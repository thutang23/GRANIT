// =============================================================================
// Plain Wall Contact Forces for DEM Simulation
// =============================================================================
// Implements contact forces between particles and infinite plane walls using
// policy-based design for extensibility.
//
// Key Features:
// - Template Policy Composition for compile-time wall type selection
// - Generic calculate_wall_force function reusing contact physics
// - Static and Moving wall support (velocity from config or data file)
// - Hertz-Mindlin contact model with rolling/twisting friction
//
// Main Differences from Particle-Particle Contact:
// - Mass: Wall has infinite mass → M_eff = m_particle (not m_i*m_j/(m_i+m_j))
// - Radius: Wall has infinite radius → R_eff = R_particle (not R_i*R_j/(R_i+R_j))
// - Relative velocity: Particle velocity minus wall velocity
//
// Reference: lib/forceContactWalls.f90, lib/forceMovingWalls.f90
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../particle_model/particle.h"
#include "../math/linear_interpolation.cuh"
#include "wall_contact_force.cuh"
#include "rigid_pose_trajectory.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstdint>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace cudadem {

static SimConfig g_plain_wall_config;
static bool g_plain_wall_config_valid = false;

// =============================================================================
// Wall Type Policies - Compile-Time Selection
// =============================================================================
// These policies define how wall geometry and physics are computed
// =============================================================================

// -----------------------------------------------------------------------------
// Mass Policy: How to compute effective mass for wall contact
// -----------------------------------------------------------------------------

// Infinite wall mass: M_eff = m_particle only
struct InfiniteWallMassPolicy {
    __device__ __forceinline__
    static float effective_mass(float m_particle) {
        return m_particle;  // Wall has infinite mass
    }

    __device__ __forceinline__
    static float effective_radius(float R_particle) {
        return R_particle;  // Wall has infinite radius (flat surface)
    }

    __device__ __forceinline__
    static float effective_inertia(float I_particle) {
        return I_particle;  // Only particle inertia matters
    }
};

// Finite wall mass (for future: moving objects with mass)
// Note: Uses runtime parameter instead of template to avoid nonstandard float template
struct FiniteWallMassPolicy {
    float wall_mass;

    __host__ __device__ FiniteWallMassPolicy(float mass = 1e30f) : wall_mass(mass) {}

    __device__ __forceinline__
    float effective_mass(float m_particle) const {
        return m_particle * wall_mass / (m_particle + wall_mass);
    }

    __device__ __forceinline__
    float effective_radius(float R_particle, float R_wall) const {
        return R_particle * R_wall / (R_particle + R_wall);
    }
};

// -----------------------------------------------------------------------------
// Motion Policy: Wall velocity behavior
// -----------------------------------------------------------------------------

// Static wall: zero velocity
struct StaticWallPolicy {
    static constexpr bool is_moving = false;

    __device__ __forceinline__
    static void get_velocity(float& vx, float& vy, float& vz) {
        vx = 0.0f;
        vy = 0.0f;
        vz = 0.0f;
    }
};

// Moving wall: velocity from parameter
struct MovingWallPolicy {
    static constexpr bool is_moving = true;

    __device__ __forceinline__
    static void get_velocity(
        double& vx, double& vy, double& vz,
        double wall_vx, double wall_vy, double wall_vz
    ) {
        vx = wall_vx;
        vy = wall_vy;
        vz = wall_vz;
    }
};

// =============================================================================
// Note: plane_signed_distance and compute_plane_overlap are defined in wall.h
// =============================================================================

// =============================================================================
// Moving Wall Velocity Data Loader
// =============================================================================
// Loads velocity time series from binary file for moving walls
// Format: [num_frames][time0, vx0, vy0, vz0][time1, vx1, vy1, vz1]...
// =============================================================================

// Use shared definition from WallManager
using MovingWallVelocityData = WallManager::MovingWallVelocityData;

// Note: interpolate_wall_velocity is defined in wall.h

// =============================================================================
// GPU Wall Data Structure
// =============================================================================
// Compact representation for GPU constant memory / global memory
// =============================================================================

struct PlainWallGPU {
    int32_t id;
    int32_t material_id;
    bool enabled;
    bool is_smooth;       // Frictionless wall
    bool is_moving;       // Has velocity
    bool is_finite;       // Finite-area wall flag

    // Geometry: plane defined by point and normal
    float point_x, point_y, point_z;
    float normal_x, normal_y, normal_z;
    // World body tangents for a finite rectangle.  Keeping them explicitly
    // is required for a rotating rigid wall; a normal alone loses yaw.
    float axis_u_x, axis_u_y, axis_u_z;
    float axis_v_x, axis_v_y, axis_v_z;

    // Finite wall bounds
    float half_extent_x;  // Half-length in local X direction
    float half_extent_y;  // Half-length in local Y direction

    // Static velocity (if not using time series)
    double velocity_x, velocity_y, velocity_z;
    // Prescribed rigid-body angular velocity and reference-origin world point.
    double omega_x, omega_y, omega_z;
    double center_x, center_y, center_z;

    // Moving wall velocity data (if using time series)
    const float* d_times;
    const double* d_vx;
    const double* d_vy;
    const double* d_vz;
    int32_t num_frames;

    __host__ __device__ PlainWallGPU()
        : id(-1), material_id(-1), enabled(false), is_smooth(false), is_moving(false),
          is_finite(false),
          point_x(0), point_y(0), point_z(0),
          normal_x(0), normal_y(1), normal_z(0),
          axis_u_x(1), axis_u_y(0), axis_u_z(0),
          axis_v_x(0), axis_v_y(0), axis_v_z(1),
          half_extent_x(1e10f), half_extent_y(1e10f),
          velocity_x(0), velocity_y(0), velocity_z(0),
          omega_x(0), omega_y(0), omega_z(0), center_x(0), center_y(0), center_z(0),
          d_times(nullptr), d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
          num_frames(0) {}
};

// Note: MAX_PLAIN_WALLS is defined in wall.h

// GPU constant memory for wall data
// Use raw storage to avoid dynamic initialization (CUDA limitation)
__constant__ __align__(16) unsigned char dev_plain_walls_storage[MAX_PLAIN_WALLS * sizeof(PlainWallGPU)];
__constant__ int32_t dev_num_plain_walls;

// Accessor macro for plain walls array
#define dev_plain_walls (reinterpret_cast<PlainWallGPU*>(const_cast<unsigned char*>(dev_plain_walls_storage)))

// =============================================================================
// Wall Contact Force Kernel
// =============================================================================

template<bool EnableFriction>
__global__ void compute_plain_wall_forces_kernel(
    // Particle positions (double precision)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle velocities
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    // Angular velocities
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    // Particle properties
    const float* __restrict__ radius,
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const DemInteractionProperties* __restrict__ interactions,
    uint32_t material_count,
    // Force/torque output
    // Force/torque output
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    // Wall contact history
    WallContactHistory* __restrict__ wall_history,
    const int32_t* __restrict__ wall_history_offset,
    int32_t* __restrict__ num_wall_contacts,
    int32_t max_wall_contacts_per_particle,
    bool use_visco,
    float dt,
    float current_time,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Load particle data
    float pi_x = static_cast<float>(pos_x[idx]);
    float pi_y = static_cast<float>(pos_y[idx]);
    float pi_z = static_cast<float>(pos_z[idx]);
    double vi_x = vel_x[idx];
    double vi_y = vel_y[idx];
    double vi_z = vel_z[idx];
    double wi_x = omega_x[idx];
    double wi_y = omega_y[idx];
    double wi_z = omega_z[idx];
    float R_i = radius[idx];
    float m_i = mass[idx];
    float I_i = moment_of_inertia[idx];
    const int32_t particle_material_id = material_id[idx];

    // Wall history for this particle
    int32_t hist_offset = wall_history_offset[idx];
    WallContactHistory* my_history = wall_history + hist_offset;
    int32_t my_num_contacts = num_wall_contacts[idx];

    // Accumulate forces
    double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
    double tx_acc = 0.0, ty_acc = 0.0, tz_acc = 0.0;

    int32_t new_contact_count = 0;

    // Read wall count from constant memory
    int32_t num_walls = dev_num_plain_walls;

    // Loop over all plain walls
    for (int32_t w = 0; w < num_walls; ++w) {
        const PlainWallGPU& wall = dev_plain_walls[w];

        if (!wall.enabled) continue;

        float E_star, G_star;
        DemInteractionProperties pair{};
        if (!resolve_wall_material_pair(particle_material_id, wall.material_id,
                                        materials, interactions, material_count,
                                        E_star, G_star, pair)) continue;

        // Get wall velocity
        double wall_vx, wall_vy, wall_vz;
        if (wall.is_moving && wall.num_frames > 0) {
            interpolate_wall_velocity(
                wall_vx, wall_vy, wall_vz,
                current_time,
                wall.d_times, wall.d_vx, wall.d_vy, wall.d_vz,
                wall.num_frames
            );
        } else {
            wall_vx = wall.velocity_x;
            wall_vy = wall.velocity_y;
            wall_vz = wall.velocity_z;
        }
        // The trajectory is evaluated host-side and copied into constant
        // memory.  Add the rigid-body velocity field here; a surface-only
        // conveyor has zero omega and unchanged geometry.
        const double rx = static_cast<double>(pi_x) - wall.center_x;
        const double ry = static_cast<double>(pi_y) - wall.center_y;
        const double rz = static_cast<double>(pi_z) - wall.center_z;
        wall_vx += wall.omega_y * rz - wall.omega_z * ry;
        wall_vy += wall.omega_z * rx - wall.omega_x * rz;
        wall_vz += wall.omega_x * ry - wall.omega_y * rx;

        // Find contact history for this wall
        int slot = -1;
        float H_t_x = 0.0f, H_t_y = 0.0f, H_t_z = 0.0f;
        float H_r_x = 0.0f, H_r_y = 0.0f, H_r_z = 0.0f;
        float H_tw_x = 0.0f, H_tw_y = 0.0f, H_tw_z = 0.0f;

        for (int32_t k = 0; k < my_num_contacts; ++k) {
            if (my_history[k].wall_id == wall.id && my_history[k].is_active) {
                slot = k;
                H_t_x = my_history[k].delta_t_x;
                H_t_y = my_history[k].delta_t_y;
                H_t_z = my_history[k].delta_t_z;
                H_r_x = my_history[k].delta_r_x;
                H_r_y = my_history[k].delta_r_y;
                H_r_z = my_history[k].delta_r_z;
                H_tw_x = my_history[k].delta_tw_x;
                H_tw_y = my_history[k].delta_tw_y;
                H_tw_z = my_history[k].delta_tw_z;
                break;
            }
        }

        // Compute wall force
        WallForceResult result;

        if (wall.is_finite) {
            // Finite wall: closest-point-on-rectangle for correct edge/corner geometry
            FiniteWallContactResult cr = compute_oriented_finite_wall_contact(
                pi_x, pi_y, pi_z, R_i,
                wall.point_x, wall.point_y, wall.point_z,
                wall.normal_x, wall.normal_y, wall.normal_z,
                wall.axis_u_x, wall.axis_u_y, wall.axis_u_z,
                wall.axis_v_x, wall.axis_v_y, wall.axis_v_z,
                wall.half_extent_x, wall.half_extent_y
            );

            if (cr.delta_n <= 0.0f) {
                WallForceResult cohesion;
                add_wall_cohesion(cohesion, cr.delta_n,
                                  cr.normal_x, cr.normal_y, cr.normal_z,
                                  R_i, pair);
                fx_acc += cohesion.force_x;
                fy_acc += cohesion.force_y;
                fz_acc += cohesion.force_z;
                if (slot >= 0) {
                    my_history[slot].is_active = false;
                }
                continue;
            }

            if constexpr (EnableFriction) {
                if (!wall.is_smooth) {
                    result = calculate_wall_force_impl<FullFrictionPolicy>(
                        cr.delta_n, cr.dist,
                        vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                        R_i, m_i, I_i,
                        cr.normal_x, cr.normal_y, cr.normal_z,
                        wall_vx, wall_vy, wall_vz,
                        H_t_x, H_t_y, H_t_z,
                        H_r_x, H_r_y, H_r_z,
                        H_tw_x, H_tw_y, H_tw_z,
                        E_star, G_star, pair.restitution,
                        pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                        use_visco, pair.viscous_coefficient, dt
                    );
                } else {
                    result = calculate_wall_force_impl<SmoothWallPolicy>(
                        cr.delta_n, cr.dist,
                        vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                        R_i, m_i, I_i,
                        cr.normal_x, cr.normal_y, cr.normal_z,
                        wall_vx, wall_vy, wall_vz,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        E_star, G_star, pair.restitution,
                        pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                        use_visco, pair.viscous_coefficient, dt
                    );
                }
            } else {
                result = calculate_wall_force_impl<SmoothWallPolicy>(
                    cr.delta_n, cr.dist,
                    vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                    R_i, m_i, I_i,
                    cr.normal_x, cr.normal_y, cr.normal_z,
                    wall_vx, wall_vy, wall_vz,
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    E_star, G_star, pair.restitution,
                    pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                    use_visco, pair.viscous_coefficient, dt
                );
            }
            add_wall_cohesion(result, cr.delta_n, cr.normal_x, cr.normal_y, cr.normal_z, R_i, pair);
        } else {
            // Infinite wall: plane distance + overlap
            if constexpr (EnableFriction) {
                if (!wall.is_smooth) {
                    result = calculate_wall_force<FullFrictionPolicy>(
                        pi_x, pi_y, pi_z, vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                        R_i, m_i, I_i,
                        wall.normal_x, wall.normal_y, wall.normal_z,
                        wall.point_x, wall.point_y, wall.point_z,
                        wall_vx, wall_vy, wall_vz,
                        H_t_x, H_t_y, H_t_z,
                        H_r_x, H_r_y, H_r_z,
                        H_tw_x, H_tw_y, H_tw_z,
                        E_star, G_star, pair.restitution,
                        pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                        use_visco, pair.viscous_coefficient, dt
                    );
                } else {
                    result = calculate_wall_force<SmoothWallPolicy>(
                        pi_x, pi_y, pi_z, vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                        R_i, m_i, I_i,
                        wall.normal_x, wall.normal_y, wall.normal_z,
                        wall.point_x, wall.point_y, wall.point_z,
                        wall_vx, wall_vy, wall_vz,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        E_star, G_star, pair.restitution,
                        pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                        use_visco, pair.viscous_coefficient, dt
                    );
                }
            } else {
                result = calculate_wall_force<SmoothWallPolicy>(
                    pi_x, pi_y, pi_z, vi_x, vi_y, vi_z, wi_x, wi_y, wi_z,
                    R_i, m_i, I_i,
                    wall.normal_x, wall.normal_y, wall.normal_z,
                    wall.point_x, wall.point_y, wall.point_z,
                    wall_vx, wall_vy, wall_vz,
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    E_star, G_star, pair.restitution,
                    pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                    use_visco, pair.viscous_coefficient, dt
                );
            }
        }

        // This is deliberately after the contact-law calculation so cohesive
        // attraction never enlarges the friction, rolling, or twisting load.
        if (!wall.is_finite) {
            const float delta_n = compute_plane_overlap(
                plane_signed_distance(pi_x, pi_y, pi_z,
                                      wall.normal_x, wall.normal_y, wall.normal_z,
                                      wall.point_x, wall.point_y, wall.point_z), R_i);
            add_wall_cohesion(result, delta_n, wall.normal_x, wall.normal_y, wall.normal_z, R_i, pair);
        }

        // Accumulate forces
        fx_acc += result.force_x;
        fy_acc += result.force_y;
        fz_acc += result.force_z;
        tx_acc += result.torque_x;
        ty_acc += result.torque_y;
        tz_acc += result.torque_z;

        // Update contact history
        if (result.is_contact) {
            if (slot < 0 && new_contact_count + my_num_contacts < max_wall_contacts_per_particle) {
                slot = my_num_contacts + new_contact_count;
                new_contact_count++;
            }

            if (slot >= 0 && slot < max_wall_contacts_per_particle) {
                my_history[slot].wall_id = wall.id;
                my_history[slot].delta_t_x = result.H_t_x;
                my_history[slot].delta_t_y = result.H_t_y;
                my_history[slot].delta_t_z = result.H_t_z;
                my_history[slot].delta_r_x = result.H_r_x;
                my_history[slot].delta_r_y = result.H_r_y;
                my_history[slot].delta_r_z = result.H_r_z;
                my_history[slot].delta_tw_x = result.H_tw_x;
                my_history[slot].delta_tw_y = result.H_tw_y;
                my_history[slot].delta_tw_z = result.H_tw_z;
                my_history[slot].is_active = true;
                my_history[slot].is_slipping = result.is_slipping;
                my_history[slot].is_rolling = result.is_rolling;
                my_history[slot].is_twisting = result.is_twisting;
            }
        } else if (slot >= 0) {
            my_history[slot].is_active = false;
        }
    }

    // Atomic add to global arrays
    // Atomic add to global arrays
    atomic_add_force(force_x, force_y, force_z, fx_acc, fy_acc, fz_acc, idx);
    atomic_add_torque(torque_x, torque_y, torque_z, tx_acc, ty_acc, tz_acc, idx);

    // Update contact count
    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// Wall Force Dispatcher
// =============================================================================

class PlainWallDispatcher {
public:
    struct RuntimeWall {
        PlainWallGPU body_wall;
        bool has_trajectory = false;
        RigidPoseTrajectory trajectory;
    };
private:
    static std::vector<RuntimeWall> runtime_walls;

    static void basis_from_normal(PlainWallGPU& wall) {
        const float ax = fabsf(wall.normal_x) < 0.9f ? 1.0f : 0.0f;
        const float ay = fabsf(wall.normal_x) < 0.9f ? 0.0f : 1.0f;
        const float az = 0.0f;
        const float dot = ax*wall.normal_x + ay*wall.normal_y + az*wall.normal_z;
        wall.axis_u_x=ax-dot*wall.normal_x; wall.axis_u_y=ay-dot*wall.normal_y; wall.axis_u_z=az-dot*wall.normal_z;
        const float norm=sqrtf(wall.axis_u_x*wall.axis_u_x+wall.axis_u_y*wall.axis_u_y+wall.axis_u_z*wall.axis_u_z);
        wall.axis_u_x/=norm; wall.axis_u_y/=norm; wall.axis_u_z/=norm;
        wall.axis_v_x=wall.normal_y*wall.axis_u_z-wall.normal_z*wall.axis_u_y;
        wall.axis_v_y=wall.normal_z*wall.axis_u_x-wall.normal_x*wall.axis_u_z;
        wall.axis_v_z=wall.normal_x*wall.axis_u_y-wall.normal_y*wall.axis_u_x;
    }
    static void rotate(const std::array<double,9>& r, float x, float y, float z, float& ox, float& oy, float& oz) {
        ox=static_cast<float>(r[0]*x+r[1]*y+r[2]*z); oy=static_cast<float>(r[3]*x+r[4]*y+r[5]*z); oz=static_cast<float>(r[6]*x+r[7]*y+r[8]*z);
    }
    static void upload(double time) {
        std::vector<PlainWallGPU> walls; walls.reserve(runtime_walls.size());
        for (auto& runtime : runtime_walls) {
            PlainWallGPU wall=runtime.body_wall;
            if (runtime.has_trajectory) {
                const RigidPoseSample sample=runtime.trajectory.sample(time);
                const auto r=RigidPoseTrajectory::rotation_matrix(sample.quaternion_wxyz);
                const auto& origin=runtime.trajectory.reference_origin();
                float bx=wall.point_x-static_cast<float>(origin[0]), by=wall.point_y-static_cast<float>(origin[1]), bz=wall.point_z-static_cast<float>(origin[2]);
                rotate(r,bx,by,bz,wall.point_x,wall.point_y,wall.point_z);
                wall.point_x+=static_cast<float>(sample.translation[0]); wall.point_y+=static_cast<float>(sample.translation[1]); wall.point_z+=static_cast<float>(sample.translation[2]);
                rotate(r, runtime.body_wall.normal_x, runtime.body_wall.normal_y, runtime.body_wall.normal_z, wall.normal_x, wall.normal_y, wall.normal_z);
                rotate(r, runtime.body_wall.axis_u_x, runtime.body_wall.axis_u_y, runtime.body_wall.axis_u_z, wall.axis_u_x, wall.axis_u_y, wall.axis_u_z);
                rotate(r, runtime.body_wall.axis_v_x, runtime.body_wall.axis_v_y, runtime.body_wall.axis_v_z, wall.axis_v_x, wall.axis_v_y, wall.axis_v_z);
                wall.velocity_x=sample.linear_velocity[0]; wall.velocity_y=sample.linear_velocity[1]; wall.velocity_z=sample.linear_velocity[2];
                wall.omega_x=sample.angular_velocity[0]; wall.omega_y=sample.angular_velocity[1]; wall.omega_z=sample.angular_velocity[2];
                wall.center_x=sample.translation[0]; wall.center_y=sample.translation[1]; wall.center_z=sample.translation[2]; wall.is_moving=true;
            }
            walls.push_back(wall);
        }
        const int32_t count=static_cast<int32_t>(walls.size());
        if (count) cudaMemcpyToSymbol(dev_plain_walls_storage, walls.data(), count*sizeof(PlainWallGPU));
        cudaMemcpyToSymbol(dev_num_plain_walls, &count, sizeof(count));
    }
public:
    // Convert SimConfig walls to GPU format
    static void initialize_walls(
        const SimConfig& config,
        std::vector<MovingWallVelocityData>& velocity_data
    ) {
        runtime_walls.clear();
        velocity_data.clear(); // Retained only for ABI; legacy data is rejected by config_host.

        for (int32_t i = 0; i < config.num_walls; ++i) {
            const WallConfig& wc = config.walls[i];

            if (wc.type == WallType::PLAIN || wc.type == WallType::MOVING || wc.type == WallType::FINITE_PLAIN) {
                PlainWallGPU wall;
                wall.id = wc.id;
                wall.material_id = wc.material_id;
                wall.enabled = wc.enabled;
                wall.is_smooth = false;  // Could add to config
                wall.is_finite = (wc.type == WallType::FINITE_PLAIN);

                // Geometry
                wall.point_x = static_cast<float>(wc.position.x);
                wall.point_y = static_cast<float>(wc.position.y);
                wall.point_z = static_cast<float>(wc.position.z);
                wall.normal_x = static_cast<float>(wc.normal.x);
                wall.normal_y = static_cast<float>(wc.normal.y);
                wall.normal_z = static_cast<float>(wc.normal.z);

                // Normalize
                float len = sqrtf(wall.normal_x * wall.normal_x +
                                  wall.normal_y * wall.normal_y +
                                  wall.normal_z * wall.normal_z);
                if (len > 1e-10f) {
                    wall.normal_x /= len;
                    wall.normal_y /= len;
                    wall.normal_z /= len;
                }

                // Set finite wall bounds
                if (wall.is_finite) {
                    wall.half_extent_x = wc.half_extents[0];
                    wall.half_extent_y = wc.half_extents[1];
                }

                basis_from_normal(wall);
                wall.is_moving = wc.surface_only_motion;
                wall.velocity_x = wc.surface_velocity.x; wall.velocity_y = wc.surface_velocity.y; wall.velocity_z = wc.surface_velocity.z;
                RuntimeWall runtime; runtime.body_wall=wall;
                if (wc.rigid_pose_path[0] != '\0') { runtime.trajectory.load(wc.rigid_pose_path); runtime.has_trajectory=true; }
                runtime_walls.push_back(std::move(runtime));
            }
        }

        // Copy to constant memory
        int32_t num_walls = static_cast<int32_t>(runtime_walls.size());
        if (num_walls > MAX_PLAIN_WALLS) {
            std::cerr << "WARNING: Too many plain walls (" << num_walls
                      << "), truncating to " << MAX_PLAIN_WALLS << std::endl;
            num_walls = MAX_PLAIN_WALLS;
        }

        if (num_walls != static_cast<int32_t>(runtime_walls.size())) throw std::runtime_error("too many plain walls");
        upload(0.0);

        std::cout << "Initialized " << num_walls << " plain/moving walls" << std::endl;
    }

    // Compute wall forces
    static void compute_forces(
        const SimulationState& state,
        const SimConfig& config,
        float dt,
        float current_time,
        cudaStream_t stream = 0
    ) {
        upload(current_time);
        int32_t num_particles = state.particles.num_particles;
        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        bool use_visco = config.dem_damping_model == DemDampingModel::VISCOUS;

        // Select kernel based on friction mode
        LAUNCH_KERNEL(compute_plain_wall_forces_kernel<true>, num_blocks, block_size, 0, stream,
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
            state.particles.omega_x, state.particles.omega_y, state.particles.omega_z,
            state.particles.radius, state.particles.mass, state.particles.moment_of_inertia,
            state.particles.material_id, state.material_database.materials,
            state.material_database.dem_interactions, state.material_database.material_count,
            state.particles.force_x, state.particles.force_y, state.particles.force_z,
            state.particles.torque_x, state.particles.torque_y, state.particles.torque_z,
            state.wall_contact_history.data,
            state.particles.wall_history_offset,
            state.particles.num_wall_contacts,
            state.wall_contact_history.max_contacts_per_particle,
            use_visco, dt, current_time,
            num_particles
        );
    }
};

std::vector<PlainWallDispatcher::RuntimeWall> PlainWallDispatcher::runtime_walls;

void plain_wall_initialize(const SimConfig& config) {
    static std::vector<MovingWallVelocityData> velocity_data;
    g_plain_wall_config = config;
    g_plain_wall_config_valid = true;
    PlainWallDispatcher::initialize_walls(g_plain_wall_config, velocity_data);
}

void plain_wall_copy_to_current_device() {
    if (!g_plain_wall_config_valid) return;
    static std::vector<MovingWallVelocityData> velocity_data;
    PlainWallDispatcher::initialize_walls(g_plain_wall_config, velocity_data);
}

void plain_wall_compute_forces(const SimulationState& state, const SimConfig& config,
                               float dt, float current_time, cudaStream_t stream) {
    PlainWallDispatcher::compute_forces(state, config, dt, current_time, stream);
}

// =============================================================================
// Wall Contact History Cleanup
// =============================================================================

__global__ void cleanup_wall_contact_history_kernel(
    WallContactHistory* __restrict__ wall_history,
    const int32_t* __restrict__ wall_history_offset,
    int32_t* __restrict__ num_wall_contacts,
    int32_t max_contacts_per_particle,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    int32_t offset = wall_history_offset[idx];
    int32_t n = num_wall_contacts[idx];
    int32_t write_idx = 0;

    for (int32_t k = 0; k < n; ++k) {
        if (wall_history[offset + k].is_active) {
            if (write_idx != k) {
                wall_history[offset + write_idx] = wall_history[offset + k];
            }
            write_idx++;
        }
    }

    num_wall_contacts[idx] = write_idx;
}

inline void cleanup_wall_contact_history(
    SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream = 0
) {
    int32_t num_particles = state.particles.num_particles;
    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(cleanup_wall_contact_history_kernel, num_blocks, block_size, 0, stream,
        state.wall_contact_history.data,
        state.particles.wall_history_offset,
        state.particles.num_wall_contacts,
        state.wall_contact_history.max_contacts_per_particle,
        num_particles
    );
}

// =============================================================================
// Verification Functions (Host)
// =============================================================================

#ifndef __CUDACC__

inline void print_plain_walls(const SimConfig& config) {
    std::cout << "=== Plain Walls ===" << std::endl;
    int count = 0;
    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];
        if (wc.type == WallType::PLAIN || wc.type == WallType::MOVING) {
            std::cout << "Wall " << wc.id << ": "
                      << (wc.type == WallType::MOVING ? "MOVING" : "STATIC")
                      << std::endl;
            std::cout << "  Point: (" << wc.position.x << ", " << wc.position.y
                      << ", " << wc.position.z << ")" << std::endl;
            std::cout << "  Normal: (" << wc.normal.x << ", " << wc.normal.y
                      << ", " << wc.normal.z << ")" << std::endl;
            if (wc.type == WallType::MOVING) {
                std::cout << "  Velocity: (" << wc.velocity.x << ", "
                          << wc.velocity.y << ", " << wc.velocity.z << ")" << std::endl;
            }
            count++;
        }
    }
    std::cout << "Total: " << count << " walls" << std::endl;
    std::cout << "===================" << std::endl;
}

#endif // __CUDACC__

} // namespace cudadem

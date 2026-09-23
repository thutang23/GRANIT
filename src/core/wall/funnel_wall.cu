// =============================================================================
// Funnel Wall Contact Forces for DEM Simulation
// =============================================================================
// Implements contact forces between particles and conical frustum walls
// (interior contact). Geometry detection is a direct port from
// lib/forceFunnelWalls.f90 (Fortran). Force computation reuses the shared
// calculate_wall_force_impl() from wall_contact_force.cuh.
//
// A funnel wall is defined by:
//   - A base point (bottom center of frustum)
//   - A unit axis direction (base → top)
//   - Radius at base (radius_bottom) and top (radius_top)
//   - Length (height along axis)
//
// Interior contact means the surface normal points inward (toward the axis).
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../particle_model/particle.h"
#include "wall_contact_force.cuh"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

namespace cudadem {

// =============================================================================
// GPU Wall Data Structure
// =============================================================================

struct FunnelWallGPU {
    int32_t id;
    int32_t material_id;
    bool enabled;
    bool is_smooth;

    float point_x, point_y, point_z;
    float axis_x, axis_y, axis_z;       // Unit axis (normalized on host)
    float radius_bottom, radius_top;
    float length;

    __host__ __device__ FunnelWallGPU()
        : id(-1), material_id(-1), enabled(false), is_smooth(false),
          point_x(0), point_y(0), point_z(0),
          axis_x(0), axis_y(0), axis_z(1),
          radius_bottom(1), radius_top(0.5f), length(1) {}
};

// =============================================================================
// Constant Memory
// =============================================================================

constexpr int MAX_FUNNEL_WALLS = 16;

__constant__ __align__(16) unsigned char dev_funnel_walls_storage[MAX_FUNNEL_WALLS * sizeof(FunnelWallGPU)];
__constant__ int32_t dev_num_funnel_walls;

#define dev_funnel_walls (reinterpret_cast<FunnelWallGPU*>(const_cast<unsigned char*>(dev_funnel_walls_storage)))

static int32_t host_num_funnel_walls = 0;
static std::vector<FunnelWallGPU> host_funnel_walls_cache;

// =============================================================================
// Funnel Contact Geometry
// =============================================================================
// Direct port of the geometry detection from lib/forceFunnelWalls.f90
// lines 66–103. Variable names preserved for traceability.
//
// Algorithm:
//   1. RV = particle_pos - funnel_base_point
//   2. ERR = dot(RV, axis) — axial projection
//   3. If ERR <= 0 or ERR >= length: no contact
//   4. If ERR < 1e-14: no contact (degenerate)
//   5. RVb = (RV - ERR*axis) normalized, then scaled to r_bot magnitude
//   6. RVu = same direction, scaled to r_top magnitude
//   7. RVg = normalize(-RVb + axis*length + RVu) — generatrix direction
//   8. ERR2 = dot(RVg, RVb)
//   9. RVn = normalize(ERR2*RVg - RVb) — inward-pointing surface normal
//  10. RVc = RV - RVb
//  11. If dot(RVc, RVg) <= 0 OR dot(RVc, RVn) <= 0: no contact
//  12. dist = dot(RVc, RVn), delta_n = R - dist
// =============================================================================

struct FunnelContactResult {
    float delta_n;                          // Overlap (>0 = contact)
    float dist;                             // Distance from center to surface
    float normal_x, normal_y, normal_z;    // Surface normal pointing into interior
};

__device__ __forceinline__
FunnelContactResult compute_funnel_contact(
    float px, float py, float pz, float R,
    float fp_x, float fp_y, float fp_z,
    float fa_x, float fa_y, float fa_z,
    float r_bot, float r_top, float length
)
{
    FunnelContactResult result;
    result.delta_n = -1.0f;
    result.dist = 0.0f;
    result.normal_x = 0.0f;
    result.normal_y = 0.0f;
    result.normal_z = 0.0f;

    // Step 1: RV = particle_pos - funnel_base_point
    float RV_x = px - fp_x;
    float RV_y = py - fp_y;
    float RV_z = pz - fp_z;

    // Step 2: ERR = dot(RV, axis) — axial projection
    float ERR = RV_x * fa_x + RV_y * fa_y + RV_z * fa_z;

    // Step 3: Check axial bounds
    if (ERR <= 0.0f || ERR >= length) {
        return result;
    }

    // Step 4: Degeneracy check
    if (ERR < 1e-14f) {
        return result;
    }

    // Step 5: RVb = radial component, then normalize and scale to r_bot
    float RVb_x = RV_x - ERR * fa_x;
    float RVb_y = RV_y - ERR * fa_y;
    float RVb_z = RV_z - ERR * fa_z;

    float norm = sqrtf(RVb_x * RVb_x + RVb_y * RVb_y + RVb_z * RVb_z);

    // Step 6: RVu = direction * r_top, rescale RVb to r_bot
    float inv_norm = 1.0f / norm;
    float RVu_x = RVb_x * inv_norm * r_top;
    float RVu_y = RVb_y * inv_norm * r_top;
    float RVu_z = RVb_z * inv_norm * r_top;
    RVb_x = RVb_x * inv_norm * r_bot;
    RVb_y = RVb_y * inv_norm * r_bot;
    RVb_z = RVb_z * inv_norm * r_bot;

    // Step 7: RVg = generatrix direction, then normalize
    float RVg_x = -RVb_x + fa_x * length + RVu_x;
    float RVg_y = -RVb_y + fa_y * length + RVu_y;
    float RVg_z = -RVb_z + fa_z * length + RVu_z;

    norm = sqrtf(RVg_x * RVg_x + RVg_y * RVg_y + RVg_z * RVg_z);
    RVg_x /= norm;
    RVg_y /= norm;
    RVg_z /= norm;

    // Step 8: ERR2 = dot(RVg, RVb)
    float ERR2 = RVg_x * RVb_x + RVg_y * RVb_y + RVg_z * RVb_z;

    // Step 9: RVn = inward-pointing surface normal
    float RVn_x = ERR2 * RVg_x - RVb_x;
    float RVn_y = ERR2 * RVg_y - RVb_y;
    float RVn_z = ERR2 * RVg_z - RVb_z;

    norm = sqrtf(RVn_x * RVn_x + RVn_y * RVn_y + RVn_z * RVn_z);
    RVn_x /= norm;
    RVn_y /= norm;
    RVn_z /= norm;

    // Step 10: RVc = RV - RVb
    float RVc_x = RV_x - RVb_x;
    float RVc_y = RV_y - RVb_y;
    float RVc_z = RV_z - RVb_z;

    // Step 11: Check if particle is inside the frustum cone
    float dot_RVc_RVg = RVc_x * RVg_x + RVc_y * RVg_y + RVc_z * RVg_z;
    if (dot_RVc_RVg <= 0.0f) {
        return result;
    }

    float dist = RVc_x * RVn_x + RVc_y * RVn_y + RVc_z * RVn_z;
    if (dist <= 0.0f) {
        return result;
    }

    // Step 12: Compute overlap
    result.dist = dist;
    result.delta_n = R - dist;
    result.normal_x = RVn_x;
    result.normal_y = RVn_y;
    result.normal_z = RVn_z;

    return result;
}

// =============================================================================
// Funnel Wall Force Kernel
// =============================================================================

template<bool EnableFriction>
__global__ void compute_funnel_wall_forces_kernel(
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
    int32_t num_particles
)
{
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
    int32_t num_walls = dev_num_funnel_walls;

    // Loop over all funnel walls
    for (int32_t w = 0; w < num_walls; ++w) {
        const FunnelWallGPU& wall = dev_funnel_walls[w];

        if (!wall.enabled) continue;

        float E_star, G_star;
        DemInteractionProperties pair{};
        if (!resolve_wall_material_pair(particle_material_id, wall.material_id,
                                        materials, interactions, material_count,
                                        E_star, G_star, pair)) continue;

        // Compute funnel contact geometry
        FunnelContactResult cr = compute_funnel_contact(
            pi_x, pi_y, pi_z, R_i,
            wall.point_x, wall.point_y, wall.point_z,
            wall.axis_x, wall.axis_y, wall.axis_z,
            wall.radius_bottom, wall.radius_top, wall.length
        );

        // No contact
        if (cr.delta_n <= 0.0f) {
            WallForceResult cohesion;
            add_wall_cohesion(cohesion, cr.delta_n,
                              cr.normal_x, cr.normal_y, cr.normal_z, R_i, pair);
            fx_acc += cohesion.force_x;
            fy_acc += cohesion.force_y;
            fz_acc += cohesion.force_z;
            // Mark existing contact as inactive
            for (int32_t k = 0; k < my_num_contacts; ++k) {
                if (my_history[k].wall_id == wall.id) {
                    my_history[k].is_active = false;
                    break;
                }
            }
            continue;
        }

        // Find contact history slot
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

        // Funnel walls are static — wall velocity is always zero
        double wall_vx = 0.0, wall_vy = 0.0, wall_vz = 0.0;

        // Compute wall force using shared implementation
        WallForceResult fresult;

        if constexpr (EnableFriction) {
            if (!wall.is_smooth) {
                fresult = calculate_wall_force_impl<FullFrictionPolicy>(
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
                fresult = calculate_wall_force_impl<SmoothWallPolicy>(
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
            fresult = calculate_wall_force_impl<SmoothWallPolicy>(
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
        add_wall_cohesion(fresult, cr.delta_n,
                          cr.normal_x, cr.normal_y, cr.normal_z, R_i, pair);

        // Accumulate forces
        fx_acc += fresult.force_x;
        fy_acc += fresult.force_y;
        fz_acc += fresult.force_z;
        tx_acc += fresult.torque_x;
        ty_acc += fresult.torque_y;
        tz_acc += fresult.torque_z;

        // Update contact history
        if (fresult.is_contact) {
            if (slot < 0 && new_contact_count + my_num_contacts < max_wall_contacts_per_particle) {
                slot = my_num_contacts + new_contact_count;
                new_contact_count++;
            }

            if (slot >= 0 && slot < max_wall_contacts_per_particle) {
                my_history[slot].wall_id = wall.id;
                my_history[slot].delta_t_x = fresult.H_t_x;
                my_history[slot].delta_t_y = fresult.H_t_y;
                my_history[slot].delta_t_z = fresult.H_t_z;
                my_history[slot].delta_r_x = fresult.H_r_x;
                my_history[slot].delta_r_y = fresult.H_r_y;
                my_history[slot].delta_r_z = fresult.H_r_z;
                my_history[slot].delta_tw_x = fresult.H_tw_x;
                my_history[slot].delta_tw_y = fresult.H_tw_y;
                my_history[slot].delta_tw_z = fresult.H_tw_z;
                my_history[slot].is_active = true;
                my_history[slot].is_slipping = fresult.is_slipping;
                my_history[slot].is_rolling = fresult.is_rolling;
                my_history[slot].is_twisting = fresult.is_twisting;
            }
        } else if (slot >= 0) {
            my_history[slot].is_active = false;
        }
    }

    // Atomic add to global arrays
    atomic_add_force(force_x, force_y, force_z, fx_acc, fy_acc, fz_acc, idx);
    atomic_add_torque(torque_x, torque_y, torque_z, tx_acc, ty_acc, tz_acc, idx);

    // Update contact count
    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// Funnel Wall Dispatcher
// =============================================================================

class FunnelWallDispatcher {
public:
    static void initialize_walls(const SimConfig& config);
    static void compute_forces(const SimulationState& state, const SimConfig& config,
                               float dt, cudaStream_t stream = 0);
    static int32_t get_num_walls();
};

// =============================================================================
// FunnelWallDispatcher Implementation
// =============================================================================

void FunnelWallDispatcher::initialize_walls(const SimConfig& config) {
    std::vector<FunnelWallGPU> walls;

    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];

        if (wc.type == WallType::FUNNEL) {
            FunnelWallGPU wall;
            wall.id = wc.id;
            wall.material_id = wc.material_id;
            wall.enabled = wc.enabled;
            wall.is_smooth = false;

            // Geometry
            wall.point_x = static_cast<float>(wc.position.x);
            wall.point_y = static_cast<float>(wc.position.y);
            wall.point_z = static_cast<float>(wc.position.z);

            // Normalize axis direction
            float ax = static_cast<float>(wc.normal.x);
            float ay = static_cast<float>(wc.normal.y);
            float az = static_cast<float>(wc.normal.z);
            float alen = sqrtf(ax * ax + ay * ay + az * az);
            if (alen > 1e-10f) {
                wall.axis_x = ax / alen;
                wall.axis_y = ay / alen;
                wall.axis_z = az / alen;
            } else {
                wall.axis_x = 0.0f;
                wall.axis_y = 0.0f;
                wall.axis_z = 1.0f;
            }

            wall.radius_bottom = wc.funnel_radius_bottom;
            wall.radius_top = wc.funnel_radius_top;
            wall.length = wc.funnel_length;

            walls.push_back(wall);
        }
    }

    // Copy to constant memory
    int32_t num_walls = static_cast<int32_t>(walls.size());
    if (num_walls > MAX_FUNNEL_WALLS) {
        std::cerr << "WARNING: Too many funnel walls (" << num_walls
                  << "), truncating to " << MAX_FUNNEL_WALLS << std::endl;
        num_walls = MAX_FUNNEL_WALLS;
    }

    if (num_walls > 0) {
        cudaMemcpyToSymbol(dev_funnel_walls_storage, walls.data(),
                           num_walls * sizeof(FunnelWallGPU));
    }
    cudaMemcpyToSymbol(dev_num_funnel_walls, &num_walls, sizeof(int32_t));
    host_num_funnel_walls = num_walls;
    host_funnel_walls_cache = walls;

    if (num_walls > 0) {
        std::cout << "Initialized " << num_walls << " funnel walls" << std::endl;
    }
}

void FunnelWallDispatcher::compute_forces(
    const SimulationState& state,
    const SimConfig& config,
    float dt,
    cudaStream_t stream
) {
        if (host_num_funnel_walls <= 0) return;

        int32_t num_particles = state.particles.num_particles;
        if (num_particles == 0) return;

        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        bool use_visco = config.dem_damping_model == DemDampingModel::VISCOUS;

        LAUNCH_KERNEL(compute_funnel_wall_forces_kernel<true>, num_blocks, block_size, 0, stream,
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
            use_visco, dt,
            num_particles
        );
}

int32_t FunnelWallDispatcher::get_num_walls() {
    return host_num_funnel_walls;
}

void copy_funnel_to_current_device() {
    if (host_num_funnel_walls > 0 && !host_funnel_walls_cache.empty()) {
        cudaMemcpyToSymbol(dev_funnel_walls_storage, host_funnel_walls_cache.data(),
                           host_num_funnel_walls * sizeof(FunnelWallGPU));
    }
    cudaMemcpyToSymbol(dev_num_funnel_walls, &host_num_funnel_walls, sizeof(int32_t));
}

} // namespace cudadem

// =============================================================================
// Wall Manager Implementation
// =============================================================================
// Unified orchestration of all wall subsystems:
// - Plain walls (static infinite planes)
// - Moving walls (time-varying velocity planes)
// - Periodic boundaries (coordinate wrapping)
//
// This file implements the WallManager class defined in wall.h
// =============================================================================

#include "wall.h"
#include "trimesh_wall.h"
#include "bonded_rect_wall.h"
#include "sphere_wall.h"
#include "../particle_model/particle.h"
#include "../acceleration/acceleration.h"
#include "../constant/system_initialize.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <fstream>
#include <cstring>

namespace cudadem {

// Forward declaration for funnel wall dispatcher (defined in funnel_wall.cu)
class FunnelWallDispatcher {
public:
    static void initialize_walls(const SimConfig& config);
    static void compute_forces(const SimulationState& state, const SimConfig& config,
                               float dt, cudaStream_t stream = 0);
    static int32_t get_num_walls();
};

// =============================================================================
// Static Member Initialization
// =============================================================================

std::vector<WallManager::MovingWallVelocityData> WallManager::velocity_data_storage;
PeriodicBoundaryConfig WallManager::pbc_config;
bool WallManager::initialized = false;

// =============================================================================
// GPU Constant Memory
// =============================================================================

// Plain walls constant memory - use raw storage to avoid dynamic initialization
__constant__ __align__(16) unsigned char dev_wall_data_storage[MAX_PLAIN_WALLS * sizeof(WallData)];
__constant__ int32_t dev_num_contact_walls;
__constant__ __align__(16) unsigned char dev_pbc_config_storage[sizeof(PeriodicBoundaryConfig)];

// Host-side cache for num_contact_walls to avoid cudaMemcpyFromSymbol
static int32_t host_num_contact_walls = 0;

// Host-side cache of contact wall data for per-GPU replication
static std::vector<WallData> h_contact_walls_cache;

// Accessor macros
#define dev_wall_data (reinterpret_cast<WallData*>(const_cast<unsigned char*>(dev_wall_data_storage)))
#define dev_pbc_config (*reinterpret_cast<PeriodicBoundaryConfig*>(const_cast<unsigned char*>(dev_pbc_config_storage)))

// =============================================================================
// Moving Wall Velocity Data Loader
// =============================================================================

static bool load_velocity_data(
    WallManager::MovingWallVelocityData& vd,
    const char* path,
    WallType expected_type
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open velocity data file: " << path << std::endl;
        return false;
    }

    // Read header
    int32_t file_type;
    file.read(reinterpret_cast<char*>(&file_type), sizeof(int32_t));

    // Validate type
    if (static_cast<WallType>(file_type) != expected_type) {
        std::cerr << "ERROR: Velocity data type mismatch. Expected: "
                  << static_cast<int>(expected_type)
                  << ", Got: " << file_type << std::endl;
        file.close();
        return false;
    }

    file.read(reinterpret_cast<char*>(&vd.num_frames), sizeof(int32_t));

    if (vd.num_frames <= 0) {
        std::cerr << "ERROR: Invalid frame count in velocity data: "
                  << vd.num_frames << std::endl;
        file.close();
        return false;
    }

    vd.times.resize(vd.num_frames);
    vd.velocities_x.resize(vd.num_frames);
    vd.velocities_y.resize(vd.num_frames);
    vd.velocities_z.resize(vd.num_frames);

    for (int32_t i = 0; i < vd.num_frames; ++i) {
        float t, vx, vy, vz;
        file.read(reinterpret_cast<char*>(&t), sizeof(float));
        file.read(reinterpret_cast<char*>(&vx), sizeof(float));
        file.read(reinterpret_cast<char*>(&vy), sizeof(float));
        file.read(reinterpret_cast<char*>(&vz), sizeof(float));
        vd.times[i] = t;
        vd.velocities_x[i] = vx;
        vd.velocities_y[i] = vy;
        vd.velocities_z[i] = vz;
    }

    file.close();

    std::cout << "Loaded " << vd.num_frames << " velocity frames from " << path << std::endl;
    return true;
}

// =============================================================================
// Wall Force Kernel (Unified for Plain and Moving Walls)
// =============================================================================

// Wall material parameters (Hertz-Mindlin for wall contact)
struct WallMaterialParams {
    float R_eff;          // = R_particle (wall has infinite radius)
    float M_eff;          // = m_particle (wall has infinite mass)
    float I_eff;          // = I_particle

    float K_n, K_s, K_r, K_t;  // Spring constants
    float C_n, C_s, C_r, C_t;  // Damping coefficients
    float mu_s, mu_d;          // Friction
    float beta;                // Shape parameter

    __device__ WallMaterialParams()
        : R_eff(0), M_eff(0), I_eff(0),
          K_n(0), K_s(0), K_r(0), K_t(0),
          C_n(0), C_s(0), C_r(0), C_t(0),
          mu_s(1.0f), mu_d(1.0f), beta(0.7f) {}
};

// Compute wall material parameters (wall = infinite mass/radius)
__device__ __forceinline__
WallMaterialParams compute_wall_material_params(
    float R_particle, float m_particle, float I_particle,
    float delta_n,
    float E, float nu, float COR,
    float mu_s, float beta
) {
    WallMaterialParams params;

    // Wall has infinite mass/radius
    params.R_eff = R_particle;
    params.M_eff = m_particle;
    params.I_eff = I_particle;
    params.mu_s = mu_s;
    params.mu_d = mu_s;
    params.beta = beta;

    // Normal spring constant (Hertz)
    float sqrt_Rd = sqrtf(params.R_eff * delta_n);
    params.K_n = 2.0f * E * sqrt_Rd / (3.0f * (1.0f - nu * nu));

    // Normal damping (from COR)
    if (COR > 0.0f && COR < 1.0f) {
        float ln_COR = logf(COR);
        float coeff = 2.0f * sqrtf(5.0f / 6.0f) * ln_COR /
                      sqrtf(ln_COR * ln_COR + 9.8696044f);
        float sqrt_delta = sqrtf(sqrtf(delta_n));
        float sqrt_Reff = sqrtf(sqrtf(params.R_eff));
        params.C_n = coeff * sqrtf(params.M_eff * E / (1.0f - nu * nu)) *
                     sqrt_Reff * sqrt_delta;
    } else {
        params.C_n = 0.0f;
    }

    // Tangential spring constant (Mindlin)
    params.K_s = 2.0f * E / (1.0f + nu) / (2.0f - nu) *
                 sqrtf(params.R_eff) * sqrtf(delta_n);

    // Tangential damping
    if (COR > 0.0f && COR < 1.0f) {
        float ln_COR = logf(COR);
        float coeff = 2.0f * sqrtf(5.0f / 6.0f) * ln_COR /
                      sqrtf(ln_COR * ln_COR + 9.8696044f);
        float sqrt_delta = sqrtf(sqrtf(delta_n));
        float sqrt_Reff = sqrtf(sqrtf(params.R_eff));
        params.C_s = coeff * sqrtf(2.0f * params.M_eff * E / (1.0f + nu) / (2.0f - nu)) *
                     sqrt_Reff * sqrt_delta;
    } else {
        params.C_s = 0.0f;
    }

    // Rolling and twisting springs/damping
    float beta_R_sq = (beta * params.R_eff) * (beta * params.R_eff);
    params.K_r = 0.25f * params.K_n * beta_R_sq;
    params.C_r = 0.25f * params.C_n * beta_R_sq;
    params.K_t = 0.5f * params.K_s * beta_R_sq;
    params.C_t = 0.5f * params.C_s * beta_R_sq;

    return params;
}

// Compute wall contact force kernel
template<bool EnableFriction>
__global__ void compute_wall_forces_kernel(
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
    // Material parameters
    float E, float nu, float COR,
    float mu_s, float beta,
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

    // Wall history for this particle
    int32_t hist_offset = wall_history_offset[idx];
    WallContactHistory* my_history = wall_history + hist_offset;
    int32_t my_num_contacts = num_wall_contacts[idx];

    // Accumulate forces
    double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
    double tx_acc = 0.0, ty_acc = 0.0, tz_acc = 0.0;

    int32_t new_contact_count = 0;
    int32_t num_walls = dev_num_contact_walls;

    // Loop over all contact walls
    for (int32_t w = 0; w < num_walls; ++w) {
        const WallData& wall = dev_wall_data[w];

        if (!wall.enabled) continue;
        if (wall.type != WallType::PLAIN && wall.type != WallType::MOVING && wall.type != WallType::FINITE_PLAIN) continue;

        // Get wall velocity
        double wall_vx, wall_vy, wall_vz;
        if (wall.num_velocity_frames > 0) {
            interpolate_wall_velocity(
                wall_vx, wall_vy, wall_vz,
                current_time,
                wall.d_velocity_times,
                wall.d_velocity_x,
                wall.d_velocity_y,
                wall.d_velocity_z,
                wall.num_velocity_frames
            );
        } else {
            wall_vx = wall.velocity_x;
            wall_vy = wall.velocity_y;
            wall_vz = wall.velocity_z;
        }

        // Compute signed distance to wall
        float signed_dist = plane_signed_distance(
            pi_x, pi_y, pi_z,
            wall.plane.normal_x, wall.plane.normal_y, wall.plane.normal_z,
            wall.plane.point_x, wall.plane.point_y, wall.plane.point_z
        );

        float delta_n = compute_plane_overlap(signed_dist, R_i);

        // No contact
        if (delta_n <= 0.0f) {
            // Mark existing contact as inactive
            for (int32_t k = 0; k < my_num_contacts; ++k) {
                if (my_history[k].wall_id == wall.id) {
                    my_history[k].is_active = false;
                    break;
                }
            }
            continue;
        }

        // Finite wall area check — skip if particle is outside shelf bounds
        if (wall.is_finite) {
            bool in_area = is_in_finite_wall_area(
                pi_x, pi_y,
                wall.plane.point_x, wall.plane.point_y,
                wall.half_extent_x, wall.half_extent_y,
                R_i
            );
            if (!in_area) {
                for (int32_t k = 0; k < my_num_contacts; ++k) {
                    if (my_history[k].wall_id == wall.id) {
                        my_history[k].is_active = false;
                        break;
                    }
                }
                continue;
            }
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

        // Compute material parameters
        WallMaterialParams params = compute_wall_material_params(
            R_i, m_i, I_i, delta_n, E, nu, COR, mu_s, beta
        );

        float Ap = signed_dist;  // Distance from center to contact point

        // Relative velocity
        double Vrel_x = vi_x - wall_vx;
        double Vrel_y = vi_y - wall_vy;
        double Vrel_z = vi_z - wall_vz;

        // Rotational contribution
        double Vrot_x = (wall.plane.normal_y * wi_z - wall.plane.normal_z * wi_y) * Ap;
        double Vrot_y = (wall.plane.normal_z * wi_x - wall.plane.normal_x * wi_z) * Ap;
        double Vrot_z = (wall.plane.normal_x * wi_y - wall.plane.normal_y * wi_x) * Ap;

        double Vtot_x = Vrel_x + Vrot_x;
        double Vtot_y = Vrel_y + Vrot_y;
        double Vtot_z = Vrel_z + Vrot_z;

        // Normal component
        double Vn_dot = Vtot_x * wall.plane.normal_x +
                       Vtot_y * wall.plane.normal_y +
                       Vtot_z * wall.plane.normal_z;
        double Vn_x = Vn_dot * wall.plane.normal_x;
        double Vn_y = Vn_dot * wall.plane.normal_y;
        double Vn_z = Vn_dot * wall.plane.normal_z;

        // Tangential component
        double Vt_x = Vtot_x - Vn_x;
        double Vt_y = Vtot_y - Vn_y;
        double Vt_z = Vtot_z - Vn_z;

        // Normal force
        // C_n < 0 (from log(COR)); Vn_dot > 0 when separating.
        // Must use signed Vn_dot so damping opposes motion in both directions.
        float Fn_mag = params.K_n * delta_n + params.C_n * static_cast<float>(Vn_dot);

        float Fn_x = Fn_mag * wall.plane.normal_x;
        float Fn_y = Fn_mag * wall.plane.normal_y;
        float Fn_z = Fn_mag * wall.plane.normal_z;

        fx_acc += Fn_x;
        fy_acc += Fn_y;
        fz_acc += Fn_z;

        // Updated history values
        float H_t_new_x = 0.0f, H_t_new_y = 0.0f, H_t_new_z = 0.0f;
        float H_r_new_x = 0.0f, H_r_new_y = 0.0f, H_r_new_z = 0.0f;
        float H_tw_new_x = 0.0f, H_tw_new_y = 0.0f, H_tw_new_z = 0.0f;
        bool is_slipping = false, is_rolling = false, is_twisting = false;

        if constexpr (EnableFriction) {
            if (!wall.is_smooth) {
                // Tangential force
                float Ds_x = Vt_x * dt;
                float Ds_y = Vt_y * dt;
                float Ds_z = Vt_z * dt;

                float Ft_x = -params.K_s * Ds_x + params.C_s * Vt_x + H_t_x;
                float Ft_y = -params.K_s * Ds_y + params.C_s * Vt_y + H_t_y;
                float Ft_z = -params.K_s * Ds_z + params.C_s * Vt_z + H_t_z;

                float Ft_mag = sqrtf(Ft_x * Ft_x + Ft_y * Ft_y + Ft_z * Ft_z);
                float Ft_max = params.mu_s * Fn_mag;

                if (Ft_mag > Ft_max && Ft_mag > 1e-10f) {
                    float scale = params.mu_d * Fn_mag / Ft_mag;
                    Ft_x *= scale;
                    Ft_y *= scale;
                    Ft_z *= scale;
                    H_t_new_x = Ft_x;
                    H_t_new_y = Ft_y;
                    H_t_new_z = Ft_z;
                    is_slipping = true;
                } else {
                    H_t_new_x = Ft_x - params.C_s * Vt_x;
                    H_t_new_y = Ft_y - params.C_s * Vt_y;
                    H_t_new_z = Ft_z - params.C_s * Vt_z;
                }

                fx_acc += Ft_x;
                fy_acc += Ft_y;
                fz_acc += Ft_z;

                // Torque from tangential force
                tx_acc -= Ap * (wall.plane.normal_y * Ft_z - wall.plane.normal_z * Ft_y);
                ty_acc -= Ap * (wall.plane.normal_z * Ft_x - wall.plane.normal_x * Ft_z);
                tz_acc -= Ap * (wall.plane.normal_x * Ft_y - wall.plane.normal_y * Ft_x);

                // Rolling moment
                double dtheta_x = wi_x * dt;
                double dtheta_y = wi_y * dt;
                double dtheta_z = wi_z * dt;

                double dtheta_n = dtheta_x * wall.plane.normal_x +
                                 dtheta_y * wall.plane.normal_y +
                                 dtheta_z * wall.plane.normal_z;
                double dtheta_t_x = dtheta_n * wall.plane.normal_x;
                double dtheta_t_y = dtheta_n * wall.plane.normal_y;
                double dtheta_t_z = dtheta_n * wall.plane.normal_z;

                double dtheta_r_x = dtheta_x - dtheta_t_x;
                double dtheta_r_y = dtheta_y - dtheta_t_y;
                double dtheta_r_z = dtheta_z - dtheta_t_z;

                float Mr_x = -params.K_r * dtheta_r_x + H_r_x;
                float Mr_y = -params.K_r * dtheta_r_y + H_r_y;
                float Mr_z = -params.K_r * dtheta_r_z + H_r_z;

                float Mr_mag = sqrtf(Mr_x * Mr_x + Mr_y * Mr_y + Mr_z * Mr_z);
                float Mr_max = 2.1f * 0.25f * params.beta * params.R_eff * Fn_mag;

                if (Mr_mag > Mr_max && Mr_mag > 1e-10f) {
                    float scale = Mr_max / Mr_mag;
                    Mr_x *= scale;
                    Mr_y *= scale;
                    Mr_z *= scale;
                    is_rolling = true;
                }
                H_r_new_x = Mr_x;
                H_r_new_y = Mr_y;
                H_r_new_z = Mr_z;

                tx_acc += Mr_x;
                ty_acc += Mr_y;
                tz_acc += Mr_z;

                // Twisting moment
                float Mt_x = -params.K_t * dtheta_t_x + H_tw_x;
                float Mt_y = -params.K_t * dtheta_t_y + H_tw_y;
                float Mt_z = -params.K_t * dtheta_t_z + H_tw_z;

                float Mt_mag = sqrtf(Mt_x * Mt_x + Mt_y * Mt_y + Mt_z * Mt_z);
                float Mt_max = 0.65f * params.mu_s * params.beta * params.R_eff * Fn_mag;

                if (Mt_mag > Mt_max && Mt_mag > 1e-10f) {
                    float scale = Mt_max / Mt_mag;
                    Mt_x *= scale;
                    Mt_y *= scale;
                    Mt_z *= scale;
                    is_twisting = true;
                }
                H_tw_new_x = Mt_x;
                H_tw_new_y = Mt_y;
                H_tw_new_z = Mt_z;

                tx_acc += Mt_x;
                ty_acc += Mt_y;
                tz_acc += Mt_z;
            }
        }

        // Update contact history
        if (slot < 0 && new_contact_count + my_num_contacts < max_wall_contacts_per_particle) {
            slot = my_num_contacts + new_contact_count;
            new_contact_count++;
        }

        if (slot >= 0 && slot < max_wall_contacts_per_particle) {
            my_history[slot].wall_id = wall.id;
            my_history[slot].delta_t_x = H_t_new_x;
            my_history[slot].delta_t_y = H_t_new_y;
            my_history[slot].delta_t_z = H_t_new_z;
            my_history[slot].delta_r_x = H_r_new_x;
            my_history[slot].delta_r_y = H_r_new_y;
            my_history[slot].delta_r_z = H_r_new_z;
            my_history[slot].delta_tw_x = H_tw_new_x;
            my_history[slot].delta_tw_y = H_tw_new_y;
            my_history[slot].delta_tw_z = H_tw_new_z;
            my_history[slot].is_active = true;
            my_history[slot].is_slipping = is_slipping;
            my_history[slot].is_rolling = is_rolling;
            my_history[slot].is_twisting = is_twisting;
        }
    }

    // Atomic add to global arrays
    atomic_add_force(force_x, force_y, force_z, fx_acc, fy_acc, fz_acc, idx);
    atomic_add_torque(torque_x, torque_y, torque_z, tx_acc, ty_acc, tz_acc, idx);

    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// Periodic Boundary Wrapping Kernel
// =============================================================================

static __global__ void wrap_positions_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    PeriodicBoundaryConfig pbc = dev_pbc_config;

    if (!pbc.is_enabled()) return;

    double x = pos_x[idx];
    double y = pos_y[idx];
    double z = pos_z[idx];

    wrap_position_double(x, y, z, pbc);

    pos_x[idx] = x;
    pos_y[idx] = y;
    pos_z[idx] = z;
}

// =============================================================================
// Wall Contact History Cleanup Kernel
// =============================================================================

__global__ void cleanup_wall_history_kernel(
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

// =============================================================================
// WallManager Implementation
// =============================================================================

void WallManager::initialize(
    SimulationState& state,
    const SimConfig& config
) {
    if (initialized) {
        std::cerr << "WARNING: WallManager already initialized, reinitializing..." << std::endl;
        shutdown();
    }

    // Extract periodic boundary config
    pbc_config = extract_pbc_config(config);
    if (config.shear_box.enabled) {
        pbc_config.periodic_x = true;
        pbc_config.periodic_y = true;
        pbc_config.periodic_z = false;
        pbc_config.box_min_x = config.domain.box_min.x;
        pbc_config.box_max_x = config.domain.box_max.x;
        pbc_config.box_min_y = config.domain.box_min.y;
        pbc_config.box_max_y = config.domain.box_max.y;
        pbc_config.box_min_z = config.domain.box_min.z;
        pbc_config.box_max_z = config.domain.box_max.z;
        pbc_config.compute_derived();
    }

    // Copy PBC config to constant memory
    cudaMemcpyToSymbol(dev_pbc_config_storage, &pbc_config, sizeof(PeriodicBoundaryConfig));

    // Prepare contact walls (plain and moving)
    std::vector<WallData> contact_walls;
    velocity_data_storage.clear();

    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];

        if (wc.type == WallType::PLAIN || wc.type == WallType::MOVING || wc.type == WallType::FINITE_PLAIN) {
            WallData wd = wall_config_to_data(wc);

            // Load velocity data for moving walls
            if (wc.type == WallType::MOVING && wc.velocity_data_path[0] != '\0') {
                MovingWallVelocityData vd;
                if (load_velocity_data(vd, wc.velocity_data_path, wc.type)) {
                    vd.copy_to_device();
                    wd.d_velocity_times = vd.d_times;
                    wd.d_velocity_x = vd.d_vx;
                    wd.d_velocity_y = vd.d_vy;
                    wd.d_velocity_z = vd.d_vz;
                    wd.num_velocity_frames = vd.num_frames;
                    velocity_data_storage.push_back(std::move(vd));
                }
            }

            contact_walls.push_back(wd);
        }
    }

    // Copy contact walls to constant memory
    int32_t num_contact_walls = static_cast<int32_t>(contact_walls.size());
    if (num_contact_walls > MAX_PLAIN_WALLS) {
        std::cerr << "WARNING: Too many contact walls (" << num_contact_walls
                  << "), truncating to " << MAX_PLAIN_WALLS << std::endl;
        num_contact_walls = MAX_PLAIN_WALLS;
    }

    if (num_contact_walls > 0) {
        cudaMemcpyToSymbol(dev_wall_data_storage, contact_walls.data(),
                           num_contact_walls * sizeof(WallData));
    }
    cudaMemcpyToSymbol(dev_num_contact_walls, &num_contact_walls, sizeof(int32_t));
    host_num_contact_walls = num_contact_walls;

    // Cache contact walls data for multi-GPU per-device replication
    h_contact_walls_cache = contact_walls;

    // The material-aware plain dispatcher owns its compact constant-memory
    // representation separately from this legacy manager cache.
    plain_wall_initialize(config);

    // Allocate wall contact history if not already done
    if (state.wall_contact_history.data == nullptr && state.particles.num_particles > 0) {
        allocate_wall_contact_history(state.wall_contact_history,
                                      state.particles.num_particles,
                                      16);  // 16 wall contacts per particle max (plain/moving + bonded rect)

        // Allocate wall history offset array if not pre-allocated by allocate_particle_soa
        if (state.particles.wall_history_offset == nullptr) {
            cudaMalloc(&state.particles.wall_history_offset,
                       state.particles.num_particles * sizeof(int32_t));
        }
        // Always initialize offsets: particle i starts at i * max_contacts_per_particle.
        // allocate_particle_soa pre-allocates this array but leaves it uninitialized,
        // so the stride pattern must always be written here.
        {
            std::vector<int32_t> offsets(state.particles.num_particles);
            for (int32_t i = 0; i < state.particles.num_particles; ++i) {
                offsets[i] = i * state.wall_contact_history.max_contacts_per_particle;
            }
            cudaMemcpy(state.particles.wall_history_offset, offsets.data(),
                       state.particles.num_particles * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
        }

        if (state.particles.num_wall_contacts == nullptr) {
            cudaMalloc(&state.particles.num_wall_contacts,
                       state.particles.num_particles * sizeof(int32_t));
        }
        // Always zero num_wall_contacts (may have been pre-allocated but not zeroed properly)
        cudaMemset(state.particles.num_wall_contacts, 0,
                   state.particles.num_particles * sizeof(int32_t));
    }

    initialized = true;

    // Initialize trimesh walls
    trimesh_wall_initialize(config, state.particles.num_particles);
    if (trimesh_wall_is_initialized()) {
        trimesh_wall_build_all_bvhs(config.particle.radius_max, config.gpu.block_size);
    }

    // Bonded rect walls
    bonded_rect_wall_initialize(config, state.particles.num_particles);

    // Funnel walls
    FunnelWallDispatcher::initialize_walls(config);

    // Sphere wall
    SphereWallManager::initialize(config, state.particles.num_particles);

    // Print summary
    WallStatistics stats = compute_wall_statistics(config);
    std::cout << "WallManager initialized:" << std::endl;
    std::cout << "  Contact walls: " << num_contact_walls
              << " (plain: " << stats.num_plain_walls
              << ", moving: " << stats.num_moving_walls
              << ", finite: " << stats.num_finite_plain_walls
              << ", funnel: " << stats.num_funnel_walls << ")" << std::endl;
    std::cout << "  Periodic axes: " << stats.num_periodic_axes << std::endl;

    if (pbc_config.is_enabled()) {
        print_pbc_config(pbc_config);
    }
}

void WallManager::compute_forces(
    SimulationState& state,
    const SimConfig& config,
    float dt,
    float current_time,
    cudaStream_t stream
) {
    if (!initialized) {
        std::cerr << "ERROR: WallManager not initialized" << std::endl;
        return;
    }

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    // Plain/moving wall forces
    int32_t num_contact_walls = host_num_contact_walls;
    if (num_contact_walls > 0) {
        plain_wall_compute_forces(state, config, dt, current_time, stream);
    }

    // Trimesh wall forces — NOT supported in multi-GPU mode
    // BVH/mesh data lives on GPU 0 only; other GPUs cannot safely access via P2P
    // without replication. Skip to avoid illegal memory access.
    if (trimesh_wall_is_initialized() && !config.multigpu.enabled) {
        trimesh_wall_compute_forces(state, config, dt, current_time, stream);
    } else if (trimesh_wall_is_initialized() && config.multigpu.enabled) {
        // Trimesh walls disabled for multi-GPU — data not replicated across devices
        std::cerr << "WARNING: Trimesh wall forces skipped in multi-GPU mode "
                  << "(BVH/mesh data not replicated to all GPUs)" << std::endl;
    }

    // Bonded rect wall forces + 6-DOF integration — NOT supported in multi-GPU mode
    // d_bonded_states/d_bonded_reactions live on GPU 0; skip to avoid P2P issues.
    if (bonded_rect_wall_is_initialized() && !config.multigpu.enabled) {
        bonded_rect_wall_compute_forces(state, config, dt, current_time, stream);
        cudaStreamSynchronize(stream);   // reactions must complete before integration reads them

        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        int32_t num_walls = bonded_rect_wall_get_states(nullptr, nullptr);
        if (num_walls > 0) {
            std::vector<BondedRectWallState> host_states(num_walls);
            bonded_rect_wall_get_states(host_states.data(), nullptr);
            // Query gravity at the center of the first bonded wall
            AccelerationManager::query_gravity_at_position(
                host_states[0].pos_x, host_states[0].pos_y, host_states[0].pos_z,
                gx, gy, gz
            );
        }

        bonded_rect_wall_integrate(dt, gx, gy, gz, stream);
    } else if (bonded_rect_wall_is_initialized() && config.multigpu.enabled) {
        // Bonded rect walls disabled for multi-GPU — data not replicated across devices
        std::cerr << "WARNING: Bonded rect wall forces skipped in multi-GPU mode "
                  << "(device data not replicated to all GPUs)" << std::endl;
    }

    // Funnel wall forces
    if (FunnelWallDispatcher::get_num_walls() > 0) {
        FunnelWallDispatcher::compute_forces(state, config, dt, stream);
    }

    // Sphere wall: update state then compute contact forces
    SphereWallManager::update_state(current_time);
    SphereWallManager::compute_contact_forces(state, config, dt, stream);
}

void WallManager::apply_periodic_boundaries(
    SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream
) {
    // ShearBoxManager performs the coupled x/y remap and velocity jump.
    if (config.shear_box.enabled) return;
    if (!pbc_config.is_enabled()) return;

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(wrap_positions_kernel, num_blocks, block_size, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        num_particles
    );
}

void WallManager::cleanup_contact_history(
    SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream
) {
    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;
    if (state.wall_contact_history.data == nullptr) return;

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(cleanup_wall_history_kernel, num_blocks, block_size, 0, stream,
        state.wall_contact_history.data,
        state.particles.wall_history_offset,
        state.particles.num_wall_contacts,
        state.wall_contact_history.max_contacts_per_particle,
        num_particles
    );
}

void WallManager::copy_to_current_device() {
    // Recreate the compact plain-wall representation on this device.  Its
    // moving-wall time-series buffers are device-local, so a raw constant
    // memory copy would leave invalid pointers on secondary GPUs.
    plain_wall_copy_to_current_device();
    // Copy all wall-related __constant__ memory to the current CUDA device.
    // Called during multi-GPU per-device initialization so each GPU has
    // its own copy of wall data, PBC config, etc.
    // __constant__ memory is device-local and NOT shared via P2P.
    if (!initialized) {
        std::cerr << "WARNING: WallManager not initialized, cannot copy to device" << std::endl;
        return;
    }

    // Copy wall data (plain/moving/finite) to this device's constant memory
    if (host_num_contact_walls > 0 && !h_contact_walls_cache.empty()) {
        cudaMemcpyToSymbol(dev_wall_data_storage, h_contact_walls_cache.data(),
                           host_num_contact_walls * sizeof(WallData));
    }
    cudaMemcpyToSymbol(dev_num_contact_walls, &host_num_contact_walls, sizeof(int32_t));

    // Copy PBC config to this device's constant memory
    cudaMemcpyToSymbol(dev_pbc_config_storage, &pbc_config, sizeof(PeriodicBoundaryConfig));

    // Copy funnel wall data to this device's constant memory
    copy_funnel_to_current_device();
    SphereWallManager::copy_to_current_device();
}

void WallManager::shutdown() {
    trimesh_wall_shutdown();
    bonded_rect_wall_shutdown();
    SphereWallManager::shutdown();
    velocity_data_storage.clear();
    pbc_config = PeriodicBoundaryConfig();
    initialized = false;
}

} // namespace cudadem

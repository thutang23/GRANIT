// =============================================================================
// Sphere Wall Implementation — Prescribed-trajectory massive sphere
// =============================================================================
// One-way contact coupling (same Hertz-Mindlin as particle-particle)
// and point-mass gravity on all DEM particles.
// Wall state (pos, vel, omega) is prescribed from a binary trajectory file.
// =============================================================================

#include "sphere_wall.h"
#include "wall_contact_force.cuh"
#include "../particle_model/particle.h"
#include "../acceleration/acceleration.h"
#include "../constant/system_initialize.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstring>
#include <algorithm>

namespace cudadem {

// =============================================================================
// GPU Constant Memory — Raw storage pattern (avoids dynamic initialization)
// =============================================================================

__constant__ __align__(16) unsigned char dev_sphere_wall_state_storage[sizeof(SphereWallState)];
#define dev_sphere_wall_state (*reinterpret_cast<SphereWallState*>(const_cast<unsigned char*>(dev_sphere_wall_state_storage)))

// =============================================================================
// Static Member Definitions
// =============================================================================

SphereWallTrajectoryData SphereWallManager::trajectory_data_;
bool SphereWallManager::initialized_ = false;
SphereWallState SphereWallManager::host_state_;

// =============================================================================
// Contact Force Kernel
// =============================================================================
// Per-particle kernel. Uses compute_contact_forces() from particle.h
// treating particle i as "i" and sphere wall as "j".
// One-way coupling: forces/torques applied only to particle i.
// =============================================================================

__global__ void compute_sphere_wall_contact_forces_kernel(
    // Particle SoA arrays
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const float*  __restrict__ radius,
    const float*  __restrict__ mass,
    const float*  __restrict__ moment_of_inertia,
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
    // Timestep
    float dt,
    // PBC config (passed as individual fields)
    bool pbc_x, bool pbc_y, bool pbc_z,
    float pbc_box_x, float pbc_box_y, float pbc_box_z,
    float pbc_half_x, float pbc_half_y, float pbc_half_z,
    // Particle count
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Read sphere wall state from constant memory
    SphereWallState wall = dev_sphere_wall_state;
    if (!wall.enabled) return;

    // Load particle data
    float  pi_x = static_cast<float>(pos_x[idx]);
    float  pi_y = static_cast<float>(pos_y[idx]);
    float  pi_z = static_cast<float>(pos_z[idx]);
    double vi_x = vel_x[idx];
    double vi_y = vel_y[idx];
    double vi_z = vel_z[idx];
    double wi_x = omega_x[idx];
    double wi_y = omega_y[idx];
    double wi_z = omega_z[idx];
    float  R_i  = radius[idx];
    float  m_i  = mass[idx];
    float  I_i  = moment_of_inertia[idx];
    const int32_t particle_material_id = material_id[idx];

    // Sphere wall properties (as "particle j")
    float  pj_x = static_cast<float>(wall.pos_x);
    float  pj_y = static_cast<float>(wall.pos_y);
    float  pj_z = static_cast<float>(wall.pos_z);
    double vj_x = wall.vel_x;
    double vj_y = wall.vel_y;
    double vj_z = wall.vel_z;
    double wj_x = wall.omega_x;
    double wj_y = wall.omega_y;
    double wj_z = wall.omega_z;
    float  R_j  = wall.radius;
    float  m_j  = wall.mass;
    float  I_j  = wall.moment_of_inertia;
    float E_star, G_star;
    DemInteractionProperties pair{};
    if (!resolve_wall_material_pair(particle_material_id, wall.material_id,
                                    materials, interactions, material_count,
                                    E_star, G_star, pair)) return;

    // Distance vector from i to j
    float dx = pj_x - pi_x;
    float dy = pj_y - pi_y;
    float dz = pj_z - pi_z;

    // Minimum image convention for PBC
    if (pbc_x && pbc_box_x > 0.0f) {
        if (dx > pbc_half_x) dx -= pbc_box_x;
        else if (dx < -pbc_half_x) dx += pbc_box_x;
    }
    if (pbc_y && pbc_box_y > 0.0f) {
        if (dy > pbc_half_y) dy -= pbc_box_y;
        else if (dy < -pbc_half_y) dy += pbc_box_y;
    }
    if (pbc_z && pbc_box_z > 0.0f) {
        if (dz > pbc_half_z) dz -= pbc_box_z;
        else if (dz < -pbc_half_z) dz += pbc_box_z;
    }

    // Wall contact history for this particle
    int32_t hist_offset = wall_history_offset[idx];
    WallContactHistory* my_history = wall_history + hist_offset;
    int32_t my_num_contacts = num_wall_contacts[idx];

    // Look up WallContactHistory slot for sphere wall
    int slot = -1;
    float H_t_x = 0.0f, H_t_y = 0.0f, H_t_z = 0.0f;
    float H_r_x = 0.0f, H_r_y = 0.0f, H_r_z = 0.0f;
    float H_tw_x = 0.0f, H_tw_y = 0.0f, H_tw_z = 0.0f;

    for (int32_t k = 0; k < my_num_contacts; ++k) {
        if (my_history[k].wall_id == SPHERE_WALL_ID && my_history[k].is_active) {
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

    // Call compute_contact_forces() — particle i vs sphere wall j
    ContactForceResult result = compute_contact_forces(
        // Particle i
        pi_x, pi_y, pi_z,
        vi_x, vi_y, vi_z,
        wi_x, wi_y, wi_z,
        R_i, m_i, I_i,
        // Particle j (sphere wall)
        pi_x + dx, pi_y + dy, pi_z + dz,  // pos_j adjusted for PBC
        vj_x, vj_y, vj_z,
        wj_x, wj_y, wj_z,
        R_j, m_j, I_j,
        // Contact history
        H_t_x, H_t_y, H_t_z,
        H_r_x, H_r_y, H_r_z,
        H_tw_x, H_tw_y, H_tw_z,
        // Material parameters
        E_star, G_star, pair.restitution,
        pair.friction_static, pair.friction_rolling, pair.friction_twisting,
        pair.shape_parameter, pair.cohesion_cutoff_factor,
        use_visco, pair.viscous_coefficient,
        pair.cohesion_enabled != 0, pair.cohesion_strength,
        // Timestep
        dt
    );

    // One-way coupling: apply forces/torques only to particle i
    atomicAdd(&force_x[idx],  result.force_x);
    atomicAdd(&force_y[idx],  result.force_y);
    atomicAdd(&force_z[idx],  result.force_z);
    atomicAdd(&torque_x[idx], result.torque_x);
    atomicAdd(&torque_y[idx], result.torque_y);
    atomicAdd(&torque_z[idx], result.torque_z);

    // Update WallContactHistory slot
    int32_t new_contact_count = 0;
    if (result.is_contact) {
        if (slot < 0 && my_num_contacts + new_contact_count < max_wall_contacts_per_particle) {
            slot = my_num_contacts + new_contact_count;
            new_contact_count++;
        }

        if (slot >= 0 && slot < max_wall_contacts_per_particle) {
            my_history[slot].wall_id = SPHERE_WALL_ID;
            my_history[slot].delta_t_x  = result.delta_t_x;
            my_history[slot].delta_t_y  = result.delta_t_y;
            my_history[slot].delta_t_z  = result.delta_t_z;
            my_history[slot].delta_r_x  = result.delta_r_x;
            my_history[slot].delta_r_y  = result.delta_r_y;
            my_history[slot].delta_r_z  = result.delta_r_z;
            my_history[slot].delta_tw_x = result.delta_tw_x;
            my_history[slot].delta_tw_y = result.delta_tw_y;
            my_history[slot].delta_tw_z = result.delta_tw_z;
            my_history[slot].is_active    = true;
            my_history[slot].is_slipping  = result.is_slipping;
            my_history[slot].is_rolling   = result.is_rolling;
            my_history[slot].is_twisting  = result.is_twisting;
        }
    } else {
        // Mark existing contact as inactive if no overlap
        if (slot >= 0) {
            my_history[slot].is_active = false;
        }
    }

    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// Gravity Kernel
// =============================================================================
// Per-particle kernel. Computes G*M/r² gravitational acceleration
// from sphere wall on each particle i. One-way (no back-reaction).
// =============================================================================

__global__ void compute_sphere_wall_gravity_kernel(
    // Particle positions
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle masses
    const float*  __restrict__ mass,
    // Force output
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    // Softening
    float softening,
    // PBC config
    bool pbc_x, bool pbc_y, bool pbc_z,
    float pbc_box_x, float pbc_box_y, float pbc_box_z,
    float pbc_half_x, float pbc_half_y, float pbc_half_z,
    // Particle count
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    SphereWallState wall = dev_sphere_wall_state;
    if (!wall.enabled) return;

    float pi_x = static_cast<float>(pos_x[idx]);
    float pi_y = static_cast<float>(pos_y[idx]);
    float pi_z = static_cast<float>(pos_z[idx]);
    float m_i  = mass[idx];

    // Distance vector from particle to sphere wall
    float dx = static_cast<float>(wall.pos_x) - pi_x;
    float dy = static_cast<float>(wall.pos_y) - pi_y;
    float dz = static_cast<float>(wall.pos_z) - pi_z;

    // Minimum image convention for PBC
    if (pbc_x && pbc_box_x > 0.0f) {
        if (dx > pbc_half_x) dx -= pbc_box_x;
        else if (dx < -pbc_half_x) dx += pbc_box_x;
    }
    if (pbc_y && pbc_box_y > 0.0f) {
        if (dy > pbc_half_y) dy -= pbc_box_y;
        else if (dy < -pbc_half_y) dy += pbc_box_y;
    }
    if (pbc_z && pbc_box_z > 0.0f) {
        if (dz > pbc_half_z) dz -= pbc_box_z;
        else if (dz < -pbc_half_z) dz += pbc_box_z;
    }

    // Distance with softening
    float r2 = dx * dx + dy * dy + dz * dz + softening * softening;
    float r  = sqrtf(r2);
    float r3 = r2 * r;

    // Gravitational force: F = G * m_i * M_wall / r^2 * r_hat
    // factor = G * M / r^3 (so force_i = m_i * factor * dx/dy/dz)
    double G = GRAVITATIONAL_CONSTANT;
    float M_wall = wall.mass;
    double factor = G * static_cast<double>(M_wall) / static_cast<double>(r3);

    // Apply force: atomicAdd since other kernels may also write to force arrays
    atomicAdd(&force_x[idx], m_i * factor * dx);
    atomicAdd(&force_y[idx], m_i * factor * dy);
    atomicAdd(&force_z[idx], m_i * factor * dz);
}

// =============================================================================
// SphereWallManager — Static Method Implementations
// =============================================================================

void SphereWallManager::initialize(const SimConfig& config, int32_t num_particles) {
    if (initialized_) {
        std::cerr << "WARNING: SphereWallManager already initialized, reinitializing..." << std::endl;
        shutdown();
    }

    // Search for a SPHERE wall in config
    bool found = false;
    const WallConfig* sphere_wc = nullptr;
    for (int32_t i = 0; i < config.num_walls; ++i) {
        if (config.walls[i].type == WallType::SPHERE && config.walls[i].enabled) {
            found = true;
            sphere_wc = &config.walls[i];
            break;
        }
    }

    if (!found) {
        // No sphere wall — leave disabled
        host_state_.enabled = false;
        cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
        initialized_ = true;
        return;
    }

    // Load trajectory from binary file
    bool loaded = trajectory_data_.load_from_file(sphere_wc->sphere_trajectory_path);
    if (!loaded) {
        throw std::runtime_error("sphere wall trajectory initialization failed");
    }

    // Copy trajectory data to device (for device-side binary search if needed)
    trajectory_data_.copy_to_device();

    // Set up host state
    host_state_.enabled = true;
    host_state_.material_id = sphere_wc->material_id;
    host_state_.radius = sphere_wc->sphere_radius;
    host_state_.mass = sphere_wc->sphere_mass;
    host_state_.moment_of_inertia = 0.4f * sphere_wc->sphere_mass * sphere_wc->sphere_radius * sphere_wc->sphere_radius;

    // Initialize position/velocity/omega from first trajectory frame
    if (trajectory_data_.num_frames > 0) {
        host_state_.pos_x = trajectory_data_.pos_x[0];
        host_state_.pos_y = trajectory_data_.pos_y[0];
        host_state_.pos_z = trajectory_data_.pos_z[0];
        host_state_.vel_x = trajectory_data_.vel_x[0];
        host_state_.vel_y = trajectory_data_.vel_y[0];
        host_state_.vel_z = trajectory_data_.vel_z[0];
        host_state_.omega_x = trajectory_data_.omega_x[0];
        host_state_.omega_y = trajectory_data_.omega_y[0];
        host_state_.omega_z = trajectory_data_.omega_z[0];
    } else {
        host_state_.pos_x = 0; host_state_.pos_y = 0; host_state_.pos_z = 0;
        host_state_.vel_x = 0; host_state_.vel_y = 0; host_state_.vel_z = 0;
        host_state_.omega_x = 0; host_state_.omega_y = 0; host_state_.omega_z = 0;
    }

    // Copy initial state to constant memory
    cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));

    initialized_ = true;

    std::cout << "SphereWallManager initialized: radius=" << host_state_.radius
              << " mass=" << host_state_.mass
              << " I=" << host_state_.moment_of_inertia
              << " trajectory_frames=" << trajectory_data_.num_frames
              << std::endl;
}

void SphereWallManager::update_state(double current_time) {
    if (!initialized_ || !host_state_.enabled) return;

    int32_t n = trajectory_data_.num_frames;
    if (n <= 0) return;

    // Clamp to first frame
    if (current_time < trajectory_data_.times[0] || current_time > trajectory_data_.times[n - 1]) {
        if (!trajectory_data_.hold_out_of_range)
            throw std::runtime_error("sphere wall RIGIDPS2 trajectory does not cover current simulation time");
        const int32_t endpoint = current_time < trajectory_data_.times[0] ? 0 : n - 1;
        host_state_.pos_x = trajectory_data_.pos_x[endpoint]; host_state_.pos_y = trajectory_data_.pos_y[endpoint]; host_state_.pos_z = trajectory_data_.pos_z[endpoint];
        host_state_.vel_x = host_state_.vel_y = host_state_.vel_z = 0.0;
        host_state_.omega_x = host_state_.omega_y = host_state_.omega_z = 0.0;
        cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
        return;
    }
    if (n == 1 || current_time == trajectory_data_.times[0]) {
        host_state_.pos_x = trajectory_data_.pos_x[0];
        host_state_.pos_y = trajectory_data_.pos_y[0];
        host_state_.pos_z = trajectory_data_.pos_z[0];
        host_state_.vel_x = trajectory_data_.vel_x[0];
        host_state_.vel_y = trajectory_data_.vel_y[0];
        host_state_.vel_z = trajectory_data_.vel_z[0];
        host_state_.omega_x = trajectory_data_.omega_x[0];
        host_state_.omega_y = trajectory_data_.omega_y[0];
        host_state_.omega_z = trajectory_data_.omega_z[0];
        cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
        return;
    }

    // Clamp to last frame
    if (current_time == trajectory_data_.times[n - 1]) {
        host_state_.pos_x = trajectory_data_.pos_x[n - 1];
        host_state_.pos_y = trajectory_data_.pos_y[n - 1];
        host_state_.pos_z = trajectory_data_.pos_z[n - 1];
        host_state_.vel_x = trajectory_data_.vel_x[n - 1];
        host_state_.vel_y = trajectory_data_.vel_y[n - 1];
        host_state_.vel_z = trajectory_data_.vel_z[n - 1];
        host_state_.omega_x = trajectory_data_.omega_x[n - 1];
        host_state_.omega_y = trajectory_data_.omega_y[n - 1];
        host_state_.omega_z = trajectory_data_.omega_z[n - 1];
        cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
        return;
    }

    // Binary search for time bracket
    int lo = 0, hi = n - 1;
    while (lo < hi - 1) {
        int mid = (lo + hi) / 2;
        if (trajectory_data_.times[mid] <= current_time) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation
    double t0 = trajectory_data_.times[lo];
    double t1 = trajectory_data_.times[hi];
    double alpha = (current_time - t0) / (t1 - t0);

    host_state_.pos_x = trajectory_data_.pos_x[lo] + alpha * (trajectory_data_.pos_x[hi] - trajectory_data_.pos_x[lo]);
    host_state_.pos_y = trajectory_data_.pos_y[lo] + alpha * (trajectory_data_.pos_y[hi] - trajectory_data_.pos_y[lo]);
    host_state_.pos_z = trajectory_data_.pos_z[lo] + alpha * (trajectory_data_.pos_z[hi] - trajectory_data_.pos_z[lo]);
    host_state_.vel_x = trajectory_data_.vel_x[lo] + alpha * (trajectory_data_.vel_x[hi] - trajectory_data_.vel_x[lo]);
    host_state_.vel_y = trajectory_data_.vel_y[lo] + alpha * (trajectory_data_.vel_y[hi] - trajectory_data_.vel_y[lo]);
    host_state_.vel_z = trajectory_data_.vel_z[lo] + alpha * (trajectory_data_.vel_z[hi] - trajectory_data_.vel_z[lo]);
    host_state_.omega_x = trajectory_data_.omega_x[lo] + alpha * (trajectory_data_.omega_x[hi] - trajectory_data_.omega_x[lo]);
    host_state_.omega_y = trajectory_data_.omega_y[lo] + alpha * (trajectory_data_.omega_y[hi] - trajectory_data_.omega_y[lo]);
    host_state_.omega_z = trajectory_data_.omega_z[lo] + alpha * (trajectory_data_.omega_z[hi] - trajectory_data_.omega_z[lo]);

    // Copy interpolated state to constant memory
    cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
}

void SphereWallManager::compute_contact_forces(
    SimulationState& state, const SimConfig& config,
    float dt, cudaStream_t stream
) {
    if (!initialized_ || !host_state_.enabled) return;

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    bool use_visco = config.dem_damping_model == DemDampingModel::VISCOUS;

    // PBC config from WallManager
    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(compute_sphere_wall_contact_forces_kernel,
        num_blocks, block_size, 0, stream,
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
        use_visco,
        dt,
        pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
        pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
        pbc.half_length_x, pbc.half_length_y, pbc.half_length_z,
        num_particles
    );
}

void SphereWallManager::compute_gravity(
    SimulationState& state, const SimConfig& config,
    cudaStream_t stream
) {
    if (!initialized_ || !host_state_.enabled) return;

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    // Determine softening length
    // Use SPH gravity_softening_len if available and > 0
    // Otherwise use 0.001 * sphere wall radius as default
    float softening = 0.001f * host_state_.radius;  // Default
    if (config.sph.enabled && config.sph.gravity_softening_len > 0.0f) {
        softening = config.sph.gravity_softening_len;
    } else if (config.acceleration.enable_self_gravity) {
        // Use self-gravity softening if available
        // SelfGravityConfig.softening is in AccelerationManager (not directly accessible here)
        // Use the displacement threshold as a heuristic scale
        softening = config.acceleration.displacement_threshold * host_state_.radius;
    }

    // PBC config from WallManager
    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(compute_sphere_wall_gravity_kernel,
        num_blocks, block_size, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.mass,
        state.particles.force_x, state.particles.force_y, state.particles.force_z,
        softening,
        pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
        pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
        pbc.half_length_x, pbc.half_length_y, pbc.half_length_z,
        num_particles
    );
}

void SphereWallManager::copy_to_current_device() {
    if (!initialized_) {
        std::cerr << "WARNING: SphereWallManager not initialized, cannot copy to device" << std::endl;
        return;
    }
    cudaMemcpyToSymbol(dev_sphere_wall_state_storage, &host_state_, sizeof(SphereWallState));
}

void SphereWallManager::shutdown() {
    trajectory_data_.free_device();
    host_state_ = SphereWallState();  // Reset to disabled
    initialized_ = false;
}

bool SphereWallManager::is_enabled() {
    return host_state_.enabled;
}

const SphereWallState& SphereWallManager::get_host_state() {
    return host_state_;
}

} // namespace cudadem

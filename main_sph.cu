#include "src/core/constant/adaptive_fmm_depth_lifecycle.h"
#include "src/core/sph/sph_respa_schedule.h"
#include "src/core/sph/sph_boundary.h"
#include "src/core/sph/sph_stage.h"
#include "src/core/sph/sph_history.h"
// =============================================================================
// CudaDEM Main Entry Point
// =============================================================================
// GPU-Accelerated Discrete Element Method for Asteroid Dynamics
//
// Simulation Pipeline:
// 1. Parse YAML configuration -> SimConfig
// 2. Copy SimConfig to GPU constant memory
// 3. Initialize system (particles, walls, neighbor structures)
// 4. Main time loop with Velocity Verlet integration
// 5. Termination by time OR energy threshold
//
// Design Philosophy:
// - Runtime Configuration: All physics toggles in SimConfig (no recompilation)
// - Simple kernel launches (no complex template policy per force combination)
// - SoA data layout for coalesced GPU memory access
// =============================================================================

#include <nvtx3/nvToolsExt.h>

#ifdef CUDADEM_CPU_MOCK
#define DEFINE_CUDA_MOCK_GLOBALS
#endif
#include "cuda_utils.h"

#include "src/core/constant/command_line_interface.h"
#include "src/core/constant/system_initialize.h"
#include "src/core/constant/load_or_resume.h"
#include "src/core/constant/time_step_schedule.h"
#include "src/core/constant/binary_format.h"
#include "src/core/neighbor/neighbor_api.h"
#include "src/core/wall/wall.h"
#include "src/core/acceleration/acceleration.h"
#include "src/core/math/verlet_integration.cuh"
#include "src/core/particle_model/particle.h"
#include "src/core/sph/sph_physics.h"
#include "src/core/math/verlet_respa_kernels.cuh"
#include "src/core/sph/sph_input.h"
#include "src/core/sph/sph_timestep.h"
#include "src/core/sph/sph_config_printer.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <filesystem>

// Thrust for reductions
#ifndef CUDADEM_CPU_MOCK
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#endif

namespace cudadem {

// =============================================================================
// DEM Contact Force Kernels (Sorted Owner-Centric, Register Accumulation)
// =============================================================================
// These kernels are used for DEM and DEM+SPH hybrid mode.
// In pure SPH mode, these are NOT called (skipped via pure_sph_mode_ flag).
//
// Contacts are sorted by owner (contact.i) so each thread processes ALL
// contacts for one owner particle. Owner forces/torques are accumulated in
// registers and flushed with a single atomicAdd per particle — inspired by
// the FMM P2P kernel's register accumulation optimization.
// =============================================================================

// =============================================================================
// Kernel: Extract sort keys and init permutation indices
// =============================================================================
__global__ void extract_sort_keys_kernel(
    const ContactPair* __restrict__ contacts,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ indices,
    int32_t num_contacts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;
    keys[idx] = ((uint64_t)(uint32_t)contacts[idx].i << 32) | (uint64_t)(uint32_t)contacts[idx].j;
    indices[idx] = idx;
}

// =============================================================================
// Kernel: Permute contacts using sorted indices
// =============================================================================
__global__ void permute_contacts_kernel(
    const ContactPair* __restrict__ src,
    const int32_t* __restrict__ indices,
    ContactPair* __restrict__ dst,
    int32_t num_contacts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;
    dst[idx] = src[indices[idx]];
}

// =============================================================================
// Kernel: Build owner offsets from sorted keys (mark phase)
// =============================================================================
__global__ void build_owner_offsets_from_keys_kernel(
    const uint64_t* __restrict__ keys,
    int32_t* __restrict__ owner_offsets,
    int32_t num_contacts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;
    int32_t owner = (int32_t)(keys[idx] >> 32);
    if (idx == 0 || owner != (int32_t)(keys[idx - 1] >> 32)) {
        owner_offsets[owner] = idx;
    }
}

// =============================================================================
// Kernel: Init offsets to sentinel
// =============================================================================
__global__ void init_owner_offsets_kernel(
    int32_t* __restrict__ offsets, int32_t val, int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) offsets[i] = val;
}

// =============================================================================
// Kernel: Fill owner offset gaps (parallel)
// =============================================================================
__global__ void fill_owner_offset_gaps_kernel(
    int32_t* __restrict__ owner_offsets,
    int32_t num_particles, int32_t num_contacts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p > num_particles) return;
    if (p == num_particles) {
        owner_offsets[num_particles] = num_contacts;
        return;
    }
    if (owner_offsets[p] != num_contacts) return;
    int32_t next_val = num_contacts;
    for (int q = p + 1; q <= num_particles; ++q) {
        if (q == num_particles || owner_offsets[q] != num_contacts) {
            next_val = owner_offsets[q]; break;
        }
    }
    owner_offsets[p] = next_val;
}

// =============================================================================
// Host Helper: Sort contacts by owner via key-based radix sort
// =============================================================================
static void sort_contacts_by_owner(
    ContactPair* contacts, ContactPair* contacts_sorted,
    const int32_t* d_num_contacts,
    int32_t* d_owner_offsets,
    uint64_t* d_sort_keys, int32_t* d_sort_indices,
    int32_t num_particles, int32_t max_contacts,
    int block_size, cudaStream_t stream)
{
    int32_t h_num_contacts = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_contacts, d_num_contacts,
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_num_contacts = min(h_num_contacts, max_contacts);

    if (h_num_contacts == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_owner_offsets, 0,
            (num_particles + 1) * sizeof(int32_t), stream));
        return;
    }

    // Step 1: Extract keys + init indices
    int nb = (h_num_contacts + block_size - 1) / block_size;
    LAUNCH_KERNEL(extract_sort_keys_kernel, nb, block_size, 0, stream,
        contacts, d_sort_keys, d_sort_indices, h_num_contacts);

    // Step 2: Radix sort by key
    thrust::sort_by_key(thrust::cuda::par.on(stream),
        d_sort_keys, d_sort_keys + h_num_contacts, d_sort_indices);

    // Step 3: Build offsets from sorted keys (skip permute — force kernel gathers)
    int nbi = (num_particles + 1 + block_size - 1) / block_size;
    LAUNCH_KERNEL(init_owner_offsets_kernel, nbi, block_size, 0, stream,
        d_owner_offsets, h_num_contacts, num_particles + 1);
    LAUNCH_KERNEL(build_owner_offsets_from_keys_kernel, nb, block_size, 0, stream,
        d_sort_keys, d_owner_offsets, h_num_contacts);
    LAUNCH_KERNEL(fill_owner_offset_gaps_kernel, nbi, block_size, 0, stream,
        d_owner_offsets, num_particles, h_num_contacts);
}

// =============================================================================
// Kernel: Sorted Contact Force Computation (Owner-Centric)
// =============================================================================
__global__ void compute_contact_forces_from_pairs_kernel(
    // Contact pairs + permutation indices (sorted by owner i)
    const ContactPair* __restrict__ contacts,
    const int32_t* __restrict__ contact_indices,  // sorted perm: contacts[indices[pos]]
    const int32_t* __restrict__ num_contacts_ptr,
    const int32_t* __restrict__ owner_offsets,    // owner start offsets
    int32_t num_particles,                         // particle count
    // Particle positions
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
    const float* __restrict__ cohesive_strength,
    // Global ID for contact matching
    const int32_t* __restrict__ global_id,
    // Force/torque output (accumulated via atomicAdd)
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    // Contact history buffer
    ContactHistory* __restrict__ contact_history,
    const int32_t* __restrict__ contact_history_offset,
    int32_t* __restrict__ num_contacts_per_particle,
    int32_t max_contacts_per_particle,
    // Material parameters (Hertz-Mindlin)
    float E, float nu, float COR,
    float mu_s, float mu_r, float mu_t,
    float beta, float r_cut,
    bool use_visco, float visco_A,
    bool cohesion_enabled,
    float dt,
    // Periodic boundary config
    bool pbc_x, bool pbc_y, bool pbc_z,
    float pbc_box_len_x, float pbc_box_len_y, float pbc_box_len_z
) {
    // Thread-to-owner mapping (one thread per particle)
    int owner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner_idx >= num_particles) return;

    // Read contact range from offset array
    int32_t start = owner_offsets[owner_idx];
    int32_t end = owner_offsets[owner_idx + 1];
    if (start >= end) return;  // No contacts

    // Overflow protection
    int32_t actual_count = *num_contacts_ptr;
    if (start >= actual_count) return;
    int actual_end = min(end, actual_count);

    // Hoist owner properties (loaded once, reused across all contacts)
    float po_x = static_cast<float>(pos_x[owner_idx]);
    float po_y = static_cast<float>(pos_y[owner_idx]);
    float po_z = static_cast<float>(pos_z[owner_idx]);
    float vo_x = vel_x[owner_idx], vo_y = vel_y[owner_idx], vo_z = vel_z[owner_idx];
    float wo_x = omega_x[owner_idx], wo_y = omega_y[owner_idx], wo_z = omega_z[owner_idx];
    float R_o = radius[owner_idx];
    float m_o = mass[owner_idx];
    float I_o = moment_of_inertia[owner_idx];
    float gamma_o = cohesion_enabled ? cohesive_strength[owner_idx] : 0.0f;

    int32_t hist_offset = contact_history_offset[owner_idx];

    // Register accumulation for owner forces/torques (FMM-inspired)
    double owner_fx = 0.0, owner_fy = 0.0, owner_fz = 0.0;
    double owner_tx = 0.0, owner_ty = 0.0, owner_tz = 0.0;

    // Process all contacts for this owner
    for (int32_t c = start; c < actual_end; ++c) {
        const ContactPair& contact = contacts[contact_indices[c]];
        int32_t partner_idx = contact.j;
        if (partner_idx < 0) continue;

        // Load partner properties
        float pp_x = static_cast<float>(pos_x[partner_idx]);
        float pp_y = static_cast<float>(pos_y[partner_idx]);
        float pp_z = static_cast<float>(pos_z[partner_idx]);

        // Apply periodic image offset
        {
            int8_t img_x = contact.image_x;
            int8_t img_y = contact.image_y;
            int8_t img_z = contact.image_z;

            if (img_x != 0 && pbc_x && pbc_box_len_x > 0.0f) {
                pp_x += img_x * pbc_box_len_x;
            }
            if (img_y != 0 && pbc_y && pbc_box_len_y > 0.0f) {
                pp_y += img_y * pbc_box_len_y;
            }
            if (img_z != 0 && pbc_z && pbc_box_len_z > 0.0f) {
                pp_z += img_z * pbc_box_len_z;
            }
        }

        float vp_x = vel_x[partner_idx], vp_y = vel_y[partner_idx], vp_z = vel_z[partner_idx];
        float wp_x = omega_x[partner_idx], wp_y = omega_y[partner_idx], wp_z = omega_z[partner_idx];
        float R_p = radius[partner_idx];
        float m_p = mass[partner_idx];
        float I_p = moment_of_inertia[partner_idx];
        float gamma_p = cohesion_enabled ? cohesive_strength[partner_idx] : 0.0f;

        // Contact History Lookup
        float H_t_x = 0.0f, H_t_y = 0.0f, H_t_z = 0.0f;
        float H_r_x = 0.0f, H_r_y = 0.0f, H_r_z = 0.0f;
        float H_tw_x = 0.0f, H_tw_y = 0.0f, H_tw_z = 0.0f;

        int32_t num_hist = num_contacts_per_particle[owner_idx];
        int32_t slot = -1;
        int32_t gid_partner = global_id[partner_idx];

        for (int32_t k = 0; k < num_hist && k < max_contacts_per_particle; ++k) {
            const ContactHistory& hist = contact_history[hist_offset + k];
            if (hist.partner_global_id == gid_partner && hist.is_active) {
                H_t_x = hist.delta_t_x; H_t_y = hist.delta_t_y; H_t_z = hist.delta_t_z;
                H_r_x = hist.delta_r_x; H_r_y = hist.delta_r_y; H_r_z = hist.delta_r_z;
                H_tw_x = hist.delta_tw_x; H_tw_y = hist.delta_tw_y; H_tw_z = hist.delta_tw_z;
                slot = k;
                break;
            }
        }

        // Compute contact forces
        ContactForceResult result = compute_contact_forces(
            po_x, po_y, po_z, vo_x, vo_y, vo_z, wo_x, wo_y, wo_z,
            R_o, m_o, I_o, gamma_o,
            pp_x, pp_y, pp_z, vp_x, vp_y, vp_z, wp_x, wp_y, wp_z,
            R_p, m_p, I_p, gamma_p,
            H_t_x, H_t_y, H_t_z,
            H_r_x, H_r_y, H_r_z,
            H_tw_x, H_tw_y, H_tw_z,
            E, nu, COR, mu_s, mu_r, mu_t, beta, r_cut,
            use_visco, visco_A, cohesion_enabled, dt
        );

        // Update Contact History (non-atomic: single writer per owner)
        if (result.is_contact) {
            if (slot >= 0) {
                ContactHistory& hist = contact_history[hist_offset + slot];
                hist.delta_t_x = result.delta_t_x;
                hist.delta_t_y = result.delta_t_y;
                hist.delta_t_z = result.delta_t_z;
                hist.delta_r_x = result.delta_r_x;
                hist.delta_r_y = result.delta_r_y;
                hist.delta_r_z = result.delta_r_z;
                hist.delta_tw_x = result.delta_tw_x;
                hist.delta_tw_y = result.delta_tw_y;
                hist.delta_tw_z = result.delta_tw_z;
                hist.is_active = true;
            } else {
                int32_t new_slot = num_contacts_per_particle[owner_idx];
                if (new_slot < max_contacts_per_particle) {
                    num_contacts_per_particle[owner_idx] = new_slot + 1;
                    ContactHistory& hist = contact_history[hist_offset + new_slot];
                    hist.partner_global_id = gid_partner;
                    hist.partner_local_id = partner_idx;
                    hist.delta_t_x = result.delta_t_x;
                    hist.delta_t_y = result.delta_t_y;
                    hist.delta_t_z = result.delta_t_z;
                    hist.delta_r_x = result.delta_r_x;
                    hist.delta_r_y = result.delta_r_y;
                    hist.delta_r_z = result.delta_r_z;
                    hist.delta_tw_x = result.delta_tw_x;
                    hist.delta_tw_y = result.delta_tw_y;
                    hist.delta_tw_z = result.delta_tw_z;
                    hist.is_active = true;
                } else {
                    num_contacts_per_particle[owner_idx] = max_contacts_per_particle;
                }
            }
        } else if (slot >= 0) {
            contact_history[hist_offset + slot].is_active = false;
        }

        // Accumulate OWNER forces/torques in registers
        owner_fx += result.force_x;
        owner_fy += result.force_y;
        owner_fz += result.force_z;
        owner_tx += result.torque_x;
        owner_ty += result.torque_y;
        owner_tz += result.torque_z;

        // PARTNER reaction forces: atomicAdd per contact (Phase 2 target)
        atomicAdd(&force_x[partner_idx], -result.force_x);
        atomicAdd(&force_y[partner_idx], -result.force_y);
        atomicAdd(&force_z[partner_idx], -result.force_z);
        atomicAdd(&torque_x[partner_idx], result.torque_j_x);
        atomicAdd(&torque_y[partner_idx], result.torque_j_y);
        atomicAdd(&torque_z[partner_idx], result.torque_j_z);
    }

    // Flush accumulated owner forces (1 atomicAdd each)
    if (owner_fx != 0.0 || owner_fy != 0.0 || owner_fz != 0.0 ||
        owner_tx != 0.0 || owner_ty != 0.0 || owner_tz != 0.0) {
        atomicAdd(&force_x[owner_idx], owner_fx);
        atomicAdd(&force_y[owner_idx], owner_fy);
        atomicAdd(&force_z[owner_idx], owner_fz);
        atomicAdd(&torque_x[owner_idx], owner_tx);
        atomicAdd(&torque_y[owner_idx], owner_ty);
        atomicAdd(&torque_z[owner_idx], owner_tz);
    }
}

// =============================================================================
// SPH-Specific Translational Verlet Kernels
// =============================================================================
// Pure SPH uses translational-only Verlet integration (no rotation).
// These kernels skip angular velocity and quaternion updates.
// =============================================================================

// Translational-only Verlet step 1 for pure SPH.
// Applies acceleration magnitude limit (E5) and NaN/Inf detection (E6)
// mirroring the DEM fused kernel pattern in verlet_integration.cuh:489-498.
static __global__ void verlet_step1_translational_kernel(
    double* __restrict__ pos_x, double* __restrict__ pos_y, double* __restrict__ pos_z,
    double* __restrict__ vel_x, double* __restrict__ vel_y, double* __restrict__ vel_z,
    double* __restrict__ force_x, double* __restrict__ force_y, double* __restrict__ force_z,
    const float* __restrict__ mass,
    double dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double inv_m = 1.0 / static_cast<double>(mass[idx]);
    double ax = force_x[idx] * inv_m;
    double ay = force_y[idx] * inv_m;
    double az = force_z[idx] * inv_m;

    // E6: NaN/Inf detection. If any force component is non-finite, zero the
    // acceleration for this particle to prevent silent propagation through
    // velocity and position. isfinite returns false for both NaN and Inf.
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) {
        ax = 0.0; ay = 0.0; az = 0.0;
    }

    // E5: Acceleration magnitude limit (mirrors verlet_integration.cuh:489-498).
    // Config sets dev_config.acceleration.max_accel_magnitude = 1.0e5 m/s^2.
    // Without this clamp, a single bad force value can launch a particle across
    // the domain and trigger cascading NaN in the next neighbor search.
    if (dev_config.acceleration.enable_accel_limit) {
        double a_mag = sqrt(ax * ax + ay * ay + az * az);
        if (a_mag > dev_config.acceleration.max_accel_magnitude) {
            double scale = dev_config.acceleration.max_accel_magnitude / a_mag;
            ax *= scale; ay *= scale; az *= scale;
        }
    }

    double half_dt = dt * 0.5;
    double vx = vel_x[idx] + ax * half_dt;
    double vy = vel_y[idx] + ay * half_dt;
    double vz = vel_z[idx] + az * half_dt;

    vel_x[idx] = vx;
    vel_y[idx] = vy;
    vel_z[idx] = vz;

    pos_x[idx] += vx * dt;
    pos_y[idx] += vy * dt;
    pos_z[idx] += vz * dt;

    // Reset forces for next step
    force_x[idx] = 0.0;
    force_y[idx] = 0.0;
    force_z[idx] = 0.0;
}

// Translational-only Verlet step 2 for pure SPH.
// Applies NaN/Inf detection (E6). Acceleration limit is not re-applied here
// because step 2 only completes the velocity half-kick; the limit in step 1
// already bounds the per-step velocity change.
static __global__ void verlet_step2_translational_kernel(
    double* __restrict__ vel_x, double* __restrict__ vel_y, double* __restrict__ vel_z,
    const double* __restrict__ force_x, const double* __restrict__ force_y, const double* __restrict__ force_z,
    const float* __restrict__ mass,
    double dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double inv_m = 1.0 / static_cast<double>(mass[idx]);
    double half_dt = dt * 0.5;

    double dax = force_x[idx] * inv_m * half_dt;
    double day = force_y[idx] * inv_m * half_dt;
    double daz = force_z[idx] * inv_m * half_dt;

    // E6: NaN/Inf detection. Skip the velocity update for this particle if any
    // force component is non-finite. This prevents a single NaN force from
    // contaminating the velocity array (which would then propagate to position
    // in the next step 1).
    if (isfinite(dax) && isfinite(day) && isfinite(daz)) {
        vel_x[idx] += dax;
        vel_y[idx] += day;
        vel_z[idx] += daz;
    }
}

// Kernel to initialize SPH support extent
static __global__ void init_sph_support_extent_kernel(
    float* __restrict__ support_extent,
    const float* __restrict__ smoothing_length,
    float kernel_cutoff_factor,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    support_extent[idx] = smoothing_length[idx] * kernel_cutoff_factor;
}

// Helper function to initialize SPH support extent
void initialize_sph_support_extent(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    int32_t num_particles = state.particles.num_particles;
    if (num_particles <= 0) return;

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    init_sph_support_extent_kernel<<<num_blocks, block_size, 0, stream>>>(
        state.particles.support_extent,
        state.particles.smoothing_length,
        config.sph.kernel_cutoff_factor,
        num_particles
    );
}

// =============================================================================
// Simulation Runner
// =============================================================================

class SimulationRunner {
private:
    SimConfig config_;
    SimulationState state_;
    ContactListBuffer contacts_;
    VerletIntegrator integrator_;

    // Pure SPH mode flag - when true, skip DEM contact forces and use CFL timestep
    bool pure_sph_mode_;
    bool resumed_;

    // Optimization: Cache contact count (Pinned Memory)
    int32_t* h_pinned_num_contacts_;
    bool contacts_dirty_; // True if we need to re-read count from GPU
    int32_t* d_owner_offsets_;     // Device: owner start offsets
    ContactPair* d_contacts_sorted_; // Device: sorted contacts buffer
    uint64_t*  d_sort_keys_;       // Device: sort keys workspace
    int32_t*  d_sort_indices_;     // Device: sort permutation workspace
    uint64_t fallback_counter_; // Counter for fallback events

    // Timing
    cudaEvent_t start_event_, stop_event_, copy_done_event_;
    cudaStream_t compute_stream_;
    AdaptiveFMMDepthLifecycle adaptive_depth_io_{config_, state_, compute_stream_};
    int sph_inner_remaining_ = 0;
    SphMacroStep sph_macro_{};
    uint64_t cfl_evaluations_ = 0;
    uint64_t gravity_evaluations_ = 0;

    // Energy buffer for termination check
    float* d_kinetic_energy_per_particle_;
    // Ephemeral Grady--Kipp rate/cap state.  Authoritative restart state stays
    // in ParticleSoA; these values are recomputed from each closed state.
    SphDamageWorkspace damage_workspace_;
    SphBoundaryWorkspace boundary_workspace_;
    SPHTimestepResult last_sph_timestep_;

    // Total mass for specific KE calculation
    double total_mass_;
    int32_t num_total_particles_;
    int64_t last_scalar_step_;
    int64_t last_checkpoint_step_ = -1;
    int64_t checkpoint_cadence_step_ = 0;
    int last_sph_stage_ = -2;

    // Status
    bool initialized_;

    // Host output buffers (pre-allocated)
    std::vector<double> h_pos_x_, h_pos_y_, h_pos_z_;
    std::vector<double> h_vel_x_, h_vel_y_, h_vel_z_;
    std::vector<double> h_force_x_, h_force_y_, h_force_z_;
    std::vector<double> h_omega_x_, h_omega_y_, h_omega_z_;
    std::vector<float> h_radius_;

    // V3 SPH diagnostic output buffers (only used when config_.sph.enabled)
    std::vector<double> h_density_;
    std::vector<float> h_pressure_;
    std::vector<float> h_c_sound_;
    std::vector<float> h_smoothing_length_;
    std::vector<int32_t> h_flag_;  // V3 core requires flag array (zeros for pure SPH)

public:
    SimulationRunner()
        : pure_sph_mode_(false), resumed_(false),
          d_kinetic_energy_per_particle_(nullptr),
          total_mass_(0.0),
          num_total_particles_(0),
          last_scalar_step_(-1),
          initialized_(false),
          h_pinned_num_contacts_(nullptr),
          contacts_dirty_(true),
          d_owner_offsets_(nullptr),
          d_contacts_sorted_(nullptr),
          d_sort_keys_(nullptr),
          d_sort_indices_(nullptr),
          fallback_counter_(0) {}

    ~SimulationRunner() {
        shutdown();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    bool initialize(const std::string& config_path) {
        std::cout << "\n";
        std::cout << "============================================" << std::endl;
        std::cout << "       CudaDEM - SPH/DEM Simulation         " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "\n";

        // -------------------------------------------------------------------------
        // 1. Parse YAML configuration
        // -------------------------------------------------------------------------
        std::cout << "Loading configuration from: " << config_path << std::endl;

        try {
            config_ = parse_yaml(config_path);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to parse config - " << e.what() << std::endl;
            return false;
        }
        if (config_.physics != SimulationPhysics::SPH) {
            std::cerr << "ERROR: cudadem_sph is the single-GPU SPH runner; use "
                         "cudadem for simulation.physics: dem" << std::endl;
            return false;
        }

        print_config(config_);

        // -------------------------------------------------------------------------
        // 1.5 Detect pure SPH mode
        // -------------------------------------------------------------------------
        // Pure SPH mode: SPH enabled AND input_path specified AND no DEM particles
        // In pure SPH mode:
        //   - Use dedicated SPH input parser
        //   - Use CFL-limited timestep
        //   - Skip DEM contact forces
        //   - Use translational-only Verlet integration
        // -------------------------------------------------------------------------
        pure_sph_mode_ = config_.physics == SimulationPhysics::SPH;

        if (pure_sph_mode_) {
            std::cout << "\n*** PURE SPH MODE DETECTED ***" << std::endl;
            std::cout << "  - Using SPH-dedicated input parser" << std::endl;
            std::cout << "  - Using CFL-limited timestep" << std::endl;
            std::cout << "  - Skipping DEM contact forces" << std::endl;
            std::cout << "  - Using translational-only Verlet integration" << std::endl;
        }

        // Print SPH-specific configuration if enabled
        if (config_.physics == SimulationPhysics::SPH) {
            print_sph_config(config_);
            if (config_.sph.gravity_softening_len != 0.0f) {
                std::cerr << "WARNING: sph_parameters.gravity_softening_len does not "
                             "control particle self-gravity; particle pairs use their "
                             "mean smoothing length. The scalar applies only to sphere "
                             "walls.\n";
            }
        }

        // -------------------------------------------------------------------------
        // 2. Copy SimConfig to GPU constant memory
        // -------------------------------------------------------------------------
        std::cout << "\nCopying configuration to GPU constant memory..." << std::endl;
        CUDA_CHECK(cudaSetDevice(config_.gpu.device_id));
        copy_config_to_device(config_);

        // -------------------------------------------------------------------------
        // 3. Initialize simulation state
        // -------------------------------------------------------------------------
        std::cout << "\nInitializing simulation state..." << std::endl;

        // =================================================================
        // PURE SPH PATH: Use dedicated SPH input parser
        // =================================================================
        try {
            int32_t num_loaded = 0;
            if (config_.io.checkpoint_resume_from[0] != '\0') {
                if (config_.io.checkpoint_force_fresh)
                    throw std::runtime_error(
                        "io.checkpoint.force_fresh conflicts with io.checkpoint.resume_from");
                state_ = load_checkpoint(config_.io.checkpoint_resume_from, config_);
                resumed_ = true;
                checkpoint_cadence_step_ = state_.current_step;
                num_loaded = state_.particles.num_particles;
                std::cout << "Loaded " << num_loaded
                          << " SPH particles from explicit checkpoint" << std::endl;
            } else {
                num_loaded = load_sph_particles(state_.particles, config_);
                std::cout << "Loaded " << num_loaded
                          << " SPH particles from dedicated parser" << std::endl;
            }
            num_total_particles_ = num_loaded;
            canonicalize_particle_clump_order(state_.particles, nullptr, true);
            std::vector<int32_t> clump_ids(num_loaded);
            std::vector<uint8_t> lifecycle(num_loaded);
            if (num_loaded > 0) CUDA_CHECK(cudaMemcpy(clump_ids.data(), state_.particles.clump_id,
                                                     num_loaded*sizeof(int32_t), cudaMemcpyDeviceToHost));
            if (num_loaded > 0) CUDA_CHECK(cudaMemcpy(lifecycle.data(), state_.particles.lifecycle_state,
                                                     num_loaded*sizeof(uint8_t), cudaMemcpyDeviceToHost));
            int32_t num_live = 0;
            int32_t max_part_id = -1;
            while (num_live < num_loaded && clump_ids[num_live] >= 0 && lifecycle[num_live] == 0) {
                max_part_id = std::max(max_part_id, clump_ids[num_live]);
                ++num_live;
            }
            std::vector<uint8_t> seen(static_cast<size_t>(max_part_id + 1), 0);
            for (int32_t i = 0; i < num_live; ++i) seen[clump_ids[i]] = 1;
            for (int32_t part = 0; part <= max_part_id; ++part)
                if (!seen[part]) throw std::runtime_error("SPH Part IDs must be dense 0..K");
            for (int32_t i = num_live; i < num_loaded; ++i) {
                if (clump_ids[i] != -1 && lifecycle[i] == 0)
                    throw std::runtime_error("SPH active Part rows must precede inactive rows");
            }
            state_.particles.num_particles = num_live;
            state_.particles.num_local = num_live;
            state_.particles.num_ghost = 0;

            if (config_.sph.symmetry_enabled) {
                if (num_live != num_loaded || num_live > INT32_MAX/2)
                    throw std::runtime_error("SPH symmetry requires live-only input and room for mirror rows");
                resize_particle_soa_preserving(state_.particles, 2*num_live);
            }

            // SPH kernels consume the same immutable, per-GPU material table
            // as DEM.  It must be present before SPH properties are initialized.
            if (!resumed_) {
                const MaterialDatabase material_database = load_material_database(
                    config_.material_database.path, config_.dem_damping_model);
                state_.material_database = upload_material_database(
                    material_database, config_.gpu.device_id);
            }

            // Initialize SPH timestep module
            initialize_sph_timestep(num_live);
            CUDA_CHECK(cudaMalloc(&damage_workspace_.qdot,
                                  static_cast<size_t>(num_live) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&damage_workspace_.qcap,
                                  static_cast<size_t>(num_live) * sizeof(float)));

            // Set initial timestep from SPH CFL config
            if (!resumed_) state_.dt = get_initial_sph_timestep(config_);

        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to load SPH particles - " << e.what() << std::endl;
            return false;
        }

        // -------------------------------------------------------------------------
        // 4. Initialize subsystems
        // -------------------------------------------------------------------------

        // Create CUDA stream and events
        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
        CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_event_, cudaEventDisableTiming));

        // Wall manager must publish periodic bounds before CLL initializes:
        // CLL snapshots WallManager::get_pbc_config() into its device neighbor
        // parameters and otherwise misses periodic image cells on first use.
        WallManager::initialize(state_, config_);

        // Neighbor search
        if (config_.sph.symmetry_enabled) state_.particles.num_particles *= 2;
        NeighborSearch::initialize(config_, state_);
        if (config_.sph.symmetry_enabled) state_.particles.num_particles /= 2;

        // Fresh starts initialize persistent SPH state.  Checkpoints already
        // contain h, thermodynamics, stress, damage, rates, and closed-step
        // force and must not be overwritten here.
        if (!resumed_) {
            cudadem::initialize_sph_properties(state_, config_);
            cudadem::initialize_sph_strength(state_, config_);
        }
        // New checkpoints persist the compact Weibull population.  Legacy
        // checkpoints (including the accepted hydro relaxation state) do not,
        // so regenerate only when the four-field set was absent.
        if (!resumed_ || !state_.particles.input_has_damage_flaw_metadata)
            cudadem::initialize_sph_damage(state_, config_);

        initialize_sph_boundaries(state_, config_, boundary_workspace_, compute_stream_);
        reflect_sph_domain(state_, config_, compute_stream_);

        // Initialize SPH support extent
        initialize_sph_support_extent(state_, config_, compute_stream_);

        // Initialize global h_max for symmetric SPH neighbor queries (must happen
        // before the first neighbor search, otherwise dev_sph_h_max_global=0.0
        // yields half the correct query radius on step 0).
        if (config_.physics == SimulationPhysics::SPH) {
            cudadem::update_sph_h_max_global(state_, config_, compute_stream_);
        }

        // Acceleration manager
        AccelerationManager::initialize(config_);

        // Establish the epoch before force bootstrap because acceleration
        // providers may be explicitly time dependent.
        if (!resumed_) {
            state_.current_time = config_.time.start_time;
            state_.current_step = 0;
        }

        if (resumed_)
            adaptive_depth_io_.restore_adaptive_fmm_depth_checkpoint_sidecar(config_.io.checkpoint_resume_from);
        else
            adaptive_depth_io_.reset_adaptive_fmm_depth_csv_for_fresh_start();

        if (!resumed_ || config_.io.checkpoint_rebootstrap_forces ||
            sph_stage_transition_at(config_, state_.current_time)) {
            bootstrap_sph_forces();
        } else {
            // A normal checkpoint resume intentionally preserves its closed
            // force state, but P-alpha rates live in non-checkpointed device
            // workspace.  Recreate only that derived closure so the very
            // first CFL evaluation includes dt6/dt7 without touching particle
            // state or force arrays.
            cudadem::evaluate_sph_porosity_rates(state_, config_, compute_stream_);
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        }

        // Kinetic energy buffer for termination check
        size_t ke_size = state_.particles.num_particles * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_kinetic_energy_per_particle_, ke_size));



        // Compute total mass for specific KE calculation
        total_mass_ = compute_total_mass();

        initialized_ = true;

        std::cout << "\n========== Initialization Complete ==========" << std::endl;
        std::cout << "Mode:             Pure SPH" << std::endl;
        std::cout << "Particles:        " << num_total_particles_ << " total, "
                  << state_.particles.num_particles << " live" << std::endl;
        std::cout << "Timestep:         " << state_.dt << " s" << std::endl;
        std::cout << "CFL number:       " << config_.sph.cfl_number << std::endl;
        std::cout << "dt range:         [" << config_.sph.dt_min << ", " << config_.sph.dt_max << "] s" << std::endl;
        std::cout << "Start time:       " << state_.current_time << " s"
                  << (resumed_ ? " (checkpoint)" : "") << std::endl;
        std::cout << "Stop time:        " << config_.time.stop_time << " s" << std::endl;
        std::cout << "Termination:      " << (config_.termination == TerminationCondition::TIME ?
                                               "Time" : "Energy threshold") << std::endl;
        if (config_.termination == TerminationCondition::ENERGY_THRESHOLD) {
            std::cout << "Energy threshold: " << config_.energy_threshold << " J" << std::endl;
        }
        std::cout << "==============================================" << std::endl;

        // Pre-allocate host buffers
        int32_t num_particles = num_total_particles_;
        h_pos_x_.resize(num_particles); h_pos_y_.resize(num_particles); h_pos_z_.resize(num_particles);
        h_vel_x_.resize(num_particles); h_vel_y_.resize(num_particles); h_vel_z_.resize(num_particles);
        h_force_x_.resize(num_particles); h_force_y_.resize(num_particles); h_force_z_.resize(num_particles);
        h_omega_x_.resize(num_particles); h_omega_y_.resize(num_particles); h_omega_z_.resize(num_particles);
        h_radius_.resize(num_particles);

        // V3 SPH diagnostic buffers
        h_density_.resize(num_particles);
        h_pressure_.resize(num_particles);
        h_c_sound_.resize(num_particles);
        h_smoothing_length_.resize(num_particles);
        h_flag_.assign(num_particles, 0);  // V3 core flag — zeros for pure SPH, set once, never modified

        if (config_.io.output_initial_state) {
            save_output();
            record_sph_scalar_receipt(/*force=*/true);
        }

        return true;
    }

    // Compute the force at the imported closed state before the first Verlet
    // drift.  Fresh runs previously entered step 1 with zero force.  Derived
    // checkpoints can request the same rebuild after rigid assembly.
    void find_sph_neighbors_with_boundaries() {
        update_sph_mirrors(state_, config_, compute_stream_);
        const int n = state_.particles.num_particles;
        if (config_.sph.symmetry_enabled) state_.particles.num_particles = 2*n;
        NeighborSearch::find_sph_neighbors(state_, config_, compute_stream_);
        state_.particles.num_particles = n;
    }

    void bootstrap_sph_forces() {
        const int32_t n = state_.particles.num_particles;
        if (n <= 0) return;

        find_sph_neighbors_with_boundaries();
        cudadem::compute_sph_density(state_, config_, compute_stream_);
        CUDA_CHECK(cudaMemsetAsync(state_.particles.force_x, 0,
                                   static_cast<size_t>(n) * sizeof(double), compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(state_.particles.force_y, 0,
                                   static_cast<size_t>(n) * sizeof(double), compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(state_.particles.force_z, 0,
                                   static_cast<size_t>(n) * sizeof(double), compute_stream_));
        cudadem::update_sph_kernel_correction(state_, config_, compute_stream_);
        restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
        cudadem::compute_sph_pressure(state_, config_, compute_stream_);
        update_sph_mirrors(state_, config_, compute_stream_);
        // Populate instantaneous constitutive rates before the first adaptive
        // CFL query without advancing rho, e, S, D, or alpha.  This must also
        // run after a checkpoint force rebootstrap: checkpointed rate fields
        // may predate a rebuilt pressure/neighbor closure.
        cudadem::evaluate_sph_damage(state_, config_, damage_workspace_, compute_stream_);
        cudadem::compute_sph_density_rate(state_, config_, compute_stream_);
        cudadem::predict_sph_strength_and_work(state_, config_, 0.0f, compute_stream_);
        cudadem::compute_sph_energy_rate(state_, config_, compute_stream_);
        cudadem::evaluate_sph_porosity_rates(state_, config_, compute_stream_);
        update_sph_mirrors(state_, config_, compute_stream_);
        cudadem::compute_sph_forces(state_, config_, compute_stream_);
        AccelerationManager::compute_all_without_gravity(
            state_, config_, state_.current_time, compute_stream_);
        refresh_slow_gravity();
        cudadem::update_sph_history(state_, config_, 0.0f, compute_stream_);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    void refresh_slow_gravity() {
        if (!config_.acceleration.enable_self_gravity) return;
        AccelerationManager::compute_outer_gravity(state_, config_, state_.current_time, compute_stream_);
        ++gravity_evaluations_;
    }

    void slow_half_kick(double span) {
        if (!config_.acceleration.enable_self_gravity) return;
        const int n = state_.particles.num_particles;
        if (n > 0) apply_half_gravity_impulse_kernel<<<(n+255)/256,256,0,compute_stream_>>>(
            state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
            state_.particles.gravity_force_x, state_.particles.gravity_force_y,
            state_.particles.gravity_force_z, 0.5*span, n);
        restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
    }

    void record_macro() {
        if (config_.io.scalar_interval <= 0) return;
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = std::string(config_.io.output_directory) + "/sph_macros.csv";
        const bool header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        std::ofstream out(path, std::ios::app);
        if (header) out << "start_step,start_time,inner_count,inner_dt,end_time,cfl_dt,cfl_evaluations,gravity_evaluations\n";
        out << std::setprecision(17) << state_.current_step << ',' << state_.current_time << ','
            << sph_macro_.count << ',' << sph_macro_.dt << ',' << sph_macro_.end_time << ','
            << last_sph_timestep_.dt << ',' << cfl_evaluations_ << ',' << gravity_evaluations_ << '\n';
        if (!out) throw std::runtime_error("Cannot write SPH macro receipt");
    }

    void record_sph_scalar_receipt(bool force) {
        const int32_t interval = config_.io.scalar_interval;
        if (!force && (interval <= 0 || state_.current_step % interval != 0)) return;
        if (state_.current_step == last_scalar_step_) return;
        if (config_.io.output_directory[0] == '\0') return;

        const int32_t n = state_.particles.num_particles;
        if (n <= 0) return;
        std::vector<float> mass(n), pressure(n), h(n), damage(n);
        std::vector<double> density(n), x(n), y(n), z(n), vx(n), vy(n), vz(n),
                            fx(n), fy(n), fz(n), energy(n);
        std::vector<int32_t> material_id(n);
        CUDA_CHECK(cudaMemcpy(mass.data(), state_.particles.mass,
                              static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(density.data(), state_.particles.density,
                              static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pressure.data(), state_.particles.pressure,
                              static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data(), state_.particles.smoothing_length,
                              static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(damage.data(), state_.particles.damage,
                              static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(material_id.data(), state_.particles.material_id,
                              static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyDeviceToHost));
        const double* device_double_fields[] = {
            state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
            state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
            state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
            state_.particles.energy};
        double* host_double_fields[] = {
            x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
            fx.data(), fy.data(), fz.data(), energy.data()};
        for (int field = 0; field < 10; ++field)
            CUDA_CHECK(cudaMemcpy(host_double_fields[field], device_double_fields[field],
                                  static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));

        if (config_.acceleration.enable_self_gravity) {
            std::vector<double> gravity(n);
            const double* slow[] = {state_.particles.gravity_force_x, state_.particles.gravity_force_y, state_.particles.gravity_force_z};
            double* total[] = {fx.data(), fy.data(), fz.data()};
            for (int axis=0; axis<3; ++axis) {
                CUDA_CHECK(cudaMemcpy(gravity.data(), slow[axis], n*sizeof(double), cudaMemcpyDeviceToHost));
                for (int i=0; i<n; ++i) total[axis][i] += static_cast<double>(mass[i])*gravity[i];
            }
        }
        const MaterialDatabase materials = load_material_database(
            config_.material_database.path, config_.dem_damping_model);
        long double total_mass = 0.0L, com_x = 0.0L, com_y = 0.0L, com_z = 0.0L;
        long double mom_x = 0.0L, mom_y = 0.0L, mom_z = 0.0L;
        long double kinetic = 0.0L, internal = 0.0L;
        long double net_fx = 0.0L, net_fy = 0.0L, net_fz = 0.0L;
        double velocity_max = 0.0;
        double h_min = std::numeric_limits<double>::infinity(), h_max = 0.0;
        double rho_min = std::numeric_limits<double>::infinity(), rho_max = 0.0;
        double p_min = std::numeric_limits<double>::infinity();
        double p_max = -std::numeric_limits<double>::infinity();
        double d_min = std::numeric_limits<double>::infinity(), d_max = 0.0;
        int64_t density_floor_count = 0;
        int64_t damage_nonzero_count = 0;
        int64_t nonfinite_count = 0;
        for (int32_t i = 0; i < n; ++i) {
            const long double mi = mass[i];
            const double values[] = {
                static_cast<double>(mass[i]), x[i], y[i], z[i], vx[i], vy[i], vz[i],
                fx[i], fy[i], fz[i], energy[i], static_cast<double>(density[i]),
                static_cast<double>(pressure[i]), static_cast<double>(h[i]),
                static_cast<double>(damage[i])};
            bool particle_nonfinite = false;
            for (const double value : values)
                particle_nonfinite = particle_nonfinite || !std::isfinite(value);
            if (particle_nonfinite) {
                ++nonfinite_count;
                continue;
            }
            total_mass += mi;
            com_x += mi * x[i]; com_y += mi * y[i]; com_z += mi * z[i];
            mom_x += mi * vx[i]; mom_y += mi * vy[i]; mom_z += mi * vz[i];
            kinetic += 0.5L * mi * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
            internal += mi * energy[i];
            net_fx += fx[i]; net_fy += fy[i]; net_fz += fz[i];
            const double speed = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
            velocity_max = std::max(velocity_max, speed);
            h_min = std::min(h_min, static_cast<double>(h[i]));
            h_max = std::max(h_max, static_cast<double>(h[i]));
            rho_min = std::min(rho_min, static_cast<double>(density[i]));
            rho_max = std::max(rho_max, static_cast<double>(density[i]));
            p_min = std::min(p_min, static_cast<double>(pressure[i]));
            p_max = std::max(p_max, static_cast<double>(pressure[i]));
            d_min = std::min(d_min, static_cast<double>(damage[i]));
            d_max = std::max(d_max, static_cast<double>(damage[i]));
            if (damage[i] > 0.0f) ++damage_nonzero_count;
            const int32_t mid = material_id[i];
            if (mid < 0 || static_cast<size_t>(mid) >= materials.materials.size())
                throw std::runtime_error("invalid material id while writing SPH scalar receipt");
            const double floor = 0.8 * materials.materials[mid].sph_tillotson.reference_density;
            if (density[i] <= floor) ++density_floor_count;
        }
        if (!(total_mass > 0.0L) && nonfinite_count == 0)
            throw std::runtime_error("non-positive mass while writing SPH scalar receipt");
        if (total_mass > 0.0L) {
            com_x /= total_mass; com_y /= total_mass; com_z /= total_mass;
        } else {
            com_x = 0.0L; com_y = 0.0L; com_z = 0.0L;
        }
        const long double vrms = total_mass > 0.0L
            ? std::sqrt(2.0L * kinetic / total_mass) : 0.0L;
        const long double net_force = std::sqrt(net_fx * net_fx + net_fy * net_fy + net_fz * net_fz);

        const std::filesystem::path directory(config_.io.output_directory);
        std::filesystem::create_directories(directory);
        std::ofstream file(directory / "sph_scalars.jsonl", std::ios::app);
        if (!file) throw std::runtime_error("cannot append SPH scalar receipt");
        file << std::setprecision(17)
             << "{\"schema_version\":\"granit-sph-scalar-v1\",\"step\":"
             << state_.current_step << ",\"time_seconds\":" << state_.current_time
             << ",\"particles\":" << n << ",\"mass_kg\":" << static_cast<double>(total_mass)
             << ",\"com_m\":[" << static_cast<double>(com_x) << "," << static_cast<double>(com_y) << "," << static_cast<double>(com_z) << "]"
             << ",\"momentum_kg_m_s\":[" << static_cast<double>(mom_x) << "," << static_cast<double>(mom_y) << "," << static_cast<double>(mom_z) << "]"
             << ",\"velocity_rms_m_s\":" << static_cast<double>(vrms)
             << ",\"velocity_max_m_s\":" << velocity_max
             << ",\"kinetic_energy_j\":" << static_cast<double>(kinetic)
             << ",\"internal_energy_j\":" << static_cast<double>(internal)
             << ",\"h_min_m\":" << h_min << ",\"h_max_m\":" << h_max
             << ",\"density_min_kg_m3\":" << rho_min << ",\"density_max_kg_m3\":" << rho_max
             << ",\"pressure_min_pa\":" << p_min << ",\"pressure_max_pa\":" << p_max
             << ",\"damage_min\":" << d_min << ",\"damage_max\":" << d_max
             << ",\"damage_nonzero_count\":" << damage_nonzero_count
             << ",\"density_floor_count\":" << density_floor_count
             << ",\"nonfinite_particle_count\":" << nonfinite_count
             << ",\"net_force_norm_n\":" << static_cast<double>(net_force)
             << ",\"timestep_limiter\":{\"particle\":"
             << last_sph_timestep_.limiting_particle
             << ",\"constraint\":\""
             << sph_timestep_constraint_name(
                    last_sph_timestep_.limiting_diagnostic.constraint)
             << "\",\"dt_unclamped_s\":" << last_sph_timestep_.dt_unclamped
             << ",\"dt_cfl_s\":" << last_sph_timestep_.limiting_diagnostic.dt_cfl
             << ",\"dt_density_s\":" << last_sph_timestep_.limiting_diagnostic.dt_density
             << ",\"dt_energy_s\":" << last_sph_timestep_.limiting_diagnostic.dt_energy
             << ",\"dt_damage_s\":" << last_sph_timestep_.limiting_diagnostic.dt_damage
             << ",\"dt_porosity_alpha_s\":" << last_sph_timestep_.limiting_diagnostic.dt_porosity_alpha
             << ",\"dt_porosity_pressure_s\":" << last_sph_timestep_.limiting_diagnostic.dt_porosity_pressure
             << ",\"longitudinal_sound_speed_m_s\":"
             << last_sph_timestep_.limiting_diagnostic.longitudinal_sound_speed
             << ",\"particle_speed_m_s\":"
             << last_sph_timestep_.limiting_diagnostic.particle_speed << "}}\n";
        if (!file) throw std::runtime_error("failed writing SPH scalar receipt");
        last_scalar_step_ = state_.current_step;
    }

    // =========================================================================
    // Main Simulation Loop
    // =========================================================================

    void run() {
        if (!initialized_) {
            std::cerr << "ERROR: Simulation not initialized" << std::endl;
            return;
        }

        // Warm-up CUDA context
        cudaFree(0);

        std::cout << "\n========== Starting Simulation ===========" << std::endl;
        std::cout << "Running in PURE SPH mode" << std::endl;

        auto wall_start = std::chrono::high_resolution_clock::now();
        double next_output_time = next_sph_output_time(state_.current_time, config_.time.start_time, config_.time.output_step);
        int64_t steps_since_output = 0;

        int32_t num_particles = state_.particles.num_particles;
        int block_size = config_.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        // -------------------------------------------------------------------------
        // Main time stepping loop
        // -------------------------------------------------------------------------
        while (sph_inner_remaining_ > 0 || !should_terminate()) {
            const int stage_index = sph_stage_index(config_.sph.stages, config_.sph.stage_count, state_.current_time);
            if (stage_index != last_sph_stage_) {
                if (last_sph_stage_ != -2) bootstrap_sph_forces();
                last_sph_stage_ = stage_index;
                if (stage_index >= 0)
                    std::cout << "SPH stage: " << sph_stage_name(config_.sph.stages[stage_index].mode)
                              << " at t=" << state_.current_time << " s" << std::endl;
            }
            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePushA("SimulationStep");
            #endif


            // =====================================================================
            // Pure SPH Mode
            // =====================================================================
                // -----------------------------------------------------------------
                // Step 0: Compute SPH CFL timestep
                // -----------------------------------------------------------------
                if (sph_inner_remaining_ == 0) {
                    cudadem::compute_sph_timestep(state_, config_, last_sph_timestep_, compute_stream_);
                    ++cfl_evaluations_;
                    double event = next_output_time;
                    event = std::min(event, config_.time.stop_time);
                    if (config_.sph.stage_count > 0)
                        event = std::min(event, sph_current_stage(config_, state_.current_time).end_time);
                    sph_macro_ = plan_sph_macro(state_.current_time, last_sph_timestep_.dt,
                        config_.acceleration.gravity_update_rate, event,
                        config_.acceleration.gravity_macro_dt);
                    state_.dt = sph_macro_.dt;
                    sph_inner_remaining_ = sph_macro_.count;
                    record_macro();
                    slow_half_kick(sph_macro_.span);
                }
                damp_sph_relaxation(state_, config_, 0.5*state_.dt, compute_stream_);

                // -----------------------------------------------------------------
                // Step 1: Velocity Verlet - First half (TRANSLATIONAL ONLY)
                // -----------------------------------------------------------------
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("Verlet1_SPH");
                #endif
                LAUNCH_KERNEL(verlet_step1_translational_kernel, num_blocks, block_size, 0, compute_stream_,
                    state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
                    state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                    state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                    state_.particles.mass,
                    state_.dt, num_particles
                );
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // Verlet1_SPH
                #endif

                // -----------------------------------------------------------------
                // Step 2: Apply periodic boundaries
                // -----------------------------------------------------------------
                nvtxRangePushA("Boundary Condition");
                restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
                reflect_sph_domain(state_, config_, compute_stream_);
                WallManager::apply_periodic_boundaries(state_, config_, compute_stream_);
                nvtxRangePop();

                // -----------------------------------------------------------------
                // Step 3: Neighbor search (SPH support extent based)
                // -----------------------------------------------------------------
                nvtxRangePushA("Neighbor Search");
                if (config_.sph.symmetry_enabled || config_.neighbor.method == NeighborMethod::LBVH) {
                    find_sph_neighbors_with_boundaries();
                } else if (NeighborSearch::needs_rebuild(state_, config_, compute_stream_)) {
                    NeighborSearch::rebuild(state_, config_, compute_stream_);
                }
                nvtxRangePop();

                // -----------------------------------------------------------------
                // Step 4: SPH Physics Updates
                // -----------------------------------------------------------------
                nvtxRangePushA("SPH Physics");
#ifdef SPH_DEBUG
                {
                    // One-time diagnostic: pass neighbor-count buffer on first step
                    static bool sph_debug_first_step = true;
                    static int32_t* d_debug_neighbor_count = nullptr;

                    if (sph_debug_first_step) {
                        CUDA_CHECK(cudaMalloc(&d_debug_neighbor_count, num_particles * sizeof(int32_t)));
                        CUDA_CHECK(cudaMemsetAsync(d_debug_neighbor_count, 0, num_particles * sizeof(int32_t), compute_stream_));
                        sph_debug_first_step = false;
                    }

                    cudadem::compute_sph_density(state_, config_, compute_stream_, d_debug_neighbor_count);

                    if (d_debug_neighbor_count != nullptr) {
                        // Copy density to host
                        cudaMemcpyAsync(h_density_.data(), state_.particles.density, num_particles * sizeof(double), cudaMemcpyDeviceToHost, compute_stream_);

                        // Copy neighbor counts to host
                        std::vector<int32_t> h_neighbor_count(num_particles);
                        cudaMemcpyAsync(h_neighbor_count.data(), d_debug_neighbor_count, num_particles * sizeof(int32_t), cudaMemcpyDeviceToHost, compute_stream_);

                        cudaStreamSynchronize(compute_stream_);

                        // Compute stats
                        double dmin = 1e300, dmax = -1e300;
                        double dsum = 0.0;
                        int32_t nmin = 2147483647, nmax = 0, nzero = 0;
                        double nsum = 0.0;
                        for (int32_t k = 0; k < num_particles; k++) {
                            double d = h_density_[k];
                            if (d < dmin) dmin = d;
                            if (d > dmax) dmax = d;
                            dsum += d;

                            int32_t nc = h_neighbor_count[k];
                            if (nc < nmin) nmin = nc;
                            if (nc > nmax) nmax = nc;
                            nsum += nc;
                            if (nc == 0) nzero++;
                        }

                        std::cout << "\n========== SPH_DEBUG: First-Step Density Diagnostic ==========" << std::endl;
                        std::cout << "Step: " << state_.current_step << std::endl;
                        std::cout << "Smoothing length (h): " << config_.sph.smoothing_length << " m" << std::endl;
                        std::cout << "Kernel cutoff factor: " << config_.sph.kernel_cutoff_factor << std::endl;
                        std::cout << "Rest density (rho0): " << config_.sph.rest_density << " kg/m^3" << std::endl;
                        const char* method_names[] = {"???", "CLL", "LBVH", "DIRECT"};
                        int m = (int)config_.neighbor.method;
                        std::cout << "Neighbor method: " << (m >= 1 && m <= 3 ? method_names[m] : "???" )
                                  << " (" << m << ")" << std::endl;
                        std::cout << "---" << std::endl;
                        std::cout << "Density:   min=" << dmin << " max=" << dmax << " mean=" << (float)(dsum / num_particles) << std::endl;
                        std::cout << "Neighbors: min=" << nmin << " max=" << nmax << " mean=" << (float)(nsum / num_particles) << std::endl;
                        std::cout << "Neighbors == 0: " << nzero << "/" << num_particles << std::endl;
                        if (nzero > 0) {
                            // Print first 5 particles with 0 neighbors
                            int printed = 0;
                            for (int32_t k = 0; k < num_particles && printed < 5; k++) {
                                if (h_neighbor_count[k] == 0) {
                                    std::cout << "  Particle " << k << ": density=" << h_density_[k] << std::endl;
                                    printed++;
                                }
                            }
                        }
                        if (nzero > 0) {
                            std::cout << "*** DIAGNOSIS: " << nzero << " particles have ZERO neighbors! ***" << std::endl;
                            std::cout << "*** This confirms H1: Neighbor search is broken. ***" << std::endl;
                        } else if ((float)(dsum / num_particles) < 0.9f * config_.sph.rest_density) {
                            std::cout << "*** DIAGNOSIS: Density too low with neighbors present ***" << std::endl;
                            std::cout << "*** This suggests H2: Density kernel has bug. ***" << std::endl;
                        }
                        std::cout << "=================================================================\n" << std::endl;

                        // Free the debug buffer after first use
                        CUDA_CHECK(cudaFree(d_debug_neighbor_count));
                        d_debug_neighbor_count = nullptr;
                    }
                }
#else
                cudadem::compute_sph_density(state_, config_, compute_stream_);
#endif
                // Evaluate from the prior closed stress state.  Pressure and
                // porosity/strength below intentionally consume D_old; D
                // advances only after those constitutive updates and force
                // evaluation.
                cudadem::evaluate_sph_damage(state_, config_, damage_workspace_, compute_stream_);
                cudadem::update_sph_kernel_correction(state_, config_, compute_stream_);
                cudadem::compute_sph_density_rate(state_, config_, compute_stream_);
                cudadem::predict_sph_strength_and_work(state_, config_,
                                                       static_cast<float>(state_.dt), compute_stream_);
                cudadem::compute_sph_energy_rate(state_, config_, compute_stream_);
                // Rate closure belongs to the pre-integration state.  The
                // resulting alpha_dot is the only porosity state used by this
                // step's alpha commit.
                cudadem::evaluate_sph_porosity_rates(state_, config_, compute_stream_);
                cudadem::advance_sph_cold_energy(state_, config_, compute_stream_);
                cudadem::integrate_sph_energy(state_, config_, compute_stream_);
                cudadem::integrate_sph_density(state_, config_, static_cast<float>(state_.dt), compute_stream_);
                restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
                cudadem::compute_sph_pressure(state_, config_, compute_stream_);
                cudadem::integrate_sph_porosity(state_, config_, static_cast<float>(state_.dt), compute_stream_);
                restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
                cudadem::compute_sph_pressure(state_, config_, compute_stream_);
                cudadem::apply_sph_strength_radial_return(state_, config_, compute_stream_);
                // Do not carry the pre-commit rate into next step's CFL.  This
                // state-neutral refresh observes the final alpha/pressure/S.
                cudadem::evaluate_sph_porosity_rates(state_, config_, compute_stream_);
                restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
                update_sph_mirrors(state_, config_, compute_stream_);
                cudadem::compute_sph_forces(state_, config_, compute_stream_);
                cudadem::integrate_sph_damage(state_, config_, damage_workspace_,
                                              static_cast<float>(state_.dt), compute_stream_);
                nvtxRangePop();

                // -----------------------------------------------------------------
                // Update smoothing length for next step
                // -----------------------------------------------------------------
                cudadem::update_sph_smoothing_length(state_, config_, static_cast<float>(state_.dt), compute_stream_);
                cudadem::update_sph_h_max_global(state_, config_, compute_stream_);

                // Update support extent for neighbor rebuild detection
                initialize_sph_support_extent(state_, config_, compute_stream_);

                // -----------------------------------------------------------------
                // Step 5: External accelerations (gravity, etc.)
                // -----------------------------------------------------------------
                nvtxRangePushA("Acceleration");
                AccelerationManager::compute_all_without_gravity(state_, config_, state_.current_time + state_.dt, compute_stream_);
                nvtxRangePop();

                // -----------------------------------------------------------------
                // Step 6: Velocity Verlet - Second half (TRANSLATIONAL ONLY)
                // -----------------------------------------------------------------
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("Verlet2_SPH");
                #endif
                LAUNCH_KERNEL(verlet_step2_translational_kernel, num_blocks, block_size, 0, compute_stream_,
                    state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                    state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                    state_.particles.mass,
                    state_.dt, num_particles
                );
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // Verlet2_SPH
                #endif

                // -----------------------------------------------------------------
                // Step 7: Advance time
                // -----------------------------------------------------------------
                restore_sph_fixed(state_, config_, boundary_workspace_, compute_stream_);
                cudadem::update_sph_history(state_, config_, static_cast<float>(state_.dt), compute_stream_);
                damp_sph_relaxation(state_, config_, 0.5*state_.dt, compute_stream_);
                state_.current_time += state_.dt;
                state_.current_step++;
                steps_since_output++;
                --sph_inner_remaining_;
                if (sph_inner_remaining_ == 0) {
                    state_.current_time = sph_macro_.end_time;
                    refresh_slow_gravity();
                    slow_half_kick(sph_macro_.span);
                }

            // -----------------------------------------------------------------
            // Output and Visualization (common to both modes)
            // -----------------------------------------------------------------
            if (sph_inner_remaining_ == 0 && state_.current_time >= next_output_time) {
                nvtxRangePushA("Output Data");
                save_output();
                record_sph_scalar_receipt(/*force=*/true);
                nvtxRangePop();

                next_output_time = next_sph_output_time(state_.current_time, config_.time.start_time, config_.time.output_step);

                // Show progress
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - wall_start).count();
                std::cout << "Step " << state_.current_step
                          << " (t=" << state_.current_time << "s, dt=" << state_.dt << "s) - "
                          << elapsed << " ms elapsed" << std::endl;
            }

            // -----------------------------------------------------------------
            // Checkpoint (at configured intervals)
            // -----------------------------------------------------------------
            if (sph_inner_remaining_ == 0 && config_.io.checkpoint_enabled &&
                state_.current_step - checkpoint_cadence_step_ >= config_.io.checkpoint_interval) {
                save_checkpoint();
            }
            if (sph_inner_remaining_ == 0) record_sph_scalar_receipt(/*force=*/false);
            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePop(); // SimulationStep
            #endif
        }

        // Final synchronization and a complete terminal checkpoint, including
        // a zero-step load/save or a terminal step outside the output cadence.
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        if (config_.io.checkpoint_enabled && last_checkpoint_step_ != state_.current_step)
            save_checkpoint();
        record_sph_scalar_receipt(/*force=*/true);

        adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();

        // Compute final specific KE for output
        double final_specific_ke = (total_mass_ > 0) ? state_.kinetic_energy / total_mass_ : 0.0;

        std::cout << "\n========== Simulation Complete ===========" << std::endl;
        std::cout << "Mode:        Pure SPH" << std::endl;
        std::cout << "Final time:  " << state_.current_time << " s" << std::endl;
        std::cout << "Total steps: " << state_.current_step << std::endl;
        std::cout << "Final KE:    " << state_.kinetic_energy << " J" << std::endl;
        std::cout << "Final ratio = " << std::scientific << std::setprecision(4) << final_specific_ke << std::endl;
        std::cout << "============================================" << std::endl;
    }

    // =========================================================================
    // Termination Condition
    // =========================================================================

    bool should_terminate() {
        // Check time-based termination
        if (config_.termination == TerminationCondition::TIME) {
            return state_.current_time >= config_.time.stop_time;
        }

        if (config_.sph.stage_count > 0) return state_.current_time >= config_.time.stop_time;

        // Check energy-based termination
        if (config_.termination == TerminationCondition::ENERGY_THRESHOLD) {
            // Time limit still applies as fallback
            if (state_.current_time >= config_.time.stop_time) {
                return true;
            }

            // Check specific kinetic energy periodically (every 100 steps to avoid overhead)
            // Use specific KE (KE per unit mass) for proper dimensionless comparison
            if (state_.current_step % 100 == 0) {
                nvtxRangePushA("Energy Check (Thrust)");
                double ke = compute_total_kinetic_energy();
                nvtxRangePop();
                
                state_.kinetic_energy = ke;

                // Compute specific KE (J/kg = m²/s²)
                double specific_ke = (total_mass_ > 0) ? ke / total_mass_ : ke;

                if (specific_ke < config_.energy_threshold) {
                    std::cout << "Energy threshold reached: specific_KE = " << specific_ke
                              << " m²/s² < " << config_.energy_threshold << std::endl;
                    return true;
                }
            }
        }

        return false;
    }

    // =========================================================================
    // Energy Computation
    // =========================================================================

    double compute_total_kinetic_energy() {
        int32_t num_particles = state_.particles.num_particles;
        int block_size = config_.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        // Compute per-particle kinetic energy
        LAUNCH_KERNEL(compute_kinetic_energy_kernel, num_blocks, block_size, 0, compute_stream_,
            d_kinetic_energy_per_particle_,
            state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
            state_.particles.omega_x, state_.particles.omega_y, state_.particles.omega_z,
            state_.particles.mass, state_.particles.moment_of_inertia,
            num_particles
        );

        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // Sum using Thrust
        thrust::device_ptr<float> d_ptr(d_kinetic_energy_per_particle_);
        float total_ke = thrust::reduce(thrust::cuda::par.on(compute_stream_), d_ptr, d_ptr + num_particles, 0.0f, thrust::plus<float>());

        return static_cast<double>(total_ke);
    }

    // Compute total mass of all particles (for specific KE calculation)
    double compute_total_mass() {
        int32_t num_particles = state_.particles.num_particles;

        // Sum particle masses using Thrust
        thrust::device_ptr<float> d_mass_ptr(state_.particles.mass);
        float total_mass = thrust::reduce(thrust::cuda::par.on(compute_stream_), d_mass_ptr, d_mass_ptr + num_particles, 0.0f, thrust::plus<float>());

        return static_cast<double>(total_mass);
    }

    // Compute specific kinetic energy (KE per unit mass, dimensionless-like)
    double compute_specific_kinetic_energy() {
        double ke = compute_total_kinetic_energy();
        if (total_mass_ > 0) {
            return ke / total_mass_;  // J/kg = m²/s²
        }
        return ke;
    }

    // =========================================================================
    // Output
    // =========================================================================

    void prepare_inactive_sph_tail_for_serialization() {
        sanitize_inactive_sph_tail_for_serialization(
            state_.particles, state_.particles.num_particles,
            num_total_particles_, config_.sph.smoothing_length);
    }

    void save_output() {
        adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();
        // Generate output file path with step number
        std::ostringstream oss;
        oss << config_.io.output_directory << "/output_step_"
            << std::setfill('0') << std::setw(10) << state_.current_step << ".bin";
        std::string filepath = oss.str();

        // Create output directory if needed
        std::filesystem::path output_dir(config_.io.output_directory);
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        // Write all stored rows, including dead particles excluded from SPH.
        const int32_t live_particles = state_.particles.num_particles;
        prepare_inactive_sph_tail_for_serialization();
        state_.particles.num_particles = num_total_particles_;
        const bool wrote_output = write_unified_binary(filepath, state_.particles, state_, config_, false,
            nullptr, 0, 0);
        state_.particles.num_particles = live_particles;
        if (!wrote_output) {
            std::cerr << "WARNING: Failed to write unified binary output: " << filepath << std::endl;
            return;
        }
        std::cout << "Output saved: " << filepath << std::endl;
    }

    void save_checkpoint() {
        if (sph_inner_remaining_ != 0) throw std::logic_error("SPH checkpoint requires a closed macro");
        adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();
        const std::string path = generate_checkpoint_path(config_, state_.current_step);
        std::filesystem::remove(adaptive_fmm_depth_checkpoint_sidecar_path(path));
        const int32_t live_particles = state_.particles.num_particles;
        prepare_inactive_sph_tail_for_serialization();
        state_.particles.num_particles = num_total_particles_;
        cudadem::save_checkpoint(state_, config_);
        state_.particles.num_particles = live_particles;
        adaptive_depth_io_.save_adaptive_fmm_depth_checkpoint_sidecar(path);
        last_checkpoint_step_ = state_.current_step;
        checkpoint_cadence_step_ = state_.current_step;
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    void shutdown() {
        if (!initialized_) return;

        std::cout << "\nShutting down simulation..." << std::endl;

        shutdown_sph_boundaries(boundary_workspace_);

        // Free kinetic energy buffer
        if (d_kinetic_energy_per_particle_) {
            cudaFree(d_kinetic_energy_per_particle_);
            d_kinetic_energy_per_particle_ = nullptr;
        }
        if (damage_workspace_.qdot) {
            cudaFree(damage_workspace_.qdot);
            damage_workspace_.qdot = nullptr;
        }
        if (damage_workspace_.qcap) {
            cudaFree(damage_workspace_.qcap);
            damage_workspace_.qcap = nullptr;
        }

        if (h_pinned_num_contacts_) {
            cudaFreeHost(h_pinned_num_contacts_);
            h_pinned_num_contacts_ = nullptr;
        }

        if (d_owner_offsets_) { cudaFree(d_owner_offsets_); d_owner_offsets_ = nullptr; }
        if (d_contacts_sorted_) { cudaFree(d_contacts_sorted_); d_contacts_sorted_ = nullptr; }
        if (d_sort_keys_) { cudaFree(d_sort_keys_); d_sort_keys_ = nullptr; }
        if (d_sort_indices_) { cudaFree(d_sort_indices_); d_sort_indices_ = nullptr; }



        // Shutdown SPH timestep module
        shutdown_sph_timestep();
        shutdown_sph_strength_workspace();
        shutdown_sph_porosity();

        // Shutdown subsystems
        NeighborSearch::shutdown(state_);
        AccelerationManager::shutdown();

        // Free simulation state
        free_simulation_state(state_);

        // Destroy CUDA resources
        if (compute_stream_) {
            cudaStreamDestroy(compute_stream_);
            compute_stream_ = nullptr;
        }
        if (start_event_) {
            cudaEventDestroy(start_event_);
            start_event_ = nullptr;
        }
        if (stop_event_) {
            cudaEventDestroy(stop_event_);
            stop_event_ = nullptr;
        }
        if (copy_done_event_) {
            cudaEventDestroy(copy_done_event_);
            copy_done_event_ = nullptr;
        }

        initialized_ = false;
        std::cout << "Shutdown complete." << std::endl;
    }
};

} // namespace cudadem

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string config_path = "config.yaml";

    if (argc > 1) {
        config_path = argv[1];
    }

    // Check if config file exists
    std::ifstream config_check(config_path);
    if (!config_check.good()) {
        std::cerr << "ERROR: Configuration file not found: " << config_path << std::endl;
        std::cerr << "Usage: " << argv[0] << " [config.yaml]" << std::endl;
        return 1;
    }
    config_check.close();

    // Print CUDA device info
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA-capable devices found" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << std::endl;

    // Create and run simulation
    cudadem::SimulationRunner sim;

    if (!sim.initialize(config_path)) {
        std::cerr << "ERROR: Failed to initialize simulation" << std::endl;
        return 1;
    }

    try {
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Simulation failed - " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#include "src/core/constant/adaptive_fmm_depth_lifecycle.h"
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
#include "src/core/constant/binary_format.h"
#include "src/core/constant/final_output_tracker.h"
#include "src/core/constant/time_step_schedule.h"
#include "src/core/neighbor/neighbor_api.h"
#include "src/core/neighbor/contact_geometry.cuh"
#include "src/core/wall/wall.h"
#include "src/core/wall/bonded_rect_wall.h"
#include "src/core/wall/sphere_wall.h"
#include "src/core/acceleration/acceleration.h"
#include "src/core/acceleration/external_bodies.h"
#include "src/core/math/verlet_integration.cuh"
#include "src/core/math/verlet_respa_kernels.cuh"
#include "src/core/particle_model/particle.h"
#include "src/core/particle_model/clump.h"
#include "src/core/particle_model/clump_rigid_math.cuh"
#include "src/core/shear/shear_box.h"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <cstring>

// Thrust for reductions
#ifndef CUDADEM_CPU_MOCK
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>
#endif

namespace cudadem {

// Keep the integer suffix-min scan compatible with both CUB 2 and CUB 3.
struct OwnerOffsetMinimum {
    __host__ __device__ int32_t operator()(int32_t a, int32_t b) const {
        return b < a ? b : a;
    }
};

__device__ __forceinline__ void respa_rotate_vector(
    double qw, double qx, double qy, double qz,
    double vx, double vy, double vz,
    double& ox, double& oy, double& oz) {
    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

__global__ void compose_respa_output_force_kernel(
    const double* fast_x, const double* fast_y, const double* fast_z,
    const double* gravity_acc_x, const double* gravity_acc_y, const double* gravity_acc_z,
    const double* pos_x, const double* pos_y, const double* pos_z,
    const float* particle_mass, const int32_t* clump_id,
    const double* clump_com_x, const double* clump_com_y, const double* clump_com_z,
    const double* clump_qw, const double* clump_qx, const double* clump_qy, const double* clump_qz,
    const double* clump_force_x, const double* clump_force_y, const double* clump_force_z,
    const double* clump_torque_x, const double* clump_torque_y, const double* clump_torque_z,
    const double* clump_mass,
    const double* ixx, const double* ixy, const double* ixz,
    const double* iyy, const double* iyz, const double* izz,
    double* total_x, double* total_y, double* total_z, int32_t count) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double gx = 0.0, gy = 0.0, gz = 0.0;
    const int32_t id = clump_id[i];
    const double m = static_cast<double>(particle_mass[i]);
    if (id == 0) {
        gx = m * gravity_acc_x[i];
        gy = m * gravity_acc_y[i];
        gz = m * gravity_acc_z[i];
    } else if (id > 0) {
        const int32_t c = id - 1;
        const double qw = clump_qw[c], qx = clump_qx[c];
        const double qy = clump_qy[c], qz = clump_qz[c];
        double tbx, tby, tbz;
        respa_rotate_vector(qw, -qx, -qy, -qz,
                            clump_torque_x[c], clump_torque_y[c], clump_torque_z[c],
                            tbx, tby, tbz);
        ClumpRigidVec3 alpha_body{};
        if (clump_tensor_torque_alpha(
                ixx[c], ixy[c], ixz[c], iyy[c], iyz[c], izz[c],
                {tbx, tby, tbz}, alpha_body)) {
            double awx, awy, awz;
            respa_rotate_vector(qw, qx, qy, qz,
                                alpha_body.x, alpha_body.y, alpha_body.z,
                                awx, awy, awz);
            const double rx = pos_x[i] - clump_com_x[c];
            const double ry = pos_y[i] - clump_com_y[c];
            const double rz = pos_z[i] - clump_com_z[c];
            gx += m * (awy * rz - awz * ry);
            gy += m * (awz * rx - awx * rz);
            gz += m * (awx * ry - awy * rx);
        }
        const double share = m / clump_mass[c];
        gx += share * clump_force_x[c];
        gy += share * clump_force_y[c];
        gz += share * clump_force_z[c];
    }
    total_x[i] = fast_x[i] + gx;
    total_y[i] = fast_y[i] + gy;
    total_z[i] = fast_z[i] + gz;
}

// =============================================================================
// Sorted Contact Force Kernel (Owner-Centric, Register Accumulation)
// =============================================================================
// Contacts are sorted by owner (contact.i) so each thread processes ALL
// contacts for one owner particle. Owner forces/torques are accumulated in
// registers and flushed with a single atomicAdd per particle — inspired by
// the FMM P2P kernel's register accumulation optimization.
//
// Partner reaction forces still use per-contact atomicAdd (Phase 2 target).
// Contact history is updated non-atomically since only one thread touches
// each owner's history slots.
// =============================================================================

// =============================================================================
// Kernel: Extract sort keys and initialize permutation indices
// =============================================================================
// Builds uint64_t key = ((uint64_t)i << 32) | (uint32_t)j for each contact.
// Radix-sorting these keys groups contacts by owner deterministically.
// Also initializes indices[idx] = idx for the sort permutation.
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
// Deterministic gather-scatter: reads from src at permuted positions,
// writes to dst sequentially. Each thread handles one output position.
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
// Kernel: Build owner offset array from sorted keys (mark phase)
// =============================================================================
__global__ void build_owner_offsets_from_keys_kernel(
    const uint64_t* __restrict__ keys,
    int32_t* __restrict__ owner_offsets,
    int32_t num_contacts,
    int32_t num_owners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    int32_t owner = static_cast<int32_t>(keys[idx] >> 32);
    if (owner < 0 || owner >= num_owners) return;  // R4: bounds check (defensive)
    if (idx == 0 || owner != static_cast<int32_t>(keys[idx - 1] >> 32)) {
        owner_offsets[owner] = idx;
    }
}

// =============================================================================
// Kernel: Init offsets to sentinel value
// =============================================================================
__global__ void init_owner_offsets_kernel(
    int32_t* __restrict__ offsets,
    int32_t val,
    int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) offsets[i] = val;
}

// =============================================================================
// Kernel: Reverse-copy owner offsets for stream-ordered CUB inclusive-min scan
// =============================================================================
// CUB scans forward.  Reversing before and after the scan implements the
// required suffix minimum without a host-side temporary allocation or sync.
// =============================================================================
__global__ void reverse_copy_offsets_kernel(
    const int32_t* __restrict__ source,
    int32_t* __restrict__ destination,
    int32_t count)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = source[count - 1 - index];
}

// =============================================================================
// Host Helper: Sort contacts by owner using key-based radix sort
// =============================================================================
// Uses CUB DeviceRadixSort on uint64_t keys = (i<<32)|j + permutation indices.
// Its caller preallocates every output and temporary buffer, so steady-state
// contact sorting neither allocates nor synchronizes the compute stream.
//
// Force kernel uses gather: contact = contacts[indices[pos]] (one extra read)
// =============================================================================
static int32_t sort_contacts_by_owner(
    ContactPair* contacts,            // input: unsorted contacts
    int32_t h_num_contacts,           // already synchronized host mirror
    int32_t* d_owner_offsets,         // output: owner start offsets (size num_particles+1)
    uint64_t* d_sort_keys,            // workspace: sort keys (size max_contacts)
    uint64_t* d_sorted_sort_keys,     // sorted keys (size max_contacts)
    int32_t* d_sort_indices,          // workspace: sort permutation (size max_contacts)
    int32_t* d_sorted_sort_indices,   // sorted permutation (size max_contacts)
    int32_t* d_reversed_owner_offsets,// workspace: num_particles + 1
    void* d_cub_temp_storage,         // preallocated max(sort, scan) temporary storage
    size_t cub_temp_storage_bytes,
    int32_t num_particles,
    int32_t max_contacts,
    int block_size,
    cudaStream_t stream)
{
    h_num_contacts = min(h_num_contacts, max_contacts);

    if (h_num_contacts == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_owner_offsets, 0,
            (num_particles + 1) * sizeof(int32_t), stream));
        return 0;
    }

    // Step 1: Extract sort keys (i<<32)|j + init permutation indices
    int num_blocks_extract = (h_num_contacts + block_size - 1) / block_size;
    LAUNCH_KERNEL(extract_sort_keys_kernel, num_blocks_extract, block_size, 0, stream,
        contacts, d_sort_keys, d_sort_indices, h_num_contacts);

    // Step 2: Radix sort keys + reorder indices (O(N), deterministic).
    // CUB's explicit input/output buffers and cached temp allocation avoid the
    // implicit Thrust temporary allocation/synchronization on every step.
#ifndef CUDADEM_CPU_MOCK
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_cub_temp_storage, cub_temp_storage_bytes,
        d_sort_keys, d_sorted_sort_keys,
        d_sort_indices, d_sorted_sort_indices,
        h_num_contacts, 0, 64, stream));
#else
    std::sort(d_sort_indices, d_sort_indices + h_num_contacts,
        [d_sort_keys](int32_t left, int32_t right) {
            return d_sort_keys[left] < d_sort_keys[right];
        });
    for (int32_t index = 0; index < h_num_contacts; ++index) {
        const int32_t source_index = d_sort_indices[index];
        d_sorted_sort_keys[index] = d_sort_keys[source_index];
        d_sorted_sort_indices[index] = source_index;
    }
#endif

    // Step 3: Build owner offset array from sorted keys (mark + fill gaps)
    // Skip permute — force kernel gathers via d_sort_indices directly
    int num_blocks_init = (num_particles + 1 + block_size - 1) / block_size;
    LAUNCH_KERNEL(init_owner_offsets_kernel, num_blocks_init, block_size, 0, stream,
        d_owner_offsets, h_num_contacts, num_particles + 1);

    int num_blocks_mark = (h_num_contacts + block_size - 1) / block_size;
    LAUNCH_KERNEL(build_owner_offsets_from_keys_kernel, num_blocks_mark, block_size, 0, stream,
        d_sorted_sort_keys, d_owner_offsets, h_num_contacts, num_particles);

    // Reverse inclusive-min scan: offset[p] = min_{q>=p} offset[q].
    // Replaces the O(N^2) forward-scan fill_owner_offset_gaps_kernel.
    // In-place on the [0, num_particles] range (sentinel slot included).
    // Bit-identical to the old kernel because min is associative+commutative
    // and the sentinel M=num_contacts is the max value.
    const int32_t offset_count = num_particles + 1;
#ifndef CUDADEM_CPU_MOCK
    const int32_t num_blocks_reverse = (offset_count + block_size - 1) / block_size;
    LAUNCH_KERNEL(reverse_copy_offsets_kernel, num_blocks_reverse, block_size, 0, stream,
        d_owner_offsets, d_reversed_owner_offsets, offset_count);
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        d_cub_temp_storage, cub_temp_storage_bytes,
        d_reversed_owner_offsets, d_reversed_owner_offsets,
        OwnerOffsetMinimum{}, offset_count, stream));
    LAUNCH_KERNEL(reverse_copy_offsets_kernel, num_blocks_reverse, block_size, 0, stream,
        d_reversed_owner_offsets, d_owner_offsets, offset_count);
#else
    // CPU mock arrays are ordinary host allocations.  Preserve the exact
    // suffix-min semantics without pulling real CUB into the mock build.
    for (int32_t owner = num_particles - 1; owner >= 0; --owner) {
        d_owner_offsets[owner] = std::min(
            d_owner_offsets[owner], d_owner_offsets[owner + 1]);
    }
#endif

    return h_num_contacts;
}

// =============================================================================
// Kernel: Sorted Contact Force Computation (Owner-Centric)
// =============================================================================
// Each thread processes all contacts for one owner particle.
// Owner forces/torques accumulated in registers → 1 atomicAdd flush per owner.
// Partner reaction forces still use per-contact atomicAdd (Phase 2 target).
// =============================================================================
__global__ void compute_contact_forces_from_pairs_kernel(
    // Contact pairs + permutation indices (sorted by owner i)
    const ContactPair* __restrict__ contacts,
    const int32_t* __restrict__ contact_indices,  // sorted perm: contacts[indices[pos]]
    const int32_t* __restrict__ num_contacts_ptr,
    const int32_t* __restrict__ owner_offsets,    // owner start offsets [0..num_particles]
    int32_t num_particles,                         // number of free particles
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
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const DemInteractionProperties* __restrict__ interactions,
    uint32_t material_count,
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
    bool use_visco,
    float dt,
    int32_t max_contact_pairs,  // Hard cap: size of contacts[] array
    // Periodic boundary config
    bool pbc_x, bool pbc_y, bool pbc_z,
    float pbc_box_len_x, float pbc_box_len_y, float pbc_box_len_z,
    float pbc_half_len_x, float pbc_half_len_y, float pbc_half_len_z,
    bool shear_enabled, float shear_reduced_shift, float shear_velocity
) {
    // Thread-to-owner mapping (one thread per free particle)
    int owner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner_idx >= num_particles) return;

    // Read contact range for this owner from offset array
    int32_t start = owner_offsets[owner_idx];
    int32_t end = owner_offsets[owner_idx + 1];

    // Clamp to max_contact_pairs (overflow protection)
    int32_t actual_count = min(*num_contacts_ptr, max_contact_pairs);
    const int32_t material_o = material_id[owner_idx];
    const int32_t hist_offset = contact_history_offset[owner_idx];
    const int32_t previous_rows = min(num_contacts_per_particle[owner_idx],
                                      max_contacts_per_particle);
    const bool no_usable_pairs = start >= end || start >= actual_count ||
        material_o < 0 || static_cast<uint32_t>(material_o) >= material_count;
    // Begin this owner's contact epoch without discarding active spring
    // memory before lookup. Owners with no candidates must also retire their
    // histories: a separated pair can disappear from the neighbor list.
    for (int32_t k = 0; k < previous_rows; ++k) {
        ContactHistory& hist = contact_history[hist_offset + k];
        hist.seen_this_step = false;
        if (no_usable_pairs) hist.is_active = false;
    }
    if (no_usable_pairs) return;

    // =========================================================================
    // Hoist owner properties OUTSIDE the contact loop (loaded once)
    // =========================================================================
    float po_x = static_cast<float>(pos_x[owner_idx]);
    float po_y = static_cast<float>(pos_y[owner_idx]);
    float po_z = static_cast<float>(pos_z[owner_idx]);
    float vo_x = vel_x[owner_idx], vo_y = vel_y[owner_idx], vo_z = vel_z[owner_idx];
    float wo_x = omega_x[owner_idx], wo_y = omega_y[owner_idx], wo_z = omega_z[owner_idx];
    float R_o = radius[owner_idx];
    float m_o = mass[owner_idx];
    float I_o = moment_of_inertia[owner_idx];
    // =========================================================================
    // Register accumulation for owner forces/torques (FMM-inspired)
    // =========================================================================
    double owner_fx = 0.0, owner_fy = 0.0, owner_fz = 0.0;
    double owner_tx = 0.0, owner_ty = 0.0, owner_tz = 0.0;

    // =========================================================================
    // Process all contacts for this owner
    // =========================================================================
    int actual_end = min(end, actual_count);
    for (int32_t c = start; c < actual_end; ++c) {
        const ContactPair& contact = contacts[contact_indices[c]];

        // By construction, contact.i == owner_idx (contacts are sorted by i)
        int32_t partner_idx = contact.j;
        if (partner_idx < 0) continue;

        // Load partner particle properties (per-contact)
        float pp_x = static_cast<float>(pos_x[partner_idx]);
        float pp_y = static_cast<float>(pos_y[partner_idx]);
        float pp_z = static_cast<float>(pos_z[partner_idx]);

        apply_contact_image(contact, pp_x, pp_y, pp_z,
            pbc_box_len_x, pbc_box_len_y, pbc_box_len_z,
            pbc_x, pbc_y, pbc_z, shear_enabled, shear_reduced_shift);

        float vp_x = vel_x[partner_idx], vp_y = vel_y[partner_idx], vp_z = vel_z[partner_idx];
        if (shear_enabled) vp_y -= contact.image_x * shear_velocity;
        float wp_x = omega_x[partner_idx], wp_y = omega_y[partner_idx], wp_z = omega_z[partner_idx];
        float R_p = radius[partner_idx];
        float m_p = mass[partner_idx];
        float I_p = moment_of_inertia[partner_idx];
        const int32_t material_p = material_id[partner_idx];
        if (material_p < 0 || static_cast<uint32_t>(material_p) >= material_count) continue;
        const DemInteractionProperties& pair = interactions[
            static_cast<size_t>(material_o) * material_count + material_p];
        const DemMaterialProperties& dem_o = materials[material_o].dem;
        const DemMaterialProperties& dem_p = materials[material_p].dem;
        const float G_o = dem_o.youngs_modulus / (2.0f * (1.0f + dem_o.poisson_ratio));
        const float G_p = dem_p.youngs_modulus / (2.0f * (1.0f + dem_p.poisson_ratio));
        const float E_star = 1.0f /
            ((1.0f - dem_o.poisson_ratio * dem_o.poisson_ratio) / dem_o.youngs_modulus +
             (1.0f - dem_p.poisson_ratio * dem_p.poisson_ratio) / dem_p.youngs_modulus);
        const float G_star = 1.0f /
            ((2.0f - dem_o.poisson_ratio) / G_o +
             (2.0f - dem_p.poisson_ratio) / G_p);

        // =====================================================================
        // Contact History Lookup (single-writer: safe, no atomics needed)
        // =====================================================================
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

        // =====================================================================
        // Compute contact forces (identical physics to original kernel)
        // =====================================================================
        ContactForceResult result = compute_contact_forces(
            po_x, po_y, po_z, vo_x, vo_y, vo_z, wo_x, wo_y, wo_z,
            R_o, m_o, I_o,
            pp_x, pp_y, pp_z, vp_x, vp_y, vp_z, wp_x, wp_y, wp_z,
            R_p, m_p, I_p,
            H_t_x, H_t_y, H_t_z,
            H_r_x, H_r_y, H_r_z,
            H_tw_x, H_tw_y, H_tw_z,
            E_star, G_star, pair.restitution,
            pair.friction_static, pair.friction_rolling, pair.friction_twisting,
            pair.shape_parameter,
            pair.cohesion_enabled ? pair.cohesion_cutoff_factor : 1.0f,
            use_visco, pair.viscous_coefficient,
            pair.cohesion_enabled != 0, pair.cohesion_strength, dt
        );

        // =====================================================================
        // Update Contact History (NON-ATOMIC: single writer per owner)
        // =====================================================================
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
                hist.seen_this_step = true;
            } else {
                // Non-atomic slot allocation (safe: single writer per owner)
                int32_t new_slot = num_contacts_per_particle[owner_idx];
                if (new_slot < max_contacts_per_particle) {
                    num_contacts_per_particle[owner_idx] = new_slot + 1;
                    ContactHistory& hist = contact_history[hist_offset + new_slot];
                    hist.partner_global_id = gid_partner;
                    hist.partner_local_id = partner_idx;
                    hist.seen_this_step = true;
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

        // =====================================================================
        // Accumulate OWNER forces/torques in registers
        // =====================================================================
        owner_fx += result.force_x;
        owner_fy += result.force_y;
        owner_fz += result.force_z;
        owner_tx += result.torque_x;
        owner_ty += result.torque_y;
        owner_tz += result.torque_z;

        // =====================================================================
        // PARTNER reaction forces: atomicAdd per contact (Phase 2 target)
        // =====================================================================
        atomicAdd(&force_x[partner_idx], -result.force_x);
        atomicAdd(&force_y[partner_idx], -result.force_y);
        atomicAdd(&force_z[partner_idx], -result.force_z);
        atomicAdd(&torque_x[partner_idx], result.torque_j_x);
        atomicAdd(&torque_y[partner_idx], result.torque_j_y);
        atomicAdd(&torque_z[partner_idx], result.torque_j_z);
    }

    // =========================================================================
    // Retire prior contacts absent from this evaluation. New and retained
    // physical contacts were marked seen by the single owner thread above.
    // This touches particle histories only; wall histories have their own
    // lifecycle and storage.
    const int32_t current_rows = min(num_contacts_per_particle[owner_idx],
                                     max_contacts_per_particle);
    for (int32_t k = 0; k < current_rows; ++k) {
        ContactHistory& hist = contact_history[hist_offset + k];
        if (!hist.seen_this_step) hist.is_active = false;
    }

    // Flush accumulated owner forces (1 atomicAdd each, not per-contact)
    // =========================================================================
    // Use if-guard to skip zero accumulations (common for particles with
    // only non-contact overlaps where is_contact == false)
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
// Simulation Runner
// =============================================================================

class SimulationRunner {
private:
    SimConfig config_;
    SimulationState state_;
    ContactListBuffer contacts_;
    VerletIntegrator integrator_;
    
    // Optimization: Cache contact count (Pinned Memory)
    int32_t* h_pinned_num_contacts_;
    bool contacts_dirty_; // True if we need to re-read count from GPU
    uint64_t fallback_counter_; // Counter for fallback events

    // Timing
    cudaEvent_t start_event_, stop_event_, copy_done_event_;
    cudaStream_t compute_stream_;
    AdaptiveFMMDepthLifecycle adaptive_depth_io_{config_, state_, compute_stream_};

    // Energy buffer for termination check
    float* d_kinetic_energy_per_particle_;

    // Total mass for specific KE calculation
    double total_mass_;

    // Status
    bool initialized_;

    // Clump (multi-sphere rigid body) state
    ClumpStateSoA       clump_state_;
    ClumpSegmentOffsets clump_segments_;
    ClumpStagingBuffer  clump_staging_;
    int32_t             num_free_particles_; // Ordinary live particles (clump_id == 0)
    int32_t             num_total_particles_; // Includes checkpointed dead particles
    int32_t*            d_owner_offsets_;     // Device: owner start offsets (size: num_live+1)
    uint64_t*           d_sort_keys_;          // Device: sort keys workspace (size: max_contacts)
    uint64_t*           d_sorted_sort_keys_;   // Device: sorted sort keys (size: max_contacts)
    int32_t*            d_sort_indices_;       // Device: sort permutation workspace (size: max_contacts)
    int32_t*            d_sorted_sort_indices_;// Device: sorted permutation (size: max_contacts)
    int32_t*            d_reversed_owner_offsets_; // Device: reverse scan workspace (size num_live + 1)
    void*               d_contact_sort_temp_;  // CUB sort/scan temporary storage
    size_t              contact_sort_temp_bytes_;
    int32_t             contact_sort_capacity_;

    // Loop timing/output state (promoted from run() locals for run_dem_step access)
    std::chrono::high_resolution_clock::time_point wall_start_;
    double next_output_time_;
    int32_t next_output_schedule_index_;
    int64_t steps_since_output_;
    FinalOutputTracker final_output_tracker_;
    int64_t timing_start_step_;
    double timing_start_time_;
    bool timing_window_started_, timing_window_finished_, timing_window_force_segment_active_;
    int64_t timing_window_start_step_, timing_window_end_step_;
    std::chrono::high_resolution_clock::time_point timing_window_start_;
    double timing_window_wall_seconds_;
    uint64_t timing_window_external_gravity_start_, timing_window_external_gravity_end_;
    uint64_t self_gravity_evaluations_;

    // Host output buffers (pre-allocated)
    std::vector<double> h_pos_x_, h_pos_y_, h_pos_z_;
    std::vector<double> h_vel_x_, h_vel_y_, h_vel_z_;
    std::vector<double> h_force_x_, h_force_y_, h_force_z_;
    std::vector<double> h_omega_x_, h_omega_y_, h_omega_z_;
    std::vector<float> h_radius_;
    std::vector<int32_t> h_clump_id_;  // For flag output
    double* d_output_force_x_;
    double* d_output_force_y_;
    double* d_output_force_z_;

    static uint64_t adaptive_yorp_config_fingerprint(const SimConfig& config) {
        // Bind exactly the runtime knobs that change adaptive decisions.
        uint64_t h = UINT64_C(1469598103934665603);
        const auto mix = [&h](const void* data, size_t bytes) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
        };
        const YorpConfig& y = config.acceleration.yorp;
        mix(&y.enabled, sizeof(y.enabled)); mix(&y.adaptive, sizeof(y.adaptive));
        mix(&y.mode, sizeof(y.mode));
        mix(&y.start_time, sizeof(y.start_time)); mix(&y.stop_time, sizeof(y.stop_time));
        mix(&y.epsilon, sizeof(y.epsilon)); mix(&y.check_interval_seconds, sizeof(y.check_interval_seconds));
        mix(&y.pause_specific_energy_j_per_kg, sizeof(y.pause_specific_energy_j_per_kg));
        mix(&y.resume_specific_energy_j_per_kg, sizeof(y.resume_specific_energy_j_per_kg));
        mix(&y.pause_confirmation_checks, sizeof(y.pause_confirmation_checks));
        mix(&y.resume_confirmation_checks, sizeof(y.resume_confirmation_checks));
        mix(&y.axis_ratio_drop_fraction, sizeof(y.axis_ratio_drop_fraction));
        mix(&y.epsilon_reduction_fraction_per_cycle, sizeof(y.epsilon_reduction_fraction_per_cycle));
        mix(&y.minimum_epsilon_fraction, sizeof(y.minimum_epsilon_fraction));
        mix(&y.max_omega_rad_s, sizeof(y.max_omega_rad_s));
        mix(&config.physics, sizeof(config.physics));
        mix(&config.termination, sizeof(config.termination));
        mix(&config.time.start_time, sizeof(config.time.start_time));
        mix(&config.time.stop_time, sizeof(config.time.stop_time));
        mix(&config.time.output_step, sizeof(config.time.output_step));
        mix(&config.rotating_frame.enabled, sizeof(config.rotating_frame.enabled));
        mix(&config.rotating_frame.omega, sizeof(config.rotating_frame.omega));
        mix(&config.acceleration.enable_inertial, sizeof(config.acceleration.enable_inertial));
        mix(&config.acceleration.inertial_constant_omega,
            sizeof(config.acceleration.inertial_constant_omega));
        return h;
    }

    static const char* adaptive_state_name(int32_t value) {
        switch (static_cast<AdaptiveYorpState>(value)) {
        case AdaptiveYorpState::RELAXING: return "relaxing";
        case AdaptiveYorpState::ACCELERATING: return "accelerating";
        case AdaptiveYorpState::PROVISIONAL_PAUSE: return "provisional_pause";
        case AdaptiveYorpState::CEILING_RELAX: return "ceiling_relax";
        case AdaptiveYorpState::COMPLETE: return "complete";
        case AdaptiveYorpState::FATAL: return "fatal";
        } return "unknown";
    }

    static const char* adaptive_transition_name(int32_t value) {
        switch (static_cast<AdaptiveYorpTransition>(value)) {
        case AdaptiveYorpTransition::NONE: return "none";
        case AdaptiveYorpTransition::INITIAL_BASELINE: return "initial_baseline";
        case AdaptiveYorpTransition::PROVISIONAL_PAUSE: return "provisional_pause";
        case AdaptiveYorpTransition::PAUSE_CONFIRMED: return "pause_confirmed";
        case AdaptiveYorpTransition::PROVISIONAL_CLEARED: return "provisional_cleared";
        case AdaptiveYorpTransition::RESUMED: return "resumed";
        case AdaptiveYorpTransition::CEILING_REACHED: return "ceiling_reached";
        case AdaptiveYorpTransition::COMPLETE: return "complete";
        case AdaptiveYorpTransition::INVALID_METRICS: return "invalid_metrics";
        } return "unknown";
    }

    std::string adaptive_yorp_csv_path() const {
        return std::string(config_.io.output_directory) + "/adaptive_yorp.csv";
    }

    std::string adaptive_yorp_checkpoint_path(const std::string& checkpoint_path) const {
        return checkpoint_path + ".adaptive_yorp";
    }

    static const char* adaptive_yorp_csv_header() {
        return "time,state,fitted_omega_x_rad_s,fitted_omega_y_rad_s,fitted_omega_z_rad_s,"
               "specific_energy_j_per_kg,current_baseline_axis_ratio,effective_epsilon_rad_s2,"
               "confirmed_cycles,transition_reason,transition_time";
    }

    void reset_adaptive_yorp_csv_for_fresh_start() {
        if (!config_.acceleration.yorp.adaptive) return;
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = adaptive_yorp_csv_path();
        if (!std::filesystem::exists(path)) return;
        std::ofstream file(path, std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot reset stale adaptive YORP diagnostic: " + path);
    }

    void rewind_adaptive_yorp_csv_for_resume(double checkpoint_metric_time) {
        const std::string path = adaptive_yorp_csv_path();
        if (!std::filesystem::exists(path)) return;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot read adaptive YORP diagnostic for resume: " + path);
        std::string header;
        if (!std::getline(input, header) || header != adaptive_yorp_csv_header()) {
            throw std::runtime_error("Adaptive YORP diagnostic header is invalid on resume: " + path);
        }
        std::vector<std::string> retained;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const size_t comma = line.find(',');
            if (comma == std::string::npos)
                throw std::runtime_error("Adaptive YORP diagnostic row is invalid on resume: " + path);
            size_t parsed = 0;
            double metric_time = 0.0;
            try { metric_time = std::stod(line.substr(0, comma), &parsed); }
            catch (...) {
                throw std::runtime_error("Adaptive YORP diagnostic time is invalid on resume: " + path);
            }
            if (parsed != comma || !std::isfinite(metric_time))
                throw std::runtime_error("Adaptive YORP diagnostic time is invalid on resume: " + path);
            const double tolerance = 1e-12 * std::max(1.0, std::fabs(checkpoint_metric_time));
            if (metric_time <= checkpoint_metric_time + tolerance) retained.push_back(line);
        }
        input.close();
        const std::string temporary = path + ".resume_tmp";
        {
            std::ofstream output(temporary, std::ios::trunc);
            if (!output) throw std::runtime_error("Cannot rewrite adaptive YORP diagnostic on resume: " + temporary);
            output << header << '\n';
            for (const std::string& row : retained) output << row << '\n';
            if (!output) throw std::runtime_error("Failed rewriting adaptive YORP diagnostic on resume: " + temporary);
        }
        std::filesystem::rename(temporary, path);
    }

    bool append_adaptive_yorp_diagnostic() {
        if (!config_.acceleration.yorp.adaptive) return false;
        AdaptiveYorpDiagnostic d;
        if (!AccelerationManager::consume_adaptive_yorp_diagnostic(&d, compute_stream_)) return false;
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = adaptive_yorp_csv_path();
        const bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        std::ofstream file(path, std::ios::app);
        if (!file) throw std::runtime_error("Cannot append adaptive YORP diagnostic: " + path);
        if (write_header) {
            file << adaptive_yorp_csv_header() << '\n';
        }
        // `d.time` is the completed integration boundary at which the GPU
        // reduction ran.  It can precede an arbitrary output boundary, so
        // recording state_.current_time here would mislabel the measurement.
        file << std::setprecision(17) << d.time << ',' << adaptive_state_name(d.state) << ','
             << d.omega_x << ',' << d.omega_y << ',' << d.omega_z << ','
             << d.specific_energy_j_per_kg << ',' << d.current_baseline_axis_ratio << ','
             << d.effective_epsilon << ',' << d.confirmed_cycles << ','
             << adaptive_transition_name(d.transition_reason) << ',' << d.transition_time << '\n';
        file.flush();
        if (!file) throw std::runtime_error("Failed writing adaptive YORP diagnostic: " + path);
        // Let the output boundary persist the matching particle state before
        // the runner turns this device-side failure into a nonzero exit.
        return d.fatal || d.state == static_cast<int32_t>(AdaptiveYorpState::FATAL);
    }

    void save_adaptive_yorp_checkpoint_sidecar(const std::string& checkpoint_path) {
        if (!config_.acceleration.yorp.adaptive) return;
        AdaptiveYorpCheckpoint checkpoint;
        if (!AccelerationManager::get_adaptive_yorp_checkpoint(&checkpoint, compute_stream_))
            throw std::runtime_error("Adaptive YORP checkpoint state is unavailable");
        struct Header { char magic[8]; uint32_t version; int64_t step; double time; uint64_t config_hash; };
        Header header{{'A','Y','O','R','P','1','\0','\0'}, 1, state_.current_step,
                      state_.current_time, adaptive_yorp_config_fingerprint(config_)};
        const std::string path = adaptive_yorp_checkpoint_path(checkpoint_path);
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot write adaptive YORP checkpoint sidecar: " + path);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.write(reinterpret_cast<const char*>(&checkpoint), sizeof(checkpoint));
        if (!file) throw std::runtime_error("Failed writing adaptive YORP checkpoint sidecar: " + path);
    }

    void restore_adaptive_yorp_checkpoint_sidecar(const std::string& checkpoint_path) {
        if (!config_.acceleration.yorp.adaptive) return;
        struct Header { char magic[8]; uint32_t version; int64_t step; double time; uint64_t config_hash; };
        Header header{}; AdaptiveYorpCheckpoint checkpoint{};
        const std::string path = adaptive_yorp_checkpoint_path(checkpoint_path);
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Adaptive YORP resume requires checkpoint sidecar: " + path);
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        file.read(reinterpret_cast<char*>(&checkpoint), sizeof(checkpoint));
        const char expected_magic[8] = {'A','Y','O','R','P','1','\0','\0'};
        if (!file || std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
            header.version != 1 || checkpoint.version != 1 || header.step != state_.current_step ||
            header.time != state_.current_time || header.config_hash != adaptive_yorp_config_fingerprint(config_)) {
            throw std::runtime_error("Adaptive YORP checkpoint sidecar is invalid, orphaned, or mismatched: " + path);
        }
        if (!AccelerationManager::restore_adaptive_yorp_checkpoint(checkpoint, compute_stream_))
            throw std::runtime_error("Unsupported adaptive YORP checkpoint sidecar: " + path);
        rewind_adaptive_yorp_csv_for_resume(checkpoint.diagnostic.time);
    }


public:
    SimulationRunner()
        : d_kinetic_energy_per_particle_(nullptr),
          total_mass_(0.0),
          initialized_(false),
          h_pinned_num_contacts_(nullptr),
          contacts_dirty_(true),
          fallback_counter_(0),
          num_free_particles_(0),
          num_total_particles_(0),
          d_owner_offsets_(nullptr),
          d_sort_keys_(nullptr),
          d_sorted_sort_keys_(nullptr),
          d_sort_indices_(nullptr),
          d_sorted_sort_indices_(nullptr),
          d_reversed_owner_offsets_(nullptr),
          d_contact_sort_temp_(nullptr),
          contact_sort_temp_bytes_(0),
          contact_sort_capacity_(0),
          d_output_force_x_(nullptr),
          d_output_force_y_(nullptr),
          d_output_force_z_(nullptr), timing_start_step_(0),
          timing_start_time_(0.0) {}

    ~SimulationRunner() {
        shutdown();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    bool initialize(const std::string& config_path) {
        std::cout << "\n";
        std::cout << "============================================" << std::endl;
        std::cout << "       CudaDEM - Asteroid DEM Simulation    " << std::endl;
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
        if (config_.physics != SimulationPhysics::DEM) {
            std::cerr << "ERROR: cudadem is the single-GPU DEM runner; use "
                         "cudadem_sph for simulation.physics: sph" << std::endl;
            return false;
        }
        if (config_.shear_box.enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled) {
            std::cerr << "ERROR: adaptive shear gravity requires cudadem_multigpu's common "
                         "depth controller, including for one GPU" << std::endl;
            return false;
        }

        print_config(config_);

        // -------------------------------------------------------------------------
        // 2. Copy SimConfig to GPU constant memory
        // -------------------------------------------------------------------------
        std::cout << "\nCopying configuration to GPU constant memory..." << std::endl;
        try {
            // Constant memory is device-local.  Select the configured device
            // before the first symbol copy; otherwise CUDA uses the process
            // default (normally GPU 0) even when gpu.device_id requests another
            // device.  initialize_system repeats this selection defensively.
            CUDA_CHECK(cudaSetDevice(config_.gpu.device_id));
            copy_config_to_device(config_);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to initialize configured GPU - " << e.what() << std::endl;
            return false;
        }

        // -------------------------------------------------------------------------
        // 3. Initialize simulation state
        // -------------------------------------------------------------------------
        std::cout << "\nInitializing simulation state..." << std::endl;

        try {
            // Preserve the existing automatic-resume policy.  Unified-v3
            // checkpoint fingerprints are validated before any state is used.
            state_ = load_or_resume(config_);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to initialize system - " << e.what() << std::endl;
            return false;
        }

        // -------------------------------------------------------------------------
        // 3b. Canonicalize imported members/dead rows and initialize clumps.
        // -------------------------------------------------------------------------
        num_total_particles_ = state_.particles.num_particles;
        try {
            canonicalize_particle_clump_order(
                state_.particles, &state_.contact_history);
            const int32_t clump_capacity = std::max<int32_t>(config_.clump.num_instances, 1);
            clump_state_    = allocate_clump_state(clump_capacity);
            clump_segments_ = allocate_clump_segments(clump_capacity);
            initialize_clumps(config_, state_.material_database, state_.particles,
                              clump_state_, clump_segments_);
            num_free_particles_ = clump_segments_.normal_end;
            const int32_t num_members = clump_segments_.member_end - clump_segments_.member_begin;
            clump_staging_ = allocate_clump_staging(std::max<int32_t>(num_members, 1));
            if (clump_state_.num_clumps > 0)
                initialize_clump_cub_temp(clump_staging_, clump_state_, clump_segments_);

            // Every calculation sees only the live prefix. Unified output and
            // checkpoints temporarily restore the total count.
            state_.particles.num_particles = clump_segments_.dead_begin;
            state_.particles.num_local = state_.particles.num_particles;
            state_.particles.num_ghost = 0;
            if (config_.acceleration.yorp.adaptive && clump_state_.num_clumps > 0) {
                throw std::runtime_error(
                    "adaptive YORP supports only unclumped DEM particles");
            }
            std::cout << "Clumps:           " << clump_state_.num_clumps
                      << " (members: " << num_members << ", dead: "
                      << num_total_particles_ - clump_segments_.dead_begin << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to initialize clumps - " << e.what() << std::endl;
            free_clump_staging(clump_staging_);
            free_clump_segments(clump_segments_);
            free_clump_state(clump_state_);
            return false;
        }

        const bool resumed = state_.particles.source_record_kind ==
            static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT);
        if (config_.physics == SimulationPhysics::DEM) {
            const auto bounds = derive_particle_radius_bounds(
                state_.particles, state_.particles.num_particles);
            config_.particle.radius_min = bounds.first;
            config_.particle.radius_max = bounds.second;
            const MaterialDatabase materials = load_material_database(
                config_.material_database.path, config_.dem_damping_model);
            const double allowed_dt = calculate_critical_timestep(
                config_, state_.particles, materials, state_.particles.num_particles);
            if (!resumed) state_.dt = allowed_dt;
            else if (!(state_.dt > 0.0) || state_.dt > allowed_dt * (1.0 + 1.0e-12))
                throw std::runtime_error("checkpoint timestep exceeds current contact/orbital/configured limit");
            copy_config_to_device(config_);
        }

        // -------------------------------------------------------------------------
        // 4. Initialize subsystems
        // -------------------------------------------------------------------------

        // Create CUDA stream and events
        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
        CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_event_, cudaEventDisableTiming));
        sync_clump_members();
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // Establish the epoch before any time-dependent canonicalization.
        if (!resumed) {
            state_.current_time = config_.time.start_time;
            state_.current_step = 0;
        }

        // Wall manager (periodic boundaries + wall contacts, includes sphere wall init)
        WallManager::initialize(state_, config_);
        ShearBoxManager::initialize(state_, config_, compute_stream_);
        if (config_.shear_box.enabled) {
            ShearBoxManager::wrap(state_,config_,WallManager::get_pbc_config(),
                                  state_.current_time,state_.particles.num_local,compute_stream_);
        }

        // Neighbor search
        NeighborSearch::initialize(config_, state_);

        // Contact buffer
        int32_t max_contacts = estimate_max_contacts(state_.particles.num_particles);
        NeighborSearch::allocate_contact_buffer(contacts_, max_contacts);

        // Every live sphere can own a contact.  Restricting this to ordinary
        // particles silently drops clump-member/clump-member contacts.
        CUDA_CHECK(cudaMalloc(&d_owner_offsets_,
                              (state_.particles.num_particles + 1) * sizeof(int32_t)));
        // Sort workspace: CUB needs separate input/output key/value buffers.
        CUDA_CHECK(cudaMalloc(&d_sort_keys_, max_contacts * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sorted_sort_keys_, max_contacts * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sort_indices_, max_contacts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sorted_sort_indices_, max_contacts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_reversed_owner_offsets_,
                              (state_.particles.num_particles + 1) * sizeof(int32_t)));
#ifndef CUDADEM_CPU_MOCK
        size_t radix_sort_temp_bytes = 0;
        size_t owner_scan_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, radix_sort_temp_bytes,
            static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            max_contacts, 0, 64, compute_stream_));
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(
            nullptr, owner_scan_temp_bytes,
            static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            OwnerOffsetMinimum{}, state_.particles.num_particles + 1, compute_stream_));
        contact_sort_temp_bytes_ = std::max(radix_sort_temp_bytes, owner_scan_temp_bytes);
        CUDA_CHECK(cudaMalloc(&d_contact_sort_temp_, contact_sort_temp_bytes_));
#endif
        contact_sort_capacity_=max_contacts;

        // Acceleration manager
        AccelerationManager::initialize(config_);
        AccelerationManager::set_clump_runtime(&clump_state_, &clump_segments_);
        if (resumed && config_.acceleration.yorp.adaptive) {
            // load_or_resume selects the latest checkpoint for this runner;
            // fail closed if its adaptive state sidecar was not preserved.
            restore_adaptive_yorp_checkpoint_sidecar(loaded_checkpoint_path());
        } else if (config_.acceleration.yorp.adaptive) {
            reset_adaptive_yorp_csv_for_fresh_start();
        }

        // Integrator setup
        integrator_.set_timestep(state_.dt, true);
        integrator_.block_size = config_.gpu.block_size;
        state_.dt = integrator_.dt;

        if (config_.acceleration.gravity_macro_dt > 0.0) {
            const double ratio = config_.acceleration.gravity_macro_dt / state_.dt;
            const int64_t effective = std::max<int64_t>(
                1, static_cast<int64_t>(std::floor(ratio + 1.0e-12)));
            if (effective > std::numeric_limits<int>::max()) {
                throw std::runtime_error("gravity_macro_dt produces too many RESPA inner steps");
            }
            config_.acceleration.gravity_update_rate = static_cast<int>(effective);
            copy_config_to_device(config_);
            std::cout << "RESPA macro target: "
                      << config_.acceleration.gravity_macro_dt << " s -> K="
                      << effective << " (effective H=" << effective * state_.dt
                      << " s)" << std::endl;
        }
        // Validate against the final cadence, after gravity_macro_dt has
        // become the effective integer K used by the integrator.
        if (config_.io.timing_measure_steps > 0 && config_.acceleration.enable_self_gravity &&
            config_.acceleration.gravity_update_rate > 1 &&
            ((config_.io.timing_warmup_steps % config_.acceleration.gravity_update_rate) != 0 ||
             (config_.io.timing_measure_steps % config_.acceleration.gravity_update_rate) != 0)) {
            throw std::runtime_error("timing warmup_steps and measure_steps must align to complete RESPA outer ranges");
        }

        // The adaptive-depth fingerprint includes the effective RESPA
        // cadence.  Restore only after gravity_macro_dt has been converted to
        // the same integer update rate that was fingerprinted at save time.
        if (resumed && adaptive_fmm_depth_checkpoint_requested(config_)) {
            adaptive_depth_io_.restore_adaptive_fmm_depth_checkpoint_sidecar(
                loaded_checkpoint_path());
        } else if (adaptive_fmm_depth_checkpoint_requested(config_)) {
            adaptive_depth_io_.reset_adaptive_fmm_depth_csv_for_fresh_start();
        }

        // Damping is config-driven: Cundall local damping (LOCAL) is applied
        // inside the fused verlet kernels via dev_config. Viscous damping
        // (VISCOUS) is applied as a separate post-step kernel below.

        // Kinetic energy buffer for termination check
        size_t ke_size = state_.particles.num_particles * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_kinetic_energy_per_particle_, ke_size));

        // Allocate pinned memory for contact count
        CUDA_CHECK(cudaMallocHost(&h_pinned_num_contacts_, sizeof(int32_t)));
        *h_pinned_num_contacts_ = 0;

        // Compute total mass for specific KE calculation
        total_mass_ = compute_total_mass();

        // Global prescribed bodies are configured independently of contact
        // walls. Their loss ledger is deliberately restored only from the
        // matching checkpoint sidecar, so an interrupted restart cannot
        // silently double-count or forget Saturn absorptions.
        ExternalBodyManager::initialize(config_, num_free_particles_, total_mass_);
        if (resumed && config_.acceleration.external_bodies.enabled) {
            ExternalBodyManager::restore_checkpoint_sidecar(
                loaded_checkpoint_path(), state_, config_);
        }

        // -------------------------------------------------------------------------
        // 5. Set initial time
        // -------------------------------------------------------------------------
        if (config_.acceleration.yorp.adaptive) {
            // Establish the t=0 baseline in stream order.  No host readback:
            // the first torque update observes this queued device state.
            AccelerationManager::observe_adaptive_yorp(
                state_, config_, state_.current_time, compute_stream_);
        }

        initialized_ = true;

        std::cout << "\n========== Initialization Complete ==========" << std::endl;
        std::cout << "Particles:        " << num_total_particles_ << " total, "
                  << state_.particles.num_particles << " live" << std::endl;
        std::cout << "Timestep:         " << state_.dt << " s" << std::endl;
        std::cout << "Start time:       " << config_.time.start_time << " s" << std::endl;
        std::cout << "Stop time:        " << config_.time.stop_time << " s" << std::endl;
        const char* termination_name = config_.termination == TerminationCondition::TIME ? "Time" :
            config_.termination == TerminationCondition::ENERGY_THRESHOLD ? "Energy threshold" : "Adaptive YORP";
        std::cout << "Termination:      " << termination_name << std::endl;
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
        if ((config_.acceleration.enable_self_gravity &&
             config_.acceleration.gravity_update_rate > 1) || config_.shear_box.enabled) {
            const size_t bytes = static_cast<size_t>(num_particles) * sizeof(double);
            CUDA_CHECK(cudaMalloc(&d_output_force_x_, bytes));
            CUDA_CHECK(cudaMalloc(&d_output_force_y_, bytes));
            CUDA_CHECK(cudaMalloc(&d_output_force_z_, bytes));
        }

        return true;
    }

    // =========================================================================
    // Main Simulation Loop
    // =========================================================================

    int steps_until_time(double target_time) const {
        const double remaining = target_time - state_.current_time;
        if (!(remaining > 0.0) || !(state_.dt > 0.0)) return 1;
        const double scaled = remaining / state_.dt;
        return std::max(1, static_cast<int>(std::ceil(scaled - 1.0e-12)));
    }

    int synchronized_respa_inner_count(int nominal_k) const {
        int count = nominal_k;
        count = std::min(count, steps_until_time(config_.time.stop_time));

        if (config_.io.output_schedule_count > 0 || config_.time.output_step > 0.0) {
            count = std::min(count, steps_until_time(next_output_time_));
        }
        if (config_.io.checkpoint_enabled && config_.io.checkpoint_interval > 0) {
            const int64_t interval = config_.io.checkpoint_interval;
            const int64_t mod = state_.current_step % interval;
            const int64_t to_checkpoint = mod == 0 ? interval : interval - mod;
            count = std::min<int64_t>(count, to_checkpoint);
        }
        if (config_.termination == TerminationCondition::ENERGY_THRESHOLD) {
            const int64_t mod = state_.current_step % 100;
            const int64_t to_check = mod == 0 ? 100 : 100 - mod;
            count = std::min<int64_t>(count, to_check);
        }
        return std::max(count, 1);
    }

    void complete_synchronized_boundary() {
        // Metrics and persistence must observe a fully closed velocity state.
        AccelerationManager::observe_adaptive_yorp(
            state_, config_, state_.current_time, compute_stream_);
        if(config_.shear_box.enabled && state_.current_step % 100 == 0) {
            const ShearDiagnostics d=ShearBoxManager::compute_diagnostics(
                state_,config_,state_.particles.num_local,compute_stream_);
            std::cout << "Shear diagnostics: peculiar_KE=" << d.peculiar_kinetic_energy
                      << ", Hill/Jacobi=" << d.hill_jacobi_invariant << std::endl;
        }

        if ((config_.io.output_schedule_count > 0 || config_.time.output_step > 0.0) &&
            state_.current_time + 1.0e-12 * std::max(1.0, std::fabs(state_.current_time))
                >= next_output_time_) {
            adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();
            const bool adaptive_fatal = append_adaptive_yorp_diagnostic();
            if (save_output()) final_output_tracker_.mark_written(state_.current_step);
            save_bonded_wall_state();
            save_clump_state();
            if (config_.io.output_schedule_count > 0) {
                while (next_output_schedule_index_ < config_.io.output_schedule_count && state_.current_time >= config_.io.output_schedule_times[next_output_schedule_index_]) ++next_output_schedule_index_;
                next_output_time_ = next_output_schedule_index_ < config_.io.output_schedule_count ? config_.io.output_schedule_times[next_output_schedule_index_] : std::numeric_limits<double>::infinity();
            } else { do { next_output_time_ += config_.time.output_step; } while (state_.current_time >= next_output_time_); }

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - wall_start_).count();
            std::cout << "Step " << state_.current_step
                      << " (t=" << state_.current_time << "s) - "
                      << elapsed << " ms elapsed" << std::endl;
            if (adaptive_fatal) {
                throw std::runtime_error(
                    "Adaptive YORP encountered invalid metrics at an output boundary");
            }
        }

        if (config_.io.checkpoint_enabled && config_.io.checkpoint_interval > 0 &&
            state_.current_step % config_.io.checkpoint_interval == 0) {
            save_checkpoint();
        }
        record_scalar_receipt(/*force=*/false);
    }

    void begin_timing_window_if_due() {
        if (config_.io.timing_measure_steps <= 0 || timing_window_finished_) return;
        if (!timing_window_started_) {
            if (state_.current_step - timing_start_step_ < config_.io.timing_warmup_steps) return;
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            timing_window_started_ = true;
            timing_window_start_step_ = state_.current_step;
            timing_window_external_gravity_start_ = ExternalBodyManager::gravity_evaluations();
        }
        if (!timing_window_force_segment_active_) {
            timing_window_start_ = std::chrono::high_resolution_clock::now();
            timing_window_force_segment_active_ = true;
        }
    }

    void end_timing_window_if_due() {
        if (!timing_window_started_ || timing_window_finished_ || !timing_window_force_segment_active_ || config_.io.timing_measure_steps <= 0) return;
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        timing_window_wall_seconds_ += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - timing_window_start_).count();
        timing_window_force_segment_active_ = false;
        // Always pause here: complete_synchronized_boundary may write output,
        // checkpoints, diagnostics, and scalar receipts. A later outer range
        // restarts the force-only segment if the requested inner-step count
        // has not yet been reached.
        if (state_.current_step - timing_window_start_step_ < config_.io.timing_measure_steps) return;
        timing_window_end_step_ = state_.current_step;
        timing_window_external_gravity_end_ = ExternalBodyManager::gravity_evaluations();
        timing_window_finished_ = true;
    }

    void run() {
        if (!initialized_) {
            std::cerr << "ERROR: Simulation not initialized" << std::endl;
            return;
        }

        // Warm-up CUDA context
        cudaFree(0);

        std::cout << "\n========== Starting Simulation ===========" << std::endl;

        wall_start_ = std::chrono::high_resolution_clock::now();
        timing_start_step_ = state_.current_step;
        timing_start_time_ = state_.current_time;
        timing_window_started_=timing_window_finished_=timing_window_force_segment_active_=false; timing_window_start_step_=timing_window_end_step_=state_.current_step;
        timing_window_wall_seconds_=0.0; timing_window_external_gravity_start_=timing_window_external_gravity_end_=0; self_gravity_evaluations_=0;
        // Keep the output grid anchored to the configured start time.  A
        // resumed run must close RESPA at the same future output boundaries
        // as an uninterrupted run; shifting the grid to checkpoint_time +
        // output_step would otherwise change the slow-force schedule.
        if (config_.io.output_schedule_count > 0) {
            next_output_schedule_index_=0;
            while(next_output_schedule_index_<config_.io.output_schedule_count && config_.io.output_schedule_times[next_output_schedule_index_] <= state_.current_time + 1.e-12) ++next_output_schedule_index_;
            next_output_time_=next_output_schedule_index_<config_.io.output_schedule_count ? config_.io.output_schedule_times[next_output_schedule_index_] : std::numeric_limits<double>::infinity();
        } else if (config_.time.output_step > 0.0) {
            const double elapsed = std::max(
                0.0, state_.current_time - config_.time.start_time);
            const double completed = std::floor(
                elapsed / config_.time.output_step + 1.0e-12);
            next_output_time_ = config_.time.start_time +
                (completed + 1.0) * config_.time.output_step;
        } else {
            next_output_time_ = std::numeric_limits<double>::infinity();
        }
        steps_since_output_ = 0;
        final_output_tracker_.reset();

        // -------------------------------------------------------------------------
        // Main time stepping loop
        // -------------------------------------------------------------------------
        const int respa_k = config_.acceleration.gravity_update_rate;
        const bool respa_enabled = config_.acceleration.enable_self_gravity && respa_k > 1;

        // Ordinary inputs/outputs do not carry an integrator-level force
        // history. Evaluate the retained force at q(t0) before any opening
        // kick or drift, just as LAMMPS initializes every RESPA level.
        const bool resumed = state_.particles.source_record_kind ==
            static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT);
        if (!resumed) {
            run_dem_step(/*include_gravity=*/!respa_enabled,
                         /*force_only=*/true);
        }

        if(resumed && clump_state_.num_clumps>0 &&
           config_.acceleration.enable_self_gravity && !respa_enabled){
            AccelerationManager::restore_checkpoint_clump_gravity(
                state_,config_,state_.current_time,compute_stream_);
            ++self_gravity_evaluations_;
        }

        if (respa_enabled) {
            // Bootstrap the outer force at the current configuration. The
            // cached normal acceleration and clump gravity wrench remain
            // separate from the inner contact-force Verlet state.
            AccelerationManager::compute_outer_gravity(
                state_,config_,state_.current_time,compute_stream_);
            ++self_gravity_evaluations_;
        }

        // Initial output must include the bootstrapped slow force when RESPA is active.
        if (config_.io.output_initial_state &&
            final_output_tracker_.needs_output(state_.current_step)) {
            if (save_output()) final_output_tracker_.mark_written(state_.current_step);
            record_scalar_receipt(/*force=*/true);
        }

        while (!should_terminate()) {
            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePushA("SimulationStep");
            #endif

            // A RESPA measurement is an entire outer range: both gravity
            // impulses, all inner steps, and the refreshed outer gravity.
            begin_timing_window_if_due();
            if (respa_enabled) {
                const int inner_count = synchronized_respa_inner_count(respa_k);
                const double half_outer_dt = 0.5 * terminal_limited_respa_span(
                    state_.current_time, state_.dt, inner_count,
                    config_.time.stop_time);
                apply_respa_gravity_impulse(half_outer_dt);
                for (int inner=0;inner<inner_count;++inner) run_time_limited_dem_step(/*include_gravity=*/false);
                AccelerationManager::compute_outer_gravity(
                    state_,config_,state_.current_time,compute_stream_);
                ++self_gravity_evaluations_;
                apply_respa_gravity_impulse(half_outer_dt);
                end_timing_window_if_due();
                complete_synchronized_boundary();
            } else {
                // ── STANDARD SINGLE STEP (k=1 or RESPA disabled) ─────
                run_time_limited_dem_step(/*include_gravity=*/true);
                end_timing_window_if_due();
                complete_synchronized_boundary();
            }

            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePop(); // SimulationStep
            #endif
        }

        // Final synchronization
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // Persist the exact terminating state unless the regular interval
        // output path already wrote this step. This also covers an immediate
        // successful termination before the first integration step.
        bool adaptive_fatal = false;
        if (final_output_tracker_.needs_output(state_.current_step)) {
            adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();
            adaptive_fatal = append_adaptive_yorp_diagnostic();
            if (save_output()) {
                final_output_tracker_.mark_written(state_.current_step);
            }
            save_bonded_wall_state();
            save_clump_state();
        }
        if (adaptive_fatal) {
            throw std::runtime_error("Adaptive YORP encountered invalid metrics at the final output boundary");
        }
        write_timing_receipt();

        // Compute final specific KE for output
        double final_specific_ke = (total_mass_ > 0) ? state_.kinetic_energy / total_mass_ : 0.0;

        std::cout << "\n========== Simulation Complete ===========" << std::endl;
        std::cout << "Final time:  " << state_.current_time << " s" << std::endl;
        std::cout << "Total steps: " << state_.current_step << std::endl;
        std::cout << "Final KE:    " << state_.kinetic_energy << " J" << std::endl;
        std::cout << "Final ratio = " << std::scientific << std::setprecision(4) << final_specific_ke << std::endl;
        std::cout << "============================================" << std::endl;
    }

    // =========================================================================
    // Single DEM Step (extracted for RESPA outer/inner composition)
    // =========================================================================
    // Contains the full verlet + force pipeline for one timestep.
    // include_gravity: true for full physics, false for inner steps (no gravity).
    // The outer SimulationStep NVTX range is managed by run() (Task 8).
    // =========================================================================

    void sync_clump_members() {
        const int32_t base=clump_segments_.member_begin;
        const int32_t count=clump_segments_.member_end-base;
        if(clump_state_.num_clumps<=0||count<=0)return;
        const int block=config_.gpu.block_size,grid=(count+block-1)/block;
        LAUNCH_KERNEL(clump_sync_kinematics_kernel,grid,block,0,compute_stream_,
            state_.particles.pos_x+base,state_.particles.pos_y+base,state_.particles.pos_z+base,
            state_.particles.vel_x+base,state_.particles.vel_y+base,state_.particles.vel_z+base,
            state_.particles.omega_x+base,state_.particles.omega_y+base,state_.particles.omega_z+base,
            state_.particles.quat_w+base,state_.particles.quat_x+base,state_.particles.quat_y+base,state_.particles.quat_z+base,
            state_.particles.clump_off_x+base,state_.particles.clump_off_y+base,state_.particles.clump_off_z+base,
            state_.particles.clump_id+base,clump_state_.com_x,clump_state_.com_y,clump_state_.com_z,
            clump_state_.vel_x,clump_state_.vel_y,clump_state_.vel_z,
            clump_state_.omega_x,clump_state_.omega_y,clump_state_.omega_z,
            clump_state_.quat_w,clump_state_.quat_x,clump_state_.quat_y,clump_state_.quat_z,count);
    }

    void apply_respa_gravity_impulse(double impulse_dt) {
        const int block=config_.gpu.block_size;
        if(num_free_particles_>0) {
            const int grid=(num_free_particles_+block-1)/block;
            LAUNCH_KERNEL(apply_half_gravity_impulse_kernel,grid,block,0,compute_stream_,
                state_.particles.vel_x,state_.particles.vel_y,state_.particles.vel_z,
                state_.particles.gravity_force_x,state_.particles.gravity_force_y,state_.particles.gravity_force_z,
                impulse_dt,num_free_particles_);
        }
        if(clump_state_.num_clumps>0) {
            const int grid=(clump_state_.num_clumps+block-1)/block;
            LAUNCH_KERNEL(clump_apply_gravity_impulse_kernel,grid,block,0,compute_stream_,
                clump_state_.vel_x,clump_state_.vel_y,clump_state_.vel_z,
                clump_state_.omega_x,clump_state_.omega_y,clump_state_.omega_z,
                clump_state_.quat_w,clump_state_.quat_x,clump_state_.quat_y,clump_state_.quat_z,
                clump_state_.gravity_force_x,clump_state_.gravity_force_y,clump_state_.gravity_force_z,
                clump_state_.gravity_torque_x,clump_state_.gravity_torque_y,clump_state_.gravity_torque_z,
                clump_state_.mass,
                clump_state_.inertia_xx,clump_state_.inertia_xy,clump_state_.inertia_xz,
                clump_state_.inertia_yy,clump_state_.inertia_yz,clump_state_.inertia_zz,
                impulse_dt,clump_state_.num_clumps);
            sync_clump_members();
        }
    }

    void apply_half_viscous_damping() {
        if (!config_.damping.enabled || config_.damping.type != DampingType::VISCOUS)
            return;
        const double full_factor = std::max(
            0.0, 1.0 - static_cast<double>(config_.damping.coefficient) * state_.dt);
        const float half_factor = static_cast<float>(std::sqrt(full_factor));
        const int block = config_.gpu.block_size;
        if (num_free_particles_ > 0) {
            const int grid = (num_free_particles_ + block - 1) / block;
            LAUNCH_KERNEL(apply_velocity_damping_kernel, grid, block, 0, compute_stream_,
                state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                state_.particles.omega_x, state_.particles.omega_y, state_.particles.omega_z,
                half_factor, num_free_particles_);
        }
        if (clump_state_.num_clumps > 0) {
            const int grid = (clump_state_.num_clumps + block - 1) / block;
            LAUNCH_KERNEL(apply_velocity_damping_kernel, grid, block, 0, compute_stream_,
                clump_state_.vel_x, clump_state_.vel_y, clump_state_.vel_z,
                clump_state_.omega_x, clump_state_.omega_y, clump_state_.omega_z,
                half_factor, clump_state_.num_clumps);
            sync_clump_members();
        }
    }

    void process_saturn_absorption() {
        const ExternalBodyAbsorption delta = ExternalBodyManager::mark_absorbed(
            state_, config_, num_free_particles_, state_.current_step,
            state_.current_time, state_.dt, compute_stream_);
        if (delta.count == 0) return;
        if (delta.count < 0 || delta.count > num_free_particles_ ||
            !std::isfinite(delta.mass) || delta.mass < 0.0 ||
            delta.mass > total_mass_ + 1.0e-8 * std::max(1.0, total_mass_)) {
            throw std::runtime_error("invalid Saturn absorption delta");
        }

        // Canonicalization moves the dedicated absorbed lifecycle rows behind
        // the live prefix while preserving clump_id and immutable global IDs.
        const int32_t old_live = num_free_particles_;
        state_.particles.num_particles = num_total_particles_;
        state_.particles.num_local = num_total_particles_;
        state_.particles.num_ghost = 0;
        canonicalize_particle_clump_order(state_.particles, &state_.contact_history);
        num_free_particles_ = old_live - delta.count;
        state_.particles.num_particles = num_free_particles_;
        state_.particles.num_local = num_free_particles_;
        total_mass_ = std::max(0.0, total_mass_ - delta.mass);

        // The permutation preserves ordinary particle contact history, but a
        // new neighbor epoch must not reuse any stale index cache. Wall history
        // is irrelevant to this gravity-only case and is reset defensively.
        NeighborSearch::invalidate_cache();
        NeighborSearch::reset_contact_buffer(contacts_, compute_stream_);
        CUDA_CHECK(cudaMemsetAsync(state_.particles.num_contacts, 0,
                                   static_cast<size_t>(num_total_particles_) * sizeof(int32_t),
                                   compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(state_.particles.num_wall_contacts, 0,
                                   static_cast<size_t>(num_total_particles_) * sizeof(int32_t),
                                   compute_stream_));
        std::cout << "Saturn absorption: step=" << state_.current_step
                  << " count=" << delta.count << " mass=" << delta.mass
                  << " live=" << num_free_particles_ << std::endl;
        ExternalBodyManager::write_event_receipts(config_.io.output_directory);
    }

    void record_scalar_receipt(bool force) {
        const int32_t interval = config_.io.scalar_interval;
        if (!force && (interval <= 0 || state_.current_step % interval != 0)) return;
        if (config_.io.output_directory[0] == '\0') return;
        std::filesystem::path directory(config_.io.output_directory);
        std::filesystem::create_directories(directory);
        std::ofstream file(directory / "scalars.jsonl", std::ios::app);
        if (!file) throw std::runtime_error("cannot append scalar receipt");
        const ExternalBodyLedger loss = ExternalBodyManager::ledger();
        file << std::setprecision(17)
             << "{\"schema_version\":\"granit-scalar-v1\",\"step\":"
             << state_.current_step << ",\"time_seconds\":" << state_.current_time
             << ",\"live_particles\":" << num_free_particles_
             << ",\"live_mass_kg\":" << total_mass_
             << ",\"absorbed_particles\":" << loss.absorbed_particles
             << ",\"absorbed_mass_kg\":" << loss.absorbed_mass
             << ",\"absorbed_momentum_kg_m_s\":[" << loss.absorbed_momentum_x << "," << loss.absorbed_momentum_y << "," << loss.absorbed_momentum_z << "]"
             << ",\"absorbed_angular_momentum_kg_m2_s\":[" << loss.absorbed_angular_momentum_x << "," << loss.absorbed_angular_momentum_y << "," << loss.absorbed_angular_momentum_z << "]"
             << ",\"absorbed_orbital_energy_j\":" << loss.absorbed_orbital_energy
             << ",\"external_gravity_evaluations\":" << ExternalBodyManager::gravity_evaluations() << "}\n";
        if (!file) throw std::runtime_error("failed writing scalar receipt");
    }

    void write_timing_receipt() {
        if (config_.io.output_directory[0] == '\0' &&
            config_.io.timing_receipt_path[0] == '\0') return;
        const auto now = std::chrono::high_resolution_clock::now();
        const double wall_seconds = std::chrono::duration<double>(now - wall_start_).count();
        const int64_t completed_steps = state_.current_step - timing_start_step_;
        const double simulated_seconds = state_.current_time - timing_start_time_;
        std::filesystem::path path = config_.io.timing_receipt_path[0] != '\0'
            ? std::filesystem::path(config_.io.timing_receipt_path)
            : std::filesystem::path(config_.io.output_directory) / "timing_receipt.json";
        if (std::filesystem::exists(path))
            throw std::runtime_error("refusing to overwrite timing receipt: " + path.string());
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        if (!file) throw std::runtime_error("cannot write timing receipt");
        const ExternalBodyLedger loss = ExternalBodyManager::ledger();
        file << std::setprecision(17)
             << "{\n  \"schema_version\": \"granit-timing-v1\",\n"
             << "  \"start_step\": " << timing_start_step_ << ",\n"
             << "  \"end_step\": " << state_.current_step << ",\n"
             << "  \"completed_steps\": " << completed_steps << ",\n"
             << "  \"simulated_seconds\": " << simulated_seconds << ",\n"
             << "  \"wall_seconds\": " << wall_seconds << ",\n"
             << "  \"configured_warmup_steps\": " << config_.io.timing_warmup_steps << ",\n"
             << "  \"configured_measure_steps\": " << config_.io.timing_measure_steps << ",\n"
             << "  \"compute_window_completed\": " << (timing_window_finished_ ? "true" : "false") << ",\n"
             << "  \"compute_window_start_step\": " << timing_window_start_step_ << ",\n"
             << "  \"compute_window_end_step\": " << timing_window_end_step_ << ",\n"
             << "  \"compute_window_wall_seconds\": " << timing_window_wall_seconds_ << ",\n"
             << "  \"compute_window_external_gravity_evaluations\": " << (timing_window_finished_ ? timing_window_external_gravity_end_ - timing_window_external_gravity_start_ : 0) << ",\n"
             << "  \"self_gravity_evaluations\": " << self_gravity_evaluations_ << ",\n"
             << "  \"wall_seconds_per_step\": "
             << (completed_steps > 0 ? wall_seconds / completed_steps : 0.0) << ",\n"
             << "  \"simulated_seconds_per_wall_second\": "
             << (wall_seconds > 0.0 ? simulated_seconds / wall_seconds : 0.0) << ",\n"
             << "  \"live_particles\": " << num_free_particles_ << ",\n"
             << "  \"external_gravity_evaluations\": " << ExternalBodyManager::gravity_evaluations() << ",\n"
             << "  \"absorbed_particles\": " << loss.absorbed_particles << ",\n"
             << "  \"absorbed_mass_kg\": " << loss.absorbed_mass << "\n}\n";
        if (!file) throw std::runtime_error("failed writing timing receipt");
    }

    void run_time_limited_dem_step(bool include_gravity) {
        const double nominal_dt = state_.dt;
        const double step_dt = terminal_limited_step_dt(
            state_.current_time, nominal_dt, config_.time.stop_time);
        const bool shortened = step_dt < nominal_dt;
        state_.dt = step_dt;
        try {
            run_dem_step(include_gravity);
        } catch (...) {
            state_.dt = nominal_dt;
            throw;
        }
        state_.dt = nominal_dt;
        if (shortened) {
            // Remove a possible subtraction/addition roundoff residue after
            // force evaluation over the exact terminal interval.
            state_.current_time = config_.time.stop_time;
        }
    }

    void run_dem_step(bool include_gravity, bool force_only = false) {
        int32_t num_free = num_free_particles_;
        int block_size = config_.gpu.block_size;
        int num_free_blocks = (num_free + block_size - 1) / block_size;
        const float force_dt = force_only ? 0.0f : static_cast<float>(state_.dt);
        const double force_time = force_only ? state_.current_time
                                             : state_.current_time + state_.dt;
        if (!force_only && config_.shear_box.enabled)
            ShearBoxManager::validate_step(state_,config_,WallManager::get_pbc_config());

        if (!force_only) apply_half_viscous_damping();

        // Step 1: Velocity Verlet - First half
        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePushA("Verlet1");
        #endif
        if (!force_only && num_free > 0 && config_.shear_box.enabled) {
            ShearBoxManager::drift_step1(state_,config_,state_.dt,num_free,compute_stream_);
        } else if (!force_only && num_free > 0) {
            LAUNCH_KERNEL(verlet_step1_fused_kernel, num_free_blocks, block_size, 0, compute_stream_,
                state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
                state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                state_.particles.omega_x, state_.particles.omega_y, state_.particles.omega_z,
                state_.particles.quat_w, state_.particles.quat_x,
                state_.particles.quat_y, state_.particles.quat_z,
                state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                state_.particles.torque_x, state_.particles.torque_y, state_.particles.torque_z,
                state_.particles.mass, state_.particles.moment_of_inertia,
                state_.dt, num_free
            );
        }
        if (!force_only && clump_state_.num_clumps > 0) {
            int clump_blocks = (clump_state_.num_clumps + block_size - 1) / block_size;
            LAUNCH_KERNEL(clump_verlet_step1_kernel, clump_blocks, block_size, 0, compute_stream_,
                clump_state_.com_x, clump_state_.com_y, clump_state_.com_z,
                clump_state_.vel_x, clump_state_.vel_y, clump_state_.vel_z,
                clump_state_.omega_x, clump_state_.omega_y, clump_state_.omega_z,
                clump_state_.net_force_x, clump_state_.net_force_y, clump_state_.net_force_z,
                clump_state_.net_torque_x, clump_state_.net_torque_y, clump_state_.net_torque_z,
                clump_state_.quat_w, clump_state_.quat_x, clump_state_.quat_y, clump_state_.quat_z,
                clump_state_.mass,
                clump_state_.inertia_xx, clump_state_.inertia_xy, clump_state_.inertia_xz,
                clump_state_.inertia_yy, clump_state_.inertia_yz, clump_state_.inertia_zz,
                static_cast<double>(state_.dt), clump_state_.num_clumps
            );
            sync_clump_members();
        }

        // A Saturn centre crossing is an absorbing lifecycle event, not a
        // force or a mechanical wall collision.  It must occur after drift
        // but before neighbor/contact work so the removed row cannot take a
        // second force, collision, or loss count. The global-ring contract
        // forbids clumps, making the existing canonical live/dead ordering a
        // deterministic, checkpoint-safe compaction mechanism.
        if (!force_only && ExternalBodyManager::is_enabled()) {
            process_saturn_absorption();
            num_free = num_free_particles_;
            num_free_blocks = (num_free + block_size - 1) / block_size;
        }
        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePop();
        #endif

        if (force_only) {
            const size_t bytes = static_cast<size_t>(state_.particles.num_particles) * sizeof(double);
            CUDA_CHECK(cudaMemsetAsync(state_.particles.force_x, 0, bytes, compute_stream_));
            CUDA_CHECK(cudaMemsetAsync(state_.particles.force_y, 0, bytes, compute_stream_));
            CUDA_CHECK(cudaMemsetAsync(state_.particles.force_z, 0, bytes, compute_stream_));
            CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_x, 0, bytes, compute_stream_));
            CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_y, 0, bytes, compute_stream_));
            CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_z, 0, bytes, compute_stream_));
        }

        // Step 2: The first half-kick consumed the force retained from the
        // previous step.  Clear body/member accumulators only now, before
        // assembling the force at the new position for the second half-kick.
        if (clump_state_.num_clumps > 0) {
            int clump_blocks = (clump_state_.num_clumps + block_size - 1) / block_size;
            LAUNCH_KERNEL(clump_zero_accumulators_kernel, clump_blocks, block_size, 0, compute_stream_,
                clump_state_.net_force_x, clump_state_.net_force_y, clump_state_.net_force_z,
                clump_state_.net_torque_x, clump_state_.net_torque_y, clump_state_.net_torque_z,
                clump_state_.num_clumps
            );
        }

        // Reset clump member forces.
        if (clump_state_.num_clumps > 0) {
            int32_t num_sub = clump_segments_.member_end - clump_segments_.member_begin;
            if (num_sub > 0) {
                size_t bytes = num_sub * sizeof(double);
                const int32_t base = clump_segments_.member_begin;
                CUDA_CHECK(cudaMemsetAsync(state_.particles.force_x + base, 0, bytes, compute_stream_));
                CUDA_CHECK(cudaMemsetAsync(state_.particles.force_y + base, 0, bytes, compute_stream_));
                CUDA_CHECK(cudaMemsetAsync(state_.particles.force_z + base, 0, bytes, compute_stream_));
                CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_x + base, 0, bytes, compute_stream_));
                CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_y + base, 0, bytes, compute_stream_));
                CUDA_CHECK(cudaMemsetAsync(state_.particles.torque_z + base, 0, bytes, compute_stream_));
            }
        }

        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePushA("ForceCalculation");
        #endif

        // Step 3: Periodic boundaries
        const int32_t live_particles=state_.particles.num_particles;
        state_.particles.num_particles=num_free_particles_;
        ShearPhase shear_phase;
        if(config_.shear_box.enabled)
            shear_phase=ShearBoxManager::wrap(state_,config_,WallManager::get_pbc_config(),
                force_time,num_free_particles_,compute_stream_);
        else
            WallManager::apply_periodic_boundaries(state_, config_, compute_stream_);
        state_.particles.num_particles=live_particles;
        if(clump_state_.num_clumps>0){const auto& pbc=WallManager::get_pbc_config();const int grid=(clump_state_.num_clumps+block_size-1)/block_size;LAUNCH_KERNEL(clump_wrap_com_kernel,grid,block_size,0,compute_stream_,clump_state_.com_x,clump_state_.com_y,clump_state_.com_z,clump_state_.num_clumps,pbc.periodic_x,pbc.periodic_y,pbc.periodic_z,pbc.box_min_x,pbc.box_min_y,pbc.box_min_z,pbc.box_length_x,pbc.box_length_y,pbc.box_length_z);sync_clump_members();}

        // Step 4: Neighbor search
        NeighborSearch::find_contacts(state_, config_, contacts_, force_time, compute_stream_);
        if(contacts_.max_contacts>contact_sort_capacity_) {
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            cudaFree(d_sort_keys_); cudaFree(d_sorted_sort_keys_);
            cudaFree(d_sort_indices_); cudaFree(d_sorted_sort_indices_);
            cudaFree(d_contact_sort_temp_);
            CUDA_CHECK(cudaMalloc(&d_sort_keys_,static_cast<size_t>(contacts_.max_contacts)*sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_sorted_sort_keys_,static_cast<size_t>(contacts_.max_contacts)*sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_sort_indices_,static_cast<size_t>(contacts_.max_contacts)*sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_sorted_sort_indices_,static_cast<size_t>(contacts_.max_contacts)*sizeof(int32_t)));
#ifndef CUDADEM_CPU_MOCK
            size_t radix_sort_temp_bytes = 0;
            size_t owner_scan_temp_bytes = 0;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr, radix_sort_temp_bytes,
                static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
                static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
                contacts_.max_contacts, 0, 64, compute_stream_));
            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                nullptr, owner_scan_temp_bytes,
                static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
                OwnerOffsetMinimum{}, state_.particles.num_particles + 1, compute_stream_));
            contact_sort_temp_bytes_ = std::max(radix_sort_temp_bytes, owner_scan_temp_bytes);
            CUDA_CHECK(cudaMalloc(&d_contact_sort_temp_, contact_sort_temp_bytes_));
#else
            d_contact_sort_temp_ = nullptr;
            contact_sort_temp_bytes_ = 0;
#endif
            contact_sort_capacity_=contacts_.max_contacts;
        }
        contacts_dirty_ = true;
        if (!force_only) dump_contacts_to_file(shear_phase);
        const int32_t num_live = state_.particles.num_particles;

        // Step 4.5: Compact contact history (every 5 steps instead of every step)
        // Particles typically have 4-8 contacts; max is 16. Between compactions,
        // num_contacts grows by at most ~1 per step (new contacts replacing old).
        // With 10+ free slots, compacting every 5 steps reduces kernel cost by
        // 80% (from 17.4% to ~3.5% of total time) while staying safely below limit.
        constexpr int32_t COMPACTION_INTERVAL = 5;
        if (config_.shear_box.enabled || state_.current_step % COMPACTION_INTERVAL == 0) {
            launch_compact_contact_history_kernel(
                state_.contact_history.data,
                state_.particles.contact_history_offset,
                state_.particles.num_contacts,
                state_.contact_history.max_contacts_per_particle,
                num_live,
                config_.gpu.block_size,
                compute_stream_
            );
        }
        // Step 5: Contact forces
        // Sort contacts by owner to enable register accumulation (FMM-inspired).
        // Grid is sized by particle count, not contact count, since each thread
        // processes all contacts for one owner particle.
        // LBVH has just completed a synchronized count readback to maintain
        // its cache/statistics.  Reuse that valid host mirror.  Other methods
        // retain their existing explicit readback behavior.
        int32_t h_num_contacts = contacts_.h_num_contacts;
        if (config_.neighbor.method != NeighborMethod::LBVH) {
            h_num_contacts = NeighborSearch::get_contact_count(contacts_, compute_stream_);
        } else if (h_num_contacts == 0) {
            // An LBVH trivial-return path resets the host mirror without a
            // D2H count read.  Keep the former sort-boundary error semantics
            // for that no-contact case without performing a new readback.
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        }
        // Retain the previous terminal-FMM error observation at a completed
        // stream boundary without a second sort-helper synchronization.
        AccelerationManager::throw_if_fmm_terminal_error();
        int32_t h_num_contacts_from_sort = sort_contacts_by_owner(
            contacts_.pairs, h_num_contacts,
            d_owner_offsets_,
            d_sort_keys_, d_sorted_sort_keys_,
            d_sort_indices_, d_sorted_sort_indices_,
            d_reversed_owner_offsets_, d_contact_sort_temp_, contact_sort_temp_bytes_,
            num_live, contacts_.max_contacts,
            config_.gpu.block_size, compute_stream_
        );
        *h_pinned_num_contacts_ = h_num_contacts_from_sort;
        contacts_dirty_ = false;

        {
            int contact_block_size = config_.gpu.block_size;
            int num_blocks = (num_live + contact_block_size - 1) / contact_block_size;
            bool use_visco = config_.dem_damping_model == DemDampingModel::VISCOUS;
            const auto& pbc = WallManager::get_pbc_config();

            LAUNCH_KERNEL(compute_contact_forces_from_pairs_kernel, num_blocks, contact_block_size, 0, compute_stream_,
                contacts_.pairs, d_sorted_sort_indices_,       // contacts + sorted perm indices
                contacts_.num_contacts,
                d_owner_offsets_, num_live,
                state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
                state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                state_.particles.omega_x, state_.particles.omega_y, state_.particles.omega_z,
                state_.particles.radius, state_.particles.mass, state_.particles.moment_of_inertia,
                state_.particles.material_id, state_.material_database.materials,
                state_.material_database.dem_interactions, state_.material_database.material_count,
                state_.particles.global_id,
                state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                state_.particles.torque_x, state_.particles.torque_y, state_.particles.torque_z,
                state_.contact_history.data,
                state_.particles.contact_history_offset, state_.particles.num_contacts,
                state_.contact_history.max_contacts_per_particle,
                use_visco,
                force_dt, contacts_.max_contacts,
                pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                pbc.half_length_x, pbc.half_length_y, pbc.half_length_z,
                config_.shear_box.enabled,
                static_cast<float>(shear_phase.reduced_shift),
                static_cast<float>(shear_phase.shear_velocity)
            );
        }
        // Step 5.5: Compact wall contact history (recycle inactive wall slots)
        // Must run BEFORE wall force computation so wall kernels see a clean history.
        // Without this, wall history slots fill up over time: contacts that ended
        // occupy slots forever, new contacts get slot=-1 (no history), and friction
        // loses tangential spring memory → particles slide/shake instead of settling.
        WallManager::cleanup_contact_history(state_, config_, compute_stream_);

        // Step 6: Wall forces
        WallManager::compute_forces(
            state_, config_, force_dt, force_time, compute_stream_);

        // Step 7: Non-gravity accelerations. Body gravity is evaluated after
        // member-force reduction so its clump wrench is not overwritten.
        AccelerationManager::compute_all_without_gravity(
            state_, config_, force_time, compute_stream_, !force_only);

        // Step 7.5: Sphere wall gravity
        SphereWallManager::compute_gravity(state_, config_, compute_stream_);

        // Prescribed Saturn/satellite gravity is a fast-level force and is
        // therefore evaluated on every inner step, outside the RESPA cache.
        ExternalBodyManager::add_forces(
            state_, config_, force_time, num_free, compute_stream_);

        // Clump force aggregation
        if (clump_state_.num_clumps > 0) {
            const int32_t base = clump_segments_.member_begin;
            int32_t num_sub_spheres = clump_segments_.member_end - base;
            if (num_sub_spheres > 0) {
                int sub_blocks = (num_sub_spheres + block_size - 1) / block_size;
                LAUNCH_KERNEL(clump_pre_aggregate_kernel, sub_blocks, block_size, 0, compute_stream_,
                    state_.particles.force_x + base,
                    state_.particles.force_y + base,
                    state_.particles.force_z + base,
                    state_.particles.torque_x + base,
                    state_.particles.torque_y + base,
                    state_.particles.torque_z + base,
                    state_.particles.pos_x + base,
                    state_.particles.pos_y + base,
                    state_.particles.pos_z + base,
                    state_.particles.clump_id + base,
                    clump_state_.com_x, clump_state_.com_y, clump_state_.com_z,
                    clump_staging_.fx, clump_staging_.fy, clump_staging_.fz,
                    clump_staging_.tx, clump_staging_.ty, clump_staging_.tz,
                    num_sub_spheres
                );
                clump_aggregate_forces(clump_staging_, clump_state_, clump_segments_, num_sub_spheres, compute_stream_);
            }
        }
        if (include_gravity) {
            AccelerationManager::compute_self_gravity(
                state_, config_, force_time, compute_stream_);
            ++self_gravity_evaluations_;
        }

        NeighborSearch::launch_async_displacement_monitor(state_, config_, compute_stream_);
        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePop(); // ForceCalculation
        #endif

        if (force_only) return;

        // Step 8: Verlet step 2
        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePushA("Verlet2");
        #endif
        if (num_free > 0 && config_.shear_box.enabled) {
            ShearBoxManager::kick_step2(state_,config_,state_.dt,num_free,compute_stream_);
        } else if (num_free > 0) {
            LAUNCH_KERNEL(verlet_step2_fused_kernel, num_free_blocks, block_size, 0, compute_stream_,
                state_.particles.vel_x, state_.particles.vel_y, state_.particles.vel_z,
                state_.particles.omega_x, state_.particles.omega_y, state_.particles.omega_z,
                state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                state_.particles.torque_x, state_.particles.torque_y, state_.particles.torque_z,
                state_.particles.mass, state_.particles.moment_of_inertia,
                state_.dt, num_free
            );
        }

        if (clump_state_.num_clumps > 0) {
            int clump_blocks = (clump_state_.num_clumps + block_size - 1) / block_size;
            LAUNCH_KERNEL(clump_verlet_step2_kernel, clump_blocks, block_size, 0, compute_stream_,
                clump_state_.com_x, clump_state_.com_y, clump_state_.com_z,
                clump_state_.vel_x, clump_state_.vel_y, clump_state_.vel_z,
                clump_state_.omega_x, clump_state_.omega_y, clump_state_.omega_z,
                clump_state_.quat_w, clump_state_.quat_x, clump_state_.quat_y, clump_state_.quat_z,
                clump_state_.net_force_x, clump_state_.net_force_y, clump_state_.net_force_z,
                clump_state_.net_torque_x, clump_state_.net_torque_y, clump_state_.net_torque_z,
                clump_state_.mass,
                clump_state_.inertia_xx, clump_state_.inertia_xy, clump_state_.inertia_xz,
                clump_state_.inertia_yy, clump_state_.inertia_yz, clump_state_.inertia_zz,
                static_cast<double>(state_.dt), clump_state_.num_clumps
            );

            sync_clump_members();
        }

        // Symmetric split of separable viscous damping around the inner step.
        apply_half_viscous_damping();
        #ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePop(); // Verlet2
        #endif

        // Step 10: Advance time
        state_.current_time = force_time;
        state_.current_step++;
        steps_since_output_++;

    }

    // =========================================================================
    // Termination Condition
    // =========================================================================

    bool should_terminate() {
        // Adaptive status is intentionally cached by the output/checkpoint
        // boundary consumer; this predicate never adds a 50-second sync.
        if (config_.termination == TerminationCondition::ADAPTIVE_YORP &&
            config_.acceleration.yorp.adaptive &&
            AccelerationManager::adaptive_yorp_should_terminate(compute_stream_)) {
            return true;
        }
        // Check time-based termination
        if (config_.termination == TerminationCondition::TIME) {
            return state_.current_time >= config_.time.stop_time;
        }

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

        if (config_.termination == TerminationCondition::ADAPTIVE_YORP) {
            // A malformed/unreachable adaptive state must still respect the
            // configured physical time horizon.
            return state_.current_time >= config_.time.stop_time;
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

    bool save_output() {
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

        const bool respa_enabled = config_.acceleration.enable_self_gravity &&
            config_.acceleration.gravity_update_rate > 1;
        if (respa_enabled) {
            const int block = config_.gpu.block_size;
            const int grid = (num_total_particles_ + block - 1) / block;
            LAUNCH_KERNEL(compose_respa_output_force_kernel, grid, block, 0, compute_stream_,
                state_.particles.force_x, state_.particles.force_y, state_.particles.force_z,
                state_.particles.gravity_force_x, state_.particles.gravity_force_y,
                state_.particles.gravity_force_z,
                state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
                state_.particles.mass, state_.particles.clump_id,
                clump_state_.com_x, clump_state_.com_y, clump_state_.com_z,
                clump_state_.quat_w, clump_state_.quat_x, clump_state_.quat_y, clump_state_.quat_z,
                clump_state_.gravity_force_x, clump_state_.gravity_force_y, clump_state_.gravity_force_z,
                clump_state_.gravity_torque_x, clump_state_.gravity_torque_y, clump_state_.gravity_torque_z,
                clump_state_.mass,
                clump_state_.inertia_xx, clump_state_.inertia_xy, clump_state_.inertia_xz,
                clump_state_.inertia_yy, clump_state_.inertia_yz, clump_state_.inertia_zz,
                d_output_force_x_, d_output_force_y_, d_output_force_z_, num_total_particles_);
        }
        if(config_.shear_box.enabled) {
            ShearBoxManager::compose_output_force(
                state_,config_,respa_enabled?d_output_force_x_:state_.particles.force_x,
                respa_enabled?d_output_force_y_:state_.particles.force_y,
                respa_enabled?d_output_force_z_:state_.particles.force_z,
                d_output_force_x_,d_output_force_y_,d_output_force_z_,
                num_total_particles_,compute_stream_);
        }

        // Write every stored row, including inert dead particles. RESPA uses
        // staging arrays so the retained fast-force history is never altered.
        const int32_t live_particles = state_.particles.num_particles;
        state_.particles.num_particles = num_total_particles_;
        const bool wrote_output = write_unified_binary(
            filepath, state_.particles, state_, config_, false, nullptr, 0, 0,
            (respa_enabled||config_.shear_box.enabled) ? d_output_force_x_ : nullptr,
            (respa_enabled||config_.shear_box.enabled) ? d_output_force_y_ : nullptr,
            (respa_enabled||config_.shear_box.enabled) ? d_output_force_z_ : nullptr);
        state_.particles.num_particles = live_particles;
        if (!wrote_output) {
            std::cerr << "WARNING: Failed to write unified binary output: " << filepath << std::endl;
            return false;
        }
        std::cout << "Output saved: " << filepath << std::endl;

        // =====================================================================
        // Optional gravity-force sidecar (for LET single-vs-multi A/B validation)
        // =====================================================================
        // Writes per-particle gravitational force (gravity_force_x/y/z, double,
        // contact-free) to "<base>.gravity_force" when io.output.fields includes
        // "gravity_force". Sorted by global_id so the index aligns with the
        // multi-GPU sidecar (which is also global_id-sorted) for direct A/B
        // comparison. Identical format to main_multigpu.cu's sidecar.
        // =====================================================================
        if (config_.io.output_gravity_force) {
            const auto& p = state_.particles;
            int32_t N = num_total_particles_;
            std::vector<double> h_gx(N), h_gy(N), h_gz(N);
            std::vector<int32_t> h_gid(N);
            if (respa_enabled && config_.shear_box.enabled) {
                // In shear mode the unified FORCE staging array also contains
                // the output-only Hill force. The RESPA cache is pure gravity
                // acceleration, and shear mode forbids clumps, so convert that
                // cache directly instead of subtracting from a Hill-composed
                // total force.
                std::vector<float> h_mass(N);
                CUDA_CHECK(cudaMemcpy(h_gx.data(), p.gravity_force_x, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gy.data(), p.gravity_force_y, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gz.data(), p.gravity_force_z, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_mass.data(), p.mass, N * sizeof(float), cudaMemcpyDeviceToHost));
                for (int32_t i = 0; i < N; ++i) {
                    const double mass = static_cast<double>(h_mass[i]);
                    h_gx[i] *= mass;
                    h_gy[i] *= mass;
                    h_gz[i] *= mass;
                }
            } else if (respa_enabled) {
                // The slow cache stores acceleration for ordinary particles.
                // Derive physical gravity force from the already staged total
                // force so this sidecar keeps its historical force units.
                std::vector<double> h_fast(N);
                CUDA_CHECK(cudaMemcpy(h_gx.data(), d_output_force_x_, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_fast.data(), p.force_x, N * sizeof(double), cudaMemcpyDeviceToHost));
                for (int32_t i = 0; i < N; ++i) h_gx[i] -= h_fast[i];
                CUDA_CHECK(cudaMemcpy(h_gy.data(), d_output_force_y_, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_fast.data(), p.force_y, N * sizeof(double), cudaMemcpyDeviceToHost));
                for (int32_t i = 0; i < N; ++i) h_gy[i] -= h_fast[i];
                CUDA_CHECK(cudaMemcpy(h_gz.data(), d_output_force_z_, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_fast.data(), p.force_z, N * sizeof(double), cudaMemcpyDeviceToHost));
                for (int32_t i = 0; i < N; ++i) h_gz[i] -= h_fast[i];
            } else {
                CUDA_CHECK(cudaMemcpy(h_gx.data(), p.gravity_force_x, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gy.data(), p.gravity_force_y, N * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gz.data(), p.gravity_force_z, N * sizeof(double), cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaMemcpy(h_gid.data(), p.global_id, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

            // Sort by global_id so the sidecar is index-aligned across runs.
            std::vector<int32_t> idx(N);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return h_gid[a] < h_gid[b]; });

            std::string gpath = filepath + ".gravity_force";
            std::ofstream gfile(gpath, std::ios::binary);
            if (gfile.is_open()) {
                const char magic[8] = "DEMOUT\0";
                uint32_t version = 1;
                int32_t n32 = N;
                int64_t step = state_.current_step;
                double t = state_.current_time, dt = state_.dt;
                gfile.write(magic, 8);
                gfile.write(reinterpret_cast<const char*>(&version), sizeof(version));
                gfile.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
                gfile.write(reinterpret_cast<const char*>(&step), sizeof(step));
                gfile.write(reinterpret_cast<const char*>(&t), sizeof(t));
                gfile.write(reinterpret_cast<const char*>(&dt), sizeof(dt));
                for (int32_t i : idx) { double v = h_gx[i]; gfile.write(reinterpret_cast<const char*>(&v), sizeof(double)); }
                for (int32_t i : idx) { double v = h_gy[i]; gfile.write(reinterpret_cast<const char*>(&v), sizeof(double)); }
                for (int32_t i : idx) { double v = h_gz[i]; gfile.write(reinterpret_cast<const char*>(&v), sizeof(double)); }
                gfile.close();
            } else {
                std::cerr << "WARNING: Cannot create gravity-force sidecar: " << gpath << std::endl;
            }
        }

        return true;
    }

    // =========================================================================
    // Contact Dump for Validation
    // =========================================================================

    void dump_contacts_to_file(const ShearPhase& shear_phase) {
        if (!config_.io.dump_contacts) return;

        // Generate output file path with step number
        std::ostringstream oss;
        oss << config_.io.output_directory << "/contacts_step_"
            << std::setfill('0') << std::setw(10) << state_.current_step << ".bin";
        std::string filepath = oss.str();

        // Create output directory if needed (save_output handles this, but be safe)
        std::filesystem::path output_dir(config_.io.output_directory);
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        // Get contact count (synchronizes stream)
        int32_t num_contacts = NeighborSearch::get_contact_count(contacts_, compute_stream_);
        int32_t num_particles = state_.particles.num_particles;

        // Materialize physical geometry at the same post-drift positions and
        // shear phase as force evaluation. No geometry buffer exists otherwise.
        std::vector<ContactDumpRecord> h_contacts(num_contacts);
        if (num_contacts > 0) {
            ContactDumpRecord* d_records = nullptr;
            CUDA_CHECK(cudaMalloc(&d_records,
                static_cast<size_t>(num_contacts) * sizeof(ContactDumpRecord)));
            const auto& pbc = WallManager::get_pbc_config();
            const int block_size = config_.gpu.block_size;
            const int blocks = (num_contacts + block_size - 1) / block_size;
            LAUNCH_KERNEL(materialize_contact_dump_kernel, blocks, block_size, 0, compute_stream_,
                contacts_.pairs, num_contacts,
                state_.particles.pos_x, state_.particles.pos_y, state_.particles.pos_z,
                state_.particles.radius, d_records,
                pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                config_.shear_box.enabled, static_cast<float>(shear_phase.reduced_shift));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_contacts.data(), d_records,
                static_cast<size_t>(num_contacts) * sizeof(ContactDumpRecord),
                cudaMemcpyDeviceToHost, compute_stream_));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            CUDA_CHECK(cudaFree(d_records));
        }

        // Write binary file
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "WARNING: Cannot create contact dump file: " << filepath << std::endl;
            return;
        }

        // Header: magic(8) + version(4) + num_contacts(4) + num_particles(4) + time(8) + reserved(4) = 32 bytes
        const char magic[8] = "CONTACT";
        int32_t version = 1;
        int32_t reserved = 0;
        file.write(magic, 8);
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&num_contacts), sizeof(num_contacts));
        file.write(reinterpret_cast<const char*>(&num_particles), sizeof(num_particles));
        file.write(reinterpret_cast<const char*>(&state_.current_time), sizeof(state_.current_time));
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));

        // Write each contact pair (28 bytes each)
        for (int32_t i = 0; i < num_contacts; ++i) {
            const ContactDumpRecord& c = h_contacts[i];
            file.write(reinterpret_cast<const char*>(&c.i), sizeof(c.i));
            file.write(reinterpret_cast<const char*>(&c.j), sizeof(c.j));
            file.write(reinterpret_cast<const char*>(&c.overlap), sizeof(c.overlap));
            file.write(reinterpret_cast<const char*>(&c.distance), sizeof(c.distance));
            file.write(reinterpret_cast<const char*>(&c.nx), sizeof(c.nx));
            file.write(reinterpret_cast<const char*>(&c.ny), sizeof(c.ny));
            file.write(reinterpret_cast<const char*>(&c.nz), sizeof(c.nz));
        }

        file.close();
        std::cout << "Contacts dumped: " << filepath << " (" << num_contacts << " contacts)" << std::endl;
    }

    void save_bonded_wall_state() {
        int32_t num_walls = cudadem::bonded_rect_wall_get_states(nullptr, nullptr);
        if (num_walls <= 0) return;

        std::vector<cudadem::BondedRectWallState> host_states(num_walls);
        std::vector<cudadem::BondedRectReaction> host_reactions(num_walls);
        cudadem::bonded_rect_wall_get_states(host_states.data(), host_reactions.data());

        std::ostringstream oss;
        oss << config_.io.output_directory << "/bonded_wall_states_step_"
            << std::setfill('0') << std::setw(10) << state_.current_step << ".bin";
        std::string filepath = oss.str();

        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "WARNING: Cannot create bonded wall output file: " << filepath << std::endl;
            return;
        }

        const char magic[8] = {'B', 'W', 'A', 'L', 'L', '\0', '\0', '\0'};
        uint32_t version = 1;
        file.write(magic, 8);
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&num_walls), sizeof(num_walls));
        file.write(reinterpret_cast<const char*>(&state_.current_step), sizeof(state_.current_step));
        file.write(reinterpret_cast<const char*>(&state_.current_time), sizeof(state_.current_time));

        for (int i = 0; i < num_walls; ++i) {
            file.write(reinterpret_cast<const char*>(&host_states[i].id), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&host_states[i].pos_x), 3 * sizeof(double));
            file.write(reinterpret_cast<const char*>(&host_states[i].vel_x), 3 * sizeof(double));
            file.write(reinterpret_cast<const char*>(&host_states[i].rot), 9 * sizeof(float));
            file.write(reinterpret_cast<const char*>(&host_states[i].omega_x), 3 * sizeof(float));
            file.write(reinterpret_cast<const char*>(&host_reactions[i].force_x), 3 * sizeof(double));
            file.write(reinterpret_cast<const char*>(&host_reactions[i].torque_x), 3 * sizeof(double));
        }
        file.close();
    }

    void save_clump_state() {
        if (clump_state_.num_clumps <= 0) return;

        // Get clump states from GPU
        ClumpStateSoA host_states;
        int32_t num_clumps = cudadem::clump_get_states(host_states, nullptr, clump_state_);

        if (num_clumps <= 0) {
            cudadem::free_clump_host_state(host_states, nullptr);
            return;
        }

        // Build output filename
        std::ostringstream oss;
        oss << config_.io.output_directory << "/clump_states_step_"
            << std::setfill('0') << std::setw(10) << state_.current_step << ".bin";
        std::string filepath = oss.str();

        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "WARNING: Cannot create clump output file: " << filepath << std::endl;
            cudadem::free_clump_host_state(host_states, nullptr);
            return;
        }

        // Version 2 is analysis-only and contains no generated-member template.
        const char magic[8] = {'C', 'L', 'U', 'M', 'P', '\0', '\0', '\0'};
        uint32_t version = 2;
        file.write(magic, 8);
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&num_clumps), sizeof(num_clumps));
        file.write(reinterpret_cast<const char*>(&state_.current_step), sizeof(state_.current_step));
        file.write(reinterpret_cast<const char*>(&state_.current_time), sizeof(state_.current_time));

        std::vector<double> gravity_fx,gravity_fy,gravity_fz,gravity_tx,gravity_ty,gravity_tz;
        if(config_.acceleration.enable_self_gravity&&config_.acceleration.gravity_update_rate>1) {
            gravity_fx.resize(num_clumps);gravity_fy.resize(num_clumps);gravity_fz.resize(num_clumps);
            gravity_tx.resize(num_clumps);gravity_ty.resize(num_clumps);gravity_tz.resize(num_clumps);
            const size_t bytes=static_cast<size_t>(num_clumps)*sizeof(double);
            CUDA_CHECK(cudaMemcpy(gravity_fx.data(),clump_state_.gravity_force_x,bytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gravity_fy.data(),clump_state_.gravity_force_y,bytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gravity_fz.data(),clump_state_.gravity_force_z,bytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gravity_tx.data(),clump_state_.gravity_torque_x,bytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gravity_ty.data(),clump_state_.gravity_torque_y,bytes,cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gravity_tz.data(),clump_state_.gravity_torque_z,bytes,cudaMemcpyDeviceToHost));
        }

        // Per clump: positive ID, CoM, velocity, quaternion, body omega,
        // double mass, six body-inertia values, p3, total force, total torque.
        for (int i = 0; i < num_clumps; ++i) {
            int32_t clump_id = i + 1;
            file.write(reinterpret_cast<const char*>(&clump_id), sizeof(int32_t));

            // CoM position
            double com[3] = {host_states.com_x[i], host_states.com_y[i], host_states.com_z[i]};
            file.write(reinterpret_cast<const char*>(com), 3 * sizeof(double));

            // Linear velocity
            double vel[3] = {host_states.vel_x[i], host_states.vel_y[i], host_states.vel_z[i]};
            file.write(reinterpret_cast<const char*>(vel), 3 * sizeof(double));

            // Orientation quaternion (w, x, y, z)
            double quat[4] = {host_states.quat_w[i], host_states.quat_x[i], host_states.quat_y[i], host_states.quat_z[i]};
            file.write(reinterpret_cast<const char*>(quat), 4 * sizeof(double));

            double omega[3] = {host_states.omega_x[i], host_states.omega_y[i], host_states.omega_z[i]};
            file.write(reinterpret_cast<const char*>(omega), sizeof(omega));
            file.write(reinterpret_cast<const char*>(&host_states.mass[i]), sizeof(double));
            double inertia[6] = {host_states.inertia_xx[i], host_states.inertia_yy[i], host_states.inertia_zz[i],
                                 host_states.inertia_xy[i], host_states.inertia_xz[i], host_states.inertia_yz[i]};
            file.write(reinterpret_cast<const char*>(inertia), sizeof(inertia));
            for (int q = 0; q < 19; ++q)
                file.write(reinterpret_cast<const char*>(&host_states.multipole_p3[q][i]), sizeof(double));

            // Net force
            double net_force[3] = {host_states.net_force_x[i], host_states.net_force_y[i], host_states.net_force_z[i]};
            if(!gravity_fx.empty()) {net_force[0]+=gravity_fx[i];net_force[1]+=gravity_fy[i];net_force[2]+=gravity_fz[i];}
            file.write(reinterpret_cast<const char*>(net_force), 3 * sizeof(double));

            // Net torque
            double net_torque[3] = {host_states.net_torque_x[i], host_states.net_torque_y[i], host_states.net_torque_z[i]};
            if(!gravity_tx.empty()) {net_torque[0]+=gravity_tx[i];net_torque[1]+=gravity_ty[i];net_torque[2]+=gravity_tz[i];}
            file.write(reinterpret_cast<const char*>(net_torque), 3 * sizeof(double));

        }

        file.close();
        cudadem::free_clump_host_state(host_states, nullptr);
        std::cout << "Clump states saved: " << filepath << std::endl;
    }

    void save_checkpoint() {
        // Call the full implementation from load_or_resume.h
        const std::string checkpoint_path =
            generate_checkpoint_path(config_, state_.current_step);
        adaptive_depth_io_.append_adaptive_fmm_depth_diagnostics();
        if (adaptive_fmm_depth_checkpoint_requested(config_)) {
            // Remove any controller state from an earlier write of this step
            // before replacing the particle binary.  The new binary is not a
            // published adaptive checkpoint until its new sidecar is renamed.
            for (const std::string& stale : {
                     adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path),
                     adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path) + ".tmp"}) {
                std::error_code ec;
                std::filesystem::remove(stale, ec);
                if (ec) {
                    throw std::runtime_error(
                        "Cannot remove stale adaptive FMM depth sidecar: " +
                        stale + ": " + ec.message());
                }
            }
        }
        const int32_t live_particles = state_.particles.num_particles;
        state_.particles.num_particles = num_total_particles_;
        cudadem::save_checkpoint(state_, config_);
        state_.particles.num_particles = live_particles;
        save_adaptive_yorp_checkpoint_sidecar(
            checkpoint_path);
        adaptive_depth_io_.save_adaptive_fmm_depth_checkpoint_sidecar(checkpoint_path);
        ExternalBodyManager::write_checkpoint_sidecar(checkpoint_path, state_, config_);
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    void shutdown() {
        if (!initialized_) return;

        std::cout << "\nShutting down simulation..." << std::endl;
        std::cout << "Optimized Contact Count Fallback Usage: " << fallback_counter_ << " steps" << std::endl;

        // All normal DEM resources below are released with CUDA APIs that may
        // require context-wide progress.  Stop the FMM scheduler first; it is
        // intentionally resident only while a submitted tree command waits
        // for its doorbell.  Draining the consumer stream also guarantees no
        // FMM input/output is still in flight.
        if (compute_stream_) {
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        }
        ExternalBodyManager::shutdown();
        AccelerationManager::shutdown();
        ShearBoxManager::shutdown();

        // Free kinetic energy buffer
        if (d_kinetic_energy_per_particle_) {
            cudaFree(d_kinetic_energy_per_particle_);
            d_kinetic_energy_per_particle_ = nullptr;
        }

        if (h_pinned_num_contacts_) {
            cudaFreeHost(h_pinned_num_contacts_);
            h_pinned_num_contacts_ = nullptr;
        }

        if (d_owner_offsets_) {
            cudaFree(d_owner_offsets_);
            d_owner_offsets_ = nullptr;
        }
        if (d_sort_keys_) {
            cudaFree(d_sort_keys_);
            d_sort_keys_ = nullptr;
        }
        if (d_sorted_sort_keys_) {
            cudaFree(d_sorted_sort_keys_);
            d_sorted_sort_keys_ = nullptr;
        }
        if (d_sort_indices_) {
            cudaFree(d_sort_indices_);
            d_sort_indices_ = nullptr;
        }
        if (d_sorted_sort_indices_) {
            cudaFree(d_sorted_sort_indices_);
            d_sorted_sort_indices_ = nullptr;
        }
        if (d_reversed_owner_offsets_) {
            cudaFree(d_reversed_owner_offsets_);
            d_reversed_owner_offsets_ = nullptr;
        }
        if (d_contact_sort_temp_) {
            cudaFree(d_contact_sort_temp_);
            d_contact_sort_temp_ = nullptr;
        }
        if (d_output_force_x_) { cudaFree(d_output_force_x_); d_output_force_x_ = nullptr; }
        if (d_output_force_y_) { cudaFree(d_output_force_y_); d_output_force_y_ = nullptr; }
        if (d_output_force_z_) { cudaFree(d_output_force_z_); d_output_force_z_ = nullptr; }

        // Free contact buffer
        NeighborSearch::free_contact_buffer(contacts_);

        // Shutdown subsystems
        SphereWallManager::shutdown();
        NeighborSearch::shutdown(state_);

        // Free clump resources
        free_clump_staging(clump_staging_);
        free_clump_segments(clump_segments_);
        free_clump_state(clump_state_);

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
    // Load application kernels eagerly before CUDA work begins.  This avoids a
    // lazy-load context synchronization racing the brief scheduler wait used
    // to hand an FMM command to its device-launched build graph.
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);

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

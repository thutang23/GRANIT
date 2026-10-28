#pragma once

#include "cuda_utils.h"
#include "../core/constant/system_initialize.h"
#include "../core/particle_model/particle.h"
#include <cuda_runtime.h>

namespace cudadem {

// One diagnostic per rank. flag is published only after all kernels complete.
struct ContactHistoryOverflow {
    int32_t flag = 0;
    int32_t owner_global_id = -1;
    int32_t partner_global_id = -1;
    int32_t stride = 0;
    int32_t required_count = 0;
};

__device__ inline void report_contact_history_overflow(ContactHistoryOverflow* error,
    int32_t owner, int32_t partner, int32_t stride, int32_t required) {
    if (atomicCAS(&error->flag, 0, 1) == 0) {
        error->owner_global_id = owner;
        error->partner_global_id = partner;
        error->stride = stride;
        error->required_count = required;
    }
}

__device__ inline int32_t reserve_contact_history_slot(int32_t* count, int32_t stride) {
    int32_t old = atomicAdd(count, 0);
    while (old < stride) {
        int32_t observed = atomicCAS(count, old, old + 1);
        if (observed == old) return old;
        old = observed;
    }
    return stride;
}

struct ContactHistoryWorkspace {
    int32_t* lookup_counts = nullptr;
    ContactHistoryOverflow* overflow = nullptr;
    int32_t capacity = 0;
    int allocation_device = -1;
    ContactHistoryWorkspace() = default;
    ContactHistoryWorkspace(const ContactHistoryWorkspace&) = delete;
    ContactHistoryWorkspace& operator=(const ContactHistoryWorkspace&) = delete;
    ~ContactHistoryWorkspace() {
        if (allocation_device < 0) return;
        int previous_device = -1;
        const bool have_previous_device = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool switch_device = !have_previous_device || previous_device != allocation_device;
        if (switch_device && cudaSetDevice(allocation_device) != cudaSuccess) return;
        cudaFree(lookup_counts);
        cudaFree(overflow);
        if (switch_device && have_previous_device) cudaSetDevice(previous_device);
    }
    void ensure_capacity(int32_t n) {
        if (allocation_device < 0)
            CUDA_CHECK(cudaGetDevice(&allocation_device));
        if (!overflow) CUDA_CHECK(cudaMalloc(&overflow, sizeof(ContactHistoryOverflow)));
        if (n > capacity) {
            int32_t* replacement = nullptr;
            CUDA_CHECK(cudaMalloc(&replacement, sizeof(int32_t) * n));
            CUDA_CHECK(cudaFree(lookup_counts));
            lookup_counts = replacement;
            capacity = n;
        }
    }
};

__global__ void clear_distributed_history_seen(ContactHistory* history,
    const int32_t* offsets, const int32_t* counts, int32_t stride, int32_t n) {
    int32_t owner = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner >= n) return;
    for (int32_t k = 0; k < counts[owner] && k < stride; ++k)
        history[offsets[owner] + k].seen_this_step = false;
}

// Identical position narrowing, canonical order, image transform and FP32
// overlap arithmetic to the force kernel and compute_contact_forces below.
// Both owner passes finish marking before any row is compacted.
template<bool FlipParticles>
__global__ void mark_distributed_history_contacts(const ContactPair* contacts, int32_t n_pairs,
    const double* px, const double* py, const double* pz, const float* radius,
    const int32_t* ids, ContactHistory* history, const int32_t* offsets,
    const int32_t* counts, int32_t stride, int32_t n_local,
    bool shear, float lx, float ly, float shift, bool canonical) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_pairs) return;
    const ContactPair& pair = contacts[index];
    int32_t owner = FlipParticles ? pair.j : pair.i;
    int32_t partner = FlipParticles ? pair.i : pair.j;
    if (owner < 0 || partner < 0 || owner >= n_local) return;
    bool reverse = canonical && ids[owner] > ids[partner];
    int32_t eo = reverse ? partner : owner, ep = reverse ? owner : partner;
    float ox = static_cast<float>(px[eo]), oy = static_cast<float>(py[eo]);
    float oz = static_cast<float>(pz[eo]);
    float x = static_cast<float>(px[ep]), y = static_cast<float>(py[ep]);
    float z = static_cast<float>(pz[ep]);
    if (shear) {
        bool reverse_image = FlipParticles != reverse;
        int ix = reverse_image ? -pair.image_x : pair.image_x;
        int iy = reverse_image ? -pair.image_y : pair.image_y;
        x += ix * lx;
        y += iy * ly - ix * shift;
    }
    float dx = x - ox, dy = y - oy, dz = z - oz;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    float dist = sqrtf(dist_sq);
    float delta_n = radius[eo] + radius[ep] - dist;
    if (!(delta_n > 0.0f)) return;
    for (int32_t k = 0; k < counts[owner] && k < stride; ++k) {
        ContactHistory& h = history[offsets[owner] + k];
        if (h.is_active && h.partner_global_id == ids[partner]) {
            h.seen_this_step = true;
            break;
        }
    }
}

__global__ void compact_distributed_contact_history(ContactHistory* history,
    const int32_t* offsets, int32_t* counts, int32_t* lookup_counts,
    int32_t stride, int32_t n) {
    int32_t owner = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner >= n) return;
    int32_t retained = 0;
    const int32_t old_count = counts[owner];
    for (int32_t k = 0; k < old_count && k < stride; ++k) {
        ContactHistory h = history[offsets[owner] + k];
        if (h.is_active && h.seen_this_step) {
            if (retained != k) history[offsets[owner] + retained] = h;
            ++retained;
        }
    }
    for (int32_t k = retained; k < old_count && k < stride; ++k) {
        history[offsets[owner] + k].is_active = false;
        history[offsets[owner] + k].seen_this_step = false;
    }
    counts[owner] = retained;
    lookup_counts[owner] = retained;
}

// Call after the final candidate list is ready, before either force pass.
// Ordered on stream: reset -> both geometry marks -> stable compaction. The
// frozen lookup prefix excludes new allocations throughout both force passes.
// Zero candidates still retires every old record. Springs of retained contacts
// are copied verbatim; no contact law integration occurs during preparation.
inline void prepare_distributed_contact_history(const ContactPair* pairs, int32_t n_pairs,
    SimulationState& state, int32_t n_local, ContactHistoryWorkspace& workspace,
    cudaStream_t stream, bool shear = false, float lx = 0, float ly = 0,
    float shift = 0, bool canonical = false) {
    workspace.ensure_capacity(n_local);
    CUDA_CHECK(cudaMemsetAsync(workspace.overflow, 0, sizeof(ContactHistoryOverflow), stream));
    if (n_local == 0) return;
    const int blocks = (n_local + 255) / 256;
    auto& p = state.particles;
    auto& h = state.contact_history;
    clear_distributed_history_seen<<<blocks, 256, 0, stream>>>(h.data,
        p.contact_history_offset, p.num_contacts, h.max_contacts_per_particle, n_local);
    if (n_pairs > 0) {
        const int pair_blocks = (n_pairs + 255) / 256;
        mark_distributed_history_contacts<false><<<pair_blocks, 256, 0, stream>>>(pairs,
            n_pairs, p.pos_x, p.pos_y, p.pos_z, p.radius, p.global_id, h.data,
            p.contact_history_offset, p.num_contacts, h.max_contacts_per_particle,
            n_local, shear, lx, ly, shift, canonical);
        mark_distributed_history_contacts<true><<<pair_blocks, 256, 0, stream>>>(pairs,
            n_pairs, p.pos_x, p.pos_y, p.pos_z, p.radius, p.global_id, h.data,
            p.contact_history_offset, p.num_contacts, h.max_contacts_per_particle,
            n_local, shear, lx, ly, shift, canonical);
    }
    compact_distributed_contact_history<<<blocks, 256, 0, stream>>>(h.data,
        p.contact_history_offset, p.num_contacts, workspace.lookup_counts,
        h.max_contacts_per_particle, n_local);
    CUDA_CHECK(cudaGetLastError());
}

template<bool FlipParticles>
__global__ void compute_contact_forces_multigpu_kernel(
    const ContactPair* __restrict__ contacts,
    int32_t num_contacts,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const float* __restrict__ radius,
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const DemInteractionProperties* __restrict__ interactions,
    uint32_t material_count,
    const int32_t* __restrict__ global_id,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    ContactHistory* __restrict__ contact_history,
    const int32_t* __restrict__ contact_history_offset,
    int32_t* __restrict__ num_contacts_per_particle,
    int32_t max_contacts_per_particle,
    const int32_t* __restrict__ lookup_counts,
    ContactHistoryOverflow* overflow,
    bool use_visco,
    float dt,
    int32_t num_local_particles,  // particles with index >= this are ghosts
    bool shear_enabled = false,
    float shear_length_x = 0.0f, float shear_length_y = 0.0f,
    float shear_reduced_shift = 0.0f, float shear_velocity = 0.0f,
    bool canonical_pair_evaluation = false
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    const ContactPair& contact = contacts[idx];

    // Select owner/partner based on template parameter
    int32_t owner_idx, partner_idx;
    if constexpr (!FlipParticles) {
        owner_idx = contact.i;
        partner_idx = contact.j;
    } else {
        owner_idx = contact.j;
        partner_idx = contact.i;
    }

    if (owner_idx < 0 || partner_idx < 0) return;

    // CRITICAL: Skip if owner is a ghost particle (only compute forces for locals)
    if (owner_idx >= num_local_particles) return;

    // A clump scene evaluates both directed owner passes in the same stable
    // global-ID order. Recomputing the reversed pair changes the FP32 contact
    // arms Ap/An by ULPs and loses exact Newton force/torque correspondence.
    // The directed history still belongs to owner_idx and migrates with it.
    const bool reverse_evaluation = canonical_pair_evaluation &&
        global_id[owner_idx] > global_id[partner_idx];
    const int32_t evaluation_owner = reverse_evaluation ? partner_idx : owner_idx;
    const int32_t evaluation_partner = reverse_evaluation ? owner_idx : partner_idx;

    // Load owner particle properties
    float po_x = static_cast<float>(pos_x[evaluation_owner]);
    float po_y = static_cast<float>(pos_y[evaluation_owner]);
    float po_z = static_cast<float>(pos_z[evaluation_owner]);
    float vo_x = vel_x[evaluation_owner], vo_y = vel_y[evaluation_owner], vo_z = vel_z[evaluation_owner];
    float wo_x = omega_x[evaluation_owner], wo_y = omega_y[evaluation_owner], wo_z = omega_z[evaluation_owner];
    float R_o = radius[evaluation_owner];
    float m_o = mass[evaluation_owner];
    float I_o = moment_of_inertia[evaluation_owner];
    const int32_t material_o = material_id[evaluation_owner];
    if (material_o < 0 || static_cast<uint32_t>(material_o) >= material_count) return;

    // Load partner particle properties
    float pp_x = static_cast<float>(pos_x[evaluation_partner]);
    float pp_y = static_cast<float>(pos_y[evaluation_partner]);
    float pp_z = static_cast<float>(pos_z[evaluation_partner]);
    float vp_x = vel_x[evaluation_partner], vp_y = vel_y[evaluation_partner], vp_z = vel_z[evaluation_partner];
    if (shear_enabled) {
        // The pair stores the image of j relative to i.  The second pass
        // changes owner and therefore needs the inverse lattice transform.
        const bool reverse_image=FlipParticles != reverse_evaluation;
        const int ix = reverse_image ? -contact.image_x : contact.image_x;
        const int iy = reverse_image ? -contact.image_y : contact.image_y;
        pp_x += ix * shear_length_x;
        pp_y += iy * shear_length_y - ix * shear_reduced_shift;
        vp_y -= ix * shear_velocity;
    }
    float wp_x = omega_x[evaluation_partner], wp_y = omega_y[evaluation_partner], wp_z = omega_z[evaluation_partner];
    float R_p = radius[evaluation_partner];
    float m_p = mass[evaluation_partner];
    float I_p = moment_of_inertia[evaluation_partner];
    const int32_t material_p = material_id[evaluation_partner];
    if (material_p < 0 || static_cast<uint32_t>(material_p) >= material_count) return;
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

    // Contact history lookup
    float H_t_x = 0.0f, H_t_y = 0.0f, H_t_z = 0.0f;
    float H_r_x = 0.0f, H_r_y = 0.0f, H_r_z = 0.0f;
    float H_tw_x = 0.0f, H_tw_y = 0.0f, H_tw_z = 0.0f;

    int32_t slot = -1;
    int32_t gid_partner = global_id[partner_idx];
    int32_t hist_offset = contact_history_offset[owner_idx];
    int32_t num_hist = lookup_counts[owner_idx];

    for (int32_t k = 0; k < num_hist && k < max_contacts_per_particle; ++k) {
        const ContactHistory& hist = contact_history[hist_offset + k];
        if (hist.partner_global_id == gid_partner && hist.is_active) {
            H_t_x = hist.delta_t_x;
            H_t_y = hist.delta_t_y;
            H_t_z = hist.delta_t_z;
            H_r_x = hist.delta_r_x;
            H_r_y = hist.delta_r_y;
            H_r_z = hist.delta_r_z;
            H_tw_x = hist.delta_tw_x;
            H_tw_y = hist.delta_tw_y;
            H_tw_z = hist.delta_tw_z;
            slot = k;
            break;
        }
    }

    // Tangential spring force, rolling moment, and twisting moment histories
    // are world-frame vectors. Swapping the two particles reverses all three;
    // the tangent plane itself is unchanged under n -> -n.
    if (reverse_evaluation) {
        H_t_x=-H_t_x; H_t_y=-H_t_y; H_t_z=-H_t_z;
        H_r_x=-H_r_x; H_r_y=-H_r_y; H_r_z=-H_r_z;
        H_tw_x=-H_tw_x; H_tw_y=-H_tw_y; H_tw_z=-H_tw_z;
    }

    // Compute contact forces
    ContactForceResult result = compute_contact_forces(
        po_x, po_y, po_z, vo_x, vo_y, vo_z, wo_x, wo_y, wo_z,
        R_o, m_o, I_o,
        pp_x, pp_y, pp_z, vp_x, vp_y, vp_z, wp_x, wp_y, wp_z,
        R_p, m_p, I_p,
        H_t_x, H_t_y, H_t_z, H_r_x, H_r_y, H_r_z, H_tw_x, H_tw_y, H_tw_z,
        E_star, G_star, pair.restitution,
        pair.friction_static, pair.friction_rolling, pair.friction_twisting,
        pair.shape_parameter,
        pair.cohesion_enabled ? pair.cohesion_cutoff_factor : 1.0f,
        use_visco, pair.viscous_coefficient,
        pair.cohesion_enabled != 0, pair.cohesion_strength, dt
    );

    if (reverse_evaluation) {
        // Select the canonical partner's physical force and torque. Its arm
        // is An, which is deliberately computed alongside Ap in one call.
        result.force_x=-result.force_x;
        result.force_y=-result.force_y;
        result.force_z=-result.force_z;
        result.torque_x=result.torque_j_x;
        result.torque_y=result.torque_j_y;
        result.torque_z=result.torque_j_z;
        result.delta_t_x=-result.delta_t_x;
        result.delta_t_y=-result.delta_t_y;
        result.delta_t_z=-result.delta_t_z;
        result.delta_r_x=-result.delta_r_x;
        result.delta_r_y=-result.delta_r_y;
        result.delta_r_z=-result.delta_r_z;
        result.delta_tw_x=-result.delta_tw_x;
        result.delta_tw_y=-result.delta_tw_y;
        result.delta_tw_z=-result.delta_tw_z;
    }

    // Update Contact History for the owner particle
    // =========================================================================
    // Each particle has its own independent history buffer.
    // Both passes must update their respective owner's history.
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
            int32_t new_slot = reserve_contact_history_slot(
                &num_contacts_per_particle[owner_idx], max_contacts_per_particle);
            if (new_slot < max_contacts_per_particle) {
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
                hist.seen_this_step = true;
            } else {
                report_contact_history_overflow(overflow, global_id[owner_idx],
                    gid_partner, max_contacts_per_particle, max_contacts_per_particle + 1);
            }
        }
    } else if (slot >= 0) {
        contact_history[hist_offset + slot].is_active = false;
    }

    // Apply forces and torques to owner particle
    // compute_contact_forces(owner, partner, ...) returns force AND torque
    // on the owner particle. Both passes must apply their respective torques.
    // Canonical evaluation selects the appropriate side explicitly; the
    // established unclumped path retains its directed owner evaluation.
    atomicAdd(&force_x[owner_idx], result.force_x);
    atomicAdd(&force_y[owner_idx], result.force_y);
    atomicAdd(&force_z[owner_idx], result.force_z);
    atomicAdd(&torque_x[owner_idx], result.torque_x);
    atomicAdd(&torque_y[owner_idx], result.torque_y);
    atomicAdd(&torque_z[owner_idx], result.torque_z);
}


} // namespace cudadem

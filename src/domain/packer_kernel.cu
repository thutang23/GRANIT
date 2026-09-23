#include "communicator.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

namespace cudadem {

// =============================================================================
// Pack Particles Kernel (SoA → AoS)
// =============================================================================

__global__ void pack_particles_kernel(
    PackedParticle* __restrict__ buffer,
    const int32_t* __restrict__ indices_to_pack,
    int32_t num_to_pack,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const double* __restrict__ quat_w,
    const double* __restrict__ quat_x,
    const double* __restrict__ quat_y,
    const double* __restrict__ quat_z,
    const float* __restrict__ radius,
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    const float* __restrict__ cohesive_strength,
    const int32_t* __restrict__ global_id,
    const int32_t* __restrict__ material_id,
    int32_t source_gpu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_pack) return;

    int32_t p_idx = indices_to_pack[idx];

    PackedParticle p{};
    p.pos_x = pos_x[p_idx];
    p.pos_y = pos_y[p_idx];
    p.pos_z = pos_z[p_idx];
    p.vel_x = vel_x[p_idx];
    p.vel_y = vel_y[p_idx];
    p.vel_z = vel_z[p_idx];
    p.omega_x = omega_x[p_idx];
    p.omega_y = omega_y[p_idx];
    p.omega_z = omega_z[p_idx];
    p.quat_w = quat_w[p_idx];
    p.quat_x = quat_x[p_idx];
    p.quat_y = quat_y[p_idx];
    p.quat_z = quat_z[p_idx];
    p.radius = radius[p_idx];
    p.mass = mass[p_idx];
    p.moment_of_inertia = moment_of_inertia[p_idx];
    p.cohesive_strength = cohesive_strength[p_idx];
    p.global_id = global_id[p_idx];
    p.material_id = material_id[p_idx];
    p.source_gpu = source_gpu;
    p.num_active_contacts = 0;  // Will be set separately if migrating
    p.num_active_wall_contacts = 0;
    p.contact_payload_offset = 0;
    p.wall_payload_offset = 0;

    buffer[idx] = p;
}

// =============================================================================
// Unpack Ghost Particles Kernel (AoS → SoA, append to ghost region)
// =============================================================================

__global__ void unpack_ghost_particles_kernel(
    const PackedParticle* __restrict__ buffer,
    int32_t num_received,
    int32_t ghost_start_offset,
    ParticleSoA particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_received) return;

    int32_t target_idx = ghost_start_offset + idx;
    const PackedParticle& p = buffer[idx];

    particles.pos_x[target_idx] = p.pos_x; particles.pos_y[target_idx] = p.pos_y; particles.pos_z[target_idx] = p.pos_z;
    particles.gravity_force_x[target_idx] = p.gravity_force_x; particles.gravity_force_y[target_idx] = p.gravity_force_y; particles.gravity_force_z[target_idx] = p.gravity_force_z;
    particles.last_pos_x[target_idx] = p.last_pos_x; particles.last_pos_y[target_idx] = p.last_pos_y; particles.last_pos_z[target_idx] = p.last_pos_z;
    particles.winding_x[target_idx] = p.winding_x;
    particles.winding_y[target_idx] = p.winding_y;
    particles.winding_z[target_idx] = p.winding_z;
    particles.vel_x[target_idx] = p.vel_x; particles.vel_y[target_idx] = p.vel_y; particles.vel_z[target_idx] = p.vel_z;
    particles.omega_x[target_idx] = p.omega_x; particles.omega_y[target_idx] = p.omega_y; particles.omega_z[target_idx] = p.omega_z;
    particles.quat_w[target_idx] = p.quat_w; particles.quat_x[target_idx] = p.quat_x; particles.quat_y[target_idx] = p.quat_y; particles.quat_z[target_idx] = p.quat_z;
    particles.radius[target_idx] = p.radius; particles.mass[target_idx] = p.mass; particles.moment_of_inertia[target_idx] = p.moment_of_inertia;
    particles.cohesive_strength[target_idx] = p.cohesive_strength;
    particles.support_extent[target_idx] = p.support_extent;
    particles.density[target_idx] = p.density; particles.pressure[target_idx] = p.pressure; particles.viscosity[target_idx] = p.viscosity;
    particles.c_sound[target_idx] = p.c_sound; particles.drhodt[target_idx] = p.drhodt; particles.rho_rate[target_idx] = p.rho_rate;
    particles.smoothing_length[target_idx] = p.smoothing_length; particles.energy[target_idx] = p.energy; particles.dedt[target_idx] = p.dedt;
    particles.energy_cold[target_idx] = p.energy_cold;
    particles.energy_thermal_peak[target_idx] = p.energy_thermal_peak;
    particles.pressure_peak[target_idx] = p.pressure_peak;
    particles.fracture_area[target_idx] = p.fracture_area;
    particles.fragment_length[target_idx] = p.fragment_length;
    particles.damping_energy[target_idx] = p.damping_energy;

    particles.S_xx[target_idx] = p.S_xx; particles.S_xy[target_idx] = p.S_xy; particles.S_xz[target_idx] = p.S_xz;
    particles.S_yy[target_idx] = p.S_yy; particles.S_yz[target_idx] = p.S_yz; particles.S_zz[target_idx] = p.S_zz;
    particles.damage[target_idx] = p.damage; particles.crack_strain[target_idx] = p.crack_strain; particles.flaws_ntot[target_idx] = p.flaws_ntot;
    particles.flaws_strain_min[target_idx] = p.flaws_strain_min; particles.flaws_strain_max[target_idx] = p.flaws_strain_max; particles.flaws_m0[target_idx] = p.flaws_m0;
    particles.distension[target_idx] = p.distension;
    particles.L_00[target_idx] = p.L_00; particles.L_01[target_idx] = p.L_01; particles.L_02[target_idx] = p.L_02;
    particles.L_10[target_idx] = p.L_10; particles.L_11[target_idx] = p.L_11; particles.L_12[target_idx] = p.L_12;
    particles.L_20[target_idx] = p.L_20; particles.L_21[target_idx] = p.L_21; particles.L_22[target_idx] = p.L_22;
    particles.global_id[target_idx] = p.global_id; particles.material_id[target_idx] = p.material_id;
    particles.clump_id[target_idx] = p.clump_id; particles.clump_off_x[target_idx] = p.clump_off_x;
    particles.clump_off_y[target_idx] = p.clump_off_y; particles.clump_off_z[target_idx] = p.clump_off_z;
}

// =============================================================================
// Pack Contact History Kernel (local ID → global ID)
// =============================================================================

__global__ void pack_contact_history_kernel(
    ContactHistoryPacked* __restrict__ packed_history,
    int32_t* __restrict__ num_packed,
    const ContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ migrate_indices,
    int32_t num_migrating,
    const int32_t* __restrict__ local_to_global,
    int32_t max_contacts_per_particle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_migrating) return;

    int32_t local_idx = migrate_indices[idx];
    int32_t global_id = local_to_global[local_idx];
    int32_t offset = history_offset[local_idx];
    int32_t n_contacts = num_contacts[local_idx];

    for (int c = 0; c < n_contacts && c < max_contacts_per_particle; ++c) {
        const ContactHistory& h = history_buffer[offset + c];
        if (!h.is_active) continue;

        // Allocate slot in output using atomic
        int32_t out_idx = atomicAdd(num_packed, 1);

        ContactHistoryPacked& ph = packed_history[out_idx];
        ph.particle_global_id = global_id;
        ph.partner_global_id = h.partner_global_id;
        ph.delta_t_x = h.delta_t_x;
        ph.delta_t_y = h.delta_t_y;
        ph.delta_t_z = h.delta_t_z;
        ph.delta_r_x = h.delta_r_x;
        ph.delta_r_y = h.delta_r_y;
        ph.delta_r_z = h.delta_r_z;
        ph.delta_tw_x = h.delta_tw_x;
        ph.delta_tw_y = h.delta_tw_y;
        ph.delta_tw_z = h.delta_tw_z;
        ph.is_active = true;
    }
}

// =============================================================================
// Unpack Contact History Kernel (global ID → local ID via lookup)
// =============================================================================

// Device function for binary search in sorted global_to_local array
__device__ int32_t binary_search_global_to_local(
    int32_t global_id,
    const int32_t* global_to_local,
    int32_t size
) {
    // Simplified: assume global_to_local is indexed by global_id
    // In production, use hash table or actual binary search
    if (global_id >= 0 && global_id < size) {
        return global_to_local[global_id];
    }
    return -1;
}

__global__ void unpack_contact_history_kernel(
    const ContactHistoryPacked* __restrict__ packed_history,
    int32_t num_packed,
    ContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ global_to_local,
    int32_t global_to_local_size,
    int32_t max_contacts_per_particle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packed) return;

    const ContactHistoryPacked& ph = packed_history[idx];

    // Find local index of owner particle
    int32_t owner_local = binary_search_global_to_local(
        ph.particle_global_id, global_to_local, global_to_local_size);
    if (owner_local < 0) return;

    // Find local index of partner
    int32_t partner_local = binary_search_global_to_local(
        ph.partner_global_id, global_to_local, global_to_local_size);

    // Allocate history slot
    int32_t slot = atomicAdd(&num_contacts[owner_local], 1);
    if (slot >= max_contacts_per_particle) return;

    int32_t offset = history_offset[owner_local];
    ContactHistory& h = history_buffer[offset + slot];
    h.partner_global_id = ph.partner_global_id;
    h.partner_local_id = partner_local;
    h.delta_t_x = ph.delta_t_x;
    h.delta_t_y = ph.delta_t_y;
    h.delta_t_z = ph.delta_t_z;
    h.delta_r_x = ph.delta_r_x;
    h.delta_r_y = ph.delta_r_y;
    h.delta_r_z = ph.delta_r_z;
        h.delta_tw_x = ph.delta_tw_x;
        h.delta_tw_y = ph.delta_tw_y;
        h.delta_tw_z = ph.delta_tw_z;
    h.is_active = true;
}

// =============================================================================
// Fixed-stride migration history kernels
// =============================================================================

__device__ inline ContactHistoryPacked inactive_contact_history_packed() {
    ContactHistoryPacked h{};
    h.particle_global_id = -1;
    h.partner_global_id = -1;
    h.is_active = false;
    return h;
}

__device__ inline WallContactHistoryPacked inactive_wall_history_packed() {
    WallContactHistoryPacked h{};
    h.wall_id = -1;
    h.is_active = false;
    h.is_slipping = false;
    h.is_rolling = false;
    h.is_twisting = false;
    return h;
}

__global__ void validate_migration_history_counts_kernel(
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ num_wall_contacts,
    int32_t max_contacts_per_particle,
    int32_t max_wall_contacts_per_particle,
    int32_t* __restrict__ overflow_status
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const int32_t particle = particle_indices ? particle_indices[idx] : idx;
    if (num_contacts[particle] < 0 || num_contacts[particle] > max_contacts_per_particle)
        atomicCAS(overflow_status, 0, 1);
    if (num_wall_contacts[particle] < 0 || num_wall_contacts[particle] > max_wall_contacts_per_particle)
        atomicCAS(overflow_status, 0, 2);
}

__global__ void pack_contact_history_fixed_stride_kernel(
    ContactHistoryPacked* __restrict__ packed_history,
    PackedParticle* __restrict__ packed_particles,
    const ContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    const int32_t* __restrict__ particle_global_id,
    int32_t max_contacts_per_particle
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const int32_t particle = particle_indices[idx];
    ContactHistoryPacked* out = packed_history + static_cast<size_t>(idx) * max_contacts_per_particle;
    int32_t dense = 0;
    const int32_t offset = history_offset[particle];
    const int32_t count = num_contacts[particle];
    for (int32_t slot = 0; slot < count; ++slot) {
        const ContactHistory& source = history_buffer[offset + slot];
        if (!source.is_active) continue;
        ContactHistoryPacked& target = out[dense++];
        target.particle_global_id = particle_global_id[particle];
        target.partner_global_id = source.partner_global_id;
        target.delta_t_x = source.delta_t_x; target.delta_t_y = source.delta_t_y; target.delta_t_z = source.delta_t_z;
        target.delta_r_x = source.delta_r_x; target.delta_r_y = source.delta_r_y; target.delta_r_z = source.delta_r_z;
        target.delta_tw_x = source.delta_tw_x; target.delta_tw_y = source.delta_tw_y; target.delta_tw_z = source.delta_tw_z;
        target.is_active = true;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        out[slot] = inactive_contact_history_packed();
    packed_particles[idx].num_active_contacts = dense;
}

__global__ void pack_wall_contact_history_fixed_stride_kernel(
    WallContactHistoryPacked* __restrict__ packed_history,
    PackedParticle* __restrict__ packed_particles,
    const WallContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    int32_t max_contacts_per_particle
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const int32_t particle = particle_indices[idx];
    WallContactHistoryPacked* out = packed_history + static_cast<size_t>(idx) * max_contacts_per_particle;
    int32_t dense = 0;
    const int32_t offset = history_offset[particle];
    const int32_t count = num_contacts[particle];
    for (int32_t slot = 0; slot < count; ++slot) {
        const WallContactHistory& source = history_buffer[offset + slot];
        if (!source.is_active) continue;
        WallContactHistoryPacked& target = out[dense++];
        target.wall_id = source.wall_id;
        target.delta_t_x = source.delta_t_x; target.delta_t_y = source.delta_t_y; target.delta_t_z = source.delta_t_z;
        target.delta_r_x = source.delta_r_x; target.delta_r_y = source.delta_r_y; target.delta_r_z = source.delta_r_z;
        target.delta_tw_x = source.delta_tw_x; target.delta_tw_y = source.delta_tw_y; target.delta_tw_z = source.delta_tw_z;
        target.is_active = true;
        target.is_slipping = source.is_slipping;
        target.is_rolling = source.is_rolling;
        target.is_twisting = source.is_twisting;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        out[slot] = inactive_wall_history_packed();
    packed_particles[idx].num_active_wall_contacts = dense;
}

__global__ void unpack_contact_history_fixed_stride_kernel(
    const ContactHistoryPacked* __restrict__ packed_history,
    int32_t num_particles,
    int32_t first_target_particle,
    ContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts_per_particle
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const int32_t target_particle = first_target_particle + idx;
    const ContactHistoryPacked* source = packed_history + static_cast<size_t>(idx) * max_contacts_per_particle;
    ContactHistory* target = history_buffer + static_cast<size_t>(target_particle) * max_contacts_per_particle;
    int32_t dense = 0;
    for (int32_t slot = 0; slot < max_contacts_per_particle; ++slot) {
        if (!source[slot].is_active) continue;
        const ContactHistoryPacked& in = source[slot];
        ContactHistory& out = target[dense++];
        out.partner_global_id = in.partner_global_id;
        // Local particle indices are GPU-local and invalid after migration.
        out.partner_local_id = -1;
        out.delta_t_x = in.delta_t_x; out.delta_t_y = in.delta_t_y; out.delta_t_z = in.delta_t_z;
        out.delta_r_x = in.delta_r_x; out.delta_r_y = in.delta_r_y; out.delta_r_z = in.delta_r_z;
        out.delta_tw_x = in.delta_tw_x; out.delta_tw_y = in.delta_tw_y; out.delta_tw_z = in.delta_tw_z;
        out.is_active = true;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        target[slot] = ContactHistory();
    history_offset[target_particle] = target_particle * max_contacts_per_particle;
    num_contacts[target_particle] = dense;
}

__global__ void unpack_wall_contact_history_fixed_stride_kernel(
    const WallContactHistoryPacked* __restrict__ packed_history,
    int32_t num_particles,
    int32_t first_target_particle,
    WallContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts_per_particle
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    const int32_t target_particle = first_target_particle + idx;
    const WallContactHistoryPacked* source = packed_history + static_cast<size_t>(idx) * max_contacts_per_particle;
    WallContactHistory* target = history_buffer + static_cast<size_t>(target_particle) * max_contacts_per_particle;
    int32_t dense = 0;
    for (int32_t slot = 0; slot < max_contacts_per_particle; ++slot) {
        if (!source[slot].is_active) continue;
        const WallContactHistoryPacked& in = source[slot];
        WallContactHistory& out = target[dense++];
        out.wall_id = in.wall_id;
        out.delta_t_x = in.delta_t_x; out.delta_t_y = in.delta_t_y; out.delta_t_z = in.delta_t_z;
        out.delta_r_x = in.delta_r_x; out.delta_r_y = in.delta_r_y; out.delta_r_z = in.delta_r_z;
        out.delta_tw_x = in.delta_tw_x; out.delta_tw_y = in.delta_tw_y; out.delta_tw_z = in.delta_tw_z;
        out.is_active = true;
        out.is_slipping = in.is_slipping; out.is_rolling = in.is_rolling; out.is_twisting = in.is_twisting;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        target[slot] = WallContactHistory();
    history_offset[target_particle] = target_particle * max_contacts_per_particle;
    num_contacts[target_particle] = dense;
}

__global__ void compact_contact_history_kernel(
    const int32_t* __restrict__ keep_flags,
    const int32_t* __restrict__ new_indices,
    int32_t num_particles,
    const ContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    ContactHistory* __restrict__ compacted_history,
    int32_t* __restrict__ compacted_offset,
    int32_t* __restrict__ compacted_count,
    int32_t max_contacts_per_particle
) {
    const int32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= num_particles || !keep_flags[particle]) return;
    const int32_t new_particle = new_indices[particle];
    const ContactHistory* source = history_buffer + history_offset[particle];
    ContactHistory* target = compacted_history + static_cast<size_t>(new_particle) * max_contacts_per_particle;
    int32_t dense = 0;
    for (int32_t slot = 0; slot < num_contacts[particle] && slot < max_contacts_per_particle; ++slot) {
        const ContactHistory& in = source[slot];
        if (!in.is_active) continue;
        ContactHistory& out = target[dense++];
        out = in;
        out.partner_local_id = -1;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        target[slot] = ContactHistory();
    compacted_offset[new_particle] = new_particle * max_contacts_per_particle;
    compacted_count[new_particle] = dense;
}

__global__ void compact_wall_contact_history_kernel(
    const int32_t* __restrict__ keep_flags,
    const int32_t* __restrict__ new_indices,
    int32_t num_particles,
    const WallContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    WallContactHistory* __restrict__ compacted_history,
    int32_t* __restrict__ compacted_offset,
    int32_t* __restrict__ compacted_count,
    int32_t max_contacts_per_particle
) {
    const int32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= num_particles || !keep_flags[particle]) return;
    const int32_t new_particle = new_indices[particle];
    const WallContactHistory* source = history_buffer + history_offset[particle];
    WallContactHistory* target = compacted_history + static_cast<size_t>(new_particle) * max_contacts_per_particle;
    int32_t dense = 0;
    for (int32_t slot = 0; slot < num_contacts[particle] && slot < max_contacts_per_particle; ++slot) {
        const WallContactHistory& in = source[slot];
        if (!in.is_active) continue;
        target[dense++] = in;
    }
    for (int32_t slot = dense; slot < max_contacts_per_particle; ++slot)
        target[slot] = WallContactHistory();
    if (compacted_offset) compacted_offset[new_particle] = new_particle * max_contacts_per_particle;
    compacted_count[new_particle] = dense;
}

__global__ void finalize_compacted_wall_history_metadata_kernel(
    int32_t num_particles,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ compacted_count,
    int32_t max_contacts_per_particle
) {
    const int32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= num_particles) return;
    history_offset[particle] = particle * max_contacts_per_particle;
    num_contacts[particle] = compacted_count[particle];
}

// =============================================================================
// Compact Particles Kernel (stream compaction)
// =============================================================================

__global__ void compact_particles_kernel(
    const int32_t* __restrict__ keep_flags,
    const int32_t* __restrict__ new_indices,
    int32_t num_particles,
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    double* __restrict__ quat_w,
    double* __restrict__ quat_x,
    double* __restrict__ quat_y,
    double* __restrict__ quat_z,
    float* __restrict__ radius,
    float* __restrict__ mass,
    float* __restrict__ moment_of_inertia,
    float* __restrict__ cohesive_strength,
    int32_t* __restrict__ global_id,
    int32_t* __restrict__ contact_history_offset,
    int32_t* __restrict__ num_contacts,
    double* __restrict__ last_pos_x,
    double* __restrict__ last_pos_y,
    double* __restrict__ last_pos_z,
    double* __restrict__ temp_pos_x,
    double* __restrict__ temp_pos_y,
    double* __restrict__ temp_pos_z,
    double* __restrict__ temp_last_pos_x,
    double* __restrict__ temp_last_pos_y,
    double* __restrict__ temp_last_pos_z,
    double* __restrict__ temp_vel_x,
    double* __restrict__ temp_vel_y,
    double* __restrict__ temp_vel_z,
    double* __restrict__ temp_omega_x,
    double* __restrict__ temp_omega_y,
    double* __restrict__ temp_omega_z,
    double* __restrict__ temp_quat_w,
    double* __restrict__ temp_quat_x,
    double* __restrict__ temp_quat_y,
    double* __restrict__ temp_quat_z,
    float* __restrict__ temp_radius,
    float* __restrict__ temp_mass,
    float* __restrict__ temp_moi,
    float* __restrict__ temp_cohesive,
    int32_t* __restrict__ temp_global_id,
    int32_t* __restrict__ temp_contact_offset,
    int32_t* __restrict__ temp_num_contacts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (keep_flags[idx]) {
        int32_t new_idx = new_indices[idx];

        temp_pos_x[new_idx] = pos_x[idx];
        temp_pos_y[new_idx] = pos_y[idx];
        temp_pos_z[new_idx] = pos_z[idx];
        temp_last_pos_x[new_idx] = last_pos_x[idx];
        temp_last_pos_y[new_idx] = last_pos_y[idx];
        temp_last_pos_z[new_idx] = last_pos_z[idx];

        temp_vel_x[new_idx] = vel_x[idx];
        temp_vel_y[new_idx] = vel_y[idx];
        temp_vel_z[new_idx] = vel_z[idx];
        temp_omega_x[new_idx] = omega_x[idx];
        temp_omega_y[new_idx] = omega_y[idx];
        temp_omega_z[new_idx] = omega_z[idx];
        temp_quat_w[new_idx] = quat_w[idx];
        temp_quat_x[new_idx] = quat_x[idx];
        temp_quat_y[new_idx] = quat_y[idx];
        temp_quat_z[new_idx] = quat_z[idx];
        temp_radius[new_idx] = radius[idx];
        temp_mass[new_idx] = mass[idx];
        temp_moi[new_idx] = moment_of_inertia[idx];
        temp_cohesive[new_idx] = cohesive_strength[idx];
        temp_global_id[new_idx] = global_id[idx];
        temp_contact_offset[new_idx] = contact_history_offset[idx];
        temp_num_contacts[new_idx] = num_contacts[idx];
    }
}

// =============================================================================
// Direct Scatter Pack Kernels
// =============================================================================

__device__ inline PackedParticle make_packed_particle(
    int32_t particle_idx,
    ParticlePackSource source,
    int32_t source_gpu
) {
    PackedParticle p{};
    p.pos_x = source.pos_x[particle_idx];
    p.pos_y = source.pos_y[particle_idx];
    p.pos_z = source.pos_z[particle_idx];
    p.gravity_force_x = source.gravity_force_x ? source.gravity_force_x[particle_idx] : 0;
    p.gravity_force_y = source.gravity_force_y ? source.gravity_force_y[particle_idx] : 0;
    p.gravity_force_z = source.gravity_force_z ? source.gravity_force_z[particle_idx] : 0;
    p.last_pos_x = source.last_pos_x[particle_idx]; p.last_pos_y = source.last_pos_y[particle_idx]; p.last_pos_z = source.last_pos_z[particle_idx];
    p.winding_x = source.winding_x ? source.winding_x[particle_idx] : 0;
    p.winding_y = source.winding_y ? source.winding_y[particle_idx] : 0;
    p.winding_z = source.winding_z ? source.winding_z[particle_idx] : 0;
    p.vel_x = source.vel_x[particle_idx];
    p.vel_y = source.vel_y[particle_idx];
    p.vel_z = source.vel_z[particle_idx];
    p.omega_x = source.omega_x[particle_idx];
    p.omega_y = source.omega_y[particle_idx];
    p.omega_z = source.omega_z[particle_idx];
    p.quat_w = source.quat_w[particle_idx];
    p.quat_x = source.quat_x[particle_idx];
    p.quat_y = source.quat_y[particle_idx];
    p.quat_z = source.quat_z[particle_idx];
    p.radius = source.radius[particle_idx];
    p.mass = source.mass[particle_idx];
    p.moment_of_inertia = source.moment_of_inertia[particle_idx];
    p.cohesive_strength = source.cohesive_strength[particle_idx];
    p.support_extent = source.support_extent[particle_idx];
    p.density = source.density[particle_idx]; p.pressure = source.pressure[particle_idx]; p.viscosity = source.viscosity[particle_idx];
    p.c_sound = source.c_sound[particle_idx]; p.drhodt = source.drhodt[particle_idx]; p.rho_rate = source.rho_rate[particle_idx];
    p.smoothing_length = source.smoothing_length[particle_idx]; p.energy = source.energy[particle_idx]; p.dedt = source.dedt[particle_idx];
    p.energy_cold = source.energy_cold[particle_idx];
    p.energy_thermal_peak = source.energy_thermal_peak[particle_idx];
    p.pressure_peak = source.pressure_peak[particle_idx];
    p.fracture_area = source.fracture_area[particle_idx];
    p.fragment_length = source.fragment_length[particle_idx];
    p.damping_energy = source.damping_energy[particle_idx];

    p.S_xx = source.S_xx[particle_idx]; p.S_xy = source.S_xy[particle_idx]; p.S_xz = source.S_xz[particle_idx];
    p.S_yy = source.S_yy[particle_idx]; p.S_yz = source.S_yz[particle_idx]; p.S_zz = source.S_zz[particle_idx];
    p.damage = source.damage[particle_idx]; p.crack_strain = source.crack_strain[particle_idx]; p.flaws_ntot = source.flaws_ntot[particle_idx];
    p.flaws_strain_min = source.flaws_strain_min[particle_idx]; p.flaws_strain_max = source.flaws_strain_max[particle_idx]; p.flaws_m0 = source.flaws_m0[particle_idx];
    p.distension = source.distension[particle_idx];
    p.L_00 = source.L_00[particle_idx]; p.L_01 = source.L_01[particle_idx]; p.L_02 = source.L_02[particle_idx];
    p.L_10 = source.L_10[particle_idx]; p.L_11 = source.L_11[particle_idx]; p.L_12 = source.L_12[particle_idx];
    p.L_20 = source.L_20[particle_idx]; p.L_21 = source.L_21[particle_idx]; p.L_22 = source.L_22[particle_idx];
    p.global_id = source.global_id[particle_idx];
    p.material_id = source.material_id[particle_idx];
    p.clump_id = source.clump_id[particle_idx]; p.clump_off_x = source.clump_off_x[particle_idx];
    p.clump_off_y = source.clump_off_y[particle_idx]; p.clump_off_z = source.clump_off_z[particle_idx];
    p.source_gpu = source_gpu;
    p.num_active_contacts = 0;
    p.num_active_wall_contacts = 0;
    p.contact_payload_offset = 0;
    p.wall_payload_offset = 0;
    return p;
}

__global__ void direct_scatter_halo_particles_kernel(
    const int32_t* __restrict__ is_halo,
    const int32_t* __restrict__ halo_dest_mask,
    PackedParticle* const* __restrict__ send_particle_buffers,
    int32_t* const* __restrict__ send_index_buffers,
    int32_t* __restrict__ send_counts,
    int32_t buffer_capacity,
    ParticlePackSource source,
    int32_t num_particles,
    int32_t source_gpu,
    int32_t num_gpus
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !is_halo[idx]) return;

    const int words = (num_gpus + 31) / 32;
    const auto* destinations = reinterpret_cast<const uint32_t*>(halo_dest_mask) + static_cast<size_t>(idx) * words;
    const PackedParticle particle = make_packed_particle(idx, source, source_gpu);
    for (int32_t peer = 0; peer < num_gpus; ++peer) {
        if (peer == source_gpu || !(destinations[peer / 32] & (uint32_t{1} << (peer % 32)))) continue;

        int32_t slot = atomicAdd(send_counts + peer, 1);
        // Keep incrementing counts after capacity is exhausted so the host can
        // report a fatal overflow, but never write beyond a peer's buffer.
        if (slot < buffer_capacity) {
            send_index_buffers[peer][slot] = idx;
            send_particle_buffers[peer][slot] = particle;
        }
    }
}

__global__ void direct_scatter_migrating_particles_kernel(
    const int32_t* __restrict__ is_migrating,
    const int32_t* __restrict__ migrate_dest_gpu,
    PackedParticle* const* __restrict__ send_particle_buffers,
    int32_t* const* __restrict__ send_index_buffers,
    int32_t* __restrict__ send_counts,
    int32_t buffer_capacity,
    ParticlePackSource source,
    int32_t num_particles,
    int32_t source_gpu,
    int32_t num_gpus
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || !is_migrating[idx]) return;

    const int32_t peer = migrate_dest_gpu[idx];
    if (peer == source_gpu || peer < 0 || peer >= num_gpus) return;

    const PackedParticle particle = make_packed_particle(idx, source, source_gpu);
    int32_t slot = atomicAdd(send_counts + peer, 1);
    // Keep incrementing counts after capacity is exhausted so the host can
    // report a fatal overflow, but never write beyond a peer's buffer.
    if (slot < buffer_capacity) {
        send_index_buffers[peer][slot] = idx;
        send_particle_buffers[peer][slot] = particle;
    }
}

// Count active rows before allocating the compact wire arenas.  The counts are
// per destination slice, so a sparse history never reserves stride-sized
// transport storage.
__global__ void count_active_migration_history_kernel(
    const int32_t* indices, int32_t n, const ContactHistory* contacts,
    const WallContactHistory* walls, const int32_t* contact_offsets,
    const int32_t* wall_offsets, const int32_t* contact_counts,
    const int32_t* wall_counts, int32_t contact_stride, int32_t wall_stride,
    int32_t* total_contacts, int32_t* total_walls, int32_t* overflow) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t particle = indices[i];
    const int32_t nc = contact_counts[particle], nw = wall_counts[particle];
    if (nc < 0 || nc > contact_stride) { atomicCAS(overflow, 0, 1); return; }
    if (nw < 0 || nw > wall_stride) { atomicCAS(overflow, 0, 2); return; }
    int32_t active_contacts = 0, active_walls = 0;
    for (int32_t j = 0; j < nc; ++j) active_contacts += contacts[contact_offsets[particle] + j].is_active;
    for (int32_t j = 0; j < nw; ++j) active_walls += walls[wall_offsets[particle] + j].is_active;
    atomicAdd(total_contacts, active_contacts);
    atomicAdd(total_walls, active_walls);
}

// Each packed particle owns an exact subrange in its peer's active-only
// payload.  A thread writes only its own slice, so packing is fully parallel.
__global__ void pack_active_migration_history_kernel(
    ContactHistoryPacked* contacts, WallContactHistoryPacked* walls,
    PackedParticle* particles, const ContactHistory* contact_history,
    const WallContactHistory* wall_history, const int32_t* contact_offsets,
    const int32_t* wall_offsets, const int32_t* contact_counts,
    const int32_t* wall_counts, int32_t n, const int32_t* particle_indices,
    int32_t contact_stride, int32_t wall_stride, int32_t* contact_counter,
    int32_t* wall_counter) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t particle = particle_indices[i];
    const int32_t nc = contact_counts[particle], nw = wall_counts[particle];
    int32_t active_contacts = 0, active_walls = 0;
    for (int32_t j = 0; j < nc; ++j) active_contacts += contact_history[contact_offsets[particle] + j].is_active;
    for (int32_t j = 0; j < nw; ++j) active_walls += wall_history[wall_offsets[particle] + j].is_active;
    const int32_t c = atomicAdd(contact_counter, active_contacts);
    const int32_t w = atomicAdd(wall_counter, active_walls);
    particles[i].contact_payload_offset = c;
    particles[i].wall_payload_offset = w;
    int32_t cj = 0;
    for (int32_t j = 0; j < nc; ++j) {
        const ContactHistory& in = contact_history[contact_offsets[particle] + j];
        if (!in.is_active) continue;
        ContactHistoryPacked& out = contacts[c + cj++];
        out.particle_global_id = particles[i].global_id; out.partner_global_id = in.partner_global_id;
        out.delta_t_x=in.delta_t_x; out.delta_t_y=in.delta_t_y; out.delta_t_z=in.delta_t_z;
        out.delta_r_x=in.delta_r_x; out.delta_r_y=in.delta_r_y; out.delta_r_z=in.delta_r_z;
        out.delta_tw_x=in.delta_tw_x; out.delta_tw_y=in.delta_tw_y; out.delta_tw_z=in.delta_tw_z; out.is_active=in.is_active;
    }
    int32_t wj = 0;
    for (int32_t j = 0; j < nw; ++j) {
        const WallContactHistory& in = wall_history[wall_offsets[particle] + j];
        if (!in.is_active) continue;
        WallContactHistoryPacked& out = walls[w + wj++];
        out.wall_id=in.wall_id; out.delta_t_x=in.delta_t_x; out.delta_t_y=in.delta_t_y; out.delta_t_z=in.delta_t_z;
        out.delta_r_x=in.delta_r_x; out.delta_r_y=in.delta_r_y; out.delta_r_z=in.delta_r_z;
        out.delta_tw_x=in.delta_tw_x; out.delta_tw_y=in.delta_tw_y; out.delta_tw_z=in.delta_tw_z;
        out.is_active=in.is_active; out.is_slipping=in.is_slipping; out.is_rolling=in.is_rolling; out.is_twisting=in.is_twisting;
    }
    particles[i].num_active_contacts = active_contacts;
    particles[i].num_active_wall_contacts = active_walls;
}

__global__ void unpack_contact_history_active_kernel(const ContactHistoryPacked* packed, const PackedParticle* particles,
    int32_t n, int32_t first, ContactHistory* history, int32_t* offsets, int32_t* counts, int32_t stride) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t dst = first + i, count = particles[i].num_active_contacts;
    ContactHistory* row = history + static_cast<size_t>(dst) * stride;
    for (int32_t j = 0; j < count; ++j) {
        const auto& in = packed[particles[i].contact_payload_offset + j];
        auto& out = row[j];
        out.partner_global_id = in.partner_global_id;
        out.partner_local_id = -1;
        out.delta_t_x = in.delta_t_x; out.delta_t_y = in.delta_t_y; out.delta_t_z = in.delta_t_z;
        out.delta_r_x = in.delta_r_x; out.delta_r_y = in.delta_r_y; out.delta_r_z = in.delta_r_z;
        out.delta_tw_x = in.delta_tw_x; out.delta_tw_y = in.delta_tw_y; out.delta_tw_z = in.delta_tw_z;
        out.is_active = true;
    }
    for (int32_t j = count; j < stride; ++j) row[j] = ContactHistory();
    offsets[dst] = dst * stride;
    counts[dst] = count;
}

__global__ void unpack_wall_history_active_kernel(const WallContactHistoryPacked* packed, const PackedParticle* particles,
    int32_t n, int32_t first, WallContactHistory* history, int32_t* offsets, int32_t* counts, int32_t stride) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t dst = first + i, count = particles[i].num_active_wall_contacts;
    WallContactHistory* row = history + static_cast<size_t>(dst) * stride;
    for (int32_t j = 0; j < count; ++j) {
        const auto& in = packed[particles[i].wall_payload_offset + j];
        auto& out = row[j];
        out.wall_id = in.wall_id;
        out.delta_t_x = in.delta_t_x; out.delta_t_y = in.delta_t_y; out.delta_t_z = in.delta_t_z;
        out.delta_r_x = in.delta_r_x; out.delta_r_y = in.delta_r_y; out.delta_r_z = in.delta_r_z;
        out.delta_tw_x = in.delta_tw_x; out.delta_tw_y = in.delta_tw_y; out.delta_tw_z = in.delta_tw_z;
        out.is_active = true;
        out.is_slipping = in.is_slipping;
        out.is_rolling = in.is_rolling;
        out.is_twisting = in.is_twisting;
    }
    for (int32_t j = count; j < stride; ++j) row[j] = WallContactHistory();
    offsets[dst] = dst * stride;
    counts[dst] = count;
}

// =============================================================================
// Build Halo Send List Kernel
// =============================================================================

__global__ void build_halo_send_list_kernel(
    const int32_t* __restrict__ is_halo,
    const int32_t* __restrict__ halo_dest_mask,
    int32_t* __restrict__ halo_indices,
    int32_t* __restrict__ halo_count,
    int32_t num_particles,
    int32_t target_gpu,
    int32_t buffer_capacity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (is_halo[idx] && (halo_dest_mask[idx] & (int32_t{1} << target_gpu))) {
        int32_t slot = atomicAdd(halo_count, 1);
        // Keep counting after capacity is exhausted so the host overflow check
        // can fail safely, but do not write beyond this peer's send buffer.
        if (slot < buffer_capacity) {
            halo_indices[slot] = idx;
        }
    }
}

// =============================================================================
// Build Migration Send List Kernel
// =============================================================================

__global__ void build_migration_send_list_kernel(
    const int32_t* __restrict__ is_migrating,
    const int32_t* __restrict__ migrate_dest_gpu,
    int32_t* __restrict__ migrate_indices,
    int32_t* __restrict__ migrate_count,
    int32_t num_particles,
    int32_t target_gpu,
    int32_t buffer_capacity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (is_migrating[idx] && migrate_dest_gpu[idx] == target_gpu) {
        int32_t slot = atomicAdd(migrate_count, 1);
        // Keep counting after capacity is exhausted so the host overflow check
        // can fail safely, but do not write beyond this peer's send buffer.
        if (slot < buffer_capacity) {
            migrate_indices[slot] = idx;
        }
    }
}

// =============================================================================
// Copy Temp to All Kernel (for compaction finalization)
// =============================================================================

__global__ void copy_temp_all_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    double* __restrict__ last_pos_x,
    double* __restrict__ last_pos_y,
    double* __restrict__ last_pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    double* __restrict__ quat_w,
    double* __restrict__ quat_x,
    double* __restrict__ quat_y,
    double* __restrict__ quat_z,
    float* __restrict__ radius,
    float* __restrict__ mass,
    float* __restrict__ moment_of_inertia,
    float* __restrict__ cohesive_strength,
    int32_t* __restrict__ global_id,
    int32_t* __restrict__ contact_history_offset,
    int32_t* __restrict__ num_contacts,
    const double* __restrict__ temp_pos_x,
    const double* __restrict__ temp_pos_y,
    const double* __restrict__ temp_pos_z,
    const double* __restrict__ temp_last_x,
    const double* __restrict__ temp_last_y,
    const double* __restrict__ temp_last_z,
    const double* __restrict__ temp_vel_x,
    const double* __restrict__ temp_vel_y,
    const double* __restrict__ temp_vel_z,
    const double* __restrict__ temp_omega_x,
    const double* __restrict__ temp_omega_y,
    const double* __restrict__ temp_omega_z,
    const double* __restrict__ temp_quat_w,
    const double* __restrict__ temp_quat_x,
    const double* __restrict__ temp_quat_y,
    const double* __restrict__ temp_quat_z,
    const float* __restrict__ temp_radius,
    const float* __restrict__ temp_mass,
    const float* __restrict__ temp_moi,
    const float* __restrict__ temp_cohesive,
    const int32_t* __restrict__ temp_global_id,
    const int32_t* __restrict__ temp_contact_offset,
    const int32_t* __restrict__ temp_num_contacts,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    pos_x[idx] = temp_pos_x[idx];
    pos_y[idx] = temp_pos_y[idx];
    pos_z[idx] = temp_pos_z[idx];
    last_pos_x[idx] = temp_last_x[idx];
    last_pos_y[idx] = temp_last_y[idx];
    last_pos_z[idx] = temp_last_z[idx];
    vel_x[idx] = temp_vel_x[idx];
    vel_y[idx] = temp_vel_y[idx];
    vel_z[idx] = temp_vel_z[idx];
    omega_x[idx] = temp_omega_x[idx];
    omega_y[idx] = temp_omega_y[idx];
    omega_z[idx] = temp_omega_z[idx];
    quat_w[idx] = temp_quat_w[idx];
    quat_x[idx] = temp_quat_x[idx];
    quat_y[idx] = temp_quat_y[idx];
    quat_z[idx] = temp_quat_z[idx];
    radius[idx] = temp_radius[idx];
    mass[idx] = temp_mass[idx];
    moment_of_inertia[idx] = temp_moi[idx];
    cohesive_strength[idx] = temp_cohesive[idx];
    global_id[idx] = temp_global_id[idx];
    contact_history_offset[idx] = temp_contact_offset[idx];
    num_contacts[idx] = temp_num_contacts[idx];
}

__global__ void repack_halo_particles_kernel(
    PackedParticle* destination, const int32_t* indices, int32_t count,
    ParticlePackSource source, int32_t source_gpu) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) destination[i] = make_packed_particle(indices[i], source, source_gpu);
}


} // namespace cudadem

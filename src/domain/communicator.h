#ifndef CUDADEM_COMMUNICATOR_H
#define CUDADEM_COMMUNICATOR_H

#include <vector>
#include <cuda_runtime.h>
#include "domain_manager.h"
#include "p2p_topology.h"

namespace cudadem {

// =============================================================================
// Packed Particle Structure for P2P Transfer
// =============================================================================
// AoS layout for contiguous memory transfer via cudaMemcpyPeerAsync
// =============================================================================

struct PackedParticle {
    // Position (double precision for accuracy)
    double pos_x, pos_y, pos_z;
    double last_pos_x, last_pos_y, last_pos_z;

    // Retained RESPA slow acceleration follows ownership between endpoints.
    double gravity_force_x, gravity_force_y, gravity_force_z;

    // Velocity
    double vel_x, vel_y, vel_z;

    // Angular velocity
    double omega_x, omega_y, omega_z;

    // Orientation quaternion
    double quat_w, quat_x, quat_y, quat_z;

    // Properties
    float radius;
    float mass;
    float moment_of_inertia;
    float cohesive_strength;

    // SPH state.  The same packet is used for halos and migrants so a
    // cross-GPU SPH interaction observes exactly the same material state as a
    // local interaction.
    float support_extent;
    double density, rho_rate;
    float pressure, viscosity, c_sound, drhodt;
    float smoothing_length;
    double energy, dedt;
    double energy_cold, energy_thermal_peak, pressure_peak, fracture_area, fragment_length, damping_energy;
    float S_xx, S_xy, S_xz, S_yy, S_yz, S_zz;
    float damage, crack_strain;
    int32_t flaws_ntot;
    float flaws_strain_min, flaws_strain_max, flaws_m0;
    float distension;
    float L_00, L_01, L_02, L_10, L_11, L_12, L_20, L_21, L_22;

    // Clump membership is immutable particle metadata.  Rigid-body state is
    // managed separately, but these fields must not change under migration.
    int32_t clump_id;
    float clump_off_x, clump_off_y, clump_off_z;

    // Tracking
    int32_t global_id;           // Permanent unique ID
    int32_t material_id;         // Immutable dense material database ID
    int32_t source_gpu;          // GPU sending this particle
    int32_t num_active_contacts; // Number of contact history entries
    int32_t num_active_wall_contacts; // Number of wall-contact history entries
    int32_t contact_payload_offset;   // Active-only transit payload offset
    int32_t wall_payload_offset;      // Active-only transit payload offset
    int64_t winding_x, winding_y, winding_z;
};

// Read-only particle fields consumed by the direct-scatter pack kernels.
// Passing this descriptor keeps the halo and migration kernels' signatures
// manageable while preserving the existing SoA-to-AoS packing layout.
struct ParticlePackSource {
    const double* pos_x;
    const double* pos_y;
    const double* pos_z;
    const double* last_pos_x;
    const double* last_pos_y;
    const double* last_pos_z;
    const double* vel_x;
    const double* vel_y;
    const double* vel_z;
    const double* omega_x;
    const double* omega_y;
    const double* omega_z;
    const double* quat_w;
    const double* quat_x;
    const double* quat_y;
    const double* quat_z;
    const float* radius;
    const float* mass;
    const float* moment_of_inertia;
    const float* cohesive_strength;
    const float* support_extent;
    const double* density;
    const float* pressure;
    const float* viscosity;
    const float* c_sound;
    const float* drhodt;
    const double* rho_rate;
    const float* smoothing_length;
    const double* energy;
    const double* dedt;
    const double* energy_cold;
    const double* energy_thermal_peak;
    const double* pressure_peak;
    const double* fracture_area;
    const double* fragment_length;
    const double* damping_energy;

    const float* S_xx; const float* S_xy; const float* S_xz;
    const float* S_yy; const float* S_yz; const float* S_zz;
    const float* damage;
    const float* crack_strain;
    const int32_t* flaws_ntot;
    const float* flaws_strain_min;
    const float* flaws_strain_max;
    const float* flaws_m0;
    const float* distension;
    const float* L_00; const float* L_01; const float* L_02;
    const float* L_10; const float* L_11; const float* L_12;
    const float* L_20; const float* L_21; const float* L_22;
    const int32_t* global_id;
    const int32_t* material_id;
    const int32_t* clump_id;
    const float* clump_off_x;
    const float* clump_off_y;
    const float* clump_off_z;
    const double* gravity_force_x;
    const double* gravity_force_y;
    const double* gravity_force_z;
    const int64_t* winding_x;
    const int64_t* winding_y;
    const int64_t* winding_z;
};

// =============================================================================
// Packed Contact History for Migration
// =============================================================================
// Uses global IDs instead of local indices for cross-GPU transfer
// =============================================================================

struct ContactHistoryPacked {
    int32_t particle_global_id;  // Owner particle (global ID)
    int32_t partner_global_id;   // Contact partner (global ID)

    // Tangential spring displacement
    float delta_t_x, delta_t_y, delta_t_z;

    // Rolling displacement
    float delta_r_x, delta_r_y, delta_r_z;

    // Twisting displacement
    float delta_tw_x, delta_tw_y, delta_tw_z;

    bool is_active;

    // Padding
    uint8_t _pad[3];
};

// Active-only migration representation of wall-contact spring memory.
struct WallContactHistoryPacked {
    int32_t wall_id;
    float delta_t_x, delta_t_y, delta_t_z;
    float delta_r_x, delta_r_y, delta_r_z;
    float delta_tw_x, delta_tw_y, delta_tw_z;
    bool is_active;
    bool is_slipping;
    bool is_rolling;
    bool is_twisting;
};

// =============================================================================
// Communication Buffer for GPU Pair
// =============================================================================

struct CommBuffer {
    // Send buffers (device memory on source GPU)
    PackedParticle* d_send_particles;
    ContactHistoryPacked* d_send_history;
    WallContactHistoryPacked* d_send_wall_history;
    int32_t* d_send_indices;     // Indices of particles to send
    int32_t* d_send_count;       // Number of particles to send

    // Receive buffers (device memory on this GPU)
    PackedParticle* d_recv_particles;
    ContactHistoryPacked* d_recv_history;
    WallContactHistoryPacked* d_recv_wall_history;
    int32_t* d_recv_count;       // Number of particles received

    // Host staging for count synchronization
    int32_t h_send_count;
    int32_t h_recv_count;
    int32_t h_send_history_count;
    int32_t h_send_wall_history_count;
    int32_t h_recv_history_count;
    int32_t h_recv_wall_history_count;

    // Capacity
    int32_t particle_capacity;
    int32_t history_capacity;
    int32_t wall_history_capacity;
    int32_t contact_history_stride;
    int32_t wall_history_stride;

    CommBuffer()
        : d_send_particles(nullptr), d_send_history(nullptr), d_send_wall_history(nullptr),
          d_send_indices(nullptr), d_send_count(nullptr),
          d_recv_particles(nullptr), d_recv_history(nullptr), d_recv_wall_history(nullptr),
          d_recv_count(nullptr),
          h_send_count(0), h_recv_count(0), h_send_history_count(0), h_send_wall_history_count(0),
          h_recv_history_count(0), h_recv_wall_history_count(0),
          particle_capacity(0), history_capacity(0), wall_history_capacity(0),
          contact_history_stride(0), wall_history_stride(0) {}
};

// Snapshot exported for allocation telemetry.  Payload capacity is intentionally
// reported separately from local particle capacity: it is communication
// workspace, not a reservation of simulation particles.
struct CommunicationAllocationTelemetry {
    size_t send_particle_bytes = 0;
    size_t recv_particle_bytes = 0;
    size_t send_history_bytes = 0;
    size_t recv_history_bytes = 0;
    size_t send_index_bytes = 0;
    size_t workspace_bytes = 0;
    int32_t active_send_peers = 0;
    int32_t active_recv_peers = 0;
};

struct HaloTransferSummary {
    int64_t sent_particles = 0;
    int64_t received_particles = 0;
    size_t sent_bytes = 0;
    size_t received_bytes = 0;
    int32_t active_send_peers = 0;
    int32_t active_receive_peers = 0;
};

// =============================================================================
// P2P Topology Information
// =============================================================================

struct P2PTopology {
    std::vector<std::vector<bool>> can_access;  // can_access[i][j] = GPU i can access GPU j
    // Compatibility view derived exclusively from link_class.  It is never
    // inferred from CUDA native-atomic support.
    std::vector<std::vector<bool>> is_nvlink;
    std::vector<std::vector<LinkClass>> link_class;
    bool nvlink_available;

    P2PTopology() : nvlink_available(false) {}
};

// =============================================================================
// Communicator Class
// =============================================================================
// Handles P2P memory transfers for halo exchange and particle migration
// =============================================================================

class Communicator {
private:
    int32_t device_id_;
    int32_t num_gpus_;
    std::vector<CommBuffer> buffers_;  // One per peer GPU
    P2PTopology topology_;

    // Halo identification buffers
    int32_t* d_halo_count_;
    int32_t* d_halo_dest_mask_;        // Bitmask of GPUs each halo overlaps

    // Migration buffers
    int32_t* d_migrate_indices_;
    int32_t* d_migrate_count_;
    int32_t* d_migrate_dest_gpu_;

    // Compaction buffers
    int32_t* d_keep_flags_;            // 1 = keep, 0 = migrating out
    int32_t* d_new_indices_;           // Prefix sum for compaction

    int32_t max_particles_;

    // ORB box staging and migration history accounting.
    double*   d_gpu_box_min_x_;       // MAX_MULTI_GPUS doubles for ORB box data
    double*   d_gpu_box_min_y_;
    double*   d_gpu_box_min_z_;
    double*   d_gpu_box_max_x_;
    double*   d_gpu_box_max_y_;
    double*   d_gpu_box_max_z_;
    int32_t contact_history_stride_;
    int32_t wall_history_stride_;
    int32_t* d_history_overflow_status_;
    int32_t* d_history_payload_counters_;
    int32_t* d_send_contact_history_counts_;
    int32_t* d_send_wall_history_counts_;

    // =========================================================================
    // CUB temp storage (Phase 5: eliminate thrust temp alloc per step)
    // =========================================================================
    void*   d_cub_scan_temp_;          // Pre-allocated CUB temp storage
    size_t  cub_scan_temp_bytes_;      // Size of temp storage

    // =========================================================================
    // Contiguous send counts (Phase 7: batch memset + memcpy)
    // =========================================================================
    int32_t* d_all_send_counts_;       // num_gpus_ entries on device

    // Device-side views of each peer's existing send buffers.  The direct
    // scatter kernels index these tables by destination GPU.
    PackedParticle** d_send_particle_buffers_;
    int32_t** d_send_index_buffers_;
    int32_t send_particle_capacity_;

    // One grow-only arena per direction. CommBuffer pointers are non-owning
    // peer slices into these arenas; this avoids reserving a full payload for
    // every possible GPU pair.
    PackedParticle* d_send_particle_arena_;
    PackedParticle* d_recv_particle_arena_;
    ContactHistoryPacked* d_send_history_arena_;
    ContactHistoryPacked* d_recv_history_arena_;
    WallContactHistoryPacked* d_send_wall_history_arena_;
    WallContactHistoryPacked* d_recv_wall_history_arena_;
    int32_t* d_send_index_arena_;
    int32_t send_arena_capacity_;
    int32_t recv_arena_capacity_;
    int32_t send_history_arena_capacity_;
    int32_t send_wall_history_arena_capacity_;
    int32_t recv_history_arena_capacity_;
    int32_t recv_wall_history_arena_capacity_;

    // =========================================================================
    // Pinned host memory (Phase 2: eliminate synchronous D2H scalar copies)
    // =========================================================================
    int32_t* h_pinned_send_counts_;    // num_gpus_ entries for batched count readback
    int32_t* h_pinned_scan_result_;    // 2 entries: [last_scan, last_keep] for compaction
    CommunicationAllocationTelemetry allocation_telemetry_;
    // Reset when migration classification starts.  It records ownership
    // membership changes, not a change in the final local particle count.
    bool particle_membership_changed_;

public:
    Communicator()
        : device_id_(-1), num_gpus_(0),
          d_halo_count_(nullptr), d_halo_dest_mask_(nullptr),
          d_migrate_indices_(nullptr), d_migrate_count_(nullptr), d_migrate_dest_gpu_(nullptr),
          d_keep_flags_(nullptr), d_new_indices_(nullptr),
          max_particles_(0),
          d_gpu_box_min_x_(nullptr), d_gpu_box_min_y_(nullptr), d_gpu_box_min_z_(nullptr),
          d_gpu_box_max_x_(nullptr), d_gpu_box_max_y_(nullptr), d_gpu_box_max_z_(nullptr),
          contact_history_stride_(0), wall_history_stride_(0), d_history_overflow_status_(nullptr), d_history_payload_counters_(nullptr),
          d_send_contact_history_counts_(nullptr), d_send_wall_history_counts_(nullptr),
          d_cub_scan_temp_(nullptr), cub_scan_temp_bytes_(0),
          d_all_send_counts_(nullptr),
          d_send_particle_buffers_(nullptr), d_send_index_buffers_(nullptr),
          send_particle_capacity_(0), d_send_particle_arena_(nullptr), d_recv_particle_arena_(nullptr),
          d_send_history_arena_(nullptr), d_recv_history_arena_(nullptr),
          d_send_wall_history_arena_(nullptr), d_recv_wall_history_arena_(nullptr),
          d_send_index_arena_(nullptr), send_arena_capacity_(0), recv_arena_capacity_(0),
          send_history_arena_capacity_(0), send_wall_history_arena_capacity_(0),
          recv_history_arena_capacity_(0), recv_wall_history_arena_capacity_(0),
          h_pinned_send_counts_(nullptr), h_pinned_scan_result_(nullptr),
          particle_membership_changed_(false) {}

    ~Communicator() {
        shutdown();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    void initialize(int32_t device_id, int32_t num_gpus, int32_t max_particles,
                    int32_t contact_history_stride, int32_t wall_history_stride);

    // Grow particle-indexed classification/compaction workspace after the
    // owning GPUDomainState grows. Call only at the same local phase boundary.
    void ensure_workspace_capacity(int32_t required_capacity);

    // Query P2P topology
    void query_topology();

    // =========================================================================
    // Halo Exchange (every timestep before neighbor search)
    // =========================================================================

    // Step 1: Identify halo particles (near subdomain boundary)
    void identify_halo_particles(
        GPUDomainState& gpu_state,
        const DomainParamsGPU& params,
        float halo_width,
        cudaStream_t stream = nullptr,
        bool use_support_extent = false,
        float interaction_extent_scale = 1.0f
    );

    // Step 2: Pack halo particles into send buffers
    void pack_halo_particles(
        const GPUDomainState& gpu_state,
        cudaStream_t stream = nullptr
    );

    // Refresh the current halo's payload in the existing slot order. No
    // positions/ownership change: neighbor indices remain valid.
    void repack_halo_particles(const GPUDomainState& gpu_state,
                               cudaStream_t stream = nullptr);

    // Step 3: Exchange counts between GPUs (synchronization point)
    void exchange_halo_counts(
        std::vector<GPUDomainState>& all_gpu_states
    );

    // Step 4: P2P transfer of packed particles
    // NOTE: Requires access to ALL communicators to read peer's send buffers
    void transfer_halos(
        std::vector<Communicator>& all_communicators,
        cudaStream_t stream = nullptr
    );

    // Step 5: Unpack received halos as ghost particles
    void unpack_ghost_particles(
        GPUDomainState& gpu_state,
        cudaStream_t stream = nullptr
    );

    // Convenience: Full halo exchange pipeline
    // NOTE: Requires access to ALL communicators for P2P transfers
    void exchange_halos(
        std::vector<GPUDomainState>& all_gpu_states,
        std::vector<Communicator>& all_communicators,
        const DomainParamsGPU& params,
        float halo_width
    );

    // =========================================================================
    // Particle Migration (after integration, when particles cross boundaries)
    // =========================================================================

    // Identify particles that left this GPU's domain
    void identify_migrating_particles(
        GPUDomainState& gpu_state,
        const DomainParamsGPU& params,
        cudaStream_t stream = nullptr
    );

    // Pack migrating particles WITH their contact history
    void pack_migrating_particles(
        GPUDomainState& gpu_state,
        cudaStream_t stream = nullptr
    );

    // Transfer migrating particles to new owners
    // NOTE: Requires access to ALL communicators to read peer's send buffers
    void transfer_migrants(
        std::vector<Communicator>& all_communicators,
        cudaStream_t stream
    );

    // Unpack received migrants and integrate into local arrays
    void unpack_migrating_particles(
        GPUDomainState& gpu_state,
        cudaStream_t stream = nullptr
    );

    // Remove migrated particles from local arrays (compact)
    void compact_local_particles(
        GPUDomainState& gpu_state,
        cudaStream_t stream = nullptr
    );

    // Convenience: Full migration pipeline
    // NOTE: Requires access to ALL communicators for P2P transfers
    void migrate_particles(
        std::vector<GPUDomainState>& all_gpu_states,
        std::vector<Communicator>& all_communicators,
        const DomainParamsGPU& params
    );

    // =========================================================================
    // Cleanup
    // =========================================================================

    void shutdown();

    // =========================================================================
    // Accessors
    // =========================================================================

    int32_t device_id() const { return device_id_; }
    int32_t num_gpus() const { return num_gpus_; }
    const P2PTopology& topology() const { return topology_; }
    LinkClass link_class_to(int32_t peer) const {
        return (device_id_ >= 0 && device_id_ < num_gpus_ &&
                peer >= 0 && peer < num_gpus_ && !topology_.link_class.empty())
            ? topology_.link_class[device_id_][peer] : LinkClass::Unavailable;
    }
    // Valid after pack_migrating_particles/transfer_migrants for the current
    // migration epoch.  A balanced send/receive exchange still returns true.
    bool particle_membership_changed() const { return particle_membership_changed_; }
    // Per-owned-particle destination bitmasks produced by
    // identify_halo_particles.  The caller owns neither this storage nor its
    // lifetime; only entries [0, halo_destination_mask_capacity()) are valid.
    const int32_t* halo_destination_mask() const { return d_halo_dest_mask_; }
    int32_t halo_destination_mask_capacity() const { return max_particles_; }
    // Snapshot after the halo pack/transfer lifecycle.  Counts are exact
    // packet units from existing per-peer buffers; no telemetry storage is
    // allocated for this view.
    HaloTransferSummary halo_transfer_summary() const;

    // Access communication buffers for diagnostic readout of send/receive counts
    const std::vector<CommBuffer>& buffers() const { return buffers_; }
    CommunicationAllocationTelemetry allocation_telemetry() const;

private:
    void layout_send_arena(const std::vector<int32_t>& counts);
    void layout_recv_arena(const std::vector<int32_t>& counts);
    void layout_send_history_arenas(const std::vector<int32_t>& contact_counts,
                                    const std::vector<int32_t>& wall_counts);
    void layout_recv_history_arenas(const std::vector<int32_t>& contact_counts,
                                    const std::vector<int32_t>& wall_counts);
};

// =============================================================================
// Kernel Declarations (implemented in packer_kernel.cu)
// =============================================================================

__global__ void pack_particles_kernel(
    PackedParticle* __restrict__ buffer,
    const int32_t* __restrict__ indices_to_pack,
    int32_t num_to_pack,
    // SoA source arrays
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
);

__global__ void unpack_ghost_particles_kernel(
    const PackedParticle* __restrict__ buffer,
    int32_t num_received,
    int32_t ghost_start_offset,
    ParticleSoA particles
);

__global__ void pack_contact_history_kernel(
    ContactHistoryPacked* __restrict__ packed_history,
    int32_t* __restrict__ num_packed,
    const ContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ migrate_indices,
    int32_t num_migrating,
    const int32_t* __restrict__ particle_global_id,
    int32_t max_contacts_per_particle
);

__global__ void unpack_contact_history_kernel(
    const ContactHistoryPacked* __restrict__ packed_history,
    int32_t num_packed,
    ContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ global_to_local,
    int32_t global_to_local_size,
    int32_t max_contacts_per_particle
);

__global__ void validate_migration_history_counts_kernel(
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ num_wall_contacts,
    int32_t max_contacts_per_particle,
    int32_t max_wall_contacts_per_particle,
    int32_t* __restrict__ overflow_status);

__global__ void pack_contact_history_fixed_stride_kernel(
    ContactHistoryPacked* __restrict__ packed_history,
    PackedParticle* __restrict__ packed_particles,
    const ContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    const int32_t* __restrict__ local_to_global,
    int32_t max_contacts_per_particle);

__global__ void pack_wall_contact_history_fixed_stride_kernel(
    WallContactHistoryPacked* __restrict__ packed_history,
    PackedParticle* __restrict__ packed_particles,
    const WallContactHistory* __restrict__ history_buffer,
    const int32_t* __restrict__ history_offset,
    const int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ particle_indices,
    int32_t num_particles,
    int32_t max_contacts_per_particle);

__global__ void unpack_contact_history_fixed_stride_kernel(
    const ContactHistoryPacked* __restrict__ packed_history,
    int32_t num_particles,
    int32_t first_target_particle,
    ContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts_per_particle);

__global__ void unpack_wall_contact_history_fixed_stride_kernel(
    const WallContactHistoryPacked* __restrict__ packed_history,
    int32_t num_particles,
    int32_t first_target_particle,
    WallContactHistory* __restrict__ history_buffer,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts_per_particle);

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
    int32_t max_contacts_per_particle);

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
    int32_t max_contacts_per_particle);

__global__ void finalize_compacted_wall_history_metadata_kernel(
    int32_t num_particles,
    int32_t* __restrict__ history_offset,
    int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ compacted_count,
    int32_t max_contacts_per_particle);

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
);

__global__ void build_halo_send_list_kernel(
    const int32_t* __restrict__ is_halo,
    const int32_t* __restrict__ halo_dest_mask,
    int32_t* __restrict__ halo_indices,
    int32_t* __restrict__ halo_count,
    int32_t num_particles,
    int32_t target_gpu,
    int32_t buffer_capacity
);

__global__ void build_migration_send_list_kernel(
    const int32_t* __restrict__ is_migrating,
    const int32_t* __restrict__ migrate_dest_gpu,
    int32_t* __restrict__ migrate_indices,
    int32_t* __restrict__ migrate_count,
    int32_t num_particles,
    int32_t target_gpu,
    int32_t buffer_capacity
);

// One particle-parallel direct-scatter launch packs every selected particle
// into its destination peer buffer(s).  Counts intentionally advance past
// capacity so the host-side fatal-overflow checks retain their semantics.
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
);

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
);

__global__ void count_active_migration_history_kernel(const int32_t*, int32_t, const ContactHistory*,
    const WallContactHistory*, const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    int32_t, int32_t, int32_t*, int32_t*, int32_t*);
__global__ void pack_active_migration_history_kernel(ContactHistoryPacked*, WallContactHistoryPacked*, PackedParticle*,
    const ContactHistory*, const WallContactHistory*, const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    int32_t, const int32_t*, int32_t, int32_t, int32_t*, int32_t*);
__global__ void unpack_contact_history_active_kernel(const ContactHistoryPacked*, const PackedParticle*, int32_t,
    int32_t, ContactHistory*, int32_t*, int32_t*, int32_t);
__global__ void unpack_wall_history_active_kernel(const WallContactHistoryPacked*, const PackedParticle*, int32_t,
    int32_t, WallContactHistory*, int32_t*, int32_t*, int32_t);

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
);

__global__ void repack_halo_particles_kernel(
    PackedParticle* destination, const int32_t* indices, int32_t count,
    ParticlePackSource source, int32_t source_gpu);

} // namespace cudadem

#endif // CUDADEM_COMMUNICATOR_H

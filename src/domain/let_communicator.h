// src/domain/let_communicator.h
#ifndef CUDADEM_LET_COMMUNICATOR_H
#define CUDADEM_LET_COMMUNICATOR_H

#include <vector>
#include <cstdint>
#include "let_types.h"

namespace cudadem {

class MultiGPUPhaseCoordinator;

// =============================================================================
// LET Communicator Class (One-Shot FMM Push)
// =============================================================================
// Pushes FMM local-essential-tree data to peer GPUs over P2P / NVLink.
// (The legacy Barnes-Hut push_lets() / MPI path has been removed following
//  consolidation onto the body-fitted FMM as the sole runtime gravity solver.)
// =============================================================================

class LETCommunicator {
private:
    LETPushState state_;
    // Owned by the multi-GPU runner.  All communicators participating in a
    // timestep share this coordinator; it supplies the directed publish /
    // consumed handshakes that make the single LET buffers reusable.
    MultiGPUPhaseCoordinator* coordinator_ = nullptr;
    uint64_t next_fmm_let_epoch_ = 0;
    cudaEvent_t let_compute_ready_ = nullptr;
    cudaEvent_t let_counts_ready_ = nullptr;
    LETTimingSlot* acquire_timing_slot(std::vector<LETTimingSlot>& ring);
    void invalidate_export_timing(uint64_t epoch, bool grew, bool retried);

public:
    LETCommunicator();
    ~LETCommunicator();

    // Prevent copying
    LETCommunicator(const LETCommunicator&) = delete;
    LETCommunicator& operator=(const LETCommunicator&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    void initialize(
        int32_t gpu_id,
        int32_t num_gpus,
        int32_t max_particles,
        int32_t max_nodes,
        int32_t start_level = 2,
        float theta = 0.5f,
        bool use_one_shot_push = true,
        MultiGPUPhaseCoordinator* coordinator = nullptr,
        size_t min_saved_bytes = 65536,
        int32_t max_shared_buckets = 8,
        bool common_tree = false
    );

    void shutdown();

    // =========================================================================
    // FMM Path: Push LETs from FMM octree
    // =========================================================================
    // GPU traverses its local unified FMM tree once, writes final LET wire
    // records to aggregate per-peer send segments, and publishes exact counts
    // for receiver-pull P2P copies.  The source view is non-owning and makes
    // allocation capacity distinct from metadata's live node count.
    // pos_x/y/z and mass are SoA particle data pointers.
    // =========================================================================

    // Returns the monotonically increasing epoch published for this sender.
    uint64_t push_lets_fmm(
        const FMMLETSourceView& source,
        const double* pos_x, const double* pos_y, const double* pos_z,
        const float* mass,
        cudaStream_t stream = nullptr
    );
    // Participate in the same directed LET epoch when this GPU owns no
    // particles.  Publishes exact zero counts without constructing a tree.
    uint64_t push_empty_lets_fmm(cudaStream_t stream = nullptr);

    // =========================================================================
    // FMM Path: acknowledge direct received FMM LET arena
    // =========================================================================
    // Kept as a lifecycle boundary: it records coordinator acknowledgements
    // after count publication.  It performs no payload merge or allocation.
    // =========================================================================

    bool merge_fmm_lets(uint64_t epoch, cudaStream_t stream = nullptr);

    // Waits for every peer's LET publication for epoch, then stages the small
    // count arrays used by the segment-aware remote evaluator. The coordinator's publish /
    // wait release-acquire pairing makes peer pinned counts safe to read.
    bool wait_and_stage_fmm_recv_counts(uint64_t epoch, cudaStream_t stream = nullptr);

    // =========================================================================
    // Accessors
    // =========================================================================

    MultiGPUPhaseCoordinator* coordinator() const { return coordinator_; }
    int32_t gpu_id() const { return state_.gpu_id; }
    int32_t num_gpus() const { return state_.num_gpus; }
    bool is_initialized() const { return state_.initialized; }
    bool is_single_gpu() const { return state_.num_gpus <= 1; }

    const LETPushPerPeer* peers() const { return state_.peers; }
    const LETPushPerPeer& peer(int32_t g) const { return state_.peers[g]; }

    int32_t* h_pinned_my_send_node_counts() { return state_.h_pinned_my_send_node_counts; }
    int32_t* h_pinned_my_send_particle_counts() { return state_.h_pinned_my_send_particle_counts; }

    cudaStream_t compute_stream() const { return state_.compute_stream; }
    cudaStream_t comm_stream() const { return state_.comm_stream; }

    // Direct FMM receive arena metadata.  Entries remain in producer-owned
    // segments; consumers must use these offsets/counts rather than assuming
    // a compacted replica exists.
    const FMMLETNode* fmm_recv_nodes() const { return state_.d_fmm_recv_nodes; }
    const LETParticle* fmm_recv_particles() const { return state_.d_fmm_recv_particles; }
    const int32_t* fmm_recv_node_offsets() const { return state_.d_fmm_recv_node_offsets; }
    const int32_t* fmm_recv_particle_offsets() const { return state_.d_fmm_recv_particle_offsets; }
    const int32_t* fmm_recv_node_counts() const { return state_.d_fmm_recv_node_counts; }
    const int32_t* fmm_recv_particle_counts() const { return state_.d_fmm_recv_particle_counts; }
    size_t fmm_arena_bytes() const { return state_.fmm_arena_bytes; }
    int32_t fmm_send_node_capacity(int32_t peer) const { return state_.h_fmm_send_node_capacities[peer]; }
    int32_t fmm_send_particle_capacity(int32_t peer) const { return state_.h_fmm_send_particle_capacities[peer]; }
    int32_t fmm_recv_node_capacity(int32_t peer) const { return state_.h_fmm_recv_node_capacities[peer]; }
    int32_t fmm_recv_particle_capacity(int32_t peer) const { return state_.h_fmm_recv_particle_capacities[peer]; }
    // Valid until the next LET export starts. Entries are indexed by local FMM
    // source-node slot; fmm_source_work_count() is the fixed node capacity and
    // slots beyond metadata's live node count are current zero records.
    const LETSourceWork* fmm_source_work() const { return state_.d_fmm_source_work; }
    int32_t fmm_source_work_count() const { return state_.fmm_source_work_count; }
    LETTransferSummary fmm_send_summary() const;
    LETTransferSummary fmm_send_peer_summary(int32_t peer) const;
    LETTransferSummary fmm_recv_summary() const;
    LETTransferSummary fmm_recv_peer_summary(int32_t peer) const;
    const LETSharingSummary& fmm_sharing_summary() const { return state_.fmm_sharing_summary; }
    bool last_fmm_send_grew() const { return state_.last_fmm_send_grew; }
    bool last_fmm_send_retried() const { return state_.last_fmm_send_retried; }
    bool last_fmm_recv_grew() const { return state_.last_fmm_recv_grew; }
    void set_timing_sample_interval(uint32_t interval) { state_.timing_sample_interval = interval ? interval : 1; }
    uint32_t timing_sample_interval() const { return state_.timing_sample_interval; }
    void set_peer_link_class(int32_t peer, LinkClass link);
    size_t poll_timing_records(LETTimingRecord* output, size_t capacity);
};

} // namespace cudadem

#endif // CUDADEM_LET_COMMUNICATOR_H

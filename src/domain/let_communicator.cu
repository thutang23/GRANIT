// src/domain/let_communicator.cu
// =============================================================================
// LET Communicator Implementation - One-Shot P2P Push Pipeline
// =============================================================================
// Handles Local Essential Tree exchange for multi-GPU self-gravity via
// direct P2P memory copies over NVLink/PCIe. Replaces the old 3-phase
// MPI boundary-tree protocol entirely.
// =============================================================================
// Pipeline:
//   1. compute_let_one_shot_kernel: single-kernel tree walk fills combined
//      per-peer output arrays via atomicAdd.
//   2. Count exchange: each GPU thread D2H copies only its scalar send counts.
//   3. Publish: producer exposes its completed direct send arena and counts.
//   4. Receiver pull: each consumer waits, demand-grows its compact aggregate
//      receive arena, then cudaMemcpyPeerAsync-pulls its exact segment.
// =============================================================================

#include "let_communicator.h"
#include "peer_transfer.h"
#include "multigpu_phase_coordinator.h"
#include "domain_manager.h"
#include "cuda_utils.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

// FMM types for push_lets_fmm() — opaque in the header, concrete here
#include "core/acceleration/fmm/fmm_types.h"
#include "core/acceleration/body_fitted_fmm/unified_node.cuh"

namespace body_fitted_fmm {
struct PersistentTreeMetadata;
}

namespace cudadem {

namespace {

int32_t grow_let_capacity(const int32_t required, const int32_t current,
                          const int32_t gpu, const int32_t peer,
                          const char* payload) {
    if (required < 0) {
        throw std::runtime_error(
            "Negative LET " + std::string(payload) + " count on GPU "
            + std::to_string(gpu) + " for peer " + std::to_string(peer));
    }
    const int64_t grown = static_cast<int64_t>(current)
                        + (static_cast<int64_t>(current) + 1) / 2;
    const int64_t capacity = std::max<int64_t>(required, grown);
    if (capacity > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "LET " + std::string(payload) + " capacity exceeds int32 indexing on GPU "
            + std::to_string(gpu) + " for peer " + std::to_string(peer));
    }
    return static_cast<int32_t>(capacity);
}

void log_let_growth(const int32_t gpu, const char* direction,
                    const int64_t old_nodes, const int64_t required_nodes,
                    const int64_t new_nodes, const int64_t old_particles,
                    const int64_t required_particles,
                    const int64_t new_particles, const size_t bytes) {
    std::printf("[memory] GPU %d LET grow direction=%s old_nodes=%lld "
                "required_nodes=%lld new_nodes=%lld old_particles=%lld "
                "required_particles=%lld new_particles=%lld bytes=%zu\n",
                gpu, direction, static_cast<long long>(old_nodes),
                static_cast<long long>(required_nodes),
                static_cast<long long>(new_nodes),
                static_cast<long long>(old_particles),
                static_cast<long long>(required_particles),
                static_cast<long long>(new_particles), bytes);
}

} // namespace

// FMM LET kernel forward declarations (defined in let_kernels_fmm.cu)
__global__ void discover_let_fmm_kernel(
    const ::body_fitted_fmm::UnifiedNode*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const ::body_fitted_fmm::PersistentTreeMetadata*, int32_t, int, float, int, int,
    LETExportDescriptor*, int32_t*, int32_t, int32_t*, int32_t*, LETSourceWork*, int32_t*);
__global__ void materialize_let_fmm_kernel(
    const LETExportDescriptor*, int, const ::body_fitted_fmm::UnifiedNode*, const fmm::MultipoleExpansion*,
    const double*, const double*, const double*, const float*, const int32_t*,
    const ::body_fitted_fmm::PersistentTreeMetadata*, int32_t, FMMLETNode*, LETParticle*,
    const int32_t*, const uint8_t*, const int32_t*, const int32_t*, int32_t*, LETSourceWork*, int);

// =============================================================================
// Cross-Communicator Shared State (OpenMP threads share host memory)
// =============================================================================
// Static arrays indexed by GPU ID. Each OpenMP thread (managing one GPU)
// publishes pointers during initialize() so other threads can read peer
// send-buffer addresses for cudaMemcpyPeerAsync.
// =============================================================================

// Direct FMM send arenas published by each GPU for peer receiver-pulls.
static FMMLETNode* g_fmm_send_nodes[MAX_MULTI_GPUS];
static LETParticle* g_fmm_send_particles[MAX_MULTI_GPUS];
static LETSenderSegmentPlan* g_fmm_sender_plans[MAX_MULTI_GPUS];

// =============================================================================
// Constructor / Destructor
// =============================================================================

LETCommunicator::LETCommunicator() {}
LETCommunicator::~LETCommunicator() { shutdown(); }

LETTimingSlot* LETCommunicator::acquire_timing_slot(std::vector<LETTimingSlot>& ring) {
    for (LETTimingSlot& slot : ring) if (!slot.pending) return &slot;
    return nullptr; // fixed ring: skip a sample rather than wait
}

void LETCommunicator::invalidate_export_timing(const uint64_t epoch, const bool grew, const bool retried) {
    for (LETTimingSlot& slot : state_.export_timing_ring) {
        if (slot.pending && slot.record.epoch == epoch) {
            slot.record.grew = slot.record.grew || grew;
            slot.record.retried = slot.record.retried || retried;
            slot.record.valid_for_calibration = !(slot.record.grew || slot.record.retried);
        }
    }
}

void LETCommunicator::set_peer_link_class(const int32_t peer, const LinkClass link) {
    if (peer >= 0 && peer < state_.num_gpus) state_.peer_link_classes[peer] = link;
}

size_t LETCommunicator::poll_timing_records(LETTimingRecord* output, const size_t capacity) {
    if (!output || !capacity) return 0;
    size_t emitted = 0;
    auto harvest = [&output, &capacity, &emitted](std::vector<LETTimingSlot>& ring) {
        for (LETTimingSlot& slot : ring) {
            if (!slot.pending || emitted == capacity) continue;
            const cudaError_t status = cudaEventQuery(slot.stop);
            if (status == cudaErrorNotReady) continue;
            if (status != cudaSuccess) { slot.pending = false; continue; }
            cudaEventElapsedTime(&slot.record.milliseconds, slot.start, slot.stop);
            output[emitted++] = slot.record;
            slot.pending = false;
        }
    };
    harvest(state_.export_timing_ring);
    harvest(state_.receive_timing_ring);
    return emitted;
}

namespace {
LETTransferSummary make_let_summary(const int32_t nodes, const int32_t particles) {
    LETTransferSummary summary;
    summary.node_records = nodes;
    summary.particle_records = particles;
    summary.node_bytes = static_cast<size_t>(nodes) * sizeof(FMMLETNode);
    summary.particle_bytes = static_cast<size_t>(particles) * sizeof(LETParticle);
    summary.total_bytes = summary.node_bytes + summary.particle_bytes;
    summary.active_peers = (nodes > 0 || particles > 0) ? 1 : 0;
    return summary;
}
}

LETTransferSummary LETCommunicator::fmm_send_peer_summary(const int32_t peer) const {
    if (peer < 0 || peer >= state_.num_gpus || !state_.h_pinned_my_send_node_counts)
        return {};
    return make_let_summary(state_.h_pinned_my_send_node_counts[peer],
                            state_.h_pinned_my_send_particle_counts[peer]);
}

LETTransferSummary LETCommunicator::fmm_send_summary() const {
    LETTransferSummary total;
    for (int p = 0; p < state_.num_gpus; ++p) {
        if (p == state_.gpu_id) continue;
        const LETTransferSummary one = fmm_send_peer_summary(p);
        total.node_records += one.node_records; total.particle_records += one.particle_records;
        total.node_bytes += one.node_bytes; total.particle_bytes += one.particle_bytes;
        total.total_bytes += one.total_bytes; total.active_peers += one.active_peers;
    }
    return total;
}

LETTransferSummary LETCommunicator::fmm_recv_peer_summary(const int32_t peer) const {
    if (peer < 0 || peer >= state_.num_gpus || peer == state_.gpu_id || !state_.peers)
        return {};
    return make_let_summary(*state_.peers[peer].h_pinned_recv_node_count,
                            *state_.peers[peer].h_pinned_recv_particle_count);
}

LETTransferSummary LETCommunicator::fmm_recv_summary() const {
    LETTransferSummary total;
    for (int p = 0; p < state_.num_gpus; ++p) {
        const LETTransferSummary one = fmm_recv_peer_summary(p);
        total.node_records += one.node_records; total.particle_records += one.particle_records;
        total.node_bytes += one.node_bytes; total.particle_bytes += one.particle_bytes;
        total.total_bytes += one.total_bytes; total.active_peers += one.active_peers;
    }
    return total;
}

// =============================================================================
// Initialization
// =============================================================================

void LETCommunicator::initialize(
    int32_t gpu_id,
    int32_t num_gpus,
    int32_t max_particles,
    int32_t max_nodes,
    int32_t start_level,
    float theta,
    bool use_one_shot_push,
    MultiGPUPhaseCoordinator* coordinator,
    size_t min_saved_bytes,
    int32_t max_shared_buckets,
    bool common_tree
) {
    // --- Single GPU early exit ---
    if (num_gpus <= 1 || common_tree) {
        state_.peer_link_classes.assign(num_gpus, LinkClass::GenericPcieP2P);
        state_.gpu_id = gpu_id;
        state_.num_gpus = num_gpus;
        state_.start_level = start_level;
        state_.theta = theta;
        state_.min_shared_saved_bytes = min_saved_bytes;
        state_.max_shared_buckets = std::min<int32_t>(8, std::max<int32_t>(0, max_shared_buckets));
        state_.initialized = true;
        state_.use_legacy_mpi = !use_one_shot_push;
        coordinator_ = coordinator;
        return;
    }

    if (!coordinator || !coordinator->initialized()) {
        std::fprintf(stderr,
                     "LETCommunicator requires an initialized MultiGPUPhaseCoordinator "
                     "when num_gpus=%d\n", num_gpus);
        std::abort();
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    state_.gpu_id = gpu_id;
    state_.num_gpus = num_gpus;
    state_.start_level = start_level;
    state_.theta = theta;
    state_.min_shared_saved_bytes = min_saved_bytes;
    state_.max_shared_buckets = std::min<int32_t>(8, std::max<int32_t>(0, max_shared_buckets));
    state_.use_legacy_mpi = !use_one_shot_push;
    coordinator_ = coordinator;
    // Coordinator epoch zero is the "nothing published" sentinel.
    next_fmm_let_epoch_ = 1;

    // --- Create CUDA streams ---
    CUDA_CHECK(cudaStreamCreate(&state_.compute_stream));
    CUDA_CHECK(cudaStreamCreate(&state_.comm_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&let_compute_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&let_counts_ready_, cudaEventDisableTiming));
    state_.peer_link_classes.assign(num_gpus, LinkClass::GenericPcieP2P);
    state_.export_timing_ring.resize(16);
    state_.receive_timing_ring.resize(16);
    for (LETTimingSlot& slot : state_.export_timing_ring) {
        CUDA_CHECK(cudaEventCreate(&slot.start)); CUDA_CHECK(cudaEventCreate(&slot.stop));
    }
    for (LETTimingSlot& slot : state_.receive_timing_ring) {
        CUDA_CHECK(cudaEventCreate(&slot.start)); CUDA_CHECK(cudaEventCreate(&slot.stop));
    }

    // Start with no payload reservation.  The first export is a count-only
    // pass (capacity guards prevent writes), then only traffic-bearing peer
    // segments grow and retry.  This is essential for zero-traffic GPUs.
    (void)max_particles;
    (void)max_nodes;
    const int32_t per_peer_nodes = 0;
    const int32_t per_peer_particles = 0;

    // Exact counters are deliberately separate from capacity.  The kernel
    // increments them even when a high-water segment is too small, so a retry
    // can grow only the affected segment without publishing clipped payload.
    CUDA_CHECK(cudaMalloc(&state_.d_let_node_counts,
                          num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_let_particle_counts,
                          num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_descriptor_count, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_export_status, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_mask_node_counts, 256 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_mask_write_counts, (256 + 16) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_mask_offsets, 256 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_mask_shared, 256 * sizeof(uint8_t)));

    // --- Host-pinned count staging ---
    CUDA_CHECK(cudaMallocHost(&state_.h_pinned_my_send_node_counts,
                              num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&state_.h_pinned_my_send_particle_counts,
                              num_gpus * sizeof(int32_t)));

    // Per-peer scalar receive count staging; payloads live only in the direct
    // aggregate FMM arenas below.
    state_.peers = new LETPushPerPeer[num_gpus];
    for (int32_t p = 0; p < num_gpus; p++) {
        state_.peers[p].max_nodes = per_peer_nodes;
        state_.peers[p].max_particles = per_peer_particles;
        CUDA_CHECK(cudaMallocHost(&state_.peers[p].h_pinned_recv_node_count,
                                  sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&state_.peers[p].h_pinned_recv_particle_count,
                                  sizeof(int32_t)));
        state_.peers[p].max_recv_nodes = per_peer_nodes;
        state_.peers[p].max_recv_particles = per_peer_particles;
    }

    // One send and one receive allocation, segmented by peer.  Initial
    // high-water marks are intentionally conservative; they are independently
    // growable at a phase boundary by the producer retry path.
    const int32_t node_total = num_gpus * per_peer_nodes;
    const int32_t particle_total = num_gpus * per_peer_particles;
    auto host_array = [num_gpus]() { return new int32_t[num_gpus](); };
    state_.h_fmm_send_node_offsets = host_array(); state_.h_fmm_send_particle_offsets = host_array();
    state_.h_fmm_recv_node_offsets = host_array(); state_.h_fmm_recv_particle_offsets = host_array();
    state_.h_fmm_send_node_capacities = host_array(); state_.h_fmm_send_particle_capacities = host_array();
    state_.h_fmm_recv_node_capacities = host_array(); state_.h_fmm_recv_particle_capacities = host_array();
    for (int p = 0; p < num_gpus; ++p) {
        state_.h_fmm_send_node_offsets[p] = state_.h_fmm_recv_node_offsets[p] = p * per_peer_nodes;
        state_.h_fmm_send_particle_offsets[p] = state_.h_fmm_recv_particle_offsets[p] = p * per_peer_particles;
        state_.h_fmm_send_node_capacities[p] = state_.h_fmm_recv_node_capacities[p] = per_peer_nodes;
        state_.h_fmm_send_particle_capacities[p] = state_.h_fmm_recv_particle_capacities[p] = per_peer_particles;
    }
    if (node_total) {
        CUDA_CHECK(cudaMalloc(&state_.d_fmm_send_nodes, node_total * sizeof(FMMLETNode)));
        CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_nodes, node_total * sizeof(FMMLETNode)));
    }
    if (particle_total) {
        CUDA_CHECK(cudaMalloc(&state_.d_fmm_send_particles, particle_total * sizeof(LETParticle)));
        CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_particles, particle_total * sizeof(LETParticle)));
    }
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_send_node_offsets, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_send_particle_offsets, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_node_offsets, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_particle_offsets, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_node_counts, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_fmm_recv_particle_counts, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(state_.d_fmm_send_node_offsets, state_.h_fmm_send_node_offsets, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state_.d_fmm_send_particle_offsets, state_.h_fmm_send_particle_offsets, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state_.d_fmm_recv_node_offsets, state_.h_fmm_recv_node_offsets, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state_.d_fmm_recv_particle_offsets, state_.h_fmm_recv_particle_offsets, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    // Capacities are kernel inputs; reuse the historical receive count slots
    // as no legacy FMM payload path remains.
    CUDA_CHECK(cudaMalloc(&state_.d_recv_node_counts, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&state_.d_recv_particle_counts, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(state_.d_recv_node_counts, state_.h_fmm_send_node_capacities, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state_.d_recv_particle_counts, state_.h_fmm_send_particle_capacities, num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice));
    // Two payload arenas plus all live per-peer capacity/offset/count metadata
    // (device counters and host-pinned count exchange included).
    state_.fmm_arena_bytes = static_cast<size_t>(22 * num_gpus + 1) * sizeof(int32_t);
    g_fmm_send_nodes[gpu_id] = state_.d_fmm_send_nodes;
    g_fmm_send_particles[gpu_id] = state_.d_fmm_send_particles;
    g_fmm_sender_plans[gpu_id] = &state_.fmm_sender_plan;

    state_.initialized = true;
}

// =============================================================================
// Shutdown
// =============================================================================

void LETCommunicator::shutdown() {
    if (!state_.initialized) return;

    if (let_compute_ready_) cudaEventDestroy(let_compute_ready_);
    if (let_counts_ready_) cudaEventDestroy(let_counts_ready_);
    let_compute_ready_ = nullptr;
    let_counts_ready_ = nullptr;
    for (LETTimingSlot& slot : state_.export_timing_ring) { cudaEventDestroy(slot.start); cudaEventDestroy(slot.stop); }
    for (LETTimingSlot& slot : state_.receive_timing_ring) { cudaEventDestroy(slot.start); cudaEventDestroy(slot.stop); }
    state_.export_timing_ring.clear(); state_.receive_timing_ring.clear(); state_.peer_link_classes.clear();
    coordinator_ = nullptr;

    // --- Unregister from shared host arrays ---
    if (state_.gpu_id >= 0 && state_.gpu_id < MAX_MULTI_GPUS) {
        g_fmm_send_nodes[state_.gpu_id] = nullptr;
        g_fmm_send_particles[state_.gpu_id] = nullptr;
        g_fmm_sender_plans[state_.gpu_id] = nullptr;
    }

    // --- Destroy CUDA streams ---
    if (state_.compute_stream) {
        cudaStreamDestroy(state_.compute_stream);
        state_.compute_stream = nullptr;
    }
    if (state_.comm_stream) {
        cudaStreamDestroy(state_.comm_stream);
        state_.comm_stream = nullptr;
    }

    // --- Free combined send-side arrays ---
    cudaFree(state_.d_all_let_centers);
    cudaFree(state_.d_all_let_sizes);
    cudaFree(state_.d_all_let_multipoles);
    cudaFree(state_.d_all_let_particles);
    cudaFree(state_.d_let_node_counts);
    cudaFree(state_.d_let_particle_counts);
    state_.d_all_let_centers = nullptr;
    state_.d_all_let_sizes = nullptr;
    state_.d_all_let_multipoles = nullptr;
    state_.d_all_let_particles = nullptr;
    state_.d_let_node_counts = nullptr;
    cudaFree(state_.d_fmm_descriptors);
    cudaFree(state_.d_fmm_descriptor_count);
    cudaFree(state_.d_fmm_export_status);
    cudaFree(state_.d_fmm_mask_node_counts);
    cudaFree(state_.d_fmm_mask_write_counts);
    cudaFree(state_.d_fmm_mask_offsets);
    cudaFree(state_.d_fmm_mask_shared);
    state_.d_fmm_descriptors = nullptr;
    state_.d_fmm_descriptor_count = nullptr;
    state_.d_fmm_export_status = nullptr;
    state_.d_fmm_mask_node_counts = nullptr;
    state_.d_fmm_mask_write_counts = nullptr;
    state_.d_fmm_mask_offsets = nullptr;
    state_.d_fmm_mask_shared = nullptr;
    state_.d_let_particle_counts = nullptr;

    // --- Free combined receive-side arrays ---
    cudaFree(state_.d_all_recv_centers);
    cudaFree(state_.d_all_recv_sizes);
    cudaFree(state_.d_all_recv_multipoles);
    cudaFree(state_.d_all_recv_particles);
    cudaFree(state_.d_recv_node_counts);
    cudaFree(state_.d_recv_particle_counts);
    state_.d_all_recv_centers = nullptr;
    state_.d_all_recv_sizes = nullptr;
    state_.d_all_recv_multipoles = nullptr;
    state_.d_all_recv_particles = nullptr;
    state_.d_recv_node_counts = nullptr;
    state_.d_recv_particle_counts = nullptr;

    // --- Free host-pinned count staging ---
    cudaFreeHost(state_.h_pinned_my_send_node_counts);
    cudaFreeHost(state_.h_pinned_my_send_particle_counts);
    state_.h_pinned_my_send_node_counts = nullptr;
    state_.h_pinned_my_send_particle_counts = nullptr;

    // --- Free per-peer buffers ---
    if (state_.peers) {
        for (int32_t p = 0; p < state_.num_gpus; p++) {
            cudaFree(state_.peers[p].d_send_nodes);
            cudaFree(state_.peers[p].d_send_particles);
            cudaFree(state_.peers[p].d_send_node_count);
            cudaFree(state_.peers[p].d_send_particle_count);

            cudaFree(state_.peers[p].d_recv_nodes);
            cudaFree(state_.peers[p].d_recv_particles);
            cudaFreeHost(state_.peers[p].h_pinned_recv_node_count);
            cudaFreeHost(state_.peers[p].h_pinned_recv_particle_count);
        }
        delete[] state_.peers;
        state_.peers = nullptr;
    }

    cudaFree(state_.d_fmm_send_nodes); cudaFree(state_.d_fmm_send_particles);
    cudaFree(state_.d_fmm_recv_nodes); cudaFree(state_.d_fmm_recv_particles);
    cudaFree(state_.d_fmm_send_node_offsets); cudaFree(state_.d_fmm_send_particle_offsets);
    cudaFree(state_.d_fmm_recv_node_offsets); cudaFree(state_.d_fmm_recv_particle_offsets);
    cudaFree(state_.d_fmm_recv_node_counts); cudaFree(state_.d_fmm_recv_particle_counts);
    cudaFree(state_.d_fmm_source_work);
    delete[] state_.h_fmm_send_node_offsets; delete[] state_.h_fmm_send_particle_offsets;
    delete[] state_.h_fmm_recv_node_offsets; delete[] state_.h_fmm_recv_particle_offsets;
    delete[] state_.h_fmm_send_node_capacities; delete[] state_.h_fmm_send_particle_capacities;
    delete[] state_.h_fmm_recv_node_capacities; delete[] state_.h_fmm_recv_particle_capacities;
    state_.d_fmm_send_nodes = nullptr; state_.d_fmm_send_particles = nullptr;
    state_.d_fmm_recv_nodes = nullptr; state_.d_fmm_recv_particles = nullptr;
    state_.d_fmm_send_node_offsets = nullptr; state_.d_fmm_send_particle_offsets = nullptr;
    state_.d_fmm_recv_node_offsets = nullptr; state_.d_fmm_recv_particle_offsets = nullptr;
    state_.d_fmm_recv_node_counts = nullptr; state_.d_fmm_recv_particle_counts = nullptr;
    state_.d_fmm_source_work = nullptr;
    state_.fmm_source_work_capacity = state_.fmm_source_work_count = 0;
    state_.fmm_arena_bytes = 0;

    state_.initialized = false;
}

// =============================================================================
// FMM Push LETs
// =============================================================================

uint64_t LETCommunicator::push_empty_lets_fmm(cudaStream_t stream) {
    if (is_single_gpu()) return 0;
    cudaStream_t s = stream ? stream : state_.compute_stream;
    const uint64_t epoch = next_fmm_let_epoch_++;
    state_.last_fmm_send_grew = false;
    state_.last_fmm_send_retried = false;
    state_.fmm_source_work_count = 0;
    state_.fmm_sharing_summary = {};
    if (!coordinator_->prepare_producer(PeerPhase::Let, state_.gpu_id, epoch, s)) {
        coordinator_->abort();
        return 0;
    }
    CUDA_CHECK(cudaMemsetAsync(state_.d_let_node_counts, 0,
                               state_.num_gpus * sizeof(int32_t), s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_let_particle_counts, 0,
                               state_.num_gpus * sizeof(int32_t), s));
    CUDA_CHECK(cudaEventRecord(let_compute_ready_, s));
    CUDA_CHECK(cudaStreamWaitEvent(state_.comm_stream, let_compute_ready_, 0));
    CUDA_CHECK(cudaMemcpyAsync(state_.h_pinned_my_send_node_counts,
                               state_.d_let_node_counts,
                               state_.num_gpus * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, state_.comm_stream));
    CUDA_CHECK(cudaMemcpyAsync(state_.h_pinned_my_send_particle_counts,
                               state_.d_let_particle_counts,
                               state_.num_gpus * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, state_.comm_stream));
    CUDA_CHECK(cudaEventRecord(let_counts_ready_, state_.comm_stream));
    CUDA_CHECK(cudaEventSynchronize(let_counts_ready_));

    // Empty exports have no bucket candidates and no copies.  Keep this
    // explicit so receivers and telemetry never observe a stale prior epoch.
    state_.fmm_sender_plan = {};
    state_.fmm_sharing_summary = {};
    if (!coordinator_->publish(PeerPhase::Let, state_.gpu_id, epoch,
                               state_.comm_stream)) {
        coordinator_->abort();
        return 0;
    }
    return epoch;
}

uint64_t LETCommunicator::push_lets_fmm(
    const FMMLETSourceView& source,
    const double* pos_x, const double* pos_y,
    const double* pos_z, const float* mass,
    cudaStream_t stream
) {
    const bool invalid_source = !source.unified_nodes || !source.children_compact
        || !source.multipoles || !source.sorted_indices || !source.level_start
        || !source.level_count || !source.level_list || !source.metadata
        || source.node_capacity <= 0;
    if (invalid_source) {
        if (coordinator_) coordinator_->abort();
        throw std::invalid_argument(
            "push_lets_fmm requires complete FMMLETSourceView with positive node_capacity");
    }
    if (is_single_gpu()) return 0;

    // The public view retains only header-safe types.  These casts stay in
    // this translation unit, where the concrete FMM headers are available.
    const auto* nodes = static_cast<const ::body_fitted_fmm::UnifiedNode*>(source.unified_nodes);
    const auto* multipoles = static_cast<const fmm::MultipoleExpansion*>(source.multipoles);
    const auto* metadata = static_cast<const ::body_fitted_fmm::PersistentTreeMetadata*>(source.metadata);

    cudaStream_t s = stream ? stream : state_.compute_stream;
    const uint64_t epoch = next_fmm_let_epoch_++;
    state_.last_fmm_send_grew = false;
    state_.last_fmm_send_retried = false;
    state_.fmm_sharing_summary = {};

    // Source attribution is O(N_nodes), independent of destination count and
    // wire capacity.  Grow before the count-only pass so every retry writes
    // the same source-indexed records.
    if (source.node_capacity > state_.fmm_source_work_capacity) {
        const int32_t capacity = grow_let_capacity(source.node_capacity,
            state_.fmm_source_work_capacity, state_.gpu_id, -1, "source-work");
        const int32_t old_capacity = state_.fmm_source_work_capacity;
        LETSourceWork* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(&replacement, static_cast<size_t>(capacity) * sizeof(LETSourceWork)));
        CUDA_CHECK(cudaFree(state_.d_fmm_source_work));
        state_.d_fmm_source_work = replacement;
        state_.fmm_source_work_capacity = capacity;
        state_.fmm_arena_bytes += static_cast<size_t>(capacity - old_capacity) * sizeof(LETSourceWork);
        state_.last_fmm_send_grew = true;
    }
    // At most one discovery record is emitted for each visited source node.
    // Allocate once at the same grow-only boundary as source attribution.
    if (source.node_capacity > state_.fmm_descriptor_capacity) {
        LETExportDescriptor* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(&replacement,
                              static_cast<size_t>(source.node_capacity) * sizeof(LETExportDescriptor)));
        CUDA_CHECK(cudaFree(state_.d_fmm_descriptors));
        state_.d_fmm_descriptors = replacement;
        state_.fmm_descriptor_capacity = source.node_capacity;
        state_.last_fmm_send_grew = true;
    }
    state_.fmm_source_work_count = source.node_capacity;

    // The coordinator waits only on acknowledgements for this producer's
    // single reusable LET buffer.  It enqueues the wait on `s`, so the next
    // tree walk cannot overwrite data a peer has not consumed.
    if (!coordinator_->prepare_producer(PeerPhase::Let, state_.gpu_id, epoch, s)) {
        coordinator_->abort();
        return 0;
    }
    const bool sample_timing = (epoch % state_.timing_sample_interval) == 0;
    // --- 1. Zero send counters ---
    CUDA_CHECK(cudaMemsetAsync(state_.d_let_node_counts, 0,
                                state_.num_gpus * sizeof(int32_t), s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_let_particle_counts, 0,
                                state_.num_gpus * sizeof(int32_t), s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_fmm_descriptor_count, 0, sizeof(int32_t), s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_fmm_export_status, 0, sizeof(int32_t), s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_fmm_mask_node_counts, 0, 256 * sizeof(int32_t), s));
    if (source.node_capacity > 0) {
        CUDA_CHECK(cudaMemsetAsync(state_.d_fmm_source_work, 0,
                                    static_cast<size_t>(source.node_capacity) * sizeof(LETSourceWork), s));
    }

    // Discovery is count-only: it records each source node exactly once with
    // its exact destination mask.  The host only waits for these scalar bins
    // to choose a compact physical layout; payloads remain stream ordered.
    const int block_size = 128;

    LETTimingSlot* export_timing = sample_timing ? acquire_timing_slot(state_.export_timing_ring) : nullptr;
    if (export_timing) CUDA_CHECK(cudaEventRecord(export_timing->start, s));
    // Capacity bounds every launch and device traversal; metadata supplies
    // the current live tree count without relying on stale host mirrors.
    discover_let_fmm_kernel<<<std::max(1, (source.node_capacity + block_size - 1) / block_size), block_size, 0, s>>>(
        nodes, source.children_compact, source.sorted_indices,
        source.level_start, source.level_count, source.level_list, metadata,
        source.node_capacity, state_.start_level, state_.theta, state_.num_gpus, state_.gpu_id,
        state_.d_fmm_descriptors, state_.d_fmm_descriptor_count,
        state_.fmm_descriptor_capacity, state_.d_fmm_mask_node_counts,
        state_.d_let_particle_counts, state_.d_fmm_source_work,
        state_.d_fmm_export_status);
    CUDA_CHECK(cudaGetLastError());

    // Bridge the caller's FMM stream to the dedicated communication stream.
    // This is an event dependency rather than a host/device-wide fence.
    CUDA_CHECK(cudaEventRecord(let_compute_ready_, s));
    CUDA_CHECK(cudaStreamWaitEvent(state_.comm_stream, let_compute_ready_, 0));

    // --- 4. Copy send counts to host-pinned ---
    CUDA_CHECK(cudaMemcpyAsync(state_.h_pinned_my_send_node_counts,
                                state_.d_let_node_counts,
                                state_.num_gpus * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, state_.comm_stream));
    CUDA_CHECK(cudaMemcpyAsync(state_.h_pinned_my_send_particle_counts,
                                state_.d_let_particle_counts,
                                state_.num_gpus * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, state_.comm_stream));
    int32_t descriptor_count = 0;
    int32_t export_status = 0;
    int32_t mask_counts[256]{};
    CUDA_CHECK(cudaMemcpyAsync(&descriptor_count, state_.d_fmm_descriptor_count,
                               sizeof(descriptor_count), cudaMemcpyDeviceToHost, state_.comm_stream));
    CUDA_CHECK(cudaMemcpyAsync(mask_counts, state_.d_fmm_mask_node_counts,
                               sizeof(mask_counts), cudaMemcpyDeviceToHost, state_.comm_stream));
    CUDA_CHECK(cudaMemcpyAsync(&export_status, state_.d_fmm_export_status,
                               sizeof(export_status), cudaMemcpyDeviceToHost, state_.comm_stream));

    // A narrow host wait is necessary only for the scalar counts that decide
    // each P2P copy size.  Payload completion stays asynchronous.
    CUDA_CHECK(cudaEventRecord(let_counts_ready_, state_.comm_stream));
    CUDA_CHECK(cudaEventSynchronize(let_counts_ready_));

    if (export_status != 0) {
        coordinator_->abort();
        throw std::runtime_error("LET FMM export validation failed on GPU "
                                 + std::to_string(state_.gpu_id)
                                 + " (status " + std::to_string(export_status) + ")");
    }
    if (descriptor_count < 0 || descriptor_count > state_.fmm_descriptor_capacity) {
        coordinator_->abort();
        throw std::runtime_error("LET descriptor discovery overflow on GPU " + std::to_string(state_.gpu_id));
    }

    // Rank only masks that can save a serialization.  The selected buckets
    // occupy one copy apiece at the beginning of the node arena; every other
    // node is appended once per consumer in its residual region.
    std::vector<std::pair<int64_t, int>> ranked;
    for (int m = 1; m < 256; ++m) {
        const int fanout = __builtin_popcount(static_cast<unsigned>(m));
        if (mask_counts[m] < 0) throw std::runtime_error("negative LET mask count");
        const int64_t saved = static_cast<int64_t>(fanout - 1) * mask_counts[m] * sizeof(FMMLETNode);
        if (mask_counts[m] > 0 && fanout >= 2 && saved >= static_cast<int64_t>(state_.min_shared_saved_bytes)) ranked.emplace_back(saved, m);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    uint8_t selected[256]{};
    for (int i = 0; i < std::min<int>(state_.max_shared_buckets, ranked.size()); ++i) selected[ranked[i].second] = 1;
    LETSenderSegmentPlan plan{};
    int64_t node_total = 0, particle_total = 0;
    const int64_t i32max = std::numeric_limits<int32_t>::max();
    for (int m = 1; m < 256; ++m) if (selected[m]) {
        if (node_total > i32max || mask_counts[m] < 0 || node_total + mask_counts[m] > i32max) throw std::runtime_error("LET shared layout exceeds int32");
        plan.shared_offset[m] = static_cast<int32_t>(node_total); plan.shared_count[m] = mask_counts[m]; node_total += mask_counts[m];
    }
    for (int p = 0; p < state_.num_gpus; ++p) {
        if (node_total > i32max || particle_total > i32max) throw std::runtime_error("LET layout offset exceeds int32");
        plan.residual_offset[p] = static_cast<int32_t>(node_total);
        int64_t residual = 0;
        for (int m = 1; m < 256; ++m) if (!selected[m] && (m & (1 << p))) residual += mask_counts[m];
        if (residual > std::numeric_limits<int32_t>::max()) throw std::runtime_error("LET residual exceeds int32");
        if (node_total + residual > i32max) throw std::runtime_error("LET residual layout exceeds int32");
        plan.residual_count[p] = static_cast<int32_t>(residual); node_total += residual;
        plan.particle_offset[p] = static_cast<int32_t>(particle_total);
        plan.particle_count[p] = state_.h_pinned_my_send_particle_counts[p];
        if (plan.particle_count[p] < 0 || particle_total + plan.particle_count[p] > i32max) throw std::runtime_error("LET particle layout exceeds int32");
        particle_total += plan.particle_count[p];
    }
    if (node_total > std::numeric_limits<int32_t>::max() || particle_total > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("Aggregate LET send arena exceeds int32 indexing on GPU " + std::to_string(state_.gpu_id));
    for (int p = 0; p < state_.num_gpus; ++p) {
        int64_t delivered = plan.residual_count[p];
        for (int m = 1; m < 256; ++m) if (selected[m] && (m & (1 << p))) delivered += plan.shared_count[m];
        state_.h_pinned_my_send_node_counts[p] = static_cast<int32_t>(delivered);
        // Compatibility metadata remains logical per-consumer capacity; the
        // physical allocation itself is the aggregate capacity below.
        state_.h_fmm_send_node_capacities[p] = static_cast<int32_t>(delivered);
        state_.h_fmm_send_particle_capacities[p] = plan.particle_count[p];
        state_.h_fmm_send_node_offsets[p] = plan.residual_offset[p];
        state_.h_fmm_send_particle_offsets[p] = plan.particle_offset[p];
    }
    state_.fmm_sender_plan = plan;
    const int32_t required_node_total = static_cast<int32_t>(node_total), required_particle_total = static_cast<int32_t>(particle_total);
    const int32_t old_node_total = state_.fmm_send_node_capacity_total, old_particle_total = state_.fmm_send_particle_capacity_total;
    const bool grow_nodes = required_node_total > state_.fmm_send_node_capacity_total;
    const bool grow_particles = required_particle_total > state_.fmm_send_particle_capacity_total;
    if (grow_nodes) state_.fmm_send_node_capacity_total = grow_let_capacity(required_node_total, old_node_total, state_.gpu_id, -1, "aggregate-send-node");
    if (grow_particles) state_.fmm_send_particle_capacity_total = grow_let_capacity(required_particle_total, old_particle_total, state_.gpu_id, -1, "aggregate-send-particle");
    if (grow_nodes || grow_particles) {
        FMMLETNode* new_nodes = nullptr; LETParticle* new_particles = nullptr;
        if (state_.fmm_send_node_capacity_total) CUDA_CHECK(cudaMalloc(&new_nodes, static_cast<size_t>(state_.fmm_send_node_capacity_total) * sizeof(FMMLETNode)));
        if (state_.fmm_send_particle_capacity_total) CUDA_CHECK(cudaMalloc(&new_particles, static_cast<size_t>(state_.fmm_send_particle_capacity_total) * sizeof(LETParticle)));
        CUDA_CHECK(cudaFree(state_.d_fmm_send_nodes)); CUDA_CHECK(cudaFree(state_.d_fmm_send_particles));
        state_.d_fmm_send_nodes = new_nodes; state_.d_fmm_send_particles = new_particles;
        g_fmm_send_nodes[state_.gpu_id] = new_nodes; g_fmm_send_particles[state_.gpu_id] = new_particles;
        state_.last_fmm_send_grew = true;
    }
    // Keep telemetry tied to allocations, not this epoch's compact payload.
    int64_t recv_node_capacity = 0, recv_particle_capacity = 0;
    for (int p = 0; p < state_.num_gpus; ++p) {
        recv_node_capacity += state_.h_fmm_recv_node_capacities[p];
        recv_particle_capacity += state_.h_fmm_recv_particle_capacities[p];
    }
    state_.fmm_arena_bytes = static_cast<size_t>(state_.fmm_send_node_capacity_total) * sizeof(FMMLETNode)
                           + static_cast<size_t>(state_.fmm_send_particle_capacity_total) * sizeof(LETParticle)
                           + static_cast<size_t>(recv_node_capacity) * sizeof(FMMLETNode)
                           + static_cast<size_t>(recv_particle_capacity) * sizeof(LETParticle)
                           + static_cast<size_t>(state_.fmm_source_work_capacity) * sizeof(LETSourceWork)
                           + static_cast<size_t>(state_.fmm_descriptor_capacity) * sizeof(LETExportDescriptor)
                           + static_cast<size_t>(256 + 272 + 256 + 2 + 20 * state_.num_gpus) * sizeof(int32_t)
                           + 256 * sizeof(uint8_t);
    for (LETTimingSlot& slot : state_.export_timing_ring) {
        if (slot.pending && slot.record.epoch == epoch) {
            slot.record.node_records = required_node_total;
            slot.record.particle_records = required_particle_total;
            slot.record.bytes = static_cast<size_t>(required_node_total) * sizeof(FMMLETNode)
                              + static_cast<size_t>(required_particle_total) * sizeof(LETParticle);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_mask_offsets, plan.shared_offset, sizeof(plan.shared_offset), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_mask_shared, selected, sizeof(selected), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_send_node_offsets, plan.residual_offset, state_.num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_send_particle_offsets, plan.particle_offset, state_.num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemsetAsync(state_.d_fmm_mask_write_counts, 0, (256 + 16) * sizeof(int32_t), s));
    if (descriptor_count) materialize_let_fmm_kernel<<<(descriptor_count + block_size - 1) / block_size, block_size, 0, s>>>(
        state_.d_fmm_descriptors, descriptor_count, nodes, multipoles, pos_x, pos_y, pos_z, mass,
        source.sorted_indices, metadata, source.node_capacity, state_.d_fmm_send_nodes, state_.d_fmm_send_particles,
        state_.d_fmm_mask_offsets, state_.d_fmm_mask_shared, state_.d_fmm_send_node_offsets,
        state_.d_fmm_send_particle_offsets, state_.d_fmm_mask_write_counts, state_.d_fmm_source_work, state_.num_gpus);
    CUDA_CHECK(cudaGetLastError());
    // The count exchange above only orders discovery.  Re-record the bridge
    // after materialization so publication cannot expose a partially-filled
    // arena to a receiver's P2P pull.
    CUDA_CHECK(cudaEventRecord(let_compute_ready_, s));
    CUDA_CHECK(cudaStreamWaitEvent(state_.comm_stream, let_compute_ready_, 0));
    if (export_timing) { CUDA_CHECK(cudaEventRecord(export_timing->stop, s)); export_timing->pending = true;
        export_timing->record = {LETTimingPhase::ExportKernel, epoch, -1, LinkClass::GenericPcieP2P,
            required_node_total, required_particle_total, static_cast<size_t>(required_node_total)*sizeof(FMMLETNode)+static_cast<size_t>(required_particle_total)*sizeof(LETParticle), 0.0f, state_.last_fmm_send_grew, false, !state_.last_fmm_send_grew}; }
    state_.fmm_sharing_summary = {};
    for (int m=1; m<256; ++m) if (mask_counts[m]) { const int fanout=__builtin_popcount(static_cast<unsigned>(m));
        if (fanout >= 2) ++state_.fmm_sharing_summary.candidate_mask_count;
        if (selected[m]) { ++state_.fmm_sharing_summary.shared_mask_count; state_.fmm_sharing_summary.shared_node_records += mask_counts[m]; state_.fmm_sharing_summary.avoided_serialization_bytes += static_cast<size_t>(fanout-1)*mask_counts[m]*sizeof(FMMLETNode); }
    }
    for (int p=0; p<state_.num_gpus; ++p) if (p != state_.gpu_id) { state_.fmm_sharing_summary.delivered_node_records += state_.h_pinned_my_send_node_counts[p]; state_.fmm_sharing_summary.particle_records += plan.particle_count[p]; state_.fmm_sharing_summary.residual_node_records += plan.residual_count[p]; }
    state_.fmm_sharing_summary.serialized_node_records = required_node_total;
    for (int m = 1; m < 256; ++m) state_.fmm_sharing_summary.unique_node_records += mask_counts[m];
    state_.fmm_sharing_summary.copy_operations = 0;
    for (int p = 0; p < state_.num_gpus; ++p) if (p != state_.gpu_id) {
        for (int m = 1; m < 256; ++m)
            if (selected[m] && (m & (1 << p)) && plan.shared_count[m]) ++state_.fmm_sharing_summary.copy_operations;
        if (plan.residual_count[p]) ++state_.fmm_sharing_summary.copy_operations;
        if (plan.particle_count[p]) ++state_.fmm_sharing_summary.copy_operations;
    }

    // Publish means the local send arena and its pinned exact counts are
    // ready.  Consumers pull only their segment after waiting on this event.
    if (!coordinator_->publish(PeerPhase::Let, state_.gpu_id, epoch, state_.comm_stream)) {
        coordinator_->abort();
        return 0;
    }
    return epoch;
}

// =============================================================================
// FMM Merge Received LETs
// =============================================================================

bool LETCommunicator::wait_and_stage_fmm_recv_counts(uint64_t epoch, cudaStream_t stream) {
    if (is_single_gpu()) return true;
    cudaStream_t s = stream ? stream : state_.compute_stream;
    state_.last_fmm_recv_grew = false;
    const bool sample_timing = (epoch % state_.timing_sample_interval) == 0;

    for (int32_t p = 0; p < state_.num_gpus; p++) {
        if (p == state_.gpu_id) {
            CUDA_CHECK(cudaMemsetAsync(&state_.d_fmm_recv_node_counts[p], 0,
                                       sizeof(int32_t), s));
            CUDA_CHECK(cudaMemsetAsync(&state_.d_fmm_recv_particle_counts[p], 0,
                                       sizeof(int32_t), s));
        } else {
            // wait_for_published both wires the peer's P2P completion event
            // into `s` and provides the host acquire fence for its pinned
            // counts.  No OpenMP rendezvous is required.
            if (!coordinator_->wait_for_published(
                    PeerPhase::Let, p, state_.gpu_id, epoch, s)) {
                coordinator_->abort();
                return false;
            }
            const LETSenderSegmentPlan& plan = *g_fmm_sender_plans[p];
            int64_t nodes = plan.residual_count[state_.gpu_id];
            for (int m = 1; m < 256; ++m)
                if ((m & (1 << state_.gpu_id)) && plan.shared_count[m]) nodes += plan.shared_count[m];
            if (nodes > std::numeric_limits<int32_t>::max())
                throw std::runtime_error("LET received logical node count exceeds int32");
            *state_.peers[p].h_pinned_recv_node_count = static_cast<int32_t>(nodes);
            *state_.peers[p].h_pinned_recv_particle_count = plan.particle_count[state_.gpu_id];
            CUDA_CHECK(cudaMemcpyAsync(&state_.d_fmm_recv_node_counts[p],
                                       state_.peers[p].h_pinned_recv_node_count,
                                       sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(&state_.d_fmm_recv_particle_counts[p],
                                       state_.peers[p].h_pinned_recv_particle_count,
                                       sizeof(int32_t), cudaMemcpyHostToDevice, s));
        }
    }

    // Exact prefix offsets make the final receive arena compact for this
    // epoch.  Capacity remains a per-peer high-water mark for telemetry and
    // grow-only allocation decisions, but never dictates payload placement.
    int64_t node_total = 0, particle_total = 0;
    int64_t node_capacity_total = 0, particle_capacity_total = 0;
    int64_t old_node_capacity_total = 0, old_particle_capacity_total = 0;
    bool grow = false;
    for (int p = 0; p < state_.num_gpus; ++p) {
        const int n = (p == state_.gpu_id) ? 0 : *state_.peers[p].h_pinned_recv_node_count;
        const int q = (p == state_.gpu_id) ? 0 : *state_.peers[p].h_pinned_recv_particle_count;
        if (node_total > std::numeric_limits<int32_t>::max()
            || particle_total > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Aggregate LET receive offsets exceed int32 indexing on GPU "
                                     + std::to_string(state_.gpu_id));
        }
        state_.h_fmm_recv_node_offsets[p] = static_cast<int32_t>(node_total);
        state_.h_fmm_recv_particle_offsets[p] = static_cast<int32_t>(particle_total);
        node_total += n; particle_total += q;
        old_node_capacity_total += state_.h_fmm_recv_node_capacities[p];
        old_particle_capacity_total += state_.h_fmm_recv_particle_capacities[p];
        if (n > state_.h_fmm_recv_node_capacities[p]) {
            state_.h_fmm_recv_node_capacities[p] = grow_let_capacity(
                n, state_.h_fmm_recv_node_capacities[p], state_.gpu_id, p,
                "receive-node");
            grow = true;
        }
        if (q > state_.h_fmm_recv_particle_capacities[p]) {
            state_.h_fmm_recv_particle_capacities[p] = grow_let_capacity(
                q, state_.h_fmm_recv_particle_capacities[p], state_.gpu_id, p,
                "receive-particle");
            grow = true;
        }
        node_capacity_total += state_.h_fmm_recv_node_capacities[p];
        particle_capacity_total += state_.h_fmm_recv_particle_capacities[p];
    }
    if (node_total > std::numeric_limits<int32_t>::max()
        || particle_total > std::numeric_limits<int32_t>::max()
        || node_capacity_total > std::numeric_limits<int32_t>::max()
        || particle_capacity_total > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Aggregate LET receive arena exceeds int32 indexing on GPU "
                                 + std::to_string(state_.gpu_id));
    }
    if (grow) {
        state_.last_fmm_recv_grew = true;
        // Keep the receive side symmetric with the producer: a payload type
        // with no traffic owns no allocation.  Besides avoiding cudaMalloc(0),
        // this keeps fmm_arena_bytes an exact allocation accounting rather
        // than charging a one-record sentinel arena.
        FMMLETNode* new_nodes = nullptr;
        LETParticle* new_particles = nullptr;
        if (node_capacity_total > 0) {
            CUDA_CHECK(cudaMalloc(&new_nodes,
                                  node_capacity_total * sizeof(FMMLETNode)));
        }
        if (particle_capacity_total > 0) {
            CUDA_CHECK(cudaMalloc(&new_particles,
                                  particle_capacity_total * sizeof(LETParticle)));
        }
        CUDA_CHECK(cudaFree(state_.d_fmm_recv_nodes));
        CUDA_CHECK(cudaFree(state_.d_fmm_recv_particles));
        state_.d_fmm_recv_nodes = new_nodes; state_.d_fmm_recv_particles = new_particles;
        state_.fmm_arena_bytes += (node_capacity_total - old_node_capacity_total) * sizeof(FMMLETNode)
                               + (particle_capacity_total - old_particle_capacity_total) * sizeof(LETParticle);
        log_let_growth(state_.gpu_id, "receive", old_node_capacity_total,
                       node_total, node_capacity_total,
                       old_particle_capacity_total, particle_total,
                       particle_capacity_total, state_.fmm_arena_bytes);
    }
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_recv_node_offsets, state_.h_fmm_recv_node_offsets,
                               state_.num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(state_.d_fmm_recv_particle_offsets, state_.h_fmm_recv_particle_offsets,
                               state_.num_gpus * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    int32_t* recv_nodes_host = new int32_t[state_.num_gpus]();
    int32_t* recv_particles_host = new int32_t[state_.num_gpus]();
    for (int p = 0; p < state_.num_gpus; ++p) {
        recv_nodes_host[p] = (p == state_.gpu_id) ? 0 : *state_.peers[p].h_pinned_recv_node_count;
        recv_particles_host[p] = (p == state_.gpu_id) ? 0 : *state_.peers[p].h_pinned_recv_particle_count;
        if (p == state_.gpu_id) continue;
        const size_t bytes = static_cast<size_t>(recv_nodes_host[p]) * sizeof(FMMLETNode)
                           + static_cast<size_t>(recv_particles_host[p]) * sizeof(LETParticle);
        LETTimingSlot* transfer_timing = (sample_timing && bytes)
            ? acquire_timing_slot(state_.receive_timing_ring) : nullptr;
        if (transfer_timing) CUDA_CHECK(cudaEventRecord(transfer_timing->start, s));
        const LETSenderSegmentPlan& plan = *g_fmm_sender_plans[p];
        int32_t dst_node = state_.h_fmm_recv_node_offsets[p];
        for (int m = 1; m < 256; ++m) {
            if (!(m & (1 << state_.gpu_id)) || !plan.shared_count[m]) continue;
            copy_peer_payload(state_.d_fmm_recv_nodes + dst_node, state_.gpu_id,
                g_fmm_send_nodes[p] + plan.shared_offset[m], p,
                static_cast<size_t>(plan.shared_count[m]) * sizeof(FMMLETNode), s);
            dst_node += plan.shared_count[m];
        }
        if (plan.residual_count[state_.gpu_id]) {
            copy_peer_payload(state_.d_fmm_recv_nodes + dst_node, state_.gpu_id,
                g_fmm_send_nodes[p] + plan.residual_offset[state_.gpu_id], p,
                static_cast<size_t>(plan.residual_count[state_.gpu_id]) * sizeof(FMMLETNode), s);
            dst_node += plan.residual_count[state_.gpu_id];
        }
        if (dst_node != state_.h_fmm_recv_node_offsets[p] + recv_nodes_host[p])
            throw std::runtime_error("LET sender plan/node count mismatch");
        if (recv_particles_host[p]) copy_peer_payload(state_.d_fmm_recv_particles + state_.h_fmm_recv_particle_offsets[p], state_.gpu_id,
            g_fmm_send_particles[p] + plan.particle_offset[state_.gpu_id], p,
            recv_particles_host[p] * sizeof(LETParticle), s);
        if (transfer_timing) {
            CUDA_CHECK(cudaEventRecord(transfer_timing->stop, s));
            transfer_timing->pending = true;
            transfer_timing->record = {LETTimingPhase::ReceiveTransfer, epoch, p,
                state_.peer_link_classes[p], recv_nodes_host[p], recv_particles_host[p],
                bytes, 0.0f, state_.last_fmm_recv_grew, false,
                !state_.last_fmm_recv_grew};
        }
    }
    delete[] recv_nodes_host;
    delete[] recv_particles_host;
    return true;
}

bool LETCommunicator::merge_fmm_lets(uint64_t epoch, cudaStream_t stream) {
    if (is_single_gpu()) return true;

    cudaStream_t s = stream ? stream : state_.compute_stream;
    // No merge kernel: the remote evaluator reads the aggregate receive
    // arena through its peer offsets and exact counts.

    // The merge has consumed every peer slot once this event is reached, so
    // each producer may safely reuse its single-buffer LET send region.
    for (int32_t p = 0; p < state_.num_gpus; ++p) {
        if (p != state_.gpu_id &&
            !coordinator_->acknowledge_consumed(PeerPhase::Let, p, state_.gpu_id, epoch, s)) {
            coordinator_->abort();
            return false;
        }
    }
    return true;
}

} // namespace cudadem

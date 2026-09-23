#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "../fmm/fmm_types.h"
#include "karras_radix_tree.cuh"
#include "grouped_interaction_arena.cuh"
#include "task_structs.cuh"
#include "tree_traits.cuh"
#include "morton_unified.cuh"

namespace body_fitted_fmm {

constexpr int PERSISTENT_TREE_MIN_DEPTH = 2;
constexpr int PERSISTENT_TREE_MAX_DEPTH = 20;
constexpr int PERSISTENT_TREE_DIMENSIONS = 1;

// The normal single-GPU path owns the complete timing/trial state on the
// device.  Distributed runs keep the same pre-captured depth graphs, but a
// host coordinator supplies the next launch depth instead.  Keeping this in
// the persistent context (rather than changing the graph topology) makes a
// mode transition safe at an existing synchronization boundary.
enum class AdaptiveDepthControlMode : uint8_t {
    Autonomous = 0,
    ExternalDistributed = 1,
};

struct ExternalDistributedDepthCommand {
    // A value of zero leaves the corresponding state unchanged.  launch_depth
    // must be non-zero for a command to select a graph.
    int selected_depth = 0;
    int candidate_depth = 0;
    int launch_depth = 0;
    uint64_t controller_epoch = 0;
};

struct AdaptiveDepthOptions {
    bool enabled = false;
    bool auto_initial_depth = false;
    int initial_depth = 0;
    int effective_max_depth = 0;
    bool startup_probe_complete = false;
    int min_depth = 6;
    int max_depth = 12;
    int trial_interval_evaluations = 32;
    int trial_warmup_evaluations = 1;
    int trial_samples = 4;
    float min_improvement_fraction = 0.02f;
    int confirmation_trials = 2;
    float capacity_headroom = 2.0f;
    float capacity_growth_trigger = 0.75f;
    AdaptiveDepthControlMode control_mode = AdaptiveDepthControlMode::Autonomous;
};

enum class AdaptiveDepthTransition : int {
    None = 0,
    InitialSelection = 1,
    TrialResult = 2,
    FasterNeighbor = 3,
    Confirmation = 4,
    CandidateRejected = 5,
    CapacityFallback = 6,
    CapacityGrowth = 7,
    EffectiveCeilingChanged = 8,
    FatalMinDepth = 9,
    NONE = None,
    INITIAL_SELECTION = InitialSelection,
    TRIAL_RESULT = TrialResult,
    FASTER_NEIGHBOR = FasterNeighbor,
    CONFIRMATION = Confirmation,
    CANDIDATE_REJECTED = CandidateRejected,
    CAPACITY_FALLBACK = CapacityFallback,
    CAPACITY_GROWTH = CapacityGrowth,
    EFFECTIVE_CEILING_CHANGED = EffectiveCeilingChanged,
    FATAL_MIN_DEPTH = FatalMinDepth,
};

// Device-resident state is deliberately POD: it is checkpointed without
// translating a host-only controller object and each tree dimension owns an
// independent copy.
struct AdaptiveDepthState {
    int selected_depth = 0;
    int candidate_depth = 0;
    int launch_depth = 0;
    int search_direction = 1;
    int trial_phase = 0;  // 0=selected, 1=warmup, 2=sampling
    int trial_warmups = 0;
    int trial_sample_count = 0;
    int confirmation_count = 0;
    int evaluations_since_trial = 0;
    int effective_max_depth = 0;
    int node_high_water = 0;
    int capacity_fallback_count = 0;
    uint64_t evaluation_counter = 0;
    uint64_t selected_time_ns = 0;
    uint64_t selected_sample_count = 0;
    uint64_t candidate_time_ns = 0;
    uint64_t last_elapsed_ns = 0;
    float last_relative_improvement = 0.0f;
};

struct AdaptiveDepthTelemetry {
    bool enabled = false;
    int tree_dimension = 3;
    int node_capacity = 0;
    float capacity_headroom = 0.0f;
    float capacity_growth_trigger = 0.0f;
    AdaptiveDepthState controller{};
};

// A compact, host-readable record for a distributed controller.  It is
// deliberately independent of CUDA events: the coordinator can pair it with
// an end-to-end gravity event that includes LET transfer and remote work.
struct ExternalDistributedDepthEvaluation {
    bool valid = false;
    uint64_t controller_epoch = 0;
    uint64_t evaluation_counter = 0;
    int tree_dimension = 0;
    int selected_depth = 0;
    int candidate_depth = 0;
    int launch_depth = 0;
    int effective_max_depth = 0;
    int node_count = 0;
    int node_capacity = 0;
    int capacity_fallback_count = 0;
};

struct AdaptiveDepthDiagnostic {
    uint64_t sequence = 0;
    uint64_t evaluation_counter = 0;
    double evaluation_time = 0.0;
    int tree_dimension = 0;
    int selected_depth = 0;
    int candidate_depth = 0;
    int search_direction = 0;
    int trial_phase = 0;
    double elapsed_ms = 0.0;
    double relative_improvement = 0.0;
    int node_count = 0;
    int node_capacity = 0;
    int effective_max_depth = 0;
    AdaptiveDepthTransition transition_reason = AdaptiveDepthTransition::None;
};

constexpr int ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE = 256;

struct AdaptiveDepthDiagnosticRing {
    AdaptiveDepthDiagnostic records[ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE]{};
    uint64_t next_sequence = 1;
    uint64_t consumer_sequence = 1;
    uint64_t dropped = 0;
};

// Decode a host snapshot of the device ring without mutating it.  The returned
// cursor is the first sequence not copied to `output`, so callers can publish
// it back to the device after preserving the selected records.  Keeping this
// helper independent of CUDA calls makes wrap/overflow ordering testable on
// the host and gives periodic and boundary drains identical semantics.
inline int extract_adaptive_depth_diagnostic_snapshot(
    const AdaptiveDepthDiagnosticRing& ring,
    AdaptiveDepthDiagnostic* output,
    int output_capacity,
    uint64_t& consumer_sequence)
{
    consumer_sequence = ring.consumer_sequence;
    if (!output || output_capacity <= 0) return 0;

    uint64_t begin = ring.consumer_sequence;
    const uint64_t end = ring.next_sequence;
    if (end > begin + ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE)
        begin = end - ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE;
    const uint64_t available = end > begin ? end - begin : 0;
    const uint64_t capacity = static_cast<uint64_t>(output_capacity);
    const int count = static_cast<int>(
        available < capacity ? available : capacity);
    for (int i = 0; i < count; ++i) {
        const uint64_t sequence = begin + static_cast<uint64_t>(i);
        output[i] = ring.records[
            (sequence - 1) % ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE];
    }
    consumer_sequence = begin + static_cast<uint64_t>(count);
    return count;
}

enum class PersistentBuildError : int {
    None = 0,
    InvalidParticleCount = 1,
    NodeCapacityExceeded = 2,
    M2LSameCapacityExceeded = 3,
    M2LCrossCapacityExceeded = 4,
    P2PCapacityExceeded = 5,
    P2PSelfCapacityExceeded = 6,
    DeviceGraphLaunchFailed = 7,
    SparseLookupCapacityExceeded = 8,
    ParticleOutsideFixedRoot = 9,
    DiagnosticRingOverflow = 10,
    InvalidPresortedMortonInput = 11,
    PresortedMortonMismatch = 12,
};

// An exact current-evaluation permutation prepared before graph execution.
// Keys retain the production 63-bit alignment and configured-depth mask.
struct PresortedMortonInput {
    const uint64_t* keys = nullptr;
    const int* indices = nullptr;
    int count = 0;
    int depth = 0;
    double min_x = 0.0, min_y = 0.0, min_z = 0.0, root_size = 0.0;
    int validate = 0;
};

struct PersistentTreeCommand {
    const double* pos_x = nullptr;
    const double* pos_y = nullptr;
    const double* pos_z = nullptr;
    int particle_count = 0;
    int common_planar_center = 0;
    int target_particle_count = 0;
    int current_step = 0;
    double evaluation_time = 0.0;
    // The dispatcher validates this as exactly 2 or 3.  It is immutable for
    // the lifetime of a captured runtime; no shape-driven tree switching.
    int tree_dimension = 3;
    int use_fixed_root = 0;
    double fixed_domain_min_x = 0.0;
    double fixed_domain_min_y = 0.0;
    double fixed_domain_min_z = 0.0;
    double fixed_root_cell_size = 0.0;
    double cutoff_radius = 0.0;
    PresortedMortonInput presorted;
    // Multi-rank execution uses fixed geometric M2L/P2P routing.
    int disable_direct_fallback = 0;
};

struct PersistentTreeMetadata {
    int num_nodes = 0;
    int num_children = 0;
    int num_unique_cells = 0;
    int active_tree = 3;
    int error = 0;
    int task_counts[4] = {0, 0, 0, 0};
    double domain_min_x = 0.0;
    double domain_min_y = 0.0;
    double domain_min_z = 0.0;
    double root_cell_size = 1.0;
    double extent_x = 0.0;
    double extent_y = 0.0;
    double extent_z = 0.0;
    double cutoff_radius = 0.0;
    int target_particle_count = 0;
    uint64_t tree_start_ns = 0;
    uint64_t tree_end_ns = 0;
    uint64_t stencil_end_ns = 0;
    uint64_t evaluation_start_ns = 0;
    uint64_t evaluation_end_ns = 0;
    int evaluation_depth = 0;
    uint64_t tree_phase_ns[9] = {};
    uint64_t tree_sort_end_ns = 0;
};

struct AdaptiveDepthCheckpoint {
    uint32_t version = 2;
    uint32_t reserved = 0;
    int32_t tree_dimension = 3;
    int32_t reserved_dimension = 0;
    AdaptiveDepthState controller{};
    uint64_t diagnostic_sequence = 0;
    uint64_t persisted_csv_sequence = 0;
};

struct PersistentTreeWorkspace {
    uint64_t* morton_unsorted = nullptr;
    uint64_t* morton_sorted = nullptr;
    int* index_unsorted = nullptr;

    double* block_min_x = nullptr;
    double* block_min_y = nullptr;
    double* block_min_z = nullptr;
    double* block_max_x = nullptr;
    double* block_max_y = nullptr;
    double* block_max_z = nullptr;

    KarrasBinaryNode* binary_nodes = nullptr;
    int* edge_counts = nullptr;
    int* edge_offsets = nullptr;
    int* edge_children = nullptr;
    int* incoming_edge = nullptr;

    int* unique_offsets = nullptr;
    int2* unique_ranges = nullptr;
    int* level_block_counts = nullptr;
    int* level_block_offsets = nullptr;
    uint32_t* dense_storage = nullptr;
    uint32_t* dense_linear_storage = nullptr;
    uint32_t* dense_linear_map[fmm::FMM_MAX_DENSE_LEVELS] = {};
    uint64_t* sparse_keys = nullptr;
    uint32_t* sparse_values = nullptr;
    uint64_t sparse_offset[fmm::FMM_MAX_LEVELS] = {};
    uint32_t sparse_capacity[fmm::FMM_MAX_LEVELS] = {};
    // Four emitted-task counts for each persistent stencil block.
    int* stencil_block_counts = nullptr;
};

struct PersistentTreeContext {
    PersistentTreeCommand command;
    PersistentTreeMetadata metadata;
    PersistentTreeWorkspace workspace;

    fmm::FMMDeviceData dd;
    GroupedInteractionArena arena;

    int particle_capacity = 0;
    int node_capacity = 0;
    int edge_capacity = 0;
    int dense_capacity_total = 0;
    uint64_t sparse_capacity_total = 0;
    int stencil_block_capacity = 0;
    int stencil_targets_per_block = 0;
    int level_block_capacity = 0;
    int tree_depth = 0;
    // Immutable graph mode; generic/adaptive device-selected builds sort locally.
    int use_presorted_morton = 0;

    int adaptive_depth_enabled = 0;
    int adaptive_auto_initial_depth = 0;
    int adaptive_min_depth = 0;
    int adaptive_max_depth = 0;
    int adaptive_trial_interval = 32;
    int adaptive_trial_warmups = 1;
    int adaptive_trial_samples = 4;
    int adaptive_confirmation_trials = 2;
    float adaptive_min_improvement = 0.02f;
    float adaptive_capacity_growth_trigger = 0.75f;
    int adaptive_external_control = 0;
    ExternalDistributedDepthCommand adaptive_external_command{};
    AdaptiveDepthState adaptive_depth[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_probe_active[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_probe_final[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_probe_complete[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_initial_selection_pending[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_skip_timing_once[PERSISTENT_TREE_DIMENSIONS]{};
    int adaptive_probe_nodes[PERSISTENT_TREE_DIMENSIONS]
                            [PERSISTENT_TREE_MAX_DEPTH + 1]{};
    int adaptive_probe_leaves[PERSISTENT_TREE_DIMENSIONS]
                             [PERSISTENT_TREE_MAX_DEPTH + 1]{};
    AdaptiveDepthDiagnosticRing adaptive_diagnostics{};
    cudaGraphExec_t depth_graph_exec[PERSISTENT_TREE_DIMENSIONS]
                                     [PERSISTENT_TREE_MAX_DEPTH + 1]{};
    int* terminal_error = nullptr;

    unsigned int doorbell = 0;
    unsigned int completed_epoch = 0;
    unsigned int shutdown = 0;
};

struct PersistentTreeRuntime {
    PersistentTreeContext host_context{};
    PersistentTreeContext* device_context = nullptr;
    // Consumed once by submit; stale permutations cannot survive another epoch.
    PresortedMortonInput pending_presorted;

    cudaGraph_t graph_d2 = nullptr;
    cudaGraph_t graph_d3 = nullptr;
    cudaGraph_t scheduler_graph = nullptr;
    cudaGraphExec_t graph_exec_d2 = nullptr;
    cudaGraphExec_t graph_exec_d3 = nullptr;
    cudaGraphExec_t scheduler_graph_exec = nullptr;
    cudaGraph_t depth_graph[PERSISTENT_TREE_DIMENSIONS]
                           [PERSISTENT_TREE_MAX_DEPTH + 1]{};
    cudaGraphExec_t depth_graph_exec[PERSISTENT_TREE_DIMENSIONS]
                                    [PERSISTENT_TREE_MAX_DEPTH + 1]{};
    // Host launches the device-launchable scheduler graph once for each
    // submitted command.  It waits only until that command is visible, then
    // tail-launches the selected build graph and exits.  Keeping no idle
    // polling block resident is required while normal DEM code performs
    // CUDA operations with context-wide synchronization semantics.
    cudaStream_t scheduler_stream = nullptr;
    cudaStream_t graph_upload_stream = nullptr;

    void* radix_sort_temp = nullptr;
    size_t radix_sort_temp_bytes = 0;
    void* edge_scan_temp = nullptr;
    size_t edge_scan_temp_bytes = 0;

    unsigned int next_epoch = 1;
    bool initialized = false;
    // Workspace lookup maps cover this depth even when a shallower graph runs.
    int allocated_depth = 0;
    bool cache_fixed_depth_graphs = false;
    AdaptiveDepthOptions adaptive_options{};
    int* terminal_error_managed = nullptr;
};

template<int D>
__host__ __device__ inline constexpr int persistent_dense_cutoff() {
    return D == 3 ? 7 : 10;
}

__device__ __forceinline__ uint64_t persistent_hash_morton(uint64_t key) {
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ull;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebull;
    return key ^ (key >> 31);
}

template<int D>
__device__ __forceinline__ int persistent_morton_lookup(
    const PersistentTreeContext* ctx, int level, uint64_t compact)
{
    if (level <= persistent_dense_cutoff<D>()) {
        if (level < 0 || level >= fmm::FMM_MAX_DENSE_LEVELS ||
            ctx->dd.level_dense_map[level] == nullptr)
            return -1;
        const uint32_t source =
            ctx->dd.level_dense_map[level][static_cast<uint32_t>(compact)];
        return source == 0xFFFFFFFFu ? -1 : static_cast<int>(source);
    }
    if (level < 0 || level >= fmm::FMM_MAX_LEVELS) return -1;
    const uint32_t capacity = ctx->workspace.sparse_capacity[level];
    if (capacity == 0) return -1;
    const uint64_t offset = ctx->workspace.sparse_offset[level];
    const uint32_t mask = capacity - 1u;
    uint32_t slot = static_cast<uint32_t>(persistent_hash_morton(compact)) & mask;
    for (uint32_t probe = 0; probe < capacity; ++probe) {
        const uint64_t found = ctx->workspace.sparse_keys[offset + slot];
        if (found == compact)
            return static_cast<int>(ctx->workspace.sparse_values[offset + slot]);
        if (found == UINT64_MAX) return -1;
        slot = (slot + 1u) & mask;
    }
    return -1;
}

template<int D>
__device__ __forceinline__ bool persistent_morton_insert(
    PersistentTreeContext* ctx, int level, uint64_t compact, uint32_t node)
{
    if (level <= persistent_dense_cutoff<D>()) {
        ctx->dd.level_dense_map[level][static_cast<uint32_t>(compact)] = node;
        return true;
    }
    const uint32_t capacity = ctx->workspace.sparse_capacity[level];
    if (capacity == 0) return false;
    const uint64_t offset = ctx->workspace.sparse_offset[level];
    const uint32_t mask = capacity - 1u;
    uint32_t slot = static_cast<uint32_t>(persistent_hash_morton(compact)) & mask;
    for (uint32_t probe = 0; probe < capacity; ++probe) {
        auto* address = reinterpret_cast<unsigned long long*>(
            ctx->workspace.sparse_keys + offset + slot);
        const uint64_t old = atomicCAS(address,
            static_cast<unsigned long long>(UINT64_MAX),
            static_cast<unsigned long long>(compact));
        if (old == UINT64_MAX || old == compact) {
            ctx->workspace.sparse_values[offset + slot] = node;
            return true;
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

template<int D>
__device__ __forceinline__ int persistent_coordinate_lookup(
    const PersistentTreeContext* ctx, int level, int x, int y, int z)
{
    if (level < 0 || level >= 31) return -1;
    const int side = 1 << level;
    if (x < 0 || x >= side || y < 0 || y >= side) return -1;
    if constexpr (D == 3) if (z < 0 || z >= side) return -1;
    if (level <= persistent_dense_cutoff<D>()) {
        uint32_t linear = static_cast<uint32_t>(x * side + y);
        if constexpr (D == 3)
            linear = linear * static_cast<uint32_t>(side) +
                     static_cast<uint32_t>(z);
        const uint32_t source = ctx->workspace.dense_linear_map[level][linear];
        return source == 0xFFFFFFFFu ? -1 : static_cast<int>(source);
    }
    return persistent_morton_lookup<D>(
        ctx, level, morton_encode<D>(static_cast<uint32_t>(x),
                                     static_cast<uint32_t>(y),
                                     static_cast<uint32_t>(z)));
}

int fixed_depth_node_upper_bound(int particle_capacity, int tree_depth,
                                 int tree_dimension);

// Change an externally selected fixed depth without replacing its arenas.
// Caller must keep the dimension and particle/node capacities unchanged.
void switch_persistent_tree_depth(PersistentTreeRuntime& runtime, int depth,
                                  int tree_dimension);

void initialize_persistent_tree_runtime(
    PersistentTreeRuntime& runtime,
    fmm::FMMDeviceData& dd,
    GroupedInteractionArena& arena,
    int particle_capacity,
    int tree_depth,
    int tree_dimension,
    float theta_far,
    uint64_t direct_threshold_leaf,
    uint64_t direct_threshold_internal);

// Adaptive overload.  The legacy overload above is intentionally retained so
// fixed-depth callers keep their original allocation/capture/launch path.
void initialize_persistent_tree_runtime(
    PersistentTreeRuntime& runtime,
    fmm::FMMDeviceData& dd,
    GroupedInteractionArena& arena,
    int particle_capacity,
    int tree_depth,
    int tree_dimension,
    float theta_far,
    uint64_t direct_threshold_leaf,
    uint64_t direct_threshold_internal,
    const AdaptiveDepthOptions& adaptive_options,
    int shared_node_capacity);

void submit_persistent_tree_runtime(
    PersistentTreeRuntime& runtime,
    const double* pos_x,
    const double* pos_y,
    const double* pos_z,
    int particle_count,
    int target_particle_count,
    int current_step,
    int tree_dimension,
    bool use_fixed_root,
    double fixed_domain_min_x,
    double fixed_domain_min_y,
    double fixed_domain_min_z,
    double fixed_root_cell_size,
    double cutoff_radius,
    cudaStream_t consumer_stream,
    double evaluation_time = 0.0,
    bool common_planar_center = false,
    bool disable_direct_fallback = false);

// Validation/debug only.  Production execution never calls this function.
// The caller must first complete the consumer stream for the submitted epoch;
// the copy itself uses the runtime's nonblocking diagnostic stream and avoids
// introducing a device-wide synchronization.
PersistentTreeMetadata snapshot_persistent_tree_metadata(
    const PersistentTreeRuntime& runtime);

// Place this marker on the FMM consumer stream after the last depth-dependent
// force kernel.  It closes the globaltimer interval and advances the device
// timing controller without a host synchronization.
void record_persistent_depth_evaluation(PersistentTreeRuntime& runtime,
                                        cudaStream_t consumer_stream);

// Switch an initialized adaptive runtime between its device-resident
// single-GPU controller and host-provided distributed commands.  No graphs
// are recaptured by this operation.
void set_persistent_external_depth_control(
    PersistentTreeRuntime& runtime, bool enabled, cudaStream_t stream);
void set_persistent_external_depth_command(
    PersistentTreeRuntime& runtime,
    const ExternalDistributedDepthCommand& command,
    cudaStream_t stream);
ExternalDistributedDepthEvaluation
snapshot_persistent_external_depth_evaluation(
    const PersistentTreeRuntime& runtime,
    cudaStream_t ordered_after_stream = 0);

AdaptiveDepthTelemetry snapshot_adaptive_depth_telemetry(
    const PersistentTreeRuntime& runtime,
    cudaStream_t ordered_after_stream = 0);

// Sparse staged-allocation recommendation based on the observed high-water
// count.  It is capped by the strict mathematical bound.  The caller may use
// it when recreating address-stable graph storage at a synchronization point.
int recommended_sparse_node_capacity(const PersistentTreeRuntime& runtime);

AdaptiveDepthCheckpoint snapshot_adaptive_depth_checkpoint(
    const PersistentTreeRuntime& runtime,
    cudaStream_t ordered_after_stream = 0);
void restore_adaptive_depth_checkpoint(
    PersistentTreeRuntime& runtime,
    const AdaptiveDepthCheckpoint& checkpoint,
    cudaStream_t stream);
void consume_adaptive_depth_diagnostics(
    PersistentTreeRuntime& runtime,
    AdaptiveDepthDiagnostic* output,
    int output_capacity,
    int& output_count,
    uint64_t& dropped,
    cudaStream_t stream);
void emit_adaptive_capacity_transition(
    PersistentTreeRuntime& runtime,
    AdaptiveDepthTransition transition,
    int tree_dimension,
    cudaStream_t stream);
void throw_if_persistent_terminal_error(const PersistentTreeRuntime& runtime);

void shutdown_persistent_tree_runtime(PersistentTreeRuntime& runtime);

}  // namespace body_fitted_fmm

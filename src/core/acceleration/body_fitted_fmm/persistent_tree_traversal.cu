#include "persistent_tree_traversal.cuh"

#include "karras_octree_build.cuh"

#include "morton_unified.cuh"
#include "task_structs.cuh"
#include "grouped_stencil_traits.cuh"

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>

namespace body_fitted_fmm {

namespace {

constexpr int PERSISTENT_BLOCK = 256;
constexpr int PERSISTENT_STENCIL_BLOCK = 256;

inline void driver_check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : "unknown") + " (" +
                             (message ? message : "no description") + ")");
}

// Scheduler and stencil error/timing hooks remain traversal-owned.
__device__ __forceinline__ void publish_error(PersistentTreeContext* ctx,
                                               PersistentBuildError error) {
    atomicCAS(&ctx->metadata.error, 0, static_cast<int>(error));
}

__device__ __forceinline__ uint64_t persistent_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}


template<int D>
__device__ __forceinline__ int persistent_linear_lookup(
    const PersistentTreeContext* ctx, int level, int x, int y, int z)
{
    return persistent_coordinate_lookup<D>(ctx, level, x, y, z);
}

enum class CutoffRelation : uint8_t { Inside, Outside, Straddles };

__device__ __forceinline__ CutoffRelation persistent_cutoff_relation(
    const UnifiedNode& target, const UnifiedNode& source, double cutoff, double root_size)
{
    if (!(cutoff > 0.0)) return CutoffRelation::Inside;
    const double dx = static_cast<double>(target.center.x) - source.center.x;
    const double dy = static_cast<double>(target.center.y) - source.center.y;
    const double dz = static_cast<double>(target.center.z) - source.center.z;
    const double distance = sqrt(dx * dx + dy * dy + dz * dz);
    // Multipole construction replaces node.radius with the occupied-particle
    // radius, which differs across owners. Fixed stencil/cutoff routing must
    // instead use the same immutable cell bounds before and after exchange.
    const float target_radius=static_cast<float>(0.5*sqrt(3.0)*ldexp(root_size,-target.level));
    const float source_radius=static_cast<float>(0.5*sqrt(3.0)*ldexp(root_size,-source.level));
    const double radii = static_cast<double>(target_radius) + source_radius;
    if (distance - radii > cutoff) return CutoffRelation::Outside;
    if (distance + radii <= cutoff) return CutoffRelation::Inside;
    return CutoffRelation::Straddles;
}

__global__ void persistent_stencil_finalize_counts_kernel(
    PersistentTreeContext* ctx, int stencil_blocks)
{
    const int cls = threadIdx.x;
    if (blockIdx.x != 0 || cls >= 4) return;
    int count = 0;
    for (int block = 0; block < stencil_blocks; ++block) {
        count += ctx->workspace.stencil_block_counts[
            static_cast<size_t>(cls) * stencil_blocks + block];
    }
    ctx->arena.d_counters[cls] = count;
    ctx->metadata.task_counts[cls] = count;
}

template<int D>
__global__ void persistent_stencil_emit_kernel(PersistentTreeContext* ctx) {
    using T = GroupedStencilTraits<D>;
    __shared__ int block_counts[4];
    if (threadIdx.x < 4) block_counts[threadIdx.x] = 0;
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int target_begin = blockIdx.x * ctx->stencil_targets_per_block;
    const int target_end = min(target_begin + ctx->stencil_targets_per_block,
                               ctx->node_capacity);

    for (int target_idx = target_begin + warp_in_block;
         target_idx < target_end; target_idx += warps_per_block) {
        int valid_target = 0;
        int level = 0;
        int flags = 0;
        uint32_t target_count = 0;
        uint64_t compact_morton = 0;
        if (lane == 0 && target_idx < ctx->metadata.num_nodes &&
            ctx->metadata.error == 0) {
            const UnifiedNode& target = ctx->dd.unified_nodes[target_idx];
            valid_target =
                (target.flags & UNIFIED_NODE_FLAG_VALID) && target.level >= 2 &&
                target.target_count > 0;
            level = target.level;
            flags = target.flags;
            // Route with the complete source-cell population on every rank.
            // In shear cells count also includes image-only sources, whereas
            // target_count only gates work for owned central-image targets.
            target_count = target.global_count ? target.global_count : target.count;
            compact_morton = target.compact_morton;
        }
        valid_target = __shfl_sync(0xFFFFFFFFu, valid_target, 0);
        if (!valid_target) continue;
        level = __shfl_sync(0xFFFFFFFFu, level, 0);
        flags = __shfl_sync(0xFFFFFFFFu, flags, 0);
        target_count = __shfl_sync(0xFFFFFFFFu, target_count, 0);
        compact_morton = __shfl_sync(0xFFFFFFFFu,
                                     static_cast<uint32_t>(compact_morton), 0) |
            (static_cast<uint64_t>(__shfl_sync(
                 0xFFFFFFFFu, static_cast<uint32_t>(compact_morton >> 32), 0)) << 32);

        uint32_t c0 = 0, c1 = 0, c2 = 0;
        if (lane == 0)
            morton_decode<D>(compact_morton, level, c0, c1, c2);
        c0 = __shfl_sync(0xFFFFFFFFu, c0, 0);
        c1 = __shfl_sync(0xFFFFFFFFu, c1, 0);
        c2 = __shfl_sync(0xFFFFFFFFu, c2, 0);
        int child_q = (c0 & 1u) << (D - 1);
        if constexpr (D >= 2) child_q |= (c1 & 1u) << (D - 2);
        if constexpr (D >= 3) child_q |= (c2 & 1u) << (D - 3);
        const bool is_leaf = (flags & UNIFIED_NODE_FLAG_IS_LEAF) != 0;

        GroupedTargetInteractions& out =
            ctx->arena.d_targets[target_idx];
        int same_count = 0;
        int cross_count = 0;
        int p2p_regular_count = 0;
        int p2p_self_count = 0;
        if (lane == 0) out.cross_direct_bits = 0u;
        __syncwarp();

        if (lane == 0) {
            out.target_parent = persistent_linear_lookup<D>(
                ctx, level - 1, c0 >> 1, c1 >> 1, c2 >> 1);
            if (out.target_parent < 0)
                publish_error(ctx, PersistentBuildError::NodeCapacityExceeded);
        }

        if (lane < GROUPED_MAX_CROSS_COUNT) {
            GroupedSourceRoute route = grouped_invalid_route();
            if (lane < T::CROSS_COUNT) {
                const GeneratedCrossStencilEntry entry = T::cross(child_q, lane);
                const int source = persistent_linear_lookup<D>(
                    ctx, level - 1,
                    (c0 >> 1) + static_cast<int>(entry.parent[0]),
                    (c1 >> 1) + static_cast<int>(entry.parent[1]),
                    (c2 >> 1) + static_cast<int>(entry.parent[2]));
                if (source >= 0) {
                    const CutoffRelation cutoff = persistent_cutoff_relation(
                        ctx->dd.unified_nodes[target_idx],
                        ctx->dd.unified_nodes[source],
                        ctx->command.cutoff_radius,ctx->metadata.root_cell_size);
                    if (cutoff != CutoffRelation::Outside) {
                        const uint64_t work = static_cast<uint64_t>(
                            (ctx->dd.unified_nodes[source].global_count ?
                             ctx->dd.unified_nodes[source].global_count : ctx->dd.unified_nodes[source].count)) *
                            static_cast<uint64_t>(target_count);
                        const bool direct = cutoff == CutoffRelation::Straddles ||
                            (!ctx->command.disable_direct_fallback &&
                             work <= grouped_direct_threshold(level, is_leaf));
                        route = grouped_route(static_cast<uint32_t>(source),
                                              direct ? 1u : 0u);
                        direct ? ++p2p_regular_count : ++cross_count;
                    }
                }
            }
            out.cross_node[lane] = route.node;
            if (route.direct_mask)
                atomicOr(&out.cross_direct_bits, 1u << lane);
        }

        if (lane < GROUPED_MAX_BLOCK_COUNT) {
            GroupedSourceRoute route = grouped_invalid_route();
            uint8_t omit_mask = 0;
            if (lane < T::BLOCK_COUNT) {
                const GeneratedChildBlockStencilEntry entry = T::block(child_q, lane);
                const int source_parent = persistent_linear_lookup<D>(
                    ctx, level - 1,
                    (c0 >> 1) + static_cast<int>(entry.parent[0]),
                    (c1 >> 1) + static_cast<int>(entry.parent[1]),
                    (c2 >> 1) + static_cast<int>(entry.parent[2]));
                if (source_parent >= 0) {
                    uint8_t direct_mask = 0;
                    #pragma unroll
                    for (int child = 0; child < T::CHILD_COUNT; ++child) {
                        const int source = grouped_child_node<D>(
                            ctx->dd.unified_nodes, ctx->dd.children_compact,
                            source_parent, child);
                        if (source < 0) continue;
                        const uint8_t bit = static_cast<uint8_t>(1u << child);
                        const CutoffRelation cutoff = persistent_cutoff_relation(
                            ctx->dd.unified_nodes[target_idx],
                            ctx->dd.unified_nodes[source],
                            ctx->command.cutoff_radius,ctx->metadata.root_cell_size);
                        if (cutoff == CutoffRelation::Outside) {
                            if (entry.far_mask & bit) omit_mask |= bit;
                            continue;
                        }
                        if (entry.far_mask & bit) {
                            const uint64_t work = static_cast<uint64_t>(
                                (ctx->dd.unified_nodes[source].global_count ?
                             ctx->dd.unified_nodes[source].global_count : ctx->dd.unified_nodes[source].count)) *
                                static_cast<uint64_t>(target_count);
                            if (cutoff == CutoffRelation::Straddles ||
                                (!ctx->command.disable_direct_fallback &&
                                 work <= grouped_direct_threshold(level, is_leaf))) {
                                direct_mask |= bit;
                                ++p2p_regular_count;
                            } else {
                                ++same_count;
                            }
                        } else if (is_leaf && (entry.near_mask & bit)) {
                            direct_mask |= bit;
                            if (source == target_idx)
                                ++p2p_self_count;
                            else
                                ++p2p_regular_count;
                        }
                    }
                    route = grouped_route(static_cast<uint32_t>(source_parent),
                                          direct_mask);
                }
            }
            grouped_set_block_route(out, lane, route);
            grouped_set_block_omit_mask(out, lane, omit_mask);
        }

        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            same_count += __shfl_down_sync(0xffffffffu, same_count, delta);
            cross_count += __shfl_down_sync(0xffffffffu, cross_count, delta);
            p2p_regular_count += __shfl_down_sync(
                0xffffffffu, p2p_regular_count, delta);
            p2p_self_count += __shfl_down_sync(
                0xffffffffu, p2p_self_count, delta);
        }
        if (lane == 0) {
            atomicAdd(&block_counts[0], same_count);
            atomicAdd(&block_counts[1], cross_count);
            atomicAdd(&block_counts[2], p2p_regular_count);
            atomicAdd(&block_counts[3], p2p_self_count);
        }
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        ctx->workspace.stencil_block_counts[
            static_cast<size_t>(threadIdx.x) * gridDim.x + blockIdx.x] =
            block_counts[threadIdx.x];
    }
}

__device__ __forceinline__ int adaptive_dimension_slot(int dimension) {
    (void)dimension;
    return 0;
}

__device__ __forceinline__ int adaptive_neighbor_depth(
    AdaptiveDepthState& state, const PersistentTreeContext* ctx)
{
    int candidate = state.selected_depth + state.search_direction;
    if (candidate < ctx->adaptive_min_depth ||
        candidate > state.effective_max_depth) {
        // Reaching an intentionally staged ceiling is a sticky request for
        // the next safe-boundary capacity service.  Keep using the fitting
        // neighbor in the meantime; normal evaluations remain asynchronous.
        if (candidate > state.effective_max_depth &&
            state.effective_max_depth < ctx->adaptive_max_depth)
            ++state.capacity_fallback_count;
        state.search_direction = -state.search_direction;
        candidate = state.selected_depth + state.search_direction;
    }
    return max(ctx->adaptive_min_depth,
               min(state.effective_max_depth, candidate));
}

__device__ __forceinline__ int adaptive_choose_launch_depth(
    PersistentTreeContext* ctx, int dimension)
{
    AdaptiveDepthState& state =
        ctx->adaptive_depth[adaptive_dimension_slot(dimension)];
    if (state.trial_phase == 0 &&
        state.evaluations_since_trial >= ctx->adaptive_trial_interval) {
        const int candidate = adaptive_neighbor_depth(state, ctx);
        if (candidate != state.selected_depth) {
            state.candidate_depth = candidate;
            state.trial_phase = ctx->adaptive_trial_warmups > 0 ? 1 : 2;
            state.trial_warmups = 0;
            state.trial_sample_count = 0;
            state.candidate_time_ns = 0;
        } else {
            state.evaluations_since_trial = 0;
        }
    }
    state.launch_depth = state.trial_phase == 0
        ? state.selected_depth : state.candidate_depth;
    return state.launch_depth;
}

__device__ __forceinline__ void adaptive_emit_diagnostic(
    PersistentTreeContext* ctx,
    int dimension,
    const AdaptiveDepthState& state,
    AdaptiveDepthTransition reason)
{
    AdaptiveDepthDiagnosticRing& ring = ctx->adaptive_diagnostics;
    const uint64_t sequence = ring.next_sequence;
    if (sequence - ring.consumer_sequence >=
        ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE) {
        ++ring.dropped;
        ctx->metadata.error = static_cast<int>(
            PersistentBuildError::DiagnosticRingOverflow);
        if (ctx->terminal_error)
            *ctx->terminal_error = static_cast<int>(
                PersistentBuildError::DiagnosticRingOverflow);
        __threadfence_system();
        asm volatile("trap;");
        return;
    }
    ++ring.next_sequence;
    AdaptiveDepthDiagnostic record{};
    record.sequence = sequence;
    record.evaluation_counter = state.evaluation_counter;
    record.evaluation_time = ctx->command.evaluation_time;
    record.tree_dimension = dimension;
    record.selected_depth = state.selected_depth;
    record.candidate_depth = state.candidate_depth;
    record.search_direction = state.search_direction;
    record.trial_phase = state.trial_phase;
    record.elapsed_ms = static_cast<double>(state.last_elapsed_ns) * 1.0e-6;
    record.relative_improvement = state.last_relative_improvement;
    record.node_count = ctx->metadata.num_nodes;
    record.node_capacity = ctx->node_capacity;
    record.effective_max_depth = state.effective_max_depth;
    record.transition_reason = reason;
    ring.records[(sequence - 1) % ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE] = record;
}

__global__ void persistent_complete_kernel(PersistentTreeContext* ctx) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ctx->metadata.stencil_end_ns = persistent_globaltimer_ns();

    // An adaptive graph that exhausts its sparse shared storage retries one
    // level shallower on the device.  The reset at the head of the fallback
    // graph clears metadata.error; the consumer epoch is published only by a
    // successful graph.  Fixed-depth execution retains the original fatal
    // policy below.
    if (ctx->adaptive_depth_enabled &&
        ctx->metadata.error ==
            static_cast<int>(PersistentBuildError::NodeCapacityExceeded)) {
        const int dimension = ctx->metadata.active_tree;
        AdaptiveDepthState& state =
            ctx->adaptive_depth[adaptive_dimension_slot(dimension)];
        const int fallback = ctx->tree_depth - 1;
        if (fallback >= ctx->adaptive_min_depth) {
            state.effective_max_depth = min(state.effective_max_depth, fallback);
            state.node_high_water = max(state.node_high_water,
                                        ctx->node_capacity + 1);
            ++state.capacity_fallback_count;
            state.selected_depth = min(state.selected_depth, fallback);
            state.candidate_depth = state.selected_depth;
            state.launch_depth = fallback;
            state.trial_phase = 0;
            state.evaluations_since_trial = 0;
            state.trial_warmups = 0;
            state.trial_sample_count = 0;
            state.confirmation_count = 0;
            state.selected_time_ns = 0;
            state.selected_sample_count = 0;
            state.candidate_time_ns = 0;
            ctx->adaptive_skip_timing_once[
                adaptive_dimension_slot(dimension)] = 1;
            adaptive_emit_diagnostic(ctx, dimension, state,
                                      AdaptiveDepthTransition::CapacityFallback);
            ctx->tree_depth = fallback;
            const cudaGraphExec_t graph =
                ctx->depth_graph_exec[adaptive_dimension_slot(dimension)][fallback];
            if (graph && cudaGraphLaunch(graph, cudaStreamGraphTailLaunch) ==
                             cudaSuccess)
                return;
            ctx->metadata.error = static_cast<int>(
                PersistentBuildError::DeviceGraphLaunchFailed);
        } else {
            // No shallower recoverable retry remains. Publish the terminal
            // mailbox and completion epoch before trapping so a stream-value
            // wait cannot deadlock and no invalid force can be integrated.
            if (ctx->terminal_error)
                *ctx->terminal_error = static_cast<int>(
                    PersistentBuildError::NodeCapacityExceeded);
            adaptive_emit_diagnostic(ctx, dimension, state,
                                      AdaptiveDepthTransition::FatalMinDepth);
            __threadfence_system();
            atomicExch(&ctx->completed_epoch, ctx->doorbell);
            asm volatile("trap;");
            return;
        }
    }

    if (ctx->adaptive_depth_enabled && ctx->metadata.error == 0) {
        const int dimension = ctx->metadata.active_tree;
        const int slot = adaptive_dimension_slot(dimension);
        if (ctx->adaptive_probe_active[slot]) {
            AdaptiveDepthState& state = ctx->adaptive_depth[slot];
            if (ctx->adaptive_probe_final[slot]) {
                // This is the selected-depth rebuild that follows the probe;
                // publish it normally, but exclude the one-time multi-build
                // startup cost from the timing controller.
                ctx->adaptive_probe_active[slot] = 0;
                ctx->adaptive_probe_final[slot] = 0;
                ctx->adaptive_probe_complete[slot] = 1;
                ctx->adaptive_initial_selection_pending[slot] = 1;
            } else {
                const int depth = ctx->tree_depth;
                ctx->adaptive_probe_nodes[slot][depth] =
                    ctx->metadata.num_nodes;
                ctx->adaptive_probe_leaves[slot][depth] =
                    ctx->metadata.num_unique_cells;
                state.node_high_water = max(state.node_high_water,
                                            ctx->metadata.num_nodes);
                int next_depth = depth - 1;
                if (next_depth < ctx->adaptive_min_depth) {
                    int best_depth = ctx->adaptive_min_depth;
                    double best_score = DBL_MAX;
                    for (int candidate = ctx->adaptive_min_depth;
                         candidate <= state.effective_max_depth; ++candidate) {
                        const int leaves =
                            ctx->adaptive_probe_leaves[slot][candidate];
                        if (leaves <= 0) continue;
                        const double occupancy = static_cast<double>(
                            ctx->command.particle_count) / leaves;
                        const double score = fabs(log(fmax(occupancy, 1.0e-12) /
                                                      16.0));
                        if (score < best_score) {
                            best_score = score;
                            best_depth = candidate;
                        }
                    }
                    state.selected_depth = best_depth;
                    state.candidate_depth = best_depth;
                    state.launch_depth = best_depth;
                    state.evaluations_since_trial = 0;
                    state.selected_time_ns = 0;
                    state.selected_sample_count = 0;
                    next_depth = best_depth;
                    ctx->adaptive_probe_final[slot] = 1;
                } else {
                    state.launch_depth = next_depth;
                }
                ctx->tree_depth = next_depth;
                const cudaGraphExec_t graph =
                    ctx->depth_graph_exec[slot][next_depth];
                if (graph && cudaGraphLaunch(graph, cudaStreamGraphTailLaunch) ==
                                 cudaSuccess)
                    return;
                publish_error(ctx, PersistentBuildError::DeviceGraphLaunchFailed);
            }
        }
    }

    for (int s = 0; s < 4; ++s)
        ctx->metadata.task_counts[s] = ctx->arena.d_counters[s];

    if (ctx->dd.d_stats) {
        ctx->dd.d_stats->num_octree_nodes = ctx->metadata.num_nodes;
        ctx->dd.d_stats->num_m2l_interactions =
            static_cast<int64_t>(ctx->metadata.task_counts[0]) +
            static_cast<int64_t>(ctx->metadata.task_counts[1]);
        ctx->dd.d_stats->num_p2p_interactions =
            static_cast<int64_t>(ctx->metadata.task_counts[2]) +
            static_cast<int64_t>(ctx->metadata.task_counts[3]);
    }
    __threadfence_system();
    atomicExch(&ctx->completed_epoch, ctx->doorbell);
    if (ctx->metadata.error != 0) {
        // Capacity and launch failures are fatal by policy: never truncate and
        // never fall back to CPU construction/traversal.
        asm volatile("trap;");
    }

}

__global__ void persistent_scheduler_kernel(PersistentTreeContext* ctx,
                                            cudaGraphExec_t graph_d2,
                                            cudaGraphExec_t graph_d3) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // This graph is launched by the host for one command.  The spin is only
    // needed until the command stream reaches its doorbell write; returning
    // after the build tail launch leaves no idle resident block between DEM
    // timesteps, where generic CUDA operations may require whole-context
    // progress.
    while (atomicAdd(&ctx->shutdown, 0u) == 0u) {
        const unsigned int epoch = atomicAdd(&ctx->doorbell, 0u);
        if (epoch != atomicAdd(&ctx->completed_epoch, 0u)) {
            const int dimension = ctx->command.tree_dimension;
            ctx->metadata.active_tree = dimension;
            cudaGraphExec_t selected_graph = dimension == 2 ? graph_d2 : graph_d3;
            if (ctx->adaptive_depth_enabled) {
                const int slot = adaptive_dimension_slot(dimension);
                int depth = 0;
                if (ctx->adaptive_auto_initial_depth &&
                    !ctx->adaptive_probe_complete[slot]) {
                    // The sparse, position-aware probe is also required for
                    // external distributed control.  It establishes each
                    // GPU's local initial depth and fitting ceiling from its
                    // owned prefix before the host starts issuing trials.
                    ctx->adaptive_probe_active[slot] = 1;
                    ctx->adaptive_probe_final[slot] = 0;
                    depth = ctx->adaptive_depth[slot].effective_max_depth;
                    ctx->adaptive_depth[slot].launch_depth = depth;
                } else if (ctx->adaptive_external_control) {
                    // Distributed scheduling deliberately bypasses the local
                    // probe/trial FSM.  The host command is clamped to the
                    // current fitting ceiling so a previously observed sparse
                    // overflow remains a recoverable shallower retry until a
                    // safe-boundary capacity service grows this runtime.
                    AdaptiveDepthState& state = ctx->adaptive_depth[slot];
                    const ExternalDistributedDepthCommand command =
                        ctx->adaptive_external_command;
                    const int ceiling = max(ctx->adaptive_min_depth,
                                            state.effective_max_depth);
                    if (command.selected_depth > 0)
                        state.selected_depth = max(ctx->adaptive_min_depth,
                                                   min(ceiling, command.selected_depth));
                    if (command.candidate_depth > 0)
                        state.candidate_depth = max(ctx->adaptive_min_depth,
                                                    min(ceiling, command.candidate_depth));
                    else
                        state.candidate_depth = state.selected_depth;
                    depth = command.launch_depth > 0
                        ? max(ctx->adaptive_min_depth,
                              min(ceiling, command.launch_depth))
                        : state.selected_depth;
                    state.launch_depth = depth;
                    // Never let a stale autonomous A/B trial resume if a
                    // controller is enabled midway through a run.
                    state.trial_phase = 0;
                    state.trial_warmups = 0;
                    state.trial_sample_count = 0;
                    state.selected_time_ns = 0;
                    state.selected_sample_count = 0;
                    state.candidate_time_ns = 0;
                } else {
                    depth = adaptive_choose_launch_depth(ctx, dimension);
                }
                ctx->tree_depth = depth;
                ctx->metadata.evaluation_start_ns = persistent_globaltimer_ns();
                selected_graph = ctx->depth_graph_exec[
                    adaptive_dimension_slot(dimension)][depth];
            }
            // Common distributed control stages a fixed-depth graph on each
            // rank. Publish that graph's depth too, so validation observes the
            // actual launch rather than an unset autonomous-controller field.
            ctx->metadata.evaluation_depth = ctx->tree_depth;
            const cudaError_t status = cudaGraphLaunch(
                selected_graph, cudaStreamGraphTailLaunch);
            if (status != cudaSuccess) {
                publish_error(ctx, PersistentBuildError::DeviceGraphLaunchFailed);
                __threadfence_system();
                atomicExch(&ctx->completed_epoch, epoch);
                asm volatile("trap;");
            }
            return;
        }
        __nanosleep(256);
    }
}

__global__ void persistent_record_depth_evaluation_kernel(
    PersistentTreeContext* ctx)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 ||
        !ctx->adaptive_depth_enabled)
        return;

    const uint64_t end = persistent_globaltimer_ns();
    ctx->metadata.evaluation_end_ns = end;
    if (end < ctx->metadata.evaluation_start_ns) return;

    const int slot = adaptive_dimension_slot(ctx->metadata.active_tree);
    AdaptiveDepthState& state = ctx->adaptive_depth[slot];
    if (ctx->adaptive_external_control) {
        // The distributed controller times the complete gravity epoch outside
        // this local boundary (including LET and remote evaluation).  Keep
        // the existing single-GPU timing boundary intact, but publish the
        // local rebuild metadata and high-water information for that host
        // controller without advancing the autonomous FSM.
        ++state.evaluation_counter;
        state.node_high_water = max(state.node_high_water,
                                    ctx->metadata.num_nodes);
        state.last_elapsed_ns = 0;
        ctx->adaptive_initial_selection_pending[slot] = 0;
        ctx->adaptive_skip_timing_once[slot] = 0;
        return;
    }
    if (ctx->adaptive_initial_selection_pending[slot]) {
        const int pending = ctx->adaptive_initial_selection_pending[slot];
        ctx->adaptive_initial_selection_pending[slot] = 0;
        ctx->adaptive_skip_timing_once[slot] = 0;
        state.last_elapsed_ns = 0;
        adaptive_emit_diagnostic(ctx, ctx->metadata.active_tree, state,
                                 AdaptiveDepthTransition::InitialSelection);
        if (pending == 1) return;
    }
    if (ctx->adaptive_skip_timing_once[slot]) {
        ctx->adaptive_skip_timing_once[slot] = 0;
        return;
    }
    const uint64_t elapsed = end - ctx->metadata.evaluation_start_ns;
    state.last_elapsed_ns = elapsed;
    ++state.evaluation_counter;
    state.node_high_water = max(state.node_high_water,
                                ctx->metadata.num_nodes);

    if (state.trial_phase == 0) {
        state.selected_time_ns += elapsed;
        ++state.selected_sample_count;
        ++state.evaluations_since_trial;
        return;
    }

    if (state.trial_phase == 1) {
        ++state.trial_warmups;
        if (state.trial_warmups >= ctx->adaptive_trial_warmups)
            state.trial_phase = 2;
        return;
    }

    state.candidate_time_ns += elapsed;
    ++state.trial_sample_count;
    if (state.trial_sample_count < ctx->adaptive_trial_samples) return;

    const double selected_mean = state.selected_sample_count > 0
        ? static_cast<double>(state.selected_time_ns) /
              static_cast<double>(state.selected_sample_count)
        : static_cast<double>(elapsed);
    const double candidate_mean = static_cast<double>(state.candidate_time_ns) /
                                  state.trial_sample_count;
    const double relative = selected_mean > 0.0
        ? (selected_mean - candidate_mean) / selected_mean : 0.0;
    state.last_relative_improvement = static_cast<float>(relative);
    const bool faster = relative >= ctx->adaptive_min_improvement;
    bool accepted = false;
    AdaptiveDepthTransition transition =
        AdaptiveDepthTransition::CandidateRejected;
    if (faster) {
        ++state.confirmation_count;
        transition = AdaptiveDepthTransition::Confirmation;
        if (state.confirmation_count >= ctx->adaptive_confirmation_trials) {
            state.search_direction = state.candidate_depth > state.selected_depth
                ? 1 : -1;
            state.selected_depth = state.candidate_depth;
            state.confirmation_count = 0;
            // Seed the new baseline with the accepted candidate samples.
            state.selected_time_ns = state.candidate_time_ns;
            state.selected_sample_count = state.trial_sample_count;
            accepted = true;
            transition = AdaptiveDepthTransition::FasterNeighbor;
        }
    } else {
        state.confirmation_count = 0;
        state.search_direction = -state.search_direction;
    }
    state.candidate_depth = state.selected_depth;
    if (!accepted) {
        // Each comparison uses only the selected-depth samples collected
        // since the preceding trial; moving topology must not be compared
        // against a lifetime average.
        state.selected_time_ns = 0;
        state.selected_sample_count = 0;
    }
    state.candidate_time_ns = 0;
    state.trial_sample_count = 0;
    state.trial_warmups = 0;
    state.trial_phase = 0;
    state.evaluations_since_trial = 0;
    adaptive_emit_diagnostic(ctx, ctx->metadata.active_tree, state, transition);
}

__global__ void persistent_emit_capacity_transition_kernel(
    PersistentTreeContext* ctx, AdaptiveDepthTransition transition,
    int requested_dimension)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 ||
        !ctx->adaptive_depth_enabled)
        return;
    const int dimension = requested_dimension == 2 || requested_dimension == 3
        ? requested_dimension
        : (ctx->metadata.active_tree == 2 ? 2 : 3);
    adaptive_emit_diagnostic(
        ctx, dimension,
        ctx->adaptive_depth[adaptive_dimension_slot(dimension)], transition);
}

template<int D>
void capture_build_graph(PersistentTreeRuntime& runtime, int depth) {
    const auto& hc = runtime.host_context;
    PersistentTreeContext* ctx = runtime.device_context;

    cudaStream_t stream = runtime.graph_upload_stream;
    // Each OpenMP worker initializes and captures an independent GPU.  A
    // global capture would reject allocations issued concurrently by workers
    // on the other devices even though those operations cannot affect this
    // stream's graph.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    capture_karras_tree_stages<D>(runtime, stream);

    const int stencil_blocks = std::min(hc.stencil_block_capacity,
                                        hc.node_capacity);
    persistent_stencil_emit_kernel<D><<<stencil_blocks, PERSISTENT_STENCIL_BLOCK,
                                        0, stream>>>(ctx);
    persistent_stencil_finalize_counts_kernel<<<1, 4, 0, stream>>>(
        ctx, stencil_blocks);
    persistent_complete_kernel<<<1, 1, 0, stream>>>(ctx);

    const int slot = 0;
    cudaGraph_t* graph = (runtime.adaptive_options.enabled || runtime.cache_fixed_depth_graphs)
        ? &runtime.depth_graph[slot][depth]
        : (D == 2 ? &runtime.graph_d2 : &runtime.graph_d3);
    CUDA_CHECK(cudaStreamEndCapture(stream, graph));
    cudaGraphExec_t* exec = (runtime.adaptive_options.enabled || runtime.cache_fixed_depth_graphs)
        ? &runtime.depth_graph_exec[slot][depth]
        : (D == 2 ? &runtime.graph_exec_d2 : &runtime.graph_exec_d3);
    CUDA_CHECK(cudaGraphInstantiate(exec, *graph,
                                    cudaGraphInstantiateFlagDeviceLaunch));
    CUDA_CHECK(cudaGraphUpload(*exec, stream));
}

void capture_scheduler_graph(PersistentTreeRuntime& runtime,
                             cudaGraphExec_t graph_d2,
                             cudaGraphExec_t graph_d3) {
    cudaStream_t stream = runtime.graph_upload_stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    persistent_scheduler_kernel<<<1, 1, 0, stream>>>(
        runtime.device_context, graph_d2, graph_d3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &runtime.scheduler_graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &runtime.scheduler_graph_exec, runtime.scheduler_graph,
        cudaGraphInstantiateFlagDeviceLaunch));
    CUDA_CHECK(cudaGraphUpload(runtime.scheduler_graph_exec, stream));
}

template<typename T>
void device_alloc(T*& pointer, size_t count) {
    CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
}

template<typename T>
void device_free(T*& pointer) {
    if (pointer) CUDA_CHECK(cudaFree(pointer));
    pointer = nullptr;
}

}  // namespace

void switch_persistent_tree_depth(PersistentTreeRuntime& runtime, int depth,
                                  int tree_dimension) {
    if (!runtime.initialized || runtime.adaptive_options.enabled ||
        depth < PERSISTENT_TREE_MIN_DEPTH || depth > runtime.allocated_depth ||
        (tree_dimension != 2 && tree_dimension != 3))
        throw std::invalid_argument("fixed tree depth switch exceeds allocated workspace");
    auto& hc = runtime.host_context;
    if (hc.tree_depth == depth) return;

    // Tail-launched children and the consumer may still hold arena pointers.
    // Fence only at a depth transition before replacing the tiny scheduler.
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!runtime.cache_fixed_depth_graphs) {
        auto& graph = tree_dimension == 2 ? runtime.graph_d2 : runtime.graph_d3;
        auto& exec = tree_dimension == 2 ? runtime.graph_exec_d2 : runtime.graph_exec_d3;
        runtime.depth_graph[0][hc.tree_depth] = graph;
        runtime.depth_graph_exec[0][hc.tree_depth] = exec;
        graph = nullptr;
        exec = nullptr;
        runtime.cache_fixed_depth_graphs = true;
    }
    hc.tree_depth = depth;
    // Sort bit ranges and per-level scans are captured by value, so each
    // depth needs its own graph; changing only ctx->tree_depth is incorrect.
    if (!runtime.depth_graph_exec[0][depth]) {
        if (tree_dimension == 2) capture_build_graph<2>(runtime, depth);
        else capture_build_graph<3>(runtime, depth);
    }
    CUDA_CHECK(cudaMemcpyAsync(&runtime.device_context->tree_depth, &hc.tree_depth,
        sizeof(hc.tree_depth), cudaMemcpyHostToDevice, runtime.graph_upload_stream));
    CUDA_CHECK(cudaGraphExecDestroy(runtime.scheduler_graph_exec));
    runtime.scheduler_graph_exec = nullptr;
    CUDA_CHECK(cudaGraphDestroy(runtime.scheduler_graph));
    runtime.scheduler_graph = nullptr;
    const auto selected = runtime.depth_graph_exec[0][depth];
    capture_scheduler_graph(runtime, selected, selected);
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
}

int fixed_depth_node_upper_bound(int particle_capacity, int tree_depth,
                                 int tree_dimension) {
    if (particle_capacity <= 0 || tree_depth <= 0)
        throw std::invalid_argument("fixed-depth tree capacity requires positive inputs");

    auto bound = [=](int dimension) {
        uint64_t cells = 1;
        uint64_t total = 1;
        const uint64_t branch = 1u << dimension;
        for (int level = 1; level <= tree_depth; ++level) {
            if (cells > static_cast<uint64_t>(particle_capacity) / branch)
                cells = particle_capacity;
            else
                cells *= branch;
            total += std::min<uint64_t>(cells, particle_capacity);
        }
        if (total > static_cast<uint64_t>(INT_MAX))
            throw std::overflow_error("fixed-depth node bound exceeds INT_MAX");
        return static_cast<int>(total);
    };

    if (tree_dimension != 2 && tree_dimension != 3)
        throw std::invalid_argument("tree_dimension must be 2 or 3");
    return bound(tree_dimension);
}

void initialize_persistent_tree_runtime(
    PersistentTreeRuntime& runtime,
    fmm::FMMDeviceData& dd,
    GroupedInteractionArena& arena,
    int particle_capacity,
    int tree_depth,
    int tree_dimension,
    float theta_far,
    uint64_t direct_threshold_leaf,
    uint64_t direct_threshold_internal)
{
    initialize_persistent_tree_runtime(
        runtime, dd, arena, particle_capacity, tree_depth, tree_dimension,
        theta_far,
        direct_threshold_leaf, direct_threshold_internal,
        AdaptiveDepthOptions{}, 0);
}

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
    int shared_node_capacity)
{
    if (runtime.initialized)
        throw std::logic_error("persistent tree runtime initialized twice");
    if (!adaptive_options.enabled &&
        (tree_depth < PERSISTENT_TREE_MIN_DEPTH ||
         tree_depth > PERSISTENT_TREE_MAX_DEPTH))
        throw std::invalid_argument("tree_depth is outside the selected tree's supported range");
    if (adaptive_options.enabled &&
        (adaptive_options.min_depth < PERSISTENT_TREE_MIN_DEPTH ||
         adaptive_options.max_depth > PERSISTENT_TREE_MAX_DEPTH ||
         adaptive_options.min_depth > adaptive_options.max_depth))
        throw std::invalid_argument("adaptive tree-depth range is invalid");
    if (std::abs(theta_far - 0.8f) > 1.0e-6f)
        throw std::invalid_argument(
            "local grouped traversal tables require theta_far=0.8");

    runtime.adaptive_options = adaptive_options;
    const int initial_depth = adaptive_options.enabled
        ? (adaptive_options.startup_probe_complete
            ? adaptive_options.initial_depth
            : adaptive_options.auto_initial_depth
            ? (adaptive_options.min_depth + adaptive_options.max_depth) / 2
            : std::max(adaptive_options.min_depth,
                       std::min(adaptive_options.max_depth,
                                adaptive_options.initial_depth > 0
                                    ? adaptive_options.initial_depth : tree_depth)))
        : tree_depth;
    const int allocation_depth = adaptive_options.enabled
        ? adaptive_options.max_depth : tree_depth;
    runtime.allocated_depth = allocation_depth;

    auto& hc = runtime.host_context;
    if (hc.use_presorted_morton && (tree_dimension != 3 || adaptive_options.enabled))
        throw std::invalid_argument("presorted Morton input requires an externally selected fixed 3D depth");
    hc.particle_capacity = particle_capacity;
    const int strict_node_capacity = fixed_depth_node_upper_bound(
        particle_capacity, allocation_depth, tree_dimension);
    hc.node_capacity = shared_node_capacity > 0
        ? shared_node_capacity : strict_node_capacity;
    if (hc.node_capacity <= 0)
        throw std::invalid_argument("shared adaptive node capacity must be positive");
    if (hc.node_capacity >= INT_MAX)
        throw std::overflow_error("fixed-depth node capacity exceeds signed index width");
    hc.edge_capacity = 2 * particle_capacity - 1;
    hc.stencil_block_capacity = std::min(8192, hc.node_capacity);
    hc.stencil_targets_per_block =
        (hc.node_capacity + hc.stencil_block_capacity - 1) /
        hc.stencil_block_capacity;
    hc.level_block_capacity =
        (hc.node_capacity + PERSISTENT_BLOCK - 1) / PERSISTENT_BLOCK;
    // Workspace maps must cover the deepest captured graph.  The scheduler
    // replaces tree_depth immediately before every adaptive launch.
    hc.tree_depth = allocation_depth;
    hc.command.tree_dimension = tree_dimension;
    hc.dd = dd;

    if (adaptive_options.enabled) {
        hc.adaptive_depth_enabled = 1;
        hc.adaptive_auto_initial_depth =
            adaptive_options.auto_initial_depth ? 1 : 0;
        hc.adaptive_min_depth = adaptive_options.min_depth;
        hc.adaptive_max_depth = adaptive_options.max_depth;
        hc.adaptive_trial_interval =
            std::max(1, adaptive_options.trial_interval_evaluations);
        hc.adaptive_trial_warmups =
            std::max(0, adaptive_options.trial_warmup_evaluations);
        hc.adaptive_trial_samples = std::max(1, adaptive_options.trial_samples);
        hc.adaptive_confirmation_trials =
            std::max(1, adaptive_options.confirmation_trials);
        hc.adaptive_min_improvement =
            std::max(0.0f, adaptive_options.min_improvement_fraction);
        hc.adaptive_capacity_growth_trigger = std::max(
            0.01f, std::min(1.0f, adaptive_options.capacity_growth_trigger));
        hc.adaptive_external_control =
            adaptive_options.control_mode ==
                    AdaptiveDepthControlMode::ExternalDistributed ? 1 : 0;
        for (int slot = 0; slot < PERSISTENT_TREE_DIMENSIONS; ++slot) {
            AdaptiveDepthState& state = hc.adaptive_depth[slot];
            const int probed_initial = adaptive_options.initial_depth;
            const int slot_initial = adaptive_options.startup_probe_complete &&
                    probed_initial >= adaptive_options.min_depth &&
                    probed_initial <= adaptive_options.max_depth
                ? probed_initial : initial_depth;
            state.selected_depth = slot_initial;
            state.candidate_depth = slot_initial;
            state.launch_depth = slot_initial;
            state.search_direction = 1;
            const int probed_ceiling = adaptive_options.effective_max_depth;
            state.effective_max_depth = adaptive_options.startup_probe_complete &&
                    probed_ceiling >= slot_initial &&
                    probed_ceiling <= adaptive_options.max_depth
                ? probed_ceiling : adaptive_options.max_depth;
            hc.adaptive_probe_complete[slot] =
                adaptive_options.startup_probe_complete ? 1 : 0;
            if (adaptive_options.startup_probe_complete)
                hc.adaptive_initial_selection_pending[slot] = 2;
        }
    }

    device_alloc(hc.workspace.stencil_block_counts,
                 static_cast<size_t>(4) * hc.stencil_block_capacity);
    arena.allocate(hc.node_capacity);
    hc.arena = arena;

    // The Karras module owns all topology, sort, scan, and dense-map storage.
    // It updates both the captured host context and the caller's FMM view.
    initialize_karras_tree_workspace(runtime, dd, tree_dimension);

    CUDA_CHECK(cudaMemcpyToSymbol(c_grouped_direct_threshold_leaf,
                                  &direct_threshold_leaf, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grouped_direct_threshold_internal,
                                  &direct_threshold_internal, sizeof(uint64_t)));

    CUDA_CHECK(cudaMallocManaged(&runtime.terminal_error_managed, sizeof(int)));
    *runtime.terminal_error_managed = 0;
    hc.terminal_error = runtime.terminal_error_managed;

    CUDA_CHECK(cudaMalloc(&runtime.device_context,
                          sizeof(PersistentTreeContext)));
    CUDA_CHECK(cudaMemcpy(runtime.device_context, &hc, sizeof(hc),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&runtime.graph_upload_stream,
                                         cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&runtime.scheduler_stream,
                                         cudaStreamNonBlocking));

    if (adaptive_options.enabled) {
        for (int depth = adaptive_options.min_depth;
             depth <= adaptive_options.max_depth; ++depth) {
            hc.tree_depth = depth;
            if (tree_dimension == 2) capture_build_graph<2>(runtime, depth);
            else capture_build_graph<3>(runtime, depth);
        }
        hc.tree_depth = initial_depth;
        for (int slot = 0; slot < PERSISTENT_TREE_DIMENSIONS; ++slot)
            for (int depth = adaptive_options.min_depth;
                 depth <= adaptive_options.max_depth; ++depth)
                hc.depth_graph_exec[slot][depth] =
                    runtime.depth_graph_exec[slot][depth];
        CUDA_CHECK(cudaMemcpy(runtime.device_context, &hc, sizeof(hc),
                              cudaMemcpyHostToDevice));
    } else {
        if (tree_dimension == 2) capture_build_graph<2>(runtime, tree_depth);
        else capture_build_graph<3>(runtime, tree_depth);
    }
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));

    const cudaGraphExec_t adaptive_graph = adaptive_options.enabled
        ? runtime.depth_graph_exec[0][initial_depth] : nullptr;
    const cudaGraphExec_t selected_graph = adaptive_options.enabled
        ? adaptive_graph
        : (tree_dimension == 2 ? runtime.graph_exec_d2 : runtime.graph_exec_d3);
    const cudaGraphExec_t scheduler_d2 = selected_graph;
    const cudaGraphExec_t scheduler_d3 = selected_graph;
    capture_scheduler_graph(runtime, scheduler_d2, scheduler_d3);
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));

    dd.num_nodes = hc.node_capacity;
    dd.max_level = allocation_depth;
    dd.children_compact_count = hc.node_capacity * 8;
    runtime.initialized = true;
}

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
    double evaluation_time,
    bool common_planar_center,
    bool disable_direct_fallback)
{
    if (!runtime.initialized)
        throw std::logic_error("persistent tree runtime is not initialized");
    if (particle_count <= 0 ||
        particle_count > runtime.host_context.particle_capacity)
        throw std::out_of_range("particle count exceeds persistent tree capacity");

    PersistentTreeCommand command{};
    command.pos_x = pos_x;
    command.pos_y = pos_y;
    command.pos_z = pos_z;
    command.particle_count = particle_count;
    command.common_planar_center = common_planar_center;
    command.disable_direct_fallback = disable_direct_fallback;
    command.target_particle_count = target_particle_count;
    command.current_step = current_step;
    command.evaluation_time = evaluation_time;
    command.tree_dimension = tree_dimension;
    command.use_fixed_root = use_fixed_root ? 1 : 0;
    command.fixed_domain_min_x = fixed_domain_min_x;
    command.fixed_domain_min_y = fixed_domain_min_y;
    command.fixed_domain_min_z = fixed_domain_min_z;
    command.fixed_root_cell_size = fixed_root_cell_size;
    command.cutoff_radius = cutoff_radius;
    if (runtime.host_context.use_presorted_morton) {
        const auto& sorted = runtime.pending_presorted;
        if (!sorted.keys || !sorted.indices || sorted.count != particle_count ||
            sorted.depth != runtime.host_context.tree_depth || !use_fixed_root ||
            sorted.min_x != fixed_domain_min_x || sorted.min_y != fixed_domain_min_y ||
            sorted.min_z != fixed_domain_min_z || sorted.root_size != fixed_root_cell_size)
            throw std::logic_error("presorted Morton input does not match the current tree evaluation");
        command.presorted = sorted;
        runtime.pending_presorted = PresortedMortonInput{};
    }

    PersistentTreeMetadata metadata_patch{};
    metadata_patch.target_particle_count = target_particle_count;
    metadata_patch.cutoff_radius = cutoff_radius;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->metadata.target_particle_count,
        &metadata_patch.target_particle_count,
        sizeof(metadata_patch.target_particle_count),
        cudaMemcpyHostToDevice, consumer_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->metadata.cutoff_radius,
        &metadata_patch.cutoff_radius, sizeof(metadata_patch.cutoff_radius),
        cudaMemcpyHostToDevice, consumer_stream));

    PersistentTreeCommand* device_command =
        &runtime.device_context->command;
    CUDA_CHECK(cudaMemcpyAsync(device_command, &command, sizeof(command),
                               cudaMemcpyHostToDevice, consumer_stream));

    const unsigned int epoch = runtime.next_epoch++;
    driver_check(cuStreamWriteValue32(
        reinterpret_cast<CUstream>(consumer_stream),
        reinterpret_cast<CUdeviceptr>(&runtime.device_context->doorbell),
        epoch, CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue32(BFFMM doorbell)");
    // Start a scheduler graph for this command only.  It may begin before the
    // consumer stream reaches the doorbell, in which case it spins briefly;
    // it exits after tail-launching the matching build graph.
    CUDA_CHECK(cudaGraphLaunch(runtime.scheduler_graph_exec,
                               runtime.scheduler_stream));
    driver_check(cuStreamWaitValue32(
        reinterpret_cast<CUstream>(consumer_stream),
        reinterpret_cast<CUdeviceptr>(&runtime.device_context->completed_epoch),
        epoch, CU_STREAM_WAIT_VALUE_EQ),
        "cuStreamWaitValue32(BFFMM completion)");
}

PersistentTreeMetadata snapshot_persistent_tree_metadata(
    const PersistentTreeRuntime& runtime)
{
    if (!runtime.initialized || !runtime.device_context)
        throw std::logic_error("persistent tree runtime is not initialized");

    PersistentTreeMetadata metadata{};
    CUDA_CHECK(cudaMemcpyAsync(
        &metadata, &runtime.device_context->metadata, sizeof(metadata),
        cudaMemcpyDeviceToHost, runtime.graph_upload_stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
    return metadata;
}

void record_persistent_depth_evaluation(PersistentTreeRuntime& runtime,
                                        cudaStream_t consumer_stream)
{
    if (!runtime.initialized || !runtime.adaptive_options.enabled) return;
    persistent_record_depth_evaluation_kernel<<<1, 1, 0, consumer_stream>>>(
        runtime.device_context);
    CUDA_CHECK(cudaGetLastError());
}

void set_persistent_external_depth_control(
    PersistentTreeRuntime& runtime, bool enabled, cudaStream_t stream)
{
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled)
        throw std::logic_error(
            "external depth control requires an initialized adaptive tree runtime");
    const int value = enabled ? 1 : 0;
    runtime.adaptive_options.control_mode = enabled
        ? AdaptiveDepthControlMode::ExternalDistributed
        : AdaptiveDepthControlMode::Autonomous;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_external_control, &value,
        sizeof(value), cudaMemcpyHostToDevice, stream));
}

void set_persistent_external_depth_command(
    PersistentTreeRuntime& runtime,
    const ExternalDistributedDepthCommand& command,
    cudaStream_t stream)
{
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled)
        throw std::logic_error(
            "external depth command requires an initialized adaptive tree runtime");
    const auto valid_depth = [&](int depth) {
        return depth == 0 || (depth >= runtime.adaptive_options.min_depth &&
                              depth <= runtime.adaptive_options.max_depth);
    };
    if (command.launch_depth == 0 || !valid_depth(command.selected_depth) ||
        !valid_depth(command.candidate_depth) ||
        !valid_depth(command.launch_depth))
        throw std::invalid_argument(
            "external FMM depth command is outside the captured range");
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_external_command, &command,
        sizeof(command), cudaMemcpyHostToDevice, stream));
}

ExternalDistributedDepthEvaluation
snapshot_persistent_external_depth_evaluation(
    const PersistentTreeRuntime& runtime, cudaStream_t ordered_after_stream)
{
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled)
        return {};
    CUDA_CHECK(cudaStreamSynchronize(ordered_after_stream));
    PersistentTreeContext context{};
    CUDA_CHECK(cudaMemcpyAsync(&context, runtime.device_context, sizeof(context),
                               cudaMemcpyDeviceToHost,
                               runtime.graph_upload_stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
    if (!context.adaptive_external_control) return {};
    const int dimension = context.metadata.active_tree == 2 ? 2 : 3;
    const AdaptiveDepthState& state =
        context.adaptive_depth[0];
    ExternalDistributedDepthEvaluation result{};
    result.valid = context.metadata.error == 0 &&
                   context.metadata.evaluation_depth > 0;
    result.controller_epoch = context.adaptive_external_command.controller_epoch;
    result.evaluation_counter = state.evaluation_counter;
    result.tree_dimension = dimension;
    result.selected_depth = state.selected_depth;
    result.candidate_depth = state.candidate_depth;
    result.launch_depth = context.metadata.evaluation_depth;
    result.effective_max_depth = state.effective_max_depth;
    result.node_count = context.metadata.num_nodes;
    result.node_capacity = context.node_capacity;
    result.capacity_fallback_count = state.capacity_fallback_count;
    return result;
}

AdaptiveDepthTelemetry snapshot_adaptive_depth_telemetry(
    const PersistentTreeRuntime& runtime,
    cudaStream_t ordered_after_stream)
{
    if (!runtime.initialized || !runtime.device_context)
        throw std::logic_error("persistent tree runtime is not initialized");
    CUDA_CHECK(cudaStreamSynchronize(ordered_after_stream));
    PersistentTreeContext context{};
    CUDA_CHECK(cudaMemcpyAsync(
        &context, runtime.device_context, sizeof(context),
        cudaMemcpyDeviceToHost, runtime.graph_upload_stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
    AdaptiveDepthTelemetry telemetry{};
    telemetry.enabled = context.adaptive_depth_enabled != 0;
    telemetry.tree_dimension = runtime.host_context.command.tree_dimension;
    telemetry.node_capacity = context.node_capacity;
    telemetry.capacity_headroom = runtime.adaptive_options.capacity_headroom;
    telemetry.capacity_growth_trigger =
        context.adaptive_capacity_growth_trigger;
    telemetry.controller = context.adaptive_depth[0];
    return telemetry;
}

int recommended_sparse_node_capacity(const PersistentTreeRuntime& runtime)
{
    const AdaptiveDepthTelemetry telemetry =
        snapshot_adaptive_depth_telemetry(runtime);
    if (!telemetry.enabled) return telemetry.node_capacity;
    const int high_water = telemetry.controller.node_high_water;
    if (high_water <= 0) return telemetry.node_capacity;
    const int requested = static_cast<int>(std::ceil(
        static_cast<double>(high_water) *
        std::max(1.0f, telemetry.capacity_headroom)));
    const int strict = fixed_depth_node_upper_bound(
        runtime.host_context.particle_capacity,
        runtime.adaptive_options.max_depth,
        telemetry.tree_dimension);
    return std::max(high_water, std::min(strict, requested));
}

AdaptiveDepthCheckpoint snapshot_adaptive_depth_checkpoint(
    const PersistentTreeRuntime& runtime,
    cudaStream_t ordered_after_stream)
{
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled)
        throw std::logic_error("adaptive tree-depth runtime is not initialized");
    CUDA_CHECK(cudaStreamSynchronize(ordered_after_stream));
    PersistentTreeContext context{};
    CUDA_CHECK(cudaMemcpyAsync(
        &context, runtime.device_context, sizeof(context),
        cudaMemcpyDeviceToHost, runtime.graph_upload_stream));
    CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
    AdaptiveDepthCheckpoint checkpoint{};
    checkpoint.tree_dimension = runtime.host_context.command.tree_dimension;
    checkpoint.controller = context.adaptive_depth[0];
    checkpoint.diagnostic_sequence =
        context.adaptive_diagnostics.next_sequence > 0
            ? context.adaptive_diagnostics.next_sequence - 1 : 0;
    checkpoint.persisted_csv_sequence =
        context.adaptive_diagnostics.consumer_sequence > 0
            ? context.adaptive_diagnostics.consumer_sequence - 1 : 0;
    return checkpoint;
}

void restore_adaptive_depth_checkpoint(
    PersistentTreeRuntime& runtime,
    const AdaptiveDepthCheckpoint& checkpoint,
    cudaStream_t stream)
{
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled)
        throw std::logic_error("adaptive tree-depth runtime is not initialized");
    if (checkpoint.version != 2)
        throw std::invalid_argument("unsupported adaptive tree-depth checkpoint version");
    if (checkpoint.tree_dimension != runtime.host_context.command.tree_dimension)
        throw std::invalid_argument(
            "adaptive checkpoint tree dimension does not match the configured dimension");
    if (checkpoint.persisted_csv_sequence != checkpoint.diagnostic_sequence)
        throw std::invalid_argument(
            "adaptive checkpoint diagnostics were not drained losslessly");
    const AdaptiveDepthState& state = checkpoint.controller;
    if (state.selected_depth < runtime.adaptive_options.min_depth ||
        state.selected_depth > runtime.adaptive_options.max_depth ||
        state.candidate_depth < runtime.adaptive_options.min_depth ||
        state.candidate_depth > runtime.adaptive_options.max_depth ||
        state.effective_max_depth < runtime.adaptive_options.min_depth ||
        state.effective_max_depth > runtime.adaptive_options.max_depth)
        throw std::invalid_argument(
            "adaptive tree-depth checkpoint is outside the captured range");
    const int slot = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_depth[slot], &checkpoint.controller,
        sizeof(checkpoint.controller), cudaMemcpyHostToDevice, stream));
    const int probe_complete = 1;
    const int probe_clear = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_probe_complete[slot], &probe_complete,
        sizeof(probe_complete), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_probe_active[slot], &probe_clear,
        sizeof(probe_clear), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_probe_final[slot], &probe_clear,
        sizeof(probe_clear), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_initial_selection_pending[slot], &probe_clear,
        sizeof(probe_clear), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_skip_timing_once[slot], &probe_clear,
        sizeof(probe_clear), cudaMemcpyHostToDevice, stream));
    const uint64_t next_sequence = checkpoint.diagnostic_sequence + 1;
    const uint64_t consumer_sequence = next_sequence;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_diagnostics.next_sequence,
        &next_sequence, sizeof(next_sequence), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_diagnostics.consumer_sequence,
        &consumer_sequence, sizeof(consumer_sequence),
        cudaMemcpyHostToDevice, stream));
    const uint64_t dropped = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_diagnostics.dropped, &dropped,
        sizeof(dropped), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void consume_adaptive_depth_diagnostics(
    PersistentTreeRuntime& runtime,
    AdaptiveDepthDiagnostic* output,
    int output_capacity,
    int& output_count,
    uint64_t& dropped,
    cudaStream_t stream)
{
    output_count = 0;
    dropped = 0;
    if (!runtime.initialized || !runtime.device_context ||
        !runtime.adaptive_options.enabled || output_capacity <= 0 || !output)
        return;
    AdaptiveDepthDiagnosticRing ring{};
    CUDA_CHECK(cudaMemcpyAsync(
        &ring, &runtime.device_context->adaptive_diagnostics, sizeof(ring),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    uint64_t consumed = ring.consumer_sequence;
    output_count = extract_adaptive_depth_diagnostic_snapshot(
        ring, output, output_capacity, consumed);
    CUDA_CHECK(cudaMemcpyAsync(
        &runtime.device_context->adaptive_diagnostics.consumer_sequence,
        &consumed, sizeof(consumed), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    dropped = ring.dropped;
}

void emit_adaptive_capacity_transition(
    PersistentTreeRuntime& runtime,
    AdaptiveDepthTransition transition,
    int tree_dimension,
    cudaStream_t stream)
{
    if (!runtime.initialized || !runtime.adaptive_options.enabled) return;
    persistent_emit_capacity_transition_kernel<<<1, 1, 0, stream>>>(
        runtime.device_context, transition, tree_dimension);
    CUDA_CHECK(cudaGetLastError());
}

void throw_if_persistent_terminal_error(const PersistentTreeRuntime& runtime)
{
    if (!runtime.terminal_error_managed ||
        *runtime.terminal_error_managed == 0)
        return;
    const int error = *runtime.terminal_error_managed;
    std::ostringstream out;
    out << "adaptive BFFMM stopped cleanly after persistent tree error="
        << error << ": even min_depth="
        << runtime.adaptive_options.min_depth
        << " could not fit node_capacity="
        << runtime.host_context.node_capacity
        << ". No truncated tree or force result was accepted.";
    throw std::runtime_error(out.str());
}

void shutdown_persistent_tree_runtime(PersistentTreeRuntime& runtime) {
    if (!runtime.initialized && !runtime.device_context) return;

    if (runtime.device_context && runtime.scheduler_stream) {
        const unsigned int stop = 1;
        CUDA_CHECK(cudaMemcpyAsync(&runtime.device_context->shutdown, &stop,
                                   sizeof(stop), cudaMemcpyHostToDevice,
                                   runtime.graph_upload_stream));
        // The stop write and scheduler execute on independent streams.  Make
        // the write visible before waiting for a scheduler that may be
        // polling it, otherwise teardown has no cross-stream happens-before
        // edge and can wait indefinitely.
        CUDA_CHECK(cudaStreamSynchronize(runtime.graph_upload_stream));
        CUDA_CHECK(cudaStreamSynchronize(runtime.scheduler_stream));
        // Shutdown is outside the persistent execution interval; fence before
        // destroying graph executables or their captured workspace.
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (runtime.graph_exec_d2) CUDA_CHECK(cudaGraphExecDestroy(runtime.graph_exec_d2));
    if (runtime.graph_exec_d3) CUDA_CHECK(cudaGraphExecDestroy(runtime.graph_exec_d3));
    if (runtime.scheduler_graph_exec)
        CUDA_CHECK(cudaGraphExecDestroy(runtime.scheduler_graph_exec));
    if (runtime.graph_d2) CUDA_CHECK(cudaGraphDestroy(runtime.graph_d2));
    if (runtime.graph_d3) CUDA_CHECK(cudaGraphDestroy(runtime.graph_d3));
    for (int slot = 0; slot < PERSISTENT_TREE_DIMENSIONS; ++slot) {
        for (int depth = PERSISTENT_TREE_MIN_DEPTH;
             depth <= PERSISTENT_TREE_MAX_DEPTH; ++depth) {
            if (runtime.depth_graph_exec[slot][depth])
                CUDA_CHECK(cudaGraphExecDestroy(
                    runtime.depth_graph_exec[slot][depth]));
            if (runtime.depth_graph[slot][depth])
                CUDA_CHECK(cudaGraphDestroy(runtime.depth_graph[slot][depth]));
        }
    }
    if (runtime.scheduler_graph) CUDA_CHECK(cudaGraphDestroy(runtime.scheduler_graph));
    if (runtime.scheduler_stream) CUDA_CHECK(cudaStreamDestroy(runtime.scheduler_stream));
    if (runtime.graph_upload_stream) CUDA_CHECK(cudaStreamDestroy(runtime.graph_upload_stream));

    shutdown_karras_tree_workspace(runtime);

    auto& ws = runtime.host_context.workspace;
    device_free(ws.stencil_block_counts);
    if (runtime.device_context) CUDA_CHECK(cudaFree(runtime.device_context));
    if (runtime.terminal_error_managed)
        CUDA_CHECK(cudaFree(runtime.terminal_error_managed));

    runtime = PersistentTreeRuntime{};
}

}  // namespace body_fitted_fmm

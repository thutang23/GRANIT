// =============================================================================
// Target-Owned Grouped M2L — Single-Stage Contraction with Full G-Table
// =============================================================================
// Replaces the old 3-stage folded contraction with a single-stage grouped
// kernel against a precomputed full 48-orientation G-table (d_G_full).
//
// The shipped full G-table is composed offline from the canonical G
// tables and the A_R/B_R rotation matrices:
//   G_full[sc][lane] = sum_k sum_tc A_R[v2][k][sc] * G_canon[k][tc]
//                      * B_R[v2][lane][tc]
// computed in double precision, stored as float. This bakes the 48-fold
// rotation into ~12 MiB (reference modes) or ~14 MiB (TF32_TC) of device
// memory, replacing 3 stages of per-pair
// computation (84 FMA + shuffle, 84 float loads) with a single stage
// (28 FMA + shuffle, 28 float loads).
//
// Architecture:
//   - Reference/pre-sm80 path: 1 warp owns one target; lanes 0..27 compute its
//     coefficients with SIMT and lanes 28..31 are idle.
//   - TF32_TC on sm80+: a 256-thread block owns disjoint 256-node chunks, one
//     warp per child class collects batches of 16 targets, and all 32 lanes
//     issue WMMA m16n16k8 operations.
//   - Both paths drain generated cross/child-block routes in their original
//     order and perform one final local-expansion store; no target atomics.
//
// Single-stage contraction:
//   L_intermediate[lane] = Σ_sc G_full[sc][lane] · (M_source[sc] * 1/r^order_sc)
//                         · 1/r^(1+order_lane) · h^order_lane
//
// G_full layout per gap segment (indexed via __constant__ d_canonG_gap_info):
//   Gcb = float_offset + (cid * 48 + oid_gen) * G_STRIDE;
//   G_full[sc][lane] = d_G_full[Gcb + sc * 32 + lane];
//   G_STRIDE is 896 for reference modes and 1024 for TF32_TC, whose padded K
//   rows 28..31 are zero.
// The FMM_M2L_MODE compile-time switch affects only this 28-term G-table dot;
// source/post scaling and all expansion storage and accumulation remain FP64.
// =============================================================================

#include "../fmm/fmm_types.h"
#include "body_fitted_fmm_types.h"
#include "persistent_tree_traversal.cuh"
#include "grouped_stencil_traits.cuh"
#include "cuda_utils.h"
#include "remote_interaction_lists.cuh"
#include <cuda_runtime.h>
namespace body_fitted_fmm {
#include "m2l_group_diagnostics.cuh"
#ifndef FMM_M2L_MODE
#define FMM_M2L_MODE 3
#endif
#define FMM_M2L_MODE_FP64 0
#define FMM_M2L_MODE_FP32_SIMT_LINEAR 1
#define FMM_M2L_MODE_FP32_SIMT_DUAL 2
#define FMM_M2L_MODE_TF32_TC 3
#if FMM_M2L_MODE != FMM_M2L_MODE_FP64 && \
    FMM_M2L_MODE != FMM_M2L_MODE_FP32_SIMT_LINEAR && \
    FMM_M2L_MODE != FMM_M2L_MODE_FP32_SIMT_DUAL && \
    FMM_M2L_MODE != FMM_M2L_MODE_TF32_TC
#error "Unsupported FMM_M2L_MODE"
#endif
static constexpr int NC = fmm::FMM_ACTIVE_COEFFS;
static constexpr unsigned int M2L_SHFL_MASK = ((1u << NC) - 1u);
// Existing modes retain their 28x32 table.  The Tensor Core mode pads the K
// dimension as well, giving WMMA an aligned 32x32 matrix without touching the
// layouts used by the reference modes.
static constexpr int G_STRIDE =
    FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC ? 32 * 32 : NC * 32;
struct RotGGapInfo { int count; int float_offset; };
__constant__ RotGGapInfo d_canonG_gap_info[9];

template<bool Remote = false>
__device__ __forceinline__ far_real_t grouped_m2l_contribution(
    int target, int source, uint16_t packed_operator, int level_delta,
    int lane, const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full, far_real_t root_cell_size,
    const FMMRemoteRef* refs = nullptr,const FMMRemoteView* peers = nullptr)
{
    uint16_t cid = 0;
    uint8_t oid = 0;
    unpack_operator_id(packed_operator, cid, oid);
    const int gi = level_delta + 4;
    const RotGGapInfo& inf = d_canonG_gap_info[gi];
    if (oid >= 48 || cid >= static_cast<uint16_t>(inf.count))
        return FAR_REAL(0.0);

    const UnifiedNode& nA = nodes[target];
    const UnifiedNode& nB = grouped_m2l_source_node<Remote>(source, source_nodes, refs, peers);
    const far_real_t dx = FAR_CAST(nA.center.x - nB.center.x);
    const far_real_t dy = FAR_CAST(nA.center.y - nB.center.y);
    const far_real_t dz = FAR_CAST(nA.center.z - nB.center.z);
    const far_real_t distance = FAR_SQRT(dx * dx + dy * dy + dz * dz);
    far_real_t scales[7];
    if (distance >= FAR_REAL(1e-10)) {
        fmm::compute_scale_factors(distance, scales);
    } else {
        #pragma unroll
        for (int i = 0; i < 7; ++i) scales[i] = FAR_REAL(0.0);
    }

    const int order_j = fmm::get_coeff_order(lane);
    const far_real_t source_scale = order_j <= 0 ? scales[0]
        : (order_j == 1 ? scales[1]
        : (order_j == 2 ? scales[2]
        : (order_j == 3 ? scales[3] : scales[4])));
    const far_real_t mv = grouped_m2l_source_coefficient<Remote>(source, lane, multipoles, refs, peers) * source_scale;
    const size_t Gcb = static_cast<size_t>(inf.float_offset) +
        (static_cast<size_t>(cid) * 48 + oid) * static_cast<size_t>(G_STRIDE);
#if FMM_M2L_MODE == FMM_M2L_MODE_FP64
    far_real_t result = FAR_REAL(0.0);
    #pragma unroll
    for (int sc = 0; sc < NC; ++sc) {
        const far_real_t g = static_cast<far_real_t>(
            d_G_full[Gcb + static_cast<size_t>(sc) * 32 + lane]);
        result = fma(g, __shfl_sync(M2L_SHFL_MASK, mv, sc), result);
    }
#else
    // Keep the scale product above in FP64; quantize only the completed source
    // coefficient and perform the G-table dot with FP32 SIMT FMA.
    const float mv_fp32 = static_cast<float>(mv);
#if FMM_M2L_MODE == FMM_M2L_MODE_FP32_SIMT_LINEAR
    float result_fp32 = 0.0f;
    #pragma unroll
    for (int sc = 0; sc < NC; ++sc) {
        const float g = d_G_full[Gcb + static_cast<size_t>(sc) * 32 + lane];
        result_fp32 = fmaf(g, __shfl_sync(M2L_SHFL_MASK, mv_fp32, sc),
                           result_fp32);
    }
#else
    float result_even_fp32 = 0.0f;
    float result_odd_fp32 = 0.0f;
    #pragma unroll
    for (int sc = 0; sc < NC; sc += 2) {
        const float source_even = __shfl_sync(M2L_SHFL_MASK, mv_fp32, sc);
        const float source_odd = __shfl_sync(M2L_SHFL_MASK, mv_fp32, sc + 1);
        result_even_fp32 = fmaf(
            d_G_full[Gcb + static_cast<size_t>(sc) * 32 + lane], source_even,
            result_even_fp32);
        result_odd_fp32 = fmaf(
            d_G_full[Gcb + static_cast<size_t>(sc + 1) * 32 + lane], source_odd,
            result_odd_fp32);
    }
    const float result_fp32 = result_even_fp32 + result_odd_fp32;
#endif
    const far_real_t result = static_cast<far_real_t>(result_fp32);
#endif

    const int order_i = fmm::get_coeff_order(lane);
    const int power = 1 + order_i;
    far_real_t post_scale = power <= 1 ? scales[1]
        : (power == 2 ? scales[2]
        : (power == 3 ? scales[3]
        : (power == 4 ? scales[4]
        : (power == 5 ? scales[5] : scales[5] * scales[1]))));
    for (int p = 6; p < power; ++p) post_scale *= scales[1];

    const far_real_t cell_size = root_cell_size /
        static_cast<far_real_t>(1ull << nA.level);
    far_real_t h_pow = FAR_REAL(1.0);
    #pragma unroll
    for (int p = 0; p < 4; ++p)
        if (p < order_i) h_pow *= cell_size;
    return result * post_scale * h_pow;
}

template<int D>
__device__ __forceinline__ far_real_t grouped_m2l_target(
    int target, int lane, const GroupedInteractionArena& arena,
    const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const int* __restrict__ children,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full, far_real_t root_cell_size)
{
    using T = GroupedStencilTraits<D>;
    const UnifiedNode& target_node = nodes[target];
    const int child_q = grouped_target_child<D>(target_node.compact_morton);
    const GroupedTargetInteractions& work = arena.d_targets[target];
    far_real_t total = FAR_REAL(0.0);

    #pragma unroll
    for (int i = 0; i < T::CROSS_COUNT; ++i) {
        const GroupedSourceRoute route = grouped_cross_route(work, i);
        if (!grouped_route_valid(route) || (grouped_route_mask(route) & 1u))
            continue;
        total += grouped_m2l_contribution(
            target, grouped_route_node(route), T::cross(child_q, i).operator_id,
            1, lane, nodes, source_nodes, multipoles, d_G_full, root_cell_size);
    }
    #pragma unroll
    for (int block = 0; block < T::BLOCK_COUNT; ++block) {
        const GroupedSourceRoute route = grouped_block_route(work, block);
        if (!grouped_route_valid(route)) continue;
        const int source_parent = grouped_route_node(route);
        const uint8_t direct = grouped_route_mask(route);
        const uint8_t omit = grouped_block_omit_mask(work, block);
        const GeneratedChildBlockStencilEntry stencil = T::block(child_q, block);
        #pragma unroll
        for (int child = 0; child < T::CHILD_COUNT; ++child) {
            const uint8_t bit = static_cast<uint8_t>(1u << child);
            if (!(stencil.far_mask & bit) || (direct & bit) || (omit & bit))
                continue;
            const int source = grouped_child_node<D>(
                source_nodes, children, source_parent, child);
            if (source < 0) continue;
            total += grouped_m2l_contribution(
                target, source, stencil.operator_id[child], 0, lane, nodes, source_nodes,
                multipoles, d_G_full, root_cell_size);
        }
    }
    return total;
}

#if FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Eight warps let one block cover every octree child class.  A quadtree uses
// the first four warps.  Each block owns a disjoint grid-stride sequence of
// node chunks, so collecting a new batch never rescans the complete tree.
static constexpr int M2L_TC_BATCH = 16;
static constexpr int M2L_TC_PADDED_NC = 32;
static constexpr int M2L_TC_WARPS_PER_BLOCK = 8;
static constexpr int M2L_TC_NODE_CHUNK = 256;
// Measured route sparsity makes very small tiles slower than SIMT.  Require at
// least four live columns (25% N utilization); this still sends about 30% of
// representative active pairs through WMMA while avoiding the worst padding.
static constexpr int M2L_TC_SIMT_COLUMN_THRESHOLD = 4;
static constexpr int M2L_TC_THREAD_TOTALS =
    (NC * M2L_TC_BATCH) / 32;
static_assert(NC * M2L_TC_BATCH == M2L_TC_THREAD_TOTALS * 32,
              "Tensor Core output ownership must divide evenly over a warp");

__device__ __forceinline__ far_real_t grouped_m2l_source_scale(
    int coeff, const far_real_t* scales)
{
    const int order = fmm::get_coeff_order(coeff);
    return order <= 0 ? scales[0]
        : (order == 1 ? scales[1]
        : (order == 2 ? scales[2]
        : (order == 3 ? scales[3] : scales[4])));
}

// Apply one route slot to a same-child-class batch.  Lanes 0..15 have already
// placed their route-specific source node (or -1) in sources.  The G operator
// is common to all columns because child_q and route position are common.
template<bool Remote = false>
__device__ __forceinline__ void grouped_m2l_tc_apply_sources(
    const int* __restrict__ targets, int batch_count,
    uint16_t packed_operator, int level_delta,
    const int* __restrict__ sources,
    const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full,
    far_real_t* __restrict__ scales_shared,
    const far_real_t* __restrict__ h_shared,
    float* __restrict__ b_shared, float* __restrict__ c_shared,
    far_real_t (&totals)[M2L_TC_THREAD_TOTALS],
    const FMMRemoteRef* refs = nullptr,const FMMRemoteView* peers = nullptr)
{
    const int lane = threadIdx.x & 31;
    const unsigned int valid_columns = __ballot_sync(
        0xffffffffu, lane < batch_count && sources[lane] >= 0);
    const unsigned int active_columns = valid_columns & 0xffffu;
    if (active_columns == 0u) return;

    uint16_t cid = 0;
    uint8_t oid = 0;
    unpack_operator_id(packed_operator, cid, oid);
    const int gi = level_delta + 4;
    const RotGGapInfo& inf = d_canonG_gap_info[gi];
    if (oid >= 48 || cid >= static_cast<uint16_t>(inf.count)) return;

    // Compute distance powers once per active target/route, in FP64, and make
    // them available to both the B staging and the FP64 post-scaling step.
    if (lane < M2L_TC_BATCH) {
        far_real_t* column_scales =
            scales_shared + static_cast<size_t>(lane) * 7;
        if (lane < batch_count && sources[lane] >= 0) {
            const UnifiedNode& nA = nodes[targets[lane]];
            const UnifiedNode& nB = grouped_m2l_source_node<Remote>(sources[lane], source_nodes, refs, peers);
            const far_real_t dx = FAR_CAST(nA.center.x - nB.center.x);
            const far_real_t dy = FAR_CAST(nA.center.y - nB.center.y);
            const far_real_t dz = FAR_CAST(nA.center.z - nB.center.z);
            const far_real_t distance = FAR_SQRT(dx * dx + dy * dy + dz * dz);
            if (distance >= FAR_REAL(1e-10)) {
                fmm::compute_scale_factors(distance, column_scales);
            } else {
                #pragma unroll
                for (int p = 0; p < 7; ++p)
                    column_scales[p] = FAR_REAL(0.0);
            }
        } else {
            #pragma unroll
            for (int p = 0; p < 7; ++p)
                column_scales[p] = FAR_REAL(0.0);
        }
    }
    __syncwarp();

    const size_t Gcb = static_cast<size_t>(inf.float_offset) +
        (static_cast<size_t>(cid) * 48 + oid) * static_cast<size_t>(G_STRIDE);
    if (__popc(active_columns) < M2L_TC_SIMT_COLUMN_THRESHOLD) {
        // Sparse-route hybrid: every live column is a conventional 28-lane
        // matrix-vector product.  Source scaling is still completed in FP64,
        // and the dot uses the same dual FP32 accumulators as SIMT_DUAL.
        // Match the dense route's TF32 operands so changing batch occupancy
        // through domain partitioning does not change source precision.
        #pragma unroll 1
        for (int column = 0; column < M2L_TC_BATCH; ++column) {
            if (!(active_columns & (1u << column))) continue;
            if (lane < NC) {
                const far_real_t* column_scales =
                    scales_shared + static_cast<size_t>(column) * 7;
                const far_real_t scaled =
                    grouped_m2l_source_coefficient<Remote>(sources[column], lane, multipoles, refs, peers) *
                    grouped_m2l_source_scale(lane, column_scales);
                const float source_value = ::nvcuda::wmma::__float_to_tf32(
                    static_cast<float>(scaled));
                float result_even = 0.0f;
                float result_odd = 0.0f;
                #pragma unroll
                for (int sc = 0; sc < NC; sc += 2) {
                    const float source_even =
                        __shfl_sync(M2L_SHFL_MASK, source_value, sc);
                    const float source_odd =
                        __shfl_sync(M2L_SHFL_MASK, source_value, sc + 1);
                    result_even = fmaf(
                        d_G_full[Gcb + static_cast<size_t>(sc) * 32 + lane],
                        source_even, result_even);
                    result_odd = fmaf(
                        d_G_full[Gcb + static_cast<size_t>(sc + 1) * 32 + lane],
                        source_odd, result_odd);
                }
                c_shared[lane * M2L_TC_BATCH + column] =
                    result_even + result_odd;
            }
        }
    } else {
        // Dense route: B is [source coefficient][target].  Quantize the
        // completed FP64 scale product to TF32; padded entries remain zero.
        #pragma unroll
        for (int element = lane;
             element < M2L_TC_PADDED_NC * M2L_TC_BATCH; element += 32) {
            const int sc = element / M2L_TC_BATCH;
            const int column = element & (M2L_TC_BATCH - 1);
            float value = 0.0f;
            if (sc < NC && column < batch_count && sources[column] >= 0) {
                const far_real_t* column_scales =
                    scales_shared + static_cast<size_t>(column) * 7;
                const far_real_t scaled =
                    grouped_m2l_source_coefficient<Remote>(sources[column], sc, multipoles, refs, peers) *
                    grouped_m2l_source_scale(sc, column_scales);
                value = ::nvcuda::wmma::__float_to_tf32(
                    static_cast<float>(scaled));
            }
            b_shared[element] = value;
        }
        __syncwarp();

        using namespace ::nvcuda;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> accum_lo;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> accum_hi;
        wmma::fill_fragment(accum_lo, 0.0f);
        wmma::fill_fragment(accum_hi, 0.0f);

        #pragma unroll
        for (int k = 0; k < M2L_TC_PADDED_NC; k += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> a_lo;
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> a_hi;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> b;
            wmma::load_matrix_sync(a_lo, d_G_full + Gcb +
                static_cast<size_t>(k) * M2L_TC_PADDED_NC,
                M2L_TC_PADDED_NC);
            wmma::load_matrix_sync(a_hi, d_G_full + Gcb +
                static_cast<size_t>(k) * M2L_TC_PADDED_NC + 16,
                M2L_TC_PADDED_NC);
            wmma::load_matrix_sync(b, b_shared + k * M2L_TC_BATCH,
                                   M2L_TC_BATCH);
            wmma::mma_sync(accum_lo, a_lo, b, accum_lo);
            wmma::mma_sync(accum_hi, a_hi, b, accum_hi);
        }

        wmma::store_matrix_sync(c_shared, accum_lo, M2L_TC_BATCH,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(c_shared + 16 * M2L_TC_BATCH, accum_hi,
                                M2L_TC_BATCH, wmma::mem_row_major);
    }
    __syncwarp();

    // Source scaling no longer needs scales[0..4].  Replace them with the five
    // route/output-order post factors.  Cell-size powers were hoisted to batch
    // setup, so this costs five FP64 multiplies per live target/route instead
    // of fourteen divisions and power loops per lane/route.
    if (lane < M2L_TC_BATCH && (active_columns & (1u << lane))) {
        far_real_t* column_scales =
            scales_shared + static_cast<size_t>(lane) * 7;
        const far_real_t* column_h =
            h_shared + static_cast<size_t>(lane) * 5;
        #pragma unroll
        for (int order = 0; order < 5; ++order)
            column_scales[order] = column_scales[order + 1] * column_h[order];
    }
    __syncwarp();

    // Distribute the 28x16 active result evenly: every lane owns fourteen
    // (coefficient,target) pairs and keeps their route sums in FP64 registers.
    #pragma unroll
    for (int item = 0; item < M2L_TC_THREAD_TOTALS; ++item) {
        const int element = lane + item * 32;
        const int coeff = element / M2L_TC_BATCH;
        const int column = element & (M2L_TC_BATCH - 1);
        if (column < batch_count && sources[column] >= 0) {
            const far_real_t* column_scales =
                scales_shared + static_cast<size_t>(column) * 7;
            const far_real_t factor =
                column_scales[fmm::get_coeff_order(coeff)];
            totals[item] += static_cast<far_real_t>(c_shared[element]) * factor;
        }
    }
    __syncwarp();
}

template<int D>
__device__ __forceinline__ void grouped_m2l_tc_batch(
    int child_q, const int* __restrict__ targets, int batch_count,
    const GroupedInteractionArena& arena,
    const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const int* __restrict__ children,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full,
    fmm::LocalExpansion* __restrict__ local_expansions,
    far_real_t root_cell_size, int* __restrict__ sources,
    far_real_t* __restrict__ scales_shared,
    far_real_t* __restrict__ h_shared,
    float* __restrict__ b_shared, float* __restrict__ c_shared)
{
    using T = GroupedStencilTraits<D>;
    const int lane = threadIdx.x & 31;
    BF_M2L_CENSUS_BATCH(0,child_q,batch_count);
    far_real_t totals[M2L_TC_THREAD_TOTALS];
    #pragma unroll
    for (int item = 0; item < M2L_TC_THREAD_TOTALS; ++item)
        totals[item] = FAR_REAL(0.0);

    // Target level and root size do not change across routes.  Compute the
    // cell-size powers once for the batch and retain them in shared memory.
    if (lane < M2L_TC_BATCH && lane < batch_count) {
        const far_real_t cell_size = root_cell_size /
            static_cast<far_real_t>(1ull << nodes[targets[lane]].level);
        far_real_t h_pow = FAR_REAL(1.0);
        #pragma unroll
        for (int order = 0; order < 5; ++order) {
            h_shared[lane * 5 + order] = h_pow;
            h_pow *= cell_size;
        }
    }
    __syncwarp();

    // Keep the original per-target route order: cross slots first, followed by
    // block slots and their children.  Empty batch-wide routes return before
    // staging B or issuing any Tensor Core instruction.
    #pragma unroll 1
    for (int i = 0; i < T::CROSS_COUNT; ++i) {
        if (lane < M2L_TC_BATCH) {
            int source = -1;
            if (lane < batch_count) {
                const GroupedTargetInteractions& work =
                    arena.d_targets[targets[lane]];
                const GroupedSourceRoute route = grouped_cross_route(work, i);
                if (grouped_route_valid(route) &&
                    !(grouped_route_mask(route) & 1u))
                    source = grouped_route_node(route);
            }
            sources[lane] = source;
        }
        __syncwarp();
        BF_M2L_CENSUS_ROUTE(0,child_q,i,sources,batch_count);
        grouped_m2l_tc_apply_sources(
            targets, batch_count, T::cross(child_q, i).operator_id, 1,
            sources, nodes, source_nodes, multipoles, d_G_full,
            scales_shared, h_shared, b_shared, c_shared, totals);
    }

    #pragma unroll 1
    for (int block = 0; block < T::BLOCK_COUNT; ++block) {
        const GeneratedChildBlockStencilEntry stencil = T::block(child_q, block);
        #pragma unroll 1
        for (int child = 0; child < T::CHILD_COUNT; ++child) {
            if (lane < M2L_TC_BATCH) {
                int source = -1;
                if (lane < batch_count) {
                    const GroupedTargetInteractions& work =
                        arena.d_targets[targets[lane]];
                    const GroupedSourceRoute route =
                        grouped_block_route(work, block);
                    const uint8_t bit = static_cast<uint8_t>(1u << child);
                    if (grouped_route_valid(route) &&
                        (stencil.far_mask & bit) &&
                        !(grouped_route_mask(route) & bit) &&
                        !(grouped_block_omit_mask(work, block) & bit)) {
                        source = grouped_child_node<D>(
                            source_nodes, children, grouped_route_node(route), child);
                    }
                }
                sources[lane] = source;
            }
            __syncwarp();
            BF_M2L_CENSUS_ROUTE(0,child_q,T::CROSS_COUNT+block*T::CHILD_COUNT+child,sources,batch_count);
            grouped_m2l_tc_apply_sources(
                targets, batch_count, stencil.operator_id[child], 0,
                sources, nodes, source_nodes, multipoles, d_G_full,
                scales_shared, h_shared, b_shared, c_shared, totals);
        }
    }

    // Target ownership is exclusive to this block chunk and child-class warp.
    // Consequently these are ordinary FP64 additions, with no target atomics.
    #pragma unroll
    for (int item = 0; item < M2L_TC_THREAD_TOTALS; ++item) {
        const int element = lane + item * 32;
        const int coeff = element / M2L_TC_BATCH;
        const int column = element & (M2L_TC_BATCH - 1);
        if (column < batch_count)
            local_expansions[targets[column]].coeffs[coeff] += totals[item];
    }
    __syncwarp();
}

template<int D>
__device__ __forceinline__ void grouped_m2l_tc_scan_chunks(
    int child_q, const GroupedInteractionArena& arena,
    const PersistentTreeMetadata* __restrict__ metadata,
    const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const int* __restrict__ children,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full,
    fmm::LocalExpansion* __restrict__ local_expansions,
    int* __restrict__ targets, int* __restrict__ sources,
    far_real_t* __restrict__ scales_shared,
    far_real_t* __restrict__ h_shared,
    float* __restrict__ b_shared, float* __restrict__ c_shared,
    far_real_t root_cell_size)
{
    const int lane = threadIdx.x & 31;
    const unsigned int lane_lower_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);
    const int num_nodes = metadata->num_nodes;

    for (int chunk = static_cast<int>(blockIdx.x) * M2L_TC_NODE_CHUNK;
         chunk < num_nodes;
         chunk += static_cast<int>(gridDim.x) * M2L_TC_NODE_CHUNK) {
        const int chunk_end = min(chunk + M2L_TC_NODE_CHUNK, num_nodes);
        BF_M2L_CENSUS_SCAN(0,child_q,chunk_end-chunk);
        int batch_count = 0;

        #pragma unroll 1
        for (int offset = 0; offset < M2L_TC_NODE_CHUNK; offset += 32) {
            const int target = chunk + offset + lane;
            bool eligible = target < chunk_end;
            if (eligible) {
                const UnifiedNode& node = nodes[target];
                eligible = (node.flags & UNIFIED_NODE_FLAG_VALID) &&
                    node.level >= 2 && node.target_count != 0 &&
                    grouped_target_child<D>(node.compact_morton) == child_q;
            }
            unsigned int pending = __ballot_sync(0xffffffffu, eligible);

            // One 32-node slice can finish the current batch and seed another.
            // All lanes manipulate the same pending mask, keeping every WMMA
            // call warp-uniform even for sparse and partial batches.
            while (pending != 0u) {
                const int available = M2L_TC_BATCH - batch_count;
                const int selected = min(__popc(pending), available);
                const int rank = __popc(pending & lane_lower_mask);
                if ((pending & (1u << lane)) && rank < selected)
                    targets[batch_count + rank] = target;
                __syncwarp();
                batch_count += selected;
                #pragma unroll
                for (int drop = 0; drop < M2L_TC_BATCH; ++drop)
                    if (drop < selected) pending &= pending - 1u;

                if (batch_count == M2L_TC_BATCH) {
                    grouped_m2l_tc_batch<D>(
                        child_q, targets, batch_count, arena, nodes, source_nodes, children,
                        multipoles, d_G_full, local_expansions, root_cell_size,
                        sources, scales_shared, h_shared, b_shared, c_shared);
                    batch_count = 0;
                }
            }
        }

        if (batch_count != 0)
            grouped_m2l_tc_batch<D>(
                child_q, targets, batch_count, arena, nodes, source_nodes, children,
                multipoles, d_G_full, local_expansions, root_cell_size,
                sources, scales_shared, h_shared, b_shared, c_shared);
    }
}

#endif  // TF32 Tensor Core device implementation

__global__ void m2l_grouped_target_kernel(
    const GroupedInteractionArena arena,
    const PersistentTreeMetadata* __restrict__ metadata,
    const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const int* __restrict__ children,
    int node_capacity,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const float* __restrict__ d_G_full,
    fmm::LocalExpansion* __restrict__ local_expansions)
{
    (void)node_capacity;
#if FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(32) int
        targets_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_BATCH];
    __shared__ __align__(32) int
        sources_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_BATCH];
    __shared__ __align__(32) far_real_t
        scales_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_BATCH][7];
    __shared__ __align__(32) far_real_t
        h_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_BATCH][5];
    __shared__ __align__(32) float
        b_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_PADDED_NC * M2L_TC_BATCH];
    __shared__ __align__(32) float
        c_shared[M2L_TC_WARPS_PER_BLOCK][M2L_TC_PADDED_NC * M2L_TC_BATCH];

    const int warp_in_block = threadIdx.x >> 5;
    const int child_count = metadata->active_tree == 2 ? 4 : 8;
    if (warp_in_block >= child_count) return;
    const far_real_t root_cell_size =
        static_cast<far_real_t>(metadata->root_cell_size);
    if (metadata->active_tree == 2) {
        grouped_m2l_tc_scan_chunks<2>(
            warp_in_block, arena, metadata, nodes, source_nodes, children, multipoles,
            d_G_full, local_expansions, targets_shared[warp_in_block],
            sources_shared[warp_in_block], &scales_shared[warp_in_block][0][0],
            &h_shared[warp_in_block][0][0], b_shared[warp_in_block],
            c_shared[warp_in_block], root_cell_size);
    } else {
        grouped_m2l_tc_scan_chunks<3>(
            warp_in_block, arena, metadata, nodes, source_nodes, children, multipoles,
            d_G_full, local_expansions, targets_shared[warp_in_block],
            sources_shared[warp_in_block], &scales_shared[warp_in_block][0][0],
            &h_shared[warp_in_block][0][0], b_shared[warp_in_block],
            c_shared[warp_in_block], root_cell_size);
    }
#else
    // Reference modes, and the TF32_TC pre-sm80 fallback, retain the original
    // target-owned FP64/FP32-SIMT implementation.  In mode 3 the compile-time
    // dot selection above resolves to the dual-accumulator FP32 path.
    const int lane = threadIdx.x & 31;
    if (lane >= NC) return;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warp_stride = (gridDim.x * blockDim.x) >> 5;
    const far_real_t root_cell_size =
        static_cast<far_real_t>(metadata->root_cell_size);
    for (int target = warp; target < metadata->num_nodes; target += warp_stride) {
        const UnifiedNode& node = nodes[target];
        if (!(node.flags & UNIFIED_NODE_FLAG_VALID) || node.level < 2 ||
            node.target_count == 0) continue;
        const far_real_t contribution = metadata->active_tree == 2
            ? grouped_m2l_target<2>(target, lane, arena, nodes, source_nodes, children,
                                    multipoles, d_G_full, root_cell_size)
            : grouped_m2l_target<3>(target, lane, arena, nodes, source_nodes, children,
                                    multipoles, d_G_full, root_cell_size);
        local_expansions[target].coeffs[lane] += contribution;
    }
#endif
}

void run_m2l_grouped_targets(
    const GroupedInteractionArena& arena,
    const PersistentTreeMetadata* metadata,
    const UnifiedNode* nodes,
    const int* children,
    int node_capacity,
    const fmm::MultipoleExpansion* multipoles,
    const float* d_G_full,
    fmm::LocalExpansion* local_expansions,
    cudaStream_t stream,
    const UnifiedNode* source_nodes = nullptr,
    const int* source_children = nullptr)
{
    if (!source_nodes) source_nodes = nodes;
    // Target ancestry belongs to L2L; grouped M2L child routes index sources.
    if (!source_children) source_children = children;
    constexpr int block =
        FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC ? 256 : 128;
    constexpr int grid = 2048;
    m2l_grouped_target_kernel<<<grid, block, 0, stream>>>(
        arena, metadata, nodes, source_nodes, source_children, node_capacity, multipoles, d_G_full,
        local_expansions);
    CUDA_CHECK(cudaGetLastError());
}

#include "m2l_remote_kernel.cuh"

} // namespace

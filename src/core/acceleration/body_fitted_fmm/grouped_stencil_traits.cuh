#pragma once

#include "grouped_interaction_arena.cuh"
#include "unified_node.cuh"

namespace body_fitted_fmm {

template<int D> struct GroupedStencilTraits;

template<> struct GroupedStencilTraits<2> {
    static constexpr int CHILD_COUNT = GROUPED_D2_CHILD_COUNT;
    static constexpr int CROSS_COUNT = GROUPED_D2_CROSS_COUNT;
    static constexpr int ALL_FAR_COUNT = GROUPED_D2_ALL_FAR_COUNT;
    static constexpr int MIXED_COUNT = GROUPED_D2_MIXED_COUNT;
    static constexpr int BLOCK_COUNT = GROUPED_D2_BLOCK_COUNT;

    __device__ static GeneratedCrossStencilEntry cross(int q, int i) {
        return GROUPED_D2_CROSS[q][i];
    }
    __device__ static GeneratedChildBlockStencilEntry block(int q, int i) {
        return i < ALL_FAR_COUNT ? GROUPED_D2_ALL_FAR[q][i]
                                 : GROUPED_D2_MIXED[q][i - ALL_FAR_COUNT];
    }
    static GeneratedCrossStencilEntry host_cross(int q, int i) {
        return HOST_GROUPED_D2_CROSS[q][i];
    }
    static GeneratedChildBlockStencilEntry host_block(int q, int i) {
        return i < ALL_FAR_COUNT ? HOST_GROUPED_D2_ALL_FAR[q][i]
                                 : HOST_GROUPED_D2_MIXED[q][i - ALL_FAR_COUNT];
    }
};

template<> struct GroupedStencilTraits<3> {
    static constexpr int CHILD_COUNT = GROUPED_D3_CHILD_COUNT;
    static constexpr int CROSS_COUNT = GROUPED_D3_CROSS_COUNT;
    static constexpr int ALL_FAR_COUNT = GROUPED_D3_ALL_FAR_COUNT;
    static constexpr int MIXED_COUNT = GROUPED_D3_MIXED_COUNT;
    static constexpr int BLOCK_COUNT = GROUPED_D3_BLOCK_COUNT;

    __device__ static GeneratedCrossStencilEntry cross(int q, int i) {
        return GROUPED_D3_CROSS[q][i];
    }
    __device__ static GeneratedChildBlockStencilEntry block(int q, int i) {
        return i < ALL_FAR_COUNT ? GROUPED_D3_ALL_FAR[q][i]
                                 : GROUPED_D3_MIXED[q][i - ALL_FAR_COUNT];
    }
    static GeneratedCrossStencilEntry host_cross(int q, int i) {
        return HOST_GROUPED_D3_CROSS[q][i];
    }
    static GeneratedChildBlockStencilEntry host_block(int q, int i) {
        return i < ALL_FAR_COUNT ? HOST_GROUPED_D3_ALL_FAR[q][i]
                                 : HOST_GROUPED_D3_MIXED[q][i - ALL_FAR_COUNT];
    }
};

template<int D>
__host__ __device__ __forceinline__ int grouped_morton_slot(int generated_child) {
    int slot = 0;
    #pragma unroll
    for (int axis = 0; axis < D; ++axis)
        slot |= ((generated_child >> (D - 1 - axis)) & 1) << axis;
    return slot;
}

template<int D>
__host__ __device__ __forceinline__ int grouped_target_child(
    uint64_t compact_morton) {
    return grouped_morton_slot<D>(
        static_cast<int>(compact_morton & ((1u << D) - 1u)));
}

template<int D>
__device__ __forceinline__ int grouped_child_node(
    const UnifiedNode* nodes, const int* children, int parent, int child_slot) {
    return get_child_idx<D>(nodes, children, parent,
                            grouped_morton_slot<D>(child_slot));
}

}  // namespace body_fitted_fmm

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "grouped_stencil_tables.cuh"

namespace body_fitted_fmm {

struct GroupedSourceRoute {
    uint32_t node;
    uint8_t direct_mask;
};
inline constexpr uint32_t GROUPED_ROUTE_INVALID_NODE = 0xffffffffu;

__host__ __device__ inline constexpr GroupedSourceRoute grouped_invalid_route() {
    return {GROUPED_ROUTE_INVALID_NODE, 0u};
}
__host__ __device__ inline constexpr GroupedSourceRoute grouped_route(
    uint32_t node, uint8_t direct_mask) {
    return {node, direct_mask};
}
__host__ __device__ inline constexpr bool grouped_route_valid(
    GroupedSourceRoute route) {
    return route.node != GROUPED_ROUTE_INVALID_NODE;
}
__host__ __device__ inline constexpr int grouped_route_node(
    GroupedSourceRoute route) {
    return static_cast<int>(route.node);
}
__host__ __device__ inline constexpr uint8_t grouped_route_mask(
    GroupedSourceRoute route) {
    return route.direct_mask;
}

inline constexpr int GROUPED_MAX_CROSS_COUNT =
    GROUPED_D3_CROSS_COUNT > GROUPED_D2_CROSS_COUNT
        ? GROUPED_D3_CROSS_COUNT : GROUPED_D2_CROSS_COUNT;
inline constexpr int GROUPED_MAX_BLOCK_COUNT =
    GROUPED_D3_BLOCK_COUNT > GROUPED_D2_BLOCK_COUNT
        ? GROUPED_D3_BLOCK_COUNT : GROUPED_D2_BLOCK_COUNT;

struct GroupedTargetInteractions {
    uint32_t cross_node[GROUPED_MAX_CROSS_COUNT];
    uint32_t block_parent[GROUPED_MAX_BLOCK_COUNT];
    uint32_t cross_direct_bits;
    uint8_t block_direct_mask[GROUPED_MAX_BLOCK_COUNT];
    uint8_t block_omit_mask[GROUPED_MAX_BLOCK_COUNT];
    uint8_t _pad[2];
    int32_t target_parent;
};
static_assert(sizeof(GroupedTargetInteractions) == 176,
              "split grouped target record layout changed unexpectedly");

__host__ __device__ inline GroupedSourceRoute grouped_cross_route(
    const GroupedTargetInteractions& target, int slot) {
    return {target.cross_node[slot], static_cast<uint8_t>(
        (target.cross_direct_bits >> slot) & 1u)};
}
__host__ __device__ inline void grouped_set_cross_route(
    GroupedTargetInteractions& target, int slot, GroupedSourceRoute route) {
    target.cross_node[slot] = route.node;
    const uint32_t bit = 1u << slot;
    target.cross_direct_bits = route.direct_mask
        ? (target.cross_direct_bits | bit) : (target.cross_direct_bits & ~bit);
}
__host__ __device__ inline GroupedSourceRoute grouped_block_route(
    const GroupedTargetInteractions& target, int slot) {
    return {target.block_parent[slot], target.block_direct_mask[slot]};
}
__host__ __device__ inline void grouped_set_block_route(
    GroupedTargetInteractions& target, int slot, GroupedSourceRoute route) {
    target.block_parent[slot] = route.node;
    target.block_direct_mask[slot] = route.direct_mask;
}
__host__ __device__ inline uint8_t grouped_block_omit_mask(
    const GroupedTargetInteractions& target, int slot) {
    return target.block_omit_mask[slot];
}
__host__ __device__ inline void grouped_set_block_omit_mask(
    GroupedTargetInteractions& target, int slot, uint8_t mask) {
    target.block_omit_mask[slot] = mask;
}

struct GroupedInteractionArena {
    GroupedTargetInteractions* d_targets = nullptr;
    int* d_counters = nullptr;  // M2L same, M2L cross, P2P regular, P2P self
    int node_capacity = 0;

    void allocate(int capacity);
    void free();
    void reset(cudaStream_t stream = nullptr);

    size_t bytes() const {
        return static_cast<size_t>(node_capacity) *
                   sizeof(GroupedTargetInteractions) +
               4 * sizeof(int);
    }
};

}  // namespace body_fitted_fmm

#include "grouped_interaction_arena.cuh"

namespace body_fitted_fmm {

__constant__ uint64_t c_grouped_direct_threshold_leaf = 16;
__constant__ uint64_t c_grouped_direct_threshold_internal = 64;

__device__ __forceinline__ uint64_t grouped_direct_threshold(
    int target_level, bool target_is_leaf) {
    (void)target_level;
    return target_is_leaf ? c_grouped_direct_threshold_leaf
                          : c_grouped_direct_threshold_internal;
}

void GroupedInteractionArena::allocate(int capacity) {
    if (capacity <= 0) return;
    node_capacity = capacity;
    cudaMalloc(&d_targets, static_cast<size_t>(capacity) *
        sizeof(GroupedTargetInteractions));
    cudaMalloc(&d_counters, 4 * sizeof(int));
    reset();
}

void GroupedInteractionArena::free() {
    if (d_targets) cudaFree(d_targets);
    if (d_counters) cudaFree(d_counters);
    *this = GroupedInteractionArena{};
}

void GroupedInteractionArena::reset(cudaStream_t stream) {
    if (d_counters)
        cudaMemsetAsync(d_counters, 0, 4 * sizeof(int), stream);
}

}  // namespace body_fitted_fmm

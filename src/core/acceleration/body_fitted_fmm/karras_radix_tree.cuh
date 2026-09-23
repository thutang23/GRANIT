// =============================================================================
// karras_radix_tree.cuh — Karras 2012 binary-radix primitives
// =============================================================================
// Kernel-free device helpers for building a binary radix tree from sorted
// Morton codes. Ported from lbvh_fmm/bvh8_builder.cu, adapted for:
//   - 63-bit Morton codes (21-bit/axis) via 64-bit delta
//   - Index-fallback delta for duplicate Morton keys
//   - Non-batched (no clusters, no SAH top, no merge)
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace body_fitted_fmm {

// =============================================================================
// Minimal temporary node. Child indices are consumed in the constructing
// thread, so the persistent conversion stores only ancestry and leaf ranges.
// =============================================================================
struct KarrasBinaryNode {
    int32_t parent_idx;     // -1 for root
    int32_t particle_begin; // first unique-cell index (inclusive)
    int32_t particle_end;   // one past last unique-cell index (exclusive)
};

// =============================================================================
// Delta function with index-fallback for duplicate Morton codes.
// =============================================================================
// Returns the length of the longest common prefix (LCP) between sorted_codes[i]
// and sorted_codes[j].
//   - If codes differ: returns __clzll(codes[i] ^ codes[j]) ∈ [0, 63]
//   - If codes are equal: fallback to 63 + __clzll(i ^ j) ∈ [64, 127]
//   - Returns -1 if j is out of bounds [0, N)
//   - Returns 64 if i == j (sentinel — shouldn't happen for internal nodes)
// =============================================================================
__device__ __forceinline__
int karras_delta(const uint64_t* __restrict__ sorted_codes, int N, int i, int j) {
    if (j < 0 || j >= N) return -1;
    if (i == j) return 64;
    uint64_t xor_code = sorted_codes[i] ^ sorted_codes[j];
    if (xor_code != 0) {
        return __clzll(xor_code);
    }
    // Codes are equal — fall back to index bits
    uint64_t xor_idx = (uint64_t)i ^ (uint64_t)j;
    if (xor_idx == 0) return 128;
    return 63 + __clzll(xor_idx);
}

// =============================================================================
// Determine the range [first, last] of leaves covered by internal node `idx`.
// Port of bvh8_determine_range, adapted to use karras_delta.
// Returns make_uint2(first, last) where first < last.
// =============================================================================
__device__ __forceinline__
uint2 karras_determine_range(const uint64_t* sorted_codes, int N, int idx) {
    const int L_delta = karras_delta(sorted_codes, N, idx, idx - 1);
    const int R_delta = karras_delta(sorted_codes, N, idx, idx + 1);
    const int d = (R_delta > L_delta) ? 1 : -1;

    const int delta_min = (L_delta < R_delta) ? L_delta : R_delta;

    int l_max = 2;
    int delta = karras_delta(sorted_codes, N, idx, idx + d * l_max);
    while (delta > delta_min) {
        l_max <<= 1;
        delta = karras_delta(sorted_codes, N, idx, idx + d * l_max);
    }

    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        delta = karras_delta(sorted_codes, N, idx, idx + d * (l + t));
        if (delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0) {
        unsigned int tmp = idx; idx = jdx; jdx = tmp;
    }
    return make_uint2(idx, jdx);
}

// =============================================================================
// Find the split position within range [first, last].
// Port of bvh8_find_split, adapted to use karras_delta for duplicate robustness.
// Returns gamma ∈ [first, last-1] where the prefix changes.
// =============================================================================
__device__ __forceinline__
unsigned int karras_find_split(const uint64_t* sorted_codes, int N,
                               unsigned int first, unsigned int last) {
    // Use the same augmented-key LCP as range construction.  Falling back to
    // the sorted position for equal Morton codes is Karras's conceptual
    // (Morton,index) key and must also drive the split search; a midpoint split
    // is inconsistent for non-power-of-two duplicate runs.
    const int delta_node = karras_delta(sorted_codes, N, (int)first, (int)last);

    int split  = first;
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < (int)last) {
            const int delta = karras_delta(sorted_codes, N, (int)first, middle);
            if (delta > delta_node) {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

} // namespace body_fitted_fmm

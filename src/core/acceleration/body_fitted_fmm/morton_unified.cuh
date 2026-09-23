// =============================================================================
// Morton Unified: template<int D> Morton encoder/decoder
// =============================================================================
// Supports D=2 (quadtree) and D=3 (octree), both with 64-bit keys.
//
// This replaces the 3D-only morton_helpers.cuh with a dimension-generic version
// for the Phase 2 unified FMM.
// =============================================================================

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace body_fitted_fmm {

// =============================================================================
// Key type per dimension
// =============================================================================
template<int D> struct key_t;
template<> struct key_t<2> { using type = uint64_t; };
template<> struct key_t<3> { using type = uint64_t; };

template<int D> using key_t_v = typename key_t<D>::type;

// =============================================================================
// Bit spread functions
// =============================================================================

// 2D: bit spread (bit i → position 2*i)
// Maps a 32-bit integer to its 64-bit Morton interleaved form (bits 0-62).
// Each input bit i goes to position 2*i in the output.
__device__ __forceinline__ uint64_t expand2(uint64_t x) {
    x = (x | (x << 32)) & 0x00000000FFFFFFFFull;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FFull;
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x <<  2)) & 0x3333333333333333ull;
    x = (x | (x <<  1)) & 0x5555555555555555ull;
    return x;
}

// 3D: bit spread (bit i → position 3*i) — ported from bf_expand3 in morton_helpers.cuh
// Maps a 21-bit coordinate to its 63-bit Morton interleaved form.
__device__ __forceinline__ uint64_t expand3(uint64_t v) {
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

// =============================================================================
// Encode: (c0, c1, c2) → Morton key
// Only the first D coordinates are used; unused dims are ignored.
// =============================================================================

template<int D>
__device__ __forceinline__ key_t_v<D> morton_encode(uint32_t c0, uint32_t c1, uint32_t c2);

template<>
__device__ __forceinline__ uint64_t morton_encode<2>(uint32_t c0, uint32_t c1, uint32_t) {
    return expand2(c0) | (expand2(c1) << 1);
}

template<>
__device__ __forceinline__ uint64_t morton_encode<3>(uint32_t c0, uint32_t c1, uint32_t c2) {
    return expand3(c0) | (expand3(c1) << 1) | (expand3(c2) << 2);
}

// =============================================================================
// Decode: Morton key → (c0, c1, c2) at a given level
// Extracts `level` bits per axis from the Morton key.
// =============================================================================

template<int D>
__device__ __forceinline__ void morton_decode(key_t_v<D> code, int level,
                                               uint32_t& c0, uint32_t& c1, uint32_t& c2);

template<>
__device__ __forceinline__ void morton_decode<2>(uint64_t code, int level,
                                                   uint32_t& c0, uint32_t& c1, uint32_t&) {
    c0 = 0; c1 = 0;
    for (int l = 0; l < level; l++) {
        c0 |= ((uint32_t)((code >> (2 * l))     & 1ULL)) << l;
        c1 |= ((uint32_t)((code >> (2 * l + 1)) & 1ULL)) << l;
    }
}

template<>
__device__ __forceinline__ void morton_decode<3>(uint64_t code, int level,
                                                   uint32_t& c0, uint32_t& c1, uint32_t& c2) {
    c0 = 0; c1 = 0; c2 = 0;
    for (int l = 0; l < level; l++) {
        c0 |= ((uint32_t)((code >> (3 * l))     & 1ULL)) << l;
        c1 |= ((uint32_t)((code >> (3 * l + 1)) & 1ULL)) << l;
        c2 |= ((uint32_t)((code >> (3 * l + 2)) & 1ULL)) << l;
    }
}

// =============================================================================
// Encode cell Morton at a specific level (for cellToNode lookup)
// Shifts the level-bit cell path to the top of the key, matching particle
// Morton code alignment.
// =============================================================================

template<int D>
__device__ __forceinline__ key_t_v<D> encode_cell_morton(uint32_t c0, uint32_t c1, uint32_t c2, int level);

template<>
__device__ __forceinline__ uint64_t encode_cell_morton<2>(uint32_t c0, uint32_t c1, uint32_t, int level) {
    uint64_t morton = expand2(c0) | (expand2(c1) << 1);
    int shift = 62 - 2 * level;
    return (shift > 0) ? (morton << shift) : morton;
}

template<>
__device__ __forceinline__ uint64_t encode_cell_morton<3>(uint32_t c0, uint32_t c1, uint32_t c2, int level) {
    uint64_t morton = expand3(c0) | (expand3(c1) << 1) | (expand3(c2) << 2);
    int shift = 63 - 3 * level;
    return (shift > 0) ? (morton << shift) : morton;
}

}  // namespace body_fitted_fmm

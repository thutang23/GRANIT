// =============================================================================
// Dense Level Map — O(1) Morton→node lookup via flat indexed arrays
// =============================================================================
// Each level L has a flat array of size (1<<D)^L, indexed by compact_morton.
// Invalid entries use sentinel 0xFFFFFFFF (-1 as uint32_t).
// Max level: 7 for D=3 and 10 for D=2.
// Total memory: ~9.14 MiB for levels 0..7 in D=3.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "../fmm/fmm_types.h"

// =============================================================================
// Bit spread functions — host+device compatible
// =============================================================================

// 2D: bit spread (bit i → position 2*i)
__host__ __device__ inline uint64_t dense_expand2(uint64_t x) {
    x = (x | (x << 32)) & 0x00000000FFFFFFFFull;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FFull;
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x <<  2)) & 0x3333333333333333ull;
    x = (x | (x <<  1)) & 0x5555555555555555ull;
    return x;
}

// 3D: bit spread (bit i → position 3*i)
__host__ __device__ inline uint64_t dense_expand3(uint64_t v) {
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

// =============================================================================
// D-templated compact Morton computation
// =============================================================================

// Max dense levels per D
template<int D> struct DenseLevelTraits;
template<> struct DenseLevelTraits<2> { static constexpr int max_levels = 11; };
template<> struct DenseLevelTraits<3> { static constexpr int max_levels =  8; };

template<int D>
__host__ __device__ inline uint32_t dense_compute_compact_morton_d(int cx, int cy, int cz, int level) {
    (void)level;
    if constexpr (D == 2) {
        uint64_t ex = dense_expand2((uint64_t)(uint32_t)cx);
        uint64_t ey = dense_expand2((uint64_t)(uint32_t)cy);
        return (uint32_t)(ex | (ey << 1));
    } else {  // D == 3
        uint64_t ex = dense_expand3((uint64_t)(uint32_t)cx);
        uint64_t ey = dense_expand3((uint64_t)(uint32_t)cy);
        uint64_t ez = dense_expand3((uint64_t)(uint32_t)cz);
        return (uint32_t)(ex | (ey << 1) | (ez << 2));
    }
}

// Legacy alias (kept for backward compatibility)
__host__ __device__ inline uint32_t dense_compute_compact_morton(int cx, int cy, int cz, int level) {
    return dense_compute_compact_morton_d<3>(cx, cy, cz, level);
}

// =============================================================================
// Device Function: O(1) level→node lookup (D-aware level bounds)
// =============================================================================

template<int D>
__device__ __forceinline__
int dense_lookup(const fmm::FMMDeviceData& dd, int level, uint32_t compact_morton) {
    constexpr int MAX = DenseLevelTraits<D>::max_levels;
    if (level < 0 || level >= MAX) return -1;
    if (dd.level_dense_map[level] == nullptr) return -1;
    uint32_t idx = dd.level_dense_map[level][compact_morton];
    return (idx == 0xFFFFFFFF) ? -1 : (int)idx;
}

// Retain non-templated overload for D=3 (backward compat with older callers)
__device__ __forceinline__
int dense_lookup(const fmm::FMMDeviceData& dd, int level, uint32_t compact_morton) {
    return dense_lookup<3>(dd, level, compact_morton);
}

// =============================================================================
// Host Function: Compute level capacity per D
// =============================================================================

template<int D>
inline int dense_level_capacity_for(int level) {
    if (level < 0 || level >= fmm::FMM_MAX_DENSE_LEVELS) return 0;
    int cap = 1;
    for (int l = 0; l < level; l++) {
        cap *= (1 << D);  // 2^D per level
    }
    return cap;
}

// =============================================================================
// Host Function: Build dense level maps from FMMOctreeNode (D=3 only)
// =============================================================================

inline void build_dense_level_maps(
    fmm::FMMDeviceData& dd,
    const fmm::FMMOctreeNode* h_nodes,
    int num_nodes,
    int max_level,
    cudaStream_t stream)
{
    int cap[fmm::FMM_MAX_DENSE_LEVELS];
    cap[0] = 1;
    for (int l = 1; l < fmm::FMM_MAX_DENSE_LEVELS; l++) {
        cap[l] = cap[l-1] * 8;
    }

    int active_levels = (max_level < fmm::FMM_MAX_DENSE_LEVELS) ? (max_level + 1) : fmm::FMM_MAX_DENSE_LEVELS;

    uint32_t* h_dense[fmm::FMM_MAX_DENSE_LEVELS] = {nullptr};

    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        h_dense[l] = (uint32_t*)malloc(sz);
        if (!h_dense[l]) {
            fprintf(stderr, "[dense_level_map] malloc failed for level %d (size %zu)\n", l, sz);
            for (int pl = 0; pl < l; pl++) { free(h_dense[pl]); }
            return;
        }
        for (int i = 0; i < cap[l]; i++) {
            h_dense[l][i] = 0xFFFFFFFF;
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        int lvl = (int)h_nodes[i].level;
        if (lvl < 0 || lvl >= fmm::FMM_MAX_DENSE_LEVELS) continue;
        uint32_t cm = h_nodes[i].compact_morton;
        if (cm < (uint32_t)cap[lvl]) {
            h_dense[lvl][cm] = (uint32_t)i;
        }
    }

    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&dd.level_dense_map[l], sz));
        CUDA_CHECK(cudaMemcpyAsync(dd.level_dense_map[l], h_dense[l], sz,
                                    cudaMemcpyHostToDevice, stream));
        dd.level_dense_capacity[l] = cap[l];
        free(h_dense[l]);
    }
}

// =============================================================================
// Host Function: Build dense level maps from host-side UnifiedNode
// =============================================================================

template<int D>
inline void build_dense_level_maps_from_unified(
    fmm::FMMDeviceData& dd,
    const body_fitted_fmm::UnifiedNode* h_nodes,
    int num_nodes,
    int max_level,
    cudaStream_t stream)
{
    constexpr int MAX = DenseLevelTraits<D>::max_levels;
    int cap[MAX];
    cap[0] = 1;
    for (int l = 1; l < MAX; l++) {
        cap[l] = cap[l-1] * (1 << D);
    }

    int active_levels = (max_level < MAX) ? (max_level + 1) : MAX;

    uint32_t* h_dense[MAX] = {};
    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        h_dense[l] = (uint32_t*)malloc(sz);
        if (!h_dense[l]) {
            fprintf(stderr, "[dense_level_map] malloc failed for level %d (size %zu)\n", l, sz);
            for (int pl = 0; pl < l; pl++) { free(h_dense[pl]); }
            return;
        }
        for (int i = 0; i < cap[l]; i++) {
            h_dense[l][i] = 0xFFFFFFFF;
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        int lvl = (int)h_nodes[i].level;
        if (lvl < 0 || lvl >= MAX) continue;
        uint32_t cm = h_nodes[i].compact_morton;
        if (cm < (uint32_t)cap[lvl]) {
            h_dense[lvl][cm] = (uint32_t)i;
        }
    }

    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&dd.level_dense_map[l], sz));
        CUDA_CHECK(cudaMemcpyAsync(dd.level_dense_map[l], h_dense[l], sz,
                                    cudaMemcpyHostToDevice, stream));
        dd.level_dense_capacity[l] = cap[l];
        free(h_dense[l]);
    }
}

// =============================================================================
// Host Function: Free dense level map device memory
// =============================================================================

inline void free_dense_level_maps(fmm::FMMDeviceData& dd) {
    for (int l = 0; l < fmm::FMM_MAX_DENSE_LEVELS; l++) {
        if (dd.level_dense_map[l]) {
            cudaFree(dd.level_dense_map[l]);
            dd.level_dense_map[l] = nullptr;
        }
        dd.level_dense_capacity[l] = 0;
    }
}

// =============================================================================
// Device-side: build dense level maps from device nodes
// =============================================================================
// Declared here, defined in body_fitted_fmm_manager.cu (single TU).
// =============================================================================

#ifdef __CUDACC__
void build_dense_level_maps_device(
    fmm::FMMDeviceData& dd,
    int num_nodes,
    int max_level,
    cudaStream_t stream);

template<int D>
void build_dense_level_maps_device_unified(
    fmm::FMMDeviceData& dd,
    int num_nodes,
    int max_level,
    cudaStream_t stream);
#endif // __CUDACC__

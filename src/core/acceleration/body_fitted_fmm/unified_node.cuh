// =============================================================================
// UnifiedNode: coordinate-safe unified tree node for octree/quadtree
// =============================================================================
// Part of Phase 2 unified FMM. Replaces FMMOctreeNode for the new BFFMM path.
// 16-byte aligned for coalesced GPU access.
//
// Compact child allocation:
//   - childBase points into children_compact[] array
//   - childMask is a bitmask of which children exist
//   - get_child_idx<D>() uses popc to compute the offset into children_compact
// =============================================================================

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace body_fitted_fmm {

// 64-byte unified node struct for octree/quadtree.
// 16-byte aligned for coalesced access.
// Replaces FMMOctreeNode for Phase 2 unified FMM.
struct alignas(16) UnifiedNode {
    uint32_t begin;          // particle range start (sorted index)
    uint32_t count;          // particle count in this node
    uint32_t childBase;      // index into children_compact[] array (UINT32_MAX for leaf)
    uint8_t  childMask;      // bitmask of valid children (bit c set = child c exists)
    uint8_t  level;          // tree level (0 = root)
    uint8_t  flags;          // bit0: is_leaf, bit1: has_multipole,
                              // bit2: has_local, bit3: reserved (Phase 3: is_refined),
                              // bit4: reserved (Phase 3: is_virtual)
    uint8_t  _pad;            // explicit padding for alignment
    // World coordinates must remain FP64. Global-ring coordinates are large
    // compared with the local separations used by P2P/M2L; converting these
    // centers to FP32 would erase those separations before the kernels can
    // form their local differences.
    double3  center;          // 3D bounding sphere center (world frame)
    float    radius;          // 3D bounding sphere radius
    uint64_t compact_morton;  // dim-specific Morton encoding (D bits/axis x D axes)
    uint32_t target_count;    // central-image targets in this source node
    uint32_t global_count;    // complete cell population; zero uses legacy local population
};
static_assert(sizeof(UnifiedNode) == 64, "UnifiedNode must be 64 bytes");
static_assert(alignof(UnifiedNode) == 16 || alignof(UnifiedNode) == 8,
              "UnifiedNode should be at least 8-byte aligned");

// Flag bit definitions
constexpr uint8_t UNIFIED_NODE_FLAG_IS_LEAF      = 0x01;
constexpr uint8_t UNIFIED_NODE_FLAG_HAS_MULTIPOLE = 0x02;
constexpr uint8_t UNIFIED_NODE_FLAG_HAS_LOCAL    = 0x04;
constexpr uint8_t UNIFIED_NODE_FLAG_IS_REFINED   = 0x08;  // Phase 3
constexpr uint8_t UNIFIED_NODE_FLAG_IS_VIRTUAL  = 0x10;  // Phase 3
// Set for slots populated by the fixed-capacity persistent builder. Slots in
// [num_nodes, node_capacity) remain invalid and are ignored by capacity-grid
// downstream launches.
constexpr uint8_t UNIFIED_NODE_FLAG_VALID       = 0x20;

// Sentinel for "no children" (leaf node)
constexpr uint32_t UNIFIED_NODE_LEAF_CHILDBASE = 0xFFFFFFFFu;

// =============================================================================
// Device Helper: Get child node index via compact allocation
// =============================================================================
// Uses childBase + popc(childMask & ((1<<c)-1)) to index into children_compact[].
// D appears only in the caller's loop bound: for (int c = 0; c < (1<<D); c++)
//
// Returns -1 if child c doesn't exist (bit not set in childMask).
// =============================================================================
template<int D>
__device__ __forceinline__ int get_child_idx(
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children_compact,
    int parent_idx,
    int child_c)
{
    const UnifiedNode& p = nodes[parent_idx];
    if (p.childBase == UNIFIED_NODE_LEAF_CHILDBASE) return -1;
    if (!(p.childMask & (1u << child_c))) return -1;
    int offset = __popc(p.childMask & ((1u << child_c) - 1u));
    return children_compact[p.childBase + offset];
}

// Non-templated version (D not needed in the function itself, only in caller's loop)
__device__ __forceinline__ int get_child_idx(
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children_compact,
    int parent_idx,
    int child_c)
{
    const UnifiedNode& p = nodes[parent_idx];
    if (p.childBase == UNIFIED_NODE_LEAF_CHILDBASE) return -1;
    if (!(p.childMask & (1u << child_c))) return -1;
    int offset = __popc(p.childMask & ((1u << child_c) - 1u));
    return children_compact[p.childBase + offset];
}

}  // namespace body_fitted_fmm

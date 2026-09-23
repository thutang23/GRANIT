// =============================================================================
// LBVH (Linear Bounding Volume Hierarchy) Types
// =============================================================================
// Types for LBVH neighbor search implementation.

#ifndef CUDADEM_LBVH_TYPES_H
#define CUDADEM_LBVH_TYPES_H

#include "../neighbor_types.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace cudadem { namespace rt { class Backend; } }

namespace cudadem {
namespace lbvh {

// =============================================================================
// LBVH Node Structure
// =============================================================================

struct LBVHNode {
    uint32_t parent_idx;   // Parent node index
    uint32_t left_idx;     // Left child index
    uint32_t right_idx;    // Right child index
    uint32_t object_idx;   // Leaf: particle index; Internal: 0xFFFFFFFF
};

// =============================================================================
// Axis-Aligned Bounding Box
// =============================================================================

struct LBVHAABB {
    float4 upper;  // max corner (x, y, z, unused)
    float4 lower;  // min corner (x, y, z, unused)
};

// Kept in cache order, independently of particle-array migration/reordering.
struct LBVHCacheParticle {
    int32_t global_id;
    int32_t clump_id;
    float support_extent;
    float x, y, z;
};

// =============================================================================
// LBVH GPU State - Per-GPU Internal Tracking
// =============================================================================

struct LBVHGPUState {
    int32_t gpu_id;
    bool initialized;
    int32_t capacity;

    // RT owns only acceleration/traversal; the Verlet lifecycle below is
    // shared with the CUDA backend. CUDA buffers remain available for fallback.
    rt::Backend* rt_backend;
    NeighborBackend active_backend;
    bool rt_attempted;
    bool backend_timing_enabled;
    std::string rt_unavailable_reason;
    std::string backend_reason;
    std::string backend_log;
    cudaEvent_t timing_start, timing_build_end, timing_query_start, timing_query_end;

    // Particle ownership layout used by the cached topology and Verlet pairs.
    // Locals must occupy [0, tracked_num_local); ghosts follow them.
    int32_t tracked_num_particles;
    int32_t tracked_num_local;

    // BVH tree buffers (device memory)
    LBVHNode* nodes;
    LBVHAABB* aabbs;

    // Morton code buffers
    uint32_t* morton_codes;
    uint32_t* morton_codes_alt;
    unsigned long long* temp_unique_codes;  // scratch for (code<<32)|k merge in rebuild()
    void*  d_sort_temp;                     // CUB DeviceRadixSort temp storage
    size_t sort_temp_bytes;
    uint32_t* sorted_indices;
    uint32_t* sorted_indices_alt;

    // AABB propagation
    int32_t* atomic_flags;
    LBVHAABB* whole_domain;
    void* domain_reduce_temp;
    size_t domain_reduce_temp_bytes;

    // Topology caching
    float* snapshot_pos_x, *snapshot_pos_y, *snapshot_pos_z;
    float* d_max_disp_sq;
    float* h_max_disp_sq;
    cudaEvent_t disp_event;
    bool async_readback_pending;
    float cached_max_disp_sq;
    float rebuild_threshold_sq;
    bool threshold_resolved;

    // CUB workspace
    void* d_cub_disp_temp;
    size_t cub_disp_temp_bytes;
    float* d_disp_scratch;
    int32_t steps_since_rebuild;

    // Verlet buffer tracking
    int32_t* d_verlet_disp_scaled;
    float* h_pinned_verlet_disp;
    cudaEvent_t verlet_copy_event;
    bool verlet_readback_pending;
    float cached_verlet_disp_sq;
    int32_t steps_since_verlet_rebuild;
    uint64_t current_generation;        // Incremented on each rebuild

    // Verlet neighbor cache
    VerletNeighborCache verlet_cache;
    LBVHCacheParticle* verlet_particles;
    float cached_verlet_skin;
    float cached_periodic_bounds[6];
    uint8_t cached_periodic_axes;

    // Lazily allocated SPH candidate and CSR storage. DEM never consumes it.
    ContactListBuffer sph_pairs;
    unsigned long long* sph_pair_count = nullptr;
    int32_t* sph_overflow = nullptr;
    uint64_t* sph_offsets = nullptr;
    uint64_t* sph_cursors = nullptr;
    int32_t* sph_neighbors = nullptr;
    size_t sph_neighbor_capacity = 0;
    void* sph_scan_temp = nullptr;
    size_t sph_scan_bytes = 0;
    bool sph_ready = false;
    bool sph_cache_valid = false;
    float sph_cached_skin = 0.0f;
    float sph_cached_cutoff = 0.0f;

    // Phase 1 & 2: Diagnostics for optimization
    int64_t* debug_stack_overflows;      // Stack overflow counter
    int64_t* debug_contact_overflows;    // Contact buffer overflow counter

    // Statistics
    NeighborSearchStats stats;

    LBVHGPUState()
        : gpu_id(-1), initialized(false), capacity(0),
          rt_backend(nullptr), active_backend(NeighborBackend::CUDA), rt_attempted(false),
          backend_timing_enabled(false),
          timing_start(nullptr), timing_build_end(nullptr),
          timing_query_start(nullptr), timing_query_end(nullptr),
          tracked_num_particles(-1), tracked_num_local(-1),
          nodes(nullptr), aabbs(nullptr),
          morton_codes(nullptr), morton_codes_alt(nullptr),
          temp_unique_codes(nullptr),
          d_sort_temp(nullptr), sort_temp_bytes(0),
          sorted_indices(nullptr), sorted_indices_alt(nullptr),
          atomic_flags(nullptr),
          whole_domain(nullptr), domain_reduce_temp(nullptr), domain_reduce_temp_bytes(0),
          snapshot_pos_x(nullptr), snapshot_pos_y(nullptr), snapshot_pos_z(nullptr),
          d_max_disp_sq(nullptr), h_max_disp_sq(nullptr),
          disp_event(nullptr),
          async_readback_pending(false), cached_max_disp_sq(1e30f),
          rebuild_threshold_sq(0.0f), threshold_resolved(false),
          d_cub_disp_temp(nullptr), cub_disp_temp_bytes(0),
          d_disp_scratch(nullptr), steps_since_rebuild(0),
          d_verlet_disp_scaled(nullptr), h_pinned_verlet_disp(nullptr),
          verlet_copy_event(nullptr), verlet_readback_pending(false),
          cached_verlet_disp_sq(0.0f), steps_since_verlet_rebuild(0),
          current_generation(0), verlet_particles(nullptr),
          cached_verlet_skin(0.0f), cached_periodic_bounds{}, cached_periodic_axes(0),
          debug_stack_overflows(nullptr), debug_contact_overflows(nullptr) {}
};

// =============================================================================
// AABB Operations
// =============================================================================

__device__ __host__ __forceinline__
bool lbvh_intersects(const LBVHAABB& a, const LBVHAABB& b) {
    if (a.upper.x < b.lower.x || b.upper.x < a.lower.x) return false;
    if (a.upper.y < b.lower.y || b.upper.y < a.lower.y) return false;
    if (a.upper.z < b.lower.z || b.upper.z < a.lower.z) return false;
    return true;
}

__device__ __host__ __forceinline__
LBVHAABB lbvh_merge(const LBVHAABB& a, const LBVHAABB& b) {
    LBVHAABB merged;
    merged.upper.x = fmaxf(a.upper.x, b.upper.x);
    merged.upper.y = fmaxf(a.upper.y, b.upper.y);
    merged.upper.z = fmaxf(a.upper.z, b.upper.z);
    merged.upper.w = 0.0f;
    merged.lower.x = fminf(a.lower.x, b.lower.x);
    merged.lower.y = fminf(a.lower.y, b.lower.y);
    merged.lower.z = fminf(a.lower.z, b.lower.z);
    merged.lower.w = 0.0f;
    return merged;
}

} // namespace lbvh
} // namespace cudadem

#endif // CUDADEM_LBVH_TYPES_H

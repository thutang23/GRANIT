// =============================================================================
// Neighbor Search Shared Types
// =============================================================================
// Types shared between CLL and LBVH implementations.
// Single header for all neighbor-related type definitions.

#ifndef CUDADEM_NEIGHBOR_TYPES_H
#define CUDADEM_NEIGHBOR_TYPES_H

#include "../constant/command_line_interface.h"  // For NeighborMethod enum
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace cudadem {

// =============================================================================
// Forward Declarations (for type safety without header coupling)
// =============================================================================

namespace lbvh {
struct LBVHNode;
struct LBVHAABB;
} // namespace lbvh

// Note: NeighborMethod enum is defined in command_line_interface.h
// We include that header to use it here.

// =============================================================================
// Contact Pair - Output of Narrow Phase Detection
// =============================================================================

// Geometry belongs to force evaluation (or explicit diagnostics), not the
// candidate list. Indices retain their canonical ownership/history meaning.
struct ContactPair {
    int32_t i;
    int32_t j;
    // Image of j: position_j += image * box_length, with the shear correction
    // applied by the force kernel when Lees-Edwards boundaries are enabled.
    int8_t image_x, image_y, image_z;
    uint8_t reserved; // Explicit padding keeps copied records fully initialized.

    __host__ __device__ ContactPair()
        : i(-1), j(-1), image_x(0), image_y(0), image_z(0), reserved(0) {}

    __host__ __device__ ContactPair(int32_t _i, int32_t _j,
                                  int8_t _ix = 0, int8_t _iy = 0, int8_t _iz = 0)
        : i(_i), j(_j), image_x(_ix), image_y(_iy), image_z(_iz), reserved(0) {}
};
static_assert(sizeof(ContactPair) == 12, "Contact candidates must remain compact");
static_assert(alignof(ContactPair) == alignof(int32_t), "Contact pair alignment changed");
static_assert(std::is_trivially_copyable<ContactPair>::value, "Pairs must support device copies");
static_assert(std::is_standard_layout<ContactPair>::value, "Pairs must have a stable layout");


// =============================================================================
// Contact List Buffer - Dynamic Storage for Detected Pairs
// =============================================================================

struct ContactListBuffer {
    ContactPair* pairs;          // Device array of contact pairs
    int32_t* num_contacts;       // Device counter (atomically incremented)
    int32_t max_contacts;        // Capacity
    int32_t h_num_contacts;      // Host mirror for readback

    ContactListBuffer()
        : pairs(nullptr), num_contacts(nullptr),
          max_contacts(0), h_num_contacts(0) {}
};

// =============================================================================
// Verlet Neighbor List Cache
// =============================================================================
// Caches neighbor pairs (i, j, image) for reuse when particles haven't moved.
// Geometry (overlap, distance, normal) is recomputed from current positions.

struct VerletNeighborCache {
    // Cached neighbor pairs (indices only)
    int32_t* pair_i;           // Device array: first particle index
    int32_t* pair_j;           // Device array: second particle index
    int8_t* pair_image_x;      // Periodic image offset (-1, 0, +1)
    int8_t* pair_image_y;
    int8_t* pair_image_z;
    int32_t* num_pairs;        // Device counter (not used, kept for future)
    int32_t max_pairs;         // Capacity
    int32_t h_num_pairs;       // Host mirror
    
    // Cache state
    bool cache_valid;          // True if pairs can be reused
    bool cache_allocated;      // True if memory allocated
    
    VerletNeighborCache()
        : pair_i(nullptr), pair_j(nullptr),
          pair_image_x(nullptr), pair_image_y(nullptr), pair_image_z(nullptr),
          num_pairs(nullptr), max_pairs(0), h_num_pairs(0),
          cache_valid(false), cache_allocated(false) {}
};

// =============================================================================
// Neighbor Search Statistics
// =============================================================================

struct NeighborSearchStats {
    int32_t num_rebuilds;
    int32_t num_contacts_found;
    int32_t cells_with_particles;
    float avg_particles_per_cell;
    float rebuild_time_ms;
    float search_time_ms;
    int32_t lbvh_full_rebuilds;
    int32_t lbvh_fast_path_steps;
    // Cumulative milliseconds: build/update/query use CUDA events, while init
    // uses host elapsed time. Cache-only calls add zero to device components.
    // Build/update include bounds generation; query includes candidate emission.
    double backend_build_time_ms;
    double backend_update_time_ms;
    double backend_query_time_ms;
    double backend_init_time_ms;
    
    // Verlet cache statistics
    int32_t cache_hits;           // Steps where cache was reused
    int32_t cache_misses;         // Steps where cache was rebuilt
    int32_t cache_invalidations;  // Times cache was invalidated

    NeighborSearchStats()
        : num_rebuilds(0), num_contacts_found(0),
          cells_with_particles(0), avg_particles_per_cell(0),
          rebuild_time_ms(0), search_time_ms(0),
          lbvh_full_rebuilds(0), lbvh_fast_path_steps(0),
          backend_build_time_ms(0), backend_update_time_ms(0),
          backend_query_time_ms(0), backend_init_time_ms(0),
          cache_hits(0), cache_misses(0), cache_invalidations(0) {}
};

// =============================================================================
// NeighborData - Data Carrier for SimulationState
// =============================================================================
// Replaces CellLinkedList with a method-agnostic interface.
// Populated by neighbor search implementations, accessible for kernel use.

struct NeighborData {
    // ----- Common Data -----
    int32_t num_particles;       // Particles this search was built for
    NeighborMethod active_method; // Which method populated this data
    ContactListBuffer* contacts;  // Set by caller before simulation_step()

    // ----- Cell-Linked List Data (CLL method) -----
    float cell_size;
    int32_t grid_dim_x, grid_dim_y, grid_dim_z;
    int32_t total_cells;
    int32_t morton_capacity;
    float domain_min_x, domain_min_y, domain_min_z;
    float domain_max_x, domain_max_y, domain_max_z;

    // Cell arrays (device memory)
    int32_t* cell_start;         // First particle index per cell (Morton indexed)
    int32_t* cell_end;           // Last particle index + 1 per cell
    int32_t* particle_indices;   // Particles sorted by cell
    int32_t* cell_ids;           // Cell ID for each particle

    // Backward compatibility aliases
    int32_t* particle_cell;      // Alias for cell_ids
    int32_t cells_x, cells_y, cells_z;  // Aliases for grid_dim_*

    // ----- LBVH Data (LBVH method) -----
    lbvh::LBVHNode* nodes;             // BVH tree nodes (2N-1)
    lbvh::LBVHAABB* aabbs;             // Bounding boxes (2N-1)
    uint32_t* sorted_indices;    // Particle indices sorted by Morton code
    int32_t num_internal_nodes;  // N-1 internal nodes

    // Directed SPH candidates, shared by CUDA and RT and reused by every
    // physics pass. Geometry and the exact current h_ij cutoff are evaluated
    // by the consumer; offsets are 64-bit because each pair has two entries.
    uint64_t* sph_offsets;
    int32_t* sph_neighbors;
    bool sph_adjacency_valid;

    // ===== Shared Morton codes for FMM =====
    unsigned long long* morton_codes_64;  // 64-bit Morton codes (shared)
    uint64_t rebuild_generation;          // Incremented on each rebuild
    bool morton_data_valid;               // True if codes are current

    NeighborData()
        : num_particles(0),
          active_method(NeighborMethod::CELL_LINKED_LIST),
          contacts(nullptr),
          cell_size(0), grid_dim_x(0), grid_dim_y(0), grid_dim_z(0),
          total_cells(0), morton_capacity(0),
          domain_min_x(0), domain_min_y(0), domain_min_z(0),
          domain_max_x(0), domain_max_y(0), domain_max_z(0),
          cell_start(nullptr), cell_end(nullptr),
          particle_indices(nullptr), cell_ids(nullptr),
          particle_cell(nullptr), cells_x(0), cells_y(0), cells_z(0),
          nodes(nullptr), aabbs(nullptr), sorted_indices(nullptr),
          num_internal_nodes(0),
          sph_offsets(nullptr), sph_neighbors(nullptr), sph_adjacency_valid(false),
          morton_codes_64(nullptr),
          rebuild_generation(0),
          morton_data_valid(false) {}
};

// =============================================================================
// GPU Constant Memory Parameters
// =============================================================================

struct NeighborParamsGPU {
    int32_t grid_dim_x, grid_dim_y, grid_dim_z;
    int32_t total_cells;
    int32_t morton_capacity;
    float cell_size;
    float inv_cell_size;
    float domain_min_x, domain_min_y, domain_min_z;
    float domain_max_x, domain_max_y, domain_max_z;
    bool periodic_x, periodic_y, periodic_z;
    float contact_threshold;
    float verlet_skin;
    float kernel_cutoff_factor;   // SPH support radius factor (default 2.0)
};

// extern declaration - defined in cll/cll_kernels.cu
extern __constant__ NeighborParamsGPU dev_neighbor_params;

// =============================================================================
// Host-side Morton Code Utilities
// =============================================================================

inline uint32_t expand_bits_host(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint32_t morton_3d_host(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits_host(x) | (expand_bits_host(y) << 1) | (expand_bits_host(z) << 2);
}

inline int32_t compute_morton_capacity(int32_t gx, int32_t gy, int32_t gz) {
    if (gx <= 0 || gy <= 0 || gz <= 0) return 1;
    uint32_t max_code = morton_3d_host(
        static_cast<uint32_t>(gx - 1),
        static_cast<uint32_t>(gy - 1),
        static_cast<uint32_t>(gz - 1));
    return static_cast<int32_t>(max_code) + 1;
}

// =============================================================================
// Utility Functions
// =============================================================================

inline int32_t estimate_max_contacts(int32_t num_particles) {
    return num_particles * 6;  // ~6 contacts per particle in 3D packing
}

inline float compute_cell_size(float max_radius, float factor = 1.1f) {
    return 2.0f * max_radius * factor;
}

// =============================================================================
// Device Functions for Cell ID Computation
// =============================================================================

#ifdef __CUDACC__
// Compute 3D cell indices from position
__device__ __forceinline__
void position_to_cell_3d(
    float px, float py, float pz,
    int32_t& ix, int32_t& iy, int32_t& iz
) {
    ix = static_cast<int32_t>((px - dev_neighbor_params.domain_min_x) *
                               dev_neighbor_params.inv_cell_size);
    iy = static_cast<int32_t>((py - dev_neighbor_params.domain_min_y) *
                               dev_neighbor_params.inv_cell_size);
    iz = static_cast<int32_t>((pz - dev_neighbor_params.domain_min_z) *
                               dev_neighbor_params.inv_cell_size);

    // Clamp to valid range
    ix = max(0, min(ix, dev_neighbor_params.grid_dim_x - 1));
    iy = max(0, min(iy, dev_neighbor_params.grid_dim_y - 1));
    iz = max(0, min(iz, dev_neighbor_params.grid_dim_z - 1));
}

// Convert 3D cell indices to linear cell ID
__device__ __forceinline__
int32_t cell_3d_to_linear(int32_t ix, int32_t iy, int32_t iz) {
    return iz * dev_neighbor_params.grid_dim_y * dev_neighbor_params.grid_dim_x +
           iy * dev_neighbor_params.grid_dim_x + ix;
}

// Compute linear cell ID from position
__device__ __forceinline__
int32_t position_to_cell_id(float px, float py, float pz) {
    int32_t ix, iy, iz;
    position_to_cell_3d(px, py, pz, ix, iy, iz);
    return cell_3d_to_linear(ix, iy, iz);
}
#endif // __CUDACC__

} // namespace cudadem

#endif // CUDADEM_NEIGHBOR_TYPES_H

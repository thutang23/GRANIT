// =============================================================================
// Cell-Linked List Implementation for DEM Simulation
// =============================================================================
// GPU-optimized neighbor search using:
// - Thrust sorting for particle reordering by cell ID
// - Historical state comparison to skip unnecessary rebuilds
// - 27-cell neighborhood search with periodic boundary support
//
// Performance Optimizations:
// - Track previous cell IDs to detect actual cell changes
// - Reorder all particle attributes after sorting for coalesced access
// - Use shared memory for cell boundary storage in broad phase
// - Warp-level primitives for efficient reduction
//
// Reference: GPU Gems 3, Chapter 32 - Broad-Phase Collision Detection
// =============================================================================

#include "cll_manager.h"
#include "../neighbor_types.h"
#include "../neighbor_api.h"  // For NeighborSearch contact buffer helpers
#include "../../wall/wall.h"  // For PeriodicBoundaryConfig
#include "../../shear/shear_box.h"
#include "../lbvh/max_support_reduce.cuh"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#ifndef CUDADEM_CPU_MOCK
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
// CUB for efficient reduction
#include <cub/cub.cuh>
#endif
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

namespace cudadem {

// =============================================================================
// GPU Constant Memory Definition
// =============================================================================
// Must be in cudadem namespace (not cudadem::cll) because device functions in
// neighbor.h reference this symbol and they are in the cudadem namespace.
// =============================================================================

__constant__ NeighborParamsGPU dev_neighbor_params;

namespace cll {

struct ShearProfileRange {
    explicit ShearProfileRange(const char* name) {
#ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ShearProfileRange() {
#ifdef CUDADEM_ENABLE_PROFILING
        nvtxRangePop();
#endif
    }
};

static void shear_cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct ShearGrid {
    double min_x, min_y, max_x, max_y, cell_size, inverse_cell_size;
    double length_x, length_y, reduced_shift, search_range;
    int32_t nx, ny;
};

__global__ void shear_bin_kernel(const double* x, const double* y,
                                  int32_t* heads, int32_t* next,
                                  int32_t count, ShearGrid grid) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int32_t cx = static_cast<int32_t>(floor((x[i] - grid.min_x) * grid.inverse_cell_size));
    int32_t cy = static_cast<int32_t>(floor((y[i] - grid.min_y) * grid.inverse_cell_size));
    cx = max(0, min(cx, grid.nx - 1));
    cy = max(0, min(cy, grid.ny - 1));
    next[i] = atomicExch(&heads[static_cast<int64_t>(cy) * grid.nx + cx], i);
}

__global__ void shear_query_kernel(
    const double* x, const double* y, const double* z,
    const float* support, const float* radius, const int32_t* clump_id,
    const int32_t* heads, const int32_t* next,
    ContactPair* pairs, int32_t* count, int32_t capacity,
    int32_t num_local, ShearGrid grid, float skin) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local) return;
    const double xi = x[i], yi = y[i], zi = z[i];
    const float ri = support ? support[i] : radius[i];
    const int32_t ci = clump_id ? clump_id[i] : -2;
    for (int ix = -1; ix <= 1; ++ix) {
        const double qx = xi - ix * grid.length_x;
        if (qx < grid.min_x - grid.cell_size ||
            qx > grid.max_x + grid.cell_size) continue;
        for (int iy = -1; iy <= 1; ++iy) {
            const double qy = yi - iy * grid.length_y + ix * grid.reduced_shift;
            if (qy < grid.min_y - grid.cell_size ||
                qy > grid.max_y + grid.cell_size) continue;
            const int32_t cx = static_cast<int32_t>(floor((qx - grid.min_x) * grid.inverse_cell_size));
            const int32_t cy = static_cast<int32_t>(floor((qy - grid.min_y) * grid.inverse_cell_size));
            for (int32_t dy = -1; dy <= 1; ++dy) {
                const int32_t by = cy + dy;
                if (by < 0 || by >= grid.ny) continue;
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    const int32_t bx = cx + dx;
                    if (bx < 0 || bx >= grid.nx) continue;
                    for (int32_t j = heads[static_cast<int64_t>(by) * grid.nx + bx];
                         j != -1; j = next[j]) {
                        if (j <= i || (ci > 0 && clump_id[j] == ci)) continue;
                        // Match the periodic LBVH narrow-phase arithmetic and
                        // coincident-particle exclusion at the same epoch.
                        const float qfx = static_cast<float>(xi)
                            - static_cast<float>(ix * grid.length_x);
                        const float qfy = static_cast<float>(yi)
                            - static_cast<float>(iy * grid.length_y
                                - ix * grid.reduced_shift);
                        const float ddx = static_cast<float>(x[j]) - qfx;
                        const float ddy = static_cast<float>(y[j]) - qfy;
                        const float ddz = static_cast<float>(z[j])
                            - static_cast<float>(zi);
                        const float dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                        const float rj = support ? support[j] : radius[j];
                        const float limit = ri + rj + skin;
                        if (!(dist_sq < limit * limit && dist_sq > 1e-12f)) continue;
                        if (!(ri + rj - sqrtf(dist_sq) > -skin)) continue;
                        const int32_t slot = atomicAdd(count, 1);
                        if (slot >= 0 && slot < capacity)
                            pairs[slot] = ContactPair(i, j, ix, iy, 0);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Morton Code (Z-Order Curve) Functions
// =============================================================================
// These functions compute space-filling curve indices that improve cache
// locality by keeping spatially adjacent cells near each other in memory.
// This can improve memory access patterns by 20-30% in neighbor search.
// =============================================================================

// Expand 10-bit integer to 30 bits by inserting 2 zeros between each bit
__device__ __forceinline__ uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Compute 30-bit Morton code from 3D cell coordinates (each max 10 bits = 1024)
__device__ __forceinline__ uint32_t morton_3d(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

// Compute Morton-ordered cell ID from position
__device__ __forceinline__
uint32_t position_to_morton_cell_id(float px, float py, float pz) {
    // Compute 3D cell indices
    int32_t ix = static_cast<int32_t>((px - dev_neighbor_params.domain_min_x) *
                                       dev_neighbor_params.inv_cell_size);
    int32_t iy = static_cast<int32_t>((py - dev_neighbor_params.domain_min_y) *
                                       dev_neighbor_params.inv_cell_size);
    int32_t iz = static_cast<int32_t>((pz - dev_neighbor_params.domain_min_z) *
                                       dev_neighbor_params.inv_cell_size);

    // Clamp to valid range
    ix = max(0, min(ix, dev_neighbor_params.grid_dim_x - 1));
    iy = max(0, min(iy, dev_neighbor_params.grid_dim_y - 1));
    iz = max(0, min(iz, dev_neighbor_params.grid_dim_z - 1));

    // Return Morton code for Z-order memory layout
    return morton_3d(static_cast<uint32_t>(ix),
                     static_cast<uint32_t>(iy),
                     static_cast<uint32_t>(iz));
}

// =============================================================================
// Cell ID Computation (Linear indexing)
// =============================================================================

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

// =============================================================================
// Per-GPU State Map (Multi-GPU Safe)
// =============================================================================
// Each GPU maintains its own state to avoid cross-GPU corruption.
// Thread-safe access via mutex for OpenMP parallelism.
// =============================================================================

std::unordered_map<int32_t, CLLGPUState> CLLSearchImpl::gpu_states_;
std::mutex CLLSearchImpl::state_mutex_;
NeighborSearchStats CLLSearchImpl::default_stats_;

// Morton ordering enable flag (shared, read-only after init)
static bool use_morton_binning = true;

// Get or create state for current GPU (thread-safe)
CLLGPUState& CLLSearchImpl::get_gpu_state() {
    int32_t gpu_id;
    cudaGetDevice(&gpu_id);
    return get_gpu_state(gpu_id);
}

// Get or create state for specific GPU (thread-safe)
CLLGPUState& CLLSearchImpl::get_gpu_state(int32_t gpu_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = gpu_states_.find(gpu_id);
    if (it == gpu_states_.end()) {
        // Create new state for this GPU
        auto [inserted_it, success] = gpu_states_.emplace(gpu_id, CLLGPUState());
        inserted_it->second.gpu_id = gpu_id;
        return inserted_it->second;
    }
    return it->second;
}



// =============================================================================
// CUDA Kernels
// =============================================================================

// -----------------------------------------------------------------------------
// Compute Cell IDs Kernel
// -----------------------------------------------------------------------------
// Converts particle positions to cell linear indices

__global__ void compute_cell_ids_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    int32_t* __restrict__ cell_ids,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Load position
    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);

    // Compute cell ID
    cell_ids[idx] = position_to_cell_id(px, py, pz);
}

// -----------------------------------------------------------------------------
// Detect Cell Changes Kernel
// -----------------------------------------------------------------------------
// Compares current cell IDs with previous to detect if rebuild is needed

__global__ void detect_cell_changes_kernel(
    const int32_t* __restrict__ current_cell_ids,
    const int32_t* __restrict__ previous_cell_ids,
    int32_t* __restrict__ changed_flag,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // If any particle changed cells, set flag
    if (current_cell_ids[idx] != previous_cell_ids[idx]) {
        atomicOr(changed_flag, 1);
    }
}

// -----------------------------------------------------------------------------
// Check Displacement Kernel (Verlet Buffer)
// -----------------------------------------------------------------------------
// Computes max displacement since last rebuild to check against skin depth
// Uses block reduction to find local max, then atomicMax for global

__global__ void __launch_bounds__(256, 4) check_displacement_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ last_pos_x,
    const double* __restrict__ last_pos_y,
    const double* __restrict__ last_pos_z,
    int32_t* __restrict__ max_disp_sq_scaled, // Stores max_disp^2 * 1e6 as int for atomicMax
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float my_disp_sq = 0.0f;

    if (idx < num_particles) {
        float dx = static_cast<float>(pos_x[idx] - last_pos_x[idx]);
        float dy = static_cast<float>(pos_y[idx] - last_pos_y[idx]);
        float dz = static_cast<float>(pos_z[idx] - last_pos_z[idx]);
        my_disp_sq = dx*dx + dy*dy + dz*dz;
    }

#ifdef CUDADEM_CPU_MOCK
    // CPU Mock: Global Atomic Max
    int32_t int_val = __float_as_int(my_disp_sq);
    atomicMax(max_disp_sq_scaled, int_val);
#else
    // Block-level reduction
    typedef cub::BlockReduce<float, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    float block_max = BlockReduceT(temp_storage).Reduce(my_disp_sq, cub::Max());

    // Atomic update to global max (only first thread of block)
    if (threadIdx.x == 0) {
        // Re-interpreting float bits as int works for positive floats comparison
        int32_t int_val = __float_as_int(block_max);
        atomicMax(max_disp_sq_scaled, int_val);
    }
#endif
}

// -----------------------------------------------------------------------------
// Store Last Positions Kernel
// -----------------------------------------------------------------------------
// Updates reference positions after a rebuild

__global__ void store_last_positions_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ last_pos_x,
    double* __restrict__ last_pos_y,
    double* __restrict__ last_pos_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    last_pos_x[idx] = pos_x[idx];
    last_pos_y[idx] = pos_y[idx];
    last_pos_z[idx] = pos_z[idx];
}

// -----------------------------------------------------------------------------
// Build Cell Ranges Kernel
// -----------------------------------------------------------------------------
// Finds start/end indices for each cell from sorted particle array

__global__ void build_cell_ranges_kernel(
    const int32_t* __restrict__ sorted_cell_ids,
    int32_t* __restrict__ cell_start,
    int32_t* __restrict__ cell_end,
    int32_t num_particles,
    int32_t total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize cell_start to -1 (empty cell marker)
    if (idx < total_cells) {
        cell_start[idx] = -1;
        cell_end[idx] = -1;
    }

    __syncthreads();

    // Each thread handles one particle
    if (idx >= num_particles) return;

    int32_t cell_id = sorted_cell_ids[idx];

    // Check if this is the first particle in the cell
    if (idx == 0 || sorted_cell_ids[idx - 1] != cell_id) {
        cell_start[cell_id] = idx;
    }

    // Check if this is the last particle in the cell
    if (idx == num_particles - 1 || sorted_cell_ids[idx + 1] != cell_id) {
        cell_end[cell_id] = idx + 1;  // Exclusive end
    }
}

// -----------------------------------------------------------------------------
// Reorder Array Kernel (Template)
// -----------------------------------------------------------------------------
// Reorders any array based on sorted indices

template<typename T>
__global__ void reorder_array_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int32_t* __restrict__ indices,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    dst[idx] = src[indices[idx]];
}

// Explicit instantiations for common types
template __global__ void reorder_array_kernel<double>(
    const double*, double*, const int32_t*, int32_t);
template __global__ void reorder_array_kernel<float>(
    const float*, float*, const int32_t*, int32_t);
template __global__ void reorder_array_kernel<int32_t>(
    const int32_t*, int32_t*, const int32_t*, int32_t);

// =============================================================================
// Atomic Binning Kernels (O(N) Counting Sort with Morton Code)
// =============================================================================
// These kernels replace thrust::sort_by_key with O(N) atomic operations:
// 1. count_cells_kernel: Count particles per Morton cell
// 2. bin_particles_kernel: Place particles into sorted positions atomically
// This reduces rebuild time from O(N log N) to O(N), typically 5-10x faster.
// =============================================================================

// -----------------------------------------------------------------------------
// Count Cells Kernel - Phase 1 of Atomic Binning
// -----------------------------------------------------------------------------
// Counts how many particles belong to each cell using Morton-ordered cell IDs
__global__ void count_cells_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    int32_t* __restrict__ cell_counts,
    int32_t* __restrict__ particle_cell_ids,  // Output: store Morton cell ID per particle
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Compute Morton-ordered cell ID
    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);

    uint32_t morton_id = position_to_morton_cell_id(px, py, pz);

    // Store cell ID for this particle (used in binning phase)
    particle_cell_ids[idx] = static_cast<int32_t>(morton_id);

    // Atomically increment count for this cell
    atomicAdd(&cell_counts[morton_id], 1);
}

// -----------------------------------------------------------------------------
// Bin Particles Kernel - Phase 3 of Atomic Binning (after prefix sum)
// -----------------------------------------------------------------------------
// Places particle indices into their sorted positions using atomic increments
__global__ void bin_particles_kernel(
    const int32_t* __restrict__ particle_cell_ids,  // Morton cell ID per particle
    const int32_t* __restrict__ cell_start,         // Start position per cell (from prefix sum)
    int32_t* __restrict__ cell_offsets,             // Atomic write cursors (initialized = cell_start)
    int32_t* __restrict__ sorted_indices,           // Output: sorted particle indices
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    int32_t cell_id = particle_cell_ids[idx];

    // Atomically get write position and increment cursor
    int32_t write_pos = atomicAdd(&cell_offsets[cell_id], 1);

    // Write particle index to sorted position
    sorted_indices[write_pos] = idx;
}

// -----------------------------------------------------------------------------
// Build Cell End from Counts Kernel
// -----------------------------------------------------------------------------
// Computes cell_end = cell_start + cell_count for each cell
__global__ void build_cell_end_kernel(
    const int32_t* __restrict__ cell_start,
    const int32_t* __restrict__ cell_counts,
    int32_t* __restrict__ cell_end,
    int32_t total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;

    int32_t start = cell_start[idx];
    int32_t count = cell_counts[idx];

    // cell_end is exclusive: cell_end = cell_start + count
    // Use -1 for empty cells (where count is 0)
    cell_end[idx] = (count > 0) ? (start + count) : -1;
}

// -----------------------------------------------------------------------------
// Find Contacts Kernel (Broad + Narrow Phase Combined)
// -----------------------------------------------------------------------------
// For each particle, search 27 neighboring cells and detect contacts
// Optimized with __launch_bounds__ for high occupancy and warp-aggregated atomics

__global__ void __launch_bounds__(256, 4) find_contacts_kernel(
    // Positions (in current order, not necessarily sorted)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Radii
    const float* __restrict__ radius,
    const float* __restrict__ support_extent,
    // Cell structure
    const int32_t* __restrict__ cell_start,
    const int32_t* __restrict__ cell_end,
    const int32_t* __restrict__ particle_cell_ids,
    // Output
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts,
    // Clump membership (nullptr if clumps disabled)
    const int32_t* __restrict__ clump_id,
    // Params
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Load particle i data
    float pi_x = static_cast<float>(pos_x[idx]);
    float pi_y = static_cast<float>(pos_y[idx]);
    float pi_z = static_cast<float>(pos_z[idx]);
    float ri = support_extent[idx];

    // Hoist clump membership for particle i (constant for this thread)
    const int32_t ci_clump = (clump_id != nullptr) ? clump_id[idx] : -2;

    // Get cell indices for particle i
    int32_t ix, iy, iz;
    position_to_cell_3d(pi_x, pi_y, pi_z, ix, iy, iz);

    // Periodic boundary flags
    bool periodic_x = dev_neighbor_params.periodic_x;
    bool periodic_y = dev_neighbor_params.periodic_y;
    bool periodic_z = dev_neighbor_params.periodic_z;

    int32_t grid_dim_x = dev_neighbor_params.grid_dim_x;
    int32_t grid_dim_y = dev_neighbor_params.grid_dim_y;
    int32_t grid_dim_z = dev_neighbor_params.grid_dim_z;

    float cell_size = dev_neighbor_params.cell_size;
    float domain_size_x = dev_neighbor_params.domain_max_x - dev_neighbor_params.domain_min_x;
    float domain_size_y = dev_neighbor_params.domain_max_y - dev_neighbor_params.domain_min_y;
    float domain_size_z = dev_neighbor_params.domain_max_z - dev_neighbor_params.domain_min_z;

    // Search 27 neighboring cells (including self)
    // Contact accumulator for this thread
    // int32_t local_contact_count = 0;
    // We can't easily batch writes without an intermediate buffer, but we CAN batch the ATOMIC ADD.
    // However, since we need the correct index for EACH contact to write it,
    // we need to reserve a Range of indices.

    // 1. First pass: Search and buffer contacts locally (limiting register usage)
    // Practical limit: 32 max neighbors per step typically, or we can just iterate.
    // Better approach for large scale: Just optimize the atomic.

    // We will use a "Warp-Aggregated Atomic" approach.
    // But since the loop is complex, we'll do it per-contact or batched?
    // Batching per particle is hard because number of contacts is unknown and variable.

    // Simplest robust opt: Just check "potential contact" using skin depth.

    float verlet_skin = dev_neighbor_params.verlet_skin;
    // float effective_threshold = -(verlet_skin);
    // If overlap > -skin, it's a POTENTIAL contact we must track.
    // Note: dev_neighbor_params.contact_threshold is normally 0.
    // If verlet logic is enabled, we store ANY pair within skin depth.
    // If disabled (legacy), we store only overlap > 0.

    // Using a simple bool check from config would be slow in kernel,
    // so we assume dev_neighbor_params.contact_threshold has been set to -skin on host if verlet enabled.
    float threshold = dev_neighbor_params.contact_threshold;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                // Apply periodic boundary condition to neighbor cell index
                // neigh_x = (my_x + offset_x + N_x) % N_x
                int32_t nx = ix + dx;
                int32_t ny = iy + dy;
                int32_t nz = iz + dz;

                // Handle periodic boundaries (logic identical to original)
                float shift_x = 0.0f, shift_y = 0.0f, shift_z = 0.0f;

                if (periodic_x) {
                    if (nx < 0) { nx += grid_dim_x; shift_x = -domain_size_x; }
                    else if (nx >= grid_dim_x) { nx -= grid_dim_x; shift_x = domain_size_x; }
                } else { if (nx < 0 || nx >= grid_dim_x) continue; }

                if (periodic_y) {
                    if (ny < 0) { ny += grid_dim_y; shift_y = -domain_size_y; }
                    else if (ny >= grid_dim_y) { ny -= grid_dim_y; shift_y = domain_size_y; }
                } else { if (ny < 0 || ny >= grid_dim_y) continue; }

                if (periodic_z) {
                    if (nz < 0) { nz += grid_dim_z; shift_z = -domain_size_z; }
                    else if (nz >= grid_dim_z) { nz -= grid_dim_z; shift_z = domain_size_z; }
                } else { if (nz < 0 || nz >= grid_dim_z) continue; }

                // Get Morton-ordered cell ID (must match rebuild's Morton ordering)
                int32_t neighbor_cell = static_cast<int32_t>(
                    morton_3d(static_cast<uint32_t>(nx),
                              static_cast<uint32_t>(ny),
                              static_cast<uint32_t>(nz)));

                // Get particle range in this cell
                int32_t start = cell_start[neighbor_cell];
                int32_t end = cell_end[neighbor_cell];

                if (start < 0) continue;  // Empty cell

                // Check all particles in this cell
                for (int32_t k = start; k < end; ++k) {
                    // Indirection: Get original particle index from sorted slot
                    int32_t j = particle_cell_ids[k];

                    // Skip self and ensure i < j to avoid duplicates
                    if (j <= idx) continue;

                    // Skip intra-clump pairs (sub-spheres of the same clump)
                    if (ci_clump > 0) {
                        if (clump_id[j] == ci_clump) continue;
                    }

                    // Load particle j data
                    float pj_x = static_cast<float>(pos_x[j]) + shift_x;
                    float pj_y = static_cast<float>(pos_y[j]) + shift_y;
                    float pj_z = static_cast<float>(pos_z[j]) + shift_z;
                    float rj = support_extent[j];

                    // Compute distance vector
                    float dx_ij = pj_x - pi_x;
                    float dy_ij = pj_y - pi_y;
                    float dz_ij = pj_z - pi_z;

                    // Compute distance: d_ij = |r_i - r_j|
                    float dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
                    float contact_dist = ri + rj;

                    // Pre-check with skin!
                    // We need to keep pair if dist < contact_dist + skin
                    // effectively: overlap = (ri+rj) - dist > -skin
                    // dist < contact_dist + skin  => dist_sq < (contact_dist+skin)^2

                    // Optimization: check squared distance first
                    float range_limit = contact_dist - threshold; // threshold is -skin or 0
                    if (dist_sq >= range_limit * range_limit) continue;

                    // Preserve the final distance predicate, including its rounding.
                    float dist = sqrtf(dist_sq);

                    // Only store if within threshold (overlap > -skin)
                    if (dist < range_limit) {
                        // Standard global atomic ADD is a bottleneck.
                        // Ideally, we would use a warp-aggregated atomic here.
                        // But implementing that inside a triple loop structure is tricky for code size.
                        // The simple legacy way:
                        int32_t contact_idx = atomicAdd(num_contacts, 1);
                        if (contact_idx < max_contacts) {
                            // Compute image offsets from cell-wrap shifts
                            // shift = +domain → img = +1 (j's +X ghost was used)
                            // shift = -domain → img = -1 (j's -X ghost was used)
                            int8_t img_x = (shift_x > 0.0f) ? 1 : (shift_x < 0.0f ? -1 : 0);
                            int8_t img_y = (shift_y > 0.0f) ? 1 : (shift_y < 0.0f ? -1 : 0);
                            int8_t img_z = (shift_z > 0.0f) ? 1 : (shift_z < 0.0f ? -1 : 0);
                            contacts[contact_idx] = ContactPair(idx, j, img_x, img_y, img_z);
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Verlet Cache Kernels
// =============================================================================

// -----------------------------------------------------------------------------
// Cache Pairs Kernel - Extract (i, j, image) from ContactPair buffer
// -----------------------------------------------------------------------------
__global__ void cache_pairs_kernel(
    const ContactPair* __restrict__ contacts,
    int32_t num_contacts,
    int32_t* __restrict__ pair_i,
    int32_t* __restrict__ pair_j,
    int8_t* __restrict__ pair_image_x,
    int8_t* __restrict__ pair_image_y,
    int8_t* __restrict__ pair_image_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;
    
    pair_i[idx] = contacts[idx].i;
    pair_j[idx] = contacts[idx].j;
    pair_image_x[idx] = contacts[idx].image_x;
    pair_image_y[idx] = contacts[idx].image_y;
    pair_image_z[idx] = contacts[idx].image_z;
}

// -----------------------------------------------------------------------------
// Copy candidate identity from the Verlet cache. Force evaluation uses current
// positions; cache reuse must not load or recompute any particle geometry.
// -----------------------------------------------------------------------------
__global__ void copy_cached_contact_pairs_kernel(
    const int32_t* __restrict__ pair_i,
    const int32_t* __restrict__ pair_j,
    const int8_t* __restrict__ pair_image_x,
    const int8_t* __restrict__ pair_image_y,
    const int8_t* __restrict__ pair_image_z,
    int32_t num_pairs,
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    contacts[idx] = ContactPair(pair_i[idx], pair_j[idx],
        pair_image_x[idx], pair_image_y[idx], pair_image_z[idx]);
    if (idx == 0) *num_contacts = num_pairs;
}

// =============================================================================
// CLLSearchImpl Implementation
// =============================================================================

// Forward declarations for static helper functions
static void cache_neighbor_pairs(
    const ContactPair* __restrict__ contacts,
    int32_t num_contacts,
    int32_t* __restrict__ pair_i,
    int32_t* __restrict__ pair_j,
    int8_t* __restrict__ pair_image_x,
    int8_t* __restrict__ pair_image_y,
    int8_t* __restrict__ pair_image_z,
    cudaStream_t stream
);

static void copy_cached_pairs_to_contacts(
    const int32_t* __restrict__ pair_i,
    const int32_t* __restrict__ pair_j,
    const int8_t* __restrict__ pair_image_x,
    const int8_t* __restrict__ pair_image_y,
    const int8_t* __restrict__ pair_image_z,
    int32_t num_pairs,
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts,
    cudaStream_t stream
);

void CLLSearchImpl::initialize(
    const SimConfig& config,
    SimulationState& state
) {
    // Get per-GPU state (creates if not exists)
    CLLGPUState& gpu_state = get_gpu_state();
    int32_t gpu_id = gpu_state.gpu_id;

    if (gpu_state.initialized) {
        std::cerr << "WARNING: CLLSearchImpl already initialized on GPU " << gpu_id
                  << ", reinitializing..." << std::endl;
        shutdown(state, gpu_id);
    }

    if (config.shear_box.enabled) {
        // The wide, vertically thin shear box cannot use the 10-bit/axis
        // Morton grid below. Its bins are allocated from the current support
        // radius on each evaluation, after migration and ghost exchange.
        gpu_state.shear_max_support.initialize();
        gpu_state.initialized = true;
        gpu_state.grid_valid = false;
        gpu_state.stats = NeighborSearchStats();
        state.neighbor.active_method = NeighborMethod::CELL_LINKED_LIST;
        return;
    }

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) {
        std::cerr << "WARNING: No particles for neighbor search on GPU " << gpu_id << std::endl;
        gpu_state.initialized = true;
        return;
    }

    // DEM cohesion and SPH kernels may both reach beyond the particle radius.
    // Use the actual per-particle support field so the 27-cell stencil remains
    // conservative for either physics mode.
    float max_extent = config.particle.radius_max;
    if (state.particles.support_extent) {
        thrust::device_ptr<float> extent(state.particles.support_extent);
        max_extent = thrust::reduce(thrust::device, extent, extent + num_particles,
                                    0.0f, thrust::maximum<float>());
    }
    if (config.physics == SimulationPhysics::SPH) {
        max_extent = std::max(max_extent,
            config.sph.hsml_max * config.sph.kernel_cutoff_factor);
    }
    // Compute cell size from the conservative interaction extent.
    float cell_size = compute_cell_size(
        max_extent,
        config.neighbor.cell_size_factor
    );

    // ==========================================================================
    // Compute domain bounds from particles if needed
    // ==========================================================================
    float domain_min_x, domain_min_y, domain_min_z;
    float domain_max_x, domain_max_y, domain_max_z;
    
    // Get config bounds
    float config_min_x = config.domain.box_min.x;
    float config_min_y = config.domain.box_min.y;
    float config_min_z = config.domain.box_min.z;
    float config_max_x = config.domain.box_max.x;
    float config_max_y = config.domain.box_max.y;
    float config_max_z = config.domain.box_max.z;
    
    // Check if bounds need to be computed from particles:
    // 1. auto_bounds flag is set, OR
    // 2. Bounds are inverted (min >= max), OR
    // 3. Domain is degenerate (grid would be 1x1x1 or very small)
    float config_size_x = config_max_x - config_min_x;
    float config_size_y = config_max_y - config_min_y;
    float config_size_z = config_max_z - config_min_z;
    bool bounds_inverted = (config_min_x >= config_max_x || config_min_y >= config_max_y || config_min_z >= config_max_z);
    bool degenerate_grid = (config_size_x < cell_size || config_size_y < cell_size || config_size_z < cell_size);
    bool compute_from_particles = config.domain.auto_bounds || bounds_inverted || degenerate_grid;
    
    if (compute_from_particles) {
        // Compute bounding box from particle positions on GPU
        // Use CUB reduction for min/max
        // CRITICAL: CUB output pointers must be device memory, not host arrays
        double* d_bbox_min = nullptr;
        double* d_bbox_max = nullptr;
        cudaMalloc(&d_bbox_min, 3 * sizeof(double));
        cudaMalloc(&d_bbox_max, 3 * sizeof(double));
        
        // Allocate temporary storage for reduction
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        // Get temp storage size for min reduction
        cub::DeviceReduce::Min(nullptr, temp_storage_bytes,
                               state.particles.pos_x, d_bbox_min, num_particles);
        
        // Allocate temp storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        
        // Compute min for each coordinate
        cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_x, d_bbox_min, num_particles);
        cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_y, d_bbox_min + 1, num_particles);
        cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_z, d_bbox_min + 2, num_particles);
        
        // Compute max for each coordinate
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_x, d_bbox_max, num_particles);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_y, d_bbox_max + 1, num_particles);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                               state.particles.pos_z, d_bbox_max + 2, num_particles);
        
        // Copy results to host
        double h_bbox_min[3], h_bbox_max[3];
        cudaMemcpy(h_bbox_min, d_bbox_min, 3 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bbox_max, d_bbox_max, 3 * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Free temp storage and output arrays
        cudaFree(d_temp_storage);
        cudaFree(d_bbox_min);
        cudaFree(d_bbox_max);
        
        // Add padding based on max particle radius
        float padding = max_extent * 2.0f;
        domain_min_x = static_cast<float>(h_bbox_min[0] - padding);
        domain_min_y = static_cast<float>(h_bbox_min[1] - padding);
        domain_min_z = static_cast<float>(h_bbox_min[2] - padding);
        domain_max_x = static_cast<float>(h_bbox_max[0] + padding);
        domain_max_y = static_cast<float>(h_bbox_max[1] + padding);
        domain_max_z = static_cast<float>(h_bbox_max[2] + padding);
        
        std::cout << "  Domain bounds computed from particles:"
                  << " [" << domain_min_x << ", " << domain_min_y << ", " << domain_min_z << "]"
                  << " to [" << domain_max_x << ", " << domain_max_y << ", " << domain_max_z << "]"
                  << std::endl;
    } else {
        // Use config values
        domain_min_x = config.domain.box_min.x;
        domain_min_y = config.domain.box_min.y;
        domain_min_z = config.domain.box_min.z;
        domain_max_x = config.domain.box_max.x;
        domain_max_y = config.domain.box_max.y;
        domain_max_z = config.domain.box_max.z;
    }

    // Compute grid dimensions
    int32_t grid_dim_x = static_cast<int32_t>(
        ceilf((domain_max_x - domain_min_x) / cell_size));
    int32_t grid_dim_y = static_cast<int32_t>(
        ceilf((domain_max_y - domain_min_y) / cell_size));
    int32_t grid_dim_z = static_cast<int32_t>(
        ceilf((domain_max_z - domain_min_z) / cell_size));

    // Ensure at least 1 cell per dimension
    grid_dim_x = std::max(1, grid_dim_x);
    grid_dim_y = std::max(1, grid_dim_y);
    grid_dim_z = std::max(1, grid_dim_z);

    int32_t total_cells = grid_dim_x * grid_dim_y * grid_dim_z;
    // Morton codes for non-power-of-2 grids can exceed total_cells.
    // Compute the true upper bound for cell array indexing.
    int32_t morton_capacity = compute_morton_capacity(grid_dim_x, grid_dim_y, grid_dim_z);

    // Get periodic boundary configuration
    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    // Prepare GPU parameters
    NeighborParamsGPU params;
    params.grid_dim_x = grid_dim_x;
    params.grid_dim_y = grid_dim_y;
    params.grid_dim_z = grid_dim_z;
    params.total_cells = total_cells;
    params.morton_capacity = morton_capacity;
    params.cell_size = cell_size;
    params.inv_cell_size = 1.0f / cell_size;
    params.domain_min_x = domain_min_x;
    params.domain_min_y = domain_min_y;
    params.domain_min_z = domain_min_z;
    params.domain_max_x = domain_max_x;
    params.domain_max_y = domain_max_y;
    params.domain_max_z = domain_max_z;
    params.periodic_x = pbc.periodic_x;
    params.periodic_y = pbc.periodic_y;
    params.periodic_z = pbc.periodic_z;
    // NOTE: Contact threshold logic
    // If Verlet buffer is enabled, we need to store POTENTIAL contacts within the skin depth.
    // Overlap = (ri+rj) - dist.
    // Potential contact: dist < ri+rj + skin  => overlap > -skin
    // So threshold = -skin
    if (config.neighbor.verlet_buffer_enabled) {
        params.contact_threshold = -config.neighbor.verlet_skin;
        params.verlet_skin = config.neighbor.verlet_skin;
    } else {
        params.contact_threshold = 0.0f;
        params.verlet_skin = 0.0f;
    }
    params.kernel_cutoff_factor = config.sph.kernel_cutoff_factor;

    // Copy to constant memory
    cudaMemcpyToSymbol(dev_neighbor_params, &params, sizeof(NeighborParamsGPU));

    // Allocate previous cell ID buffer for change detection
    cudaMalloc(&gpu_state.prev_cell_ids, num_particles * sizeof(int32_t));
    cudaMemset(gpu_state.prev_cell_ids, 0xFF, num_particles * sizeof(int32_t));  // Initialize to -1

    // Allocate temporary buffers for sorting
    cudaMalloc(&gpu_state.temp_keys, num_particles * sizeof(int32_t));
    cudaMalloc(&gpu_state.temp_values, num_particles * sizeof(int32_t));

    // Allocate reorder buffers
    cudaMalloc(&gpu_state.temp_pos_x, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_pos_y, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_pos_z, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_vel_x, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_vel_y, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_vel_z, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_radius, num_particles * sizeof(float));
    cudaMalloc(&gpu_state.temp_mass, num_particles * sizeof(float));

    // Additional reorder buffers for complete particle data synchronization
    cudaMalloc(&gpu_state.temp_omega_x, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_omega_y, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_omega_z, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_moment_of_inertia, num_particles * sizeof(float));
    cudaMalloc(&gpu_state.temp_torque_x, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_torque_y, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_torque_z, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_quat_w, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_quat_x, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_quat_y, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_quat_z, num_particles * sizeof(double));
    cudaMalloc(&gpu_state.temp_cohesive_strength, num_particles * sizeof(float));
    cudaMalloc(&gpu_state.temp_material_id, num_particles * sizeof(int32_t));
    cudaMalloc(&gpu_state.temp_contact_history_offset, num_particles * sizeof(int32_t));
    cudaMalloc(&gpu_state.temp_num_contacts, num_particles * sizeof(int32_t));

    // Allocate change detection flag
    cudaMalloc(&gpu_state.d_changed, sizeof(int32_t));

    // ==========================================================================
    // Atomic Binning Buffer Initialization (O(N) Counting Sort)
    // ==========================================================================
    // Allocate buffers for the counting sort pipeline:
    // 1. cell_counts: particles per cell
    // 2. cell_offsets: atomic write cursors during binning
    // 3. CUB temp storage: for prefix sum
    // ==========================================================================
    // Allocate with morton_capacity (not total_cells): Morton codes can index beyond total_cells
    cudaMalloc(&gpu_state.d_cell_counts, morton_capacity * sizeof(int32_t));
    cudaMalloc(&gpu_state.d_cell_offsets, morton_capacity * sizeof(int32_t));

    // ==========================================================================
    // Allocate state.neighbor arrays (used by kernels and sorting pipeline)
    // ==========================================================================
    cudaMalloc(&state.neighbor.cell_start, morton_capacity * sizeof(int32_t));
    cudaMalloc(&state.neighbor.cell_end, morton_capacity * sizeof(int32_t));
    cudaMalloc(&state.neighbor.particle_indices, num_particles * sizeof(int32_t));
    cudaMalloc(&state.neighbor.cell_ids, num_particles * sizeof(int32_t));

    // Store neighbor data parameters in state
    state.neighbor.num_particles = num_particles;
    state.neighbor.cell_size = cell_size;
    state.neighbor.grid_dim_x = grid_dim_x;
    state.neighbor.grid_dim_y = grid_dim_y;
    state.neighbor.grid_dim_z = grid_dim_z;
    state.neighbor.total_cells = total_cells;
    state.neighbor.morton_capacity = morton_capacity;
    state.neighbor.domain_min_x = domain_min_x;
    state.neighbor.domain_min_y = domain_min_y;
    state.neighbor.domain_min_z = domain_min_z;
    state.neighbor.domain_max_x = domain_max_x;
    state.neighbor.domain_max_y = domain_max_y;
    state.neighbor.domain_max_z = domain_max_z;

    // Query CUB for required temp storage size (first call with nullptr)
    gpu_state.cub_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, gpu_state.cub_temp_storage_bytes,
                                   gpu_state.d_cell_counts, state.neighbor.cell_start,
                                   morton_capacity);
    // Allocate CUB temp storage
    cudaMalloc(&gpu_state.d_cub_temp_storage, gpu_state.cub_temp_storage_bytes);

    std::cout << "  Atomic binning: Enabled (O(N) rebuild)" << std::endl;
    std::cout << "  Morton Z-order: Enabled (cache optimization)" << std::endl;

    // ==========================================================================
    // Async Lagged Readback Initialization
    // ==========================================================================
    // Allocate pinned memory for async D2H copy of max displacement
    cudaMallocHost(&gpu_state.h_pinned_max_disp_sq, sizeof(float));
    *gpu_state.h_pinned_max_disp_sq = 0.0f;

    // Create event for checking async copy completion
    cudaEventCreateWithFlags(&gpu_state.disp_copy_done_event, cudaEventDisableTiming);

    // Initialize state
    gpu_state.cached_max_disp_sq = 0.0f;
    gpu_state.async_readback_pending = false;
    gpu_state.steps_since_rebuild = 0;

    // Reset statistics
    gpu_state.stats = NeighborSearchStats();

    // ==========================================================================
    // Verlet Neighbor Cache Initialization
    // ==========================================================================
    int32_t max_pairs = num_particles * 12;  // Conservative: ~12 neighbors per particle
    cudaMalloc(&gpu_state.verlet_cache.pair_i, max_pairs * sizeof(int32_t));
    cudaMalloc(&gpu_state.verlet_cache.pair_j, max_pairs * sizeof(int32_t));
    cudaMalloc(&gpu_state.verlet_cache.pair_image_x, max_pairs * sizeof(int8_t));
    cudaMalloc(&gpu_state.verlet_cache.pair_image_y, max_pairs * sizeof(int8_t));
    cudaMalloc(&gpu_state.verlet_cache.pair_image_z, max_pairs * sizeof(int8_t));
    gpu_state.verlet_cache.max_pairs = max_pairs;
    gpu_state.verlet_cache.cache_allocated = true;
    gpu_state.verlet_cache.cache_valid = false;
    
    std::cout << "  Verlet cache: " << max_pairs << " pairs capacity" << std::endl;

    gpu_state.initialized = true;
    gpu_state.grid_valid = false;

    // Print configuration
    std::cout << "CLLSearchImpl initialized on GPU " << gpu_id << ":" << std::endl;
    std::cout << "  Cell size: " << cell_size << " m" << std::endl;
    std::cout << "  Grid dimensions: " << grid_dim_x << " x " << grid_dim_y
              << " x " << grid_dim_z << std::endl;
    std::cout << "  Total cells: " << total_cells << std::endl;
    std::cout << "  Particles: " << num_particles << std::endl;
    std::cout << "  Periodic: (" << (pbc.periodic_x ? "X" : "-") << ", "
              << (pbc.periodic_y ? "Y" : "-") << ", "
              << (pbc.periodic_z ? "Z" : "-") << ")" << std::endl;
}

bool CLLSearchImpl::needs_rebuild(
    const SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream
) {
    CLLGPUState& gpu_state = get_gpu_state();

    if (config.shear_box.enabled) return true;

    if (!gpu_state.grid_valid) return true;

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return false;

    // Increment step counter
    gpu_state.steps_since_rebuild++;

    if (config.neighbor.verlet_buffer_enabled) {
        // ==========================================================================
        // ASYNC LAGGED READBACK STRATEGY (Zero-Latency)
        // ==========================================================================
        // We use PREVIOUS step's max displacement to decide THIS step's rebuild.
        // This eliminates CPU-GPU synchronization completely.
        //
        // Decision Logic:
        // 1. Check if the async copy from previous step is done
        // 2. If done, read the value and update cached_max_disp_sq
        // 3. Use cached value to decide if rebuild is needed
        // 4. Safety: Force rebuild if too many steps have passed
        // ==========================================================================

        bool need_rebuild = false;

        // Check if we have a pending async copy that completed
        if (gpu_state.async_readback_pending) {
            if (cudaEventQuery(gpu_state.disp_copy_done_event) == cudaSuccess) {
                // Data is ready - read it
                gpu_state.cached_max_disp_sq = *gpu_state.h_pinned_max_disp_sq;
                gpu_state.async_readback_pending = false;
            }
            // If not ready, we'll use the old cached value (slightly stale but safe)
        }

        // Decision based on cached (lagged) displacement
        float max_disp = sqrtf(gpu_state.cached_max_disp_sq);
        float skin = config.neighbor.verlet_skin;

        // Criterion: 2 * max_disp > skin (worst case: two particles approaching)
        if (2.0f * max_disp > skin) {
            need_rebuild = true;
        }

        // Safety fallback: force rebuild every N steps to prevent tunneling
        // Uses update_frequency from config (default: 1, user set: 10)
        if (gpu_state.steps_since_rebuild >= config.neighbor.update_frequency) {
            need_rebuild = true;
        }

        return need_rebuild;

    } else {
        // --- Cell Change Strategy (Legacy) ---
        // This path still uses sync, but is only for non-Verlet mode
        
        // Force rebuild if update_frequency steps have passed
        if (gpu_state.steps_since_rebuild >= config.neighbor.update_frequency) {
            return true;
        }
        
        cudaMemsetAsync(gpu_state.d_changed, 0, sizeof(int32_t), stream);

        int block_size = 256;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        LAUNCH_KERNEL(detect_cell_changes_kernel, num_blocks, block_size, 0, stream,
            state.neighbor.cell_ids,
            gpu_state.prev_cell_ids,
            gpu_state.d_changed,
            num_particles
        );

        int32_t h_changed = 0;
        cudaMemcpyAsync(&h_changed, gpu_state.d_changed, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        return (h_changed != 0);
    }
}

void CLLSearchImpl::find_contacts(
    SimulationState& state,
    const SimConfig& config,
    ContactListBuffer& contacts,
    double evaluation_time,
    cudaStream_t stream
) {
    CLLGPUState& gpu_state = get_gpu_state();

    if (!gpu_state.initialized) {
        std::cerr << "ERROR: CLLSearchImpl not initialized on GPU " << gpu_state.gpu_id << std::endl;
        return;
    }

    if (config.shear_box.enabled) {
        const int32_t n = state.particles.num_particles;
        const int32_t num_local = state.particles.num_local;
        if (n < 0 || num_local < 0 || num_local > n)
            throw std::runtime_error("invalid shear CLL local/ghost particle counts");
        if (!contacts.pairs || !contacts.num_contacts || contacts.max_contacts <= 0)
            throw std::runtime_error("shear CLL contact buffer is unallocated");
        if (n == 0 || num_local == 0) {
            NeighborSearch::reset_contact_buffer(contacts, stream);
            gpu_state.stats.num_contacts_found = 0;
            return;
        }

        float max_support = 0.0f;
        {
            ShearProfileRange range("CLL Shear: Max Support");
            gpu_state.shear_max_support.enqueue(state.particles.support_extent,
                state.particles.radius, n, stream);
            max_support = gpu_state.shear_max_support.wait(n, stream);
        }
        const double skin = config.neighbor.verlet_buffer_enabled
            ? static_cast<double>(config.neighbor.verlet_skin) : 0.0;
        const double search_range = 2.0 * static_cast<double>(max_support) + skin;
        const double min_x = config.domain.box_min.x;
        const double min_y = config.domain.box_min.y;
        const double max_x = config.domain.box_max.x;
        const double max_y = config.domain.box_max.y;
        const double lx = max_x - min_x, ly = max_y - min_y;
        if (!std::isfinite(search_range) || search_range <= 0.0 ||
            !(search_range < 0.5 * lx) || !(search_range < 0.5 * ly))
            throw std::runtime_error("shear CLL unique-image condition violated: 2*max_support+skin must be positive and below half Lx and Ly");
        const ShearPhase phase = compute_shear_phase_checked(
            config.shear_box, lx, ly, evaluation_time);
        // The extra rounding margin keeps the 3x3 cell stencil conservative.
        const double cell_size = std::nextafter(search_range * 1.001,
            std::numeric_limits<double>::infinity());
        const double cells_x = std::ceil(lx / cell_size);
        const double cells_y = std::ceil(ly / cell_size);
        if (!(cells_x >= 1.0) || !(cells_y >= 1.0) ||
            cells_x * cells_y > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("shear CLL XY bin grid exceeds int32 capacity");
        const int32_t nx = static_cast<int32_t>(cells_x);
        const int32_t ny = static_cast<int32_t>(cells_y);
        const int32_t num_bins = static_cast<int32_t>(cells_x * cells_y);
        const ShearGrid grid{min_x, min_y, max_x, max_y, cell_size,
            1.0 / cell_size, lx, ly, phase.reduced_shift, search_range, nx, ny};

        {
            ShearProfileRange range("CLL Shear: Resize");
            if (num_bins > gpu_state.shear_heads_capacity) {
                const int64_t grown = std::max<int64_t>(num_bins,
                    2LL * gpu_state.shear_heads_capacity);
                const int32_t target = static_cast<int32_t>(std::min<int64_t>(
                    grown, std::numeric_limits<int32_t>::max()));
                int32_t* larger = nullptr;
                shear_cuda_check(cudaMalloc(&larger, static_cast<size_t>(target) * sizeof(int32_t)));
                if (gpu_state.shear_heads) shear_cuda_check(cudaFree(gpu_state.shear_heads));
                gpu_state.shear_heads = larger;
                gpu_state.shear_heads_capacity = target;
            }
            if (n > gpu_state.shear_particle_capacity) {
                const int64_t grown = std::max<int64_t>(n,
                    std::max<int64_t>(1024,
                        2LL * gpu_state.shear_particle_capacity));
                const int32_t target = static_cast<int32_t>(std::min<int64_t>(
                    grown, std::numeric_limits<int32_t>::max()));
                int32_t* larger = nullptr;
                shear_cuda_check(cudaMalloc(&larger, static_cast<size_t>(target) * sizeof(int32_t)));
                if (gpu_state.shear_next) shear_cuda_check(cudaFree(gpu_state.shear_next));
                gpu_state.shear_next = larger;
                gpu_state.shear_particle_capacity = target;
            }
        }
        {
            ShearProfileRange range("CLL Shear: Clear Bins");
            shear_cuda_check(cudaMemsetAsync(gpu_state.shear_heads, 0xff,
                static_cast<size_t>(num_bins) * sizeof(int32_t), stream));
        }
        const int threads = 256;
        {
            ShearProfileRange range("CLL Shear: Bin");
            shear_bin_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y,
                gpu_state.shear_heads, gpu_state.shear_next, n, grid);
            shear_cuda_check(cudaGetLastError());
        }
        for (;;) {
            NeighborSearch::reset_contact_buffer(contacts, stream);
            {
                ShearProfileRange range("CLL Shear: Query");
                shear_query_kernel<<<(num_local + threads - 1) / threads, threads, 0, stream>>>(
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.support_extent, state.particles.radius,
                    state.particles.clump_id, gpu_state.shear_heads, gpu_state.shear_next,
                    contacts.pairs, contacts.num_contacts, contacts.max_contacts,
                    // Every shear evaluation rebuilds, so history must reflect
                    // current material support without a future-motion shell.
                    num_local, grid, 0.0f);
                shear_cuda_check(cudaGetLastError());
            }
            int32_t found;
            {
                ShearProfileRange range("CLL Shear: Count");
                found = NeighborSearch::get_contact_count(contacts, stream);
            }
            if (found < 0)
                throw std::overflow_error("shear CLL contact count overflowed int32");
            if (found <= contacts.max_contacts) {
                gpu_state.stats.num_contacts_found = found;
                gpu_state.stats.num_rebuilds++;
                gpu_state.stats.cache_misses++;
                state.neighbor.num_particles = n;
                return;
            }
            const int64_t grown = std::max<int64_t>(found,
                2LL * contacts.max_contacts);
            if (grown > std::numeric_limits<int32_t>::max())
                throw std::overflow_error("shear CLL contact buffer exceeds int32 capacity");
            {
                ShearProfileRange range("CLL Shear: Overflow Resize");
                NeighborSearch::free_contact_buffer(contacts);
                NeighborSearch::allocate_contact_buffer(contacts,
                    static_cast<int32_t>(grown));
            }
        }
    }

    // Compute current cell IDs
    compute_cell_ids(state, stream);

    // Check if rebuild is needed
    if (needs_rebuild(state, config, stream)) {
        // Rebuild was needed - perform full contact search and cache pairs
        rebuild(state, config, stream);
        
        // After rebuild, do full contact search and cache
        NeighborSearch::reset_contact_buffer(contacts, stream);
        
        int32_t num_local = state.particles.num_local;
        int block_size = config.gpu.block_size;
        int num_blocks = (num_local + block_size - 1) / block_size;

        LAUNCH_KERNEL(find_contacts_kernel, num_blocks, block_size, 0, stream,
            state.particles.pos_x,
            state.particles.pos_y,
            state.particles.pos_z,
            state.particles.radius,
            state.particles.support_extent,
            state.neighbor.cell_start,
            state.neighbor.cell_end,
            state.neighbor.particle_indices,
            contacts.pairs,
            contacts.num_contacts,
            contacts.max_contacts,
            state.particles.clump_id,
            num_local
        );

        // Cache the pairs for future use
        int32_t num_contacts_found = NeighborSearch::get_contact_count(contacts, stream);
        
        // Always increment cache_misses for rebuild path (even if 0 contacts)
        gpu_state.stats.cache_misses++;
        
        if (num_contacts_found > 0 && num_contacts_found <= gpu_state.verlet_cache.max_pairs) {
            cache_neighbor_pairs(
                contacts.pairs,
                num_contacts_found,
                gpu_state.verlet_cache.pair_i,
                gpu_state.verlet_cache.pair_j,
                gpu_state.verlet_cache.pair_image_x,
                gpu_state.verlet_cache.pair_image_y,
                gpu_state.verlet_cache.pair_image_z,
                stream
            );
            gpu_state.verlet_cache.h_num_pairs = num_contacts_found;
            gpu_state.verlet_cache.cache_valid = true;
        } else if (num_contacts_found > gpu_state.verlet_cache.max_pairs) {
            // Too many contacts - invalidate cache
            gpu_state.verlet_cache.cache_valid = false;
            gpu_state.stats.cache_invalidations++;
        } else {
            // 0 contacts - cache remains valid but empty
            gpu_state.verlet_cache.h_num_pairs = 0;
            gpu_state.verlet_cache.cache_valid = true;
        }
        
        gpu_state.stats.num_contacts_found = num_contacts_found;
        return;
    }

    // No rebuild needed - check if we can use cached pairs (fast path)
    if (config.neighbor.verlet_buffer_enabled && 
        gpu_state.verlet_cache.cache_allocated && 
        gpu_state.verlet_cache.cache_valid) {
        // Fast path: copy only cached candidate identities
        copy_cached_pairs_to_contacts(
            gpu_state.verlet_cache.pair_i,
            gpu_state.verlet_cache.pair_j,
            gpu_state.verlet_cache.pair_image_x,
            gpu_state.verlet_cache.pair_image_y,
            gpu_state.verlet_cache.pair_image_z,
            gpu_state.verlet_cache.h_num_pairs,
            contacts.pairs,
            contacts.num_contacts,
            stream
        );
        
        gpu_state.stats.cache_hits++;
        gpu_state.stats.num_contacts_found = gpu_state.verlet_cache.h_num_pairs;
        return;
    }

    // Fallback: full contact search (shouldn't reach here in normal operation)
    NeighborSearch::reset_contact_buffer(contacts, stream);

    int32_t num_local = state.particles.num_local;
    int block_size = config.gpu.block_size;
    int num_blocks = (num_local + block_size - 1) / block_size;

    LAUNCH_KERNEL(find_contacts_kernel, num_blocks, block_size, 0, stream,
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        state.particles.radius,
        state.particles.support_extent,
        state.neighbor.cell_start,
        state.neighbor.cell_end,
        state.neighbor.particle_indices,
        contacts.pairs,
        contacts.num_contacts,
        contacts.max_contacts,
        state.particles.clump_id,
        num_local
    );

    gpu_state.stats.num_contacts_found = NeighborSearch::get_contact_count(contacts, stream);
}

// =============================================================================
// Verlet Cache Helper Functions
// =============================================================================

// Cache neighbor pairs after a full contact search
static void cache_neighbor_pairs(
    const ContactPair* __restrict__ contacts,
    int32_t num_contacts,
    int32_t* __restrict__ pair_i,
    int32_t* __restrict__ pair_j,
    int8_t* __restrict__ pair_image_x,
    int8_t* __restrict__ pair_image_y,
    int8_t* __restrict__ pair_image_z,
    cudaStream_t stream
) {
    if (num_contacts == 0) return;
    
    int block_size = 256;
    int num_blocks = (num_contacts + block_size - 1) / block_size;
    
    LAUNCH_KERNEL(cache_pairs_kernel, num_blocks, block_size, 0, stream,
        contacts,
        num_contacts,
        pair_i,
        pair_j,
        pair_image_x,
        pair_image_y,
        pair_image_z
    );
}

// Copy cached pairs to contact buffer (for fast path)
static void copy_cached_pairs_to_contacts(
    const int32_t* __restrict__ pair_i,
    const int32_t* __restrict__ pair_j,
    const int8_t* __restrict__ pair_image_x,
    const int8_t* __restrict__ pair_image_y,
    const int8_t* __restrict__ pair_image_z,
    int32_t num_pairs,
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts,
    cudaStream_t stream
) {
    if (num_pairs == 0) {
        // Reset count
        cudaMemsetAsync(num_contacts, 0, sizeof(int32_t), stream);
        return;
    }
    
    int block_size = 256;
    int num_blocks = (num_pairs + block_size - 1) / block_size;
    
    LAUNCH_KERNEL(copy_cached_contact_pairs_kernel, num_blocks, block_size, 0, stream,
        pair_i, pair_j, pair_image_x, pair_image_y, pair_image_z,
        num_pairs, contacts, num_contacts
    );
}

void CLLSearchImpl::compute_cell_ids(
    SimulationState& state,
    cudaStream_t stream
) {
    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(compute_cell_ids_kernel, num_blocks, block_size, 0, stream,
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        state.neighbor.cell_ids,
        num_particles
    );
}

void CLLSearchImpl::rebuild(
    SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream
) {
    // Shear bins depend on the current phase and are rebuilt by find_contacts.
    if (config.shear_box.enabled) return;
    CLLGPUState& gpu_state = get_gpu_state();

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    // CUDA events for timing
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    */

    // Step 1: Sort particles by cell ID AND Build cell ranges (Atomic Pipeline)
    // The atomic binning pipeline in sort_particles_by_cell now computes
    // cell_start (via Scan) and cell_end (via build_cell_end_kernel) directly.
    sort_particles_by_cell(state, stream);

    // Step 2: Build cell start/end ranges - SKIPPED (Handled in Step 1)
    // build_cell_ranges(state, stream);

    // Step 3: Reorder particle data
    // MULTI-GPU CHANGE: Disable reordering to preserve Local/Ghost layout
    // We now use indirection in find_contacts_kernel
    /*
    reorder_particle_data(state, stream);
    */

    // Step 4: Store current cell IDs as previous for next iteration
    // Use the UNSORTED cell_ids which match the position array order
    cudaMemcpyAsync(gpu_state.prev_cell_ids, state.neighbor.cell_ids,
                    num_particles * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Step 5: Store current positions as "last positions" (reference for Verlet buffer)
    if (config.neighbor.verlet_buffer_enabled) {
        int block_size = 256;
        int num_blocks = (num_particles + block_size - 1) / block_size;
        LAUNCH_KERNEL(store_last_positions_kernel, num_blocks, block_size, 0, stream,
            state.particles.pos_x,
            state.particles.pos_y,
            state.particles.pos_z,
            state.particles.last_pos_x,
            state.particles.last_pos_y,
            state.particles.last_pos_z,
            num_particles
        );

        // Reset cached displacement after rebuild (positions are now reference)
        gpu_state.cached_max_disp_sq = 0.0f;
        *gpu_state.h_pinned_max_disp_sq = 0.0f;
        gpu_state.async_readback_pending = false;
    }

    // Reset step counter after rebuild
    gpu_state.steps_since_rebuild = 0;

    // Update statistics
    // PERFORMANCE OPTIMIZATION: Removed blocking cudaEventSynchronize
    // We don't want to stall the CPU just to measure rebuild time per step.
    /*
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_state.stats.rebuild_time_ms, start, stop);
    */
    gpu_state.stats.num_rebuilds++;

    /*
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */

    gpu_state.grid_valid = true;
}

// =============================================================================
// Async Displacement Monitoring
// =============================================================================
// This function launches the displacement computation and async copy.
// It should be called AFTER force calculation each step to prepare data
// for the NEXT step's rebuild decision.
// =============================================================================

void CLLSearchImpl::launch_async_displacement_monitor(
    const SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream
) {
    if (config.shear_box.enabled) return;
    if (!config.neighbor.verlet_buffer_enabled) return;

    CLLGPUState& gpu_state = get_gpu_state();
    if (!gpu_state.initialized) return;

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    // Don't launch if previous copy hasn't been consumed yet
    // (This shouldn't happen in normal operation, but safety check)
    if (gpu_state.async_readback_pending) {
        // Check if previous copy completed
        if (cudaEventQuery(gpu_state.disp_copy_done_event) == cudaSuccess) {
            gpu_state.cached_max_disp_sq = *gpu_state.h_pinned_max_disp_sq;
            gpu_state.async_readback_pending = false;
        } else {
            // Previous copy still in flight - skip this update
            // The stale cached value will be used (safe, just slightly conservative)
            return;
        }
    }

    // 1. Reset reduction buffer
    cudaMemsetAsync(gpu_state.d_changed, 0, sizeof(int32_t), stream);

    // 2. Launch displacement reduction kernel
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    LAUNCH_KERNEL(check_displacement_kernel, num_blocks, block_size, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.last_pos_x, state.particles.last_pos_y, state.particles.last_pos_z,
        gpu_state.d_changed,
        num_particles
    );

    // 3. Async copy result to pinned memory (NO SYNC!)
    // We reinterpret the int32 as float bits for the copy
    cudaMemcpyAsync(gpu_state.h_pinned_max_disp_sq, gpu_state.d_changed,
                    sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 4. Record event so we can check completion next step
    cudaEventRecord(gpu_state.disp_copy_done_event, stream);
    gpu_state.async_readback_pending = true;
}

void CLLSearchImpl::sort_particles_by_cell(
    SimulationState& state,
    cudaStream_t stream
) {
    CLLGPUState& gpu_state = get_gpu_state();

    int32_t num_particles = state.particles.num_particles;
    int32_t total_cells = state.neighbor.total_cells;
    int32_t morton_capacity = state.neighbor.morton_capacity;

    // ==========================================================================
    // O(N) Atomic Binning Pipeline with Morton Code (Z-Order)
    // ==========================================================================
    // This replaces O(N log N) thrust::sort with O(N) counting sort:
    // 1. Count: Count particles per Morton-ordered cell
    // 2. Scan: Prefix sum to get cell_start positions
    // 3. Bin: Atomically place particles into sorted positions
    // NOTE: All cell-indexed arrays use morton_capacity, NOT total_cells, because
    // Morton codes for non-power-of-2 grids can exceed total_cells.
    // ==========================================================================

    int block_size = 256;
    int num_particle_blocks = (num_particles + block_size - 1) / block_size;
    int num_cell_blocks = (morton_capacity + block_size - 1) / block_size;

    // Step 1: Clear cell counts (must cover full morton_capacity range)
    cudaMemsetAsync(gpu_state.d_cell_counts, 0, morton_capacity * sizeof(int32_t), stream);

    // Step 2: Count particles per cell (also stores Morton cell ID per particle)
    LAUNCH_KERNEL(count_cells_kernel, num_particle_blocks, block_size, 0, stream,
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        gpu_state.d_cell_counts,
        gpu_state.temp_keys,  // Reuse temp_keys to store particle Morton cell IDs
        num_particles
    );

    // Step 3: Prefix sum (exclusive scan) to get cell_start positions
    // cell_start[i] = sum of counts for cells 0..i-1
    // Must cover morton_capacity elements (not total_cells) since Morton codes can exceed total_cells
#ifndef CUDADEM_CPU_MOCK
    cub::DeviceScan::ExclusiveSum(gpu_state.d_cub_temp_storage, gpu_state.cub_temp_storage_bytes,
                                   gpu_state.d_cell_counts, state.neighbor.cell_start,
                                   morton_capacity, stream);
#else
    // CPU mock: simple sequential prefix sum
    std::vector<int32_t> counts(morton_capacity);
    std::vector<int32_t> starts(morton_capacity);
    cudaMemcpy(counts.data(), gpu_state.d_cell_counts, morton_capacity * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t sum = 0;
    for (int i = 0; i < morton_capacity; ++i) {
        starts[i] = sum;
        sum += counts[i];
    }
    cudaMemcpy(state.neighbor.cell_start, starts.data(), morton_capacity * sizeof(int32_t), cudaMemcpyHostToDevice);
#endif

    // Step 4: Copy cell_start to cell_offsets (atomic write cursors)
    cudaMemcpyAsync(gpu_state.d_cell_offsets, state.neighbor.cell_start,
                    morton_capacity * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    // Step 5: Bin particles into sorted order using atomic increments
    LAUNCH_KERNEL(bin_particles_kernel, num_particle_blocks, block_size, 0, stream,
        gpu_state.temp_keys,  // Morton cell IDs per particle
        state.neighbor.cell_start,
        gpu_state.d_cell_offsets,
        state.neighbor.particle_indices,  // Output: sorted indices
        num_particles
    );

    // Step 6: Build cell_end from cell_start + cell_counts (use morton_capacity for bounds)
    LAUNCH_KERNEL(build_cell_end_kernel, num_cell_blocks, block_size, 0, stream,
        state.neighbor.cell_start,
        gpu_state.d_cell_counts,
        state.neighbor.cell_end,
        morton_capacity
    );

    // Note: We keep state.neighbor.cell_ids in original particle order for change detection
    // The Morton cell IDs in temp_keys are used only for sorting
}

void CLLSearchImpl::build_cell_ranges(
    SimulationState& state,
    cudaStream_t stream
) {
    CLLGPUState& gpu_state = get_gpu_state();

    int32_t num_particles = state.particles.num_particles;
    int32_t morton_capacity = state.neighbor.morton_capacity;

    int block_size = 256;

    // Initialize cell_start and cell_end to -1 across full morton_capacity range
    thrust::device_ptr<int32_t> d_start(state.neighbor.cell_start);
    thrust::device_ptr<int32_t> d_end(state.neighbor.cell_end);
#ifndef CUDADEM_CPU_MOCK
    thrust::fill(thrust::cuda::par.on(stream), d_start, d_start + morton_capacity, -1);
    thrust::fill(thrust::cuda::par.on(stream), d_end, d_end + morton_capacity, -1);
#else
    thrust::fill(d_start, d_start + morton_capacity, -1);
    thrust::fill(d_end, d_end + morton_capacity, -1);
#endif

    // Step 2: Build cell ranges
    // Use temp_keys (sorted cell IDs) instead of state.neighbor.cell_ids
    int num_particle_blocks = (num_particles + block_size - 1) / block_size;
    LAUNCH_KERNEL(build_cell_ranges_kernel, num_particle_blocks, block_size, 0, stream,
        gpu_state.temp_keys,
        state.neighbor.cell_start,
        state.neighbor.cell_end,
        num_particles,
        morton_capacity
    );
}

void CLLSearchImpl::reorder_particle_data(
    SimulationState& state,
    cudaStream_t stream
) {
    CLLGPUState& gpu_state = get_gpu_state();

    int32_t num_particles = state.particles.num_particles;
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    // ==========================================================================
    // Step 1: Reorder all particle attribute arrays using sorted indices
    // ==========================================================================
    // CRITICAL: All particle attributes must be reordered together to maintain
    // index consistency. After sorting, particle_indices[new_idx] = old_idx,
    // so dst[new_idx] = src[old_idx].
    // ==========================================================================

    // Reorder positions (double precision)
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.pos_x, gpu_state.temp_pos_x,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.pos_y, gpu_state.temp_pos_y,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.pos_z, gpu_state.temp_pos_z,
        state.neighbor.particle_indices, num_particles);

    // Reorder linear velocities
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.vel_x, gpu_state.temp_vel_x,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.vel_y, gpu_state.temp_vel_y,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.vel_z, gpu_state.temp_vel_z,
        state.neighbor.particle_indices, num_particles);

    // Reorder angular velocities (omega)
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.omega_x, gpu_state.temp_omega_x,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.omega_y, gpu_state.temp_omega_y,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.omega_z, gpu_state.temp_omega_z,
        state.neighbor.particle_indices, num_particles);

    // Reorder radii, masses, and moment of inertia
    LAUNCH_KERNEL(reorder_array_kernel<float>, num_blocks, block_size, 0, stream,
        state.particles.radius, gpu_state.temp_radius,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<float>, num_blocks, block_size, 0, stream,
        state.particles.mass, gpu_state.temp_mass,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<float>, num_blocks, block_size, 0, stream,
        state.particles.moment_of_inertia, gpu_state.temp_moment_of_inertia,
        state.neighbor.particle_indices, num_particles);

    // Reorder torques
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.torque_x, (double*)gpu_state.temp_torque_x,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.torque_y, (double*)gpu_state.temp_torque_y,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.torque_z, (double*)gpu_state.temp_torque_z,
        state.neighbor.particle_indices, num_particles);

    // Reorder orientation quaternion
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.quat_w, gpu_state.temp_quat_w,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.quat_x, gpu_state.temp_quat_x,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.quat_y, gpu_state.temp_quat_y,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<double>, num_blocks, block_size, 0, stream,
        state.particles.quat_z, gpu_state.temp_quat_z,
        state.neighbor.particle_indices, num_particles);

    // Reorder cohesive strength
    LAUNCH_KERNEL(reorder_array_kernel<float>, num_blocks, block_size, 0, stream,
        state.particles.cohesive_strength, gpu_state.temp_cohesive_strength,
        state.neighbor.particle_indices, num_particles);

    LAUNCH_KERNEL(reorder_array_kernel<int32_t>, num_blocks, block_size, 0, stream,
        state.particles.material_id, gpu_state.temp_material_id,
        state.neighbor.particle_indices, num_particles);

    // Reorder contact history offsets and counts
    // NOTE: The actual contact history data is NOT reordered here because the
    // history entries reference partner particle indices which would also need
    // to be remapped. Instead, after reordering the offset arrays, the contact
    // force code must use the original_index lookup to find history entries.
    LAUNCH_KERNEL(reorder_array_kernel<int32_t>, num_blocks, block_size, 0, stream,
        state.particles.contact_history_offset, gpu_state.temp_contact_history_offset,
        state.neighbor.particle_indices, num_particles);
    LAUNCH_KERNEL(reorder_array_kernel<int32_t>, num_blocks, block_size, 0, stream,
        state.particles.num_contacts, gpu_state.temp_num_contacts,
        state.neighbor.particle_indices, num_particles);

    // ==========================================================================
    // Step 2: Copy back all reordered data to particle arrays
    // ==========================================================================

    // Copy back positions
    cudaMemcpyAsync(state.particles.pos_x, gpu_state.temp_pos_x,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.pos_y, gpu_state.temp_pos_y,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.pos_z, gpu_state.temp_pos_z,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    // Copy back linear velocities
    cudaMemcpyAsync(state.particles.vel_x, gpu_state.temp_vel_x,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.vel_y, gpu_state.temp_vel_y,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.vel_z, gpu_state.temp_vel_z,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    // Copy back angular velocities
    cudaMemcpyAsync(state.particles.omega_x, gpu_state.temp_omega_x,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.omega_y, gpu_state.temp_omega_y,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.omega_z, gpu_state.temp_omega_z,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    // Copy back radii, masses, and moment of inertia
    cudaMemcpyAsync(state.particles.radius, gpu_state.temp_radius,
                    num_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.mass, gpu_state.temp_mass,
                    num_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.moment_of_inertia, gpu_state.temp_moment_of_inertia,
                    num_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Copy back torques
    cudaMemcpyAsync(state.particles.torque_x, gpu_state.temp_torque_x,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.torque_y, gpu_state.temp_torque_y,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.torque_z, gpu_state.temp_torque_z,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    // Copy back quaternion
    cudaMemcpyAsync(state.particles.quat_w, gpu_state.temp_quat_w,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.quat_x, gpu_state.temp_quat_x,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.quat_y, gpu_state.temp_quat_y,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.quat_z, gpu_state.temp_quat_z,
                    num_particles * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    // Copy back cohesive strength
    cudaMemcpyAsync(state.particles.cohesive_strength, gpu_state.temp_cohesive_strength,
                    num_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.material_id, gpu_state.temp_material_id,
                    num_particles * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    // Copy back contact history offsets and counts
    cudaMemcpyAsync(state.particles.contact_history_offset, gpu_state.temp_contact_history_offset,
                    num_particles * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.particles.num_contacts, gpu_state.temp_num_contacts,
                    num_particles * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    // Update sorted_index to reflect new ordering (identity mapping after reorder)
    thrust::device_ptr<int32_t> d_sorted_idx(state.particles.sorted_index);
#ifndef CUDADEM_CPU_MOCK
    thrust::sequence(thrust::cuda::par.on(stream), d_sorted_idx,
                     d_sorted_idx + num_particles);
#else
    thrust::sequence(d_sorted_idx, d_sorted_idx + num_particles);
#endif
}

// =============================================================================
// Shutdown
// =============================================================================

// Shutdown current GPU's state (queries cudaGetDevice)
void CLLSearchImpl::shutdown(SimulationState& state) {
    int32_t gpu_id;
    cudaGetDevice(&gpu_id);
    shutdown(state, gpu_id);
}

// Shutdown specific GPU's state
void CLLSearchImpl::shutdown(SimulationState& state, int32_t gpu_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    auto it = gpu_states_.find(gpu_id);
    if (it == gpu_states_.end()) {
        return;  // No state for this GPU
    }

    CLLGPUState& gpu_state = it->second;
    if (!gpu_state.initialized) {
        return;
    }

    // Ensure we're on the correct GPU for CUDA operations
    int32_t current_gpu;
    cudaGetDevice(&current_gpu);
    if (current_gpu != gpu_id) {
        cudaSetDevice(gpu_id);
    }

    if (gpu_state.shear_heads) cudaFree(gpu_state.shear_heads);
    if (gpu_state.shear_next) cudaFree(gpu_state.shear_next);
    gpu_state.shear_heads = nullptr;
    gpu_state.shear_next = nullptr;
    gpu_state.shear_heads_capacity = 0;
    gpu_state.shear_particle_capacity = 0;
    if (gpu_state.shear_max_support.device_result)
        gpu_state.shear_max_support.release();

    // Free sorting buffers
    if (gpu_state.prev_cell_ids) {
        cudaFree(gpu_state.prev_cell_ids);
        gpu_state.prev_cell_ids = nullptr;
    }
    if (gpu_state.temp_keys) {
        cudaFree(gpu_state.temp_keys);
        gpu_state.temp_keys = nullptr;
    }
    if (gpu_state.temp_values) {
        cudaFree(gpu_state.temp_values);
        gpu_state.temp_values = nullptr;
    }

    // Free reorder buffers
    if (gpu_state.temp_pos_x) cudaFree(gpu_state.temp_pos_x); gpu_state.temp_pos_x = nullptr;
    if (gpu_state.temp_pos_y) cudaFree(gpu_state.temp_pos_y); gpu_state.temp_pos_y = nullptr;
    if (gpu_state.temp_pos_z) cudaFree(gpu_state.temp_pos_z); gpu_state.temp_pos_z = nullptr;
    if (gpu_state.temp_vel_x) cudaFree(gpu_state.temp_vel_x); gpu_state.temp_vel_x = nullptr;
    if (gpu_state.temp_vel_y) cudaFree(gpu_state.temp_vel_y); gpu_state.temp_vel_y = nullptr;
    if (gpu_state.temp_vel_z) cudaFree(gpu_state.temp_vel_z); gpu_state.temp_vel_z = nullptr;
    if (gpu_state.temp_radius) cudaFree(gpu_state.temp_radius); gpu_state.temp_radius = nullptr;
    if (gpu_state.temp_mass) cudaFree(gpu_state.temp_mass); gpu_state.temp_mass = nullptr;

    // Free additional reorder buffers for complete particle data synchronization
    if (gpu_state.temp_omega_x) cudaFree(gpu_state.temp_omega_x); gpu_state.temp_omega_x = nullptr;
    if (gpu_state.temp_omega_y) cudaFree(gpu_state.temp_omega_y); gpu_state.temp_omega_y = nullptr;
    if (gpu_state.temp_omega_z) cudaFree(gpu_state.temp_omega_z); gpu_state.temp_omega_z = nullptr;
    if (gpu_state.temp_moment_of_inertia) cudaFree(gpu_state.temp_moment_of_inertia); gpu_state.temp_moment_of_inertia = nullptr;
    if (gpu_state.temp_torque_x) cudaFree(gpu_state.temp_torque_x); gpu_state.temp_torque_x = nullptr;
    if (gpu_state.temp_torque_y) cudaFree(gpu_state.temp_torque_y); gpu_state.temp_torque_y = nullptr;
    if (gpu_state.temp_torque_z) cudaFree(gpu_state.temp_torque_z); gpu_state.temp_torque_z = nullptr;
    if (gpu_state.temp_quat_w) cudaFree(gpu_state.temp_quat_w); gpu_state.temp_quat_w = nullptr;
    if (gpu_state.temp_quat_x) cudaFree(gpu_state.temp_quat_x); gpu_state.temp_quat_x = nullptr;
    if (gpu_state.temp_quat_y) cudaFree(gpu_state.temp_quat_y); gpu_state.temp_quat_y = nullptr;
    if (gpu_state.temp_quat_z) cudaFree(gpu_state.temp_quat_z); gpu_state.temp_quat_z = nullptr;
    if (gpu_state.temp_cohesive_strength) cudaFree(gpu_state.temp_cohesive_strength); gpu_state.temp_cohesive_strength = nullptr;
    if (gpu_state.temp_material_id) cudaFree(gpu_state.temp_material_id); gpu_state.temp_material_id = nullptr;
    if (gpu_state.temp_contact_history_offset) cudaFree(gpu_state.temp_contact_history_offset); gpu_state.temp_contact_history_offset = nullptr;
    if (gpu_state.temp_num_contacts) cudaFree(gpu_state.temp_num_contacts); gpu_state.temp_num_contacts = nullptr;

    // Free change flag
    if (gpu_state.d_changed) cudaFree(gpu_state.d_changed); gpu_state.d_changed = nullptr;

    // Free atomic binning buffers
    if (gpu_state.d_cell_counts) cudaFree(gpu_state.d_cell_counts); gpu_state.d_cell_counts = nullptr;
    if (gpu_state.d_cell_offsets) cudaFree(gpu_state.d_cell_offsets); gpu_state.d_cell_offsets = nullptr;
    if (gpu_state.d_cub_temp_storage) cudaFree(gpu_state.d_cub_temp_storage); gpu_state.d_cub_temp_storage = nullptr;
    gpu_state.cub_temp_storage_bytes = 0;

    // Free state.neighbor arrays
    if (state.neighbor.cell_start) cudaFree(state.neighbor.cell_start); state.neighbor.cell_start = nullptr;
    if (state.neighbor.cell_end) cudaFree(state.neighbor.cell_end); state.neighbor.cell_end = nullptr;
    if (state.neighbor.particle_indices) cudaFree(state.neighbor.particle_indices); state.neighbor.particle_indices = nullptr;
    if (state.neighbor.cell_ids) cudaFree(state.neighbor.cell_ids); state.neighbor.cell_ids = nullptr;

    // Free async readback resources
    if (gpu_state.h_pinned_max_disp_sq) {
        cudaFreeHost(gpu_state.h_pinned_max_disp_sq);
        gpu_state.h_pinned_max_disp_sq = nullptr;
    }
    if (gpu_state.disp_copy_done_event) {
        cudaEventDestroy(gpu_state.disp_copy_done_event);
        gpu_state.disp_copy_done_event = nullptr;
    }
    gpu_state.cached_max_disp_sq = 0.0f;
    gpu_state.async_readback_pending = false;
    gpu_state.steps_since_rebuild = 0;

    // Free Verlet cache
    if (gpu_state.verlet_cache.cache_allocated) {
        cudaFree(gpu_state.verlet_cache.pair_i);
        cudaFree(gpu_state.verlet_cache.pair_j);
        cudaFree(gpu_state.verlet_cache.pair_image_x);
        cudaFree(gpu_state.verlet_cache.pair_image_y);
        cudaFree(gpu_state.verlet_cache.pair_image_z);
        gpu_state.verlet_cache.cache_allocated = false;
        gpu_state.verlet_cache.cache_valid = false;
    }

    // Reset state
    gpu_state.initialized = false;
    gpu_state.grid_valid = false;
    gpu_state.last_rebuild_step = -1;
    gpu_state.stats = NeighborSearchStats();

    // Restore original GPU if we switched
    if (current_gpu != gpu_id) {
        cudaSetDevice(current_gpu);
    }
}

// Shutdown all GPUs (call at end of simulation)
void CLLSearchImpl::shutdown_all() {
    // Get list of GPU IDs first (to avoid iterator invalidation)
    std::vector<int32_t> gpu_ids;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (const auto& pair : gpu_states_) {
            gpu_ids.push_back(pair.first);
        }
    }

    // Shutdown each GPU
    SimulationState dummy_state;  // Placeholder for API compatibility
    for (int32_t gpu_id : gpu_ids) {
        shutdown(dummy_state, gpu_id);
    }

    // Clear the map
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        gpu_states_.clear();
    }
}

// =============================================================================
// Query Functions
// =============================================================================

bool CLLSearchImpl::is_initialized() {
    int32_t gpu_id;
    if (cudaGetDevice(&gpu_id) != cudaSuccess) {
        return false;
    }
    return is_initialized(gpu_id);
}

bool CLLSearchImpl::is_initialized(int32_t gpu_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = gpu_states_.find(gpu_id);
    if (it == gpu_states_.end()) {
        return false;
    }
    return it->second.initialized;
}

const NeighborSearchStats& CLLSearchImpl::get_stats() {
    int32_t gpu_id;
    cudaGetDevice(&gpu_id);
    return get_stats(gpu_id);
}

const NeighborSearchStats& CLLSearchImpl::get_stats(int32_t gpu_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = gpu_states_.find(gpu_id);
    if (it == gpu_states_.end()) {
        return default_stats_;
    }
    return it->second.stats;
}

} // namespace cll
} // namespace cudadem

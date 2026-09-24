// =============================================================================
// Production Karras fixed-depth D-ary tree builder
// =============================================================================
// The builder follows Karras 2012 directly:
//   1. generate and sort configured-depth Morton cell keys;
//   2. compact equal cells into unique radix leaves while retaining particle
//      ranges;
//   3. build the binary radix tree and count D-ary prefixes on its edges in the
//      same kernel;
//   4. prefix-sum edge counts and materialize the occupied D-ary prefixes;
//   5. link edge chains directly, then emit child masks and level lists.
//
// All addresses are allocated once and captured in the persistent CUDA graphs.
// No rebuild performs allocation, host reduction, metadata readback, or host
// synchronization.
// =============================================================================

#include "karras_octree_build.cuh"

#include "dense_level_map.cuh"
#include "morton_unified.cuh"
#include "persistent_tree_traversal.cuh"
#include "tree_traits.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace body_fitted_fmm {

namespace {

constexpr int KARRAS_BLOCK = 256;

template<typename T>
void karras_device_alloc(T*& pointer, size_t count) {
    if (count == 0) {
        pointer = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
}

template<typename T>
void karras_device_free(T*& pointer) {
    if (pointer) CUDA_CHECK(cudaFree(pointer));
    pointer = nullptr;
}

__device__ __forceinline__ void karras_publish_error(
    PersistentTreeContext* ctx, PersistentBuildError error)
{
    atomicCAS(&ctx->metadata.error, 0, static_cast<int>(error));
}

__device__ __forceinline__ uint64_t karras_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void karras_reset_kernel(PersistentTreeContext* ctx) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (tid == 0) {
        ctx->metadata.tree_start_ns = karras_globaltimer_ns();
        ctx->metadata.tree_end_ns = 0;
        ctx->metadata.stencil_end_ns = 0;
        ctx->metadata.tree_sort_end_ns = 0;
        ctx->metadata.num_children = 0;
        ctx->metadata.num_unique_cells = 0;
        ctx->metadata.error = 0;
        for (int phase = 0; phase < 9; ++phase)
            ctx->metadata.tree_phase_ns[phase] = 0;
        for (int s = 0; s < 4; ++s) {
            ctx->arena.d_counters[s] = 0;
            ctx->metadata.task_counts[s] = 0;
        }
    }

    for (int i = tid; i < ctx->dense_capacity_total; i += stride) {
        ctx->workspace.dense_storage[i] = 0xFFFFFFFFu;
        ctx->workspace.dense_linear_storage[i] = 0xFFFFFFFFu;
    }
    for (uint64_t i = static_cast<uint64_t>(tid);
        i < ctx->sparse_capacity_total; i += static_cast<uint64_t>(stride)) {
        ctx->workspace.sparse_keys[i] = UINT64_MAX;
        // Empty keys make old values unreachable; insertion overwrites values
        // before any later kernel can look up the rebuilt tree.
    }
    for (int i = tid; i < fmm::FMM_MAX_LEVELS; i += stride) {
        ctx->dd.level_start[i] = 0;
        ctx->dd.level_count[i] = 0;
    }
    // The allocation starts invalid, and materialization fully assigns every
    // new node. Only the previous live range needs invalidating, including LET
    // imports when present. Keep num_nodes stable until the next kernel so
    // these threads cannot race a thread-0 reset of their loop bound.
    for (int i = tid; i < ctx->metadata.num_nodes; i += stride) {
        ctx->dd.unified_nodes[i].flags = 0;
    }
}

__global__ void karras_children_reset_kernel(PersistentTreeContext* ctx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ctx->metadata.num_nodes || ctx->metadata.error != 0) return;
    #pragma unroll
    for (int c = 0; c < 8; ++c)
        ctx->dd.children_compact[static_cast<size_t>(idx) * 8 + c] = -1;
}

__global__ void karras_phase_marker_kernel(PersistentTreeContext* ctx,
                                            int phase) {
    // Keep the original nine phase boundaries stable for existing consumers.
    if (blockIdx.x == 0 && threadIdx.x == 0 && phase == 9)
        ctx->metadata.tree_sort_end_ns = karras_globaltimer_ns();
    if (blockIdx.x == 0 && threadIdx.x == 0 && phase >= 0 && phase < 9) {
        if (phase == 0) ctx->metadata.num_nodes = 0;
        ctx->metadata.tree_phase_ns[phase] = karras_globaltimer_ns();
    }
}

__global__ void karras_timing_end_kernel(PersistentTreeContext* ctx) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        ctx->metadata.tree_end_ns = karras_globaltimer_ns();
}

__global__ void karras_bounds_block_kernel(PersistentTreeContext* ctx) {
    extern __shared__ double shared[];
    double* sminx = shared;
    double* sminy = sminx + blockDim.x;
    double* sminz = sminy + blockDim.x;
    double* smaxx = sminz + blockDim.x;
    double* smaxy = smaxx + blockDim.x;
    double* smaxz = smaxy + blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = ctx->command.particle_count;
    double x = DBL_MAX, y = DBL_MAX, z = DBL_MAX;
    double X = -DBL_MAX, Y = -DBL_MAX, Z = -DBL_MAX;
    if (i < n) {
        x = X = ctx->command.pos_x[i];
        y = Y = ctx->command.pos_y[i];
        z = Z = ctx->command.pos_z[i];
    }
    sminx[threadIdx.x] = x; sminy[threadIdx.x] = y; sminz[threadIdx.x] = z;
    smaxx[threadIdx.x] = X; smaxy[threadIdx.x] = Y; smaxz[threadIdx.x] = Z;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sminx[threadIdx.x] = fmin(sminx[threadIdx.x], sminx[threadIdx.x + offset]);
            sminy[threadIdx.x] = fmin(sminy[threadIdx.x], sminy[threadIdx.x + offset]);
            sminz[threadIdx.x] = fmin(sminz[threadIdx.x], sminz[threadIdx.x + offset]);
            smaxx[threadIdx.x] = fmax(smaxx[threadIdx.x], smaxx[threadIdx.x + offset]);
            smaxy[threadIdx.x] = fmax(smaxy[threadIdx.x], smaxy[threadIdx.x + offset]);
            smaxz[threadIdx.x] = fmax(smaxz[threadIdx.x], smaxz[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ctx->workspace.block_min_x[blockIdx.x] = sminx[0];
        ctx->workspace.block_min_y[blockIdx.x] = sminy[0];
        ctx->workspace.block_min_z[blockIdx.x] = sminz[0];
        ctx->workspace.block_max_x[blockIdx.x] = smaxx[0];
        ctx->workspace.block_max_y[blockIdx.x] = smaxy[0];
        ctx->workspace.block_max_z[blockIdx.x] = smaxz[0];
    }
}

template<int D>
__global__ void karras_bounds_final_kernel(PersistentTreeContext* ctx,
                                            int block_capacity) {
    __shared__ double sminx[KARRAS_BLOCK], sminy[KARRAS_BLOCK], sminz[KARRAS_BLOCK];
    __shared__ double smaxx[KARRAS_BLOCK], smaxy[KARRAS_BLOCK], smaxz[KARRAS_BLOCK];
    const int active = (ctx->command.particle_count + KARRAS_BLOCK - 1) /
                       KARRAS_BLOCK;
    double minx = DBL_MAX, miny = DBL_MAX, minz = DBL_MAX;
    double maxx = -DBL_MAX, maxy = -DBL_MAX, maxz = -DBL_MAX;
    for (int i = threadIdx.x; i < active && i < block_capacity;
         i += blockDim.x) {
        minx = fmin(minx, ctx->workspace.block_min_x[i]);
        miny = fmin(miny, ctx->workspace.block_min_y[i]);
        minz = fmin(minz, ctx->workspace.block_min_z[i]);
        maxx = fmax(maxx, ctx->workspace.block_max_x[i]);
        maxy = fmax(maxy, ctx->workspace.block_max_y[i]);
        maxz = fmax(maxz, ctx->workspace.block_max_z[i]);
    }
    sminx[threadIdx.x] = minx; sminy[threadIdx.x] = miny; sminz[threadIdx.x] = minz;
    smaxx[threadIdx.x] = maxx; smaxy[threadIdx.x] = maxy; smaxz[threadIdx.x] = maxz;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sminx[threadIdx.x] = fmin(sminx[threadIdx.x], sminx[threadIdx.x + offset]);
            sminy[threadIdx.x] = fmin(sminy[threadIdx.x], sminy[threadIdx.x + offset]);
            sminz[threadIdx.x] = fmin(sminz[threadIdx.x], sminz[threadIdx.x + offset]);
            smaxx[threadIdx.x] = fmax(smaxx[threadIdx.x], smaxx[threadIdx.x + offset]);
            smaxy[threadIdx.x] = fmax(smaxy[threadIdx.x], smaxy[threadIdx.x + offset]);
            smaxz[threadIdx.x] = fmax(smaxz[threadIdx.x], smaxz[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (ctx->command.particle_count <= 0 ||
            ctx->command.particle_count > ctx->particle_capacity) {
            karras_publish_error(ctx, PersistentBuildError::InvalidParticleCount);
            return;
        }
        if (ctx->command.use_fixed_root) {
            ctx->metadata.extent_x = smaxx[0] - sminx[0];
            ctx->metadata.extent_y = smaxy[0] - sminy[0];
            ctx->metadata.extent_z = smaxz[0] - sminz[0];
            ctx->metadata.domain_min_x = ctx->command.fixed_domain_min_x;
            ctx->metadata.domain_min_y = ctx->command.fixed_domain_min_y;
            if constexpr (D == 2) {
                // The quadtree does not key on z.  Keep every planar
                // expansion center on the actual particle slab instead of
                // the much taller fixed supertree envelope; otherwise L2P
                // evaluates far outside its normalized local cell.
                const double pad_z = 0.01 * fmax(
                    ctx->metadata.extent_x,
                    fmax(ctx->metadata.extent_y, ctx->metadata.extent_z));
                ctx->metadata.domain_min_z = sminz[0] - pad_z;
            } else {
                ctx->metadata.domain_min_z = ctx->command.fixed_domain_min_z;
            }
            ctx->metadata.root_cell_size = ctx->command.fixed_root_cell_size;
            return;
        }
        const double ex = smaxx[0] - sminx[0];
        const double ey = smaxy[0] - sminy[0];
        const double ez = smaxz[0] - sminz[0];
        double max_extent = fmax(ex, fmax(ey, ez));
        if (max_extent < 1e-30) max_extent = 1.0;
        const double pad = max_extent * 0.01;
        ctx->metadata.extent_x = ex;
        ctx->metadata.extent_y = ey;
        ctx->metadata.extent_z = ez;
        ctx->metadata.domain_min_x = sminx[0] - pad;
        ctx->metadata.domain_min_y = sminy[0] - pad;
        ctx->metadata.domain_min_z = sminz[0] - pad;
        ctx->metadata.root_cell_size = max_extent + 2.0 * pad;
    }
}

template<int D>
__global__ void karras_morton_kernel(PersistentTreeContext* ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ctx->particle_capacity) return;
    const int n = ctx->command.particle_count;
    if (i >= n || ctx->metadata.error != 0) {
        ctx->workspace.morton_unsorted[i] = UINT64_MAX;
        ctx->workspace.index_unsorted[i] = -1;
        return;
    }

    const double inv = 1.0 / ctx->metadata.root_cell_size;
    const double raw_fx = (ctx->command.pos_x[i] - ctx->metadata.domain_min_x) * inv;
    const double raw_fy = (ctx->command.pos_y[i] - ctx->metadata.domain_min_y) * inv;
    const double raw_fz = (ctx->command.pos_z[i] - ctx->metadata.domain_min_z) * inv;
    if (ctx->command.use_fixed_root &&
        (raw_fx < 0.0 || raw_fx >= 1.0 || raw_fy < 0.0 || raw_fy >= 1.0 ||
         (D == 3 && (raw_fz < 0.0 || raw_fz >= 1.0)))) {
        karras_publish_error(ctx, PersistentBuildError::ParticleOutsideFixedRoot);
        ctx->workspace.morton_unsorted[i] = UINT64_MAX;
        ctx->workspace.index_unsorted[i] = -1;
        return;
    }
    const double fx = fmin(fmax(raw_fx, 0.0), 0.9999999999999999);
    const double fy = fmin(fmax(raw_fy, 0.0), 0.9999999999999999);
    const double fz = fmin(fmax(raw_fz, 0.0), 0.9999999999999999);
    constexpr int bits_per_axis = TreeTraits<D>::MORTON_BITS / D;
    constexpr int quant_bits = bits_per_axis > 31 ? 31 : bits_per_axis;
    constexpr double scale = static_cast<double>(1ull << quant_bits);
    const uint32_t x = static_cast<uint32_t>(fx * scale);
    const uint32_t y = static_cast<uint32_t>(fy * scale);
    const uint32_t z = static_cast<uint32_t>(fz * scale);
    uint64_t code = morton_encode<D>(x, y, z);
    const int shift = TreeTraits<D>::MORTON_BITS - D * ctx->tree_depth;
    if (shift > 0) code &= ~((1ull << shift) - 1ull);
    ctx->workspace.morton_unsorted[i] = code;
    ctx->workspace.index_unsorted[i] = i;
}

__global__ void karras_import_presorted_kernel(PersistentTreeContext* ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ctx->particle_capacity || ctx->metadata.error) return;
    const auto& input = ctx->command.presorted;
    const int n = ctx->command.particle_count;
    if (!input.keys || !input.indices || input.count != n || input.depth != ctx->tree_depth ||
        input.min_x != ctx->metadata.domain_min_x || input.min_y != ctx->metadata.domain_min_y ||
        input.min_z != ctx->metadata.domain_min_z || input.root_size != ctx->metadata.root_cell_size) {
        if (i == 0) karras_publish_error(ctx, PersistentBuildError::InvalidPresortedMortonInput);
        return;
    }
    if (i >= n) {
        ctx->workspace.morton_sorted[i] = UINT64_MAX;
        ctx->dd.sorted_to_original[i] = -1;
        return;
    }
    // The producer normally writes these exact allocations before the graph.
    // Preserve support for separate input arrays without requiring a copy.
    if (input.keys != ctx->workspace.morton_sorted)
        ctx->workspace.morton_sorted[i] = input.keys[i];
    if (input.indices != ctx->dd.sorted_to_original)
        ctx->dd.sorted_to_original[i] = input.indices[i];
    if (!input.validate) return;
    const int original = input.indices[i];
    const uint64_t key = input.keys[i];
    if (original < 0 || original >= n ||
        (i && (input.keys[i - 1] > key ||
               (input.keys[i - 1] == key && input.indices[i - 1] >= original)))) {
        karras_publish_error(ctx, PersistentBuildError::PresortedMortonMismatch);
        return;
    }
    const double inv = 1.0 / ctx->metadata.root_cell_size;
    const double fx = fmin(fmax((ctx->command.pos_x[original] - input.min_x) * inv, 0.0), 0.9999999999999999);
    const double fy = fmin(fmax((ctx->command.pos_y[original] - input.min_y) * inv, 0.0), 0.9999999999999999);
    const double fz = fmin(fmax((ctx->command.pos_z[original] - input.min_z) * inv, 0.0), 0.9999999999999999);
    uint64_t expected = morton_encode<3>(static_cast<uint32_t>(fx * double(1u << 21)),
        static_cast<uint32_t>(fy * double(1u << 21)), static_cast<uint32_t>(fz * double(1u << 21)));
    const int shift = 63 - 3 * ctx->tree_depth;
    expected &= ~((uint64_t{1} << shift) - 1);
    if (expected != key) karras_publish_error(ctx, PersistentBuildError::PresortedMortonMismatch);
}

struct KarrasUniqueRunStart {
    const PersistentTreeContext* ctx;

    __device__ __forceinline__ bool operator()(int i) const {
        // Graphs sort their fixed capacity. Exclude the sentinel tail even
        // when its restricted sort bits tie the highest occupied Morton cell.
        return i < ctx->command.particle_count && ctx->metadata.error == 0 &&
            (i == 0 || ctx->workspace.morton_sorted[i] !=
                       ctx->workspace.morton_sorted[i - 1]);
    }
};

__global__ void karras_compact_unique_cells_kernel(PersistentTreeContext* ctx) {
    const int unique = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = ctx->metadata.num_unique_cells;
    if (unique >= count || ctx->metadata.error != 0) return;

    // Stable selection emits monotonically increasing run starts. Only live
    // unique cells materialize keys/ranges, rather than every source particle.
    const int begin = ctx->workspace.unique_offsets[unique];
    const int end = unique + 1 < count
        ? ctx->workspace.unique_offsets[unique + 1]
        : ctx->command.particle_count;
    ctx->workspace.morton_unsorted[unique] = ctx->workspace.morton_sorted[begin];
    ctx->workspace.unique_ranges[unique] = make_int2(begin, end);
}

__global__ void karras_mark_target_particles_kernel(PersistentTreeContext* ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ctx->particle_capacity) return;
    // The selected run starts are dead after unique_ranges has been emitted.
    // Reuse the buffers for an exclusive target prefix; initialize the tail
    // as well because the device-launchable graph scans a fixed allocation.
    ctx->workspace.index_unsorted[i] =
        i < ctx->command.particle_count && ctx->metadata.error == 0 &&
        ctx->dd.sorted_to_original[i] < ctx->command.target_particle_count;
}

template<int D>
__device__ __forceinline__ int karras_range_level(
    const PersistentTreeContext* ctx, int first, int last_exclusive)
{
    if (last_exclusive - first <= 1) return ctx->tree_depth;
    const uint64_t a = ctx->workspace.morton_unsorted[first];
    const uint64_t b = ctx->workspace.morton_unsorted[last_exclusive - 1];
    if (a == b) return ctx->tree_depth;
    constexpr int padding = 64 - TreeTraits<D>::MORTON_BITS;
    int logical_lcp = __clzll(a ^ b) - padding;
    logical_lcp = max(0, logical_lcp);
    return min(ctx->tree_depth, logical_lcp / D);
}

template<int D>
__global__ void karras_build_radix_and_edges_kernel(PersistentTreeContext* ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = ctx->metadata.num_unique_cells;
    if (ctx->metadata.error != 0 || n <= 1 || i >= n - 1) return;

    KarrasBinaryNode* nodes = ctx->workspace.binary_nodes;
    const uint64_t* keys = ctx->workspace.morton_unsorted;
    const uint2 range = karras_determine_range(keys, n, i);
    const unsigned int gamma = karras_find_split(keys, n, range.x, range.y);
    const unsigned int first = min(range.x, range.y);
    const unsigned int last = max(range.x, range.y);

    int left = static_cast<int>(gamma);
    int right = static_cast<int>(gamma + 1);
    if (first == gamma) left += n - 1;
    if (last == gamma + 1) right += n - 1;

    KarrasBinaryNode& node = nodes[i];
    if (i == 0) node.parent_idx = -1;
    node.particle_begin = static_cast<int>(first);
    node.particle_end = static_cast<int>(last) + 1;

    nodes[left].parent_idx = i;
    nodes[right].parent_idx = i;
    if (left >= n - 1) {
        const int leaf = left - (n - 1);
        nodes[left].particle_begin = leaf;
        nodes[left].particle_end = leaf + 1;
    }
    if (right >= n - 1) {
        const int leaf = right - (n - 1);
        nodes[right].particle_begin = leaf;
        nodes[right].particle_end = leaf + 1;
    }

    const int parent_level = karras_range_level<D>(ctx, first, last + 1);
    const int child_idx[2] = {left, right};
    const int child_first[2] = {static_cast<int>(first), static_cast<int>(gamma + 1)};
    const int child_last[2] = {static_cast<int>(gamma + 1), static_cast<int>(last + 1)};
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const int edge = 1 + 2 * i + side;
        const int child_level = karras_range_level<D>(
            ctx, child_first[side], child_last[side]);
        ctx->workspace.edge_counts[edge] = max(0, child_level - parent_level);
        ctx->workspace.edge_children[edge] = child_idx[side];
        ctx->workspace.incoming_edge[child_idx[side]] = edge;
    }
    if (i == 0) {
        ctx->workspace.edge_counts[0] = parent_level;
        ctx->workspace.edge_children[0] = 0;
        ctx->workspace.incoming_edge[0] = 0;
    }
}

template<int D>
__global__ void karras_single_unique_cell_kernel(PersistentTreeContext* ctx) {
    if (blockIdx.x != 0 || threadIdx.x != 0 ||
        ctx->metadata.num_unique_cells != 1 || ctx->metadata.error != 0)
        return;
    KarrasBinaryNode node{};
    node.parent_idx = -1;
    node.particle_begin = 0;
    node.particle_end = 1;
    ctx->workspace.binary_nodes[0] = node;
    ctx->workspace.edge_counts[0] = ctx->tree_depth;
    ctx->workspace.edge_children[0] = 0;
    ctx->workspace.incoming_edge[0] = 0;
}

template<int D>
__global__ void karras_finalize_count_and_root_kernel(PersistentTreeContext* ctx) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || ctx->metadata.error != 0) return;
    const int unique = ctx->metadata.num_unique_cells;
    if (unique <= 0) {
        karras_publish_error(ctx, PersistentBuildError::InvalidParticleCount);
        return;
    }
    const int used_edges = 2 * unique - 1;
    const int last = used_edges - 1;
    const int count = 1 + ctx->workspace.edge_offsets[last] +
                          ctx->workspace.edge_counts[last];
    if (count <= 0 || count > ctx->node_capacity) {
        karras_publish_error(ctx, PersistentBuildError::NodeCapacityExceeded);
        return;
    }
    ctx->metadata.num_nodes = count;
    ctx->metadata.num_children = count - 1;

    UnifiedNode root{};
    root.begin = 0;
    root.count = static_cast<uint32_t>(ctx->command.particle_count);
    root.childBase = UNIFIED_NODE_LEAF_CHILDBASE;
    root.level = 0;
    root.flags = UNIFIED_NODE_FLAG_VALID;
    double center_z = ctx->metadata.domain_min_z + 0.5 * ctx->metadata.root_cell_size;
    if constexpr (D == 2)
        center_z = ctx->metadata.domain_min_z + 0.5 * ctx->metadata.extent_z +
            0.01 * fmax(ctx->metadata.extent_x,
                        fmax(ctx->metadata.extent_y, ctx->metadata.extent_z));
    if constexpr (D == 2) if (ctx->command.common_planar_center)
        center_z = ctx->command.fixed_domain_min_z + 0.5 * ctx->command.fixed_root_cell_size;
    root.center = make_double3(
        ctx->metadata.domain_min_x + 0.5 * ctx->metadata.root_cell_size,
        ctx->metadata.domain_min_y + 0.5 * ctx->metadata.root_cell_size,
        center_z);
    root.radius = static_cast<float>(0.5 * sqrt(3.0) * ctx->metadata.root_cell_size);
    root.compact_morton = 0;
    root.target_count = static_cast<uint32_t>(ctx->command.target_particle_count);
    ctx->dd.unified_nodes[0] = root;
    ctx->dd.level_dense_map[0][0] = 0;
    ctx->workspace.dense_linear_map[0][0] = 0;
}

__device__ __forceinline__ int karras_edge_parent_node(
    const PersistentTreeContext* ctx, int edge, int child)
{
    if (edge == 0) return 0;
    int ancestor = ctx->workspace.binary_nodes[child].parent_idx;
    while (ancestor >= 0) {
        const int incoming = ctx->workspace.incoming_edge[ancestor];
        const int count = ctx->workspace.edge_counts[incoming];
        if (count > 0)
            return 1 + ctx->workspace.edge_offsets[incoming] + count - 1;
        ancestor = ctx->workspace.binary_nodes[ancestor].parent_idx;
    }
    return 0;
}

template<int D>
__global__ void karras_materialize_edges_kernel(PersistentTreeContext* ctx) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    const int unique = ctx->metadata.num_unique_cells;
    const int used_edges = unique > 0 ? 2 * unique - 1 : 0;
    if (ctx->metadata.error != 0 || edge >= used_edges) return;
    const int count = ctx->workspace.edge_counts[edge];
    if (count <= 0) return;
    const int child = ctx->workspace.edge_children[edge];
    if (child < 0) return;

    int parent_level = 0;
    if (edge != 0) {
        const int binary_parent = ctx->workspace.binary_nodes[child].parent_idx;
        parent_level = karras_range_level<D>(
            ctx,
            ctx->workspace.binary_nodes[binary_parent].particle_begin,
            ctx->workspace.binary_nodes[binary_parent].particle_end);
    }
    int parent_node = karras_edge_parent_node(ctx, edge, child);
    const int base = 1 + ctx->workspace.edge_offsets[edge];
    const KarrasBinaryNode& child_node = ctx->workspace.binary_nodes[child];
    const int unique_begin = child_node.particle_begin;
    const int unique_end = child_node.particle_end;
    const int particle_begin = ctx->workspace.unique_ranges[unique_begin].x;
    const int particle_end = ctx->workspace.unique_ranges[unique_end - 1].y;
    const uint64_t code = ctx->workspace.morton_unsorted[unique_begin];

    for (int j = 0; j < count; ++j) {
        const int idx = base + j;
        const int level = parent_level + 1 + j;
        const int shift = TreeTraits<D>::MORTON_BITS - D * level;
        const uint64_t compact = shift > 0 ? code >> shift : code;
        uint32_t x = 0, y = 0, z = 0;
        morton_decode<D>(compact, level, x, y, z);
        const double cell_size = ctx->metadata.root_cell_size /
                                 static_cast<double>(1u << level);
        double center_z = ctx->metadata.domain_min_z + (static_cast<double>(z) + 0.5) * cell_size;
        if constexpr (D == 2)
            center_z = ctx->metadata.domain_min_z + 0.5 *
                (ctx->metadata.extent_z + 0.02 *
                 fmax(ctx->metadata.extent_x,
                      fmax(ctx->metadata.extent_y, ctx->metadata.extent_z)));

        if constexpr (D == 2) if (ctx->command.common_planar_center)
            center_z = ctx->command.fixed_domain_min_z + 0.5 * ctx->command.fixed_root_cell_size;
        UnifiedNode node{};
        node.begin = static_cast<uint32_t>(particle_begin);
        node.count = static_cast<uint32_t>(particle_end - particle_begin);
        node.childBase = UNIFIED_NODE_LEAF_CHILDBASE;
        node.level = static_cast<uint8_t>(level);
        node.flags = UNIFIED_NODE_FLAG_VALID;
        node.center = make_double3(
            ctx->metadata.domain_min_x + (static_cast<double>(x) + 0.5) * cell_size,
            ctx->metadata.domain_min_y + (static_cast<double>(y) + 0.5) * cell_size,
            center_z);
        node.radius = static_cast<float>(0.5 * sqrt(3.0) * cell_size);
        node.compact_morton = compact;
        uint32_t targets = static_cast<uint32_t>(particle_end - particle_begin);
        if (ctx->command.target_particle_count != ctx->command.particle_count) {
            // Target counts are FMM metadata, not part of Karras's edge
            // conversion. Avoid serial descendant scans in coarse nodes.
            // The exclusive prefix has capacity entries; use the known total
            // at n so that a full-capacity particle range stays in bounds.
            const int end_targets = particle_end == ctx->command.particle_count
                ? ctx->command.target_particle_count
                : ctx->workspace.unique_offsets[particle_end];
            targets = static_cast<uint32_t>(end_targets -
                ctx->workspace.unique_offsets[particle_begin]);
        }
        node.target_count = targets;
        ctx->dd.unified_nodes[idx] = node;
        if (!persistent_morton_insert<D>(ctx, level, compact,
                                          static_cast<uint32_t>(idx))) {
            karras_publish_error(ctx,
                PersistentBuildError::SparseLookupCapacityExceeded);
            return;
        }
        if (level <= persistent_dense_cutoff<D>()) {
            const uint32_t side = 1u << level;
            uint32_t linear = x * side + y;
            if constexpr (D == 3) linear = linear * side + z;
            ctx->workspace.dense_linear_map[level][linear] =
                static_cast<uint32_t>(idx);
        }

        const int slot = static_cast<int>(compact & ((1ull << D) - 1ull));
        ctx->dd.children_compact[static_cast<size_t>(parent_node) * 8 + slot] = idx;
        parent_node = idx;
    }
}

template<int D>
__global__ void karras_finalize_nodes_kernel(PersistentTreeContext* ctx) {
    __shared__ int histogram[fmm::FMM_MAX_LEVELS];
    if (threadIdx.x < fmm::FMM_MAX_LEVELS) histogram[threadIdx.x] = 0;
    __syncthreads();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ctx->metadata.num_nodes && ctx->metadata.error == 0) {
        UnifiedNode& node = ctx->dd.unified_nodes[idx];
        uint8_t mask = 0;
        int packed = 0;
        const size_t base = static_cast<size_t>(idx) * 8;
        for (int c = 0; c < (1 << D); ++c) {
            const int child = ctx->dd.children_compact[base + c];
            if (child >= 0) {
                mask |= static_cast<uint8_t>(1u << c);
                ctx->dd.children_compact[base + packed++] = child;
            }
        }
        node.childMask = mask;
        if (packed == 0) {
            node.childBase = UNIFIED_NODE_LEAF_CHILDBASE;
            node.flags |= UNIFIED_NODE_FLAG_IS_LEAF;
        } else {
            node.childBase = static_cast<uint32_t>(base);
            node.flags &= static_cast<uint8_t>(~UNIFIED_NODE_FLAG_IS_LEAF);
        }
        atomicAdd(&histogram[node.level], 1);
    }
    __syncthreads();
    if (threadIdx.x < fmm::FMM_MAX_LEVELS)
        ctx->workspace.level_block_counts[
            static_cast<size_t>(threadIdx.x) * gridDim.x + blockIdx.x] =
            histogram[threadIdx.x];
}

__global__ void karras_finalize_levels_kernel(PersistentTreeContext* ctx,
                                               int block_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int start = 0;
    for (int level = 0; level < fmm::FMM_MAX_LEVELS; ++level) {
        if (level > ctx->tree_depth) {
            ctx->dd.level_start[level] = start;
            ctx->dd.level_count[level] = 0;
            continue;
        }
        const size_t base = static_cast<size_t>(level) * block_count;
        const int last = block_count - 1;
        const int count = ctx->workspace.level_block_offsets[base + last] +
                          ctx->workspace.level_block_counts[base + last];
        ctx->dd.level_start[level] = start;
        ctx->dd.level_count[level] = count;
        start += count;
    }
}

__global__ void karras_emit_level_list_kernel(PersistentTreeContext* ctx) {
    __shared__ int cursor[fmm::FMM_MAX_LEVELS];
    if (threadIdx.x < fmm::FMM_MAX_LEVELS) {
        const int level = threadIdx.x;
        cursor[level] = ctx->dd.level_start[level] +
            ctx->workspace.level_block_offsets[
                static_cast<size_t>(level) * gridDim.x + blockIdx.x];
    }
    __syncthreads();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ctx->metadata.num_nodes || ctx->metadata.error != 0) return;
    const int level = ctx->dd.unified_nodes[idx].level;
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, level);
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(peers) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(&cursor[level], __popc(peers));
    base = __shfl_sync(peers, base, leader);
    const unsigned lower = peers & ((1u << lane) - 1u);
    ctx->dd.level_list[base + __popc(lower)] = idx;
}

template<int D>
__global__ void karras_particle_leaf_map_kernel(PersistentTreeContext* ctx) {
    const int sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted >= ctx->command.particle_count || ctx->metadata.error != 0) return;
    const uint64_t code = ctx->workspace.morton_sorted[sorted];
    const int shift = TreeTraits<D>::MORTON_BITS - D * ctx->tree_depth;
    const uint64_t compact = shift > 0 ? code >> shift : code;
    ctx->dd.particle_to_leaf[sorted] =
        persistent_morton_lookup<D>(ctx, ctx->tree_depth, compact);
}

}  // namespace

size_t karras_unique_selection_scratch_bytes(int particle_capacity) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, bytes, thrust::counting_iterator<int>(0),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        particle_capacity, KarrasUniqueRunStart{nullptr}));
    return bytes;
}

void initialize_karras_tree_workspace(PersistentTreeRuntime& runtime,
                                      fmm::FMMDeviceData& dd,
                                      int tree_dimension) {
    auto& hc = runtime.host_context;
    const int particles = hc.particle_capacity;
    // Establish the invalid-tail invariant once, rather than clearing all
    // provisioned node slots on every rebuild. The first reset has zero live
    // nodes; subsequent resets invalidate the preceding tree's live range.
    CUDA_CHECK(cudaMemset(dd.unified_nodes, 0,
        static_cast<size_t>(hc.node_capacity) * sizeof(UnifiedNode)));
    const int bounds_blocks = (particles + KARRAS_BLOCK - 1) / KARRAS_BLOCK;
    auto& ws = hc.workspace;
    karras_device_alloc(ws.morton_unsorted, particles);
    karras_device_alloc(ws.morton_sorted, particles);
    karras_device_alloc(ws.index_unsorted, particles);
    karras_device_alloc(ws.unique_offsets, particles);
    karras_device_alloc(ws.unique_ranges, particles);
    karras_device_alloc(ws.block_min_x, bounds_blocks);
    karras_device_alloc(ws.block_min_y, bounds_blocks);
    karras_device_alloc(ws.block_min_z, bounds_blocks);
    karras_device_alloc(ws.block_max_x, bounds_blocks);
    karras_device_alloc(ws.block_max_y, bounds_blocks);
    karras_device_alloc(ws.block_max_z, bounds_blocks);
    karras_device_alloc(ws.binary_nodes, 2 * particles - 1);
    karras_device_alloc(ws.edge_counts, hc.edge_capacity);
    karras_device_alloc(ws.edge_offsets, hc.edge_capacity);
    karras_device_alloc(ws.edge_children, hc.edge_capacity);
    karras_device_alloc(ws.incoming_edge, 2 * particles - 1);
    karras_device_alloc(ws.level_block_counts,
        static_cast<size_t>(fmm::FMM_MAX_LEVELS) * hc.level_block_capacity);
    karras_device_alloc(ws.level_block_offsets,
        static_cast<size_t>(fmm::FMM_MAX_LEVELS) * hc.level_block_capacity);

    uint64_t sparse_total = 0;
    for (int level = 0; level < fmm::FMM_MAX_LEVELS; ++level) {
        ws.sparse_offset[level] = sparse_total;
        uint32_t capacity = 0;
        if (level <= hc.tree_depth &&
            level > (tree_dimension == 2 ? persistent_dense_cutoff<2>()
                                         : persistent_dense_cutoff<3>())) {
            // No level can contain more nodes than the shared node arena.
            // Thin image trees may have many sources in each occupied cell;
            // sizing every sparse level from particle capacity wastes memory.
            capacity = ::cudadem::shear_fmm_capacity::sparse_slots(
                particles, hc.node_capacity, level, tree_dimension);
            if (sparse_total > SIZE_MAX - capacity)
                throw std::overflow_error("sparse lookup capacity overflow");
            sparse_total += capacity;
        }
        ws.sparse_capacity[level] = capacity;
    }
    hc.sparse_capacity_total = sparse_total;
    karras_device_alloc(ws.sparse_keys, sparse_total);
    karras_device_alloc(ws.sparse_values, sparse_total);

    int dense_total = 0;
    for (int level = 0; level < fmm::FMM_MAX_DENSE_LEVELS; ++level) {
        uint64_t capacity = 0;
        if (level <= hc.tree_depth) {
            if (tree_dimension == 3 && level <= 7)
                capacity = 1ull << (3 * level);
            if (tree_dimension == 2)
                capacity = static_cast<uint64_t>(1) << (2 * level);
        }
        if (capacity > static_cast<uint64_t>(INT_MAX - dense_total))
            throw std::overflow_error("dense-map capacity exceeds INT_MAX");
        hc.dd.level_dense_capacity[level] = static_cast<int>(capacity);
        dense_total += static_cast<int>(capacity);
    }
    hc.dense_capacity_total = dense_total;
    karras_device_alloc(ws.dense_storage, dense_total);
    karras_device_alloc(ws.dense_linear_storage, dense_total);
    int offset = 0;
    for (int level = 0; level < fmm::FMM_MAX_DENSE_LEVELS; ++level) {
        hc.dd.level_dense_map[level] = ws.dense_storage + offset;
        ws.dense_linear_map[level] = ws.dense_linear_storage + offset;
        dd.level_dense_map[level] = hc.dd.level_dense_map[level];
        dd.level_dense_capacity[level] = hc.dd.level_dense_capacity[level];
        offset += hc.dd.level_dense_capacity[level];
    }

    cub::DeviceRadixSort::SortPairs(
        nullptr, runtime.radix_sort_temp_bytes,
        ws.morton_unsorted, ws.morton_sorted,
        ws.index_unsorted, hc.dd.sorted_to_original,
        particles, 0, 64);
    CUDA_CHECK(cudaMalloc(&runtime.radix_sort_temp, runtime.radix_sort_temp_bytes));
    size_t edge_bytes = 0;
    size_t target_bytes = 0;
    size_t level_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, edge_bytes,
        ws.edge_counts, ws.edge_offsets, hc.edge_capacity);
    cub::DeviceScan::ExclusiveSum(
        nullptr, target_bytes,
        ws.index_unsorted, ws.unique_offsets, particles);
    cub::DeviceScan::ExclusiveSum(
        nullptr, level_bytes,
        ws.level_block_counts, ws.level_block_offsets,
        hc.level_block_capacity);
    runtime.edge_scan_temp_bytes = std::max(
        karras_unique_selection_scratch_bytes(particles),
        std::max(edge_bytes, std::max(target_bytes, level_bytes)));
    CUDA_CHECK(cudaMalloc(&runtime.edge_scan_temp, runtime.edge_scan_temp_bytes));
}

void shutdown_karras_tree_workspace(PersistentTreeRuntime& runtime) {
    auto& ws = runtime.host_context.workspace;
    karras_device_free(ws.morton_unsorted);
    karras_device_free(ws.morton_sorted);
    karras_device_free(ws.index_unsorted);
    karras_device_free(ws.unique_offsets);
    karras_device_free(ws.unique_ranges);
    karras_device_free(ws.block_min_x); karras_device_free(ws.block_min_y);
    karras_device_free(ws.block_min_z); karras_device_free(ws.block_max_x);
    karras_device_free(ws.block_max_y); karras_device_free(ws.block_max_z);
    karras_device_free(ws.binary_nodes);
    karras_device_free(ws.edge_counts); karras_device_free(ws.edge_offsets);
    karras_device_free(ws.edge_children); karras_device_free(ws.incoming_edge);
    karras_device_free(ws.level_block_counts); karras_device_free(ws.level_block_offsets);
    karras_device_free(ws.sparse_keys); karras_device_free(ws.sparse_values);
    karras_device_free(ws.dense_storage); karras_device_free(ws.dense_linear_storage);
    if (runtime.radix_sort_temp) CUDA_CHECK(cudaFree(runtime.radix_sort_temp));
    if (runtime.edge_scan_temp) CUDA_CHECK(cudaFree(runtime.edge_scan_temp));
    runtime.radix_sort_temp = nullptr;
    runtime.edge_scan_temp = nullptr;
    runtime.radix_sort_temp_bytes = 0;
    runtime.edge_scan_temp_bytes = 0;
}

template<int D>
void capture_karras_tree_stages(PersistentTreeRuntime& runtime,
                                cudaStream_t stream) {
    PersistentTreeContext* ctx = runtime.device_context;
    const auto& hc = runtime.host_context;
    const int particle_blocks = (hc.particle_capacity + KARRAS_BLOCK - 1) / KARRAS_BLOCK;
    const int node_blocks = (hc.node_capacity + KARRAS_BLOCK - 1) / KARRAS_BLOCK;
    const int edge_blocks = (hc.edge_capacity + KARRAS_BLOCK - 1) / KARRAS_BLOCK;
    const int init_blocks = std::max(node_blocks, std::max(edge_blocks, particle_blocks * 2));
    const size_t bounds_shared = 6 * KARRAS_BLOCK * sizeof(double);

    karras_reset_kernel<<<init_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 0);
    karras_bounds_block_kernel<<<particle_blocks, KARRAS_BLOCK,
                                 bounds_shared, stream>>>(ctx);
    karras_bounds_final_kernel<D><<<1, KARRAS_BLOCK, 0, stream>>>(
        ctx, particle_blocks);
    if (hc.use_presorted_morton) {
        karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 1);
        karras_import_presorted_kernel<<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    } else {
        karras_morton_kernel<D><<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
        karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 1);
        const int begin_bit = TreeTraits<D>::MORTON_BITS - D * hc.tree_depth;
        cub::DeviceRadixSort::SortPairs(
            runtime.radix_sort_temp, runtime.radix_sort_temp_bytes,
            hc.workspace.morton_unsorted, hc.workspace.morton_sorted,
            hc.workspace.index_unsorted, hc.dd.sorted_to_original,
            hc.particle_capacity, begin_bit, TreeTraits<D>::MORTON_BITS, stream);
    }
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 9);
    CUDA_CHECK(cub::DeviceSelect::If(
        runtime.edge_scan_temp, runtime.edge_scan_temp_bytes,
        thrust::counting_iterator<int>(0), hc.workspace.unique_offsets,
        &ctx->metadata.num_unique_cells, hc.particle_capacity,
        KarrasUniqueRunStart{ctx}, stream));
    karras_compact_unique_cells_kernel<<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 2);

    karras_build_radix_and_edges_kernel<D><<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_single_unique_cell_kernel<D><<<1, 1, 0, stream>>>(ctx);
    cub::DeviceScan::ExclusiveSum(
        runtime.edge_scan_temp, runtime.edge_scan_temp_bytes,
        hc.workspace.edge_counts, hc.workspace.edge_offsets,
        hc.edge_capacity, stream);
    karras_finalize_count_and_root_kernel<D><<<1, 1, 0, stream>>>(ctx);
    karras_children_reset_kernel<<<node_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 3);

    karras_mark_target_particles_kernel<<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    cub::DeviceScan::ExclusiveSum(
        runtime.edge_scan_temp, runtime.edge_scan_temp_bytes,
        hc.workspace.index_unsorted, hc.workspace.unique_offsets,
        hc.particle_capacity, stream);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 4);
    karras_materialize_edges_kernel<D><<<edge_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 5);
    karras_finalize_nodes_kernel<D><<<node_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 6);
    for (int level = 0; level <= hc.tree_depth; ++level) {
        cub::DeviceScan::ExclusiveSum(
            runtime.edge_scan_temp, runtime.edge_scan_temp_bytes,
            hc.workspace.level_block_counts + static_cast<size_t>(level) * node_blocks,
            hc.workspace.level_block_offsets + static_cast<size_t>(level) * node_blocks,
            node_blocks, stream);
    }
    karras_finalize_levels_kernel<<<1, 1, 0, stream>>>(ctx, node_blocks);
    karras_emit_level_list_kernel<<<node_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 7);
    karras_particle_leaf_map_kernel<D><<<particle_blocks, KARRAS_BLOCK, 0, stream>>>(ctx);
    karras_phase_marker_kernel<<<1, 1, 0, stream>>>(ctx, 8);
    karras_timing_end_kernel<<<1, 1, 0, stream>>>(ctx);
}

template void capture_karras_tree_stages<2>(PersistentTreeRuntime&, cudaStream_t);
template void capture_karras_tree_stages<3>(PersistentTreeRuntime&, cudaStream_t);

}  // namespace body_fitted_fmm

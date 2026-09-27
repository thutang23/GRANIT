// =============================================================================
// LBVH (Linear Bounding Volume Hierarchy) Neighbor Search Implementation
// =============================================================================

#include "lbvh_manager.h"
#include "max_support_reduce.cuh"
#include "contact_dedup.cuh"
#include "lbvh_periodic.cuh"
#include "../rt/rt_backend.h"
#include "../neighbor_types.h"
#include "../sph_pair.cuh"
#include "../neighbor_api.h"  // For NeighborSearch contact buffer helpers
#include "../../constant/system_initialize.h"
#include "../../constant/command_line_interface.h"
#include "../../wall/wall.h"
#include <chrono>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <stdexcept>
#include <string>
#include <limits>
#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)
#endif

namespace cudadem {
namespace lbvh {

struct LBVHProfileRange {
  explicit LBVHProfileRange(const char* name) {
#ifdef CUDADEM_ENABLE_PROFILING
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~LBVHProfileRange() {
#ifdef CUDADEM_ENABLE_PROFILING
    nvtxRangePop();
#endif
  }
};

// Preserve the former cub::Max comparison, including ties and NaNs.
struct DisplacementMaximum {
    __host__ __device__ float operator()(float a, float b) const {
        return b > a ? b : a;
    }
};

// =============================================================================
// Morton Code Utilities (for LBVH)
// =============================================================================

__device__ __host__ __forceinline__
uint32_t lbvh_expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// 30-bit morton code for normalized [0,1]³ coordinates
__device__ __host__ __forceinline__
uint32_t lbvh_morton_code(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = lbvh_expand_bits(static_cast<uint32_t>(x));
    uint32_t yy = lbvh_expand_bits(static_cast<uint32_t>(y));
    uint32_t zz = lbvh_expand_bits(static_cast<uint32_t>(z));
    return xx * 4 + yy * 2 + zz;
}

// Count leading zeros for common prefix length
__device__ __forceinline__
int lbvh_common_upper_bits(uint32_t lhs, uint32_t rhs) {
    return __clz(lhs ^ rhs);
}

__device__ __forceinline__
int lbvh_common_upper_bits(unsigned long long int lhs, unsigned long long int rhs) {
    return __clzll(lhs ^ rhs);
}

// =============================================================================
// LBVHSearchImpl Static Members
// =============================================================================

std::unordered_map<int32_t, LBVHGPUState> LBVHSearchImpl::gpu_states_;
std::mutex LBVHSearchImpl::state_mutex_;

NeighborSearchStats LBVHSearchImpl::default_stats_;

LBVHGPUState &LBVHSearchImpl::get_gpu_state() {
  int32_t device;
  CUDA_CHECK(cudaGetDevice(&device));
  return get_gpu_state(device);
}

LBVHGPUState &LBVHSearchImpl::get_gpu_state(int32_t gpu_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto &state = gpu_states_[gpu_id];
  if (state.gpu_id == -1) {
    state.gpu_id = gpu_id;
  }
  return state;
}

// =============================================================================
// Kernel Implementations
// =============================================================================

// =============================================================================
// Verlet Buffer Kernels
// =============================================================================

__global__ void lbvh_verlet_displacement_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const double* __restrict__ last_pos_x, const double* __restrict__ last_pos_y, const double* __restrict__ last_pos_z,
    int32_t* __restrict__ max_disp_sq_scaled, int32_t num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float my_disp_sq = 0.0f;
    if (idx < num_particles) {
        float dx = static_cast<float>(pos_x[idx] - last_pos_x[idx]);
        float dy = static_cast<float>(pos_y[idx] - last_pos_y[idx]);
        float dz = static_cast<float>(pos_z[idx] - last_pos_z[idx]);
        my_disp_sq = dx*dx + dy*dy + dz*dz;
    }
    typedef cub::BlockReduce<float, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    float block_max = BlockReduceT(temp_storage).Reduce(my_disp_sq, DisplacementMaximum{});
    if (threadIdx.x == 0) {
        int32_t int_val = __float_as_int(block_max);
        atomicMax(max_disp_sq_scaled, int_val);
    }
}

__global__ void lbvh_store_last_positions_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    double* __restrict__ last_pos_x, double* __restrict__ last_pos_y, double* __restrict__ last_pos_z,
    int32_t num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    last_pos_x[idx] = pos_x[idx];
    last_pos_y[idx] = pos_y[idx];
    last_pos_z[idx] = pos_z[idx];
}

// Validate against the CURRENT query coordinates, after drift and wrapping.
// An end-of-step asynchronous monitor describes the previous force position and
// cannot certify a list for the next force evaluation. Also bind the cache to
// particle identity and interaction support: count equality is not sufficient
// after a halo exchange or a same-size particle-array permutation.
__global__ void lbvh_current_cache_displacement_kernel(
    const double* pos_x, const double* pos_y, const double* pos_z,
    const int32_t* global_id, const int32_t* clump_id, const float* support_extent,
    const LBVHCacheParticle* reference, int32_t* max_disp_bits, int32_t count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float displacement_sq = 0.0f;
    if (i < count) {
        // Match the float coordinates used by traversal, with an upward-rounded
        // bound on their squared displacement. Raw wrapped displacement makes
        // a periodic wrap invalidate cached image offsets conservatively.
        const LBVHCacheParticle old = reference[i];
        // Keep references private to this device's list. Particle last_pos
        // fields can be overwritten by a halo refresh from another device.
        const double dx = double(float(pos_x[i])) - double(old.x);
        const double dy = double(float(pos_y[i])) - double(old.y);
        const double dz = double(float(pos_z[i])) - double(old.z);
        displacement_sq = __double2float_ru(dx*dx + dy*dy + dz*dz);
        if (!isfinite(displacement_sq) ||
            old.global_id != (global_id ? global_id[i] : i) ||
            old.clump_id != (clump_id ? clump_id[i] : 0) ||
            old.support_extent != support_extent[i]) {
            displacement_sq = CUDART_INF_F;
        }
    }
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage storage;
    const float block_max = Reduce(storage).Reduce(displacement_sq, DisplacementMaximum{});
    if (threadIdx.x == 0) atomicMax(max_disp_bits, __float_as_int(block_max));
}

__global__ void lbvh_capture_cache_particles_kernel(
    const double* pos_x, const double* pos_y, const double* pos_z,
    const int32_t* global_id, const int32_t* clump_id, const float* support_extent,
    LBVHCacheParticle* reference, int32_t count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) reference[i] = {
        global_id ? global_id[i] : i, clump_id ? clump_id[i] : 0, support_extent[i],
        float(pos_x[i]), float(pos_y[i]), float(pos_z[i])};
}

#include "lbvh_leaf_bounds.cuh"

// Write per-particle squared displacement from float snapshot into scratch
__global__ void lbvh_displacement_check_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ snap_x,
    const float*  __restrict__ snap_y,
    const float*  __restrict__ snap_z,
    float*        __restrict__ disp_sq_buf,
    int32_t num_particles)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float dx = static_cast<float>(pos_x[idx]) - snap_x[idx];
    float dy = static_cast<float>(pos_y[idx]) - snap_y[idx];
    float dz = static_cast<float>(pos_z[idx]) - snap_z[idx];
    disp_sq_buf[idx] = dx*dx + dy*dy + dz*dz;
}

// Capture current positions (double) into float snapshot arrays
__global__ void lbvh_copy_snapshot_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    float*        __restrict__ snap_x,
    float*        __restrict__ snap_y,
    float*        __restrict__ snap_z,
    int32_t num_particles)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    snap_x[idx] = static_cast<float>(pos_x[idx]);
    snap_y[idx] = static_cast<float>(pos_y[idx]);
    snap_z[idx] = static_cast<float>(pos_z[idx]);
}

// 2. Compute 30-bit morton codes based on global AABB
__global__ void
lbvh_compute_morton_codes_kernel(const LBVHAABB *__restrict__ leaf_aabbs,
                                 uint32_t *__restrict__ morton_codes,
                                 int32_t num_particles, int32_t internal_offset,
                                 const LBVHAABB* __restrict__ domain_ptr) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  const LBVHAABB whole_domain = *domain_ptr;
  const LBVHAABB &box = leaf_aabbs[internal_offset + idx];
  float cx = (box.upper.x + box.lower.x) * 0.5f;
  float cy = (box.upper.y + box.lower.y) * 0.5f;
  float cz = (box.upper.z + box.lower.z) * 0.5f;

  // Normalize
  float nx = (cx - whole_domain.lower.x) /
             (whole_domain.upper.x - whole_domain.lower.x);
  float ny = (cy - whole_domain.lower.y) /
             (whole_domain.upper.y - whole_domain.lower.y);
  float nz = (cz - whole_domain.lower.z) /
             (whole_domain.upper.z - whole_domain.lower.z);

  morton_codes[idx] = lbvh_morton_code(nx, ny, nz);
}

// 3. Leaf node init
__global__ void
lbvh_construct_leaves_kernel(const uint32_t *__restrict__ sorted_indices,
                             LBVHNode *__restrict__ nodes,
                             int32_t num_particles, int32_t internal_offset) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  LBVHNode n;
  n.parent_idx = 0xFFFFFFFF;
  n.left_idx = 0xFFFFFFFF;
  n.right_idx = 0xFFFFFFFF;
  n.object_idx = sorted_indices[idx];

  nodes[internal_offset + idx] = n;
}

// 4. Karras 2012 determine_range + find_split
template <typename UInt>
__device__ uint2 lbvh_determine_range(const UInt *node_code, int32_t num_leaves,
                                      int32_t idx) {
  if (idx == 0)
    return make_uint2(0, num_leaves - 1);

  const UInt self_code = node_code[idx];
  const int L_delta = lbvh_common_upper_bits(self_code, node_code[idx - 1]);
  const int R_delta = lbvh_common_upper_bits(self_code, node_code[idx + 1]);
  const int d = (R_delta > L_delta) ? 1 : -1;

  const int delta_min = min(L_delta, R_delta);
  int l_max = 2;
  int delta = -1;
  int i_tmp = idx + d * l_max;
  if (0 <= i_tmp && i_tmp < num_leaves)
    delta = lbvh_common_upper_bits(self_code, node_code[i_tmp]);

  while (delta > delta_min) {
    l_max <<= 1;
    i_tmp = idx + d * l_max;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves)
      delta = lbvh_common_upper_bits(self_code, node_code[i_tmp]);
  }

  int l = 0;
  int t = l_max >> 1;
  while (t > 0) {
    i_tmp = idx + (l + t) * d;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves)
      delta = lbvh_common_upper_bits(self_code, node_code[i_tmp]);
    if (delta > delta_min)
      l += t;
    t >>= 1;
  }

  uint32_t jdx = idx + l * d;
  if (d < 0) {
    uint32_t tmp = idx;
    idx = jdx;
    jdx = tmp;
  }
  return make_uint2(idx, jdx);
}

template <typename UInt>
__device__ int32_t lbvh_find_split(const UInt *node_code, int32_t num_leaves,
                                   int32_t first, int32_t last) {
  const UInt first_code = node_code[first];
  const UInt last_code = node_code[last];
  if (first_code == last_code)
    return (first + last) >> 1;

  const int delta_node = lbvh_common_upper_bits(first_code, last_code);
  int split = first;
  int stride = last - first;

  do {
    stride = (stride + 1) >> 1;
    int middle = split + stride;
    if (middle < last) {
      int delta = lbvh_common_upper_bits(first_code, node_code[middle]);
      if (delta > delta_node)
        split = middle;
    }
  } while (stride > 1);

  return split;
}

template <typename UInt>
__global__ void lbvh_construct_internal_nodes_kernel(
    const UInt *__restrict__ morton_codes, LBVHNode *__restrict__ nodes,
    LBVHAABB *__restrict__ aabbs, int32_t num_particles) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t num_internal = num_particles - 1;
  if (idx >= num_internal)
    return;

  nodes[idx].object_idx = 0xFFFFFFFF; // Internal node

  uint2 ij = lbvh_determine_range(morton_codes, num_particles, idx);
  int gamma = lbvh_find_split(morton_codes, num_particles, ij.x, ij.y);

  uint32_t left_child = gamma;
  uint32_t right_child = gamma + 1;

  if (min(ij.x, ij.y) == gamma)
    left_child += num_internal;
  if (max(ij.x, ij.y) == gamma + 1)
    right_child += num_internal;

  nodes[idx].left_idx = left_child;
  nodes[idx].right_idx = right_child;

  nodes[left_child].parent_idx = idx;
  nodes[right_child].parent_idx = idx;
}

// 5. Bottom up AABB
// Karras 2012 §4 optimization: use shared-memory atomics for first-level
// leaf→parent propagation when sibling leaves land in the same thread block.
// Both siblings CAS on the SAME shared flag (indexed by min thread index) so
// exactly one thread merges and the other terminates.
__global__ void lbvh_propagate_aabbs_kernel(LBVHNode *__restrict__ nodes,
                                             LBVHAABB *__restrict__ aabbs,
                                             int32_t *__restrict__ flags,
                                             int32_t num_internal_nodes,
                                             int32_t num_particles) {
  // 1 KB per block — one flag per potential sibling pair in the block
  __shared__ int32_t shared_flags[256];

  // All threads participate: inactive threads' slots must be zeroed so that
  // active siblings never CAS against garbage.
  shared_flags[threadIdx.x] = 0;
  __syncthreads();

  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  uint32_t leaf_idx = num_internal_nodes + idx;
  uint32_t parent = nodes[leaf_idx].parent_idx;

  bool first_iteration = true;

  while (parent != 0xFFFFFFFF) {
    // ---------------------------------------------------------------
    //  First-level (leaf → parent) shared-memory fast path.
    //  Condition: both children are leaves AND sibling is in this block.
    // ---------------------------------------------------------------
    if (first_iteration) {
      first_iteration = false;

      uint32_t left = nodes[parent].left_idx;
      uint32_t right = nodes[parent].right_idx;
      bool left_is_leaf = (left >= num_internal_nodes);
      bool right_is_leaf = (right >= num_internal_nodes);

      if (left_is_leaf && right_is_leaf) {
        uint32_t sibling_leaf = (left == leaf_idx) ? right : left;
        int32_t sibling_idx = sibling_leaf - num_internal_nodes;
        int32_t block_start = blockIdx.x * blockDim.x;

        if (sibling_idx >= block_start &&
            sibling_idx < block_start + blockDim.x &&
            sibling_idx < num_particles) {
          int32_t sibling_thread = sibling_idx - block_start;

          // Both siblings use the SAME shared flag (min of the two
          // thread indices), avoiding the bug where each CAS'd on the
          // other's flag (different addresses → both return, no merge).
          int32_t flag_idx = min(threadIdx.x, sibling_thread);
          int old = atomicCAS(&shared_flags[flag_idx], 0, 1);
          if (old == 0)
            return; // First sibling terminates — the other will merge

          // Second sibling merges
          aabbs[parent] = lbvh_merge(aabbs[left], aabbs[right]);
          // The next parent readiness flag is global: another block may
          // consume this box. Publish it device-wide before signaling.
          __threadfence();

          parent = nodes[parent].parent_idx;
          continue;
        }
      }
    }

    // ---------------------------------------------------------------
    //  Global-memory path (original behaviour).
    //  Used for: first-level when siblings are in different blocks,
    //            and all higher levels.
    // ---------------------------------------------------------------
    int old = atomicCAS(&flags[parent], 0, 1);
    if (old == 0)
      return; // First thread terminates

    // Second thread merges
    uint32_t lidx = nodes[parent].left_idx;
    uint32_t ridx = nodes[parent].right_idx;
    aabbs[parent] = lbvh_merge(aabbs[lidx], aabbs[ridx]);

    __threadfence();

    parent = nodes[parent].parent_idx;
  }
}

// 6. Broad + Narrow query
__global__ void lbvh_find_contacts_kernel(
    const double *__restrict__ pos_x, const double *__restrict__ pos_y,
    const double *__restrict__ pos_z, const float *__restrict__ radius,
    const LBVHNode *__restrict__ nodes, const LBVHAABB *__restrict__ aabbs,
    ContactPair *__restrict__ contacts, int32_t *__restrict__ num_contacts,
    const int32_t *__restrict__ clump_id,
    const uint32_t* __restrict__ sorted_indices,
    int32_t max_contacts, float contact_threshold, int32_t num_local, int32_t num_particles) {
  int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= num_particles)
    return;

  // Nearby queries follow nearby tree paths. Scan every Morton rank and
  // explicitly exclude ghosts: the first num_local ranks are not necessarily
  // owned particles. Canonical particle/history indices remain unchanged.
  uint32_t i = sorted_indices[thread_idx];
  if (i >= static_cast<uint32_t>(num_local)) return;

  float px = static_cast<float>(pos_x[i]);
  float py = static_cast<float>(pos_y[i]);
  float pz = static_cast<float>(pos_z[i]);
  float r_i = radius[i];
  // Hoist clump membership for particle i (constant for this thread)
  const int32_t ci_clump = (clump_id != nullptr) ? clump_id[i] : -2;
  // Match CLL neighbor-list semantics:
  // threshold = -skin (Verlet) or 0 (exact contact).
  // Query AABB must include this margin to avoid broad-phase false negatives.
  float query_r = r_i - contact_threshold;

  LBVHAABB query_box;
  query_box.lower = make_float4(px - query_r, py - query_r, pz - query_r, 0);
  query_box.upper = make_float4(px + query_r, py + query_r, pz + query_r, 0);

  // Stack based traversal
  // Note: Stack size must accommodate maximum tree depth. For a balanced LBVH with N particles,
  // depth is O(log N). However, degenerate Morton codes can create unbalanced trees.
  // Using 128 to handle worst-case scenarios (original 64 was insufficient for N=823).
  constexpr int STACK_SIZE = 128;
  uint32_t stack[STACK_SIZE];
  uint32_t *stack_ptr = stack;
  uint32_t *stack_end = stack + STACK_SIZE;
  *stack_ptr++ = 0; // Root is ALWAYS 0

  while (stack_ptr > stack) {
    uint32_t node = *--stack_ptr;
    uint32_t L_idx = nodes[node].left_idx;
    uint32_t R_idx = nodes[node].right_idx;
    
    // Skip invalid child indices (corrupted tree)
    bool L_valid = (L_idx != 0xFFFFFFFF);
    bool R_valid = (R_idx != 0xFFFFFFFF);
    
    if (!L_valid && !R_valid) continue;

    bool L_overlap = L_valid && lbvh_intersects(query_box, aabbs[L_idx]);
    bool R_overlap = R_valid && lbvh_intersects(query_box, aabbs[R_idx]);

    if (L_overlap) {
      uint32_t obj_j = nodes[L_idx].object_idx;
      if (obj_j != 0xFFFFFFFF) { // Leaf
        if (i < obj_j) {
          // Skip intra-clump pairs
          if (!(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
            float j_px = static_cast<float>(pos_x[obj_j]);
            float j_py = static_cast<float>(pos_y[obj_j]);
            float j_pz = static_cast<float>(pos_z[obj_j]);
            float dx = j_px - px;
            float dy = j_py - py;
            float dz = j_pz - pz;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float range_limit = r_i + radius[obj_j] - contact_threshold;

            if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
              float dist = sqrtf(dist_sq);
              float overlap = r_i + radius[obj_j] - dist;
              if (overlap > contact_threshold) {
                int32_t write_idx = atomicAdd(num_contacts, 1);
                if (write_idx < max_contacts) {
                  contacts[write_idx] = ContactPair(i, obj_j);
                }
              }
            }
          }
        }
      } else { // Internal
        if (stack_ptr < stack_end) *stack_ptr++ = L_idx;
      }
    }

    if (R_overlap) {
      uint32_t obj_j = nodes[R_idx].object_idx;
      if (obj_j != 0xFFFFFFFF) { // Leaf
        if (i < obj_j) {
          // Skip intra-clump pairs
          if (!(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
            float j_px = static_cast<float>(pos_x[obj_j]);
            float j_py = static_cast<float>(pos_y[obj_j]);
            float j_pz = static_cast<float>(pos_z[obj_j]);
            float dx = j_px - px;
            float dy = j_py - py;
            float dz = j_pz - pz;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float range_limit = r_i + radius[obj_j] - contact_threshold;

            if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
              float dist = sqrtf(dist_sq);
              float overlap = r_i + radius[obj_j] - dist;
              if (overlap > contact_threshold) {
                int32_t write_idx = atomicAdd(num_contacts, 1);
                if (write_idx < max_contacts) {
                  contacts[write_idx] = ContactPair(i, obj_j);
                }
              }
            }
          }
        }
      } else { // Internal
        if (stack_ptr < stack_end) *stack_ptr++ = R_idx;
      }
    }
  }
}

// =============================================================================
// Optimized Kernel: Block-Level Contact Reduction
// =============================================================================
// Eliminates global atomic contention by:
// 1. Each thread stores contacts locally
// 2. Block-level reduction aggregates counts
// 3. Single atomic reserves space for entire block
// 4. Batch write to global memory
// =============================================================================

__global__ void lbvh_find_contacts_optimized_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ radius,
    const LBVHNode* __restrict__ nodes,
    const LBVHAABB* __restrict__ aabbs,
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts,
    const int32_t* __restrict__ clump_id,
    int32_t max_contacts,
    float contact_threshold,
    int32_t num_local,
    int64_t* __restrict__ debug_contact_overflows,
    int64_t* __restrict__ debug_stack_overflows)
{
    // Thread-local contact storage
    constexpr int MAX_CONTACTS_PER_THREAD = 32;
    ContactPair my_contacts[MAX_CONTACTS_PER_THREAD];
    int my_count = 0;
    
    // Process particle
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < num_local) {
        // Local particles own queries directly; ghosts remain tree leaves only.
        uint32_t i = static_cast<uint32_t>(thread_idx);
        
        float px = static_cast<float>(pos_x[i]);
        float py = static_cast<float>(pos_y[i]);
        float pz = static_cast<float>(pos_z[i]);
        float r_i = radius[i];
        const int32_t ci_clump = (clump_id != nullptr) ? clump_id[i] : -2;
        float query_r = r_i - contact_threshold;
        
        LBVHAABB query_box;
        query_box.lower = make_float4(px - query_r, py - query_r, pz - query_r, 0);
        query_box.upper = make_float4(px + query_r, py + query_r, pz + query_r, 0);
        
        // Stack-based traversal (same as original)
        constexpr int STACK_SIZE = 128;
        uint32_t stack[STACK_SIZE];
        uint32_t* stack_ptr = stack;
        uint32_t* stack_end = stack + STACK_SIZE;
        *stack_ptr++ = 0;
        
        while (stack_ptr > stack) {
            uint32_t node = *--stack_ptr;
            uint32_t L_idx = nodes[node].left_idx;
            uint32_t R_idx = nodes[node].right_idx;
            
            bool L_valid = (L_idx != 0xFFFFFFFF);
            bool R_valid = (R_idx != 0xFFFFFFFF);
            
            if (!L_valid && !R_valid) continue;
            
            bool L_overlap = L_valid && lbvh_intersects(query_box, aabbs[L_idx]);
            bool R_overlap = R_valid && lbvh_intersects(query_box, aabbs[R_idx]);
            
            // Process left child
            if (L_overlap) {
                uint32_t obj_j = nodes[L_idx].object_idx;
                if (obj_j != 0xFFFFFFFF) { // Leaf
                    if (i < obj_j && !(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
                        float j_px = static_cast<float>(pos_x[obj_j]);
                        float j_py = static_cast<float>(pos_y[obj_j]);
                        float j_pz = static_cast<float>(pos_z[obj_j]);
                        float dx = j_px - px;
                        float dy = j_py - py;
                        float dz = j_pz - pz;
                        float dist_sq = dx*dx + dy*dy + dz*dz;
                        float range_limit = r_i + radius[obj_j] - contact_threshold;
                        
                        if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
                            float dist = sqrtf(dist_sq);
                            float overlap = r_i + radius[obj_j] - dist;
                            if (overlap > contact_threshold) {
                                // Store locally instead of atomic
                                if (my_count < MAX_CONTACTS_PER_THREAD) {
                                    my_contacts[my_count++] = ContactPair(i, obj_j);
                                } else {
                                    // Overflow: drop the contact to avoid
                                    // write-index race with block-reduce path.
                                    // Track for diagnostics so we know if
                                    // MAX_CONTACTS_PER_THREAD needs raising.
                                    if (debug_contact_overflows) {
                                        atomicAdd((unsigned long long*)debug_contact_overflows, 1);
                                    }
                                }
                            }
                        }
                    }
                } else { // Internal
                    if (stack_ptr < stack_end) {
                        *stack_ptr++ = L_idx;
                    } else {
                        // Stack overflow - track for diagnostics
                        if (debug_stack_overflows) {
                            atomicAdd((unsigned long long*)debug_stack_overflows, 1);
                        }
                    }
                }
            }
            
            // Process right child (same pattern)
            if (R_overlap) {
                uint32_t obj_j = nodes[R_idx].object_idx;
                if (obj_j != 0xFFFFFFFF) { // Leaf
                    if (i < obj_j && !(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
                        float j_px = static_cast<float>(pos_x[obj_j]);
                        float j_py = static_cast<float>(pos_y[obj_j]);
                        float j_pz = static_cast<float>(pos_z[obj_j]);
                        float dx = j_px - px;
                        float dy = j_py - py;
                        float dz = j_pz - pz;
                        float dist_sq = dx*dx + dy*dy + dz*dz;
                        float range_limit = r_i + radius[obj_j] - contact_threshold;
                        
                        if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
                            float dist = sqrtf(dist_sq);
                            float overlap = r_i + radius[obj_j] - dist;
                            if (overlap > contact_threshold) {
                                if (my_count < MAX_CONTACTS_PER_THREAD) {
                                    my_contacts[my_count++] = ContactPair(i, obj_j);
                                } else {
                                    if (debug_contact_overflows) {
                                        atomicAdd((unsigned long long*)debug_contact_overflows, 1);
                                    }
                                    int32_t write_idx = atomicAdd(num_contacts, 1);
                                    if (write_idx < max_contacts) {
                                        contacts[write_idx] = ContactPair(i, obj_j);
                                    }
                                }
                            }
                        }
                    }
                } else { // Internal
                    if (stack_ptr < stack_end) {
                        *stack_ptr++ = R_idx;
                    } else {
                        // Stack overflow - track for diagnostics
                        if (debug_stack_overflows) {
                            atomicAdd((unsigned long long*)debug_stack_overflows, 1);
                        }
                    }
                }
            }
        }
    }
    
    // Block-level reduction
    // Note: Overflow contacts already incremented num_contacts via atomicAdd.
    // Block reduction only counts thread-local contacts (my_count).
    // Total contacts = overflow_count + sum(my_count), no double-counting.
    __syncthreads();
    
    typedef cub::BlockReduce<int, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    int total_contacts = BlockReduceT(temp_storage).Sum(my_count);
    
    // Single thread reserves space for entire block
    __shared__ int block_contact_base;
    if (threadIdx.x == 0) {
        block_contact_base = atomicAdd(num_contacts, total_contacts);
    }
    __syncthreads();
    
    // Compute each thread's offset using exclusive scan
    typedef cub::BlockScan<int, 256> BlockScanT;
    __shared__ typename BlockScanT::TempStorage scan_storage;
    
    int my_offset;
    BlockScanT(scan_storage).ExclusiveSum(my_count, my_offset);
    
    // Batch write contacts to global memory
    for (int i = 0; i < my_count; i++) {
        int global_idx = block_contact_base + my_offset + i;
        if (global_idx < max_contacts) {
            contacts[global_idx] = my_contacts[i];
        }
    }
}

// NOTE: lbvh_find_contacts_optimized_v2_kernel (shared-memory stack variant) removed.
// The kernel used __shared__ uint32_t shared_stack[256][48] = 49KB + CUB overhead ≈ 51KB,
// exceeding the default 48KB per-block shared memory limit on sm_70 and causing nvlink error.
// Use the original lbvh_find_contacts_kernel with local-memory stack (128 entries).

// Reduction op for merging AABBs during domain sizing
struct AABBMergeOp {
  __device__ LBVHAABB operator()(const LBVHAABB &a, const LBVHAABB &b) const {
    return lbvh_merge(a, b);
  }
};

// =============================================================================
// Periodic-Aware Contact Finding Kernel
// =============================================================================
// Handles periodic boundaries by generating shifted queries for particles
// near periodic boundaries. Each particle may perform multiple BVH traversals.

__global__ void lbvh_find_contacts_periodic_kernel(
    const double *__restrict__ pos_x, const double *__restrict__ pos_y,
    const double *__restrict__ pos_z, const float *__restrict__ radius,
    const LBVHNode *__restrict__ nodes, const LBVHAABB *__restrict__ aabbs,
    ContactPair *__restrict__ contacts, int32_t *__restrict__ num_contacts,
    const int32_t *__restrict__ clump_id,
    int32_t max_contacts, float contact_threshold, int32_t num_local,
    PeriodicBoundaryConfig pbc,
    float max_radius, bool shear_enabled, ShearPhase shear_phase
) {
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_local) return;

    // Local particles own queries directly; ghosts remain tree leaves only.
    uint32_t i = static_cast<uint32_t>(thread_idx);

    float px = static_cast<float>(pos_x[i]);
    float py = static_cast<float>(pos_y[i]);
    float pz = static_cast<float>(pos_z[i]);
    float r_i = radius[i];

    const int32_t ci_clump = (clump_id != nullptr) ? clump_id[i] : -2;
    float query_r = r_i - contact_threshold;

    // Generate periodic images to query
    PeriodicImageShift images[27];
    int num_images = shear_enabled
        ? generate_shear_images(pbc, shear_phase, images)
        // Image selection must include the same Verlet shell as traversal;
        // otherwise a valid cache can miss a later cross-boundary contact.
        : generate_periodic_images(px, py, pz, query_r, max_radius, pbc, images);

    // Traverse BVH for each image
    for (int img = 0; img < num_images; img++) {
        float shift_x = images[img].shift_x;
        float shift_y = images[img].shift_y;
        float shift_z = images[img].shift_z;
        int8_t img_x = images[img].image_x;
        int8_t img_y = images[img].image_y;
        int8_t img_z = images[img].image_z;

        // Build shifted query AABB
        float qpx = px + shift_x;
        float qpy = py + shift_y;
        float qpz = pz + shift_z;

        LBVHAABB query_box;
        query_box.lower = make_float4(qpx - query_r, qpy - query_r, qpz - query_r, 0);
        query_box.upper = make_float4(qpx + query_r, qpy + query_r, qpz + query_r, 0);

        // Stack-based traversal
        constexpr int STACK_SIZE = 128;
        uint32_t stack[STACK_SIZE];
        uint32_t *stack_ptr = stack;
        uint32_t *stack_end = stack + STACK_SIZE;
        *stack_ptr++ = 0;  // Root is ALWAYS 0

        while (stack_ptr > stack) {
            uint32_t node = *--stack_ptr;
            uint32_t L_idx = nodes[node].left_idx;
            uint32_t R_idx = nodes[node].right_idx;

            bool L_valid = (L_idx != 0xFFFFFFFF);
            bool R_valid = (R_idx != 0xFFFFFFFF);

            if (!L_valid && !R_valid) continue;

            bool L_overlap = L_valid && lbvh_intersects(query_box, aabbs[L_idx]);
            bool R_overlap = R_valid && lbvh_intersects(query_box, aabbs[R_idx]);

            if (L_overlap) {
                uint32_t obj_j = nodes[L_idx].object_idx;
                if (obj_j != 0xFFFFFFFF) {  // Leaf
                    if (i < obj_j) {
                        if (!(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
                            float j_px = static_cast<float>(pos_x[obj_j]);
                            float j_py = static_cast<float>(pos_y[obj_j]);
                            float j_pz = static_cast<float>(pos_z[obj_j]);

                            // Distance from shifted query to j
                            float dx = j_px - qpx;
                            float dy = j_py - qpy;
                            float dz = j_pz - qpz;

                            // Apply minimum image to get true distance
                            if (!shear_enabled && pbc.periodic_x && pbc.box_length_x > 0.0f) {
                                if (dx > pbc.half_length_x) dx -= pbc.box_length_x;
                                else if (dx < -pbc.half_length_x) dx += pbc.box_length_x;
                            }
                            if (!shear_enabled && pbc.periodic_y && pbc.box_length_y > 0.0f) {
                                if (dy > pbc.half_length_y) dy -= pbc.box_length_y;
                                else if (dy < -pbc.half_length_y) dy += pbc.box_length_y;
                            }
                            if (!shear_enabled && pbc.periodic_z && pbc.box_length_z > 0.0f) {
                                if (dz > pbc.half_length_z) dz -= pbc.box_length_z;
                                else if (dz < -pbc.half_length_z) dz += pbc.box_length_z;
                            }

                            float dist_sq = dx * dx + dy * dy + dz * dz;
                            float range_limit = r_i + radius[obj_j] - contact_threshold;

                            if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
                                float dist = sqrtf(dist_sq);
                                float overlap = r_i + radius[obj_j] - dist;
                                if (overlap > contact_threshold) {
                                    int32_t write_idx = atomicAdd(num_contacts, 1);
                                    if (write_idx < max_contacts) {
                                        contacts[write_idx] = ContactPair(i, obj_j, img_x, img_y, img_z);
                                    }
                                }
                            }
                        }
                    }
                } else {  // Internal
                    if (stack_ptr < stack_end) *stack_ptr++ = L_idx;
                }
            }

            if (R_overlap) {
                uint32_t obj_j = nodes[R_idx].object_idx;
                if (obj_j != 0xFFFFFFFF) {  // Leaf
                    if (i < obj_j) {
                        if (!(ci_clump > 0 && clump_id[obj_j] == ci_clump)) {
                            float j_px = static_cast<float>(pos_x[obj_j]);
                            float j_py = static_cast<float>(pos_y[obj_j]);
                            float j_pz = static_cast<float>(pos_z[obj_j]);

                            float dx = j_px - qpx;
                            float dy = j_py - qpy;
                            float dz = j_pz - qpz;

                            if (!shear_enabled && pbc.periodic_x && pbc.box_length_x > 0.0f) {
                                if (dx > pbc.half_length_x) dx -= pbc.box_length_x;
                                else if (dx < -pbc.half_length_x) dx += pbc.box_length_x;
                            }
                            if (!shear_enabled && pbc.periodic_y && pbc.box_length_y > 0.0f) {
                                if (dy > pbc.half_length_y) dy -= pbc.box_length_y;
                                else if (dy < -pbc.half_length_y) dy += pbc.box_length_y;
                            }
                            if (!shear_enabled && pbc.periodic_z && pbc.box_length_z > 0.0f) {
                                if (dz > pbc.half_length_z) dz -= pbc.box_length_z;
                                else if (dz < -pbc.half_length_z) dz += pbc.box_length_z;
                            }

                            float dist_sq = dx * dx + dy * dy + dz * dz;
                            float range_limit = r_i + radius[obj_j] - contact_threshold;

                            if (dist_sq < range_limit * range_limit && dist_sq > 1e-12f) {
                                float dist = sqrtf(dist_sq);
                                float overlap = r_i + radius[obj_j] - dist;
                                if (overlap > contact_threshold) {
                                    int32_t write_idx = atomicAdd(num_contacts, 1);
                                    if (write_idx < max_contacts) {
                                        contacts[write_idx] = ContactPair(i, obj_j, img_x, img_y, img_z);
                                    }
                                }
                            }
                        }
                    }
                } else {  // Internal
                    if (stack_ptr < stack_end) *stack_ptr++ = R_idx;
                }
            }
        }
    }
}

// =============================================================================
// Verlet Cache Kernels (reused from CLL pattern)
// =============================================================================

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
// Clamp Contact Count Kernel — prevents inflated atomic counter from
// propagating downstream after narrow-phase overflow.
// The narrow-phase kernels increment contacts.num_contacts via atomicAdd
// for EVERY detected contact, but only write to contacts.pairs when
// write_idx < max_contacts. This kernel caps the counter to max_contacts
// so downstream reads don't see an inflated count.
// -----------------------------------------------------------------------------
__global__ void clamp_contact_count_kernel(int32_t* num_contacts, int32_t max_contacts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t current = *num_contacts;
        if (current > max_contacts) {
            *num_contacts = max_contacts;
        }
    }
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
    
    cache_pairs_kernel<<<num_blocks, block_size, 0, stream>>>(
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
    
    copy_cached_contact_pairs_kernel<<<num_blocks, block_size, 0, stream>>>(
        pair_i, pair_j, pair_image_x, pair_image_y, pair_image_z,
        num_pairs, contacts, num_contacts
    );
}

// =============================================================================
// Implementation wrapper logic
// =============================================================================

static rt::SearchInput rt_input(const SimulationState& state, const SimConfig& config) {
  const auto& p = state.particles;
  const float skin = config.neighbor.verlet_buffer_enabled && config.neighbor.update_frequency != 1
      ? config.neighbor.verlet_skin : 0.0f;
  rt::SearchInput input{p.pos_x, p.pos_y, p.pos_z, p.support_extent, p.clump_id, p.num_particles, skin};
  if (config.physics == SimulationPhysics::SPH) {
    input.smoothing_length = p.smoothing_length;
    input.kernel_cutoff_factor = config.sph.kernel_cutoff_factor;
  }
  return input;
}

static void select_backend(LBVHGPUState& gpu, const SimulationState& state, const SimConfig& config) {
  std::string reason;
  if (config.neighbor.backend == NeighborBackend::CUDA) reason = "explicit CUDA request";
  else if (config.multigpu.enabled) reason = "multi-GPU neighbor search uses CUDA LBVH";
  else if (config.shear_box.enabled) reason = "Lees-Edwards shear uses CUDA LBVH";
  else if (extract_pbc_config(config).is_enabled()) reason = "periodic boundaries use CUDA LBVH";
  else if (state.particles.num_local != state.particles.num_particles)
    reason = "local/ghost particle layouts use CUDA LBVH";

  NeighborBackend selected = NeighborBackend::CUDA;
  if (reason.empty()) {
    if (!gpu.rt_attempted) {
      const auto start = std::chrono::steady_clock::now();
      gpu.rt_attempted = true;
      try {
        gpu.rt_backend = rt::Backend::create(gpu.rt_unavailable_reason);
      } catch (const std::exception& error) {
        gpu.rt_unavailable_reason = std::string("OptiX initialization failed: ") + error.what();
      }
      gpu.stats.backend_init_time_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    }
    if (gpu.rt_backend) selected = NeighborBackend::RT;
    else reason = gpu.rt_unavailable_reason;
  }
  if (config.neighbor.backend == NeighborBackend::RT && selected != NeighborBackend::RT)
    throw std::runtime_error("neighbor.backend: rt unavailable: " + reason);

  if (selected != gpu.active_backend) {
    gpu.sph_ready = gpu.sph_cache_valid = false;
    gpu.verlet_cache.cache_valid = false;
    gpu.cached_max_disp_sq = 1e30f;
    gpu.async_readback_pending = false;
    gpu.tracked_num_particles = -1; // Force a build of the selected backend.
  }
  gpu.active_backend = selected;
  gpu.backend_reason = reason;
  const std::string message = selected == NeighborBackend::RT
      ? "rt (OptiX hardware traversal)" : "cuda (" + reason + ")";
  if (message != gpu.backend_log) {
    std::cout << "[neighbor] GPU " << gpu.gpu_id << " LBVH backend: " << message << '\n';
    gpu.backend_log = message;
  }
}

static void destroy_rt_resources(LBVHGPUState& gpu) {
  delete gpu.rt_backend;
  gpu.rt_backend = nullptr;
  gpu.rt_attempted = false;
  gpu.rt_unavailable_reason.clear();
  gpu.backend_log.clear();
  for (cudaEvent_t event : {gpu.timing_start, gpu.timing_build_end,
                           gpu.timing_query_start, gpu.timing_query_end})
    if (event) cudaEventDestroy(event);
  gpu.timing_start = gpu.timing_build_end = gpu.timing_query_start = gpu.timing_query_end = nullptr;
}

static void ensure_contact_capacity(ContactListBuffer& contacts, int32_t required) {
  if (required <= contacts.max_contacts) return;
  ContactPair* next = nullptr;
  CUDA_CHECK(cudaMalloc(&next, size_t(required) * sizeof(ContactPair)));
  const auto result = cudaFree(contacts.pairs);
  if (result != cudaSuccess) { cudaFree(next); CUDA_CHECK(result); }
  contacts.pairs = next;
  contacts.max_contacts = required;
}

static void destroy_sph_resources(LBVHGPUState& gpu) {
  cudaFree(gpu.sph_pairs.pairs);
  cudaFree(gpu.sph_pairs.num_contacts);
  gpu.sph_pairs = ContactListBuffer{};
  cudaFree(gpu.sph_pair_count); gpu.sph_pair_count = nullptr;
  cudaFree(gpu.sph_overflow); gpu.sph_overflow = nullptr;
  cudaFree(gpu.sph_offsets); gpu.sph_offsets = nullptr;
  cudaFree(gpu.sph_cursors); gpu.sph_cursors = nullptr;
  cudaFree(gpu.sph_neighbors); gpu.sph_neighbors = nullptr;
  cudaFree(gpu.sph_scan_temp); gpu.sph_scan_temp = nullptr;
  gpu.sph_neighbor_capacity = gpu.sph_scan_bytes = 0;
  gpu.sph_ready = gpu.sph_cache_valid = false;
}

void LBVHSearchImpl::initialize(const SimConfig& config, SimulationState& state) {
  if (config.neighbor.method != NeighborMethod::LBVH)
    return;

  auto &gpu_state = get_gpu_state();
  if (gpu_state.initialized) {
    std::cerr << "WARNING: LBVHSearchImpl already initialized" << std::endl;
    return;
  }

  int32_t num_particles = state.particles.num_particles;
  int32_t num_local = state.particles.num_local;
  if (num_local < 0 || num_local > num_particles) {
    throw std::runtime_error(
        "LBVH invalid local particle count: num_local=" +
        std::to_string(num_local) + ", num_particles=" +
        std::to_string(num_particles));
  }
  gpu_state.stats = NeighborSearchStats();
  select_backend(gpu_state, state, config);
  CUDA_CHECK(cudaEventCreate(&gpu_state.timing_start));
  CUDA_CHECK(cudaEventCreate(&gpu_state.timing_build_end));
  CUDA_CHECK(cudaEventCreate(&gpu_state.timing_query_start));
  CUDA_CHECK(cudaEventCreate(&gpu_state.timing_query_end));
  int32_t capacity = std::max(
      1024, static_cast<int32_t>(num_particles * 1.5f)); // Give 50% headroom

  gpu_state.capacity = capacity;
  int32_t num_nodes = 2 * capacity - 1;

  CUDA_CHECK(cudaMalloc(&gpu_state.nodes, num_nodes * sizeof(LBVHNode)));
  CUDA_CHECK(cudaMalloc(&gpu_state.aabbs, num_nodes * sizeof(LBVHAABB)));
  CUDA_CHECK(cudaMalloc(&gpu_state.morton_codes, capacity * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.morton_codes_alt, capacity * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.temp_unique_codes,   capacity * sizeof(unsigned long long)));

  // CUB DeviceRadixSort temp storage (pre-allocated, eliminates per-rebuild cudaMalloc/cudaFree)
  gpu_state.sort_temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, gpu_state.sort_temp_bytes,
      (uint32_t*)nullptr, (uint32_t*)nullptr,
      (uint32_t*)nullptr, (uint32_t*)nullptr,
      capacity);
  CUDA_CHECK(cudaMalloc(&gpu_state.d_sort_temp, gpu_state.sort_temp_bytes));

  CUDA_CHECK(cudaMalloc(&gpu_state.sorted_indices, capacity * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.sorted_indices_alt, capacity * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.atomic_flags, capacity * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.whole_domain, sizeof(LBVHAABB)));
  CUDA_CHECK(cub::DeviceReduce::Reduce(
      nullptr, gpu_state.domain_reduce_temp_bytes,
      static_cast<const LBVHAABB*>(nullptr), gpu_state.whole_domain,
      capacity, AABBMergeOp(), LBVHAABB{}));
  CUDA_CHECK(cudaMalloc(&gpu_state.domain_reduce_temp, gpu_state.domain_reduce_temp_bytes));
  gpu_state.max_support.initialize();
  if (num_particles > 0) {
    const float* support = state.particles.support_extent
        ? state.particles.support_extent : state.particles.radius;
    gpu_state.max_support.reserve(support, capacity, nullptr);
  }

  // Snapshot position buffers (float precision)
  CUDA_CHECK(cudaMalloc(&gpu_state.snapshot_pos_x, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&gpu_state.snapshot_pos_y, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&gpu_state.snapshot_pos_z, capacity * sizeof(float)));

  // Displacement scalar
  CUDA_CHECK(cudaMalloc(&gpu_state.d_max_disp_sq, sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&gpu_state.h_max_disp_sq, sizeof(float)));
  *gpu_state.h_max_disp_sq = 1e30f;

  // CUDA event for async readback
  CUDA_CHECK(cudaEventCreateWithFlags(&gpu_state.disp_event, cudaEventDisableTiming));

  // CUB temp storage: query size with a dry-run DeviceReduce::Max call
  float* dummy_in  = nullptr;
  float* dummy_out = nullptr;
  gpu_state.cub_disp_temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, gpu_state.cub_disp_temp_bytes,
                         dummy_in, dummy_out, capacity);
  CUDA_CHECK(cudaMalloc(&gpu_state.d_cub_disp_temp, gpu_state.cub_disp_temp_bytes));

  // Per-particle scratch buffer for disp² values
  CUDA_CHECK(cudaMalloc(&gpu_state.d_disp_scratch, capacity * sizeof(float)));

  // --- Verlet Buffer Tracking Variables ---
  CUDA_CHECK(cudaMalloc(&gpu_state.d_verlet_disp_scaled, sizeof(int32_t)));
  CUDA_CHECK(cudaMallocHost(&gpu_state.h_pinned_verlet_disp, sizeof(float)));
  CUDA_CHECK(cudaEventCreateWithFlags(&gpu_state.verlet_copy_event, cudaEventDisableTiming));

  // Threshold resolved at runtime in find_contacts()
  gpu_state.rebuild_threshold_sq = 0.0f;
  gpu_state.threshold_resolved   = false;
  gpu_state.cached_max_disp_sq   = 1e30f;
  gpu_state.async_readback_pending = false;
  gpu_state.steps_since_rebuild    = 0;

  // Verlet cache allocation
  int32_t max_pairs = num_particles * 12;
  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_cache.pair_i, max_pairs * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_cache.pair_j, max_pairs * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_cache.pair_image_x, max_pairs * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_cache.pair_image_y, max_pairs * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_cache.pair_image_z, max_pairs * sizeof(int8_t)));
  gpu_state.verlet_cache.max_pairs = max_pairs;
  gpu_state.verlet_cache.cache_allocated = true;
  gpu_state.verlet_cache.cache_valid = false;

  CUDA_CHECK(cudaMalloc(&gpu_state.verlet_particles, capacity * sizeof(LBVHCacheParticle)));

  // Allocate diagnostic counters
  CUDA_CHECK(cudaMalloc(&gpu_state.debug_stack_overflows, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&gpu_state.debug_contact_overflows, sizeof(int64_t)));
  CUDA_CHECK(cudaMemset(gpu_state.debug_stack_overflows, 0, sizeof(int64_t)));
  CUDA_CHECK(cudaMemset(gpu_state.debug_contact_overflows, 0, sizeof(int64_t)));


  // Publish LBVH device pointers to NeighborData so SPH kernels can use them
  // even before the first rebuild call. Without this, compute_sph_density_kernel
  // (called before the first needs_rebuild/rebuild cycle) dereferences null
  // nodes/aabbs/sorted_indices and crashes with cudaErrorIllegalAddress.
  state.neighbor.nodes          = gpu_state.nodes;
  state.neighbor.aabbs          = gpu_state.aabbs;
  state.neighbor.sorted_indices = gpu_state.sorted_indices;
  state.neighbor.num_internal_nodes = state.particles.num_particles - 1;

  // Force the first needs_rebuild() check to return true so the tree is
  // actually built on step 1. Without this, cached_verlet_disp_sq=0 and
  // steps_since_verlet_rebuild=0 < update_frequency, so needs_rebuild
  // returns false and the tree is never built.
  gpu_state.steps_since_verlet_rebuild = config.neighbor.update_frequency;

  gpu_state.initialized = true;
  gpu_state.stats.num_rebuilds = 0;
}

void LBVHSearchImpl::shutdown(SimulationState& state, int32_t gpu_id) {
  auto &gpu_state = get_gpu_state(gpu_id);
  if (!gpu_state.initialized)
    return;

  int previous_device = 0;
  CUDA_CHECK(cudaGetDevice(&previous_device));
  CUDA_CHECK(cudaSetDevice(gpu_id));
  destroy_rt_resources(gpu_state);
  destroy_sph_resources(gpu_state);

  cudaFree(gpu_state.nodes);          gpu_state.nodes = nullptr;
  cudaFree(gpu_state.aabbs);          gpu_state.aabbs = nullptr;
  cudaFree(gpu_state.morton_codes);    gpu_state.morton_codes = nullptr;
  cudaFree(gpu_state.morton_codes_alt); gpu_state.morton_codes_alt = nullptr;
  cudaFree(gpu_state.temp_unique_codes); gpu_state.temp_unique_codes = nullptr;
  if (gpu_state.d_sort_temp)       { cudaFree(gpu_state.d_sort_temp);       gpu_state.d_sort_temp = nullptr; }
  cudaFree(gpu_state.sorted_indices); gpu_state.sorted_indices = nullptr;
  cudaFree(gpu_state.sorted_indices_alt); gpu_state.sorted_indices_alt = nullptr;
  cudaFree(gpu_state.atomic_flags);   gpu_state.atomic_flags = nullptr;
  cudaFree(gpu_state.whole_domain); gpu_state.whole_domain = nullptr;
  cudaFree(gpu_state.domain_reduce_temp); gpu_state.domain_reduce_temp = nullptr;
  gpu_state.max_support.release();
  gpu_state.contact_dedup.release();
  cudaFree(gpu_state.verlet_particles); gpu_state.verlet_particles = nullptr;

  if (gpu_state.snapshot_pos_x)   { cudaFree(gpu_state.snapshot_pos_x);   gpu_state.snapshot_pos_x = nullptr; }
  if (gpu_state.snapshot_pos_y)   { cudaFree(gpu_state.snapshot_pos_y);   gpu_state.snapshot_pos_y = nullptr; }
  if (gpu_state.snapshot_pos_z)   { cudaFree(gpu_state.snapshot_pos_z);   gpu_state.snapshot_pos_z = nullptr; }
  if (gpu_state.d_max_disp_sq)    { cudaFree(gpu_state.d_max_disp_sq);    gpu_state.d_max_disp_sq = nullptr; }
  if (gpu_state.h_max_disp_sq)    { cudaFreeHost(gpu_state.h_max_disp_sq); gpu_state.h_max_disp_sq = nullptr; }
  if (gpu_state.disp_event)       { cudaEventDestroy(gpu_state.disp_event); gpu_state.disp_event = nullptr; }
  if (gpu_state.d_cub_disp_temp)  { cudaFree(gpu_state.d_cub_disp_temp);  gpu_state.d_cub_disp_temp = nullptr; }
  if (gpu_state.d_disp_scratch)   { cudaFree(gpu_state.d_disp_scratch);   gpu_state.d_disp_scratch = nullptr; }
  if (gpu_state.d_verlet_disp_scaled) { cudaFree(gpu_state.d_verlet_disp_scaled); gpu_state.d_verlet_disp_scaled = nullptr; }
  if (gpu_state.h_pinned_verlet_disp) { cudaFreeHost(gpu_state.h_pinned_verlet_disp); gpu_state.h_pinned_verlet_disp = nullptr; }
  if (gpu_state.verlet_copy_event) { cudaEventDestroy(gpu_state.verlet_copy_event); gpu_state.verlet_copy_event = nullptr; }

  if (gpu_state.debug_stack_overflows) {
    cudaFree(gpu_state.debug_stack_overflows);
    gpu_state.debug_stack_overflows = nullptr;
  }
  if (gpu_state.debug_contact_overflows) {
    cudaFree(gpu_state.debug_contact_overflows);
    gpu_state.debug_contact_overflows = nullptr;
  }

  if (gpu_state.verlet_cache.cache_allocated) {
    cudaFree(gpu_state.verlet_cache.pair_i);  gpu_state.verlet_cache.pair_i = nullptr;
    cudaFree(gpu_state.verlet_cache.pair_j);  gpu_state.verlet_cache.pair_j = nullptr;
    cudaFree(gpu_state.verlet_cache.pair_image_x); gpu_state.verlet_cache.pair_image_x = nullptr;
    cudaFree(gpu_state.verlet_cache.pair_image_y); gpu_state.verlet_cache.pair_image_y = nullptr;
    cudaFree(gpu_state.verlet_cache.pair_image_z); gpu_state.verlet_cache.pair_image_z = nullptr;
    gpu_state.verlet_cache.cache_allocated = false;
    gpu_state.verlet_cache.cache_valid = false;
  }

  gpu_state.initialized = false;
  gpu_state.active_backend = NeighborBackend::CUDA;
  gpu_state.tracked_num_particles = gpu_state.tracked_num_local = -1;
  gpu_state.threshold_resolved = false;
  gpu_state.backend_timing_enabled = false;
  state.neighbor.nodes = nullptr;
  state.neighbor.aabbs = nullptr;
  state.neighbor.sorted_indices = nullptr;
  state.neighbor.sph_offsets = nullptr;
  state.neighbor.sph_neighbors = nullptr;
  state.neighbor.sph_adjacency_valid = false;
  CUDA_CHECK(cudaSetDevice(previous_device));
}

void LBVHSearchImpl::shutdown(SimulationState& state) {
  int32_t gpu_id = 0;
  cudaGetDevice(&gpu_id);
  shutdown(state, gpu_id);
}

void LBVHSearchImpl::invalidate_cache() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto &pair : gpu_states_) {
    if (pair.second.initialized) {
      pair.second.cached_max_disp_sq = 1e30f;
      pair.second.sph_ready = pair.second.sph_cache_valid = false;
      pair.second.verlet_cache.cache_valid = false;
      pair.second.async_readback_pending = false;
    }
  }
}

void LBVHSearchImpl::shutdown_all() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto &pair : gpu_states_) {
    if (pair.second.initialized) {
      cudaSetDevice(pair.first);
      destroy_rt_resources(pair.second);
      destroy_sph_resources(pair.second);
      cudaFree(pair.second.nodes);
      cudaFree(pair.second.aabbs);
      cudaFree(pair.second.morton_codes);
      cudaFree(pair.second.morton_codes_alt);
      cudaFree(pair.second.temp_unique_codes);
      if (pair.second.d_sort_temp)       { cudaFree(pair.second.d_sort_temp);       pair.second.d_sort_temp = nullptr; }
      cudaFree(pair.second.sorted_indices);
      cudaFree(pair.second.sorted_indices_alt);
      cudaFree(pair.second.atomic_flags);
      cudaFree(pair.second.whole_domain);
      cudaFree(pair.second.domain_reduce_temp);
      pair.second.max_support.release();
      pair.second.contact_dedup.release();
      cudaFree(pair.second.verlet_particles); pair.second.verlet_particles = nullptr;

      if (pair.second.snapshot_pos_x)   { cudaFree(pair.second.snapshot_pos_x);   pair.second.snapshot_pos_x = nullptr; }
      if (pair.second.snapshot_pos_y)   { cudaFree(pair.second.snapshot_pos_y);   pair.second.snapshot_pos_y = nullptr; }
      if (pair.second.snapshot_pos_z)   { cudaFree(pair.second.snapshot_pos_z);   pair.second.snapshot_pos_z = nullptr; }
      if (pair.second.d_max_disp_sq)    { cudaFree(pair.second.d_max_disp_sq);    pair.second.d_max_disp_sq = nullptr; }
      if (pair.second.h_max_disp_sq)    { cudaFreeHost(pair.second.h_max_disp_sq); pair.second.h_max_disp_sq = nullptr; }
      if (pair.second.disp_event)       { cudaEventDestroy(pair.second.disp_event); pair.second.disp_event = nullptr; }
      if (pair.second.d_cub_disp_temp)  { cudaFree(pair.second.d_cub_disp_temp);  pair.second.d_cub_disp_temp = nullptr; }
      if (pair.second.d_disp_scratch)   { cudaFree(pair.second.d_disp_scratch);   pair.second.d_disp_scratch = nullptr; }
      if (pair.second.d_verlet_disp_scaled) { cudaFree(pair.second.d_verlet_disp_scaled); pair.second.d_verlet_disp_scaled = nullptr; }
      if (pair.second.h_pinned_verlet_disp) { cudaFreeHost(pair.second.h_pinned_verlet_disp); pair.second.h_pinned_verlet_disp = nullptr; }
      if (pair.second.verlet_copy_event) { cudaEventDestroy(pair.second.verlet_copy_event); pair.second.verlet_copy_event = nullptr; }

      if (pair.second.debug_stack_overflows) {
        cudaFree(pair.second.debug_stack_overflows);
        pair.second.debug_stack_overflows = nullptr;
      }
      if (pair.second.debug_contact_overflows) {
        cudaFree(pair.second.debug_contact_overflows);
        pair.second.debug_contact_overflows = nullptr;
      }

      if (pair.second.verlet_cache.cache_allocated) {
        cudaFree(pair.second.verlet_cache.pair_i);
        cudaFree(pair.second.verlet_cache.pair_j);
        cudaFree(pair.second.verlet_cache.pair_image_x);
        cudaFree(pair.second.verlet_cache.pair_image_y);
        cudaFree(pair.second.verlet_cache.pair_image_z);
        pair.second.verlet_cache.cache_allocated = false;
        pair.second.verlet_cache.cache_valid = false;
      }

      pair.second.initialized = false;
    }
  }
  gpu_states_.clear();
}

bool LBVHSearchImpl::is_initialized() {
  int32_t gpu_id;
  if (cudaGetDevice(&gpu_id) != cudaSuccess)
    return false;
  return is_initialized(gpu_id);
}

bool LBVHSearchImpl::is_initialized(int32_t gpu_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto it = gpu_states_.find(gpu_id);
  if (it != gpu_states_.end()) {
    return it->second.initialized;
  }
  return false;
}

// Make Morton codes unique by merging 30-bit code with sorted index.
// Approach: unique_codes[k] = (morton_30bit[k] << 32) | k
// This is the standard technique used by ToruNiina/lbvh and KittenGpuLBVH.
//
// Guarantees strictly increasing codes because:
// - morton_30bit[k] is non-decreasing (already sorted by CUB radix sort)
// - If morton_30bit[k] == morton_30bit[k+1], the lower 32 bits (k vs k+1) break the tie
// - No overflow: 30-bit code << 32 = 62 bits, plus 32-bit index = 64 bits total
//
// This replaces the previous prefix-max approach which had an unsigned wraparound bug:
// adjusted[k] = sorted_codes[k] - k would wrap to ~2^64 when sorted_codes[k] < k,
// producing non-strictly-increasing codes → corrupted tree with self-referencing
// nodes → infinite loop in traversal and AABB propagation kernels.
__global__ void lbvh_make_unique_codes_kernel(
    const uint32_t* __restrict__ sorted_codes_30bit,
    unsigned long long* __restrict__ unique_codes,
    int32_t N) {
  int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  unique_codes[k] = (static_cast<unsigned long long>(sorted_codes_30bit[k]) << 32)
                  | static_cast<unsigned long long>(k);
}

__global__ void init_nodes_kernel(LBVHNode *nodes, int32_t num) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num)
    return;
  nodes[idx].parent_idx = 0xFFFFFFFF;
  nodes[idx].left_idx = 0xFFFFFFFF;
  nodes[idx].right_idx = 0xFFFFFFFF;
  nodes[idx].object_idx = 0xFFFFFFFF;
}

void LBVHSearchImpl::rebuild(SimulationState &state, const SimConfig &config,
                                cudaStream_t stream) {
  auto &gpu_state = get_gpu_state();
  if (!gpu_state.initialized)
    return;
  state.neighbor.sph_adjacency_valid = false;
  gpu_state.sph_ready = gpu_state.sph_cache_valid = false;
  select_backend(gpu_state, state, config);

  int32_t N = state.particles.num_particles;
  int32_t num_local = state.particles.num_local;
  if (num_local < 0 || num_local > N) {
    throw std::runtime_error(
        "LBVH invalid local particle count: num_local=" +
        std::to_string(num_local) + ", num_particles=" + std::to_string(N));
  }
  // N <= 1: No tree to build (single particle = single leaf, no internal nodes)
  if (N <= 1) {
    state.neighbor.num_particles = N;
    state.neighbor.num_internal_nodes = 0;
    gpu_state.tracked_num_particles = N;
    gpu_state.tracked_num_local = num_local;
    return;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  int32_t num_internal = N - 1;
  int32_t num_nodes = 2 * N - 1;

  // Resize if needed
  if (N > gpu_state.capacity) {
    const NeighborSearchStats previous = gpu_state.stats;
    const bool timing_enabled = gpu_state.backend_timing_enabled;
    shutdown(state);
    initialize(config, state); // reallocates larger buffers
    const double init_ms = gpu_state.stats.backend_init_time_ms;
    gpu_state.stats = previous;
    gpu_state.stats.backend_init_time_ms += init_ms;
    gpu_state.backend_timing_enabled = timing_enabled;
  }

  if (gpu_state.active_backend == NeighborBackend::RT) {
    gpu_state.rt_backend->update(rt_input(state, config), true, stream);
    state.neighbor.nodes = nullptr;
    state.neighbor.aabbs = nullptr;
    state.neighbor.sorted_indices = nullptr;
    state.neighbor.num_internal_nodes = 0;
    state.neighbor.num_particles = N;
    state.neighbor.morton_data_valid = false;
    gpu_state.stats.num_rebuilds++;
    gpu_state.current_generation++;
    gpu_state.tracked_num_particles = N;
    gpu_state.tracked_num_local = num_local;
    gpu_state.stats.rebuild_time_ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    return;
  }

  int threads = config.gpu.block_size;
  int blocks_N = (N + threads - 1) / threads;
  int blocks_internal = (num_internal + threads - 1) / threads;
  int blocks_nodes = (num_nodes + threads - 1) / threads;

  // 0. Initialize nodes
  init_nodes_kernel<<<blocks_nodes, threads, 0, stream>>>(gpu_state.nodes,
                                                          num_nodes);

  // 1. Compute leaf AABBs
  // DEM support_extent includes the material-row maximum cohesive cutoff;
  // SPH stores full K*h for halo exchange, but each tree leaf needs only
  // half of that extent: the pair support is K*(hi+hj)/2.
  const float* aabb_extent = state.particles.support_extent;
  const float extent_scale = config.physics == SimulationPhysics::SPH ? 0.5f : 1.0f;
  lbvh_compute_aabbs_kernel<<<blocks_N, threads, 0, stream>>>(
      state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
      aabb_extent, gpu_state.aabbs, N, num_internal, extent_scale);

  // 2. Reduce bounds entirely on the device. Morton generation consumes this
  // result on the same stream, so no host readback/allocation barrier is needed.
  LBVHAABB default_aabb;
  default_aabb.lower = make_float4(1e30f, 1e30f, 1e30f, 0);
  default_aabb.upper = make_float4(-1e30f, -1e30f, -1e30f, 0);

  CUDA_CHECK(cub::DeviceReduce::Reduce(
      gpu_state.domain_reduce_temp, gpu_state.domain_reduce_temp_bytes,
      static_cast<const LBVHAABB*>(gpu_state.aabbs + num_internal), gpu_state.whole_domain,
      N, AABBMergeOp(), default_aabb, stream));

  // 2a. Compute 30-bit morton codes (used for tree construction + uniqueness merge)
  lbvh_compute_morton_codes_kernel<<<blocks_N, threads, 0, stream>>>(
      gpu_state.aabbs, gpu_state.morton_codes, N, num_internal, gpu_state.whole_domain);

  // 3. Sort indices by 30-bit morton code — CUB out-of-place
  thrust::device_ptr<uint32_t> d_indices(gpu_state.sorted_indices);
  thrust::sequence(thrust::cuda::par.on(stream), d_indices, d_indices + N);

  // Keep the CUB output buffers as the canonical sorted arrays. Each allocation
  // has one owner; neither output aliases the later 64-bit uniqueness keys.
  cub::DeviceRadixSort::SortPairs(
      gpu_state.d_sort_temp, gpu_state.sort_temp_bytes,
      gpu_state.morton_codes, gpu_state.morton_codes_alt,
      gpu_state.sorted_indices, gpu_state.sorted_indices_alt,
      N, 0, 32, stream);
  std::swap(gpu_state.morton_codes, gpu_state.morton_codes_alt);
  std::swap(gpu_state.sorted_indices, gpu_state.sorted_indices_alt);

  // Produce leaves directly in their final order from immutable particle
  // arrays. Reading the old leaf array here would require permutation scratch.
  lbvh_refresh_leaf_aabbs_kernel<<<blocks_N, threads, 0, stream>>>(
      state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
      aabb_extent, gpu_state.sorted_indices, gpu_state.aabbs, N, num_internal,
      0.0f, extent_scale);

  // 4. Construct leaf nodes
  lbvh_construct_leaves_kernel<<<blocks_N, threads, 0, stream>>>(
      gpu_state.sorted_indices, gpu_state.nodes, N, num_internal);

  // 5. Construct internal nodes
  // CRITICAL: Karras algorithm requires UNIQUE Morton codes.
  // We ALWAYS merge the sorted 30-bit morton codes with the sorted index:
  //   unique_codes[k] = (sorted_codes_30bit[k] << 32) | k
  // This guarantees strictly increasing codes without any conditional duplicate
  // detection. The previous approach had a bug: when duplicate codes existed,
  // the prefix-max handler used unsigned subtraction (codes[k] - k) which
  // wraps to ~2^64 when codes[k] < k, producing non-strictly-increasing codes
  // that corrupt the tree with self-referencing nodes → infinite loop.
  lbvh_make_unique_codes_kernel<<<blocks_N, threads, 0, stream>>>(
      gpu_state.morton_codes, gpu_state.temp_unique_codes, N);

  lbvh_construct_internal_nodes_kernel<unsigned long long>
      <<<blocks_internal, threads, 0, stream>>>(
          gpu_state.temp_unique_codes,
          gpu_state.nodes, gpu_state.aabbs, N);

  // 6. Bottom-up AABB propagation
  CUDA_CHECK(cudaMemsetAsync(gpu_state.atomic_flags, 0,
                             num_internal * sizeof(int32_t), stream));

  lbvh_propagate_aabbs_kernel<<<blocks_N, threads, 0, stream>>>(
      gpu_state.nodes, gpu_state.aabbs, gpu_state.atomic_flags, num_internal,
      N);

  // Export BVH tree to NeighborData so SPH neighbor query kernels can access it.
  // CRITICAL: Without these, for_each_sph_neighbor_lbvh() dereferences null
  // pointers → illegal memory access at address 0x4 (nodes/node].left_idx).
  state.neighbor.nodes          = gpu_state.nodes;
  state.neighbor.aabbs          = gpu_state.aabbs;
  state.neighbor.sorted_indices = gpu_state.sorted_indices;
  state.neighbor.num_internal_nodes = num_internal;
  state.neighbor.num_particles  = N;

  gpu_state.stats.num_rebuilds++;
  gpu_state.current_generation++;
  gpu_state.tracked_num_particles = N;
  gpu_state.tracked_num_local = state.particles.num_local;
  auto t_end = std::chrono::high_resolution_clock::now();
  gpu_state.stats.rebuild_time_ms =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
}

// Keep CUB scratch and the result scalar per GPU. A zero-particle reduction
// retains the old Thrust initial value; support extent retains radius fallback.
static float compute_max_radius(const ParticleSoA& particles, LBVHGPUState& gpu,
                                cudaStream_t stream) {
    const int32_t n = particles.num_particles;
    {
      LBVHProfileRange range("LBVH: Max Support Reduce");
      gpu.max_support.enqueue(particles.support_extent, particles.radius, n, stream);
    }
    {
      LBVHProfileRange range("LBVH: Max Support Wait");
      return gpu.max_support.wait(n, stream);
    }
}

void LBVHSearchImpl::find_contacts(SimulationState &state,
                                      const SimConfig &config,
                                      ContactListBuffer &contacts,
                                      double evaluation_time,
                                      cudaStream_t stream) {
  auto &gpu_state = get_gpu_state();
  if (!gpu_state.initialized)
    return;

  int32_t N = state.particles.num_particles;
  int32_t num_local = state.particles.num_local;
  if (num_local < 0 || num_local > N) {
    throw std::runtime_error(
        "LBVH invalid local particle count: num_local=" +
        std::to_string(num_local) + ", num_particles=" + std::to_string(N));
  }

  // Resolve feature/hardware changes before trusting either cached topology
  // or cached pairs. In particular, adding ghosts must switch auto to CUDA.
  select_backend(gpu_state, state, config);
  if (N > gpu_state.capacity) {
    const NeighborSearchStats previous = gpu_state.stats;
    const bool timing_enabled = gpu_state.backend_timing_enabled;
    shutdown(state);
    initialize(config, state);
    const double init_ms = gpu_state.stats.backend_init_time_ms;
    gpu_state.stats = previous;
    gpu_state.stats.backend_init_time_ms += init_ms;
    gpu_state.backend_timing_enabled = timing_enabled;
  }

  const bool particle_layout_changed =
      N != gpu_state.tracked_num_particles ||
      num_local != gpu_state.tracked_num_local;
  if (particle_layout_changed) {
    // A changed total count invalidates the topology; a changed local count
    // changes query ownership. Either also invalidates cached Verlet pairs.
    gpu_state.cached_max_disp_sq = 1e30f;
    gpu_state.async_readback_pending = false;
    gpu_state.verlet_cache.cache_valid = false;
    gpu_state.verlet_cache.h_num_pairs = 0;
    gpu_state.steps_since_verlet_rebuild = 0;
    gpu_state.stats.cache_invalidations++;
  }

  // Always clear the result before any trivial return so a shrinking domain
  // cannot expose contacts from a previous call.
  {
    LBVHProfileRange range("LBVH: Contact Buffer Reset");
    NeighborSearch::reset_contact_buffer(contacts, stream);
  }
  if (num_local == 0 || N <= 1) {
    gpu_state.tracked_num_particles = N;
    gpu_state.tracked_num_local = num_local;
    gpu_state.stats.num_contacts_found = 0;
    return;
  }

  // A frequency of one forces a topology rebuild for every contact search.
  // In that mode neither the topology-displacement tracker nor cached Verlet
  // pairs can be consumed before the next rebuild, so do not enqueue or wait
  // for any of their work. Keep the cache invalid in case a caller changes
  // this runtime configuration after a previous reusable-cache step.
  const bool force_rebuild_every_step = config.neighbor.update_frequency == 1;
  if (force_rebuild_every_step) {
    gpu_state.verlet_cache.cache_valid = false;
    gpu_state.verlet_cache.h_num_pairs = 0;
    gpu_state.async_readback_pending = false;
    gpu_state.verlet_readback_pending = false;
  }

  // [A] Read async displacement result from previous step
  if (!force_rebuild_every_step && gpu_state.async_readback_pending) {
    if (cudaEventQuery(gpu_state.disp_event) == cudaSuccess) {
      gpu_state.cached_max_disp_sq = *gpu_state.h_max_disp_sq;
      gpu_state.async_readback_pending = false;
    }
    // If not ready yet, keep using the cached value from the step before
  }

  // [B] Resolve rebuild_threshold_sq on first call
  if (!force_rebuild_every_step && !gpu_state.threshold_resolved) {
    float max_r = compute_max_radius(state.particles, gpu_state, stream);
    float threshold;
    if (config.neighbor.lbvh_rebuild_threshold > 0.0f) {
      threshold = config.neighbor.lbvh_rebuild_threshold;
    } else {
      threshold = max_r * config.neighbor.lbvh_skin_factor / 2.0f;
    }
    gpu_state.rebuild_threshold_sq = threshold * threshold;
    gpu_state.threshold_resolved   = true;
  }

  // [C] Decide path
  // cached_max_disp_sq is initialized to 1e30f, so first call always rebuilds
  // Also force rebuild if update_frequency steps have passed
  if (!force_rebuild_every_step) {
    gpu_state.steps_since_rebuild++;  // Increment before decision
  }
  bool do_full_rebuild = force_rebuild_every_step || config.shear_box.enabled ||
                         (gpu_state.cached_max_disp_sq > gpu_state.rebuild_threshold_sq) ||
                         (gpu_state.steps_since_rebuild >= config.neighbor.update_frequency) ||
                         (N > gpu_state.capacity) || particle_layout_changed;

  // Get periodic boundary config early for cache fast path
  PeriodicBoundaryConfig pbc = config.shear_box.enabled
      ? WallManager::get_pbc_config() : extract_pbc_config(config);

  int threads      = config.gpu.block_size;
  int blocks_N     = (N + threads - 1) / threads;
  int blocks_local = (num_local + threads - 1) / threads;
  int32_t num_internal = N - 1;

  auto t_start = std::chrono::high_resolution_clock::now();

  // [C-extra] A cached list is safe only at the current force positions. Never
  // use an older asynchronous displacement result to approve a new drift.
  const uint8_t periodic_axes = uint8_t(pbc.periodic_x) |
      (uint8_t(pbc.periodic_y) << 1) | (uint8_t(pbc.periodic_z) << 2);
  const float periodic_bounds[6] = {pbc.box_min_x, pbc.box_min_y, pbc.box_min_z,
                                   pbc.box_length_x, pbc.box_length_y, pbc.box_length_z};
  bool cache_config_changed = gpu_state.cached_verlet_skin != config.neighbor.verlet_skin ||
                             gpu_state.cached_periodic_axes != periodic_axes;
  for (int axis = 0; axis < 6; ++axis)
      cache_config_changed |= gpu_state.cached_periodic_bounds[axis] != periodic_bounds[axis];
  bool verlet_cache_valid = !do_full_rebuild && !cache_config_changed &&
      gpu_state.verlet_cache.cache_valid && config.neighbor.verlet_buffer_enabled;
  if (verlet_cache_valid) {
      CUDA_CHECK(cudaMemsetAsync(gpu_state.d_verlet_disp_scaled, 0, sizeof(int32_t), stream));
      // The reduction has a fixed 256-thread layout, independent of the user's
      // general launch block size.
      lbvh_current_cache_displacement_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          state.particles.global_id, state.particles.clump_id, state.particles.support_extent,
          gpu_state.verlet_particles, gpu_state.d_verlet_disp_scaled, N);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(gpu_state.h_pinned_verlet_disp, gpu_state.d_verlet_disp_scaled,
                                sizeof(float), cudaMemcpyDeviceToHost, stream));
      {
          LBVHProfileRange range("LBVH: Cache Displacement Wait");
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      gpu_state.cached_verlet_disp_sq = *gpu_state.h_pinned_verlet_disp;
      gpu_state.verlet_readback_pending = false;
      gpu_state.steps_since_verlet_rebuild++;
      const double skin = config.neighbor.verlet_skin;
      // Two particles can approach by twice the largest displacement. Equality
      // rebuilds conservatively rather than depending on boundary rounding.
      if (!(skin > 0.0) || 4.0 * double(gpu_state.cached_verlet_disp_sq) >= skin * skin ||
          gpu_state.steps_since_verlet_rebuild >= config.neighbor.update_frequency) {
          verlet_cache_valid = false;
          gpu_state.verlet_cache.cache_valid = false;
          gpu_state.stats.cache_invalidations++;
          do_full_rebuild = true;
      }
  }

  // [C-extra] Verlet cache fast path - use cached pairs if valid
  if (!config.shear_box.enabled && !do_full_rebuild &&
      config.neighbor.verlet_buffer_enabled &&
      gpu_state.verlet_cache.cache_allocated &&
      verlet_cache_valid) {
      // Fast path: copy only cached candidate identities
      ensure_contact_capacity(contacts, gpu_state.verlet_cache.h_num_pairs);
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
      contacts.h_num_contacts = gpu_state.verlet_cache.h_num_pairs;

      // FMM builds its own spatial keys. Keep the optional export explicitly
      // unavailable instead of generating an unconsumed 64-bit code array.
      state.neighbor.morton_codes_64 = nullptr;
      state.neighbor.sorted_indices = gpu_state.active_backend == NeighborBackend::CUDA
          ? gpu_state.sorted_indices : nullptr;
      state.neighbor.rebuild_generation = gpu_state.current_generation;
      state.neighbor.morton_data_valid = false;

      // NOTE: Verlet reference positions are NOT reset here.
      // They are only reset after a full contact search (cache population),
      // so displacement accumulates across cache uses and triggers
      // invalidation when 2*max_disp >= verlet_skin.

      auto t_end = std::chrono::high_resolution_clock::now();
      gpu_state.stats.search_time_ms =
          std::chrono::duration<float, std::milli>(t_end - t_start).count();

      return;
  }


  if (gpu_state.backend_timing_enabled)
    CUDA_CHECK(cudaEventRecord(gpu_state.timing_start, stream));
  if (do_full_rebuild) {
    // [C-full] Full rebuild (existing pipeline)
    LBVHProfileRange range("LBVH: Full Rebuild");
    rebuild(state, config, stream);

    // The position snapshot is only useful when a later step can take the
    // topology fast path. update_frequency == 1 cannot do so.
    if (!force_rebuild_every_step) {
      lbvh_copy_snapshot_kernel<<<blocks_N, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          gpu_state.snapshot_pos_x, gpu_state.snapshot_pos_y, gpu_state.snapshot_pos_z,
          N);
      gpu_state.cached_max_disp_sq = 0.0f;
    }
    gpu_state.steps_since_rebuild = 0;
    gpu_state.stats.lbvh_full_rebuilds++;
  } else if (gpu_state.active_backend == NeighborBackend::RT) {
    LBVHProfileRange range("LBVH: RT Update");
    gpu_state.rt_backend->update(rt_input(state, config), false, stream);
    gpu_state.stats.lbvh_fast_path_steps++;
  } else {
    // [C-fast] Fast path: refresh leaf AABBs + re-propagate
    LBVHProfileRange range("LBVH: Refit");

    // skin_half is the rebuild threshold distance (used to inflate AABBs on fast path)
    float skin_half = sqrtf(gpu_state.rebuild_threshold_sq);

    const float* aabb_extent = state.particles.support_extent;
    lbvh_refresh_leaf_aabbs_kernel<<<blocks_N, threads, 0, stream>>>(
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        aabb_extent, gpu_state.sorted_indices, gpu_state.aabbs,
        N, num_internal, skin_half,
        config.physics == SimulationPhysics::SPH ? 0.5f : 1.0f);

    // Reset atomic flags before re-propagation
    CUDA_CHECK(cudaMemsetAsync(gpu_state.atomic_flags, 0,
                               num_internal * sizeof(int32_t), stream));

    lbvh_propagate_aabbs_kernel<<<blocks_N, threads, 0, stream>>>(
        gpu_state.nodes, gpu_state.aabbs, gpu_state.atomic_flags,
        num_internal, N);

    // steps_since_rebuild already incremented before rebuild decision
    gpu_state.stats.lbvh_fast_path_steps++;
  }
  if (gpu_state.backend_timing_enabled)
    CUDA_CHECK(cudaEventRecord(gpu_state.timing_build_end, stream));


  // Topology indices remain available to SPH; FMM does not consume LBVH codes.
  state.neighbor.morton_codes_64 = nullptr;
  state.neighbor.sorted_indices = gpu_state.active_backend == NeighborBackend::CUDA
      ? gpu_state.sorted_indices : nullptr;
  state.neighbor.rebuild_generation = gpu_state.current_generation;
  state.neighbor.morton_data_valid = false;

  // [D] Traversal - dispatch periodic vs non-periodic path
  // A list rebuilt every step has no future motion to cover. Preserve the
  // material support extent, but omit the extra Verlet-only shell. Force
  // owners retire histories absent from this evaluation, including empty lists.
  float contact_threshold = config.neighbor.verlet_buffer_enabled && !force_rebuild_every_step
                          ? -config.neighbor.verlet_skin
                          : 0.0f;

  // Get periodic boundary config - already declared before cache check

  // The periodic traversal needs a pre-deduplication count for its overflow
  // policy. When no deduplication is needed, retain that result instead of
  // synchronizing the same counter again below.
  int32_t known_final_contact_count = -1;
  if (gpu_state.backend_timing_enabled)
    CUDA_CHECK(cudaEventRecord(gpu_state.timing_query_start, stream));
  if (gpu_state.active_backend == NeighborBackend::RT) {
      gpu_state.rt_backend->find_contacts(contacts, stream);
      known_final_contact_count = contacts.h_num_contacts;
  } else if (pbc.is_enabled()) {
      // Periodic path: use periodic-aware kernel + deduplication
      float max_r = compute_max_radius(state.particles, gpu_state, stream);
      ShearPhase shear_phase;
      if (config.shear_box.enabled) {
          const double search_distance = 2.0 * static_cast<double>(max_r)
              + (config.neighbor.verlet_buffer_enabled
                 ? static_cast<double>(config.neighbor.verlet_skin) : 0.0);
          if (!(search_distance < 0.5 * pbc.box_length_x) ||
              !(search_distance < 0.5 * pbc.box_length_y)) {
              throw std::runtime_error("shear unique-image condition violated: 2*max_support+skin must be below half Lx and Ly");
          }
          shear_phase = compute_shear_phase_checked(config.shear_box,
              pbc.box_length_x,pbc.box_length_y,evaluation_time);
      }

      auto launch_periodic = [&]() {
          lbvh_find_contacts_periodic_kernel<<<blocks_local, threads, 0, stream>>>(
              state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
              state.particles.support_extent, gpu_state.nodes, gpu_state.aabbs,
              contacts.pairs, contacts.num_contacts,
              state.particles.clump_id,
              contacts.max_contacts, contact_threshold, num_local,
              pbc, max_r, config.shear_box.enabled, shear_phase);
          CUDA_CHECK(cudaGetLastError());
      };
      { LBVHProfileRange range("LBVH: Periodic Query"); launch_periodic(); }

      // Deduplicate contacts (same i,j from different images)
      int32_t raw_count;
      {
          LBVHProfileRange range("LBVH: Raw Count");
          raw_count = NeighborSearch::get_contact_count(contacts, stream);
      }
      while (config.shear_box.enabled && raw_count > contacts.max_contacts) {
          const int64_t doubled=2LL*contacts.max_contacts;
          const int64_t requested=std::max<int64_t>(raw_count,doubled);
          if(requested>std::numeric_limits<int32_t>::max())
              throw std::overflow_error("shear contact buffer capacity exceeds int32");
          {
              LBVHProfileRange range("LBVH: Contact Overflow Resize");
              NeighborSearch::free_contact_buffer(contacts);
              NeighborSearch::allocate_contact_buffer(contacts,static_cast<int32_t>(requested));
              NeighborSearch::reset_contact_buffer(contacts,stream);
          }
          { LBVHProfileRange range("LBVH: Periodic Query"); launch_periodic(); }
          { LBVHProfileRange range("LBVH: Raw Count");
            raw_count=NeighborSearch::get_contact_count(contacts,stream); }
      }
      if (raw_count > contacts.max_contacts) {
          throw std::runtime_error("periodic contact buffer overflow: required="
              + std::to_string(raw_count) + ", capacity=" + std::to_string(contacts.max_contacts));
      }
      if (raw_count > 1) {
          LBVHProfileRange range("LBVH: Deduplicate");
          deduplicate_contacts(contacts.pairs, raw_count, contacts.num_contacts,
                               stream, gpu_state.contact_dedup);
      } else {
          known_final_contact_count = raw_count;
      }
  } else {
      // Morton-ordered queries with direct pair emission. Keep the dormant
      // staging-buffer kernel disabled: its overflow path can drop pairs.
      lbvh_find_contacts_kernel<<<blocks_N, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          state.particles.support_extent, gpu_state.nodes, gpu_state.aabbs,
          contacts.pairs, contacts.num_contacts,
          state.particles.clump_id, gpu_state.sorted_indices,
          contacts.max_contacts, contact_threshold, num_local, N);
  }

  // Catch kernel launch failures (e.g., shared memory overflow) early
  // rather than silently producing zero contacts.
  CUDA_CHECK(cudaGetLastError());

  // Clamp the contact counter to max_contacts. The narrow-phase kernels
  // increment contacts.num_contacts via atomicAdd for EVERY detected
  // contact, but only write to contacts.pairs when write_idx < max_contacts.
  // An inflated counter causes downstream cache/force kernels to read
  // beyond the allocated buffer → illegal memory access.
  if (!config.shear_box.enabled && gpu_state.active_backend != NeighborBackend::RT)
      clamp_contact_count_kernel<<<1, 1, 0, stream>>>(contacts.num_contacts, contacts.max_contacts);

  // Read the final count once and reuse it for cache population, statistics,
  // and the host mirror consumed by the main loop. A periodic deduplication
  // must complete before this readback; non-periodic paths only need the
  // clamp above.
  int32_t num_contacts_found;
  {
    LBVHProfileRange range("LBVH: Final Count");
    num_contacts_found = known_final_contact_count >= 0
        ? known_final_contact_count
        : NeighborSearch::get_contact_count(contacts, stream);
  }
  contacts.h_num_contacts = num_contacts_found;

  if (gpu_state.backend_timing_enabled) {
    CUDA_CHECK(cudaEventRecord(gpu_state.timing_query_end, stream));
    CUDA_CHECK(cudaEventSynchronize(gpu_state.timing_query_end));
    float build_ms = 0.0f, query_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&build_ms, gpu_state.timing_start, gpu_state.timing_build_end));
    CUDA_CHECK(cudaEventElapsedTime(&query_ms, gpu_state.timing_query_start, gpu_state.timing_query_end));
    if (do_full_rebuild) gpu_state.stats.backend_build_time_ms += build_ms;
    else gpu_state.stats.backend_update_time_ms += build_ms;
    gpu_state.stats.backend_query_time_ms += query_ms;
  }

  // Cache the contact pairs only when a later step can reuse them.
  if (!force_rebuild_every_step && !config.shear_box.enabled &&
      config.neighbor.verlet_buffer_enabled && gpu_state.verlet_cache.cache_allocated) {
      if (num_contacts_found >= 0 && num_contacts_found <= gpu_state.verlet_cache.max_pairs && num_contacts_found <= contacts.max_contacts) {
          if (num_contacts_found > 0) {
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
          }
          lbvh_capture_cache_particles_kernel<<<blocks_N, threads, 0, stream>>>(
              state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
              state.particles.global_id, state.particles.clump_id, state.particles.support_extent,
              gpu_state.verlet_particles, N);
          gpu_state.cached_verlet_skin = config.neighbor.verlet_skin;
          gpu_state.cached_periodic_axes = periodic_axes;
          std::copy(periodic_bounds, periodic_bounds + 6, gpu_state.cached_periodic_bounds);
          gpu_state.verlet_cache.h_num_pairs = num_contacts_found;
          gpu_state.verlet_cache.cache_valid = true;
          gpu_state.stats.cache_misses++;
      } else if (num_contacts_found > gpu_state.verlet_cache.max_pairs) {
          // Too many contacts - invalidate cache
          gpu_state.verlet_cache.cache_valid = false;
          gpu_state.stats.cache_invalidations++;
      }
  }

  gpu_state.stats.num_contacts_found = num_contacts_found;

  auto t_end = std::chrono::high_resolution_clock::now();
  gpu_state.stats.search_time_ms =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();

  // [E] Reset Verlet tracker reference position
  if (!force_rebuild_every_step && config.neighbor.verlet_buffer_enabled) {
      lbvh_store_last_positions_kernel<<<blocks_N, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          state.particles.last_pos_x, state.particles.last_pos_y, state.particles.last_pos_z,
          N);
      gpu_state.cached_verlet_disp_sq = 0.0f;
      *gpu_state.h_pinned_verlet_disp = 0.0f;
      gpu_state.verlet_readback_pending = false;
  }
  gpu_state.steps_since_verlet_rebuild = 0;
}

bool LBVHSearchImpl::needs_rebuild(const SimulationState &state,
                                      const SimConfig &config,
                                      cudaStream_t stream) {
  auto &gpu_state = get_gpu_state();
  if (!gpu_state.initialized) return true;

  int32_t N = state.particles.num_particles;
  int32_t num_local = state.particles.num_local;
  if (num_local < 0 || num_local > N) {
    throw std::runtime_error(
        "LBVH invalid local particle count: num_local=" +
        std::to_string(num_local) + ", num_particles=" + std::to_string(N));
  }
  if (N != gpu_state.tracked_num_particles ||
      num_local != gpu_state.tracked_num_local) {
    return true;
  }
  // N <= 1: No rebuild needed (single particle has no contacts)
  if (N <= 1) return false;

  // A forced per-step rebuild makes the cache and displacement readback
  // unobservable. Return before touching either, including in CPU mocks.
  if (config.neighbor.update_frequency == 1) return true;

  gpu_state.steps_since_verlet_rebuild++;

  if (config.neighbor.verlet_buffer_enabled) {
      if (gpu_state.verlet_readback_pending) {
          if (cudaEventQuery(gpu_state.verlet_copy_event) == cudaSuccess) {
              gpu_state.cached_verlet_disp_sq = *gpu_state.h_pinned_verlet_disp;
              gpu_state.verlet_readback_pending = false;
          }
      }

      float max_disp = sqrtf(gpu_state.cached_verlet_disp_sq);
      float skin = config.neighbor.verlet_skin;

      if (2.0f * max_disp > skin) {
          return true;
      }

      // Use update_frequency from config instead of hardcoded 100
      if (gpu_state.steps_since_verlet_rebuild >= config.neighbor.update_frequency) {
          return true;
      }

      return false;
  }

  return true;
}

const NeighborSearchStats &LBVHSearchImpl::get_stats() {
  int32_t gpu_id = 0;
  cudaGetDevice(&gpu_id);
  return get_stats(gpu_id);
}

NeighborBackend LBVHSearchImpl::get_backend() {
  int32_t gpu_id = 0;
  CUDA_CHECK(cudaGetDevice(&gpu_id));
  return get_backend(gpu_id);
}

NeighborBackend LBVHSearchImpl::get_backend(int32_t gpu_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  const auto it = gpu_states_.find(gpu_id);
  return it != gpu_states_.end() ? it->second.active_backend : NeighborBackend::CUDA;
}

std::string LBVHSearchImpl::get_backend_reason() {
  return get_gpu_state().backend_reason;
}

void LBVHSearchImpl::set_backend_timing_enabled(bool enabled) {
  get_gpu_state().backend_timing_enabled = enabled;
}

const NeighborSearchStats &LBVHSearchImpl::get_stats(int32_t gpu_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto it = gpu_states_.find(gpu_id);
  if (it != gpu_states_.end()) {
    return it->second.stats;
  }
  return default_stats_;
}

void LBVHSearchImpl::launch_async_displacement_monitor(
    const SimulationState& state,
    const SimConfig& config,
    cudaStream_t stream)
{
    auto& gpu_state = get_gpu_state();
    if (!gpu_state.initialized) return;

    int32_t N = state.particles.num_particles;
    // N <= 1: No displacement monitoring needed
    if (N <= 1) return;

    // find_contacts() must fully rebuild at this cadence, so the snapshots
    // and Verlet displacement monitor cannot contribute to any later step.
    if (config.neighbor.update_frequency == 1) return;

    int threads = config.gpu.block_size;
    int blocks  = (N + threads - 1) / threads;

    // Step 1: fill d_disp_scratch with per-particle disp²
    lbvh_displacement_check_kernel<<<blocks, threads, 0, stream>>>(
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        gpu_state.snapshot_pos_x, gpu_state.snapshot_pos_y, gpu_state.snapshot_pos_z,
        gpu_state.d_disp_scratch, N);

    // Step 2: CUB DeviceReduce::Max → gpu_state.d_max_disp_sq
    cub::DeviceReduce::Max(
        gpu_state.d_cub_disp_temp, gpu_state.cub_disp_temp_bytes,
        gpu_state.d_disp_scratch, gpu_state.d_max_disp_sq,
        N, stream);

    // Step 3: async copy to pinned host buffer
    CUDA_CHECK(cudaMemcpyAsync(
        gpu_state.h_max_disp_sq, gpu_state.d_max_disp_sq,
        sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Step 4: record event to signal completion
    CUDA_CHECK(cudaEventRecord(gpu_state.disp_event, stream));
    gpu_state.async_readback_pending = true;

    // Step 5: Verlet Buffer Monitor
    if (config.neighbor.verlet_buffer_enabled) {
        if (!gpu_state.verlet_readback_pending) {
            CUDA_CHECK(cudaMemsetAsync(gpu_state.d_verlet_disp_scaled, 0, sizeof(int32_t), stream));
            lbvh_verlet_displacement_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.last_pos_x, state.particles.last_pos_y, state.particles.last_pos_z,
                gpu_state.d_verlet_disp_scaled, N);

            CUDA_CHECK(cudaMemcpyAsync(gpu_state.h_pinned_verlet_disp, gpu_state.d_verlet_disp_scaled,
                                       sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(gpu_state.verlet_copy_event, stream));
            gpu_state.verlet_readback_pending = true;
        }
    }
}

#include "lbvh_sph.cuh"

} // namespace lbvh
} // namespace cudadem

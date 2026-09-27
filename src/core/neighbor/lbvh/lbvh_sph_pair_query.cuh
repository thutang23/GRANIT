#pragma once
// Included inside cudadem::lbvh. Shared by the solver and GPU pair-set regression.

__global__ void sph_find_pairs_kernel(
    const double* x, const double* y, const double* z, const float* h,
    const LBVHNode* nodes, const LBVHAABB* boxes, const uint32_t* order,
    ContactPair* pairs, unsigned long long* count, int32_t capacity,
    int32_t n, int32_t num_local, float cutoff, float skin, int32_t* overflow) {
  const int32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= n) return;
  const uint32_t i = order[rank];
  // All n leaves remain searchable, but only owned particles initiate queries.
  // With locals in [0, num_local), j > i still emits each owned pair once.
  if (i >= static_cast<uint32_t>(num_local)) return;
  const float px = float(x[i]), py = float(y[i]), pz = float(z[i]);
  // SPH tree leaves and queries each contribute half of K*h. The inflation
  // covers rounding in the exact half-sum and float distance predicate.
  // Fast-math squared distances can flush to zero even for distinct centers.
  // Cover the same absolute underflow region as the RT proxy bounds.
  const float r = nextafterf((0.5f * cutoff * h[i] + skin) * (1.0f + 8.0f * FLT_EPSILON)
                             + 2.0f * sqrtf(FLT_MIN), CUDART_INF_F);
  LBVHAABB query;
  query.lower = make_float4(nextafterf(px-r, -CUDART_INF_F),
                           nextafterf(py-r, -CUDART_INF_F),
                           nextafterf(pz-r, -CUDART_INF_F), 0);
  query.upper = make_float4(nextafterf(px+r, CUDART_INF_F),
                           nextafterf(py+r, CUDART_INF_F),
                           nextafterf(pz+r, CUDART_INF_F), 0);
  uint32_t stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top) {
    const LBVHNode node = nodes[stack[--top]];
    const uint32_t children[2] = {node.left_idx, node.right_idx};
    for (int side = 0; side < 2; ++side) {
      const uint32_t child = children[side];
      if (child == 0xFFFFFFFF || !lbvh_intersects(query, boxes[child])) continue;
      const uint32_t j = nodes[child].object_idx;
      if (j != 0xFFFFFFFF) {
        if (j <= i) continue;
        if (sph_pair_candidate(px-float(x[j]), py-float(y[j]), pz-float(z[j]),
                               h[i], h[j], cutoff, skin)) {
          // Active lanes reserve a contiguous range with one global atomic.
          // Each accepted pair still gets a slot; capacity retry reads the
          // complete count even when the current buffer cannot hold them all.
          const unsigned int mask = __activemask();
          const int lane = threadIdx.x & 31;
          const int leader = __ffs(mask) - 1;
          unsigned long long base = 0;
          if (lane == leader)
            base = atomicAdd(count, static_cast<unsigned long long>(__popc(mask)));
          base = __shfl_sync(mask, base, leader);
          const unsigned int preceding = mask & ((1u << lane) - 1u);
          const unsigned long long edge = base + __popc(preceding);
          if (edge < static_cast<unsigned long long>(capacity)) pairs[edge] = ContactPair(i, j);
        }
      } else if (top < 128) {
        stack[top++] = child;
      } else {
        atomicExch(overflow, 1);
      }
    }
  }
}

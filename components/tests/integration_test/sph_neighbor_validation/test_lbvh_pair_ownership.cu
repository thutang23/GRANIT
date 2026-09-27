// Standalone dense SPH LBVH pair regression with a shuffled leaf order.
#include "src/core/neighbor/lbvh/lbvh_types.h"
#include "src/core/neighbor/sph_pair.cuh"
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace cudadem::lbvh {
#include "src/core/neighbor/lbvh/lbvh_leaf_bounds.cuh"
#include "src/core/neighbor/lbvh/lbvh_sph_pair_query.cuh"
}

using cudadem::ContactPair;
using cudadem::lbvh::LBVHAABB;
using cudadem::lbvh::LBVHNode;
constexpr int N = 96;
constexpr int MAX_PAIRS = N * (N - 1) / 2;

#define CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  std::exit(1); } } while (0)

__global__ void brute_pairs(const double* x, const double* y, const double* z,
                            const float* h, ContactPair* pairs,
                            unsigned long long* count, int num_local) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_local) return;
  for (int j = i + 1; j < N; ++j) {
    if (cudadem::sph_pair_candidate(float(x[i])-float(x[j]),
                                    float(y[i])-float(y[j]),
                                    float(z[i])-float(z[j]),
                                    h[i], h[j], 2.0f, 0.0f)) {
      const auto slot = atomicAdd(count, 1ULL);
      pairs[slot] = ContactPair(i, j);
    }
  }
}

static std::vector<std::pair<int, int>> read_pairs(ContactPair* device,
                                                    unsigned long long count) {
  if (count > MAX_PAIRS) std::abort();
  std::vector<ContactPair> raw(count);
  if (count) CHECK(cudaMemcpy(raw.data(), device, count*sizeof(ContactPair), cudaMemcpyDeviceToHost));
  std::vector<std::pair<int, int>> result;
  for (const auto p : raw) result.emplace_back(p.i, p.j);
  std::sort(result.begin(), result.end());
  return result;
}

int main() {
  std::array<double, N> x{}, y{}, z{};
  std::array<float, N> h{}, support{};
  std::array<uint32_t, N> order{};
  for (int i = 0; i < N; ++i) {
    x[i] = 0.001 * (i % 8);
    y[i] = 0.001 * (i / 8);
    h[i] = 0.1f;
    support[i] = 2.0f * h[i];
    // 37 and 96 are coprime. Owned and ghost ranks are interleaved.
    order[i] = (i * 37) % N;
  }

  double *dx, *dy, *dz;
  float *dh, *dsupport;
  uint32_t* dorder;
  LBVHNode* dnodes;
  LBVHAABB* dboxes;
  ContactPair *found, *reference;
  unsigned long long *found_count, *reference_count;
  int32_t* overflow;
  CHECK(cudaMalloc(&dx, N*sizeof(double)));
  CHECK(cudaMalloc(&dy, N*sizeof(double)));
  CHECK(cudaMalloc(&dz, N*sizeof(double)));
  CHECK(cudaMalloc(&dh, N*sizeof(float)));
  CHECK(cudaMalloc(&dsupport, N*sizeof(float)));
  CHECK(cudaMalloc(&dorder, N*sizeof(uint32_t)));
  CHECK(cudaMalloc(&dnodes, (2*N-1)*sizeof(LBVHNode)));
  CHECK(cudaMalloc(&dboxes, (2*N-1)*sizeof(LBVHAABB)));
  CHECK(cudaMalloc(&found, MAX_PAIRS*sizeof(ContactPair)));
  CHECK(cudaMalloc(&reference, MAX_PAIRS*sizeof(ContactPair)));
  CHECK(cudaMalloc(&found_count, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&reference_count, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&overflow, sizeof(int32_t)));
  CHECK(cudaMemcpy(dx, x.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, y.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dz, z.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dh, h.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dsupport, support.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dorder, order.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice));
  cudadem::lbvh::lbvh_refresh_leaf_aabbs_kernel<<<2, 64>>>(
      dx, dy, dz, dsupport, dorder, dboxes, N, N-1, 0.0f, 0.5f);
  CHECK(cudaDeviceSynchronize());

  std::array<LBVHAABB, 2*N-1> boxes{};
  std::array<LBVHNode, 2*N-1> nodes{};
  CHECK(cudaMemcpy(boxes.data(), dboxes, sizeof(boxes), cudaMemcpyDeviceToHost));
  for (int k = 0; k < N; ++k)
    nodes[N-1+k] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, order[k]};
  for (int k = N-2; k >= 0; --k) {
    const uint32_t left = N-1+k;
    const uint32_t right = k == N-2 ? 2*N-2 : k+1;
    nodes[k] = {0xFFFFFFFFu, left, right, 0xFFFFFFFFu};
    boxes[k] = cudadem::lbvh::lbvh_merge(boxes[left], boxes[right]);
  }
  CHECK(cudaMemcpy(dnodes, nodes.data(), sizeof(nodes), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dboxes, boxes.data(), sizeof(boxes), cudaMemcpyHostToDevice));

  for (int num_local : {N, 23, 1, 0}) {
    CHECK(cudaMemset(reference_count, 0, sizeof(unsigned long long)));
    brute_pairs<<<2, 64>>>(dx, dy, dz, dh, reference, reference_count, num_local);
    CHECK(cudaDeviceSynchronize());
    unsigned long long actual = 0, expected = 0;
    CHECK(cudaMemcpy(&expected, reference_count, sizeof(expected), cudaMemcpyDeviceToHost));
    const unsigned long long dense_count = MAX_PAIRS -
        (N-num_local)*(N-num_local-1)/2;
    if (expected != dense_count) {
      std::fprintf(stderr, "Dense oracle count mismatch with %d owned particles\n", num_local);
      return 1;
    }

    // A short buffer must report the complete count and leave its guard intact.
    ContactPair guard(123456, 789012);
    CHECK(cudaMemcpy(found + 7, &guard, sizeof(guard), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(found_count, 0, sizeof(unsigned long long)));
    CHECK(cudaMemset(overflow, 0, sizeof(int32_t)));
    cudadem::lbvh::sph_find_pairs_kernel<<<2, 64>>>(
        dx, dy, dz, dh, dnodes, dboxes, dorder, found, found_count,
        7, N, num_local, 2.0f, 0.0f, overflow);
    CHECK(cudaDeviceSynchronize());
    ContactPair after(0, 0);
    CHECK(cudaMemcpy(&after, found + 7, sizeof(after), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&actual, found_count, sizeof(actual), cudaMemcpyDeviceToHost));
    if (actual != expected || after.i != guard.i || after.j != guard.j) {
      std::fprintf(stderr, "Dense SPH LBVH capacity/count mismatch with %d owned particles\n", num_local);
      return 1;
    }

    // Match production's retry with enough space against the exact GPU oracle.
    CHECK(cudaMemset(found_count, 0, sizeof(unsigned long long)));
    cudadem::lbvh::sph_find_pairs_kernel<<<2, 64>>>(
        dx, dy, dz, dh, dnodes, dboxes, dorder, found, found_count,
        MAX_PAIRS, N, num_local, 2.0f, 0.0f, overflow);
    CHECK(cudaDeviceSynchronize());
    int32_t overflowed = 0;
    CHECK(cudaMemcpy(&actual, found_count, sizeof(actual), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&overflowed, overflow, sizeof(overflowed), cudaMemcpyDeviceToHost));
    if (overflowed || actual != expected ||
        read_pairs(found, actual) != read_pairs(reference, expected)) {
      std::fprintf(stderr, "Dense SPH LBVH pair-set mismatch with %d owned particles\n", num_local);
      return 1;
    }
  }
  std::puts("Dense SPH LBVH owned pair sets, zero-owned case, and capacity retry count match oracle");
  return 0;
}

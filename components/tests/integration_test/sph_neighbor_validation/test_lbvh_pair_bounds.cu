// Standalone CUDA regression for the production SPH LBVH leaf and pair kernels.
// Compile: nvcc -std=c++17 -O3 --use_fast_math -I. test_lbvh_pair_bounds.cu -o test_lbvh_pair_bounds
#include "src/core/neighbor/lbvh/lbvh_types.h"
#include "src/core/neighbor/sph_pair.cuh"
#include "src/core/sph/sph_neighbor_query.cuh"
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cudadem::lbvh {
#include "src/core/neighbor/lbvh/lbvh_leaf_bounds.cuh"
#include "src/core/neighbor/lbvh/lbvh_sph_pair_query.cuh"
}

using cudadem::ContactPair;
using cudadem::lbvh::LBVHAABB;
using cudadem::lbvh::LBVHNode;
constexpr int N = 8;

#define CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  std::exit(1); } } while (0)

__global__ void brute_pairs(const double* x, const double* y, const double* z,
                            const float* h, ContactPair* pairs,
                            unsigned long long* count, float cutoff, float skin) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  for (int j = i + 1; j < N; ++j) {
    if (cudadem::sph_pair_candidate(float(x[i])-float(x[j]),
                                     float(y[i])-float(y[j]),
                                     float(z[i])-float(z[j]),
                                     h[i], h[j], cutoff, skin)) {
      const auto slot = atomicAdd(count, 1ULL);
      pairs[slot] = ContactPair(i, j);
    }
  }
}

__global__ void gravity_fallback_pairs(const double* x, const double* y, const double* z,
                                      const float* h, cudadem::NeighborData neighbor,
                                      unsigned char* visited) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  cudadem::for_each_sph_neighbor_lbvh(i, float(x[i]), float(y[i]), float(z[i]),
      h[i], neighbor, x, y, z, h, 2.0f,
      [&](int j, float, float, float, float) {
        const double ddx = x[i]-x[j], ddy = y[i]-y[j], ddz = z[i]-z[j];
        const double r2 = ddx*ddx + ddy*ddy + ddz*ddz;
        const double hij = 0.5*(double(h[i])+double(h[j]));
        if (r2 > 0.0 && r2 < 4.0*hij*hij) visited[i*N+j] = 1;
      });
}

static std::vector<std::pair<int, int>> read_pairs(ContactPair* device,
                                                    unsigned long long* device_count) {
  unsigned long long count = 0;
  CHECK(cudaMemcpy(&count, device_count, sizeof(count), cudaMemcpyDeviceToHost));
  if (count > N*(N-1)/2) std::abort();
  std::vector<ContactPair> raw(count);
  if (count) CHECK(cudaMemcpy(raw.data(), device, count*sizeof(ContactPair), cudaMemcpyDeviceToHost));
  std::vector<std::pair<int, int>> out;
  for (auto p : raw) out.emplace_back(p.i, p.j);
  std::sort(out.begin(), out.end());
  return out;
}

int main() {
  const float cutoff = 2.0f;
  const float edge = cutoff * (0.5f * (0.005f + 0.025f));
  const float inside = nextafterf(edge, 0.0f);
  const float outside = nextafterf(edge, INFINITY);
  std::array<double, N> x{0.0, inside, 0.0, outside, 0.0, inside, 0.0, outside};
  std::array<double, N> y{0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6};
  std::array<double, N> z{};
  std::array<float, N> h{0.005f, 0.025f, 0.025f, 0.005f,
                         0.005f, 0.025f, 0.025f, 0.005f};
  std::array<float, N> support{};
  std::array<uint32_t, N> order{};
  for (int i = 0; i < N; ++i) { support[i] = cutoff*h[i]; order[i] = i; }

  double *dx, *dy, *dz;
  float *dh, *dsupport;
  uint32_t* dorder;
  LBVHNode* dnodes;
  LBVHAABB* dboxes;
  ContactPair *found, *reference;
  unsigned long long *found_count, *reference_count;
  int32_t* overflow;
  unsigned char* dvisited;
  CHECK(cudaMalloc(&dx, N*sizeof(double)));
  CHECK(cudaMalloc(&dy, N*sizeof(double)));
  CHECK(cudaMalloc(&dz, N*sizeof(double)));
  CHECK(cudaMalloc(&dh, N*sizeof(float)));
  CHECK(cudaMalloc(&dsupport, N*sizeof(float)));
  CHECK(cudaMalloc(&dorder, N*sizeof(uint32_t)));
  CHECK(cudaMalloc(&dnodes, (2*N-1)*sizeof(LBVHNode)));
  CHECK(cudaMalloc(&dboxes, (2*N-1)*sizeof(LBVHAABB)));
  CHECK(cudaMalloc(&found, 28*sizeof(ContactPair)));
  CHECK(cudaMalloc(&reference, 28*sizeof(ContactPair)));
  CHECK(cudaMalloc(&found_count, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&reference_count, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&overflow, sizeof(int32_t)));
  CHECK(cudaMalloc(&dvisited, N*N));

  for (int phase = 0; phase < 4; ++phase) {
    // Phase 1 refits Morton-ordered leaves after h and position change;
    // phase 2 tests FTZ underflow: dist^2 rounds to zero despite dist >> h.
    if (phase == 1) {
      h[0] = 0.025f; support[0] = cutoff*h[0];
      x[3] = double(nextafterf(outside, 0.0f));
      std::reverse(order.begin(), order.end());
    } else if (phase == 2) {
      x[1] = 1.0e-22;
      h[0] = h[1] = 1.0e-25f;
      support[0] = cutoff*h[0];
      support[1] = cutoff*h[1];
    } else if (phase == 3) {
      // Double positions differ inside gravity's support while their float
      // centers coincide at this scale. Tree K=3 exceeds gravity query K=2.
      x[0] = 1.0e6;
      x[1] = 1.0e6 + 0.02999999;
      h[0] = 0.005f; h[1] = 0.025f;
      for (int i = 0; i < N; ++i) support[i] = 3.0f*h[i];
    }
    CHECK(cudaMemcpy(dx, x.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy, y.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dz, z.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dh, h.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsupport, support.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dorder, order.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (phase == 0)
      cudadem::lbvh::lbvh_compute_aabbs_kernel<<<1, 32>>>(
          dx, dy, dz, dsupport, dboxes, N, N-1, 0.5f);
    else
      cudadem::lbvh::lbvh_refresh_leaf_aabbs_kernel<<<1, 32>>>(
          dx, dy, dz, dsupport, dorder, dboxes, N, N-1, 0.0f, 0.5f);
    CHECK(cudaDeviceSynchronize());

    std::array<LBVHAABB, 2*N-1> boxes{};
    std::array<LBVHNode, 2*N-1> nodes{};
    CHECK(cudaMemcpy(boxes.data(), dboxes, sizeof(boxes), cudaMemcpyDeviceToHost));
    for (int k = 0; k < N; ++k)
      nodes[N-1+k] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, order[k]};
    for (int k = N-2; k >= 0; --k) {
      const uint32_t left = N-1+k;
      const uint32_t right = k == N-2 ? N-1+(N-1) : k+1;
      nodes[k] = {0xFFFFFFFFu, left, right, 0xFFFFFFFFu};
      boxes[k] = cudadem::lbvh::lbvh_merge(boxes[left], boxes[right]);
    }
    CHECK(cudaMemcpy(dnodes, nodes.data(), sizeof(nodes), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dboxes, boxes.data(), sizeof(boxes), cudaMemcpyHostToDevice));

    for (float skin : {0.0f, 0.002f}) {
      CHECK(cudaMemset(found_count, 0, sizeof(unsigned long long)));
      CHECK(cudaMemset(reference_count, 0, sizeof(unsigned long long)));
      CHECK(cudaMemset(overflow, 0, sizeof(int32_t)));
      cudadem::lbvh::sph_find_pairs_kernel<<<1, 32>>>(
          dx, dy, dz, dh, dnodes, dboxes, dorder, found, found_count,
          28, N, cutoff, skin, overflow);
      brute_pairs<<<1, 32>>>(dx, dy, dz, dh, reference, reference_count, cutoff, skin);
      CHECK(cudaDeviceSynchronize());
      int32_t overflowed = 0;
      CHECK(cudaMemcpy(&overflowed, overflow, sizeof(overflowed), cudaMemcpyDeviceToHost));
      if (overflowed || read_pairs(found, found_count) != read_pairs(reference, reference_count)) {
        std::fprintf(stderr, "SPH LBVH pair-set mismatch at phase %d, skin %g\n", phase, skin);
        return 1;
      }
    }
    if (phase == 3) {
      cudadem::NeighborData neighbor{};
      neighbor.nodes = dnodes;
      neighbor.aabbs = dboxes;
      neighbor.num_particles = N;
      CHECK(cudaMemset(dvisited, 0, N*N));
      gravity_fallback_pairs<<<1, 32>>>(dx, dy, dz, dh, neighbor, dvisited);
      CHECK(cudaDeviceSynchronize());
      std::array<unsigned char, N*N> visited{};
      CHECK(cudaMemcpy(visited.data(), dvisited, N*N, cudaMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        const double ddx = x[i]-x[j], ddy = y[i]-y[j], ddz = z[i]-z[j];
        const double r2 = ddx*ddx + ddy*ddy + ddz*ddz;
        const double hij = 0.5*(double(h[i])+double(h[j]));
        const bool expected = i != j && r2 > 0.0 && r2 < 4.0*hij*hij;
        if (bool(visited[i*N+j]) != expected) {
          std::fprintf(stderr, "gravity LBVH fallback mismatch at %d -> %d\n", i, j);
          return 1;
        }
      }
    }
  }
  std::puts("SPH LBVH GPU pair sets and double gravity fallback match brute force");
  return 0;
}

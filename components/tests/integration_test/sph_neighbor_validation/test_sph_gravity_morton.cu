// Exercise the production SPH gravity correction kernel through the solver link.
// The Morton order deliberately puts owned targets behind interleaved ghosts.
#include "src/core/constant/system_initialize.h"
#include "src/core/neighbor/lbvh/lbvh_types.h"
#include "src/core/sph/sph_gravity_kernel.cuh"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cudadem {
__global__ void sph_gravity_softening_correction_kernel(
    ParticleSoA, NeighborData, double, double, int, bool, int, bool);
}

#define CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

template <typename T> struct DeviceArray {
  T* ptr = nullptr;
  explicit DeviceArray(size_t n) { CHECK(cudaMalloc(&ptr, std::max(size_t(1), n)*sizeof(T))); }
  ~DeviceArray() { if (ptr) cudaFree(ptr); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  void set(const std::vector<T>& values) {
    if (!values.empty()) CHECK(cudaMemcpy(ptr, values.data(),
        values.size()*sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get(size_t n) const {
    std::vector<T> out(n);
    if (n) CHECK(cudaMemcpy(out.data(), ptr, n*sizeof(T), cudaMemcpyDeviceToHost));
    return out;
  }
};

struct Case {
  std::vector<double> x, y, z;
  std::vector<float> h, mass;
  std::vector<uint32_t> order;
  int owned;
};

using Vec3 = std::array<double, 3>;

static Vec3 brute_delta(const Case& c, int i, bool outer, double G, double epsilon) {
  Vec3 result{};
  for (int j = 0; j < static_cast<int>(c.x.size()); ++j) {
    const double dx = c.x[j]-c.x[i], dy = c.y[j]-c.y[i], dz = c.z[j]-c.z[i];
    const double r2 = dx*dx + dy*dy + dz*dz;
    const double h = 0.5*(static_cast<double>(c.h[i])+c.h[j]);
    if (!(r2 > 0.0) || r2 >= 4.0*h*h) continue;
    const double r = std::sqrt(r2);
    const double inverse = 1.0/std::sqrt(std::max(r2, epsilon*epsilon));
    const double difference =
        cudadem::gravity_softening_force_factor_double(r, h) - inverse*inverse*inverse;
    const double scale = G*(outer ? 1.0 : c.mass[i])*c.mass[j]*difference;
    result[0] += scale*dx; result[1] += scale*dy; result[2] += scale*dz;
  }
  return result;
}

static bool close(double got, double want) {
  return std::abs(got-want) <= 2e-10*std::max(1.0, std::abs(want));
}

static void run_case(const Case& c, const char* label) {
  const int n = static_cast<int>(c.x.size());
  if (c.y.size()!=c.x.size() || c.z.size()!=c.x.size() ||
      c.h.size()!=c.x.size() || c.mass.size()!=c.x.size() ||
      c.order.size()!=c.x.size() || c.owned>n) std::abort();
  auto permutation = c.order;
  std::sort(permutation.begin(), permutation.end());
  for (int i=0; i<n; ++i) if (permutation[i] != static_cast<uint32_t>(i)) std::abort();

  DeviceArray<double> dx(n), dy(n), dz(n);
  DeviceArray<float> dh(n), dm(n);
  DeviceArray<uint32_t> dorder(n);
  DeviceArray<cudadem::lbvh::LBVHNode> dnodes(std::max(1,2*n-1));
  DeviceArray<cudadem::lbvh::LBVHAABB> dboxes(std::max(1,2*n-1));
  DeviceArray<double> fx(n), fy(n), fz(n), gx(n), gy(n), gz(n);
  dx.set(c.x); dy.set(c.y); dz.set(c.z); dh.set(c.h); dm.set(c.mass);
  dorder.set(c.order);

  if (n>1) {
    std::vector<cudadem::lbvh::LBVHNode> nodes(2*n-1);
    std::vector<cudadem::lbvh::LBVHAABB> boxes(2*n-1);
    for (int rank=0; rank<n; ++rank) {
      const int i=c.order[rank], leaf=n-1+rank;
      nodes[leaf]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,static_cast<uint32_t>(i)};
      // Outward boxes contain the exact particle positions and all support.
      // The tree shape, independent of Morton target order, is fixed for both launches.
      const float pad=c.h[i]+0.01f;
      boxes[leaf].lower=make_float4(float(c.x[i])-pad,float(c.y[i])-pad,
                                    float(c.z[i])-pad,0.0f);
      boxes[leaf].upper=make_float4(float(c.x[i])+pad,float(c.y[i])+pad,
                                    float(c.z[i])+pad,0.0f);
    }
    for (int k=n-2; k>=0; --k) {
      const uint32_t left=n-1+k, right=k==n-2 ? 2*n-2 : k+1;
      nodes[k]={0xFFFFFFFFu,left,right,0xFFFFFFFFu};
      boxes[k]=cudadem::lbvh::lbvh_merge(boxes[left], boxes[right]);
    }
    dnodes.set(nodes); dboxes.set(boxes);
  }

  cudadem::ParticleSoA p{};
  p.num_particles=n; p.num_local=c.owned; p.num_ghost=n-c.owned;
  p.pos_x=dx.ptr; p.pos_y=dy.ptr; p.pos_z=dz.ptr;
  p.smoothing_length=dh.ptr; p.mass=dm.ptr;
  p.force_x=fx.ptr; p.force_y=fy.ptr; p.force_z=fz.ptr;
  p.gravity_force_x=gx.ptr; p.gravity_force_y=gy.ptr; p.gravity_force_z=gz.ptr;
  cudadem::NeighborData neighbors;
  neighbors.num_particles=n;
  neighbors.active_method=cudadem::NeighborMethod::LBVH;
  neighbors.nodes=n>1 ? dnodes.ptr : nullptr;
  neighbors.aabbs=n>1 ? dboxes.ptr : nullptr;
  neighbors.sorted_indices=n>1 ? dorder.ptr : nullptr;
  // A stale hydro adjacency cannot be used by this gravity stage.
  neighbors.sph_adjacency_valid=true;

  constexpr double G=2.3, epsilon=0.2;
  for (bool outer : {false,true}) {
    std::array<std::vector<double>,6> initial;
    for (int component=0; component<6; ++component) {
      initial[component].resize(n);
      for (int i=0; i<n; ++i) initial[component][i]=
          (component+1)*100.0 + i*0.125;
    }
    const auto reset=[&] {
      fx.set(initial[0]); fy.set(initial[1]); fz.set(initial[2]);
      gx.set(initial[3]); gy.set(initial[4]); gz.set(initial[5]);
    };
    const auto snapshot=[&] {
      return std::array<std::vector<double>,6>{
          fx.get(n),fy.get(n),fz.get(n),gx.get(n),gy.get(n),gz.get(n)};
    };
    const auto launch=[&](bool morton) {
      const int queries=morton ? n : c.owned;
      if (queries) {
        cudadem::sph_gravity_softening_correction_kernel<<<(queries+127)/128,128>>>(
            p,neighbors,G,epsilon,c.owned,outer,queries,morton);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
      }
      return snapshot();
    };
    reset();
    const auto canonical=launch(false);
    reset();
    const auto sorted=launch(n>1);
    for (int component=0; component<6; ++component) {
      for (int i=0; i<n; ++i) {
        if (std::memcmp(&canonical[component][i], &sorted[component][i],
                        sizeof(double)) != 0) {
          std::fprintf(stderr,"%s %s component %d index %d: canonical %.17g Morton %.17g\n",
              label,outer?"outer":"inner",component,i,
              canonical[component][i],sorted[component][i]);
          std::exit(1);
        }
        const bool active=outer ? component>=3 : component<3;
        const double delta=(i<c.owned && active)
            ? brute_delta(c,i,outer,G,epsilon)[component%3] : 0.0;
        const double expected=initial[component][i]+delta;
        if (!close(sorted[component][i],expected)) {
          std::fprintf(stderr,"%s %s component %d index %d: got %.17g expected %.17g\n",
              label,outer?"outer":"inner",component,i,
              sorted[component][i],expected);
          std::exit(1);
        }
        if ((i>=c.owned || !active) &&
            sorted[component][i] != initial[component][i]) {
          std::fprintf(stderr,"%s ghost/inactive force buffer modified\n",label);
          std::exit(1);
        }
      }
    }
  }
  std::printf("%s: canonical, Morton, brute-force, and ghost sentinels pass\n",label);
}

int main() {
  run_case({{0.0,0.4,1.0,2.5,0.25,0.85,2.4,5.0},
            {0.0,0.1,0.0,0.0,0.1,-0.1,0.0,0.0},
            {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
            {0.35f,0.65f,0.45f,0.40f,0.50f,0.50f,0.30f,0.20f},
            {2.0f,3.0f,4.0f,5.0f,1.5f,2.5f,3.5f,4.5f},
            {5,0,6,1,7,2,4,3},4},"interleaved owned/ghost");
  Case multiple_blocks;
  multiple_blocks.owned=23;
  for (int i=0; i<270; ++i) {
    multiple_blocks.x.push_back(0.01*i);
    multiple_blocks.y.push_back(0.001*(i%3));
    multiple_blocks.z.push_back(0.0);
    multiple_blocks.h.push_back(0.025f+0.001f*(i%5));
    multiple_blocks.mass.push_back(1.0f+0.1f*(i%7));
    multiple_blocks.order.push_back((i*37)%270);
  }
  run_case(multiple_blocks,"three Morton blocks with late owned targets");
  run_case({{0.0,0.3},{0.0,0.0},{0.0,0.0},
            {0.5f,0.5f},{2.0f,3.0f},{1,0},0},"zero-owned ghosts");
  run_case({{0.0},{0.0},{0.0},{0.5f},{2.0f},{0},1},"single-particle fallback");
  run_case({{},{},{},{},{},{},0},"empty-domain no-launch");
}

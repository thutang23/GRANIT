#pragma once

#include "../neighbor_types.h"
#include "contact_dedup_workspace.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cstdint>
#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

namespace cudadem::lbvh {

struct ContactDedupRange {
  explicit ContactDedupRange(const char* name) {
#ifdef CUDADEM_ENABLE_PROFILING
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~ContactDedupRange() {
#ifdef CUDADEM_ENABLE_PROFILING
    nvtxRangePop();
#endif
  }
};

inline int32_t deduplicate_contacts(ContactPair* contacts, int32_t raw_count,
                                    int32_t* d_num_contacts, cudaStream_t stream,
                                    ContactDedupWorkspace& workspace) {
  if (raw_count <= 1) return raw_count;
  ContactDedupAllocator allocator{&workspace};
  {
    ContactDedupRange range("LBVH: Deduplicate Sort");
    thrust::sort(thrust::cuda::par(allocator).on(stream),
        contacts, contacts + raw_count,
        [] __device__ (const ContactPair& a, const ContactPair& b) {
          return (a.i < b.i) || (a.i == b.i && a.j < b.j);
        });
  }
  ContactPair* new_end;
  {
    ContactDedupRange range("LBVH: Deduplicate Unique");
    new_end = thrust::unique(thrust::cuda::par(allocator).on(stream),
        contacts, contacts + raw_count,
        [] __device__ (const ContactPair& a, const ContactPair& b) {
          return a.i == b.i && a.j == b.j;
        });
  }
  const int32_t new_count = static_cast<int32_t>(new_end - contacts);
  {
    ContactDedupRange range("LBVH: Deduplicate Count Upload");
    ContactDedupWorkspace::checked(cudaMemcpyAsync(
        d_num_contacts, &new_count, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  }
  return new_count;
}

} // namespace cudadem::lbvh

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem::lbvh {

struct MaxSupportReduction {
  void* temp = nullptr;
  size_t temp_bytes = 0;
  int32_t capacity = 0;
  float* device_result = nullptr;
  float* host_result = nullptr;

  static void checked(cudaError_t error);
  void initialize();
  void release();
  void reserve(const float* values, int32_t n, cudaStream_t stream);
  void enqueue(const float* support_extent, const float* radius, int32_t n,
               cudaStream_t stream);
  float wait(int32_t n, cudaStream_t stream);
};

} // namespace cudadem::lbvh

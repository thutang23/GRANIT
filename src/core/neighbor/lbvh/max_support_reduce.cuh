#pragma once

#include "max_support_reduce.h"
#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>

namespace cudadem::lbvh {

// Match cub::Max's comparison, including ties and NaNs.
struct SupportMaximum {
  __host__ __device__ float operator()(float a, float b) const {
    return b > a ? b : a;
  }
};

inline void MaxSupportReduction::checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

inline void MaxSupportReduction::initialize() {
  checked(cudaMalloc(&device_result, sizeof(float)));
  checked(cudaMallocHost(&host_result, sizeof(float)));
}

inline void MaxSupportReduction::release() {
  if (temp) checked(cudaFree(temp));
  if (device_result) checked(cudaFree(device_result));
  if (host_result) checked(cudaFreeHost(host_result));
  temp = nullptr; temp_bytes = 0; capacity = 0;
  device_result = nullptr; host_result = nullptr;
}

inline void MaxSupportReduction::reserve(const float* values, int32_t n, cudaStream_t stream) {
  if (n <= capacity) return;
  if (!values || !device_result) throw std::runtime_error("LBVH maximum support reserve is uninitialized");
  const int64_t grown = static_cast<int64_t>(capacity) + capacity / 2;
  const int32_t target = static_cast<int32_t>(grown > std::numeric_limits<int32_t>::max()
      ? n : (n > grown ? n : grown));
  size_t needed = 0;
  checked(cub::DeviceReduce::Reduce(nullptr, needed, values, device_result,
      target, SupportMaximum{}, 0.0f, stream));
  if (needed > temp_bytes) {
    void* larger = nullptr;
    checked(cudaMalloc(&larger, needed));
    if (temp) checked(cudaFree(temp));
    temp = larger; temp_bytes = needed;
  }
  capacity = target;
}

inline void MaxSupportReduction::enqueue(const float* support_extent, const float* radius,
                                          int32_t n, cudaStream_t stream) {
  if (n <= 0) { if (host_result) *host_result = 0.0f; return; }
  const float* values = support_extent ? support_extent : radius;
  if (!values || !device_result || !host_result)
    throw std::runtime_error("LBVH maximum support input or output is null");
  reserve(values, n, stream);
  checked(cub::DeviceReduce::Reduce(temp, temp_bytes, values, device_result,
      n, SupportMaximum{}, 0.0f, stream));
  checked(cudaMemcpyAsync(host_result, device_result, sizeof(float),
      cudaMemcpyDeviceToHost, stream));
}

inline float MaxSupportReduction::wait(int32_t n, cudaStream_t stream) {
  if (n <= 0) return 0.0f;
  checked(cudaStreamSynchronize(stream));
  return *host_result;
}

} // namespace cudadem::lbvh

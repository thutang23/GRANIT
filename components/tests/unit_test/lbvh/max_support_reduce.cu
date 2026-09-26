#include "../../../../src/core/neighbor/lbvh/max_support_reduce.cuh"
#include <algorithm>
#include <exception>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using cudadem::lbvh::MaxSupportReduction;

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static void exercise(int gpu) {
  MaxSupportReduction::checked(cudaSetDevice(gpu));
  cudaStream_t stream;
  MaxSupportReduction::checked(cudaStreamCreate(&stream));
  float *support = nullptr, *radius = nullptr;
  constexpr int large_n = 100000;
  std::vector<float> h_support(large_n, -3.0f), h_radius(large_n, -2.0f);
  h_support[0] = 1.0f; h_support[1] = 3.0f; h_support[2] = 2.0f; h_support[3] = 5.0f;
  h_radius[0] = 7.0f; h_radius[1] = 2.0f; h_radius[2] = 4.0f; h_radius[3] = 1.0f;
  h_support.back() = 9.0f;
  MaxSupportReduction::checked(cudaMalloc(&support, large_n * sizeof(float)));
  MaxSupportReduction::checked(cudaMalloc(&radius, large_n * sizeof(float)));
  MaxSupportReduction::checked(cudaMemcpy(support, h_support.data(), large_n * sizeof(float), cudaMemcpyHostToDevice));
  MaxSupportReduction::checked(cudaMemcpy(radius, h_radius.data(), large_n * sizeof(float), cudaMemcpyHostToDevice));

  MaxSupportReduction reduction;
  reduction.initialize();
  reduction.enqueue(support, radius, 4, stream);
  require(reduction.wait(4, stream) == 5.0f, "support must take precedence over radius");
  auto small_temp = reduction.temp;
  reduction.enqueue(support, radius, 4, stream);
  require(reduction.wait(4, stream) == 5.0f && reduction.temp == small_temp,
          "same-size call must reuse scratch");

  // Same count, changed device contents: the maximum cannot be cached.
  float replacement = 8.0f;
  MaxSupportReduction::checked(cudaMemcpyAsync(support + 2, &replacement, sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
  reduction.enqueue(support, radius, 4, stream);
  require(reduction.wait(4, stream) == 8.0f, "same-count value change was missed");

  reduction.enqueue(support, radius, large_n, stream);
  require(reduction.wait(large_n, stream) == 9.0f && reduction.capacity >= large_n,
          "large live input or scratch growth failed");
  auto large_temp = reduction.temp;
  reduction.enqueue(support, radius, 2, stream);
  require(reduction.wait(2, stream) == 3.0f && reduction.temp == large_temp,
          "shrinking input should reuse grown scratch");
  reduction.enqueue(nullptr, radius, 4, stream);
  require(reduction.wait(4, stream) == 7.0f, "radius fallback failed");

  std::vector<float> negative(large_n, -4.0f);
  MaxSupportReduction::checked(cudaMemcpyAsync(support, negative.data(), large_n * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
  reduction.enqueue(support, radius, large_n, stream);
  require(reduction.wait(large_n, stream) == 0.0f, "initial zero must dominate negative values");
  reduction.enqueue(nullptr, nullptr, 0, stream);
  require(reduction.wait(0, stream) == 0.0f, "empty input must return zero");

  reduction.release();
  require(!reduction.temp && !reduction.device_result && !reduction.host_result && !reduction.capacity,
          "release did not clear workspace");
  reduction.initialize();
  // Query a capacity larger than the live input; CUB's size query must not read it.
  reduction.reserve(support, large_n, stream);
  reduction.enqueue(nullptr, radius, 4, stream);
  require(reduction.wait(4, stream) == 7.0f, "reinitialize or reserved fallback failed");
  reduction.release();
  MaxSupportReduction::checked(cudaFree(support));
  MaxSupportReduction::checked(cudaFree(radius));
  MaxSupportReduction::checked(cudaStreamDestroy(stream));
}

int main() {
  int devices = 0;
  MaxSupportReduction::checked(cudaGetDeviceCount(&devices));
  require(devices > 0, "test requires a visible GPU");
  const int workers_count = std::min(devices, 4);
  std::vector<std::exception_ptr> errors(workers_count);
  std::vector<std::thread> workers;
  workers.reserve(workers_count);
  for (int gpu = 0; gpu < workers_count; ++gpu) {
    workers.emplace_back([gpu, &errors] {
      try { exercise(gpu); }
      catch (...) { errors[gpu] = std::current_exception(); }
    });
  }
  for (auto& worker : workers) worker.join();
  for (const auto& error : errors) if (error) std::rethrow_exception(error);
  std::printf("PASS %d-GPU max-support reduction: reuse, growth, live values, fallback, empty, lifecycle\n", workers_count);
}

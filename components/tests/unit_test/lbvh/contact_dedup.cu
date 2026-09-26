#include "../../../../src/core/neighbor/lbvh/contact_dedup.cuh"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

using cudadem::ContactPair;
using cudadem::lbvh::ContactDedupWorkspace;
using cudadem::lbvh::deduplicate_contacts;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static bool same(const ContactPair& a, const ContactPair& b) {
  return a.i == b.i && a.j == b.j && a.image_x == b.image_x &&
         a.image_y == b.image_y && a.image_z == b.image_z;
}

static size_t retained_bytes(const ContactDedupWorkspace& workspace) {
  size_t result = 0;
  for (const auto& block : workspace.blocks) result += block.bytes;
  return result;
}

static void run_case(ContactDedupWorkspace& workspace, ContactPair* device_pairs,
                     int32_t* device_count, cudaStream_t stream,
                     const std::vector<ContactPair>& input,
                     const std::vector<ContactPair>& expected) {
  auto check = ContactDedupWorkspace::checked;
  const int32_t raw_count = static_cast<int32_t>(input.size());
  check(cudaMemcpyAsync(device_count, &raw_count, sizeof(raw_count),
                        cudaMemcpyHostToDevice, stream));
  if (raw_count)
    check(cudaMemcpyAsync(device_pairs, input.data(), input.size() * sizeof(ContactPair),
                          cudaMemcpyHostToDevice, stream));
  const auto returned = deduplicate_contacts(device_pairs, raw_count, device_count,
                                             stream, workspace);
  int32_t count = -1;
  check(cudaMemcpyAsync(&count, device_count, sizeof(count),
                        cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  require(returned == static_cast<int32_t>(expected.size()), "returned count changed");
  require(count == static_cast<int32_t>(expected.size()), "device count changed");
  std::vector<ContactPair> output(count);
  if (count)
    check(cudaMemcpy(output.data(), device_pairs, output.size() * sizeof(ContactPair),
                     cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i)
    require(same(output[i], expected[i]), "pair identity or image changed");
}

static void exercise(int gpu) {
  auto check = ContactDedupWorkspace::checked;
  check(cudaSetDevice(gpu));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  constexpr int capacity = 8192;
  ContactPair* device_pairs = nullptr;
  int32_t* device_count = nullptr;
  check(cudaMalloc(&device_pairs, capacity * sizeof(ContactPair)));
  check(cudaMalloc(&device_count, sizeof(int32_t)));
  ContactDedupWorkspace workspace;

  run_case(workspace, device_pairs, device_count, stream, {}, {});
  run_case(workspace, device_pairs, device_count, stream,
           {ContactPair(7, 8, -1, 1, 0)}, {ContactPair(7, 8, -1, 1, 0)});
  require(workspace.blocks.empty(), "zero and one pairs should need no scratch");

  const std::vector<ContactPair> small = {
      {4, 5, 0, -1, 1}, {1, 9, -1, 0, 0}, {4, 5, 0, -1, 1},
      {2, 8, 1, 0, -1}, {1, 9, -1, 0, 0}, {2, 8, 1, 0, -1}};
  const std::vector<ContactPair> small_expected = {
      {1, 9, -1, 0, 0}, {2, 8, 1, 0, -1}, {4, 5, 0, -1, 1}};
  run_case(workspace, device_pairs, device_count, stream, small, small_expected);
  require(!workspace.blocks.empty(), "sort and unique did not use retained scratch");
  const size_t small_blocks = workspace.blocks.size();
  run_case(workspace, device_pairs, device_count, stream, small, small_expected);
  require(workspace.blocks.size() == small_blocks, "repeat allocated new scratch");

  std::vector<ContactPair> large;
  std::vector<ContactPair> large_expected;
  large.reserve(capacity);
  for (int i = 0; i < capacity / 2; ++i) {
    ContactPair pair(i / 64, i % 64, static_cast<int8_t>(i % 3 - 1),
                     static_cast<int8_t>((i / 3) % 3 - 1), 0);
    large.push_back(pair);
    large.push_back(pair);
    large_expected.push_back(pair);
  }
  std::reverse(large.begin(), large.end());
  run_case(workspace, device_pairs, device_count, stream, large, large_expected);
  const size_t grown_blocks = workspace.blocks.size();
  require(grown_blocks > small_blocks, "workspace growth failed");
  run_case(workspace, device_pairs, device_count, stream, small, small_expected);
  require(workspace.blocks.size() == grown_blocks, "shrinking input allocated scratch");

  // Many gradually increasing contact counts should not accumulate one
  // retained allocation per count. Check full sort/unique output each time.
  for (int n = 33; n <= 4096; n += 127) {
    std::vector<ContactPair> input, expected;
    for (int i = 0; i < n; ++i) {
      ContactPair pair(i / 2, i / 2 + 1, static_cast<int8_t>((i / 2) % 3 - 1), 0, 1);
      input.push_back(pair);
      if ((i % 2) == 0) expected.push_back(pair);
    }
    std::reverse(input.begin(), input.end());
    run_case(workspace, device_pairs, device_count, stream, input, expected);
  }
  require(retained_bytes(workspace) <= 16 * workspace.max_request,
          "gradual contact growth retained too much scratch");
  const size_t growth_blocks = workspace.blocks.size();
  run_case(workspace, device_pairs, device_count, stream, large, large_expected);
  require(workspace.blocks.size() == growth_blocks,
          "repeat after gradual growth allocated new scratch");

  // Compare tied images to the original default-allocator Thrust path. Its
  // comparator only sees (i,j), so the exact survivor is algorithm-dependent.
  const std::vector<ContactPair> tied = {
      {2, 4, -1, 0, 0}, {1, 3, 0, 0, 0}, {2, 4, 1, 0, 0}};
  ContactPair* legacy_pairs = nullptr;
  check(cudaMalloc(&legacy_pairs, tied.size() * sizeof(ContactPair)));
  check(cudaMemcpyAsync(legacy_pairs, tied.data(), tied.size() * sizeof(ContactPair),
                        cudaMemcpyHostToDevice, stream));
  thrust::sort(thrust::cuda::par.on(stream), legacy_pairs, legacy_pairs + tied.size(),
      [] __device__ (const ContactPair& a, const ContactPair& b) {
        return (a.i < b.i) || (a.i == b.i && a.j < b.j);
      });
  auto legacy_end = thrust::unique(thrust::cuda::par.on(stream),
      legacy_pairs, legacy_pairs + tied.size(),
      [] __device__ (const ContactPair& a, const ContactPair& b) {
        return a.i == b.i && a.j == b.j;
      });
  const auto legacy_count = legacy_end - legacy_pairs;
  check(cudaMemcpyAsync(device_pairs, tied.data(), tied.size() * sizeof(ContactPair),
                        cudaMemcpyHostToDevice, stream));
  const auto tied_count = deduplicate_contacts(device_pairs, 3, device_count, stream, workspace);
  check(cudaStreamSynchronize(stream));
  require(tied_count == 2, "different images of one pair were not deduplicated");
  ContactPair tied_output[2];
  ContactPair legacy_output[2];
  check(cudaMemcpy(tied_output, device_pairs, sizeof(tied_output), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(legacy_output, legacy_pairs, sizeof(legacy_output), cudaMemcpyDeviceToHost));
  require(legacy_count == tied_count && same(tied_output[0], legacy_output[0]) &&
          same(tied_output[1], legacy_output[1]),
          "tied-image output differs from original Thrust sort and unique");
  check(cudaFree(legacy_pairs));

  workspace.release();
  require(workspace.blocks.empty() && workspace.device == -1, "release retained scratch");
  run_case(workspace, device_pairs, device_count, stream, small, small_expected);
  require(workspace.device == gpu, "workspace did not rebind to its GPU");
  workspace.release();
  check(cudaFree(device_pairs));
  check(cudaFree(device_count));
  check(cudaStreamDestroy(stream));
}

static void independent_gpus() {
  auto check = ContactDedupWorkspace::checked;
  ContactDedupWorkspace first, second;
  check(cudaSetDevice(0));
  char* first_pointer = first.allocate(128);
  first.deallocate(first_pointer);
  check(cudaSetDevice(1));
  char* second_pointer = second.allocate(256);
  second.deallocate(second_pointer);
  require(first.device == 0 && second.device == 1 &&
          first.blocks.size() == 1 && second.blocks.size() == 1,
          "independent GPU workspaces lost their owners");
  bool rejected_wrong_device = false;
  try { first.allocate(16); }
  catch (const std::runtime_error&) { rejected_wrong_device = true; }
  require(rejected_wrong_device, "workspace accepted a different GPU");
  check(cudaSetDevice(0));
  first.release();
  check(cudaSetDevice(1));
  second.release();
}

static void gradual_allocator_growth() {
  auto check = ContactDedupWorkspace::checked;
  check(cudaSetDevice(0));
  ContactDedupWorkspace workspace;
  for (size_t bytes = 1; bytes <= 8192; bytes += 17) {
    char* pointer = workspace.allocate(bytes);
    workspace.deallocate(pointer);
  }
  require(retained_bytes(workspace) < 4 * workspace.max_request,
          "geometric bins did not bound retained allocation growth");
  const size_t blocks = workspace.blocks.size();
  char* best = nullptr;
  for (const auto& block : workspace.blocks)
    if (block.bytes == 512) best = block.pointer;
  require(best != nullptr, "expected 512-byte bin was not allocated");
  char* reused = workspace.allocate(300);
  require(reused == best, "allocator did not use the smallest fitting block");
  workspace.deallocate(reused);
  for (size_t bytes = 1; bytes <= 8192; bytes += 17) {
    char* pointer = workspace.allocate(bytes);
    workspace.deallocate(pointer);
  }
  require(workspace.blocks.size() == blocks, "repeated gradual growth allocated scratch");
  workspace.release();
}

int main() {
  int devices = 0;
  const auto discovery = cudaGetDeviceCount(&devices);
  if (discovery == cudaErrorNoDevice || discovery == cudaErrorInsufficientDriver ||
      discovery == cudaErrorInitializationError ||
      (discovery == cudaSuccess && devices == 0)) {
    std::fprintf(stderr, "SKIP periodic contact deduplication: no usable CUDA device\n");
    return 77;
  }
  ContactDedupWorkspace::checked(discovery);
  const int workers_count = std::min(devices, 2);
  std::vector<std::exception_ptr> errors(workers_count);
  std::vector<std::thread> workers;
  for (int gpu = 0; gpu < workers_count; ++gpu)
    workers.emplace_back([gpu, &errors] {
      try { exercise(gpu); }
      catch (...) { errors[gpu] = std::current_exception(); }
    });
  for (auto& worker : workers) worker.join();
  for (const auto& error : errors) if (error) std::rethrow_exception(error);
  gradual_allocator_growth();
  if (devices > 1) independent_gpus();
  std::printf("PASS %d-GPU periodic contact deduplication and retained scratch\n",
              std::min(devices, 2));
}

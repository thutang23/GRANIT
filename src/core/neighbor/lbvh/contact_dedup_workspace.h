#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudadem::lbvh {

// Thrust may request several temporary allocations in one algorithm. Retain
// each block until LBVH shutdown; a released block can be reused on the same
// stream after Thrust has queued its last use. The final contact-count readback
// synchronizes that stream before a later query can use this workspace again.
struct ContactDedupWorkspace {
  struct Block {
    char* pointer;
    size_t bytes;
    bool in_use;
  };

  std::vector<Block> blocks;
  int device = -1;
  size_t max_request = 0;

  ContactDedupWorkspace() = default;
  ContactDedupWorkspace(const ContactDedupWorkspace&) = delete;
  ContactDedupWorkspace& operator=(const ContactDedupWorkspace&) = delete;

  static void checked(cudaError_t result) {
    if (result != cudaSuccess)
      throw std::runtime_error(std::string("LBVH contact deduplication: ") +
                               cudaGetErrorString(result));
  }

  char* allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    int current_device = -1;
    checked(cudaGetDevice(&current_device));
    if (device >= 0 && current_device != device)
      throw std::runtime_error("LBVH contact deduplication workspace used on another GPU");
    if (bytes > max_request) max_request = bytes;
    Block* best = nullptr;
    for (auto& block : blocks)
      if (!block.in_use && block.bytes >= bytes &&
          (!best || block.bytes < best->bytes)) best = &block;
    if (best) {
      best->in_use = true;
      return best->pointer;
    }
    // A slowly rising contact count should occupy logarithmically many bins,
    // rather than retaining one exact-sized allocation for every new count.
    size_t capacity = 256;
    while (capacity < bytes) {
      if (capacity > std::numeric_limits<size_t>::max() / 2) {
        capacity = bytes;
        break;
      }
      capacity *= 2;
    }
    char* pointer = nullptr;
    checked(cudaMalloc(&pointer, capacity));
    try {
      blocks.push_back({pointer, capacity, true});
    } catch (...) {
      cudaFree(pointer);
      throw;
    }
    device = current_device;
    return pointer;
  }

  void deallocate(char* pointer) {
    if (!pointer) return;
    for (auto& block : blocks) {
      if (block.pointer == pointer) {
        if (!block.in_use)
          throw std::runtime_error("LBVH contact deduplication released a block twice");
        block.in_use = false;
        return;
      }
    }
    throw std::runtime_error("LBVH contact deduplication released an unknown block");
  }

  void release() {
    if (device >= 0) {
      int current_device = -1;
      checked(cudaGetDevice(&current_device));
      if (current_device != device)
        throw std::runtime_error("LBVH contact deduplication released on another GPU");
    }
    cudaError_t first_error = cudaSuccess;
    for (auto& block : blocks) {
      const cudaError_t result = cudaFree(block.pointer);
      if (first_error == cudaSuccess) first_error = result;
    }
    blocks.clear();
    device = -1;
    max_request = 0;
    checked(first_error);
  }
};

// Copies of Thrust's allocator must share the GPU state's retained blocks.
struct ContactDedupAllocator {
  using value_type = char;
  ContactDedupWorkspace* workspace;
  char* allocate(std::size_t bytes) { return workspace->allocate(bytes); }
  void deallocate(char* pointer, std::size_t) { workspace->deallocate(pointer); }
};

} // namespace cudadem::lbvh

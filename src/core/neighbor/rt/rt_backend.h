#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace cudadem {
struct ContactListBuffer;
namespace rt {

struct SearchInput {
    const double* pos_x;
    const double* pos_y;
    const double* pos_z;
    const float* support_extent;
    const int32_t* clump_id;
    int32_t count;
    float skin;
    // Non-null selects the SPH symmetric smoothing-length cutoff. SPH Part IDs
    // are not DEM clumps and do not suppress neighbor pairs.
    const float* smoothing_length = nullptr;
    float kernel_cutoff_factor = 2.0f;
};

// OptiX types and driver loading are confined to the implementation. The same
// interface has a CUDA-only stub, so fallback executables need no OptiX library.
class Backend {
public:
    static Backend* create(std::string& unavailable_reason);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    void update(const SearchInput& input, bool rebuild, cudaStream_t stream);
    void find_contacts(ContactListBuffer& contacts, cudaStream_t stream);

private:
    struct Impl;
    explicit Backend(Impl* impl) : impl_(impl) {}
    Impl* impl_;
};

} // namespace rt
} // namespace cudadem

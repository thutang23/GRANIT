#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace cudadem {

// Lazy initialization is serialized once, before handing out stable references.
// Each participating worker subsequently touches only its device's entry.
template<class T> class DeviceStateArray {
    std::once_flag once_;
    std::unique_ptr<T[]> values_;
    int count_ = 0;
    void initialize() {
        std::call_once(once_, [&] {
            if (cudaGetDeviceCount(&count_) != cudaSuccess || count_ < 1)
                throw std::runtime_error("Cannot enumerate CUDA device state");
            values_ = std::make_unique<T[]>(count_);
        });
    }
public:
    T& operator[](int device) {
        initialize();
        if (device < 0 || device >= count_)
            throw std::out_of_range("CUDA device state index");
        return values_[device];
    }
    int size() { initialize(); return count_; }
    T* begin() { initialize(); return values_.get(); }
    T* end() { initialize(); return values_.get() + count_; }
    void fill(const T& value) { for (auto& entry : *this) entry = value; }
};

} // namespace cudadem

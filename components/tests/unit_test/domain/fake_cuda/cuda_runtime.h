#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorPeerAccessAlreadyEnabled = 1,
    cudaErrorTooManyPeers = 2,
    cudaErrorPeerAccessUnsupported = 3,
    cudaErrorUnknown = 4,
};
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
};
struct FakeCudaStream { int device; };
struct FakeCudaEvent { int device; };
using cudaStream_t = FakeCudaStream*;
using cudaEvent_t = FakeCudaEvent*;
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDisableTiming = 2;

namespace fake_cuda {
inline thread_local int current_device = 0;
inline thread_local cudaError_t last_error = cudaSuccess;
inline int can_access_calls = 0;
inline int enable_calls = 0;
inline int peer_copies = 0;
inline int staged_d2h = 0;
inline int staged_h2d = 0;
inline int accessible = 1;
inline cudaError_t enable_result = cudaSuccess;

inline void reset() {
    can_access_calls = enable_calls = peer_copies = staged_d2h = staged_h2d = 0;
    accessible = 1;
    enable_result = cudaSuccess;
    last_error = cudaSuccess;
}
} // namespace fake_cuda

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
    case cudaSuccess: return "success";
    case cudaErrorPeerAccessAlreadyEnabled: return "already enabled";
    case cudaErrorTooManyPeers: return "too many peers";
    case cudaErrorPeerAccessUnsupported: return "peer access unsupported";
    default: return "unknown CUDA error";
    }
}
inline cudaError_t cudaGetDevice(int* device) {
    *device = fake_cuda::current_device;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) {
    fake_cuda::current_device = device;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceGetPCIBusId(char* bus_id, int length, int device) {
    if (length <= 0 || device < 0 || device > 255) return cudaErrorUnknown;
    const int written = std::snprintf(bus_id, static_cast<size_t>(length),
                                      "0000:%02x:00.0", device);
    return written >= 0 && written < length ? cudaSuccess : cudaErrorUnknown;
}
inline cudaError_t cudaDeviceCanAccessPeer(int* accessible, int, int) {
    ++fake_cuda::can_access_calls;
    *accessible = fake_cuda::accessible;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceEnablePeerAccess(int, unsigned) {
    ++fake_cuda::enable_calls;
    fake_cuda::last_error = fake_cuda::enable_result;
    return fake_cuda::enable_result;
}
inline cudaError_t cudaGetLastError() {
    const auto error = fake_cuda::last_error;
    fake_cuda::last_error = cudaSuccess;
    return error;
}
inline cudaError_t cudaMemcpyPeerAsync(void* dst, int, const void* src, int,
                                     size_t bytes, cudaStream_t) {
    ++fake_cuda::peer_copies;
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes,
                                  cudaMemcpyKind kind, cudaStream_t) {
    if (kind == cudaMemcpyDeviceToHost) ++fake_cuda::staged_d2h;
    if (kind == cudaMemcpyHostToDevice) ++fake_cuda::staged_h2d;
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    *stream = new FakeCudaStream{fake_cuda::current_device};
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    delete stream;
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
    *event = new FakeCudaEvent{fake_cuda::current_device};
    return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** storage, size_t bytes) {
    *storage = std::malloc(bytes);
    return *storage ? cudaSuccess : cudaErrorUnknown;
}
inline cudaError_t cudaFreeHost(void* storage) {
    std::free(storage);
    return cudaSuccess;
}

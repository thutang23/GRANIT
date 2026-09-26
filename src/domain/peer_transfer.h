#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

namespace cudadem {

inline void peer_transfer_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// One instance per destination worker; staging is never shared between workers.
// The caller establishes source readiness before calling copy(). Direct copies
// remain asynchronous. The fallback waits only for this destination's stream
// and its private source-copy stream, and retains pinned storage through H2D.
class PeerTransfer {
    void* staging_ = nullptr;
    size_t capacity_ = 0;
    int destination_ = -1;
    cudaStream_t last_destination_stream_ = nullptr;
    cudaEvent_t staging_consumed_ = nullptr;
    std::vector<cudaStream_t> source_streams_;
    struct PeerRoute {
        int accessible = -1; // -1 unknown, 0 unavailable, 1 accessible
        int mapping = -1;    // -1 untried, 0 staged, 1 mapped
    };
    std::vector<std::vector<PeerRoute>> peer_routes_;

    // Called with the destination selected. Copy and startup preparation use
    // the same route cache, including the handled mapping-limit fallback.
    bool direct_route(int destination_device, int source_device) {
        if (peer_routes_.size() <= static_cast<size_t>(destination_device))
            peer_routes_.resize(static_cast<size_t>(destination_device) + 1);
        auto& routes = peer_routes_[destination_device];
        if (routes.size() <= static_cast<size_t>(source_device))
            routes.resize(static_cast<size_t>(source_device) + 1);
        PeerRoute& route = routes[source_device];
        if (route.accessible < 0) {
            int accessible = 0;
            peer_transfer_check(cudaDeviceCanAccessPeer(&accessible, destination_device,
                                                        source_device), "query peer access");
            route.accessible = accessible ? 1 : 0;
        }
        // Read this on every call: changing the runtime override still takes
        // effect after a route has been prepared or used.
        const char* force_staging = std::getenv("GRANIT_FORCE_HOST_STAGING");
        if (!route.accessible || (force_staging && std::string(force_staging) == "1"))
            return false;
        if (route.mapping < 0) {
            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePushA("Peer Transfer: Enable Mapping");
            #endif
            const cudaError_t enabled = cudaDeviceEnablePeerAccess(source_device, 0);
            #ifdef CUDADEM_ENABLE_PROFILING
            nvtxRangePop();
            #endif
            if (enabled == cudaSuccess) route.mapping = 1;
            else if (enabled == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError(); route.mapping = 1;
            } else if (enabled == cudaErrorTooManyPeers ||
                       enabled == cudaErrorPeerAccessUnsupported) {
                cudaGetLastError(); route.mapping = 0;
            } else peer_transfer_check(enabled, "enable peer payload mapping");
        }
        return route.mapping == 1;
    }
public:
    PeerTransfer() = default;
    PeerTransfer(const PeerTransfer&) = delete;
    PeerTransfer& operator=(const PeerTransfer&) = delete;
    ~PeerTransfer() {
        int previous = 0;
        cudaGetDevice(&previous);
        if (staging_) {
            cudaSetDevice(destination_);
            if(staging_consumed_)cudaEventSynchronize(staging_consumed_);
            cudaFreeHost(staging_);
            if(staging_consumed_)cudaEventDestroy(staging_consumed_);
        }
        for (size_t i = 0; i < source_streams_.size(); ++i)
            if (source_streams_[i]) {
                cudaSetDevice(static_cast<int>(i));
                cudaStreamDestroy(source_streams_[i]);
            }
        cudaSetDevice(previous);
    }
    // Prepare a directed destination <- source route without moving payloads
    // or changing the caller's selected device. A mapping-limit rejection is
    // cached as a staged route exactly as it is during copy().
    void prepare(int destination_device, int source_device) {
        if (destination_device == source_device) return;
        int previous_device = -1;
        peer_transfer_check(cudaGetDevice(&previous_device), "read current device");
        peer_transfer_check(cudaSetDevice(destination_device), "select peer destination");
        try {
            direct_route(destination_device, source_device);
        } catch (...) {
            cudaSetDevice(previous_device);
            throw;
        }
        peer_transfer_check(cudaSetDevice(previous_device), "restore current device");
    }
    void copy(void* destination, int destination_device, const void* source,
              int source_device, size_t bytes, cudaStream_t stream) {
        if (!bytes) return;
        if (!destination || !source) throw std::invalid_argument("null peer payload");
        peer_transfer_check(cudaSetDevice(destination_device), "select peer destination");
        if (source_device == destination_device) {
            peer_transfer_check(cudaMemcpyAsync(destination, source, bytes,
                                                cudaMemcpyDeviceToDevice, stream), "local payload");
            return;
        }
        if (direct_route(destination_device, source_device)) {
            peer_transfer_check(cudaMemcpyPeerAsync(destination, destination_device,
                                                    source, source_device, bytes, stream), "peer payload");
            return;
        }
        // This also resolves the caller's cross-device source-ready event.
        peer_transfer_check(cudaStreamSynchronize(stream), "staged destination readiness");
        if (staging_ && (destination_ != destination_device || last_destination_stream_ != stream)) {
            peer_transfer_check(cudaSetDevice(destination_), "select prior staging owner");
            peer_transfer_check(cudaEventSynchronize(staging_consumed_), "staging reuse");
            if(destination_!=destination_device){
                peer_transfer_check(cudaEventDestroy(staging_consumed_), "release staging completion");
                staging_consumed_=nullptr;
            }
            peer_transfer_check(cudaSetDevice(destination_device), "restore staging owner");
        }
        if (capacity_ < bytes) {
            if (staging_) peer_transfer_check(cudaFreeHost(staging_), "release staging");
            staging_ = nullptr;
            capacity_ = bytes;
            peer_transfer_check(cudaMallocHost(&staging_, capacity_), "allocate staging");
        }
        if(!staging_consumed_)peer_transfer_check(cudaEventCreateWithFlags(&staging_consumed_,cudaEventDisableTiming), "staging completion event");
        destination_ = destination_device;
        last_destination_stream_ = stream;
        if (source_streams_.size() <= static_cast<size_t>(source_device))
            source_streams_.resize(static_cast<size_t>(source_device) + 1, nullptr);
        peer_transfer_check(cudaSetDevice(source_device), "select staging source");
        if (!source_streams_[source_device])
            peer_transfer_check(cudaStreamCreateWithFlags(&source_streams_[source_device],
                                                          cudaStreamNonBlocking), "source staging stream");
        peer_transfer_check(cudaMemcpyAsync(staging_, source, bytes, cudaMemcpyDeviceToHost,
                                            source_streams_[source_device]), "stage source payload");
        peer_transfer_check(cudaStreamSynchronize(source_streams_[source_device]), "staged source completion");
        peer_transfer_check(cudaSetDevice(destination_device), "restore peer destination");
        peer_transfer_check(cudaMemcpyAsync(destination, staging_, bytes,
                                            cudaMemcpyHostToDevice, stream), "deliver staged payload");
        peer_transfer_check(cudaEventRecord(staging_consumed_,stream), "record staging consumption");
    }
};

inline PeerTransfer& worker_peer_transfer() {
    thread_local PeerTransfer transfer;
    return transfer;
}

inline void prepare_peer_payload_route(int dst_device, int src_device) {
    worker_peer_transfer().prepare(dst_device, src_device);
}

inline void copy_peer_payload(void* dst, int dst_device, const void* src,
                              int src_device, size_t bytes, cudaStream_t stream) {
    worker_peer_transfer().copy(dst, dst_device, src, src_device, bytes, stream);
}

} // namespace cudadem

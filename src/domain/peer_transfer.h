#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::vector<std::vector<int>> peer_mapping_; // -1 unknown, 0 staged, 1 mapped
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
        int accessible = 0;
        peer_transfer_check(cudaDeviceCanAccessPeer(&accessible, destination_device,
                                                    source_device), "query peer access");
        const char* force_staging = std::getenv("GRANIT_FORCE_HOST_STAGING");
        if (accessible && !(force_staging && std::string(force_staging) == "1")) {
            if (peer_mapping_.size() <= static_cast<size_t>(destination_device))
                peer_mapping_.resize(static_cast<size_t>(destination_device)+1);
            auto& mappings=peer_mapping_[destination_device];
            if (mappings.size() <= static_cast<size_t>(source_device))
                mappings.resize(static_cast<size_t>(source_device)+1,-1);
            if (mappings[source_device]<0) {
                const cudaError_t enabled=cudaDeviceEnablePeerAccess(source_device,0);
                if (enabled==cudaSuccess) mappings[source_device]=1;
                else if (enabled==cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError(); mappings[source_device]=1;
                } else if (enabled==cudaErrorTooManyPeers || enabled==cudaErrorPeerAccessUnsupported) {
                    cudaGetLastError(); mappings[source_device]=0;
                } else peer_transfer_check(enabled,"enable peer payload mapping");
            }
            if (mappings[source_device]) {
                peer_transfer_check(cudaMemcpyPeerAsync(destination, destination_device,
                                                        source, source_device, bytes, stream), "peer payload");
                return;
            }
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

inline void copy_peer_payload(void* dst, int dst_device, const void* src,
                              int src_device, size_t bytes, cudaStream_t stream) {
    thread_local PeerTransfer transfer;
    transfer.copy(dst, dst_device, src, src_device, bytes, stream);
}

} // namespace cudadem

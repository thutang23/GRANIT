#pragma once
#include "peer_transport_policy.h"
#include <cuda_runtime.h>
#include <cstdlib>
#if defined(__linux__)
#include <cerrno>
#include <fstream>
#include <sys/stat.h>
#endif
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudadem {

inline void peer_transfer_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

namespace peer_transport_detail {

struct State {
    std::mutex mutex;
    std::map<int, std::string> pci_ids;
    std::map<std::string, IommuMode> iommu_modes;
    std::map<std::pair<std::string, std::string>, LinkClass> fabrics;
    std::set<std::pair<int, int>> logged_routes;
};

inline State& state() {
    static State instance;
    return instance;
}

inline std::string pci_id_locked(State& state, int device) {
    auto found = state.pci_ids.find(device);
    if (found != state.pci_ids.end()) return found->second;
    char id[32] = {};
    const cudaError_t status = cudaDeviceGetPCIBusId(id, sizeof(id), device);
    return state.pci_ids.emplace(device,
        status == cudaSuccess ? normalize_pci_bus_id(id) : std::string()).first->second;
}

inline IommuMode iommu_mode_locked(State& state, const std::string& id) {
#if !defined(__linux__)
    (void)state;
    (void)id;
    return IommuMode::Absent;
#else
    if (id.empty()) return IommuMode::Unknown;
    const auto found = state.iommu_modes.find(id);
    if (found != state.iommu_modes.end()) return found->second;

    const std::string device_path = "/sys/bus/pci/devices/" + id;
    struct stat info {};
    IommuMode mode = IommuMode::Unknown;
    if (stat(device_path.c_str(), &info) == 0) {
        const std::string group_path = device_path + "/iommu_group";
        if (stat(group_path.c_str(), &info) == 0) {
            std::ifstream type_file(group_path + "/type");
            std::string type;
            if (std::getline(type_file, type)) mode = parse_iommu_type(type);
        } else if (errno == ENOENT) {
            mode = IommuMode::Absent;
        }
    }
    state.iommu_modes.emplace(id, mode);
    return mode;
#endif
}

inline bool stage_for_route(int destination, int source) {
    const char* force = std::getenv("GRANIT_FORCE_HOST_STAGING");
    const bool forced = force && std::string(force) == "1";
    State& data = state();
    std::lock_guard<std::mutex> guard(data.mutex);
    const std::string dst_id = pci_id_locked(data, destination);
    const std::string src_id = pci_id_locked(data, source);
    const auto fabric = data.fabrics.find({dst_id, src_id});
    const LinkClass link = fabric == data.fabrics.end()
        ? LinkClass::GenericPcieP2P : fabric->second;
    const IommuMode dst_mode = iommu_mode_locked(data, dst_id);
    const IommuMode src_mode = iommu_mode_locked(data, src_id);
    const bool staged = should_stage_peer_copy(forced, link, dst_mode, src_mode);
    if (data.logged_routes.emplace(destination, source).second) {
        std::clog << "[peer-transfer] GPU " << source << " -> " << destination
                  << " (" << src_id << " -> " << dst_id << ") "
                  << (staged ? "host staging" : "direct P2P")
                  << (forced ? " [forced]" : is_nvlink_fabric(link) ? " [NVLink/NVSwitch]"
                     : staged ? " [IOMMU translated or unknown]" : " [IOMMU absent or identity]")
                  << '\n';
    }
    return staged;
}

} // namespace peer_transport_detail

// Only NVML-proven fabric routes are registered. CUDA indices are resolved to
// PCI identities at copy time, so device visibility/remapping cannot mix routes.
inline void register_peer_link_class(const std::string& destination_pci,
                                     const std::string& source_pci, LinkClass link) {
    if (!is_nvlink_fabric(link)) return;
    const std::string destination = normalize_pci_bus_id(destination_pci);
    const std::string source = normalize_pci_bus_id(source_pci);
    if (destination.empty() || source.empty()) return;
    auto& data = peer_transport_detail::state();
    std::lock_guard<std::mutex> guard(data.mutex);
    data.fabrics[{destination, source}] = link;
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
        if (accessible && !peer_transport_detail::stage_for_route(destination_device,
                                                                   source_device)) {
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

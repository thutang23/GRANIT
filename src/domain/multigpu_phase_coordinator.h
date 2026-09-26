#ifndef CUDADEM_MULTIGPU_PHASE_COORDINATOR_H
#define CUDADEM_MULTIGPU_PHASE_COORDINATOR_H

// A small host-side coordinator for the directed, single-buffer exchanges used
// by the multi-GPU runner.  It deliberately provides dependencies, rather than
// a collective barrier: each GPU host thread only waits for the producer or
// consumer that owns the buffer it is about to use.

#include <cuda_runtime.h>
#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#endif

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cudadem {

enum class PeerPhase : uint8_t {
    Migration = 0,
    Halo = 1,
    Let = 2,
    // Adaptive YORP uses two narrow, directed exchanges.  Keeping these
    // separate prevents a slow scalar observation from reusing a migration
    // or LET event generation.
    YorpReduction = 3,
    YorpState = 4,
    FmmSource = 5,
    FmmProjection = 6,
    // Independent generations for the two source-owned batched LET rounds.
    FmmCellRequests = 7,
    FmmCellReplies = 8,
    FmmParticleRequests = 9,
    FmmParticleReplies = 10,
};

// This object is shared by the persistent host thread assigned to each GPU.
// The normal path never synchronizes a stream or a device.  A host-side wait
// is used only to make sure an event has been recorded before another stream
// can wait on that particular event generation.
class MultiGPUPhaseCoordinator {
public:
    explicit MultiGPUPhaseCoordinator(
        std::chrono::milliseconds stall_period = std::chrono::milliseconds(5000))
        : stall_period_(stall_period.count() > 0 ? stall_period
                                                : std::chrono::milliseconds(5000)) {}

    MultiGPUPhaseCoordinator(const MultiGPUPhaseCoordinator&) = delete;
    MultiGPUPhaseCoordinator& operator=(const MultiGPUPhaseCoordinator&) = delete;

    ~MultiGPUPhaseCoordinator() { shutdown(); }

    // device_ids maps logical GPU indices used by the runner to CUDA device
    // ordinals.  Call once before entering the persistent OpenMP region.
    bool initialize(const std::vector<int>& device_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return fail_locked("initialize called more than once");
        }
        if (device_ids.empty()) {
            return fail_locked("initialize requires at least one GPU");
        }

        device_ids_ = device_ids;
        num_gpus_ = static_cast<int>(device_ids.size());
        aborted_ = false;
        error_message_.clear();
        for (Channel& channel : channels_) {
            channel.reset(num_gpus_);
        }
        abort_streams_.assign(num_gpus_, nullptr);

        // Events are owned by the device on which they are recorded.  Ready
        // events live on the producer; consumed events live on the consumer.
        for (int gpu = 0; gpu < num_gpus_; ++gpu) {
            if (!cuda_ok_locked(cudaSetDevice(device_ids_[gpu]), "cudaSetDevice during initialize")) {
                cleanup_locked();
                return false;
            }
            if (!cuda_ok_locked(cudaStreamCreateWithFlags(&abort_streams_[gpu], cudaStreamNonBlocking),
                                "cudaStreamCreateWithFlags during initialize")) {
                cleanup_locked();
                return false;
            }
            for (Channel& channel : channels_) {
                if (!cuda_ok_locked(cudaEventCreateWithFlags(&channel.ready[gpu], cudaEventDisableTiming),
                                    "cudaEventCreateWithFlags for ready event")) {
                    cleanup_locked();
                    return false;
                }
            }
        }
        for (int producer = 0; producer < num_gpus_; ++producer) {
            for (int consumer = 0; consumer < num_gpus_; ++consumer) {
                if (producer == consumer) continue;
                if (!cuda_ok_locked(cudaSetDevice(device_ids_[consumer]),
                                    "cudaSetDevice for consumed event")) {
                    cleanup_locked();
                    return false;
                }
                for (Channel& channel : channels_) {
                    cudaEvent_t& event = channel.consumed[channel.edge(producer, consumer)];
                    if (!cuda_ok_locked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                        "cudaEventCreateWithFlags for consumed event") ||
                        !cuda_ok_locked(cudaEventRecord(event, abort_streams_[consumer]),
                                        "initial cudaEventRecord for consumed event")) {
                        cleanup_locked();
                        return false;
                    }
                }
            }
        }

        initialized_ = true;
        return true;
    }

    bool initialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_;
    }

    bool aborted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return aborted_;
    }

    int num_gpus() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_gpus_;
    }

    std::string error_message() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_message_;
    }

    // Convenience for a producer that reuses its own phase buffer on the next
    // epoch.  For buffers shared across phases, use wait_for_consumed with the
    // actual predecessor phase instead.
    bool prepare_producer(PeerPhase phase, int producer, uint64_t epoch,
                          cudaStream_t producer_stream) {
        uint64_t predecessor = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!validate_gpu_locked(producer) || !validate_phase_locked(phase)) return false;
            predecessor = channel_locked(phase).published[producer];
        }
        if (predecessor == 0 || epoch == 0) return epoch != 0;
        for (int consumer = 0; consumer < num_gpus_; ++consumer) {
            if (consumer != producer &&
                !wait_for_consumed(phase, producer, consumer, predecessor, producer_stream)) {
                return false;
            }
        }
        return true;
    }

    // Enqueue the directed consumer acknowledgement dependency before the
    // producer overwrites that source buffer.  It is valid to use this across
    // phases (migration -> halo, halo -> migration), which is needed because
    // Communicator reuses the same send storage for both exchanges.
    bool wait_for_consumed(PeerPhase consumed_phase, int producer, int consumer,
                           uint64_t expected_epoch, cudaStream_t producer_stream) {
        if (expected_epoch == 0) return true;

        std::unique_lock<std::mutex> lock(mutex_);
        if (!validate_edge_locked(consumed_phase, producer, consumer)) return false;
        Channel& channel = channel_locked(consumed_phase);
        const size_t edge = channel.edge(producer, consumer);
        auto next_report = std::chrono::steady_clock::now() + stall_period_;
        while (!aborted_ && channel.acknowledged[edge] < expected_epoch) {
            std::cv_status wait_status;
            {
#ifdef CUDADEM_ENABLE_PROFILING
                ScopedWaitNvtx wait_range("wait_for_consumed", consumed_phase,
                                          producer, consumer, expected_epoch);
#endif
                wait_status = cv_.wait_until(lock, next_report);
            }
            if (wait_status == std::cv_status::timeout) {
                report_stall_locked("consumed acknowledgement", consumed_phase, producer, consumer,
                                    expected_epoch, channel.acknowledged[edge]);
                next_report = std::chrono::steady_clock::now() + stall_period_;
            }
        }
        if (aborted_) return false;
        if (!cuda_ok_locked(cudaSetDevice(device_ids_[producer]),
                            "cudaSetDevice while waiting for consumed event")) {
            return false;
        }
        return cuda_ok_locked(cudaStreamWaitEvent(producer_stream, channel.consumed[edge], 0),
                              "cudaStreamWaitEvent for consumed event");
    }

    // Records the producer-ready event after its pack/send work.  Epochs must
    // strictly increase independently for every (phase, producer); gaps are
    // allowed because a phase need not run on every simulation step.
    bool publish(PeerPhase phase, int producer, uint64_t epoch, cudaStream_t producer_stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validate_gpu_locked(producer) || !validate_phase_locked(phase) || epoch == 0) return false;
        Channel& channel = channel_locked(phase);
        if (epoch <= channel.published[producer]) {
            return fail_locked("non-monotonic producer epoch in publish");
        }
        if (!cuda_ok_locked(cudaSetDevice(device_ids_[producer]), "cudaSetDevice while publishing") ||
            !cuda_ok_locked(cudaEventRecord(channel.ready[producer], producer_stream),
                            "cudaEventRecord for producer-ready event")) {
            return false;
        }
        channel.published[producer] = epoch;
#ifdef CUDADEM_ENABLE_PROFILING
        mark_epoch_nvtx("published", phase, producer, -1, epoch);
#endif
        cv_.notify_all();
        return true;
    }

    // Establishes the producer-ready dependency in a consumer stream.  Host
    // metadata written by the producer before publish is also visible after
    // this method returns, so it can safely carry narrow scalar counts.
    bool wait_for_published(PeerPhase phase, int producer, int consumer,
                            uint64_t epoch, cudaStream_t consumer_stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!validate_edge_locked(phase, producer, consumer) || epoch == 0) return false;
        Channel& channel = channel_locked(phase);
        auto next_report = std::chrono::steady_clock::now() + stall_period_;
        while (!aborted_ && channel.published[producer] < epoch) {
            std::cv_status wait_status;
            {
#ifdef CUDADEM_ENABLE_PROFILING
                ScopedWaitNvtx wait_range("wait_for_published", phase,
                                          producer, consumer, epoch);
#endif
                wait_status = cv_.wait_until(lock, next_report);
            }
            if (wait_status == std::cv_status::timeout) {
                report_stall_locked("producer-ready event", phase, producer, consumer,
                                    epoch, channel.published[producer]);
                next_report = std::chrono::steady_clock::now() + stall_period_;
            }
        }
        if (aborted_) return false;
        if (!cuda_ok_locked(cudaSetDevice(device_ids_[consumer]),
                            "cudaSetDevice while waiting for producer-ready event")) {
            return false;
        }
        if (!cuda_ok_locked(cudaStreamWaitEvent(consumer_stream, channel.ready[producer], 0),
                            "cudaStreamWaitEvent for producer-ready event")) {
            return false;
        }
        channel.waited[channel.edge(producer, consumer)] = epoch;
        return true;
    }

    // Record this only after the consumer stream has completed every read of
    // the producer's single buffer (a receiver-pull copy or LET merge).
    bool acknowledge_consumed(PeerPhase phase, int producer, int consumer,
                              uint64_t epoch, cudaStream_t consumer_stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validate_edge_locked(phase, producer, consumer) || epoch == 0) return false;
        Channel& channel = channel_locked(phase);
        const size_t edge = channel.edge(producer, consumer);
        if (channel.waited[edge] < epoch) {
            return fail_locked("consumer acknowledgement without producer-ready dependency");
        }
        if (epoch <= channel.acknowledged[edge]) {
            return fail_locked("non-monotonic consumer acknowledgement epoch");
        }
        if (!cuda_ok_locked(cudaSetDevice(device_ids_[consumer]), "cudaSetDevice while acknowledging") ||
            !cuda_ok_locked(cudaEventRecord(channel.consumed[edge], consumer_stream),
                            "cudaEventRecord for consumed acknowledgement")) {
            return false;
        }
        channel.acknowledged[edge] = epoch;
#ifdef CUDADEM_ENABLE_PROFILING
        mark_epoch_nvtx("acknowledged", phase, producer, consumer, epoch);
#endif
        cv_.notify_all();
        return true;
    }

    // Releases all host waiters and publishes terminal event generations for
    // dependencies that have not yet been enqueued.  Already-enqueued CUDA
    // work is drained during shutdown.  This is error-path coordination, not
    // a timeout policy.
    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_) return;
        aborted_ = true;
        if (initialized_) release_all_events_locked();
        cv_.notify_all();
    }

    // Call after all GPU host threads have left their parallel region.  The
    // shutdown-only per-device synchronize makes CUDA event destruction safe.
    void shutdown() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ && device_ids_.empty()) return;
        aborted_ = true;
        if (initialized_) release_all_events_locked();
        cv_.notify_all();
        cleanup_locked();
    }

private:
#ifdef CUDADEM_ENABLE_PROFILING
    static const char* phase_name_nvtx(PeerPhase phase) {
        switch (phase) {
            case PeerPhase::Migration: return "Migration";
            case PeerPhase::Halo: return "Halo";
            case PeerPhase::Let: return "Let";
            case PeerPhase::YorpReduction: return "YorpReduction";
            case PeerPhase::YorpState: return "YorpState";
            case PeerPhase::FmmSource: return "FmmSource";
            case PeerPhase::FmmProjection: return "FmmProjection";
            case PeerPhase::FmmCellRequests: return "FmmCellRequests";
            case PeerPhase::FmmCellReplies: return "FmmCellReplies";
            case PeerPhase::FmmParticleRequests: return "FmmParticleRequests";
            case PeerPhase::FmmParticleReplies: return "FmmParticleReplies";
        }
        return "Unknown";
    }

    static nvtxEventAttributes_t epoch_attributes_nvtx(const char* label, uint64_t epoch) {
        nvtxEventAttributes_t attributes = {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = label;
        attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        attributes.payload.ullValue = epoch;
        return attributes;
    }

    class ScopedWaitNvtx {
    public:
        ScopedWaitNvtx(const char* operation, PeerPhase phase, int producer,
                       int consumer, uint64_t epoch) {
            char label[160];
            std::snprintf(label, sizeof(label),
                          "PhaseCoordinator %s cv.wait_until phase=%s producer=%d consumer=%d epoch=%llu",
                          operation, phase_name_nvtx(phase), producer, consumer,
                          static_cast<unsigned long long>(epoch));
            const auto attributes = epoch_attributes_nvtx(label, epoch);
            range_ = nvtxRangeStartEx(&attributes);
        }

        ~ScopedWaitNvtx() { nvtxRangeEnd(range_); }

        ScopedWaitNvtx(const ScopedWaitNvtx&) = delete;
        ScopedWaitNvtx& operator=(const ScopedWaitNvtx&) = delete;

    private:
        nvtxRangeId_t range_;
    };

    static void mark_epoch_nvtx(const char* operation, PeerPhase phase, int producer,
                                int consumer, uint64_t epoch) {
        char label[160];
        std::snprintf(label, sizeof(label),
                      "PhaseCoordinator %s phase=%s producer=%d consumer=%d epoch=%llu",
                      operation, phase_name_nvtx(phase), producer, consumer,
                      static_cast<unsigned long long>(epoch));
        const auto attributes = epoch_attributes_nvtx(label, epoch);
        nvtxMarkEx(&attributes);
    }
#endif

    struct Channel {
        std::vector<cudaEvent_t> ready;
        std::vector<cudaEvent_t> consumed;
        std::vector<uint64_t> published;
        std::vector<uint64_t> acknowledged;
        std::vector<uint64_t> waited;

        void reset(int n) {
            ready.assign(n, nullptr);
            consumed.assign(static_cast<size_t>(n) * n, nullptr);
            published.assign(n, 0);
            acknowledged.assign(static_cast<size_t>(n) * n, 0);
            waited.assign(static_cast<size_t>(n) * n, 0);
        }
        size_t edge(int producer, int consumer) const {
            return static_cast<size_t>(producer) * ready.size() + consumer;
        }
    };

    static constexpr size_t phase_index(PeerPhase phase) {
        return static_cast<size_t>(phase);
    }

    Channel& channel_locked(PeerPhase phase) { return channels_[phase_index(phase)]; }
    const Channel& channel_locked(PeerPhase phase) const { return channels_[phase_index(phase)]; }

    bool validate_phase_locked(PeerPhase phase) {
        if (phase_index(phase) >= channels_.size()) return fail_locked("unknown peer phase");
        if (!initialized_) return fail_locked("coordinator is not initialized");
        if (aborted_) return false;
        return true;
    }

    bool validate_gpu_locked(int gpu) {
        if (!initialized_) return fail_locked("coordinator is not initialized");
        if (aborted_) return false;
        if (gpu < 0 || gpu >= num_gpus_) return fail_locked("invalid logical GPU index");
        return true;
    }

    bool validate_edge_locked(PeerPhase phase, int producer, int consumer) {
        if (!validate_phase_locked(phase) || !validate_gpu_locked(producer) || !validate_gpu_locked(consumer)) {
            return false;
        }
        if (producer == consumer) return fail_locked("peer dependency requires distinct producer and consumer");
        return true;
    }

    bool cuda_ok_locked(cudaError_t status, const char* operation) {
        if (status == cudaSuccess) return true;
        error_message_ = std::string(operation) + ": " + cudaGetErrorString(status);
        std::fprintf(stderr, "[multigpu phase coordinator] %s\n", error_message_.c_str());
        aborted_ = true;
        if (initialized_) release_all_events_locked();
        cv_.notify_all();
        return false;
    }

    bool fail_locked(const char* message) {
        error_message_ = message;
        std::fprintf(stderr, "[multigpu phase coordinator] %s\n", message);
        aborted_ = true;
        if (initialized_) release_all_events_locked();
        cv_.notify_all();
        return false;
    }

    void report_stall_locked(const char* wait_name, PeerPhase phase, int producer,
                             int consumer, uint64_t wanted, uint64_t observed) const {
        std::fprintf(stderr,
                     "[multigpu phase coordinator] stall waiting for %s: phase=%u producer=%d "
                     "consumer=%d wanted_epoch=%llu observed_epoch=%llu\n",
                     wait_name, static_cast<unsigned>(phase), producer, consumer,
                     static_cast<unsigned long long>(wanted),
                     static_cast<unsigned long long>(observed));
    }

    void release_all_events_locked() noexcept {
        for (int owner = 0; owner < num_gpus_; ++owner) {
            if (owner >= static_cast<int>(abort_streams_.size()) || !abort_streams_[owner]) continue;
            cudaSetDevice(device_ids_[owner]);
            for (Channel& channel : channels_) {
                if (channel.ready[owner]) cudaEventRecord(channel.ready[owner], abort_streams_[owner]);
                for (int producer = 0; producer < num_gpus_; ++producer) {
                    if (producer == owner) continue;
                    cudaEvent_t event = channel.consumed[channel.edge(producer, owner)];
                    if (event) cudaEventRecord(event, abort_streams_[owner]);
                }
            }
        }
    }

    void cleanup_locked() noexcept {
        // Events have an owning device; synchronizing only at shutdown avoids
        // destroying an event still used by a cross-device stream wait.
        for (int gpu = 0; gpu < num_gpus_; ++gpu) {
            cudaSetDevice(device_ids_[gpu]);
            cudaDeviceSynchronize();
            for (Channel& channel : channels_) {
                if (gpu < static_cast<int>(channel.ready.size()) && channel.ready[gpu]) {
                    cudaEventDestroy(channel.ready[gpu]);
                    channel.ready[gpu] = nullptr;
                }
                for (int producer = 0; producer < num_gpus_; ++producer) {
                    if (producer == gpu || channel.consumed.empty()) continue;
                    cudaEvent_t& event = channel.consumed[channel.edge(producer, gpu)];
                    if (event) {
                        cudaEventDestroy(event);
                        event = nullptr;
                    }
                }
            }
            if (gpu < static_cast<int>(abort_streams_.size()) && abort_streams_[gpu]) {
                cudaStreamDestroy(abort_streams_[gpu]);
                abort_streams_[gpu] = nullptr;
            }
        }
        for (Channel& channel : channels_) channel.reset(0);
        abort_streams_.clear();
        device_ids_.clear();
        num_gpus_ = 0;
        initialized_ = false;
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::array<Channel, 11> channels_;
    std::vector<int> device_ids_;
    std::vector<cudaStream_t> abort_streams_;
    int num_gpus_ = 0;
    std::chrono::milliseconds stall_period_;
    bool initialized_ = false;
    bool aborted_ = false;
    std::string error_message_;
};

}  // namespace cudadem

#endif  // CUDADEM_MULTIGPU_PHASE_COORDINATOR_H

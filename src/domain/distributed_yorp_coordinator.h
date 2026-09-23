#ifndef CUDADEM_DISTRIBUTED_YORP_COORDINATOR_H
#define CUDADEM_DISTRIBUTED_YORP_COORDINATOR_H

// Distributed adaptive YORP is intentionally a small two-stage directed
// protocol, rather than a general collective.  Each worker reduces its owned
// prefix locally, GPU 0 pulls those compact records in logical-GPU order and
// evaluates the existing device FSM, then every worker pulls that one state
// record.  CUDA stream dependencies carry the data ordering; host waits only
// establish that a producer has recorded its event and are not a barrier.

#include "multigpu_phase_coordinator.h"
#include "peer_transfer.h"
#include "../core/math/yorp.cuh"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cudadem {

class DistributedYorpCoordinator {
public:
    DistributedYorpCoordinator() = default;
    DistributedYorpCoordinator(const DistributedYorpCoordinator&) = delete;
    DistributedYorpCoordinator& operator=(const DistributedYorpCoordinator&) = delete;
    ~DistributedYorpCoordinator() { shutdown(); }

    bool initialize(const std::vector<int>& device_ids,
                    MultiGPUPhaseCoordinator* phase_coordinator) {
        shutdown();
        if (!phase_coordinator || !phase_coordinator->initialized() ||
            device_ids.empty() ||
            phase_coordinator->num_gpus() != static_cast<int>(device_ids.size())) {
            return fail("invalid distributed YORP coordinator initialization");
        }
        device_ids_ = device_ids;
        phase_ = phase_coordinator;
        managers_.assign(device_ids.size(), nullptr);
        // The shared transport maps available peers on demand and stages
        // links that are unavailable or exceed the device mapping limit.
        cudaError_t status = cudaSetDevice(device_ids_[0]);
        if (status != cudaSuccess) return cuda_fail("cudaSetDevice for YORP gather", status);
        status = cudaMalloc(&d_gather_, device_ids.size() * sizeof(YorpReduction));
        if (status != cudaSuccess) return cuda_fail("cudaMalloc for YORP gather", status);
        initialized_ = true;
        return true;
    }

    bool attach_manager(int logical_gpu, YorpManager* manager) {
        if (!validate_gpu(logical_gpu) || !manager) return fail("invalid YORP manager attachment");
        managers_[logical_gpu] = manager;
        return true;
    }

    // Call once from each persistent GPU worker at an agreed observation
    // epoch.  `owned_count` is the local prefix before ghosts.  The caller
    // owns cadence (including the mandatory t=0 observation); this object
    // owns only one all-participant protocol execution.
    bool observe(int logical_gpu, ParticleSoA& particles, int32_t owned_count,
                 double now, const SimConfig& config, int block_size,
                 cudaStream_t stream, uint64_t epoch) {
        if (!config.acceleration.yorp.adaptive ||
            now < config.acceleration.yorp.start_time) return true;
        if (!validate_attached(logical_gpu) || epoch == 0) return false;
        if (!set_device(logical_gpu)) return false;

        if (epoch > 1 && logical_gpu != 0 &&
            !phase_->wait_for_consumed(PeerPhase::YorpReduction, logical_gpu, 0,
                                       epoch - 1, stream)) {
            return fail("previous YORP reduction was not consumed");
        }
        managers_[logical_gpu]->enqueue_owned_reduction(particles, owned_count,
                                                        config, block_size, stream);
        if (!phase_->publish(PeerPhase::YorpReduction, logical_gpu, epoch, stream)) {
            return fail("could not publish local YORP reduction");
        }

        if (logical_gpu != 0) {
            if (!phase_->wait_for_published(PeerPhase::YorpState, 0, logical_gpu,
                                            epoch, stream)) {
                return fail("could not wait for authoritative YORP state");
            }
            copy_peer_payload(
                managers_[logical_gpu]->device_state(), device_ids_[logical_gpu],
                managers_[0]->device_state(), device_ids_[0], sizeof(YorpDeviceState), stream);
            if (!phase_->acknowledge_consumed(PeerPhase::YorpState, 0, logical_gpu,
                                              epoch, stream)) {
                return fail("could not acknowledge authoritative YORP state");
            }
            return true;
        }

        // GPU 0 waits only on each producer it consumes.  The state buffer is
        // not overwritten until every mirror has consumed its prior epoch.
        for (int peer = 1; peer < static_cast<int>(managers_.size()); ++peer) {
            if (epoch > 1 && !phase_->wait_for_consumed(PeerPhase::YorpState, 0, peer,
                                                        epoch - 1, stream)) {
                return fail("previous YORP state was not consumed");
            }
            if (!phase_->wait_for_published(PeerPhase::YorpReduction, peer, 0,
                                            epoch, stream)) {
                return fail("could not wait for peer YORP reduction");
            }
        }
        cudaError_t status = cudaMemcpyAsync(d_gather_, managers_[0]->device_reduction(),
                                             sizeof(YorpReduction), cudaMemcpyDeviceToDevice,
                                             stream);
        if (status != cudaSuccess) return cuda_fail("YORP local gather copy", status);
        for (int peer = 1; peer < static_cast<int>(managers_.size()); ++peer) {
            copy_peer_payload(d_gather_ + peer, device_ids_[0],
                                         managers_[peer]->device_reduction(), device_ids_[peer],
                                         sizeof(YorpReduction), stream);
            if (!phase_->acknowledge_consumed(PeerPhase::YorpReduction, peer, 0,
                                              epoch, stream)) {
                return fail("could not acknowledge peer YORP reduction");
            }
        }
        yorp_sum_reductions_kernel<<<1, 1, 0, stream>>>(d_gather_,
            static_cast<int32_t>(managers_.size()), managers_[0]->device_reduction());
        status = cudaGetLastError();
        if (status != cudaSuccess) return cuda_fail("YORP deterministic reduction", status);
        managers_[0]->evaluate_adaptive_reduction(managers_[0]->device_reduction(), now,
                                                  config, stream);
        if (!phase_->publish(PeerPhase::YorpState, 0, epoch, stream)) {
            return fail("could not publish authoritative YORP state");
        }
        published_state_epoch_ = epoch;
        return true;
    }

    // Invoked on every worker after `observe` has made its state mirror
    // stream-visible.  It intentionally applies only to the owned prefix.
    bool apply_increment(int logical_gpu, ParticleSoA& particles, int32_t owned_count,
                         double now, double dt, const SimConfig& config,
                         int block_size, cudaStream_t stream) {
        if (!config.acceleration.yorp.adaptive ||
            now < config.acceleration.yorp.start_time ||
            now >= config.acceleration.yorp.stop_time) return true;
        if (!validate_attached(logical_gpu) || !set_device(logical_gpu)) return false;
        if (logical_gpu == 0 &&
            published_state_epoch_ > safely_consumed_state_epoch_) {
            // The authoritative state is also the root manager's live FSM
            // storage.  Before its next increment mutates that record, order
            // the root stream after every peer's current broadcast copy.  The
            // wait occurs only once per sparse observation epoch and retains
            // the directed-event protocol without a global host barrier.
            for (int peer = 1; peer < static_cast<int>(managers_.size()); ++peer) {
                if (!phase_->wait_for_consumed(
                        PeerPhase::YorpState, 0, peer,
                        published_state_epoch_, stream)) {
                    return fail("authoritative YORP state was not consumed before mutation");
                }
            }
            safely_consumed_state_epoch_ = published_state_epoch_;
        }
        managers_[logical_gpu]->apply_adaptive_increment_owned(
            particles, owned_count, now, dt, config, block_size, stream);
        return true;
    }

    bool initialized() const { return initialized_; }
    const std::string& error_message() const { return error_message_; }

    void shutdown() noexcept {
        if (d_gather_) {
            cudaSetDevice(device_ids_.empty() ? 0 : device_ids_[0]);
            cudaFree(d_gather_);
        }
        d_gather_ = nullptr;
        managers_.clear();
        device_ids_.clear();
        phase_ = nullptr;
        initialized_ = false;
        published_state_epoch_ = 0;
        safely_consumed_state_epoch_ = 0;
        error_message_.clear();
    }

private:
    bool validate_gpu(int logical_gpu) const {
        return initialized_ && logical_gpu >= 0 &&
               logical_gpu < static_cast<int>(device_ids_.size());
    }
    bool validate_attached(int logical_gpu) {
        if (!validate_gpu(logical_gpu) || !managers_[logical_gpu])
            return fail("YORP worker has no attached manager");
        // GPU 0 dereferences every manager during its gather, so reject a
        // partial attachment immediately instead of leaving a worker blocked.
        if (logical_gpu == 0) {
            for (YorpManager* manager : managers_)
                if (!manager) return fail("not all YORP managers are attached");
        }
        return true;
    }
    bool set_device(int logical_gpu) {
        cudaError_t status = cudaSetDevice(device_ids_[logical_gpu]);
        return status == cudaSuccess ? true : cuda_fail("cudaSetDevice for YORP worker", status);
    }
    bool cuda_fail(const char* context, cudaError_t status) {
        error_message_ = std::string(context) + ": " + cudaGetErrorString(status);
        std::fprintf(stderr, "[distributed YORP] %s\n", error_message_.c_str());
        if (phase_) phase_->abort();
        return false;
    }
    bool fail(const char* message) {
        error_message_ = message;
        std::fprintf(stderr, "[distributed YORP] %s\n", message);
        if (phase_) phase_->abort();
        return false;
    }

    std::vector<int> device_ids_;
    std::vector<YorpManager*> managers_;
    MultiGPUPhaseCoordinator* phase_ = nullptr;
    YorpReduction* d_gather_ = nullptr;
    uint64_t published_state_epoch_ = 0;
    uint64_t safely_consumed_state_epoch_ = 0;
    bool initialized_ = false;
    std::string error_message_;
};

}  // namespace cudadem

#endif  // CUDADEM_DISTRIBUTED_YORP_COORDINATOR_H

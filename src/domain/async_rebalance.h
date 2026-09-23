#ifndef CUDADEM_ASYNC_REBALANCE_H
#define CUDADEM_ASYNC_REBALANCE_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace cudadem {

// Lock-free host mailbox for per-GPU step timings.  Writers never wait for
// peers.  The coordinator thread collects the newest step still present in
// every GPU's small ring, then may schedule a collective rebalance far enough
// in the future that no worker can miss the rendezvous.
class AsyncRebalanceMailbox {
public:
    static constexpr int32_t kTimingSlots = 4;

    void initialize(int32_t num_gpus) {
        num_gpus_ = num_gpus;
        slots_.reset(new TimingSlot[static_cast<size_t>(num_gpus_) * kTimingSlots]);
        latest_epoch_.reset(new std::atomic<int64_t>[num_gpus_]);
        progress_epoch_.reset(new std::atomic<int64_t>[num_gpus_]);
        for (int32_t g = 0; g < num_gpus_; ++g) {
            latest_epoch_[g].store(kNoEpoch, std::memory_order_relaxed);
            progress_epoch_[g].store(kNoEpoch, std::memory_order_relaxed);
        }
        pending_rebalance_epoch_.store(kNoEpoch, std::memory_order_relaxed);
    }

    void publish_progress(int32_t gpu, int64_t epoch) {
        if (gpu < 0 || gpu >= num_gpus_ || epoch < 0) return;
        progress_epoch_[gpu].store(epoch, std::memory_order_release);
    }

    void publish(int32_t gpu, int64_t epoch, float milliseconds) {
        if (gpu < 0 || gpu >= num_gpus_ || epoch < 0) return;

        TimingSlot& slot = slot_for(gpu, epoch);
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(milliseconds), "float must be 32-bit");
        std::memcpy(&bits, &milliseconds, sizeof(bits));
        slot.value_bits.store(bits, std::memory_order_relaxed);
        slot.epoch.store(epoch, std::memory_order_release);
        latest_epoch_[gpu].store(epoch, std::memory_order_release);
    }

    // Returns the newest common timing epoch newer than after_epoch.  A small
    // ring is sufficient because same-step halo/LET dependencies bound normal
    // host-thread skew; overwritten samples are simply skipped.
    bool try_collect_latest(int64_t after_epoch,
                            int64_t& collected_epoch,
                            std::vector<float>& milliseconds) const {
        if (num_gpus_ <= 0) return false;

        int64_t newest_common = std::numeric_limits<int64_t>::max();
        for (int32_t g = 0; g < num_gpus_; ++g) {
            int64_t latest = latest_epoch_[g].load(std::memory_order_acquire);
            if (latest == kNoEpoch) return false;
            if (latest < newest_common) newest_common = latest;
        }

        const int64_t oldest_candidate =
            (newest_common >= kTimingSlots - 1)
                ? newest_common - (kTimingSlots - 1)
                : 0;
        for (int64_t epoch = newest_common;
             epoch > after_epoch && epoch >= oldest_candidate;
             --epoch) {
            std::vector<float> candidate(static_cast<size_t>(num_gpus_));
            bool complete = true;
            for (int32_t g = 0; g < num_gpus_; ++g) {
                const TimingSlot& slot = slot_for(g, epoch);
                int64_t stamp_before = slot.epoch.load(std::memory_order_acquire);
                if (stamp_before != epoch) {
                    complete = false;
                    break;
                }
                uint32_t bits = slot.value_bits.load(std::memory_order_relaxed);
                int64_t stamp_after = slot.epoch.load(std::memory_order_acquire);
                if (stamp_after != epoch) {
                    complete = false;
                    break;
                }
                std::memcpy(&candidate[static_cast<size_t>(g)], &bits, sizeof(bits));
            }
            if (complete) {
                collected_epoch = epoch;
                milliseconds.swap(candidate);
                return true;
            }
        }
        return false;
    }

    int64_t max_published_epoch() const {
        int64_t maximum = kNoEpoch;
        for (int32_t g = 0; g < num_gpus_; ++g) {
            int64_t epoch = latest_epoch_[g].load(std::memory_order_acquire);
            if (epoch > maximum) maximum = epoch;
        }
        return maximum;
    }

    int64_t max_progress_epoch() const {
        int64_t maximum = kNoEpoch;
        for (int32_t g = 0; g < num_gpus_; ++g) {
            int64_t epoch = progress_epoch_[g].load(std::memory_order_acquire);
            if (epoch > maximum) maximum = epoch;
        }
        return maximum;
    }

    bool schedule(int64_t epoch) {
        int64_t expected = kNoEpoch;
        return pending_rebalance_epoch_.compare_exchange_strong(
            expected, epoch, std::memory_order_release, std::memory_order_relaxed);
    }

    int64_t pending_epoch() const {
        return pending_rebalance_epoch_.load(std::memory_order_acquire);
    }

    void complete(int64_t epoch) {
        int64_t expected = epoch;
        pending_rebalance_epoch_.compare_exchange_strong(
            expected, kNoEpoch, std::memory_order_release, std::memory_order_relaxed);
    }

private:
    static constexpr int64_t kNoEpoch = -1;

    struct TimingSlot {
        std::atomic<int64_t> epoch{kNoEpoch};
        std::atomic<uint32_t> value_bits{0};
    };

    TimingSlot& slot_for(int32_t gpu, int64_t epoch) {
        size_t index = static_cast<size_t>(gpu) * kTimingSlots
                     + static_cast<size_t>(epoch % kTimingSlots);
        return slots_[index];
    }

    const TimingSlot& slot_for(int32_t gpu, int64_t epoch) const {
        size_t index = static_cast<size_t>(gpu) * kTimingSlots
                     + static_cast<size_t>(epoch % kTimingSlots);
        return slots_[index];
    }

    int32_t num_gpus_ = 0;
    std::unique_ptr<TimingSlot[]> slots_;
    std::unique_ptr<std::atomic<int64_t>[]> latest_epoch_;
    std::unique_ptr<std::atomic<int64_t>[]> progress_epoch_;
    std::atomic<int64_t> pending_rebalance_epoch_{kNoEpoch};
};

} // namespace cudadem

#endif // CUDADEM_ASYNC_REBALANCE_H

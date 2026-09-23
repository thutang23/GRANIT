#ifndef CUDADEM_REBALANCE_TRIGGER_H
#define CUDADEM_REBALANCE_TRIGGER_H

// =============================================================================
// Pure host rebalance decision logic (no CUDA). Decides WHEN to rebalance from
// per-GPU compute-time measurements. See SP1 design §5.
// =============================================================================
#include <vector>
#include <cstdint>

namespace cudadem {

class RebalanceController {
public:
    void configure(float threshold, int32_t min_interval) {
        threshold_ = threshold;
        min_interval_ = min_interval;
    }

    // True if imbalance exceeds threshold AND min_interval has elapsed since last fire.
    bool should_rebalance(const std::vector<float>& gpu_times, int64_t step) const {
        if (gpu_times.size() < 2) return false;
        if (step - last_step_ < min_interval_) return false;

        float sum = 0.0f, mx = 0.0f;
        for (float t : gpu_times) {
            sum += t;
            if (t > mx) mx = t;
        }
        float avg = sum / static_cast<float>(gpu_times.size());
        if (avg <= 0.0f) return false;
        return (mx / avg) > threshold_;
    }

    void notify_rebalanced(int64_t step) { last_step_ = step; }

private:
    float   threshold_ = 1.15f;
    int32_t min_interval_ = 50;
    int64_t last_step_ = -1000000;
};

} // namespace cudadem
#endif // CUDADEM_REBALANCE_TRIGGER_H

#pragma once

#include <cstdint>

namespace cudadem {

// Records the step for the last successfully written particle output.  A
// failed write intentionally leaves the step pending so shutdown can retry it.
class FinalOutputTracker {
public:
    void reset() { last_written_step_ = -1; }

    bool needs_output(int64_t current_step) const {
        return last_written_step_ != current_step;
    }

    void mark_written(int64_t current_step) {
        last_written_step_ = current_step;
    }

private:
    int64_t last_written_step_ = -1;
};

}  // namespace cudadem

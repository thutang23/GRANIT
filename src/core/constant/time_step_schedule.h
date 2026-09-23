#pragma once

#include <algorithm>
#include <cmath>

namespace cudadem {

// Preserve the configured fixed timestep except for the last step before a
// finite time limit. The shortened step makes force sampling and the stored
// terminating state land on the requested endpoint instead of overshooting it.
inline double terminal_limited_step_dt(
    double current_time,
    double nominal_dt,
    double stop_time) {
    if (!(nominal_dt > 0.0) || !std::isfinite(nominal_dt)) return nominal_dt;
    const double remaining = stop_time - current_time;
    if (remaining > 0.0 && remaining < nominal_dt) return remaining;
    return nominal_dt;
}

// A RESPA macro-step uses symmetric outer kicks. When its last inner step is
// shortened, both outer half-kicks must be based on the actual accumulated
// inner-step span rather than inner_count * nominal_dt.
inline double terminal_limited_respa_span(
    double current_time,
    double nominal_dt,
    int inner_count,
    double stop_time) {
    const double nominal_span =
        static_cast<double>(std::max(inner_count, 0)) * nominal_dt;
    if (!(nominal_span > 0.0) || !std::isfinite(nominal_span)) {
        return nominal_span;
    }
    const double remaining = stop_time - current_time;
    if (remaining > 0.0 && remaining < nominal_span) return remaining;
    return nominal_span;
}

}  // namespace cudadem

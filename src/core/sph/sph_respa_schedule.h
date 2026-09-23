#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudadem {
struct SphMacroStep {
    int count;
    double dt;
    double end_time;
    double span;
};

// CFL is sampled once per closed macro. All its inner steps have equal dt,
// including a macro shortened to land exactly on an output/stage/stop event.
inline SphMacroStep plan_sph_macro(double time, double cfl_dt, int count,
                                   double event_time, double max_span = 0.0) {
    if (!(cfl_dt > 0.0) || !std::isfinite(cfl_dt) || count < 1)
        throw std::runtime_error("Invalid SPH macro timestep");
    if (max_span > 0.0) {
        const double steps = std::ceil(max_span / cfl_dt);
        if (!std::isfinite(steps) || steps > std::numeric_limits<int>::max())
            throw std::runtime_error("SPH gravity_macro_dt requires too many inner steps");
        count = std::max(1, static_cast<int>(steps));
    }
    double span = count * cfl_dt;
    if (max_span > 0.0) span = std::min(span, max_span);
    double end = time + span;
    if (event_time <= end) { end = event_time; span = end - time; }
    const double dt = span / count;
    if (!(dt > 0.0) || time + dt == time)
        throw std::runtime_error("SPH macro timestep cannot advance time");
    return {count, dt, end, span};
}
inline double next_sph_output_time(double time, double origin, double interval) {
    if (!(interval > 0.0)) return std::numeric_limits<double>::infinity();
    const double tick = std::floor((time - origin) / interval + 1.e-12) + 1.0;
    return origin + tick * interval;
}
}

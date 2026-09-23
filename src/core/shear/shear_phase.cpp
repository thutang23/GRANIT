#include "shear_box.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cudadem {

ShearPhase compute_shear_phase_checked(
    const ShearBoxConfig& shear, double Lx, double Ly, double time) {
    if (!(Lx > 0.0) || !(Ly > 0.0) || !std::isfinite(time))
        throw std::runtime_error("invalid shear phase domain/time");
    const long double gamma = static_cast<long double>(shear.shear_parameter)
                            * static_cast<long double>(shear.omega)
                            * static_cast<long double>(Lx);
    const long double full = gamma * (static_cast<long double>(time)
                           - static_cast<long double>(shear.time_origin));
    const long double ratio = full / static_cast<long double>(Ly);
    if (!std::isfinite(ratio)) throw std::runtime_error("shear phase is not finite");
    long double cyc = floorl(ratio + 0.5L);
    constexpr long double lo = static_cast<long double>(INT64_MIN);
    constexpr long double hi = static_cast<long double>(INT64_MAX);
    if (cyc < lo || cyc > hi)
        throw std::overflow_error("shear phase cycle exceeds int64");
    int64_t cycle = static_cast<int64_t>(cyc);
    long double reduced = full - cyc * static_cast<long double>(Ly);
    const long double half = 0.5L * static_cast<long double>(Ly);
    if (reduced >= half) {
        if (cycle == INT64_MAX)
            throw std::overflow_error("shear phase cycle addition overflow");
        reduced -= static_cast<long double>(Ly); ++cycle;
    } else if (reduced < -half) {
        if (cycle == INT64_MIN)
            throw std::overflow_error("shear phase cycle addition overflow");
        reduced += static_cast<long double>(Ly); --cycle;
    }
    ShearPhase phase;
    phase.reduced_shift = static_cast<double>(reduced);
    phase.shear_velocity = static_cast<double>(gamma);
    phase.cycle = cycle;
    if (!std::isfinite(phase.reduced_shift) || !std::isfinite(phase.shear_velocity))
        throw std::runtime_error("reduced shear phase is not finite");
    return phase;
}

} // namespace cudadem

#ifndef CUDADEM_DEM_TIMESTEP_SELECTION_H
#define CUDADEM_DEM_TIMESTEP_SELECTION_H

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cudadem {
// A supplied seconds-valued timestep is exact. Automatic legacy selection alone
// applies the safety factor and clips to the orbital limit.
inline double select_dem_timestep(double dt, double factor, double critical, double orbital) {
    if (!std::isfinite(dt) || dt < 0.0)
        throw std::runtime_error("simulation.time.dt must be finite and positive when supplied");
    if (!std::isfinite(factor) || !(factor > 0.0))
        throw std::runtime_error("simulation.time.dt_factor must be finite and positive");
    if (!std::isfinite(critical) || !(critical > 0.0))
        throw std::runtime_error("Unable to derive a positive DEM critical timestep");
    if (dt > 0.0) {
        if (dt > critical)
            throw std::runtime_error("simulation.time.dt exceeds the physical DEM critical timestep");
        if (dt > orbital)
            throw std::runtime_error("simulation.time.dt exceeds the shear orbital timestep limit");
        return dt;
    }
    return std::min(factor * critical, orbital);
}

inline void validate_dem_checkpoint_timestep(double saved, double allowed, double requested) {
    if (!std::isfinite(saved) || !(saved > 0.0))
        throw std::runtime_error("checkpoint timestep must be finite and positive");
    if (requested > 0.0 && std::fabs(saved - requested) >
            1.0e-12 * std::max(saved, requested))
        throw std::runtime_error("simulation.time.dt differs from the checkpoint timestep; resume preserves the saved base timestep");
    if (saved > allowed * (1.0 + 1.0e-12))
        throw std::runtime_error("checkpoint timestep exceeds current contact/orbital/configured limit");
}
}
#endif

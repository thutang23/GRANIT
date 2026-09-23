#ifndef CUDADEM_SPH_ENERGY_H
#define CUDADEM_SPH_ENERGY_H

#include <cuda_runtime.h>
#include "src/core/constant/system_initialize.h"

namespace cudadem {

// Compute de/dt: internal energy rate including PdV work, viscous heating, and shear heating.
// de/dt = -(P/rho^2)*div_v + visc_heat + shear_heat
void compute_sph_energy_rate(SimulationState& state, const SimConfig& config,
                               cudaStream_t stream);

// Integrate energy: e += de/dt * dt  (forward Euler)
void integrate_sph_energy(SimulationState& state, const SimConfig& config,
                           cudaStream_t stream);

} // namespace cudadem

#endif // CUDADEM_SPH_ENERGY_H

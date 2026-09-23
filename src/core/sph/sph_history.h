#ifndef CUDADEM_SPH_HISTORY_H
#define CUDADEM_SPH_HISTORY_H

#include "src/core/constant/system_initialize.h"

namespace cudadem {
void advance_sph_cold_energy(SimulationState& state, const SimConfig& config, cudaStream_t stream = 0);
void update_sph_history(SimulationState& state, const SimConfig& config, float dt, cudaStream_t stream = 0);
void damp_sph_relaxation(SimulationState& state, const SimConfig& config, double dt, cudaStream_t stream = 0);
} // namespace cudadem

#endif // CUDADEM_SPH_HISTORY_H

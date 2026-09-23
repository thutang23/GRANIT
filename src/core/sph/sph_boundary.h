#ifndef CUDADEM_SPH_BOUNDARY_H
#define CUDADEM_SPH_BOUNDARY_H

#include "src/core/constant/system_initialize.h"

namespace cudadem {
struct SphFixedState;
struct SphBoundaryWorkspace { SphFixedState* fixed = nullptr; };
void initialize_sph_boundaries(SimulationState& state, const SimConfig& config, SphBoundaryWorkspace& workspace, cudaStream_t stream);
void restore_sph_fixed(SimulationState& state, const SimConfig& config, const SphBoundaryWorkspace& workspace, cudaStream_t stream);
void reflect_sph_domain(SimulationState& state, const SimConfig& config, cudaStream_t stream);
void update_sph_mirrors(SimulationState& state, const SimConfig& config, cudaStream_t stream);
void shutdown_sph_boundaries(SphBoundaryWorkspace& workspace);
} // namespace cudadem

#endif // CUDADEM_SPH_BOUNDARY_H

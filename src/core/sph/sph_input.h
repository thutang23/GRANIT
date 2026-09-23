// =============================================================================
// SPH v2 Unified-Binary Input
// =============================================================================
// Runtime SPH input is deliberately shared with DEM.  Legacy CSV belongs in
// the offline migration script, never in the solver.
// =============================================================================
#ifndef CUDADEM_SPH_INPUT_H
#define CUDADEM_SPH_INPUT_H

#include "src/core/constant/command_line_interface.h"
#include "src/core/constant/system_initialize.h"

namespace cudadem {

// Reads config.io.input_particle_path as a strict v2 SPH unified binary.  The
// file is required to contain global_id, material_id, mass, and density.
int32_t load_sph_particles(ParticleSoA& particles, const SimConfig& config);

} // namespace cudadem

#endif // CUDADEM_SPH_INPUT_H

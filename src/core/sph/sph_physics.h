#ifndef CUDADEM_SPH_PHYSICS_H
#define CUDADEM_SPH_PHYSICS_H

#include <cuda_runtime.h>
#include "src/core/constant/command_line_interface.h"
#include "src/core/constant/system_initialize.h"

// Device-only headers (inline device functions)
#include "src/core/sph/sph_kernel_correction.cuh"
#include "src/core/sph/sph_artificial_stress.cuh"
#include "src/core/sph/sph_gravity_kernel.cuh"

// Host/device SPH physics modules
#include "src/core/sph/sph_strength.h"
#include "src/core/sph/sph_damage.h"
#include "src/core/sph/sph_porosity.h"
#include "src/core/sph/sph_energy.h"

namespace cudadem {

void compute_sph_density(SimulationState& state, const SimConfig& config, cudaStream_t stream, int32_t* debug_neighbor_count = nullptr);
void compute_sph_density_rate(SimulationState& state, const SimConfig& config, cudaStream_t stream);
void integrate_sph_density(SimulationState& state, const SimConfig& config, float dt, cudaStream_t stream);
void compute_sph_pressure(SimulationState& state, const SimConfig& config, cudaStream_t stream);
void compute_sph_forces(SimulationState& state, const SimConfig& config, cudaStream_t stream);
void initialize_sph_properties(SimulationState& state, const SimConfig& config);
void update_sph_smoothing_length(SimulationState& state, const SimConfig& config, float dt, cudaStream_t stream);
void update_sph_h_max_global(SimulationState& state, const SimConfig& config, cudaStream_t stream);
float compute_sph_h_min(const SimulationState& state, int32_t num_particles, cudaStream_t stream);
float compute_sph_h_max(const SimulationState& state, int32_t num_particles,
                        cudaStream_t stream);
void set_sph_h_max_global(float h_max, cudaStream_t stream);
void update_sph_support_extent(SimulationState& state, const SimConfig& config,
                               int32_t num_particles, cudaStream_t stream);

} // namespace cudadem

#endif // CUDADEM_SPH_PHYSICS_H

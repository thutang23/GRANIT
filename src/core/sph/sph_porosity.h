#ifndef CUDADEM_SPH_POROSITY_H
#define CUDADEM_SPH_POROSITY_H

#include <cuda_runtime.h>
#include "src/core/constant/system_initialize.h"

namespace cudadem {

// P-alpha rate buffers are derived each force evaluation and deliberately do
// not belong to ParticleSoA or checkpoints.  They provide the SPHSOL dt6/dt7
// constraints without carrying a stale constitutive rate across a restart or
// a domain migration.
struct SphPorosityRates {
    const float* dalpha_dt = nullptr;
    const float* dpdt = nullptr;       // porous-pressure rate
    int32_t capacity = 0;
};

void initialize_sph_porosity(int32_t max_particles);
void shutdown_sph_porosity();
SphPorosityRates sph_porosity_rates_for_current_device();

// Evaluate the closed-state P-alpha rates from the current rho-dot and
// e-dot.  This does not mutate alpha or pressure.
void evaluate_sph_porosity_rates(SimulationState& state, const SimConfig& config,
                                 cudaStream_t stream);

// Commit alpha after all rates for this state have been consumed.  The caller
// must recompute the EOS afterwards so pressure and sound speed use alpha_new.
void integrate_sph_porosity(SimulationState& state, const SimConfig& config,
                            float dt, cudaStream_t stream);

} // namespace cudadem

#endif // CUDADEM_SPH_POROSITY_H

#ifndef CUDADEM_SPH_STRENGTH_H
#define CUDADEM_SPH_STRENGTH_H

// =============================================================================
// SPH Elasto-Plastic Shear Strength
// =============================================================================
// Implements Hooke's law deviatoric stress evolution + plastic return mapping.
// Supports:
//   1. Von Mises yield: Y = Y0 (constant)
//   2. Lundborg model: Y = min(Y0, c0 + mu_i * P) (pressure-dependent)
// Damage scaling: Y_eff = (1 - D) * Y
//
// References:
//   Benz & Asphaug (1995), Libersky et al. (1993), Randles & Libersky (1996)
// =============================================================================

#include <cuda_runtime.h>
#include "src/core/constant/system_initialize.h"

namespace cudadem {

// Rebuild the Bonet--Lok correction matrices at the current particle state.
// This remains separate from strength so hydro-only runs use the same corrected
// gradients without advancing any constitutive state.
void update_sph_kernel_correction(SimulationState& state, const SimConfig& config,
                                  cudaStream_t stream);

// Trial stress and stress work are evaluated separately from the pressure-
// dependent radial return.  This lets the EOS close rho/e/alpha first, so the
// yield law consumes P_new while the energy work remains GRANIT's existing
// time-centred old/old-pressure-returned convention.
void initialize_sph_strength_workspace(int32_t max_particles);
void shutdown_sph_strength_workspace();
void predict_sph_strength_and_work(SimulationState& state, const SimConfig& config,
                                   float stress_dt, cudaStream_t stream);
void apply_sph_strength_radial_return(SimulationState& state, const SimConfig& config,
                                      cudaStream_t stream);

// Initialize deviatoric stress tensor to zero (called once at start)
// Also initializes distension (porosity) to alpha_initial
void initialize_sph_strength(SimulationState& state, const SimConfig& config);

} // namespace cudadem

#endif // CUDADEM_SPH_STRENGTH_H

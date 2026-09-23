#pragma once

#include <cuda_runtime.h>

namespace cudadem {

// Per-neighbor coefficient in the asymmetric geometric-density specific
// internal-energy equation.  The pressure term belongs to particle i.  The
// symmetric artificial-viscosity work is shared equally by both endpoints so
// the pair's thermal gain matches its kinetic-energy loss.
__host__ __device__ __forceinline__ float sph_energy_pair_coefficient(
    float pressure_i, float rho_i, float rho_j, float artificial_viscosity)
{
    return pressure_i / (rho_i * rho_j) + 0.5f * artificial_viscosity;
}

__host__ __device__ __forceinline__ double sph_energy_pair_coefficient(
    float pressure_i, double rho_i, double rho_j, float artificial_viscosity)
{
    return static_cast<double>(pressure_i) / (rho_i * rho_j)
         + 0.5 * static_cast<double>(artificial_viscosity);
}

} // namespace cudadem

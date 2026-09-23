#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

// Longitudinal elastic signal speed for an isotropic solid.  eos_sound_speed
// supplies the volumetric response; strength supplies the 4G/3 contribution.
// alpha>=1 scales the shear response for the P-alpha convention.
__host__ __device__ __forceinline__ float sph_longitudinal_signal_speed(
    float eos_sound_speed, float density, float shear_modulus,
    float alpha, bool strength_enabled)
{
    const double c = (isfinite(eos_sound_speed) && eos_sound_speed > 0.0f)
        ? static_cast<double>(eos_sound_speed) : 1.0;
    const double rho = (isfinite(density) && density > 0.0f)
        ? static_cast<double>(density) : 1.0;
    const double safe_alpha = (isfinite(alpha) && alpha > 1.0f)
        ? static_cast<double>(alpha) : 1.0;
    const double G = strength_enabled && isfinite(shear_modulus) && shear_modulus > 0.0f
        ? static_cast<double>(shear_modulus) / safe_alpha : 0.0;
    return static_cast<float>(sqrt(c * c + 4.0 * G / (3.0 * rho)));
}

} // namespace cudadem

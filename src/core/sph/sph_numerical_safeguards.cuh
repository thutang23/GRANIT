#pragma once
// =============================================================================
// SPH Numerical Safeguards
// =============================================================================
// Device-inline sanitize helpers that replace NaN/Inf with safe floor values.
// Applied at kernel I/O boundaries to prevent 1 NaN particle from
// contaminating all neighbors within 1 step (SoA array propagation).
//
// Pattern matches SPHSOL's 15+ MY_ASSERT(std::isfinite(...)) checks
// but as GPU-friendly device functions rather than CPU assertions.
// =============================================================================

#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

// Sanitize float: replace NaN/Inf with floor_val
__device__ __forceinline__ float sanitize_f(float v, float floor_val = 0.0f) {
    if (!isfinite(v)) return floor_val;
    return v;
}

// Sanitize double: replace NaN/Inf with floor_val
__device__ __forceinline__ double sanitize_d(double v, double floor_val = 0.0) {
    if (!isfinite(v)) return floor_val;
    return v;
}

// Sanitize density: clamp to [rho_floor, rho_max] and replace NaN
__device__ __forceinline__ float sanitize_density(float rho, float rho_floor, float rho_max = 1e10f) {
    if (!isfinite(rho) || rho < rho_floor) return rho_floor;
    if (rho > rho_max) return rho_max;
    return rho;
}

__device__ __forceinline__ double sanitize_density(
    double rho, double rho_floor, double rho_max = 1e10) {
    if (!isfinite(rho) || rho < rho_floor) return rho_floor;
    if (rho > rho_max) return rho_max;
    return rho;
}

// Sanitize pressure: replace NaN with 0, clamp to [p_min, p_max]
__device__ __forceinline__ float sanitize_pressure(float p, float p_min = -1e15f, float p_max = 1e15f) {
    if (!isfinite(p)) return 0.0f;
    if (p < p_min) return p_min;
    if (p > p_max) return p_max;
    return p;
}

// Sanitize smoothing length: clamp to [h_min, h_max] and replace NaN
__device__ __forceinline__ float sanitize_hsml(float h, float h_min, float h_max) {
    if (!isfinite(h) || h < h_min) return h_min;
    if (h > h_max) return h_max;
    return h;
}

// Sanitize sound speed: replace NaN with floor, clamp to [0, c_max]
__device__ __forceinline__ float sanitize_c_sound(float c, float c_floor = 1.0f, float c_max = 1e10f) {
    if (!isfinite(c) || c < c_floor) return c_floor;
    if (c > c_max) return c_max;
    return c;
}

} // namespace cudadem

#pragma once
// =============================================================================
// SPH Device Helper Functions (shared across all SPH translation units)
// Include this header in any .cu file that uses SPH kernel evaluations or
// Morton-code cell lookups.
// =============================================================================

#include <cstdint>

namespace cudadem {

// Global max smoothing length for symmetric SPH neighbor queries.
// Updated each step via update_sph_h_max_global(). Matches SPHSOL grid.cc:87 pattern.
extern __constant__ float dev_sph_h_max_global;

// Expand 10-bit integer to 30 bits by inserting 2 zeros between each bit
__device__ __forceinline__ uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Compute 30-bit Morton code from 3D cell coordinates (each max 10 bits = 1024)
__device__ __forceinline__ uint32_t morton_3d(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

// Cubic Spline Kernel W(r, h)
__device__ inline float sph_kernel_w(float r, float h) {
    float q = r / h;
    float norm = 1.0f / (3.14159265359f * h * h * h);
    if (q < 1.0f) {
        return norm * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    } else if (q < 2.0f) {
        float d = 2.0f - q;
        return norm * 0.25f * d * d * d;
    }
    return 0.0f;
}

// Gradient magnitude of Cubic Spline Kernel: |dW/dr|
// Actual gradient vector: |dW/dr| * (x_ij / r)
__device__ inline float sph_kernel_grad(float r, float h) {
    float q = r / h;
    float norm = 1.0f / (3.14159265359f * h * h * h * h);
    if (q < 1.0f) {
        return norm * (-3.0f * q + 2.25f * q * q);
    } else if (q < 2.0f) {
        float d = 2.0f - q;
        return norm * -0.75f * d * d;
    }
    return 0.0f;
}

} // namespace cudadem

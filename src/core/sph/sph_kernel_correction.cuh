#ifndef CUDADEM_SPH_KERNEL_CORRECTION_CUH
#define CUDADEM_SPH_KERNEL_CORRECTION_CUH

// =============================================================================
// SPH Kernel Gradient Correction (CSPH / Bonet-Lok)
// =============================================================================
// Computes per-particle renormalization matrix L_i = (sum_j m_j/rho_j * r_ji ⊗ grad_W_ij)^{-1}
// Corrected gradient: grad_W_corrected_ij = L_i * grad_W_ij
// This restores first-order consistency and eliminates the bias at boundaries.
// Reference: Bonet & Lok (1999), Randles & Libersky (1996)
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "src/core/neighbor/neighbor_types.h"

namespace cudadem {

// -----------------------------------------------------------------------------
// 3x3 Matrix Inverse via Cramer's rule (device inline)
// Input:  row-major 3x3 matrix M[row*3+col]
// Output: row-major 3x3 inverse Minv[row*3+col]
// Returns false if matrix is singular (det ~ 0); Minv set to identity.
//
// NOTE: No blend/regularization is applied here. The caller (compute_csph_
// correction_kernel) handles the completeness-based blend with identity to
// match SPHSOL equations.cc:421-432.
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool mat3_inverse(const float M[9], float Minv[9]) {
    // Cofactors
    float C00 = M[4]*M[8] - M[5]*M[7];
    float C01 = M[5]*M[6] - M[3]*M[8];
    float C02 = M[3]*M[7] - M[4]*M[6];
    float C10 = M[2]*M[7] - M[1]*M[8];
    float C11 = M[0]*M[8] - M[2]*M[6];
    float C12 = M[1]*M[6] - M[0]*M[7];
    float C20 = M[1]*M[5] - M[2]*M[4];
    float C21 = M[2]*M[3] - M[0]*M[5];
    float C22 = M[0]*M[4] - M[1]*M[3];

    float det = M[0]*C00 + M[1]*C01 + M[2]*C02;
    float abs_det = fabsf(det);

    if (abs_det < 1.0e-30f) {
        // Singular matrix: return identity, signal failure.
        Minv[0] = 1.0f; Minv[1] = 0.0f; Minv[2] = 0.0f;
        Minv[3] = 0.0f; Minv[4] = 1.0f; Minv[5] = 0.0f;
        Minv[6] = 0.0f; Minv[7] = 0.0f; Minv[8] = 1.0f;
        return false;
    }

    float inv_det = 1.0f / det;

    // Raw adjugate / det — NO blend (blend moved to caller)
    Minv[0] = C00 * inv_det;
    Minv[1] = C10 * inv_det;
    Minv[2] = C20 * inv_det;
    Minv[3] = C01 * inv_det;
    Minv[4] = C11 * inv_det;
    Minv[5] = C21 * inv_det;
    Minv[6] = C02 * inv_det;
    Minv[7] = C12 * inv_det;
    Minv[8] = C22 * inv_det;

    return true;
}

// -----------------------------------------------------------------------------
// Apply CSPH correction: corrected grad = L_i * grad_ij
// L_i stored as SoA 9 components (row-major): L_00..L_22
// Input grad_x/y/z: standard kernel gradient components (grad_W * e_ij / r)
// Output: corrected gradient in-place
// -----------------------------------------------------------------------------
__device__ __forceinline__ void apply_kernel_correction(
    float& grad_x, float& grad_y, float& grad_z,
    float L00, float L01, float L02,
    float L10, float L11, float L12,
    float L20, float L21, float L22)
{
    float gx = L00 * grad_x + L01 * grad_y + L02 * grad_z;
    float gy = L10 * grad_x + L11 * grad_y + L12 * grad_z;
    float gz = L20 * grad_x + L21 * grad_y + L22 * grad_z;
    grad_x = gx;
    grad_y = gy;
    grad_z = gz;
}

// -----------------------------------------------------------------------------
// Kernel: Build correction matrix B_i = sum_j (m_j/rho_j) * r_ji ⊗ grad_W_ij
// then invert to get L_i = B_i^{-1}
// If singular, L_i = identity (fall back to uncorrected)
// -----------------------------------------------------------------------------
__global__ void compute_csph_correction_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float*  __restrict__ mass,
    const double* __restrict__ density,
    const float*  __restrict__ smoothing_length,
    float* __restrict__ L_00, float* __restrict__ L_01, float* __restrict__ L_02,
    float* __restrict__ L_10, float* __restrict__ L_11, float* __restrict__ L_12,
    float* __restrict__ L_20, float* __restrict__ L_21, float* __restrict__ L_22,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    int32_t num_particles
);

} // namespace cudadem

#endif // CUDADEM_SPH_KERNEL_CORRECTION_CUH

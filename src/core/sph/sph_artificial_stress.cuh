#ifndef CUDADEM_SPH_ARTIFICIAL_STRESS_CUH
#define CUDADEM_SPH_ARTIFICIAL_STRESS_CUH

// =============================================================================
// SPH Artificial Stress (Tension Instability Control)
// =============================================================================
// Implements the Swegle-Attaway-Gray (1995) artificial stress method.
// When particles are in tension, the pressure gradient term becomes repulsive
// instead of attractive, causing particle clumping (tension instability).
// The artificial stress adds a small repulsive term proportional to W/W_delta
// raised to the power n, eliminating tensile instability.
//
// Reference: Swegle, Hicks & Attaway (1995), Gray, Monaghan & Swift (2001)
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "src/core/sph/sph_eigen_3x3.cuh"

namespace cudadem {

// -----------------------------------------------------------------------------
// Compute principal artificial stress for particle i
// For each principal stress sigma_a:
//   if sigma_a > 0 (tension): R_aa = -epsilon * sigma_a / rho_i^2
//   if sigma_a <= 0 (compression): R_aa = 0
// The full Cauchy stress is rotated to its principal frame, tensile
// eigenvalues are retained, and the correction is reconstructed in the
// simulation frame.  This follows SPHSOL's strength-aware treatment and is
// required for non-axis-aligned shear tension.
//
// Returns the symmetric tensor R_ij components needed in:
//   a_i += m_j * (R_i + R_j) * f^n * grad_W_ij
// where f = W(r, h) / W(dp, h) and dp is mean particle spacing = h / 1.2
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__ void compute_artificial_stress_tensor(
    float pressure_i,        // scalar pressure (positive = compression)
    float S_xx_i, float S_xy_i, float S_xz_i,
    float S_yy_i, float S_yz_i, float S_zz_i,
    float rho_i,
    float epsilon,           // artificial stress magnitude (~0.3)
    float& R_xx, float& R_xy, float& R_xz,
    float& R_yy, float& R_yz, float& R_zz)
{
    // Full Cauchy stress components: sigma_ab = -P * delta_ab + S_ab
    float sigma_xx = -pressure_i + S_xx_i;
    float sigma_xy = S_xy_i;
    float sigma_xz = S_xz_i;
    float sigma_yy = -pressure_i + S_yy_i;
    float sigma_yz = S_yz_i;
    float sigma_zz = -pressure_i + S_zz_i;

    float rho2 = rho_i * rho_i;
    if (rho2 < 1.0e-30f) rho2 = 1.0e-30f;

    float eigenvalues[3];
    float eigenvectors[9];
    jacobi_eigen_3x3(sigma_xx, sigma_xy, sigma_xz,
                     sigma_yy, sigma_yz, sigma_zz,
                     eigenvalues, eigenvectors);

    for (int k = 0; k < 3; ++k) {
        eigenvalues[k] = fmaxf(eigenvalues[k], 0.0f);
    }

    float fac = -epsilon / rho2;

    // eigenvectors are stored by columns: R_ab = fac sum_k V_ak l_k V_bk.
    R_xx = fac * (eigenvectors[0] * eigenvalues[0] * eigenvectors[0] +
                  eigenvectors[1] * eigenvalues[1] * eigenvectors[1] +
                  eigenvectors[2] * eigenvalues[2] * eigenvectors[2]);
    R_xy = fac * (eigenvectors[0] * eigenvalues[0] * eigenvectors[3] +
                  eigenvectors[1] * eigenvalues[1] * eigenvectors[4] +
                  eigenvectors[2] * eigenvalues[2] * eigenvectors[5]);
    R_xz = fac * (eigenvectors[0] * eigenvalues[0] * eigenvectors[6] +
                  eigenvectors[1] * eigenvalues[1] * eigenvectors[7] +
                  eigenvectors[2] * eigenvalues[2] * eigenvectors[8]);
    R_yy = fac * (eigenvectors[3] * eigenvalues[0] * eigenvectors[3] +
                  eigenvectors[4] * eigenvalues[1] * eigenvectors[4] +
                  eigenvectors[5] * eigenvalues[2] * eigenvectors[5]);
    R_yz = fac * (eigenvectors[3] * eigenvalues[0] * eigenvectors[6] +
                  eigenvectors[4] * eigenvalues[1] * eigenvectors[7] +
                  eigenvectors[5] * eigenvalues[2] * eigenvectors[8]);
    R_zz = fac * (eigenvectors[6] * eigenvalues[0] * eigenvectors[6] +
                  eigenvectors[7] * eigenvalues[1] * eigenvectors[7] +
                  eigenvectors[8] * eigenvalues[2] * eigenvectors[8]);
}

// -----------------------------------------------------------------------------
// Compute the artificial stress correction to acceleration for pair (i,j)
// and add it to the force accumulator.
//
// f_factor = (W_ij / W_dp)^n  where W_dp = W(spacing, h_ij)
// Contribution: da_i += m_j * (R_i + R_j) : (e_ij ⊗ grad_W_ij_vec)
//            = m_j * [(R_i_ab + R_j_ab) * e_ij_b] * dW_dr_vec_a
// For pressure-gradient style:
//   accel_artstr_x += m_j * [(Ri_xx+Rj_xx)*nx + (Ri_xy+Rj_xy)*ny + (Ri_xz+Rj_xz)*nz] * dW_mag * f^n
// where nx,ny,nz = e_ij direction
// -----------------------------------------------------------------------------
__device__ __forceinline__ void accumulate_artificial_stress(
    float m_j,
    float dW_dr,         // magnitude of kernel gradient (scalar, from sph_kernel_grad)
    float dist,          // |r_ij|
    float dx_ij, float dy_ij, float dz_ij, // r_i - r_j (not normalized)
    float W_ij,          // W(r_ij, h_ij)
    float W_dp,          // W(dp, h_ij)  (particle-spacing kernel value)
    float artstr_n,      // exponent (typically 4)
    // Artificial stress tensors for i and j
    float Ri_xx, float Ri_xy, float Ri_xz,
    float Ri_yy, float Ri_yz, float Ri_zz,
    float Rj_xx, float Rj_xy, float Rj_xz,
    float Rj_yy, float Rj_yz, float Rj_zz,
    float& fx, float& fy, float& fz)   // output: accumulated acceleration
{
    if (dist < 1.0e-15f) return;

    // f^n weight
    float f_ratio = (W_dp > 1.0e-30f) ? (W_ij / W_dp) : 0.0f;
    float fn = powf(fabsf(f_ratio), artstr_n);

    float inv_dist = 1.0f / dist;
    float nx = dx_ij * inv_dist;
    float ny = dy_ij * inv_dist;
    float nz = dz_ij * inv_dist;

    // Combined tensor
    float Rxx = Ri_xx + Rj_xx;
    float Rxy = Ri_xy + Rj_xy;
    float Rxz = Ri_xz + Rj_xz;
    float Ryy = Ri_yy + Rj_yy;
    float Ryz = Ri_yz + Rj_yz;
    float Rzz = Ri_zz + Rj_zz;

    // Contract R with e_ij direction: (R * n)_a = R_ab * n_b
    float Rn_x = Rxx * nx + Rxy * ny + Rxz * nz;
    float Rn_y = Rxy * nx + Ryy * ny + Ryz * nz;
    float Rn_z = Rxz * nx + Ryz * ny + Rzz * nz;

    // dW/dr * e_ij gives the gradient vector (pointing from j to i)
    float scale = m_j * dW_dr * fn;

    fx += scale * Rn_x;
    fy += scale * Rn_y;
    fz += scale * Rn_z;
}

} // namespace cudadem

#endif // CUDADEM_SPH_ARTIFICIAL_STRESS_CUH

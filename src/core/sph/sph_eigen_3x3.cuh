#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

// Jacobi eigenvalue decomposition for symmetric 3x3 matrix.
// Input: symmetric matrix A (row-major, 6 unique values: xx, xy, xz, yy, yz, zz)
// Output: eigenvalues[3], eigenvectors[9] (row-major, columns are eigenvectors)
// Convergence: ~5-10 iterations for 3x3 (empirically: 3 for diagonal matrices,
// ~6-8 for matrices with moderate off-diagonal coupling).
//
// This enables SPHSOL-style eigenvalue-based artificial stress (solver.cc:178-189)
// where the stress tensor is eigendecomposed, tensile eigenvalues are clamped to
// >= 0, and the artificial stress tensor is reconstructed from the clamped
// eigenvalues. This correctly handles non-axis-aligned tension that the simpler
// coordinate-frame diagonal-only check would miss.
//
// Note: For pure-pressure SPH (no strength model, S = 0), the stress tensor
// is -P * I (diagonal). The eigenvalues are all equal (-P), so the eigenvalue
// version gives identical results to the scalar R = 0.2 * max(-P, 0) / rho^2.
// The eigenvalue version only differs when deviatoric stress (off-diagonal
// components) is present (use_strength = true).
__host__ __device__ __forceinline__ void jacobi_eigen_3x3(
    float Axx, float Axy, float Axz, float Ayy, float Ayz, float Azz,
    float eigenvalues[3], float eigenvectors[9])
{
    // Initialize eigenvectors to identity
    float V[9] = {1,0,0, 0,1,0, 0,0,1};

    // Working copy of A
    float a[9] = {Axx, Axy, Axz,
                  Axy, Ayy, Ayz,
                  Axz, Ayz, Azz};

    const int MAX_ITER = 20;
    const float EPS = 1e-6f;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest off-diagonal element
        float max_off = fabsf(a[1]);  // a[0][1]
        int p = 0, q = 1;
        if (fabsf(a[2]) > max_off) { max_off = fabsf(a[2]); p = 0; q = 2; }
        if (fabsf(a[5]) > max_off) { max_off = fabsf(a[5]); p = 1; q = 2; }

        const float diagonal_scale = fmaxf(
            1.0f, fmaxf(fabsf(a[0]), fmaxf(fabsf(a[4]), fabsf(a[8]))));
        if (max_off <= EPS * diagonal_scale) break;

        // Stable symmetric Jacobi rotation A' = J^T A J.  Updating only the
        // affected rows/columns preserves symmetry and keeps each eigenvalue
        // paired with the accumulated eigenvector column.
        float app = a[p*3+p], aqq = a[q*3+q], apq = a[p*3+q];
        const float tau = (aqq - app) / (2.0f * apq);
        const float sign = tau >= 0.0f ? 1.0f : -1.0f;
        const float t = sign / (fabsf(tau) + sqrtf(1.0f + tau*tau));
        const float c = 1.0f / sqrtf(1.0f + t*t);
        const float s = t*c;

        a[p*3+p] = app - t*apq;
        a[q*3+q] = aqq + t*apq;
        a[p*3+q] = 0.0f;
        a[q*3+p] = 0.0f;
        for (int r = 0; r < 3; ++r) {
            if (r == p || r == q) continue;
            const float arp = a[r*3+p];
            const float arq = a[r*3+q];
            const float new_rp = c*arp - s*arq;
            const float new_rq = s*arp + c*arq;
            a[r*3+p] = a[p*3+r] = new_rp;
            a[r*3+q] = a[q*3+r] = new_rq;
        }

        // Accumulate eigenvectors by columns using the same J.
        for (int i = 0; i < 3; i++) {
            float vip = V[i*3+p], viq = V[i*3+q];
            V[i*3+p] = c*vip - s*viq;
            V[i*3+q] = s*vip + c*viq;
        }
    }

    // Eigenvalues are diagonal of A
    eigenvalues[0] = a[0];
    eigenvalues[1] = a[4];
    eigenvalues[2] = a[8];

    // Eigenvectors are columns of V
    for (int i = 0; i < 9; i++) eigenvectors[i] = V[i];
}

} // namespace cudadem

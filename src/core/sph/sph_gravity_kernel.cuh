#ifndef CUDADEM_SPH_GRAVITY_KERNEL_CUH
#define CUDADEM_SPH_GRAVITY_KERNEL_CUH

// =============================================================================
// Gravity Softening Kernel (Cubic Spline)
// =============================================================================
// Replaces the 1/r^2 Newtonian singularity in self-gravity with a cubic-spline
// softened kernel. This avoids numerical divergence when particles are close.
//
// Reference: Price & Monaghan (2007), Hernquist & Katz (1989)
//
// For h_soft = 0, falls back to 1/r^2 (no softening).
// The softened force matches the exact 1/r^2 for r > 2*h_soft.
//
// Gravitational acceleration on particle i due to particle j:
//   g_i += -G * m_j * f(r, h_soft) * r_ij / r
// where f(r, h_soft) = phi'(r, h_soft) / r  (effective 1/r^3 replacement)
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem {

// Select the pair softening length.  query_particle is deliberately
// directional and exists to reproduce SPHSOL's h_i gravity operator; the
// symmetric default preserves reciprocal pair softening in GRANIT.
__host__ __device__ __forceinline__
double select_sph_gravity_softening_length(
    double h_query, double h_neighbor, bool query_particle)
{
    return query_particle ? h_query : 0.5 * (h_query + h_neighbor);
}

// -----------------------------------------------------------------------------
// Cubic spline gravity softening: force factor phi'(r) / r
// Returns the value used in: g_ij = -G * m_j * phi_force(r, h) * r_vec
// Units: [1/m^2] when r in meters, h in meters
//
// q = r / h_soft
// For q < 1:    phi' / r = (4/3 - (6/5)*q^2 + (1/2)*q^3) / h^3
// For 1 <= q < 2: phi' / r = (8/3 - 3*q + (6/5)*q^2 - (1/3)*q^3 - 1/(15*q^3)) / h^3
// For q >= 2:   phi' / r = 1/r^3  (exact Newtonian)
// -----------------------------------------------------------------------------
__device__ __forceinline__ float gravity_softening_force_factor(float r, float h_soft) {
    if (h_soft <= 0.0f || r <= 0.0f) {
        // No softening: return 1/r^3 (caller multiplies by r_vec to get g/r => -G*m/r^2 * e_r)
        float r2 = r * r + 1.0e-60f;
        return 1.0f / (r2 * sqrtf(r2));
    }

    float q = r / h_soft;
    float h3 = h_soft * h_soft * h_soft;

    if (q < 1.0f) {
        // Fully softened inner region
        return (4.0f/3.0f - (6.0f/5.0f)*q*q + 0.5f*q*q*q) / h3;
    } else if (q < 2.0f) {
        // Transition region
        float q2 = q * q;
        float q3 = q2 * q;
        return (8.0f/3.0f - 3.0f*q + (6.0f/5.0f)*q2 - (1.0f/6.0f)*q3 - 1.0f/(15.0f*q3)) / h3;
    } else {
        // Outside softening radius: exact Newtonian
        float r2 = r * r;
        return 1.0f / (r2 * r);
    }
}

// -----------------------------------------------------------------------------
// Cubic spline gravity softening: gravitational potential phi(r)
// phi(r) = -G*m * Phi_soft(r, h) / r  (the potential, not the force)
//
// q = r / h_soft
// For q < 1:    -1/h * [7/5 - (2/3)*q^2 + (3/10)*q^4 - (1/10)*q^5]  <- Price&Monaghan
// For 1 <= q < 2: -1/h * [8/5 - (4/3)*q^2 + q^3 - (3/10)*q^4 + (1/30)*q^5 - 1/(15*q)]
// For q >= 2:   -1/r
// Returns the dimensionless Phi such that phi = G*m * Phi (negative)
// -----------------------------------------------------------------------------
__device__ __forceinline__ float gravity_softening_potential(float r, float h_soft) {
    if (h_soft <= 0.0f || r <= 0.0f) {
        return -1.0f / fmaxf(r, 1.0e-30f);
    }

    float q = r / h_soft;

    if (q < 1.0f) {
        float q2 = q * q;
        float q4 = q2 * q2;
        float q5 = q4 * q;
        return -(1.0f / h_soft) * (1.4f - (2.0f/3.0f)*q2 + 0.3f*q4 - 0.1f*q5);
    } else if (q < 2.0f) {
        float q2 = q * q;
        float q3 = q2 * q;
        float q4 = q3 * q;
        float q5 = q4 * q;
        return -(1.0f / h_soft) * (8.0f/5.0f - (4.0f/3.0f)*q2 + q3 - 0.3f*q4
                                   + (1.0f/30.0f)*q5 - 1.0f/(15.0f*q));
    } else {
        return -1.0f / r;
    }
}

// -----------------------------------------------------------------------------
// Compute softened gravitational acceleration contribution from particle j on i
// Adds to (gx, gy, gz):
//   delta_g = -G * m_j * phi_force_factor(r, h) * r_vec_ij
// where r_vec_ij = pos_i - pos_j  (pointing from j to i)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void accumulate_softened_gravity(
    float G_const,      // gravitational constant
    float m_j,          // mass of particle j
    float dx, float dy, float dz,  // pos_i - pos_j
    float r,            // |r_ij|
    float h_soft,       // softening length
    float& gx, float& gy, float& gz)
{
    float ff = gravity_softening_force_factor(r, h_soft);
    float scale = -G_const * m_j * ff;
    gx += scale * dx;
    gy += scale * dy;
    gz += scale * dz;
}

// -----------------------------------------------------------------------------
// Double-precision variant of gravity_softening_force_factor.
// Host-callable (__host__ __device__) for unit-test access.
//
// Returns phi'(r, h) / r, the scalar multiplied by r_vec to get the
// gravitational acceleration contribution:
//   a_i += -G * m_j * gravity_softening_force_factor_double(r, h) * (pos_i - pos_j)
//
// For h_soft <= 0: returns 1/r^3 (exact Newtonian, DEM-safe fallback).
// For r >= 2*h:    returns 1/r^3 (exact Newtonian).
// For r < 2*h:     cubic-spline softened form (Price & Monaghan 2007).
//
// Matches SPHSOL KernelSoften(r, h) / r — see lib/SPHSOL/src/kernel.cc:93-122.
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
double gravity_softening_force_factor_double(double r, double h_soft) {
    if (h_soft <= 0.0 || r <= 0.0) {
        double r2 = r * r + 1.0e-60;
        return 1.0 / (r2 * sqrt(r2));
    }

    double q = r / h_soft;
    double h3 = h_soft * h_soft * h_soft;

    if (q < 1.0) {
        return (4.0/3.0 - (6.0/5.0)*q*q + 0.5*q*q*q) / h3;
    } else if (q < 2.0) {
        double q2 = q * q;
        double q3 = q2 * q;
        return (8.0/3.0 - 3.0*q + (6.0/5.0)*q2 - (1.0/6.0)*q3 - 1.0/(15.0*q3)) / h3;
    } else {
        double r2 = r * r;
        return 1.0 / (r2 * r);
    }
}

} // namespace cudadem

#endif // CUDADEM_SPH_GRAVITY_KERNEL_CUH

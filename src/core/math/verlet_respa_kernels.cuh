// =============================================================================
// RESPA (Reversible Reference System Propagator Algorithm) Kernels
// =============================================================================
// Half-gravity-impulse kernel for RESPA multiple time stepping.
//
// In RESPA, self-gravity is computed every K inner steps and applied as
// bracketing half-impulses. Between the half-impulses, inner steps integrate
// only contact/wall/inertial/tidal/field forces (gravity-free).
//
// Usage: Called from the outer step in SimulationRunner::run() (main.cu).
// The gravity array (gravity_force_x/y/z) stores GRAVITATIONAL ACCELERATION
// in m/s², not force. This kernel applies: v += grav_acc * (0.5 * K * dt).
// =============================================================================

#ifndef CUDADEM_VERLET_RESPA_KERNELS_H
#define CUDADEM_VERLET_RESPA_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// apply_half_gravity_impulse_kernel
// =============================================================================
// Applies a velocity half-impulse from cached gravitational acceleration.
//
// Called twice per outer step:
//   1. With gravity computed at the START of the outer step (old positions)
//   2. With gravity re-computed at the END of the outer step (new positions)
//
// Parameters:
//   vel_x/y/z      : particle velocity arrays (read + write)
//   grav_acc_x/y/z : cached gravitational acceleration in m/s² (read only)
//   half_outer_dt  : 0.5 * k * dt  (half of the outer timestep)
//   num_particles  : number of particles to process (free particles only)
// =============================================================================

__global__ void apply_half_gravity_impulse_kernel(
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    const double* __restrict__ grav_acc_x,
    const double* __restrict__ grav_acc_y,
    const double* __restrict__ grav_acc_z,
    double half_outer_dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    vel_x[idx] += grav_acc_x[idx] * half_outer_dt;
    vel_y[idx] += grav_acc_y[idx] * half_outer_dt;
    vel_z[idx] += grav_acc_z[idx] * half_outer_dt;
}

#endif // CUDADEM_VERLET_RESPA_KERNELS_H

#ifndef CUDADEM_VERLET_INTEGRATION_CUH
#define CUDADEM_VERLET_INTEGRATION_CUH

// =============================================================================
// Velocity Verlet Integration for DEM Simulations
// =============================================================================
// Implements time integration for translational and rotational motion
//
// Key Features:
// - Velocity Verlet (symplectic, time-reversible, good energy conservation)
// - Double precision accumulation for position updates (critical for accuracy)
// - Quaternion integration for orientation (avoids gimbal lock)
// - Automatic timestep calculation (10-20% of Rayleigh wave time)
//
// Reference:
// - Swope et al. (1982) "A computer simulation method for the calculation
//   of equilibrium constants for the formation of physical clusters"
// - Allen & Tildesley, "Computer Simulation of Liquids" (2017)
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../constant/command_line_interface.h"
#include "../sph/sph_numerical_safeguards.cuh"

namespace cudadem {

// =============================================================================
// Critical Timestep Calculation
// =============================================================================
// The critical timestep for DEM is based on the Rayleigh wave speed
// dt_crit = π * R_min * sqrt(ρ / G) / (0.1631*ν + 0.8766)
//
// For linear spring model: dt_crit ∝ sqrt(m_min / k_n)
// Use 10-20% of critical timestep for stability with damping
// =============================================================================

struct TimestepParams {
    float safety_factor;        // 0.1 - 0.2 (10-20% of critical)
    float min_particle_mass;    // Minimum particle mass in system
    float density;              // Bulk density
    float min_radius;           // Minimum particle radius
    float poisson_ratio;        // Poisson's ratio (for Rayleigh calculation)
    float youngs_modulus;       // Young's modulus E (for Rayleigh calculation)

    TimestepParams()
        : safety_factor(0.15f),   // 15% of critical (middle of range)
          min_particle_mass(0),
          density(0),
          min_radius(0),
          poisson_ratio(0.25f),
          youngs_modulus(0) {}
};

// Calculate critical timestep using linear spring approximation
// For Hertz-Mindlin, we approximate k_n ≈ E * R_min
// dt_crit = sqrt(m / k) * π
__host__ inline
double calculate_critical_timestep_linear(const TimestepParams& params) {
    if (params.min_particle_mass <= 0 || params.youngs_modulus <= 0 || params.min_radius <= 0) {
        return 1e-6;  // Default fallback
    }

    // Rough approximation of normal stiffness for Hertz contact
    double k_n = static_cast<double>(params.youngs_modulus) * static_cast<double>(params.min_radius);
    double dt_crit = M_PI * sqrt(static_cast<double>(params.min_particle_mass) / k_n);
    return params.safety_factor * dt_crit;
}

// Calculate critical timestep using Rayleigh wave criterion
// dt_crit = π * R * sqrt(ρ/E) / (0.163*ν + 0.877)
__host__ inline
double calculate_critical_timestep_rayleigh(const TimestepParams& params) {
    if (params.min_radius <= 0 || params.density <= 0 || params.youngs_modulus <= 0) {
        // Fall back to linear spring method
        return calculate_critical_timestep_linear(params);
    }

    double nu = params.poisson_ratio;
    double rayleigh_factor = 0.163 * nu + 0.877;

    double dt_crit = M_PI * (static_cast<double>(params.min_radius) / rayleigh_factor) *
                     sqrt(static_cast<double>(params.density) /
                          static_cast<double>(params.youngs_modulus));

    return params.safety_factor * dt_crit;
}

// Auto-select best timestep calculation method
__host__ inline
double calculate_timestep(const TimestepParams& params) {
    double dt_linear = calculate_critical_timestep_linear(params);

    if (params.youngs_modulus > 0) {
        double dt_rayleigh = calculate_critical_timestep_rayleigh(params);
        // Use the more conservative (smaller) timestep
        return std::min(dt_linear, dt_rayleigh);
    }

    return dt_linear;
}

// =============================================================================
// Velocity Verlet Integration - Translational Motion
// =============================================================================
// Two-step algorithm:
//   Step 1 (before force calculation):
//     v(t + dt/2) = v(t) + (1/2) * a(t) * dt
//     r(t + dt) = r(t) + v(t + dt/2) * dt
//
//   Step 2 (after force calculation):
//     v(t + dt) = v(t + dt/2) + (1/2) * a(t + dt) * dt
//
// CRITICAL: Position updates use double precision accumulation to prevent
// loss of precision when particles have large absolute coordinates.
// =============================================================================

// Half-step velocity update and full-step position update
// Call BEFORE force calculation
static __global__ void velocity_verlet_step1_kernel(
    // Position (double precision - CRITICAL)
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    // Velocity (double precision for consistency with particle data)
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    // Acceleration from previous step (double)
    const double* __restrict__ acc_x,
    const double* __restrict__ acc_y,
    const double* __restrict__ acc_z,
    // Mass (for computing acceleration from force)
    const float* __restrict__ mass,
    // Force (if using force instead of acceleration)
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    // Use force or acceleration
    bool use_force,
    // Timestep (double for precision)
    double dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double ax, ay, az;
    if (use_force) {
        double inv_m = 1.0 / static_cast<double>(mass[idx]);
        ax = force_x[idx] * inv_m;
        ay = force_y[idx] * inv_m;
        az = force_z[idx] * inv_m;
    } else {
        ax = acc_x[idx];
        ay = acc_y[idx];
        az = acc_z[idx];
    }

    // Half-step velocity update: v(t + dt/2) = v(t) + (1/2) * a(t) * dt
    double half_dt = dt * 0.5;
    double vx = vel_x[idx] + ax * half_dt;
    double vy = vel_y[idx] + ay * half_dt;
    double vz = vel_z[idx] + az * half_dt;

    vel_x[idx] = sanitize_d(vx);
    vel_y[idx] = sanitize_d(vy);
    vel_z[idx] = sanitize_d(vz);

    // Full-step position update: r(t + dt) = r(t) + v(t + dt/2) * dt
    // CRITICAL: Use double precision arithmetic for position accumulation
    // to prevent catastrophic cancellation with large coordinates
    pos_x[idx] = sanitize_d(pos_x[idx] + vx * dt);
    pos_y[idx] = sanitize_d(pos_y[idx] + vy * dt);
    pos_z[idx] = sanitize_d(pos_z[idx] + vz * dt);
}

// Complete the velocity update after force calculation
// Call AFTER force calculation
static __global__ void velocity_verlet_step2_kernel(
    // Velocity (double precision)
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    // New acceleration (double)
    const double* __restrict__ acc_x,
    const double* __restrict__ acc_y,
    const double* __restrict__ acc_z,
    // Mass
    const float* __restrict__ mass,
    // New force
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    // Use force or acceleration
    bool use_force,
    // Timestep
    double dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double ax, ay, az;
    if (use_force) {
        double inv_m = 1.0 / static_cast<double>(mass[idx]);
        ax = force_x[idx] * inv_m;
        ay = force_y[idx] * inv_m;
        az = force_z[idx] * inv_m;
    } else {
        ax = acc_x[idx];
        ay = acc_y[idx];
        az = acc_z[idx];
    }

    // Clamp acceleration magnitude for numerical stability
    if (dev_config.acceleration.enable_accel_limit) {
        double a_mag = sqrt(ax*ax + ay*ay + az*az);
        if (a_mag > dev_config.acceleration.max_accel_magnitude) {
            double scale = dev_config.acceleration.max_accel_magnitude / a_mag;
            ax *= scale;
            ay *= scale;
            az *= scale;
        }
    }

    // Complete velocity update: v(t + dt) = v(t + dt/2) + (1/2) * a(t + dt) * dt
    double half_dt = dt * 0.5;
    vel_x[idx] = sanitize_d(vel_x[idx] + ax * half_dt);
    vel_y[idx] = sanitize_d(vel_y[idx] + ay * half_dt);
    vel_z[idx] = sanitize_d(vel_z[idx] + az * half_dt);
}

// =============================================================================
// Quaternion Integration for Rotational Motion
// =============================================================================
// For spherical particles, orientation is tracked via unit quaternion q = (w, x, y, z)
//
// Quaternion rate of change:
//   dq/dt = (1/2) * q ⊗ ω_body
//
// where ω_body is the angular velocity in body frame.
// For spheres with uniform mass distribution, body frame = world frame.
//
// Integration uses quaternion exponential map for numerical stability.
// =============================================================================

// Quaternion multiplication: q1 ⊗ q2
__device__ __forceinline__
void quat_multiply(
    float q1w, float q1x, float q1y, float q1z,
    float q2w, float q2x, float q2y, float q2z,
    float& rw, float& rx, float& ry, float& rz
) {
    rw = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z;
    rx = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y;
    ry = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x;
    rz = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w;
}

// Normalize quaternion to unit length
__device__ __forceinline__
void quat_normalize(float& w, float& x, float& y, float& z) {
    float norm = sqrtf(w*w + x*x + y*y + z*z);
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    } else {
        // Reset to identity if degenerate
        w = 1.0f;
        x = y = z = 0.0f;
    }
}

// Angular velocity half-step update and quaternion integration (Step 1)
// Uses exponential map: q(t+dt) = exp(ω*dt/2) ⊗ q(t)
static __global__ void angular_verlet_step1_kernel(
    // Quaternion (w, x, y, z) - double precision
    double* __restrict__ quat_w,
    double* __restrict__ quat_x,
    double* __restrict__ quat_y,
    double* __restrict__ quat_z,
    // Angular velocity - double precision
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    // Angular acceleration (alpha = torque / I) (double)
    const double* __restrict__ alpha_x,
    const double* __restrict__ alpha_y,
    const double* __restrict__ alpha_z,
    // Moment of inertia
    const float* __restrict__ moment_of_inertia,
    // Torque (if using torque instead of alpha)
    const double* __restrict__ torque_x,
    const double* __restrict__ torque_y,
    const double* __restrict__ torque_z,
    // Use torque or alpha
    bool use_torque,
    // Timestep
    float dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Get angular acceleration
    double ax, ay, az;
    if (use_torque) {
        double inv_I = 1.0 / static_cast<double>(moment_of_inertia[idx]);
        ax = torque_x[idx] * inv_I;
        ay = torque_y[idx] * inv_I;
        az = torque_z[idx] * inv_I;
    } else {
        ax = alpha_x[idx];
        ay = alpha_y[idx];
        az = alpha_z[idx];
    }

    // Half-step angular velocity update
    double half_dt = dt * 0.5;
    double wx = omega_x[idx] + ax * half_dt;
    double wy = omega_y[idx] + ay * half_dt;
    double wz = omega_z[idx] + az * half_dt;

    omega_x[idx] = wx;
    omega_y[idx] = wy;
    omega_z[idx] = wz;

    // Quaternion integration using exponential map
    // exp(ω*dt/2) = cos(|ω|*dt/2) + sin(|ω|*dt/2) * (ω/|ω|)
    double omega_mag = sqrt(wx*wx + wy*wy + wz*wz);

    double exp_w, exp_x, exp_y, exp_z;
    if (omega_mag > 1e-10) {
        double half_angle = omega_mag * half_dt;
        double s = sin(half_angle) / omega_mag;
        exp_w = cos(half_angle);
        exp_x = s * wx;
        exp_y = s * wy;
        exp_z = s * wz;
    } else {
        // First-order approximation for small angles
        exp_w = 1.0;
        exp_x = wx * half_dt;
        exp_y = wy * half_dt;
        exp_z = wz * half_dt;
    }

    // Apply rotation: q_new = exp(ω*dt/2) ⊗ q_old
    double qw = quat_w[idx];
    double qx = quat_x[idx];
    double qy = quat_y[idx];
    double qz = quat_z[idx];

    double new_w = exp_w * qw - exp_x * qx - exp_y * qy - exp_z * qz;
    double new_x = exp_w * qx + exp_x * qw + exp_y * qz - exp_z * qy;
    double new_y = exp_w * qy - exp_x * qz + exp_y * qw + exp_z * qx;
    double new_z = exp_w * qz + exp_x * qy - exp_y * qx + exp_z * qw;

    // Normalize to maintain unit quaternion
    double norm = sqrt(new_w*new_w + new_x*new_x + new_y*new_y + new_z*new_z);
    if (norm > 1e-10) {
        double inv_norm = 1.0 / norm;
        new_w *= inv_norm;
        new_x *= inv_norm;
        new_y *= inv_norm;
        new_z *= inv_norm;
    } else {
        new_w = 1.0;
        new_x = new_y = new_z = 0.0;
    }

    quat_w[idx] = new_w;
    quat_x[idx] = new_x;
    quat_y[idx] = new_y;
    quat_z[idx] = new_z;
}

// Complete angular velocity update after torque calculation (Step 2)
static __global__ void angular_verlet_step2_kernel(
    // Angular velocity - double precision
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    // New angular acceleration (double)
    const double* __restrict__ alpha_x,
    const double* __restrict__ alpha_y,
    const double* __restrict__ alpha_z,
    // Moment of inertia
    const float* __restrict__ moment_of_inertia,
    // New torque
    const double* __restrict__ torque_x,
    const double* __restrict__ torque_y,
    const double* __restrict__ torque_z,
    // Use torque or alpha
    bool use_torque,
    // Timestep
    float dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double ax, ay, az;
    if (use_torque) {
        double inv_I = 1.0 / static_cast<double>(moment_of_inertia[idx]);
        ax = torque_x[idx] * inv_I;
        ay = torque_y[idx] * inv_I;
        az = torque_z[idx] * inv_I;
    } else {
        ax = alpha_x[idx];
        ay = alpha_y[idx];
        az = alpha_z[idx];
    }

    double half_dt = dt * 0.5;
    omega_x[idx] += ax * half_dt;
    omega_y[idx] += ay * half_dt;
    omega_z[idx] += az * half_dt;
}

// =============================================================================
// Combined Single-Pass Kernels (for efficiency)
// =============================================================================
// When force/torque arrays are reused each step, we can combine Step 1 into
// a single kernel that does position update before clearing forces.
// =============================================================================

// Fused Step 1: Position + Quaternion update in single kernel
static __global__ void verlet_step1_fused_kernel(
    // Position (double precision)
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    // Velocity (double precision)
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    // Angular velocity (double precision)
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    // Quaternion (double precision)
    double* __restrict__ quat_w,
    double* __restrict__ quat_x,
    double* __restrict__ quat_y,
    double* __restrict__ quat_z,
    // Force and Torque (input) (double)
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    // Mass and MOI
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    // Timestep
    double dt,
    int32_t num_particles,
    const int32_t* clump_ids = nullptr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || (clump_ids && clump_ids[idx] > 0)) return;

    double m = static_cast<double>(mass[idx]);
    double I = static_cast<double>(moment_of_inertia[idx]);
    double inv_m = 1.0 / m;
    double inv_I = 1.0 / I;

    double half_dt = dt * 0.5;

    // === Translational motion ===
    // Acceleration from force (unbalanced force at this step)
    double ax = force_x[idx] * inv_m;
    double ay = force_y[idx] * inv_m;
    double az = force_z[idx] * inv_m;

    // Cundall Non-viscous Local Damping (per-component)
    // F_damp,i = -alpha * |F_unbalanced,i| * sgn(v_i)
    // Applied as force before integration: adds to acceleration
    if (dev_config.damping.enabled && dev_config.damping.type == DampingType::LOCAL) {
        float alpha = dev_config.damping.coefficient;
        double Fx = force_x[idx];
        double Fy = force_y[idx];
        double Fz = force_z[idx];
        double sgn_vx = (vel_x[idx] > 0.0) ? 1.0 : ((vel_x[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_vy = (vel_y[idx] > 0.0) ? 1.0 : ((vel_y[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_vz = (vel_z[idx] > 0.0) ? 1.0 : ((vel_z[idx] < 0.0) ? -1.0 : 0.0);
        ax += (-alpha * fabs(Fx) * sgn_vx) * inv_m;
        ay += (-alpha * fabs(Fy) * sgn_vy) * inv_m;
        az += (-alpha * fabs(Fz) * sgn_vz) * inv_m;
    }

    // Clamp acceleration magnitude for numerical stability
    if (dev_config.acceleration.enable_accel_limit) {
        double a_mag = sqrt(ax*ax + ay*ay + az*az);
        if (a_mag > dev_config.acceleration.max_accel_magnitude) {
            double scale = dev_config.acceleration.max_accel_magnitude / a_mag;
            ax *= scale;
            ay *= scale;
            az *= scale;
        }
    }

    // Half-step velocity
    double vx = vel_x[idx] + ax * half_dt;
    double vy = vel_y[idx] + ay * half_dt;
    double vz = vel_z[idx] + az * half_dt;

    vel_x[idx] = sanitize_d(vx);
    vel_y[idx] = sanitize_d(vy);
    vel_z[idx] = sanitize_d(vz);

    // Full-step position (double precision accumulation)
    pos_x[idx] = sanitize_d(pos_x[idx] + vx * dt);
    pos_y[idx] = sanitize_d(pos_y[idx] + vy * dt);
    pos_z[idx] = sanitize_d(pos_z[idx] + vz * dt);

    // === Rotational motion ===
    // Angular acceleration from torque
    double alpha_x_val = torque_x[idx] * inv_I;
    double alpha_y_val = torque_y[idx] * inv_I;
    double alpha_z_val = torque_z[idx] * inv_I;

    // Cundall local damping for rotation (per-component)
    // tau_damp,i = -alpha * |tau_unbalanced,i| * sgn(omega_i)
    if (dev_config.damping.enabled && dev_config.damping.type == DampingType::LOCAL) {
        float alpha = dev_config.damping.coefficient;
        double Tx = torque_x[idx];
        double Ty = torque_y[idx];
        double Tz = torque_z[idx];
        double sgn_wx = (omega_x[idx] > 0.0) ? 1.0 : ((omega_x[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_wy = (omega_y[idx] > 0.0) ? 1.0 : ((omega_y[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_wz = (omega_z[idx] > 0.0) ? 1.0 : ((omega_z[idx] < 0.0) ? -1.0 : 0.0);
        alpha_x_val += (-alpha * fabs(Tx) * sgn_wx) * inv_I;
        alpha_y_val += (-alpha * fabs(Ty) * sgn_wy) * inv_I;
        alpha_z_val += (-alpha * fabs(Tz) * sgn_wz) * inv_I;
    }

    // Half-step angular velocity
    double wx = omega_x[idx] + alpha_x_val * half_dt;
    double wy = omega_y[idx] + alpha_y_val * half_dt;
    double wz = omega_z[idx] + alpha_z_val * half_dt;

    omega_x[idx] = wx;
    omega_y[idx] = wy;
    omega_z[idx] = wz;

    // Quaternion update via exponential map
    double omega_mag = sqrt(wx*wx + wy*wy + wz*wz);

    double exp_w, exp_x, exp_y, exp_z;
    if (omega_mag > 1e-10) {
        double half_angle = omega_mag * half_dt;
        double s = sin(half_angle) / omega_mag;
        exp_w = cos(half_angle);
        exp_x = s * wx;
        exp_y = s * wy;
        exp_z = s * wz;
    } else {
        exp_w = 1.0;
        exp_x = wx * half_dt;
        exp_y = wy * half_dt;
        exp_z = wz * half_dt;
    }

    double qw = quat_w[idx];
    double qx_val = quat_x[idx];
    double qy_val = quat_y[idx];
    double qz_val = quat_z[idx];

    double new_w = exp_w * qw - exp_x * qx_val - exp_y * qy_val - exp_z * qz_val;
    double new_x = exp_w * qx_val + exp_x * qw + exp_y * qz_val - exp_z * qy_val;
    double new_y = exp_w * qy_val - exp_x * qz_val + exp_y * qw + exp_z * qx_val;
    double new_z = exp_w * qz_val + exp_x * qy_val - exp_y * qx_val + exp_z * qw;

    // Normalize
    double norm = sqrt(new_w*new_w + new_x*new_x + new_y*new_y + new_z*new_z);
    if (norm > 1e-10) {
        double inv_norm = 1.0 / norm;
        new_w *= inv_norm;
        new_x *= inv_norm;
        new_y *= inv_norm;
        new_z *= inv_norm;
    } else {
        new_w = 1.0;
        new_x = new_y = new_z = 0.0;
    }

    quat_w[idx] = new_w;
    quat_x[idx] = new_x;
    quat_y[idx] = new_y;
    quat_z[idx] = new_z;

    // Reset forces and torques for the next step
    force_x[idx] = 0.0;
    force_y[idx] = 0.0;
    force_z[idx] = 0.0;
    torque_x[idx] = 0.0;
    torque_y[idx] = 0.0;
    torque_z[idx] = 0.0;
}

// Fused Step 2: Complete velocity updates
static __global__ void verlet_step2_fused_kernel(
    // Velocity (double precision)
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    // Angular velocity (double precision)
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    // New force and torque (double)
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    const double* __restrict__ torque_x,
    const double* __restrict__ torque_y,
    const double* __restrict__ torque_z,
    // Mass and MOI
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    // Timestep
    double dt,
    int32_t num_particles,
    const int32_t* clump_ids = nullptr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || (clump_ids && clump_ids[idx] > 0)) return;

    double inv_m = 1.0 / static_cast<double>(mass[idx]);
    double inv_I = 1.0 / static_cast<double>(moment_of_inertia[idx]);
    double half_dt = dt * 0.5;

    // Complete translational velocity
    // Acceleration from force
    double ax = force_x[idx] * inv_m;
    double ay = force_y[idx] * inv_m;
    double az = force_z[idx] * inv_m;

    // Cundall local damping: F_damp,i = -alpha * |F_i| * sgn(v_half,i)
    if (dev_config.damping.enabled && dev_config.damping.type == DampingType::LOCAL) {
        float alpha = dev_config.damping.coefficient;
        double Fx = force_x[idx];
        double Fy = force_y[idx];
        double Fz = force_z[idx];
        double sgn_vx = (vel_x[idx] > 0.0) ? 1.0 : ((vel_x[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_vy = (vel_y[idx] > 0.0) ? 1.0 : ((vel_y[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_vz = (vel_z[idx] > 0.0) ? 1.0 : ((vel_z[idx] < 0.0) ? -1.0 : 0.0);
        ax += (-alpha * fabs(Fx) * sgn_vx) * inv_m;
        ay += (-alpha * fabs(Fy) * sgn_vy) * inv_m;
        az += (-alpha * fabs(Fz) * sgn_vz) * inv_m;
    }

    // Clamp acceleration magnitude for numerical stability
    if (dev_config.acceleration.enable_accel_limit) {
        double a_mag = sqrt(ax*ax + ay*ay + az*az);
        if (a_mag > dev_config.acceleration.max_accel_magnitude) {
            double scale = dev_config.acceleration.max_accel_magnitude / a_mag;
            ax *= scale;
            ay *= scale;
            az *= scale;
        }
    }

    vel_x[idx] = sanitize_d(vel_x[idx] + ax * half_dt);
    vel_y[idx] = sanitize_d(vel_y[idx] + ay * half_dt);
    vel_z[idx] = sanitize_d(vel_z[idx] + az * half_dt);

    // Complete angular velocity
    double dw_x = torque_x[idx] * inv_I * half_dt;
    double dw_y = torque_y[idx] * inv_I * half_dt;
    double dw_z = torque_z[idx] * inv_I * half_dt;

    // Cundall local damping for rotation
    if (dev_config.damping.enabled && dev_config.damping.type == DampingType::LOCAL) {
        float alpha = dev_config.damping.coefficient;
        double Tx = torque_x[idx];
        double Ty = torque_y[idx];
        double Tz = torque_z[idx];
        double sgn_wx = (omega_x[idx] > 0.0) ? 1.0 : ((omega_x[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_wy = (omega_y[idx] > 0.0) ? 1.0 : ((omega_y[idx] < 0.0) ? -1.0 : 0.0);
        double sgn_wz = (omega_z[idx] > 0.0) ? 1.0 : ((omega_z[idx] < 0.0) ? -1.0 : 0.0);
        dw_x += (-alpha * fabs(Tx) * sgn_wx) * inv_I * half_dt;
        dw_y += (-alpha * fabs(Ty) * sgn_wy) * inv_I * half_dt;
        dw_z += (-alpha * fabs(Tz) * sgn_wz) * inv_I * half_dt;
    }

    omega_x[idx] += dw_x;
    omega_y[idx] += dw_y;
    omega_z[idx] += dw_z;
}

// =============================================================================
// Viscous Velocity Damping (DampingType::VISCOUS)
// =============================================================================
// Apply global viscous velocity damping: v_new = v_old * (1 - gamma * dt)
// Useful for reaching equilibrium states faster.
// NOTE: Cundall local damping (DampingType::LOCAL) is applied inside the fused
// verlet kernels (verlet_step1_fused_kernel / verlet_step2_fused_kernel) and
// does NOT use this kernel.
// =============================================================================

static __global__ void apply_velocity_damping_kernel(
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    double* __restrict__ omega_x,
    double* __restrict__ omega_y,
    double* __restrict__ omega_z,
    float damping_factor,  // (1 - gamma * dt)
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    vel_x[idx] *= damping_factor;
    vel_y[idx] *= damping_factor;
    vel_z[idx] *= damping_factor;

    omega_x[idx] *= damping_factor;
    omega_y[idx] *= damping_factor;
    omega_z[idx] *= damping_factor;
}

// =============================================================================
// Energy Calculation
// =============================================================================
// Compute kinetic energy for monitoring simulation health
// =============================================================================

// Single-particle kinetic energy (translational + rotational)
__device__ __forceinline__
float particle_kinetic_energy(
    float vx, float vy, float vz,
    float wx, float wy, float wz,
    float mass, float moment_of_inertia
) {
    // Translational: (1/2) * m * v²
    float v_sq = vx*vx + vy*vy + vz*vz;
    float ke_trans = 0.5f * mass * v_sq;

    // Rotational: (1/2) * I * ω²
    float omega_sq = wx*wx + wy*wy + wz*wz;
    float ke_rot = 0.5f * moment_of_inertia * omega_sq;

    return ke_trans + ke_rot;
}

// Kernel to compute per-particle kinetic energy (for reduction)
static __global__ void compute_kinetic_energy_kernel(
    float* __restrict__ ke_per_particle,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    ke_per_particle[idx] = particle_kinetic_energy(
        static_cast<float>(vel_x[idx]), static_cast<float>(vel_y[idx]), static_cast<float>(vel_z[idx]),
        static_cast<float>(omega_x[idx]), static_cast<float>(omega_y[idx]), static_cast<float>(omega_z[idx]),
        mass[idx], moment_of_inertia[idx]
    );
}

// =============================================================================
// Force/Torque Reset Kernel
// =============================================================================
// Clear force and torque arrays before accumulation
// =============================================================================

static __global__ void reset_forces_kernel(
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    force_x[idx] = 0.0;
    force_y[idx] = 0.0;
    force_z[idx] = 0.0;
    torque_x[idx] = 0.0;
    torque_y[idx] = 0.0;
    torque_z[idx] = 0.0;
}

// =============================================================================
// OPTIMIZED Force Reset using cudaMemsetAsync
// =============================================================================
// Performance Note: The reset_forces_kernel can become slow in later simulation
// phases due to:
//   1. TLB thrashing from previous kernel access patterns
//   2. Kernel launch overhead (6 separate memory writes per thread)
//   3. Branching at boundary threads
//
// cudaMemsetAsync advantages:
//   - Uses the GPU's DMA engine (separate from compute)
//   - Guaranteed coalesced 128-byte writes
//   - Zero branching
//   - Lower launch overhead (uses copy engine, not SM)
//   - Can overlap with compute on different streams
//
// IMPORTANT: This only works because 0.0 in IEEE 754 double is all-zero bits.
// =============================================================================

inline void reset_forces_memset_async(
    double* force_x,
    double* force_y,
    double* force_z,
    double* torque_x,
    double* torque_y,
    double* torque_z,
    int32_t num_particles,
    cudaStream_t stream = 0
) {
    size_t bytes = num_particles * sizeof(double);
    
    // Zero all force/torque arrays using DMA engine
    // Note: cudaMemsetAsync sets bytes, and 0.0 in double is 8 bytes of 0x00
    cudaMemsetAsync(force_x,  0, bytes, stream);
    cudaMemsetAsync(force_y,  0, bytes, stream);
    cudaMemsetAsync(force_z,  0, bytes, stream);
    cudaMemsetAsync(torque_x, 0, bytes, stream);
    cudaMemsetAsync(torque_y, 0, bytes, stream);
    cudaMemsetAsync(torque_z, 0, bytes, stream);
}

// =============================================================================
// Velocity Limiting (for stability)
// =============================================================================
// Clamp velocities to prevent numerical instabilities
// =============================================================================

static __global__ void limit_velocities_kernel(
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    float* __restrict__ omega_x,
    float* __restrict__ omega_y,
    float* __restrict__ omega_z,
    float max_velocity,
    float max_omega,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Limit linear velocity
    float v_mag = sqrtf(vel_x[idx]*vel_x[idx] +
                        vel_y[idx]*vel_y[idx] +
                        vel_z[idx]*vel_z[idx]);
    if (v_mag > max_velocity) {
        float scale = max_velocity / v_mag;
        vel_x[idx] *= scale;
        vel_y[idx] *= scale;
        vel_z[idx] *= scale;
    }

    // Limit angular velocity
    float w_mag = sqrtf(omega_x[idx]*omega_x[idx] +
                        omega_y[idx]*omega_y[idx] +
                        omega_z[idx]*omega_z[idx]);
    if (w_mag > max_omega) {
        float scale = max_omega / w_mag;
        omega_x[idx] *= scale;
        omega_y[idx] *= scale;
        omega_z[idx] *= scale;
    }
}

// =============================================================================
// Integrator Class (Host-side orchestration)
// =============================================================================

class VerletIntegrator {
public:
    double dt;                  // Current timestep
    double dt_critical;         // Critical timestep
    int32_t block_size;         // CUDA block size
    bool use_fused_kernels;     // Use fused kernels for efficiency

    // Velocity limits for stability
    float max_velocity;
    float max_omega;
    bool limit_velocities;

    // NOTE: Damping is now config-driven via dev_config.damping (SimConfig).
    // Cundall local damping (DampingType::LOCAL) is applied inside the fused
    // verlet kernels. Viscous damping (DampingType::VISCOUS) is applied as a
    // separate post-step kernel in main.cu.

    VerletIntegrator()
        : dt(1e-5), dt_critical(1e-4), block_size(256),
          use_fused_kernels(true),
          max_velocity(100.0f), max_omega(1000.0f), limit_velocities(false) {}

    // Calculate and set timestep from particle/material properties
    void set_timestep(const TimestepParams& params) {
        dt_critical = calculate_timestep(params);
        dt = dt_critical;

        std::cout << "Verlet Integrator: dt = " << dt << " s "
                  << "(critical: " << dt_critical << " s, safety: "
                  << params.safety_factor * 100 << "%)" << std::endl;
    }

    // Set explicit timestep (with safety check)
    void set_timestep(double new_dt, bool force = false) {
        if (!force && new_dt > dt_critical) {
            std::cerr << "WARNING: Requested timestep " << new_dt
                      << " exceeds critical timestep " << dt_critical << std::endl;
        }
        dt = new_dt;
    }

    // Enable velocity limiting
    void enable_velocity_limit(float max_v, float max_w) {
        max_velocity = max_v;
        max_omega = max_w;
        limit_velocities = true;
    }

    void disable_velocity_limit() {
        limit_velocities = false;
    }
};

// =============================================================================
// Utility: Quaternion-to-Rotation Matrix (for visualization/output)
// =============================================================================

__device__ __forceinline__
void quat_to_rotation_matrix(
    float qw, float qx, float qy, float qz,
    float& r00, float& r01, float& r02,
    float& r10, float& r11, float& r12,
    float& r20, float& r21, float& r22
) {
    float xx = qx * qx;
    float yy = qy * qy;
    float zz = qz * qz;
    float xy = qx * qy;
    float xz = qx * qz;
    float yz = qy * qz;
    float wx = qw * qx;
    float wy = qw * qy;
    float wz = qw * qz;

    r00 = 1.0f - 2.0f * (yy + zz);
    r01 = 2.0f * (xy - wz);
    r02 = 2.0f * (xz + wy);

    r10 = 2.0f * (xy + wz);
    r11 = 1.0f - 2.0f * (xx + zz);
    r12 = 2.0f * (yz - wx);

    r20 = 2.0f * (xz - wy);
    r21 = 2.0f * (yz + wx);
    r22 = 1.0f - 2.0f * (xx + yy);
}

} // namespace cudadem

#endif // CUDADEM_VERLET_INTEGRATION_CUH

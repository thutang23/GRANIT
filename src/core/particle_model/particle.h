#ifndef CUDADEM_PARTICLE_H
#define CUDADEM_PARTICLE_H

// =============================================================================
// Particle Contact Physics Interface
// =============================================================================
// DESIGN: Template Policy Composition for compile-time physics selection
//
// Reference implementations:
// - Hertz-Mindlin: Wada et al., DEM simulations for asteroids
// - Rolling friction: LIGGGHTS (LAMMPS granular)
// - Cohesion: Scheeres et al., asteroid cohesive strength
// - Damping: applicable for ice ball
//
// Key formulas:
// - Normal: F_n = K_n * δ_n * n̂ + C_n * V_n
// - Tangential: F_t = -K_s * δ_s + C_s * V_t + H (history)
// - Rolling: M_r = K_r * Δθ_r + M_r_history
// - Twisting: M_t = K_t * Δθ_t + M_t_history
// - Cohesion: F_c = -γ * (β*R_eff)² * n̂
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#endif
#include <cmath>

namespace cudadem {

// =============================================================================
// Contact Force Result Structure
// =============================================================================
struct ContactForceResult {
    // Forces (on particle i)
    double force_x, force_y, force_z;

    // Torque on particle i (owner)
    double torque_x, torque_y, torque_z;

    // Torque on particle j (partner) — computed once, applied via atomicAdd
    // Force-arm part uses An (not Ap), pure-couple part (Mr+Mt) is negated
    double torque_j_x, torque_j_y, torque_j_z;

    // Contact history updates (for next timestep)
    float delta_t_x, delta_t_y, delta_t_z;  // Tangential spring displacement
    float delta_r_x, delta_r_y, delta_r_z;  // Rolling displacement
    float delta_tw_x, delta_tw_y, delta_tw_z;  // Twisting displacement (3-component)

    // Contact state flags
    bool is_contact;
    bool is_slipping;
    bool is_rolling;
    bool is_twisting;

    // Energy dissipation (for monitoring)
    float energy_elastic;
    float energy_dissipated;

    __device__ ContactForceResult()
        : force_x(0), force_y(0), force_z(0),
          torque_x(0), torque_y(0), torque_z(0),
          torque_j_x(0), torque_j_y(0), torque_j_z(0),
          delta_t_x(0), delta_t_y(0), delta_t_z(0),
          delta_r_x(0), delta_r_y(0), delta_r_z(0),
          delta_tw_x(0), delta_tw_y(0), delta_tw_z(0),
          is_contact(false), is_slipping(false), is_rolling(false), is_twisting(false),
          energy_elastic(0), energy_dissipated(0) {}
};

// =============================================================================
// Material Parameters (computed from SimConfig)
// =============================================================================
struct ContactMaterialParams {
    // Effective properties
    float R_eff;      // Effective radius: R_i * R_j / (R_i + R_j)
    float M_eff;      // Effective mass: m_i * m_j / (m_i + m_j)
    float I_eff;      // Effective moment of inertia

    // Spring constants (updated each contact based on overlap)
    float K_n;        // Normal spring
    float K_s;        // Tangential spring
    float K_r;        // Rolling spring
    float K_t;        // Twisting spring

    // Damping coefficients
    float C_n;        // Normal damping
    float C_s;        // Tangential damping
    float C_r;        // Rolling damping
    float C_t;        // Twisting damping

    // Friction limits
    float mu_s;       // Sliding friction coefficient
    float mu_d;       // Dynamic friction coefficient (currently = mu_s)
    float mu_r;       // Rolling friction coefficient
    float mu_t;       // Twisting friction coefficient
    float beta;       // Shape parameter

    // Cohesion
    float gamma;      // Cohesive strength
    float r_cut;      // Cutoff range for cohesion

    __device__ ContactMaterialParams()
        : R_eff(0), M_eff(0), I_eff(0),
          K_n(0), K_s(0), K_r(0), K_t(0),
          C_n(0), C_s(0), C_r(0), C_t(0),
          mu_s(1.0f), mu_d(1.0f), mu_r(0.0f), mu_t(0.0f), beta(0.7f),
          gamma(0), r_cut(1.0f) {}
};

// =============================================================================
// Hertz-Mindlin Material Constants Calculation
// =============================================================================
// Two modes:
// 1. HertzMindlinResti: Damping from coefficient of restitution
// 2. HertzMindlinVisco: Damping from viscous parameter (A)
// =============================================================================

// Calculate Hertz-Mindlin material constants with restitution-based damping
__device__ __forceinline__
ContactMaterialParams compute_hertz_mindlin_resti(
    float R_i, float R_j,           // Radii
    float m_i, float m_j,           // Masses
    float I_i, float I_j,           // Moments of inertia
    float delta_n,                   // Normal overlap (positive when in contact)
    float E_star,                    // Pair effective Young's modulus
    float G_star,                    // Pair effective shear modulus
    float COR,                       // Coefficient of restitution
    float mu_s,                      // Static friction
    float mu_r,                      // Rolling friction
    float mu_t,                      // Twisting friction
    float beta                       // Shape parameter
) {
    ContactMaterialParams params;

    // Effective properties
    params.R_eff = R_i * R_j / (R_i + R_j);
    params.M_eff = m_i * m_j / (m_i + m_j);
    params.I_eff = I_i * I_j / (I_i + I_j);

    // Friction
    params.mu_s = mu_s;
    params.mu_d = mu_s;  // Assume dynamic = static for simplicity
    params.mu_r = mu_r;
    params.mu_t = mu_t;
    params.beta = beta;

    // Normal spring constant (Hertz)
    // K_n = 4/3 E* sqrt(R_eff*delta_n).  For identical materials this
    // is exactly the legacy 2E/[3(1-nu²)] expression.
    float sqrt_Rd = sqrtf(params.R_eff * delta_n);
    params.K_n = (4.0f / 3.0f) * E_star * sqrt_Rd;

    // Normal damping (from COR)
    // C_n = 2*sqrt(5/6) * ln(COR)/sqrt(ln²(COR)+π²) * sqrt(M_eff*K_n)
    if (COR > 0.0f && COR < 1.0f) {
        float ln_COR = logf(COR);
        float coeff = 2.0f * sqrtf(5.0f / 6.0f) * ln_COR /
                      sqrtf(ln_COR * ln_COR + 9.8696044f);  // π² ≈ 9.8696
        float sqrt_delta = sqrtf(sqrtf(delta_n));  // delta^0.25
        float sqrt_Reff = sqrtf(sqrtf(params.R_eff));  // R^0.25
        params.C_n = coeff * sqrtf(2.0f * params.M_eff * E_star) *
                     sqrt_Reff * sqrt_delta;
    } else {
        params.C_n = 0.0f;
    }

    // Tangential spring constant (Mindlin)
    // K_s = 8 G* sqrt(R_eff*delta_n).
    params.K_s = 8.0f * G_star * sqrtf(params.R_eff) * sqrtf(delta_n);

    // Tangential damping
    if (COR > 0.0f && COR < 1.0f) {
        float ln_COR = logf(COR);
        float coeff = 2.0f * sqrtf(5.0f / 6.0f) * ln_COR /
                      sqrtf(ln_COR * ln_COR + 9.8696044f);
        float sqrt_delta = sqrtf(sqrtf(delta_n));
        float sqrt_Reff = sqrtf(sqrtf(params.R_eff));
        params.C_s = coeff * sqrtf(8.0f * params.M_eff * G_star) *
                     sqrt_Reff * sqrt_delta;
    } else {
        params.C_s = 0.0f;
    }

    // Rolling spring and damping (LIGGGHTS style)
    // K_r = 0.25 * K_n * (beta*R_eff)²
    float beta_R_sq = (beta * params.R_eff) * (beta * params.R_eff);
    params.K_r = 0.25f * params.K_n * beta_R_sq;
    params.C_r = 0.25f * params.C_n * beta_R_sq;

    // Twisting spring and damping
    // K_t = 0.5 * K_s * (beta*R_eff)²
    params.K_t = 0.5f * params.K_s * beta_R_sq;
    params.C_t = 0.5f * params.C_s * beta_R_sq;

    return params;
}

// Calculate Hertz-Mindlin material constants with viscous damping (for ice ball)
__device__ __forceinline__
ContactMaterialParams compute_hertz_mindlin_visco(
    float R_i, float R_j,
    float m_i, float m_j,
    float I_i, float I_j,
    float delta_n,
    float E_star,
    float G_star,
    float A,                         // Viscous damping parameter
    float mu_s,
    float mu_r,
    float mu_t,
    float beta
) {
    ContactMaterialParams params;

    // Effective properties
    params.R_eff = R_i * R_j / (R_i + R_j);
    params.M_eff = m_i * m_j / (m_i + m_j);
    params.I_eff = I_i * I_j / (I_i + I_j);

    params.mu_s = mu_s;
    params.mu_d = mu_s;
    params.mu_r = mu_r;
    params.mu_t = mu_t;
    params.beta = beta;

    // Spring constants (same as restitution mode)
    float sqrt_Rd = sqrtf(params.R_eff * delta_n);
    params.K_n = (4.0f / 3.0f) * E_star * sqrt_Rd;
    params.K_s = 8.0f * G_star * sqrtf(params.R_eff) * sqrtf(delta_n);

    // Viscous damping: C = -K * A
    params.C_n = -params.K_n * A;
    params.C_s = -8.0f * G_star * sqrtf(delta_n) * A;

    // Rolling and twisting
    float beta_R_sq = (beta * params.R_eff) * (beta * params.R_eff);
    params.K_r = 0.25f * params.K_n * beta_R_sq;
    params.C_r = 0.25f * params.C_n * beta_R_sq;
    params.K_t = 0.5f * params.K_s * beta_R_sq;
    params.C_t = 0.5f * params.C_s * beta_R_sq;

    return params;
}

// =============================================================================
// Normal Force Calculation (Hertz + Damping)
// =============================================================================
// F_n = K_n * δ_n * n̂ + C_n * V_n
// Force direction: positive = repulsive (pushes particles apart)
// =============================================================================

__device__ __forceinline__
void compute_normal_force(
    // Outputs
    float& Fn_x, float& Fn_y, float& Fn_z,
    float& Fn_magnitude,
    // Inputs
    float delta_n,                   // Overlap (positive)
    float nx, float ny, float nz,    // Unit normal (from i to j)
    double Vn_x, double Vn_y, double Vn_z,  // Normal velocity component
    const ContactMaterialParams& params
) {
    // Elastic force: F_elastic = K_n * delta_n * n
    float F_elastic = params.K_n * delta_n;

    // Damping force: F_damp = C_n * Vn_signed
    // C_n < 0 (from COR formula); Vn_signed > 0 when separating (j moves away from i).
    // This correctly increases repulsion during approach and decreases during separation.
    float Vn_signed = static_cast<float>(Vn_x * nx + Vn_y * ny + Vn_z * nz);
    float F_damp = params.C_n * Vn_signed;

    // Total normal force (signed: positive=repulsive, negative=attractive during separation)
    // Damping (C_n < 0) can make this negative during strong separation — physically correct
    // for the force vector, but the magnitude for friction limits must be non-negative.
    float Fn_signed = F_elastic + F_damp;

    // Force on particle i (points away from j when repulsive)
    Fn_x = -Fn_signed * nx;
    Fn_y = -Fn_signed * ny;
    Fn_z = -Fn_signed * nz;

    // Magnitude for Coulomb/rolling/twisting limits: always non-negative (matches DEMBody vector magnitude)
    Fn_magnitude = fabsf(Fn_signed);
}

// =============================================================================
// Tangential Force Calculation (Mindlin + History)
// =============================================================================

// Rotate a stored vector history into the current contact subspace.  The
// renormalization preserves the spring state under a changing contact normal;
// simply projecting it would spuriously dissipate stored elastic energy.
__device__ __forceinline__
void rotate_history_to_tangent(float& x, float& y, float& z,
                               float nx, float ny, float nz) {
    const float old_mag = sqrtf(x * x + y * y + z * z);
    const float normal = x * nx + y * ny + z * nz;
    x -= normal * nx;
    y -= normal * ny;
    z -= normal * nz;
    const float projected_mag = sqrtf(x * x + y * y + z * z);
    if (old_mag > 1e-20f && projected_mag > 1e-20f) {
        const float scale = old_mag / projected_mag;
        x *= scale; y *= scale; z *= scale;
    } else {
        x = 0.0f; y = 0.0f; z = 0.0f;
    }
}

__device__ __forceinline__
void rotate_history_to_normal(float& x, float& y, float& z,
                              float nx, float ny, float nz) {
    const float old_mag = sqrtf(x * x + y * y + z * z);
    const float signed_projection = x * nx + y * ny + z * nz;
    if (old_mag > 1e-20f && fabsf(signed_projection) > 1e-20f) {
        const float signed_mag = copysignf(old_mag, signed_projection);
        x = signed_mag * nx; y = signed_mag * ny; z = signed_mag * nz;
    } else {
        x = 0.0f; y = 0.0f; z = 0.0f;
    }
}
// F_t = -K_s * δ_s + C_s * V_t + H (accumulated history)
// With Coulomb slip limit: |F_t| ≤ μ * |F_n|
// =============================================================================

__device__ __forceinline__
void compute_tangential_force(
    // Outputs
    float& Ft_x, float& Ft_y, float& Ft_z,
    float& Ft_magnitude,
    float& H_new_x, float& H_new_y, float& H_new_z,  // Updated history
    bool& is_slipping,
    // Inputs
    double Vt_x, double Vt_y, double Vt_z,  // Tangential velocity
    float H_x, float H_y, float H_z,      // History from previous step
    float dt,
    float Fn_magnitude,
    const ContactMaterialParams& params
) {
    // Tangential displacement increment
    float ds_x = Vt_x * dt;
    float ds_y = Vt_y * dt;
    float ds_z = Vt_z * dt;

    // Tangential force from spring + damping + history
    // F_t = -K_s * ds + C_s * Vt + H
    Ft_x = -params.K_s * ds_x + params.C_s * Vt_x + H_x;
    Ft_y = -params.K_s * ds_y + params.C_s * Vt_y + H_y;
    Ft_z = -params.K_s * ds_z + params.C_s * Vt_z + H_z;

    Ft_magnitude = sqrtf(Ft_x * Ft_x + Ft_y * Ft_y + Ft_z * Ft_z);

    // Coulomb friction limit
    float Ft_max = params.mu_s * Fn_magnitude;

    if (Ft_magnitude > Ft_max && Ft_magnitude > 1e-10f) {
        // Sliding: limit force to kinetic friction
        float scale = params.mu_d * Fn_magnitude / Ft_magnitude;
        Ft_x *= scale;
        Ft_y *= scale;
        Ft_z *= scale;
        Ft_magnitude = params.mu_d * Fn_magnitude;

        // Store only the clipped elastic spring force.  Keeping the damping
        // contribution in history double-counts it on the following step.
        H_new_x = Ft_x - params.C_s * Vt_x;
        H_new_y = Ft_y - params.C_s * Vt_y;
        H_new_z = Ft_z - params.C_s * Vt_z;

        is_slipping = true;
    } else {
        // Sticking: accumulate history (without damping)
        H_new_x = Ft_x - params.C_s * Vt_x;
        H_new_y = Ft_y - params.C_s * Vt_y;
        H_new_z = Ft_z - params.C_s * Vt_z;

        is_slipping = false;
    }
}

// =============================================================================
// Rolling Moment Calculation (LIGGGHTS style)
// =============================================================================
// M_r = K_r * Δθ_r + M_history
// With limit: |M_r| ≤ 2.1 * 0.25 * β * R_eff * |F_n|
// =============================================================================

__device__ __forceinline__
void compute_rolling_moment(
    // Outputs
    float& Mr_x, float& Mr_y, float& Mr_z,
    float& Mr_magnitude,
    float& Mr_new_x, float& Mr_new_y, float& Mr_new_z,  // Updated history
    bool& is_rolling,
    // Inputs
    double omega_i_x, double omega_i_y, double omega_i_z,  // Angular velocity particle i
    double omega_j_x, double omega_j_y, double omega_j_z,  // Angular velocity particle j
    float nx, float ny, float nz,                        // Contact normal
    float Mr_hist_x, float Mr_hist_y, float Mr_hist_z,  // Previous history
    float dt,
    float Fn_magnitude,
    const ContactMaterialParams& params
) {
    // Relative angular velocity
    float dtheta_x = (omega_i_x - omega_j_x) * dt;
    float dtheta_y = (omega_i_y - omega_j_y) * dt;
    float dtheta_z = (omega_i_z - omega_j_z) * dt;

    // Project onto normal (twisting component)
    float dtheta_n = dtheta_x * nx + dtheta_y * ny + dtheta_z * nz;
    float dtheta_t_x = dtheta_n * nx;
    float dtheta_t_y = dtheta_n * ny;
    float dtheta_t_z = dtheta_n * nz;

    // Rolling component (perpendicular to normal)
    float dtheta_r_x = dtheta_x - dtheta_t_x;
    float dtheta_r_y = dtheta_y - dtheta_t_y;
    float dtheta_r_z = dtheta_z - dtheta_t_z;

    rotate_history_to_tangent(Mr_hist_x, Mr_hist_y, Mr_hist_z, nx, ny, nz);

    // Rolling moment
    Mr_x = params.K_r * dtheta_r_x + Mr_hist_x;
    Mr_y = params.K_r * dtheta_r_y + Mr_hist_y;
    Mr_z = params.K_r * dtheta_r_z + Mr_hist_z;

    Mr_magnitude = sqrtf(Mr_x * Mr_x + Mr_y * Mr_y + Mr_z * Mr_z);

    // Rolling resistance limit (LIGGGHTS: 2.1 * 0.25 * beta * R_eff * |Fn|)
    float Mr_max = 2.1f * 0.25f * params.mu_r * params.beta * params.R_eff * Fn_magnitude;

    if (Mr_magnitude > Mr_max && Mr_magnitude > 1e-10f) {
        // Rolling: limit moment
        float scale = Mr_max / Mr_magnitude;
        Mr_x *= scale;
        Mr_y *= scale;
        Mr_z *= scale;
        Mr_magnitude = Mr_max;

        Mr_new_x = Mr_x;
        Mr_new_y = Mr_y;
        Mr_new_z = Mr_z;

        is_rolling = true;
    } else {
        // Sticking
        Mr_new_x = Mr_x;
        Mr_new_y = Mr_y;
        Mr_new_z = Mr_z;

        // Apply damping
        float dtheta_r_mag = sqrtf(dtheta_r_x * dtheta_r_x +
                                    dtheta_r_y * dtheta_r_y +
                                    dtheta_r_z * dtheta_r_z);
        if (dtheta_r_mag > 1e-10f) {
            Mr_x -= params.C_r * dtheta_r_x / dt;
            Mr_y -= params.C_r * dtheta_r_y / dt;
            Mr_z -= params.C_r * dtheta_r_z / dt;
        }

        is_rolling = false;
    }
}

// =============================================================================
// Twisting Moment Calculation
// =============================================================================
// M_t = K_t * Δθ_t + M_history
// With limit: |M_t| ≤ 0.65 * μ * β * R_eff * |F_n|
// =============================================================================

__device__ __forceinline__
void compute_twisting_moment(
    // Outputs
    float& Mt_x, float& Mt_y, float& Mt_z,
    float& Mt_magnitude,
    float& Mt_new_x, float& Mt_new_y, float& Mt_new_z,
    bool& is_twisting,
    // Inputs
    double omega_i_x, double omega_i_y, double omega_i_z,
    double omega_j_x, double omega_j_y, double omega_j_z,
    float nx, float ny, float nz,
    float Mt_hist_x, float Mt_hist_y, float Mt_hist_z,
    float dt,
    float Fn_magnitude,
    const ContactMaterialParams& params
) {
    // Relative angular velocity
    float dtheta_x = (omega_i_x - omega_j_x) * dt;
    float dtheta_y = (omega_i_y - omega_j_y) * dt;
    float dtheta_z = (omega_i_z - omega_j_z) * dt;

    // Twisting component (along normal)
    float dtheta_n = dtheta_x * nx + dtheta_y * ny + dtheta_z * nz;
    float dtheta_t_x = dtheta_n * nx;
    float dtheta_t_y = dtheta_n * ny;
    float dtheta_t_z = dtheta_n * nz;

    rotate_history_to_normal(Mt_hist_x, Mt_hist_y, Mt_hist_z, nx, ny, nz);

    // Twisting moment
    Mt_x = params.K_t * dtheta_t_x + Mt_hist_x;
    Mt_y = params.K_t * dtheta_t_y + Mt_hist_y;
    Mt_z = params.K_t * dtheta_t_z + Mt_hist_z;

    Mt_magnitude = sqrtf(Mt_x * Mt_x + Mt_y * Mt_y + Mt_z * Mt_z);

    // Twisting resistance limit
    float Mt_max = 0.65f * params.mu_t * params.beta * params.R_eff * Fn_magnitude;

    if (Mt_magnitude > Mt_max && Mt_magnitude > 1e-10f) {
        float scale = Mt_max / Mt_magnitude;
        Mt_x *= scale;
        Mt_y *= scale;
        Mt_z *= scale;
        Mt_magnitude = Mt_max;

        Mt_new_x = Mt_x;
        Mt_new_y = Mt_y;
        Mt_new_z = Mt_z;

        is_twisting = true;
    } else {
        Mt_new_x = Mt_x;
        Mt_new_y = Mt_y;
        Mt_new_z = Mt_z;

        // Apply damping
        float dtheta_t_mag = sqrtf(dtheta_t_x * dtheta_t_x +
                                    dtheta_t_y * dtheta_t_y +
                                    dtheta_t_z * dtheta_t_z);
        if (dtheta_t_mag > 1e-10f) {
            Mt_x -= params.C_t * dtheta_t_x / dt;
            Mt_y -= params.C_t * dtheta_t_y / dt;
            Mt_z -= params.C_t * dtheta_t_z / dt;
        }

        is_twisting = false;
    }
}

// =============================================================================
// Cohesion Force (Scheeres Model)
// =============================================================================
// F_c = -γ * (β*R_eff)² * n̂
//
// Even if not in contact, cohesion applies within cutoff range:
// Region 1: delta_n > -R_eff*(r_cut-1)/2  -> full cohesion
// Region 2: delta_n > -R_eff*(r_cut-1)    -> linear ramp
// Beyond Region 2: no cohesion
// =============================================================================

__device__ __forceinline__
void compute_cohesion_force(
    // Outputs
    float& Fc_x, float& Fc_y, float& Fc_z,
    // Inputs
    float delta_n,                   // Overlap (can be negative)
    float nx, float ny, float nz,    // Unit normal (from i to j)
    float gamma,                     // Pair-specific cohesive strength
    float R_eff,
    float beta,
    float r_cut                      // Cutoff range parameter
) {
    if (gamma <= 1e-10f) {
        Fc_x = 0.0f;
        Fc_y = 0.0f;
        Fc_z = 0.0f;
        return;
    }

    // Paper Eq. 1: F_C = c * A_eff = c * (beta*r_eff)^2
    float beta_R_sq = (beta * R_eff) * (beta * R_eff);
    float F_cohesion;

    // In contact: full cohesion
    if (delta_n >= 0.0f) {
        F_cohesion = -gamma * beta_R_sq;
    }
    // r_cut == 1 is the valid contact-only limit: touching/overlapping
    // particles receive full cohesion, while any positive gap receives none.
    // Keep this branch after the contact case to avoid both disabling contact
    // cohesion and dividing by a zero-width cohesive range below.
    else if (r_cut <= 1.0f) {
        Fc_x = 0.0f;
        Fc_y = 0.0f;
        Fc_z = 0.0f;
        return;
    }
    // Region 1: close but not touching
    else if (delta_n > -R_eff * (r_cut - 1.0f) * 0.5f) {
        F_cohesion = -gamma * beta_R_sq;
    }
    // Region 2: linear ramp down
    else if (delta_n > -R_eff * (r_cut - 1.0f)) {
        // Linear interpolation from full to zero
        float factor = 2.0f * (delta_n / (r_cut - 1.0f) + R_eff);
        F_cohesion = -gamma * (beta * beta) * R_eff * factor;
    }
    // Beyond cutoff: no cohesion
    else {
        Fc_x = 0.0f;
        Fc_y = 0.0f;
        Fc_z = 0.0f;
        return;
    }

    // Cohesion force (attractive, points toward j)
    Fc_x = F_cohesion * nx;
    Fc_y = F_cohesion * ny;
    Fc_z = F_cohesion * nz;
}

// =============================================================================
// Complete Contact Force Calculation
// =============================================================================
// Computes all forces and torques for a single particle pair
// =============================================================================

__device__ __forceinline__
ContactForceResult compute_contact_forces(
    // Particle i properties
    float pos_i_x, float pos_i_y, float pos_i_z,
    double vel_i_x, double vel_i_y, double vel_i_z,
    double omega_i_x, double omega_i_y, double omega_i_z,
    float R_i, float m_i, float I_i,

    // Particle j properties
    float pos_j_x, float pos_j_y, float pos_j_z,
    double vel_j_x, double vel_j_y, double vel_j_z,
    double omega_j_x, double omega_j_y, double omega_j_z,
    float R_j, float m_j, float I_j,

    // Contact history
    float H_t_x, float H_t_y, float H_t_z,    // Tangential
    float H_r_x, float H_r_y, float H_r_z,    // Rolling
    float H_tw_x, float H_tw_y, float H_tw_z, // Twisting

    // Material parameters
    float E_star, float G_star, float COR,
    float mu_s, float mu_r, float mu_t,
    float beta, float r_cut,
    bool use_visco, float visco_A,
    bool cohesion_enabled, float cohesion_strength,

    // Timestep
    float dt
) {
    ContactForceResult result;

    // Distance vector (from i to j)
    float dist_x = pos_j_x - pos_i_x;
    float dist_y = pos_j_y - pos_i_y;
    float dist_z = pos_j_z - pos_i_z;
    float dist_sq = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
    float dist = sqrtf(dist_sq);

    // Overlap (positive when in contact)
    float delta_n = R_i + R_j - dist;

    // Unit normal (from i to j)
    float inv_dist = (dist > 1e-10f) ? (1.0f / dist) : 0.0f;
    float nx = dist_x * inv_dist;
    float ny = dist_y * inv_dist;
    float nz = dist_z * inv_dist;

    // Calculate effective radius and check cohesion range
    float R_eff = R_i * R_j / (R_i + R_j);

    // Cohesion even without contact
    if (cohesion_enabled) {
        float Fc_x, Fc_y, Fc_z;
        compute_cohesion_force(Fc_x, Fc_y, Fc_z,
                               delta_n, nx, ny, nz,
                               cohesion_strength, R_eff, beta, r_cut);
        result.force_x = -Fc_x;  // Force on i (opposite to cohesion toward j)
        result.force_y = -Fc_y;
        result.force_z = -Fc_z;
    }

    // Contact check
    if (delta_n <= 0.0f) {
        result.is_contact = false;
        // Invalidate history for non-contact
        result.delta_t_x = 0.0f; result.delta_t_y = 0.0f; result.delta_t_z = 0.0f;
        result.delta_r_x = 0.0f; result.delta_r_y = 0.0f; result.delta_r_z = 0.0f;
        result.delta_tw_x = 0.0f;
        result.delta_tw_y = 0.0f;
        result.delta_tw_z = 0.0f;
        return result;
    }

    result.is_contact = true;

    // Compute material parameters
    ContactMaterialParams params;
    if (use_visco) {
        params = compute_hertz_mindlin_visco(R_i, R_j, m_i, m_j, I_i, I_j, delta_n,
                                              E_star, G_star, visco_A, mu_s, mu_r, mu_t, beta);
    } else {
        params = compute_hertz_mindlin_resti(R_i, R_j, m_i, m_j, I_i, I_j, delta_n,
                                              E_star, G_star, COR, mu_s, mu_r, mu_t, beta);
    }
    params.gamma = cohesion_strength;
    params.r_cut = r_cut;

    // Contact point arms
    float Ap = (R_i * R_i - R_j * R_j + dist_sq) / (2.0f * dist);  // Distance from i to contact
    float An = dist - Ap;  // Distance from j to contact

    // Relative velocity at contact point
    // V_rel = V_j - V_i + ω_j × (r_j→contact) - ω_i × (r_i→contact)
    double Vrel_x = vel_j_x - vel_i_x;
    double Vrel_y = vel_j_y - vel_i_y;
    double Vrel_z = vel_j_z - vel_i_z;

    // Rotational contribution to velocity
    // ω × r = (ω_y*r_z - ω_z*r_y, ω_z*r_x - ω_x*r_z, ω_x*r_y - ω_y*r_x)
    double Vrot_x = (ny * omega_j_z - nz * omega_j_y) * An +
                   (ny * omega_i_z - nz * omega_i_y) * Ap;
    double Vrot_y = (nz * omega_j_x - nx * omega_j_z) * An +
                   (nz * omega_i_x - nx * omega_i_z) * Ap;
    double Vrot_z = (nx * omega_j_y - ny * omega_j_x) * An +
                   (nx * omega_i_y - ny * omega_i_x) * Ap;

    // Total relative velocity
    double Vtot_x = Vrel_x + Vrot_x;
    double Vtot_y = Vrel_y + Vrot_y;
    double Vtot_z = Vrel_z + Vrot_z;

    // Normal component
    double Vn_dot = Vtot_x * nx + Vtot_y * ny + Vtot_z * nz;
    double Vn_x = Vn_dot * nx;
    double Vn_y = Vn_dot * ny;
    double Vn_z = Vn_dot * nz;

    // Tangential component
    double Vt_x = Vtot_x - Vn_x;
    double Vt_y = Vtot_y - Vn_y;
    double Vt_z = Vtot_z - Vn_z;

    // === Normal Force ===
    float Fn_x, Fn_y, Fn_z, Fn_mag;
    compute_normal_force(Fn_x, Fn_y, Fn_z, Fn_mag,
                         delta_n, nx, ny, nz, Vn_x, Vn_y, Vn_z, params);

    // Elastic energy
    result.energy_elastic = 0.2f * params.K_n * delta_n * delta_n;

    // === Tangential Force ===
    rotate_history_to_tangent(H_t_x, H_t_y, H_t_z, nx, ny, nz);
    float Ft_x, Ft_y, Ft_z, Ft_mag;
    float H_t_new_x, H_t_new_y, H_t_new_z;
    bool slipping;
    compute_tangential_force(Ft_x, Ft_y, Ft_z, Ft_mag,
                             H_t_new_x, H_t_new_y, H_t_new_z, slipping,
                             Vt_x, Vt_y, Vt_z,
                             H_t_x, H_t_y, H_t_z,
                             dt, Fn_mag, params);

    result.is_slipping = slipping;
    result.delta_t_x = H_t_new_x;
    result.delta_t_y = H_t_new_y;
    result.delta_t_z = H_t_new_z;

    // === Rolling Moment ===
    float Mr_x, Mr_y, Mr_z, Mr_mag;
    float Mr_new_x, Mr_new_y, Mr_new_z;
    bool rolling;
    compute_rolling_moment(Mr_x, Mr_y, Mr_z, Mr_mag,
                           Mr_new_x, Mr_new_y, Mr_new_z, rolling,
                           omega_i_x, omega_i_y, omega_i_z,
                           omega_j_x, omega_j_y, omega_j_z,
                           nx, ny, nz,
                           H_r_x, H_r_y, H_r_z,
                           dt, Fn_mag, params);

    result.is_rolling = rolling;
    result.delta_r_x = Mr_new_x;
    result.delta_r_y = Mr_new_y;
    result.delta_r_z = Mr_new_z;

    // === Twisting Moment ===
    float Mt_x, Mt_y, Mt_z, Mt_mag;
    float Mt_new_x, Mt_new_y, Mt_new_z;
    bool twisting;
    compute_twisting_moment(Mt_x, Mt_y, Mt_z, Mt_mag,
                            Mt_new_x, Mt_new_y, Mt_new_z, twisting,
                            omega_i_x, omega_i_y, omega_i_z,
                            omega_j_x, omega_j_y, omega_j_z,
                            nx, ny, nz,
                            H_tw_x, H_tw_y, H_tw_z,
                            dt, Fn_mag, params);

    result.is_twisting = twisting;
    result.delta_tw_x = Mt_new_x;
    result.delta_tw_y = Mt_new_y;
    result.delta_tw_z = Mt_new_z;

    // === Accumulate Forces ===
    // Force on particle i (normal + tangential)
    result.force_x += Fn_x - Ft_x;
    result.force_y += Fn_y - Ft_y;
    result.force_z += Fn_z - Ft_z;

    // === Torque on particle i (owner) ===
    // Force-arm torque: τ = -Ap * (n × F_t)
    // Pure couples (rolling + twisting): subtracted (Mr, Mt defined as magnitudes)
    result.torque_x = -Ap * (ny * Ft_z - nz * Ft_y) - Mr_x - Mt_x;
    result.torque_y = -Ap * (nz * Ft_x - nx * Ft_z) - Mr_y - Mt_y;
    result.torque_z = -Ap * (nx * Ft_y - ny * Ft_x) - Mr_z - Mt_z;

    // === Torque on particle j (partner) ===
    // Force-arm torque: uses An (j's arm to contact), same cross product n × Ft
    // Pure couples: NEGATED relative to i (Newton's 3rd law for couples)
    result.torque_j_x = -An * (ny * Ft_z - nz * Ft_y) + Mr_x + Mt_x;
    result.torque_j_y = -An * (nz * Ft_x - nx * Ft_z) + Mr_y + Mt_y;
    result.torque_j_z = -An * (nx * Ft_y - ny * Ft_x) + Mr_z + Mt_z;

    return result;
}

// Transitional wall-only adapter.  Particle-particle contacts always use the
// database-backed overload above; this keeps specialised sphere-wall modules
// buildable until they are migrated to the same wall material table.
__device__ __forceinline__
ContactForceResult compute_contact_forces(
    float pix, float piy, float piz, double vix, double viy, double viz,
    double wix, double wiy, double wiz, float Ri, float mi, float Ii, float gamma_i,
    float pjx, float pjy, float pjz, double vjx, double vjy, double vjz,
    double wjx, double wjy, double wjz, float Rj, float mj, float Ij, float gamma_j,
    float Htx, float Hty, float Htz, float Hrx, float Hry, float Hrz,
    float Htwx, float Htwy, float Htwz, float E, float nu, float cor,
    float mu_s, float mu_r, float mu_t, float beta, float r_cut,
    bool use_visco, float visco_A, bool cohesion_enabled, float dt) {
    const float E_star = E / (2.0f * (1.0f - nu * nu));
    const float G_star = E / (4.0f * (1.0f + nu) * (2.0f - nu));
    return compute_contact_forces(
        pix, piy, piz, vix, viy, viz, wix, wiy, wiz, Ri, mi, Ii,
        pjx, pjy, pjz, vjx, vjy, vjz, wjx, wjy, wjz, Rj, mj, Ij,
        Htx, Hty, Htz, Hrx, Hry, Hrz, Htwx, Htwy, Htwz,
        E_star, G_star, cor, mu_s, mu_r, mu_t, beta, r_cut,
        use_visco, visco_A, cohesion_enabled, 0.5f * (gamma_i + gamma_j), dt);
}

// =============================================================================
// Force Policy Traits (Compile-time Configuration)
// =============================================================================
// These allow compile-time selection of physics components via templates
// Runtime branching is kept minimal for performance
// =============================================================================

// Contact model policies
struct RestitutionPolicy {
    static constexpr bool use_viscous = false;
    static constexpr float viscous_A = 0.0f;
};

struct ViscousPolicy {
    static constexpr bool use_viscous = true;
    static constexpr float viscous_A = 0.1f;  // Typical value for ice
};

// Cohesion policies
struct CohesionEnabledPolicy {
    static constexpr bool enabled = true;
};

struct CohesionDisabledPolicy {
    static constexpr bool enabled = false;
};

// Rolling friction policies
struct RollingEnabledPolicy {
    static constexpr bool enabled = true;
};

struct RollingDisabledPolicy {
    static constexpr bool enabled = false;
};

// =============================================================================
// Force Accumulation Utilities
// =============================================================================

// Atomic add for float (CUDA native)
__device__ __forceinline__
void atomic_add_force(double* force_x, double* force_y, double* force_z,
                      double fx, double fy, double fz, int idx) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(&force_x[idx], fx);
    atomicAdd(&force_y[idx], fy);
    atomicAdd(&force_z[idx], fz);
#else
    // Fallback for older architectures (cas loop) if needed
    // But for this project we likely target Pascal or newer
    atomicAdd(force_x + idx, fx); // Should fail compile if not supported, giving feedback
    atomicAdd(force_y + idx, fy);
    atomicAdd(force_z + idx, fz);
#endif
}

__device__ __forceinline__
void atomic_add_torque(double* torque_x, double* torque_y, double* torque_z,
                       double tx, double ty, double tz, int idx) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(&torque_x[idx], tx);
    atomicAdd(&torque_y[idx], ty);
    atomicAdd(&torque_z[idx], tz);
#else
    atomicAdd(torque_x + idx, tx);
    atomicAdd(torque_y + idx, ty);
    atomicAdd(torque_z + idx, tz);
#endif
}


// Atomic add for float (CUDA native)
__device__ __forceinline__
void atomic_add_force(float* force_x, float* force_y, float* force_z,
                      float fx, float fy, float fz, int idx) {
    atomicAdd(&force_x[idx], fx);
    atomicAdd(&force_y[idx], fy);
    atomicAdd(&force_z[idx], fz);
}

__device__ __forceinline__
void atomic_add_torque(float* torque_x, float* torque_y, float* torque_z,
                       float tx, float ty, float tz, int idx) {
    atomicAdd(&torque_x[idx], tx);
    atomicAdd(&torque_y[idx], ty);
    atomicAdd(&torque_z[idx], tz);
}



// Contact history compaction launcher
void launch_compact_contact_history_kernel(
    ContactHistory* contact_history,
    const int32_t* contact_history_offset,
    int32_t* num_contacts,
    int32_t max_contacts_per_particle,
    int32_t num_particles,
    int block_size,
    cudaStream_t stream = 0
);

// =============================================================================
// Force Kernel Dispatcher (Interface)
// =============================================================================
class ForceKernelDispatcher {
public:
    static void compute_particle_contacts(
        const SimulationState& state,
        const SimConfig& config,
        float dt,
        cudaStream_t stream = 0
    );

    static void compute_wall_contacts(
        SimulationState& state,
        const SimConfig& config,
        float dt,
        cudaStream_t stream = 0
    );

    static void compute_accelerations(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    );
};

} // namespace cudadem

#endif // CUDADEM_PARTICLE_H

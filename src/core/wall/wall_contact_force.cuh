#ifndef CUDADEM_WALL_CONTACT_FORCE_CUH
#define CUDADEM_WALL_CONTACT_FORCE_CUH

// =============================================================================
// Shared Wall Contact Force Computation
// =============================================================================
// Device functions used by both plain_wall.cu and trimesh_wall.cu.
// Extracted to a .cuh header so both translation units can see them.
// =============================================================================

#include "wall.h"
#include "../material/material_database.h"
#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

// =============================================================================
// Friction Policies
// =============================================================================

struct FullFrictionPolicy {
    static constexpr bool enable_tangential = true;
    static constexpr bool enable_rolling = true;
    static constexpr bool enable_twisting = true;
};

struct SmoothWallPolicy {
    static constexpr bool enable_tangential = false;
    static constexpr bool enable_rolling = false;
    static constexpr bool enable_twisting = false;
};

__device__ __forceinline__
void rotate_wall_history_to_tangent(float& x, float& y, float& z,
                                    float nx, float ny, float nz) {
    const float old_mag = sqrtf(x * x + y * y + z * z);
    const float normal = x * nx + y * ny + z * nz;
    x -= normal * nx; y -= normal * ny; z -= normal * nz;
    const float projected_mag = sqrtf(x * x + y * y + z * z);
    if (old_mag > 1e-20f && projected_mag > 1e-20f) {
        const float scale = old_mag / projected_mag;
        x *= scale; y *= scale; z *= scale;
    } else {
        x = 0.0f; y = 0.0f; z = 0.0f;
    }
}

__device__ __forceinline__
void rotate_wall_history_to_normal(float& x, float& y, float& z,
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

// =============================================================================
// Wall Material Parameters
// =============================================================================

struct WallMaterialParams {
    float R_eff;
    float M_eff;
    float I_eff;
    float K_n, K_s, K_r, K_t;
    float C_n, C_s, C_r, C_t;
    float mu_s, mu_d, mu_r, mu_t;
    float beta;

    __device__ WallMaterialParams()
        : R_eff(0), M_eff(0), I_eff(0),
          K_n(0), K_s(0), K_r(0), K_t(0),
          C_n(0), C_s(0), C_r(0), C_t(0),
          mu_s(1.0f), mu_d(1.0f), mu_r(0.0f), mu_t(0.0f), beta(0.7f) {}
};

// Resolve the explicitly configured material pair for a particle and wall.
// The wall is geometrically infinite for plane/mesh/funnel contacts, but its
// constitutive response is still the symmetric DEM pair response.  There is
// intentionally no fallback or mixing rule for interface properties.
__device__ __forceinline__
bool resolve_wall_material_pair(
    int32_t particle_material_id, int32_t wall_material_id,
    const MaterialRecord* materials,
    const DemInteractionProperties* interactions,
    uint32_t material_count,
    float& E_star, float& G_star,
    DemInteractionProperties& pair
) {
    if (particle_material_id < 0 || wall_material_id < 0 ||
        static_cast<uint32_t>(particle_material_id) >= material_count ||
        static_cast<uint32_t>(wall_material_id) >= material_count) return false;
    const MaterialRecord& particle = materials[particle_material_id];
    const MaterialRecord& wall = materials[wall_material_id];
    if (!particle.has_dem || !wall.has_dem) return false;
    pair = interactions[static_cast<size_t>(particle_material_id) * material_count + wall_material_id];
    const DemMaterialProperties& a = particle.dem;
    const DemMaterialProperties& b = wall.dem;
    const float Ga = a.youngs_modulus / (2.0f * (1.0f + a.poisson_ratio));
    const float Gb = b.youngs_modulus / (2.0f * (1.0f + b.poisson_ratio));
    E_star = 1.0f / ((1.0f - a.poisson_ratio * a.poisson_ratio) / a.youngs_modulus +
                     (1.0f - b.poisson_ratio * b.poisson_ratio) / b.youngs_modulus);
    G_star = 1.0f / ((2.0f - a.poisson_ratio) / Ga +
                     (2.0f - b.poisson_ratio) / Gb);
    return true;
}

// Scheeres cohesion for an infinite wall uses the particle radius as the
// effective radius.  It intentionally contributes only to the normal force:
// Coulomb, rolling, and twisting limits continue to use the validated
// non-cohesive normal load.
__device__ __forceinline__
void add_wall_cohesion(WallForceResult& result, float delta_n,
                       float nx, float ny, float nz, float particle_radius,
                       const DemInteractionProperties& pair) {
    if (!pair.cohesion_enabled || pair.cohesion_strength <= 0.0f) return;
    float reach = 0.0f;
    if (delta_n < 0.0f) {
        // A unit cutoff is the contact-only limit.  It has no finite gap over
        // which to ramp and must not disable cohesion at contact.
        if (pair.cohesion_cutoff_factor <= 1.0f) return;
        reach = particle_radius * (pair.cohesion_cutoff_factor - 1.0f);
        if (delta_n <= -reach) return;
    }
    const float full = pair.cohesion_strength *
                       (pair.shape_parameter * particle_radius) *
                       (pair.shape_parameter * particle_radius);
    float magnitude = full;
    if (delta_n < -0.5f * reach) {
        magnitude *= 2.0f * (delta_n + reach) / reach;
    }
    // Wall normal points from the wall toward the particle; cohesion is the
    // opposite direction, toward the wall.
    result.force_x -= magnitude * nx;
    result.force_y -= magnitude * ny;
    result.force_z -= magnitude * nz;
}

// =============================================================================
// Hertz-Mindlin Material Params (Restitution-based)
// =============================================================================

__device__ __forceinline__
WallMaterialParams compute_wall_hertz_mindlin_resti(
    float R_particle, float m_particle, float I_particle,
    float delta_n,
    float E_star, float G_star, float COR,
    float mu_s, float mu_r, float mu_t, float beta
) {
    WallMaterialParams params;
    params.R_eff = R_particle;
    params.M_eff = m_particle;
    params.I_eff = I_particle;
    params.mu_s = mu_s;
    params.mu_d = mu_s;
    params.mu_r = mu_r;
    params.mu_t = mu_t;
    params.beta = beta;

    float sqrt_Rd = sqrtf(params.R_eff * delta_n);
    params.K_n = (4.0f / 3.0f) * E_star * sqrt_Rd;

    if (COR > 0.0f && COR < 1.0f) {
        float ln_COR = logf(COR);
        float coeff = 2.0f * sqrtf(5.0f / 6.0f) * ln_COR /
                      sqrtf(ln_COR * ln_COR + 9.8696044f);
        float sqrt_delta = sqrtf(sqrtf(delta_n));
        float sqrt_Reff = sqrtf(sqrtf(params.R_eff));
        params.C_n = coeff * sqrtf(2.0f * params.M_eff * E_star) *
                     sqrt_Reff * sqrt_delta;
    } else {
        params.C_n = 0.0f;
    }

    params.K_s = 8.0f * G_star * sqrtf(params.R_eff) * sqrtf(delta_n);

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

    float beta_R_sq = (beta * params.R_eff) * (beta * params.R_eff);
    params.K_r = 0.25f * params.K_n * beta_R_sq;
    params.C_r = 0.25f * params.C_n * beta_R_sq;
    params.K_t = 0.5f * params.K_s * beta_R_sq;
    params.C_t = 0.5f * params.C_s * beta_R_sq;

    return params;
}

// =============================================================================
// Hertz-Mindlin Material Params (Viscous damping)
// =============================================================================

__device__ __forceinline__
WallMaterialParams compute_wall_hertz_mindlin_visco(
    float R_particle, float m_particle, float I_particle,
    float delta_n,
    float E_star, float G_star, float A, float COR,
    float mu_s, float mu_r, float mu_t, float beta
) {
    WallMaterialParams params;
    params.R_eff = R_particle;
    params.M_eff = m_particle;
    params.I_eff = I_particle;
    params.mu_s = mu_s;
    params.mu_d = mu_s;
    params.mu_r = mu_r;
    params.mu_t = mu_t;
    params.beta = beta;

    float sqrt_Rd = sqrtf(params.R_eff * delta_n);
    params.K_n = (4.0f / 3.0f) * E_star * sqrt_Rd;
    params.K_s = 8.0f * G_star * sqrtf(params.R_eff) * sqrtf(delta_n);
    params.C_n = -params.K_n * A;

    if (COR > 1.0f) {
        params.C_s = -8.0f * G_star * sqrtf(delta_n) * A;
    } else if (COR >= 0.0f) {
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

    float beta_R_sq = (beta * params.R_eff) * (beta * params.R_eff);
    params.K_r = 0.25f * params.K_n * beta_R_sq;
    params.C_r = 0.25f * params.C_n * beta_R_sq;
    params.K_t = 0.5f * params.K_s * beta_R_sq;
    params.C_t = 0.5f * params.C_s * beta_R_sq;

    return params;
}

// =============================================================================
// Core Wall Force Calculation (takes pre-computed overlap and distance)
// =============================================================================
// Used directly by trimesh_wall.cu with overlap from compute_triangle_contact.
// Plain walls call this via the calculate_wall_force wrapper below.
// =============================================================================

template<typename FrictionPolicy>
__device__ __forceinline__
WallForceResult calculate_wall_force_impl(
    float delta_n, float Ap,
    double vel_x, double vel_y, double vel_z,
    double omega_x, double omega_y, double omega_z,
    float R_particle, float m_particle, float I_particle,
    float wall_nx, float wall_ny, float wall_nz,
    double wall_vx, double wall_vy, double wall_vz,
    float H_t_x, float H_t_y, float H_t_z,
    float H_r_x, float H_r_y, float H_r_z,
    float H_tw_x, float H_tw_y, float H_tw_z,
    float E_star, float G_star, float COR,
    float mu_s, float mu_r, float mu_t, float beta,
    bool use_visco, float visco_A,
    float dt
) {
    WallForceResult result;

    if (delta_n <= 0.0f) {
        result.is_contact = false;
        return result;
    }
    result.is_contact = true;

    WallMaterialParams params;
    if (use_visco) {
        params = compute_wall_hertz_mindlin_visco(
            R_particle, m_particle, I_particle, delta_n,
            E_star, G_star, visco_A, COR, mu_s, mu_r, mu_t, beta);
    } else {
        params = compute_wall_hertz_mindlin_resti(
            R_particle, m_particle, I_particle, delta_n,
            E_star, G_star, COR, mu_s, mu_r, mu_t, beta);
    }
    double Vrel_x = vel_x - wall_vx;
    double Vrel_y = vel_y - wall_vy;
    double Vrel_z = vel_z - wall_vz;
    double Vrot_x = (wall_ny * omega_z - wall_nz * omega_y) * Ap;
    double Vrot_y = (wall_nz * omega_x - wall_nx * omega_z) * Ap;
    double Vrot_z = (wall_nx * omega_y - wall_ny * omega_x) * Ap;
    double Vtot_x = Vrel_x + Vrot_x;
    double Vtot_y = Vrel_y + Vrot_y;
    double Vtot_z = Vrel_z + Vrot_z;

    double Vn_dot = Vtot_x * wall_nx + Vtot_y * wall_ny + Vtot_z * wall_nz;
    double Vt_x = Vtot_x - Vn_dot * wall_nx;
    double Vt_y = Vtot_y - Vn_dot * wall_ny;
    double Vt_z = Vtot_z - Vn_dot * wall_nz;

    // C_n < 0 (from COR formula); Vn_dot > 0 when separating.
    // Correct sign: + C_n * Vn_dot  (was - C_n, which inverted damping)
    float Fn_mag = params.K_n * delta_n + params.C_n * static_cast<float>(Vn_dot);
    float Fn_mag_abs = fabsf(Fn_mag);  // Absolute magnitude for friction limits (matches DEMBody)
    result.energy_elastic = 0.2f * params.K_n * delta_n * delta_n;
    result.force_x = Fn_mag * wall_nx;
    result.force_y = Fn_mag * wall_ny;
    result.force_z = Fn_mag * wall_nz;
    result.torque_x = 0.0f;
    result.torque_y = 0.0f;
    result.torque_z = 0.0f;

    if constexpr (FrictionPolicy::enable_tangential) {
        rotate_wall_history_to_tangent(H_t_x, H_t_y, H_t_z,
                                       wall_nx, wall_ny, wall_nz);
        float Ds_x = Vt_x * dt, Ds_y = Vt_y * dt, Ds_z = Vt_z * dt;
        float Ft_x = -params.K_s * Ds_x + params.C_s * Vt_x + H_t_x;
        float Ft_y = -params.K_s * Ds_y + params.C_s * Vt_y + H_t_y;
        float Ft_z = -params.K_s * Ds_z + params.C_s * Vt_z + H_t_z;
        float Ft_mag = sqrtf(Ft_x * Ft_x + Ft_y * Ft_y + Ft_z * Ft_z);
        float Ft_max = params.mu_s * Fn_mag_abs;
        if (Ft_mag > Ft_max && Ft_mag > 1e-10f) {
            float scale = params.mu_d * Fn_mag_abs / Ft_mag;
            Ft_x *= scale; Ft_y *= scale; Ft_z *= scale;
            result.H_t_x = Ft_x - params.C_s * Vt_x;
            result.H_t_y = Ft_y - params.C_s * Vt_y;
            result.H_t_z = Ft_z - params.C_s * Vt_z;
            result.is_slipping = true;
        } else {
            result.H_t_x = Ft_x - params.C_s * Vt_x;
            result.H_t_y = Ft_y - params.C_s * Vt_y;
            result.H_t_z = Ft_z - params.C_s * Vt_z;
            result.is_slipping = false;
        }
        result.force_x += Ft_x; result.force_y += Ft_y; result.force_z += Ft_z;
        result.torque_x = -Ap * (wall_ny * Ft_z - wall_nz * Ft_y);
        result.torque_y = -Ap * (wall_nz * Ft_x - wall_nx * Ft_z);
        result.torque_z = -Ap * (wall_nx * Ft_y - wall_ny * Ft_x);
    }

    if constexpr (FrictionPolicy::enable_rolling) {
        rotate_wall_history_to_tangent(H_r_x, H_r_y, H_r_z,
                                       wall_nx, wall_ny, wall_nz);
        double dtheta_x = omega_x * dt, dtheta_y = omega_y * dt, dtheta_z = omega_z * dt;
        double dtheta_n = dtheta_x * wall_nx + dtheta_y * wall_ny + dtheta_z * wall_nz;
        double dtheta_r_x = dtheta_x - dtheta_n * wall_nx;
        double dtheta_r_y = dtheta_y - dtheta_n * wall_ny;
        double dtheta_r_z = dtheta_z - dtheta_n * wall_nz;
        float Mr_x = -params.K_r * dtheta_r_x + H_r_x;
        float Mr_y = -params.K_r * dtheta_r_y + H_r_y;
        float Mr_z = -params.K_r * dtheta_r_z + H_r_z;
        float Mr_mag = sqrtf(Mr_x * Mr_x + Mr_y * Mr_y + Mr_z * Mr_z);
        float Mr_max = 2.1f * 0.25f * params.mu_r * params.beta * params.R_eff * Fn_mag_abs;
        if (Mr_mag > Mr_max && Mr_mag > 1e-10f) {
            float s = Mr_max / Mr_mag;
            Mr_x *= s; Mr_y *= s; Mr_z *= s;
            result.is_rolling = true;
        } else {
            result.is_rolling = false;
            result.H_r_x = Mr_x; result.H_r_y = Mr_y; result.H_r_z = Mr_z;
            float dr_mag = sqrtf(dtheta_r_x*dtheta_r_x + dtheta_r_y*dtheta_r_y + dtheta_r_z*dtheta_r_z);
            if (dr_mag > 1e-10f) {
                Mr_x += params.C_r * dtheta_r_x / dt;
                Mr_y += params.C_r * dtheta_r_y / dt;
                Mr_z += params.C_r * dtheta_r_z / dt;
            }
        }
        if (result.is_rolling) {
            result.H_r_x = Mr_x; result.H_r_y = Mr_y; result.H_r_z = Mr_z;
        }
        result.torque_x += Mr_x; result.torque_y += Mr_y; result.torque_z += Mr_z;
    }

    if constexpr (FrictionPolicy::enable_twisting) {
        rotate_wall_history_to_normal(H_tw_x, H_tw_y, H_tw_z,
                                      wall_nx, wall_ny, wall_nz);
        double dtheta_x = omega_x * dt, dtheta_y = omega_y * dt, dtheta_z = omega_z * dt;
        double dtheta_n = dtheta_x * wall_nx + dtheta_y * wall_ny + dtheta_z * wall_nz;
        double dtheta_t_x = dtheta_n * wall_nx;
        double dtheta_t_y = dtheta_n * wall_ny;
        double dtheta_t_z = dtheta_n * wall_nz;
        float Mt_x = -params.K_t * dtheta_t_x + H_tw_x;
        float Mt_y = -params.K_t * dtheta_t_y + H_tw_y;
        float Mt_z = -params.K_t * dtheta_t_z + H_tw_z;
        float Mt_mag = sqrtf(Mt_x * Mt_x + Mt_y * Mt_y + Mt_z * Mt_z);
        float Mt_max = 0.65f * params.mu_t * params.beta * params.R_eff * Fn_mag_abs;
        if (Mt_mag > Mt_max && Mt_mag > 1e-10f) {
            float s = Mt_max / Mt_mag;
            Mt_x *= s; Mt_y *= s; Mt_z *= s;
            result.is_twisting = true;
        } else {
            result.is_twisting = false;
            result.H_tw_x = Mt_x; result.H_tw_y = Mt_y; result.H_tw_z = Mt_z;
            float dt_mag = sqrtf(dtheta_t_x*dtheta_t_x + dtheta_t_y*dtheta_t_y + dtheta_t_z*dtheta_t_z);
            if (dt_mag > 1e-10f) {
                Mt_x += params.C_t * dtheta_t_x / dt;
                Mt_y += params.C_t * dtheta_t_y / dt;
                Mt_z += params.C_t * dtheta_t_z / dt;
            }
        }
        if (result.is_twisting) {
            result.H_tw_x = Mt_x; result.H_tw_y = Mt_y; result.H_tw_z = Mt_z;
        }
        result.torque_x += Mt_x; result.torque_y += Mt_y; result.torque_z += Mt_z;
    }

    return result;
}

// =============================================================================
// Plane Wall Wrapper (computes overlap from plane geometry, then delegates)
// =============================================================================

template<typename FrictionPolicy>
__device__ __forceinline__
WallForceResult calculate_wall_force(
    double pos_x, double pos_y, double pos_z,
    double vel_x, double vel_y, double vel_z,
    double omega_x, double omega_y, double omega_z,
    float R_particle, float m_particle, float I_particle,
    float wall_nx, float wall_ny, float wall_nz,
    float wall_px, float wall_py, float wall_pz,
    double wall_vx, double wall_vy, double wall_vz,
    float H_t_x, float H_t_y, float H_t_z,
    float H_r_x, float H_r_y, float H_r_z,
    float H_tw_x, float H_tw_y, float H_tw_z,
    float E_star, float G_star, float COR,
    float mu_s, float mu_r, float mu_t, float beta,
    bool use_visco, float visco_A,
    float dt
) {
    float signed_dist = plane_signed_distance(
        pos_x, pos_y, pos_z,
        wall_nx, wall_ny, wall_nz,
        wall_px, wall_py, wall_pz
    );
    float delta_n = compute_plane_overlap(signed_dist, R_particle);

    return calculate_wall_force_impl<FrictionPolicy>(
        delta_n, signed_dist,
        vel_x, vel_y, vel_z, omega_x, omega_y, omega_z,
        R_particle, m_particle, I_particle,
        wall_nx, wall_ny, wall_nz,
        wall_vx, wall_vy, wall_vz,
        H_t_x, H_t_y, H_t_z,
        H_r_x, H_r_y, H_r_z,
        H_tw_x, H_tw_y, H_tw_z,
        E_star, G_star, COR, mu_s, mu_r, mu_t, beta,
        use_visco, visco_A, dt
    );
}

} // namespace cudadem

#endif // CUDADEM_WALL_CONTACT_FORCE_CUH

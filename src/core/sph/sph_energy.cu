#include "sph_stage.h"
#include "src/core/sph/sph_energy.h"
#include "src/core/neighbor/neighbor_api.h"

#include "src/core/sph/sph_device_helpers.cuh"
#include "src/core/sph/sph_neighbor_query.cuh"
#include "src/core/sph/sph_kernel_correction.cuh"
#include "src/core/sph/sph_numerical_safeguards.cuh"
#include "src/core/sph/sph_energy_math.cuh"
#include "src/core/material/material_database.h"
#include "cuda_utils.h"
#include <cmath>

namespace cudadem {

// ---------------------------------------------------------------------------
// Kernel: compute de/dt for each particle
//
// de/dt = (P/rho^2) * sum_j m_j * (v_i - v_j) . grad_W_ij
//       + 0.5 * sum_j m_j * Pi_ij * (v_i - v_j) . grad_W_ij   (viscous heating)
//       + (1/rho) * S_ab * eps_dot_ab                           (shear heating, if strength)
//
// Note: the pressure work term uses (v_i - v_j) to match the antisymmetric form
// consistent with the momentum equation (Monaghan 1992, Eq. 3.20).
// ---------------------------------------------------------------------------
__global__ void compute_sph_energy_rate_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const double* __restrict__ vel_x, const double* __restrict__ vel_y, const double* __restrict__ vel_z,
    const float*  __restrict__ mass,
    const double* __restrict__ density,
    const float*  __restrict__ pressure,
    const float*  __restrict__ c_sound,
    const float*  __restrict__ smoothing_length,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const float*  __restrict__ S_xx, const float* __restrict__ S_xy, const float* __restrict__ S_xz,
    const float*  __restrict__ S_yy, const float* __restrict__ S_yz, const float* __restrict__ S_zz,
    const float* __restrict__ L_00, const float* __restrict__ L_01, const float* __restrict__ L_02,
    const float* __restrict__ L_10, const float* __restrict__ L_11, const float* __restrict__ L_12,
    const float* __restrict__ L_20, const float* __restrict__ L_21, const float* __restrict__ L_22,
    double* __restrict__ dedt,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    float alpha_visc, float beta_visc,
    bool strength_enabled,
    bool use_kernel_correction,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float pi_x = static_cast<float>(pos_x[i]);
    float pi_y = static_cast<float>(pos_y[i]);
    float pi_z = static_cast<float>(pos_z[i]);
    float vi_x = static_cast<float>(vel_x[i]);
    float vi_y = static_cast<float>(vel_y[i]);
    float vi_z = static_cast<float>(vel_z[i]);
    float hi    = smoothing_length[i];
    double rho_i = density[i];
    float p_i   = pressure[i];
    float c_i   = c_sound[i];

    float L00 = L_00[i];
    float L01 = L_01[i];
    float L02 = L_02[i];
    float L10 = L_10[i];
    float L11 = L_11[i];
    float L12 = L_12[i];
    float L20 = L_20[i];
    float L21 = L_21[i];
    float L22 = L_22[i];

    // advance_sph_strength seeds dedt with the time-centred SPHSOL S:D/rho
    // contribution.  Hydro-only runs must ignore any stale checkpoint stress.
    double dedt_i = strength_enabled ? sanitize_d(dedt[i]) : 0.0;

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float hj   = smoothing_length[j];
            float h_ij = 0.5f * (hi + hj);
            if (dist >= kernel_cutoff_factor * h_ij) return;

            float m_j   = mass[j];
            double rho_j = sanitize_density(
                density[j], 0.8 * materials[material_id[j]].sph_tillotson.reference_density, 1e10);
            float p_j   = sanitize_pressure(pressure[j]);

            float dW_dr = sph_kernel_grad(dist, h_ij);
            float inv_d = 1.0f / dist;
            float gW_x = dW_dr * dx_ij * inv_d;
            float gW_y = dW_dr * dy_ij * inv_d;
            float gW_z = dW_dr * dz_ij * inv_d;

            if (use_kernel_correction) {
            apply_kernel_correction(gW_x, gW_y, gW_z, L00, L01, L02, L10, L11, L12, L20, L21, L22);
            }

            float dvx = vi_x - static_cast<float>(vel_x[j]);
            float dvy = vi_y - static_cast<float>(vel_y[j]);
            float dvz = vi_z - static_cast<float>(vel_z[j]);

            float v_dot_gW = dvx * gW_x + dvy * gW_y + dvz * gW_z;

            float Pi_ij_term = 0.0f;
            float dot_vr = dvx * dx_ij + dvy * dy_ij + dvz * dz_ij;
            if (dot_vr < 0.0f) {
                float dist_sq = dist * dist;
                float mu_ij = (h_ij * dot_vr) / (dist_sq + 0.01f * h_ij * h_ij);
                float c_ij  = 0.5f * (c_i + sanitize_c_sound(c_sound[j]));
            double rho_ij = 0.5 * (rho_i + rho_j);
                Pi_ij_term = (-alpha_visc * c_ij * mu_ij + beta_visc * mu_ij * mu_ij) / rho_ij;
            }

            // Geometric pressure work plus conservative pairwise viscous heat.
            // The 1/2 applies only to Pi: both endpoints receive this term, so
            // together they recover exactly the kinetic energy removed by the
            // symmetric Monaghan viscosity force.  The pressure term remains
            // the local geometric-density work used by GRANIT.
            dedt_i += m_j * sph_energy_pair_coefficient(
                p_i, rho_i, rho_j, Pi_ij_term) * v_dot_gW;
        });

    dedt[i] = sanitize_d(dedt_i);
}

// ---------------------------------------------------------------------------
// Kernel: integrate energy: e[i] += dedt[i] * dt
// ---------------------------------------------------------------------------
__global__ void integrate_energy_kernel(
    double* __restrict__ energy,
    const double* __restrict__ dedt,
    float dt,
    int32_t num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    energy[i] += dedt[i] * dt;
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void compute_sph_energy_rate(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;

    if (!sph_current_stage(config, state.current_time).evolve_energy) {
        CUDA_CHECK(cudaMemsetAsync(state.particles.dedt, 0,
            state.particles.num_particles * sizeof(double), stream));
        return;
    }

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(compute_sph_energy_rate_kernel, nb, bs, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
        state.particles.mass,
        state.particles.density,
        state.particles.pressure,
        state.particles.c_sound,
        state.particles.smoothing_length,
        state.particles.material_id, state.material_database.materials,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz,
        state.particles.L_00, state.particles.L_01, state.particles.L_02,
        state.particles.L_10, state.particles.L_11, state.particles.L_12,
        state.particles.L_20, state.particles.L_21, state.particles.L_22,
        state.particles.dedt,
        state.neighbor,
        config.sph.kernel_cutoff_factor,
        config.sph.viscosity,
        2.0f * config.sph.viscosity,
        sph_current_stage(config, state.current_time).use_strength,
        config.sph.use_kernel_correction,
        num
    );
}

void integrate_sph_energy(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;

    if (!sph_current_stage(config, state.current_time).evolve_energy) return;

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(integrate_energy_kernel, nb, bs, 0, stream,
        state.particles.energy,
        state.particles.dedt,
        static_cast<float>(state.dt),
        num
    );
}

} // namespace cudadem

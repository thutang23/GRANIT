#include "src/core/sph/sph_physics.h"
#include "src/core/sph/sph_eos_math.cuh"
#include "src/core/sph/sph_stage.h"
#include "src/core/sph/sph_porosity_math.cuh"
#include "src/core/sph/sph_device_helpers.cuh"
#include "src/core/sph/sph_neighbor_query.cuh"
#include "src/core/sph/sph_numerical_safeguards.cuh"
#include "src/core/sph/sph_artificial_stress.cuh"
#include "src/core/material/material_database.h"
#include "src/core/neighbor/neighbor_api.h"
#include "cuda_utils.h"
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace cudadem {

// Global max smoothing length for symmetric SPH neighbor queries.
// Host writes via cudaMemcpyToSymbolAsync in update_sph_h_max_global().
__constant__ float dev_sph_h_max_global = 0.0f;


// =============================================================================
// Density Rate (Continuous Density)
// =============================================================================
__global__ void compute_sph_density_rate_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const double* __restrict__ vel_x, const double* __restrict__ vel_y, const double* __restrict__ vel_z,
    const float* __restrict__ mass, const double* __restrict__ density,
    const float* __restrict__ smoothing_length,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    double* __restrict__ rho_rate,
    const float* __restrict__ L_00, const float* __restrict__ L_01, const float* __restrict__ L_02,
    const float* __restrict__ L_10, const float* __restrict__ L_11, const float* __restrict__ L_12,
    const float* __restrict__ L_20, const float* __restrict__ L_21, const float* __restrict__ L_22,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
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
    float hi = smoothing_length[i];
    double rho_i = density[i];

    float L00 = L_00[i]; float L01 = L_01[i]; float L02 = L_02[i];
    float L10 = L_10[i]; float L11 = L_11[i]; float L12 = L_12[i];
    float L20 = L_20[i]; float L21 = L_21[i]; float L22 = L_22[i];

    double div_v = 0.0;

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float m_j = mass[j];
            const MaterialRecord& material_j = materials[material_id[j]];
            double rho_j = sanitize_density(
                density[j], 0.8 * material_j.sph_tillotson.reference_density, 1e10);
            float hj = smoothing_length[j];
            float h_ij = 0.5f * (hi + hj);

            // Re-check with h_ij since for_each_sph_neighbor filtered by kernel_cutoff_factor * hi
            if (dist >= kernel_cutoff_factor * h_ij) return;

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

            double v_dot_gW = static_cast<double>(dvx) * gW_x
                            + static_cast<double>(dvy) * gW_y
                            + static_cast<double>(dvz) * gW_z;
            div_v += (m_j / rho_j) * v_dot_gW;
        });

    rho_rate[i] = sanitize_d(rho_i * div_v);  // dρ/dt = ρ·Σ(m_j/ρ_j)(v_i-v_j)·∇W (Monaghan 1992 eq.3.4)
}

// =============================================================================
// Density Integration (Continuous Density)
// =============================================================================
// Integrates dρ/dt → ρ using forward Euler, matching SPHSOL solver.cc:157-166.
// Clamped to [0.8*ρ₀, ∞) to prevent negative/zero density.
__global__ void integrate_sph_density_kernel(
    double* __restrict__ density,
    const double* __restrict__ rho_rate,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    float dt,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double rho_floor = 0.8 * materials[material_id[i]].sph_tillotson.reference_density;
    double new_rho = density[i] + rho_rate[i] * static_cast<double>(dt);
    density[i] = sanitize_density(new_rho, rho_floor);
}

// =============================================================================
// 1. Density Summation
// =============================================================================
__global__ void compute_sph_density_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass, const float* __restrict__ smoothing_length,
    double* __restrict__ density,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    bool enable_shepard_filter,
    int32_t num_particles,
    int32_t* __restrict__ debug_neighbor_count = nullptr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float pi_x = static_cast<float>(pos_x[i]);
    float pi_y = static_cast<float>(pos_y[i]);
    float pi_z = static_cast<float>(pos_z[i]);
    float hi = smoothing_length[i];
    float m_i = mass[i];

    // Self-density contribution
    double rho_i = static_cast<double>(m_i) * sph_kernel_w(0.0f, hi);
    double w_sum = enable_shepard_filter
        ? (static_cast<double>(m_i) / density[i] * sph_kernel_w(0.0f, hi)) : 0.0;

    // SPHSOL density floor (solver.cc:145-151): clamp rho >= 0.8*rho0
    // Prevents tension cascade when surface particles have reduced neighbor count
    double rho_floor = 0.8 * materials[material_id[i]].sph_tillotson.reference_density;

    int32_t neighbor_count = 0;

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float m_j = mass[j];
            double rho_j = sanitize_density(
                density[j], 0.8 * materials[material_id[j]].sph_tillotson.reference_density, 1e10);
            const float hj = smoothing_length[j];
            const float h_ij = 0.5f * (hi + hj);

            // The broad query is sized from h_i and the global maximum h;
            // recheck the actual symmetric pair support before summing.
            if (dist >= kernel_cutoff_factor * h_ij) return;

            const float w_ij = sph_kernel_w(dist, h_ij);
            rho_i += m_j * w_ij;
            if (enable_shepard_filter) {
                w_sum += (static_cast<double>(m_j) / rho_j) * w_ij;
            }
            ++neighbor_count;
        });

    if (enable_shepard_filter && w_sum > 1e-6) {
        density[i] = sanitize_density(rho_i / w_sum, rho_floor);
    } else {
        density[i] = sanitize_density(rho_i, rho_floor);
    }

    if (debug_neighbor_count != nullptr) {
        debug_neighbor_count[i] = neighbor_count;
    }
}

// No separate debug kernel — neighbor counting is embedded in compute_sph_density_kernel
// via the optional debug_neighbor_count parameter (see SPH_DEBUG blocks below).

// =============================================================================
// 2. Pressure EOS (Tillotson Equation)
// =============================================================================
__global__ void compute_sph_pressure_kernel(
    const double* __restrict__ density,
    const double* __restrict__ energy,
    const float* __restrict__ damage,
    const float* __restrict__ distension,
    float* __restrict__ pressure, float* __restrict__ c_sound,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    bool elastic_eos,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const MaterialRecord& material = materials[material_id[i]];
    const SphEosResult result = sph_eos_state(material, density[i], energy[i],
        damage ? damage[i] : 0.0, distension[i], elastic_eos);
    pressure[i] = sanitize_pressure(static_cast<float>(result.pressure));
    c_sound[i] = sanitize_c_sound(static_cast<float>(result.c_sound));
}

// =============================================================================
// 3. Force Calculation (Pressure Gradient + Artificial Viscosity)
// =============================================================================
__global__ void compute_sph_forces_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const double* __restrict__ vel_x, const double* __restrict__ vel_y, const double* __restrict__ vel_z,
    const float* __restrict__ mass, const double* __restrict__ density,
    const float* __restrict__ pressure, const float* __restrict__ c_sound,
    const float* __restrict__ S_xx, const float* __restrict__ S_xy, const float* __restrict__ S_xz,
    const float* __restrict__ S_yy, const float* __restrict__ S_yz, const float* __restrict__ S_zz,
    const float* __restrict__ smoothing_length,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    double* __restrict__ force_x, double* __restrict__ force_y, double* __restrict__ force_z,
    const float* __restrict__ L_00, const float* __restrict__ L_01, const float* __restrict__ L_02,
    const float* __restrict__ L_10, const float* __restrict__ L_11, const float* __restrict__ L_12,
    const float* __restrict__ L_20, const float* __restrict__ L_21, const float* __restrict__ L_22,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    bool use_kernel_correction,
    float alpha, float beta,
    bool strength_enabled,
    bool use_artificial_stress,
    float artstr_epsilon,
    float artstr_n,
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
    
    float hi = smoothing_length[i];
    double rho_i = sanitize_density(
        density[i], 0.8 * materials[material_id[i]].sph_tillotson.reference_density, 1e10);
    float p_i = pressure[i];
    float c_i = c_sound[i];
    const float Si_xx = strength_enabled ? S_xx[i] : 0.0f;
    const float Si_xy = strength_enabled ? S_xy[i] : 0.0f;
    const float Si_xz = strength_enabled ? S_xz[i] : 0.0f;
    const float Si_yy = strength_enabled ? S_yy[i] : 0.0f;
    const float Si_yz = strength_enabled ? S_yz[i] : 0.0f;
    const float Si_zz = strength_enabled ? S_zz[i] : 0.0f;

    // Load kernel correction matrix L_i (row-major 3x3).
    // When use_kernel_correction is false, L_i should be identity (set by caller).
    float L00 = L_00[i]; float L01 = L_01[i]; float L02 = L_02[i];
    float L10 = L_10[i]; float L11 = L_11[i]; float L12 = L_12[i];
    float L20 = L_20[i]; float L21 = L_21[i]; float L22 = L_22[i];

    float f_x = 0.0f, f_y = 0.0f, f_z = 0.0f;

    // Artificial stress tensor for particle i (computed once before neighbor loop)
    float Ri_xx = 0.0f, Ri_xy = 0.0f, Ri_xz = 0.0f;
    float Ri_yy = 0.0f, Ri_yz = 0.0f, Ri_zz = 0.0f;
    if (use_artificial_stress) {
        compute_artificial_stress_tensor(
            p_i,                       // scalar pressure (positive = compression)
            Si_xx, Si_xy, Si_xz,
            Si_yy, Si_yz, Si_zz,
            static_cast<float>(rho_i),
            artstr_epsilon,            // configurable epsilon (default 0.3)
            Ri_xx, Ri_xy, Ri_xz,
            Ri_yy, Ri_yz, Ri_zz
        );
    }

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float m_j = mass[j];
            double rho_j = sanitize_density(
                density[j], 0.8 * materials[material_id[j]].sph_tillotson.reference_density, 1e10);
            float p_j = sanitize_pressure(pressure[j]);
            float hj = smoothing_length[j];
            float h_ij = 0.5f * (hi + hj);

            // Re-check with h_ij since for_each_sph_neighbor filtered by kernel_cutoff_factor * hi
            if (dist >= kernel_cutoff_factor * h_ij) return;

            // Kernel gradient (vector form: dW/dr * r_ij / |r_ij|)
            float dW_dr = sph_kernel_grad(dist, h_ij);
            float inv_d = 1.0f / dist;
            float gW_x = dW_dr * dx_ij * inv_d;
            float gW_y = dW_dr * dy_ij * inv_d;
            float gW_z = dW_dr * dz_ij * inv_d;

            // Apply Bonet-Lok kernel gradient correction (1st-order consistency).
            // Must match density_rate kernel (sph_physics.cu:64-66) to keep
            // density evolution and pressure-gradient force consistent.
            if (use_kernel_correction) {
                apply_kernel_correction(gW_x, gW_y, gW_z,
                    L00, L01, L02, L10, L11, L12, L20, L21, L22);
            }

            // Pressure gradient term: m_j * ((P_i+P_j)/(rho_i*rho_j)) * grad_W
            // GRANIT geometric-density form, paired with its local pressure work.
            double press_term = static_cast<double>(p_i + p_j) / (rho_i * rho_j);

            // Deviatoric part of the same geometric total-stress form.  Keeping
            // pressure and strength on one density pairing avoids an interface
            // traction bias when rho_i != rho_j.
            const double inv_rho_pair = 1.0 / (rho_i * rho_j);
            const float Sj_xx = strength_enabled ? S_xx[j] : 0.0f;
            const float Sj_xy = strength_enabled ? S_xy[j] : 0.0f;
            const float Sj_xz = strength_enabled ? S_xz[j] : 0.0f;
            const float Sj_yy = strength_enabled ? S_yy[j] : 0.0f;
            const float Sj_yz = strength_enabled ? S_yz[j] : 0.0f;
            const float Sj_zz = strength_enabled ? S_zz[j] : 0.0f;
            const double Ssum_xx = (Si_xx + Sj_xx) * inv_rho_pair;
            const double Ssum_xy = (Si_xy + Sj_xy) * inv_rho_pair;
            const double Ssum_xz = (Si_xz + Sj_xz) * inv_rho_pair;
            const double Ssum_yy = (Si_yy + Sj_yy) * inv_rho_pair;
            const double Ssum_yz = (Si_yz + Sj_yz) * inv_rho_pair;
            const double Ssum_zz = (Si_zz + Sj_zz) * inv_rho_pair;

            // Artificial stress (Tensile Instability correction)
            // SPHSOL-style: dvdt += m_j * zeta * grad_W where zeta = (R_i + R_j) * (W/W_dp)^n
            // For pure-pressure SPH (S=0): R is diagonal with R_aa = -epsilon/rho^2 * max(-P, 0)
            if (use_artificial_stress) {
                // Compute R tensor for particle j
                float Rj_xx, Rj_xy, Rj_xz, Rj_yy, Rj_yz, Rj_zz;
                compute_artificial_stress_tensor(
                    p_j,
                    Sj_xx, Sj_xy, Sj_xz,
                    Sj_yy, Sj_yz, Sj_zz,
                    static_cast<float>(rho_j),
                    artstr_epsilon,
                    Rj_xx, Rj_xy, Rj_xz,
                    Rj_yy, Rj_yz, Rj_zz
                );

                // Sum R tensors
                float Rxx = Ri_xx + Rj_xx;
                float Rxy = Ri_xy + Rj_xy;
                float Rxz = Ri_xz + Rj_xz;
                float Ryy = Ri_yy + Rj_yy;
                float Ryz = Ri_yz + Rj_yz;
                float Rzz = Ri_zz + Rj_zz;

                // (W/W_dp)^n weighting factor
                float W_dist = sph_kernel_w(dist, h_ij);
                float W_dp = sph_kernel_w(h_ij / 1.2f, h_ij);
                float f_ratio = (W_dp > 1.0e-30f) ? (W_dist / W_dp) : 0.0f;
                float fn = powf(fabsf(f_ratio), artstr_n);

                // zeta * grad_W: contract R tensor with kernel gradient
                float art_x = Rxx * gW_x + Rxy * gW_y + Rxz * gW_z;
                float art_y = Rxy * gW_x + Ryy * gW_y + Ryz * gW_z;
                float art_z = Rxz * gW_x + Ryz * gW_y + Rzz * gW_z;

                f_x += m_j * fn * art_x;
                f_y += m_j * fn * art_y;
                f_z += m_j * fn * art_z;
            }

            // Artificial viscosity (Monaghan)
            float dist_sq = dist * dist;
            float visc_term = 0.0f;
            float vj_x = static_cast<float>(vel_x[j]);
            float vj_y = static_cast<float>(vel_y[j]);
            float vj_z = static_cast<float>(vel_z[j]);

            float dv_x = vi_x - vj_x;
            float dv_y = vi_y - vj_y;
            float dv_z = vi_z - vj_z;
            float dot_v_x = dv_x * dx_ij + dv_y * dy_ij + dv_z * dz_ij;

            if (dot_v_x < 0.0f) {
                float mu_ij = (h_ij * dot_v_x) / (dist_sq + 0.01f * h_ij * h_ij);
                float c_ij = 0.5f * (c_i + sanitize_c_sound(c_sound[j]));
                double rho_ij = 0.5 * (rho_i + rho_j);
                visc_term = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / rho_ij;
            }

            // Total acceleration (pressure + viscosity). Artificial stress (if enabled)
            // is applied directly to f_x,f_y,f_z in the block above.
            double coeff = -static_cast<double>(m_j) * (press_term + visc_term);

            f_x += coeff * gW_x;
            f_y += coeff * gW_y;
            f_z += coeff * gW_z;

            f_x += m_j * (Ssum_xx*gW_x + Ssum_xy*gW_y + Ssum_xz*gW_z);
            f_y += m_j * (Ssum_xy*gW_x + Ssum_yy*gW_y + Ssum_yz*gW_z);
            f_z += m_j * (Ssum_xz*gW_x + Ssum_yz*gW_y + Ssum_zz*gW_z);
        });
    
    // Convert acceleration to force
    float m_i = mass[i];
    force_x[i] += sanitize_d(f_x * m_i);
    force_y[i] += sanitize_d(f_y * m_i);
    force_z[i] += sanitize_d(f_z * m_i);
}

__global__ void init_sph_arrays_kernel(
    float* __restrict__ smoothing_length, double* __restrict__ density,
    float* __restrict__ pressure, float* __restrict__ viscosity,
    float* __restrict__ c_sound,
    double* __restrict__ energy, double* __restrict__ dedt,
    float* __restrict__ distension,
    float* __restrict__ L_00, float* __restrict__ L_01, float* __restrict__ L_02,
    float* __restrict__ L_10, float* __restrict__ L_11, float* __restrict__ L_12,
    float* __restrict__ L_20, float* __restrict__ L_21, float* __restrict__ L_22,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    float h0, float mu0, bool preserve_input_h,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    const MaterialRecord& material = materials[material_id[i]];
    if (!preserve_input_h) smoothing_length[i] = h0;
    pressure[i] = 0.0f;
    viscosity[i] = mu0;
    c_sound[i] = sqrtf(static_cast<float>(sph_bulk_modulus(material)) / material.sph_tillotson.reference_density);
    // ENERGY is optional in a v2 SPH input and allocate_particle_soa already
    // supplies the specified zero default.  Preserve an explicitly loaded
    // nonzero value here.
    (void)energy;
    dedt[i] = 0.0;
    distension[i] = material.sph_porosity.model == SphPorosityModel::P_ALPHA
        ? material.sph_porosity.alpha_initial : 1.0f;

    // Initialize kernel correction matrix to identity.
    // When use_kernel_correction is disabled, the force kernel reads
    // these arrays expecting identity so the correction is a no-op.
    L_00[i] = 1.0f; L_01[i] = 0.0f; L_02[i] = 0.0f;
    L_10[i] = 0.0f; L_11[i] = 1.0f; L_12[i] = 0.0f;
    L_20[i] = 0.0f; L_21[i] = 0.0f; L_22[i] = 1.0f;
}

// =============================================================================
// Host Wrappers
// =============================================================================

void initialize_sph_properties(SimulationState& state, const SimConfig& config) {
    if (config.physics != SimulationPhysics::SPH) return;
    
    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;
    
    float h0 = config.sph.smoothing_length;
    float mu0 = config.sph.viscosity;
    
    LAUNCH_KERNEL(init_sph_arrays_kernel, nb, bs, 0, 0,
        state.particles.smoothing_length, state.particles.density,
        state.particles.pressure, state.particles.viscosity,
        state.particles.c_sound,
        state.particles.energy, state.particles.dedt,
        state.particles.distension,
        state.particles.L_00, state.particles.L_01, state.particles.L_02,
        state.particles.L_10, state.particles.L_11, state.particles.L_12,
        state.particles.L_20, state.particles.L_21, state.particles.L_22,
        state.particles.material_id, state.material_database.materials,
        h0, mu0, state.particles.input_has_smoothing_length,
        num
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
void compute_sph_density(SimulationState& state, const SimConfig& config, cudaStream_t stream, int32_t* debug_neighbor_count) {
    if (config.physics != SimulationPhysics::SPH) return;
    if (config.sph.density_update_mode == 1) return; // Skip density summation if using continuous density

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;
    
    LAUNCH_KERNEL(compute_sph_density_kernel, nb, bs, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.mass, state.particles.smoothing_length,
        state.particles.density,
        state.particles.material_id, state.material_database.materials,
        state.neighbor,
        config.sph.kernel_cutoff_factor,
        config.sph.enable_shepard_filter,
        num,
        debug_neighbor_count
    );
}

void compute_sph_density_rate(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH || config.sph.density_update_mode != 1) return;

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(compute_sph_density_rate_kernel, nb, bs, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
        state.particles.mass, state.particles.density,
        state.particles.smoothing_length,
        state.particles.material_id, state.material_database.materials,
        state.particles.rho_rate,
        state.particles.L_00, state.particles.L_01, state.particles.L_02,
        state.particles.L_10, state.particles.L_11, state.particles.L_12,
        state.particles.L_20, state.particles.L_21, state.particles.L_22,
        state.neighbor,
        config.sph.kernel_cutoff_factor,
        config.sph.use_kernel_correction,
        num
    );
}

void integrate_sph_density(SimulationState& state, const SimConfig& config, float dt, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    if (config.sph.density_update_mode != 1) return; // Only continuous density mode integrates

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(integrate_sph_density_kernel, nb, bs, 0, stream,
        state.particles.density,
        state.particles.rho_rate,
        state.particles.material_id, state.material_database.materials,
        dt,
        num
    );
}

void compute_sph_pressure(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;
    
    LAUNCH_KERNEL(compute_sph_pressure_kernel, nb, bs, 0, stream,
        state.particles.density,
        state.particles.energy,
        state.particles.damage,
        state.particles.distension,
        state.particles.pressure, state.particles.c_sound,
        state.particles.material_id, state.material_database.materials,
        sph_current_stage(config, state.current_time).elastic_eos,
        num
    );
}

void compute_sph_forces(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;
    
    LAUNCH_KERNEL(compute_sph_forces_kernel, nb, bs, 0, stream,
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
        state.particles.mass, state.particles.density,
        state.particles.pressure, state.particles.c_sound,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz,
        state.particles.smoothing_length,
        state.particles.material_id, state.material_database.materials,
        state.particles.force_x, state.particles.force_y, state.particles.force_z,
        state.particles.L_00, state.particles.L_01, state.particles.L_02,
        state.particles.L_10, state.particles.L_11, state.particles.L_12,
        state.particles.L_20, state.particles.L_21, state.particles.L_22,
        state.neighbor,
        config.sph.kernel_cutoff_factor,
        config.sph.use_kernel_correction,
        config.sph.viscosity,
        2.0f * config.sph.viscosity,
        sph_current_stage(config, state.current_time).use_strength,
        config.sph.use_artificial_stress,
        config.sph.artstr_epsilon,
        config.sph.artstr_n,
        num
    );
}

// =============================================================================
// Smooth Length Update (matching SPHSOL UpdateHsml)
// =============================================================================
// kEta = 1.2 (from SPHSOL/src/global.h)
#define SPH_KETA 1.2f

__global__ void update_sph_smoothing_length_kernel(
    const float* __restrict__ mass,
    const double* __restrict__ density,
    const double* __restrict__ rho_rate,
    float* __restrict__ smoothing_length,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    int32_t density_update_mode,
    float dt,
    float hsml_min,
    float hsml_max,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float h = smoothing_length[i];
    double rho = density[i];
    // SPHSOL solver.cc:145: rho_limit = 0.8 * rho0 — uniform density floor.
    double rho_floor = 0.8 * materials[material_id[i]].sph_tillotson.reference_density;

    // Update smooth length based on density update mode
    if (density_update_mode == 1) {
        // Continuous density: h = h - dt * h * drhodt / (3 * rho)
        // SPHSOL: particles_[p].hsml -= dt_ * particles_[p].hsml * particles_[p].drhodt / (3.0 * particles_[p].rho)
        double drhodt = (rho_rate != nullptr) ? rho_rate[i] : 0.0;
        double safe_rho = sanitize_density(rho, rho_floor, 1e10);
        h -= static_cast<float>(static_cast<double>(dt) * h * drhodt / (3.0 * safe_rho));
    } else {
        // Density summation: h = kEta * pow(mass / rho, 1/3)
        // SPHSOL: particles_[p].hsml = kEta * pow(particles_[p].mass / particles_[p].rho, 1.0 / 3.0)
        double safe_rho = sanitize_density(rho, rho_floor, 1e10);
        h = SPH_KETA * static_cast<float>(pow(static_cast<double>(mass[i]) / safe_rho, 1.0 / 3.0));
    }

    // Clamp smooth length (sanitize_hsml catches NaN, fmaxf/fminf do not)
    smoothing_length[i] = sanitize_hsml(h, hsml_min, hsml_max);
}

#undef SPH_KETA

void update_sph_smoothing_length(
    SimulationState& state,
    const SimConfig& config,
    float dt,
    cudaStream_t stream
) {
    if (config.physics != SimulationPhysics::SPH) return;

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(update_sph_smoothing_length_kernel, nb, bs, 0, stream,
        state.particles.mass,
        state.particles.density,
        state.particles.rho_rate,
        state.particles.smoothing_length,
        state.particles.material_id, state.material_database.materials,
        config.sph.density_update_mode,
        dt,
        config.sph.hsml_min,
        config.sph.hsml_max,
        num
    );
}

// =============================================================================
// Global h_max update for symmetric SPH neighbor queries (matches SPHSOL grid.cc:87)
// =============================================================================
__global__ void update_sph_support_extent_kernel(
    float* __restrict__ support_extent,
    const float* __restrict__ smoothing_length,
    float kernel_cutoff_factor,
    int32_t num_particles
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    support_extent[i] = kernel_cutoff_factor * smoothing_length[i];
}

void update_sph_support_extent(
    SimulationState& state,
    const SimConfig& config,
    int32_t num_particles,
    cudaStream_t stream
) {
    if (config.physics != SimulationPhysics::SPH || num_particles <= 0) return;
    const int32_t bs = config.gpu.block_size;
    const int32_t nb = (num_particles + bs - 1) / bs;
    LAUNCH_KERNEL(update_sph_support_extent_kernel, nb, bs, 0, stream,
        state.particles.support_extent,
        state.particles.smoothing_length,
        config.sph.kernel_cutoff_factor,
        num_particles
    );
}

float compute_sph_h_max(
    const SimulationState& state,
    int32_t num_particles,
    cudaStream_t stream
) {
    if (num_particles <= 0) return 0.0f;
    thrust::device_ptr<float> h_ptr(state.particles.smoothing_length);
    return thrust::reduce(thrust::cuda::par.on(stream),
                          h_ptr, h_ptr + num_particles,
                          0.0f, thrust::maximum<float>());
}

void set_sph_h_max_global(float h_max, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyToSymbolAsync(dev_sph_h_max_global, &h_max,
                                       sizeof(float), 0,
                                       cudaMemcpyHostToDevice, stream));
}

void update_sph_h_max_global(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    set_sph_h_max_global(compute_sph_h_max(state, num, stream), stream);
}

float compute_sph_h_min(const SimulationState& state, int32_t count, cudaStream_t stream) {
    if (count <= 0) return INFINITY;
    thrust::device_ptr<float> h(state.particles.smoothing_length);
    return thrust::reduce(thrust::cuda::par.on(stream), h, h + count,
                          INFINITY, thrust::minimum<float>());
}


} // namespace cudadem

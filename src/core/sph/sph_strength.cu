#include "sph_stage.h"
#include "src/core/sph/sph_strength.h"
#include "src/core/sph/sph_device_helpers.cuh"
#include "src/core/sph/sph_kernel_correction.cuh"
#include "src/core/sph/sph_neighbor_query.cuh"
#include "src/core/sph/sph_numerical_safeguards.cuh"
#include "src/core/neighbor/neighbor_api.h"
#include "src/core/material/material_database.h"
#include "cuda_utils.h"
#include <cmath>
#include <array>
#include <stdexcept>

namespace cudadem {

namespace {
constexpr int32_t kMaxTrackedCudaDevices = 64;
struct SphStrengthWorkspace {
    float *xx = nullptr, *xy = nullptr, *xz = nullptr, *yy = nullptr, *yz = nullptr, *zz = nullptr;
    int32_t capacity = 0;
};
std::array<SphStrengthWorkspace, kMaxTrackedCudaDevices> strength_workspaces;

SphStrengthWorkspace& current_strength_workspace() {
    int32_t device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxTrackedCudaDevices)
        throw std::runtime_error("SPH strength device index is outside workspace table");
    return strength_workspaces[static_cast<size_t>(device)];
}
} // namespace

// ---------------------------------------------------------------------------
// Helper: compute Lundborg or von Mises yield strength
// ---------------------------------------------------------------------------
__device__ __forceinline__ float compute_yield_strength(
    float P,              // pressure (positive = compression)
    float D,              // damage [0,1]
    float Y0,             // von Mises cohesive limit
    bool  use_lundborg,
    float Yi0, float Yd0, float Ym, float mui, float mud,
    float& J2_factor)     // output: whether to use sqrt(J2) or sqrt(3*J2)
{
    float Y;
    if (use_lundborg) {
        J2_factor = 1.0f; // Lundborg uses sqrt(J2) in reference
        float Yi = Yi0;
        if (Ym > Yi0) {
            float denom = 1.0f + mui * P / (Ym - Yi0);
            // Protect against negative denominator in extreme tension
            if (denom > 1e-5f) {
                Yi = Yi0 + mui * P / denom;
            } else {
                Yi = 0.0f; // Material has failed in tension
            }
        } else {
            Yi = Yi0 + mui * P;
        }
        float Yd = fminf(Yd0 + mud * P, Yi);
        Y = fmaxf(Yi * (1.0f - D) + Yd * D, 0.0f);
    } else {
        J2_factor = 3.0f; // von Mises uses sqrt(3*J2) in reference
        Y = fmaxf(Y0 * (1.0f - D), 0.0f);
    }
    return Y;
}

__device__ __forceinline__ void radial_return_sph_stress(
    float& xx, float& xy, float& xz, float& yy, float& yz, float& zz,
    float pressure, float damage, const SphStrengthProperties& strength) {
    const float J2 = 0.5f * (xx * xx + yy * yy + zz * zz +
        2.0f * (xy * xy + xz * xz + yz * yz));
    float factor = 3.0f;
    const float yield = compute_yield_strength(
        pressure, damage, strength.yield_strength_Y0,
        strength.model == SphStrengthModel::LUNDBORG,
        strength.Yi0, strength.Yd0, strength.Ym, strength.mui, strength.mud,
        factor);
    const float norm = sqrtf(factor * J2);
    if (norm > yield && norm > 1.0e-30f) {
        const float scale = yield / norm;
        xx *= scale; xy *= scale; xz *= scale;
        yy *= scale; yz *= scale; zz *= scale;
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute a trial deviatoric stress (Jaumann) and seed de/dt with its
// time-centred work.  The pressure-dependent return mapping deliberately runs
// later, after the EOS has closed the new rho/e/alpha state.
// ---------------------------------------------------------------------------
__global__ void predict_sph_strength_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const double* __restrict__ vel_x, const double* __restrict__ vel_y, const double* __restrict__ vel_z,
    const float*  __restrict__ mass,
    const double* __restrict__ density,
    const float*  __restrict__ smoothing_length,
    const float*  __restrict__ distension,
    const float*  __restrict__ pressure,
    const float*  __restrict__ damage,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    // Correction matrix (identity if correction disabled)
    const float* __restrict__ L_00, const float* __restrict__ L_01, const float* __restrict__ L_02,
    const float* __restrict__ L_10, const float* __restrict__ L_11, const float* __restrict__ L_12,
    const float* __restrict__ L_20, const float* __restrict__ L_21, const float* __restrict__ L_22,
    // Deviatoric stress (in/out) and energy-rate seed
    const float* __restrict__ S_xx, const float* __restrict__ S_xy, const float* __restrict__ S_xz,
    const float* __restrict__ S_yy, const float* __restrict__ S_yz, const float* __restrict__ S_zz,
    float* __restrict__ trial_xx, float* __restrict__ trial_xy, float* __restrict__ trial_xz,
    float* __restrict__ trial_yy, float* __restrict__ trial_yz, float* __restrict__ trial_zz,
    double* __restrict__ dedt,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    bool  use_kernel_correction,
    float dt,
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
    double rho_i = sanitize_density(
        density[i], 0.8 * materials[material_id[i]].sph_tillotson.reference_density, 1e10);
    const SphStrengthProperties& strength = materials[material_id[i]].sph_strength;
    const float strength_alpha = materials[material_id[i]].sph_porosity.model ==
        SphPorosityModel::P_ALPHA ? fmaxf(distension[i], 1.0f) : 1.0f;
    const float G_shear = strength.model == SphStrengthModel::NONE ? 0.0f :
        strength.shear_modulus / strength_alpha;

    if (strength.model == SphStrengthModel::NONE) {
        trial_xx[i] = 0.0f; trial_xy[i] = 0.0f; trial_xz[i] = 0.0f;
        trial_yy[i] = 0.0f; trial_yz[i] = 0.0f; trial_zz[i] = 0.0f;
        dedt[i] = 0.0;
        return;
    }

    // Velocity gradient tensor: L_ab = dv_a/dx_b
    // L_ab += sum_j m_j/rho_j * (v_j - v_i)_a * dW/dx_b
    float L00=0,L01=0,L02=0, L10=0,L11=0,L12=0, L20=0,L21=0,L22=0;
    float Si_xx = S_xx[i]; float Si_xy = S_xy[i]; float Si_xz = S_xz[i];
    float Si_yy = S_yy[i]; float Si_yz = S_yz[i]; float Si_zz = S_zz[i];

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float hj   = smoothing_length[j];
            float h_ij = 0.5f * (hi + hj);

            // Re-check with h_ij since for_each_sph_neighbor filtered by kernel_cutoff_factor * hi
            if (dist >= kernel_cutoff_factor * h_ij) return;

            float m_j   = mass[j];
            // SPHSOL solver.cc:145: rho_limit = 0.8 * rho0 — uniform density floor.
            double rho_j = sanitize_density(
                density[j], 0.8 * materials[material_id[j]].sph_tillotson.reference_density, 1e10);
            float dW_dr = sph_kernel_grad(dist, h_ij);

            // Kernel gradient vector (i -> j direction)
            float inv_dist = 1.0f / dist;
            float gW_x = dW_dr * dx_ij * inv_dist;
            float gW_y = dW_dr * dy_ij * inv_dist;
            float gW_z = dW_dr * dz_ij * inv_dist;

            // Apply kernel correction if enabled
            if (use_kernel_correction) {
                apply_kernel_correction(gW_x, gW_y, gW_z,
                    L_00[i], L_01[i], L_02[i],
                    L_10[i], L_11[i], L_12[i],
                    L_20[i], L_21[i], L_22[i]);
            }

            // Velocity difference: dv_a = v_j_a - v_i_a
            float dvx = static_cast<float>(vel_x[j]) - vi_x;
            float dvy = static_cast<float>(vel_y[j]) - vi_y;
            float dvz = static_cast<float>(vel_z[j]) - vi_z;

            double vol_j = static_cast<double>(m_j) / rho_j;

            // Velocity gradient L_ab = sum_j vol_j * dv_a * gW_b
            L00 += vol_j * dvx * gW_x;  L01 += vol_j * dvx * gW_y;  L02 += vol_j * dvx * gW_z;
            L10 += vol_j * dvy * gW_x;  L11 += vol_j * dvy * gW_y;  L12 += vol_j * dvy * gW_z;
            L20 += vol_j * dvz * gW_x;  L21 += vol_j * dvz * gW_y;  L22 += vol_j * dvz * gW_z;

        });

    // -----------------------------------------------------------------------
    // Jaumann objective stress rate:
    // dS_ab/dt = 2G * eps_dot_ab + S_ac*omega_cb + omega_ac*S_cb
    //   strain rate: eps_dot_ab = 0.5*(L_ab + L_ba) - (1/3)*delta_ab*tr(L)
    //   rotation rate: omega_ab = 0.5*(L_ab - L_ba)
    // -----------------------------------------------------------------------
    float tr_L = L00 + L11 + L22;

    float eps_xx = L00 - (1.0f/3.0f)*tr_L;
    float eps_yy = L11 - (1.0f/3.0f)*tr_L;
    float eps_zz = L22 - (1.0f/3.0f)*tr_L;
    float eps_xy = 0.5f * (L01 + L10);
    float eps_xz = 0.5f * (L02 + L20);
    float eps_yz = 0.5f * (L12 + L21);

    float omega_xy = 0.5f * (L01 - L10);
    float omega_xz = 0.5f * (L02 - L20);
    float omega_yz = 0.5f * (L12 - L21);

    // Jaumann rotation correction for S: dS = S*W^T + W*S
    float dS_rot_xx = 2.0f*(Si_xy*omega_xy + Si_xz*omega_xz);   // simplified for symmetric S
    float dS_rot_yy = -2.0f*(Si_xy*omega_xy) + 2.0f*(Si_yz*omega_yz);
    float dS_rot_zz = -2.0f*(Si_xz*omega_xz) - 2.0f*(Si_yz*omega_yz);
    float dS_rot_xy = (Si_yy - Si_xx)*omega_xy + Si_xz*omega_yz + Si_yz*omega_xz;
    float dS_rot_xz = (Si_zz - Si_xx)*omega_xz + Si_yz*omega_xy - Si_xy*omega_yz;
    float dS_rot_yz = (Si_zz - Si_yy)*omega_yz - Si_xz*omega_xy - Si_xy*omega_xz;

    // Update stress tensor
    float two_G = 2.0f * G_shear;
    float new_Sxx = Si_xx + dt * (two_G * eps_xx + dS_rot_xx);
    float new_Sxy = Si_xy + dt * (two_G * eps_xy + dS_rot_xy);
    float new_Sxz = Si_xz + dt * (two_G * eps_xz + dS_rot_xz);
    float new_Syy = Si_yy + dt * (two_G * eps_yy + dS_rot_yy);
    float new_Syz = Si_yz + dt * (two_G * eps_yz + dS_rot_yz);
    float new_Szz = Si_zz + dt * (two_G * eps_zz + dS_rot_zz);

    // Preserve the established GRANIT/SPHSOL midpoint-work convention: its
    // endpoint is the trial stress returned with the old pressure and damage.
    // Keep the unreturned trial separately for the later final-state return
    // after rho/e/alpha/EOS are committed.
    float work_Sxx = new_Sxx, work_Sxy = new_Sxy, work_Sxz = new_Sxz;
    float work_Syy = new_Syy, work_Syz = new_Syz, work_Szz = new_Szz;
    radial_return_sph_stress(work_Sxx, work_Sxy, work_Sxz,
        work_Syy, work_Syz, work_Szz, pressure[i], damage[i], strength);
    const float mid_Sxx = 0.5f * (Si_xx + work_Sxx);
    const float mid_Sxy = 0.5f * (Si_xy + work_Sxy);
    const float mid_Sxz = 0.5f * (Si_xz + work_Sxz);
    const float mid_Syy = 0.5f * (Si_yy + work_Syy);
    const float mid_Syz = 0.5f * (Si_yz + work_Syz);
    const float mid_Szz = 0.5f * (Si_zz + work_Szz);
    const double shear_power = (
        static_cast<double>(mid_Sxx) * eps_xx +
        static_cast<double>(mid_Syy) * eps_yy +
        static_cast<double>(mid_Szz) * eps_zz +
        2.0 * (static_cast<double>(mid_Sxy) * eps_xy +
               static_cast<double>(mid_Sxz) * eps_xz +
               static_cast<double>(mid_Syz) * eps_yz)) /
        static_cast<double>(rho_i);
    dedt[i] = sanitize_d(shear_power);

    trial_xx[i] = sanitize_f(new_Sxx);  trial_xy[i] = sanitize_f(new_Sxy);  trial_xz[i] = sanitize_f(new_Sxz);
    trial_yy[i] = sanitize_f(new_Syy);  trial_yz[i] = sanitize_f(new_Syz);  trial_zz[i] = sanitize_f(new_Szz);

}

__global__ void apply_sph_strength_radial_return_kernel(
    const float* __restrict__ trial_xx, const float* __restrict__ trial_xy, const float* __restrict__ trial_xz,
    const float* __restrict__ trial_yy, const float* __restrict__ trial_yz, const float* __restrict__ trial_zz,
    const float* __restrict__ pressure, const float* __restrict__ damage,
    const int32_t* __restrict__ material_id, const MaterialRecord* __restrict__ materials,
    float* __restrict__ S_xx, float* __restrict__ S_xy, float* __restrict__ S_xz,
    float* __restrict__ S_yy, float* __restrict__ S_yz, float* __restrict__ S_zz,
    int32_t num_particles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const SphStrengthProperties& strength = materials[material_id[i]].sph_strength;
    if (strength.model == SphStrengthModel::NONE) {
        S_xx[i]=S_xy[i]=S_xz[i]=S_yy[i]=S_yz[i]=S_zz[i]=0.0f;
        return;
    }
    float xx=trial_xx[i], xy=trial_xy[i], xz=trial_xz[i], yy=trial_yy[i], yz=trial_yz[i], zz=trial_zz[i];
    radial_return_sph_stress(xx, xy, xz, yy, yz, zz,
        pressure[i], damage[i], strength);
    S_xx[i]=sanitize_f(xx); S_xy[i]=sanitize_f(xy); S_xz[i]=sanitize_f(xz);
    S_yy[i]=sanitize_f(yy); S_yz[i]=sanitize_f(yz); S_zz[i]=sanitize_f(zz);
}

// ---------------------------------------------------------------------------
// Kernel: build CSPH correction matrix (needed when use_kernel_correction)
// ---------------------------------------------------------------------------
__global__ void compute_csph_correction_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float*  __restrict__ mass,
    const double* __restrict__ density,
    const float*  __restrict__ smoothing_length,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    float* __restrict__ L_00, float* __restrict__ L_01, float* __restrict__ L_02,
    float* __restrict__ L_10, float* __restrict__ L_11, float* __restrict__ L_12,
    float* __restrict__ L_20, float* __restrict__ L_21, float* __restrict__ L_22,
    const NeighborData neighbor,
    float kernel_cutoff_factor,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float pi_x = static_cast<float>(pos_x[i]);
    float pi_y = static_cast<float>(pos_y[i]);
    float pi_z = static_cast<float>(pos_z[i]);
    float hi = smoothing_length[i];

    // Build B matrix: B_ab = sum_j vol_j * (-r_ij_a) * gW_b (tensor product of relative position and kernel gradient)
    float B[9] = {0};
    // SPHSOL completeness metric: sum_j vol_j * W_ij  (equations.cc:412)
    float completeness = 0.0f;

    for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
        pos_x, pos_y, pos_z, smoothing_length,
        kernel_cutoff_factor,
        [&](int32_t j, float dx_ij, float dy_ij, float dz_ij, float dist) {
            float hj   = smoothing_length[j];
            float h_ij = 0.5f * (hi + hj);

            // Re-check with h_ij since for_each_sph_neighbor filtered by kernel_cutoff_factor * hi
            if (dist >= kernel_cutoff_factor * h_ij) return;

            // SPHSOL solver.cc:145: rho_limit = 0.8 * rho0 — uniform density floor.
            double safe_rho_j = sanitize_density(
                density[j], 0.8 * materials[material_id[j]].sph_tillotson.reference_density, 1e10);
            double vol_j = static_cast<double>(mass[j]) / safe_rho_j;
            float dW_dr = sph_kernel_grad(dist, h_ij);
            float inv_d = 1.0f / dist;
            float gW_x = dW_dr * dx_ij * inv_d;
            float gW_y = dW_dr * dy_ij * inv_d;
            float gW_z = dW_dr * dz_ij * inv_d;

            // B_ab += vol_j * r_ji_a * gW_b  (r_ji = -r_ij)
            float rx = -dx_ij, ry = -dy_ij, rz = -dz_ij;
            B[0] += vol_j * rx * gW_x;  B[1] += vol_j * rx * gW_y;  B[2] += vol_j * rx * gW_z;
            B[3] += vol_j * ry * gW_x;  B[4] += vol_j * ry * gW_y;  B[5] += vol_j * ry * gW_z;
            B[6] += vol_j * rz * gW_x;  B[7] += vol_j * rz * gW_y;  B[8] += vol_j * rz * gW_z;

            // completeness = sum_j vol_j * W_ij  (SPHSOL equations.cc:412)
            float W_ij = sph_kernel_w(dist, h_ij);
            completeness += vol_j * W_ij;
        });

    // Invert B -> L (Cramer's rule, no blend — blend is applied afterwards)
    float Linv[9];
    if (!mat3_inverse(B, Linv)) {
        // Singular: fall back to identity
        Linv[0]=1; Linv[1]=0; Linv[2]=0;
        Linv[3]=0; Linv[4]=1; Linv[5]=0;
        Linv[6]=0; Linv[7]=0; Linv[8]=1;
    }

    // Completeness blend: matches SPHSOL equations.cc:421-432
    // blend = clamp((completeness - 0.6) * 2.5, 0, 1)
    // L = blend * Linv + (1 - blend) * I
    float blended_comp = fmaxf(0.0f, fminf(1.0f, (completeness - 0.6f) * 2.5f));

    L_00[i] = blended_comp * Linv[0] + (1.0f - blended_comp) * 1.0f;
    L_01[i] = blended_comp * Linv[1];
    L_02[i] = blended_comp * Linv[2];
    L_10[i] = blended_comp * Linv[3];
    L_11[i] = blended_comp * Linv[4] + (1.0f - blended_comp) * 1.0f;
    L_12[i] = blended_comp * Linv[5];
    L_20[i] = blended_comp * Linv[6];
    L_21[i] = blended_comp * Linv[7];
    L_22[i] = blended_comp * Linv[8] + (1.0f - blended_comp) * 1.0f;
}

// ---------------------------------------------------------------------------
// Kernel: initialize deviatoric stress and distension
// ---------------------------------------------------------------------------
__global__ void init_strength_arrays_kernel(
    float* __restrict__ S_xx, float* __restrict__ S_xy, float* __restrict__ S_xz,
    float* __restrict__ S_yy, float* __restrict__ S_yz, float* __restrict__ S_zz,
    float* __restrict__ distension,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    int32_t num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    S_xx[i] = 0.0f; S_xy[i] = 0.0f; S_xz[i] = 0.0f;
    S_yy[i] = 0.0f; S_yz[i] = 0.0f; S_zz[i] = 0.0f;
    const SphPorosityProperties& porosity = materials[material_id[i]].sph_porosity;
    distension[i] = porosity.model == SphPorosityModel::P_ALPHA ? porosity.alpha_initial : 1.0f;
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void initialize_sph_strength(SimulationState& state, const SimConfig& config) {
    if (config.physics != SimulationPhysics::SPH) return;

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(init_strength_arrays_kernel, nb, bs, 0, 0,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz,
        state.particles.distension,
        state.particles.material_id, state.material_database.materials,
        num
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void update_sph_kernel_correction(SimulationState& state, const SimConfig& config,
                                  cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;

    int32_t num = state.particles.num_particles;
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    if (config.sph.use_kernel_correction) {
        LAUNCH_KERNEL(compute_csph_correction_kernel, nb, bs, 0, stream,
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.mass, state.particles.density, state.particles.smoothing_length,
            state.particles.material_id, state.material_database.materials,
            state.particles.L_00, state.particles.L_01, state.particles.L_02,
            state.particles.L_10, state.particles.L_11, state.particles.L_12,
            state.particles.L_20, state.particles.L_21, state.particles.L_22,
            state.neighbor,
            config.sph.kernel_cutoff_factor,
            num
        );
    }
}

void initialize_sph_strength_workspace(int32_t max_particles) {
    if (max_particles <= 0) return;
    SphStrengthWorkspace& w = current_strength_workspace();
    if (max_particles <= w.capacity) return;
    if (w.xx) CUDA_CHECK(cudaFree(w.xx)); if (w.xy) CUDA_CHECK(cudaFree(w.xy));
    if (w.xz) CUDA_CHECK(cudaFree(w.xz)); if (w.yy) CUDA_CHECK(cudaFree(w.yy));
    if (w.yz) CUDA_CHECK(cudaFree(w.yz)); if (w.zz) CUDA_CHECK(cudaFree(w.zz));
    const size_t bytes = static_cast<size_t>(max_particles) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&w.xx, bytes)); CUDA_CHECK(cudaMalloc(&w.xy, bytes));
    CUDA_CHECK(cudaMalloc(&w.xz, bytes)); CUDA_CHECK(cudaMalloc(&w.yy, bytes));
    CUDA_CHECK(cudaMalloc(&w.yz, bytes)); CUDA_CHECK(cudaMalloc(&w.zz, bytes));
    w.capacity = max_particles;
}

void shutdown_sph_strength_workspace() {
    SphStrengthWorkspace& w = current_strength_workspace();
    if (w.xx) cudaFree(w.xx); if (w.xy) cudaFree(w.xy); if (w.xz) cudaFree(w.xz);
    if (w.yy) cudaFree(w.yy); if (w.yz) cudaFree(w.yz); if (w.zz) cudaFree(w.zz);
    w = SphStrengthWorkspace{};
}

void predict_sph_strength_and_work(SimulationState& state, const SimConfig& config,
                                   float stress_dt, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH || !sph_current_stage(config, state.current_time).use_strength) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    initialize_sph_strength_workspace(num);
    const SphStrengthWorkspace& w = current_strength_workspace();
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(predict_sph_strength_kernel, nb, bs, 0, stream,
        state.particles.pos_x,   state.particles.pos_y,   state.particles.pos_z,
        state.particles.vel_x,   state.particles.vel_y,   state.particles.vel_z,
        state.particles.mass,
        state.particles.density,
        state.particles.smoothing_length,
        state.particles.distension,
        state.particles.pressure,
        state.particles.damage,
        state.particles.material_id, state.material_database.materials,
        state.particles.L_00, state.particles.L_01, state.particles.L_02,
        state.particles.L_10, state.particles.L_11, state.particles.L_12,
        state.particles.L_20, state.particles.L_21, state.particles.L_22,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz,
        w.xx, w.xy, w.xz, w.yy, w.yz, w.zz,
        state.particles.dedt,
        state.neighbor,
        config.sph.kernel_cutoff_factor,
        config.sph.use_kernel_correction,
        stress_dt,
        num
    );
}

void apply_sph_strength_radial_return(SimulationState& state, const SimConfig& config,
                                      cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH || !sph_current_stage(config, state.current_time).use_strength) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    initialize_sph_strength_workspace(num);
    const SphStrengthWorkspace& w = current_strength_workspace();
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(apply_sph_strength_radial_return_kernel, nb, bs, 0, stream,
        w.xx, w.xy, w.xz, w.yy, w.yz, w.zz,
        state.particles.pressure, state.particles.damage,
        state.particles.material_id, state.material_database.materials,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz, num);
}

} // namespace cudadem

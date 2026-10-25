#include "sph_eos_math.cuh"
#include "sph_stage.h"
// =============================================================================
// SPH CFL Timestep Implementation
// =============================================================================

#include "sph_timestep.h"
#include "sph_damage_math.cuh"
#include "sph_porosity.h"
#include "sph_timestep_math.cuh"
#include "sph_numerical_safeguards.cuh"
#include "src/core/material/material_database.h"
#include "cuda_utils.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <array>
#include <stdexcept>

namespace cudadem {

namespace {

constexpr int32_t kMaxTrackedCudaDevices = 64;

struct SPHTimestepWorkspace {
    float* particle_dt = nullptr;
    int32_t* limiting_idx = nullptr;
    SPHTimestepDiagnostic* diagnostics = nullptr;
    float* diagnostic_value = nullptr;
    int32_t capacity = 0;
    int32_t dt_min_hit_count = 0;
};

std::array<SPHTimestepWorkspace, kMaxTrackedCudaDevices> timestep_workspaces;

SPHTimestepWorkspace& current_timestep_workspace() {
    int32_t device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxTrackedCudaDevices) {
        throw std::runtime_error("SPH timestep device index is outside workspace table");
    }
    return timestep_workspaces[static_cast<size_t>(device)];
}

} // namespace

// Reduction functor for finding minimum dt and its index
struct MinDtWithIndex {
    __device__ thrust::tuple<float, int32_t> operator()(
        const thrust::tuple<float, int32_t>& a,
        const thrust::tuple<float, int32_t>& b
    ) {
        float dt_a = thrust::get<0>(a);
        float dt_b = thrust::get<0>(b);
        if (dt_a <= dt_b) return a;
        return b;
    }
};

// Kernel to compute per-particle CFL timestep (full SPHSOL constraints)
// Matches SPHSOL solver.cc:938-1021 for RELAX (dt1+dt4) and SHOCK (dt1+dt3+dt4) modes.
__global__ void compute_sph_dt_per_particle_kernel(
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const float* __restrict__ smoothing_length,
    const float* __restrict__ c_sound,
    const double* __restrict__ density,
    const double* __restrict__ rho_rate,
    const double* __restrict__ energy,
    const double* __restrict__ dedt,
    const float* __restrict__ mass,
    const float* __restrict__ pressure,
    const float* __restrict__ damage,
    const float* __restrict__ distension,
    const float* __restrict__ dalpha_dt,
    const float* __restrict__ dpdt,
    const float* __restrict__ S_xx,
    const float* __restrict__ S_xy,
    const float* __restrict__ S_xz,
    const float* __restrict__ S_yy,
    const float* __restrict__ S_yz,
    const float* __restrict__ S_zz,
    const int32_t* __restrict__ flaws_ntot,
    const float* __restrict__ flaws_strain_min,
    const float* __restrict__ flaws_strain_max,
    const float* __restrict__ flaws_m0,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    float* __restrict__ particle_dt,
    int32_t* __restrict__ limiting_idx,
    SPHTimestepDiagnostic* __restrict__ diagnostics,
    float cfl_number,
    bool is_relax_mode,
    bool damage_enabled,
    bool strength_enabled,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float h = sanitize_hsml(smoothing_length[i], 1e-10f, 1e10f);
    const MaterialRecord& material = materials[material_id[i]];
    float cs = sanitize_c_sound(c_sound[i]);
    float vx = static_cast<float>(vel_x[i]);
    float vy = static_cast<float>(vel_y[i]);
    float vz = static_cast<float>(vel_z[i]);
    float v_mag = sqrtf(vx*vx + vy*vy + vz*vz);

    // The signal-speed constraint is a user-facing CFL control.  The other
    // rate constraints deliberately retain SPHSOL's 0.7 coefficient.
    float dt1_coef = cfl_number;
    float coef2 = 0.7f;  // SPHSOL dt3, dt4, and dt5 coefficient

    // dt3 = coef2 * (rho + 0.1*rho0) / |drhodt|
    // Only used in SHOCK mode (RELAX mode excludes dt3 per solver.cc:1010)
    // SPHSOL solver.cc:145: rho_limit = 0.8 * rho0 — uniform density floor
    // prevents low-density leakage into CFL timestep (causes dt→0 or dt→∞).
    float rho0 = material.sph_tillotson.reference_density;
    double rho = sanitize_density(density[i], 0.8 * rho0);

    // Elastic longitudinal signal speed.  SPHSOL's EOS sound speed contains
    // the volumetric response; strength adds 4G/(3 rho).  Porous material uses
    // the same alpha_inv scaling as its elastic stress-rate convention.
    const float strength_alpha = material.sph_porosity.model == SphPorosityModel::P_ALPHA
        ? fmaxf(distension[i], 1.0f) : 1.0f;
    const float c_long = sanitize_c_sound(sph_longitudinal_signal_speed(
        cs, static_cast<float>(rho), material.sph_strength.shear_modulus, strength_alpha,
        strength_enabled && material.sph_strength.model != SphStrengthModel::NONE), cs);

    // dt1 = coef * h / (c_long + |v|)
    float dt1 = dt1_coef * h / (c_long + v_mag + 1.0e-10f);

    float dt3 = FLT_MAX;
    if (rho_rate != nullptr) {
        double drhodt = fabs(rho_rate[i]);
        if (drhodt > 1.0e-9) {
            dt3 = static_cast<float>(coef2 * (rho + 0.1 * rho0) / drhodt);
        }
    }

    // dt4 = coef2 * (|e| + 0.1*c²) / |dedt|
    // Used in both RELAX and SHOCK modes
    float dt4 = FLT_MAX;
    if (dedt != nullptr) {
        float dedt_val = fabsf(static_cast<float>(dedt[i]));
        if (dedt_val > 1.0e-9f) {
            float e_abs = fabsf(static_cast<float>(energy[i]));
            dt4 = coef2 * (e_abs + 0.1f * cs * cs) / dedt_val;
        }
    }

    // dt5 = 0.7 * (D^(1/3) + 0.1) / dD^(1/3)/dt.  Re-evaluate the
    // instantaneous rate from the closed state so qdot remains ephemeral and
    // cannot become stale across restart, migration, or constitutive updates.
    float dt5 = FLT_MAX;
    if (damage_enabled && material.sph_damage.model == SphDamageModel::GRADY_KIPP) {
        const float lambda = sph_damage_lambda_max(
            S_xx[i] - pressure[i], S_xy[i], S_xz[i],
            S_yy[i] - pressure[i], S_yz[i], S_zz[i] - pressure[i]);
        const float smin = flaws_strain_min[i];
        const SphDamageRate rate = sph_damage_rate(
            lambda, damage[i], static_cast<float>(sph_bulk_modulus(material)),
            material.sph_strength.shear_modulus, material.sph_tillotson.reference_density,
            material.sph_porosity.model == SphPorosityModel::P_ALPHA
                ? fmaxf(distension[i], 1.0f) : 1.0f,
            smoothing_length[i], flaws_ntot[i], smin,
            fmaxf(smin, material.sph_damage.min_strain),
            flaws_strain_max[i], flaws_m0[i],
            material.sph_damage.crack_speed_fraction,
        material.sph_eos.elastic_modulus, material.sph_eos.continuous_activation);
        if (rate.qdot > 1.0e-9f && isfinite(rate.qdot)) {
            const float q = cbrtf(fminf(1.0f, fmaxf(0.0f, damage[i])));
            dt5 = coef2 * (q + 0.1f) / rate.qdot;
        }
    }

    // SPHSOL SHOCK constraints for the P-alpha crush curve.  Rates live in
    // ephemeral workspace, so they are always from this closed state.
    float dt6 = FLT_MAX, dt7 = FLT_MAX;
    if (material.sph_porosity.model == SphPorosityModel::P_ALPHA &&
        distension[i] > 1.0f + 1.0e-9f) {
        const float alpha_dot = dalpha_dt ? dalpha_dt[i] : 0.0f;
        if (alpha_dot < -1.0e-9f && isfinite(alpha_dot))
            dt6 = -coef2 * distension[i] / alpha_dot;
        const float pressure_dot = dpdt ? dpdt[i] : 0.0f;
        if (fabsf(pressure_dot) > 1.0e-9f && isfinite(pressure_dot) &&
            material.sph_porosity.P_solid_crush > 0.0f &&
            isfinite(material.sph_porosity.P_solid_crush))
            dt7 = 0.1f * material.sph_porosity.P_solid_crush / fabsf(pressure_dot);
    }

    // SPHSOL solver.cc:1010 (RELAX): dt = min(dt1, dt4)
    // SPHSOL solver.cc:948-987 (SHOCK): dt = min(dt1, dt3, dt4, dt5, dt6, dt7)
    float dt;
    SPHTimestepConstraint constraint = SPHTimestepConstraint::ELASTIC_CFL;
    if (is_relax_mode) {
        dt = dt1;
        if (dt4 < dt) { dt = dt4; constraint = SPHTimestepConstraint::ENERGY_RATE; }
    } else {
        dt = dt1;
        if (dt3 < dt) { dt = dt3; constraint = SPHTimestepConstraint::DENSITY_RATE; }
        if (dt4 < dt) { dt = dt4; constraint = SPHTimestepConstraint::ENERGY_RATE; }
        if (dt5 < dt) { dt = dt5; constraint = SPHTimestepConstraint::DAMAGE_RATE; }
        if (dt6 < dt) { dt = dt6; constraint = SPHTimestepConstraint::POROSITY_ALPHA_RATE; }
        if (dt7 < dt) { dt = dt7; constraint = SPHTimestepConstraint::POROSITY_PRESSURE_RATE; }
    }

    particle_dt[i] = sanitize_f(dt, 1.0e-15f);
    limiting_idx[i] = i;
    diagnostics[i] = SPHTimestepDiagnostic{
        dt1, dt3, dt4, dt5, dt6, dt7, c_long, v_mag, constraint};
}

// Extract compact diagnostics into float arrays for independent max reductions.
__global__ void extract_timestep_diagnostic_values_kernel(
    const SPHTimestepDiagnostic* __restrict__ diagnostics,
    float* __restrict__ max_speed_out,
    float* __restrict__ max_c_sound_out,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    max_speed_out[i] = diagnostics[i].particle_speed;
    max_c_sound_out[i] = diagnostics[i].longitudinal_sound_speed;
}

// Initialize timestep module
void initialize_sph_timestep(int32_t max_particles) {
    if (max_particles <= 0) return;
    SPHTimestepWorkspace& workspace = current_timestep_workspace();
    if (max_particles <= workspace.capacity) return;

    if (workspace.particle_dt) CUDA_CHECK(cudaFree(workspace.particle_dt));
    if (workspace.limiting_idx) CUDA_CHECK(cudaFree(workspace.limiting_idx));
    if (workspace.diagnostics) CUDA_CHECK(cudaFree(workspace.diagnostics));
    if (workspace.diagnostic_value) CUDA_CHECK(cudaFree(workspace.diagnostic_value));
    CUDA_CHECK(cudaMalloc(&workspace.particle_dt,
                          static_cast<size_t>(max_particles) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace.limiting_idx,
                          static_cast<size_t>(max_particles) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&workspace.diagnostics,
                          static_cast<size_t>(max_particles) * sizeof(SPHTimestepDiagnostic)));
    CUDA_CHECK(cudaMalloc(&workspace.diagnostic_value,
                          static_cast<size_t>(max_particles) * sizeof(float)));
    workspace.capacity = max_particles;
}

void shutdown_sph_timestep() {
    SPHTimestepWorkspace& workspace = current_timestep_workspace();
    if (workspace.particle_dt) {
        cudaFree(workspace.particle_dt);
        workspace.particle_dt = nullptr;
    }
    if (workspace.limiting_idx) {
        cudaFree(workspace.limiting_idx);
        workspace.limiting_idx = nullptr;
    }
    if (workspace.diagnostics) {
        cudaFree(workspace.diagnostics);
        workspace.diagnostics = nullptr;
    }
    if (workspace.diagnostic_value) {
        cudaFree(workspace.diagnostic_value);
        workspace.diagnostic_value = nullptr;
    }
    workspace.capacity = 0;
    workspace.dt_min_hit_count = 0;
}

// Compute SPH timestep
void compute_sph_timestep(
    SimulationState& state,
    const SimConfig& config,
    SPHTimestepResult& result,
    cudaStream_t stream
) {
    int32_t num_particles = state.particles.num_particles;
    if (num_particles <= 0) {
        result.dt = config.sph.dt_max;
        result.dt_unclamped = config.sph.dt_max;
        return;
    }

    // Ensure buffers are allocated
    initialize_sph_timestep(num_particles);
    SPHTimestepWorkspace& workspace = current_timestep_workspace();
    const SphPorosityRates porosity_rates = sph_porosity_rates_for_current_device();

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    // Compute per-particle dt (full CFL with SPHSOL constraints)
    // is_relax_mode: set to false to enable dt3 (viscous constraint) for stability.
    // SPHSOL uses SHOCK mode (all 7 constraints) for continuous density. 
    const bool is_relax_mode = sph_current_stage(config, state.current_time).mode == SphRunStage::RELAX;
    compute_sph_dt_per_particle_kernel<<<num_blocks, block_size, 0, stream>>>(
        state.particles.vel_x,
        state.particles.vel_y,
        state.particles.vel_z,
        state.particles.smoothing_length,
        state.particles.c_sound,
        state.particles.density,
        state.particles.rho_rate,
        state.particles.energy,
        state.particles.dedt,
        state.particles.mass,
        state.particles.pressure,
        state.particles.damage,
        state.particles.distension,
        porosity_rates.capacity >= num_particles ? porosity_rates.dalpha_dt : nullptr,
        porosity_rates.capacity >= num_particles ? porosity_rates.dpdt : nullptr,
        state.particles.S_xx,
        state.particles.S_xy,
        state.particles.S_xz,
        state.particles.S_yy,
        state.particles.S_yz,
        state.particles.S_zz,
        state.particles.flaws_ntot,
        state.particles.flaws_strain_min,
        state.particles.flaws_strain_max,
        state.particles.flaws_m0,
        state.particles.material_id, state.material_database.materials,
        workspace.particle_dt,
        workspace.limiting_idx,
        workspace.diagnostics,
        config.sph.cfl_number,
        is_relax_mode,
        sph_current_stage(config, state.current_time).evolve_damage,
        sph_current_stage(config, state.current_time).use_strength,
        num_particles
    );

    // Reduce to find minimum dt and its index using Thrust
    thrust::device_ptr<float> dt_ptr(workspace.particle_dt);
    thrust::device_ptr<int32_t> idx_ptr(workspace.limiting_idx);

    // Create zip iterator for (dt, idx) pairs
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(dt_ptr, idx_ptr));
    auto end = begin + num_particles;

    auto min_result = thrust::reduce(
        thrust::cuda::par.on(stream),
        begin, end,
        thrust::make_tuple(FLT_MAX, -1),
        MinDtWithIndex()
    );

    float dt_min = thrust::get<0>(min_result);
    int32_t limiting_idx = thrust::get<1>(min_result);

    // Clamp to [dt_min, dt_max]
    result.dt_unclamped = dt_min;
    result.dt = fmaxf(config.sph.dt_min, fminf(config.sph.dt_max, dt_min));
    result.limiting_particle = limiting_idx;
    if (limiting_idx >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            &result.limiting_diagnostic,
            workspace.diagnostics + limiting_idx,
            sizeof(SPHTimestepDiagnostic), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Track dt_min hits
    if (result.dt == config.sph.dt_min) {
        workspace.dt_min_hit_count++;
        if (workspace.dt_min_hit_count >= 10) {
            std::cout << "WARNING: SPH timestep hit dt_min "
                      << workspace.dt_min_hit_count << " consecutive times" << std::endl;
        }
    } else {
        workspace.dt_min_hit_count = 0;
    }
    result.dt_min_hits = workspace.dt_min_hit_count;

    extract_timestep_diagnostic_values_kernel<<<num_blocks, block_size, 0, stream>>>(
        workspace.diagnostics, workspace.particle_dt,
        workspace.diagnostic_value, num_particles);
    thrust::device_ptr<float> value_ptr(workspace.diagnostic_value);
    result.max_speed = thrust::reduce(
        thrust::cuda::par.on(stream), dt_ptr, dt_ptr + num_particles,
        0.0f, thrust::maximum<float>());
    result.max_c_sound = thrust::reduce(
        thrust::cuda::par.on(stream), value_ptr, value_ptr + num_particles,
        0.0f, thrust::maximum<float>());
}

const char* sph_timestep_constraint_name(SPHTimestepConstraint constraint) {
    switch (constraint) {
        case SPHTimestepConstraint::ELASTIC_CFL: return "elastic_cfl";
        case SPHTimestepConstraint::DENSITY_RATE: return "density_rate";
        case SPHTimestepConstraint::ENERGY_RATE: return "energy_rate";
        case SPHTimestepConstraint::DAMAGE_RATE: return "damage_rate";
        case SPHTimestepConstraint::POROSITY_ALPHA_RATE: return "porosity_alpha_rate";
        case SPHTimestepConstraint::POROSITY_PRESSURE_RATE: return "porosity_pressure_rate";
        default: return "none";
    }
}

float get_initial_sph_timestep(const SimConfig& config) {
    // Use dt_max as initial estimate, actual dt computed after first step
    return config.sph.dt_max;
}

} // namespace cudadem

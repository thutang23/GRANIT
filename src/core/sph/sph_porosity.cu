#include "sph_eos_math.cuh"
#include "sph_stage.h"
#include "src/core/sph/sph_porosity.h"
#include "src/core/sph/sph_porosity_math.cuh"
#include "src/core/material/material_database.h"
#include "cuda_utils.h"
#include <cmath>
#include <array>
#include <stdexcept>

namespace cudadem {

// ---------------------------------------------------------------------------
// Kernel: P-alpha porosity model (Jutzi 2008)
//
// The P-alpha model replaces the solid EOS with a porous one:
//   P_porous(rho, e) = P_solid(rho * alpha, e) / alpha
//
// Crush curve (polynomial, Carroll & Holt 1972):
//   For P_e < P < P_s:
//     alpha = 1 + (alpha_0 - 1) * ((P_s - P) / (P_s - P_e))^n
//   For P >= P_s: alpha = 1 (fully compacted)
//   For P <= P_e: alpha = alpha_0 (elastic distension)
//
namespace {
constexpr int32_t kMaxTrackedCudaDevices = 64;

struct SphPorosityWorkspace {
    float* dalpha_dt = nullptr;
    float* dpdt = nullptr;
    int32_t capacity = 0;
};

std::array<SphPorosityWorkspace, kMaxTrackedCudaDevices> porosity_workspaces;

SphPorosityWorkspace& current_porosity_workspace() {
    int32_t device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxTrackedCudaDevices)
        throw std::runtime_error("SPH porosity device index is outside workspace table");
    return porosity_workspaces[static_cast<size_t>(device)];
}
} // namespace

// Evaluate a P-alpha rate from the same current rho-dot and e-dot used by
// density/energy integration.  P = P_solid(alpha*rho,e)/alpha and the
// crush curve alpha(P) make the rates implicit.  sph_palpha_rate solves the
// differentiated porous-pressure form without changing persistent state.
// ---------------------------------------------------------------------------
__global__ void evaluate_sph_porosity_rates_kernel(
    const double* __restrict__ density,
    const float* __restrict__ pressure,
    const float* __restrict__ distension,
    const double* __restrict__ rho_rate,
    const double* __restrict__ energy,
    const double* __restrict__ dedt,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    float* __restrict__ dalpha_dt_out,
    float* __restrict__ dpdt_out,
    bool elastic_eos,
    int32_t num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const MaterialRecord& material = materials[material_id[i]];
    const SphPorosityProperties& porosity = material.sph_porosity;
    dalpha_dt_out[i] = 0.0f;
    dpdt_out[i] = 0.0f;
    if (porosity.model != SphPorosityModel::P_ALPHA) return;
    const SphTillotsonProperties& eos = material.sph_tillotson;
    const float alpha_0 = porosity.alpha_initial;
    const float P_e = porosity.P_elastic_limit;
    const float P_s = porosity.P_solid_crush;
    const float crush_n = porosity.crushing_exponent;
    const float alpha_i = fmaxf(distension[i], 1.0f);
    const float P_porous = pressure[i];
    if (alpha_i > 1.0f && P_porous >= sph_palpha_crush_pressure(
            alpha_i, alpha_0, P_e, P_s, crush_n) &&
        P_porous > P_e && P_porous < P_s && P_s > P_e) {
        const float frac = (P_s - P_porous) / (P_s - P_e);
        float frac_clamped = fmaxf(frac, 0.0f);
        float dalpha_dp = 0.0f;
        if (frac_clamped > 0.0f) {
            dalpha_dp = -crush_n * (alpha_0 - 1.0f) * powf(frac_clamped, crush_n - 1.0f) / (P_s - P_e);
        }

        const double rho_i = density[i];
        const double drho_dt = rho_rate ? rho_rate[i] : 0.0;
        const double de_dt = dedt ? dedt[i] : 0.0;
        const SphTillotsonSolidDerivatives solid =
            sph_solid_eos_derivatives(material, rho_i * alpha_i,
                static_cast<double>(eos.reference_density) * alpha_0,
                energy[i], elastic_eos);
        SphPorosityRate rate = sph_palpha_rate(
            rho_i, alpha_i, P_porous, drho_dt, de_dt, dalpha_dp, solid);

        // The crush curve only compacts.  A derivative which would increase
        // alpha is not a physical P-alpha rate and must not constrain dt.
        if (rate.dalpha_dt > 0.0) rate.dalpha_dt = 0.0;
        dalpha_dt_out[i] = static_cast<float>(rate.dalpha_dt);
        dpdt_out[i] = static_cast<float>(rate.dpdt);
    }
}

__global__ void integrate_sph_porosity_kernel(
    float* __restrict__ distension, const float* __restrict__ pressure,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const float* __restrict__ dalpha_dt, float dt, int32_t num_particles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const SphPorosityProperties& porosity = materials[material_id[i]].sph_porosity;
    if (porosity.model != SphPorosityModel::P_ALPHA) {
        distension[i] = 1.0f;
        return;
    }
    distension[i] = static_cast<float>(sph_palpha_commit(
        distension[i], porosity.alpha_initial, pressure[i],
        porosity.P_elastic_limit, porosity.P_solid_crush, porosity.crushing_exponent,
        dalpha_dt && isfinite(dalpha_dt[i]) ? dalpha_dt[i] : 0.0, dt));
}

void initialize_sph_porosity(int32_t max_particles) {
    if (max_particles <= 0) return;
    SphPorosityWorkspace& workspace = current_porosity_workspace();
    if (max_particles <= workspace.capacity) return;
    if (workspace.dalpha_dt) CUDA_CHECK(cudaFree(workspace.dalpha_dt));
    if (workspace.dpdt) CUDA_CHECK(cudaFree(workspace.dpdt));
    CUDA_CHECK(cudaMalloc(&workspace.dalpha_dt, static_cast<size_t>(max_particles) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace.dpdt, static_cast<size_t>(max_particles) * sizeof(float)));
    workspace.capacity = max_particles;
}

void shutdown_sph_porosity() {
    SphPorosityWorkspace& workspace = current_porosity_workspace();
    if (workspace.dalpha_dt) cudaFree(workspace.dalpha_dt);
    if (workspace.dpdt) cudaFree(workspace.dpdt);
    workspace = SphPorosityWorkspace{};
}

SphPorosityRates sph_porosity_rates_for_current_device() {
    const SphPorosityWorkspace& workspace = current_porosity_workspace();
    return {workspace.dalpha_dt, workspace.dpdt, workspace.capacity};
}

void evaluate_sph_porosity_rates(SimulationState& state, const SimConfig& config,
                                 cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;

    int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    initialize_sph_porosity(num);
    const SphPorosityWorkspace& workspace = current_porosity_workspace();
    if (!sph_current_stage(config, state.current_time).evolve_porosity) {
        CUDA_CHECK(cudaMemsetAsync(workspace.dalpha_dt, 0, num * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(workspace.dpdt, 0, num * sizeof(float), stream));
        return;
    }
    int bs = config.gpu.block_size;
    int nb = (num + bs - 1) / bs;

    LAUNCH_KERNEL(evaluate_sph_porosity_rates_kernel, nb, bs, 0, stream,
        state.particles.density,
        state.particles.pressure,
        state.particles.distension,
        state.particles.rho_rate,
        state.particles.energy,
        state.particles.dedt,
        state.particles.material_id, state.material_database.materials,
        workspace.dalpha_dt, workspace.dpdt,
        sph_current_stage(config, state.current_time).elastic_eos,
        num
    );
}

void integrate_sph_porosity(SimulationState& state, const SimConfig& config,
                            float dt, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    if (!sph_current_stage(config, state.current_time).evolve_porosity) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    initialize_sph_porosity(num);
    const SphPorosityWorkspace& workspace = current_porosity_workspace();
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(integrate_sph_porosity_kernel, nb, bs, 0, stream,
        state.particles.distension, state.particles.pressure,
        state.particles.material_id, state.material_database.materials,
        workspace.dalpha_dt, dt, num);
}

} // namespace cudadem

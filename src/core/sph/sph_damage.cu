#include "sph_eos_math.cuh"
#include "sph_stage.h"
#include "src/core/sph/sph_damage.h"
#include "src/core/sph/sph_damage_math.cuh"
#include "src/core/sph/sph_device_helpers.cuh"
#include "src/core/neighbor/neighbor_api.h"
#include "src/core/material/material_database.h"
#include "cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cudadem {

__device__ __forceinline__ unsigned long long damage_mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float damage_uniform_open(unsigned long long key) {
    return (static_cast<float>((damage_mix64(key) >> 40) & 0x00ffffffULL) + 0.5f) /
           16777216.0f;
}

// Knuth's sampler is exact for the deliberately compact lambda=log(N) used
// here (lambda is small for practical SPH bodies), and is deterministic from
// damage_seed + global_id rather than particle order or GPU decomposition.
__device__ __forceinline__ int compact_poisson(float lambda, unsigned long long key) {
    const float stop = expf(-fmaxf(lambda, 0.0f));
    float product = 1.0f;
    int n = 0;
    do {
        ++n;
        product *= damage_uniform_open(key + static_cast<unsigned long long>(n));
    } while (product > stop && n < 64);
    return max(1, n - 1);
}

// Compact deterministic Weibull intervals.  Stats are whole-Part values for
// each material, supplied by the host wrapper or by a decomposed driver.
__global__ void init_weibull_flaws_kernel(
    int32_t* flaws_ntot,
    float* flaws_strain_min,
    float* flaws_strain_max,
    float* flaws_m0,
    const float* mass,
    const int32_t* global_id,
    const int32_t* material_id,
    const int32_t* part_id,
    const MaterialRecord* materials,
    const SphDamagePartMaterialStats* part_material_stats,
    int32_t part_material_stats_count, int32_t material_count,
    int32_t num_particles, int32_t fallback_particle_count,
    unsigned long long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const int mid = material_id[i];
    const int pid = part_id[i];
    const SphDamageProperties& damage = materials[mid].sph_damage;
    if (damage.model != SphDamageModel::GRADY_KIPP || !(damage.k > 0.0f) ||
        !(damage.m > 0.0f) || !isfinite(damage.k) || !isfinite(damage.m)) {
        flaws_ntot[i] = 0; flaws_strain_min[i] = 0.0f;
        flaws_strain_max[i] = 0.0f; flaws_m0[i] = 1.0f;
        return;
    }

    int body_count = fallback_particle_count > 0 ? fallback_particle_count : num_particles;
    double body_mass = 0.0;
    const int64_t stats_index = static_cast<int64_t>(pid) * material_count + mid;
    if (part_material_stats != nullptr && pid >= 0 && mid >= 0 && material_count > 0 &&
        stats_index >= 0 && stats_index < part_material_stats_count) {
        const SphDamagePartMaterialStats stats = part_material_stats[stats_index];
        if (stats.particle_count > 0) body_count = stats.particle_count;
        body_mass = stats.total_mass;
    }
    const float rho0 = materials[mid].sph_tillotson.reference_density;
    const double local_volume = static_cast<double>(mass[i]) / rho0;
    const float alpha0 = materials[mid].sph_porosity.model == SphPorosityModel::P_ALPHA
        ? fmaxf(materials[mid].sph_porosity.alpha_initial, 1.0f) : 1.0f;
    const double body_volume = body_mass > 0.0 && isfinite(body_mass)
        ? body_mass / static_cast<double>(rho0) / alpha0
        : fmax(local_volume * static_cast<double>(body_count) / alpha0, local_volume);
    const unsigned long long key = seed ^ static_cast<unsigned long long>(global_id[i]);
    const int ntot = compact_poisson(logf(static_cast<float>(max(body_count, 1))), key);
    const double N = static_cast<double>(max(body_count, 1));
    const double r = static_cast<double>(damage_uniform_open(
        key ^ 0xd1b54a32d192ed03ULL));
    const double first_order = -N * log(1.0-r);
    const double last_order = N * log(1.0+N*r);
    const SphFlawInterval interval = sph_damage_flaw_interval(
        damage.k, damage.m, body_volume, ntot, first_order, last_order,
        damage.min_strain);

    flaws_ntot[i] = ntot;
    flaws_strain_min[i] = interval.strain_min;
    flaws_strain_max[i] = interval.strain_max;
    flaws_m0[i] = interval.m0;
}

__global__ void evaluate_sph_damage_kernel(
    const float*  __restrict__ S_xx, const float* __restrict__ S_xy, const float* __restrict__ S_xz,
    const float*  __restrict__ S_yy, const float* __restrict__ S_yz, const float* __restrict__ S_zz,
    const float*  __restrict__ pressure,
    const float*  __restrict__ smoothing_length,
    const float*  __restrict__ distension,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const float* __restrict__ damage,
    float* __restrict__ crack_strain,
    const int32_t* __restrict__ flaws_ntot,
    const float* __restrict__ flaws_strain_min,
    const float* __restrict__ flaws_strain_max,
    const float* __restrict__ flaws_m0,
    float* __restrict__ qdot,
    float* __restrict__ qcap,
    int32_t num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    qdot[i] = 0.0f; qcap[i] = 0.0f; crack_strain[i] = 0.0f;
    const MaterialRecord& material = materials[material_id[i]];
    if (material.sph_damage.model != SphDamageModel::GRADY_KIPP) return;
    const float K = static_cast<float>(sph_bulk_modulus(material)), mu = material.sph_strength.shear_modulus;
    const float alpha = material.sph_porosity.model == SphPorosityModel::P_ALPHA ? fmaxf(distension[i], 1.0f) : 1.0f;
    if (!(K > 0.0f) || !(mu > 0.0f)) return;
    const float D = fminf(1.0f, fmaxf(0.0f, damage[i]));
    const float lambda = sph_damage_lambda_max(S_xx[i]-pressure[i], S_xy[i], S_xz[i],
        S_yy[i]-pressure[i], S_yz[i], S_zz[i]-pressure[i]);
    const int ntot = flaws_ntot[i];
    const float smin = flaws_strain_min[i];
    const float activation_floor = fmaxf(smin, material.sph_damage.min_strain);
    const SphDamageRate rate = sph_damage_rate(lambda, D, K, mu,
        material.sph_tillotson.reference_density, alpha, smoothing_length[i], ntot,
        smin, activation_floor, flaws_strain_max[i], flaws_m0[i], material.sph_damage.crack_speed_fraction,
        material.sph_eos.elastic_modulus, material.sph_eos.continuous_activation);
    crack_strain[i] = rate.crack_strain;
    qdot[i] = rate.qdot;
    qcap[i] = rate.qcap;
}

__global__ void integrate_sph_damage_kernel(float* damage, const float* qdot,
                                             const float* qcap, float dt,
                                             int32_t num_particles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const float Dold = fminf(1.0f, fmaxf(0.0f, damage[i]));
    const float qold = cbrtf(Dold);
    const float rate = qdot[i] > 0.0f && isfinite(qdot[i]) ? qdot[i] : 0.0f;
    const float cap = fminf(1.0f, fmaxf(0.0f, qcap[i]));
    const float qnew = fmaxf(qold, fminf(qold + rate*fmaxf(dt, 0.0f), cap));
    const float Dnew = qnew*qnew*qnew;
    damage[i] = fminf(1.0f, fmaxf(Dold, isfinite(Dnew) ? Dnew : Dold));
}

void initialize_sph_damage(SimulationState& state, const SimConfig& config,
                           const SphDamageFlawInitialization& initialization,
                           cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(init_weibull_flaws_kernel, nb, bs, 0, stream,
        state.particles.flaws_ntot,
        state.particles.flaws_strain_min,
        state.particles.flaws_strain_max,
        state.particles.flaws_m0,
        state.particles.mass,
        state.particles.global_id,
        state.particles.material_id, state.particles.clump_id,
        state.material_database.materials,
        initialization.part_material_stats, initialization.part_material_stats_count,
        initialization.material_count,
        num, initialization.fallback_particle_count, initialization.damage_seed
    );
}

void initialize_sph_damage(SimulationState& state, const SimConfig& config,
                           int32_t global_particle_count) {
    if (config.physics != SimulationPhysics::SPH) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    std::vector<int32_t> material_id(num), part_id(num);
    std::vector<float> mass(num);
    CUDA_CHECK(cudaMemcpy(material_id.data(), state.particles.material_id,
                          static_cast<size_t>(num)*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass.data(), state.particles.mass,
                          static_cast<size_t>(num)*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part_id.data(), state.particles.clump_id,
                          static_cast<size_t>(num)*sizeof(int32_t), cudaMemcpyDeviceToHost));
    int32_t max_material_id = -1, max_part_id = -1;
    for (int32_t i = 0; i < num; ++i) {
        max_material_id = std::max(max_material_id, material_id[i]);
        max_part_id = std::max(max_part_id, part_id[i]);
    }
    const int64_t stats_size64 = static_cast<int64_t>(max_part_id + 1) * (max_material_id + 1);
    if (stats_size64 < 0 || stats_size64 > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("SPH Part/material damage statistics exceed int32 indexing");
    const int32_t material_count = max_material_id + 1;
    std::vector<SphDamagePartMaterialStats> stats(static_cast<size_t>(stats_size64), {0, 0.0});
    for (int32_t i = 0; i < num; ++i) {
        const int32_t mid = material_id[i];
        const int32_t pid = part_id[i];
        if (mid >= 0 && pid >= 0) {
            SphDamagePartMaterialStats& group = stats[static_cast<size_t>(pid) * material_count + mid];
            ++group.particle_count;
            // The kernel converts this material mass to the SPHSOL solid
            // Weibull volume with rho0 and alpha_initial.
            group.total_mass += static_cast<double>(mass[i]);
        }
    }
    SphDamagePartMaterialStats* device_stats = nullptr;
    if (!stats.empty()) {
        CUDA_CHECK(cudaMalloc(&device_stats, stats.size()*sizeof(SphDamagePartMaterialStats)));
        CUDA_CHECK(cudaMemcpy(device_stats, stats.data(), stats.size()*sizeof(SphDamagePartMaterialStats),
                              cudaMemcpyHostToDevice));
    }
    SphDamageFlawInitialization initialization{};
    initialization.part_material_stats = device_stats;
    initialization.part_material_stats_count = static_cast<int32_t>(stats.size());
    initialization.material_count = material_count;
    initialization.fallback_particle_count = global_particle_count > 0 ? global_particle_count : num;
    initialization.damage_seed = config.sph.damage_seed;
    initialize_sph_damage(state, config, initialization, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (device_stats != nullptr) CUDA_CHECK(cudaFree(device_stats));
}

void evaluate_sph_damage(SimulationState& state, const SimConfig& config,
                         const SphDamageWorkspace& workspace, cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH || !sph_current_stage(config, state.current_time).evolve_damage ||
        workspace.qdot == nullptr || workspace.qcap == nullptr) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(evaluate_sph_damage_kernel, nb, bs, 0, stream,
        state.particles.S_xx, state.particles.S_xy, state.particles.S_xz,
        state.particles.S_yy, state.particles.S_yz, state.particles.S_zz,
        state.particles.pressure, state.particles.smoothing_length,
        state.particles.distension,
        state.particles.material_id, state.material_database.materials,
        state.particles.damage,
        state.particles.crack_strain,
        state.particles.flaws_ntot,
        state.particles.flaws_strain_min,
        state.particles.flaws_strain_max,
        state.particles.flaws_m0, workspace.qdot, workspace.qcap, num);
}

void integrate_sph_damage(SimulationState& state, const SimConfig& config,
                          const SphDamageWorkspace& workspace, float dt,
                          cudaStream_t stream) {
    if (config.physics != SimulationPhysics::SPH || !sph_current_stage(config, state.current_time).evolve_damage ||
        workspace.qdot == nullptr || workspace.qcap == nullptr) return;
    const int32_t num = state.particles.num_particles;
    if (num <= 0) return;
    const int bs = config.gpu.block_size;
    const int nb = (num + bs - 1) / bs;
    LAUNCH_KERNEL(integrate_sph_damage_kernel, nb, bs, 0, stream,
                  state.particles.damage, workspace.qdot, workspace.qcap, dt, num);
}

} // namespace cudadem

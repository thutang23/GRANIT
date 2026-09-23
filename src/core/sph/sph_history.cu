#include "sph_history.h"
#include "sph_history_math.cuh"
#include "sph_eos_math.cuh"
#include "sph_stage.h"
#include "cuda_utils.h"

namespace cudadem {

// Cold compression follows the same material EOS and tensile-damage closure at
// the cold-energy state.  It is diagnostic and adds no heat to total energy.
__global__ void advance_sph_cold_energy_kernel(
    ParticleSoA particles, const MaterialRecord* materials, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles.num_particles) return;
    const auto& material = materials[particles.material_id[i]];
    const double rho = particles.density[i];
    const auto cold = sph_eos_state(material, rho, particles.energy_cold[i],
        particles.damage[i], particles.distension[i]);
    particles.energy_cold[i] += dt * cold.pressure / (rho*rho) * particles.rho_rate[i];
}

__global__ void update_sph_history_kernel(
    ParticleSoA particles, const MaterialRecord* materials, float dt, bool evolve_damage) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles.num_particles) return;
    particles.pressure_peak[i] = fmax(particles.pressure_peak[i], static_cast<double>(particles.pressure[i]));
    particles.energy_thermal_peak[i] = fmax(particles.energy_thermal_peak[i],
        particles.energy[i] - particles.energy_cold[i]);
    const auto& material = materials[particles.material_id[i]];
    const auto& damage = material.sph_damage;
    if (evolve_damage && damage.model == SphDamageModel::GRADY_KIPP && dt > 0.0f &&
        particles.damage[i] > 0.0f && particles.damage[i] < 1.0f) {
        const double alpha = material.sph_porosity.model == SphPorosityModel::P_ALPHA
            ? particles.distension[i] : 1.0;
        const double cg = damage.crack_speed_fraction * sqrt((sph_bulk_modulus(material) +
            4.0/3.0*material.sph_strength.shear_modulus) /
            (alpha*material.sph_tillotson.reference_density));
        // crack_strain is evaluated before the damage commit; q is the new
        // damage root, following the SPHSOL crack-area diagnostic convention.
        particles.fracture_area[i] += dt * sph_fracture_area_rate(
            particles.crack_strain[i], cbrt(static_cast<double>(particles.damage[i])),
            damage.m, damage.k, cg);
        particles.fragment_length[i] = sph_fragment_length(particles.fracture_area[i], damage.m);
    }
}

__global__ void damp_sph_relaxation_kernel(ParticleSoA particles, double decay, bool track_history) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles.num_particles) return;
    const double vx = particles.vel_x[i], vy = particles.vel_y[i], vz = particles.vel_z[i];
    if (track_history)
        particles.damping_energy[i] += 0.5*(1.0-decay*decay)*(vx*vx+vy*vy+vz*vz);
    particles.vel_x[i] *= decay;
    particles.vel_y[i] *= decay;
    particles.vel_z[i] *= decay;
}

void advance_sph_cold_energy(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    const auto stage = sph_current_stage(config, state.current_time);
    if (!config.sph.track_history || stage.mode != SphRunStage::SHOCK || !stage.evolve_energy) return;
    const int n = state.particles.num_particles;
    if (n <= 0) return;
    const int bs = config.gpu.block_size;
    LAUNCH_KERNEL(advance_sph_cold_energy_kernel, (n+bs-1)/bs, bs, 0, stream,
        state.particles, state.material_database.materials, static_cast<float>(state.dt));
}

void update_sph_history(SimulationState& state, const SimConfig& config, float dt, cudaStream_t stream) {
    const auto stage = sph_current_stage(config, state.current_time);
    if (!config.sph.track_history || stage.mode != SphRunStage::SHOCK) return;
    const int n = state.particles.num_particles;
    if (n <= 0) return;
    const int bs = config.gpu.block_size;
    LAUNCH_KERNEL(update_sph_history_kernel, (n+bs-1)/bs, bs, 0, stream,
        state.particles, state.material_database.materials, dt, stage.evolve_damage);
}

void damp_sph_relaxation(SimulationState& state, const SimConfig& config, double dt, cudaStream_t stream) {
    const auto stage = sph_current_stage(config, state.current_time);
    if (stage.mode != SphRunStage::RELAX || !(stage.damping_rate > 0.0)) return;
    const int n = config.multigpu.enabled ? state.particles.num_local : state.particles.num_particles;
    if (n <= 0) return;
    ParticleSoA owned = state.particles;
    owned.num_particles = n;
    const int bs = config.gpu.block_size;
    LAUNCH_KERNEL(damp_sph_relaxation_kernel, (n+bs-1)/bs, bs, 0, stream,
        owned, exp(-stage.damping_rate*dt), config.sph.track_history);
}

} // namespace cudadem

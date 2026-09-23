#ifndef CUDADEM_SPH_DAMAGE_H
#define CUDADEM_SPH_DAMAGE_H

#include <cstdint>
#include <cuda_runtime.h>
#include "src/core/constant/system_initialize.h"

namespace cudadem {

// Device-resident table indexed by (SPHSOL Part ID, material ID), derived
// from each whole Part.  It is initialization metadata, never particle or
// restart state.  The historical clump_id storage is the Part ID in SPH.
struct SphDamagePartMaterialStats {
    int32_t particle_count;
    double total_mass;
};

struct SphDamageFlawInitialization {
    const SphDamagePartMaterialStats* part_material_stats = nullptr;
    int32_t part_material_stats_count = 0;
    int32_t material_count = 0;
    int32_t fallback_particle_count = -1;
    uint64_t damage_seed = 0;
};

// Caller-owned ephemeral device workspace; qdot/qcap must contain one value
// per particle and are deliberately excluded from ParticleSoA/checkpoints.
struct SphDamageWorkspace {
    float* qdot = nullptr;
    float* qcap = nullptr;
};

void initialize_sph_damage(SimulationState& state, const SimConfig& config,
                           const SphDamageFlawInitialization& initialization,
                           cudaStream_t stream = 0);

// Compatibility wrapper derives current-state (Part, material) statistics.
// Drivers with a decomposed body should use the explicit contract.
void initialize_sph_damage(SimulationState& state, const SimConfig& config,
                           int32_t global_particle_count = -1);

// Evaluate closed-state damage derivatives before EOS/strength. This writes
// instantaneous crack_strain and workspace values, but leaves D unchanged.
void evaluate_sph_damage(SimulationState& state, const SimConfig& config,
                         const SphDamageWorkspace& workspace, cudaStream_t stream);

// Advance D only after EOS and strength have consumed the old value.
void integrate_sph_damage(SimulationState& state, const SimConfig& config,
                          const SphDamageWorkspace& workspace, float dt,
                          cudaStream_t stream);

} // namespace cudadem

#endif // CUDADEM_SPH_DAMAGE_H

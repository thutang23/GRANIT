#ifndef CUDADEM_SPH_STAGE_H
#define CUDADEM_SPH_STAGE_H

#include "src/core/constant/command_line_interface.h"

namespace cudadem {

inline SphStageParameters sph_current_stage(const SimConfig& config, double time) {
    const int index = sph_stage_index(config.sph.stages, config.sph.stage_count, time);
    SphStageParameters stage = index >= 0 ? config.sph.stages[index] : SphStageParameters{};
    if (index < 0) stage.end_time = config.time.stop_time;
    // Legacy no-sequence relaxation controls retain their original whole-run
    // meaning.  In an explicit sequence they apply to RELAX only.
    if (index < 0 || stage.mode == SphRunStage::RELAX) {
        stage.use_strength = stage.use_strength && !config.sph.relaxation_disable_strength;
        stage.evolve_damage = stage.evolve_damage && !config.sph.relaxation_disable_damage;
    }
    return stage;
}

// A boundary checkpoint contains the force closed under the preceding stage.
// Mid-stage checkpoints already contain the correct closed force/rates and
// must not be re-evaluated at full-step velocity during ordinary continuation.
inline bool sph_stage_transition_at(const SimConfig& config, double time) {
    for (int i = 0; i + 1 < config.sph.stage_count; ++i)
        if (time == config.sph.stages[i].end_time) return true;
    return false;
}

} // namespace cudadem

#endif // CUDADEM_SPH_STAGE_H

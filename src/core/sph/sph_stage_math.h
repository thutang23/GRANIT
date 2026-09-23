#ifndef CUDADEM_SPH_STAGE_MATH_H
#define CUDADEM_SPH_STAGE_MATH_H

#include <cmath>
#include <cstdint>

namespace cudadem {

enum class SphRunStage : uint8_t { RELAX = 0, SHOCK = 1, LATE = 2 };

struct SphStageParameters {
    SphRunStage mode = SphRunStage::SHOCK;
    double end_time = 0.0;       // Absolute simulation time (s)
    double damping_rate = 0.0;   // RELAX velocity decay rate (1/s)
    bool evolve_energy = true;
    bool evolve_damage = true;
    bool evolve_porosity = true;
    bool use_strength = true;
    bool elastic_eos = false;
};

inline SphStageParameters sph_stage_defaults(SphRunStage mode) {
    SphStageParameters stage;
    stage.mode = mode;
    stage.evolve_energy = mode != SphRunStage::RELAX;
    stage.evolve_damage = mode == SphRunStage::SHOCK;
    stage.evolve_porosity = mode == SphRunStage::SHOCK;
    stage.elastic_eos = mode != SphRunStage::SHOCK;
    return stage;
}

inline const char* sph_stage_name(SphRunStage mode) {
    switch (mode) {
        case SphRunStage::RELAX: return "RELAX";
        case SphRunStage::LATE: return "LATE";
        default: return "SHOCK";
    }
}

// Right-continuous stages: a closed state on a boundary belongs to the
// following stage.  The driver lands exactly on that boundary before switching.
inline int sph_stage_index(const SphStageParameters* stages, int count, double time) {
    for (int i = 0; i < count; ++i)
        if (time < stages[i].end_time) return i;
    return count > 0 ? count - 1 : -1;
}

inline double sph_stage_step(double time, double dt, double end_time) {
    return fmax(0.0, fmin(dt, end_time - time));
}

} // namespace cudadem

#endif // CUDADEM_SPH_STAGE_MATH_H

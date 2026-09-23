#ifndef CUDADEM_EXTERNAL_BODIES_H
#define CUDADEM_EXTERNAL_BODIES_H

// Prescribed external massive bodies for the global, inertial-frame DEM path.
// The module is intentionally independent of WallManager: a Saturn monopole
// and its optional satellite are gravity-only point bodies, while Saturn loss
// is a particle-centre lifecycle event rather than a mechanical contact.

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cudadem {

struct ExternalBodyState {
    bool enabled;
    bool saturn_enabled;
    double saturn_x, saturn_y, saturn_z;
    double saturn_mass;
    double saturn_absorb_radius;
    bool satellite_enabled;
    double satellite_x, satellite_y, satellite_z;
    double satellite_mass;
};

struct ExternalBodyLedger {
    int64_t initial_live_particles;
    double initial_live_mass;
    int64_t absorbed_particles;
    double absorbed_mass;
    double absorbed_momentum_x, absorbed_momentum_y, absorbed_momentum_z;
    double absorbed_angular_momentum_x, absorbed_angular_momentum_y, absorbed_angular_momentum_z;
    double absorbed_orbital_energy;
};

// One immutable, centre-crossing record per particle.  `velocity` is the
// drift half-step velocity; energy is total two-body orbital energy relative
// to Saturn, E=m(0.5 |v|^2-GM/|r|), evaluated at the interpolated crossing.
struct ExternalBodyEvent {
    int32_t particle_global_id;
    int32_t body_id;
    int64_t step;
    double time;
    double position_x, position_y, position_z;
    double velocity_x, velocity_y, velocity_z;
    double mass;
    double momentum_x, momentum_y, momentum_z;
    double angular_momentum_x, angular_momentum_y, angular_momentum_z;
    double orbital_energy;
};

struct ExternalBodyAbsorption {
    int32_t count;
    double mass;
};

// Returns the exact prescribed state at `time`.  The satellite orbit is a
// circular Kepler orbit around the configured Saturn monopole; phase is
// anchored to simulation.time.start_time, avoiding any trajectory file or
// interpolation/coverage ambiguity on restart.
ExternalBodyState evaluate_external_body_state(const SimConfig& config,
                                               double time);

class ExternalBodyManager {
public:
    static void initialize(const SimConfig& config, int32_t initial_live_particles,
                           double initial_live_mass);
    static void shutdown();

    // Fast-level force: call on every inner DEM step, including the force
    // bootstrap.  Self gravity remains the sole RESPA-cached force.
    static void add_forces(SimulationState& state, const SimConfig& config,
                           double time, int32_t particle_count,
                           cudaStream_t stream = 0);

    // Detect centre crossings after the drift and before neighbor/contact
    // work.  A particle is marked dead atomically once; the caller owns the
    // subsequent canonical compaction at this closed host/device boundary.
    static ExternalBodyAbsorption mark_absorbed(SimulationState& state,
                                                const SimConfig& config,
                                                int32_t particle_count,
                                                int64_t step, double step_start_time,
                                                double dt,
                                                cudaStream_t stream = 0);

    static void write_checkpoint_sidecar(const std::string& checkpoint_path,
                                         const SimulationState& state,
                                         const SimConfig& config);
    static void restore_checkpoint_sidecar(const std::string& checkpoint_path,
                                           const SimulationState& state,
                                           const SimConfig& config);
    static bool checkpoint_sidecar_matches(const std::string& checkpoint_path,
                                           int64_t step, double time,
                                           const SimConfig& config);
    static bool is_enabled();
    static ExternalBodyLedger ledger();
    static const std::vector<ExternalBodyEvent>& events();
    static uint64_t gravity_evaluations();
    static void write_event_receipts(const std::string& output_directory);
};

}  // namespace cudadem

#endif  // CUDADEM_EXTERNAL_BODIES_H

// =============================================================================
// SPH CFL Timestep Module
// =============================================================================
// CFL-limited timestep computation for SPH simulations.
//
// Rule: dt = cfl_number * min_i( h_i / (c_sound_i + |v_i| + eps) )
//
// - Recomputes dt each step based on current particle state
// - Clamps to [dt_min, dt_max]
// - Warns when dt repeatedly hits dt_min
// =============================================================================

#ifndef CUDADEM_SPH_TIMESTEP_H
#define CUDADEM_SPH_TIMESTEP_H

#include <cuda_runtime.h>
#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"

namespace cudadem {

enum class SPHTimestepConstraint : int32_t {
    NONE = 0,
    ELASTIC_CFL = 1,
    DENSITY_RATE = 2,
    ENERGY_RATE = 3,
    DAMAGE_RATE = 4,
    POROSITY_ALPHA_RATE = 5,
    POROSITY_PRESSURE_RATE = 6,
};

struct SPHTimestepDiagnostic {
    float dt_cfl;
    float dt_density;
    float dt_energy;
    float dt_damage;
    float dt_porosity_alpha;
    float dt_porosity_pressure;
    float longitudinal_sound_speed;
    float particle_speed;
    SPHTimestepConstraint constraint;
};

// SPH timestep parameters
struct SPHTimestepParams {
    float cfl_number;      // Signal-speed CFL coefficient, finite in (0, 1]
    float dt_min;          // Minimum timestep
    float dt_max;          // Maximum timestep
    float epsilon;         // Small number to avoid division by zero

    SPHTimestepParams()
        : cfl_number(0.3f), dt_min(1.0e-10f), dt_max(1.0e-3f), epsilon(1.0e-10f) {}
};

// SPH timestep result (for host readback)
struct SPHTimestepResult {
    float dt;              // Computed timestep
    float dt_unclamped;    // Before clamping
    int32_t limiting_particle;  // Index of particle with min dt
    float max_speed;       // Maximum |v| found
    float max_c_sound;     // Maximum longitudinal elastic sound speed found
    int32_t dt_min_hits;   // Count of consecutive dt_min hits
    SPHTimestepDiagnostic limiting_diagnostic;

    SPHTimestepResult()
        : dt(0), dt_unclamped(0), limiting_particle(-1),
          max_speed(0), max_c_sound(0), dt_min_hits(0),
          limiting_diagnostic{0, 0, 0, 0, 0, 0, 0, 0,
                              SPHTimestepConstraint::NONE} {}
};

const char* sph_timestep_constraint_name(SPHTimestepConstraint constraint);

// Compute SPH CFL timestep on device
// Returns per-particle dt values, reduces to find minimum
void compute_sph_timestep(
    SimulationState& state,
    const SimConfig& config,
    SPHTimestepResult& result,
    cudaStream_t stream = 0
);

// Initialize SPH timestep module (allocate work buffers)
void initialize_sph_timestep(int32_t max_particles);

// Shutdown SPH timestep module (free work buffers)
void shutdown_sph_timestep();

// Get recommended initial timestep from config
float get_initial_sph_timestep(const SimConfig& config);

} // namespace cudadem

#endif // CUDADEM_SPH_TIMESTEP_H

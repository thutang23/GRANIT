#ifndef CUDADEM_BONDED_RECT_WALL_H
#define CUDADEM_BONDED_RECT_WALL_H

// =============================================================================
// Bonded Rectangular Prism Wall — Device Structures and Host Declarations
// =============================================================================
// A finite rigid box with 6-DOF dynamics:
//   - Particles collide with any face via box closest-point test
//   - Contact forces computed via wall_contact_force.cuh (Hertz-Mindlin)
//   - Reaction forces accumulated with double atomicAdd
//   - Integration kernel solves Euler's equations + Rodrigues rotation update
//
// Memory layout:
//   d_bonded_rect_states[MAX_BONDED_RECT_WALLS]   — mutable 6-DOF state
//   d_bonded_rect_reactions[MAX_BONDED_RECT_WALLS] — zeroed before each step
// =============================================================================

#include "../constant/command_line_interface.h"
#include "wall.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem {

constexpr int MAX_BONDED_RECT_WALLS = 8;

// Forward declaration
struct SimulationState;

// =============================================================================
// BondedRectWallState — mutable 6-DOF rigid body state (device global)
// =============================================================================

struct BondedRectWallState {
    int32_t id;
    int32_t material_id;
    bool    enabled;

    // Fixed geometry: half-extents in body frame (m)
    float half_x, half_y, half_z;

    // Rigid body properties
    float mass;                              // kg
    float inertia_x, inertia_y, inertia_z;  // Principal moments (body frame, kg·m²)

    // Mutable 6-DOF state
    double pos_x, pos_y, pos_z;             // CoM position (world frame, m)
    float  rot[9];                           // Body-to-world rotation matrix, row-major
                                             //   rot[r*3+c] = element at row r, column c
                                             //   v_world = R * v_body
    double vel_x, vel_y, vel_z;             // Linear velocity (world frame, m/s)
    float  omega_x, omega_y, omega_z;       // Angular velocity (body frame, rad/s)
                                             //   kept in body frame for Euler's equations

    __host__ __device__ BondedRectWallState()
        : id(-1), material_id(-1), enabled(false),
          half_x(1.0f), half_y(1.0f), half_z(1.0f),
          mass(1.0f), inertia_x(1.0f), inertia_y(1.0f), inertia_z(1.0f),
          pos_x(0), pos_y(0), pos_z(0),
          vel_x(0), vel_y(0), vel_z(0),
          omega_x(0), omega_y(0), omega_z(0) {
        // Identity rotation matrix
        rot[0]=1; rot[1]=0; rot[2]=0;
        rot[3]=0; rot[4]=1; rot[5]=0;
        rot[6]=0; rot[7]=0; rot[8]=1;
    }
};

// =============================================================================
// BondedRectReaction — per-wall force/torque accumulator (zeroed each step)
// =============================================================================

struct BondedRectReaction {
    double force_x,  force_y,  force_z;   // Net reaction force (world frame, N)
    double torque_x, torque_y, torque_z;  // Net torque about CoM (world frame, N·m)

    __host__ __device__ BondedRectReaction()
        : force_x(0), force_y(0), force_z(0),
          torque_x(0), torque_y(0), torque_z(0) {}
};

// =============================================================================
// Host Function Declarations (implemented in bonded_rect_wall.cu)
// =============================================================================

// Initialize device arrays from config; allocates d_states and d_reactions.
void bonded_rect_wall_initialize(
    const SimConfig& config,
    int32_t num_particles
);

// Returns true after a successful initialize call.
bool bonded_rect_wall_is_initialized();

// Get the current state and reaction forces of all bonded walls on the host.
// Returns the number of bonded walls.
int32_t bonded_rect_wall_get_states(
    BondedRectWallState* host_states,
    BondedRectReaction* host_reactions
);

// Restore a closed-step checkpoint after initialization. Fixed wall identity,
// geometry and inertia must still match the current configuration.
void bonded_rect_wall_restore_states(
    const BondedRectWallState* host_states,
    const BondedRectReaction* host_reactions,
    int32_t count
);

// Launch contact force kernel: per-particle box test → Hertz-Mindlin → atomicAdd.
// Must be called before bonded_rect_wall_integrate on the same stream.
void bonded_rect_wall_compute_forces(
    SimulationState& state,
    const SimConfig& config,
    float dt,
    float current_time,
    cudaStream_t stream = 0
);

// Launch integration kernel: Euler's equations + Rodrigues rotation update.
// gx/gy/gz: gravity vector (m/s²) in world frame.
void bonded_rect_wall_integrate(
    float dt,
    float gx, float gy, float gz,
    cudaStream_t stream = 0
);

// Free device memory and reset state.
void bonded_rect_wall_shutdown();

} // namespace cudadem

#endif // CUDADEM_BONDED_RECT_WALL_H

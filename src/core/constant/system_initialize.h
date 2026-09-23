#ifndef CUDADEM_SYSTEM_INITIALIZE_H
#define CUDADEM_SYSTEM_INITIALIZE_H

#include "command_line_interface.h"
#include "../wall/wall.h"
#include "../neighbor/neighbor_types.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <utility>

namespace cudadem {

// =============================================================================
// SoA (Structure of Arrays) Particle Data
// =============================================================================
// CRITICAL: SoA layout is essential for coalesced GPU memory access
// Each array stores one attribute for ALL particles contiguously
// =============================================================================

// -----------------------------------------------------------------------------
// SoA Particle Arrays (Device Pointers)
// -----------------------------------------------------------------------------
struct ParticleSoA {
    // Particle count
    int32_t num_particles;       // Total particles (Local + Ghost)
    int32_t num_local;           // Local particles (0 ... num_local-1)
    int32_t num_ghost;           // Ghost particles (num_local ... num_particles-1)
    int32_t capacity;            // Allocated capacity (may be > num_particles)
    bool input_has_omega;        // Descriptor presence, not inferred from values
    bool input_has_smoothing_length; // Fresh SPH INPUT supplied heterogeneous h
    bool input_has_clump_offset; // True for DEM v3 checkpoints
    bool input_has_winding;      // All three shearing-box winding descriptors present
    // True only when an SPH checkpoint supplied all compact flaw descriptors.
    // False permits callers to regenerate flaw metadata from the configured seed.
    bool input_has_damage_flaw_metadata;
    uint8_t source_record_kind;  // BinaryRecordKind value from the loaded header

    // === Position (double precision) ===
    double* pos_x;
    double* pos_y;
    double* pos_z;

    // === Last position (for Verlet buffer displacement check) ===
    double* last_pos_x;
    double* last_pos_y;
    double* last_pos_z;

    // === Velocity (double) ===
    double* vel_x;
    double* vel_y;
    double* vel_z;

    // === Integer image winding (canonical shearing-periodic coordinates) ===
    int64_t* winding_x;
    int64_t* winding_y;
    int64_t* winding_z;

    // === Force/Acceleration (double) ===
    double* force_x;
    double* force_y;
    double* force_z;

    // === RESPA outer-loop gravity acceleration (double) ===
    // Stores gravitational acceleration (m/s²) computed every K steps.
    // NOT force — this is acceleration = F_grav / mass.
    // Persists across inner steps; overwritten only at outer-step boundaries.
    double* gravity_force_x;
    double* gravity_force_y;
    double* gravity_force_z;

    // Acceleration aliases (for backward compatibility)
    double* acc_x;
    double* acc_y;
    double* acc_z;

    // === Angular velocity (double) ===
    double* omega_x;
    double* omega_y;
    double* omega_z;

    // === Torque (double) ===
    double* torque_x;
    double* torque_y;
    double* torque_z;

    // Angular acceleration aliases (for backward compatibility)
    double* alpha_x;
    double* alpha_y;
    double* alpha_z;

    // === Orientation quaternion (double) ===
    double* quat_w;
    double* quat_x;
    double* quat_y;
    double* quat_z;

    // === Particle properties ===
    float* radius;
    float* mass;
    float* moment_of_inertia;    // I = 2/5 * m * r^2 for sphere

    // === Interaction support extent ===
    float* support_extent;        // DEM: radius, SPH: kernel_cutoff_factor * h

    // === Material properties (per-particle for heterogeneous systems) ===
    float* cohesive_strength;    // Individual cohesion strength
    // === SPH properties (basic) ===
    // Density and its time derivative require FP64 for weakly compressed,
    // self-gravitating bodies.  Their hydrostatic signal can be O(1e-6*rho0),
    // so FP32 rho/rho0 - 1 both quantizes pressure and can swallow rho updates.
    double* density;             // rho (density)
    float* pressure;             // P (pressure)
    float* viscosity;            // mu (dynamic viscosity)
    float* c_sound;              // Speed of sound
    float* drhodt;
    double* rho_rate;              // Density rate of change
    float* smoothing_length;     // h (SPH smoothing length)
    double* energy;              // e (specific internal energy J/kg)
    double* dedt;                // de/dt (energy rate)

    // === Optional SPH shock and relaxation history (specific energies) ===
    double* energy_cold;
    double* energy_thermal_peak;
    double* pressure_peak;
    double* fracture_area;
    double* fragment_length;
    double* damping_energy;


    // === SPH material state ===
    // Deviatoric stress tensor S (symmetric, 6 independent SoA components)
    float* S_xx;                 // S_11
    float* S_xy;                 // S_12 = S_21
    float* S_xz;                 // S_13 = S_31
    float* S_yy;                 // S_22
    float* S_yz;                 // S_23 = S_32
    float* S_zz;                 // S_33
    // Grady-Kipp damage
    float* damage;               // D in [0,1] scalar damage
    float* crack_strain;         // instantaneous nonnegative max-principal elastic tensile strain
    int32_t* flaws_ntot;
    float* flaws_strain_min;
    float* flaws_strain_max;
    float* flaws_m0;
    // P-alpha porosity
    float* distension;           // alpha >= 1 (1 = fully compacted)
    // CSPH kernel gradient correction matrix (3x3 row-major SoA)
    float* L_00; float* L_01; float* L_02;
    float* L_10; float* L_11; float* L_12;
    float* L_20; float* L_21; float* L_22;
    // === Contact history tracking ===
    int32_t* contact_history_offset;
    int32_t* num_contacts;
    int32_t* wall_history_offset;
    int32_t* num_wall_contacts;
    int32_t* cell_id;
    int32_t* sorted_index;
    // === Multi-GPU support ===
    int32_t* global_id;
    // Immutable dense material identifier loaded from unified-v3 input.
    int32_t* material_id;
    // Independent runtime lifecycle: 0 live, 1 externally absorbed.  This
    // deliberately does not overload clump_id, whose -1 value remains a
    // legacy input/checkpoint row state.
    uint8_t* lifecycle_state;
    // === Contextual particle identity ===
    // DEM: -1 dead, 0 ordinary live, 1..K persistent rigid-clump ID.
    // SPH: -1 inactive/dead, 0..K dense SPHSOL Part ID.  The storage name is
    // retained for unified-v3 compatibility; it is not a clump membership in
    // SPH mode.
    int32_t* clump_id;
    float* clump_off_x;      // Sub-sphere body-frame offset from CoM (constant)
    float* clump_off_y;
    float* clump_off_z;
    // Default constructor
    ParticleSoA() : num_particles(0), num_local(0), num_ghost(0), capacity(0),
        input_has_omega(false), input_has_smoothing_length(false),
        input_has_clump_offset(false), input_has_winding(false),
        input_has_damage_flaw_metadata(false),
        source_record_kind(0),
        pos_x(nullptr), pos_y(nullptr), pos_z(nullptr),
        last_pos_x(nullptr), last_pos_y(nullptr), last_pos_z(nullptr),
        vel_x(nullptr), vel_y(nullptr), vel_z(nullptr),
        winding_x(nullptr), winding_y(nullptr), winding_z(nullptr),
        force_x(nullptr), force_y(nullptr), force_z(nullptr),
        gravity_force_x(nullptr), gravity_force_y(nullptr), gravity_force_z(nullptr),
        acc_x(nullptr), acc_y(nullptr), acc_z(nullptr),
        omega_x(nullptr), omega_y(nullptr), omega_z(nullptr),
        torque_x(nullptr), torque_y(nullptr), torque_z(nullptr),
        alpha_x(nullptr), alpha_y(nullptr), alpha_z(nullptr),
        quat_w(nullptr), quat_x(nullptr), quat_y(nullptr), quat_z(nullptr),
        radius(nullptr), mass(nullptr), moment_of_inertia(nullptr),
        support_extent(nullptr),
        cohesive_strength(nullptr),
        density(nullptr), pressure(nullptr), viscosity(nullptr),
        c_sound(nullptr), drhodt(nullptr), rho_rate(nullptr), smoothing_length(nullptr),
        energy(nullptr), dedt(nullptr),
        energy_cold(nullptr), energy_thermal_peak(nullptr), pressure_peak(nullptr), fracture_area(nullptr), fragment_length(nullptr), damping_energy(nullptr),
        S_xx(nullptr), S_xy(nullptr), S_xz(nullptr),
        S_yy(nullptr), S_yz(nullptr), S_zz(nullptr),
        damage(nullptr), crack_strain(nullptr),
        flaws_ntot(nullptr), flaws_strain_min(nullptr), flaws_strain_max(nullptr), flaws_m0(nullptr),
        distension(nullptr),
        L_00(nullptr), L_01(nullptr), L_02(nullptr),
        L_10(nullptr), L_11(nullptr), L_12(nullptr),
        L_20(nullptr), L_21(nullptr), L_22(nullptr),
        contact_history_offset(nullptr), num_contacts(nullptr),
        wall_history_offset(nullptr), num_wall_contacts(nullptr),
        cell_id(nullptr), sorted_index(nullptr),
        global_id(nullptr), material_id(nullptr), lifecycle_state(nullptr),
        clump_id(nullptr), clump_off_x(nullptr), clump_off_y(nullptr), clump_off_z(nullptr) {}
};

// -----------------------------------------------------------------------------
// Contact History for tangential/rolling/twisting spring memory
// -----------------------------------------------------------------------------
struct ContactHistory {
    int32_t partner_global_id;   // Contact partner Global ID (-1 if invalid)
    int32_t partner_local_id;    // Cached local index (optimization, verify before use)
    bool is_active;              // Whether contact is active
    bool seen_this_step;         // Shear contact-epoch marker (fits existing padding)

    // Tangential spring displacement
    float delta_t_x;
    float delta_t_y;
    float delta_t_z;

    // Rolling displacement
    float delta_r_x;
    float delta_r_y;
    float delta_r_z;

    // Twisting displacement (3-component vector for directional spring memory)
    float delta_tw_x;
    float delta_tw_y;
    float delta_tw_z;

    __host__ __device__ ContactHistory()
        : partner_global_id(-1), partner_local_id(-1), is_active(false), seen_this_step(false),
          delta_t_x(0), delta_t_y(0), delta_t_z(0),
          delta_r_x(0), delta_r_y(0), delta_r_z(0),
          delta_tw_x(0), delta_tw_y(0), delta_tw_z(0) {}
};

struct ContactHistoryBuffer {
    ContactHistory* data;
    int32_t capacity;            // Total slots
    int32_t max_contacts_per_particle;

    ContactHistoryBuffer()
        : data(nullptr), capacity(0), max_contacts_per_particle(32) {}
};

// Forward declarations for wall structures (defined in wall.h)
struct WallContactHistory;
struct WallContactHistoryBuffer;
struct PeriodicBoundaryConfig;

// =============================================================================
// Wall Runtime Data
// =============================================================================
struct WallSoA {
    int32_t num_walls;

    // Wall properties (device arrays)
    int32_t* id;
    uint8_t* type;               // WallType enum
    bool* enabled;

    // Position and orientation
    double* pos_x;
    double* pos_y;
    double* pos_z;
    double* normal_x;
    double* normal_y;
    double* normal_z;

    // Velocity (for moving walls)
    double* vel_x;
    double* vel_y;
    double* vel_z;

    // Periodical boundary info
    uint8_t* axis;
    float* bound_min;
    float* bound_max;

    // Legacy data pointer (backward compatibility)
    WallData* data;

    WallSoA() : num_walls(0),
        id(nullptr), type(nullptr), enabled(nullptr),
        pos_x(nullptr), pos_y(nullptr), pos_z(nullptr),
        normal_x(nullptr), normal_y(nullptr), normal_z(nullptr),
        vel_x(nullptr), vel_y(nullptr), vel_z(nullptr),
        axis(nullptr), bound_min(nullptr), bound_max(nullptr),
        data(nullptr) {}
};

// =============================================================================
// Neighbor Search Data Structures
// =============================================================================
// NeighborData is defined in src/core/neighbor/neighbor_types.h
// No longer defined here - moved to neighbor module

// =============================================================================
// Complete Simulation State
// =============================================================================
struct SimulationState {
    ParticleSoA particles;
    ContactHistoryBuffer contact_history;
    WallContactHistoryBuffer wall_contact_history;
    WallSoA walls;
    NeighborData neighbor;  // Unified neighbor search data (replaces CellLinkedList)
    // Dense material and pair tables owned by the CUDA device holding this
    // state.  They are deliberately not put in constant memory: the table is
    // runtime-sized and one complete copy is required per GPU.
    DeviceMaterialDatabase material_database;

    // Simulation time state
    double current_time;
    int64_t current_step;
    double dt;                   // Current timestep

    // Energy tracking (for convergence checks)
    double kinetic_energy;
    double potential_energy;
    
    // Verlet buffer state
    float max_displacement;      // Maximum particle displacement since last rebuild

    SimulationState()
        : current_time(0), current_step(0), dt(1e-5),
          kinetic_energy(0), potential_energy(0), max_displacement(0) {}
};

// =============================================================================
// CUDA Error Checking Macro
// =============================================================================
// Only define if not already defined (cuda_utils.h may define it first)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)
#endif

// =============================================================================
// Host-Only Function Declarations
// =============================================================================
// Implementations are in system_initialize_host.cpp (compiled without nvcc)
// =============================================================================

// Memory allocation
void allocate_particle_soa(ParticleSoA& p, int32_t num_particles);
// Grow a particle SoA without losing any active local or ghost field.  This is
// intentionally host-synchronous: callers must invoke it only at a phase
// boundary after their compute and communication streams have quiesced.
void resize_particle_soa_preserving(ParticleSoA& p, int32_t new_capacity);
// One-time stable canonical ordering: by default, active id 0 then positive
// IDs grouped in ascending order, followed by inactive/dead rows (id -1 or
// absorbed).  This is the DEM clump layout.  SPH callers can preserve their
// active input order because SPHSOL Part is metadata, not a sorting key; dead
// rows are still stably partitioned to the tail.
// Every SoA column, including global_id, is permuted identically.
// When a contact buffer is supplied its fixed-stride rows are permuted with
// the particles so a checkpoint round-trip preserves tangential history.
void canonicalize_particle_clump_order(
    ParticleSoA& p, ContactHistoryBuffer* contact_history = nullptr,
    bool preserve_active_order = false);

// Repair only serialization-only inactive SPH rows.  Live rows must have
// passed normal input validation and are deliberately outside this helper's
// scope.  A null distension pointer is allowed for output paths that do not
// serialize it (currently the multi-GPU snapshot path).
inline void sanitize_inactive_sph_tail_values(
    float* smoothing_length, float* distension, int32_t count,
    float fallback_smoothing_length) {
    if (count < 0 || !(std::isfinite(fallback_smoothing_length) &&
                       fallback_smoothing_length > 0.0f) ||
        (count > 0 && smoothing_length == nullptr)) {
        throw std::runtime_error("invalid SPH inactive-tail serialization request");
    }
    for (int32_t i = 0; i < count; ++i) {
        if (!(std::isfinite(smoothing_length[i]) && smoothing_length[i] > 0.0f)) {
            smoothing_length[i] = fallback_smoothing_length;
        }
        if (distension != nullptr &&
            !(std::isfinite(distension[i]) && distension[i] >= 1.0f)) {
            distension[i] = 1.0f;
        }
    }
}

// Device-backed counterpart used by single-GPU output and checkpoint writers.
// It copies and repairs only [live_particles, total_particles), preserving
// every valid stored tail value and leaving active rows untouched.
void sanitize_inactive_sph_tail_for_serialization(
    ParticleSoA& p, int32_t live_particles, int32_t total_particles,
    float fallback_smoothing_length);
void allocate_contact_history(ContactHistoryBuffer& ch, int32_t num_particles, int32_t max_contacts_per_particle = 32);
void resize_contact_history_preserving(ContactHistoryBuffer& ch, int32_t num_particles);
void resize_wall_contact_history_preserving(WallContactHistoryBuffer& wch, int32_t num_particles);
void allocate_wall_soa(WallSoA& w, int32_t num_walls);

// Memory deallocation
void free_particle_soa(ParticleSoA& p);
void free_contact_history(ContactHistoryBuffer& ch);
void free_wall_soa(WallSoA& w);
void free_simulation_state(SimulationState& state);

// Particle loading (strict GRANIT unified v3 only)
int32_t load_particles_from_binary(ParticleSoA& particles, const std::string& filepath, const SimConfig& config, int32_t extra_capacity = 0);

// Clump capacity computation
int32_t compute_total_clump_sub_spheres(const SimConfig& config);

// Wall initialization
void initialize_walls(WallSoA& walls, const SimConfig& config);

// Input-derived DEM size bounds and material-aware timestep calculation.
std::pair<float, float> derive_particle_radius_bounds(const ParticleSoA& particles,
                                                       int32_t num_particles);
double calculate_critical_timestep(const SimConfig& config,
                                   const ParticleSoA& particles,
                                   const MaterialDatabase& materials,
                                   int32_t num_particles);

// Main initialization function
SimulationState initialize_system(const SimConfig& config);

} // namespace cudadem

#endif // CUDADEM_SYSTEM_INITIALIZE_H

#ifndef CUDADEM_COMMAND_LINE_INTERFACE_H
#define CUDADEM_COMMAND_LINE_INTERFACE_H

#include <cuda_runtime.h>
#include "src/core/sph/sph_stage_math.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>

// yaml-cpp for YAML parsing (host only)
// Include yaml-cpp for host code, but not for device code
// __CUDA_ARCH__ is only defined during device compilation, not host compilation
#ifndef __CUDA_ARCH__
#include <yaml-cpp/yaml.h>
#endif

#include "../../domain/rcb_work_model.h"
#include "../material/material_database.h"

// =============================================================================
// GPU-Compatible Configuration Structures (POD types only)
// =============================================================================
// IMPORTANT: All structs must be Plain Old Data (POD) for CUDA constant memory
// - No virtual functions
// - No std::string (use fixed-size char arrays)
// - No pointers to dynamic memory
// - No STL containers
// =============================================================================

namespace cudadem {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int MAX_NAME_LENGTH = 64;
constexpr int MAX_PATH_LENGTH = 256;
constexpr int MAX_WALLS = 8;
constexpr int MAX_ACCEL_FIELDS = 4;
constexpr int MAX_OUTPUT_FIELDS = 16;

// -----------------------------------------------------------------------------
// Enumerations for type-safe configuration
// -----------------------------------------------------------------------------
enum class SimulationType : uint8_t {
    SURFACE_SLOPE = 0,
    ASTEROID_SHAKE = 1
};

// A run has exactly one particle physics implementation.  This replaces the
// historical independent SPH enabled flag and prevents unsupported hybrids.
enum class SimulationPhysics : uint8_t {
    DEM = 0,
    SPH = 1
};

enum class CoordinateSystem : uint8_t {
    SURFACE = 0,
    MASS_CENTER = 1,
    ROTATING = 2       // Frame co-rotating at constant Ω; inertial forces applied
};

enum class ContactModel : uint8_t {
    RESTITUTION = 0,          // "resti"
    VISCOUS = 1,             // "visco"
    HERTZ_MINDLIN_RESTI = 0,  // Alias for RESTITUTION
    HERTZ_MINDLIN_VISCO = 1   // Alias for VISCOUS
};

enum class WallType : uint8_t {
    NONE = 0,
    PLAIN = 1,
    PERIODICAL = 2,
    MOVING = 3,
    MESH = 4,
    FREE = 5,
    BONDED_RECT = 6,   // finite rigid box with 6-DOF GPU dynamics
    FINITE_PLAIN = 7,  // finite-area plain wall (rectangular shelf)
    FUNNEL = 8,        // Conical frustum (interior contact)
    SPHERE = 9         // Sphere wall (prescribed trajectory, full two-body contact)
};

enum class LoadMethod : uint8_t {
    ONCE = 0,          // Load all data at start
    SEQUENTIAL = 1     // Load in batches during simulation
};

enum class InterpolationMethod : uint8_t {
    NONE = 0,
    LINEAR = 1,
    CUBIC = 2
};

enum class DampingType : uint8_t {
    NONE = 0,
    VISCOUS = 1,
    LOCAL = 2
};

enum class TerminationCondition : uint8_t {
    TIME = 0,
    ENERGY_THRESHOLD = 1,
    ADAPTIVE_YORP = 2
};

enum class QueueOverflowPolicy : uint8_t {
    RESIZE = 0,
    SINGLE_PASS = 1
};

enum class SelfGravityMethod : uint8_t {
    DIRECT    = 0,   // "direct"  — O(N^2) particle-pair gravity
    BFFMM     = 2    // "unified_fmm" — body-fitted FMM (Karras octree + warp-sync M2L)
};

// The query-particle mode reproduces SPHSOL's directional h_i softening for
// comparison runs.  Symmetric mean is the conservative GRANIT default.
enum class DirectSphSofteningMode : uint8_t {
    SYMMETRIC_MEAN = 0,
    QUERY_PARTICLE = 1
};

// -----------------------------------------------------------------------------
// Vector Types (POD compatible)
// -----------------------------------------------------------------------------
struct Vec3d {
    double x, y, z;

    __host__ __device__ Vec3d() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Mat3d {
    double data[9];  // Row-major: [0,1,2] = row0, [3,4,5] = row1, [6,7,8] = row2

    __host__ __device__ Mat3d() {
        // Identity matrix
        data[0] = 1; data[1] = 0; data[2] = 0;
        data[3] = 0; data[4] = 1; data[5] = 0;
        data[6] = 0; data[7] = 0; data[8] = 1;
    }
};

// -----------------------------------------------------------------------------
// Time Configuration
// -----------------------------------------------------------------------------
struct TimeConfig {
    double start_time;           // Simulation start (seconds from epoch)
    double stop_time;            // Simulation end
    double output_step;          // Output interval (seconds)
    double dt_factor;            // Verlet timestep = dt_factor * dt_critical (typically 0.1-0.2)
    double dt_upper_bound;       // Optional explicit ceiling; <= 0 means no ceiling

    __host__ __device__ TimeConfig()
        : start_time(0), stop_time(0), output_step(60.0), dt_factor(0.2),
          dt_upper_bound(0.0) {}
};

// -----------------------------------------------------------------------------
// Local planetary-ring shearing sheet
// -----------------------------------------------------------------------------
enum class ShearInputVelocity : uint8_t {
    TOTAL = 0,
    PECULIAR = 1
};

struct ShearBoxConfig {
    bool enabled;
    double omega;                   // Orbital angular frequency (rad/s)
    double shear_parameter;         // q = -d ln(Omega) / d ln(r)
    double vertical_frequency;      // Vertical oscillator frequency (rad/s)
    double time_origin;             // Epoch at which the shear phase is zero
    ShearInputVelocity input_velocity;
    double orbital_dt_factor;       // Max resolved fraction of orbital timescales
    int32_t gravity_layers_x;       // Complete finite replica layers in radial x
    int32_t gravity_layers_y;       // Complete finite replica layers in azimuthal y
    double gravity_cutoff_radius;   // <= 0 disables the optional physical cutoff
    double force_balance_warn_threshold;

    __host__ __device__ ShearBoxConfig()
        : enabled(false), omega(0.0), shear_parameter(1.5),
          vertical_frequency(0.0), time_origin(0.0),
          input_velocity(ShearInputVelocity::TOTAL), orbital_dt_factor(0.05),
          gravity_layers_x(1), gravity_layers_y(1),
          gravity_cutoff_radius(0.0), force_balance_warn_threshold(1.0e-5) {}
};

// -----------------------------------------------------------------------------
// Particle Configuration
// -----------------------------------------------------------------------------
struct ParticleConfig {
    // Radius distribution
    float radius_min;            // meters
    float radius_max;            // meters
    float size_dist_slope;       // alpha (typically -3.0)

    // Material properties (Hertz-Mindlin)
    float youngs_modulus;       // E (Pa or kg/m/s²)
    float poisson_ratio;         // ν (typical 0.3)
    float density_bulk;          // kg/m³
    float shape_parameter;       // beta

    // Contact model
    ContactModel contact_model;

    // Restitution
    float restitution_normal;    // εn
    float restitution_tangent;   // εt

    // Friction coefficients
    float friction_static;       // μs
    float friction_rolling;      // μr
    float friction_twisting;     // μt
    float friction_angle_deg;    // φf in degrees

    // Cohesion (Scheeres model)
    bool cohesion_enabled;
    float cohesion_range;        // Range parameter
    float cohesion_strength;     // Pa

    // Internal aliases/parameters used by kernels
    float damping_normal;         // Alias for restitution_normal
    float beta_shape;            // Alias for shape_parameter
    float visco_A;              // Viscous damping parameter

    __host__ __device__ ParticleConfig()
        : radius_min(0.0596f), radius_max(0.1786f), size_dist_slope(-3.0f),
          youngs_modulus(1.58e9f), poisson_ratio(0.3f),
          density_bulk(2200.0f), shape_parameter(0.7f),
          contact_model(ContactModel::RESTITUTION),
          restitution_normal(0.55f), restitution_tangent(0.55f),
          friction_static(1.0f), friction_rolling(1.05f),
          friction_twisting(1.3f), friction_angle_deg(35.1f),
          cohesion_enabled(true), cohesion_range(1.0f), cohesion_strength(1.0f),
          damping_normal(0.55f), beta_shape(0.7f), visco_A(0.0f) {}
};

// -----------------------------------------------------------------------------
// Domain Configuration
// -----------------------------------------------------------------------------
struct DomainConfig {
    // Patch dimensions (for surface simulation)
    Vec3f patch_size;            // length, width, height in meters

    // Domain bounding box (for neighbor search and periodic boundaries)
    Vec3f box_min;               // Minimum corner of domain
    Vec3f box_max;               // Maximum corner of domain
    bool auto_bounds;            // If true, compute bounds from particle positions

    // Surface coordinate transform
    Vec3d surface_origin;        // Position from mass center
    Mat3d rotation_matrix;       // Body-fixed to surface-fixed

    // Grid resolution for gravity field
    Vec3f grid_resolution;       // Cell size in meters

    __host__ __device__ DomainConfig()
        : patch_size(8.0f, 8.0f, 3.0f),
          box_min(-4.0f, -4.0f, 0.0f),    // Default: centered patch, z starts at 0
          box_max(4.0f, 4.0f, 3.0f),       // Default: matches patch_size
          auto_bounds(false),
          grid_resolution(0.25f, 0.25f, 0.25f) {}
};

// -----------------------------------------------------------------------------
// Wall Configuration
// -----------------------------------------------------------------------------
struct WallConfig {
    int32_t id;
    int32_t material_id;          // required YAML material id for every wall
    WallType type;
    bool enabled;

    Vec3d position;              // Wall position/origin
    Vec3d normal;                // Normal direction (for plain walls)
    Vec3d velocity;              // For moving walls

    // Periodical boundary
    uint8_t axis;                // 0=x, 1=y, 2=z
    float bound_min;
    float bound_max;

    // Data path for moving walls (fixed-size)
    char velocity_data_path[MAX_PATH_LENGTH];

    // Canonical prescribed rigid geometry motion.  RIGIDPS2 is the only
    // accepted file trajectory for contact walls; legacy velocity files are
    // intentionally rejected by the YAML reader.
    char rigid_pose_path[MAX_PATH_LENGTH];
    bool surface_only_motion;       // Explicit conveyor/surface velocity; geometry remains fixed.
    Vec3d surface_velocity;         // World-frame m/s, valid only when surface_only_motion is true.

    // Mesh wall specific
    char mesh_path[MAX_PATH_LENGTH];          // Path to OBJ file
    char motion_data_path[MAX_PATH_LENGTH];   // Optional motion data
    Vec3d angular_velocity;                   // For rotating mesh

    // Bonded rectangular prism fields
    float half_extents[3];        // Half-extents in body frame (m), default 1m each
    float bonded_mass;             // Wall mass (kg), default 1 kg
    float inertia_body[3];         // Principal moments of inertia (kg·m²), default 1 each
    Vec3d initial_omega_body;      // Initial angular velocity, body frame (rad/s)
    Vec3d euler_zyx_deg;           // Initial orientation as ZYX Euler angles (degrees)

    // Funnel wall specific
    float funnel_radius_bottom;    // Radius at base (m)
    float funnel_radius_top;       // Radius at top (m)
    float funnel_length;           // Frustum height along axis (m)

    // Sphere wall specific
    float sphere_radius;
    float sphere_mass;

    char sphere_trajectory_path[MAX_PATH_LENGTH];

    __host__ __device__ WallConfig()
        : id(-1), material_id(-1), type(WallType::NONE), enabled(false),
          axis(0), bound_min(0), bound_max(0) {
        velocity_data_path[0] = '\0';
        rigid_pose_path[0] = '\0';
        surface_only_motion = false;
        mesh_path[0] = '\0';
        motion_data_path[0] = '\0';
        half_extents[0] = half_extents[1] = half_extents[2] = 1.0f;
        bonded_mass = 1.0f;
        inertia_body[0] = inertia_body[1] = inertia_body[2] = 1.0f;
        // initial_omega_body and euler_zyx_deg zero-init via Vec3d default ctor
        funnel_radius_bottom = 0.0f;
        funnel_radius_top = 0.0f;
        funnel_length = 0.0f;
        sphere_radius = 0.0f;
        sphere_mass = 0.0f;
        sphere_trajectory_path[0] = '\0';
    }
};

// -----------------------------------------------------------------------------
// Acceleration Field Configuration
// -----------------------------------------------------------------------------
struct AccelFieldConfig {
    int32_t id;
    bool enabled;
    char name[MAX_NAME_LENGTH];
    char data_path[MAX_PATH_LENGTH];

    LoadMethod load_method;
    float time_interval;         // seconds
    int32_t batch_size;          // Frames to load at once

    bool time_varying;
    bool space_varying;

    InterpolationMethod time_interp;
    InterpolationMethod space_interp;

    __host__ __device__ AccelFieldConfig()
        : id(-1), enabled(false), load_method(LoadMethod::SEQUENTIAL),
          time_interval(0.01f), batch_size(100),
          time_varying(true), space_varying(true),
          time_interp(InterpolationMethod::LINEAR),
          space_interp(InterpolationMethod::LINEAR) {
        name[0] = '\0';
        data_path[0] = '\0';
    }
};

// -----------------------------------------------------------------------------
// External Acceleration Configuration
// -----------------------------------------------------------------------------
// YORP Configuration
struct YorpConfig {
    bool enabled;
    int32_t mode;            // 1: Increment, 2: Continuous, 3: Prescribed spin period

    // Common mode params
    double start_time;
    double stop_time;
    double omega_init;       // Initial YORP omega (rad/s)

    // Mode 1 params
    double increment;        // Omega increment (rad/s)
    double dt_check;         // Time interval (s)

    // Mode 2 params
    double epsilon;          // Acceleration rate (rad/s²)

    // Mode 3 params: prescribed spin period (3-stage)
    // Stage 0: T_start → T_inter, duration fast_spinup_time
    // Stage 1: T_inter → T_final, duration slow_spinup_time
    // Stage 2: maintain at T_final
    double T_start_period;   // Initial spin period (hours)
    double T_inter_period;   // Intermediate spin period (hours)
    double T_final_period;   // Final spin period (hours)
    double fast_spinup_time; // Stage 0 duration (seconds)
    double slow_spinup_time; // Stage 1 duration (seconds)

    // Adaptive YORP control.  This is deliberately expressed in physical
    // seconds and specific energy, never in integration-step counts.
    bool adaptive;
    double check_interval_seconds;
    double pause_specific_energy_j_per_kg;
    double resume_specific_energy_j_per_kg;
    int32_t pause_confirmation_checks;
    int32_t resume_confirmation_checks;
    double axis_ratio_drop_fraction;
    double epsilon_reduction_fraction_per_cycle;
    double minimum_epsilon_fraction;
    double max_omega_rad_s;

    __host__ __device__ YorpConfig()
        : enabled(false), mode(0), start_time(0.0), stop_time(0.0),
          omega_init(0.0), increment(0.0), dt_check(1.0), epsilon(0.0),
          T_start_period(5.0), T_inter_period(3.0), T_final_period(2.26),
          fast_spinup_time(7000.0), slow_spinup_time(140000.0),
          adaptive(false), check_interval_seconds(50.0),
          pause_specific_energy_j_per_kg(0.0), resume_specific_energy_j_per_kg(0.0),
          pause_confirmation_checks(1), resume_confirmation_checks(1),
          axis_ratio_drop_fraction(0.01),
          epsilon_reduction_fraction_per_cycle(0.0),
          minimum_epsilon_fraction(1.0), max_omega_rad_s(0.0) {}
};

// Distributed adaptive-depth policy is deliberately represented by enums, not
// strings: SimConfig is copied verbatim to device constant memory.
enum class AdaptiveDepthSecondaryObjective : uint8_t {
    MeanEpochTime = 0,
};

enum class AdaptiveDepthTrialOrder : uint8_t {
    RoundRobin = 0,
};

struct AdaptiveDepthMultiGPUConfig {
    float local_count_reset_fraction;
    AdaptiveDepthSecondaryObjective secondary_objective;
    AdaptiveDepthTrialOrder trial_order;

    __host__ __device__ AdaptiveDepthMultiGPUConfig()
        : local_count_reset_fraction(0.10f),
          secondary_objective(AdaptiveDepthSecondaryObjective::MeanEpochTime),
          trial_order(AdaptiveDepthTrialOrder::RoundRobin) {}
};

// Timing-only adaptive tree-depth controller.  Keeping this POD configuration
// nested under UnifiedFMMConfig makes it safe to copy with SimConfig to device
// constant memory and gives the runtime one stable configuration contract.
struct AdaptiveDepthConfig {
    bool  enabled;                       // normalized true for omitted/auto depth; false for fixed integer
    bool  initial_depth_auto;            // tree_depth: auto; startup probe chooses the seed
    bool  tree_depth_was_omitted;        // omitted auto is permitted only for a fresh launch
    int   min_depth;                     // shallowest captured candidate (default: 6)
    int   max_depth;                     // deepest captured candidate (default: 12)
    int   trial_interval_evaluations;     // selected-depth evaluations between trials
    int   trial_warmup_evaluations;       // untimed candidate evaluations per trial
    int   trial_samples;                  // timed candidate evaluations per trial
    float min_improvement_fraction;       // required relative speedup (default: 0.02)
    int   confirmation_trials;            // consecutive wins required to switch
    float capacity_headroom;              // multiplier on sparse startup probe
    float capacity_growth_trigger;        // request growth at this utilization
    AdaptiveDepthMultiGPUConfig multigpu; // distributed host-controller policy

    __host__ __device__ AdaptiveDepthConfig()
        : enabled(true), initial_depth_auto(true), tree_depth_was_omitted(true),
          min_depth(6), max_depth(12),
          trial_interval_evaluations(32), trial_warmup_evaluations(1),
          trial_samples(4), min_improvement_fraction(0.02f),
          confirmation_trials(2), capacity_headroom(2.0f),
          capacity_growth_trigger(0.75f) {}
};

// Unified FMM Configuration (selected via gravity_method == SelfGravityMethod::BFFMM)
struct UnifiedFMMConfig {
    // Accuracy
    float theta_far;                    // MAC threshold (default: 0.8)
    float softening_factor;             // eps = radius_min * softening_factor (default: 0.01)
    double gravitational_constant;      // G (default: 6.67430e-11)

    // Tree structure.  BFFMM is deliberately explicit: 2 selects the X-Y
    // quadtree and 3 selects the octree.  There is no topology auto-selector.
    int tree_dimension;
    int tree_depth;                     // fixed/explicit initial depth; 0 when initial depth is auto
    AdaptiveDepthConfig adaptive_depth; // optional timing-only neighboring-depth controller

    // Memory allocation
    float particle_overallocate_factor; // particle buffer headroom (default: 1.2)

    int min_nodes_estimate;             // absolute floor for node count estimate (default: 65536)

    // Global
    bool enable_global_projection;       // enforce Newton's 3rd law via global projection (default: true)
    bool verbose;                        // control BFFMM print output (default: false)
    // Direct kernel thresholds
    int   direct_threshold_leaf;          // P2P direct evaluation leaf threshold (default: 16)
    int   direct_threshold_internal;      // P2P direct evaluation internal threshold (default: 64)

    __host__ __device__ UnifiedFMMConfig()
        : theta_far(0.8f), softening_factor(0.01f), gravitational_constant(6.67430e-11),
          tree_dimension(3), tree_depth(0),
          particle_overallocate_factor(1.2f), min_nodes_estimate(65536),
          enable_global_projection(true), verbose(false),
          direct_threshold_leaf(16), direct_threshold_internal(64) {}
};

// -----------------------------------------------------------------------------
// External Acceleration Configuration
// -----------------------------------------------------------------------------
// Prescribed massive bodies used by global, inertial-frame DEM problems.
// These are intentionally separate from contact walls: they supply analytic
// gravity only, in double precision, and never receive a back reaction.
struct ExternalBodiesConfig {
    bool enabled;
    bool saturn_enabled;
    Vec3d saturn_position;
    double saturn_mass;
    double saturn_absorb_radius;  // particle-centre crossing, m; 0 disables

    bool satellite_enabled;
    double satellite_mass;
    double satellite_orbit_radius;
    double satellite_phase;       // radians at simulation.time.start
    bool satellite_prograde;

    __host__ __device__ ExternalBodiesConfig()
        : enabled(false), saturn_enabled(false), saturn_position(),
          saturn_mass(0.0), saturn_absorb_radius(0.0),
          satellite_enabled(false), satellite_mass(0.0),
          satellite_orbit_radius(0.0), satellite_phase(0.0),
          satellite_prograde(true) {}
};

struct DirectGravityConfig {
    double gravitational_constant;
    DirectSphSofteningMode sph_softening_mode;

    __host__ __device__ DirectGravityConfig()
        : gravitational_constant(6.67430e-11),
          sph_softening_mode(DirectSphSofteningMode::SYMMETRIC_MEAN) {}
};

struct AccelerationConfig {
    // Inertial acceleration (centrifugal, Coriolis, Euler)
    bool enable_inertial;
    bool inertial_constant_omega;    // true → use RotatingFrameConfig::omega, skip binary file
    char inertial_data_path[MAX_PATH_LENGTH];
    LoadMethod inertial_load_method;
    float inertial_time_interval;
    InterpolationMethod inertial_interp;

    // Tidal acceleration
    bool enable_tidal;
    char tidal_data_path[MAX_PATH_LENGTH];
    LoadMethod tidal_load_method;
    float tidal_time_interval;
    InterpolationMethod tidal_interp;
    double perturber_mass;       // kg (Earth mass for Apophis flyby)

    // Self-gravity
    bool enable_self_gravity;
    SelfGravityMethod gravity_method;    // BFFMM (default) or DIRECT
    int gravity_update_rate;       // RESPA inner steps per gravity macro-step (default 2; 1 = disabled)
    double gravity_macro_dt;       // requested max macro-step seconds; 0 = use gravity_update_rate

    float displacement_threshold; // Sphere-wall gravity softening fraction (legacy name)
    DirectGravityConfig direct_gravity;

    // Static central monopole plus an optional analytic circular point
    // satellite. This is the global-ring path; it is not a contact wall.
    ExternalBodiesConfig external_bodies;

    // YORP Effect
    YorpConfig yorp;

    // Unified FMM gravity
    UnifiedFMMConfig unified_fmm;

    // Acceleration fields
    int32_t num_accel_fields;
    AccelFieldConfig fields[MAX_ACCEL_FIELDS];

    // Acceleration limiting for numerical stability
    bool enable_accel_limit;        // Toggle acceleration magnitude limiting
    float max_accel_magnitude;      // Maximum acceleration magnitude (m/s²)

    __host__ __device__ AccelerationConfig()
        : enable_inertial(false), inertial_constant_omega(false),
          inertial_load_method(LoadMethod::ONCE),
          inertial_time_interval(60.0f), inertial_interp(InterpolationMethod::LINEAR),
          enable_tidal(false), tidal_load_method(LoadMethod::ONCE),
          tidal_time_interval(60.0f), tidal_interp(InterpolationMethod::LINEAR),
          perturber_mass(5.972e24),  // Earth mass
          enable_self_gravity(false), gravity_method(SelfGravityMethod::BFFMM), gravity_update_rate(2), gravity_macro_dt(0.0),
          displacement_threshold(0.3f), num_accel_fields(0),
          enable_accel_limit(false), max_accel_magnitude(1.0e6f) {
        inertial_data_path[0] = '\0';
        tidal_data_path[0] = '\0';
    }
};

// -----------------------------------------------------------------------------
// SPH Parameters Configuration
// -----------------------------------------------------------------------------
struct SPHConfig {
    bool enabled;                  // Toggle SPH on/off
    float rest_density;            // rho_0 kg/m3
    float stiffness;               // B Pa (Tait EOS stiffness)
    float viscosity;               // alpha (Monaghan artificial viscosity)
    float smoothing_length;        // h m
    int32_t num_neighbors;         // target neighbor count

    // --- EOS (Tillotson) ---
    float tillotson_a;
    float tillotson_b;
    float tillotson_A;
    float tillotson_B;
    float tillotson_alpha;
    float tillotson_beta;
    float tillotson_E0;
    float tillotson_Eiv;
    float tillotson_Ecv;
    // --- Shear Strength ---
    bool  use_strength;            // Enable elasto-plastic strength
    float shear_modulus;           // G Pa (elastic shear modulus)
    float yield_strength_Y0;       // Y0 Pa (von Mises cohesive limit)
    bool  use_lundborg;            // Use Lundborg model (pressure-dependent)
    float lundborg_Yi0;            // Intact cohesion
    float lundborg_Yd0;            // Damaged cohesion
    float lundborg_Ym;             // Maximum yield strength
    float lundborg_mui;            // Intact friction coefficient
    float lundborg_mud;            // Damaged friction coefficient

    // --- Damage realization provenance ---
    // Grady-Kipp constitutive values live exclusively in the material database.
    uint64_t damage_seed;          // deterministic Weibull-flaw initialization provenance

    // --- Porosity (P-alpha) ---
    bool  use_porosity;            // Enable P-alpha compaction model
    float alpha_initial;           // alpha_0 initial distension ratio (>=1)
    float P_elastic_limit;         // P_e Pa (onset of pore crushing)
    float P_solid_crush;           // P_s Pa (fully compacted pressure)
    float crushing_exponent;       // n  polynomial exponent in crush curve

    // --- Kernel Gradient Correction ---
    bool use_kernel_correction;    // Enable CSPH/Bonet-Lok correction matrix

    // --- Artificial Stress (tension instability) ---
    bool  use_artificial_stress;   // Enable Swegle-Gray artificial stress
    float artstr_epsilon;          // epsilon ~0.3 (artificial stress magnitude)
    float artstr_n;                // n ~4 (exponent W/W_delta)

    // --- Gravity Softening Kernel ---
    float gravity_softening_len;   // h_soft m (cubic-spline gravity kernel, 0=off)
    int32_t density_update_mode;
    bool enable_shepard_filter;

    // --- CFL Timestep ---
    float cfl_number;             // Signal-speed CFL coefficient (finite in (0, 1])
    float dt_min;                 // Minimum timestep (s)
    float dt_max;                 // Maximum timestep (s)
    float kernel_cutoff_factor;   // Support radius = factor * h (typically 2.0)
    float hsml_min;               // Minimum smoothing length (m)
    float hsml_max;               // Maximum smoothing length (m)

    // --- Relaxation-only numerical controls ---
    // These switches do not alter the immutable material database.  They let
    // a hydrostatic body settle before the constitutive operators are restored
    // from the same checkpoint for the production-physics control.
    bool relaxation_disable_strength;
    bool relaxation_disable_damage;

    // --- Optional RELAX / SHOCK / LATE sequence ---
    int32_t stage_count = 0;      // Zero preserves the legacy single-stage path
    SphStageParameters stages[3];
    bool track_history = false;   // Peak pressure, cold/thermal energy, crack area

    // --- Single-GPU continuum boundaries ---
    int32_t fixed_part_count = 0;
    int32_t fixed_parts[16] = {};
    bool symmetry_enabled = false;
    double symmetry_normal[3] = {1.0, 0.0, 0.0};
    double symmetry_offset = 0.0; // n.x + offset = 0; real particles on positive side

    // Resolved input path (set internally from io.input.particle_file).
    char input_path[MAX_PATH_LENGTH]; // Path to SPH particle input

    __host__ __device__ SPHConfig()
        : enabled(false), rest_density(1000.0f), stiffness(50.0f),
          viscosity(0.1f), smoothing_length(0.01f), num_neighbors(64),
          tillotson_a(0.5f), tillotson_b(1.5f), tillotson_A(2.67e10f),
          tillotson_B(2.67e10f), tillotson_alpha(5.0f), tillotson_beta(5.0f),
          tillotson_E0(4.87e8f), tillotson_Eiv(4.72e6f), tillotson_Ecv(1.82e7f),
          use_strength(false), shear_modulus(1.0e8f), yield_strength_Y0(1.0e5f),
          use_lundborg(false), lundborg_Yi0(1.0e5f), lundborg_Yd0(0.0f),
          lundborg_Ym(1.0e9f), lundborg_mui(0.65f), lundborg_mud(0.65f),
          damage_seed(UINT64_C(12345678)),
          use_porosity(false), alpha_initial(1.3f),
          P_elastic_limit(1.0e5f), P_solid_crush(5.0e8f), crushing_exponent(3.0f),
          use_kernel_correction(true),
          use_artificial_stress(false), artstr_epsilon(0.3f), artstr_n(4.0f),
          density_update_mode(0), enable_shepard_filter(false),
          gravity_softening_len(0.0f),
          // CFL timestep defaults
          cfl_number(0.3f), dt_min(1.0e-10f), dt_max(1.0e-3f),
          kernel_cutoff_factor(2.0f),
          hsml_min(1.0e-6f), hsml_max(1.0f),
          relaxation_disable_strength(false), relaxation_disable_damage(false)
    {
        input_path[0] = '\0';
    }
};

// -----------------------------------------------------------------------------
// Neighbor Search Configuration
// -----------------------------------------------------------------------------
enum class NeighborMethod : uint8_t {
    CELL_LINKED_LIST = 0,
    LBVH = 1,
    DIRECT = 2  // Brute-force O(N²) for validation
};

enum class NeighborBackend : uint8_t {
    AUTO = 0,  // Select RT when available for the supported LBVH workload
    CUDA = 1,  // Existing CUDA-core implementation
    RT = 2     // Require the OptiX RT-core implementation
};

struct NeighborConfig {
    NeighborMethod method;       // Which neighbor search algorithm to use
    NeighborBackend backend;     // Execution backend; RT supports single-GPU nonperiodic DEM/SPH LBVH
    float cell_size_factor;      // L_cell > max(2*R_max)*(1+ε)
    int32_t update_frequency;    // Steps between rebuild
    bool verlet_buffer_enabled;  // Use Verlet list (skin depth)
    float verlet_skin;           // Skin depth in meters

    // LBVH topology caching
    float lbvh_skin_factor;        // threshold = max_radius * skin_factor / 2; default 0.3
    float lbvh_rebuild_threshold;  // absolute override in meters; 0.0 = use skin_factor

    __host__ __device__ NeighborConfig()
        : method(NeighborMethod::LBVH), backend(NeighborBackend::AUTO),
          cell_size_factor(1.1f), update_frequency(1),
          verlet_buffer_enabled(true), verlet_skin(0.01f), lbvh_skin_factor(0.3f), lbvh_rebuild_threshold(0.0f) {}
};

// -----------------------------------------------------------------------------
// Damping Configuration
// -----------------------------------------------------------------------------
struct DampingConfig {
    bool enabled;
    DampingType type;
    float coefficient;

    __host__ __device__ DampingConfig()
        : enabled(false), type(DampingType::LOCAL), coefficient(0.7f) {}
};

// -----------------------------------------------------------------------------
// I/O Configuration
// -----------------------------------------------------------------------------
struct IOConfig {
    char input_particle_path[MAX_PATH_LENGTH];
    char output_directory[MAX_PATH_LENGTH];
    bool validate_input_types;

    // Checkpoint
    bool checkpoint_enabled;
    bool checkpoint_force_fresh;  // ignore prior checkpoints for this launch
    int32_t checkpoint_interval;
    char checkpoint_path[MAX_PATH_LENGTH];
    char checkpoint_resume_from[MAX_PATH_LENGTH]; // explicit checkpoint; empty = fresh
    bool checkpoint_rebootstrap_forces; // recompute force after a derived-state resume

    // Contact dump for validation
    bool dump_contacts;

    // Optional output sidecars

    bool output_gravity_force;    // Write per-particle gravity force sidecar (for LET single-vs-multi A/B)
    bool output_initial_state;    // Write output_step_0000000000.bin after force bootstrap
    int32_t scalar_interval;      // closed-step scalar receipt cadence; 0 disables
    int32_t output_schedule_count; // exact physical-time boundaries; 0 uses output_step
    double output_schedule_times[16];
    int32_t timing_warmup_steps;
    int32_t timing_measure_steps;
    char timing_receipt_path[MAX_PATH_LENGTH];

    __host__ __device__ IOConfig()
        : validate_input_types(true), checkpoint_enabled(true), checkpoint_force_fresh(false),
          checkpoint_interval(1000), checkpoint_rebootstrap_forces(false), dump_contacts(false),
          output_gravity_force(false), output_initial_state(false),
          scalar_interval(0), output_schedule_count(0), timing_warmup_steps(0), timing_measure_steps(0) {
        input_particle_path[0] = '\0';
        output_directory[0] = '\0';
        checkpoint_path[0] = '\0';
        checkpoint_resume_from[0] = '\0';
        timing_receipt_path[0] = '\0';
        for (double& value : output_schedule_times) value = 0.0;
    }
};

// -----------------------------------------------------------------------------
// GPU Configuration
// -----------------------------------------------------------------------------
struct GPUConfig {
    int32_t device_id;
    int32_t block_size;

    __host__ __device__ GPUConfig()
        : device_id(0), block_size(256) {}
};

// -----------------------------------------------------------------------------
// Multi-GPU Configuration
// -----------------------------------------------------------------------------
struct MultiGPUConfig {
    bool enabled;                    // Enable multi-GPU mode
    int32_t max_gpus;                // Maximum number of GPUs to use
    int32_t partition_depth = 0;    // 0 resolves once to initial computation depth
    bool morton_ownership = true;   // Shared whole-leaf ownership for DEM and FMM

    float halo_width_factor;         // Multiplier for 2*R_max for halo region
    float buffer_capacity_factor;    // Capacity = local_particles * factor (>= 2.0)

    // --- LET communication ---
    bool let_use_one_shot_push;      // Use one-shot push for LET particle transfer
    // Share only destination-mask buckets whose avoided re-serialization meets
    // this threshold.  Zero permits every eligible multi-destination bucket.
    uint64_t shared_bucket_min_saved_bytes;
    // Bounded number of shared mask buckets per sender/epoch; zero leaves the
    // exact per-peer residual path active while disabling bucket sharing.
    uint8_t max_shared_buckets;

    // --- RCB cost-weighted balancing (SP1) ---
    RCBCostModel rcb_cost_model;     // adaptive is the default; alpha/beta/gamma are legacy-only
    int32_t rcb_calibration_sample_interval;
    float   rcb_calibration_half_life;
    float   rcb_min_predicted_improvement;
    bool    rcb_calibration_verbose;
    float   rcb_alpha;               // legacy base integration weight per particle
    float   rcb_beta;                // legacy weight per active contact
    float   rcb_gamma;               // legacy weight per FMM neighbor
    int32_t rcb_num_bins;            // histogram resolution for weighted median
    float   rebalance_threshold;     // trigger when T_max/T_avg exceeds this
    int32_t rebalance_min_interval;  // min steps between rebalances (anti-thrash)

    __host__ __device__ MultiGPUConfig()
        : enabled(false), max_gpus(8), halo_width_factor(1.5f), buffer_capacity_factor(2.0f),
          let_use_one_shot_push(true),
          shared_bucket_min_saved_bytes(65536), max_shared_buckets(8),
          rcb_cost_model(RCBCostModel::Adaptive),
          rcb_calibration_sample_interval(8), rcb_calibration_half_life(8.0f),
          rcb_min_predicted_improvement(0.05f), rcb_calibration_verbose(false),
          rcb_alpha(1.0f), rcb_beta(0.1f), rcb_gamma(0.0f),
          rcb_num_bins(1024), rebalance_threshold(1.15f),
          rebalance_min_interval(50) {}
};

// =============================================================================
// Clump (Multi-Sphere Rigid Body) Configuration
// =============================================================================

struct ClumpTemplate {
    char    name[64];
    bool    has_inertia;
    // Symmetric body-frame inertia: xx, yy, zz, xy, xz, yz.
    double  inertia[6];
    bool    has_multipole;
    char    multipole_basis[32];
    // Compact trace_free_p3_v1 layout: FMM slots 0..15, then 25..27.
    double  multipole_p3[19];

    __host__ __device__ ClumpTemplate() : has_inertia(false), inertia{},
                                           has_multipole(false), multipole_p3{} {
        name[0] = '\0';
        multipole_basis[0] = '\0';
    }
};

struct ClumpInstance {
    int32_t id;                 // Persistent on-disk ID, contiguous 1..K.
    char    template_name[64];
    bool    has_com;
    double  com[3];             // Optional world-frame CoM override.
    bool    has_orientation;
    double  orientation[4];     // Unit quaternion (w, x, y, z).

    __host__ __device__ ClumpInstance()
        : id(0), has_com(false), com{}, has_orientation(false),
          orientation{1.0, 0.0, 0.0, 0.0} {
        template_name[0] = '\0';
    }
};

struct ClumpConfig {
    int32_t num_templates;
    int32_t num_instances;
    // Host-owned immutable metadata. These pointers are copied as inert bytes
    // with SimConfig but are never dereferenced by device code.
    ClumpTemplate* templates;
    ClumpInstance* instances;
    // Hash of canonicalized clump YAML metadata.  It is persisted in unified
    // v3 checkpoints so a restart cannot bind member IDs to different bodies.
    uint64_t clump_metadata_fingerprint;

    __host__ __device__ ClumpConfig()
        : num_templates(0), num_instances(0), templates(nullptr), instances(nullptr),
          clump_metadata_fingerprint(0) {}
};

// =============================================================================
// Rotating Frame Configuration
// =============================================================================
// Defines the rotating coordinate frame for tidal encounter simulations.
// When coord_system == ROTATING, particles experience fictitious forces
// (centrifugal + Coriolis) in a frame co-rotating at constant Ω.
// =============================================================================
struct RotatingFrameConfig {
    bool enabled;                    // true when coord_system == ROTATING
    Vec3d omega;                     // rad/s, angular velocity vector
    float rotation_period_h;         // hours, alternative to omega (auto-converted)

    __host__ __device__ RotatingFrameConfig()
        : enabled(false), omega{0.0, 0.0, 0.0}, rotation_period_h(4.3f) {}
};

// =============================================================================
// Main SimConfig Structure
// =============================================================================
// This is the primary configuration struct that gets copied to GPU constant memory
// It contains all runtime-configurable physics toggles and parameters
// =============================================================================
struct SimConfig {
    // Simulation identity
    char name[MAX_NAME_LENGTH];
    SimulationType sim_type;
    SimulationPhysics physics;
    DemDampingModel dem_damping_model;
    // POD metadata copied with the configuration. Tables remain separate and
    // are uploaded per GPU through upload_material_database().
    MaterialDatabaseInfo material_database;
    CoordinateSystem coord_system;
    RotatingFrameConfig rotating_frame;

    // Sub-configurations
    TimeConfig time;
    ShearBoxConfig shear_box;
    ParticleConfig particle;
    DomainConfig domain;
    AccelerationConfig acceleration;
    SPHConfig sph;
    NeighborConfig neighbor;
    DampingConfig damping;
    IOConfig io;
    GPUConfig gpu;
    MultiGPUConfig multigpu;
    ClumpConfig clump;

    // Walls
    int32_t num_walls;
    WallConfig walls[MAX_WALLS];

    // Termination
    TerminationCondition termination;
    double energy_threshold;

    // Runtime state (not from YAML, set during initialization)
    double current_time;
    int64_t current_step;
    double dt;                   // Calculated Verlet timestep

    __host__ __device__ SimConfig()
        : sim_type(SimulationType::SURFACE_SLOPE), physics(SimulationPhysics::DEM),
          dem_damping_model(DemDampingModel::RESTITUTION),
          coord_system(CoordinateSystem::SURFACE),
          num_walls(0), termination(TerminationCondition::TIME),
          energy_threshold(1.0e-8), current_time(0), current_step(0), dt(1.0e-5) {
        name[0] = '\0';
    }
};

// =============================================================================
// GPU Constant Memory Declaration
// =============================================================================
// Declared in config_device.cu as raw storage to avoid dynamic initialization.
// Access via dev_config macro/reference.
extern __constant__ unsigned char dev_config_storage[sizeof(SimConfig)];

// Provide access as SimConfig reference
#define dev_config (*reinterpret_cast<SimConfig*>(const_cast<unsigned char*>(dev_config_storage)))

// =============================================================================
// Host-Only Function Declarations
// =============================================================================
// Implementations are in config_host.cpp (compiled without nvcc)
// =============================================================================

// Parse YAML file into SimConfig
SimConfig parse_yaml(const std::string& yaml_path);

// Copy SimConfig to GPU constant memory
void copy_config_to_device(const SimConfig& host_config);

// Print configuration summary
void print_config(const SimConfig& config);

// End of header - host function implementations moved to config_host.cpp

} // namespace cudadem

#endif // CUDADEM_COMMAND_LINE_INTERFACE_H

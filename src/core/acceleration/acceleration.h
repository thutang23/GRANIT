// =============================================================================
// Acceleration Interface for DEM Simulation
// =============================================================================
// Unified interface for all acceleration sources:
// - Self-gravity (particle-particle N² interaction)
// - Inertial acceleration (centrifugal, Coriolis, Euler)
// - Tidal acceleration (from perturber, e.g., Earth during Apophis flyby)
// - Pre-calculated acceleration fields (3D texture lookup)
//
// Design Philosophy:
// - Template Policy Composition for compile-time acceleration selection
// - Runtime Configuration via SimConfig for togglable physics
// - All accelerations are additive to particle force arrays
//
// Usage:
//   AccelerationManager::initialize(config);
//   AccelerationManager::compute_all(state, config, current_time, stream);
// =============================================================================

#ifndef CUDADEM_ACCELERATION_H
#define CUDADEM_ACCELERATION_H

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../math/yorp.cuh"
#include "domain/domain_manager.h"
#include "device_state_array.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <vector>

namespace cudadem {

// =============================================================================
// Physical Constants
// =============================================================================

// Gravitational constant (SI units: m³/(kg·s²))
constexpr double GRAVITATIONAL_CONSTANT = 6.67430e-11;

// Softening parameter to prevent singularities (as fraction of min radius)
constexpr float DEFAULT_SOFTENING_FACTOR = 0.01f;

// =============================================================================
// Acceleration Type Traits (Compile-Time Properties)
// =============================================================================

enum class AccelerationType : uint8_t {
    SELF_GRAVITY = 0,
    INERTIAL = 1,
    TIDAL = 2,
    FIELD_3D = 3,
    SEISMIC = 4
};

template<AccelerationType Type>
struct AccelerationTypeTraits;

template<>
struct AccelerationTypeTraits<AccelerationType::SELF_GRAVITY> {
    static constexpr bool is_pairwise = true;
    static constexpr bool is_position_dependent = true;
    static constexpr bool is_time_varying = false;
    static constexpr const char* name = "SelfGravity";
};

template<>
struct AccelerationTypeTraits<AccelerationType::INERTIAL> {
    static constexpr bool is_pairwise = false;
    static constexpr bool is_position_dependent = true;
    static constexpr bool is_time_varying = true;
    static constexpr const char* name = "Inertial";
};

template<>
struct AccelerationTypeTraits<AccelerationType::TIDAL> {
    static constexpr bool is_pairwise = false;
    static constexpr bool is_position_dependent = true;
    static constexpr bool is_time_varying = true;
    static constexpr const char* name = "Tidal";
};

template<>
struct AccelerationTypeTraits<AccelerationType::FIELD_3D> {
    static constexpr bool is_pairwise = false;
    static constexpr bool is_position_dependent = true;
    static constexpr bool is_time_varying = true;
    static constexpr const char* name = "Field3D";
};

// =============================================================================
// Self-Gravity Configuration
// =============================================================================

// Last completed aligned-tree evaluation on the current GPU. Counts describe
// real component/union contexts; source audit violations must remain zero.
struct AlignedClumpGravityDiagnostics {
    uint64_t epoch = 0;
    bool used = false;
    int component_count = 0, large_count = 0, green_members = 0, live_members = 0;
    double root_min_x = 0, root_min_y = 0, root_min_z = 0, root_size = 0;
    int depth = 0;
    uint64_t target_node_count = 0, source_node_count = 0;
    uint64_t m2l_route_count = 0, p2p_route_count = 0;
    uint64_t alignment_checked_count = 0, alignment_violation_count = 0;
    uint64_t self_exclusion_checked_count = 0;
    uint64_t self_source_coefficient_violations = 0;
    // Compatibility alias for coefficient audit violations, not geometric
    // routes: a union source cell may contain excluded members with zero
    // contribution to its target-specific source expansion.
    uint64_t self_source_route_count = 0;
    uint64_t context_remap_count = 0, index_remap_checked_count = 0;
    // Conservative upper estimate for retained capacities, not measured VRAM.
    uint64_t reserved_bytes = 0;
    // Canonical live-prefix forces before body projection. Read-only; valid
    // until the next registered evaluation or AccelerationManager shutdown.
    const double *member_force_x = nullptr, *member_force_y = nullptr, *member_force_z = nullptr;
    int member_force_count = 0;
};

struct SelfGravityConfig {
    bool enabled;
    float softening;           // Softening parameter (meters)
    bool use_symmetry;         // Compute each pair once, apply Newton's 3rd law
    bool respect_periodic;     // Apply minimum image convention

    __host__ __device__ SelfGravityConfig()
        : enabled(false), softening(1e-4f),
          use_symmetry(true), respect_periodic(true) {}
};

// Stable host-facing adaptive-depth lifecycle types.  The body-fitted runtime
// owns its device representation; keeping this public POD contract here lets
// SimulationRunner persist and report controller state without including the
// implementation-only FMM headers (which are compiled inside namespace
// cudadem by the single body-fitted translation unit).
enum class AdaptiveFMMDepthTransition : int32_t {
    NONE = 0,
    INITIAL_SELECTION = 1,
    TRIAL_RESULT = 2,
    FASTER_NEIGHBOR = 3,
    CONFIRMATION = 4,
    CANDIDATE_REJECTED = 5,
    CAPACITY_FALLBACK = 6,
    CAPACITY_GROWTH = 7,
    EFFECTIVE_CEILING_CHANGED = 8,
};

struct AdaptiveFMMDepthControllerState {
    int32_t selected_depth = 0;
    int32_t candidate_depth = 0;
    int32_t launch_depth = 0;
    int32_t search_direction = 1;
    int32_t trial_phase = 0;
    int32_t trial_warmups = 0;
    int32_t trial_sample_count = 0;
    int32_t confirmation_count = 0;
    int32_t evaluations_since_trial = 0;
    int32_t effective_max_depth = 0;
    int32_t node_high_water = 0;
    int32_t capacity_fallback_count = 0;
    uint64_t evaluation_counter = 0;
    uint64_t selected_time_ns = 0;
    uint64_t selected_sample_count = 0;
    uint64_t candidate_time_ns = 0;
    uint64_t last_elapsed_ns = 0;
    float last_relative_improvement = 0.0f;
};

struct AdaptiveFMMDepthDiagnostic {
    uint64_t sequence = 0;
    uint64_t evaluation_counter = 0;
    double evaluation_time = 0.0;
    int32_t tree_dimension = 0;
    int32_t selected_depth = 0;
    int32_t candidate_depth = 0;
    int32_t search_direction = 0;
    int32_t trial_phase = 0;
    double elapsed_ms = 0.0;
    double relative_improvement = 0.0;
    int32_t node_count = 0;
    int32_t node_capacity = 0;
    int32_t effective_max_depth = 0;
    AdaptiveFMMDepthTransition transition_reason =
        AdaptiveFMMDepthTransition::NONE;
};

// Version 2 is intentionally a single-controller snapshot.  Tree dimension is
// configuration, not an adaptive FSM, and producer/CSV watermarks make the
// telemetry hand-off lossless and resumable.
struct AdaptiveFMMDepthCheckpoint {
    uint32_t version = 2;
    int32_t tree_dimension = 3;
    AdaptiveFMMDepthControllerState controller{};
    uint64_t producer_sequence = 0;
    uint64_t persisted_csv_sequence = 0;
};

// =============================================================================
// Acceleration Result (for single particle)
// =============================================================================

struct AccelerationResult {
    double accel_x, accel_y, accel_z;
    double potential_energy;     // For self-gravity tracking

    __device__ AccelerationResult()
        : accel_x(0.0), accel_y(0.0), accel_z(0.0), potential_energy(0.0) {}
};

// =============================================================================
// Forward Declarations for Implementation Files
// =============================================================================

class SelfGravityDispatcher;
class InertialAccelerationDispatcher;
class TidalAccelerationDispatcher;
class AccelerationFieldDispatcher;
struct ClumpStateSoA;
struct ClumpSegmentOffsets;

// =============================================================================
// Unified Acceleration Manager
// =============================================================================
// Orchestrates all acceleration sources through a single interface
// =============================================================================

class AccelerationManager {
private:
    static bool initialized;
    static SelfGravityConfig self_gravity_config;
    // Device-affine YORP storage.  The multi-GPU runner owns one persistent
    // host thread per CUDA device, so a single process-global manager would
    // race its pointers and host-side cadence counters across those threads.
    static DeviceStateArray<YorpManager> yorp_managers_;
    static YorpManager& current_yorp_manager();
    
    // Track actual gravity dispatcher used (may differ from config if FMM override)
    static DeviceStateArray<bool> fmm_initialized_;
    // Non-owning runtime registration.  Kept out of SimulationState because
    // clump storage is owned by SimulationRunner and is optional for SPH.
    static ClumpStateSoA* clump_runtime_state_;
    static const ClumpSegmentOffsets* clump_runtime_segments_;

public:
    // Initialize all acceleration subsystems
    static void initialize(const SimConfig& config);

    // Non-owning access used by the distributed YORP coordinator.  The
    // manager itself remains owned and shut down by AccelerationManager.
    static YorpManager* yorp_manager_for_gpu(int32_t gpu);

    static void set_clump_runtime(ClumpStateSoA* state,
                                  const ClumpSegmentOffsets* segments);

    // Compute all enabled accelerations
    static void compute_all(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream = 0
    );

#ifdef CUDADEM_HAVE_LET
    static void compute_all_multigpu(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        uint64_t gravity_evaluation_epoch,
        class LETCommunicator* let_comm,
        cudaStream_t stream = 0
    );
#endif

    // Individual acceleration computations
    static void compute_self_gravity(
        SimulationState& state,
        const SimConfig& config,
        double evaluation_time,
        cudaStream_t stream = 0
    );

    // Compute self-gravity for RESPA outer step.
    // Writes GRAVITATIONAL ACCELERATION (m/s²) to gravity_force_x/y/z
    // instead of force to force_x/y/z.
    static void compute_outer_gravity(
        SimulationState& state,
        const SimConfig& config,
        double evaluation_time,
        cudaStream_t stream = 0
    );

    // Restore only the clump gravity portion absent from member checkpoint
    // force rows. Normal checkpoint forces already retain their full force.
    static AlignedClumpGravityDiagnostics get_aligned_clump_gravity_diagnostics();

    static void restore_checkpoint_clump_gravity(
        SimulationState& state, const SimConfig& config,
        double evaluation_time, cudaStream_t stream = 0);

    // Compute all non-gravity accelerations: inertial, tidal, fields, YORP.
    // Does NOT call compute_self_gravity.
    static void compute_all_without_gravity(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream = 0,
        bool include_incremental_effects = true
    );

    // Adaptive YORP is observed only after a completed Verlet step.  This is
    // deliberately separate from compute_all(), which injects torque during
    // the force phase.
    static void observe_adaptive_yorp(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream = 0
    );
    // These methods synchronize only when the caller already needs a host
    // boundary (output/checkpoint/final termination), never at check cadence.
    static bool consume_adaptive_yorp_diagnostic(AdaptiveYorpDiagnostic* diagnostic,
                                                  cudaStream_t stream = 0);
    static bool adaptive_yorp_should_terminate(cudaStream_t stream = 0);
    static bool get_adaptive_yorp_checkpoint(AdaptiveYorpCheckpoint* checkpoint,
                                             cudaStream_t stream = 0);
    static bool restore_adaptive_yorp_checkpoint(const AdaptiveYorpCheckpoint& checkpoint,
                                                 cudaStream_t stream = 0);

    // Adaptive FMM depth state is read only at existing output/checkpoint
    // boundaries. Restore may be called before the lazily-created FMM runtime;
    // the dispatcher stages it and applies it before the first gravity build.
    static bool get_adaptive_fmm_depth_checkpoint(
        AdaptiveFMMDepthCheckpoint* checkpoint,
        cudaStream_t stream = 0);
    static bool restore_adaptive_fmm_depth_checkpoint(
        const AdaptiveFMMDepthCheckpoint& checkpoint,
        cudaStream_t stream = 0);
    static void consume_adaptive_fmm_depth_diagnostics(
        std::vector<AdaptiveFMMDepthDiagnostic>& diagnostics,
        uint64_t* dropped,
        cudaStream_t stream = 0);
    static bool service_adaptive_fmm_capacity(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0);

    // Inspect the device-resident FMM terminal mailbox after a synchronization
    // boundary. This call never introduces a synchronization of its own.
    static void throw_if_fmm_terminal_error();

#ifdef CUDADEM_HAVE_LET
    // Multi-GPU: BH tree self-gravity with LET exchange between build and walk.
    // Caller must call this from within #pragma omp parallel (one thread per GPU).
    // let_comm must be initialized and configured on this GPU's device.
    static void compute_self_gravity_multigpu(
        SimulationState& state,
        const SimConfig& config,
        double evaluation_time,
        uint64_t gravity_evaluation_epoch,
        class LETCommunicator* let_comm,
        cudaStream_t stream = 0
    );
#endif

    static void compute_inertial(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream = 0
    );

    static void compute_tidal(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream = 0
    );

    static void compute_field_acceleration(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        int field_id,
        cudaStream_t stream = 0
    );

    // Query gravity at a specific position (for walls)
    static void query_gravity_at_position(
        double x, double y, double z,
        float& gx, float& gy, float& gz
    );

    // Query functions
    static bool is_self_gravity_enabled() {
        return self_gravity_config.enabled;
    }

    // Shutdown and free resources
    static void shutdown();
};

// =============================================================================
// Device Utility Functions
// =============================================================================

// Compute gravitational acceleration from particle j on particle i
// Returns acceleration vector on particle i
__device__ __forceinline__
void compute_gravity_pair(
    double& ax, double& ay, double& az,    // Output: acceleration on i
    double& potential,                     // Output: potential energy contribution
    double xi, double yi, double zi,       // Position of i
    double xj, double yj, double zj,       // Position of j
    double mj,                             // Mass of j (double)
    double G,                              // Gravitational constant
    double softening                       // Softening parameter
) {
    // Distance vector (from i to j)
    double dx = xj - xi;
    double dy = yj - yi;
    double dz = zj - zi;

    // Distance squared with softening
    double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
    double r = sqrt(r2);
    double r3 = r2 * r;

    // Gravitational acceleration: a = G * m_j / r² * r̂
    double factor = G * mj / r3;

    ax = factor * dx;
    ay = factor * dy;
    az = factor * dz;

    // Potential energy: U = -G * m_i * m_j / r (computed per pair)
    // We return -G * m_j / r, multiply by m_i outside
    potential = -G * mj / r;
}

// Apply minimum image convention to distance vector
// Modifies dx, dy, dz in place
__device__ __forceinline__
void apply_minimum_image(
    float& dx, float& dy, float& dz,
    float box_x, float box_y, float box_z,
    bool periodic_x, bool periodic_y, bool periodic_z
) {
    if (periodic_x && box_x > 0.0f) {
        float half_box = box_x * 0.5f;
        if (dx > half_box) dx -= box_x;
        else if (dx < -half_box) dx += box_x;
    }

    if (periodic_y && box_y > 0.0f) {
        float half_box = box_y * 0.5f;
        if (dy > half_box) dy -= box_y;
        else if (dy < -half_box) dy += box_y;
    }

    if (periodic_z && box_z > 0.0f) {
        float half_box = box_z * 0.5f;
        if (dz > half_box) dz -= box_z;
        else if (dz < -half_box) dz += box_z;
    }
}

// =============================================================================
// Verification Functions (Host-only)
// =============================================================================

#ifndef __CUDACC__

#include <iostream>

inline void print_acceleration_config(const SimConfig& config) {
    std::cout << "=== Acceleration Configuration ===" << std::endl;
    std::cout << "Self-gravity: " << (config.acceleration.enable_self_gravity ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Inertial:     " << (config.acceleration.enable_inertial ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Tidal:        " << (config.acceleration.enable_tidal ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Accel fields: " << config.acceleration.num_accel_fields << std::endl;
    std::cout << "=================================" << std::endl;
}

#endif // __CUDACC__

} // namespace cudadem

#endif // CUDADEM_ACCELERATION_H

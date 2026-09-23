#ifndef CUDADEM_CLUMP_H
#define CUDADEM_CLUMP_H

// =============================================================================
// Clump (Multi-Sphere Rigid Body) — State SoA and Kernel Declarations
// =============================================================================
// Each clump is a set of sub-spheres that move as a single rigid body.
// Sub-spheres live in ParticleSoA with positive clump_id values.
// ClumpStateSoA holds the 6-DOF state for each active clump.
//
// Pipeline (per timestep):
// 1. clump_zero_accumulators_kernel  — zero net_force/torque
// 2. clump_verlet_step1_kernel       — half-kick, drift CoM, rotate body
// 3. [contact + wall + gravity kernels accumulate forces on sub-spheres]
// 4. clump_pre_aggregate_kernel      — compute (F, τ) per sub-sphere in staging buffer
// 5. clump_aggregate_forces          — CUB segmented reduce → net_force/torque per clump
// 6. clump_verlet_step2_kernel       — second half-kick CoM vel + body omega
// 7. clump_sync_positions_kernel     — re-derive sub-sphere world positions from CoM + quat + offset
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../material/material_database.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem {

// =============================================================================
// ClumpStateSoA — dynamically sized 6-DOF rigid body state
// =============================================================================
struct ClumpStateSoA {
    int32_t num_clumps;
    int32_t capacity;

    // CoM position (world frame, double precision)
    double* com_x;
    double* com_y;
    double* com_z;

    // Linear velocity (world frame)
    double* vel_x;
    double* vel_y;
    double* vel_z;

    // Angular velocity (body frame — Euler's equations work in body frame)
    double* omega_x;
    double* omega_y;
    double* omega_z;

    // Orientation quaternion (w, x, y, z) — rotates body → world
    double* quat_w;
    double* quat_x;
    double* quat_y;
    double* quat_z;

    // Rigid body properties (constant after init)
    // These deliberately remain double: clump properties are reductions of
    // potentially heterogeneous member spheres and must survive restarts.
    double* mass;
    // Symmetric body-frame inertia, stored xx, xy, xz, yy, yz, zz.
    // Do not silently diagonalise this tensor: imported clumps need not be in
    // their principal-axis frame.
    double* inertia_xx; double* inertia_xy; double* inertia_xz;
    double* inertia_yy; double* inertia_yz; double* inertia_zz;
    // trace_free_p3_v1 intrinsic mass multipole (19 independent values).
    double* multipole_p3[19];

    // Per-step net force/torque accumulators (zeroed before each step)
    double* net_force_x;
    double* net_force_y;
    double* net_force_z;
    double* net_torque_x;
    double* net_torque_y;
    double* net_torque_z;

    // RESPA outer-step gravity wrench. Linear entries are force (not
    // acceleration); rotational entries are world-frame torque.
    double* gravity_force_x;
    double* gravity_force_y;
    double* gravity_force_z;
    double* gravity_torque_x;
    double* gravity_torque_y;
    double* gravity_torque_z;

    ClumpStateSoA()
        : num_clumps(0), capacity(0),
          com_x(nullptr), com_y(nullptr), com_z(nullptr),
          vel_x(nullptr), vel_y(nullptr), vel_z(nullptr),
          omega_x(nullptr), omega_y(nullptr), omega_z(nullptr),
          quat_w(nullptr), quat_x(nullptr), quat_y(nullptr), quat_z(nullptr),
          mass(nullptr), inertia_xx(nullptr), inertia_xy(nullptr), inertia_xz(nullptr),
          inertia_yy(nullptr), inertia_yz(nullptr), inertia_zz(nullptr),
          net_force_x(nullptr), net_force_y(nullptr), net_force_z(nullptr),
          net_torque_x(nullptr), net_torque_y(nullptr), net_torque_z(nullptr),
          gravity_force_x(nullptr), gravity_force_y(nullptr), gravity_force_z(nullptr),
          gravity_torque_x(nullptr), gravity_torque_y(nullptr), gravity_torque_z(nullptr) {}
};

// =============================================================================
// ClumpSegmentOffsets — precomputed at init, immutable during simulation
// Sub-spheres are sorted by clump_id once at init; these offsets index into
// the sorted region of ParticleSoA for CUB DeviceSegmentedReduce.
// =============================================================================
struct ClumpSegmentOffsets {
    int32_t* seg_begin;   // [num_clumps]: index of first sub-sphere in sorted SoA
    int32_t* seg_end;     // [num_clumps]: one-past-last sub-sphere index
    int32_t  num_segments;
    // Member region after the one-time canonical grouping.  All ranges are
    // half-open and refer to ParticleSoA local indices.
    int32_t normal_begin, normal_end;
    int32_t member_begin, member_end;
    int32_t dead_begin, dead_end;

    ClumpSegmentOffsets()
        : seg_begin(nullptr), seg_end(nullptr), num_segments(0),
          normal_begin(0), normal_end(0), member_begin(0), member_end(0), dead_begin(0), dead_end(0) {}
};

// =============================================================================
// ClumpStagingBuffer — temporary (F, τ) per sub-sphere before segmented reduce
// Written by clump_pre_aggregate_kernel, consumed by clump_aggregate_forces.
// Sized to max sub-sphere count; allocated once at init.
// =============================================================================
struct ClumpStagingBuffer {
    double* fx;
    double* fy;
    double* fz;
    double* tx;
    double* ty;
    double* tz;
    int32_t capacity;

    // Persistent CUB temp buffer for DeviceSegmentedReduce (pre-allocated at init)
    void*  cub_temp_storage;
    size_t cub_temp_bytes;

    ClumpStagingBuffer()
        : fx(nullptr), fy(nullptr), fz(nullptr),
          tx(nullptr), ty(nullptr), tz(nullptr),
          capacity(0), cub_temp_storage(nullptr), cub_temp_bytes(0) {}
};

// =============================================================================
// Host API — allocation and initialization
// =============================================================================

ClumpStateSoA       allocate_clump_state(int32_t max_clumps);
ClumpSegmentOffsets allocate_clump_segments(int32_t max_clumps);
ClumpStagingBuffer  allocate_clump_staging(int32_t max_sub_spheres);

void free_clump_state(ClumpStateSoA& cs);
void free_clump_segments(ClumpSegmentOffsets& seg);
void free_clump_staging(ClumpStagingBuffer& buf);

// Main initialization entry point: parses config, instantiates sub-spheres
// into ParticleSoA, computes inertia, uploads ClumpState to GPU, builds
// segment offset arrays. Called once during SimulationRunner::initialize().
void initialize_clumps(
    const SimConfig&    config,
    const DeviceMaterialDatabase& material_database,
    ParticleSoA&        particles,
    ClumpStateSoA&      clump_state,
    ClumpSegmentOffsets& segments
);

// Call after initialize_clumps to pre-allocate CUB temp storage.
// Must be called before the simulation loop starts.
void initialize_clump_cub_temp(
    ClumpStagingBuffer& staging,
    const ClumpStateSoA& cs,
    const ClumpSegmentOffsets& seg
);

// =============================================================================
// CUDA Kernel Declarations
// =============================================================================

// Zero net_force/torque accumulators — call before force kernels each step
__global__ void clump_zero_accumulators_kernel(
    double* net_fx, double* net_fy, double* net_fz,
    double* net_tx, double* net_ty, double* net_tz,
    int32_t num_clumps
);

// Apply one RESPA gravity half-impulse without drifting the body.
__global__ void clump_apply_gravity_impulse_kernel(
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    const double* quat_w, const double* quat_x, const double* quat_y, const double* quat_z,
    const double* gravity_force_x, const double* gravity_force_y, const double* gravity_force_z,
    const double* gravity_torque_x, const double* gravity_torque_y, const double* gravity_torque_z,
    const double* mass,
    const double* inertia_xx, const double* inertia_xy, const double* inertia_xz,
    const double* inertia_yy, const double* inertia_yz, const double* inertia_zz,
    double impulse_dt, int32_t num_clumps);

// For each sub-sphere (indexed in the sorted SoA region 0..num_sub_spheres-1):
// compute torque arm τ_i = (pos_i - com_{cid}) × force_i (world frame)
// and write (fx, fy, fz, τx, τy, τz) to the staging buffer.
// Must be called after all force kernels have finished.
__global__ void clump_pre_aggregate_kernel(
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    const double* __restrict__ torque_x,
    const double* __restrict__ torque_y,
    const double* __restrict__ torque_z,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int32_t* __restrict__ clump_id,
    const double* __restrict__ com_x,
    const double* __restrict__ com_y,
    const double* __restrict__ com_z,
    double* __restrict__ stg_fx, double* __restrict__ stg_fy, double* __restrict__ stg_fz,
    double* __restrict__ stg_tx, double* __restrict__ stg_ty, double* __restrict__ stg_tz,
    int32_t num_sub_spheres
);

// Velocity Verlet first half-kick and drift for ClumpState:
// vel   += 0.5 * dt * net_force / mass
// omega += 0.5 * dt * I^-1 * (τ_body - ω × Iω)  [Euler's equations, body frame]
// com   += vel * dt
// quat   = exp(0.5 * dt * omega_body) applied to q, then normalized
// where τ_body = R^T * net_torque_world (conjugate quaternion rotation)
__global__ void clump_verlet_step1_kernel(
    double* com_x, double* com_y, double* com_z,
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    const double* net_force_x, const double* net_force_y, const double* net_force_z,
    const double* net_torque_x, const double* net_torque_y, const double* net_torque_z,
    double* quat_w, double* quat_x, double* quat_y, double* quat_z,
    const double* mass,
    const double* inertia_xx, const double* inertia_xy, const double* inertia_xz,
    const double* inertia_yy, const double* inertia_yz, const double* inertia_zz,
    double dt, int32_t num_clumps
);

// Velocity Verlet second half-kick:
// vel   += 0.5 * dt * net_force / mass
// omega += 0.5 * dt * I^-1 * (τ_body - ω × Iω)
__global__ void clump_verlet_step2_kernel(
    double* com_x, double* com_y, double* com_z,
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    double* quat_w, double* quat_x, double* quat_y, double* quat_z,
    const double* net_force_x, const double* net_force_y, const double* net_force_z,
    const double* net_torque_x, const double* net_torque_y, const double* net_torque_z,
    const double* mass,
    const double* inertia_xx, const double* inertia_xy, const double* inertia_xz,
    const double* inertia_yy, const double* inertia_yz, const double* inertia_zz,
    double dt, int32_t num_clumps
);

// Re-derive sub-sphere world positions: pos_i = com_{cid} + R(quat_{cid}) * body_offset_i
__global__ void clump_sync_positions_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const float* __restrict__ off_x,
    const float* __restrict__ off_y,
    const float* __restrict__ off_z,
    const int32_t* __restrict__ clump_id,
    const double* __restrict__ com_x,
    const double* __restrict__ com_y,
    const double* __restrict__ com_z,
    const double* __restrict__ quat_w,
    const double* __restrict__ quat_x,
    const double* __restrict__ quat_y,
    const double* __restrict__ quat_z,
    int32_t num_sub_spheres
);

// Host dispatcher: runs CUB DeviceSegmentedReduce on the staging buffer
// to accumulate net_force/torque per clump. Uses a persistent temp buffer
// allocated inside ClumpStagingBuffer (see clump.cu).
void clump_aggregate_forces(
    const ClumpStagingBuffer& staging,
    ClumpStateSoA& clump_state,
    const ClumpSegmentOffsets& segments,
    int32_t num_sub_spheres,
    cudaStream_t stream
);

// Synchronise all member kinematics after a rigid-body update.  `clump_id`
// uses persistent on-disk IDs (1..K), while state arrays use id-1.
__global__ void clump_sync_kinematics_kernel(
    double* pos_x, double* pos_y, double* pos_z,
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    double* particle_qw, double* particle_qx, double* particle_qy, double* particle_qz,
    const float* off_x, const float* off_y, const float* off_z,
    const int32_t* clump_id, const double* com_x, const double* com_y, const double* com_z,
    const double* com_vel_x, const double* com_vel_y, const double* com_vel_z,
    const double* body_omega_x, const double* body_omega_y, const double* body_omega_z,
    const double* qw, const double* qx, const double* qy, const double* qz, int32_t count);

__global__ void clump_wrap_com_kernel(
    double* com_x, double* com_y, double* com_z, int32_t count,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double min_x, double min_y, double min_z,
    double length_x, double length_y, double length_z);

// Host API — state retrieval for visualization
int32_t clump_get_states(
    ClumpStateSoA& host_states,
    int32_t** template_indices,   // output: caller should free with cudaFreeHost
    const ClumpStateSoA& device_states
);

// Free host-side clump state buffers allocated by clump_get_states
void free_clump_host_state(ClumpStateSoA& host_states, int32_t* template_indices);

} // namespace cudadem

#endif // CUDADEM_CLUMP_H

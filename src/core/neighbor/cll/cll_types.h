// =============================================================================
// Cell-Linked List Types
// =============================================================================
// Internal types for CLL neighbor search implementation.

#ifndef CUDADEM_CLL_TYPES_H
#define CUDADEM_CLL_TYPES_H

#include "../neighbor_types.h"
#include "../lbvh/max_support_reduce.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem {
namespace cll {

// =============================================================================
// CLL GPU State - Per-GPU Internal Tracking
// =============================================================================

struct CLLGPUState {
    int32_t gpu_id;
    bool initialized;
    bool grid_valid;
    int32_t last_rebuild_step;

    // Shear-only XY bins. The sparse vertical direction is filtered exactly
    // during the query; these buffers grow after migration or ghost exchange.
    int32_t* shear_heads;
    int32_t* shear_next;
    int32_t shear_heads_capacity;
    int32_t shear_particle_capacity;
    lbvh::MaxSupportReduction shear_max_support;

    // Internal buffers (not exposed in NeighborData)
    int32_t* prev_cell_ids;        // Previous cell IDs for change detection
    int32_t* temp_keys;            // Sorting temporaries
    int32_t* temp_values;

    // Particle reordering temporaries
    double* temp_pos_x, *temp_pos_y, *temp_pos_z;
    double* temp_vel_x, *temp_vel_y, *temp_vel_z;
    float* temp_radius, *temp_mass;
    double* temp_omega_x, *temp_omega_y, *temp_omega_z;
    float* temp_moment_of_inertia;
    double* temp_torque_x, *temp_torque_y, *temp_torque_z;
    double* temp_quat_w, *temp_quat_x, *temp_quat_y, *temp_quat_z;
    float* temp_cohesive_strength;
    int32_t* temp_material_id;
    int32_t* temp_contact_history_offset, *temp_num_contacts;

    // Change detection
    int32_t* d_changed;

    // Atomic binning (CUB)
    int32_t* d_cell_counts;
    int32_t* d_cell_offsets;
    void* d_cub_temp_storage;
    size_t cub_temp_storage_bytes;

    // Async displacement readback
    float* h_pinned_max_disp_sq;
    cudaEvent_t disp_copy_done_event;
    float cached_max_disp_sq;
    bool async_readback_pending;
    int32_t steps_since_rebuild;

    // Verlet neighbor cache
    VerletNeighborCache verlet_cache;

    // Statistics
    NeighborSearchStats stats;

    CLLGPUState()
        : gpu_id(-1), initialized(false), grid_valid(false), last_rebuild_step(-1),
          shear_heads(nullptr), shear_next(nullptr), shear_heads_capacity(0),
          shear_particle_capacity(0),
          prev_cell_ids(nullptr), temp_keys(nullptr), temp_values(nullptr),
          temp_pos_x(nullptr), temp_pos_y(nullptr), temp_pos_z(nullptr),
          temp_vel_x(nullptr), temp_vel_y(nullptr), temp_vel_z(nullptr),
          temp_radius(nullptr), temp_mass(nullptr),
          temp_omega_x(nullptr), temp_omega_y(nullptr), temp_omega_z(nullptr),
          temp_moment_of_inertia(nullptr),
          temp_torque_x(nullptr), temp_torque_y(nullptr), temp_torque_z(nullptr),
          temp_quat_w(nullptr), temp_quat_x(nullptr), temp_quat_y(nullptr), temp_quat_z(nullptr),
          temp_cohesive_strength(nullptr),
          temp_material_id(nullptr),
          temp_contact_history_offset(nullptr), temp_num_contacts(nullptr),
          d_changed(nullptr),
          d_cell_counts(nullptr), d_cell_offsets(nullptr),
          d_cub_temp_storage(nullptr), cub_temp_storage_bytes(0),
          h_pinned_max_disp_sq(nullptr), disp_copy_done_event(nullptr),
          cached_max_disp_sq(0.0f), async_readback_pending(false), steps_since_rebuild(0) {}
};

} // namespace cll
} // namespace cudadem

#endif // CUDADEM_CLL_TYPES_H

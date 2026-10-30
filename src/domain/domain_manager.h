#ifndef CUDADEM_DOMAIN_MANAGER_H
#define CUDADEM_DOMAIN_MANAGER_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "../core/constant/system_initialize.h"
#include "../core/constant/command_line_interface.h"
#include "rcb_balancer.h"
#include "morton_partition.h"

namespace cudadem {

// =============================================================================
// GPU Constant Memory Structure for Domain Parameters
// =============================================================================
// This struct is copied to __constant__ memory for fast kernel access
// =============================================================================

constexpr int32_t MAX_MULTI_GPUS = 8;

struct DomainParamsGPU {
    int32_t num_gpus;
    int32_t morton_enabled;
    bool restore_morton_partition; // Checkpoint ownership is authoritative (Morton cuts or RCB boxes).
    MortonGrid morton_grid;
    const uint64_t* morton_cuts;
    const MortonCoverCell* morton_cells;
    int32_t morton_cell_count;

    // Per-GPU sub-domain AABBs (ORB-style)
    // GPU g owns particles where box_min <= pos < box_max on all axes
    double gpu_box_min_x[MAX_MULTI_GPUS];
    double gpu_box_min_y[MAX_MULTI_GPUS];
    double gpu_box_min_z[MAX_MULTI_GPUS];
    double gpu_box_max_x[MAX_MULTI_GPUS];
    double gpu_box_max_y[MAX_MULTI_GPUS];
    double gpu_box_max_z[MAX_MULTI_GPUS];

    // Global domain bounds (computed from particles if auto_bounds)
    double domain_min_x, domain_min_y, domain_min_z;
    double domain_max_x, domain_max_y, domain_max_z;
};

// =============================================================================
// SubDomain - Represents one GPU's responsibility
// =============================================================================

struct SubDomain {
    int32_t gpu_id;
    // ORB AABB sub-domain box: particles INSIDE this box belong to this GPU
    // Convention: box_min <= position < box_max (min inclusive, max exclusive on split axis)
    double box_min_x, box_min_y, box_min_z;
    double box_max_x, box_max_y, box_max_z;

    int32_t num_local_particles;   // Particles owned by this GPU
    int32_t num_ghost_particles;   // Ghost copies from neighbor GPUs
    int32_t ghost_start_index;     // Where ghosts begin in local array
    int32_t capacity;              // Allocated capacity for particles
    int32_t num_local_before_migration;  // Snapshot of local count before migration

    __host__ __device__ SubDomain()
        : gpu_id(0),
          box_min_x(-1e30), box_min_y(-1e30), box_min_z(-1e30),
          box_max_x(1e30), box_max_y(1e30), box_max_z(1e30),
          num_local_particles(0), num_ghost_particles(0),
          ghost_start_index(0), capacity(0), num_local_before_migration(0) {}
};

// =============================================================================
// GPUDomainState - Per-GPU state for multi-GPU simulation
// =============================================================================

struct GPUDomainState {
    int32_t gpu_id;
    cudaStream_t compute_stream;   // Main compute stream
    cudaStream_t comm_stream;      // Separate stream for P2P transfers

    SimulationState local_state;   // This GPU's particles + ghosts
    SubDomain subdomain;

    // Index mappings for ghost/migration
    int32_t* d_owner_gpu;          // Which GPU owns each particle
    int32_t* d_local_to_global;    // Map local index -> global particle ID
    int32_t* d_global_to_local;    // Map global ID -> local index (hash table)

    // Flags for halo/migration identification
    int32_t* d_is_halo;            // Flag: 1 if particle is near boundary
    int32_t* d_is_migrating;       // Flag: 1 if particle left this domain

    GPUDomainState()
        : gpu_id(0), compute_stream(nullptr), comm_stream(nullptr),
          d_owner_gpu(nullptr),
          d_local_to_global(nullptr), d_global_to_local(nullptr),
          d_is_halo(nullptr), d_is_migrating(nullptr) {}
};

// Capacity/accounting snapshot used by multi-GPU allocation telemetry.
struct CoreAllocationTelemetry {
    int32_t particle_capacity = 0;
    int32_t contact_history_capacity = 0;
    int32_t wall_history_capacity = 0;
    size_t particle_bytes = 0;
    size_t persistent_history_bytes = 0;
    size_t mapping_and_flag_bytes = 0;
};

CoreAllocationTelemetry core_allocation_telemetry(const GPUDomainState& gpu_state);

// Grow one GPU's core storage at a locally synchronized phase boundary.  Call
// this before a halo receive or migration insertion whose final local+ghost
// count is known.  The function is grow-only and preserves all active particle
// fields and fixed-stride persistent histories.
void ensure_particle_capacity(GPUDomainState& gpu_state,
                              int32_t required_total,
                              const char* phase_name);
// Backward-compatible descriptive alias for call sites that refer to domain
// capacity rather than the ParticleSoA explicitly.
inline void ensure_gpu_domain_capacity(GPUDomainState& gpu_state,
                                       int32_t required_capacity,
                                       const char* phase_name) {
    ensure_particle_capacity(gpu_state, required_capacity, phase_name);
}

// =============================================================================
// DomainManager - SFC-based Domain Decomposition Manager
// =============================================================================

class DomainManager {
public:
    // =========================================================================
    // ORB Sub-domain Ownership Check (device-friendly)
    // =========================================================================

    // Determine which GPU owns a particle based on AABB containment
    // Returns my_gpu_id if no box contains this particle (should not happen)
    __device__ __host__ static inline int32_t position_to_gpu(
        double x, double y, double z,
        const DomainParamsGPU& params,
        int32_t my_gpu_id
    ) {
        int32_t num_gpus = params.num_gpus;
        if (params.morton_enabled)
            return morton_owner(params.morton_grid.key(x, y, z), params.morton_cuts, num_gpus);
        for (int32_t g = 0; g < num_gpus; ++g) {
            if (x >= params.gpu_box_min_x[g] && x < params.gpu_box_max_x[g] &&
                y >= params.gpu_box_min_y[g] && y < params.gpu_box_max_y[g] &&
                z >= params.gpu_box_min_z[g] && z < params.gpu_box_max_z[g]) {
                return g;
            }
        }
        // Fallback: closest box center
        double best_dist = 1e30;
        int32_t best_gpu = my_gpu_id;
        for (int32_t g = 0; g < num_gpus; ++g) {
            double cx = (params.gpu_box_min_x[g] + params.gpu_box_max_x[g]) * 0.5;
            double cy = (params.gpu_box_min_y[g] + params.gpu_box_max_y[g]) * 0.5;
            double cz = (params.gpu_box_min_z[g] + params.gpu_box_max_z[g]) * 0.5;
            double dx = x - cx, dy = y - cy, dz = z - cz;
            double dist = dx * dx + dy * dy + dz * dz;
            if (dist < best_dist) { best_dist = dist; best_gpu = g; }
        }
        return best_gpu;
    }

    static void set_morton_partition(DomainParamsGPU& params, const MortonGrid& grid,
                                     const std::vector<uint64_t>& cuts);

    // Publish the contact-image phase on the current CUDA device. Call after
    // selecting that device, before halo classification for these positions.
    // This does not mutate shared ownership geometry or the FMM root.
    static void set_shear_time(const SimConfig& config, double time);

    // Check if a particle's influence region overlaps a neighbor GPU's box
    __device__ __host__ static inline bool aabb_overlaps(
        double p_min_x, double p_min_y, double p_min_z,
        double p_max_x, double p_max_y, double p_max_z,
        double box_min_x, double box_min_y, double box_min_z,
        double box_max_x, double box_max_y, double box_max_z
    ) {
        return (p_max_x >= box_min_x && p_min_x <= box_max_x &&
                p_max_y >= box_min_y && p_min_y <= box_max_y &&
                p_max_z >= box_min_z && p_min_z <= box_max_z);
    }

    // =========================================================================
    // Initialization and Lifecycle
    // =========================================================================

    // Initialize multi-GPU domain decomposition
    static DomainParamsGPU initialize(
        const SimConfig& config,
        int32_t num_gpus,
        std::vector<GPUDomainState>& gpu_states
    );

    // Initial distribution of particles across GPUs using ORB
    static void distribute_particles(
        const SimulationState& global_state,
        std::vector<GPUDomainState>& gpu_states,
        DomainParamsGPU& params,
        const SimConfig& config
    );

    // Periodic load rebalancing - recompute splits on-GPU via RCB (no host roundtrip).
    static void update_load_balance(
        std::vector<GPUDomainState>& gpu_states,
        DomainParamsGPU& params,
        RCBBalancer& balancer
    );

    // Commit boxes that have already passed the adaptive proposal/readiness
    // gates.  This only updates ownership bounds and device constants; the
    // caller's normal migration phase performs particle movement afterwards.
    static void apply_accepted_rcb_boxes(
        std::vector<GPUDomainState>& gpu_states,
        DomainParamsGPU& params,
        const RcbBox* boxes,
        int32_t box_count
    );

    // Cleanup resources
    static void shutdown(std::vector<GPUDomainState>& gpu_states);

    // Copy domain params to GPU constant memory
    static void copy_params_to_device(const DomainParamsGPU& params);

private:
    // Single commit path for legacy and proposal-based balancing.
    static void apply_domain_params_to_states(
        std::vector<GPUDomainState>& gpu_states,
        const DomainParamsGPU& params
    );
};

// =============================================================================
// Kernel Declarations
// =============================================================================

__global__ void classify_particles_orb_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    int32_t* __restrict__ owner_gpu,
    int32_t num_particles,
    int32_t num_gpus
);

__global__ void identify_halo_particles_orb_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ interaction_extent,
    float interaction_extent_scale,
    int32_t* __restrict__ is_halo,
    int32_t* __restrict__ halo_dest_mask,
    int32_t num_local_particles,
    float halo_width,
    int32_t my_gpu_id,
    int32_t num_gpus,
    int32_t* send_counts
);

__global__ void identify_migrating_particles_orb_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    int32_t* __restrict__ is_migrating,
    int32_t* __restrict__ dest_gpu,
    int32_t num_particles,
    int32_t num_gpus,
    int32_t my_gpu_id,
    int32_t* send_counts
);

} // namespace cudadem

#endif // CUDADEM_DOMAIN_MANAGER_H

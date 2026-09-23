#include "domain_manager.h"
#include "let_communicator.h"
#include "../core/shear/shear_box.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <mutex>

namespace cudadem {

CoreAllocationTelemetry core_allocation_telemetry(const GPUDomainState& gpu_state) {
    const auto& state = gpu_state.local_state;
    const size_t capacity = static_cast<size_t>(std::max(0, state.particles.capacity));
    CoreAllocationTelemetry telemetry;
    telemetry.particle_capacity = state.particles.capacity;
    telemetry.contact_history_capacity = state.contact_history.capacity;
    telemetry.wall_history_capacity = state.wall_contact_history.capacity;
    // Include the three integer winding arrays in particle storage telemetry.
    telemetry.particle_bytes = capacity * (29 * sizeof(double) + 34 * sizeof(float)
        + 9 * sizeof(int32_t) + 3 * sizeof(int64_t));
    telemetry.persistent_history_bytes = static_cast<size_t>(std::max(0, state.contact_history.capacity)) * sizeof(ContactHistory)
        + static_cast<size_t>(std::max(0, state.wall_contact_history.capacity)) * sizeof(WallContactHistory);
    const size_t indexed_array_bytes = capacity * sizeof(int32_t);
    telemetry.mapping_and_flag_bytes = (gpu_state.d_owner_gpu ? indexed_array_bytes : 0)
        + (gpu_state.d_local_to_global ? indexed_array_bytes : 0)
        + (gpu_state.d_is_halo ? indexed_array_bytes : 0)
        + (gpu_state.d_is_migrating ? indexed_array_bytes : 0);
    return telemetry;
}

namespace {
void resize_capacity_index_array(int32_t*& array, int32_t old_capacity,
                                 int32_t new_capacity, const char* name) {
    if (!array) return;
    int32_t* replacement = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&replacement, static_cast<size_t>(new_capacity) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(replacement, array, static_cast<size_t>(old_capacity) * sizeof(int32_t),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(replacement + old_capacity, 0,
                              static_cast<size_t>(new_capacity - old_capacity) * sizeof(int32_t)));
    } catch (...) {
        if (replacement) cudaFree(replacement);
        throw;
    }
    CUDA_CHECK(cudaFree(array));
    array = replacement;
    (void)name;
}
} // namespace

void ensure_particle_capacity(GPUDomainState& gpu_state,
                              int32_t required_capacity,
                              const char* phase_name) {
    const int32_t old_capacity = gpu_state.local_state.particles.capacity;
    if (required_capacity <= old_capacity) return;
    if (required_capacity <= 0 || old_capacity <= 0) {
        throw std::runtime_error("FATAL core resize has invalid capacity on GPU "
                                 + std::to_string(gpu_state.gpu_id));
    }
    const int64_t growth = (static_cast<int64_t>(old_capacity) * 3 + 1) / 2;
    const int64_t target64 = std::max<int64_t>(required_capacity, growth);
    if (target64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("FATAL core resize capacity overflow on GPU "
                                 + std::to_string(gpu_state.gpu_id));
    }
    const int32_t new_capacity = static_cast<int32_t>(target64);
    try {
        CUDA_CHECK(cudaSetDevice(gpu_state.gpu_id));
        CUDA_CHECK(cudaStreamSynchronize(gpu_state.compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_state.comm_stream));
        resize_particle_soa_preserving(gpu_state.local_state.particles, new_capacity);
        resize_contact_history_preserving(gpu_state.local_state.contact_history, new_capacity);
        resize_wall_contact_history_preserving(gpu_state.local_state.wall_contact_history, new_capacity);
        resize_capacity_index_array(gpu_state.d_owner_gpu, old_capacity, new_capacity, "d_owner_gpu");
        resize_capacity_index_array(gpu_state.d_local_to_global, old_capacity, new_capacity, "d_local_to_global");
        resize_capacity_index_array(gpu_state.d_is_halo, old_capacity, new_capacity, "d_is_halo");
        resize_capacity_index_array(gpu_state.d_is_migrating, old_capacity, new_capacity, "d_is_migrating");
        // d_global_to_local is keyed by global ID and is not capacity-indexed.
        gpu_state.subdomain.capacity = new_capacity;
        const CoreAllocationTelemetry telemetry = core_allocation_telemetry(gpu_state);
        std::cout << "[memory] GPU " << gpu_state.gpu_id
                  << " core grow phase=" << (phase_name ? phase_name : "unspecified")
                  << " old_capacity=" << old_capacity
                  << " required=" << required_capacity
                  << " new_capacity=" << new_capacity
                  << " tracked_bytes="
                  << (telemetry.particle_bytes + telemetry.persistent_history_bytes
                      + telemetry.mapping_and_flag_bytes)
                  << std::endl;
    } catch (const std::exception& e) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream msg;
        msg << "FATAL core capacity resize: gpu=" << gpu_state.gpu_id
            << ", phase=" << (phase_name ? phase_name : "unspecified")
            << ", old_capacity=" << old_capacity
            << ", required_capacity=" << required_capacity
            << ", requested_capacity=" << new_capacity
            << ", device_free_bytes=" << (memory_status == cudaSuccess ? free_bytes : 0)
            << ", device_total_bytes=" << (memory_status == cudaSuccess ? total_bytes : 0)
            << ", cuda_error=" << e.what();
        throw std::runtime_error(msg.str());
    }
}

// =============================================================================
// GPU Constant Memory for Domain Parameters
// =============================================================================
__constant__ DomainParamsGPU dev_domain_params;

// Each worker publishes only its device's phase; ownership cuts remain shared
// host geometry and can be updated independently at a collective boundary.
struct DomainShearPhaseGPU {
    int enabled;
    double length_x, length_y, reduced_shift;
    double min_x, min_y, max_x, max_y;
};
__constant__ DomainShearPhaseGPU dev_domain_shear;

void DomainManager::set_shear_time(const SimConfig& config, double time) {
    DomainShearPhaseGPU phase{};
    phase.enabled = config.shear_box.enabled;
    if (phase.enabled) {
        phase.min_x = config.domain.box_min.x;
        phase.min_y = config.domain.box_min.y;
        phase.max_x = config.domain.box_max.x;
        phase.max_y = config.domain.box_max.y;
        phase.length_x = config.domain.box_max.x - config.domain.box_min.x;
        phase.length_y = config.domain.box_max.y - config.domain.box_min.y;
        phase.reduced_shift = compute_shear_phase_checked(
            config.shear_box, phase.length_x, phase.length_y, time).reduced_shift;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(dev_domain_shear, &phase, sizeof(phase)));
}

__device__ inline bool domain_halo_overlaps(
    double x, double y, double z, double inflate,
    double lo_x, double lo_y, double lo_z,
    double hi_x, double hi_y, double hi_z) {
    const auto& shear = dev_domain_shear;
    if (shear.enabled) {
        // The square XY Morton grid can extend outside a rectangular physical
        // box. Its padding contains no targets, even when an image lands there.
        lo_x = fmax(lo_x, shear.min_x);
        lo_y = fmax(lo_y, shear.min_y);
        hi_x = fmin(hi_x, shear.max_x);
        hi_y = fmin(hi_y, shear.max_y);
        if (!(lo_x < hi_x && lo_y < hi_y)) return false;
    }
    const int layers = shear.enabled ? 1 : 0;
    for (int ix = -layers; ix <= layers; ++ix) {
        const double image_x = x + ix * shear.length_x;
        for (int iy = -layers; iy <= layers; ++iy) {
            const double image_y = y + iy * shear.length_y - ix * shear.reduced_shift;
            if (DomainManager::aabb_overlaps(
                    image_x - inflate, image_y - inflate, z - inflate,
                    image_x + inflate, image_y + inflate, z + inflate,
                    lo_x, lo_y, lo_z, hi_x, hi_y, hi_z)) return true;
        }
    }
    return false;
}

// =============================================================================
// LET Exchange Integration Point
// =============================================================================
// In main_multigpu.cu's timestep loop, the LET exchange runs inside
// AccelerationManager::compute_self_gravity_multigpu ->
// body_fitted_fmm::UnifiedFMMDispatcher::compute_multigpu:
//   1. Build local FMM octree + multipoles (compute())
//   2. Push LETs and publish the LET epoch through the phase coordinator
//   3. Wait on each peer's CUDA event and receiver-pull its exact LET segment
//   4. Evaluate aggregate receive segments directly, then acknowledge consumption
// The LETCommunicator instance is created per-GPU inside GPUDomainState or
// managed externally in main_multigpu.cu.
// =============================================================================

// =============================================================================
// Kernel Implementations
// =============================================================================

__global__ void classify_particles_orb_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    int32_t* __restrict__ owner_gpu,
    int32_t num_particles,
    int32_t num_gpus
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double x = pos_x[idx];
    double y = pos_y[idx];
    double z = pos_z[idx];

    const DomainParamsGPU& p = dev_domain_params;
    if (p.morton_enabled) {
        owner_gpu[idx] = DomainManager::position_to_gpu(x, y, z, p, 0);
        return;
    }
    for (int32_t g = 0; g < num_gpus; ++g) {
        if (x >= p.gpu_box_min_x[g] && x < p.gpu_box_max_x[g] &&
            y >= p.gpu_box_min_y[g] && y < p.gpu_box_max_y[g] &&
            z >= p.gpu_box_min_z[g] && z < p.gpu_box_max_z[g]) {
            owner_gpu[idx] = g;
            return;
        }
    }
    // Fallback: closest box center
    double best_dist = 1e30;
    int32_t best_gpu = 0;
    for (int32_t g = 0; g < num_gpus; ++g) {
        double cx = (p.gpu_box_min_x[g] + p.gpu_box_max_x[g]) * 0.5;
        double cy = (p.gpu_box_min_y[g] + p.gpu_box_max_y[g]) * 0.5;
        double cz = (p.gpu_box_min_z[g] + p.gpu_box_max_z[g]) * 0.5;
        double dx = x - cx, dy = y - cy, dz = z - cz;
        double dist = dx * dx + dy * dy + dz * dz;
        if (dist < best_dist) { best_dist = dist; best_gpu = g; }
    }
    owner_gpu[idx] = best_gpu;
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_local_particles) return;

    is_halo[idx] = 0;
    const int words = (num_gpus + 31) / 32;
    auto* destinations = reinterpret_cast<uint32_t*>(halo_dest_mask) + static_cast<size_t>(idx) * words;
    for (int word = 0; word < words; ++word) destinations[word] = 0;

    double x = pos_x[idx];
    double y = pos_y[idx];
    double z = pos_z[idx];
    const float extent = interaction_extent[idx] * interaction_extent_scale;
    double inflate = static_cast<double>(halo_width) + static_cast<double>(extent);

    const DomainParamsGPU& p = dev_domain_params;

    if (p.morton_enabled) {
        // Prefix cells are the exact (possibly disconnected) ownership cover.
        // A particle can touch several cells of one peer; send it only once.
        for (int c = 0; c < p.morton_cell_count; ++c) {
            const auto& cell = p.morton_cells[c];
            const int g = cell.owner;
            const uint32_t bit = uint32_t{1} << (g % 32);
            if (g == my_gpu_id || (destinations[g / 32] & bit)) continue;
            if (domain_halo_overlaps(x, y, z, inflate,
                    cell.lo[0], cell.lo[1], cell.lo[2],
                    cell.hi[0], cell.hi[1], cell.hi[2])) {
                destinations[g / 32] |= bit;
                is_halo[idx] = 1;
                if (send_counts) atomicAdd(send_counts + g, 1);
            }
        }
    } else {
        for (int32_t g = 0; g < num_gpus; ++g) {
            if (g == my_gpu_id) continue;
            if (domain_halo_overlaps(x, y, z, inflate,
                    p.gpu_box_min_x[g], p.gpu_box_min_y[g], p.gpu_box_min_z[g],
                    p.gpu_box_max_x[g], p.gpu_box_max_y[g], p.gpu_box_max_z[g])) {
                destinations[g / 32] |= uint32_t{1} << (g % 32);
                is_halo[idx] = 1;
                if (send_counts) atomicAdd(send_counts + g, 1);
            }
        }
    }
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double x = pos_x[idx];
    double y = pos_y[idx];
    double z = pos_z[idx];

    const DomainParamsGPU& p = dev_domain_params;

    const int32_t owner = DomainManager::position_to_gpu(x, y, z, p, my_gpu_id);

    if (owner != my_gpu_id) {
        is_migrating[idx] = 1;
        dest_gpu[idx] = owner;
        if (send_counts) atomicAdd(send_counts + owner, 1);
    } else {
        is_migrating[idx] = 0;
        dest_gpu[idx] = my_gpu_id;
    }
}

// =============================================================================
// DomainManager Implementation
// =============================================================================

namespace {
std::vector<uint64_t> host_morton_cuts;
std::vector<MortonCoverCell> host_morton_cells;
struct MortonDeviceStorage {
    uint64_t* cuts = nullptr;
    MortonCoverCell* cells = nullptr;
    size_t cut_capacity = 0, cell_capacity = 0;
};
std::vector<MortonDeviceStorage> morton_devices;
std::mutex morton_storage_mutex;
}

void DomainManager::set_morton_partition(DomainParamsGPU& params,
                                         const MortonGrid& grid,
                                         const std::vector<uint64_t>& cuts) {
    if (cuts.size() != static_cast<size_t>(params.num_gpus) + 1 ||
        cuts.front() != 0 || cuts.back() != grid.end_key() ||
        !std::is_sorted(cuts.begin(), cuts.end()))
        throw std::invalid_argument("invalid Morton ownership cuts");
    host_morton_cuts = cuts;
    host_morton_cells = morton_cover(grid, cuts);
    params.morton_enabled = 1;
    params.morton_grid = grid;
    params.morton_cuts = host_morton_cuts.data();
    params.morton_cells = host_morton_cells.data();
    params.morton_cell_count = static_cast<int>(host_morton_cells.size());
}

void DomainManager::copy_params_to_device(const DomainParamsGPU& params) {
    DomainParamsGPU device = params;
    if (params.morton_enabled && params.morton_cuts) {
        int rank = 0;
        CUDA_CHECK(cudaGetDevice(&rank));
        std::lock_guard<std::mutex> lock(morton_storage_mutex);
        if (morton_devices.size() <= static_cast<size_t>(rank))
            morton_devices.resize(static_cast<size_t>(rank) + 1);
        auto& storage = morton_devices[rank];
        const size_t cut_count = static_cast<size_t>(params.num_gpus) + 1;
        if (storage.cut_capacity < cut_count) {
            if (storage.cuts) CUDA_CHECK(cudaFree(storage.cuts));
            CUDA_CHECK(cudaMalloc(&storage.cuts, cut_count * sizeof(uint64_t)));
            storage.cut_capacity = cut_count;
        }
        if (storage.cell_capacity < static_cast<size_t>(params.morton_cell_count)) {
            if (storage.cells) CUDA_CHECK(cudaFree(storage.cells));
            CUDA_CHECK(cudaMalloc(&storage.cells, params.morton_cell_count * sizeof(MortonCoverCell)));
            storage.cell_capacity = params.morton_cell_count;
        }
        CUDA_CHECK(cudaMemcpy(storage.cuts, params.morton_cuts,
                             cut_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
        if (params.morton_cell_count)
            CUDA_CHECK(cudaMemcpy(storage.cells, params.morton_cells,
                                 params.morton_cell_count * sizeof(MortonCoverCell), cudaMemcpyHostToDevice));
        device.morton_cuts = storage.cuts;
        device.morton_cells = storage.cells;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(dev_domain_params, &device, sizeof(device)));
}

void DomainManager::apply_domain_params_to_states(
    std::vector<GPUDomainState>& gpu_states,
    const DomainParamsGPU& params)
{
    const int32_t num_gpus = static_cast<int32_t>(gpu_states.size());
    if (params.num_gpus != num_gpus) {
        throw std::runtime_error("DomainParamsGPU GPU count does not match domain states");
    }
    for (int32_t g = 0; g < num_gpus; ++g) {
        SubDomain& subdomain = gpu_states[g].subdomain;
        subdomain.box_min_x = params.morton_enabled ? params.domain_min_x : params.gpu_box_min_x[g];
        subdomain.box_min_y = params.morton_enabled ? params.domain_min_y : params.gpu_box_min_y[g];
        subdomain.box_min_z = params.morton_enabled ? params.domain_min_z : params.gpu_box_min_z[g];
        subdomain.box_max_x = params.morton_enabled ? params.domain_max_x : params.gpu_box_max_x[g];
        subdomain.box_max_y = params.morton_enabled ? params.domain_max_y : params.gpu_box_max_y[g];
        subdomain.box_max_z = params.morton_enabled ? params.domain_max_z : params.gpu_box_max_z[g];

        CUDA_CHECK(cudaSetDevice(g));
        copy_params_to_device(params);
    }
    // Preserve the legacy postcondition used by callers that immediately work
    // with GPU 0 after a rebalance.
    CUDA_CHECK(cudaSetDevice(0));
}

void DomainManager::apply_accepted_rcb_boxes(
    std::vector<GPUDomainState>& gpu_states,
    DomainParamsGPU& params,
    const RcbBox* boxes,
    int32_t box_count)
{
    const int32_t num_gpus = static_cast<int32_t>(gpu_states.size());
    if (!boxes || box_count != num_gpus || box_count <= 0 || box_count > MAX_MULTI_GPUS) {
        throw std::runtime_error("Accepted RCB boxes do not match active GPU domains");
    }
    if (params.num_gpus != num_gpus) {
        throw std::runtime_error("Accepted RCB boxes cannot be applied to mismatched DomainParamsGPU");
    }

    for (int32_t g = 0; g < std::min(num_gpus, MAX_MULTI_GPUS); ++g) {
        params.gpu_box_min_x[g] = boxes[g].min[0];
        params.gpu_box_min_y[g] = boxes[g].min[1];
        params.gpu_box_min_z[g] = boxes[g].min[2];
        params.gpu_box_max_x[g] = boxes[g].max[0];
        params.gpu_box_max_y[g] = boxes[g].max[1];
        params.gpu_box_max_z[g] = boxes[g].max[2];
    }
    apply_domain_params_to_states(gpu_states, params);
}

// =============================================================================
// ORB Split Computation (Host Utility)
// =============================================================================

// =============================================================================
// Initialization
// =============================================================================

DomainParamsGPU DomainManager::initialize(
    const SimConfig& config,
    int32_t num_gpus,
    std::vector<GPUDomainState>& gpu_states
) {
    gpu_states.resize(num_gpus);

    DomainParamsGPU params{};
    params.num_gpus = num_gpus;
    params.morton_enabled = config.shear_box.enabled || (config.multigpu.morton_ownership &&
        config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM);
    if (!params.morton_enabled && num_gpus > MAX_MULTI_GPUS)
        throw std::invalid_argument("legacy RCB supports at most eight GPUs");

    // Start with config bounds; auto_bounds will override in distribute_particles
    params.domain_min_x = config.domain.box_min.x;
    params.domain_min_y = config.domain.box_min.y;
    params.domain_min_z = config.domain.box_min.z;
    params.domain_max_x = config.domain.box_max.x;
    params.domain_max_y = config.domain.box_max.y;
    params.domain_max_z = config.domain.box_max.z;

    // Initialize the bounded legacy box representation only.
    for (int32_t g = 0; g < std::min(num_gpus, MAX_MULTI_GPUS); ++g) {
        params.gpu_box_min_x[g] = params.domain_min_x;
        params.gpu_box_min_y[g] = params.domain_min_y;
        params.gpu_box_min_z[g] = params.domain_min_z;
        params.gpu_box_max_x[g] = params.domain_max_x;
        params.gpu_box_max_y[g] = params.domain_max_y;
        params.gpu_box_max_z[g] = params.domain_max_z;
    }

    // Initialize each GPU's state
    for (int32_t g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));

        GPUDomainState& state = gpu_states[g];
        state.gpu_id = g;

        // Create streams
        CUDA_CHECK(cudaStreamCreate(&state.compute_stream));
        CUDA_CHECK(cudaStreamCreate(&state.comm_stream));

        // Initialize subdomain with default box
        state.subdomain.gpu_id = g;
        state.subdomain.box_min_x = params.morton_enabled ? params.domain_min_x : params.gpu_box_min_x[g];
        state.subdomain.box_min_y = params.morton_enabled ? params.domain_min_y : params.gpu_box_min_y[g];
        state.subdomain.box_min_z = params.morton_enabled ? params.domain_min_z : params.gpu_box_min_z[g];
        state.subdomain.box_max_x = params.morton_enabled ? params.domain_max_x : params.gpu_box_max_x[g];
        state.subdomain.box_max_y = params.morton_enabled ? params.domain_max_y : params.gpu_box_max_y[g];
        state.subdomain.box_max_z = params.morton_enabled ? params.domain_max_z : params.gpu_box_max_z[g];

        // Copy domain params to this GPU's constant memory
        copy_params_to_device(params);
        set_shear_time(config, 0.0);

        // Morton transport enables peer mappings lazily and stages payloads
        // when the device's mapping limit is reached. Do not eagerly consume
        // every mapping or leave a handled TooManyPeers error pending here.
        if (params.morton_enabled) continue;
        // Enable P2P access with other GPUs on the legacy path.
        for (int32_t peer = 0; peer < num_gpus; ++peer) {
            if (peer != g) {
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, g, peer));
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        std::cerr << "Warning: Could not enable P2P access from GPU "
                                  << g << " to GPU " << peer << std::endl;
                    }
                }
            }
        }
    }

    std::cout << "DomainManager initialized with " << num_gpus << " GPUs ("
              << (params.morton_enabled ? "Morton FMM ownership" : "ORB mode") << ")" << std::endl;

    return params;
}

// =============================================================================
// Initial Particle Distribution via ORB
// =============================================================================

void DomainManager::distribute_particles(
    const SimulationState& global_state,
    std::vector<GPUDomainState>& gpu_states,
    DomainParamsGPU& params,
    const SimConfig& config
) {
    int32_t num_gpus = static_cast<int32_t>(gpu_states.size());
    int32_t total_particles = global_state.particles.num_particles;


    // Step 1: Copy all particle positions from GPU 0 to host
    CUDA_CHECK(cudaSetDevice(0));

    std::vector<double> h_pos_x(total_particles);
    std::vector<double> h_pos_y(total_particles);
    std::vector<double> h_pos_z(total_particles);

    CUDA_CHECK(cudaMemcpy(h_pos_x.data(), global_state.particles.pos_x,
                          total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y.data(), global_state.particles.pos_y,
                          total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z.data(), global_state.particles.pos_z,
                          total_particles * sizeof(double), cudaMemcpyDeviceToHost));

    // Step 2: If auto_bounds, compute domain bounds from particles
    if (config.domain.auto_bounds) {
        double gmin_x = 1e30, gmin_y = 1e30, gmin_z = 1e30;
        double gmax_x = -1e30, gmax_y = -1e30, gmax_z = -1e30;
        for (int32_t i = 0; i < total_particles; ++i) {
            double px = h_pos_x[i];
            double py = h_pos_y[i];
            double pz = h_pos_z[i];
            gmin_x = fmin(gmin_x, px); gmax_x = fmax(gmax_x, px);
            gmin_y = fmin(gmin_y, py); gmax_y = fmax(gmax_y, py);
            gmin_z = fmin(gmin_z, pz); gmax_z = fmax(gmax_z, pz);
        }
        double margin_x = fmax((gmax_x - gmin_x) * 0.01, 0.01);
        double margin_y = fmax((gmax_y - gmin_y) * 0.01, 0.01);
        double margin_z = fmax((gmax_z - gmin_z) * 0.01, 0.01);
        params.domain_min_x = gmin_x - margin_x;
        params.domain_min_y = gmin_y - margin_y;
        params.domain_min_z = gmin_z - margin_z;
        params.domain_max_x = gmax_x + margin_x;
        params.domain_max_y = gmax_y + margin_y;
        params.domain_max_z = gmax_z + margin_z;
        std::cout << "Auto bounds: ["
                  << params.domain_min_x << ", " << params.domain_max_x << "] x ["
                  << params.domain_min_y << ", " << params.domain_max_y << "] x ["
                  << params.domain_min_z << ", " << params.domain_max_z << "]" << std::endl;
    }

    if (params.morton_enabled) {
        MortonGrid grid{0, 0, 0, 1, 3, 6, 0};
        grid.dimension = config.shear_box.enabled ? 2 : config.acceleration.unified_fmm.tree_dimension;
        grid.depth = std::max(2, config.acceleration.unified_fmm.tree_depth);
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            grid.depth = config.acceleration.unified_fmm.adaptive_depth.min_depth;
        if (config.multigpu.partition_depth > 0) grid.depth = config.multigpu.partition_depth;
        grid.size = std::max(params.domain_max_x - params.domain_min_x,
                            params.domain_max_y - params.domain_min_y);
        if (grid.dimension == 3)
            grid.size = std::max(grid.size, params.domain_max_z - params.domain_min_z);
        if (!(grid.size > 0)) grid.size = 1;
        grid.x = (params.domain_min_x + params.domain_max_x - grid.size) * 0.5;
        grid.y = (params.domain_min_y + params.domain_max_y - grid.size) * 0.5;
        grid.z = (params.domain_min_z + params.domain_max_z - grid.size) * 0.5;
        std::vector<MortonLeafWeight> weights;
        weights.reserve(total_particles);
        for (int i = 0; i < total_particles; ++i)
            weights.push_back({grid.leaf_begin(grid.key(h_pos_x[i], h_pos_y[i], h_pos_z[i])), 1.0});
        set_morton_partition(params, grid, weighted_morton_cuts(std::move(weights), grid, num_gpus));
    } else
    // Step 3: Compute initial N-way splits with the same RCB math used at runtime.
    // (Unit-weight here; t=0 scatter below is host-staged, a one-time cost.)
    {
        RcbBox root;
        root.min[0] = params.domain_min_x; root.max[0] = params.domain_max_x;
        root.min[1] = params.domain_min_y; root.max[1] = params.domain_max_y;
        root.min[2] = params.domain_min_z; root.max[2] = params.domain_max_z;

        HistogramFn histogram =
            [&](const RcbBox& box, int axis, int nbins, double lo, double hi, double* out) {
            for (int i = 0; i < nbins; ++i) out[i] = 0.0;
            double span = hi - lo;
            if (span <= 0.0) return;
            for (int32_t i = 0; i < total_particles; ++i) {
                double p[3] = {h_pos_x[i], h_pos_y[i], h_pos_z[i]};
                bool inside = true;
                for (int a = 0; a < 3; ++a)
                    if (!(p[a] >= box.min[a] && p[a] < box.max[a])) inside = false;
                if (!inside) continue;
                int b = static_cast<int>((p[axis] - lo) / span * nbins);
                if (b < 0) b = 0; if (b >= nbins) b = nbins - 1;
                out[b] += 1.0;
            }
        };

        RcbBox boxes[MAX_MULTI_GPUS];
        rcb_compute_boxes(root, num_gpus, config.multigpu.rcb_num_bins,
                          /*refine_passes=*/1, histogram, boxes);
        for (int32_t g = 0; g < num_gpus; ++g) {
            params.gpu_box_min_x[g] = boxes[g].min[0];
            params.gpu_box_min_y[g] = boxes[g].min[1];
            params.gpu_box_min_z[g] = boxes[g].min[2];
            params.gpu_box_max_x[g] = boxes[g].max[0];
            params.gpu_box_max_y[g] = boxes[g].max[1];
            params.gpu_box_max_z[g] = boxes[g].max[2];
        }
    }

    // Update constant memory on all GPUs
    for (int32_t g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        copy_params_to_device(params);
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Step 4: Classify particles on GPU 0 using the ORB kernel
    int32_t* d_gpu_assignment;
    CUDA_CHECK(cudaMalloc(&d_gpu_assignment, total_particles * sizeof(int32_t)));

    int block_size = 256;
    int num_blocks = (total_particles + block_size - 1) / block_size;

    classify_particles_orb_kernel<<<num_blocks, block_size>>>(
        global_state.particles.pos_x,
        global_state.particles.pos_y,
        global_state.particles.pos_z,
        d_gpu_assignment,
        total_particles,
        num_gpus
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Count particles per GPU
    std::vector<int32_t> h_gpu_assignment(total_particles);
    CUDA_CHECK(cudaMemcpy(h_gpu_assignment.data(), d_gpu_assignment,
                          total_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::vector<int32_t> counts(num_gpus, 0);
    for (int32_t i = 0; i < total_particles; ++i) {
        counts[h_gpu_assignment[i]]++;
    }

    for (int32_t g = 0; g < num_gpus; ++g) {
        gpu_states[g].subdomain.num_local_particles = counts[g];
        gpu_states[g].subdomain.box_min_x = params.morton_enabled ? params.domain_min_x : params.gpu_box_min_x[g];
        gpu_states[g].subdomain.box_min_y = params.morton_enabled ? params.domain_min_y : params.gpu_box_min_y[g];
        gpu_states[g].subdomain.box_min_z = params.morton_enabled ? params.domain_min_z : params.gpu_box_min_z[g];
        gpu_states[g].subdomain.box_max_x = params.morton_enabled ? params.domain_max_x : params.gpu_box_max_x[g];
        gpu_states[g].subdomain.box_max_y = params.morton_enabled ? params.domain_max_y : params.gpu_box_max_y[g];
        gpu_states[g].subdomain.box_max_z = params.morton_enabled ? params.domain_max_z : params.gpu_box_max_z[g];
    }

    // Step 6: Build index lists for each GPU
    std::vector<std::vector<int32_t>> particle_indices(num_gpus);
    for (int32_t g = 0; g < num_gpus; ++g) {
        particle_indices[g].reserve(counts[g]);
    }
    for (int32_t i = 0; i < total_particles; ++i) {
        particle_indices[h_gpu_assignment[i]].push_back(i);
    }

    // Step 7: Copy particle data from global_state to host
    std::vector<double> g_pos_x(total_particles), g_pos_y(total_particles), g_pos_z(total_particles);
    std::vector<double> g_vel_x(total_particles), g_vel_y(total_particles), g_vel_z(total_particles);
    std::vector<double> g_force_x(total_particles), g_force_y(total_particles), g_force_z(total_particles);
    std::vector<double> g_omega_x(total_particles), g_omega_y(total_particles), g_omega_z(total_particles);
    std::vector<double> g_torque_x(total_particles), g_torque_y(total_particles), g_torque_z(total_particles);
    std::vector<double> g_quat_w(total_particles), g_quat_x(total_particles), g_quat_y(total_particles), g_quat_z(total_particles);
    std::vector<float> g_radius(total_particles), g_mass(total_particles), g_moi(total_particles);
    std::vector<float> g_cohesive(total_particles), g_support_extent(total_particles);
    std::vector<double> g_density(total_particles), g_rho_rate(total_particles);
    std::vector<float> g_pressure(total_particles), g_viscosity(total_particles), g_c_sound(total_particles);
    std::vector<float> g_drhodt(total_particles), g_smoothing_length(total_particles);
    std::vector<double> g_energy(total_particles), g_dedt(total_particles);
    std::vector<double> g_energy_cold(total_particles);
    std::vector<double> g_energy_thermal_peak(total_particles);
    std::vector<double> g_pressure_peak(total_particles);
    std::vector<double> g_fracture_area(total_particles);
    std::vector<double> g_fragment_length(total_particles);
    std::vector<double> g_damping_energy(total_particles);

    std::vector<float> g_S_xx(total_particles), g_S_xy(total_particles), g_S_xz(total_particles);
    std::vector<float> g_S_yy(total_particles), g_S_yz(total_particles), g_S_zz(total_particles);
    std::vector<float> g_damage(total_particles), g_crack_strain(total_particles);
    std::vector<int32_t> g_flaws_ntot(total_particles);
    std::vector<float> g_flaws_strain_min(total_particles), g_flaws_strain_max(total_particles), g_flaws_m0(total_particles);
    std::vector<float> g_distension(total_particles);
    std::vector<float> g_L_00(total_particles), g_L_01(total_particles), g_L_02(total_particles);
    std::vector<float> g_L_10(total_particles), g_L_11(total_particles), g_L_12(total_particles);
    std::vector<float> g_L_20(total_particles), g_L_21(total_particles), g_L_22(total_particles);
    std::vector<int32_t> g_global_id(total_particles);
    std::vector<int32_t> g_material_id(total_particles);
    std::vector<int64_t> g_winding_x(total_particles), g_winding_y(total_particles), g_winding_z(total_particles);
    // Contextual identity: DEM clump membership or SPHSOL Part ID.  This must
    // survive the initial host-staged scatter just as it survives migration.
    std::vector<int32_t> g_clump_id(total_particles);

    const auto& src = global_state.particles;
    CUDA_CHECK(cudaMemcpy(g_pos_x.data(), src.pos_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_pos_y.data(), src.pos_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_pos_z.data(), src.pos_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_vel_x.data(), src.vel_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_vel_y.data(), src.vel_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_vel_z.data(), src.vel_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_force_x.data(), src.force_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_force_y.data(), src.force_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_force_z.data(), src.force_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_omega_x.data(), src.omega_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_omega_y.data(), src.omega_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_omega_z.data(), src.omega_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_torque_x.data(), src.torque_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_torque_y.data(), src.torque_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_torque_z.data(), src.torque_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_quat_w.data(), src.quat_w, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_quat_x.data(), src.quat_x, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_quat_y.data(), src.quat_y, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_quat_z.data(), src.quat_z, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_radius.data(), src.radius, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_mass.data(), src.mass, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_moi.data(), src.moment_of_inertia, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_cohesive.data(), src.cohesive_strength, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_support_extent.data(), src.support_extent, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_density.data(), src.density, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_pressure.data(), src.pressure, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_viscosity.data(), src.viscosity, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_c_sound.data(), src.c_sound, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_drhodt.data(), src.drhodt, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_rho_rate.data(), src.rho_rate, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_smoothing_length.data(), src.smoothing_length, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_energy.data(), src.energy, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_energy_cold.data(), src.energy_cold, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_energy_thermal_peak.data(), src.energy_thermal_peak, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_pressure_peak.data(), src.pressure_peak, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_fracture_area.data(), src.fracture_area, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_fragment_length.data(), src.fragment_length, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_damping_energy.data(), src.damping_energy, total_particles * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(g_dedt.data(), src.dedt, total_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_xx.data(), src.S_xx, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_xy.data(), src.S_xy, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_xz.data(), src.S_xz, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_yy.data(), src.S_yy, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_yz.data(), src.S_yz, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_S_zz.data(), src.S_zz, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_damage.data(), src.damage, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_crack_strain.data(), src.crack_strain, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_flaws_ntot.data(), src.flaws_ntot, total_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_flaws_strain_min.data(), src.flaws_strain_min, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_flaws_strain_max.data(), src.flaws_strain_max, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_flaws_m0.data(), src.flaws_m0, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_distension.data(), src.distension, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_00.data(), src.L_00, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_01.data(), src.L_01, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_02.data(), src.L_02, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_10.data(), src.L_10, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_11.data(), src.L_11, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_12.data(), src.L_12, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_20.data(), src.L_20, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_21.data(), src.L_21, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_L_22.data(), src.L_22, total_particles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_global_id.data(), src.global_id, total_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_material_id.data(), src.material_id, total_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_winding_x.data(), src.winding_x, total_particles * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_winding_y.data(), src.winding_y, total_particles * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_winding_z.data(), src.winding_z, total_particles * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_clump_id.data(), src.clump_id, total_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Step 8: Allocate and copy to each GPU
    // Material tables are dense but runtime-sized.  Replicate the same
    // immutable database on every GPU rather than trying to use a cross-device
    // pointer (which is not valid for normal kernel reads on all systems).
    const MaterialDatabase material_database = load_material_database(
        config.material_database.path, config.dem_damping_model);
    for (int32_t gpu = 0; gpu < num_gpus; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));

        int32_t n = counts[gpu];
        auto& dst = gpu_states[gpu].local_state.particles;
        gpu_states[gpu].local_state.material_database =
            upload_material_database(material_database, gpu);

        if (config.multigpu.buffer_capacity_factor < 1.0f) {
            throw std::runtime_error("multigpu.buffer_capacity_factor must be >= 1.0");
        }
        // Capacity is local by design.  Migration paths must call the grow-only
        // phase-boundary resize before writing beyond this allocation.
        const double scaled = std::ceil(static_cast<double>(n) * config.multigpu.buffer_capacity_factor);
        if (n > std::numeric_limits<int32_t>::max() - 1000 ||
            scaled > static_cast<double>(std::numeric_limits<int32_t>::max() - 1000)) {
            throw std::runtime_error("Initial local ParticleSoA capacity overflows int32");
        }
        int32_t capacity = std::max(static_cast<int32_t>(scaled), n + 1000);
        allocate_particle_soa(dst, capacity);
        dst.num_particles = n;
        dst.num_local = n;
        dst.num_ghost = 0;
        dst.capacity = capacity;
        dst.input_has_winding = src.input_has_winding;
        dst.source_record_kind = src.source_record_kind;
        gpu_states[gpu].local_state.current_time = global_state.current_time;
        gpu_states[gpu].local_state.current_step = global_state.current_step;
        gpu_states[gpu].local_state.dt = global_state.dt;
        set_shear_time(config, global_state.current_time);

        gpu_states[gpu].subdomain.capacity = capacity;

        if (n == 0) continue;

        // Build per-GPU arrays
        std::vector<double> p_x(n), p_y(n), p_z(n);
        std::vector<double> v_x(n), v_y(n), v_z(n);
        std::vector<double> f_x(n), f_y(n), f_z(n);
        std::vector<double> w_x(n), w_y(n), w_z(n);
        std::vector<double> t_x(n), t_y(n), t_z(n);
        std::vector<double> q_w(n), q_x(n), q_y(n), q_z(n);
        std::vector<float> r(n), m(n), mi(n), c(n), support_extent(n);
        std::vector<double> density(n), rho_rate(n);
        std::vector<float> pressure(n), viscosity(n), c_sound(n);
        std::vector<float> drhodt(n), smoothing_length(n);
        std::vector<double> energy(n), dedt(n);
        std::vector<double> energy_cold(n);
        std::vector<double> energy_thermal_peak(n);
        std::vector<double> pressure_peak(n);
        std::vector<double> fracture_area(n);
        std::vector<double> fragment_length(n);
        std::vector<double> damping_energy(n);

        std::vector<float> S_xx(n), S_xy(n), S_xz(n), S_yy(n), S_yz(n), S_zz(n);
        std::vector<float> damage(n), crack_strain(n), flaws_strain_min(n), flaws_strain_max(n), flaws_m0(n), distension(n);
        std::vector<int32_t> flaws_ntot(n);
        std::vector<float> L_00(n), L_01(n), L_02(n), L_10(n), L_11(n), L_12(n), L_20(n), L_21(n), L_22(n);
        std::vector<int32_t> id(n), material_id(n), clump_id(n);
        std::vector<int64_t> winding_x(n), winding_y(n), winding_z(n);

        for (int32_t local_idx = 0; local_idx < n; ++local_idx) {
            int32_t global_idx = particle_indices[gpu][local_idx];
            p_x[local_idx] = g_pos_x[global_idx];
            p_y[local_idx] = g_pos_y[global_idx];
            p_z[local_idx] = g_pos_z[global_idx];
            v_x[local_idx] = g_vel_x[global_idx];
            v_y[local_idx] = g_vel_y[global_idx];
            v_z[local_idx] = g_vel_z[global_idx];
            f_x[local_idx] = g_force_x[global_idx];
            f_y[local_idx] = g_force_y[global_idx];
            f_z[local_idx] = g_force_z[global_idx];
            w_x[local_idx] = g_omega_x[global_idx];
            w_y[local_idx] = g_omega_y[global_idx];
            w_z[local_idx] = g_omega_z[global_idx];
            t_x[local_idx] = g_torque_x[global_idx];
            t_y[local_idx] = g_torque_y[global_idx];
            t_z[local_idx] = g_torque_z[global_idx];
            q_w[local_idx] = g_quat_w[global_idx];
            q_x[local_idx] = g_quat_x[global_idx];
            q_y[local_idx] = g_quat_y[global_idx];
            q_z[local_idx] = g_quat_z[global_idx];
            r[local_idx] = g_radius[global_idx];
            m[local_idx] = g_mass[global_idx];
            mi[local_idx] = g_moi[global_idx];
            c[local_idx] = g_cohesive[global_idx];
            support_extent[local_idx] = g_support_extent[global_idx];
            density[local_idx] = g_density[global_idx];
            pressure[local_idx] = g_pressure[global_idx];
            viscosity[local_idx] = g_viscosity[global_idx];
            c_sound[local_idx] = g_c_sound[global_idx];
            drhodt[local_idx] = g_drhodt[global_idx];
            rho_rate[local_idx] = g_rho_rate[global_idx];
            smoothing_length[local_idx] = g_smoothing_length[global_idx];
            energy[local_idx] = g_energy[global_idx];
            energy_cold[local_idx] = g_energy_cold[global_idx];
            energy_thermal_peak[local_idx] = g_energy_thermal_peak[global_idx];
            pressure_peak[local_idx] = g_pressure_peak[global_idx];
            fracture_area[local_idx] = g_fracture_area[global_idx];
            fragment_length[local_idx] = g_fragment_length[global_idx];
            damping_energy[local_idx] = g_damping_energy[global_idx];

            dedt[local_idx] = g_dedt[global_idx];
            S_xx[local_idx] = g_S_xx[global_idx];
            S_xy[local_idx] = g_S_xy[global_idx];
            S_xz[local_idx] = g_S_xz[global_idx];
            S_yy[local_idx] = g_S_yy[global_idx];
            S_yz[local_idx] = g_S_yz[global_idx];
            S_zz[local_idx] = g_S_zz[global_idx];
            damage[local_idx] = g_damage[global_idx];
            crack_strain[local_idx] = g_crack_strain[global_idx];
            flaws_ntot[local_idx] = g_flaws_ntot[global_idx];
            flaws_strain_min[local_idx] = g_flaws_strain_min[global_idx];
            flaws_strain_max[local_idx] = g_flaws_strain_max[global_idx];
            flaws_m0[local_idx] = g_flaws_m0[global_idx];
            distension[local_idx] = g_distension[global_idx];
            L_00[local_idx] = g_L_00[global_idx]; L_01[local_idx] = g_L_01[global_idx]; L_02[local_idx] = g_L_02[global_idx];
            L_10[local_idx] = g_L_10[global_idx]; L_11[local_idx] = g_L_11[global_idx]; L_12[local_idx] = g_L_12[global_idx];
            L_20[local_idx] = g_L_20[global_idx]; L_21[local_idx] = g_L_21[global_idx]; L_22[local_idx] = g_L_22[global_idx];
            id[local_idx] = g_global_id[global_idx];
            material_id[local_idx] = g_material_id[global_idx];
            winding_x[local_idx] = g_winding_x[global_idx];
            winding_y[local_idx] = g_winding_y[global_idx];
            winding_z[local_idx] = g_winding_z[global_idx];
            clump_id[local_idx] = g_clump_id[global_idx];
        }

        // Copy to device
        CUDA_CHECK(cudaMemcpy(dst.pos_x, p_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.pos_y, p_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.pos_z, p_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.winding_x, winding_x.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.winding_y, winding_y.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.winding_z, winding_z.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.vel_x, v_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.vel_y, v_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.vel_z, v_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.force_x, f_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.force_y, f_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.force_z, f_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.omega_x, w_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.omega_y, w_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.omega_z, w_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.torque_x, t_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.torque_y, t_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.torque_z, t_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.quat_w, q_w.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.quat_x, q_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.quat_y, q_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.quat_z, q_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.radius, r.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.mass, m.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.moment_of_inertia, mi.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.cohesive_strength, c.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.support_extent, support_extent.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.density, density.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.pressure, pressure.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.viscosity, viscosity.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.c_sound, c_sound.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.drhodt, drhodt.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.rho_rate, rho_rate.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.smoothing_length, smoothing_length.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.energy, energy.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.energy_cold, energy_cold.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.energy_thermal_peak, energy_thermal_peak.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.pressure_peak, pressure_peak.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.fracture_area, fracture_area.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.fragment_length, fragment_length.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.damping_energy, damping_energy.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(dst.dedt, dedt.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_xx, S_xx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_xy, S_xy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_xz, S_xz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_yy, S_yy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_yz, S_yz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.S_zz, S_zz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.damage, damage.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.crack_strain, crack_strain.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.flaws_ntot, flaws_ntot.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.flaws_strain_min, flaws_strain_min.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.flaws_strain_max, flaws_strain_max.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.flaws_m0, flaws_m0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.distension, distension.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_00, L_00.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_01, L_01.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_02, L_02.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_10, L_10.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_11, L_11.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_12, L_12.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_20, L_20.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_21, L_21.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.L_22, L_22.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.global_id, id.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.material_id, material_id.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.clump_id, clump_id.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

        // Initialize last_pos to current pos
        CUDA_CHECK(cudaMemcpy(dst.last_pos_x, p_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.last_pos_y, p_y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.last_pos_z, p_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // Initialize contact history offsets
        std::vector<int32_t> zero_contacts(n, 0);
        std::vector<int32_t> offsets(n);
        for (int32_t i = 0; i < n; ++i) offsets[i] = i * 16;
        CUDA_CHECK(cudaMemcpy(dst.num_contacts, zero_contacts.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst.contact_history_offset, offsets.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d_gpu_assignment));
}

// =============================================================================
// Load Balancing
// =============================================================================

void DomainManager::update_load_balance(
    std::vector<GPUDomainState>& gpu_states,
    DomainParamsGPU& params,
    RCBBalancer& balancer
) {
    int32_t num_gpus = static_cast<int32_t>(gpu_states.size());
    if (num_gpus <= 1) return;

    // Compute new disjoint AABBs entirely on-GPU (distributed histograms; no host
    // particle gather). Particles are redistributed by the next migration pass.
    balancer.compute_splits(gpu_states, params);

    // Legacy balancing and accepted adaptive proposals share the exact same
    // subdomain/constant-memory commit boundary.
    apply_domain_params_to_states(gpu_states, params);
}

// =============================================================================
// Shutdown
// =============================================================================

void DomainManager::shutdown(std::vector<GPUDomainState>& gpu_states) {
    std::lock_guard<std::mutex> lock(morton_storage_mutex);
    for (auto& state : gpu_states) {
        CUDA_CHECK(cudaSetDevice(state.gpu_id));

        if (state.compute_stream) {
            cudaStreamDestroy(state.compute_stream);
            state.compute_stream = nullptr;
        }
        if (state.comm_stream) {
            cudaStreamDestroy(state.comm_stream);
            state.comm_stream = nullptr;
        }

        if (state.d_owner_gpu) {
            cudaFree(state.d_owner_gpu);
            state.d_owner_gpu = nullptr;
        }
        if (state.d_local_to_global) {
            cudaFree(state.d_local_to_global);
            state.d_local_to_global = nullptr;
        }
        if (state.d_global_to_local) {
            cudaFree(state.d_global_to_local);
            state.d_global_to_local = nullptr;
        }
        if (state.d_is_halo) {
            cudaFree(state.d_is_halo);
            state.d_is_halo = nullptr;
        }
        if (state.d_is_migrating) {
            cudaFree(state.d_is_migrating);
            state.d_is_migrating = nullptr;
        }
        release_material_database(&state.local_state.material_database);
        if (static_cast<size_t>(state.gpu_id) < morton_devices.size()) {
            auto& storage = morton_devices[state.gpu_id];
            if (storage.cuts) CUDA_CHECK(cudaFree(storage.cuts));
            if (storage.cells) CUDA_CHECK(cudaFree(storage.cells));
            storage = {};
        }
    }

    gpu_states.clear();
    std::cout << "DomainManager shutdown complete" << std::endl;
}

} // namespace cudadem

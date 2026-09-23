// =============================================================================
// Direct O(N^2) Neighbor Search Kernels
// =============================================================================
// Brute-force contact detection checking all particle pairs.
// Each thread processes one unique pair (i, j) where j > i.

#include "direct_manager.h"
#include "../../constant/system_initialize.h"
#include <cuda_runtime.h>
#include <cassert>

namespace cudadem {
namespace direct {

// =============================================================================
// O(N^2) Contact Detection Kernel
// =============================================================================
// Each thread checks one unique pair (i, j) where j > i
// Linear index k maps to pair (i, j) via triangular number computation

__global__ void direct_find_contacts_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ radius,
    int32_t num_particles,
    ContactPair* __restrict__ contacts,
    int32_t* __restrict__ num_contacts,
    int32_t max_contacts
) {
    // Total pairs: N*(N-1)/2
    const int64_t total_pairs = (int64_t)num_particles * (num_particles - 1) / 2;
    const int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_idx >= total_pairs) return;

    // Convert linear index to pair (i, j)
    // Find i such that: i*(2N - i - 1)/2 <= pair_idx < (i+1)*(2N - i - 2)/2
    // Using iterative search (O(N) worst case, acceptable for validation)
    int32_t i = 0;
    int64_t remaining = pair_idx;
    while (i < num_particles - 1) {
        int64_t pairs_in_row = num_particles - 1 - i;
        if (remaining < pairs_in_row) break;
        remaining -= pairs_in_row;
        i++;
    }
    int32_t j = i + 1 + (int32_t)remaining;

    // Load positions (double precision)
    double pi_x = pos_x[i];
    double pi_y = pos_y[i];
    double pi_z = pos_z[i];
    double pj_x = pos_x[j];
    double pj_y = pos_y[j];
    double pj_z = pos_z[j];

    // Compute distance vector
    double dx = pj_x - pi_x;
    double dy = pj_y - pi_y;
    double dz = pj_z - pi_z;
    double dist_sq = dx*dx + dy*dy + dz*dz;
    double dist = sqrt(dist_sq);

    // Compute overlap (cast radii to double for precision matching with CLL/LBVH)
    double sum_radii = (double)radius[i] + (double)radius[j];
    double overlap = sum_radii - dist;

    // Check for contact
    if (overlap > 0.0) {
        // Store contact atomically
        int32_t idx = atomicAdd(num_contacts, 1);
        if (idx < max_contacts) {
            contacts[idx] = ContactPair(i, j);
        }
    }
}

// =============================================================================
// Host Launch Function
// =============================================================================

void launch_direct_find_contacts(
    const double* pos_x,
    const double* pos_y,
    const double* pos_z,
    const float* radius,
    int32_t num_particles,
    ContactPair* contacts,
    int32_t* num_contacts,
    int32_t max_contacts,
    cudaStream_t stream
) {
    const int64_t total_pairs = (int64_t)num_particles * (num_particles - 1) / 2;
    const int32_t block_size = 256;
    const int64_t num_blocks_64 = (total_pairs + block_size - 1) / block_size;

    // Safety check for grid dimension overflow
    assert(num_blocks_64 <= INT32_MAX && "Too many pairs for single kernel launch");

    // Clear contact counter
    cudaMemsetAsync(num_contacts, 0, sizeof(int32_t), stream);

    // Launch kernel
    direct_find_contacts_kernel<<<(int32_t)num_blocks_64, block_size, 0, stream>>>(
        pos_x, pos_y, pos_z, radius,
        num_particles, contacts, num_contacts, max_contacts
    );
}

// =============================================================================
// Static Member Definitions
// =============================================================================

int32_t DirectSearchImpl::s_initialized_gpu = -1;
NeighborSearchStats DirectSearchImpl::s_stats;

// =============================================================================
// Lifecycle Implementation
// =============================================================================

void DirectSearchImpl::initialize(const SimConfig& config, SimulationState& state) {
    s_initialized_gpu = config.gpu.device_id;

    // Mark NeighborData as using DIRECT method
    state.neighbor.active_method = NeighborMethod::DIRECT;
    state.neighbor.num_particles = state.particles.num_particles;

    // No data structures to allocate for direct method
    s_stats = NeighborSearchStats();
}

void DirectSearchImpl::shutdown(SimulationState& state) {
    s_initialized_gpu = -1;
}

void DirectSearchImpl::shutdown(SimulationState& state, int32_t gpu_id) {
    if (s_initialized_gpu == gpu_id) {
        s_initialized_gpu = -1;
    }
}

void DirectSearchImpl::shutdown_all() {
    s_initialized_gpu = -1;
}

// =============================================================================
// Contact Detection Implementation
// =============================================================================

void DirectSearchImpl::find_contacts(
    SimulationState& state,
    const SimConfig& config,
    ContactListBuffer& contacts,
    cudaStream_t stream
) {
    launch_direct_find_contacts(
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        // Conservative candidates; the DEM force kernel applies the exact
        // material-pair cohesive cutoff afterwards.
        state.particles.support_extent,
        state.particles.num_particles,
        contacts.pairs,
        contacts.num_contacts,
        contacts.max_contacts,
        stream
    );

    // Synchronize and read back contact count
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(&contacts.h_num_contacts, contacts.num_contacts,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    s_stats.num_contacts_found = contacts.h_num_contacts;
}

// =============================================================================
// Query Implementation
// =============================================================================

bool DirectSearchImpl::is_initialized() {
    return s_initialized_gpu >= 0;
}

bool DirectSearchImpl::is_initialized(int32_t gpu_id) {
    return s_initialized_gpu == gpu_id;
}

const NeighborSearchStats& DirectSearchImpl::get_stats() {
    return s_stats;
}

const NeighborSearchStats& DirectSearchImpl::get_stats(int32_t gpu_id) {
    return s_stats;
}

} // namespace direct
} // namespace cudadem

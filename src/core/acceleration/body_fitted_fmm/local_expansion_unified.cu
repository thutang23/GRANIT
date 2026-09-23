// =============================================================================
// L2L + L2P Unified Kernels (uses UnifiedNode + children_compact)
// =============================================================================
// Included only by body_fitted_fmm_manager.cu.
// =============================================================================

#include "local_expansion_math.cuh"

namespace body_fitted_fmm {

constexpr int LU_BLOCK_SIZE = 256;

// The host view carries allocation capacity; the completed tree's actual
// count stays on the GPU. Clear every coefficient (including padding) in the
// live prefix before M2L/L2L accumulate into it, without a host count readback.
__global__ void fmm_zero_active_local_expansions_kernel(
    fmm::LocalExpansion* local_expansions,
    const PersistentTreeMetadata* metadata,
    int node_capacity)
{
    const int nodes = metadata->num_nodes;
    if (metadata->error != 0) return;
    if (nodes < 0 || nodes > node_capacity) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("BF-FMM active local-expansion count exceeds capacity: %d/%d\n",
                   nodes, node_capacity);
            asm volatile("trap;");
        }
        return;
    }
    const size_t count = static_cast<size_t>(nodes) * fmm::FMM_MAX_COEFFS;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        local_expansions[i / fmm::FMM_MAX_COEFFS].coeffs[i % fmm::FMM_MAX_COEFFS] =
            FAR_REAL(0.0);
    }
}

void zero_active_local_expansions(
    const fmm::FMMDeviceData& dd,
    const PersistentTreeMetadata* metadata,
    cudaStream_t stream)
{
    if (dd.num_nodes <= 0) return;
    const size_t coefficients = static_cast<size_t>(dd.num_nodes) * fmm::FMM_MAX_COEFFS;
    const int blocks = static_cast<int>(std::min<size_t>(4096,
        (coefficients + LU_BLOCK_SIZE - 1) / LU_BLOCK_SIZE));
    fmm_zero_active_local_expansions_kernel<<<blocks, LU_BLOCK_SIZE, 0, stream>>>(
        dd.local_expansions, metadata, dd.num_nodes);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Device Function: L2L Translation for Trace-Free p=4
// (Cell-size-normalized: parent in normalized form, child accumulated normalized)
// =============================================================================

// =============================================================================
// Kernel: L2L Unified (uses get_child_idx)
// =============================================================================

__global__ void fmm_l2l_kernel_unified(
    const UnifiedNode* __restrict__ nodes,
    fmm::LocalExpansion* __restrict__ local_expansions,
    const int* __restrict__ children_compact,
    const int* __restrict__ level_list,
    const int* __restrict__ level_start,
    const int* __restrict__ level_count,
    int current_level,
    int num_nodes,
    const double* __restrict__ device_root_cell_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lvl_count = level_count[current_level];
    if (tid >= lvl_count) return;

    int lvl_start = level_start[current_level];
    int node_idx = level_list[lvl_start + tid];
    if (node_idx < 0 || node_idx >= num_nodes) return;

    const UnifiedNode& node = nodes[node_idx];
    if (node.target_count == 0) return;
    if (node.childBase == body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE) return;

    const fmm::LocalExpansion& parent_L = local_expansions[node_idx];

    for (int o = 0; o < 8; o++) {
        int child_idx = get_child_idx<3>(nodes, children_compact, node_idx, o);
        if (child_idx < 0 || child_idx >= num_nodes) continue;

        const UnifiedNode& child = nodes[child_idx];
        if (child.target_count == 0) continue;
        far_real_t dx = FAR_CAST(child.center.x - node.center.x);
        far_real_t dy = FAR_CAST(child.center.y - node.center.y);
        far_real_t dz = FAR_CAST(child.center.z - node.center.z);

        // Cell size from level: h = root_cell_size / 2^level
        // (correct for ALL tree builders, not just Karras)
        const far_real_t root_cell_size =
            static_cast<far_real_t>(*device_root_cell_size);
        far_real_t hp = root_cell_size / (far_real_t)(1ull << node.level);
        far_real_t hc = hp * FAR_REAL(0.5);  // h_child = h_parent / 2 for all regular trees

        l2l_translate_coefficients_unified(
            parent_L.coeffs,
            local_expansions[child_idx].coeffs,
            dx, dy, dz, hp, hc);
    }
}

// =============================================================================
// Device Function: Evaluate Local Expansion Force (same math as local_expansion.cu)
// =============================================================================

// =============================================================================
// Kernel: L2P Unified (uses UnifiedNode)
// =============================================================================

__global__ void fmm_l2p_kernel_unified(
    const UnifiedNode* __restrict__ nodes,
    const fmm::LocalExpansion* __restrict__ local_expansions,
    const far_real_t* __restrict__ pos_x,
    const far_real_t* __restrict__ pos_y,
    const far_real_t* __restrict__ pos_z,
    const float* __restrict__ mass,
    far_real_t* __restrict__ force_x,
    far_real_t* __restrict__ force_y,
    far_real_t* __restrict__ force_z,
    const int* __restrict__ sorted_to_original,
    const int* __restrict__ particle_to_leaf,
    int num_particles,
    int num_nodes,
    int target_particle_count,
    far_real_t G,
    const double* __restrict__ device_root_cell_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    int sorted_idx = tid;
    int orig_idx = sorted_to_original[sorted_idx];
    if (orig_idx >= target_particle_count) return;
    int leaf_idx = particle_to_leaf[sorted_idx];
    if (leaf_idx < 0) return;

    const UnifiedNode& leaf = nodes[leaf_idx];
    const fmm::LocalExpansion& L = local_expansions[leaf_idx];

    far_real_t dx = pos_x[orig_idx] - static_cast<far_real_t>(leaf.center.x);
    far_real_t dy = pos_y[orig_idx] - static_cast<far_real_t>(leaf.center.y);
    far_real_t dz = pos_z[orig_idx] - static_cast<far_real_t>(leaf.center.z);

    far_real_t fx, fy, fz;
    // Cell size from level: h = root_cell_size / 2^level
    const far_real_t root_cell_size =
        static_cast<far_real_t>(*device_root_cell_size);
    far_real_t hl = root_cell_size / (far_real_t)(1ull << leaf.level);
    evaluate_local_expansion_force_unified(L.coeffs, dx, dy, dz, fx, fy, fz, hl);

    far_real_t mi = static_cast<far_real_t>(mass[orig_idx]);
    atomicAdd(&force_x[orig_idx], G * mi * fx);
    atomicAdd(&force_y[orig_idx], G * mi * fy);
    atomicAdd(&force_z[orig_idx], G * mi * fz);
}

// Outer variant (acceleration, no mass multiplication)
__global__ void evaluate_l2p_outer_kernel_unified(
    const UnifiedNode* __restrict__ nodes,
    const fmm::LocalExpansion* __restrict__ local_expansions,
    const far_real_t* __restrict__ pos_x,
    const far_real_t* __restrict__ pos_y,
    const far_real_t* __restrict__ pos_z,
    const float* __restrict__ mass,
    far_real_t* __restrict__ grav_acc_x,
    far_real_t* __restrict__ grav_acc_y,
    far_real_t* __restrict__ grav_acc_z,
    const int* __restrict__ sorted_to_original,
    const int* __restrict__ particle_to_leaf,
    int num_particles,
    int num_nodes,
    int target_particle_count,
    far_real_t G,
    const double* __restrict__ device_root_cell_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    int sorted_idx = tid;
    int orig_idx = sorted_to_original[sorted_idx];
    if (orig_idx >= target_particle_count) return;
    int leaf_idx = particle_to_leaf[sorted_idx];
    if (leaf_idx < 0) return;

    const UnifiedNode& leaf = nodes[leaf_idx];
    const fmm::LocalExpansion& L = local_expansions[leaf_idx];

    far_real_t dx = pos_x[orig_idx] - static_cast<far_real_t>(leaf.center.x);
    far_real_t dy = pos_y[orig_idx] - static_cast<far_real_t>(leaf.center.y);
    far_real_t dz = pos_z[orig_idx] - static_cast<far_real_t>(leaf.center.z);

    far_real_t fx, fy, fz;
    // Cell size from level: h = root_cell_size / 2^level
    const far_real_t root_cell_size =
        static_cast<far_real_t>(*device_root_cell_size);
    far_real_t hl = root_cell_size / (far_real_t)(1ull << leaf.level);
    evaluate_local_expansion_force_unified(L.coeffs, dx, dy, dz, fx, fy, fz, hl);

    atomicAdd(&grav_acc_x[orig_idx], G * fx);
    atomicAdd(&grav_acc_y[orig_idx], G * fy);
    atomicAdd(&grav_acc_z[orig_idx], G * fz);
}

// =============================================================================
// Host Functions: Run L2L Pass and L2P Evaluation (Unified)
// =============================================================================

void run_l2l_pass_unified(
    const fmm::FMMDeviceData& dd,
    const double* device_root_cell_size,
    cudaStream_t stream)
{
    const int BLOCK = LU_BLOCK_SIZE;
    if (dd.num_nodes <= 0 || dd.max_level <= 0) return;

    for (int level = 0; level < dd.max_level; level++) {
        // Capacity launch; the kernel reads the actual level count on device.
        int grid = (dd.num_nodes + BLOCK - 1) / BLOCK;
        fmm_l2l_kernel_unified<<<grid, BLOCK, 0, stream>>>(
            dd.unified_nodes, dd.local_expansions, dd.children_compact,
            dd.level_list, dd.level_start, dd.level_count,
            level, dd.num_nodes, device_root_cell_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

void run_l2p_evaluation_unified(
    const fmm::FMMDeviceData& dd,
    const far_real_t* pos_x, const far_real_t* pos_y, const far_real_t* pos_z,
    const float* mass,
    far_real_t* force_x, far_real_t* force_y, far_real_t* force_z,
    const int* particle_to_leaf,
    int target_particle_count,
    far_real_t G,
    const double* device_root_cell_size,
    cudaStream_t stream)
{
    const int BLOCK = LU_BLOCK_SIZE;
    int N = dd.num_particles;
    if (N <= 0 || dd.num_nodes <= 0) return;
    int grid = (N + BLOCK - 1) / BLOCK;
    fmm_l2p_kernel_unified<<<grid, BLOCK, 0, stream>>>(
        dd.unified_nodes, dd.local_expansions,
        pos_x, pos_y, pos_z, mass,
        force_x, force_y, force_z,
        dd.sorted_to_original, particle_to_leaf,
        N, dd.num_nodes, target_particle_count, G, device_root_cell_size);
    CUDA_CHECK(cudaGetLastError());
}

void run_l2p_evaluation_outer_unified(
    const fmm::FMMDeviceData& dd,
    const far_real_t* pos_x, const far_real_t* pos_y, const far_real_t* pos_z,
    const float* mass,
    far_real_t* grav_acc_x, far_real_t* grav_acc_y, far_real_t* grav_acc_z,
    const int* particle_to_leaf,
    int target_particle_count,
    far_real_t G,
    const double* device_root_cell_size,
    cudaStream_t stream)
{
    const int BLOCK = LU_BLOCK_SIZE;
    int N = dd.num_particles;
    if (N <= 0 || dd.num_nodes <= 0) return;
    int grid = (N + BLOCK - 1) / BLOCK;
    evaluate_l2p_outer_kernel_unified<<<grid, BLOCK, 0, stream>>>(
        dd.unified_nodes, dd.local_expansions,
        pos_x, pos_y, pos_z, mass,
        grav_acc_x, grav_acc_y, grav_acc_z,
        dd.sorted_to_original, particle_to_leaf,
        N, dd.num_nodes, target_particle_count, G, device_root_cell_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace body_fitted_fmm

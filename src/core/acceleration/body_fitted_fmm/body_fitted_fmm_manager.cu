#include "src/core/sph/sph_physics.h"
#include "src/core/sph/sph_gravity_correction.h"
#ifdef GRANIT_FMM_TEST_HOOKS
#include <thread>
#endif
#include <chrono>
// =============================================================================
// Body-Fitted FMM Manager — Full Pipeline Implementation (Step 7)
// =============================================================================
// Orchestrates the complete body-fitted FMM pipeline.
// Structure mirrors lbvh_fmm_manager.cu: all .cu files are #included into this
// single translation unit inside `namespace cudadem {}`, avoiding symbol
// duplication and namespace mismatch.
// =============================================================================

#include "../acceleration.h"
#include "../../particle_model/particle.h"
#include "../../wall/wall.h"
#include "cuda_utils.h"
#include "../shear_gravity_images.cuh"
#include "shear_fmm_capacity.h"
#include "g_table_binary.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <climits>
#include <cstddef>
#include <limits>
#include <string>

// WMMA must be declared in the global namespace.  This manager intentionally
// includes its component .cu files inside namespace cudadem below.
#include <mma.h>

#ifdef CUDADEM_ENABLE_PROFILING
#include <nvtx3/nvToolsExt.h>
#define BF_FMM_NVTX_RANGE(name) nvtxRangePush(name)
#define BF_FMM_NVTX_END()       nvtxRangePop()
#else
#define BF_FMM_NVTX_RANGE(name) ((void)0)
#define BF_FMM_NVTX_END()       ((void)0)
#endif

// MAX_MULTI_GPUS
#include "../device_state_array.h"

// STL includes before namespace cudadem (avoid cudadem::std pollution)
#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>
#include <memory>
#include <new>
#include "domain/peer_transfer.h"
#include "domain/shear_source_routing.h"
#include "domain/multigpu_phase_coordinator.h"

// Thrust headers MUST be included at global scope (before namespace cudadem)
// to prevent thrust symbols from being placed in cudadem::thrust:: namespace
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#ifdef CUDADEM_HAVE_LET
#include "domain/let_communicator.h"
#endif

namespace cudadem {

// (lbvh_fmm headers + host-wrapper forward declarations removed — body_fitted_fmm
//  is now self-contained. The canonical FMM formulas formerly in lbvh_fmm/ are
//  preserved under lib/lbvh_fmm_reference/ as reference-only documentation.
//  BFFMM uses its own _unified kernels for P2M/M2M/L2L/L2P/P2P.)


// Topology-freezing displacement kernels (migrated from the old lbvh_fmm object;
// the only live symbols BFFMM ever pulled from it).

// Include our own header (body_fitted_fmm types)
#include "body_fitted_fmm_manager.h"

// Include our own .cu files (component implementations)
// All these .cu files are #included into this single TU inside namespace cudadem.
#include "g_table_lookup.cuh"
#include "planar_gap1_operator_table.h"
#include "m2l_warpsync_kernel.cu"

// Dense level map for O(1) Morton lookup
#include "dense_level_map.cuh"

// Phase 2: Unified FMM kernels (use UnifiedNode + children_compact)
#include "multipole_compute_unified.cu"
#include "local_expansion_unified.cu"
#include "p2p_unified.cu"

// MAC-generated, target-owned grouped interaction records.
#include "grouped_interaction_arena.cu"

// Sole production Karras topology builder.
#include "karras_octree_build.cu"

// Fixed-depth, device-resident tree construction + stencil traversal.
#include "persistent_tree_traversal.cu"
#include "distributed_fmm.cuh"

// =============================================================================
// Device Kernel: Populate dense level maps from device-side nodes
// =============================================================================
// One thread per node writes its index into dense_map[level][compact_morton].
// Defined here (not in the header) to avoid multiple-definition linker errors
// when included from multiple translation units.
// =============================================================================

__global__ void dense_populate_from_device_nodes_kernel(
    uint32_t** level_dense_map,
    const int* level_dense_capacity,
    const fmm::FMMOctreeNode* nodes,
    int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int lvl = (int)nodes[idx].level;
    if (lvl < 0 || lvl >= fmm::FMM_MAX_DENSE_LEVELS) return;

    uint32_t cm = nodes[idx].compact_morton;
    if (cm < (uint32_t)level_dense_capacity[lvl]) {
        level_dense_map[lvl][cm] = (uint32_t)idx;
    }
}

// Host wrapper: allocate dense maps, fill with sentinel, launch populate kernel
void build_dense_level_maps_device(
    fmm::FMMDeviceData& dd,
    int num_nodes,
    int max_level,
    cudaStream_t stream)
{
    constexpr int D3_MAX_LEVELS = 8;  // D=3: cap per level = 8^L
    int cap[D3_MAX_LEVELS];
    cap[0] = 1;
    for (int l = 1; l < D3_MAX_LEVELS; l++) {
        cap[l] = cap[l-1] * 8;
    }

    int active_levels = (max_level < D3_MAX_LEVELS) ? (max_level + 1) : D3_MAX_LEVELS;

    // Free any prior dense maps before reallocating to prevent memory leak.
    for (int l = 0; l < fmm::FMM_MAX_DENSE_LEVELS; l++) {
        if (dd.level_dense_map[l]) {
            cudaFree(dd.level_dense_map[l]);
            dd.level_dense_map[l] = nullptr;
            dd.level_dense_capacity[l] = 0;
        }
    }

    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&dd.level_dense_map[l], sz));
        CUDA_CHECK(cudaMemsetAsync(dd.level_dense_map[l], 0xFF, sz, stream));
        dd.level_dense_capacity[l] = cap[l];
    }

    int* d_cap;
    CUDA_CHECK(cudaMalloc(&d_cap, D3_MAX_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_cap, dd.level_dense_capacity, D3_MAX_LEVELS * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    uint32_t** d_dense_ptrs;
    CUDA_CHECK(cudaMalloc(&d_dense_ptrs, D3_MAX_LEVELS * sizeof(uint32_t*)));
    CUDA_CHECK(cudaMemcpyAsync(d_dense_ptrs, dd.level_dense_map, D3_MAX_LEVELS * sizeof(uint32_t*),
                                cudaMemcpyHostToDevice, stream));

    const int BLOCK = 256;
    int grid = (num_nodes + BLOCK - 1) / BLOCK;
    dense_populate_from_device_nodes_kernel<<<grid, BLOCK, 0, stream>>>(
        d_dense_ptrs, d_cap, dd.nodes, num_nodes);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_cap));
    CUDA_CHECK(cudaFree(d_dense_ptrs));
}

// =============================================================================
// Device Kernel: Populate dense level maps from UnifiedNode (D-agnostic)
// =============================================================================

__global__ void dense_populate_from_unified_kernel(
    uint32_t** level_dense_map,
    const int* level_dense_capacity,
    const body_fitted_fmm::UnifiedNode* nodes,
    int num_nodes,
    int max_levels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int lvl = (int)nodes[idx].level;
    if (lvl < 0 || lvl >= max_levels) return;

    uint32_t cm = nodes[idx].compact_morton;
    if (cm < (uint32_t)level_dense_capacity[lvl]) {
        level_dense_map[lvl][cm] = (uint32_t)idx;
    }
}

// Host wrapper: allocate dense maps from UnifiedNode with D-specific capacity
template<int D>
void build_dense_level_maps_device_unified(
    fmm::FMMDeviceData& dd,
    int num_nodes,
    int max_level,
    cudaStream_t stream)
{
    constexpr int MAX = DenseLevelTraits<D>::max_levels;
    int cap[MAX];
    cap[0] = 1;
    for (int l = 1; l < MAX; l++) {
        cap[l] = cap[l-1] * (1 << D);
    }

    int active_levels = (max_level < MAX) ? (max_level + 1) : MAX;

    // Free prior dense maps
    for (int l = 0; l < fmm::FMM_MAX_DENSE_LEVELS; l++) {
        if (dd.level_dense_map[l]) {
            cudaFree(dd.level_dense_map[l]);
            dd.level_dense_map[l] = nullptr;
            dd.level_dense_capacity[l] = 0;
        }
    }

    for (int l = 0; l < active_levels; l++) {
        size_t sz = (size_t)cap[l] * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&dd.level_dense_map[l], sz));
        CUDA_CHECK(cudaMemsetAsync(dd.level_dense_map[l], 0xFF, sz, stream));
        dd.level_dense_capacity[l] = cap[l];
    }

    int* d_cap;
    CUDA_CHECK(cudaMalloc(&d_cap, MAX * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_cap, dd.level_dense_capacity, MAX * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    uint32_t** d_dense_ptrs;
    CUDA_CHECK(cudaMalloc(&d_dense_ptrs, MAX * sizeof(uint32_t*)));
    CUDA_CHECK(cudaMemcpyAsync(d_dense_ptrs, dd.level_dense_map, MAX * sizeof(uint32_t*),
                                cudaMemcpyHostToDevice, stream));

    const int BLOCK = 256;
    int grid = (num_nodes + BLOCK - 1) / BLOCK;
    dense_populate_from_unified_kernel<<<grid, BLOCK, 0, stream>>>(
        d_dense_ptrs, d_cap, dd.unified_nodes, num_nodes, MAX);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_cap));
    CUDA_CHECK(cudaFree(d_dense_ptrs));
}

// Explicit template instantiations
template void build_dense_level_maps_device_unified<2>(fmm::FMMDeviceData&, int, int, cudaStream_t);
template void build_dense_level_maps_device_unified<3>(fmm::FMMDeviceData&, int, int, cudaStream_t);


// =============================================================================
// Copied kernels (not available through external linkage)
// =============================================================================
// These kernels are defined in lbvh_fmm_manager.cu itself (not in separate .cu
// files), or are static functions in .cu files included by lbvh_fmm_manager.cu.
// We copy them here with `bf_` prefix to avoid any symbol conflicts.
// =============================================================================

// --- Tree topology conversion kernel (TEST-ONLY: used by test_karras_vs_bh) ---
// Not called by build_tree() — retained for the topology comparison test.
__global__ void bf_convert_bh_topology_to_octree(
    const uint64_t* __restrict__ sorted_morton_keys,
    const int2*     __restrict__ node_range,
    const int*      __restrict__ node_children,
    const int*      __restrict__ node_parent,
    const int*      __restrict__ node_level,
    const int*      __restrict__ node_is_leaf,
    const int*      __restrict__ node_first_child,
    const int*      __restrict__ node_n_children,
    const float*    __restrict__ node_cell_size,
    fmm::FMMOctreeNode*  __restrict__ nodes,
    int*            __restrict__ children_indices,
    double          domain_min_x,
    double          domain_min_y,
    double          domain_min_z,
    double          root_cell_size,
    int             num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    fmm::FMMOctreeNode node;
    node.parent_idx    = node_parent[idx];
    node.first_child   = idx * 8;
    node.particle_begin = node_is_leaf[idx] ? node_range[idx].x : 0;
    node.particle_end   = node_is_leaf[idx] ? (node_range[idx].x + node_range[idx].y) : 0;
    node.level         = (uint8_t)node_level[idx];
    node.n_children    = (uint8_t)node_n_children[idx];
    node.flags         = node_is_leaf[idx] ? 1 : 0;

    int level = node_level[idx];
    float half_size = (float)(root_cell_size / (double)(1ULL << (level + 1)));

    int2 nr = node_range[idx];
    int first = nr.x;

    double cx, cy, cz;
    if (first >= 0 && nr.y > 0) {
        uint64_t morton = sorted_morton_keys[first];
        uint32_t cell_x = 0, cell_y = 0, cell_z = 0;
        for (int l = 0; l < level; l++) {
            int shift = 60 - 3 * l;
            if (shift >= 0) {
                cell_x = (cell_x << 1) | (uint32_t)((morton >> shift) & 1);
                cell_y = (cell_y << 1) | (uint32_t)((morton >> (shift + 1)) & 1);
                cell_z = (cell_z << 1) | (uint32_t)((morton >> (shift + 2)) & 1);
            }
        }
        double cell_size_d = root_cell_size / (double)(1ULL << level);
        cx = domain_min_x + ((double)cell_x + 0.5) * cell_size_d;
        cy = domain_min_y + ((double)cell_y + 0.5) * cell_size_d;
        cz = domain_min_z + ((double)cell_z + 0.5) * cell_size_d;

        if (level <= 7) {
            uint64_t ex = (uint64_t)cell_x;
            uint64_t ey = (uint64_t)cell_y;
            uint64_t ez = (uint64_t)cell_z;
            ex = (ex | (ex << 32)) & 0x1F00000000FFFFull;
            ex = (ex | (ex << 16)) & 0x1F0000FF0000FFull;
            ex = (ex | (ex <<  8)) & 0x100F00F00F00F00Full;
            ex = (ex | (ex <<  4)) & 0x10C30C30C30C30C3ull;
            ex = (ex | (ex <<  2)) & 0x1249249249249249ull;
            ey = (ey | (ey << 32)) & 0x1F00000000FFFFull;
            ey = (ey | (ey << 16)) & 0x1F0000FF0000FFull;
            ey = (ey | (ey <<  8)) & 0x100F00F00F00F00Full;
            ey = (ey | (ey <<  4)) & 0x10C30C30C30C30C3ull;
            ey = (ey | (ey <<  2)) & 0x1249249249249249ull;
            ez = (ez | (ez << 32)) & 0x1F00000000FFFFull;
            ez = (ez | (ez << 16)) & 0x1F0000FF0000FFull;
            ez = (ez | (ez <<  8)) & 0x100F00F00F00F00Full;
            ez = (ez | (ez <<  4)) & 0x10C30C30C30C30C3ull;
            ez = (ez | (ez <<  2)) & 0x1249249249249249ull;
            node.compact_morton = (uint32_t)(ex | (ey << 1) | (ez << 2));
        } else {
            node.compact_morton = 0;
        }
    } else {
        cx = domain_min_x + root_cell_size * 0.5;
        cy = domain_min_y + root_cell_size * 0.5;
        cz = domain_min_z + root_cell_size * 0.5;
        node.compact_morton = 0;
    }

    node.aabb_min = make_float3((float)(cx - half_size), (float)(cy - half_size), (float)(cz - half_size));
    node.aabb_max = make_float3((float)(cx + half_size), (float)(cy + half_size), (float)(cz + half_size));
    nodes[idx] = node;

    for (int c = 0; c < 8; c++) {
        children_indices[idx * 8 + c] = node_children[idx * 8 + c];
    }
}

// --- Particle-to-leaf mapping kernel (from lbvh_fmm_manager.cu) ---
__global__ void bf_build_particle_to_leaf_kernel(
    const body_fitted_fmm::UnifiedNode* __restrict__ nodes,
    int num_nodes,
    int* __restrict__ particle_to_leaf)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;

    const body_fitted_fmm::UnifiedNode& node = nodes[node_idx];
    if (node.childBase == body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE) {
        for (int i = node.begin; i < node.begin + node.count; i++) {
            particle_to_leaf[i] = node_idx;
        }
    }
}

// --- Global projection kernels (copied from global_projection.cu, was static) ---
#define BF_GP_REDUCE_NUM_SCALARS 16
#define BF_GP_TOTAL_NUM_SCALARS 26
#define BF_GP_SOLUTION_OFFSET 16

__global__ void bf_global_projection_reduce(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const double* __restrict__ force_x,
    const double* __restrict__ force_y, const double* __restrict__ force_z,
    const float* __restrict__ mass, double* d_scalars, int N,
    bool input_is_acceleration, double origin_x=0,double origin_y=0,double origin_z=0)
{
    __shared__ double sdata[256 * BF_GP_REDUCE_NUM_SCALARS];
    int tid = threadIdx.x, bsz = blockDim.x;
    int gid = blockIdx.x * bsz + tid, grid_sz = gridDim.x * bsz;
    for (int k = 0; k < BF_GP_REDUCE_NUM_SCALARS; k++) sdata[tid * BF_GP_REDUCE_NUM_SCALARS + k] = 0.0;
    for (int i = gid; i < N; i += grid_sz) {
        double px = pos_x[i]-origin_x, py = pos_y[i]-origin_y, pz = pos_z[i]-origin_z;
        double fx = force_x[i], fy = force_y[i], fz = force_z[i], m = (double)mass[i];
        if (input_is_acceleration) { fx *= m; fy *= m; fz *= m; }
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+0] += fx;  sdata[tid*BF_GP_REDUCE_NUM_SCALARS+1] += fy;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+2] += fz;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+3] += py*fz-pz*fy; sdata[tid*BF_GP_REDUCE_NUM_SCALARS+4] += pz*fx-px*fz;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+5] += px*fy-py*fx;  sdata[tid*BF_GP_REDUCE_NUM_SCALARS+6] += m;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+7] += m*px; sdata[tid*BF_GP_REDUCE_NUM_SCALARS+8] += m*py;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+9] += m*pz;
        double r2 = px*px+py*py+pz*pz;
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+10] += m*(r2-px*px); sdata[tid*BF_GP_REDUCE_NUM_SCALARS+11] += m*(r2-py*py);
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+12] += m*(r2-pz*pz); sdata[tid*BF_GP_REDUCE_NUM_SCALARS+13] += m*(-px*py);
        sdata[tid*BF_GP_REDUCE_NUM_SCALARS+14] += m*(-px*pz);  sdata[tid*BF_GP_REDUCE_NUM_SCALARS+15] += m*(-py*pz);
    }
    __syncthreads();
    for (int stride = bsz/2; stride > 0; stride >>= 1) {
        if (tid < stride) for (int k = 0; k < BF_GP_REDUCE_NUM_SCALARS; k++)
            sdata[tid*BF_GP_REDUCE_NUM_SCALARS+k] += sdata[(tid+stride)*BF_GP_REDUCE_NUM_SCALARS+k];
        __syncthreads();
    }
    if (tid == 0) for (int k = 0; k < BF_GP_REDUCE_NUM_SCALARS; k++) atomicAdd(&d_scalars[k], sdata[k]);
}

__global__ void bf_global_projection_apply(
    double* __restrict__ force_x, double* __restrict__ force_y, double* __restrict__ force_z,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    const double* __restrict__ d_scalars, int N,
    bool output_is_acceleration, double origin_x=0,double origin_y=0,double origin_z=0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double Fex = d_scalars[BF_GP_SOLUTION_OFFSET + 0];
    const double Fey = d_scalars[BF_GP_SOLUTION_OFFSET + 1];
    const double Fez = d_scalars[BF_GP_SOLUTION_OFFSET + 2];
    const double ax = d_scalars[BF_GP_SOLUTION_OFFSET + 3];
    const double ay = d_scalars[BF_GP_SOLUTION_OFFSET + 4];
    const double az = d_scalars[BF_GP_SOLUTION_OFFSET + 5];
    const double cmx = d_scalars[BF_GP_SOLUTION_OFFSET + 6];
    const double cmy = d_scalars[BF_GP_SOLUTION_OFFSET + 7];
    const double cmz = d_scalars[BF_GP_SOLUTION_OFFSET + 8];
    const double inv_M = d_scalars[BF_GP_SOLUTION_OFFSET + 9];
    double m = output_is_acceleration ? 1.0 : (double)mass[i];
    double rx = (pos_x[i]-origin_x)-cmx, ry = (pos_y[i]-origin_y)-cmy, rz = (pos_z[i]-origin_z)-cmz;
    double dFx = m*inv_M*Fex + m*(ay*rz-az*ry);
    double dFy = m*inv_M*Fey + m*(az*rx-ax*rz);
    double dFz = m*inv_M*Fez + m*(ax*ry-ay*rx);
    force_x[i] -= dFx; force_y[i] -= dFy; force_z[i] -= dFz;
}

__global__ void bf_global_projection_solve(double* d_scalars)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const double Fx=d_scalars[0], Fy=d_scalars[1], Fz=d_scalars[2];
    const double tox=d_scalars[3], toy=d_scalars[4], toz=d_scalars[5];
    const double M=d_scalars[6], MRx=d_scalars[7], MRy=d_scalars[8], MRz=d_scalars[9];
    double* solution = d_scalars + BF_GP_SOLUTION_OFFSET;
    if (M < 1e-30) {
        for (int k = 0; k < 10; ++k) solution[k] = 0.0;
        return;
    }

    const double inv_M=1.0/M;
    const double cmx=MRx*inv_M, cmy=MRy*inv_M, cmz=MRz*inv_M;
    const double tx=tox-(cmy*Fz-cmz*Fy);
    const double ty=toy-(cmz*Fx-cmx*Fz);
    const double tz=toz-(cmx*Fy-cmy*Fx);
    const double cm2=cmx*cmx+cmy*cmy+cmz*cmz;
    const double a=d_scalars[10]-M*(cm2-cmx*cmx);
    const double b=d_scalars[13]+M*cmx*cmy;
    const double c=d_scalars[14]+M*cmx*cmz;
    const double d=b;
    const double e=d_scalars[11]-M*(cm2-cmy*cmy);
    const double f=d_scalars[15]+M*cmy*cmz;
    const double g=c;
    const double h=f;
    const double i=d_scalars[12]-M*(cm2-cmz*cmz);
    const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);

    double ax=0.0, ay=0.0, az=0.0;
    if (fabs(det) >= 1e-40) {
        const double inv=1.0/det;
        const double ai0=(e*i-f*h)*inv, ai1=(c*h-b*i)*inv, ai2=(b*f-c*e)*inv;
        const double ai3=(f*g-d*i)*inv, ai4=(a*i-c*g)*inv, ai5=(c*d-a*f)*inv;
        const double ai6=(d*h-e*g)*inv, ai7=(b*g-a*h)*inv, ai8=(a*e-b*d)*inv;
        ax=ai0*tx+ai1*ty+ai2*tz;
        ay=ai3*tx+ai4*ty+ai5*tz;
        az=ai6*tx+ai7*ty+ai8*tz;
    }

    solution[0]=Fx; solution[1]=Fy; solution[2]=Fz;
    solution[3]=ax; solution[4]=ay; solution[5]=az;
    solution[6]=cmx; solution[7]=cmy; solution[8]=cmz;
    solution[9]=inv_M;
}

static void bf_run_global_projection(
    double* pos_x, double* pos_y, double* pos_z,
    double* force_x, double* force_y, double* force_z,
    float* mass, int N, double* d_scalars, cudaStream_t stream,
    bool values_are_acceleration)
{
    if (N <= 0) return;
    const int BLOCK = 256;
    CUDA_CHECK(cudaMemsetAsync(d_scalars, 0, BF_GP_REDUCE_NUM_SCALARS*sizeof(double), stream));
    int grid = (N + BLOCK*8 - 1) / (BLOCK*8); if (grid < 1) grid = 1; if (grid > 65535) grid = 65535;
    bf_global_projection_reduce<<<grid, BLOCK, 0, stream>>>(
        pos_x,pos_y,pos_z,force_x,force_y,force_z,mass,d_scalars,N,
        values_are_acceleration);
    bf_global_projection_solve<<<1, 1, 0, stream>>>(d_scalars);
    int ag = (N+BLOCK-1)/BLOCK;
    bf_global_projection_apply<<<ag, BLOCK, 0, stream>>>(
        force_x,force_y,force_z,pos_x,pos_y,pos_z,mass,d_scalars,N,
        values_are_acceleration);
}

// The correction acts on gravity alone. It must be solved from global
// additive moments, never independently on each owner's irregular domain.
static void bf_distributed_projection(int gpu,SimulationState& state,bool outer,cudaStream_t stream) {
    auto& common=body_fitted_fmm::distributed_fmm_[gpu];const int n=state.particles.num_local;
    const auto phase=::cudadem::PeerPhase::FmmProjection;
    if(common.ranks>1)body_fitted_fmm::fmm_collective_check(common.coordinator->prepare_producer(phase,gpu,common.epoch,stream),common);
    common.projection_scalars.reserve(BF_GP_TOTAL_NUM_SCALARS);
    double* scalar=common.projection_scalars.data;
    CUDA_CHECK(cudaMemsetAsync(scalar,0,BF_GP_TOTAL_NUM_SCALARS*sizeof(double),stream));
    const double ox=common.grid.x+0.5*common.grid.size,oy=common.grid.y+0.5*common.grid.size,oz=common.grid.z+0.5*common.grid.size;
    auto& p=state.particles;double* fx=outer?p.gravity_force_x:p.force_x;
    double* fy=outer?p.gravity_force_y:p.force_y;double* fz=outer?p.gravity_force_z:p.force_z;
    if(n)bf_global_projection_reduce<<<std::min(65535,(n+2047)/2048),256,0,stream>>>(p.pos_x,p.pos_y,p.pos_z,fx,fy,fz,p.mass,scalar,n,outer,ox,oy,oz);
    CUDA_CHECK(cudaMemcpyAsync(common.projection_partial,scalar,16*sizeof(double),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if(common.ranks>1){
        body_fitted_fmm::fmm_collective_check(common.coordinator->publish(phase,gpu,common.epoch,stream),common);
        for(int peer=0;peer<common.ranks;++peer)if(peer!=gpu)
            body_fitted_fmm::fmm_collective_check(common.coordinator->wait_for_published(phase,peer,gpu,common.epoch,stream),common);
    }
    double sums[16]{};
    for(int peer=0;peer<common.ranks;++peer)for(int i=0;i<16;++i)sums[i]+=body_fitted_fmm::distributed_fmm_[peer].projection_partial[i];
    CUDA_CHECK(cudaMemcpyAsync(scalar,sums,sizeof(sums),cudaMemcpyHostToDevice,stream));
    bf_global_projection_solve<<<1,1,0,stream>>>(scalar);
    if(n)bf_global_projection_apply<<<(n+255)/256,256,0,stream>>>(fx,fy,fz,p.pos_x,p.pos_y,p.pos_z,p.mass,scalar,n,outer,ox,oy,oz);
    if(common.ranks>1)for(int peer=0;peer<common.ranks;++peer)if(peer!=gpu)
        body_fitted_fmm::fmm_collective_check(common.coordinator->acknowledge_consumed(phase,peer,gpu,common.epoch,stream),common);
    // Protect the host sums backing the asynchronous upload.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
__global__ void bf_add_distributed_gravity(double* fx,double* fy,double* fz,
    const double* gx,const double* gy,const double* gz,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){fx[i]+=gx[i];fy[i]+=gy[i];fz[i]+=gz[i];}
}

// Census keys use the same fixed root, image transformation and quantization
// as the production Karras builder. No image-coordinate arrays are needed.
template<int D>
__global__ void shear_census_keys_kernel(const double* x, const double* y, const double* z,
    int targets, int sources, ShearGravityImages images, double4 root, int depth,
    uint64_t* keys, unsigned long long* histogram)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sources) return;
    int32_t nx=0, ny=0;
    shear_gravity_image_indices(i / targets, images.layers_x, 0, nx, ny);
    const int row = i % targets;
    const double px=x[row]+double(nx)*images.length_x;
    const double py=shear_gravity_canonical_y(y[row],nx,images);
    const double inv=1.0/root.w;
    const double fx=(px-root.x)*inv, fy=(py-root.y)*inv, fz=(z[row]-root.z)*inv;
    if (!isfinite(fx)||!isfinite(fy)||!isfinite(fz)||fx<0||fx>=1||fy<0||fy>=1||(D==3&&(fz<0||fz>=1))) {
        atomicExch(histogram+21,1ull); keys[i]=0; return;
    }
    constexpr int bits=body_fitted_fmm::TreeTraits<D>::MORTON_BITS;
    constexpr int quant=bits/D>31?31:bits/D;
    constexpr double scale=double(1ull<<quant);
    uint64_t key=body_fitted_fmm::morton_encode<D>(uint32_t(fx*scale),uint32_t(fy*scale),
        uint32_t(fmin(fmax(fz,0.0),0.9999999999999999)*scale));
    const int shift=bits-D*depth;
    if(shift>0)key&=~((1ull<<shift)-1ull);
    keys[i]=key;
}

template<int D>
__global__ void shear_census_prefix_kernel(const uint64_t* keys, int n, unsigned long long* histogram)
{
    __shared__ unsigned int counts[21];
    if(threadIdx.x<21)counts[threadIdx.x]=0;
    __syncthreads();
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>0&&i<n){
        const uint64_t different=keys[i]^keys[i-1];
        if(different){
            constexpr int padding=64-body_fitted_fmm::TreeTraits<D>::MORTON_BITS;
            const int first=(__clzll(different)-padding)/D+1;
            if(first<=20)atomicAdd(counts+first,1u);
        }
    }
    __syncthreads();
    if(threadIdx.x<21&&counts[threadIdx.x])atomicAdd(histogram+threadIdx.x,
        static_cast<unsigned long long>(counts[threadIdx.x]));
}

// A scheduler graph may briefly spin while a submitted command waits behind
// earlier consumer-stream work. CUDA lazy function loading may synchronize the
// context when a kernel is first used, so resolve every post-startup kernel
// before initialize_persistent_tree_runtime() makes scheduler launches
// available.
static void preload_post_scheduler_kernels()
{
    auto preload = [](const void* kernel, const char* name) {
        cudaFuncAttributes attributes{};
        const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to preload CUDA kernel ") + name + ": " +
                cudaGetErrorString(status));
        }
    };

#define PRELOAD_LET(kernel) preload((const void*)body_fitted_fmm::kernel, #kernel)
    preload((const void*)body_fitted_fmm::fmm_image_parts_kernel<2,false>, "image parts2 count");
    preload((const void*)body_fitted_fmm::fmm_image_parts_kernel<3,false>, "image parts3 count");
    preload((const void*)body_fitted_fmm::fmm_image_parts_kernel<2,true>, "image parts2 emit");
    preload((const void*)body_fitted_fmm::fmm_image_parts_kernel<3,true>, "image parts3 emit");
    PRELOAD_LET(fmm_refresh_image_replies_kernel<2>);
    PRELOAD_LET(fmm_refresh_image_replies_kernel<3>);
    PRELOAD_LET(fmm_pack_image_particles_kernel);
    PRELOAD_LET(fmm_check_image_parts_kernel);
    PRELOAD_LET(fmm_halo_local_image_policy_kernel<2>);
    PRELOAD_LET(fmm_halo_local_image_policy_kernel<3>);
    PRELOAD_LET(fmm_halo_count_routes_kernel<2>);
    PRELOAD_LET(fmm_halo_count_routes_kernel<3>);
    PRELOAD_LET(fmm_halo_publish_counts_kernel);
    PRELOAD_LET(fmm_halo_moment_bind_kernel<2>);
    PRELOAD_LET(fmm_halo_moment_bind_kernel<3>);
    PRELOAD_LET(fmm_halo_moment_routes_kernel<2>);
    PRELOAD_LET(fmm_halo_moment_routes_kernel<3>);
    PRELOAD_LET(fmm_halo_moment_reduce_kernel);
    PRELOAD_LET(fmm_p2p_remote_leaves_kernel<false>);
    PRELOAD_LET(fmm_p2p_remote_leaves_kernel<true>);
    preload((const void*)body_fitted_fmm::fmm_p2p_remote_leaves_kernel<false,double>, "SPH remote force");
    preload((const void*)body_fitted_fmm::fmm_p2p_remote_leaves_kernel<true,double>, "SPH remote acceleration");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<false,double>, "SPH local force");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<true,double>, "SPH local acceleration");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<false,double,true>, "SPH leaf-only local force");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<true,double,true>, "SPH leaf-only local acceleration");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<false,body_fitted_fmm::p2p_compute_t,true>, "DEM leaf-only local force");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<true,body_fitted_fmm::p2p_compute_t,true>, "DEM leaf-only local acceleration");
    PRELOAD_LET(fmm_verify_leaf_only_kernel);
    PRELOAD_LET(fmm_halo_decode_kernel<2>);
    PRELOAD_LET(fmm_halo_decode_kernel<3>);
    PRELOAD_LET(fmm_halo_nodes_kernel<2>);
    PRELOAD_LET(fmm_halo_nodes_kernel<3>);
    PRELOAD_LET(fmm_halo_inverse_routes_kernel<2>);
    PRELOAD_LET(fmm_halo_inverse_routes_kernel<3>);
    preload((const void*)body_fitted_fmm::fmm_halo_inverse_routes_kernel<2,true>, "halo mark compact targets 2");
    preload((const void*)body_fitted_fmm::fmm_halo_inverse_routes_kernel<3,true>, "halo mark compact targets 3");
    PRELOAD_LET(fmm_plan_identity_kernel<2>);
    PRELOAD_LET(fmm_plan_identity_kernel<3>);
    PRELOAD_LET(fmm_halo_select_kernel);
    PRELOAD_LET(fmm_halo_gather_kernel);
    PRELOAD_LET(fmm_halo_initialize_routes_kernel);
    PRELOAD_LET(fmm_halo_particle_ranges_kernel);
    PRELOAD_LET(fmm_halo_particle_requests_kernel);
    PRELOAD_LET(fmm_narrow_checked_offsets_kernel);
    PRELOAD_LET(fmm_m2l_remote_targets_kernel);
    PRELOAD_LET(fmm_remote_list_counts_kernel<2>);
    PRELOAD_LET(fmm_remote_list_counts_kernel<3>);
    PRELOAD_LET(fmm_remote_far_keys_kernel<2>);
    PRELOAD_LET(fmm_remote_far_keys_kernel<3>);
    PRELOAD_LET(fmm_remote_far_targets_kernel);
    PRELOAD_LET(fmm_remote_far_sources_kernel<2>);
    PRELOAD_LET(fmm_remote_far_sources_kernel<3>);
    PRELOAD_LET(fmm_far_cache_source_remap_kernel<2>);
    PRELOAD_LET(fmm_far_cache_source_remap_kernel<3>);
    PRELOAD_LET(fmm_remote_far_cached_sources_kernel<2>);
    PRELOAD_LET(fmm_remote_far_cached_sources_kernel<3>);
    PRELOAD_LET(fmm_far_cache_target_keys_kernel<2>);
    PRELOAD_LET(fmm_far_cache_target_keys_kernel<3>);
    PRELOAD_LET(fmm_far_cache_snapshot_targets_kernel);
    PRELOAD_LET(fmm_far_cache_snapshot_children_kernel);
    PRELOAD_LET(fmm_remote_direct_sources_kernel<2>);
    PRELOAD_LET(fmm_remote_direct_sources_kernel<3>);
    PRELOAD_LET(fmm_halo_refresh_nodes_kernel);
    PRELOAD_LET(fmm_halo_identity_order_kernel);
    preload((const void*)body_fitted_fmm::fmm_halo_candidate_kernel<2,false>, "halo candidates 2 false");
    preload((const void*)body_fitted_fmm::fmm_halo_candidate_kernel<2,true>, "halo candidates 2 true");
    preload((const void*)body_fitted_fmm::fmm_halo_candidate_kernel<3,false>, "halo candidates 3 false");
    preload((const void*)body_fitted_fmm::fmm_halo_candidate_kernel<3,true>, "halo candidates 3 true");
#undef PRELOAD_LET

    preload((const void*)shear_census_keys_kernel<2>, "shear_census_keys_kernel<2>");
    preload((const void*)shear_census_keys_kernel<3>, "shear_census_keys_kernel<3>");
    preload((const void*)shear_census_prefix_kernel<2>, "shear_census_prefix_kernel<2>");
    preload((const void*)shear_census_prefix_kernel<3>, "shear_census_prefix_kernel<3>");
    preload((const void*)body_fitted_fmm::fmm_zero_active_local_expansions_kernel,
            "fmm_zero_active_local_expansions_kernel");
    preload((const void*)body_fitted_fmm::fmm_clear_R_max_unified,
            "fmm_clear_R_max_unified");
    preload((const void*)body_fitted_fmm::fmm_p2m_leaf_multipoles_unified,
            "fmm_p2m_leaf_multipoles_unified");
    preload((const void*)body_fitted_fmm::fmm_mass_weighted_com_unified,
            "fmm_mass_weighted_com_unified");
    preload((const void*)body_fitted_fmm::fmm_m2m_translate_parent_side,
            "fmm_m2m_translate_parent_side");
    preload((const void*)body_fitted_fmm::fmm_pack_multipoles_unified,
            "fmm_pack_multipoles_unified");
    preload((const void*)body_fitted_fmm::m2l_grouped_target_kernel,
            "m2l_grouped_target_kernel");
    preload((const void*)body_fitted_fmm::fmm_l2l_kernel_unified,
            "fmm_l2l_kernel_unified");
    preload((const void*)body_fitted_fmm::fmm_l2p_kernel_unified,
            "fmm_l2p_kernel_unified");
    preload((const void*)body_fitted_fmm::evaluate_l2p_outer_kernel_unified,
            "evaluate_l2p_outer_kernel_unified");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<false>,
            "fmm_p2p_grouped_leaf_unified<false>");
    preload((const void*)body_fitted_fmm::fmm_p2p_grouped_leaf_unified<true>,
            "fmm_p2p_grouped_leaf_unified<true>");
    preload((const void*)bf_global_projection_reduce,
            "bf_global_projection_reduce");
    preload((const void*)bf_global_projection_solve,
            "bf_global_projection_solve");
    preload((const void*)bf_global_projection_apply,
            "bf_global_projection_apply");
}

// =============================================================================
// UnifiedFMMDispatcher implementation
// =============================================================================

namespace body_fitted_fmm {

static void release_aligned_clump_pool(int gpu);

// Static member definitions — per-GPU arrays
static ::cudadem::DeviceStateArray<fmm::FMMHostData> host_data_;
static ::cudadem::DeviceStateArray<fmm::FMMDeviceData> device_data_;
static ::cudadem::DeviceStateArray<BodyFittedFMMDeviceData> bf_device_data_;
static ::cudadem::DeviceStateArray<bool> initialized_;
static ::cudadem::DeviceStateArray<int> max_particles_;
static ::cudadem::DeviceStateArray<bool> verbose_;
static ::cudadem::DeviceStateArray<int> initialized_tree_type_;

// M2L pair arrays (aggregated across all levels for M2L dispatch)
// Max nodes capacity per GPU
static ::cudadem::DeviceStateArray<int> max_nodes_capacity_;

// Task arena (4-segment typed, allocated after tree build)
static ::cudadem::DeviceStateArray<GroupedInteractionArena> grouped_arena_;

// Resident device scheduler and its pre-uploaded D=2/D=3 graph executables.
static ::cudadem::DeviceStateArray<PersistentTreeRuntime> persistent_tree_runtime_;
static ::cudadem::DeviceStateArray<AdaptiveFMMDepthCheckpoint> pending_adaptive_depth_checkpoint_;
static ::cudadem::DeviceStateArray<bool> has_pending_adaptive_depth_checkpoint_;

// Keep the device ring small and nonblocking while decoupling its lifetime
// from physical-time output intervals.  Every N completed autonomous depth
// evaluations, the dispatcher snapshots the ring into pinned host memory at
// the synchronization the FMM path already performs.  Rows remain in the
// host backlog until the normal output/checkpoint/final writer consumes them.
static constexpr int ADAPTIVE_DEPTH_DIAGNOSTIC_DRAIN_EVALUATIONS = 16;
struct AdaptiveDepthDiagnosticDrainState {
    AdaptiveDepthDiagnosticRing* snapshot = nullptr;
    cudaEvent_t snapshot_ready = nullptr;
    bool snapshot_pending = false;
    std::vector<AdaptiveFMMDepthDiagnostic> backlog;
    int evaluations_since_drain = 0;
};
static ::cudadem::DeviceStateArray<AdaptiveDepthDiagnosticDrainState> adaptive_depth_diagnostic_drain_;

struct AdaptiveStartupProbeResult {
    bool valid = false;
    int initial_depth = 0;
    int effective_max_depth = 0;
    int node_count[PERSISTENT_TREE_MAX_DEPTH + 1] = {};
    int leaf_count[PERSISTENT_TREE_MAX_DEPTH + 1] = {};
    int node_capacity = 0;
};
static ::cudadem::DeviceStateArray<AdaptiveStartupProbeResult> adaptive_startup_probe_;
static ::cudadem::DeviceStateArray<int> adaptive_capacity_override_;

template<int D>
__global__ void adaptive_probe_morton_kernel(
    const double* pos_x, const double* pos_y, const double* pos_z,
    int count, double min_x, double min_y, double min_z, double root_size,
    int max_depth, uint64_t* keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double inv = 1.0 / root_size;
    const double fx = fmin(fmax((pos_x[i] - min_x) * inv, 0.0),
                            0.9999999999999999);
    const double fy = fmin(fmax((pos_y[i] - min_y) * inv, 0.0),
                            0.9999999999999999);
    const double fz = fmin(fmax((pos_z[i] - min_z) * inv, 0.0),
                            0.9999999999999999);
    constexpr int bits_per_axis = TreeTraits<D>::MORTON_BITS / D;
    constexpr int quant_bits = bits_per_axis > 31 ? 31 : bits_per_axis;
    constexpr double scale = static_cast<double>(1ull << quant_bits);
    uint64_t code = morton_encode<D>(
        static_cast<uint32_t>(fx * scale),
        static_cast<uint32_t>(fy * scale),
        static_cast<uint32_t>(fz * scale));
    const int shift = TreeTraits<D>::MORTON_BITS - D * max_depth;
    if (shift > 0) code &= ~((1ull << shift) - 1ull);
    keys[i] = code;
}

__global__ void adaptive_probe_prefix_flags_kernel(
    const uint64_t* sorted_keys, int count, int shift, int* flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint64_t prefix = shift > 0 ? sorted_keys[i] >> shift : sorted_keys[i];
    const uint64_t previous = i > 0
        ? (shift > 0 ? sorted_keys[i - 1] >> shift : sorted_keys[i - 1])
        : UINT64_MAX;
    flags[i] = (i == 0 || prefix != previous) ? 1 : 0;
}

static uint64_t adaptive_next_pow2(uint64_t value)
{
    uint64_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

struct AdaptiveMemoryEstimate {
    uint64_t particles = 0;
    uint64_t nodes = 0;
    uint64_t lookup = 0;
    uint64_t cub = 0;
    uint64_t fixed = 0;
    uint64_t total() const { return particles + nodes + lookup + cub + fixed; }
};

static AdaptiveMemoryEstimate estimate_adaptive_fmm_memory(
    int particle_capacity, int node_capacity, int depth, int tree_dimension)
{
    AdaptiveMemoryEstimate estimate;
    const uint64_t p = static_cast<uint64_t>(particle_capacity);
    const uint64_t n = static_cast<uint64_t>(node_capacity);
    const uint64_t legacy_nodes = static_cast<uint64_t>(
        std::max(2 * particle_capacity, 65536)) * 12ull / 10ull;
    estimate.particles = p * (12ull + 4ull + 12ull + 4ull +
        2ull * sizeof(uint64_t) + 3ull * sizeof(int) +
        6ull * sizeof(double) + 2ull * sizeof(KarrasBinaryNode));
    estimate.particles += legacy_nodes *
        (sizeof(int2) + 12ull * sizeof(int) + sizeof(float));
    estimate.nodes = n * (
        3ull * sizeof(far_real_t) + sizeof(fmm::FMMOctreeNode) +
        sizeof(int) + sizeof(fmm::MultipoleExpansion) +
        fmm::FMM_ACTIVE_COEFFS * sizeof(double) +
        sizeof(fmm::LocalExpansion) + 16ull * sizeof(int) +
        sizeof(UnifiedNode) + sizeof(GroupedTargetInteractions));
    estimate.nodes += static_cast<uint64_t>(fmm::FMM_MAX_LEVELS) *
        ((n + 255ull) / 256ull) * 2ull * sizeof(int);

    uint64_t dense_entries = 0;
    for (int level = 0; level < fmm::FMM_MAX_DENSE_LEVELS; ++level) {
        uint64_t entries = 0;
        if (level <= depth) {
            if (tree_dimension == 3 && level <= 7)
                entries = 1ull << (3 * level);
            if (tree_dimension == 2)
                entries = std::max<uint64_t>(
                    entries, static_cast<uint64_t>(1) << (2 * level));
        }
        dense_entries += entries;
    }
    estimate.lookup = dense_entries * 2ull * sizeof(uint32_t);
    for (int level = 0; level <= depth; ++level) {
        const bool sparse =
            (tree_dimension == 2 && level > 10) ||
            (tree_dimension == 3 && level > 7);
        if (!sparse) continue;
        uint64_t occupied = p;
        const int dimension = tree_dimension;
        if (dimension * level < 63)
            occupied = std::min<uint64_t>(occupied,
                                           1ull << (dimension * level));
        estimate.lookup += adaptive_next_pow2(
            std::max<uint64_t>(2, 2 * occupied)) *
            (sizeof(uint64_t) + sizeof(uint32_t));
    }
    estimate.cub = 6ull * p * (sizeof(uint64_t) + sizeof(int));
    estimate.fixed = 64ull * 1024ull * 1024ull;
    return estimate;
}


// Adaptive RCB consumes these as a borrowed, device-resident view.  A view is
// deliberately published only once both the unified tree and grouped records
// have been rebuilt on this GPU.
static ::cudadem::DeviceStateArray<bool> workload_view_valid_;
static ::cudadem::DeviceStateArray<uint64_t> workload_generation_;
// Set only around compute_multigpu().  Ghosts remain in ParticleSoA for
// contact, but gravity's local tree and target work are strictly [0,num_local).
static ::cudadem::DeviceStateArray<bool> multigpu_owned_only_;
// Staged before lazy initialization so a coordinator can enable external
// control before the first multi-GPU gravity epoch.
static ::cudadem::DeviceStateArray<AdaptiveFMMDepthControlMode> adaptive_depth_control_mode_;
static ::cudadem::DeviceStateArray<ExternalDistributedFMMDepthCommand> staged_external_depth_command_;
static ::cudadem::DeviceStateArray<bool> has_staged_external_depth_command_;

static void ensure_adaptive_depth_diagnostic_staging(int gpu_id)
{
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    if (drain.snapshot) return;
    CUDA_CHECK(cudaHostAlloc(
        &drain.snapshot, sizeof(*drain.snapshot), cudaHostAllocPortable));
    *drain.snapshot = AdaptiveDepthDiagnosticRing{};
    CUDA_CHECK(cudaEventCreateWithFlags(
        &drain.snapshot_ready, cudaEventDisableTiming));
}

static void append_adaptive_depth_diagnostic_snapshot(int gpu_id)
{
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    if (!drain.snapshot) return;
    std::array<AdaptiveFMMDepthDiagnostic,
               ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE> records{};
    uint64_t consumed = drain.snapshot->consumer_sequence;
    const int count = extract_adaptive_depth_diagnostic_snapshot(
        *drain.snapshot, records.data(), static_cast<int>(records.size()),
        consumed);
    drain.backlog.insert(
        drain.backlog.end(), records.data(), records.data() + count);
}

static bool harvest_adaptive_depth_diagnostic_snapshot(
    int gpu_id, bool wait)
{
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    if (!drain.snapshot_pending) return true;
    const cudaError_t status = wait
        ? cudaEventSynchronize(drain.snapshot_ready)
        : cudaEventQuery(drain.snapshot_ready);
    if (!wait && status == cudaErrorNotReady) return false;
    CUDA_CHECK(status);
    append_adaptive_depth_diagnostic_snapshot(gpu_id);
    drain.snapshot_pending = false;
    return true;
}

static bool enqueue_adaptive_depth_diagnostic_snapshot(
    int gpu_id, cudaStream_t stream)
{
    ensure_adaptive_depth_diagnostic_staging(gpu_id);
    auto& runtime = persistent_tree_runtime_[gpu_id];
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    // Never overwrite the single pinned snapshot until its rows have been
    // transferred to the host backlog.  Advancing the device consumer and
    // then reusing this buffer used to create a fatal CSV sequence gap.
    if (!harvest_adaptive_depth_diagnostic_snapshot(gpu_id, false))
        return false;
    auto* device_ring = &runtime.device_context->adaptive_diagnostics;
    CUDA_CHECK(cudaMemcpyAsync(
        drain.snapshot, device_ring, sizeof(*drain.snapshot),
        cudaMemcpyDeviceToHost, stream));
    // The cursor update follows the snapshot in the same stream.  It therefore
    // cannot expose a record as consumed before that record is preserved.
    CUDA_CHECK(cudaMemcpyAsync(
        &device_ring->consumer_sequence, &device_ring->next_sequence,
        sizeof(device_ring->consumer_sequence), cudaMemcpyDeviceToDevice,
        stream));
    CUDA_CHECK(cudaEventRecord(drain.snapshot_ready, stream));
    drain.snapshot_pending = true;
    return true;
}

static bool enqueue_periodic_adaptive_depth_diagnostic_drain(
    int gpu_id, cudaStream_t stream)
{
    auto& runtime = persistent_tree_runtime_[gpu_id];
    if (!runtime.initialized || !runtime.adaptive_options.enabled ||
        adaptive_depth_control_mode_[gpu_id] !=
            AdaptiveFMMDepthControlMode::Autonomous)
        return false;
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    ++drain.evaluations_since_drain;
    if (drain.evaluations_since_drain <
        ADAPTIVE_DEPTH_DIAGNOSTIC_DRAIN_EVALUATIONS)
        return false;
    if (!enqueue_adaptive_depth_diagnostic_snapshot(gpu_id, stream))
        return false;
    drain.evaluations_since_drain = 0;
    return true;
}

static void drain_live_adaptive_depth_diagnostics_to_backlog(
    int gpu_id, cudaStream_t stream)
{
    auto& runtime = persistent_tree_runtime_[gpu_id];
    if (!runtime.initialized || !runtime.adaptive_options.enabled ||
        adaptive_depth_control_mode_[gpu_id] !=
            AdaptiveFMMDepthControlMode::Autonomous)
        return;
    // A periodic snapshot may still be queued.  Preserve it before reusing
    // the staging buffer, then capture all newer records at this boundary.
    (void)harvest_adaptive_depth_diagnostic_snapshot(gpu_id, true);
    if (!enqueue_adaptive_depth_diagnostic_snapshot(gpu_id, stream))
        throw std::logic_error(
            "adaptive-depth diagnostic staging remained busy after synchronization");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    (void)harvest_adaptive_depth_diagnostic_snapshot(gpu_id, true);
    adaptive_depth_diagnostic_drain_[gpu_id].evaluations_since_drain = 0;
}

static void release_adaptive_depth_diagnostic_drain(int gpu_id)
{
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    if (drain.snapshot || drain.snapshot_ready)
        CUDA_CHECK(cudaSetDevice(gpu_id));
    if (drain.snapshot_pending)
        (void)harvest_adaptive_depth_diagnostic_snapshot(gpu_id, true);
    if (!drain.backlog.empty()) {
        std::fprintf(
            stderr,
            "[BF-FMM] WARNING: discarding %zu buffered adaptive-depth "
            "diagnostic record(s) during shutdown; consume diagnostics "
            "before ending the dispatcher lifetime\n",
            drain.backlog.size());
    }
    if (drain.snapshot_ready) CUDA_CHECK(cudaEventDestroy(drain.snapshot_ready));
    if (drain.snapshot) CUDA_CHECK(cudaFreeHost(drain.snapshot));
    drain.snapshot = nullptr;
    drain.snapshot_ready = nullptr;
    drain.snapshot_pending = false;
    drain.backlog.clear();
    drain.evaluations_since_drain = 0;
}

struct ShearFMMSourceArena {
    double* pos_x = nullptr;
    double* pos_y = nullptr;
    double* pos_z = nullptr;
    float* mass = nullptr;
    int* clump_ids = nullptr; // allocated only for rigid-clump scenes
    double* before_x = nullptr;
    double* before_y = nullptr;
    double* before_z = nullptr;
    double* diagnostic = nullptr; // sum Fx,Fy,Fz,sum |F_i|
    int capacity = 0;
};
static ::cudadem::DeviceStateArray<ShearFMMSourceArena> shear_source_arena_;
// Observed replaceable arena allocation, excluding the persistent LET plan.
// Do not credit a conservative capacity estimate as actually reclaimable VRAM.
static ::cudadem::DeviceStateArray<uint64_t> shear_reclaimable_bytes_;
static ::cudadem::DeviceStateArray<int> shear_target_count_;
static ::cudadem::DeviceStateArray<double> shear_cutoff_;
static ::cudadem::DeviceStateArray<double4> shear_fixed_root_;
static ::cudadem::DeviceStateArray<int> shear_node_bound_;
struct ShearCensusWorkspace {
    FMMDeviceBuffer<uint64_t> input, sorted;
    FMMDeviceBuffer<unsigned char> scratch;
    FMMDeviceBuffer<unsigned long long> histogram;
    void reset() { input.reset(); sorted.reset(); scratch.reset(); histogram.reset(); }
};
static ::cudadem::DeviceStateArray<ShearCensusWorkspace> shear_census_workspace_;
static ::cudadem::DeviceStateArray<uint64_t> shear_expected_nodes_;

static std::array<uint64_t,21> shear_occupied_census(const SimulationState& state,
    int targets, const ShearGravityImages& images, double4 root, int depth,
    int dimension, cudaStream_t stream)
{
    if(depth<1||depth>20)throw std::invalid_argument("invalid shear census depth");
    if(targets==0)return {};
    const int64_t sources64=int64_t(targets)*(2*images.layers_x+1);
    if(sources64<=0||sources64>INT_MAX/2)throw std::overflow_error("shear census source indexing");
    const int sources=static_cast<int>(sources64);
    int gpu=0;CUDA_CHECK(cudaGetDevice(&gpu));
    auto& ws=shear_census_workspace_[gpu];
    size_t scratch_bytes=0;
    const int bits=dimension==2?TreeTraits<2>::MORTON_BITS:TreeTraits<3>::MORTON_BITS;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,scratch_bytes,
        static_cast<uint64_t*>(nullptr),static_cast<uint64_t*>(nullptr),sources,bits-dimension*depth,bits,stream));
    uint64_t additional=0;
    auto growth=[&](size_t wanted,size_t current,size_t width){
        if(wanted>current)additional+=(wanted+wanted/4-current)*width;
    };
    growth(sources,ws.input.capacity,8);growth(sources,ws.sorted.capacity,8);
    growth(scratch_bytes,ws.scratch.capacity,1);growth(22,ws.histogram.capacity,8);
    size_t free_bytes=0,total_bytes=0;CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    if(additional>shear_fmm_capacity::memory_allowance(free_bytes))
        throw std::runtime_error("shear occupied-prefix census workspace cannot fit with reserve");
    ws.input.reserve(sources);ws.sorted.reserve(sources);ws.scratch.reserve(scratch_bytes);ws.histogram.reserve(22);
    BF_FMM_NVTX_RANGE("BF-FMM: Shear occupied-prefix census");
    CUDA_CHECK(cudaMemsetAsync(ws.histogram.data,0,22*sizeof(unsigned long long),stream));
    const int blocks=(sources+255)/256;
    if(dimension==2)shear_census_keys_kernel<2><<<blocks,256,0,stream>>>(state.particles.pos_x,state.particles.pos_y,state.particles.pos_z,
        targets,sources,images,root,depth,ws.input.data,ws.histogram.data);
    else shear_census_keys_kernel<3><<<blocks,256,0,stream>>>(state.particles.pos_x,state.particles.pos_y,state.particles.pos_z,
        targets,sources,images,root,depth,ws.input.data,ws.histogram.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(ws.scratch.data,scratch_bytes,ws.input.data,ws.sorted.data,
        sources,bits-dimension*depth,bits,stream));
    if(dimension==2)shear_census_prefix_kernel<2><<<blocks,256,0,stream>>>(ws.sorted.data,sources,ws.histogram.data);
    else shear_census_prefix_kernel<3><<<blocks,256,0,stream>>>(ws.sorted.data,sources,ws.histogram.data);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long histogram[22]{};
    CUDA_CHECK(cudaMemcpyAsync(histogram,ws.histogram.data,sizeof(histogram),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));BF_FMM_NVTX_END();
    if(histogram[21])throw std::runtime_error("shear census found nonfinite particle or source outside fixed root");
    std::array<uint64_t,21> changes{};
    std::copy(histogram,histogram+21,changes.begin());
    return shear_fmm_capacity::occupied_nodes(changes,true);
}

static int shear_occupied_node_bound(const SimulationState& state, const SimConfig& config,
    int targets, int sources, int depth, const ShearGravityImages& images, double4 root,
    cudaStream_t stream)
{
    const auto counts=shear_occupied_census(state,targets,images,root,depth,
        config.acceleration.unified_fmm.tree_dimension,stream);
    const auto& fmm=config.acceleration.unified_fmm;
    const int particles=shear_fmm_capacity::padded_particles(sources,fmm.particle_overallocate_factor);
    const int strict=fixed_depth_node_upper_bound(particles,depth,fmm.tree_dimension);
    int gpu=0;CUDA_CHECK(cudaGetDevice(&gpu));
    shear_expected_nodes_[gpu]=counts[depth];
    if(initialized_[gpu] && depth<=persistent_tree_runtime_[gpu].allocated_depth &&
       counts[depth]<=uint64_t(persistent_tree_runtime_[gpu].host_context.node_capacity))
        return persistent_tree_runtime_[gpu].host_context.node_capacity;
    // Recount before every build: moving particles/image phase cannot reuse a
    // stale bound. Headroom avoids resizing for small population changes.
    return shear_fmm_capacity::occupied_capacity(counts[depth],strict,
        std::max(1.0f,fmm.adaptive_depth.capacity_headroom));
}

__global__ void materialize_shear_gravity_sources_kernel(
    const double* source_x, const double* source_y, const double* source_z,
    const float* source_mass, int source_count, ShearGravityImages images,
    double* image_x, double* image_y, double* image_z, float* image_mass,
    const int* source_clump_ids = nullptr, int* image_clump_ids = nullptr)
{
    const int out = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = source_count * (2*images.layers_x+1);
    if (out >= total) return;
    const int slot = out / source_count;
    const int source = out - slot * source_count;
    int32_t nx = 0, ny = 0;
    shear_gravity_image_indices(slot, images.layers_x, 0, nx, ny);
    image_x[out] = source_x[source] + static_cast<double>(nx) * images.length_x;
    image_y[out] = shear_gravity_canonical_y(source_y[source],nx,images);
    image_z[out] = source_z[source];
    image_mass[out] = source_mass[source];
    if (image_clump_ids) image_clump_ids[out] = source_clump_ids[source];
}

static void ensure_shear_source_arena(int gpu, int count)
{
    auto& arena = shear_source_arena_[gpu];
    if (arena.capacity >= count) return;
    if (arena.pos_x) cudaFree(arena.pos_x);
    if (arena.pos_y) cudaFree(arena.pos_y);
    if (arena.pos_z) cudaFree(arena.pos_z);
    if (arena.mass) cudaFree(arena.mass);
    if (arena.clump_ids) cudaFree(arena.clump_ids);
    arena.clump_ids = nullptr;
    if (arena.before_x) cudaFree(arena.before_x);
    if (arena.before_y) cudaFree(arena.before_y);
    if (arena.before_z) cudaFree(arena.before_z);
    if (arena.diagnostic) cudaFree(arena.diagnostic);
    CUDA_CHECK(cudaMalloc(&arena.pos_x, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.pos_y, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.pos_z, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.mass, sizeof(float) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.before_x, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.before_y, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.before_z, sizeof(double) * static_cast<size_t>(count)));
    CUDA_CHECK(cudaMalloc(&arena.diagnostic, 4 * sizeof(double)));
    arena.capacity = count;
}

// The non-RESPA path adds gravity directly to force_*. Keep the optional
// output cache in force units, excluding the fast/contact force that was
// already present. Reuse the shear wrapper's existing before-force snapshot.
__global__ void cache_shear_gravity_force_kernel(
    const double* force_x, const double* force_y, const double* force_z,
    const double* before_x, const double* before_y, const double* before_z,
    double* gravity_x, double* gravity_y, double* gravity_z, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    gravity_x[i] = force_x[i] - before_x[i];
    gravity_y[i] = force_y[i] - before_y[i];
    gravity_z[i] = force_z[i] - before_z[i];
}

template<bool OUTER>
__global__ void shear_force_balance_reduce_kernel(
    const double* out_x, const double* out_y, const double* out_z,
    const double* before_x, const double* before_y, const double* before_z,
    const float* mass, int count, double* diagnostic)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double scale = OUTER ? static_cast<double>(mass[i]) : 1.0;
    const double fx = scale * (out_x[i] - before_x[i]);
    const double fy = scale * (out_y[i] - before_y[i]);
    const double fz = scale * (out_z[i] - before_z[i]);
    atomicAdd(diagnostic + 0, fx);
    atomicAdd(diagnostic + 1, fy);
    atomicAdd(diagnostic + 2, fz);
    atomicAdd(diagnostic + 3, sqrt(fx * fx + fy * fy + fz * fz));
}

__global__ void shear_force_balance_warn_kernel(
    const double* diagnostic, double threshold)
{
    if (blockIdx.x || threadIdx.x) return;
    const double numerator = sqrt(diagnostic[0] * diagnostic[0] +
                                  diagnostic[1] * diagnostic[1] +
                                  diagnostic[2] * diagnostic[2]);
    const double epsilon = diagnostic[3] > 0.0
        ? numerator / diagnostic[3] : 0.0;
    if (epsilon > threshold)
        printf("WARNING: finite shear-image gravity force imbalance "
               "epsilon_F=%.9e exceeds %.9e (diagnostic only; force unchanged)\n",
               epsilon, threshold);
}

template<bool OUTER>
static void launch_shear_force_balance(
    ShearFMMSourceArena& arena, const SimulationState& state,
    int count, double threshold, cudaStream_t stream)
{
    if (!(threshold > 0.0) || count <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(arena.diagnostic, 0, 4 * sizeof(double), stream));
    const double* x = OUTER ? state.particles.gravity_force_x
                            : state.particles.force_x;
    const double* y = OUTER ? state.particles.gravity_force_y
                            : state.particles.force_y;
    const double* z = OUTER ? state.particles.gravity_force_z
                            : state.particles.force_z;
    shear_force_balance_reduce_kernel<OUTER><<<(count + 255) / 256, 256, 0,
        stream>>>(x, y, z, arena.before_x, arena.before_y, arena.before_z,
                  state.particles.mass, count, arena.diagnostic);
    shear_force_balance_warn_kernel<<<1, 1, 0, stream>>>(
        arena.diagnostic, threshold);
    CUDA_CHECK(cudaGetLastError());
}

static int shear_effective_depth(const UnifiedFMMConfig& fmm,
                                 const ShearGravityImages& images,
                                 const PeriodicBoundaryConfig& pbc)
{
    return shear_fmm_effective_depth(fmm, images, pbc);
}

static double4 shear_static_root(const ShearGravityImages& images,
                                 const PeriodicBoundaryConfig& pbc,
                                 const SimConfig& config)
{
    return shear_fmm_static_root(images, pbc, config);
}

static void configure_distributed_shear_geometry(int gpu,const SimConfig& config,
    const PeriodicBoundaryConfig& pbc,double time)
{
    auto& state=body_fitted_fmm::distributed_fmm_[gpu];
    const auto images=make_shear_gravity_images(config,pbc,time);
    const auto root=images.enabled?shear_static_root(images,pbc,config):make_double4(0,0,0,0);
    const bool changed=state.routing.enabled!=images.enabled ||
        (images.enabled && (state.routing.length_x!=images.length_x ||
        state.routing.length_y!=images.length_y ||
        state.routing.min_x!=pbc.box_min_x || state.routing.max_x!=pbc.box_max_x ||
        state.routing.min_y!=pbc.box_min_y || state.routing.max_y!=pbc.box_max_y ||
        state.routing.gravity.x!=root.x || state.routing.gravity.y!=root.y ||
        state.routing.gravity.z!=root.z || state.routing.gravity.size!=root.w ||
        state.routing.layers_x!=images.layers_x || state.routing.layers_y!=images.layers_y));
    // Shear phase moves the native canonical sources. Their occupied-cell
    // identity check invalidates bindings when needed; movement within the
    // same cells only refreshes values and does not rebuild the geometric halo.
    if(changed){state.gpu_plan.cached_nodes=-1;}
    state.routing.enabled=images.enabled;
    state.routing.local_owner_mask=0;
    if(!images.enabled){state.grid=state.ownership_grid;return;}
    if(state.ranks>32)throw std::invalid_argument("Shear source routing supports at most 32 GPU owners");
    state.grid=state.ownership_grid;
    state.grid.x=root.x;state.grid.y=root.y;state.grid.z=root.z;state.grid.size=root.w;
    state.grid.dimension=config.acceleration.unified_fmm.tree_dimension;
    state.routing.gravity=state.grid;
    state.routing.layers_x=images.layers_x;state.routing.layers_y=images.layers_y;
    state.routing.length_x=images.length_x;state.routing.length_y=images.length_y;
    state.routing.shift=images.shear_offset;
    state.routing.min_x=pbc.box_min_x;state.routing.max_x=pbc.box_max_x;
    state.routing.min_y=pbc.box_min_y;state.routing.max_y=pbc.box_max_y;
}

// Single-device shear uses the same independent source halo, without a peer
// coordinator. Its sole producer is this CUDA device, including device != 0.
struct LocalShearHaloScope {
    DistributedFMMState& state;
    bool active, restore;
    int saved_ranks;
    std::vector<uint64_t> saved_cuts;
    ::cudadem::MortonGrid saved_grid;
    ::cudadem::ShearSourceRouting saved_routing;
    LocalShearHaloScope(int gpu, const SimConfig& config,
        const PeriodicBoundaryConfig& pbc, double time)
        : state(distributed_fmm_[gpu]), active(!state.evaluating),
          restore(active && state.enabled), saved_ranks(state.ranks),
          saved_grid(state.grid), saved_routing(state.routing) {
        if (!active) return;
        if (restore) saved_cuts = state.cuts;
        state.local_only = true; state.evaluating = true; state.ranks = gpu + 1;
        state.cuts.assign(state.ranks + 1, 0);
        state.cuts.back() = uint64_t{1} << (3 * ::cudadem::MortonGrid::key_depth);
        if (state.peers.size() != size_t(state.ranks)) {
            state.peers.clear(); state.peers.resize(state.ranks);
            for (auto& peer : state.peers) peer = std::make_unique<FMMQueryPeer>();
            state.gpu_plan.cached_nodes = -1;
        }
        configure_distributed_shear_geometry(gpu, config, pbc, time);
        state.routing.local_owner_mask = uint32_t{1} << gpu;
    }
    ~LocalShearHaloScope() {
        if (!active) return;
        state.evaluating = false; state.local_only = false;
        if (restore) {
            state.ranks = saved_ranks; state.cuts = std::move(saved_cuts);
            state.grid = saved_grid; state.routing = saved_routing;
            state.gpu_plan.cached_nodes = -1;
            state.peers.clear(); state.peers.resize(state.ranks);
            for (auto& peer : state.peers) peer = std::make_unique<FMMQueryPeer>();
        }
    }
};

struct ShearFMMMemoryEstimate {
    uint64_t legacy_tree, routes, expansions, tree_work, source_staging, diagnostic_staging, sparse_dense, cub, fixed, reuse_work;
    uint64_t total() const { return legacy_tree + routes + expansions + tree_work + source_staging + diagnostic_staging + sparse_dense + cub + fixed + reuse_work; }
};

static ShearFMMMemoryEstimate estimate_shear_fmm_memory(const SimConfig& config,
    int particles, int depth, int bound, bool distributed)
{
    const auto& fmm = config.acceleration.unified_fmm;
    const int pcap = shear_fmm_capacity::padded_particles(particles,fmm.particle_overallocate_factor);
    const int strict_nodes = body_fitted_fmm::fixed_depth_node_upper_bound(
        pcap, depth, fmm.tree_dimension);
    int nodes = bound > 0 ? std::min(strict_nodes, bound) : strict_nodes;
    if (distributed) nodes = std::max(nodes, std::min(65536,
        body_fitted_fmm::fixed_depth_node_upper_bound(std::max(pcap,4096), depth, fmm.tree_dimension)));
    const uint64_t p = static_cast<uint64_t>(pcap);
    const uint64_t n = static_cast<uint64_t>(nodes);
    // Match allocate_memory(), including legacy topology arrays still owned
    // by FMMHostData. P remains particle capacity; N is occupied-node capacity.
    const uint64_t legacy_tree = p * (sizeof(uint64_t) + 3 * sizeof(int)) +
        n * (sizeof(fmm::FMMOctreeNode) + sizeof(UnifiedNode) + sizeof(int2) +
             31 * sizeof(int) + sizeof(float) + 3 * sizeof(far_real_t));
    const uint64_t routes = n * sizeof(GroupedTargetInteractions) + 4 * sizeof(int);
    const uint64_t expansions = n * (sizeof(fmm::MultipoleExpansion) +
        sizeof(fmm::LocalExpansion) + fmm::FMM_ACTIVE_COEFFS * sizeof(double));
    const uint64_t edges = 2 * p - 1;
    const uint64_t level_blocks = (n + 255) / 256;
    const uint64_t tree_work = p * (2 * sizeof(uint64_t) + 2 * sizeof(int) + sizeof(int2)) +
        edges * (sizeof(KarrasBinaryNode) + 4 * sizeof(int)) +
        ((p + 255) / 256) * 6 * sizeof(double) +
        2 * fmm::FMM_MAX_LEVELS * level_blocks * sizeof(int) +
        4 * std::min<uint64_t>(8192,n) * sizeof(int);
    const uint64_t source_staging = p * (3 * sizeof(double) + sizeof(float));
    const uint64_t diagnostic_staging = p * 3 * sizeof(double) + 4 * sizeof(double);
    uint64_t dense_entries = 0, sparse_entries = 0;
    const int dense_depth = fmm.tree_dimension == 2 ? 10 : 7;
    const int dimension = fmm.tree_dimension;
    for (int level = 0; level <= depth; ++level) {
        if(level<=dense_depth) dense_entries += 1ull << (dimension * level);
        else sparse_entries += shear_fmm_capacity::sparse_slots(p,n,level,dimension);
    }
    const uint64_t sparse_dense = dense_entries * 2 * sizeof(uint32_t) +
        sparse_entries * (sizeof(uint64_t) + sizeof(uint32_t));
    // Query the same CUB operations and capacities used by the allocator.
    size_t sort_bytes=0, edge_bytes=0, target_bytes=0, level_bytes=0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,
        static_cast<uint64_t*>(nullptr),static_cast<uint64_t*>(nullptr),
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),pcap,0,64));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,edge_bytes,
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),static_cast<int>(edges)));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,target_bytes,
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),pcap));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,level_bytes,
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),static_cast<int>(level_blocks)));
    const uint64_t cub = sort_bytes + std::max(
        body_fitted_fmm::karras_unique_selection_scratch_bytes(pcap),
        std::max(edge_bytes,std::max(target_bytes,level_bytes)));
    const uint64_t fixed = 32ull * 1024ull * 1024ull;
    uint64_t reuse_work = 0;
    return {legacy_tree, routes, expansions, tree_work, source_staging, diagnostic_staging, sparse_dense, cub, fixed, reuse_work};
}

static void preflight_shear_fmm_memory(const SimConfig& config, int particles, int depth)
{
    int gpu = 0;
    CUDA_CHECK(cudaGetDevice(&gpu));
    const auto estimate = estimate_shear_fmm_memory(config, particles, depth,
        shear_node_bound_[gpu], distributed_fmm_[gpu].enabled);
    const uint64_t estimated = estimate.total();
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const long double allowed = std::min<long double>(
        0.80L * static_cast<long double>(free_bytes),
        free_bytes > 2ull * 1024ull * 1024ull * 1024ull
            ? static_cast<long double>(free_bytes -
                2ull * 1024ull * 1024ull * 1024ull) : 0.0L);
    if (estimated > allowed) {
        auto mib = [](long double bytes) { return bytes / (1024.0L * 1024.0L); };
        std::ostringstream out;
        out << "finite shear-image BFFMM memory preflight failed: estimated "
            << static_cast<double>(mib(estimated)) << " MiB > allowed "
            << static_cast<double>(mib(allowed)) << " MiB (free "
            << static_cast<double>(mib(free_bytes)) << " MiB; reserve=max(20%,2GiB)). "
            << "Breakdown MiB: legacy/tree=" << static_cast<double>(mib(estimate.legacy_tree))
            << ", routes=" << static_cast<double>(mib(estimate.routes))
            << ", multipole+local=" << static_cast<double>(mib(estimate.expansions))
            << ", Karras work=" << static_cast<double>(mib(estimate.tree_work))
            << ", source staging=" << static_cast<double>(mib(estimate.source_staging))
            << ", diagnostic staging=" << static_cast<double>(mib(estimate.diagnostic_staging))
            << ", dense+sparse lookup=" << static_cast<double>(mib(estimate.sparse_dense))
            << ", CUB estimate=" << static_cast<double>(mib(estimate.cub))
            << ", fixed=" << static_cast<double>(mib(estimate.fixed))
            << ", image reuse=" << static_cast<double>(mib(estimate.reuse_work))
            << ". Reduce particle capacity/image layers or use direct gravity; "
               "the solver will not reduce image physics automatically.";
        throw std::runtime_error(out.str());
    }
}

static void ensure_shear_fmm_capacity(const SimConfig& config, int particles, int gpu)
{
    const auto& fmm = config.acceleration.unified_fmm;
    size_t allocation_free_before = 0, allocation_total = 0;
    const bool rebuild_tree = !initialized_[gpu] ||
        persistent_tree_runtime_[gpu].host_context.use_presorted_morton != 0 ||
        particles > host_data_[gpu].allocated_max_particles ||
        shear_node_bound_[gpu] > persistent_tree_runtime_[gpu].host_context.node_capacity ||
        initialized_tree_type_[gpu] != fmm.tree_dimension ||
        persistent_tree_runtime_[gpu].allocated_depth < fmm.tree_depth;

    if (rebuild_tree) {
        // initialize() already replaces these arenas. Release them before
        // measuring free memory so the full-allocation guard does not count
        // the allocation being replaced twice. A failed guard leaves the
        // dispatcher uninitialized, with no replacement allocation started.
        if (initialized_[gpu]) UnifiedFMMDispatcher::shutdown();
        preflight_shear_fmm_memory(config, particles, fmm.tree_depth);
        if(std::getenv("GRANIT_SHEAR_CAPACITY_DIAGNOSTICS"))std::printf(
            "[Shear allocation] gpu=%d effective=%d sources=%d node_capacity=%d\n",
            gpu,fmm.tree_depth,particles,shear_node_bound_[gpu]);
        CUDA_CHECK(cudaMemGetInfo(&allocation_free_before, &allocation_total));
        UnifiedFMMDispatcher::initialize(config, particles);
    } else if (host_data_[gpu].allocated_max_particles > shear_source_arena_[gpu].capacity) {
        // The FMM particle buffers have headroom; source staging may still
        // need growth. ensure_shear_source_arena frees every old buffer before
        // allocating replacements, so only the capacity difference is new
        // device memory. Its diagnostic buffer is reused in size (4 doubles).
        const int old_capacity = shear_source_arena_[gpu].capacity;
        const uint64_t additional =
            static_cast<uint64_t>(host_data_[gpu].allocated_max_particles - old_capacity) *
                (6 * sizeof(double) + sizeof(float)) +
            (old_capacity == 0 ? 4 * sizeof(double) : 0);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const long double allowed = std::min<long double>(
            0.80L * static_cast<long double>(free_bytes),
            free_bytes > 2ull * 1024ull * 1024ull * 1024ull
                ? static_cast<long double>(free_bytes -
                    2ull * 1024ull * 1024ull * 1024ull) : 0.0L);
        if (static_cast<long double>(additional) > allowed) {
            std::ostringstream out;
            out << "finite shear-image BFFMM source growth memory preflight failed: "
                << additional << " additional bytes > allowed "
                << static_cast<double>(allowed) << " bytes (free "
                << free_bytes << " bytes; reserve=max(20%,2GiB)); capacity "
                << old_capacity << " -> " << host_data_[gpu].allocated_max_particles;
            throw std::runtime_error(out.str());
        }
        allocation_free_before = free_bytes;
    }
    // Repeated force evaluations reuse both arenas and need no new budget.
    ensure_shear_source_arena(gpu, host_data_[gpu].allocated_max_particles);
    if (!rebuild_tree)
        body_fitted_fmm::switch_persistent_tree_depth(
            persistent_tree_runtime_[gpu], fmm.tree_depth, fmm.tree_dimension);
    if (allocation_free_before) {
        size_t after = 0;
        CUDA_CHECK(cudaMemGetInfo(&after, &allocation_total));
        const uint64_t delta = allocation_free_before > after ? allocation_free_before - after : 0;
        if (rebuild_tree) shear_reclaimable_bytes_[gpu] = delta;
        else shear_reclaimable_bytes_[gpu] += delta;
    }
}

static void preflight_shear_float_centers(double4 root, int depth)
{
    const double leaf = std::ldexp(root.w, -depth);
    auto separated = [leaf, root](double min) {
        return static_cast<float>(min + 0.5 * leaf) !=
                   static_cast<float>(min + 1.5 * leaf) &&
               static_cast<float>(min + root.w - 0.5 * leaf) !=
                   static_cast<float>(min + root.w - 1.5 * leaf);
    };
    if (!(leaf > 0.0) || !separated(root.x) || !separated(root.y) ||
        !separated(root.z)) {
        std::ostringstream out;
        out << "finite shear-image BFFMM float cell centers are not distinct: "
            << "root_min=(" << root.x << ',' << root.y << ',' << root.z
            << "), side=" << root.w << ", depth=" << depth
            << ", leaf_width=" << leaf;
        throw std::invalid_argument(out.str());
    }
}

// Fixed, nonblocking telemetry ring.  A full ring drops a measurement rather
// than waiting for a prior one, keeping calibration invisible to simulation
// scheduling.
static constexpr int BFFMM_TELEMETRY_RING_SIZE = 16;
struct BFFMMTelemetrySlot {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    bool in_flight = false;
    BFFMMActiveSample completed{};
};
static ::cudadem::DeviceStateArray<std::array<BFFMMTelemetrySlot, BFFMM_TELEMETRY_RING_SIZE>> telemetry_ring_;

static ::cudadem::DeviceStateArray<double> last_validation_m2l_ms_;

static int configured_initial_tree_depth(const UnifiedFMMConfig& fmm)
{
    if (!fmm.adaptive_depth.enabled) return fmm.tree_depth;
    if (fmm.adaptive_depth.initial_depth_auto || fmm.tree_depth <= 0)
        return (fmm.adaptive_depth.min_depth +
                fmm.adaptive_depth.max_depth) / 2;
    return fmm.tree_depth;
}

static body_fitted_fmm::AdaptiveDepthOptions adaptive_depth_options(
    const UnifiedFMMConfig& fmm)
{
    body_fitted_fmm::AdaptiveDepthOptions options;
    options.enabled = fmm.adaptive_depth.enabled;
    options.auto_initial_depth = fmm.adaptive_depth.initial_depth_auto;
    options.initial_depth = configured_initial_tree_depth(fmm);
    options.min_depth = fmm.adaptive_depth.min_depth;
    options.max_depth = fmm.adaptive_depth.max_depth;
    options.trial_interval_evaluations =
        fmm.adaptive_depth.trial_interval_evaluations;
    options.trial_warmup_evaluations =
        fmm.adaptive_depth.trial_warmup_evaluations;
    options.trial_samples = fmm.adaptive_depth.trial_samples;
    options.min_improvement_fraction =
        fmm.adaptive_depth.min_improvement_fraction;
    options.confirmation_trials = fmm.adaptive_depth.confirmation_trials;
    options.capacity_headroom = fmm.adaptive_depth.capacity_headroom;
    options.capacity_growth_trigger =
        fmm.adaptive_depth.capacity_growth_trigger;
    return options;
}

static bool persistent_depth_configuration_matches(
    const body_fitted_fmm::PersistentTreeRuntime& runtime,
    const UnifiedFMMConfig& fmm)
{
    if (!runtime.initialized ||
        runtime.adaptive_options.enabled != fmm.adaptive_depth.enabled)
        return false;
    if (!fmm.adaptive_depth.enabled)
        return runtime.host_context.tree_depth == fmm.tree_depth;
    if (runtime.adaptive_options.min_depth != fmm.adaptive_depth.min_depth ||
        runtime.adaptive_options.max_depth != fmm.adaptive_depth.max_depth ||
        runtime.adaptive_options.auto_initial_depth !=
            fmm.adaptive_depth.initial_depth_auto)
        return false;
    // An automatic startup probe replaces the configured midpoint with the
    // measured depth.  That measured value is runtime state, not a
    // configuration mismatch that should tear down and rebuild the entire
    // persistent dispatcher on the next gravity epoch.
    return fmm.adaptive_depth.initial_depth_auto ||
           runtime.adaptive_options.initial_depth ==
               configured_initial_tree_depth(fmm);
}

static int sph_physical_depth_ceiling(const SimulationState& state, const SimConfig& config,
                                      int count, cudaStream_t stream) {
    double extent = 0.0;
    for (const double* axis : {state.particles.pos_x, state.particles.pos_y, state.particles.pos_z}) {
        const auto begin = thrust::device_pointer_cast(axis);
        const auto mm = thrust::minmax_element(thrust::cuda::par.on(stream), begin, begin + count);
        double low, high;
        CUDA_CHECK(cudaMemcpyAsync(&low, thrust::raw_pointer_cast(mm.first), sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&high, thrust::raw_pointer_cast(mm.second), sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        extent = std::max(extent, high-low);
    }
    const double root = (extent < 1.e-30 ? 1.0 : extent)*1.02;
    const double hmax = compute_sph_h_max(state, count, stream);
    if (!(hmax > 0.0) || hmax > 128.0*config.sph.hsml_min)
        throw std::runtime_error("SPH FMM requires h_max/hsml_min <= 128");
    int ceiling = 20;
    while (ceiling >= 2 && hmax > 32.0*std::ldexp(root, -ceiling)) --ceiling;
    if (ceiling < config.acceleration.unified_fmm.adaptive_depth.min_depth)
        throw std::runtime_error("SPH smoothing lengths require a shallower adaptive min_depth");
    return ceiling;
}

static AdaptiveStartupProbeResult run_adaptive_startup_probe(
    const SimulationState& state, const UnifiedFMMConfig& fmm,
    int particle_count, int particle_capacity, cudaStream_t stream)
{
    if (particle_count <= 0)
        throw std::invalid_argument("adaptive FMM startup probe requires particles");
    AdaptiveStartupProbeResult result;
    double bounds[6] = {};
    const double* axes[3] = {state.particles.pos_x, state.particles.pos_y,
                             state.particles.pos_z};
    for (int axis = 0; axis < 3; ++axis) {
        auto begin = thrust::device_pointer_cast(axes[axis]);
        auto extrema = thrust::minmax_element(
            thrust::cuda::par.on(stream), begin, begin + particle_count);
        CUDA_CHECK(cudaMemcpyAsync(bounds + 2 * axis,
            thrust::raw_pointer_cast(extrema.first), sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(bounds + 2 * axis + 1,
            thrust::raw_pointer_cast(extrema.second), sizeof(double),
            cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double ex = bounds[1] - bounds[0];
    const double ey = bounds[3] - bounds[2];
    const double ez = bounds[5] - bounds[4];
    double max_extent = std::max(ex, std::max(ey, ez));
    if (max_extent < 1.0e-30) max_extent = 1.0;
    const double pad = 0.01 * max_extent;
    const double root_size = max_extent + 2.0 * pad;
    const double min_x = bounds[0] - pad;
    const double min_y = bounds[2] - pad;
    const double min_z = bounds[4] - pad;

    uint64_t* keys = nullptr;
    int* flags = nullptr;
    CUDA_CHECK(cudaMalloc(&keys, static_cast<size_t>(particle_count) *
                                  sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&flags, static_cast<size_t>(particle_count) *
                                   sizeof(int)));
    const int block = 256;
    const int grid = (particle_count + block - 1) / block;

    auto count_dimension = [&](int dimension) {
        if (dimension == 2) {
            adaptive_probe_morton_kernel<2><<<grid, block, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y,
                state.particles.pos_z, particle_count, min_x, min_y, min_z,
                root_size, fmm.adaptive_depth.max_depth, keys);
        } else {
            adaptive_probe_morton_kernel<3><<<grid, block, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y,
                state.particles.pos_z, particle_count, min_x, min_y, min_z,
                root_size, fmm.adaptive_depth.max_depth, keys);
        }
        CUDA_CHECK(cudaGetLastError());
        thrust::sort(thrust::cuda::par.on(stream),
                     thrust::device_pointer_cast(keys),
                     thrust::device_pointer_cast(keys + particle_count));
        int cumulative_nodes = 1;
        double best_score = std::numeric_limits<double>::infinity();
        int best_depth = fmm.adaptive_depth.min_depth;
        for (int level = 1; level <= fmm.adaptive_depth.max_depth; ++level) {
            const int morton_bits = dimension == 2
                ? TreeTraits<2>::MORTON_BITS : TreeTraits<3>::MORTON_BITS;
            const int shift = morton_bits - dimension * level;
            adaptive_probe_prefix_flags_kernel<<<grid, block, 0, stream>>>(
                keys, particle_count, shift, flags);
            CUDA_CHECK(cudaGetLastError());
            const int occupied = thrust::reduce(
                thrust::cuda::par.on(stream),
                thrust::device_pointer_cast(flags),
                thrust::device_pointer_cast(flags + particle_count), 0,
                thrust::plus<int>());
            cumulative_nodes += occupied;
            if (level < fmm.adaptive_depth.min_depth) continue;
            result.node_count[level] = cumulative_nodes;
            result.leaf_count[level] = occupied;
            const double occupancy = static_cast<double>(particle_count) /
                                     std::max(1, occupied);
            const double score = std::abs(std::log(
                std::max(occupancy, 1.0e-12) / 16.0));
            if (score < best_score) {
                best_score = score;
                best_depth = level;
            }
        }
        result.initial_depth = best_depth;
    };

    // Probe exactly the configured topology.  D2 and D3 implementations stay
    // compiled, but adaptive allocation/control has one device-resident
    // controller for this runtime.
    count_dimension(fmm.tree_dimension);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(flags));
    CUDA_CHECK(cudaFree(keys));

    if (!fmm.adaptive_depth.initial_depth_auto)
        result.initial_depth = fmm.tree_depth;

    auto capacity_at = [&](int depth) {
        const int observed = result.node_count[depth];
        const int headroom = static_cast<int>(std::ceil(
            static_cast<double>(observed) *
            std::max(1.0f, fmm.adaptive_depth.capacity_headroom)));
        // Guarantee that min_depth can never overflow, so a disrupted body
        // always has a correct, controlled shallower escape path.
        const int min_guarantee = fixed_depth_node_upper_bound(
            particle_capacity, fmm.adaptive_depth.min_depth,
            fmm.tree_dimension);
        const int strict = fixed_depth_node_upper_bound(
            particle_capacity, depth, fmm.tree_dimension);
        return std::min(strict, std::max(min_guarantee, headroom));
    };

    const int provision_depth = std::min(fmm.adaptive_depth.max_depth,
        result.initial_depth + 1);
    result.node_capacity = capacity_at(provision_depth);
    int ceiling = fmm.adaptive_depth.min_depth;
    for (int depth = fmm.adaptive_depth.min_depth;
         depth <= fmm.adaptive_depth.max_depth; ++depth) {
        if (result.node_count[depth] > 0 &&
            result.node_count[depth] <= result.node_capacity)
            ceiling = depth;
    }
    result.effective_max_depth = ceiling;
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const uint64_t reserve = 2ull * 1024ull * 1024ull * 1024ull;
    const uint64_t allowed = std::min<uint64_t>(
        static_cast<uint64_t>(0.8 * static_cast<double>(free_bytes)),
        free_bytes > reserve ? free_bytes - reserve : free_bytes / 2);
    const int eventual_max_capacity = capacity_at(
        fmm.adaptive_depth.max_depth);
    const AdaptiveMemoryEstimate requested = estimate_adaptive_fmm_memory(
        particle_capacity, eventual_max_capacity,
        fmm.adaptive_depth.max_depth, fmm.tree_dimension);
    int largest_feasible = 0;
    for (int depth = fmm.adaptive_depth.min_depth;
         depth <= fmm.adaptive_depth.max_depth; ++depth) {
        const AdaptiveMemoryEstimate candidate = estimate_adaptive_fmm_memory(
            particle_capacity, capacity_at(depth), depth, fmm.tree_dimension);
        if (candidate.total() <= allowed) largest_feasible = depth;
    }
    if (requested.total() > allowed) {
        auto mib = [](uint64_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };
        std::ostringstream out;
        out << "adaptive BFFMM memory preflight failed for max_depth="
            << fmm.adaptive_depth.max_depth << ": required="
            << mib(requested.total()) << " MiB, allowed=" << mib(allowed)
            << " MiB, free=" << mib(free_bytes) << " MiB, total_gpu="
            << mib(total_bytes) << " MiB, largest_estimated_feasible_depth="
            << largest_feasible << ". Breakdown MiB: particles/workspace="
            << mib(requested.particles) << ", sparse_node_arrays="
            << mib(requested.nodes) << ", dense+sparse_lookup="
            << mib(requested.lookup) << ", CUB=" << mib(requested.cub)
            << ", reserve/fixed=" << mib(requested.fixed)
            << ". Eventual max-depth node_capacity=" << eventual_max_capacity
            << ", startup node_capacity=" << result.node_capacity
            << ", observed_nodes_at_max_depth="
            << result.node_count[fmm.adaptive_depth.max_depth]
            << ", headroom=" << fmm.adaptive_depth.capacity_headroom
            << ". Increase available GPU memory or explicitly lower max_depth; "
               "the runtime will not silently remove candidate depths.";
        throw std::runtime_error(out.str());
    }
    result.valid = true;
    return result;
}

static int begin_bffmm_sample(int gpu_id, BFFMMActivePhase phase,
                              int64_t epoch,
                              double work_units, RCBWorkPhase rcb_phase,
                              bool allocation_grew, cudaStream_t stream)
{
    auto& ring = telemetry_ring_[gpu_id];
    for (int i = 0; i < BFFMM_TELEMETRY_RING_SIZE; ++i) {
        BFFMMTelemetrySlot& slot = ring[i];
        if (slot.in_flight) continue;
        if (!slot.start) {
            CUDA_CHECK(cudaEventCreate(&slot.start));
            CUDA_CHECK(cudaEventCreate(&slot.end));
        }
        slot.completed.phase = phase;
        slot.completed.sample.epoch = epoch;
        slot.completed.sample.key.phase = rcb_phase;
        slot.completed.sample.key.fmm_mode = RCBFmmMode::Rebuild;
        slot.completed.sample.key.link_class = RCBLinkClass::PCIe;
        slot.completed.sample.active_seconds = 0.0;
        slot.completed.sample.work_units = work_units;
        slot.completed.sample.valid = work_units > 0.0;
        slot.completed.sample.allocation_grew = allocation_grew;
        slot.completed.sample.retried = false;
        CUDA_CHECK(cudaEventRecord(slot.start, stream));
        slot.in_flight = true;
        return i;
    }
    return -1;
}

static void end_bffmm_sample(int gpu_id, int slot_index, cudaStream_t stream,
                             double final_work_units = -1.0)
{
    if (slot_index >= 0) {
        if (final_work_units >= 0.0) {
            auto& sample = telemetry_ring_[gpu_id][slot_index].completed.sample;
            sample.work_units = final_work_units;
            sample.valid = final_work_units > 0.0;
        }
        CUDA_CHECK(cudaEventRecord(telemetry_ring_[gpu_id][slot_index].end, stream));
    }
}

static void free_bffmm_telemetry_ring(int gpu_id)
{
    for (BFFMMTelemetrySlot& slot : telemetry_ring_[gpu_id]) {
        if (slot.start) cudaEventDestroy(slot.start);
        if (slot.end) cudaEventDestroy(slot.end);
        slot = BFFMMTelemetrySlot{};
    }
}

static void invalidate_workload_view(int gpu_id)
{
    workload_view_valid_[gpu_id] = false;
    ++workload_generation_[gpu_id];
}

static void publish_rebuilt_workload_view(int gpu_id, int expected_owned_particles)
{
    // The published mapping must describe exactly the owned prefix used to
    // build this tree. Keep the view invalid rather than exposing a
    // partially rebuilt or mismatched tree.
    if (device_data_[gpu_id].num_particles != expected_owned_particles) {
        fprintf(stderr, "[BF-FMM] refusing workload view: tree has %d particles, expected %d\n",
                device_data_[gpu_id].num_particles, expected_owned_particles);
        invalidate_workload_view(gpu_id);
        return;
    }
    workload_view_valid_[gpu_id] = true;
    ++workload_generation_[gpu_id];
}

class MultigpuOwnedOnlyScope {
public:
    explicit MultigpuOwnedOnlyScope(int gpu_id) : gpu_id_(gpu_id), saved_(multigpu_owned_only_[gpu_id])
    {
        multigpu_owned_only_[gpu_id_] = true;
    }
    ~MultigpuOwnedOnlyScope() { multigpu_owned_only_[gpu_id_] = saved_; }
    MultigpuOwnedOnlyScope(const MultigpuOwnedOnlyScope&) = delete;
    MultigpuOwnedOnlyScope& operator=(const MultigpuOwnedOnlyScope&) = delete;
private:
    int gpu_id_;
    bool saved_;
};

int UnifiedFMMDispatcher::distributed_depth_ceiling(const ::cudadem::SimConfig& config,
    int global_particles, const ::cudadem::SimulationState* local_state,
    const ::cudadem::PeriodicBoundaryConfig* pbc, cudaStream_t stream)
{
    const int gpu = current_gpu();
    const auto& fmm = config.acceleration.unified_fmm;
    size_t free_bytes=0, total_bytes=0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    if (config.shear_box.enabled) {
        if (!local_state || !pbc)
            throw std::invalid_argument("adaptive shear memory preflight requires live state and periodic bounds");
        const auto images = make_shear_gravity_images(config, *pbc, local_state->current_time);
        const int local=local_state->particles.num_local;
        const int64_t sources64=int64_t(std::max(local,1))*(2*images.layers_x+1);
        const int64_t global_sources=int64_t(std::max(global_particles,1))*(2*images.layers_x+1);
        if(sources64>INT_MAX/2||global_sources>INT_MAX/2)
            throw std::overflow_error("adaptive shear source count exceeds edge indexing");
        const int sources=static_cast<int>(sources64);
        const auto root=shear_static_root(images,*pbc,config);
        auto deepest=config;deepest.acceleration.unified_fmm.tree_depth=fmm.adaptive_depth.max_depth;
        const int offset=shear_effective_depth(deepest.acceleration.unified_fmm,images,*pbc)-fmm.adaptive_depth.max_depth;
        const int max_base=std::min(fmm.adaptive_depth.max_depth,20-offset);
        if(!local)return max_base;
        const auto occupied=shear_occupied_census(*local_state,local,images,root,max_base+offset,
            fmm.tree_dimension,stream);
        // Census scratch is live and charged in free_bytes; only arenas that
        // will really be replaced are credited as reclaimable.
        CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
        const uint64_t reclaim=initialized_[gpu]?shear_reclaimable_bytes_[gpu]:0;
        const uint64_t available=std::min<uint64_t>(total_bytes,free_bytes+reclaim);
        const uint64_t allowed=shear_fmm_capacity::memory_allowance(available);
        const int particles=shear_fmm_capacity::padded_particles(sources,fmm.particle_overallocate_factor);
        int ceiling=0;
        for(int base=fmm.adaptive_depth.min_depth;base<=max_base;++base){
            auto candidate=config;candidate.acceleration.unified_fmm.tree_depth=base;
            const int depth=shear_effective_depth(candidate.acceleration.unified_fmm,images,*pbc);
            try {
                const int strict=fixed_depth_node_upper_bound(particles,depth,fmm.tree_dimension);
                const int bound=shear_fmm_capacity::occupied_capacity(occupied[depth],strict,
                    std::max(1.0f,fmm.adaptive_depth.capacity_headroom));
                auto candidate=config;candidate.acceleration.unified_fmm.tree_depth=base;
                const auto estimate=estimate_shear_fmm_memory(candidate,sources,depth,bound,true);
                const int peers=std::max(0,distributed_fmm_[gpu].ranks-1);
                // Multi-rank source indices may repeat at each level; union
                // coordinates do not. Include the buffers' 25% growth policy.
                const uint64_t source_copies=peers?uint64_t(global_sources)*5/4*
                    ((depth+1)*sizeof(int)+3*sizeof(double)+sizeof(float)+
                     2*uint64_t(peers)*sizeof(FMMSourceParticle)):0;
                const uint64_t needed=estimate.total()+source_copies;
                if(std::getenv("GRANIT_SHEAR_CAPACITY_DIAGNOSTICS"))std::printf(
                    "[Shear depth capacity] gpu=%d base=%d effective=%d local_sources=%d occupied_nodes=%llu node_capacity=%d estimate_bytes=%llu allowed_bytes=%llu fit=%d\n",
                    gpu,base,depth,sources,(unsigned long long)occupied[depth],bound,
                    (unsigned long long)needed,(unsigned long long)allowed,needed<=allowed);
                if(needed>allowed)break;
                ceiling=base;
            }catch(const std::overflow_error&){break;}
        }
        return ceiling;
    }
    const double padded=std::ceil(std::max(global_particles,1)*double(fmm.particle_overallocate_factor));
    if(padded>INT_MAX)throw std::overflow_error("common FMM particle capacity exceeds indexing");
    const int particles=static_cast<int>(padded);
    uint64_t reclaim=0;
    if(initialized_[gpu])reclaim=estimate_adaptive_fmm_memory(host_data_[gpu].allocated_max_particles,
        max_nodes_capacity_[gpu],persistent_tree_runtime_[gpu].host_context.tree_depth,fmm.tree_dimension).total();
    const uint64_t available=std::min<uint64_t>(total_bytes,free_bytes+reclaim);
    const uint64_t allowed=shear_fmm_capacity::memory_allowance(available);
    int ceiling=0;
    for(int depth=fmm.adaptive_depth.min_depth;depth<=fmm.adaptive_depth.max_depth;++depth){
        const int nodes=fixed_depth_node_upper_bound(particles,depth,fmm.tree_dimension);
        const auto estimate=estimate_adaptive_fmm_memory(particles,nodes,depth,fmm.tree_dimension);
        const uint64_t source=uint64_t(particles)*(depth+1)*(sizeof(FMMSourceParticle)+sizeof(int));
        if(estimate.total()+source<=allowed)ceiling=depth;
    }
    return ceiling;
}

DistributedFMMMetrics UnifiedFMMDispatcher::distributed_metrics(cudaStream_t stream) {
    const int gpu=current_gpu();auto& state=distributed_fmm_[gpu];
    if(!state.phase_events[0]||!state.epoch)return {};
    CUDA_CHECK(cudaEventSynchronize(state.phase_events[6]));
    auto elapsed=[&](int first,int last){float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,state.phase_events[first],state.phase_events[last]));return double(ms);};
    auto& m=state.metrics;m.epoch=state.epoch;m.owned=state.publication.targets;
    m.total_ms=elapsed(0,6);m.tree_ms=elapsed(0,1);m.exchange_ms=elapsed(2,3);
    m.far_ms=elapsed(1,2)+elapsed(3,4);m.near_ms=elapsed(4,5);
    m.communication_ms=state.communication_ms;
    m.received_bytes=state.received_particle_count*sizeof(FMMSourceParticle);
    for(int peer=0;peer<state.ranks;++peer)if(peer!=gpu)
        m.received_bytes+=state.peers[peer]->request_count*(sizeof(FMMCellKey)+sizeof(FMMSourceReply));
    if(state.let_wire_bytes)m.received_bytes=state.let_wire_bytes;
    return m;
}

struct FMMWorkSum {
    __host__ __device__ double4 operator()(double4 a,double4 b)const{return make_double4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
};
__global__ void fmm_work_difference_kernel(const UnifiedNode* nodes,const double4* work,
    int count,int n,double4* difference,int* error,bool shear){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const auto node=nodes[i];
    if(shear && node.count){
        if(uint64_t(node.begin)+node.count>uint64_t(n)){atomicExch(error,1);return;}
        atomicAdd(&difference[node.begin].x,1.0/node.count);
        atomicAdd(&difference[node.begin+node.count].x,-1.0/node.count);
    }
    if(!node.target_count)return;
    const uint32_t span=shear?node.count:node.target_count;
    if(uint64_t(node.begin)+span>uint64_t(n)){atomicExch(error,1);return;}
    const double inv=1.0/node.target_count;
    const double value[4]={shear?0.0:inv,work[i].x*inv,work[i].y*inv,work[i].z*inv};
    auto* first=reinterpret_cast<double*>(difference+node.begin);auto* last=reinterpret_cast<double*>(difference+node.begin+span);
    for(int c=0;c<4;++c){atomicAdd(first+c,value[c]);atomicAdd(last+c,-value[c]);}
}
__global__ void fmm_work_scatter_kernel(double4* prefix,const int* order,int sources,int targets,
    DistributedFMMParticleWork* work,int* error,bool shear){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=sources)return;const auto p=prefix[i];
    const bool target=order[i]<targets;
    const auto positive=make_double4(fmax(0.0,p.x)+(shear?1.0:0.0),
        target?fmax(0.0,p.y):0.0,target?fmax(0.0,p.z):0.0,target?fmax(0.0,p.w):0.0);
    if(order[i]<0||order[i]>=sources){atomicExch(error,2);return;}
    prefix[i]=positive;
    if(shear){auto& out=work[order[i]%targets];
        atomicAdd(&out.tree,positive.x);atomicAdd(&out.far,positive.y);
        atomicAdd(&out.near,positive.z);atomicAdd(&out.communication,positive.w);
    }else work[order[i]]={positive.x,positive.y,positive.z,positive.w};
}
const DistributedFMMParticleWork* UnifiedFMMDispatcher::distributed_particle_work_device(int& n,double totals[4],cudaStream_t stream){
    const int gpu=current_gpu();auto& state=distributed_fmm_[gpu];auto& p=state.gpu_plan;
    const int sources=state.publication.count;
    n=state.publication.targets;for(int c=0;c<4;++c)totals[c]=0;if(!n)return nullptr;
    const int nodes=p.owned_nodes;if(!nodes)throw std::runtime_error("missing owned FMM work snapshot");
    p.scalar.reserve(4);CUDA_CHECK(cudaMemsetAsync(p.scalar.data,0,sizeof(int),stream));
    state.node_work.reserve(nodes);p.work_difference.reserve(sources+1);p.work_prefix.reserve(sources+1);p.particle_work.reserve(n);
    const auto ctx=persistent_tree_runtime_[gpu].device_context;

    if(state.grid.dimension==2)fmm_owned_node_work_kernel<2><<<(nodes+255)/256,256,0,stream>>>(ctx,nodes,state.cuts[gpu],state.cuts[gpu+1],state.node_work.data,state.routing,gpu);
    else fmm_owned_node_work_kernel<3><<<(nodes+255)/256,256,0,stream>>>(ctx,nodes,state.cuts[gpu],state.cuts[gpu+1],state.node_work.data,state.routing,gpu);
    for(int peer=0;peer<state.ranks;++peer) {
        if((peer==gpu&&!state.routing.enabled)||(state.local_only&&peer!=gpu))continue;
        const auto& q=*state.peers[peer];if(!q.request_count)continue;
        if(state.grid.dimension==2)fmm_owned_node_work_kernel<2><<<(nodes+255)/256,256,0,stream>>>(ctx,nodes,0,0,
            state.node_work.data,state.routing,gpu,q.halo_nodes.data,q.halo_children.data,q.moment_routes.data,peer!=gpu,q.route_index.data);
        else fmm_owned_node_work_kernel<3><<<(nodes+255)/256,256,0,stream>>>(ctx,nodes,0,0,
            state.node_work.data,state.routing,gpu,q.halo_nodes.data,q.halo_children.data,q.moment_routes.data,peer!=gpu,q.route_index.data);
    }
    CUDA_CHECK(cudaMemsetAsync(p.work_difference.data,0,(sources+1)*sizeof(double4),stream));
    CUDA_CHECK(cudaMemsetAsync(p.particle_work.data,0,n*sizeof(DistributedFMMParticleWork),stream));
    fmm_work_difference_kernel<<<(nodes+255)/256,256,0,stream>>>(device_data_[gpu].unified_nodes,state.node_work.data,nodes,sources,p.work_difference.data,p.scalar.data,state.routing.enabled);
    fmm_cub(p,[&](void* tmp,size_t& bytes){return cub::DeviceScan::InclusiveScan(tmp,bytes,p.work_difference.data,p.work_prefix.data,FMMWorkSum{},sources,stream);});
    fmm_work_scatter_kernel<<<(sources+255)/256,256,0,stream>>>(p.work_prefix.data,state.publication.sorted,sources,n,p.particle_work.data,p.scalar.data,state.routing.enabled);
    fmm_cub(p,[&](void* tmp,size_t& bytes){return cub::DeviceReduce::Reduce(tmp,bytes,p.work_prefix.data,p.work_difference.data,sources,FMMWorkSum{},make_double4(0,0,0,0),stream);});
    CUDA_CHECK(cudaMemcpyAsync(totals,p.work_difference.data,4*sizeof(double),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if(fmm_read_scalar(p.scalar.data,stream))throw std::runtime_error("invalid owned FMM work range or permutation");
    return p.particle_work.data;
}
void UnifiedFMMDispatcher::distributed_particle_work(std::vector<DistributedFMMParticleWork>& work,cudaStream_t stream){
    int n=0;double totals[4];const auto* device=distributed_particle_work_device(n,totals,stream);work.resize(n);
    if(n)CUDA_CHECK(cudaMemcpyAsync(work.data(),device,n*sizeof(DistributedFMMParticleWork),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void UnifiedFMMDispatcher::validate_distributed_particle_work(cudaStream_t stream){
    const int gpu=current_gpu();auto& state=distributed_fmm_[gpu];auto& p=state.gpu_plan;
    int n=0;double totals[4];const auto* device=distributed_particle_work_device(n,totals,stream);if(!n)return;
    const int sources=state.publication.count;
    std::vector<DistributedFMMParticleWork> actual(n);std::vector<double4> units(p.owned_nodes);
    std::vector<UnifiedNode> nodes(p.owned_nodes);std::vector<int> order(sources);
    CUDA_CHECK(cudaMemcpy(actual.data(),device,n*sizeof(actual[0]),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(units.data(),state.node_work.data,units.size()*sizeof(double4),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nodes.data(),device_data_[gpu].unified_nodes,nodes.size()*sizeof(UnifiedNode),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(order.data(),state.publication.sorted,sources*sizeof(int),cudaMemcpyDeviceToHost));
    std::vector<std::array<double,4>> expected(n);
    std::array<double,4> expected_totals{};
    // Independent membership oracle: enumerate each node's actual sources,
    // and allocate interaction work only to the central physical targets.
    if(state.routing.enabled)for(int i=0;i<sources;++i)expected.at(order.at(i)%n)[0]+=1.0;
    for(size_t i=0;i<nodes.size();++i){const auto node=nodes[i];
        for(uint32_t j=0;j<node.count;++j){const int original=order.at(node.begin+j);
            auto& out=expected.at(original%n);
            if(state.routing.enabled)out[0]+=1.0/node.count;
            if(original<n && node.target_count){
                const double inv=1.0/node.target_count;
                if(!state.routing.enabled)out[0]+=inv;
                out[1]+=units[i].x*inv;out[2]+=units[i].y*inv;out[3]+=units[i].z*inv;
            }
        }
    }
    for(const auto& row:expected)for(int c=0;c<4;++c)expected_totals[c]+=row[c];
    for(int i=0;i<n;++i){const auto a=actual[i];const double v[4]={a.tree,a.far,a.near,a.communication};
        for(int c=0;c<4;++c)if(std::abs(v[c]-expected[i][c])>1e-9*std::max(1.0,std::abs(expected[i][c])))
            throw std::runtime_error("GPU FMM work differs from CPU image range oracle");}
    for(int c=0;c<4;++c)if(std::abs(totals[c]-expected_totals[c])>1e-9*std::max(1.0,std::abs(expected_totals[c])))
        throw std::runtime_error("GPU FMM work total differs from CPU oracle");
    std::printf("gpu_fmm_work_oracle rank=%d epoch=%llu particles=%d sources=%d PASS\n",gpu,(unsigned long long)state.epoch,n,sources);
}

void UnifiedFMMDispatcher::set_distributed_compute_depth(int depth) {
    auto& state=distributed_fmm_[current_gpu()];
    if(depth<1 || depth>20 ||
       (state.ownership_grid.dimension==3 && depth<state.ownership_grid.depth))
        throw std::invalid_argument("Invalid common FMM computation depth");
    state.compute_depth=depth;
}

void UnifiedFMMDispatcher::set_distributed_clump_exclusion_diameter(double diameter) {
    if (!std::isfinite(diameter) || diameter < 0)
        throw std::invalid_argument("Invalid distributed clump exclusion diameter");
    distributed_fmm_[current_gpu()].clump_exclusion_diameter = diameter;
}

void UnifiedFMMDispatcher::set_distributed_partition(const ::cudadem::DomainParamsGPU& params) {
    set_distributed_fmm_partition(current_gpu(), params);
}

int UnifiedFMMDispatcher::current_gpu() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= initialized_.size())
        throw std::out_of_range("FMM CUDA device state index");
    return dev;
}
// =============================================================================
// Load the versioned, precomposed table shipped beside the executable.
// All modes share unrounded FP32 coefficients; TF32 pads and rounds on upload.
// =============================================================================
static void load_full_g_table(BodyFittedFMMDeviceData& bfd)
{
    static_assert(G_GAP0_COUNT == granit_g_table::counts[0] &&
                  G_GAP1_FWD_COUNT + G_GAP1_PLANAR_COUNT == granit_g_table::counts[1] &&
                  G_GAP1_REV_COUNT == granit_g_table::counts[2], "G-table canonical counts changed");
    constexpr bool tf32 = FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC;
    const auto path = granit_g_table::installed_path();
    const auto coefficients = granit_g_table::load(path);
    bfd.d_G_full_bytes = size_t(75) * 48 * (tf32 ? 32 : 28) * 32 * sizeof(float);
    std::cout << "Loading full 48-orientation G-table: " << path << std::endl;
    CUDA_CHECK(cudaMalloc(&bfd.d_G_full, bfd.d_G_full_bytes));
    RotGGapInfo gap_info[9] = {};
    size_t offset = 0;
    for (int gi = 0; gi < 3; ++gi) {
        const auto segment = granit_g_table::segment(coefficients, gi, tf32);
        const int slot = granit_g_table::gaps[gi] + 4;
        gap_info[slot].count = granit_g_table::counts[gi];
        gap_info[slot].float_offset = static_cast<int>(offset);
        CUDA_CHECK(cudaMemcpy(bfd.d_G_full + offset, segment.data(),
                              segment.size() * sizeof(float), cudaMemcpyHostToDevice));
        offset += segment.size();
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_canonG_gap_info, gap_info, sizeof(gap_info)));
}

// =============================================================================
// allocate_memory
// =============================================================================

void UnifiedFMMDispatcher::allocate_memory(int gpu_id, int num_particles,
                                               int max_nodes,
                                               const ::cudadem::SimConfig& config)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));
    invalidate_workload_view(gpu_id);
    free_memory(gpu_id);

    fmm::FMMHostData& hd = host_data_[gpu_id];
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    BodyFittedFMMDeviceData& bfd = bf_device_data_[gpu_id];

    const int np = shear_node_bound_[gpu_id]>0
        ? shear_fmm_capacity::padded_particles(num_particles,config.acceleration.unified_fmm.particle_overallocate_factor)
        : static_cast<int>(std::ceil(num_particles * config.acceleration.unified_fmm.particle_overallocate_factor));
    // Adaptive graphs execute serially over one shared backing allocation.
    // Until the startup probe has produced a smaller staged recommendation,
    // initialize with the strict bound for the deepest captured graph.
    const int allocation_depth =
        config.acceleration.unified_fmm.adaptive_depth.enabled
            ? config.acceleration.unified_fmm.adaptive_depth.max_depth
            : config.acceleration.unified_fmm.tree_depth;
    const int strict_nn = body_fitted_fmm::fixed_depth_node_upper_bound(
        np, allocation_depth,
        config.acceleration.unified_fmm.tree_dimension);
    const int local_nn = shear_node_bound_[gpu_id] > 0
        ? std::min(strict_nn, shear_node_bound_[gpu_id])
        : (config.acceleration.unified_fmm.adaptive_depth.enabled
            ? std::min(strict_nn, max_nodes) : strict_nn);
    // A local essential tree includes remote stencil cells. Its capacity is
    // independent of the number of owned particles (including tiny owners).
    const int nn = distributed_fmm_[gpu_id].enabled
        ? std::max(local_nn, std::min(65536, fixed_depth_node_upper_bound(
            std::max(np, 4096), allocation_depth, config.acceleration.unified_fmm.tree_dimension)))
        : local_nn;
    if (nn <= 0)
        throw std::invalid_argument("BFFMM node capacity must be positive");
    max_nodes_capacity_[gpu_id] = nn;

    // ===== Tree topology arrays (owned directly by FMMHostData) =====
    // (Previously aliased from a BHTreeBuildData container; the BH build data
    //  struct has been retired — these arrays are now first-class FMM fields,
    //  allocated here with the same 1.2x headroom policy.)
    int max_nodes_est = std::max(2 * np, config.acceleration.unified_fmm.min_nodes_estimate);
    const int nn_est = shear_node_bound_[gpu_id] > 0
        ? nn : static_cast<int>(max_nodes_est * 1.2);

    // Morton keys & sort buffers
    CUDA_CHECK(cudaMalloc(&hd.morton_keys,    sizeof(uint64_t) * np));
    CUDA_CHECK(cudaMalloc(&hd.sorted_indices, sizeof(int)      * np));

    // Tree topology
    CUDA_CHECK(cudaMalloc(&hd.node_range,       sizeof(int2) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_children,    sizeof(int) * 8 * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_parent,      sizeof(int) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_level,       sizeof(int) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_is_leaf,     sizeof(int) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_first_child, sizeof(int) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_n_children,  sizeof(int) * nn_est));
    CUDA_CHECK(cudaMalloc(&hd.node_cell_size,   sizeof(float) * nn_est));

    // Level-order processing
    CUDA_CHECK(cudaMalloc(&hd.h_level_start, sizeof(int) * fmm::FMM_MAX_LEVELS));
    CUDA_CHECK(cudaMalloc(&hd.h_level_count, sizeof(int) * fmm::FMM_MAX_LEVELS));
    CUDA_CHECK(cudaMalloc(&hd.h_level_list,  sizeof(int) * nn_est));

    // Atomic counters
    CUDA_CHECK(cudaMalloc(&hd.d_num_nodes, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hd.d_max_depth, sizeof(int)));

    // Topology freezing support
    CUDA_CHECK(cudaMalloc(&hd.particle_to_leaf,      sizeof(int)   * np));

    // node_com_x/y/z are FMM-owned (far_real_t, distinct from the float cell size)
    CUDA_CHECK(cudaMalloc(&hd.node_com_x, sizeof(far_real_t) * nn));
    CUDA_CHECK(cudaMalloc(&hd.node_com_y, sizeof(far_real_t) * nn));
    CUDA_CHECK(cudaMalloc(&hd.node_com_z, sizeof(far_real_t) * nn));

    hd.allocated_max_particles = np;
    hd.allocated_max_nodes = nn;

    // ===== Device data: Tree structure =====
    CUDA_CHECK(cudaMalloc(&dd.nodes, sizeof(fmm::FMMOctreeNode) * nn));
    CUDA_CHECK(cudaMalloc(&dd.level_start, sizeof(int) * fmm::FMM_MAX_LEVELS));
    CUDA_CHECK(cudaMalloc(&dd.level_count, sizeof(int) * fmm::FMM_MAX_LEVELS));
    CUDA_CHECK(cudaMalloc(&dd.level_list, sizeof(int) * nn));

    // ===== Multipole and local expansions =====
    CUDA_CHECK(cudaMalloc(&dd.multipoles, sizeof(fmm::MultipoleExpansion) * nn));
    CUDA_CHECK(cudaMalloc(
        &dd.multipoles_dbl,
        sizeof(double) * fmm::FMM_ACTIVE_COEFFS * static_cast<size_t>(nn)));
    CUDA_CHECK(cudaMalloc(&dd.local_expansions, sizeof(fmm::LocalExpansion) * nn));

    // ===== Index mappings =====
    // The radix sort writes the production permutation directly.  Keep one
    // allocation and one owner instead of copying between duplicate arrays.
    dd.sorted_to_original = hd.sorted_indices;
    CUDA_CHECK(cudaMalloc(&dd.particle_to_leaf, sizeof(int) * np));
    CUDA_CHECK(cudaMalloc(&dd.children_indices, sizeof(int) * 8 * nn));
    // Phase 2: Unified node arrays (converted from FMMOctreeNode after build)
    CUDA_CHECK(cudaMalloc(&dd.unified_nodes, sizeof(body_fitted_fmm::UnifiedNode) * nn));
    CUDA_CHECK(cudaMalloc(&dd.children_compact, sizeof(int) * 8 * nn));
    dd.children_compact_count = 0;
    CUDA_CHECK(cudaMalloc(&dd.d_stats, sizeof(fmm::FMMStats)));
    CUDA_CHECK(cudaMemset(dd.d_stats, 0, sizeof(fmm::FMMStats)));

    // ===== Global projection scalars =====
    CUDA_CHECK(cudaMalloc(&dd.gp_scalars, BF_GP_TOTAL_NUM_SCALARS * sizeof(double)));

    // Verbose flag
    verbose_[gpu_id] = config.acceleration.unified_fmm.verbose;

    // The grouped M2L consumer uses the same precomposed 48-orientation
    // operator matrices as the legacy task consumer.  This is live runtime
    // state, not part of the removed flat queue pipeline.
    load_full_g_table(bfd);

    preload_post_scheduler_kernels();

    auto adaptive_options = adaptive_depth_options(
        config.acceleration.unified_fmm);
    adaptive_options.control_mode = adaptive_depth_control_mode_[gpu_id];
    if (adaptive_options.enabled && adaptive_startup_probe_[gpu_id].valid) {
        if (adaptive_options.auto_initial_depth)
            adaptive_options.initial_depth =
                adaptive_startup_probe_[gpu_id].initial_depth;
        adaptive_options.effective_max_depth =
            adaptive_startup_probe_[gpu_id].effective_max_depth;
        adaptive_options.startup_probe_complete = true;
    }
    persistent_tree_runtime_[gpu_id].host_context.use_presorted_morton = 0;
    body_fitted_fmm::initialize_persistent_tree_runtime(
        persistent_tree_runtime_[gpu_id], dd, grouped_arena_[gpu_id], np,
        configured_initial_tree_depth(config.acceleration.unified_fmm),
        config.acceleration.unified_fmm.tree_dimension,
        config.acceleration.unified_fmm.theta_far,
        static_cast<uint64_t>(config.acceleration.unified_fmm.direct_threshold_leaf),
        static_cast<uint64_t>(config.acceleration.unified_fmm.direct_threshold_internal),
        adaptive_options, nn);

    if (adaptive_options.enabled &&
        has_pending_adaptive_depth_checkpoint_[gpu_id]) {
        body_fitted_fmm::restore_adaptive_depth_checkpoint(
            persistent_tree_runtime_[gpu_id],
            pending_adaptive_depth_checkpoint_[gpu_id],
            persistent_tree_runtime_[gpu_id].graph_upload_stream);
        has_pending_adaptive_depth_checkpoint_[gpu_id] = false;
    }

    if (verbose_[gpu_id]) {
        std::cout << "[BF-FMM] Allocated memory for " << np << " particles, "
                  << nn << " max nodes" << std::endl;
    }
}

// =============================================================================
// free_memory
// =============================================================================

void UnifiedFMMDispatcher::free_memory(int gpu_id) {
    invalidate_workload_view(gpu_id);
    CUDA_CHECK(cudaSetDevice(gpu_id));
    if (host_data_[gpu_id].allocated_max_particles == 0 &&
        host_data_[gpu_id].allocated_max_nodes == 0) {
        body_fitted_fmm::shutdown_persistent_tree_runtime(
            persistent_tree_runtime_[gpu_id]);
        free_bffmm_telemetry_ring(gpu_id);
        return;
    }

    fmm::FMMHostData& hd = host_data_[gpu_id];
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    BodyFittedFMMDeviceData& bfd = bf_device_data_[gpu_id];

    // Stop the reusable scheduler runtime before releasing pointers captured
    // by its graph executables. Dense maps are one contiguous allocation.
    body_fitted_fmm::shutdown_persistent_tree_runtime(
        persistent_tree_runtime_[gpu_id]);
    // Telemetry owns CUDA events/buffers too; release it only after the
    // scheduler has stopped so no context-wide teardown operation can block
    // behind a resident polling graph.
    free_bffmm_telemetry_ring(gpu_id);
    for (int level = 0; level < fmm::FMM_MAX_DENSE_LEVELS; ++level) {
        dd.level_dense_map[level] = nullptr;
        dd.level_dense_capacity[level] = 0;
    }


    // Free full G-table
    if (bfd.d_G_full) { cudaFree(bfd.d_G_full); bfd.d_G_full = nullptr; bfd.d_G_full_bytes = 0; }

    grouped_arena_[gpu_id].free();

    // Free device data buffers
    cudaFree(dd.nodes);
    cudaFree(dd.level_start);
    cudaFree(dd.level_count);
    cudaFree(dd.level_list);
    cudaFree(dd.multipoles);
    cudaFree(dd.multipoles_dbl);
    cudaFree(dd.local_expansions);
    // dd.sorted_to_original aliases hd.sorted_indices and is freed below.
    cudaFree(dd.particle_to_leaf);
    cudaFree(dd.children_indices);
    cudaFree(dd.unified_nodes);
    cudaFree(dd.children_compact);
    cudaFree(dd.d_stats);

    // Free dense level maps
    free_dense_level_maps(dd);

    cudaFree(dd.gp_scalars);

    // Free host data — node center-of-mass
    cudaFree(hd.node_com_x);
    cudaFree(hd.node_com_y);
    cudaFree(hd.node_com_z);

    // Free tree topology arrays (owned directly by FMMHostData)
    cudaFree(hd.morton_keys);
    cudaFree(hd.sorted_indices);
    cudaFree(hd.node_range);
    cudaFree(hd.node_children);
    cudaFree(hd.node_parent);
    cudaFree(hd.node_level);
    cudaFree(hd.node_is_leaf);
    cudaFree(hd.node_first_child);
    cudaFree(hd.node_n_children);
    cudaFree(hd.node_cell_size);
    cudaFree(hd.h_level_start);
    cudaFree(hd.h_level_count);
    cudaFree(hd.h_level_list);
    cudaFree(hd.d_num_nodes);
    cudaFree(hd.d_max_depth);
    cudaFree(hd.particle_to_leaf);

    std::memset(&hd, 0, sizeof(fmm::FMMHostData));
    std::memset(&dd, 0, sizeof(fmm::FMMDeviceData));
    std::memset(&bfd, 0, sizeof(BodyFittedFMMDeviceData));
}

// =============================================================================
// initialize
// =============================================================================

void UnifiedFMMDispatcher::initialize(const ::cudadem::SimConfig& config, int max_particles) {
    int gpu_id = current_gpu();
    const auto& fmm_config = config.acceleration.unified_fmm;

    if (fmm_config.adaptive_depth.enabled) {
        throw std::invalid_argument(
            "adaptive BFFMM initialization requires particle positions; "
            "use the lazy compute/compute_outer path so sparse startup probing "
            "runs before heavy allocation");
    }

    if (fmm_config.tree_dimension != 2 &&
        fmm_config.tree_dimension != 3) {
        throw std::invalid_argument(
            "Body-fitted FMM tree_dimension must be 2 (X-Y quadtree) or 3 (octree)");
    }

    if (initialized_[gpu_id]) {
        shutdown();
    }

    verbose_[gpu_id] = fmm_config.verbose;
    max_particles_[gpu_id] = max_particles;

    if (!fmm_config.adaptive_depth.enabled && fmm_config.tree_depth < 2) {
        throw std::invalid_argument(
            "Body-fitted FMM requires unified_fmm.tree_depth >= 2");
    }
    const int max_depth = 20;
    if (!fmm_config.adaptive_depth.enabled && fmm_config.tree_depth > max_depth) {
        throw std::invalid_argument(
            "Body-fitted FMM tree_depth exceeds the selected Morton encoding limit");
    }
    const int allocation_depth = fmm_config.adaptive_depth.enabled
        ? fmm_config.adaptive_depth.max_depth : fmm_config.tree_depth;
    const int max_nodes = body_fitted_fmm::fixed_depth_node_upper_bound(
        static_cast<int>(std::ceil(
            max_particles * fmm_config.particle_overallocate_factor)),
        allocation_depth, fmm_config.tree_dimension);

    allocate_memory(gpu_id, max_particles, max_nodes, config);

    initialized_[gpu_id] = true;
    initialized_tree_type_[gpu_id] = fmm_config.tree_dimension;

    if (verbose_[gpu_id]) {
        std::cout << "[BF-FMM] Initialized GPU " << gpu_id << " for " << max_particles
                  << " particles, " << max_nodes << " max nodes" << std::endl;
    }
}

void UnifiedFMMDispatcher::initialize_adaptive(
    const ::cudadem::SimConfig& config,
    const ::cudadem::SimulationState& state,
    int particle_count,
    cudaStream_t stream)
{
    const int gpu_id = current_gpu();
    const auto& fmm = config.acceleration.unified_fmm;
    if (!fmm.adaptive_depth.enabled) {
        initialize(config, particle_count);
        return;
    }
    if (initialized_[gpu_id]) {
        // Capacity service replaces the persistent/FMM arenas, but the
        // multi-GPU timing mailbox remains valid and can still be polled for
        // completed pre-service epochs.  Final shutdown owns its teardown.
        // A compatible particle-capacity reallocation must also retain the
        // controller and diagnostic sequence; otherwise host-backlogged rows
        // would be followed by a replacement ring restarting at sequence 1.
        if (!has_pending_adaptive_depth_checkpoint_[gpu_id] &&
            persistent_depth_configuration_matches(
                persistent_tree_runtime_[gpu_id], fmm)) {
            pending_adaptive_depth_checkpoint_[gpu_id] =
                body_fitted_fmm::snapshot_adaptive_depth_checkpoint(
                    persistent_tree_runtime_[gpu_id], stream);
            has_pending_adaptive_depth_checkpoint_[gpu_id] = true;
        }
        drain_live_adaptive_depth_diagnostics_to_backlog(gpu_id, stream);
        free_memory(gpu_id);
    }
    verbose_[gpu_id] = fmm.verbose;
    max_particles_[gpu_id] = particle_count;
    const int particle_capacity = static_cast<int>(std::ceil(
        particle_count * fmm.particle_overallocate_factor));

    if (!adaptive_startup_probe_[gpu_id].valid ||
        adaptive_capacity_override_[gpu_id] <= 0) {
        adaptive_startup_probe_[gpu_id] = run_adaptive_startup_probe(
            state, fmm, particle_count, particle_capacity, stream);
    }
    int node_capacity = adaptive_startup_probe_[gpu_id].node_capacity;
    if (adaptive_capacity_override_[gpu_id] > 0)
        node_capacity = std::max(node_capacity,
                                 adaptive_capacity_override_[gpu_id]);
    const int strict = fixed_depth_node_upper_bound(
        particle_capacity, fmm.adaptive_depth.max_depth,
        fmm.tree_dimension);
    node_capacity = std::min(node_capacity, strict);
    allocate_memory(gpu_id, particle_count, node_capacity, config);
    if (adaptive_depth_control_mode_[gpu_id] ==
            AdaptiveFMMDepthControlMode::ExternalDistributed &&
        has_staged_external_depth_command_[gpu_id]) {
        body_fitted_fmm::set_persistent_external_depth_command(
            persistent_tree_runtime_[gpu_id],
            staged_external_depth_command_[gpu_id], stream);
    }
    initialized_[gpu_id] = true;
    initialized_tree_type_[gpu_id] = fmm.tree_dimension;
    adaptive_capacity_override_[gpu_id] = 0;
    if (adaptive_depth_control_mode_[gpu_id] ==
        AdaptiveFMMDepthControlMode::Autonomous)
        ensure_adaptive_depth_diagnostic_staging(gpu_id);

    if (verbose_[gpu_id]) {
        std::cout << "[BF-FMM adaptive] startup probe selected D"
                  << fmm.tree_dimension << " depth "
                  << adaptive_startup_probe_[gpu_id].initial_depth
                  << ", shared node capacity " << node_capacity << std::endl;
    }
}

// =============================================================================
// shutdown
// =============================================================================

void UnifiedFMMDispatcher::shutdown() {
    int gpu_id = current_gpu();
    release_aligned_clump_pool(gpu_id);
    shear_reclaimable_bytes_[gpu_id] = 0;
    if (!initialized_[gpu_id]) {
        if (!distributed_fmm_[gpu_id].evaluating) shear_census_workspace_[gpu_id].reset();
        release_adaptive_depth_diagnostic_drain(gpu_id);
        if (!distributed_fmm_[gpu_id].evaluating)
            reset_distributed_fmm_state(gpu_id);
        return;
    }

    if (persistent_tree_runtime_[gpu_id].adaptive_options.enabled &&
        adaptive_depth_control_mode_[gpu_id] ==
            AdaptiveFMMDepthControlMode::Autonomous) {
        CUDA_CHECK(cudaDeviceSynchronize());
        drain_live_adaptive_depth_diagnostics_to_backlog(
            gpu_id, persistent_tree_runtime_[gpu_id].graph_upload_stream);
    }
    free_memory(gpu_id);
    if (!distributed_fmm_[gpu_id].evaluating) shear_census_workspace_[gpu_id].reset();
    // Unlike free_memory(), an explicit per-GPU shutdown ends the mailbox's
    // lifetime.  This keeps repeated validation/setup cycles bounded while
    // capacity reinitialization above retains outstanding completed samples.
    release_adaptive_depth_diagnostic_drain(gpu_id);

    auto& shear_arena = shear_source_arena_[gpu_id];
    if (shear_arena.pos_x) cudaFree(shear_arena.pos_x);
    if (shear_arena.pos_y) cudaFree(shear_arena.pos_y);
    if (shear_arena.pos_z) cudaFree(shear_arena.pos_z);
    if (shear_arena.mass) cudaFree(shear_arena.mass);
    if (shear_arena.clump_ids) cudaFree(shear_arena.clump_ids);
    if (shear_arena.before_x) cudaFree(shear_arena.before_x);
    if (shear_arena.before_y) cudaFree(shear_arena.before_y);
    if (shear_arena.before_z) cudaFree(shear_arena.before_z);
    if (shear_arena.diagnostic) cudaFree(shear_arena.diagnostic);
    shear_arena = ShearFMMSourceArena{};

    if (verbose_[gpu_id]) {
        std::cout << "[BF-FMM] GPU " << gpu_id << " shutdown complete" << std::endl;
    }

    initialized_[gpu_id] = false;
    initialized_tree_type_[gpu_id] = 0;
    max_particles_[gpu_id] = 0;
    verbose_[gpu_id] = false;
    if (!distributed_fmm_[gpu_id].evaluating)
        reset_distributed_fmm_state(gpu_id);
}

void UnifiedFMMDispatcher::shutdown_all() {
    for (int g = 0; g < initialized_.size(); ++g) {
        release_aligned_clump_pool(g);
        if (initialized_[g]) {
            CUDA_CHECK(cudaSetDevice(g));
            if (persistent_tree_runtime_[g].adaptive_options.enabled &&
                adaptive_depth_control_mode_[g] ==
                    AdaptiveFMMDepthControlMode::Autonomous) {
                CUDA_CHECK(cudaDeviceSynchronize());
                drain_live_adaptive_depth_diagnostics_to_backlog(
                    g, persistent_tree_runtime_[g].graph_upload_stream);
            }
            free_memory(g);
            initialized_[g] = false;
            initialized_tree_type_[g] = 0;
            max_particles_[g] = 0;
            verbose_[g] = false;
        }
        release_adaptive_depth_diagnostic_drain(g);
        auto& shear_arena = shear_source_arena_[g];
        if (shear_arena.pos_x || shear_arena.pos_y || shear_arena.pos_z ||
            shear_arena.mass || shear_arena.before_x || shear_arena.before_y ||
            shear_arena.before_z || shear_arena.diagnostic)
            CUDA_CHECK(cudaSetDevice(g));
        if (shear_arena.pos_x) cudaFree(shear_arena.pos_x);
        if (shear_arena.pos_y) cudaFree(shear_arena.pos_y);
        if (shear_arena.pos_z) cudaFree(shear_arena.pos_z);
        if (shear_arena.mass) cudaFree(shear_arena.mass);
    if (shear_arena.clump_ids) cudaFree(shear_arena.clump_ids);
        if (shear_arena.before_x) cudaFree(shear_arena.before_x);
        if (shear_arena.before_y) cudaFree(shear_arena.before_y);
        if (shear_arena.before_z) cudaFree(shear_arena.before_z);
        if (shear_arena.diagnostic) cudaFree(shear_arena.diagnostic);
        shear_arena = ShearFMMSourceArena{};
        reset_distributed_fmm_state(g);
    }
}

// =============================================================================
// build_tree
// =============================================================================

void UnifiedFMMDispatcher::build_tree(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    cudaStream_t stream)
{
    const int gpu_id = current_gpu();
    invalidate_workload_view(gpu_id);

    const int particle_count = multigpu_owned_only_[gpu_id] && shear_target_count_[gpu_id] == 0
        ? state.particles.num_local : state.particles.num_particles;
    if (!initialized_[gpu_id]) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            initialize_adaptive(config, state, particle_count, stream);
        else
            initialize(config, particle_count);
    }
    if (particle_count > host_data_[gpu_id].allocated_max_particles) {
        throw std::overflow_error(
            "Body-fitted FMM particle capacity exceeded; persistent execution "
            "does not resize or fall back to a host build");
    }

    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    dd.num_particles = particle_count;
    dd.sorted_to_original = host_data_[gpu_id].sorted_indices;

    const bool shear_active = shear_target_count_[gpu_id] > 0;
    const bool common = distributed_fmm_[gpu_id].evaluating;
    const auto grid = distributed_fmm_[gpu_id].grid;
    const double4 fixed = shear_active ? shear_fixed_root_[gpu_id] : make_double4(grid.x, grid.y, grid.z, grid.size);

    body_fitted_fmm::submit_persistent_tree_runtime(
        persistent_tree_runtime_[gpu_id],
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        particle_count,
        shear_active ? shear_target_count_[gpu_id] : particle_count,
        static_cast<int>(state.current_step),
        config.acceleration.unified_fmm.tree_dimension,
        shear_active || common, fixed.x, fixed.y, fixed.z, fixed.w,
        shear_active ? shear_cutoff_[gpu_id] : 0.0, stream,
        state.current_time, common,
        distributed_fmm_[gpu_id].evaluating && distributed_fmm_[gpu_id].ranks > 1 &&
        !distributed_fmm_[gpu_id].local_only);

    // The stream wait above is satisfied by the device completion epoch. No
    // topology/count/domain metadata is copied to or reduced on the host.
}


static void run_m2l_phase(int gpu_id, cudaStream_t stream)
{
#ifdef GRANIT_FMM_M2L_DIAGNOSTICS
    body_fitted_fmm::m2l_census_begin(gpu_id,stream);
#endif
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    BodyFittedFMMDeviceData& bfd = bf_device_data_[gpu_id];
    body_fitted_fmm::GroupedInteractionArena& arena = grouped_arena_[gpu_id];

    // Focused migration adapter: consume GPU-published counts directly.  The
    // fixed grid is independent of the runtime count and every warp drains a
    // grid-stride sequence, so there is no D2H metadata path here.
    const bool time_m2l = getenv("BF_FMM_VALIDATE_M2L_TIMING") != nullptr;
    cudaEvent_t timing_start = nullptr, timing_end = nullptr;
    if (time_m2l) {
        CUDA_CHECK(cudaEventCreate(&timing_start));
        CUDA_CHECK(cudaEventCreate(&timing_end));
        CUDA_CHECK(cudaEventRecord(timing_start, stream));
    }
    auto& distributed = distributed_fmm_[gpu_id];
    const auto* moments=distributed.evaluating && distributed.has_moment_overlay
        ?distributed.moment_overlay.data:dd.multipoles;
    body_fitted_fmm::run_m2l_grouped_targets(
        arena, &persistent_tree_runtime_[gpu_id].device_context->metadata,
        dd.unified_nodes, dd.children_compact, max_nodes_capacity_[gpu_id], moments,
        bfd.d_G_full, dd.local_expansions, stream);
    if (distributed.evaluating) body_fitted_fmm::run_m2l_remote_targets(
        distributed.remote_lists.far_view(),
        &persistent_tree_runtime_[gpu_id].device_context->metadata,
        dd.unified_nodes,bfd.d_G_full,dd.local_expansions,stream);
    if (time_m2l) {
        CUDA_CHECK(cudaEventRecord(timing_end, stream));
        CUDA_CHECK(cudaEventSynchronize(timing_end));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timing_start, timing_end));
        last_validation_m2l_ms_[gpu_id] = static_cast<double>(elapsed_ms);
        CUDA_CHECK(cudaEventDestroy(timing_start));
        CUDA_CHECK(cudaEventDestroy(timing_end));
    } else {
        last_validation_m2l_ms_[gpu_id] = 0.0;
    }
#ifdef GRANIT_FMM_M2L_DIAGNOSTICS
    body_fitted_fmm::m2l_census_end(gpu_id,
        &persistent_tree_runtime_[gpu_id].device_context->metadata,stream);
#endif
}


#include "aligned_clump_fmm.cuh"

// =============================================================================
// Helper: Run P2P phase (inner — writes to force_*)
// =============================================================================

double sph_compact_baseline_length(const SimConfig& config, cudaStream_t stream) {
    int gpu = 0;
    CUDA_CHECK(cudaGetDevice(&gpu));
    body_fitted_fmm::PersistentTreeMetadata metadata;
    CUDA_CHECK(cudaMemcpyAsync(&metadata,
        &persistent_tree_runtime_[gpu].device_context->metadata,
        sizeof(metadata), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Every generated far route separates cell AABBs by at least a leaf width.
    // The compact baseline is therefore exactly Newtonian for every M2L pair.
    return std::min(static_cast<double>(config.sph.hsml_min),
                    0.25 * std::ldexp(metadata.root_cell_size, -metadata.evaluation_depth));
}

static void run_p2p_inner(int gpu_id, ::cudadem::SimulationState& state,
                          const ::cudadem::SimConfig& config, cudaStream_t stream,
                          const int* gravity_clump_ids)
{
    fmm::FMMDeviceData& dd = device_data_[gpu_id];

    double G = config.acceleration.unified_fmm.gravitational_constant;
    double softening = config.physics == SimulationPhysics::SPH
        ? -sph_compact_baseline_length(config, stream) :
        config.particle.radius_min * config.acceleration.unified_fmm.softening_factor;

    const auto& distributed = distributed_fmm_[gpu_id];
    body_fitted_fmm::launch_p2p_grouped_leaves<false>(
        dd, grouped_arena_[gpu_id],
        &persistent_tree_runtime_[gpu_id].device_context->metadata,
        max_nodes_capacity_[gpu_id],
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        state.particles.mass, state.particles.force_x, state.particles.force_y,
        state.particles.force_z, G, softening, stream, persistent_tree_runtime_[gpu_id].device_context,
        body_fitted_fmm::grouped_p2p_leaf_only_allowed(
            persistent_tree_runtime_[gpu_id].host_context.command.disable_direct_fallback,
            persistent_tree_runtime_[gpu_id].host_context.command.cutoff_radius),
        gravity_clump_ids, shear_target_count_[gpu_id] > 0 ?
            distributed.clump_exclusion_diameter*distributed.clump_exclusion_diameter : 0.0);
    if (distributed.evaluating) body_fitted_fmm::launch_p2p_remote_leaves<false>(
        distributed.remote_lists.near_view(),dd,
        &persistent_tree_runtime_[gpu_id].device_context->metadata,
        state.particles.pos_x,state.particles.pos_y,state.particles.pos_z,state.particles.mass,
        state.particles.force_x,state.particles.force_y,state.particles.force_z,
        G,softening,stream,gravity_clump_ids, shear_target_count_[gpu_id] > 0 ?
            distributed.clump_exclusion_diameter*distributed.clump_exclusion_diameter : 0.0);

}

// =============================================================================
// Helper: Run P2P phase (outer — writes to gravity_force_*)
// =============================================================================

static void run_p2p_outer(int gpu_id, ::cudadem::SimulationState& state,
                           const ::cudadem::SimConfig& config, cudaStream_t stream,
                          const int* gravity_clump_ids)
{
    fmm::FMMDeviceData& dd = device_data_[gpu_id];

    double G = config.acceleration.unified_fmm.gravitational_constant;
    double softening = config.physics == SimulationPhysics::SPH
        ? -sph_compact_baseline_length(config, stream) :
        config.particle.radius_min * config.acceleration.unified_fmm.softening_factor;

    const auto& distributed = distributed_fmm_[gpu_id];
    body_fitted_fmm::launch_p2p_grouped_leaves<true>(
        dd, grouped_arena_[gpu_id],
        &persistent_tree_runtime_[gpu_id].device_context->metadata,
        max_nodes_capacity_[gpu_id],
        state.particles.pos_x,
        state.particles.pos_y,
        state.particles.pos_z,
        state.particles.mass, state.particles.gravity_force_x,
        state.particles.gravity_force_y, state.particles.gravity_force_z,
        G, softening, stream, persistent_tree_runtime_[gpu_id].device_context,
        body_fitted_fmm::grouped_p2p_leaf_only_allowed(
            persistent_tree_runtime_[gpu_id].host_context.command.disable_direct_fallback,
            persistent_tree_runtime_[gpu_id].host_context.command.cutoff_radius),
        gravity_clump_ids, shear_target_count_[gpu_id] > 0 ?
            distributed.clump_exclusion_diameter*distributed.clump_exclusion_diameter : 0.0);
    if (distributed.evaluating) body_fitted_fmm::launch_p2p_remote_leaves<true>(
        distributed.remote_lists.near_view(),dd,
        &persistent_tree_runtime_[gpu_id].device_context->metadata,
        state.particles.pos_x,state.particles.pos_y,state.particles.pos_z,state.particles.mass,
        state.particles.gravity_force_x,state.particles.gravity_force_y,state.particles.gravity_force_z,
        G,softening,stream,gravity_clump_ids, shear_target_count_[gpu_id] > 0 ?
            distributed.clump_exclusion_diameter*distributed.clump_exclusion_diameter : 0.0);

}


// =============================================================================
// compute (inner — writes to force_*)
// =============================================================================


void UnifiedFMMDispatcher::compute(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    const ::cudadem::NeighborData& neighbor_data,
    const ::cudadem::PeriodicBoundaryConfig& pbc,
    double evaluation_time,
    cudaStream_t stream,
    bool synchronize_at_end,
    const int* gravity_clump_ids)
{
    (void)neighbor_data;
    if (gravity_clump_ids) {
        const bool controlled_shear = distributed_fmm_[current_gpu()].evaluating &&
            (config.shear_box.enabled || shear_target_count_[current_gpu()] > 0);
        if (config.physics != SimulationPhysics::DEM ||
            ((!controlled_shear) && (pbc.periodic_x || pbc.periodic_y || pbc.periodic_z)) ||
            config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument("clump gravity requires an explicitly controlled DEM FMM depth and supported boundaries");
    }
    const ShearGravityImages shear_images =
        make_shear_gravity_images(config, pbc, evaluation_time);


    if (shear_images.enabled) {
        LocalShearHaloScope halo_scope(current_gpu(), config, pbc, evaluation_time);
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument("adaptive shear gravity must be staged by the common multi-GPU depth controller");
        const int central_count = multigpu_owned_only_[current_gpu()]
            ? state.particles.num_local : state.particles.num_particles;
        const int64_t total64 = static_cast<int64_t>(central_count) *
                                (2*shear_images.layers_x+1);
        if (total64 <= 0 || total64 > INT_MAX / 2)
            throw std::overflow_error(
                "finite shear-image BFFMM source count exceeds edge/index capacity");
        const int total = static_cast<int>(total64);
        if (int64_t(central_count)*shear_images.image_count>INT_MAX/2)
            throw std::overflow_error("logical shear image population exceeds indexing capacity");
        if (std::getenv("GRANIT_SHEAR_SOURCE_DIAGNOSTICS"))
            std::printf("[Shear canonical] gpu=%d physical=%d stored=%d logical=%lld columns=%d rows=%d\n",
                current_gpu(),central_count,total,static_cast<long long>(central_count)*shear_images.image_count,
                2*shear_images.layers_x+1,2*shear_images.layers_y+1);

        SimConfig staged_config = config;
        staged_config.shear_box.enabled = false;
        // The configured dimension is immutable for a gravity epoch; shear
        // replicas do not trigger a host AABB/D2H selector.
        staged_config.acceleration.unified_fmm.tree_dimension =
            config.acceleration.unified_fmm.tree_dimension;
        staged_config.acceleration.unified_fmm.tree_depth = shear_effective_depth(
            config.acceleration.unified_fmm, shear_images, pbc);
        if (staged_config.acceleration.unified_fmm.tree_depth > 20)
            throw std::invalid_argument(
                "finite shear-image BFFMM effective depth exceeds the float-center limit 20");
        staged_config.acceleration.unified_fmm.enable_global_projection = false;
        shear_fixed_root_[current_gpu()] = shear_static_root(shear_images, pbc, config);
        shear_node_bound_[current_gpu()] = shear_occupied_node_bound(state, config,
            central_count, total, staged_config.acceleration.unified_fmm.tree_depth, shear_images,
            shear_fixed_root_[current_gpu()], stream);
        ensure_shear_fmm_capacity(staged_config, total, current_gpu());
        const int gpu = current_gpu();
        auto& arena = shear_source_arena_[gpu];
        if (gravity_clump_ids && !arena.clump_ids)
            CUDA_CHECK(cudaMalloc(&arena.clump_ids, sizeof(int)*size_t(arena.capacity)));
        const size_t target_bytes = static_cast<size_t>(central_count) *
                                    sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(arena.before_x, state.particles.force_x,
                                   target_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(arena.before_y, state.particles.force_y,
                                   target_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(arena.before_z, state.particles.force_z,
                                   target_bytes, cudaMemcpyDeviceToDevice, stream));
        const int block = 256;
        materialize_shear_gravity_sources_kernel<<<(total + block - 1) / block,
            block, 0, stream>>>(
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.mass, central_count, shear_images,
            arena.pos_x, arena.pos_y, arena.pos_z, arena.mass,
            gravity_clump_ids, gravity_clump_ids ? arena.clump_ids : nullptr);
        CUDA_CHECK(cudaGetLastError());

        SimulationState staged = state;
        staged.particles.pos_x = arena.pos_x;
        staged.particles.pos_y = arena.pos_y;
        staged.particles.pos_z = arena.pos_z;
        staged.particles.mass = arena.mass;
        staged.particles.num_particles = total;
        staged.particles.num_local = central_count;
        staged.particles.num_ghost = total - central_count;
        staged.particles.capacity = total;
        shear_target_count_[gpu] = central_count;
        shear_cutoff_[gpu] = shear_images.cutoff_radius;
        shear_fixed_root_[gpu] = shear_static_root(shear_images, pbc, config);
        preflight_shear_float_centers(
            shear_fixed_root_[gpu],
            staged_config.acceleration.unified_fmm.tree_depth);
        const int* staged_ids = gravity_clump_ids ? arena.clump_ids : nullptr;
        distributed_fmm_[gpu].gravity_clump_ids = staged_ids;
        try {
            compute(staged, staged_config, neighbor_data, pbc, evaluation_time,
                    stream, synchronize_at_end, staged_ids);
        } catch (...) {
            distributed_fmm_[gpu].gravity_clump_ids = gravity_clump_ids;
            shear_node_bound_[gpu] = 0;
            persistent_tree_runtime_[gpu].pending_presorted = body_fitted_fmm::PresortedMortonInput{};
            shear_target_count_[gpu] = 0;
            shear_cutoff_[gpu] = 0.0;
            shear_fixed_root_[gpu] = make_double4(0.0, 0.0, 0.0, 0.0);
            throw;
        }
        distributed_fmm_[gpu].gravity_clump_ids = gravity_clump_ids;
        if (config.io.output_gravity_force) {
            cache_shear_gravity_force_kernel<<<(central_count + block - 1) / block,
                block, 0, stream>>>(
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                arena.before_x, arena.before_y, arena.before_z,
                state.particles.gravity_force_x, state.particles.gravity_force_y,
                state.particles.gravity_force_z, central_count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (!distributed_fmm_[gpu].evaluating || distributed_fmm_[gpu].local_only) launch_shear_force_balance<false>(arena, state, central_count,
            config.shear_box.force_balance_warn_threshold, stream);
        shear_node_bound_[gpu] = 0;
        shear_target_count_[gpu] = 0;
        shear_cutoff_[gpu] = 0.0;
        shear_fixed_root_[gpu] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    int gpu_id = current_gpu();

    const int requested_particle_count = multigpu_owned_only_[gpu_id] && shear_target_count_[gpu_id] == 0
        ? state.particles.num_local : state.particles.num_particles;
    if (!initialized_[gpu_id] ||
        persistent_tree_runtime_[gpu_id].host_context.use_presorted_morton != 0 ||
        initialized_tree_type_[gpu_id] !=
            config.acceleration.unified_fmm.tree_dimension ||
        !persistent_depth_configuration_matches(
            persistent_tree_runtime_[gpu_id],
            config.acceleration.unified_fmm)) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            initialize_adaptive(config, state, requested_particle_count,
                                stream);
        else
            initialize(config, requested_particle_count);
    }

    int N = requested_particle_count;
    if (N > host_data_[gpu_id].allocated_max_particles) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            initialize_adaptive(config, state, N, stream);
        else
            initialize(config, N);
    }

    if (config.physics == SimulationPhysics::SPH && config.acceleration.unified_fmm.adaptive_depth.enabled) {
        limit_persistent_sph_depth(persistent_tree_runtime_[gpu_id],
            sph_physical_depth_ceiling(state, config, requested_particle_count, stream), stream);
    }

    fmm::FMMHostData& hd = host_data_[gpu_id];
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    BodyFittedFMMDeviceData& bfd = bf_device_data_[gpu_id];

    const ::cudadem::UnifiedFMMConfig& fmm_config = config.acceleration.unified_fmm;
    bool verbose = fmm_config.verbose;
    dd.verbose = verbose;

    // Every actual gravity evaluation rebuilds the full Karras/stencil tree.
    {
        const int tree_task_sample = begin_bffmm_sample(
            gpu_id, BFFMMActivePhase::TreeTaskRebuild, state.current_step,
            0.0, RCBWorkPhase::FmmNode, false, stream);

        // --- Phase 0: Tree Build ---
        BF_FMM_NVTX_RANGE("BF-FMM: Tree Build");
        build_tree(state, config, stream);
        if(shear_target_count_[gpu_id]>0 && std::getenv("GRANIT_SHEAR_CENSUS_VALIDATE")) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto metadata=snapshot_persistent_tree_metadata(persistent_tree_runtime_[gpu_id]);
            if(metadata.error || uint64_t(metadata.num_nodes)!=shear_expected_nodes_[gpu_id])
                throw std::runtime_error("occupied-prefix census differs from actual Karras tree");
            std::printf("[Shear census validation] gpu=%d effective=%d nodes=%d PASS\n",
                gpu_id,config.acceleration.unified_fmm.tree_depth,metadata.num_nodes);
        }
        mark_distributed_fmm_phase(gpu_id,1,stream);
        BF_FMM_NVTX_END();
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
            fprintf(stderr, "[BF-FMM validate] wait: tree+stencil completion\n");
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "[BF-FMM validate] done: tree+stencil completion\n");
            const auto debug_metadata =
                body_fitted_fmm::snapshot_persistent_tree_metadata(
                    persistent_tree_runtime_[gpu_id]);
            std::vector<body_fitted_fmm::UnifiedNode> debug_nodes(
                debug_metadata.num_nodes);
            CUDA_CHECK(cudaMemcpyAsync(
                debug_nodes.data(), dd.unified_nodes,
                debug_nodes.size() * sizeof(debug_nodes[0]),
                cudaMemcpyDeviceToHost,
                persistent_tree_runtime_[gpu_id].graph_upload_stream));
            CUDA_CHECK(cudaStreamSynchronize(
                persistent_tree_runtime_[gpu_id].graph_upload_stream));
            int debug_leaves = 0, debug_invalid_ranges = 0;
            uint64_t debug_leaf_particles = 0;
            uint32_t debug_max_leaf = 0;
            for (const auto& node : debug_nodes) {
                if (!(node.flags & body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
                    static_cast<uint64_t>(node.begin) + node.count >
                        static_cast<uint64_t>(N)) {
                    ++debug_invalid_ranges;
                }
                if (node.flags & body_fitted_fmm::UNIFIED_NODE_FLAG_IS_LEAF) {
                    ++debug_leaves;
                    debug_leaf_particles += node.count;
                    debug_max_leaf = std::max(debug_max_leaf, node.count);
                }
            }
            fprintf(stderr,
                    "[BF-FMM validate] nodes=%d leaves=%d leaf_particles=%llu "
                    "max_leaf=%u invalid_ranges=%d tasks=[%d,%d,%d,%d]\n",
                    debug_metadata.num_nodes, debug_leaves,
                    static_cast<unsigned long long>(debug_leaf_particles),
                    debug_max_leaf, debug_invalid_ranges,
                    debug_metadata.task_counts[0], debug_metadata.task_counts[1],
                    debug_metadata.task_counts[2], debug_metadata.task_counts[3]);
        }
        // --- Phase 2: Publish the completed grouped workload ---
        BF_FMM_NVTX_RANGE("BF-FMM: Grouped Work");
        publish_rebuilt_workload_view(gpu_id, N);
        end_bffmm_sample(gpu_id, tree_task_sample, stream,
                         static_cast<double>(max_nodes_capacity_[gpu_id]));
        BF_FMM_NVTX_END();

    }

    // =====================================================================
    // Phases 3-9: Far-field pipeline after the mandatory rebuild.
    // =====================================================================


    // --- Phase 3: Zero local expansions before the far-field passes. ---

    // The host view is the captured capacity; the reset kernel reads the
    // completed tree's actual count and checks it against this allocation.
    if (dd.num_nodes > max_nodes_capacity_[gpu_id]) {
        throw std::runtime_error(
            "BF-FMM node count exceeds local-expansion capacity before the far-field pass");
    }

    body_fitted_fmm::zero_active_local_expansions(
        dd, &persistent_tree_runtime_[gpu_id].device_context->metadata, stream);

    // --- Phase 4: P2M + M2M ---
    const int far_sample = begin_bffmm_sample(
        gpu_id, BFFMMActivePhase::M2LFarField, state.current_step,
        (double)dd.num_nodes,
        RCBWorkPhase::FmmNode, false, stream);
    BF_FMM_NVTX_RANGE("BF-FMM: P2M+M2M");
    body_fitted_fmm::fmm_compute_all_multipoles_unified(state, config, hd, dd, stream);
    BF_FMM_NVTX_END();
    if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
        fprintf(stderr, "[BF-FMM validate] wait: P2M+M2M\n");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[BF-FMM validate] done: P2M+M2M\n");
    }


    mark_distributed_fmm_phase(gpu_id,2,stream);
    if (distributed_fmm_[gpu_id].evaluating) {
        exchange_distributed_fmm_sources(gpu_id, &dd, &persistent_tree_runtime_[gpu_id], state, config, stream);
        publish_rebuilt_workload_view(gpu_id, N);
    }
    mark_distributed_fmm_phase(gpu_id,3,stream);

    // --- Phase 5: target-owned grouped M2L ---
    BF_FMM_NVTX_RANGE("BF-FMM: M2L");
    run_m2l_phase(gpu_id, stream);
    BF_FMM_NVTX_END();
    if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
        fprintf(stderr, "[BF-FMM validate] wait: M2L\n");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[BF-FMM validate] done: M2L\n");
    }

    BF_FMM_NVTX_RANGE("BF-FMM: L2L");
    body_fitted_fmm::run_l2l_pass_unified(
        dd, &persistent_tree_runtime_[gpu_id].device_context->metadata.root_cell_size,
        stream);
    BF_FMM_NVTX_END();
    if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
        fprintf(stderr, "[BF-FMM validate] wait: L2L\n");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[BF-FMM validate] done: L2L\n");
    }
    end_bffmm_sample(gpu_id, far_sample, stream);

    // --- Phase 7: L2P ---
    BF_FMM_NVTX_RANGE("BF-FMM: L2P");
    {
        const int l2p_sample = begin_bffmm_sample(
            gpu_id, BFFMMActivePhase::ParticleL2P, state.current_step,
            (double)N, RCBWorkPhase::FmmParticle, false, stream);
        body_fitted_fmm::run_l2p_evaluation_unified(
            dd,
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.mass,
            state.particles.force_x, state.particles.force_y, state.particles.force_z,
            dd.particle_to_leaf,
            shear_target_count_[gpu_id] > 0 ? shear_target_count_[gpu_id] : N,
            FAR_REAL(config.acceleration.unified_fmm.gravitational_constant),
            &persistent_tree_runtime_[gpu_id].device_context->metadata.root_cell_size,
            stream);
        end_bffmm_sample(gpu_id, l2p_sample, stream);
    }
    BF_FMM_NVTX_END();
    if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
        fprintf(stderr, "[BF-FMM validate] wait: L2P\n");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[BF-FMM validate] done: L2P\n");
    }

    mark_distributed_fmm_phase(gpu_id,4,stream);
    // --- Phase 8: P2P ---
    BF_FMM_NVTX_RANGE("BF-FMM: P2P");
    const int p2p_sample = begin_bffmm_sample(
        gpu_id, BFFMMActivePhase::P2P, state.current_step,
        (double)N,
        RCBWorkPhase::FmmLeaf, false, stream);
    run_p2p_inner(gpu_id, state, config, stream, gravity_clump_ids);
    mark_distributed_fmm_phase(gpu_id,5,stream);
    end_bffmm_sample(gpu_id, p2p_sample, stream);
    BF_FMM_NVTX_END();
    if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
        fprintf(stderr, "[BF-FMM validate] wait: P2P\n");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, "[BF-FMM validate] done: P2P\n");
    }

    // Q24 timing boundary: tree through final P2P/L2P.  Global projection is
    // depth-independent and would dilute the neighboring-depth comparison.
    body_fitted_fmm::record_persistent_depth_evaluation(
        persistent_tree_runtime_[gpu_id], stream);

    add_sph_gravity_softening_correction(state, config, stream, false);

    // --- Phase 9: Global Projection ---
    if (fmm_config.enable_global_projection) {
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC"))
            fprintf(stderr, "[BF-FMM validate] launch: global projection\n");
        bf_run_global_projection(
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.force_x, state.particles.force_y, state.particles.force_z,
            state.particles.mass, N,
            dd.gp_scalars, stream, false);
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC")) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "[BF-FMM validate] done: global projection\n");
        }
    }

    const bool diagnostic_drain_enqueued = synchronize_at_end &&
        enqueue_periodic_adaptive_depth_diagnostic_drain(gpu_id, stream);
    if (synchronize_at_end) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (diagnostic_drain_enqueued)
            (void)harvest_adaptive_depth_diagnostic_snapshot(gpu_id, true);
        body_fitted_fmm::throw_if_persistent_terminal_error(
            persistent_tree_runtime_[gpu_id]);
        if (getenv("BF_FMM_VALIDATE_PHASE_SYNC"))
            fprintf(stderr, "[BF-FMM validate] done: complete FMM stream\n");
    }
}

// =============================================================================
// compute_outer (RESPRA — writes to gravity_force_*)
// =============================================================================
// compute_outer (RESPA — writes to gravity_force_*)
// =============================================================================


void UnifiedFMMDispatcher::compute_outer(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    const ::cudadem::NeighborData& neighbor_data,
    const ::cudadem::PeriodicBoundaryConfig& pbc,
    double evaluation_time,
    cudaStream_t stream,
    const int* gravity_clump_ids)
{

    (void)neighbor_data;
    if (gravity_clump_ids) {
        const bool controlled_shear = distributed_fmm_[current_gpu()].evaluating &&
            (config.shear_box.enabled || shear_target_count_[current_gpu()] > 0);
        if (config.physics != SimulationPhysics::DEM ||
            ((!controlled_shear) && (pbc.periodic_x || pbc.periodic_y || pbc.periodic_z)) ||
            config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument("clump gravity requires an explicitly controlled DEM FMM depth and supported boundaries");
    }
    const ShearGravityImages shear_images =
        make_shear_gravity_images(config, pbc, evaluation_time);
    if (shear_images.enabled) {
        LocalShearHaloScope halo_scope(current_gpu(), config, pbc, evaluation_time);
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument("adaptive shear gravity must be staged by the common multi-GPU depth controller");
        const int central_count = multigpu_owned_only_[current_gpu()]
            ? state.particles.num_local : state.particles.num_particles;
        const int64_t total64 = static_cast<int64_t>(central_count) *
                                (2*shear_images.layers_x+1);
        if (total64 <= 0 || total64 > INT_MAX / 2)
            throw std::overflow_error(
                "finite shear-image BFFMM source count exceeds edge/index capacity");
        const int total = static_cast<int>(total64);
        if (int64_t(central_count)*shear_images.image_count>INT_MAX/2)
            throw std::overflow_error("logical shear image population exceeds indexing capacity");
        if (std::getenv("GRANIT_SHEAR_SOURCE_DIAGNOSTICS"))
            std::printf("[Shear canonical] gpu=%d physical=%d stored=%d logical=%lld columns=%d rows=%d\n",
                current_gpu(),central_count,total,static_cast<long long>(central_count)*shear_images.image_count,
                2*shear_images.layers_x+1,2*shear_images.layers_y+1);

        SimConfig staged_config = config;
        staged_config.shear_box.enabled = false;
        staged_config.acceleration.unified_fmm.tree_dimension =
            config.acceleration.unified_fmm.tree_dimension;
        staged_config.acceleration.unified_fmm.tree_depth = shear_effective_depth(
            config.acceleration.unified_fmm, shear_images, pbc);
        staged_config.acceleration.unified_fmm.enable_global_projection = false;
        if (staged_config.acceleration.unified_fmm.tree_depth > 20)
            throw std::invalid_argument(
                "finite shear-image BFFMM effective depth exceeds the float-center limit 20");
        shear_fixed_root_[current_gpu()] = shear_static_root(shear_images, pbc, config);
        shear_node_bound_[current_gpu()] = shear_occupied_node_bound(state, config,
            central_count, total, staged_config.acceleration.unified_fmm.tree_depth, shear_images,
            shear_fixed_root_[current_gpu()], stream);
        ensure_shear_fmm_capacity(staged_config, total, current_gpu());
        const int gpu = current_gpu();
        auto& arena = shear_source_arena_[gpu];
        if (gravity_clump_ids && !arena.clump_ids)
            CUDA_CHECK(cudaMalloc(&arena.clump_ids, sizeof(int)*size_t(arena.capacity)));
        const size_t target_bytes = static_cast<size_t>(central_count) *
                                    sizeof(double);
        // compute_outer overwrites the retained acceleration with the new
        // gravity field, so its diagnostic baseline is explicitly zero.
        CUDA_CHECK(cudaMemsetAsync(arena.before_x, 0, target_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(arena.before_y, 0, target_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(arena.before_z, 0, target_bytes, stream));
        constexpr int block = 256;
        materialize_shear_gravity_sources_kernel<<<(total + block - 1) / block,
            block, 0, stream>>>(state.particles.pos_x, state.particles.pos_y,
            state.particles.pos_z, state.particles.mass, central_count,
            shear_images, arena.pos_x, arena.pos_y, arena.pos_z, arena.mass,
            gravity_clump_ids, gravity_clump_ids ? arena.clump_ids : nullptr);
        CUDA_CHECK(cudaGetLastError());
        SimulationState staged = state;
        staged.particles.pos_x = arena.pos_x; staged.particles.pos_y = arena.pos_y;
        staged.particles.pos_z = arena.pos_z; staged.particles.mass = arena.mass;
        staged.particles.num_particles = total;
        staged.particles.num_local = central_count;
        staged.particles.num_ghost = total - central_count;
        staged.particles.capacity = total;
        shear_target_count_[gpu] = central_count;
        shear_cutoff_[gpu] = shear_images.cutoff_radius;
        shear_fixed_root_[gpu] = shear_static_root(shear_images, pbc, config);
        preflight_shear_float_centers(
            shear_fixed_root_[gpu],
            staged_config.acceleration.unified_fmm.tree_depth);
        const int* staged_ids = gravity_clump_ids ? arena.clump_ids : nullptr;
        distributed_fmm_[gpu].gravity_clump_ids = staged_ids;
        try {
            compute_outer(staged, staged_config, neighbor_data, pbc,
                          evaluation_time, stream, staged_ids);
        } catch (...) {
            distributed_fmm_[gpu].gravity_clump_ids = gravity_clump_ids;
            shear_node_bound_[gpu] = 0;
            persistent_tree_runtime_[gpu].pending_presorted = body_fitted_fmm::PresortedMortonInput{};
            shear_target_count_[gpu] = 0;
            shear_cutoff_[gpu] = 0.0;
            shear_fixed_root_[gpu] = make_double4(0.0, 0.0, 0.0, 0.0);
            throw;
        }
        distributed_fmm_[gpu].gravity_clump_ids = gravity_clump_ids;
        if (!distributed_fmm_[gpu].evaluating || distributed_fmm_[gpu].local_only) launch_shear_force_balance<true>(arena, state, central_count,
            config.shear_box.force_balance_warn_threshold, stream);
        shear_node_bound_[gpu] = 0;
        shear_target_count_[gpu] = 0;
        shear_cutoff_[gpu] = 0.0;
        shear_fixed_root_[gpu] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    int gpu_id = current_gpu();
    const int requested_particle_count = multigpu_owned_only_[gpu_id] && shear_target_count_[gpu_id] == 0
        ? state.particles.num_local : state.particles.num_particles;

    if (!initialized_[gpu_id] ||
        persistent_tree_runtime_[gpu_id].host_context.use_presorted_morton != 0 ||
        initialized_tree_type_[gpu_id] !=
            config.acceleration.unified_fmm.tree_dimension ||
        !persistent_depth_configuration_matches(
            persistent_tree_runtime_[gpu_id],
            config.acceleration.unified_fmm)) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            initialize_adaptive(config, state, requested_particle_count,
                                stream);
        else
            initialize(config, requested_particle_count);
    }

    int N = requested_particle_count;
    if (N > host_data_[gpu_id].allocated_max_particles) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            initialize_adaptive(config, state, N, stream);
        else
            initialize(config, N);
    }

    if (config.physics == SimulationPhysics::SPH && config.acceleration.unified_fmm.adaptive_depth.enabled) {
        limit_persistent_sph_depth(persistent_tree_runtime_[gpu_id],
            sph_physical_depth_ceiling(state, config, requested_particle_count, stream), stream);
    }

    fmm::FMMHostData& hd = host_data_[gpu_id];
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    BodyFittedFMMDeviceData& bfd = bf_device_data_[gpu_id];

    const ::cudadem::UnifiedFMMConfig& fmm_config = config.acceleration.unified_fmm;
    bool verbose = fmm_config.verbose;
    dd.verbose = verbose;

    // Replicas are sources only; the output storage still belongs to the
    // central image and therefore has only target_particle_count entries.
    const int output_count = shear_target_count_[gpu_id] > 0
        ? shear_target_count_[gpu_id] : N;
    CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_x, 0,
                               output_count * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_y, 0,
                               output_count * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_z, 0,
                               output_count * sizeof(double), stream));

    {

        // Phase 0: Tree Build
        BF_FMM_NVTX_RANGE("BF-FMM: Tree Build");
        build_tree(state, config, stream);
        if(shear_target_count_[gpu_id]>0 && std::getenv("GRANIT_SHEAR_CENSUS_VALIDATE")) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto metadata=snapshot_persistent_tree_metadata(persistent_tree_runtime_[gpu_id]);
            if(metadata.error || uint64_t(metadata.num_nodes)!=shear_expected_nodes_[gpu_id])
                throw std::runtime_error("occupied-prefix census differs from actual Karras tree");
            std::printf("[Shear census validation] gpu=%d effective=%d nodes=%d PASS\n",
                gpu_id,config.acceleration.unified_fmm.tree_depth,metadata.num_nodes);
        }
        mark_distributed_fmm_phase(gpu_id,1,stream);
        BF_FMM_NVTX_END();

        // Phase 2: publish the completed grouped workload
        BF_FMM_NVTX_RANGE("BF-FMM: Grouped Work");
        publish_rebuilt_workload_view(gpu_id, N);
        BF_FMM_NVTX_END();

    }

    // ===== Phases 3-9: Far-field pipeline after the mandatory rebuild =====


    // Phase 3: Zero local expansions before the far-field passes.
    if (dd.num_nodes > max_nodes_capacity_[gpu_id]) {
        throw std::runtime_error(
            "BF-FMM node count exceeds local-expansion capacity before the outer far-field pass");
    }
    body_fitted_fmm::zero_active_local_expansions(
        dd, &persistent_tree_runtime_[gpu_id].device_context->metadata, stream);

    // Phase 4: P2M + M2M
    BF_FMM_NVTX_RANGE("BF-FMM: P2M+M2M");
    body_fitted_fmm::fmm_compute_all_multipoles_unified(state, config, hd, dd, stream);
    BF_FMM_NVTX_END();


    mark_distributed_fmm_phase(gpu_id,2,stream);
    if (distributed_fmm_[gpu_id].evaluating) {
        exchange_distributed_fmm_sources(gpu_id, &dd, &persistent_tree_runtime_[gpu_id], state, config, stream);
        publish_rebuilt_workload_view(gpu_id, N);
    }
    mark_distributed_fmm_phase(gpu_id,3,stream);

    // Phase 5: target-owned grouped M2L
    BF_FMM_NVTX_RANGE("BF-FMM: M2L");
    run_m2l_phase(gpu_id, stream);
    BF_FMM_NVTX_END();

    // Phase 6: L2L
    BF_FMM_NVTX_RANGE("BF-FMM: L2L");
    body_fitted_fmm::run_l2l_pass_unified(
        dd, &persistent_tree_runtime_[gpu_id].device_context->metadata.root_cell_size,
        stream);
    BF_FMM_NVTX_END();

    // Phase 7: L2P (outer — writes to gravity_force_*)
    BF_FMM_NVTX_RANGE("BF-FMM: L2P");
    {
        body_fitted_fmm::run_l2p_evaluation_outer_unified(
            dd,
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.mass,
            state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
            dd.particle_to_leaf,
            shear_target_count_[gpu_id] > 0 ? shear_target_count_[gpu_id] : N,
            FAR_REAL(config.acceleration.unified_fmm.gravitational_constant),
            &persistent_tree_runtime_[gpu_id].device_context->metadata.root_cell_size,
            stream);
    }
    BF_FMM_NVTX_END();

    mark_distributed_fmm_phase(gpu_id,4,stream);
    // Phase 8: P2P (outer — writes to gravity_force_*)
    BF_FMM_NVTX_RANGE("BF-FMM: P2P");
    run_p2p_outer(gpu_id, state, config, stream, gravity_clump_ids);
    mark_distributed_fmm_phase(gpu_id,5,stream);
    BF_FMM_NVTX_END();

    body_fitted_fmm::record_persistent_depth_evaluation(
        persistent_tree_runtime_[gpu_id], stream);

    add_sph_gravity_softening_correction(state, config, stream, true);

    // Phase 9: Global Projection
    if (fmm_config.enable_global_projection) {
        bf_run_global_projection(
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
            state.particles.mass, N,
            dd.gp_scalars, stream, true);
    }

    // Snapshot copies are queued on the consumer stream.  The writer/boundary
    // drain harvests them once ready; gravity submission never waits here.
    (void)enqueue_periodic_adaptive_depth_diagnostic_drain(gpu_id, stream);
}

#ifdef CUDADEM_HAVE_LET
// =============================================================================
// Multi-GPU common-tree halo exchange. Both entrypoints retain the one-rank
// local dispatcher; multiple ranks require a shared Morton partition.
// =============================================================================
void UnifiedFMMDispatcher::compute_multigpu(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    const ::cudadem::NeighborData& neighbor_data,
    const ::cudadem::PeriodicBoundaryConfig& pbc,
    ::cudadem::LETCommunicator& let_comm,
    uint64_t gravity_evaluation_epoch,
    double evaluation_time,
    cudaStream_t stream,
    const int* gravity_clump_ids)
{
    const int distributed_gpu = current_gpu();
    auto& distributed = distributed_fmm_[distributed_gpu];
    if (gravity_clump_ids) {
        const auto& f = config.acceleration.unified_fmm;
        if (config.physics != SimulationPhysics::DEM ||
            (!config.shear_box.enabled && (pbc.periodic_x || pbc.periodic_y || pbc.periodic_z)) ||
            f.tree_dimension != 3 || f.enable_global_projection ||
            f.direct_threshold_leaf != 0 || f.direct_threshold_internal != 0 ||
            (f.adaptive_depth.enabled && (!distributed.enabled || distributed.compute_depth < 2)))
            throw std::invalid_argument("multi-GPU clump gravity requires controlled 3D DEM, supported boundaries, projection disabled, and zero direct occupancy thresholds");
        if (config.shear_box.enabled && (!(distributed.clump_exclusion_diameter > 0) ||
            !(2*distributed.clump_exclusion_diameter < std::min(double(pbc.box_length_x), double(pbc.box_length_y)))))
            throw std::invalid_argument("shear clump gravity requires twice the maximum clump diameter below both planar box lengths");
    }
    if (distributed.enabled) {
        distributed.gravity_clump_ids = gravity_clump_ids;
        distributed.coordinator = let_comm.coordinator();
        distributed.epoch = gravity_evaluation_epoch;
        distributed.evaluating = true;
        mark_distributed_fmm_phase(distributed_gpu,0,stream);
        if (distributed.ranks > 1 && !distributed.coordinator)
            throw std::logic_error("Distributed FMM requires its phase coordinator");
        try {
            if (distributed.ranks > 1) {
                fmm_collective_check(distributed.coordinator->prepare_producer(
                    ::cudadem::PeerPhase::FmmSource, distributed_gpu,
                    gravity_evaluation_epoch, stream), distributed);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            SimConfig staged = config;
            staged.acceleration.unified_fmm.adaptive_depth.enabled = false;
            staged.acceleration.unified_fmm.enable_global_projection = false;
            staged.acceleration.unified_fmm.tree_depth = distributed.compute_depth ? distributed.compute_depth : config.acceleration.unified_fmm.tree_depth;
            configure_distributed_shear_geometry(distributed_gpu,staged,pbc,evaluation_time);
            SimulationState gravity_state=state;
            const bool projection=config.acceleration.unified_fmm.enable_global_projection && !config.shear_box.enabled;
            const int n=state.particles.num_local;
            const bool cache_clump_gravity=gravity_clump_ids && config.io.output_gravity_force &&
                !config.shear_box.enabled && !projection;
            if(cache_clump_gravity && n){
                distributed.gravity_x.reserve(n);distributed.gravity_y.reserve(n);distributed.gravity_z.reserve(n);
                CUDA_CHECK(cudaMemcpyAsync(distributed.gravity_x.data,state.particles.force_x,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
                CUDA_CHECK(cudaMemcpyAsync(distributed.gravity_y.data,state.particles.force_y,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
                CUDA_CHECK(cudaMemcpyAsync(distributed.gravity_z.data,state.particles.force_z,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            }
            if(projection&&n){
                distributed.gravity_x.reserve(n);distributed.gravity_y.reserve(n);distributed.gravity_z.reserve(n);
                gravity_state.particles.force_x=distributed.gravity_x.data;
                gravity_state.particles.force_y=distributed.gravity_y.data;
                gravity_state.particles.force_z=distributed.gravity_z.data;
                CUDA_CHECK(cudaMemsetAsync(gravity_state.particles.force_x,0,n*sizeof(double),stream));
                CUDA_CHECK(cudaMemsetAsync(gravity_state.particles.force_y,0,n*sizeof(double),stream));
                CUDA_CHECK(cudaMemsetAsync(gravity_state.particles.force_z,0,n*sizeof(double),stream));
            }
            if (n > 0) {
                MultigpuOwnedOnlyScope owned_only(distributed_gpu);
                compute(gravity_state, staged, neighbor_data, pbc, evaluation_time, stream, false, gravity_clump_ids);
            } else {
                invalidate_workload_view(distributed_gpu);
                mark_distributed_fmm_phase(distributed_gpu,1,stream);
                mark_distributed_fmm_phase(distributed_gpu,2,stream);
                exchange_distributed_fmm_sources(distributed_gpu, nullptr, nullptr, state, staged, stream);
                mark_distributed_fmm_phase(distributed_gpu,3,stream);
                mark_distributed_fmm_phase(distributed_gpu,4,stream);
                mark_distributed_fmm_phase(distributed_gpu,5,stream);
            }
            if(projection){
                bf_distributed_projection(distributed_gpu,gravity_state,false,stream);
                if(n)bf_add_distributed_gravity<<<(n+255)/256,256,0,stream>>>(state.particles.force_x,state.particles.force_y,state.particles.force_z,
                    gravity_state.particles.force_x,gravity_state.particles.force_y,gravity_state.particles.force_z,n);
            }
            if(cache_clump_gravity && n)
                cache_shear_gravity_force_kernel<<<(n+255)/256,256,0,stream>>>(
                    state.particles.force_x,state.particles.force_y,state.particles.force_z,
                    distributed.gravity_x.data,distributed.gravity_y.data,distributed.gravity_z.data,
                    state.particles.gravity_force_x,state.particles.gravity_force_y,state.particles.gravity_force_z,n);
        } catch (...) {
            distributed.evaluating = false;
            distributed.gravity_clump_ids = nullptr;
            if (distributed.coordinator) distributed.coordinator->abort();
            throw;
        }
        mark_distributed_fmm_phase(distributed_gpu,6,stream);
        distributed.evaluating = false;
        distributed.gravity_clump_ids = nullptr;
        return;
    }

    if (!let_comm.is_single_gpu())
        throw std::invalid_argument("Multi-GPU BFFMM requires multigpu.morton_ownership=true");
    if (state.particles.num_local > 0) {
        MultigpuOwnedOnlyScope owned_only(distributed_gpu);
        compute(state,config,neighbor_data,pbc,evaluation_time,stream,false,gravity_clump_ids);
    } else invalidate_workload_view(distributed_gpu);
}

void UnifiedFMMDispatcher::compute_outer_multigpu(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    const ::cudadem::NeighborData& neighbor_data,
    const ::cudadem::PeriodicBoundaryConfig& pbc,
    ::cudadem::LETCommunicator& let_comm,
    uint64_t gravity_evaluation_epoch,
    double evaluation_time,
    cudaStream_t stream,
    const int* gravity_clump_ids)
{
    const int distributed_gpu = current_gpu();
    auto& distributed = distributed_fmm_[distributed_gpu];
    if (gravity_clump_ids) {
        const auto& f = config.acceleration.unified_fmm;
        if (config.physics != SimulationPhysics::DEM ||
            (!config.shear_box.enabled && (pbc.periodic_x || pbc.periodic_y || pbc.periodic_z)) ||
            f.tree_dimension != 3 || f.enable_global_projection ||
            f.direct_threshold_leaf != 0 || f.direct_threshold_internal != 0 ||
            (f.adaptive_depth.enabled && (!distributed.enabled || distributed.compute_depth < 2)))
            throw std::invalid_argument("multi-GPU clump gravity requires controlled 3D DEM, supported boundaries, projection disabled, and zero direct occupancy thresholds");
        if (config.shear_box.enabled && (!(distributed.clump_exclusion_diameter > 0) ||
            !(2*distributed.clump_exclusion_diameter < std::min(double(pbc.box_length_x), double(pbc.box_length_y)))))
            throw std::invalid_argument("shear clump gravity requires twice the maximum clump diameter below both planar box lengths");
    }
    if (distributed.enabled) {
        distributed.gravity_clump_ids = gravity_clump_ids;
        distributed.coordinator = let_comm.coordinator();
        distributed.epoch = gravity_evaluation_epoch;
        distributed.evaluating = true;
        mark_distributed_fmm_phase(distributed_gpu,0,stream);
        if (distributed.ranks > 1 && !distributed.coordinator)
            throw std::logic_error("Distributed FMM requires its phase coordinator");
        try {
            if (distributed.ranks > 1) {
                fmm_collective_check(distributed.coordinator->prepare_producer(
                    ::cudadem::PeerPhase::FmmSource, distributed_gpu,
                    gravity_evaluation_epoch, stream), distributed);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            SimConfig staged = config;
            staged.acceleration.unified_fmm.adaptive_depth.enabled = false;
            staged.acceleration.unified_fmm.enable_global_projection = false;
            staged.acceleration.unified_fmm.tree_depth = distributed.compute_depth ? distributed.compute_depth : config.acceleration.unified_fmm.tree_depth;
            configure_distributed_shear_geometry(distributed_gpu,staged,pbc,evaluation_time);
            if (state.particles.num_local > 0) {
                MultigpuOwnedOnlyScope owned_only(distributed_gpu);
                compute_outer(state, staged, neighbor_data, pbc, evaluation_time, stream, gravity_clump_ids);
            } else {
                invalidate_workload_view(distributed_gpu);
                mark_distributed_fmm_phase(distributed_gpu,1,stream);
                mark_distributed_fmm_phase(distributed_gpu,2,stream);
                exchange_distributed_fmm_sources(distributed_gpu, nullptr, nullptr, state, staged, stream);
                mark_distributed_fmm_phase(distributed_gpu,3,stream);
                mark_distributed_fmm_phase(distributed_gpu,4,stream);
                mark_distributed_fmm_phase(distributed_gpu,5,stream);
            }
            if(config.acceleration.unified_fmm.enable_global_projection && !config.shear_box.enabled)
                bf_distributed_projection(distributed_gpu,state,true,stream);
        } catch (...) {
            distributed.evaluating = false;
            distributed.gravity_clump_ids = nullptr;
            if (distributed.coordinator) distributed.coordinator->abort();
            throw;
        }
        mark_distributed_fmm_phase(distributed_gpu,6,stream);
        distributed.evaluating = false;
        distributed.gravity_clump_ids = nullptr;
        return;
    }

    if (!let_comm.is_single_gpu())
        throw std::invalid_argument("Multi-GPU BFFMM requires multigpu.morton_ownership=true");
    if (state.particles.num_local > 0) {
        MultigpuOwnedOnlyScope owned_only(distributed_gpu);
        compute_outer(state,config,neighbor_data,pbc,evaluation_time,stream,gravity_clump_ids);
    } else invalidate_workload_view(distributed_gpu);
}
#endif // CUDADEM_HAVE_LET

// =============================================================================
// Control
// =============================================================================

bool UnifiedFMMDispatcher::needs_rebuild(const ::cudadem::SimulationState& state) {
    (void)state;
    int gpu_id = current_gpu();
    return false;
}

void UnifiedFMMDispatcher::force_rebuild() {
    int gpu_id = current_gpu();
    invalidate_workload_view(gpu_id);
}

void UnifiedFMMDispatcher::invalidate_current_gpu_workload() {
    int gpu_id = current_gpu();
    // Membership changes can alter the owned prefix and invalidate LET source
    // coverage after any communicator-membership change.
    invalidate_workload_view(gpu_id);
}

FMMWorkloadView UnifiedFMMDispatcher::get_workload_view() {
    const int gpu_id = current_gpu();
    const fmm::FMMDeviceData& dd = device_data_[gpu_id];
    const GroupedInteractionArena& arena = grouped_arena_[gpu_id];
    FMMWorkloadView view;
    view.nodes = dd.unified_nodes;
    view.sorted_to_original = dd.sorted_to_original;
    view.num_nodes = dd.num_nodes;
    view.num_particles = dd.num_particles;
    view.m2l_same = nullptr;
    view.m2l_cross = nullptr;
    view.p2p_regular = nullptr;
    view.p2p_self = nullptr;
    view.m2l_same_count = 0;
    view.m2l_cross_count = 0;
    view.p2p_regular_count = 0;
    view.p2p_self_count = 0;
    view.grouped_targets = arena.d_targets;
    view.children_compact = dd.children_compact;
    view.metadata = &persistent_tree_runtime_[gpu_id].device_context->metadata;
    view.node_capacity = arena.node_capacity;
    view.generation = workload_generation_[gpu_id];
    view.valid = workload_view_valid_[gpu_id];
    return view;
}

void UnifiedFMMDispatcher::poll_active_samples(std::vector<BFFMMActiveSample>& out) {
    const int gpu_id = current_gpu();
    for (BFFMMTelemetrySlot& slot : telemetry_ring_[gpu_id]) {
        if (!slot.in_flight) continue;
        const cudaError_t status = cudaEventQuery(slot.end);
        if (status == cudaErrorNotReady) continue;
        if (status != cudaSuccess) {
            // Do not turn a telemetry failure into a simulation-stream wait.
            cudaGetLastError();
            slot.in_flight = false;
            continue;
        }
        float elapsed_ms = 0.0f;
        cudaError_t elapsed_status = cudaEventElapsedTime(&elapsed_ms, slot.start, slot.end);
        if (elapsed_status == cudaSuccess) {
            slot.completed.sample.active_seconds = (double)elapsed_ms * 1.0e-3;
            out.push_back(slot.completed);
        } else {
            cudaGetLastError();
        }
        slot.in_flight = false;
    }
}

void UnifiedFMMDispatcher::throw_if_terminal_error() {
    const int gpu_id = current_gpu();
    if (!initialized_[gpu_id]) return;
    body_fitted_fmm::throw_if_persistent_terminal_error(
        persistent_tree_runtime_[gpu_id]);
}

fmm::FMMStats UnifiedFMMDispatcher::get_statistics() {
    const int gpu_id = current_gpu();
    fmm::FMMStats stats{};
    const auto metadata = body_fitted_fmm::snapshot_persistent_tree_metadata(
        persistent_tree_runtime_[gpu_id]);
    if (device_data_[gpu_id].d_stats) {
        CUDA_CHECK(cudaMemcpyAsync(
            &stats, device_data_[gpu_id].d_stats, sizeof(fmm::FMMStats),
            cudaMemcpyDeviceToHost,
            persistent_tree_runtime_[gpu_id].graph_upload_stream));
        CUDA_CHECK(cudaStreamSynchronize(
            persistent_tree_runtime_[gpu_id].graph_upload_stream));
    }
    stats.num_octree_nodes = metadata.num_nodes;
    stats.num_m2l_interactions =
        static_cast<int64_t>(metadata.task_counts[
            static_cast<int>(body_fitted_fmm::SegClass::M2L_SAME)]) +
        static_cast<int64_t>(metadata.task_counts[
            static_cast<int>(body_fitted_fmm::SegClass::M2L_CROSS)]);
    stats.num_p2p_interactions =
        static_cast<int64_t>(metadata.task_counts[
            static_cast<int>(body_fitted_fmm::SegClass::P2P_REG)]) +
        static_cast<int64_t>(metadata.task_counts[
            static_cast<int>(body_fitted_fmm::SegClass::P2P_SELF)]);

    const auto& distributed=distributed_fmm_[gpu_id];
    if(distributed.remote_interactions.data) {
        int64_t remote[4]{};
        CUDA_CHECK(cudaMemcpy(remote,distributed.remote_interactions.data,sizeof(remote),cudaMemcpyDeviceToHost));
        stats.num_m2l_interactions+=remote[0]+remote[1];
        stats.num_p2p_interactions+=remote[2]+remote[3];
    }
    return stats;
}

void UnifiedFMMDispatcher::get_last_persistent_phase_times(
    double& tree_ms, double& stencil_ms)
{
    const int gpu_id = current_gpu();
    const auto metadata = body_fitted_fmm::snapshot_persistent_tree_metadata(
        persistent_tree_runtime_[gpu_id]);
    tree_ms = metadata.tree_end_ns >= metadata.tree_start_ns
        ? static_cast<double>(metadata.tree_end_ns - metadata.tree_start_ns) * 1.0e-6
        : -1.0;
    stencil_ms = metadata.stencil_end_ns >= metadata.tree_end_ns
        ? static_cast<double>(metadata.stencil_end_ns - metadata.tree_end_ns) * 1.0e-6
        : -1.0;
    if (std::getenv("BF_FMM_TREE_PHASES")) {
        static constexpr const char* phase_names[9] = {
            "reset", "bounds_morton", "sort_unique", "radix_edges",
            "target_prefix", "materialize", "finalize_nodes", "level_lists", "leaf_map"
        };
        uint64_t previous = metadata.tree_start_ns;
        std::fprintf(stderr, "[BF-FMM tree phases]");
        for (int phase = 0; phase < 9; ++phase) {
            const uint64_t stamp = metadata.tree_phase_ns[phase];
            const double elapsed = stamp >= previous
                ? static_cast<double>(stamp - previous) * 1.0e-6 : -1.0;
            std::fprintf(stderr, " %s=%.3fms", phase_names[phase], elapsed);
            previous = stamp;
        }
        std::fprintf(stderr, "\n");
        const uint64_t sort_begin = metadata.tree_phase_ns[1];
        const uint64_t sort_end = metadata.tree_sort_end_ns;
        const uint64_t unique_end = metadata.tree_phase_ns[2];
        const bool valid = sort_begin <= sort_end && sort_end <= unique_end;
        std::fprintf(stderr, "[BF-FMM tree sort] sort=%.3fms unique=%.3fms\n",
            valid ? static_cast<double>(sort_end - sort_begin) * 1.0e-6 : -1.0,
            valid ? static_cast<double>(unique_end - sort_end) * 1.0e-6 : -1.0);
    }
}

void UnifiedFMMDispatcher::get_last_m2l_phase_time(double& m2l_ms)
{
    m2l_ms = last_validation_m2l_ms_[current_gpu()];
}

bool UnifiedFMMDispatcher::get_adaptive_fmm_depth_checkpoint(
    AdaptiveFMMDepthCheckpoint* checkpoint, cudaStream_t stream)
{
    (void)stream;
    if (!checkpoint) return false;
    const int gpu_id = current_gpu();
    if (!initialized_[gpu_id] && has_pending_adaptive_depth_checkpoint_[gpu_id]) {
        *checkpoint = pending_adaptive_depth_checkpoint_[gpu_id];
        return true;
    }
    if (!initialized_[gpu_id] ||
        !persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        return false;
    // Checkpoints are taken at an existing synchronization boundary.  Order
    // the D2H snapshot after the caller's FMM stream before the runtime uses
    // its independent nonblocking diagnostic stream.
    *checkpoint = body_fitted_fmm::snapshot_adaptive_depth_checkpoint(
        persistent_tree_runtime_[gpu_id], stream);
    return true;
}

bool UnifiedFMMDispatcher::restore_adaptive_fmm_depth_checkpoint(
    const AdaptiveFMMDepthCheckpoint& checkpoint, cudaStream_t stream)
{
    if (checkpoint.version != 2) return false;
    const int gpu_id = current_gpu();
    if (!initialized_[gpu_id]) {
        pending_adaptive_depth_checkpoint_[gpu_id] = checkpoint;
        has_pending_adaptive_depth_checkpoint_[gpu_id] = true;
        return true;
    }
    if (!persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        return false;
    body_fitted_fmm::restore_adaptive_depth_checkpoint(
        persistent_tree_runtime_[gpu_id], checkpoint, stream);
    return true;
}

void UnifiedFMMDispatcher::consume_adaptive_fmm_depth_diagnostics(
    std::vector<AdaptiveFMMDepthDiagnostic>& diagnostics,
    uint64_t* dropped, cudaStream_t stream)
{
    if (dropped) *dropped = 0;
    const int gpu_id = current_gpu();
    auto& drain = adaptive_depth_diagnostic_drain_[gpu_id];
    if (initialized_[gpu_id] &&
        persistent_tree_runtime_[gpu_id].adaptive_options.enabled &&
        adaptive_depth_control_mode_[gpu_id] ==
            AdaptiveFMMDepthControlMode::Autonomous) {
        drain_live_adaptive_depth_diagnostics_to_backlog(gpu_id, stream);
        diagnostics.insert(
            diagnostics.end(), drain.backlog.begin(), drain.backlog.end());
        drain.backlog.clear();
        return;
    }
    diagnostics.insert(
        diagnostics.end(), drain.backlog.begin(), drain.backlog.end());
    drain.backlog.clear();
    // Start a fresh periodic window at an explicit output/checkpoint drain.
    drain.evaluations_since_drain = 0;
    if (!initialized_[gpu_id] ||
        !persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        return;
    AdaptiveFMMDepthDiagnostic records[ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE];
    int count = 0;
    uint64_t dropped_count = 0;
    body_fitted_fmm::consume_adaptive_depth_diagnostics(
        persistent_tree_runtime_[gpu_id], records,
        ADAPTIVE_DEPTH_DIAGNOSTIC_RING_SIZE, count, dropped_count, stream);
    diagnostics.insert(diagnostics.end(), records, records + count);
    if (dropped) *dropped = dropped_count;
}

bool UnifiedFMMDispatcher::get_adaptive_fmm_depth_telemetry(
    AdaptiveDepthTelemetry* telemetry, cudaStream_t stream)
{
    if (!telemetry) return false;
    const int gpu_id = current_gpu();
    if (!initialized_[gpu_id] ||
        !persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        return false;
    *telemetry = body_fitted_fmm::snapshot_adaptive_depth_telemetry(
        persistent_tree_runtime_[gpu_id], stream);
    return true;
}

bool UnifiedFMMDispatcher::service_adaptive_fmm_capacity(
    ::cudadem::SimulationState& state,
    const ::cudadem::SimConfig& config,
    cudaStream_t stream)
{
    const int gpu_id = current_gpu();
    const auto& fmm = config.acceleration.unified_fmm;
    if (!initialized_[gpu_id] || !fmm.adaptive_depth.enabled ||
        !persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        return false;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const AdaptiveDepthTelemetry telemetry =
        body_fitted_fmm::snapshot_adaptive_depth_telemetry(
            persistent_tree_runtime_[gpu_id], stream);
    const int high_water = telemetry.controller.node_high_water;
    const int current_capacity = telemetry.node_capacity;
    // A reduced startup ceiling is intentional: storage initially covers the
    // selected depth and one neighbor.  Grow for a reduced ceiling only after
    // the device has actually overflowed or requested an unavailable neighbor,
    // not merely because max_depth has not been provisioned yet.
    const bool fallback_requested =
        telemetry.controller.capacity_fallback_count > 0;
    const bool trigger = high_water >= static_cast<int>(std::floor(
        current_capacity * fmm.adaptive_depth.capacity_growth_trigger));
    if (!trigger && !fallback_requested) return false;

    const int particle_capacity = host_data_[gpu_id].allocated_max_particles;
    const int strict = fixed_depth_node_upper_bound(
        particle_capacity, fmm.adaptive_depth.max_depth,
        fmm.tree_dimension);
    int next_required = high_water;
    if (fallback_requested && adaptive_startup_probe_[gpu_id].valid) {
        const int next_depth = std::min(
            fmm.adaptive_depth.max_depth,
            telemetry.controller.effective_max_depth + 1);
        next_required = std::max(next_required,
            adaptive_startup_probe_[gpu_id].node_count[next_depth]);
    }
    const int desired = std::min(strict, std::max(
        current_capacity + 1, 2 * std::max(next_required, 1)));
    if (desired <= current_capacity) return false;

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const AdaptiveMemoryEstimate current = estimate_adaptive_fmm_memory(
        particle_capacity, current_capacity, fmm.adaptive_depth.max_depth,
        fmm.tree_dimension);
    const AdaptiveMemoryEstimate requested = estimate_adaptive_fmm_memory(
        particle_capacity, desired, fmm.adaptive_depth.max_depth,
        fmm.tree_dimension);
    const uint64_t after_release = std::min<uint64_t>(
        static_cast<uint64_t>(total_bytes), free_bytes + current.total());
    const uint64_t reserve = 2ull * 1024ull * 1024ull * 1024ull;
    const uint64_t allowed = std::min<uint64_t>(
        static_cast<uint64_t>(0.8 * static_cast<double>(after_release)),
        after_release > reserve ? after_release - reserve : after_release / 2);
    if (requested.total() > allowed) {
        // Q32 policy: retain the reduced effective ceiling and continue at a
        // fitting depth when growth is unavailable.  Diagnostics preserve the
        // reason without turning this recoverable condition into a stop.
        body_fitted_fmm::emit_adaptive_capacity_transition(
            persistent_tree_runtime_[gpu_id],
            AdaptiveDepthTransition::EffectiveCeilingChanged,
            fmm.tree_dimension, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return false;
    }

    AdaptiveFMMDepthCheckpoint checkpoint =
        body_fitted_fmm::snapshot_adaptive_depth_checkpoint(
            persistent_tree_runtime_[gpu_id], stream);
    int feasible_ceiling = fmm.adaptive_depth.min_depth;
    if (adaptive_startup_probe_[gpu_id].valid) {
        for (int depth = fmm.adaptive_depth.min_depth;
             depth <= fmm.adaptive_depth.max_depth; ++depth) {
            const int nodes = adaptive_startup_probe_[gpu_id]
                .node_count[depth];
            if (nodes > 0 && nodes <= desired)
                feasible_ceiling = depth;
        }
    } else {
        feasible_ceiling = fmm.adaptive_depth.max_depth;
    }
    checkpoint.controller.effective_max_depth = feasible_ceiling;
    checkpoint.controller.capacity_fallback_count = 0;
    checkpoint.controller.confirmation_count = 0;
    checkpoint.controller.trial_phase = 0;
    checkpoint.controller.candidate_depth =
        checkpoint.controller.selected_depth;
    checkpoint.controller.selected_time_ns = 0;
    checkpoint.controller.selected_sample_count = 0;
    checkpoint.controller.candidate_time_ns = 0;
    pending_adaptive_depth_checkpoint_[gpu_id] = checkpoint;
    has_pending_adaptive_depth_checkpoint_[gpu_id] = true;
    adaptive_capacity_override_[gpu_id] = desired;
    const int service_particle_count =
        (multigpu_owned_only_[gpu_id] ||
         adaptive_depth_control_mode_[gpu_id] ==
             AdaptiveFMMDepthControlMode::ExternalDistributed) &&
        state.particles.num_local > 0
        ? state.particles.num_local : state.particles.num_particles;
    initialize_adaptive(config, state, service_particle_count, stream);
    const int diagnostic_dimension = fmm.tree_dimension;
    body_fitted_fmm::emit_adaptive_capacity_transition(
        persistent_tree_runtime_[gpu_id],
        AdaptiveDepthTransition::CapacityGrowth,
        diagnostic_dimension, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void UnifiedFMMDispatcher::set_adaptive_fmm_depth_control_mode(
    AdaptiveFMMDepthControlMode mode, cudaStream_t stream)
{
    const int gpu_id = current_gpu();
    adaptive_depth_control_mode_[gpu_id] = mode;
    if (!initialized_[gpu_id]) return;
    if (!persistent_tree_runtime_[gpu_id].adaptive_options.enabled)
        throw std::logic_error(
            "external FMM depth control requires adaptive_depth.enabled");
    body_fitted_fmm::set_persistent_external_depth_control(
        persistent_tree_runtime_[gpu_id],
        mode == AdaptiveFMMDepthControlMode::ExternalDistributed, stream);
}

bool UnifiedFMMDispatcher::set_external_adaptive_fmm_depth_command(
    const ExternalDistributedFMMDepthCommand& command, cudaStream_t stream)
{
    const int gpu_id = current_gpu();
    if (command.launch_depth <= 0 || command.selected_depth < 0 ||
        command.candidate_depth < 0)
        return false;
    staged_external_depth_command_[gpu_id] = command;
    has_staged_external_depth_command_[gpu_id] = true;
    if (!initialized_[gpu_id]) return true;
    if (!persistent_tree_runtime_[gpu_id].adaptive_options.enabled ||
        adaptive_depth_control_mode_[gpu_id] !=
            AdaptiveFMMDepthControlMode::ExternalDistributed)
        return false;
    try {
        body_fitted_fmm::set_persistent_external_depth_command(
            persistent_tree_runtime_[gpu_id], command, stream);
    } catch (const std::invalid_argument&) {
        return false;
    }
    return true;
}

bool UnifiedFMMDispatcher::get_external_adaptive_fmm_depth_evaluation(
    ExternalDistributedFMMDepthEvaluation* evaluation,
    cudaStream_t ordered_after_stream)
{
    if (!evaluation) return false;
    const int gpu_id = current_gpu();
    if (!initialized_[gpu_id]) return false;
    *evaluation = body_fitted_fmm::snapshot_persistent_external_depth_evaluation(
        persistent_tree_runtime_[gpu_id], ordered_after_stream);
    return evaluation->valid;
}

void UnifiedFMMDispatcher::poll_multigpu_gravity_epoch_timings(
    std::vector<MultiGPUFMMGravityEpochTiming>& completed, uint64_t* dropped)
{
    // Compatibility for the retired non-Morton telemetry consumer. Current
    // common-tree execution reports DistributedFMMMetrics directly.
    (void)completed;
    if (dropped) *dropped=0;
}

template<int D>
static void expand_grouped_validation_work(
    const std::vector<body_fitted_fmm::GroupedTargetInteractions>& work,
    const std::vector<body_fitted_fmm::UnifiedNode>& nodes,
    const std::vector<int>& children,
    std::vector<int2>& m2l_pairs, std::vector<int>& m2l_ti,
    std::vector<int2>& p2p_pairs, std::vector<int>& p2p_self_nodes)
{
    using T = body_fitted_fmm::GroupedStencilTraits<D>;
    auto child_node = [&](int parent, int slot) {
        const auto& node = nodes[parent];
        const uint8_t bit = static_cast<uint8_t>(1u << slot);
        if (node.childBase == body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE ||
            !(node.childMask & bit)) return -1;
        const int offset = __builtin_popcount(
            node.childMask & static_cast<uint8_t>(bit - 1u));
        return children[static_cast<size_t>(node.childBase) + offset];
    };

    for (int target = 0; target < static_cast<int>(nodes.size()); ++target) {
        const auto& target_node = nodes[target];
        if (!(target_node.flags & body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
            target_node.level < 2 || target_node.target_count == 0) continue;
        const int q = body_fitted_fmm::grouped_target_child<D>(
            target_node.compact_morton);
        const auto& target_work = work[target];
        for (int i = 0; i < T::CROSS_COUNT; ++i) {
            const auto route = body_fitted_fmm::grouped_cross_route(target_work, i);
            if (!body_fitted_fmm::grouped_route_valid(route)) continue;
            const int source = body_fitted_fmm::grouped_route_node(route);
            if (body_fitted_fmm::grouped_route_mask(route) & 1u) {
                p2p_pairs.push_back(make_int2(target, source));
            } else {
                m2l_pairs.push_back(make_int2(target, source));
                m2l_ti.push_back(static_cast<int>(T::host_cross(q, i).operator_id));
            }
        }
        for (int block = 0; block < T::BLOCK_COUNT; ++block) {
            const auto route = body_fitted_fmm::grouped_block_route(target_work, block);
            if (!body_fitted_fmm::grouped_route_valid(route)) continue;
            const int parent = body_fitted_fmm::grouped_route_node(route);
            const uint8_t direct = body_fitted_fmm::grouped_route_mask(route);
            const uint8_t omit =
                body_fitted_fmm::grouped_block_omit_mask(target_work, block);
            const auto stencil = T::host_block(q, block);
            for (int child = 0; child < T::CHILD_COUNT; ++child) {
                const int source = child_node(
                    parent, body_fitted_fmm::grouped_morton_slot<D>(child));
                if (source < 0) continue;
                const uint8_t bit = static_cast<uint8_t>(1u << child);
                if (omit & bit) continue;
                if (stencil.far_mask & bit) {
                    if (direct & bit) {
                        p2p_pairs.push_back(make_int2(target, source));
                    } else {
                        m2l_pairs.push_back(make_int2(target, source));
                        m2l_ti.push_back(
                            static_cast<int>(stencil.operator_id[child]));
                    }
                } else if ((target_node.flags &
                            body_fitted_fmm::UNIFIED_NODE_FLAG_IS_LEAF) &&
                           (stencil.near_mask & bit)) {
                    if (!(direct & bit))
                        throw std::runtime_error(
                            "leaf near interaction is missing its direct route");
                    if (source == target)
                        p2p_self_nodes.push_back(target);
                    else
                        p2p_pairs.push_back(make_int2(target, source));
                }
            }
        }
    }
}

void UnifiedFMMDispatcher::validate_distributed_coverage(int global_particles) {
    const int gpu=current_gpu();auto& state=distributed_fmm_[gpu];auto& plan=state.gpu_plan;
    if(!state.debug_coverage)throw std::runtime_error("enable GRANIT_FMM_VALIDATE_COVERAGE before evaluation");
    const int owned=state.publication.count;if(!owned)return;
    if(global_particles>8192 || global_particles<owned)throw std::runtime_error("invalid small coverage fixture size");
    auto& runtime=persistent_tree_runtime_[gpu];const auto metadata=snapshot_persistent_tree_metadata(runtime);
    const int count=metadata.num_nodes;const auto& dd=device_data_[gpu];
    std::vector<UnifiedNode> nodes(count);std::vector<GroupedTargetInteractions> work(plan.owned_nodes);
    std::vector<int> children(static_cast<size_t>(count)*8),indices(state.debug_index_count);
    CUDA_CHECK(cudaMemcpy(nodes.data(),dd.unified_nodes,nodes.size()*sizeof(UnifiedNode),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(work.data(),runtime.host_context.arena.d_targets,work.size()*sizeof(GroupedTargetInteractions),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(children.data(),dd.children_compact,children.size()*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(),dd.sorted_to_original,indices.size()*sizeof(int),cudaMemcpyDeviceToHost));
    auto cell_code=[&](const UnifiedNode& node) {
        return (uint64_t{1}<<(state.grid.dimension*node.level))|node.compact_morton;
    };
    std::unordered_map<uint64_t,std::vector<int>> complete_members;
    for(int i=0;i<count;++i)complete_members[cell_code(nodes[i])]=state.debug_node_members.at(i);
    for(int peer=0;peer<state.ranks;++peer) {
        if(peer==gpu)continue;
        const auto& q=*state.peers[peer];if(!q.request_count)continue;
        std::vector<UnifiedNode> remote(q.request_count);
        CUDA_CHECK(cudaMemcpy(remote.data(),q.halo_nodes.data,remote.size()*sizeof(UnifiedNode),cudaMemcpyDeviceToHost));
        for(int i=0;i<q.request_count;++i) {
            auto& ids=complete_members[cell_code(remote[i])];const auto& partial=q.debug_memberships.at(i);
            ids.insert(ids.end(),partial.begin(),partial.end());
        }
    }
    std::vector<int2> far,direct;std::vector<int> translations,self;
    if(state.grid.dimension==2)expand_grouped_validation_work<2>(work,nodes,children,far,translations,direct,self);
    else expand_grouped_validation_work<3>(work,nodes,children,far,translations,direct,self);
    std::vector<int> owned_slot(global_particles,-1);
    for(int i=0;i<owned;++i){const int gid=state.debug_arena_ids.at(i);
        if(gid<0||gid>=global_particles||owned_slot[gid]>=0)throw std::runtime_error("invalid owned coverage identity");
        owned_slot[gid]=i;}
    std::vector<uint16_t> coverage(static_cast<size_t>(owned)*global_particles,0);
    auto increment=[&](int target,int source){if(source<0||source>=global_particles||target<0||target>=global_particles||owned_slot[target]<0)
            throw std::runtime_error("invalid coverage particle ID");
        auto& value=coverage[static_cast<size_t>(owned_slot[target])*global_particles+source];
        if(value==UINT16_MAX)throw std::runtime_error("coverage multiplicity overflow");++value;};
    for(const auto pair:far){
        const auto& sources=complete_members.at(cell_code(nodes.at(pair.y)));
        for(int target:state.debug_target_members.at(pair.x))for(int source:sources)increment(target,source);
    }
    for(int node:self)direct.push_back(make_int2(node,node));
    for(const auto pair:direct){const auto& node=nodes.at(pair.y);
        if(uint64_t(node.begin)+node.count>indices.size())throw std::runtime_error("direct range outside source arena");
        for(int target:state.debug_target_members.at(pair.x))for(uint32_t j=0;j<node.count;++j){
            const int slot=indices.at(node.begin+j);
            if(slot==owned_slot[target])continue; // The same exclusion used by the actual P2P kernel.
            increment(target,state.debug_arena_ids.at(slot));}
    }
    // Validate the references consumed by the fused numerical kernels, including
    // direct routes inherited by each target leaf from its ancestors.
    const auto& combined=state.remote_lists;
    std::vector<int> remote_targets(combined.far_count),remote_leaves(combined.near_count);
    std::vector<uint64_t> direct_offsets(size_t(plan.owned_nodes)+1);
    if(combined.far_count)CUDA_CHECK(cudaMemcpy(remote_targets.data(),combined.far_targets.data,
        remote_targets.size()*sizeof(int),cudaMemcpyDeviceToHost));
    if(combined.near_count) {
        CUDA_CHECK(cudaMemcpy(remote_leaves.data(),combined.near_leaves.data,remote_leaves.size()*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(direct_offsets.data(),combined.direct_offsets.data,direct_offsets.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    }
    const int slots=state.grid.dimension==2 ?
        GroupedStencilTraits<2>::CROSS_COUNT+GroupedStencilTraits<2>::BLOCK_COUNT*GroupedStencilTraits<2>::CHILD_COUNT :
        GroupedStencilTraits<3>::CROSS_COUNT+GroupedStencilTraits<3>::BLOCK_COUNT*GroupedStencilTraits<3>::CHILD_COUNT;
    std::vector<FMMRemoteRef> remote_far(size_t(slots)*combined.far_count),remote_direct(direct_offsets.back());
    if(!remote_far.empty())CUDA_CHECK(cudaMemcpy(remote_far.data(),combined.far_sources.data,
        remote_far.size()*sizeof(FMMRemoteRef),cudaMemcpyDeviceToHost));
    if(!remote_direct.empty())CUDA_CHECK(cudaMemcpy(remote_direct.data(),combined.direct_sources.data,
        remote_direct.size()*sizeof(FMMRemoteRef),cudaMemcpyDeviceToHost));
    std::vector<std::vector<UnifiedNode>> peer_nodes(state.ranks);
    std::vector<std::vector<FMMSourceParticle>> peer_particles(state.ranks);
    for(int peer=0;peer<state.ranks;++peer) {
        const auto& q=*state.peers[peer];if(!q.request_count)continue;
        peer_nodes[peer].resize(q.request_count);peer_particles[peer].resize(q.particle_count);
        CUDA_CHECK(cudaMemcpy(peer_nodes[peer].data(),q.halo_nodes.data,q.request_count*sizeof(UnifiedNode),cudaMemcpyDeviceToHost));
        if(q.particle_count)CUDA_CHECK(cudaMemcpy(peer_particles[peer].data(),q.halo_particles.data,
            q.particle_count*sizeof(FMMSourceParticle),cudaMemcpyDeviceToHost));
    }
    auto consume_remote=[&](int target,FMMRemoteRef ref,bool is_direct) {
        if(ref.node<0)return;
        const auto& q=*state.peers.at(ref.peer);const auto& node=peer_nodes.at(ref.peer).at(ref.node);
        if(q.debug_memberships.at(ref.node).size()!=node.count)
            throw std::runtime_error("independent halo membership count mismatch");
        for(int tid:state.debug_target_members.at(target)) {
            if(is_direct) {
                const auto& payload=peer_particles.at(ref.peer);
                if(uint64_t(node.begin)+node.count>payload.size())throw std::runtime_error("halo direct range outside payload");
                for(uint32_t j=0;j<node.count;++j)
                    increment(tid,q.debug_source_ids.at(payload.at(node.begin+j).original));
            } else for(int sid:complete_members.at(cell_code(node)))increment(tid,sid);
        }
    };
    for(int row=0;row<combined.far_count;++row)for(int slot=0;slot<slots;++slot)
        consume_remote(remote_targets[row],remote_far[size_t(slot)*combined.far_count+row],false);
    for(int leaf:remote_leaves)for(uint64_t i=direct_offsets.at(leaf);i<direct_offsets.at(leaf+1);++i)
        consume_remote(leaf,remote_direct.at(i),true);
    uint64_t missing=0,duplicate=0,diagonal=0;
    for(int target=0;target<owned;++target)for(int source=0;source<global_particles;++source){
        const auto actual=coverage[static_cast<size_t>(target)*global_particles+source];
        if(source==state.debug_arena_ids[target])diagonal+=actual;
        else if(!actual)++missing;else if(actual!=1)++duplicate;}
    if(missing||duplicate||diagonal)throw std::runtime_error("distributed exact coverage: missing="+std::to_string(missing)+
        " duplicate="+std::to_string(duplicate)+" self="+std::to_string(diagonal));
    std::printf("distributed_exact_coverage rank=%d epoch=%llu targets=%d sources=%d pairs=%llu PASS\n",gpu,
        (unsigned long long)state.epoch,owned,global_particles,(unsigned long long)owned*(global_particles-1));
}

void UnifiedFMMDispatcher::get_completeness_data(
    std::vector<int2>& h_m2l_pairs,
    std::vector<int>&  h_m2l_ti,
    std::vector<int2>&  h_p2p_pairs,
    std::vector<int>&  h_p2p_self_nodes,
    std::vector<fmm::FMMOctreeNode>& h_nodes,
    std::vector<int>&  h_children_indices,
    std::vector<int>&  h_particle_to_leaf,
    int& num_nodes,
    int& num_particles)
{
    const int gpu_id = current_gpu();
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    const auto metadata = body_fitted_fmm::snapshot_persistent_tree_metadata(
        persistent_tree_runtime_[gpu_id]);
    const cudaStream_t validation_stream =
        persistent_tree_runtime_[gpu_id].graph_upload_stream;
    auto copy_from_device = [validation_stream](void* destination,
                                                const void* source,
                                                size_t bytes) {
        if (bytes == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes,
                                   cudaMemcpyDeviceToHost, validation_stream));
        CUDA_CHECK(cudaStreamSynchronize(validation_stream));
    };
    if (metadata.error != 0) {
        throw std::runtime_error(
            "cannot export validation snapshot: persistent tree/task build failed with error " +
            std::to_string(metadata.error));
    }

    num_nodes     = metadata.num_nodes;
    num_particles = dd.num_particles;
    if (num_nodes <= 0 ||
        num_nodes > persistent_tree_runtime_[gpu_id].host_context.node_capacity) {
        throw std::runtime_error(
            "persistent validation snapshot contains an invalid node count");
    }

    // Copy the grouped, address-stable work records and expand them only for
    // diagnostics. Production arithmetic consumes the records directly.
    const auto& arena = grouped_arena_[gpu_id];
    if (!arena.d_targets || arena.node_capacity < num_nodes)
        throw std::runtime_error("persistent validation snapshot has no grouped work records");
    std::vector<body_fitted_fmm::GroupedTargetInteractions> h_work(num_nodes);
    std::vector<body_fitted_fmm::UnifiedNode> h_uni(num_nodes);
    std::vector<int> h_compact(static_cast<size_t>(num_nodes) * 8);
    copy_from_device(h_work.data(), arena.d_targets,
                     h_work.size() * sizeof(h_work[0]));
    copy_from_device(h_uni.data(), dd.unified_nodes,
                     h_uni.size() * sizeof(h_uni[0]));
    copy_from_device(h_compact.data(), dd.children_compact,
                     h_compact.size() * sizeof(h_compact[0]));
    h_m2l_pairs.clear(); h_m2l_ti.clear();
    h_p2p_pairs.clear(); h_p2p_self_nodes.clear();
    if (metadata.active_tree == 2)
        expand_grouped_validation_work<2>(h_work, h_uni, h_compact,
            h_m2l_pairs, h_m2l_ti, h_p2p_pairs, h_p2p_self_nodes);
    else
        expand_grouped_validation_work<3>(h_work, h_uni, h_compact,
            h_m2l_pairs, h_m2l_ti, h_p2p_pairs, h_p2p_self_nodes);
    const int expanded_counts[4] = {
        metadata.task_counts[0], metadata.task_counts[1],
        metadata.task_counts[2], metadata.task_counts[3]};
    if (static_cast<int>(h_m2l_pairs.size()) != expanded_counts[0] + expanded_counts[1] ||
        static_cast<int>(h_p2p_pairs.size()) != expanded_counts[2] ||
        static_cast<int>(h_p2p_self_nodes.size()) != expanded_counts[3])
        throw std::runtime_error(
            "grouped validation expansion disagrees with published counts: expanded=" +
            std::to_string(h_m2l_pairs.size()) + "/" +
            std::to_string(h_p2p_pairs.size()) + "/" +
            std::to_string(h_p2p_self_nodes.size()) + " published=" +
            std::to_string(expanded_counts[0] + expanded_counts[1]) + "/" +
            std::to_string(expanded_counts[2]) + "/" +
            std::to_string(expanded_counts[3]));

    // Copy exactly the published node prefix, reject invalid slots, and rebuild
    // the host-only parent/AABB fields required by the validator.
    h_nodes.resize(num_nodes);
    if (num_nodes > 0 && dd.unified_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            if (!(h_uni[i].flags & body_fitted_fmm::UNIFIED_NODE_FLAG_VALID))
                throw std::runtime_error("invalid node inside published device tree prefix");
            const float half_cell = h_uni[i].radius / 1.7320508075688772f;
            // FMMOctreeNode is a legacy FP32 diagnostic view. The active
            // UnifiedNode retains its FP64 world center; narrow only while
            // exporting this non-computational compatibility representation.
            const float center_x = static_cast<float>(h_uni[i].center.x);
            const float center_y = static_cast<float>(h_uni[i].center.y);
            const float center_z = static_cast<float>(h_uni[i].center.z);
            h_nodes[i].aabb_min = make_float3(
                center_x - half_cell, center_y - half_cell, center_z - half_cell);
            h_nodes[i].aabb_max = make_float3(
                center_x + half_cell, center_y + half_cell, center_z + half_cell);
            h_nodes[i].com = make_float3(center_x, center_y, center_z);
            h_nodes[i].R_max = h_uni[i].radius;
            h_nodes[i].parent_idx = -1;
            h_nodes[i].first_child = -1;
            h_nodes[i].particle_begin = h_uni[i].begin;
            h_nodes[i].particle_end   = h_uni[i].begin + h_uni[i].count;
            h_nodes[i].n_children     = static_cast<uint8_t>(__builtin_popcount(h_uni[i].childMask));
            h_nodes[i].level          = h_uni[i].level;
            h_nodes[i].flags          = h_uni[i].flags;
            h_nodes[i].compact_morton = h_uni[i].compact_morton;
        }
    } else {
        throw std::runtime_error("persistent validation snapshot has no unified node buffer");
    }

    if (!dd.children_compact)
        throw std::runtime_error("persistent validation snapshot has no child buffer");
    h_children_indices.assign(static_cast<size_t>(num_nodes) * 8, -1);
    for (int i = 0; i < num_nodes; ++i) {
        const auto& node = h_uni[i];
        if (node.childBase == body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE) continue;
        h_nodes[i].first_child = static_cast<int>(node.childBase);
        for (int c = 0; c < 8; ++c) {
            if (!(node.childMask & (1u << c))) continue;
            const int offset = __builtin_popcount(node.childMask & ((1u << c) - 1u));
            const size_t compact_index = static_cast<size_t>(node.childBase) + offset;
            if (compact_index >= h_compact.size())
                throw std::runtime_error("device child offset is outside the published tree prefix");
            const int child = h_compact[compact_index];
            if (child < 0 || child >= num_nodes)
                throw std::runtime_error("device child index is outside the published tree prefix");
            h_children_indices[static_cast<size_t>(i) * 8 + c] = child;
            if (h_nodes[child].parent_idx != -1)
                throw std::runtime_error("device tree node has multiple parents");
            h_nodes[child].parent_idx = i;
        }
    }

    // Particle-to-leaf mapping
    h_particle_to_leaf.resize(num_particles);
    if (num_particles > 0 && dd.particle_to_leaf) {
        copy_from_device(h_particle_to_leaf.data(), dd.particle_to_leaf,
                         static_cast<size_t>(num_particles) * sizeof(int));
        for (const int leaf : h_particle_to_leaf) {
            if (leaf < 0 || leaf >= num_nodes || h_nodes[leaf].n_children != 0)
                throw std::runtime_error("invalid particle-to-leaf entry in device snapshot");
        }
    } else if (num_particles > 0) {
        throw std::runtime_error("persistent validation snapshot has no particle-to-leaf map");
    }
}

void UnifiedFMMDispatcher::get_expansion_data(
    std::vector<fmm::MultipoleExpansion>& h_multipoles,
    std::vector<fmm::LocalExpansion>& h_local,
    int& num_nodes)
{
    int gpu_id = current_gpu();
    fmm::FMMDeviceData& dd = device_data_[gpu_id];

    const auto metadata = body_fitted_fmm::snapshot_persistent_tree_metadata(
        persistent_tree_runtime_[gpu_id]);
    if (metadata.error != 0)
        throw std::runtime_error("cannot export expansions from a failed persistent build");
    num_nodes = metadata.num_nodes;
    h_multipoles.resize(num_nodes);
    h_local.resize(num_nodes);

    if (num_nodes > 0 && dd.multipoles) {
        CUDA_CHECK(cudaMemcpyAsync(
            h_multipoles.data(), dd.multipoles,
            static_cast<size_t>(num_nodes) * sizeof(fmm::MultipoleExpansion),
            cudaMemcpyDeviceToHost,
            persistent_tree_runtime_[gpu_id].graph_upload_stream));
    }
    if (num_nodes > 0 && dd.local_expansions) {
        CUDA_CHECK(cudaMemcpyAsync(
            h_local.data(), dd.local_expansions,
            static_cast<size_t>(num_nodes) * sizeof(fmm::LocalExpansion),
            cudaMemcpyDeviceToHost,
            persistent_tree_runtime_[gpu_id].graph_upload_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(
        persistent_tree_runtime_[gpu_id].graph_upload_stream));
}

void UnifiedFMMDispatcher::debug_let_inputs() {
    int gpu_id = current_gpu();
    fmm::FMMDeviceData& dd = device_data_[gpu_id];
    fprintf(stderr, "[debug_let_inputs] gpu=%d num_nodes=%d num_particles=%d max_level=%d\n",
            gpu_id, dd.num_nodes, dd.num_particles, dd.max_level);
    if (dd.level_start && dd.level_count) {
        int ls[fmm::FMM_MAX_LEVELS], lc[fmm::FMM_MAX_LEVELS];
        CUDA_CHECK(cudaMemcpy(ls, dd.level_start, sizeof(ls), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lc, dd.level_count, sizeof(lc), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[debug_let_inputs] FMM_MAX_LEVELS=%d\n  level_start =", (int)fmm::FMM_MAX_LEVELS);
        for (int i = 0; i < fmm::FMM_MAX_LEVELS; i++) fprintf(stderr, " %d", ls[i]);
        fprintf(stderr, "\n  level_count =");
        for (int i = 0; i < fmm::FMM_MAX_LEVELS; i++) fprintf(stderr, " %d", lc[i]);
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "[debug_let_inputs] level_start=%p level_count=%p (NULL!)\n",
                (void*)dd.level_start, (void*)dd.level_count);
    }
    if (dd.nodes && dd.num_nodes > 0) {
        int k = std::min(4, dd.num_nodes);
        std::vector<fmm::FMMOctreeNode> hn(k);
        CUDA_CHECK(cudaMemcpy(hn.data(), dd.nodes, k * sizeof(fmm::FMMOctreeNode), cudaMemcpyDeviceToHost));
        for (int i = 0; i < k; i++) {
            fprintf(stderr, "  node[%d] lvl=%d nchild=%d begin=%d end=%d first_child=%d aabb=(%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n",
                    i, (int)hn[i].level, (int)hn[i].n_children, hn[i].particle_begin, hn[i].particle_end,
                    hn[i].first_child,
                    hn[i].aabb_min.x, hn[i].aabb_min.y, hn[i].aabb_min.z,
                    hn[i].aabb_max.x, hn[i].aabb_max.y, hn[i].aabb_max.z);
        }
    }
}

} // namespace body_fitted_fmm

double sph_gravity_regularization_length(const SimConfig& config, cudaStream_t stream) {
    return body_fitted_fmm::sph_compact_baseline_length(config, stream);
}
} // namespace cudadem

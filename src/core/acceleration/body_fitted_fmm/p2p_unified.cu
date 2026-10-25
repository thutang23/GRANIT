// =============================================================================
// Grouped P2P Kernels (leaf-owned accumulation plus benchmark reference kernels)
// =============================================================================
// Included only by body_fitted_fmm_manager.cu.
// =============================================================================

#include "grouped_interaction_arena.cuh"
#include "persistent_tree_traversal.cuh"
#include "grouped_stencil_traits.cuh"

#include "remote_interaction_lists.cuh"
namespace body_fitted_fmm {

// Public particle storage remains FP64.  This switch changes only the
// registers/shared-memory arithmetic used by near-field kernels.
// It is deliberately independent of FMM_FLOAT_P2P, which stages storage.
#ifdef FMM_P2P_LOCAL_FP32
using p2p_compute_t = float;
#define BF_P2P_REAL(x) (x##f)
#define BF_P2P_SQRT(x) sqrtf(x)
#else
using p2p_compute_t = double;
#define BF_P2P_REAL(x) (x)
#define BF_P2P_SQRT(x) sqrt(x)
#endif

// Alias local arithmetic only; all input/output pointer declarations remain
// explicitly double.
#define p2p_real_t p2p_compute_t

template<class Compute>
__device__ __forceinline__
void compute_gravitational_acceleration_from_delta_unified(
    Compute dx, Compute dy, Compute dz,
    Compute mj, Compute softening_sq, Compute G,
    Compute& fx, Compute& fy, Compute& fz)
{
    Compute raw_r_sq = dx*dx + dy*dy + dz*dz;
    // Negative softening_sq encodes the SPH compact, finite baseline. DEM
    // retains its original Plummer denominator for nonnegative values.
    Compute r_sq = softening_sq < 0 ? fmax(raw_r_sq, -softening_sq)
                                      : raw_r_sq + softening_sq;
    if (!(r_sq > Compute(0))) {
        fx = fy = fz = Compute(0);
        return;
    }
    Compute r_inv = Compute(1) / sqrt(r_sq);
    Compute r_inv3 = r_inv * r_inv * r_inv;
    Compute f_mag = G * mj * r_inv3;
    fx = f_mag * dx; fy = f_mag * dy; fz = f_mag * dz;
}

// Shared-memory P2P kernel (leaf pair, writes force)
__global__ void fmm_p2p_shared_unified(
    const int2* __restrict__ p2p_pairs, int num_pairs,
    const UnifiedNode* __restrict__ nodes, int num_nodes, int num_particles,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    double* __restrict__ force_x, double* __restrict__ force_y, double* __restrict__ force_z,
    const int* __restrict__ sorted_indices,
    p2p_real_t G, p2p_real_t softening)
{
    // Keep world positions in FP64 until each source-target subtraction is
    // complete. FP32 local arithmetic below is then applied only to a local
    // displacement, so global translations cannot quantize near contacts.
    __shared__ double sh_pos_x[64], sh_pos_y[64], sh_pos_z[64];
    __shared__ float sh_mass[64];

    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    int2 pair = p2p_pairs[pair_idx];
    int leafA = pair.x, leafB = pair.y;
    if (leafA < 0 || leafA >= num_nodes || leafB < 0 || leafB >= num_nodes) return;
    if (leafA == leafB) return;

    const UnifiedNode& nA = nodes[leafA];
    const UnifiedNode& nB = nodes[leafB];

    int beginA = nA.begin, endA = nA.begin + nA.count;
    int beginB = nB.begin, endB = nB.begin + nB.count;
    int countB = endB - beginB;
    if (beginA < 0 || endA < beginA || beginB < 0 || endB < beginB) return;

    p2p_real_t softening_sq = softening < 0 ? -softening * softening : softening * softening;

    for (int tile_start = 0; tile_start < countB; tile_start += 64) {
        int tile_end = (tile_start + 64 < countB) ? (tile_start + 64) : countB;
        int tile_size = tile_end - tile_start;

        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            int j_sorted = beginB + tile_start + j;
            if (j_sorted < 0 || j_sorted >= num_particles) continue;
            int j_orig = sorted_indices[j_sorted];
            if (j_orig >= 0 && j_orig < num_particles) {
                sh_pos_x[j] = pos_x[j_orig]; sh_pos_y[j] = pos_y[j_orig];
                sh_pos_z[j] = pos_z[j_orig]; sh_mass[j] = mass[j_orig];
            }
        }
        __syncthreads();

        for (int i_sorted = beginA + threadIdx.x; i_sorted < endA; i_sorted += blockDim.x) {
            int i = sorted_indices[i_sorted];
            if (i < 0 || i >= num_particles) continue;

            const double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
            p2p_real_t mi = static_cast<p2p_real_t>(mass[i]);
            p2p_real_t fx = BF_P2P_REAL(0.0), fy = BF_P2P_REAL(0.0), fz = BF_P2P_REAL(0.0);

            for (int j = 0; j < tile_size; j++) {
                int j_sorted = beginB + tile_start + j;
                if (j_sorted < 0 || j_sorted >= num_particles) continue;
                int j_orig = sorted_indices[j_sorted];
                if (i == j_orig) continue;

                const p2p_real_t dx = static_cast<p2p_real_t>(sh_pos_x[j] - xi);
                const p2p_real_t dy = static_cast<p2p_real_t>(sh_pos_y[j] - yi);
                const p2p_real_t dz = static_cast<p2p_real_t>(sh_pos_z[j] - zi);
                p2p_real_t dfx, dfy, dfz;
                compute_gravitational_acceleration_from_delta_unified(
                    dx, dy, dz,
                    static_cast<p2p_real_t>(sh_mass[j]),
                    softening_sq, G, dfx, dfy, dfz);
                fx += mi * dfx; fy += mi * dfy; fz += mi * dfz;
            }
            atomicAdd(&force_x[i], fx);
            atomicAdd(&force_y[i], fy);
            atomicAdd(&force_z[i], fz);
        }
        __syncthreads();
    }
}

// Shared-memory P2P outer kernel (writes acceleration)
__global__ void fmm_p2p_shared_outer_unified(
    const int2* __restrict__ p2p_pairs, int num_pairs,
    const UnifiedNode* __restrict__ nodes, int num_nodes, int num_particles,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    double* __restrict__ grav_acc_x, double* __restrict__ grav_acc_y, double* __restrict__ grav_acc_z,
    const int* __restrict__ sorted_indices,
    p2p_real_t G, p2p_real_t softening)
{
    // World-coordinate subtraction must precede the optional FP32 local P2P
    // path; see fmm_p2p_shared_unified above.
    __shared__ double sh_pos_x[64], sh_pos_y[64], sh_pos_z[64];
    __shared__ float sh_mass[64];

    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    int2 pair = p2p_pairs[pair_idx];
    int leafA = pair.x, leafB = pair.y;
    if (leafA < 0 || leafA >= num_nodes || leafB < 0 || leafB >= num_nodes) return;
    if (leafA == leafB) return;

    const UnifiedNode& nA = nodes[leafA];
    const UnifiedNode& nB = nodes[leafB];

    int beginA = nA.begin, endA = nA.begin + nA.count;
    int beginB = nB.begin, endB = nB.begin + nB.count;
    int countB = endB - beginB;
    if (beginA < 0 || endA < beginA || beginB < 0 || endB < beginB) return;

    p2p_real_t softening_sq = softening < 0 ? -softening * softening : softening * softening;

    for (int tile_start = 0; tile_start < countB; tile_start += 64) {
        int tile_end = (tile_start + 64 < countB) ? (tile_start + 64) : countB;
        int tile_size = tile_end - tile_start;

        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            int j_sorted = beginB + tile_start + j;
            if (j_sorted < 0 || j_sorted >= num_particles) continue;
            int j_orig = sorted_indices[j_sorted];
            if (j_orig >= 0 && j_orig < num_particles) {
                sh_pos_x[j] = pos_x[j_orig]; sh_pos_y[j] = pos_y[j_orig];
                sh_pos_z[j] = pos_z[j_orig]; sh_mass[j] = mass[j_orig];
            }
        }
        __syncthreads();

        for (int i_sorted = beginA + threadIdx.x; i_sorted < endA; i_sorted += blockDim.x) {
            int i = sorted_indices[i_sorted];
            if (i < 0 || i >= num_particles) continue;

            const double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
            p2p_real_t ax = BF_P2P_REAL(0.0), ay = BF_P2P_REAL(0.0), az = BF_P2P_REAL(0.0);

            for (int j = 0; j < tile_size; j++) {
                int j_sorted = beginB + tile_start + j;
                if (j_sorted < 0 || j_sorted >= num_particles) continue;
                int j_orig = sorted_indices[j_sorted];
                if (i == j_orig) continue;

                const p2p_real_t dx = static_cast<p2p_real_t>(sh_pos_x[j] - xi);
                const p2p_real_t dy = static_cast<p2p_real_t>(sh_pos_y[j] - yi);
                const p2p_real_t dz = static_cast<p2p_real_t>(sh_pos_z[j] - zi);
                p2p_real_t dfx, dfy, dfz;
                compute_gravitational_acceleration_from_delta_unified(
                    dx, dy, dz,
                    static_cast<p2p_real_t>(sh_mass[j]),
                    softening_sq, G, dfx, dfy, dfz);
                ax += dfx; ay += dfy; az += dfz;
            }
            atomicAdd(&grav_acc_x[i], ax);
            atomicAdd(&grav_acc_y[i], ay);
            atomicAdd(&grav_acc_z[i], az);
        }
        __syncthreads();
    }
}

// Self-interaction P2P kernel (writes force)
__global__ void fmm_p2p_self_unified(
    const UnifiedNode* __restrict__ nodes, int num_leaves,
    const int* __restrict__ leaf_indices,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    double* __restrict__ force_x, double* __restrict__ force_y, double* __restrict__ force_z,
    const int* __restrict__ sorted_indices,
    p2p_real_t G, p2p_real_t softening)
{
    constexpr int TILE = 64;
    __shared__ double sh_pos_x[TILE], sh_pos_y[TILE], sh_pos_z[TILE];
    __shared__ p2p_real_t sh_mass[TILE];

    int leaf_idx = blockIdx.x;
    if (leaf_idx >= num_leaves) return;

    int leaf = leaf_indices[leaf_idx];
    const UnifiedNode& node = nodes[leaf];
    int begin = node.begin, end = node.begin + node.count;
    int N = end - begin;
    if (N <= 0) return;

    p2p_real_t softening_sq = softening < 0 ? -softening * softening : softening * softening;

    for (int tile = 0; tile < N; tile += TILE) {
        int t_end = min(tile + TILE, N);
        int t_size = t_end - tile;

        for (int k = threadIdx.x; k < t_size; k += blockDim.x) {
            int idx = sorted_indices[begin + tile + k];
            sh_pos_x[k] = pos_x[idx]; sh_pos_y[k] = pos_y[idx];
            sh_pos_z[k] = pos_z[idx]; sh_mass[k] = static_cast<p2p_real_t>(mass[idx]);
        }
        __syncthreads();

        for (int i_local = threadIdx.x; i_local < N; i_local += blockDim.x) {
            int i = sorted_indices[begin + i_local];
            const double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
            p2p_real_t mi = static_cast<p2p_real_t>(mass[i]);
            p2p_real_t fx = BF_P2P_REAL(0.0), fy = BF_P2P_REAL(0.0), fz = BF_P2P_REAL(0.0);

            for (int k = 0; k < t_size; k++) {
                int j = sorted_indices[begin + tile + k];
                if (i == j) continue;
                const p2p_real_t dx = static_cast<p2p_real_t>(sh_pos_x[k] - xi);
                const p2p_real_t dy = static_cast<p2p_real_t>(sh_pos_y[k] - yi);
                const p2p_real_t dz = static_cast<p2p_real_t>(sh_pos_z[k] - zi);
                p2p_real_t raw_r_sq = dx*dx + dy*dy + dz*dz;
                p2p_real_t r_sq = softening_sq < 0 ? fmax(raw_r_sq, -softening_sq)
                                                  : raw_r_sq + softening_sq;
                if (!(r_sq > BF_P2P_REAL(0.0))) continue;
                p2p_real_t r_inv = BF_P2P_REAL(1.0) / BF_P2P_SQRT(r_sq);
                p2p_real_t r_inv3 = r_inv * r_inv * r_inv;
                p2p_real_t f_mag = G * mi * sh_mass[k] * r_inv3;
                fx += f_mag * dx; fy += f_mag * dy; fz += f_mag * dz;
            }
            atomicAdd(&force_x[i], fx);
            atomicAdd(&force_y[i], fy);
            atomicAdd(&force_z[i], fz);
        }
        __syncthreads();
    }
}

// Self-interaction P2P outer kernel (writes acceleration)
__global__ void fmm_p2p_self_outer_unified(
    const UnifiedNode* __restrict__ nodes, int num_leaves,
    const int* __restrict__ leaf_indices,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    double* __restrict__ grav_acc_x, double* __restrict__ grav_acc_y, double* __restrict__ grav_acc_z,
    const int* __restrict__ sorted_indices,
    p2p_real_t G, p2p_real_t softening)
{
    constexpr int TILE = 64;
    __shared__ double sh_pos_x[TILE], sh_pos_y[TILE], sh_pos_z[TILE];
    __shared__ p2p_real_t sh_mass[TILE];

    int leaf_idx = blockIdx.x;
    if (leaf_idx >= num_leaves) return;

    int leaf = leaf_indices[leaf_idx];
    const UnifiedNode& node = nodes[leaf];
    int begin = node.begin, end = node.begin + node.count;
    int N = end - begin;
    if (N <= 0) return;

    p2p_real_t softening_sq = softening < 0 ? -softening * softening : softening * softening;

    for (int tile = 0; tile < N; tile += TILE) {
        int t_end = min(tile + TILE, N);
        int t_size = t_end - tile;

        for (int k = threadIdx.x; k < t_size; k += blockDim.x) {
            int idx = sorted_indices[begin + tile + k];
            sh_pos_x[k] = pos_x[idx]; sh_pos_y[k] = pos_y[idx];
            sh_pos_z[k] = pos_z[idx]; sh_mass[k] = static_cast<p2p_real_t>(mass[idx]);
        }
        __syncthreads();

        for (int i_local = threadIdx.x; i_local < N; i_local += blockDim.x) {
            int i = sorted_indices[begin + i_local];
            const double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
            p2p_real_t ax = BF_P2P_REAL(0.0), ay = BF_P2P_REAL(0.0), az = BF_P2P_REAL(0.0);

            for (int k = 0; k < t_size; k++) {
                int j = sorted_indices[begin + tile + k];
                if (i == j) continue;
                const p2p_real_t dx = static_cast<p2p_real_t>(sh_pos_x[k] - xi);
                const p2p_real_t dy = static_cast<p2p_real_t>(sh_pos_y[k] - yi);
                const p2p_real_t dz = static_cast<p2p_real_t>(sh_pos_z[k] - zi);
                p2p_real_t raw_r_sq = dx*dx + dy*dy + dz*dz;
                p2p_real_t r_sq = softening_sq < 0 ? fmax(raw_r_sq, -softening_sq)
                                                  : raw_r_sq + softening_sq;
                if (!(r_sq > BF_P2P_REAL(0.0))) continue;
                p2p_real_t r_inv = BF_P2P_REAL(1.0) / BF_P2P_SQRT(r_sq);
                p2p_real_t r_inv3 = r_inv * r_inv * r_inv;
                p2p_real_t f_mag = G * sh_mass[k] * r_inv3;
                ax += f_mag * dx; ay += f_mag * dy; az += f_mag * dz;
            }
            atomicAdd(&grav_acc_x[i], ax);
            atomicAdd(&grav_acc_y[i], ay);
            atomicAdd(&grav_acc_z[i], az);
        }
        __syncthreads();
    }
}


template<bool OUTER, class Compute = p2p_compute_t>
__device__ __forceinline__ void grouped_p2p_source_warp(
    int source, bool target_active, unsigned int target_mask, int orig_i,
    double xi, double yi, double zi, Compute mi,
    Compute& ax, Compute& ay, Compute& az,
    const UnifiedNode* __restrict__ nodes,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const float* __restrict__ mass,
    const int* __restrict__ sorted_indices, int lane,
    double* sh_pos_x, double* sh_pos_y, double* sh_pos_z,
    Compute* sh_mass,
    Compute G, Compute softening_sq, double cutoff_sq,
    const int* gravity_clump_ids = nullptr)
{
    const UnifiedNode& source_node = nodes[source];
    const int source_begin = static_cast<int>(source_node.begin);
    const int source_count = static_cast<int>(source_node.count);
    for (int tile = 0; tile < source_count; tile += 32) {
        const int tile_size = min(32, source_count - tile);
        if (lane < tile_size) {
            const int orig_j = sorted_indices[source_begin + tile + lane];
            sh_pos_x[lane] = pos_x[orig_j];
            sh_pos_y[lane] = pos_y[orig_j];
            sh_pos_z[lane] = pos_z[orig_j];
            sh_mass[lane] = static_cast<Compute>(mass[orig_j]);
        }
        __syncwarp();
        if (__popc(target_mask) <= 4) {
            // Fine shearing-box leaves commonly have only one or a few
            // central targets. Use the otherwise idle lanes to evaluate one
            // source each, then gather in source order into each target lane.
            // The gather deliberately retains the original sequential local
            // arithmetic; a warp reduction would change its summation order.
            unsigned int remaining = target_mask;
            while (remaining != 0u) {
                const int target_lane = __ffs(remaining) - 1;
                remaining &= remaining - 1u;
                const int target_orig = __shfl_sync(0xffffffffu, orig_i, target_lane);
                const double target_x = __shfl_sync(0xffffffffu, xi, target_lane);
                const double target_y = __shfl_sync(0xffffffffu, yi, target_lane);
                const double target_z = __shfl_sync(0xffffffffu, zi, target_lane);
                Compute dax = 0, day = 0, daz = 0;
                bool contributes = lane < tile_size;
                if (contributes) {
                    const int orig_j = sorted_indices[source_begin + tile + lane];
                    contributes = target_orig != orig_j &&
                        !(gravity_clump_ids && gravity_clump_ids[target_orig] > 0 &&
                          gravity_clump_ids[target_orig] == gravity_clump_ids[orig_j]);
                    if (contributes) {
                        const double rdx = (sh_pos_x[lane] - target_x);
                        const double rdy = (sh_pos_y[lane] - target_y);
                        const double rdz = (sh_pos_z[lane] - target_z);
                        contributes = !(cutoff_sq > 0.0 &&
                            rdx * rdx + rdy * rdy + rdz * rdz > cutoff_sq);
                        if (contributes) {
                            compute_gravitational_acceleration_from_delta_unified(
                                static_cast<Compute>(rdx),
                                static_cast<Compute>(rdy),
                                static_cast<Compute>(rdz), sh_mass[lane],
                                softening_sq, G, dax, day, daz);
                        }
                    }
                }
                unsigned int source_mask = __ballot_sync(0xffffffffu, contributes);
                while (source_mask != 0u) {
                    const int j = __ffs(source_mask) - 1;
                    source_mask &= source_mask - 1u;
                    // All lanes participate in every shuffle, including image
                    // lanes and lanes beyond this source tile's last particle.
                    const Compute sx = __shfl_sync(0xffffffffu, dax, j);
                    const Compute sy = __shfl_sync(0xffffffffu, day, j);
                    const Compute sz = __shfl_sync(0xffffffffu, daz, j);
                    if (lane == target_lane) {
                        if constexpr (OUTER) {
                            ax += sx; ay += sy; az += sz;
                        } else {
                            ax += mi * sx; ay += mi * sy; az += mi * sz;
                        }
                    }
                }
            }
        } else if (target_active) {
            for (int j = 0; j < tile_size; ++j) {
                const int orig_j = sorted_indices[source_begin + tile + j];
                if (orig_i == orig_j ||
                    (gravity_clump_ids && gravity_clump_ids[orig_i] > 0 &&
                     gravity_clump_ids[orig_i] == gravity_clump_ids[orig_j])) continue;
                const double rdx = (sh_pos_x[j] - xi);
                const double rdy = (sh_pos_y[j] - yi);
                const double rdz = (sh_pos_z[j] - zi);
                if (cutoff_sq > 0.0 &&
                    rdx * rdx + rdy * rdy + rdz * rdz > cutoff_sq)
                    continue;
                const Compute dx = static_cast<Compute>(rdx);
                const Compute dy = static_cast<Compute>(rdy);
                const Compute dz = static_cast<Compute>(rdz);
                Compute dax, day, daz;
                compute_gravitational_acceleration_from_delta_unified(
                    dx, dy, dz, sh_mass[j], softening_sq, G, dax, day, daz);
                if constexpr (OUTER) {
                    ax += dax; ay += day; az += daz;
                } else {
                    ax += mi * dax; ay += mi * day; az += mi * daz;
                }
            }
        }
        __syncwarp();
    }
}

// Independent source view; targets always retain the local tree and order.
struct GroupedParticleSources {
    const UnifiedNode* nodes;
    const int* children;
    const double *x,*y,*z;
    const float* mass;
    const int* order;
    bool remote;
    const int* target_canonical = nullptr;
};

template<int D, bool OUTER, class Compute = p2p_compute_t>
__device__ __forceinline__ void grouped_p2p_leaf_routes(
    int leaf, int num_nodes, bool target_active, unsigned int target_mask, int orig_i,
    double xi, double yi, double zi, Compute mi,
    Compute& ax, Compute& ay, Compute& az,
    const GroupedInteractionArena& arena, const UnifiedNode* __restrict__ nodes,
    const UnifiedNode* __restrict__ source_nodes,
    const int* __restrict__ children,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const float* __restrict__ mass,
    const int* __restrict__ sorted_indices, int lane,
    double* sh_pos_x, double* sh_pos_y, double* sh_pos_z,
    Compute* sh_mass,
    Compute G, Compute softening_sq, double cutoff_sq,
    const int* gravity_clump_ids)
{
    using T = GroupedStencilTraits<D>;
    int target = leaf;
    // Karras materializes every occupied level. A valid target's ancestry
    // therefore descends exactly one level per record, ending at level 1
    // (which has no grouped record). Bound the walk by that invariant so a
    // stale/corrupt parent cannot hang the GPU or silently omit forces.
    for (int level = nodes[leaf].level; level >= 1; --level) {
        if (target < 0 || target >= num_nodes ||
            !(nodes[target].flags & UNIFIED_NODE_FLAG_VALID) ||
            nodes[target].target_count == 0 || nodes[target].level != level) {
            if (lane == 0)
                printf("BF-FMM invalid P2P target chain: leaf=%d target=%d "
                       "expected_level=%d num_nodes=%d\n",
                       leaf, target, level, num_nodes);
            __trap();
            return;
        }
        if (level == 1) break;
        const GroupedTargetInteractions& work = arena.d_targets[target];
        #pragma unroll
        for (int i = 0; i < T::CROSS_COUNT; ++i) {
            const GroupedSourceRoute route = grouped_cross_route(work, i);
            if (!grouped_route_valid(route) || !(grouped_route_mask(route) & 1u))
                continue;
            grouped_p2p_source_warp<OUTER>(
                grouped_route_node(route), target_active, target_mask, orig_i, xi, yi, zi,
                mi, ax, ay, az, source_nodes, pos_x, pos_y, pos_z, mass,
                sorted_indices, lane, sh_pos_x, sh_pos_y, sh_pos_z, sh_mass,
                G, softening_sq, cutoff_sq, gravity_clump_ids);
        }
        #pragma unroll
        for (int block = 0; block < T::BLOCK_COUNT; ++block) {
            const GroupedSourceRoute route = grouped_block_route(work, block);
            if (!grouped_route_valid(route)) continue;
            const int source_parent = grouped_route_node(route);
            const uint8_t direct_mask = grouped_route_mask(route);
            #pragma unroll
            for (int child = 0; child < T::CHILD_COUNT; ++child) {
                if (!(direct_mask & static_cast<uint8_t>(1u << child))) continue;
                const int source = grouped_child_node<D>(
                    source_nodes, children, source_parent, child);
                if (source < 0) continue;
                grouped_p2p_source_warp<OUTER>(
                    source, target_active, target_mask, orig_i, xi, yi, zi, mi,
                    ax, ay, az, source_nodes, pos_x, pos_y, pos_z, mass,
                    sorted_indices, lane, sh_pos_x, sh_pos_y, sh_pos_z, sh_mass,
                    G, softening_sq, cutoff_sq, gravity_clump_ids);
            }
        }
        target = work.target_parent;
    }
}

template<bool OUTER, class Compute = p2p_compute_t>
__device__ __forceinline__ void fmm_p2p_grouped_leaf_body(
    const GroupedInteractionArena arena,
    const PersistentTreeMetadata* __restrict__ metadata,
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children, int node_capacity,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const float* __restrict__ mass,
    double* __restrict__ out_x, double* __restrict__ out_y,
    double* __restrict__ out_z, const int* __restrict__ sorted_indices,
    Compute G, Compute softening, GroupedParticleSources sources,
    const int* gravity_clump_ids = nullptr)
{
    constexpr int WARPS_PER_BLOCK = 4;
    __shared__ double sh_pos_x[WARPS_PER_BLOCK][32];
    __shared__ double sh_pos_y[WARPS_PER_BLOCK][32];
    __shared__ double sh_pos_z[WARPS_PER_BLOCK][32];
    __shared__ Compute sh_mass[WARPS_PER_BLOCK][32];
    const Compute softening_sq = softening < 0 ? -softening * softening : softening * softening;
    const double cutoff = metadata->cutoff_radius;
    const double cutoff_sq = cutoff > 0.0 ? cutoff * cutoff : 0.0;
    const int target_particle_count = metadata->target_particle_count;
    const UnifiedNode* target_nodes = nodes;
    const int* target_order = sorted_indices;
    const int actual_nodes = min(metadata->num_nodes, node_capacity);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_leaf = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int leaf_stride = gridDim.x * WARPS_PER_BLOCK;
    for (int leaf = first_leaf; leaf < actual_nodes; leaf += leaf_stride) {
        const UnifiedNode& leaf_node = target_nodes[leaf];
        if (!(leaf_node.flags & UNIFIED_NODE_FLAG_VALID) ||
            !(leaf_node.flags & UNIFIED_NODE_FLAG_IS_LEAF) ||
            leaf_node.level < 2 || leaf_node.target_count == 0)
            continue;
        // Image-only leaves are sources, not destinations: the stencil never
        // initializes their grouped records. Mixed leaves still need every
        // lane for cooperative source loads, including inactive target lanes.
        const int begin = static_cast<int>(leaf_node.begin);
        const int count = static_cast<int>(leaf_node.count);
        for (int target_tile = 0; target_tile < count; target_tile += 32) {
            const int local_i = target_tile + lane;
            bool active = local_i < count;
            int orig_i = -1;
            double xi = 0.0, yi = 0.0, zi = 0.0;
            Compute mi = 0;
            if (active) {
                orig_i = target_order[begin + local_i];
                active = orig_i < target_particle_count;
            }
            const unsigned int target_mask = __ballot_sync(0xffffffffu, active);
            if (target_mask == 0u) continue;
            if (active) {
                xi = pos_x[orig_i];
                yi = pos_y[orig_i];
                zi = pos_z[orig_i];
                mi = static_cast<Compute>(mass[orig_i]);
            }
            Compute ax = 0, ay = 0, az = 0;
            if (metadata->active_tree == 2) {
                grouped_p2p_leaf_routes<2, OUTER>(
                    leaf, actual_nodes, active, target_mask, (active && sources.target_canonical) ? sources.target_canonical[orig_i] : (sources.remote ? -1 : orig_i), xi, yi, zi, mi, ax, ay, az, arena,
                    nodes, sources.nodes, sources.children, sources.x, sources.y, sources.z, sources.mass, sources.order,
                    lane, sh_pos_x[warp], sh_pos_y[warp], sh_pos_z[warp],
                    sh_mass[warp], G, softening_sq, cutoff_sq, gravity_clump_ids);
            } else {
                grouped_p2p_leaf_routes<3, OUTER>(
                    leaf, actual_nodes, active, target_mask, (active && sources.target_canonical) ? sources.target_canonical[orig_i] : (sources.remote ? -1 : orig_i), xi, yi, zi, mi, ax, ay, az, arena,
                    nodes, sources.nodes, sources.children, sources.x, sources.y, sources.z, sources.mass, sources.order,
                    lane, sh_pos_x[warp], sh_pos_y[warp], sh_pos_z[warp],
                    sh_mass[warp], G, softening_sq, cutoff_sq, gravity_clump_ids);
            }
            if (active) {
                out_x[orig_i] += static_cast<double>(ax);
                out_y[orig_i] += static_cast<double>(ay);
                out_z[orig_i] += static_cast<double>(az);
            }
        }
    }
}

template<bool OUTER, class Compute = p2p_compute_t>
__global__ void fmm_p2p_grouped_leaf_unified(
    const GroupedInteractionArena arena,
    const PersistentTreeMetadata* __restrict__ metadata,
    const UnifiedNode* __restrict__ nodes,
    const int* __restrict__ children, int node_capacity,
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const float* __restrict__ mass,
    double* __restrict__ out_x, double* __restrict__ out_y,
    double* __restrict__ out_z, const int* __restrict__ sorted_indices,
    Compute G, Compute softening, const int* gravity_clump_ids)
{
    const GroupedParticleSources sources{nodes,children,pos_x,pos_y,pos_z,mass,sorted_indices,false};
    fmm_p2p_grouped_leaf_body<OUTER>(
        arena,metadata,nodes,children,node_capacity,pos_x,pos_y,pos_z,mass,
        out_x,out_y,out_z,sorted_indices,G,softening,sources,gravity_clump_ids);
}

template<bool OUTER>
void launch_p2p_grouped_leaves(
    const fmm::FMMDeviceData& dd, const GroupedInteractionArena& arena,
    const PersistentTreeMetadata* metadata, int node_capacity,
    const double* pos_x, const double* pos_y, const double* pos_z,
    const float* mass, double* out_x, double* out_y, double* out_z,
    double G, double softening, cudaStream_t stream,
    const int* gravity_clump_ids = nullptr)
{
    constexpr int grid = 2048;
    if (softening < 0) {
    fmm_p2p_grouped_leaf_unified<OUTER, double><<<grid, 128, 0, stream>>>(
        arena, metadata, dd.unified_nodes, dd.children_compact, node_capacity,
        pos_x, pos_y, pos_z, mass, out_x, out_y, out_z,
        dd.sorted_to_original, static_cast<double>(G),
        static_cast<double>(softening), gravity_clump_ids);
    } else {
    fmm_p2p_grouped_leaf_unified<OUTER><<<grid, 128, 0, stream>>>(
        arena, metadata, dd.unified_nodes, dd.children_compact, node_capacity,
        pos_x, pos_y, pos_z, mass, out_x, out_y, out_z,
        dd.sorted_to_original, static_cast<p2p_compute_t>(G),
        static_cast<p2p_compute_t>(softening), gravity_clump_ids);
    }
    CUDA_CHECK(cudaGetLastError());
}


__global__ void fmm_p2p_cross_leaf_kernel(
    GroupedInteractionArena arena, const PersistentTreeMetadata* metadata,
    const UnifiedNode* targets, const int* target_children, int capacity,
    const double* x, const double* y, const double* z, const float* mass,
    double* fx, double* fy, double* fz, const int* target_order,
    GroupedParticleSources sources, const int* clump_ids,
    p2p_compute_t G, p2p_compute_t eps) {
    fmm_p2p_grouped_leaf_body<false>(
        arena, metadata, targets, target_children, capacity, x, y, z, mass,
        fx, fy, fz, target_order, G, eps, sources, clump_ids);
}

void launch_p2p_cross_leaves(
    const fmm::FMMDeviceData& target, const GroupedInteractionArena& arena,
    const PersistentTreeMetadata* metadata, int capacity,
    const double* x, const double* y, const double* z, const float* mass,
    double* fx, double* fy, double* fz, const int* target_canonical,
    const fmm::FMMDeviceData& source,
    const double* sx, const double* sy, const double* sz, const float* sm,
    const int* clump_ids, double G, double eps, cudaStream_t stream) {
    const GroupedParticleSources sources{
        source.unified_nodes, source.children_compact, sx, sy, sz, sm,
        source.sorted_to_original, false, target_canonical};
    fmm_p2p_cross_leaf_kernel<<<2048, 128, 0, stream>>>(
        arena, metadata, target.unified_nodes, target.children_compact, capacity,
        x, y, z, mass, fx, fy, fz, target.sorted_to_original, sources, clump_ids,
        static_cast<p2p_compute_t>(G), static_cast<p2p_compute_t>(eps));
    CUDA_CHECK(cudaGetLastError());
}


#include "p2p_remote_kernel.cuh"

} // namespace body_fitted_fmm

#undef p2p_real_t
#undef BF_P2P_REAL
#undef BF_P2P_SQRT

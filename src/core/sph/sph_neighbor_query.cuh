#pragma once
// =============================================================================
// SPH Neighbor Query — Unified Device-Function Abstraction
// =============================================================================
// Provides for_each_sph_neighbor() that dispatches on NeighborData::active_method
// to either CLL 27-cell iteration or LBVH BVH stack traversal.
//
// Usage:
//   for_each_sph_neighbor(i, pi_x, pi_y, pi_z, hi, neighbor,
//       pos_x, pos_y, pos_z, smoothing_length,
//       kernel_cutoff_factor,
//       [&](int32_t j, float dx, float dy, float dz, float dist) {
//           // ... physics computation using captured variables ...
//       });
// =============================================================================

#include "src/core/neighbor/neighbor_types.h"  // provides NeighborData, NeighborMethod (via command_line_interface.h)
#include "src/core/neighbor/lbvh/lbvh_types.h"
#include "src/core/neighbor/sph_pair.cuh"
#include "src/core/sph/sph_device_helpers.cuh"

namespace cudadem {

// =============================================================================
// CLL Path: 27-cell Morton neighborhood iteration
// =============================================================================
// Extracted from existing SPH kernels. Behavior must be bit-identical.
// Uses extern __constant__ dev_neighbor_params for grid/periodicity info.

template<typename Lambda>
__device__ void for_each_sph_neighbor_cll(
    int32_t i,
    float pi_x, float pi_y, float pi_z,
    float hi,
    const NeighborData& neighbor,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ smoothing_length,
    float kernel_cutoff_factor,
    Lambda&& lambda)
{
    float support_radius = 0.5f * (hi + dev_sph_h_max_global) * kernel_cutoff_factor;
    float support_radius_sq = support_radius * support_radius;

    int32_t ix, iy, iz;
    position_to_cell_3d(pi_x, pi_y, pi_z, ix, iy, iz);

    bool periodic_x = dev_neighbor_params.periodic_x;
    bool periodic_y = dev_neighbor_params.periodic_y;
    bool periodic_z = dev_neighbor_params.periodic_z;
    int32_t grid_dim_x = dev_neighbor_params.grid_dim_x;
    int32_t grid_dim_y = dev_neighbor_params.grid_dim_y;
    int32_t grid_dim_z = dev_neighbor_params.grid_dim_z;
    float domain_size_x = dev_neighbor_params.domain_max_x - dev_neighbor_params.domain_min_x;
    float domain_size_y = dev_neighbor_params.domain_max_y - dev_neighbor_params.domain_min_y;
    float domain_size_z = dev_neighbor_params.domain_max_z - dev_neighbor_params.domain_min_z;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int32_t nx = ix + dx;
                int32_t ny = iy + dy;
                int32_t nz = iz + dz;
                float shift_x = 0.0f, shift_y = 0.0f, shift_z = 0.0f;

                if (periodic_x) {
                    if (nx < 0) { nx += grid_dim_x; shift_x = -domain_size_x; }
                    else if (nx >= grid_dim_x) { nx -= grid_dim_x; shift_x = domain_size_x; }
                } else { if (nx < 0 || nx >= grid_dim_x) continue; }

                if (periodic_y) {
                    if (ny < 0) { ny += grid_dim_y; shift_y = -domain_size_y; }
                    else if (ny >= grid_dim_y) { ny -= grid_dim_y; shift_y = domain_size_y; }
                } else { if (ny < 0 || ny >= grid_dim_y) continue; }

                if (periodic_z) {
                    if (nz < 0) { nz += grid_dim_z; shift_z = -domain_size_z; }
                    else if (nz >= grid_dim_z) { nz -= grid_dim_z; shift_z = domain_size_z; }
                } else { if (nz < 0 || nz >= grid_dim_z) continue; }

                int32_t neighbor_cell = static_cast<int32_t>(morton_3d(nx, ny, nz));
                int32_t start = neighbor.cell_start[neighbor_cell];
                int32_t end = neighbor.cell_end[neighbor_cell];
                if (start < 0) continue;

                for (int32_t k = start; k < end; ++k) {
                    int32_t j = neighbor.particle_indices[k];
                    if (i == j) continue;

                    float pj_x = static_cast<float>(pos_x[j]) + shift_x;
                    float pj_y = static_cast<float>(pos_y[j]) + shift_y;
                    float pj_z = static_cast<float>(pos_z[j]) + shift_z;

                    float dx_ij = pi_x - pj_x;
                    float dy_ij = pi_y - pj_y;
                    float dz_ij = pi_z - pj_z;
                    float dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

                    if (dist_sq < support_radius_sq) {
                        float dist = sqrtf(dist_sq);
                        lambda(j, dx_ij, dy_ij, dz_ij, dist);
                    }
                }
            }
        }
    }
}

// =============================================================================
// LBVH Path: BVH stack traversal
// =============================================================================
// Modeled after lbvh_find_contacts_kernel in lbvh_kernels.cu.
// Reuses nodes, aabbs, sorted_indices already in NeighborData.

template<typename Lambda>
__device__ void for_each_sph_neighbor_lbvh(
    int32_t i,
    float pi_x, float pi_y, float pi_z,
    float hi,
    const NeighborData& neighbor,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ smoothing_length,
    float kernel_cutoff_factor,
    Lambda&& lambda)
{
    // LBVH rebuild intentionally has no tree for zero/one-particle domains.
    // Such sparse domains are normal after multi-GPU partitioning, and there
    // cannot be a distinct neighbor to visit.
    if (neighbor.num_particles <= 1) return;

    float query_r = 0.5f * (hi + dev_sph_h_max_global) * kernel_cutoff_factor;

    lbvh::LBVHAABB query_box;
    query_box.lower = make_float4(pi_x - query_r, pi_y - query_r, pi_z - query_r, 0.0f);
    query_box.upper = make_float4(pi_x + query_r, pi_y + query_r, pi_z + query_r, 0.0f);

    constexpr int STACK_SIZE = 128;
    uint32_t stack[STACK_SIZE];
    uint32_t* stack_ptr = stack;
    uint32_t* stack_end = stack + STACK_SIZE;
    *stack_ptr++ = 0;  // Root is ALWAYS 0

#if defined(DEBUG) || defined(_DEBUG)
    int debug_stack_overflow_count = 0;
#endif

    const lbvh::LBVHNode* nodes = neighbor.nodes;
    const lbvh::LBVHAABB* aabbs = neighbor.aabbs;

    while (stack_ptr > stack) {
        uint32_t node = *--stack_ptr;
        uint32_t L_idx = nodes[node].left_idx;
        uint32_t R_idx = nodes[node].right_idx;

        bool L_valid = (L_idx != 0xFFFFFFFF);
        bool R_valid = (R_idx != 0xFFFFFFFF);
        if (!L_valid && !R_valid) continue;

        bool L_overlap = L_valid && lbvh_intersects(query_box, aabbs[L_idx]);
        bool R_overlap = R_valid && lbvh_intersects(query_box, aabbs[R_idx]);

        if (L_overlap) {
            uint32_t obj_j = nodes[L_idx].object_idx;
            if (obj_j != 0xFFFFFFFF) {  // Leaf
                int32_t j = static_cast<int32_t>(obj_j);
                if (i != j) {
                    float pj_x = static_cast<float>(pos_x[j]);
                    float pj_y = static_cast<float>(pos_y[j]);
                    float pj_z = static_cast<float>(pos_z[j]);
                    float dx_ij = pi_x - pj_x;
                    float dy_ij = pi_y - pj_y;
                    float dz_ij = pi_z - pj_z;
                    float dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
                    float query_r_sq = query_r * query_r;
                    if (dist_sq < query_r_sq) {
                        float dist = sqrtf(dist_sq);
                        lambda(j, dx_ij, dy_ij, dz_ij, dist);
                    }
                }
            } else {  // Internal node
                if (stack_ptr < stack_end) {
                    *stack_ptr++ = L_idx;
                }
#if defined(DEBUG) || defined(_DEBUG)
                else { ++debug_stack_overflow_count; }
#endif
            }
        }

        if (R_overlap) {
            uint32_t obj_j = nodes[R_idx].object_idx;
            if (obj_j != 0xFFFFFFFF) {  // Leaf
                int32_t j = static_cast<int32_t>(obj_j);
                if (i != j) {
                    float pj_x = static_cast<float>(pos_x[j]);
                    float pj_y = static_cast<float>(pos_y[j]);
                    float pj_z = static_cast<float>(pos_z[j]);
                    float dx_ij = pi_x - pj_x;
                    float dy_ij = pi_y - pj_y;
                    float dz_ij = pi_z - pj_z;
                    float dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
                    float query_r_sq = query_r * query_r;
                    if (dist_sq < query_r_sq) {
                        float dist = sqrtf(dist_sq);
                        lambda(j, dx_ij, dy_ij, dz_ij, dist);
                    }
                }
            } else {  // Internal node
                if (stack_ptr < stack_end) {
                    *stack_ptr++ = R_idx;
                }
#if defined(DEBUG) || defined(_DEBUG)
                else { ++debug_stack_overflow_count; }
#endif
            }
        }
    }
}

// =============================================================================
// Unified Dispatch
// =============================================================================

template<typename Lambda>
__device__ void for_each_sph_neighbor(
    int32_t i,
    float pi_x, float pi_y, float pi_z,
    float hi,
    const NeighborData& neighbor,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ smoothing_length,
    float kernel_cutoff_factor,
    Lambda&& lambda)
{
    if (neighbor.sph_adjacency_valid) {
        for (uint64_t edge = neighbor.sph_offsets[i];
             edge < neighbor.sph_offsets[i + 1]; ++edge) {
            const int32_t j = neighbor.sph_neighbors[edge];
            const float dx = pi_x - static_cast<float>(pos_x[j]);
            const float dy = pi_y - static_cast<float>(pos_y[j]);
            const float dz = pi_z - static_cast<float>(pos_z[j]);
            if (sph_pair_candidate(dx, dy, dz, hi, smoothing_length[j],
                                   kernel_cutoff_factor, 0.0f)) {
                lambda(j, dx, dy, dz, sqrtf(dx * dx + dy * dy + dz * dz));
            }
        }
        return;
    }
    switch (neighbor.active_method) {
        case NeighborMethod::CELL_LINKED_LIST:
            for_each_sph_neighbor_cll(i, pi_x, pi_y, pi_z, hi, neighbor,
                pos_x, pos_y, pos_z, smoothing_length,
                kernel_cutoff_factor,
                static_cast<Lambda&&>(lambda));
            break;
        case NeighborMethod::LBVH:
            for_each_sph_neighbor_lbvh(i, pi_x, pi_y, pi_z, hi, neighbor,
                pos_x, pos_y, pos_z, smoothing_length,
                kernel_cutoff_factor,
                static_cast<Lambda&&>(lambda));
            break;
        case NeighborMethod::DIRECT:
            // O(N²) fallback — iterate all particles
            {
                float query_r = 0.5f * (hi + dev_sph_h_max_global) * kernel_cutoff_factor;
                float query_r_sq = query_r * query_r;
                for (int32_t j = 0; j < neighbor.num_particles; ++j) {
                    if (i == j) continue;
                    float pj_x = static_cast<float>(pos_x[j]);
                    float pj_y = static_cast<float>(pos_y[j]);
                    float pj_z = static_cast<float>(pos_z[j]);
                    float dx_ij = pi_x - pj_x;
                    float dy_ij = pi_y - pj_y;
                    float dz_ij = pi_z - pj_z;
                    float dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
                    if (dist_sq < query_r_sq) {
                        float dist = sqrtf(dist_sq);
                        lambda(j, dx_ij, dy_ij, dz_ij, dist);
                    }
                }
            }
            break;
    }
}

} // namespace cudadem

#include "rcb_balancer.h"
#include "core/neighbor/neighbor_types.h"
#include "core/acceleration/body_fitted_fmm/unified_node.cuh"
#include "core/acceleration/body_fitted_fmm/task_structs.cuh"
#include "core/acceleration/body_fitted_fmm/persistent_tree_traversal.cuh"
#include "core/acceleration/body_fitted_fmm/grouped_stencil_traits.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace cudadem {

template <class FmmWeightPolicy>
__global__ void compute_weights_kernel(
    const int32_t* __restrict__ num_contacts,
    int32_t n, float alpha, float beta, float gamma,
    FmmWeightPolicy fmm, float* __restrict__ out_weight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float nc = static_cast<float>(num_contacts[idx]);
    out_weight[idx] = alpha + beta * nc + gamma * fmm(idx);
}

// Explicit instantiation for the SP1 policy.
template __global__ void compute_weights_kernel<ZeroFmmWeight>(
    const int32_t*, int32_t, float, float, float, ZeroFmmWeight, float*);

// Fixed-point scale so weights accumulate deterministically via atomicAdd(ull).
__device__ __forceinline__ unsigned long long to_fixed(float w) {
    return static_cast<unsigned long long>(w * 1024.0f + 0.5f);
}

__global__ void axis_histogram_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ weight,
    int32_t n, int axis, int num_bins,
    double lo, double hi,
    double box_min_x, double box_min_y, double box_min_z,
    double box_max_x, double box_max_y, double box_max_z,
    unsigned long long* __restrict__ bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double x = pos_x[idx], y = pos_y[idx], z = pos_z[idx];
    if (!(x >= box_min_x && x < box_max_x &&
          y >= box_min_y && y < box_max_y &&
          z >= box_min_z && z < box_max_z)) return;

    double span = hi - lo;
    if (span <= 0.0) return;
    double coord = (axis == 0) ? x : (axis == 1) ? y : z;
    int b = static_cast<int>((coord - lo) / span * num_bins);
    if (b < 0) b = 0;
    if (b >= num_bins) b = num_bins - 1;
    atomicAdd(&bins[b], to_fixed(weight[idx]));
}

struct AdaptiveHaloPeerCosts { float value[RCB_MAX_PEERS]; };

__global__ void adaptive_base_kernel(float* weights, float* difference, int n, float base,
                                     const uint32_t* halo_mask, AdaptiveHaloPeerCosts peer_cost,
                                     int32_t* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    if (i == n) { difference[i] = 0.0f; return; }
    float halo = 0.0f;
    if (halo_mask) { const uint32_t mask = halo_mask[i]; for (int p = 0; p < RCB_MAX_PEERS; ++p)
        if (mask & (uint32_t(1) << p)) halo += peer_cost.value[p]; }
    const float value = base + halo;
    if (!isfinite(value) || value < 0.0f) {
        atomicExch(status, 1);
        weights[i] = 0.0f;
    } else {
        weights[i] = value;
    }
    difference[i] = 0.0f;
}
__global__ void adaptive_contacts_kernel(const ContactPair* pairs, int count, int owned, float cost, float* weights) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= count) return;
    const ContactPair p = pairs[k];
    const bool owned_i = p.i >= 0 && p.i < owned;
    const bool owned_j = p.j >= 0 && p.j < owned;
    const int owned_endpoints = static_cast<int>(owned_i) + static_cast<int>(owned_j);
    if (owned_endpoints == 0) return;
    // Two force passes launch for every contact.  Preserve both units even at
    // a halo boundary by charging the remote endpoint's early-return work to
    // the local endpoint instead of silently dropping it.
    const float endpoint_cost = cost * (2.0f / static_cast<float>(owned_endpoints));
    if (owned_i) atomicAdd(weights + p.i, endpoint_cost);
    if (owned_j) atomicAdd(weights + p.j, endpoint_cost);
}
__global__ void adaptive_sph_neighbors_kernel(const uint64_t* offsets, int owned,
                                              float cost, float* weights) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < owned) weights[i] += cost *
        (1.0f + static_cast<float>(offsets ? offsets[i+1] - offsets[i] : 0));
}
template <class Task> __global__ void adaptive_tasks_kernel(const Task* tasks, int count,
    const body_fitted_fmm::UnifiedNode* nodes, int num_nodes, int owned,
    float cost, float* difference, int32_t* status) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= count) return;
    const uint32_t target = tasks[k].target;
    if (target >= static_cast<uint32_t>(num_nodes)) { atomicExch(status, 1); return; }
    const auto node = nodes[target]; if (node.count == 0) return;
    if (static_cast<uint64_t>(node.begin) + node.count > static_cast<uint64_t>(owned)) {
        atomicExch(status, 1); return;
    }
    const float share = cost / static_cast<float>(node.count);
    if (!isfinite(share) || share < 0.0f) { atomicExch(status, 1); return; }
    atomicAdd(difference + node.begin, share); atomicAdd(difference + node.begin + node.count, -share);
}
template<int D>
__device__ __forceinline__ void grouped_target_units(
    int target, const body_fitted_fmm::GroupedTargetInteractions& work,
    const body_fitted_fmm::UnifiedNode* nodes, const int* children,
    int& m2l_units, int& p2p_units)
{
    using T = body_fitted_fmm::GroupedStencilTraits<D>;
    const auto& target_node = nodes[target];
    const int q = body_fitted_fmm::grouped_target_child<D>(
        target_node.compact_morton);
    for (int i = 0; i < T::CROSS_COUNT; ++i) {
        const auto route = body_fitted_fmm::grouped_cross_route(work, i);
        if (!body_fitted_fmm::grouped_route_valid(route)) continue;
        (body_fitted_fmm::grouped_route_mask(route) & 1u)
            ? ++p2p_units : ++m2l_units;
    }
    for (int block = 0; block < T::BLOCK_COUNT; ++block) {
        const auto route = body_fitted_fmm::grouped_block_route(work, block);
        if (!body_fitted_fmm::grouped_route_valid(route)) continue;
        const int parent = body_fitted_fmm::grouped_route_node(route);
        const uint8_t direct = body_fitted_fmm::grouped_route_mask(route);
        const uint8_t omit =
            body_fitted_fmm::grouped_block_omit_mask(work, block);
        const auto stencil = T::block(q, block);
        for (int child = 0; child < T::CHILD_COUNT; ++child) {
            const int source = body_fitted_fmm::grouped_child_node<D>(
                nodes, children, parent, child);
            if (source < 0) continue;
            const uint8_t bit = static_cast<uint8_t>(1u << child);
            if (omit & bit) continue;
            if (stencil.far_mask & bit)
                (direct & bit) ? ++p2p_units : ++m2l_units;
            else if ((target_node.flags &
                      body_fitted_fmm::UNIFIED_NODE_FLAG_IS_LEAF) &&
                     (stencil.near_mask & bit))
                ++p2p_units;
        }
    }
}
__global__ void adaptive_grouped_fmm_kernel(
    const body_fitted_fmm::GroupedTargetInteractions* work,
    const body_fitted_fmm::PersistentTreeMetadata* metadata,
    const body_fitted_fmm::UnifiedNode* nodes, const int* children,
    int node_capacity, int owned, float m2l_cost, float p2p_cost,
    float* difference, int32_t* status)
{
    const int target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= node_capacity || target >= metadata->num_nodes) return;
    const auto node = nodes[target];
    if (!(node.flags & body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
        node.level < 2 || node.count == 0) return;
    if (static_cast<uint64_t>(node.begin) + node.count >
        static_cast<uint64_t>(owned)) {
        atomicExch(status, 1); return;
    }
    int m2l_units = 0, p2p_units = 0;
    if (metadata->active_tree == 2)
        grouped_target_units<2>(target, work[target], nodes, children,
                                m2l_units, p2p_units);
    else
        grouped_target_units<3>(target, work[target], nodes, children,
                                m2l_units, p2p_units);
    const float total = m2l_cost * static_cast<float>(m2l_units) +
                        p2p_cost * static_cast<float>(p2p_units);
    const float share = total / static_cast<float>(node.count);
    if (!isfinite(total) || !isfinite(share) || share < 0.0f) {
        atomicExch(status, 1); return;
    }
    atomicAdd(difference + node.begin, share);
    atomicAdd(difference + node.begin + node.count, -share);
}
__global__ void adaptive_node_units_kernel(const uint2* units, const body_fitted_fmm::UnifiedNode* nodes,
    int count, int num_nodes, int owned, float node_cost, float particle_cost,
    float* difference, int32_t* status) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= count) return;
    if (k >= num_nodes) { atomicExch(status, 1); return; }
    const auto record=units[k]; const auto node = nodes[k];
    if (node.count == 0 || (record.x == 0 && record.y == 0)) return;
    if (static_cast<uint64_t>(node.begin) + node.count > static_cast<uint64_t>(owned)) {
        atomicExch(status, 1); return;
    }
    const float total = node_cost * static_cast<float>(record.x)
                      + particle_cost * static_cast<float>(record.y);
    const float share = total / static_cast<float>(node.count);
    if (!isfinite(total) || !isfinite(share) || share < 0.0f) {
        atomicExch(status, 1); return;
    }
    atomicAdd(difference + node.begin, share); atomicAdd(difference + node.begin + node.count, -share);
}
__global__ void adaptive_scatter_kernel(const int* sorted_to_original, const float* prefix,
                                        int n, float* weights, int32_t* status) {
    const int sorted = blockIdx.x * blockDim.x + threadIdx.x; if (sorted >= n) return;
    const int original = sorted_to_original[sorted];
    if (original < 0 || original >= n || !isfinite(prefix[sorted])) {
        atomicExch(status, 1); return;
    }
    weights[original] += prefix[sorted];
}
__global__ void scale_weights_kernel(float* weights, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) weights[i] *= scale;
}

void build_adaptive_rcb_weights(const AdaptiveRCBInput& input, float* weights, float* difference,
                                int32_t* status, cudaStream_t stream) {
    if (input.owned_count <= 0 || !weights || !difference || !status) return;
    constexpr int bs = 256; const int nblocks = (input.owned_count + bs - 1) / bs;
    AdaptiveHaloPeerCosts peer_cost{};
    for (int p = 0; p < RCB_MAX_PEERS; ++p) peer_cost.value[p] = input.halo_peer_cost[p];
    adaptive_base_kernel<<<(input.owned_count + 1 + bs - 1) / bs, bs, 0, stream>>>(weights, difference, input.owned_count,
        input.coefficients.base + input.coefficients.let_receiver * input.receiver_uniform_units,
        input.halo_destination_mask, peer_cost, status);
    if (input.sph_mode) adaptive_sph_neighbors_kernel<<<nblocks,bs,0,stream>>>(
        input.sph_offsets,input.owned_count,input.coefficients.contact_pair,weights);
    else if (input.contact_pairs && input.contact_pair_count > 0) adaptive_contacts_kernel<<<(input.contact_pair_count + bs - 1)/bs,bs,0,stream>>>(input.contact_pairs,input.contact_pair_count,input.owned_count,input.coefficients.contact_pair,weights);
    if (input.fmm.nodes) {
        if (input.fmm.grouped_targets && input.fmm.metadata &&
            input.fmm.children_compact && input.fmm.node_capacity > 0)
            adaptive_grouped_fmm_kernel<<<(input.fmm.node_capacity+bs-1)/bs,bs,0,stream>>>(
                input.fmm.grouped_targets,input.fmm.metadata,input.fmm.nodes,
                input.fmm.children_compact,input.fmm.node_capacity,input.owned_count,
                input.coefficients.m2l,input.coefficients.p2p,difference,status);
        if (input.fmm.m2l_same && input.fmm.m2l_same_count > 0) adaptive_tasks_kernel<<<(input.fmm.m2l_same_count+bs-1)/bs,bs,0,stream>>>(input.fmm.m2l_same,input.fmm.m2l_same_count,input.fmm.nodes,input.fmm.num_nodes,input.owned_count,input.coefficients.m2l,difference,status);
        if (input.fmm.m2l_cross && input.fmm.m2l_cross_count > 0) adaptive_tasks_kernel<<<(input.fmm.m2l_cross_count+bs-1)/bs,bs,0,stream>>>(input.fmm.m2l_cross,input.fmm.m2l_cross_count,input.fmm.nodes,input.fmm.num_nodes,input.owned_count,input.coefficients.m2l,difference,status);
        if (input.fmm.p2p_regular && input.fmm.p2p_regular_count > 0) adaptive_tasks_kernel<<<(input.fmm.p2p_regular_count+bs-1)/bs,bs,0,stream>>>(input.fmm.p2p_regular,input.fmm.p2p_regular_count,input.fmm.nodes,input.fmm.num_nodes,input.owned_count,input.coefficients.p2p,difference,status);
        if (input.fmm.p2p_self && input.fmm.p2p_self_count > 0) adaptive_tasks_kernel<<<(input.fmm.p2p_self_count+bs-1)/bs,bs,0,stream>>>(input.fmm.p2p_self,input.fmm.p2p_self_count,input.fmm.nodes,input.fmm.num_nodes,input.owned_count,input.coefficients.p2p,difference,status);
        if (input.let_source_work && input.let_source_work_count > 0) adaptive_node_units_kernel<<<(input.let_source_work_count+bs-1)/bs,bs,0,stream>>>(input.let_source_work,input.fmm.nodes,input.let_source_work_count,input.fmm.num_nodes,input.owned_count,input.coefficients.let_source_node,input.coefficients.let_source_particle,difference,status);
        thrust::inclusive_scan(thrust::cuda::par.on(stream), difference, difference + input.owned_count, difference);
        if (input.fmm.sorted_to_original) adaptive_scatter_kernel<<<nblocks,bs,0,stream>>>(input.fmm.sorted_to_original,difference,input.owned_count,weights,status);
    }
}

} // namespace cudadem

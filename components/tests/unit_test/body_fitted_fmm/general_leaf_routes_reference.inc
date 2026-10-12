// Frozen general local P2P walker before Stage 3 specialization, 2026-10-08.
// Same walker as baseline 51af3606fac9bc54bb87a2345642a14a615b9e16.
template<int D, bool OUTER, class Compute = p2p_compute_t>
__device__ __forceinline__ void frozen_grouped_p2p_leaf_routes(
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
    Compute G, Compute softening_sq, double cutoff_sq)
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
                G, softening_sq, cutoff_sq);
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
                    G, softening_sq, cutoff_sq);
            }
        }
        target = work.target_parent;
    }
}

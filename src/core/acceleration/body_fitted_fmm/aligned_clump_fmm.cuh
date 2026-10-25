#pragma once
// Included by the manager after its shared operators. Each component owns its
// topology, upward moments and target locals; only immutable G tables are shared.

struct AlignedComponentView {
    const PersistentTreeContext* context = nullptr;
    const fmm::MultipoleExpansion* moments = nullptr;
    int clump_id = 0;
};

struct AlignedGreenPredicate {
    const int* ids;
    const int* large;
    __device__ bool operator()(int i) const {
        return ids[i] <= 0 || large[ids[i] - 1] == 0;
    }
};

__global__ void aligned_pack_component(
    int n, int begin, bool green, int* canonical,
    const double* sx, const double* sy, const double* sz, const float* sm,
    double* x, double* y, double* z, float* mass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int original = green ? canonical[i] : begin + i;
    canonical[i] = original;
    x[i] = sx[original]; y[i] = sy[original]; z[i] = sz[original]; mass[i] = sm[original];
}

// counters: checked mappings, unequal local IDs, alignment errors, audited
// excluded-source rows, source coefficient violations, canonical index errors.
__global__ void aligned_map_component(
    const PersistentTreeContext* component, const PersistentTreeContext* united,
    int* mapping, unsigned long long* counters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= component->metadata.num_nodes) return;
    const auto node = component->dd.unified_nodes[i];
    const int j = persistent_morton_lookup<3>(united, node.level, node.compact_morton);
    mapping[i] = j;
    atomicAdd(counters, 1ull);
    if (j < 0) { atomicAdd(counters + 2, 1ull); return; }
    const auto other = united->dd.unified_nodes[j];
    if (node.level != other.level || node.center.x != other.center.x ||
        node.center.y != other.center.y || node.center.z != other.center.z)
        atomicAdd(counters + 2, 1ull);
    if (i != j) atomicAdd(counters + 1, 1ull);
}

__global__ void aligned_add_component(
    const PersistentTreeContext* component, const int* mapping,
    const fmm::MultipoleExpansion* moments, fmm::MultipoleExpansion* sum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= component->metadata.num_nodes || mapping[i] < 0) return;
    // One component has one node per key. Components launch serially, so each
    // output has one writer and the accumulation order is reproducible.
    auto& target = sum[mapping[i]];
    for (int q = 0; q < fmm::FMM_ACTIVE_COEFFS; ++q)
        target.coeffs[q] += moments[i].coeffs[q];
}

__global__ void aligned_audit_excluded_sources(
    const PersistentTreeContext* united, const AlignedComponentView* components,
    int component_count, int excluded_id, const fmm::MultipoleExpansion* sum,
    unsigned long long* counters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= united->metadata.num_nodes) return;
    const auto node = united->dd.unified_nodes[i];
    double expected[fmm::FMM_ACTIVE_COEFFS] = {};
    for (int c = 0; c < component_count; ++c) {
        const auto source = components[c];
        if (source.clump_id == excluded_id) continue;
        const int j = persistent_morton_lookup<3>(source.context, node.level, node.compact_morton);
        if (j < 0) continue;
        for (int q = 0; q < fmm::FMM_ACTIVE_COEFFS; ++q)
            expected[q] += source.moments[j].coeffs[q];
    }
    bool valid = true;
    for (int q = 0; q < fmm::FMM_ACTIVE_COEFFS; ++q)
        valid = valid && expected[q] == sum[i].coeffs[q];
    atomicAdd(counters + 3, 1ull);
    if (!valid) atomicAdd(counters + 4, 1ull);
}

__global__ void aligned_scatter_component(
    int n, int total, const int* canonical,
    const double* fx, const double* fy, const double* fz,
    double* ox, double* oy, double* oz, unsigned long long* counters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int j = canonical[i];
    if (j < 0 || j >= total) { atomicAdd(counters + 5, 1ull); return; }
    ox[j] = fx[i]; oy[j] = fy[i]; oz[j] = fz[i];
}

template<class T> static void aligned_allocate(T*& pointer, size_t count) {
    CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
}

struct AlignedClumpTree {
    PersistentTreeRuntime runtime{};
    GroupedInteractionArena arena{};
    fmm::FMMDeviceData dd{};
    fmm::FMMHostData hd{};
    double *x = nullptr, *y = nullptr, *z = nullptr;
    float* mass = nullptr;
    double *fx = nullptr, *fy = nullptr, *fz = nullptr;
    int *canonical = nullptr, *union_mapping = nullptr;
    int capacity = 0, node_capacity = 0, depth = 0;
    int threshold_leaf = 0, threshold_internal = 0;
    bool source_only = false;

    AlignedClumpTree(int count, const SimConfig& config, bool only_source)
        : capacity(count), depth(config.acceleration.unified_fmm.tree_depth),
          threshold_leaf(config.acceleration.unified_fmm.direct_threshold_leaf),
          threshold_internal(config.acceleration.unified_fmm.direct_threshold_internal),
          source_only(only_source) {
        node_capacity = fixed_depth_node_upper_bound(capacity, depth, 3);
        aligned_allocate(dd.level_start, fmm::FMM_MAX_LEVELS);
        aligned_allocate(dd.level_count, fmm::FMM_MAX_LEVELS);
        aligned_allocate(dd.level_list, node_capacity);
        aligned_allocate(dd.sorted_to_original, capacity);
        aligned_allocate(dd.particle_to_leaf, capacity);
        aligned_allocate(dd.unified_nodes, node_capacity);
        aligned_allocate(dd.children_compact, size_t(node_capacity) * 8);
        aligned_allocate(dd.multipoles, node_capacity);
        if (!source_only) {
            aligned_allocate(dd.multipoles_dbl, size_t(node_capacity) * fmm::FMM_ACTIVE_COEFFS);
            aligned_allocate(dd.local_expansions, node_capacity);
            aligned_allocate(x, capacity); aligned_allocate(y, capacity); aligned_allocate(z, capacity);
            aligned_allocate(mass, capacity);
            aligned_allocate(fx, capacity); aligned_allocate(fy, capacity); aligned_allocate(fz, capacity);
            aligned_allocate(canonical, capacity); aligned_allocate(union_mapping, node_capacity);
        }
        hd.sorted_indices = dd.sorted_to_original;
        dd.num_nodes = node_capacity;
        dd.max_level = depth;
        initialize_persistent_tree_runtime(runtime, dd, arena, capacity, depth, 3,
            config.acceleration.unified_fmm.theta_far, threshold_leaf, threshold_internal);
    }

    ~AlignedClumpTree() {
        shutdown_persistent_tree_runtime(runtime);
        arena.free();
        cudaFree(dd.level_start); cudaFree(dd.level_count); cudaFree(dd.level_list);
        cudaFree(dd.sorted_to_original); cudaFree(dd.particle_to_leaf);
        cudaFree(dd.unified_nodes); cudaFree(dd.children_compact); cudaFree(dd.multipoles);
        cudaFree(dd.multipoles_dbl); cudaFree(dd.local_expansions);
        cudaFree(x); cudaFree(y); cudaFree(z); cudaFree(mass);
        cudaFree(fx); cudaFree(fy); cudaFree(fz); cudaFree(canonical); cudaFree(union_mapping);
    }
    AlignedClumpTree(const AlignedClumpTree&) = delete;
    AlignedClumpTree& operator=(const AlignedClumpTree&) = delete;

    bool fits(int n, const SimConfig& config) const {
        return n <= capacity && depth == config.acceleration.unified_fmm.tree_depth &&
            threshold_leaf == config.acceleration.unified_fmm.direct_threshold_leaf &&
            threshold_internal == config.acceleration.unified_fmm.direct_threshold_internal;
    }

    PersistentTreeMetadata build(
        int n, const double* px, const double* py, const double* pz,
        double4 root, int step, double time, cudaStream_t stream) {
        dd.num_particles = n;
        submit_persistent_tree_runtime(runtime, px, py, pz, n, source_only ? 0 : n,
            step, 3, true, root.x, root.y, root.z, root.w, 0.0, stream, time, false, false);
        PersistentTreeMetadata metadata{};
        CUDA_CHECK(cudaMemcpyAsync(&metadata, &runtime.device_context->metadata,
            sizeof(metadata), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (metadata.error || metadata.num_nodes <= 0 || metadata.num_nodes > node_capacity)
            throw std::runtime_error("aligned clump Karras build failed or exceeded capacity");
        dd.num_nodes = metadata.num_nodes;
        return metadata;
    }
};

struct AlignedClumpPool {
    std::unique_ptr<AlignedClumpTree> united;
    std::vector<std::unique_ptr<AlignedClumpTree>> components;
    BodyFittedFMMDeviceData operators{};
    unsigned long long* counters = nullptr;
    AlignedComponentView* views = nullptr;
    int view_capacity = 0;
    void* select_temp = nullptr;
    size_t select_capacity = 0;
    int* selected_count = nullptr;
    uint64_t reserved_estimate = 0;
    ~AlignedClumpPool() {
        components.clear(); united.reset();
        cudaFree(operators.d_G_full); cudaFree(counters); cudaFree(views);
        cudaFree(select_temp); cudaFree(selected_count);
    }
};

static DeviceStateArray<std::unique_ptr<AlignedClumpPool>> aligned_clump_pools;
static DeviceStateArray<AlignedClumpGravityDiagnostics> aligned_clump_reports;
static DeviceStateArray<uint64_t> aligned_clump_epochs;

static void release_aligned_clump_pool(int gpu) {
    aligned_clump_reports[gpu] = {};
    if (!aligned_clump_pools[gpu]) return;
    int saved = 0;
    CUDA_CHECK(cudaGetDevice(&saved));
    CUDA_CHECK(cudaSetDevice(gpu));
    aligned_clump_pools[gpu].reset();
    CUDA_CHECK(cudaSetDevice(saved));
}

AlignedClumpGravityDiagnostics UnifiedFMMDispatcher::aligned_clump_diagnostics() {
    return aligned_clump_reports[current_gpu()];
}
void UnifiedFMMDispatcher::reset_aligned_clump_diagnostics() {
    const int gpu = current_gpu();
    aligned_clump_reports[gpu] = {};
    // Component contexts have a separate view. Never expose a previous
    // ordinary-tree epoch as the latest aligned or direct evaluation.
    invalidate_workload_view(gpu);
}

void UnifiedFMMDispatcher::compute_aligned_clumps(
    SimulationState& members, const SimConfig& config,
    int clump_count, int member_begin, const int* segment_begin,
    const int* segment_end, const int* clump_ids, const int* large_flags,
    const double* root_geometry, cudaStream_t stream) {
    const int gpu = current_gpu(), total = members.particles.num_particles;
    if (config.physics != SimulationPhysics::DEM || config.shear_box.enabled ||
        config.acceleration.unified_fmm.tree_dimension != 3 ||
        config.acceleration.unified_fmm.adaptive_depth.enabled || distributed_fmm_[gpu].evaluating)
        throw std::invalid_argument("aligned clump trees require fixed-depth single-GPU 3D DEM");
    std::vector<int> large(clump_count), begin(clump_count), end(clump_count);
    double geometry[4];
    CUDA_CHECK(cudaMemcpyAsync(large.data(), large_flags, clump_count * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(begin.data(), segment_begin, clump_count * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(end.data(), segment_end, clump_count * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(geometry, root_geometry, sizeof(geometry), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double4 root = make_double4(geometry[1], geometry[2], geometry[3], geometry[0]);
    struct Branch { int id, begin, count; };
    std::vector<Branch> branches;
    int green_count = total, cursor = 0, large_count = 0;
    for (int c = 0; c < clump_count; ++c) {
        if (begin[c] != cursor || end[c] <= begin[c] || member_begin + end[c] > total)
            throw std::runtime_error("invalid canonical aligned clump segment partition");
        cursor = end[c];
        if (large[c]) {
            ++large_count; green_count -= end[c] - begin[c];
            branches.push_back({c + 1, member_begin + begin[c], end[c] - begin[c]});
        }
    }
    if (member_begin + cursor != total || large_count == 0 || green_count < 0)
        throw std::runtime_error("aligned clump dispatch requires a complete large-member partition");
    if (green_count) branches.insert(branches.begin(), {0, 0, green_count});
    const int depth = config.acceleration.unified_fmm.tree_depth;

    // Conservative preflight includes every live pool allocation, dense maps,
    // sort workspaces and graph headroom. Failure is explicit, never a hidden
    // substitution of direct summation for this aligned-tree implementation.
    auto& owner = aligned_clump_pools[gpu];
    const int union_capacity = owner && owner->united && owner->united->fits(total, config)
        ? owner->united->capacity : total;
    uint64_t estimate = estimate_adaptive_fmm_memory(union_capacity,
        fixed_depth_node_upper_bound(union_capacity, depth, 3), depth, 3).total();
    for (size_t i = 0; i < branches.size(); ++i) {
        const int count = branches[i].count;
        const int retained = owner && i < owner->components.size() && owner->components[i] &&
            owner->components[i]->fits(count, config) ? owner->components[i]->capacity : count;
        estimate += estimate_adaptive_fmm_memory(retained,
            fixed_depth_node_upper_bound(retained, depth, 3), depth, 3).total();
    }
    estimate += 32ull * 1024 * 1024; // shared G table, graph and audit reserve
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const uint64_t available = std::min<uint64_t>(total_bytes,
        uint64_t(free_bytes) + (owner ? owner->reserved_estimate : 0));
    if (estimate > available * 3 / 4)
        throw std::runtime_error("aligned clump tree memory preflight exceeds 75% available budget");
    if (!owner) {
        owner = std::make_unique<AlignedClumpPool>();
        load_full_g_table(owner->operators);
        aligned_allocate(owner->counters, 6);
        aligned_allocate(owner->selected_count, 1);
    }
    auto& pool = *owner;
    auto reserve_tree = [&](std::unique_ptr<AlignedClumpTree>& tree, int n, bool source_only) {
        if (tree && !tree->fits(n, config)) tree.reset();
        if (!tree) tree = std::make_unique<AlignedClumpTree>(n, config, source_only);
    };
    reserve_tree(pool.united, total, true);
    pool.components.resize(branches.size());
    for (size_t i = 0; i < branches.size(); ++i)
        reserve_tree(pool.components[i], branches[i].count, false);
    pool.reserved_estimate = estimate;
    if (pool.view_capacity < int(branches.size())) {
        cudaFree(pool.views); pool.views = nullptr;
        aligned_allocate(pool.views, branches.size()); pool.view_capacity = branches.size();
    }
    CUDA_CHECK(cudaMemsetAsync(pool.counters, 0, 6 * sizeof(unsigned long long), stream));

    AlignedClumpGravityDiagnostics report{};
    report.epoch = ++aligned_clump_epochs[gpu]; report.used = true;
    report.component_count = branches.size(); report.large_count = large_count;
    report.green_members = green_count; report.live_members = total; report.depth = depth;
    report.root_min_x = root.x; report.root_min_y = root.y; report.root_min_z = root.z; report.root_size = root.w;
    report.reserved_bytes = estimate;
    report.member_force_x = members.particles.force_x;
    report.member_force_y = members.particles.force_y;
    report.member_force_z = members.particles.force_z;
    report.member_force_count = total;
    auto validate_metadata = [&](const PersistentTreeMetadata& metadata) {
        ++report.alignment_checked_count;
        if (metadata.domain_min_x != root.x || metadata.domain_min_y != root.y ||
            metadata.domain_min_z != root.z || metadata.root_cell_size != root.w ||
            metadata.evaluation_depth != depth) ++report.alignment_violation_count;
    };
    const auto union_metadata = pool.united->build(total, members.particles.pos_x,
        members.particles.pos_y, members.particles.pos_z, root,
        int(members.current_step), members.current_time, stream);
    validate_metadata(union_metadata);
    report.source_node_count = union_metadata.num_nodes;
    std::vector<AlignedComponentView> views;
    for (size_t c = 0; c < branches.size(); ++c) {
        auto& tree = *pool.components[c]; const auto branch = branches[c];
        if (branch.id == 0) {
            size_t bytes = 0;
            CUDA_CHECK(cub::DeviceSelect::If(nullptr, bytes, thrust::counting_iterator<int>(0),
                tree.canonical, pool.selected_count, total, AlignedGreenPredicate{clump_ids, large_flags}, stream));
            if (bytes > pool.select_capacity) {
                cudaFree(pool.select_temp); pool.select_temp = nullptr;
                CUDA_CHECK(cudaMalloc(&pool.select_temp, bytes)); pool.select_capacity = bytes;
            }
            CUDA_CHECK(cub::DeviceSelect::If(pool.select_temp, pool.select_capacity,
                thrust::counting_iterator<int>(0), tree.canonical, pool.selected_count,
                total, AlignedGreenPredicate{clump_ids, large_flags}, stream));
            int selected = 0;
            CUDA_CHECK(cudaMemcpyAsync(&selected, pool.selected_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (selected != branch.count) throw std::runtime_error("aligned green member selection count mismatch");
        }
        aligned_pack_component<<<(branch.count + 127) / 128, 128, 0, stream>>>(
            branch.count, branch.begin, branch.id == 0, tree.canonical,
            members.particles.pos_x, members.particles.pos_y, members.particles.pos_z, members.particles.mass,
            tree.x, tree.y, tree.z, tree.mass);
        const auto metadata = tree.build(branch.count, tree.x, tree.y, tree.z, root,
            int(members.current_step), members.current_time, stream);
        validate_metadata(metadata); report.target_node_count += metadata.num_nodes;
        SimulationState component;
        component.particles.num_particles = branch.count;
        component.particles.pos_x = tree.x; component.particles.pos_y = tree.y; component.particles.pos_z = tree.z;
        component.particles.mass = tree.mass;
        fmm_compute_all_multipoles_unified(component, config, tree.hd, tree.dd, stream);
        aligned_map_component<<<(tree.dd.num_nodes + 127) / 128, 128, 0, stream>>>(
            tree.runtime.device_context, pool.united->runtime.device_context, tree.union_mapping, pool.counters);
        rebuild_cross_context_stencil(tree.runtime, pool.united->runtime, stream);
        PersistentTreeMetadata cross{};
        CUDA_CHECK(cudaMemcpyAsync(&cross, &tree.runtime.device_context->metadata, sizeof(cross), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (cross.error) throw std::runtime_error("aligned cross-context stencil emission failed");
        report.m2l_route_count += uint64_t(cross.task_counts[0]) + cross.task_counts[1];
        report.p2p_route_count += uint64_t(cross.task_counts[2]) + cross.task_counts[3];
        views.push_back({tree.runtime.device_context, tree.dd.multipoles, branch.id});
    }
    CUDA_CHECK(cudaMemcpyAsync(pool.views, views.data(), views.size() * sizeof(views[0]), cudaMemcpyHostToDevice, stream));
    const double G = config.acceleration.unified_fmm.gravitational_constant;
    const double eps = config.particle.radius_min * config.acceleration.unified_fmm.softening_factor;
    for (size_t c = 0; c < branches.size(); ++c) {
        auto& tree = *pool.components[c];
        auto& united = *pool.united;
        CUDA_CHECK(cudaMemsetAsync(united.dd.multipoles, 0,
            size_t(union_metadata.num_nodes) * sizeof(fmm::MultipoleExpansion), stream));
        for (size_t s = 0; s < branches.size(); ++s) {
            if (branches[c].id > 0 && branches[s].id == branches[c].id) continue;
            const auto& source = *pool.components[s];
            aligned_add_component<<<(source.dd.num_nodes + 127) / 128, 128, 0, stream>>>(
                source.runtime.device_context, source.union_mapping, source.dd.multipoles, united.dd.multipoles);
        }
        if (branches[c].id > 0)
            aligned_audit_excluded_sources<<<(united.dd.num_nodes + 127) / 128, 128, 0, stream>>>(
                united.runtime.device_context, pool.views, branches.size(), branches[c].id,
                united.dd.multipoles, pool.counters);
        zero_active_local_expansions(tree.dd, &tree.runtime.device_context->metadata, stream);
        run_m2l_grouped_targets(tree.arena, &tree.runtime.device_context->metadata,
            tree.dd.unified_nodes, tree.dd.children_compact, tree.node_capacity,
            united.dd.multipoles, pool.operators.d_G_full, tree.dd.local_expansions, stream,
            united.dd.unified_nodes, united.dd.children_compact);
        const auto* root_size = &tree.runtime.device_context->metadata.root_cell_size;
        run_l2l_pass_unified(tree.dd, root_size, stream);
        const size_t bytes = size_t(branches[c].count) * sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(tree.fx, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(tree.fy, 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(tree.fz, 0, bytes, stream));
        run_l2p_evaluation_unified(tree.dd, tree.x, tree.y, tree.z, tree.mass,
            tree.fx, tree.fy, tree.fz, tree.dd.particle_to_leaf, branches[c].count, G, root_size, stream);
        launch_p2p_cross_leaves(tree.dd, tree.arena, &tree.runtime.device_context->metadata,
            tree.node_capacity, tree.x, tree.y, tree.z, tree.mass, tree.fx, tree.fy, tree.fz,
            tree.canonical, united.dd, members.particles.pos_x, members.particles.pos_y,
            members.particles.pos_z, members.particles.mass, clump_ids, G, eps, stream);
        aligned_scatter_component<<<(branches[c].count + 127) / 128, 128, 0, stream>>>(
            branches[c].count, total, tree.canonical, tree.fx, tree.fy, tree.fz,
            members.particles.force_x, members.particles.force_y, members.particles.force_z, pool.counters);
    }
    unsigned long long counters[6] = {};
    CUDA_CHECK(cudaMemcpyAsync(counters, pool.counters, sizeof(counters), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    report.index_remap_checked_count = counters[0]; report.context_remap_count = counters[1];
    report.alignment_violation_count += counters[2];
    report.self_exclusion_checked_count = counters[3];
    report.self_source_coefficient_violations = counters[4];
    report.self_source_route_count = counters[4];
    if (report.alignment_violation_count || report.self_source_route_count || counters[5])
        throw std::runtime_error("aligned clump geometry, source exclusion or canonical-index audit failed");
    aligned_clump_reports[gpu] = report;
}

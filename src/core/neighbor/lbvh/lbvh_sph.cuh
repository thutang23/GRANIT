#pragma once
// Included inside cudadem::lbvh after the common CUDA/RT lifecycle helpers.
// SPH keeps its gather-style physics kernels. Both backends emit unique pairs,
// which are expanded once into directed CSR and filtered at every evaluation.

__global__ void sph_find_pairs_kernel(
    const double* x, const double* y, const double* z, const float* h,
    const LBVHNode* nodes, const LBVHAABB* boxes, const uint32_t* order,
    ContactPair* pairs, unsigned long long* count, int32_t capacity,
    int32_t n, float cutoff, float skin, int32_t* overflow) {
  const int32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= n) return;
  const uint32_t i = order[rank];
  const float px = float(x[i]), py = float(y[i]), pz = float(z[i]);
  // Leaves contain the full k*h extent. A half-support query is sufficient
  // for k*(hi+hj)/2; use the full support here as a conservative broad phase
  // independent of the rounded half-sum in the exact SPH predicate.
  // Fast-math squared distances can flush to zero even for distinct centers.
  // Cover the same absolute underflow region as the RT proxy bounds.
  const float r = nextafterf((cutoff * h[i] + skin) * (1.0f + 8.0f * FLT_EPSILON)
                             + 2.0f * sqrtf(FLT_MIN), CUDART_INF_F);
  LBVHAABB query;
  query.lower = make_float4(nextafterf(px-r, -CUDART_INF_F),
                           nextafterf(py-r, -CUDART_INF_F),
                           nextafterf(pz-r, -CUDART_INF_F), 0);
  query.upper = make_float4(nextafterf(px+r, CUDART_INF_F),
                           nextafterf(py+r, CUDART_INF_F),
                           nextafterf(pz+r, CUDART_INF_F), 0);
  uint32_t stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top) {
    const LBVHNode node = nodes[stack[--top]];
    const uint32_t children[2] = {node.left_idx, node.right_idx};
    for (int side = 0; side < 2; ++side) {
      const uint32_t child = children[side];
      if (child == 0xFFFFFFFF || !lbvh_intersects(query, boxes[child])) continue;
      const uint32_t j = nodes[child].object_idx;
      if (j != 0xFFFFFFFF) {
        if (j <= i) continue;
        if (sph_pair_candidate(px-float(x[j]), py-float(y[j]), pz-float(z[j]),
                               h[i], h[j], cutoff, skin)) {
          const unsigned long long edge = atomicAdd(count, 1ULL);
          if (edge < static_cast<unsigned long long>(capacity)) pairs[edge] = ContactPair(i, j);
        }
      } else if (top < 128) {
        stack[top++] = child;
      } else {
        atomicExch(overflow, 1);
      }
    }
  }
}

__global__ void sph_count_rows_kernel(const ContactPair* pairs, int32_t count,
                                      uint64_t* counts) {
  const int32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= count) return;
  const ContactPair p = pairs[edge];
  atomicAdd(reinterpret_cast<unsigned long long*>(counts + p.i), 1ULL);
  atomicAdd(reinterpret_cast<unsigned long long*>(counts + p.j), 1ULL);
}

__global__ void sph_fill_rows_kernel(const ContactPair* pairs, int32_t count,
                                     uint64_t* cursor, int32_t* neighbors) {
  const int32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= count) return;
  const ContactPair p = pairs[edge];
  const auto a = atomicAdd(reinterpret_cast<unsigned long long*>(cursor + p.i), 1ULL);
  const auto b = atomicAdd(reinterpret_cast<unsigned long long*>(cursor + p.j), 1ULL);
  neighbors[a] = p.j;
  neighbors[b] = p.i;
}

__global__ void sph_cache_motion_kernel(
    const double* x, const double* y, const double* z, const float* h,
    const int32_t* ids, const int32_t* parts, const LBVHCacheParticle* reference,
    int32_t* max_motion_bits, int32_t n, float cutoff, float skin) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float bound_sq = 0.0f;
  if (i < n) {
    const auto old = reference[i];
    const double px = float(x[i]), py = float(y[i]), pz = float(z[i]);
    const double dx = px - old.x, dy = py - old.y, dz = pz - old.z;
    // A strict skin test based only on real displacement is insufficient for
    // the rounded sqrt predicate: a pair can cross both rounded cutoffs while
    // moving just less than skin. Reserve a conservative error allowance for
    // subtraction, norm, cutoff+skin rounding, and fast-math underflow at both
    // evaluations. Each endpoint contributes its own allowance.
    const double scale = fabs(px) + fabs(py) + fabs(pz)
        + fabs(double(old.x)) + fabs(double(old.y)) + fabs(double(old.z))
        + double(cutoff) * double(h[i]) + double(skin);
    const double margin = 64.0 * double(FLT_EPSILON) * scale
        + 4.0 * sqrt(double(FLT_MIN));
    const double bound = sqrt(dx*dx + dy*dy + dz*dz) + margin;
    bound_sq = __double2float_ru(bound * bound);
    if (!isfinite(bound_sq) || old.global_id != (ids ? ids[i] : i) ||
        old.clump_id != (parts ? parts[i] : 0) || old.support_extent != h[i])
      bound_sq = CUDART_INF_F;
  }
  using Reduce = cub::BlockReduce<float, 256>;
  __shared__ typename Reduce::TempStorage storage;
  const float block_max = Reduce(storage).Reduce(bound_sq, cub::Max());
  if (threadIdx.x == 0) atomicMax(max_motion_bits, __float_as_int(block_max));
}

static void allocate_sph_storage(LBVHGPUState& gpu) {
  if (gpu.sph_offsets) return;
  CUDA_CHECK(cudaMalloc(&gpu.sph_offsets, (size_t(gpu.capacity)+1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&gpu.sph_cursors, (size_t(gpu.capacity)+1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&gpu.sph_pair_count, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&gpu.sph_overflow, sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&gpu.sph_pairs.num_contacts, sizeof(int32_t)));
  // Dense SPH domains can exceed DEM contact estimates by orders of magnitude.
  // The count/retry policy below grows this buffer without dropping candidates.
  ensure_contact_capacity(gpu.sph_pairs, std::max(1024, gpu.capacity));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, gpu.sph_scan_bytes,
      gpu.sph_cursors, gpu.sph_offsets, gpu.capacity + 1));
  CUDA_CHECK(cudaMalloc(&gpu.sph_scan_temp, gpu.sph_scan_bytes));
}

static void publish_sph_adjacency(LBVHGPUState& gpu, SimulationState& state) {
  state.neighbor.num_particles = state.particles.num_particles;
  state.neighbor.sph_offsets = gpu.sph_offsets;
  state.neighbor.sph_neighbors = gpu.sph_neighbors;
  state.neighbor.sph_adjacency_valid = true;
  state.neighbor.morton_codes_64 = nullptr;
  state.neighbor.morton_data_valid = false;
  state.neighbor.rebuild_generation = gpu.current_generation;
}

void LBVHSearchImpl::find_sph_neighbors(SimulationState& state, const SimConfig& config,
                                       cudaStream_t stream) {
  auto& gpu = get_gpu_state();
  state.neighbor.sph_adjacency_valid = false;
  if (!gpu.initialized) throw std::runtime_error("SPH LBVH search is not initialized");
  const int32_t n = state.particles.num_particles;
  if (n < 0 || state.particles.num_local < 0 || state.particles.num_local > n)
    throw std::runtime_error("SPH LBVH invalid particle ownership layout");
  select_backend(gpu, state, config);
  // Retain the existing distributed/image traversal paths outside phase one.
  if (config.multigpu.enabled || state.particles.num_local != n ||
      extract_pbc_config(config).is_enabled() || config.shear_box.enabled) {
    rebuild(state, config, stream);
    return;
  }
  if (config.physics != SimulationPhysics::SPH)
    throw std::runtime_error("find_sph_neighbors requires SPH physics");
  if (n > 0 && !state.particles.smoothing_length)
    throw std::runtime_error("SPH neighbor search requires smoothing lengths");
  if (!(config.sph.kernel_cutoff_factor > 0.0f) ||
      !std::isfinite(config.sph.kernel_cutoff_factor))
    throw std::runtime_error("SPH neighbor cutoff must be finite and positive");
  if (n > gpu.capacity) {
    const auto stats = gpu.stats;
    const bool timing = gpu.backend_timing_enabled;
    shutdown(state);
    initialize(config, state);
    const double init_ms = gpu.stats.backend_init_time_ms;
    gpu.stats = stats;
    gpu.stats.backend_init_time_ms += init_ms;
    gpu.backend_timing_enabled = timing;
  }
  allocate_sph_storage(gpu);
  const bool layout_changed = gpu.tracked_num_particles != n || gpu.tracked_num_local != n;
  if (n <= 1) {
    CUDA_CHECK(cudaMemsetAsync(gpu.sph_offsets, 0, (size_t(n)+1)*sizeof(uint64_t), stream));
    gpu.sph_ready = gpu.sph_cache_valid = false;
    gpu.tracked_num_particles = gpu.tracked_num_local = n;
    gpu.stats.num_contacts_found = 0;
    publish_sph_adjacency(gpu, state);
    return;
  }
  const int threads = config.gpu.block_size;
  const int blocks = (n + threads - 1) / threads;
  const int frequency = std::max(1, config.neighbor.update_frequency);
  const float skin = config.neighbor.verlet_buffer_enabled && frequency > 1
      ? config.neighbor.verlet_skin : 0.0f;
  if (!(skin >= 0.0f) || !std::isfinite(skin))
    throw std::runtime_error("SPH Verlet skin must be finite and nonnegative");
  const auto start = std::chrono::steady_clock::now();
  bool full = !gpu.sph_ready || layout_changed || frequency == 1 ||
              ++gpu.steps_since_rebuild >= frequency;
  // Topology quality may use a current displacement measurement. Bounds are
  // always refreshed before querying, so this threshold never certifies a list.
  if (!full) {
    lbvh_displacement_check_kernel<<<blocks, threads, 0, stream>>>(
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        gpu.snapshot_pos_x, gpu.snapshot_pos_y, gpu.snapshot_pos_z,
        gpu.d_disp_scratch, n);
    CUDA_CHECK(cub::DeviceReduce::Max(gpu.d_cub_disp_temp, gpu.cub_disp_temp_bytes,
                                     gpu.d_disp_scratch, gpu.d_max_disp_sq, n, stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu.h_max_disp_sq, gpu.d_max_disp_sq, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    full = !std::isfinite(*gpu.h_max_disp_sq) || *gpu.h_max_disp_sq > gpu.rebuild_threshold_sq;
  }
  bool reuse = !full && gpu.sph_cache_valid && skin > 0.0f &&
      gpu.sph_cached_skin == skin && gpu.sph_cached_cutoff == config.sph.kernel_cutoff_factor;
  if (reuse) {
    CUDA_CHECK(cudaMemsetAsync(gpu.d_verlet_disp_scaled, 0, sizeof(int32_t), stream));
    // Store/check h itself rather than k*h: distinct h values can round to the
    // same support extent but have different pair cutoffs. Part IDs are metadata.
    sph_cache_motion_kernel<<<(n+255)/256, 256, 0, stream>>>(
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.smoothing_length, state.particles.global_id, state.particles.clump_id,
        gpu.verlet_particles, gpu.d_verlet_disp_scaled, n, config.sph.kernel_cutoff_factor, skin);
    CUDA_CHECK(cudaMemcpyAsync(gpu.h_pinned_verlet_disp, gpu.d_verlet_disp_scaled, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double displacement_sq = *gpu.h_pinned_verlet_disp;
    reuse = 4.0 * displacement_sq < double(skin) * skin;
  }
  if (reuse) {
    ++gpu.stats.cache_hits;
    gpu.stats.num_contacts_found = gpu.sph_pairs.h_num_contacts;
    publish_sph_adjacency(gpu, state);
    gpu.stats.search_time_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now()-start).count();
    return;
  }
  if (gpu.sph_cache_valid) ++gpu.stats.cache_invalidations;
  gpu.sph_cache_valid = false;
  if (gpu.backend_timing_enabled) CUDA_CHECK(cudaEventRecord(gpu.timing_start, stream));
  if (full) {
    rebuild(state, config, stream);
    ++gpu.stats.lbvh_full_rebuilds;
    gpu.steps_since_rebuild = 0;
    if (frequency > 1) {
      lbvh_copy_snapshot_kernel<<<blocks, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          gpu.snapshot_pos_x, gpu.snapshot_pos_y, gpu.snapshot_pos_z, n);
      const float threshold = config.neighbor.lbvh_rebuild_threshold > 0.0f
          ? config.neighbor.lbvh_rebuild_threshold
          : compute_max_radius(state.particles, gpu, stream) * config.neighbor.lbvh_skin_factor * 0.5f;
      gpu.rebuild_threshold_sq = threshold * threshold;
    }
  } else {
    if (gpu.active_backend == NeighborBackend::RT) {
      gpu.rt_backend->update(rt_input(state, config), false, stream);
    } else {
      lbvh_refresh_leaf_aabbs_kernel<<<blocks, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          state.particles.support_extent, gpu.sorted_indices, gpu.aabbs, n, n-1, 0.0f);
      CUDA_CHECK(cudaMemsetAsync(gpu.atomic_flags, 0, size_t(n-1)*sizeof(int32_t), stream));
      lbvh_propagate_aabbs_kernel<<<blocks, threads, 0, stream>>>(
          gpu.nodes, gpu.aabbs, gpu.atomic_flags, n-1, n);
    }
    ++gpu.stats.lbvh_fast_path_steps;
  }
  if (gpu.backend_timing_enabled) {
    CUDA_CHECK(cudaEventRecord(gpu.timing_build_end, stream));
    CUDA_CHECK(cudaEventRecord(gpu.timing_query_start, stream));
  }
  if (gpu.active_backend == NeighborBackend::RT) {
    gpu.rt_backend->find_contacts(gpu.sph_pairs, stream);
  } else {
    unsigned long long count = 0;
    int overflow = 0;
    do {
      CUDA_CHECK(cudaMemsetAsync(gpu.sph_pair_count, 0, sizeof(count), stream));
      CUDA_CHECK(cudaMemsetAsync(gpu.sph_overflow, 0, sizeof(overflow), stream));
      sph_find_pairs_kernel<<<blocks, threads, 0, stream>>>(
          state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
          state.particles.smoothing_length, gpu.nodes, gpu.aabbs, gpu.sorted_indices,
          gpu.sph_pairs.pairs, gpu.sph_pair_count, gpu.sph_pairs.max_contacts,
          n, config.sph.kernel_cutoff_factor, skin, gpu.sph_overflow);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(&count, gpu.sph_pair_count, sizeof(count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(&overflow, gpu.sph_overflow, sizeof(overflow), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (overflow) throw std::runtime_error("SPH LBVH traversal stack overflow");
      if (count > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error("SPH neighbor pair count exceeds int32 capacity");
      if (count <= static_cast<unsigned long long>(gpu.sph_pairs.max_contacts)) break;
      const int64_t requested = std::max<int64_t>(count, 2LL*gpu.sph_pairs.max_contacts);
      ensure_contact_capacity(gpu.sph_pairs, static_cast<int32_t>(
          std::min<int64_t>(requested, std::numeric_limits<int32_t>::max())));
    } while (true);
    gpu.sph_pairs.h_num_contacts = static_cast<int32_t>(count);
  }
  const int32_t pairs = gpu.sph_pairs.h_num_contacts;
  const size_t edges = size_t(pairs) * 2;
  if (edges > gpu.sph_neighbor_capacity) {
    int32_t* next = nullptr;
    CUDA_CHECK(cudaMalloc(&next, edges*sizeof(int32_t)));
    CUDA_CHECK(cudaFree(gpu.sph_neighbors));
    gpu.sph_neighbors = next;
    gpu.sph_neighbor_capacity = edges;
  }
  CUDA_CHECK(cudaMemsetAsync(gpu.sph_cursors, 0, (size_t(n)+1)*sizeof(uint64_t), stream));
  const int pair_blocks = static_cast<int>((int64_t(pairs)+threads-1)/threads);
  if (pairs) sph_count_rows_kernel<<<pair_blocks, threads, 0, stream>>>(
      gpu.sph_pairs.pairs, pairs, gpu.sph_cursors);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(gpu.sph_scan_temp, gpu.sph_scan_bytes,
      gpu.sph_cursors, gpu.sph_offsets, n+1, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu.sph_cursors, gpu.sph_offsets,
      (size_t(n)+1)*sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
  if (pairs) sph_fill_rows_kernel<<<pair_blocks, threads, 0, stream>>>(
      gpu.sph_pairs.pairs, pairs, gpu.sph_cursors, gpu.sph_neighbors);
  if (skin > 0.0f) {
    lbvh_capture_cache_particles_kernel<<<blocks, threads, 0, stream>>>(
        state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
        state.particles.global_id, state.particles.clump_id, state.particles.smoothing_length,
        gpu.verlet_particles, n);
    gpu.sph_cache_valid = true;
  }
  CUDA_CHECK(cudaGetLastError());
  gpu.sph_ready = true;
  gpu.sph_cached_skin = skin;
  gpu.sph_cached_cutoff = config.sph.kernel_cutoff_factor;
  ++gpu.stats.cache_misses;
  gpu.stats.num_contacts_found = pairs;
  publish_sph_adjacency(gpu, state);
  if (gpu.backend_timing_enabled) {
    CUDA_CHECK(cudaEventRecord(gpu.timing_query_end, stream));
    CUDA_CHECK(cudaEventSynchronize(gpu.timing_query_end));
    float build_ms = 0, query_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&build_ms, gpu.timing_start, gpu.timing_build_end));
    CUDA_CHECK(cudaEventElapsedTime(&query_ms, gpu.timing_query_start, gpu.timing_query_end));
    if (full) gpu.stats.backend_build_time_ms += build_ms;
    else gpu.stats.backend_update_time_ms += build_ms;
    gpu.stats.backend_query_time_ms += query_ms;
  }
  gpu.stats.search_time_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now()-start).count();
}

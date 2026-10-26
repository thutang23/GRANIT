#pragma once

// A requested global cell is a disjoint cover of translated canonical cells.
// Fine requests have at most one component per Y image. Coarse requests use
// the canonical alignment level, where every Y translation is an integer key.
template<int D, bool Emit>
__global__ void fmm_image_parts_kernel(FMMSourcePublication source,
    const FMMCellKey* keys, int count, uint64_t* counts, const int* offsets,
    FMMSourcePart* parts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { if constexpr (!Emit) { if (i == count) counts[i] = 0; } return; }
    const auto* ctx = source.context;
    uint64_t found = 0;
    if (ctx) {
        const auto key = keys[i];
        const int k = source.image_alignment_level;
        if (key.level >= k) {
            uint32_t x=0, y=0, z=0; morton_decode<D>(key.morton, key.level, x, y, z);
            const int step = 1 << (key.level - k), side = 1 << key.level;
            for (int ny = -source.image_layers_y; ny <= source.image_layers_y; ++ny) {
                const int cy = int(y) - ny * step;
                if (cy < 0 || cy >= side) continue;
                const int node = persistent_morton_lookup<D>(ctx, key.level,
                    morton_encode<D>(x, uint32_t(cy), z));
                if (node < 0) continue;
                if constexpr (Emit) parts[offsets[i] + found] = {node, ny};
                ++found;
            }
        } else {
            const int side = 1 << k;
            for (int at = 0; at < ctx->dd.level_count[k]; ++at) {
                const int node = ctx->dd.level_list[ctx->dd.level_start[k] + at];
                uint32_t x=0, y=0, z=0;
                morton_decode<D>(ctx->dd.unified_nodes[node].compact_morton, k, x, y, z);
                for (int ny = -source.image_layers_y; ny <= source.image_layers_y; ++ny) {
                    const int gy = int(y) + ny;
                    if (gy < 0 || gy >= side) continue;
                    const auto global = morton_encode<D>(x, uint32_t(gy), z);
                    if ((global >> (D * (k - key.level))) != key.morton) continue;
                    if constexpr (Emit) parts[offsets[i] + found] = {node, ny};
                    ++found;
                }
            }
        }
    }
    if constexpr (!Emit) counts[i] = found;
}

template<int D>
__global__ void fmm_refresh_image_replies_kernel(FMMSourcePublication source,
    const FMMCellKey* keys, const int* offsets, const FMMSourcePart* parts,
    int count, FMMSourceReply* replies, bool skip_central = false)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (i >= count) return;
    const auto* ctx = source.context;
    if (!ctx) {
        replies[i].coeffs[lane] = 0;
        if (!lane) { replies[i].begin = i; replies[i].count = replies[i].global_count = 0; }
        return;
    }
    if (keys[i].level >= source.image_alignment_level) {
        far_real_t coefficient = 0;
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
            if (skip_central && parts[p].image_y == 0) continue;
            coefficient += ctx->dd.multipoles[parts[p].node].coeffs[lane];
        }
        replies[i].coeffs[lane] = coefficient;
    } else if (lane == 0) {
        uint32_t x=0, y=0, z=0; morton_decode<D>(keys[i].morton, keys[i].level, x, y, z);
        const double h = ldexp(ctx->metadata.root_cell_size, -keys[i].level);
        const double cx = ctx->metadata.domain_min_x + (x + 0.5) * h;
        const double cy = ctx->metadata.domain_min_y + (y + 0.5) * h;
        const double cz = D==2 ? ctx->dd.unified_nodes[0].center.z : ctx->metadata.domain_min_z + (z + 0.5) * h;
        double coefficients[fmm::FMM_MAX_COEFFS]{};
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
            const auto part = parts[p];
            if (skip_central && part.image_y == 0) continue;
            const auto node = ctx->dd.unified_nodes[part.node];
            const double* m = ctx->dd.multipoles_dbl + part.node * fmm::FMM_ACTIVE_COEFFS;
            coefficients[0] += m[0];
            translate_multipole_p4_unified(coefficients, m, node.center.x - cx,
                node.center.y + part.image_y * source.image_length_y - cy, node.center.z - cz);
        }
        for (int c = 0; c < fmm::FMM_MAX_COEFFS; ++c)
            replies[i].coeffs[c] = far_real_t(coefficients[c]);
    }
    if (lane == 0) {
        uint32_t population = 0, global_population = 0;
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
            const auto n = ctx->dd.unified_nodes[parts[p].node].count;
            global_population += n;
            if (!skip_central || parts[p].image_y != 0) population += n;
        }
        replies[i].global_count = global_population;
        // For virtual sources begin is a component-list handle, not a range.
        replies[i].begin = i; replies[i].count = population;
    }
}

__global__ void fmm_pack_image_particles_kernel(FMMSourcePublication source,
    const int* offsets, const FMMSourcePart* parts,
    const FMMParticleRequest* requests, FMMSourceParticle* particles, int count,
    bool skip_central = false, int* packed_clump_ids = nullptr)
{
    for (int i = blockIdx.x; i < count; i += gridDim.x) {
        const auto request = requests[i];
        if (!request.count) continue;
        uint32_t out = request.offset;
        for (int p = offsets[request.begin]; p < offsets[request.begin + 1]; ++p) {
            const auto part = parts[p];
            if (skip_central && part.image_y == 0) continue;
            const auto node = source.context->dd.unified_nodes[part.node];
            for (uint32_t j = threadIdx.x; j < node.count; j += blockDim.x) {
                const int original = source.sorted[node.begin + j];
                particles[out + j] = {source.x[original], source.y[original] + part.image_y * source.image_length_y,
                    source.z[original], source.mass[original], original};
                if (packed_clump_ids) packed_clump_ids[out + j] = source.clump_ids[original];
            }
            out += node.count;
        }
    }
}

__global__ void fmm_narrow_checked_offsets_kernel(const uint64_t* prefix,int* offsets,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<=count)offsets[i]=int(prefix[i]); // total was checked on the host
}

template<int D> inline void prepare_fmm_image_parts(FMMQueryPeer& peer, FMMSourcePublication source,
    int count, bool reuse)
{
    if (reuse) return;
    peer.part_counts.reserve(count + 1); peer.part_prefix.reserve(count + 1); peer.part_offsets.reserve(count + 1);
    fmm_image_parts_kernel<D,false><<<(count + 256) / 256, 256, 0, peer.stream>>>(
        source, peer.keys.data, count, peer.part_counts.data, nullptr, nullptr);
    fmm_cub(peer, [&](void* tmp, size_t& bytes) { return cub::DeviceScan::ExclusiveSum(
        tmp, bytes, peer.part_counts.data, peer.part_prefix.data, count + 1, peer.stream); });
    const uint64_t total = fmm_read_scalar(peer.part_prefix.data + count, peer.stream);
    if (total > INT_MAX) throw std::overflow_error("canonical source component count overflow");
    const int parts=int(total);
    fmm_narrow_checked_offsets_kernel<<<(count+256)/256,256,0,peer.stream>>>(
        peer.part_prefix.data,peer.part_offsets.data,count);
    peer.parts.reserve(parts);
    fmm_image_parts_kernel<D,true><<<(count + 255) / 256, 256, 0, peer.stream>>>(
        source, peer.keys.data, count, nullptr, peer.part_offsets.data, peer.parts.data);
}

__global__ void fmm_check_image_parts_kernel(const int* old_offsets,const int* fresh_offsets,
    const FMMSourcePart* old_parts,const FMMSourcePart* fresh_parts,int count,int* error) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>count)return;
    if(old_offsets[i]!=fresh_offsets[i]) { atomicOr(error,1);return; }
    if(i==count || old_offsets[i+1]!=fresh_offsets[i+1])return;
    for(int p=old_offsets[i];p<old_offsets[i+1];++p) {
        if(old_parts[p].node==fresh_parts[p].node && old_parts[p].image_y==fresh_parts[p].image_y)continue;
        // Level-list compaction may reorder nodes while their Morton/node
        // identities remain unchanged. Compare the disjoint covers as sets.
        bool old_found=false,fresh_found=false;
        for(int q=old_offsets[i];q<old_offsets[i+1];++q) {
            old_found|=old_parts[p].node==fresh_parts[q].node && old_parts[p].image_y==fresh_parts[q].image_y;
            fresh_found|=fresh_parts[p].node==old_parts[q].node && fresh_parts[p].image_y==old_parts[q].image_y;
        }
        atomicOr(error,(!old_found||!fresh_found)?1:2);
    }
}

template<int D> inline void validate_fmm_image_parts(FMMQueryPeer& peer,
    FMMSourcePublication source,int count) {
    const int parts=fmm_read_scalar(peer.part_offsets.data+count,peer.stream);
    peer.validation_part_offsets.reserve(count+1);peer.validation_parts.reserve(parts);
    peer.validation_error.reserve(1);
    CUDA_CHECK(cudaMemcpyAsync(peer.validation_part_offsets.data,peer.part_offsets.data,
        (count+1)*sizeof(int),cudaMemcpyDeviceToDevice,peer.stream));
    if(parts)CUDA_CHECK(cudaMemcpyAsync(peer.validation_parts.data,peer.parts.data,
        parts*sizeof(FMMSourcePart),cudaMemcpyDeviceToDevice,peer.stream));
    prepare_fmm_image_parts<D>(peer,source,count,false);
    CUDA_CHECK(cudaMemsetAsync(peer.validation_error.data,0,sizeof(int),peer.stream));
    fmm_check_image_parts_kernel<<<(count+256)/256,256,0,peer.stream>>>(
        peer.validation_part_offsets.data,peer.part_offsets.data,peer.validation_parts.data,
        peer.parts.data,count,peer.validation_error.data);
    const int verdict=fmm_read_scalar(peer.validation_error.data,peer.stream);
    if(verdict&1)
        throw std::runtime_error("canonical source schedule differs from fresh lookup");
    if(verdict&2)std::printf("[FMM image schedule] equivalent_cover_reordered=1 PASS\n");
}

#pragma once

// Included inside body_fitted_fmm after independent peer route construction.
template<int D>
__device__ FMMRemoteRef fmm_remote_far_ref(int target,int slot,
    const UnifiedNode* targets,const FMMRemoteView* peers,int ranks,int* error) {
    using T=GroupedStencilTraits<D>;
    FMMRemoteRef result;
    const int q=grouped_target_child<D>(targets[target].compact_morton);
    for(int peer=0;peer<ranks;++peer) {
        const auto view=peers[peer];if(!view.count)continue;
        const auto work=view.far_routes[target];int source=-1;
        if(slot<T::CROSS_COUNT) {
            const auto route=grouped_cross_route(work,slot);
            if(grouped_route_valid(route) && !(grouped_route_mask(route)&1u))source=grouped_route_node(route);
        } else {
            const int at=slot-T::CROSS_COUNT,b=at/T::CHILD_COUNT,c=at%T::CHILD_COUNT;
            const auto route=grouped_block_route(work,b);const uint8_t bit=uint8_t(1u<<c);
            if(grouped_route_valid(route) && (T::block(q,b).far_mask&bit) &&
                !(grouped_route_mask(route)&bit) && !(grouped_block_omit_mask(work,b)&bit))
                source=grouped_child_node<D>(view.nodes,view.children,grouped_route_node(route),c);
        }
        if(source>=0) {
            // Shared coefficients already have exactly one numerical owner.
            if(result.node>=0 && error)atomicExch(error,1);
            result={peer,source};
        }
    }
    return result;
}

template<int D,bool Emit>
__device__ uint64_t fmm_remote_direct_row(int leaf,const PersistentTreeContext* local,
    const FMMRemoteView* peers,int ranks,FMMRemoteRef* output) {
    using T=GroupedStencilTraits<D>;
    uint64_t count=0;
    for(int peer=0;peer<ranks;++peer) {
        const auto view=peers[peer];if(!view.count)continue;
        int target=leaf;
        for(int level=local->dd.unified_nodes[leaf].level;level>=2;--level) {
            if(target<0 || target>=local->metadata.num_nodes ||
                local->dd.unified_nodes[target].level!=level || !local->dd.unified_nodes[target].target_count) {
                printf("BF-FMM invalid remote direct target chain: leaf=%d target=%d level=%d\n",leaf,target,level);
                __trap();return 0;
            }
            const auto work=view.direct_routes[target];
            for(int c=0;c<T::CROSS_COUNT;++c) {
                const auto route=grouped_cross_route(work,c);
                if(grouped_route_valid(route) && (grouped_route_mask(route)&1u)) {
                    if constexpr(Emit)output[count]={peer,grouped_route_node(route)};
                    ++count;
                }
            }
            for(int b=0;b<T::BLOCK_COUNT;++b) {
                const auto route=grouped_block_route(work,b);if(!grouped_route_valid(route))continue;
                for(int c=0;c<T::CHILD_COUNT;++c)if(grouped_route_mask(route)&(1u<<c)) {
                    const int source=grouped_child_node<D>(view.nodes,view.children,grouped_route_node(route),c);
                    if(source>=0) {if constexpr(Emit)output[count]={peer,source};++count;}
                }
            }
            target=local->arena.d_targets[target].target_parent;
        }
    }
    return count;
}

template<int D> __global__ void fmm_remote_list_counts_kernel(
    const PersistentTreeContext* local,const FMMRemoteView* peers,int ranks,int nodes,
    int* indices,int* far_flags,int* near_flags,uint64_t* direct_counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>nodes)return;
    direct_counts[i]=0;if(i==nodes)return;
    indices[i]=i;far_flags[i]=near_flags[i]=0;
    const auto node=local->dd.unified_nodes[i];
    if(!node.target_count || node.level<2)return;
    using T=GroupedStencilTraits<D>;
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot)
        if(fmm_remote_far_ref<D>(i,slot,local->dd.unified_nodes,peers,ranks,nullptr).node>=0) {
            far_flags[i]=1;break;
        }
    if(node.flags&UNIFIED_NODE_FLAG_IS_LEAF) {
        direct_counts[i]=fmm_remote_direct_row<D,false>(i,local,peers,ranks,nullptr);
        near_flags[i]=direct_counts[i]!=0;
    }
}
template<int D> __global__ void fmm_remote_far_keys_kernel(
    const int* ids,int count,const UnifiedNode* nodes,uint64_t* keys) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int target=ids[i];
    keys[i]=(uint64_t(grouped_target_child<D>(nodes[target].compact_morton))<<32)|uint32_t(target);
}
__global__ void fmm_remote_far_targets_kernel(const uint64_t* keys,int count,int* targets,int* classes) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    targets[i]=int(uint32_t(keys[i]));atomicAdd(classes+(keys[i]>>32),1);
}
template<int D> __global__ void fmm_remote_far_sources_kernel(
    const int* targets,int count,const UnifiedNode* nodes,const FMMRemoteView* peers,int ranks,
    FMMRemoteRef* sources,int* error) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
    using T=GroupedStencilTraits<D>;
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot)
        sources[size_t(slot)*count+row]=fmm_remote_far_ref<D>(targets[row],slot,nodes,peers,ranks,error);
}
template<int D> __global__ void fmm_remote_direct_sources_kernel(
    const int* leaves,int count,const PersistentTreeContext* local,const FMMRemoteView* peers,int ranks,
    const uint64_t* offsets,FMMRemoteRef* sources) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
    const int leaf=leaves[row];
    fmm_remote_direct_row<D,true>(leaf,local,peers,ranks,sources+offsets[leaf]);
}

template<int D> inline void fmm_prepare_remote_lists(int gpu,DistributedFMMState& s,
    PersistentTreeRuntime* runtime,cudaStream_t stream) {
    auto& plan=s.remote_lists;const int nodes=s.gpu_plan.owned_nodes;
    if(!nodes || !s.received_cells) {
        plan.far_count=plan.near_count=0;plan.valid=false;return;
    }
    plan.host_views.assign(s.ranks,{});
    std::vector<uint64_t> generations(s.ranks,~uint64_t{0});
    for(int peer=0;peer<s.ranks;++peer) {
        if((peer==gpu&&!s.routing.enabled)||(s.local_only&&peer!=gpu))continue;
        const auto& q=*s.peers[peer];generations[peer]=q.source_generation;
        if(!q.request_count)continue;
        plan.host_views[peer]={q.halo_nodes.data,q.halo_children.data,q.halo_multipoles.data,
            q.moment_routes.data,q.halo_routes.data,q.halo_x.data,q.halo_y.data,q.halo_z.data,
            q.halo_mass.data,q.halo_order.data,q.request_count};
    }
    plan.views.reserve(s.ranks);
    CUDA_CHECK(cudaMemcpyAsync(plan.views.data,plan.host_views.data(),s.ranks*sizeof(FMMRemoteView),cudaMemcpyHostToDevice,stream));
    const bool rebuild=!plan.valid || plan.local_stamp!=s.publication_generation ||
        plan.topology_stamp!=s.gpu_plan.topology_rebuilds || plan.generations!=generations;
    if(!rebuild)return;
    plan.indices.reserve(nodes);plan.far_flags.reserve(nodes);plan.near_flags.reserve(nodes);
    plan.direct_counts.reserve(size_t(nodes)+1);plan.direct_offsets.reserve(size_t(nodes)+1);
    plan.far_targets.reserve(nodes);plan.near_leaves.reserve(nodes);plan.scalar.reserve(1);
    plan.class_counts.reserve(8);plan.error.reserve(1);
    fmm_remote_list_counts_kernel<D><<<(nodes+256)/256,256,0,stream>>>(runtime->device_context,
        plan.views.data,s.ranks,nodes,plan.indices.data,plan.far_flags.data,plan.near_flags.data,plan.direct_counts.data);
    fmm_cub(plan,[&](void* tmp,size_t& bytes) {return cub::DeviceSelect::Flagged(tmp,bytes,
        plan.indices.data,plan.far_flags.data,plan.far_targets.data,plan.scalar.data,nodes,stream);});
    plan.far_count=fmm_read_scalar(plan.scalar.data,stream);
    fmm_cub(plan,[&](void* tmp,size_t& bytes) {return cub::DeviceSelect::Flagged(tmp,bytes,
        plan.indices.data,plan.near_flags.data,plan.near_leaves.data,plan.scalar.data,nodes,stream);});
    plan.near_count=fmm_read_scalar(plan.scalar.data,stream);
    fmm_cub(plan,[&](void* tmp,size_t& bytes) {return cub::DeviceScan::ExclusiveSum(tmp,bytes,
        plan.direct_counts.data,plan.direct_offsets.data,nodes+1,stream);});
    const uint64_t direct_count=fmm_read_scalar(plan.direct_offsets.data+nodes,stream);
    plan.direct_sources.reserve(direct_count);
    if(plan.near_count)fmm_remote_direct_sources_kernel<D><<<(plan.near_count+255)/256,256,0,stream>>>(
        plan.near_leaves.data,plan.near_count,runtime->device_context,plan.views.data,s.ranks,
        plan.direct_offsets.data,plan.direct_sources.data);
    std::fill(plan.class_offsets,plan.class_offsets+9,0);
    if(plan.far_count) {
        using T=GroupedStencilTraits<D>;
        constexpr int slots=T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;
        const size_t entries=size_t(slots)*plan.far_count;
        if(entries>INT_MAX)throw std::overflow_error("FMM remote M2L reference count exceeds signed indexing");
        plan.far_sources.reserve(entries);plan.far_keys.reserve(plan.far_count);plan.sorted_far_keys.reserve(plan.far_count);
        fmm_remote_far_keys_kernel<D><<<(plan.far_count+255)/256,256,0,stream>>>(plan.far_targets.data,
            plan.far_count,runtime->host_context.dd.unified_nodes,plan.far_keys.data);
        fmm_cub(plan,[&](void* tmp,size_t& bytes) {return cub::DeviceRadixSort::SortKeys(tmp,bytes,
            plan.far_keys.data,plan.sorted_far_keys.data,plan.far_count,32,35,stream);});
        CUDA_CHECK(cudaMemsetAsync(plan.class_counts.data,0,8*sizeof(int),stream));
        CUDA_CHECK(cudaMemsetAsync(plan.error.data,0,sizeof(int),stream));
        fmm_remote_far_targets_kernel<<<(plan.far_count+255)/256,256,0,stream>>>(
            plan.sorted_far_keys.data,plan.far_count,plan.far_targets.data,plan.class_counts.data);
        fmm_remote_far_sources_kernel<D><<<(plan.far_count+255)/256,256,0,stream>>>(plan.far_targets.data,
            plan.far_count,runtime->host_context.dd.unified_nodes,plan.views.data,s.ranks,plan.far_sources.data,plan.error.data);
        int counts[8];CUDA_CHECK(cudaMemcpyAsync(counts,plan.class_counts.data,sizeof(counts),cudaMemcpyDeviceToHost,stream));
        if(fmm_read_scalar(plan.error.data,stream))throw std::logic_error("duplicate remote M2L source ownership");
        for(int q=0;q<8;++q)plan.class_offsets[q+1]=plan.class_offsets[q]+counts[q];
    }
    plan.local_stamp=s.publication_generation;plan.topology_stamp=s.gpu_plan.topology_rebuilds;
    plan.generations=std::move(generations);plan.valid=true;
    if(std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS"))
        std::printf("[FMM fused remote] rank=%d far_targets=%d direct_leaves=%d direct_refs=%llu\n",
            gpu,plan.far_count,plan.near_count,(unsigned long long)direct_count);
}

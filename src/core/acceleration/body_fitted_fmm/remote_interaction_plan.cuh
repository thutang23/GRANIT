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
        const int row=fmm_remote_route_row(view,target);if(row<0)continue;
        const auto work=view.far_routes[row];int source=-1;
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
            const int row=fmm_remote_route_row(view,target);
            if(row<0) {target=local->arena.d_targets[target].target_parent;continue;}
            const auto work=view.direct_routes[row];
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

// Only the existence of a remote far source is needed to select a target.
// Keep fmm_remote_far_ref() for the ordered source list and its duplicate-owner
// check; changing the traversal order here cannot change numerical ownership.
template<int D>
__device__ bool fmm_remote_has_far(int target,const UnifiedNode& node,
    const FMMRemoteView* peers,int ranks) {
    using T=GroupedStencilTraits<D>;
    const int q=grouped_target_child<D>(node.compact_morton);
    for(int peer=0;peer<ranks;++peer) {
        const FMMRemoteView view=peers[peer];
        if(!view.count)continue;
        const int row=fmm_remote_route_row(view,target);if(row<0)continue;
        const auto& work=view.far_routes[row];
        for(int cross=0;cross<T::CROSS_COUNT;++cross) {
            const auto route=grouped_cross_route(work,cross);
            if(grouped_route_valid(route) && !(grouped_route_mask(route)&1u))return true;
        }
        for(int block=0;block<T::BLOCK_COUNT;++block) {
            const auto route=grouped_block_route(work,block);
            if(!grouped_route_valid(route))continue;
            uint8_t candidates=static_cast<uint8_t>(T::block(q,block).far_mask &
                ~grouped_route_mask(route) & ~grouped_block_omit_mask(work,block));
            while(candidates) {
                const int child=__ffs(static_cast<int>(candidates))-1;
                candidates=static_cast<uint8_t>(candidates & (candidates-1));
                if(grouped_child_node<D>(view.nodes,view.children,
                    grouped_route_node(route),child)>=0)return true;
            }
        }
    }
    return false;
}

#ifdef GRANIT_FMM_TEST_HOOKS
template<int D> __global__ void fmm_remote_legacy_far_flags_kernel(
    const UnifiedNode* nodes,const FMMRemoteView* peers,int ranks,int count,int* flags) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto node=nodes[i];
    flags[i]=0;
    if(!node.target_count || node.level<2)return;
    using T=GroupedStencilTraits<D>;
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot)
        if(fmm_remote_far_ref<D>(i,slot,nodes,peers,ranks,nullptr).node>=0) {
            flags[i]=1;break;
        }
}

__global__ void fmm_remote_compare_far_flags_kernel(const int* candidate,
    const int* reference,int count,int* mismatches,int* first_mismatch) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && candidate[i]!=reference[i]) {
        atomicAdd(mismatches,1);
        atomicMin(first_mismatch,i);
    }
}
#endif

template<int D> __global__ void fmm_remote_list_counts_kernel(
    const PersistentTreeContext* local,const FMMRemoteView* peers,int ranks,int nodes,
    int* indices,int* far_flags,int* near_flags,uint64_t* direct_counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>nodes)return;
    direct_counts[i]=0;if(i==nodes)return;
    indices[i]=i;far_flags[i]=near_flags[i]=0;
    const auto node=local->dd.unified_nodes[i];
    if(!node.target_count || node.level<2)return;
    far_flags[i]=fmm_remote_has_far<D>(i,node,peers,ranks)?1:0;
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
// Lookups use exact cell codes, not probabilistic fingerprints. The sorted
// key index decouples cache rows from both native IDs and child-class ordering.
__device__ inline int fmm_far_cache_find_row(uint64_t key,const uint64_t* keys,
    const int* rows,int count) {
    int lo=0,hi=count;
    while(lo<hi) {const int mid=lo+(hi-lo)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
    return lo<count && keys[lo]==key?rows[lo]:-1;
}
__device__ inline bool fmm_far_cache_same_source(uint32_t before,uint32_t now,
    const FMMFarCachePeerView& cache) {
    if(before==GROUPED_ROUTE_INVALID_NODE || now==GROUPED_ROUTE_INVALID_NODE)return before==now;
    return before<uint32_t(cache.count) && cache.source_remap[before]==int(now);
}
template<int D> __device__ bool fmm_far_cache_target_matches(int target,int old_row,
    const FMMRemoteView* current,const FMMFarCachePeerView* previous,
    const FMMFarCacheTargetPeer* snapshots,int old_count,int ranks) {
    using T=GroupedStencilTraits<D>;
    for(int peer=0;peer<ranks;++peer) {
        const auto view=current[peer];const auto cache=previous[peer];
        const int row=view.count?fmm_remote_route_row(view,target):-1;
        const auto& old=snapshots[size_t(peer)*old_count+old_row];
        if((row>=0)!=old.present)return false;
        if(row<0)continue;
        const auto& a=old.routes;const auto& b=view.far_routes[row];
        if(a.cross_direct_bits!=b.cross_direct_bits)return false;
        for(int cross=0;cross<T::CROSS_COUNT;++cross)
            if(!fmm_far_cache_same_source(a.cross_node[cross],b.cross_node[cross],cache))return false;
        for(int block=0;block<T::BLOCK_COUNT;++block) {
            const uint32_t before=a.block_parent[block],now=b.block_parent[block];
            if(a.block_direct_mask[block]!=b.block_direct_mask[block] ||
                a.block_omit_mask[block]!=b.block_omit_mask[block] ||
                !fmm_far_cache_same_source(before,now,cache))return false;
            // A parent identity alone is insufficient: an occupied child may
            // appear/disappear while the parent persists. Equal child masks
            // prove equal canonical child keys after source-index rebinding.
            if(before!=GROUPED_ROUTE_INVALID_NODE && cache.child_masks[before]!=view.nodes[now].childMask)return false;
        }
    }
    return true;
}
template<int D> __global__ void fmm_far_cache_source_remap_kernel(
    const FMMCellKey* old_keys,int old_count,const FMMCellKey* keys,int count,int* remap) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=old_count)return;
    remap[i]=fmm_halo_lookup<D>(keys,count,old_keys[i]);
}
template<int D> __global__ void fmm_remote_far_cached_sources_kernel(
    const int* targets,int count,const UnifiedNode* nodes,const FMMRemoteView* peers,int ranks,
    FMMRemoteRef* sources,int* error,const uint64_t* old_keys,const int* old_rows,int old_count,
    const FMMFarCachePeerView* previous,const FMMFarCacheTargetPeer* snapshots,
    const FMMRemoteRef* old_sources,bool verify,int* statistics) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
    const int target=targets[row];const auto node=nodes[target];
    const int old_row=fmm_far_cache_find_row(gpu_cell_code<D>({node.compact_morton,node.level}),old_keys,old_rows,old_count);
    const bool reuse=old_row>=0 && fmm_far_cache_target_matches<D>(target,old_row,peers,previous,snapshots,old_count,ranks);
    if(statistics)atomicAdd(statistics+(reuse?0:1),1);
    using T=GroupedStencilTraits<D>;
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
        FMMRemoteRef result;
        if(reuse) {
            result=old_sources[size_t(slot)*old_count+old_row];
            if(result.node>=0)result.node=previous[result.peer].source_remap[result.node];
        } else result=fmm_remote_far_ref<D>(target,slot,nodes,peers,ranks,error);
        sources[size_t(slot)*count+row]=result;
        if(verify) {
            const auto expected=fmm_remote_far_ref<D>(target,slot,nodes,peers,ranks,error);
            if(result.peer!=expected.peer || result.node!=expected.node) {
                if(atomicCAS(error,0,2)==0)
                    printf("FMM far cache mismatch target=%d slot=%d reused=%d actual=(%d,%d) expected=(%d,%d)\n",
                        target,slot,int(reuse),result.peer,result.node,expected.peer,expected.node);
            }
        }
    }
}
template<int D> __global__ void fmm_far_cache_target_keys_kernel(
    const int* targets,int count,const UnifiedNode* nodes,uint64_t* keys,int* rows) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
    const auto node=nodes[targets[row]];
    keys[row]=gpu_cell_code<D>({node.compact_morton,node.level});rows[row]=row;
}
__global__ void fmm_far_cache_snapshot_targets_kernel(const int* targets,int count,
    const FMMRemoteView* peers,int ranks,FMMFarCacheTargetPeer* snapshots) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(count)*ranks)return;
    const int peer=int(i/count),row=int(i%count);const auto view=peers[peer];
    const int route=view.count?fmm_remote_route_row(view,targets[row]):-1;
    snapshots[i].present=route>=0;
    if(route>=0)snapshots[i].routes=view.far_routes[route];
}
__global__ void fmm_far_cache_snapshot_children_kernel(const UnifiedNode* nodes,int count,uint8_t* masks) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)masks[i]=nodes[i].childMask;
}

template<int D> inline bool fmm_far_cache_geometry_matches(const FMMRemoteLists& plan,
    const DistributedFMMState& s,int gpu) {
    return plan.cache_valid && plan.cache_dimension==D && plan.cache_peers.size()==size_t(s.ranks) &&
        plan.cache_cutoff==s.gpu_plan.cutoff && plan.cache_layers_x==s.routing.layers_x &&
        plan.cache_layers_y==s.routing.layers_y && plan.cache_length_x==s.routing.length_x &&
        plan.cache_length_y==s.routing.length_y && plan.cache_geometry.matches(s.grid,s.ownership_grid,
            s.cuts,s.routing,s.local_only,gpu,s.gpu_plan.depth);
}
template<int D> inline void fmm_far_cache_prepare_views(DistributedFMMState& s,bool eligible,cudaStream_t stream) {
    auto& plan=s.remote_lists;plan.cache_host_views.assign(s.ranks,{});
    if(eligible)for(int peer=0;peer<s.ranks;++peer) {
        auto& previous=*plan.cache_peers[peer];const auto current=plan.host_views[peer];
        if(!previous.count)continue;
        previous.source_remap.reserve(previous.count);
        const auto& q=*s.peers[peer];
        fmm_far_cache_source_remap_kernel<D><<<(previous.count+255)/256,256,0,stream>>>(
            previous.keys.data,previous.count,q.remote_keys.data,current.count,previous.source_remap.data);
        plan.cache_host_views[peer]={previous.source_remap.data,previous.child_masks.data,previous.count};
    }
    plan.cache_views.reserve(s.ranks);
    CUDA_CHECK(cudaMemcpyAsync(plan.cache_views.data,plan.cache_host_views.data(),
        s.ranks*sizeof(FMMFarCachePeerView),cudaMemcpyHostToDevice,stream));
}
template<int D> inline void fmm_far_cache_snapshot(int gpu,DistributedFMMState& s,
    const UnifiedNode* nodes,cudaStream_t stream) {
    auto& plan=s.remote_lists;const int count=plan.far_count;
    if(!count) {plan.cache_valid=false;plan.cache_count=0;return;}
    plan.cache_unsorted_keys.reserve(count);plan.cache_unsorted_rows.reserve(count);
    plan.cache_keys.reserve(count);plan.cache_rows.reserve(count);
    fmm_far_cache_target_keys_kernel<D><<<(count+255)/256,256,0,stream>>>(plan.far_targets.data,
        count,nodes,plan.cache_unsorted_keys.data,plan.cache_unsorted_rows.data);
    fmm_cub(plan,[&](void* tmp,size_t& bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,
        plan.cache_unsorted_keys.data,plan.cache_keys.data,plan.cache_unsorted_rows.data,
        plan.cache_rows.data,count,0,64,stream);});
    plan.cache_targets.reserve(size_t(count)*s.ranks);
    fmm_far_cache_snapshot_targets_kernel<<<(size_t(count)*s.ranks+255)/256,256,0,stream>>>(
        plan.far_targets.data,count,plan.views.data,s.ranks,plan.cache_targets.data);
    if(plan.cache_peers.size()!=size_t(s.ranks)) {
        plan.cache_peers.clear();plan.cache_peers.resize(s.ranks);
        for(auto& peer:plan.cache_peers)peer=std::make_unique<FMMFarCachePeerStorage>();
    }
    for(int peer=0;peer<s.ranks;++peer) {
        auto& previous=*plan.cache_peers[peer];const auto view=plan.host_views[peer];
        previous.count=view.count;if(!view.count)continue;
        const auto& q=*s.peers[peer];previous.keys.reserve(view.count);previous.child_masks.reserve(view.count);
        CUDA_CHECK(cudaMemcpyAsync(previous.keys.data,q.remote_keys.data,view.count*sizeof(FMMCellKey),cudaMemcpyDeviceToDevice,stream));
        fmm_far_cache_snapshot_children_kernel<<<(view.count+255)/256,256,0,stream>>>(view.nodes,view.count,previous.child_masks.data);
    }
    plan.cache_geometry.capture(s.grid,s.ownership_grid,s.cuts,s.routing,s.local_only,gpu,s.gpu_plan.depth);
    plan.cache_dimension=D;plan.cache_cutoff=s.gpu_plan.cutoff;
    plan.cache_layers_x=s.routing.layers_x;plan.cache_layers_y=s.routing.layers_y;
    plan.cache_length_x=s.routing.length_x;plan.cache_length_y=s.routing.length_y;
    plan.cache_count=count;plan.cache_valid=true;
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
        plan.far_count=plan.near_count=0;plan.valid=false;plan.cache_valid=false;return;
    }
    plan.host_views.assign(s.ranks,{});
    std::vector<uint64_t> generations(s.ranks,~uint64_t{0});
    for(int peer=0;peer<s.ranks;++peer) {
        if((peer==gpu&&!s.routing.enabled)||(s.local_only&&peer!=gpu))continue;
        const auto& q=*s.peers[peer];generations[peer]=q.source_generation;
        if(!q.request_count)continue;
        plan.host_views[peer]={q.halo_nodes.data,q.halo_children.data,q.halo_multipoles.data,
            q.moment_routes.data,q.halo_routes.data,q.halo_x.data,q.halo_y.data,q.halo_z.data,
            q.halo_mass.data,q.halo_order.data,q.request_count,q.route_index.data};
    }
    plan.views.reserve(s.ranks);
    CUDA_CHECK(cudaMemcpyAsync(plan.views.data,plan.host_views.data(),s.ranks*sizeof(FMMRemoteView),cudaMemcpyHostToDevice,stream));
    const bool cache_eligible=fmm_far_cache_geometry_matches<D>(plan,s,gpu);
    const bool rebuild=!plan.valid || plan.local_stamp!=s.publication_generation ||
        plan.topology_stamp!=s.gpu_plan.topology_rebuilds || plan.generations!=generations ||
        (plan.cache_valid && !cache_eligible);
    if(!rebuild)return;
    // Swap storage, not contents. The old epoch remains immutable until the
    // cached-source kernel finishes; snapshots are replaced later on-stream.
    std::swap(plan.far_sources.data,plan.cached_far_sources.data);
    std::swap(plan.far_sources.capacity,plan.cached_far_sources.capacity);
    std::swap(plan.far_sources.device,plan.cached_far_sources.device);
    plan.indices.reserve(nodes);plan.far_flags.reserve(nodes);plan.near_flags.reserve(nodes);
    plan.direct_counts.reserve(size_t(nodes)+1);plan.direct_offsets.reserve(size_t(nodes)+1);
    plan.far_targets.reserve(nodes);plan.near_leaves.reserve(nodes);plan.scalar.reserve(1);
    plan.class_counts.reserve(8);plan.error.reserve(1);
    fmm_remote_list_counts_kernel<D><<<(nodes+256)/256,256,0,stream>>>(runtime->device_context,
        plan.views.data,s.ranks,nodes,plan.indices.data,plan.far_flags.data,plan.near_flags.data,plan.direct_counts.data);
#ifdef GRANIT_FMM_TEST_HOOKS
    if(std::getenv("GRANIT_FMM_VERIFY_REMOTE_FAR")) {
        FMMDeviceBuffer<int> reference_flags;
        reference_flags.reserve(nodes);
        CUDA_CHECK(cudaMemsetAsync(plan.error.data,0,sizeof(int),stream));
        CUDA_CHECK(cudaMemcpyAsync(plan.scalar.data,&nodes,sizeof(int),cudaMemcpyHostToDevice,stream));
        fmm_remote_legacy_far_flags_kernel<D><<<(nodes+255)/256,256,0,stream>>>(
            runtime->host_context.dd.unified_nodes,plan.views.data,s.ranks,nodes,reference_flags.data);
        fmm_remote_compare_far_flags_kernel<<<(nodes+255)/256,256,0,stream>>>(
            plan.far_flags.data,reference_flags.data,nodes,plan.error.data,plan.scalar.data);
        int mismatches=0,first=nodes;
        CUDA_CHECK(cudaMemcpyAsync(&mismatches,plan.error.data,sizeof(int),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaMemcpyAsync(&first,plan.scalar.data,sizeof(int),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("[FMM remote-list verify] rank=%d epoch=%llu nodes=%d mismatches=%d first=%d\n",
            gpu,(unsigned long long)s.epoch,nodes,mismatches,first);
        if(mismatches) {
            std::fprintf(stderr,"FMM remote far flags differ: rank=%d epoch=%llu first=%d\n",
                gpu,(unsigned long long)s.epoch,first);
            std::fflush(stderr);
            std::abort(); // End every rank immediately; a local exception can strand collective peers.
        }
    }
#endif
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
        const bool verify_cache=std::getenv("GRANIT_FMM_VERIFY_FAR_CACHE")!=nullptr;
        const bool cache_diagnostics=verify_cache || std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS");
        fmm_far_cache_prepare_views<D>(s,cache_eligible,stream);
        int* statistics=nullptr;
        if(cache_diagnostics) {
            plan.cache_statistics.reserve(2);statistics=plan.cache_statistics.data;
            CUDA_CHECK(cudaMemsetAsync(statistics,0,2*sizeof(int),stream));
        }
        fmm_remote_far_cached_sources_kernel<D><<<(plan.far_count+255)/256,256,0,stream>>>(plan.far_targets.data,
            plan.far_count,runtime->host_context.dd.unified_nodes,plan.views.data,s.ranks,plan.far_sources.data,plan.error.data,
            plan.cache_keys.data,plan.cache_rows.data,cache_eligible?plan.cache_count:0,plan.cache_views.data,
            plan.cache_targets.data,plan.cached_far_sources.data,verify_cache,statistics);
        int cache_counts[2]{};
        if(cache_diagnostics)CUDA_CHECK(cudaMemcpyAsync(cache_counts,statistics,sizeof(cache_counts),cudaMemcpyDeviceToHost,stream));
        int counts[8];CUDA_CHECK(cudaMemcpyAsync(counts,plan.class_counts.data,sizeof(counts),cudaMemcpyDeviceToHost,stream));
        const int error=fmm_read_scalar(plan.error.data,stream);
        if(error==2) {
            std::fprintf(stderr,"FMM far cache oracle failed: rank=%d epoch=%llu\n",gpu,(unsigned long long)s.epoch);
            std::fflush(stderr);
            // All GPU ranks share this process. A failed verification must
            // terminate it rather than strand peers in a later collective.
            std::abort();
        }
        if(error)throw std::logic_error("duplicate remote M2L source ownership");
        if(cache_diagnostics)std::printf("[FMM far cache] rank=%d epoch=%llu eligible=%d reused=%d changed=%d verify=%d%s\n",
            gpu,(unsigned long long)s.epoch,int(cache_eligible),cache_counts[0],cache_counts[1],int(verify_cache),verify_cache?" PASS":"");
        for(int q=0;q<8;++q)plan.class_offsets[q+1]=plan.class_offsets[q]+counts[q];
    }
    // All old-cache reads precede these writes on the same consumer stream.
    fmm_far_cache_snapshot<D>(gpu,s,runtime->host_context.dd.unified_nodes,stream);
    plan.local_stamp=s.publication_generation;plan.topology_stamp=s.gpu_plan.topology_rebuilds;
    plan.generations=std::move(generations);plan.valid=true;
    if(std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS"))
        std::printf("[FMM fused remote] rank=%d far_targets=%d direct_leaves=%d direct_refs=%llu\n",
            gpu,plan.far_count,plan.near_count,(unsigned long long)direct_count);
}

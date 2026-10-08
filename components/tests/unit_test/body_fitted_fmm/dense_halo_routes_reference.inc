// Frozen dense reference, commit 51af3606fac9bc54bb87a2345642a14a615b9e16.
// Kept independent of subsequent compact-route edits. CUDA qualifiers are host-defined by harness.

// From distributed_fmm_gpu.cuh
__global__ void dense_fmm_halo_initialize_routes_kernel(const PersistentTreeContext* target,int n,
    GroupedTargetInteractions* routes,int* flags,int sources) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<sources)flags[i]=0;
    if(i>=n)return;
    GroupedTargetInteractions out{};
    for(int j=0;j<GROUPED_MAX_CROSS_COUNT;++j)out.cross_node[j]=GROUPED_ROUTE_INVALID_NODE;
    for(int j=0;j<GROUPED_MAX_BLOCK_COUNT;++j)out.block_parent[j]=GROUPED_ROUTE_INVALID_NODE;
    const auto node=target->dd.unified_nodes[i];
    out.target_parent=node.target_count && node.level>=2?target->arena.d_targets[i].target_parent:-1;
    routes[i]=out;
}

// From distributed_fmm_gpu.cuh
template<int D> __global__ void dense_fmm_halo_count_routes_kernel(const PersistentTreeContext* local,
    const UnifiedNode* sources,const int* children,const GroupedTargetInteractions* routes,
    int targets,int* counts,bool local_sources) {
    const int target=blockIdx.x*blockDim.x+threadIdx.x;if(target>=targets)return;
    const auto node=local->dd.unified_nodes[target];if(!node.target_count || node.level<2)return;
    using T=GroupedStencilTraits<D>;const auto work=routes[target];
    int result[4]{};const int q=grouped_target_child<D>(node.compact_morton);
    for(int b=0;b<T::CROSS_COUNT;++b) {const auto r=grouped_cross_route(work,b);
        if(grouped_route_valid(r))++result[(grouped_route_mask(r)&1u)?2:1];}
    for(int b=0;b<T::BLOCK_COUNT;++b) {
        const auto r=grouped_block_route(work,b);if(!grouped_route_valid(r))continue;
        const auto e=T::block(q,b);
        for(int c=0;c<T::CHILD_COUNT;++c) {
            const int source=grouped_child_node<D>(sources,children,grouped_route_node(r),c);if(source<0)continue;
            const uint8_t bit=uint8_t(1u<<c);if(grouped_block_omit_mask(work,b)&bit)continue;
            if(e.far_mask&bit)++result[(grouped_route_mask(r)&bit)?2:0];
            else if((node.flags&UNIFIED_NODE_FLAG_IS_LEAF) && (grouped_route_mask(r)&bit))
                ++result[local_sources && source==target?3:2];
        }
    }
    for(int c=0;c<4;++c)if(result[c])atomicAdd(counts+c,result[c]);
}

// From distributed_fmm_gpu.cuh
template<int D> __global__ void dense_fmm_halo_inverse_routes_kernel(
    const PersistentTreeContext* local,const UnifiedNode* sources,const int* children,
    const FMMCellKey* keys,int source_count,int depth,bool disable_fallback,GroupedTargetInteractions* routes,int* needed) {
    using T=GroupedStencilTraits<D>;
    const int task=blockIdx.x*blockDim.x+threadIdx.x;
    const int source=task/T::CHILD_COUNT,q=task%T::CHILD_COUNT;
    if(source>=source_count)return;
    const auto parent=sources[source];const int level=parent.level+1;if(level>depth)return;
    uint32_t x=0,y=0,z=0;morton_decode<D>(parent.compact_morton,parent.level,x,y,z);
    const int source_xyz[3]={int(x),int(y),int(z)};
    const int slot=grouped_morton_slot<D>(q);
    for(int b=0;b<T::CROSS_COUNT+T::BLOCK_COUNT;++b) {
        const bool cross=b<T::CROSS_COUNT;
        int pos[3]{};
        if(cross) {const auto e=T::cross(q,b);for(int a=0;a<D;++a)pos[a]=2*(source_xyz[a]-e.parent[a])+((slot>>a)&1);}
        else {const auto e=T::block(q,b-T::CROSS_COUNT);for(int a=0;a<D;++a)pos[a]=2*(source_xyz[a]-e.parent[a])+((slot>>a)&1);}
        const int target=persistent_coordinate_lookup<D>(local,level,pos[0],pos[1],pos[2]);
        if(target<0)continue;
        const auto t=local->dd.unified_nodes[target];if(!t.target_count)continue;
        uint32_t target_population=t.count;
        if(!disable_fallback) {const int at=fmm_halo_lookup<D>(keys,source_count,{t.compact_morton,t.level});
            if(at>=0)target_population=sources[at].global_count;}
        auto& out=routes[target];const bool leaf=t.flags&UNIFIED_NODE_FLAG_IS_LEAF;
        if(cross) {
            const auto relation=persistent_cutoff_relation(t,parent,local->command.cutoff_radius,local->metadata.root_cell_size);
            if(relation==CutoffRelation::Outside)continue;
            const bool direct=relation==CutoffRelation::Straddles || (!disable_fallback &&
                uint64_t(parent.global_count)*target_population<=grouped_direct_threshold(level,leaf));
            out.cross_node[b]=source;
            if(direct) {atomicOr(&out.cross_direct_bits,1u<<b);atomicExch(needed+source,1);}
        } else {
            const int block=b-T::CROSS_COUNT;const auto e=T::block(q,block);
            uint8_t direct=0,omit=0;
            for(int c=0;c<T::CHILD_COUNT;++c) {
                const int child=grouped_child_node<D>(sources,children,source,c);if(child<0)continue;
                const auto s=sources[child];const uint8_t bit=uint8_t(1u<<c);
                const auto relation=persistent_cutoff_relation(t,s,local->command.cutoff_radius,local->metadata.root_cell_size);
                if(relation==CutoffRelation::Outside) {if(e.far_mask&bit)omit|=bit;continue;}
                if((e.far_mask&bit) && (relation==CutoffRelation::Straddles || (!disable_fallback &&
                    uint64_t(s.global_count)*target_population<=grouped_direct_threshold(level,leaf))))direct|=bit;
                if(leaf && (e.near_mask&bit))direct|=bit;
                if(direct&bit)atomicExch(needed+child,1);
            }
            out.block_parent[block]=source;out.block_direct_mask[block]=direct;out.block_omit_mask[block]=omit;
        }
    }
}

// From distributed_fmm_gpu.cuh
template<int D> __global__ void dense_fmm_halo_moment_routes_kernel(
    const PersistentTreeContext* local,const UnifiedNode* sources,const int* children,
    const GroupedTargetInteractions* routes,GroupedTargetInteractions* output,
    int targets,int peer,int peers,const int* slots,const int* owners) {
    const int target=blockIdx.x*blockDim.x+threadIdx.x;if(target>=targets)return;
    auto route=routes[target];const auto node=local->dd.unified_nodes[target];
    if(node.target_count && node.level>=2) {
        using T=GroupedStencilTraits<D>;const int q=grouped_target_child<D>(node.compact_morton);
        for(int b=0;b<T::CROSS_COUNT;++b) {
            const auto r=grouped_cross_route(route,b);if(!grouped_route_valid(r) || (grouped_route_mask(r)&1u))continue;
            const int i=grouped_route_node(r);
            if(slots[size_t(i)*(peers+1)]>=0 || owners[i]!=peer)route.cross_node[b]=GROUPED_ROUTE_INVALID_NODE;
        }
        for(int b=0;b<T::BLOCK_COUNT;++b) {
            const auto r=grouped_block_route(route,b);if(!grouped_route_valid(r))continue;
            const auto e=T::block(q,b);
            for(int c=0;c<T::CHILD_COUNT;++c) {
                const uint8_t bit=uint8_t(1u<<c);if(!(e.far_mask&bit) || (grouped_route_mask(r)&bit))continue;
                const int i=grouped_child_node<D>(sources,children,grouped_route_node(r),c);if(i<0)continue;
                if(slots[size_t(i)*(peers+1)]>=0 || owners[i]!=peer)route.block_omit_mask[b]|=bit;
            }
        }
    }
    output[target]=route;
}

// From remote_interaction_plan.cuh
template<int D>
__device__ FMMRemoteRef dense_fmm_remote_far_ref(int target,int slot,
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

// From remote_interaction_plan.cuh
template<int D,bool Emit>
__device__ uint64_t dense_fmm_remote_direct_row(int leaf,const PersistentTreeContext* local,
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

// From remote_interaction_plan.cuh
template<int D>
__device__ bool dense_fmm_remote_has_far(int target,const UnifiedNode& node,
    const FMMRemoteView* peers,int ranks) {
    using T=GroupedStencilTraits<D>;
    const int q=grouped_target_child<D>(node.compact_morton);
    for(int peer=0;peer<ranks;++peer) {
        const FMMRemoteView view=peers[peer];
        if(!view.count)continue;
        const auto& work=view.far_routes[target];
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

// From distributed_fmm.cuh
template<int D> __global__ void dense_fmm_owned_node_work_kernel(
    const PersistentTreeContext* ctx,int owned_nodes,uint64_t own_begin,uint64_t own_end,double4* output,
    ::cudadem::ShearSourceRouting routing,int gpu,
    const UnifiedNode* remote_nodes=nullptr,const int* remote_children=nullptr,
    const GroupedTargetInteractions* remote_routes=nullptr,bool remote_wire=false) {
    const int target=blockIdx.x*blockDim.x+threadIdx.x;if(target>=owned_nodes)return;
    const auto& node=ctx->dd.unified_nodes[target];double far=0,near=0,communication=0;
    const auto* nodes=remote_nodes?remote_nodes:ctx->dd.unified_nodes;
    const auto* children=remote_children?remote_children:ctx->dd.children_compact;
    if(node.level>=2 && node.target_count){
        using T=GroupedStencilTraits<D>;const auto work=remote_routes?remote_routes[target]:ctx->arena.d_targets[target];
        const int q=grouped_target_child<D>(node.compact_morton);
        for(int i=0;i<T::CROSS_COUNT;++i){const auto route=grouped_cross_route(work,i);
            if(!grouped_route_valid(route))continue;
            const auto source=nodes[grouped_route_node(route)];
            const bool foreign=remote_nodes && remote_wire;
            if(grouped_route_mask(route)&1u){near+=source.count;if(foreign)communication+=source.count*sizeof(FMMSourceParticle);}
            else {++far;if(foreign)communication+=sizeof(FMMSourceReply);}
        }
        for(int b=0;b<T::BLOCK_COUNT;++b){const auto route=grouped_block_route(work,b);
            if(!grouped_route_valid(route))continue;
            const auto stencil=T::block(q,b);const auto omit=grouped_block_omit_mask(work,b);
            for(int c=0;c<T::CHILD_COUNT;++c){const uint8_t bit=uint8_t(1u<<c);if(omit&bit)continue;
                const int source=grouped_child_node<D>(nodes,children,grouped_route_node(route),c);
                if(source<0)continue;
                const auto cell=nodes[source];
                const bool foreign=remote_nodes && remote_wire;
                if(stencil.far_mask&bit){if(grouped_route_mask(route)&bit){near+=cell.count;if(foreign)communication+=cell.count*sizeof(FMMSourceParticle);}
                    else {++far;if(foreign)communication+=sizeof(FMMSourceReply);}}
                else if((node.flags&UNIFIED_NODE_FLAG_IS_LEAF)&&(stencil.near_mask&bit)){near+=cell.count;if(foreign)communication+=cell.count*sizeof(FMMSourceParticle);}
            }
        }
    }
    if(remote_nodes) {output[target].x+=far;output[target].y+=near*node.target_count;output[target].z+=communication;}
    else output[target]=make_double4(far,near*node.target_count,communication,0);
}

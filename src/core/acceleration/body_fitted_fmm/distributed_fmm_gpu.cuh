#pragma once

namespace body_fitted_fmm {

template<int D> __host__ __device__ inline uint64_t gpu_cell_code(FMMCellKey key) {
    return (uint64_t{1} << (D*key.level)) | key.morton;
}
template<int D> __device__ inline FMMCellKey gpu_cell_key(uint64_t code) {
    const int level=(63-__clzll(code))/D;
    return {code^(uint64_t{1}<<(D*level)),level};
}
template<class Plan,class F> inline void fmm_cub(Plan& plan,F function) {
    size_t bytes=0;CUDA_CHECK(function(nullptr,bytes));
    plan.scratch.reserve(bytes);CUDA_CHECK(function(plan.scratch.data,bytes));
}
template<class T> inline T fmm_read_scalar(const T* device,cudaStream_t stream) {
    T value{};CUDA_CHECK(cudaMemcpyAsync(&value,device,sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));return value;
}
#include "fmm_image_sources.cuh"

template<int D> __global__ void fmm_plan_identity_kernel(const PersistentTreeContext* ctx,
    int nodes,uint64_t* previous,int* changed,bool replace) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nodes)return;
    const auto node=ctx->dd.unified_nodes[i];
    const uint64_t key=gpu_cell_code<D>({node.compact_morton,node.level}) |
        (uint64_t(bool(node.target_count))<<62) |
        (uint64_t(bool(node.flags&UNIFIED_NODE_FLAG_IS_LEAF))<<63);
    if(!replace && previous[i]!=key)atomicExch(changed,1);
    previous[i]=key;
}

template<int D,bool Emit> __global__ void fmm_halo_candidate_kernel(
    FMMSourcePublication source,int producer,int nodes,const FMMHaloBox* boxes,int box_count,
    uint64_t* counts,const uint64_t* offsets,uint64_t* keys) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nodes)return;
    const auto node=source.context->dd.unified_nodes[i];
    const int k=source.image_alignment_level;
    uint64_t n=0;
    if(node.level>=k && node.level>0) {
        uint32_t x=0,y=0,z=0;morton_decode<D>(node.compact_morton,node.level,x,y,z);
        for(int image=-source.image_layers_y;image<=source.image_layers_y;++image) {
            const int gy=int(y)+image*(1<<(node.level-k));
            if(gy<0 || gy>=(1<<node.level))continue;
            const int lowest=node.level==k?1:node.level;
            for(int level=lowest;level<=node.level;++level) {
                const int shift=node.level-level;
                const uint32_t cx=x>>shift,cy=uint32_t(gy)>>shift,cz=z>>shift;
                uint32_t owners=fmm_halo_owners(boxes,box_count,level,cx,cy,cz);
                // The producer's native cells already belong to its local
                // tree. Only translated images need publication to itself.
                // A shared key emitted by another image still refreshes its
                // complete component sum for foreign consumers.
                if(image==0)owners&=~(uint32_t{1}<<producer);
                if(!owners)continue;
                if constexpr(Emit) keys[offsets[i]+n]=gpu_cell_code<D>({morton_encode<D>(cx,cy,cz),level});
                ++n;
            }
        }
    }
    if constexpr(!Emit)counts[i]=n;
}

template<int D> __global__ void fmm_halo_decode_kernel(const uint64_t* codes,int n,
    const FMMHaloBox* boxes,int box_count,FMMCellKey* keys,int* owners) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto key=gpu_cell_key<D>(codes[i]);keys[i]=key;
    uint32_t x=0,y=0,z=0;morton_decode<D>(key.morton,key.level,x,y,z);
    owners[i]=int(fmm_halo_owners(boxes,box_count,key.level,x,y,z));
}

inline void fmm_make_query_stream(FMMQueryPeer& q,int device) {
    if(q.stream)return;
    q.device=device;CUDA_CHECK(cudaStreamCreateWithFlags(&q.stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&q.end,cudaEventDisableTiming));
}

inline std::vector<FMMHaloBox> fmm_make_halo_geometry(const DistributedFMMState& s,int gpu,int depth) {
    std::vector<::cudadem::MortonCoverCell> cover;
    if(s.local_only) {
        cover.push_back({{s.routing.min_x,s.routing.min_y,-INFINITY},
                         {s.routing.max_x,s.routing.max_y,INFINITY},gpu});
    } else cover=::cudadem::morton_cover(s.ownership_grid,s.cuts);
    if(s.routing.enabled)for(auto& cell:cover) {
        cell.lo[0]=std::max(cell.lo[0],s.routing.min_x);cell.hi[0]=std::min(cell.hi[0],s.routing.max_x);
        cell.lo[1]=std::max(cell.lo[1],s.routing.min_y);cell.hi[1]=std::min(cell.hi[1],s.routing.max_y);
    }
    return fmm_halo_boxes(s.grid,depth,cover);
}

template<int D> inline void fmm_publish_halo(DistributedFMMState& s,int gpu,bool changed,cudaStream_t stream) {
    auto& h=s.halo;auto& q=h.table;auto& p=s.gpu_plan;
    fmm_make_query_stream(q,gpu);
    if(changed) {
        h.geometry=fmm_make_halo_geometry(s,gpu,p.depth);
        h.boxes.reserve(h.geometry.size());
        if(!h.geometry.empty())CUDA_CHECK(cudaMemcpyAsync(h.boxes.data,h.geometry.data(),
            h.geometry.size()*sizeof(FMMHaloBox),cudaMemcpyHostToDevice,q.stream));
        p.counts.reserve(size_t(p.owned_nodes)+1);p.offsets.reserve(size_t(p.owned_nodes)+1);
        CUDA_CHECK(cudaMemsetAsync(p.counts.data+p.owned_nodes,0,sizeof(uint64_t),q.stream));
        if(p.owned_nodes)fmm_halo_candidate_kernel<D,false><<<(p.owned_nodes+255)/256,256,0,q.stream>>>(
            s.publication,gpu,p.owned_nodes,h.boxes.data,int(h.geometry.size()),p.counts.data,nullptr,nullptr);
        fmm_cub(h,[&](void* tmp,size_t& bytes){return cub::DeviceScan::ExclusiveSum(tmp,bytes,
            p.counts.data,p.offsets.data,p.owned_nodes+1,q.stream);});
        const uint64_t raw=fmm_read_scalar(p.offsets.data+p.owned_nodes,q.stream);
        if(raw>INT_MAX)throw std::overflow_error("FMM source halo candidate count exceeds CUB indexing");
        h.raw_keys.reserve(raw);h.sorted_keys.reserve(raw);h.unique_keys.reserve(raw);h.scalar.reserve(1);
        h.cells=0;p.raw_requests=raw;
        if(raw) {
            fmm_halo_candidate_kernel<D,true><<<(p.owned_nodes+255)/256,256,0,q.stream>>>(
                s.publication,gpu,p.owned_nodes,h.boxes.data,int(h.geometry.size()),nullptr,p.offsets.data,h.raw_keys.data);
            fmm_cub(h,[&](void* tmp,size_t& bytes){return cub::DeviceRadixSort::SortKeys(tmp,bytes,
                h.raw_keys.data,h.sorted_keys.data,int(raw),0,64,q.stream);});
            fmm_cub(h,[&](void* tmp,size_t& bytes){return cub::DeviceSelect::Unique(tmp,bytes,
                h.sorted_keys.data,h.unique_keys.data,h.scalar.data,int(raw),q.stream);});
            h.cells=fmm_read_scalar(h.scalar.data,q.stream);
        }
        q.keys.reserve(h.cells);q.replies.reserve(h.cells);h.images_only.reserve(s.routing.enabled?h.cells:0);
        h.owner_masks.reserve(h.cells);
        if(h.cells) {
            fmm_halo_decode_kernel<D><<<(h.cells+255)/256,256,0,q.stream>>>(h.unique_keys.data,h.cells,
                h.boxes.data,int(h.geometry.size()),q.keys.data,h.owner_masks.data);
            prepare_fmm_image_parts<D>(q,s.publication,h.cells,false);
        }
        ++h.generation;++p.rebuilds;
    } else ++p.reuses;
#ifdef GRANIT_FMM_TEST_HOOKS
    if(gpu==0 && s.ranks>1 && s.epoch==3 && std::getenv("GRANIT_FMM_TEST_FORCE_SOURCE_GROWTH")) {
        const auto delayed=distributed_fmm_[1].test_delay_event;
        if(!delayed || cudaEventQuery(delayed)!=cudaSuccess)
            throw std::runtime_error("producer publication grew before previous consumer completed");
        std::printf("fmm_test_consumer_event_complete_before_growth PASS\n");
        const size_t before=q.replies.capacity;q.replies.reserve(before+1);
        if(q.replies.capacity<=before)throw std::runtime_error("publication growth hook did not grow");
        std::printf("fmm_test_source_growth rank=0 epoch=3 before=%zu after=%zu PASS\n",before,q.replies.capacity);
    }
#endif
    if(h.cells) {
        if(std::getenv("GRANIT_FMM_VALIDATE_SCHEDULE") && !changed)
            validate_fmm_image_parts<D>(q,s.publication,h.cells);
        fmm_refresh_image_replies_kernel<D><<<(h.cells+7)/8,256,0,q.stream>>>(s.publication,
            q.keys.data,q.part_offsets.data,q.parts.data,h.cells,q.replies.data);
        if(s.routing.enabled)fmm_refresh_image_replies_kernel<D><<<(h.cells+7)/8,256,0,q.stream>>>(s.publication,
            q.keys.data,q.part_offsets.data,q.parts.data,h.cells,h.images_only.data,true);
    }
    CUDA_CHECK(cudaEventRecord(q.end,q.stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream,q.end,0));
}

__global__ void fmm_halo_select_kernel(const int* owners,const FMMSourceReply* values,int n,
    int consumer,int* ids,int* flags) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    ids[i]=i;flags[i]=bool(uint32_t(owners[i])&(uint32_t{1}<<consumer)) && values[i].count>0;
}
__global__ void fmm_halo_gather_kernel(const int* ids,int n,const FMMCellKey* keys,
    const FMMSourceReply* values,FMMCellKey* out_keys,FMMSourceReply* out_values) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    out_keys[i]=keys[ids[i]];out_values[i]=values[ids[i]];
}

template<int D> __device__ int fmm_halo_lookup(const FMMCellKey* keys,int count,FMMCellKey key) {
    const uint64_t code=gpu_cell_code<D>(key);int lo=0,hi=count;
    while(lo<hi) {const int mid=lo+(hi-lo)/2;
        if(gpu_cell_code<D>(keys[mid])<code)lo=mid+1;else hi=mid;}
    return lo<count && gpu_cell_code<D>(keys[lo])==code?lo:-1;
}

template<int D> __global__ void fmm_halo_nodes_kernel(const FMMCellKey* keys,
    const FMMSourceReply* values,int count,int depth,::cudadem::MortonGrid grid,
    const PersistentTreeContext* target,UnifiedNode* nodes,int* children) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto key=keys[i];UnifiedNode node{};
    uint32_t x=0,y=0,z=0;morton_decode<D>(key.morton,key.level,x,y,z);
    const double h=ldexp(grid.size,-key.level);
    node.center=make_double3(grid.x+(x+0.5)*h,grid.y+(y+0.5)*h,
        D==2?target->dd.unified_nodes[0].center.z:grid.z+(z+0.5)*h);
    node.radius=float(0.5*sqrt(3.0)*h);node.level=key.level;node.compact_morton=key.morton;
    node.count=values[i].count;node.global_count=values[i].global_count;
    node.flags=UNIFIED_NODE_FLAG_VALID|(key.level==depth?UNIFIED_NODE_FLAG_IS_LEAF:0);
    int child_count=0;
    if(key.level<depth)for(int c=0;c<(1<<D);++c) {
        const int child=fmm_halo_lookup<D>(keys,count,{(key.morton<<D)|uint64_t(c),key.level+1});
        if(child>=0) {node.childMask|=uint8_t(1u<<c);children[size_t(i)*8+child_count++]=child;}
    }
    node.childBase=child_count?uint32_t(i*8):UNIFIED_NODE_LEAF_CHILDBASE;
    nodes[i]=node;
}
__global__ void fmm_halo_initialize_routes_kernel(const PersistentTreeContext* target,int n,
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

template<int D> __global__ void fmm_halo_count_routes_kernel(const PersistentTreeContext* local,
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
__global__ void fmm_halo_publish_counts_kernel(PersistentTreeContext* local,
    const int* counts,int64_t* remote) {
    const int c=threadIdx.x;if(c>=4)return;
    if(remote)remote[c]+=counts[c];else local->metadata.task_counts[c]=counts[c];
}

// One source parent owns each (target, stencil-slot) write. Different source
// parents may touch one target record, hence only its packed CROSS bits need
// atomic OR. Numerical output remains exclusively target-owned.
template<int D> __global__ void fmm_halo_inverse_routes_kernel(
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

// Single-device shear retains its population threshold using the complete
// logical cell population. Only route masks change; local nodes/moments do not.
template<int D> __global__ void fmm_halo_local_image_policy_kernel(
    PersistentTreeContext* local,const FMMCellKey* keys,const FMMSourceReply* values,int n,int targets) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=targets)return;
    const auto target=local->dd.unified_nodes[i];
    if(!target.target_count || target.level<2)return;
    using T=GroupedStencilTraits<D>;auto& out=local->arena.d_targets[i];
    const bool leaf=target.flags&UNIFIED_NODE_FLAG_IS_LEAF;
    const int target_slot=fmm_halo_lookup<D>(keys,n,{target.compact_morton,target.level});
    const uint32_t target_population=target_slot>=0?values[target_slot].global_count:target.count;
    uint32_t cross_bits=0;
    for(int b=0;b<T::CROSS_COUNT;++b) {
        const auto route=grouped_cross_route(out,b);if(!grouped_route_valid(route))continue;
        const auto source=local->dd.unified_nodes[grouped_route_node(route)];
        const int h=fmm_halo_lookup<D>(keys,n,{source.compact_morton,source.level});
        const uint32_t count=h>=0?values[h].global_count:source.count;
        const auto cutoff=persistent_cutoff_relation(target,source,local->command.cutoff_radius,local->metadata.root_cell_size);
        if(cutoff==CutoffRelation::Straddles || uint64_t(count)*target_population<=grouped_direct_threshold(target.level,leaf))cross_bits|=1u<<b;
    }
    out.cross_direct_bits=cross_bits;
    const int q=grouped_target_child<D>(target.compact_morton);
    for(int b=0;b<T::BLOCK_COUNT;++b) {
        const auto route=grouped_block_route(out,b);if(!grouped_route_valid(route))continue;
        const auto e=T::block(q,b);uint8_t direct=0;
        for(int c=0;c<T::CHILD_COUNT;++c) {
            const int index=grouped_child_node<D>(local->dd.unified_nodes,local->dd.children_compact,grouped_route_node(route),c);
            if(index<0)continue;
            const auto source=local->dd.unified_nodes[index];const uint8_t bit=uint8_t(1u<<c);
            const auto cutoff=persistent_cutoff_relation(target,source,local->command.cutoff_radius,local->metadata.root_cell_size);
            if(cutoff==CutoffRelation::Outside)continue;
            const int h=fmm_halo_lookup<D>(keys,n,{source.compact_morton,source.level});
            const uint32_t count=h>=0?values[h].global_count:source.count;
            if((e.far_mask&bit) && (cutoff==CutoffRelation::Straddles ||
                uint64_t(count)*target_population<=grouped_direct_threshold(target.level,leaf)))direct|=bit;
            if(leaf && (e.near_mask&bit))direct|=bit;
        }
        out.block_direct_mask[b]=direct;
    }
}

__global__ void fmm_halo_particle_ranges_kernel(const int* ids,const int* direct,int n,
    const FMMSourceReply* values,uint64_t* counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>n)return;
    if(i==n) {counts[i]=0;return;}
    counts[i]=direct[i]?uint64_t(values[ids[i]].count):0;
}
__global__ void fmm_halo_particle_requests_kernel(const int* ids,const uint64_t* counts,
    const uint64_t* offsets,int n,FMMParticleRequest* requests,int* packed_offsets) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>n)return;
    packed_offsets[i]=int(offsets[i]); // host checked the entire prefix before narrowing
    if(i==n)return;
    requests[i]={uint32_t(ids[i]),uint32_t(counts[i]),uint32_t(offsets[i])};
}
__global__ void fmm_halo_refresh_nodes_kernel(const FMMSourceReply* values,const int* offsets,int n,
    UnifiedNode* nodes,fmm::MultipoleExpansion* moments) {
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;if(i>=n)return;
    moments[i].coeffs[lane]=values[i].coeffs[lane];
    if(!lane) {nodes[i].begin=offsets[i];nodes[i].count=values[i].count;nodes[i].global_count=values[i].global_count;}
}
__global__ void fmm_halo_identity_order_kernel(int* order,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)order[i]=i;
}

inline GroupedInteractionArena fmm_halo_arena(FMMQueryPeer& q,int targets) {
    GroupedInteractionArena arena{};arena.d_targets=q.halo_routes.data;arena.node_capacity=targets;return arena;
}

template<int D> inline void fmm_receive_halo(int gpu,int peer,DistributedFMMState& s,
    PersistentTreeRuntime* runtime,cudaStream_t stream,bool collective) {
    auto& producer=distributed_fmm_[peer];auto& h=producer.halo;auto& q=*s.peers[peer];
    const bool self=peer==gpu;const auto* values=self?h.images_only.data:h.table.replies.data;
    const bool selection_changed=q.source_generation!=h.generation;
    CUDA_CHECK(cudaSetDevice(peer));fmm_make_query_stream(q,peer);
    if(selection_changed) {
        q.source_nodes.reserve(h.cells);q.select_flags.reserve(h.cells);q.selected.reserve(h.cells);q.validation_error.reserve(1);
        q.request_count=0;
        if(h.cells) {
            fmm_halo_select_kernel<<<(h.cells+255)/256,256,0,q.stream>>>(h.owner_masks.data,values,h.cells,gpu,
                q.source_nodes.data,q.select_flags.data);
            fmm_cub(q,[&](void* tmp,size_t& bytes){return cub::DeviceSelect::Flagged(tmp,bytes,
                q.source_nodes.data,q.select_flags.data,q.selected.data,q.validation_error.data,h.cells,q.stream);});
            q.request_count=fmm_read_scalar(q.validation_error.data,q.stream);
        }
        q.source_generation=h.generation;++s.let_schedule_lookups;
    } else ++s.let_schedule_reuses;
    const int n=q.request_count;
    if(!n) {q.particle_count=0;CUDA_CHECK(cudaSetDevice(gpu));return;}
    q.keys.reserve(n);q.replies.reserve(n);
    fmm_halo_gather_kernel<<<(n+255)/256,256,0,q.stream>>>(q.selected.data,n,h.table.keys.data,values,q.keys.data,q.replies.data);
    CUDA_CHECK(cudaEventRecord(q.end,q.stream));
    CUDA_CHECK(cudaSetDevice(gpu));
    q.remote_keys.reserve(n);q.remote_values.reserve(n);q.halo_nodes.reserve(n);q.halo_children.reserve(size_t(n)*8);
    q.halo_multipoles.reserve(n);q.direct_flags.reserve(n);q.halo_routes.reserve(s.gpu_plan.owned_nodes);q.route_counts.reserve(4);
    const bool population_dependent=!collective; // single-device thresholds use live full counts
    const bool rebuild=selection_changed || q.route_generation!=s.publication_generation || population_dependent;
    CUDA_CHECK(cudaStreamWaitEvent(stream,q.end,0));
    if(selection_changed)::cudadem::copy_peer_payload(q.remote_keys.data,gpu,q.keys.data,peer,n*sizeof(FMMCellKey),stream);
    ::cudadem::copy_peer_payload(q.remote_values.data,gpu,q.replies.data,peer,n*sizeof(FMMSourceReply),stream);
    if(!self)s.let_wire_bytes+=n*sizeof(FMMSourceReply)+(selection_changed?n*sizeof(FMMCellKey):0);
    if(self && !collective) {
        fmm_halo_local_image_policy_kernel<D><<<(s.gpu_plan.owned_nodes+255)/256,256,0,stream>>>(
            runtime->device_context,q.remote_keys.data,q.remote_values.data,n,s.gpu_plan.owned_nodes);
        CUDA_CHECK(cudaMemsetAsync(runtime->host_context.arena.d_counters,0,4*sizeof(int),stream));
        fmm_halo_count_routes_kernel<D><<<(s.gpu_plan.owned_nodes+255)/256,256,0,stream>>>(runtime->device_context,
            runtime->host_context.dd.unified_nodes,runtime->host_context.dd.children_compact,runtime->host_context.arena.d_targets,
            s.gpu_plan.owned_nodes,runtime->host_context.arena.d_counters,true);
        fmm_halo_publish_counts_kernel<<<1,4,0,stream>>>(runtime->device_context,runtime->host_context.arena.d_counters,nullptr);
    }
    if(rebuild) {
        fmm_halo_nodes_kernel<D><<<(n+255)/256,256,0,stream>>>(q.remote_keys.data,q.remote_values.data,n,s.gpu_plan.depth,
            s.grid,runtime->device_context,q.halo_nodes.data,q.halo_children.data);
        fmm_halo_initialize_routes_kernel<<<(std::max(n,s.gpu_plan.owned_nodes)+255)/256,256,0,stream>>>(runtime->device_context,
            s.gpu_plan.owned_nodes,q.halo_routes.data,q.direct_flags.data,n);
        fmm_halo_inverse_routes_kernel<D><<<(n*(1<<D)+255)/256,256,0,stream>>>(runtime->device_context,q.halo_nodes.data,
            q.halo_children.data,q.remote_keys.data,n,s.gpu_plan.depth,collective,q.halo_routes.data,q.direct_flags.data);
        CUDA_CHECK(cudaMemsetAsync(q.route_counts.data,0,4*sizeof(int),stream));
        fmm_halo_count_routes_kernel<D><<<(s.gpu_plan.owned_nodes+255)/256,256,0,stream>>>(runtime->device_context,
            q.halo_nodes.data,q.halo_children.data,q.halo_routes.data,s.gpu_plan.owned_nodes,q.route_counts.data,false);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaSetDevice(peer));q.direct_on_producer.reserve(n);
        ::cudadem::copy_peer_payload(q.direct_on_producer.data,peer,q.direct_flags.data,gpu,n*sizeof(int),q.stream);
        q.route_generation=s.publication_generation;++s.gpu_plan.topology_rebuilds;
    } else {CUDA_CHECK(cudaSetDevice(peer));++s.gpu_plan.topology_reuses;}
    q.particle_counts.reserve(n+1);q.particle_offsets.reserve(n+1);q.particle_requests.reserve(n);
    q.packed_particle_offsets.reserve(n+1);
    fmm_halo_particle_ranges_kernel<<<(n+256)/256,256,0,q.stream>>>(q.selected.data,q.direct_on_producer.data,n,values,q.particle_counts.data);
    fmm_cub(q,[&](void* tmp,size_t& bytes){return cub::DeviceScan::ExclusiveSum(tmp,bytes,
        q.particle_counts.data,q.particle_offsets.data,n+1,q.stream);});
    const uint64_t total_particles=fmm_read_scalar(q.particle_offsets.data+n,q.stream);
    if(total_particles>INT_MAX)throw std::overflow_error("FMM halo particle payload exceeds signed indexing");
    const int particles=int(total_particles);
    q.particles.reserve(particles);q.particle_count=particles;
    if(s.debug_coverage) {
        q.debug_source_ids=producer.debug_arena_ids;
        std::vector<int> ids(n),offsets(h.cells+1);
        CUDA_CHECK(cudaMemcpy(ids.data(),q.selected.data,n*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(offsets.data(),h.table.part_offsets.data,offsets.size()*sizeof(int),cudaMemcpyDeviceToHost));
        std::vector<FMMSourcePart> parts(offsets.back());
        if(!parts.empty())CUDA_CHECK(cudaMemcpy(parts.data(),h.table.parts.data,parts.size()*sizeof(FMMSourcePart),cudaMemcpyDeviceToHost));
        q.debug_memberships.assign(n,{});
        for(int i=0;i<n;++i)for(int at=offsets.at(ids[i]);at<offsets.at(ids[i]+1);++at) {
            const auto& members=producer.debug_node_members.at(parts[at].node);
            q.debug_memberships[i].insert(q.debug_memberships[i].end(),members.begin(),members.end());
        }
    }
    fmm_halo_particle_requests_kernel<<<(n+256)/256,256,0,q.stream>>>(q.selected.data,q.particle_counts.data,
        q.particle_offsets.data,n,q.particle_requests.data,q.packed_particle_offsets.data);
    if(particles)fmm_pack_image_particles_kernel<<<std::min(n,8192),256,0,q.stream>>>(producer.publication,
        h.table.part_offsets.data,h.table.parts.data,q.particle_requests.data,q.particles.data,n,self);
    CUDA_CHECK(cudaEventRecord(q.end,q.stream));
    CUDA_CHECK(cudaSetDevice(gpu));
    q.local_ids.reserve(n+1);q.halo_particles.reserve(particles);q.halo_order.reserve(particles);
    q.halo_x.reserve(particles);q.halo_y.reserve(particles);q.halo_z.reserve(particles);q.halo_mass.reserve(particles);
    CUDA_CHECK(cudaStreamWaitEvent(stream,q.end,0));
    ::cudadem::copy_peer_payload(q.local_ids.data,gpu,q.packed_particle_offsets.data,peer,(n+1)*sizeof(int),stream);
    fmm_halo_refresh_nodes_kernel<<<(n+7)/8,256,0,stream>>>(q.remote_values.data,q.local_ids.data,n,q.halo_nodes.data,q.halo_multipoles.data);
    if(particles) {
        ::cudadem::copy_peer_payload(q.halo_particles.data,gpu,q.particles.data,peer,size_t(particles)*sizeof(FMMSourceParticle),stream);
        fmm_unpack_source_particles_kernel<<<(size_t(particles)+255)/256,256,0,stream>>>(q.halo_particles.data,particles,0,
            q.halo_x.data,q.halo_y.data,q.halo_z.data,q.halo_mass.data);
        fmm_halo_identity_order_kernel<<<(size_t(particles)+255)/256,256,0,stream>>>(q.halo_order.data,particles);
    }
    if(!self)s.let_wire_bytes+=(n+1)*sizeof(int)+size_t(particles)*sizeof(FMMSourceParticle);
    s.received_cells+=n;s.received_particle_count+=particles;
    fmm_halo_publish_counts_kernel<<<1,4,0,stream>>>(runtime->device_context,q.route_counts.data,s.remote_interactions.data);
}

// Bind identical cell coefficients without combining any tree topology. The
// first contributing peer owns the reduction. A native cell stays on the local
// traversal; otherwise exactly one independent remote route contracts it.
template<int D> __global__ void fmm_halo_moment_bind_kernel(
    const PersistentTreeContext* local,const FMMHaloMomentView* views,int peers,
    int peer,int* slots,int* owners) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=views[peer].count)return;
    const auto key=views[peer].keys[i];uint32_t x=0,y=0,z=0;
    morton_decode<D>(key.morton,key.level,x,y,z);
    slots[size_t(i)*(peers+1)]=persistent_coordinate_lookup<D>(local,key.level,x,y,z);
    int owner=peer;
    for(int p=0;p<peers;++p) {
        const int index=fmm_halo_lookup<D>(views[p].keys,views[p].count,key);
        slots[size_t(i)*(peers+1)+p+1]=index;
        if(index>=0)owner=std::min(owner,p);
    }
    owners[i]=owner;
}
__global__ void fmm_halo_moment_reduce_kernel(const FMMHaloMomentView* views,int peers,
    int peer,const int* slots,const int* owners,const fmm::MultipoleExpansion* native,
    fmm::MultipoleExpansion* overlay,fmm::MultipoleExpansion* remote) {
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    if(i>=views[peer].count || owners[i]!=peer)return;
    const int* row=slots+size_t(i)*(peers+1);const int local=row[0];
    far_real_t value=local>=0?native[local].coeffs[lane]:FAR_REAL(0.0);
    for(int p=0;p<peers;++p)if(row[p+1]>=0)value+=views[p].values[row[p+1]].coeffs[lane];
    if(local>=0)overlay[local].coeffs[lane]=value;
    else remote[i].coeffs[lane]=value;
}
template<int D> __global__ void fmm_halo_moment_routes_kernel(
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

template<int D> inline void fmm_prepare_halo_moments(int gpu,DistributedFMMState& s,
    const PersistentTreeRuntime* runtime,const fmm::FMMDeviceData* dd,cudaStream_t stream) {
    const int targets=s.gpu_plan.owned_nodes;
    s.has_moment_overlay=targets>0 && s.received_cells>0;
    if(!s.has_moment_overlay) {s.moment_local_generation=~uint64_t{0};return;}
    auto& views=s.host_moment_views;views.assign(s.ranks,{});
    std::vector<uint64_t> generations(s.ranks,~uint64_t{0});
    for(int peer=0;peer<s.ranks;++peer) {
        if((peer==gpu&&!s.routing.enabled)||(s.local_only&&peer!=gpu))continue;
        const auto& q=*s.peers[peer];views[peer]={q.remote_keys.data,q.remote_values.data,q.request_count};
        generations[peer]=q.source_generation;
    }
    const bool rebuild=s.moment_local_generation!=s.publication_generation || generations!=s.moment_generations;
    s.moment_views.reserve(s.ranks);s.moment_overlay.reserve(targets);
    CUDA_CHECK(cudaMemcpyAsync(s.moment_views.data,views.data(),s.ranks*sizeof(FMMHaloMomentView),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(s.moment_overlay.data,dd->multipoles,targets*sizeof(fmm::MultipoleExpansion),cudaMemcpyDeviceToDevice,stream));
    // Population-dependent single-device masks refresh even when cell keys do not.
    const bool route_rebuild=rebuild || s.ranks==1 || s.local_only;
    CUDA_CHECK(cudaMemsetAsync(s.remote_interactions.data,0,4*sizeof(int64_t),stream));
    for(int peer=0;peer<s.ranks;++peer) {
        const int n=views[peer].count;if(!n)continue;
        auto& q=*s.peers[peer];
        if(rebuild) {
            q.moment_slots.reserve(size_t(n)*(s.ranks+1));q.moment_owner.reserve(n);
            fmm_halo_moment_bind_kernel<D><<<(n+255)/256,256,0,stream>>>(runtime->device_context,
                s.moment_views.data,s.ranks,peer,q.moment_slots.data,q.moment_owner.data);
        }
        if(route_rebuild) {
            q.moment_routes.reserve(targets);
            fmm_halo_moment_routes_kernel<D><<<(targets+255)/256,256,0,stream>>>(runtime->device_context,
                q.halo_nodes.data,q.halo_children.data,q.halo_routes.data,q.moment_routes.data,
                targets,peer,s.ranks,q.moment_slots.data,q.moment_owner.data);
            CUDA_CHECK(cudaMemsetAsync(q.route_counts.data,0,4*sizeof(int),stream));
            fmm_halo_count_routes_kernel<D><<<(targets+255)/256,256,0,stream>>>(runtime->device_context,
                q.halo_nodes.data,q.halo_children.data,q.moment_routes.data,targets,q.route_counts.data,false);
        }
        fmm_halo_moment_reduce_kernel<<<(n+7)/8,256,0,stream>>>(s.moment_views.data,s.ranks,peer,
            q.moment_slots.data,q.moment_owner.data,dd->multipoles,s.moment_overlay.data,q.halo_multipoles.data);
        fmm_halo_publish_counts_kernel<<<1,4,0,stream>>>(runtime->device_context,q.route_counts.data,s.remote_interactions.data);
    }
    s.moment_local_generation=s.publication_generation;s.moment_generations=std::move(generations);
}

#include "remote_interaction_plan.cuh"

inline void exchange_distributed_fmm_sources(int gpu,fmm::FMMDeviceData* dd,
    PersistentTreeRuntime* runtime,::cudadem::SimulationState& particles,
    const ::cudadem::SimConfig& config,cudaStream_t stream) {
    auto& s=distributed_fmm_[gpu];auto& p=s.gpu_plan;
    const bool collective=s.ranks>1&&!s.local_only,images=s.routing.enabled;
    s.received_cells=s.received_particle_count=s.let_wire_bytes=0;s.communication_ms=0;
    s.has_moment_overlay=false;
    s.remote_interactions.reserve(4);CUDA_CHECK(cudaMemsetAsync(s.remote_interactions.data,0,4*sizeof(int64_t),stream));
    const int targets=particles.particles.num_local;
    const int owned=images?(runtime?particles.particles.num_particles:0):targets;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    s.publication={owned?runtime->device_context:nullptr,owned?dd->sorted_to_original:nullptr,
        particles.particles.pos_x,particles.particles.pos_y,particles.particles.pos_z,
        particles.particles.mass,owned,particles.particles.global_id,targets};
    p.owned_nodes=0;p.depth=config.acceleration.unified_fmm.tree_depth;
    double effective_cutoff=0;
    if(owned) {
        const auto metadata=snapshot_persistent_tree_metadata(*runtime);
        if(metadata.error)throw std::runtime_error("invalid local FMM tree before halo publication");
        p.owned_nodes=metadata.num_nodes;p.depth=runtime->host_context.tree_depth;
        effective_cutoff=metadata.cutoff_radius;
    }
    if(images) {
        s.publication.image_layers_y=s.routing.layers_y;s.publication.image_length_y=s.routing.length_y;
        int level=0;while(std::ldexp(s.routing.length_y,level)<s.grid.size)++level;
        if(std::ldexp(s.routing.length_y,level)!=s.grid.size || (owned && level>p.depth))
            throw std::logic_error("canonical image grid must align with Y translations");
        s.publication.image_alignment_level=level;
    }
    s.debug_coverage=std::getenv("GRANIT_FMM_VALIDATE_COVERAGE")!=nullptr;
    if(s.debug_coverage && images)throw std::invalid_argument("use finite-image force oracle for shear coverage");
    if(s.debug_coverage) {
        if(owned>8192)throw std::runtime_error("coverage debug is restricted to small fixtures");
        s.debug_arena_ids.resize(owned);std::vector<int> order(owned);std::vector<UnifiedNode> nodes(p.owned_nodes);
        if(owned) {
            CUDA_CHECK(cudaMemcpy(s.debug_arena_ids.data(),s.publication.global_ids,owned*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(order.data(),s.publication.sorted,owned*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(nodes.data(),dd->unified_nodes,nodes.size()*sizeof(UnifiedNode),cudaMemcpyDeviceToHost));
        }
        s.debug_index_count=owned;s.debug_node_members.assign(p.owned_nodes,{});s.debug_target_members.assign(p.owned_nodes,{});
        for(int i=0;i<p.owned_nodes;++i)for(uint32_t j=0;j<nodes[i].count;++j) {
            const int original=order.at(nodes[i].begin+j);const int id=s.debug_arena_ids.at(original);
            s.debug_node_members[i].push_back(id);
            if(original<targets)s.debug_target_members[i].push_back(id);
        }
    }
    if(!collective&&!images) {
        s.remote_lists.far_count=s.remote_lists.near_count=0;
        s.remote_lists.valid=false;
        return;
    }
    BF_FMM_NVTX_RANGE("BF-FMM: Distributed Exchange");
    const auto started=std::chrono::steady_clock::now();
    const auto& old=s.published_grid;const auto& grid=s.grid;
    const bool structural=p.cached_nodes<0 || p.cutoff!=effective_cutoff || s.published_nodes!=p.owned_nodes || s.published_depth!=p.depth ||
        s.published_context!=s.publication.context || s.published_arena!=(owned?dd->unified_nodes:nullptr) ||
        old.generation!=grid.generation || old.dimension!=grid.dimension || old.x!=grid.x ||
        old.y!=grid.y || old.z!=grid.z || old.size!=grid.size;
    s.published_changed.reserve(1);s.published_identity.reserve(p.owned_nodes);
    CUDA_CHECK(cudaMemsetAsync(s.published_changed.data,0,sizeof(int),stream));
    if(p.owned_nodes) {
        if(grid.dimension==2)fmm_plan_identity_kernel<2><<<(p.owned_nodes+255)/256,256,0,stream>>>(
            runtime->device_context,p.owned_nodes,s.published_identity.data,s.published_changed.data,structural);
        else fmm_plan_identity_kernel<3><<<(p.owned_nodes+255)/256,256,0,stream>>>(
            runtime->device_context,p.owned_nodes,s.published_identity.data,s.published_changed.data,structural);
    }
    const bool changed=bool(fmm_read_scalar(s.published_changed.data,stream))||structural;
    if(changed)++s.publication_generation;
    p.cached_nodes=p.owned_nodes;p.cutoff=effective_cutoff;s.published_nodes=p.owned_nodes;s.published_depth=p.depth;
    s.published_context=s.publication.context;s.published_arena=owned?dd->unified_nodes:nullptr;s.published_grid=grid;
    if(grid.dimension==2)fmm_publish_halo<2>(s,gpu,changed,stream);else fmm_publish_halo<3>(s,gpu,changed,stream);
    std::vector<unsigned char> original_nodes,original_moments,original_children;
    if(owned && std::getenv("GRANIT_FMM_VALIDATE_SCHEDULE")) {
        original_nodes.resize(size_t(p.owned_nodes)*sizeof(UnifiedNode));
        original_moments.resize(size_t(p.owned_nodes)*sizeof(fmm::MultipoleExpansion));
        original_children.resize(size_t(p.owned_nodes)*8*sizeof(int));
        CUDA_CHECK(cudaMemcpy(original_nodes.data(),dd->unified_nodes,original_nodes.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(original_moments.data(),dd->multipoles,original_moments.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(original_children.data(),dd->children_compact,original_children.size(),cudaMemcpyDeviceToHost));
    }
    const auto phase=::cudadem::PeerPhase::FmmSource;
    if(collective) {
        fmm_collective_check(s.coordinator->publish(phase,gpu,s.epoch,stream),s);
        for(int peer=0;peer<s.ranks;++peer)if(peer!=gpu)
            fmm_collective_check(s.coordinator->wait_for_published(phase,peer,gpu,s.epoch,stream),s);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int peer=0;peer<s.ranks;++peer) {
        if((peer==gpu&&!images)||(s.local_only&&peer!=gpu))continue;
        if(owned) {
            if(grid.dimension==2)fmm_receive_halo<2>(gpu,peer,s,runtime,stream,collective);
            else fmm_receive_halo<3>(gpu,peer,s,runtime,stream,collective);
        } else {s.peers[peer]->request_count=0;s.peers[peer]->particle_count=0;
            s.peers[peer]->source_generation=~uint64_t{0};s.peers[peer]->route_generation=~uint64_t{0};}
    }
    if(grid.dimension==2)fmm_prepare_halo_moments<2>(gpu,s,runtime,dd,stream);
    else fmm_prepare_halo_moments<3>(gpu,s,runtime,dd,stream);
    if(grid.dimension==2)fmm_prepare_remote_lists<2>(gpu,s,runtime,stream);
    else fmm_prepare_remote_lists<3>(gpu,s,runtime,stream);
#ifdef GRANIT_FMM_TEST_HOOKS
    if(gpu==1 && s.epoch==2 && std::getenv("GRANIT_FMM_TEST_DELAY_CONSUMER")) {
        if(!s.test_delay_event)CUDA_CHECK(cudaEventCreateWithFlags(&s.test_delay_event,cudaEventDisableTiming));
        CUDA_CHECK(cudaLaunchHostFunc(stream,[](void*) {std::this_thread::sleep_for(std::chrono::milliseconds(50));},nullptr));
        CUDA_CHECK(cudaEventRecord(s.test_delay_event,stream));
        std::printf("fmm_test_delayed_consumer rank=1 epoch=2 delay_ms=50\n");
    }
#endif
    // Independent receive arrays outlive peer publication. There is no local
    // arena mutation requiring a same-epoch all-consumer barrier. The existing
    // prepare_producer dependency protects the next local tree rebuild.
    for(int peer=0;peer<s.ranks;++peer)if(collective&&peer!=gpu)
        fmm_collective_check(s.coordinator->acknowledge_consumed(phase,peer,gpu,s.epoch,stream),s);
    if(!original_nodes.empty()) {
        const auto unchanged=[&](const std::vector<unsigned char>& before,const void* device) {
            std::vector<unsigned char> after(before.size());
            CUDA_CHECK(cudaMemcpyAsync(after.data(),device,after.size(),cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if(before!=after)throw std::runtime_error("source halo modified the local tree or original moments");
        };
        unchanged(original_nodes,dd->unified_nodes);unchanged(original_moments,dd->multipoles);
        unchanged(original_children,dd->children_compact);
        std::printf("fmm_local_tree_unchanged rank=%d epoch=%llu PASS\n",gpu,(unsigned long long)s.epoch);
    }
    s.communication_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS"))std::printf(
        "[FMM source halo] rank=%d epoch=%llu local_nodes=%d published=%d received=%zu particles=%zu rebuilds=%llu reuses=%llu wire_bytes=%zu ms=%.6f\n",
        gpu,(unsigned long long)s.epoch,p.owned_nodes,s.halo.cells,s.received_cells,s.received_particle_count,
        (unsigned long long)p.rebuilds,(unsigned long long)p.reuses,s.let_wire_bytes,s.communication_ms);
    if(std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS"))std::printf(
        "[FMM halo] rank=%d contributors=%zu topology_rebuilds=%llu topology_reuses=%llu source_hits=%llu source_lookups=%llu wire_bytes=%zu\n",
        gpu,s.received_cells,(unsigned long long)p.topology_rebuilds,(unsigned long long)p.topology_reuses,
        (unsigned long long)s.let_schedule_reuses,(unsigned long long)s.let_schedule_lookups,s.let_wire_bytes);
    BF_FMM_NVTX_END();
}
} // namespace body_fitted_fmm

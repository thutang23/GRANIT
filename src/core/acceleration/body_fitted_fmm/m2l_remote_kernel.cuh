#pragma once

// Included inside body_fitted_fmm after the shared M2L arithmetic helpers.
#if FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<int D>
__device__ __forceinline__ void remote_m2l_tc_batches(
    int q,FMMRemoteM2LList list,const UnifiedNode* nodes,const float* table,
    fmm::LocalExpansion* output,far_real_t root_size,int* targets,int* sources,
    far_real_t* scales,far_real_t* powers,float* b,float* c) {
    using T=GroupedStencilTraits<D>;
    const int lane=threadIdx.x&31;
    const int end=list.class_offsets[q+1];
    for(int row=list.class_offsets[q]+int(blockIdx.x)*M2L_TC_BATCH;
        row<end;row+=int(gridDim.x)*M2L_TC_BATCH) {
        const int count=min(M2L_TC_BATCH,end-row);
        BF_M2L_CENSUS_BATCH(1,q,count);
        if(lane<count) {
            targets[lane]=list.targets[row+lane];
            const far_real_t h=root_size/static_cast<far_real_t>(1ull<<nodes[targets[lane]].level);
            far_real_t power=FAR_REAL(1.0);
            for(int order=0;order<5;++order) {powers[lane*5+order]=power;power*=h;}
        }
        __syncwarp();
        far_real_t totals[M2L_TC_THREAD_TOTALS]{};
        for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
            const bool cross=slot<T::CROSS_COUNT;
            const int child_slot=slot-T::CROSS_COUNT;
            uint16_t op;
            if(cross)op=T::cross(q,slot).operator_id;
            else {
                const auto stencil=T::block(q,child_slot/T::CHILD_COUNT);
                const int child=child_slot%T::CHILD_COUNT;
                if(!(stencil.far_mask&(1u<<child)))continue;
                op=stencil.operator_id[child];
            }
            if(lane<M2L_TC_BATCH) {
                const int ref=slot*list.count+row+lane;
                sources[lane]=lane<count && list.sources[ref].node>=0?ref:-1;
            }
            __syncwarp();
            BF_M2L_CENSUS_ROUTE(1,q,slot,sources,count);
            grouped_m2l_tc_apply_sources<true>(targets,count,op,cross?1:0,sources,
                nodes,nullptr,nullptr,table,scales,powers,b,c,totals,list.sources,list.peers);
        }
        // Exactly one warp owns this target batch, across every source peer.
        for(int item=0;item<M2L_TC_THREAD_TOTALS;++item) {
            const int element=lane+item*32,column=element&(M2L_TC_BATCH-1);
            if(column<count)output[targets[column]].coeffs[element/M2L_TC_BATCH]+=totals[item];
        }
        __syncwarp();
    }
}
#endif

template<int D>
__device__ __forceinline__ far_real_t remote_m2l_reference(
    int row,int lane,FMMRemoteM2LList list,const UnifiedNode* nodes,
    const float* table,far_real_t root_size) {
    using T=GroupedStencilTraits<D>;
    const int target=list.targets[row];
    const int q=grouped_target_child<D>(nodes[target].compact_morton);
    far_real_t value=FAR_REAL(0.0);
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
        const int ref=slot*list.count+row;
        if(list.sources[ref].node<0)continue;
        const bool cross=slot<T::CROSS_COUNT;
        const int at=slot-T::CROSS_COUNT;
        const uint16_t op=cross?T::cross(q,slot).operator_id:
            T::block(q,at/T::CHILD_COUNT).operator_id[at%T::CHILD_COUNT];
        value+=grouped_m2l_contribution<true>(target,ref,op,cross?1:0,lane,
            nodes,nullptr,nullptr,table,root_size,list.sources,list.peers);
    }
    return value;
}

__global__ void fmm_m2l_remote_targets_kernel(
    FMMRemoteM2LList list,const PersistentTreeMetadata* metadata,
    const UnifiedNode* nodes,const float* table,fmm::LocalExpansion* output) {
#if FMM_M2L_MODE == FMM_M2L_MODE_TF32_TC && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(32) int targets[8][M2L_TC_BATCH],sources[8][M2L_TC_BATCH];
    __shared__ __align__(32) far_real_t scales[8][M2L_TC_BATCH*7],powers[8][M2L_TC_BATCH*5];
    __shared__ __align__(32) float b[8][M2L_TC_PADDED_NC*M2L_TC_BATCH],c[8][M2L_TC_PADDED_NC*M2L_TC_BATCH];
    const int q=threadIdx.x>>5;
    if(q>=(metadata->active_tree==2?4:8))return;
    if(metadata->active_tree==2)remote_m2l_tc_batches<2>(q,list,nodes,table,output,
        FAR_CAST(metadata->root_cell_size),targets[q],sources[q],scales[q],powers[q],b[q],c[q]);
    else remote_m2l_tc_batches<3>(q,list,nodes,table,output,
        FAR_CAST(metadata->root_cell_size),targets[q],sources[q],scales[q],powers[q],b[q],c[q]);
#else
    const int lane=threadIdx.x&31;if(lane>=NC)return;
    const int first=(blockIdx.x*blockDim.x+threadIdx.x)/32,stride=gridDim.x*blockDim.x/32;
    for(int row=first;row<list.count;row+=stride) {
        const far_real_t value=metadata->active_tree==2
            ?remote_m2l_reference<2>(row,lane,list,nodes,table,FAR_CAST(metadata->root_cell_size))
            :remote_m2l_reference<3>(row,lane,list,nodes,table,FAR_CAST(metadata->root_cell_size));
        output[list.targets[row]].coeffs[lane]+=value;
    }
#endif
}

inline void run_m2l_remote_targets(FMMRemoteM2LList list,const PersistentTreeMetadata* metadata,
    const UnifiedNode* nodes,const float* table,fmm::LocalExpansion* output,cudaStream_t stream) {
    if(!list.count)return;
    int largest=0;
    for(int q=0;q<8;++q)largest=std::max(largest,list.class_offsets[q+1]-list.class_offsets[q]);
    const int grid=FMM_M2L_MODE==FMM_M2L_MODE_TF32_TC
        ?std::min(2048,(largest+15)/16):std::min(2048,(list.count+3)/4);
    const int block=FMM_M2L_MODE==FMM_M2L_MODE_TF32_TC?256:128;
    fmm_m2l_remote_targets_kernel<<<grid,block,0,stream>>>(list,metadata,nodes,table,output);
    CUDA_CHECK(cudaGetLastError());
}

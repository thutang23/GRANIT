#pragma once

// Included in body_fitted_fmm before the local P2P precision macros are cleared.
template<bool OUTER>
__global__ void fmm_p2p_remote_leaves_kernel(
    FMMRemoteP2PList list,const PersistentTreeMetadata* metadata,
    const UnifiedNode* nodes,const int* target_order,
    const double* pos_x,const double* pos_y,const double* pos_z,const float* mass,
    double* out_x,double* out_y,double* out_z,p2p_real_t G,p2p_real_t softening) {
    __shared__ double sx[4][32],sy[4][32],sz[4][32];
    __shared__ p2p_real_t sm[4][32];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const double cutoff=metadata->cutoff_radius,cutoff_sq=cutoff>0?cutoff*cutoff:0;
    for(int row=int(blockIdx.x)*4+warp;row<list.count;row+=int(gridDim.x)*4) {
        const int leaf=list.leaves[row];const auto node=nodes[leaf];
        for(int tile=0;tile<int(node.count);tile+=32) {
            const int at=tile+lane;
            int original=at<int(node.count)?target_order[node.begin+at]:-1;
            const bool active=original>=0 && original<metadata->target_particle_count;
            const unsigned mask=__ballot_sync(0xffffffffu,active);if(!mask)continue;
            double x=0,y=0,z=0,fx=0,fy=0,fz=0;
            p2p_real_t mi=0;
            if(active) {
                x=pos_x[original];y=pos_y[original];z=pos_z[original];mi=p2p_real_t(mass[original]);
                fx=out_x[original];fy=out_y[original];fz=out_z[original];
            }
            int peer=-1;p2p_real_t ax=0,ay=0,az=0;
            for(uint64_t i=list.offsets[leaf];i<list.offsets[leaf+1];++i) {
                const auto ref=list.sources[i];
                if(ref.peer!=peer) {
                    // Keep the validated FP32-per-peer summation boundaries,
                    // with the final FP64 output retained in registers.
                    fx+=double(ax);fy+=double(ay);fz+=double(az);ax=ay=az=0;peer=ref.peer;
                }
                const auto& source=list.peers[peer];
                grouped_p2p_source_warp<OUTER>(ref.node,active,mask,-1,x,y,z,mi,ax,ay,az,
                    source.nodes,source.x,source.y,source.z,source.mass,source.order,lane,
                    sx[warp],sy[warp],sz[warp],sm[warp],G,copysign(softening*softening, softening),cutoff_sq);
            }
            if(active) {
                out_x[original]=fx+double(ax);out_y[original]=fy+double(ay);out_z[original]=fz+double(az);
            }
        }
    }
}

template<bool OUTER>
inline void launch_p2p_remote_leaves(FMMRemoteP2PList list,const fmm::FMMDeviceData& dd,
    const PersistentTreeMetadata* metadata,const double* x,const double* y,const double* z,
    const float* mass,double* out_x,double* out_y,double* out_z,
    double G,double softening,cudaStream_t stream) {
    if(!list.count)return;
    const int grid=std::min(2048,(list.count+3)/4);
    fmm_p2p_remote_leaves_kernel<OUTER><<<grid,128,0,stream>>>(list,metadata,dd.unified_nodes,
        dd.sorted_to_original,x,y,z,mass,out_x,out_y,out_z,p2p_real_t(G),p2p_real_t(softening));
    CUDA_CHECK(cudaGetLastError());
}

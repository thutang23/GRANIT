#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "domain_manager.h"

namespace cudadem {
struct MortonBounds {
    double lo[3], hi[3];
    int invalid;
};
__global__ void morton_bounds_kernel(const double* x,const double* y,const double* z,int count,MortonBounds* output) {
    __shared__ double lo[3][256],hi[3][256];
    __shared__ int bad[256];
    const int lane=threadIdx.x,i=blockIdx.x*blockDim.x+lane;
    const double p[3]={i<count?x[i]:0,i<count?y[i]:0,i<count?z[i]:0};
    bad[lane]=0;
    for(int a=0;a<3;++a){lo[a][lane]=i<count?p[a]:CUDART_INF;hi[a][lane]=i<count?p[a]:-CUDART_INF;
        if(i<count&&!isfinite(p[a]))bad[lane]=1;}
    __syncthreads();
    for(int stride=128;stride;stride>>=1){if(lane<stride){
        for(int a=0;a<3;++a){lo[a][lane]=fmin(lo[a][lane],lo[a][lane+stride]);hi[a][lane]=fmax(hi[a][lane],hi[a][lane+stride]);}
        bad[lane]|=bad[lane+stride];}__syncthreads();}
    if(lane==0){MortonBounds value{};for(int a=0;a<3;++a){value.lo[a]=lo[a][0];value.hi[a]=hi[a][0];}value.invalid=bad[0];output[blockIdx.x]=value;}
}

// Only six bounds per rank are shared. Enlargement preserves the old root as
// one child of the new root, so existing Morton intervals remain ordered.
class MortonRootService {
    struct Rank {
        MortonBounds* device=nullptr;
        size_t capacity=0;
        std::vector<MortonBounds> blocks;
        MortonBounds bounds{};
    };
    std::vector<std::unique_ptr<Rank>> ranks_;
    bool changed_=false;
public:
    void initialize(int count){ranks_.resize(count);for(auto& rank:ranks_)rank=std::make_unique<Rank>();}
    void shutdown(){int saved=0;cudaGetDevice(&saved);for(size_t i=0;i<ranks_.size();++i){cudaSetDevice(i);cudaFree(ranks_[i]->device);ranks_[i]->device=nullptr;}cudaSetDevice(saved);}
    void collect(int rank,const ParticleSoA& particles,cudaStream_t stream){
        auto& r=*ranks_[rank];const int count=particles.num_local;
        const double inf=std::numeric_limits<double>::infinity();
        r.bounds={{inf,inf,inf},{-inf,-inf,-inf},0};
        if(!count)return;
        const size_t blocks=(count+255)/256;
        if(blocks>r.capacity){if(r.device)CUDA_CHECK(cudaFree(r.device));CUDA_CHECK(cudaMalloc(&r.device,blocks*sizeof(MortonBounds)));r.capacity=blocks;}
        r.blocks.resize(blocks);
        morton_bounds_kernel<<<blocks,256,0,stream>>>(particles.pos_x,particles.pos_y,particles.pos_z,count,r.device);
        CUDA_CHECK(cudaMemcpyAsync(r.blocks.data(),r.device,blocks*sizeof(MortonBounds),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(const auto& block:r.blocks){for(int a=0;a<3;++a){r.bounds.lo[a]=std::min(r.bounds.lo[a],block.lo[a]);r.bounds.hi[a]=std::max(r.bounds.hi[a],block.hi[a]);}r.bounds.invalid|=block.invalid;}
    }
    bool commit(DomainParamsGPU& params, std::vector<uint64_t>* saved_cuts=nullptr,
                int saved_depth=0){
        changed_=false;
        const double inf=std::numeric_limits<double>::infinity();
        MortonBounds bounds{{inf,inf,inf},{-inf,-inf,-inf},0};
        for(const auto& rank:ranks_){for(int a=0;a<3;++a){bounds.lo[a]=std::min(bounds.lo[a],rank->bounds.lo[a]);bounds.hi[a]=std::max(bounds.hi[a],rank->bounds.hi[a]);}bounds.invalid|=rank->bounds.invalid;}
        if(bounds.invalid)throw std::runtime_error("Morton ownership encountered nonfinite particle coordinates");
        if(!std::isfinite(bounds.lo[0]))return false;
        auto grid=params.morton_grid;
        if(grid.dimension==2) {
            const double plane=0.5*(bounds.lo[2]+bounds.hi[2]);
            const double origin=plane-0.5*grid.size;
            if(origin!=grid.z){grid.z=origin;changed_=true;}
        }
        std::vector<uint64_t> cuts(params.morton_cuts,params.morton_cuts+params.num_gpus+1);
        while(!grid.contains(bounds.lo[0],bounds.lo[1],bounds.lo[2]) || !grid.contains(bounds.hi[0],bounds.hi[1],bounds.hi[2])){
            double* origin[3]={&grid.x,&grid.y,&grid.z};uint64_t child=0;
            for(int a=0;a<grid.dimension;++a)if(bounds.lo[a]<*origin[a]){*origin[a]-=grid.size;child|=uint64_t{1}<<a;}
            const uint64_t prefix=child<<(grid.dimension*(MortonGrid::key_depth-1));
            for(int rank=1;rank<params.num_gpus;++rank)cuts[rank]=grid.leaf_begin(prefix+(cuts[rank]>>grid.dimension));
            if (saved_cuts && !saved_cuts->empty()) {
                auto saved_grid = grid; saved_grid.depth = saved_depth;
                for (int rank=1; rank<params.num_gpus; ++rank)
                    (*saved_cuts)[rank] = saved_grid.leaf_begin(
                        prefix + ((*saved_cuts)[rank] >> grid.dimension));
            }
            if(grid.dimension==2)grid.z-=0.5*grid.size; // keep the common expansion plane fixed
            grid.size*=2;
            if(!std::isfinite(grid.size))throw std::overflow_error("Morton root enlargement overflow");
            changed_=true;
        }
        if(changed_){
            ++grid.generation;
            params.domain_min_x=grid.x;params.domain_min_y=grid.y;params.domain_min_z=grid.z;
            params.domain_max_x=grid.x+grid.size;params.domain_max_y=grid.y+grid.size;params.domain_max_z=grid.z+grid.size;
            DomainManager::set_morton_partition(params,grid,cuts);
        }
        return changed_;
    }
    bool changed()const{return changed_;}
};
} // namespace cudadem

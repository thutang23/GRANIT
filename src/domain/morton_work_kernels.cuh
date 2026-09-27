#pragma once
#include "morton_work_balance.h"
#include "morton_gpu_work_balance.cuh"
#include "communicator.h"
#include <memory>

namespace cudadem {
struct MortonParticleCostRecord { uint64_t key, migration_bytes; double contact_units, halo_units; };
__global__ void morton_contact_units_kernel(const ContactPair* pairs,int count,int owned,double* units){
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
    const auto pair=pairs[k];const bool a=pair.i>=0&&pair.i<owned,b=pair.j>=0&&pair.j<owned;
    if(!a&&!b)return;const double cost=2.0/(int(a)+int(b));
    if(a)atomicAdd(units+pair.i,cost);if(b)atomicAdd(units+pair.j,cost);
}
__global__ void morton_sph_units_kernel(const uint64_t* offsets,int owned,double* units){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<owned)units[i]=1.0+double(offsets ? offsets[i+1]-offsets[i] : 0);
}
__global__ void morton_particle_cost_kernel(ParticleSoA p,MortonGrid grid,int ranks,
    const int32_t* halo,const ContactHistory* contacts,const WallContactHistory* walls,
    const double* units,MortonParticleCostRecord* records){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.num_local)return;
    uint64_t bytes=sizeof(PackedParticle);int destinations=0;
    for(int s=0;s<p.num_contacts[i];++s)if(contacts[p.contact_history_offset[i]+s].is_active)bytes+=sizeof(ContactHistoryPacked);
    for(int s=0;s<p.num_wall_contacts[i];++s)if(walls[p.wall_history_offset[i]+s].is_active)bytes+=sizeof(WallContactHistoryPacked);
    const int words=(ranks+31)/32;
    if(halo)for(int w=0;w<words;++w)destinations+=__popc(static_cast<unsigned int>(halo[static_cast<size_t>(i)*words+w]));
    records[i]={grid.leaf_begin(grid.key(p.pos_x[i],p.pos_y[i],p.pos_z[i])),bytes,units[i],double(destinations)};
}
class MortonWorkCollector {
    double* units_=nullptr;MortonParticleCostRecord* records_=nullptr;size_t capacity_=0;
public:
    void shutdown(){cudaFree(units_);cudaFree(records_);units_=nullptr;records_=nullptr;capacity_=0;}
    const MortonParticleCostRecord* collect_device(const GPUDomainState& state,const DomainParamsGPU& params,
        const ContactPair* pairs,int contacts,const uint64_t* sph_offsets,
        bool sph_mode,const int32_t* halo,cudaStream_t stream){
        const int n=state.local_state.particles.num_local;if(!n)return nullptr;
        if(size_t(n)>capacity_){shutdown();capacity_=n;CUDA_CHECK(cudaMalloc(&units_,n*sizeof(double)));CUDA_CHECK(cudaMalloc(&records_,n*sizeof(MortonParticleCostRecord)));}
        CUDA_CHECK(cudaMemsetAsync(units_,0,n*sizeof(double),stream));
        if(sph_mode)morton_sph_units_kernel<<<(n+255)/256,256,0,stream>>>(sph_offsets,n,units_);
        else if(contacts)morton_contact_units_kernel<<<(contacts+255)/256,256,0,stream>>>(pairs,contacts,n,units_);
        morton_particle_cost_kernel<<<(n+255)/256,256,0,stream>>>(state.local_state.particles,params.morton_grid,params.num_gpus,halo,
            state.local_state.contact_history.data,state.local_state.wall_contact_history.data,units_,records_);
        return records_;
    }
};
} // namespace cudadem

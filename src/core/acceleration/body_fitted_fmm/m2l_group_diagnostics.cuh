#pragma once

// Included inside body_fitted_fmm. The default build has no census state or
// instructions. Instrumented executables are for counting, never timing.
#ifdef GRANIT_FMM_M2L_DIAGNOSTICS
static constexpr int M2L_CENSUS_SLOTS = GROUPED_D3_CROSS_COUNT +
    GROUPED_D3_BLOCK_COUNT * GROUPED_D3_CHILD_COUNT;
struct M2LGroupCensus {
    unsigned long long scans[2][8];
    unsigned long long batches[2][8][17];
    unsigned long long routes[2][8][M2L_CENSUS_SLOTS][17];
};
__device__ M2LGroupCensus d_m2l_group_census;
__device__ int d_m2l_group_census_enabled;
static ::cudadem::DeviceStateArray<uint64_t> m2l_census_calls;
static ::cudadem::DeviceStateArray<bool> m2l_census_active;

__device__ __forceinline__ void m2l_census_scan(int path,int q,int count) {
    if(d_m2l_group_census_enabled && (threadIdx.x&31)==0)
        atomicAdd(&d_m2l_group_census.scans[path][q],static_cast<unsigned long long>(count));
}
__device__ __forceinline__ void m2l_census_batch(int path,int q,int count) {
    if(d_m2l_group_census_enabled && (threadIdx.x&31)==0)
        atomicAdd(&d_m2l_group_census.batches[path][q][count],1ull);
}
__device__ __forceinline__ void m2l_census_route(
    int path,int q,int slot,const int* sources,int count) {
    if(!d_m2l_group_census_enabled)return;
    const int lane=threadIdx.x&31;
    const unsigned active=__ballot_sync(0xffffffffu,lane<count && sources[lane]>=0);
    if(lane==0)atomicAdd(&d_m2l_group_census.routes[path][q][slot][__popc(active)],1ull);
}
inline void m2l_census_begin(int gpu,cudaStream_t stream) {
    const char* directory=std::getenv("BF_FMM_M2L_CENSUS_DIR");
    const char* requested=std::getenv("BF_FMM_M2L_CENSUS_EPOCH");
    const uint64_t epoch=requested?std::stoull(requested):1;
    const int enabled=directory && m2l_census_calls[gpu]==epoch;
    m2l_census_active[gpu]=enabled;
    if(enabled) {
        void* address=nullptr;
        CUDA_CHECK(cudaGetSymbolAddress(&address,d_m2l_group_census));
        CUDA_CHECK(cudaMemsetAsync(address,0,sizeof(M2LGroupCensus),stream));
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_m2l_group_census_enabled,&enabled,sizeof(enabled),0,
        cudaMemcpyHostToDevice,stream));
}
template<int D> inline void m2l_census_write(
    std::ostream& out,const M2LGroupCensus& data,uint64_t epoch) {
    using T=GroupedStencilTraits<D>;
    for(int path=0;path<2;++path)for(int q=0;q<T::CHILD_COUNT;++q) {
        if(data.scans[path][q])out<<epoch<<','<<D<<','<<path<<",scan,"<<q<<",-1,-1,1,0,"
            <<data.scans[path][q]<<'\n';
        for(int fill=1;fill<=16;++fill)if(data.batches[path][q][fill])
            out<<epoch<<','<<D<<','<<path<<",batch,"<<q<<",-1,-1,1,"<<fill<<','
               <<data.batches[path][q][fill]<<'\n';
        for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
            bool far=true;uint16_t op;
            if(slot<T::CROSS_COUNT)op=T::host_cross(q,slot).operator_id;
            else {
                const int at=slot-T::CROSS_COUNT,child=at%T::CHILD_COUNT;
                const auto entry=T::host_block(q,at/T::CHILD_COUNT);
                far=(entry.far_mask&(1u<<child))!=0;op=entry.operator_id[child];
            }
            for(int fill=0;fill<=16;++fill)if(data.routes[path][q][slot][fill])
                out<<epoch<<','<<D<<','<<path<<",route,"<<q<<','<<slot<<','<<op<<','<<far<<','
                   <<fill<<','<<data.routes[path][q][slot][fill]<<'\n';
        }
    }
}
inline void m2l_census_end(int gpu,const PersistentTreeMetadata* metadata,cudaStream_t stream) {
    const uint64_t epoch=m2l_census_calls[gpu]++;
    if(!m2l_census_active[gpu])return;
    auto counts=std::make_unique<M2LGroupCensus>();PersistentTreeMetadata host{};
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(counts.get(),d_m2l_group_census,sizeof(*counts),0,
        cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaMemcpyAsync(&host,metadata,sizeof(host),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::string file=std::string(std::getenv("BF_FMM_M2L_CENSUS_DIR"))+"/m2l_g"+
        std::to_string(gpu)+"_e"+std::to_string(epoch)+".csv";
    std::ofstream out(file);
    if(!out)throw std::runtime_error("cannot create M2L census: "+file);
    out<<"epoch,dimension,path,kind,child_class,slot,operator,far_slot,fill,count\n";
    if(host.active_tree==2)m2l_census_write<2>(out,*counts,epoch);
    else m2l_census_write<3>(out,*counts,epoch);
    if(!out)throw std::runtime_error("cannot write M2L census: "+file);
}
#define BF_M2L_CENSUS_SCAN(...) m2l_census_scan(__VA_ARGS__)
#define BF_M2L_CENSUS_BATCH(...) m2l_census_batch(__VA_ARGS__)
#define BF_M2L_CENSUS_ROUTE(...) m2l_census_route(__VA_ARGS__)
#else
#define BF_M2L_CENSUS_SCAN(...) ((void)0)
#define BF_M2L_CENSUS_BATCH(...) ((void)0)
#define BF_M2L_CENSUS_ROUTE(...) ((void)0)
#endif

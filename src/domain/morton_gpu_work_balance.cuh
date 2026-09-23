#pragma once
#include <cub/cub.cuh>
#include <memory>
#include "morton_work_balance.h"
#include "cuda_utils.h"

namespace cudadem {
template<class T> struct MortonGpuBuffer {
    T* data=nullptr;size_t capacity=0;int device=-1;
    ~MortonGpuBuffer(){if(data){int saved=0;cudaGetDevice(&saved);cudaSetDevice(device);cudaFree(data);cudaSetDevice(saved);}}
    void reserve(size_t n){if(n<=capacity)return;int current=0;CUDA_CHECK(cudaGetDevice(&current));
        if(device>=0&&device!=current)throw std::logic_error("Morton buffer device mismatch");
        CUDA_CHECK(cudaFree(data));data=nullptr;device=current;capacity=n+n/4;
        CUDA_CHECK(cudaMalloc(&data,capacity*sizeof(T)));}
};
struct MortonGpuValue {double seconds;unsigned long long bytes;};
struct MortonGpuValueAdd {__host__ __device__ MortonGpuValue operator()(MortonGpuValue a,MortonGpuValue b)const{
    return {a.seconds+b.seconds,a.bytes+b.bytes};}};
template<class Record,class Work> __global__ void morton_gpu_weights_kernel(
    const Record* records,const Work* work,int n,double4 coefficient,double base,double contact,
    double halo,double frequency,uint64_t* keys,MortonGpuValue* values,MortonGrid grid,
    uint64_t lo,uint64_t hi,int* invalid){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto r=records[i];const auto w=work[i];
    const double gravity=w.tree*coefficient.x+w.far*coefficient.y+w.near*coefficient.z+w.communication*coefficient.w;
    const double seconds=base+contact*r.contact_units+halo*r.halo_units+frequency*gravity;
    if(r.key<lo||r.key>=hi||grid.leaf_begin(r.key)!=r.key||!isfinite(seconds)||seconds<0)atomicExch(invalid,1);
    keys[i]=r.key;values[i]={seconds,r.migration_bytes};
}
__global__ void morton_gpu_seconds_kernel(const MortonGpuValue* values,int n,double* seconds){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<=n)seconds[i]=i<n?values[i].seconds:0;
}
__global__ void morton_gpu_cut_kernel(const uint64_t* keys,const MortonGpuValue* values,const double* prefix,int n,
    int ranks,double total,double rank_begin,double rank_end,uint64_t old_end,uint64_t* cuts){
    const int cut=blockIdx.x*blockDim.x+threadIdx.x;if(cut>=ranks||cut==0)return;
    cuts[cut]=~uint64_t{0};const double target=total*double(cut)/ranks;
    if(!n||rank_end<=rank_begin||target<rank_begin||target>=rank_end)return;
    const double local=target-rank_begin;int lo=0,hi=n+1;
    while(lo<hi){const int mid=lo+(hi-lo)/2;if(prefix[mid]<=local)lo=mid+1;else hi=mid;}
    int i=min(n-1,lo-1);
    if(local>=prefix[i]+values[i].seconds*0.5)++i;
    cuts[cut]=i<n?keys[i]:old_end;
}
__global__ void morton_gpu_predict_kernel(const uint64_t* keys,const MortonGpuValue* values,
    int n,const uint64_t* cuts,int ranks,int rank,double* loads,unsigned long long* bytes){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int owner=morton_owner(keys[i],cuts,ranks);atomicAdd(loads+owner,values[i].seconds);
    if(owner!=rank)atomicAdd(bytes+owner,values[i].bytes);
}

// Only totals, cut keys and small load/byte summaries leave device storage.
// The CPU MortonWorkBalance remains the policy oracle for unit tests.
class MortonGpuWorkBalance {
    struct Rank {
        MortonGpuBuffer<uint64_t> keys,sorted_keys,unique_keys,cuts;
        MortonGpuBuffer<MortonGpuValue> values,sorted_values,unique_values;
        MortonGpuBuffer<double> seconds,prefix,loads;
        MortonGpuBuffer<unsigned long long> bytes;
        MortonGpuBuffer<int> scalar;
        MortonGpuBuffer<unsigned char> scratch;
        int leaves=0;double total=0;cudaStream_t stream=nullptr;
        std::vector<double> proposed_seconds;
        std::vector<unsigned long long> moved_bytes;
    };
    std::vector<std::unique_ptr<Rank>> ranks_;
    std::vector<double> rank_prefix_;
    std::vector<uint64_t> old_cuts_,proposed_cuts_;
    MortonGrid grid_{};
    template<class F> static void cub_call(Rank& r,F function){size_t bytes=0;CUDA_CHECK(function(nullptr,bytes));
        r.scratch.reserve(bytes);CUDA_CHECK(function(r.scratch.data,bytes));}
    template<class T> static T read(const T* data,cudaStream_t stream){T value{};
        CUDA_CHECK(cudaMemcpyAsync(&value,data,sizeof(value),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));return value;}
public:
    double current_max=0,proposed_max=0,migration_seconds=0;
    void initialize(int n){ranks_.resize(n);for(auto& r:ranks_)r=std::make_unique<Rank>();rank_prefix_.resize(n+1);}
    void shutdown(){ranks_.clear();}
    void begin(const MortonGrid& grid,const uint64_t* cuts){grid_=grid;
        old_cuts_.assign(cuts,cuts+ranks_.size()+1);proposed_cuts_=old_cuts_;}
    template<class Record,class Work> void set_local(int rank,const Record* records,const Work* work,
        int n,const double coefficient[4],double base,double contact,double halo,double frequency,cudaStream_t stream){
        auto& r=*ranks_.at(rank);r.stream=stream;r.total=0;r.leaves=0;
        r.proposed_seconds.assign(ranks_.size(),0);r.moved_bytes.assign(ranks_.size(),0);
        r.scalar.reserve(2);if(!n)return;
        r.keys.reserve(n);r.sorted_keys.reserve(n);r.unique_keys.reserve(n);
        r.values.reserve(n);r.sorted_values.reserve(n);r.unique_values.reserve(n);
        CUDA_CHECK(cudaMemsetAsync(r.scalar.data,0,2*sizeof(int),stream));
        morton_gpu_weights_kernel<<<(n+255)/256,256,0,stream>>>(records,work,n,
            make_double4(coefficient[0],coefficient[1],coefficient[2],coefficient[3]),base,contact,halo,frequency,
            r.keys.data,r.values.data,grid_,old_cuts_[rank],old_cuts_[rank+1],r.scalar.data+1);
        cub_call(r,[&](void* tmp,size_t& bytes){return cub::DeviceRadixSort::SortPairs(tmp,bytes,
            r.keys.data,r.sorted_keys.data,r.values.data,r.sorted_values.data,n,0,64,stream);});
        cub_call(r,[&](void* tmp,size_t& bytes){return cub::DeviceReduce::ReduceByKey(tmp,bytes,
            r.sorted_keys.data,r.unique_keys.data,r.sorted_values.data,r.unique_values.data,
            r.scalar.data,MortonGpuValueAdd{},n,stream);});
        r.leaves=read(r.scalar.data,stream);
        if(read(r.scalar.data+1,stream))throw std::runtime_error("invalid GPU Morton work record");
        r.seconds.reserve(r.leaves+1);r.prefix.reserve(r.leaves+1);
        morton_gpu_seconds_kernel<<<(r.leaves+256)/256,256,0,stream>>>(r.unique_values.data,r.leaves,r.seconds.data);
        cub_call(r,[&](void* tmp,size_t& bytes){return cub::DeviceScan::ExclusiveSum(tmp,bytes,
            r.seconds.data,r.prefix.data,r.leaves+1,stream);});
        r.total=read(r.prefix.data+r.leaves,stream);
        if(!std::isfinite(r.total))throw std::overflow_error("GPU Morton work total overflow");
    }
    void sum_rank_totals(){rank_prefix_[0]=0;current_max=0;
        for(size_t i=0;i<ranks_.size();++i){rank_prefix_[i+1]=rank_prefix_[i]+ranks_[i]->total;current_max=std::max(current_max,ranks_[i]->total);}
        if(!std::isfinite(rank_prefix_.back()))throw std::overflow_error("GPU Morton collective total overflow");}
    void choose_local_cuts(int rank){auto& r=*ranks_.at(rank);if(r.total<=0)return;
        const int ranks=ranks_.size();r.cuts.reserve(ranks+1);
        morton_gpu_cut_kernel<<<(ranks+255)/256,256,0,r.stream>>>(r.unique_keys.data,r.unique_values.data,r.prefix.data,r.leaves,
            ranks,rank_prefix_.back(),rank_prefix_[rank],rank_prefix_[rank+1],old_cuts_[rank+1],r.cuts.data);
        std::vector<uint64_t> cuts(ranks);CUDA_CHECK(cudaMemcpyAsync(cuts.data(),r.cuts.data,ranks*sizeof(uint64_t),cudaMemcpyDeviceToHost,r.stream));
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
        for(int cut=1;cut<ranks;++cut)if(cuts[cut]!=~uint64_t{0})proposed_cuts_[cut]=cuts[cut];
    }
    void predict_local(int rank){auto& r=*ranks_.at(rank);if(!r.leaves)return;
        const int ranks=ranks_.size();r.cuts.reserve(ranks+1);r.loads.reserve(ranks);r.bytes.reserve(ranks);
        CUDA_CHECK(cudaMemcpyAsync(r.cuts.data,proposed_cuts_.data(),(ranks+1)*sizeof(uint64_t),cudaMemcpyHostToDevice,r.stream));
        CUDA_CHECK(cudaMemsetAsync(r.loads.data,0,ranks*sizeof(double),r.stream));
        CUDA_CHECK(cudaMemsetAsync(r.bytes.data,0,ranks*sizeof(unsigned long long),r.stream));
        morton_gpu_predict_kernel<<<(r.leaves+255)/256,256,0,r.stream>>>(r.unique_keys.data,r.unique_values.data,
            r.leaves,r.cuts.data,ranks,rank,r.loads.data,r.bytes.data);
        CUDA_CHECK(cudaMemcpyAsync(r.proposed_seconds.data(),r.loads.data,ranks*sizeof(double),cudaMemcpyDeviceToHost,r.stream));
        CUDA_CHECK(cudaMemcpyAsync(r.moved_bytes.data(),r.bytes.data,ranks*sizeof(unsigned long long),cudaMemcpyDeviceToHost,r.stream));
        CUDA_CHECK(cudaStreamSynchronize(r.stream));
    }
    bool accept(double minimum_fraction,int amortization_steps,double seconds_per_byte){
        if(!std::is_sorted(proposed_cuts_.begin(),proposed_cuts_.end()))throw std::runtime_error("nonmonotonic GPU Morton cuts");
        if(current_max<=0||proposed_cuts_==old_cuts_)return false;
        std::vector<double> loads(ranks_.size(),0),transfer(ranks_.size(),0);
        for(size_t from=0;from<ranks_.size();++from)for(size_t to=0;to<ranks_.size();++to){
            loads[to]+=ranks_[from]->proposed_seconds[to];const double cost=ranks_[from]->moved_bytes[to]*seconds_per_byte;
            transfer[from]+=cost;transfer[to]+=cost;}
        proposed_max=*std::max_element(loads.begin(),loads.end());migration_seconds=*std::max_element(transfer.begin(),transfer.end());
        return proposed_max<current_max*(1-minimum_fraction)&&
            (current_max-proposed_max)*std::max(1,amortization_steps)>migration_seconds;
    }
    const std::vector<uint64_t>& cuts()const{return proposed_cuts_;}
};
} // namespace cudadem

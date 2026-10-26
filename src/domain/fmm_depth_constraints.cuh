#pragma once

#include "fmm_depth_constraints.h"
#include "peer_transfer.h"
#include "../core/constant/system_initialize.h"
#include "../core/acceleration/shear_gravity_images.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <array>
#include <climits>
#include <memory>
#include <string>
#include <vector>

namespace cudadem {

struct FMMDepthSourceGeometry {
    MortonGrid root{}; // actual shared source-tree root, not XY ownership geometry
    ShearGravityImages images{};
};

static __global__ void fmm_constraint_keys_kernel(
    ParticleSoA particles, int physical_count, int source_count,
    FMMDepthSourceGeometry geometry, uint64_t* keys, int* error) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < source_count;
         i += int64_t(gridDim.x)*blockDim.x) {
        const int row = int(i % physical_count);
        int nx = 0, ny = 0;
        if (geometry.images.enabled) {
            const int slot = int(i/physical_count);
            if (slot) {
                const int width = 2*geometry.images.layers_y+1;
                const int center = geometry.images.layers_x*width+geometry.images.layers_y;
                int ordinal=slot-1;if (ordinal>=center) ++ordinal;
                nx=ordinal/width-geometry.images.layers_x;
                ny=ordinal%width-geometry.images.layers_y;
            }
        }
        const double x = particles.pos_x[row] + double(nx)*geometry.images.length_x;
        const double y = geometry.images.enabled
            ? shear_gravity_canonical_y(particles.pos_y[row],nx,geometry.images)+double(ny)*geometry.images.length_y : particles.pos_y[row];
        uint64_t key = 0;
        if ((particles.lifecycle_state && particles.lifecycle_state[row] != 0) ||
            !fmm_constraint_source_key(geometry.root,x,y,particles.pos_z[row],key))
            atomicExch(error,1);
        keys[i] = key;
    }
}

static __global__ void fmm_constraint_partition_offsets(
    const uint64_t* keys, int count, int ranks, int* offsets) {
    const int rank = threadIdx.x;
    if (rank > ranks) return;
    const uint64_t end = uint64_t{1}<<60;
    const uint64_t cut = (end/uint64_t(ranks))*rank + ((end%uint64_t(ranks))*rank)/ranks;
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = lo+(hi-lo)/2;
        if (keys[mid] < cut) lo = mid+1; else hi = mid;
    }
    offsets[rank] = lo;
}

static __global__ void fmm_constraint_leaf_summary(
    const uint64_t* keys, int count, int depth, FMMLeafOccupancySummary* summary) {
    const int shift = 3*(20-depth);
    for (int64_t at = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; at < count;
         at += int64_t(gridDim.x)*blockDim.x) {
        const int i=int(at);
        const uint64_t leaf = keys[i]>>shift;
        if (i && (keys[i-1]>>shift) == leaf) continue;
        int lo = i+1, hi = count;
        while (lo < hi) {
            const int mid = lo+(hi-lo)/2;
            if ((keys[mid]>>shift) == leaf) lo = mid+1; else hi = mid;
        }
        const uint64_t population = uint64_t(lo-i);
        atomicMax(reinterpret_cast<unsigned long long*>(&summary->maximum_count),
                  static_cast<unsigned long long>(population));
        if (!i) {
            summary->first_key=leaf;summary->first_count=population;summary->source_count=count;
        }
        if (lo == count) { summary->last_key=leaf;summary->last_count=population; }
    }
}

// Exact collective census. Common 3D owners normally already form ordered
// source-key intervals: only keys' extrema and O(depths) summaries leave GPUs.
// Shear XY ownership and image slabs may interleave in 3D Morton order. In that
// case only 8-byte keys are repartitioned between GPUs into ordered census
// intervals, then re-sorted; no N-particle/key list is gathered on the host.
class FMMDepthConstraintCensus {
    template<class T> struct Buffer {
        T* data=nullptr;size_t capacity=0;
        void reserve(size_t n) {
            if (n<=capacity) return;
            if (n>std::numeric_limits<size_t>::max()/sizeof(T))
                throw std::overflow_error("FMM occupancy census allocation overflow");
            size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));
            if (n*sizeof(T)>free+capacity*sizeof(T))
                throw std::runtime_error("FMM occupancy census exceeds available device memory");
            if (data) CUDA_CHECK(cudaFree(data));
            data=nullptr;capacity=0;
            CUDA_CHECK(cudaMalloc(&data,n*sizeof(T)));capacity=n;
        }
        void release() noexcept { if (data) cudaFree(data);data=nullptr;capacity=0; }
    };
    struct Rank {
        Buffer<uint64_t> input,sorted,received,received_sorted;
        Buffer<unsigned char> scratch;
        Buffer<int> offsets_device,error_device;
        Buffer<FMMLeafOccupancySummary> summaries_device;
        std::vector<int> offsets;
        std::array<FMMLeafOccupancySummary,21> summaries{};
        FMMDepthSourceGeometry geometry{};
        int physical=0,sources=0,minimum=0,maximum=0;
        uint64_t first=0,last=0;
        cudaStream_t stream=nullptr;
    };
    std::vector<std::unique_ptr<Rank>> ranks_;
    std::vector<std::string> errors_;
    std::string error_;
    FMMLeafCounts maxima_{};
    uint64_t sources_=0,physical_=0,logical_=0,stored_=0;
    bool redistribute_=false;

    template<class Work> void local_phase(int rank,Work work) {
        errors_[rank].clear();
        try { work(); }
        catch (const std::exception& error) { errors_[rank]=error.what(); }
        catch (...) { errors_[rank]="unknown occupancy census error"; }
    }
    void agree() {
        #pragma omp barrier
        #pragma omp single
        {
            error_.clear();
            for (size_t rank=0;rank<errors_.size();++rank)
                if (!errors_[rank].empty()) {
                    error_="FMM hard-depth census rank "+std::to_string(rank)+": "+errors_[rank];break;
                }
        }
        #pragma omp barrier
        if (!error_.empty()) throw std::runtime_error(error_);
    }
    static int blocks(int count) { return int(std::min<int64_t>(65535,(int64_t(count)+255)/256)); }
    static bool same_geometry(const FMMDepthSourceGeometry& a,const FMMDepthSourceGeometry& b) {
        return a.root.x==b.root.x&&a.root.y==b.root.y&&a.root.z==b.root.z&&a.root.size==b.root.size&&
            a.root.dimension==b.root.dimension&&a.images.enabled==b.images.enabled&&
            a.images.layers_x==b.images.layers_x&&a.images.layers_y==b.images.layers_y&&
            a.images.length_x==b.images.length_x&&a.images.length_y==b.images.length_y&&
            a.images.min_y==b.images.min_y&&a.images.shear_offset==b.images.shear_offset&&
            a.images.image_count==b.images.image_count;
    }
    static void sort_keys(Rank& rank,uint64_t* input,uint64_t* output,int count,cudaStream_t stream) {
        if (!count) return;
        size_t bytes=0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,bytes,input,output,count,0,60,stream));
        rank.scratch.reserve(bytes);
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(rank.scratch.data,bytes,input,output,count,0,60,stream));
    }
public:
    FMMDepthConstraintCensus()=default;
    FMMDepthConstraintCensus(const FMMDepthConstraintCensus&)=delete;
    FMMDepthConstraintCensus& operator=(const FMMDepthConstraintCensus&)=delete;
    ~FMMDepthConstraintCensus(){shutdown();}
    void initialize(int ranks) {
        if (!ranks_.empty() || ranks<1 || ranks>8) throw std::invalid_argument("invalid FMM census rank count");
        for (int rank=0;rank<ranks;++rank) ranks_.push_back(std::make_unique<Rank>());
        errors_.resize(ranks);maxima_.fill(UINT64_MAX);
    }
    const FMMLeafCounts& maximum_leaf_counts()const{return maxima_;}
    uint64_t source_particles()const{return sources_;}
    uint64_t physical_particles()const{return physical_;}
    uint64_t logical_source_particles()const{return logical_;}
    uint64_t stored_source_particles()const{return stored_;}
    bool used_key_redistribution()const{return redistribute_;}

    // All persistent workers call in identical order before every gravity
    // endpoint, after wrapping/root service and with current owned positions.
    void collect(int rank,const ParticleSoA& particles,const FMMDepthSourceGeometry& geometry,
                 int minimum_actual_depth,int maximum_actual_depth,cudaStream_t stream) {
        auto& local=*ranks_.at(rank);const int workers=ranks_.size();
        local_phase(rank,[&] {
            if (minimum_actual_depth<2 || maximum_actual_depth>20 || minimum_actual_depth>maximum_actual_depth ||
                geometry.root.dimension!=3 || !std::isfinite(geometry.root.size) || geometry.root.size<=0 ||
                particles.num_local<0)
                throw std::invalid_argument("invalid source geometry/depth for FMM hard-depth census");
            if (geometry.images.enabled) {
                if (geometry.images.layers_x<0 || geometry.images.layers_y<0 ||
                    !(geometry.images.length_x>0) || !(geometry.images.length_y>0))
                    throw std::invalid_argument("invalid shear source image geometry for FMM census");
                // Count every logical X/Y image at its actual shifted position.
                // At the minimum Y-alignment level the canonical slab begins
                // halfway through a leaf, so virtual images can share a leaf;
                // counting only stored X columns would undercount that case.
                const int64_t columns=2ll*geometry.images.layers_x+1;
                const int64_t rows=2ll*geometry.images.layers_y+1;
                if (columns>INT_MAX/rows) throw std::overflow_error("full shear image census exceeds signed indexing");
                const int64_t full=columns*rows;
                if (full!=geometry.images.image_count)
                    throw std::invalid_argument("inconsistent full shear image population");
            }
            local.geometry=geometry;local.minimum=minimum_actual_depth;local.maximum=maximum_actual_depth;
            local.physical=particles.num_local;local.stream=stream;
            const int64_t columns=geometry.images.enabled?geometry.images.image_count:1;
            if (columns*local.physical>INT_MAX)
                throw std::overflow_error("FMM census source count exceeds signed GPU indexing");
            local.sources=int(columns*local.physical);
            local.input.reserve(local.sources);local.sorted.reserve(local.sources);
            local.error_device.reserve(1);local.offsets_device.reserve(workers+1);
            CUDA_CHECK(cudaMemsetAsync(local.error_device.data,0,sizeof(int),stream));
            if (local.sources) {
                fmm_constraint_keys_kernel<<<blocks(local.sources),256,0,stream>>>(particles,local.physical,
                    local.sources,geometry,local.input.data,local.error_device.data);
                sort_keys(local,local.input.data,local.sorted.data,local.sources,stream);
                CUDA_CHECK(cudaMemcpyAsync(&local.first,local.sorted.data,sizeof(uint64_t),cudaMemcpyDeviceToHost,stream));
                CUDA_CHECK(cudaMemcpyAsync(&local.last,local.sorted.data+local.sources-1,sizeof(uint64_t),cudaMemcpyDeviceToHost,stream));
            }
            fmm_constraint_partition_offsets<<<1,32,0,stream>>>(local.sorted.data,local.sources,workers,local.offsets_device.data);
            local.offsets.resize(workers+1);int invalid=0;
            CUDA_CHECK(cudaMemcpyAsync(local.offsets.data(),local.offsets_device.data,(workers+1)*sizeof(int),cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaMemcpyAsync(&invalid,local.error_device.data,sizeof(int),cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (invalid) throw std::runtime_error("nonfinite/inactive source or particle outside actual FMM source root");
        });
        agree();
        #pragma omp single
        {
            local_phase(rank,[&] {
                redistribute_=false;sources_=physical_=logical_=stored_=0;
                bool previous=false;uint64_t last=0;
                const auto& first=*ranks_[0];
                for (const auto& entry:ranks_) {
                    const auto& other=*entry;
                    if (!same_geometry(first.geometry,other.geometry) || first.minimum!=other.minimum || first.maximum!=other.maximum)
                        throw std::runtime_error("workers disagree on actual FMM census geometry/depth range");
                    sources_+=other.sources;physical_+=other.physical;
                    if (other.sources) {
                        if (previous && other.first<last) redistribute_=true;
                        previous=true;last=other.last;
                    }
                }
                if (!sources_) throw std::runtime_error("FMM occupancy census has no live source particles");
                const uint64_t images=geometry.images.enabled?uint64_t(geometry.images.image_count):1;
                if (images && physical_>UINT64_MAX/images) throw std::overflow_error("logical FMM source population overflow");
                logical_=physical_*images;
                const uint64_t columns=geometry.images.enabled?uint64_t(2ll*geometry.images.layers_x+1):1;
                if (physical_>UINT64_MAX/columns) throw std::overflow_error("stored FMM source population overflow");
                stored_=physical_*columns;
                if (logical_!=sources_) throw std::runtime_error("logical FMM image census population mismatch");
            });
        }
        agree();
        local_phase(rank,[&] {
            int received_count=local.sources;
            const uint64_t* keys=local.sorted.data;
            if (redistribute_) {
                int64_t total=0;
                for (const auto& entry:ranks_) total+=entry->offsets[rank+1]-entry->offsets[rank];
                if (total>INT_MAX) throw std::overflow_error("FMM census key shard exceeds signed GPU indexing");
                received_count=int(total);
                local.received.reserve(received_count);local.received_sorted.reserve(received_count);
                int offset=0;
                for (int producer=0;producer<workers;++producer) {
                    const auto& source=*ranks_[producer];
                    const int begin=source.offsets[rank],count=source.offsets[rank+1]-begin;
                    if (count) copy_peer_payload(local.received.data+offset,rank,source.sorted.data+begin,producer,
                                                 size_t(count)*sizeof(uint64_t),stream);
                    offset+=count;
                }
                sort_keys(local,local.received.data,local.received_sorted.data,received_count,stream);
                keys=local.received_sorted.data;
            }
            local.summaries_device.reserve(21);
            CUDA_CHECK(cudaMemsetAsync(local.summaries_device.data,0,21*sizeof(FMMLeafOccupancySummary),stream));
            if (received_count) for (int depth=minimum_actual_depth;depth<=maximum_actual_depth;++depth)
                fmm_constraint_leaf_summary<<<blocks(received_count),256,0,stream>>>(keys,received_count,depth,local.summaries_device.data+depth);
            CUDA_CHECK(cudaMemcpyAsync(local.summaries.data(),local.summaries_device.data,
                21*sizeof(FMMLeafOccupancySummary),cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        agree(); // every consumer has finished reading producer keys
        #pragma omp single
        {
            local_phase(rank,[&] {
                maxima_.fill(UINT64_MAX);
                for (int depth=minimum_actual_depth;depth<=maximum_actual_depth;++depth) {
                    std::vector<FMMLeafOccupancySummary> shards;
                    uint64_t count=0;
                    for (const auto& entry:ranks_) {shards.push_back(entry->summaries[depth]);count+=entry->summaries[depth].source_count;}
                    if (count!=sources_) throw std::runtime_error("FMM source census lost or duplicated particles");
                    maxima_[depth]=fmm_global_maximum_leaf_count(shards);
                    if (depth>minimum_actual_depth && maxima_[depth]>maxima_[depth-1])
                        throw std::runtime_error("FMM maximum leaf population increased under refinement");
                }
            });
        }
        agree();
    }
    void shutdown()noexcept {
        int saved=0;cudaGetDevice(&saved);
        for (size_t rank=0;rank<ranks_.size();++rank) {
            cudaSetDevice(int(rank));auto& local=*ranks_[rank];
            if (local.stream) cudaStreamSynchronize(local.stream);
            local.input.release();local.sorted.release();local.received.release();local.received_sorted.release();
            local.scratch.release();local.offsets_device.release();local.error_device.release();local.summaries_device.release();
        }
        ranks_.clear();errors_.clear();cudaSetDevice(saved);
    }
};

} // namespace cudadem

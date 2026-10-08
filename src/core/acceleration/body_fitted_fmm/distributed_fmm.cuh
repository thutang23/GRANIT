#pragma once
#include "fmm_halo_geometry.h"
#include "remote_interaction_lists.cuh"

namespace body_fitted_fmm {

// Cell identities are independent of the rank count and local tree indexing.
struct FMMCellKey {
    uint64_t morton;
    int level;
    bool operator<(const FMMCellKey& b) const {
        return level < b.level || (level == b.level && morton < b.morton);
    }
    bool operator==(const FMMCellKey& b) const { return level==b.level && morton==b.morton; }
};
template<class T> struct FMMDeviceBuffer {
    T* data = nullptr;
    size_t capacity = 0;
    int device = -1;
    FMMDeviceBuffer() = default;
    FMMDeviceBuffer(const FMMDeviceBuffer&) = delete;
    FMMDeviceBuffer& operator=(const FMMDeviceBuffer&) = delete;
    ~FMMDeviceBuffer() { reset(); }
    void reset() {
        if (!data) return;
        int saved = 0;
        cudaGetDevice(&saved);
        cudaSetDevice(device);
        cudaFree(data);
        cudaSetDevice(saved);
        data = nullptr; capacity = 0;
    }
    void reserve(size_t count) {
        if (count <= capacity) return;
        int current = 0;
        CUDA_CHECK(cudaGetDevice(&current));
        if (device >= 0 && device != current)
            throw std::logic_error("FMM buffer used on a different device");
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(T);
        if (count > limit || count / 4 > limit - count)
            throw std::overflow_error("FMM GPU buffer byte size overflow");
        const size_t next_capacity = count + count / 4;
        size_t free_bytes=0, total_bytes=0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t bytes = next_capacity * sizeof(T);
        // reserve discards this scratch buffer; its previous allocation is
        // reclaimable. Use actual request sizes, not a dense-grid estimate.
        if (bytes > free_bytes + capacity * sizeof(T))
            throw std::runtime_error("FMM GPU buffer capacity preflight: need " +
                std::to_string(bytes) + " bytes, free plus reclaimable " +
                std::to_string(free_bytes + capacity * sizeof(T)));
        const size_t additional=bytes>capacity*sizeof(T)?bytes-capacity*sizeof(T):0;
        if(additional>::cudadem::shear_fmm_capacity::memory_allowance(free_bytes))
            throw std::runtime_error("FMM actual scratch-buffer growth exceeds memory reserve");
        reset(); device = current;
        CUDA_CHECK(cudaMalloc(&data, bytes));
        capacity = next_capacity;
    }
};

struct FMMSourceReply {
    uint32_t begin, count, global_count;
    far_real_t coeffs[fmm::FMM_MAX_COEFFS];
};
struct FMMParticleRequest { uint32_t begin, count, offset; };
struct FMMSourceParticle { double x, y, z; float mass; int original; };
struct FMMSourcePart { int node, image_y; };
struct FMMSourcePublication {
    PersistentTreeContext* context = nullptr;
    const int* sorted = nullptr;
    const double *x = nullptr, *y = nullptr, *z = nullptr;
    const float* mass = nullptr;
    int count = 0;
    const int32_t* global_ids = nullptr;
    int targets = 0;
    int image_layers_y = 0, image_alignment_level = 0;
    double image_length_y = 0;
};
struct FMMQueryPeer {
    cudaStream_t stream = nullptr;
    int device = -1;
    cudaEvent_t end=nullptr;
    FMMDeviceBuffer<FMMCellKey> keys;
    FMMDeviceBuffer<FMMSourceReply> replies;
    FMMDeviceBuffer<FMMParticleRequest> particle_requests;
    FMMDeviceBuffer<FMMSourceParticle> particles;
    std::vector<int> debug_source_ids;
    std::vector<std::vector<int>> debug_memberships;
    int request_count = 0;
    FMMDeviceBuffer<int> local_ids;
    FMMDeviceBuffer<int> source_nodes;
    FMMDeviceBuffer<uint64_t> part_counts, part_prefix;
    FMMDeviceBuffer<int> part_offsets;
    FMMDeviceBuffer<FMMSourcePart> parts;
    FMMDeviceBuffer<int> validation_part_offsets;
    FMMDeviceBuffer<FMMSourcePart> validation_parts;
    FMMDeviceBuffer<unsigned char> scratch;
    FMMDeviceBuffer<int> validation_error;
    uint64_t source_generation=~uint64_t{0};
    // Producer-side dense gather schedule, consumer-side independent source view.
    FMMDeviceBuffer<int> selected, select_flags, direct_flags, direct_on_producer;
    FMMDeviceBuffer<uint64_t> particle_counts, particle_offsets;
    FMMDeviceBuffer<int> packed_particle_offsets;
    FMMDeviceBuffer<FMMCellKey> remote_keys;
    FMMDeviceBuffer<FMMSourceReply> remote_values;
    FMMDeviceBuffer<UnifiedNode> halo_nodes;
    FMMDeviceBuffer<int> halo_children, halo_order, route_counts;
    FMMDeviceBuffer<fmm::MultipoleExpansion> halo_multipoles;
    // Only inverse-stencil targets own the large route planes. A dense int map
    // preserves O(1) lookup for ordered far slots and ancestor direct routes.
    FMMDeviceBuffer<int> route_targets, route_index;
    int route_target_count=0;
    FMMDeviceBuffer<GroupedTargetInteractions> halo_routes, moment_routes;
    FMMDeviceBuffer<int> moment_slots, moment_owner;
    FMMDeviceBuffer<double> halo_x, halo_y, halo_z;
    FMMDeviceBuffer<float> halo_mass;
    FMMDeviceBuffer<FMMSourceParticle> halo_particles;
    uint64_t route_generation=~uint64_t{0};
    size_t particle_count=0;
    ~FMMQueryPeer() {
        if (!stream) return;
        int saved = 0; cudaGetDevice(&saved); cudaSetDevice(device);
        cudaStreamSynchronize(stream); cudaEventDestroy(end); cudaStreamDestroy(stream); cudaSetDevice(saved);
    }
};
struct FMMGpuPlan {
    int owned_nodes=0,depth=0,cached_nodes=-1;
    double cutoff=-1;
    uint64_t rebuilds=0,reuses=0,raw_requests=0;
    uint64_t topology_rebuilds=0,topology_reuses=0;
    FMMDeviceBuffer<double4> work_difference,work_prefix;
    FMMDeviceBuffer<DistributedFMMParticleWork> particle_work;
    FMMDeviceBuffer<uint64_t> counts,offsets;
    FMMDeviceBuffer<int> scalar;
    FMMDeviceBuffer<unsigned char> scratch;
};
struct FMMProducerHalo {
    FMMQueryPeer table;
    FMMDeviceBuffer<FMMSourceReply> images_only;
    FMMDeviceBuffer<FMMHaloBox> boxes;
    FMMDeviceBuffer<uint64_t> raw_keys, sorted_keys, unique_keys;
    FMMDeviceBuffer<int> scalar, owner_masks;
    FMMDeviceBuffer<unsigned char> scratch;
    std::vector<FMMHaloBox> geometry;
    FMMHaloGeometryKey geometry_key;
    int cells=0;
    uint64_t generation=0;
};
struct FMMHaloMomentView {
    const FMMCellKey* keys=nullptr;
    const FMMSourceReply* values=nullptr;
    int count=0;
};
struct FMMRemoteLists {
    FMMDeviceBuffer<FMMRemoteView> views;
    std::vector<FMMRemoteView> host_views;
    FMMDeviceBuffer<int> indices,far_flags,near_flags,far_targets,near_leaves,scalar,class_counts,error;
    FMMDeviceBuffer<uint64_t> far_keys,sorted_far_keys,direct_counts,direct_offsets;
    FMMDeviceBuffer<FMMRemoteRef> far_sources,direct_sources;
    FMMDeviceBuffer<unsigned char> scratch;
    int far_count=0,near_count=0;
    int class_offsets[9]{};
    uint64_t topology_stamp=~uint64_t{0},local_stamp=~uint64_t{0};
    std::vector<uint64_t> generations;
    bool valid=false;
    FMMRemoteM2LList far_view() const {
        FMMRemoteM2LList result{far_targets.data,far_sources.data,views.data,far_count,{}};
        std::copy(class_offsets,class_offsets+9,result.class_offsets);return result;
    }
    FMMRemoteP2PList near_view() const {
        return {near_leaves.data,direct_offsets.data,direct_sources.data,views.data,near_count};
    }
};
struct DistributedFMMState {
#ifdef GRANIT_FMM_TEST_HOOKS
    cudaEvent_t test_delay_event=nullptr;
#endif
    bool debug_coverage = false;
    std::vector<int> debug_arena_ids;
    std::vector<std::vector<int>> debug_node_members, debug_target_members;
    size_t debug_index_count = 0;
    bool enabled = false;
    int compute_depth = 0;
    FMMGpuPlan gpu_plan;
    cudaEvent_t phase_events[9]{};
    int event_device=-1;
    DistributedFMMMetrics metrics{};
    FMMDeviceBuffer<double4> node_work;
    FMMDeviceBuffer<int64_t> remote_interactions;
    double communication_ms=0;
    FMMDeviceBuffer<double> projection_scalars, gravity_x, gravity_y, gravity_z;
    double projection_partial[16]{};
    ~DistributedFMMState(){
        if(event_device<0)return;
        int saved=0;cudaGetDevice(&saved);cudaSetDevice(event_device);
#ifdef GRANIT_FMM_TEST_HOOKS
        if(test_delay_event)cudaEventDestroy(test_delay_event);
#endif
        for(auto event:phase_events)if(event)cudaEventDestroy(event);
        cudaSetDevice(saved);
    }
    ::cudadem::MortonGrid grid{};
    ::cudadem::MortonGrid ownership_grid{};
    ::cudadem::ShearSourceRouting routing{};
    FMMDeviceBuffer<::cudadem::MortonCoverCell> ownership_cells;
    std::vector<uint64_t> cuts;
    std::vector<std::unique_ptr<FMMQueryPeer>> peers;
    FMMSourcePublication publication;
    FMMProducerHalo halo;
    FMMRemoteLists remote_lists;
    // Numerical overlay only: local topology, routes and original moments
    // remain owned by the local tree. Shared cells are contracted once after
    // FP64 reduction, before the selected M2L mode rounds its operands.
    FMMDeviceBuffer<fmm::MultipoleExpansion> moment_overlay;
    FMMDeviceBuffer<FMMHaloMomentView> moment_views;
    std::vector<FMMHaloMomentView> host_moment_views;
    std::vector<uint64_t> moment_generations;
    uint64_t moment_local_generation=~uint64_t{0};
    bool has_moment_overlay=false;
    // Producer identity is independent of the consumer's request-plan cache.
    FMMDeviceBuffer<uint64_t> published_identity;
    FMMDeviceBuffer<int> published_changed;
    int published_nodes=-1,published_depth=-1;
    const void* published_arena=nullptr;
    const void* published_context=nullptr;
    ::cudadem::MortonGrid published_grid{};
    uint64_t publication_generation=0;
    size_t let_wire_bytes=0;
    uint64_t let_schedule_reuses=0,let_schedule_lookups=0;
    ::cudadem::MultiGPUPhaseCoordinator* coordinator = nullptr;
    uint64_t epoch = 0;
    int ranks = 1;
    bool evaluating = false;
    bool local_only = false;
    size_t received_cells = 0, received_particle_count = 0;
};
static ::cudadem::DeviceStateArray<DistributedFMMState> distributed_fmm_;

inline void reset_distributed_fmm_state(int gpu) {
    auto& state = distributed_fmm_[gpu];
    // The per-device slot remains address-stable. Run-wide teardown releases
    // every independent source allocation, including an always-empty owner.
    // Internal depth/arena replacements retain the live distributed slot.
    state.~DistributedFMMState();
    new (&state) DistributedFMMState();
}

inline void mark_distributed_fmm_phase(int gpu,int phase,cudaStream_t stream) {
    auto& state=distributed_fmm_[gpu];if(!state.evaluating)return;
    if(!state.phase_events[0]){
        state.event_device=gpu;
        for(auto& event:state.phase_events)CUDA_CHECK(cudaEventCreate(&event));
    }
    CUDA_CHECK(cudaEventRecord(state.phase_events[phase],stream));
}

template<int D> __global__ void fmm_owned_node_work_kernel(
    const PersistentTreeContext* ctx,int owned_nodes,uint64_t own_begin,uint64_t own_end,double4* output,
    ::cudadem::ShearSourceRouting routing,int gpu,
    const UnifiedNode* remote_nodes=nullptr,const int* remote_children=nullptr,
    const GroupedTargetInteractions* remote_routes=nullptr,bool remote_wire=false,
    const int* remote_route_index=nullptr) {
    const int target=blockIdx.x*blockDim.x+threadIdx.x;if(target>=owned_nodes)return;
    const auto& node=ctx->dd.unified_nodes[target];double far=0,near=0,communication=0;
    const auto* nodes=remote_nodes?remote_nodes:ctx->dd.unified_nodes;
    const auto* children=remote_children?remote_children:ctx->dd.children_compact;
    if(node.level>=2 && node.target_count &&
        (!remote_nodes || !remote_route_index || remote_route_index[target]>=0)){
        const int row=remote_route_index?remote_route_index[target]:target;
        using T=GroupedStencilTraits<D>;const auto work=remote_routes?remote_routes[row]:ctx->arena.d_targets[target];
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

inline void set_distributed_fmm_partition(int gpu, const ::cudadem::DomainParamsGPU& params) {
    auto& state = distributed_fmm_[gpu];
    if (!params.morton_enabled) { state.enabled = false; return; }
    std::vector<uint64_t> cuts(params.morton_cuts, params.morton_cuts + params.num_gpus + 1);
    if (!state.enabled || params.morton_grid.generation != state.ownership_grid.generation ||
        params.morton_grid.depth != state.ownership_grid.depth || params.morton_grid.dimension != state.ownership_grid.dimension ||
        params.morton_grid.x != state.ownership_grid.x || params.morton_grid.y != state.ownership_grid.y ||
        params.morton_grid.z != state.ownership_grid.z || params.morton_grid.size != state.ownership_grid.size || cuts != state.cuts) {
        state.gpu_plan.cached_nodes=-1;
        const auto cells = ::cudadem::morton_cover(params.morton_grid, cuts);
        state.ownership_cells.reserve(cells.size());
        if (!cells.empty()) CUDA_CHECK(cudaMemcpy(state.ownership_cells.data, cells.data(),
            cells.size() * sizeof(::cudadem::MortonCoverCell), cudaMemcpyHostToDevice));
        state.routing.cells = state.ownership_cells.data;
        state.routing.cell_count = static_cast<int>(cells.size());
    }
    state.enabled = true; state.grid = state.ownership_grid = params.morton_grid;
    state.cuts = std::move(cuts); state.ranks = params.num_gpus;
    if (state.peers.size() != static_cast<size_t>(params.num_gpus)) {
        state.peers.clear(); state.peers.resize(params.num_gpus);
        for (auto& peer : state.peers) peer = std::make_unique<FMMQueryPeer>();
    }
}

__global__ void fmm_unpack_source_particles_kernel(
    const FMMSourceParticle* particles, int count, int offset,
    double* x, double* y, double* z, float* mass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto p = particles[i];
    x[offset+i] = p.x; y[offset+i] = p.y; z[offset+i] = p.z; mass[offset+i] = p.mass;
}

inline void fmm_collective_check(bool ok, const DistributedFMMState& state) {
    if (!ok) throw std::runtime_error("Distributed FMM exchange: " + state.coordinator->error_message());
}

} // namespace body_fitted_fmm

#include "distributed_fmm_gpu.cuh"

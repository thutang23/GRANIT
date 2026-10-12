#pragma once

#include "../fmm/fmm_types.h"
#include "unified_node.cuh"
#include "grouped_interaction_arena.cuh"

namespace body_fitted_fmm {

// References remain valid when live particle ranges move within receive buffers.
// Peer views are refreshed after every exchange, including buffer reallocations.
struct FMMRemoteRef {
    int peer = -1;
    int node = -1;
};
struct FMMRemoteView {
    const UnifiedNode* nodes = nullptr;
    const int* children = nullptr;
    const fmm::MultipoleExpansion* moments = nullptr;
    const GroupedTargetInteractions* far_routes = nullptr;
    const GroupedTargetInteractions* direct_routes = nullptr;
    const double *x = nullptr, *y = nullptr, *z = nullptr;
    const float* mass = nullptr;
    const int* order = nullptr;
    int count = 0;
    const int* route_index = nullptr; // local target -> compact row; -1 means absent
};
__host__ __device__ __forceinline__ int fmm_remote_route_row(const FMMRemoteView& view,int target) {
    return view.route_index?view.route_index[target]:target;
}
// Previous-epoch far schedules contain no live coefficients or populations.
// Source references are rebound to current receive-buffer indexes before use.
struct FMMFarCachePeerView {
    const int* source_remap = nullptr;
    const uint8_t* child_masks = nullptr;
    int count = 0;
};
struct FMMFarCacheTargetPeer {
    GroupedTargetInteractions routes{};
    bool present = false;
};

struct FMMRemoteM2LList {
    const int* targets = nullptr; // child-class sorted active target indices
    const FMMRemoteRef* sources = nullptr; // [stencil slot][active target]
    const FMMRemoteView* peers = nullptr;
    int count = 0;
    int class_offsets[9]{};
};
struct FMMRemoteP2PList {
    const int* leaves = nullptr;
    const uint64_t* offsets = nullptr; // indexed by local node, with end sentinel
    const FMMRemoteRef* sources = nullptr;
    const FMMRemoteView* peers = nullptr;
    int count = 0;
};

template<bool Remote>
__device__ __forceinline__ const UnifiedNode& grouped_m2l_source_node(
    int source,const UnifiedNode* nodes,const FMMRemoteRef* refs,const FMMRemoteView* peers) {
    if constexpr(Remote) {const auto r=refs[source];return peers[r.peer].nodes[r.node];}
    else return nodes[source];
}
template<bool Remote>
__device__ __forceinline__ far_real_t grouped_m2l_source_coefficient(
    int source,int coefficient,const fmm::MultipoleExpansion* moments,
    const FMMRemoteRef* refs,const FMMRemoteView* peers) {
    if constexpr(Remote) {const auto r=refs[source];return peers[r.peer].moments[r.node].coeffs[coefficient];}
    else return moments[source].coeffs[coefficient];
}

} // namespace body_fitted_fmm

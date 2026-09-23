// =============================================================================
// Local Essential Tree (LET) Data Structures for Multi-GPU Exchange
// =============================================================================
// Structures for exchanging boundary trees and full LETs between GPUs.
// Compatible with Bonsai's float4 layout for tree nodes.
// =============================================================================

#ifndef CUDADEM_LET_TYPES_H
#define CUDADEM_LET_TYPES_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "p2p_topology.h"

namespace cudadem {

// =============================================================================
// LET Node Structure (compatible with Bonsai's real4 layout)
// =============================================================================

struct LETNode {
    float4 center;        // xyz = center, w = opening_criterion (s/d ratio squared)
    float4 size;          // xyz = half-size, w = packed child info
    float4 multipole[3];  // [0]=COM+mass, [1]=quad_diag, [2]=quad_offdiag
};

// =============================================================================
// FMM LET Node Structure (carries p=4 trace-free Cartesian multipoles)
// =============================================================================
// Used by the FMM multi-GPU path.  World positions stay FP64 across the wire:
// a global ring may sit at 1e8 m while physically relevant local separations
// are many orders smaller.  Coefficients remain FP32 by design and are
// promoted by the evaluator; they are local expansion data, not world points.
// =============================================================================

struct alignas(16) FMMLETNode {
    double3 center;              // FP64 world-space expansion center
    float radius;                // conservative half-size
    float multipole[28];        // promoted to double by the evaluator
};
static_assert(sizeof(FMMLETNode) == 144, "FMM LET wire node must be 144 bytes");
static_assert(alignof(FMMLETNode) == 16, "FMM LET wire node must be 16-byte aligned");

struct alignas(16) LETParticle {
    double3 pos;          // FP64 world-space position
    float mass;
};
static_assert(sizeof(LETParticle) == 32, "FMM LET particle wire record must be 32 bytes");

// A discovery record is deliberately tiny and contains no wire payload.  A
// mask describes exactly the consumers for which this source node was
// accepted; leaf records are materialised as sparse particles instead.
struct LETExportDescriptor {
    int32_t source_node;
    uint8_t destination_mask;
    uint8_t is_leaf;
    uint16_t reserved;
};
static_assert(sizeof(LETExportDescriptor) == 8, "LET discovery records stay compact");

// Non-owning device view consumed by the FMM LET exporter.  The communicator
// deliberately does not include body-fitted FMM headers: those headers are
// built under more than one namespace in this project.  The three concrete
// FMM pointees therefore remain opaque here and are cast only in the .cu
// implementation, while topology arrays keep their stable int32_t type.
// `metadata` supplies the current tree's actual node count; `node_capacity`
// is the allocation bound used for host-side LET workspace sizing and every
// device-side traversal/index validation.
struct FMMLETSourceView {
    const void* unified_nodes = nullptr;       // body_fitted_fmm::UnifiedNode[]
    const int32_t* children_compact = nullptr;
    const void* multipoles = nullptr;          // fmm::MultipoleExpansion[]
    const int32_t* sorted_indices = nullptr;
    const int32_t* level_start = nullptr;
    const int32_t* level_count = nullptr;
    const int32_t* level_list = nullptr;
    const void* metadata = nullptr;            // body_fitted_fmm::PersistentTreeMetadata
    int32_t node_capacity = 0;
};

// Host-visible sender plan.  It is published before the coordinator release;
// consumers use it to pull any selected shared bucket followed by their
// residual segment into one contiguous local producer range.
struct LETSenderSegmentPlan {
    int32_t shared_offset[256]{};
    int32_t shared_count[256]{};
    int32_t residual_offset[8]{};
    int32_t residual_count[8]{};
    int32_t particle_offset[8]{};
    int32_t particle_count[8]{};
};

// Exact wire-work units produced for one local FMM source node during the
// latest successful LET export. x counts accepted multipole records; y counts
// leaf particle records.  Both include every destination peer.
using LETSourceWork = uint2;

struct LETTransferSummary {
    int64_t node_records = 0;
    int64_t particle_records = 0;
    size_t node_bytes = 0;
    size_t particle_bytes = 0;
    size_t total_bytes = 0;
    int32_t active_peers = 0;
};

struct LETSharingSummary {
    int32_t candidate_mask_count = 0;
    int32_t shared_mask_count = 0;
    int64_t unique_node_records = 0;
    int64_t delivered_node_records = 0;
    int64_t serialized_node_records = 0;
    int64_t shared_node_records = 0;
    int64_t residual_node_records = 0;
    int64_t particle_records = 0;
    size_t avoided_serialization_bytes = 0;
    int64_t copy_operations = 0;
};

enum class LETTimingPhase : uint8_t { ExportKernel, ReceiveTransfer };
struct LETTimingRecord {
    LETTimingPhase phase = LETTimingPhase::ExportKernel;
    uint64_t epoch = 0;
    int32_t peer_gpu = -1;
    LinkClass link_class = LinkClass::GenericPcieP2P;
    int64_t node_records = 0;
    int64_t particle_records = 0;
    size_t bytes = 0;
    float milliseconds = 0.0f;
    bool grew = false;
    bool retried = false;
    bool valid_for_calibration = true;
};
struct LETTimingSlot {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool pending = false;
    LETTimingRecord record{};
};

// =============================================================================
// LET Buffer Header (16 bytes, MPI-transferable)
// =============================================================================

struct LETBufferHeader {
    int32_t num_particles;
    int32_t num_nodes;
    int32_t start_node;    // First node in top-level list
    int32_t end_node;      // Last node in top-level list
};

// =============================================================================
// LET Push Per-Peer Buffer (one per remote GPU)
// =============================================================================
// Producer GPU writes LET data here; then cudaMemcpyPeerAsync pushes to peer.
// =============================================================================

struct LETPushPerPeer {
    // --- Send side (kernel writes here on producer GPU) ---
    LETNode*     d_send_nodes;           // Pre-allocated: max_nodes * sizeof(LETNode)
    LETParticle*  d_send_particles;      // Pre-allocated: max_particles * sizeof(LETParticle)
    int32_t*     d_send_node_count;      // Device atomic counter
    int32_t*     d_send_particle_count;  // Device atomic counter
    int32_t      max_nodes;
    int32_t      max_particles;

    // --- Receive side (peer pushes data here via cudaMemcpyPeerAsync) ---
    LETNode*     d_recv_nodes;
    LETParticle*  d_recv_particles;
    int32_t*     h_pinned_recv_node_count;     // Host-pinned: set before P2P copy
    int32_t*     h_pinned_recv_particle_count; // Host-pinned
    int32_t      max_recv_nodes;
    int32_t      max_recv_particles;

    LETPushPerPeer()
        : d_send_nodes(nullptr), d_send_particles(nullptr),
          d_send_node_count(nullptr), d_send_particle_count(nullptr),
          max_nodes(0), max_particles(0),
          d_recv_nodes(nullptr), d_recv_particles(nullptr),
          h_pinned_recv_node_count(nullptr), h_pinned_recv_particle_count(nullptr),
          max_recv_nodes(0), max_recv_particles(0) {}
};

// =============================================================================
// LET Push State (replaces LETCommState)
// =============================================================================

struct LETPushState {
    int32_t          gpu_id;
    int32_t          num_gpus;
    LETPushPerPeer*  peers;                       // [num_gpus], indexed by peer GPU ID

    // Combined device arrays for kernel access (contiguous across all peers)
    // Allocated as: [num_gpus * per_peer_size]
    float4*          d_all_let_centers;           // send: let node centers
    float4*          d_all_let_sizes;             // send: let node sizes
    float4*          d_all_let_multipoles;        // send: let node multipoles (3 float4 per node)
    float4*          d_all_let_particles;         // send: let particles
    int32_t*         d_let_node_counts;           // send: [num_gpus] atomic counters
    int32_t*         d_let_particle_counts;       // send: [num_gpus] atomic counters

    // Receive-side combined arrays (peer data pushed here)
    float4*          d_all_recv_centers;
    float4*          d_all_recv_sizes;
    float4*          d_all_recv_multipoles;
    float4*          d_all_recv_particles;
    int32_t*         d_recv_node_counts;          // receive: [num_gpus]
    int32_t*         d_recv_particle_counts;      // receive: [num_gpus]

    // Host-pinned count staging (one entry per peer, shared host memory on single node)
    int32_t*         h_pinned_my_send_node_counts;     // [num_gpus]
    int32_t*         h_pinned_my_send_particle_counts; // [num_gpus]

    cudaStream_t     compute_stream;
    cudaStream_t     comm_stream;

    // Direct FMM LET arenas.  Each is one allocation with peer-owned
    // segments.  Unlike the historical SoA staging arrays these hold the
    // final wire format consumed by the remote gravity walk.
    FMMLETNode*      d_fmm_send_nodes;
    LETParticle*     d_fmm_send_particles;
    FMMLETNode*      d_fmm_recv_nodes;
    LETParticle*     d_fmm_recv_particles;
    int32_t*         d_fmm_send_node_offsets;
    int32_t*         d_fmm_send_particle_offsets;
    int32_t*         d_fmm_recv_node_offsets;
    int32_t*         d_fmm_recv_particle_offsets;
    // One count per producer/consumer.  Counts are exact even when the
    // producer discovers a high-water overflow; publication is deferred.
    int32_t*         d_fmm_recv_node_counts;
    int32_t*         d_fmm_recv_particle_counts;
    int32_t*         h_fmm_send_node_offsets;
    int32_t*         h_fmm_send_particle_offsets;
    int32_t*         h_fmm_recv_node_offsets;
    int32_t*         h_fmm_recv_particle_offsets;
    int32_t*         h_fmm_send_node_capacities;
    int32_t*         h_fmm_send_particle_capacities;
    int32_t*         h_fmm_recv_node_capacities;
    int32_t*         h_fmm_recv_particle_capacities;
    size_t           fmm_arena_bytes;

    // Capacity-indexed source attribution. Live Karras nodes carry work and
    // the invalid suffix is cleared to zero; this stays independent of the
    // per-peer wire arenas without reading the live node count back to host.
    LETSourceWork*   d_fmm_source_work;
    LETExportDescriptor* d_fmm_descriptors;
    int32_t*          d_fmm_descriptor_count;
    // Device-side traversal validation status.  Any nonzero value prevents
    // publication, so malformed tree metadata cannot silently drop forces.
    int32_t*          d_fmm_export_status;
    int32_t           fmm_descriptor_capacity;
    int32_t           fmm_send_node_capacity_total;
    int32_t           fmm_send_particle_capacity_total;
    int32_t*          d_fmm_mask_node_counts; // 256 exact discovery bins
    int32_t*          d_fmm_mask_write_counts;
    int32_t*          d_fmm_mask_offsets;
    uint8_t*          d_fmm_mask_shared;
    LETSenderSegmentPlan fmm_sender_plan;
    int32_t          fmm_source_work_capacity;
    int32_t          fmm_source_work_count;
    bool             last_fmm_send_grew;
    bool             last_fmm_send_retried;
    bool             last_fmm_recv_grew;
    LETSharingSummary fmm_sharing_summary;
    uint32_t          timing_sample_interval;
    std::vector<LETTimingSlot> export_timing_ring;
    std::vector<LETTimingSlot> receive_timing_ring;
    std::vector<LinkClass> peer_link_classes;

    // Configuration (unchanged semantics)
    int32_t          start_level;
    float            theta;
    size_t           min_shared_saved_bytes;
    int32_t          max_shared_buckets;

    // Status
    bool             initialized;
    bool             use_legacy_mpi;   // Config gate: true = old MPI path

    LETPushState()
        : gpu_id(-1), num_gpus(0), peers(nullptr),
          d_all_let_centers(nullptr), d_all_let_sizes(nullptr),
          d_all_let_multipoles(nullptr), d_all_let_particles(nullptr),
          d_let_node_counts(nullptr), d_let_particle_counts(nullptr),
          d_all_recv_centers(nullptr), d_all_recv_sizes(nullptr),
          d_all_recv_multipoles(nullptr), d_all_recv_particles(nullptr),
          d_recv_node_counts(nullptr), d_recv_particle_counts(nullptr),
          h_pinned_my_send_node_counts(nullptr),
          h_pinned_my_send_particle_counts(nullptr),
          compute_stream(nullptr), comm_stream(nullptr),
          d_fmm_send_nodes(nullptr), d_fmm_send_particles(nullptr),
          d_fmm_recv_nodes(nullptr), d_fmm_recv_particles(nullptr),
          d_fmm_send_node_offsets(nullptr), d_fmm_send_particle_offsets(nullptr),
          d_fmm_recv_node_offsets(nullptr), d_fmm_recv_particle_offsets(nullptr),
          d_fmm_recv_node_counts(nullptr), d_fmm_recv_particle_counts(nullptr),
          h_fmm_send_node_offsets(nullptr), h_fmm_send_particle_offsets(nullptr),
          h_fmm_recv_node_offsets(nullptr), h_fmm_recv_particle_offsets(nullptr),
          h_fmm_send_node_capacities(nullptr), h_fmm_send_particle_capacities(nullptr),
          h_fmm_recv_node_capacities(nullptr), h_fmm_recv_particle_capacities(nullptr),
          fmm_arena_bytes(0), d_fmm_source_work(nullptr),
          d_fmm_descriptors(nullptr), d_fmm_descriptor_count(nullptr),
          d_fmm_export_status(nullptr),
          fmm_descriptor_capacity(0), d_fmm_mask_node_counts(nullptr),
          fmm_send_node_capacity_total(0), fmm_send_particle_capacity_total(0),
          d_fmm_mask_write_counts(nullptr), d_fmm_mask_offsets(nullptr),
          d_fmm_mask_shared(nullptr), fmm_sender_plan{},
          fmm_source_work_capacity(0), fmm_source_work_count(0),
          last_fmm_send_grew(false), last_fmm_send_retried(false),
          last_fmm_recv_grew(false), timing_sample_interval(8),
          fmm_sharing_summary{},
          start_level(2), theta(0.5f),
          min_shared_saved_bytes(65536), max_shared_buckets(8),
          initialized(false), use_legacy_mpi(false) {}
};

} // namespace cudadem

#endif // CUDADEM_LET_TYPES_H

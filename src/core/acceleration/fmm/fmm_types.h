// =============================================================================
// LBVH-FMM (Linear Bounding Volume Hierarchy - Fast Multipole Method) Types
// =============================================================================
// Core data structures for the LBVH-FMM self-gravity module.
// Follows the pattern from bh_tree_types.h for consistency.
// All structs are POD-compatible for GPU constant memory and device use.
// =============================================================================

#ifndef CUDADEM_FMM_TYPES_H
#define CUDADEM_FMM_TYPES_H

#include "core/constant/command_line_interface.h"
#include <cuda_runtime.h>

// UnifiedNode for Phase 2 unified FMM (octree/quadtree).
// Coexists with FMMOctreeNode during migration.
#include "../body_fitted_fmm/unified_node.cuh"

#include <cstdint>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// =============================================================================
// Precision Type System (two named types)
// =============================================================================
// Spec: docs/superpowers/specs/2026-06-15-fmm-precision-refactor-design.md
//
// far_real_t: Far-field (P2M, M2M, M2L, L2L, L2P) — ALWAYS double.
//              Must remain double for numerical stability in multipole
//              accumulation, G-table GEMM, and high-order Taylor expansions.
// p2p_real_t: P2P near-field — float if FMM_FLOAT_P2P is defined, else double.
//             P2P (direct 1/r^2 summation) tolerates float without measurable
//             accuracy loss.
// =============================================================================

// --- Far-Field Type: ALWAYS double ---
using far_real_t = double;
#define FAR_REAL(x)    (x)
#define FAR_CAST(x)    ((far_real_t)(x))
#define FAR_SQRT(x)    sqrt(x)
#define FAR_FMAX(x, y) fmax(x, y)
#define FAR_FMIN(x, y) fmin(x, y)
#define FAR_FABS(x)    fabs(x)

#ifdef __CUDACC__
__device__ __forceinline__
far_real_t FAR_FMA(far_real_t x, far_real_t y, far_real_t z) {
    return fma(x, y, z);
}
#endif

// --- P2P Type: configurable at build time via FMM_FLOAT_P2P ---
#ifdef FMM_FLOAT_P2P
using p2p_real_t = float;
#define P2P_REAL(x)    (x##f)
#define P2P_CAST(x)    ((p2p_real_t)(x))
#define P2P_SQRT(x)    sqrtf(x)
#define P2P_FMAX(x, y) fmaxf(x, y)
#define P2P_FMIN(x, y) fminf(x, y)
#define P2P_FABS(x)    fabsf(x)
#ifdef __CUDACC__
__device__ __forceinline__
p2p_real_t P2P_FMA(p2p_real_t x, p2p_real_t y, p2p_real_t z) {
    return fmaf(x, y, z);
}
#endif
#else
using p2p_real_t = double;
#define P2P_REAL(x)    (x)
#define P2P_CAST(x)    ((p2p_real_t)(x))
#define P2P_SQRT(x)    sqrt(x)
#define P2P_FMAX(x, y) fmax(x, y)
#define P2P_FMIN(x, y) fmin(x, y)
#define P2P_FABS(x)    fabs(x)
#ifdef __CUDACC__
__device__ __forceinline__
p2p_real_t P2P_FMA(p2p_real_t x, p2p_real_t y, p2p_real_t z) {
    return fma(x, y, z);
}
#endif
#endif

namespace fmm {

// =============================================================================
// Constants
// =============================================================================

constexpr int FMM_MAX_LEVELS = 30;
constexpr int FMM_MAX_DENSE_LEVELS = 11;  // D=2 levels 0..10 (D=3 uses 0..7)
constexpr int FMM_MAX_COEFFS = 32;
constexpr int FMM_ACTIVE_COEFFS = 28;  // p=4 with supplementary octupole: 1+3+5+7+3+9
constexpr int FMM_INTERACTION_LIST_SIZE = 316;  // 7³ - 3³ (valid M2L grid positions)

// =============================================================================
// FMM Octree Node Structure (64-byte aligned for GPU efficiency)
// =============================================================================
// Packed node structure for the LBVH-FMM octree.
// Total size: 64 bytes (cache-line aligned)
// =============================================================================

struct __align__(64) FMMOctreeNode {
    float3 aabb_min;           // 12 bytes: bounding box min
    float3 aabb_max;           // 12 bytes: bounding box max
    float3 com;                // 12 bytes: center of mass (P2M/M2M reference)
    float  R_max;              //  4 bytes: bounding sphere radius from com (MAC)
    int32_t parent_idx;       //  4 bytes: parent index (-1 for root)
    int32_t first_child;       //  4 bytes: index into children_indices[] (-1 for leaf)
    int32_t particle_begin;    //  4 bytes: start index in particle list
    int32_t particle_end;      //  4 bytes: end index in particle list
    uint8_t level;             //  1 byte:  tree level (0=root)
    uint8_t n_children;        //  1 byte:  number of valid children (0-8)
    uint8_t flags;             //  1 byte:  node flags (leaf, has_multipole, etc.)
    uint8_t _pad0;             //  1 byte:  padding
    uint32_t compact_morton;   //  4 bytes: 21-bit Morton for dense level map (O(1) lookup)

    __host__ __device__
    FMMOctreeNode()
        : aabb_min(make_float3(0, 0, 0)), aabb_max(make_float3(0, 0, 0)),
          com(make_float3(0, 0, 0)), R_max(0),
          parent_idx(-1), first_child(-1),
          particle_begin(0), particle_end(0),
          level(0), n_children(0), flags(0), _pad0(0), compact_morton(0) {}

    __host__ __device__
    FMMOctreeNode(const float3& min, const float3& max, uint8_t lvl)
        : aabb_min(min), aabb_max(max),
          com(make_float3(0, 0, 0)), R_max(0),
          parent_idx(-1), first_child(-1),
          particle_begin(0), particle_end(0),
          level(lvl), n_children(0), flags(0), _pad0(0), compact_morton(0) {}
};

static_assert(sizeof(FMMOctreeNode) == 64, "FMMOctreeNode must be 64 bytes");

// =============================================================================
// Multipole Expansion Structure (128 bytes)
// =============================================================================
// Stores the multipole coefficients for a single node.
// Uses trace-free coefficients up to p=4 plus 3 supplementary octupole
// components (O_xyy, O_xzz, O_yzz) needed for complete Cartesian shift.
// Layout: M(1) + D(3) + Q_tf(5) + O_tf(7) + H_tf(9) + O_supp(3) = 28 active + 4 padding
// Total size: 128 bytes (32 floats)
// =============================================================================

struct MultipoleExpansion {
    far_real_t coeffs[FMM_MAX_COEFFS];  // 28 active + 4 padding (float: 128B, double: 256B)

    __host__ __device__
    MultipoleExpansion() {
        std::memset(coeffs, 0, sizeof(coeffs));
    }

    __host__ __device__
    void zero() {
        std::memset(coeffs, 0, sizeof(coeffs));
    }

    __host__ __device__
    far_real_t get_coeff(int index) const {
        return (index >= 0 && index < FMM_MAX_COEFFS) ? coeffs[index] : FAR_REAL(0.0);
    }

    __host__ __device__
    void set_coeff(int index, far_real_t value) {
        if (index >= 0 && index < FMM_MAX_COEFFS) {
            coeffs[index] = value;
        }
    }
};

static_assert(sizeof(MultipoleExpansion) == (sizeof(far_real_t) * FMM_MAX_COEFFS), "MultipoleExpansion size must match far_real_t layout");

// =============================================================================
// Local Expansion Structure (128 bytes)
// =============================================================================
// Stores the local expansion coefficients for a single node.
// Same layout as MultipoleExpansion (28 active + 4 padding) for M2L translation efficiency.
// =============================================================================

struct LocalExpansion {
    far_real_t coeffs[FMM_MAX_COEFFS];  // 28 active + 4 padding (float: 128B, double: 256B)

    __host__ __device__
    LocalExpansion() {
        std::memset(coeffs, 0, sizeof(coeffs));
    }

    __host__ __device__
    void zero() {
        std::memset(coeffs, 0, sizeof(coeffs));
    }
};

static_assert(sizeof(LocalExpansion) == (sizeof(far_real_t) * FMM_MAX_COEFFS), "LocalExpansion size must match far_real_t layout");

// =============================================================================
// M2L (Multipole-to-Local) Queue Structure
// =============================================================================
// Tracks pending M2L translation pairs for batch processing.
// =============================================================================

struct M2LQueue {
    int2* pairs;           // Device: (target_idx, source_idx) pairs
    int32_t* count;        // Device: Current queue count
    int32_t* h_count;      // Host mirror of count
    int32_t capacity;      // Maximum queue capacity

    __host__ __device__
    M2LQueue()
        : pairs(nullptr), count(nullptr), h_count(nullptr), capacity(0) {}
};

// =============================================================================
// P2P (Particle-to-Particle) Queue Structure
// =============================================================================
// Tracks pending P2P interaction pairs for near-field computation.
// =============================================================================

struct P2PQueue {
    int2* pairs;           // Device: (particle_i, particle_j) pairs
    int32_t* count;        // Device: Current queue count
    int32_t* h_count;      // Host mirror of count
    int32_t capacity;      // Maximum queue capacity

    __host__ __device__
    P2PQueue()
        : pairs(nullptr), count(nullptr), h_count(nullptr), capacity(0) {}
};

// =============================================================================
// =============================================================================
// Work Queue for Single-Pass Dual Tree Traversal
// =============================================================================
// Global work queue for distributing (nodeA, nodeB) pairs across warps.
// Uses monotonically-increasing head/tail counters with ring buffer indexing.
// =============================================================================

struct WorkQueue {
    int2*    pairs;          // Device: (nodeA, nodeB) pairs
    int32_t* head;           // Device: Next slot to claim (atomic, monotonically increasing)
    int32_t* tail;           // Device: Next slot to produce (atomic, monotonically increasing)
    int32_t  capacity;       // Ring buffer size (index via slot % capacity)
    int32_t* active_warps;   // Device: Count of warps currently processing a pair

    __host__ __device__
    WorkQueue()
        : pairs(nullptr), head(nullptr), tail(nullptr),
          capacity(0), active_warps(nullptr) {}
};

// =============================================================================
// FMM Statistics Structure
// =============================================================================
// Runtime statistics for performance monitoring and debugging.
// =============================================================================

struct FMMStats {
    int64_t num_octree_nodes;
    int64_t num_m2l_interactions;
    int64_t num_p2p_interactions;
    int64_t num_p2m_operations;
    int64_t num_m2m_operations;
    int64_t num_l2p_operations;
    int64_t num_l2l_operations;
    // Overflow diagnostics - track lost interactions
    int64_t dropped_p2p;      // P2P pairs silently discarded due to queue overflow
    int64_t dropped_m2l;      // M2L pairs silently discarded due to queue overflow
    int64_t fallback_p2p;     // P2P pairs computed directly on overflow (fallback)
    int64_t num_m2l_internal; // M2L pairs where target is internal node
    int64_t num_m2l_leaf;     // M2L pairs where target is leaf node
    double  total_build_time;
    double  total_translation_time;
    double  total_evaluation_time;

    __host__ __device__
    FMMStats()
        : num_octree_nodes(0), num_m2l_interactions(0), num_p2p_interactions(0),
          num_p2m_operations(0), num_m2m_operations(0),
          num_l2p_operations(0), num_l2l_operations(0),
          dropped_p2p(0), dropped_m2l(0), fallback_p2p(0),
          num_m2l_internal(0), num_m2l_leaf(0),
          total_build_time(0.0), total_translation_time(0.0),
          total_evaluation_time(0.0) {}
};

// =============================================================================
// FMM Device Data Structure
// =============================================================================
// Device-side data for FMM computation.
// All pointers reference device memory.
// =============================================================================

struct FMMDeviceData {
    // Tree structure
    FMMOctreeNode* nodes;          // Octree nodes (Morton-sorted)
    int* level_start;              // [FMM_MAX_LEVELS] Starting index per level
    int* level_count;              // [FMM_MAX_LEVELS] Node count per level
    int* level_list;               // [num_nodes] Node indices ordered by level

    // Dense level maps for O(1) Morton→node lookup
    // level_dense_map[L] has (1<<D)^L entries, indexed by compact_morton
    uint32_t* level_dense_map[FMM_MAX_DENSE_LEVELS];          // [level] → array of node indices
    int       level_dense_capacity[FMM_MAX_DENSE_LEVELS];     // [level] → capacity ((1<<D)^level)

    // Multipole and local expansions
    MultipoleExpansion* multipoles;    // [num_nodes] Multipole expansions
    double* multipoles_dbl;            // [num_nodes * FMM_ACTIVE_COEFFS] P2M/M2M scratch
    LocalExpansion* local_expansions;  // [num_nodes] Local expansions

    // Interaction queues
    M2LQueue m2l_queue;
    P2PQueue p2p_queue;

    // Index mappings
    int* sorted_to_original;       // [num_particles] Morton sort mapping
    int* particle_to_leaf;         // [num_particles] Particle -> leaf mapping

    // Children indices (per octant, 8 per node)
    int* children_indices;         // [8 * num_nodes] Child index per octant (-1 if empty)

    // Phase 2: Unified node arrays (converted from FMMOctreeNode after build)
    body_fitted_fmm::UnifiedNode* unified_nodes;        // [num_nodes] converted nodes
    int* children_compact;             // [total_children] compact children array
    int  children_compact_count;       // actual count

    // Statistics
    FMMStats* d_stats;             // Device-side statistics

    // Counts
    int num_nodes;
    int num_particles;
    int max_level;
    bool verbose;

    // =====================================================================
    // Equivalent-density FMM fields (np=4, Chebyshev equivalent densities)
    // =====================================================================

    // Precomputed sampling points (read-only, uploaded once)
    far_real_t* sp_uc;           // Upward check points [3 × sp_uc_n_padded]
    far_real_t* sp_ue;           // Upward equivalent points [3 × sp_ue_n_padded]
    far_real_t* sp_dc;           // Downward check points [3 × sp_dc_n_padded]
    far_real_t* sp_de;           // Downward evaluation points [3 × sp_de_n_padded]

    // Precomputed translation matrices (read-only)
    far_real_t* UC2UE;           // Check → Equivalent (upward) [sp_n_padded × sp_n_padded]
    far_real_t* UE2UC;           // Equivalent → Check (upward) [8 × sp_n_padded × sp_n_padded]
    far_real_t* DC2DE;           // Check → Evaluation (downward) [sp_n_padded × sp_n_padded]
    far_real_t* DE2DC;           // Evaluation → Check (downward) [8 × sp_n_padded × sp_n_padded]

    // Per-node equivalent/check densities (replaces MultipoleExpansion / LocalExpansion)
    far_real_t* upw_equ_den;     // [num_nodes × sp_ue_n_padded] upward equivalent densities
    far_real_t* dwn_chk_val;     // [num_nodes × sp_dc_n_padded] downward check-point values
    far_real_t* dwn_equ_den;     // [num_nodes × sp_de_n_padded] downward evaluation densities

    // FFT staging arrays (batched, stride-based)
    far_real_t* fft_reg_den;     // [num_nodes × 512] real density grid (8×8×8)
    cufftDoubleComplex* vlist_src;    // [num_nodes × 320] FFT'd source densities (complex)
    cufftDoubleComplex* vlist_trans;  // [343 × 320] precomputed transfer functions
    cufftDoubleComplex* vlist_trg;    // [num_nodes × 320] FFT'd target (temporary)

    // cuFFT plan handles
    cufftHandle fft_plan_d2z;    // Real → Complex batched plan
    cufftHandle fft_plan_z2d;    // Complex → Real batched plan

    // V-list output from traversal
    int* vlist_targets;          // [num_vlist_entries] target node indices
    int* vlist_sources_arr;      // [num_vlist_entries] source node indices
    int* vlist_ti;               // [num_vlist_entries] transfer indices (optional, diagnostics)
    int* vlist_count;            // [1] device-side V-list entry count
    int* h_vlist_count;          // Host mirror of vlist_count
    int  vlist_capacity;         // Maximum V-list capacity

    // FMM dimension constants (set during init)
    int sp_n;                    // 56 (np=4)
    int sp_n_padded;             // 64
    int fft_grid_n;              // 8
    int fft_grid_size;           // 512
    int fft_complex_size;        // 320

    // Global projection reduction scalars
    far_real_t* gp_scalars;         // [16] reduction scalars for global projection

    __host__ __device__
    FMMDeviceData()
        : nodes(nullptr), level_start(nullptr), level_count(nullptr),
          level_list(nullptr), multipoles(nullptr), multipoles_dbl(nullptr),
          local_expansions(nullptr),
          sorted_to_original(nullptr), particle_to_leaf(nullptr), children_indices(nullptr),
          unified_nodes(nullptr), children_compact(nullptr), children_compact_count(0),
          d_stats(nullptr),
          num_nodes(0), num_particles(0), max_level(0), verbose(false)
          , sp_uc(nullptr), sp_ue(nullptr), sp_dc(nullptr), sp_de(nullptr)
          , UC2UE(nullptr), UE2UC(nullptr), DC2DE(nullptr), DE2DC(nullptr)
          , upw_equ_den(nullptr), dwn_chk_val(nullptr), dwn_equ_den(nullptr)
          , fft_reg_den(nullptr), vlist_src(nullptr), vlist_trans(nullptr), vlist_trg(nullptr)
          , fft_plan_d2z(0), fft_plan_z2d(0)
          , vlist_targets(nullptr), vlist_sources_arr(nullptr), vlist_ti(nullptr)
          , vlist_count(nullptr)
          , h_vlist_count(nullptr), vlist_capacity(0)
          , gp_scalars(nullptr)
          , sp_n(56), sp_n_padded(64), fft_grid_n(8), fft_grid_size(512), fft_complex_size(320)
    {
        for (int l = 0; l < FMM_MAX_DENSE_LEVELS; l++) {
            level_dense_map[l] = nullptr;
            level_dense_capacity[l] = 0;
        }
    }
};

// =============================================================================
// FMM Host Data Structure
// =============================================================================
// Host-side build and state management data.
// Contains intermediate buffers for tree construction.
// =============================================================================

struct FMMHostData {
    // ===== Tree topology device arrays (owned directly by FMM; formerly aliased
    //       from a BHTreeBuildData container, now first-class fields) =====
    uint64_t* morton_keys;           // [num_particles] Morton codes
    int* sorted_indices;             // [num_particles] Particle indices after sort
    int2* node_range;                // [max_nodes] (first_particle, last_particle)
    int* node_children;              // [8 * max_nodes] Octant children indices
    int* node_parent;                // [max_nodes] Parent indices
    int* node_level;                 // [max_nodes] Tree levels
    int* node_is_leaf;               // [max_nodes] Leaf flag (0/1)
    float* node_cell_size;           // [max_nodes] Cell size at node
    far_real_t* node_com_x;              // [max_nodes] Center of mass x
    far_real_t* node_com_y;              // [max_nodes] Center of mass y
    far_real_t* node_com_z;              // [max_nodes] Center of mass z
    int* node_first_child;           // [max_nodes] First child index
    int* node_n_children;            // [max_nodes] Child count

    // Level-order processing
    int* h_level_start;              // [FMM_MAX_LEVELS] Host level starts
    int* h_level_count;              // [FMM_MAX_LEVELS] Host level counts
    int* h_level_list;               // [max_nodes] Host level-ordered nodes

    // Atomic counters
    int* d_num_nodes;
    int* d_max_depth;

    // ===== Domain Information =====
    far_real_t domain_min_x;             // Padded bounding box min x
    far_real_t domain_min_y;             // Padded bounding box min y
    far_real_t domain_min_z;             // Padded bounding box min z
    far_real_t domain_inv_size;          // 1 / padded domain size
    p2p_real_t* p2p_pos_x;    // P2P position staging (float or double depending on build)
    p2p_real_t* p2p_pos_y;
    p2p_real_t* p2p_pos_z;
    p2p_real_t* p2p_force_x;  // P2P force staging (float or double depending on build)
    p2p_real_t* p2p_force_y;
    p2p_real_t* p2p_force_z;
    float domain_root_cell_size;     // Root cell size

    // ===== Particle-to-leaf map =====
    int* particle_to_leaf;           // [num_particles] Particle -> leaf mapping

    // ===== Capacity Tracking =====
    int allocated_max_particles;
    int allocated_max_nodes;

    // ===== Statistics (host mirror) =====
    FMMStats h_stats;

    __host__ __device__
    FMMHostData()
        : morton_keys(nullptr), sorted_indices(nullptr),
          node_range(nullptr), node_children(nullptr),
          node_parent(nullptr), node_level(nullptr),
          node_is_leaf(nullptr), node_cell_size(nullptr),
          node_com_x(nullptr), node_com_y(nullptr), node_com_z(nullptr),
          node_first_child(nullptr), node_n_children(nullptr),
          h_level_start(nullptr), h_level_count(nullptr), h_level_list(nullptr),
          d_num_nodes(nullptr), d_max_depth(nullptr),
          domain_min_x(0), domain_min_y(0), domain_min_z(0),
          domain_inv_size(0),
          p2p_pos_x(nullptr), p2p_pos_y(nullptr), p2p_pos_z(nullptr),
          p2p_force_x(nullptr), p2p_force_y(nullptr), p2p_force_z(nullptr),
          domain_root_cell_size(0),
          particle_to_leaf(nullptr),
          allocated_max_particles(0), allocated_max_nodes(0) {}
};

// =============================================================================
// Device Helper Functions
// =============================================================================

#ifdef __CUDACC__

// Coefficient order for trace-free multipoles (p=4)
// Returns the expansion order (0-4) for a given coefficient index
__device__ __forceinline__
int get_coeff_order(int idx) {
    // Orders: M(1) + D(3) + Q(5) + O_tf(7) + O_supp(3) + H(9) = 28 active coefficients
    // Indices 25-27 are supplementary (non-trace-free) octupole components
    // needed for complete Cartesian Taylor contraction in L2L and L2P.
    if (idx == 0) return 0;                          // M (monopole)
    if (idx >= 1 && idx <= 3) return 1;              // D (dipole)
    if (idx >= 4 && idx <= 8) return 2;              // Q (quadrupole)
    if (idx >= 9 && idx <= 15) return 3;             // O (trace-free octupole)
    if (idx >= 16 && idx <= 24) return 4;            // H (hexadecapole)
    if (idx >= 25 && idx <= 27) return 3;            // O (supplementary octupole)
    return 0;                                        // padding
}

// Get octree level from Morton code
// Returns the level (0=root) based on leading zeros
__device__ __forceinline__
int get_octree_level(uint64_t morton_code) {
    return __clzll(morton_code) / 3;
}

// Get octant at a specific level from Morton code
__device__ __forceinline__
int get_octant(uint64_t morton_code, int level) {
    // Extract 3 bits at the given level position
    int shift = 63 - (level * 3);
    return (morton_code >> shift) & 0x7;
}

// Check if two Morton codes are in the same octree cell at given level
__device__ __forceinline__
bool same_octree_cell(uint64_t morton_a, uint64_t morton_b, int level) {
    // Mask out bits below the given level
    uint64_t mask = ~((1ULL << (63 - level * 3 + 1)) - 1);
    return (morton_a & mask) == (morton_b & mask);
}

// Evaluate FMM Interaction Criterion (NOT Barnes-Hut MAC!)
// For FMM, the interaction list is defined by Chebyshev distance in cell units.
// 
// For cells at the SAME level (same size):
//   - Distance 0: same cell (skip)
//   - Distance 1: adjacent cells → P2P (near-field)
//   - Distance 2-3: interaction list → M2L (far-field at this level)
//   - Distance > 3: should interact at higher level (descend)
//
// For cells at DIFFERENT levels:
//   - Use MAC with adaptive theta based on relative sizes
//
// Returns: true = well-separated (M2L), false = near-field (P2P or descend)
//
// CRITICAL FIX: Use conservative MAC that is MONOTONIC with respect to subtree.
// If a parent is well-separated, ALL descendants must also be well-separated.
// This prevents double-counting where M2L(parent) and P2P(descendant) both cover
// the same particle pairs.
//
// Conservative MAC: dist - r_a - r_b >= (r_a + r_b) / theta
// This ensures the minimum distance between particles in A and B is large enough
// that even the closest descendants will be well-separated.
__device__ __forceinline__
bool evaluate_fmm_interaction(
    const float3& center_a, float size_a, int level_a,
    const float3& center_b, float size_b, int level_b,
    float theta_far_sq)
{
    float3 dx;
    dx.x = center_b.x - center_a.x;
    dx.y = center_b.y - center_a.y;
    dx.z = center_b.z - center_a.z;

    float dist_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
    float dist = sqrtf(dist_sq);

    float theta_far = sqrtf(theta_far_sq);

    // Conservative MAC: ensure minimum particle distance is sufficient
    // min_dist = dist - r_a - r_b (distance between closest points)
    // We require: min_dist >= (r_a + r_b) / theta
    // Rearranging: dist - r_a - r_b >= (r_a + r_b) / theta
    // Or: dist >= (r_a + r_b) * (1 + 1/theta)

    float r_sum = size_a + size_b;
    float min_dist = dist - r_sum;  // Distance between closest points

    // Check if cells are well-separated
    // Using theta_far = 0.75, we require min_dist >= r_sum / 0.75 = 1.33 * r_sum
    return (min_dist >= r_sum / theta_far);
}

__device__ __forceinline__
bool evaluate_mac(const float3& center_a, float size_a,
                  const float3& center_b, float size_b,
                  float theta_far_sq) {
    float3 dx;
    dx.x = center_b.x - center_a.x;
    dx.y = center_b.y - center_a.y;
    dx.z = center_b.z - center_a.z;

    float dist_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
    
    float r_sum = size_a + size_b;
    float r_sum_sq = r_sum * r_sum;

    // MAC: (r_a + r_b)^2 <= theta^2 * R^2
    return r_sum_sq <= theta_far_sq * dist_sq;
}

// CoM-based MAC using bounding sphere radii from center of mass
// (R_max_A + R_max_B) / ||com_A - com_B|| <= theta_far
__device__ __forceinline__
bool evaluate_com_mac(const FMMOctreeNode& nA, const FMMOctreeNode& nB, float theta_far) {
    float3 dx;
    dx.x = nB.com.x - nA.com.x;
    dx.y = nB.com.y - nA.com.y;
    dx.z = nB.com.z - nA.com.z;
    float dist_sq = dx.x*dx.x + dx.y*dx.y + dx.z*dx.z;
    float r_sum  = nA.R_max + nB.R_max;
    float r_sum_sq = r_sum * r_sum;
    return r_sum_sq <= theta_far * theta_far * dist_sq;
}

// Compute scale factors for M2L translation
// Precomputes distance-dependent scaling terms
// Computes 1/r^k for k=0..6 (7 entries), sufficient for p=4 M2L evaluation.
// M2L translation: L_i = sum_j M_j * mat[i][j] / r^(order_i + order_j + 1)
// For multipole_eval_device (evaluates via dipole row): power = 1 + order_j + 1 = order_j + 2
//   order_j=3 → power=5 → scales[4] = 1/r^4
//   order_j=4 → power=6 → scales[5] = 1/r^5
__device__ __forceinline__
void compute_scale_factors(far_real_t distance, far_real_t* scales) {
    far_real_t inv_dist = FAR_REAL(1.0) / distance;
    far_real_t inv_dist_sq = inv_dist * inv_dist;

    // Precompute powers of 1/r for translation operators
    scales[0] = FAR_REAL(1.0);           // 1/r^0 = 1
    scales[1] = inv_dist;                 // 1/r^1
    scales[2] = inv_dist_sq;              // 1/r^2
    scales[3] = inv_dist * inv_dist_sq;   // 1/r^3
    scales[4] = inv_dist_sq * inv_dist_sq; // 1/r^4
    scales[5] = inv_dist * inv_dist_sq * inv_dist_sq; // 1/r^5
    scales[6] = inv_dist_sq * inv_dist_sq * inv_dist_sq; // 1/r^6
}

// Compute direction-only M2L matrix entries (no 1/r distance factors).
// Stores result in mat[32][32] padded format — only rows/cols 0..27 are meaningful.
// Auto-generated from bvh8_fmm/derivation/generate_m2l_matrix.py
// 784 non-zero entries for p=4 trace-free Cartesian M2L.
__device__ __forceinline__
void compute_m2l_matrix_device(far_real_t nx, far_real_t ny, far_real_t nz,
                                far_real_t mat[32][32]) {
    // Zero all entries
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            mat[i][j] = FAR_REAL(0.0);
        }
    }

    // Precompute direction products
    far_real_t nxx = nx * nx, nyy = ny * ny, nzz = nz * nz;

    // Corrected M2L direction matrix from derivation/derive_m2l.py
    // 784 non-zero entries for p=4 trace-free Cartesian M2L

    // Row 0 (target order 0)
    mat[0][0] = FAR_REAL(1);  // power=1
    mat[0][1] = FAR_REAL(nx);  // power=2
    mat[0][2] = FAR_REAL(ny);  // power=2
    mat[0][3] = FAR_REAL(nz);  // power=2
    mat[0][4] = FAR_REAL((1.0/3.0)*nxx - 1.0/6.0*nyy - 1.0/6.0*nzz);  // power=3
    mat[0][5] = FAR_REAL((1.0/6.0)*nxx + (1.0/6.0)*nyy - 1.0/3.0*nzz);  // power=3
    mat[0][6] = FAR_REAL(nx*ny);  // power=3
    mat[0][7] = FAR_REAL(nx*nz);  // power=3
    mat[0][8] = FAR_REAL(ny*nz);  // power=3
    mat[0][9] = FAR_REAL((1.0/5.0)*nxx*nx - 3.0/10.0*nx*nyy - 3.0/10.0*nx*nzz);  // power=4
    mat[0][10] = FAR_REAL(-3.0/10.0*nxx*ny + (1.0/5.0)*nyy*ny - 3.0/10.0*ny*nzz);  // power=4
    mat[0][11] = FAR_REAL(-3.0/10.0*nxx*nz - 3.0/10.0*nyy*nz + (1.0/5.0)*nzz*nz);  // power=4
    mat[0][12] = FAR_REAL((6.0/5.0)*nxx*ny - 3.0/10.0*nyy*ny - 3.0/10.0*ny*nzz);  // power=4
    mat[0][13] = FAR_REAL((6.0/5.0)*nxx*nz - 3.0/10.0*nyy*nz - 3.0/10.0*nzz*nz);  // power=4
    mat[0][14] = FAR_REAL(-3.0/10.0*nxx*nz + (6.0/5.0)*nyy*nz - 3.0/10.0*nzz*nz);  // power=4
    mat[0][15] = FAR_REAL(3*nx*ny*nz);  // power=4
    mat[0][16] = FAR_REAL((1.0/8.0)*nxx*nxx - 3.0/4.0*nxx*nzz + (1.0/8.0)*nzz*nzz);  // power=5
    mat[0][17] = FAR_REAL((1.0/8.0)*nyy*nyy - 3.0/4.0*nyy*nzz + (1.0/8.0)*nzz*nzz);  // power=5
    mat[0][18] = FAR_REAL((1.0/2.0)*nx*nyy*ny - 3.0/2.0*nx*ny*nzz);  // power=5
    mat[0][19] = FAR_REAL((1.0/2.0)*nxx*nx*ny - 3.0/2.0*nx*ny*nzz);  // power=5
    mat[0][20] = FAR_REAL((1.0/2.0)*nxx*nx*nz - 1.0/2.0*nx*nzz*nz);  // power=5
    mat[0][21] = FAR_REAL((1.0/2.0)*nyy*ny*nz - 1.0/2.0*ny*nzz*nz);  // power=5
    mat[0][22] = FAR_REAL((3.0/4.0)*nxx*nyy - 3.0/4.0*nxx*nzz - 3.0/4.0*nyy*nzz + (1.0/4.0)*nzz*nzz);  // power=5
    mat[0][23] = FAR_REAL((3.0/2.0)*nxx*ny*nz - 1.0/2.0*ny*nzz*nz);  // power=5
    mat[0][24] = FAR_REAL((3.0/2.0)*nx*nyy*nz - 1.0/2.0*nx*nzz*nz);  // power=5
    mat[0][25] = FAR_REAL(-3.0/10.0*nxx*nx + (6.0/5.0)*nx*nyy - 3.0/10.0*nx*nzz);  // power=4
    mat[0][26] = FAR_REAL(-3.0/10.0*nxx*nx - 3.0/10.0*nx*nyy + (6.0/5.0)*nx*nzz);  // power=4
    mat[0][27] = FAR_REAL(-3.0/10.0*nxx*ny - 3.0/10.0*nyy*ny + (6.0/5.0)*ny*nzz);  // power=4

    // Row 1 (target order 1)
    mat[1][0] = FAR_REAL(-nx);  // power=2
    mat[1][1] = FAR_REAL(-2*nxx + nyy + nzz);  // power=3
    mat[1][2] = FAR_REAL(-3*nx*ny);  // power=3
    mat[1][3] = FAR_REAL(-3*nx*nz);  // power=3
    mat[1][4] = FAR_REAL(-nxx*nx + (3.0/2.0)*nx*nyy + (3.0/2.0)*nx*nzz);  // power=4
    mat[1][5] = FAR_REAL(-1.0/2.0*nxx*nx - 1.0/2.0*nx*nyy + 2*nx*nzz);  // power=4
    mat[1][6] = FAR_REAL(-4*nxx*ny + nyy*ny + ny*nzz);  // power=4
    mat[1][7] = FAR_REAL(-4*nxx*nz + nyy*nz + nzz*nz);  // power=4
    mat[1][8] = FAR_REAL(-5*nx*ny*nz);  // power=4
    mat[1][9] = FAR_REAL(-7.0/2.0*nxx*nxx + 3*nxx - 3.0/10.0);  // power=5
    mat[1][10] = FAR_REAL((3.0/2.0)*nxx*nx*ny - 2*nx*nyy*ny + (3.0/2.0)*nx*ny*nzz);  // power=5
    mat[1][11] = FAR_REAL((3.0/2.0)*nxx*nx*nz + (3.0/2.0)*nx*nyy*nz - 2*nx*nzz*nz);  // power=5
    mat[1][12] = FAR_REAL(-6*nxx*nx*ny + (9.0/2.0)*nx*nyy*ny + (9.0/2.0)*nx*ny*nzz);  // power=5
    mat[1][13] = FAR_REAL(-6*nxx*nx*nz + (9.0/2.0)*nx*nyy*nz + (9.0/2.0)*nx*nzz*nz);  // power=5
    mat[1][14] = FAR_REAL((3.0/2.0)*nxx*nx*nz - 9*nx*nyy*nz + (3.0/2.0)*nx*nzz*nz);  // power=5
    mat[1][15] = FAR_REAL(-18*nxx*ny*nz + 3*nyy*ny*nz + 3*ny*nzz*nz);  // power=5
    mat[1][16] = FAR_REAL(-9.0/8.0*nxx*nxx*nx + (27.0/4.0)*nxx*nx*nzz + (1.0/2.0)*nxx*nx - 9.0/8.0*nx*nzz*nzz - 3.0/2.0*nx*nzz);  // power=6
    mat[1][17] = FAR_REAL((3.0/28.0)*nxx*nxx*nx - 15.0/28.0*nxx*nx*nyy - 15.0/28.0*nxx*nx*nzz - 99.0/56.0*nx*nyy*nyy + (153.0/28.0)*nx*nyy*nzz + (3.0/4.0)*nx*nyy - 99.0/56.0*nx*nzz*nzz + (3.0/4.0)*nx*nzz - 3.0/28.0*nx);  // power=6
    mat[1][18] = FAR_REAL(-9.0/2.0*nxx*nyy*ny + (27.0/2.0)*nxx*ny*nzz + (1.0/2.0)*nyy*ny - 3.0/2.0*ny*nzz);  // power=6
    mat[1][19] = FAR_REAL(-9.0/2.0*nxx*nxx*ny + (27.0/2.0)*nxx*ny*nzz + (3.0/2.0)*nxx*ny - 3.0/2.0*ny*nzz);  // power=6
    mat[1][20] = FAR_REAL(-9.0/2.0*nxx*nxx*nz + (9.0/2.0)*nxx*nzz*nz + (3.0/2.0)*nxx*nz - 1.0/2.0*nzz*nz);  // power=6
    mat[1][21] = FAR_REAL(-9.0/2.0*nx*nyy*ny*nz + (9.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[1][22] = FAR_REAL((3.0/28.0)*nxx*nxx*nx - 51.0/7.0*nxx*nx*nyy + (87.0/14.0)*nxx*nx*nzz - 9.0/14.0*nx*nyy*nyy + (153.0/28.0)*nx*nyy*nzz + (9.0/4.0)*nx*nyy - 81.0/28.0*nx*nzz*nzz - 3.0/4.0*nx*nzz - 3.0/28.0*nx);  // power=6
    mat[1][23] = FAR_REAL(-21.0/2.0*nxx*nx*ny*nz + 3*nx*nyy*ny*nz + (15.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[1][24] = FAR_REAL(-27.0/2.0*nxx*nyy*nz + (9.0/2.0)*nxx*nzz*nz + (3.0/2.0)*nyy*nz - 1.0/2.0*nzz*nz);  // power=6
    mat[1][25] = FAR_REAL(-21.0/2.0*nxx*nyy + (3.0/2.0)*nxx + (3.0/2.0)*nyy - 3.0/10.0);  // power=5
    mat[1][26] = FAR_REAL(-21.0/2.0*nxx*nzz + (3.0/2.0)*nxx + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
    mat[1][27] = FAR_REAL((3.0/2.0)*nxx*nx*ny + (3.0/2.0)*nx*nyy*ny - 9*nx*ny*nzz);  // power=5

    // Row 2 (target order 1)
    mat[2][0] = FAR_REAL(-ny);  // power=2
    mat[2][1] = FAR_REAL(-3*nx*ny);  // power=3
    mat[2][2] = FAR_REAL(nxx - 2*nyy + nzz);  // power=3
    mat[2][3] = FAR_REAL(-3*ny*nz);  // power=3
    mat[2][4] = FAR_REAL(-2*nxx*ny + (1.0/2.0)*nyy*ny + (1.0/2.0)*ny*nzz);  // power=4
    mat[2][5] = FAR_REAL(-1.0/2.0*nxx*ny - 1.0/2.0*nyy*ny + 2*ny*nzz);  // power=4
    mat[2][6] = FAR_REAL(nxx*nx - 4*nx*nyy + nx*nzz);  // power=4
    mat[2][7] = FAR_REAL(-5*nx*ny*nz);  // power=4
    mat[2][8] = FAR_REAL(nxx*nz - 4*nyy*nz + nzz*nz);  // power=4
    mat[2][9] = FAR_REAL(-2*nxx*nx*ny + (3.0/2.0)*nx*nyy*ny + (3.0/2.0)*nx*ny*nzz);  // power=5
    mat[2][10] = FAR_REAL(-7.0/2.0*nyy*nyy + 3*nyy - 3.0/10.0);  // power=5
    mat[2][11] = FAR_REAL((3.0/2.0)*nxx*ny*nz + (3.0/2.0)*nyy*ny*nz - 2*ny*nzz*nz);  // power=5
    mat[2][12] = FAR_REAL(-21.0/2.0*nxx*nyy + (3.0/2.0)*nxx + (3.0/2.0)*nyy - 3.0/10.0);  // power=5
    mat[2][13] = FAR_REAL(-9*nxx*ny*nz + (3.0/2.0)*nyy*ny*nz + (3.0/2.0)*ny*nzz*nz);  // power=5
    mat[2][14] = FAR_REAL((9.0/2.0)*nxx*ny*nz - 6*nyy*ny*nz + (9.0/2.0)*ny*nzz*nz);  // power=5
    mat[2][15] = FAR_REAL(3*nxx*nx*nz - 18*nx*nyy*nz + 3*nx*nzz*nz);  // power=5
    mat[2][16] = FAR_REAL(-9.0/8.0*nxx*nxx*ny + (27.0/4.0)*nxx*ny*nzz - 9.0/8.0*ny*nzz*nzz);  // power=6
    mat[2][17] = FAR_REAL(-9.0/8.0*nyy*nyy*ny + (27.0/4.0)*nyy*ny*nzz + (1.0/2.0)*nyy*ny - 9.0/8.0*ny*nzz*nzz - 3.0/2.0*ny*nzz);  // power=6
    mat[2][18] = FAR_REAL((3.0/14.0)*nxx*nxx*nx - 15.0/14.0*nxx*nx*nyy - 15.0/14.0*nxx*nx*nzz - 81.0/14.0*nx*nyy*nyy + (153.0/14.0)*nx*nyy*nzz + 3*nx*nyy - 9.0/7.0*nx*nzz*nzz - 3.0/14.0*nx);  // power=6
    mat[2][19] = FAR_REAL((3.0/14.0)*nxx*nxx*nx - 39.0/7.0*nxx*nx*nyy - 15.0/14.0*nxx*nx*nzz + (1.0/2.0)*nxx*nx - 9.0/7.0*nx*nyy*nyy + (153.0/14.0)*nx*nyy*nzz + (3.0/2.0)*nx*nyy - 9.0/7.0*nx*nzz*nzz - 3.0/14.0*nx);  // power=6
    mat[2][20] = FAR_REAL(-9.0/2.0*nxx*nx*ny*nz + (9.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[2][21] = FAR_REAL(-9.0/2.0*nyy*nyy*nz + (9.0/2.0)*nyy*nzz*nz + (3.0/2.0)*nyy*nz - 1.0/2.0*nzz*nz);  // power=6
    mat[2][22] = FAR_REAL(-27.0/4.0*nxx*nyy*ny + (27.0/4.0)*nxx*ny*nzz + (3.0/2.0)*nxx*ny + (27.0/4.0)*nyy*ny*nzz - 9.0/4.0*ny*nzz*nzz - 3.0/2.0*ny*nzz);  // power=6
    mat[2][23] = FAR_REAL(-27.0/2.0*nxx*nyy*nz + (3.0/2.0)*nxx*nz + (9.0/2.0)*nyy*nzz*nz - 1.0/2.0*nzz*nz);  // power=6
    mat[2][24] = FAR_REAL(3*nxx*nx*ny*nz - 21.0/2.0*nx*nyy*ny*nz + (15.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[2][25] = FAR_REAL((9.0/2.0)*nxx*nx*ny - 6*nx*nyy*ny + (9.0/2.0)*nx*ny*nzz);  // power=5
    mat[2][26] = FAR_REAL((3.0/2.0)*nxx*nx*ny + (3.0/2.0)*nx*nyy*ny - 9*nx*ny*nzz);  // power=5
    mat[2][27] = FAR_REAL(-21.0/2.0*nyy*nzz + (3.0/2.0)*nyy + (3.0/2.0)*nzz - 3.0/10.0);  // power=5

    // Row 3 (target order 1)
    mat[3][0] = FAR_REAL(-nz);  // power=2
    mat[3][1] = FAR_REAL(-3*nx*nz);  // power=3
    mat[3][2] = FAR_REAL(-3*ny*nz);  // power=3
    mat[3][3] = FAR_REAL(nxx + nyy - 2*nzz);  // power=3
    mat[3][4] = FAR_REAL(-2*nxx*nz + (1.0/2.0)*nyy*nz + (1.0/2.0)*nzz*nz);  // power=4
    mat[3][5] = FAR_REAL(-3.0/2.0*nxx*nz - 3.0/2.0*nyy*nz + nzz*nz);  // power=4
    mat[3][6] = FAR_REAL(-5*nx*ny*nz);  // power=4
    mat[3][7] = FAR_REAL(nxx*nx + nx*nyy - 4*nx*nzz);  // power=4
    mat[3][8] = FAR_REAL(nxx*ny + nyy*ny - 4*ny*nzz);  // power=4
    mat[3][9] = FAR_REAL(-2*nxx*nx*nz + (3.0/2.0)*nx*nyy*nz + (3.0/2.0)*nx*nzz*nz);  // power=5
    mat[3][10] = FAR_REAL((3.0/2.0)*nxx*ny*nz - 2*nyy*ny*nz + (3.0/2.0)*ny*nzz*nz);  // power=5
    mat[3][11] = FAR_REAL(-7.0/2.0*nzz*nzz + 3*nzz - 3.0/10.0);  // power=5
    mat[3][12] = FAR_REAL(-9*nxx*ny*nz + (3.0/2.0)*nyy*ny*nz + (3.0/2.0)*ny*nzz*nz);  // power=5
    mat[3][13] = FAR_REAL(-21.0/2.0*nxx*nzz + (3.0/2.0)*nxx + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
    mat[3][14] = FAR_REAL(-21.0/2.0*nyy*nzz + (3.0/2.0)*nyy + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
    mat[3][15] = FAR_REAL(3*nxx*nx*ny + 3*nx*nyy*ny - 18*nx*ny*nzz);  // power=5
    mat[3][16] = FAR_REAL(-9.0/8.0*nxx*nxx*nz + (27.0/4.0)*nxx*nzz*nz - 3.0/2.0*nxx*nz - 9.0/8.0*nzz*nzz*nz + (1.0/2.0)*nzz*nz);  // power=6
    mat[3][17] = FAR_REAL(-9.0/8.0*nyy*nyy*nz + (27.0/4.0)*nyy*nzz*nz - 3.0/2.0*nyy*nz - 9.0/8.0*nzz*nzz*nz + (1.0/2.0)*nzz*nz);  // power=6
    mat[3][18] = FAR_REAL(-3*nxx*nx*ny*nz - 15.0/2.0*nx*nyy*ny*nz + (21.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[3][19] = FAR_REAL(-15.0/2.0*nxx*nx*ny*nz - 3*nx*nyy*ny*nz + (21.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[3][20] = FAR_REAL(-9.0/2.0*nxx*nx*nzz + (1.0/2.0)*nxx*nx + (9.0/2.0)*nx*nzz*nzz - 3.0/2.0*nx*nzz);  // power=6
    mat[3][21] = FAR_REAL(-9.0/2.0*nyy*ny*nzz + (1.0/2.0)*nyy*ny + (9.0/2.0)*ny*nzz*nzz - 3.0/2.0*ny*nzz);  // power=6
    mat[3][22] = FAR_REAL(-27.0/4.0*nxx*nyy*nz + (27.0/4.0)*nxx*nzz*nz - 3.0/2.0*nxx*nz + (27.0/4.0)*nyy*nzz*nz - 3.0/2.0*nyy*nz - 9.0/4.0*nzz*nzz*nz + nzz*nz);  // power=6
    mat[3][23] = FAR_REAL(-27.0/2.0*nxx*ny*nzz + (3.0/2.0)*nxx*ny + (9.0/2.0)*ny*nzz*nzz - 3.0/2.0*ny*nzz);  // power=6
    mat[3][24] = FAR_REAL(-3.0/14.0*nxx*nxx*nx + (15.0/14.0)*nxx*nx*nyy + (15.0/14.0)*nxx*nx*nzz + (9.0/7.0)*nx*nyy*nyy - 153.0/14.0*nx*nyy*nzz + (81.0/14.0)*nx*nzz*nzz - 3*nx*nzz + (3.0/14.0)*nx);  // power=6
    mat[3][25] = FAR_REAL((3.0/2.0)*nxx*nx*nz - 9*nx*nyy*nz + (3.0/2.0)*nx*nzz*nz);  // power=5
    mat[3][26] = FAR_REAL((9.0/2.0)*nxx*nx*nz + (9.0/2.0)*nx*nyy*nz - 6*nx*nzz*nz);  // power=5
    mat[3][27] = FAR_REAL((9.0/2.0)*nxx*ny*nz + (9.0/2.0)*nyy*ny*nz - 6*ny*nzz*nz);  // power=5

    // Row 4 (target order 2)
    mat[4][0] = FAR_REAL(9*nxx - 9*nyy);  // power=3
    mat[4][1] = FAR_REAL(27*nxx*nx - 63*nx*nyy - 18*nx*nzz);  // power=4
    mat[4][2] = FAR_REAL(63*nxx*ny - 27*nyy*ny + 18*ny*nzz);  // power=4
    mat[4][3] = FAR_REAL(45*nxx*nz - 45*nyy*nz);  // power=4
    mat[4][4] = FAR_REAL(35*nxx*nxx - 105.0/2.0*nxx*nyy - 35.0/2.0*nxx*nzz - 20*nxx + (35.0/2.0)*nyy*nyy + (35.0/2.0)*nyy*nzz - 10*nyy + 3);  // power=5
    mat[4][5] = FAR_REAL((35.0/2.0)*nxx*nxx - 35*nxx*nzz - 10*nxx - 35.0/2.0*nyy*nyy + 35*nyy*nzz + 10*nyy);  // power=5
    mat[4][6] = FAR_REAL(105*nxx*nx*ny - 105*nx*nyy*ny);  // power=5
    mat[4][7] = FAR_REAL(75*nxx*nx*nz - 135*nx*nyy*nz - 30*nx*nzz*nz);  // power=5
    mat[4][8] = FAR_REAL(135*nxx*ny*nz - 75*nyy*ny*nz + 30*ny*nzz*nz);  // power=5
    mat[4][9] = FAR_REAL((189.0/2.0)*nxx*nxx*nx - 189.0/2.0*nxx*nx*nyy - 189.0/2.0*nxx*nx + (63.0/2.0)*nx*nyy + 18*nx);  // power=6
    mat[4][10] = FAR_REAL((189.0/2.0)*nxx*nyy*ny - 63.0/2.0*nxx*ny - 189.0/2.0*nyy*nyy*ny + (189.0/2.0)*nyy*ny - 18*ny);  // power=6
    mat[4][11] = FAR_REAL((189.0/2.0)*nxx*nzz*nz - 63.0/2.0*nxx*nz - 189.0/2.0*nyy*nzz*nz + (63.0/2.0)*nyy*nz);  // power=6
    mat[4][12] = FAR_REAL((567.0/2.0)*nxx*nxx*ny - 567.0/2.0*nxx*nyy*ny - 189.0/2.0*nxx*ny + (63.0/2.0)*nyy*ny);  // power=6
    mat[4][13] = FAR_REAL((567.0/2.0)*nxx*nxx*nz - 567.0/2.0*nxx*nyy*nz - 315.0/2.0*nxx*nz + (63.0/2.0)*nyy*nz + 9*nz);  // power=6
    mat[4][14] = FAR_REAL((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 567.0/2.0*nyy*nyy*nz + (315.0/2.0)*nyy*nz - 9*nz);  // power=6
    mat[4][15] = FAR_REAL(567*nxx*nx*ny*nz - 567*nx*nyy*ny*nz);  // power=6
    mat[4][16] = FAR_REAL((297.0/8.0)*nxx*nxx*nxx - 297.0/8.0*nxx*nxx*nyy - 891.0/4.0*nxx*nxx*nzz - 27*nxx*nxx + (891.0/4.0)*nxx*nyy*nzz + (297.0/8.0)*nxx*nzz*nzz + 81*nxx*nzz + (9.0/4.0)*nxx - 297.0/8.0*nyy*nzz*nzz - 9.0/4.0*nyy - 27.0/4.0*nzz + 9.0/4.0);  // power=7
    mat[4][17] = FAR_REAL((297.0/8.0)*nxx*nyy*nyy - 891.0/4.0*nxx*nyy*nzz + (297.0/8.0)*nxx*nzz*nzz + (9.0/4.0)*nxx - 297.0/8.0*nyy*nyy*nyy + (891.0/4.0)*nyy*nyy*nzz + 27*nyy*nyy - 297.0/8.0*nyy*nzz*nzz - 81*nyy*nzz - 9.0/4.0*nyy + (27.0/4.0)*nzz - 9.0/4.0);  // power=7
    mat[4][18] = FAR_REAL((27.0/2.0)*nxx*nxx*nx*ny + 135*nxx*nx*nyy*ny - 540*nxx*nx*ny*nzz - 351.0/2.0*nx*nyy*nyy*ny + (621.0/2.0)*nx*nyy*ny*nzz + (189.0/2.0)*nx*nyy*ny - 108*nx*ny*nzz*nzz + (243.0/2.0)*nx*ny*nzz - 45.0/2.0*nx*ny);  // power=7
    mat[4][19] = FAR_REAL(162*nxx*nxx*nx*ny - 162*nxx*nx*nyy*ny - 540*nxx*nx*ny*nzz - 54*nxx*nx*ny - 27*nx*nyy*nyy*ny + (621.0/2.0)*nx*nyy*ny*nzz + (81.0/2.0)*nx*nyy*ny - 108*nx*ny*nzz*nzz + (243.0/2.0)*nx*ny*nzz - 9.0/2.0*nx*ny);  // power=7
    mat[4][20] = FAR_REAL(153*nxx*nxx*nx*nz - 180*nxx*nx*nyy*nz - 153*nxx*nx*nzz*nz - 81*nxx*nx*nz - 36*nx*nyy*nyy*nz + (207.0/2.0)*nx*nyy*nzz*nz + (81.0/2.0)*nx*nyy*nz - 9*nx*nzz*nzz*nz + (81.0/2.0)*nx*nzz*nz + (9.0/2.0)*nx*nz);  // power=7
    mat[4][21] = FAR_REAL((297.0/2.0)*nxx*nyy*ny*nz - 297.0/2.0*nxx*ny*nzz*nz - 297.0/2.0*nyy*nyy*ny*nz + (297.0/2.0)*nyy*ny*nzz*nz + 81*nyy*ny*nz - 27*ny*nzz*nz - 9*ny*nz);  // power=7
    mat[4][22] = FAR_REAL((891.0/4.0)*nxx*nxx*nyy - 891.0/4.0*nxx*nxx*nzz - 891.0/4.0*nxx*nyy*nyy + (297.0/4.0)*nxx*nzz*nzz + 81*nxx*nzz - 9.0/2.0*nxx + (891.0/4.0)*nyy*nyy*nzz - 297.0/4.0*nyy*nzz*nzz - 81*nyy*nzz + (9.0/2.0)*nyy);  // power=7
    mat[4][23] = FAR_REAL((891.0/2.0)*nxx*nxx*ny*nz - 891.0/2.0*nxx*nyy*ny*nz - 297.0/2.0*nxx*ny*nzz*nz - 81*nxx*ny*nz + (297.0/2.0)*nyy*ny*nzz*nz - 27*ny*nzz*nz + 9*ny*nz);  // power=7
    mat[4][24] = FAR_REAL((9.0/2.0)*nxx*nxx*nx*nz + 468*nxx*nx*nyy*nz - 153*nxx*nx*nzz*nz - 855.0/2.0*nx*nyy*nyy*nz + (315.0/2.0)*nx*nyy*nzz*nz + (135.0/2.0)*nx*nyy*nz - 9*nx*nzz*nzz*nz + (81.0/2.0)*nx*nzz*nz - 27.0/2.0*nx*nz);  // power=7
    mat[4][25] = FAR_REAL((567.0/2.0)*nxx*nx*nyy - 63.0/2.0*nxx*nx - 567.0/2.0*nx*nyy*nyy + (189.0/2.0)*nx*nyy);  // power=6
    mat[4][26] = FAR_REAL(-9.0/2.0*nxx*nxx*nx + (45.0/2.0)*nxx*nx*nyy + 306*nxx*nx*nzz - 63.0/2.0*nxx*nx + 27*nx*nyy*nyy - 459.0/2.0*nx*nyy*nzz + 27*nx*nzz*nzz - 189.0/2.0*nx*nzz + (27.0/2.0)*nx);  // power=6
    mat[4][27] = FAR_REAL((567.0/2.0)*nxx*ny*nzz - 63.0/2.0*nxx*ny - 567.0/2.0*nyy*ny*nzz + (63.0/2.0)*nyy*ny + 63*ny*nzz - 9*ny);  // power=6

    // Row 5 (target order 2)
    mat[5][0] = FAR_REAL(9*nyy - 9*nzz);  // power=3
    mat[5][1] = FAR_REAL(45*nx*nyy - 45*nx*nzz);  // power=4
    mat[5][2] = FAR_REAL(-18*nxx*ny + 27*nyy*ny - 63*ny*nzz);  // power=4
    mat[5][3] = FAR_REAL(18*nxx*nz + 63*nyy*nz - 27*nzz*nz);  // power=4
    mat[5][4] = FAR_REAL(35*nxx*nyy - 35*nxx*nzz - 35.0/2.0*nyy*nyy + 10*nyy + (35.0/2.0)*nzz*nzz - 10*nzz);  // power=5
    mat[5][5] = FAR_REAL((35.0/2.0)*nxx*nyy - 35.0/2.0*nxx*nzz + (35.0/2.0)*nyy*nyy - 105.0/2.0*nyy*nzz - 10*nyy + 35*nzz*nzz - 20*nzz + 3);  // power=5
    mat[5][6] = FAR_REAL(-30*nxx*nx*ny + 75*nx*nyy*ny - 135*nx*ny*nzz);  // power=5
    mat[5][7] = FAR_REAL(30*nxx*nx*nz + 135*nx*nyy*nz - 75*nx*nzz*nz);  // power=5
    mat[5][8] = FAR_REAL(105*nyy*ny*nz - 105*ny*nzz*nz);  // power=5
    mat[5][9] = FAR_REAL((189.0/2.0)*nxx*nx*nyy - 189.0/2.0*nxx*nx*nzz - 63.0/2.0*nx*nyy + (63.0/2.0)*nx*nzz);  // power=6
    mat[5][10] = FAR_REAL((189.0/2.0)*nyy*nyy*ny - 189.0/2.0*nyy*ny*nzz - 189.0/2.0*nyy*ny + (63.0/2.0)*ny*nzz + 18*ny);  // power=6
    mat[5][11] = FAR_REAL((189.0/2.0)*nyy*nzz*nz - 63.0/2.0*nyy*nz - 189.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 18*nz);  // power=6
    mat[5][12] = FAR_REAL((567.0/2.0)*nxx*nyy*ny - 567.0/2.0*nxx*ny*nzz - 63*nxx*ny - 63.0/2.0*nyy*ny + (63.0/2.0)*ny*nzz + 9*ny);  // power=6
    mat[5][13] = FAR_REAL((567.0/2.0)*nxx*nyy*nz - 567.0/2.0*nxx*nzz*nz + 63*nxx*nz - 63.0/2.0*nyy*nz + (63.0/2.0)*nzz*nz - 9*nz);  // power=6
    mat[5][14] = FAR_REAL((567.0/2.0)*nyy*nyy*nz - 567.0/2.0*nyy*nzz*nz - 189.0/2.0*nyy*nz + (63.0/2.0)*nzz*nz);  // power=6
    mat[5][15] = FAR_REAL(567*nx*nyy*ny*nz - 567*nx*ny*nzz*nz);  // power=6
    mat[5][16] = FAR_REAL((297.0/8.0)*nxx*nxx*nyy - 297.0/8.0*nxx*nxx*nzz - 891.0/4.0*nxx*nyy*nzz + (891.0/4.0)*nxx*nzz*nzz - 81*nxx*nzz + (27.0/4.0)*nxx + (297.0/8.0)*nyy*nzz*nzz + (9.0/4.0)*nyy - 297.0/8.0*nzz*nzz*nzz + 27*nzz*nzz - 9.0/4.0*nzz - 9.0/4.0);  // power=7
    mat[5][17] = FAR_REAL((297.0/8.0)*nyy*nyy*nyy - 2079.0/8.0*nyy*nyy*nzz - 27*nyy*nyy + (2079.0/8.0)*nyy*nzz*nzz + 9*nyy - 297.0/8.0*nzz*nzz*nzz + 27*nzz*nzz - 9*nzz);  // power=7
    mat[5][18] = FAR_REAL(-18*nxx*nxx*nx*ny + 18*nxx*nx*nyy*ny + 126*nxx*nx*ny*nzz + (369.0/2.0)*nx*nyy*nyy*ny - 414*nx*nyy*ny*nzz - 135*nx*nyy*ny + (1179.0/2.0)*nx*ny*nzz*nzz - 243*nx*ny*nzz + 36*nx*ny);  // power=7
    mat[5][19] = FAR_REAL(-27.0/2.0*nxx*nxx*nx*ny + 162*nxx*nx*nyy*ny - 54*nxx*nx*ny*nzz - 27*nxx*nx*ny + 27*nx*nyy*nyy*ny - 621.0/2.0*nx*nyy*ny*nzz - 81.0/2.0*nx*nyy*ny + (1107.0/2.0)*nx*ny*nzz*nzz - 405.0/2.0*nx*ny*nzz + (45.0/2.0)*nx*ny);  // power=7
    mat[5][20] = FAR_REAL(-9.0/2.0*nxx*nxx*nx*nz + 180*nxx*nx*nyy*nz - 144*nxx*nx*nzz*nz + 27*nxx*nx*nz + 36*nx*nyy*nyy*nz - 207.0/2.0*nx*nyy*nzz*nz - 81.0/2.0*nx*nyy*nz + (315.0/2.0)*nx*nzz*nzz*nz - 189.0/2.0*nx*nzz*nz + (27.0/2.0)*nx*nz);  // power=7
    mat[5][21] = FAR_REAL((297.0/2.0)*nyy*nyy*ny*nz - 297*nyy*ny*nzz*nz - 54*nyy*ny*nz + (297.0/2.0)*ny*nzz*nzz*nz - 54*ny*nzz*nz + 18*ny*nz);  // power=7
    mat[5][22] = FAR_REAL((891.0/4.0)*nxx*nyy*nyy - 891.0/2.0*nxx*nyy*nzz - 81*nxx*nyy + (891.0/4.0)*nxx*nzz*nzz - 81*nxx*nzz + (27.0/2.0)*nxx - 891.0/4.0*nyy*nyy*nzz + 297*nyy*nzz*nzz + 9*nyy - 297.0/4.0*nzz*nzz*nzz + 54*nzz*nzz - 9*nzz - 9.0/2.0);  // power=7
    mat[5][23] = FAR_REAL((891.0/2.0)*nxx*nyy*ny*nz - 891.0/2.0*nxx*ny*nzz*nz - 297.0/2.0*nyy*ny*nzz*nz + (297.0/2.0)*ny*nzz*nzz*nz - 54*ny*nzz*nz + 9*ny*nz);  // power=7
    mat[5][24] = FAR_REAL(-18*nxx*nxx*nx*nz + 72*nxx*nx*nyy*nz + 18*nxx*nx*nzz*nz + (1071.0/2.0)*nx*nyy*nyy*nz - 468*nx*nyy*nzz*nz - 189*nx*nyy*nz + (369.0/2.0)*nx*nzz*nzz*nz - 135*nx*nzz*nz + 36*nx*nz);  // power=7
    mat[5][25] = FAR_REAL(-9.0/2.0*nxx*nxx*nx + (45.0/2.0)*nxx*nx*nyy + (45.0/2.0)*nxx*nx*nzz + (621.0/2.0)*nx*nyy*nyy - 459.0/2.0*nx*nyy*nzz - 189*nx*nyy + 27*nx*nzz*nzz + (27.0/2.0)*nx);  // power=6
    mat[5][26] = FAR_REAL((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nx*nyy - 45.0/2.0*nxx*nx*nzz - 27*nx*nyy*nyy + (459.0/2.0)*nx*nyy*nzz - 621.0/2.0*nx*nzz*nzz + 189*nx*nzz - 27.0/2.0*nx);  // power=6
    mat[5][27] = FAR_REAL((567.0/2.0)*nyy*ny*nzz - 63.0/2.0*nyy*ny - 567.0/2.0*ny*nzz*nzz + (189.0/2.0)*ny*nzz);  // power=6

    // Row 6 (target order 2)
    mat[6][0] = FAR_REAL(9*nx*ny);  // power=3
    mat[6][1] = FAR_REAL(36*nxx*ny - 9*nyy*ny - 9*ny*nzz);  // power=4
    mat[6][2] = FAR_REAL(-9*nxx*nx + 36*nx*nyy - 9*nx*nzz);  // power=4
    mat[6][3] = FAR_REAL(45*nx*ny*nz);  // power=4
    mat[6][4] = FAR_REAL(30*nxx*nx*ny - 45.0/2.0*nx*nyy*ny - 45.0/2.0*nx*ny*nzz);  // power=5
    mat[6][5] = FAR_REAL((15.0/2.0)*nxx*nx*ny + (15.0/2.0)*nx*nyy*ny - 45*nx*ny*nzz);  // power=5
    mat[6][6] = FAR_REAL(105*nxx*nyy - 15*nxx - 15*nyy + 3);  // power=5
    mat[6][7] = FAR_REAL(90*nxx*ny*nz - 15*nyy*ny*nz - 15*ny*nzz*nz);  // power=5
    mat[6][8] = FAR_REAL(-15*nxx*nx*nz + 90*nx*nyy*nz - 15*nx*nzz*nz);  // power=5
    mat[6][9] = FAR_REAL((189.0/2.0)*nxx*nxx*ny - 63*nxx*ny + (9.0/2.0)*ny);  // power=6
    mat[6][10] = FAR_REAL((189.0/2.0)*nx*nyy*nyy - 63*nx*nyy + (9.0/2.0)*nx);  // power=6
    mat[6][11] = FAR_REAL(-63.0/2.0*nxx*nx*ny*nz - 63.0/2.0*nx*nyy*ny*nz + 63*nx*ny*nzz*nz);  // power=6
    mat[6][12] = FAR_REAL((567.0/2.0)*nxx*nx*nyy - 63.0/2.0*nxx*nx - 189.0/2.0*nx*nyy + (27.0/2.0)*nx);  // power=6
    mat[6][13] = FAR_REAL(189*nxx*nx*ny*nz - 189.0/2.0*nx*nyy*ny*nz - 189.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[6][14] = FAR_REAL(-189.0/2.0*nxx*nx*ny*nz + 189*nx*nyy*ny*nz - 189.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[6][15] = FAR_REAL(567*nxx*nyy*nz - 63*nxx*nz - 63*nyy*nz + 9*nz);  // power=6
    mat[6][16] = FAR_REAL((297.0/8.0)*nxx*nxx*nx*ny - 891.0/4.0*nxx*nx*ny*nzz - 27.0/2.0*nxx*nx*ny + (297.0/8.0)*nx*ny*nzz*nzz + (81.0/2.0)*nx*ny*nzz);  // power=7
    mat[6][17] = FAR_REAL(-27.0/4.0*nxx*nxx*nx*ny + (27.0/4.0)*nxx*nx*nyy*ny + (189.0/4.0)*nxx*nx*ny*nzz + (405.0/8.0)*nx*nyy*nyy*ny - 621.0/4.0*nx*nyy*ny*nzz - 135.0/4.0*nx*nyy*ny + (729.0/8.0)*nx*ny*nzz*nzz - 81.0/4.0*nx*ny*nzz + (27.0/4.0)*nx*ny);  // power=7
    mat[6][18] = FAR_REAL((297.0/2.0)*nxx*nyy*nyy - 891.0/2.0*nxx*nyy*nzz - 81.0/2.0*nxx*nyy + (81.0/2.0)*nxx*nzz + (9.0/2.0)*nxx - 27.0/2.0*nyy*nyy + (81.0/2.0)*nyy*nzz + 9*nyy - 9.0/2.0);  // power=7
    mat[6][19] = FAR_REAL((297.0/2.0)*nxx*nxx*nyy - 27.0/2.0*nxx*nxx - 891.0/2.0*nxx*nyy*nzz - 81.0/2.0*nxx*nyy + (81.0/2.0)*nxx*nzz + 9*nxx + (81.0/2.0)*nyy*nzz + (9.0/2.0)*nyy - 9.0/2.0);  // power=7
    mat[6][20] = FAR_REAL((297.0/2.0)*nxx*nxx*ny*nz - 297.0/2.0*nxx*ny*nzz*nz - 81.0/2.0*nxx*ny*nz + (27.0/2.0)*ny*nzz*nz);  // power=7
    mat[6][21] = FAR_REAL(-9.0/2.0*nxx*nxx*nx*nz + (27.0/2.0)*nxx*nx*nyy*nz + (9.0/2.0)*nxx*nx*nzz*nz + (333.0/2.0)*nx*nyy*nyy*nz - 243.0/2.0*nx*nyy*nzz*nz - 63*nx*nyy*nz + 9*nx*nzz*nzz*nz + (9.0/2.0)*nx*nz);  // power=7
    mat[6][22] = FAR_REAL(-27.0/4.0*nxx*nxx*nx*ny + (459.0/2.0)*nxx*nx*nyy*ny - 351.0/2.0*nxx*nx*ny*nzz - 81.0/2.0*nxx*nx*ny + (27.0/2.0)*nx*nyy*nyy*ny - 621.0/4.0*nx*nyy*ny*nzz - 243.0/4.0*nx*nyy*ny + (513.0/4.0)*nx*ny*nzz*nzz + (81.0/4.0)*nx*ny*nzz + (63.0/4.0)*nx*ny);  // power=7
    mat[6][23] = FAR_REAL(-9.0/2.0*nxx*nxx*nx*nz + 477*nxx*nx*nyy*nz + (9.0/2.0)*nxx*nx*nzz*nz - 81.0/2.0*nxx*nx*nz + 36*nx*nyy*nyy*nz - 207.0/2.0*nx*nyy*nzz*nz - 243.0/2.0*nx*nyy*nz + 9*nx*nzz*nzz*nz + (27.0/2.0)*nx*nz);  // power=7
    mat[6][24] = FAR_REAL((891.0/2.0)*nxx*nyy*ny*nz - 297.0/2.0*nxx*ny*nzz*nz - 81*nxx*ny*nz - 81.0/2.0*nyy*ny*nz + (27.0/2.0)*ny*nzz*nz + 9*ny*nz);  // power=7
    mat[6][25] = FAR_REAL((567.0/2.0)*nxx*nyy*ny - 189.0/2.0*nxx*ny - 63.0/2.0*nyy*ny + (27.0/2.0)*ny);  // power=6
    mat[6][26] = FAR_REAL((567.0/2.0)*nxx*ny*nzz - 63.0/2.0*nxx*ny - 63.0/2.0*ny*nzz + (9.0/2.0)*ny);  // power=6
    mat[6][27] = FAR_REAL((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nx*nyy - 45.0/2.0*nxx*nx*nzz - 27*nx*nyy*nyy + (459.0/2.0)*nx*nyy*nzz - 27*nx*nzz*nzz);  // power=6

    // Row 7 (target order 2)
    mat[7][0] = FAR_REAL(9*nx*nz);  // power=3
    mat[7][1] = FAR_REAL(36*nxx*nz - 9*nyy*nz - 9*nzz*nz);  // power=4
    mat[7][2] = FAR_REAL(45*nx*ny*nz);  // power=4
    mat[7][3] = FAR_REAL(-9*nxx*nx - 9*nx*nyy + 36*nx*nzz);  // power=4
    mat[7][4] = FAR_REAL(30*nxx*nx*nz - 45.0/2.0*nx*nyy*nz - 45.0/2.0*nx*nzz*nz);  // power=5
    mat[7][5] = FAR_REAL((45.0/2.0)*nxx*nx*nz + (45.0/2.0)*nx*nyy*nz - 30*nx*nzz*nz);  // power=5
    mat[7][6] = FAR_REAL(90*nxx*ny*nz - 15*nyy*ny*nz - 15*ny*nzz*nz);  // power=5
    mat[7][7] = FAR_REAL(105*nxx*nzz - 15*nxx - 15*nzz + 3);  // power=5
    mat[7][8] = FAR_REAL(-15*nxx*nx*ny - 15*nx*nyy*ny + 90*nx*ny*nzz);  // power=5
    mat[7][9] = FAR_REAL((189.0/2.0)*nxx*nxx*nz - 63*nxx*nz + (9.0/2.0)*nz);  // power=6
    mat[7][10] = FAR_REAL(-63.0/2.0*nxx*nx*ny*nz + 63*nx*nyy*ny*nz - 63.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[7][11] = FAR_REAL((189.0/2.0)*nx*nzz*nzz - 63*nx*nzz + (9.0/2.0)*nx);  // power=6
    mat[7][12] = FAR_REAL(189*nxx*nx*ny*nz - 189.0/2.0*nx*nyy*ny*nz - 189.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[7][13] = FAR_REAL((567.0/2.0)*nxx*nx*nzz - 63.0/2.0*nxx*nx - 189.0/2.0*nx*nzz + (27.0/2.0)*nx);  // power=6
    mat[7][14] = FAR_REAL((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nx*nyy - 45.0/2.0*nxx*nx*nzz - 27*nx*nyy*nyy + (459.0/2.0)*nx*nyy*nzz - 27*nx*nzz*nzz);  // power=6
    mat[7][15] = FAR_REAL(567*nxx*ny*nzz - 63*nxx*ny - 63*ny*nzz + 9*ny);  // power=6
    mat[7][16] = FAR_REAL((297.0/8.0)*nxx*nxx*nx*nz - 891.0/4.0*nxx*nx*nzz*nz + 27*nxx*nx*nz + (297.0/8.0)*nx*nzz*nzz*nz + 27*nx*nzz*nz - 9*nx*nz);  // power=7
    mat[7][17] = FAR_REAL(-27.0/4.0*nxx*nxx*nx*nz + (171.0/4.0)*nxx*nx*nyy*nz + (27.0/4.0)*nxx*nx*nzz*nz + (693.0/8.0)*nx*nyy*nyy*nz - 639.0/4.0*nx*nyy*nzz*nz - 63.0/4.0*nx*nyy*nz + (405.0/8.0)*nx*nzz*nzz*nz - 135.0/4.0*nx*nzz*nz + (27.0/4.0)*nx*nz);  // power=7
    mat[7][18] = FAR_REAL((297.0/2.0)*nxx*nyy*ny*nz - 891.0/2.0*nxx*ny*nzz*nz + 81*nxx*ny*nz - 27.0/2.0*nyy*ny*nz + (81.0/2.0)*ny*nzz*nz - 9*ny*nz);  // power=7
    mat[7][19] = FAR_REAL((297.0/2.0)*nxx*nxx*ny*nz - 891.0/2.0*nxx*ny*nzz*nz + (81.0/2.0)*nxx*ny*nz + (81.0/2.0)*ny*nzz*nz - 9*ny*nz);  // power=7
    mat[7][20] = FAR_REAL((297.0/2.0)*nxx*nxx*nzz - 27.0/2.0*nxx*nxx - 297.0/2.0*nxx*nzz*nzz + (9.0/2.0)*nxx + (27.0/2.0)*nzz*nzz - 9.0/2.0*nzz);  // power=7
    mat[7][21] = FAR_REAL((9.0/2.0)*nxx*nxx*nx*ny - 9.0/2.0*nxx*nx*nyy*ny - 63.0/2.0*nxx*nx*ny*nzz - 9*nx*nyy*nyy*ny + (207.0/2.0)*nx*nyy*ny*nzz - 369.0/2.0*nx*ny*nzz*nzz + 81*nx*ny*nzz - 9.0/2.0*nx*ny);  // power=7
    mat[7][22] = FAR_REAL(-27.0/4.0*nxx*nxx*nx*nz + 270*nxx*nx*nyy*nz - 216*nxx*nx*nzz*nz + (81.0/2.0)*nxx*nx*nz + 54*nx*nyy*nyy*nz - 621.0/4.0*nx*nyy*nzz*nz - 243.0/4.0*nx*nyy*nz + (351.0/4.0)*nx*nzz*nzz*nz - 27.0/4.0*nx*nzz*nz - 9.0/4.0*nx*nz);  // power=7
    mat[7][23] = FAR_REAL((891.0/2.0)*nxx*nx*ny*nzz - 81.0/2.0*nxx*nx*ny - 297.0/2.0*nx*ny*nzz*nzz - 81.0/2.0*nx*ny*nzz + 9*nx*ny);  // power=7
    mat[7][24] = FAR_REAL((891.0/2.0)*nxx*nyy*nzz - 81.0/2.0*nxx*nyy - 297.0/2.0*nxx*nzz*nzz + (81.0/2.0)*nxx*nzz - 9.0/2.0*nxx - 81.0/2.0*nyy*nzz + (27.0/2.0)*nzz*nzz - 9*nzz + 9.0/2.0);  // power=7
    mat[7][25] = FAR_REAL((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 63.0/2.0*nyy*nz + (9.0/2.0)*nz);  // power=6
    mat[7][26] = FAR_REAL((567.0/2.0)*nxx*nzz*nz - 189.0/2.0*nxx*nz - 63.0/2.0*nzz*nz + (27.0/2.0)*nz);  // power=6
    mat[7][27] = FAR_REAL(-189.0/2.0*nxx*nx*ny*nz - 189.0/2.0*nx*nyy*ny*nz + 189*nx*ny*nzz*nz);  // power=6

    // Row 8 (target order 2)
    mat[8][0] = FAR_REAL(9*ny*nz);  // power=3
    mat[8][1] = FAR_REAL(45*nx*ny*nz);  // power=4
    mat[8][2] = FAR_REAL(-9*nxx*nz + 36*nyy*nz - 9*nzz*nz);  // power=4
    mat[8][3] = FAR_REAL(-9*nxx*ny - 9*nyy*ny + 36*ny*nzz);  // power=4
    mat[8][4] = FAR_REAL(45*nxx*ny*nz - 15.0/2.0*nyy*ny*nz - 15.0/2.0*ny*nzz*nz);  // power=5
    mat[8][5] = FAR_REAL((45.0/2.0)*nxx*ny*nz + (45.0/2.0)*nyy*ny*nz - 30*ny*nzz*nz);  // power=5
    mat[8][6] = FAR_REAL(-15*nxx*nx*nz + 90*nx*nyy*nz - 15*nx*nzz*nz);  // power=5
    mat[8][7] = FAR_REAL(-15*nxx*nx*ny - 15*nx*nyy*ny + 90*nx*ny*nzz);  // power=5
    mat[8][8] = FAR_REAL(105*nyy*nzz - 15*nyy - 15*nzz + 3);  // power=5
    mat[8][9] = FAR_REAL(63*nxx*nx*ny*nz - 63.0/2.0*nx*nyy*ny*nz - 63.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[8][10] = FAR_REAL((189.0/2.0)*nyy*nyy*nz - 63*nyy*nz + (9.0/2.0)*nz);  // power=6
    mat[8][11] = FAR_REAL((189.0/2.0)*ny*nzz*nzz - 63*ny*nzz + (9.0/2.0)*ny);  // power=6
    mat[8][12] = FAR_REAL((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 63.0/2.0*nyy*nz + (9.0/2.0)*nz);  // power=6
    mat[8][13] = FAR_REAL((567.0/2.0)*nxx*ny*nzz - 63.0/2.0*nxx*ny - 63.0/2.0*ny*nzz + (9.0/2.0)*ny);  // power=6
    mat[8][14] = FAR_REAL((567.0/2.0)*nyy*ny*nzz - 63.0/2.0*nyy*ny - 189.0/2.0*ny*nzz + (27.0/2.0)*ny);  // power=6
    mat[8][15] = FAR_REAL(9*nxx*nxx*nx - 45*nxx*nx*nyy - 45*nxx*nx*nzz - 54*nx*nyy*nyy + 459*nx*nyy*nzz - 54*nx*nzz*nzz);  // power=6
    mat[8][16] = FAR_REAL((297.0/8.0)*nxx*nxx*ny*nz - 891.0/4.0*nxx*ny*nzz*nz + (81.0/2.0)*nxx*ny*nz + (297.0/8.0)*ny*nzz*nzz*nz - 27.0/2.0*ny*nzz*nz);  // power=7
    mat[8][17] = FAR_REAL((297.0/8.0)*nyy*nyy*ny*nz - 891.0/4.0*nyy*ny*nzz*nz + 27*nyy*ny*nz + (297.0/8.0)*ny*nzz*nzz*nz + 27*ny*nzz*nz - 9*ny*nz);  // power=7
    mat[8][18] = FAR_REAL(-27.0/2.0*nxx*nxx*nx*nz + (153.0/2.0)*nxx*nx*nyy*nz + (27.0/2.0)*nxx*nx*nzz*nz + (477.0/2.0)*nx*nyy*nyy*nz - 657.0/2.0*nx*nyy*nzz*nz - 63*nx*nyy*nz + 27*nx*nzz*nzz*nz + (9.0/2.0)*nx*nz);  // power=7
    mat[8][19] = FAR_REAL(-27.0/2.0*nxx*nxx*nx*nz + 243*nxx*nx*nyy*nz + (27.0/2.0)*nxx*nx*nzz*nz - 27.0/2.0*nxx*nx*nz + 108*nx*nyy*nyy*nz - 621.0/2.0*nx*nyy*nzz*nz - 81.0/2.0*nx*nyy*nz + 27*nx*nzz*nzz*nz + (9.0/2.0)*nx*nz);  // power=7
    mat[8][20] = FAR_REAL((297.0/2.0)*nxx*nx*ny*nzz - 27.0/2.0*nxx*nx*ny - 297.0/2.0*nx*ny*nzz*nzz + (81.0/2.0)*nx*ny*nzz);  // power=7
    mat[8][21] = FAR_REAL((297.0/2.0)*nyy*nyy*nzz - 27.0/2.0*nyy*nyy - 297.0/2.0*nyy*nzz*nzz + (9.0/2.0)*nyy + (27.0/2.0)*nzz*nzz - 9.0/2.0*nzz);  // power=7
    mat[8][22] = FAR_REAL((891.0/4.0)*nxx*nyy*ny*nz - 891.0/4.0*nxx*ny*nzz*nz - 891.0/4.0*nyy*ny*nzz*nz + (81.0/2.0)*nyy*ny*nz + (297.0/4.0)*ny*nzz*nzz*nz + (27.0/2.0)*ny*nzz*nz - 9*ny*nz);  // power=7
    mat[8][23] = FAR_REAL((891.0/2.0)*nxx*nyy*nzz - 81.0/2.0*nxx*nyy - 81.0/2.0*nxx*nzz - 297.0/2.0*nyy*nzz*nzz + (81.0/2.0)*nyy*nzz - 9.0/2.0*nyy + (27.0/2.0)*nzz*nzz - 9*nzz + 9.0/2.0);  // power=7
    mat[8][24] = FAR_REAL((27.0/2.0)*nxx*nxx*nx*ny - 27.0/2.0*nxx*nx*nyy*ny - 189.0/2.0*nxx*nx*ny*nzz - 27*nx*nyy*nyy*ny + (621.0/2.0)*nx*nyy*ny*nzz - 513.0/2.0*nx*ny*nzz*nzz + 81*nx*ny*nzz - 9.0/2.0*nx*ny);  // power=7
    mat[8][25] = FAR_REAL(-189.0/2.0*nxx*nx*ny*nz + 189*nx*nyy*ny*nz - 189.0/2.0*nx*ny*nzz*nz);  // power=6
    mat[8][26] = FAR_REAL(-189.0/2.0*nxx*nx*ny*nz - 189.0/2.0*nx*nyy*ny*nz + 189*nx*ny*nzz*nz);  // power=6
    mat[8][27] = FAR_REAL((567.0/2.0)*nyy*nzz*nz - 189.0/2.0*nyy*nz - 63.0/2.0*nzz*nz + (27.0/2.0)*nz);  // power=6

    // Row 9 (target order 3)
    mat[9][0] = FAR_REAL(-30*nxx*nx + 45*nx*nyy + 45*nx*nzz);  // power=4
    mat[9][1] = FAR_REAL(-210*nxx*nxx + 315*nxx*nyy + 315*nxx*nzz + 90*nxx - 45*nyy - 45*nzz);  // power=5
    mat[9][2] = FAR_REAL(-300*nxx*nx*ny + 225*nx*nyy*ny + 225*nx*ny*nzz);  // power=5
    mat[9][3] = FAR_REAL(-300*nxx*nx*nz + 225*nx*nyy*nz + 225*nx*nzz*nz);  // power=5
    mat[9][4] = FAR_REAL(-215*nxx*nxx*nx + 445*nxx*nx*nyy + 445*nxx*nx*nzz + 140*nxx*nx - 255.0/2.0*nx*nyy*nyy - 255*nx*nyy*nzz - 35*nx*nyy - 255.0/2.0*nx*nzz*nzz - 35*nx*nzz - 25*nx);  // power=6
    mat[9][5] = FAR_REAL(-215.0/2.0*nxx*nxx*nx + 65*nxx*nx*nyy + 380*nxx*nx*nzz + 70*nxx*nx + (345.0/2.0)*nx*nyy*nyy - 255.0/2.0*nx*nyy*nzz - 245.0/2.0*nx*nyy - 300*nx*nzz*nzz + (175.0/2.0)*nx*nzz - 25.0/2.0*nx);  // power=6
    mat[9][6] = FAR_REAL(-630*nxx*nxx*ny + 945*nxx*nyy*ny + 945*nxx*ny*nzz - 105*nyy*ny - 105*ny*nzz + 30*ny);  // power=6
    mat[9][7] = FAR_REAL(-630*nxx*nxx*nz + 945*nxx*nyy*nz + 945*nxx*nzz*nz - 105*nyy*nz - 105*nzz*nz + 30*nz);  // power=6
    mat[9][8] = FAR_REAL(-1050*nxx*nx*ny*nz + 525*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[9][9] = FAR_REAL(-693*nxx*nxx*nxx + (2079.0/2.0)*nxx*nxx*nyy + (2079.0/2.0)*nxx*nxx*nzz + 756*nxx*nxx - 567*nxx*nyy - 567*nxx*nzz - 189*nxx + (63.0/2.0)*nyy + (63.0/2.0)*nzz + 6);  // power=7
    mat[9][10] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*ny - 1449.0/2.0*nxx*nx*nyy*ny - 441.0/2.0*nxx*nx*ny*nzz + 189*nxx*nx*ny + (1953.0/2.0)*nx*nyy*nyy*ny + (1449.0/2.0)*nx*nyy*ny*nzz - 756*nx*nyy*ny - 252*nx*ny*nzz*nzz + (189.0/2.0)*nx*ny);  // power=7
    mat[9][11] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*nz - 441.0/2.0*nxx*nx*nyy*nz - 1449.0/2.0*nxx*nx*nzz*nz + 189*nxx*nx*nz - 252*nx*nyy*nyy*nz + (1449.0/2.0)*nx*nyy*nzz*nz + (1953.0/2.0)*nx*nzz*nzz*nz - 756*nx*nzz*nz + (189.0/2.0)*nx*nz);  // power=7
    mat[9][12] = FAR_REAL(-2079*nxx*nxx*nx*ny + (6237.0/2.0)*nxx*nx*nyy*ny + (6237.0/2.0)*nxx*nx*ny*nzz + 756*nxx*nx*ny - 1701.0/2.0*nx*nyy*ny - 1701.0/2.0*nx*ny*nzz + 63*nx*ny);  // power=7
    mat[9][13] = FAR_REAL(-2079*nxx*nxx*nx*nz + (6237.0/2.0)*nxx*nx*nyy*nz + (6237.0/2.0)*nxx*nx*nzz*nz + 756*nxx*nx*nz - 1701.0/2.0*nx*nyy*nz - 1701.0/2.0*nx*nzz*nz + 63*nx*nz);  // power=7
    mat[9][14] = FAR_REAL((189.0/2.0)*nxx*nxx*nx*nz - 6237.0/2.0*nxx*nx*nyy*nz - 189.0/2.0*nxx*nx*nzz*nz + 189*nxx*nx*nz + (3969.0/2.0)*nx*nyy*nyy*nz + (3591.0/2.0)*nx*nyy*nzz*nz - 756*nx*nyy*nz - 189*nx*nzz*nzz*nz + (63.0/2.0)*nx*nz);  // power=7
    mat[9][15] = FAR_REAL(-4158*nxx*nxx*ny*nz + 6237*nxx*nyy*ny*nz + 6237*nxx*ny*nzz*nz - 1134000000000001.0/1000000000000.0*nxx*ny*nz - 567*nyy*ny*nz - 567*ny*nzz*nz + 252*ny*nz);  // power=7
    mat[9][16] = FAR_REAL(-1287.0/4.0*nxx*nxx*nxx*nx - 9.0/8.0*nxx*nxx*nxx*nx + (482624999998552.0/999999999997.0)*nxx*nxx*nx*nyy + (27.0/4.0)*nxx*nxx*nx*nyy + (19305.0/8.0)*nxx*nxx*nx*nzz + (135.0/8.0)*nxx*nxx*nx*nzz + 297*nxx*nxx*nx + (135.0/8.0)*nxx*nx*nyy*nyy - 11583.0/4.0*nxx*nx*nyy*nzz - 675.0/4.0*nxx*nx*nyy*nzz - 297.0/2.0*nxx*nx*nyy - 6435.0/2.0*nxx*nx*nzz*nzz - 148499999999851.0/999999999999.0*nxx*nx*nzz - 243.0/2.0*nxx*nx + 9*nx*nyy*nyy*nyy - 1485.0/8.0*nx*nyy*nyy*nzz + 270*nx*nyy*nzz*nzz + (2673.0/4.0)*nx*nyy*nzz - 81.0/8.0*nx*nyy + (482624999998552.0/999999999997.0)*nx*nzz*nzz*nzz - 18*nx*nzz*nzz*nzz + (1485.0/8.0)*nx*nzz*nzz - 243.0/4.0*nx*nzz + (129.0/8.0)*nx);  // power=8
    mat[9][17] = FAR_REAL((45.0/8.0)*nxx*nxx*nxx*nx - 135.0/4.0*nxx*nxx*nx*nyy - 675.0/8.0*nxx*nxx*nx*nzz - 1629.0/8.0*nxx*nx*nyy*nyy - 675.0/8.0*nxx*nx*nyy*nyy + (4491.0/2.0)*nxx*nx*nyy*nzz + (3375.0/4.0)*nxx*nx*nyy*nzz - 315.0/4.0*nxx*nx*nyy - 1287.0/4.0*nxx*nx*nzz*nzz + (45.0/8.0)*nxx*nx + (2403.0/4.0)*nx*nyy*nyy*nyy - 45*nx*nyy*nyy*nyy - 1980*nx*nyy*nyy*nzz + (7425.0/8.0)*nx*nyy*nyy*nzz - 493874999999506.0/999999999999.0*nx*nyy*nyy + 315*nx*nyy*nzz*nzz - 1350*nx*nyy*nzz*nzz + (549.0/2.0)*nx*nyy*nzz + 135*nx*nyy + (482624999998552.0/999999999997.0)*nx*nzz*nzz*nzz + 90*nx*nzz*nzz*nzz - 482624999998552.0/999999999997.0*nx*nzz*nzz + (855.0/8.0)*nx*nzz - 45.0/4.0*nx);  // power=8
    mat[9][18] = FAR_REAL(-1287*nxx*nxx*nyy*ny + 3861*nxx*nxx*ny*nzz + (1930499999998069.0/999999999999.0)*nxx*nyy*nyy*ny - 3861*nxx*nyy*ny*nzz - 594*nxx*nyy*ny - 11583.0/2.0*nxx*ny*nzz*nzz + 1782*nxx*ny*nzz - 297.0/2.0*nyy*nyy*ny + 297*nyy*ny*nzz + 81*nyy*ny + (891.0/2.0)*ny*nzz*nzz - 243*ny*nzz);  // power=8
    mat[9][19] = FAR_REAL(-1287*nxx*nxx*nxx*ny + (1930499999998069.0/999999999999.0)*nxx*nxx*nyy*ny + (11583.0/2.0)*nxx*nxx*ny*nzz + 594*nxx*nxx*ny - 11583.0/2.0*nxx*nyy*ny*nzz - 891.0/2.0*nxx*nyy*ny - 11583.0/2.0*nxx*ny*nzz*nzz + (2673.0/2.0)*nxx*ny*nzz - 162*nxx*ny + (891.0/2.0)*nyy*ny*nzz + (891.0/2.0)*ny*nzz*nzz - 243*ny*nzz + 15*ny);  // power=8
    mat[9][20] = FAR_REAL(-1287*nxx*nxx*nxx*nz + (1930499999998069.0/999999999999.0)*nxx*nxx*nyy*nz + (3217499999996782.0/999999999999.0)*nxx*nxx*nzz*nz + 594*nxx*nxx*nz - 1930499999998069.0/999999999999.0*nxx*nyy*nzz*nz - 891.0/2.0*nxx*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nzz*nzz*nz + (297.0/2.0)*nxx*nzz*nz - 162*nxx*nz + (297.0/2.0)*nyy*nzz*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz + 15*nz);  // power=8
    mat[9][21] = FAR_REAL(-1557*nxx*nx*nyy*ny*nz + 1287*nxx*nx*ny*nzz*nz + (3321.0/2.0)*nx*nyy*nyy*ny*nz - 270*nx*nyy*ny*nzz*nz - 918*nx*nyy*ny*nz - 1930499999998069.0/999999999999.0*nx*ny*nzz*nzz*nz + 1188*nx*ny*nzz*nz);  // power=8
    mat[9][22] = FAR_REAL((9.0/2.0)*nxx*nxx*nxx*nx - 3861.0/2.0*nxx*nxx*nx*nyy - 27*nxx*nxx*nx*nyy + (3861.0/2.0)*nxx*nxx*nx*nzz - 135.0/2.0*nxx*nxx*nx*nzz + (6075.0/2.0)*nxx*nx*nyy*nyy - 135.0/2.0*nxx*nx*nyy*nyy + (4617.0/2.0)*nxx*nx*nyy*nzz + 675*nxx*nx*nyy*nzz - 94499999999906.0/999999999999.0*nxx*nx*nyy - 1945723264306643.0/549755813889.0*nxx*nx*nzz*nzz + (27.0/4.0)*nxx*nx + (567.0/4.0)*nx*nyy*nyy*nyy - 36*nx*nyy*nyy*nyy - 2376*nx*nyy*nyy*nzz + (1485.0/2.0)*nx*nyy*nyy*nzz - 2727.0/4.0*nx*nyy*nyy + 378*nx*nyy*nzz*nzz - 1080*nx*nyy*nzz*nzz + (837.0/2.0)*nx*nyy*nzz + (729.0/4.0)*nx*nyy + (965249999997104.0/999999999997.0)*nx*nzz*nzz*nzz + 72*nx*nzz*nzz*nzz - 297*nx*nzz*nzz + (189.0/4.0)*nx*nzz - 45.0/4.0*nx);  // power=8
    mat[9][23] = FAR_REAL(-7965.0/2.0*nxx*nxx*nx*ny*nz + 5994*nxx*nx*nyy*ny*nz + (7280999999999999.0/1000000000000.0)*nxx*nx*ny*nzz*nz + 324*nx*nyy*nyy*ny*nz - 2565.0/2.0*nx*nyy*ny*nzz*nz - 2673.0/2.0*nx*nyy*ny*nz - 3213.0/2.0*nx*ny*nzz*nzz*nz - 297.0/2.0*nx*ny*nzz*nz + (405.0/2.0)*nx*ny*nz);  // power=8
    mat[9][24] = FAR_REAL(-3861*nxx*nxx*nyy*nz + 1287*nxx*nxx*nzz*nz + (11583.0/2.0)*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 1782*nxx*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nzz*nzz*nz + 594*nxx*nzz*nz - 891.0/2.0*nyy*nyy*nz - 297*nyy*nzz*nz + 243*nyy*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz);  // power=8
    mat[9][25] = FAR_REAL(-2079*nxx*nxx*nyy + 189*nxx*nxx + (6237.0/2.0)*nxx*nyy*nyy + (6237.0/2.0)*nxx*nyy*nzz - 1701.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz - 63.0/2.0*nxx - 567.0/2.0*nyy*nyy - 567.0/2.0*nyy*nzz + 126*nyy + 45.0/2.0);  // power=7
    mat[9][26] = FAR_REAL(-2079*nxx*nxx*nzz + 189*nxx*nxx + (6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy + (6237.0/2.0)*nxx*nzz*nzz - 1701.0/2.0*nxx*nzz - 63.0/2.0*nxx - 567.0/2.0*nyy*nzz - 567.0/2.0*nzz*nzz + 126*nzz + 45.0/2.0);  // power=7
    mat[9][27] = FAR_REAL((189.0/2.0)*nxx*nxx*nx*ny - 189.0/2.0*nxx*nx*nyy*ny - 2740499999997259.0/999999999999.0*nxx*nx*ny*nzz + 189*nxx*nx*ny - 189*nx*nyy*nyy*ny + (4347.0/2.0)*nx*nyy*ny*nzz + (4725.0/2.0)*nx*ny*nzz*nzz - 1134*nx*ny*nzz + (63.0/2.0)*nx*ny);  // power=7

    // Row 10 (target order 3)
    mat[10][0] = FAR_REAL(45*nxx*ny - 30*nyy*ny + 45*ny*nzz);  // power=4
    mat[10][1] = FAR_REAL(225*nxx*nx*ny - 300*nx*nyy*ny + 225*nx*ny*nzz);  // power=5
    mat[10][2] = FAR_REAL(315*nxx*nyy - 45*nxx - 210*nyy*nyy + 315*nyy*nzz + 90*nyy - 45*nzz);  // power=5
    mat[10][3] = FAR_REAL(225*nxx*ny*nz - 300*nyy*ny*nz + 225*ny*nzz*nz);  // power=5
    mat[10][4] = FAR_REAL(315*nxx*nxx*ny - 735.0/2.0*nxx*nyy*ny + (315.0/2.0)*nxx*ny*nzz - 105*nxx*ny + 105*nyy*nyy*ny - 105.0/2.0*nyy*ny*nzz - 70*nyy*ny - 315.0/2.0*ny*nzz*nzz + 105*ny*nzz + 15*ny);  // power=6
    mat[10][5] = FAR_REAL((315.0/2.0)*nxx*nxx*ny + (105.0/2.0)*nxx*nyy*ny - 315.0/2.0*nxx*ny*nzz - 105*nxx*ny - 105*nyy*nyy*ny + (735.0/2.0)*nyy*ny*nzz + 70*nyy*ny - 315*ny*nzz*nzz + 105*ny*nzz - 15*ny);  // power=6
    mat[10][6] = FAR_REAL(15*nxx*nxx*nx + 870*nxx*nx*nyy - 75*nxx*nx*nzz - 105*nxx*nx - 720*nx*nyy*nyy + 765*nx*nyy*nzz + 105*nx*nyy - 90*nx*nzz*nzz + 15*nx);  // power=6
    mat[10][7] = FAR_REAL(525*nxx*nx*ny*nz - 1050*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[10][8] = FAR_REAL(945*nxx*nyy*nz - 105*nxx*nz - 630*nyy*nyy*nz + 945*nyy*nzz*nz - 105*nzz*nz + 30*nz);  // power=6
    mat[10][9] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*ny - 693*nxx*nx*nyy*ny + (2079.0/2.0)*nxx*nx*ny*nzz - 1701.0/2.0*nxx*nx*ny + 189*nx*nyy*ny - 567.0/2.0*nx*ny*nzz + 126*nx*ny);  // power=7
    mat[10][10] = FAR_REAL((2079.0/2.0)*nxx*nyy*nyy - 567*nxx*nyy + (63.0/2.0)*nxx - 693*nyy*nyy*nyy + (2079.0/2.0)*nyy*nyy*nzz + 756*nyy*nyy - 567*nyy*nzz - 189*nyy + (63.0/2.0)*nzz + 6);  // power=7
    mat[10][11] = FAR_REAL((2079.0/2.0)*nxx*ny*nzz*nz - 567.0/2.0*nxx*ny*nz - 693*nyy*ny*nzz*nz + 189*nyy*ny*nz + (2079.0/2.0)*ny*nzz*nzz*nz - 1701.0/2.0*ny*nzz*nz + 126*ny*nz);  // power=7
    mat[10][12] = FAR_REAL((6237.0/2.0)*nxx*nxx*nyy - 567.0/2.0*nxx*nxx - 2079*nxx*nyy*nyy + (6237.0/2.0)*nxx*nyy*nzz - 1701.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz + 126*nxx + 189*nyy*nyy - 567.0/2.0*nyy*nzz - 63.0/2.0*nyy + 45.0/2.0);  // power=7
    mat[10][13] = FAR_REAL((6237.0/2.0)*nxx*nxx*ny*nz - 2079*nxx*nyy*ny*nz + (6237.0/2.0)*nxx*ny*nzz*nz - 1984499999998016.0/999999999999.0*nxx*ny*nz + 189*nyy*ny*nz - 567.0/2.0*ny*nzz*nz + 126*ny*nz);  // power=7
    mat[10][14] = FAR_REAL((6237.0/2.0)*nxx*nyy*ny*nz - 1701.0/2.0*nxx*ny*nz - 2079*nyy*nyy*ny*nz + (6237.0/2.0)*nyy*ny*nzz*nz + 756*nyy*ny*nz - 1701.0/2.0*ny*nzz*nz + 63*ny*nz);  // power=7
    mat[10][15] = FAR_REAL(189*nxx*nxx*nx*nz + 5418*nxx*nx*nyy*nz - 189*nxx*nx*nzz*nz - 567*nxx*nx*nz - 5165999999999999.0/1000000000000.0*nx*nyy*nyy*nz + 4851*nx*nyy*nzz*nz + 63*nx*nyy*nz - 378*nx*nzz*nzz*nz + 63*nx*nz);  // power=7
    mat[10][16] = FAR_REAL((482624999998552.0/999999999997.0)*nxx*nxx*nxx*ny - 1287.0/4.0*nxx*nxx*nyy*ny - 19305.0/8.0*nxx*nxx*ny*nzz - 297*nxx*nxx*ny + (3861.0/2.0)*nxx*nyy*ny*nzz - 19305.0/8.0*nxx*ny*nzz*nzz + 1782*nxx*ny*nzz - 1287.0/4.0*nyy*ny*nzz*nzz + (482624999998552.0/999999999997.0)*ny*nzz*nzz*nzz - 297*ny*nzz*nzz);  // power=8
    mat[10][17] = FAR_REAL((482624999998552.0/999999999997.0)*nxx*nyy*nyy*ny - 11583.0/4.0*nxx*nyy*ny*nzz - 297.0/2.0*nxx*nyy*ny + (482624999998552.0/999999999997.0)*nxx*ny*nzz*nzz + (891.0/2.0)*nxx*ny*nzz - 1287.0/4.0*nyy*nyy*nyy*ny + (19305.0/8.0)*nyy*nyy*ny*nzz + 297*nyy*nyy*ny - 6435.0/2.0*nyy*ny*nzz*nzz - 148499999999851.0/999999999999.0*nyy*ny*nzz - 243.0/2.0*nyy*ny + (482624999998552.0/999999999997.0)*ny*nzz*nzz*nzz + (297.0/2.0)*ny*nzz*nzz - 81.0/2.0*ny*nzz + 15*ny);  // power=8
    mat[10][18] = FAR_REAL((27.0/2.0)*nxx*nxx*nxx*nx - 81*nxx*nxx*nx*nyy - 405.0/2.0*nxx*nxx*nx*nzz + 1647*nxx*nx*nyy*nyy - 405.0/2.0*nxx*nx*nyy*nyy - 13095.0/2.0*nxx*nx*nyy*nzz + 2025*nxx*nx*nyy*nzz - 513.0/2.0*nxx*nx*nyy + (891.0/2.0)*nxx*nx*nzz - 27.0/2.0*nxx*nx - 3141.0/2.0*nx*nyy*nyy*nyy - 108*nx*nyy*nyy*nyy + 4752*nx*nyy*nyy*nzz + (4455.0/2.0)*nx*nyy*nyy*nzz + (2133.0/2.0)*nx*nyy*nyy - 756*nx*nyy*nzz*nzz - 3240*nx*nyy*nzz*nzz - 783.0/2.0*nx*nyy*nzz - 243*nx*nyy + 216*nx*nzz*nzz*nzz - 27.0/2.0*nx*nzz + 15*nx);  // power=8
    mat[10][19] = FAR_REAL((27.0/2.0)*nxx*nxx*nxx*nx + (1930499999998069.0/999999999999.0)*nxx*nxx*nx*nyy - 81*nxx*nxx*nx*nyy - 405.0/2.0*nxx*nxx*nx*nzz - 297.0/2.0*nxx*nxx*nx - 1476*nxx*nx*nyy*nyy - 405.0/2.0*nxx*nx*nyy*nyy - 4365*nxx*nx*nyy*nzz + 2025*nxx*nx*nyy*nzz - 468*nxx*nx*nyy + 297*nxx*nx*nzz + 72*nxx*nx - 189*nx*nyy*nyy*nyy - 108*nx*nyy*nyy*nyy + 3168*nx*nyy*nyy*nzz + (4455.0/2.0)*nx*nyy*nyy*nzz + 315*nx*nyy*nyy - 504*nx*nyy*nzz*nzz - 3240*nx*nyy*nzz*nzz - 261*nx*nyy*nzz - 27.0/2.0*nx*nyy + 216*nx*nzz*nzz*nzz - 9*nx*nzz - 9.0/2.0*nx);  // power=8
    mat[10][20] = FAR_REAL((4023.0/2.0)*nxx*nxx*nx*ny*nz - 1422*nxx*nx*nyy*ny*nz - 135*nxx*nx*ny*nzz*nz - 1188*nxx*nx*ny*nz - 216*nx*nyy*nyy*ny*nz + 855*nx*nyy*ny*nzz*nz + 297*nx*nyy*ny*nz - 4293.0/2.0*nx*ny*nzz*nzz*nz + 1485*nx*ny*nzz*nz - 81*nx*ny*nz);  // power=8
    mat[10][21] = FAR_REAL((1930499999998069.0/999999999999.0)*nxx*nyy*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nyy*nzz*nz - 891.0/2.0*nxx*nyy*nz + (297.0/2.0)*nxx*nzz*nz - 1287*nyy*nyy*nyy*nz + (3217499999996782.0/999999999999.0)*nyy*nyy*nzz*nz + 594*nyy*nyy*nz - 1930499999998069.0/999999999999.0*nyy*nzz*nzz*nz + (297.0/2.0)*nyy*nzz*nz - 162*nyy*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz + 15*nz);  // power=8
    mat[10][22] = FAR_REAL((11583.0/4.0)*nxx*nxx*nyy*ny - 11583.0/4.0*nxx*nxx*ny*nzz - 891.0/2.0*nxx*nxx*ny - 3861.0/2.0*nxx*nyy*nyy*ny + (3861.0/2.0)*nxx*nyy*ny*nzz - 3861.0/2.0*nxx*ny*nzz*nzz + 1782*nxx*ny*nzz + (81.0/2.0)*nxx*ny + (3861.0/2.0)*nyy*nyy*ny*nzz - 1945723264306643.0/549755813889.0*nyy*ny*nzz*nzz + (965249999997104.0/999999999997.0)*ny*nzz*nzz*nzz - 297.0/2.0*ny*nzz*nzz - 81.0/2.0*ny*nzz);  // power=8
    mat[10][23] = FAR_REAL((11583.0/2.0)*nxx*nxx*nyy*nz - 891.0/2.0*nxx*nxx*nz - 3861*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 1782*nxx*nyy*nz - 297*nxx*nzz*nz + 243*nxx*nz + 1287*nyy*nyy*nzz*nz - 1930499999998069.0/999999999999.0*nyy*nzz*nzz*nz + 594*nyy*nzz*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz);  // power=8
    mat[10][24] = FAR_REAL((891.0/2.0)*nxx*nxx*nx*ny*nz + (5589000000000001.0/1000000000000.0)*nxx*nx*nyy*ny*nz - 2673*nxx*nx*ny*nzz*nz - 891*nxx*nx*ny*nz - 4509*nx*nyy*nyy*ny*nz + (10485.0/2.0)*nx*nyy*ny*nzz*nz + (1093499999998906.0/999999999999.0)*nx*nyy*ny*nz - 3118499999996881.0/999999999999.0*nx*ny*nzz*nzz*nz + (3861.0/2.0)*nx*ny*nzz*nz - 729.0/2.0*nx*ny*nz);  // power=8
    mat[10][25] = FAR_REAL((189.0/2.0)*nxx*nxx*nx*ny + 3024*nxx*nx*nyy*ny - 1323.0/2.0*nxx*nx*ny*nzz - 1701.0/2.0*nxx*nx*ny - 2268*nx*nyy*nyy*ny + (4347.0/2.0)*nx*nyy*ny*nzz + (2079.0/2.0)*nx*nyy*ny - 756*nx*ny*nzz*nzz - 63.0/2.0*nx*ny);  // power=7
    mat[10][26] = FAR_REAL(-63*nxx*nxx*nx*ny + 63*nxx*nx*nyy*ny + (3559499999996441.0/999999999999.0)*nxx*nx*ny*nzz - 567.0/2.0*nxx*nx*ny + 126*nx*nyy*nyy*ny - 1449*nx*nyy*ny*nzz + (7245.0/2.0)*nx*ny*nzz*nzz - 5103.0/2.0*nx*ny*nzz + 189*nx*ny);  // power=7
    mat[10][27] = FAR_REAL((6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz - 2079*nyy*nyy*nzz + 189*nyy*nyy + (6237.0/2.0)*nyy*nzz*nzz - 1701.0/2.0*nyy*nzz - 63.0/2.0*nyy - 567.0/2.0*nzz*nzz + 126*nzz + 45.0/2.0);  // power=7

    // Row 11 (target order 3)
    mat[11][0] = FAR_REAL(45*nxx*nz + 45*nyy*nz - 30*nzz*nz);  // power=4
    mat[11][1] = FAR_REAL(225*nxx*nx*nz + 225*nx*nyy*nz - 300*nx*nzz*nz);  // power=5
    mat[11][2] = FAR_REAL(225*nxx*ny*nz + 225*nyy*ny*nz - 300*ny*nzz*nz);  // power=5
    mat[11][3] = FAR_REAL(315*nxx*nzz - 45*nxx + 315*nyy*nzz - 45*nyy - 210*nzz*nzz + 90*nzz);  // power=5
    mat[11][4] = FAR_REAL(315*nxx*nxx*nz + (315.0/2.0)*nxx*nyy*nz - 735.0/2.0*nxx*nzz*nz - 105*nxx*nz - 315.0/2.0*nyy*nyy*nz - 105.0/2.0*nyy*nzz*nz + 105*nyy*nz + 105*nzz*nzz*nz - 70*nzz*nz + 15*nz);  // power=6
    mat[11][5] = FAR_REAL((315.0/2.0)*nxx*nxx*nz + 315*nxx*nyy*nz - 420*nxx*nzz*nz + (315.0/2.0)*nyy*nyy*nz - 420*nyy*nzz*nz + 210*nzz*nzz*nz - 140*nzz*nz + 30*nz);  // power=6
    mat[11][6] = FAR_REAL(525*nxx*nx*ny*nz + 525*nx*nyy*ny*nz - 1050*nx*ny*nzz*nz);  // power=6
    mat[11][7] = FAR_REAL(15*nxx*nxx*nx - 75*nxx*nx*nyy + 870*nxx*nx*nzz - 105*nxx*nx - 90*nx*nyy*nyy + 765*nx*nyy*nzz - 720*nx*nzz*nzz + 105*nx*nzz + 15*nx);  // power=6
    mat[11][8] = FAR_REAL(945*nxx*ny*nzz - 105*nxx*ny + 945*nyy*ny*nzz - 105*nyy*ny - 630*ny*nzz*nzz + 30*ny);  // power=6
    mat[11][9] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*nz + (2079.0/2.0)*nxx*nx*nyy*nz - 693*nxx*nx*nzz*nz - 1701.0/2.0*nxx*nx*nz - 567.0/2.0*nx*nyy*nz + 189*nx*nzz*nz + 126*nx*nz);  // power=7
    mat[11][10] = FAR_REAL((2079.0/2.0)*nxx*nyy*ny*nz - 567.0/2.0*nxx*ny*nz + (2079.0/2.0)*nyy*nyy*ny*nz - 693*nyy*ny*nzz*nz - 1701.0/2.0*nyy*ny*nz + 189*ny*nzz*nz + 126*ny*nz);  // power=7
    mat[11][11] = FAR_REAL((2079.0/2.0)*nxx*nzz*nzz - 567*nxx*nzz + (63.0/2.0)*nxx + (2079.0/2.0)*nyy*nzz*nzz - 567*nyy*nzz + (63.0/2.0)*nyy - 693*nzz*nzz*nzz + 756*nzz*nzz - 189*nzz + 6);  // power=7
    mat[11][12] = FAR_REAL((6237.0/2.0)*nxx*nxx*ny*nz + (6237.0/2.0)*nxx*nyy*ny*nz - 2079*nxx*ny*nzz*nz - 1984499999998016.0/999999999999.0*nxx*ny*nz - 567.0/2.0*nyy*ny*nz + 189*ny*nzz*nz + 126*ny*nz);  // power=7
    mat[11][13] = FAR_REAL((6237.0/2.0)*nxx*nxx*nzz - 567.0/2.0*nxx*nxx + (6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 2079*nxx*nzz*nzz - 1701.0/2.0*nxx*nzz + 126*nxx - 567.0/2.0*nyy*nzz + 189*nzz*nzz - 63.0/2.0*nzz + 45.0/2.0);  // power=7
    mat[11][14] = FAR_REAL((6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz + (6237.0/2.0)*nyy*nyy*nzz - 567.0/2.0*nyy*nyy - 2079*nyy*nzz*nzz - 1701.0/2.0*nyy*nzz + 126*nyy + 189*nzz*nzz - 63.0/2.0*nzz + 45.0/2.0);  // power=7
    mat[11][15] = FAR_REAL(189*nxx*nxx*nx*ny - 189*nxx*nx*nyy*ny + 4914*nxx*nx*ny*nzz - 567*nxx*nx*ny - 378*nx*nyy*nyy*ny + 4347*nx*nyy*ny*nzz - 5670*nx*ny*nzz*nzz + 567*nx*ny*nzz + 63*nx*ny);  // power=7
    mat[11][16] = FAR_REAL((482624999998552.0/999999999997.0)*nxx*nxx*nxx*nz + (482624999998552.0/999999999997.0)*nxx*nxx*nyy*nz - 6435.0/2.0*nxx*nxx*nzz*nz + (297.0/2.0)*nxx*nxx*nz - 11583.0/4.0*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz + (19305.0/8.0)*nxx*nzz*nzz*nz - 148499999999851.0/999999999999.0*nxx*nzz*nz - 81.0/2.0*nxx*nz + (482624999998552.0/999999999997.0)*nyy*nzz*nzz*nz - 297.0/2.0*nyy*nzz*nz - 1287.0/4.0*nzz*nzz*nzz*nz + 297*nzz*nzz*nz - 243.0/2.0*nzz*nz + 15*nz);  // power=8
    mat[11][17] = FAR_REAL((482624999998552.0/999999999997.0)*nxx*nyy*nyy*nz - 11583.0/4.0*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz + (482624999998552.0/999999999997.0)*nxx*nzz*nzz*nz - 297.0/2.0*nxx*nzz*nz + (482624999998552.0/999999999997.0)*nyy*nyy*nyy*nz - 6435.0/2.0*nyy*nyy*nzz*nz + (297.0/2.0)*nyy*nyy*nz + (19305.0/8.0)*nyy*nzz*nzz*nz - 148499999999851.0/999999999999.0*nyy*nzz*nz - 81.0/2.0*nyy*nz - 1287.0/4.0*nzz*nzz*nzz*nz + 297*nzz*nzz*nz - 243.0/2.0*nzz*nz + 15*nz);  // power=8
    mat[11][18] = FAR_REAL(-891.0/2.0*nxx*nxx*nx*ny*nz + 2403*nxx*nx*nyy*ny*nz - 5049*nxx*nx*ny*nzz*nz + 891*nxx*nx*ny*nz + (5697.0/2.0)*nx*nyy*nyy*ny*nz - 9945.0/2.0*nx*nyy*ny*nzz*nz - 3321.0/2.0*nx*nyy*ny*nz + 5049*nx*ny*nzz*nzz*nz - 3267.0/2.0*nx*ny*nzz*nz + (729.0/2.0)*nx*ny*nz);  // power=8
    mat[11][19] = FAR_REAL(1566*nxx*nxx*nx*ny*nz + 2538*nxx*nx*nyy*ny*nz - 6470999999999999.0/1000000000000.0*nxx*nx*ny*nzz*nz - 297*nxx*nx*ny*nz + 972*nx*nyy*nyy*ny*nz - 7695.0/2.0*nx*nyy*ny*nzz*nz - 891.0/2.0*nx*nyy*ny*nz + 4833*nx*ny*nzz*nzz*nz - 2673.0/2.0*nx*ny*nzz*nz + (567.0/2.0)*nx*ny*nz);  // power=8
    mat[11][20] = FAR_REAL((9.0/2.0)*nxx*nxx*nxx*nx - 27*nxx*nxx*nx*nyy + (1930499999998069.0/999999999999.0)*nxx*nxx*nx*nzz - 135.0/2.0*nxx*nxx*nx*nzz - 297.0/2.0*nxx*nxx*nx - 135.0/2.0*nxx*nx*nyy*nyy + (1930499999998069.0/999999999999.0)*nxx*nx*nyy*nzz + 675*nxx*nx*nyy*nzz - 297.0/2.0*nxx*nx*nyy - 3217499999996782.0/999999999999.0*nxx*nx*nzz*nzz - 297.0/2.0*nxx*nx*nzz + 81*nxx*nx - 36*nx*nyy*nyy*nyy + (1485.0/2.0)*nx*nyy*nyy*nzz - 1080*nx*nyy*nzz*nzz - 891.0/2.0*nx*nyy*nzz + (81.0/2.0)*nx*nyy + 1287*nx*nzz*nzz*nzz + 72*nx*nzz*nzz*nzz - 1485.0/2.0*nx*nzz*nzz + 243*nx*nzz - 39.0/2.0*nx);  // power=8
    mat[11][21] = FAR_REAL((1930499999998069.0/999999999999.0)*nxx*nyy*ny*nzz - 297.0/2.0*nxx*nyy*ny - 1930499999998069.0/999999999999.0*nxx*ny*nzz*nzz + (891.0/2.0)*nxx*ny*nzz + (1930499999998069.0/999999999999.0)*nyy*nyy*ny*nzz - 297.0/2.0*nyy*nyy*ny - 3217499999996782.0/999999999999.0*nyy*ny*nzz*nzz - 297.0/2.0*nyy*ny*nzz + 81*nyy*ny + 1287*ny*nzz*nzz*nzz - 594*ny*nzz*nzz + 162*ny*nzz - 15*ny);  // power=8
    mat[11][22] = FAR_REAL((11583.0/4.0)*nxx*nxx*nyy*nz - 11583.0/4.0*nxx*nxx*nzz*nz + (891.0/2.0)*nxx*nxx*nz + (11583.0/4.0)*nxx*nyy*nyy*nz - 7722*nxx*nyy*nzz*nz - 891*nxx*nyy*nz + (11583.0/4.0)*nxx*nzz*nzz*nz - 297*nxx*nzz*nz - 81.0/2.0*nxx*nz - 11583.0/4.0*nyy*nyy*nzz*nz + (891.0/2.0)*nyy*nyy*nz + (11583.0/4.0)*nyy*nzz*nzz*nz - 297*nyy*nzz*nz - 81.0/2.0*nyy*nz - 1287.0/2.0*nzz*nzz*nzz*nz + 594*nzz*nzz*nz - 243*nzz*nz + 30*nz);  // power=8
    mat[11][23] = FAR_REAL((11583.0/2.0)*nxx*nxx*ny*nzz - 891.0/2.0*nxx*nxx*ny + (11583.0/2.0)*nxx*nyy*ny*nzz - 891.0/2.0*nxx*nyy*ny - 11583.0/2.0*nxx*ny*nzz*nzz - 2673.0/2.0*nxx*ny*nzz + 243*nxx*ny - 1930499999998069.0/999999999999.0*nyy*ny*nzz*nzz + (891.0/2.0)*nyy*ny*nzz + 1287*ny*nzz*nzz*nzz - 594*ny*nzz*nzz + 162*ny*nzz - 15*ny);  // power=8
    mat[11][24] = FAR_REAL((27.0/2.0)*nxx*nxx*nxx*nx - 81*nxx*nxx*nx*nyy - 405.0/2.0*nxx*nxx*nx*nzz - 567.0/2.0*nxx*nx*nyy*nyy - 405.0/2.0*nxx*nx*nyy*nyy + (10071.0/2.0)*nxx*nx*nyy*nzz + 2025*nxx*nx*nyy*nzz - 513.0/2.0*nxx*nx*nyy - 1930499999998069.0/999999999999.0*nxx*nx*nzz*nzz + (891.0/2.0)*nxx*nx*nzz - 27.0/2.0*nxx*nx - 567.0/2.0*nx*nyy*nyy*nyy - 108*nx*nyy*nyy*nyy + 4752*nx*nyy*nyy*nzz + (4455.0/2.0)*nx*nyy*nyy*nzz + 27*nx*nyy*nyy - 756*nx*nyy*nzz*nzz - 3240*nx*nyy*nzz*nzz - 6129.0/2.0*nx*nyy*nzz + 162*nx*nyy + 1287*nx*nzz*nzz*nzz + 216*nx*nzz*nzz*nzz - 2079.0/2.0*nx*nzz*nzz + (783.0/2.0)*nx*nzz - 15*nx);  // power=8
    mat[11][25] = FAR_REAL(-63*nxx*nxx*nx*nz + (6363.0/2.0)*nxx*nx*nyy*nz + 63*nxx*nx*nzz*nz - 567.0/2.0*nxx*nx*nz + (3244499999996756.0/999999999999.0)*nx*nyy*nyy*nz - 1827*nx*nyy*nzz*nz - 4347.0/2.0*nx*nyy*nz + 126*nx*nzz*nzz*nz + 189*nx*nz);  // power=7
    mat[11][26] = FAR_REAL((189.0/2.0)*nxx*nxx*nx*nz - 1323.0/2.0*nxx*nx*nyy*nz + 3024*nxx*nx*nzz*nz - 1701.0/2.0*nxx*nx*nz - 756*nx*nyy*nyy*nz + (4347.0/2.0)*nx*nyy*nzz*nz - 2268*nx*nzz*nzz*nz + (2079.0/2.0)*nx*nzz*nz - 63.0/2.0*nx*nz);  // power=7
    mat[11][27] = FAR_REAL((6237.0/2.0)*nxx*ny*nzz*nz - 1701.0/2.0*nxx*ny*nz + (6237.0/2.0)*nyy*ny*nzz*nz - 1701.0/2.0*nyy*ny*nz - 2079*ny*nzz*nzz*nz + 756*ny*nzz*nz + 63*ny*nz);  // power=7

    // Row 12 (target order 3)
    mat[12][0] = FAR_REAL(-60*nxx*ny + 15*nyy*ny + 15*ny*nzz);  // power=4
    mat[12][1] = FAR_REAL(-300*nxx*nx*ny + 225*nx*nyy*ny + 225*nx*ny*nzz);  // power=5
    mat[12][2] = FAR_REAL(-420*nxx*nyy + 60*nxx + 105*nyy*nyy + 105*nyy*nzz - 45*nyy - 15*nzz);  // power=5
    mat[12][3] = FAR_REAL(-450*nxx*ny*nz + 75*nyy*ny*nz + 75*ny*nzz*nz);  // power=5
    mat[12][4] = FAR_REAL(-420*nxx*nxx*ny + 315*nxx*nyy*ny + 315*nxx*ny*nzz + 140*nxx*ny - 105.0/2.0*nyy*nyy*ny - 105*nyy*ny*nzz + 35*nyy*ny - 105.0/2.0*ny*nzz*nzz + 35*ny*nzz - 20*ny);  // power=6
    mat[12][5] = FAR_REAL(-210*nxx*nxx*ny - 315.0/2.0*nxx*nyy*ny + (945.0/2.0)*nxx*ny*nzz + 140*nxx*ny + (105.0/2.0)*nyy*nyy*ny - 105.0/2.0*nyy*ny*nzz - 35*nyy*ny - 105*ny*nzz*nzz + 35*ny*nzz - 5*ny);  // power=6
    mat[12][6] = FAR_REAL(5*nxx*nxx*nx - 1285*nxx*nx*nyy - 25*nxx*nx*nzz + 140*nxx*nx + 285*nx*nyy*nyy + 255*nx*nyy*nzz + 210*nx*nyy - 30*nx*nzz*nzz - 45*nx);  // power=6
    mat[12][7] = FAR_REAL(-1050*nxx*nx*ny*nz + 525*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[12][8] = FAR_REAL(-1260*nxx*nyy*nz + 140*nxx*nz + 315*nyy*nyy*nz + 315*nyy*nzz*nz - 175*nyy*nz - 35*nzz*nz + 10*nz);  // power=6
    mat[12][9] = FAR_REAL(-1386*nxx*nxx*nx*ny + (693.0/2.0)*nxx*nx*nyy*ny + (693.0/2.0)*nxx*nx*ny*nzz + 1134*nxx*nx*ny - 189.0/2.0*nx*nyy*ny - 189.0/2.0*nx*ny*nzz - 168*nx*ny);  // power=7
    mat[12][10] = FAR_REAL(-1386*nxx*nyy*nyy + 756*nxx*nyy - 42*nxx + (693.0/2.0)*nyy*nyy*nyy + (693.0/2.0)*nyy*nyy*nzz - 378*nyy*nyy - 189*nyy*nzz + (189.0/2.0)*nyy + (21.0/2.0)*nzz - 3);  // power=7
    mat[12][11] = FAR_REAL(-1386*nxx*ny*nzz*nz + 378*nxx*ny*nz + (693.0/2.0)*nyy*ny*nzz*nz - 189.0/2.0*nyy*ny*nz + (693.0/2.0)*ny*nzz*nzz*nz - 567.0/2.0*ny*nzz*nz + 42*ny*nz);  // power=7
    mat[12][12] = FAR_REAL(-4158*nxx*nxx*nyy + 378*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy + (2079.0/2.0)*nxx*nyy*nzz + (3213.0/2.0)*nxx*nyy - 189.0/2.0*nxx*nzz - 441.0/2.0*nxx - 189.0/2.0*nyy*nyy - 189.0/2.0*nyy*nzz - 63*nyy + 45.0/2.0);  // power=7
    mat[12][13] = FAR_REAL(-4158*nxx*nxx*ny*nz + (2079.0/2.0)*nxx*nyy*ny*nz + (2079.0/2.0)*nxx*ny*nzz*nz + 1701*nxx*ny*nz - 189.0/2.0*nyy*ny*nz - 189.0/2.0*ny*nzz*nz - 63*ny*nz);  // power=7
    mat[12][14] = FAR_REAL(-4158*nxx*nyy*ny*nz + 1134*nxx*ny*nz + (2079.0/2.0)*nyy*nyy*ny*nz + (2079.0/2.0)*nyy*ny*nzz*nz - 1701.0/2.0*nyy*ny*nz - 567.0/2.0*ny*nzz*nz + 126*ny*nz);  // power=7
    mat[12][15] = FAR_REAL(63*nxx*nxx*nx*nz - 9009*nxx*nx*nyy*nz - 63*nxx*nx*nzz*nz + 756*nxx*nx*nz + 1323*nx*nyy*nyy*nz + 1197*nx*nyy*nzz*nz + 1386*nx*nyy*nz - 126*nx*nzz*nzz*nz - 189*nx*nz);  // power=7
    mat[12][16] = FAR_REAL(-1287.0/2.0*nxx*nxx*nxx*ny + (1287.0/8.0)*nxx*nxx*nyy*ny + (32175.0/8.0)*nxx*nxx*ny*nzz + 396*nxx*nxx*ny - 3861.0/4.0*nxx*nyy*ny*nzz - 6435.0/4.0*nxx*ny*nzz*nzz - 891*nxx*ny*nzz - 135.0/2.0*nxx*ny + (1287.0/8.0)*nyy*ny*nzz*nzz + (1287.0/8.0)*ny*nzz*nzz*nzz - 99*ny*nzz*nzz + (135.0/2.0)*ny*nzz);  // power=8
    mat[12][17] = FAR_REAL(-1287.0/2.0*nxx*nyy*nyy*ny + 3861*nxx*nyy*ny*nzz + 198*nxx*nyy*ny - 1287.0/2.0*nxx*ny*nzz*nzz - 594*nxx*ny*nzz + (1287.0/8.0)*nyy*nyy*nyy*ny - 6435.0/8.0*nyy*nyy*ny*nzz - 297.0/2.0*nyy*nyy*ny - 6435.0/8.0*nyy*ny*nzz*nzz + 693*nyy*ny*nzz + 27*nyy*ny + (1287.0/8.0)*ny*nzz*nzz*nzz + (99.0/2.0)*ny*nzz*nzz - 81*ny*nzz);  // power=8
    mat[12][18] = FAR_REAL((9.0/2.0)*nxx*nxx*nxx*nx - 27*nxx*nxx*nx*nyy - 135.0/2.0*nxx*nxx*nx*nzz - 2511*nxx*nx*nyy*nyy - 135.0/2.0*nxx*nx*nyy*nyy + 7890*nxx*nx*nyy*nzz + 675*nxx*nx*nyy*nzz + 552*nxx*nx*nyy - 594*nxx*nx*nzz + 3*nxx*nx + (1413.0/2.0)*nx*nyy*nyy*nyy - 36*nx*nyy*nyy*nyy - 1056*nx*nyy*nyy*nzz + (1485.0/2.0)*nx*nyy*nyy*nzz - 309.0/2.0*nx*nyy*nyy + 168*nx*nyy*nzz*nzz - 1080*nx*nyy*nzz*nzz - 1398*nx*nyy*nzz + (63.0/2.0)*nx*nyy + 72*nx*nzz*nzz*nzz + 138*nx*nzz - 15.0/2.0*nx);  // power=8
    mat[12][19] = FAR_REAL((9.0/2.0)*nxx*nxx*nxx*nx - 2574*nxx*nxx*nx*nyy - 27*nxx*nxx*nx*nyy - 135.0/2.0*nxx*nxx*nx*nzz + 198*nxx*nxx*nx + 738*nxx*nx*nyy*nyy - 135.0/2.0*nxx*nx*nyy*nyy + (17235.0/2.0)*nxx*nx*nyy*nzz + 675*nxx*nx*nyy*nzz + (976499999999023.0/999999999999.0)*nxx*nx*nyy - 1287.0/2.0*nxx*nx*nzz - 207.0/2.0*nxx*nx + (189.0/2.0)*nx*nyy*nyy*nyy - 36*nx*nyy*nyy*nyy - 1584*nx*nyy*nyy*nzz + (1485.0/2.0)*nx*nyy*nyy*nzz - 315.0/2.0*nx*nyy*nyy + 252*nx*nyy*nzz*nzz - 1080*nx*nyy*nzz*nzz - 2709.0/2.0*nx*nyy*nzz - 27*nx*nyy + 72*nx*nzz*nzz*nzz + (279.0/2.0)*nx*nzz + 6*nx);  // power=8
    mat[12][20] = FAR_REAL(-1437336575412790.0/549755813889.0*nxx*nxx*nx*ny*nz + 711*nxx*nx*nyy*ny*nz + 3285*nxx*nx*ny*nzz*nz + 1089*nxx*nx*ny*nz + 108*nx*nyy*nyy*ny*nz - 855.0/2.0*nx*nyy*ny*nzz*nz - 297.0/2.0*nx*nyy*ny*nz - 1071.0/2.0*nx*ny*nzz*nzz*nz - 495.0/2.0*nx*ny*nzz*nz - 189.0/2.0*nx*ny*nz);  // power=8
    mat[12][21] = FAR_REAL(-2574*nxx*nyy*nyy*nz + 2574*nxx*nyy*nzz*nz + 594*nxx*nyy*nz - 198*nxx*nzz*nz + (1287.0/2.0)*nyy*nyy*nyy*nz - 1089.0/2.0*nyy*nyy*nz - 1287.0/2.0*nyy*nzz*nzz*nz + 297*nyy*nzz*nz + 81*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz);  // power=8
    mat[12][22] = FAR_REAL(-3861*nxx*nxx*nyy*ny + 3861*nxx*nxx*ny*nzz + 594*nxx*nxx*ny + (3861.0/4.0)*nxx*nyy*nyy*ny + 3861*nxx*nyy*ny*nzz + (1485.0/2.0)*nxx*nyy*ny - 1238187531831500.0/549755813889.0*nxx*ny*nzz*nzz - 3267.0/2.0*nxx*ny*nzz - 189*nxx*ny - 3861.0/4.0*nyy*nyy*ny*nzz - 1287.0/2.0*nyy*ny*nzz*nzz + (1485.0/2.0)*nyy*ny*nzz - 135.0/2.0*nyy*ny + (1287.0/4.0)*ny*nzz*nzz*nzz - 99.0/2.0*ny*nzz*nzz - 27.0/2.0*ny*nzz + 15*ny);  // power=8
    mat[12][23] = FAR_REAL(-7722*nxx*nxx*nyy*nz + 594*nxx*nxx*nz + (3861.0/2.0)*nxx*nyy*nyy*nz + (2476375063663000.0/549755813889.0)*nxx*nyy*nzz*nz + (3267.0/2.0)*nxx*nyy*nz - 693.0/2.0*nxx*nzz*nz - 189*nxx*nz - 1287.0/2.0*nyy*nyy*nzz*nz - 1287.0/2.0*nyy*nzz*nzz*nz + (891.0/2.0)*nyy*nzz*nz - 135*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz + 15*nz);  // power=8
    mat[12][24] = FAR_REAL(81*nxx*nxx*nx*ny*nz - 8127*nxx*nx*nyy*ny*nz + 2439*nxx*nx*ny*nzz*nz + 1188*nxx*nx*ny*nz + (2889.0/2.0)*nx*nyy*nyy*ny*nz + 585*nx*nyy*ny*nzz*nz + 567*nx*nyy*ny*nz - 1719.0/2.0*nx*ny*nzz*nzz*nz - 99*nx*ny*nzz*nz - 189*nx*ny*nz);  // power=8
    mat[12][25] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*ny - 8379.0/2.0*nxx*nx*nyy*ny - 441.0/2.0*nxx*nx*ny*nzz + 1134*nxx*nx*ny + (1953.0/2.0)*nx*nyy*nyy*ny + (1449.0/2.0)*nx*nyy*ny*nzz + 189*nx*nyy*ny - 252*nx*ny*nzz*nzz - 441.0/2.0*nx*ny);  // power=7
    mat[12][26] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*ny - 63.0/2.0*nxx*nx*nyy*ny - 8757.0/2.0*nxx*nx*ny*nzz + 378*nxx*nx*ny - 63*nx*nyy*nyy*ny + (1449.0/2.0)*nx*nyy*ny*nzz + (1575.0/2.0)*nx*ny*nzz*nzz + 567*nx*ny*nzz - 189.0/2.0*nx*ny);  // power=7
    mat[12][27] = FAR_REAL(-4158*nxx*nyy*nzz + 378*nxx*nyy + 378*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz - 756*nyy*nzz + (189.0/2.0)*nyy - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7

    // Row 13 (target order 3)
    mat[13][0] = FAR_REAL(-60*nxx*nz + 15*nyy*nz + 15*nzz*nz);  // power=4
    mat[13][1] = FAR_REAL(-300*nxx*nx*nz + 225*nx*nyy*nz + 225*nx*nzz*nz);  // power=5
    mat[13][2] = FAR_REAL(-450*nxx*ny*nz + 75*nyy*ny*nz + 75*ny*nzz*nz);  // power=5
    mat[13][3] = FAR_REAL(-420*nxx*nzz + 60*nxx + 105*nyy*nzz - 15*nyy + 105*nzz*nzz - 45*nzz);  // power=5
    mat[13][4] = FAR_REAL(-420*nxx*nxx*nz + 315*nxx*nyy*nz + 315*nxx*nzz*nz + 140*nxx*nz - 105.0/2.0*nyy*nyy*nz - 105*nyy*nzz*nz + 35*nyy*nz - 105.0/2.0*nzz*nzz*nz + 35*nzz*nz - 20*nz);  // power=6
    mat[13][5] = FAR_REAL(-210*nxx*nxx*nz - 315.0/2.0*nxx*nyy*nz + (945.0/2.0)*nxx*nzz*nz + (105.0/2.0)*nyy*nyy*nz - 105.0/2.0*nyy*nzz*nz - 105*nzz*nzz*nz + 70*nzz*nz - 15*nz);  // power=6
    mat[13][6] = FAR_REAL(-1050*nxx*nx*ny*nz + 525*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[13][7] = FAR_REAL(5*nxx*nxx*nx - 25*nxx*nx*nyy - 1285*nxx*nx*nzz + 140*nxx*nx - 30*nx*nyy*nyy + 255*nx*nyy*nzz + 285*nx*nzz*nzz + 210*nx*nzz - 45*nx);  // power=6
    mat[13][8] = FAR_REAL(-1260*nxx*ny*nzz + 140*nxx*ny + 315*nyy*ny*nzz - 35*nyy*ny + 315*ny*nzz*nzz - 175*ny*nzz + 10*ny);  // power=6
    mat[13][9] = FAR_REAL(-1386*nxx*nxx*nx*nz + (693.0/2.0)*nxx*nx*nyy*nz + (693.0/2.0)*nxx*nx*nzz*nz + 1134*nxx*nx*nz - 189.0/2.0*nx*nyy*nz - 189.0/2.0*nx*nzz*nz - 168*nx*nz);  // power=7
    mat[13][10] = FAR_REAL(-1386*nxx*nyy*ny*nz + 378*nxx*ny*nz + (693.0/2.0)*nyy*nyy*ny*nz + (693.0/2.0)*nyy*ny*nzz*nz - 567.0/2.0*nyy*ny*nz - 189.0/2.0*ny*nzz*nz + 42*ny*nz);  // power=7
    mat[13][11] = FAR_REAL(-1386*nxx*nzz*nzz + 756*nxx*nzz - 42*nxx + (693.0/2.0)*nyy*nzz*nzz - 189*nyy*nzz + (21.0/2.0)*nyy + (693.0/2.0)*nzz*nzz*nzz - 378*nzz*nzz + (189.0/2.0)*nzz - 3);  // power=7
    mat[13][12] = FAR_REAL(-4158*nxx*nxx*ny*nz + (2079.0/2.0)*nxx*nyy*ny*nz + (2079.0/2.0)*nxx*ny*nzz*nz + 1701*nxx*ny*nz - 189.0/2.0*nyy*ny*nz - 189.0/2.0*ny*nzz*nz - 63*ny*nz);  // power=7
    mat[13][13] = FAR_REAL(-4158*nxx*nxx*nzz + 378*nxx*nxx + (2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz + (3213.0/2.0)*nxx*nzz - 441.0/2.0*nxx - 189.0/2.0*nyy*nzz - 189.0/2.0*nzz*nzz - 63*nzz + 45.0/2.0);  // power=7
    mat[13][14] = FAR_REAL(-4158*nxx*nyy*nzz + 378*nxx*nyy + 378*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz - 756*nyy*nzz + (189.0/2.0)*nyy - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
    mat[13][15] = FAR_REAL(63*nxx*nxx*nx*ny - 63*nxx*nx*nyy*ny - 8757*nxx*nx*ny*nzz + 756*nxx*nx*ny - 126*nx*nyy*nyy*ny + 1449*nx*nyy*ny*nzz + 1575*nx*ny*nzz*nzz + 1134*nx*ny*nzz - 189*nx*ny);  // power=7
    mat[13][16] = FAR_REAL(-1287.0/2.0*nxx*nxx*nxx*nz + (1287.0/8.0)*nxx*nxx*nyy*nz + (32175.0/8.0)*nxx*nxx*nzz*nz - 198*nxx*nxx*nz - 3861.0/4.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz - 6435.0/4.0*nxx*nzz*nzz*nz - 1089.0/2.0*nxx*nzz*nz + (243.0/2.0)*nxx*nz + (1287.0/8.0)*nyy*nzz*nzz*nz - 99.0/2.0*nyy*nzz*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[13][17] = FAR_REAL(-1287.0/2.0*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 594*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + 198*nxx*nzz*nz + (1287.0/8.0)*nyy*nyy*nyy*nz - 6435.0/8.0*nyy*nyy*nzz*nz + (99.0/2.0)*nyy*nyy*nz - 6435.0/8.0*nyy*nzz*nzz*nz + 693*nyy*nzz*nz - 81*nyy*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + 27*nzz*nz);  // power=8
    mat[13][18] = FAR_REAL(-81*nxx*nxx*nx*ny*nz - 2529*nxx*nx*nyy*ny*nz + 7857*nxx*nx*ny*nzz*nz - 1188*nxx*nx*ny*nz + (1539.0/2.0)*nx*nyy*nyy*ny*nz - 945*nx*nyy*ny*nzz*nz + 189*nx*nyy*ny*nz - 3429.0/2.0*nx*ny*nzz*nzz*nz - 297*nx*ny*nzz*nz + 189*nx*ny*nz);  // power=8
    mat[13][19] = FAR_REAL(-2695499999997304.0/999999999999.0*nxx*nxx*nx*ny*nz + 846*nxx*nx*nyy*ny*nz + 8568*nxx*nx*ny*nzz*nz - 99*nxx*nx*ny*nz + 324*nx*nyy*nyy*ny*nz - 2565.0/2.0*nx*nyy*ny*nzz*nz - 297.0/2.0*nx*nyy*ny*nz - 3213.0/2.0*nx*ny*nzz*nzz*nz - 891.0/2.0*nx*ny*nzz*nz + (189.0/2.0)*nx*ny*nz);  // power=8
    mat[13][20] = FAR_REAL((3.0/2.0)*nxx*nxx*nxx*nx - 9*nxx*nxx*nx*nyy - 2574*nxx*nxx*nx*nzz - 45.0/2.0*nxx*nxx*nx*nzz + 198*nxx*nxx*nx - 45.0/2.0*nxx*nx*nyy*nyy + (1287.0/2.0)*nxx*nx*nyy*nzz + 225*nxx*nx*nyy*nzz - 99.0/2.0*nxx*nx*nyy + (3217499999996782.0/999999999999.0)*nxx*nx*nzz*nzz + (445499999999554.0/999999999999.0)*nxx*nx*nzz - 108*nxx*nx - 12*nx*nyy*nyy*nyy + (495.0/2.0)*nx*nyy*nyy*nzz - 360*nx*nyy*nzz*nzz - 297.0/2.0*nx*nyy*nzz + (27.0/2.0)*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz + 24*nx*nzz*nzz*nzz - 54*nx*nzz + (27.0/2.0)*nx);  // power=8
    mat[13][21] = FAR_REAL(-2574*nxx*nyy*ny*nzz + 198*nxx*nyy*ny + 2574*nxx*ny*nzz*nzz - 594*nxx*ny*nzz + (1287.0/2.0)*nyy*nyy*ny*nzz - 99.0/2.0*nyy*nyy*ny - 297*nyy*ny*nzz + 27*nyy*ny - 1287.0/2.0*ny*nzz*nzz*nzz + (1089.0/2.0)*ny*nzz*nzz - 81*ny*nzz);  // power=8
    mat[13][22] = FAR_REAL(-3861*nxx*nxx*nyy*nz + 3861*nxx*nxx*nzz*nz - 594*nxx*nxx*nz + (3861.0/4.0)*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz - 1238187531831500.0/549755813889.0*nxx*nzz*nzz*nz - 693.0/2.0*nxx*nzz*nz + 189*nxx*nz - 3861.0/4.0*nyy*nyy*nzz*nz + (297.0/2.0)*nyy*nyy*nz - 1287.0/2.0*nyy*nzz*nzz*nz + (1287.0/2.0)*nyy*nzz*nz - 297.0/2.0*nyy*nz + (1287.0/4.0)*nzz*nzz*nzz*nz - 297*nzz*nzz*nz + (243.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[13][23] = FAR_REAL(-7722*nxx*nxx*ny*nzz + 594*nxx*nxx*ny + (3861.0/2.0)*nxx*nyy*ny*nzz - 297.0/2.0*nxx*nyy*ny + (2476375063663000.0/549755813889.0)*nxx*ny*nzz*nzz + (2079.0/2.0)*nxx*ny*nzz - 189*nxx*ny - 1287.0/2.0*nyy*ny*nzz*nzz + (297.0/2.0)*nyy*ny*nzz - 1287.0/2.0*ny*nzz*nzz*nzz + (1089.0/2.0)*ny*nzz*nzz - 216*ny*nzz + 15*ny);  // power=8
    mat[13][24] = FAR_REAL(-3*nxx*nxx*nxx*nx + 18*nxx*nxx*nx*nyy + 45*nxx*nxx*nx*nzz - 189.0/2.0*nxx*nx*nyy*nyy + 45*nxx*nx*nyy*nyy - 7974*nxx*nx*nyy*nzz - 450*nxx*nx*nyy*nzz + 657*nxx*nx*nyy + 2574*nxx*nx*nzz*nzz - 594*nxx*nx*nzz - 9.0/2.0*nxx*nx - 189.0/2.0*nx*nyy*nyy*nyy + 24*nx*nyy*nyy*nyy + 1584*nx*nyy*nyy*nzz - 495*nx*nyy*nyy*nzz + 9*nx*nyy*nyy - 252*nx*nyy*nzz*nzz + 720*nx*nyy*nzz*nzz + 1206*nx*nyy*nzz - 297.0/2.0*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz - 48*nx*nzz*nzz*nzz + (297.0/2.0)*nx*nzz*nzz - 9.0/2.0*nx*nzz + (15.0/2.0)*nx);  // power=8
    mat[13][25] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*nz - 9009.0/2.0*nxx*nx*nyy*nz - 63.0/2.0*nxx*nx*nzz*nz + 378*nxx*nx*nz + (1323.0/2.0)*nx*nyy*nyy*nz + (1197.0/2.0)*nx*nyy*nzz*nz + 693*nx*nyy*nz - 63*nx*nzz*nzz*nz - 189.0/2.0*nx*nz);  // power=7
    mat[13][26] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*nz - 441.0/2.0*nxx*nx*nyy*nz - 8379.0/2.0*nxx*nx*nzz*nz + 1134*nxx*nx*nz - 252*nx*nyy*nyy*nz + (1449.0/2.0)*nx*nyy*nzz*nz + (1953.0/2.0)*nx*nzz*nzz*nz + 189*nx*nzz*nz - 441.0/2.0*nx*nz);  // power=7
    mat[13][27] = FAR_REAL(-4158*nxx*ny*nzz*nz + 1134*nxx*ny*nz + (2079.0/2.0)*nyy*ny*nzz*nz - 567.0/2.0*nyy*ny*nz + (2079.0/2.0)*ny*nzz*nzz*nz - 1701.0/2.0*ny*nzz*nz + 126*ny*nz);  // power=7

    // Row 14 (target order 3)
    mat[14][0] = FAR_REAL(15*nxx*nz - 60*nyy*nz + 15*nzz*nz);  // power=4
    mat[14][1] = FAR_REAL(75*nxx*nx*nz - 450*nx*nyy*nz + 75*nx*nzz*nz);  // power=5
    mat[14][2] = FAR_REAL(225*nxx*ny*nz - 300*nyy*ny*nz + 225*ny*nzz*nz);  // power=5
    mat[14][3] = FAR_REAL(105*nxx*nzz - 15*nxx - 420*nyy*nzz + 60*nyy + 105*nzz*nzz - 45*nzz);  // power=5
    mat[14][4] = FAR_REAL(105*nxx*nxx*nz - 945.0/2.0*nxx*nyy*nz + (105.0/2.0)*nxx*nzz*nz - 35*nxx*nz + 210*nyy*nyy*nz + (315.0/2.0)*nyy*nzz*nz - 140*nyy*nz - 105.0/2.0*nzz*nzz*nz + 35*nzz*nz + 5*nz);  // power=6
    mat[14][5] = FAR_REAL((105.0/2.0)*nxx*nxx*nz - 315.0/2.0*nxx*nyy*nz - 105.0/2.0*nxx*nzz*nz - 210*nyy*nyy*nz + (945.0/2.0)*nyy*nzz*nz - 105*nzz*nzz*nz + 70*nzz*nz - 15*nz);  // power=6
    mat[14][6] = FAR_REAL(525*nxx*nx*ny*nz - 1050*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[14][7] = FAR_REAL(-20*nxx*nxx*nx + 100*nxx*nx*nyy + 415*nxx*nx*nzz - 35*nxx*nx + 120*nx*nyy*nyy - 1020*nx*nyy*nzz + 435*nx*nzz*nzz - 315*nx*nzz + 30*nx);  // power=6
    mat[14][8] = FAR_REAL(315*nxx*ny*nzz - 35*nxx*ny - 1260*nyy*ny*nzz + 140*nyy*ny + 315*ny*nzz*nzz + 175*ny*nzz - 40*ny);  // power=6
    mat[14][9] = FAR_REAL((693.0/2.0)*nxx*nxx*nx*nz - 1386*nxx*nx*nyy*nz + (693.0/2.0)*nxx*nx*nzz*nz - 567.0/2.0*nxx*nx*nz + 378*nx*nyy*nz - 189.0/2.0*nx*nzz*nz + 42*nx*nz);  // power=7
    mat[14][10] = FAR_REAL((693.0/2.0)*nxx*nyy*ny*nz - 189.0/2.0*nxx*ny*nz - 1386*nyy*nyy*ny*nz + (693.0/2.0)*nyy*ny*nzz*nz + 1134*nyy*ny*nz - 189.0/2.0*ny*nzz*nz - 168*ny*nz);  // power=7
    mat[14][11] = FAR_REAL((693.0/2.0)*nxx*nzz*nzz - 189*nxx*nzz + (21.0/2.0)*nxx - 1386*nyy*nzz*nzz + 756*nyy*nzz - 42*nyy + (693.0/2.0)*nzz*nzz*nzz - 378*nzz*nzz + (189.0/2.0)*nzz - 3);  // power=7
    mat[14][12] = FAR_REAL((2079.0/2.0)*nxx*nxx*ny*nz - 4158*nxx*nyy*ny*nz + (2079.0/2.0)*nxx*ny*nzz*nz + (567.0/2.0)*nxx*ny*nz + 378*nyy*ny*nz - 189.0/2.0*ny*nzz*nz - 63*ny*nz);  // power=7
    mat[14][13] = FAR_REAL((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nzz + 378*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz - 756*nxx*nzz + (189.0/2.0)*nxx + 378*nyy*nzz - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
    mat[14][14] = FAR_REAL((2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 189.0/2.0*nxx*nzz - 4158*nyy*nyy*nzz + 378*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz + (3213.0/2.0)*nyy*nzz - 441.0/2.0*nyy - 189.0/2.0*nzz*nzz - 63*nzz + 45.0/2.0);  // power=7
    mat[14][15] = FAR_REAL(-252*nxx*nxx*nx*ny + 252*nxx*nx*nyy*ny + 3843*nxx*nx*ny*nzz - 189*nxx*nx*ny + 504*nx*nyy*nyy*ny - 5796*nx*nyy*ny*nzz + 4095*nx*ny*nzz*nzz - 1701*nx*ny*nzz + 126*nx*ny);  // power=7
    mat[14][16] = FAR_REAL((1287.0/8.0)*nxx*nxx*nxx*nz - 1287.0/2.0*nxx*nxx*nyy*nz - 6435.0/8.0*nxx*nxx*nzz*nz + (99.0/2.0)*nxx*nxx*nz + 3861*nxx*nyy*nzz*nz - 594*nxx*nyy*nz - 6435.0/8.0*nxx*nzz*nzz*nz + 693*nxx*nzz*nz - 81*nxx*nz - 1287.0/2.0*nyy*nzz*nzz*nz + 198*nyy*nzz*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + 27*nzz*nz);  // power=8
    mat[14][17] = FAR_REAL((1287.0/8.0)*nxx*nyy*nyy*nz - 3861.0/4.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + (1287.0/8.0)*nxx*nzz*nzz*nz - 99.0/2.0*nxx*nzz*nz - 1287.0/2.0*nyy*nyy*nyy*nz + (32175.0/8.0)*nyy*nyy*nzz*nz - 198*nyy*nyy*nz - 6435.0/4.0*nyy*nzz*nzz*nz - 1089.0/2.0*nyy*nzz*nz + (243.0/2.0)*nyy*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[14][18] = FAR_REAL((1053.0/2.0)*nxx*nxx*nx*ny*nz + 126*nxx*nx*nyy*ny*nz - 2808*nxx*nx*ny*nzz*nz + 297*nxx*nx*ny*nz - 3618*nx*nyy*nyy*ny*nz + (11835.0/2.0)*nx*nyy*ny*nzz*nz + (2943.0/2.0)*nx*nyy*ny*nz - 3334499999996666.0/999999999999.0*nx*ny*nzz*nzz*nz + (3861.0/2.0)*nx*ny*nzz*nz - 1107.0/2.0*nx*ny*nz);  // power=8
    mat[14][19] = FAR_REAL((2259.0/2.0)*nxx*nxx*nx*ny*nz - 3384*nxx*nx*nyy*ny*nz - 2097*nxx*nx*ny*nzz*nz + 396*nxx*nx*ny*nz - 1296*nx*nyy*nyy*ny*nz + 5130*nx*nyy*ny*nzz*nz + 594*nx*nyy*ny*nz - 6453.0/2.0*nx*ny*nzz*nzz*nz + 1782*nx*ny*nzz*nz - 378*nx*ny*nz);  // power=8
    mat[14][20] = FAR_REAL(-6*nxx*nxx*nxx*nx + 36*nxx*nxx*nx*nyy + (1287.0/2.0)*nxx*nxx*nx*nzz + 90*nxx*nxx*nx*nzz - 99.0/2.0*nxx*nxx*nx + 90*nxx*nx*nyy*nyy - 2574*nxx*nx*nyy*nzz - 900*nxx*nx*nyy*nzz + 198*nxx*nx*nyy - 297*nxx*nx*nzz + 27*nxx*nx + 48*nx*nyy*nyy*nyy - 990*nx*nyy*nyy*nzz + 1440*nx*nyy*nzz*nzz + 594*nx*nyy*nzz - 54*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz - 96*nx*nzz*nzz*nzz + (1485.0/2.0)*nx*nzz*nzz - 189*nx*nzz + 6*nx);  // power=8
    mat[14][21] = FAR_REAL((1287.0/2.0)*nxx*nyy*ny*nzz - 99.0/2.0*nxx*nyy*ny - 1287.0/2.0*nxx*ny*nzz*nzz + (297.0/2.0)*nxx*ny*nzz - 2574*nyy*nyy*ny*nzz + 198*nyy*nyy*ny + (3217499999996782.0/999999999999.0)*nyy*ny*nzz*nzz + (445499999999554.0/999999999999.0)*nyy*ny*nzz - 108*nyy*ny - 1287.0/2.0*ny*nzz*nzz*nzz + (99.0/2.0)*ny*nzz*nzz - 81*ny*nzz + 15*ny);  // power=8
    mat[14][22] = FAR_REAL((3861.0/4.0)*nxx*nxx*nyy*nz - 3861.0/4.0*nxx*nxx*nzz*nz + (297.0/2.0)*nxx*nxx*nz - 3861*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + (1287.0/2.0)*nxx*nzz*nz - 297.0/2.0*nxx*nz + 3861*nyy*nyy*nzz*nz - 594*nyy*nyy*nz - 1238187531831500.0/549755813889.0*nyy*nzz*nzz*nz - 693.0/2.0*nyy*nzz*nz + 189*nyy*nz + (1287.0/4.0)*nzz*nzz*nzz*nz - 297*nzz*nzz*nz + (243.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[14][23] = FAR_REAL((3861.0/2.0)*nxx*nxx*ny*nzz - 297.0/2.0*nxx*nxx*ny - 7722*nxx*nyy*ny*nzz + 594*nxx*nyy*ny + 1287*nxx*ny*nzz*nzz + 297*nxx*ny*nzz - 54*nxx*ny + 2574*nyy*ny*nzz*nzz - 594*nyy*ny*nzz - 1287.0/2.0*ny*nzz*nzz*nzz + (99.0/2.0)*ny*nzz*nzz + 54*ny*nzz);  // power=8
    mat[14][24] = FAR_REAL(-21.0/2.0*nxx*nxx*nxx*nx + 63*nxx*nxx*nx*nyy + (315.0/2.0)*nxx*nxx*nx*nzz + 378*nxx*nx*nyy*nyy + (315.0/2.0)*nxx*nx*nyy*nyy + (5877.0/2.0)*nxx*nx*nyy*nzz - 1575*nxx*nx*nyy*nzz - 801.0/2.0*nxx*nx*nyy - 1287.0/2.0*nxx*nx*nzz*nzz + (297.0/2.0)*nxx*nx*nzz + 18*nxx*nx + 378*nx*nyy*nyy*nyy + 84*nx*nyy*nyy*nyy - 6336*nx*nyy*nyy*nzz - 3465.0/2.0*nx*nyy*nyy*nzz - 36*nx*nyy*nyy + 1008*nx*nyy*nzz*nzz + 2520*nx*nyy*nzz*nzz + (3717.0/2.0)*nx*nyy*nzz - 27.0/2.0*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz - 168*nx*nzz*nzz*nzz + 891*nx*nzz*nzz - 387*nx*nzz + (15.0/2.0)*nx);  // power=8
    mat[14][25] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*nz + 1323*nxx*nx*nyy*nz - 63.0/2.0*nxx*nx*nzz*nz - 189.0/2.0*nxx*nx*nz - 3906*nx*nyy*nyy*nz + (2457.0/2.0)*nx*nyy*nzz*nz + (2961.0/2.0)*nx*nyy*nz - 63*nx*nzz*nzz*nz - 189.0/2.0*nx*nz);  // power=7
    mat[14][26] = FAR_REAL(-126*nxx*nxx*nx*nz + 882*nxx*nx*nyy*nz + (2331.0/2.0)*nxx*nx*nzz*nz - 567.0/2.0*nxx*nx*nz + 1008*nx*nyy*nyy*nz - 2898*nx*nyy*nzz*nz + (2583.0/2.0)*nx*nzz*nzz*nz - 2457.0/2.0*nx*nzz*nz + 252*nx*nz);  // power=7
    mat[14][27] = FAR_REAL((2079.0/2.0)*nxx*ny*nzz*nz - 567.0/2.0*nxx*ny*nz - 4158*nyy*ny*nzz*nz + 1134*nyy*ny*nz + (2079.0/2.0)*ny*nzz*nzz*nz + (189.0/2.0)*ny*nzz*nz - 189*ny*nz);  // power=7

    // Row 15 (target order 3)
    mat[15][0] = FAR_REAL(-75*nx*ny*nz);  // power=4
    mat[15][1] = FAR_REAL(-450*nxx*ny*nz + 75*nyy*ny*nz + 75*ny*nzz*nz);  // power=5
    mat[15][2] = FAR_REAL(75*nxx*nx*nz - 450*nx*nyy*nz + 75*nx*nzz*nz);  // power=5
    mat[15][3] = FAR_REAL(75*nxx*nx*ny + 75*nx*nyy*ny - 450*nx*ny*nzz);  // power=5
    mat[15][4] = FAR_REAL(-525*nxx*nx*ny*nz + (525.0/2.0)*nx*nyy*ny*nz + (525.0/2.0)*nx*ny*nzz*nz);  // power=6
    mat[15][5] = FAR_REAL(-525.0/2.0*nxx*nx*ny*nz - 525.0/2.0*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[15][6] = FAR_REAL(-1575*nxx*nyy*nz + 175*nxx*nz + 175*nyy*nz - 25*nz);  // power=6
    mat[15][7] = FAR_REAL(-1575*nxx*ny*nzz + 175*nxx*ny + 175*ny*nzz - 25*ny);  // power=6
    mat[15][8] = FAR_REAL(-25*nxx*nxx*nx + 125*nxx*nx*nyy + 125*nxx*nx*nzz + 150*nx*nyy*nyy - 1275*nx*nyy*nzz + 150*nx*nzz*nzz);  // power=6
    mat[15][9] = FAR_REAL(-3465.0/2.0*nxx*nxx*ny*nz + 945*nxx*ny*nz - 105.0/2.0*ny*nz);  // power=7
    mat[15][10] = FAR_REAL(210*nxx*nx*nyy*nz - 3045.0/2.0*nx*nyy*nyy*nz + 210*nx*nyy*nzz*nz + 735*nx*nyy*nz - 105.0/2.0*nx*nz);  // power=7
    mat[15][11] = FAR_REAL(-3465.0/2.0*nx*ny*nzz*nzz + 945*nx*ny*nzz - 105.0/2.0*nx*ny);  // power=7
    mat[15][12] = FAR_REAL(-10395.0/2.0*nxx*nx*nyy*nz + (945.0/2.0)*nxx*nx*nz + (2835.0/2.0)*nx*nyy*nz - 315.0/2.0*nx*nz);  // power=7
    mat[15][13] = FAR_REAL(-10395.0/2.0*nxx*nx*ny*nzz + (945.0/2.0)*nxx*nx*ny + (2835.0/2.0)*nx*ny*nzz - 315.0/2.0*nx*ny);  // power=7
    mat[15][14] = FAR_REAL(-315.0/2.0*nxx*nxx*nx*ny + (315.0/2.0)*nxx*nx*nyy*ny + (2205.0/2.0)*nxx*nx*ny*nzz + 315*nx*nyy*nyy*ny - 7245.0/2.0*nx*nyy*ny*nzz + 1260*nx*ny*nzz*nzz);  // power=7
    mat[15][15] = FAR_REAL(-10395*nxx*nyy*nzz + 945*nxx*nyy + 945*nxx*nzz + 945*nyy*nzz - 90);  // power=7
    mat[15][16] = FAR_REAL(-804374999995978.0/999999999995.0*nxx*nxx*nx*ny*nz + (19305.0/4.0)*nxx*nx*ny*nzz*nz - 495*nxx*nx*ny*nz - 804374999995978.0/999999999995.0*nx*ny*nzz*nzz*nz - 495*nx*ny*nzz*nz + 135*nx*ny*nz);  // power=8
    mat[15][17] = FAR_REAL((1215.0/4.0)*nxx*nxx*nx*ny*nz - 1575.0/4.0*nxx*nx*nyy*ny*nz - 2025.0/4.0*nxx*nx*ny*nzz*nz - 12015.0/8.0*nx*nyy*nyy*ny*nz + (13275.0/4.0)*nx*nyy*ny*nzz*nz + (2025.0/4.0)*nx*nyy*ny*nz - 12915.0/8.0*nx*ny*nzz*nzz*nz + (2475.0/4.0)*nx*ny*nzz*nz - 675.0/4.0*nx*ny*nz);  // power=8
    mat[15][18] = FAR_REAL(-3217499999996782.0/999999999999.0*nxx*nyy*nyy*nz + (19305.0/2.0)*nxx*nyy*nzz*nz - 1485.0/2.0*nxx*nyy*nz - 1485.0/2.0*nxx*nzz*nz + 135*nxx*nz + (495.0/2.0)*nyy*nyy*nz - 1485.0/2.0*nyy*nzz*nz + (135.0/2.0)*nyy*nz + (135.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[15][19] = FAR_REAL(-3217499999996782.0/999999999999.0*nxx*nxx*nyy*nz + (495.0/2.0)*nxx*nxx*nz + (19305.0/2.0)*nxx*nyy*nzz*nz - 1485.0/2.0*nxx*nyy*nz - 1485.0/2.0*nxx*nzz*nz + (135.0/2.0)*nxx*nz - 1485.0/2.0*nyy*nzz*nz + 135*nyy*nz + (135.0/2.0)*nzz*nz - 15*nz);  // power=8
    mat[15][20] = FAR_REAL(-3217499999996782.0/999999999999.0*nxx*nxx*ny*nzz + (495.0/2.0)*nxx*nxx*ny + (3217499999996782.0/999999999999.0)*nxx*ny*nzz*nzz - 135.0/2.0*nxx*ny - 495.0/2.0*ny*nzz*nzz + (135.0/2.0)*ny*nzz);  // power=8
    mat[15][21] = FAR_REAL(-15.0/2.0*nxx*nxx*nxx*nx + 45*nxx*nxx*nx*nyy + (225.0/2.0)*nxx*nxx*nx*nzz + (315.0/2.0)*nxx*nx*nyy*nyy + (225.0/2.0)*nxx*nx*nyy*nyy + 420*nxx*nx*nyy*nzz - 1125*nxx*nx*nyy*nzz - 105*nxx*nx*nyy + (15.0/2.0)*nxx*nx + (315.0/2.0)*nx*nyy*nyy*nyy + 60*nx*nyy*nyy*nyy - 2640*nx*nyy*nyy*nzz - 2475.0/2.0*nx*nyy*nyy*nzz - 15*nx*nyy*nyy + 420*nx*nyy*nzz*nzz + 1800*nx*nyy*nzz*nzz + 960*nx*nyy*nzz - 45.0/2.0*nx*nyy - 120*nx*nzz*nzz*nzz - 60*nx*nzz);  // power=8
    mat[15][22] = FAR_REAL((1215.0/4.0)*nxx*nxx*nx*ny*nz - 10665.0/2.0*nxx*nx*nyy*ny*nz + 4320*nxx*nx*ny*nzz*nz - 810*nx*nyy*nyy*ny*nz + (12825.0/4.0)*nx*nyy*ny*nzz*nz + (4455.0/4.0)*nx*nyy*ny*nz - 1329721874839181.0/549755813887.0*nx*ny*nzz*nzz*nz + (123749999999876.0/999999999999.0)*nx*ny*nzz*nz - 675.0/4.0*nx*ny*nz);  // power=8
    mat[15][23] = FAR_REAL(-15.0/2.0*nxx*nxx*nxx*nx + 45*nxx*nxx*nx*nyy + (225.0/2.0)*nxx*nxx*nx*nzz + (225.0/2.0)*nxx*nx*nyy*nyy - 19305.0/2.0*nxx*nx*nyy*nzz - 1125*nxx*nx*nyy*nzz + (1485.0/2.0)*nxx*nx*nyy + (1485.0/2.0)*nxx*nx*nzz - 135.0/2.0*nxx*nx + 60*nx*nyy*nyy*nyy - 2475.0/2.0*nx*nyy*nyy*nzz + 1800*nx*nyy*nzz*nzz + (4455.0/2.0)*nx*nyy*nzz - 405.0/2.0*nx*nyy - 120*nx*nzz*nzz*nzz - 405.0/2.0*nx*nzz + (45.0/2.0)*nx);  // power=8
    mat[15][24] = FAR_REAL(-19305.0/2.0*nxx*nyy*ny*nzz + (1485.0/2.0)*nxx*nyy*ny + (3217499999996782.0/999999999999.0)*nxx*ny*nzz*nzz + (1485.0/2.0)*nxx*ny*nzz - 135*nxx*ny + (1485.0/2.0)*nyy*ny*nzz - 135.0/2.0*nyy*ny - 495.0/2.0*ny*nzz*nzz - 135.0/2.0*ny*nzz + 15*ny);  // power=8
    mat[15][25] = FAR_REAL(-10395.0/2.0*nxx*nyy*ny*nz + (2835.0/2.0)*nxx*ny*nz + (945.0/2.0)*nyy*ny*nz - 315.0/2.0*ny*nz);  // power=7
    mat[15][26] = FAR_REAL(-10395.0/2.0*nxx*ny*nzz*nz + (2835.0/2.0)*nxx*ny*nz + (945.0/2.0)*ny*nzz*nz - 315.0/2.0*ny*nz);  // power=7
    mat[15][27] = FAR_REAL(-315.0/2.0*nxx*nxx*nx*nz + (2205.0/2.0)*nxx*nx*nyy*nz + (315.0/2.0)*nxx*nx*nzz*nz + 1260*nx*nyy*nyy*nz - 7245.0/2.0*nx*nyy*nzz*nz + 315*nx*nzz*nzz*nz);  // power=7

    // Row 16 (target order 4)
    mat[16][0] = FAR_REAL(525*nxx*nxx - 3150*nxx*nyy - 3150*nxx*nzz + 450*nxx + 450*nyy + 450*nzz - 135);  // power=5
    mat[16][1] = FAR_REAL(4725*nxx*nxx*nx - 28350*nxx*nx*nyy - 28350*nxx*nx*nzz + 1050*nxx*nx + 9450*nx*nyy + 9450*nx*nzz - 1575*nx);  // power=6
    mat[16][2] = FAR_REAL(4725*nxx*nxx*ny - 28350*nxx*nyy*ny - 28350*nxx*ny*nzz + 9450*nxx*ny + 3150*nyy*ny + 3150*ny*nzz - 1575*ny);  // power=6
    mat[16][3] = FAR_REAL(4725*nxx*nxx*nz - 28350*nxx*nyy*nz - 28350*nxx*nzz*nz + 9450*nxx*nz + 3150*nyy*nz + 3150*nzz*nz - 1575*nz);  // power=6
    mat[16][4] = FAR_REAL(5775*nxx*nxx*nxx - 75075.0/2.0*nxx*nxx*nyy - 75075.0/2.0*nxx*nxx*nzz - 1050*nxx*nxx + 17325*nxx*nyy*nyy + 34650*nxx*nyy*nzz + 7875*nxx*nyy + 17325*nxx*nzz*nzz + 7875*nxx*nzz - 875*nxx - 1575*nyy*nyy - 3150*nyy*nzz - 175.0/2.0*nyy - 1575*nzz*nzz - 175.0/2.0*nzz + 350);  // power=7
    mat[16][5] = FAR_REAL((5775.0/2.0)*nxx*nxx*nxx - 28875.0/2.0*nxx*nxx*nyy - 23100*nxx*nxx*nzz - 525*nxx*nxx - 17325*nxx*nyy*nyy + 17325*nxx*nyy*nzz + 15750*nxx*nyy + 34650*nxx*nzz*nzz - 7875*nxx*nzz - 875.0/2.0*nxx + 1575*nyy*nyy - 1575*nyy*nzz - 2975.0/2.0*nyy - 3150*nzz*nzz + 1400*nzz + 175);  // power=7
    mat[16][6] = FAR_REAL(17325*nxx*nxx*nx*ny - 103950*nxx*nx*nyy*ny - 103950*nxx*nx*ny*nzz + 22050*nxx*nx*ny + 28350*nx*nyy*ny + 28350*nx*ny*nzz - 9975*nx*ny);  // power=7
    mat[16][7] = FAR_REAL(17325*nxx*nxx*nx*nz - 103950*nxx*nx*nyy*nz - 103950*nxx*nx*nzz*nz + 22050*nxx*nx*nz + 28350*nx*nyy*nz + 28350*nx*nzz*nz - 9975*nx*nz);  // power=7
    mat[16][8] = FAR_REAL(17325*nxx*nxx*ny*nz - 103950*nxx*nyy*ny*nz - 103950*nxx*ny*nzz*nz + 47250*nxx*ny*nz + 9450*nyy*ny*nz + 9450*ny*nzz*nz - 5775*ny*nz);  // power=7
    mat[16][9] = FAR_REAL((45045.0/2.0)*nxx*nxx*nxx*nx - 135135*nxx*nxx*nx*nyy - 135135*nxx*nxx*nx*nzz - 31185.0/2.0*nxx*nxx*nx + 103950*nxx*nx*nyy + 103950*nxx*nx*nzz - 4725.0/2.0*nxx*nx - 14175*nx*nyy - 14175*nx*nzz + (2625.0/2.0)*nx);  // power=8
    mat[16][10] = FAR_REAL((45045.0/2.0)*nxx*nxx*nyy*ny - 10395.0/2.0*nxx*nxx*ny - 135135*nxx*nyy*nyy*ny - 135135*nxx*nyy*ny*nzz + 103950*nxx*nyy*ny + 31185*nxx*ny*nzz - 14175*nxx*ny + 10395*nyy*nyy*ny + 10395*nyy*ny*nzz - 19845.0/2.0*nyy*ny - 2835*ny*nzz + (3465.0/2.0)*ny);  // power=8
    mat[16][11] = FAR_REAL((45045.0/2.0)*nxx*nxx*nzz*nz - 10395.0/2.0*nxx*nxx*nz - 135135*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz - 135135*nxx*nzz*nzz*nz + 103950*nxx*nzz*nz - 14175*nxx*nz + 10395*nyy*nzz*nz - 2835*nyy*nz + 10395*nzz*nzz*nz - 19845.0/2.0*nzz*nz + (3465.0/2.0)*nz);  // power=8
    mat[16][12] = FAR_REAL((135135.0/2.0)*nxx*nxx*nxx*ny - 405405*nxx*nxx*nyy*ny - 405405*nxx*nxx*ny*nzz + (93555.0/2.0)*nxx*nxx*ny + 187110*nxx*nyy*ny + 187110*nxx*ny*nzz - 93555.0/2.0*nxx*ny - 8505*nyy*ny - 8505*ny*nzz + (5985.0/2.0)*ny);  // power=8
    mat[16][13] = FAR_REAL((135135.0/2.0)*nxx*nxx*nxx*nz - 405405*nxx*nxx*nyy*nz - 405405*nxx*nxx*nzz*nz + (93555.0/2.0)*nxx*nxx*nz + 187110*nxx*nyy*nz + 187110*nxx*nzz*nz - 93555.0/2.0*nxx*nz - 8505*nyy*nz - 8505*nzz*nz + (5985.0/2.0)*nz);  // power=8
    mat[16][14] = FAR_REAL((135135.0/2.0)*nxx*nxx*nyy*nz - 10395.0/2.0*nxx*nxx*nz - 405405*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 249480*nxx*nyy*nz + 31185*nxx*nzz*nz - 14175*nxx*nz + 31185*nyy*nyy*nz + 31185*nyy*nzz*nz - 48195.0/2.0*nyy*nz - 2835*nzz*nz + (3465.0/2.0)*nz);  // power=8
    mat[16][15] = FAR_REAL(135135*nxx*nxx*nx*ny*nz - 810810*nxx*nx*nyy*ny*nz - 810810*nxx*nx*ny*nzz*nz + 270270*nxx*nx*ny*nz + 187110*nx*nyy*ny*nz + 187110*nx*ny*nzz*nz - 87885*nx*ny*nz);  // power=8
    mat[16][16] = FAR_REAL((6633147491942405.0/549755813888.0)*nxx*nxx*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy - 2487430309478401.0/17179869184.0*nxx*nxx*nxx*nzz - 2211049163980803.0/274877906944.0*nxx*nxx*nxx + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (173745.0/4.0)*nxx*nxx*nyy + (7669576787558399.0/17179869184.0)*nxx*nxx*nzz*nzz - 1077886467440639.0/17179869184.0*nxx*nxx*nzz + (1479702132817919.0/274877906944.0)*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nyy*nzz*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nzz - 4455*nxx*nyy - 4974860618956799.0/68719476736.0*nxx*nzz*nzz*nzz - 405405.0/4.0*nxx*nzz*nzz + (2984916371374079.0/68719476736.0)*nxx*nzz - 2159999999999999.0/1000000000000.0*nxx + (3979888495160613.0/824633720831.0)*nyy*nzz*nzz + 4455*nyy*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz + (892923700839329.0/962072674305.0)*nzz*nzz - 1620*nzz + 105);  // power=9
    mat[16][17] = FAR_REAL((6633147491942405.0/549755813888.0)*nxx*nxx*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nyy*nzz + (1.0/1000000000000.0)*nxx*nxx*nyy + (6633147491942405.0/549755813888.0)*nxx*nxx*nzz*nzz + (1.0/1000000000000.0)*nxx*nxx*nzz - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nyy*nzz + (173745.0/4.0)*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 4974860618956799.0/68719476736.0*nxx*nzz*nzz*nzz + (173745.0/4.0)*nxx*nzz*nzz + (3979888495160613.0/824633720831.0)*nyy*nyy*nyy - 96525.0/4.0*nyy*nyy*nzz - 3393110063189446.0/962072674305.0*nyy*nyy - 96525.0/4.0*nyy*nzz*nzz + (5816760108318719.0/274877906944.0)*nyy*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz - 3393110063189446.0/962072674305.0*nzz*nzz);  // power=9
    mat[16][18] = FAR_REAL((6633147491942405.0/137438953472.0)*nxx*nxx*nx*nyy*ny - 1243715154739201.0/8589934592.0*nxx*nxx*nx*ny*nzz + (1.0/879609302221.0)*nxx*nxx*nx*ny - 4974860618956799.0/17179869184.0*nxx*nx*nyy*nyy*ny + (4974860618956799.0/8589934592.0)*nxx*nx*nyy*ny*nzz + (8401986823127037.0/68719476736.0)*nxx*nx*nyy*ny + (7462290928435199.0/8589934592.0)*nxx*nx*ny*nzz*nzz - 6301490117345279.0/17179869184.0*nxx*nx*ny*nzz + (2.0/879609302221.0)*nxx*nx*ny + (3979888495165439.0/68719476736.0)*nx*nyy*nyy*ny - 7959776990330879.0/68719476736.0*nx*nyy*ny*nzz - 8776164373954557.0/274877906944.0*nx*nyy*ny - 173745*nx*ny*nzz*nzz + (6582123280465919.0/68719476736.0)*nx*ny*nzz - 1.0/1000000000000.0*nx*ny);  // power=9
    mat[16][19] = FAR_REAL((6633147491942405.0/137438953472.0)*nxx*nxx*nxx*nx*ny - 4974860618956799.0/17179869184.0*nxx*nxx*nx*nyy*ny - 7462290928435201.0/17179869184.0*nxx*nxx*nx*ny*nzz + (2653258996776955.0/137438953472.0)*nxx*nxx*nx*ny + (7462290928435199.0/8589934592.0)*nxx*nx*nyy*ny*nzz + (4643203244359679.0/34359738368.0)*nxx*nx*nyy*ny + (7462290928435199.0/8589934592.0)*nxx*nx*ny*nzz*nzz - 3979888495165439.0/17179869184.0*nxx*nx*ny*nzz - 3877840072212479.0/274877906944.0*nxx*nx*ny - 173745*nx*nyy*ny*nzz - 4898324301742079.0/549755813888.0*nx*nyy*ny - 173745*nx*ny*nzz*nzz + (173745.0/2.0)*nx*ny*nzz - 540000000000001.0/1000000000000.0*nx*ny);  // power=9
    mat[16][20] = FAR_REAL((6633147491942405.0/137438953472.0)*nxx*nxx*nxx*nx*nz - 4974860618956799.0/17179869184.0*nxx*nxx*nx*nyy*nz - 675675.0/2.0*nxx*nxx*nx*nzz*nz + (2653258996776955.0/137438953472.0)*nxx*nxx*nx*nz + (4974860618956799.0/17179869184.0)*nxx*nx*nyy*nzz*nz + (4643203244359679.0/34359738368.0)*nxx*nx*nyy*nz + (4974860618956799.0/17179869184.0)*nxx*nx*nzz*nzz*nz + (3537678662369285.0/274877906944.0)*nxx*nx*nzz*nz - 3877840072212479.0/274877906944.0*nxx*nx*nz - 3979888495165439.0/68719476736.0*nx*nyy*nzz*nz - 4898324301742079.0/549755813888.0*nx*nyy*nz - 3979888495165439.0/68719476736.0*nx*nzz*nzz*nz + (6327002223083519.0/274877906944.0)*nx*nzz*nz - 540000000000001.0/1000000000000.0*nx*nz);  // power=9
    mat[16][21] = FAR_REAL((6633147491942405.0/137438953472.0)*nxx*nxx*nyy*ny*nz - 6633147491942405.0/137438953472.0*nxx*nxx*ny*nzz*nz - 4974860618956799.0/17179869184.0*nxx*nyy*nyy*ny*nz + (2984916371374079.0/17179869184.0)*nxx*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nxx*ny*nzz*nzz*nz - 2984916371374079.0/17179869184.0*nxx*ny*nzz*nz + (5306517993553919.0/274877906944.0)*nyy*nyy*ny*nz - 28215.0/2.0*nyy*ny*nz - 5306517993553919.0/274877906944.0*ny*nzz*nzz*nz + (28215.0/2.0)*ny*nzz*nz);  // power=9
    mat[16][22] = FAR_REAL((1243715154739201.0/17179869184.0)*nxx*nxx*nxx*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nxx*nzz - 7462290928435199.0/17179869184.0*nxx*nxx*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nyy*nzz + (7296462241136635.0/68719476736.0)*nxx*nxx*nyy + (7876862646681599.0/17179869184.0)*nxx*nxx*nzz*nzz - 7296462241136635.0/68719476736.0*nxx*nxx*nzz + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 6582123280465919.0/137438953472.0*nxx*nyy - 4974860618956799.0/34359738368.0*nxx*nzz*nzz*nzz - 57915*nxx*nzz*nzz + (6582123280465919.0/137438953472.0)*nxx*nzz - 7959776990330879.0/274877906944.0*nyy*nyy*nzz - 4455*nyy*nyy - 19305*nyy*nzz*nzz + (5816760108318719.0/274877906944.0)*nyy*nzz + 1620*nyy + (5306517993553919.0/549755813888.0)*nzz*nzz*nzz - 2143016882014758.0/824633720833.0*nzz*nzz - 1620*nzz);  // power=9
    mat[16][23] = FAR_REAL((1243715154739201.0/8589934592.0)*nxx*nxx*nxx*ny*nz - 7462290928435199.0/8589934592.0*nxx*nxx*nyy*ny*nz - 7876862646681599.0/8589934592.0*nxx*nxx*ny*nzz*nz + (7296462241136635.0/34359738368.0)*nxx*nxx*ny*nz + (4974860618956799.0/17179869184.0)*nxx*nyy*ny*nzz*nz + (4974860618956799.0/17179869184.0)*nxx*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nxx*ny*nzz*nzz*nz + (3979888495165441.0/34359738368.0)*nxx*ny*nzz*nz - 6582123280465919.0/68719476736.0*nxx*ny*nz - 5306517993553919.0/274877906944.0*nyy*ny*nzz*nz - 8910*nyy*ny*nz - 5306517993553919.0/274877906944.0*ny*nzz*nzz*nz + (2857355842682879.0/549755813888.0)*ny*nzz*nz + 3240*ny*nz);  // power=9
    mat[16][24] = FAR_REAL((1243715154739201.0/8589934592.0)*nxx*nxx*nx*nyy*nz - 6633147491942405.0/137438953472.0*nxx*nxx*nx*nzz*nz - 1.0/879609302221.0*nxx*nxx*nx*nz - 7462290928435199.0/8589934592.0*nxx*nx*nyy*nyy*nz - 4974860618956799.0/8589934592.0*nxx*nx*nyy*nzz*nz + (6301490117345279.0/17179869184.0)*nxx*nx*nyy*nz + (4974860618956799.0/17179869184.0)*nxx*nx*nzz*nzz*nz - 8401986823127037.0/68719476736.0*nxx*nx*nzz*nz - 2.0/879609302221.0*nxx*nx*nz + 173745*nx*nyy*nyy*nz + (7959776990330879.0/68719476736.0)*nx*nyy*nzz*nz - 6582123280465919.0/68719476736.0*nx*nyy*nz - 3979888495165439.0/68719476736.0*nx*nzz*nzz*nz + (8776164373954557.0/274877906944.0)*nx*nzz*nz + (1.0/1000000000000.0)*nx*nz);  // power=9
    mat[16][25] = FAR_REAL((135135.0/2.0)*nxx*nxx*nx*nyy - 10395.0/2.0*nxx*nxx*nx - 405405*nxx*nx*nyy*nyy - 405405*nxx*nx*nyy*nzz + 166320*nxx*nx*nyy + 31185*nxx*nx*nzz - 6615*nxx*nx + 93555*nx*nyy*nyy + 93555*nx*nyy*nzz - 104895.0/2.0*nx*nyy - 8505*nx*nzz + (5985.0/2.0)*nx);  // power=8
    mat[16][26] = FAR_REAL((135135.0/2.0)*nxx*nxx*nx*nzz - 10395.0/2.0*nxx*nxx*nx - 405405*nxx*nx*nyy*nzz + 31185*nxx*nx*nyy - 405405*nxx*nx*nzz*nzz + 166320*nxx*nx*nzz - 6615*nxx*nx + 93555*nx*nyy*nzz - 8505*nx*nyy + 93555*nx*nzz*nzz - 104895.0/2.0*nx*nzz + (5985.0/2.0)*nx);  // power=8
    mat[16][27] = FAR_REAL((135135.0/2.0)*nxx*nxx*ny*nzz - 10395.0/2.0*nxx*nxx*ny - 405405*nxx*nyy*ny*nzz + 31185*nxx*nyy*ny - 405405*nxx*ny*nzz*nzz + 249480*nxx*ny*nzz - 14175*nxx*ny + 31185*nyy*ny*nzz - 2835*nyy*ny + 31185*ny*nzz*nzz - 48195.0/2.0*ny*nzz + (3465.0/2.0)*ny);  // power=8

    // Row 17 (target order 4)
    mat[17][0] = FAR_REAL(-3150*nxx*nyy + 450*nxx + 525*nyy*nyy - 3150*nyy*nzz + 450*nyy + 450*nzz - 135);  // power=5
    mat[17][1] = FAR_REAL(-450*nxx*nxx*nx - 26100*nxx*nx*nyy + 2250*nxx*nx*nzz + 3150*nxx*nx + (5442582557493677.0/733007751851.0)*nx*nyy*nyy - 22950*nx*nyy*nzz + 6300*nx*nyy + 2700*nx*nzz*nzz - 1125*nx);  // power=6
    mat[17][2] = FAR_REAL(-28350*nxx*nyy*ny + 9450*nxx*ny + 4725*nyy*nyy*ny - 28350*nyy*ny*nzz + 1050*nyy*ny + 9450*ny*nzz - 1575*ny);  // power=6
    mat[17][3] = FAR_REAL(-28350*nxx*nyy*nz + 3150*nxx*nz + 4725*nyy*nyy*nz - 28350*nyy*nzz*nz + 9450*nyy*nz + 3150*nzz*nz - 1575*nz);  // power=6
    mat[17][4] = FAR_REAL(-34650*nxx*nxx*nyy + 3150*nxx*nxx + 23100*nxx*nyy*nyy - 17325*nxx*nyy*nzz + 7875*nxx*nyy + 1575*nxx*nzz - 1400*nxx - 5775.0/2.0*nyy*nyy*nyy + (28875.0/2.0)*nyy*nyy*nzz + 525*nyy*nyy + 17325*nyy*nzz*nzz - 15750*nyy*nzz + (875.0/2.0)*nyy - 1575*nzz*nzz + (2975.0/2.0)*nzz - 175);  // power=7
    mat[17][5] = FAR_REAL(-17325*nxx*nxx*nyy + 1575*nxx*nxx - 28875.0/2.0*nxx*nyy*nyy + 17325*nxx*nyy*nzz + 15750*nxx*nyy - 1575*nxx*nzz - 2975.0/2.0*nxx + (5775.0/2.0)*nyy*nyy*nyy - 23100*nyy*nyy*nzz - 525*nyy*nyy + 34650*nyy*nzz*nzz - 7875*nyy*nzz - 875.0/2.0*nyy - 3150*nzz*nzz + 1400*nzz + 175);  // power=7
    mat[17][6] = FAR_REAL(-3150*nxx*nxx*nx*ny - 100800*nxx*nx*nyy*ny + 22050*nxx*nx*ny*nzz + 28350*nxx*nx*ny + 23625*nx*nyy*nyy*ny - 72450*nx*nyy*ny*nzz + 12600*nx*nyy*ny + 25200*nx*ny*nzz*nzz - 6825*nx*ny);  // power=7
    mat[17][7] = FAR_REAL(-3150*nxx*nxx*nx*nz - 84000*nxx*nx*nyy*nz + 3150*nxx*nx*nzz*nz + 9450*nxx*nx*nz + 40425*nx*nyy*nyy*nz - 74550*nx*nyy*nzz*nz + 21000*nx*nyy*nz + 6300*nx*nzz*nzz*nz - 2625*nx*nz);  // power=7
    mat[17][8] = FAR_REAL(-103950*nxx*nyy*ny*nz + 28350*nxx*ny*nz + 17325*nyy*nyy*ny*nz - 103950*nyy*ny*nzz*nz + 22050*nyy*ny*nz + 28350*ny*nzz*nz - 9975*ny*nz);  // power=7
    mat[17][9] = FAR_REAL(-135135*nxx*nxx*nx*nyy + 10395*nxx*nxx*nx + (45045.0/2.0)*nxx*nx*nyy*nyy - 135135*nxx*nx*nyy*nzz + 103950*nxx*nx*nyy + 10395*nxx*nx*nzz - 19845.0/2.0*nxx*nx - 10395.0/2.0*nx*nyy*nyy + 31185*nx*nyy*nzz - 14175*nx*nyy - 2835*nx*nzz + (3465.0/2.0)*nx);  // power=8
    mat[17][10] = FAR_REAL(-135135*nxx*nyy*nyy*ny + 103950*nxx*nyy*ny - 14175*nxx*ny + (45045.0/2.0)*nyy*nyy*nyy*ny - 135135*nyy*nyy*ny*nzz - 31185.0/2.0*nyy*nyy*ny + 103950*nyy*ny*nzz - 4725.0/2.0*nyy*ny - 14175*ny*nzz + (2625.0/2.0)*ny);  // power=8
    mat[17][11] = FAR_REAL(-135135*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 10395*nxx*nzz*nz - 2835*nxx*nz + (45045.0/2.0)*nyy*nyy*nzz*nz - 10395.0/2.0*nyy*nyy*nz - 135135*nyy*nzz*nzz*nz + 103950*nyy*nzz*nz - 14175*nyy*nz + 10395*nzz*nzz*nz - 19845.0/2.0*nzz*nz + (3465.0/2.0)*nz);  // power=8
    mat[17][12] = FAR_REAL(-405405*nxx*nxx*nyy*ny + 93555*nxx*nxx*ny + (135135.0/2.0)*nxx*nyy*nyy*ny - 405405*nxx*nyy*ny*nzz + 166320*nxx*nyy*ny + 93555*nxx*ny*nzz - 104895.0/2.0*nxx*ny - 10395.0/2.0*nyy*nyy*ny + 31185*nyy*ny*nzz - 6615*nyy*ny - 8505*ny*nzz + (5985.0/2.0)*ny);  // power=8
    mat[17][13] = FAR_REAL(-405405*nxx*nxx*nyy*nz + 31185*nxx*nxx*nz + (135135.0/2.0)*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 249480*nxx*nyy*nz + 31185*nxx*nzz*nz - 48195.0/2.0*nxx*nz - 10395.0/2.0*nyy*nyy*nz + 31185*nyy*nzz*nz - 14175*nyy*nz - 2835*nzz*nz + (3465.0/2.0)*nz);  // power=8
    mat[17][14] = FAR_REAL(-405405*nxx*nyy*nyy*nz + 187110*nxx*nyy*nz - 8505*nxx*nz + (135135.0/2.0)*nyy*nyy*nyy*nz - 405405*nyy*nyy*nzz*nz + (93555.0/2.0)*nyy*nyy*nz + 187110*nyy*nzz*nz - 93555.0/2.0*nyy*nz - 8505*nzz*nz + (5985.0/2.0)*nz);  // power=8
    mat[17][15] = FAR_REAL(-51030*nxx*nxx*nx*ny*nz - 744660*nxx*nx*nyy*ny*nz + 85050*nxx*nx*ny*nzz*nz + 187110*nxx*nx*ny*nz + 252315*nx*nyy*nyy*ny*nz - 557550*nx*nyy*ny*nzz*nz + 102060*nx*nyy*ny*nz + 136080*nx*ny*nzz*nzz*nz - 36855*nx*ny*nz);  // power=8
    mat[17][16] = FAR_REAL(-4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy + (3979888495160613.0/824633720831.0)*nxx*nxx*nxx + (6633147491942405.0/549755813888.0)*nxx*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nxx*nyy*nzz + (173745.0/4.0)*nxx*nxx*nyy - 96525.0/4.0*nxx*nxx*nzz - 3393110063189446.0/962072674305.0*nxx*nxx - 1243715154739201.0/17179869184.0*nxx*nyy*nyy*nzz + (1.0/1000000000000.0)*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 96525.0/4.0*nxx*nzz*nzz + (5816760108318719.0/274877906944.0)*nxx*nzz + (6633147491942405.0/549755813888.0)*nyy*nyy*nzz*nzz + (1.0/1000000000000.0)*nyy*nyy*nzz - 4974860618956799.0/68719476736.0*nyy*nzz*nzz*nzz + (173745.0/4.0)*nyy*nzz*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz - 3393110063189446.0/962072674305.0*nzz*nzz);  // power=9
    mat[17][17] = FAR_REAL(-4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (173745.0/4.0)*nxx*nyy*nyy - 4974860618956799.0/68719476736.0*nxx*nyy*nzz*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nzz - 4455*nxx*nyy + (3979888495160613.0/824633720831.0)*nxx*nzz*nzz + 4455*nxx*nzz + (6633147491942405.0/549755813888.0)*nyy*nyy*nyy*nyy - 2487430309478401.0/17179869184.0*nyy*nyy*nyy*nzz - 2211049163980803.0/274877906944.0*nyy*nyy*nyy + (7669576787558399.0/17179869184.0)*nyy*nyy*nzz*nzz - 1077886467440639.0/17179869184.0*nyy*nyy*nzz + (1479702132817919.0/274877906944.0)*nyy*nyy - 4974860618956799.0/68719476736.0*nyy*nzz*nzz*nzz - 405405.0/4.0*nyy*nzz*nzz + (2984916371374079.0/68719476736.0)*nyy*nzz - 2159999999999999.0/1000000000000.0*nyy + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz + (892923700839329.0/962072674305.0)*nzz*nzz - 1620*nzz + 105);  // power=9
    mat[17][18] = FAR_REAL(3780*nxx*nxx*nx*nyy*ny - 9001134760919039.0/34359738368.0*nxx*nx*nyy*nyy*ny + 801765*nxx*nx*nyy*ny*nzz + (8033994025205759.0/68719476736.0)*nxx*nx*nyy*ny - 173745*nxx*nx*ny*nzz*nzz - 93555*nxx*nx*ny*nzz - 2616150479339519.0/274877906944.0*nxx*nx*ny + (4953987077898241.0/68719476736.0)*nx*nyy*nyy*nyy*ny - 481275*nx*nyy*nyy*ny*nzz - 2606873349980165.0/137438953472.0*nx*nyy*nyy*ny + 624240*nx*nyy*ny*nzz*nzz - 6540376198348799.0/68719476736.0*nx*nyy*ny*nzz - 257285720899607.0/952910077406.0*nx*nyy*ny - 173745*nx*ny*nzz*nzz*nzz + (5510614839459839.0/68719476736.0)*nx*ny*nzz*nzz + (5343626510992571.0/879609302221.0)*nx*ny*nzz + (47498902319879.0/703687441776.0)*nx*ny);  // power=9
    mat[17][19] = FAR_REAL(-285795*nxx*nxx*nx*nyy*ny + (3979888495165439.0/68719476736.0)*nxx*nxx*nx*ny + (8590621786767361.0/137438953472.0)*nxx*nx*nyy*nyy*ny + 463590*nxx*nx*nyy*ny*nzz + (7984516001955835.0/68719476736.0)*nxx*nx*nyy*ny - 173745*nxx*nx*ny*nzz*nzz - 35640*nxx*nx*ny*nzz - 4749890231992319.0/137438953472.0*nxx*nx*ny + (2875910101401601.0/274877906944.0)*nx*nyy*nyy*nyy*ny - 8715863033118721.0/34359738368.0*nx*nyy*nyy*ny*nzz - 25515.0/2.0*nx*nyy*nyy*ny + (4944709948538881.0/8589934592.0)*nx*nyy*ny*nzz*nzz - 5826037237678079.0/34359738368.0*nx*nyy*ny*nzz - 296868139499249.0/879609302220.0*nx*nyy*ny - 173745*nx*ny*nzz*nzz*nzz + (5510614839459839.0/68719476736.0)*nx*ny*nzz*nzz + (7124835347988479.0/549755813888.0)*nx*ny*nzz + (1447232180062792.0/549755813889.0)*nx*ny);  // power=9
    mat[17][20] = FAR_REAL(-2025*nxx*nxx*nxx*nx*nz - 4985683936542719.0/17179869184.0*nxx*nxx*nx*nyy*nz + 16200*nxx*nxx*nx*nyy*nz + 8775*nxx*nxx*nx*nzz*nz + (5306517993553919.0/274877906944.0)*nxx*nxx*nx*nz + (7304193182269445.0/137438953472.0)*nxx*nx*nyy*nyy*nz + (5287963734835199.0/137438953472.0)*nxx*nx*nyy*nyy*nz + (1212211569623039.0/274877906944.0)*nxx*nx*nyy*nzz*nz - 8998815478579199.0/68719476736.0*nxx*nx*nyy*nzz*nz + 170910*nxx*nx*nyy*nz + 4320*nxx*nx*nzz*nzz*nz - 7644354592112639.0/549755813888.0*nxx*nx*nz + (11025.0/2.0)*nx*nyy*nyy*nyy*nz + (5566277615615999.0/274877906944.0)*nx*nyy*nyy*nyy*nz - 5182822935429121.0/137438953472.0*nx*nyy*nyy*nzz*nz - 139725*nx*nyy*nyy*nzz*nz - 1233858204794881.0/137438953472.0*nx*nyy*nyy*nz + (1385384650997761.0/274877906944.0)*nx*nyy*nzz*nzz*nz + (8460741975736319.0/68719476736.0)*nx*nyy*nzz*nzz*nz + (2968681394995201.0/274877906944.0)*nx*nyy*nzz*nz - 5114790653460479.0/274877906944.0*nx*nyy*nz - 6479999999999999.0/1000000000000.0*nx*nzz*nzz*nzz*nz - 540*nx*nzz*nz + (3645.0/2.0)*nx*nz);  // power=9
    mat[17][21] = FAR_REAL(-4974860618956799.0/17179869184.0*nxx*nyy*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nxx*nyy*ny*nzz*nz + (4643203244359679.0/34359738368.0)*nxx*nyy*ny*nz - 3979888495165439.0/68719476736.0*nxx*ny*nzz*nz - 4898324301742079.0/549755813888.0*nxx*ny*nz + (6633147491942405.0/137438953472.0)*nyy*nyy*nyy*ny*nz - 675675.0/2.0*nyy*nyy*ny*nzz*nz + (2653258996776955.0/137438953472.0)*nyy*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nyy*ny*nzz*nzz*nz + (3537678662369285.0/274877906944.0)*nyy*ny*nzz*nz - 3877840072212479.0/274877906944.0*nyy*ny*nz - 3979888495165439.0/68719476736.0*ny*nzz*nzz*nz + (6327002223083519.0/274877906944.0)*ny*nzz*nz - 540000000000001.0/1000000000000.0*ny*nz);  // power=9
    mat[17][22] = FAR_REAL(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nxx*nzz - 4455*nxx*nxx + (1243715154739201.0/17179869184.0)*nxx*nyy*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nyy*nyy*nzz + (7296462241136635.0/68719476736.0)*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 6582123280465919.0/137438953472.0*nxx*nyy - 19305*nxx*nzz*nzz + (5816760108318719.0/274877906944.0)*nxx*nzz + 1620*nxx - 1243715154739201.0/17179869184.0*nyy*nyy*nyy*nzz + (7876862646681599.0/17179869184.0)*nyy*nyy*nzz*nzz - 7296462241136635.0/68719476736.0*nyy*nyy*nzz - 4974860618956799.0/34359738368.0*nyy*nzz*nzz*nzz - 57915*nyy*nzz*nzz + (6582123280465919.0/137438953472.0)*nyy*nzz + (5306517993553919.0/549755813888.0)*nzz*nzz*nzz - 2143016882014758.0/824633720833.0*nzz*nzz - 1620*nzz);  // power=9
    mat[17][23] = FAR_REAL(-7462290928435199.0/8589934592.0*nxx*nxx*nyy*ny*nz + 173745*nxx*nxx*ny*nz + (1243715154739201.0/8589934592.0)*nxx*nyy*nyy*ny*nz - 4974860618956799.0/8589934592.0*nxx*nyy*ny*nzz*nz + (6301490117345279.0/17179869184.0)*nxx*nyy*ny*nz + (7959776990330879.0/68719476736.0)*nxx*ny*nzz*nz - 6582123280465919.0/68719476736.0*nxx*ny*nz - 6633147491942405.0/137438953472.0*nyy*nyy*ny*nzz*nz - 1.0/879609302221.0*nyy*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nyy*ny*nzz*nzz*nz - 8401986823127037.0/68719476736.0*nyy*ny*nzz*nz - 2.0/879609302221.0*nyy*ny*nz - 3979888495165439.0/68719476736.0*ny*nzz*nzz*nz + (8776164373954557.0/274877906944.0)*ny*nzz*nz + (1.0/1000000000000.0)*ny*nz);  // power=9
    mat[17][24] = FAR_REAL(-2025*nxx*nxx*nxx*nx*nz - 11970*nxx*nxx*nx*nyy*nz + 16200*nxx*nxx*nx*nyy*nz + 8775*nxx*nxx*nx*nzz*nz - 920745*nxx*nx*nyy*nyy*nz + (5287963734835199.0/137438953472.0)*nxx*nx*nyy*nyy*nz + (6414361857884159.0/17179869184.0)*nxx*nx*nyy*nzz*nz - 8998815478579199.0/68719476736.0*nxx*nx*nyy*nzz*nz + 235710*nxx*nx*nyy*nz + 4320*nxx*nx*nzz*nzz*nz - 5306517993553919.0/274877906944.0*nxx*nx*nzz*nz - 10125.0/2.0*nxx*nx*nz + (209475.0/2.0)*nx*nyy*nyy*nyy*nz + (5566277615615999.0/274877906944.0)*nx*nyy*nyy*nyy*nz - 1722555.0/2.0*nx*nyy*nyy*nzz*nz - 139725*nx*nyy*nyy*nzz*nz + 186570*nx*nyy*nyy*nz + 95760*nx*nyy*nzz*nzz*nz + (8460741975736319.0/68719476736.0)*nx*nyy*nzz*nzz*nz + 147285*nx*nyy*nzz*nz - 6648609374207999.0/137438953472.0*nx*nyy*nz - 6479999999999999.0/1000000000000.0*nx*nzz*nzz*nzz*nz - 5804999999999999.0/1000000000000.0*nx*nzz*nz + (2835.0/2.0)*nx*nz);  // power=9
    mat[17][25] = FAR_REAL(-385560*nxx*nx*nyy*nyy + 52920*nxx*nx*nyy*nzz + 173880*nxx*nx*nyy - 7560*nxx*nx + (174825.0/2.0)*nx*nyy*nyy*nyy - 332640*nx*nyy*nyy*nzz + (27405.0/2.0)*nx*nyy*nyy + 52920*nx*nyy*nzz*nzz + 120960*nx*nyy*nzz - 65205.0/2.0*nx*nyy - 7560*nx*nzz + (4095.0/2.0)*nx);  // power=8
    mat[17][26] = FAR_REAL(945*nxx*nxx*nxx*nx - 5670*nxx*nxx*nx*nyy - 14175*nxx*nxx*nx*nzz - 6615.0/2.0*nxx*nx*nyy*nyy - 14175*nxx*nx*nyy*nyy - 414225*nxx*nx*nyy*nzz + 141750*nxx*nx*nyy*nzz + 33390*nxx*nx*nyy + 31185*nxx*nx*nzz - 5985.0/2.0*nxx*nx - 6615.0/2.0*nx*nyy*nyy*nyy - 7560*nx*nyy*nyy*nyy + 55440*nx*nyy*nyy*nzz + 155925*nx*nyy*nyy*nzz + 315*nx*nyy*nyy - 8820*nx*nyy*nzz*nzz - 226800*nx*nyy*nzz*nzz + 73395*nx*nyy*nzz - 16065.0/2.0*nx*nyy + 15120*nx*nzz*nzz*nzz - 7245*nx*nzz + 945*nx);  // power=8
    mat[17][27] = FAR_REAL(-405405*nxx*nyy*ny*nzz + 31185*nxx*nyy*ny + 93555*nxx*ny*nzz - 8505*nxx*ny + (135135.0/2.0)*nyy*nyy*ny*nzz - 10395.0/2.0*nyy*nyy*ny - 405405*nyy*ny*nzz*nzz + 166320*nyy*ny*nzz - 6615*nyy*ny + 93555*ny*nzz*nzz - 104895.0/2.0*ny*nzz + (5985.0/2.0)*ny);  // power=8

    // Row 18 (target order 4)
    mat[18][0] = FAR_REAL(-1575*nxx*nx*ny + 2100*nx*nyy*ny - 1575*nx*ny*nzz);  // power=5
    mat[18][1] = FAR_REAL(-14175*nxx*nxx*ny + 18900*nxx*nyy*ny - 14175*nxx*ny*nzz + 4725*nxx*ny - 2100*nyy*ny + 1575*ny*nzz);  // power=6
    mat[18][2] = FAR_REAL(-225*nxx*nxx*nx - 13050*nxx*nx*nyy + 1125*nxx*nx*nzz + 1575*nxx*nx + (5566277615616001.0/274877906944.0)*nx*nyy*nyy - 11475*nx*nyy*nzz - 7875*nx*nyy + 1350*nx*nzz*nzz + 225*nx);  // power=6
    mat[18][3] = FAR_REAL(-11025*nxx*nx*ny*nz + 22050*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[18][4] = FAR_REAL(-34825.0/2.0*nxx*nxx*nx*ny + 31850*nxx*nx*nyy*ny - 8050*nxx*nx*ny*nzz + 7875*nxx*nx*ny - 11375*nx*nyy*nyy*ny - 4025.0/2.0*nx*nyy*ny*nzz + (3675.0/2.0)*nx*nyy*ny + (18725.0/2.0)*nx*ny*nzz*nzz - 4725.0/2.0*nx*ny*nzz - 2975.0/2.0*nx*ny);  // power=7
    mat[18][5] = FAR_REAL(-9625*nxx*nxx*nx*ny + 3850*nxx*nx*nyy*ny + 15400*nxx*nx*ny*nzz + 6300*nxx*nx*ny + 13475*nx*nyy*nyy*ny - 44275.0/2.0*nx*nyy*ny*nzz - 22575.0/2.0*nx*nyy*ny + 25025*nx*ny*nzz*nzz - 23625.0/2.0*nx*ny*nzz + (2975.0/2.0)*nx*ny);  // power=7
    mat[18][6] = FAR_REAL(-51975*nxx*nxx*nyy + 4725*nxx*nxx + 69300*nxx*nyy*nyy - 51975*nxx*nyy*nzz - 4725*nxx*nyy + 4725*nxx*nzz - 1050*nxx - 6300*nyy*nyy + 4725*nyy*nzz + 2625*nyy - 525);  // power=7
    mat[18][7] = FAR_REAL(-51975*nxx*nxx*ny*nz + 69300*nxx*nyy*ny*nz - 51975*nxx*ny*nzz*nz + 23625*nxx*ny*nz - 6300*nyy*ny*nz + 4725*ny*nzz*nz - 1050*ny*nz);  // power=7
    mat[18][8] = FAR_REAL(-1575*nxx*nxx*nx*nz - 49350*nxx*nx*nyy*nz + 1575*nxx*nx*nzz*nz + 4725*nxx*nx*nz + 73500*nx*nyy*nyy*nz - 44625*nx*nyy*nzz*nz - 15225*nx*nyy*nz + 3150*nx*nzz*nzz*nz + 525*nx*nz);  // power=7
    mat[18][9] = FAR_REAL(-135135.0/2.0*nxx*nxx*nxx*ny + 90090*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*ny*nzz + 62370*nxx*nxx*ny - 41580*nxx*nyy*ny + 31185*nxx*ny*nzz - 25515.0/2.0*nxx*ny + 1890*nyy*ny - 2835.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[18][10] = FAR_REAL(-64260*nxx*nx*nyy*nyy + 8820*nxx*nx*nyy*nzz + 28980*nxx*nx*nyy - 1260*nxx*nx + (186795.0/2.0)*nx*nyy*nyy*nyy - 55440*nx*nyy*nyy*nzz - 177345.0/2.0*nx*nyy*nyy + 8820*nx*nyy*nzz*nzz + 20160*nx*nyy*nzz + (38745.0/2.0)*nx*nyy - 1260*nx*nzz - 1155.0/2.0*nx);  // power=8
    mat[18][11] = FAR_REAL(5670*nxx*nxx*nx*ny*nz - 9450*nxx*nx*nyy*ny*nz - 154035.0/2.0*nxx*nx*ny*nzz*nz + (31185.0/2.0)*nxx*nx*ny*nz - 15120*nx*nyy*nyy*ny*nz + 59850*nx*nyy*ny*nzz*nz - 165375.0/2.0*nx*ny*nzz*nzz*nz + (135135.0/2.0)*nx*ny*nzz*nz - 11340*nx*ny*nz);  // power=8
    mat[18][12] = FAR_REAL(-405405.0/2.0*nxx*nxx*nx*nyy + (31185.0/2.0)*nxx*nxx*nx + 270270*nxx*nx*nyy*nyy - 405405.0/2.0*nxx*nx*nyy*nzz + (93555.0/2.0)*nxx*nx*nyy + (31185.0/2.0)*nxx*nx*nzz - 19845.0/2.0*nxx*nx - 62370*nx*nyy*nyy + (93555.0/2.0)*nx*nyy*nzz + 8505*nx*nyy - 8505.0/2.0*nx*nzz + 945*nx);  // power=8
    mat[18][13] = FAR_REAL(-405405.0/2.0*nxx*nxx*nx*ny*nz + 270270*nxx*nx*nyy*ny*nz - 405405.0/2.0*nxx*nx*ny*nzz*nz + (280665.0/2.0)*nxx*nx*ny*nz - 62370*nx*nyy*ny*nz + (93555.0/2.0)*nx*ny*nzz*nz - 17010*nx*ny*nz);  // power=8
    mat[18][14] = FAR_REAL(-25515.0/2.0*nxx*nxx*nx*ny*nz - 219240*nxx*nx*nyy*ny*nz + (42525.0/2.0)*nxx*nx*ny*nzz*nz + (93555.0/2.0)*nxx*nx*ny*nz + 266490*nx*nyy*nyy*ny*nz - 344925.0/2.0*nx*nyy*ny*nzz*nz - 246645.0/2.0*nx*nyy*ny*nz + 34020*nx*ny*nzz*nzz*nz + (31185.0/2.0)*nx*ny*nz);  // power=8
    mat[18][15] = FAR_REAL(-405405*nxx*nxx*nyy*nz + 31185*nxx*nxx*nz + 540540*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz - 14175*nxx*nz - 41580*nyy*nyy*nz + 31185*nyy*nzz*nz + 5670*nyy*nz - 2835*nzz*nz + 630*nz);  // power=8
    mat[18][16] = FAR_REAL(-4974860618956799.0/137438953472.0*nxx*nxx*nxx*nx*ny + (6662009672171521.0/137438953472.0)*nxx*nxx*nx*nyy*ny + (6218575773695999.0/34359738368.0)*nxx*nxx*nx*ny*nzz + (1989944247582719.0/68719476736.0)*nxx*nxx*nx*ny + 420*nxx*nx*nyy*nyy*ny - 297345*nxx*nx*nyy*ny*nzz - 3537678662369279.0/274877906944.0*nxx*nx*nyy*ny + (1370655.0/8.0)*nxx*nx*ny*nzz*nzz - 8980261219860479.0/68719476736.0*nxx*nx*ny*nzz - 4267479505304045.0/916259689813.0*nxx*nx*ny + 210*nx*nyy*nyy*nyy*ny - 7769999999999999.0/1000000000000.0*nx*nyy*nyy*ny*nzz + 30630*nx*nyy*ny*nzz*nzz + (7044433560207361.0/137438953472.0)*nx*nyy*ny*nzz - 362838837166163.0/879609302221.0*nx*nyy*ny - 6301490117345279.0/137438953472.0*nx*ny*nzz*nzz*nzz + (4454999999999999.0/1000000000000.0)*nx*ny*nzz*nzz + (4786998749429759.0/549755813888.0)*nx*ny*nzz + (405.0/2.0)*nx*ny);  // power=9
    mat[18][17] = FAR_REAL(1155*nxx*nxx*nx*nyy*ny - 75405.0/4.0*nxx*nx*nyy*nyy*ny + (916485.0/4.0)*nxx*nx*nyy*ny*nzz - 849214149048800.0/211444543803.0*nxx*nx*nyy*ny - 6135661430046719.0/68719476736.0*nxx*nx*ny*nzz*nzz - 4455000000000001.0/1000000000000.0*nxx*nx*ny*nzz + (853495901062673.0/733007751852.0)*nxx*nx*ny + (515625.0/8.0)*nx*nyy*nyy*nyy*ny - 1195185.0/4.0*nx*nyy*nyy*ny*nzz - 538515.0/8.0*nx*nyy*nyy*ny + (7667032019435519.0/34359738368.0)*nx*nyy*ny*nzz*nzz + (2362962157240321.0/34359738368.0)*nx*nyy*ny*nzz + (198525.0/8.0)*nx*nyy*ny - 6135661430046719.0/68719476736.0*nx*ny*nzz*nzz*nzz + (271755.0/4.0)*nx*ny*nzz*nzz - 7027425489715201.0/274877906944.0*nx*ny*nzz - 16875.0/8.0*nx*ny);  // power=9
    mat[18][18] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (7959776990330877.0/274877906944.0)*nxx*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nxx*nzz + (6633147491942401.0/34359738368.0)*nxx*nyy*nyy*nyy - 1447875.0/2.0*nxx*nyy*nyy*nzz - 173745.0/2.0*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (31185.0/2.0)*nxx*nzz - 945*nxx - 7075357324738559.0/549755813888.0*nyy*nyy*nyy + (6633147491942401.0/137438953472.0)*nyy*nyy*nzz + 8910*nyy*nyy - 7959776990330879.0/274877906944.0*nyy*nzz*nzz - 2040968459058460.0/916259689813.0*nyy*nzz - 2025*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz + 105);  // power=9
    mat[18][19] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*nyy + (5306517993553919.0/549755813888.0)*nxx*nxx*nxx + (6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nxx*nyy*nzz + (6633147491942403.0/137438953472.0)*nxx*nxx*nyy - 19305*nxx*nxx*nzz - 1836871613153279.0/274877906944.0*nxx*nxx - 579150*nxx*nyy*nyy*nzz - 5306517993553919.0/137438953472.0*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz - 6.0/753950830475.0*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (3673743226306559.0/274877906944.0)*nxx*nzz + 810*nxx + (5306517993553921.0/137438953472.0)*nyy*nyy*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (4455.0/2.0)*nzz*nzz - 810*nzz);  // power=9
    mat[18][20] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*ny*nz + (6633147491942401.0/34359738368.0)*nxx*nxx*nyy*ny*nz + (7296462241136639.0/68719476736.0)*nxx*nxx*ny*nz - 6633147491942401.0/34359738368.0*nxx*nyy*ny*nzz*nz - 38610*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz - 7959776990330875.0/137438953472.0*nxx*ny*nzz*nz - 13365*nxx*ny*nz + (7075357324738559.0/549755813888.0)*nyy*ny*nzz*nz - 5306517993553919.0/549755813888.0*ny*nzz*nzz*nz + 4455*ny*nzz*nz);  // power=9
    mat[18][21] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz - 4409999999999999.0/1000000000000.0*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz - 303660*nxx*nx*nyy*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nx*nyy*nyy*nz + (6035545742376959.0/34359738368.0)*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz + (2504824927027199.0/274877906944.0)*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz - 5306517993553919.0/549755813888.0*nxx*nx*nzz*nz + (2835.0/2.0)*nxx*nx*nz + (77175.0/2.0)*nx*nyy*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nx*nyy*nyy*nyy*nz - 457020*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz + (4629287550320639.0/137438953472.0)*nx*nyy*nyy*nz + (4848846278492159.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (6410496387317761.0/137438953472.0)*nx*nyy*nzz*nz + (8893674679173121.0/274877906944.0)*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 1552499999998447.0/999999999999.0*nx*nzz*nz - 1350*nx*nz);  // power=9
    mat[18][22] = FAR_REAL(-7415389885562879.0/34359738368.0*nxx*nxx*nx*nyy*ny + (7462290928435199.0/34359738368.0)*nxx*nxx*nx*ny*nzz + (7959776990330879.0/274877906944.0)*nxx*nxx*nx*ny + (5251370613473281.0/17179869184.0)*nxx*nx*nyy*nyy*ny - 291480*nxx*nx*nyy*ny*nzz - 82215.0/2.0*nxx*nx*nyy*ny + (5638175368151039.0/68719476736.0)*nxx*nx*ny*nzz*nzz - 3979888495165439.0/34359738368.0*nxx*nx*ny*nzz - 333976656936656.0/219902325555.0*nxx*nx*ny + (8097903138570241.0/549755813888.0)*nx*nyy*nyy*nyy*ny - 278115*nx*nyy*nyy*ny*nzz - 64125*nx*nyy*nyy*ny + 247695*nx*nyy*ny*nzz*nzz + 146925*nx*nyy*ny*nzz + (99435.0/4.0)*nx*nyy*ny - 135135*nx*ny*nzz*nzz*nzz + (4974860618956799.0/68719476736.0)*nx*ny*nzz*nzz - 5928085660631041.0/274877906944.0*nx*ny*nzz - 1923749999994229.0/999999999997.0*nx*ny);  // power=9
    mat[18][23] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz - 7505584198778879.0/17179869184.0*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz + (7959776990330879.0/274877906944.0)*nxx*nxx*nx*nz + 598680*nxx*nx*nyy*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nx*nyy*nyy*nz - 4671807726551039.0/17179869184.0*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz + (3590249062072319.0/34359738368.0)*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz + 19305*nxx*nx*nzz*nz - 17010*nxx*nx*nz + 22050*nx*nyy*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nx*nyy*nyy*nyy*nz - 150840*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz - 113130*nx*nyy*nyy*nz + 20160*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + 101115*nx*nyy*nzz*nz + (2718817377582401.0/769658139443.0)*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 6615*nx*nzz*nz + (1822499999998177.0/999999999999.0)*nx*nz);  // power=9
    mat[18][24] = FAR_REAL(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*nxx*ny*nzz*nz + (7959776990330879.0/137438953472.0)*nxx*nxx*ny*nz + 579150*nxx*nyy*nyy*ny*nz - 1254825.0/2.0*nxx*nyy*ny*nzz*nz - 2984916371374081.0/34359738368.0*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz - 994972123791359.0/34359738368.0*nxx*ny*nzz*nz - 5306517993553921.0/137438953472.0*nyy*nyy*ny*nz + (5748727826350079.0/137438953472.0)*nyy*ny*nzz*nz + (7347486452613121.0/549755813888.0)*nyy*ny*nz - 5306517993553919.0/549755813888.0*ny*nzz*nzz*nz - 1.0/1000000000000.0*ny*nzz*nz - 1215*ny*nz);  // power=9
    mat[18][25] = FAR_REAL(-405405.0/2.0*nxx*nxx*nyy*ny + (93555.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*nyy*ny - 405405.0/2.0*nxx*nyy*ny*nzz - 197505.0/2.0*nxx*nyy*ny + (93555.0/2.0)*nxx*ny*nzz - 2835.0/2.0*nxx*ny - 20790*nyy*nyy*ny + (31185.0/2.0)*nyy*ny*nzz + 13230*nyy*ny - 8505.0/2.0*ny*nzz - 1260*ny);  // power=8
    mat[18][26] = FAR_REAL(-405405.0/2.0*nxx*nxx*ny*nzz + (31185.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*ny*nzz - 20790*nxx*nyy*ny - 405405.0/2.0*nxx*ny*nzz*nzz + 124740*nxx*ny*nzz - 14175.0/2.0*nxx*ny - 20790*nyy*ny*nzz + 1890*nyy*ny + (31185.0/2.0)*ny*nzz*nzz - 14175.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[18][27] = FAR_REAL((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nx*nyy - 14175.0/2.0*nxx*nxx*nx*nzz - 13230*nxx*nx*nyy*nyy - 14175.0/2.0*nxx*nx*nyy*nyy - 475965.0/2.0*nxx*nx*nyy*nzz + 70875*nxx*nx*nyy*nzz + (48825.0/2.0)*nxx*nx*nyy + (31185.0/2.0)*nxx*nx*nzz - 4095.0/2.0*nxx*nx - 13230*nx*nyy*nyy*nyy - 3780*nx*nyy*nyy*nyy + 221760*nx*nyy*nyy*nzz + (155925.0/2.0)*nx*nyy*nyy*nzz + 1260*nx*nyy*nyy - 35280*nx*nyy*nzz*nzz - 113400*nx*nyy*nzz*nzz - 67725.0/2.0*nx*nyy*nzz - 4725.0/2.0*nx*nyy + 7560*nx*nzz*nzz*nzz + (1575.0/2.0)*nx*nzz + (945.0/2.0)*nx);  // power=8

    // Row 19 (target order 4)
    mat[19][0] = FAR_REAL(2100*nxx*nx*ny - 1575*nx*nyy*ny - 1575*nx*ny*nzz);  // power=5
    mat[19][1] = FAR_REAL(18900*nxx*nxx*ny - 14175*nxx*nyy*ny - 14175*nxx*ny*nzz - 6300*nxx*ny + 1575*nyy*ny + 1575*ny*nzz);  // power=6
    mat[19][2] = FAR_REAL(-225*nxx*nxx*nx + 20025*nxx*nx*nyy + 1125*nxx*nx*nzz - 2100*nxx*nx - 7050618313113599.0/549755813888.0*nx*nyy*nyy - 11475*nx*nyy*nzz + 3150*nx*nyy + 1350*nx*nzz*nzz + 225*nx);  // power=6
    mat[19][3] = FAR_REAL(22050*nxx*nx*ny*nz - 11025*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[19][4] = FAR_REAL(23625*nxx*nxx*nx*ny - 29400*nxx*nx*nyy*ny - 32550*nxx*nx*ny*nzz - 10500*nxx*nx*ny + (15225.0/2.0)*nx*nyy*nyy*ny + 12075*nx*nyy*ny*nzz + (8925.0/2.0)*nx*ny*nzz*nzz + 3150*nx*ny*nzz + 1575*nx*ny);  // power=7
    mat[19][5] = FAR_REAL((23625.0/2.0)*nxx*nxx*nx*ny + 2625*nxx*nx*nyy*ny - 33600*nxx*nx*ny*nzz - 8400*nxx*nx*ny - 18375.0/2.0*nx*nyy*nyy*ny + (12075.0/2.0)*nx*nyy*ny*nzz + (14175.0/2.0)*nx*nyy*ny + 15225*nx*ny*nzz*nzz - 1575.0/2.0*nx*ny*nzz + (525.0/2.0)*nx*ny);  // power=7
    mat[19][6] = FAR_REAL(69300*nxx*nxx*nyy - 6300*nxx*nxx - 51975*nxx*nyy*nyy - 51975*nxx*nyy*nzz - 4725*nxx*nyy + 4725*nxx*nzz + 2625*nxx + 4725*nyy*nyy + 4725*nyy*nzz - 1050*nyy - 525);  // power=7
    mat[19][7] = FAR_REAL(69300*nxx*nxx*ny*nz - 51975*nxx*nyy*ny*nz - 51975*nxx*ny*nzz*nz - 9450*nxx*ny*nz + 4725*nyy*ny*nz + 4725*ny*nzz*nz - 1050*ny*nz);  // power=7
    mat[19][8] = FAR_REAL(-1575*nxx*nxx*nx*nz + 86625*nxx*nx*nyy*nz + 1575*nxx*nx*nzz*nz - 6300*nxx*nx*nz - 33075*nx*nyy*nyy*nz - 29925*nx*nyy*nzz*nz + 3150*nx*nyy*nz + 3150*nx*nzz*nzz*nz + 525*nx*nz);  // power=7
    mat[19][9] = FAR_REAL(90090*nxx*nxx*nxx*ny - 135135.0/2.0*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*ny*nzz - 83160*nxx*nxx*ny + 31185*nxx*nyy*ny + 31185*nxx*ny*nzz + 17010*nxx*ny - 2835.0/2.0*nyy*ny - 2835.0/2.0*ny*nzz - 420*ny);  // power=8
    mat[19][10] = FAR_REAL((186795.0/2.0)*nxx*nx*nyy*nyy + 8820*nxx*nx*nyy*nzz - 43785*nxx*nx*nyy + (4095.0/2.0)*nxx*nx - 64260*nx*nyy*nyy*nyy - 55440*nx*nyy*nyy*nzz + (113715.0/2.0)*nx*nyy*nyy + 8820*nx*nyy*nzz*nzz + 20160*nx*nyy*nzz - 10395*nx*nyy - 1260*nx*nzz + (315.0/2.0)*nx);  // power=8
    mat[19][11] = FAR_REAL(-8505.0/2.0*nxx*nxx*nx*ny*nz + (14175.0/2.0)*nxx*nx*nyy*ny*nz + (194355.0/2.0)*nxx*nx*ny*nzz*nz - 20790*nxx*nx*ny*nz + 11340*nx*nyy*nyy*ny*nz - 89775.0/2.0*nx*nyy*ny*nzz*nz - 112455.0/2.0*nx*ny*nzz*nzz*nz + 31185*nx*ny*nzz*nz - 2835.0/2.0*nx*ny*nz);  // power=8
    mat[19][12] = FAR_REAL(270270*nxx*nxx*nx*nyy - 20790*nxx*nxx*nx - 405405.0/2.0*nxx*nx*nyy*nyy - 405405.0/2.0*nxx*nx*nyy*nzz - 197505.0/2.0*nxx*nx*nyy + (31185.0/2.0)*nxx*nx*nzz + 13230*nxx*nx + (93555.0/2.0)*nx*nyy*nyy + (93555.0/2.0)*nx*nyy*nzz - 2835.0/2.0*nx*nyy - 8505.0/2.0*nx*nzz - 1260*nx);  // power=8
    mat[19][13] = FAR_REAL(270270*nxx*nxx*nx*ny*nz - 405405.0/2.0*nxx*nx*nyy*ny*nz - 405405.0/2.0*nxx*nx*ny*nzz*nz - 114345*nxx*nx*ny*nz + (93555.0/2.0)*nx*nyy*ny*nz + (93555.0/2.0)*nx*ny*nzz*nz + 2835*nx*ny*nz);  // power=8
    mat[19][14] = FAR_REAL(-25515.0/2.0*nxx*nxx*nx*ny*nz + (639765.0/2.0)*nxx*nx*nyy*ny*nz + (42525.0/2.0)*nxx*nx*ny*nzz*nz - 62370*nxx*nx*ny*nz - 280665.0/2.0*nx*nyy*nyy*ny*nz - 212625.0/2.0*nx*nyy*ny*nzz*nz + 65205*nx*nyy*ny*nz + 34020*nx*ny*nzz*nzz*nz - 8505.0/2.0*nx*ny*nz);  // power=8
    mat[19][15] = FAR_REAL(540540*nxx*nxx*nyy*nz - 41580*nxx*nxx*nz - 405405*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz + 5670*nxx*nz + 31185*nyy*nyy*nz + 31185*nyy*nzz*nz - 14175*nyy*nz - 2835*nzz*nz + 630*nz);  // power=8
    mat[19][16] = FAR_REAL((6633147491942401.0/137438953472.0)*nxx*nxx*nxx*nx*ny - 4996507254128639.0/137438953472.0*nxx*nxx*nx*nyy*ny - 5596718196326401.0/17179869184.0*nxx*nxx*nx*ny*nzz - 38610*nxx*nxx*nx*ny - 315*nxx*nx*nyy*nyy*ny + (7662522303774719.0/34359738368.0)*nxx*nx*nyy*ny*nzz + (2653258996776959.0/274877906944.0)*nxx*nx*nyy*ny + (2181465.0/8.0)*nxx*nx*ny*nzz*nzz + (4413594292715521.0/68719476736.0)*nxx*nx*ny*nzz + (3103199770705919.0/274877906944.0)*nxx*nx*ny - 315.0/2.0*nx*nyy*nyy*nyy*ny + (3203702005438147.0/549755813889.0)*nx*nyy*nyy*ny*nzz - 3157316358635519.0/137438953472.0*nx*nyy*ny*nzz*nzz - 5283325170155519.0/137438953472.0*nx*nyy*ny*nzz + (97188974241047.0/314146179365.0)*nx*nyy*ny - 7959776990330879.0/274877906944.0*nx*ny*nzz*nzz*nzz - 161865.0/8.0*nx*ny*nzz*nzz + (69578470195453.0/54975581389.0)*nx*ny*nzz - 603013408357303.0/549755813887.0*nx*ny);  // power=9
    mat[19][17] = FAR_REAL((1575.0/2.0)*nxx*nxx*nx*nyy*ny + (331875.0/8.0)*nxx*nx*nyy*nyy*ny - 5997277583769601.0/17179869184.0*nxx*nx*nyy*ny*nzz - 2900649113026559.0/549755813888.0*nxx*nx*nyy*ny + (6633147491942405.0/549755813888.0)*nxx*nx*ny*nzz*nzz + (221265.0/4.0)*nxx*nx*ny*nzz - 2025*nxx*nx*ny - 87525.0/2.0*nx*nyy*nyy*nyy*ny + (7773461309030399.0/68719476736.0)*nx*nyy*nyy*ny*nzz + (6171610306314237.0/137438953472.0)*nx*nyy*nyy*ny + (5805936790732799.0/68719476736.0)*nx*nyy*ny*nzz*nzz - 6837244337848317.0/137438953472.0*nx*nyy*ny*nzz - 8170058589143037.0/549755813888.0*nx*nyy*ny - 4974860618956799.0/68719476736.0*nx*ny*nzz*nzz*nzz + (5944320637009919.0/137438953472.0)*nx*ny*nzz*nzz - 2950127136276481.0/549755813888.0*nx*ny*nzz + 2025*nx*ny);  // power=9
    mat[19][18] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyy - 579150*nxx*nxx*nyy*nzz - 5306517993553919.0/137438953472.0*nxx*nxx*nyy + (5306517993553921.0/137438953472.0)*nxx*nxx*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nyy*nzz + (6633147491942403.0/137438953472.0)*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz - 6.0/753950830475.0*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (5306517993553919.0/549755813888.0)*nyy*nyy*nyy - 19305*nyy*nyy*nzz - 1836871613153279.0/274877906944.0*nyy*nyy - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (3673743226306559.0/274877906944.0)*nyy*nzz + 810*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz);  // power=9
    mat[19][19] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nyy - 7075357324738559.0/549755813888.0*nxx*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyy - 1447875.0/2.0*nxx*nxx*nyy*nzz - 173745.0/2.0*nxx*nxx*nyy + (6633147491942401.0/137438953472.0)*nxx*nxx*nzz + 8910*nxx*nxx + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (7959776990330877.0/274877906944.0)*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz - 2040968459058460.0/916259689813.0*nxx*nzz - 2025*nxx - 7959776990330879.0/274877906944.0*nyy*nyy*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (31185.0/2.0)*nyy*nzz - 945*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz + 105);  // power=9
    mat[19][20] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*ny*nz - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*ny*nz - 675675.0/2.0*nxx*nxx*ny*nzz*nz - 6633147491942401.0/68719476736.0*nxx*nxx*ny*nz + (4974860618956799.0/34359738368.0)*nxx*nyy*ny*nzz*nz + (3979888495165439.0/137438953472.0)*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz + (331657374597121.0/34359738368.0)*nxx*ny*nzz*nz + (4898324301742081.0/274877906944.0)*nxx*ny*nz - 5306517993553919.0/549755813888.0*nyy*ny*nzz*nz - 5306517993553919.0/549755813888.0*ny*nzz*nzz*nz + 4455*ny*nzz*nz - 945*ny*nz);  // power=9
    mat[19][21] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz + (675675.0/2.0)*nxx*nx*nyy*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nx*nyy*nyy*nz - 6633147491942401.0/34359738368.0*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz - 5306517993553919.0/137438953472.0*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz + (7075357324738559.0/549755813888.0)*nxx*nx*nzz*nz + (5566277615615999.0/549755813888.0)*nx*nyy*nyy*nyy*nz + (4974860618956799.0/34359738368.0)*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz - 38610*nx*nyy*nyy*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (5306517993553919.0/137438953472.0)*nx*nyy*nzz*nz - 6735195914895359.0/274877906944.0*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 2970*nx*nzz*nz + (2025.0/2.0)*nx*nz);  // power=9
    mat[19][22] = FAR_REAL(290205*nxx*nxx*nx*nyy*ny - 289575*nxx*nxx*nx*ny*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nx*ny - 225945*nxx*nx*nyy*nyy*ny - 349335*nxx*nx*nyy*ny*nzz - 19845*nxx*nx*nyy*ny + (1138995.0/4.0)*nxx*nx*ny*nzz*nzz + (4770763773050881.0/34359738368.0)*nxx*nx*ny*nzz + (6178568153333761.0/549755813888.0)*nxx*nx*ny - 5164268676710399.0/549755813888.0*nx*nyy*nyy*nyy*ny + (5064539536097279.0/34359738368.0)*nx*nyy*nyy*ny*nzz + (192375.0/4.0)*nx*nyy*nyy*ny + 55440*nx*nyy*ny*nzz*nzz - 8423633458298879.0/137438953472.0*nx*nyy*ny*nzz - 56385.0/4.0*nx*nyy*ny - 405405.0/4.0*nx*ny*nzz*nzz*nzz + (6327002223083521.0/274877906944.0)*nx*ny*nzz*nzz - 2421330762792961.0/274877906944.0*nx*ny*nzz + (911249999997266.0/999999999997.0)*nx*ny);  // power=9
    mat[19][23] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz + 581040*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz - 5306517993553921.0/137438953472.0*nxx*nxx*nx*nz - 7713933062307839.0/17179869184.0*nxx*nx*nyy*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nx*nyy*nyy*nz - 1281285.0/2.0*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz - 2692686796554241.0/34359738368.0*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz + (5748727826350079.0/137438953472.0)*nxx*nx*nzz*nz + (25515.0/2.0)*nxx*nx*nz - 33075.0/2.0*nx*nyy*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nx*nyy*nyy*nyy*nz + (7774234403143679.0/68719476736.0)*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz + (169695.0/2.0)*nx*nyy*nyy*nz - 4156153952993279.0/274877906944.0*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (186165.0/2.0)*nx*nyy*nzz*nz - 6085796859740159.0/274877906944.0*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 16065.0/2.0*nx*nzz*nz + (404999999999999.0/1000000000000.0)*nx*nz);  // power=9
    mat[19][24] = FAR_REAL(579150*nxx*nxx*nyy*ny*nz - 6633147491942401.0/34359738368.0*nxx*nxx*ny*nzz*nz - 5306517993553921.0/68719476736.0*nxx*nxx*ny*nz - 7462290928435199.0/17179869184.0*nxx*nyy*nyy*ny*nz - 4974860618956799.0/17179869184.0*nxx*nyy*ny*nzz*nz + (3979888495165439.0/34359738368.0)*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz + (5306517993553923.0/137438953472.0)*nxx*ny*nzz*nz + (7959776990330879.0/274877906944.0)*nyy*nyy*ny*nz + 19305*nyy*ny*nzz*nz - 17820*nyy*ny*nz - 5306517993553919.0/549755813888.0*ny*nzz*nzz*nz - 1.0/1000000000000.0*ny*nzz*nz + 1620*ny*nz);  // power=9
    mat[19][25] = FAR_REAL(270270*nxx*nxx*nyy*ny - 62370*nxx*nxx*ny - 405405.0/2.0*nxx*nyy*nyy*ny - 405405.0/2.0*nxx*nyy*ny*nzz + (93555.0/2.0)*nxx*nyy*ny + (93555.0/2.0)*nxx*ny*nzz + 8505*nxx*ny + (31185.0/2.0)*nyy*nyy*ny + (31185.0/2.0)*nyy*ny*nzz - 19845.0/2.0*nyy*ny - 8505.0/2.0*ny*nzz + 945*ny);  // power=8
    mat[19][26] = FAR_REAL(270270*nxx*nxx*ny*nzz - 20790*nxx*nxx*ny - 405405.0/2.0*nxx*nyy*ny*nzz + (31185.0/2.0)*nxx*nyy*ny - 405405.0/2.0*nxx*ny*nzz*nzz + (31185.0/2.0)*nxx*ny*nzz + 2835*nxx*ny + (31185.0/2.0)*nyy*ny*nzz - 2835.0/2.0*nyy*ny + (31185.0/2.0)*ny*nzz*nzz - 14175.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[19][27] = FAR_REAL((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nx*nyy - 14175.0/2.0*nxx*nxx*nx*nzz + (19845.0/2.0)*nxx*nx*nyy*nyy - 14175.0/2.0*nxx*nx*nyy*nyy + 296730*nxx*nx*nyy*nzz + 70875*nxx*nx*nyy*nzz - 27405*nxx*nx*nyy - 20790*nxx*nx*nzz + (4725.0/2.0)*nxx*nx + (19845.0/2.0)*nx*nyy*nyy*nyy - 3780*nx*nyy*nyy*nyy - 166320*nx*nyy*nyy*nzz + (155925.0/2.0)*nx*nyy*nyy*nzz - 945*nx*nyy*nyy + 26460*nx*nyy*nzz*nzz - 113400*nx*nyy*nzz*nzz - 1890*nx*nyy*nzz + (8505.0/2.0)*nx*nyy + 7560*nx*nzz*nzz*nzz + 1890*nx*nzz - 630*nx);  // power=8

    // Row 20 (target order 4)
    mat[20][0] = FAR_REAL(2100*nxx*nx*nz - 1575*nx*nyy*nz - 1575*nx*nzz*nz);  // power=5
    mat[20][1] = FAR_REAL(18900*nxx*nxx*nz - 14175*nxx*nyy*nz - 14175*nxx*nzz*nz - 6300*nxx*nz + 1575*nyy*nz + 1575*nzz*nz);  // power=6
    mat[20][2] = FAR_REAL(22050*nxx*nx*ny*nz - 11025*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[20][3] = FAR_REAL(-225*nxx*nxx*nx + 1125*nxx*nx*nyy + 20025*nxx*nx*nzz - 2100*nxx*nx + 1350*nx*nyy*nyy - 11475*nx*nyy*nzz - 7050618313113599.0/549755813888.0*nx*nzz*nzz + 3150*nx*nzz + 225*nx);  // power=6
    mat[20][4] = FAR_REAL(23625*nxx*nxx*nx*nz - 33600*nxx*nx*nyy*nz - 29400*nxx*nx*nzz*nz - 10500*nxx*nx*nz + (6825.0/2.0)*nx*nyy*nyy*nz + 11025*nx*nyy*nzz*nz + 4200*nx*nyy*nz + (15225.0/2.0)*nx*nzz*nzz*nz + 1575*nx*nz);  // power=7
    mat[20][5] = FAR_REAL((23625.0/2.0)*nxx*nxx*nx*nz + 2100*nxx*nx*nyy*nz - 32025*nxx*nx*nzz*nz - 2100*nxx*nx*nz - 19425.0/2.0*nx*nyy*nyy*nz + (14175.0/2.0)*nx*nyy*nzz*nz + (5775.0/2.0)*nx*nyy*nz + 16800*nx*nzz*nzz*nz - 14175.0/2.0*nx*nzz*nz + (2625.0/2.0)*nx*nz);  // power=7
    mat[20][6] = FAR_REAL(69300*nxx*nxx*ny*nz - 51975*nxx*nyy*ny*nz - 51975*nxx*ny*nzz*nz - 9450*nxx*ny*nz + 4725*nyy*ny*nz + 4725*ny*nzz*nz - 1050*ny*nz);  // power=7
    mat[20][7] = FAR_REAL(69300*nxx*nxx*nzz - 6300*nxx*nxx - 51975*nxx*nyy*nzz + 4725*nxx*nyy - 51975*nxx*nzz*nzz - 4725*nxx*nzz + 2625*nxx + 4725*nyy*nzz + 4725*nzz*nzz - 1050*nzz - 525);  // power=7
    mat[20][8] = FAR_REAL(-1575*nxx*nxx*nx*ny + 1575*nxx*nx*nyy*ny + 80325*nxx*nx*ny*nzz - 6300*nxx*nx*ny + 3150*nx*nyy*nyy*ny - 36225*nx*nyy*ny*nzz - 39375*nx*ny*nzz*nzz + 9450*nx*ny*nzz + 525*nx*ny);  // power=7
    mat[20][9] = FAR_REAL(90090*nxx*nxx*nxx*nz - 135135.0/2.0*nxx*nxx*nyy*nz - 135135.0/2.0*nxx*nxx*nzz*nz - 83160*nxx*nxx*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz + 17010*nxx*nz - 2835.0/2.0*nyy*nz - 2835.0/2.0*nzz*nz - 420*nz);  // power=8
    mat[20][10] = FAR_REAL(-8505.0/2.0*nxx*nxx*nx*ny*nz + (213255.0/2.0)*nxx*nx*nyy*ny*nz + (14175.0/2.0)*nxx*nx*ny*nzz*nz - 20790*nxx*nx*ny*nz - 93555.0/2.0*nx*nyy*nyy*ny*nz - 70875.0/2.0*nx*nyy*ny*nzz*nz + 21735*nx*nyy*ny*nz + 11340*nx*ny*nzz*nzz*nz - 2835.0/2.0*nx*ny*nz);  // power=8
    mat[20][11] = FAR_REAL((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nx*nyy - 4725.0/2.0*nxx*nxx*nx*nzz - 4725.0/2.0*nxx*nx*nyy*nyy + 23625*nxx*nx*nyy*nzz + 90090*nxx*nx*nzz*nzz - 41580*nxx*nx*nzz + 1890*nxx*nx - 1260*nx*nyy*nyy*nyy + (51975.0/2.0)*nx*nyy*nyy*nzz - 37800*nx*nyy*nzz*nzz - 135135.0/2.0*nx*nzz*nzz*nzz + 2520*nx*nzz*nzz*nzz + (114345.0/2.0)*nx*nzz*nzz - 19845.0/2.0*nx*nzz + (315.0/2.0)*nx);  // power=8
    mat[20][12] = FAR_REAL(270270*nxx*nxx*nx*ny*nz - 405405.0/2.0*nxx*nx*nyy*ny*nz - 405405.0/2.0*nxx*nx*ny*nzz*nz - 114345*nxx*nx*ny*nz + (93555.0/2.0)*nx*nyy*ny*nz + (93555.0/2.0)*nx*ny*nzz*nz + 2835*nx*ny*nz);  // power=8
    mat[20][13] = FAR_REAL(270270*nxx*nxx*nx*nzz - 20790*nxx*nxx*nx - 405405.0/2.0*nxx*nx*nyy*nzz + (31185.0/2.0)*nxx*nx*nyy - 405405.0/2.0*nxx*nx*nzz*nzz - 197505.0/2.0*nxx*nx*nzz + 13230*nxx*nx + (93555.0/2.0)*nx*nyy*nzz - 8505.0/2.0*nx*nyy + (93555.0/2.0)*nx*nzz*nzz - 2835.0/2.0*nx*nzz - 1260*nx);  // power=8
    mat[20][14] = FAR_REAL((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nx*nyy - 14175.0/2.0*nxx*nxx*nx*nzz + (19845.0/2.0)*nxx*nx*nyy*nyy - 14175.0/2.0*nxx*nx*nyy*nyy + 296730*nxx*nx*nyy*nzz + 70875*nxx*nx*nyy*nzz - 27405*nxx*nx*nyy - 20790*nxx*nx*nzz + (4725.0/2.0)*nxx*nx + (19845.0/2.0)*nx*nyy*nyy*nyy - 3780*nx*nyy*nyy*nyy - 166320*nx*nyy*nyy*nzz + (155925.0/2.0)*nx*nyy*nyy*nzz - 945*nx*nyy*nyy + 26460*nx*nyy*nzz*nzz - 113400*nx*nyy*nzz*nzz - 1890*nx*nyy*nzz + (8505.0/2.0)*nx*nyy + 7560*nx*nzz*nzz*nzz + 1890*nx*nzz - 630*nx);  // power=8
    mat[20][15] = FAR_REAL(540540*nxx*nxx*ny*nzz - 41580*nxx*nxx*ny - 405405*nxx*nyy*ny*nzz + 31185*nxx*nyy*ny - 405405*nxx*ny*nzz*nzz + 31185*nxx*ny*nzz + 5670*nxx*ny + 31185*nyy*ny*nzz - 2835*nyy*ny + 31185*ny*nzz*nzz - 14175*ny*nzz + 630*ny);  // power=8
    mat[20][16] = FAR_REAL((6633147491942401.0/137438953472.0)*nxx*nxx*nxx*nx*nz + (2025.0/8.0)*nxx*nxx*nxx*nx*nz - 4974860618956799.0/137438953472.0*nxx*nxx*nx*nyy*nz - 2025*nxx*nxx*nx*nyy*nz - 5596718196326401.0/17179869184.0*nxx*nxx*nx*nzz*nz - 8775.0/8.0*nxx*nxx*nx*nzz*nz + (6.0/753950830475.0)*nxx*nxx*nx*nz - 3304977334267190.0/687194767359.0*nxx*nx*nyy*nyy*nz + (7462290928435199.0/34359738368.0)*nxx*nx*nyy*nzz*nz + (8998815478579199.0/549755813888.0)*nxx*nx*nyy*nzz*nz - 5306517993553921.0/274877906944.0*nxx*nx*nyy*nz + (1061775.0/4.0)*nxx*nx*nzz*nzz*nz - 540*nxx*nx*nzz*nzz*nz + (7075357324738561.0/274877906944.0)*nxx*nx*nzz*nz - 2040968459058460.0/916259689813.0*nxx*nx*nz - 1391569403906531.0/549755813889.0*nx*nyy*nyy*nyy*nz + (139725.0/8.0)*nx*nyy*nyy*nzz*nz - 8460741975736319.0/549755813888.0*nx*nyy*nzz*nzz*nz - 173745.0/4.0*nx*nyy*nzz*nz + (4974860618949560.0/687194767359.0)*nx*nyy*nz - 4974860618956799.0/137438953472.0*nx*nzz*nzz*nzz*nz + 810*nx*nzz*nzz*nzz*nz - 1326629498388483.0/549755813888.0*nx*nzz*nzz*nz - 359210448794461.0/967570232443.0*nx*nzz*nz + (14843406974912.0/879609302217.0)*nx*nz);  // power=9
    mat[20][17] = FAR_REAL(-10125.0/8.0*nxx*nxx*nxx*nx*nz + (4725.0/2.0)*nxx*nxx*nx*nyy*nz + 10125*nxx*nxx*nx*nyy*nz + (3768833802234515.0/687194767359.0)*nxx*nxx*nx*nzz*nz + 66150*nxx*nx*nyy*nyy*nz + (192375.0/8.0)*nxx*nx*nyy*nyy*nz - 5258972705587199.0/17179869184.0*nxx*nx*nyy*nzz*nz - 5624259674111999.0/68719476736.0*nxx*nx*nyy*nzz*nz + (6767665867653121.0/137438953472.0)*nxx*nx*nyy*nz + (6633147491942401.0/137438953472.0)*nxx*nx*nzz*nzz*nz + 2700*nxx*nx*nzz*nzz*nz - 3537678662369279.0/274877906944.0*nxx*nx*nzz*nz - 667953313873313.0/879609302220.0*nxx*nx*nz - 165375.0/8.0*nx*nyy*nyy*nyy*nz + (50625.0/4.0)*nx*nyy*nyy*nyy*nz + (1420875.0/8.0)*nx*nyy*nyy*nzz*nz - 698625.0/8.0*nx*nyy*nyy*nzz*nz - 48735.0/4.0*nx*nyy*nyy*nz - 18900*nx*nyy*nzz*nzz*nz + 76950*nx*nyy*nzz*nzz*nz + (4786998749429763.0/274877906944.0)*nx*nyy*nzz*nz - 7867005696737281.0/549755813888.0*nx*nyy*nz - 4974860618956799.0/137438953472.0*nx*nzz*nzz*nzz*nz - 4050*nx*nzz*nzz*nzz*nz + (2819087684075519.0/68719476736.0)*nx*nzz*nzz*nz - 3984527059845119.0/274877906944.0*nx*nzz*nz + 2025*nx*nz);  // power=9
    mat[20][18] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nyy*ny*nz - 579150*nxx*nxx*ny*nzz*nz + (5306517993553921.0/68719476736.0)*nxx*nxx*ny*nz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*ny*nz + (4974860618956799.0/17179869184.0)*nxx*nyy*ny*nzz*nz - 5306517993553923.0/137438953472.0*nxx*nyy*ny*nz + (7462290928435199.0/17179869184.0)*nxx*ny*nzz*nzz*nz - 3979888495165439.0/34359738368.0*nxx*ny*nzz*nz + (5306517993553919.0/549755813888.0)*nyy*nyy*ny*nz - 19305*nyy*ny*nzz*nz + (1.0/1000000000000.0)*nyy*ny*nz - 7959776990330879.0/274877906944.0*ny*nzz*nzz*nz + 17820*ny*nzz*nz - 1620*ny*nz);  // power=9
    mat[20][19] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*ny*nz - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*ny*nz - 1447875.0/2.0*nxx*nxx*ny*nzz*nz - 5306517993553923.0/274877906944.0*nxx*nxx*ny*nz + (7462290928435199.0/17179869184.0)*nxx*nyy*ny*nzz*nz - 3979888495165441.0/137438953472.0*nxx*nyy*ny*nz + (7462290928435199.0/17179869184.0)*nxx*ny*nzz*nzz*nz - 2984916371374079.0/34359738368.0*nxx*ny*nzz*nz + 17820*nxx*ny*nz - 7959776990330879.0/274877906944.0*nyy*ny*nzz*nz + 4455*nyy*ny*nz - 7959776990330879.0/274877906944.0*ny*nzz*nzz*nz + 17820*ny*nzz*nz - 2565*ny*nz);  // power=9
    mat[20][20] = FAR_REAL((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nzz - 7075357324738559.0/549755813888.0*nxx*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nzz + (5306517993553919.0/549755813888.0)*nxx*nxx*nyy - 675675.0/2.0*nxx*nxx*nzz*nzz - 6633147491942401.0/137438953472.0*nxx*nxx*nzz + 8910*nxx*nxx + (4974860618956799.0/34359738368.0)*nxx*nyy*nzz*nzz - 4455.0/2.0*nxx*nyy + (4974860618956799.0/34359738368.0)*nxx*nzz*nzz*nzz - 5306517993553923.0/274877906944.0*nxx*nzz*nzz + (2143016882012161.0/137438953472.0)*nxx*nzz - 2025*nxx - 5306517993553919.0/549755813888.0*nyy*nzz*nzz + (4455.0/2.0)*nyy*nzz - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 1755*nzz + 105);  // power=9
    mat[20][21] = FAR_REAL((6122905377175370.0/916259689813.0)*nxx*nx*nyy*nyy*ny + 217350*nxx*nx*nyy*ny*nzz - 5207561947054079.0/274877906944.0*nxx*nx*nyy*ny - 6633147491942401.0/34359738368.0*nxx*nx*ny*nzz*nzz + (5306517993553919.0/137438953472.0)*nxx*nx*ny*nzz + (2025.0/2.0)*nxx*nx*ny + (3673743226306559.0/549755813888.0)*nx*nyy*nyy*nyy*ny - 3910310024970239.0/34359738368.0*nx*nyy*nyy*ny*nzz - 3104999999999999.0/1000000000000.0*nx*nyy*nyy*ny + (6679533138739199.0/274877906944.0)*nx*nyy*ny*nzz*nzz + 27540*nx*nyy*ny*nzz + (2632499999997368.0/999999999999.0)*nx*nyy*ny + (4974860618956799.0/34359738368.0)*nx*ny*nzz*nzz*nzz - 7296462241136639.0/68719476736.0*nx*ny*nzz*nzz + (28755.0/2.0)*nx*ny*nzz - 2025.0/2.0*nx*ny);  // power=9
    mat[20][22] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz + 292410*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz - 289575*nxx*nxx*nx*nzz*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*nxx*nx*nz - 8217217330053119.0/34359738368.0*nxx*nx*nyy*nyy*nz + (38475.0/2.0)*nxx*nx*nyy*nyy*nz - 5315795122913279.0/17179869184.0*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz - 4879770043023361.0/137438953472.0*nxx*nx*nyy*nz + (1254825.0/4.0)*nxx*nx*nzz*nzz*nz + 2160*nxx*nx*nzz*nzz*nz + (2476375063657856.0/769658139443.0)*nxx*nx*nzz*nz - 3924225719009281.0/274877906944.0*nxx*nx*nz - 99225.0/4.0*nx*nyy*nyy*nyy*nz + 10125*nx*nyy*nyy*nyy*nz + 169695*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz + (161595.0/4.0)*nx*nyy*nyy*nz - 22680*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (1280243851591681.0/137438953472.0)*nx*nyy*nzz*nz - 4174708211711999.0/549755813888.0*nx*nyy*nz - 4974860618956799.0/68719476736.0*nx*nzz*nzz*nzz*nz - 3240*nx*nzz*nzz*nzz*nz + (5306517993553917.0/137438953472.0)*nx*nzz*nzz*nz - 8256645129830397.0/549755813888.0*nx*nzz*nz + (1725546060837821.0/549755813887.0)*nx*nz);  // power=9
    mat[20][23] = FAR_REAL(630*nxx*nxx*nx*nyy*ny + 579150*nxx*nxx*nx*ny*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nx*ny + 1260*nxx*nx*nyy*nyy*ny - 7862753679114239.0/17179869184.0*nxx*nx*nyy*ny*nzz + (7959776990330879.0/274877906944.0)*nxx*nx*nyy*ny - 656370*nxx*nx*ny*nzz*nzz - 1199068969697281.0/34359738368.0*nxx*nx*ny*nzz + (25515.0/2.0)*nxx*nx*ny + 630*nx*nyy*nyy*nyy*ny - 6407404010864639.0/274877906944.0*nx*nyy*nyy*ny*nzz + (3157316358635519.0/34359738368.0)*nx*nyy*ny*nzz*nzz + (133785.0/2.0)*nx*nyy*ny*nzz - 11385.0/2.0*nx*nyy*ny + (7959776990330879.0/68719476736.0)*nx*ny*nzz*nzz*nzz - 5937362789990950.0/999556025251.0*nx*ny*nzz*nzz - 534362651098651.0/879609302220.0*nx*ny*nzz - 1215.0/2.0*nx*ny);  // power=9
    mat[20][24] = FAR_REAL(579150*nxx*nxx*nyy*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nyy - 6633147491942401.0/34359738368.0*nxx*nxx*nzz*nzz + (5306517993553919.0/137438953472.0)*nxx*nxx*nzz - 7462290928435199.0/17179869184.0*nxx*nyy*nyy*nzz + (7959776990330879.0/274877906944.0)*nxx*nyy*nyy - 4974860618956799.0/17179869184.0*nxx*nyy*nzz*nzz + (3979888495165437.0/68719476736.0)*nxx*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nzz*nzz*nzz - 6633147491942403.0/137438953472.0*nxx*nzz*nzz + (6.0/753950830475.0)*nxx*nzz + (7959776990330879.0/274877906944.0)*nyy*nyy*nzz - 4455.0/2.0*nyy*nyy + 19305*nyy*nzz*nzz - 3673743226306559.0/274877906944.0*nyy*nzz + 810*nyy - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 810*nzz);  // power=9
    mat[20][25] = FAR_REAL(270270*nxx*nxx*nyy*nz - 20790*nxx*nxx*nz - 405405.0/2.0*nxx*nyy*nyy*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (31185.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz + 2835*nxx*nz + (31185.0/2.0)*nyy*nyy*nz + (31185.0/2.0)*nyy*nzz*nz - 14175.0/2.0*nyy*nz - 2835.0/2.0*nzz*nz + 315*nz);  // power=8
    mat[20][26] = FAR_REAL(270270*nxx*nxx*nzz*nz - 62370*nxx*nxx*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (93555.0/2.0)*nxx*nyy*nz - 405405.0/2.0*nxx*nzz*nzz*nz + (93555.0/2.0)*nxx*nzz*nz + 8505*nxx*nz + (31185.0/2.0)*nyy*nzz*nz - 8505.0/2.0*nyy*nz + (31185.0/2.0)*nzz*nzz*nz - 19845.0/2.0*nzz*nz + 945*nz);  // power=8
    mat[20][27] = FAR_REAL(-25515.0/2.0*nxx*nxx*nx*ny*nz + (42525.0/2.0)*nxx*nx*nyy*ny*nz + (583065.0/2.0)*nxx*nx*ny*nzz*nz - 62370*nxx*nx*ny*nz + 34020*nx*nyy*nyy*ny*nz - 269325.0/2.0*nx*nyy*ny*nzz*nz - 337365.0/2.0*nx*ny*nzz*nzz*nz + 93555*nx*ny*nzz*nz - 8505.0/2.0*nx*ny*nz);  // power=8

    // Row 21 (target order 4)
    mat[21][0] = FAR_REAL(-1575*nxx*ny*nz + 2100*nyy*ny*nz - 1575*ny*nzz*nz);  // power=5
    mat[21][1] = FAR_REAL(-11025*nxx*nx*ny*nz + 22050*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[21][2] = FAR_REAL(-14175*nxx*nyy*nz + 1575*nxx*nz + 18900*nyy*nyy*nz - 14175*nyy*nzz*nz - 6300*nyy*nz + 1575*nzz*nz);  // power=6
    mat[21][3] = FAR_REAL(-14175*nxx*ny*nzz + 1575*nxx*ny + 18900*nyy*ny*nzz - 2100*nyy*ny - 14175*ny*nzz*nzz + 4725*ny*nzz);  // power=6
    mat[21][4] = FAR_REAL(-17325*nxx*nxx*ny*nz + (63525.0/2.0)*nxx*nyy*ny*nz - 17325.0/2.0*nxx*ny*nzz*nz + 3150*nxx*ny*nz - 11550*nyy*nyy*ny*nz - 5775.0/2.0*nyy*ny*nzz*nz + 8400*nyy*ny*nz + (17325.0/2.0)*ny*nzz*nzz*nz - 6300*ny*nzz*nz - 525*ny*nz);  // power=7
    mat[21][5] = FAR_REAL(-17325.0/2.0*nxx*nxx*ny*nz + (5775.0/2.0)*nxx*nyy*ny*nz + (17325.0/2.0)*nxx*ny*nzz*nz + 1575*nxx*ny*nz + 11550*nyy*nyy*ny*nz - 63525.0/2.0*nyy*ny*nzz*nz - 2100*nyy*ny*nz + 17325*ny*nzz*nzz*nz - 7875*ny*nzz*nz + 1575*ny*nz);  // power=7
    mat[21][6] = FAR_REAL(-1575*nxx*nxx*nx*nz - 49350*nxx*nx*nyy*nz + 1575*nxx*nx*nzz*nz + 4725*nxx*nx*nz + 73500*nx*nyy*nyy*nz - 44625*nx*nyy*nzz*nz - 15225*nx*nyy*nz + 3150*nx*nzz*nzz*nz + 525*nx*nz);  // power=7
    mat[21][7] = FAR_REAL(2100*nxx*nxx*nx*ny - 2100*nxx*nx*nyy*ny - 66675*nxx*nx*ny*nzz + 4725*nxx*nx*ny - 4200*nx*nyy*nyy*ny + 48300*nx*nyy*ny*nzz - 68775*nx*ny*nzz*nzz + 42525*nx*ny*nzz - 3150*nx*ny);  // power=7
    mat[21][8] = FAR_REAL(-51975*nxx*nyy*nzz + 4725*nxx*nyy + 4725*nxx*nzz + 69300*nyy*nyy*nzz - 6300*nyy*nyy - 51975*nyy*nzz*nzz - 4725*nyy*nzz + 2625*nyy + 4725*nzz*nzz - 1050*nzz - 525);  // power=7
    mat[21][9] = FAR_REAL(-135135.0/2.0*nxx*nxx*nx*ny*nz + 90090*nxx*nx*nyy*ny*nz - 135135.0/2.0*nxx*nx*ny*nzz*nz + (93555.0/2.0)*nxx*nx*ny*nz - 20790*nx*nyy*ny*nz + (31185.0/2.0)*nx*ny*nzz*nz - 5670*nx*ny*nz);  // power=8
    mat[21][10] = FAR_REAL(-135135.0/2.0*nxx*nyy*nyy*nz + 31185*nxx*nyy*nz - 2835.0/2.0*nxx*nz + 90090*nyy*nyy*nyy*nz - 135135.0/2.0*nyy*nyy*nzz*nz - 83160*nyy*nyy*nz + 31185*nyy*nzz*nz + 17010*nyy*nz - 2835.0/2.0*nzz*nz - 420*nz);  // power=8
    mat[21][11] = FAR_REAL(-135135.0/2.0*nxx*ny*nzz*nzz + 31185*nxx*ny*nzz - 2835.0/2.0*nxx*ny + 90090*nyy*ny*nzz*nzz - 41580*nyy*ny*nzz + 1890*nyy*ny - 135135.0/2.0*ny*nzz*nzz*nzz + 62370*ny*nzz*nzz - 25515.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[21][12] = FAR_REAL(-405405.0/2.0*nxx*nxx*nyy*nz + (31185.0/2.0)*nxx*nxx*nz + 270270*nxx*nyy*nyy*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (31185.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz - 14175.0/2.0*nxx*nz - 20790*nyy*nyy*nz + (31185.0/2.0)*nyy*nzz*nz + 2835*nyy*nz - 2835.0/2.0*nzz*nz + 315*nz);  // power=8
    mat[21][13] = FAR_REAL(-405405.0/2.0*nxx*nxx*ny*nzz + (31185.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*ny*nzz - 20790*nxx*nyy*ny - 405405.0/2.0*nxx*ny*nzz*nzz + 124740*nxx*ny*nzz - 14175.0/2.0*nxx*ny - 20790*nyy*ny*nzz + 1890*nyy*ny + (31185.0/2.0)*ny*nzz*nzz - 14175.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[21][14] = FAR_REAL(-405405.0/2.0*nxx*nyy*ny*nzz + (31185.0/2.0)*nxx*nyy*ny + (93555.0/2.0)*nxx*ny*nzz - 8505.0/2.0*nxx*ny + 270270*nyy*nyy*ny*nzz - 20790*nyy*nyy*ny - 405405.0/2.0*nyy*ny*nzz*nzz - 197505.0/2.0*nyy*ny*nzz + 13230*nyy*ny + (93555.0/2.0)*ny*nzz*nzz - 2835.0/2.0*ny*nzz - 1260*ny);  // power=8
    mat[21][15] = FAR_REAL(945*nxx*nxx*nxx*nx - 5670*nxx*nxx*nx*nyy - 14175*nxx*nxx*nx*nzz - 26460*nxx*nx*nyy*nyy - 14175*nxx*nx*nyy*nyy - 475965*nxx*nx*nyy*nzz + 141750*nxx*nx*nyy*nzz + 48825*nxx*nx*nyy + 31185*nxx*nx*nzz - 4095*nxx*nx - 26460*nx*nyy*nyy*nyy - 7560*nx*nyy*nyy*nyy + 443520*nx*nyy*nyy*nzz + 155925*nx*nyy*nyy*nzz + 2520*nx*nyy*nyy - 70560*nx*nyy*nzz*nzz - 226800*nx*nyy*nzz*nzz - 67725*nx*nyy*nzz - 4725*nx*nyy + 15120*nx*nzz*nzz*nzz + 1575*nx*nzz + 945*nx);  // power=8
    mat[21][16] = FAR_REAL(-4974860618956799.0/137438953472.0*nxx*nxx*nxx*ny*nz + (6633147491942401.0/137438953472.0)*nxx*nxx*nyy*ny*nz + (6218575773695999.0/34359738368.0)*nxx*nxx*ny*nzz*nz - 5306517993553921.0/549755813888.0*nxx*nxx*ny*nz - 289575*nxx*nyy*ny*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*nyy*ny*nz + (6218575773695999.0/34359738368.0)*nxx*ny*nzz*nzz*nz - 135135*nxx*ny*nzz*nz + (3673743226306559.0/274877906944.0)*nxx*ny*nz + (6633147491942401.0/137438953472.0)*nyy*ny*nzz*nzz*nz - 3537678662369279.0/274877906944.0*nyy*ny*nzz*nz - 1.0/1000000000000.0*nyy*ny*nz - 4974860618956799.0/137438953472.0*ny*nzz*nzz*nzz*nz + (1989944247582719.0/68719476736.0)*ny*nzz*nzz*nz - 2449162150871039.0/549755813888.0*ny*nzz*nz);  // power=9
    mat[21][17] = FAR_REAL(-4974860618956799.0/137438953472.0*nxx*nyy*nyy*ny*nz + (7462290928435199.0/34359738368.0)*nxx*nyy*ny*nzz*nz - 5306517993553921.0/274877906944.0*nxx*nyy*ny*nz - 4974860618956799.0/137438953472.0*nxx*ny*nzz*nzz*nz - 5306517993553921.0/274877906944.0*nxx*ny*nzz*nz + 4455*nxx*ny*nz + (6633147491942401.0/137438953472.0)*nyy*nyy*nyy*ny*nz - 5596718196326401.0/17179869184.0*nyy*nyy*ny*nzz*nz + (6.0/753950830475.0)*nyy*nyy*ny*nz + (1061775.0/4.0)*nyy*ny*nzz*nzz*nz + (7075357324738561.0/274877906944.0)*nyy*ny*nzz*nz - 2040968459058460.0/916259689813.0*nyy*ny*nz - 4974860618956799.0/137438953472.0*ny*nzz*nzz*nzz*nz - 5.0/837723144972.0*ny*nzz*nzz*nz - 222651104624437.0/99955602525.0*ny*nzz*nz + 270*ny*nz);  // power=9
    mat[21][18] = FAR_REAL(-3037499999996962.0/999999999999.0*nxx*nxx*nxx*nx*nz - 9450*nxx*nxx*nx*nyy*nz + (6679533138739199.0/274877906944.0)*nxx*nxx*nx*nyy*nz + (26325.0/2.0)*nxx*nxx*nx*nzz*nz - 264600*nxx*nx*nyy*nyy*nz + (115425.0/2.0)*nxx*nx*nyy*nyy*nz + (1001025.0/2.0)*nxx*nx*nyy*nzz*nz - 196425*nxx*nx*nyy*nzz*nz - 4912239995781121.0/68719476736.0*nxx*nx*nyy*nz + 6480*nxx*nx*nzz*nzz*nz - 7959776990330879.0/274877906944.0*nxx*nx*nzz*nz + (4119045435563332.0/549755813889.0)*nxx*nx*nz + (165375.0/2.0)*nx*nyy*nyy*nyy*nz + (8349416423423999.0/274877906944.0)*nx*nyy*nyy*nyy*nz - 6517183374950399.0/8589934592.0*nx*nyy*nyy*nzz*nz - 419175.0/2.0*nx*nyy*nyy*nzz*nz + (5371457899069439.0/137438953472.0)*nx*nyy*nyy*nz + 75600*nx*nyy*nzz*nzz*nz + 184680*nx*nyy*nzz*nzz*nz + (150255.0/2.0)*nx*nyy*nzz*nz + 39420*nx*nyy*nz - 9720*nx*nzz*nzz*nzz*nz - 2835.0/2.0*nx*nzz*nz - 2565*nx*nz);  // power=9
    mat[21][19] = FAR_REAL(-3037499999996962.0/999999999999.0*nxx*nxx*nxx*nx*nz - 5234620241018879.0/34359738368.0*nxx*nxx*nx*nyy*nz + (6679533138739199.0/274877906944.0)*nxx*nxx*nx*nyy*nz + (26325.0/2.0)*nxx*nxx*nx*nzz*nz + (5306517993553919.0/549755813888.0)*nxx*nxx*nx*nz + (8646284562923521.0/34359738368.0)*nxx*nx*nyy*nyy*nz + (115425.0/2.0)*nxx*nx*nyy*nyy*nz + (5884019296174079.0/17179869184.0)*nxx*nx*nyy*nzz*nz - 196425*nxx*nx*nyy*nzz*nz - 2495547797667841.0/34359738368.0*nxx*nx*nyy*nz + 6480*nxx*nx*nzz*nzz*nz - 19305*nxx*nx*nzz*nz + (1781208836997931.0/733007751851.0)*nxx*nx*nz + 66150*nx*nyy*nyy*nyy*nz + (8349416423423999.0/274877906944.0)*nx*nyy*nyy*nyy*nz - 452520*nx*nyy*nyy*nzz*nz - 419175.0/2.0*nx*nyy*nyy*nzz*nz - 30510*nx*nyy*nyy*nz + 60480*nx*nyy*nzz*nzz*nz + 184680*nx*nyy*nzz*nzz*nz + (2463077844910081.0/34359738368.0)*nx*nyy*nzz*nz + (5826037237678081.0/549755813888.0)*nx*nyy*nz - 9720*nx*nzz*nzz*nzz*nz - 1781208836997526.0/879609302221.0*nx*nzz*nz - 2025.0/2.0*nx*nz);  // power=9
    mat[21][20] = FAR_REAL(-840*nxx*nxx*nx*nyy*ny - 4974860618956799.0/34359738368.0*nxx*nxx*nx*ny*nzz + (5306517993553919.0/549755813888.0)*nxx*nxx*nx*ny - 1680*nxx*nx*nyy*nyy*ny + (7701048160419841.0/34359738368.0)*nxx*nx*nyy*ny*nzz - 7075357324738559.0/549755813888.0*nxx*nx*nyy*ny + (5306517993553919.0/137438953472.0)*nxx*nx*ny*nzz*nzz + (5510614839459837.0/137438953472.0)*nxx*nx*ny*nzz - 3644999999999999.0/1000000000000.0*nxx*nx*ny - 840*nx*nyy*nyy*nyy*ny + (8543205347819519.0/274877906944.0)*nx*nyy*nyy*ny*nzz - 122520*nx*nyy*ny*nzz*nzz - 11970*nx*nyy*ny*nzz + 1650*nx*nyy*ny + (6301490117345279.0/34359738368.0)*nx*ny*nzz*nzz*nzz - 5587151156674559.0/34359738368.0*nx*ny*nzz*nzz + 31995*nx*ny*nzz - 810*nx*ny);  // power=9
    mat[21][21] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nyy*nyy*nzz + (5306517993553919.0/549755813888.0)*nxx*nyy*nyy + (4974860618956799.0/34359738368.0)*nxx*nyy*nzz*nzz - 4455.0/2.0*nxx*nyy - 5306517993553919.0/549755813888.0*nxx*nzz*nzz + (4455.0/2.0)*nxx*nzz + (6633147491942401.0/34359738368.0)*nyy*nyy*nyy*nzz - 7075357324738559.0/549755813888.0*nyy*nyy*nyy - 675675.0/2.0*nyy*nyy*nzz*nzz - 6633147491942401.0/137438953472.0*nyy*nyy*nzz + 8910*nyy*nyy + (4974860618956799.0/34359738368.0)*nyy*nzz*nzz*nzz - 5306517993553923.0/274877906944.0*nyy*nzz*nzz + (2143016882012161.0/137438953472.0)*nyy*nzz - 2025*nyy - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 1755*nzz + 105);  // power=9
    mat[21][22] = FAR_REAL(-7462290928435199.0/34359738368.0*nxx*nxx*nyy*ny*nz + (7462290928435199.0/34359738368.0)*nxx*nxx*ny*nzz*nz + 289575*nxx*nyy*nyy*ny*nz - 289575*nxx*nyy*ny*nzz*nz - 6633147491942401.0/137438953472.0*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz - 8623091739525119.0/68719476736.0*nxx*ny*nzz*nz + (5510614839459839.0/274877906944.0)*nxx*ny*nz - 289575*nyy*nyy*ny*nzz*nz + (5306517993553921.0/137438953472.0)*nyy*nyy*ny*nz + (1254825.0/4.0)*nyy*ny*nzz*nzz*nz + (2476375063657856.0/769658139443.0)*nyy*ny*nzz*nz - 7347486452613121.0/549755813888.0*nyy*ny*nz - 4974860618956799.0/68719476736.0*ny*nzz*nzz*nzz*nz + (7959776990330875.0/274877906944.0)*ny*nzz*nzz*nz - 1836871613153279.0/274877906944.0*ny*nzz*nz + 1215*ny*nz);  // power=9
    mat[21][23] = FAR_REAL(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*nzz + (7959776990330879.0/274877906944.0)*nxx*nxx*nyy + (7959776990330879.0/274877906944.0)*nxx*nxx*nzz - 4455.0/2.0*nxx*nxx + 579150*nxx*nyy*nyy*nzz - 5306517993553921.0/137438953472.0*nxx*nyy*nyy - 4974860618956799.0/17179869184.0*nxx*nyy*nzz*nzz + (3979888495165437.0/68719476736.0)*nxx*nyy*nzz + 19305*nxx*nzz*nzz - 3673743226306559.0/274877906944.0*nxx*nzz + 810*nxx - 6633147491942401.0/34359738368.0*nyy*nyy*nzz*nzz + (5306517993553919.0/137438953472.0)*nyy*nyy*nzz + (4974860618956799.0/34359738368.0)*nyy*nzz*nzz*nzz - 6633147491942403.0/137438953472.0*nyy*nzz*nzz + (6.0/753950830475.0)*nyy*nzz - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 810*nzz);  // power=9
    mat[21][24] = FAR_REAL(-2730*nxx*nxx*nx*nyy*ny - 8848319824527361.0/274877906944.0*nxx*nx*nyy*nyy*ny - 861105.0/2.0*nxx*nx*nyy*ny*nzz + (106515.0/2.0)*nxx*nx*nyy*ny + 270270*nxx*nx*ny*nzz*nzz - 7959776990330877.0/274877906944.0*nxx*nx*ny*nzz - 11745.0/2.0*nxx*nx*ny - 8097903138570241.0/274877906944.0*nx*nyy*nyy*nyy*ny + 556230*nx*nyy*nyy*ny*nzz + (6827967208488961.0/549755813888.0)*nx*nyy*nyy*ny - 495390*nx*nyy*ny*nzz*nzz - 6263608505794561.0/68719476736.0*nx*nyy*ny*nzz - 5290025319137279.0/549755813888.0*nx*nyy*ny + 270270*nx*ny*nzz*nzz*nzz - 7959776990330879.0/34359738368.0*nx*ny*nzz*nzz + (139725.0/2.0)*nx*ny*nzz + (405.0/2.0)*nx*ny);  // power=9
    mat[21][25] = FAR_REAL(-25515.0/2.0*nxx*nxx*nx*ny*nz - 219240*nxx*nx*nyy*ny*nz + (42525.0/2.0)*nxx*nx*ny*nzz*nz + (93555.0/2.0)*nxx*nx*ny*nz + 266490*nx*nyy*nyy*ny*nz - 344925.0/2.0*nx*nyy*ny*nzz*nz - 246645.0/2.0*nx*nyy*ny*nz + 34020*nx*ny*nzz*nzz*nz + (31185.0/2.0)*nx*ny*nz);  // power=8
    mat[21][26] = FAR_REAL(17010*nxx*nxx*nx*ny*nz - 28350*nxx*nx*nyy*ny*nz - 462105.0/2.0*nxx*nx*ny*nzz*nz + (93555.0/2.0)*nxx*nx*ny*nz - 45360*nx*nyy*nyy*ny*nz + 179550*nx*nyy*ny*nzz*nz - 496125.0/2.0*nx*ny*nzz*nzz*nz + (405405.0/2.0)*nx*ny*nzz*nz - 34020*nx*ny*nz);  // power=8
    mat[21][27] = FAR_REAL(-405405.0/2.0*nxx*nyy*nzz*nz + (93555.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz - 8505.0/2.0*nxx*nz + 270270*nyy*nyy*nzz*nz - 62370*nyy*nyy*nz - 405405.0/2.0*nyy*nzz*nzz*nz + (93555.0/2.0)*nyy*nzz*nz + 8505*nyy*nz + (31185.0/2.0)*nzz*nzz*nz - 19845.0/2.0*nzz*nz + 945*nz);  // power=8

    // Row 22 (target order 4)
    mat[22][0] = FAR_REAL(-525*nxx*nxx + 2625*nxx*nyy - 525*nxx*nzz + 150*nxx - 525*nyy*nyy - 525*nyy*nzz + 150*nyy + 150*nzz - 45);  // power=5
    mat[22][1] = FAR_REAL(-4800*nxx*nxx*nx + (6597069766656001.0/274877906944.0)*nxx*nx*nyy - 4350*nxx*nx*nzz + 3150*nxx*nx - 4275*nx*nyy*nyy - 3825*nx*nyy*nzz - 4725*nx*nyy + 450*nx*nzz*nzz + 1575*nx*nzz - 450*nx);  // power=6
    mat[22][2] = FAR_REAL(-4725*nxx*nxx*ny + 23625*nxx*nyy*ny - 4725*nxx*ny*nzz - 4200*nxx*ny - 4725*nyy*nyy*ny - 4725*nyy*ny*nzz + 3150*nyy*ny + 2100*ny*nzz - 525*ny);  // power=6
    mat[22][3] = FAR_REAL(-4725*nxx*nxx*nz + 23625*nxx*nyy*nz - 4725*nxx*nzz*nz + 2100*nxx*nz - 4725*nyy*nyy*nz - 4725*nyy*nzz*nz + 2100*nyy*nz + 1050*nzz*nz - 525*nz);  // power=6
    mat[22][4] = FAR_REAL(-5775*nxx*nxx*nxx + (63525.0/2.0)*nxx*nxx*nyy - 5775.0/2.0*nxx*nxx*nzz + 5250*nxx*nxx - 40425.0/2.0*nxx*nyy*nyy - 17325*nxx*nyy*nzz - 4725*nxx*nyy + (5775.0/2.0)*nxx*nzz*nzz + 1575*nxx*nzz - 1400*nxx + (5775.0/2.0)*nyy*nyy*nyy + 5775*nyy*nyy*nzz - 2625*nyy*nyy + (5775.0/2.0)*nyy*nzz*nzz - 3150*nyy*nzz + (2975.0/2.0)*nyy - 525*nzz*nzz + (875.0/2.0)*nzz - 175);  // power=7
    mat[22][5] = FAR_REAL(-5775.0/2.0*nxx*nxx*nxx + 11550*nxx*nxx*nyy + (5775.0/2.0)*nxx*nxx*nzz + 2625*nxx*nxx + 11550*nxx*nyy*nyy - 34650*nxx*nyy*nzz - 9450*nxx*nyy + 5775*nxx*nzz*nzz - 1575*nxx*nzz + (175.0/2.0)*nxx - 5775.0/2.0*nyy*nyy*nyy + (5775.0/2.0)*nyy*nyy*nzz + 2625*nyy*nyy + 5775*nyy*nzz*nzz - 1575*nyy*nzz + (175.0/2.0)*nyy - 1050*nzz*nzz + 875*nzz - 350);  // power=7
    mat[22][6] = FAR_REAL(-17850*nxx*nxx*nx*ny + 87150*nxx*nx*nyy*ny - 13650*nxx*nx*ny*nzz - 6300*nxx*nx*ny - 16275*nx*nyy*nyy*ny - 12075*nx*nyy*ny*nzz - 7875*nx*nyy*ny + 4200*nx*ny*nzz*nzz + 4725*nx*ny*nzz + 2100*nx*ny);  // power=7
    mat[22][7] = FAR_REAL(-17850*nxx*nxx*nx*nz + 92400*nxx*nx*nyy*nz - 16800*nxx*nx*nzz*nz + 12600*nxx*nx*nz - 11025*nx*nyy*nyy*nz - 9975*nx*nyy*nzz*nz - 16275*nx*nyy*nz + 1050*nx*nzz*nzz*nz + 4725*nx*nzz*nz - 2100*nx*nz);  // power=7
    mat[22][8] = FAR_REAL(-17325*nxx*nxx*ny*nz + 86625*nxx*nyy*ny*nz - 17325*nxx*ny*nzz*nz - 9450*nxx*ny*nz - 17325*nyy*nyy*ny*nz - 17325*nyy*ny*nzz*nz + 12600*nyy*ny*nz + 6300*ny*nzz*nz - 2625*ny*nz);  // power=7
    mat[22][9] = FAR_REAL(-45045.0/2.0*nxx*nxx*nxx*nx + (225225.0/2.0)*nxx*nxx*nx*nyy - 45045.0/2.0*nxx*nxx*nx*nzz + (58905.0/2.0)*nxx*nxx*nx - 45045.0/2.0*nxx*nx*nyy*nyy - 45045.0/2.0*nxx*nx*nyy*nzz - 148995.0/2.0*nxx*nx*nyy + (38115.0/2.0)*nxx*nx*nzz - 21735.0/2.0*nxx*nx + (10395.0/2.0)*nx*nyy*nyy + (10395.0/2.0)*nx*nyy*nzz + 8505*nx*nyy - 2835*nx*nzz + (1995.0/2.0)*nx);  // power=8
    mat[22][10] = FAR_REAL(-45045.0/2.0*nxx*nxx*nyy*ny + (10395.0/2.0)*nxx*nxx*ny + (225225.0/2.0)*nxx*nyy*nyy*ny - 45045.0/2.0*nxx*nyy*ny*nzz - 148995.0/2.0*nxx*nyy*ny + (10395.0/2.0)*nxx*ny*nzz + 8505*nxx*ny - 45045.0/2.0*nyy*nyy*nyy*ny - 45045.0/2.0*nyy*nyy*ny*nzz + (58905.0/2.0)*nyy*nyy*ny + (38115.0/2.0)*nyy*ny*nzz - 21735.0/2.0*nyy*ny - 2835*ny*nzz + (1995.0/2.0)*ny);  // power=8
    mat[22][11] = FAR_REAL(-45045.0/2.0*nxx*nxx*nzz*nz + (10395.0/2.0)*nxx*nxx*nz + (225225.0/2.0)*nxx*nyy*nzz*nz - 51975.0/2.0*nxx*nyy*nz - 45045.0/2.0*nxx*nzz*nzz*nz + (38115.0/2.0)*nxx*nzz*nz - 2835*nxx*nz - 45045.0/2.0*nyy*nyy*nzz*nz + (10395.0/2.0)*nyy*nyy*nz - 45045.0/2.0*nyy*nzz*nzz*nz + (38115.0/2.0)*nyy*nzz*nz - 2835*nyy*nz + 3465*nzz*nzz*nz - 6615.0/2.0*nzz*nz + (1155.0/2.0)*nz);  // power=8
    mat[22][12] = FAR_REAL(-135135.0/2.0*nxx*nxx*nxx*ny + (675675.0/2.0)*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*ny*nzz + (10395.0/2.0)*nxx*nxx*ny - 135135.0/2.0*nxx*nyy*nyy*ny - 135135.0/2.0*nxx*nyy*ny*nzz - 197505.0/2.0*nxx*nyy*ny + (93555.0/2.0)*nxx*ny*nzz + (19845.0/2.0)*nxx*ny + (10395.0/2.0)*nyy*nyy*ny + (10395.0/2.0)*nyy*ny*nzz + 1890*nyy*ny - 2835*ny*nzz - 945.0/2.0*ny);  // power=8
    mat[22][13] = FAR_REAL(-135135.0/2.0*nxx*nxx*nxx*nz + (675675.0/2.0)*nxx*nxx*nyy*nz - 135135.0/2.0*nxx*nxx*nzz*nz + (135135.0/2.0)*nxx*nxx*nz - 135135.0/2.0*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 218295.0/2.0*nxx*nyy*nz + (72765.0/2.0)*nxx*nzz*nz - 36855.0/2.0*nxx*nz + (10395.0/2.0)*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + 2835*nyy*nz - 1890*nzz*nz + (1575.0/2.0)*nz);  // power=8
    mat[22][14] = FAR_REAL(-135135.0/2.0*nxx*nxx*nyy*nz + (10395.0/2.0)*nxx*nxx*nz + (675675.0/2.0)*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 218295.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + 2835*nxx*nz - 135135.0/2.0*nyy*nyy*nyy*nz - 135135.0/2.0*nyy*nyy*nzz*nz + (135135.0/2.0)*nyy*nyy*nz + (72765.0/2.0)*nyy*nzz*nz - 36855.0/2.0*nyy*nz - 1890*nzz*nz + (1575.0/2.0)*nz);  // power=8
    mat[22][15] = FAR_REAL(-143640*nxx*nxx*nx*ny*nz + 708750*nxx*nx*nyy*ny*nz - 120960*nxx*nx*ny*nzz*nz - 20790*nxx*nx*ny*nz - 93555*nx*nyy*nyy*ny*nz - 70875*nx*nyy*ny*nzz*nz - 70875*nx*nyy*ny*nz + 22680*nx*ny*nzz*nzz*nz + 31185*nx*ny*nzz*nz + 5670*nx*ny*nz);  // power=8
    mat[22][16] = FAR_REAL(-96525.0/8.0*nxx*nxx*nxx*nxx + (482625.0/8.0)*nxx*nxx*nxx*nyy + (482625.0/8.0)*nxx*nxx*nxx*nzz + (57915.0/4.0)*nxx*nxx*nxx - 96525.0/8.0*nxx*nxx*nyy*nyy - 2992275.0/8.0*nxx*nxx*nyy*nzz - 122265.0/4.0*nxx*nxx*nyy + (482625.0/8.0)*nxx*nxx*nzz*nzz - 4090440953364479.0/68719476736.0*nxx*nxx*nzz - 4702731491077351.0/962072674303.0*nxx*nxx + (4974860618956799.0/68719476736.0)*nxx*nyy*nyy*nzz - 1.0/1000000000000.0*nxx*nyy*nyy + (1061775.0/8.0)*nxx*nyy*nzz*nzz + (135135.0/2.0)*nxx*nyy*nzz + 4455*nxx*nyy - 96525.0/8.0*nxx*nzz*nzz*nzz - 5748727826350079.0/274877906944.0*nxx*nzz*nzz + (4388082186977279.0/274877906944.0)*nxx*nzz + 270*nxx - 96525.0/8.0*nyy*nyy*nzz*nzz - 1.0/1000000000000.0*nyy*nyy*nzz - 96525.0/8.0*nyy*nzz*nzz*nzz + (32175.0/4.0)*nyy*nzz*nzz - 4455*nyy*nzz + (884419665590711.0/549755813887.0)*nzz*nzz*nzz - 380980779023171.0/879609302218.0*nzz*nzz - 270*nzz);  // power=9
    mat[22][17] = FAR_REAL(-96525.0/8.0*nxx*nxx*nyy*nyy + (4974860618956799.0/68719476736.0)*nxx*nxx*nyy*nzz - 1.0/1000000000000.0*nxx*nxx*nyy - 96525.0/8.0*nxx*nxx*nzz*nzz - 1.0/1000000000000.0*nxx*nxx*nzz + (482625.0/8.0)*nxx*nyy*nyy*nyy - 2992275.0/8.0*nxx*nyy*nyy*nzz - 122265.0/4.0*nxx*nyy*nyy + (1061775.0/8.0)*nxx*nyy*nzz*nzz + (135135.0/2.0)*nxx*nyy*nzz + 4455*nxx*nyy - 96525.0/8.0*nxx*nzz*nzz*nzz + (32175.0/4.0)*nxx*nzz*nzz - 4455*nxx*nzz - 96525.0/8.0*nyy*nyy*nyy*nyy + (482625.0/8.0)*nyy*nyy*nyy*nzz + (57915.0/4.0)*nyy*nyy*nyy + (482625.0/8.0)*nyy*nyy*nzz*nzz - 4090440953364479.0/68719476736.0*nyy*nyy*nzz - 4702731491077351.0/962072674303.0*nyy*nyy - 96525.0/8.0*nyy*nzz*nzz*nzz - 5748727826350079.0/274877906944.0*nyy*nzz*nzz + (4388082186977279.0/274877906944.0)*nyy*nzz + 270*nyy + (884419665590711.0/549755813887.0)*nzz*nzz*nzz - 380980779023171.0/879609302218.0*nzz*nzz - 270*nzz);  // power=9
    mat[22][18] = FAR_REAL(-95265.0/2.0*nxx*nxx*nx*nyy*ny + (4974860618956799.0/34359738368.0)*nxx*nxx*nx*ny*nzz - 1.0/879609302221.0*nxx*nxx*nx*ny + (476235.0/2.0)*nxx*nx*nyy*nyy*ny - 811710*nxx*nx*nyy*ny*nzz - 73170*nxx*nx*nyy*ny + (7959776990330879.0/68719476736.0)*nxx*nx*ny*nzz*nzz + (7347486452613125.0/549755813888.0)*nxx*nx*ny*nzz + (4193262470430719.0/549755813888.0)*nxx*nx*ny - 7158851488972799.0/137438953472.0*nx*nyy*nyy*nyy*ny + (7223791394488319.0/137438953472.0)*nx*nyy*nyy*ny*nzz + (2937757630463997.0/137438953472.0)*nx*nyy*nyy*ny + 75690*nx*nyy*ny*nzz*nzz + 90360*nx*nyy*ny*nzz - 2849934139196036.0/879609302221.0*nx*nyy*ny - 7959776990330879.0/274877906944.0*nx*ny*nzz*nzz*nzz - 31185.0/2.0*nx*ny*nzz*nzz - 2746030290370561.0/549755813888.0*nx*ny*nzz - 135.0/2.0*nx*ny);  // power=9
    mat[22][19] = FAR_REAL(-96525.0/2.0*nxx*nxx*nxx*nx*ny + (483885.0/2.0)*nxx*nxx*nx*nyy*ny + 96525*nxx*nxx*nx*ny*nzz + (7075357324738557.0/549755813888.0)*nxx*nxx*nx*ny - 7378410217144319.0/137438953472.0*nxx*nx*nyy*nyy*ny - 819810*nxx*nx*nyy*ny*nzz - 4889047172382719.0/68719476736.0*nxx*nx*nyy*ny + (7959776990330879.0/68719476736.0)*nxx*nx*ny*nzz*nzz + (6258969941114883.0/137438953472.0)*nxx*nx*ny*nzz + (673450872012925.0/610839793209.0)*nxx*nx*ny - 12105.0/2.0*nx*nyy*nyy*nyy*ny + 90495*nx*nyy*nyy*ny*nzz + (25515.0/2.0)*nx*nyy*nyy*ny + 67590*nx*nyy*ny*nzz*nzz + 68310*nx*nyy*ny*nzz + (343047627866384.0/586406201481.0)*nx*nyy*ny - 7959776990330879.0/274877906944.0*nx*ny*nzz*nzz*nzz - 31185.0/2.0*nx*ny*nzz*nzz - 2931572877563093.0/549755813889.0*nx*ny*nzz + 270*nx*ny);  // power=9
    mat[22][20] = FAR_REAL(-96525.0/2.0*nxx*nxx*nxx*nx*nz - 675.0/2.0*nxx*nxx*nxx*nx*nz + (483885.0/2.0)*nxx*nxx*nx*nyy*nz + 2700*nxx*nxx*nx*nyy*nz + (2925.0/2.0)*nxx*nxx*nx*nzz*nz + (7075357324738559.0/137438953472.0)*nxx*nxx*nx*nz - 53145*nxx*nx*nyy*nyy*nz + (3525309156556799.0/549755813888.0)*nxx*nx*nyy*nyy*nz - 293985*nxx*nx*nyy*nzz*nz - 21825*nxx*nx*nyy*nzz*nz - 2776954054901759.0/34359738368.0*nxx*nx*nyy*nz + (96525.0/2.0)*nxx*nx*nzz*nzz*nz + 720*nxx*nx*nzz*nzz*nz - 7075357324738559.0/549755813888.0*nxx*nx*nzz*nz - 4205631976243199.0/274877906944.0*nxx*nx*nz - 11025.0/2.0*nx*nyy*nyy*nyy*nz + (3374999999999999.0/1000000000000.0)*nx*nyy*nyy*nyy*nz + 37710*nx*nyy*nyy*nzz*nz - 6401219257958399.0/274877906944.0*nx*nyy*nyy*nzz*nz + (4935432819179519.0/549755813888.0)*nx*nyy*nyy*nz - 5039999999999999.0/1000000000000.0*nx*nyy*nzz*nzz*nz + (5640494650490879.0/274877906944.0)*nx*nyy*nzz*nzz*nz + (6475436292833279.0/137438953472.0)*nx*nyy*nzz*nz + (1529999999999999.0/1000000000000.0)*nx*nyy*nz - 1080*nx*nzz*nzz*nzz*nz - 5306517993553919.0/549755813888.0*nx*nzz*nzz*nz + (2572857208997009.0/733007751851.0)*nx*nzz*nz + 1080*nx*nz);  // power=9
    mat[22][21] = FAR_REAL(-96525.0/2.0*nxx*nxx*nyy*ny*nz + (96525.0/2.0)*nxx*nxx*ny*nzz*nz + (482625.0/2.0)*nxx*nyy*nyy*ny*nz - 289575*nxx*nyy*ny*nzz*nz - 5748727826350079.0/68719476736.0*nxx*nyy*ny*nz + (96525.0/2.0)*nxx*ny*nzz*nzz*nz + (6435000000000001.0/1000000000000.0)*nxx*ny*nzz*nz + (4898324301742079.0/549755813888.0)*nxx*ny*nz - 96525.0/2.0*nyy*nyy*nyy*ny*nz + (7075357324738559.0/137438953472.0)*nyy*nyy*ny*nz + (96525.0/2.0)*nyy*ny*nzz*nzz*nz - 7075357324738559.0/549755813888.0*nyy*ny*nzz*nz - 8299938400174077.0/549755813888.0*nyy*ny*nz - 3537678662369279.0/274877906944.0*ny*nzz*nzz*nz + (12375.0/2.0)*ny*nzz*nz + 540*ny*nz);  // power=9
    mat[22][22] = FAR_REAL(-4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy + (4974860618956799.0/68719476736.0)*nxx*nxx*nxx*nzz + (1447875.0/4.0)*nxx*nxx*nyy*nyy - 1447875.0/4.0*nxx*nxx*nyy*nzz - 6633147491942401.0/137438953472.0*nxx*nxx*nyy + (96525.0/2.0)*nxx*nxx*nzz*nzz - 135135.0/2.0*nxx*nxx*nzz + 4455*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy - 1447875.0/4.0*nxx*nyy*nyy*nzz - 6633147491942401.0/137438953472.0*nxx*nyy*nyy + (1061775.0/4.0)*nxx*nyy*nzz*nzz + (366795.0/2.0)*nxx*nyy*nzz + (8367970682142719.0/549755813888.0)*nxx*nyy - 96525.0/4.0*nxx*nzz*nzz*nzz - 12870*nxx*nzz*nzz + (46035.0/4.0)*nxx*nzz - 1620*nxx + (4974860618956799.0/68719476736.0)*nyy*nyy*nyy*nzz + (96525.0/2.0)*nyy*nyy*nzz*nzz - 135135.0/2.0*nyy*nyy*nzz + 4455*nyy*nyy - 96525.0/4.0*nyy*nzz*nzz*nzz - 12870*nyy*nzz*nzz + (46035.0/4.0)*nyy*nzz - 1620*nyy + (1768839331187857.0/549755813889.0)*nzz*nzz*nzz - 761961558049807.0/879609302222.0*nzz*nzz - 540*nzz + 105);  // power=9
    mat[22][23] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*ny*nz + (1447875.0/2.0)*nxx*nxx*nyy*ny*nz - 96525*nxx*nxx*ny*nzz*nz + (5306517993553919.0/274877906944.0)*nxx*nxx*ny*nz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*ny*nz - 386100*nxx*nyy*ny*nzz*nz - 115830*nxx*nyy*ny*nz + (96525.0/2.0)*nxx*ny*nzz*nzz*nz + 83655*nxx*ny*nzz*nz - 163277476724588.0/219902325555.0*nxx*ny*nz + (96525.0/2.0)*nyy*nyy*ny*nzz*nz + (1.0/879609302221.0)*nyy*nyy*ny*nz + (96525.0/2.0)*nyy*ny*nzz*nzz*nz - 5306517993553919.0/137438953472.0*nyy*ny*nzz*nz + (4898324301742079.0/549755813888.0)*nyy*ny*nz - 3537678662369279.0/274877906944.0*ny*nzz*nzz*nz + (3401614098431999.0/549755813888.0)*ny*nzz*nz - 1350*ny*nz);  // power=9
    mat[22][24] = FAR_REAL(-675.0/2.0*nxx*nxx*nxx*nx*nz - 5018153889300479.0/34359738368.0*nxx*nxx*nx*nyy*nz + 2700*nxx*nxx*nx*nyy*nz + (96525.0/2.0)*nxx*nxx*nx*nzz*nz + (2925.0/2.0)*nxx*nxx*nx*nzz*nz + (1.0/879609302221.0)*nxx*nxx*nx*nz + 878490*nxx*nx*nyy*nyy*nz + (3525309156556799.0/549755813888.0)*nxx*nx*nyy*nyy*nz - 377280*nxx*nx*nyy*nzz*nz - 21825*nxx*nx*nyy*nzz*nz - 121500*nxx*nx*nyy*nz + (96525.0/2.0)*nxx*nx*nzz*nzz*nz + 720*nxx*nx*nzz*nzz*nz - 5306517993553919.0/137438953472.0*nxx*nx*nzz*nz + (5120975406366719.0/549755813888.0)*nxx*nx*nz + 11025*nx*nyy*nyy*nyy*nz + (3374999999999999.0/1000000000000.0)*nx*nyy*nyy*nyy*nz + (138735.0/2.0)*nx*nyy*nyy*nzz*nz - 6401219257958399.0/274877906944.0*nx*nyy*nyy*nzz*nz - 4928474972160001.0/34359738368.0*nx*nyy*nyy*nz + 10080*nx*nyy*nzz*nzz*nz + (5640494650490879.0/274877906944.0)*nx*nyy*nzz*nzz*nz + (6790858691051519.0/68719476736.0)*nx*nyy*nzz*nz + (2533274790396478.0/879609302221.0)*nx*nyy*nz - 1080*nx*nzz*nzz*nzz*nz - 5306517993553919.0/549755813888.0*nx*nzz*nzz*nz + (3772699272805025.0/916259689813.0)*nx*nzz*nz - 2835.0/2.0*nx*nz);  // power=9
    mat[22][25] = FAR_REAL(-135135.0/2.0*nxx*nxx*nx*nyy + (10395.0/2.0)*nxx*nxx*nx + 341145*nxx*nx*nyy*nyy - 117495.0/2.0*nxx*nx*nyy*nzz - 201915.0/2.0*nxx*nx*nyy + (10395.0/2.0)*nxx*nx*nzz + (4095.0/2.0)*nxx*nx - 64260*nx*nyy*nyy*nyy - 55440*nx*nyy*nyy*nzz - 315*nx*nyy*nyy + 8820*nx*nyy*nzz*nzz + (71505.0/2.0)*nx*nyy*nzz + 12285*nx*nyy - 5355.0/2.0*nx*nzz - 630*nx);  // power=8
    mat[22][26] = FAR_REAL((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nx*nyy - 135135.0/2.0*nxx*nxx*nx*nzz - 4725.0/2.0*nxx*nxx*nx*nzz + (10395.0/2.0)*nxx*nxx*nx + (6615.0/2.0)*nxx*nx*nyy*nyy - 4725.0/2.0*nxx*nx*nyy*nyy + (693315.0/2.0)*nxx*nx*nyy*nzz + 23625*nxx*nx*nyy*nzz - 56385.0/2.0*nxx*nx*nyy - 135135.0/2.0*nxx*nx*nzz*nzz + (114345.0/2.0)*nxx*nx*nzz - 7245.0/2.0*nxx*nx + (6615.0/2.0)*nx*nyy*nyy*nyy - 1260*nx*nyy*nyy*nyy - 55440*nx*nyy*nyy*nzz + (51975.0/2.0)*nx*nyy*nyy*nzz - 315*nx*nyy*nyy + 8820*nx*nyy*nzz*nzz - 37800*nx*nyy*nzz*nzz - 115605.0/2.0*nx*nyy*nzz + 6615*nx*nyy + 2520*nx*nzz*nzz*nzz + (31185.0/2.0)*nx*nzz*nzz - 9765*nx*nzz + (945.0/2.0)*nx);  // power=8
    mat[22][27] = FAR_REAL(-135135.0/2.0*nxx*nxx*ny*nzz + (10395.0/2.0)*nxx*nxx*ny + (675675.0/2.0)*nxx*nyy*ny*nzz - 51975.0/2.0*nxx*nyy*ny - 135135.0/2.0*nxx*ny*nzz*nzz - 31185.0/2.0*nxx*ny*nzz + 2835*nxx*ny - 135135.0/2.0*nyy*nyy*ny*nzz + (10395.0/2.0)*nyy*nyy*ny - 135135.0/2.0*nyy*ny*nzz*nzz + (114345.0/2.0)*nyy*ny*nzz - 3780*nyy*ny + 20790*ny*nzz*nzz - 25515.0/2.0*ny*nzz + (1575.0/2.0)*ny);  // power=8

    // Row 23 (target order 4)
    mat[23][0] = FAR_REAL(3150*nxx*ny*nz - 525*nyy*ny*nz - 525*ny*nzz*nz);  // power=5
    mat[23][1] = FAR_REAL(22050*nxx*nx*ny*nz - 11025*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[23][2] = FAR_REAL(28350*nxx*nyy*nz - 3150*nxx*nz - 4725*nyy*nyy*nz - 4725*nyy*nzz*nz + 1575*nyy*nz + 525*nzz*nz);  // power=6
    mat[23][3] = FAR_REAL(28350*nxx*ny*nzz - 3150*nxx*ny - 4725*nyy*ny*nzz + 525*nyy*ny - 4725*ny*nzz*nzz + 1575*ny*nzz);  // power=6
    mat[23][4] = FAR_REAL(34650*nxx*nxx*ny*nz - 23100*nxx*nyy*ny*nz - 23100*nxx*ny*nzz*nz - 6300*nxx*ny*nz + (5775.0/2.0)*nyy*nyy*ny*nz + 5775*nyy*ny*nzz*nz - 2100*nyy*ny*nz + (5775.0/2.0)*ny*nzz*nzz*nz - 2100*ny*nzz*nz + 1050*ny*nz);  // power=7
    mat[23][5] = FAR_REAL(17325*nxx*nxx*ny*nz + (28875.0/2.0)*nxx*nyy*ny*nz - 75075.0/2.0*nxx*ny*nzz*nz - 3150*nxx*ny*nz - 5775.0/2.0*nyy*nyy*ny*nz + (5775.0/2.0)*nyy*ny*nzz*nz + 525*nyy*ny*nz + 5775*ny*nzz*nzz*nz - 2625*ny*nzz*nz + 525*ny*nz);  // power=7
    mat[23][6] = FAR_REAL(-525*nxx*nxx*nx*nz + 109725*nxx*nx*nyy*nz + 525*nxx*nx*nzz*nz - 9450*nxx*nx*nz - 11025*nx*nyy*nyy*nz - 9975*nx*nyy*nzz*nz - 21000*nx*nyy*nz + 1050*nx*nzz*nzz*nz + 2625*nx*nz);  // power=7
    mat[23][7] = FAR_REAL(-525*nxx*nxx*nx*ny + 525*nxx*nx*nyy*ny + 107625*nxx*nx*ny*nzz - 9450*nxx*nx*ny + 1050*nx*nyy*nyy*ny - 12075*nx*nyy*ny*nzz - 13125*nx*ny*nzz*nzz - 18900*nx*ny*nzz + 2625*nx*ny);  // power=7
    mat[23][8] = FAR_REAL(103950*nxx*nyy*nzz - 9450*nxx*nyy - 9450*nxx*nzz - 17325*nyy*nyy*nzz + 1575*nyy*nyy - 17325*nyy*nzz*nzz + 9450*nyy*nzz - 1575*nyy + 1575*nzz*nzz - 1575*nzz + 1050);  // power=7
    mat[23][9] = FAR_REAL(135135*nxx*nxx*nx*ny*nz - 45045.0/2.0*nxx*nx*nyy*ny*nz - 45045.0/2.0*nxx*nx*ny*nzz*nz - 93555*nxx*nx*ny*nz + (10395.0/2.0)*nx*nyy*ny*nz + (10395.0/2.0)*nx*ny*nzz*nz + 11340*nx*ny*nz);  // power=8
    mat[23][10] = FAR_REAL(135135*nxx*nyy*nyy*nz - 62370*nxx*nyy*nz + 2835*nxx*nz - 45045.0/2.0*nyy*nyy*nyy*nz - 45045.0/2.0*nyy*nyy*nzz*nz + 20790*nyy*nyy*nz + 10395*nyy*nzz*nz - 8505.0/2.0*nyy*nz - 945.0/2.0*nzz*nz + 105*nz);  // power=8
    mat[23][11] = FAR_REAL(135135*nxx*ny*nzz*nzz - 62370*nxx*ny*nzz + 2835*nxx*ny - 45045.0/2.0*nyy*ny*nzz*nzz + 10395*nyy*ny*nzz - 945.0/2.0*nyy*ny - 45045.0/2.0*ny*nzz*nzz*nzz + 20790*ny*nzz*nzz - 8505.0/2.0*ny*nzz + 105*ny);  // power=8
    mat[23][12] = FAR_REAL(405405*nxx*nxx*nyy*nz - 31185*nxx*nxx*nz - 135135.0/2.0*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 280665.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + 14175*nxx*nz + (10395.0/2.0)*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + (8505.0/2.0)*nyy*nz - 945.0/2.0*nzz*nz - 630*nz);  // power=8
    mat[23][13] = FAR_REAL(405405*nxx*nxx*ny*nzz - 31185*nxx*nxx*ny - 135135.0/2.0*nxx*nyy*ny*nzz + (10395.0/2.0)*nxx*nyy*ny - 135135.0/2.0*nxx*ny*nzz*nzz - 280665.0/2.0*nxx*ny*nzz + 14175*nxx*ny + (10395.0/2.0)*nyy*ny*nzz - 945.0/2.0*nyy*ny + (10395.0/2.0)*ny*nzz*nzz + (8505.0/2.0)*ny*nzz - 630*ny);  // power=8
    mat[23][14] = FAR_REAL(405405*nxx*nyy*ny*nzz - 31185*nxx*nyy*ny - 93555*nxx*ny*nzz + 8505*nxx*ny - 135135.0/2.0*nyy*nyy*ny*nzz + (10395.0/2.0)*nyy*nyy*ny - 135135.0/2.0*nyy*ny*nzz*nzz + 51975*nyy*ny*nzz - 6615.0/2.0*nyy*ny + (31185.0/2.0)*ny*nzz*nzz - 14175.0/2.0*ny*nzz + 315*ny);  // power=8
    mat[23][15] = FAR_REAL(315*nxx*nxx*nxx*nx - 1890*nxx*nxx*nx*nyy - 4725*nxx*nxx*nx*nzz + 6615*nxx*nx*nyy*nyy - 4725*nxx*nx*nyy*nyy + 828450*nxx*nx*nyy*nzz + 47250*nxx*nx*nyy*nzz - 66780*nxx*nx*nyy - 62370*nxx*nx*nzz + 5985*nxx*nx + 6615*nx*nyy*nyy*nyy - 2520*nx*nyy*nyy*nyy - 110880*nx*nyy*nyy*nzz + 51975*nx*nyy*nyy*nzz - 630*nx*nyy*nyy + 17640*nx*nyy*nzz*nzz - 75600*nx*nyy*nzz*nzz - 146790*nx*nyy*nzz + 16065*nx*nyy + 5040*nx*nzz*nzz*nzz + 14490*nx*nzz - 1890*nx);  // power=8
    mat[23][16] = FAR_REAL((289575.0/4.0)*nxx*nxx*nxx*ny*nz - 96525.0/8.0*nxx*nxx*nyy*ny*nz - 3571425.0/8.0*nxx*nxx*ny*nzz*nz + 19305*nxx*nxx*ny*nz + (4974860618956799.0/68719476736.0)*nxx*nyy*ny*nzz*nz - 19305.0/2.0*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz + (135135.0/2.0)*nxx*ny*nzz*nz - 22275.0/2.0*nxx*ny*nz - 96525.0/8.0*nyy*ny*nzz*nzz*nz + (1179226220788687.0/366503875925.0)*nyy*ny*nzz*nz - 96525.0/8.0*ny*nzz*nzz*nzz*nz + (5306517993553919.0/549755813888.0)*ny*nzz*nzz*nz - 3673743226306559.0/549755813888.0*ny*nzz*nz + 945*ny*nz);  // power=9
    mat[23][17] = FAR_REAL((289575.0/4.0)*nxx*nyy*nyy*ny*nz - 868725.0/2.0*nxx*nyy*ny*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*nyy*ny*nz + (289575.0/4.0)*nxx*ny*nzz*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*ny*nzz*nz - 8910*nxx*ny*nz - 96525.0/8.0*nyy*nyy*nyy*ny*nz + (482625.0/8.0)*nyy*nyy*ny*nzz*nz - 1.0/502633886983.0*nyy*nyy*ny*nz + (482625.0/8.0)*nyy*ny*nzz*nzz*nz - 7075357324738559.0/137438953472.0*nyy*ny*nzz*nz + 4455*nyy*ny*nz - 96525.0/8.0*ny*nzz*nzz*nzz*nz - 1.0/502633886983.0*ny*nzz*nzz*nz + 4455*ny*nzz*nz - 540*ny*nz);  // power=9
    mat[23][18] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz + 1260*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz + (656145.0/2.0)*nxx*nx*nyy*nyy*nz + (38475.0/2.0)*nxx*nx*nyy*nyy*nz - 877545*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz + (8739055856517121.0/137438953472.0)*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz + 57915*nxx*nx*nzz*nz - 9315*nxx*nx*nz - 11025*nx*nyy*nyy*nyy*nz + 10125*nx*nyy*nyy*nyy*nz + (247365.0/2.0)*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz - 59265*nx*nyy*nyy*nz - 10080*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + 152145*nx*nyy*nzz*nz - 6302263211458561.0/274877906944.0*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 12285*nx*nzz*nz + (5535.0/2.0)*nx*nz);  // power=9
    mat[23][19] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz + 291465*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz - 5306517993553919.0/274877906944.0*nxx*nxx*nx*nz - 62910*nxx*nx*nyy*nyy*nz + (38475.0/2.0)*nxx*nx*nyy*nyy*nz - 1860435.0/2.0*nxx*nx*nyy*nzz*nz - 8998815478579199.0/137438953472.0*nxx*nx*nyy*nzz*nz + (2495547797667843.0/137438953472.0)*nxx*nx*nyy*nz + 2160*nxx*nx*nzz*nzz*nz + (122265.0/2.0)*nxx*nx*nzz*nz - 572531411891499.0/942438538093.0*nxx*nx*nz - 33075.0/2.0*nx*nyy*nyy*nyy*nz + 10125*nx*nyy*nyy*nyy*nz + 113130*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz + (4193262470430719.0/549755813888.0)*nx*nyy*nyy*nz - 15120*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (5188234594222081.0/34359738368.0)*nx*nyy*nzz*nz - 6085796859740161.0/274877906944.0*nx*nyy*nz - 3240*nx*nzz*nzz*nzz*nz - 24975.0/2.0*nx*nzz*nz + 2025*nx*nz);  // power=9
    mat[23][20] = FAR_REAL(210*nxx*nxx*nx*nyy*ny + 289575*nxx*nxx*nx*ny*nzz - 5306517993553919.0/274877906944.0*nxx*nxx*nx*ny + 420*nxx*nx*nyy*nyy*ny - 112065.0/2.0*nxx*nx*nyy*ny*nzz + (1768839331187857.0/549755813889.0)*nxx*nx*nyy*ny - 347490*nxx*nx*ny*nzz*nzz - 87615.0/2.0*nxx*nx*ny*nzz + (4786998749429759.0/549755813888.0)*nxx*nx*ny + 210*nx*nyy*nyy*nyy*ny - 4271602673909759.0/549755813888.0*nx*nyy*nyy*ny*nzz + (8419510289694719.0/274877906944.0)*nx*nyy*ny*nzz*nzz + (5985.0/2.0)*nx*nyy*ny*nzz - 825.0/2.0*nx*nyy*ny + 38610*nx*ny*nzz*nzz*nzz + (3265549534494719.0/137438953472.0)*nx*ny*nzz*nzz - 142496706959843.0/703687441777.0*nx*ny*nzz - 1485.0/2.0*nx*ny);  // power=9
    mat[23][21] = FAR_REAL(289575*nxx*nyy*nyy*nzz - 5306517993553919.0/274877906944.0*nxx*nyy*nyy - 289575*nxx*nyy*nzz*nzz + 4455*nxx*nyy + (5306517993553919.0/274877906944.0)*nxx*nzz*nzz - 4455*nxx*nzz - 96525.0/2.0*nyy*nyy*nyy*nzz + (1768839331187857.0/549755813889.0)*nyy*nyy*nyy + (7959776990330879.0/274877906944.0)*nyy*nyy*nzz - 2227499999997772.0/999999999999.0*nyy*nyy + (96525.0/2.0)*nyy*nzz*nzz*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + 270*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 270*nzz);  // power=9
    mat[23][22] = FAR_REAL((868725.0/2.0)*nxx*nxx*nyy*ny*nz - 868725.0/2.0*nxx*nxx*ny*nzz*nz - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*ny*nz - 868725.0/2.0*nxx*nyy*ny*nzz*nz - 5306517993553921.0/137438953472.0*nxx*nyy*ny*nz + (868725.0/4.0)*nxx*ny*nzz*nzz*nz + 115830*nxx*ny*nzz*nz - 8910*nxx*ny*nz + (4974860618956799.0/68719476736.0)*nyy*nyy*ny*nzz*nz - 19305.0/2.0*nyy*nyy*ny*nz + (96525.0/2.0)*nyy*ny*nzz*nzz*nz - 7075357324738559.0/137438953472.0*nyy*ny*nzz*nz + (22275.0/2.0)*nyy*ny*nz - 96525.0/4.0*ny*nzz*nzz*nzz*nz + (2653258996776959.0/274877906944.0)*ny*nzz*nzz*nz - 1224581075437747.0/549755813889.0*ny*nzz*nz - 540*ny*nz);  // power=9
    mat[23][23] = FAR_REAL(868725*nxx*nxx*nyy*nzz - 57915*nxx*nxx*nyy - 57915*nxx*nxx*nzz + 4455*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nzz + (19305.0/2.0)*nxx*nyy*nyy - 868725.0/2.0*nxx*nyy*nzz*nzz - 115830*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy + (3979888495165439.0/137438953472.0)*nxx*nzz*nzz + (22275.0/2.0)*nxx*nzz - 1620*nxx + (96525.0/2.0)*nyy*nyy*nzz*nzz - 5306517993553919.0/549755813888.0*nyy*nyy*nzz + (96525.0/2.0)*nyy*nzz*nzz*nzz - 5306517993553921.0/137438953472.0*nyy*nzz*nzz + (31185.0/2.0)*nyy*nzz - 945*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 1215*nzz + 105);  // power=9
    mat[23][24] = FAR_REAL(-420*nxx*nxx*nx*nyy*ny + (11685.0/2.0)*nxx*nx*nyy*nyy*ny + 908565*nxx*nx*nyy*ny*nzz - 63990*nxx*nx*nyy*ny - 270270*nxx*nx*ny*nzz*nzz - 4592179032883201.0/68719476736.0*nxx*nx*ny*nzz + (20655.0/2.0)*nxx*nx*ny + (12525.0/2.0)*nx*nyy*nyy*nyy*ny - 98265*nx*nyy*nyy*ny*nzz - 3105*nx*nyy*nyy*ny - 36960*nx*nyy*ny*nzz*nzz - 6478528669286401.0/68719476736.0*nx*nyy*ny*nzz + (6799105028259841.0/549755813888.0)*nx*nyy*ny + (135135.0/2.0)*nx*ny*nzz*nzz*nzz + (718420897588922.0/967570232443.0)*nx*ny*nzz*nzz + (4063382659399681.0/274877906944.0)*nx*ny*nzz - 5535.0/2.0*nx*ny);  // power=9
    mat[23][25] = FAR_REAL(-8505.0/2.0*nxx*nxx*nx*ny*nz + (843885.0/2.0)*nxx*nx*nyy*ny*nz + (14175.0/2.0)*nxx*nx*ny*nzz*nz - 93555*nxx*nx*ny*nz - 93555.0/2.0*nx*nyy*nyy*ny*nz - 70875.0/2.0*nx*nyy*ny*nzz*nz - 51030*nx*nyy*ny*nz + 11340*nx*ny*nzz*nzz*nz + (36855.0/2.0)*nx*ny*nz);  // power=8
    mat[23][26] = FAR_REAL(-8505.0/2.0*nxx*nxx*nx*ny*nz + (14175.0/2.0)*nxx*nx*nyy*ny*nz + (824985.0/2.0)*nxx*nx*ny*nzz*nz - 93555*nxx*nx*ny*nz + 11340*nx*nyy*nyy*ny*nz - 89775.0/2.0*nx*nyy*ny*nzz*nz - 112455.0/2.0*nx*ny*nzz*nzz*nz - 41580*nx*ny*nzz*nz + (36855.0/2.0)*nx*ny*nz);  // power=8
    mat[23][27] = FAR_REAL(405405*nxx*nyy*nzz*nz - 93555*nxx*nyy*nz - 31185*nxx*nzz*nz + 8505*nxx*nz - 135135.0/2.0*nyy*nyy*nzz*nz + (31185.0/2.0)*nyy*nyy*nz - 135135.0/2.0*nyy*nzz*nzz*nz + 51975*nyy*nzz*nz - 14175.0/2.0*nyy*nz + (10395.0/2.0)*nzz*nzz*nz - 6615.0/2.0*nzz*nz + 315*nz);  // power=8

    // Row 24 (target order 4)
    mat[24][0] = FAR_REAL(-525*nxx*nx*nz + 3150*nx*nyy*nz - 525*nx*nzz*nz);  // power=5
    mat[24][1] = FAR_REAL(-4725*nxx*nxx*nz + 28350*nxx*nyy*nz - 4725*nxx*nzz*nz + 1575*nxx*nz - 3150*nyy*nz + 525*nzz*nz);  // power=6
    mat[24][2] = FAR_REAL(-11025*nxx*nx*ny*nz + 22050*nx*nyy*ny*nz - 11025*nx*ny*nzz*nz);  // power=6
    mat[24][3] = FAR_REAL(450*nxx*nxx*nx - 2250*nxx*nx*nyy - 6975*nxx*nx*nzz + 525*nxx*nx - 2700*nx*nyy*nyy + (6308447964364801.0/274877906944.0)*nx*nyy*nzz - 7425*nx*nzz*nzz + 4725*nx*nzz - 450*nx);  // power=6
    mat[24][4] = FAR_REAL(-12425.0/2.0*nxx*nxx*nx*nz + 42700*nxx*nx*nyy*nz - 2450*nxx*nx*nzz*nz + 2625*nxx*nx*nz - 11725*nx*nyy*nyy*nz - 15925.0/2.0*nx*nyy*nzz*nz - 5775.0/2.0*nx*nyy*nz + (7525.0/2.0)*nx*nzz*nzz*nz - 3675.0/2.0*nx*nzz*nz - 175.0/2.0*nx*nz);  // power=7
    mat[24][5] = FAR_REAL(-4025*nxx*nxx*nx*nz + 20300*nxx*nx*nyy*nz + 4025*nxx*nx*nzz*nz + 525*nxx*nx*nz + 24325*nx*nyy*nyy*nz - 56525.0/2.0*nx*nyy*nzz*nz - 22575.0/2.0*nx*nyy*nz + 8050*nx*nzz*nzz*nz - 12075.0/2.0*nx*nzz*nz + (3325.0/2.0)*nx*nz);  // power=7
    mat[24][6] = FAR_REAL(-17325*nxx*nxx*ny*nz + 103950*nxx*nyy*ny*nz - 17325*nxx*ny*nzz*nz - 14175*nxx*ny*nz - 9450*nyy*ny*nz + 1575*ny*nzz*nz + 2100*ny*nz);  // power=7
    mat[24][7] = FAR_REAL(-17325*nxx*nxx*nzz + 1575*nxx*nxx + 103950*nxx*nyy*nzz - 9450*nxx*nyy - 17325*nxx*nzz*nzz + 9450*nxx*nzz - 1575*nxx - 9450*nyy*nzz + 1575*nzz*nzz - 1575*nzz + 1050);  // power=7
    mat[24][8] = FAR_REAL(3150*nxx*nxx*nx*ny - 3150*nxx*nx*nyy*ny - 39375*nxx*nx*ny*nzz + 1575*nxx*nx*ny - 6300*nx*nyy*nyy*ny + 72450*nx*nyy*ny*nzz - 42525*nx*ny*nzz*nzz + 14175*nx*ny*nzz - 1050*nx*ny);  // power=7
    mat[24][9] = FAR_REAL(-45045.0/2.0*nxx*nxx*nxx*nz + 135135*nxx*nxx*nyy*nz - 45045.0/2.0*nxx*nxx*nzz*nz + 20790*nxx*nxx*nz - 62370*nxx*nyy*nz + 10395*nxx*nzz*nz - 8505.0/2.0*nxx*nz + 2835*nyy*nz - 945.0/2.0*nzz*nz + 105*nz);  // power=8
    mat[24][10] = FAR_REAL(-2835.0/2.0*nxx*nxx*nx*ny*nz - 39060*nxx*nx*nyy*ny*nz + (4725.0/2.0)*nxx*nx*ny*nzz*nz + (10395.0/2.0)*nxx*nx*ny*nz + 120015*nx*nyy*nyy*ny*nz - 67725.0/2.0*nx*nyy*ny*nzz*nz - 159705.0/2.0*nx*nyy*ny*nz + 3780*nx*ny*nzz*nzz*nz + (25515.0/2.0)*nx*ny*nz);  // power=8
    mat[24][11] = FAR_REAL(-315*nxx*nxx*nxx*nx + 1890*nxx*nxx*nx*nyy + 4725*nxx*nxx*nx*nzz + 4725*nxx*nx*nyy*nyy - 47250*nxx*nx*nyy*nzz - 45045.0/2.0*nxx*nx*nzz*nzz + 10395*nxx*nx*nzz - 945.0/2.0*nxx*nx + 2520*nx*nyy*nyy*nyy - 51975*nx*nyy*nyy*nzz + 75600*nx*nyy*nzz*nzz - 45045.0/2.0*nx*nzz*nzz*nzz - 5040*nx*nzz*nzz*nzz + 31185*nx*nzz*nzz - 19845.0/2.0*nx*nzz + 420*nx);  // power=8
    mat[24][12] = FAR_REAL(-135135.0/2.0*nxx*nxx*nx*ny*nz + 405405*nxx*nx*nyy*ny*nz - 135135.0/2.0*nxx*nx*ny*nzz*nz - 51975.0/2.0*nxx*nx*ny*nz - 93555*nx*nyy*ny*nz + (31185.0/2.0)*nx*ny*nzz*nz + 14175*nx*ny*nz);  // power=8
    mat[24][13] = FAR_REAL(-135135.0/2.0*nxx*nxx*nx*nzz + (10395.0/2.0)*nxx*nxx*nx + 405405*nxx*nx*nyy*nzz - 31185*nxx*nx*nyy - 135135.0/2.0*nxx*nx*nzz*nzz + 51975*nxx*nx*nzz - 6615.0/2.0*nxx*nx - 93555*nx*nyy*nzz + 8505*nx*nyy + (31185.0/2.0)*nx*nzz*nzz - 14175.0/2.0*nx*nzz + 315*nx);  // power=8
    mat[24][14] = FAR_REAL((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nx*nyy - 4725.0/2.0*nxx*nxx*nx*nzz - 19845*nxx*nx*nyy*nyy - 4725.0/2.0*nxx*nx*nyy*nyy - 240975.0/2.0*nxx*nx*nyy*nzz + 23625*nxx*nx*nyy*nzz + (36855.0/2.0)*nxx*nx*nyy + (10395.0/2.0)*nxx*nx*nzz - 2835.0/2.0*nxx*nx - 19845*nx*nyy*nyy*nyy - 1260*nx*nyy*nyy*nyy + 332640*nx*nyy*nyy*nzz + (51975.0/2.0)*nx*nyy*nyy*nzz + 1890*nx*nyy*nyy - 52920*nx*nyy*nzz*nzz - 37800*nx*nyy*nzz*nzz - 210735.0/2.0*nx*nyy*nzz + (2835.0/2.0)*nx*nyy + 2520*nx*nzz*nzz*nzz + (12285.0/2.0)*nx*nzz + (315.0/2.0)*nx);  // power=8
    mat[24][15] = FAR_REAL(-135135*nxx*nxx*ny*nzz + 10395*nxx*nxx*ny + 810810*nxx*nyy*ny*nzz - 62370*nxx*nyy*ny - 135135*nxx*ny*nzz*nzz - 62370*nxx*ny*nzz + 8505*nxx*ny - 62370*nyy*ny*nzz + 5670*nyy*ny + 10395*ny*nzz*nzz + 8505*ny*nzz - 1260*ny);  // power=8
    mat[24][16] = FAR_REAL(-96525.0/8.0*nxx*nxx*nxx*nx*nz - 2025.0/4.0*nxx*nxx*nxx*nx*nz + (289575.0/4.0)*nxx*nxx*nx*nyy*nz + 4050*nxx*nxx*nx*nyy*nz + (482625.0/8.0)*nxx*nxx*nx*nzz*nz + (8775.0/4.0)*nxx*nxx*nx*nzz*nz - 1.0/502633886983.0*nxx*nxx*nx*nz + (38475.0/4.0)*nxx*nx*nyy*nyy*nz - 868725.0/2.0*nxx*nx*nyy*nzz*nz - 65475.0/2.0*nxx*nx*nyy*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*nx*nyy*nz + (482625.0/8.0)*nxx*nx*nzz*nzz*nz + 1080*nxx*nx*nzz*nzz*nz - 7075357324738559.0/137438953472.0*nxx*nx*nzz*nz + 4455*nxx*nx*nz + (2783138807813062.0/549755813889.0)*nx*nyy*nyy*nyy*nz - 139725.0/4.0*nx*nyy*nyy*nzz*nz + 30780*nx*nyy*nzz*nzz*nz + (173745.0/2.0)*nx*nyy*nzz*nz - 57915.0/4.0*nx*nyy*nz - 96525.0/8.0*nx*nzz*nzz*nzz*nz - 1620*nx*nzz*nzz*nzz*nz + (2653258996776959.0/549755813888.0)*nx*nzz*nzz*nz + (1485.0/2.0)*nx*nzz*nz - 135.0/4.0*nx*nz);  // power=9
    mat[24][17] = FAR_REAL(-2025.0/2.0*nxx*nxx*nxx*nx*nz - 11655.0/2.0*nxx*nxx*nx*nyy*nz + 8100*nxx*nxx*nx*nyy*nz + (8775.0/2.0)*nxx*nxx*nx*nzz*nz - 157185.0/4.0*nxx*nx*nyy*nyy*nz + (38475.0/2.0)*nxx*nx*nyy*nyy*nz + (7778099873710079.0/68719476736.0)*nxx*nx*nyy*nzz*nz - 65475*nxx*nx*nyy*nzz*nz - 143505.0/4.0*nxx*nx*nyy*nz - 96525.0/8.0*nxx*nx*nzz*nzz*nz + 2160*nxx*nx*nzz*nzz*nz + (1179226220788687.0/366503875925.0)*nxx*nx*nzz*nz + (14985.0/8.0)*nxx*nx*nz + (407925.0/8.0)*nx*nyy*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nx*nyy*nyy*nyy*nz - 1684845.0/4.0*nx*nyy*nyy*nzz*nz - 139725.0/2.0*nx*nyy*nyy*nzz*nz + (69255.0/8.0)*nx*nyy*nyy*nz + 46620*nx*nyy*nzz*nzz*nz + (8460741975736319.0/137438953472.0)*nx*nyy*nzz*nzz*nz + (5870103602135041.0/68719476736.0)*nx*nyy*nzz*nz + (93195.0/8.0)*nx*nyy*nz - 96525.0/8.0*nx*nzz*nzz*nzz*nz - 3240*nx*nzz*nzz*nzz*nz + (5306517993553919.0/274877906944.0)*nx*nzz*nzz*nz - 6725918785535999.0/549755813888.0*nx*nzz*nz + (67470031704628.0/799644820203.0)*nx*nz);  // power=9
    mat[24][18] = FAR_REAL(-96525.0/2.0*nxx*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*nxx*ny*nzz*nz - 19305*nxx*nxx*ny*nz + 289575*nxx*nyy*nyy*ny*nz - 1833975.0/2.0*nxx*nyy*ny*nzz*nz + (1326629498388483.0/137438953472.0)*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz + (3979888495165441.0/137438953472.0)*nxx*ny*nzz*nz - 3.0/753950830475.0*nxx*ny*nz - 5306517993553919.0/274877906944.0*nyy*nyy*ny*nz + (122265.0/2.0)*nyy*ny*nzz*nz - 1.0/670178515978.0*nyy*ny*nz - 19305.0/2.0*ny*nzz*nzz*nz - 4455*ny*nzz*nz + 405*ny*nz);  // power=9
    mat[24][19] = FAR_REAL(-96525.0/2.0*nxx*nxx*nxx*ny*nz + 289575*nxx*nxx*nyy*ny*nz + 96525*nxx*nxx*ny*nzz*nz - 3979888495165439.0/137438953472.0*nxx*nxx*ny*nz - 868725*nxx*nyy*ny*nzz*nz + (3979888495165441.0/68719476736.0)*nxx*nyy*ny*nz + (4974860618956799.0/34359738368.0)*nxx*ny*nzz*nzz*nz + (5306517993553923.0/137438953472.0)*nxx*ny*nzz*nz - 1224581075435521.0/274877906944.0*nxx*ny*nz + 57915*nyy*ny*nzz*nz - 8910*nyy*ny*nz - 19305.0/2.0*ny*nzz*nzz*nz - 4455*ny*nzz*nz + 1350*ny*nz);  // power=9
    mat[24][20] = FAR_REAL(-96525.0/2.0*nxx*nxx*nxx*nzz + (1768839331187857.0/549755813889.0)*nxx*nxx*nxx + 289575*nxx*nxx*nyy*nzz - 5306517993553919.0/274877906944.0*nxx*nxx*nyy + (7959776990330879.0/274877906944.0)*nxx*nxx*nzz - 2227499999997772.0/999999999999.0*nxx*nxx - 289575*nxx*nyy*nzz*nzz + 4455*nxx*nyy + (96525.0/2.0)*nxx*nzz*nzz*nzz - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + 270*nxx + (5306517993553919.0/274877906944.0)*nyy*nzz*nzz - 4455*nyy*nzz - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 270*nzz);  // power=9
    mat[24][21] = FAR_REAL(-1470*nxx*nxx*nx*nyy*ny - 4481884272721919.0/274877906944.0*nxx*nx*nyy*nyy*ny - 84945.0/2.0*nxx*nx*nyy*ny*nzz + (8448372469923839.0/549755813888.0)*nxx*nx*nyy*ny + (7959776990330879.0/68719476736.0)*nxx*nx*ny*nzz*nzz - 81675.0/2.0*nxx*nx*ny*nzz - 1215.0/2.0*nxx*nx*ny - 14835*nx*nyy*nyy*nyy*ny + (4844723109887999.0/17179869184.0)*nx*nyy*nyy*ny*nzz + 6210*nx*nyy*nyy*ny - 4518477394083839.0/17179869184.0*nx*nyy*ny*nzz*nzz - 152055.0/2.0*nx*nyy*ny*nzz - 4755.0/2.0*nx*nyy*ny + (7959776990330879.0/68719476736.0)*nx*ny*nzz*nzz*nzz - 6122905377177599.0/68719476736.0*nx*ny*nzz*nzz + (70065.0/2.0)*nx*ny*nzz - 675.0/2.0*nx*ny);  // power=9
    mat[24][22] = FAR_REAL(-1518749999998481.0/999999999999.0*nxx*nxx*nxx*nx*nz - 5364500052049919.0/68719476736.0*nxx*nxx*nx*nyy*nz + (6679533138739199.0/549755813888.0)*nxx*nxx*nx*nyy*nz + (4974860618956799.0/68719476736.0)*nxx*nxx*nx*nzz*nz + (26325.0/4.0)*nxx*nxx*nx*nzz*nz - 19305.0/2.0*nxx*nxx*nx*nz + 478305*nxx*nx*nyy*nyy*nz + (3965972801126399.0/137438953472.0)*nxx*nx*nyy*nyy*nz - 789345.0/2.0*nxx*nx*nyy*nzz*nz - 196425.0/2.0*nxx*nx*nyy*nzz*nz - 8813272891392001.0/137438953472.0*nxx*nx*nyy*nz + (96525.0/2.0)*nxx*nx*nzz*nzz*nz + 3240*nxx*nx*nzz*nzz*nz - 7075357324738559.0/137438953472.0*nxx*nx*nzz*nz + 12960*nxx*nx*nz + (99225.0/2.0)*nx*nyy*nyy*nyy*nz + (4174708211711999.0/274877906944.0)*nx*nyy*nyy*nyy*nz - 339390*nx*nyy*nyy*nzz*nz - 419175.0/4.0*nx*nyy*nyy*nzz*nz - 161595.0/2.0*nx*nyy*nyy*nz + 45360*nx*nyy*nzz*nzz*nz + 92340*nx*nyy*nzz*nzz*nz + (6324682940743681.0/34359738368.0)*nx*nyy*nzz*nz + (6095073989106912.0/824633720833.0)*nx*nyy*nz - 96525.0/4.0*nx*nzz*nzz*nzz*nz - 4859999999999999.0/1000000000000.0*nx*nzz*nzz*nzz*nz + (3316573745971199.0/137438953472.0)*nx*nzz*nzz*nz - 6345556481802239.0/549755813888.0*nx*nzz*nz - 3375.0/4.0*nx*nz);  // power=9
    mat[24][23] = FAR_REAL(-1260*nxx*nxx*nx*nyy*ny - 4974860618956799.0/34359738368.0*nxx*nxx*nx*ny*nzz + (19305.0/2.0)*nxx*nxx*nx*ny - 2520*nxx*nx*nyy*nyy*ny + 915345*nxx*nx*nyy*ny*nzz - 57915*nxx*nx*nyy*ny - 38610*nxx*nx*ny*nzz*nzz - 8980261219860481.0/137438953472.0*nxx*nx*ny*nzz + 5670*nxx*nx*ny - 1260*nx*nyy*nyy*nyy*ny + (6407404010864639.0/137438953472.0)*nx*nyy*nyy*ny*nzz - 6314632717271039.0/34359738368.0*nx*nyy*ny*nzz*nzz - 133785*nx*nyy*ny*nzz + 11385*nx*nyy*ny + (7296462241136639.0/68719476736.0)*nx*ny*nzz*nzz*nzz - 7653631721471999.0/137438953472.0*nx*ny*nzz*nzz + (8906044184985601.0/274877906944.0)*nx*ny*nzz - 2565*nx*ny);  // power=9
    mat[24][24] = FAR_REAL(-4974860618956799.0/34359738368.0*nxx*nxx*nyy*nzz + (19305.0/2.0)*nxx*nxx*nyy + (96525.0/2.0)*nxx*nxx*nzz*nzz - 5306517993553919.0/549755813888.0*nxx*nxx*nzz + 868725*nxx*nyy*nyy*nzz - 57915*nxx*nyy*nyy - 868725.0/2.0*nxx*nyy*nzz*nzz - 115830*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy + (96525.0/2.0)*nxx*nzz*nzz*nzz - 5306517993553921.0/137438953472.0*nxx*nzz*nzz + (31185.0/2.0)*nxx*nzz - 945*nxx - 57915*nyy*nyy*nzz + 4455*nyy*nyy + (3979888495165439.0/137438953472.0)*nyy*nzz*nzz + (22275.0/2.0)*nyy*nzz - 1620*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 1215*nzz + 105);  // power=9
    mat[24][25] = FAR_REAL(-135135.0/2.0*nxx*nxx*nyy*nz + (10395.0/2.0)*nxx*nxx*nz + 405405*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 280665.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + (8505.0/2.0)*nxx*nz - 31185*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + 14175*nyy*nz - 945.0/2.0*nzz*nz - 630*nz);  // power=8
    mat[24][26] = FAR_REAL(-135135.0/2.0*nxx*nxx*nzz*nz + (31185.0/2.0)*nxx*nxx*nz + 405405*nxx*nyy*nzz*nz - 93555*nxx*nyy*nz - 135135.0/2.0*nxx*nzz*nzz*nz + 51975*nxx*nzz*nz - 14175.0/2.0*nxx*nz - 31185*nyy*nzz*nz + 8505*nyy*nz + (10395.0/2.0)*nzz*nzz*nz - 6615.0/2.0*nzz*nz + 315*nz);  // power=8
    mat[24][27] = FAR_REAL(25515*nxx*nxx*nx*ny*nz - 42525*nxx*nx*nyy*ny*nz - 220185.0/2.0*nxx*nx*ny*nzz*nz + (31185.0/2.0)*nxx*nx*ny*nz - 68040*nx*nyy*nyy*ny*nz + 269325*nx*nyy*ny*nzz*nz - 271215.0/2.0*nx*ny*nzz*nzz*nz + (135135.0/2.0)*nx*ny*nzz*nz - 11340*nx*ny*nz);  // power=8

    // Row 25 (target order 3)
    mat[25][0] = FAR_REAL(15*nxx*nx - 60*nx*nyy + 15*nx*nzz);  // power=4
    mat[25][1] = FAR_REAL(105*nxx*nxx - 420*nxx*nyy + 105*nxx*nzz - 45*nxx + 60*nyy - 15*nzz);  // power=5
    mat[25][2] = FAR_REAL(225*nxx*nx*ny - 300*nx*nyy*ny + 225*nx*ny*nzz);  // power=5
    mat[25][3] = FAR_REAL(75*nxx*nx*nz - 450*nx*nyy*nz + 75*nx*nzz*nz);  // power=5
    mat[25][4] = FAR_REAL((215.0/2.0)*nxx*nxx*nx - 485*nxx*nx*nyy + 40*nxx*nx*nzz - 70*nxx*nx + 195*nx*nyy*nyy + (255.0/2.0)*nx*nyy*nzz + (35.0/2.0)*nx*nyy - 135.0/2.0*nx*nzz*nzz + (35.0/2.0)*nx*nzz + (25.0/2.0)*nx);  // power=6
    mat[25][5] = FAR_REAL(60*nxx*nxx*nx - 195*nxx*nx*nyy - 90*nxx*nx*nzz - 35*nxx*nx - 255*nx*nyy*nyy + (765.0/2.0)*nx*nyy*nzz + (385.0/2.0)*nx*nyy - 150*nx*nzz*nzz + (175.0/2.0)*nx*nzz - 25.0/2.0*nx);  // power=6
    mat[25][6] = FAR_REAL(315*nxx*nxx*ny - 1260*nxx*nyy*ny + 315*nxx*ny*nzz + 175*nxx*ny + 140*nyy*ny - 35*ny*nzz - 40*ny);  // power=6
    mat[25][7] = FAR_REAL(315*nxx*nxx*nz - 1260*nxx*nyy*nz + 315*nxx*nzz*nz - 175*nxx*nz + 140*nyy*nz - 35*nzz*nz + 10*nz);  // power=6
    mat[25][8] = FAR_REAL(525*nxx*nx*ny*nz - 1050*nx*nyy*ny*nz + 525*nx*ny*nzz*nz);  // power=6
    mat[25][9] = FAR_REAL((693.0/2.0)*nxx*nxx*nxx - 1386*nxx*nxx*nyy + (693.0/2.0)*nxx*nxx*nzz - 378*nxx*nxx + 756*nxx*nyy - 189*nxx*nzz + (189.0/2.0)*nxx - 42*nyy + (21.0/2.0)*nzz - 3);  // power=7
    mat[25][10] = FAR_REAL((21.0/2.0)*nxx*nxx*nx*ny + 336*nxx*nx*nyy*ny - 147.0/2.0*nxx*nx*ny*nzz - 189.0/2.0*nxx*nx*ny - 1407*nx*nyy*nyy*ny + (483.0/2.0)*nx*nyy*ny*nzz + (2331.0/2.0)*nx*nyy*ny - 84*nx*ny*nzz*nzz - 357.0/2.0*nx*ny);  // power=7
    mat[25][11] = FAR_REAL(-42*nxx*nxx*nx*nz + 294*nxx*nx*nyy*nz + (777.0/2.0)*nxx*nx*nzz*nz - 189.0/2.0*nxx*nx*nz + 336*nx*nyy*nyy*nz - 966*nx*nyy*nzz*nz + (861.0/2.0)*nx*nzz*nzz*nz - 819.0/2.0*nx*nzz*nz + 84*nx*nz);  // power=7
    mat[25][12] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*ny - 4158*nxx*nx*nyy*ny + (2079.0/2.0)*nxx*nx*ny*nzz + (189.0/2.0)*nxx*nx*ny + 1134*nx*nyy*ny - 567.0/2.0*nx*ny*nzz - 189*nx*ny);  // power=7
    mat[25][13] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*nz - 4158*nxx*nx*nyy*nz + (2079.0/2.0)*nxx*nx*nzz*nz - 1701.0/2.0*nxx*nx*nz + 1134*nx*nyy*nz - 567.0/2.0*nx*nzz*nz + 126*nx*nz);  // power=7
    mat[25][14] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*nz + 1323*nxx*nx*nyy*nz - 63.0/2.0*nxx*nx*nzz*nz - 189.0/2.0*nxx*nx*nz - 3906*nx*nyy*nyy*nz + (2457.0/2.0)*nx*nyy*nzz*nz + (2961.0/2.0)*nx*nyy*nz - 63*nx*nzz*nzz*nz - 189.0/2.0*nx*nz);  // power=7
    mat[25][15] = FAR_REAL(2079*nxx*nxx*ny*nz - 8316*nxx*nyy*ny*nz + 2079*nxx*ny*nzz*nz + 567*nxx*ny*nz + 756*nyy*ny*nz - 189*ny*nzz*nz - 126*ny*nz);  // power=7
    mat[25][16] = FAR_REAL((1287.0/8.0)*nxx*nxx*nxx*nx + (3.0/2.0)*nxx*nxx*nxx*nx - 1287.0/2.0*nxx*nxx*nx*nyy - 9*nxx*nxx*nx*nyy - 6435.0/8.0*nxx*nxx*nx*nzz - 45.0/2.0*nxx*nxx*nx*nzz - 297.0/2.0*nxx*nxx*nx - 45.0/2.0*nxx*nx*nyy*nyy + 3861*nxx*nx*nyy*nzz + 225*nxx*nx*nyy*nzz + 198*nxx*nx*nyy - 6435.0/8.0*nxx*nx*nzz*nzz + 693*nxx*nx*nzz + 27*nxx*nx - 12*nx*nyy*nyy*nyy + (495.0/2.0)*nx*nyy*nyy*nzz - 360*nx*nyy*nzz*nzz - 891*nx*nyy*nzz + (27.0/2.0)*nx*nyy + (1287.0/8.0)*nx*nzz*nzz*nzz + 24*nx*nzz*nzz*nzz - 54*nx*nzz - 3.0/2.0*nx);  // power=8
    mat[25][17] = FAR_REAL((15.0/4.0)*nxx*nxx*nxx*nx - 45.0/2.0*nxx*nxx*nx*nyy - 225.0/4.0*nxx*nxx*nx*nzz - 36*nxx*nx*nyy*nyy - 225.0/4.0*nxx*nx*nyy*nyy - 5961.0/4.0*nxx*nx*nyy*nzz + (1125.0/2.0)*nxx*nx*nyy*nzz + (525.0/4.0)*nxx*nx*nyy + (1287.0/8.0)*nxx*nx*nzz*nzz - 75.0/8.0*nxx*nx - 6723.0/8.0*nx*nyy*nyy*nyy - 30*nx*nyy*nyy*nyy + 3300*nx*nyy*nyy*nzz + (2475.0/4.0)*nx*nyy*nyy*nzz + (5793.0/8.0)*nx*nyy*nyy - 525*nx*nyy*nzz*nzz - 900*nx*nyy*nzz*nzz - 3909.0/4.0*nx*nyy*nzz - 1395.0/8.0*nx*nyy + (1287.0/8.0)*nx*nzz*nzz*nzz + 60*nx*nzz*nzz*nzz - 891.0/4.0*nx*nzz*nzz + (1005.0/8.0)*nx*nzz + (45.0/8.0)*nx);  // power=8
    mat[25][18] = FAR_REAL((1287.0/2.0)*nxx*nxx*nyy*ny - 3861.0/2.0*nxx*nxx*ny*nzz - 2574*nxx*nyy*nyy*ny + (16731.0/2.0)*nxx*nyy*ny*nzz + (2079.0/2.0)*nxx*nyy*ny - 3861.0/2.0*nxx*ny*nzz*nzz - 297.0/2.0*nxx*ny*nzz - 135*nxx*ny + 198*nyy*nyy*ny - 1287.0/2.0*nyy*ny*nzz - 108*nyy*ny + (297.0/2.0)*ny*nzz*nzz + 54*ny*nzz + 15*ny);  // power=8
    mat[25][19] = FAR_REAL((1287.0/2.0)*nxx*nxx*nxx*ny - 2574*nxx*nxx*nyy*ny - 1287*nxx*nxx*ny*nzz - 99.0/2.0*nxx*nxx*ny + 7722*nxx*nyy*ny*nzz + 594*nxx*nyy*ny - 3861.0/2.0*nxx*ny*nzz*nzz - 297*nxx*ny*nzz - 54*nxx*ny - 594*nyy*ny*nzz + (297.0/2.0)*ny*nzz*nzz + 54*ny*nzz);  // power=8
    mat[25][20] = FAR_REAL((1287.0/2.0)*nxx*nxx*nxx*nz - 2574*nxx*nxx*nyy*nz - 1089.0/2.0*nxx*nxx*nz + 2574*nxx*nyy*nzz*nz + 594*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + 297*nxx*nzz*nz + 81*nxx*nz - 198*nyy*nzz*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz);  // power=8
    mat[25][21] = FAR_REAL((405.0/2.0)*nxx*nxx*nx*ny*nz + 666*nxx*nx*nyy*ny*nz - 981*nxx*nx*ny*nzz*nz - 2754*nx*nyy*nyy*ny*nz + (2497499999997502.0/999999999999.0)*nx*nyy*ny*nzz*nz + (2943.0/2.0)*nx*nyy*ny*nz - 2367.0/2.0*nx*ny*nzz*nzz*nz + (1287.0/2.0)*nx*ny*nzz*nz - 675.0/2.0*nx*ny*nz);  // power=8
    mat[25][22] = FAR_REAL((21.0/4.0)*nxx*nxx*nxx*nx + (3861.0/4.0)*nxx*nxx*nx*nyy - 63.0/2.0*nxx*nxx*nx*nyy - 3861.0/4.0*nxx*nxx*nx*nzz - 315.0/4.0*nxx*nxx*nx*nzz - 4050*nxx*nx*nyy*nyy - 315.0/4.0*nxx*nx*nyy*nyy + 3357*nxx*nx*nyy*nzz + (1575.0/2.0)*nxx*nx*nyy*nzz + (1737.0/2.0)*nxx*nx*nyy - 1287.0/2.0*nxx*nx*nzz*nzz + (1485.0/2.0)*nxx*nx*nzz - 153.0/2.0*nxx*nx - 189*nx*nyy*nyy*nyy - 42*nx*nyy*nyy*nyy + 3168*nx*nyy*nyy*nzz + (3465.0/4.0)*nx*nyy*nyy*nzz + 909*nx*nyy*nyy - 504*nx*nyy*nzz*nzz - 1260*nx*nyy*nzz*nzz - 2043*nx*nyy*nzz - 1107.0/4.0*nx*nyy + (1287.0/4.0)*nx*nzz*nzz*nzz + 84*nx*nzz*nzz*nzz - 891.0/4.0*nx*nzz*nzz + 72*nx*nzz + (75.0/4.0)*nx);  // power=8
    mat[25][23] = FAR_REAL((4185.0/2.0)*nxx*nxx*nx*ny*nz - 7992*nxx*nx*nyy*ny*nz + 1017*nxx*nx*ny*nzz*nz - 432*nx*nyy*nyy*ny*nz + 1710*nx*nyy*ny*nzz*nz + 1782*nx*nyy*ny*nz - 2151.0/2.0*nx*ny*nzz*nzz*nz + 198*nx*ny*nzz*nz - 270*nx*ny*nz);  // power=8
    mat[25][24] = FAR_REAL((3861.0/2.0)*nxx*nxx*nyy*nz - 1287.0/2.0*nxx*nxx*nzz*nz - 7722*nxx*nyy*nyy*nz + (2476375063663000.0/549755813889.0)*nxx*nyy*nzz*nz + (3267.0/2.0)*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + (891.0/2.0)*nxx*nzz*nz - 135*nxx*nz + 594*nyy*nyy*nz - 693.0/2.0*nyy*nzz*nz - 189*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz + 15*nz);  // power=8
    mat[25][25] = FAR_REAL((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nyy + (2079.0/2.0)*nxx*nyy*nzz + (3213.0/2.0)*nxx*nyy - 189.0/2.0*nxx*nzz - 63*nxx + 378*nyy*nyy - 189.0/2.0*nyy*nzz - 441.0/2.0*nyy + 45.0/2.0);  // power=7
    mat[25][26] = FAR_REAL((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nzz + 378*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz - 756*nxx*nzz + (189.0/2.0)*nxx + 378*nyy*nzz - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
    mat[25][27] = FAR_REAL(-126*nxx*nxx*nx*ny + 126*nxx*nx*nyy*ny + (3843.0/2.0)*nxx*nx*ny*nzz - 189.0/2.0*nxx*nx*ny + 252*nx*nyy*nyy*ny - 2898*nx*nyy*ny*nzz + (4095.0/2.0)*nx*ny*nzz*nzz - 1701.0/2.0*nx*ny*nzz + 63*nx*ny);  // power=7

    // Row 26 (target order 3)
    mat[26][0] = FAR_REAL(15*nxx*nx + 15*nx*nyy - 60*nx*nzz);  // power=4
    mat[26][1] = FAR_REAL(105*nxx*nxx + 105*nxx*nyy - 420*nxx*nzz - 45*nxx - 15*nyy + 60*nzz);  // power=5
    mat[26][2] = FAR_REAL(75*nxx*nx*ny + 75*nx*nyy*ny - 450*nx*ny*nzz);  // power=5
    mat[26][3] = FAR_REAL(225*nxx*nx*nz + 225*nx*nyy*nz - 300*nx*nzz*nz);  // power=5
    mat[26][4] = FAR_REAL((215.0/2.0)*nxx*nxx*nx + 40*nxx*nx*nyy - 485*nxx*nx*nzz - 70*nxx*nx - 135.0/2.0*nx*nyy*nyy + (255.0/2.0)*nx*nyy*nzz + (35.0/2.0)*nx*nyy + 195*nx*nzz*nzz + (35.0/2.0)*nx*nzz + (25.0/2.0)*nx);  // power=6
    mat[26][5] = FAR_REAL((95.0/2.0)*nxx*nxx*nx + 130*nxx*nx*nyy - 290*nxx*nx*nzz - 35*nxx*nx + (165.0/2.0)*nx*nyy*nyy - 255*nx*nyy*nzz - 70*nx*nyy + 450*nx*nzz*nzz - 175*nx*nzz + 25*nx);  // power=6
    mat[26][6] = FAR_REAL(315*nxx*nxx*ny + 315*nxx*nyy*ny - 1260*nxx*ny*nzz - 175*nxx*ny - 35*nyy*ny + 140*ny*nzz + 10*ny);  // power=6
    mat[26][7] = FAR_REAL(315*nxx*nxx*nz + 315*nxx*nyy*nz - 1260*nxx*nzz*nz + 175*nxx*nz - 35*nyy*nz + 140*nzz*nz - 40*nz);  // power=6
    mat[26][8] = FAR_REAL(525*nxx*nx*ny*nz + 525*nx*nyy*ny*nz - 1050*nx*ny*nzz*nz);  // power=6
    mat[26][9] = FAR_REAL((693.0/2.0)*nxx*nxx*nxx + (693.0/2.0)*nxx*nxx*nyy - 1386*nxx*nxx*nzz - 378*nxx*nxx - 189*nxx*nyy + 756*nxx*nzz + (189.0/2.0)*nxx + (21.0/2.0)*nyy - 42*nzz - 3);  // power=7
    mat[26][10] = FAR_REAL(-42*nxx*nxx*nx*ny + (777.0/2.0)*nxx*nx*nyy*ny + 294*nxx*nx*ny*nzz - 189.0/2.0*nxx*nx*ny + (861.0/2.0)*nx*nyy*nyy*ny - 966*nx*nyy*ny*nzz - 819.0/2.0*nx*nyy*ny + 336*nx*ny*nzz*nzz + 84*nx*ny);  // power=7
    mat[26][11] = FAR_REAL((21.0/2.0)*nxx*nxx*nx*nz - 147.0/2.0*nxx*nx*nyy*nz + 336*nxx*nx*nzz*nz - 189.0/2.0*nxx*nx*nz - 84*nx*nyy*nyy*nz + (483.0/2.0)*nx*nyy*nzz*nz - 1407*nx*nzz*nzz*nz + (2331.0/2.0)*nx*nzz*nz - 357.0/2.0*nx*nz);  // power=7
    mat[26][12] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*ny + (2079.0/2.0)*nxx*nx*nyy*ny - 4158*nxx*nx*ny*nzz - 1701.0/2.0*nxx*nx*ny - 567.0/2.0*nx*nyy*ny + 1134*nx*ny*nzz + 126*nx*ny);  // power=7
    mat[26][13] = FAR_REAL((2079.0/2.0)*nxx*nxx*nx*nz + (2079.0/2.0)*nxx*nx*nyy*nz - 4158*nxx*nx*nzz*nz + (189.0/2.0)*nxx*nx*nz - 567.0/2.0*nx*nyy*nz + 1134*nx*nzz*nz - 189*nx*nz);  // power=7
    mat[26][14] = FAR_REAL(-126*nxx*nxx*nx*nz + (3591.0/2.0)*nxx*nx*nyy*nz + 126*nxx*nx*nzz*nz - 189.0/2.0*nxx*nx*nz + (3843.0/2.0)*nx*nyy*nyy*nz - 3024*nx*nyy*nzz*nz - 1449.0/2.0*nx*nyy*nz + 252*nx*nzz*nzz*nz + 63*nx*nz);  // power=7
    mat[26][15] = FAR_REAL(2079*nxx*nxx*ny*nz + 2079*nxx*nyy*ny*nz - 8316*nxx*ny*nzz*nz + 567*nxx*ny*nz - 189*nyy*ny*nz + 756*ny*nzz*nz - 126*ny*nz);  // power=7
    mat[26][16] = FAR_REAL((1287.0/8.0)*nxx*nxx*nxx*nx - 3.0/8.0*nxx*nxx*nxx*nx + (1287.0/8.0)*nxx*nxx*nx*nyy + (9.0/4.0)*nxx*nxx*nx*nyy - 6435.0/4.0*nxx*nxx*nx*nzz + (45.0/8.0)*nxx*nxx*nx*nzz - 297.0/2.0*nxx*nxx*nx + (45.0/8.0)*nxx*nx*nyy*nyy - 3861.0/4.0*nxx*nx*nyy*nzz - 225.0/4.0*nxx*nx*nyy*nzz - 99.0/2.0*nxx*nx*nyy + (32175.0/8.0)*nxx*nx*nzz*nzz - 1089.0/2.0*nxx*nx*nzz + (189.0/2.0)*nxx*nx + 3*nx*nyy*nyy*nyy - 495.0/8.0*nx*nyy*nyy*nzz + 90*nx*nyy*nzz*nzz + (891.0/4.0)*nx*nyy*nzz - 27.0/8.0*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz - 6*nx*nzz*nzz*nzz - 185624999999072.0/999999999995.0*nx*nzz*nzz + (459.0/4.0)*nx*nzz - 117.0/8.0*nx);  // power=8
    mat[26][17] = FAR_REAL(-75.0/8.0*nxx*nxx*nxx*nx + (225.0/4.0)*nxx*nxx*nx*nyy + (1125.0/8.0)*nxx*nxx*nx*nzz + (1917.0/8.0)*nxx*nx*nyy*nyy + (1125.0/8.0)*nxx*nx*nyy*nyy - 3021.0/4.0*nxx*nx*nyy*nzz - 5625.0/4.0*nxx*nx*nyy*nzz - 105.0/2.0*nxx*nx*nyy + (1287.0/8.0)*nxx*nx*nzz*nzz + (15.0/4.0)*nxx*nx + (1917.0/8.0)*nx*nyy*nyy*nyy + 75*nx*nyy*nyy*nyy - 1320*nx*nyy*nyy*nzz - 12375.0/8.0*nx*nyy*nyy*nzz - 921.0/4.0*nx*nyy*nyy + 210*nx*nyy*nzz*nzz + 2250*nx*nyy*nzz*nzz + (2811.0/4.0)*nx*nyy*nzz + (315.0/8.0)*nx*nyy - 1287.0/2.0*nx*nzz*nzz*nzz - 150*nx*nzz*nzz*nzz + (5643.0/8.0)*nx*nzz*nzz - 465.0/2.0*nx*nzz + (45.0/8.0)*nx);  // power=8
    mat[26][18] = FAR_REAL((1287.0/2.0)*nxx*nxx*nyy*ny - 3861.0/2.0*nxx*nxx*ny*nzz + (1287.0/2.0)*nxx*nyy*nyy*ny - 2476375063663000.0/549755813889.0*nxx*nyy*ny*nzz - 891.0/2.0*nxx*nyy*ny + 7722*nxx*ny*nzz*nzz - 3267.0/2.0*nxx*ny*nzz + 135*nxx*ny - 99.0/2.0*nyy*nyy*ny + (693.0/2.0)*nyy*ny*nzz + 27*nyy*ny - 594*ny*nzz*nzz + 189*ny*nzz - 15*ny);  // power=8
    mat[26][19] = FAR_REAL((1287.0/2.0)*nxx*nxx*nxx*ny + (1287.0/2.0)*nxx*nxx*nyy*ny - 2476375063663000.0/549755813889.0*nxx*nxx*ny*nzz - 1089.0/2.0*nxx*nxx*ny - 3861.0/2.0*nxx*nyy*ny*nzz - 297.0/2.0*nxx*nyy*ny + 7722*nxx*ny*nzz*nzz - 2079.0/2.0*nxx*ny*nzz + 216*nxx*ny + (297.0/2.0)*nyy*ny*nzz - 594*ny*nzz*nzz + 189*ny*nzz - 15*ny);  // power=8
    mat[26][20] = FAR_REAL((1287.0/2.0)*nxx*nxx*nxx*nz + (1287.0/2.0)*nxx*nxx*nyy*nz - 3217499999996782.0/999999999999.0*nxx*nxx*nzz*nz - 99.0/2.0*nxx*nxx*nz - 1287.0/2.0*nxx*nyy*nzz*nz - 297.0/2.0*nxx*nyy*nz + 2574*nxx*nzz*nzz*nz - 445499999999554.0/999999999999.0*nxx*nzz*nz + 81*nxx*nz + (99.0/2.0)*nyy*nzz*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
    mat[26][21] = FAR_REAL(-405.0/2.0*nxx*nxx*nx*ny*nz + 891*nxx*nx*nyy*ny*nz - 306*nxx*nx*ny*nzz*nz + (2187.0/2.0)*nx*nyy*nyy*ny*nz - 2227499999997772.0/999999999999.0*nx*nyy*ny*nzz*nz - 1107.0/2.0*nx*nyy*ny*nz + 3114*nx*ny*nzz*nzz*nz - 3663.0/2.0*nx*ny*nzz*nz + (675.0/2.0)*nx*ny*nz);  // power=8
    mat[26][22] = FAR_REAL(-39.0/4.0*nxx*nxx*nxx*nx + (3861.0/4.0)*nxx*nxx*nx*nyy + (117.0/2.0)*nxx*nxx*nx*nyy - 3861.0/4.0*nxx*nxx*nx*nzz + (585.0/4.0)*nxx*nxx*nx*nzz + (2025.0/2.0)*nxx*nx*nyy*nyy + (585.0/4.0)*nxx*nx*nyy*nyy - 11331.0/2.0*nxx*nx*nyy*nzz - 2925.0/2.0*nxx*nx*nyy*nzz - 774*nxx*nx*nyy + (16731.0/4.0)*nxx*nx*nzz*nzz - 1485.0/2.0*nxx*nx*nzz + (279.0/4.0)*nxx*nx + (189.0/4.0)*nx*nyy*nyy*nyy + 78*nx*nyy*nyy*nyy - 792*nx*nyy*nyy*nzz - 6435.0/4.0*nx*nyy*nyy*nzz - 909.0/4.0*nx*nyy*nyy + 126*nx*nyy*nzz*nzz + 2340*nx*nyy*nzz*nzz + (3249.0/2.0)*nx*nyy*nzz + (189.0/2.0)*nx*nyy - 1287*nx*nzz*nzz*nzz - 156*nx*nzz*nzz*nzz + (519749999999480.0/999999999999.0)*nx*nzz*nzz - 477.0/4.0*nx*nzz - 15.0/2.0*nx);  // power=8
    mat[26][23] = FAR_REAL(1890*nxx*nxx*nx*ny*nz + 1998*nxx*nx*nyy*ny*nz - 8298*nxx*nx*ny*nzz*nz + 108*nx*nyy*nyy*ny*nz - 855.0/2.0*nx*nyy*ny*nzz*nz - 891.0/2.0*nx*nyy*ny*nz + (2681999999999999.0/1000000000000.0)*nx*ny*nzz*nzz*nz - 49499999999950.0/999999999999.0*nx*ny*nzz*nz + (135.0/2.0)*nx*ny*nz);  // power=8
    mat[26][24] = FAR_REAL((3861.0/2.0)*nxx*nxx*nyy*nz - 1287.0/2.0*nxx*nxx*nzz*nz + (3861.0/2.0)*nxx*nyy*nyy*nz - 16731.0/2.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + 2574*nxx*nzz*nzz*nz - 2079.0/2.0*nxx*nzz*nz + 135*nxx*nz - 297.0/2.0*nyy*nyy*nz + (1287.0/2.0)*nyy*nzz*nz - 54*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
    mat[26][25] = FAR_REAL((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy - 4158*nxx*nyy*nzz - 756*nxx*nyy + 378*nxx*nzz + (189.0/2.0)*nxx - 189.0/2.0*nyy*nyy + 378*nyy*nzz + (189.0/2.0)*nyy - 45);  // power=7
    mat[26][26] = FAR_REAL((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 4158*nxx*nzz*nzz + (3213.0/2.0)*nxx*nzz - 63*nxx - 189.0/2.0*nyy*nzz + 378*nzz*nzz - 441.0/2.0*nzz + 45.0/2.0);  // power=7
    mat[26][27] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*ny - 63.0/2.0*nxx*nx*nyy*ny + 819*nxx*nx*ny*nzz - 189.0/2.0*nxx*nx*ny - 63*nx*nyy*nyy*ny + (1449.0/2.0)*nx*nyy*ny*nzz - 4410*nx*ny*nzz*nzz + (3969.0/2.0)*nx*ny*nzz - 189.0/2.0*nx*ny);  // power=7

    // Row 27 (target order 3)
    mat[27][0] = FAR_REAL(15*nxx*ny + 15*nyy*ny - 60*ny*nzz);  // power=4
    mat[27][1] = FAR_REAL(75*nxx*nx*ny + 75*nx*nyy*ny - 450*nx*ny*nzz);  // power=5
    mat[27][2] = FAR_REAL(105*nxx*nyy - 15*nxx + 105*nyy*nyy - 420*nyy*nzz - 45*nyy + 60*nzz);  // power=5
    mat[27][3] = FAR_REAL(225*nxx*ny*nz + 225*nyy*ny*nz - 300*ny*nzz*nz);  // power=5
    mat[27][4] = FAR_REAL(105*nxx*nxx*ny + (105.0/2.0)*nxx*nyy*ny - 945.0/2.0*nxx*ny*nzz - 35*nxx*ny - 105.0/2.0*nyy*nyy*ny + (315.0/2.0)*nyy*ny*nzz + 35*nyy*ny + 210*ny*nzz*nzz - 140*ny*nzz + 5*ny);  // power=6
    mat[27][5] = FAR_REAL((105.0/2.0)*nxx*nxx*ny + 105*nxx*nyy*ny - 315*nxx*ny*nzz - 35*nxx*ny + (105.0/2.0)*nyy*nyy*ny - 315*nyy*ny*nzz - 35*nyy*ny + 420*ny*nzz*nzz - 140*ny*nzz + 20*ny);  // power=6
    mat[27][6] = FAR_REAL(-20*nxx*nxx*nx + 415*nxx*nx*nyy + 100*nxx*nx*nzz - 35*nxx*nx + 435*nx*nyy*nyy - 1020*nx*nyy*nzz - 315*nx*nyy + 120*nx*nzz*nzz + 30*nx);  // power=6
    mat[27][7] = FAR_REAL(525*nxx*nx*ny*nz + 525*nx*nyy*ny*nz - 1050*nx*ny*nzz*nz);  // power=6
    mat[27][8] = FAR_REAL(315*nxx*nyy*nz - 35*nxx*nz + 315*nyy*nyy*nz - 1260*nyy*nzz*nz + 175*nyy*nz + 140*nzz*nz - 40*nz);  // power=6
    mat[27][9] = FAR_REAL((693.0/2.0)*nxx*nxx*nx*ny + (693.0/2.0)*nxx*nx*nyy*ny - 1386*nxx*nx*ny*nzz - 567.0/2.0*nxx*nx*ny - 189.0/2.0*nx*nyy*ny + 378*nx*ny*nzz + 42*nx*ny);  // power=7
    mat[27][10] = FAR_REAL((693.0/2.0)*nxx*nyy*nyy - 189*nxx*nyy + (21.0/2.0)*nxx + (693.0/2.0)*nyy*nyy*nyy - 1386*nyy*nyy*nzz - 378*nyy*nyy + 756*nyy*nzz + (189.0/2.0)*nyy - 42*nzz - 3);  // power=7
    mat[27][11] = FAR_REAL((693.0/2.0)*nxx*ny*nzz*nz - 189.0/2.0*nxx*ny*nz + (693.0/2.0)*nyy*ny*nzz*nz - 189.0/2.0*nyy*ny*nz - 1386*ny*nzz*nzz*nz + 1134*ny*nzz*nz - 168*ny*nz);  // power=7
    mat[27][12] = FAR_REAL((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy - 4158*nxx*nyy*nzz - 756*nxx*nyy + 378*nxx*nzz + (189.0/2.0)*nxx - 189.0/2.0*nyy*nyy + 378*nyy*nzz + (189.0/2.0)*nyy - 45);  // power=7
    mat[27][13] = FAR_REAL((2079.0/2.0)*nxx*nxx*ny*nz + (2079.0/2.0)*nxx*nyy*ny*nz - 4158*nxx*ny*nzz*nz + (567.0/2.0)*nxx*ny*nz - 189.0/2.0*nyy*ny*nz + 378*ny*nzz*nz - 63*ny*nz);  // power=7
    mat[27][14] = FAR_REAL((2079.0/2.0)*nxx*nyy*ny*nz - 567.0/2.0*nxx*ny*nz + (2079.0/2.0)*nyy*nyy*ny*nz - 4158*nyy*ny*nzz*nz + (189.0/2.0)*nyy*ny*nz + 1134*ny*nzz*nz - 189*ny*nz);  // power=7
    mat[27][15] = FAR_REAL(-252*nxx*nxx*nx*nz + 3591*nxx*nx*nyy*nz + 252*nxx*nx*nzz*nz - 189*nxx*nx*nz + 3843*nx*nyy*nyy*nz - 6048*nx*nyy*nzz*nz - 1449*nx*nyy*nz + 504*nx*nzz*nzz*nz + 126*nx*nz);  // power=7
    mat[27][16] = FAR_REAL((1287.0/8.0)*nxx*nxx*nxx*ny + (1287.0/8.0)*nxx*nxx*nyy*ny - 6435.0/4.0*nxx*nxx*ny*nzz - 99*nxx*nxx*ny - 3861.0/4.0*nxx*nyy*ny*nzz + (32175.0/8.0)*nxx*ny*nzz*nzz - 891*nxx*ny*nzz + (135.0/2.0)*nxx*ny + (1287.0/8.0)*nyy*ny*nzz*nzz - 1287.0/2.0*ny*nzz*nzz*nzz + 396*ny*nzz*nzz - 135.0/2.0*ny*nzz);  // power=8
    mat[27][17] = FAR_REAL((1287.0/8.0)*nxx*nyy*nyy*ny - 3861.0/4.0*nxx*nyy*ny*nzz - 99.0/2.0*nxx*nyy*ny + (1287.0/8.0)*nxx*ny*nzz*nzz + (297.0/2.0)*nxx*ny*nzz + (1287.0/8.0)*nyy*nyy*nyy*ny - 6435.0/4.0*nyy*nyy*ny*nzz - 297.0/2.0*nyy*nyy*ny + (32175.0/8.0)*nyy*ny*nzz*nzz - 1089.0/2.0*nyy*ny*nzz + (189.0/2.0)*nyy*ny - 1287.0/2.0*ny*nzz*nzz*nzz - 198*ny*nzz*nzz + (243.0/2.0)*ny*nzz - 15*ny);  // power=8
    mat[27][18] = FAR_REAL(-18*nxx*nxx*nxx*nx + 108*nxx*nxx*nx*nyy + 270*nxx*nxx*nx*nzz + 864*nxx*nx*nyy*nyy + 270*nxx*nx*nyy*nyy - 2685.0/2.0*nxx*nx*nyy*nzz - 2700*nxx*nx*nyy*nzz - 591.0/2.0*nxx*nx*nyy + (297.0/2.0)*nxx*nx*nzz + (21.0/2.0)*nxx*nx + 864*nx*nyy*nyy*nyy + 144*nx*nyy*nyy*nyy - 3696*nx*nyy*nyy*nzz - 2970*nx*nyy*nyy*nzz - 912*nx*nyy*nyy + 588*nx*nyy*nzz*nzz + 4320*nx*nyy*nzz*nzz + (3579.0/2.0)*nx*nyy*nzz + (423.0/2.0)*nx*nyy - 288*nx*nzz*nzz*nzz - 249.0/2.0*nx*nzz - 15.0/2.0*nx);  // power=8
    mat[27][19] = FAR_REAL(-18*nxx*nxx*nxx*nx + (1287.0/2.0)*nxx*nxx*nx*nyy + 108*nxx*nxx*nx*nyy + 270*nxx*nxx*nx*nzz - 99.0/2.0*nxx*nxx*nx + 738*nxx*nx*nyy*nyy + 270*nxx*nx*nyy*nyy - 2337836598562972.0/549755813889.0*nxx*nx*nyy*nzz - 2700*nxx*nx*nyy*nzz - 1017.0/2.0*nxx*nx*nyy + (693.0/2.0)*nxx*nx*nzz + (63.0/2.0)*nxx*nx + (189.0/2.0)*nx*nyy*nyy*nyy + 144*nx*nyy*nyy*nyy - 1584*nx*nyy*nyy*nzz - 2970*nx*nyy*nyy*nzz - 315.0/2.0*nx*nyy*nyy + 252*nx*nyy*nzz*nzz + 4320*nx*nyy*nzz*nzz + (3231.0/2.0)*nx*nyy*nzz + (81.0/2.0)*nx*nyy - 288*nx*nzz*nzz*nzz - 261.0/2.0*nx*nzz - 3.0/2.0*nx);  // power=8
    mat[27][20] = FAR_REAL(603*nxx*nxx*nx*ny*nz + 711*nxx*nx*nyy*ny*nz - 3150*nxx*nx*ny*nzz*nz + 99*nxx*nx*ny*nz + 108*nx*nyy*nyy*ny*nz - 855.0/2.0*nx*nyy*ny*nzz*nz - 297.0/2.0*nx*nyy*ny*nz + (2681999999999999.0/1000000000000.0)*nx*ny*nzz*nzz*nz - 2475.0/2.0*nx*ny*nzz*nz + (351.0/2.0)*nx*ny*nz);  // power=8
    mat[27][21] = FAR_REAL((1287.0/2.0)*nxx*nyy*nyy*nz - 1287.0/2.0*nxx*nyy*nzz*nz - 297.0/2.0*nxx*nyy*nz + (99.0/2.0)*nxx*nzz*nz + (1287.0/2.0)*nyy*nyy*nyy*nz - 3217499999996782.0/999999999999.0*nyy*nyy*nzz*nz - 99.0/2.0*nyy*nyy*nz + 2574*nyy*nzz*nzz*nz - 445499999999554.0/999999999999.0*nyy*nzz*nz + 81*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
    mat[27][22] = FAR_REAL((3861.0/4.0)*nxx*nxx*nyy*ny - 3861.0/4.0*nxx*nxx*ny*nzz - 297.0/2.0*nxx*nxx*ny + (3861.0/4.0)*nxx*nyy*nyy*ny - 11583.0/2.0*nxx*nyy*ny*nzz - 1485.0/2.0*nxx*nyy*ny + (16731.0/4.0)*nxx*ny*nzz*nzz - 297.0/2.0*nxx*ny*nzz + (297.0/2.0)*nxx*ny - 3861.0/4.0*nyy*nyy*ny*nzz + (16731.0/4.0)*nyy*ny*nzz*nzz - 1485.0/2.0*nyy*ny*nzz + (135.0/2.0)*nyy*ny - 1287*ny*nzz*nzz*nzz + 198*ny*nzz*nzz + 54*ny*nzz - 15*ny);  // power=8
    mat[27][23] = FAR_REAL((3861.0/2.0)*nxx*nxx*nyy*nz - 297.0/2.0*nxx*nxx*nz + (3861.0/2.0)*nxx*nyy*nyy*nz - 16731.0/2.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + (1287.0/2.0)*nxx*nzz*nz - 54*nxx*nz - 1287.0/2.0*nyy*nyy*nzz*nz + 2574*nyy*nzz*nzz*nz - 2079.0/2.0*nyy*nzz*nz + 135*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
    mat[27][24] = FAR_REAL(-1053.0/2.0*nxx*nxx*nx*ny*nz + 2538*nxx*nx*nyy*ny*nz + 234*nxx*nx*ny*nzz*nz - 297*nxx*nx*ny*nz + (3064499999996936.0/999999999999.0)*nx*nyy*nyy*ny*nz - 11655.0/2.0*nx*nyy*ny*nzz*nz - 3321.0/2.0*nx*nyy*ny*nz + 3978*nx*ny*nzz*nzz*nz - 3663.0/2.0*nx*ny*nzz*nz + (1107.0/2.0)*nx*ny*nz);  // power=8
    mat[27][25] = FAR_REAL(-126*nxx*nxx*nx*ny + (2331.0/2.0)*nxx*nx*nyy*ny + 882*nxx*nx*ny*nzz - 567.0/2.0*nxx*nx*ny + (2583.0/2.0)*nx*nyy*nyy*ny - 2898*nx*nyy*ny*nzz - 2457.0/2.0*nx*nyy*ny + 1008*nx*ny*nzz*nzz + 252*nx*ny);  // power=7
    mat[27][26] = FAR_REAL((63.0/2.0)*nxx*nxx*nx*ny - 63.0/2.0*nxx*nx*nyy*ny + 819*nxx*nx*ny*nzz - 189.0/2.0*nxx*nx*ny - 63*nx*nyy*nyy*ny + (1449.0/2.0)*nx*nyy*ny*nzz - 4410*nx*ny*nzz*nzz + (3969.0/2.0)*nx*ny*nzz - 189.0/2.0*nx*ny);  // power=7
    mat[27][27] = FAR_REAL((2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 189.0/2.0*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy - 4158*nyy*nzz*nzz + (3213.0/2.0)*nyy*nzz - 63*nyy + 378*nzz*nzz - 441.0/2.0*nzz + 45.0/2.0);  // power=7

}

// ===========================================================================
// Per-row M2L matrix computation (parallel variant of compute_m2l_matrix_device)
// ===========================================================================
// Each of 28 active threads computes its own row, eliminating thread-0
// serialization and the 8KB shared memory allocation.
// Generated by gen_per_row_m2l.py from the same source expressions.
// ===========================================================================

// Per-row M2L matrix computation for parallel warp execution.
// Each thread computes one row (28 active entries) of the M2L matrix.
// Replaces the serial thread-0 compute_m2l_matrix_device() call,
// distributing work across all 28 active threads in the warp.
// Auto-generated from bvh8_fmm/derivation/generate_m2l_matrix.py
// via gen_per_row_m2l.py — same 784 non-zero entries.
template<typename T>
__host__ __device__ __forceinline__
void compute_m2l_row_device(int row, T nx, T ny, T nz,
                             T out[32]) {
    // Zero all entries in the output row
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = T(0.0);
    }

    // Precompute direction products
    T nxx = nx * nx, nyy = ny * ny, nzz = nz * nz;
    T nxy = nx * ny, nxz = nx * nz, nyz = ny * nz;

    // Dispatch to row-specific computation
    switch (row) {
    case 0: {
        out[0] = T(1);  // power=1
        out[1] = T(nx);  // power=2
        out[2] = T(ny);  // power=2
        out[3] = T(nz);  // power=2
        out[4] = T((1.0/3.0)*nxx - 1.0/6.0*nyy - 1.0/6.0*nzz);  // power=3
        out[5] = T((1.0/6.0)*nxx + (1.0/6.0)*nyy - 1.0/3.0*nzz);  // power=3
        out[6] = T(nxy);  // power=3
        out[7] = T(nxz);  // power=3
        out[8] = T(nyz);  // power=3
        out[9] = T((1.0/5.0)*nxx*nx - 3.0/10.0*nxy*ny - 3.0/10.0*nxz*nz);  // power=4
        out[10] = T(-3.0/10.0*nxx*ny + (1.0/5.0)*nyy*ny - 3.0/10.0*nyz*nz);  // power=4
        out[11] = T(-3.0/10.0*nxx*nz - 3.0/10.0*nyy*nz + (1.0/5.0)*nzz*nz);  // power=4
        out[12] = T((6.0/5.0)*nxx*ny - 3.0/10.0*nyy*ny - 3.0/10.0*nyz*nz);  // power=4
        out[13] = T((6.0/5.0)*nxx*nz - 3.0/10.0*nyy*nz - 3.0/10.0*nzz*nz);  // power=4
        out[14] = T(-3.0/10.0*nxx*nz + (6.0/5.0)*nyy*nz - 3.0/10.0*nzz*nz);  // power=4
        out[15] = T(3*nxy*nz);  // power=4
        out[16] = T((1.0/8.0)*nxx*nxx - 3.0/4.0*nxx*nzz + (1.0/8.0)*nzz*nzz);  // power=5
        out[17] = T((1.0/8.0)*nyy*nyy - 3.0/4.0*nyy*nzz + (1.0/8.0)*nzz*nzz);  // power=5
        out[18] = T((1.0/2.0)*nxy*ny*ny - 3.0/2.0*nxy*nzz);  // power=5
        out[19] = T((1.0/2.0)*nxx*nxy - 3.0/2.0*nxy*nzz);  // power=5
        out[20] = T((1.0/2.0)*nxx*nxz - 1.0/2.0*nxz*nz*nz);  // power=5
        out[21] = T((1.0/2.0)*nyy*nyz - 1.0/2.0*nyz*nz*nz);  // power=5
        out[22] = T((3.0/4.0)*nxx*nyy - 3.0/4.0*nxx*nzz - 3.0/4.0*nyy*nzz + (1.0/4.0)*nzz*nzz);  // power=5
        out[23] = T((3.0/2.0)*nxx*nyz - 1.0/2.0*nyz*nz*nz);  // power=5
        out[24] = T((3.0/2.0)*nxy*nyz - 1.0/2.0*nxz*nz*nz);  // power=5
        out[25] = T(-3.0/10.0*nxx*nx + (6.0/5.0)*nxy*ny - 3.0/10.0*nxz*nz);  // power=4
        out[26] = T(-3.0/10.0*nxx*nx - 3.0/10.0*nxy*ny + (6.0/5.0)*nxz*nz);  // power=4
        out[27] = T(-3.0/10.0*nxx*ny - 3.0/10.0*nyy*ny + (6.0/5.0)*nyz*nz);  // power=4
    } break;

    case 1: {
        out[0] = T(-nx);  // power=2
        out[1] = T(-2*nxx + nyy + nzz);  // power=3
        out[2] = T(-3*nxy);  // power=3
        out[3] = T(-3*nxz);  // power=3
        out[4] = T(-nxx*nx + (3.0/2.0)*nxy*ny + (3.0/2.0)*nxz*nz);  // power=4
        out[5] = T(-1.0/2.0*nxx*nx - 1.0/2.0*nxy*ny + 2*nxz*nz);  // power=4
        out[6] = T(-4*nxx*ny + nyy*ny + nyz*nz);  // power=4
        out[7] = T(-4*nxx*nz + nyy*nz + nzz*nz);  // power=4
        out[8] = T(-5*nxy*nz);  // power=4
        out[9] = T(-7.0/2.0*nxx*nxx + 3*nxx - 3.0/10.0);  // power=5
        out[10] = T((3.0/2.0)*nxx*nxy - 2*nxy*ny*ny + (3.0/2.0)*nxy*nzz);  // power=5
        out[11] = T((3.0/2.0)*nxx*nxz + (3.0/2.0)*nxy*nyz - 2*nxz*nz*nz);  // power=5
        out[12] = T(-6*nxx*nxy + (9.0/2.0)*nxy*ny*ny + (9.0/2.0)*nxy*nzz);  // power=5
        out[13] = T(-6*nxx*nxz + (9.0/2.0)*nxy*nyz + (9.0/2.0)*nxz*nz*nz);  // power=5
        out[14] = T((3.0/2.0)*nxx*nxz - 9*nxy*nyz + (3.0/2.0)*nxz*nz*nz);  // power=5
        out[15] = T(-18*nxx*nyz + 3*nyy*nyz + 3*nyz*nz*nz);  // power=5
        out[16] = T(-9.0/8.0*nxx*nxx*nx + (27.0/4.0)*nxx*nxz*nz + (1.0/2.0)*nxx*nx - 9.0/8.0*nxz*nz*nzz - 3.0/2.0*nxz*nz);  // power=6
        out[17] = T((3.0/28.0)*nxx*nxx*nx - 15.0/28.0*nxx*nxy*ny - 15.0/28.0*nxx*nxz*nz - 99.0/56.0*nxy*ny*nyy + (153.0/28.0)*nxy*nyz*nz + (3.0/4.0)*nxy*ny - 99.0/56.0*nxz*nz*nzz + (3.0/4.0)*nxz*nz - 3.0/28.0*nx);  // power=6
        out[18] = T(-9.0/2.0*nxx*nyy*ny + (27.0/2.0)*nxx*nyz*nz + (1.0/2.0)*nyy*ny - 3.0/2.0*nyz*nz);  // power=6
        out[19] = T(-9.0/2.0*nxx*nxx*ny + (27.0/2.0)*nxx*nyz*nz + (3.0/2.0)*nxx*ny - 3.0/2.0*nyz*nz);  // power=6
        out[20] = T(-9.0/2.0*nxx*nxx*nz + (9.0/2.0)*nxx*nzz*nz + (3.0/2.0)*nxx*nz - 1.0/2.0*nzz*nz);  // power=6
        out[21] = T(-9.0/2.0*nxy*ny*nyz + (9.0/2.0)*nxy*nzz*nz);  // power=6
        out[22] = T((3.0/28.0)*nxx*nxx*nx - 51.0/7.0*nxx*nxy*ny + (87.0/14.0)*nxx*nxz*nz - 9.0/14.0*nxy*ny*nyy + (153.0/28.0)*nxy*nyz*nz + (9.0/4.0)*nxy*ny - 81.0/28.0*nxz*nz*nzz - 3.0/4.0*nxz*nz - 3.0/28.0*nx);  // power=6
        out[23] = T(-21.0/2.0*nxx*nxy*nz + 3*nxy*ny*nyz + (15.0/2.0)*nxy*nzz*nz);  // power=6
        out[24] = T(-27.0/2.0*nxx*nyy*nz + (9.0/2.0)*nxx*nzz*nz + (3.0/2.0)*nyy*nz - 1.0/2.0*nzz*nz);  // power=6
        out[25] = T(-21.0/2.0*nxx*nyy + (3.0/2.0)*nxx + (3.0/2.0)*nyy - 3.0/10.0);  // power=5
        out[26] = T(-21.0/2.0*nxx*nzz + (3.0/2.0)*nxx + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
        out[27] = T((3.0/2.0)*nxx*nxy + (3.0/2.0)*nxy*ny*ny - 9*nxy*nzz);  // power=5
    } break;

    case 2: {
        out[0] = T(-ny);  // power=2
        out[1] = T(-3*nxy);  // power=3
        out[2] = T(nxx - 2*nyy + nzz);  // power=3
        out[3] = T(-3*nyz);  // power=3
        out[4] = T(-2*nxx*ny + (1.0/2.0)*nyy*ny + (1.0/2.0)*nyz*nz);  // power=4
        out[5] = T(-1.0/2.0*nxx*ny - 1.0/2.0*nyy*ny + 2*nyz*nz);  // power=4
        out[6] = T(nxx*nx - 4*nxy*ny + nxz*nz);  // power=4
        out[7] = T(-5*nxy*nz);  // power=4
        out[8] = T(nxx*nz - 4*nyy*nz + nzz*nz);  // power=4
        out[9] = T(-2*nxx*nxy + (3.0/2.0)*nxy*ny*ny + (3.0/2.0)*nxy*nzz);  // power=5
        out[10] = T(-7.0/2.0*nyy*nyy + 3*nyy - 3.0/10.0);  // power=5
        out[11] = T((3.0/2.0)*nxx*nyz + (3.0/2.0)*nyy*nyz - 2*nyz*nz*nz);  // power=5
        out[12] = T(-21.0/2.0*nxx*nyy + (3.0/2.0)*nxx + (3.0/2.0)*nyy - 3.0/10.0);  // power=5
        out[13] = T(-9*nxx*nyz + (3.0/2.0)*nyy*nyz + (3.0/2.0)*nyz*nz*nz);  // power=5
        out[14] = T((9.0/2.0)*nxx*nyz - 6*nyy*nyz + (9.0/2.0)*nyz*nz*nz);  // power=5
        out[15] = T(3*nxx*nxz - 18*nxy*nyz + 3*nxz*nz*nz);  // power=5
        out[16] = T(-9.0/8.0*nxx*nxx*ny + (27.0/4.0)*nxx*nyz*nz - 9.0/8.0*nyz*nz*nzz);  // power=6
        out[17] = T(-9.0/8.0*nyy*nyy*ny + (27.0/4.0)*nyy*nyz*nz + (1.0/2.0)*nyy*ny - 9.0/8.0*nyz*nz*nzz - 3.0/2.0*nyz*nz);  // power=6
        out[18] = T((3.0/14.0)*nxx*nxx*nx - 15.0/14.0*nxx*nxy*ny - 15.0/14.0*nxx*nxz*nz - 81.0/14.0*nxy*ny*nyy + (153.0/14.0)*nxy*nyz*nz + 3*nxy*ny - 9.0/7.0*nxz*nz*nzz - 3.0/14.0*nx);  // power=6
        out[19] = T((3.0/14.0)*nxx*nxx*nx - 39.0/7.0*nxx*nxy*ny - 15.0/14.0*nxx*nxz*nz + (1.0/2.0)*nxx*nx - 9.0/7.0*nxy*ny*nyy + (153.0/14.0)*nxy*nyz*nz + (3.0/2.0)*nxy*ny - 9.0/7.0*nxz*nz*nzz - 3.0/14.0*nx);  // power=6
        out[20] = T(-9.0/2.0*nxx*nxy*nz + (9.0/2.0)*nxy*nzz*nz);  // power=6
        out[21] = T(-9.0/2.0*nyy*nyy*nz + (9.0/2.0)*nyy*nzz*nz + (3.0/2.0)*nyy*nz - 1.0/2.0*nzz*nz);  // power=6
        out[22] = T(-27.0/4.0*nxx*nyy*ny + (27.0/4.0)*nxx*nyz*nz + (3.0/2.0)*nxx*ny + (27.0/4.0)*nyy*nyz*nz - 9.0/4.0*nyz*nz*nzz - 3.0/2.0*nyz*nz);  // power=6
        out[23] = T(-27.0/2.0*nxx*nyy*nz + (3.0/2.0)*nxx*nz + (9.0/2.0)*nyy*nzz*nz - 1.0/2.0*nzz*nz);  // power=6
        out[24] = T(3*nxx*nxy*nz - 21.0/2.0*nxy*ny*nyz + (15.0/2.0)*nxy*nzz*nz);  // power=6
        out[25] = T((9.0/2.0)*nxx*nxy - 6*nxy*ny*ny + (9.0/2.0)*nxy*nzz);  // power=5
        out[26] = T((3.0/2.0)*nxx*nxy + (3.0/2.0)*nxy*ny*ny - 9*nxy*nzz);  // power=5
        out[27] = T(-21.0/2.0*nyy*nzz + (3.0/2.0)*nyy + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
    } break;

    case 3: {
        out[0] = T(-nz);  // power=2
        out[1] = T(-3*nxz);  // power=3
        out[2] = T(-3*nyz);  // power=3
        out[3] = T(nxx + nyy - 2*nzz);  // power=3
        out[4] = T(-2*nxx*nz + (1.0/2.0)*nyy*nz + (1.0/2.0)*nzz*nz);  // power=4
        out[5] = T(-3.0/2.0*nxx*nz - 3.0/2.0*nyy*nz + nzz*nz);  // power=4
        out[6] = T(-5*nxy*nz);  // power=4
        out[7] = T(nxx*nx + nxy*ny - 4*nxz*nz);  // power=4
        out[8] = T(nxx*ny + nyy*ny - 4*nyz*nz);  // power=4
        out[9] = T(-2*nxx*nxz + (3.0/2.0)*nxy*nyz + (3.0/2.0)*nxz*nz*nz);  // power=5
        out[10] = T((3.0/2.0)*nxx*nyz - 2*nyy*nyz + (3.0/2.0)*nyz*nz*nz);  // power=5
        out[11] = T(-7.0/2.0*nzz*nzz + 3*nzz - 3.0/10.0);  // power=5
        out[12] = T(-9*nxx*nyz + (3.0/2.0)*nyy*nyz + (3.0/2.0)*nyz*nz*nz);  // power=5
        out[13] = T(-21.0/2.0*nxx*nzz + (3.0/2.0)*nxx + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
        out[14] = T(-21.0/2.0*nyy*nzz + (3.0/2.0)*nyy + (3.0/2.0)*nzz - 3.0/10.0);  // power=5
        out[15] = T(3*nxx*nxy + 3*nxy*ny*ny - 18*nxy*nzz);  // power=5
        out[16] = T(-9.0/8.0*nxx*nxx*nz + (27.0/4.0)*nxx*nzz*nz - 3.0/2.0*nxx*nz - 9.0/8.0*nzz*nzz*nz + (1.0/2.0)*nzz*nz);  // power=6
        out[17] = T(-9.0/8.0*nyy*nyy*nz + (27.0/4.0)*nyy*nzz*nz - 3.0/2.0*nyy*nz - 9.0/8.0*nzz*nzz*nz + (1.0/2.0)*nzz*nz);  // power=6
        out[18] = T(-3*nxx*nxy*nz - 15.0/2.0*nxy*ny*nyz + (21.0/2.0)*nxy*nzz*nz);  // power=6
        out[19] = T(-15.0/2.0*nxx*nxy*nz - 3*nxy*ny*nyz + (21.0/2.0)*nxy*nzz*nz);  // power=6
        out[20] = T(-9.0/2.0*nxx*nxz*nz + (1.0/2.0)*nxx*nx + (9.0/2.0)*nxz*nz*nzz - 3.0/2.0*nxz*nz);  // power=6
        out[21] = T(-9.0/2.0*nyy*nyz*nz + (1.0/2.0)*nyy*ny + (9.0/2.0)*nyz*nz*nzz - 3.0/2.0*nyz*nz);  // power=6
        out[22] = T(-27.0/4.0*nxx*nyy*nz + (27.0/4.0)*nxx*nzz*nz - 3.0/2.0*nxx*nz + (27.0/4.0)*nyy*nzz*nz - 3.0/2.0*nyy*nz - 9.0/4.0*nzz*nzz*nz + nzz*nz);  // power=6
        out[23] = T(-27.0/2.0*nxx*nyz*nz + (3.0/2.0)*nxx*ny + (9.0/2.0)*nyz*nz*nzz - 3.0/2.0*nyz*nz);  // power=6
        out[24] = T(-3.0/14.0*nxx*nxx*nx + (15.0/14.0)*nxx*nxy*ny + (15.0/14.0)*nxx*nxz*nz + (9.0/7.0)*nxy*ny*nyy - 153.0/14.0*nxy*nyz*nz + (81.0/14.0)*nxz*nz*nzz - 3*nxz*nz + (3.0/14.0)*nx);  // power=6
        out[25] = T((3.0/2.0)*nxx*nxz - 9*nxy*nyz + (3.0/2.0)*nxz*nz*nz);  // power=5
        out[26] = T((9.0/2.0)*nxx*nxz + (9.0/2.0)*nxy*nyz - 6*nxz*nz*nz);  // power=5
        out[27] = T((9.0/2.0)*nxx*nyz + (9.0/2.0)*nyy*nyz - 6*nyz*nz*nz);  // power=5
    } break;

    case 4: {
        out[0] = T(9*nxx - 9*nyy);  // power=3
        out[1] = T(27*nxx*nx - 63*nxy*ny - 18*nxz*nz);  // power=4
        out[2] = T(63*nxx*ny - 27*nyy*ny + 18*nyz*nz);  // power=4
        out[3] = T(45*nxx*nz - 45*nyy*nz);  // power=4
        out[4] = T(35*nxx*nxx - 105.0/2.0*nxx*nyy - 35.0/2.0*nxx*nzz - 20*nxx + (35.0/2.0)*nyy*nyy + (35.0/2.0)*nyy*nzz - 10*nyy + 3);  // power=5
        out[5] = T((35.0/2.0)*nxx*nxx - 35*nxx*nzz - 10*nxx - 35.0/2.0*nyy*nyy + 35*nyy*nzz + 10*nyy);  // power=5
        out[6] = T(105*nxx*nxy - 105*nxy*ny*ny);  // power=5
        out[7] = T(75*nxx*nxz - 135*nxy*nyz - 30*nxz*nz*nz);  // power=5
        out[8] = T(135*nxx*nyz - 75*nyy*nyz + 30*nyz*nz*nz);  // power=5
        out[9] = T((189.0/2.0)*nxx*nxx*nx - 189.0/2.0*nxx*nxy*ny - 189.0/2.0*nxx*nx + (63.0/2.0)*nxy*ny + 18*nx);  // power=6
        out[10] = T((189.0/2.0)*nxx*nyy*ny - 63.0/2.0*nxx*ny - 189.0/2.0*nyy*nyy*ny + (189.0/2.0)*nyy*ny - 18*ny);  // power=6
        out[11] = T((189.0/2.0)*nxx*nzz*nz - 63.0/2.0*nxx*nz - 189.0/2.0*nyy*nzz*nz + (63.0/2.0)*nyy*nz);  // power=6
        out[12] = T((567.0/2.0)*nxx*nxx*ny - 567.0/2.0*nxx*nyy*ny - 189.0/2.0*nxx*ny + (63.0/2.0)*nyy*ny);  // power=6
        out[13] = T((567.0/2.0)*nxx*nxx*nz - 567.0/2.0*nxx*nyy*nz - 315.0/2.0*nxx*nz + (63.0/2.0)*nyy*nz + 9*nz);  // power=6
        out[14] = T((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 567.0/2.0*nyy*nyy*nz + (315.0/2.0)*nyy*nz - 9*nz);  // power=6
        out[15] = T(567*nxx*nxy*nz - 567*nxy*ny*nyz);  // power=6
        out[16] = T((297.0/8.0)*nxx*nxx*nxx - 297.0/8.0*nxx*nxx*nyy - 891.0/4.0*nxx*nxx*nzz - 27*nxx*nxx + (891.0/4.0)*nxx*nyy*nzz + (297.0/8.0)*nxx*nzz*nzz + 81*nxx*nzz + (9.0/4.0)*nxx - 297.0/8.0*nyy*nzz*nzz - 9.0/4.0*nyy - 27.0/4.0*nzz + 9.0/4.0);  // power=7
        out[17] = T((297.0/8.0)*nxx*nyy*nyy - 891.0/4.0*nxx*nyy*nzz + (297.0/8.0)*nxx*nzz*nzz + (9.0/4.0)*nxx - 297.0/8.0*nyy*nyy*nyy + (891.0/4.0)*nyy*nyy*nzz + 27*nyy*nyy - 297.0/8.0*nyy*nzz*nzz - 81*nyy*nzz - 9.0/4.0*nyy + (27.0/4.0)*nzz - 9.0/4.0);  // power=7
        out[18] = T((27.0/2.0)*nxx*nxx*nxy + 135*nxx*nxy*ny*ny - 540*nxx*nxy*nzz - 351.0/2.0*nxy*ny*nyy*ny + (621.0/2.0)*nxy*ny*nyz*nz + (189.0/2.0)*nxy*ny*ny - 108*nxy*nzz*nzz + (243.0/2.0)*nxy*nzz - 45.0/2.0*nxy);  // power=7
        out[19] = T(162*nxx*nxx*nxy - 162*nxx*nxy*ny*ny - 540*nxx*nxy*nzz - 54*nxx*nxy - 27*nxy*ny*nyy*ny + (621.0/2.0)*nxy*ny*nyz*nz + (81.0/2.0)*nxy*ny*ny - 108*nxy*nzz*nzz + (243.0/2.0)*nxy*nzz - 9.0/2.0*nxy);  // power=7
        out[20] = T(153*nxx*nxx*nxz - 180*nxx*nxy*nyz - 153*nxx*nxz*nz*nz - 81*nxx*nxz - 36*nxy*ny*nyy*nz + (207.0/2.0)*nxy*nyz*nz*nz + (81.0/2.0)*nxy*nyz - 9*nxz*nz*nzz*nz + (81.0/2.0)*nxz*nz*nz + (9.0/2.0)*nxz);  // power=7
        out[21] = T((297.0/2.0)*nxx*nyy*nyz - 297.0/2.0*nxx*nyz*nz*nz - 297.0/2.0*nyy*nyy*nyz + (297.0/2.0)*nyy*nyz*nz*nz + 81*nyy*nyz - 27*nyz*nz*nz - 9*nyz);  // power=7
        out[22] = T((891.0/4.0)*nxx*nxx*nyy - 891.0/4.0*nxx*nxx*nzz - 891.0/4.0*nxx*nyy*nyy + (297.0/4.0)*nxx*nzz*nzz + 81*nxx*nzz - 9.0/2.0*nxx + (891.0/4.0)*nyy*nyy*nzz - 297.0/4.0*nyy*nzz*nzz - 81*nyy*nzz + (9.0/2.0)*nyy);  // power=7
        out[23] = T((891.0/2.0)*nxx*nxx*nyz - 891.0/2.0*nxx*nyy*nyz - 297.0/2.0*nxx*nyz*nz*nz - 81*nxx*nyz + (297.0/2.0)*nyy*nyz*nz*nz - 27*nyz*nz*nz + 9*nyz);  // power=7
        out[24] = T((9.0/2.0)*nxx*nxx*nxz + 468*nxx*nxy*nyz - 153*nxx*nxz*nz*nz - 855.0/2.0*nxy*ny*nyy*nz + (315.0/2.0)*nxy*nyz*nz*nz + (135.0/2.0)*nxy*nyz - 9*nxz*nz*nzz*nz + (81.0/2.0)*nxz*nz*nz - 27.0/2.0*nxz);  // power=7
        out[25] = T((567.0/2.0)*nxx*nxy*ny - 63.0/2.0*nxx*nx - 567.0/2.0*nxy*ny*nyy + (189.0/2.0)*nxy*ny);  // power=6
        out[26] = T(-9.0/2.0*nxx*nxx*nx + (45.0/2.0)*nxx*nxy*ny + 306*nxx*nxz*nz - 63.0/2.0*nxx*nx + 27*nxy*ny*nyy - 459.0/2.0*nxy*nyz*nz + 27*nxz*nz*nzz - 189.0/2.0*nxz*nz + (27.0/2.0)*nx);  // power=6
        out[27] = T((567.0/2.0)*nxx*nyz*nz - 63.0/2.0*nxx*ny - 567.0/2.0*nyy*nyz*nz + (63.0/2.0)*nyy*ny + 63*nyz*nz - 9*ny);  // power=6
    } break;

    case 5: {
        out[0] = T(9*nyy - 9*nzz);  // power=3
        out[1] = T(45*nxy*ny - 45*nxz*nz);  // power=4
        out[2] = T(-18*nxx*ny + 27*nyy*ny - 63*nyz*nz);  // power=4
        out[3] = T(18*nxx*nz + 63*nyy*nz - 27*nzz*nz);  // power=4
        out[4] = T(35*nxx*nyy - 35*nxx*nzz - 35.0/2.0*nyy*nyy + 10*nyy + (35.0/2.0)*nzz*nzz - 10*nzz);  // power=5
        out[5] = T((35.0/2.0)*nxx*nyy - 35.0/2.0*nxx*nzz + (35.0/2.0)*nyy*nyy - 105.0/2.0*nyy*nzz - 10*nyy + 35*nzz*nzz - 20*nzz + 3);  // power=5
        out[6] = T(-30*nxx*nxy + 75*nxy*ny*ny - 135*nxy*nzz);  // power=5
        out[7] = T(30*nxx*nxz + 135*nxy*nyz - 75*nxz*nz*nz);  // power=5
        out[8] = T(105*nyy*nyz - 105*nyz*nz*nz);  // power=5
        out[9] = T((189.0/2.0)*nxx*nxy*ny - 189.0/2.0*nxx*nxz*nz - 63.0/2.0*nxy*ny + (63.0/2.0)*nxz*nz);  // power=6
        out[10] = T((189.0/2.0)*nyy*nyy*ny - 189.0/2.0*nyy*nyz*nz - 189.0/2.0*nyy*ny + (63.0/2.0)*nyz*nz + 18*ny);  // power=6
        out[11] = T((189.0/2.0)*nyy*nzz*nz - 63.0/2.0*nyy*nz - 189.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 18*nz);  // power=6
        out[12] = T((567.0/2.0)*nxx*nyy*ny - 567.0/2.0*nxx*nyz*nz - 63*nxx*ny - 63.0/2.0*nyy*ny + (63.0/2.0)*nyz*nz + 9*ny);  // power=6
        out[13] = T((567.0/2.0)*nxx*nyy*nz - 567.0/2.0*nxx*nzz*nz + 63*nxx*nz - 63.0/2.0*nyy*nz + (63.0/2.0)*nzz*nz - 9*nz);  // power=6
        out[14] = T((567.0/2.0)*nyy*nyy*nz - 567.0/2.0*nyy*nzz*nz - 189.0/2.0*nyy*nz + (63.0/2.0)*nzz*nz);  // power=6
        out[15] = T(567*nxy*ny*nyz - 567*nxy*nzz*nz);  // power=6
        out[16] = T((297.0/8.0)*nxx*nxx*nyy - 297.0/8.0*nxx*nxx*nzz - 891.0/4.0*nxx*nyy*nzz + (891.0/4.0)*nxx*nzz*nzz - 81*nxx*nzz + (27.0/4.0)*nxx + (297.0/8.0)*nyy*nzz*nzz + (9.0/4.0)*nyy - 297.0/8.0*nzz*nzz*nzz + 27*nzz*nzz - 9.0/4.0*nzz - 9.0/4.0);  // power=7
        out[17] = T((297.0/8.0)*nyy*nyy*nyy - 2079.0/8.0*nyy*nyy*nzz - 27*nyy*nyy + (2079.0/8.0)*nyy*nzz*nzz + 9*nyy - 297.0/8.0*nzz*nzz*nzz + 27*nzz*nzz - 9*nzz);  // power=7
        out[18] = T(-18*nxx*nxx*nxy + 18*nxx*nxy*ny*ny + 126*nxx*nxy*nzz + (369.0/2.0)*nxy*ny*nyy*ny - 414*nxy*ny*nyz*nz - 135*nxy*ny*ny + (1179.0/2.0)*nxy*nzz*nzz - 243*nxy*nzz + 36*nxy);  // power=7
        out[19] = T(-27.0/2.0*nxx*nxx*nxy + 162*nxx*nxy*ny*ny - 54*nxx*nxy*nzz - 27*nxx*nxy + 27*nxy*ny*nyy*ny - 621.0/2.0*nxy*ny*nyz*nz - 81.0/2.0*nxy*ny*ny + (1107.0/2.0)*nxy*nzz*nzz - 405.0/2.0*nxy*nzz + (45.0/2.0)*nxy);  // power=7
        out[20] = T(-9.0/2.0*nxx*nxx*nxz + 180*nxx*nxy*nyz - 144*nxx*nxz*nz*nz + 27*nxx*nxz + 36*nxy*ny*nyy*nz - 207.0/2.0*nxy*nyz*nz*nz - 81.0/2.0*nxy*nyz + (315.0/2.0)*nxz*nz*nzz*nz - 189.0/2.0*nxz*nz*nz + (27.0/2.0)*nxz);  // power=7
        out[21] = T((297.0/2.0)*nyy*nyy*nyz - 297*nyy*nyz*nz*nz - 54*nyy*nyz + (297.0/2.0)*nyz*nz*nzz*nz - 54*nyz*nz*nz + 18*nyz);  // power=7
        out[22] = T((891.0/4.0)*nxx*nyy*nyy - 891.0/2.0*nxx*nyy*nzz - 81*nxx*nyy + (891.0/4.0)*nxx*nzz*nzz - 81*nxx*nzz + (27.0/2.0)*nxx - 891.0/4.0*nyy*nyy*nzz + 297*nyy*nzz*nzz + 9*nyy - 297.0/4.0*nzz*nzz*nzz + 54*nzz*nzz - 9*nzz - 9.0/2.0);  // power=7
        out[23] = T((891.0/2.0)*nxx*nyy*nyz - 891.0/2.0*nxx*nyz*nz*nz - 297.0/2.0*nyy*nyz*nz*nz + (297.0/2.0)*nyz*nz*nzz*nz - 54*nyz*nz*nz + 9*nyz);  // power=7
        out[24] = T(-18*nxx*nxx*nxz + 72*nxx*nxy*nyz + 18*nxx*nxz*nz*nz + (1071.0/2.0)*nxy*ny*nyy*nz - 468*nxy*nyz*nz*nz - 189*nxy*nyz + (369.0/2.0)*nxz*nz*nzz*nz - 135*nxz*nz*nz + 36*nxz);  // power=7
        out[25] = T(-9.0/2.0*nxx*nxx*nx + (45.0/2.0)*nxx*nxy*ny + (45.0/2.0)*nxx*nxz*nz + (621.0/2.0)*nxy*ny*nyy - 459.0/2.0*nxy*nyz*nz - 189*nxy*ny + 27*nxz*nz*nzz + (27.0/2.0)*nx);  // power=6
        out[26] = T((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nxy*ny - 45.0/2.0*nxx*nxz*nz - 27*nxy*ny*nyy + (459.0/2.0)*nxy*nyz*nz - 621.0/2.0*nxz*nz*nzz + 189*nxz*nz - 27.0/2.0*nx);  // power=6
        out[27] = T((567.0/2.0)*nyy*nyz*nz - 63.0/2.0*nyy*ny - 567.0/2.0*nyz*nz*nzz + (189.0/2.0)*nyz*nz);  // power=6
    } break;

    case 6: {
        out[0] = T(9*nxy);  // power=3
        out[1] = T(36*nxx*ny - 9*nyy*ny - 9*nyz*nz);  // power=4
        out[2] = T(-9*nxx*nx + 36*nxy*ny - 9*nxz*nz);  // power=4
        out[3] = T(45*nxy*nz);  // power=4
        out[4] = T(30*nxx*nxy - 45.0/2.0*nxy*ny*ny - 45.0/2.0*nxy*nzz);  // power=5
        out[5] = T((15.0/2.0)*nxx*nxy + (15.0/2.0)*nxy*ny*ny - 45*nxy*nzz);  // power=5
        out[6] = T(105*nxx*nyy - 15*nxx - 15*nyy + 3);  // power=5
        out[7] = T(90*nxx*nyz - 15*nyy*nyz - 15*nyz*nz*nz);  // power=5
        out[8] = T(-15*nxx*nxz + 90*nxy*nyz - 15*nxz*nz*nz);  // power=5
        out[9] = T((189.0/2.0)*nxx*nxx*ny - 63*nxx*ny + (9.0/2.0)*ny);  // power=6
        out[10] = T((189.0/2.0)*nxy*ny*nyy - 63*nxy*ny + (9.0/2.0)*nx);  // power=6
        out[11] = T(-63.0/2.0*nxx*nxy*nz - 63.0/2.0*nxy*ny*nyz + 63*nxy*nzz*nz);  // power=6
        out[12] = T((567.0/2.0)*nxx*nxy*ny - 63.0/2.0*nxx*nx - 189.0/2.0*nxy*ny + (27.0/2.0)*nx);  // power=6
        out[13] = T(189*nxx*nxy*nz - 189.0/2.0*nxy*ny*nyz - 189.0/2.0*nxy*nzz*nz);  // power=6
        out[14] = T(-189.0/2.0*nxx*nxy*nz + 189*nxy*ny*nyz - 189.0/2.0*nxy*nzz*nz);  // power=6
        out[15] = T(567*nxx*nyy*nz - 63*nxx*nz - 63*nyy*nz + 9*nz);  // power=6
        out[16] = T((297.0/8.0)*nxx*nxx*nxy - 891.0/4.0*nxx*nxy*nzz - 27.0/2.0*nxx*nxy + (297.0/8.0)*nxy*nzz*nzz + (81.0/2.0)*nxy*nzz);  // power=7
        out[17] = T(-27.0/4.0*nxx*nxx*nxy + (27.0/4.0)*nxx*nxy*ny*ny + (189.0/4.0)*nxx*nxy*nzz + (405.0/8.0)*nxy*ny*nyy*ny - 621.0/4.0*nxy*ny*nyz*nz - 135.0/4.0*nxy*ny*ny + (729.0/8.0)*nxy*nzz*nzz - 81.0/4.0*nxy*nzz + (27.0/4.0)*nxy);  // power=7
        out[18] = T((297.0/2.0)*nxx*nyy*nyy - 891.0/2.0*nxx*nyy*nzz - 81.0/2.0*nxx*nyy + (81.0/2.0)*nxx*nzz + (9.0/2.0)*nxx - 27.0/2.0*nyy*nyy + (81.0/2.0)*nyy*nzz + 9*nyy - 9.0/2.0);  // power=7
        out[19] = T((297.0/2.0)*nxx*nxx*nyy - 27.0/2.0*nxx*nxx - 891.0/2.0*nxx*nyy*nzz - 81.0/2.0*nxx*nyy + (81.0/2.0)*nxx*nzz + 9*nxx + (81.0/2.0)*nyy*nzz + (9.0/2.0)*nyy - 9.0/2.0);  // power=7
        out[20] = T((297.0/2.0)*nxx*nxx*nyz - 297.0/2.0*nxx*nyz*nz*nz - 81.0/2.0*nxx*nyz + (27.0/2.0)*nyz*nz*nz);  // power=7
        out[21] = T(-9.0/2.0*nxx*nxx*nxz + (27.0/2.0)*nxx*nxy*nyz + (9.0/2.0)*nxx*nxz*nz*nz + (333.0/2.0)*nxy*ny*nyy*nz - 243.0/2.0*nxy*nyz*nz*nz - 63*nxy*nyz + 9*nxz*nz*nzz*nz + (9.0/2.0)*nxz);  // power=7
        out[22] = T(-27.0/4.0*nxx*nxx*nxy + (459.0/2.0)*nxx*nxy*ny*ny - 351.0/2.0*nxx*nxy*nzz - 81.0/2.0*nxx*nxy + (27.0/2.0)*nxy*ny*nyy*ny - 621.0/4.0*nxy*ny*nyz*nz - 243.0/4.0*nxy*ny*ny + (513.0/4.0)*nxy*nzz*nzz + (81.0/4.0)*nxy*nzz + (63.0/4.0)*nxy);  // power=7
        out[23] = T(-9.0/2.0*nxx*nxx*nxz + 477*nxx*nxy*nyz + (9.0/2.0)*nxx*nxz*nz*nz - 81.0/2.0*nxx*nxz + 36*nxy*ny*nyy*nz - 207.0/2.0*nxy*nyz*nz*nz - 243.0/2.0*nxy*nyz + 9*nxz*nz*nzz*nz + (27.0/2.0)*nxz);  // power=7
        out[24] = T((891.0/2.0)*nxx*nyy*nyz - 297.0/2.0*nxx*nyz*nz*nz - 81*nxx*nyz - 81.0/2.0*nyy*nyz + (27.0/2.0)*nyz*nz*nz + 9*nyz);  // power=7
        out[25] = T((567.0/2.0)*nxx*nyy*ny - 189.0/2.0*nxx*ny - 63.0/2.0*nyy*ny + (27.0/2.0)*ny);  // power=6
        out[26] = T((567.0/2.0)*nxx*nyz*nz - 63.0/2.0*nxx*ny - 63.0/2.0*nyz*nz + (9.0/2.0)*ny);  // power=6
        out[27] = T((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nxy*ny - 45.0/2.0*nxx*nxz*nz - 27*nxy*ny*nyy + (459.0/2.0)*nxy*nyz*nz - 27*nxz*nz*nzz);  // power=6
    } break;

    case 7: {
        out[0] = T(9*nxz);  // power=3
        out[1] = T(36*nxx*nz - 9*nyy*nz - 9*nzz*nz);  // power=4
        out[2] = T(45*nxy*nz);  // power=4
        out[3] = T(-9*nxx*nx - 9*nxy*ny + 36*nxz*nz);  // power=4
        out[4] = T(30*nxx*nxz - 45.0/2.0*nxy*nyz - 45.0/2.0*nxz*nz*nz);  // power=5
        out[5] = T((45.0/2.0)*nxx*nxz + (45.0/2.0)*nxy*nyz - 30*nxz*nz*nz);  // power=5
        out[6] = T(90*nxx*nyz - 15*nyy*nyz - 15*nyz*nz*nz);  // power=5
        out[7] = T(105*nxx*nzz - 15*nxx - 15*nzz + 3);  // power=5
        out[8] = T(-15*nxx*nxy - 15*nxy*ny*ny + 90*nxy*nzz);  // power=5
        out[9] = T((189.0/2.0)*nxx*nxx*nz - 63*nxx*nz + (9.0/2.0)*nz);  // power=6
        out[10] = T(-63.0/2.0*nxx*nxy*nz + 63*nxy*ny*nyz - 63.0/2.0*nxy*nzz*nz);  // power=6
        out[11] = T((189.0/2.0)*nxz*nz*nzz - 63*nxz*nz + (9.0/2.0)*nx);  // power=6
        out[12] = T(189*nxx*nxy*nz - 189.0/2.0*nxy*ny*nyz - 189.0/2.0*nxy*nzz*nz);  // power=6
        out[13] = T((567.0/2.0)*nxx*nxz*nz - 63.0/2.0*nxx*nx - 189.0/2.0*nxz*nz + (27.0/2.0)*nx);  // power=6
        out[14] = T((9.0/2.0)*nxx*nxx*nx - 45.0/2.0*nxx*nxy*ny - 45.0/2.0*nxx*nxz*nz - 27*nxy*ny*nyy + (459.0/2.0)*nxy*nyz*nz - 27*nxz*nz*nzz);  // power=6
        out[15] = T(567*nxx*nyz*nz - 63*nxx*ny - 63*nyz*nz + 9*ny);  // power=6
        out[16] = T((297.0/8.0)*nxx*nxx*nxz - 891.0/4.0*nxx*nxz*nz*nz + 27*nxx*nxz + (297.0/8.0)*nxz*nz*nzz*nz + 27*nxz*nz*nz - 9*nxz);  // power=7
        out[17] = T(-27.0/4.0*nxx*nxx*nxz + (171.0/4.0)*nxx*nxy*nyz + (27.0/4.0)*nxx*nxz*nz*nz + (693.0/8.0)*nxy*ny*nyy*nz - 639.0/4.0*nxy*nyz*nz*nz - 63.0/4.0*nxy*nyz + (405.0/8.0)*nxz*nz*nzz*nz - 135.0/4.0*nxz*nz*nz + (27.0/4.0)*nxz);  // power=7
        out[18] = T((297.0/2.0)*nxx*nyy*nyz - 891.0/2.0*nxx*nyz*nz*nz + 81*nxx*nyz - 27.0/2.0*nyy*nyz + (81.0/2.0)*nyz*nz*nz - 9*nyz);  // power=7
        out[19] = T((297.0/2.0)*nxx*nxx*nyz - 891.0/2.0*nxx*nyz*nz*nz + (81.0/2.0)*nxx*nyz + (81.0/2.0)*nyz*nz*nz - 9*nyz);  // power=7
        out[20] = T((297.0/2.0)*nxx*nxx*nzz - 27.0/2.0*nxx*nxx - 297.0/2.0*nxx*nzz*nzz + (9.0/2.0)*nxx + (27.0/2.0)*nzz*nzz - 9.0/2.0*nzz);  // power=7
        out[21] = T((9.0/2.0)*nxx*nxx*nxy - 9.0/2.0*nxx*nxy*ny*ny - 63.0/2.0*nxx*nxy*nzz - 9*nxy*ny*nyy*ny + (207.0/2.0)*nxy*ny*nyz*nz - 369.0/2.0*nxy*nzz*nzz + 81*nxy*nzz - 9.0/2.0*nxy);  // power=7
        out[22] = T(-27.0/4.0*nxx*nxx*nxz + 270*nxx*nxy*nyz - 216*nxx*nxz*nz*nz + (81.0/2.0)*nxx*nxz + 54*nxy*ny*nyy*nz - 621.0/4.0*nxy*nyz*nz*nz - 243.0/4.0*nxy*nyz + (351.0/4.0)*nxz*nz*nzz*nz - 27.0/4.0*nxz*nz*nz - 9.0/4.0*nxz);  // power=7
        out[23] = T((891.0/2.0)*nxx*nxy*nzz - 81.0/2.0*nxx*nxy - 297.0/2.0*nxy*nzz*nzz - 81.0/2.0*nxy*nzz + 9*nxy);  // power=7
        out[24] = T((891.0/2.0)*nxx*nyy*nzz - 81.0/2.0*nxx*nyy - 297.0/2.0*nxx*nzz*nzz + (81.0/2.0)*nxx*nzz - 9.0/2.0*nxx - 81.0/2.0*nyy*nzz + (27.0/2.0)*nzz*nzz - 9*nzz + 9.0/2.0);  // power=7
        out[25] = T((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 63.0/2.0*nyy*nz + (9.0/2.0)*nz);  // power=6
        out[26] = T((567.0/2.0)*nxx*nzz*nz - 189.0/2.0*nxx*nz - 63.0/2.0*nzz*nz + (27.0/2.0)*nz);  // power=6
        out[27] = T(-189.0/2.0*nxx*nxy*nz - 189.0/2.0*nxy*ny*nyz + 189*nxy*nzz*nz);  // power=6
    } break;

    case 8: {
        out[0] = T(9*nyz);  // power=3
        out[1] = T(45*nxy*nz);  // power=4
        out[2] = T(-9*nxx*nz + 36*nyy*nz - 9*nzz*nz);  // power=4
        out[3] = T(-9*nxx*ny - 9*nyy*ny + 36*nyz*nz);  // power=4
        out[4] = T(45*nxx*nyz - 15.0/2.0*nyy*nyz - 15.0/2.0*nyz*nz*nz);  // power=5
        out[5] = T((45.0/2.0)*nxx*nyz + (45.0/2.0)*nyy*nyz - 30*nyz*nz*nz);  // power=5
        out[6] = T(-15*nxx*nxz + 90*nxy*nyz - 15*nxz*nz*nz);  // power=5
        out[7] = T(-15*nxx*nxy - 15*nxy*ny*ny + 90*nxy*nzz);  // power=5
        out[8] = T(105*nyy*nzz - 15*nyy - 15*nzz + 3);  // power=5
        out[9] = T(63*nxx*nxy*nz - 63.0/2.0*nxy*ny*nyz - 63.0/2.0*nxy*nzz*nz);  // power=6
        out[10] = T((189.0/2.0)*nyy*nyy*nz - 63*nyy*nz + (9.0/2.0)*nz);  // power=6
        out[11] = T((189.0/2.0)*nyz*nz*nzz - 63*nyz*nz + (9.0/2.0)*ny);  // power=6
        out[12] = T((567.0/2.0)*nxx*nyy*nz - 63.0/2.0*nxx*nz - 63.0/2.0*nyy*nz + (9.0/2.0)*nz);  // power=6
        out[13] = T((567.0/2.0)*nxx*nyz*nz - 63.0/2.0*nxx*ny - 63.0/2.0*nyz*nz + (9.0/2.0)*ny);  // power=6
        out[14] = T((567.0/2.0)*nyy*nyz*nz - 63.0/2.0*nyy*ny - 189.0/2.0*nyz*nz + (27.0/2.0)*ny);  // power=6
        out[15] = T(9*nxx*nxx*nx - 45*nxx*nxy*ny - 45*nxx*nxz*nz - 54*nxy*ny*nyy + 459*nxy*nyz*nz - 54*nxz*nz*nzz);  // power=6
        out[16] = T((297.0/8.0)*nxx*nxx*nyz - 891.0/4.0*nxx*nyz*nz*nz + (81.0/2.0)*nxx*nyz + (297.0/8.0)*nyz*nz*nzz*nz - 27.0/2.0*nyz*nz*nz);  // power=7
        out[17] = T((297.0/8.0)*nyy*nyy*nyz - 891.0/4.0*nyy*nyz*nz*nz + 27*nyy*nyz + (297.0/8.0)*nyz*nz*nzz*nz + 27*nyz*nz*nz - 9*nyz);  // power=7
        out[18] = T(-27.0/2.0*nxx*nxx*nxz + (153.0/2.0)*nxx*nxy*nyz + (27.0/2.0)*nxx*nxz*nz*nz + (477.0/2.0)*nxy*ny*nyy*nz - 657.0/2.0*nxy*nyz*nz*nz - 63*nxy*nyz + 27*nxz*nz*nzz*nz + (9.0/2.0)*nxz);  // power=7
        out[19] = T(-27.0/2.0*nxx*nxx*nxz + 243*nxx*nxy*nyz + (27.0/2.0)*nxx*nxz*nz*nz - 27.0/2.0*nxx*nxz + 108*nxy*ny*nyy*nz - 621.0/2.0*nxy*nyz*nz*nz - 81.0/2.0*nxy*nyz + 27*nxz*nz*nzz*nz + (9.0/2.0)*nxz);  // power=7
        out[20] = T((297.0/2.0)*nxx*nxy*nzz - 27.0/2.0*nxx*nxy - 297.0/2.0*nxy*nzz*nzz + (81.0/2.0)*nxy*nzz);  // power=7
        out[21] = T((297.0/2.0)*nyy*nyy*nzz - 27.0/2.0*nyy*nyy - 297.0/2.0*nyy*nzz*nzz + (9.0/2.0)*nyy + (27.0/2.0)*nzz*nzz - 9.0/2.0*nzz);  // power=7
        out[22] = T((891.0/4.0)*nxx*nyy*nyz - 891.0/4.0*nxx*nyz*nz*nz - 891.0/4.0*nyy*nyz*nz*nz + (81.0/2.0)*nyy*nyz + (297.0/4.0)*nyz*nz*nzz*nz + (27.0/2.0)*nyz*nz*nz - 9*nyz);  // power=7
        out[23] = T((891.0/2.0)*nxx*nyy*nzz - 81.0/2.0*nxx*nyy - 81.0/2.0*nxx*nzz - 297.0/2.0*nyy*nzz*nzz + (81.0/2.0)*nyy*nzz - 9.0/2.0*nyy + (27.0/2.0)*nzz*nzz - 9*nzz + 9.0/2.0);  // power=7
        out[24] = T((27.0/2.0)*nxx*nxx*nxy - 27.0/2.0*nxx*nxy*ny*ny - 189.0/2.0*nxx*nxy*nzz - 27*nxy*ny*nyy*ny + (621.0/2.0)*nxy*ny*nyz*nz - 513.0/2.0*nxy*nzz*nzz + 81*nxy*nzz - 9.0/2.0*nxy);  // power=7
        out[25] = T(-189.0/2.0*nxx*nxy*nz + 189*nxy*ny*nyz - 189.0/2.0*nxy*nzz*nz);  // power=6
        out[26] = T(-189.0/2.0*nxx*nxy*nz - 189.0/2.0*nxy*ny*nyz + 189*nxy*nzz*nz);  // power=6
        out[27] = T((567.0/2.0)*nyy*nzz*nz - 189.0/2.0*nyy*nz - 63.0/2.0*nzz*nz + (27.0/2.0)*nz);  // power=6
    } break;

    case 9: {
        out[0] = T(-30*nxx*nx + 45*nxy*ny + 45*nxz*nz);  // power=4
        out[1] = T(-210*nxx*nxx + 315*nxx*nyy + 315*nxx*nzz + 90*nxx - 45*nyy - 45*nzz);  // power=5
        out[2] = T(-300*nxx*nxy + 225*nxy*ny*ny + 225*nxy*nzz);  // power=5
        out[3] = T(-300*nxx*nxz + 225*nxy*nyz + 225*nxz*nz*nz);  // power=5
        out[4] = T(-215*nxx*nxx*nx + 445*nxx*nxy*ny + 445*nxx*nxz*nz + 140*nxx*nx - 255.0/2.0*nxy*ny*nyy - 255*nxy*nyz*nz - 35*nxy*ny - 255.0/2.0*nxz*nz*nzz - 35*nxz*nz - 25*nx);  // power=6
        out[5] = T(-215.0/2.0*nxx*nxx*nx + 65*nxx*nxy*ny + 380*nxx*nxz*nz + 70*nxx*nx + (345.0/2.0)*nxy*ny*nyy - 255.0/2.0*nxy*nyz*nz - 245.0/2.0*nxy*ny - 300*nxz*nz*nzz + (175.0/2.0)*nxz*nz - 25.0/2.0*nx);  // power=6
        out[6] = T(-630*nxx*nxx*ny + 945*nxx*nyy*ny + 945*nxx*nyz*nz - 105*nyy*ny - 105*nyz*nz + 30*ny);  // power=6
        out[7] = T(-630*nxx*nxx*nz + 945*nxx*nyy*nz + 945*nxx*nzz*nz - 105*nyy*nz - 105*nzz*nz + 30*nz);  // power=6
        out[8] = T(-1050*nxx*nxy*nz + 525*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[9] = T(-693*nxx*nxx*nxx + (2079.0/2.0)*nxx*nxx*nyy + (2079.0/2.0)*nxx*nxx*nzz + 756*nxx*nxx - 567*nxx*nyy - 567*nxx*nzz - 189*nxx + (63.0/2.0)*nyy + (63.0/2.0)*nzz + 6);  // power=7
        out[10] = T((63.0/2.0)*nxx*nxx*nxy - 1449.0/2.0*nxx*nxy*ny*ny - 441.0/2.0*nxx*nxy*nzz + 189*nxx*nxy + (1953.0/2.0)*nxy*ny*nyy*ny + (1449.0/2.0)*nxy*ny*nyz*nz - 756*nxy*ny*ny - 252*nxy*nzz*nzz + (189.0/2.0)*nxy);  // power=7
        out[11] = T((63.0/2.0)*nxx*nxx*nxz - 441.0/2.0*nxx*nxy*nyz - 1449.0/2.0*nxx*nxz*nz*nz + 189*nxx*nxz - 252*nxy*ny*nyy*nz + (1449.0/2.0)*nxy*nyz*nz*nz + (1953.0/2.0)*nxz*nz*nzz*nz - 756*nxz*nz*nz + (189.0/2.0)*nxz);  // power=7
        out[12] = T(-2079*nxx*nxx*nxy + (6237.0/2.0)*nxx*nxy*ny*ny + (6237.0/2.0)*nxx*nxy*nzz + 756*nxx*nxy - 1701.0/2.0*nxy*ny*ny - 1701.0/2.0*nxy*nzz + 63*nxy);  // power=7
        out[13] = T(-2079*nxx*nxx*nxz + (6237.0/2.0)*nxx*nxy*nyz + (6237.0/2.0)*nxx*nxz*nz*nz + 756*nxx*nxz - 1701.0/2.0*nxy*nyz - 1701.0/2.0*nxz*nz*nz + 63*nxz);  // power=7
        out[14] = T((189.0/2.0)*nxx*nxx*nxz - 6237.0/2.0*nxx*nxy*nyz - 189.0/2.0*nxx*nxz*nz*nz + 189*nxx*nxz + (3969.0/2.0)*nxy*ny*nyy*nz + (3591.0/2.0)*nxy*nyz*nz*nz - 756*nxy*nyz - 189*nxz*nz*nzz*nz + (63.0/2.0)*nxz);  // power=7
        out[15] = T(-4158*nxx*nxx*nyz + 6237*nxx*nyy*nyz + 6237*nxx*nyz*nz*nz - 1134000000000001.0/1000000000000.0*nxx*nyz - 567*nyy*nyz - 567*nyz*nz*nz + 252*nyz);  // power=7
        out[16] = T(-1287.0/4.0*nxx*nxx*nxx*nx - 9.0/8.0*nxx*nxx*nxx*nx + (482624999998552.0/999999999997.0)*nxx*nxx*nxy*ny + (27.0/4.0)*nxx*nxx*nxy*ny + (19305.0/8.0)*nxx*nxx*nxz*nz + (135.0/8.0)*nxx*nxx*nxz*nz + 297*nxx*nxx*nx + (135.0/8.0)*nxx*nxy*ny*nyy - 11583.0/4.0*nxx*nxy*nyz*nz - 675.0/4.0*nxx*nxy*nyz*nz - 297.0/2.0*nxx*nxy*ny - 6435.0/2.0*nxx*nxz*nz*nzz - 148499999999851.0/999999999999.0*nxx*nxz*nz - 243.0/2.0*nxx*nx + 9*nxy*ny*nyy*nyy - 1485.0/8.0*nxy*ny*nyy*nzz + 270*nxy*nyz*nz*nzz + (2673.0/4.0)*nxy*nyz*nz - 81.0/8.0*nxy*ny + (482624999998552.0/999999999997.0)*nxz*nz*nzz*nzz - 18*nxz*nz*nzz*nzz + (1485.0/8.0)*nxz*nz*nzz - 243.0/4.0*nxz*nz + (129.0/8.0)*nx);  // power=8
        out[17] = T((45.0/8.0)*nxx*nxx*nxx*nx - 135.0/4.0*nxx*nxx*nxy*ny - 675.0/8.0*nxx*nxx*nxz*nz - 1629.0/8.0*nxx*nxy*ny*nyy - 675.0/8.0*nxx*nxy*ny*nyy + (4491.0/2.0)*nxx*nxy*nyz*nz + (3375.0/4.0)*nxx*nxy*nyz*nz - 315.0/4.0*nxx*nxy*ny - 1287.0/4.0*nxx*nxz*nz*nzz + (45.0/8.0)*nxx*nx + (2403.0/4.0)*nxy*ny*nyy*nyy - 45*nxy*ny*nyy*nyy - 1980*nxy*ny*nyy*nzz + (7425.0/8.0)*nxy*ny*nyy*nzz - 493874999999506.0/999999999999.0*nxy*ny*nyy + 315*nxy*nyz*nz*nzz - 1350*nxy*nyz*nz*nzz + (549.0/2.0)*nxy*nyz*nz + 135*nxy*ny + (482624999998552.0/999999999997.0)*nxz*nz*nzz*nzz + 90*nxz*nz*nzz*nzz - 482624999998552.0/999999999997.0*nxz*nz*nzz + (855.0/8.0)*nxz*nz - 45.0/4.0*nx);  // power=8
        out[18] = T(-1287*nxx*nxx*nyy*ny + 3861*nxx*nxx*nyz*nz + (1930499999998069.0/999999999999.0)*nxx*nyy*nyy*ny - 3861*nxx*nyy*nyz*nz - 594*nxx*nyy*ny - 11583.0/2.0*nxx*nyz*nz*nzz + 1782*nxx*nyz*nz - 297.0/2.0*nyy*nyy*ny + 297*nyy*nyz*nz + 81*nyy*ny + (891.0/2.0)*nyz*nz*nzz - 243*nyz*nz);  // power=8
        out[19] = T(-1287*nxx*nxx*nxx*ny + (1930499999998069.0/999999999999.0)*nxx*nxx*nyy*ny + (11583.0/2.0)*nxx*nxx*nyz*nz + 594*nxx*nxx*ny - 11583.0/2.0*nxx*nyy*nyz*nz - 891.0/2.0*nxx*nyy*ny - 11583.0/2.0*nxx*nyz*nz*nzz + (2673.0/2.0)*nxx*nyz*nz - 162*nxx*ny + (891.0/2.0)*nyy*nyz*nz + (891.0/2.0)*nyz*nz*nzz - 243*nyz*nz + 15*ny);  // power=8
        out[20] = T(-1287*nxx*nxx*nxx*nz + (1930499999998069.0/999999999999.0)*nxx*nxx*nyy*nz + (3217499999996782.0/999999999999.0)*nxx*nxx*nzz*nz + 594*nxx*nxx*nz - 1930499999998069.0/999999999999.0*nxx*nyy*nzz*nz - 891.0/2.0*nxx*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nzz*nzz*nz + (297.0/2.0)*nxx*nzz*nz - 162*nxx*nz + (297.0/2.0)*nyy*nzz*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz + 15*nz);  // power=8
        out[21] = T(-1557*nxx*nxy*ny*nyz + 1287*nxx*nxy*nzz*nz + (3321.0/2.0)*nxy*ny*nyy*nyz - 270*nxy*ny*nyz*nz*nz - 918*nxy*ny*nyz - 1930499999998069.0/999999999999.0*nxy*nzz*nzz*nz + 1188*nxy*nzz*nz);  // power=8
        out[22] = T((9.0/2.0)*nxx*nxx*nxx*nx - 3861.0/2.0*nxx*nxx*nxy*ny - 27*nxx*nxx*nxy*ny + (3861.0/2.0)*nxx*nxx*nxz*nz - 135.0/2.0*nxx*nxx*nxz*nz + (6075.0/2.0)*nxx*nxy*ny*nyy - 135.0/2.0*nxx*nxy*ny*nyy + (4617.0/2.0)*nxx*nxy*nyz*nz + 675*nxx*nxy*nyz*nz - 94499999999906.0/999999999999.0*nxx*nxy*ny - 1945723264306643.0/549755813889.0*nxx*nxz*nz*nzz + (27.0/4.0)*nxx*nx + (567.0/4.0)*nxy*ny*nyy*nyy - 36*nxy*ny*nyy*nyy - 2376*nxy*ny*nyy*nzz + (1485.0/2.0)*nxy*ny*nyy*nzz - 2727.0/4.0*nxy*ny*nyy + 378*nxy*nyz*nz*nzz - 1080*nxy*nyz*nz*nzz + (837.0/2.0)*nxy*nyz*nz + (729.0/4.0)*nxy*ny + (965249999997104.0/999999999997.0)*nxz*nz*nzz*nzz + 72*nxz*nz*nzz*nzz - 297*nxz*nz*nzz + (189.0/4.0)*nxz*nz - 45.0/4.0*nx);  // power=8
        out[23] = T(-7965.0/2.0*nxx*nxx*nxy*nz + 5994*nxx*nxy*ny*nyz + (7280999999999999.0/1000000000000.0)*nxx*nxy*nzz*nz + 324*nxy*ny*nyy*nyz - 2565.0/2.0*nxy*ny*nyz*nz*nz - 2673.0/2.0*nxy*ny*nyz - 3213.0/2.0*nxy*nzz*nzz*nz - 297.0/2.0*nxy*nzz*nz + (405.0/2.0)*nxy*nz);  // power=8
        out[24] = T(-3861*nxx*nxx*nyy*nz + 1287*nxx*nxx*nzz*nz + (11583.0/2.0)*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 1782*nxx*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nzz*nzz*nz + 594*nxx*nzz*nz - 891.0/2.0*nyy*nyy*nz - 297*nyy*nzz*nz + 243*nyy*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz);  // power=8
        out[25] = T(-2079*nxx*nxx*nyy + 189*nxx*nxx + (6237.0/2.0)*nxx*nyy*nyy + (6237.0/2.0)*nxx*nyy*nzz - 1701.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz - 63.0/2.0*nxx - 567.0/2.0*nyy*nyy - 567.0/2.0*nyy*nzz + 126*nyy + 45.0/2.0);  // power=7
        out[26] = T(-2079*nxx*nxx*nzz + 189*nxx*nxx + (6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy + (6237.0/2.0)*nxx*nzz*nzz - 1701.0/2.0*nxx*nzz - 63.0/2.0*nxx - 567.0/2.0*nyy*nzz - 567.0/2.0*nzz*nzz + 126*nzz + 45.0/2.0);  // power=7
        out[27] = T((189.0/2.0)*nxx*nxx*nxy - 189.0/2.0*nxx*nxy*ny*ny - 2740499999997259.0/999999999999.0*nxx*nxy*nzz + 189*nxx*nxy - 189*nxy*ny*nyy*ny + (4347.0/2.0)*nxy*ny*nyz*nz + (4725.0/2.0)*nxy*nzz*nzz - 1134*nxy*nzz + (63.0/2.0)*nxy);  // power=7
    } break;

    case 10: {
        out[0] = T(45*nxx*ny - 30*nyy*ny + 45*nyz*nz);  // power=4
        out[1] = T(225*nxx*nxy - 300*nxy*ny*ny + 225*nxy*nzz);  // power=5
        out[2] = T(315*nxx*nyy - 45*nxx - 210*nyy*nyy + 315*nyy*nzz + 90*nyy - 45*nzz);  // power=5
        out[3] = T(225*nxx*nyz - 300*nyy*nyz + 225*nyz*nz*nz);  // power=5
        out[4] = T(315*nxx*nxx*ny - 735.0/2.0*nxx*nyy*ny + (315.0/2.0)*nxx*nyz*nz - 105*nxx*ny + 105*nyy*nyy*ny - 105.0/2.0*nyy*nyz*nz - 70*nyy*ny - 315.0/2.0*nyz*nz*nzz + 105*nyz*nz + 15*ny);  // power=6
        out[5] = T((315.0/2.0)*nxx*nxx*ny + (105.0/2.0)*nxx*nyy*ny - 315.0/2.0*nxx*nyz*nz - 105*nxx*ny - 105*nyy*nyy*ny + (735.0/2.0)*nyy*nyz*nz + 70*nyy*ny - 315*nyz*nz*nzz + 105*nyz*nz - 15*ny);  // power=6
        out[6] = T(15*nxx*nxx*nx + 870*nxx*nxy*ny - 75*nxx*nxz*nz - 105*nxx*nx - 720*nxy*ny*nyy + 765*nxy*nyz*nz + 105*nxy*ny - 90*nxz*nz*nzz + 15*nx);  // power=6
        out[7] = T(525*nxx*nxy*nz - 1050*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[8] = T(945*nxx*nyy*nz - 105*nxx*nz - 630*nyy*nyy*nz + 945*nyy*nzz*nz - 105*nzz*nz + 30*nz);  // power=6
        out[9] = T((2079.0/2.0)*nxx*nxx*nxy - 693*nxx*nxy*ny*ny + (2079.0/2.0)*nxx*nxy*nzz - 1701.0/2.0*nxx*nxy + 189*nxy*ny*ny - 567.0/2.0*nxy*nzz + 126*nxy);  // power=7
        out[10] = T((2079.0/2.0)*nxx*nyy*nyy - 567*nxx*nyy + (63.0/2.0)*nxx - 693*nyy*nyy*nyy + (2079.0/2.0)*nyy*nyy*nzz + 756*nyy*nyy - 567*nyy*nzz - 189*nyy + (63.0/2.0)*nzz + 6);  // power=7
        out[11] = T((2079.0/2.0)*nxx*nyz*nz*nz - 567.0/2.0*nxx*nyz - 693*nyy*nyz*nz*nz + 189*nyy*nyz + (2079.0/2.0)*nyz*nz*nzz*nz - 1701.0/2.0*nyz*nz*nz + 126*nyz);  // power=7
        out[12] = T((6237.0/2.0)*nxx*nxx*nyy - 567.0/2.0*nxx*nxx - 2079*nxx*nyy*nyy + (6237.0/2.0)*nxx*nyy*nzz - 1701.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz + 126*nxx + 189*nyy*nyy - 567.0/2.0*nyy*nzz - 63.0/2.0*nyy + 45.0/2.0);  // power=7
        out[13] = T((6237.0/2.0)*nxx*nxx*nyz - 2079*nxx*nyy*nyz + (6237.0/2.0)*nxx*nyz*nz*nz - 1984499999998016.0/999999999999.0*nxx*nyz + 189*nyy*nyz - 567.0/2.0*nyz*nz*nz + 126*nyz);  // power=7
        out[14] = T((6237.0/2.0)*nxx*nyy*nyz - 1701.0/2.0*nxx*nyz - 2079*nyy*nyy*nyz + (6237.0/2.0)*nyy*nyz*nz*nz + 756*nyy*nyz - 1701.0/2.0*nyz*nz*nz + 63*nyz);  // power=7
        out[15] = T(189*nxx*nxx*nxz + 5418*nxx*nxy*nyz - 189*nxx*nxz*nz*nz - 567*nxx*nxz - 5165999999999999.0/1000000000000.0*nxy*ny*nyy*nz + 4851*nxy*nyz*nz*nz + 63*nxy*nyz - 378*nxz*nz*nzz*nz + 63*nxz);  // power=7
        out[16] = T((482624999998552.0/999999999997.0)*nxx*nxx*nxx*ny - 1287.0/4.0*nxx*nxx*nyy*ny - 19305.0/8.0*nxx*nxx*nyz*nz - 297*nxx*nxx*ny + (3861.0/2.0)*nxx*nyy*nyz*nz - 19305.0/8.0*nxx*nyz*nz*nzz + 1782*nxx*nyz*nz - 1287.0/4.0*nyy*nyz*nz*nzz + (482624999998552.0/999999999997.0)*nyz*nz*nzz*nzz - 297*nyz*nz*nzz);  // power=8
        out[17] = T((482624999998552.0/999999999997.0)*nxx*nyy*nyy*ny - 11583.0/4.0*nxx*nyy*nyz*nz - 297.0/2.0*nxx*nyy*ny + (482624999998552.0/999999999997.0)*nxx*nyz*nz*nzz + (891.0/2.0)*nxx*nyz*nz - 1287.0/4.0*nyy*nyy*nyy*ny + (19305.0/8.0)*nyy*nyy*nyz*nz + 297*nyy*nyy*ny - 6435.0/2.0*nyy*nyz*nz*nzz - 148499999999851.0/999999999999.0*nyy*nyz*nz - 243.0/2.0*nyy*ny + (482624999998552.0/999999999997.0)*nyz*nz*nzz*nzz + (297.0/2.0)*nyz*nz*nzz - 81.0/2.0*nyz*nz + 15*ny);  // power=8
        out[18] = T((27.0/2.0)*nxx*nxx*nxx*nx - 81*nxx*nxx*nxy*ny - 405.0/2.0*nxx*nxx*nxz*nz + 1647*nxx*nxy*ny*nyy - 405.0/2.0*nxx*nxy*ny*nyy - 13095.0/2.0*nxx*nxy*nyz*nz + 2025*nxx*nxy*nyz*nz - 513.0/2.0*nxx*nxy*ny + (891.0/2.0)*nxx*nxz*nz - 27.0/2.0*nxx*nx - 3141.0/2.0*nxy*ny*nyy*nyy - 108*nxy*ny*nyy*nyy + 4752*nxy*ny*nyy*nzz + (4455.0/2.0)*nxy*ny*nyy*nzz + (2133.0/2.0)*nxy*ny*nyy - 756*nxy*nyz*nz*nzz - 3240*nxy*nyz*nz*nzz - 783.0/2.0*nxy*nyz*nz - 243*nxy*ny + 216*nxz*nz*nzz*nzz - 27.0/2.0*nxz*nz + 15*nx);  // power=8
        out[19] = T((27.0/2.0)*nxx*nxx*nxx*nx + (1930499999998069.0/999999999999.0)*nxx*nxx*nxy*ny - 81*nxx*nxx*nxy*ny - 405.0/2.0*nxx*nxx*nxz*nz - 297.0/2.0*nxx*nxx*nx - 1476*nxx*nxy*ny*nyy - 405.0/2.0*nxx*nxy*ny*nyy - 4365*nxx*nxy*nyz*nz + 2025*nxx*nxy*nyz*nz - 468*nxx*nxy*ny + 297*nxx*nxz*nz + 72*nxx*nx - 189*nxy*ny*nyy*nyy - 108*nxy*ny*nyy*nyy + 3168*nxy*ny*nyy*nzz + (4455.0/2.0)*nxy*ny*nyy*nzz + 315*nxy*ny*nyy - 504*nxy*nyz*nz*nzz - 3240*nxy*nyz*nz*nzz - 261*nxy*nyz*nz - 27.0/2.0*nxy*ny + 216*nxz*nz*nzz*nzz - 9*nxz*nz - 9.0/2.0*nx);  // power=8
        out[20] = T((4023.0/2.0)*nxx*nxx*nxy*nz - 1422*nxx*nxy*ny*nyz - 135*nxx*nxy*nzz*nz - 1188*nxx*nxy*nz - 216*nxy*ny*nyy*nyz + 855*nxy*ny*nyz*nz*nz + 297*nxy*ny*nyz - 4293.0/2.0*nxy*nzz*nzz*nz + 1485*nxy*nzz*nz - 81*nxy*nz);  // power=8
        out[21] = T((1930499999998069.0/999999999999.0)*nxx*nyy*nyy*nz - 1930499999998069.0/999999999999.0*nxx*nyy*nzz*nz - 891.0/2.0*nxx*nyy*nz + (297.0/2.0)*nxx*nzz*nz - 1287*nyy*nyy*nyy*nz + (3217499999996782.0/999999999999.0)*nyy*nyy*nzz*nz + 594*nyy*nyy*nz - 1930499999998069.0/999999999999.0*nyy*nzz*nzz*nz + (297.0/2.0)*nyy*nzz*nz - 162*nyy*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz + 15*nz);  // power=8
        out[22] = T((11583.0/4.0)*nxx*nxx*nyy*ny - 11583.0/4.0*nxx*nxx*nyz*nz - 891.0/2.0*nxx*nxx*ny - 3861.0/2.0*nxx*nyy*nyy*ny + (3861.0/2.0)*nxx*nyy*nyz*nz - 3861.0/2.0*nxx*nyz*nz*nzz + 1782*nxx*nyz*nz + (81.0/2.0)*nxx*ny + (3861.0/2.0)*nyy*nyy*nyz*nz - 1945723264306643.0/549755813889.0*nyy*nyz*nz*nzz + (965249999997104.0/999999999997.0)*nyz*nz*nzz*nzz - 297.0/2.0*nyz*nz*nzz - 81.0/2.0*nyz*nz);  // power=8
        out[23] = T((11583.0/2.0)*nxx*nxx*nyy*nz - 891.0/2.0*nxx*nxx*nz - 3861*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 1782*nxx*nyy*nz - 297*nxx*nzz*nz + 243*nxx*nz + 1287*nyy*nyy*nzz*nz - 1930499999998069.0/999999999999.0*nyy*nzz*nzz*nz + 594*nyy*nzz*nz + (297.0/2.0)*nzz*nzz*nz - 81*nzz*nz);  // power=8
        out[24] = T((891.0/2.0)*nxx*nxx*nxy*nz + (5589000000000001.0/1000000000000.0)*nxx*nxy*ny*nyz - 2673*nxx*nxy*nzz*nz - 891*nxx*nxy*nz - 4509*nxy*ny*nyy*nyz + (10485.0/2.0)*nxy*ny*nyz*nz*nz + (1093499999998906.0/999999999999.0)*nxy*ny*nyz - 3118499999996881.0/999999999999.0*nxy*nzz*nzz*nz + (3861.0/2.0)*nxy*nzz*nz - 729.0/2.0*nxy*nz);  // power=8
        out[25] = T((189.0/2.0)*nxx*nxx*nxy + 3024*nxx*nxy*ny*ny - 1323.0/2.0*nxx*nxy*nzz - 1701.0/2.0*nxx*nxy - 2268*nxy*ny*nyy*ny + (4347.0/2.0)*nxy*ny*nyz*nz + (2079.0/2.0)*nxy*ny*ny - 756*nxy*nzz*nzz - 63.0/2.0*nxy);  // power=7
        out[26] = T(-63*nxx*nxx*nxy + 63*nxx*nxy*ny*ny + (3559499999996441.0/999999999999.0)*nxx*nxy*nzz - 567.0/2.0*nxx*nxy + 126*nxy*ny*nyy*ny - 1449*nxy*ny*nyz*nz + (7245.0/2.0)*nxy*nzz*nzz - 5103.0/2.0*nxy*nzz + 189*nxy);  // power=7
        out[27] = T((6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz - 2079*nyy*nyy*nzz + 189*nyy*nyy + (6237.0/2.0)*nyy*nzz*nzz - 1701.0/2.0*nyy*nzz - 63.0/2.0*nyy - 567.0/2.0*nzz*nzz + 126*nzz + 45.0/2.0);  // power=7
    } break;

    case 11: {
        out[0] = T(45*nxx*nz + 45*nyy*nz - 30*nzz*nz);  // power=4
        out[1] = T(225*nxx*nxz + 225*nxy*nyz - 300*nxz*nz*nz);  // power=5
        out[2] = T(225*nxx*nyz + 225*nyy*nyz - 300*nyz*nz*nz);  // power=5
        out[3] = T(315*nxx*nzz - 45*nxx + 315*nyy*nzz - 45*nyy - 210*nzz*nzz + 90*nzz);  // power=5
        out[4] = T(315*nxx*nxx*nz + (315.0/2.0)*nxx*nyy*nz - 735.0/2.0*nxx*nzz*nz - 105*nxx*nz - 315.0/2.0*nyy*nyy*nz - 105.0/2.0*nyy*nzz*nz + 105*nyy*nz + 105*nzz*nzz*nz - 70*nzz*nz + 15*nz);  // power=6
        out[5] = T((315.0/2.0)*nxx*nxx*nz + 315*nxx*nyy*nz - 420*nxx*nzz*nz + (315.0/2.0)*nyy*nyy*nz - 420*nyy*nzz*nz + 210*nzz*nzz*nz - 140*nzz*nz + 30*nz);  // power=6
        out[6] = T(525*nxx*nxy*nz + 525*nxy*ny*nyz - 1050*nxy*nzz*nz);  // power=6
        out[7] = T(15*nxx*nxx*nx - 75*nxx*nxy*ny + 870*nxx*nxz*nz - 105*nxx*nx - 90*nxy*ny*nyy + 765*nxy*nyz*nz - 720*nxz*nz*nzz + 105*nxz*nz + 15*nx);  // power=6
        out[8] = T(945*nxx*nyz*nz - 105*nxx*ny + 945*nyy*nyz*nz - 105*nyy*ny - 630*nyz*nz*nzz + 30*ny);  // power=6
        out[9] = T((2079.0/2.0)*nxx*nxx*nxz + (2079.0/2.0)*nxx*nxy*nyz - 693*nxx*nxz*nz*nz - 1701.0/2.0*nxx*nxz - 567.0/2.0*nxy*nyz + 189*nxz*nz*nz + 126*nxz);  // power=7
        out[10] = T((2079.0/2.0)*nxx*nyy*nyz - 567.0/2.0*nxx*nyz + (2079.0/2.0)*nyy*nyy*nyz - 693*nyy*nyz*nz*nz - 1701.0/2.0*nyy*nyz + 189*nyz*nz*nz + 126*nyz);  // power=7
        out[11] = T((2079.0/2.0)*nxx*nzz*nzz - 567*nxx*nzz + (63.0/2.0)*nxx + (2079.0/2.0)*nyy*nzz*nzz - 567*nyy*nzz + (63.0/2.0)*nyy - 693*nzz*nzz*nzz + 756*nzz*nzz - 189*nzz + 6);  // power=7
        out[12] = T((6237.0/2.0)*nxx*nxx*nyz + (6237.0/2.0)*nxx*nyy*nyz - 2079*nxx*nyz*nz*nz - 1984499999998016.0/999999999999.0*nxx*nyz - 567.0/2.0*nyy*nyz + 189*nyz*nz*nz + 126*nyz);  // power=7
        out[13] = T((6237.0/2.0)*nxx*nxx*nzz - 567.0/2.0*nxx*nxx + (6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 2079*nxx*nzz*nzz - 1701.0/2.0*nxx*nzz + 126*nxx - 567.0/2.0*nyy*nzz + 189*nzz*nzz - 63.0/2.0*nzz + 45.0/2.0);  // power=7
        out[14] = T((6237.0/2.0)*nxx*nyy*nzz - 567.0/2.0*nxx*nyy - 567.0/2.0*nxx*nzz + (6237.0/2.0)*nyy*nyy*nzz - 567.0/2.0*nyy*nyy - 2079*nyy*nzz*nzz - 1701.0/2.0*nyy*nzz + 126*nyy + 189*nzz*nzz - 63.0/2.0*nzz + 45.0/2.0);  // power=7
        out[15] = T(189*nxx*nxx*nxy - 189*nxx*nxy*ny*ny + 4914*nxx*nxy*nzz - 567*nxx*nxy - 378*nxy*ny*nyy*ny + 4347*nxy*ny*nyz*nz - 5670*nxy*nzz*nzz + 567*nxy*nzz + 63*nxy);  // power=7
        out[16] = T((482624999998552.0/999999999997.0)*nxx*nxx*nxx*nz + (482624999998552.0/999999999997.0)*nxx*nxx*nyy*nz - 6435.0/2.0*nxx*nxx*nzz*nz + (297.0/2.0)*nxx*nxx*nz - 11583.0/4.0*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz + (19305.0/8.0)*nxx*nzz*nzz*nz - 148499999999851.0/999999999999.0*nxx*nzz*nz - 81.0/2.0*nxx*nz + (482624999998552.0/999999999997.0)*nyy*nzz*nzz*nz - 297.0/2.0*nyy*nzz*nz - 1287.0/4.0*nzz*nzz*nzz*nz + 297*nzz*nzz*nz - 243.0/2.0*nzz*nz + 15*nz);  // power=8
        out[17] = T((482624999998552.0/999999999997.0)*nxx*nyy*nyy*nz - 11583.0/4.0*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz + (482624999998552.0/999999999997.0)*nxx*nzz*nzz*nz - 297.0/2.0*nxx*nzz*nz + (482624999998552.0/999999999997.0)*nyy*nyy*nyy*nz - 6435.0/2.0*nyy*nyy*nzz*nz + (297.0/2.0)*nyy*nyy*nz + (19305.0/8.0)*nyy*nzz*nzz*nz - 148499999999851.0/999999999999.0*nyy*nzz*nz - 81.0/2.0*nyy*nz - 1287.0/4.0*nzz*nzz*nzz*nz + 297*nzz*nzz*nz - 243.0/2.0*nzz*nz + 15*nz);  // power=8
        out[18] = T(-891.0/2.0*nxx*nxx*nxy*nz + 2403*nxx*nxy*ny*nyz - 5049*nxx*nxy*nzz*nz + 891*nxx*nxy*nz + (5697.0/2.0)*nxy*ny*nyy*nyz - 9945.0/2.0*nxy*ny*nyz*nz*nz - 3321.0/2.0*nxy*ny*nyz + 5049*nxy*nzz*nzz*nz - 3267.0/2.0*nxy*nzz*nz + (729.0/2.0)*nxy*nz);  // power=8
        out[19] = T(1566*nxx*nxx*nxy*nz + 2538*nxx*nxy*ny*nyz - 6470999999999999.0/1000000000000.0*nxx*nxy*nzz*nz - 297*nxx*nxy*nz + 972*nxy*ny*nyy*nyz - 7695.0/2.0*nxy*ny*nyz*nz*nz - 891.0/2.0*nxy*ny*nyz + 4833*nxy*nzz*nzz*nz - 2673.0/2.0*nxy*nzz*nz + (567.0/2.0)*nxy*nz);  // power=8
        out[20] = T((9.0/2.0)*nxx*nxx*nxx*nx - 27*nxx*nxx*nxy*ny + (1930499999998069.0/999999999999.0)*nxx*nxx*nxz*nz - 135.0/2.0*nxx*nxx*nxz*nz - 297.0/2.0*nxx*nxx*nx - 135.0/2.0*nxx*nxy*ny*nyy + (1930499999998069.0/999999999999.0)*nxx*nxy*nyz*nz + 675*nxx*nxy*nyz*nz - 297.0/2.0*nxx*nxy*ny - 3217499999996782.0/999999999999.0*nxx*nxz*nz*nzz - 297.0/2.0*nxx*nxz*nz + 81*nxx*nx - 36*nxy*ny*nyy*nyy + (1485.0/2.0)*nxy*ny*nyy*nzz - 1080*nxy*nyz*nz*nzz - 891.0/2.0*nxy*nyz*nz + (81.0/2.0)*nxy*ny + 1287*nxz*nz*nzz*nzz + 72*nxz*nz*nzz*nzz - 1485.0/2.0*nxz*nz*nzz + 243*nxz*nz - 39.0/2.0*nx);  // power=8
        out[21] = T((1930499999998069.0/999999999999.0)*nxx*nyy*nyz*nz - 297.0/2.0*nxx*nyy*ny - 1930499999998069.0/999999999999.0*nxx*nyz*nz*nzz + (891.0/2.0)*nxx*nyz*nz + (1930499999998069.0/999999999999.0)*nyy*nyy*nyz*nz - 297.0/2.0*nyy*nyy*ny - 3217499999996782.0/999999999999.0*nyy*nyz*nz*nzz - 297.0/2.0*nyy*nyz*nz + 81*nyy*ny + 1287*nyz*nz*nzz*nzz - 594*nyz*nz*nzz + 162*nyz*nz - 15*ny);  // power=8
        out[22] = T((11583.0/4.0)*nxx*nxx*nyy*nz - 11583.0/4.0*nxx*nxx*nzz*nz + (891.0/2.0)*nxx*nxx*nz + (11583.0/4.0)*nxx*nyy*nyy*nz - 7722*nxx*nyy*nzz*nz - 891*nxx*nyy*nz + (11583.0/4.0)*nxx*nzz*nzz*nz - 297*nxx*nzz*nz - 81.0/2.0*nxx*nz - 11583.0/4.0*nyy*nyy*nzz*nz + (891.0/2.0)*nyy*nyy*nz + (11583.0/4.0)*nyy*nzz*nzz*nz - 297*nyy*nzz*nz - 81.0/2.0*nyy*nz - 1287.0/2.0*nzz*nzz*nzz*nz + 594*nzz*nzz*nz - 243*nzz*nz + 30*nz);  // power=8
        out[23] = T((11583.0/2.0)*nxx*nxx*nyz*nz - 891.0/2.0*nxx*nxx*ny + (11583.0/2.0)*nxx*nyy*nyz*nz - 891.0/2.0*nxx*nyy*ny - 11583.0/2.0*nxx*nyz*nz*nzz - 2673.0/2.0*nxx*nyz*nz + 243*nxx*ny - 1930499999998069.0/999999999999.0*nyy*nyz*nz*nzz + (891.0/2.0)*nyy*nyz*nz + 1287*nyz*nz*nzz*nzz - 594*nyz*nz*nzz + 162*nyz*nz - 15*ny);  // power=8
        out[24] = T((27.0/2.0)*nxx*nxx*nxx*nx - 81*nxx*nxx*nxy*ny - 405.0/2.0*nxx*nxx*nxz*nz - 567.0/2.0*nxx*nxy*ny*nyy - 405.0/2.0*nxx*nxy*ny*nyy + (10071.0/2.0)*nxx*nxy*nyz*nz + 2025*nxx*nxy*nyz*nz - 513.0/2.0*nxx*nxy*ny - 1930499999998069.0/999999999999.0*nxx*nxz*nz*nzz + (891.0/2.0)*nxx*nxz*nz - 27.0/2.0*nxx*nx - 567.0/2.0*nxy*ny*nyy*nyy - 108*nxy*ny*nyy*nyy + 4752*nxy*ny*nyy*nzz + (4455.0/2.0)*nxy*ny*nyy*nzz + 27*nxy*ny*nyy - 756*nxy*nyz*nz*nzz - 3240*nxy*nyz*nz*nzz - 6129.0/2.0*nxy*nyz*nz + 162*nxy*ny + 1287*nxz*nz*nzz*nzz + 216*nxz*nz*nzz*nzz - 2079.0/2.0*nxz*nz*nzz + (783.0/2.0)*nxz*nz - 15*nx);  // power=8
        out[25] = T(-63*nxx*nxx*nxz + (6363.0/2.0)*nxx*nxy*nyz + 63*nxx*nxz*nz*nz - 567.0/2.0*nxx*nxz + (3244499999996756.0/999999999999.0)*nxy*ny*nyy*nz - 1827*nxy*nyz*nz*nz - 4347.0/2.0*nxy*nyz + 126*nxz*nz*nzz*nz + 189*nxz);  // power=7
        out[26] = T((189.0/2.0)*nxx*nxx*nxz - 1323.0/2.0*nxx*nxy*nyz + 3024*nxx*nxz*nz*nz - 1701.0/2.0*nxx*nxz - 756*nxy*ny*nyy*nz + (4347.0/2.0)*nxy*nyz*nz*nz - 2268*nxz*nz*nzz*nz + (2079.0/2.0)*nxz*nz*nz - 63.0/2.0*nxz);  // power=7
        out[27] = T((6237.0/2.0)*nxx*nyz*nz*nz - 1701.0/2.0*nxx*nyz + (6237.0/2.0)*nyy*nyz*nz*nz - 1701.0/2.0*nyy*nyz - 2079*nyz*nz*nzz*nz + 756*nyz*nz*nz + 63*nyz);  // power=7
    } break;

    case 12: {
        out[0] = T(-60*nxx*ny + 15*nyy*ny + 15*nyz*nz);  // power=4
        out[1] = T(-300*nxx*nxy + 225*nxy*ny*ny + 225*nxy*nzz);  // power=5
        out[2] = T(-420*nxx*nyy + 60*nxx + 105*nyy*nyy + 105*nyy*nzz - 45*nyy - 15*nzz);  // power=5
        out[3] = T(-450*nxx*nyz + 75*nyy*nyz + 75*nyz*nz*nz);  // power=5
        out[4] = T(-420*nxx*nxx*ny + 315*nxx*nyy*ny + 315*nxx*nyz*nz + 140*nxx*ny - 105.0/2.0*nyy*nyy*ny - 105*nyy*nyz*nz + 35*nyy*ny - 105.0/2.0*nyz*nz*nzz + 35*nyz*nz - 20*ny);  // power=6
        out[5] = T(-210*nxx*nxx*ny - 315.0/2.0*nxx*nyy*ny + (945.0/2.0)*nxx*nyz*nz + 140*nxx*ny + (105.0/2.0)*nyy*nyy*ny - 105.0/2.0*nyy*nyz*nz - 35*nyy*ny - 105*nyz*nz*nzz + 35*nyz*nz - 5*ny);  // power=6
        out[6] = T(5*nxx*nxx*nx - 1285*nxx*nxy*ny - 25*nxx*nxz*nz + 140*nxx*nx + 285*nxy*ny*nyy + 255*nxy*nyz*nz + 210*nxy*ny - 30*nxz*nz*nzz - 45*nx);  // power=6
        out[7] = T(-1050*nxx*nxy*nz + 525*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[8] = T(-1260*nxx*nyy*nz + 140*nxx*nz + 315*nyy*nyy*nz + 315*nyy*nzz*nz - 175*nyy*nz - 35*nzz*nz + 10*nz);  // power=6
        out[9] = T(-1386*nxx*nxx*nxy + (693.0/2.0)*nxx*nxy*ny*ny + (693.0/2.0)*nxx*nxy*nzz + 1134*nxx*nxy - 189.0/2.0*nxy*ny*ny - 189.0/2.0*nxy*nzz - 168*nxy);  // power=7
        out[10] = T(-1386*nxx*nyy*nyy + 756*nxx*nyy - 42*nxx + (693.0/2.0)*nyy*nyy*nyy + (693.0/2.0)*nyy*nyy*nzz - 378*nyy*nyy - 189*nyy*nzz + (189.0/2.0)*nyy + (21.0/2.0)*nzz - 3);  // power=7
        out[11] = T(-1386*nxx*nyz*nz*nz + 378*nxx*nyz + (693.0/2.0)*nyy*nyz*nz*nz - 189.0/2.0*nyy*nyz + (693.0/2.0)*nyz*nz*nzz*nz - 567.0/2.0*nyz*nz*nz + 42*nyz);  // power=7
        out[12] = T(-4158*nxx*nxx*nyy + 378*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy + (2079.0/2.0)*nxx*nyy*nzz + (3213.0/2.0)*nxx*nyy - 189.0/2.0*nxx*nzz - 441.0/2.0*nxx - 189.0/2.0*nyy*nyy - 189.0/2.0*nyy*nzz - 63*nyy + 45.0/2.0);  // power=7
        out[13] = T(-4158*nxx*nxx*nyz + (2079.0/2.0)*nxx*nyy*nyz + (2079.0/2.0)*nxx*nyz*nz*nz + 1701*nxx*nyz - 189.0/2.0*nyy*nyz - 189.0/2.0*nyz*nz*nz - 63*nyz);  // power=7
        out[14] = T(-4158*nxx*nyy*nyz + 1134*nxx*nyz + (2079.0/2.0)*nyy*nyy*nyz + (2079.0/2.0)*nyy*nyz*nz*nz - 1701.0/2.0*nyy*nyz - 567.0/2.0*nyz*nz*nz + 126*nyz);  // power=7
        out[15] = T(63*nxx*nxx*nxz - 9009*nxx*nxy*nyz - 63*nxx*nxz*nz*nz + 756*nxx*nxz + 1323*nxy*ny*nyy*nz + 1197*nxy*nyz*nz*nz + 1386*nxy*nyz - 126*nxz*nz*nzz*nz - 189*nxz);  // power=7
        out[16] = T(-1287.0/2.0*nxx*nxx*nxx*ny + (1287.0/8.0)*nxx*nxx*nyy*ny + (32175.0/8.0)*nxx*nxx*nyz*nz + 396*nxx*nxx*ny - 3861.0/4.0*nxx*nyy*nyz*nz - 6435.0/4.0*nxx*nyz*nz*nzz - 891*nxx*nyz*nz - 135.0/2.0*nxx*ny + (1287.0/8.0)*nyy*nyz*nz*nzz + (1287.0/8.0)*nyz*nz*nzz*nzz - 99*nyz*nz*nzz + (135.0/2.0)*nyz*nz);  // power=8
        out[17] = T(-1287.0/2.0*nxx*nyy*nyy*ny + 3861*nxx*nyy*nyz*nz + 198*nxx*nyy*ny - 1287.0/2.0*nxx*nyz*nz*nzz - 594*nxx*nyz*nz + (1287.0/8.0)*nyy*nyy*nyy*ny - 6435.0/8.0*nyy*nyy*nyz*nz - 297.0/2.0*nyy*nyy*ny - 6435.0/8.0*nyy*nyz*nz*nzz + 693*nyy*nyz*nz + 27*nyy*ny + (1287.0/8.0)*nyz*nz*nzz*nzz + (99.0/2.0)*nyz*nz*nzz - 81*nyz*nz);  // power=8
        out[18] = T((9.0/2.0)*nxx*nxx*nxx*nx - 27*nxx*nxx*nxy*ny - 135.0/2.0*nxx*nxx*nxz*nz - 2511*nxx*nxy*ny*nyy - 135.0/2.0*nxx*nxy*ny*nyy + 7890*nxx*nxy*nyz*nz + 675*nxx*nxy*nyz*nz + 552*nxx*nxy*ny - 594*nxx*nxz*nz + 3*nxx*nx + (1413.0/2.0)*nxy*ny*nyy*nyy - 36*nxy*ny*nyy*nyy - 1056*nxy*ny*nyy*nzz + (1485.0/2.0)*nxy*ny*nyy*nzz - 309.0/2.0*nxy*ny*nyy + 168*nxy*nyz*nz*nzz - 1080*nxy*nyz*nz*nzz - 1398*nxy*nyz*nz + (63.0/2.0)*nxy*ny + 72*nxz*nz*nzz*nzz + 138*nxz*nz - 15.0/2.0*nx);  // power=8
        out[19] = T((9.0/2.0)*nxx*nxx*nxx*nx - 2574*nxx*nxx*nxy*ny - 27*nxx*nxx*nxy*ny - 135.0/2.0*nxx*nxx*nxz*nz + 198*nxx*nxx*nx + 738*nxx*nxy*ny*nyy - 135.0/2.0*nxx*nxy*ny*nyy + (17235.0/2.0)*nxx*nxy*nyz*nz + 675*nxx*nxy*nyz*nz + (976499999999023.0/999999999999.0)*nxx*nxy*ny - 1287.0/2.0*nxx*nxz*nz - 207.0/2.0*nxx*nx + (189.0/2.0)*nxy*ny*nyy*nyy - 36*nxy*ny*nyy*nyy - 1584*nxy*ny*nyy*nzz + (1485.0/2.0)*nxy*ny*nyy*nzz - 315.0/2.0*nxy*ny*nyy + 252*nxy*nyz*nz*nzz - 1080*nxy*nyz*nz*nzz - 2709.0/2.0*nxy*nyz*nz - 27*nxy*ny + 72*nxz*nz*nzz*nzz + (279.0/2.0)*nxz*nz + 6*nx);  // power=8
        out[20] = T(-1437336575412790.0/549755813889.0*nxx*nxx*nxy*nz + 711*nxx*nxy*ny*nyz + 3285*nxx*nxy*nzz*nz + 1089*nxx*nxy*nz + 108*nxy*ny*nyy*nyz - 855.0/2.0*nxy*ny*nyz*nz*nz - 297.0/2.0*nxy*ny*nyz - 1071.0/2.0*nxy*nzz*nzz*nz - 495.0/2.0*nxy*nzz*nz - 189.0/2.0*nxy*nz);  // power=8
        out[21] = T(-2574*nxx*nyy*nyy*nz + 2574*nxx*nyy*nzz*nz + 594*nxx*nyy*nz - 198*nxx*nzz*nz + (1287.0/2.0)*nyy*nyy*nyy*nz - 1089.0/2.0*nyy*nyy*nz - 1287.0/2.0*nyy*nzz*nzz*nz + 297*nyy*nzz*nz + 81*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz);  // power=8
        out[22] = T(-3861*nxx*nxx*nyy*ny + 3861*nxx*nxx*nyz*nz + 594*nxx*nxx*ny + (3861.0/4.0)*nxx*nyy*nyy*ny + 3861*nxx*nyy*nyz*nz + (1485.0/2.0)*nxx*nyy*ny - 1238187531831500.0/549755813889.0*nxx*nyz*nz*nzz - 3267.0/2.0*nxx*nyz*nz - 189*nxx*ny - 3861.0/4.0*nyy*nyy*nyz*nz - 1287.0/2.0*nyy*nyz*nz*nzz + (1485.0/2.0)*nyy*nyz*nz - 135.0/2.0*nyy*ny + (1287.0/4.0)*nyz*nz*nzz*nzz - 99.0/2.0*nyz*nz*nzz - 27.0/2.0*nyz*nz + 15*ny);  // power=8
        out[23] = T(-7722*nxx*nxx*nyy*nz + 594*nxx*nxx*nz + (3861.0/2.0)*nxx*nyy*nyy*nz + (2476375063663000.0/549755813889.0)*nxx*nyy*nzz*nz + (3267.0/2.0)*nxx*nyy*nz - 693.0/2.0*nxx*nzz*nz - 189*nxx*nz - 1287.0/2.0*nyy*nyy*nzz*nz - 1287.0/2.0*nyy*nzz*nzz*nz + (891.0/2.0)*nyy*nzz*nz - 135*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz + 15*nz);  // power=8
        out[24] = T(81*nxx*nxx*nxy*nz - 8127*nxx*nxy*ny*nyz + 2439*nxx*nxy*nzz*nz + 1188*nxx*nxy*nz + (2889.0/2.0)*nxy*ny*nyy*nyz + 585*nxy*ny*nyz*nz*nz + 567*nxy*ny*nyz - 1719.0/2.0*nxy*nzz*nzz*nz - 99*nxy*nzz*nz - 189*nxy*nz);  // power=8
        out[25] = T((63.0/2.0)*nxx*nxx*nxy - 8379.0/2.0*nxx*nxy*ny*ny - 441.0/2.0*nxx*nxy*nzz + 1134*nxx*nxy + (1953.0/2.0)*nxy*ny*nyy*ny + (1449.0/2.0)*nxy*ny*nyz*nz + 189*nxy*ny*ny - 252*nxy*nzz*nzz - 441.0/2.0*nxy);  // power=7
        out[26] = T((63.0/2.0)*nxx*nxx*nxy - 63.0/2.0*nxx*nxy*ny*ny - 8757.0/2.0*nxx*nxy*nzz + 378*nxx*nxy - 63*nxy*ny*nyy*ny + (1449.0/2.0)*nxy*ny*nyz*nz + (1575.0/2.0)*nxy*nzz*nzz + 567*nxy*nzz - 189.0/2.0*nxy);  // power=7
        out[27] = T(-4158*nxx*nyy*nzz + 378*nxx*nyy + 378*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz - 756*nyy*nzz + (189.0/2.0)*nyy - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
    } break;

    case 13: {
        out[0] = T(-60*nxx*nz + 15*nyy*nz + 15*nzz*nz);  // power=4
        out[1] = T(-300*nxx*nxz + 225*nxy*nyz + 225*nxz*nz*nz);  // power=5
        out[2] = T(-450*nxx*nyz + 75*nyy*nyz + 75*nyz*nz*nz);  // power=5
        out[3] = T(-420*nxx*nzz + 60*nxx + 105*nyy*nzz - 15*nyy + 105*nzz*nzz - 45*nzz);  // power=5
        out[4] = T(-420*nxx*nxx*nz + 315*nxx*nyy*nz + 315*nxx*nzz*nz + 140*nxx*nz - 105.0/2.0*nyy*nyy*nz - 105*nyy*nzz*nz + 35*nyy*nz - 105.0/2.0*nzz*nzz*nz + 35*nzz*nz - 20*nz);  // power=6
        out[5] = T(-210*nxx*nxx*nz - 315.0/2.0*nxx*nyy*nz + (945.0/2.0)*nxx*nzz*nz + (105.0/2.0)*nyy*nyy*nz - 105.0/2.0*nyy*nzz*nz - 105*nzz*nzz*nz + 70*nzz*nz - 15*nz);  // power=6
        out[6] = T(-1050*nxx*nxy*nz + 525*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[7] = T(5*nxx*nxx*nx - 25*nxx*nxy*ny - 1285*nxx*nxz*nz + 140*nxx*nx - 30*nxy*ny*nyy + 255*nxy*nyz*nz + 285*nxz*nz*nzz + 210*nxz*nz - 45*nx);  // power=6
        out[8] = T(-1260*nxx*nyz*nz + 140*nxx*ny + 315*nyy*nyz*nz - 35*nyy*ny + 315*nyz*nz*nzz - 175*nyz*nz + 10*ny);  // power=6
        out[9] = T(-1386*nxx*nxx*nxz + (693.0/2.0)*nxx*nxy*nyz + (693.0/2.0)*nxx*nxz*nz*nz + 1134*nxx*nxz - 189.0/2.0*nxy*nyz - 189.0/2.0*nxz*nz*nz - 168*nxz);  // power=7
        out[10] = T(-1386*nxx*nyy*nyz + 378*nxx*nyz + (693.0/2.0)*nyy*nyy*nyz + (693.0/2.0)*nyy*nyz*nz*nz - 567.0/2.0*nyy*nyz - 189.0/2.0*nyz*nz*nz + 42*nyz);  // power=7
        out[11] = T(-1386*nxx*nzz*nzz + 756*nxx*nzz - 42*nxx + (693.0/2.0)*nyy*nzz*nzz - 189*nyy*nzz + (21.0/2.0)*nyy + (693.0/2.0)*nzz*nzz*nzz - 378*nzz*nzz + (189.0/2.0)*nzz - 3);  // power=7
        out[12] = T(-4158*nxx*nxx*nyz + (2079.0/2.0)*nxx*nyy*nyz + (2079.0/2.0)*nxx*nyz*nz*nz + 1701*nxx*nyz - 189.0/2.0*nyy*nyz - 189.0/2.0*nyz*nz*nz - 63*nyz);  // power=7
        out[13] = T(-4158*nxx*nxx*nzz + 378*nxx*nxx + (2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz + (3213.0/2.0)*nxx*nzz - 441.0/2.0*nxx - 189.0/2.0*nyy*nzz - 189.0/2.0*nzz*nzz - 63*nzz + 45.0/2.0);  // power=7
        out[14] = T(-4158*nxx*nyy*nzz + 378*nxx*nyy + 378*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz - 756*nyy*nzz + (189.0/2.0)*nyy - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
        out[15] = T(63*nxx*nxx*nxy - 63*nxx*nxy*ny*ny - 8757*nxx*nxy*nzz + 756*nxx*nxy - 126*nxy*ny*nyy*ny + 1449*nxy*ny*nyz*nz + 1575*nxy*nzz*nzz + 1134*nxy*nzz - 189*nxy);  // power=7
        out[16] = T(-1287.0/2.0*nxx*nxx*nxx*nz + (1287.0/8.0)*nxx*nxx*nyy*nz + (32175.0/8.0)*nxx*nxx*nzz*nz - 198*nxx*nxx*nz - 3861.0/4.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz - 6435.0/4.0*nxx*nzz*nzz*nz - 1089.0/2.0*nxx*nzz*nz + (243.0/2.0)*nxx*nz + (1287.0/8.0)*nyy*nzz*nzz*nz - 99.0/2.0*nyy*nzz*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[17] = T(-1287.0/2.0*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz - 594*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + 198*nxx*nzz*nz + (1287.0/8.0)*nyy*nyy*nyy*nz - 6435.0/8.0*nyy*nyy*nzz*nz + (99.0/2.0)*nyy*nyy*nz - 6435.0/8.0*nyy*nzz*nzz*nz + 693*nyy*nzz*nz - 81*nyy*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + 27*nzz*nz);  // power=8
        out[18] = T(-81*nxx*nxx*nxy*nz - 2529*nxx*nxy*ny*nyz + 7857*nxx*nxy*nzz*nz - 1188*nxx*nxy*nz + (1539.0/2.0)*nxy*ny*nyy*nyz - 945*nxy*ny*nyz*nz*nz + 189*nxy*ny*nyz - 3429.0/2.0*nxy*nzz*nzz*nz - 297*nxy*nzz*nz + 189*nxy*nz);  // power=8
        out[19] = T(-2695499999997304.0/999999999999.0*nxx*nxx*nxy*nz + 846*nxx*nxy*ny*nyz + 8568*nxx*nxy*nzz*nz - 99*nxx*nxy*nz + 324*nxy*ny*nyy*nyz - 2565.0/2.0*nxy*ny*nyz*nz*nz - 297.0/2.0*nxy*ny*nyz - 3213.0/2.0*nxy*nzz*nzz*nz - 891.0/2.0*nxy*nzz*nz + (189.0/2.0)*nxy*nz);  // power=8
        out[20] = T((3.0/2.0)*nxx*nxx*nxx*nx - 9*nxx*nxx*nxy*ny - 2574*nxx*nxx*nxz*nz - 45.0/2.0*nxx*nxx*nxz*nz + 198*nxx*nxx*nx - 45.0/2.0*nxx*nxy*ny*nyy + (1287.0/2.0)*nxx*nxy*nyz*nz + 225*nxx*nxy*nyz*nz - 99.0/2.0*nxx*nxy*ny + (3217499999996782.0/999999999999.0)*nxx*nxz*nz*nzz + (445499999999554.0/999999999999.0)*nxx*nxz*nz - 108*nxx*nx - 12*nxy*ny*nyy*nyy + (495.0/2.0)*nxy*ny*nyy*nzz - 360*nxy*nyz*nz*nzz - 297.0/2.0*nxy*nyz*nz + (27.0/2.0)*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz + 24*nxz*nz*nzz*nzz - 54*nxz*nz + (27.0/2.0)*nx);  // power=8
        out[21] = T(-2574*nxx*nyy*nyz*nz + 198*nxx*nyy*ny + 2574*nxx*nyz*nz*nzz - 594*nxx*nyz*nz + (1287.0/2.0)*nyy*nyy*nyz*nz - 99.0/2.0*nyy*nyy*ny - 297*nyy*nyz*nz + 27*nyy*ny - 1287.0/2.0*nyz*nz*nzz*nzz + (1089.0/2.0)*nyz*nz*nzz - 81*nyz*nz);  // power=8
        out[22] = T(-3861*nxx*nxx*nyy*nz + 3861*nxx*nxx*nzz*nz - 594*nxx*nxx*nz + (3861.0/4.0)*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz - 1238187531831500.0/549755813889.0*nxx*nzz*nzz*nz - 693.0/2.0*nxx*nzz*nz + 189*nxx*nz - 3861.0/4.0*nyy*nyy*nzz*nz + (297.0/2.0)*nyy*nyy*nz - 1287.0/2.0*nyy*nzz*nzz*nz + (1287.0/2.0)*nyy*nzz*nz - 297.0/2.0*nyy*nz + (1287.0/4.0)*nzz*nzz*nzz*nz - 297*nzz*nzz*nz + (243.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[23] = T(-7722*nxx*nxx*nyz*nz + 594*nxx*nxx*ny + (3861.0/2.0)*nxx*nyy*nyz*nz - 297.0/2.0*nxx*nyy*ny + (2476375063663000.0/549755813889.0)*nxx*nyz*nz*nzz + (2079.0/2.0)*nxx*nyz*nz - 189*nxx*ny - 1287.0/2.0*nyy*nyz*nz*nzz + (297.0/2.0)*nyy*nyz*nz - 1287.0/2.0*nyz*nz*nzz*nzz + (1089.0/2.0)*nyz*nz*nzz - 216*nyz*nz + 15*ny);  // power=8
        out[24] = T(-3*nxx*nxx*nxx*nx + 18*nxx*nxx*nxy*ny + 45*nxx*nxx*nxz*nz - 189.0/2.0*nxx*nxy*ny*nyy + 45*nxx*nxy*ny*nyy - 7974*nxx*nxy*nyz*nz - 450*nxx*nxy*nyz*nz + 657*nxx*nxy*ny + 2574*nxx*nxz*nz*nzz - 594*nxx*nxz*nz - 9.0/2.0*nxx*nx - 189.0/2.0*nxy*ny*nyy*nyy + 24*nxy*ny*nyy*nyy + 1584*nxy*ny*nyy*nzz - 495*nxy*ny*nyy*nzz + 9*nxy*ny*nyy - 252*nxy*nyz*nz*nzz + 720*nxy*nyz*nz*nzz + 1206*nxy*nyz*nz - 297.0/2.0*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz - 48*nxz*nz*nzz*nzz + (297.0/2.0)*nxz*nz*nzz - 9.0/2.0*nxz*nz + (15.0/2.0)*nx);  // power=8
        out[25] = T((63.0/2.0)*nxx*nxx*nxz - 9009.0/2.0*nxx*nxy*nyz - 63.0/2.0*nxx*nxz*nz*nz + 378*nxx*nxz + (1323.0/2.0)*nxy*ny*nyy*nz + (1197.0/2.0)*nxy*nyz*nz*nz + 693*nxy*nyz - 63*nxz*nz*nzz*nz - 189.0/2.0*nxz);  // power=7
        out[26] = T((63.0/2.0)*nxx*nxx*nxz - 441.0/2.0*nxx*nxy*nyz - 8379.0/2.0*nxx*nxz*nz*nz + 1134*nxx*nxz - 252*nxy*ny*nyy*nz + (1449.0/2.0)*nxy*nyz*nz*nz + (1953.0/2.0)*nxz*nz*nzz*nz + 189*nxz*nz*nz - 441.0/2.0*nxz);  // power=7
        out[27] = T(-4158*nxx*nyz*nz*nz + 1134*nxx*nyz + (2079.0/2.0)*nyy*nyz*nz*nz - 567.0/2.0*nyy*nyz + (2079.0/2.0)*nyz*nz*nzz*nz - 1701.0/2.0*nyz*nz*nz + 126*nyz);  // power=7
    } break;

    case 14: {
        out[0] = T(15*nxx*nz - 60*nyy*nz + 15*nzz*nz);  // power=4
        out[1] = T(75*nxx*nxz - 450*nxy*nyz + 75*nxz*nz*nz);  // power=5
        out[2] = T(225*nxx*nyz - 300*nyy*nyz + 225*nyz*nz*nz);  // power=5
        out[3] = T(105*nxx*nzz - 15*nxx - 420*nyy*nzz + 60*nyy + 105*nzz*nzz - 45*nzz);  // power=5
        out[4] = T(105*nxx*nxx*nz - 945.0/2.0*nxx*nyy*nz + (105.0/2.0)*nxx*nzz*nz - 35*nxx*nz + 210*nyy*nyy*nz + (315.0/2.0)*nyy*nzz*nz - 140*nyy*nz - 105.0/2.0*nzz*nzz*nz + 35*nzz*nz + 5*nz);  // power=6
        out[5] = T((105.0/2.0)*nxx*nxx*nz - 315.0/2.0*nxx*nyy*nz - 105.0/2.0*nxx*nzz*nz - 210*nyy*nyy*nz + (945.0/2.0)*nyy*nzz*nz - 105*nzz*nzz*nz + 70*nzz*nz - 15*nz);  // power=6
        out[6] = T(525*nxx*nxy*nz - 1050*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[7] = T(-20*nxx*nxx*nx + 100*nxx*nxy*ny + 415*nxx*nxz*nz - 35*nxx*nx + 120*nxy*ny*nyy - 1020*nxy*nyz*nz + 435*nxz*nz*nzz - 315*nxz*nz + 30*nx);  // power=6
        out[8] = T(315*nxx*nyz*nz - 35*nxx*ny - 1260*nyy*nyz*nz + 140*nyy*ny + 315*nyz*nz*nzz + 175*nyz*nz - 40*ny);  // power=6
        out[9] = T((693.0/2.0)*nxx*nxx*nxz - 1386*nxx*nxy*nyz + (693.0/2.0)*nxx*nxz*nz*nz - 567.0/2.0*nxx*nxz + 378*nxy*nyz - 189.0/2.0*nxz*nz*nz + 42*nxz);  // power=7
        out[10] = T((693.0/2.0)*nxx*nyy*nyz - 189.0/2.0*nxx*nyz - 1386*nyy*nyy*nyz + (693.0/2.0)*nyy*nyz*nz*nz + 1134*nyy*nyz - 189.0/2.0*nyz*nz*nz - 168*nyz);  // power=7
        out[11] = T((693.0/2.0)*nxx*nzz*nzz - 189*nxx*nzz + (21.0/2.0)*nxx - 1386*nyy*nzz*nzz + 756*nyy*nzz - 42*nyy + (693.0/2.0)*nzz*nzz*nzz - 378*nzz*nzz + (189.0/2.0)*nzz - 3);  // power=7
        out[12] = T((2079.0/2.0)*nxx*nxx*nyz - 4158*nxx*nyy*nyz + (2079.0/2.0)*nxx*nyz*nz*nz + (567.0/2.0)*nxx*nyz + 378*nyy*nyz - 189.0/2.0*nyz*nz*nz - 63*nyz);  // power=7
        out[13] = T((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nzz + 378*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz - 756*nxx*nzz + (189.0/2.0)*nxx + 378*nyy*nzz - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
        out[14] = T((2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 189.0/2.0*nxx*nzz - 4158*nyy*nyy*nzz + 378*nyy*nyy + (2079.0/2.0)*nyy*nzz*nzz + (3213.0/2.0)*nyy*nzz - 441.0/2.0*nyy - 189.0/2.0*nzz*nzz - 63*nzz + 45.0/2.0);  // power=7
        out[15] = T(-252*nxx*nxx*nxy + 252*nxx*nxy*ny*ny + 3843*nxx*nxy*nzz - 189*nxx*nxy + 504*nxy*ny*nyy*ny - 5796*nxy*ny*nyz*nz + 4095*nxy*nzz*nzz - 1701*nxy*nzz + 126*nxy);  // power=7
        out[16] = T((1287.0/8.0)*nxx*nxx*nxx*nz - 1287.0/2.0*nxx*nxx*nyy*nz - 6435.0/8.0*nxx*nxx*nzz*nz + (99.0/2.0)*nxx*nxx*nz + 3861*nxx*nyy*nzz*nz - 594*nxx*nyy*nz - 6435.0/8.0*nxx*nzz*nzz*nz + 693*nxx*nzz*nz - 81*nxx*nz - 1287.0/2.0*nyy*nzz*nzz*nz + 198*nyy*nzz*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + 27*nzz*nz);  // power=8
        out[17] = T((1287.0/8.0)*nxx*nyy*nyy*nz - 3861.0/4.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + (1287.0/8.0)*nxx*nzz*nzz*nz - 99.0/2.0*nxx*nzz*nz - 1287.0/2.0*nyy*nyy*nyy*nz + (32175.0/8.0)*nyy*nyy*nzz*nz - 198*nyy*nyy*nz - 6435.0/4.0*nyy*nzz*nzz*nz - 1089.0/2.0*nyy*nzz*nz + (243.0/2.0)*nyy*nz + (1287.0/8.0)*nzz*nzz*nzz*nz - 297.0/2.0*nzz*nzz*nz + (189.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[18] = T((1053.0/2.0)*nxx*nxx*nxy*nz + 126*nxx*nxy*ny*nyz - 2808*nxx*nxy*nzz*nz + 297*nxx*nxy*nz - 3618*nxy*ny*nyy*nyz + (11835.0/2.0)*nxy*ny*nyz*nz*nz + (2943.0/2.0)*nxy*ny*nyz - 3334499999996666.0/999999999999.0*nxy*nzz*nzz*nz + (3861.0/2.0)*nxy*nzz*nz - 1107.0/2.0*nxy*nz);  // power=8
        out[19] = T((2259.0/2.0)*nxx*nxx*nxy*nz - 3384*nxx*nxy*ny*nyz - 2097*nxx*nxy*nzz*nz + 396*nxx*nxy*nz - 1296*nxy*ny*nyy*nyz + 5130*nxy*ny*nyz*nz*nz + 594*nxy*ny*nyz - 6453.0/2.0*nxy*nzz*nzz*nz + 1782*nxy*nzz*nz - 378*nxy*nz);  // power=8
        out[20] = T(-6*nxx*nxx*nxx*nx + 36*nxx*nxx*nxy*ny + (1287.0/2.0)*nxx*nxx*nxz*nz + 90*nxx*nxx*nxz*nz - 99.0/2.0*nxx*nxx*nx + 90*nxx*nxy*ny*nyy - 2574*nxx*nxy*nyz*nz - 900*nxx*nxy*nyz*nz + 198*nxx*nxy*ny - 297*nxx*nxz*nz + 27*nxx*nx + 48*nxy*ny*nyy*nyy - 990*nxy*ny*nyy*nzz + 1440*nxy*nyz*nz*nzz + 594*nxy*nyz*nz - 54*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz - 96*nxz*nz*nzz*nzz + (1485.0/2.0)*nxz*nz*nzz - 189*nxz*nz + 6*nx);  // power=8
        out[21] = T((1287.0/2.0)*nxx*nyy*nyz*nz - 99.0/2.0*nxx*nyy*ny - 1287.0/2.0*nxx*nyz*nz*nzz + (297.0/2.0)*nxx*nyz*nz - 2574*nyy*nyy*nyz*nz + 198*nyy*nyy*ny + (3217499999996782.0/999999999999.0)*nyy*nyz*nz*nzz + (445499999999554.0/999999999999.0)*nyy*nyz*nz - 108*nyy*ny - 1287.0/2.0*nyz*nz*nzz*nzz + (99.0/2.0)*nyz*nz*nzz - 81*nyz*nz + 15*ny);  // power=8
        out[22] = T((3861.0/4.0)*nxx*nxx*nyy*nz - 3861.0/4.0*nxx*nxx*nzz*nz + (297.0/2.0)*nxx*nxx*nz - 3861*nxx*nyy*nyy*nz + 3861*nxx*nyy*nzz*nz + (891.0/2.0)*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + (1287.0/2.0)*nxx*nzz*nz - 297.0/2.0*nxx*nz + 3861*nyy*nyy*nzz*nz - 594*nyy*nyy*nz - 1238187531831500.0/549755813889.0*nyy*nzz*nzz*nz - 693.0/2.0*nyy*nzz*nz + 189*nyy*nz + (1287.0/4.0)*nzz*nzz*nzz*nz - 297*nzz*nzz*nz + (243.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[23] = T((3861.0/2.0)*nxx*nxx*nyz*nz - 297.0/2.0*nxx*nxx*ny - 7722*nxx*nyy*nyz*nz + 594*nxx*nyy*ny + 1287*nxx*nyz*nz*nzz + 297*nxx*nyz*nz - 54*nxx*ny + 2574*nyy*nyz*nz*nzz - 594*nyy*nyz*nz - 1287.0/2.0*nyz*nz*nzz*nzz + (99.0/2.0)*nyz*nz*nzz + 54*nyz*nz);  // power=8
        out[24] = T(-21.0/2.0*nxx*nxx*nxx*nx + 63*nxx*nxx*nxy*ny + (315.0/2.0)*nxx*nxx*nxz*nz + 378*nxx*nxy*ny*nyy + (315.0/2.0)*nxx*nxy*ny*nyy + (5877.0/2.0)*nxx*nxy*nyz*nz - 1575*nxx*nxy*nyz*nz - 801.0/2.0*nxx*nxy*ny - 1287.0/2.0*nxx*nxz*nz*nzz + (297.0/2.0)*nxx*nxz*nz + 18*nxx*nx + 378*nxy*ny*nyy*nyy + 84*nxy*ny*nyy*nyy - 6336*nxy*ny*nyy*nzz - 3465.0/2.0*nxy*ny*nyy*nzz - 36*nxy*ny*nyy + 1008*nxy*nyz*nz*nzz + 2520*nxy*nyz*nz*nzz + (3717.0/2.0)*nxy*nyz*nz - 27.0/2.0*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz - 168*nxz*nz*nzz*nzz + 891*nxz*nz*nzz - 387*nxz*nz + (15.0/2.0)*nx);  // power=8
        out[25] = T((63.0/2.0)*nxx*nxx*nxz + 1323*nxx*nxy*nyz - 63.0/2.0*nxx*nxz*nz*nz - 189.0/2.0*nxx*nxz - 3906*nxy*ny*nyy*nz + (2457.0/2.0)*nxy*nyz*nz*nz + (2961.0/2.0)*nxy*nyz - 63*nxz*nz*nzz*nz - 189.0/2.0*nxz);  // power=7
        out[26] = T(-126*nxx*nxx*nxz + 882*nxx*nxy*nyz + (2331.0/2.0)*nxx*nxz*nz*nz - 567.0/2.0*nxx*nxz + 1008*nxy*ny*nyy*nz - 2898*nxy*nyz*nz*nz + (2583.0/2.0)*nxz*nz*nzz*nz - 2457.0/2.0*nxz*nz*nz + 252*nxz);  // power=7
        out[27] = T((2079.0/2.0)*nxx*nyz*nz*nz - 567.0/2.0*nxx*nyz - 4158*nyy*nyz*nz*nz + 1134*nyy*nyz + (2079.0/2.0)*nyz*nz*nzz*nz + (189.0/2.0)*nyz*nz*nz - 189*nyz);  // power=7
    } break;

    case 15: {
        out[0] = T(-75*nxy*nz);  // power=4
        out[1] = T(-450*nxx*nyz + 75*nyy*nyz + 75*nyz*nz*nz);  // power=5
        out[2] = T(75*nxx*nxz - 450*nxy*nyz + 75*nxz*nz*nz);  // power=5
        out[3] = T(75*nxx*nxy + 75*nxy*ny*ny - 450*nxy*nzz);  // power=5
        out[4] = T(-525*nxx*nxy*nz + (525.0/2.0)*nxy*ny*nyz + (525.0/2.0)*nxy*nzz*nz);  // power=6
        out[5] = T(-525.0/2.0*nxx*nxy*nz - 525.0/2.0*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[6] = T(-1575*nxx*nyy*nz + 175*nxx*nz + 175*nyy*nz - 25*nz);  // power=6
        out[7] = T(-1575*nxx*nyz*nz + 175*nxx*ny + 175*nyz*nz - 25*ny);  // power=6
        out[8] = T(-25*nxx*nxx*nx + 125*nxx*nxy*ny + 125*nxx*nxz*nz + 150*nxy*ny*nyy - 1275*nxy*nyz*nz + 150*nxz*nz*nzz);  // power=6
        out[9] = T(-3465.0/2.0*nxx*nxx*nyz + 945*nxx*nyz - 105.0/2.0*nyz);  // power=7
        out[10] = T(210*nxx*nxy*nyz - 3045.0/2.0*nxy*ny*nyy*nz + 210*nxy*nyz*nz*nz + 735*nxy*nyz - 105.0/2.0*nxz);  // power=7
        out[11] = T(-3465.0/2.0*nxy*nzz*nzz + 945*nxy*nzz - 105.0/2.0*nxy);  // power=7
        out[12] = T(-10395.0/2.0*nxx*nxy*nyz + (945.0/2.0)*nxx*nxz + (2835.0/2.0)*nxy*nyz - 315.0/2.0*nxz);  // power=7
        out[13] = T(-10395.0/2.0*nxx*nxy*nzz + (945.0/2.0)*nxx*nxy + (2835.0/2.0)*nxy*nzz - 315.0/2.0*nxy);  // power=7
        out[14] = T(-315.0/2.0*nxx*nxx*nxy + (315.0/2.0)*nxx*nxy*ny*ny + (2205.0/2.0)*nxx*nxy*nzz + 315*nxy*ny*nyy*ny - 7245.0/2.0*nxy*ny*nyz*nz + 1260*nxy*nzz*nzz);  // power=7
        out[15] = T(-10395*nxx*nyy*nzz + 945*nxx*nyy + 945*nxx*nzz + 945*nyy*nzz - 90);  // power=7
        out[16] = T(-804374999995978.0/999999999995.0*nxx*nxx*nxy*nz + (19305.0/4.0)*nxx*nxy*nzz*nz - 495*nxx*nxy*nz - 804374999995978.0/999999999995.0*nxy*nzz*nzz*nz - 495*nxy*nzz*nz + 135*nxy*nz);  // power=8
        out[17] = T((1215.0/4.0)*nxx*nxx*nxy*nz - 1575.0/4.0*nxx*nxy*ny*nyz - 2025.0/4.0*nxx*nxy*nzz*nz - 12015.0/8.0*nxy*ny*nyy*nyz + (13275.0/4.0)*nxy*ny*nyz*nz*nz + (2025.0/4.0)*nxy*ny*nyz - 12915.0/8.0*nxy*nzz*nzz*nz + (2475.0/4.0)*nxy*nzz*nz - 675.0/4.0*nxy*nz);  // power=8
        out[18] = T(-3217499999996782.0/999999999999.0*nxx*nyy*nyy*nz + (19305.0/2.0)*nxx*nyy*nzz*nz - 1485.0/2.0*nxx*nyy*nz - 1485.0/2.0*nxx*nzz*nz + 135*nxx*nz + (495.0/2.0)*nyy*nyy*nz - 1485.0/2.0*nyy*nzz*nz + (135.0/2.0)*nyy*nz + (135.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[19] = T(-3217499999996782.0/999999999999.0*nxx*nxx*nyy*nz + (495.0/2.0)*nxx*nxx*nz + (19305.0/2.0)*nxx*nyy*nzz*nz - 1485.0/2.0*nxx*nyy*nz - 1485.0/2.0*nxx*nzz*nz + (135.0/2.0)*nxx*nz - 1485.0/2.0*nyy*nzz*nz + 135*nyy*nz + (135.0/2.0)*nzz*nz - 15*nz);  // power=8
        out[20] = T(-3217499999996782.0/999999999999.0*nxx*nxx*nyz*nz + (495.0/2.0)*nxx*nxx*ny + (3217499999996782.0/999999999999.0)*nxx*nyz*nz*nzz - 135.0/2.0*nxx*ny - 495.0/2.0*nyz*nz*nzz + (135.0/2.0)*nyz*nz);  // power=8
        out[21] = T(-15.0/2.0*nxx*nxx*nxx*nx + 45*nxx*nxx*nxy*ny + (225.0/2.0)*nxx*nxx*nxz*nz + (315.0/2.0)*nxx*nxy*ny*nyy + (225.0/2.0)*nxx*nxy*ny*nyy + 420*nxx*nxy*nyz*nz - 1125*nxx*nxy*nyz*nz - 105*nxx*nxy*ny + (15.0/2.0)*nxx*nx + (315.0/2.0)*nxy*ny*nyy*nyy + 60*nxy*ny*nyy*nyy - 2640*nxy*ny*nyy*nzz - 2475.0/2.0*nxy*ny*nyy*nzz - 15*nxy*ny*nyy + 420*nxy*nyz*nz*nzz + 1800*nxy*nyz*nz*nzz + 960*nxy*nyz*nz - 45.0/2.0*nxy*ny - 120*nxz*nz*nzz*nzz - 60*nxz*nz);  // power=8
        out[22] = T((1215.0/4.0)*nxx*nxx*nxy*nz - 10665.0/2.0*nxx*nxy*ny*nyz + 4320*nxx*nxy*nzz*nz - 810*nxy*ny*nyy*nyz + (12825.0/4.0)*nxy*ny*nyz*nz*nz + (4455.0/4.0)*nxy*ny*nyz - 1329721874839181.0/549755813887.0*nxy*nzz*nzz*nz + (123749999999876.0/999999999999.0)*nxy*nzz*nz - 675.0/4.0*nxy*nz);  // power=8
        out[23] = T(-15.0/2.0*nxx*nxx*nxx*nx + 45*nxx*nxx*nxy*ny + (225.0/2.0)*nxx*nxx*nxz*nz + (225.0/2.0)*nxx*nxy*ny*nyy - 19305.0/2.0*nxx*nxy*nyz*nz - 1125*nxx*nxy*nyz*nz + (1485.0/2.0)*nxx*nxy*ny + (1485.0/2.0)*nxx*nxz*nz - 135.0/2.0*nxx*nx + 60*nxy*ny*nyy*nyy - 2475.0/2.0*nxy*ny*nyy*nzz + 1800*nxy*nyz*nz*nzz + (4455.0/2.0)*nxy*nyz*nz - 405.0/2.0*nxy*ny - 120*nxz*nz*nzz*nzz - 405.0/2.0*nxz*nz + (45.0/2.0)*nx);  // power=8
        out[24] = T(-19305.0/2.0*nxx*nyy*nyz*nz + (1485.0/2.0)*nxx*nyy*ny + (3217499999996782.0/999999999999.0)*nxx*nyz*nz*nzz + (1485.0/2.0)*nxx*nyz*nz - 135*nxx*ny + (1485.0/2.0)*nyy*nyz*nz - 135.0/2.0*nyy*ny - 495.0/2.0*nyz*nz*nzz - 135.0/2.0*nyz*nz + 15*ny);  // power=8
        out[25] = T(-10395.0/2.0*nxx*nyy*nyz + (2835.0/2.0)*nxx*nyz + (945.0/2.0)*nyy*nyz - 315.0/2.0*nyz);  // power=7
        out[26] = T(-10395.0/2.0*nxx*nyz*nz*nz + (2835.0/2.0)*nxx*nyz + (945.0/2.0)*nyz*nz*nz - 315.0/2.0*nyz);  // power=7
        out[27] = T(-315.0/2.0*nxx*nxx*nxz + (2205.0/2.0)*nxx*nxy*nyz + (315.0/2.0)*nxx*nxz*nz*nz + 1260*nxy*ny*nyy*nz - 7245.0/2.0*nxy*nyz*nz*nz + 315*nxz*nz*nzz*nz);  // power=7
    } break;

    case 16: {
        out[0] = T(525*nxx*nxx - 3150*nxx*nyy - 3150*nxx*nzz + 450*nxx + 450*nyy + 450*nzz - 135);  // power=5
        out[1] = T(4725*nxx*nxx*nx - 28350*nxx*nxy*ny - 28350*nxx*nxz*nz + 1050*nxx*nx + 9450*nxy*ny + 9450*nxz*nz - 1575*nx);  // power=6
        out[2] = T(4725*nxx*nxx*ny - 28350*nxx*nyy*ny - 28350*nxx*nyz*nz + 9450*nxx*ny + 3150*nyy*ny + 3150*nyz*nz - 1575*ny);  // power=6
        out[3] = T(4725*nxx*nxx*nz - 28350*nxx*nyy*nz - 28350*nxx*nzz*nz + 9450*nxx*nz + 3150*nyy*nz + 3150*nzz*nz - 1575*nz);  // power=6
        out[4] = T(5775*nxx*nxx*nxx - 75075.0/2.0*nxx*nxx*nyy - 75075.0/2.0*nxx*nxx*nzz - 1050*nxx*nxx + 17325*nxx*nyy*nyy + 34650*nxx*nyy*nzz + 7875*nxx*nyy + 17325*nxx*nzz*nzz + 7875*nxx*nzz - 875*nxx - 1575*nyy*nyy - 3150*nyy*nzz - 175.0/2.0*nyy - 1575*nzz*nzz - 175.0/2.0*nzz + 350);  // power=7
        out[5] = T((5775.0/2.0)*nxx*nxx*nxx - 28875.0/2.0*nxx*nxx*nyy - 23100*nxx*nxx*nzz - 525*nxx*nxx - 17325*nxx*nyy*nyy + 17325*nxx*nyy*nzz + 15750*nxx*nyy + 34650*nxx*nzz*nzz - 7875*nxx*nzz - 875.0/2.0*nxx + 1575*nyy*nyy - 1575*nyy*nzz - 2975.0/2.0*nyy - 3150*nzz*nzz + 1400*nzz + 175);  // power=7
        out[6] = T(17325*nxx*nxx*nxy - 103950*nxx*nxy*ny*ny - 103950*nxx*nxy*nzz + 22050*nxx*nxy + 28350*nxy*ny*ny + 28350*nxy*nzz - 9975*nxy);  // power=7
        out[7] = T(17325*nxx*nxx*nxz - 103950*nxx*nxy*nyz - 103950*nxx*nxz*nz*nz + 22050*nxx*nxz + 28350*nxy*nyz + 28350*nxz*nz*nz - 9975*nxz);  // power=7
        out[8] = T(17325*nxx*nxx*nyz - 103950*nxx*nyy*nyz - 103950*nxx*nyz*nz*nz + 47250*nxx*nyz + 9450*nyy*nyz + 9450*nyz*nz*nz - 5775*nyz);  // power=7
        out[9] = T((45045.0/2.0)*nxx*nxx*nxx*nx - 135135*nxx*nxx*nxy*ny - 135135*nxx*nxx*nxz*nz - 31185.0/2.0*nxx*nxx*nx + 103950*nxx*nxy*ny + 103950*nxx*nxz*nz - 4725.0/2.0*nxx*nx - 14175*nxy*ny - 14175*nxz*nz + (2625.0/2.0)*nx);  // power=8
        out[10] = T((45045.0/2.0)*nxx*nxx*nyy*ny - 10395.0/2.0*nxx*nxx*ny - 135135*nxx*nyy*nyy*ny - 135135*nxx*nyy*nyz*nz + 103950*nxx*nyy*ny + 31185*nxx*nyz*nz - 14175*nxx*ny + 10395*nyy*nyy*ny + 10395*nyy*nyz*nz - 19845.0/2.0*nyy*ny - 2835*nyz*nz + (3465.0/2.0)*ny);  // power=8
        out[11] = T((45045.0/2.0)*nxx*nxx*nzz*nz - 10395.0/2.0*nxx*nxx*nz - 135135*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz - 135135*nxx*nzz*nzz*nz + 103950*nxx*nzz*nz - 14175*nxx*nz + 10395*nyy*nzz*nz - 2835*nyy*nz + 10395*nzz*nzz*nz - 19845.0/2.0*nzz*nz + (3465.0/2.0)*nz);  // power=8
        out[12] = T((135135.0/2.0)*nxx*nxx*nxx*ny - 405405*nxx*nxx*nyy*ny - 405405*nxx*nxx*nyz*nz + (93555.0/2.0)*nxx*nxx*ny + 187110*nxx*nyy*ny + 187110*nxx*nyz*nz - 93555.0/2.0*nxx*ny - 8505*nyy*ny - 8505*nyz*nz + (5985.0/2.0)*ny);  // power=8
        out[13] = T((135135.0/2.0)*nxx*nxx*nxx*nz - 405405*nxx*nxx*nyy*nz - 405405*nxx*nxx*nzz*nz + (93555.0/2.0)*nxx*nxx*nz + 187110*nxx*nyy*nz + 187110*nxx*nzz*nz - 93555.0/2.0*nxx*nz - 8505*nyy*nz - 8505*nzz*nz + (5985.0/2.0)*nz);  // power=8
        out[14] = T((135135.0/2.0)*nxx*nxx*nyy*nz - 10395.0/2.0*nxx*nxx*nz - 405405*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 249480*nxx*nyy*nz + 31185*nxx*nzz*nz - 14175*nxx*nz + 31185*nyy*nyy*nz + 31185*nyy*nzz*nz - 48195.0/2.0*nyy*nz - 2835*nzz*nz + (3465.0/2.0)*nz);  // power=8
        out[15] = T(135135*nxx*nxx*nxy*nz - 810810*nxx*nxy*ny*nyz - 810810*nxx*nxy*nzz*nz + 270270*nxx*nxy*nz + 187110*nxy*ny*nyz + 187110*nxy*nzz*nz - 87885*nxy*nz);  // power=8
        out[16] = T((6633147491942405.0/549755813888.0)*nxx*nxx*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy - 2487430309478401.0/17179869184.0*nxx*nxx*nxx*nzz - 2211049163980803.0/274877906944.0*nxx*nxx*nxx + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (173745.0/4.0)*nxx*nxx*nyy + (7669576787558399.0/17179869184.0)*nxx*nxx*nzz*nzz - 1077886467440639.0/17179869184.0*nxx*nxx*nzz + (1479702132817919.0/274877906944.0)*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nyy*nzz*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nzz - 4455*nxx*nyy - 4974860618956799.0/68719476736.0*nxx*nzz*nzz*nzz - 405405.0/4.0*nxx*nzz*nzz + (2984916371374079.0/68719476736.0)*nxx*nzz - 2159999999999999.0/1000000000000.0*nxx + (3979888495160613.0/824633720831.0)*nyy*nzz*nzz + 4455*nyy*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz + (892923700839329.0/962072674305.0)*nzz*nzz - 1620*nzz + 105);  // power=9
        out[17] = T((6633147491942405.0/549755813888.0)*nxx*nxx*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nyy*nzz + (1.0/1000000000000.0)*nxx*nxx*nyy + (6633147491942405.0/549755813888.0)*nxx*nxx*nzz*nzz + (1.0/1000000000000.0)*nxx*nxx*nzz - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nyy*nzz + (173745.0/4.0)*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 4974860618956799.0/68719476736.0*nxx*nzz*nzz*nzz + (173745.0/4.0)*nxx*nzz*nzz + (3979888495160613.0/824633720831.0)*nyy*nyy*nyy - 96525.0/4.0*nyy*nyy*nzz - 3393110063189446.0/962072674305.0*nyy*nyy - 96525.0/4.0*nyy*nzz*nzz + (5816760108318719.0/274877906944.0)*nyy*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz - 3393110063189446.0/962072674305.0*nzz*nzz);  // power=9
        out[18] = T((6633147491942405.0/137438953472.0)*nxx*nxx*nxy*ny*ny - 1243715154739201.0/8589934592.0*nxx*nxx*nxy*nzz + (1.0/879609302221.0)*nxx*nxx*nxy - 4974860618956799.0/17179869184.0*nxx*nxy*ny*nyy*ny + (4974860618956799.0/8589934592.0)*nxx*nxy*ny*nyz*nz + (8401986823127037.0/68719476736.0)*nxx*nxy*ny*ny + (7462290928435199.0/8589934592.0)*nxx*nxy*nzz*nzz - 6301490117345279.0/17179869184.0*nxx*nxy*nzz + (2.0/879609302221.0)*nxx*nxy + (3979888495165439.0/68719476736.0)*nxy*ny*nyy*ny - 7959776990330879.0/68719476736.0*nxy*ny*nyz*nz - 8776164373954557.0/274877906944.0*nxy*ny*ny - 173745*nxy*nzz*nzz + (6582123280465919.0/68719476736.0)*nxy*nzz - 1.0/1000000000000.0*nxy);  // power=9
        out[19] = T((6633147491942405.0/137438953472.0)*nxx*nxx*nxx*nxy - 4974860618956799.0/17179869184.0*nxx*nxx*nxy*ny*ny - 7462290928435201.0/17179869184.0*nxx*nxx*nxy*nzz + (2653258996776955.0/137438953472.0)*nxx*nxx*nxy + (7462290928435199.0/8589934592.0)*nxx*nxy*ny*nyz*nz + (4643203244359679.0/34359738368.0)*nxx*nxy*ny*ny + (7462290928435199.0/8589934592.0)*nxx*nxy*nzz*nzz - 3979888495165439.0/17179869184.0*nxx*nxy*nzz - 3877840072212479.0/274877906944.0*nxx*nxy - 173745*nxy*ny*nyz*nz - 4898324301742079.0/549755813888.0*nxy*ny*ny - 173745*nxy*nzz*nzz + (173745.0/2.0)*nxy*nzz - 540000000000001.0/1000000000000.0*nxy);  // power=9
        out[20] = T((6633147491942405.0/137438953472.0)*nxx*nxx*nxx*nxz - 4974860618956799.0/17179869184.0*nxx*nxx*nxy*nyz - 675675.0/2.0*nxx*nxx*nxz*nz*nz + (2653258996776955.0/137438953472.0)*nxx*nxx*nxz + (4974860618956799.0/17179869184.0)*nxx*nxy*nyz*nz*nz + (4643203244359679.0/34359738368.0)*nxx*nxy*nyz + (4974860618956799.0/17179869184.0)*nxx*nxz*nz*nzz*nz + (3537678662369285.0/274877906944.0)*nxx*nxz*nz*nz - 3877840072212479.0/274877906944.0*nxx*nxz - 3979888495165439.0/68719476736.0*nxy*nyz*nz*nz - 4898324301742079.0/549755813888.0*nxy*nyz - 3979888495165439.0/68719476736.0*nxz*nz*nzz*nz + (6327002223083519.0/274877906944.0)*nxz*nz*nz - 540000000000001.0/1000000000000.0*nxz);  // power=9
        out[21] = T((6633147491942405.0/137438953472.0)*nxx*nxx*nyy*nyz - 6633147491942405.0/137438953472.0*nxx*nxx*nyz*nz*nz - 4974860618956799.0/17179869184.0*nxx*nyy*nyy*nyz + (2984916371374079.0/17179869184.0)*nxx*nyy*nyz + (4974860618956799.0/17179869184.0)*nxx*nyz*nz*nzz*nz - 2984916371374079.0/17179869184.0*nxx*nyz*nz*nz + (5306517993553919.0/274877906944.0)*nyy*nyy*nyz - 28215.0/2.0*nyy*nyz - 5306517993553919.0/274877906944.0*nyz*nz*nzz*nz + (28215.0/2.0)*nyz*nz*nz);  // power=9
        out[22] = T((1243715154739201.0/17179869184.0)*nxx*nxx*nxx*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nxx*nzz - 7462290928435199.0/17179869184.0*nxx*nxx*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nxx*nyy*nzz + (7296462241136635.0/68719476736.0)*nxx*nxx*nyy + (7876862646681599.0/17179869184.0)*nxx*nxx*nzz*nzz - 7296462241136635.0/68719476736.0*nxx*nxx*nzz + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 6582123280465919.0/137438953472.0*nxx*nyy - 4974860618956799.0/34359738368.0*nxx*nzz*nzz*nzz - 57915*nxx*nzz*nzz + (6582123280465919.0/137438953472.0)*nxx*nzz - 7959776990330879.0/274877906944.0*nyy*nyy*nzz - 4455*nyy*nyy - 19305*nyy*nzz*nzz + (5816760108318719.0/274877906944.0)*nyy*nzz + 1620*nyy + (5306517993553919.0/549755813888.0)*nzz*nzz*nzz - 2143016882014758.0/824633720833.0*nzz*nzz - 1620*nzz);  // power=9
        out[23] = T((1243715154739201.0/8589934592.0)*nxx*nxx*nxx*nyz - 7462290928435199.0/8589934592.0*nxx*nxx*nyy*nyz - 7876862646681599.0/8589934592.0*nxx*nxx*nyz*nz*nz + (7296462241136635.0/34359738368.0)*nxx*nxx*nyz + (4974860618956799.0/17179869184.0)*nxx*nyy*nyz*nz*nz + (4974860618956799.0/17179869184.0)*nxx*nyy*nyz + (4974860618956799.0/17179869184.0)*nxx*nyz*nz*nzz*nz + (3979888495165441.0/34359738368.0)*nxx*nyz*nz*nz - 6582123280465919.0/68719476736.0*nxx*nyz - 5306517993553919.0/274877906944.0*nyy*nyz*nz*nz - 8910*nyy*nyz - 5306517993553919.0/274877906944.0*nyz*nz*nzz*nz + (2857355842682879.0/549755813888.0)*nyz*nz*nz + 3240*nyz);  // power=9
        out[24] = T((1243715154739201.0/8589934592.0)*nxx*nxx*nxy*nyz - 6633147491942405.0/137438953472.0*nxx*nxx*nxz*nz*nz - 1.0/879609302221.0*nxx*nxx*nxz - 7462290928435199.0/8589934592.0*nxx*nxy*ny*nyy*nz - 4974860618956799.0/8589934592.0*nxx*nxy*nyz*nz*nz + (6301490117345279.0/17179869184.0)*nxx*nxy*nyz + (4974860618956799.0/17179869184.0)*nxx*nxz*nz*nzz*nz - 8401986823127037.0/68719476736.0*nxx*nxz*nz*nz - 2.0/879609302221.0*nxx*nxz + 173745*nxy*ny*nyy*nz + (7959776990330879.0/68719476736.0)*nxy*nyz*nz*nz - 6582123280465919.0/68719476736.0*nxy*nyz - 3979888495165439.0/68719476736.0*nxz*nz*nzz*nz + (8776164373954557.0/274877906944.0)*nxz*nz*nz + (1.0/1000000000000.0)*nxz);  // power=9
        out[25] = T((135135.0/2.0)*nxx*nxx*nxy*ny - 10395.0/2.0*nxx*nxx*nx - 405405*nxx*nxy*ny*nyy - 405405*nxx*nxy*nyz*nz + 166320*nxx*nxy*ny + 31185*nxx*nxz*nz - 6615*nxx*nx + 93555*nxy*ny*nyy + 93555*nxy*nyz*nz - 104895.0/2.0*nxy*ny - 8505*nxz*nz + (5985.0/2.0)*nx);  // power=8
        out[26] = T((135135.0/2.0)*nxx*nxx*nxz*nz - 10395.0/2.0*nxx*nxx*nx - 405405*nxx*nxy*nyz*nz + 31185*nxx*nxy*ny - 405405*nxx*nxz*nz*nzz + 166320*nxx*nxz*nz - 6615*nxx*nx + 93555*nxy*nyz*nz - 8505*nxy*ny + 93555*nxz*nz*nzz - 104895.0/2.0*nxz*nz + (5985.0/2.0)*nx);  // power=8
        out[27] = T((135135.0/2.0)*nxx*nxx*nyz*nz - 10395.0/2.0*nxx*nxx*ny - 405405*nxx*nyy*nyz*nz + 31185*nxx*nyy*ny - 405405*nxx*nyz*nz*nzz + 249480*nxx*nyz*nz - 14175*nxx*ny + 31185*nyy*nyz*nz - 2835*nyy*ny + 31185*nyz*nz*nzz - 48195.0/2.0*nyz*nz + (3465.0/2.0)*ny);  // power=8
    } break;

    case 17: {
        out[0] = T(-3150*nxx*nyy + 450*nxx + 525*nyy*nyy - 3150*nyy*nzz + 450*nyy + 450*nzz - 135);  // power=5
        out[1] = T(-450*nxx*nxx*nx - 26100*nxx*nxy*ny + 2250*nxx*nxz*nz + 3150*nxx*nx + (5442582557493677.0/733007751851.0)*nxy*ny*nyy - 22950*nxy*nyz*nz + 6300*nxy*ny + 2700*nxz*nz*nzz - 1125*nx);  // power=6
        out[2] = T(-28350*nxx*nyy*ny + 9450*nxx*ny + 4725*nyy*nyy*ny - 28350*nyy*nyz*nz + 1050*nyy*ny + 9450*nyz*nz - 1575*ny);  // power=6
        out[3] = T(-28350*nxx*nyy*nz + 3150*nxx*nz + 4725*nyy*nyy*nz - 28350*nyy*nzz*nz + 9450*nyy*nz + 3150*nzz*nz - 1575*nz);  // power=6
        out[4] = T(-34650*nxx*nxx*nyy + 3150*nxx*nxx + 23100*nxx*nyy*nyy - 17325*nxx*nyy*nzz + 7875*nxx*nyy + 1575*nxx*nzz - 1400*nxx - 5775.0/2.0*nyy*nyy*nyy + (28875.0/2.0)*nyy*nyy*nzz + 525*nyy*nyy + 17325*nyy*nzz*nzz - 15750*nyy*nzz + (875.0/2.0)*nyy - 1575*nzz*nzz + (2975.0/2.0)*nzz - 175);  // power=7
        out[5] = T(-17325*nxx*nxx*nyy + 1575*nxx*nxx - 28875.0/2.0*nxx*nyy*nyy + 17325*nxx*nyy*nzz + 15750*nxx*nyy - 1575*nxx*nzz - 2975.0/2.0*nxx + (5775.0/2.0)*nyy*nyy*nyy - 23100*nyy*nyy*nzz - 525*nyy*nyy + 34650*nyy*nzz*nzz - 7875*nyy*nzz - 875.0/2.0*nyy - 3150*nzz*nzz + 1400*nzz + 175);  // power=7
        out[6] = T(-3150*nxx*nxx*nxy - 100800*nxx*nxy*ny*ny + 22050*nxx*nxy*nzz + 28350*nxx*nxy + 23625*nxy*ny*nyy*ny - 72450*nxy*ny*nyz*nz + 12600*nxy*ny*ny + 25200*nxy*nzz*nzz - 6825*nxy);  // power=7
        out[7] = T(-3150*nxx*nxx*nxz - 84000*nxx*nxy*nyz + 3150*nxx*nxz*nz*nz + 9450*nxx*nxz + 40425*nxy*ny*nyy*nz - 74550*nxy*nyz*nz*nz + 21000*nxy*nyz + 6300*nxz*nz*nzz*nz - 2625*nxz);  // power=7
        out[8] = T(-103950*nxx*nyy*nyz + 28350*nxx*nyz + 17325*nyy*nyy*nyz - 103950*nyy*nyz*nz*nz + 22050*nyy*nyz + 28350*nyz*nz*nz - 9975*nyz);  // power=7
        out[9] = T(-135135*nxx*nxx*nxy*ny + 10395*nxx*nxx*nx + (45045.0/2.0)*nxx*nxy*ny*nyy - 135135*nxx*nxy*nyz*nz + 103950*nxx*nxy*ny + 10395*nxx*nxz*nz - 19845.0/2.0*nxx*nx - 10395.0/2.0*nxy*ny*nyy + 31185*nxy*nyz*nz - 14175*nxy*ny - 2835*nxz*nz + (3465.0/2.0)*nx);  // power=8
        out[10] = T(-135135*nxx*nyy*nyy*ny + 103950*nxx*nyy*ny - 14175*nxx*ny + (45045.0/2.0)*nyy*nyy*nyy*ny - 135135*nyy*nyy*nyz*nz - 31185.0/2.0*nyy*nyy*ny + 103950*nyy*nyz*nz - 4725.0/2.0*nyy*ny - 14175*nyz*nz + (2625.0/2.0)*ny);  // power=8
        out[11] = T(-135135*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 10395*nxx*nzz*nz - 2835*nxx*nz + (45045.0/2.0)*nyy*nyy*nzz*nz - 10395.0/2.0*nyy*nyy*nz - 135135*nyy*nzz*nzz*nz + 103950*nyy*nzz*nz - 14175*nyy*nz + 10395*nzz*nzz*nz - 19845.0/2.0*nzz*nz + (3465.0/2.0)*nz);  // power=8
        out[12] = T(-405405*nxx*nxx*nyy*ny + 93555*nxx*nxx*ny + (135135.0/2.0)*nxx*nyy*nyy*ny - 405405*nxx*nyy*nyz*nz + 166320*nxx*nyy*ny + 93555*nxx*nyz*nz - 104895.0/2.0*nxx*ny - 10395.0/2.0*nyy*nyy*ny + 31185*nyy*nyz*nz - 6615*nyy*ny - 8505*nyz*nz + (5985.0/2.0)*ny);  // power=8
        out[13] = T(-405405*nxx*nxx*nyy*nz + 31185*nxx*nxx*nz + (135135.0/2.0)*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 249480*nxx*nyy*nz + 31185*nxx*nzz*nz - 48195.0/2.0*nxx*nz - 10395.0/2.0*nyy*nyy*nz + 31185*nyy*nzz*nz - 14175*nyy*nz - 2835*nzz*nz + (3465.0/2.0)*nz);  // power=8
        out[14] = T(-405405*nxx*nyy*nyy*nz + 187110*nxx*nyy*nz - 8505*nxx*nz + (135135.0/2.0)*nyy*nyy*nyy*nz - 405405*nyy*nyy*nzz*nz + (93555.0/2.0)*nyy*nyy*nz + 187110*nyy*nzz*nz - 93555.0/2.0*nyy*nz - 8505*nzz*nz + (5985.0/2.0)*nz);  // power=8
        out[15] = T(-51030*nxx*nxx*nxy*nz - 744660*nxx*nxy*ny*nyz + 85050*nxx*nxy*nzz*nz + 187110*nxx*nxy*nz + 252315*nxy*ny*nyy*nyz - 557550*nxy*ny*nyz*nz*nz + 102060*nxy*ny*nyz + 136080*nxy*nzz*nzz*nz - 36855*nxy*nz);  // power=8
        out[16] = T(-4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy + (3979888495160613.0/824633720831.0)*nxx*nxx*nxx + (6633147491942405.0/549755813888.0)*nxx*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nxx*nyy*nzz + (173745.0/4.0)*nxx*nxx*nyy - 96525.0/4.0*nxx*nxx*nzz - 3393110063189446.0/962072674305.0*nxx*nxx - 1243715154739201.0/17179869184.0*nxx*nyy*nyy*nzz + (1.0/1000000000000.0)*nxx*nyy*nyy + (6218575773695999.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 96525.0/4.0*nxx*nzz*nzz + (5816760108318719.0/274877906944.0)*nxx*nzz + (6633147491942405.0/549755813888.0)*nyy*nyy*nzz*nzz + (1.0/1000000000000.0)*nyy*nyy*nzz - 4974860618956799.0/68719476736.0*nyy*nzz*nzz*nzz + (173745.0/4.0)*nyy*nzz*nzz + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz - 3393110063189446.0/962072674305.0*nzz*nzz);  // power=9
        out[17] = T(-4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (173745.0/4.0)*nxx*nyy*nyy - 4974860618956799.0/68719476736.0*nxx*nyy*nzz*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nzz - 4455*nxx*nyy + (3979888495160613.0/824633720831.0)*nxx*nzz*nzz + 4455*nxx*nzz + (6633147491942405.0/549755813888.0)*nyy*nyy*nyy*nyy - 2487430309478401.0/17179869184.0*nyy*nyy*nyy*nzz - 2211049163980803.0/274877906944.0*nyy*nyy*nyy + (7669576787558399.0/17179869184.0)*nyy*nyy*nzz*nzz - 1077886467440639.0/17179869184.0*nyy*nyy*nzz + (1479702132817919.0/274877906944.0)*nyy*nyy - 4974860618956799.0/68719476736.0*nyy*nzz*nzz*nzz - 405405.0/4.0*nyy*nzz*nzz + (2984916371374079.0/68719476736.0)*nyy*nzz - 2159999999999999.0/1000000000000.0*nyy + (3979888495160613.0/824633720831.0)*nzz*nzz*nzz + (892923700839329.0/962072674305.0)*nzz*nzz - 1620*nzz + 105);  // power=9
        out[18] = T(3780*nxx*nxx*nxy*ny*ny - 9001134760919039.0/34359738368.0*nxx*nxy*ny*nyy*ny + 801765*nxx*nxy*ny*nyz*nz + (8033994025205759.0/68719476736.0)*nxx*nxy*ny*ny - 173745*nxx*nxy*nzz*nzz - 93555*nxx*nxy*nzz - 2616150479339519.0/274877906944.0*nxx*nxy + (4953987077898241.0/68719476736.0)*nxy*ny*nyy*nyy*ny - 481275*nxy*ny*nyy*nyz*nz - 2606873349980165.0/137438953472.0*nxy*ny*nyy*ny + 624240*nxy*ny*nyz*nz*nzz - 6540376198348799.0/68719476736.0*nxy*ny*nyz*nz - 257285720899607.0/952910077406.0*nxy*ny*ny - 173745*nxy*nzz*nzz*nzz + (5510614839459839.0/68719476736.0)*nxy*nzz*nzz + (5343626510992571.0/879609302221.0)*nxy*nzz + (47498902319879.0/703687441776.0)*nxy);  // power=9
        out[19] = T(-285795*nxx*nxx*nxy*ny*ny + (3979888495165439.0/68719476736.0)*nxx*nxx*nxy + (8590621786767361.0/137438953472.0)*nxx*nxy*ny*nyy*ny + 463590*nxx*nxy*ny*nyz*nz + (7984516001955835.0/68719476736.0)*nxx*nxy*ny*ny - 173745*nxx*nxy*nzz*nzz - 35640*nxx*nxy*nzz - 4749890231992319.0/137438953472.0*nxx*nxy + (2875910101401601.0/274877906944.0)*nxy*ny*nyy*nyy*ny - 8715863033118721.0/34359738368.0*nxy*ny*nyy*nyz*nz - 25515.0/2.0*nxy*ny*nyy*ny + (4944709948538881.0/8589934592.0)*nxy*ny*nyz*nz*nzz - 5826037237678079.0/34359738368.0*nxy*ny*nyz*nz - 296868139499249.0/879609302220.0*nxy*ny*ny - 173745*nxy*nzz*nzz*nzz + (5510614839459839.0/68719476736.0)*nxy*nzz*nzz + (7124835347988479.0/549755813888.0)*nxy*nzz + (1447232180062792.0/549755813889.0)*nxy);  // power=9
        out[20] = T(-2025*nxx*nxx*nxx*nxz - 4985683936542719.0/17179869184.0*nxx*nxx*nxy*nyz + 16200*nxx*nxx*nxy*nyz + 8775*nxx*nxx*nxz*nz*nz + (5306517993553919.0/274877906944.0)*nxx*nxx*nxz + (7304193182269445.0/137438953472.0)*nxx*nxy*ny*nyy*nz + (5287963734835199.0/137438953472.0)*nxx*nxy*ny*nyy*nz + (1212211569623039.0/274877906944.0)*nxx*nxy*nyz*nz*nz - 8998815478579199.0/68719476736.0*nxx*nxy*nyz*nz*nz + 170910*nxx*nxy*nyz + 4320*nxx*nxz*nz*nzz*nz - 7644354592112639.0/549755813888.0*nxx*nxz + (11025.0/2.0)*nxy*ny*nyy*nyy*nz + (5566277615615999.0/274877906944.0)*nxy*ny*nyy*nyy*nz - 5182822935429121.0/137438953472.0*nxy*ny*nyy*nzz*nz - 139725*nxy*ny*nyy*nzz*nz - 1233858204794881.0/137438953472.0*nxy*ny*nyy*nz + (1385384650997761.0/274877906944.0)*nxy*nyz*nz*nzz*nz + (8460741975736319.0/68719476736.0)*nxy*nyz*nz*nzz*nz + (2968681394995201.0/274877906944.0)*nxy*nyz*nz*nz - 5114790653460479.0/274877906944.0*nxy*nyz - 6479999999999999.0/1000000000000.0*nxz*nz*nzz*nzz*nz - 540*nxz*nz*nz + (3645.0/2.0)*nxz);  // power=9
        out[21] = T(-4974860618956799.0/17179869184.0*nxx*nyy*nyy*nyz + (4974860618956799.0/17179869184.0)*nxx*nyy*nyz*nz*nz + (4643203244359679.0/34359738368.0)*nxx*nyy*nyz - 3979888495165439.0/68719476736.0*nxx*nyz*nz*nz - 4898324301742079.0/549755813888.0*nxx*nyz + (6633147491942405.0/137438953472.0)*nyy*nyy*nyy*nyz - 675675.0/2.0*nyy*nyy*nyz*nz*nz + (2653258996776955.0/137438953472.0)*nyy*nyy*nyz + (4974860618956799.0/17179869184.0)*nyy*nyz*nz*nzz*nz + (3537678662369285.0/274877906944.0)*nyy*nyz*nz*nz - 3877840072212479.0/274877906944.0*nyy*nyz - 3979888495165439.0/68719476736.0*nyz*nz*nzz*nz + (6327002223083519.0/274877906944.0)*nyz*nz*nz - 540000000000001.0/1000000000000.0*nyz);  // power=9
        out[22] = T(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nxx*nzz - 4455*nxx*nxx + (1243715154739201.0/17179869184.0)*nxx*nyy*nyy*nyy - 1243715154739201.0/17179869184.0*nxx*nyy*nyy*nzz + (7296462241136635.0/68719476736.0)*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nzz*nzz - 4477374557061119.0/17179869184.0*nxx*nyy*nzz - 6582123280465919.0/137438953472.0*nxx*nyy - 19305*nxx*nzz*nzz + (5816760108318719.0/274877906944.0)*nxx*nzz + 1620*nxx - 1243715154739201.0/17179869184.0*nyy*nyy*nyy*nzz + (7876862646681599.0/17179869184.0)*nyy*nyy*nzz*nzz - 7296462241136635.0/68719476736.0*nyy*nyy*nzz - 4974860618956799.0/34359738368.0*nyy*nzz*nzz*nzz - 57915*nyy*nzz*nzz + (6582123280465919.0/137438953472.0)*nyy*nzz + (5306517993553919.0/549755813888.0)*nzz*nzz*nzz - 2143016882014758.0/824633720833.0*nzz*nzz - 1620*nzz);  // power=9
        out[23] = T(-7462290928435199.0/8589934592.0*nxx*nxx*nyy*nyz + 173745*nxx*nxx*nyz + (1243715154739201.0/8589934592.0)*nxx*nyy*nyy*nyz - 4974860618956799.0/8589934592.0*nxx*nyy*nyz*nz*nz + (6301490117345279.0/17179869184.0)*nxx*nyy*nyz + (7959776990330879.0/68719476736.0)*nxx*nyz*nz*nz - 6582123280465919.0/68719476736.0*nxx*nyz - 6633147491942405.0/137438953472.0*nyy*nyy*nyz*nz*nz - 1.0/879609302221.0*nyy*nyy*nyz + (4974860618956799.0/17179869184.0)*nyy*nyz*nz*nzz*nz - 8401986823127037.0/68719476736.0*nyy*nyz*nz*nz - 2.0/879609302221.0*nyy*nyz - 3979888495165439.0/68719476736.0*nyz*nz*nzz*nz + (8776164373954557.0/274877906944.0)*nyz*nz*nz + (1.0/1000000000000.0)*nyz);  // power=9
        out[24] = T(-2025*nxx*nxx*nxx*nxz - 11970*nxx*nxx*nxy*nyz + 16200*nxx*nxx*nxy*nyz + 8775*nxx*nxx*nxz*nz*nz - 920745*nxx*nxy*ny*nyy*nz + (5287963734835199.0/137438953472.0)*nxx*nxy*ny*nyy*nz + (6414361857884159.0/17179869184.0)*nxx*nxy*nyz*nz*nz - 8998815478579199.0/68719476736.0*nxx*nxy*nyz*nz*nz + 235710*nxx*nxy*nyz + 4320*nxx*nxz*nz*nzz*nz - 5306517993553919.0/274877906944.0*nxx*nxz*nz*nz - 10125.0/2.0*nxx*nxz + (209475.0/2.0)*nxy*ny*nyy*nyy*nz + (5566277615615999.0/274877906944.0)*nxy*ny*nyy*nyy*nz - 1722555.0/2.0*nxy*ny*nyy*nzz*nz - 139725*nxy*ny*nyy*nzz*nz + 186570*nxy*ny*nyy*nz + 95760*nxy*nyz*nz*nzz*nz + (8460741975736319.0/68719476736.0)*nxy*nyz*nz*nzz*nz + 147285*nxy*nyz*nz*nz - 6648609374207999.0/137438953472.0*nxy*nyz - 6479999999999999.0/1000000000000.0*nxz*nz*nzz*nzz*nz - 5804999999999999.0/1000000000000.0*nxz*nz*nz + (2835.0/2.0)*nxz);  // power=9
        out[25] = T(-385560*nxx*nxy*ny*nyy + 52920*nxx*nxy*nyz*nz + 173880*nxx*nxy*ny - 7560*nxx*nx + (174825.0/2.0)*nxy*ny*nyy*nyy - 332640*nxy*ny*nyy*nzz + (27405.0/2.0)*nxy*ny*nyy + 52920*nxy*nyz*nz*nzz + 120960*nxy*nyz*nz - 65205.0/2.0*nxy*ny - 7560*nxz*nz + (4095.0/2.0)*nx);  // power=8
        out[26] = T(945*nxx*nxx*nxx*nx - 5670*nxx*nxx*nxy*ny - 14175*nxx*nxx*nxz*nz - 6615.0/2.0*nxx*nxy*ny*nyy - 14175*nxx*nxy*ny*nyy - 414225*nxx*nxy*nyz*nz + 141750*nxx*nxy*nyz*nz + 33390*nxx*nxy*ny + 31185*nxx*nxz*nz - 5985.0/2.0*nxx*nx - 6615.0/2.0*nxy*ny*nyy*nyy - 7560*nxy*ny*nyy*nyy + 55440*nxy*ny*nyy*nzz + 155925*nxy*ny*nyy*nzz + 315*nxy*ny*nyy - 8820*nxy*nyz*nz*nzz - 226800*nxy*nyz*nz*nzz + 73395*nxy*nyz*nz - 16065.0/2.0*nxy*ny + 15120*nxz*nz*nzz*nzz - 7245*nxz*nz + 945*nx);  // power=8
        out[27] = T(-405405*nxx*nyy*nyz*nz + 31185*nxx*nyy*ny + 93555*nxx*nyz*nz - 8505*nxx*ny + (135135.0/2.0)*nyy*nyy*nyz*nz - 10395.0/2.0*nyy*nyy*ny - 405405*nyy*nyz*nz*nzz + 166320*nyy*nyz*nz - 6615*nyy*ny + 93555*nyz*nz*nzz - 104895.0/2.0*nyz*nz + (5985.0/2.0)*ny);  // power=8
    } break;

    case 18: {
        out[0] = T(-1575*nxx*nxy + 2100*nxy*ny*ny - 1575*nxy*nzz);  // power=5
        out[1] = T(-14175*nxx*nxx*ny + 18900*nxx*nyy*ny - 14175*nxx*nyz*nz + 4725*nxx*ny - 2100*nyy*ny + 1575*nyz*nz);  // power=6
        out[2] = T(-225*nxx*nxx*nx - 13050*nxx*nxy*ny + 1125*nxx*nxz*nz + 1575*nxx*nx + (5566277615616001.0/274877906944.0)*nxy*ny*nyy - 11475*nxy*nyz*nz - 7875*nxy*ny + 1350*nxz*nz*nzz + 225*nx);  // power=6
        out[3] = T(-11025*nxx*nxy*nz + 22050*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[4] = T(-34825.0/2.0*nxx*nxx*nxy + 31850*nxx*nxy*ny*ny - 8050*nxx*nxy*nzz + 7875*nxx*nxy - 11375*nxy*ny*nyy*ny - 4025.0/2.0*nxy*ny*nyz*nz + (3675.0/2.0)*nxy*ny*ny + (18725.0/2.0)*nxy*nzz*nzz - 4725.0/2.0*nxy*nzz - 2975.0/2.0*nxy);  // power=7
        out[5] = T(-9625*nxx*nxx*nxy + 3850*nxx*nxy*ny*ny + 15400*nxx*nxy*nzz + 6300*nxx*nxy + 13475*nxy*ny*nyy*ny - 44275.0/2.0*nxy*ny*nyz*nz - 22575.0/2.0*nxy*ny*ny + 25025*nxy*nzz*nzz - 23625.0/2.0*nxy*nzz + (2975.0/2.0)*nxy);  // power=7
        out[6] = T(-51975*nxx*nxx*nyy + 4725*nxx*nxx + 69300*nxx*nyy*nyy - 51975*nxx*nyy*nzz - 4725*nxx*nyy + 4725*nxx*nzz - 1050*nxx - 6300*nyy*nyy + 4725*nyy*nzz + 2625*nyy - 525);  // power=7
        out[7] = T(-51975*nxx*nxx*nyz + 69300*nxx*nyy*nyz - 51975*nxx*nyz*nz*nz + 23625*nxx*nyz - 6300*nyy*nyz + 4725*nyz*nz*nz - 1050*nyz);  // power=7
        out[8] = T(-1575*nxx*nxx*nxz - 49350*nxx*nxy*nyz + 1575*nxx*nxz*nz*nz + 4725*nxx*nxz + 73500*nxy*ny*nyy*nz - 44625*nxy*nyz*nz*nz - 15225*nxy*nyz + 3150*nxz*nz*nzz*nz + 525*nxz);  // power=7
        out[9] = T(-135135.0/2.0*nxx*nxx*nxx*ny + 90090*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*nyz*nz + 62370*nxx*nxx*ny - 41580*nxx*nyy*ny + 31185*nxx*nyz*nz - 25515.0/2.0*nxx*ny + 1890*nyy*ny - 2835.0/2.0*nyz*nz + 315*ny);  // power=8
        out[10] = T(-64260*nxx*nxy*ny*nyy + 8820*nxx*nxy*nyz*nz + 28980*nxx*nxy*ny - 1260*nxx*nx + (186795.0/2.0)*nxy*ny*nyy*nyy - 55440*nxy*ny*nyy*nzz - 177345.0/2.0*nxy*ny*nyy + 8820*nxy*nyz*nz*nzz + 20160*nxy*nyz*nz + (38745.0/2.0)*nxy*ny - 1260*nxz*nz - 1155.0/2.0*nx);  // power=8
        out[11] = T(5670*nxx*nxx*nxy*nz - 9450*nxx*nxy*ny*nyz - 154035.0/2.0*nxx*nxy*nzz*nz + (31185.0/2.0)*nxx*nxy*nz - 15120*nxy*ny*nyy*nyz + 59850*nxy*ny*nyz*nz*nz - 165375.0/2.0*nxy*nzz*nzz*nz + (135135.0/2.0)*nxy*nzz*nz - 11340*nxy*nz);  // power=8
        out[12] = T(-405405.0/2.0*nxx*nxx*nxy*ny + (31185.0/2.0)*nxx*nxx*nx + 270270*nxx*nxy*ny*nyy - 405405.0/2.0*nxx*nxy*nyz*nz + (93555.0/2.0)*nxx*nxy*ny + (31185.0/2.0)*nxx*nxz*nz - 19845.0/2.0*nxx*nx - 62370*nxy*ny*nyy + (93555.0/2.0)*nxy*nyz*nz + 8505*nxy*ny - 8505.0/2.0*nxz*nz + 945*nx);  // power=8
        out[13] = T(-405405.0/2.0*nxx*nxx*nxy*nz + 270270*nxx*nxy*ny*nyz - 405405.0/2.0*nxx*nxy*nzz*nz + (280665.0/2.0)*nxx*nxy*nz - 62370*nxy*ny*nyz + (93555.0/2.0)*nxy*nzz*nz - 17010*nxy*nz);  // power=8
        out[14] = T(-25515.0/2.0*nxx*nxx*nxy*nz - 219240*nxx*nxy*ny*nyz + (42525.0/2.0)*nxx*nxy*nzz*nz + (93555.0/2.0)*nxx*nxy*nz + 266490*nxy*ny*nyy*nyz - 344925.0/2.0*nxy*ny*nyz*nz*nz - 246645.0/2.0*nxy*ny*nyz + 34020*nxy*nzz*nzz*nz + (31185.0/2.0)*nxy*nz);  // power=8
        out[15] = T(-405405*nxx*nxx*nyy*nz + 31185*nxx*nxx*nz + 540540*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz - 14175*nxx*nz - 41580*nyy*nyy*nz + 31185*nyy*nzz*nz + 5670*nyy*nz - 2835*nzz*nz + 630*nz);  // power=8
        out[16] = T(-4974860618956799.0/137438953472.0*nxx*nxx*nxx*nxy + (6662009672171521.0/137438953472.0)*nxx*nxx*nxy*ny*ny + (6218575773695999.0/34359738368.0)*nxx*nxx*nxy*nzz + (1989944247582719.0/68719476736.0)*nxx*nxx*nxy + 420*nxx*nxy*ny*nyy*ny - 297345*nxx*nxy*ny*nyz*nz - 3537678662369279.0/274877906944.0*nxx*nxy*ny*ny + (1370655.0/8.0)*nxx*nxy*nzz*nzz - 8980261219860479.0/68719476736.0*nxx*nxy*nzz - 4267479505304045.0/916259689813.0*nxx*nxy + 210*nxy*ny*nyy*nyy*ny - 7769999999999999.0/1000000000000.0*nxy*ny*nyy*nyz*nz + 30630*nxy*ny*nyz*nz*nzz + (7044433560207361.0/137438953472.0)*nxy*ny*nyz*nz - 362838837166163.0/879609302221.0*nxy*ny*ny - 6301490117345279.0/137438953472.0*nxy*nzz*nzz*nzz + (4454999999999999.0/1000000000000.0)*nxy*nzz*nzz + (4786998749429759.0/549755813888.0)*nxy*nzz + (405.0/2.0)*nxy);  // power=9
        out[17] = T(1155*nxx*nxx*nxy*ny*ny - 75405.0/4.0*nxx*nxy*ny*nyy*ny + (916485.0/4.0)*nxx*nxy*ny*nyz*nz - 849214149048800.0/211444543803.0*nxx*nxy*ny*ny - 6135661430046719.0/68719476736.0*nxx*nxy*nzz*nzz - 4455000000000001.0/1000000000000.0*nxx*nxy*nzz + (853495901062673.0/733007751852.0)*nxx*nxy + (515625.0/8.0)*nxy*ny*nyy*nyy*ny - 1195185.0/4.0*nxy*ny*nyy*nyz*nz - 538515.0/8.0*nxy*ny*nyy*ny + (7667032019435519.0/34359738368.0)*nxy*ny*nyz*nz*nzz + (2362962157240321.0/34359738368.0)*nxy*ny*nyz*nz + (198525.0/8.0)*nxy*ny*ny - 6135661430046719.0/68719476736.0*nxy*nzz*nzz*nzz + (271755.0/4.0)*nxy*nzz*nzz - 7027425489715201.0/274877906944.0*nxy*nzz - 16875.0/8.0*nxy);  // power=9
        out[18] = T(-4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nxx*nyy*nzz + (7959776990330877.0/274877906944.0)*nxx*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nxx*nzz + (6633147491942401.0/34359738368.0)*nxx*nyy*nyy*nyy - 1447875.0/2.0*nxx*nyy*nyy*nzz - 173745.0/2.0*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (31185.0/2.0)*nxx*nzz - 945*nxx - 7075357324738559.0/549755813888.0*nyy*nyy*nyy + (6633147491942401.0/137438953472.0)*nyy*nyy*nzz + 8910*nyy*nyy - 7959776990330879.0/274877906944.0*nyy*nzz*nzz - 2040968459058460.0/916259689813.0*nyy*nzz - 2025*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz + 105);  // power=9
        out[19] = T(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*nyy + (5306517993553919.0/549755813888.0)*nxx*nxx*nxx + (6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nxx*nyy*nzz + (6633147491942403.0/137438953472.0)*nxx*nxx*nyy - 19305*nxx*nxx*nzz - 1836871613153279.0/274877906944.0*nxx*nxx - 579150*nxx*nyy*nyy*nzz - 5306517993553919.0/137438953472.0*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz - 6.0/753950830475.0*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (3673743226306559.0/274877906944.0)*nxx*nzz + 810*nxx + (5306517993553921.0/137438953472.0)*nyy*nyy*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (4455.0/2.0)*nzz*nzz - 810*nzz);  // power=9
        out[20] = T(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*nyz + (6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyz + (7296462241136639.0/68719476736.0)*nxx*nxx*nyz - 6633147491942401.0/34359738368.0*nxx*nyy*nyz*nz*nz - 38610*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz - 7959776990330875.0/137438953472.0*nxx*nyz*nz*nz - 13365*nxx*nyz + (7075357324738559.0/549755813888.0)*nyy*nyz*nz*nz - 5306517993553919.0/549755813888.0*nyz*nz*nzz*nz + 4455*nyz*nz*nz);  // power=9
        out[21] = T(-2025.0/2.0*nxx*nxx*nxx*nxz - 4409999999999999.0/1000000000000.0*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz - 303660*nxx*nxy*ny*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nxy*ny*nyy*nz + (6035545742376959.0/34359738368.0)*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz + (2504824927027199.0/274877906944.0)*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz - 5306517993553919.0/549755813888.0*nxx*nxz*nz*nz + (2835.0/2.0)*nxx*nxz + (77175.0/2.0)*nxy*ny*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nxy*ny*nyy*nyy*nz - 457020*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz + (4629287550320639.0/137438953472.0)*nxy*ny*nyy*nz + (4848846278492159.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (6410496387317761.0/137438953472.0)*nxy*nyz*nz*nz + (8893674679173121.0/274877906944.0)*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 1552499999998447.0/999999999999.0*nxz*nz*nz - 1350*nxz);  // power=9
        out[22] = T(-7415389885562879.0/34359738368.0*nxx*nxx*nxy*ny*ny + (7462290928435199.0/34359738368.0)*nxx*nxx*nxy*nzz + (7959776990330879.0/274877906944.0)*nxx*nxx*nxy + (5251370613473281.0/17179869184.0)*nxx*nxy*ny*nyy*ny - 291480*nxx*nxy*ny*nyz*nz - 82215.0/2.0*nxx*nxy*ny*ny + (5638175368151039.0/68719476736.0)*nxx*nxy*nzz*nzz - 3979888495165439.0/34359738368.0*nxx*nxy*nzz - 333976656936656.0/219902325555.0*nxx*nxy + (8097903138570241.0/549755813888.0)*nxy*ny*nyy*nyy*ny - 278115*nxy*ny*nyy*nyz*nz - 64125*nxy*ny*nyy*ny + 247695*nxy*ny*nyz*nz*nzz + 146925*nxy*ny*nyz*nz + (99435.0/4.0)*nxy*ny*ny - 135135*nxy*nzz*nzz*nzz + (4974860618956799.0/68719476736.0)*nxy*nzz*nzz - 5928085660631041.0/274877906944.0*nxy*nzz - 1923749999994229.0/999999999997.0*nxy);  // power=9
        out[23] = T(-2025.0/2.0*nxx*nxx*nxx*nxz - 7505584198778879.0/17179869184.0*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz + (7959776990330879.0/274877906944.0)*nxx*nxx*nxz + 598680*nxx*nxy*ny*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nxy*ny*nyy*nz - 4671807726551039.0/17179869184.0*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz + (3590249062072319.0/34359738368.0)*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz + 19305*nxx*nxz*nz*nz - 17010*nxx*nxz + 22050*nxy*ny*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nxy*ny*nyy*nyy*nz - 150840*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz - 113130*nxy*ny*nyy*nz + 20160*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + 101115*nxy*nyz*nz*nz + (2718817377582401.0/769658139443.0)*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 6615*nxz*nz*nz + (1822499999998177.0/999999999999.0)*nxz);  // power=9
        out[24] = T(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nxx*nyz*nz*nz + (7959776990330879.0/137438953472.0)*nxx*nxx*nyz + 579150*nxx*nyy*nyy*nyz - 1254825.0/2.0*nxx*nyy*nyz*nz*nz - 2984916371374081.0/34359738368.0*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz - 994972123791359.0/34359738368.0*nxx*nyz*nz*nz - 5306517993553921.0/137438953472.0*nyy*nyy*nyz + (5748727826350079.0/137438953472.0)*nyy*nyz*nz*nz + (7347486452613121.0/549755813888.0)*nyy*nyz - 5306517993553919.0/549755813888.0*nyz*nz*nzz*nz - 1.0/1000000000000.0*nyz*nz*nz - 1215*nyz);  // power=9
        out[25] = T(-405405.0/2.0*nxx*nxx*nyy*ny + (93555.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*nyy*ny - 405405.0/2.0*nxx*nyy*nyz*nz - 197505.0/2.0*nxx*nyy*ny + (93555.0/2.0)*nxx*nyz*nz - 2835.0/2.0*nxx*ny - 20790*nyy*nyy*ny + (31185.0/2.0)*nyy*nyz*nz + 13230*nyy*ny - 8505.0/2.0*nyz*nz - 1260*ny);  // power=8
        out[26] = T(-405405.0/2.0*nxx*nxx*nyz*nz + (31185.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*nyz*nz - 20790*nxx*nyy*ny - 405405.0/2.0*nxx*nyz*nz*nzz + 124740*nxx*nyz*nz - 14175.0/2.0*nxx*ny - 20790*nyy*nyz*nz + 1890*nyy*ny + (31185.0/2.0)*nyz*nz*nzz - 14175.0/2.0*nyz*nz + 315*ny);  // power=8
        out[27] = T((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nxy*ny - 14175.0/2.0*nxx*nxx*nxz*nz - 13230*nxx*nxy*ny*nyy - 14175.0/2.0*nxx*nxy*ny*nyy - 475965.0/2.0*nxx*nxy*nyz*nz + 70875*nxx*nxy*nyz*nz + (48825.0/2.0)*nxx*nxy*ny + (31185.0/2.0)*nxx*nxz*nz - 4095.0/2.0*nxx*nx - 13230*nxy*ny*nyy*nyy - 3780*nxy*ny*nyy*nyy + 221760*nxy*ny*nyy*nzz + (155925.0/2.0)*nxy*ny*nyy*nzz + 1260*nxy*ny*nyy - 35280*nxy*nyz*nz*nzz - 113400*nxy*nyz*nz*nzz - 67725.0/2.0*nxy*nyz*nz - 4725.0/2.0*nxy*ny + 7560*nxz*nz*nzz*nzz + (1575.0/2.0)*nxz*nz + (945.0/2.0)*nx);  // power=8
    } break;

    case 19: {
        out[0] = T(2100*nxx*nxy - 1575*nxy*ny*ny - 1575*nxy*nzz);  // power=5
        out[1] = T(18900*nxx*nxx*ny - 14175*nxx*nyy*ny - 14175*nxx*nyz*nz - 6300*nxx*ny + 1575*nyy*ny + 1575*nyz*nz);  // power=6
        out[2] = T(-225*nxx*nxx*nx + 20025*nxx*nxy*ny + 1125*nxx*nxz*nz - 2100*nxx*nx - 7050618313113599.0/549755813888.0*nxy*ny*nyy - 11475*nxy*nyz*nz + 3150*nxy*ny + 1350*nxz*nz*nzz + 225*nx);  // power=6
        out[3] = T(22050*nxx*nxy*nz - 11025*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[4] = T(23625*nxx*nxx*nxy - 29400*nxx*nxy*ny*ny - 32550*nxx*nxy*nzz - 10500*nxx*nxy + (15225.0/2.0)*nxy*ny*nyy*ny + 12075*nxy*ny*nyz*nz + (8925.0/2.0)*nxy*nzz*nzz + 3150*nxy*nzz + 1575*nxy);  // power=7
        out[5] = T((23625.0/2.0)*nxx*nxx*nxy + 2625*nxx*nxy*ny*ny - 33600*nxx*nxy*nzz - 8400*nxx*nxy - 18375.0/2.0*nxy*ny*nyy*ny + (12075.0/2.0)*nxy*ny*nyz*nz + (14175.0/2.0)*nxy*ny*ny + 15225*nxy*nzz*nzz - 1575.0/2.0*nxy*nzz + (525.0/2.0)*nxy);  // power=7
        out[6] = T(69300*nxx*nxx*nyy - 6300*nxx*nxx - 51975*nxx*nyy*nyy - 51975*nxx*nyy*nzz - 4725*nxx*nyy + 4725*nxx*nzz + 2625*nxx + 4725*nyy*nyy + 4725*nyy*nzz - 1050*nyy - 525);  // power=7
        out[7] = T(69300*nxx*nxx*nyz - 51975*nxx*nyy*nyz - 51975*nxx*nyz*nz*nz - 9450*nxx*nyz + 4725*nyy*nyz + 4725*nyz*nz*nz - 1050*nyz);  // power=7
        out[8] = T(-1575*nxx*nxx*nxz + 86625*nxx*nxy*nyz + 1575*nxx*nxz*nz*nz - 6300*nxx*nxz - 33075*nxy*ny*nyy*nz - 29925*nxy*nyz*nz*nz + 3150*nxy*nyz + 3150*nxz*nz*nzz*nz + 525*nxz);  // power=7
        out[9] = T(90090*nxx*nxx*nxx*ny - 135135.0/2.0*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*nyz*nz - 83160*nxx*nxx*ny + 31185*nxx*nyy*ny + 31185*nxx*nyz*nz + 17010*nxx*ny - 2835.0/2.0*nyy*ny - 2835.0/2.0*nyz*nz - 420*ny);  // power=8
        out[10] = T((186795.0/2.0)*nxx*nxy*ny*nyy + 8820*nxx*nxy*nyz*nz - 43785*nxx*nxy*ny + (4095.0/2.0)*nxx*nx - 64260*nxy*ny*nyy*nyy - 55440*nxy*ny*nyy*nzz + (113715.0/2.0)*nxy*ny*nyy + 8820*nxy*nyz*nz*nzz + 20160*nxy*nyz*nz - 10395*nxy*ny - 1260*nxz*nz + (315.0/2.0)*nx);  // power=8
        out[11] = T(-8505.0/2.0*nxx*nxx*nxy*nz + (14175.0/2.0)*nxx*nxy*ny*nyz + (194355.0/2.0)*nxx*nxy*nzz*nz - 20790*nxx*nxy*nz + 11340*nxy*ny*nyy*nyz - 89775.0/2.0*nxy*ny*nyz*nz*nz - 112455.0/2.0*nxy*nzz*nzz*nz + 31185*nxy*nzz*nz - 2835.0/2.0*nxy*nz);  // power=8
        out[12] = T(270270*nxx*nxx*nxy*ny - 20790*nxx*nxx*nx - 405405.0/2.0*nxx*nxy*ny*nyy - 405405.0/2.0*nxx*nxy*nyz*nz - 197505.0/2.0*nxx*nxy*ny + (31185.0/2.0)*nxx*nxz*nz + 13230*nxx*nx + (93555.0/2.0)*nxy*ny*nyy + (93555.0/2.0)*nxy*nyz*nz - 2835.0/2.0*nxy*ny - 8505.0/2.0*nxz*nz - 1260*nx);  // power=8
        out[13] = T(270270*nxx*nxx*nxy*nz - 405405.0/2.0*nxx*nxy*ny*nyz - 405405.0/2.0*nxx*nxy*nzz*nz - 114345*nxx*nxy*nz + (93555.0/2.0)*nxy*ny*nyz + (93555.0/2.0)*nxy*nzz*nz + 2835*nxy*nz);  // power=8
        out[14] = T(-25515.0/2.0*nxx*nxx*nxy*nz + (639765.0/2.0)*nxx*nxy*ny*nyz + (42525.0/2.0)*nxx*nxy*nzz*nz - 62370*nxx*nxy*nz - 280665.0/2.0*nxy*ny*nyy*nyz - 212625.0/2.0*nxy*ny*nyz*nz*nz + 65205*nxy*ny*nyz + 34020*nxy*nzz*nzz*nz - 8505.0/2.0*nxy*nz);  // power=8
        out[15] = T(540540*nxx*nxx*nyy*nz - 41580*nxx*nxx*nz - 405405*nxx*nyy*nyy*nz - 405405*nxx*nyy*nzz*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz + 5670*nxx*nz + 31185*nyy*nyy*nz + 31185*nyy*nzz*nz - 14175*nyy*nz - 2835*nzz*nz + 630*nz);  // power=8
        out[16] = T((6633147491942401.0/137438953472.0)*nxx*nxx*nxx*nxy - 4996507254128639.0/137438953472.0*nxx*nxx*nxy*ny*ny - 5596718196326401.0/17179869184.0*nxx*nxx*nxy*nzz - 38610*nxx*nxx*nxy - 315*nxx*nxy*ny*nyy*ny + (7662522303774719.0/34359738368.0)*nxx*nxy*ny*nyz*nz + (2653258996776959.0/274877906944.0)*nxx*nxy*ny*ny + (2181465.0/8.0)*nxx*nxy*nzz*nzz + (4413594292715521.0/68719476736.0)*nxx*nxy*nzz + (3103199770705919.0/274877906944.0)*nxx*nxy - 315.0/2.0*nxy*ny*nyy*nyy*ny + (3203702005438147.0/549755813889.0)*nxy*ny*nyy*nyz*nz - 3157316358635519.0/137438953472.0*nxy*ny*nyz*nz*nzz - 5283325170155519.0/137438953472.0*nxy*ny*nyz*nz + (97188974241047.0/314146179365.0)*nxy*ny*ny - 7959776990330879.0/274877906944.0*nxy*nzz*nzz*nzz - 161865.0/8.0*nxy*nzz*nzz + (69578470195453.0/54975581389.0)*nxy*nzz - 603013408357303.0/549755813887.0*nxy);  // power=9
        out[17] = T((1575.0/2.0)*nxx*nxx*nxy*ny*ny + (331875.0/8.0)*nxx*nxy*ny*nyy*ny - 5997277583769601.0/17179869184.0*nxx*nxy*ny*nyz*nz - 2900649113026559.0/549755813888.0*nxx*nxy*ny*ny + (6633147491942405.0/549755813888.0)*nxx*nxy*nzz*nzz + (221265.0/4.0)*nxx*nxy*nzz - 2025*nxx*nxy - 87525.0/2.0*nxy*ny*nyy*nyy*ny + (7773461309030399.0/68719476736.0)*nxy*ny*nyy*nyz*nz + (6171610306314237.0/137438953472.0)*nxy*ny*nyy*ny + (5805936790732799.0/68719476736.0)*nxy*ny*nyz*nz*nzz - 6837244337848317.0/137438953472.0*nxy*ny*nyz*nz - 8170058589143037.0/549755813888.0*nxy*ny*ny - 4974860618956799.0/68719476736.0*nxy*nzz*nzz*nzz + (5944320637009919.0/137438953472.0)*nxy*nzz*nzz - 2950127136276481.0/549755813888.0*nxy*nzz + 2025*nxy);  // power=9
        out[18] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyy - 579150*nxx*nxx*nyy*nzz - 5306517993553919.0/137438953472.0*nxx*nxx*nyy + (5306517993553921.0/137438953472.0)*nxx*nxx*nzz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nyy + (4974860618956799.0/17179869184.0)*nxx*nyy*nyy*nzz + (6633147491942403.0/137438953472.0)*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz - 6.0/753950830475.0*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + (5306517993553919.0/549755813888.0)*nyy*nyy*nyy - 19305*nyy*nyy*nzz - 1836871613153279.0/274877906944.0*nyy*nyy - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (3673743226306559.0/274877906944.0)*nyy*nzz + 810*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz);  // power=9
        out[19] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nyy - 7075357324738559.0/549755813888.0*nxx*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyy - 1447875.0/2.0*nxx*nxx*nyy*nzz - 173745.0/2.0*nxx*nxx*nyy + (6633147491942401.0/137438953472.0)*nxx*nxx*nzz + 8910*nxx*nxx + (7462290928435199.0/17179869184.0)*nxx*nyy*nyy*nzz + (7959776990330877.0/274877906944.0)*nxx*nyy*nyy + (7462290928435199.0/17179869184.0)*nxx*nyy*nzz*nzz - 3979888495165437.0/68719476736.0*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy - 7959776990330879.0/274877906944.0*nxx*nzz*nzz - 2040968459058460.0/916259689813.0*nxx*nzz - 2025*nxx - 7959776990330879.0/274877906944.0*nyy*nyy*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + (31185.0/2.0)*nyy*nzz - 945*nyy + (4455.0/2.0)*nzz*nzz - 810*nzz + 105);  // power=9
        out[20] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nyz - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyz - 675675.0/2.0*nxx*nxx*nyz*nz*nz - 6633147491942401.0/68719476736.0*nxx*nxx*nyz + (4974860618956799.0/34359738368.0)*nxx*nyy*nyz*nz*nz + (3979888495165439.0/137438953472.0)*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz + (331657374597121.0/34359738368.0)*nxx*nyz*nz*nz + (4898324301742081.0/274877906944.0)*nxx*nyz - 5306517993553919.0/549755813888.0*nyy*nyz*nz*nz - 5306517993553919.0/549755813888.0*nyz*nz*nzz*nz + 4455*nyz*nz*nz - 945*nyz);  // power=9
        out[21] = T(-2025.0/2.0*nxx*nxx*nxx*nxz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz + (675675.0/2.0)*nxx*nxy*ny*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nxy*ny*nyy*nz - 6633147491942401.0/34359738368.0*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz - 5306517993553919.0/137438953472.0*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz + (7075357324738559.0/549755813888.0)*nxx*nxz*nz*nz + (5566277615615999.0/549755813888.0)*nxy*ny*nyy*nyy*nz + (4974860618956799.0/34359738368.0)*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz - 38610*nxy*ny*nyy*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (5306517993553919.0/137438953472.0)*nxy*nyz*nz*nz - 6735195914895359.0/274877906944.0*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 2970*nxz*nz*nz + (2025.0/2.0)*nxz);  // power=9
        out[22] = T(290205*nxx*nxx*nxy*ny*ny - 289575*nxx*nxx*nxy*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nxy - 225945*nxx*nxy*ny*nyy*ny - 349335*nxx*nxy*ny*nyz*nz - 19845*nxx*nxy*ny*ny + (1138995.0/4.0)*nxx*nxy*nzz*nzz + (4770763773050881.0/34359738368.0)*nxx*nxy*nzz + (6178568153333761.0/549755813888.0)*nxx*nxy - 5164268676710399.0/549755813888.0*nxy*ny*nyy*nyy*ny + (5064539536097279.0/34359738368.0)*nxy*ny*nyy*nyz*nz + (192375.0/4.0)*nxy*ny*nyy*ny + 55440*nxy*ny*nyz*nz*nzz - 8423633458298879.0/137438953472.0*nxy*ny*nyz*nz - 56385.0/4.0*nxy*ny*ny - 405405.0/4.0*nxy*nzz*nzz*nzz + (6327002223083521.0/274877906944.0)*nxy*nzz*nzz - 2421330762792961.0/274877906944.0*nxy*nzz + (911249999997266.0/999999999997.0)*nxy);  // power=9
        out[23] = T(-2025.0/2.0*nxx*nxx*nxx*nxz + 581040*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz - 5306517993553921.0/137438953472.0*nxx*nxx*nxz - 7713933062307839.0/17179869184.0*nxx*nxy*ny*nyy*nz + (5287963734835199.0/274877906944.0)*nxx*nxy*ny*nyy*nz - 1281285.0/2.0*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz - 2692686796554241.0/34359738368.0*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz + (5748727826350079.0/137438953472.0)*nxx*nxz*nz*nz + (25515.0/2.0)*nxx*nxz - 33075.0/2.0*nxy*ny*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nxy*ny*nyy*nyy*nz + (7774234403143679.0/68719476736.0)*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz + (169695.0/2.0)*nxy*ny*nyy*nz - 4156153952993279.0/274877906944.0*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (186165.0/2.0)*nxy*nyz*nz*nz - 6085796859740159.0/274877906944.0*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 16065.0/2.0*nxz*nz*nz + (404999999999999.0/1000000000000.0)*nxz);  // power=9
        out[24] = T(579150*nxx*nxx*nyy*nyz - 6633147491942401.0/34359738368.0*nxx*nxx*nyz*nz*nz - 5306517993553921.0/68719476736.0*nxx*nxx*nyz - 7462290928435199.0/17179869184.0*nxx*nyy*nyy*nyz - 4974860618956799.0/17179869184.0*nxx*nyy*nyz*nz*nz + (3979888495165439.0/34359738368.0)*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz + (5306517993553923.0/137438953472.0)*nxx*nyz*nz*nz + (7959776990330879.0/274877906944.0)*nyy*nyy*nyz + 19305*nyy*nyz*nz*nz - 17820*nyy*nyz - 5306517993553919.0/549755813888.0*nyz*nz*nzz*nz - 1.0/1000000000000.0*nyz*nz*nz + 1620*nyz);  // power=9
        out[25] = T(270270*nxx*nxx*nyy*ny - 62370*nxx*nxx*ny - 405405.0/2.0*nxx*nyy*nyy*ny - 405405.0/2.0*nxx*nyy*nyz*nz + (93555.0/2.0)*nxx*nyy*ny + (93555.0/2.0)*nxx*nyz*nz + 8505*nxx*ny + (31185.0/2.0)*nyy*nyy*ny + (31185.0/2.0)*nyy*nyz*nz - 19845.0/2.0*nyy*ny - 8505.0/2.0*nyz*nz + 945*ny);  // power=8
        out[26] = T(270270*nxx*nxx*nyz*nz - 20790*nxx*nxx*ny - 405405.0/2.0*nxx*nyy*nyz*nz + (31185.0/2.0)*nxx*nyy*ny - 405405.0/2.0*nxx*nyz*nz*nzz + (31185.0/2.0)*nxx*nyz*nz + 2835*nxx*ny + (31185.0/2.0)*nyy*nyz*nz - 2835.0/2.0*nyy*ny + (31185.0/2.0)*nyz*nz*nzz - 14175.0/2.0*nyz*nz + 315*ny);  // power=8
        out[27] = T((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nxy*ny - 14175.0/2.0*nxx*nxx*nxz*nz + (19845.0/2.0)*nxx*nxy*ny*nyy - 14175.0/2.0*nxx*nxy*ny*nyy + 296730*nxx*nxy*nyz*nz + 70875*nxx*nxy*nyz*nz - 27405*nxx*nxy*ny - 20790*nxx*nxz*nz + (4725.0/2.0)*nxx*nx + (19845.0/2.0)*nxy*ny*nyy*nyy - 3780*nxy*ny*nyy*nyy - 166320*nxy*ny*nyy*nzz + (155925.0/2.0)*nxy*ny*nyy*nzz - 945*nxy*ny*nyy + 26460*nxy*nyz*nz*nzz - 113400*nxy*nyz*nz*nzz - 1890*nxy*nyz*nz + (8505.0/2.0)*nxy*ny + 7560*nxz*nz*nzz*nzz + 1890*nxz*nz - 630*nx);  // power=8
    } break;

    case 20: {
        out[0] = T(2100*nxx*nxz - 1575*nxy*nyz - 1575*nxz*nz*nz);  // power=5
        out[1] = T(18900*nxx*nxx*nz - 14175*nxx*nyy*nz - 14175*nxx*nzz*nz - 6300*nxx*nz + 1575*nyy*nz + 1575*nzz*nz);  // power=6
        out[2] = T(22050*nxx*nxy*nz - 11025*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[3] = T(-225*nxx*nxx*nx + 1125*nxx*nxy*ny + 20025*nxx*nxz*nz - 2100*nxx*nx + 1350*nxy*ny*nyy - 11475*nxy*nyz*nz - 7050618313113599.0/549755813888.0*nxz*nz*nzz + 3150*nxz*nz + 225*nx);  // power=6
        out[4] = T(23625*nxx*nxx*nxz - 33600*nxx*nxy*nyz - 29400*nxx*nxz*nz*nz - 10500*nxx*nxz + (6825.0/2.0)*nxy*ny*nyy*nz + 11025*nxy*nyz*nz*nz + 4200*nxy*nyz + (15225.0/2.0)*nxz*nz*nzz*nz + 1575*nxz);  // power=7
        out[5] = T((23625.0/2.0)*nxx*nxx*nxz + 2100*nxx*nxy*nyz - 32025*nxx*nxz*nz*nz - 2100*nxx*nxz - 19425.0/2.0*nxy*ny*nyy*nz + (14175.0/2.0)*nxy*nyz*nz*nz + (5775.0/2.0)*nxy*nyz + 16800*nxz*nz*nzz*nz - 14175.0/2.0*nxz*nz*nz + (2625.0/2.0)*nxz);  // power=7
        out[6] = T(69300*nxx*nxx*nyz - 51975*nxx*nyy*nyz - 51975*nxx*nyz*nz*nz - 9450*nxx*nyz + 4725*nyy*nyz + 4725*nyz*nz*nz - 1050*nyz);  // power=7
        out[7] = T(69300*nxx*nxx*nzz - 6300*nxx*nxx - 51975*nxx*nyy*nzz + 4725*nxx*nyy - 51975*nxx*nzz*nzz - 4725*nxx*nzz + 2625*nxx + 4725*nyy*nzz + 4725*nzz*nzz - 1050*nzz - 525);  // power=7
        out[8] = T(-1575*nxx*nxx*nxy + 1575*nxx*nxy*ny*ny + 80325*nxx*nxy*nzz - 6300*nxx*nxy + 3150*nxy*ny*nyy*ny - 36225*nxy*ny*nyz*nz - 39375*nxy*nzz*nzz + 9450*nxy*nzz + 525*nxy);  // power=7
        out[9] = T(90090*nxx*nxx*nxx*nz - 135135.0/2.0*nxx*nxx*nyy*nz - 135135.0/2.0*nxx*nxx*nzz*nz - 83160*nxx*nxx*nz + 31185*nxx*nyy*nz + 31185*nxx*nzz*nz + 17010*nxx*nz - 2835.0/2.0*nyy*nz - 2835.0/2.0*nzz*nz - 420*nz);  // power=8
        out[10] = T(-8505.0/2.0*nxx*nxx*nxy*nz + (213255.0/2.0)*nxx*nxy*ny*nyz + (14175.0/2.0)*nxx*nxy*nzz*nz - 20790*nxx*nxy*nz - 93555.0/2.0*nxy*ny*nyy*nyz - 70875.0/2.0*nxy*ny*nyz*nz*nz + 21735*nxy*ny*nyz + 11340*nxy*nzz*nzz*nz - 2835.0/2.0*nxy*nz);  // power=8
        out[11] = T((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nxy*ny - 4725.0/2.0*nxx*nxx*nxz*nz - 4725.0/2.0*nxx*nxy*ny*nyy + 23625*nxx*nxy*nyz*nz + 90090*nxx*nxz*nz*nzz - 41580*nxx*nxz*nz + 1890*nxx*nx - 1260*nxy*ny*nyy*nyy + (51975.0/2.0)*nxy*ny*nyy*nzz - 37800*nxy*nyz*nz*nzz - 135135.0/2.0*nxz*nz*nzz*nzz + 2520*nxz*nz*nzz*nzz + (114345.0/2.0)*nxz*nz*nzz - 19845.0/2.0*nxz*nz + (315.0/2.0)*nx);  // power=8
        out[12] = T(270270*nxx*nxx*nxy*nz - 405405.0/2.0*nxx*nxy*ny*nyz - 405405.0/2.0*nxx*nxy*nzz*nz - 114345*nxx*nxy*nz + (93555.0/2.0)*nxy*ny*nyz + (93555.0/2.0)*nxy*nzz*nz + 2835*nxy*nz);  // power=8
        out[13] = T(270270*nxx*nxx*nxz*nz - 20790*nxx*nxx*nx - 405405.0/2.0*nxx*nxy*nyz*nz + (31185.0/2.0)*nxx*nxy*ny - 405405.0/2.0*nxx*nxz*nz*nzz - 197505.0/2.0*nxx*nxz*nz + 13230*nxx*nx + (93555.0/2.0)*nxy*nyz*nz - 8505.0/2.0*nxy*ny + (93555.0/2.0)*nxz*nz*nzz - 2835.0/2.0*nxz*nz - 1260*nx);  // power=8
        out[14] = T((945.0/2.0)*nxx*nxx*nxx*nx - 2835*nxx*nxx*nxy*ny - 14175.0/2.0*nxx*nxx*nxz*nz + (19845.0/2.0)*nxx*nxy*ny*nyy - 14175.0/2.0*nxx*nxy*ny*nyy + 296730*nxx*nxy*nyz*nz + 70875*nxx*nxy*nyz*nz - 27405*nxx*nxy*ny - 20790*nxx*nxz*nz + (4725.0/2.0)*nxx*nx + (19845.0/2.0)*nxy*ny*nyy*nyy - 3780*nxy*ny*nyy*nyy - 166320*nxy*ny*nyy*nzz + (155925.0/2.0)*nxy*ny*nyy*nzz - 945*nxy*ny*nyy + 26460*nxy*nyz*nz*nzz - 113400*nxy*nyz*nz*nzz - 1890*nxy*nyz*nz + (8505.0/2.0)*nxy*ny + 7560*nxz*nz*nzz*nzz + 1890*nxz*nz - 630*nx);  // power=8
        out[15] = T(540540*nxx*nxx*nyz*nz - 41580*nxx*nxx*ny - 405405*nxx*nyy*nyz*nz + 31185*nxx*nyy*ny - 405405*nxx*nyz*nz*nzz + 31185*nxx*nyz*nz + 5670*nxx*ny + 31185*nyy*nyz*nz - 2835*nyy*ny + 31185*nyz*nz*nzz - 14175*nyz*nz + 630*ny);  // power=8
        out[16] = T((6633147491942401.0/137438953472.0)*nxx*nxx*nxx*nxz + (2025.0/8.0)*nxx*nxx*nxx*nxz - 4974860618956799.0/137438953472.0*nxx*nxx*nxy*nyz - 2025*nxx*nxx*nxy*nyz - 5596718196326401.0/17179869184.0*nxx*nxx*nxz*nz*nz - 8775.0/8.0*nxx*nxx*nxz*nz*nz + (6.0/753950830475.0)*nxx*nxx*nxz - 3304977334267190.0/687194767359.0*nxx*nxy*ny*nyy*nz + (7462290928435199.0/34359738368.0)*nxx*nxy*nyz*nz*nz + (8998815478579199.0/549755813888.0)*nxx*nxy*nyz*nz*nz - 5306517993553921.0/274877906944.0*nxx*nxy*nyz + (1061775.0/4.0)*nxx*nxz*nz*nzz*nz - 540*nxx*nxz*nz*nzz*nz + (7075357324738561.0/274877906944.0)*nxx*nxz*nz*nz - 2040968459058460.0/916259689813.0*nxx*nxz - 1391569403906531.0/549755813889.0*nxy*ny*nyy*nyy*nz + (139725.0/8.0)*nxy*ny*nyy*nzz*nz - 8460741975736319.0/549755813888.0*nxy*nyz*nz*nzz*nz - 173745.0/4.0*nxy*nyz*nz*nz + (4974860618949560.0/687194767359.0)*nxy*nyz - 4974860618956799.0/137438953472.0*nxz*nz*nzz*nzz*nz + 810*nxz*nz*nzz*nzz*nz - 1326629498388483.0/549755813888.0*nxz*nz*nzz*nz - 359210448794461.0/967570232443.0*nxz*nz*nz + (14843406974912.0/879609302217.0)*nxz);  // power=9
        out[17] = T(-10125.0/8.0*nxx*nxx*nxx*nxz + (4725.0/2.0)*nxx*nxx*nxy*nyz + 10125*nxx*nxx*nxy*nyz + (3768833802234515.0/687194767359.0)*nxx*nxx*nxz*nz*nz + 66150*nxx*nxy*ny*nyy*nz + (192375.0/8.0)*nxx*nxy*ny*nyy*nz - 5258972705587199.0/17179869184.0*nxx*nxy*nyz*nz*nz - 5624259674111999.0/68719476736.0*nxx*nxy*nyz*nz*nz + (6767665867653121.0/137438953472.0)*nxx*nxy*nyz + (6633147491942401.0/137438953472.0)*nxx*nxz*nz*nzz*nz + 2700*nxx*nxz*nz*nzz*nz - 3537678662369279.0/274877906944.0*nxx*nxz*nz*nz - 667953313873313.0/879609302220.0*nxx*nxz - 165375.0/8.0*nxy*ny*nyy*nyy*nz + (50625.0/4.0)*nxy*ny*nyy*nyy*nz + (1420875.0/8.0)*nxy*ny*nyy*nzz*nz - 698625.0/8.0*nxy*ny*nyy*nzz*nz - 48735.0/4.0*nxy*ny*nyy*nz - 18900*nxy*nyz*nz*nzz*nz + 76950*nxy*nyz*nz*nzz*nz + (4786998749429763.0/274877906944.0)*nxy*nyz*nz*nz - 7867005696737281.0/549755813888.0*nxy*nyz - 4974860618956799.0/137438953472.0*nxz*nz*nzz*nzz*nz - 4050*nxz*nz*nzz*nzz*nz + (2819087684075519.0/68719476736.0)*nxz*nz*nzz*nz - 3984527059845119.0/274877906944.0*nxz*nz*nz + 2025*nxz);  // power=9
        out[18] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nyy*nyz - 579150*nxx*nxx*nyz*nz*nz + (5306517993553921.0/68719476736.0)*nxx*nxx*nyz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nyz + (4974860618956799.0/17179869184.0)*nxx*nyy*nyz*nz*nz - 5306517993553923.0/137438953472.0*nxx*nyy*nyz + (7462290928435199.0/17179869184.0)*nxx*nyz*nz*nzz*nz - 3979888495165439.0/34359738368.0*nxx*nyz*nz*nz + (5306517993553919.0/549755813888.0)*nyy*nyy*nyz - 19305*nyy*nyz*nz*nz + (1.0/1000000000000.0)*nyy*nyz - 7959776990330879.0/274877906944.0*nyz*nz*nzz*nz + 17820*nyz*nz*nz - 1620*nyz);  // power=9
        out[19] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nyz - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nyz - 1447875.0/2.0*nxx*nxx*nyz*nz*nz - 5306517993553923.0/274877906944.0*nxx*nxx*nyz + (7462290928435199.0/17179869184.0)*nxx*nyy*nyz*nz*nz - 3979888495165441.0/137438953472.0*nxx*nyy*nyz + (7462290928435199.0/17179869184.0)*nxx*nyz*nz*nzz*nz - 2984916371374079.0/34359738368.0*nxx*nyz*nz*nz + 17820*nxx*nyz - 7959776990330879.0/274877906944.0*nyy*nyz*nz*nz + 4455*nyy*nyz - 7959776990330879.0/274877906944.0*nyz*nz*nzz*nz + 17820*nyz*nz*nz - 2565*nyz);  // power=9
        out[20] = T((6633147491942401.0/34359738368.0)*nxx*nxx*nxx*nzz - 7075357324738559.0/549755813888.0*nxx*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nxx*nyy*nzz + (5306517993553919.0/549755813888.0)*nxx*nxx*nyy - 675675.0/2.0*nxx*nxx*nzz*nzz - 6633147491942401.0/137438953472.0*nxx*nxx*nzz + 8910*nxx*nxx + (4974860618956799.0/34359738368.0)*nxx*nyy*nzz*nzz - 4455.0/2.0*nxx*nyy + (4974860618956799.0/34359738368.0)*nxx*nzz*nzz*nzz - 5306517993553923.0/274877906944.0*nxx*nzz*nzz + (2143016882012161.0/137438953472.0)*nxx*nzz - 2025*nxx - 5306517993553919.0/549755813888.0*nyy*nzz*nzz + (4455.0/2.0)*nyy*nzz - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 1755*nzz + 105);  // power=9
        out[21] = T((6122905377175370.0/916259689813.0)*nxx*nxy*ny*nyy*ny + 217350*nxx*nxy*ny*nyz*nz - 5207561947054079.0/274877906944.0*nxx*nxy*ny*ny - 6633147491942401.0/34359738368.0*nxx*nxy*nzz*nzz + (5306517993553919.0/137438953472.0)*nxx*nxy*nzz + (2025.0/2.0)*nxx*nxy + (3673743226306559.0/549755813888.0)*nxy*ny*nyy*nyy*ny - 3910310024970239.0/34359738368.0*nxy*ny*nyy*nyz*nz - 3104999999999999.0/1000000000000.0*nxy*ny*nyy*ny + (6679533138739199.0/274877906944.0)*nxy*ny*nyz*nz*nzz + 27540*nxy*ny*nyz*nz + (2632499999997368.0/999999999999.0)*nxy*ny*ny + (4974860618956799.0/34359738368.0)*nxy*nzz*nzz*nzz - 7296462241136639.0/68719476736.0*nxy*nzz*nzz + (28755.0/2.0)*nxy*nzz - 2025.0/2.0*nxy);  // power=9
        out[22] = T(-2025.0/2.0*nxx*nxx*nxx*nxz + 292410*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz - 289575*nxx*nxx*nxz*nz*nz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz + (5306517993553921.0/137438953472.0)*nxx*nxx*nxz - 8217217330053119.0/34359738368.0*nxx*nxy*ny*nyy*nz + (38475.0/2.0)*nxx*nxy*ny*nyy*nz - 5315795122913279.0/17179869184.0*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz - 4879770043023361.0/137438953472.0*nxx*nxy*nyz + (1254825.0/4.0)*nxx*nxz*nz*nzz*nz + 2160*nxx*nxz*nz*nzz*nz + (2476375063657856.0/769658139443.0)*nxx*nxz*nz*nz - 3924225719009281.0/274877906944.0*nxx*nxz - 99225.0/4.0*nxy*ny*nyy*nyy*nz + 10125*nxy*ny*nyy*nyy*nz + 169695*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz + (161595.0/4.0)*nxy*ny*nyy*nz - 22680*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (1280243851591681.0/137438953472.0)*nxy*nyz*nz*nz - 4174708211711999.0/549755813888.0*nxy*nyz - 4974860618956799.0/68719476736.0*nxz*nz*nzz*nzz*nz - 3240*nxz*nz*nzz*nzz*nz + (5306517993553917.0/137438953472.0)*nxz*nz*nzz*nz - 8256645129830397.0/549755813888.0*nxz*nz*nz + (1725546060837821.0/549755813887.0)*nxz);  // power=9
        out[23] = T(630*nxx*nxx*nxy*ny*ny + 579150*nxx*nxx*nxy*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nxy + 1260*nxx*nxy*ny*nyy*ny - 7862753679114239.0/17179869184.0*nxx*nxy*ny*nyz*nz + (7959776990330879.0/274877906944.0)*nxx*nxy*ny*ny - 656370*nxx*nxy*nzz*nzz - 1199068969697281.0/34359738368.0*nxx*nxy*nzz + (25515.0/2.0)*nxx*nxy + 630*nxy*ny*nyy*nyy*ny - 6407404010864639.0/274877906944.0*nxy*ny*nyy*nyz*nz + (3157316358635519.0/34359738368.0)*nxy*ny*nyz*nz*nzz + (133785.0/2.0)*nxy*ny*nyz*nz - 11385.0/2.0*nxy*ny*ny + (7959776990330879.0/68719476736.0)*nxy*nzz*nzz*nzz - 5937362789990950.0/999556025251.0*nxy*nzz*nzz - 534362651098651.0/879609302220.0*nxy*nzz - 1215.0/2.0*nxy);  // power=9
        out[24] = T(579150*nxx*nxx*nyy*nzz - 5306517993553921.0/137438953472.0*nxx*nxx*nyy - 6633147491942401.0/34359738368.0*nxx*nxx*nzz*nzz + (5306517993553919.0/137438953472.0)*nxx*nxx*nzz - 7462290928435199.0/17179869184.0*nxx*nyy*nyy*nzz + (7959776990330879.0/274877906944.0)*nxx*nyy*nyy - 4974860618956799.0/17179869184.0*nxx*nyy*nzz*nzz + (3979888495165437.0/68719476736.0)*nxx*nyy*nzz + (4974860618956799.0/34359738368.0)*nxx*nzz*nzz*nzz - 6633147491942403.0/137438953472.0*nxx*nzz*nzz + (6.0/753950830475.0)*nxx*nzz + (7959776990330879.0/274877906944.0)*nyy*nyy*nzz - 4455.0/2.0*nyy*nyy + 19305*nyy*nzz*nzz - 3673743226306559.0/274877906944.0*nyy*nzz + 810*nyy - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 810*nzz);  // power=9
        out[25] = T(270270*nxx*nxx*nyy*nz - 20790*nxx*nxx*nz - 405405.0/2.0*nxx*nyy*nyy*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (31185.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz + 2835*nxx*nz + (31185.0/2.0)*nyy*nyy*nz + (31185.0/2.0)*nyy*nzz*nz - 14175.0/2.0*nyy*nz - 2835.0/2.0*nzz*nz + 315*nz);  // power=8
        out[26] = T(270270*nxx*nxx*nzz*nz - 62370*nxx*nxx*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (93555.0/2.0)*nxx*nyy*nz - 405405.0/2.0*nxx*nzz*nzz*nz + (93555.0/2.0)*nxx*nzz*nz + 8505*nxx*nz + (31185.0/2.0)*nyy*nzz*nz - 8505.0/2.0*nyy*nz + (31185.0/2.0)*nzz*nzz*nz - 19845.0/2.0*nzz*nz + 945*nz);  // power=8
        out[27] = T(-25515.0/2.0*nxx*nxx*nxy*nz + (42525.0/2.0)*nxx*nxy*ny*nyz + (583065.0/2.0)*nxx*nxy*nzz*nz - 62370*nxx*nxy*nz + 34020*nxy*ny*nyy*nyz - 269325.0/2.0*nxy*ny*nyz*nz*nz - 337365.0/2.0*nxy*nzz*nzz*nz + 93555*nxy*nzz*nz - 8505.0/2.0*nxy*nz);  // power=8
    } break;

    case 21: {
        out[0] = T(-1575*nxx*nyz + 2100*nyy*nyz - 1575*nyz*nz*nz);  // power=5
        out[1] = T(-11025*nxx*nxy*nz + 22050*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[2] = T(-14175*nxx*nyy*nz + 1575*nxx*nz + 18900*nyy*nyy*nz - 14175*nyy*nzz*nz - 6300*nyy*nz + 1575*nzz*nz);  // power=6
        out[3] = T(-14175*nxx*nyz*nz + 1575*nxx*ny + 18900*nyy*nyz*nz - 2100*nyy*ny - 14175*nyz*nz*nzz + 4725*nyz*nz);  // power=6
        out[4] = T(-17325*nxx*nxx*nyz + (63525.0/2.0)*nxx*nyy*nyz - 17325.0/2.0*nxx*nyz*nz*nz + 3150*nxx*nyz - 11550*nyy*nyy*nyz - 5775.0/2.0*nyy*nyz*nz*nz + 8400*nyy*nyz + (17325.0/2.0)*nyz*nz*nzz*nz - 6300*nyz*nz*nz - 525*nyz);  // power=7
        out[5] = T(-17325.0/2.0*nxx*nxx*nyz + (5775.0/2.0)*nxx*nyy*nyz + (17325.0/2.0)*nxx*nyz*nz*nz + 1575*nxx*nyz + 11550*nyy*nyy*nyz - 63525.0/2.0*nyy*nyz*nz*nz - 2100*nyy*nyz + 17325*nyz*nz*nzz*nz - 7875*nyz*nz*nz + 1575*nyz);  // power=7
        out[6] = T(-1575*nxx*nxx*nxz - 49350*nxx*nxy*nyz + 1575*nxx*nxz*nz*nz + 4725*nxx*nxz + 73500*nxy*ny*nyy*nz - 44625*nxy*nyz*nz*nz - 15225*nxy*nyz + 3150*nxz*nz*nzz*nz + 525*nxz);  // power=7
        out[7] = T(2100*nxx*nxx*nxy - 2100*nxx*nxy*ny*ny - 66675*nxx*nxy*nzz + 4725*nxx*nxy - 4200*nxy*ny*nyy*ny + 48300*nxy*ny*nyz*nz - 68775*nxy*nzz*nzz + 42525*nxy*nzz - 3150*nxy);  // power=7
        out[8] = T(-51975*nxx*nyy*nzz + 4725*nxx*nyy + 4725*nxx*nzz + 69300*nyy*nyy*nzz - 6300*nyy*nyy - 51975*nyy*nzz*nzz - 4725*nyy*nzz + 2625*nyy + 4725*nzz*nzz - 1050*nzz - 525);  // power=7
        out[9] = T(-135135.0/2.0*nxx*nxx*nxy*nz + 90090*nxx*nxy*ny*nyz - 135135.0/2.0*nxx*nxy*nzz*nz + (93555.0/2.0)*nxx*nxy*nz - 20790*nxy*ny*nyz + (31185.0/2.0)*nxy*nzz*nz - 5670*nxy*nz);  // power=8
        out[10] = T(-135135.0/2.0*nxx*nyy*nyy*nz + 31185*nxx*nyy*nz - 2835.0/2.0*nxx*nz + 90090*nyy*nyy*nyy*nz - 135135.0/2.0*nyy*nyy*nzz*nz - 83160*nyy*nyy*nz + 31185*nyy*nzz*nz + 17010*nyy*nz - 2835.0/2.0*nzz*nz - 420*nz);  // power=8
        out[11] = T(-135135.0/2.0*nxx*nyz*nz*nzz + 31185*nxx*nyz*nz - 2835.0/2.0*nxx*ny + 90090*nyy*nyz*nz*nzz - 41580*nyy*nyz*nz + 1890*nyy*ny - 135135.0/2.0*nyz*nz*nzz*nzz + 62370*nyz*nz*nzz - 25515.0/2.0*nyz*nz + 315*ny);  // power=8
        out[12] = T(-405405.0/2.0*nxx*nxx*nyy*nz + (31185.0/2.0)*nxx*nxx*nz + 270270*nxx*nyy*nyy*nz - 405405.0/2.0*nxx*nyy*nzz*nz + (31185.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz - 14175.0/2.0*nxx*nz - 20790*nyy*nyy*nz + (31185.0/2.0)*nyy*nzz*nz + 2835*nyy*nz - 2835.0/2.0*nzz*nz + 315*nz);  // power=8
        out[13] = T(-405405.0/2.0*nxx*nxx*nyz*nz + (31185.0/2.0)*nxx*nxx*ny + 270270*nxx*nyy*nyz*nz - 20790*nxx*nyy*ny - 405405.0/2.0*nxx*nyz*nz*nzz + 124740*nxx*nyz*nz - 14175.0/2.0*nxx*ny - 20790*nyy*nyz*nz + 1890*nyy*ny + (31185.0/2.0)*nyz*nz*nzz - 14175.0/2.0*nyz*nz + 315*ny);  // power=8
        out[14] = T(-405405.0/2.0*nxx*nyy*nyz*nz + (31185.0/2.0)*nxx*nyy*ny + (93555.0/2.0)*nxx*nyz*nz - 8505.0/2.0*nxx*ny + 270270*nyy*nyy*nyz*nz - 20790*nyy*nyy*ny - 405405.0/2.0*nyy*nyz*nz*nzz - 197505.0/2.0*nyy*nyz*nz + 13230*nyy*ny + (93555.0/2.0)*nyz*nz*nzz - 2835.0/2.0*nyz*nz - 1260*ny);  // power=8
        out[15] = T(945*nxx*nxx*nxx*nx - 5670*nxx*nxx*nxy*ny - 14175*nxx*nxx*nxz*nz - 26460*nxx*nxy*ny*nyy - 14175*nxx*nxy*ny*nyy - 475965*nxx*nxy*nyz*nz + 141750*nxx*nxy*nyz*nz + 48825*nxx*nxy*ny + 31185*nxx*nxz*nz - 4095*nxx*nx - 26460*nxy*ny*nyy*nyy - 7560*nxy*ny*nyy*nyy + 443520*nxy*ny*nyy*nzz + 155925*nxy*ny*nyy*nzz + 2520*nxy*ny*nyy - 70560*nxy*nyz*nz*nzz - 226800*nxy*nyz*nz*nzz - 67725*nxy*nyz*nz - 4725*nxy*ny + 15120*nxz*nz*nzz*nzz + 1575*nxz*nz + 945*nx);  // power=8
        out[16] = T(-4974860618956799.0/137438953472.0*nxx*nxx*nxx*nyz + (6633147491942401.0/137438953472.0)*nxx*nxx*nyy*nyz + (6218575773695999.0/34359738368.0)*nxx*nxx*nyz*nz*nz - 5306517993553921.0/549755813888.0*nxx*nxx*nyz - 289575*nxx*nyy*nyz*nz*nz + (5306517993553921.0/137438953472.0)*nxx*nyy*nyz + (6218575773695999.0/34359738368.0)*nxx*nyz*nz*nzz*nz - 135135*nxx*nyz*nz*nz + (3673743226306559.0/274877906944.0)*nxx*nyz + (6633147491942401.0/137438953472.0)*nyy*nyz*nz*nzz*nz - 3537678662369279.0/274877906944.0*nyy*nyz*nz*nz - 1.0/1000000000000.0*nyy*nyz - 4974860618956799.0/137438953472.0*nyz*nz*nzz*nzz*nz + (1989944247582719.0/68719476736.0)*nyz*nz*nzz*nz - 2449162150871039.0/549755813888.0*nyz*nz*nz);  // power=9
        out[17] = T(-4974860618956799.0/137438953472.0*nxx*nyy*nyy*nyz + (7462290928435199.0/34359738368.0)*nxx*nyy*nyz*nz*nz - 5306517993553921.0/274877906944.0*nxx*nyy*nyz - 4974860618956799.0/137438953472.0*nxx*nyz*nz*nzz*nz - 5306517993553921.0/274877906944.0*nxx*nyz*nz*nz + 4455*nxx*nyz + (6633147491942401.0/137438953472.0)*nyy*nyy*nyy*nyz - 5596718196326401.0/17179869184.0*nyy*nyy*nyz*nz*nz + (6.0/753950830475.0)*nyy*nyy*nyz + (1061775.0/4.0)*nyy*nyz*nz*nzz*nz + (7075357324738561.0/274877906944.0)*nyy*nyz*nz*nz - 2040968459058460.0/916259689813.0*nyy*nyz - 4974860618956799.0/137438953472.0*nyz*nz*nzz*nzz*nz - 5.0/837723144972.0*nyz*nz*nzz*nz - 222651104624437.0/99955602525.0*nyz*nz*nz + 270*nyz);  // power=9
        out[18] = T(-3037499999996962.0/999999999999.0*nxx*nxx*nxx*nxz - 9450*nxx*nxx*nxy*nyz + (6679533138739199.0/274877906944.0)*nxx*nxx*nxy*nyz + (26325.0/2.0)*nxx*nxx*nxz*nz*nz - 264600*nxx*nxy*ny*nyy*nz + (115425.0/2.0)*nxx*nxy*ny*nyy*nz + (1001025.0/2.0)*nxx*nxy*nyz*nz*nz - 196425*nxx*nxy*nyz*nz*nz - 4912239995781121.0/68719476736.0*nxx*nxy*nyz + 6480*nxx*nxz*nz*nzz*nz - 7959776990330879.0/274877906944.0*nxx*nxz*nz*nz + (4119045435563332.0/549755813889.0)*nxx*nxz + (165375.0/2.0)*nxy*ny*nyy*nyy*nz + (8349416423423999.0/274877906944.0)*nxy*ny*nyy*nyy*nz - 6517183374950399.0/8589934592.0*nxy*ny*nyy*nzz*nz - 419175.0/2.0*nxy*ny*nyy*nzz*nz + (5371457899069439.0/137438953472.0)*nxy*ny*nyy*nz + 75600*nxy*nyz*nz*nzz*nz + 184680*nxy*nyz*nz*nzz*nz + (150255.0/2.0)*nxy*nyz*nz*nz + 39420*nxy*nyz - 9720*nxz*nz*nzz*nzz*nz - 2835.0/2.0*nxz*nz*nz - 2565*nxz);  // power=9
        out[19] = T(-3037499999996962.0/999999999999.0*nxx*nxx*nxx*nxz - 5234620241018879.0/34359738368.0*nxx*nxx*nxy*nyz + (6679533138739199.0/274877906944.0)*nxx*nxx*nxy*nyz + (26325.0/2.0)*nxx*nxx*nxz*nz*nz + (5306517993553919.0/549755813888.0)*nxx*nxx*nxz + (8646284562923521.0/34359738368.0)*nxx*nxy*ny*nyy*nz + (115425.0/2.0)*nxx*nxy*ny*nyy*nz + (5884019296174079.0/17179869184.0)*nxx*nxy*nyz*nz*nz - 196425*nxx*nxy*nyz*nz*nz - 2495547797667841.0/34359738368.0*nxx*nxy*nyz + 6480*nxx*nxz*nz*nzz*nz - 19305*nxx*nxz*nz*nz + (1781208836997931.0/733007751851.0)*nxx*nxz + 66150*nxy*ny*nyy*nyy*nz + (8349416423423999.0/274877906944.0)*nxy*ny*nyy*nyy*nz - 452520*nxy*ny*nyy*nzz*nz - 419175.0/2.0*nxy*ny*nyy*nzz*nz - 30510*nxy*ny*nyy*nz + 60480*nxy*nyz*nz*nzz*nz + 184680*nxy*nyz*nz*nzz*nz + (2463077844910081.0/34359738368.0)*nxy*nyz*nz*nz + (5826037237678081.0/549755813888.0)*nxy*nyz - 9720*nxz*nz*nzz*nzz*nz - 1781208836997526.0/879609302221.0*nxz*nz*nz - 2025.0/2.0*nxz);  // power=9
        out[20] = T(-840*nxx*nxx*nxy*ny*ny - 4974860618956799.0/34359738368.0*nxx*nxx*nxy*nzz + (5306517993553919.0/549755813888.0)*nxx*nxx*nxy - 1680*nxx*nxy*ny*nyy*ny + (7701048160419841.0/34359738368.0)*nxx*nxy*ny*nyz*nz - 7075357324738559.0/549755813888.0*nxx*nxy*ny*ny + (5306517993553919.0/137438953472.0)*nxx*nxy*nzz*nzz + (5510614839459837.0/137438953472.0)*nxx*nxy*nzz - 3644999999999999.0/1000000000000.0*nxx*nxy - 840*nxy*ny*nyy*nyy*ny + (8543205347819519.0/274877906944.0)*nxy*ny*nyy*nyz*nz - 122520*nxy*ny*nyz*nz*nzz - 11970*nxy*ny*nyz*nz + 1650*nxy*ny*ny + (6301490117345279.0/34359738368.0)*nxy*nzz*nzz*nzz - 5587151156674559.0/34359738368.0*nxy*nzz*nzz + 31995*nxy*nzz - 810*nxy);  // power=9
        out[21] = T(-4974860618956799.0/34359738368.0*nxx*nyy*nyy*nzz + (5306517993553919.0/549755813888.0)*nxx*nyy*nyy + (4974860618956799.0/34359738368.0)*nxx*nyy*nzz*nzz - 4455.0/2.0*nxx*nyy - 5306517993553919.0/549755813888.0*nxx*nzz*nzz + (4455.0/2.0)*nxx*nzz + (6633147491942401.0/34359738368.0)*nyy*nyy*nyy*nzz - 7075357324738559.0/549755813888.0*nyy*nyy*nyy - 675675.0/2.0*nyy*nyy*nzz*nzz - 6633147491942401.0/137438953472.0*nyy*nyy*nzz + 8910*nyy*nyy + (4974860618956799.0/34359738368.0)*nyy*nzz*nzz*nzz - 5306517993553923.0/274877906944.0*nyy*nzz*nzz + (2143016882012161.0/137438953472.0)*nyy*nzz - 2025*nyy - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 1755*nzz + 105);  // power=9
        out[22] = T(-7462290928435199.0/34359738368.0*nxx*nxx*nyy*nyz + (7462290928435199.0/34359738368.0)*nxx*nxx*nyz*nz*nz + 289575*nxx*nyy*nyy*nyz - 289575*nxx*nyy*nyz*nz*nz - 6633147491942401.0/137438953472.0*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz - 8623091739525119.0/68719476736.0*nxx*nyz*nz*nz + (5510614839459839.0/274877906944.0)*nxx*nyz - 289575*nyy*nyy*nyz*nz*nz + (5306517993553921.0/137438953472.0)*nyy*nyy*nyz + (1254825.0/4.0)*nyy*nyz*nz*nzz*nz + (2476375063657856.0/769658139443.0)*nyy*nyz*nz*nz - 7347486452613121.0/549755813888.0*nyy*nyz - 4974860618956799.0/68719476736.0*nyz*nz*nzz*nzz*nz + (7959776990330875.0/274877906944.0)*nyz*nz*nzz*nz - 1836871613153279.0/274877906944.0*nyz*nz*nz + 1215*nyz);  // power=9
        out[23] = T(-7462290928435199.0/17179869184.0*nxx*nxx*nyy*nzz + (7959776990330879.0/274877906944.0)*nxx*nxx*nyy + (7959776990330879.0/274877906944.0)*nxx*nxx*nzz - 4455.0/2.0*nxx*nxx + 579150*nxx*nyy*nyy*nzz - 5306517993553921.0/137438953472.0*nxx*nyy*nyy - 4974860618956799.0/17179869184.0*nxx*nyy*nzz*nzz + (3979888495165437.0/68719476736.0)*nxx*nyy*nzz + 19305*nxx*nzz*nzz - 3673743226306559.0/274877906944.0*nxx*nzz + 810*nxx - 6633147491942401.0/34359738368.0*nyy*nyy*nzz*nzz + (5306517993553919.0/137438953472.0)*nyy*nyy*nzz + (4974860618956799.0/34359738368.0)*nyy*nzz*nzz*nzz - 6633147491942403.0/137438953472.0*nyy*nzz*nzz + (6.0/753950830475.0)*nyy*nzz - 5306517993553919.0/549755813888.0*nzz*nzz*nzz + (1836871613153279.0/274877906944.0)*nzz*nzz - 810*nzz);  // power=9
        out[24] = T(-2730*nxx*nxx*nxy*ny*ny - 8848319824527361.0/274877906944.0*nxx*nxy*ny*nyy*ny - 861105.0/2.0*nxx*nxy*ny*nyz*nz + (106515.0/2.0)*nxx*nxy*ny*ny + 270270*nxx*nxy*nzz*nzz - 7959776990330877.0/274877906944.0*nxx*nxy*nzz - 11745.0/2.0*nxx*nxy - 8097903138570241.0/274877906944.0*nxy*ny*nyy*nyy*ny + 556230*nxy*ny*nyy*nyz*nz + (6827967208488961.0/549755813888.0)*nxy*ny*nyy*ny - 495390*nxy*ny*nyz*nz*nzz - 6263608505794561.0/68719476736.0*nxy*ny*nyz*nz - 5290025319137279.0/549755813888.0*nxy*ny*ny + 270270*nxy*nzz*nzz*nzz - 7959776990330879.0/34359738368.0*nxy*nzz*nzz + (139725.0/2.0)*nxy*nzz + (405.0/2.0)*nxy);  // power=9
        out[25] = T(-25515.0/2.0*nxx*nxx*nxy*nz - 219240*nxx*nxy*ny*nyz + (42525.0/2.0)*nxx*nxy*nzz*nz + (93555.0/2.0)*nxx*nxy*nz + 266490*nxy*ny*nyy*nyz - 344925.0/2.0*nxy*ny*nyz*nz*nz - 246645.0/2.0*nxy*ny*nyz + 34020*nxy*nzz*nzz*nz + (31185.0/2.0)*nxy*nz);  // power=8
        out[26] = T(17010*nxx*nxx*nxy*nz - 28350*nxx*nxy*ny*nyz - 462105.0/2.0*nxx*nxy*nzz*nz + (93555.0/2.0)*nxx*nxy*nz - 45360*nxy*ny*nyy*nyz + 179550*nxy*ny*nyz*nz*nz - 496125.0/2.0*nxy*nzz*nzz*nz + (405405.0/2.0)*nxy*nzz*nz - 34020*nxy*nz);  // power=8
        out[27] = T(-405405.0/2.0*nxx*nyy*nzz*nz + (93555.0/2.0)*nxx*nyy*nz + (31185.0/2.0)*nxx*nzz*nz - 8505.0/2.0*nxx*nz + 270270*nyy*nyy*nzz*nz - 62370*nyy*nyy*nz - 405405.0/2.0*nyy*nzz*nzz*nz + (93555.0/2.0)*nyy*nzz*nz + 8505*nyy*nz + (31185.0/2.0)*nzz*nzz*nz - 19845.0/2.0*nzz*nz + 945*nz);  // power=8
    } break;

    case 22: {
        out[0] = T(-525*nxx*nxx + 2625*nxx*nyy - 525*nxx*nzz + 150*nxx - 525*nyy*nyy - 525*nyy*nzz + 150*nyy + 150*nzz - 45);  // power=5
        out[1] = T(-4800*nxx*nxx*nx + (6597069766656001.0/274877906944.0)*nxx*nxy*ny - 4350*nxx*nxz*nz + 3150*nxx*nx - 4275*nxy*ny*nyy - 3825*nxy*nyz*nz - 4725*nxy*ny + 450*nxz*nz*nzz + 1575*nxz*nz - 450*nx);  // power=6
        out[2] = T(-4725*nxx*nxx*ny + 23625*nxx*nyy*ny - 4725*nxx*nyz*nz - 4200*nxx*ny - 4725*nyy*nyy*ny - 4725*nyy*nyz*nz + 3150*nyy*ny + 2100*nyz*nz - 525*ny);  // power=6
        out[3] = T(-4725*nxx*nxx*nz + 23625*nxx*nyy*nz - 4725*nxx*nzz*nz + 2100*nxx*nz - 4725*nyy*nyy*nz - 4725*nyy*nzz*nz + 2100*nyy*nz + 1050*nzz*nz - 525*nz);  // power=6
        out[4] = T(-5775*nxx*nxx*nxx + (63525.0/2.0)*nxx*nxx*nyy - 5775.0/2.0*nxx*nxx*nzz + 5250*nxx*nxx - 40425.0/2.0*nxx*nyy*nyy - 17325*nxx*nyy*nzz - 4725*nxx*nyy + (5775.0/2.0)*nxx*nzz*nzz + 1575*nxx*nzz - 1400*nxx + (5775.0/2.0)*nyy*nyy*nyy + 5775*nyy*nyy*nzz - 2625*nyy*nyy + (5775.0/2.0)*nyy*nzz*nzz - 3150*nyy*nzz + (2975.0/2.0)*nyy - 525*nzz*nzz + (875.0/2.0)*nzz - 175);  // power=7
        out[5] = T(-5775.0/2.0*nxx*nxx*nxx + 11550*nxx*nxx*nyy + (5775.0/2.0)*nxx*nxx*nzz + 2625*nxx*nxx + 11550*nxx*nyy*nyy - 34650*nxx*nyy*nzz - 9450*nxx*nyy + 5775*nxx*nzz*nzz - 1575*nxx*nzz + (175.0/2.0)*nxx - 5775.0/2.0*nyy*nyy*nyy + (5775.0/2.0)*nyy*nyy*nzz + 2625*nyy*nyy + 5775*nyy*nzz*nzz - 1575*nyy*nzz + (175.0/2.0)*nyy - 1050*nzz*nzz + 875*nzz - 350);  // power=7
        out[6] = T(-17850*nxx*nxx*nxy + 87150*nxx*nxy*ny*ny - 13650*nxx*nxy*nzz - 6300*nxx*nxy - 16275*nxy*ny*nyy*ny - 12075*nxy*ny*nyz*nz - 7875*nxy*ny*ny + 4200*nxy*nzz*nzz + 4725*nxy*nzz + 2100*nxy);  // power=7
        out[7] = T(-17850*nxx*nxx*nxz + 92400*nxx*nxy*nyz - 16800*nxx*nxz*nz*nz + 12600*nxx*nxz - 11025*nxy*ny*nyy*nz - 9975*nxy*nyz*nz*nz - 16275*nxy*nyz + 1050*nxz*nz*nzz*nz + 4725*nxz*nz*nz - 2100*nxz);  // power=7
        out[8] = T(-17325*nxx*nxx*nyz + 86625*nxx*nyy*nyz - 17325*nxx*nyz*nz*nz - 9450*nxx*nyz - 17325*nyy*nyy*nyz - 17325*nyy*nyz*nz*nz + 12600*nyy*nyz + 6300*nyz*nz*nz - 2625*nyz);  // power=7
        out[9] = T(-45045.0/2.0*nxx*nxx*nxx*nx + (225225.0/2.0)*nxx*nxx*nxy*ny - 45045.0/2.0*nxx*nxx*nxz*nz + (58905.0/2.0)*nxx*nxx*nx - 45045.0/2.0*nxx*nxy*ny*nyy - 45045.0/2.0*nxx*nxy*nyz*nz - 148995.0/2.0*nxx*nxy*ny + (38115.0/2.0)*nxx*nxz*nz - 21735.0/2.0*nxx*nx + (10395.0/2.0)*nxy*ny*nyy + (10395.0/2.0)*nxy*nyz*nz + 8505*nxy*ny - 2835*nxz*nz + (1995.0/2.0)*nx);  // power=8
        out[10] = T(-45045.0/2.0*nxx*nxx*nyy*ny + (10395.0/2.0)*nxx*nxx*ny + (225225.0/2.0)*nxx*nyy*nyy*ny - 45045.0/2.0*nxx*nyy*nyz*nz - 148995.0/2.0*nxx*nyy*ny + (10395.0/2.0)*nxx*nyz*nz + 8505*nxx*ny - 45045.0/2.0*nyy*nyy*nyy*ny - 45045.0/2.0*nyy*nyy*nyz*nz + (58905.0/2.0)*nyy*nyy*ny + (38115.0/2.0)*nyy*nyz*nz - 21735.0/2.0*nyy*ny - 2835*nyz*nz + (1995.0/2.0)*ny);  // power=8
        out[11] = T(-45045.0/2.0*nxx*nxx*nzz*nz + (10395.0/2.0)*nxx*nxx*nz + (225225.0/2.0)*nxx*nyy*nzz*nz - 51975.0/2.0*nxx*nyy*nz - 45045.0/2.0*nxx*nzz*nzz*nz + (38115.0/2.0)*nxx*nzz*nz - 2835*nxx*nz - 45045.0/2.0*nyy*nyy*nzz*nz + (10395.0/2.0)*nyy*nyy*nz - 45045.0/2.0*nyy*nzz*nzz*nz + (38115.0/2.0)*nyy*nzz*nz - 2835*nyy*nz + 3465*nzz*nzz*nz - 6615.0/2.0*nzz*nz + (1155.0/2.0)*nz);  // power=8
        out[12] = T(-135135.0/2.0*nxx*nxx*nxx*ny + (675675.0/2.0)*nxx*nxx*nyy*ny - 135135.0/2.0*nxx*nxx*nyz*nz + (10395.0/2.0)*nxx*nxx*ny - 135135.0/2.0*nxx*nyy*nyy*ny - 135135.0/2.0*nxx*nyy*nyz*nz - 197505.0/2.0*nxx*nyy*ny + (93555.0/2.0)*nxx*nyz*nz + (19845.0/2.0)*nxx*ny + (10395.0/2.0)*nyy*nyy*ny + (10395.0/2.0)*nyy*nyz*nz + 1890*nyy*ny - 2835*nyz*nz - 945.0/2.0*ny);  // power=8
        out[13] = T(-135135.0/2.0*nxx*nxx*nxx*nz + (675675.0/2.0)*nxx*nxx*nyy*nz - 135135.0/2.0*nxx*nxx*nzz*nz + (135135.0/2.0)*nxx*nxx*nz - 135135.0/2.0*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 218295.0/2.0*nxx*nyy*nz + (72765.0/2.0)*nxx*nzz*nz - 36855.0/2.0*nxx*nz + (10395.0/2.0)*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + 2835*nyy*nz - 1890*nzz*nz + (1575.0/2.0)*nz);  // power=8
        out[14] = T(-135135.0/2.0*nxx*nxx*nyy*nz + (10395.0/2.0)*nxx*nxx*nz + (675675.0/2.0)*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 218295.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + 2835*nxx*nz - 135135.0/2.0*nyy*nyy*nyy*nz - 135135.0/2.0*nyy*nyy*nzz*nz + (135135.0/2.0)*nyy*nyy*nz + (72765.0/2.0)*nyy*nzz*nz - 36855.0/2.0*nyy*nz - 1890*nzz*nz + (1575.0/2.0)*nz);  // power=8
        out[15] = T(-143640*nxx*nxx*nxy*nz + 708750*nxx*nxy*ny*nyz - 120960*nxx*nxy*nzz*nz - 20790*nxx*nxy*nz - 93555*nxy*ny*nyy*nyz - 70875*nxy*ny*nyz*nz*nz - 70875*nxy*ny*nyz + 22680*nxy*nzz*nzz*nz + 31185*nxy*nzz*nz + 5670*nxy*nz);  // power=8
        out[16] = T(-96525.0/8.0*nxx*nxx*nxx*nxx + (482625.0/8.0)*nxx*nxx*nxx*nyy + (482625.0/8.0)*nxx*nxx*nxx*nzz + (57915.0/4.0)*nxx*nxx*nxx - 96525.0/8.0*nxx*nxx*nyy*nyy - 2992275.0/8.0*nxx*nxx*nyy*nzz - 122265.0/4.0*nxx*nxx*nyy + (482625.0/8.0)*nxx*nxx*nzz*nzz - 4090440953364479.0/68719476736.0*nxx*nxx*nzz - 4702731491077351.0/962072674303.0*nxx*nxx + (4974860618956799.0/68719476736.0)*nxx*nyy*nyy*nzz - 1.0/1000000000000.0*nxx*nyy*nyy + (1061775.0/8.0)*nxx*nyy*nzz*nzz + (135135.0/2.0)*nxx*nyy*nzz + 4455*nxx*nyy - 96525.0/8.0*nxx*nzz*nzz*nzz - 5748727826350079.0/274877906944.0*nxx*nzz*nzz + (4388082186977279.0/274877906944.0)*nxx*nzz + 270*nxx - 96525.0/8.0*nyy*nyy*nzz*nzz - 1.0/1000000000000.0*nyy*nyy*nzz - 96525.0/8.0*nyy*nzz*nzz*nzz + (32175.0/4.0)*nyy*nzz*nzz - 4455*nyy*nzz + (884419665590711.0/549755813887.0)*nzz*nzz*nzz - 380980779023171.0/879609302218.0*nzz*nzz - 270*nzz);  // power=9
        out[17] = T(-96525.0/8.0*nxx*nxx*nyy*nyy + (4974860618956799.0/68719476736.0)*nxx*nxx*nyy*nzz - 1.0/1000000000000.0*nxx*nxx*nyy - 96525.0/8.0*nxx*nxx*nzz*nzz - 1.0/1000000000000.0*nxx*nxx*nzz + (482625.0/8.0)*nxx*nyy*nyy*nyy - 2992275.0/8.0*nxx*nyy*nyy*nzz - 122265.0/4.0*nxx*nyy*nyy + (1061775.0/8.0)*nxx*nyy*nzz*nzz + (135135.0/2.0)*nxx*nyy*nzz + 4455*nxx*nyy - 96525.0/8.0*nxx*nzz*nzz*nzz + (32175.0/4.0)*nxx*nzz*nzz - 4455*nxx*nzz - 96525.0/8.0*nyy*nyy*nyy*nyy + (482625.0/8.0)*nyy*nyy*nyy*nzz + (57915.0/4.0)*nyy*nyy*nyy + (482625.0/8.0)*nyy*nyy*nzz*nzz - 4090440953364479.0/68719476736.0*nyy*nyy*nzz - 4702731491077351.0/962072674303.0*nyy*nyy - 96525.0/8.0*nyy*nzz*nzz*nzz - 5748727826350079.0/274877906944.0*nyy*nzz*nzz + (4388082186977279.0/274877906944.0)*nyy*nzz + 270*nyy + (884419665590711.0/549755813887.0)*nzz*nzz*nzz - 380980779023171.0/879609302218.0*nzz*nzz - 270*nzz);  // power=9
        out[18] = T(-95265.0/2.0*nxx*nxx*nxy*ny*ny + (4974860618956799.0/34359738368.0)*nxx*nxx*nxy*nzz - 1.0/879609302221.0*nxx*nxx*nxy + (476235.0/2.0)*nxx*nxy*ny*nyy*ny - 811710*nxx*nxy*ny*nyz*nz - 73170*nxx*nxy*ny*ny + (7959776990330879.0/68719476736.0)*nxx*nxy*nzz*nzz + (7347486452613125.0/549755813888.0)*nxx*nxy*nzz + (4193262470430719.0/549755813888.0)*nxx*nxy - 7158851488972799.0/137438953472.0*nxy*ny*nyy*nyy*ny + (7223791394488319.0/137438953472.0)*nxy*ny*nyy*nyz*nz + (2937757630463997.0/137438953472.0)*nxy*ny*nyy*ny + 75690*nxy*ny*nyz*nz*nzz + 90360*nxy*ny*nyz*nz - 2849934139196036.0/879609302221.0*nxy*ny*ny - 7959776990330879.0/274877906944.0*nxy*nzz*nzz*nzz - 31185.0/2.0*nxy*nzz*nzz - 2746030290370561.0/549755813888.0*nxy*nzz - 135.0/2.0*nxy);  // power=9
        out[19] = T(-96525.0/2.0*nxx*nxx*nxx*nxy + (483885.0/2.0)*nxx*nxx*nxy*ny*ny + 96525*nxx*nxx*nxy*nzz + (7075357324738557.0/549755813888.0)*nxx*nxx*nxy - 7378410217144319.0/137438953472.0*nxx*nxy*ny*nyy*ny - 819810*nxx*nxy*ny*nyz*nz - 4889047172382719.0/68719476736.0*nxx*nxy*ny*ny + (7959776990330879.0/68719476736.0)*nxx*nxy*nzz*nzz + (6258969941114883.0/137438953472.0)*nxx*nxy*nzz + (673450872012925.0/610839793209.0)*nxx*nxy - 12105.0/2.0*nxy*ny*nyy*nyy*ny + 90495*nxy*ny*nyy*nyz*nz + (25515.0/2.0)*nxy*ny*nyy*ny + 67590*nxy*ny*nyz*nz*nzz + 68310*nxy*ny*nyz*nz + (343047627866384.0/586406201481.0)*nxy*ny*ny - 7959776990330879.0/274877906944.0*nxy*nzz*nzz*nzz - 31185.0/2.0*nxy*nzz*nzz - 2931572877563093.0/549755813889.0*nxy*nzz + 270*nxy);  // power=9
        out[20] = T(-96525.0/2.0*nxx*nxx*nxx*nxz - 675.0/2.0*nxx*nxx*nxx*nxz + (483885.0/2.0)*nxx*nxx*nxy*nyz + 2700*nxx*nxx*nxy*nyz + (2925.0/2.0)*nxx*nxx*nxz*nz*nz + (7075357324738559.0/137438953472.0)*nxx*nxx*nxz - 53145*nxx*nxy*ny*nyy*nz + (3525309156556799.0/549755813888.0)*nxx*nxy*ny*nyy*nz - 293985*nxx*nxy*nyz*nz*nz - 21825*nxx*nxy*nyz*nz*nz - 2776954054901759.0/34359738368.0*nxx*nxy*nyz + (96525.0/2.0)*nxx*nxz*nz*nzz*nz + 720*nxx*nxz*nz*nzz*nz - 7075357324738559.0/549755813888.0*nxx*nxz*nz*nz - 4205631976243199.0/274877906944.0*nxx*nxz - 11025.0/2.0*nxy*ny*nyy*nyy*nz + (3374999999999999.0/1000000000000.0)*nxy*ny*nyy*nyy*nz + 37710*nxy*ny*nyy*nzz*nz - 6401219257958399.0/274877906944.0*nxy*ny*nyy*nzz*nz + (4935432819179519.0/549755813888.0)*nxy*ny*nyy*nz - 5039999999999999.0/1000000000000.0*nxy*nyz*nz*nzz*nz + (5640494650490879.0/274877906944.0)*nxy*nyz*nz*nzz*nz + (6475436292833279.0/137438953472.0)*nxy*nyz*nz*nz + (1529999999999999.0/1000000000000.0)*nxy*nyz - 1080*nxz*nz*nzz*nzz*nz - 5306517993553919.0/549755813888.0*nxz*nz*nzz*nz + (2572857208997009.0/733007751851.0)*nxz*nz*nz + 1080*nxz);  // power=9
        out[21] = T(-96525.0/2.0*nxx*nxx*nyy*nyz + (96525.0/2.0)*nxx*nxx*nyz*nz*nz + (482625.0/2.0)*nxx*nyy*nyy*nyz - 289575*nxx*nyy*nyz*nz*nz - 5748727826350079.0/68719476736.0*nxx*nyy*nyz + (96525.0/2.0)*nxx*nyz*nz*nzz*nz + (6435000000000001.0/1000000000000.0)*nxx*nyz*nz*nz + (4898324301742079.0/549755813888.0)*nxx*nyz - 96525.0/2.0*nyy*nyy*nyy*nyz + (7075357324738559.0/137438953472.0)*nyy*nyy*nyz + (96525.0/2.0)*nyy*nyz*nz*nzz*nz - 7075357324738559.0/549755813888.0*nyy*nyz*nz*nz - 8299938400174077.0/549755813888.0*nyy*nyz - 3537678662369279.0/274877906944.0*nyz*nz*nzz*nz + (12375.0/2.0)*nyz*nz*nz + 540*nyz);  // power=9
        out[22] = T(-4974860618956799.0/68719476736.0*nxx*nxx*nxx*nyy + (4974860618956799.0/68719476736.0)*nxx*nxx*nxx*nzz + (1447875.0/4.0)*nxx*nxx*nyy*nyy - 1447875.0/4.0*nxx*nxx*nyy*nzz - 6633147491942401.0/137438953472.0*nxx*nxx*nyy + (96525.0/2.0)*nxx*nxx*nzz*nzz - 135135.0/2.0*nxx*nxx*nzz + 4455*nxx*nxx - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyy - 1447875.0/4.0*nxx*nyy*nyy*nzz - 6633147491942401.0/137438953472.0*nxx*nyy*nyy + (1061775.0/4.0)*nxx*nyy*nzz*nzz + (366795.0/2.0)*nxx*nyy*nzz + (8367970682142719.0/549755813888.0)*nxx*nyy - 96525.0/4.0*nxx*nzz*nzz*nzz - 12870*nxx*nzz*nzz + (46035.0/4.0)*nxx*nzz - 1620*nxx + (4974860618956799.0/68719476736.0)*nyy*nyy*nyy*nzz + (96525.0/2.0)*nyy*nyy*nzz*nzz - 135135.0/2.0*nyy*nyy*nzz + 4455*nyy*nyy - 96525.0/4.0*nyy*nzz*nzz*nzz - 12870*nyy*nzz*nzz + (46035.0/4.0)*nyy*nzz - 1620*nyy + (1768839331187857.0/549755813889.0)*nzz*nzz*nzz - 761961558049807.0/879609302222.0*nzz*nzz - 540*nzz + 105);  // power=9
        out[23] = T(-4974860618956799.0/34359738368.0*nxx*nxx*nxx*nyz + (1447875.0/2.0)*nxx*nxx*nyy*nyz - 96525*nxx*nxx*nyz*nz*nz + (5306517993553919.0/274877906944.0)*nxx*nxx*nyz - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nyz - 386100*nxx*nyy*nyz*nz*nz - 115830*nxx*nyy*nyz + (96525.0/2.0)*nxx*nyz*nz*nzz*nz + 83655*nxx*nyz*nz*nz - 163277476724588.0/219902325555.0*nxx*nyz + (96525.0/2.0)*nyy*nyy*nyz*nz*nz + (1.0/879609302221.0)*nyy*nyy*nyz + (96525.0/2.0)*nyy*nyz*nz*nzz*nz - 5306517993553919.0/137438953472.0*nyy*nyz*nz*nz + (4898324301742079.0/549755813888.0)*nyy*nyz - 3537678662369279.0/274877906944.0*nyz*nz*nzz*nz + (3401614098431999.0/549755813888.0)*nyz*nz*nz - 1350*nyz);  // power=9
        out[24] = T(-675.0/2.0*nxx*nxx*nxx*nxz - 5018153889300479.0/34359738368.0*nxx*nxx*nxy*nyz + 2700*nxx*nxx*nxy*nyz + (96525.0/2.0)*nxx*nxx*nxz*nz*nz + (2925.0/2.0)*nxx*nxx*nxz*nz*nz + (1.0/879609302221.0)*nxx*nxx*nxz + 878490*nxx*nxy*ny*nyy*nz + (3525309156556799.0/549755813888.0)*nxx*nxy*ny*nyy*nz - 377280*nxx*nxy*nyz*nz*nz - 21825*nxx*nxy*nyz*nz*nz - 121500*nxx*nxy*nyz + (96525.0/2.0)*nxx*nxz*nz*nzz*nz + 720*nxx*nxz*nz*nzz*nz - 5306517993553919.0/137438953472.0*nxx*nxz*nz*nz + (5120975406366719.0/549755813888.0)*nxx*nxz + 11025*nxy*ny*nyy*nyy*nz + (3374999999999999.0/1000000000000.0)*nxy*ny*nyy*nyy*nz + (138735.0/2.0)*nxy*ny*nyy*nzz*nz - 6401219257958399.0/274877906944.0*nxy*ny*nyy*nzz*nz - 4928474972160001.0/34359738368.0*nxy*ny*nyy*nz + 10080*nxy*nyz*nz*nzz*nz + (5640494650490879.0/274877906944.0)*nxy*nyz*nz*nzz*nz + (6790858691051519.0/68719476736.0)*nxy*nyz*nz*nz + (2533274790396478.0/879609302221.0)*nxy*nyz - 1080*nxz*nz*nzz*nzz*nz - 5306517993553919.0/549755813888.0*nxz*nz*nzz*nz + (3772699272805025.0/916259689813.0)*nxz*nz*nz - 2835.0/2.0*nxz);  // power=9
        out[25] = T(-135135.0/2.0*nxx*nxx*nxy*ny + (10395.0/2.0)*nxx*nxx*nx + 341145*nxx*nxy*ny*nyy - 117495.0/2.0*nxx*nxy*nyz*nz - 201915.0/2.0*nxx*nxy*ny + (10395.0/2.0)*nxx*nxz*nz + (4095.0/2.0)*nxx*nx - 64260*nxy*ny*nyy*nyy - 55440*nxy*ny*nyy*nzz - 315*nxy*ny*nyy + 8820*nxy*nyz*nz*nzz + (71505.0/2.0)*nxy*nyz*nz + 12285*nxy*ny - 5355.0/2.0*nxz*nz - 630*nx);  // power=8
        out[26] = T((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nxy*ny - 135135.0/2.0*nxx*nxx*nxz*nz - 4725.0/2.0*nxx*nxx*nxz*nz + (10395.0/2.0)*nxx*nxx*nx + (6615.0/2.0)*nxx*nxy*ny*nyy - 4725.0/2.0*nxx*nxy*ny*nyy + (693315.0/2.0)*nxx*nxy*nyz*nz + 23625*nxx*nxy*nyz*nz - 56385.0/2.0*nxx*nxy*ny - 135135.0/2.0*nxx*nxz*nz*nzz + (114345.0/2.0)*nxx*nxz*nz - 7245.0/2.0*nxx*nx + (6615.0/2.0)*nxy*ny*nyy*nyy - 1260*nxy*ny*nyy*nyy - 55440*nxy*ny*nyy*nzz + (51975.0/2.0)*nxy*ny*nyy*nzz - 315*nxy*ny*nyy + 8820*nxy*nyz*nz*nzz - 37800*nxy*nyz*nz*nzz - 115605.0/2.0*nxy*nyz*nz + 6615*nxy*ny + 2520*nxz*nz*nzz*nzz + (31185.0/2.0)*nxz*nz*nzz - 9765*nxz*nz + (945.0/2.0)*nx);  // power=8
        out[27] = T(-135135.0/2.0*nxx*nxx*nyz*nz + (10395.0/2.0)*nxx*nxx*ny + (675675.0/2.0)*nxx*nyy*nyz*nz - 51975.0/2.0*nxx*nyy*ny - 135135.0/2.0*nxx*nyz*nz*nzz - 31185.0/2.0*nxx*nyz*nz + 2835*nxx*ny - 135135.0/2.0*nyy*nyy*nyz*nz + (10395.0/2.0)*nyy*nyy*ny - 135135.0/2.0*nyy*nyz*nz*nzz + (114345.0/2.0)*nyy*nyz*nz - 3780*nyy*ny + 20790*nyz*nz*nzz - 25515.0/2.0*nyz*nz + (1575.0/2.0)*ny);  // power=8
    } break;

    case 23: {
        out[0] = T(3150*nxx*nyz - 525*nyy*nyz - 525*nyz*nz*nz);  // power=5
        out[1] = T(22050*nxx*nxy*nz - 11025*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[2] = T(28350*nxx*nyy*nz - 3150*nxx*nz - 4725*nyy*nyy*nz - 4725*nyy*nzz*nz + 1575*nyy*nz + 525*nzz*nz);  // power=6
        out[3] = T(28350*nxx*nyz*nz - 3150*nxx*ny - 4725*nyy*nyz*nz + 525*nyy*ny - 4725*nyz*nz*nzz + 1575*nyz*nz);  // power=6
        out[4] = T(34650*nxx*nxx*nyz - 23100*nxx*nyy*nyz - 23100*nxx*nyz*nz*nz - 6300*nxx*nyz + (5775.0/2.0)*nyy*nyy*nyz + 5775*nyy*nyz*nz*nz - 2100*nyy*nyz + (5775.0/2.0)*nyz*nz*nzz*nz - 2100*nyz*nz*nz + 1050*nyz);  // power=7
        out[5] = T(17325*nxx*nxx*nyz + (28875.0/2.0)*nxx*nyy*nyz - 75075.0/2.0*nxx*nyz*nz*nz - 3150*nxx*nyz - 5775.0/2.0*nyy*nyy*nyz + (5775.0/2.0)*nyy*nyz*nz*nz + 525*nyy*nyz + 5775*nyz*nz*nzz*nz - 2625*nyz*nz*nz + 525*nyz);  // power=7
        out[6] = T(-525*nxx*nxx*nxz + 109725*nxx*nxy*nyz + 525*nxx*nxz*nz*nz - 9450*nxx*nxz - 11025*nxy*ny*nyy*nz - 9975*nxy*nyz*nz*nz - 21000*nxy*nyz + 1050*nxz*nz*nzz*nz + 2625*nxz);  // power=7
        out[7] = T(-525*nxx*nxx*nxy + 525*nxx*nxy*ny*ny + 107625*nxx*nxy*nzz - 9450*nxx*nxy + 1050*nxy*ny*nyy*ny - 12075*nxy*ny*nyz*nz - 13125*nxy*nzz*nzz - 18900*nxy*nzz + 2625*nxy);  // power=7
        out[8] = T(103950*nxx*nyy*nzz - 9450*nxx*nyy - 9450*nxx*nzz - 17325*nyy*nyy*nzz + 1575*nyy*nyy - 17325*nyy*nzz*nzz + 9450*nyy*nzz - 1575*nyy + 1575*nzz*nzz - 1575*nzz + 1050);  // power=7
        out[9] = T(135135*nxx*nxx*nxy*nz - 45045.0/2.0*nxx*nxy*ny*nyz - 45045.0/2.0*nxx*nxy*nzz*nz - 93555*nxx*nxy*nz + (10395.0/2.0)*nxy*ny*nyz + (10395.0/2.0)*nxy*nzz*nz + 11340*nxy*nz);  // power=8
        out[10] = T(135135*nxx*nyy*nyy*nz - 62370*nxx*nyy*nz + 2835*nxx*nz - 45045.0/2.0*nyy*nyy*nyy*nz - 45045.0/2.0*nyy*nyy*nzz*nz + 20790*nyy*nyy*nz + 10395*nyy*nzz*nz - 8505.0/2.0*nyy*nz - 945.0/2.0*nzz*nz + 105*nz);  // power=8
        out[11] = T(135135*nxx*nyz*nz*nzz - 62370*nxx*nyz*nz + 2835*nxx*ny - 45045.0/2.0*nyy*nyz*nz*nzz + 10395*nyy*nyz*nz - 945.0/2.0*nyy*ny - 45045.0/2.0*nyz*nz*nzz*nzz + 20790*nyz*nz*nzz - 8505.0/2.0*nyz*nz + 105*ny);  // power=8
        out[12] = T(405405*nxx*nxx*nyy*nz - 31185*nxx*nxx*nz - 135135.0/2.0*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 280665.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + 14175*nxx*nz + (10395.0/2.0)*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + (8505.0/2.0)*nyy*nz - 945.0/2.0*nzz*nz - 630*nz);  // power=8
        out[13] = T(405405*nxx*nxx*nyz*nz - 31185*nxx*nxx*ny - 135135.0/2.0*nxx*nyy*nyz*nz + (10395.0/2.0)*nxx*nyy*ny - 135135.0/2.0*nxx*nyz*nz*nzz - 280665.0/2.0*nxx*nyz*nz + 14175*nxx*ny + (10395.0/2.0)*nyy*nyz*nz - 945.0/2.0*nyy*ny + (10395.0/2.0)*nyz*nz*nzz + (8505.0/2.0)*nyz*nz - 630*ny);  // power=8
        out[14] = T(405405*nxx*nyy*nyz*nz - 31185*nxx*nyy*ny - 93555*nxx*nyz*nz + 8505*nxx*ny - 135135.0/2.0*nyy*nyy*nyz*nz + (10395.0/2.0)*nyy*nyy*ny - 135135.0/2.0*nyy*nyz*nz*nzz + 51975*nyy*nyz*nz - 6615.0/2.0*nyy*ny + (31185.0/2.0)*nyz*nz*nzz - 14175.0/2.0*nyz*nz + 315*ny);  // power=8
        out[15] = T(315*nxx*nxx*nxx*nx - 1890*nxx*nxx*nxy*ny - 4725*nxx*nxx*nxz*nz + 6615*nxx*nxy*ny*nyy - 4725*nxx*nxy*ny*nyy + 828450*nxx*nxy*nyz*nz + 47250*nxx*nxy*nyz*nz - 66780*nxx*nxy*ny - 62370*nxx*nxz*nz + 5985*nxx*nx + 6615*nxy*ny*nyy*nyy - 2520*nxy*ny*nyy*nyy - 110880*nxy*ny*nyy*nzz + 51975*nxy*ny*nyy*nzz - 630*nxy*ny*nyy + 17640*nxy*nyz*nz*nzz - 75600*nxy*nyz*nz*nzz - 146790*nxy*nyz*nz + 16065*nxy*ny + 5040*nxz*nz*nzz*nzz + 14490*nxz*nz - 1890*nx);  // power=8
        out[16] = T((289575.0/4.0)*nxx*nxx*nxx*nyz - 96525.0/8.0*nxx*nxx*nyy*nyz - 3571425.0/8.0*nxx*nxx*nyz*nz*nz + 19305*nxx*nxx*nyz + (4974860618956799.0/68719476736.0)*nxx*nyy*nyz*nz*nz - 19305.0/2.0*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz + (135135.0/2.0)*nxx*nyz*nz*nz - 22275.0/2.0*nxx*nyz - 96525.0/8.0*nyy*nyz*nz*nzz*nz + (1179226220788687.0/366503875925.0)*nyy*nyz*nz*nz - 96525.0/8.0*nyz*nz*nzz*nzz*nz + (5306517993553919.0/549755813888.0)*nyz*nz*nzz*nz - 3673743226306559.0/549755813888.0*nyz*nz*nz + 945*nyz);  // power=9
        out[17] = T((289575.0/4.0)*nxx*nyy*nyy*nyz - 868725.0/2.0*nxx*nyy*nyz*nz*nz + (5306517993553921.0/137438953472.0)*nxx*nyy*nyz + (289575.0/4.0)*nxx*nyz*nz*nzz*nz + (5306517993553921.0/137438953472.0)*nxx*nyz*nz*nz - 8910*nxx*nyz - 96525.0/8.0*nyy*nyy*nyy*nyz + (482625.0/8.0)*nyy*nyy*nyz*nz*nz - 1.0/502633886983.0*nyy*nyy*nyz + (482625.0/8.0)*nyy*nyz*nz*nzz*nz - 7075357324738559.0/137438953472.0*nyy*nyz*nz*nz + 4455*nyy*nyz - 96525.0/8.0*nyz*nz*nzz*nzz*nz - 1.0/502633886983.0*nyz*nz*nzz*nz + 4455*nyz*nz*nz - 540*nyz);  // power=9
        out[18] = T(-2025.0/2.0*nxx*nxx*nxx*nxz + 1260*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz + (656145.0/2.0)*nxx*nxy*ny*nyy*nz + (38475.0/2.0)*nxx*nxy*ny*nyy*nz - 877545*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz + (8739055856517121.0/137438953472.0)*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz + 57915*nxx*nxz*nz*nz - 9315*nxx*nxz - 11025*nxy*ny*nyy*nyy*nz + 10125*nxy*ny*nyy*nyy*nz + (247365.0/2.0)*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz - 59265*nxy*ny*nyy*nz - 10080*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + 152145*nxy*nyz*nz*nz - 6302263211458561.0/274877906944.0*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 12285*nxz*nz*nz + (5535.0/2.0)*nxz);  // power=9
        out[19] = T(-2025.0/2.0*nxx*nxx*nxx*nxz + 291465*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz - 5306517993553919.0/274877906944.0*nxx*nxx*nxz - 62910*nxx*nxy*ny*nyy*nz + (38475.0/2.0)*nxx*nxy*ny*nyy*nz - 1860435.0/2.0*nxx*nxy*nyz*nz*nz - 8998815478579199.0/137438953472.0*nxx*nxy*nyz*nz*nz + (2495547797667843.0/137438953472.0)*nxx*nxy*nyz + 2160*nxx*nxz*nz*nzz*nz + (122265.0/2.0)*nxx*nxz*nz*nz - 572531411891499.0/942438538093.0*nxx*nxz - 33075.0/2.0*nxy*ny*nyy*nyy*nz + 10125*nxy*ny*nyy*nyy*nz + 113130*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz + (4193262470430719.0/549755813888.0)*nxy*ny*nyy*nz - 15120*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (5188234594222081.0/34359738368.0)*nxy*nyz*nz*nz - 6085796859740161.0/274877906944.0*nxy*nyz - 3240*nxz*nz*nzz*nzz*nz - 24975.0/2.0*nxz*nz*nz + 2025*nxz);  // power=9
        out[20] = T(210*nxx*nxx*nxy*ny*ny + 289575*nxx*nxx*nxy*nzz - 5306517993553919.0/274877906944.0*nxx*nxx*nxy + 420*nxx*nxy*ny*nyy*ny - 112065.0/2.0*nxx*nxy*ny*nyz*nz + (1768839331187857.0/549755813889.0)*nxx*nxy*ny*ny - 347490*nxx*nxy*nzz*nzz - 87615.0/2.0*nxx*nxy*nzz + (4786998749429759.0/549755813888.0)*nxx*nxy + 210*nxy*ny*nyy*nyy*ny - 4271602673909759.0/549755813888.0*nxy*ny*nyy*nyz*nz + (8419510289694719.0/274877906944.0)*nxy*ny*nyz*nz*nzz + (5985.0/2.0)*nxy*ny*nyz*nz - 825.0/2.0*nxy*ny*ny + 38610*nxy*nzz*nzz*nzz + (3265549534494719.0/137438953472.0)*nxy*nzz*nzz - 142496706959843.0/703687441777.0*nxy*nzz - 1485.0/2.0*nxy);  // power=9
        out[21] = T(289575*nxx*nyy*nyy*nzz - 5306517993553919.0/274877906944.0*nxx*nyy*nyy - 289575*nxx*nyy*nzz*nzz + 4455*nxx*nyy + (5306517993553919.0/274877906944.0)*nxx*nzz*nzz - 4455*nxx*nzz - 96525.0/2.0*nyy*nyy*nyy*nzz + (1768839331187857.0/549755813889.0)*nyy*nyy*nyy + (7959776990330879.0/274877906944.0)*nyy*nyy*nzz - 2227499999997772.0/999999999999.0*nyy*nyy + (96525.0/2.0)*nyy*nzz*nzz*nzz - 7959776990330879.0/274877906944.0*nyy*nzz*nzz + 270*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 270*nzz);  // power=9
        out[22] = T((868725.0/2.0)*nxx*nxx*nyy*nyz - 868725.0/2.0*nxx*nxx*nyz*nz*nz - 4974860618956799.0/68719476736.0*nxx*nyy*nyy*nyz - 868725.0/2.0*nxx*nyy*nyz*nz*nz - 5306517993553921.0/137438953472.0*nxx*nyy*nyz + (868725.0/4.0)*nxx*nyz*nz*nzz*nz + 115830*nxx*nyz*nz*nz - 8910*nxx*nyz + (4974860618956799.0/68719476736.0)*nyy*nyy*nyz*nz*nz - 19305.0/2.0*nyy*nyy*nyz + (96525.0/2.0)*nyy*nyz*nz*nzz*nz - 7075357324738559.0/137438953472.0*nyy*nyz*nz*nz + (22275.0/2.0)*nyy*nyz - 96525.0/4.0*nyz*nz*nzz*nzz*nz + (2653258996776959.0/274877906944.0)*nyz*nz*nzz*nz - 1224581075437747.0/549755813889.0*nyz*nz*nz - 540*nyz);  // power=9
        out[23] = T(868725*nxx*nxx*nyy*nzz - 57915*nxx*nxx*nyy - 57915*nxx*nxx*nzz + 4455*nxx*nxx - 4974860618956799.0/34359738368.0*nxx*nyy*nyy*nzz + (19305.0/2.0)*nxx*nyy*nyy - 868725.0/2.0*nxx*nyy*nzz*nzz - 115830*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy + (3979888495165439.0/137438953472.0)*nxx*nzz*nzz + (22275.0/2.0)*nxx*nzz - 1620*nxx + (96525.0/2.0)*nyy*nyy*nzz*nzz - 5306517993553919.0/549755813888.0*nyy*nyy*nzz + (96525.0/2.0)*nyy*nzz*nzz*nzz - 5306517993553921.0/137438953472.0*nyy*nzz*nzz + (31185.0/2.0)*nyy*nzz - 945*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 1215*nzz + 105);  // power=9
        out[24] = T(-420*nxx*nxx*nxy*ny*ny + (11685.0/2.0)*nxx*nxy*ny*nyy*ny + 908565*nxx*nxy*ny*nyz*nz - 63990*nxx*nxy*ny*ny - 270270*nxx*nxy*nzz*nzz - 4592179032883201.0/68719476736.0*nxx*nxy*nzz + (20655.0/2.0)*nxx*nxy + (12525.0/2.0)*nxy*ny*nyy*nyy*ny - 98265*nxy*ny*nyy*nyz*nz - 3105*nxy*ny*nyy*ny - 36960*nxy*ny*nyz*nz*nzz - 6478528669286401.0/68719476736.0*nxy*ny*nyz*nz + (6799105028259841.0/549755813888.0)*nxy*ny*ny + (135135.0/2.0)*nxy*nzz*nzz*nzz + (718420897588922.0/967570232443.0)*nxy*nzz*nzz + (4063382659399681.0/274877906944.0)*nxy*nzz - 5535.0/2.0*nxy);  // power=9
        out[25] = T(-8505.0/2.0*nxx*nxx*nxy*nz + (843885.0/2.0)*nxx*nxy*ny*nyz + (14175.0/2.0)*nxx*nxy*nzz*nz - 93555*nxx*nxy*nz - 93555.0/2.0*nxy*ny*nyy*nyz - 70875.0/2.0*nxy*ny*nyz*nz*nz - 51030*nxy*ny*nyz + 11340*nxy*nzz*nzz*nz + (36855.0/2.0)*nxy*nz);  // power=8
        out[26] = T(-8505.0/2.0*nxx*nxx*nxy*nz + (14175.0/2.0)*nxx*nxy*ny*nyz + (824985.0/2.0)*nxx*nxy*nzz*nz - 93555*nxx*nxy*nz + 11340*nxy*ny*nyy*nyz - 89775.0/2.0*nxy*ny*nyz*nz*nz - 112455.0/2.0*nxy*nzz*nzz*nz - 41580*nxy*nzz*nz + (36855.0/2.0)*nxy*nz);  // power=8
        out[27] = T(405405*nxx*nyy*nzz*nz - 93555*nxx*nyy*nz - 31185*nxx*nzz*nz + 8505*nxx*nz - 135135.0/2.0*nyy*nyy*nzz*nz + (31185.0/2.0)*nyy*nyy*nz - 135135.0/2.0*nyy*nzz*nzz*nz + 51975*nyy*nzz*nz - 14175.0/2.0*nyy*nz + (10395.0/2.0)*nzz*nzz*nz - 6615.0/2.0*nzz*nz + 315*nz);  // power=8
    } break;

    case 24: {
        out[0] = T(-525*nxx*nxz + 3150*nxy*nyz - 525*nxz*nz*nz);  // power=5
        out[1] = T(-4725*nxx*nxx*nz + 28350*nxx*nyy*nz - 4725*nxx*nzz*nz + 1575*nxx*nz - 3150*nyy*nz + 525*nzz*nz);  // power=6
        out[2] = T(-11025*nxx*nxy*nz + 22050*nxy*ny*nyz - 11025*nxy*nzz*nz);  // power=6
        out[3] = T(450*nxx*nxx*nx - 2250*nxx*nxy*ny - 6975*nxx*nxz*nz + 525*nxx*nx - 2700*nxy*ny*nyy + (6308447964364801.0/274877906944.0)*nxy*nyz*nz - 7425*nxz*nz*nzz + 4725*nxz*nz - 450*nx);  // power=6
        out[4] = T(-12425.0/2.0*nxx*nxx*nxz + 42700*nxx*nxy*nyz - 2450*nxx*nxz*nz*nz + 2625*nxx*nxz - 11725*nxy*ny*nyy*nz - 15925.0/2.0*nxy*nyz*nz*nz - 5775.0/2.0*nxy*nyz + (7525.0/2.0)*nxz*nz*nzz*nz - 3675.0/2.0*nxz*nz*nz - 175.0/2.0*nxz);  // power=7
        out[5] = T(-4025*nxx*nxx*nxz + 20300*nxx*nxy*nyz + 4025*nxx*nxz*nz*nz + 525*nxx*nxz + 24325*nxy*ny*nyy*nz - 56525.0/2.0*nxy*nyz*nz*nz - 22575.0/2.0*nxy*nyz + 8050*nxz*nz*nzz*nz - 12075.0/2.0*nxz*nz*nz + (3325.0/2.0)*nxz);  // power=7
        out[6] = T(-17325*nxx*nxx*nyz + 103950*nxx*nyy*nyz - 17325*nxx*nyz*nz*nz - 14175*nxx*nyz - 9450*nyy*nyz + 1575*nyz*nz*nz + 2100*nyz);  // power=7
        out[7] = T(-17325*nxx*nxx*nzz + 1575*nxx*nxx + 103950*nxx*nyy*nzz - 9450*nxx*nyy - 17325*nxx*nzz*nzz + 9450*nxx*nzz - 1575*nxx - 9450*nyy*nzz + 1575*nzz*nzz - 1575*nzz + 1050);  // power=7
        out[8] = T(3150*nxx*nxx*nxy - 3150*nxx*nxy*ny*ny - 39375*nxx*nxy*nzz + 1575*nxx*nxy - 6300*nxy*ny*nyy*ny + 72450*nxy*ny*nyz*nz - 42525*nxy*nzz*nzz + 14175*nxy*nzz - 1050*nxy);  // power=7
        out[9] = T(-45045.0/2.0*nxx*nxx*nxx*nz + 135135*nxx*nxx*nyy*nz - 45045.0/2.0*nxx*nxx*nzz*nz + 20790*nxx*nxx*nz - 62370*nxx*nyy*nz + 10395*nxx*nzz*nz - 8505.0/2.0*nxx*nz + 2835*nyy*nz - 945.0/2.0*nzz*nz + 105*nz);  // power=8
        out[10] = T(-2835.0/2.0*nxx*nxx*nxy*nz - 39060*nxx*nxy*ny*nyz + (4725.0/2.0)*nxx*nxy*nzz*nz + (10395.0/2.0)*nxx*nxy*nz + 120015*nxy*ny*nyy*nyz - 67725.0/2.0*nxy*ny*nyz*nz*nz - 159705.0/2.0*nxy*ny*nyz + 3780*nxy*nzz*nzz*nz + (25515.0/2.0)*nxy*nz);  // power=8
        out[11] = T(-315*nxx*nxx*nxx*nx + 1890*nxx*nxx*nxy*ny + 4725*nxx*nxx*nxz*nz + 4725*nxx*nxy*ny*nyy - 47250*nxx*nxy*nyz*nz - 45045.0/2.0*nxx*nxz*nz*nzz + 10395*nxx*nxz*nz - 945.0/2.0*nxx*nx + 2520*nxy*ny*nyy*nyy - 51975*nxy*ny*nyy*nzz + 75600*nxy*nyz*nz*nzz - 45045.0/2.0*nxz*nz*nzz*nzz - 5040*nxz*nz*nzz*nzz + 31185*nxz*nz*nzz - 19845.0/2.0*nxz*nz + 420*nx);  // power=8
        out[12] = T(-135135.0/2.0*nxx*nxx*nxy*nz + 405405*nxx*nxy*ny*nyz - 135135.0/2.0*nxx*nxy*nzz*nz - 51975.0/2.0*nxx*nxy*nz - 93555*nxy*ny*nyz + (31185.0/2.0)*nxy*nzz*nz + 14175*nxy*nz);  // power=8
        out[13] = T(-135135.0/2.0*nxx*nxx*nxz*nz + (10395.0/2.0)*nxx*nxx*nx + 405405*nxx*nxy*nyz*nz - 31185*nxx*nxy*ny - 135135.0/2.0*nxx*nxz*nz*nzz + 51975*nxx*nxz*nz - 6615.0/2.0*nxx*nx - 93555*nxy*nyz*nz + 8505*nxy*ny + (31185.0/2.0)*nxz*nz*nzz - 14175.0/2.0*nxz*nz + 315*nx);  // power=8
        out[14] = T((315.0/2.0)*nxx*nxx*nxx*nx - 945*nxx*nxx*nxy*ny - 4725.0/2.0*nxx*nxx*nxz*nz - 19845*nxx*nxy*ny*nyy - 4725.0/2.0*nxx*nxy*ny*nyy - 240975.0/2.0*nxx*nxy*nyz*nz + 23625*nxx*nxy*nyz*nz + (36855.0/2.0)*nxx*nxy*ny + (10395.0/2.0)*nxx*nxz*nz - 2835.0/2.0*nxx*nx - 19845*nxy*ny*nyy*nyy - 1260*nxy*ny*nyy*nyy + 332640*nxy*ny*nyy*nzz + (51975.0/2.0)*nxy*ny*nyy*nzz + 1890*nxy*ny*nyy - 52920*nxy*nyz*nz*nzz - 37800*nxy*nyz*nz*nzz - 210735.0/2.0*nxy*nyz*nz + (2835.0/2.0)*nxy*ny + 2520*nxz*nz*nzz*nzz + (12285.0/2.0)*nxz*nz + (315.0/2.0)*nx);  // power=8
        out[15] = T(-135135*nxx*nxx*nyz*nz + 10395*nxx*nxx*ny + 810810*nxx*nyy*nyz*nz - 62370*nxx*nyy*ny - 135135*nxx*nyz*nz*nzz - 62370*nxx*nyz*nz + 8505*nxx*ny - 62370*nyy*nyz*nz + 5670*nyy*ny + 10395*nyz*nz*nzz + 8505*nyz*nz - 1260*ny);  // power=8
        out[16] = T(-96525.0/8.0*nxx*nxx*nxx*nxz - 2025.0/4.0*nxx*nxx*nxx*nxz + (289575.0/4.0)*nxx*nxx*nxy*nyz + 4050*nxx*nxx*nxy*nyz + (482625.0/8.0)*nxx*nxx*nxz*nz*nz + (8775.0/4.0)*nxx*nxx*nxz*nz*nz - 1.0/502633886983.0*nxx*nxx*nxz + (38475.0/4.0)*nxx*nxy*ny*nyy*nz - 868725.0/2.0*nxx*nxy*nyz*nz*nz - 65475.0/2.0*nxx*nxy*nyz*nz*nz + (5306517993553921.0/137438953472.0)*nxx*nxy*nyz + (482625.0/8.0)*nxx*nxz*nz*nzz*nz + 1080*nxx*nxz*nz*nzz*nz - 7075357324738559.0/137438953472.0*nxx*nxz*nz*nz + 4455*nxx*nxz + (2783138807813062.0/549755813889.0)*nxy*ny*nyy*nyy*nz - 139725.0/4.0*nxy*ny*nyy*nzz*nz + 30780*nxy*nyz*nz*nzz*nz + (173745.0/2.0)*nxy*nyz*nz*nz - 57915.0/4.0*nxy*nyz - 96525.0/8.0*nxz*nz*nzz*nzz*nz - 1620*nxz*nz*nzz*nzz*nz + (2653258996776959.0/549755813888.0)*nxz*nz*nzz*nz + (1485.0/2.0)*nxz*nz*nz - 135.0/4.0*nxz);  // power=9
        out[17] = T(-2025.0/2.0*nxx*nxx*nxx*nxz - 11655.0/2.0*nxx*nxx*nxy*nyz + 8100*nxx*nxx*nxy*nyz + (8775.0/2.0)*nxx*nxx*nxz*nz*nz - 157185.0/4.0*nxx*nxy*ny*nyy*nz + (38475.0/2.0)*nxx*nxy*ny*nyy*nz + (7778099873710079.0/68719476736.0)*nxx*nxy*nyz*nz*nz - 65475*nxx*nxy*nyz*nz*nz - 143505.0/4.0*nxx*nxy*nyz - 96525.0/8.0*nxx*nxz*nz*nzz*nz + 2160*nxx*nxz*nz*nzz*nz + (1179226220788687.0/366503875925.0)*nxx*nxz*nz*nz + (14985.0/8.0)*nxx*nxz + (407925.0/8.0)*nxy*ny*nyy*nyy*nz + (5566277615615999.0/549755813888.0)*nxy*ny*nyy*nyy*nz - 1684845.0/4.0*nxy*ny*nyy*nzz*nz - 139725.0/2.0*nxy*ny*nyy*nzz*nz + (69255.0/8.0)*nxy*ny*nyy*nz + 46620*nxy*nyz*nz*nzz*nz + (8460741975736319.0/137438953472.0)*nxy*nyz*nz*nzz*nz + (5870103602135041.0/68719476736.0)*nxy*nyz*nz*nz + (93195.0/8.0)*nxy*nyz - 96525.0/8.0*nxz*nz*nzz*nzz*nz - 3240*nxz*nz*nzz*nzz*nz + (5306517993553919.0/274877906944.0)*nxz*nz*nzz*nz - 6725918785535999.0/549755813888.0*nxz*nz*nz + (67470031704628.0/799644820203.0)*nxz);  // power=9
        out[18] = T(-96525.0/2.0*nxx*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nxx*nyz*nz*nz - 19305*nxx*nxx*nyz + 289575*nxx*nyy*nyy*nyz - 1833975.0/2.0*nxx*nyy*nyz*nz*nz + (1326629498388483.0/137438953472.0)*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz + (3979888495165441.0/137438953472.0)*nxx*nyz*nz*nz - 3.0/753950830475.0*nxx*nyz - 5306517993553919.0/274877906944.0*nyy*nyy*nyz + (122265.0/2.0)*nyy*nyz*nz*nz - 1.0/670178515978.0*nyy*nyz - 19305.0/2.0*nyz*nz*nzz*nz - 4455*nyz*nz*nz + 405*nyz);  // power=9
        out[19] = T(-96525.0/2.0*nxx*nxx*nxx*nyz + 289575*nxx*nxx*nyy*nyz + 96525*nxx*nxx*nyz*nz*nz - 3979888495165439.0/137438953472.0*nxx*nxx*nyz - 868725*nxx*nyy*nyz*nz*nz + (3979888495165441.0/68719476736.0)*nxx*nyy*nyz + (4974860618956799.0/34359738368.0)*nxx*nyz*nz*nzz*nz + (5306517993553923.0/137438953472.0)*nxx*nyz*nz*nz - 1224581075435521.0/274877906944.0*nxx*nyz + 57915*nyy*nyz*nz*nz - 8910*nyy*nyz - 19305.0/2.0*nyz*nz*nzz*nz - 4455*nyz*nz*nz + 1350*nyz);  // power=9
        out[20] = T(-96525.0/2.0*nxx*nxx*nxx*nzz + (1768839331187857.0/549755813889.0)*nxx*nxx*nxx + 289575*nxx*nxx*nyy*nzz - 5306517993553919.0/274877906944.0*nxx*nxx*nyy + (7959776990330879.0/274877906944.0)*nxx*nxx*nzz - 2227499999997772.0/999999999999.0*nxx*nxx - 289575*nxx*nyy*nzz*nzz + 4455*nxx*nyy + (96525.0/2.0)*nxx*nzz*nzz*nzz - 7959776990330879.0/274877906944.0*nxx*nzz*nzz + 270*nxx + (5306517993553919.0/274877906944.0)*nyy*nzz*nzz - 4455*nyy*nzz - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 270*nzz);  // power=9
        out[21] = T(-1470*nxx*nxx*nxy*ny*ny - 4481884272721919.0/274877906944.0*nxx*nxy*ny*nyy*ny - 84945.0/2.0*nxx*nxy*ny*nyz*nz + (8448372469923839.0/549755813888.0)*nxx*nxy*ny*ny + (7959776990330879.0/68719476736.0)*nxx*nxy*nzz*nzz - 81675.0/2.0*nxx*nxy*nzz - 1215.0/2.0*nxx*nxy - 14835*nxy*ny*nyy*nyy*ny + (4844723109887999.0/17179869184.0)*nxy*ny*nyy*nyz*nz + 6210*nxy*ny*nyy*ny - 4518477394083839.0/17179869184.0*nxy*ny*nyz*nz*nzz - 152055.0/2.0*nxy*ny*nyz*nz - 4755.0/2.0*nxy*ny*ny + (7959776990330879.0/68719476736.0)*nxy*nzz*nzz*nzz - 6122905377177599.0/68719476736.0*nxy*nzz*nzz + (70065.0/2.0)*nxy*nzz - 675.0/2.0*nxy);  // power=9
        out[22] = T(-1518749999998481.0/999999999999.0*nxx*nxx*nxx*nxz - 5364500052049919.0/68719476736.0*nxx*nxx*nxy*nyz + (6679533138739199.0/549755813888.0)*nxx*nxx*nxy*nyz + (4974860618956799.0/68719476736.0)*nxx*nxx*nxz*nz*nz + (26325.0/4.0)*nxx*nxx*nxz*nz*nz - 19305.0/2.0*nxx*nxx*nxz + 478305*nxx*nxy*ny*nyy*nz + (3965972801126399.0/137438953472.0)*nxx*nxy*ny*nyy*nz - 789345.0/2.0*nxx*nxy*nyz*nz*nz - 196425.0/2.0*nxx*nxy*nyz*nz*nz - 8813272891392001.0/137438953472.0*nxx*nxy*nyz + (96525.0/2.0)*nxx*nxz*nz*nzz*nz + 3240*nxx*nxz*nz*nzz*nz - 7075357324738559.0/137438953472.0*nxx*nxz*nz*nz + 12960*nxx*nxz + (99225.0/2.0)*nxy*ny*nyy*nyy*nz + (4174708211711999.0/274877906944.0)*nxy*ny*nyy*nyy*nz - 339390*nxy*ny*nyy*nzz*nz - 419175.0/4.0*nxy*ny*nyy*nzz*nz - 161595.0/2.0*nxy*ny*nyy*nz + 45360*nxy*nyz*nz*nzz*nz + 92340*nxy*nyz*nz*nzz*nz + (6324682940743681.0/34359738368.0)*nxy*nyz*nz*nz + (6095073989106912.0/824633720833.0)*nxy*nyz - 96525.0/4.0*nxz*nz*nzz*nzz*nz - 4859999999999999.0/1000000000000.0*nxz*nz*nzz*nzz*nz + (3316573745971199.0/137438953472.0)*nxz*nz*nzz*nz - 6345556481802239.0/549755813888.0*nxz*nz*nz - 3375.0/4.0*nxz);  // power=9
        out[23] = T(-1260*nxx*nxx*nxy*ny*ny - 4974860618956799.0/34359738368.0*nxx*nxx*nxy*nzz + (19305.0/2.0)*nxx*nxx*nxy - 2520*nxx*nxy*ny*nyy*ny + 915345*nxx*nxy*ny*nyz*nz - 57915*nxx*nxy*ny*ny - 38610*nxx*nxy*nzz*nzz - 8980261219860481.0/137438953472.0*nxx*nxy*nzz + 5670*nxx*nxy - 1260*nxy*ny*nyy*nyy*ny + (6407404010864639.0/137438953472.0)*nxy*ny*nyy*nyz*nz - 6314632717271039.0/34359738368.0*nxy*ny*nyz*nz*nzz - 133785*nxy*ny*nyz*nz + 11385*nxy*ny*ny + (7296462241136639.0/68719476736.0)*nxy*nzz*nzz*nzz - 7653631721471999.0/137438953472.0*nxy*nzz*nzz + (8906044184985601.0/274877906944.0)*nxy*nzz - 2565*nxy);  // power=9
        out[24] = T(-4974860618956799.0/34359738368.0*nxx*nxx*nyy*nzz + (19305.0/2.0)*nxx*nxx*nyy + (96525.0/2.0)*nxx*nxx*nzz*nzz - 5306517993553919.0/549755813888.0*nxx*nxx*nzz + 868725*nxx*nyy*nyy*nzz - 57915*nxx*nyy*nyy - 868725.0/2.0*nxx*nyy*nzz*nzz - 115830*nxx*nyy*nzz + (31185.0/2.0)*nxx*nyy + (96525.0/2.0)*nxx*nzz*nzz*nzz - 5306517993553921.0/137438953472.0*nxx*nzz*nzz + (31185.0/2.0)*nxx*nzz - 945*nxx - 57915*nyy*nyy*nzz + 4455*nyy*nyy + (3979888495165439.0/137438953472.0)*nyy*nzz*nzz + (22275.0/2.0)*nyy*nzz - 1620*nyy - 1768839331187857.0/549755813889.0*nzz*nzz*nzz + (2227499999997772.0/999999999999.0)*nzz*nzz - 1215*nzz + 105);  // power=9
        out[25] = T(-135135.0/2.0*nxx*nxx*nyy*nz + (10395.0/2.0)*nxx*nxx*nz + 405405*nxx*nyy*nyy*nz - 135135.0/2.0*nxx*nyy*nzz*nz - 280665.0/2.0*nxx*nyy*nz + (10395.0/2.0)*nxx*nzz*nz + (8505.0/2.0)*nxx*nz - 31185*nyy*nyy*nz + (10395.0/2.0)*nyy*nzz*nz + 14175*nyy*nz - 945.0/2.0*nzz*nz - 630*nz);  // power=8
        out[26] = T(-135135.0/2.0*nxx*nxx*nzz*nz + (31185.0/2.0)*nxx*nxx*nz + 405405*nxx*nyy*nzz*nz - 93555*nxx*nyy*nz - 135135.0/2.0*nxx*nzz*nzz*nz + 51975*nxx*nzz*nz - 14175.0/2.0*nxx*nz - 31185*nyy*nzz*nz + 8505*nyy*nz + (10395.0/2.0)*nzz*nzz*nz - 6615.0/2.0*nzz*nz + 315*nz);  // power=8
        out[27] = T(25515*nxx*nxx*nxy*nz - 42525*nxx*nxy*ny*nyz - 220185.0/2.0*nxx*nxy*nzz*nz + (31185.0/2.0)*nxx*nxy*nz - 68040*nxy*ny*nyy*nyz + 269325*nxy*ny*nyz*nz*nz - 271215.0/2.0*nxy*nzz*nzz*nz + (135135.0/2.0)*nxy*nzz*nz - 11340*nxy*nz);  // power=8
    } break;

    case 25: {
        out[0] = T(15*nxx*nx - 60*nxy*ny + 15*nxz*nz);  // power=4
        out[1] = T(105*nxx*nxx - 420*nxx*nyy + 105*nxx*nzz - 45*nxx + 60*nyy - 15*nzz);  // power=5
        out[2] = T(225*nxx*nxy - 300*nxy*ny*ny + 225*nxy*nzz);  // power=5
        out[3] = T(75*nxx*nxz - 450*nxy*nyz + 75*nxz*nz*nz);  // power=5
        out[4] = T((215.0/2.0)*nxx*nxx*nx - 485*nxx*nxy*ny + 40*nxx*nxz*nz - 70*nxx*nx + 195*nxy*ny*nyy + (255.0/2.0)*nxy*nyz*nz + (35.0/2.0)*nxy*ny - 135.0/2.0*nxz*nz*nzz + (35.0/2.0)*nxz*nz + (25.0/2.0)*nx);  // power=6
        out[5] = T(60*nxx*nxx*nx - 195*nxx*nxy*ny - 90*nxx*nxz*nz - 35*nxx*nx - 255*nxy*ny*nyy + (765.0/2.0)*nxy*nyz*nz + (385.0/2.0)*nxy*ny - 150*nxz*nz*nzz + (175.0/2.0)*nxz*nz - 25.0/2.0*nx);  // power=6
        out[6] = T(315*nxx*nxx*ny - 1260*nxx*nyy*ny + 315*nxx*nyz*nz + 175*nxx*ny + 140*nyy*ny - 35*nyz*nz - 40*ny);  // power=6
        out[7] = T(315*nxx*nxx*nz - 1260*nxx*nyy*nz + 315*nxx*nzz*nz - 175*nxx*nz + 140*nyy*nz - 35*nzz*nz + 10*nz);  // power=6
        out[8] = T(525*nxx*nxy*nz - 1050*nxy*ny*nyz + 525*nxy*nzz*nz);  // power=6
        out[9] = T((693.0/2.0)*nxx*nxx*nxx - 1386*nxx*nxx*nyy + (693.0/2.0)*nxx*nxx*nzz - 378*nxx*nxx + 756*nxx*nyy - 189*nxx*nzz + (189.0/2.0)*nxx - 42*nyy + (21.0/2.0)*nzz - 3);  // power=7
        out[10] = T((21.0/2.0)*nxx*nxx*nxy + 336*nxx*nxy*ny*ny - 147.0/2.0*nxx*nxy*nzz - 189.0/2.0*nxx*nxy - 1407*nxy*ny*nyy*ny + (483.0/2.0)*nxy*ny*nyz*nz + (2331.0/2.0)*nxy*ny*ny - 84*nxy*nzz*nzz - 357.0/2.0*nxy);  // power=7
        out[11] = T(-42*nxx*nxx*nxz + 294*nxx*nxy*nyz + (777.0/2.0)*nxx*nxz*nz*nz - 189.0/2.0*nxx*nxz + 336*nxy*ny*nyy*nz - 966*nxy*nyz*nz*nz + (861.0/2.0)*nxz*nz*nzz*nz - 819.0/2.0*nxz*nz*nz + 84*nxz);  // power=7
        out[12] = T((2079.0/2.0)*nxx*nxx*nxy - 4158*nxx*nxy*ny*ny + (2079.0/2.0)*nxx*nxy*nzz + (189.0/2.0)*nxx*nxy + 1134*nxy*ny*ny - 567.0/2.0*nxy*nzz - 189*nxy);  // power=7
        out[13] = T((2079.0/2.0)*nxx*nxx*nxz - 4158*nxx*nxy*nyz + (2079.0/2.0)*nxx*nxz*nz*nz - 1701.0/2.0*nxx*nxz + 1134*nxy*nyz - 567.0/2.0*nxz*nz*nz + 126*nxz);  // power=7
        out[14] = T((63.0/2.0)*nxx*nxx*nxz + 1323*nxx*nxy*nyz - 63.0/2.0*nxx*nxz*nz*nz - 189.0/2.0*nxx*nxz - 3906*nxy*ny*nyy*nz + (2457.0/2.0)*nxy*nyz*nz*nz + (2961.0/2.0)*nxy*nyz - 63*nxz*nz*nzz*nz - 189.0/2.0*nxz);  // power=7
        out[15] = T(2079*nxx*nxx*nyz - 8316*nxx*nyy*nyz + 2079*nxx*nyz*nz*nz + 567*nxx*nyz + 756*nyy*nyz - 189*nyz*nz*nz - 126*nyz);  // power=7
        out[16] = T((1287.0/8.0)*nxx*nxx*nxx*nx + (3.0/2.0)*nxx*nxx*nxx*nx - 1287.0/2.0*nxx*nxx*nxy*ny - 9*nxx*nxx*nxy*ny - 6435.0/8.0*nxx*nxx*nxz*nz - 45.0/2.0*nxx*nxx*nxz*nz - 297.0/2.0*nxx*nxx*nx - 45.0/2.0*nxx*nxy*ny*nyy + 3861*nxx*nxy*nyz*nz + 225*nxx*nxy*nyz*nz + 198*nxx*nxy*ny - 6435.0/8.0*nxx*nxz*nz*nzz + 693*nxx*nxz*nz + 27*nxx*nx - 12*nxy*ny*nyy*nyy + (495.0/2.0)*nxy*ny*nyy*nzz - 360*nxy*nyz*nz*nzz - 891*nxy*nyz*nz + (27.0/2.0)*nxy*ny + (1287.0/8.0)*nxz*nz*nzz*nzz + 24*nxz*nz*nzz*nzz - 54*nxz*nz - 3.0/2.0*nx);  // power=8
        out[17] = T((15.0/4.0)*nxx*nxx*nxx*nx - 45.0/2.0*nxx*nxx*nxy*ny - 225.0/4.0*nxx*nxx*nxz*nz - 36*nxx*nxy*ny*nyy - 225.0/4.0*nxx*nxy*ny*nyy - 5961.0/4.0*nxx*nxy*nyz*nz + (1125.0/2.0)*nxx*nxy*nyz*nz + (525.0/4.0)*nxx*nxy*ny + (1287.0/8.0)*nxx*nxz*nz*nzz - 75.0/8.0*nxx*nx - 6723.0/8.0*nxy*ny*nyy*nyy - 30*nxy*ny*nyy*nyy + 3300*nxy*ny*nyy*nzz + (2475.0/4.0)*nxy*ny*nyy*nzz + (5793.0/8.0)*nxy*ny*nyy - 525*nxy*nyz*nz*nzz - 900*nxy*nyz*nz*nzz - 3909.0/4.0*nxy*nyz*nz - 1395.0/8.0*nxy*ny + (1287.0/8.0)*nxz*nz*nzz*nzz + 60*nxz*nz*nzz*nzz - 891.0/4.0*nxz*nz*nzz + (1005.0/8.0)*nxz*nz + (45.0/8.0)*nx);  // power=8
        out[18] = T((1287.0/2.0)*nxx*nxx*nyy*ny - 3861.0/2.0*nxx*nxx*nyz*nz - 2574*nxx*nyy*nyy*ny + (16731.0/2.0)*nxx*nyy*nyz*nz + (2079.0/2.0)*nxx*nyy*ny - 3861.0/2.0*nxx*nyz*nz*nzz - 297.0/2.0*nxx*nyz*nz - 135*nxx*ny + 198*nyy*nyy*ny - 1287.0/2.0*nyy*nyz*nz - 108*nyy*ny + (297.0/2.0)*nyz*nz*nzz + 54*nyz*nz + 15*ny);  // power=8
        out[19] = T((1287.0/2.0)*nxx*nxx*nxx*ny - 2574*nxx*nxx*nyy*ny - 1287*nxx*nxx*nyz*nz - 99.0/2.0*nxx*nxx*ny + 7722*nxx*nyy*nyz*nz + 594*nxx*nyy*ny - 3861.0/2.0*nxx*nyz*nz*nzz - 297*nxx*nyz*nz - 54*nxx*ny - 594*nyy*nyz*nz + (297.0/2.0)*nyz*nz*nzz + 54*nyz*nz);  // power=8
        out[20] = T((1287.0/2.0)*nxx*nxx*nxx*nz - 2574*nxx*nxx*nyy*nz - 1089.0/2.0*nxx*nxx*nz + 2574*nxx*nyy*nzz*nz + 594*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + 297*nxx*nzz*nz + 81*nxx*nz - 198*nyy*nzz*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz);  // power=8
        out[21] = T((405.0/2.0)*nxx*nxx*nxy*nz + 666*nxx*nxy*ny*nyz - 981*nxx*nxy*nzz*nz - 2754*nxy*ny*nyy*nyz + (2497499999997502.0/999999999999.0)*nxy*ny*nyz*nz*nz + (2943.0/2.0)*nxy*ny*nyz - 2367.0/2.0*nxy*nzz*nzz*nz + (1287.0/2.0)*nxy*nzz*nz - 675.0/2.0*nxy*nz);  // power=8
        out[22] = T((21.0/4.0)*nxx*nxx*nxx*nx + (3861.0/4.0)*nxx*nxx*nxy*ny - 63.0/2.0*nxx*nxx*nxy*ny - 3861.0/4.0*nxx*nxx*nxz*nz - 315.0/4.0*nxx*nxx*nxz*nz - 4050*nxx*nxy*ny*nyy - 315.0/4.0*nxx*nxy*ny*nyy + 3357*nxx*nxy*nyz*nz + (1575.0/2.0)*nxx*nxy*nyz*nz + (1737.0/2.0)*nxx*nxy*ny - 1287.0/2.0*nxx*nxz*nz*nzz + (1485.0/2.0)*nxx*nxz*nz - 153.0/2.0*nxx*nx - 189*nxy*ny*nyy*nyy - 42*nxy*ny*nyy*nyy + 3168*nxy*ny*nyy*nzz + (3465.0/4.0)*nxy*ny*nyy*nzz + 909*nxy*ny*nyy - 504*nxy*nyz*nz*nzz - 1260*nxy*nyz*nz*nzz - 2043*nxy*nyz*nz - 1107.0/4.0*nxy*ny + (1287.0/4.0)*nxz*nz*nzz*nzz + 84*nxz*nz*nzz*nzz - 891.0/4.0*nxz*nz*nzz + 72*nxz*nz + (75.0/4.0)*nx);  // power=8
        out[23] = T((4185.0/2.0)*nxx*nxx*nxy*nz - 7992*nxx*nxy*ny*nyz + 1017*nxx*nxy*nzz*nz - 432*nxy*ny*nyy*nyz + 1710*nxy*ny*nyz*nz*nz + 1782*nxy*ny*nyz - 2151.0/2.0*nxy*nzz*nzz*nz + 198*nxy*nzz*nz - 270*nxy*nz);  // power=8
        out[24] = T((3861.0/2.0)*nxx*nxx*nyy*nz - 1287.0/2.0*nxx*nxx*nzz*nz - 7722*nxx*nyy*nyy*nz + (2476375063663000.0/549755813889.0)*nxx*nyy*nzz*nz + (3267.0/2.0)*nxx*nyy*nz - 1287.0/2.0*nxx*nzz*nzz*nz + (891.0/2.0)*nxx*nzz*nz - 135*nxx*nz + 594*nyy*nyy*nz - 693.0/2.0*nyy*nzz*nz - 189*nyy*nz + (99.0/2.0)*nzz*nzz*nz - 27*nzz*nz + 15*nz);  // power=8
        out[25] = T((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nyy + (2079.0/2.0)*nxx*nyy*nzz + (3213.0/2.0)*nxx*nyy - 189.0/2.0*nxx*nzz - 63*nxx + 378*nyy*nyy - 189.0/2.0*nyy*nzz - 441.0/2.0*nyy + 45.0/2.0);  // power=7
        out[26] = T((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx - 4158*nxx*nyy*nzz + 378*nxx*nyy + (2079.0/2.0)*nxx*nzz*nzz - 756*nxx*nzz + (189.0/2.0)*nxx + 378*nyy*nzz - 189.0/2.0*nzz*nzz + (189.0/2.0)*nzz - 45);  // power=7
        out[27] = T(-126*nxx*nxx*nxy + 126*nxx*nxy*ny*ny + (3843.0/2.0)*nxx*nxy*nzz - 189.0/2.0*nxx*nxy + 252*nxy*ny*nyy*ny - 2898*nxy*ny*nyz*nz + (4095.0/2.0)*nxy*nzz*nzz - 1701.0/2.0*nxy*nzz + 63*nxy);  // power=7
    } break;

    case 26: {
        out[0] = T(15*nxx*nx + 15*nxy*ny - 60*nxz*nz);  // power=4
        out[1] = T(105*nxx*nxx + 105*nxx*nyy - 420*nxx*nzz - 45*nxx - 15*nyy + 60*nzz);  // power=5
        out[2] = T(75*nxx*nxy + 75*nxy*ny*ny - 450*nxy*nzz);  // power=5
        out[3] = T(225*nxx*nxz + 225*nxy*nyz - 300*nxz*nz*nz);  // power=5
        out[4] = T((215.0/2.0)*nxx*nxx*nx + 40*nxx*nxy*ny - 485*nxx*nxz*nz - 70*nxx*nx - 135.0/2.0*nxy*ny*nyy + (255.0/2.0)*nxy*nyz*nz + (35.0/2.0)*nxy*ny + 195*nxz*nz*nzz + (35.0/2.0)*nxz*nz + (25.0/2.0)*nx);  // power=6
        out[5] = T((95.0/2.0)*nxx*nxx*nx + 130*nxx*nxy*ny - 290*nxx*nxz*nz - 35*nxx*nx + (165.0/2.0)*nxy*ny*nyy - 255*nxy*nyz*nz - 70*nxy*ny + 450*nxz*nz*nzz - 175*nxz*nz + 25*nx);  // power=6
        out[6] = T(315*nxx*nxx*ny + 315*nxx*nyy*ny - 1260*nxx*nyz*nz - 175*nxx*ny - 35*nyy*ny + 140*nyz*nz + 10*ny);  // power=6
        out[7] = T(315*nxx*nxx*nz + 315*nxx*nyy*nz - 1260*nxx*nzz*nz + 175*nxx*nz - 35*nyy*nz + 140*nzz*nz - 40*nz);  // power=6
        out[8] = T(525*nxx*nxy*nz + 525*nxy*ny*nyz - 1050*nxy*nzz*nz);  // power=6
        out[9] = T((693.0/2.0)*nxx*nxx*nxx + (693.0/2.0)*nxx*nxx*nyy - 1386*nxx*nxx*nzz - 378*nxx*nxx - 189*nxx*nyy + 756*nxx*nzz + (189.0/2.0)*nxx + (21.0/2.0)*nyy - 42*nzz - 3);  // power=7
        out[10] = T(-42*nxx*nxx*nxy + (777.0/2.0)*nxx*nxy*ny*ny + 294*nxx*nxy*nzz - 189.0/2.0*nxx*nxy + (861.0/2.0)*nxy*ny*nyy*ny - 966*nxy*ny*nyz*nz - 819.0/2.0*nxy*ny*ny + 336*nxy*nzz*nzz + 84*nxy);  // power=7
        out[11] = T((21.0/2.0)*nxx*nxx*nxz - 147.0/2.0*nxx*nxy*nyz + 336*nxx*nxz*nz*nz - 189.0/2.0*nxx*nxz - 84*nxy*ny*nyy*nz + (483.0/2.0)*nxy*nyz*nz*nz - 1407*nxz*nz*nzz*nz + (2331.0/2.0)*nxz*nz*nz - 357.0/2.0*nxz);  // power=7
        out[12] = T((2079.0/2.0)*nxx*nxx*nxy + (2079.0/2.0)*nxx*nxy*ny*ny - 4158*nxx*nxy*nzz - 1701.0/2.0*nxx*nxy - 567.0/2.0*nxy*ny*ny + 1134*nxy*nzz + 126*nxy);  // power=7
        out[13] = T((2079.0/2.0)*nxx*nxx*nxz + (2079.0/2.0)*nxx*nxy*nyz - 4158*nxx*nxz*nz*nz + (189.0/2.0)*nxx*nxz - 567.0/2.0*nxy*nyz + 1134*nxz*nz*nz - 189*nxz);  // power=7
        out[14] = T(-126*nxx*nxx*nxz + (3591.0/2.0)*nxx*nxy*nyz + 126*nxx*nxz*nz*nz - 189.0/2.0*nxx*nxz + (3843.0/2.0)*nxy*ny*nyy*nz - 3024*nxy*nyz*nz*nz - 1449.0/2.0*nxy*nyz + 252*nxz*nz*nzz*nz + 63*nxz);  // power=7
        out[15] = T(2079*nxx*nxx*nyz + 2079*nxx*nyy*nyz - 8316*nxx*nyz*nz*nz + 567*nxx*nyz - 189*nyy*nyz + 756*nyz*nz*nz - 126*nyz);  // power=7
        out[16] = T((1287.0/8.0)*nxx*nxx*nxx*nx - 3.0/8.0*nxx*nxx*nxx*nx + (1287.0/8.0)*nxx*nxx*nxy*ny + (9.0/4.0)*nxx*nxx*nxy*ny - 6435.0/4.0*nxx*nxx*nxz*nz + (45.0/8.0)*nxx*nxx*nxz*nz - 297.0/2.0*nxx*nxx*nx + (45.0/8.0)*nxx*nxy*ny*nyy - 3861.0/4.0*nxx*nxy*nyz*nz - 225.0/4.0*nxx*nxy*nyz*nz - 99.0/2.0*nxx*nxy*ny + (32175.0/8.0)*nxx*nxz*nz*nzz - 1089.0/2.0*nxx*nxz*nz + (189.0/2.0)*nxx*nx + 3*nxy*ny*nyy*nyy - 495.0/8.0*nxy*ny*nyy*nzz + 90*nxy*nyz*nz*nzz + (891.0/4.0)*nxy*nyz*nz - 27.0/8.0*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz - 6*nxz*nz*nzz*nzz - 185624999999072.0/999999999995.0*nxz*nz*nzz + (459.0/4.0)*nxz*nz - 117.0/8.0*nx);  // power=8
        out[17] = T(-75.0/8.0*nxx*nxx*nxx*nx + (225.0/4.0)*nxx*nxx*nxy*ny + (1125.0/8.0)*nxx*nxx*nxz*nz + (1917.0/8.0)*nxx*nxy*ny*nyy + (1125.0/8.0)*nxx*nxy*ny*nyy - 3021.0/4.0*nxx*nxy*nyz*nz - 5625.0/4.0*nxx*nxy*nyz*nz - 105.0/2.0*nxx*nxy*ny + (1287.0/8.0)*nxx*nxz*nz*nzz + (15.0/4.0)*nxx*nx + (1917.0/8.0)*nxy*ny*nyy*nyy + 75*nxy*ny*nyy*nyy - 1320*nxy*ny*nyy*nzz - 12375.0/8.0*nxy*ny*nyy*nzz - 921.0/4.0*nxy*ny*nyy + 210*nxy*nyz*nz*nzz + 2250*nxy*nyz*nz*nzz + (2811.0/4.0)*nxy*nyz*nz + (315.0/8.0)*nxy*ny - 1287.0/2.0*nxz*nz*nzz*nzz - 150*nxz*nz*nzz*nzz + (5643.0/8.0)*nxz*nz*nzz - 465.0/2.0*nxz*nz + (45.0/8.0)*nx);  // power=8
        out[18] = T((1287.0/2.0)*nxx*nxx*nyy*ny - 3861.0/2.0*nxx*nxx*nyz*nz + (1287.0/2.0)*nxx*nyy*nyy*ny - 2476375063663000.0/549755813889.0*nxx*nyy*nyz*nz - 891.0/2.0*nxx*nyy*ny + 7722*nxx*nyz*nz*nzz - 3267.0/2.0*nxx*nyz*nz + 135*nxx*ny - 99.0/2.0*nyy*nyy*ny + (693.0/2.0)*nyy*nyz*nz + 27*nyy*ny - 594*nyz*nz*nzz + 189*nyz*nz - 15*ny);  // power=8
        out[19] = T((1287.0/2.0)*nxx*nxx*nxx*ny + (1287.0/2.0)*nxx*nxx*nyy*ny - 2476375063663000.0/549755813889.0*nxx*nxx*nyz*nz - 1089.0/2.0*nxx*nxx*ny - 3861.0/2.0*nxx*nyy*nyz*nz - 297.0/2.0*nxx*nyy*ny + 7722*nxx*nyz*nz*nzz - 2079.0/2.0*nxx*nyz*nz + 216*nxx*ny + (297.0/2.0)*nyy*nyz*nz - 594*nyz*nz*nzz + 189*nyz*nz - 15*ny);  // power=8
        out[20] = T((1287.0/2.0)*nxx*nxx*nxx*nz + (1287.0/2.0)*nxx*nxx*nyy*nz - 3217499999996782.0/999999999999.0*nxx*nxx*nzz*nz - 99.0/2.0*nxx*nxx*nz - 1287.0/2.0*nxx*nyy*nzz*nz - 297.0/2.0*nxx*nyy*nz + 2574*nxx*nzz*nzz*nz - 445499999999554.0/999999999999.0*nxx*nzz*nz + 81*nxx*nz + (99.0/2.0)*nyy*nzz*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
        out[21] = T(-405.0/2.0*nxx*nxx*nxy*nz + 891*nxx*nxy*ny*nyz - 306*nxx*nxy*nzz*nz + (2187.0/2.0)*nxy*ny*nyy*nyz - 2227499999997772.0/999999999999.0*nxy*ny*nyz*nz*nz - 1107.0/2.0*nxy*ny*nyz + 3114*nxy*nzz*nzz*nz - 3663.0/2.0*nxy*nzz*nz + (675.0/2.0)*nxy*nz);  // power=8
        out[22] = T(-39.0/4.0*nxx*nxx*nxx*nx + (3861.0/4.0)*nxx*nxx*nxy*ny + (117.0/2.0)*nxx*nxx*nxy*ny - 3861.0/4.0*nxx*nxx*nxz*nz + (585.0/4.0)*nxx*nxx*nxz*nz + (2025.0/2.0)*nxx*nxy*ny*nyy + (585.0/4.0)*nxx*nxy*ny*nyy - 11331.0/2.0*nxx*nxy*nyz*nz - 2925.0/2.0*nxx*nxy*nyz*nz - 774*nxx*nxy*ny + (16731.0/4.0)*nxx*nxz*nz*nzz - 1485.0/2.0*nxx*nxz*nz + (279.0/4.0)*nxx*nx + (189.0/4.0)*nxy*ny*nyy*nyy + 78*nxy*ny*nyy*nyy - 792*nxy*ny*nyy*nzz - 6435.0/4.0*nxy*ny*nyy*nzz - 909.0/4.0*nxy*ny*nyy + 126*nxy*nyz*nz*nzz + 2340*nxy*nyz*nz*nzz + (3249.0/2.0)*nxy*nyz*nz + (189.0/2.0)*nxy*ny - 1287*nxz*nz*nzz*nzz - 156*nxz*nz*nzz*nzz + (519749999999480.0/999999999999.0)*nxz*nz*nzz - 477.0/4.0*nxz*nz - 15.0/2.0*nx);  // power=8
        out[23] = T(1890*nxx*nxx*nxy*nz + 1998*nxx*nxy*ny*nyz - 8298*nxx*nxy*nzz*nz + 108*nxy*ny*nyy*nyz - 855.0/2.0*nxy*ny*nyz*nz*nz - 891.0/2.0*nxy*ny*nyz + (2681999999999999.0/1000000000000.0)*nxy*nzz*nzz*nz - 49499999999950.0/999999999999.0*nxy*nzz*nz + (135.0/2.0)*nxy*nz);  // power=8
        out[24] = T((3861.0/2.0)*nxx*nxx*nyy*nz - 1287.0/2.0*nxx*nxx*nzz*nz + (3861.0/2.0)*nxx*nyy*nyy*nz - 16731.0/2.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + 2574*nxx*nzz*nzz*nz - 2079.0/2.0*nxx*nzz*nz + 135*nxx*nz - 297.0/2.0*nyy*nyy*nz + (1287.0/2.0)*nyy*nzz*nz - 54*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
        out[25] = T((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy - 4158*nxx*nyy*nzz - 756*nxx*nyy + 378*nxx*nzz + (189.0/2.0)*nxx - 189.0/2.0*nyy*nyy + 378*nyy*nzz + (189.0/2.0)*nyy - 45);  // power=7
        out[26] = T((2079.0/2.0)*nxx*nxx*nzz - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 4158*nxx*nzz*nzz + (3213.0/2.0)*nxx*nzz - 63*nxx - 189.0/2.0*nyy*nzz + 378*nzz*nzz - 441.0/2.0*nzz + 45.0/2.0);  // power=7
        out[27] = T((63.0/2.0)*nxx*nxx*nxy - 63.0/2.0*nxx*nxy*ny*ny + 819*nxx*nxy*nzz - 189.0/2.0*nxx*nxy - 63*nxy*ny*nyy*ny + (1449.0/2.0)*nxy*ny*nyz*nz - 4410*nxy*nzz*nzz + (3969.0/2.0)*nxy*nzz - 189.0/2.0*nxy);  // power=7
    } break;

    case 27: {
        out[0] = T(15*nxx*ny + 15*nyy*ny - 60*nyz*nz);  // power=4
        out[1] = T(75*nxx*nxy + 75*nxy*ny*ny - 450*nxy*nzz);  // power=5
        out[2] = T(105*nxx*nyy - 15*nxx + 105*nyy*nyy - 420*nyy*nzz - 45*nyy + 60*nzz);  // power=5
        out[3] = T(225*nxx*nyz + 225*nyy*nyz - 300*nyz*nz*nz);  // power=5
        out[4] = T(105*nxx*nxx*ny + (105.0/2.0)*nxx*nyy*ny - 945.0/2.0*nxx*nyz*nz - 35*nxx*ny - 105.0/2.0*nyy*nyy*ny + (315.0/2.0)*nyy*nyz*nz + 35*nyy*ny + 210*nyz*nz*nzz - 140*nyz*nz + 5*ny);  // power=6
        out[5] = T((105.0/2.0)*nxx*nxx*ny + 105*nxx*nyy*ny - 315*nxx*nyz*nz - 35*nxx*ny + (105.0/2.0)*nyy*nyy*ny - 315*nyy*nyz*nz - 35*nyy*ny + 420*nyz*nz*nzz - 140*nyz*nz + 20*ny);  // power=6
        out[6] = T(-20*nxx*nxx*nx + 415*nxx*nxy*ny + 100*nxx*nxz*nz - 35*nxx*nx + 435*nxy*ny*nyy - 1020*nxy*nyz*nz - 315*nxy*ny + 120*nxz*nz*nzz + 30*nx);  // power=6
        out[7] = T(525*nxx*nxy*nz + 525*nxy*ny*nyz - 1050*nxy*nzz*nz);  // power=6
        out[8] = T(315*nxx*nyy*nz - 35*nxx*nz + 315*nyy*nyy*nz - 1260*nyy*nzz*nz + 175*nyy*nz + 140*nzz*nz - 40*nz);  // power=6
        out[9] = T((693.0/2.0)*nxx*nxx*nxy + (693.0/2.0)*nxx*nxy*ny*ny - 1386*nxx*nxy*nzz - 567.0/2.0*nxx*nxy - 189.0/2.0*nxy*ny*ny + 378*nxy*nzz + 42*nxy);  // power=7
        out[10] = T((693.0/2.0)*nxx*nyy*nyy - 189*nxx*nyy + (21.0/2.0)*nxx + (693.0/2.0)*nyy*nyy*nyy - 1386*nyy*nyy*nzz - 378*nyy*nyy + 756*nyy*nzz + (189.0/2.0)*nyy - 42*nzz - 3);  // power=7
        out[11] = T((693.0/2.0)*nxx*nyz*nz*nz - 189.0/2.0*nxx*nyz + (693.0/2.0)*nyy*nyz*nz*nz - 189.0/2.0*nyy*nyz - 1386*nyz*nz*nzz*nz + 1134*nyz*nz*nz - 168*nyz);  // power=7
        out[12] = T((2079.0/2.0)*nxx*nxx*nyy - 189.0/2.0*nxx*nxx + (2079.0/2.0)*nxx*nyy*nyy - 4158*nxx*nyy*nzz - 756*nxx*nyy + 378*nxx*nzz + (189.0/2.0)*nxx - 189.0/2.0*nyy*nyy + 378*nyy*nzz + (189.0/2.0)*nyy - 45);  // power=7
        out[13] = T((2079.0/2.0)*nxx*nxx*nyz + (2079.0/2.0)*nxx*nyy*nyz - 4158*nxx*nyz*nz*nz + (567.0/2.0)*nxx*nyz - 189.0/2.0*nyy*nyz + 378*nyz*nz*nz - 63*nyz);  // power=7
        out[14] = T((2079.0/2.0)*nxx*nyy*nyz - 567.0/2.0*nxx*nyz + (2079.0/2.0)*nyy*nyy*nyz - 4158*nyy*nyz*nz*nz + (189.0/2.0)*nyy*nyz + 1134*nyz*nz*nz - 189*nyz);  // power=7
        out[15] = T(-252*nxx*nxx*nxz + 3591*nxx*nxy*nyz + 252*nxx*nxz*nz*nz - 189*nxx*nxz + 3843*nxy*ny*nyy*nz - 6048*nxy*nyz*nz*nz - 1449*nxy*nyz + 504*nxz*nz*nzz*nz + 126*nxz);  // power=7
        out[16] = T((1287.0/8.0)*nxx*nxx*nxx*ny + (1287.0/8.0)*nxx*nxx*nyy*ny - 6435.0/4.0*nxx*nxx*nyz*nz - 99*nxx*nxx*ny - 3861.0/4.0*nxx*nyy*nyz*nz + (32175.0/8.0)*nxx*nyz*nz*nzz - 891*nxx*nyz*nz + (135.0/2.0)*nxx*ny + (1287.0/8.0)*nyy*nyz*nz*nzz - 1287.0/2.0*nyz*nz*nzz*nzz + 396*nyz*nz*nzz - 135.0/2.0*nyz*nz);  // power=8
        out[17] = T((1287.0/8.0)*nxx*nyy*nyy*ny - 3861.0/4.0*nxx*nyy*nyz*nz - 99.0/2.0*nxx*nyy*ny + (1287.0/8.0)*nxx*nyz*nz*nzz + (297.0/2.0)*nxx*nyz*nz + (1287.0/8.0)*nyy*nyy*nyy*ny - 6435.0/4.0*nyy*nyy*nyz*nz - 297.0/2.0*nyy*nyy*ny + (32175.0/8.0)*nyy*nyz*nz*nzz - 1089.0/2.0*nyy*nyz*nz + (189.0/2.0)*nyy*ny - 1287.0/2.0*nyz*nz*nzz*nzz - 198*nyz*nz*nzz + (243.0/2.0)*nyz*nz - 15*ny);  // power=8
        out[18] = T(-18*nxx*nxx*nxx*nx + 108*nxx*nxx*nxy*ny + 270*nxx*nxx*nxz*nz + 864*nxx*nxy*ny*nyy + 270*nxx*nxy*ny*nyy - 2685.0/2.0*nxx*nxy*nyz*nz - 2700*nxx*nxy*nyz*nz - 591.0/2.0*nxx*nxy*ny + (297.0/2.0)*nxx*nxz*nz + (21.0/2.0)*nxx*nx + 864*nxy*ny*nyy*nyy + 144*nxy*ny*nyy*nyy - 3696*nxy*ny*nyy*nzz - 2970*nxy*ny*nyy*nzz - 912*nxy*ny*nyy + 588*nxy*nyz*nz*nzz + 4320*nxy*nyz*nz*nzz + (3579.0/2.0)*nxy*nyz*nz + (423.0/2.0)*nxy*ny - 288*nxz*nz*nzz*nzz - 249.0/2.0*nxz*nz - 15.0/2.0*nx);  // power=8
        out[19] = T(-18*nxx*nxx*nxx*nx + (1287.0/2.0)*nxx*nxx*nxy*ny + 108*nxx*nxx*nxy*ny + 270*nxx*nxx*nxz*nz - 99.0/2.0*nxx*nxx*nx + 738*nxx*nxy*ny*nyy + 270*nxx*nxy*ny*nyy - 2337836598562972.0/549755813889.0*nxx*nxy*nyz*nz - 2700*nxx*nxy*nyz*nz - 1017.0/2.0*nxx*nxy*ny + (693.0/2.0)*nxx*nxz*nz + (63.0/2.0)*nxx*nx + (189.0/2.0)*nxy*ny*nyy*nyy + 144*nxy*ny*nyy*nyy - 1584*nxy*ny*nyy*nzz - 2970*nxy*ny*nyy*nzz - 315.0/2.0*nxy*ny*nyy + 252*nxy*nyz*nz*nzz + 4320*nxy*nyz*nz*nzz + (3231.0/2.0)*nxy*nyz*nz + (81.0/2.0)*nxy*ny - 288*nxz*nz*nzz*nzz - 261.0/2.0*nxz*nz - 3.0/2.0*nx);  // power=8
        out[20] = T(603*nxx*nxx*nxy*nz + 711*nxx*nxy*ny*nyz - 3150*nxx*nxy*nzz*nz + 99*nxx*nxy*nz + 108*nxy*ny*nyy*nyz - 855.0/2.0*nxy*ny*nyz*nz*nz - 297.0/2.0*nxy*ny*nyz + (2681999999999999.0/1000000000000.0)*nxy*nzz*nzz*nz - 2475.0/2.0*nxy*nzz*nz + (351.0/2.0)*nxy*nz);  // power=8
        out[21] = T((1287.0/2.0)*nxx*nyy*nyy*nz - 1287.0/2.0*nxx*nyy*nzz*nz - 297.0/2.0*nxx*nyy*nz + (99.0/2.0)*nxx*nzz*nz + (1287.0/2.0)*nyy*nyy*nyy*nz - 3217499999996782.0/999999999999.0*nyy*nyy*nzz*nz - 99.0/2.0*nyy*nyy*nz + 2574*nyy*nzz*nzz*nz - 445499999999554.0/999999999999.0*nyy*nzz*nz + 81*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
        out[22] = T((3861.0/4.0)*nxx*nxx*nyy*ny - 3861.0/4.0*nxx*nxx*nyz*nz - 297.0/2.0*nxx*nxx*ny + (3861.0/4.0)*nxx*nyy*nyy*ny - 11583.0/2.0*nxx*nyy*nyz*nz - 1485.0/2.0*nxx*nyy*ny + (16731.0/4.0)*nxx*nyz*nz*nzz - 297.0/2.0*nxx*nyz*nz + (297.0/2.0)*nxx*ny - 3861.0/4.0*nyy*nyy*nyz*nz + (16731.0/4.0)*nyy*nyz*nz*nzz - 1485.0/2.0*nyy*nyz*nz + (135.0/2.0)*nyy*ny - 1287*nyz*nz*nzz*nzz + 198*nyz*nz*nzz + 54*nyz*nz - 15*ny);  // power=8
        out[23] = T((3861.0/2.0)*nxx*nxx*nyy*nz - 297.0/2.0*nxx*nxx*nz + (3861.0/2.0)*nxx*nyy*nyy*nz - 16731.0/2.0*nxx*nyy*nzz*nz + (297.0/2.0)*nxx*nyy*nz + (1287.0/2.0)*nxx*nzz*nz - 54*nxx*nz - 1287.0/2.0*nyy*nyy*nzz*nz + 2574*nyy*nzz*nzz*nz - 2079.0/2.0*nyy*nzz*nz + 135*nyy*nz - 198*nzz*nzz*nz + 108*nzz*nz - 15*nz);  // power=8
        out[24] = T(-1053.0/2.0*nxx*nxx*nxy*nz + 2538*nxx*nxy*ny*nyz + 234*nxx*nxy*nzz*nz - 297*nxx*nxy*nz + (3064499999996936.0/999999999999.0)*nxy*ny*nyy*nyz - 11655.0/2.0*nxy*ny*nyz*nz*nz - 3321.0/2.0*nxy*ny*nyz + 3978*nxy*nzz*nzz*nz - 3663.0/2.0*nxy*nzz*nz + (1107.0/2.0)*nxy*nz);  // power=8
        out[25] = T(-126*nxx*nxx*nxy + (2331.0/2.0)*nxx*nxy*ny*ny + 882*nxx*nxy*nzz - 567.0/2.0*nxx*nxy + (2583.0/2.0)*nxy*ny*nyy*ny - 2898*nxy*ny*nyz*nz - 2457.0/2.0*nxy*ny*ny + 1008*nxy*nzz*nzz + 252*nxy);  // power=7
        out[26] = T((63.0/2.0)*nxx*nxx*nxy - 63.0/2.0*nxx*nxy*ny*ny + 819*nxx*nxy*nzz - 189.0/2.0*nxx*nxy - 63*nxy*ny*nyy*ny + (1449.0/2.0)*nxy*ny*nyz*nz - 4410*nxy*nzz*nzz + (3969.0/2.0)*nxy*nzz - 189.0/2.0*nxy);  // power=7
        out[27] = T((2079.0/2.0)*nxx*nyy*nzz - 189.0/2.0*nxx*nyy - 189.0/2.0*nxx*nzz + (2079.0/2.0)*nyy*nyy*nzz - 189.0/2.0*nyy*nyy - 4158*nyy*nzz*nzz + (3213.0/2.0)*nyy*nzz - 63*nyy + 378*nzz*nzz - 441.0/2.0*nzz + 45.0/2.0);  // power=7
    } break;

    default: break;  // rows 28-31 are zero (padding)
    }
}

__host__ __device__ __forceinline__
void compute_m2l_row_device(int row, far_real_t nx, far_real_t ny, far_real_t nz,
                             far_real_t out[32]) {
    compute_m2l_row_device<far_real_t>(row, nx, ny, nz, out);
}


#endif // __CUDACC__

// =============================================================================
// Memory Management Helper Declarations
// =============================================================================
// Implementations are in lbvh_fmm_memory.cu
// =============================================================================

// Free all GPU buffers in FMMHostData and zero the struct
void free_fmm_host_memory(FMMHostData& hd);

// Allocate (or re-allocate) all GPU buffers in FMMHostData
void ensure_fmm_host_memory(FMMHostData& hd, int num_particles, int max_nodes);

// Free all GPU buffers in FMMDeviceData and zero the struct
void free_fmm_device_data(FMMDeviceData& dd);

// Allocate all GPU buffers in FMMDeviceData
void allocate_fmm_device_data(FMMDeviceData& dd, int num_particles, int num_nodes);

// =============================================================================
// TDT (Target-Driven Traversal) Queue Structure
// =============================================================================
// CSR-format output for target-driven traversal.
// Each target node A owns a fixed-width row in `sources`.
// `counts[A]` is the number of valid entries in row A.
// `offsets` is filled by a post-kernel thrust::exclusive_scan on counts.
// All three device pointers are allocated by the caller.
// =============================================================================

struct TDTQueue {
    int* sources;       // Device: flat CSR data [num_nodes * max_per_node]
    int* counts;        // Device: per-target interaction count [num_nodes]
    int* offsets;       // Device: prefix-sum of counts [num_nodes + 1]
    int  max_per_node;  // Fixed row width
    int  num_nodes;     // Number of target nodes

    __host__ __device__
    TDTQueue()
        : sources(nullptr), counts(nullptr), offsets(nullptr),
          max_per_node(0), num_nodes(0) {}
};

// =============================================================================
// Variable-length CSR Queue for two-pass TDT
// =============================================================================
// After pass 1 (count-only), offsets = exclusive_scan(counts).
// sources is a flat array of size offsets[num_nodes] (= sum of all counts).
// Row A occupies sources[offsets[A] .. offsets[A]+counts[A]).
// max_per_node is unused — no per-row cap, no dropped interactions.
// =============================================================================

struct TDTVarQueue {
    int* sources;          // Device: flat variable-length CSR [total_interactions]
    int* counts;           // Device: per-target count [num_nodes]
    int* offsets;          // Device: exclusive prefix-sum [num_nodes + 1]
    int  num_nodes;        // Number of target nodes
    int  total_interactions; // sum(counts) — filled after pass 1

    __host__ __device__
    TDTVarQueue()
        : sources(nullptr), counts(nullptr), offsets(nullptr),
          num_nodes(0), total_interactions(0) {}
};

} // namespace fmm

#endif // CUDADEM_FMM_TYPES_H

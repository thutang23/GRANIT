#include "let_force_polynomial.cuh"
// src/domain/let_kernels_fmm.cu
// =============================================================================
// FMM LET Kernels: Discover, Materialize, and Remote Gravity
// =============================================================================
// discover_let_fmm_kernel: walks disjoint start-level subtrees and emits exact
//   destination-mask descriptors plus sparse particle counts.
// materialize_let_fmm_kernel: serializes selected mask buckets once and folds
//   all remaining traffic into exact per-peer residual/particle segments.
// remote_gravity_kernel: Evaluates remote LET data against local particles.
// =============================================================================

#include "let_types.h"
#include "core/acceleration/fmm/fmm_types.h"
#include "core/acceleration/body_fitted_fmm/persistent_tree_traversal.cuh"
#include "core/acceleration/body_fitted_fmm/unified_node.cuh"
#include "domain_manager.h"   // for DomainParamsGPU and MAX_MULTI_GPUS
#include <cstdio>

namespace cudadem {

// =============================================================================
// Constant Memory Reference (declared in domain_manager.cu)
// =============================================================================
extern __constant__ DomainParamsGPU dev_domain_params;

// =============================================================================
// Device Helpers
// =============================================================================

// The host treats every nonzero export status as a failed LET publication.
// Keep these values local: they diagnose which device-side safety boundary
// rejected a persistent-tree snapshot without becoming part of the wire ABI.
enum LETExportStatus : int {
  LET_EXPORT_STATUS_METADATA = 1,
  LET_EXPORT_STATUS_START_LEVEL = 2,
  LET_EXPORT_STATUS_START_ROOT = 3,
  LET_EXPORT_STATUS_NODE = 4,
  LET_EXPORT_STATUS_CHILD = 5,
  LET_EXPORT_STATUS_STACK_OVERFLOW = 6,
  LET_EXPORT_STATUS_DESCRIPTOR_OVERFLOW = 7,
};

__device__ __forceinline__ void publish_let_export_status(
    int* export_status, const int status) {
  atomicCAS(export_status, 0, status);
}

__device__ inline bool box_box_overlap(
    double node_cx, double node_cy, double node_cz, double node_hs,
    double peer_cx, double peer_cy, double peer_cz, double peer_hs
) {
    const double dx = fabs(node_cx - peer_cx);
    const double dy = fabs(node_cy - peer_cy);
    const double dz = fabs(node_cz - peer_cz);
    const double sep = node_hs + peer_hs;
    return (dx < sep && dy < sep && dz < sep);
}

// Discovery deliberately emits no wire payload.  It walks each source once,
// records the exact peer mask accepted at that source, and counts only the
// small 256-bin mask histogram plus sparse leaf particle traffic.
__global__ void discover_let_fmm_kernel(
    const ::body_fitted_fmm::UnifiedNode* nodes, const int* children,
    const int* sorted, const int* level_start, const int* level_count,
    const int* level_list,
    const ::body_fitted_fmm::PersistentTreeMetadata* metadata,
    int node_capacity, int start_level, float theta,
    int num_gpus, int my_gpu, LETExportDescriptor* desc, int* desc_count,
    int desc_capacity, int* mask_counts, int* particle_counts,
    LETSourceWork* source_work, int* export_status) {
  // Start-level roots own disjoint subtrees, so one CUDA thread per root is
  // race-free apart from the compact global descriptor/counter appends.  The
  // FMM_MAX_LEVELS is 30 and an octree DFS has at most 1+7*30 pending
  // siblings.  Malformed topology is nevertheless checked before every push
  // because this exporter consumes a fixed-capacity persistent tree.
  constexpr int kStackCapacity = 256;
  const int si = blockIdx.x * blockDim.x + threadIdx.x;
  int stack[kStackCapacity]; unsigned int pending[kStackCapacity]; int top = 0;
  if (!metadata || !export_status) return;
  if (metadata->error != 0 || metadata->num_nodes <= 0 ||
      metadata->num_nodes > node_capacity || node_capacity <= 0 ||
      (metadata->active_tree != 2 && metadata->active_tree != 3) ||
      !isfinite(metadata->root_cell_size) || metadata->root_cell_size <= 0.0) {
    publish_let_export_status(export_status, LET_EXPORT_STATUS_METADATA);
    return;
  }
  if (start_level < 0 || start_level >= fmm::FMM_MAX_LEVELS) {
    publish_let_export_status(export_status, LET_EXPORT_STATUS_START_LEVEL);
    return;
  }
  if (num_gpus < 1 || num_gpus > MAX_MULTI_GPUS || my_gpu < 0 ||
      my_gpu >= num_gpus || desc_capacity < 0) {
    publish_let_export_status(export_status, LET_EXPORT_STATUS_METADATA);
    return;
  }
  const int actual_nodes = metadata->num_nodes;
  const int level_begin = level_start[start_level];
  const int roots = level_count[start_level];
  if (level_begin < 0 || roots < 0 ||
      static_cast<long long>(level_begin) + roots > actual_nodes) {
    publish_let_export_status(export_status, LET_EXPORT_STATUS_START_LEVEL);
    return;
  }
  const int children_per_node = metadata->active_tree == 2 ? 4 : 8;
  const uint8_t valid_child_mask = static_cast<uint8_t>(
      (1u << children_per_node) - 1u);
  const unsigned long long child_capacity =
      static_cast<unsigned long long>(node_capacity) * 8ull;
  const double root_size = metadata->root_cell_size;
  const double theta_sq = static_cast<double>(theta) * theta;
  if (si < roots) {
    const int root = level_list[level_begin + si];
    if (root < 0 || root >= actual_nodes) {
      publish_let_export_status(export_status, LET_EXPORT_STATUS_START_ROOT);
      return;
    }
    const auto root_node = nodes[root];
    if (!(root_node.flags & ::body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
        root_node.level != start_level) {
      publish_let_export_status(export_status, LET_EXPORT_STATUS_START_ROOT);
      return;
    }
    stack[top] = root; pending[top++] = 0;
    while (top) {
      if (*export_status != 0) return;
      --top; const int ni = stack[top]; const unsigned int resolved = pending[top];
      if (ni < 0 || ni >= actual_nodes) {
        publish_let_export_status(export_status, LET_EXPORT_STATUS_NODE);
        return;
      }
      const auto n = nodes[ni];
      const bool leaf = n.childBase ==
          ::body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE;
      if (!(n.flags & ::body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
          ((n.flags & ::body_fitted_fmm::UNIFIED_NODE_FLAG_IS_LEAF) != 0) !=
              leaf ||
          (n.childMask & ~valid_child_mask) != 0 ||
          (leaf && n.childMask != 0) || (!leaf && n.childMask == 0)) {
        publish_let_export_status(export_status, LET_EXPORT_STATUS_NODE);
        return;
      }
      const double hs = fmax(ldexp(root_size, -(static_cast<int>(n.level) + 1)),
                             static_cast<double>(n.radius));
      unsigned int accepted = 0, descend = 0;
      for (int p = 0; p < num_gpus; ++p) {
        if (p == my_gpu || (resolved & (1u << p))) continue;
        const double x = fmax(dev_domain_params.gpu_box_min_x[p],
                              fmin(dev_domain_params.gpu_box_max_x[p], n.center.x));
        const double y = fmax(dev_domain_params.gpu_box_min_y[p],
                              fmin(dev_domain_params.gpu_box_max_y[p], n.center.y));
        const double z = fmax(dev_domain_params.gpu_box_min_z[p],
                              fmin(dev_domain_params.gpu_box_max_z[p], n.center.z));
        const double dx = n.center.x - x;
        const double dy = n.center.y - y;
        const double dz = n.center.z - z;
        if (leaf || (hs * hs / (dx * dx + dy * dy + dz * dz + 1e-300)
                     <= theta_sq)) accepted |= 1u << p;
        else descend |= 1u << p;
      }
      if (accepted) {
        int di = atomicAdd(desc_count, 1);
        if (di >= 0 && di < desc_capacity)
          desc[di] = {ni, static_cast<uint8_t>(accepted), static_cast<uint8_t>(leaf), 0};
        else {
          publish_let_export_status(export_status,
                                    LET_EXPORT_STATUS_DESCRIPTOR_OVERFLOW);
          return;
        }
        const int fanout = __popc(accepted);
        if (leaf) {
          for (int p=0;p<num_gpus;++p) if (accepted & (1u<<p)) atomicAdd(&particle_counts[p], n.count);
          atomicAdd(&source_work[ni].y, static_cast<unsigned int>(n.count * fanout));
        } else {
          atomicAdd(&mask_counts[accepted], 1);
          atomicAdd(&source_work[ni].x, static_cast<unsigned int>(fanout));
        }
      }
      if (descend && !leaf) for (int c = 0; c < children_per_node; ++c) {
        if (!(n.childMask & (1u << c))) continue;
        const int ordinal = __popc(n.childMask & ((1u << c) - 1u));
        const unsigned long long compact_index =
            static_cast<unsigned long long>(n.childBase) + ordinal;
        if (compact_index >= child_capacity) {
          publish_let_export_status(export_status, LET_EXPORT_STATUS_CHILD);
          return;
        }
        const int child = children[compact_index];
        if (child < 0 || child >= actual_nodes) {
          publish_let_export_status(export_status, LET_EXPORT_STATUS_CHILD);
          return;
        }
        const auto child_node = nodes[child];
        if (!(child_node.flags & ::body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
            child_node.level != static_cast<uint8_t>(n.level + 1)) {
          publish_let_export_status(export_status, LET_EXPORT_STATUS_CHILD);
          return;
        }
        if (top >= kStackCapacity) {
          publish_let_export_status(export_status,
                                    LET_EXPORT_STATUS_STACK_OVERFLOW);
          return;
        }
        stack[top] = child;
        pending[top++] = resolved | accepted;
      }
    }
  }
}

__global__ void materialize_let_fmm_kernel(
    const LETExportDescriptor* desc, int desc_count,
    const ::body_fitted_fmm::UnifiedNode* nodes, const fmm::MultipoleExpansion* mp,
    const double* px, const double* py, const double* pz, const float* mass,
    const int* sorted,
    const ::body_fitted_fmm::PersistentTreeMetadata* metadata,
    int node_capacity, FMMLETNode* out_nodes, LETParticle* out_particles,
    const int* mask_offsets, const uint8_t* selected, const int* residual_offsets,
    const int* particle_offsets, int* cursors, LETSourceWork* source_work, int num_gpus) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=desc_count) return;
  if (!metadata || metadata->error != 0 || metadata->num_nodes <= 0 ||
      metadata->num_nodes > node_capacity || node_capacity <= 0 ||
      !isfinite(metadata->root_cell_size) || metadata->root_cell_size <= 0.0)
    return;
  LETExportDescriptor d=desc[i];
  if (d.source_node < 0 || d.source_node >= metadata->num_nodes ||
      d.destination_mask == 0 || d.is_leaf > 1) return;
  auto n=nodes[d.source_node];
  const bool node_is_leaf = n.childBase ==
      ::body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE;
  if (!(n.flags & ::body_fitted_fmm::UNIFIED_NODE_FLAG_VALID) ||
      node_is_leaf != (d.is_leaf != 0)) return;
  const double root_size = metadata->root_cell_size;
  unsigned int mask=d.destination_mask;
  if (d.is_leaf) {
    for(int p=0;p<num_gpus;++p) if(mask&(1u<<p)) for(int j=0;j<n.count;++j) {
      int w=atomicAdd(&cursors[264+p],1); int orig=sorted[n.begin+j];
      LETParticle& out = out_particles[particle_offsets[p] + w];
      out.pos = make_double3(px[orig], py[orig], pz[orig]);
      out.mass = mass[orig];
    }
    return;
  }
  const float radius = static_cast<float>(fmax(
      ldexp(root_size, -(static_cast<int>(n.level) + 1)),
      static_cast<double>(n.radius)));
  if(selected[mask]) {
    source_work[d.source_node].x=1; int w=atomicAdd(&cursors[mask],1);
    FMMLETNode& o=out_nodes[mask_offsets[mask]+w];
    o.center=n.center; o.radius=radius;
    for(int c=0;c<28;++c)o.multipole[c]=(float)mp[d.source_node].coeffs[c];
  } else for(int p=0;p<num_gpus;++p) if(mask&(1u<<p)) {
    int w=atomicAdd(&cursors[256+p],1);
    FMMLETNode& o=out_nodes[residual_offsets[p]+w];
    o.center=n.center; o.radius=radius;
    for(int c=0;c<28;++c)o.multipole[c]=(float)mp[d.source_node].coeffs[c];
  }
}

__device__ inline bool opening_criterion_passes(
    double node_cx, double node_cy, double node_cz, double node_hs,
    double peer_cx, double peer_cy, double peer_cz,
    double theta_sq
) {
    const double dx = node_cx - peer_cx;
    const double dy = node_cy - peer_cy;
    const double dz = node_cz - peer_cz;
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    const double size_sq = node_hs * node_hs;
    return (size_sq / (dist_sq + 1e-300) <= theta_sq);
}

// =============================================================================
// Retired one-pass FMM LET exporter
// =============================================================================
// Kept temporarily as off-path reference while the two-kernel destination-mask
// pipeline settles. LETCommunicator does not declare or launch this kernel.
// =============================================================================

__global__ void compute_let_fmm_kernel(
    const ::body_fitted_fmm::UnifiedNode* __restrict__ unified_nodes,
    const int* __restrict__ children_compact,
    const fmm::MultipoleExpansion* __restrict__ multipoles,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ level_start,
    const int* __restrict__ level_count,
    const int* __restrict__ level_list,
    int start_level,
    int num_nodes,
    int num_particles,
    float domain_root_cell_size,
    FMMLETNode* __restrict__ d_fmm_send_nodes,
    LETParticle* __restrict__ d_fmm_send_particles,
    int* __restrict__ d_let_node_counts,
    int* __restrict__ d_let_particle_counts,
    const int* __restrict__ d_node_offsets,
    const int* __restrict__ d_particle_offsets,
    const int* __restrict__ d_node_capacities,
    const int* __restrict__ d_particle_capacities,
    LETSourceWork* __restrict__ d_source_work,
    float theta,
    int num_gpus,
    int my_gpu_id
) {
    __shared__ int node_stack[1024];
    __shared__ unsigned int peer_mask_stack[1024];  // per-peer "already resolved by ancestor" bitmask
    __shared__ int stack_top;

    const double theta_sq = static_cast<double>(theta) * theta;

    if (threadIdx.x == 0) stack_top = 0;
    __syncthreads();

    int num_start_nodes = level_count[start_level];

    for (int si = blockIdx.x; si < num_start_nodes; si += gridDim.x) {
        int root_idx = level_list[level_start[start_level] + si];

        if (threadIdx.x == 0) {
            node_stack[0] = root_idx;
            peer_mask_stack[0] = 0u;  // root: no ancestors handled any peer yet
            stack_top = 1;
        }
        __syncthreads();

        while (stack_top > 0) {
            __shared__ ::body_fitted_fmm::UnifiedNode s_node;
            __shared__ int s_current_node;
            __shared__ unsigned int s_parent_mask;
            __shared__ unsigned int s_resolved_by_me;
            __shared__ bool s_need_descend;

            if (threadIdx.x == 0) {
                int top = stack_top - 1;
                s_current_node = node_stack[top];
                s_parent_mask = peer_mask_stack[top];
                stack_top = top;
                s_node = unified_nodes[s_current_node];
                s_resolved_by_me = 0u;
                s_need_descend = false;
            }
            __syncthreads();

            double node_cx = s_node.center.x;
            double node_cy = s_node.center.y;
            double node_cz = s_node.center.z;
            double cell_hs = ldexp(
                static_cast<double>(domain_root_cell_size),
                -(static_cast<int>(s_node.level) + 1));
            double node_size = fmax(cell_hs, static_cast<double>(s_node.radius));
            bool is_leaf = (s_node.childBase ==
                            ::body_fitted_fmm::UNIFIED_NODE_LEAF_CHILDBASE);

            for (int peer = 0; peer < num_gpus; peer++) {
                if (peer == my_gpu_id) continue;

                // Skip peers already handled by an ancestor.
                if (s_parent_mask & (1u << peer)) continue;

                // Use CLOSEST POINT in the PEER's domain box for the opening
                // criterion (not the domain center!).
                //
                // RATIONALE:
                //   The sender decides once whether each node becomes a multipole
                //   or a particle export. The RECEIVER then re-checks per-particle
                //   (per-particle criterion in remote_gravity_kernel skips nodes
                //   too close to the local particle — no fallback to particles).
                //
                //   If we used distance to domain CENTER, a node near the
                //   slab boundary appears "far" from the center → exported as
                //   multipole. But local particles right at the boundary are
                //   MUCH closer than the center, so the receiver skips the
                //   node → its contribution is silently LOST.
                //
                //   Closest-point distance = shortest distance from node center
                //   to ANY point in the peer's box. If the node passes the
                //   criterion at the closest point, then for EVERY particle in
                //   the peer's box: dist(particle, node) >= dist(closest, node),
                //   so the per-particle criterion also passes and the multipole
                //   is evaluated. No losses.
                const double closest_x = fmax(dev_domain_params.gpu_box_min_x[peer],
                               fmin(dev_domain_params.gpu_box_max_x[peer], node_cx));
                const double closest_y = fmax(dev_domain_params.gpu_box_min_y[peer],
                               fmin(dev_domain_params.gpu_box_max_y[peer], node_cy));
                const double closest_z = fmax(dev_domain_params.gpu_box_min_z[peer],
                               fmin(dev_domain_params.gpu_box_max_z[peer], node_cz));

                const double dx = node_cx - closest_x;
                const double dy = node_cy - closest_y;
                const double dz = node_cz - closest_z;
                const double dist_sq = dx*dx + dy*dy + dz*dz;
                const double size_sq = node_size * node_size;
                bool far_enough = (size_sq / (dist_sq + 1e-300)) <= theta_sq;

                if (is_leaf) {
                    // CRITICAL: cooperative particle export.
                    // Each thread handles a DIFFERENT subset of the leaf's particles
                    // (stride by blockDim.x). Otherwise all 128 threads would
                    // each export all ~16 particles → 128× double-counting.
                    int peer_particle_offset = d_particle_offsets[peer];
                    for (int p = s_node.begin + threadIdx.x;
                         p < s_node.begin + s_node.count;
                         p += blockDim.x) {
                        int orig = sorted_indices[p];
                        int pidx = atomicAdd(&d_let_particle_counts[peer], 1);
                        atomicAdd(&d_source_work[s_current_node].y, 1u);
                        if (pidx < d_particle_capacities[peer]) {
                            LETParticle& out =
                                d_fmm_send_particles[peer_particle_offset + pidx];
                            out.pos = make_double3(pos_x[orig], pos_y[orig],
                                                   pos_z[orig]);
                            out.mass = mass[orig];
                        }
                    }
                    if (threadIdx.x == 0) s_resolved_by_me |= (1u << peer);
                } else if (far_enough) {
                    // Multipole export — only ONE thread pushes the full node.
                    // Otherwise 128 threads each push the same multipole → 128×.
                    if (threadIdx.x == 0) {
                        int peer_offset = d_node_offsets[peer];
                        int nidx = atomicAdd(&d_let_node_counts[peer], 1);
                        atomicAdd(&d_source_work[s_current_node].x, 1u);
                        if (nidx < d_node_capacities[peer]) {
                            FMMLETNode& out = d_fmm_send_nodes[peer_offset + nidx];
                            out.center = make_double3(node_cx, node_cy, node_cz);
                            out.radius = static_cast<float>(node_size);
                            const auto& mp = multipoles[s_current_node];
                            for (int c = 0; c < 28; c++) {
                                out.multipole[c] = static_cast<float>(mp.coeffs[c]);
                            }
                        }
                        s_resolved_by_me |= (1u << peer);
                    }
                } else {
                    // Internal node, failed criterion → must descend for THIS peer.
                    // Single-thread set — but ALL threads need to see the flag
                    // (used below to decide whether to push children).
                    if (threadIdx.x == 0) s_need_descend = true;
                }
            }

            // Push children with the combined resolved mask (parent + this node).
            // Children will skip any peer with bit=1 in their parent_mask.
            if (s_need_descend && !is_leaf) {
                if (threadIdx.x == 0) {
                    unsigned int child_mask = s_parent_mask | s_resolved_by_me;
                    // Unified FMM supports the X-Y quadtree and octree.
                    for (int c = 0; c < 8; c++) {
                        int child = ::body_fitted_fmm::get_child_idx(
                            unified_nodes, children_compact, s_current_node, c);
                        if (child < 0) continue;
                        if (stack_top < 1024) {
                            node_stack[stack_top] = child;
                            peer_mask_stack[stack_top] = child_mask;
                            stack_top++;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// =============================================================================
// Device Multipole Field Evaluation (p=4 trace-free Cartesian)
// =============================================================================
// Mechanical port of host_eval_multipole_field from host_fmm.cpp:633-725.
// Evaluates gravitational force (not potential) of a p=4 trace-free
// multipole expansion at field point (dx,dy,dz) relative to the expansion center.
// Returns acceleration (partial derivatives of potential) — caller must scale
// by G * particle_mass to get force.
// =============================================================================

__device__ void multipole_eval_device(const double coeffs[28],
                                       double dx, double dy, double dz,
                                       double safe_dist_sq,
                                       double& fx, double& fy, double& fz) {
    double r = sqrt(safe_dist_sq);
    double inv_r = 1.0 / r;
    double inv_r2 = inv_r * inv_r;
    double inv_r3 = inv_r2 * inv_r;
    double inv_r5 = inv_r2 * inv_r3;
    double inv_r7 = inv_r2 * inv_r5;

    fx = 0.0; fy = 0.0; fz = 0.0;

    // ===== Order 0: Monopole =====
    fx += -coeffs[0] * dx * inv_r3;
    fy += -coeffs[0] * dy * inv_r3;
    fz += -coeffs[0] * dz * inv_r3;

    // ===== Order 1: Dipole =====
    double D_dot_r = coeffs[1] * dx + coeffs[2] * dy + coeffs[3] * dz;
    fx += coeffs[1] * inv_r3 - 3.0 * D_dot_r * dx * inv_r5;
    fy += coeffs[2] * inv_r3 - 3.0 * D_dot_r * dy * inv_r5;
    fz += coeffs[3] * inv_r3 - 3.0 * D_dot_r * dz * inv_r5;

    // ===== Order 2: Quadrupole =====
    // Reconstruct the full trace-free tensor Q_TF_ij from stored coefficients.
    // coeffs[4] = Q_TF_xx - Q_TF_yy, coeffs[5] = Q_TF_yy - Q_TF_zz,
    // coeffs[6..8] = Q_TF_xy, Q_TF_xz, Q_TF_yz (direct, no factor of 3).
    // The P2M formula stores coeffs[4]=3m(dx²-dy²) which equals
    // Q_TF_xx-Q_TF_yy exactly (since Q_TF_ij = m*(3r_i*r_j - r²δ_ij)).
    // Solving the trace-free constraint gives /3 (not /9 — /9 was a latent bug
    // shared with host_eval_multipole_field, undetected there because the
    // multipole term is small relative to monopole at typical test distances).
    double Q_xx = (2.0 * coeffs[4] + coeffs[5]) / 3.0;
    double Q_yy = (-coeffs[4] + coeffs[5]) / 3.0;
    double Q_zz = -(Q_xx + Q_yy);
    double Q_xy = coeffs[6];
    double Q_xz = coeffs[7];
    double Q_yz = coeffs[8];

    double Qr = Q_xx * dx * dx + Q_yy * dy * dy + Q_zz * dz * dz
              + 2.0 * Q_xy * dx * dy + 2.0 * Q_xz * dx * dz + 2.0 * Q_yz * dy * dz;

    double Q_sum_x = Q_xx * dx + Q_xy * dy + Q_xz * dz;
    double Q_sum_y = Q_xy * dx + Q_yy * dy + Q_yz * dz;
    double Q_sum_z = Q_xz * dx + Q_yz * dy + Q_zz * dz;

    fx += Q_sum_x * inv_r5 - 2.5 * Qr * dx * inv_r7;
    fy += Q_sum_y * inv_r5 - 2.5 * Qr * dy * inv_r7;
    fz += Q_sum_z * inv_r5 - 2.5 * Qr * dz * inv_r7;

    // Factor the same p=4 force operator; retain FP64 evaluation and the
    // original softened displacement. Do not assume a unit direction here.
    const double scale5 = inv_r * inv_r2 * inv_r2;
    const double scale6 = inv_r2 * inv_r2 * inv_r2;
    double L1, L2, L3;
    evaluate_let_high_orders(coeffs, dx * inv_r, dy * inv_r, dz * inv_r,
                             scale5, scale6, L1, L2, L3);

    fx += L1;
    fy += L2;
    fz += L3;
}

// =============================================================================
// Remote Gravity Kernel
// =============================================================================
// Evaluates received FMM LET data (merged nodes + particles from remote GPUs)
// against local particles. Runs AFTER the local FMM pipeline completes.
// Far-field: evaluates FMMLETNode multipoles via multipole_eval_device().
// Near-field: direct P2P from LET particles.
// =============================================================================

__global__ void remote_gravity_kernel(
    const FMMLETNode* __restrict__ recv_nodes,
    const int* __restrict__ recv_node_offsets,
    const int* __restrict__ recv_node_counts,
    const LETParticle* __restrict__ recv_particles,
    const int* __restrict__ recv_particle_offsets,
    const int* __restrict__ recv_particle_counts,
    int num_gpus, int my_gpu_id,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double G,
    double softening,
    float theta,
    int num_local_particles,
    bool acceleration_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local_particles) return;

    double px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    // The regular LET path accumulates force.  RESPA retains gravity as an
    // acceleration cache, so its remote pass deliberately omits the target
    // mass.  (Source masses remain in the LET particle records.)
    double mi = acceleration_output ? 1.0 : static_cast<double>(mass[i]);
    (void)theta;  // sender closest-point MAC already guarantees separation
    double eps2 = softening * softening;

    double fx = 0.0, fy = 0.0, fz = 0.0;

    // Far-field: evaluate FMM multipoles, guarded by per-particle opening criterion.
    //
    // WHY the per-particle check is necessary:
    //   The SENDER validated that each node is "far enough" from the peer
    //   DOMAIN CENTER (a single reference point). But a LOCAL PARTICLE near
    //   the slab boundary can be much closer than the domain center — even
    //   INSIDE the node's bounding box. In that case the multipole expansion
    //   diverges (1/r³ with tiny r) and produces forces many orders of
    //   magnitude too large.
    //
    // WHY skipping is safe:
    //   Nodes that fail the check for a particular particle are the ones
    //   near that particle — i.e., near-field interactions. Their contribution
    //   is a tiny fraction of the total (a few particles at close range out
    //   of thousands), well within the 1% tolerance.
    for (int peer = 0; peer < num_gpus; ++peer) {
      if (peer == my_gpu_id) continue;
      const int node_offset = recv_node_offsets[peer];
      const int node_count = recv_node_counts[peer];
      for (int ni = 0; ni < node_count; ++ni) {
        const FMMLETNode& node = recv_nodes[node_offset + ni];
        // Both operands are world-space FP64.  Form the local displacement
        // before evaluating the (intentionally FP32) wire multipoles.
        double dx = px - node.center.x;
        double dy = py - node.center.y;
        double dz = pz - node.center.z;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        // CRITICAL FIX: Do NOT use per-particle distance clamping.
        // The sender uses CLOSEST-POINT criterion which guarantees exported
        // nodes are well-separated from ALL particles in the peer domain.
        // Clamping (dist → max(dist, hs/theta)) causes the M2L matrix to
        // receive non-unit direction vectors, corrupting p≥2 multipole
        // evaluation. Only softening prevents exact-zero divergence.
        double safe_dist_sq = fmax(dist_sq, eps2);

        double nfx, nfy, nfz;
        // Wire coefficients are FP32; evaluation remains FP64.
        double coeffs[28];
        #pragma unroll
        for (int ci = 0; ci < 28; ++ci) coeffs[ci] = static_cast<double>(node.multipole[ci]);
        multipole_eval_device(coeffs, dx, dy, dz, safe_dist_sq, nfx, nfy, nfz);
        fx += G * mi * nfx;
        fy += G * mi * nfy;
        fz += G * mi * nfz;
      }
    }

    // Near-field: direct P2P from LET particles
    for (int peer = 0; peer < num_gpus; ++peer) {
      if (peer == my_gpu_id) continue;
      const int particle_offset = recv_particle_offsets[peer];
      const int particle_count = recv_particle_counts[peer];
      for (int p = 0; p < particle_count; p++) {
        const LETParticle& pp = recv_particles[particle_offset + p];
        // Keep peer positions in FP64 until the P2P displacement has been
        // formed; this is the multi-GPU analogue of grouped local P2P.
        double ddx = pp.pos.x - px;
        double ddy = pp.pos.y - py;
        double ddz = pp.pos.z - pz;
        double r2  = ddx*ddx + ddy*ddy + ddz*ddz + eps2;
        double r   = sqrt(r2);
        double r3  = r2 * r;
        double factor = G * mi * static_cast<double>(pp.mass) / r3;
        fx += factor * ddx;
        fy += factor * ddy;
        fz += factor * ddz;
      }
    }

    force_x[i] += fx;
    force_y[i] += fy;
    force_z[i] += fz;
}

// =============================================================================
// Launch Remote Gravity (host-side wrapper)
// =============================================================================

// launch_remote_gravity lives at cudadem scope (was formerly inside
// namespace lbvh_fmm; moved out when lbvh_fmm was retired to lib/ as
// reference-only, so the body-fitted FMM multi-GPU path can call it).

void launch_remote_gravity(
    const FMMLETNode* d_recv_nodes,
    const int* d_recv_node_offsets,
    const int* d_recv_node_counts,
    const LETParticle* d_recv_particles,
    const int* d_recv_particle_offsets,
    const int* d_recv_particle_counts,
    int num_gpus, int my_gpu_id,
    const double* pos_x, const double* pos_y, const double* pos_z,
    const float* mass,
    double* force_x, double* force_y, double* force_z,
    double G, double softening, float theta,
    int num_local_particles,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_local_particles + block - 1) / block;
    remote_gravity_kernel<<<grid, block, 0, stream>>>(
        d_recv_nodes, d_recv_node_offsets, d_recv_node_counts,
        d_recv_particles, d_recv_particle_offsets, d_recv_particle_counts,
        num_gpus, my_gpu_id,
        pos_x, pos_y, pos_z, mass,
        force_x, force_y, force_z,
        G, softening, theta,
        num_local_particles, false
    );
}

// RESPA counterpart of launch_remote_gravity.  Keep this as a distinct host
// entry point so the force-unit K=1 path cannot accidentally select the
// acceleration-unit output arrays.
void launch_remote_gravity_acceleration(
    const FMMLETNode* d_recv_nodes,
    const int* d_recv_node_offsets,
    const int* d_recv_node_counts,
    const LETParticle* d_recv_particles,
    const int* d_recv_particle_offsets,
    const int* d_recv_particle_counts,
    int num_gpus, int my_gpu_id,
    const double* pos_x, const double* pos_y, const double* pos_z,
    double* acceleration_x, double* acceleration_y, double* acceleration_z,
    double G, double softening, float theta,
    int num_local_particles,
    cudaStream_t stream
) {
    if (num_local_particles <= 0) return;
    constexpr int block = 256;
    const int grid = (num_local_particles + block - 1) / block;
    remote_gravity_kernel<<<grid, block, 0, stream>>>(
        d_recv_nodes, d_recv_node_offsets, d_recv_node_counts,
        d_recv_particles, d_recv_particle_offsets, d_recv_particle_counts,
        num_gpus, my_gpu_id, pos_x, pos_y, pos_z,
        nullptr, acceleration_x, acceleration_y, acceleration_z,
        G, softening, theta, num_local_particles, true);
}

} // namespace cudadem

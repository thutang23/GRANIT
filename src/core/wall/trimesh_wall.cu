// =============================================================================
// Triangle Mesh Wall CUDA Implementation
// =============================================================================
// Implements GPU kernels and dispatcher for triangle mesh wall contacts.
//
// Key Components:
// - AABB computation kernel for BVH construction
// - BVH construction (simplified radix-based)
// - Contact force kernel with BVH traversal
// - TriMeshWallDispatcher for orchestration
//
// Reference: lib/forceTriMeshWalls.f90
// =============================================================================

#include "trimesh_wall.h"
#include "rigid_pose_trajectory.h"
#include "wall_contact_force.cuh"
#include "../constant/system_initialize.h"
#include "../particle_model/particle.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace cudadem {

// Forward declarations
void free_trimesh_host(TriMeshSoA& mesh);

// =============================================================================
// Device Memory Allocation
// =============================================================================

void allocate_trimesh_device(TriMeshSoA& mesh, int32_t num_triangles) {
    mesh.num_triangles = num_triangles;

    size_t size_f = num_triangles * sizeof(float);
    size_t size_i = num_triangles * sizeof(int32_t);

    cudaMalloc(&mesh.v0_x, size_f);
    cudaMalloc(&mesh.v0_y, size_f);
    cudaMalloc(&mesh.v0_z, size_f);

    cudaMalloc(&mesh.edge1_x, size_f);
    cudaMalloc(&mesh.edge1_y, size_f);
    cudaMalloc(&mesh.edge1_z, size_f);
    cudaMalloc(&mesh.edge2_x, size_f);
    cudaMalloc(&mesh.edge2_y, size_f);
    cudaMalloc(&mesh.edge2_z, size_f);

    cudaMalloc(&mesh.normal_x, size_f);
    cudaMalloc(&mesh.normal_y, size_f);
    cudaMalloc(&mesh.normal_z, size_f);

    cudaMalloc(&mesh.len_edge1_sq, size_f);
    cudaMalloc(&mesh.len_edge2_sq, size_f);
    cudaMalloc(&mesh.dot_e1_e2, size_f);
    cudaMalloc(&mesh.inv_denom, size_f);

    cudaMalloc(&mesh.wall_tag, size_i);

    cudaMalloc(&mesh.aabb_min_x, size_f);
    cudaMalloc(&mesh.aabb_min_y, size_f);
    cudaMalloc(&mesh.aabb_min_z, size_f);
    cudaMalloc(&mesh.aabb_max_x, size_f);
    cudaMalloc(&mesh.aabb_max_y, size_f);
    cudaMalloc(&mesh.aabb_max_z, size_f);
}

void copy_trimesh_to_device(TriMeshSoA& d_mesh, const TriMeshSoA& h_mesh) {
    int32_t n = h_mesh.num_triangles;
    size_t size_f = n * sizeof(float);
    size_t size_i = n * sizeof(int32_t);

    d_mesh.num_triangles = n;
    d_mesh.mesh_id = h_mesh.mesh_id;
    d_mesh.material_id = h_mesh.material_id;
    d_mesh.tag_offset = h_mesh.tag_offset;
    d_mesh.enabled = h_mesh.enabled;

    cudaMemcpy(d_mesh.v0_x, h_mesh.v0_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.v0_y, h_mesh.v0_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.v0_z, h_mesh.v0_z, size_f, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.edge1_x, h_mesh.edge1_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge1_y, h_mesh.edge1_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge1_z, h_mesh.edge1_z, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge2_x, h_mesh.edge2_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge2_y, h_mesh.edge2_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge2_z, h_mesh.edge2_z, size_f, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.normal_x, h_mesh.normal_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.normal_y, h_mesh.normal_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.normal_z, h_mesh.normal_z, size_f, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.len_edge1_sq, h_mesh.len_edge1_sq, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.len_edge2_sq, h_mesh.len_edge2_sq, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.dot_e1_e2, h_mesh.dot_e1_e2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.inv_denom, h_mesh.inv_denom, size_f, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.wall_tag, h_mesh.wall_tag, size_i, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.aabb_min_x, h_mesh.aabb_min_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.aabb_min_y, h_mesh.aabb_min_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.aabb_min_z, h_mesh.aabb_min_z, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.aabb_max_x, h_mesh.aabb_max_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.aabb_max_y, h_mesh.aabb_max_y, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.aabb_max_z, h_mesh.aabb_max_z, size_f, cudaMemcpyHostToDevice);
}

void free_trimesh_device(TriMeshSoA& mesh) {
    cudaFree(mesh.v0_x);
    cudaFree(mesh.v0_y);
    cudaFree(mesh.v0_z);

    cudaFree(mesh.edge1_x);
    cudaFree(mesh.edge1_y);
    cudaFree(mesh.edge1_z);
    cudaFree(mesh.edge2_x);
    cudaFree(mesh.edge2_y);
    cudaFree(mesh.edge2_z);

    cudaFree(mesh.normal_x);
    cudaFree(mesh.normal_y);
    cudaFree(mesh.normal_z);

    cudaFree(mesh.len_edge1_sq);
    cudaFree(mesh.len_edge2_sq);
    cudaFree(mesh.dot_e1_e2);
    cudaFree(mesh.inv_denom);

    cudaFree(mesh.wall_tag);

    cudaFree(mesh.aabb_min_x);
    cudaFree(mesh.aabb_min_y);
    cudaFree(mesh.aabb_min_z);
    cudaFree(mesh.aabb_max_x);
    cudaFree(mesh.aabb_max_y);
    cudaFree(mesh.aabb_max_z);

    mesh = TriMeshSoA();
}

void allocate_bvh(TriMeshBVH& bvh, int32_t num_triangles, int32_t num_particles) {
    bvh.num_leaves = num_triangles;

    // Compute actual node count for level-by-level build.
    // For non-power-of-2 leaf counts, ceil-division at each level
    // produces more total nodes than the 2n-1 formula assumes.
    // e.g. n=12: 12 + 6 + 3 + 2 + 1 = 24, not 2*12-1 = 23.
    int32_t total_nodes = num_triangles;
    int32_t level_size = num_triangles;
    while (level_size > 1) {
        level_size = (level_size + 1) / 2;
        total_nodes += level_size;
    }
    bvh.num_nodes = total_nodes;
    bvh.root_index = 0;  // Will be set during build

    cudaMalloc(&bvh.d_nodes, bvh.num_nodes * sizeof(TriMeshBVHNode));
    cudaMalloc(&bvh.d_sorted_indices, num_triangles * sizeof(int32_t));

    // Query scratch space
    cudaMalloc(&bvh.d_candidate_counts, num_particles * sizeof(int32_t));
    cudaMalloc(&bvh.d_candidates, num_particles * MAX_CANDIDATES_PER_PARTICLE * sizeof(int32_t));

    size_t size_f = num_triangles * sizeof(float);
    cudaMalloc(&bvh.d_inflated_min_x, size_f);
    cudaMalloc(&bvh.d_inflated_min_y, size_f);
    cudaMalloc(&bvh.d_inflated_min_z, size_f);
    cudaMalloc(&bvh.d_inflated_max_x, size_f);
    cudaMalloc(&bvh.d_inflated_max_y, size_f);
    cudaMalloc(&bvh.d_inflated_max_z, size_f);

    bvh.valid = false;
}

void free_bvh(TriMeshBVH& bvh) {
    cudaFree(bvh.d_nodes);
    cudaFree(bvh.d_sorted_indices);
    cudaFree(bvh.d_candidate_counts);
    cudaFree(bvh.d_candidates);
    cudaFree(bvh.d_inflated_min_x);
    cudaFree(bvh.d_inflated_min_y);
    cudaFree(bvh.d_inflated_min_z);
    cudaFree(bvh.d_inflated_max_x);
    cudaFree(bvh.d_inflated_max_y);
    cudaFree(bvh.d_inflated_max_z);
    bvh = TriMeshBVH();
}

// =============================================================================
// AABB Inflation Kernel
// =============================================================================

__global__ void copy_and_inflate_aabbs_kernel(
    const float* __restrict__ src_min_x,
    const float* __restrict__ src_min_y,
    const float* __restrict__ src_min_z,
    const float* __restrict__ src_max_x,
    const float* __restrict__ src_max_y,
    const float* __restrict__ src_max_z,
    float* __restrict__ dst_min_x,
    float* __restrict__ dst_min_y,
    float* __restrict__ dst_min_z,
    float* __restrict__ dst_max_x,
    float* __restrict__ dst_max_y,
    float* __restrict__ dst_max_z,
    float inflation,
    int32_t num_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles) return;

    dst_min_x[idx] = src_min_x[idx] - inflation;
    dst_min_y[idx] = src_min_y[idx] - inflation;
    dst_min_z[idx] = src_min_z[idx] - inflation;
    dst_max_x[idx] = src_max_x[idx] + inflation;
    dst_max_y[idx] = src_max_y[idx] + inflation;
    dst_max_z[idx] = src_max_z[idx] + inflation;
}

// =============================================================================
// Morton Code Computation for BVH Construction
// =============================================================================

__device__ __forceinline__
uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__
uint32_t morton_code(float x, float y, float z,
                     float min_x, float min_y, float min_z,
                     float inv_range_x, float inv_range_y, float inv_range_z) {
    // Normalize to [0, 1023]
    uint32_t ix = min(max((uint32_t)((x - min_x) * inv_range_x * 1023.0f), 0u), 1023u);
    uint32_t iy = min(max((uint32_t)((y - min_y) * inv_range_y * 1023.0f), 0u), 1023u);
    uint32_t iz = min(max((uint32_t)((z - min_z) * inv_range_z * 1023.0f), 0u), 1023u);

    return (expand_bits(ix) << 2) | (expand_bits(iy) << 1) | expand_bits(iz);
}

__global__ void compute_morton_codes_kernel(
    const float* __restrict__ aabb_min_x,
    const float* __restrict__ aabb_min_y,
    const float* __restrict__ aabb_min_z,
    const float* __restrict__ aabb_max_x,
    const float* __restrict__ aabb_max_y,
    const float* __restrict__ aabb_max_z,
    uint32_t* __restrict__ morton_codes,
    int32_t* __restrict__ indices,
    float scene_min_x, float scene_min_y, float scene_min_z,
    float inv_range_x, float inv_range_y, float inv_range_z,
    int32_t num_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles) return;

    // Use AABB centroid for Morton code
    float cx = 0.5f * (aabb_min_x[idx] + aabb_max_x[idx]);
    float cy = 0.5f * (aabb_min_y[idx] + aabb_max_y[idx]);
    float cz = 0.5f * (aabb_min_z[idx] + aabb_max_z[idx]);

    morton_codes[idx] = morton_code(cx, cy, cz,
                                    scene_min_x, scene_min_y, scene_min_z,
                                    inv_range_x, inv_range_y, inv_range_z);
    indices[idx] = idx;
}

// =============================================================================
// BVH Construction Kernel (Bottom-up from sorted leaves)
// =============================================================================

__global__ void build_bvh_leaves_kernel(
    const float* __restrict__ aabb_min_x,
    const float* __restrict__ aabb_min_y,
    const float* __restrict__ aabb_min_z,
    const float* __restrict__ aabb_max_x,
    const float* __restrict__ aabb_max_y,
    const float* __restrict__ aabb_max_z,
    const int32_t* __restrict__ sorted_indices,
    TriMeshBVHNode* __restrict__ nodes,
    int32_t num_leaves,
    int32_t leaf_offset  // First leaf node index = num_leaves - 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    int32_t tri_idx = sorted_indices[idx];
    int32_t node_idx = leaf_offset + idx;

    nodes[node_idx].min_x = aabb_min_x[tri_idx];
    nodes[node_idx].min_y = aabb_min_y[tri_idx];
    nodes[node_idx].min_z = aabb_min_z[tri_idx];
    nodes[node_idx].max_x = aabb_max_x[tri_idx];
    nodes[node_idx].max_y = aabb_max_y[tri_idx];
    nodes[node_idx].max_z = aabb_max_z[tri_idx];
    nodes[node_idx].left_child = -1;
    nodes[node_idx].right_child = -1;
    nodes[node_idx].triangle_idx = tri_idx;
}

// Build internal nodes (must be called iteratively from bottom up)
__global__ void build_bvh_internal_kernel(
    TriMeshBVHNode* __restrict__ nodes,
    int32_t level_start,
    int32_t level_size,
    int32_t child_offset,
    int32_t child_level_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_size) return;

    int32_t node_idx = level_start + idx;
    int32_t left_idx = child_offset + 2 * idx;
    int32_t right_idx = left_idx + 1;

    // When child level has odd size, the last parent has only a left child
    bool has_right = (2 * idx + 1) < child_level_size;

    nodes[node_idx].left_child = left_idx;
    nodes[node_idx].right_child = has_right ? right_idx : -1;
    nodes[node_idx].triangle_idx = -1;

    if (has_right) {
        // Merge AABBs from both children
        nodes[node_idx].min_x = fminf(nodes[left_idx].min_x, nodes[right_idx].min_x);
        nodes[node_idx].min_y = fminf(nodes[left_idx].min_y, nodes[right_idx].min_y);
        nodes[node_idx].min_z = fminf(nodes[left_idx].min_z, nodes[right_idx].min_z);
        nodes[node_idx].max_x = fmaxf(nodes[left_idx].max_x, nodes[right_idx].max_x);
        nodes[node_idx].max_y = fmaxf(nodes[left_idx].max_y, nodes[right_idx].max_y);
        nodes[node_idx].max_z = fmaxf(nodes[left_idx].max_z, nodes[right_idx].max_z);
    } else {
        // Single child: copy its AABB directly
        nodes[node_idx].min_x = nodes[left_idx].min_x;
        nodes[node_idx].min_y = nodes[left_idx].min_y;
        nodes[node_idx].min_z = nodes[left_idx].min_z;
        nodes[node_idx].max_x = nodes[left_idx].max_x;
        nodes[node_idx].max_y = nodes[left_idx].max_y;
        nodes[node_idx].max_z = nodes[left_idx].max_z;
    }
}

// =============================================================================
// TriMesh Contact Force Kernel
// =============================================================================

template<bool EnableFriction>
__global__ void compute_trimesh_wall_forces_kernel(
    // Particle data (SoA)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const float* __restrict__ radius,
    const float* __restrict__ mass,
    const float* __restrict__ moment_of_inertia,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const DemInteractionProperties* __restrict__ interactions,
    uint32_t material_count,

    // Force output
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,

    // Mesh data
    const float* __restrict__ mesh_v0_x,
    const float* __restrict__ mesh_v0_y,
    const float* __restrict__ mesh_v0_z,
    const float* __restrict__ mesh_edge1_x,
    const float* __restrict__ mesh_edge1_y,
    const float* __restrict__ mesh_edge1_z,
    const float* __restrict__ mesh_edge2_x,
    const float* __restrict__ mesh_edge2_y,
    const float* __restrict__ mesh_edge2_z,
    const float* __restrict__ mesh_normal_x,
    const float* __restrict__ mesh_normal_y,
    const float* __restrict__ mesh_normal_z,
    const float* __restrict__ mesh_len1_sq,
    const float* __restrict__ mesh_len2_sq,
    const float* __restrict__ mesh_dot_e1_e2,
    const float* __restrict__ mesh_inv_denom,
    const int32_t* __restrict__ mesh_wall_tag,
    int32_t num_triangles,

    // BVH
    const TriMeshBVHNode* __restrict__ bvh_nodes,
    int32_t num_bvh_nodes,
    int32_t bvh_root_index,

    // Mesh pose and velocity.  The BVH/triangles remain in body coordinates.
    MeshMotionState motion,

    // Contact history
    WallContactHistory* __restrict__ wall_history,
    const int32_t* __restrict__ wall_history_offset,
    int32_t* __restrict__ num_wall_contacts,
    int32_t max_wall_contacts_per_particle,

    int32_t wall_material_id,
    bool use_visco,
    float dt,
    float tolerance,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Load particle data
    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);
    float R_i = radius[idx];
    float m_i = mass[idx];
    float I_i = moment_of_inertia[idx];
    const int32_t particle_material_id = material_id[idx];

    float E_star, G_star;
    DemInteractionProperties pair{};
    if (!resolve_wall_material_pair(particle_material_id, wall_material_id,
                                    materials, interactions, material_count,
                                    E_star, G_star, pair)) return;
    const float pair_cutoff = pair.cohesion_enabled
        ? fmaxf(1.0f, pair.cohesion_cutoff_factor) : 1.0f;
    const float query_radius = R_i * pair_cutoff;

    double vx_i = vel_x[idx];
    double vy_i = vel_y[idx];
    double vz_i = vel_z[idx];
    double wx_i = omega_x[idx];
    double wy_i = omega_y[idx];
    double wz_i = omega_z[idx];

    // Transform the world particle center into the invariant mesh body frame:
    // b = reference_origin + R^T (x - translation).
    const double dxw = static_cast<double>(px) - motion.translation_x;
    const double dyw = static_cast<double>(py) - motion.translation_y;
    const double dzw = static_cast<double>(pz) - motion.translation_z;
    const float bpx = static_cast<float>(motion.reference_origin_x + motion.rotation[0]*dxw + motion.rotation[3]*dyw + motion.rotation[6]*dzw);
    const float bpy = static_cast<float>(motion.reference_origin_y + motion.rotation[1]*dxw + motion.rotation[4]*dyw + motion.rotation[7]*dzw);
    const float bpz = static_cast<float>(motion.reference_origin_z + motion.rotation[2]*dxw + motion.rotation[5]*dyw + motion.rotation[8]*dzw);

    // Get contact history for this particle
    int32_t hist_offset = wall_history_offset[idx];
    WallContactHistory* my_history = wall_history + hist_offset;
    int32_t my_num_contacts = num_wall_contacts[idx];

    // Query BVH for candidate triangles
    int32_t candidates[MAX_CANDIDATES_PER_PARTICLE];
    int32_t num_candidates = query_bvh_sphere(
        bvh_nodes, num_bvh_nodes, bvh_root_index,
        bpx, bpy, bpz, query_radius,
        candidates, MAX_CANDIDATES_PER_PARTICLE
    );

    // Accumulate forces
    double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
    double tx_acc = 0.0, ty_acc = 0.0, tz_acc = 0.0;

    int32_t new_contact_count = 0;

    // Process each candidate triangle
    for (int32_t c = 0; c < num_candidates; ++c) {
        int32_t tri_idx = candidates[c];
        if (tri_idx < 0 || tri_idx >= num_triangles) continue;

        // Narrow phase: compute triangle contact
        TriangleContactResult contact = compute_triangle_contact(
            bpx, bpy, bpz, query_radius,
            mesh_v0_x[tri_idx], mesh_v0_y[tri_idx], mesh_v0_z[tri_idx],
            mesh_edge1_x[tri_idx], mesh_edge1_y[tri_idx], mesh_edge1_z[tri_idx],
            mesh_edge2_x[tri_idx], mesh_edge2_y[tri_idx], mesh_edge2_z[tri_idx],
            mesh_normal_x[tri_idx], mesh_normal_y[tri_idx], mesh_normal_z[tri_idx],
            mesh_len1_sq[tri_idx], mesh_len2_sq[tri_idx],
            mesh_dot_e1_e2[tri_idx], mesh_inv_denom[tri_idx],
            tolerance
        );

        if (!contact.has_contact) continue;

        // Convert closest point and normal back to world before the contact
        // law.  Distance/overlap are invariant under the rigid transform.
        const float body_closest_x=contact.closest_x, body_closest_y=contact.closest_y, body_closest_z=contact.closest_z;
        contact.closest_x=static_cast<float>(motion.translation_x + motion.rotation[0]*(body_closest_x-motion.reference_origin_x) + motion.rotation[1]*(body_closest_y-motion.reference_origin_y) + motion.rotation[2]*(body_closest_z-motion.reference_origin_z));
        contact.closest_y=static_cast<float>(motion.translation_y + motion.rotation[3]*(body_closest_x-motion.reference_origin_x) + motion.rotation[4]*(body_closest_y-motion.reference_origin_y) + motion.rotation[5]*(body_closest_z-motion.reference_origin_z));
        contact.closest_z=static_cast<float>(motion.translation_z + motion.rotation[6]*(body_closest_x-motion.reference_origin_x) + motion.rotation[7]*(body_closest_y-motion.reference_origin_y) + motion.rotation[8]*(body_closest_z-motion.reference_origin_z));
        const float body_nx=contact.normal_x, body_ny=contact.normal_y, body_nz=contact.normal_z;
        contact.normal_x=static_cast<float>(motion.rotation[0]*body_nx+motion.rotation[1]*body_ny+motion.rotation[2]*body_nz);
        contact.normal_y=static_cast<float>(motion.rotation[3]*body_nx+motion.rotation[4]*body_ny+motion.rotation[5]*body_nz);
        contact.normal_z=static_cast<float>(motion.rotation[6]*body_nx+motion.rotation[7]*body_ny+motion.rotation[8]*body_nz);

        const float actual_overlap = R_i - contact.distance;

        int32_t wall_tag = mesh_wall_tag[tri_idx];

        // Find existing history slot
        int32_t slot = -1;
        float H_t_x = 0.0f, H_t_y = 0.0f, H_t_z = 0.0f;
        float H_r_x = 0.0f, H_r_y = 0.0f, H_r_z = 0.0f;
        float H_tw_x = 0.0f, H_tw_y = 0.0f, H_tw_z = 0.0f;

        for (int32_t k = 0; k < my_num_contacts; ++k) {
            if (my_history[k].wall_id == wall_tag && my_history[k].is_active) {
                slot = k;
                H_t_x = my_history[k].delta_t_x;
                H_t_y = my_history[k].delta_t_y;
                H_t_z = my_history[k].delta_t_z;
                H_r_x = my_history[k].delta_r_x;
                H_r_y = my_history[k].delta_r_y;
                H_r_z = my_history[k].delta_r_z;
                H_tw_x = my_history[k].delta_tw_x;
                H_tw_y = my_history[k].delta_tw_y;
                H_tw_z = my_history[k].delta_tw_z;
                break;
            }
        }

        // Compute mesh velocity at contact point
        double wall_vx = 0.0, wall_vy = 0.0, wall_vz = 0.0;
        if (motion.is_moving) {
            wall_vx = motion.velocity_x;
            wall_vy = motion.velocity_y;
            wall_vz = motion.velocity_z;
            // Add rotational contribution
            double rx = contact.closest_x - motion.center_x;
            double ry = contact.closest_y - motion.center_y;
            double rz = contact.closest_z - motion.center_z;
            wall_vx += motion.omega_y * rz - motion.omega_z * ry;
            wall_vy += motion.omega_z * rx - motion.omega_x * rz;
            wall_vz += motion.omega_x * ry - motion.omega_y * rx;
        }

        // Use pre-computed distance from compute_triangle_contact.  Candidate
        // search was expanded for the pair cohesion range; only the true
        // particle radius participates in the contact stiffness.
        // directly, bypassing plane_signed_distance which assumes infinite planes
        WallForceResult result;
        if constexpr (EnableFriction) {
            result = calculate_wall_force_impl<FullFrictionPolicy>(
                actual_overlap, contact.distance,
                vx_i, vy_i, vz_i, wx_i, wy_i, wz_i,
                R_i, m_i, I_i,
                contact.normal_x, contact.normal_y, contact.normal_z,
                wall_vx, wall_vy, wall_vz,
                H_t_x, H_t_y, H_t_z,
                H_r_x, H_r_y, H_r_z,
                H_tw_x, H_tw_y, H_tw_z,
                E_star, G_star, pair.restitution,
                pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                use_visco, pair.viscous_coefficient, dt
            );
        } else {
            result = calculate_wall_force_impl<SmoothWallPolicy>(
                actual_overlap, contact.distance,
                vx_i, vy_i, vz_i, wx_i, wy_i, wz_i,
                R_i, m_i, I_i,
                contact.normal_x, contact.normal_y, contact.normal_z,
                wall_vx, wall_vy, wall_vz,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                E_star, G_star, pair.restitution,
                pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
                use_visco, pair.viscous_coefficient, dt
            );
        }
        add_wall_cohesion(result, actual_overlap,
                          contact.normal_x, contact.normal_y, contact.normal_z, R_i, pair);

        // Apply edge flag weighting (from Fortran)
        float weight = contact.edge_flag;
        fx_acc += weight * result.force_x;
        fy_acc += weight * result.force_y;
        fz_acc += weight * result.force_z;
        tx_acc += weight * result.torque_x;
        ty_acc += weight * result.torque_y;
        tz_acc += weight * result.torque_z;

        // Update contact history
        if (result.is_contact) {
            if (slot < 0 && (my_num_contacts + new_contact_count) < max_wall_contacts_per_particle) {
                slot = my_num_contacts + new_contact_count;
                new_contact_count++;
            }

            if (slot >= 0 && slot < max_wall_contacts_per_particle) {
                my_history[slot].wall_id = wall_tag;
                my_history[slot].delta_t_x = result.H_t_x;
                my_history[slot].delta_t_y = result.H_t_y;
                my_history[slot].delta_t_z = result.H_t_z;
                my_history[slot].delta_r_x = result.H_r_x;
                my_history[slot].delta_r_y = result.H_r_y;
                my_history[slot].delta_r_z = result.H_r_z;
                my_history[slot].delta_tw_x = result.H_tw_x;
                my_history[slot].delta_tw_y = result.H_tw_y;
                my_history[slot].delta_tw_z = result.H_tw_z;
                my_history[slot].is_active = true;
                my_history[slot].is_slipping = result.is_slipping;
                my_history[slot].is_rolling = result.is_rolling;
                my_history[slot].is_twisting = result.is_twisting;
            }
        } else if (slot >= 0) {
            my_history[slot].is_active = false;
        }
    }

    // Atomic add to global arrays
    atomic_add_force(force_x, force_y, force_z, fx_acc, fy_acc, fz_acc, idx);
    atomic_add_torque(torque_x, torque_y, torque_z, tx_acc, ty_acc, tz_acc, idx);

    // Update contact count
    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// TriMesh Wall Dispatcher
// =============================================================================

class TriMeshWallDispatcher {
private:
    static std::vector<TriMeshSoA> d_meshes;
    static std::vector<TriMeshBVH> bvhs;
    static std::vector<MeshMotionState> motion_states;
    static std::vector<RigidPoseTrajectory> trajectories;
    static std::vector<bool> has_trajectory;
    static bool initialized;

    // Temporary buffers for BVH construction
    static uint32_t* d_morton_codes;
    static int32_t* d_temp_indices;

public:
    static void initialize(
        const SimConfig& config,
        int32_t num_particles
    ) {
        shutdown();  // Clean up any previous state

        int32_t total_tag_offset = MESH_TAG_BASE;

        for (int32_t i = 0; i < config.num_walls; ++i) {
            const WallConfig& wc = config.walls[i];

            if (wc.type == WallType::MESH && wc.enabled) {
                // Load OBJ file
                std::string mesh_path(wc.mesh_path);
                if (mesh_path.empty()) {
                    std::cerr << "WARNING: Empty mesh path for wall " << wc.id << std::endl;
                    continue;
                }

                OBJMesh obj = load_obj_file(mesh_path);
                if (obj.num_faces == 0) {
                    std::cerr << "WARNING: Failed to load mesh for wall " << wc.id << std::endl;
                    continue;
                }

                // Convert to TriMeshSoA (host)
                TriMeshSoA h_mesh = convert_obj_to_trimesh_soa(obj, wc.id, total_tag_offset);
                h_mesh.material_id = wc.material_id;

                // Allocate and copy to device
                TriMeshSoA d_mesh;
                allocate_trimesh_device(d_mesh, h_mesh.num_triangles);
                copy_trimesh_to_device(d_mesh, h_mesh);
                d_meshes.push_back(d_mesh);

                // Allocate BVH
                TriMeshBVH bvh;
                allocate_bvh(bvh, h_mesh.num_triangles, num_particles);
                bvhs.push_back(bvh);

                // Setup motion state
                MeshMotionState motion;
                motion.is_moving = wc.surface_only_motion;
                motion.velocity_x = wc.surface_velocity.x;
                motion.velocity_y = wc.surface_velocity.y;
                motion.velocity_z = wc.surface_velocity.z;
                motion_states.push_back(motion);
                has_trajectory.push_back(wc.rigid_pose_path[0] != '\0');
                trajectories.emplace_back();
                if (has_trajectory.back()) trajectories.back().load(wc.rigid_pose_path);

                // Update tag offset for next mesh
                total_tag_offset += h_mesh.num_triangles;

                // Free host mesh
                free_trimesh_host(h_mesh);

                std::cout << "Initialized mesh wall " << wc.id
                          << " with " << d_mesh.num_triangles << " triangles" << std::endl;
            }
        }

        // Allocate BVH construction buffers
        if (!d_meshes.empty()) {
            int32_t max_triangles = 0;
            for (const auto& m : d_meshes) {
                max_triangles = std::max(max_triangles, m.num_triangles);
            }
            cudaMalloc(&d_morton_codes, max_triangles * sizeof(uint32_t));
            cudaMalloc(&d_temp_indices, max_triangles * sizeof(int32_t));
        }

        initialized = !d_meshes.empty();
        std::cout << "Initialized " << d_meshes.size() << " mesh walls" << std::endl;
    }

    static void build_bvh(
        TriMeshSoA& mesh,
        TriMeshBVH& bvh,
        float max_particle_radius,
        int32_t block_size,
        cudaStream_t stream = 0
    ) {
        int32_t n = mesh.num_triangles;
        if (n == 0) return;

        int32_t num_blocks = (n + block_size - 1) / block_size;

        LAUNCH_KERNEL(copy_and_inflate_aabbs_kernel, num_blocks, block_size, 0, stream,
            mesh.aabb_min_x, mesh.aabb_min_y, mesh.aabb_min_z,
            mesh.aabb_max_x, mesh.aabb_max_y, mesh.aabb_max_z,
            bvh.d_inflated_min_x, bvh.d_inflated_min_y, bvh.d_inflated_min_z,
            bvh.d_inflated_max_x, bvh.d_inflated_max_y, bvh.d_inflated_max_z,
            max_particle_radius, n
        );

        thrust::device_ptr<float> d_min_x(bvh.d_inflated_min_x);
        thrust::device_ptr<float> d_min_y(bvh.d_inflated_min_y);
        thrust::device_ptr<float> d_min_z(bvh.d_inflated_min_z);
        thrust::device_ptr<float> d_max_x(bvh.d_inflated_max_x);
        thrust::device_ptr<float> d_max_y(bvh.d_inflated_max_y);
        thrust::device_ptr<float> d_max_z(bvh.d_inflated_max_z);

        float scene_min[3], scene_max[3];
        scene_min[0] = *thrust::min_element(d_min_x, d_min_x + n);
        scene_min[1] = *thrust::min_element(d_min_y, d_min_y + n);
        scene_min[2] = *thrust::min_element(d_min_z, d_min_z + n);
        scene_max[0] = *thrust::max_element(d_max_x, d_max_x + n);
        scene_max[1] = *thrust::max_element(d_max_y, d_max_y + n);
        scene_max[2] = *thrust::max_element(d_max_z, d_max_z + n);

        float range_x = scene_max[0] - scene_min[0];
        float range_y = scene_max[1] - scene_min[1];
        float range_z = scene_max[2] - scene_min[2];
        float inv_range[3] = {
            range_x > 1e-6f ? 1.0f / range_x : 1.0f,
            range_y > 1e-6f ? 1.0f / range_y : 1.0f,
            range_z > 1e-6f ? 1.0f / range_z : 1.0f
        };

        // Compute Morton codes
        LAUNCH_KERNEL(compute_morton_codes_kernel, num_blocks, block_size, 0, stream,
            bvh.d_inflated_min_x, bvh.d_inflated_min_y, bvh.d_inflated_min_z,
            bvh.d_inflated_max_x, bvh.d_inflated_max_y, bvh.d_inflated_max_z,
            d_morton_codes, d_temp_indices,
            scene_min[0], scene_min[1], scene_min[2],
            inv_range[0], inv_range[1], inv_range[2],
            n
        );

        // Sort by Morton code
        thrust::device_ptr<uint32_t> d_keys(d_morton_codes);
        thrust::device_ptr<int32_t> d_vals(d_temp_indices);
        thrust::sort_by_key(d_keys, d_keys + n, d_vals);

        // Copy sorted indices
        cudaMemcpyAsync(bvh.d_sorted_indices, d_temp_indices,
                        n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        // Build leaf nodes (at indices 0 to n-1)
        LAUNCH_KERNEL(build_bvh_leaves_kernel, num_blocks, block_size, 0, stream,
            bvh.d_inflated_min_x, bvh.d_inflated_min_y, bvh.d_inflated_min_z,
            bvh.d_inflated_max_x, bvh.d_inflated_max_y, bvh.d_inflated_max_z,
            bvh.d_sorted_indices, bvh.d_nodes,
            n, 0  // leaf_offset = 0, leaves stored first
        );

        // Build internal nodes using radix tree approach
        // For n leaves, we have n-1 internal nodes at indices n to 2n-2
        // Each internal node i (at index n + i) has children:
        //   - left: 2*i if 2*i < n-1 then n + 2*i else 2*i - (n-1)
        //   - right: 2*i + 1 if 2*i + 1 < n-1 then n + 2*i + 1 else 2*i + 1 - (n-1)
        // Actually, simpler: use level-by-level construction with correct child indexing
        
        // Build tree level by level from leaves up, merging pairs.
        // Level 0: n leaves at indices [0, n-1]
        // Level 1: ceil(n/2) internal nodes
        // ...until root (single node)
        // When a level has odd size, the last parent gets only a left child.
        if (n > 1) {
            int32_t current_level_start = 0;  // Start of current level (leaves first)
            int32_t current_level_size = n;
            int32_t next_level_start = n;     // Internal nodes start here
            int32_t next_level_size = 0;
            
            while (current_level_size > 1) {
                // Number of parents = ceil(current_level_size / 2)
                next_level_size = (current_level_size + 1) / 2;
                
                int32_t level_blocks = (next_level_size + block_size - 1) / block_size;
                LAUNCH_KERNEL(build_bvh_internal_kernel, level_blocks, block_size, 0, stream,
                    bvh.d_nodes, next_level_start, next_level_size, current_level_start, current_level_size
                );
                
                // Move to next level
                current_level_start = next_level_start;
                current_level_size = next_level_size;
                next_level_start += next_level_size;
            }
            // Root is at current_level_start after the loop
            bvh.root_index = current_level_start;
        } else {
            // Single triangle: root is the leaf itself
            bvh.root_index = 0;
        }

        bvh.valid = true;
    }

    static void compute_forces(
        const SimulationState& state,
        const SimConfig& config,
        float dt,
        float current_time,
        cudaStream_t stream = 0
    ) {
        if (!initialized || d_meshes.empty()) return;

        int32_t num_particles = state.particles.num_particles;
        int32_t block_size = config.gpu.block_size;
        int32_t num_blocks = (num_particles + block_size - 1) / block_size;

        bool use_visco = config.dem_damping_model == DemDampingModel::VISCOUS;
        float tolerance = config.particle.radius_max;  // input-derived max radius until per-pair reach BVH inflation is available

        // Process each mesh wall
        for (size_t m = 0; m < d_meshes.size(); ++m) {
            const TriMeshSoA& mesh = d_meshes[m];
            const TriMeshBVH& bvh = bvhs[m];
            MeshMotionState& motion = motion_states[m];
            if (has_trajectory[m]) {
                const RigidPoseSample sample=trajectories[m].sample(current_time);
                const auto r=RigidPoseTrajectory::rotation_matrix(sample.quaternion_wxyz);
                for (int k=0;k<9;++k) motion.rotation[k]=r[k];
                motion.translation_x=sample.translation[0]; motion.translation_y=sample.translation[1]; motion.translation_z=sample.translation[2];
                const auto& origin=trajectories[m].reference_origin();
                motion.reference_origin_x=origin[0]; motion.reference_origin_y=origin[1]; motion.reference_origin_z=origin[2];
                motion.velocity_x=sample.linear_velocity[0]; motion.velocity_y=sample.linear_velocity[1]; motion.velocity_z=sample.linear_velocity[2];
                motion.omega_x=sample.angular_velocity[0]; motion.omega_y=sample.angular_velocity[1]; motion.omega_z=sample.angular_velocity[2];
                motion.center_x=sample.translation[0]; motion.center_y=sample.translation[1]; motion.center_z=sample.translation[2]; motion.is_moving=true;
            }

            if (!mesh.enabled || !bvh.valid) continue;

            LAUNCH_KERNEL(compute_trimesh_wall_forces_kernel<true>, num_blocks, block_size, 0, stream,
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
                state.particles.omega_x, state.particles.omega_y, state.particles.omega_z,
                state.particles.radius, state.particles.mass, state.particles.moment_of_inertia,
                state.particles.material_id, state.material_database.materials,
                state.material_database.dem_interactions, state.material_database.material_count,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                state.particles.torque_x, state.particles.torque_y, state.particles.torque_z,
                mesh.v0_x, mesh.v0_y, mesh.v0_z,
                mesh.edge1_x, mesh.edge1_y, mesh.edge1_z,
                mesh.edge2_x, mesh.edge2_y, mesh.edge2_z,
                mesh.normal_x, mesh.normal_y, mesh.normal_z,
                mesh.len_edge1_sq, mesh.len_edge2_sq, mesh.dot_e1_e2, mesh.inv_denom,
                mesh.wall_tag, mesh.num_triangles,
                bvh.d_nodes, bvh.num_nodes, bvh.root_index,
                motion,
                state.wall_contact_history.data,
                state.particles.wall_history_offset,
                state.particles.num_wall_contacts,
                state.wall_contact_history.max_contacts_per_particle,
                mesh.material_id, use_visco, dt, tolerance,
                num_particles
            );
        }
    }

    static void shutdown() {
        for (auto& mesh : d_meshes) {
            free_trimesh_device(mesh);
        }
        d_meshes.clear();

        for (auto& bvh : bvhs) {
            free_bvh(bvh);
        }
        bvhs.clear();

        motion_states.clear();
        trajectories.clear();
        has_trajectory.clear();

        if (d_morton_codes) {
            cudaFree(d_morton_codes);
            d_morton_codes = nullptr;
        }
        if (d_temp_indices) {
            cudaFree(d_temp_indices);
            d_temp_indices = nullptr;
        }

        initialized = false;
    }

    static bool is_initialized() { return initialized; }
    static size_t num_meshes() { return d_meshes.size(); }
    static TriMeshSoA& get_mesh(size_t i) { return d_meshes[i]; }
    static TriMeshBVH& get_bvh(size_t i) { return bvhs[i]; }
};

// Static member definitions
std::vector<TriMeshSoA> TriMeshWallDispatcher::d_meshes;
std::vector<TriMeshBVH> TriMeshWallDispatcher::bvhs;
std::vector<MeshMotionState> TriMeshWallDispatcher::motion_states;
std::vector<RigidPoseTrajectory> TriMeshWallDispatcher::trajectories;
std::vector<bool> TriMeshWallDispatcher::has_trajectory;
bool TriMeshWallDispatcher::initialized = false;
uint32_t* TriMeshWallDispatcher::d_morton_codes = nullptr;
int32_t* TriMeshWallDispatcher::d_temp_indices = nullptr;

// =============================================================================
// Free Function Wrappers (callable from WallManager)
// =============================================================================

void trimesh_wall_initialize(const SimConfig& config, int32_t num_particles) {
    TriMeshWallDispatcher::initialize(config, num_particles);
}

void trimesh_wall_build_all_bvhs(float max_particle_radius, int32_t block_size, cudaStream_t stream) {
    if (!TriMeshWallDispatcher::is_initialized()) return;
    for (size_t i = 0; i < TriMeshWallDispatcher::num_meshes(); ++i) {
        TriMeshWallDispatcher::build_bvh(
            TriMeshWallDispatcher::get_mesh(i),
            TriMeshWallDispatcher::get_bvh(i),
            max_particle_radius, block_size, stream
        );
    }
}

void trimesh_wall_compute_forces(
    const SimulationState& state,
    const SimConfig& config,
    float dt,
    float current_time,
    cudaStream_t stream
) {
    TriMeshWallDispatcher::compute_forces(state, config, dt, current_time, stream);
}

void trimesh_wall_shutdown() {
    TriMeshWallDispatcher::shutdown();
}

bool trimesh_wall_is_initialized() {
    return TriMeshWallDispatcher::is_initialized();
}

} // namespace cudadem

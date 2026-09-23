#ifndef CUDADEM_TRIMESH_WALL_H
#define CUDADEM_TRIMESH_WALL_H

// =============================================================================
// Triangle Mesh Wall Module for DEM Simulation
// =============================================================================
// Implements contact forces between particles and triangle mesh walls using
// LBVH for spatial indexing and Hertz-Mindlin contact model.
//
// Reference: lib/forceTriMeshWalls.f90
//
// Key Features:
// - SoA layout for coalesced GPU memory access
// - LBVH spatial indexing for O(log N) triangle queries
// - Contact region classification: Face (1.0), Edge (0.5), Corner (0.2)
// - Reuses calculate_wall_force<> template from plain_wall.cu
// - Per-triangle contact history using wall_tag pattern
// =============================================================================

#include "../constant/command_line_interface.h"
#include "wall.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

namespace cudadem {

// Forward declarations
struct SimulationState;

// =============================================================================
// Constants
// =============================================================================

constexpr int32_t MESH_TAG_BASE = 10000;           // Reserve 0-9999 for plain walls
constexpr int32_t MAX_TRIANGLES_PER_MESH = 100000; // Max triangles per mesh wall
constexpr int32_t MAX_MESH_WALLS = 4;              // Max number of mesh walls
constexpr int32_t MAX_CANDIDATES_PER_PARTICLE = 32; // Max BVH query results

// =============================================================================
// Contact Region Classification
// =============================================================================
// Matches Fortran edgeFlag values for force weighting

enum class ContactRegion : uint8_t {
    NONE = 0,
    FACE = 1,           // edgeFlag = 1.0 (inner triangle)
    EDGE_01 = 2,        // edgeFlag = 0.5 (edge from v0 to v1)
    EDGE_02 = 3,        // edgeFlag = 0.5 (edge from v0 to v2)
    EDGE_12 = 4,        // edgeFlag = 0.5 (edge from v1 to v2)
    CORNER_0 = 5,       // edgeFlag = 0.2 (vertex 0)
    CORNER_1 = 6,       // edgeFlag = 0.2 (vertex 1)
    CORNER_2 = 7        // edgeFlag = 0.2 (vertex 2)
};

__device__ __host__ __forceinline__
float get_edge_flag(ContactRegion region) {
    switch (region) {
        case ContactRegion::FACE:     return 1.0f;
        case ContactRegion::EDGE_01:
        case ContactRegion::EDGE_02:
        case ContactRegion::EDGE_12:  return 0.5f;
        case ContactRegion::CORNER_0:
        case ContactRegion::CORNER_1:
        case ContactRegion::CORNER_2: return 0.2f;
        default:                      return 0.0f;
    }
}

// =============================================================================
// Triangle Mesh SoA (Structure of Arrays) for GPU
// =============================================================================
// Stores all triangles in a mesh with precomputed geometry for fast contact detection

struct TriMeshSoA {
    int32_t num_triangles;
    int32_t mesh_id;           // Wall ID from config
    int32_t material_id;       // Required DEM material id for this wall
    int32_t tag_offset;        // Starting tag: MESH_TAG_BASE + mesh_id * MAX_TRIANGLES_PER_MESH
    bool enabled;

    // Vertex 0 positions (reference point for barycentric coords)
    float* v0_x;
    float* v0_y;
    float* v0_z;

    // Edge vectors from v0 (precomputed)
    float* edge1_x;  // v1 - v0 (VectorTx in Fortran)
    float* edge1_y;
    float* edge1_z;
    float* edge2_x;  // v2 - v0 (VectorTy in Fortran)
    float* edge2_y;
    float* edge2_z;

    // Face normals (unit vectors, precomputed)
    float* normal_x;
    float* normal_y;
    float* normal_z;

    // Precomputed geometry for barycentric coordinates
    float* len_edge1_sq;    // |edge1|^2
    float* len_edge2_sq;    // |edge2|^2
    float* dot_e1_e2;       // edge1 . edge2
    float* inv_denom;       // 1 / (len1_sq * len2_sq - dot^2)

    // Wall tags for contact history (unique per triangle)
    int32_t* wall_tag;

    // AABB for each triangle (for BVH construction)
    float* aabb_min_x;
    float* aabb_min_y;
    float* aabb_min_z;
    float* aabb_max_x;
    float* aabb_max_y;
    float* aabb_max_z;

    __host__ TriMeshSoA()
        : num_triangles(0), mesh_id(-1), material_id(-1), tag_offset(0), enabled(false),
          v0_x(nullptr), v0_y(nullptr), v0_z(nullptr),
          edge1_x(nullptr), edge1_y(nullptr), edge1_z(nullptr),
          edge2_x(nullptr), edge2_y(nullptr), edge2_z(nullptr),
          normal_x(nullptr), normal_y(nullptr), normal_z(nullptr),
          len_edge1_sq(nullptr), len_edge2_sq(nullptr),
          dot_e1_e2(nullptr), inv_denom(nullptr),
          wall_tag(nullptr),
          aabb_min_x(nullptr), aabb_min_y(nullptr), aabb_min_z(nullptr),
          aabb_max_x(nullptr), aabb_max_y(nullptr), aabb_max_z(nullptr) {}
};

// =============================================================================
// Mesh Motion State (for moving/rotating meshes)
// =============================================================================

struct MeshMotionState {
    bool is_moving;

    // Current velocity (for contact force calculation)
    double velocity_x, velocity_y, velocity_z;
    double omega_x, omega_y, omega_z;  // Angular velocity

    // Center of rotation (for computing point velocity)
    double center_x, center_y, center_z;

    // Current body-to-world pose.  The mesh and its BVH always remain in the
    // body frame; the kernel transforms each particle query into that frame.
    double translation_x, translation_y, translation_z;
    double reference_origin_x, reference_origin_y, reference_origin_z;
    double rotation[9]; // row-major body -> world

    // Time-varying motion data (optional)
    const float* d_times;
    const double* d_velocities;   // 3 * num_frames (vx, vy, vz)
    const double* d_omegas;       // 3 * num_frames (wx, wy, wz)
    int32_t num_frames;

    __host__ __device__ MeshMotionState()
        : is_moving(false),
          velocity_x(0), velocity_y(0), velocity_z(0),
          omega_x(0), omega_y(0), omega_z(0),
          center_x(0), center_y(0), center_z(0),
          translation_x(0), translation_y(0), translation_z(0),
          reference_origin_x(0), reference_origin_y(0), reference_origin_z(0),
          d_times(nullptr), d_velocities(nullptr), d_omegas(nullptr), num_frames(0) {
        rotation[0]=rotation[4]=rotation[8]=1.0;
        rotation[1]=rotation[2]=rotation[3]=rotation[5]=rotation[6]=rotation[7]=0.0;
    }
};

// =============================================================================
// BVH Node Structure (simplified for triangle mesh)
// =============================================================================

struct TriMeshBVHNode {
    // AABB bounds
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    // Node indices (-1 for leaf)
    int32_t left_child;
    int32_t right_child;

    // For leaf nodes: triangle index
    int32_t triangle_idx;

    __host__ __device__ TriMeshBVHNode()
        : min_x(0), min_y(0), min_z(0),
          max_x(0), max_y(0), max_z(0),
          left_child(-1), right_child(-1), triangle_idx(-1) {}

    __device__ __forceinline__
    bool is_leaf() const { return left_child < 0; }
};

// =============================================================================
// Triangle Mesh BVH Wrapper
// =============================================================================

struct TriMeshBVH {
    TriMeshBVHNode* d_nodes;  // Device nodes array
    int32_t* d_sorted_indices; // Sorted triangle indices
    int32_t num_nodes;
    int32_t num_leaves;
    int32_t root_index;  // Index of root node in the BVH
    bool valid;

    // Query scratch space (per-particle candidate list)
    int32_t* d_candidate_counts;  // [num_particles]
    int32_t* d_candidates;        // [num_particles * MAX_CANDIDATES_PER_PARTICLE]

    float* d_inflated_min_x;
    float* d_inflated_min_y;
    float* d_inflated_min_z;
    float* d_inflated_max_x;
    float* d_inflated_max_y;
    float* d_inflated_max_z;

    __host__ TriMeshBVH()
        : d_nodes(nullptr), d_sorted_indices(nullptr),
          num_nodes(0), num_leaves(0), root_index(0), valid(false),
          d_candidate_counts(nullptr), d_candidates(nullptr),
          d_inflated_min_x(nullptr), d_inflated_min_y(nullptr), d_inflated_min_z(nullptr),
          d_inflated_max_x(nullptr), d_inflated_max_y(nullptr), d_inflated_max_z(nullptr) {}
};

// =============================================================================
// Triangle Contact Result
// =============================================================================

struct TriangleContactResult {
    bool has_contact;
    ContactRegion region;
    float edge_flag;

    // Closest point on triangle surface (world coordinates)
    float closest_x, closest_y, closest_z;

    // Contact normal (from triangle to particle)
    float normal_x, normal_y, normal_z;

    // Distance and overlap
    float distance;
    float overlap;  // R_particle - distance (positive = contact)

    __device__ TriangleContactResult()
        : has_contact(false), region(ContactRegion::NONE), edge_flag(0.0f),
          closest_x(0), closest_y(0), closest_z(0),
          normal_x(0), normal_y(0), normal_z(0),
          distance(0), overlap(0) {}
};

// =============================================================================
// Device Functions: Triangle Contact Detection
// =============================================================================

// Compute point-to-triangle contact with region classification
// Ported from lib/forceTriMeshWalls.f90 lines 83-182
__device__ __forceinline__
TriangleContactResult compute_triangle_contact(
    // Particle position and radius
    float px, float py, float pz,
    float radius,

    // Triangle data (from TriMeshSoA)
    float v0_x, float v0_y, float v0_z,
    float e1_x, float e1_y, float e1_z,
    float e2_x, float e2_y, float e2_z,
    float n_x, float n_y, float n_z,
    float len1_sq, float len2_sq, float dot_e1_e2, float inv_denom,

    // Tolerance for extended triangle region (0.8 * cell_size in Fortran)
    float tolerance
) {
    TriangleContactResult result;

    // Vector from v0 to particle (RV in Fortran)
    float rv_x = px - v0_x;
    float rv_y = py - v0_y;
    float rv_z = pz - v0_z;

    // Signed distance to triangle plane (ERR in Fortran)
    float signed_dist = rv_x * n_x + rv_y * n_y + rv_z * n_z;

    // Early exit if too far from plane
    if (fabsf(signed_dist) > radius + tolerance) {
        return result;
    }

    // Project particle position onto triangle plane (RVb in Fortran)
    float proj_x = rv_x - signed_dist * n_x;
    float proj_y = rv_y - signed_dist * n_y;
    float proj_z = rv_z - signed_dist * n_z;

    // Compute barycentric-like coordinates
    // RV1 = proj . edge1, RV2 = proj . edge2
    float d1 = proj_x * e1_x + proj_y * e1_y + proj_z * e1_z;
    float d2 = proj_x * e2_x + proj_y * e2_y + proj_z * e2_z;

    // Edge lengths for tolerance scaling
    float len1 = sqrtf(len1_sq);
    float len2 = sqrtf(len2_sq);

    // Barycentric coordinates (RVu, RVv in Fortran)
    // Using formula from Fortran lines 99-100
    float denom = len1_sq * len2_sq - dot_e1_e2 * dot_e1_e2;
    float u, v;
    if (fabsf(denom) > 1e-20f) {
        // Cramer's rule: d1 = u*len1_sq + v*dot_e1_e2
        //                d2 = u*dot_e1_e2 + v*len2_sq
        u = (d1 * len2_sq - d2 * dot_e1_e2) / denom;
        v = (d2 * len1_sq - d1 * dot_e1_e2) / denom;
    } else {
        return result;  // Degenerate triangle
    }

    // Tolerance for extended region (from Fortran lines 101-102)
    float tol1 = tolerance / len1;
    float tol2 = tolerance / len2;

    // Check if within extended triangle region
    if (u < -tol1 || v < -tol2 || (u + v) > 1.0f + tol1 + tol2) {
        return result;  // Outside extended region
    }

    // Classify contact region (matching Fortran lines 103-182)
    float closest_x, closest_y, closest_z;

    if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
        // FACE contact (inner triangle) - Fortran lines 103-112
        result.region = ContactRegion::FACE;
        closest_x = v0_x + proj_x;
        closest_y = v0_y + proj_y;
        closest_z = v0_z + proj_z;
        result.normal_x = n_x;
        result.normal_y = n_y;
        result.normal_z = n_z;
    }
    else if (d1 <= 0.0f && d2 <= 0.0f) {
        // CORNER_0 (v0, adjacent angle) - Fortran lines 114-123
        result.region = ContactRegion::CORNER_0;
        closest_x = v0_x;
        closest_y = v0_y;
        closest_z = v0_z;
    }
    else if (d1 >= len1_sq && (d2 - dot_e1_e2) <= (d1 - len1_sq)) {
        // CORNER_1 (v1, opposite angle in Tx) - Fortran lines 125-134
        result.region = ContactRegion::CORNER_1;
        closest_x = v0_x + e1_x;
        closest_y = v0_y + e1_y;
        closest_z = v0_z + e1_z;
    }
    else if (d2 >= len2_sq && (d1 - dot_e1_e2) <= (d2 - len2_sq)) {
        // CORNER_2 (v2, opposite angle in Ty) - Fortran lines 136-145
        result.region = ContactRegion::CORNER_2;
        closest_x = v0_x + e2_x;
        closest_y = v0_y + e2_y;
        closest_z = v0_z + e2_z;
    }
    else if (d1 > 0.0f && d1 < len1_sq && v < 0.0f) {
        // EDGE_01 (edge from v0 to v1) - Fortran lines 147-157
        result.region = ContactRegion::EDGE_01;
        float t = d1 / len1_sq;
        closest_x = v0_x + t * e1_x;
        closest_y = v0_y + t * e1_y;
        closest_z = v0_z + t * e1_z;
    }
    else if (d2 > 0.0f && d2 < len2_sq && u < 0.0f) {
        // EDGE_02 (edge from v0 to v2) - Fortran lines 159-169
        result.region = ContactRegion::EDGE_02;
        float t = d2 / len2_sq;
        closest_x = v0_x + t * e2_x;
        closest_y = v0_y + t * e2_y;
        closest_z = v0_z + t * e2_z;
    }
    else if ((u + v) > 1.0f) {
        // EDGE_12 (edge from v1 to v2, opposite edge) - Fortran lines 171-182
        result.region = ContactRegion::EDGE_12;
        // Project onto edge v1-v2
        // Edge direction: e2 - e1
        float edge_x = e2_x - e1_x;
        float edge_y = e2_y - e1_y;
        float edge_z = e2_z - e1_z;
        float edge_len_sq = edge_x * edge_x + edge_y * edge_y + edge_z * edge_z;

        // Vector from v1 to projected point
        float to_proj_x = proj_x - e1_x;
        float to_proj_y = proj_y - e1_y;
        float to_proj_z = proj_z - e1_z;

        float t = (to_proj_x * edge_x + to_proj_y * edge_y + to_proj_z * edge_z) / edge_len_sq;
        t = fmaxf(0.0f, fminf(1.0f, t));

        closest_x = v0_x + e1_x + t * edge_x;
        closest_y = v0_y + e1_y + t * edge_y;
        closest_z = v0_z + e1_z + t * edge_z;
    }
    else {
        return result;  // No valid contact region
    }

    // Store closest point
    result.closest_x = closest_x;
    result.closest_y = closest_y;
    result.closest_z = closest_z;

    // Compute distance from particle to closest point
    float dx = px - closest_x;
    float dy = py - closest_y;
    float dz = pz - closest_z;
    result.distance = sqrtf(dx * dx + dy * dy + dz * dz);

    // Compute overlap
    result.overlap = radius - result.distance;

    if (result.overlap > 0.0f) {
        result.has_contact = true;
        result.edge_flag = get_edge_flag(result.region);

        // Compute contact normal (from closest point to particle)
        if (result.distance > 1e-10f) {
            float inv_dist = 1.0f / result.distance;
            result.normal_x = dx * inv_dist;
            result.normal_y = dy * inv_dist;
            result.normal_z = dz * inv_dist;
        } else {
            // Use face normal if distance is zero
            result.normal_x = n_x;
            result.normal_y = n_y;
            result.normal_z = n_z;
        }
    }

    return result;
}

// =============================================================================
// Device Functions: BVH Traversal
// =============================================================================

// Check if AABB overlaps with sphere
__device__ __forceinline__
bool aabb_sphere_overlap(
    float aabb_min_x, float aabb_min_y, float aabb_min_z,
    float aabb_max_x, float aabb_max_y, float aabb_max_z,
    float sphere_x, float sphere_y, float sphere_z,
    float sphere_radius
) {
    // Compute squared distance from sphere center to AABB
    float dist_sq = 0.0f;

    if (sphere_x < aabb_min_x) {
        float d = aabb_min_x - sphere_x;
        dist_sq += d * d;
    } else if (sphere_x > aabb_max_x) {
        float d = sphere_x - aabb_max_x;
        dist_sq += d * d;
    }

    if (sphere_y < aabb_min_y) {
        float d = aabb_min_y - sphere_y;
        dist_sq += d * d;
    } else if (sphere_y > aabb_max_y) {
        float d = sphere_y - aabb_max_y;
        dist_sq += d * d;
    }

    if (sphere_z < aabb_min_z) {
        float d = aabb_min_z - sphere_z;
        dist_sq += d * d;
    } else if (sphere_z > aabb_max_z) {
        float d = sphere_z - aabb_max_z;
        dist_sq += d * d;
    }

    return dist_sq <= sphere_radius * sphere_radius;
}

// BVH traversal to find candidate triangles
// Returns number of candidates found
__device__ __forceinline__
int32_t query_bvh_sphere(
    const TriMeshBVHNode* __restrict__ nodes,
    int32_t num_nodes,
    int32_t root_index,
    float sphere_x, float sphere_y, float sphere_z,
    float sphere_radius,
    int32_t* __restrict__ candidates,
    int32_t max_candidates
) {
    if (num_nodes == 0) return 0;

    int32_t num_found = 0;

    // Stack-based traversal (avoid recursion)
    int32_t stack[64];
    int32_t stack_ptr = 0;
    stack[stack_ptr++] = root_index;  // Start at root

    while (stack_ptr > 0 && num_found < max_candidates) {
        int32_t node_idx = stack[--stack_ptr];

        if (node_idx < 0 || node_idx >= num_nodes) continue;

        const TriMeshBVHNode& node = nodes[node_idx];

        // Check AABB overlap
        if (!aabb_sphere_overlap(
            node.min_x, node.min_y, node.min_z,
            node.max_x, node.max_y, node.max_z,
            sphere_x, sphere_y, sphere_z, sphere_radius)) {
            continue;
        }

        if (node.is_leaf()) {
            // Add triangle to candidates
            if (node.triangle_idx >= 0) {
                candidates[num_found++] = node.triangle_idx;
            }
        } else {
            // Push children onto stack
            if (node.right_child >= 0 && stack_ptr < 63) {
                stack[stack_ptr++] = node.right_child;
            }
            if (node.left_child >= 0 && stack_ptr < 63) {
                stack[stack_ptr++] = node.left_child;
            }
        }
    }

    return num_found;
}

// =============================================================================
// Device Functions: Mesh Velocity at Contact Point
// =============================================================================

__device__ __forceinline__
void compute_mesh_velocity_at_point(
    double& vx, double& vy, double& vz,
    float point_x, float point_y, float point_z,
    const MeshMotionState& motion
) {
    if (!motion.is_moving) {
        vx = 0.0;
        vy = 0.0;
        vz = 0.0;
        return;
    }

    // Translational velocity
    vx = motion.velocity_x;
    vy = motion.velocity_y;
    vz = motion.velocity_z;

    // Add rotational contribution: v = omega x r
    double rx = point_x - motion.center_x;
    double ry = point_y - motion.center_y;
    double rz = point_z - motion.center_z;

    vx += motion.omega_y * rz - motion.omega_z * ry;
    vy += motion.omega_z * rx - motion.omega_x * rz;
    vz += motion.omega_x * ry - motion.omega_y * rx;
}

// =============================================================================
// Host Functions: Memory Management
// =============================================================================

// Allocate device memory for TriMeshSoA
void allocate_trimesh_device(TriMeshSoA& mesh, int32_t num_triangles);

// Copy host mesh data to device
void copy_trimesh_to_device(TriMeshSoA& d_mesh, const TriMeshSoA& h_mesh);

// Free device memory
void free_trimesh_device(TriMeshSoA& mesh);

// Allocate BVH structures
void allocate_bvh(TriMeshBVH& bvh, int32_t num_triangles, int32_t num_particles);

// Free BVH memory
void free_bvh(TriMeshBVH& bvh);

// =============================================================================
// Host Functions: OBJ Loading (implemented in trimesh_loader.cpp)
// =============================================================================

struct OBJMesh {
    std::vector<float> vertices;    // x,y,z triplets
    std::vector<int32_t> faces;     // v0,v1,v2 triplets (0-indexed)
    int32_t num_vertices;
    int32_t num_faces;

    OBJMesh() : num_vertices(0), num_faces(0) {}
};

// Load OBJ file
OBJMesh load_obj_file(const std::string& filepath);

// Convert OBJ to GPU-ready TriMeshSoA (host memory)
TriMeshSoA convert_obj_to_trimesh_soa(
    const OBJMesh& obj,
    int32_t mesh_id,
    int32_t tag_offset
);

// =============================================================================
// Forward Declarations for Dispatcher
// =============================================================================

class TriMeshWallDispatcher;

// =============================================================================
// Free Functions (delegate to TriMeshWallDispatcher, callable from WallManager)
// =============================================================================

void trimesh_wall_initialize(const SimConfig& config, int32_t num_particles);

void trimesh_wall_build_all_bvhs(float max_particle_radius, int32_t block_size, cudaStream_t stream = 0);

void trimesh_wall_compute_forces(
    const SimulationState& state,
    const SimConfig& config,
    float dt,
    float current_time,
    cudaStream_t stream = 0
);

void trimesh_wall_shutdown();

bool trimesh_wall_is_initialized();

} // namespace cudadem

#endif // CUDADEM_TRIMESH_WALL_H

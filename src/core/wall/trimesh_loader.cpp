// =============================================================================
// Triangle Mesh Loader for DEM Simulation
// =============================================================================
// Loads OBJ mesh files and converts to GPU-ready TriMeshSoA format.
// Host-only implementation (no CUDA code).
//
// Reference: lib/forceTriMeshWalls.f90 for triangle data structure
// =============================================================================

#include "trimesh_wall.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace cudadem {

// =============================================================================
// OBJ File Parser
// =============================================================================

OBJMesh load_obj_file(const std::string& filepath) {
    OBJMesh mesh;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open OBJ file: " << filepath << std::endl;
        return mesh;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            // Vertex position
            float x, y, z;
            iss >> x >> y >> z;
            mesh.vertices.push_back(x);
            mesh.vertices.push_back(y);
            mesh.vertices.push_back(z);
            mesh.num_vertices++;
        }
        else if (token == "f") {
            // Face (triangle)
            // OBJ format: f v1 v2 v3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            std::vector<int32_t> face_vertices;

            std::string vertex_str;
            while (iss >> vertex_str) {
                // Parse vertex index (before first '/')
                size_t slash_pos = vertex_str.find('/');
                std::string idx_str = (slash_pos != std::string::npos)
                    ? vertex_str.substr(0, slash_pos)
                    : vertex_str;

                int32_t idx = std::stoi(idx_str);
                // OBJ indices are 1-based, convert to 0-based
                face_vertices.push_back(idx - 1);
            }

            // Triangulate if polygon has more than 3 vertices
            for (size_t i = 1; i + 1 < face_vertices.size(); ++i) {
                mesh.faces.push_back(face_vertices[0]);
                mesh.faces.push_back(face_vertices[i]);
                mesh.faces.push_back(face_vertices[i + 1]);
                mesh.num_faces++;
            }
        }
        // Ignore other tokens (vt, vn, mtllib, usemtl, etc.)
    }

    file.close();

    std::cout << "Loaded OBJ mesh: " << filepath << std::endl;
    std::cout << "  Vertices: " << mesh.num_vertices << std::endl;
    std::cout << "  Triangles: " << mesh.num_faces << std::endl;

    return mesh;
}

// =============================================================================
// Convert OBJ to TriMeshSoA (Host Memory)
// =============================================================================

TriMeshSoA convert_obj_to_trimesh_soa(
    const OBJMesh& obj,
    int32_t mesh_id,
    int32_t tag_offset
) {
    TriMeshSoA mesh;
    mesh.num_triangles = obj.num_faces;
    mesh.mesh_id = mesh_id;
    mesh.tag_offset = tag_offset;
    mesh.enabled = true;

    if (obj.num_faces == 0) {
        std::cerr << "WARNING: Empty mesh for mesh_id " << mesh_id << std::endl;
        return mesh;
    }

    // Allocate host arrays
    int32_t n = obj.num_faces;

    mesh.v0_x = new float[n];
    mesh.v0_y = new float[n];
    mesh.v0_z = new float[n];

    mesh.edge1_x = new float[n];
    mesh.edge1_y = new float[n];
    mesh.edge1_z = new float[n];
    mesh.edge2_x = new float[n];
    mesh.edge2_y = new float[n];
    mesh.edge2_z = new float[n];

    mesh.normal_x = new float[n];
    mesh.normal_y = new float[n];
    mesh.normal_z = new float[n];

    mesh.len_edge1_sq = new float[n];
    mesh.len_edge2_sq = new float[n];
    mesh.dot_e1_e2 = new float[n];
    mesh.inv_denom = new float[n];

    mesh.wall_tag = new int32_t[n];

    mesh.aabb_min_x = new float[n];
    mesh.aabb_min_y = new float[n];
    mesh.aabb_min_z = new float[n];
    mesh.aabb_max_x = new float[n];
    mesh.aabb_max_y = new float[n];
    mesh.aabb_max_z = new float[n];

    // Process each triangle
    for (int32_t i = 0; i < n; ++i) {
        // Get vertex indices
        int32_t i0 = obj.faces[i * 3 + 0];
        int32_t i1 = obj.faces[i * 3 + 1];
        int32_t i2 = obj.faces[i * 3 + 2];

        // Get vertex positions
        float v0x = obj.vertices[i0 * 3 + 0];
        float v0y = obj.vertices[i0 * 3 + 1];
        float v0z = obj.vertices[i0 * 3 + 2];

        float v1x = obj.vertices[i1 * 3 + 0];
        float v1y = obj.vertices[i1 * 3 + 1];
        float v1z = obj.vertices[i1 * 3 + 2];

        float v2x = obj.vertices[i2 * 3 + 0];
        float v2y = obj.vertices[i2 * 3 + 1];
        float v2z = obj.vertices[i2 * 3 + 2];

        // Store v0
        mesh.v0_x[i] = v0x;
        mesh.v0_y[i] = v0y;
        mesh.v0_z[i] = v0z;

        // Compute edge vectors
        float e1x = v1x - v0x;
        float e1y = v1y - v0y;
        float e1z = v1z - v0z;

        float e2x = v2x - v0x;
        float e2y = v2y - v0y;
        float e2z = v2z - v0z;

        mesh.edge1_x[i] = e1x;
        mesh.edge1_y[i] = e1y;
        mesh.edge1_z[i] = e1z;
        mesh.edge2_x[i] = e2x;
        mesh.edge2_y[i] = e2y;
        mesh.edge2_z[i] = e2z;

        // Compute face normal (cross product of edges)
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;

        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-10f) {
            nx /= len;
            ny /= len;
            nz /= len;
        } else {
            // Degenerate triangle
            nx = 0.0f;
            ny = 1.0f;
            nz = 0.0f;
        }

        mesh.normal_x[i] = nx;
        mesh.normal_y[i] = ny;
        mesh.normal_z[i] = nz;

        // Precompute geometry for barycentric coordinates
        float len1_sq = e1x * e1x + e1y * e1y + e1z * e1z;
        float len2_sq = e2x * e2x + e2y * e2y + e2z * e2z;
        float dot = e1x * e2x + e1y * e2y + e1z * e2z;

        mesh.len_edge1_sq[i] = len1_sq;
        mesh.len_edge2_sq[i] = len2_sq;
        mesh.dot_e1_e2[i] = dot;

        float denom = len1_sq * len2_sq - dot * dot;
        mesh.inv_denom[i] = (std::fabs(denom) > 1e-20f) ? 1.0f / denom : 0.0f;

        // Assign unique wall tag
        mesh.wall_tag[i] = tag_offset + i;

        // Compute AABB
        mesh.aabb_min_x[i] = std::min({v0x, v1x, v2x});
        mesh.aabb_min_y[i] = std::min({v0y, v1y, v2y});
        mesh.aabb_min_z[i] = std::min({v0z, v1z, v2z});
        mesh.aabb_max_x[i] = std::max({v0x, v1x, v2x});
        mesh.aabb_max_y[i] = std::max({v0y, v1y, v2y});
        mesh.aabb_max_z[i] = std::max({v0z, v1z, v2z});
    }

    std::cout << "Converted mesh to TriMeshSoA:" << std::endl;
    std::cout << "  Triangles: " << mesh.num_triangles << std::endl;
    std::cout << "  Tag offset: " << mesh.tag_offset << std::endl;

    return mesh;
}

// =============================================================================
// Memory Management (Host-side allocation for host mesh)
// =============================================================================

void free_trimesh_host(TriMeshSoA& mesh) {
    delete[] mesh.v0_x;
    delete[] mesh.v0_y;
    delete[] mesh.v0_z;

    delete[] mesh.edge1_x;
    delete[] mesh.edge1_y;
    delete[] mesh.edge1_z;
    delete[] mesh.edge2_x;
    delete[] mesh.edge2_y;
    delete[] mesh.edge2_z;

    delete[] mesh.normal_x;
    delete[] mesh.normal_y;
    delete[] mesh.normal_z;

    delete[] mesh.len_edge1_sq;
    delete[] mesh.len_edge2_sq;
    delete[] mesh.dot_e1_e2;
    delete[] mesh.inv_denom;

    delete[] mesh.wall_tag;

    delete[] mesh.aabb_min_x;
    delete[] mesh.aabb_min_y;
    delete[] mesh.aabb_min_z;
    delete[] mesh.aabb_max_x;
    delete[] mesh.aabb_max_y;
    delete[] mesh.aabb_max_z;

    mesh = TriMeshSoA();
}

} // namespace cudadem

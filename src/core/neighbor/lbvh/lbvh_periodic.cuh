// =============================================================================
// LBVH Periodic Boundary Utilities
// =============================================================================
// Device functions for periodic-aware LBVH neighbor search:
// - Boundary detection: is particle near periodic boundary?
// - Image generation: which periodic images to query?
// - Image offset computation: convert shift to image index

#ifndef CUDADEM_LBVH_PERIODIC_CUH
#define CUDADEM_LBVH_PERIODIC_CUH

#include "../neighbor_types.h"
#include "../../wall/wall.h"
#include "../../shear/shear_box.h"
#include <cstdint>

namespace cudadem {
namespace lbvh {

// =============================================================================
// Periodic Image Query Structure
// =============================================================================

struct PeriodicImageShift {
    float shift_x, shift_y, shift_z;  // Position offset for query
    int8_t image_x, image_y, image_z;  // Image index for ContactPair
};

// All nine x/y lattice images are queried in shear mode.  The accepted
// unique-image condition guarantees that at most one image of a pair is in
// contact range, so no phase-dependent cache or ambiguous history key exists.
__device__ __forceinline__ int generate_shear_images(
    const PeriodicBoundaryConfig& pbc, const ShearPhase& phase,
    PeriodicImageShift* images) {
    int count=0;
    for (int ix=-1; ix<=1; ++ix) {
        for (int iy=-1; iy<=1; ++iy) {
            const ShearImageTransform t=shear_partner_image(
                ix,iy,0,pbc.box_length_x,pbc.box_length_y,pbc.box_length_z,phase);
            images[count++]={static_cast<float>(-t.x),static_cast<float>(-t.y),0.0f,
                             static_cast<int8_t>(ix),static_cast<int8_t>(iy),0};
        }
    }
    return count;
}

// =============================================================================
// Boundary Detection
// =============================================================================

// Check if particle i is near a periodic boundary in X
// Returns true if particle could potentially contact particles across boundary
__device__ __forceinline__
bool is_near_periodic_boundary_x(
    float px, float r_i, float max_radius,
    const PeriodicBoundaryConfig& pbc
) {
    if (!pbc.periodic_x) return false;
    float search_range = r_i + max_radius;
    return (px - search_range < pbc.box_min_x) ||
           (px + search_range > pbc.box_max_x);
}

__device__ __forceinline__
bool is_near_periodic_boundary_y(
    float py, float r_i, float max_radius,
    const PeriodicBoundaryConfig& pbc
) {
    if (!pbc.periodic_y) return false;
    float search_range = r_i + max_radius;
    return (py - search_range < pbc.box_min_y) ||
           (py + search_range > pbc.box_max_y);
}

__device__ __forceinline__
bool is_near_periodic_boundary_z(
    float pz, float r_i, float max_radius,
    const PeriodicBoundaryConfig& pbc
) {
    if (!pbc.periodic_z) return false;
    float search_range = r_i + max_radius;
    return (pz - search_range < pbc.box_min_z) ||
           (pz + search_range > pbc.box_max_z);
}

// =============================================================================
// Periodic Image Generation
// =============================================================================

// Generate all periodic images to query for a particle
// Returns number of images generated (1 + 2 per near-boundary axis)
// Maximum 27 images (3x3x3), but typically 1-7 for mixed boundaries
__device__ int generate_periodic_images(
    float px, float py, float pz,
    float r_i, float max_radius,
    const PeriodicBoundaryConfig& pbc,
    PeriodicImageShift* images  // Output array (must have space for 27)
) {
    int count = 0;

    // Primary query (no shift)
    images[count++] = {0.0f, 0.0f, 0.0f, 0, 0, 0};

    // Check each axis for periodic boundaries
    bool near_x = is_near_periodic_boundary_x(px, r_i, max_radius, pbc);
    bool near_y = is_near_periodic_boundary_y(py, r_i, max_radius, pbc);
    bool near_z = is_near_periodic_boundary_z(pz, r_i, max_radius, pbc);

    // Generate shifted images
    // For each near-boundary axis, add queries shifted by +/- box_length

    // X-axis shifts
    if (near_x) {
        // +X shift: query from position shifted into +X image
        // This finds particles that are near the high-X boundary
        // Those particles need to be shifted by -box_length for minimum image
        // So image_x = -1 (partner is in -X image relative to original)
        images[count++] = {pbc.box_length_x, 0.0f, 0.0f, -1, 0, 0};
        // -X shift: opposite
        images[count++] = {-pbc.box_length_x, 0.0f, 0.0f, 1, 0, 0};
    }

    // Y-axis shifts
    if (near_y) {
        images[count++] = {0.0f, pbc.box_length_y, 0.0f, 0, -1, 0};
        images[count++] = {0.0f, -pbc.box_length_y, 0.0f, 0, 1, 0};
    }

    // Z-axis shifts
    if (near_z) {
        images[count++] = {0.0f, 0.0f, pbc.box_length_z, 0, 0, -1};
        images[count++] = {0.0f, 0.0f, -pbc.box_length_z, 0, 0, 1};
    }

    // 2D combinations (corner particles)
    if (near_x && near_y) {
        images[count++] = {pbc.box_length_x, pbc.box_length_y, 0.0f, -1, -1, 0};
        images[count++] = {pbc.box_length_x, -pbc.box_length_y, 0.0f, -1, 1, 0};
        images[count++] = {-pbc.box_length_x, pbc.box_length_y, 0.0f, 1, -1, 0};
        images[count++] = {-pbc.box_length_x, -pbc.box_length_y, 0.0f, 1, 1, 0};
    }

    if (near_x && near_z) {
        images[count++] = {pbc.box_length_x, 0.0f, pbc.box_length_z, -1, 0, -1};
        images[count++] = {pbc.box_length_x, 0.0f, -pbc.box_length_z, -1, 0, 1};
        images[count++] = {-pbc.box_length_x, 0.0f, pbc.box_length_z, 1, 0, -1};
        images[count++] = {-pbc.box_length_x, 0.0f, -pbc.box_length_z, 1, 0, 1};
    }

    if (near_y && near_z) {
        images[count++] = {0.0f, pbc.box_length_y, pbc.box_length_z, 0, -1, -1};
        images[count++] = {0.0f, pbc.box_length_y, -pbc.box_length_z, 0, -1, 1};
        images[count++] = {0.0f, -pbc.box_length_y, pbc.box_length_z, 0, 1, -1};
        images[count++] = {0.0f, -pbc.box_length_y, -pbc.box_length_z, 0, 1, 1};
    }

    // 3D combinations (corner particles in fully periodic)
    if (near_x && near_y && near_z) {
        images[count++] = {pbc.box_length_x, pbc.box_length_y, pbc.box_length_z, -1, -1, -1};
        images[count++] = {pbc.box_length_x, pbc.box_length_y, -pbc.box_length_z, -1, -1, 1};
        images[count++] = {pbc.box_length_x, -pbc.box_length_y, pbc.box_length_z, -1, 1, -1};
        images[count++] = {pbc.box_length_x, -pbc.box_length_y, -pbc.box_length_z, -1, 1, 1};
        images[count++] = {-pbc.box_length_x, pbc.box_length_y, pbc.box_length_z, 1, -1, -1};
        images[count++] = {-pbc.box_length_x, pbc.box_length_y, -pbc.box_length_z, 1, -1, 1};
        images[count++] = {-pbc.box_length_x, -pbc.box_length_y, pbc.box_length_z, 1, 1, -1};
        images[count++] = {-pbc.box_length_x, -pbc.box_length_y, -pbc.box_length_z, 1, 1, 1};
    }

    return count;
}

// =============================================================================
// Image Offset Computation
// =============================================================================

// Convert position shift to image index
// shift > 0: we queried from +image, so partner is in -image (image = -1)
// shift < 0: we queried from -image, so partner is in +image (image = +1)
// shift = 0: no shift (image = 0)
__device__ __forceinline__
int8_t shift_to_image(float shift, float box_length) {
    if (box_length <= 0.0f) return 0;
    if (shift > box_length * 0.5f) return -1;
    if (shift < -box_length * 0.5f) return 1;
    return 0;
}

} // namespace lbvh
} // namespace cudadem

#endif // CUDADEM_LBVH_PERIODIC_CUH

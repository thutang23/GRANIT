#ifndef CUDADEM_CONTACT_GEOMETRY_CUH
#define CUDADEM_CONTACT_GEOMETRY_CUH

#include "neighbor_types.h"

namespace cudadem {

// The force and diagnostic paths must interpret stored images identically.
__host__ __device__ inline void apply_contact_image(
    const ContactPair& pair, float& x, float& y, float& z,
    float box_x, float box_y, float box_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    bool shear_enabled, float shear_reduced_shift) {
    if (shear_enabled) {
        x += pair.image_x * box_x;
        y += pair.image_y * box_y - pair.image_x * shear_reduced_shift;
    } else {
        if (pair.image_x != 0 && periodic_x && box_x > 0.0f)
            x += pair.image_x * box_x;
        if (pair.image_y != 0 && periodic_y && box_y > 0.0f)
            y += pair.image_y * box_y;
    }
    if (pair.image_z != 0 && periodic_z && box_z > 0.0f)
        z += pair.image_z * box_z;
}

// Explicit diagnostic representation. CONTACT v1 serializes these seven fields
// (28 bytes); this format is independent of the in-memory candidate ABI.
struct ContactDumpRecord {
    int32_t i, j;
    float overlap, distance, nx, ny, nz;
};
static_assert(sizeof(ContactDumpRecord) == 28, "CONTACT v1 record size changed");

// Only diagnostics invoke this kernel. It uses the physical radii, float
// coordinate conversion and image transform used by DEM force evaluation.
static __global__ void materialize_contact_dump_kernel(
    const ContactPair* pairs, int32_t count,
    const double* pos_x, const double* pos_y, const double* pos_z,
    const float* radius, ContactDumpRecord* records,
    float box_x, float box_y, float box_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    bool shear_enabled, float shear_reduced_shift) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const ContactPair pair = pairs[index];
    const float ix = static_cast<float>(pos_x[pair.i]);
    const float iy = static_cast<float>(pos_y[pair.i]);
    const float iz = static_cast<float>(pos_z[pair.i]);
    float jx = static_cast<float>(pos_x[pair.j]);
    float jy = static_cast<float>(pos_y[pair.j]);
    float jz = static_cast<float>(pos_z[pair.j]);
    apply_contact_image(pair, jx, jy, jz, box_x, box_y, box_z,
        periodic_x, periodic_y, periodic_z, shear_enabled, shear_reduced_shift);
    const float dx = jx - ix, dy = jy - iy, dz = jz - iz;
    const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    const float inverse = distance > 1e-10f ? 1.0f / distance : 0.0f;
    records[index] = {pair.i, pair.j, radius[pair.i] + radius[pair.j] - distance,
        distance, dx * inverse, dy * inverse, dz * inverse};
}

} // namespace cudadem
#endif

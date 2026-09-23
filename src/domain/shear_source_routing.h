#pragma once

#include "morton_partition.h"

namespace cudadem {

#ifdef __CUDACC__
#define GRANIT_SHEAR_ROUTING_HD __host__ __device__
#else
#define GRANIT_SHEAR_ROUTING_HD
#endif

// Ownership is a union of physical XY tiles. Gravity cells live in an
// independent, common 3D image tree. A rank contributes iff an inverse image
// of the source cell intersects one of its tiles inside the central patch.
struct ShearSourceRouting {
    bool enabled = false;
    uint32_t local_owner_mask = 0;
    MortonGrid gravity{};
    const MortonCoverCell* cells = nullptr;
    int cell_count = 0;
    int layers_x = 0, layers_y = 0;
    double length_x = 0, length_y = 0, shift = 0;
    double min_x = 0, max_x = 0, min_y = 0, max_y = 0;
};

GRANIT_SHEAR_ROUTING_HD inline uint32_t shear_source_owner_mask(
    uint64_t morton, int level, const ShearSourceRouting& routing)
{
    if (routing.local_owner_mask) return routing.local_owner_mask;
    uint32_t x = 0, y = 0;
    const int dimension=routing.gravity.dimension;
    for (int bit = 0; bit < level; ++bit) {
        x |= uint32_t((morton >> (dimension * bit)) & 1u) << bit;
        y |= uint32_t((morton >> (dimension * bit + 1)) & 1u) << bit;
    }
    const double h = routing.gravity.size / double(uint64_t{1} << level);
    const double gx = routing.gravity.x + x * h;
    const double gy = routing.gravity.y + y * h;
    // A small outward tolerance protects opposite rounding at a grid face.
    const double pad = 32.0 * 2.2204460492503131e-16 *
        (fabs(routing.gravity.x) + fabs(routing.gravity.y) + routing.gravity.size);
    uint32_t mask = 0;
    for (int c = 0; c < routing.cell_count; ++c) {
        const auto cell = routing.cells[c];
        const uint32_t bit = uint32_t{1} << cell.owner;
        if (mask & bit) continue;
        const double xmin = fmax(cell.lo[0], routing.min_x);
        const double xmax = fmin(cell.hi[0], routing.max_x);
        const double ymin = fmax(cell.lo[1], routing.min_y);
        const double ymax = fmin(cell.hi[1], routing.max_y);
        if (!(xmin < xmax && ymin < ymax)) continue;
        for (int nx = -routing.layers_x; nx <= routing.layers_x; ++nx) {
            const double dx = nx * routing.length_x;
            if (gx + h + pad < xmin + dx || gx - pad > xmax + dx) continue;
            // A physical owner interval becomes at most two wrapped segments.
            const double low = ymin - nx * routing.shift;
            const double high = ymax - nx * routing.shift;
            // The central column preserves the original particle coordinates.
            const int first = nx ? int(floor((low - routing.min_y) / routing.length_y)) : 0;
            const int last = nx ? int(floor((high - routing.min_y) / routing.length_y)) : 0;
            for (int winding = first; winding <= last; ++winding) {
                const double a = fmax(routing.min_y, low - winding * routing.length_y);
                const double b = fmin(nx ? routing.min_y + routing.length_y : routing.max_y,
                                      high - winding * routing.length_y);
                if (a > b) continue;
                for (int ny = -routing.layers_y; ny <= routing.layers_y; ++ny) {
                    const double dy = ny * routing.length_y;
                    if (gy + h + pad >= a + dy && gy - pad <= b + dy) { mask |= bit; break; }
                }
                if (mask & bit) break;
            }
            if (mask & bit) break;
        }
    }
    return mask;
}

} // namespace cudadem
#undef GRANIT_SHEAR_ROUTING_HD

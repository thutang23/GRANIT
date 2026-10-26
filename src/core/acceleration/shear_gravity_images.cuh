#pragma once

#include "../constant/command_line_interface.h"
#include "../shear/shear_box.h"
#include "../wall/wall.h"
#include "body_fitted_fmm/shear_fmm_capacity.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudadem {

// Host-side description of the finite set of physical source replicas used by
// shear-box gravity.  Slot zero is always the central image.  Remaining slots
// are ordered lexicographically in (nx, ny), excluding (0, 0).
struct ShearGravityImages {
    bool enabled = false;
    int32_t layers_x = 0;
    int32_t layers_y = 0;
    int32_t image_count = 1;
    double length_x = 0.0;
    double length_y = 0.0;
    double shear_offset = 0.0;
    double cutoff_radius = 0.0;
    double min_y = 0.0;
};

inline ShearGravityImages make_shear_gravity_images(
    const SimConfig& config,
    const PeriodicBoundaryConfig& pbc,
    double evaluation_time)
{
    ShearGravityImages images;
    images.enabled = config.shear_box.enabled;
    if (!images.enabled) return images;

    images.layers_x = config.shear_box.gravity_layers_x;
    images.layers_y = config.shear_box.gravity_layers_y;
    images.cutoff_radius = config.shear_box.gravity_cutoff_radius;
    if (images.layers_x < 0 || images.layers_y < 0)
        throw std::invalid_argument(
            "shear_box gravity image layers must be non-negative");
    if (!pbc.periodic_x || !pbc.periodic_y ||
        !(pbc.box_length_x > 0.0f) || !(pbc.box_length_y > 0.0f))
        throw std::invalid_argument(
            "shear_box gravity requires positive periodic x and y box lengths");

    const int64_t nx = 2ll * images.layers_x + 1ll;
    const int64_t ny = 2ll * images.layers_y + 1ll;
    if (nx > std::numeric_limits<int32_t>::max() / ny)
        throw std::overflow_error(
            "shear_box gravity image count exceeds int32 capacity");
    const int64_t count = nx * ny;
    images.image_count = static_cast<int32_t>(count);
    images.length_x = static_cast<double>(pbc.box_length_x);
    images.length_y = static_cast<double>(pbc.box_length_y);
    images.min_y = static_cast<double>(pbc.box_min_y);

    const ShearPhase phase = compute_shear_phase_checked(
        config.shear_box, images.length_x, images.length_y, evaluation_time);
    images.shear_offset = phase.reduced_shift;
    return images;
}

// Each X column is wrapped once, then repeated at the finite Y offsets.
// Keep the zero-column coordinates exact for target/self identity.
__host__ __device__ inline double shear_gravity_canonical_y(
    double y, int nx, const ShearGravityImages& images)
{
    if (nx == 0) return y;
    const double shifted = y - nx * images.shear_offset;
    double wrapped = shifted - images.length_y * floor((shifted - images.min_y) / images.length_y);
    if (wrapped >= images.min_y + images.length_y) wrapped -= images.length_y;
    if (wrapped < images.min_y) wrapped += images.length_y;
    return wrapped;
}

__host__ __device__ inline void shear_gravity_image_indices(
    int32_t slot, int32_t layers_x, int32_t layers_y,
    int32_t& image_x, int32_t& image_y)
{
    if (slot == 0) {
        image_x = 0;
        image_y = 0;
        return;
    }
    int32_t ordinal = slot - 1;
    for (int32_t nx = -layers_x; nx <= layers_x; ++nx) {
        for (int32_t ny = -layers_y; ny <= layers_y; ++ny) {
            if (nx == 0 && ny == 0) continue;
            if (ordinal-- == 0) {
                image_x = nx;
                image_y = ny;
                return;
            }
        }
    }
    image_x = image_y = 0;
}

// Shared by the source builder and the admissible-depth census. The physical
// box/ownership grid is distinct from this finite-image gravity envelope.
inline int shear_fmm_effective_depth(const UnifiedFMMConfig& fmm,
    const ShearGravityImages& images, const PeriodicBoundaryConfig& pbc)
{
    const double zspan = pbc.box_length_z > 0.0f
        ? double(pbc.box_length_z) : std::min(images.length_x, images.length_y);
    return std::max(fmm.tree_depth + shear_fmm_capacity::depth_offset(
        images.length_x, images.length_y, zspan, images.layers_x, images.layers_y),
        shear_fmm_capacity::alignment_level(images.length_x, images.length_y,
            zspan, images.layers_x, images.layers_y));
}

inline double4 shear_fmm_static_root(const ShearGravityImages& images,
    const PeriodicBoundaryConfig& pbc, const SimConfig& config)
{
    const double zmin = pbc.periodic_z ? pbc.box_min_z : config.domain.box_min.z;
    const double zmax = pbc.periodic_z ? pbc.box_max_z : config.domain.box_max.z;
    const double side = shear_fmm_capacity::root_side(images.length_x,
        images.length_y, std::max(1.0e-30, zmax-zmin), images.layers_x, images.layers_y);
    return make_double4(0.5*(pbc.box_min_x+pbc.box_max_x)-0.5*side,
        0.5*(pbc.box_min_y+pbc.box_max_y)-0.5*side, 0.5*(zmin+zmax)-0.5*side, side);
}

} // namespace cudadem

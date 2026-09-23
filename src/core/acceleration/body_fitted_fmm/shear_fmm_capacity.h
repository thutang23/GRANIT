#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudadem {
namespace shear_fmm_capacity {

inline int padded_particles(uint64_t particles, double factor)
{
    const double padded=std::ceil(double(particles)*factor);
    if(!particles || !std::isfinite(padded) || padded<1 || padded>INT32_MAX/2)
        throw std::overflow_error("shear particle capacity exceeds edge indexing");
    return static_cast<int>(padded);
}

// Shared with the Karras allocator: power-of-two storage at <= 0.5 load.
inline uint32_t sparse_slots(uint64_t particles, uint64_t nodes, int level, int dimension)
{
    if (!particles || !nodes || level < 0 || level > 31 || (dimension != 2 && dimension != 3))
        throw std::invalid_argument("invalid sparse occupied-cell capacity");
    uint64_t occupied = std::min(particles, nodes);
    if (dimension * level < 63) occupied = std::min(occupied, uint64_t{1} << (dimension * level));
    if (occupied > UINT32_MAX / 2) throw std::overflow_error("sparse lookup level exceeds uint32 capacity");
    const uint64_t wanted = std::max<uint64_t>(2, 2 * occupied);
    uint64_t slots = 1;
    while (slots < wanted) slots <<= 1;
    if (slots > UINT32_MAX) throw std::overflow_error("sparse lookup level exceeds uint32 capacity");
    return static_cast<uint32_t>(slots);
}

// Adjacent sorted-key differences count the first level where prefixes split.
// Summing that histogram gives exact occupied cells, including unary chains.
inline std::array<uint64_t, 21> occupied_nodes(
    const std::array<uint64_t, 21>& first_difference, bool nonempty)
{
    std::array<uint64_t, 21> nodes{};
    uint64_t occupied = nonempty ? 1 : 0;
    nodes[0] = occupied;
    for (int level = 1; level <= 20; ++level) {
        occupied += first_difference[level];
        nodes[level] = nodes[level - 1] + occupied;
    }
    return nodes;
}

inline int occupied_capacity(uint64_t observed, uint64_t strict, double headroom)
{
    if (!observed || observed > strict || !std::isfinite(headroom) || headroom < 1)
        throw std::invalid_argument("invalid occupied-node census capacity");
    const double padded = std::ceil(double(observed) * headroom);
    const uint64_t capacity = padded >= double(strict) ? strict : uint64_t(padded);
    if (capacity > INT32_MAX / 8) throw std::overflow_error("occupied tree exceeds child indexing");
    return static_cast<int>(capacity);
}

// Round the padded envelope to a dyadic multiple of Ly. Y images then
// translate by integer cell offsets at every level >= alignment_level.
inline int alignment_level(double lx, double ly, double lz, int nx, int ny)
{
    if (!(lx > 0) || !(ly > 0) || !(lz > 0) ||
        !std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz) || nx < 0 || ny < 0)
        throw std::invalid_argument("invalid shear FMM extent/image layers");
    const double span = 1.02 * std::max((2.0 * nx + 1.0) * lx,
        std::max((2.0 * ny + 1.0) * ly, lz));
    if (!std::isfinite(span)) throw std::overflow_error("shear FMM envelope overflow");
    int level = 0;
    while (std::ldexp(ly, level) < span) ++level;
    if (!std::isfinite(std::ldexp(ly, level)))
        throw std::overflow_error("shear FMM aligned root overflow");
    return level;
}

inline double root_side(double lx, double ly, double lz, int nx, int ny)
{
    return std::ldexp(ly, alignment_level(lx, ly, lz, nx, ny));
}

inline int depth_offset(double lx, double ly, double lz, int nx, int ny)
{
    const double span = root_side(lx, ly, lz, nx, ny);
    const double reference = std::max(lx, std::max(ly, lz));
    int extra = 0;
    while (std::ldexp(reference, extra) < span) ++extra;
    return extra;
}

// Conservative occupied-cell bound for a source slab, with outward rounding.
// Sources retain their actual Z extent; no vertical confinement is assumed.
inline int slab_nodes(uint64_t particles, int depth, double zmin, double zmax,
                      double root_z, double root_size)
{
    if (!particles || depth < 0 || depth > 20 || !(root_size > 0) ||
        !std::isfinite(zmin) || !std::isfinite(zmax) || zmin > zmax)
        throw std::invalid_argument("invalid shear FMM slab capacity arguments");
    uint64_t total = 0;
    for (int level = 0; level <= depth; ++level) {
        const uint64_t side = uint64_t{1} << level;
        const double scale = double(side) / root_size;
        const double a = std::floor((zmin - root_z) * scale) - 1;
        const double b = std::floor((zmax - root_z) * scale) + 1;
        const uint64_t slabs = static_cast<uint64_t>(std::max(1.0,
            std::min(double(side - 1), b) - std::max(0.0, a) + 1.0));
        total += std::min(particles, side * side * slabs);
        if (total > uint64_t(std::numeric_limits<int>::max()))
            throw std::overflow_error("Shear slab node bound exceeds indexing");
    }
    return static_cast<int>(total);
}

inline uint64_t memory_allowance(uint64_t available)
{
    constexpr uint64_t reserve = 2ull * 1024 * 1024 * 1024;
    return std::min(available * 4 / 5, available > reserve ? available - reserve : 0);
}

} // namespace shear_fmm_capacity
} // namespace cudadem

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define GRANIT_MORTON_HD __host__ __device__
#else
#define GRANIT_MORTON_HD
#endif

namespace cudadem {

// Geometry is independent of rank count. Keys retain 20 bits per active axis;
// ownership cuts always fall between whole cells at the selected FMM depth.
struct MortonGrid {
    double x, y, z, size;
    int dimension;
    int depth;
    uint64_t generation;
    static constexpr int key_depth = 20;

    GRANIT_MORTON_HD uint64_t end_key() const {
        return uint64_t{1} << (dimension * key_depth);
    }
    GRANIT_MORTON_HD bool contains(double px, double py, double pz) const {
        return px >= x && px < x + size && py >= y && py < y + size &&
            (dimension == 2 || (pz >= z && pz < z + size));
    }
    GRANIT_MORTON_HD uint64_t key(double px, double py, double pz) const {
        // Callers must detect escaped particles before keying/migration. The
        // clamps handle representational roundoff only, never root expansion.
        const double scale = static_cast<double>(uint64_t{1} << key_depth) / size;
        const uint32_t limit = (uint32_t{1} << key_depth) - 1;
        const double q[3] = {(px - x) * scale, (py - y) * scale, (pz - z) * scale};
        uint32_t c[3]{};
        for (int a = 0; a < dimension; ++a)
            c[a] = q[a] <= 0 ? 0 : (q[a] >= limit ? limit : static_cast<uint32_t>(q[a]));
        uint64_t result = 0;
        for (int b = 0; b < key_depth; ++b)
            for (int a = 0; a < dimension; ++a)
                result |= uint64_t((c[a] >> b) & 1u) << (dimension * b + a);
        return result;
    }
    GRANIT_MORTON_HD uint64_t leaf_begin(uint64_t key_value) const {
        const int shift = dimension * (key_depth - depth);
        return (key_value >> shift) << shift;
    }
};

GRANIT_MORTON_HD inline int morton_owner(uint64_t key,
                                        const uint64_t* cuts, int ranks) {
    // Upper bound deliberately allows repeated cuts: empty owners are legal.
    int lo = 0, hi = ranks;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (cuts[mid + 1] <= key) lo = mid + 1;
        else hi = mid;
    }
    return lo < ranks ? lo : ranks - 1;
}

struct MortonCoverCell {
    double lo[3];
    double hi[3];
    int owner;
};

// An interval's canonical prefix cover is compact even when its spatial union
// is disconnected. It describes ownership geometry, not the global FMM tree.
inline std::vector<MortonCoverCell> morton_cover(
    const MortonGrid& grid, const std::vector<uint64_t>& cuts) {
    std::vector<MortonCoverCell> cells;
    const int bits = grid.dimension * MortonGrid::key_depth;
    for (int rank = 0; rank + 1 < static_cast<int>(cuts.size()); ++rank) {
        uint64_t begin = cuts[rank], end = cuts[rank + 1];
        while (begin < end) {
            int suffix = 0;
            while (suffix + grid.dimension <= bits) {
                const uint64_t span = uint64_t{1} << (suffix + grid.dimension);
                if ((begin & (span - 1)) || span > end - begin) break;
                suffix += grid.dimension;
            }
            const int level = MortonGrid::key_depth - suffix / grid.dimension;
            const uint64_t prefix = begin >> suffix;
            uint32_t xyz[3]{};
            for (int b = 0; b < level; ++b)
                for (int a = 0; a < grid.dimension; ++a)
                    xyz[a] |= uint32_t((prefix >> (grid.dimension * b + a)) & 1u) << b;
            const double h = std::ldexp(grid.size, -level);
            MortonCoverCell cell{{grid.x + xyz[0] * h, grid.y + xyz[1] * h,
                                   grid.z + xyz[2] * h}, {}, rank};
            for (int a = 0; a < 3; ++a) cell.hi[a] = cell.lo[a] + h;
            if (grid.dimension == 2) {
                cell.lo[2] = -std::numeric_limits<double>::infinity();
                cell.hi[2] = std::numeric_limits<double>::infinity();
            }
            cells.push_back(cell);
            begin += uint64_t{1} << suffix;
        }
    }
    return cells;
}

struct MortonLeafWeight { uint64_t key; double weight; };

// Input contains only occupied leaf summaries, not particle/tree replicas.
// Quantiles are chosen between leaves; a heavy leaf is never split across GPUs.
inline std::vector<uint64_t> weighted_morton_cuts(
    std::vector<MortonLeafWeight> leaves, const MortonGrid& grid, int ranks) {
    if (ranks < 1 || grid.depth < 1 || grid.depth > MortonGrid::key_depth ||
        (grid.dimension != 2 && grid.dimension != 3) || !std::isfinite(grid.size) ||
        grid.size <= 0) throw std::invalid_argument("invalid Morton partition");
    std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
        return a.key < b.key;
    });
    std::vector<MortonLeafWeight> unique;
    for (const auto& leaf : leaves) {
        if (leaf.key >= grid.end_key() || grid.leaf_begin(leaf.key) != leaf.key ||
            !std::isfinite(leaf.weight) || leaf.weight < 0)
            throw std::invalid_argument("invalid Morton leaf weight");
        if (!unique.empty() && unique.back().key == leaf.key)
            unique.back().weight += leaf.weight;
        else unique.push_back(leaf);
    }
    double total = 0;
    for (const auto& leaf : unique) total += leaf.weight;
    if (!std::isfinite(total)) throw std::overflow_error("Morton weight overflow");
    if (total == 0) {
        for (auto& leaf : unique) leaf.weight = 1;
        total = static_cast<double>(unique.size());
    }
    std::vector<uint64_t> cuts(static_cast<size_t>(ranks) + 1, 0);
    cuts.back() = grid.end_key();
    size_t index = 0;
    double prefix = 0;
    for (int rank = 1; rank < ranks; ++rank) {
        const double target = total * (static_cast<double>(rank) / ranks);
        while (index < unique.size() &&
               prefix + unique[index].weight * 0.5 <= target) {
            prefix += unique[index].weight;
            ++index;
        }
        cuts[rank] = index < unique.size() ? unique[index].key : grid.end_key();
    }
    return cuts;
}

} // namespace cudadem
#undef GRANIT_MORTON_HD

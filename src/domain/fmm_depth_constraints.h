#pragma once

#include "morton_partition.h"
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cudadem {

constexpr int kFMMConstraintMaxDepth = 20;
constexpr uint64_t kFMMMaximumLeafParticles = 128;
constexpr double kFMMClumpDiameterMargin = 1.e-6;
using FMMLeafCounts = std::array<uint64_t, kFMMConstraintMaxDepth + 1>;
using FMMEffectiveDepths = std::array<int, kFMMConstraintMaxDepth + 1>;

// A sorted, contiguous interval of the global source-key multiset. Empty
// intervals are legal. Boundary leaves may span any number of intervals.
struct FMMLeafOccupancySummary {
    uint64_t first_key = 0, last_key = 0;
    uint64_t first_count = 0, last_count = 0;
    uint64_t maximum_count = 0, source_count = 0;
};

inline uint64_t fmm_global_maximum_leaf_count(
    const std::vector<FMMLeafOccupancySummary>& shards) {
    uint64_t maximum = 0, carry_key = 0, carry_count = 0;
    bool have_carry = false;
    for (const auto& shard : shards) {
        if (!shard.source_count) continue;
        if (shard.first_key > shard.last_key || !shard.first_count || !shard.last_count ||
            shard.maximum_count < shard.first_count || shard.maximum_count < shard.last_count ||
            shard.maximum_count > shard.source_count ||
            (shard.first_key == shard.last_key &&
             (shard.first_count != shard.source_count || shard.last_count != shard.source_count)) ||
            (shard.first_key != shard.last_key &&
             (shard.first_count > shard.source_count - shard.last_count)))
            throw std::invalid_argument("invalid FMM occupancy shard summary");
        if (have_carry && shard.first_key < carry_key)
            throw std::invalid_argument("FMM occupancy shards are not globally key-ordered");
        maximum = std::max(maximum, shard.maximum_count);
        if (have_carry && carry_key == shard.first_key) {
            if (shard.first_count > std::numeric_limits<uint64_t>::max() - carry_count)
                throw std::overflow_error("global FMM leaf population overflow");
            carry_count += shard.first_count;
        } else {
            carry_key = shard.first_key;
            carry_count = shard.first_count;
        }
        have_carry = true;
        maximum = std::max(maximum, carry_count);
        if (shard.first_key != shard.last_key) {
            carry_key = shard.last_key;
            carry_count = shard.last_count;
        }
    }
    return maximum;
}

inline FMMEffectiveDepths fmm_identity_effective_depths() {
    FMMEffectiveDepths depths{};
    for (int depth = 0; depth <= kFMMConstraintMaxDepth; ++depth) depths[depth] = depth;
    return depths;
}

struct FMMDepthFeasibleRange {
    int minimum_depth = 0, maximum_depth = 0;
    int occupancy_minimum_depth = 0, diameter_maximum_depth = 0;
    int resolution_maximum_depth = 0;
};

// Index counts by actual launched source-tree depth, while the returned range
// uses controller/base depths. The mapping may have plateaus for elongated
// shear boxes: effective=max(base+offset, image-alignment-level).
inline FMMDepthFeasibleRange fmm_feasible_depth_range(
    const FMMLeafCounts& counts, int configured_minimum, int configured_maximum,
    int memory_maximum, double maximum_full_clump_diameter, double root_size,
    const FMMEffectiveDepths& effective = fmm_identity_effective_depths()) {
    if (configured_minimum < 2 || configured_maximum > kFMMConstraintMaxDepth ||
        configured_minimum > configured_maximum || !std::isfinite(root_size) || root_size <= 0 ||
        !std::isfinite(maximum_full_clump_diameter) || maximum_full_clump_diameter < 0)
        throw std::invalid_argument("invalid FMM hard-depth constraints");
    for (int base = configured_minimum; base <= configured_maximum; ++base) {
        if (effective[base] < 2 || (base > configured_minimum && effective[base] < effective[base-1]))
            throw std::invalid_argument("FMM effective-depth mapping must be monotone and at least two");
    }
    FMMDepthFeasibleRange result{};
    result.occupancy_minimum_depth = configured_maximum + 1;
    result.diameter_maximum_depth = configured_minimum - 1;
    result.resolution_maximum_depth = configured_minimum - 1;
    for (int base = configured_minimum; base <= configured_maximum; ++base) {
        const int actual = effective[base];
        if (actual > kFMMConstraintMaxDepth) continue;
        result.resolution_maximum_depth = base;
        if (counts[actual] <= kFMMMaximumLeafParticles)
            result.occupancy_minimum_depth = std::min(result.occupancy_minimum_depth, base);
        const double width = std::ldexp(root_size, -actual);
        if (maximum_full_clump_diameter < width * (1.0-kFMMClumpDiameterMargin))
            result.diameter_maximum_depth = base;
    }
    result.minimum_depth = std::max(configured_minimum, result.occupancy_minimum_depth);
    result.maximum_depth = std::min({configured_maximum, memory_maximum,
                                    result.diameter_maximum_depth, result.resolution_maximum_depth});
    if (result.minimum_depth > result.maximum_depth) {
        std::ostringstream message;
        message << "no feasible FMM depth: actual global leaf population must be <= "
                << kFMMMaximumLeafParticles << "; occupancy requires base depth >= "
                << result.occupancy_minimum_depth << ", full clump diameter "
                << maximum_full_clump_diameter << " requires base depth <= "
                << result.diameter_maximum_depth << ", memory ceiling=" << memory_maximum
                << ", source-tree resolution ceiling=" << result.resolution_maximum_depth
                << ", configured range=[" << configured_minimum << ',' << configured_maximum
                << "], actual root size=" << root_size;
        throw std::runtime_error(message.str());
    }
    // Population counts are exact at every candidate depth. A feasible set
    // must be an interval; never hide a missing/incorrect census behind a clamp.
    for (int base = result.minimum_depth; base <= result.maximum_depth; ++base) {
        const int actual = effective[base];
        if (counts[actual] > kFMMMaximumLeafParticles ||
            !(maximum_full_clump_diameter < std::ldexp(root_size,-actual)*(1.0-kFMMClumpDiameterMargin)))
            throw std::runtime_error("inconsistent FMM hard-constraint census within feasible interval");
    }
    return result;
}

#ifdef __CUDACC__
#define GRANIT_FMM_CONSTRAINT_HD __host__ __device__
#else
#define GRANIT_FMM_CONSTRAINT_HD
#endif

// Reproduce Karras' FP64 normalization and 21-bit quantization exactly, then
// discard its lowest xyz triplet. Prefixes through actual depth20 therefore
// identify the same leaves as the production source tree.
GRANIT_FMM_CONSTRAINT_HD inline bool fmm_constraint_source_key(
    const MortonGrid& root, double x, double y, double z, uint64_t& key) {
    const double inverse = 1.0/root.size;
    double f[3] = {(x-root.x)*inverse, (y-root.y)*inverse, (z-root.z)*inverse};
    uint32_t coordinates[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        if (!(f[axis] >= 0.0 && f[axis] < 1.0)) return false; // rejects NaN too
        if (f[axis] > 0.9999999999999999) f[axis] = 0.9999999999999999;
        coordinates[axis] = static_cast<uint32_t>(f[axis]*double(uint32_t{1}<<21));
    }
    uint64_t full = 0;
    for (int bit = 0; bit < 21; ++bit)
        for (int axis = 0; axis < 3; ++axis)
            full |= uint64_t((coordinates[axis]>>bit)&1u) << (3*bit+axis);
    key = full >> 3;
    return true;
}
#undef GRANIT_FMM_CONSTRAINT_HD

} // namespace cudadem

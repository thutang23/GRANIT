#pragma once

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define GRANIT_SPH_PAIR_HD __host__ __device__
#else
#define GRANIT_SPH_PAIR_HD
#endif

namespace cudadem {

// The caller excludes self pairs. Keep the SPH float arithmetic and strict
// sqrtf cutoff: a squared comparison alone can change boundary membership.
// Skin expands only the candidate list; physics still checks the bare cutoff.
GRANIT_SPH_PAIR_HD inline bool sph_pair_candidate(
    float dx, float dy, float dz, float hi, float hj,
    float kernel_cutoff_factor, float skin)
{
    const float dist_sq = dx * dx + dy * dy + dz * dz;
    const float cutoff = kernel_cutoff_factor * (0.5f * (hi + hj));
    const float range = cutoff + skin;
    // Leave several float ulps around the cutoff for sqrtf/fast-math rounding.
    // Overflow gives infinity and simply disables this optional rejection.
    const float rejection_radius = range * (1.0f + 8.0f * FLT_EPSILON);
    if (dist_sq > rejection_radius * rejection_radius) return false;
    return sqrtf(dist_sq) < range;
}

} // namespace cudadem

#undef GRANIT_SPH_PAIR_HD

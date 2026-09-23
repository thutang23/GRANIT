// Mochi proxy-sphere reduction, adapted for GRANIT candidate lists.
// https://doi.org/10.1145/3797905.3807836
// Radius-first ownership avoids the reference's d == 2*r_small omission.
#include <optix.h>
#include <optix_device.h>
#include "rt_launch_params.h"
#include "../sph_pair.cuh"

extern "C" {
__constant__ cudadem::rt::LaunchParams params = {};
}

extern "C" __global__ void __raygen__neighbors() {
    const unsigned int i = optixGetLaunchIndex().x;
    const auto& in = params.input;
    const float3 origin = make_float3(float(in.pos_x[i]), float(in.pos_y[i]), float(in.pos_z[i]));
    const float support = in.smoothing_length ? in.smoothing_length[i] : in.support_extent[i];
    if (!isfinite(origin.x) || !isfinite(origin.y) || !isfinite(origin.z) ||
        !isfinite(support) || support < 0.0f) return;
    optixTrace(params.traversable, origin, make_float3(0.0f, 0.0f, 1.0f),
               0.0f, 1.0e-16f, 0.0f, 255, OPTIX_RAY_FLAG_NONE, 0, 1, 0);
}

extern "C" __global__ void __miss__neighbors() {}

extern "C" __global__ void __intersection__neighbors() {
    const unsigned int query = optixGetLaunchIndex().x;
    const unsigned int primitive = optixGetPrimitiveIndex();
    if (query == primitive) return;
    const auto& in = params.input;
    const float rq = in.smoothing_length ? in.smoothing_length[query] : in.support_extent[query];
    const float rp = in.smoothing_length ? in.smoothing_length[primitive] : in.support_extent[primitive];
    // For rq <= rp, rq + rp + skin <= 2*rp + skin. Thus the
    // smaller-support particle's point ray must reach the other proxy AABB.
    // Equal supports use array ID to elect exactly one owner.
    // For SPH, hq <= hp instead gives cutoff*(hq+hp)/2 <= cutoff*hp.
    if (rq > rp || (rq == rp && query > primitive)) return;

    // Canonical identities also keep pair arithmetic independent of which
    // particle owns the ray. SPH keeps its own cutoff and close-pair semantics.
    const unsigned int i = min(query, primitive);
    const unsigned int j = max(query, primitive);
    if (in.smoothing_length) {
        const float dx = float(in.pos_x[i]) - float(in.pos_x[j]);
        const float dy = float(in.pos_y[i]) - float(in.pos_y[j]);
        const float dz = float(in.pos_z[i]) - float(in.pos_z[j]);
        if (cudadem::sph_pair_candidate(dx, dy, dz,
                in.smoothing_length[i], in.smoothing_length[j],
                in.kernel_cutoff_factor, in.skin))
            optixReportIntersection(0.0f, 0);
        return;
    }
    // Preserve the DEM LBVH arithmetic, near-coincident exclusion and clumps.
    if (in.clump_id && in.clump_id[i] > 0 && in.clump_id[i] == in.clump_id[j]) return;
    const float dx = float(in.pos_x[j]) - float(in.pos_x[i]);
    const float dy = float(in.pos_y[j]) - float(in.pos_y[i]);
    const float dz = float(in.pos_z[j]) - float(in.pos_z[i]);
    const float dist_sq = dx * dx + dy * dy + dz * dz;
    const float threshold = -in.skin;
    const float range = in.support_extent[i] + in.support_extent[j] - threshold;
    if (dist_sq < range * range && dist_sq > 1.0e-12f) {
        const float distance = sqrtf(dist_sq);
        const float overlap = in.support_extent[i] + in.support_extent[j] - distance;
        if (overlap > threshold) optixReportIntersection(0.0f, 0);
    }
}

extern "C" __global__ void __anyhit__neighbors() {
    // The GAS uses REQUIRE_SINGLE_ANYHIT_CALL. Side effects belong here,
    // rather than in an intersection callback that may be evaluated again.
    const unsigned int i = optixGetLaunchIndex().x;
    const unsigned int j = optixGetPrimitiveIndex();
    const unsigned long long index = atomicAdd(params.count, 1ULL);
    if (index < static_cast<unsigned long long>(params.capacity))
        params.pairs[index] = cudadem::ContactPair(min(i, j), max(i, j));
    optixIgnoreIntersection();
}

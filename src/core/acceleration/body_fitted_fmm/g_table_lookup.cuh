#pragma once

#include <cstdint>
#include "g_tables.h"
#include "planar_gap1_operator_table.h"

namespace body_fitted_fmm {

__host__ __device__ __forceinline__
bool g_table_lookup(int8_t signed_gap, int ux, int uy, int uz,
                    uint16_t* canonical_id, uint8_t* orientation_id)
{
    int abs_u[3] = { (ux >= 0 ? ux : -ux), (uy >= 0 ? uy : -uy), (uz >= 0 ? uz : -uz) };
    bool sgn[3]  = { ux >= 0, uy >= 0, uz >= 0 };
    int  idx[3]  = { 0, 1, 2 };

    if (abs_u[idx[0]] < abs_u[idx[1]]) { int t=idx[0]; idx[0]=idx[1]; idx[1]=t; }
    if (abs_u[idx[1]] < abs_u[idx[2]]) { int t=idx[1]; idx[1]=idx[2]; idx[2]=t; }
    if (abs_u[idx[0]] < abs_u[idx[1]]) { int t=idx[0]; idx[0]=idx[1]; idx[1]=t; }

    int cu = abs_u[idx[0]], cv = abs_u[idx[1]], cw = abs_u[idx[2]];

    int axis_perm;
    if (idx[0]==0 && idx[1]==1)      axis_perm = 0;
    else if (idx[0]==0 && idx[1]==2) axis_perm = 1;
    else if (idx[0]==1 && idx[1]==0) axis_perm = 2;
    else if (idx[0]==1 && idx[1]==2) axis_perm = 3;
    else if (idx[0]==2 && idx[1]==0) axis_perm = 4;
    else                             axis_perm = 5;

    int sign_bits = (sgn[idx[0]] ? 0 : 1) | (sgn[idx[1]] ? 0 : 2) | (sgn[idx[2]] ? 0 : 4);
    *orientation_id = (uint8_t)(axis_perm * 8 + sign_bits);

    int count;
    const int(*offsets)[3];
    if (signed_gap == 0)      { count = G_GAP0_COUNT;    offsets = G_GAP0_OFFSETS; }
    else if (signed_gap == 1) { count = G_GAP1_FWD_COUNT; offsets = G_GAP1_FWD_OFFSETS; }
    else if (signed_gap == -1){ count = G_GAP1_REV_COUNT; offsets = G_GAP1_REV_OFFSETS; }
    else                      { return false; }

    int lo = 0, hi = count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp;
        if (offsets[mid][0] != cu)      cmp = (cu > offsets[mid][0]) ? 1 : -1;
        else if (offsets[mid][1] != cv) cmp = (cv > offsets[mid][1]) ? 1 : -1;
        else if (offsets[mid][2] != cw) cmp = (cw > offsets[mid][2]) ? 1 : -1;
        else { *canonical_id = (uint16_t)mid; return true; }
        if (cmp > 0) lo = mid + 1; else hi = mid - 1;
    }

    // Keep historic gap +1 canonical IDs 0..24 untouched and append the
    // planar family at 25..28.  Gap -1 deliberately has no planar extension:
    // target-centric traversal only emits coarse-source -> fine-target work.
    if (signed_gap == 1 && cw == 0) {
        // Spell out the four append-only planar cases so this host/device
        // helper does not read a __device__ metadata array on its host path.
        if (cu == 10 && cv == 6)  { *canonical_id = G_GAP1_PLANAR_BASE + 0; return true; }
        if (cu == 10 && cv == 10) { *canonical_id = G_GAP1_PLANAR_BASE + 1; return true; }
        if (cu == 14 && cv == 2)  { *canonical_id = G_GAP1_PLANAR_BASE + 2; return true; }
        if (cu == 18 && cv == 2)  { *canonical_id = G_GAP1_PLANAR_BASE + 3; return true; }
    }
    return false;
}

} // namespace body_fitted_fmm

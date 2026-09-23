// Compact gravity-body view shared by DIRECT and BFFMM.
//
// Karras/FMM must consume only this view for a clump run.  In particular, the
// source index is never a member-particle index for a clump body, preventing a
// rigid body from being separated across leaves or interacting with itself.
#pragma once

#include <cuda_runtime.h>

namespace body_fitted_fmm {

enum class GravityBodyKind : unsigned char { NormalParticle = 0, ClumpCoM = 1 };

struct GravityBodyView {
    // Compact manager-owned SoA consumed by Karras/P2M/P2P.  Positions have
    // exactly `count = normal_count + clump_count` entries.
    double* pos_x = nullptr;
    double* pos_y = nullptr;
    double* pos_z = nullptr;
    double* mass = nullptr;
    int* source_index = nullptr;             // ParticleSoA index or clump state index
    GravityBodyKind* kind = nullptr;
    int normal_count = 0;
    int clump_count = 0;
    int count = 0;
};

__host__ __device__ inline bool gravity_body_view_valid(const GravityBodyView& v) {
    return v.count == v.normal_count + v.clump_count && v.normal_count >= 0 &&
           v.clump_count >= 0 && (v.count == 0 ||
           (v.pos_x && v.pos_y && v.pos_z && v.mass && v.source_index && v.kind));
}

__host__ __device__ inline GravityBodyKind gravity_body_kind_at(const GravityBodyView& v, int body) {
    // The range convention is intentional redundancy with kind[]: it makes
    // the normal/member/dead exclusion auditable before device allocation.
    return body < v.normal_count ? GravityBodyKind::NormalParticle : GravityBodyKind::ClumpCoM;
}

} // namespace body_fitted_fmm

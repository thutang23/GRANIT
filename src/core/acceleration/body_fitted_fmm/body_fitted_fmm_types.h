// =============================================================================
// Body-Fitted FMM Types — Grouped Traversal + Target-Owned M2L
// =============================================================================
// Extends FMM types with the canonical G-table and null-node padding.
// =============================================================================

#ifndef CUDADEM_BODY_FITTED_FMM_TYPES_H
#define CUDADEM_BODY_FITTED_FMM_TYPES_H

#include "../fmm/fmm_types.h"

namespace body_fitted_fmm {

// Reserved index for null (boundary padding) node.
// All out-of-bounds Morton lookups redirect here.
// Uses -1 to avoid collision with real node indices (especially root node at 0).
// The pack kernel checks for this sentinel and writes zero multipoles.
constexpr int32_t NULL_NODE_INDEX = -1;

// Device data for body-fitted FMM (extends FMMDeviceData conceptually)
struct BodyFittedFMMDeviceData {
    // ========================================================================
    // Full 48-orientation G-table (precomputed on host, single-stage M2L)
    // ========================================================================
    // d_G_full: precomposed full G-table (G_full[sc][lane] = A_R * G_canon * B_R
    //   baked in for all 48 orientations, computed on host in double precision).
    //   Reference modes use NC × 32 = 896 floats (rows 0..27, cols 0..31;
    //   columns 28..31 padded with 0.0f). TF32_TC uses a mode-specific 32 × 32
    //   matrix whose final four rows are also zero-padded and whose active
    //   entries are TF32-rounded on the host. Layout per signed gap segment:
    //     canonical_id 0..count-1, then for each cid the 48 orientations
    //     (mode-specific matrix stride each).
    //   Index via __constant__ d_canonG_gap_info[9] (signed_gap+4) then
    //     Gcb = float_offset + (cid*48 + oid_gen) * mode-specific stride.
    float* d_G_full = nullptr;
    size_t d_G_full_bytes = 0;

    BodyFittedFMMDeviceData() : d_G_full(nullptr),
                                 d_G_full_bytes(0) {}
};

} // namespace body_fitted_fmm

#endif // CUDADEM_BODY_FITTED_FMM_TYPES_H

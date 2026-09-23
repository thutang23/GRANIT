// =============================================================================
// Task Structs: Typed task descriptors for 4-segment task arena
// =============================================================================
// PairTask: 8-byte P2P pair descriptor (source, target, reserved tile coords)
// M2LTask:  8-byte M2L pair descriptor (source, target, operator, delta, flags)
// =============================================================================

#pragma once
#include <cstdint>

namespace body_fitted_fmm {

struct PairTask {
    uint64_t source : 24;  // source node index
    uint64_t target : 24;  // target node index
    uint64_t tile_t : 8;   // reserved; explicit tiling is disabled
    uint64_t tile_s : 8;   // reserved; explicit tiling is disabled
};
static_assert(sizeof(PairTask) == 8, "PairTask must be 8 bytes");

struct M2LTask {
    uint64_t source : 23;     // source node index
    uint64_t target : 23;     // target node index
    uint64_t operatorId : 14; // canonical G-table operator ID
    int64_t  levelDelta : 2;  // signed gap (0 or ±1)
    uint64_t flags : 2;       // reserved
};
static_assert(sizeof(M2LTask) == 8, "M2LTask must be 8 bytes");

// Pack canonical_id (max ~25) and orientation_id (max 47) into uint16_t operatorId
static __device__ __host__ __forceinline__
uint16_t pack_operator_id(uint16_t cid, uint8_t oid) {
    return (uint16_t)((oid << 8) | (cid & 0xFF));
}

// Unpack operatorId into canonical_id and orientation_id
static __device__ __host__ __forceinline__
void unpack_operator_id(uint16_t packed, uint16_t& cid, uint8_t& oid) {
    cid = packed & 0xFF;
    oid = (packed >> 8) & 0xFF;
}

// Segment classification enum (matches grouped interaction counter layout).
enum class SegClass : int {
    M2L_SAME  = 0,
    M2L_CROSS = 1,
    P2P_REG   = 2,
    P2P_SELF  = 3
};

}  // namespace body_fitted_fmm

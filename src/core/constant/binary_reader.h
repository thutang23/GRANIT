// =============================================================================
// Unified Binary Reader — read_unified_binary()
// =============================================================================
// Reads the unified GRANIT binary format and populates ParticleSoA + metadata.
// =============================================================================

#ifndef CUDADEM_BINARY_READER_H
#define CUDADEM_BINARY_READER_H

#include "binary_format.h"

namespace cudadem {

bool read_unified_binary(
    const std::string& path,
    ParticleSoA& particles,
    SimulationState& state,
    const SimConfig& config,
    int32_t extra_capacity = 0);

} // namespace cudadem

#endif // CUDADEM_BINARY_READER_H

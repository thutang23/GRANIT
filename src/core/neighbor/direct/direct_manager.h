// =============================================================================
// Direct O(N²) Neighbor Search Manager
// =============================================================================
// Brute-force neighbor search for validation purposes.
// Checks all N(N-1)/2 particle pairs - simple but slow.
// Use only for testing/verification with small particle counts (≤10K).

#ifndef CUDADEM_DIRECT_MANAGER_H
#define CUDADEM_DIRECT_MANAGER_H

#include "../neighbor_types.h"
#include "../../constant/command_line_interface.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace cudadem {

// Forward declaration
struct SimulationState;

namespace direct {

// =============================================================================
// DirectSearchImpl - O(N²) Brute-Force Implementation
// =============================================================================

class DirectSearchImpl {
public:
    // =====================================================================
    // Lifecycle
    // =====================================================================

    static void initialize(const SimConfig& config, SimulationState& state);
    static void shutdown(SimulationState& state);
    static void shutdown(SimulationState& state, int32_t gpu_id);
    static void shutdown_all();

    // =====================================================================
    // Contact Detection
    // =====================================================================

    static void find_contacts(
        SimulationState& state,
        const SimConfig& config,
        ContactListBuffer& contacts,
        cudaStream_t stream = 0
    );

    // =====================================================================
    // Rebuild Control (No-ops for direct method)
    // =====================================================================

    static bool needs_rebuild(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        // Always compute fresh - no caching
        return true;
    }

    static void rebuild(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        // No-op: direct method has no data structures to rebuild
    }

    // Cache invalidation (force full rebuild on next call - no-op for Direct)
static void invalidate_cache() {}

static void launch_async_displacement_monitor(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        // No-op: direct method doesn't track displacement
    }

    // =====================================================================
    // Query
    // =====================================================================

    static bool is_initialized();
    static bool is_initialized(int32_t gpu_id);
    static const NeighborSearchStats& get_stats();
    static const NeighborSearchStats& get_stats(int32_t gpu_id);

private:
    static int32_t s_initialized_gpu;
    static NeighborSearchStats s_stats;
};

} // namespace direct
} // namespace cudadem

#endif // CUDADEM_DIRECT_MANAGER_H

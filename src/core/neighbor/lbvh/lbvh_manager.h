// =============================================================================
// LBVH Manager
// =============================================================================
// Internal implementation class for LBVH neighbor search.

#ifndef CUDADEM_LBVH_MANAGER_H
#define CUDADEM_LBVH_MANAGER_H

#include "lbvh_types.h"
#include "../neighbor_types.h"
#include "../../constant/command_line_interface.h"
#include "../../constant/system_initialize.h"
#include <unordered_map>
#include <mutex>

namespace cudadem {
namespace lbvh {

// =============================================================================
// LBVHSearchImpl - Internal Implementation Class
// =============================================================================

class LBVHSearchImpl {
public:
    // Lifecycle
    static void initialize(const SimConfig& config, SimulationState& state);
    static void shutdown(SimulationState& state);
    static void shutdown(SimulationState& state, int32_t gpu_id);
    static void shutdown_all();

    // Contact detection
    static void find_contacts(
        SimulationState& state,
        const SimConfig& config,
        ContactListBuffer& contacts,
        double evaluation_time,
        cudaStream_t stream = 0
    );

    // SPH structure update and directed candidate preparation.
    static void find_sph_neighbors(SimulationState& state, const SimConfig& config,
                                   cudaStream_t stream = 0);

    // Refresh the existing SPH tree after h changes, without rebuilding its
    // topology or enumerating an unused gravity neighbor list.
    static void refresh_sph_bounds(SimulationState& state, cudaStream_t stream = 0);

    // Async monitoring
    static void launch_async_displacement_monitor(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    );

    // Query
    static bool is_initialized();
    static bool is_initialized(int32_t gpu_id);
    static const NeighborSearchStats& get_stats();
    static const NeighborSearchStats& get_stats(int32_t gpu_id);
    static NeighborBackend get_backend();
    static NeighborBackend get_backend(int32_t gpu_id);
    static std::string get_backend_reason();
    // Component CUDA-event timing adds synchronization; diagnostics opt in.
    static void set_backend_timing_enabled(bool enabled);

    // Rebuild control (public for NeighborSearch facade)
    static bool needs_rebuild(const SimulationState& state, const SimConfig& config, cudaStream_t stream);
    static void rebuild(SimulationState& state, const SimConfig& config, cudaStream_t stream);

    // Cache invalidation (force full rebuild on next call)
    static void invalidate_cache();

private:
    static LBVHGPUState& get_gpu_state();
    static LBVHGPUState& get_gpu_state(int32_t gpu_id);

    static std::unordered_map<int32_t, LBVHGPUState> gpu_states_;
    static std::mutex state_mutex_;
    static NeighborSearchStats default_stats_;
};

} // namespace lbvh
} // namespace cudadem

#endif // CUDADEM_LBVH_MANAGER_H

// =============================================================================
// Cell-Linked List Manager
// =============================================================================
// Internal implementation class for CLL neighbor search.

#ifndef CUDADEM_CLL_MANAGER_H
#define CUDADEM_CLL_MANAGER_H

#include "cll_types.h"
#include "../neighbor_types.h"
#include "../../constant/command_line_interface.h"
#include "../../constant/system_initialize.h"
#include <unordered_map>
#include <mutex>

namespace cudadem {
namespace cll {

// =============================================================================
// CLLSearchImpl - Internal Implementation Class
// =============================================================================

class CLLSearchImpl {
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

    // Rebuild control (public for NeighborSearch facade)
    static bool needs_rebuild(const SimulationState& state, const SimConfig& config, cudaStream_t stream);
    static void rebuild(SimulationState& state, const SimConfig& config, cudaStream_t stream);

// Cache invalidation (force full rebuild on next call - no-op for CLL)
static void invalidate_cache() {}

    // Contact buffer management (for backward compat with kernels)
    static void allocate_contact_buffer(ContactListBuffer& buffer, int32_t max_contacts);
    static void free_contact_buffer(ContactListBuffer& buffer);
    static void reset_contact_buffer(ContactListBuffer& buffer, cudaStream_t stream = 0);
    static int32_t get_contact_count(ContactListBuffer& buffer, cudaStream_t stream = 0);

private:
    static CLLGPUState& get_gpu_state();
    static CLLGPUState& get_gpu_state(int32_t gpu_id);

    // Internal operations
    static void compute_cell_ids(SimulationState& state, cudaStream_t stream);
    static void sort_particles_by_cell(SimulationState& state, cudaStream_t stream);
    static void build_cell_ranges(SimulationState& state, cudaStream_t stream);
    static void reorder_particle_data(SimulationState& state, cudaStream_t stream);

    static std::unordered_map<int32_t, CLLGPUState> gpu_states_;
    static std::mutex state_mutex_;
    static NeighborSearchStats default_stats_;
};

} // namespace cll
} // namespace cudadem

#endif // CUDADEM_CLL_MANAGER_H

// =============================================================================
// Neighbor Search Unified API
// =============================================================================
// Single entry point for all neighbor search operations.
// Dispatches to CLL or LBVH implementation based on config.neighbor.method.

#ifndef CUDADEM_NEIGHBOR_API_H
#define CUDADEM_NEIGHBOR_API_H

#include "neighbor_types.h"
#include "cll/cll_manager.h"
#include "lbvh/lbvh_manager.h"
#include "direct/direct_manager.h"
#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"

namespace cudadem {

// =============================================================================
// NeighborSearch - Unified Static Facade
// =============================================================================

class NeighborSearch {
public:
    // =========================================================================
    // Lifecycle
    // =========================================================================

    static void initialize(const SimConfig& config, SimulationState& state) {
        state.neighbor.active_method = config.neighbor.method;
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::initialize(config, state);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::initialize(config, state);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::initialize(config, state);
                break;
        }
    }

    static void shutdown(SimulationState& state) {
        switch (state.neighbor.active_method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::shutdown(state);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::shutdown(state);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::shutdown(state);
                break;
        }
    }

    static void shutdown(SimulationState& state, int32_t gpu_id) {
        switch (state.neighbor.active_method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::shutdown(state, gpu_id);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::shutdown(state, gpu_id);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::shutdown(state, gpu_id);
                break;
        }
    }

    static void shutdown_all() {
        cll::CLLSearchImpl::shutdown_all();
        lbvh::LBVHSearchImpl::shutdown_all();
        direct::DirectSearchImpl::shutdown_all();
    }

    // =========================================================================
    // Contact Detection
    // =========================================================================

    static void find_contacts(
        SimulationState& state,
        const SimConfig& config,
        ContactListBuffer& contacts,
        double evaluation_time,
        cudaStream_t stream = 0
    ) {
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::find_contacts(state, config, contacts, stream);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::find_contacts(state, config, contacts, evaluation_time, stream);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::find_contacts(state, config, contacts, stream);
                break;
        }
    }

    // =========================================================================
    // Contact Buffer Management
    // =========================================================================

    static void allocate_contact_buffer(ContactListBuffer& buffer, int32_t max_contacts) {
        cudaMalloc(&buffer.pairs, max_contacts * sizeof(ContactPair));
        cudaMalloc(&buffer.num_contacts, sizeof(int32_t));
        buffer.max_contacts = max_contacts;
        buffer.h_num_contacts = 0;
        cudaMemset(buffer.num_contacts, 0, sizeof(int32_t));
    }

    static void free_contact_buffer(ContactListBuffer& buffer) {
        if (buffer.pairs) cudaFree(buffer.pairs);
        if (buffer.num_contacts) cudaFree(buffer.num_contacts);
        buffer.pairs = nullptr;
        buffer.num_contacts = nullptr;
        buffer.max_contacts = 0;
        buffer.h_num_contacts = 0;
    }

    static void reset_contact_buffer(ContactListBuffer& buffer, cudaStream_t stream = 0) {
        cudaMemsetAsync(buffer.num_contacts, 0, sizeof(int32_t), stream);
        buffer.h_num_contacts = 0;
    }

    static int32_t get_contact_count(ContactListBuffer& buffer, cudaStream_t stream = 0) {
        cudaMemcpyAsync(&buffer.h_num_contacts, buffer.num_contacts, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return buffer.h_num_contacts;
    }

    // =========================================================================
    // Rebuild Control
    // =========================================================================

    static bool needs_rebuild(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                return cll::CLLSearchImpl::needs_rebuild(state, config, stream);
            case NeighborMethod::LBVH:
                return lbvh::LBVHSearchImpl::needs_rebuild(state, config, stream);
            case NeighborMethod::DIRECT:
                return direct::DirectSearchImpl::needs_rebuild(state, config, stream);
        }
        return false;
    }

    static void rebuild(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::rebuild(state, config, stream);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::rebuild(state, config, stream);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::rebuild(state, config, stream);
                break;
        }
    }

    // =========================================================================
    // Cache Invalidation
    // =========================================================================

    static void invalidate_cache() {
        cll::CLLSearchImpl::invalidate_cache();
        lbvh::LBVHSearchImpl::invalidate_cache();
        direct::DirectSearchImpl::invalidate_cache();
    }

    // =========================================================================
    // Async Displacement Monitoring
    // =========================================================================

    static void launch_async_displacement_monitor(
        const SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::launch_async_displacement_monitor(state, config, stream);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::launch_async_displacement_monitor(state, config, stream);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::launch_async_displacement_monitor(state, config, stream);
                break;
        }
    }

    // =========================================================================
    // SPH-Specific Neighbor Search
    // =========================================================================

    static void find_sph_neighbors(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        // LBVH prepares reusable directed candidates for supported SPH domains;
        // other methods retain their native device query representation.
        state.neighbor.sph_adjacency_valid = false;
        switch (config.neighbor.method) {
            case NeighborMethod::CELL_LINKED_LIST:
                cll::CLLSearchImpl::rebuild(state, config, stream);
                break;
            case NeighborMethod::LBVH:
                lbvh::LBVHSearchImpl::find_sph_neighbors(state, config, stream);
                break;
            case NeighborMethod::DIRECT:
                direct::DirectSearchImpl::rebuild(state, config, stream);
                break;
        }
    }

    // =========================================================================
    // Query
    // =========================================================================

    static bool is_initialized() {
        // Check all methods - whichever is active will return true
        return cll::CLLSearchImpl::is_initialized() ||
               lbvh::LBVHSearchImpl::is_initialized() ||
               direct::DirectSearchImpl::is_initialized();
    }

    static bool is_initialized(int32_t gpu_id) {
        return cll::CLLSearchImpl::is_initialized(gpu_id) ||
               lbvh::LBVHSearchImpl::is_initialized(gpu_id) ||
               direct::DirectSearchImpl::is_initialized(gpu_id);
    }

    static const NeighborSearchStats& get_stats() {
        // Return stats from whichever is initialized
        if (cll::CLLSearchImpl::is_initialized()) {
            return cll::CLLSearchImpl::get_stats();
        }
        if (lbvh::LBVHSearchImpl::is_initialized()) {
            return lbvh::LBVHSearchImpl::get_stats();
        }
        return direct::DirectSearchImpl::get_stats();
    }

    static const NeighborSearchStats& get_stats(int32_t gpu_id) {
        if (cll::CLLSearchImpl::is_initialized(gpu_id)) {
            return cll::CLLSearchImpl::get_stats(gpu_id);
        }
        if (lbvh::LBVHSearchImpl::is_initialized(gpu_id)) {
            return lbvh::LBVHSearchImpl::get_stats(gpu_id);
        }
        return direct::DirectSearchImpl::get_stats(gpu_id);
    }
};

} // namespace cudadem

#endif // CUDADEM_NEIGHBOR_API_H

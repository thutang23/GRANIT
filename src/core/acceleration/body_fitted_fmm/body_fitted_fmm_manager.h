// =============================================================================
// Body-Fitted FMM Manager — MAC-Generated Grouped Traversal
// =============================================================================

#ifndef CUDADEM_BODY_FITTED_FMM_MANAGER_H
#define CUDADEM_BODY_FITTED_FMM_MANAGER_H

#include "../acceleration.h"
#include "body_fitted_fmm_types.h"
#include "persistent_tree_traversal.cuh"
#include "../../../domain/rcb_work_model.h"
#include <cstdint>
#include <vector>

namespace body_fitted_fmm {

struct M2LTask;
struct PairTask;
struct GroupedTargetInteractions;
struct PersistentTreeMetadata;

// Snapshot of the current GPU's device-resident FMM work.  The dispatcher owns
// all storage; callers must treat pointers as read-only and discard the view
// when valid is false or generation changes.
struct FMMWorkloadView {
    const UnifiedNode* nodes = nullptr;
    const int* sorted_to_original = nullptr;
    int num_nodes = 0;
    int num_particles = 0;

    const M2LTask* m2l_same = nullptr;
    const M2LTask* m2l_cross = nullptr;
    const PairTask* p2p_regular = nullptr;
    const PairTask* p2p_self = nullptr;
    int m2l_same_count = 0;
    int m2l_cross_count = 0;
    int p2p_regular_count = 0;
    int p2p_self_count = 0;

    // Production grouped representation. Legacy flat fields above remain for
    // source compatibility and are null/zero for grouped traversal.
    const GroupedTargetInteractions* grouped_targets = nullptr;
    const int* children_compact = nullptr;
    const PersistentTreeMetadata* metadata = nullptr;
    int node_capacity = 0;

    uint64_t generation = 0;
    bool valid = false;
};

// Fine-grained active-time telemetry emitted by the dispatcher.  `sample` is
// directly consumable by the adaptive RCB calibrator; this layer does not pool
// or calibrate it.  Pointers and CUDA events remain entirely dispatcher-owned.
enum class BFFMMActivePhase : uint8_t {
    TreeTaskRebuild,
    M2LFarField,
    P2P,
    ParticleL2P,
    RemoteLET,
};

struct BFFMMActiveSample {
    BFFMMActivePhase phase = BFFMMActivePhase::TreeTaskRebuild;
    ::cudadem::RCBActiveSample sample{};
};

using AdaptiveFMMDepthTransition = AdaptiveDepthTransition;
using AdaptiveFMMDepthControllerState = AdaptiveDepthState;
using AdaptiveFMMDepthDiagnostic = AdaptiveDepthDiagnostic;
using AdaptiveFMMDepthCheckpoint = AdaptiveDepthCheckpoint;
using AdaptiveFMMDepthControlMode = AdaptiveDepthControlMode;
using ExternalDistributedFMMDepthCommand = ExternalDistributedDepthCommand;
using ExternalDistributedFMMDepthEvaluation = ExternalDistributedDepthEvaluation;

// Compatibility record for the former distributed depth controller. The common
// Morton path reports exchange and evaluation timing through DistributedFMMMetrics.
struct MultiGPUFMMGravityEpochTiming {
    bool valid = false;
    uint64_t sequence = 0;
    uint64_t gravity_evaluation_epoch = 0;
    int64_t simulation_step = 0;
    double evaluation_time = 0.0;
    int64_t owned_particles = 0;
    double elapsed_ms = 0.0;
    // Phase-4 remote LET force evaluation only.  This is zero for an empty
    // owned region and does not include LET transfer/staging waits.
    double remote_evaluation_ms = 0.0;
    int64_t let_send_node_records = 0;
    int64_t let_send_particle_records = 0;
    uint64_t let_send_bytes = 0;
    int64_t let_recv_node_records = 0;
    int64_t let_recv_particle_records = 0;
    uint64_t let_recv_bytes = 0;
    ExternalDistributedFMMDepthEvaluation depth{};
};

// One completed common-tree evaluation; phase durations use one CUDA stream.
// Exchange includes protocol waits and is never treated as an active-work rate.
struct DistributedFMMMetrics {
    uint64_t epoch=0, received_bytes=0;
    int64_t owned=0;
    double total_ms=0, tree_ms=0, far_ms=0, near_ms=0, exchange_ms=0, communication_ms=0;
};
struct DistributedFMMParticleWork { double tree=0, far=0, near=0, communication=0; };

class UnifiedFMMDispatcher {
public:
    static void validate_distributed_particle_work(cudaStream_t stream=0);
    static void validate_distributed_coverage(int global_particles);
    static int distributed_depth_ceiling(const ::cudadem::SimConfig& config, int global_particles,
        const ::cudadem::SimulationState* local_state=nullptr,
        const ::cudadem::PeriodicBoundaryConfig* pbc=nullptr, cudaStream_t stream=0);
    static DistributedFMMMetrics distributed_metrics(cudaStream_t stream=0);
    static const DistributedFMMParticleWork* distributed_particle_work_device(int& count, double totals[4], cudaStream_t stream);
    static void distributed_particle_work(std::vector<DistributedFMMParticleWork>& work,
                                          cudaStream_t stream=0);
    static void set_distributed_compute_depth(int depth);
    static void set_distributed_clump_exclusion_diameter(double diameter);
    static void set_distributed_partition(const ::cudadem::DomainParamsGPU& params);
    static void initialize(const ::cudadem::SimConfig& config, int max_particles);
    static void shutdown();
    static void shutdown_all();

    // Optional IDs are used only by the certified small-clump DEM shadow
    // view. Positive equal IDs suppress near pairs; callers must establish
    // that same-clump members cannot enter accepted far interactions.
    static void compute(::cudadem::SimulationState& state, const ::cudadem::SimConfig& config,
                        const ::cudadem::NeighborData& neighbor_data,
                        const ::cudadem::PeriodicBoundaryConfig& pbc,
                        double evaluation_time, cudaStream_t stream = 0,
                        bool synchronize_at_end = true,
                        const int* gravity_clump_ids = nullptr);

    static void compute_outer(::cudadem::SimulationState& state, const ::cudadem::SimConfig& config,
                              const ::cudadem::NeighborData& neighbor_data,
                              const ::cudadem::PeriodicBoundaryConfig& pbc,
                              double evaluation_time, cudaStream_t stream = 0,
                              const int* gravity_clump_ids = nullptr);

    // Independent aligned component trees for green and each large clump.
    // root_geometry is [size,min_x,min_y,min_z] on device; segment offsets
    // are relative to member_begin, matching ClumpSegmentOffsets.
    static void compute_aligned_clumps(
        ::cudadem::SimulationState& members, const ::cudadem::SimConfig& config,
        int clump_count, int member_begin, const int* segment_begin,
        const int* segment_end, const int* clump_ids, const int* large_flags,
        const double* root_geometry, cudaStream_t stream = 0);
    static ::cudadem::AlignedClumpGravityDiagnostics aligned_clump_diagnostics();
    static void reset_aligned_clump_diagnostics();

    // Multi-GPU compute uses the common Morton grid and compact halo exchange.
    // Empty owners still publish an epoch. The communicator must be initialized;
    // a non-Morton configuration is supported only with one rank.
    // Optional membership is refreshed after migration. The caller guarantees every
    // clump diameter is below the actual source-tree leaf width. Clump mode
    // requires controlled 3D DEM, zero occupancy fallbacks, and no projection.
    // Shear images additionally require 2*diameter < min(Lx,Ly), preserving
    // nonlocal replicas while the near filter removes the internal body copy.
#ifdef CUDADEM_HAVE_LET
    static void compute_multigpu(
        ::cudadem::SimulationState& state,
        const ::cudadem::SimConfig& config,
        const ::cudadem::NeighborData& neighbor_data,
        const ::cudadem::PeriodicBoundaryConfig& pbc,
        ::cudadem::LETCommunicator& let_comm,
        uint64_t gravity_evaluation_epoch,
        double evaluation_time,
        cudaStream_t stream = 0,
        const int* gravity_clump_ids = nullptr);

    // Collective RESPA slow-force endpoint.  Unlike compute_multigpu(), this
    // writes acceleration (not force) to gravity_force_* and receives an
    // explicit monotonic gravity epoch so it never aliases inner-step ids.
    static void compute_outer_multigpu(
        ::cudadem::SimulationState& state,
        const ::cudadem::SimConfig& config,
        const ::cudadem::NeighborData& neighbor_data,
        const ::cudadem::PeriodicBoundaryConfig& pbc,
        ::cudadem::LETCommunicator& let_comm,
        uint64_t gravity_evaluation_epoch,
        double evaluation_time,
        cudaStream_t stream = 0,
        const int* gravity_clump_ids = nullptr);
#endif

    static void build_tree(::cudadem::SimulationState& state, const ::cudadem::SimConfig& config,
                           cudaStream_t stream);

    static bool needs_rebuild(const ::cudadem::SimulationState& state);
    static void force_rebuild();

    // Called after this GPU's communicator membership changes. It invalidates
    // any published view before the next mandatory rebuild.
    static void invalidate_current_gpu_workload();

    // Returns the current GPU's read-only device work view.  It is valid only
    // after a completed tree/task rebuild and until invalidated/rebuilt.
    static FMMWorkloadView get_workload_view();

    // Harvest completed event-ring entries for the current GPU.  Never waits
    // on the stream; entries still in flight remain queued for a later poll.
    static void poll_active_samples(std::vector<BFFMMActiveSample>& out);

    // Read the managed fail-stop mailbox after an existing synchronization
    // boundary; this method never synchronizes by itself.
    static void throw_if_terminal_error();

    // Validation support: returns FMM interaction statistics (M2L/P2P pair counts)
    static fmm::FMMStats get_statistics();

    // Validation/profiling support: device-global elapsed times for the most
    // recent persistent tree build and stencil traversal.
    static void get_last_persistent_phase_times(double& tree_ms,
                                                double& stencil_ms);

    // Validation-only timing populated when BF_FMM_VALIDATE_M2L_TIMING is set.
    // The opt-in path records CUDA events immediately around the M2L launch.
    static void get_last_m2l_phase_time(double& m2l_ms);

    // Adaptive-depth checkpoint/diagnostic lifecycle.  Restore is safe before
    // lazy FMM initialization; the checkpoint is staged and applied before
    // the first gravity build on that GPU.
    static bool get_adaptive_fmm_depth_checkpoint(
        AdaptiveFMMDepthCheckpoint* checkpoint,
        cudaStream_t stream = 0);
    static bool restore_adaptive_fmm_depth_checkpoint(
        const AdaptiveFMMDepthCheckpoint& checkpoint,
        cudaStream_t stream = 0);
    static void consume_adaptive_fmm_depth_diagnostics(
        std::vector<AdaptiveFMMDepthDiagnostic>& diagnostics,
        uint64_t* dropped,
        cudaStream_t stream = 0);
    static bool get_adaptive_fmm_depth_telemetry(
        AdaptiveDepthTelemetry* telemetry,
        cudaStream_t stream = 0);
    // Call only at an existing output/checkpoint synchronization boundary.
    // Returns true when storage was grown and depth graphs were recaptured.
    static bool service_adaptive_fmm_capacity(
        ::cudadem::SimulationState& state,
        const ::cudadem::SimConfig& config,
        cudaStream_t stream = 0);

    // Explicit external distributed control is staged before lazy
    // initialization when needed.  It reuses the adaptive depth graphs while
    // suppressing only the device-local neighbor-trial FSM; the owned-particle
    // startup probe and capacity/fallback machinery remain active.
    static void set_adaptive_fmm_depth_control_mode(
        AdaptiveFMMDepthControlMode mode, cudaStream_t stream = 0);
    static bool set_external_adaptive_fmm_depth_command(
        const ExternalDistributedFMMDepthCommand& command,
        cudaStream_t stream = 0);
    static bool get_external_adaptive_fmm_depth_evaluation(
        ExternalDistributedFMMDepthEvaluation* evaluation,
        cudaStream_t ordered_after_stream = 0);

    // Compatibility API: the retired timing mailbox emits no records and leaves
    // `dropped` at zero. Use DistributedFMMMetrics for the common Morton path.
    static void poll_multigpu_gravity_epoch_timings(
        std::vector<MultiGPUFMMGravityEpochTiming>& completed,
        uint64_t* dropped = nullptr);

    // Validation support: get raw pair data + tree structure for completeness check.
    // Copies all device arrays to host vectors. Must be called AFTER compute().
    static void get_completeness_data(
        std::vector<int2>& h_m2l_pairs,          // [m2l_count] (target, source)
        std::vector<int>&  h_m2l_ti,             // [m2l_count] packed operator IDs
        std::vector<int2>& h_p2p_pairs,          // [regular_p2p_count] (target, source)
        std::vector<int>&  h_p2p_self_nodes,     // [self_p2p_count] leaf node indices
        std::vector<fmm::FMMOctreeNode>& h_nodes,  // [num_nodes]
        std::vector<int>&  h_children_indices,   // [num_nodes * 8]
        std::vector<int>&  h_particle_to_leaf,  // [num_particles]
        int& num_nodes,
        int& num_particles);

    // Validation support: get multipole and local expansion data for decomposition dump.
    // Copies device arrays to host vectors. Must be called AFTER compute().
    static void get_expansion_data(
        std::vector<fmm::MultipoleExpansion>& h_multipoles,
        std::vector<fmm::LocalExpansion>& h_local,
        int& num_nodes);

    // DEBUG: dump the exact device inputs the LET push kernel reads
    // (num_nodes, level_start/level_count arrays, first nodes' aabb/children).
    static void debug_let_inputs();

private:
    static int current_gpu();
    static void allocate_memory(int gpu_id, int num_particles, int max_nodes,
                                const ::cudadem::SimConfig& config);
    static void initialize_adaptive(
        const ::cudadem::SimConfig& config,
        const ::cudadem::SimulationState& state,
        int particle_count,
        cudaStream_t stream);
    static void free_memory(int gpu_id);
};

} // namespace body_fitted_fmm

#endif // CUDADEM_BODY_FITTED_FMM_MANAGER_H

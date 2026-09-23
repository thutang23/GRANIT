#ifndef CUDADEM_DISTRIBUTED_FMM_DEPTH_CONTROLLER_H
#define CUDADEM_DISTRIBUTED_FMM_DEPTH_CONTROLLER_H

// Pure-host scheduler for adaptive BFFMM depth in the multi-GPU runner.  This
// deliberately owns no CUDA objects: GPU workers publish completed timing
// records asynchronously and GPU 0 consumes only complete epochs.

#include "../core/constant/command_line_interface.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace cudadem {

enum class DistributedFMMDepthCancelReason : uint8_t {
    Rebalance,
    CapacityService,
    YorpTransition,
    LocalCountChange,
    IncompleteTelemetry,
};

enum class DistributedFMMDepthEvent : uint8_t {
    Initialized,
    TrialStarted,
    TrialSample,
    TrialConfirmed,
    DepthAdopted,
    CandidateRejected,
    TrialCancelled,
    SampleDropped,
    EffectiveCeilingChanged,
    Frozen,
};

// Produced after a full-rebuild evaluation. `elapsed_ns` must cover the whole local
// gravity epoch (tree build, LET work, transfers, and remote evaluation), not
// merely local tree construction.  One record is published by each GPU.
struct DistributedFMMDepthEpochTiming {
    uint64_t epoch = 0;
    // Copy these from DistributedFMMDepthCommand at launch time.  They bind
    // an asynchronously completed record to the controller state that chose
    // its depth, rather than the state that happens to consume it later.
    uint64_t command_trial_id = 0;
    int32_t launch_depth = 0;
    uint64_t elapsed_ns = 0;
    // Optional actual-rank mean when a common-tree wrapper supplies a max.
    uint64_t aggregate_mean_ns = 0;
    uint64_t let_bytes = 0;
    int64_t local_owned_particles = 0;
    bool telemetry_valid = true;
    bool rcb_accepted = false;
    bool capacity_event = false;
    bool yorp_transition = false;
    bool local_count_changed = false;
};

struct DistributedFMMDepthCommand {
    int32_t launch_depth = 0;
    bool exclude_from_rcb = false;
    uint64_t trial_id = 0;
};

struct DistributedFMMDepthGpuState {
    int32_t selected_depth = 0;
    int32_t candidate_depth = 0;
    int32_t effective_max_depth = 0;
    int32_t search_direction = 1;
    int32_t confirmation_count = 0;
    int32_t selected_evaluations_since_trial = 0;
    int64_t stable_local_owned_particles = 0;
};

struct DistributedFMMDepthDiagnostic {
    uint64_t sequence = 0;
    uint64_t epoch = 0;
    uint64_t trial_id = 0;
    int32_t gpu_id = -1;
    int32_t selected_depth = 0;
    int32_t candidate_depth = 0;
    double global_max_ms = 0.0;
    double global_mean_ms = 0.0;
    uint64_t let_bytes = 0;
    double primary_relative_improvement = 0.0;
    DistributedFMMDepthEvent event = DistributedFMMDepthEvent::Initialized;
    DistributedFMMDepthCancelReason cancel_reason =
        DistributedFMMDepthCancelReason::IncompleteTelemetry;
};

// A bounded, host-only coordinator.  It never waits for a worker: incomplete
// or overwritten mailbox entries are discarded, which makes timing telemetry
// incapable of adding a global synchronization point to a physics step.
class DistributedFMMDepthController {
public:
    static constexpr int32_t kMaxGpus = 8;
    static constexpr int32_t kMailboxSlots = 32;
    static constexpr size_t kDiagnosticCapacity = 256;

    bool initialize(const AdaptiveDepthConfig& config, int32_t gpu_count,
                    const std::vector<int32_t>& initial_depths);
    // Startup probes can finish lazily and independently.  An empty
    // initial_depths vector leaves all GPU commands unset until this method
    // supplies their measured seed; no trial starts until every GPU is ready.
    bool set_initial_depth(int32_t gpu, int32_t depth, int32_t effective_max_depth,
                           uint64_t epoch = 0);

    bool enabled() const;
    bool frozen() const;
    std::string failure_reason() const;
    int32_t gpu_count() const;

    DistributedFMMDepthCommand command_for_gpu(int32_t gpu) const;
    DistributedFMMDepthGpuState state_for_gpu(int32_t gpu) const;
    bool trial_active() const;
    uint64_t active_trial_id() const;

    // Safe for calls from the persistent worker threads.  A duplicate record
    // is a controller invariant failure; stale/overwritten timing is dropped.
    bool publish_timing(int32_t gpu, const DistributedFMMDepthEpochTiming& timing);
    // Called by GPU 0's host controller loop.  Consumes all complete mailbox
    // epochs currently available; it never blocks for missing records.
    size_t process_ready_epochs();
    // Tell the controller that an incomplete epoch cannot arrive before a
    // mailbox slot is reused.  This cancels a trial sample but never physics.
    void discard_timing_before(uint64_t epoch);

    // RCB/capacity/YORP events invalidate a paired comparison.  The selected
    // depths remain active and the next trial starts after a fresh interval.
    void notify_disruptive_event(uint64_t epoch,
                                 DistributedFMMDepthCancelReason reason);
    // A failed deeper allocation only limits its owning GPU.  The controller
    // clamps that GPU's selected/candidate depths and resumes correctly.
    void set_effective_max_depth(int32_t gpu, int32_t effective_max_depth,
                                 uint64_t epoch);
    void freeze(const std::string& reason, uint64_t epoch = 0);

    std::vector<DistributedFMMDepthDiagnostic> diagnostics() const;
    uint64_t dropped_diagnostics() const;

private:
    struct MailboxSlot {
        uint64_t epoch = 0;
        std::vector<DistributedFMMDepthEpochTiming> timings;
        std::vector<bool> present;
    };
    struct Aggregate {
        double max_ms = 0.0;
        double mean_ms = 0.0;
        uint64_t let_bytes = 0;
    };

    enum class TrialPhase : uint8_t { SelectedA, CandidateB, CandidateB2, SelectedA2 };

    bool valid_gpu_locked(int32_t gpu) const;
    void process_epoch_locked(uint64_t epoch,
                              const std::vector<DistributedFMMDepthEpochTiming>& timings);
    void begin_next_trial_locked(uint64_t epoch);
    void cancel_trial_locked(uint64_t epoch, DistributedFMMDepthCancelReason reason);
    void finish_trial_locked(uint64_t epoch);
    void freeze_locked(const std::string& reason, uint64_t epoch);
    void emit_locked(DistributedFMMDepthEvent event, uint64_t epoch, int32_t gpu,
                     const Aggregate* aggregate = nullptr,
                     double improvement = 0.0,
                     DistributedFMMDepthCancelReason reason =
                         DistributedFMMDepthCancelReason::IncompleteTelemetry);
    Aggregate aggregate_locked(const std::vector<DistributedFMMDepthEpochTiming>& timings) const;
    bool counts_changed_locked(const std::vector<DistributedFMMDepthEpochTiming>& timings) const;
    bool timing_matches_active_phase_locked(
        const std::vector<DistributedFMMDepthEpochTiming>& timings) const;
    bool timing_matches_selected_commands_locked(
        const std::vector<DistributedFMMDepthEpochTiming>& timings) const;
    bool phase_uses_candidate_locked() const;
    void advance_phase_locked();
    int32_t candidate_for_locked(int32_t gpu) const;

    mutable std::mutex mutex_;
    AdaptiveDepthConfig config_{};
    int32_t gpu_count_ = 0;
    bool initialized_ = false;
    bool frozen_ = false;
    std::string failure_reason_;
    std::vector<DistributedFMMDepthGpuState> gpu_states_;
    std::vector<bool> initial_depth_ready_;
    std::vector<bool> stable_count_valid_;
    std::vector<MailboxSlot> mailbox_;
    uint64_t last_processed_epoch_ = 0;
    uint64_t discarded_before_epoch_ = 0;
    uint64_t next_trial_id_ = 1;
    uint64_t active_trial_id_ = 0;
    int32_t active_owner_ = -1;
    int32_t next_owner_ = 0;
    TrialPhase phase_ = TrialPhase::SelectedA;
    int32_t warmups_remaining_ = 0;
    uint64_t phase_samples_ = 0;
    uint64_t samples_a_ = 0;
    uint64_t samples_b_ = 0;
    Aggregate total_a_{};
    Aggregate total_b_{};
    std::vector<DistributedFMMDepthDiagnostic> diagnostics_;
    uint64_t next_sequence_ = 1;
    uint64_t dropped_diagnostics_ = 0;
};

}  // namespace cudadem

#endif  // CUDADEM_DISTRIBUTED_FMM_DEPTH_CONTROLLER_H

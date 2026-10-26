#include "distributed_fmm_depth_controller.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace cudadem {
namespace {

bool all_ready(const std::vector<bool>& ready) {
    return !ready.empty() &&
           std::all_of(ready.begin(), ready.end(), [](bool value) { return value; });
}

}  // namespace

bool DistributedFMMDepthController::initialize(
    const AdaptiveDepthConfig& config, int32_t gpu_count,
    const std::vector<int32_t>& initial_depths) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ || !config.enabled || gpu_count <= 0 || gpu_count > kMaxGpus ||
        (!initial_depths.empty() && static_cast<int32_t>(initial_depths.size()) != gpu_count)) {
        return false;
    }
    config_ = config;
    gpu_count_ = gpu_count;
    gpu_states_.assign(gpu_count_, DistributedFMMDepthGpuState{});
    for (auto& state : gpu_states_) {
        state.effective_min_depth = config_.min_depth;
        state.effective_max_depth = config_.max_depth;
    }
    initial_depth_ready_.assign(gpu_count_, false);
    stable_count_valid_.assign(gpu_count_, false);
    mailbox_.resize(kMailboxSlots);
    for (MailboxSlot& slot : mailbox_) {
        slot.timings.resize(gpu_count_);
        slot.present.assign(gpu_count_, false);
    }
    initialized_ = true;
    if (!initial_depths.empty()) {
        for (int32_t gpu = 0; gpu < gpu_count_; ++gpu) {
            const int32_t depth = initial_depths[gpu];
            if (depth < config_.min_depth || depth > config_.max_depth) {
                freeze_locked("initial FMM depth is outside configured range", 0);
                return false;
            }
            DistributedFMMDepthGpuState& state = gpu_states_[gpu];
            state.selected_depth = depth;
            state.candidate_depth = depth;
            state.effective_max_depth = config_.max_depth;
            initial_depth_ready_[gpu] = true;
        }
    }
    emit_locked(DistributedFMMDepthEvent::Initialized, 0, -1);
    return true;
}

bool DistributedFMMDepthController::set_initial_depth(
    int32_t gpu, int32_t depth, int32_t effective_max_depth, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || frozen_ || !valid_gpu_locked(gpu) ||
        depth < gpu_states_[gpu].effective_min_depth ||
        effective_max_depth < gpu_states_[gpu].effective_min_depth ||
        effective_max_depth > config_.max_depth || depth > effective_max_depth) {
        if (initialized_ && !frozen_) freeze_locked("invalid lazy FMM startup depth", epoch);
        return false;
    }
    DistributedFMMDepthGpuState& state = gpu_states_[gpu];
    if (initial_depth_ready_[gpu] && state.selected_depth != depth) {
        freeze_locked("conflicting lazy FMM startup depth", epoch);
        return false;
    }
    state.selected_depth = depth;
    state.candidate_depth = depth;
    state.effective_max_depth = effective_max_depth;
    initial_depth_ready_[gpu] = true;
    emit_locked(DistributedFMMDepthEvent::Initialized, epoch, gpu);
    return true;
}

bool DistributedFMMDepthController::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_ && config_.enabled;
}

bool DistributedFMMDepthController::frozen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frozen_;
}

std::string DistributedFMMDepthController::failure_reason() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_reason_;
}

int32_t DistributedFMMDepthController::gpu_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_count_;
}

DistributedFMMDepthCommand
DistributedFMMDepthController::command_for_gpu(int32_t gpu) const {
    std::lock_guard<std::mutex> lock(mutex_);
    DistributedFMMDepthCommand command{};
    if (!valid_gpu_locked(gpu) || !initial_depth_ready_[gpu]) return command;
    const DistributedFMMDepthGpuState& state = gpu_states_[gpu];
    command.launch_depth = state.selected_depth;
    command.trial_id = active_trial_id_;
    command.exclude_from_rcb = active_trial_id_ != 0;
    if (!frozen_ && active_trial_id_ != 0 && gpu == active_owner_ &&
        phase_uses_candidate_locked()) {
        command.launch_depth = state.candidate_depth;
    }
    return command;
}

DistributedFMMDepthGpuState
DistributedFMMDepthController::state_for_gpu(int32_t gpu) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return valid_gpu_locked(gpu) ? gpu_states_[gpu] : DistributedFMMDepthGpuState{};
}

bool DistributedFMMDepthController::trial_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_trial_id_ != 0;
}

uint64_t DistributedFMMDepthController::active_trial_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_trial_id_;
}

bool DistributedFMMDepthController::publish_timing(
    int32_t gpu, const DistributedFMMDepthEpochTiming& timing) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !valid_gpu_locked(gpu) || timing.epoch == 0) {
        if (initialized_ && !frozen_) freeze_locked("invalid distributed FMM timing record", timing.epoch);
        return false;
    }
    if (timing.epoch <= last_processed_epoch_ ||
        timing.epoch < discarded_before_epoch_) {
        emit_locked(DistributedFMMDepthEvent::SampleDropped, timing.epoch, gpu);
        return false;
    }
    MailboxSlot& slot = mailbox_[timing.epoch % kMailboxSlots];
    if (slot.epoch != 0 && slot.epoch != timing.epoch) {
        if (slot.epoch < timing.epoch) {
            const uint64_t overwritten = slot.epoch;
            slot.epoch = 0;
            std::fill(slot.present.begin(), slot.present.end(), false);
            emit_locked(DistributedFMMDepthEvent::SampleDropped, overwritten, -1);
            cancel_trial_locked(overwritten,
                                DistributedFMMDepthCancelReason::IncompleteTelemetry);
        } else {
            emit_locked(DistributedFMMDepthEvent::SampleDropped, timing.epoch, gpu);
            return false;
        }
    }
    if (slot.epoch == 0) slot.epoch = timing.epoch;
    if (slot.present[gpu]) {
        freeze_locked("duplicate distributed FMM timing record", timing.epoch);
        return false;
    }
    slot.timings[gpu] = timing;
    slot.present[gpu] = true;
    return true;
}

size_t DistributedFMMDepthController::process_ready_epochs() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || frozen_) return 0;
    size_t processed = 0;
    for (;;) {
        MailboxSlot* ready = nullptr;
        for (MailboxSlot& slot : mailbox_) {
            if (slot.epoch == 0 || slot.epoch <= last_processed_epoch_ ||
                !std::all_of(slot.present.begin(), slot.present.end(),
                             [](bool value) { return value; })) {
                continue;
            }
            if (!ready || slot.epoch < ready->epoch) ready = &slot;
        }
        if (!ready) break;
        const uint64_t epoch = ready->epoch;
        process_epoch_locked(epoch, ready->timings);
        ready->epoch = 0;
        std::fill(ready->present.begin(), ready->present.end(), false);
        last_processed_epoch_ = epoch;
        ++processed;
        if (frozen_) break;
    }
    return processed;
}

void DistributedFMMDepthController::discard_timing_before(uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;
    discarded_before_epoch_ = std::max(discarded_before_epoch_, epoch);
    for (MailboxSlot& slot : mailbox_) {
        if (slot.epoch != 0 && slot.epoch < epoch) {
            emit_locked(DistributedFMMDepthEvent::SampleDropped, slot.epoch, -1);
            slot.epoch = 0;
            std::fill(slot.present.begin(), slot.present.end(), false);
            cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::IncompleteTelemetry);
        }
    }
}

void DistributedFMMDepthController::notify_disruptive_event(
    uint64_t epoch, DistributedFMMDepthCancelReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && !frozen_) cancel_trial_locked(epoch, reason);
}

void DistributedFMMDepthController::set_effective_max_depth(
    int32_t gpu, int32_t effective_max_depth, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int minimum = valid_gpu_locked(gpu) ? gpu_states_[gpu].effective_min_depth : config_.min_depth;
    set_effective_depth_range_locked(gpu, minimum, effective_max_depth, epoch);
}

void DistributedFMMDepthController::set_effective_depth_range(
    int32_t gpu, int32_t minimum, int32_t maximum, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_effective_depth_range_locked(gpu, minimum, maximum, epoch);
}

void DistributedFMMDepthController::set_effective_depth_range_locked(
    int32_t gpu, int32_t minimum, int32_t maximum, uint64_t epoch) {
    if (!initialized_ || frozen_ || !valid_gpu_locked(gpu) || minimum < config_.min_depth ||
        maximum > config_.max_depth || minimum > maximum || epoch == UINT64_MAX) {
        if (initialized_ && !frozen_)
            freeze_locked("invalid distributed FMM effective depth range", epoch);
        return;
    }
    auto& state = gpu_states_[gpu];
    if (minimum == state.effective_min_depth && maximum == state.effective_max_depth) return;
    const bool minimum_changed = minimum != state.effective_min_depth;
    cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::CapacityService);
    state.effective_min_depth = minimum;
    state.effective_max_depth = maximum;
    state.selected_depth = std::max(minimum, std::min(state.selected_depth, maximum));
    state.candidate_depth = state.selected_depth;
    for (auto& peer : gpu_states_) {
        peer.confirmation_count = 0;
        peer.selected_evaluations_since_trial = 0;
    }
    // Hard feasibility changes invalidate optimization history, including
    // asynchronous records that were already queued before this boundary.
    discarded_before_epoch_ = std::max(discarded_before_epoch_, epoch+1);
    for (auto& slot : mailbox_) {
        if (slot.epoch && slot.epoch < UINT64_MAX)
            discarded_before_epoch_ = std::max(discarded_before_epoch_, slot.epoch+1);
        slot.epoch = 0;
        std::fill(slot.present.begin(), slot.present.end(), false);
    }
    emit_locked(minimum_changed ? DistributedFMMDepthEvent::EffectiveBoundsChanged :
                DistributedFMMDepthEvent::EffectiveCeilingChanged, epoch, gpu);
}

void DistributedFMMDepthController::freeze(const std::string& reason, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    freeze_locked(reason, epoch);
}

std::vector<DistributedFMMDepthDiagnostic>
DistributedFMMDepthController::diagnostics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return diagnostics_;
}

uint64_t DistributedFMMDepthController::dropped_diagnostics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_diagnostics_;
}

bool DistributedFMMDepthController::valid_gpu_locked(int32_t gpu) const {
    return gpu >= 0 && gpu < gpu_count_;
}

void DistributedFMMDepthController::process_epoch_locked(
    uint64_t epoch, const std::vector<DistributedFMMDepthEpochTiming>& timings) {
    if (timings.size() != static_cast<size_t>(gpu_count_)) {
        freeze_locked("incomplete distributed FMM timing vector", epoch);
        return;
    }
    if (!all_ready(initial_depth_ready_)) return;
    if (std::any_of(timings.begin(), timings.end(),
                    [](const DistributedFMMDepthEpochTiming& t) { return t.rcb_accepted; })) {
        cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::Rebalance);
        return;
    }
    if (std::any_of(timings.begin(), timings.end(),
                    [](const DistributedFMMDepthEpochTiming& t) { return t.capacity_event; })) {
        cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::CapacityService);
        return;
    }
    if (std::any_of(timings.begin(), timings.end(),
                    [](const DistributedFMMDepthEpochTiming& t) { return t.yorp_transition; })) {
        cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::YorpTransition);
        return;
    }
    if (std::any_of(timings.begin(), timings.end(),
                    [](const DistributedFMMDepthEpochTiming& t) { return t.local_count_changed; })) {
        cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::LocalCountChange);
        return;
    }
    // Do not let an asynchronously completed record be interpreted against a
    // newer controller state.  A dropped record is an optimization miss, not
    // a physics failure, so the current selected depths remain live.
    const bool command_matches = active_trial_id_ == 0
        ? timing_matches_selected_commands_locked(timings)
        : timing_matches_active_phase_locked(timings);
    if (!command_matches) {
        emit_locked(DistributedFMMDepthEvent::SampleDropped, epoch, active_owner_);
        return;
    }
    for (const auto& timing : timings) {
        if (!timing.telemetry_valid || timing.elapsed_ns == 0) {
            freeze_locked("invalid distributed FMM telemetry", epoch);
            return;
        }
    }
    if (counts_changed_locked(timings)) {
        for (int32_t gpu = 0; gpu < gpu_count_; ++gpu) {
            gpu_states_[gpu].stable_local_owned_particles = timings[gpu].local_owned_particles;
            stable_count_valid_[gpu] = true;
        }
        cancel_trial_locked(epoch, DistributedFMMDepthCancelReason::LocalCountChange);
        return;
    }
    for (int32_t gpu = 0; gpu < gpu_count_; ++gpu) {
        gpu_states_[gpu].stable_local_owned_particles = timings[gpu].local_owned_particles;
        stable_count_valid_[gpu] = true;
    }

    if (active_trial_id_ == 0) {
        for (DistributedFMMDepthGpuState& state : gpu_states_)
            ++state.selected_evaluations_since_trial;
        if (gpu_states_[next_owner_].selected_evaluations_since_trial >=
            config_.trial_interval_evaluations) {
            begin_next_trial_locked(epoch);
        }
        return;
    }

    const Aggregate aggregate = aggregate_locked(timings);
    if (warmups_remaining_ > 0) {
        --warmups_remaining_;
        return;
    }
    if (phase_uses_candidate_locked()) {
        total_b_.max_ms += aggregate.max_ms;
        total_b_.mean_ms += aggregate.mean_ms;
        total_b_.let_bytes += aggregate.let_bytes;
        ++samples_b_;
    } else {
        total_a_.max_ms += aggregate.max_ms;
        total_a_.mean_ms += aggregate.mean_ms;
        total_a_.let_bytes += aggregate.let_bytes;
        ++samples_a_;
    }
    ++phase_samples_;
    emit_locked(DistributedFMMDepthEvent::TrialSample, epoch, active_owner_, &aggregate);
    // Every A/B/B/A phase owns a full, independently sized sample block.
    // This preserves the time-symmetric order even when trial_samples is one.
    if (phase_samples_ < static_cast<uint64_t>(config_.trial_samples)) return;
    if (phase_ == TrialPhase::SelectedA2) {
        finish_trial_locked(epoch);
        return;
    }
    advance_phase_locked();
}

void DistributedFMMDepthController::begin_next_trial_locked(uint64_t epoch) {
    if (frozen_ || !all_ready(initial_depth_ready_)) return;
    const int32_t owner = next_owner_;
    next_owner_ = (next_owner_ + 1) % gpu_count_;
    DistributedFMMDepthGpuState& state = gpu_states_[owner];
    state.candidate_depth = candidate_for_locked(owner);
    if (state.candidate_depth == state.selected_depth) {
        state.search_direction = -state.search_direction;
        state.candidate_depth = candidate_for_locked(owner);
    }
    if (state.candidate_depth == state.selected_depth) {
        state.selected_evaluations_since_trial = 0;
        return;
    }
    active_owner_ = owner;
    active_trial_id_ = next_trial_id_++;
    phase_ = TrialPhase::SelectedA;
    warmups_remaining_ = 0;
    phase_samples_ = 0;
    samples_a_ = samples_b_ = 0;
    total_a_ = Aggregate{};
    total_b_ = Aggregate{};
    state.selected_evaluations_since_trial = 0;
    emit_locked(DistributedFMMDepthEvent::TrialStarted, epoch, owner);
}

void DistributedFMMDepthController::cancel_trial_locked(
    uint64_t epoch, DistributedFMMDepthCancelReason reason) {
    if (active_trial_id_ == 0) return;
    DistributedFMMDepthGpuState& state = gpu_states_[active_owner_];
    state.candidate_depth = state.selected_depth;
    state.confirmation_count = 0;
    state.selected_evaluations_since_trial = 0;
    emit_locked(DistributedFMMDepthEvent::TrialCancelled, epoch, active_owner_, nullptr, 0.0, reason);
    active_trial_id_ = 0;
    active_owner_ = -1;
    warmups_remaining_ = 0;
    phase_samples_ = 0;
    samples_a_ = samples_b_ = 0;
    total_a_ = Aggregate{};
    total_b_ = Aggregate{};
}

void DistributedFMMDepthController::finish_trial_locked(uint64_t epoch) {
    const int32_t owner = active_owner_;
    DistributedFMMDepthGpuState& state = gpu_states_[owner];
    Aggregate baseline = total_a_;
    Aggregate candidate = total_b_;
    baseline.max_ms /= static_cast<double>(samples_a_);
    baseline.mean_ms /= static_cast<double>(samples_a_);
    baseline.let_bytes /= samples_a_;
    candidate.max_ms /= static_cast<double>(samples_b_);
    candidate.mean_ms /= static_cast<double>(samples_b_);
    candidate.let_bytes /= samples_b_;
    const double primary = baseline.max_ms > 0.0
        ? (baseline.max_ms - candidate.max_ms) / baseline.max_ms : 0.0;
    const double secondary = baseline.mean_ms > 0.0
        ? (baseline.mean_ms - candidate.mean_ms) / baseline.mean_ms : 0.0;
    const double threshold = config_.min_improvement_fraction;
    bool candidate_wins = false;
    if (std::fabs(primary) >= threshold) {
        candidate_wins = primary > 0.0;
    } else if (std::fabs(secondary) >= threshold) {
        candidate_wins = secondary > 0.0;
    } else if (candidate.let_bytes != baseline.let_bytes) {
        candidate_wins = candidate.let_bytes < baseline.let_bytes;
    }
    if (candidate_wins) {
        ++state.confirmation_count;
        emit_locked(DistributedFMMDepthEvent::TrialConfirmed, epoch, owner,
                    &candidate, primary);
        if (state.confirmation_count >= config_.confirmation_trials) {
            const int32_t previous_selected = state.selected_depth;
            state.selected_depth = state.candidate_depth;
            state.candidate_depth = state.selected_depth;
            state.confirmation_count = 0;
            state.search_direction = state.selected_depth > previous_selected ? 1 : -1;
            emit_locked(DistributedFMMDepthEvent::DepthAdopted, epoch, owner,
                        &candidate, primary);
        }
    } else {
        state.confirmation_count = 0;
        state.search_direction = -state.search_direction;
        state.candidate_depth = state.selected_depth;
        emit_locked(DistributedFMMDepthEvent::CandidateRejected, epoch, owner,
                    &candidate, primary);
    }
    state.selected_evaluations_since_trial = 0;
    active_trial_id_ = 0;
    active_owner_ = -1;
    samples_a_ = samples_b_ = 0;
    phase_samples_ = 0;
    total_a_ = Aggregate{};
    total_b_ = Aggregate{};
}

void DistributedFMMDepthController::freeze_locked(const std::string& reason, uint64_t epoch) {
    if (frozen_) return;
    frozen_ = true;
    failure_reason_ = reason;
    if (active_owner_ >= 0) gpu_states_[active_owner_].candidate_depth =
        gpu_states_[active_owner_].selected_depth;
    active_trial_id_ = 0;
    active_owner_ = -1;
    emit_locked(DistributedFMMDepthEvent::Frozen, epoch, -1);
}

void DistributedFMMDepthController::emit_locked(
    DistributedFMMDepthEvent event, uint64_t epoch, int32_t gpu,
    const Aggregate* aggregate, double improvement,
    DistributedFMMDepthCancelReason reason) {
    DistributedFMMDepthDiagnostic record{};
    record.sequence = next_sequence_++;
    record.epoch = epoch;
    record.trial_id = active_trial_id_;
    record.gpu_id = gpu;
    if (valid_gpu_locked(gpu)) {
        record.selected_depth = gpu_states_[gpu].selected_depth;
        record.candidate_depth = gpu_states_[gpu].candidate_depth;
    }
    if (aggregate) {
        record.global_max_ms = aggregate->max_ms;
        record.global_mean_ms = aggregate->mean_ms;
        record.let_bytes = aggregate->let_bytes;
    }
    record.primary_relative_improvement = improvement;
    record.event = event;
    record.cancel_reason = reason;
    if (diagnostics_.size() == kDiagnosticCapacity) {
        diagnostics_.erase(diagnostics_.begin());
        ++dropped_diagnostics_;
    }
    diagnostics_.push_back(record);
}

DistributedFMMDepthController::Aggregate
DistributedFMMDepthController::aggregate_locked(
    const std::vector<DistributedFMMDepthEpochTiming>& timings) const {
    Aggregate result{};
    for (const auto& timing : timings) {
        const double ms = static_cast<double>(timing.elapsed_ns) / 1.0e6;
        result.max_ms = std::max(result.max_ms, ms);
        result.mean_ms += timing.aggregate_mean_ns ? static_cast<double>(timing.aggregate_mean_ns)/1.0e6 : ms;
        result.let_bytes += timing.let_bytes;
    }
    result.mean_ms /= static_cast<double>(timings.size());
    return result;
}

bool DistributedFMMDepthController::counts_changed_locked(
    const std::vector<DistributedFMMDepthEpochTiming>& timings) const {
    for (int32_t gpu = 0; gpu < gpu_count_; ++gpu) {
        if (!stable_count_valid_[gpu]) continue;
        const double prior = std::fabs(static_cast<double>(
            gpu_states_[gpu].stable_local_owned_particles));
        const double delta = std::fabs(static_cast<double>(timings[gpu].local_owned_particles) -
                                       static_cast<double>(gpu_states_[gpu].stable_local_owned_particles));
        const double denominator = std::max(1.0, prior);
        if (delta / denominator > config_.multigpu.local_count_reset_fraction) return true;
    }
    return false;
}

bool DistributedFMMDepthController::timing_matches_active_phase_locked(
    const std::vector<DistributedFMMDepthEpochTiming>& timings) const {
    if (active_trial_id_ == 0 || !valid_gpu_locked(active_owner_)) return false;
    const DistributedFMMDepthGpuState& owner = gpu_states_[active_owner_];
    const int32_t expected_depth = phase_uses_candidate_locked()
        ? owner.candidate_depth : owner.selected_depth;
    const DistributedFMMDepthEpochTiming& timing = timings[active_owner_];
    return timing.command_trial_id == active_trial_id_ &&
           timing.launch_depth == expected_depth;
}

bool DistributedFMMDepthController::timing_matches_selected_commands_locked(
    const std::vector<DistributedFMMDepthEpochTiming>& timings) const {
    // A completed command from a just-finished trial must not be counted as a
    // selected-depth interval of the next trial.  Check every GPU here because
    // there is no active owner to identify the stale one.
    for (int32_t gpu = 0; gpu < gpu_count_; ++gpu) {
        if (timings[gpu].command_trial_id != 0 ||
            timings[gpu].launch_depth != gpu_states_[gpu].selected_depth) {
            return false;
        }
    }
    return true;
}

bool DistributedFMMDepthController::phase_uses_candidate_locked() const {
    return phase_ == TrialPhase::CandidateB || phase_ == TrialPhase::CandidateB2;
}

void DistributedFMMDepthController::advance_phase_locked() {
    const bool old_candidate = phase_uses_candidate_locked();
    switch (phase_) {
        case TrialPhase::SelectedA: phase_ = TrialPhase::CandidateB; break;
        case TrialPhase::CandidateB: phase_ = TrialPhase::CandidateB2; break;
        case TrialPhase::CandidateB2: phase_ = TrialPhase::SelectedA2; break;
        case TrialPhase::SelectedA2: phase_ = TrialPhase::SelectedA; break;
    }
    if (old_candidate != phase_uses_candidate_locked())
        warmups_remaining_ = config_.trial_warmup_evaluations;
    phase_samples_ = 0;
}

int32_t DistributedFMMDepthController::candidate_for_locked(int32_t gpu) const {
    const DistributedFMMDepthGpuState& state = gpu_states_[gpu];
    const int32_t raw = state.selected_depth + (state.search_direction >= 0 ? 1 : -1);
    return std::max(state.effective_min_depth, std::min(raw, state.effective_max_depth));
}


namespace {
template<class T> void depth_write(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("Cannot write distributed depth checkpoint");
}
template<class T> void depth_read(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable<T>::value);
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("Truncated distributed depth checkpoint");
}
}
void DistributedFMMDepthController::save_checkpoint(std::ostream& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || frozen_) throw std::runtime_error("Cannot checkpoint an inactive/frozen depth controller");
    depth_write(out, uint32_t{0x32434446}); // FDC2: bounds-aware controller state
    depth_write(out, uint32_t{2});
    depth_write(out, gpu_count_);
    depth_write(out, initialized_);
    depth_write(out, frozen_);
    depth_write(out, last_processed_epoch_);
    depth_write(out, discarded_before_epoch_);
    depth_write(out, next_trial_id_);
    depth_write(out, active_trial_id_);
    depth_write(out, active_owner_);
    depth_write(out, next_owner_);
    depth_write(out, phase_);
    depth_write(out, warmups_remaining_);
    depth_write(out, phase_samples_);
    depth_write(out, samples_a_);
    depth_write(out, samples_b_);
    depth_write(out, total_a_);
    depth_write(out, total_b_);
    depth_write(out, next_sequence_);
    depth_write(out, dropped_diagnostics_);

    for (int g=0; g<gpu_count_; ++g) {
        depth_write(out, gpu_states_[g]);
        depth_write(out, uint8_t(initial_depth_ready_[g]));
        depth_write(out, uint8_t(stable_count_valid_[g]));
    }
    for (const auto& slot : mailbox_) {
        depth_write(out, slot.epoch);
        for (int g=0; g<gpu_count_; ++g) {
            depth_write(out, slot.timings[g]); depth_write(out, uint8_t(slot.present[g]));
        }
    }
    depth_write(out, uint32_t(diagnostics_.size()));
    for (const auto& diagnostic : diagnostics_) depth_write(out, diagnostic);
}
void DistributedFMMDepthController::restore_checkpoint(std::istream& in, const AdaptiveDepthConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    uint32_t first = 0; depth_read(in, first);
    const bool legacy = first != uint32_t{0x32434446};
    if (legacy) gpu_count_ = static_cast<int32_t>(first);
    else {
        uint32_t version = 0; depth_read(in, version);
        if (version != 2) throw std::runtime_error("Unsupported distributed depth controller checkpoint version");
        depth_read(in, gpu_count_);
    }
    depth_read(in, initialized_);
    depth_read(in, frozen_);
    depth_read(in, last_processed_epoch_);
    depth_read(in, discarded_before_epoch_);
    depth_read(in, next_trial_id_);
    depth_read(in, active_trial_id_);
    depth_read(in, active_owner_);
    depth_read(in, next_owner_);
    depth_read(in, phase_);
    depth_read(in, warmups_remaining_);
    depth_read(in, phase_samples_);
    depth_read(in, samples_a_);
    depth_read(in, samples_b_);
    depth_read(in, total_a_);
    depth_read(in, total_b_);
    depth_read(in, next_sequence_);
    depth_read(in, dropped_diagnostics_);

    if (!config.enabled || !initialized_ || frozen_ || gpu_count_<1 || gpu_count_>kMaxGpus ||
        active_owner_ < -1 || active_owner_ >= gpu_count_ || next_owner_<0 || next_owner_>=gpu_count_ ||
        static_cast<unsigned>(phase_)>3 || warmups_remaining_<0 || !next_sequence_)
        throw std::runtime_error("Invalid distributed depth controller checkpoint");
    gpu_states_.resize(gpu_count_); initial_depth_ready_.resize(gpu_count_); stable_count_valid_.resize(gpu_count_);
    for (int g=0; g<gpu_count_; ++g) {
        if (legacy) {
            struct LegacyGpuState {
                int32_t selected_depth, candidate_depth, effective_max_depth;
                int32_t search_direction, confirmation_count, selected_evaluations_since_trial;
                int64_t stable_local_owned_particles;
            } old{};
            depth_read(in, old);
            auto& state = gpu_states_[g];
            state.selected_depth=old.selected_depth;state.candidate_depth=old.candidate_depth;
            state.effective_max_depth=old.effective_max_depth;state.effective_min_depth=config.min_depth;
            state.search_direction=old.search_direction;state.confirmation_count=old.confirmation_count;
            state.selected_evaluations_since_trial=old.selected_evaluations_since_trial;
            state.stable_local_owned_particles=old.stable_local_owned_particles;
        } else depth_read(in, gpu_states_[g]);
        uint8_t ready, valid;
        depth_read(in, ready); depth_read(in, valid);
        initial_depth_ready_[g]=ready; stable_count_valid_[g]=valid;
        const auto& state = gpu_states_[g];
        if (ready>1 || valid>1 || state.selected_depth<config.min_depth ||
            state.selected_depth>config.max_depth || state.effective_min_depth<config.min_depth ||
            state.effective_min_depth>state.effective_max_depth || state.effective_max_depth>config.max_depth ||
            state.selected_depth<state.effective_min_depth || state.selected_depth>state.effective_max_depth ||
            state.candidate_depth<state.effective_min_depth || state.candidate_depth>state.effective_max_depth)
            throw std::runtime_error("Invalid checkpoint depth range");
    }
    mailbox_.resize(kMailboxSlots);
    for (auto& slot : mailbox_) {
        depth_read(in, slot.epoch); slot.timings.resize(gpu_count_); slot.present.resize(gpu_count_);
        for (int g=0; g<gpu_count_; ++g) {
            depth_read(in, slot.timings[g]); uint8_t present; depth_read(in, present);
            if (present>1) throw std::runtime_error("Invalid checkpoint mailbox");
            slot.present[g]=present;
        }
    }
    uint32_t count; depth_read(in, count);
    if (count>kDiagnosticCapacity) throw std::runtime_error("Invalid checkpoint diagnostic count");
    diagnostics_.resize(count);
    for (auto& diagnostic : diagnostics_) depth_read(in, diagnostic);
    failure_reason_.clear();
}
}  // namespace cudadem

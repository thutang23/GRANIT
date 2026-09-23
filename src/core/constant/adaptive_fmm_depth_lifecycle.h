#pragma once
#include "load_or_resume.h"
#include "../acceleration/acceleration.h"
#include <fstream>
#include <iomanip>
#include <filesystem>
namespace cudadem {
class AdaptiveFMMDepthLifecycle {
    const SimConfig& config_;
    SimulationState& state_;
    cudaStream_t& compute_stream_;
    uint64_t adaptive_fmm_depth_persisted_sequence_ = 0;
public:
    AdaptiveFMMDepthLifecycle(const SimConfig& c, SimulationState& s, cudaStream_t& stream)
        : config_(c), state_(s), compute_stream_(stream) {}
    static uint64_t adaptive_fmm_depth_config_fingerprint(const SimConfig& config) {
        // Bind every knob that changes the controller or its timing population.
        // or the depth-dependent FMM work being compared.  The particle
        // checkpoint independently validates material/clump identity.
        uint64_t h = UINT64_C(1469598103934665603);
        const auto mix = [&h](const void* data, size_t bytes) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < bytes; ++i) {
                h ^= p[i];
                h *= UINT64_C(1099511628211);
            }
        };
        const AccelerationConfig& acceleration = config.acceleration;
        const UnifiedFMMConfig& fmm = acceleration.unified_fmm;
        const AdaptiveDepthConfig& adaptive = fmm.adaptive_depth;
        mix(&acceleration.enable_self_gravity, sizeof(acceleration.enable_self_gravity));
        mix(&acceleration.gravity_method, sizeof(acceleration.gravity_method));
        mix(&acceleration.gravity_update_rate, sizeof(acceleration.gravity_update_rate));
        mix(&acceleration.gravity_macro_dt, sizeof(acceleration.gravity_macro_dt));
        mix(&fmm.tree_dimension, sizeof(fmm.tree_dimension));
        mix(&fmm.tree_depth, sizeof(fmm.tree_depth));
        mix(&fmm.theta_far, sizeof(fmm.theta_far));
        mix(&fmm.softening_factor, sizeof(fmm.softening_factor));
        mix(&fmm.gravitational_constant, sizeof(fmm.gravitational_constant));
        mix(&fmm.direct_threshold_leaf, sizeof(fmm.direct_threshold_leaf));
        mix(&fmm.direct_threshold_internal, sizeof(fmm.direct_threshold_internal));
        mix(&adaptive.enabled, sizeof(adaptive.enabled));
        mix(&adaptive.initial_depth_auto, sizeof(adaptive.initial_depth_auto));
        mix(&adaptive.min_depth, sizeof(adaptive.min_depth));
        mix(&adaptive.max_depth, sizeof(adaptive.max_depth));
        mix(&adaptive.trial_interval_evaluations,
            sizeof(adaptive.trial_interval_evaluations));
        mix(&adaptive.trial_warmup_evaluations,
            sizeof(adaptive.trial_warmup_evaluations));
        mix(&adaptive.trial_samples, sizeof(adaptive.trial_samples));
        mix(&adaptive.min_improvement_fraction,
            sizeof(adaptive.min_improvement_fraction));
        mix(&adaptive.confirmation_trials, sizeof(adaptive.confirmation_trials));
        mix(&adaptive.capacity_headroom, sizeof(adaptive.capacity_headroom));
        mix(&adaptive.capacity_growth_trigger,
            sizeof(adaptive.capacity_growth_trigger));
        mix(&config.physics, sizeof(config.physics));
        if (config.physics == SimulationPhysics::SPH) {
            mix(&config.sph.hsml_min, sizeof(config.sph.hsml_min));
            mix(&config.sph.hsml_max, sizeof(config.sph.hsml_max));
            mix(&config.sph.cfl_number, sizeof(config.sph.cfl_number));
            mix(&config.sph.dt_min, sizeof(config.sph.dt_min));
            mix(&config.sph.dt_max, sizeof(config.sph.dt_max));
        }
        return h;
    }


    std::string adaptive_fmm_depth_csv_path() const {
        return std::string(config_.io.output_directory) + "/adaptive_fmm_depth.csv";
    }

    static const char* adaptive_fmm_depth_csv_header() {
        return "sequence,evaluation_counter,evaluation_time_s,tree_dimension,selected_depth,"
               "candidate_depth,search_direction,trial_phase,elapsed_ms,relative_improvement,"
               "node_count,node_capacity,effective_max_depth,transition_reason";
    }

    void reset_adaptive_fmm_depth_csv_for_fresh_start() {
        if (!adaptive_fmm_depth_checkpoint_requested(config_)) return;
        adaptive_fmm_depth_persisted_sequence_ = 0;
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = adaptive_fmm_depth_csv_path();
        std::filesystem::remove(path + ".start");
        if (!std::filesystem::exists(path)) return;
        std::ofstream file(path, std::ios::trunc);
        if (!file) {
            throw std::runtime_error(
                "Cannot reset stale adaptive FMM depth diagnostic: " + path);
        }
    }

    void rewind_adaptive_fmm_depth_csv_for_resume(uint64_t checkpoint_sequence) {
        const std::string path = adaptive_fmm_depth_csv_path();
        if (!std::filesystem::exists(path)) return;
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "Cannot read adaptive FMM depth diagnostic for resume: " + path);
        }
        std::string header;
        if (!std::getline(input, header) || header != adaptive_fmm_depth_csv_header()) {
            throw std::runtime_error(
                "Adaptive FMM depth diagnostic header is invalid on resume: " + path);
        }
        std::vector<std::string> retained;
        uint64_t expected_sequence = 1;
        // A continuation may start a new output directory. Its diagnostic
        // segment begins after the source checkpoint's watermark, not at 1.
        // Keep an explicit origin so missing rows still fail on later resumes.
        if (std::filesystem::exists(path + ".start")) {
            std::ifstream origin(path + ".start");
            origin >> expected_sequence;
            if (!origin || expected_sequence == 0 ||
                expected_sequence - 1 > checkpoint_sequence)
                throw std::runtime_error("Invalid adaptive FMM diagnostic segment origin: " + path);
            origin >> std::ws;
            if (!origin.eof())
                throw std::runtime_error("Invalid adaptive FMM diagnostic segment origin: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const size_t comma = line.find(',');
            if (comma == std::string::npos) {
                throw std::runtime_error(
                    "Adaptive FMM depth diagnostic row is invalid on resume: " + path);
            }
            size_t parsed = 0;
            uint64_t sequence = 0;
            try {
                sequence = std::stoull(line.substr(0, comma), &parsed);
            } catch (...) {
                throw std::runtime_error(
                    "Adaptive FMM depth diagnostic sequence is invalid on resume: " + path);
            }
            if (parsed != comma) {
                throw std::runtime_error(
                    "Adaptive FMM depth diagnostic sequence is invalid on resume: " + path);
            }
            if (sequence <= checkpoint_sequence) {
                if (sequence != expected_sequence++) {
                    throw std::runtime_error(
                        "Adaptive FMM depth diagnostic sequence gap is fatal on resume: " + path);
                }
                retained.push_back(line);
            }
        }
        if (checkpoint_sequence != 0 && expected_sequence - 1 != checkpoint_sequence) {
            throw std::runtime_error(
                "Adaptive FMM depth checkpoint CSV watermark is unavailable on resume: " + path);
        }
        input.close();
        const std::string temporary = path + ".resume_tmp";
        {
            std::ofstream output(temporary, std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "Cannot rewrite adaptive FMM depth diagnostic on resume: " + temporary);
            }
            output << header << '\n';
            for (const std::string& row : retained) output << row << '\n';
            if (!output) {
                throw std::runtime_error(
                    "Failed rewriting adaptive FMM depth diagnostic on resume: " + temporary);
            }
        }
        std::filesystem::rename(temporary, path);
    }

    static const char* adaptive_fmm_depth_transition_name(
        AdaptiveFMMDepthTransition transition) {
        switch (transition) {
        case AdaptiveFMMDepthTransition::NONE: return "none";
        case AdaptiveFMMDepthTransition::INITIAL_SELECTION: return "initial_selection";
        case AdaptiveFMMDepthTransition::TRIAL_RESULT: return "trial_result";
        case AdaptiveFMMDepthTransition::FASTER_NEIGHBOR: return "faster_neighbor";
        case AdaptiveFMMDepthTransition::CONFIRMATION: return "confirmation";
        case AdaptiveFMMDepthTransition::CANDIDATE_REJECTED: return "candidate_rejected";
        case AdaptiveFMMDepthTransition::CAPACITY_FALLBACK: return "capacity_fallback";
        case AdaptiveFMMDepthTransition::CAPACITY_GROWTH: return "capacity_growth";
        case AdaptiveFMMDepthTransition::EFFECTIVE_CEILING_CHANGED:
            return "effective_ceiling_changed";
        }
        return "unknown";
    }

    void append_adaptive_fmm_depth_diagnostics() {
        if (!adaptive_fmm_depth_checkpoint_requested(config_)) return;
        std::vector<AdaptiveFMMDepthDiagnostic> diagnostics;
        uint64_t dropped = 0;
        AccelerationManager::consume_adaptive_fmm_depth_diagnostics(
            diagnostics, &dropped, compute_stream_);
        // This function is called only at output/checkpoint/final boundaries.
        // Service a sticky high-water/fallback request here, after draining
        // the old ring and while a host synchronization is already expected.
        if (AccelerationManager::service_adaptive_fmm_capacity(
                state_, config_, compute_stream_)) {
            uint64_t after_growth_dropped = dropped;
            AccelerationManager::consume_adaptive_fmm_depth_diagnostics(
                diagnostics, &after_growth_dropped, compute_stream_);
            dropped = std::max(dropped, after_growth_dropped);
        }
        if (dropped != 0) {
            throw std::runtime_error(
                "Adaptive FMM depth diagnostic loss is fatal; cannot continue with a "
                "non-lossless controller record");
        }
        if (diagnostics.empty()) return;

        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = adaptive_fmm_depth_csv_path();
        const bool write_header = !std::filesystem::exists(path) ||
            std::filesystem::file_size(path) == 0;
        std::ofstream file(path, std::ios::app);
        if (!file) {
            throw std::runtime_error(
                "Cannot append adaptive FMM depth diagnostic: " + path);
        }
        if (write_header) {
            const std::string origin = path + ".start";
            const std::string temporary = origin + ".tmp";
            std::ofstream start(temporary, std::ios::trunc);
            start << adaptive_fmm_depth_persisted_sequence_ + 1 << '\n';
            start.close();
            if (!start) throw std::runtime_error("Cannot write adaptive FMM diagnostic segment origin: " + path);
            std::filesystem::rename(temporary, origin);
            file << adaptive_fmm_depth_csv_header() << '\n';
        }
        file << std::setprecision(17);
        for (const AdaptiveFMMDepthDiagnostic& diagnostic : diagnostics) {
            if (diagnostic.sequence != adaptive_fmm_depth_persisted_sequence_ + 1) {
                throw std::runtime_error(
                    "Adaptive FMM depth diagnostic sequence gap is fatal");
            }
            file << diagnostic.sequence << ','
                 << diagnostic.evaluation_counter << ','
                 << diagnostic.evaluation_time << ','
                 << diagnostic.tree_dimension << ','
                 << diagnostic.selected_depth << ','
                 << diagnostic.candidate_depth << ','
                 << diagnostic.search_direction << ','
                 << diagnostic.trial_phase << ','
                 << diagnostic.elapsed_ms << ','
                 << diagnostic.relative_improvement << ','
                 << diagnostic.node_count << ','
                 << diagnostic.node_capacity << ','
                 << diagnostic.effective_max_depth << ','
                 << adaptive_fmm_depth_transition_name(
                        diagnostic.transition_reason) << '\n';
            adaptive_fmm_depth_persisted_sequence_ = diagnostic.sequence;
        }
        file.flush();
        if (!file) {
            throw std::runtime_error(
                "Failed writing adaptive FMM depth diagnostic: " + path);
        }
    }


    void save_adaptive_fmm_depth_checkpoint_sidecar(
        const std::string& checkpoint_path) {
        if (!adaptive_fmm_depth_checkpoint_requested(config_)) return;
        AdaptiveFMMDepthCheckpoint checkpoint{};
        if (!AccelerationManager::get_adaptive_fmm_depth_checkpoint(
                &checkpoint, compute_stream_)) {
            throw std::runtime_error(
                "Adaptive FMM depth checkpoint state is unavailable");
        }
        checkpoint.persisted_csv_sequence = adaptive_fmm_depth_persisted_sequence_;
        if (checkpoint.producer_sequence != checkpoint.persisted_csv_sequence) {
            throw std::runtime_error(
                "Adaptive FMM depth checkpoint diagnostics were not drained losslessly");
        }
        struct Header {
            char magic[8];
            uint32_t version;
            uint32_t payload_size;
            int64_t step;
            double time;
            uint64_t config_hash;
        };
        const Header header{
            {'A','F','M','M','D','P','T','H'}, 2,
            static_cast<uint32_t>(sizeof(checkpoint)),
            state_.current_step, state_.current_time,
            adaptive_fmm_depth_config_fingerprint(config_)};
        const std::string path =
            adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path);
        const std::string temporary = path + ".tmp";
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            if (!file) {
                throw std::runtime_error(
                    "Cannot write adaptive FMM depth checkpoint sidecar: " + temporary);
            }
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.write(reinterpret_cast<const char*>(&checkpoint), sizeof(checkpoint));
            file.flush();
            if (!file) {
                throw std::runtime_error(
                    "Failed writing adaptive FMM depth checkpoint sidecar: " + temporary);
            }
        }
        std::error_code ec;
        std::filesystem::rename(temporary, path, ec);
        if (ec) {
            std::filesystem::remove(temporary);
            throw std::runtime_error(
                "Cannot publish adaptive FMM depth checkpoint sidecar: " +
                path + ": " + ec.message());
        }
    }

    void restore_adaptive_fmm_depth_checkpoint_sidecar(
        const std::string& checkpoint_path) {
        if (!adaptive_fmm_depth_checkpoint_requested(config_)) return;
        struct Header {
            char magic[8];
            uint32_t version;
            uint32_t payload_size;
            int64_t step;
            double time;
            uint64_t config_hash;
        };
        Header header{};
        AdaptiveFMMDepthCheckpoint checkpoint{};
        const std::string path =
            adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path);
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(
                "Adaptive FMM depth resume requires checkpoint sidecar: " + path);
        }
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        file.read(reinterpret_cast<char*>(&checkpoint), sizeof(checkpoint));
        const bool payload_read = static_cast<bool>(file);
        const char expected_magic[8] = {'A','F','M','M','D','P','T','H'};
        const bool has_trailing_bytes = file.peek() != std::char_traits<char>::eof();
        if (!payload_read || has_trailing_bytes ||
            std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
            header.version != 2 || header.payload_size != sizeof(checkpoint) ||
            checkpoint.version != 2 || header.step != state_.current_step ||
            header.time != state_.current_time ||
            header.config_hash != adaptive_fmm_depth_config_fingerprint(config_)) {
            throw std::runtime_error(
                "Adaptive FMM depth checkpoint sidecar is invalid, orphaned, or mismatched "
                "(version 1 sidecars cannot be resumed): " +
                path);
        }
        if (!AccelerationManager::restore_adaptive_fmm_depth_checkpoint(
                checkpoint, compute_stream_)) {
            throw std::runtime_error(
                "Unsupported adaptive FMM depth checkpoint sidecar: " + path);
        }
        adaptive_fmm_depth_persisted_sequence_ = checkpoint.persisted_csv_sequence;
        rewind_adaptive_fmm_depth_csv_for_resume(checkpoint.persisted_csv_sequence);
    }
};
}

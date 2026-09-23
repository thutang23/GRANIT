#ifndef CUDADEM_RCB_WORK_MODEL_H
#define CUDADEM_RCB_WORK_MODEL_H

// Pure-host adaptive cost calibration used by the multi-GPU RCB policy.  This
// deliberately represents active work only: phase-coordinator stalls are a
// trigger/telemetry concern and must never become a spatial work coefficient.
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cudadem {

enum class RCBCostModel : uint8_t { Adaptive, Legacy };
enum class RCBWorkPhase : uint8_t {
    Integration, Contacts, FmmNode, FmmLeaf, FmmParticle, LetSource, LetReceiver,
    HaloLifecycle, ContactPair, LinkTransfer, Count
};
// Every gravity evaluation rebuilds; non-FMM phases use NotApplicable.
enum class RCBFmmMode : uint8_t { NotApplicable, Rebuild, Count };
enum class RCBLinkClass : uint8_t { PCIe, NVLink };

// These fields mirror multigpu YAML options and are kept host-only.  The
// legacy scalar coefficients remain meaningful only when cost_model is Legacy.
struct RCBWorkModelConfig {
    RCBCostModel cost_model = RCBCostModel::Adaptive;
    int32_t calibration_sample_interval = 8;
    float calibration_half_life = 8.0f;
    float min_predicted_improvement = 0.05f;
    bool calibration_verbose = false;
};

const char* to_string(RCBCostModel value);
bool parse_rcb_cost_model(const std::string& value, RCBCostModel* output);

struct RCBWorkKey {
    RCBWorkPhase phase = RCBWorkPhase::Integration;
    RCBFmmMode fmm_mode = RCBFmmMode::NotApplicable;
    RCBLinkClass link_class = RCBLinkClass::PCIe;

    bool operator==(const RCBWorkKey& other) const {
        return phase == other.phase && fmm_mode == other.fmm_mode &&
               link_class == other.link_class;
    }
};

// A sample is one GPU's active duration and attributed work from an agreed
// sampling epoch.  Coefficients are seconds per work unit, never raw seconds.
// Samples with allocation growth or retry work are deliberately discarded:
// they describe exceptional capacity management rather than steady-state work.
struct RCBActiveSample {
    int64_t epoch = 0;
    RCBWorkKey key{};
    double active_seconds = 0.0;
    double work_units = 0.0;
    bool valid = true;
    bool allocation_grew = false;
    bool retried = false;
};

struct RCBCalibrationUpdate {
    RCBWorkKey key{};
    double global_median_seconds_per_unit = 0.0;
    double coefficient_seconds_per_unit = 0.0;
    uint32_t accepted_samples = 0;
};

class RCBWorkCalibrator {
public:
    explicit RCBWorkCalibrator(float half_life_samples = 8.0f);

    void set_half_life(float half_life_samples);
    // Exact required keys prevent a measurement in another phase or link
    // class from declaring the active configuration calibrated.
    void set_enabled_keys(const std::vector<RCBWorkKey>& keys);
    void reset();

    // All entries must have the supplied epoch.  The caller is responsible for
    // collecting the per-GPU samples without introducing a new synchronization
    // point; this method only pools the already-available common epoch.
    std::vector<RCBCalibrationUpdate> observe_global_epoch(
        int64_t epoch, const std::vector<RCBActiveSample>& samples);

    bool has_coefficient(const RCBWorkKey& key) const;
    double coefficient(const RCBWorkKey& key, double fallback = 0.0) const;
    // Compatibility helper for callers that ask for the active FMM cost.
    double blended_fmm_coefficient(RCBWorkPhase phase, RCBLinkClass link_class,
                                   double fallback = 0.0) const;
    bool ready() const;

    static bool is_usable(const RCBActiveSample& sample);

private:
    static constexpr size_t kPhaseCount = static_cast<size_t>(RCBWorkPhase::Count);
    static constexpr size_t kModeCount = static_cast<size_t>(RCBFmmMode::Count);
    static constexpr size_t kLinkCount = 2;
    static constexpr size_t kKeyCount = kPhaseCount * kModeCount * kLinkCount;

    struct Coefficient { double value = 0.0; bool initialized = false; };
    static size_t index(const RCBWorkKey& key);

    std::array<Coefficient, kKeyCount> coefficients_{};
    std::array<bool, kKeyCount> enabled_keys_{};
    double ema_alpha_ = 0.0;
};

// Per-GPU batch mailbox for completed active-work samples.  Producers publish
// independently; the coordinator only consumes an epoch already present for
// every GPU.  Slot locks protect vector storage but never form a cross-GPU
// rendezvous.
class RCBWorkSampleMailbox {
public:
    static constexpr int32_t kSlots = 32;

    void initialize(int32_t num_gpus);
    void publish(int32_t gpu, int64_t epoch,
                 const std::vector<RCBActiveSample>& samples);
    bool try_collect_latest(int64_t after_epoch, int64_t& collected_epoch,
                            std::vector<RCBActiveSample>& samples) const;
    // Ordered variant used by calibration so a newer empty/non-sampled epoch
    // cannot cause an older recurring sample to be skipped.
    bool try_collect_next(int64_t after_epoch, int64_t& collected_epoch,
                          std::vector<RCBActiveSample>& samples) const;

private:
    static constexpr int64_t kNoEpoch = -1;
    struct Slot {
        mutable std::mutex mutex;
        int64_t epoch = kNoEpoch;
        std::vector<RCBActiveSample> samples;
    };

    Slot& slot_for(int32_t gpu, int64_t epoch);
    const Slot& slot_for(int32_t gpu, int64_t epoch) const;

    int32_t num_gpus_ = 0;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<std::atomic<int64_t>[]> latest_epoch_;
};

} // namespace cudadem

#endif // CUDADEM_RCB_WORK_MODEL_H

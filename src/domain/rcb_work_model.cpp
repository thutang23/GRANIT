#include "rcb_work_model.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace cudadem {

void RCBWorkCalibrator::save_checkpoint(std::ostream& out) const {
    const uint32_t count=kKeyCount;
    out.write(reinterpret_cast<const char*>(&count),sizeof(count));
    out.write(reinterpret_cast<const char*>(&ema_alpha_),sizeof(ema_alpha_));
    for (size_t i=0; i<kKeyCount; ++i) {
        const uint8_t flags[2]={uint8_t(coefficients_[i].initialized),uint8_t(enabled_keys_[i])};
        out.write(reinterpret_cast<const char*>(&coefficients_[i].value),sizeof(double));
        out.write(reinterpret_cast<const char*>(flags),sizeof(flags));
    }
    if (!out) throw std::runtime_error("Cannot save load-balance calibration");
}
void RCBWorkCalibrator::restore_checkpoint(std::istream& in) {
    uint32_t count=0; in.read(reinterpret_cast<char*>(&count),sizeof(count));
    in.read(reinterpret_cast<char*>(&ema_alpha_),sizeof(ema_alpha_));
    if (!in || count!=kKeyCount || !std::isfinite(ema_alpha_) || !(ema_alpha_>0) || ema_alpha_>1)
        throw std::runtime_error("Invalid load-balance calibration checkpoint");
    for (size_t i=0; i<kKeyCount; ++i) {
        uint8_t flags[2]{};
        in.read(reinterpret_cast<char*>(&coefficients_[i].value),sizeof(double));
        in.read(reinterpret_cast<char*>(flags),sizeof(flags));
        if (!in || flags[0]>1 || flags[1]>1 || !std::isfinite(coefficients_[i].value) ||
            (flags[0] && !(coefficients_[i].value>0)))
            throw std::runtime_error("Invalid load-balance coefficient");
        coefficients_[i].initialized=flags[0]; enabled_keys_[i]=flags[1];
    }
}

const char* to_string(RCBCostModel value) {
    return value == RCBCostModel::Legacy ? "legacy" : "adaptive";
}

bool parse_rcb_cost_model(const std::string& value, RCBCostModel* output) {
    if (value == "adaptive") { *output = RCBCostModel::Adaptive; return true; }
    if (value == "legacy") { *output = RCBCostModel::Legacy; return true; }
    return false;
}

RCBWorkCalibrator::RCBWorkCalibrator(float half_life_samples) {
    set_half_life(half_life_samples);
}

void RCBWorkCalibrator::set_half_life(float half_life_samples) {
    // Invalid configuration is made safe here as well as rejected by YAML.
    if (!(half_life_samples > 0.0f) || !std::isfinite(half_life_samples)) half_life_samples = 8.0f;
    ema_alpha_ = 1.0 - std::exp(std::log(0.5) / static_cast<double>(half_life_samples));
}

void RCBWorkCalibrator::set_enabled_keys(const std::vector<RCBWorkKey>& keys) {
    enabled_keys_.fill(false);
    for (const RCBWorkKey& key : keys) {
        const size_t i = index(key);
        if (i < kKeyCount) enabled_keys_[i] = true;
    }
}

void RCBWorkCalibrator::reset() {
    coefficients_.fill({});
}

bool RCBWorkCalibrator::is_usable(const RCBActiveSample& sample) {
    return sample.valid && !sample.allocation_grew && !sample.retried &&
           std::isfinite(sample.active_seconds) && sample.active_seconds > 0.0 &&
           std::isfinite(sample.work_units) && sample.work_units > 0.0;
}

size_t RCBWorkCalibrator::index(const RCBWorkKey& key) {
    return ((static_cast<size_t>(key.phase) * kModeCount + static_cast<size_t>(key.fmm_mode)) *
            kLinkCount + static_cast<size_t>(key.link_class));
}

std::vector<RCBCalibrationUpdate> RCBWorkCalibrator::observe_global_epoch(
    int64_t epoch, const std::vector<RCBActiveSample>& samples) {
    std::map<size_t, std::vector<double>> grouped;
    for (const RCBActiveSample& sample : samples) {
        if (sample.epoch != epoch || !is_usable(sample)) continue;
        const size_t i = index(sample.key);
        if (i < kKeyCount) grouped[i].push_back(sample.active_seconds / sample.work_units);
    }

    std::vector<RCBCalibrationUpdate> updates;
    for (auto& [i, values] : grouped) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        const double median = values.size() % 2 ? values[middle]
            : 0.5 * (values[middle - 1] + values[middle]);
        Coefficient& coefficient = coefficients_[i];
        double next = median;
        if (coefficient.initialized) {
            const double raw = coefficient.value + ema_alpha_ * (median - coefficient.value);
            next = std::clamp(raw, coefficient.value * 0.5, coefficient.value * 2.0);
        }
        coefficient.value = next;
        coefficient.initialized = true;
        const size_t phase = i / (kModeCount * kLinkCount);
        const size_t rem = i % (kModeCount * kLinkCount);
        updates.push_back({{static_cast<RCBWorkPhase>(phase),
                            static_cast<RCBFmmMode>(rem / kLinkCount),
                            static_cast<RCBLinkClass>(rem % kLinkCount)},
                           median, next, static_cast<uint32_t>(values.size())});
    }
    return updates;
}

bool RCBWorkCalibrator::has_coefficient(const RCBWorkKey& key) const {
    const size_t i = index(key);
    return i < kKeyCount && coefficients_[i].initialized;
}

double RCBWorkCalibrator::coefficient(const RCBWorkKey& key, double fallback) const {
    const size_t i = index(key);
    return i < kKeyCount && coefficients_[i].initialized ? coefficients_[i].value : fallback;
}

double RCBWorkCalibrator::blended_fmm_coefficient(
    RCBWorkPhase phase, RCBLinkClass link_class, double fallback) const {
    return coefficient({phase, RCBFmmMode::Rebuild, link_class}, fallback);
}

bool RCBWorkCalibrator::ready() const {
    for (size_t key = 0; key < kKeyCount; ++key) {
        if (enabled_keys_[key] && !coefficients_[key].initialized) return false;
    }
    return true;
}

void RCBWorkSampleMailbox::initialize(int32_t num_gpus) {
    num_gpus_ = std::max<int32_t>(0, num_gpus);
    slots_.reset(num_gpus_ > 0
        ? new Slot[static_cast<size_t>(num_gpus_) * kSlots] : nullptr);
    latest_epoch_.reset(num_gpus_ > 0
        ? new std::atomic<int64_t>[num_gpus_] : nullptr);
    for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
        latest_epoch_[gpu].store(kNoEpoch, std::memory_order_relaxed);
    }
}

RCBWorkSampleMailbox::Slot& RCBWorkSampleMailbox::slot_for(
    int32_t gpu, int64_t epoch) {
    const size_t index = static_cast<size_t>(gpu) * kSlots
                       + static_cast<size_t>(epoch % kSlots);
    return slots_[index];
}

const RCBWorkSampleMailbox::Slot& RCBWorkSampleMailbox::slot_for(
    int32_t gpu, int64_t epoch) const {
    const size_t index = static_cast<size_t>(gpu) * kSlots
                       + static_cast<size_t>(epoch % kSlots);
    return slots_[index];
}

void RCBWorkSampleMailbox::publish(
    int32_t gpu, int64_t epoch, const std::vector<RCBActiveSample>& samples) {
    if (gpu < 0 || gpu >= num_gpus_ || epoch < 0) return;
    Slot& slot = slot_for(gpu, epoch);
    {
        std::lock_guard<std::mutex> lock(slot.mutex);
        slot.samples = samples;
        for (RCBActiveSample& sample : slot.samples) sample.epoch = epoch;
        slot.epoch = epoch;
    }
    latest_epoch_[gpu].store(epoch, std::memory_order_release);
}

bool RCBWorkSampleMailbox::try_collect_latest(
    int64_t after_epoch, int64_t& collected_epoch,
    std::vector<RCBActiveSample>& samples) const {
    if (num_gpus_ <= 0) return false;
    int64_t newest_common = std::numeric_limits<int64_t>::max();
    for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
        const int64_t latest = latest_epoch_[gpu].load(std::memory_order_acquire);
        if (latest == kNoEpoch) return false;
        newest_common = std::min(newest_common, latest);
    }
    const int64_t oldest = std::max<int64_t>(0, newest_common - (kSlots - 1));
    for (int64_t epoch = newest_common; epoch > after_epoch && epoch >= oldest; --epoch) {
        std::vector<RCBActiveSample> candidate;
        bool complete = true;
        for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
            const Slot& slot = slot_for(gpu, epoch);
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (slot.epoch != epoch) {
                complete = false;
                break;
            }
            candidate.insert(candidate.end(), slot.samples.begin(), slot.samples.end());
        }
        if (complete) {
            collected_epoch = epoch;
            samples.swap(candidate);
            return true;
        }
    }
    return false;
}

bool RCBWorkSampleMailbox::try_collect_next(
    int64_t after_epoch, int64_t& collected_epoch,
    std::vector<RCBActiveSample>& samples) const {
    if (num_gpus_ <= 0) return false;
    int64_t newest_common = std::numeric_limits<int64_t>::max();
    for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
        const int64_t latest = latest_epoch_[gpu].load(std::memory_order_acquire);
        if (latest == kNoEpoch) return false;
        newest_common = std::min(newest_common, latest);
    }
    const int64_t oldest = std::max<int64_t>(
        after_epoch + 1, std::max<int64_t>(0, newest_common - (kSlots - 1)));
    for (int64_t epoch = oldest; epoch <= newest_common; ++epoch) {
        std::vector<RCBActiveSample> candidate;
        bool complete = true;
        for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
            const Slot& slot = slot_for(gpu, epoch);
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (slot.epoch != epoch) {
                complete = false;
                break;
            }
            candidate.insert(candidate.end(), slot.samples.begin(), slot.samples.end());
        }
        if (complete) {
            collected_epoch = epoch;
            samples.swap(candidate);
            return true;
        }
    }
    return false;
}

} // namespace cudadem

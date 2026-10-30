#ifndef CUDADEM_YORP_CHECKPOINT_H
#define CUDADEM_YORP_CHECKPOINT_H
#include "../acceleration/acceleration.h"
#include "../math/yorp.cuh"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace cudadem {
// Checkpoint gathers temporarily visit other device contexts. Restore the
// caller even when a sidecar write/snapshot fails and unwinds the transaction.
class CheckpointDeviceGuard {
    int original_device_ = 0;
public:
    CheckpointDeviceGuard() { CUDA_CHECK(cudaGetDevice(&original_device_)); }
    ~CheckpointDeviceGuard() noexcept { (void)cudaSetDevice(original_device_); }
    void restore() { CUDA_CHECK(cudaSetDevice(original_device_)); }
    CheckpointDeviceGuard(const CheckpointDeviceGuard&) = delete;
    CheckpointDeviceGuard& operator=(const CheckpointDeviceGuard&) = delete;
};
inline uint64_t yorp_runtime_config_hash(const SimConfig& config) {
    const auto& y = config.acceleration.yorp;
    std::ostringstream out;
    out << std::setprecision(17) << y.enabled << ' ' << y.mode << ' '
        << y.start_time << ' ' << y.stop_time << ' ' << y.omega_init << ' '
        << y.increment << ' ' << y.dt_check << ' ' << y.epsilon << ' '
        << y.T_start_period << ' ' << y.T_inter_period << ' ' << y.T_final_period << ' '
        << y.fast_spinup_time << ' ' << y.slow_spinup_time << ' ' << y.adaptive << ' '
        << y.check_interval_seconds << ' ' << y.pause_specific_energy_j_per_kg << ' '
        << y.resume_specific_energy_j_per_kg << ' ' << y.pause_confirmation_checks << ' '
        << y.resume_confirmation_checks << ' ' << y.axis_ratio_drop_fraction << ' '
        << y.epsilon_reduction_fraction_per_cycle << ' ' << y.minimum_epsilon_fraction << ' '
        << y.max_omega_rad_s << ' ' << config.rotating_frame.enabled << ' '
        << config.rotating_frame.omega.x << ' ' << config.rotating_frame.omega.y << ' '
        << config.rotating_frame.omega.z << ' ' << config.acceleration.enable_inertial << ' '
        << config.acceleration.inertial_constant_omega;
    uint64_t hash=UINT64_C(1469598103934665603);
    for (unsigned char byte : out.str()) { hash ^= byte; hash *= UINT64_C(1099511628211); }
    return hash;
}
struct YorpRuntimeSidecarHeader {
    char magic[8];
    uint32_t version, ranks, runtime_bytes, external_schedule;
    int64_t step;
    double time;
    uint64_t config_hash;
};
inline void write_yorp_runtime_checkpoint(const std::string& path, const SimulationState& state,
    const SimConfig& config, const std::vector<int>& devices, const std::vector<cudaStream_t>& streams,
    const std::vector<double>& external_next_checks = {}) {
    if (!config.acceleration.yorp.enabled) return;
    if (devices.empty() || devices.size()!=streams.size() ||
        (!external_next_checks.empty() && devices.size()!=external_next_checks.size()))
        throw std::logic_error("Invalid YORP checkpoint rank layout");
    CheckpointDeviceGuard device_guard;
    YorpRuntimeSidecarHeader header{{'Y','O','R','P','R','U','N','1'},1,
        static_cast<uint32_t>(devices.size()),sizeof(YorpRuntimeCheckpoint),
        static_cast<uint32_t>(!external_next_checks.empty()),state.current_step,state.current_time,
        yorp_runtime_config_hash(config)};
    std::ofstream out(path+".yorp_runtime",std::ios::binary|std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&header),sizeof(header));
    for (size_t rank=0; rank<devices.size(); ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        auto* manager=AccelerationManager::yorp_manager_for_gpu(devices[rank]);
        YorpRuntimeCheckpoint checkpoint{};
        if (!manager || !manager->get_runtime_checkpoint(&checkpoint,streams[rank]))
            throw std::runtime_error("Cannot snapshot YORP runtime");
        out.write(reinterpret_cast<const char*>(&checkpoint),sizeof(checkpoint));
        if (header.external_schedule)
            out.write(reinterpret_cast<const char*>(&external_next_checks[rank]),sizeof(double));
    }
    out.close();
    device_guard.restore();
    if (!out) throw std::runtime_error("Cannot write YORP runtime checkpoint: "+path);
}
inline std::vector<YorpRuntimeCheckpoint> restore_yorp_runtime_checkpoint(const std::string& path,
    const SimulationState& state, const SimConfig& config, const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams, std::vector<double>* external_next_checks = nullptr) {
    if (!config.acceleration.yorp.enabled) return {};
    CheckpointDeviceGuard device_guard;
    YorpRuntimeSidecarHeader header{};
    std::ifstream in(path+".yorp_runtime",std::ios::binary);
    in.read(reinterpret_cast<char*>(&header),sizeof(header));
    if (!in || std::memcmp(header.magic,"YORPRUN1",8)!=0 || header.version!=1 ||
        header.ranks!=devices.size() || devices.size()!=streams.size() ||
        header.runtime_bytes!=sizeof(YorpRuntimeCheckpoint) || header.external_schedule>1 ||
        bool(header.external_schedule)!=bool(external_next_checks) || header.step!=state.current_step ||
        header.time!=state.current_time || header.config_hash!=yorp_runtime_config_hash(config))
        throw std::runtime_error("Missing or mismatched complete YORP runtime checkpoint: "+path);
    std::vector<YorpRuntimeCheckpoint> checkpoints(devices.size());
    if (external_next_checks) external_next_checks->resize(devices.size());
    for (size_t rank=0; rank<devices.size(); ++rank) {
        in.read(reinterpret_cast<char*>(&checkpoints[rank]),sizeof(YorpRuntimeCheckpoint));
        if (external_next_checks) {
            in.read(reinterpret_cast<char*>(&(*external_next_checks)[rank]),sizeof(double));
            if (!std::isfinite((*external_next_checks)[rank])) throw std::runtime_error("Invalid YORP observation cadence");
        }
        if (!in) throw std::runtime_error("Truncated complete YORP runtime checkpoint: "+path);
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        auto* manager=AccelerationManager::yorp_manager_for_gpu(devices[rank]);
        if (!manager || !manager->restore_runtime_checkpoint(checkpoints[rank],streams[rank]))
            throw std::runtime_error("Invalid complete YORP runtime checkpoint: "+path);
    }
    if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing complete YORP runtime checkpoint data");
    device_guard.restore();
    return checkpoints;
}
// Keep diagnostic history through the restored metric epoch. A resumed run
// may append subsequent observations without retaining rows from its old future.
inline void rewind_yorp_diagnostic_csv(const std::string& path, const std::string& time_column,
    double metric_time) {
    if (!std::filesystem::exists(path)) return;
    std::ifstream in(path); std::string header,line;
    if (!std::getline(in,header)) return;
    std::istringstream columns(header); size_t column=0; bool found=false;
    while (std::getline(columns,line,',')) { if(line==time_column){found=true;break;} ++column; }
    if (!found) throw std::runtime_error("Missing YORP CSV metric time column: "+path);
    std::vector<std::string> retained;
    while (std::getline(in,line)) {
        if (line.empty()) continue;
        std::istringstream row(line); std::string cell;
        for (size_t index=0; index<=column; ++index)
            if(!std::getline(row,cell,',')) throw std::runtime_error("Invalid YORP CSV row: "+path);
        size_t parsed=0; double time=std::stod(cell,&parsed);
        if(parsed!=cell.size()||!std::isfinite(time)) throw std::runtime_error("Invalid YORP CSV metric time: "+path);
        if(time<=metric_time+1.e-12*std::max(1.,std::fabs(metric_time))) retained.push_back(line);
    }
    if(in.bad())throw std::runtime_error("Cannot read YORP diagnostic: "+path);
    in.close();
    const std::string temporary=path+".resume_tmp";
    std::ofstream out(temporary,std::ios::trunc); out<<header<<'\n';
    for(const auto& row:retained)out<<row<<'\n';
    out.close();
    if(!out)throw std::runtime_error("Cannot rewind YORP diagnostic: "+path);
    std::filesystem::rename(temporary,path);
}
} // namespace cudadem
#endif

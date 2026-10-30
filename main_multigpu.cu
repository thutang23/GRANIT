#include "src/core/constant/adaptive_fmm_depth_lifecycle.h"
#include "src/core/sph/sph_respa_schedule.h"
#include "src/core/sph/sph_stage.h"
#include "src/core/sph/sph_history.h"
// =============================================================================
// CudaDEM Multi-GPU Main Entry Point
// =============================================================================

#include <nvtx3/nvToolsExt.h>

#ifdef CUDADEM_CPU_MOCK
#define DEFINE_CUDA_MOCK_GLOBALS
#endif
#include "cuda_utils.h"

#include "src/core/constant/command_line_interface.h"
#include "src/core/constant/system_initialize.h"
#include "src/core/constant/dem_timestep_selection.h"
#include "src/core/constant/load_or_resume.h"
#include "src/core/constant/cooperative_stop.h"
#include "src/core/constant/yorp_checkpoint.h"
#include "src/core/neighbor/neighbor_api.h"
#include "src/core/wall/wall.h"
#include "src/core/acceleration/acceleration.h"
#include "src/core/math/verlet_integration.cuh"
#include "src/core/math/verlet_respa_kernels.cuh"
#include "src/core/constant/time_step_schedule.h"
#include "src/core/particle_model/particle.h"
#include "src/core/shear/shear_box.h"
#include "src/core/sph/sph_physics.h"
#include "src/core/sph/sph_timestep.h"
#include "src/core/sph/sph_config_printer.h"
#include "src/core/sph/sph_gravity_correction.h"

// Multi-GPU domain module
#include "src/domain/domain_manager.h"
#include "src/domain/morton_root_service.cuh"
#include "src/domain/morton_work_kernels.cuh"
#include "src/domain/communicator.h"
#include "src/domain/peer_transfer.h"
#include "src/domain/let_communicator.h"
#include "src/domain/rcb_balancer.h"
#include "src/domain/rcb_work_model.h"
#include "src/domain/rebalance_trigger.h"
#include "src/domain/async_rebalance.h"
#include "src/domain/multigpu_phase_coordinator.h"
#include "src/domain/distributed_yorp_coordinator.h"
#include "src/domain/distributed_clumps.cuh"
#include "src/domain/distributed_contact_history.cuh"
#include "src/domain/fmm_depth_constraints.cuh"
#include "src/domain/distributed_fmm_depth_controller.h"

#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <array>
#include <numeric>
#include <limits>
#include <map>
#include <type_traits>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include <memory>

#ifndef CUDADEM_CPU_MOCK
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#endif

namespace cudadem {

// This legacy manager header intentionally uses a relative namespace and is
// included inside cudadem by its implementation and self_gravity.cu.
#include "src/core/acceleration/body_fitted_fmm/body_fitted_fmm_manager.h"

// Per-GPU histories are allocated after the global state is distributed.  The
// global WallManager initialization therefore cannot initialize these rows.
__global__ void initialize_migration_history_layout_kernel(
    int32_t* contact_history_offset,
    int32_t* wall_history_offset,
    int32_t capacity,
    int32_t contact_history_stride,
    int32_t wall_history_stride
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= capacity) return;
    contact_history_offset[idx] = idx * contact_history_stride;
    wall_history_offset[idx] = idx * wall_history_stride;
}

// Pure SPH has no rotational degrees of freedom.  These owned-particle
// kernels mirror the single-GPU SPH integrator while leaving halo particles
// read-only.
__global__ void sph_verlet_step1_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    const float* __restrict__ mass,
    double dt,
    int32_t num_particles
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const double inv_mass = 1.0 / static_cast<double>(mass[i]);
    double ax = force_x[i] * inv_mass;
    double ay = force_y[i] * inv_mass;
    double az = force_z[i] * inv_mass;
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) {
        ax = 0.0; ay = 0.0; az = 0.0;
    }
    if (dev_config.acceleration.enable_accel_limit) {
        const double magnitude = sqrt(ax * ax + ay * ay + az * az);
        if (magnitude > dev_config.acceleration.max_accel_magnitude) {
            const double scale = dev_config.acceleration.max_accel_magnitude / magnitude;
            ax *= scale; ay *= scale; az *= scale;
        }
    }
    const double half_dt = 0.5 * dt;
    const double vx = vel_x[i] + ax * half_dt;
    const double vy = vel_y[i] + ay * half_dt;
    const double vz = vel_z[i] + az * half_dt;
    vel_x[i] = vx; vel_y[i] = vy; vel_z[i] = vz;
    pos_x[i] += vx * dt; pos_y[i] += vy * dt; pos_z[i] += vz * dt;
    force_x[i] = 0.0; force_y[i] = 0.0; force_z[i] = 0.0;
}

__global__ void sph_verlet_step2_kernel(
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    const float* __restrict__ mass,
    double dt,
    int32_t num_particles
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const double half_dt_over_mass = 0.5 * dt / static_cast<double>(mass[i]);
    const double dvx = force_x[i] * half_dt_over_mass;
    const double dvy = force_y[i] * half_dt_over_mass;
    const double dvz = force_z[i] * half_dt_over_mass;
    if (isfinite(dvx) && isfinite(dvy) && isfinite(dvz)) {
        vel_x[i] += dvx; vel_y[i] += dvy; vel_z[i] += dvz;
    }
}

// =============================================================================
// Multi-GPU Contact Force Kernel - OPTIMIZED with Template Policy
// =============================================================================
// Modified to handle ghost particles - Newton's 3rd law only applied to locals
// OPTIMIZATION: Uses template parameter to avoid N3L atomic writes.
// FlipParticles=false: process i, FlipParticles=true: process j
// =============================================================================

// =============================================================================
// Multi-GPU Simulation Runner
// =============================================================================

class MultiGPUSimulationRunner {
private:
    enum class CalibrationSegment : uint8_t {
        Verlet1,
        HaloPack,
        HaloTransfer,
        HaloUnpack,
        NeighborSearch,
        ContactForce,
        SphDensity,
        SphConstitutive,
        SphForceHistory,
        WallForce,
        Verlet2,
        Count
    };

    // Backend 3 stores the common host controller and the ownership geometry.
    // Device pointers are reconstructed from values on restart.
    struct CommonDepthCheckpoint {
        uint32_t ranks = 0;
        int32_t launch_depth = 0;
        int32_t global_particles = 0;
        uint64_t gravity_epoch = 0, completed_epoch = 0;
        int64_t last_balance_step = 0, calibration_cursor = -1;
        MortonGrid grid{};
        uint64_t cuts[MAX_MULTI_GPUS+1]{};
        int64_t stable_counts[MAX_MULTI_GPUS]{};
        double bounds[6]{};
        bool partition_pending = false;
        uint64_t payload_checksum = 0;
    };
    static uint64_t depth_payload_checksum(const std::string& payload) {
        uint64_t hash = UINT64_C(1469598103934665603);
        for (unsigned char byte : payload) { hash ^= byte; hash *= UINT64_C(1099511628211); }
        return hash;
    }
    struct NonMortonDepthCheckpoint {
        uint32_t ranks = 0;
        uint32_t distributed_controller = 0;
        uint64_t gravity_epoch = 0;
        int64_t calibration_cursor = -1, pending_rebalance = -1;
        uint64_t payload_checksum = 0;
        DomainParamsGPU ownership{};
        RebalanceController rebalance{};
        uint8_t local_valid[MAX_MULTI_GPUS]{};
        AdaptiveFMMDepthCheckpoint local[MAX_MULTI_GPUS]{};
    };
    std::vector<uint64_t> adaptive_rank_diagnostic_sequences_;
    SimConfig adaptive_rank_diagnostic_config(int rank) const {
        SimConfig rank_config = config_;
        const std::filesystem::path base = config_.io.output_directory[0] != '\0'
            ? std::filesystem::path(config_.io.output_directory) : std::filesystem::path(".");
        const std::string directory = (base / ("adaptive_depth_rank_" + std::to_string(rank))).string();
        if (directory.size() >= sizeof(rank_config.io.output_directory))
            throw std::runtime_error("Per-rank adaptive diagnostic directory is too long");
        std::memcpy(rank_config.io.output_directory, directory.c_str(), directory.size()+1);
        return rank_config;
    }
    bool restored_non_morton_depth_ = false;
    int64_t restored_pending_rebalance_ = -1;

    struct RCBOwnershipCheckpoint {
        char magic[8];
        uint32_t version, ranks;
        int64_t step;
        double time;
        uint64_t gravity_epoch, calibration_bytes, calibration_checksum;
        int64_t calibration_cursor, pending_rebalance;
        DomainParamsGPU ownership;
        RebalanceController rebalance;
    };
    void save_rcb_ownership_checkpoint(const std::string& path, const SimulationState& state) {
        if (domain_params_.morton_enabled) return;
        if (num_gpus_ > MAX_MULTI_GPUS) throw std::runtime_error("RCB checkpoints support at most 8 GPUs");
        RCBOwnershipCheckpoint header{};
        std::memcpy(header.magic, "RCBOWN1", 8);
        header.version = 1; header.ranks = num_gpus_;
        header.step = state.current_step; header.time = state.current_time;
        header.gravity_epoch = gravity_evaluation_epoch_;
        header.calibration_cursor = common_calibration_cursor_;
        header.pending_rebalance = rebalance_mailbox_.pending_epoch();
        header.ownership = domain_params_;
        header.ownership.morton_cuts = nullptr; header.ownership.morton_cells = nullptr;
        header.rebalance = rebalance_ctl_;
        std::ostringstream payload(std::ios::binary);
        rcb_calibrator_.save_checkpoint(payload);
        const auto bytes = payload.str();
        header.calibration_bytes = bytes.size(); header.calibration_checksum = depth_payload_checksum(bytes);
        std::ofstream out(path + ".rcb_ownership", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(bytes.data(), bytes.size()); out.close();
        if (!out) throw std::runtime_error("Cannot save RCB ownership checkpoint");
    }
    void restore_rcb_ownership_checkpoint(const std::string& path, const SimulationState& state) {
        if (domain_params_.morton_enabled) return;
        if (!std::filesystem::exists(path + ".rcb_ownership") &&
            !checkpoint_detail::is_generation(std::filesystem::path(path))) return;
        RCBOwnershipCheckpoint header{};
        std::ifstream in(path + ".rcb_ownership", std::ios::binary);
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in || std::memcmp(header.magic, "RCBOWN1", 8) != 0 || header.version != 1 ||
            header.ranks != uint32_t(num_gpus_) || header.ranks > MAX_MULTI_GPUS || header.step != state.current_step || header.time != state.current_time ||
            header.ownership.num_gpus != num_gpus_ || header.ownership.morton_enabled ||
            header.ownership.morton_cuts || header.ownership.morton_cells || header.calibration_bytes > 1048576)
            throw std::runtime_error("Invalid RCB ownership checkpoint (rank/configuration/backend)");
        for (int rank = 0; rank < num_gpus_; ++rank) {
            if (!(header.ownership.gpu_box_min_x[rank] < header.ownership.gpu_box_max_x[rank]) ||
                !(header.ownership.gpu_box_min_y[rank] < header.ownership.gpu_box_max_y[rank]) ||
                !(header.ownership.gpu_box_min_z[rank] < header.ownership.gpu_box_max_z[rank]))
                throw std::runtime_error("Invalid RCB checkpoint ownership bounds");
        }
        std::string bytes(header.calibration_bytes, '\0'); in.read(bytes.data(), bytes.size());
        if (!in || in.peek() != std::char_traits<char>::eof() || depth_payload_checksum(bytes) != header.calibration_checksum)
            throw std::runtime_error("Corrupt RCB ownership calibration checkpoint");
        std::istringstream payload(bytes, std::ios::binary);
        rcb_calibrator_.restore_checkpoint(payload);
        if (payload.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing RCB calibration checkpoint data");
        domain_params_ = header.ownership;
        domain_params_.restore_morton_partition = true;
        gravity_evaluation_epoch_ = header.gravity_epoch;
        common_calibration_cursor_ = header.calibration_cursor;
        restored_pending_rebalance_ = header.pending_rebalance;
        rebalance_ctl_ = header.rebalance;
    }

    void save_non_morton_depth_checkpoint(const std::string& path, const SimulationState& state) {
        CheckpointDeviceGuard device_guard;
        NonMortonDepthCheckpoint checkpoint{};
        checkpoint.ranks = num_gpus_;
        checkpoint.distributed_controller = num_gpus_ > 1;
        checkpoint.gravity_epoch = gravity_evaluation_epoch_;
        checkpoint.calibration_cursor = common_calibration_cursor_;
        checkpoint.pending_rebalance = rebalance_mailbox_.pending_epoch();
        checkpoint.ownership = domain_params_;
        // Persist ownership values; device allocations are reconstructed.
        checkpoint.ownership.morton_cuts = nullptr;
        checkpoint.ownership.morton_cells = nullptr;
        checkpoint.rebalance = rebalance_ctl_;
        adaptive_rank_diagnostic_sequences_.resize(num_gpus_, 0);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            auto rank_config = adaptive_rank_diagnostic_config(rank);
            auto& rank_state = gpu_states_[rank];
            AdaptiveFMMDepthLifecycle diagnostics(rank_config, rank_state.local_state, rank_state.compute_stream);
            diagnostics.set_diagnostic_watermark(adaptive_rank_diagnostic_sequences_[rank]);
            diagnostics.append_adaptive_fmm_depth_diagnostics();
            adaptive_rank_diagnostic_sequences_[rank] = diagnostics.diagnostic_watermark();
            checkpoint.local_valid[rank] = AccelerationManager::get_adaptive_fmm_depth_checkpoint(
                &checkpoint.local[rank], rank_state.compute_stream);
            if (!checkpoint.local_valid[rank] && rank_state.subdomain.num_local_particles > 0)
                throw std::runtime_error("Cannot snapshot owned rank adaptive FMM runtime");
            if (checkpoint.local_valid[rank] &&
                (checkpoint.local[rank].producer_sequence != diagnostics.diagnostic_watermark() ||
                 checkpoint.local[rank].persisted_csv_sequence != diagnostics.diagnostic_watermark()))
                throw std::runtime_error("Per-rank adaptive checkpoint diagnostics were not persisted losslessly");
        }
        std::ostringstream payload(std::ios::binary);
        if (checkpoint.distributed_controller) distributed_fmm_depth_.save_checkpoint(payload);
        rcb_calibrator_.save_checkpoint(payload);
        const std::string bytes = payload.str();
        checkpoint.payload_checksum = depth_payload_checksum(bytes);
        AdaptiveFMMDepthCheckpointSidecarHeader header{{'A','F','M','M','D','P','T','H'},4,
            static_cast<uint32_t>(sizeof(checkpoint)+bytes.size()),state.current_step,state.current_time,
            AdaptiveFMMDepthLifecycle::adaptive_fmm_depth_config_fingerprint(config_)};
        std::ofstream out(adaptive_fmm_depth_checkpoint_sidecar_path(path), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(&checkpoint), sizeof(checkpoint));
        out.write(bytes.data(), bytes.size());
        out.close();
        if (!out) throw std::runtime_error("Cannot write non-Morton adaptive FMM checkpoint");
        device_guard.restore();
    }
    void restore_non_morton_depth_checkpoint(const std::string& path, const SimulationState& state) {
        AdaptiveFMMDepthCheckpointSidecarHeader header{};
        NonMortonDepthCheckpoint checkpoint{};
        std::ifstream in(adaptive_fmm_depth_checkpoint_sidecar_path(path), std::ios::binary);
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        in.read(reinterpret_cast<char*>(&checkpoint), sizeof(checkpoint));
        if (!in || std::memcmp(header.magic, "AFMMDPTH", 8) != 0 || header.version != 4 ||
            header.payload_size < sizeof(checkpoint) || header.payload_size > 1048576 ||
            header.step != state.current_step || header.time != state.current_time ||
            header.config_hash != AdaptiveFMMDepthLifecycle::adaptive_fmm_depth_config_fingerprint(config_) ||
            checkpoint.ranks != uint32_t(num_gpus_) || checkpoint.ranks > MAX_MULTI_GPUS ||
            checkpoint.distributed_controller != uint32_t(num_gpus_ > 1) ||
            checkpoint.ownership.morton_enabled || checkpoint.ownership.num_gpus != num_gpus_ ||
            checkpoint.ownership.morton_cuts || checkpoint.ownership.morton_cells || domain_params_.morton_enabled)
            throw std::runtime_error("Invalid non-Morton adaptive FMM checkpoint (rank/configuration/backend)");
        for (int rank = 0; rank < num_gpus_; ++rank) {
            if (checkpoint.local_valid[rank] > 1 ||
                !(checkpoint.ownership.gpu_box_min_x[rank] < checkpoint.ownership.gpu_box_max_x[rank]) ||
                !(checkpoint.ownership.gpu_box_min_y[rank] < checkpoint.ownership.gpu_box_max_y[rank]) ||
                !(checkpoint.ownership.gpu_box_min_z[rank] < checkpoint.ownership.gpu_box_max_z[rank]))
                throw std::runtime_error("Invalid checkpoint rank ownership bounds");
        }
        std::string bytes(header.payload_size-sizeof(checkpoint), '\0');
        in.read(bytes.data(), bytes.size());
        if (!in || in.peek() != std::char_traits<char>::eof() ||
            depth_payload_checksum(bytes) != checkpoint.payload_checksum)
            throw std::runtime_error("Corrupt non-Morton adaptive FMM checkpoint payload");
        std::istringstream payload(bytes, std::ios::binary);
        if (checkpoint.distributed_controller) {
            distributed_fmm_depth_.restore_checkpoint(payload, config_.acceleration.unified_fmm.adaptive_depth);
            if (distributed_fmm_depth_.gpu_count() != num_gpus_)
                throw std::runtime_error("Mismatched distributed FMM controller ranks");
        }
        rcb_calibrator_.restore_checkpoint(payload);
        if (payload.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Unexpected trailing non-Morton FMM controller state");
        domain_params_ = checkpoint.ownership;
        // Also prevents recomputing auto bounds and RCB boxes during scatter.
        domain_params_.restore_morton_partition = true;
        gravity_evaluation_epoch_ = checkpoint.gravity_epoch;
        common_calibration_cursor_ = checkpoint.calibration_cursor;
        rebalance_ctl_ = checkpoint.rebalance;
        restored_pending_rebalance_ = checkpoint.pending_rebalance;
        adaptive_rank_diagnostic_sequences_.assign(num_gpus_, 0);
        for (int rank = 0; rank < num_gpus_; ++rank) {
            if (!checkpoint.local_valid[rank]) continue;
            adaptive_rank_diagnostic_sequences_[rank] = checkpoint.local[rank].persisted_csv_sequence;
            auto rank_config = adaptive_rank_diagnostic_config(rank);
            auto& rank_state = gpu_states_[rank];
            AdaptiveFMMDepthLifecycle diagnostics(rank_config, rank_state.local_state, rank_state.compute_stream);
            diagnostics.set_diagnostic_watermark(checkpoint.local[rank].persisted_csv_sequence);
            diagnostics.rewind_adaptive_fmm_depth_csv_for_resume(checkpoint.local[rank].persisted_csv_sequence);
            CUDA_CHECK(cudaSetDevice(rank));
            if (!AccelerationManager::restore_adaptive_fmm_depth_checkpoint(checkpoint.local[rank]))
                throw std::runtime_error("Cannot restore rank adaptive FMM runtime");
        }
        CUDA_CHECK(cudaSetDevice(0));
        restored_non_morton_depth_ = true;
    }

    void save_common_depth_checkpoint(const std::string& path, const SimulationState& state) {
        if (!adaptive_fmm_depth_checkpoint_requested(config_)) return;
        if (num_gpus_ > MAX_MULTI_GPUS) throw std::runtime_error("Checkpoints support at most 8 GPUs");
        if (!domain_params_.morton_enabled) {
            save_non_morton_depth_checkpoint(path, state);
            return;
        }
        if (num_gpus_ > MAX_MULTI_GPUS) throw std::runtime_error("SPH checkpoints support at most 8 GPUs");
        CommonDepthCheckpoint checkpoint{};
        checkpoint.ranks = num_gpus_; checkpoint.launch_depth = common_launch_depth_;
        checkpoint.global_particles = common_global_particles_;
        checkpoint.gravity_epoch = gravity_evaluation_epoch_; checkpoint.completed_epoch = common_completed_epoch_;
        checkpoint.last_balance_step = common_last_balance_step_;
        checkpoint.calibration_cursor = common_calibration_cursor_;
        checkpoint.grid = domain_params_.morton_grid; checkpoint.partition_pending = common_partition_pending_;
        for (int g=0; g<=num_gpus_; ++g) checkpoint.cuts[g] = domain_params_.morton_cuts[g];
        for (int g=0; g<num_gpus_; ++g) checkpoint.stable_counts[g] = common_stable_owned_counts_.size()>size_t(g) ? common_stable_owned_counts_[g] : 0;
        checkpoint.bounds[0]=domain_params_.domain_min_x; checkpoint.bounds[1]=domain_params_.domain_min_y;
        checkpoint.bounds[2]=domain_params_.domain_min_z; checkpoint.bounds[3]=domain_params_.domain_max_x;
        checkpoint.bounds[4]=domain_params_.domain_max_y; checkpoint.bounds[5]=domain_params_.domain_max_z;
        std::ostringstream payload(std::ios::binary);
        common_fmm_depth_.save_checkpoint(payload);
        rcb_calibrator_.save_checkpoint(payload);
        const std::string bytes=payload.str(); checkpoint.payload_checksum=depth_payload_checksum(bytes);
        AdaptiveFMMDepthCheckpointSidecarHeader header{{'A','F','M','M','D','P','T','H'},3,
            static_cast<uint32_t>(sizeof(checkpoint)+bytes.size()),state.current_step,state.current_time,
            AdaptiveFMMDepthLifecycle::adaptive_fmm_depth_config_fingerprint(config_)};
        const std::string target=adaptive_fmm_depth_checkpoint_sidecar_path(path), temporary=target+".tmp";
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&header),sizeof(header));
        out.write(reinterpret_cast<const char*>(&checkpoint),sizeof(checkpoint));
        out.write(bytes.data(),bytes.size()); out.close();
        if (!out) throw std::runtime_error("Cannot write common FMM checkpoint");
        std::filesystem::rename(temporary,target);
    }
    void restore_common_depth_checkpoint(const std::string& path, const SimulationState& state) {
        if (!domain_params_.morton_enabled) {
            restore_non_morton_depth_checkpoint(path, state);
            return;
        }
        AdaptiveFMMDepthCheckpointSidecarHeader header{}; CommonDepthCheckpoint checkpoint{};
        std::ifstream in(adaptive_fmm_depth_checkpoint_sidecar_path(path),std::ios::binary);
        in.read(reinterpret_cast<char*>(&header),sizeof(header));
        in.read(reinterpret_cast<char*>(&checkpoint),sizeof(checkpoint));
        if (!in || header.version!=3 || header.step!=state.current_step || header.time!=state.current_time ||
            header.payload_size<sizeof(checkpoint) || header.payload_size>131072 ||
            header.config_hash!=AdaptiveFMMDepthLifecycle::adaptive_fmm_depth_config_fingerprint(config_) ||
            checkpoint.ranks!=uint32_t(num_gpus_) || checkpoint.ranks>MAX_MULTI_GPUS || !domain_params_.morton_enabled ||
            checkpoint.launch_depth<config_.acceleration.unified_fmm.adaptive_depth.min_depth ||
            checkpoint.launch_depth>config_.acceleration.unified_fmm.adaptive_depth.max_depth ||
            !std::isfinite(checkpoint.grid.size) || !(checkpoint.grid.size>0) ||
            checkpoint.grid.dimension!=(config_.shear_box.enabled?2:config_.acceleration.unified_fmm.tree_dimension) ||
            checkpoint.grid.depth<2 || checkpoint.grid.depth>20)
            throw std::runtime_error("Invalid/mismatched common FMM checkpoint (backend/rank/configuration)");
        std::string bytes(header.payload_size-sizeof(checkpoint),'\0');
        in.read(bytes.data(),bytes.size());
        if (!in || in.peek()!=std::char_traits<char>::eof() || depth_payload_checksum(bytes)!=checkpoint.payload_checksum)
            throw std::runtime_error("Corrupt common FMM checkpoint payload");
        std::istringstream payload(bytes,std::ios::binary);
        common_fmm_depth_.restore_checkpoint(payload,config_.acceleration.unified_fmm.adaptive_depth);
        rcb_calibrator_.restore_checkpoint(payload);
        if (common_fmm_depth_.gpu_count()!=1 || payload.peek()!=std::char_traits<char>::eof())
            throw std::runtime_error("Invalid common FMM controller payload");
        common_fmm_command_=common_fmm_depth_.command_for_gpu(0); common_launch_depth_=checkpoint.launch_depth;
        common_global_particles_=checkpoint.global_particles;
        gravity_evaluation_epoch_=checkpoint.gravity_epoch; common_completed_epoch_=checkpoint.completed_epoch;
        common_last_balance_step_=checkpoint.last_balance_step; common_calibration_cursor_=checkpoint.calibration_cursor; common_partition_pending_=checkpoint.partition_pending;
        common_stable_owned_counts_.assign(checkpoint.stable_counts,checkpoint.stable_counts+num_gpus_);
        domain_params_.domain_min_x=checkpoint.bounds[0]; domain_params_.domain_min_y=checkpoint.bounds[1];
        domain_params_.domain_min_z=checkpoint.bounds[2]; domain_params_.domain_max_x=checkpoint.bounds[3];
        domain_params_.domain_max_y=checkpoint.bounds[4]; domain_params_.domain_max_z=checkpoint.bounds[5];
        DomainManager::set_morton_partition(domain_params_,checkpoint.grid,
            std::vector<uint64_t>(checkpoint.cuts,checkpoint.cuts+num_gpus_+1));
        domain_params_.restore_morton_partition=true;
    }

    SimConfig config_;
    DomainParamsGPU domain_params_;
    MortonRootService morton_root_;
    MortonGpuWorkBalance morton_balance_;
    int common_launch_depth_ = 0;
    std::vector<MortonWorkCollector> morton_work_collectors_;
    std::vector<body_fitted_fmm::DistributedFMMMetrics> common_metrics_;
    std::vector<std::array<double,4>> common_work_totals_;
    std::array<double,4> common_work_coefficients_{};
    DistributedFMMDepthController common_fmm_depth_;
    DistributedFMMDepthCommand common_fmm_command_{};
    uint64_t common_completed_epoch_=0;
    int64_t common_calibration_cursor_=-1;
    std::vector<int> common_depth_ceilings_;
    std::vector<int64_t> common_stable_owned_counts_;
    int common_global_particles_=0;
    int64_t common_last_balance_step_=0;
    bool common_partition_pending_=false,common_partition_changed_=false,common_balance_due_=false;
    bool common_yorp_transition_pending_=false;

    std::vector<GPUDomainState> gpu_states_;
    std::vector<Communicator> communicators_;
    LETCommunicator* let_comms_;  // per-GPU, allocated with new[]
    RCBBalancer balancer_;
    RebalanceController rebalance_ctl_;
    AsyncRebalanceMailbox rebalance_mailbox_;
    RCBWorkCalibrator rcb_calibrator_;
    RCBWorkSampleMailbox rcb_sample_mailbox_;
    MultiGPUPhaseCoordinator phase_coordinator_;
    DistributedYorpCoordinator distributed_yorp_;
    DistributedClumps distributed_clumps_;
    FMMDepthConstraintCensus clump_depth_census_;
    std::string clump_collective_error_;
    int clump_logged_base_=-1,clump_logged_min_=-1,clump_logged_max_=-1;
    double clump_logged_leaf_width_=-1.0;
    std::vector<std::string> clump_depth_errors_;
    DistributedFMMDepthController distributed_fmm_depth_;
    static constexpr int32_t kTimingEventSlots = 4;
    // Flat [gpu * kTimingEventSlots + slot] rings.  Completed samples are
    // queried without blocking; a busy ring causes sampling to be skipped.
    std::vector<cudaEvent_t> step_start_events_;
    std::vector<cudaEvent_t> step_stop_events_;
    std::vector<cudaEvent_t> migration_transfer_events_;
    std::vector<cudaEvent_t> halo_transfer_events_;
    std::vector<cudaEvent_t> calibration_start_events_;
    std::vector<cudaEvent_t> calibration_stop_events_;
    std::vector<ContactListBuffer> contacts_;
    std::vector<SphDamageWorkspace> sph_damage_workspaces_;
    std::vector<int32_t> latest_contact_counts_;

    // Dead (-1) rows never enter the distributed state.  They stay host-side
    // and are appended verbatim to every unified-v3 output snapshot.
    struct DeadRows {
        std::vector<double> pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,force_x,force_y,force_z;
        std::vector<double> omega_x,omega_y,omega_z,torque_x,torque_y,torque_z,energy;
        std::vector<double> density;
        std::vector<float> radius,mass,moi,smoothing_length,pressure,c_sound,damage;
        std::vector<float> clump_off_x,clump_off_y,clump_off_z;
        std::vector<int32_t> global_id,material_id,clump_id;
        std::vector<int64_t> winding_x,winding_y,winding_z;
        int32_t size() const { return static_cast<int32_t>(clump_id.size()); }
    } dead_rows_;

    struct DEMExactRow {
        int32_t gid, wall_count;
        double force[3], torque[3];
    };
    struct DEMClumpCheckpointHeader {
        char magic[8];
        uint32_t version, body_bytes, wall_bytes, contact_stride, wall_stride, bodies;
        uint64_t particles, walls, clump_fingerprint, checkpoint_hash, payload_hash;
        MortonGrid ownership_root;
        uint64_t cuts[MAX_MULTI_GPUS+1];
        double bounds[6];
        uint32_t ranks;
        int64_t step;
        double time, dt;
    };
    std::vector<DistributedClumpBody> restored_bodies_;
    std::vector<DEMExactRow> restored_exact_rows_;
    std::vector<ContactHistory> restored_contacts_;
    std::vector<int32_t> restored_contact_counts_, restored_contact_gids_;
    std::vector<WallContactHistory> restored_wall_rows_;
    std::vector<size_t> restored_wall_offsets_;
    int restored_contact_stride_=16, restored_wall_stride_=16;
    uint32_t restored_owner_ranks_=0;

    static uint64_t checkpoint_bytes_hash(uint64_t hash,const void* bytes,size_t count) {
        const auto* data=static_cast<const unsigned char*>(bytes);
        for(size_t i=0;i<count;++i){hash^=data[i];hash*=1099511628211ULL;}
        return hash;
    }
    static uint64_t checkpoint_file_hash(const std::string& path) {
        std::ifstream file(path,std::ios::binary);
        if(!file)throw std::runtime_error("Cannot read checkpoint for clump-sidecar binding: "+path);
        uint64_t hash=1469598103934665603ULL;char bytes[65536];
        while(file){file.read(bytes,sizeof(bytes));hash=checkpoint_bytes_hash(hash,bytes,file.gcount());}
        return hash;
    }
    template<class T> static uint64_t checkpoint_vector_hash(uint64_t hash,const std::vector<T>& rows) {
        return checkpoint_bytes_hash(hash,rows.data(),rows.size()*sizeof(T));
    }
    void restore_dem_clump_checkpoint(const std::string& path,SimulationState& global) {
        DEMClumpCheckpointHeader header{};
        std::ifstream file(path+".distributed_clumps",std::ios::binary);
        file.read(reinterpret_cast<char*>(&header),sizeof(header));
        if(!file||std::memcmp(header.magic,"MGCLMP1",8)!=0||(header.version!=2&&header.version!=3)||
           header.body_bytes!=sizeof(DistributedClumpBody)||header.wall_bytes!=sizeof(WallContactHistory)||
           header.bodies!=static_cast<uint32_t>(config_.clump.num_instances)||
           header.particles!=static_cast<uint64_t>(global.particles.num_particles)||
           header.step!=global.current_step||header.time!=global.current_time||header.dt!=global.dt||
           header.clump_fingerprint!=config_.clump.clump_metadata_fingerprint||
           header.ranks==0||header.ranks>MAX_MULTI_GPUS||
           (header.ownership_root.dimension != 0 &&
            (header.ownership_root.dimension!=(config_.shear_box.enabled?2:3)||
             !(header.ownership_root.size>0)||!std::isfinite(header.ownership_root.size)||
             header.ownership_root.depth<2||header.ownership_root.depth>20))||
           header.contact_stride==0||header.contact_stride>1024||header.wall_stride==0||header.wall_stride>1024||
           header.walls>header.particles*header.wall_stride||header.checkpoint_hash!=checkpoint_file_hash(path))
            throw std::runtime_error("Missing, corrupt or mismatched distributed-clump checkpoint sidecar: "+path);
        const uint64_t expected_size=sizeof(header)+uint64_t(header.bodies)*sizeof(DistributedClumpBody)+
            header.particles*sizeof(DEMExactRow)+header.walls*sizeof(WallContactHistory);
        file.seekg(0,std::ios::end);
        if(!file||static_cast<uint64_t>(file.tellg())!=expected_size)throw std::runtime_error("Corrupt distributed-clump checkpoint size: "+path);
        file.seekg(sizeof(header));
        restored_contact_stride_=header.contact_stride;restored_wall_stride_=header.wall_stride;
        restored_owner_ranks_=header.ranks;
        domain_params_.domain_min_x=header.bounds[0];domain_params_.domain_min_y=header.bounds[1];domain_params_.domain_min_z=header.bounds[2];
        domain_params_.domain_max_x=header.bounds[3];domain_params_.domain_max_y=header.bounds[4];domain_params_.domain_max_z=header.bounds[5];
        if ((header.ownership_root.dimension != 0) != domain_params_.morton_enabled)
            throw std::runtime_error("Checkpoint ownership backend does not match current configuration");
        if (domain_params_.morton_enabled) {
        std::vector<uint64_t> cuts;
        if(header.ranks==static_cast<uint32_t>(num_gpus_))cuts.assign(header.cuts,header.cuts+num_gpus_+1);
        else {
            const int n=global.particles.num_particles;std::vector<double> x(n),y(n),z(n);
            CUDA_CHECK(cudaMemcpy(x.data(),global.particles.pos_x,n*sizeof(double),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y.data(),global.particles.pos_y,n*sizeof(double),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(z.data(),global.particles.pos_z,n*sizeof(double),cudaMemcpyDeviceToHost));
            std::vector<MortonLeafWeight> weights;weights.reserve(n);
            for(int i=0;i<n;++i)weights.push_back({header.ownership_root.leaf_begin(header.ownership_root.key(x[i],y[i],z[i])),1.});
            cuts=weighted_morton_cuts(std::move(weights),header.ownership_root,num_gpus_);
        }
        DomainManager::set_morton_partition(domain_params_,header.ownership_root,cuts);
        domain_params_.restore_morton_partition=true;
        }
        restored_bodies_.resize(header.bodies);restored_exact_rows_.resize(header.particles);restored_wall_rows_.resize(header.walls);
        auto read=[&](auto& rows){file.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(rows[0]));};
        read(restored_bodies_);read(restored_exact_rows_);read(restored_wall_rows_);
        const uint64_t expected_hash=header.payload_hash;header.payload_hash=0;
        uint64_t hash=checkpoint_bytes_hash(1469598103934665603ULL,&header,sizeof(header));
        hash=checkpoint_vector_hash(hash,restored_bodies_);
        hash=checkpoint_vector_hash(hash,restored_exact_rows_);hash=checkpoint_vector_hash(hash,restored_wall_rows_);
        if(!file||file.peek()!=std::char_traits<char>::eof()||hash!=expected_hash)
            throw std::runtime_error("Corrupt distributed-clump checkpoint payload: "+path);
        for(size_t c=0;c<restored_bodies_.size();++c){const auto& body=restored_bodies_[c];
            if(body.cid!=static_cast<int32_t>(c+1)||!(body.mass>0)||!std::isfinite(body.mass))
                throw std::runtime_error("Invalid authoritative clump body in checkpoint");
            double norm=0;for(double q:body.pose.quat)norm+=q*q;
            if(!std::isfinite(norm)||std::fabs(norm-1)>1.e-10)throw std::runtime_error("Invalid clump checkpoint orientation");
            for(double x:body.pose.com)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint COM");
            for(double x:body.pose.vel)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint velocity");
            for(double x:body.pose.omega)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint spin");
            for(double x:body.inertia)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint inertia");
            for(double x:body.wrench.f)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint force");
            for(double x:body.wrench.t)if(!std::isfinite(x))throw std::runtime_error("Invalid clump checkpoint torque");
        }
        restored_wall_offsets_.assign(restored_exact_rows_.size()+1,0);
        for(size_t i=0;i<restored_exact_rows_.size();++i){const auto& row=restored_exact_rows_[i];
            if((i&&row.gid<=restored_exact_rows_[i-1].gid)||row.wall_count<0||row.wall_count>restored_wall_stride_)
                throw std::runtime_error("Invalid particle IDs/wall counts in clump checkpoint");
            restored_wall_offsets_[i+1]=restored_wall_offsets_[i]+row.wall_count;
        }
        if(restored_wall_offsets_.back()!=restored_wall_rows_.size())throw std::runtime_error("Invalid clump checkpoint wall-history payload");
        const int n=global.particles.num_particles;
        std::vector<int32_t> gids(n);CUDA_CHECK(cudaMemcpy(gids.data(),global.particles.global_id,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
        std::vector<double> fx(n),fy(n),fz(n),tx(n),ty(n),tz(n);
        for(int i=0;i<n;++i){const auto& row=restored_exact_rows_[restart_exact_index(gids[i])];
            fx[i]=row.force[0];fy[i]=row.force[1];fz[i]=row.force[2];tx[i]=row.torque[0];ty[i]=row.torque[1];tz[i]=row.torque[2];}
        auto put=[&](auto* target,const auto& rows){CUDA_CHECK(cudaMemcpy(target,rows.data(),n*sizeof(rows[0]),cudaMemcpyHostToDevice));};
        put(global.particles.force_x,fx);put(global.particles.force_y,fy);put(global.particles.force_z,fz);
        put(global.particles.torque_x,tx);put(global.particles.torque_y,ty);put(global.particles.torque_z,tz);
        if(header.bodies && !global.particles.input_has_clump_offset)throw std::runtime_error("Distributed DEM restart requires stored body-frame offsets");
    }
    size_t restart_exact_index(int32_t gid) const {
        const auto row=std::lower_bound(restored_exact_rows_.begin(),restored_exact_rows_.end(),gid,
            [](const DEMExactRow& row,int32_t value){return row.gid<value;});
        if(row==restored_exact_rows_.end()||row->gid!=gid)throw std::runtime_error("Checkpoint particle global ID is missing");
        return row-restored_exact_rows_.begin();
    }
    void stage_dem_checkpoint_contacts(const SimulationState& global) {
        const int n=global.particles.num_particles;restored_contact_gids_.resize(n);restored_contact_counts_.resize(n);
        CUDA_CHECK(cudaMemcpy(restored_contact_gids_.data(),global.particles.global_id,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(restored_contact_counts_.data(),global.particles.num_contacts,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
        const int stride=global.contact_history.max_contacts_per_particle;
        if(stride!=restored_contact_stride_&&std::any_of(restored_contact_counts_.begin(),restored_contact_counts_.end(),[](int x){return x>0;}))
            throw std::runtime_error("DEM checkpoint contact stride does not match clump sidecar");
        restored_contacts_.resize(static_cast<size_t>(n)*restored_contact_stride_);
        if(stride==restored_contact_stride_)CUDA_CHECK(cudaMemcpy(restored_contacts_.data(),global.contact_history.data,
            restored_contacts_.size()*sizeof(ContactHistory),cudaMemcpyDeviceToHost));
    }
    void restore_dem_rank_histories(GPUDomainState& state) {
        const int n=state.subdomain.num_local_particles;std::vector<int32_t> gids(n),counts(n),wall_counts(n);
        if(n)CUDA_CHECK(cudaMemcpy(gids.data(),state.local_state.particles.global_id,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
        std::vector<ContactHistory> contacts(static_cast<size_t>(n)*restored_contact_stride_);
        std::vector<WallContactHistory> walls(static_cast<size_t>(n)*restored_wall_stride_);
        std::vector<std::pair<int32_t,size_t>> lookup;lookup.reserve(restored_contact_gids_.size());
        for(size_t i=0;i<restored_contact_gids_.size();++i)lookup.push_back({restored_contact_gids_[i],i});
        std::sort(lookup.begin(),lookup.end());
        for(int i=0;i<n;++i){const size_t exact=restart_exact_index(gids[i]);
            const auto row=std::lower_bound(lookup.begin(),lookup.end(),std::make_pair(gids[i],size_t{0}));
            if(row==lookup.end()||row->first!=gids[i])throw std::runtime_error("Missing checkpoint contact owner");
            counts[i]=restored_contact_counts_[row->second];wall_counts[i]=restored_exact_rows_[exact].wall_count;
            std::copy_n(restored_contacts_.data()+row->second*restored_contact_stride_,restored_contact_stride_,contacts.data()+static_cast<size_t>(i)*restored_contact_stride_);
            if(wall_counts[i])std::copy_n(restored_wall_rows_.data()+restored_wall_offsets_[exact],wall_counts[i],walls.data()+static_cast<size_t>(i)*restored_wall_stride_);
        }
        if(n){CUDA_CHECK(cudaMemcpy(state.local_state.contact_history.data,contacts.data(),contacts.size()*sizeof(ContactHistory),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.local_state.wall_contact_history.data,walls.data(),walls.size()*sizeof(WallContactHistory),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.local_state.particles.num_contacts,counts.data(),n*sizeof(int32_t),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.local_state.particles.num_wall_contacts,wall_counts.data(),n*sizeof(int32_t),cudaMemcpyHostToDevice));}
        #pragma omp critical(clump_diagnostics)
        std::cout<<"[clumps-restart] rank="<<state.gpu_id<<" restored_particles="<<n
                 <<" contacts="<<std::accumulate(counts.begin(),counts.end(),int64_t{0})
                 <<" wall_contacts="<<std::accumulate(wall_counts.begin(),wall_counts.end(),int64_t{0})<<'\n';
    }

    void save_dem_clump_sidecar(const std::string& path,const SimulationState& state,
                               const std::vector<DEMExactRow>& exact,
                               const std::vector<WallContactHistory>& walls,int contact_stride,int wall_stride) {
        CheckpointDeviceGuard device_guard;
        if(domain_params_.morton_enabled && !domain_params_.morton_cuts)
            throw std::runtime_error("Distributed checkpoints require initialized Morton ownership cuts");
        const auto bodies=distributed_clumps_.snapshot();DEMClumpCheckpointHeader header{};
        std::memcpy(header.magic,"MGCLMP1",8);header.version=3;header.body_bytes=sizeof(DistributedClumpBody);
        header.wall_bytes=sizeof(WallContactHistory);header.contact_stride=contact_stride;header.wall_stride=wall_stride;
        header.bodies=bodies.size();header.particles=exact.size();header.walls=walls.size();
        header.ranks=num_gpus_;
        if (domain_params_.morton_enabled) {
            header.ownership_root=domain_params_.morton_grid;
            for(int rank=0;rank<=num_gpus_;++rank)header.cuts[rank]=domain_params_.morton_cuts[rank];
        } else { header.ownership_root.dimension = 0; }
        header.bounds[0]=domain_params_.domain_min_x;header.bounds[1]=domain_params_.domain_min_y;header.bounds[2]=domain_params_.domain_min_z;
        header.bounds[3]=domain_params_.domain_max_x;header.bounds[4]=domain_params_.domain_max_y;header.bounds[5]=domain_params_.domain_max_z;
        header.clump_fingerprint=config_.clump.clump_metadata_fingerprint;header.checkpoint_hash=checkpoint_file_hash(path);
        header.step=state.current_step;header.time=state.current_time;header.dt=state.dt;
        uint64_t hash=checkpoint_bytes_hash(1469598103934665603ULL,&header,sizeof(header));
        hash=checkpoint_vector_hash(hash,bodies);hash=checkpoint_vector_hash(hash,exact);
        header.payload_hash=checkpoint_vector_hash(hash,walls);
        const std::string target=path+".distributed_clumps",temporary=target+".tmp";
        std::ofstream file(temporary,std::ios::binary|std::ios::trunc);
        file.write(reinterpret_cast<const char*>(&header),sizeof(header));
        auto write=[&](const auto& rows){file.write(reinterpret_cast<const char*>(rows.data()),rows.size()*sizeof(rows[0]));};
        write(bodies);write(exact);write(walls);file.close();
        if(!file)throw std::runtime_error("Cannot write exact distributed-clump checkpoint: "+target);
        std::filesystem::rename(temporary,target);
        device_guard.restore();
    }

    int32_t num_gpus_;
    bool initialized_;
    bool sph_mode_;
    bool resumed_ = false;
    bool restored_yorp_runtime_ = false;
    std::vector<double> next_yorp_checks_;
    int64_t last_checkpoint_step_ = -1;
    int64_t checkpoint_cadence_step_ = 0;
    double dt_;
    uint64_t gravity_evaluation_epoch_ = 0;
    int respa_shared_inner_count_ = 0;
    double respa_shared_macro_span_ = 0.0;
    double sph_shared_macro_end_ = 0.0;
    float sph_global_h_max_;
    bool adaptive_yorp_fatal_ = false;
    int64_t last_output_step_ = -1;
    uint64_t diagnostic_config_fingerprint_ = 0;
    std::string diagnostic_device_layout_ = "unknown";

    struct FMMTimingDiagnostic {
        uint64_t epoch = 0;
        int32_t gpu = -1;
        double elapsed_ms = 0.0;
        double remote_evaluation_ms = 0.0;
        int64_t owned_particles = 0;
        uint64_t trial_id = 0;
        int32_t launch_depth = 0;
        int32_t selected_depth = 0;
        int32_t candidate_depth = 0;
        int32_t effective_max_depth = 0;
        int64_t let_node_records = 0;
        int64_t let_particle_records = 0;
        uint64_t let_bytes = 0;
    };
    static constexpr size_t kFMMTimingDiagnosticCapacity = 512;
    std::vector<std::vector<FMMTimingDiagnostic>> fmm_timing_diagnostics_;
    std::vector<uint64_t> fmm_runtime_timing_dropped_;
    std::vector<uint64_t> fmm_diagnostic_rows_dropped_;

    // Phase 4: Per-GPU pinned memory + events for async contact count readback
    std::vector<int32_t*> h_pinned_num_contacts_;   // One pinned int32 per GPU
    std::vector<uint64_t*> h_pinned_sph_owned_degrees_; // One pinned scalar per GPU/timing slot
    // The hydro CSR is invalid after the gravity halo/tree refresh.  Keep its
    // owned-row prefix for the work balancer without retaining neighbor indices.
    std::vector<uint64_t*> d_sph_hydro_owned_offsets_;
    std::vector<size_t> sph_hydro_offset_capacities_;
    std::vector<int> sph_hydro_offsets_valid_;
    std::vector<cudaEvent_t> contact_count_events_;  // One event per GPU
    std::vector<bool> contact_count_ready_;           // Whether async copy completed

    static constexpr int32_t kContactCapacityPerParticle = 20;

    static uint64_t fingerprint_config_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        uint64_t hash = UINT64_C(1469598103934665603);
        char byte = 0;
        while (input.get(byte)) {
            hash ^= static_cast<unsigned char>(byte);
            hash *= UINT64_C(1099511628211);
        }
        return input.bad() ? 0 : hash;
    }

    std::string describe_visible_device_layout() const {
        std::ostringstream out;
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        out << "visible=" << (visible && *visible ? visible : "default")
            << ";bus=";
        for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
            char bus_id[32] = {};
            if (gpu) out << '+';
            if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), gpu) == cudaSuccess)
                out << bus_id;
            else
                out << "unknown";
        }
        std::string result = out.str();
        std::replace(result.begin(), result.end(), ',', '+');
        std::replace(result.begin(), result.end(), ' ', '_');
        return result;
    }

    void append_distributed_yorp_diagnostic(const AdaptiveYorpDiagnostic& d) {
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = std::string(config_.io.output_directory)
                               + "/adaptive_yorp_multigpu.csv";
        const bool header = !std::filesystem::exists(path) ||
                            std::filesystem::file_size(path) == 0;
        std::ofstream file(path, std::ios::app);
        if (!file) throw std::runtime_error(
            "cannot append distributed adaptive YORP diagnostic: " + path);
        if (header) {
            file << "schema_version,config_fingerprint,gpu_count,topology,device_layout,metric_time,state,omega_x,omega_y,omega_z,"
                    "specific_energy_j_per_kg,axis_ratio,effective_epsilon,confirmed_cycles,"
                    "transition_reason,transition_time,fatal\n";
        }
        file << std::setprecision(17)
             << 2 << ',' << std::hex << diagnostic_config_fingerprint_ << std::dec
             << ',' << num_gpus_ << ",cuda_p2p," << diagnostic_device_layout_
             << ',' << d.time << ','
             << d.state << ',' << d.omega_x << ',' << d.omega_y << ','
             << d.omega_z << ',' << d.specific_energy_j_per_kg << ','
             << d.current_baseline_axis_ratio << ',' << d.effective_epsilon << ','
             << d.confirmed_cycles << ',' << d.transition_reason << ','
             << d.transition_time << ',' << d.fatal << '\n';
        if (!file) throw std::runtime_error(
            "failed writing distributed adaptive YORP diagnostic: " + path);
    }

    void write_distributed_fmm_diagnostics() {
        if (!(config_.acceleration.unified_fmm.adaptive_depth.enabled && num_gpus_ > 1 && !domain_params_.morton_enabled)) return;
        std::filesystem::create_directories(config_.io.output_directory);
        const std::string path = std::string(config_.io.output_directory)
                               + "/adaptive_fmm_depth_multigpu.csv";
        std::ofstream file(path, std::ios::trunc);
        if (!file) throw std::runtime_error(
            "cannot write distributed FMM-depth diagnostics: " + path);
        file << "schema_version,config_fingerprint,gpu_count,topology,device_layout,epoch,gpu,elapsed_ms,remote_evaluation_ms,owned_particles,"
                "trial_id,launch_depth,selected_depth,candidate_depth,"
                "effective_max_depth,let_node_records,let_particle_records,let_bytes\n";
        std::vector<FMMTimingDiagnostic> rows;
        for (const auto& gpu_rows : fmm_timing_diagnostics_)
            rows.insert(rows.end(), gpu_rows.begin(), gpu_rows.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.epoch != b.epoch ? a.epoch < b.epoch : a.gpu < b.gpu;
        });
        for (const auto& row : rows) {
            file << std::setprecision(17) << 3 << ',' << std::hex
                 << diagnostic_config_fingerprint_ << std::dec << ',' << num_gpus_
                 << ",cuda_p2p," << diagnostic_device_layout_ << ','
                 << row.epoch << ',' << row.gpu << ','
                 << row.elapsed_ms << ',' << row.remote_evaluation_ms << ','
                 << row.owned_particles << ','
                 << row.trial_id << ','
                 << row.launch_depth << ',' << row.selected_depth << ','
                 << row.candidate_depth << ',' << row.effective_max_depth << ','
                 << row.let_node_records << ',' << row.let_particle_records << ','
                 << row.let_bytes << '\n';
        }
        const std::string event_path = std::string(config_.io.output_directory)
                                     + "/adaptive_fmm_depth_multigpu_events.csv";
        std::ofstream events(event_path, std::ios::trunc);
        if (!events) throw std::runtime_error(
            "cannot write distributed FMM-depth events: " + event_path);
        events << "schema_version,config_fingerprint,gpu_count,topology,device_layout,epoch,trial_id,gpu,selected_depth,"
                  "candidate_depth,global_max_ms,global_mean_ms,let_bytes,"
                  "relative_improvement,event,cancel_reason\n";
        for (const auto& event : distributed_fmm_depth_.diagnostics()) {
            const int cancel_reason =
                event.event == DistributedFMMDepthEvent::TrialCancelled
                ? static_cast<int>(event.cancel_reason) : -1;
            events << std::setprecision(17) << 2 << ',' << std::hex
                   << diagnostic_config_fingerprint_ << std::dec << ','
                   << num_gpus_ << ",cuda_p2p," << diagnostic_device_layout_ << ','
                   << event.epoch << ',' << event.trial_id << ',' << event.gpu_id
                   << ',' << event.selected_depth << ',' << event.candidate_depth
                   << ',' << event.global_max_ms << ',' << event.global_mean_ms
                   << ',' << event.let_bytes << ','
                   << event.primary_relative_improvement << ','
                   << static_cast<int>(event.event) << ','
                   << cancel_reason << '\n';
        }
        const uint64_t timing_dropped = std::accumulate(
            fmm_runtime_timing_dropped_.begin(),
            fmm_runtime_timing_dropped_.end(), uint64_t{0});
        const uint64_t row_dropped = std::accumulate(
            fmm_diagnostic_rows_dropped_.begin(),
            fmm_diagnostic_rows_dropped_.end(), uint64_t{0});
        const uint64_t controller_dropped = distributed_fmm_depth_.dropped_diagnostics();
        if (timing_dropped || row_dropped || controller_dropped) {
            std::cerr << "WARNING: dropped distributed FMM diagnostics: timing="
                      << timing_dropped << " rows=" << row_dropped
                      << " controller=" << controller_dropped
                      << std::endl;
        }
    }

    void capture_and_exclude_dead_rows(SimulationState& global_state) {
        ParticleSoA& p = global_state.particles;
        canonicalize_particle_clump_order(p, nullptr, sph_mode_);
        const int32_t total = p.num_particles;
        std::vector<int32_t> ids(total);
        std::vector<uint8_t> lifecycle(total);
        CUDA_CHECK(cudaMemcpy(ids.data(), p.clump_id, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lifecycle.data(), p.lifecycle_state, total * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        int32_t live = 0, max_part_id = -1;
        while (live < total && ids[live] >= 0 && lifecycle[live] == 0) {
            max_part_id = std::max(max_part_id, ids[live]);
            ++live;
        }
        if (sph_mode_) {
            std::vector<uint8_t> seen(static_cast<size_t>(max_part_id + 1), 0);
            for (int32_t i = 0; i < live; ++i) seen[ids[i]] = 1;
            for (int32_t part = 0; part <= max_part_id; ++part)
                if (!seen[part]) throw std::runtime_error("SPH Part IDs must be dense 0..K");
        }
        for (int32_t i = live; i < total; ++i)
            if (ids[i] != -1 && lifecycle[i] == 0)
                throw std::runtime_error("active particle row follows inactive particle row");
        if (live == 0) throw std::runtime_error("Multi-GPU input contains no live particles");
        const int32_t dead = total - live;
        if (dead == 0) return;
        throw std::invalid_argument("Distributed cooperative checkpoints require all rows live; inactive input rows are unsupported");
        auto copy_tail = [&](auto& dst, const auto* src) {
            using T = typename std::decay_t<decltype(dst)>::value_type;
            dst.resize(dead);
            CUDA_CHECK(cudaMemcpy(dst.data(), src + live, dead * sizeof(T), cudaMemcpyDeviceToHost));
        };
        copy_tail(dead_rows_.pos_x,p.pos_x); copy_tail(dead_rows_.pos_y,p.pos_y); copy_tail(dead_rows_.pos_z,p.pos_z);
        copy_tail(dead_rows_.vel_x,p.vel_x); copy_tail(dead_rows_.vel_y,p.vel_y); copy_tail(dead_rows_.vel_z,p.vel_z);
        copy_tail(dead_rows_.force_x,p.force_x); copy_tail(dead_rows_.force_y,p.force_y); copy_tail(dead_rows_.force_z,p.force_z);
        copy_tail(dead_rows_.omega_x,p.omega_x); copy_tail(dead_rows_.omega_y,p.omega_y); copy_tail(dead_rows_.omega_z,p.omega_z);
        copy_tail(dead_rows_.torque_x,p.torque_x); copy_tail(dead_rows_.torque_y,p.torque_y); copy_tail(dead_rows_.torque_z,p.torque_z);
        copy_tail(dead_rows_.energy,p.energy); copy_tail(dead_rows_.radius,p.radius); copy_tail(dead_rows_.mass,p.mass);
        copy_tail(dead_rows_.moi,p.moment_of_inertia); copy_tail(dead_rows_.smoothing_length,p.smoothing_length);
        copy_tail(dead_rows_.density,p.density); copy_tail(dead_rows_.pressure,p.pressure); copy_tail(dead_rows_.c_sound,p.c_sound);
        copy_tail(dead_rows_.damage,p.damage); copy_tail(dead_rows_.global_id,p.global_id);
        copy_tail(dead_rows_.material_id,p.material_id); copy_tail(dead_rows_.clump_id,p.clump_id);
        copy_tail(dead_rows_.clump_off_x,p.clump_off_x); copy_tail(dead_rows_.clump_off_y,p.clump_off_y); copy_tail(dead_rows_.clump_off_z,p.clump_off_z);
        copy_tail(dead_rows_.winding_x,p.winding_x); copy_tail(dead_rows_.winding_y,p.winding_y);
        copy_tail(dead_rows_.winding_z,p.winding_z);
        if (sph_mode_) {
            sanitize_inactive_sph_tail_values(dead_rows_.smoothing_length.data(),
                                              nullptr, dead,
                                              config_.sph.smoothing_length);
        }
        p.num_particles = live; p.num_local = live; p.num_ghost = 0;
        std::cout << "Multi-GPU: stashed " << dead << " dead rows; distributing " << live << " live rows" << std::endl;
    }

    size_t calibration_event_index(int32_t gpu, int32_t slot,
                                   CalibrationSegment segment) const {
        const size_t gpu_slot = static_cast<size_t>(gpu) * kTimingEventSlots + slot;
        return gpu_slot * static_cast<size_t>(CalibrationSegment::Count)
             + static_cast<size_t>(segment);
    }

    static RCBLinkClass calibration_link_class(LinkClass link) {
        return is_nvlink_fabric(link) ? RCBLinkClass::NVLink : RCBLinkClass::PCIe;
    }

    bool adaptive_calibration_ready() const {
        const RCBWorkKey integration{RCBWorkPhase::Integration,
                                     RCBFmmMode::NotApplicable,
                                     RCBLinkClass::PCIe};
        const RCBWorkKey physics{RCBWorkPhase::ContactPair,
                                 RCBFmmMode::NotApplicable, RCBLinkClass::PCIe};
        return rcb_calibrator_.ready() && rcb_calibrator_.has_coefficient(integration) &&
               (!sph_mode_ || rcb_calibrator_.has_coefficient(physics));
    }

    void ensure_contact_list_capacity(int32_t gpu_id, int32_t particle_capacity,
                                      cudaStream_t stream) {
        const int64_t required64 = static_cast<int64_t>(particle_capacity)
                                 * kContactCapacityPerParticle;
        if (required64 > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Contact-list capacity exceeds int32 indexing on GPU "
                                     + std::to_string(gpu_id));
        }
        ContactListBuffer& contacts = contacts_[gpu_id];
        const int32_t required = static_cast<int32_t>(required64);
        if (required <= contacts.max_contacts) return;
        const int32_t old_capacity = contacts.max_contacts;
        const int64_t grown = static_cast<int64_t>(old_capacity)
                            + (static_cast<int64_t>(old_capacity) + 1) / 2;
        const int32_t new_capacity = static_cast<int32_t>(
            std::max<int64_t>(required, grown));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ContactPair* replacement = nullptr;
        const cudaError_t status = cudaMalloc(
            &replacement, static_cast<size_t>(new_capacity) * sizeof(ContactPair));
        if (status != cudaSuccess) {
            throw std::runtime_error(
                "FATAL contact-list resize on GPU " + std::to_string(gpu_id)
                + ": old_capacity=" + std::to_string(old_capacity)
                + ", required_capacity=" + std::to_string(required)
                + ", requested_capacity=" + std::to_string(new_capacity)
                + ", cuda_error=" + cudaGetErrorString(status));
        }
        CUDA_CHECK(cudaFree(contacts.pairs));
        contacts.pairs = replacement;
        contacts.max_contacts = new_capacity;
        contacts.h_num_contacts = 0;
        CUDA_CHECK(cudaMemsetAsync(contacts.num_contacts, 0, sizeof(int32_t), stream));
        std::cout << "[memory] GPU " << gpu_id
                  << " neighbor grow arena=contact-list old_capacity=" << old_capacity
                  << " required=" << required
                  << " new_capacity=" << new_capacity
                  << " bytes=" << static_cast<size_t>(new_capacity) * sizeof(ContactPair)
                  << std::endl;
    }

public:
    MultiGPUSimulationRunner()
        : num_gpus_(0), initialized_(false), sph_mode_(false), dt_(1e-5),
          sph_global_h_max_(0.0f), let_comms_(nullptr) {}

    ~MultiGPUSimulationRunner() {
        shutdown();
    }

    bool initialize(const std::string& config_path) {
        std::cout << "\n";
        std::cout << "============================================" << std::endl;
        std::cout << "   CudaDEM - Multi-GPU DEM/SPH Simulation  " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "\n";

        // Parse configuration
        try {
            config_ = parse_yaml(config_path);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to parse config - " << e.what() << std::endl;
            return false;
        }
        diagnostic_config_fingerprint_ = fingerprint_config_file(config_path);
        sph_mode_ = config_.physics == SimulationPhysics::SPH;
        if (config_.acceleration.yorp.adaptive) {
            std::filesystem::create_directories(config_.io.output_directory);
            const std::string diagnostic_path =
                std::string(config_.io.output_directory)
                + "/adaptive_yorp_multigpu.csv";
            std::ofstream reset(diagnostic_path, std::ios::trunc);
            if (!reset) {
                std::cerr << "ERROR: Cannot reset distributed adaptive YORP diagnostic: "
                          << diagnostic_path << std::endl;
                return false;
            }
        }
        if (sph_mode_) print_sph_config(config_);
        if (sph_mode_ &&
            config_.neighbor.method == NeighborMethod::CELL_LINKED_LIST) {
            std::cerr << "ERROR: Multi-GPU SPH does not support the cell-linked-list "
                         "neighbor method because its fixed initial-local scratch "
                         "capacity is unsafe after halo exchange/migration. Use "
                         "neighbor.method: LBVH (recommended) or DIRECT."
                      << std::endl;
            return false;
        }
        if (sph_mode_) {
            for (int32_t wall = 0; wall < config_.num_walls; ++wall) {
                if (config_.walls[wall].type == WallType::PERIODICAL) {
                    std::cerr << "ERROR: Multi-GPU SPH periodic boundaries are not "
                                 "supported by the LBVH/DIRECT SPH neighbor queries. "
                                 "Remove the periodical wall configuration."
                              << std::endl;
                    return false;
                }
            }
        }

        if (!sph_mode_ && config_.clump.num_instances > 0) {
            auto clump_fail = [](const char* reason) {
                throw std::invalid_argument(std::string("Distributed DEM clumps: ") + reason);
            };
            if (config_.neighbor.method != NeighborMethod::LBVH)
                clump_fail("the initial distributed clump implementation requires neighbor.method: LBVH");
            for (int wall=0; wall<config_.num_walls; ++wall)
                if (!config_.shear_box.enabled && config_.walls[wall].enabled && config_.walls[wall].type == WallType::PERIODICAL)
                    clump_fail("periodic boundaries are unsupported");
            if (!config_.multigpu.morton_ownership) clump_fail("common Morton spatial ownership is required");
            if (config_.acceleration.yorp.enabled || config_.acceleration.yorp.adaptive)
                clump_fail("YORP is unsupported");
            if (config_.damping.enabled) clump_fail("global damping is unsupported");
            if (config_.acceleration.enable_accel_limit) clump_fail("acceleration limiting is unsupported");
            if (!config_.shear_box.enabled && (config_.coord_system == CoordinateSystem::ROTATING || config_.acceleration.enable_inertial))
                clump_fail("rotating and inertial frames are unsupported");
            if (config_.acceleration.enable_self_gravity) {
                const auto& fmm=config_.acceleration.unified_fmm;
                if (config_.acceleration.gravity_update_rate != 1 || config_.acceleration.gravity_macro_dt != 0)
                    clump_fail("self-gravity requires gravity_update_rate=1 and gravity_macro_dt=0");
                if (fmm.tree_dimension!=3 ||
                    (!fmm.adaptive_depth.enabled && fmm.tree_depth < 2))
                    clump_fail("self-gravity requires depth>=2 and explicit 3D FMM topology");
                if (fmm.enable_global_projection) clump_fail("self-gravity requires enable_global_projection=false");
                if (fmm.direct_threshold_leaf != 0 || fmm.direct_threshold_internal != 0)
                    clump_fail("self-gravity requires direct thresholds 0/0 to preserve M2L routing");
            }
        }

        // The released multi-GPU RESPA slice deliberately has a narrow
        // contract: isolated DEM/SPH with BFFMM only. Other force methods and the
        // distributed viscous-damping path have different force histories.
        if (config_.damping.enabled &&
            config_.damping.type == DampingType::VISCOUS) {
            std::cerr << "ERROR: multi-GPU does not support global viscous damping"
                      << std::endl;
            return false;
        }
        if (config_.termination == TerminationCondition::ENERGY_THRESHOLD) {
            std::cerr << "ERROR: multi-GPU does not support energy termination"
                      << std::endl;
            return false;
        }
        if (config_.acceleration.enable_self_gravity &&
            (config_.acceleration.gravity_method != SelfGravityMethod::BFFMM)) {
            std::cerr << "ERROR: multi-GPU self-gravity requires "
                         "unified BFFMM only" << std::endl;
            return false;
        }

        for (int wall = 0; wall < config_.num_walls; ++wall)
            if (config_.walls[wall].enabled && config_.walls[wall].type == WallType::BONDED_RECT)
                throw std::invalid_argument("Distributed bonded-wall dynamics and cooperative checkpoint state are unsupported");
        resumed_ = config_.io.checkpoint_resume_from[0] != '\0';
        // Cooperative stop can checkpoint even when periodic writes are off.
        // Distributed SPH does not persist/reconstruct shearing-box state.
        if (sph_mode_ && config_.shear_box.enabled)
            throw std::invalid_argument(
                "Distributed SPH cooperative checkpoints require isolated SPH; shearing boxes are unsupported");
        if (resumed_ && config_.io.checkpoint_force_fresh)
            throw std::invalid_argument("checkpoint force_fresh conflicts with resume_from");
        const ValidationResult input_validation = resumed_
            ? validate_checkpoint_file(config_.io.checkpoint_resume_from, config_)
            : validate_particle_file(config_.io.input_particle_path, config_);
        if (!input_validation.success)
            throw std::runtime_error("Invalid multi-GPU input: " + input_validation.error_message);
        if (resumed_ && sph_mode_) validate_sph_checkpoint_fields(config_.io.checkpoint_resume_from);

        // Check multi-GPU configuration
        if (!config_.multigpu.enabled) {
            std::cerr << "ERROR: Multi-GPU mode not enabled in config" << std::endl;
            return false;
        }

        // Query available GPUs
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count < 1) {
            std::cerr << "ERROR: Multi-GPU requires at least 1 GPU, found "
                      << device_count << std::endl;
            return false;
        }

        num_gpus_ = std::min(device_count, config_.multigpu.max_gpus);
        diagnostic_device_layout_ = describe_visible_device_layout();
        std::cout << "Using " << num_gpus_ << " GPUs" << std::endl;
        std::cout << "  Device layout: " << diagnostic_device_layout_ << std::endl;

        // Print GPU info
        for (int g = 0; g < num_gpus_; ++g) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, g));
            std::cout << "  GPU " << g << ": " << prop.name << std::endl;
        }

        // Initialize domain manager
        domain_params_ = DomainManager::initialize(config_, num_gpus_, gpu_states_);
        morton_root_.initialize(num_gpus_);
        if(config_.clump.num_instances>0&&config_.acceleration.enable_self_gravity&&
            config_.acceleration.unified_fmm.adaptive_depth.enabled)
            clump_depth_census_.initialize(num_gpus_);
        morton_balance_.initialize(num_gpus_);
        morton_work_collectors_.resize(num_gpus_);
        d_sph_hydro_owned_offsets_.resize(num_gpus_, nullptr);
        sph_hydro_offset_capacities_.resize(num_gpus_, 0);
        sph_hydro_offsets_valid_.resize(num_gpus_, 0);
        common_metrics_.resize(num_gpus_);
        common_depth_ceilings_.resize(num_gpus_);
        clump_depth_errors_.resize(num_gpus_);
        common_work_totals_.resize(num_gpus_);

        // Initialize communicators
        communicators_.resize(num_gpus_);
        contacts_.resize(num_gpus_);
        sph_damage_workspaces_.resize(num_gpus_);

        // Initialize global managers before splitting state (per-GPU init below)
        AccelerationManager::initialize(config_);

        // Load particles on GPU 0 first
        CUDA_CHECK(cudaSetDevice(0));
        SimulationState global_state = resumed_
            ? load_checkpoint(config_.io.checkpoint_resume_from, config_)
            : initialize_system(config_);
        if(config_.shear_box.enabled && config_.clump.num_instances>0)
            ShearBoxManager::initialize(global_state,config_);
        ClumpStateSoA initial_clump_state;
        ClumpSegmentOffsets initial_clump_segments;
        try {
            if (resumed_ && !sph_mode_) {
                restore_dem_clump_checkpoint(loaded_checkpoint_path(),global_state);
                canonicalize_particle_clump_order(global_state.particles,&global_state.contact_history);
                stage_dem_checkpoint_contacts(global_state);
            }
            if (!sph_mode_ && !resumed_) {
                const int capacity=std::max(1,config_.clump.num_instances);
                initial_clump_state=allocate_clump_state(capacity);
                initial_clump_segments=allocate_clump_segments(capacity);
                initialize_clumps(config_,global_state.material_database,global_state.particles,
                                  initial_clump_state,initial_clump_segments);
            }
            capture_and_exclude_dead_rows(global_state);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            if (!sph_mode_ && !resumed_) {
                free_clump_state(initial_clump_state);
                free_clump_segments(initial_clump_segments);
            }
            free_simulation_state(global_state);
            return false;
        }
        // initialize_system necessarily saw the on-disk total. Recompute the
        // DEM stability limit on the live canonical prefix only.
        if (!sph_mode_) {
            const MaterialDatabase materials = load_material_database(
                config_.material_database.path, config_.dem_damping_model);
            auto timestep_config = config_;
            // Resume keeps the saved base step; validate it against physical
            // limits rather than reapplying the legacy automatic safety factor.
            if (resumed_ && timestep_config.time.dt == 0.0)
                timestep_config.time.dt = global_state.dt;
            const double allowed_dt = calculate_critical_timestep(
                timestep_config, global_state.particles, materials,
                global_state.particles.num_particles);
            if (!resumed_) global_state.dt = allowed_dt;
            else validate_dem_checkpoint_timestep(global_state.dt, allowed_dt, config_.time.dt);
        }
        if (!sph_mode_) {
            const auto radius_bounds = derive_particle_radius_bounds(
                global_state.particles, global_state.particles.num_particles);
            config_.particle.radius_min = radius_bounds.first;
            config_.particle.radius_max = radius_bounds.second;
        }
        dt_ = global_state.dt;
        global_state.dt = dt_;
        if (resumed_) checkpoint_cadence_step_ = global_state.current_step;
        const int32_t global_particle_count = global_state.particles.num_particles;

        if (sph_mode_) {
            // Match the single-GPU initialization exactly once, before the
            // state is partitioned.  Distribution below preserves every SPH
            // row, including deterministic global-ID flaw samples.
            if (!resumed_) {
                initialize_sph_properties(global_state, config_);
                initialize_sph_strength(global_state, config_);
                initialize_sph_damage(global_state, config_, global_particle_count);
            }
            update_sph_support_extent(global_state, config_,
                                      global_particle_count, 0);
            sph_global_h_max_ = compute_sph_h_max(
                global_state, global_particle_count, 0);
        }

        // Wall manager uses global state for contact history + PBC config
        WallManager::initialize(global_state, config_);
        if (config_.shear_box.enabled) {
            // Convert input peculiar velocities once, then canonicalize the
            // physical box before ownership keys and initial scatter.
            if(config_.clump.num_instances<=0)ShearBoxManager::initialize(global_state, config_);
            ShearBoxManager::wrap(global_state, config_, WallManager::get_pbc_config(),
                                  global_state.current_time,
                                  global_state.particles.num_local);
        }

        // Reuse the exact ownership geometry before assigning restart rows.
        if (resumed_) restore_rcb_ownership_checkpoint(loaded_checkpoint_path(), global_state);
        // Distribute particles across GPUs
        if (resumed_ && adaptive_fmm_depth_checkpoint_requested(config_))
            restore_common_depth_checkpoint(loaded_checkpoint_path(), global_state);
        DomainManager::distribute_particles(global_state, gpu_states_, domain_params_, config_);
        if (initial_clump_state.num_clumps > 0 || !restored_bodies_.empty()) {
            CUDA_CHECK(cudaSetDevice(0));
            distributed_clumps_.initialize(initial_clump_state,global_state.particles,gpu_states_,domain_params_,
                resumed_ ? &restored_bodies_ : nullptr,&config_,global_state.current_time,
                resumed_&&restored_owner_ranks_==static_cast<uint32_t>(num_gpus_),
                &WallManager::get_pbc_config());
            if(config_.shear_box.enabled && !(2*distributed_clumps_.max_diameter()<
                std::min(double(WallManager::get_pbc_config().box_length_x),double(WallManager::get_pbc_config().box_length_y))*(1-1.e-6)))
                throw std::runtime_error("Shear clumps require twice maximum body diameter below both planar box lengths");
        }
        CUDA_CHECK(cudaSetDevice(0));
        if (!sph_mode_ && !resumed_) {
            free_clump_state(initial_clump_state);
            free_clump_segments(initial_clump_segments);
        }
        // The initial scatter preserves clump_id.  In SPH it is the exact
        // SPHSOL Part ID; in DEM it retains the existing clump semantics.
        // Distribution owns independent per-device copies.  Retaining the
        // complete staging state on GPU 0 defeats multi-GPU memory scaling and
        // is especially expensive for the expanded SPH state.
        free_simulation_state(global_state);

        std::vector<int> device_ids(static_cast<size_t>(num_gpus_));
        std::iota(device_ids.begin(), device_ids.end(), 0);
        next_yorp_checks_.assign(num_gpus_, config_.acceleration.yorp.start_time);
        if (resumed_ && config_.acceleration.yorp.enabled) {
            std::vector<cudaStream_t> streams;
            for (const auto& rank : gpu_states_) streams.push_back(rank.compute_stream);
            const auto runtime = restore_yorp_runtime_checkpoint(loaded_checkpoint_path(),
                gpu_states_[0].local_state, config_, device_ids, streams, &next_yorp_checks_);
            restored_yorp_runtime_ = true;
            if (config_.acceleration.yorp.adaptive && !runtime.empty())
                rewind_yorp_diagnostic_csv(std::string(config_.io.output_directory)+"/adaptive_yorp_multigpu.csv",
                    "metric_time", runtime[0].device_state.diagnostic.time);
        }
        if (!phase_coordinator_.initialize(device_ids)) {
            std::cerr << "ERROR: Failed to initialize multi-GPU phase coordinator: "
                      << phase_coordinator_.error_message() << std::endl;
            return false;
        }

        if (config_.acceleration.yorp.adaptive) {
            if (!distributed_yorp_.initialize(device_ids, &phase_coordinator_)) {
                std::cerr << "ERROR: Failed to initialize distributed YORP: "
                          << distributed_yorp_.error_message() << std::endl;
                return false;
            }
            for (int32_t gpu = 0; gpu < num_gpus_; ++gpu) {
                if (!distributed_yorp_.attach_manager(
                        gpu, AccelerationManager::yorp_manager_for_gpu(gpu))) {
                    std::cerr << "ERROR: Failed to attach YORP manager for GPU "
                              << gpu << std::endl;
                    return false;
                }
            }
        }

        if (!restored_non_morton_depth_ && num_gpus_ > 1 && !domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled &&
            !distributed_fmm_depth_.initialize(
                config_.acceleration.unified_fmm.adaptive_depth,
                num_gpus_, {})) {
            std::cerr << "ERROR: Failed to initialize distributed FMM-depth controller"
                      << std::endl;
            return false;
        }

        if(domain_params_.morton_enabled && (!resumed_ || !adaptive_fmm_depth_checkpoint_requested(config_))){
            for(const auto& state:gpu_states_)common_global_particles_+=state.subdomain.num_local_particles;
            common_fmm_command_.launch_depth=config_.acceleration.unified_fmm.adaptive_depth.enabled ? config_.acceleration.unified_fmm.adaptive_depth.min_depth : std::max(2,config_.acceleration.unified_fmm.tree_depth);
            common_launch_depth_=common_fmm_command_.launch_depth;
            if(config_.acceleration.unified_fmm.adaptive_depth.enabled){
                if(!common_fmm_depth_.initialize(config_.acceleration.unified_fmm.adaptive_depth,1,
                                                {common_launch_depth_}))
                    throw std::runtime_error("common FMM depth initialization failed");
                common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
            }
        }

        let_comms_ = new LETCommunicator[num_gpus_];

        // Initialize per-GPU resources
        #pragma omp parallel num_threads(num_gpus_)
        {
            int gpu_id = omp_get_thread_num();
            CUDA_CHECK(cudaSetDevice(gpu_id));

            GPUDomainState& state = gpu_states_[gpu_id];
            state.local_state.dt = dt_;

            // Copy config to this GPU's constant memory
            copy_config_to_device(config_);
            WallManager::copy_to_current_device();

            if (sph_mode_) {
                initialize_sph_timestep(state.subdomain.capacity);
                CUDA_CHECK(cudaMalloc(&d_sph_hydro_owned_offsets_[gpu_id],
                                      (static_cast<size_t>(state.subdomain.capacity) + 1) *
                                          sizeof(uint64_t)));
                sph_hydro_offset_capacities_[gpu_id] =
                    static_cast<size_t>(state.subdomain.capacity) + 1;
                CUDA_CHECK(cudaMalloc(&sph_damage_workspaces_[gpu_id].qdot,
                                      static_cast<size_t>(state.subdomain.capacity) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&sph_damage_workspaces_[gpu_id].qcap,
                                      static_cast<size_t>(state.subdomain.capacity) * sizeof(float)));
            }

            // Initialize neighbor search after SPH support extents are valid.
            NeighborSearch::initialize(config_, state.local_state);

            int32_t capacity = state.subdomain.capacity;

            // Initialize LET communicator for multi-GPU self-gravity
            int max_nodes = std::max(capacity * 4, 65536);
            let_comms_[gpu_id].initialize(
                gpu_id, num_gpus_,
                capacity, max_nodes,
                2,  // start_level
                config_.acceleration.unified_fmm.theta_far,
                config_.multigpu.let_use_one_shot_push,
                &phase_coordinator_,
                config_.multigpu.shared_bucket_min_saved_bytes,
                config_.multigpu.max_shared_buckets,
                domain_params_.morton_enabled
            );
            body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_partition(domain_params_);
            if(distributed_clumps_.enabled())
                body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_clump_exclusion_diameter(distributed_clumps_.max_diameter());
            if (num_gpus_ > 1 && !domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled) {
                body_fitted_fmm::UnifiedFMMDispatcher::
                    set_adaptive_fmm_depth_control_mode(
                        body_fitted_fmm::AdaptiveFMMDepthControlMode::
                            ExternalDistributed,
                        state.compute_stream);
            }

            // Allocate index mapping arrays (must match particle capacity)
            CUDA_CHECK(cudaMalloc(&state.d_owner_gpu, capacity * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&state.d_is_halo, capacity * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&state.d_is_migrating, capacity * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&state.d_local_to_global, capacity * sizeof(int32_t)));

            int32_t max_contacts = capacity * kContactCapacityPerParticle;
            NeighborSearch::allocate_contact_buffer(contacts_[gpu_id], max_contacts);

            // Allocate contact history buffers for this GPU (was missing — caused
            // NULL-pointer write in compute_contact_forces_multigpu_kernel at
            // contact_history[hist_offset + new_slot].partner_global_id = ...)
            allocate_contact_history(state.local_state.contact_history, capacity, resumed_&&!sph_mode_ ? restored_contact_stride_ : 16);
            allocate_wall_contact_history(state.local_state.wall_contact_history, capacity, resumed_&&!sph_mode_ ? restored_wall_stride_ : 16);

            const int32_t contact_stride = state.local_state.contact_history.max_contacts_per_particle;
            const int32_t wall_stride = state.local_state.wall_contact_history.max_contacts_per_particle;
            const int32_t layout_blocks = (capacity + 255) / 256;
            initialize_migration_history_layout_kernel<<<layout_blocks, 256>>>(
                state.local_state.particles.contact_history_offset,
                state.local_state.particles.wall_history_offset,
                capacity, contact_stride, wall_stride);
            CUDA_CHECK(cudaMemset(state.local_state.particles.num_contacts, 0,
                                  capacity * sizeof(int32_t)));
            CUDA_CHECK(cudaMemset(state.local_state.particles.num_wall_contacts, 0,
                                  capacity * sizeof(int32_t)));
            CUDA_CHECK(cudaGetLastError());

            if (resumed_ && !sph_mode_) restore_dem_rank_histories(state);

            // Migration payload rows must use the same configured strides as
            // the persistent histories they restore into.
            communicators_[gpu_id].initialize(
                gpu_id, num_gpus_, capacity,
                contact_stride, wall_stride);
            let_comms_[gpu_id].set_timing_sample_interval(
                static_cast<uint32_t>(config_.multigpu.rcb_calibration_sample_interval));
            for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                let_comms_[gpu_id].set_peer_link_class(
                    peer, communicators_[gpu_id].link_class_to(peer));
            }

            #pragma omp critical
            {
                std::cout << "GPU " << gpu_id << " initialized with "
                          << state.subdomain.num_local_particles << " particles" << std::endl;
            }
        }

        if(resumed_&&!sph_mode_){restored_bodies_.clear();restored_exact_rows_.clear();restored_contacts_.clear();
            restored_contact_counts_.clear();restored_contact_gids_.clear();restored_wall_rows_.clear();restored_wall_offsets_.clear();}

        if (sph_mode_) {
            if (!(std::isfinite(sph_global_h_max_) && sph_global_h_max_ > 0.0f)) {
                std::cerr << "ERROR: Multi-GPU SPH requires a positive smoothing length"
                          << std::endl;
                return false;
            }
            if (!resumed_) dt_ = get_initial_sph_timestep(config_);
            for (int32_t g = 0; g < num_gpus_; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                gpu_states_[g].local_state.dt = dt_;
                set_sph_h_max_global(sph_global_h_max_,
                                     gpu_states_[g].compute_stream);
                if (resumed_ && !config_.io.checkpoint_rebootstrap_forces) {
                    evaluate_sph_porosity_rates(gpu_states_[g].local_state, config_,
                                               gpu_states_[g].compute_stream);
                    CUDA_CHECK(cudaStreamSynchronize(gpu_states_[g].compute_stream));
                }
            }
        }

        // Convert after the final globally selected dt (including the SPH
        // branch) and then refresh every device's constant config.
        if (!sph_mode_ && config_.acceleration.gravity_macro_dt > 0.0) {
            const double ratio = config_.acceleration.gravity_macro_dt / dt_;
            const int64_t effective = std::max<int64_t>(
                1, static_cast<int64_t>(std::floor(ratio + 1.0e-12)));
            if (effective > std::numeric_limits<int>::max()) {
                std::cerr << "ERROR: gravity_macro_dt produces too many RESPA inner steps"
                          << std::endl;
                return false;
            }
            config_.acceleration.gravity_update_rate = static_cast<int>(effective);
            #pragma omp parallel for num_threads(num_gpus_)
            for (int32_t g = 0; g < num_gpus_; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                copy_config_to_device(config_);
            }
        }

        for (int32_t g = 0; g < num_gpus_; ++g) {
            const CoreAllocationTelemetry core = core_allocation_telemetry(gpu_states_[g]);
            const CommunicationAllocationTelemetry communication =
                communicators_[g].allocation_telemetry();
            const size_t core_bytes = core.particle_bytes + core.persistent_history_bytes
                                    + core.mapping_and_flag_bytes;
            const size_t communication_payload_bytes = communication.send_particle_bytes
                                                      + communication.recv_particle_bytes
                                                      + communication.send_history_bytes
                                                      + communication.recv_history_bytes
                                                      + communication.send_index_bytes;
            std::cout << "[memory] GPU " << g
                      << " init local=" << gpu_states_[g].subdomain.num_local_particles
                      << " core_capacity=" << core.particle_capacity
                      << " core_bytes=" << core_bytes
                      << " communication_payload_bytes=" << communication_payload_bytes
                      << " communication_workspace_bytes=" << communication.workspace_bytes
                      << " let_bytes=" << let_comms_[g].fmm_arena_bytes()
                      << std::endl;
        }

        // RCB scratch follows independent per-GPU particle capacities.
        std::vector<int32_t> rcb_capacities(num_gpus_);
        for (int g = 0; g < num_gpus_; ++g) {
            rcb_capacities[g] = gpu_states_[g].subdomain.capacity;
        }
        if(!domain_params_.morton_enabled)balancer_.initialize(num_gpus_, rcb_capacities,
                             config_.multigpu.rcb_alpha, config_.multigpu.rcb_beta,
                             config_.multigpu.rcb_gamma, config_.multigpu.rcb_num_bins);
        rebalance_ctl_.configure(config_.multigpu.rebalance_threshold,
                                 config_.multigpu.rebalance_min_interval);
        rebalance_mailbox_.initialize(num_gpus_);
        if (restored_pending_rebalance_ >= 0) rebalance_mailbox_.schedule(restored_pending_rebalance_);
        rcb_calibrator_.set_half_life(config_.multigpu.rcb_calibration_half_life);
        std::vector<RCBWorkKey> required_calibration{{RCBWorkPhase::Integration,
            RCBFmmMode::NotApplicable, RCBLinkClass::PCIe}};
        if (sph_mode_) required_calibration.push_back({RCBWorkPhase::ContactPair,
            RCBFmmMode::NotApplicable, RCBLinkClass::PCIe});
        rcb_calibrator_.set_enabled_keys(required_calibration);
        rcb_sample_mailbox_.initialize(num_gpus_);
        latest_contact_counts_.assign(num_gpus_, 0);
        fmm_timing_diagnostics_.assign(num_gpus_, {});
        fmm_runtime_timing_dropped_.assign(num_gpus_, 0);
        fmm_diagnostic_rows_dropped_.assign(num_gpus_, 0);

        // Per-GPU timing events for the compute-heavy region.
        step_start_events_.resize(static_cast<size_t>(num_gpus_) * kTimingEventSlots);
        step_stop_events_.resize(static_cast<size_t>(num_gpus_) * kTimingEventSlots);
        migration_transfer_events_.resize(num_gpus_);
        halo_transfer_events_.resize(num_gpus_);
        const size_t calibration_event_count =
            static_cast<size_t>(num_gpus_) * kTimingEventSlots
            * static_cast<size_t>(CalibrationSegment::Count);
        calibration_start_events_.resize(calibration_event_count);
        calibration_stop_events_.resize(calibration_event_count);
        for (int g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            for (int32_t slot = 0; slot < kTimingEventSlots; ++slot) {
                size_t index = static_cast<size_t>(g) * kTimingEventSlots + slot;
                CUDA_CHECK(cudaEventCreate(&step_start_events_[index]));
                CUDA_CHECK(cudaEventCreate(&step_stop_events_[index]));
                for (uint8_t raw = 0;
                     raw < static_cast<uint8_t>(CalibrationSegment::Count); ++raw) {
                    const auto segment = static_cast<CalibrationSegment>(raw);
                    const size_t calibration_index =
                        calibration_event_index(g, slot, segment);
                    CUDA_CHECK(cudaEventCreate(&calibration_start_events_[calibration_index]));
                    CUDA_CHECK(cudaEventCreate(&calibration_stop_events_[calibration_index]));
                }
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&migration_transfer_events_[g],
                                                 cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&halo_transfer_events_[g],
                                                 cudaEventDisableTiming));
        }

        initialized_ = true;

        // Phase 4: Allocate pinned memory + events for async contact count readback
        h_pinned_num_contacts_.resize(num_gpus_);
        h_pinned_sph_owned_degrees_.resize(static_cast<size_t>(num_gpus_) * kTimingEventSlots);
        contact_count_events_.resize(num_gpus_);
        contact_count_ready_.resize(num_gpus_, false);
        for (int g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMallocHost(&h_pinned_num_contacts_[g], sizeof(int32_t)));
            *h_pinned_num_contacts_[g] = 0;
            for (int slot = 0; slot < kTimingEventSlots; ++slot) {
                const size_t index = static_cast<size_t>(g) * kTimingEventSlots + slot;
                CUDA_CHECK(cudaMallocHost(&h_pinned_sph_owned_degrees_[index], sizeof(uint64_t)));
                *h_pinned_sph_owned_degrees_[index] = 0;
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&contact_count_events_[g],
                                                 cudaEventDisableTiming));
        }

        std::cout << "\n========== Multi-GPU Initialization Complete ==========" << std::endl;

        return true;
    }

    void run() {
        if (!initialized_) {
            std::cerr << "ERROR: Simulation not initialized" << std::endl;
            return;
        }

        std::cout << "\n========== Starting Multi-GPU Simulation ===========" << std::endl;

        auto wall_start = std::chrono::high_resolution_clock::now();
        const int64_t timing_start_step = gpu_states_[0].local_state.current_step;
        const double timing_start_time = gpu_states_[0].local_state.current_time;
        bool timing_window_started = false, timing_window_finished = false;
        bool timing_segment_active = false;
        int64_t timing_window_start_step = timing_start_step;
        int64_t timing_window_end_step = timing_start_step;
        double timing_window_wall_seconds = 0.0;
        std::chrono::high_resolution_clock::time_point timing_segment_start;

        const float dem_halo_width = config_.particle.radius_max * 2.0f *
                                     config_.multigpu.halo_width_factor;
        const int respa_k = config_.acceleration.gravity_update_rate;
        const bool respa_enabled = sph_mode_ || (config_.acceleration.enable_self_gravity &&
            respa_k > 1);
        std::vector<SPHTimestepResult> sph_dt_candidates(
            static_cast<size_t>(num_gpus_));
        SPHTimestepResult global_sph_timestep_diagnostic;
        std::vector<float> sph_h_max_candidates(static_cast<size_t>(num_gpus_), 0.0f);
        std::vector<ContactHistoryOverflow> contact_history_errors(num_gpus_);
        bool contact_history_overflow = false;
        bool collective_stop = false;
        bool checkpoint_failed = false;
        std::exception_ptr worker_failure;
        std::mutex worker_failure_mutex;
        const bool prepare_distributed_peer_routes = num_gpus_ > 1 &&
            domain_params_.morton_enabled &&
            config_.acceleration.enable_self_gravity &&
            config_.acceleration.gravity_method == SelfGravityMethod::BFFMM;

        // OpenMP parallel region - one thread per GPU
        #pragma omp parallel num_threads(num_gpus_)
        {
            try {
            int gpu_id = omp_get_thread_num();
            ContactHistoryWorkspace contact_history_workspace;
            if (prepare_distributed_peer_routes) {
                // The initialization region uses different OpenMP workers.
                // Prepare routes here so copy_peer_payload sees this worker's
                // thread-local cache before its first distributed FMM step.
                try {
                    CUDA_CHECK(cudaSetDevice(gpu_id));
                    #ifdef CUDADEM_ENABLE_PROFILING
                    nvtxRangePushA("Peer Transfer: Prepare Routes");
                    #endif
                    try {
                        for (int source = 0; source < num_gpus_; ++source)
                            if (source != gpu_id)
                                prepare_peer_payload_route(gpu_id, source);
                    } catch (...) {
                        #ifdef CUDADEM_ENABLE_PROFILING
                        nvtxRangePop();
                        #endif
                        throw;
                    }
                    #ifdef CUDADEM_ENABLE_PROFILING
                    nvtxRangePop();
                    #endif
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_failure_mutex);
                    if (!worker_failure) worker_failure = std::current_exception();
                }
                // Finish each destination's source-order mapping choices
                // before workers cache the reverse routes used by FMM's
                // direct-flag publication (destination peer <- source gpu).
                #pragma omp barrier
                try {
                    #ifdef CUDADEM_ENABLE_PROFILING
                    nvtxRangePushA("Peer Transfer: Prepare Reverse Routes");
                    #endif
                    try {
                        for (int destination = 0; destination < num_gpus_; ++destination)
                            if (destination != gpu_id)
                                prepare_peer_payload_route(destination, gpu_id);
                    } catch (...) {
                        #ifdef CUDADEM_ENABLE_PROFILING
                        nvtxRangePop();
                        #endif
                        throw;
                    }
                    #ifdef CUDADEM_ENABLE_PROFILING
                    nvtxRangePop();
                    #endif
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_failure_mutex);
                    if (!worker_failure) worker_failure = std::current_exception();
                }
                #pragma omp barrier
                if (worker_failure) std::rethrow_exception(worker_failure);
            } else {
                CUDA_CHECK(cudaSetDevice(gpu_id));
            }

            GPUDomainState& my_state = gpu_states_[gpu_id];
            Communicator& comm = communicators_[gpu_id];
            ContactListBuffer& my_contacts = contacts_[gpu_id];

            cudaStream_t stream = my_state.compute_stream;

            int block_size = config_.gpu.block_size;
            double dt = dt_;
            double sph_next_output = next_sph_output_time(my_state.local_state.current_time, config_.time.start_time, config_.time.output_step);
            int respa_inner_remaining = 0;
            int respa_macro_inner_count = 0;
            double respa_macro_span = 0.0;
            double sph_macro_end = 0.0;
            bool respa_macro_closing = false;
            uint64_t local_gravity_epoch = 0;
            bool force_bootstrap_pending = !resumed_ ||
                config_.io.checkpoint_rebootstrap_forces ||
                sph_stage_transition_at(config_, my_state.local_state.current_time);
            bool initial_bootstrap_pending = force_bootstrap_pending;
            int last_sph_stage = -2;
            int64_t last_timing_epoch_evaluated = -1;
            int64_t last_calibration_epoch_evaluated = -1;
            uint64_t particle_exchange_epoch = 1;
            std::array<int64_t, kTimingEventSlots> timing_slot_epochs;
            timing_slot_epochs.fill(-1);
            std::array<bool, kTimingEventSlots> timing_slot_excluded_from_rcb{};
            std::array<int64_t, kTimingEventSlots> calibration_slot_epochs;
            calibration_slot_epochs.fill(-1);
            std::array<bool, kTimingEventSlots> calibration_slot_sampled{};
            std::array<int32_t, kTimingEventSlots> calibration_owned_units{};
            std::array<uint64_t, kTimingEventSlots> calibration_contact_units{};
            std::array<bool, kTimingEventSlots> sph_degree_readback{};
            std::array<double, kTimingEventSlots> calibration_halo_units{};
            std::array<double, kTimingEventSlots> calibration_halo_bytes{};
            std::array<RCBLinkClass, kTimingEventSlots> calibration_halo_links{};
            std::map<int64_t, std::vector<RCBActiveSample>> pending_calibration_samples;

            auto calibration_event = [&](bool start, int32_t slot,
                                         CalibrationSegment segment,
                                         cudaStream_t event_stream) {
                if (slot < 0 || !calibration_slot_sampled[slot]) return;
                const size_t index = calibration_event_index(gpu_id, slot, segment);
                CUDA_CHECK(cudaEventRecord(start ? calibration_start_events_[index]
                                                 : calibration_stop_events_[index],
                                           event_stream));
            };

            auto calibration_elapsed_seconds = [&](int32_t slot,
                                                    CalibrationSegment segment) {
                const size_t index = calibration_event_index(gpu_id, slot, segment);
                float milliseconds = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds,
                                                calibration_start_events_[index],
                                                calibration_stop_events_[index]));
                return static_cast<double>(milliseconds) * 1.0e-3;
            };

            auto append_sample = [&](std::vector<RCBActiveSample>& samples,
                                     int64_t epoch, RCBWorkPhase phase,
                                     RCBFmmMode mode, RCBLinkClass link,
                                     double seconds, double units, bool valid = true) {
                RCBActiveSample sample;
                sample.epoch = epoch;
                sample.key = {phase, mode, link};
                sample.active_seconds = seconds;
                sample.work_units = units;
                sample.valid = valid && seconds > 0.0 && units > 0.0;
                samples.push_back(sample);
            };

            auto harvest_timing_samples = [&]() {
                std::vector<int32_t> completed_slots;
                for (int32_t slot = 0; slot < kTimingEventSlots; ++slot) {
                    if (timing_slot_epochs[slot] < 0) continue;
                    size_t index = static_cast<size_t>(gpu_id) * kTimingEventSlots + slot;
                    cudaError_t status = cudaEventQuery(step_stop_events_[index]);
                    if (status == cudaErrorNotReady) continue;
                    CUDA_CHECK(status);
                    float milliseconds = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(
                        &milliseconds, step_start_events_[index], step_stop_events_[index]));
                    if (!timing_slot_excluded_from_rcb[slot]) {
                        rebalance_mailbox_.publish(
                            gpu_id, timing_slot_epochs[slot], milliseconds);
                    }
                    completed_slots.push_back(slot);
                }
                if (completed_slots.empty()) return;

                // The whole-step stop event follows all compute-stream work, so
                // these query-only harvests cannot introduce a new wait.
                if (config_.acceleration.enable_self_gravity &&
                    config_.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
                    std::vector<body_fitted_fmm::BFFMMActiveSample> fmm_samples;
                    body_fitted_fmm::UnifiedFMMDispatcher::poll_active_samples(fmm_samples);
                    for (const auto& fmm_sample : fmm_samples) {
                        pending_calibration_samples[fmm_sample.sample.epoch].push_back(
                            fmm_sample.sample);
                    }
                }
                std::array<LETTimingRecord, 32> let_records{};
                const size_t num_let_records =
                    let_comms_[gpu_id].poll_timing_records(let_records.data(),
                                                           let_records.size());
                // LET's wire epoch is communicator-local. Associate completed
                // records with the newest simulation epoch whose stop event
                // proves that the corresponding remote evaluation has consumed them.
                int32_t newest_slot = completed_slots.front();
                for (int32_t slot : completed_slots) {
                    if (calibration_slot_epochs[slot] >
                        calibration_slot_epochs[newest_slot]) newest_slot = slot;
                }
                auto& let_samples = pending_calibration_samples[
                    calibration_slot_epochs[newest_slot]];
                for (size_t i = 0; i < num_let_records; ++i) {
                    const LETTimingRecord& record = let_records[i];
                    const RCBLinkClass link = calibration_link_class(record.link_class);
                    if (record.phase == LETTimingPhase::ExportKernel) {
                        append_sample(let_samples, calibration_slot_epochs[newest_slot],
                                      RCBWorkPhase::LetSource,
                                      RCBFmmMode::NotApplicable, link,
                                      static_cast<double>(record.milliseconds) * 1.0e-3,
                                      static_cast<double>(record.node_records)
                                        + static_cast<double>(record.particle_records),
                                      record.valid_for_calibration);
                        let_samples.back().allocation_grew = record.grew;
                        let_samples.back().retried = record.retried;
                    } else {
                        append_sample(let_samples, calibration_slot_epochs[newest_slot],
                                      RCBWorkPhase::LinkTransfer,
                                      RCBFmmMode::NotApplicable, link,
                                      static_cast<double>(record.milliseconds) * 1.0e-3,
                                      static_cast<double>(record.bytes),
                                      record.valid_for_calibration);
                        let_samples.back().allocation_grew = record.grew;
                        let_samples.back().retried = record.retried;
                    }
                }

                for (int32_t slot : completed_slots) {
                    const int64_t calibration_epoch = calibration_slot_epochs[slot];
                    auto& samples = pending_calibration_samples[calibration_epoch];
                    if (timing_slot_excluded_from_rcb[slot]) {
                        // FMM and LET telemetry is harvested independently of
                        // the host segment sampling flag above.  Publish an
                        // empty cohort record so ordered mailbox progress is
                        // preserved, but no A/B trial work reaches calibration.
                        samples.clear();
                    } else if (calibration_slot_sampled[slot]) {
                        const double integration_seconds =
                            calibration_elapsed_seconds(slot, CalibrationSegment::Verlet1)
                          + calibration_elapsed_seconds(slot, CalibrationSegment::WallForce)
                          + calibration_elapsed_seconds(slot, CalibrationSegment::Verlet2);
                        append_sample(samples, calibration_epoch,
                                      RCBWorkPhase::Integration,
                                      RCBFmmMode::NotApplicable, RCBLinkClass::PCIe,
                                      integration_seconds,
                                      calibration_owned_units[slot]);
                        append_sample(samples, calibration_epoch,
                                      RCBWorkPhase::Contacts,
                                      RCBFmmMode::NotApplicable, RCBLinkClass::PCIe,
                                      calibration_elapsed_seconds(
                                          slot, CalibrationSegment::NeighborSearch),
                                      calibration_owned_units[slot]);
                        const double physics_seconds = sph_mode_
                            ? calibration_elapsed_seconds(slot, CalibrationSegment::SphDensity)
                            + calibration_elapsed_seconds(slot, CalibrationSegment::SphConstitutive)
                            + calibration_elapsed_seconds(slot, CalibrationSegment::SphForceHistory)
                            : calibration_elapsed_seconds(slot, CalibrationSegment::ContactForce);
                        const uint64_t physics_units = sph_mode_ && sph_degree_readback[slot]
                            ? calibration_contact_units[slot] +
                              *h_pinned_sph_owned_degrees_[
                                  static_cast<size_t>(gpu_id) * kTimingEventSlots + slot]
                            : calibration_contact_units[slot];
                        append_sample(samples, calibration_epoch,
                                      RCBWorkPhase::ContactPair,
                                      RCBFmmMode::NotApplicable, RCBLinkClass::PCIe,
                                      physics_seconds, static_cast<double>(physics_units));
                        const double halo_lifecycle_seconds =
                            calibration_elapsed_seconds(slot, CalibrationSegment::HaloPack)
                          + calibration_elapsed_seconds(slot, CalibrationSegment::HaloUnpack);
                        append_sample(samples, calibration_epoch,
                                      RCBWorkPhase::HaloLifecycle,
                                      RCBFmmMode::NotApplicable,
                                      calibration_halo_links[slot],
                                      halo_lifecycle_seconds,
                                      calibration_halo_units[slot]);
                        append_sample(samples, calibration_epoch,
                                      RCBWorkPhase::LinkTransfer,
                                      RCBFmmMode::NotApplicable,
                                      calibration_halo_links[slot],
                                      calibration_elapsed_seconds(
                                          slot, CalibrationSegment::HaloTransfer),
                                      calibration_halo_bytes[slot]);
                    }
                    rcb_sample_mailbox_.publish(gpu_id, calibration_epoch, samples);
                    pending_calibration_samples.erase(calibration_epoch);
                    timing_slot_epochs[slot] = -1;
                    timing_slot_excluded_from_rcb[slot] = false;
                    calibration_slot_epochs[slot] = -1;
                    calibration_slot_sampled[slot] = false;
                    sph_degree_readback[slot] = false;
                }
            };

            auto harvest_distributed_fmm_samples = [&]() {
                if (!(config_.acceleration.unified_fmm.adaptive_depth.enabled && num_gpus_ > 1 && !domain_params_.morton_enabled))
                    return;
                std::vector<body_fitted_fmm::MultiGPUFMMGravityEpochTiming> samples;
                uint64_t runtime_dropped = 0;
                body_fitted_fmm::UnifiedFMMDispatcher::
                    poll_multigpu_gravity_epoch_timings(samples, &runtime_dropped);
                if (runtime_dropped > fmm_runtime_timing_dropped_[gpu_id]) {
                    const uint64_t failure_epoch = static_cast<uint64_t>(
                        std::max<uint64_t>(1, gravity_evaluation_epoch_));
                    distributed_fmm_depth_.discard_timing_before(failure_epoch);
                    distributed_fmm_depth_.freeze(
                        "distributed FMM timing ring dropped an epoch",
                        failure_epoch);
                }
                fmm_runtime_timing_dropped_[gpu_id] = std::max(
                    fmm_runtime_timing_dropped_[gpu_id], runtime_dropped);
                for (const auto& sample : samples) {
                    const uint64_t epoch = sample.gravity_evaluation_epoch;
                    // Empty ownership participates in LET epochs but does
                    // not initialize a local FMM tree.  Its capture is
                    // therefore intentionally depth-invalid until a later
                    // rebalance gives it owned particles and it completes its
                    // own startup probe.  This is not telemetry corruption:
                    // leave the controller unready rather than freezing all
                    // selected depths.
                    if (sample.owned_particles == 0 && !sample.depth.valid)
                        continue;
                    if (!sample.valid || !sample.depth.valid) {
                        distributed_fmm_depth_.discard_timing_before(epoch + 1);
                        distributed_fmm_depth_.freeze(
                            "invalid distributed FMM epoch telemetry", epoch);
                        continue;
                    }
                    auto state = distributed_fmm_depth_.state_for_gpu(gpu_id);
                    if (state.selected_depth == 0) {
                        distributed_fmm_depth_.set_initial_depth(
                            gpu_id, sample.depth.selected_depth,
                            sample.depth.effective_max_depth, epoch);
                        state = distributed_fmm_depth_.state_for_gpu(gpu_id);
                    }
                    const bool capacity_event = state.effective_max_depth != 0 &&
                        sample.depth.effective_max_depth !=
                            state.effective_max_depth;
                    if (capacity_event) {
                        distributed_fmm_depth_.set_effective_max_depth(
                            gpu_id, sample.depth.effective_max_depth, epoch);
                        state = distributed_fmm_depth_.state_for_gpu(gpu_id);
                    }

                    DistributedFMMDepthEpochTiming timing{};
                    timing.epoch = epoch;
                    timing.elapsed_ns = static_cast<uint64_t>(std::llround(
                        std::max(0.0, sample.elapsed_ms) * 1.0e6));
                    timing.let_bytes = sample.let_send_bytes + sample.let_recv_bytes;
                    timing.local_owned_particles = sample.owned_particles;
                    timing.telemetry_valid = sample.elapsed_ms > 0.0;
                    timing.capacity_event = capacity_event;
                    timing.command_trial_id = sample.depth.controller_epoch;
                    timing.launch_depth = sample.depth.launch_depth;
                    distributed_fmm_depth_.publish_timing(gpu_id, timing);

                    FMMTimingDiagnostic row{};
                    row.epoch = epoch;
                    row.gpu = gpu_id;
                    row.elapsed_ms = sample.elapsed_ms;
                    row.remote_evaluation_ms = sample.remote_evaluation_ms;
                    row.owned_particles = sample.owned_particles;
                    row.trial_id = sample.depth.controller_epoch;
                    row.launch_depth = sample.depth.launch_depth;
                    row.selected_depth = sample.depth.selected_depth;
                    row.candidate_depth = sample.depth.candidate_depth;
                    row.effective_max_depth = sample.depth.effective_max_depth;
                    row.let_node_records = sample.let_send_node_records
                                         + sample.let_recv_node_records;
                    row.let_particle_records = sample.let_send_particle_records
                                             + sample.let_recv_particle_records;
                    row.let_bytes = timing.let_bytes;
                    auto& rows = fmm_timing_diagnostics_[gpu_id];
                    if (rows.size() == kFMMTimingDiagnosticCapacity) {
                        rows.erase(rows.begin());
                        ++fmm_diagnostic_rows_dropped_[gpu_id];
                    }
                    rows.push_back(row);
                }
            };

            auto require_phase = [&](bool ok, const char* operation) {
                if (!ok) {
                    phase_coordinator_.abort();
                    throw std::runtime_error(
                        "GPU " + std::to_string(gpu_id) +
                        " phase coordination failed during " + operation +
                        ": " + phase_coordinator_.error_message());
                }
            };

            // Establish the distributed adaptive-YORP baseline before the
            // first increment.  All workers execute the same sparse protocol
            // epoch; directed acknowledgements protect its shared buffers.
            uint64_t yorp_observation_epoch = 1;
            double& next_yorp_check = next_yorp_checks_[gpu_id];
            bool local_adaptive_yorp_stop = false;
            double local_last_yorp_transition_time =
                -std::numeric_limits<double>::infinity();
            if (restored_yorp_runtime_ && config_.acceleration.yorp.adaptive &&
                my_state.local_state.current_time >= config_.acceleration.yorp.start_time) {
                AdaptiveYorpDiagnostic diagnostic{};
                if (!AccelerationManager::yorp_manager_for_gpu(gpu_id)->consume_adaptive_diagnostic(&diagnostic, stream))
                    throw std::runtime_error("Restored distributed adaptive YORP state is unavailable");
                local_adaptive_yorp_stop = diagnostic.fatal != 0 ||
                    diagnostic.state == static_cast<int32_t>(AdaptiveYorpState::COMPLETE);
                local_last_yorp_transition_time = diagnostic.transition_time;
                if (gpu_id == 0) adaptive_yorp_fatal_ = diagnostic.fatal != 0;
            }
            if (!restored_yorp_runtime_ && config_.acceleration.yorp.adaptive &&
                my_state.local_state.current_time + 1e-12 >= next_yorp_check) {
                require_phase(distributed_yorp_.observe(
                    gpu_id, my_state.local_state.particles,
                    my_state.subdomain.num_local_particles,
                    my_state.local_state.current_time, config_, block_size,
                    stream, yorp_observation_epoch++), "initial adaptive YORP observation");
                AdaptiveYorpDiagnostic diagnostic{};
                if (!AccelerationManager::yorp_manager_for_gpu(gpu_id)->
                        consume_adaptive_diagnostic(&diagnostic, stream)) {
                    throw std::runtime_error(
                        "distributed adaptive YORP diagnostic is unavailable");
                }
                local_adaptive_yorp_stop = diagnostic.fatal != 0 ||
                    diagnostic.state == static_cast<int32_t>(
                        AdaptiveYorpState::COMPLETE);
                local_last_yorp_transition_time = diagnostic.transition_time;
                if (gpu_id == 0) {
                    append_distributed_yorp_diagnostic(diagnostic);
                    adaptive_yorp_fatal_ = diagnostic.fatal != 0;
                }
                do {
                    next_yorp_check +=
                        config_.acceleration.yorp.check_interval_seconds;
                } while (next_yorp_check <= my_state.local_state.current_time);
            }

            auto apply_respa_half_kick = [&](double half_span) {
                const int32_t owned = my_state.subdomain.num_local_particles;
                if (!config_.acceleration.enable_self_gravity || owned <= 0 || !(half_span > 0.0)) return;
                const int blocks = (owned + block_size - 1) / block_size;
                LAUNCH_KERNEL(apply_half_gravity_impulse_kernel, blocks, block_size,
                              0, stream,
                              my_state.local_state.particles.vel_x,
                              my_state.local_state.particles.vel_y,
                              my_state.local_state.particles.vel_z,
                              my_state.local_state.particles.gravity_force_x,
                              my_state.local_state.particles.gravity_force_y,
                              my_state.local_state.particles.gravity_force_z,
                              half_span, owned);
            };

            auto prepare_sph_gravity_depth = [&]() {
                if (!sph_mode_ || !domain_params_.morton_enabled ||
                    !config_.acceleration.unified_fmm.adaptive_depth.enabled) return;
                // Every rank checks the same current endpoint h/root before
                // entering a barrier, so an infeasible input fails collectively.
                int ceiling = config_.acceleration.unified_fmm.adaptive_depth.max_depth;
                while (ceiling >= 2 && sph_global_h_max_ >
                       32.0*std::ldexp(domain_params_.morton_grid.size, -ceiling)) --ceiling;
                if (sph_global_h_max_ > 128.0*config_.sph.hsml_min ||
                    ceiling < config_.acceleration.unified_fmm.adaptive_depth.min_depth)
                    throw std::runtime_error("SPH smoothing lengths exceed the common FMM depth bounds");
                const int memory_ceiling = *std::min_element(common_depth_ceilings_.begin(), common_depth_ceilings_.end());
                if (memory_ceiling > 0) ceiling = std::min(ceiling, memory_ceiling);
                #pragma omp barrier
                #pragma omp single
                {
                    if (ceiling != common_fmm_depth_.state_for_gpu(0).effective_max_depth)
                        common_fmm_depth_.set_effective_max_depth(0, ceiling, gravity_evaluation_epoch_);
                    common_fmm_command_ = common_fmm_depth_.command_for_gpu(0);
                    common_launch_depth_ = common_fmm_command_.launch_depth;
                }
                #pragma omp barrier
                body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_compute_depth(common_launch_depth_);
            };

            // Main simulation loop
            while (!collective_stop && (force_bootstrap_pending || (my_state.local_state.current_time < config_.time.stop_time &&
                   !local_adaptive_yorp_stop))) {
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("SimulationStep");
                #endif

                if (sph_mode_) {
                    const int stage = sph_stage_index(config_.sph.stages, config_.sph.stage_count,
                        my_state.local_state.current_time);
                    if (stage != last_sph_stage) {
                        if (last_sph_stage != -2 && config_.sph.stage_count > 0)
                            force_bootstrap_pending = true;
                        last_sph_stage = stage;
                        if (gpu_id == 0 && stage >= 0)
                            std::cout << "SPH stage: " << sph_stage_name(config_.sph.stages[stage].mode) << std::endl;
                    }
                }
                const bool force_bootstrap_step = force_bootstrap_pending;
                const bool initial_bootstrap_step = force_bootstrap_step && initial_bootstrap_pending;

                // Match the single-GPU force-window receipt.  Drain every
                // participating device at each closed force-segment boundary;
                // output and diagnostic work between segments is excluded.
                if (!force_bootstrap_step && (!respa_enabled || respa_inner_remaining == 0) &&
                    config_.io.timing_measure_steps > 0 && !timing_window_finished &&
                    my_state.local_state.current_step - timing_start_step >= config_.io.timing_warmup_steps) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaStreamSynchronize(my_state.comm_stream));
                    #pragma omp barrier
                    #pragma omp single
                    {
                        if (!timing_window_started) {
                            timing_window_started = true;
                            timing_window_start_step = my_state.local_state.current_step;
                        }
                        timing_segment_start = std::chrono::high_resolution_clock::now();
                        timing_segment_active = true;
                    }
                }

                if (sph_mode_ && !force_bootstrap_step && respa_inner_remaining == 0) {
                    // Each device contributes the CFL minimum over its owned
                    // particles.  All workers then use one globally clamped dt.
                    const int32_t saved_count =
                        my_state.local_state.particles.num_particles;
                    my_state.local_state.particles.num_particles = my_state.subdomain.num_local_particles;
                    SPHTimestepResult timestep_result;
                    compute_sph_timestep(my_state.local_state, config_,
                                         timestep_result, stream);
                    my_state.local_state.particles.num_particles = saved_count;
                    sph_dt_candidates[static_cast<size_t>(gpu_id)] = timestep_result;
                    #pragma omp barrier
                    #pragma omp single
                    {
                        int selected_gpu = 0;
                        for (int candidate_gpu = 1; candidate_gpu < num_gpus_;
                             ++candidate_gpu) {
                            const auto& candidate = sph_dt_candidates[
                                static_cast<size_t>(candidate_gpu)];
                            const auto& selected = sph_dt_candidates[
                                static_cast<size_t>(selected_gpu)];
                            if (candidate.dt_unclamped < selected.dt_unclamped) {
                                selected_gpu = candidate_gpu;
                            }
                        }
                        global_sph_timestep_diagnostic = sph_dt_candidates[
                            static_cast<size_t>(selected_gpu)];
                        float global_dt = global_sph_timestep_diagnostic.dt_unclamped;
                        if (!(std::isfinite(global_dt) && global_dt > 0.0f)) {
                            global_dt = config_.sph.dt_min;
                        }
                        dt_ = std::max(config_.sph.dt_min,
                                       std::min(config_.sph.dt_max, global_dt));
                    }
                    #pragma omp barrier
                }

                // Plan only at a closed macro boundary.  Output, termination,
                // and adaptive-YORP observation boundaries are never crossed
                // by an open slow kick pair.  The final inner dt remains the
                // sole fractional timestep; both kicks use the summed span.
                if (respa_enabled && !force_bootstrap_step &&
                    respa_inner_remaining == 0) {
                    // The single planner prevents a round-off or stale local
                    // cadence from making one worker enter a different LET
                    // epoch.  All workers consume this exact plan below.
                    #pragma omp barrier
                    #pragma omp single
                    {
                        if (sph_mode_) {
                            double event = std::min(config_.time.stop_time,
                                sph_next_output);
                            if (config_.sph.stage_count > 0)
                                event = std::min(event, sph_current_stage(config_, my_state.local_state.current_time).end_time);
                            const auto plan = plan_sph_macro(my_state.local_state.current_time, dt_,
                                respa_k, event, config_.acceleration.gravity_macro_dt);
                            respa_shared_inner_count_ = plan.count;
                            respa_shared_macro_span_ = plan.span;
                            sph_shared_macro_end_ = plan.end_time;
                            if (config_.io.scalar_interval > 0) {
                            std::filesystem::create_directories(config_.io.output_directory);
                            const std::string path = std::string(config_.io.output_directory) + "/sph_macros.csv";
                            const bool header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
                            std::ofstream out(path, std::ios::app);
                            if (header) out << "start_step,start_time,inner_count,inner_dt,end_time,cfl_dt,gravity_evaluations\n";
                            out << std::setprecision(17) << my_state.local_state.current_step << ','
                                << my_state.local_state.current_time << ',' << plan.count << ',' << plan.dt << ','
                                << plan.end_time << ',' << dt_ << ',' << gravity_evaluation_epoch_ << '\n';
                            if (!out) throw std::runtime_error("Cannot write SPH macro receipt");
                            }
                        } else {
                        int planned = std::min(
                            respa_k,
                            std::max(1, static_cast<int>(std::ceil(
                                (config_.time.stop_time -
                                 my_state.local_state.current_time) / dt - 1e-12))));
                        if (config_.time.output_step > 0.0) {
                            const double q = std::floor(
                                my_state.local_state.current_time /
                                config_.time.output_step + 1.0e-12);
                            const double boundary = (q + 1.0) * config_.time.output_step;
                            planned = std::min(planned, std::max(1, static_cast<int>(
                                std::ceil((boundary - my_state.local_state.current_time) /
                                          dt - 1e-12))));
                        }
                        if (config_.acceleration.yorp.adaptive) {
                            planned = std::min(planned, std::max(1, static_cast<int>(
                                std::ceil((next_yorp_check -
                                           my_state.local_state.current_time) / dt -
                                          1e-12))));
                        }
                        respa_shared_inner_count_ = planned;
                        respa_shared_macro_span_ = terminal_limited_respa_span(
                            my_state.local_state.current_time, dt, planned,
                            config_.time.stop_time);
                        }
                    }
                    #pragma omp barrier
                    respa_inner_remaining = respa_shared_inner_count_;
                    respa_macro_inner_count = respa_shared_inner_count_;
                    respa_macro_span = respa_shared_macro_span_;
                    if (sph_mode_) {
                        dt = respa_macro_span / respa_macro_inner_count;
                        sph_macro_end = sph_shared_macro_end_;
                    }
                    apply_respa_half_kick(0.5 * respa_macro_span);
                }
                respa_macro_closing = !force_bootstrap_step && respa_enabled &&
                    respa_inner_remaining == 1;
                if (!sph_mode_) dt = terminal_limited_step_dt(my_state.local_state.current_time,
                                              dt_, config_.time.stop_time);
                my_state.local_state.dt = dt;

                int32_t num_local = my_state.subdomain.num_local_particles;
                int32_t num_total = num_local + my_state.subdomain.num_ghost_particles;
                int num_blocks_local = (num_local + block_size - 1) / block_size;
                int num_blocks_total = (num_total + block_size - 1) / block_size;

                const double force_evaluation_time = force_bootstrap_step
                    ? my_state.local_state.current_time
                    : my_state.local_state.current_time + dt;
                ShearPhase shear_phase;
                if (config_.shear_box.enabled && !force_bootstrap_step)
                    ShearBoxManager::validate_step(
                        my_state.local_state, config_, WallManager::get_pbc_config());

                harvest_timing_samples();
                harvest_distributed_fmm_samples();
                // A distributed depth trial changes only one owner's launch
                // depth, but its timing must be excluded from every worker's
                // RCB input.  Complete the nonblocking per-GPU harvests, then
                // advance the shared controller exactly once before any worker
                // snapshots its command/RCB marker for this epoch.  These are
                // host-only rendezvous; no CUDA stream is synchronized here.
                if (num_gpus_ > 1 && !domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled) {
                    #pragma omp barrier
                    #pragma omp single
                    {
                        distributed_fmm_depth_.process_ready_epochs();
                    }
                }
                const DistributedFMMDepthCommand fmm_depth_command =
                    (num_gpus_ > 1 && !domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled)
                    ? distributed_fmm_depth_.command_for_gpu(gpu_id)
                    : (domain_params_.morton_enabled ? common_fmm_command_ : DistributedFMMDepthCommand{});
                int32_t timing_slot = -1;
                for (int32_t slot = 0; slot < kTimingEventSlots; ++slot) {
                    if (timing_slot_epochs[slot] < 0) {
                        timing_slot = slot;
                        break;
                    }
                }
                if (timing_slot >= 0) {
                    size_t timing_index =
                        static_cast<size_t>(gpu_id) * kTimingEventSlots + timing_slot;
                    calibration_slot_epochs[timing_slot] =
                        my_state.local_state.current_step;
                    calibration_slot_sampled[timing_slot] =
                        (my_state.local_state.current_step %
                         config_.multigpu.rcb_calibration_sample_interval) == 0 &&
                        !fmm_depth_command.exclude_from_rcb;
                    timing_slot_excluded_from_rcb[timing_slot] =
                        fmm_depth_command.exclude_from_rcb;
                    calibration_owned_units[timing_slot] = num_local;
                    calibration_contact_units[timing_slot] = 0;
                    sph_degree_readback[timing_slot] = false;
                    calibration_halo_units[timing_slot] = 0.0;
                    calibration_halo_bytes[timing_slot] = 0.0;
                    calibration_halo_links[timing_slot] = RCBLinkClass::PCIe;
                    CUDA_CHECK(cudaEventRecord(step_start_events_[timing_index], stream));
                }

                // =============================================================
                // Step 1: Velocity Verlet - First half (local particles only)
                // =============================================================
                calibration_event(true, timing_slot,
                                  CalibrationSegment::Verlet1, stream);
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("Verlet1");
                #endif
                if (!force_bootstrap_step && sph_mode_ && num_local > 0) {
                    damp_sph_relaxation(my_state.local_state, config_, 0.5*dt, stream);
                    LAUNCH_KERNEL(sph_verlet_step1_kernel,
                        num_blocks_local, block_size, 0, stream,
                        my_state.local_state.particles.pos_x,
                        my_state.local_state.particles.pos_y,
                        my_state.local_state.particles.pos_z,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.mass,
                        dt, num_local);
                } else if (!force_bootstrap_step && num_local > 0 && config_.shear_box.enabled) {
                    ShearBoxManager::drift_step1(
                        my_state.local_state, config_, dt, num_local, stream,
                        distributed_clumps_.enabled()?my_state.local_state.particles.clump_id:nullptr);
                } else if (!force_bootstrap_step && num_local > 0) {
                    LAUNCH_KERNEL(verlet_step1_fused_kernel, num_blocks_local, block_size, 0, stream,
                        my_state.local_state.particles.pos_x,
                        my_state.local_state.particles.pos_y,
                        my_state.local_state.particles.pos_z,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.omega_x,
                        my_state.local_state.particles.omega_y,
                        my_state.local_state.particles.omega_z,
                        my_state.local_state.particles.quat_w,
                        my_state.local_state.particles.quat_x,
                        my_state.local_state.particles.quat_y,
                        my_state.local_state.particles.quat_z,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.torque_x,
                        my_state.local_state.particles.torque_y,
                        my_state.local_state.particles.torque_z,
                        my_state.local_state.particles.mass,
                        my_state.local_state.particles.moment_of_inertia,
                        dt, num_local,
                        distributed_clumps_.enabled() ? my_state.local_state.particles.clump_id : nullptr
                    );
                }
                if (distributed_clumps_.enabled()) {
                    if (!force_bootstrap_step)
                        distributed_clumps_.integrate(gpu_id,dt,true,stream,force_evaluation_time);
                    // Initialization can infer rigid spin from member velocities.
                    // Publish that spin and the canonical body offsets before
                    // the initial contact/friction force evaluation as well.
                    distributed_clumps_.sync(gpu_id,my_state.local_state.particles,num_local,stream,force_evaluation_time);
                }
                calibration_event(false, timing_slot,
                                  CalibrationSegment::Verlet1, stream);
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // Verlet1
                #endif

                if (config_.shear_box.enabled) {
                    shear_phase = ShearBoxManager::wrap(
                        my_state.local_state, config_, WallManager::get_pbc_config(),
                        force_evaluation_time, num_local, stream);
                    DomainManager::set_shear_time(config_, force_evaluation_time);
                } else if (!force_bootstrap_step && sph_mode_ && num_local > 0) {
                    const int32_t saved_count =
                        my_state.local_state.particles.num_particles;
                    my_state.local_state.particles.num_particles = num_local;
                    WallManager::apply_periodic_boundaries(
                        my_state.local_state, config_, stream);
                    my_state.local_state.particles.num_particles = saved_count;
                }

                // =============================================================
                // Step 2: Reset forces (local + ghost) - OPTIMIZED: single kernel
                // =============================================================
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("ForceCalculation");
                #endif
                if (num_total > 0) {
                    LAUNCH_KERNEL(reset_forces_kernel, num_blocks_total, block_size, 0, stream,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.torque_x,
                        my_state.local_state.particles.torque_y,
                        my_state.local_state.particles.torque_z,
                        num_total
                    );
                }

                if (domain_params_.morton_enabled) {
                    const bool depth_preflight=config_.acceleration.enable_self_gravity && !distributed_clumps_.enabled() &&
                        config_.acceleration.unified_fmm.adaptive_depth.enabled &&
                        (common_completed_epoch_==0||common_fmm_command_.launch_depth!=common_launch_depth_);
                    if(depth_preflight)common_depth_ceilings_[gpu_id]=body_fitted_fmm::UnifiedFMMDispatcher::distributed_depth_ceiling(
                        config_,common_global_particles_,&my_state.local_state,&WallManager::get_pbc_config(),stream);
                    if (!config_.shear_box.enabled)
                        morton_root_.collect(gpu_id, my_state.local_state.particles, stream);
                    #pragma omp barrier
                    #pragma omp single
                    {
                        if(depth_preflight){
                            const int ceiling=*std::min_element(common_depth_ceilings_.begin(),common_depth_ceilings_.end());
                            if(ceiling<config_.acceleration.unified_fmm.adaptive_depth.min_depth)
                                throw std::runtime_error("common FMM memory preflight cannot fit the minimum depth on every device");
                            // SPH applies the combined memory/physical ceiling at
                            // each gravity endpoint. Raising it here would cancel
                            // a live trial before that physical limit is reapplied.
                            if(!sph_mode_ && !distributed_clumps_.enabled() && ceiling!=common_fmm_depth_.state_for_gpu(0).effective_max_depth)
                                common_fmm_depth_.set_effective_max_depth(0,ceiling,common_completed_epoch_);
                            common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
                        }
                        // Wrapped physical XY ownership is fixed to the box.
                        // Gravity owns a separate shared envelope of replicas.
                        const bool root_changed=!config_.shear_box.enabled &&
                            morton_root_.commit(domain_params_);
                        common_partition_changed_=common_partition_pending_||root_changed;
                        common_partition_pending_=false;
                        if(root_changed && config_.acceleration.unified_fmm.adaptive_depth.enabled){
                            common_fmm_depth_.notify_disruptive_event(gravity_evaluation_epoch_,DistributedFMMDepthCancelReason::CapacityService);
                            common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
                        }
                        if(common_launch_depth_!=common_fmm_command_.launch_depth){
                            common_launch_depth_=common_fmm_command_.launch_depth;
                            std::cout<<"[FMM] common compute depth="<<common_launch_depth_
                                     <<" partition depth="<<domain_params_.morton_grid.depth
                                     <<" trial="<<common_fmm_command_.trial_id<<std::endl;
                        }
                        if (root_changed)
                            std::cout << "[FMM] collective root generation=" << domain_params_.morton_grid.generation
                                      << " size=" << domain_params_.morton_grid.size << std::endl;
                    }
                    if (common_partition_changed_) {
                        CUDA_CHECK(cudaSetDevice(gpu_id));
                        DomainManager::copy_params_to_device(domain_params_);
                        body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_partition(domain_params_);
                    }
                    body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_compute_depth(common_launch_depth_);
                    #pragma omp barrier
                }

                // =============================================================
                // Step 3-4: Particle Migration (peer-event orchestrated)
                // =============================================================
                // ORB decomposition: identify_migrating_particles uses
                // position_to_gpu() internally — no separate Morton code step needed.
                nvtxRangePushA("Particle Migration");

                // Migration and halo share Communicator's send/receive storage.
                // Wait only for each peer to finish consuming the preceding
                // halo epoch before this producer overwrites its buffers.
                if (particle_exchange_epoch > 1) {
                    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                        if (peer == gpu_id) continue;
                        require_phase(
                            phase_coordinator_.wait_for_consumed(
                                PeerPhase::Halo, gpu_id, peer,
                                particle_exchange_epoch - 1, stream),
                            "migration buffer reuse");
                    }
                }

                // Identify and pack migrating particles.  The packer's local
                // scalar count readback is the only host wait in this phase.
                comm.identify_migrating_particles(my_state, domain_params_, stream);
                comm.pack_migrating_particles(my_state, stream);
                CUDA_CHECK(cudaPeekAtLastError());
                require_phase(
                    phase_coordinator_.publish(
                        PeerPhase::Migration, gpu_id,
                        particle_exchange_epoch, stream),
                    "migration publication");

                // Receiver-pull transfers start as soon as each specific peer
                // publishes; no all-GPU pack barrier is involved.
                for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                    if (peer == gpu_id) continue;
                    require_phase(
                        phase_coordinator_.wait_for_published(
                            PeerPhase::Migration, peer, gpu_id,
                            particle_exchange_epoch, my_state.comm_stream),
                        "migration peer publication");
                }
                comm.transfer_migrants(communicators_, my_state.comm_stream);
                CUDA_CHECK(cudaEventRecord(migration_transfer_events_[gpu_id],
                                           my_state.comm_stream));
                CUDA_CHECK(cudaStreamWaitEvent(stream,
                                               migration_transfer_events_[gpu_id], 0));

                // Unpack and compaction consume both receive and source-side
                // phase storage.  Acknowledgements are recorded after those
                // kernels in this GPU's compute stream.
                my_state.subdomain.num_local_before_migration = my_state.subdomain.num_local_particles;
                comm.compact_local_particles(my_state, stream);
                comm.unpack_migrating_particles(my_state, stream);
                for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                    if (peer == gpu_id) continue;
                    require_phase(
                        phase_coordinator_.acknowledge_consumed(
                            PeerPhase::Migration, peer, gpu_id,
                            particle_exchange_epoch, stream),
                        "migration consumption acknowledgement");
                }
                nvtxRangePop();

                // Update counts after migration
                num_local = my_state.subdomain.num_local_particles;
                num_blocks_local = (num_local + block_size - 1) / block_size;
                if (timing_slot >= 0) calibration_owned_units[timing_slot] = num_local;
                if (comm.particle_membership_changed() &&
                    config_.acceleration.enable_self_gravity &&
                    config_.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
                    // Migration itself is trigger-only.  Its sole model effect
                    // is to invalidate the borrowed spatial view and force f3.
                    body_fitted_fmm::UnifiedFMMDispatcher::
                        invalidate_current_gpu_workload();
                }

                if (distributed_clumps_.enabled())
                    distributed_clumps_.refresh(gpu_id,my_state.local_state.particles,num_local,
                        comm.particle_membership_changed(),stream,force_evaluation_time);

                if(distributed_clumps_.enabled() && config_.acceleration.enable_self_gravity) {
                    const auto& fmm=config_.acceleration.unified_fmm;
                    const int minimum=fmm.adaptive_depth.enabled?fmm.adaptive_depth.min_depth:fmm.tree_depth;
                    const int maximum=fmm.adaptive_depth.enabled?fmm.adaptive_depth.max_depth:fmm.tree_depth;
                    FMMDepthSourceGeometry geometry;
                    auto effective=fmm_identity_effective_depths();
                    clump_depth_errors_[gpu_id].clear();
                    try {
                        geometry.root=domain_params_.morton_grid;
                        geometry.root.dimension=3;
                        if(config_.shear_box.enabled) {
                            const auto& pbc=WallManager::get_pbc_config();
                            geometry.images=make_shear_gravity_images(config_,pbc,force_evaluation_time);
                            const double4 root=shear_fmm_static_root(geometry.images,pbc,config_);
                            geometry.root.x=root.x;geometry.root.y=root.y;geometry.root.z=root.z;geometry.root.size=root.w;
                            for(int base=minimum;base<=maximum;++base){auto candidate=fmm;candidate.tree_depth=base;
                                effective[base]=shear_fmm_effective_depth(candidate,geometry.images,pbc);}
                        }
                        if(effective[minimum]>20)
                            throw std::runtime_error("no feasible FMM depth: minimum base depth exceeds the actual source-tree depth limit 20");
                        if(fmm.adaptive_depth.enabled) {
                            SimulationState depth_state=my_state.local_state;
                            depth_state.current_time=force_evaluation_time;
                            common_depth_ceilings_[gpu_id]=body_fitted_fmm::UnifiedFMMDispatcher::distributed_depth_ceiling(
                                config_,common_global_particles_,&depth_state,&WallManager::get_pbc_config(),stream);
                        }
                    }catch(const std::exception& error){clump_depth_errors_[gpu_id]=error.what();}
                    catch(...){clump_depth_errors_[gpu_id]="unknown FMM geometry/memory preflight failure";}
                    #pragma omp barrier
                    #pragma omp single
                    {
                        clump_collective_error_.clear();
                        for(int rank=0;rank<num_gpus_;++rank)if(!clump_depth_errors_[rank].empty()){
                            clump_collective_error_="FMM depth preflight rank "+std::to_string(rank)+": "+clump_depth_errors_[rank];break;}
                    }
                    #pragma omp barrier
                    if(!clump_collective_error_.empty())throw std::runtime_error(clump_collective_error_);
                    if(fmm.adaptive_depth.enabled)
                        clump_depth_census_.collect(gpu_id,my_state.local_state.particles,geometry,
                            effective[minimum],std::min(20,effective[maximum]),stream);
                    #pragma omp single
                    {
                        clump_collective_error_.clear();
                        try {
                            FMMDepthFeasibleRange feasible{};
                            if(fmm.adaptive_depth.enabled) {
                                const int memory=*std::min_element(common_depth_ceilings_.begin(),common_depth_ceilings_.end());
                                feasible=fmm_feasible_depth_range(clump_depth_census_.maximum_leaf_counts(),minimum,maximum,
                                    memory,distributed_clumps_.max_diameter(),geometry.root.size,effective);
                                common_fmm_depth_.set_effective_depth_range(0,feasible.minimum_depth,feasible.maximum_depth,
                                    gravity_evaluation_epoch_);
                                common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
                                if(common_fmm_depth_.frozen() || common_fmm_command_.launch_depth<feasible.minimum_depth ||
                                   common_fmm_command_.launch_depth>feasible.maximum_depth)
                                    throw std::runtime_error("FMM depth controller cannot satisfy the current hard constraints: "+
                                        common_fmm_depth_.failure_reason());
                                common_launch_depth_=common_fmm_command_.launch_depth;
                            }else distributed_clumps_.validate_leaf_width(geometry.root.size,effective[common_launch_depth_]);
                            const int actual=effective[common_launch_depth_];
                            const double leaf_width=std::ldexp(geometry.root.size,-actual);
                            const char* depth_diagnostics=std::getenv("GRANIT_CLUMP_DIAGNOSTICS");
                            if((depth_diagnostics && std::string(depth_diagnostics)=="1") || common_launch_depth_!=clump_logged_base_ ||
                               feasible.minimum_depth!=clump_logged_min_ || feasible.maximum_depth!=clump_logged_max_ ||
                               leaf_width!=clump_logged_leaf_width_) {
                            std::ostringstream row;
                            row<<std::setprecision(17)<<"[clumps-depth] epoch="<<gravity_evaluation_epoch_+1
                               <<" step="<<my_state.local_state.current_step+(force_bootstrap_step?0:1)
                               <<" time="<<force_evaluation_time<<" evaluation_time="<<force_evaluation_time
                               <<" adaptive="<<fmm.adaptive_depth.enabled<<" selected_base="<<common_launch_depth_
                               <<" actual_depth="<<actual<<" root_x="<<geometry.root.x
                               <<" root_y="<<geometry.root.y<<" root_z="<<geometry.root.z<<" root_size="<<geometry.root.size
                               <<" max_diameter="<<distributed_clumps_.max_diameter()
                               <<" leaf_width="<<leaf_width;
                            if(fmm.adaptive_depth.enabled)
                                row<<" global_max_leaf="<<clump_depth_census_.maximum_leaf_counts()[actual]
                                   <<" physical_particles="<<clump_depth_census_.physical_particles()
                                   <<" stored_source_particles="<<clump_depth_census_.stored_source_particles()
                                   <<" source_particles="<<clump_depth_census_.source_particles()
                                   <<" feasible_min="<<feasible.minimum_depth<<" feasible_max="<<feasible.maximum_depth;
                            std::cout<<row.str()<<std::endl;
                            clump_logged_base_=common_launch_depth_;clump_logged_min_=feasible.minimum_depth;
                            clump_logged_max_=feasible.maximum_depth;clump_logged_leaf_width_=leaf_width;
                            }
                        }catch(const std::exception& error){clump_collective_error_=error.what();}
                        catch(...){clump_collective_error_="unknown FMM feasible-depth validation failure";}
                    }
                    #pragma omp barrier
                    if(!clump_collective_error_.empty())throw std::runtime_error(clump_collective_error_);
                    body_fitted_fmm::UnifiedFMMDispatcher::set_distributed_compute_depth(common_launch_depth_);
                }

                // =============================================================
                // Step 5-6: Halo Exchange (peer-event orchestrated)
                // =============================================================
                nvtxRangePushA("Halo Exchange");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::HaloPack, stream);

                // Halo reuses the migration buffers, so wait for the current
                // migration consumers before overwriting them.
                nvtxRangePushA("Halo: Buffer Reuse Wait");
                for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                    if (peer == gpu_id) continue;
                    require_phase(
                        phase_coordinator_.wait_for_consumed(
                            PeerPhase::Migration, gpu_id, peer,
                            particle_exchange_epoch, stream),
                        "halo buffer reuse");
                }
                nvtxRangePop();

                nvtxRangePushA("Halo: Identify and Pack");
                const float halo_width = sph_mode_
                    ? 0.5f * config_.multigpu.halo_width_factor
                        * config_.sph.kernel_cutoff_factor * sph_global_h_max_
                    : dem_halo_width;
                const float extent_scale = sph_mode_
                    ? 0.5f * config_.multigpu.halo_width_factor : 1.0f;
                comm.identify_halo_particles(my_state, domain_params_, halo_width,
                                             stream, sph_mode_, extent_scale);
                comm.pack_halo_particles(my_state, stream, sph_mode_);
                calibration_event(false, timing_slot,
                                  CalibrationSegment::HaloPack, stream);
                nvtxRangePop();
                nvtxRangePushA("Halo: Publish");
                require_phase(
                    phase_coordinator_.publish(
                        PeerPhase::Halo, gpu_id,
                        particle_exchange_epoch, stream),
                    "halo publication");
                nvtxRangePop();

                nvtxRangePushA("Halo: Peer Publication Wait");
                for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                    if (peer == gpu_id) continue;
                    require_phase(
                        phase_coordinator_.wait_for_published(
                            PeerPhase::Halo, peer, gpu_id,
                            particle_exchange_epoch, my_state.comm_stream),
                        "halo peer publication");
                }
                nvtxRangePop();
                nvtxRangePushA("Halo: Transfer");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::HaloTransfer,
                                  my_state.comm_stream);
                comm.transfer_halos(communicators_, my_state.comm_stream);
                calibration_event(false, timing_slot,
                                  CalibrationSegment::HaloTransfer,
                                  my_state.comm_stream);
                CUDA_CHECK(cudaEventRecord(halo_transfer_events_[gpu_id],
                                           my_state.comm_stream));
                nvtxRangePop();
                nvtxRangePushA("Halo: Transfer Completion Wait");
                CUDA_CHECK(cudaStreamWaitEvent(stream, halo_transfer_events_[gpu_id], 0));
                nvtxRangePop();

                nvtxRangePushA("Halo: Unpack");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::HaloUnpack, stream);
                comm.unpack_ghost_particles(my_state, stream);
                calibration_event(false, timing_slot,
                                  CalibrationSegment::HaloUnpack, stream);
                nvtxRangePop();
                nvtxRangePushA("Halo: Consumption Acknowledgement");
                for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                    if (peer == gpu_id) continue;
                    require_phase(
                        phase_coordinator_.acknowledge_consumed(
                            PeerPhase::Halo, peer, gpu_id,
                            particle_exchange_epoch, stream),
                        "halo consumption acknowledgement");
                }
                nvtxRangePop();
                nvtxRangePop();

                if (timing_slot >= 0 && calibration_slot_sampled[timing_slot]) {
                    const HaloTransferSummary halo = comm.halo_transfer_summary();
                    calibration_halo_units[timing_slot] =
                        static_cast<double>(halo.sent_particles)
                      + static_cast<double>(halo.received_particles);
                    calibration_halo_bytes[timing_slot] =
                        static_cast<double>(halo.sent_bytes)
                      + static_cast<double>(halo.received_bytes);
                    bool saw_link = false;
                    bool all_nvlink = true;
                    const auto& buffers = comm.buffers();
                    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                        if (peer == gpu_id) continue;
                        const bool active = buffers[peer].h_send_count > 0 ||
                                            buffers[peer].h_recv_count > 0;
                        if (!active) continue;
                        saw_link = true;
                        all_nvlink = all_nvlink &&
                            is_nvlink_fabric(comm.link_class_to(peer));
                    }
                    calibration_halo_links[timing_slot] =
                        saw_link && all_nvlink ? RCBLinkClass::NVLink
                                              : RCBLinkClass::PCIe;
                }

                // Update total particle count
                num_total = num_local + my_state.subdomain.num_ghost_particles;
                num_blocks_total = (num_total + block_size - 1) / block_size;
                ensure_contact_list_capacity(gpu_id, my_state.subdomain.capacity, stream);

                CUDA_CHECK(cudaPeekAtLastError());

                // =============================================================
                // Step 6: Neighbor Search (includes ghost particles)
                // =============================================================
                nvtxRangePushA("Neighbor Search");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::NeighborSearch, stream);
                sph_hydro_offsets_valid_[gpu_id] = 0;
                if (num_total > 0) {
                    if (sph_mode_) {
                        // Halo membership changes every step, so rebuild the
                        // spatial structure over owned + ghost particles.
                        my_state.local_state.neighbor.num_particles = num_total;
                        NeighborSearch::find_sph_neighbors(
                            my_state.local_state, config_, stream);
                        // Owner indices are stable through the later halo
                        // refresh, while ghost membership and CSR indices are not.
                        // Save only the degree prefix used by the balancer.
                        if (my_state.local_state.neighbor.sph_adjacency_valid) {
                            const size_t needed = static_cast<size_t>(num_local) + 1;
                            if (needed > sph_hydro_offset_capacities_[gpu_id]) {
                                CUDA_CHECK(cudaFree(d_sph_hydro_owned_offsets_[gpu_id]));
                                sph_hydro_offset_capacities_[gpu_id] =
                                    static_cast<size_t>(my_state.subdomain.capacity) + 1;
                                CUDA_CHECK(cudaMalloc(&d_sph_hydro_owned_offsets_[gpu_id],
                                    sph_hydro_offset_capacities_[gpu_id] * sizeof(uint64_t)));
                            }
                            CUDA_CHECK(cudaMemcpyAsync(d_sph_hydro_owned_offsets_[gpu_id],
                                my_state.local_state.neighbor.sph_offsets,
                                needed * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream));
                            sph_hydro_offsets_valid_[gpu_id] = 1;
                        }
                        if (timing_slot >= 0 && calibration_slot_sampled[timing_slot] &&
                            my_state.local_state.neighbor.sph_adjacency_valid) {
                            const size_t index = static_cast<size_t>(gpu_id) *
                                kTimingEventSlots + timing_slot;
                            CUDA_CHECK(cudaMemcpyAsync(h_pinned_sph_owned_degrees_[index],
                                my_state.local_state.neighbor.sph_offsets + num_local,
                                sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
                            sph_degree_readback[timing_slot] = true;
                        }
                        contact_count_ready_[gpu_id] = false;
                        *h_pinned_num_contacts_[gpu_id] = 0;
                    } else {
                        NeighborSearch::find_contacts(my_state.local_state, config_,
                            my_contacts, config_.shear_box.enabled ? force_evaluation_time
                                : my_state.local_state.current_time, stream);

                        // Phase 4: Async contact count readback (pinned memory + event)
                        CUDA_CHECK(cudaMemcpyAsync(h_pinned_num_contacts_[gpu_id],
                                                    my_contacts.num_contacts,
                                                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                        CUDA_CHECK(cudaEventRecord(contact_count_events_[gpu_id], stream));
                        contact_count_ready_[gpu_id] = true;
                    }
                } else {
                    // BUGFIX: Reset contact_count_ready_ when domain is empty.
                    // Without this, a GPU that becomes empty after load rebalance
                    // retains stale contact_count_ready_=true from a prior step,
                    // causing contact force kernels to launch with stale/corrupt
                    // contact pair data → GPU error → cudaEventSynchronize crash.
                    contact_count_ready_[gpu_id] = false;
                    *h_pinned_num_contacts_[gpu_id] = 0;
                }
                calibration_event(false, timing_slot,
                                  CalibrationSegment::NeighborSearch, stream);
                nvtxRangePop(); // Neighbor Search

                // =============================================================
                // Step 7: Contact Forces (with ghost handling) - OPTIMIZED
                // =============================================================
                nvtxRangePushA("Contact Force Setup");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::ContactForce, stream);

                // Phase 4: Use async contact count - query event, fall back to max_contacts
                int32_t h_num_contacts = 0;
                if (contact_count_ready_[gpu_id]) {
                    cudaError_t event_status = cudaEventQuery(contact_count_events_[gpu_id]);
                    if (event_status == cudaSuccess) {
                        h_num_contacts = *h_pinned_num_contacts_[gpu_id];
                    } else {
                        // Event not ready yet - sync to get the value
                        CUDA_CHECK(cudaEventSynchronize(contact_count_events_[gpu_id]));
                        h_num_contacts = *h_pinned_num_contacts_[gpu_id];
                    }
                }
                const float force_dt = force_bootstrap_step ? 0.0f :
                    static_cast<float>(dt);

                // Retire ended contacts and reclaim their slots before births,
                // including same-step turnover in a previously full owner row.
                // Freeze the retained lookup prefix across both owner passes;
                // newly reserved tail entries must never be read by peers.
                if (!sph_mode_)
                    prepare_distributed_contact_history(my_contacts.pairs, h_num_contacts,
                        my_state.local_state, num_local, contact_history_workspace, stream,
                        config_.shear_box.enabled,
                        WallManager::get_pbc_config().box_length_x,
                        WallManager::get_pbc_config().box_length_y,
                        static_cast<float>(shear_phase.reduced_shift),
                        distributed_clumps_.enabled());

                if (h_num_contacts > 0) {
                    int contact_blocks = (h_num_contacts + block_size - 1) / block_size;

                    bool use_visco = config_.dem_damping_model == DemDampingModel::VISCOUS;

                    // Pass 1: Forces on owner particle i
                    compute_contact_forces_multigpu_kernel<false><<<contact_blocks, block_size, 0, stream>>>(
                        my_contacts.pairs,
                        h_num_contacts,
                        my_state.local_state.particles.pos_x,
                        my_state.local_state.particles.pos_y,
                        my_state.local_state.particles.pos_z,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.omega_x,
                        my_state.local_state.particles.omega_y,
                        my_state.local_state.particles.omega_z,
                        my_state.local_state.particles.radius,
                        my_state.local_state.particles.mass,
                        my_state.local_state.particles.moment_of_inertia,
                        my_state.local_state.particles.material_id,
                        my_state.local_state.material_database.materials,
                        my_state.local_state.material_database.dem_interactions,
                        my_state.local_state.material_database.material_count,
                        my_state.local_state.particles.global_id,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.torque_x,
                        my_state.local_state.particles.torque_y,
                        my_state.local_state.particles.torque_z,
                        my_state.local_state.contact_history.data,
                        my_state.local_state.particles.contact_history_offset,
                        my_state.local_state.particles.num_contacts,
                        my_state.local_state.contact_history.max_contacts_per_particle,
                        contact_history_workspace.lookup_counts,
                        contact_history_workspace.overflow,
                        use_visco,
                        force_dt,
                        num_local,  // Pass num_local for ghost detection
                        config_.shear_box.enabled,
                        WallManager::get_pbc_config().box_length_x,
                        WallManager::get_pbc_config().box_length_y,
                        static_cast<float>(shear_phase.reduced_shift),
                        static_cast<float>(shear_phase.shear_velocity),
                        distributed_clumps_.enabled()
                    );
                    
                    // Pass 2: Reaction forces on particle j (flipped perspective)
                    compute_contact_forces_multigpu_kernel<true><<<contact_blocks, block_size, 0, stream>>>(
                        my_contacts.pairs,
                        h_num_contacts,
                        my_state.local_state.particles.pos_x,
                        my_state.local_state.particles.pos_y,
                        my_state.local_state.particles.pos_z,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.omega_x,
                        my_state.local_state.particles.omega_y,
                        my_state.local_state.particles.omega_z,
                        my_state.local_state.particles.radius,
                        my_state.local_state.particles.mass,
                        my_state.local_state.particles.moment_of_inertia,
                        my_state.local_state.particles.material_id,
                        my_state.local_state.material_database.materials,
                        my_state.local_state.material_database.dem_interactions,
                        my_state.local_state.material_database.material_count,
                        my_state.local_state.particles.global_id,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.torque_x,
                        my_state.local_state.particles.torque_y,
                        my_state.local_state.particles.torque_z,
                        my_state.local_state.contact_history.data,
                        my_state.local_state.particles.contact_history_offset,
                        my_state.local_state.particles.num_contacts,
                        my_state.local_state.contact_history.max_contacts_per_particle,
                        contact_history_workspace.lookup_counts,
                        contact_history_workspace.overflow,
                        use_visco,
                        force_dt,
                        num_local,  // Pass num_local for ghost detection
                        config_.shear_box.enabled,
                        WallManager::get_pbc_config().box_length_x,
                        WallManager::get_pbc_config().box_length_y,
                        static_cast<float>(shear_phase.reduced_shift),
                        static_cast<float>(shear_phase.shear_velocity),
                        distributed_clumps_.enabled()
                    );
                }

                if (!sph_mode_) {
                    CUDA_CHECK(cudaMemcpyAsync(&contact_history_errors[gpu_id],
                        contact_history_workspace.overflow, sizeof(ContactHistoryOverflow),
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    // All ranks finish force/history work before any rank may
                    // kick, write a valid output, or checkpoint this step.
                    #pragma omp barrier
                    #pragma omp single
                    {
                        contact_history_overflow = false;
                        for (int rank = 0; rank < num_gpus_; ++rank) {
                            const auto& error = contact_history_errors[rank];
                            if (!error.flag) continue;
                            contact_history_overflow = true;
                            std::cerr << "DEM contact-history capacity exceeded: rank=" << rank
                                << " owner_global_id=" << error.owner_global_id
                                << " partner_global_id=" << error.partner_global_id
                                << " stride=" << error.stride
                                << " required_count_at_least=" << error.required_count << std::endl;
                        }
                    }
                    #pragma omp barrier
                    // Every worker takes the same failure path after the
                    // collective check. The outer catch joins all workers and
                    // rethrows before run() can write final output/checkpoint.
                    if (contact_history_overflow)
                        throw std::runtime_error("DEM contact-history capacity exceeded; step aborted before integration/output/checkpoint");
                }

                if (sph_mode_) {
                    auto& local = my_state.local_state;
                    auto refresh_sph_halos = [&](bool rebuild_support,
                                                 SphHaloPayload payload) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        #pragma omp barrier
                        if (rebuild_support) {
                            const float reach = 0.5f * config_.multigpu.halo_width_factor *
                                config_.sph.kernel_cutoff_factor * sph_global_h_max_;
                            comm.identify_halo_particles(my_state, domain_params_, reach,
                                stream, true, 0.5f * config_.multigpu.halo_width_factor);
                            comm.pack_halo_particles(my_state, stream, true);
                        } else {
                            comm.repack_halo_particles(my_state, stream, true, payload);
                        }
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        #pragma omp barrier
                        comm.transfer_halos(communicators_, my_state.comm_stream);
                        CUDA_CHECK(cudaStreamSynchronize(my_state.comm_stream));
                        #pragma omp barrier
                        comm.unpack_ghost_particles(my_state, stream);
                        num_total = num_local + my_state.subdomain.num_ghost_particles;
                        if (rebuild_support && num_total > 0) {
                            local.neighbor.num_particles = num_total;
                            // Gravity's SPH correction traverses the current
                            // tree directly.  Rebuild it for changed h and
                            // ghosts without enumerating an unused second CSR.
                            NeighborSearch::rebuild(local, config_, stream);
                        }
                        local.particles.num_particles = num_local;
                    };

                    local.particles.num_particles = num_local;
                    calibration_event(true, timing_slot,
                                      CalibrationSegment::SphDensity, stream);
                    if (num_local > 0) {
                        compute_sph_density(local, config_, stream);
                        if (force_bootstrap_step) compute_sph_pressure(local, config_, stream);
                    }
                    calibration_event(false, timing_slot,
                                      CalibrationSegment::SphDensity, stream);
                    // Continuous density is unchanged here on regular steps;
                    // its full halo from the initial exchange is still current.
                    // Bootstrap EOS and summation-density steps do update it.
                    if (force_bootstrap_step || config_.sph.density_update_mode != 1)
                        refresh_sph_halos(false, SphHaloPayload::Thermo);
                    calibration_event(true, timing_slot,
                                      CalibrationSegment::SphConstitutive, stream);
                    if (num_local > 0) {
                        evaluate_sph_damage(local, config_, sph_damage_workspaces_[gpu_id], stream);
                        update_sph_kernel_correction(local, config_, stream);
                        compute_sph_density_rate(local, config_, stream);
                        predict_sph_strength_and_work(local, config_,
                            force_bootstrap_step ? 0.0f : static_cast<float>(dt), stream);
                        compute_sph_energy_rate(local, config_, stream);
                        evaluate_sph_porosity_rates(local, config_, stream);
                        if (!force_bootstrap_step) {
                            advance_sph_cold_energy(local, config_, stream);
                            integrate_sph_energy(local, config_, stream);
                            integrate_sph_density(local, config_, static_cast<float>(dt), stream);
                            compute_sph_pressure(local, config_, stream);
                            integrate_sph_porosity(local, config_, static_cast<float>(dt), stream);
                            compute_sph_pressure(local, config_, stream);
                            apply_sph_strength_radial_return(local, config_, stream);
                            evaluate_sph_porosity_rates(local, config_, stream);
                        }
                    }
                    calibration_event(false, timing_slot,
                                      CalibrationSegment::SphConstitutive, stream);
                    // Force pairs must consume the same closed pressure/stress
                    // on both endpoints, including current stage overrides.
                    refresh_sph_halos(false, SphHaloPayload::Force);
                    calibration_event(true, timing_slot,
                                      CalibrationSegment::SphForceHistory, stream);
                    if (num_local > 0) {
                        compute_sph_forces(local, config_, stream);
                        if (!force_bootstrap_step) {
                            integrate_sph_damage(local, config_, sph_damage_workspaces_[gpu_id],
                                                 static_cast<float>(dt), stream);
                            update_sph_smoothing_length(local, config_, static_cast<float>(dt), stream);
                            update_sph_support_extent(local, config_, num_local, stream);
                        }
                        update_sph_history(local, config_,
                            force_bootstrap_step ? 0.0f : static_cast<float>(dt), stream);
                        sph_h_max_candidates[static_cast<size_t>(gpu_id)] =
                            compute_sph_h_max(local, num_local, stream);
                    } else {
                        sph_h_max_candidates[static_cast<size_t>(gpu_id)] = 0.0f;
                    }
                    calibration_event(false, timing_slot,
                                      CalibrationSegment::SphForceHistory, stream);
                    #pragma omp barrier
                    #pragma omp single
                    {
                        sph_global_h_max_ = *std::max_element(
                            sph_h_max_candidates.begin(), sph_h_max_candidates.end());
                    }
                    #pragma omp barrier
                    set_sph_h_max_global(sph_global_h_max_, stream);
                    if (config_.acceleration.enable_self_gravity)
                        refresh_sph_halos(true, SphHaloPayload::Full);
                    local.particles.num_particles = num_total;
                }
                if (distributed_clumps_.enabled())
                    distributed_clumps_.diagnose_contacts(gpu_id,my_state.local_state.particles,
                        my_contacts.pairs,h_num_contacts,num_local,my_state.local_state.current_step,stream,force_evaluation_time);
                latest_contact_counts_[gpu_id] = h_num_contacts;
                if (timing_slot >= 0) {
                    // SPH uses one base unit per current owner plus its CSR
                    // degree; DEM has two force passes per contact pair.
                    calibration_contact_units[timing_slot] = sph_mode_
                        ? static_cast<uint64_t>(num_local)
                        : 2ULL * static_cast<uint64_t>(h_num_contacts);
                }
                calibration_event(false, timing_slot,
                                  CalibrationSegment::ContactForce, stream);
                nvtxRangePop(); // Contact Force Setup

                // =============================================================
                // Step 8: Wall Forces (local particles only)
                // =============================================================
                nvtxRangePushA("Wall Forces");
                calibration_event(true, timing_slot,
                                  CalibrationSegment::WallForce, stream);
                if (!sph_mode_) {
                    WallManager::compute_forces(my_state.local_state, config_,
                                               force_dt,
                                               force_evaluation_time, stream);
                }
                calibration_event(false, timing_slot,
                                  CalibrationSegment::WallForce, stream);
                nvtxRangePop();

                // =============================================================
                // Step 9: External Accelerations (local particles only)
                // Multi-GPU: uses LET exchange for BH tree self-gravity
                // =============================================================
                nvtxRangePushA("Acceleration");
                if (num_gpus_ > 1 && !domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled &&
                    fmm_depth_command.launch_depth > 0) {
                    const auto depth_state =
                        distributed_fmm_depth_.state_for_gpu(gpu_id);
                    body_fitted_fmm::ExternalDistributedFMMDepthCommand command{};
                    command.selected_depth = depth_state.selected_depth;
                    command.candidate_depth = depth_state.candidate_depth;
                    command.launch_depth = fmm_depth_command.launch_depth;
                    command.controller_epoch = fmm_depth_command.trial_id;
                    if (!body_fitted_fmm::UnifiedFMMDispatcher::
                            set_external_adaptive_fmm_depth_command(command, stream)) {
                        distributed_fmm_depth_.freeze(
                            "runtime rejected distributed FMM depth command",
                            static_cast<uint64_t>(
                                my_state.local_state.current_step + 1));
                    }
                }
                if (respa_enabled) {
                    // Inner force is strictly fast: retained outer gravity is
                    // applied only by the bracketing slow impulses.
                    AccelerationManager::compute_all_without_gravity(
                        my_state.local_state, config_,
                        force_evaluation_time, stream,
                        /*include_incremental_effects=*/!config_.acceleration.yorp.adaptive);
                    if (force_bootstrap_step && config_.acceleration.enable_self_gravity) {
                        // q0 ordering: distributed fast/contact/wall force
                        // first, then one collective slow acceleration epoch.
                        #pragma omp barrier
                        #pragma omp single
                        { ++gravity_evaluation_epoch_; }
                        #pragma omp barrier
                        local_gravity_epoch = gravity_evaluation_epoch_;
                        prepare_sph_gravity_depth();
                        body_fitted_fmm::UnifiedFMMDispatcher::compute_outer_multigpu(
                            my_state.local_state, config_,
                            my_state.local_state.neighbor,
                            WallManager::get_pbc_config(), let_comms_[gpu_id],
                            local_gravity_epoch,
                            my_state.local_state.current_time, stream);
                    }
                } else {
                    if (force_bootstrap_step && config_.shear_box.enabled &&
                        !config_.acceleration.enable_self_gravity) {
                        AccelerationManager::compute_all_without_gravity(
                            my_state.local_state, config_, force_evaluation_time,
                            stream, /*include_incremental_effects=*/false);
                    } else if (force_bootstrap_step &&
                        config_.acceleration.enable_self_gravity &&
                        config_.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
                        // q0 must not advance YORP.  Split the regular
                        // combined dispatcher into its fast/external and LET
                        // gravity pieces only for this force-only bootstrap.
                        AccelerationManager::compute_all_without_gravity(
                            my_state.local_state, config_,
                            my_state.local_state.current_time, stream,
                            /*include_incremental_effects=*/false);
                        #pragma omp barrier
                        #pragma omp single
                        { ++gravity_evaluation_epoch_; }
                        #pragma omp barrier
                        local_gravity_epoch = gravity_evaluation_epoch_;
                        body_fitted_fmm::UnifiedFMMDispatcher::compute_multigpu(
                            my_state.local_state, config_,
                            my_state.local_state.neighbor,
                            WallManager::get_pbc_config(), let_comms_[gpu_id],
                            local_gravity_epoch,
                            my_state.local_state.current_time, stream,
                            distributed_clumps_.enabled() ? my_state.local_state.particles.clump_id : nullptr);
                        add_sph_gravity_softening_correction(my_state.local_state, config_, stream);
                    } else {
                        #pragma omp barrier
                        #pragma omp single
                        { ++gravity_evaluation_epoch_; }
                        #pragma omp barrier
                        local_gravity_epoch = gravity_evaluation_epoch_;
                        AccelerationManager::compute_all_multigpu(
                            my_state.local_state, config_,
                            my_state.local_state.current_time + dt,
                            local_gravity_epoch,
                            &let_comms_[gpu_id], stream,
                            distributed_clumps_.enabled() ? my_state.local_state.particles.clump_id : nullptr);
                    }
                }
                if (config_.acceleration.yorp.adaptive && !force_bootstrap_step) {
                    require_phase(distributed_yorp_.apply_increment(
                        gpu_id, my_state.local_state.particles,
                        my_state.subdomain.num_local_particles,
                        my_state.local_state.current_time + dt, dt, config_,
                        block_size, stream), "adaptive YORP increment");
                }
                nvtxRangePop();
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // ForceCalculation
                #endif

                // =============================================================
                // Step 10: Velocity Verlet - Second half (local particles only)
                // =============================================================
                calibration_event(true, timing_slot,
                                  CalibrationSegment::Verlet2, stream);
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePushA("Verlet2");
                #endif
                if (!force_bootstrap_step && sph_mode_ && num_local > 0) {
                    LAUNCH_KERNEL(sph_verlet_step2_kernel,
                        num_blocks_local, block_size, 0, stream,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.mass,
                        dt, num_local);
                } else if (!force_bootstrap_step && num_local > 0 && config_.shear_box.enabled) {
                    ShearBoxManager::kick_step2(
                        my_state.local_state, config_, dt, num_local, stream,
                        distributed_clumps_.enabled()?my_state.local_state.particles.clump_id:nullptr);
                } else if (!force_bootstrap_step && num_local > 0) {
                    LAUNCH_KERNEL(verlet_step2_fused_kernel, num_blocks_local, block_size, 0, stream,
                        my_state.local_state.particles.vel_x,
                        my_state.local_state.particles.vel_y,
                        my_state.local_state.particles.vel_z,
                        my_state.local_state.particles.omega_x,
                        my_state.local_state.particles.omega_y,
                        my_state.local_state.particles.omega_z,
                        my_state.local_state.particles.force_x,
                        my_state.local_state.particles.force_y,
                        my_state.local_state.particles.force_z,
                        my_state.local_state.particles.torque_x,
                        my_state.local_state.particles.torque_y,
                        my_state.local_state.particles.torque_z,
                        my_state.local_state.particles.mass,
                        my_state.local_state.particles.moment_of_inertia,
                        static_cast<float>(dt), num_local,
                        distributed_clumps_.enabled() ? my_state.local_state.particles.clump_id : nullptr
                    );
                }
                if (distributed_clumps_.enabled()) {
                    distributed_clumps_.aggregate(gpu_id,my_state.local_state.particles,num_local,stream);
                    if (!force_bootstrap_step) distributed_clumps_.integrate(gpu_id,dt,false,stream,force_evaluation_time);
                    distributed_clumps_.sync(gpu_id,my_state.local_state.particles,num_local,stream,force_evaluation_time);
                }
                calibration_event(false, timing_slot,
                                  CalibrationSegment::Verlet2, stream);
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // Verlet2
                #endif

                // =============================================================
                // Step 11: Time Advance
                // =============================================================
                if (!force_bootstrap_step && sph_mode_)
                    damp_sph_relaxation(my_state.local_state, config_, 0.5*dt, stream);

                if (!force_bootstrap_step) {
                    my_state.local_state.current_time += dt;
                    my_state.local_state.current_step++;
                    if (timing_slot >= 0) {
                        timing_slot_epochs[timing_slot] = my_state.local_state.current_step;
                    }
                } else {
                    // The bootstrap has built q0 force state but intentionally
                    // performed no kick/drift or adaptive-YORP increment.
                    force_bootstrap_pending = false;
                    initial_bootstrap_pending = false;
                    if (timing_slot >= 0) timing_slot_epochs[timing_slot] = -1;
                }

                if (respa_enabled && !force_bootstrap_step) {
                    --respa_inner_remaining;
                    if (respa_macro_closing && sph_mode_) my_state.local_state.current_time = sph_macro_end;
                    if (respa_macro_closing && config_.acceleration.enable_self_gravity) {
                        // Endpoint gravity precedes the closing kick.  Advance
                        // one shared, monotonic collective epoch even when an
                        // owner is empty, rather than recycling current_step.
                        #pragma omp barrier
                        #pragma omp single
                        { ++gravity_evaluation_epoch_; }
                        #pragma omp barrier
                        local_gravity_epoch = gravity_evaluation_epoch_;
                        prepare_sph_gravity_depth();
                        body_fitted_fmm::UnifiedFMMDispatcher::compute_outer_multigpu(
                            my_state.local_state, config_,
                            my_state.local_state.neighbor,
                            WallManager::get_pbc_config(), let_comms_[gpu_id],
                            local_gravity_epoch,
                            my_state.local_state.current_time, stream);
                        apply_respa_half_kick(0.5 * respa_macro_span);
                    }
                }

                if (!force_bootstrap_step && (!respa_enabled || respa_macro_closing) &&
                    timing_segment_active) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaStreamSynchronize(my_state.comm_stream));
                    #pragma omp barrier
                    #pragma omp single
                    {
                        timing_window_wall_seconds += std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - timing_segment_start).count();
                        timing_segment_active = false;
                        timing_window_end_step = my_state.local_state.current_step;
                        timing_window_finished = timing_window_end_step - timing_window_start_step >=
                            config_.io.timing_measure_steps;
                    }
                }

                if (timing_slot >= 0) {
                    size_t timing_index =
                        static_cast<size_t>(gpu_id) * kTimingEventSlots + timing_slot;
                    CUDA_CHECK(cudaEventRecord(step_stop_events_[timing_index], stream));
                }

                if (!force_bootstrap_step && config_.acceleration.yorp.adaptive &&
                    (!respa_enabled || respa_macro_closing) &&
                    my_state.local_state.current_time + 1e-12 >= next_yorp_check) {
                    require_phase(distributed_yorp_.observe(
                        gpu_id, my_state.local_state.particles,
                        my_state.subdomain.num_local_particles,
                        my_state.local_state.current_time, config_, block_size,
                        stream, yorp_observation_epoch++),
                        "adaptive YORP observation");
                    AdaptiveYorpDiagnostic diagnostic{};
                    if (!AccelerationManager::yorp_manager_for_gpu(gpu_id)->
                            consume_adaptive_diagnostic(&diagnostic, stream)) {
                        throw std::runtime_error(
                            "distributed adaptive YORP diagnostic is unavailable");
                    }
                    local_adaptive_yorp_stop = diagnostic.fatal != 0 ||
                        diagnostic.state == static_cast<int32_t>(
                            AdaptiveYorpState::COMPLETE);
                    if (diagnostic.transition_reason !=
                            static_cast<int32_t>(AdaptiveYorpTransition::NONE) &&
                        diagnostic.transition_time >
                            local_last_yorp_transition_time) {
                        // Every worker observes the same replicated state.  A
                        // duplicate cancellation is harmless and lets the
                        // directed CUDA protocol proceed without a global
                        // OpenMP check-boundary barrier.
                        if (domain_params_.morton_enabled) {
                            // Rank zero writes; the closed-step barrier below
                            // publishes this to the common controller owner.
                            if (gpu_id == 0) common_yorp_transition_pending_ = true;
                        } else {
                            distributed_fmm_depth_.notify_disruptive_event(
                                static_cast<uint64_t>(my_state.local_state.current_step),
                                DistributedFMMDepthCancelReason::YorpTransition);
                        }
                        local_last_yorp_transition_time =
                            diagnostic.transition_time;
                    }
                    if (gpu_id == 0) {
                        append_distributed_yorp_diagnostic(diagnostic);
                        adaptive_yorp_fatal_ = diagnostic.fatal != 0;
                    }
                    do {
                        next_yorp_check +=
                            config_.acceleration.yorp.check_interval_seconds;
                    } while (next_yorp_check <=
                             my_state.local_state.current_time);
                }

                // =============================================================
                // Step 12: Output (gather to GPU 0)
                // =============================================================
                // Stage rebootstrap must not overwrite the previous closed-step
                // output with newly initialized stage histories at the same time.
                if ((initial_bootstrap_step && config_.io.output_initial_state) ||
                    (!force_bootstrap_step && (!respa_enabled || respa_macro_closing) &&
                    config_.time.output_step > 0.0 &&
                    (sph_mode_ ? my_state.local_state.current_time >= sph_next_output :
                    static_cast<int64_t>(my_state.local_state.current_time /
                    config_.time.output_step) >
                    static_cast<int64_t>((my_state.local_state.current_time - dt) /
                    config_.time.output_step)))) {

                    // The gather below runs in one host thread and switches
                    // across devices.  Complete each device's step before any
                    // synchronous D2H copy observes its owned rows.
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    if (!domain_params_.morton_enabled && config_.acceleration.unified_fmm.adaptive_depth.enabled &&
                        body_fitted_fmm::UnifiedFMMDispatcher::
                            service_adaptive_fmm_capacity(
                                my_state.local_state, config_, stream)) {
                        if (num_gpus_ > 1) distributed_fmm_depth_.notify_disruptive_event(
                            static_cast<uint64_t>(
                                my_state.local_state.current_step),
                            DistributedFMMDepthCancelReason::CapacityService);
                    }
                    #pragma omp barrier
                    #pragma omp single
                    {
                        nvtxRangePushA("Output Data");
                        save_output();
                        nvtxRangePop(); // Output Data
                    }
                    #pragma omp barrier
                    // Restore device context after save_output may have changed it
                    CUDA_CHECK(cudaSetDevice(gpu_id));
                    if (sph_mode_ && !initial_bootstrap_step)
                        sph_next_output = next_sph_output_time(my_state.local_state.current_time,
                            config_.time.start_time, config_.time.output_step);
                }

                // =============================================================
                // Step 13: Load Rebalancing (asynchronous check, collective fire)
                // =============================================================
                if(domain_params_.morton_enabled && config_.acceleration.enable_self_gravity &&
                    (!respa_enabled || force_bootstrap_step || respa_macro_closing)){
                    common_metrics_[gpu_id]=body_fitted_fmm::UnifiedFMMDispatcher::distributed_metrics(stream);
                    harvest_timing_samples();
                    #pragma omp barrier
                    #pragma omp single
                    {
                        // Only scalar rank observations enter the common controller.
                        const uint64_t epoch=common_metrics_[0].epoch;
                        double maximum=0,mean=0;uint64_t bytes=0;int64_t owned=0;
                        bool counts_changed=false;
                        if(common_stable_owned_counts_.size()!=static_cast<size_t>(num_gpus_))
                            common_stable_owned_counts_.assign(num_gpus_,0);
                        for(int peer=0;peer<num_gpus_;++peer){
                            const int64_t count=common_metrics_[peer].owned;
                            if(common_completed_epoch_ && !common_partition_changed_ &&
                               std::fabs(static_cast<double>(count-common_stable_owned_counts_[peer])) /
                                   std::max(1.0,static_cast<double>(common_stable_owned_counts_[peer])) >
                                   config_.acceleration.unified_fmm.adaptive_depth.multigpu.local_count_reset_fraction)
                                counts_changed=true;
                        }
                        if(!common_fmm_depth_.trial_active() || common_partition_changed_ || counts_changed || !common_completed_epoch_)
                            for(int peer=0;peer<num_gpus_;++peer)common_stable_owned_counts_[peer]=common_metrics_[peer].owned;
                        for(const auto& metric:common_metrics_){
                            if(metric.epoch!=epoch||!std::isfinite(metric.total_ms))throw std::runtime_error("incomplete common FMM timing epoch");
                            maximum=std::max(maximum,metric.total_ms);mean+=metric.total_ms;bytes+=metric.received_bytes;owned+=metric.owned;
                        }
                        if(epoch>common_completed_epoch_){
                            if(config_.acceleration.unified_fmm.adaptive_depth.enabled){
                                DistributedFMMDepthEpochTiming aggregate{};
                                aggregate.epoch=epoch;aggregate.command_trial_id=common_fmm_command_.trial_id;
                                aggregate.launch_depth=common_launch_depth_;
                                aggregate.elapsed_ns=std::max<uint64_t>(1,static_cast<uint64_t>(maximum*1e6));
                                aggregate.aggregate_mean_ns=static_cast<uint64_t>(mean/num_gpus_*1e6);
                                aggregate.let_bytes=bytes;aggregate.local_owned_particles=owned;
                                aggregate.yorp_transition=common_yorp_transition_pending_;
                                aggregate.local_count_changed=counts_changed;
                                if(!common_fmm_depth_.publish_timing(0,aggregate) && common_fmm_depth_.frozen())
                                    throw std::runtime_error(common_fmm_depth_.failure_reason());
                                common_fmm_depth_.process_ready_epochs();
                                common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
                            }
                            common_completed_epoch_=epoch;
                            if(std::getenv("GRANIT_DISTRIBUTED_FMM_DIAGNOSTICS")){
                                std::cout<<"[FMM epoch] epoch="<<epoch<<" depth="<<common_launch_depth_<<" max_ms="<<maximum<<" mean_ms="<<mean/num_gpus_<<" bytes="<<bytes;
                                for(int peer=0;peer<num_gpus_;++peer){const auto& m=common_metrics_[peer];std::cout<<" rank"<<peer<<"=[tree:"<<m.tree_ms<<",far:"<<m.far_ms<<",near:"<<m.near_ms<<",exchange:"<<m.exchange_ms<<"]";}
                                std::cout<<std::endl;
                            }
                        }
                        if (common_yorp_transition_pending_) {
                            common_yorp_transition_pending_ = false;
                        }
                        int64_t calibration_epoch=-1;std::vector<RCBActiveSample> samples;
                        while(rcb_sample_mailbox_.try_collect_next(common_calibration_cursor_,calibration_epoch,samples)){
                            common_calibration_cursor_=calibration_epoch;
                            rcb_calibrator_.observe_global_epoch(calibration_epoch,samples);samples.clear();
                        }
                        common_balance_due_=num_gpus_>1 && !force_bootstrap_step && !common_fmm_depth_.trial_active() &&
                            common_fmm_command_.launch_depth==common_launch_depth_ &&
                            my_state.local_state.current_step-common_last_balance_step_>=config_.multigpu.rebalance_min_interval &&
                            rcb_calibrator_.ready();
                        if(common_balance_due_){
                            common_last_balance_step_=my_state.local_state.current_step;
                            morton_balance_.begin(domain_params_.morton_grid,domain_params_.morton_cuts);
                        }
                    }
                    if(common_balance_due_){
                        CUDA_CHECK(cudaSetDevice(gpu_id));
                        nvtxRangePushA("Morton GPU Balance Preparation");
                        int work_count=0;std::array<double,4> units{};
                        const auto* fmm_work=body_fitted_fmm::UnifiedFMMDispatcher::distributed_particle_work_device(work_count,units.data(),stream);
                        const auto* rows=morton_work_collectors_[gpu_id].collect_device(my_state,domain_params_,contacts_[gpu_id].pairs,
                            latest_contact_counts_[gpu_id],
                            sph_hydro_offsets_valid_[gpu_id]
                                ? d_sph_hydro_owned_offsets_[gpu_id] : nullptr,
                            sph_mode_,communicators_[gpu_id].halo_destination_mask(),stream);
                        if(work_count!=my_state.local_state.particles.num_local)throw std::runtime_error("stale common FMM work snapshot at repartition");
                        common_work_totals_[gpu_id]=units;
                        #pragma omp barrier
                        #pragma omp single
                        {
                            std::array<double,4> totals{},seconds{};
                            for(int peer=0;peer<num_gpus_;++peer){
                                for(int phase=0;phase<4;++phase)totals[phase]+=common_work_totals_[peer][phase];
                                seconds[0]+=common_metrics_[peer].tree_ms*1e-3;
                                seconds[1]+=common_metrics_[peer].far_ms*1e-3;
                                seconds[2]+=common_metrics_[peer].near_ms*1e-3;
                                seconds[3]+=common_metrics_[peer].communication_ms*1e-3;
                            }
                            for(int phase=0;phase<4;++phase)common_work_coefficients_[phase]=totals[phase]>0?seconds[phase]/totals[phase]:0;
                        }
                        auto coefficient=[&](RCBWorkPhase phase){return rcb_calibrator_.coefficient({phase,RCBFmmMode::NotApplicable,RCBLinkClass::PCIe});};
                        const double base=coefficient(RCBWorkPhase::Integration)+coefficient(RCBWorkPhase::Contacts);
                        const double contacts=coefficient(RCBWorkPhase::ContactPair);
                        const double transfer=coefficient(RCBWorkPhase::LinkTransfer);
                        const double sph_halo_bytes = sizeof(SphHaloParticle)
                            + sizeof(SphForceRefresh)
                            + (config_.sph.density_update_mode == 1 ? 0 : sizeof(SphThermoRefresh))
                            + (config_.acceleration.enable_self_gravity ? sizeof(SphHaloParticle) : 0);
                        const int sph_halo_rounds = 2
                            + (config_.sph.density_update_mode == 1 ? 0 : 1)
                            + (config_.acceleration.enable_self_gravity ? 1 : 0);
                        const double halo = (sph_mode_ ? sph_halo_rounds : 1)
                            * coefficient(RCBWorkPhase::HaloLifecycle)
                            + (sph_mode_ ? sph_halo_bytes : sizeof(PackedParticle)) * transfer;
                        const double gravity_frequency=1.0/std::max(1,respa_enabled?respa_macro_inner_count:1);
                        morton_balance_.set_local(gpu_id,rows,fmm_work,work_count,common_work_coefficients_.data(),
                            base,contacts,halo,gravity_frequency,stream);
                        #pragma omp barrier
                        #pragma omp single
                        {morton_balance_.sum_rank_totals();}
                        morton_balance_.choose_local_cuts(gpu_id);
                        #pragma omp barrier
                        morton_balance_.predict_local(gpu_id);
                        nvtxRangePop();
                        #pragma omp barrier
                        #pragma omp single
                        {
                            const bool accepted=morton_balance_.accept(config_.multigpu.rcb_min_predicted_improvement,
                                config_.multigpu.rebalance_min_interval,transfer);
                            std::cout<<"[FMM] Morton balance step="<<my_state.local_state.current_step<<" accepted="<<accepted
                                <<" predicted_max_ms="<<morton_balance_.current_max*1e3<<"->"<<morton_balance_.proposed_max*1e3
                                <<" migration_ms="<<morton_balance_.migration_seconds*1e3<<std::endl;
                            if(accepted){
                                DomainManager::set_morton_partition(domain_params_,domain_params_.morton_grid,morton_balance_.cuts());
                                common_partition_pending_=true;
                                common_fmm_depth_.notify_disruptive_event(common_completed_epoch_,DistributedFMMDepthCancelReason::Rebalance);
                                if(config_.acceleration.unified_fmm.adaptive_depth.enabled)common_fmm_command_=common_fmm_depth_.command_for_gpu(0);
                            }
                        }
                    }
                    #pragma omp barrier
                }

                if (!domain_params_.morton_enabled && !force_bootstrap_step && (!respa_enabled || respa_macro_closing)) {
                    nvtxRangePushA("Load Rebalance Check");

                    // Poll completed event-ring entries.  Neither an unfinished
                    // local GPU nor a lagging peer blocks timing publication.
                    harvest_timing_samples();
                    int64_t completed_step = my_state.local_state.current_step;
                    rebalance_mailbox_.publish_progress(gpu_id, completed_step);

                    if (gpu_id == 0 &&
                        config_.multigpu.rcb_cost_model == RCBCostModel::Adaptive) {
                        int64_t calibration_epoch = -1;
                        std::vector<RCBActiveSample> calibration_samples;
                        while (rcb_sample_mailbox_.try_collect_next(
                                   last_calibration_epoch_evaluated,
                                   calibration_epoch, calibration_samples)) {
                            last_calibration_epoch_evaluated = calibration_epoch;
                            const auto updates = rcb_calibrator_.observe_global_epoch(
                                calibration_epoch, calibration_samples);
                            if (config_.multigpu.rcb_calibration_verbose) {
                                for (const auto& update : updates) {
                                    std::cout << "[rcb calibration] epoch="
                                              << calibration_epoch
                                              << " phase="
                                              << static_cast<int>(update.key.phase)
                                              << " mode="
                                              << static_cast<int>(update.key.fmm_mode)
                                              << " link="
                                              << static_cast<int>(update.key.link_class)
                                              << " samples=" << update.accepted_samples
                                              << " seconds_per_unit="
                                              << update.coefficient_seconds_per_unit
                                              << std::endl;
                                }
                            }
                            calibration_samples.clear();
                        }
                    }

                    // GPU 0 evaluates the newest complete common sample without
                    // waiting for lagging publishers.  A request is placed at
                    // least two epochs beyond the most advanced worker so no
                    // thread can pass the conditional rendezvous unnoticed.
                    if (gpu_id == 0 && rebalance_mailbox_.pending_epoch() < 0 &&
                        !distributed_fmm_depth_.trial_active()) {
                        int64_t sample_step = -1;
                        std::vector<float> sample_times;
                        if (rebalance_mailbox_.try_collect_latest(
                                last_timing_epoch_evaluated,
                                sample_step, sample_times)) {
                            last_timing_epoch_evaluated = sample_step;
                            const bool model_ready =
                                config_.multigpu.rcb_cost_model == RCBCostModel::Legacy ||
                                adaptive_calibration_ready();
                            if (model_ready &&
                                rebalance_ctl_.should_rebalance(sample_times, sample_step)) {
                                int64_t target_step =
                                    rebalance_mailbox_.max_progress_epoch() + 2;
                                rebalance_mailbox_.schedule(target_step);
                            }
                        }
                    }

                    int64_t rebalance_step = rebalance_mailbox_.pending_epoch();
                    if (rebalance_step >= 0 && completed_step >= rebalance_step) {
                        // Domain boxes may change only after current-step kernels
                        // have consumed the old decomposition.  This local wait
                        // occurs solely for an actual collective RCB update.
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        #pragma omp barrier
                        if (gpu_id == 0) {
                            nvtxRangePushA("Load Rebalance");
                            if (config_.multigpu.rcb_cost_model == RCBCostModel::Legacy) {
                                DomainManager::update_load_balance(
                                    gpu_states_, domain_params_, balancer_);
                                distributed_fmm_depth_.notify_disruptive_event(
                                    static_cast<uint64_t>(rebalance_step),
                                    DistributedFMMDepthCancelReason::Rebalance);
                            } else {
                                const auto coefficient = [&](RCBWorkPhase phase,
                                                             RCBLinkClass link,
                                                             double fallback = 0.0) {
                                    return rcb_calibrator_.coefficient(
                                        {phase, RCBFmmMode::NotApplicable, link},
                                        fallback);
                                };
                                const auto blended = [&](RCBWorkPhase phase,
                                                         double fallback = 0.0) {
                                    return rcb_calibrator_.blended_fmm_coefficient(
                                        phase, RCBLinkClass::PCIe, fallback);
                                };

                                std::vector<AdaptiveRCBInput> inputs(num_gpus_);
                                // An RCB commit is delayed until a closed
                                // macro.  Its cohort represents K fast
                                // migration/halo/contact steps plus exactly
                                // one FMM/LET endpoint evaluation.
                                const double fast_work_multiplier = respa_enabled
                                    ? static_cast<double>(respa_macro_inner_count)
                                    : 1.0;
                                for (int32_t g = 0; g < num_gpus_; ++g) {
                                    CUDA_CHECK(cudaSetDevice(g));
                                    AdaptiveRCBInput& input = inputs[g];
                                    input.owned_count =
                                        gpu_states_[g].subdomain.num_local_particles;
                                    input.contact_pairs = contacts_[g].pairs;
                                    input.contact_pair_count = latest_contact_counts_[g];
                                    input.sph_mode = sph_mode_;
                                    input.sph_offsets = sph_mode_ && sph_hydro_offsets_valid_[g]
                                        ? d_sph_hydro_owned_offsets_[g] : nullptr;
                                    input.halo_destination_mask =
                                        reinterpret_cast<const uint32_t*>(
                                            communicators_[g].halo_destination_mask());

                                    input.coefficients.base = static_cast<float>(
                                        fast_work_multiplier * (
                                            coefficient(RCBWorkPhase::Integration,
                                                        RCBLinkClass::PCIe)
                                          + coefficient(RCBWorkPhase::Contacts,
                                                        RCBLinkClass::PCIe))
                                      + blended(RCBWorkPhase::FmmParticle));
                                    input.coefficients.contact_pair =
                                        static_cast<float>(fast_work_multiplier * coefficient(
                                            RCBWorkPhase::ContactPair,
                                            RCBLinkClass::PCIe));
                                    input.coefficients.m2l = static_cast<float>(
                                        blended(RCBWorkPhase::FmmNode));
                                    input.coefficients.p2p = static_cast<float>(
                                        blended(RCBWorkPhase::FmmLeaf));
                                    input.coefficients.let_source_node =
                                        static_cast<float>(coefficient(
                                            RCBWorkPhase::LetSource,
                                            RCBLinkClass::PCIe));
                                    input.coefficients.let_source_particle =
                                        input.coefficients.let_source_node;
                                    input.coefficients.let_receiver =
                                        static_cast<float>(blended(
                                            RCBWorkPhase::LetReceiver));
                                    const auto recv_summary =
                                        let_comms_[g].fmm_recv_summary();
                                    input.receiver_uniform_units = static_cast<float>(
                                        recv_summary.node_records +
                                        recv_summary.particle_records);

                                    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
                                        if (peer == g) continue;
                                        const RCBLinkClass link = calibration_link_class(
                                            communicators_[g].link_class_to(peer));
                                        const double lifecycle = coefficient(
                                            RCBWorkPhase::HaloLifecycle, link,
                                            coefficient(RCBWorkPhase::HaloLifecycle,
                                                        RCBLinkClass::PCIe));
                                        const double transfer_per_byte = coefficient(
                                            RCBWorkPhase::LinkTransfer, link,
                                            coefficient(RCBWorkPhase::LinkTransfer,
                                                        RCBLinkClass::PCIe));
                                        const double sph_halo_bytes = sizeof(SphHaloParticle)
                                            + sizeof(SphForceRefresh)
                                            + (config_.sph.density_update_mode == 1 ? 0 : sizeof(SphThermoRefresh))
                                            + (config_.acceleration.enable_self_gravity ? sizeof(SphHaloParticle) : 0);
                                        const int sph_halo_rounds = 2
                                            + (config_.sph.density_update_mode == 1 ? 0 : 1)
                                            + (config_.acceleration.enable_self_gravity ? 1 : 0);
                                        input.halo_peer_cost[peer] = static_cast<float>(
                                            fast_work_multiplier * (
                                                (sph_mode_ ? sph_halo_rounds : 1) * lifecycle
                                                + transfer_per_byte * (sph_mode_ ? sph_halo_bytes
                                                                              : sizeof(PackedParticle))));
                                    }

                                    if (config_.acceleration.enable_self_gravity &&
                                        config_.acceleration.gravity_method ==
                                            SelfGravityMethod::BFFMM) {
                                        const auto view = body_fitted_fmm::
                                            UnifiedFMMDispatcher::get_workload_view();
                                        if (view.valid &&
                                            view.num_particles == input.owned_count) {
                                            // RCB's public POD view is declared
                                            // against the legacy global relative
                                            // namespace; the manager implementation
                                            // instantiates the same layouts inside
                                            // cudadem.  These are borrowed ABI-
                                            // identical device records.
                                            input.fmm.nodes = reinterpret_cast<
                                                const ::body_fitted_fmm::UnifiedNode*>(
                                                    view.nodes);
                                            input.fmm.sorted_to_original =
                                                view.sorted_to_original;
                                            input.fmm.num_nodes = view.num_nodes;
                                            input.fmm.m2l_same = reinterpret_cast<
                                                const ::body_fitted_fmm::M2LTask*>(
                                                    view.m2l_same);
                                            input.fmm.m2l_cross = reinterpret_cast<
                                                const ::body_fitted_fmm::M2LTask*>(
                                                    view.m2l_cross);
                                            input.fmm.p2p_regular = reinterpret_cast<
                                                const ::body_fitted_fmm::PairTask*>(
                                                    view.p2p_regular);
                                            input.fmm.p2p_self = reinterpret_cast<
                                                const ::body_fitted_fmm::PairTask*>(
                                                    view.p2p_self);
                                            input.fmm.m2l_same_count =
                                                view.m2l_same_count;
                                            input.fmm.m2l_cross_count =
                                                view.m2l_cross_count;
                                            input.fmm.p2p_regular_count =
                                                view.p2p_regular_count;
                                            input.fmm.p2p_self_count =
                                                view.p2p_self_count;
                                            input.fmm.grouped_targets = reinterpret_cast<
                                                const ::body_fitted_fmm::GroupedTargetInteractions*>(
                                                    view.grouped_targets);
                                            input.fmm.children_compact =
                                                view.children_compact;
                                            input.fmm.metadata = reinterpret_cast<
                                                const ::body_fitted_fmm::PersistentTreeMetadata*>(
                                                    view.metadata);
                                            input.fmm.node_capacity =
                                                view.node_capacity;
                                            input.let_source_work =
                                                let_comms_[g].fmm_source_work();
                                            input.let_source_work_count =
                                                let_comms_[g].fmm_source_work_count();
                                        }
                                    }
                                }
                                CUDA_CHECK(cudaSetDevice(0));
                                const RCBProposal proposal =
                                    balancer_.compute_adaptive_proposal(
                                        gpu_states_, domain_params_, inputs,
                                        config_.multigpu.rcb_min_predicted_improvement);
                                if (proposal.valid && proposal.eligible) {
                                    DomainManager::apply_accepted_rcb_boxes(
                                        gpu_states_, domain_params_, proposal.boxes,
                                        num_gpus_);
                                    distributed_fmm_depth_.notify_disruptive_event(
                                        static_cast<uint64_t>(rebalance_step),
                                        DistributedFMMDepthCancelReason::Rebalance);
                                    std::cout << "[rcb adaptive] accepted epoch="
                                              << rebalance_step << " ratio="
                                              << proposal.current_tmax_over_avg << "->"
                                              << proposal.proposed_tmax_over_avg
                                              << std::endl;
                                } else if (proposal.valid) {
                                    std::cout << "[rcb adaptive] rejected epoch="
                                              << rebalance_step << " ratio="
                                              << proposal.current_tmax_over_avg << "->"
                                              << proposal.proposed_tmax_over_avg
                                              << " gate="
                                              << config_.multigpu.rcb_min_predicted_improvement
                                              << std::endl;
                                } else {
                                    std::cout << "[rcb adaptive] invalid proposal at epoch="
                                              << rebalance_step
                                              << "; retaining current boxes" << std::endl;
                                }
                            }
                            rebalance_ctl_.notify_rebalanced(rebalance_step);
                            rebalance_mailbox_.complete(rebalance_step);
                            nvtxRangePop();
                        }
                        #pragma omp barrier
                        CUDA_CHECK(cudaSetDevice(gpu_id));
                    }
                    nvtxRangePop();
                }
                if (!force_bootstrap_step && config_.io.checkpoint_enabled && (!respa_enabled || respa_macro_closing) &&
                    my_state.local_state.current_step - checkpoint_cadence_step_ >= config_.io.checkpoint_interval) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    harvest_timing_samples();
                    harvest_distributed_fmm_samples();
                    #pragma omp barrier
                    #pragma omp single
                    {
                        distributed_fmm_depth_.process_ready_epochs();
                        int64_t epoch; std::vector<RCBActiveSample> samples;
                        while (rcb_sample_mailbox_.try_collect_next(common_calibration_cursor_,epoch,samples)) {
                            common_calibration_cursor_=epoch; rcb_calibrator_.observe_global_epoch(epoch,samples); samples.clear();
                        }
                        try { save_output(true); }
                        catch (...) {
                            worker_failure = std::current_exception();
                            checkpoint_failed = true;
                        }
                    }
                    #pragma omp barrier
                    if (checkpoint_failed) std::rethrow_exception(worker_failure);
                    CUDA_CHECK(cudaSetDevice(gpu_id));
                }

                // Only this collective closed-boundary sample reads the signal
                // flag. Every worker sees the same exit decision after single's
                // implicit barrier, so none skips a later rank barrier alone.
                if (force_bootstrap_step || !respa_enabled || respa_macro_closing) {
                    #pragma omp barrier
                    #pragma omp single
                    { collective_stop = cooperative_stop_requested(); }
                }
                ++particle_exchange_epoch;
                #ifdef CUDADEM_ENABLE_PROFILING
                nvtxRangePop(); // SimulationStep
                #endif
            }

            // Drain every device before the host gathers the stop checkpoint.
            CUDA_CHECK(cudaDeviceSynchronize());
            harvest_timing_samples();
            harvest_distributed_fmm_samples();
            #pragma omp barrier
            if (gpu_id == 0)
                distributed_fmm_depth_.process_ready_epochs();
            #pragma omp barrier
            } catch (...) {
                // LET/phase failures are collective failures: wake directed
                // waiters before this worker unwinds, retain the first cause,
                // and rethrow only after the OpenMP region has joined.
                phase_coordinator_.abort();
                std::lock_guard<std::mutex> lock(worker_failure_mutex);
                if (!worker_failure) worker_failure = std::current_exception();
            }
        }

        if (worker_failure) std::rethrow_exception(worker_failure);
        collective_stop = collective_stop || cooperative_stop_requested();

        if ((collective_stop || config_.io.checkpoint_enabled) &&
            last_checkpoint_step_ != gpu_states_[0].local_state.current_step)
            save_output(true);
        collective_stop = collective_stop || cooperative_stop_requested();
        if (collective_stop) {
            if (last_checkpoint_step_ != gpu_states_[0].local_state.current_step) save_output(true);
            std::cout << "[cooperative-stop] signal=" << cooperative_stop_signal.load(std::memory_order_relaxed)
                      << " checkpoint=" << generate_checkpoint_path(config_, gpu_states_[0].local_state.current_step)
                      << " step=" << gpu_states_[0].local_state.current_step
                      << " time=" << gpu_states_[0].local_state.current_time << std::endl;
            return;
        }
        if (last_output_step_ != gpu_states_[0].local_state.current_step)
            save_output();
        write_distributed_fmm_diagnostics();

        if (config_.io.output_directory[0] != '\0' || config_.io.timing_receipt_path[0] != '\0') {
            const auto& final_state = gpu_states_[0].local_state;
            const int64_t completed_steps = final_state.current_step - timing_start_step;
            const double simulated_seconds = final_state.current_time - timing_start_time;
            const double wall_seconds = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - wall_start).count();
            const std::filesystem::path path = config_.io.timing_receipt_path[0] != '\0'
                ? std::filesystem::path(config_.io.timing_receipt_path)
                : std::filesystem::path(config_.io.output_directory) / "timing_receipt.json";
            if (std::filesystem::exists(path))
                throw std::runtime_error("refusing to overwrite timing receipt: " + path.string());
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            std::ofstream receipt(path);
            if (!receipt) throw std::runtime_error("cannot write multi-GPU timing receipt");
            int64_t live_particles = 0;
            for (const auto& rank : gpu_states_) live_particles += rank.subdomain.num_local_particles;
            receipt << std::setprecision(17)
                << "{\n  \"schema_version\": \"granit-timing-v1\",\n"
                << "  \"num_gpus\": " << num_gpus_ << ",\n"
                << "  \"start_step\": " << timing_start_step << ",\n"
                << "  \"end_step\": " << final_state.current_step << ",\n"
                << "  \"completed_steps\": " << completed_steps << ",\n"
                << "  \"simulated_seconds\": " << simulated_seconds << ",\n"
                << "  \"wall_seconds\": " << wall_seconds << ",\n"
                << "  \"configured_warmup_steps\": " << config_.io.timing_warmup_steps << ",\n"
                << "  \"configured_measure_steps\": " << config_.io.timing_measure_steps << ",\n"
                << "  \"compute_window_completed\": " << (timing_window_finished ? "true" : "false") << ",\n"
                << "  \"compute_window_start_step\": " << timing_window_start_step << ",\n"
                << "  \"compute_window_end_step\": " << timing_window_end_step << ",\n"
                << "  \"compute_window_wall_seconds\": " << timing_window_wall_seconds << ",\n"
                << "  \"compute_window_external_gravity_evaluations\": 0,\n"
                << "  \"self_gravity_evaluations\": "
                << (config_.acceleration.enable_self_gravity ? gravity_evaluation_epoch_ : 0) << ",\n"
                << "  \"wall_seconds_per_step\": " << (completed_steps > 0 ? wall_seconds / completed_steps : 0.0) << ",\n"
                << "  \"simulated_seconds_per_wall_second\": " << (wall_seconds > 0.0 ? simulated_seconds / wall_seconds : 0.0) << ",\n"
                << "  \"live_particles\": " << live_particles << ",\n"
                << "  \"external_gravity_evaluations\": 0\n}\n";
            if (!receipt) throw std::runtime_error("failed writing multi-GPU timing receipt");
        }

        std::cout << "\n========== Multi-GPU Simulation Complete ===========" << std::endl;
        if (adaptive_yorp_fatal_)
            std::cout << "Adaptive YORP reached the replicated FATAL state after "
                         "invalid physical metrics; all workers stopped at the "
                         "same completed step." << std::endl;
    }

    static void validate_sph_checkpoint_fields(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        UnifiedHeader header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input || header.physics != static_cast<uint8_t>(BinaryPhysics::SPH))
            throw std::invalid_argument("Expected a complete SPH checkpoint");
        std::vector<FieldDescriptor> descriptors(header.num_descriptors);
        input.read(reinterpret_cast<char*>(descriptors.data()),
                   descriptors.size() * sizeof(FieldDescriptor));
        if (!input) throw std::invalid_argument("Truncated SPH checkpoint descriptors");
        for (const auto required : {
                FieldID::FORCE, FieldID::DENSITY, FieldID::ENERGY,
                FieldID::SMOOTHING_LENGTH, FieldID::PRESSURE, FieldID::C_SOUND,
                FieldID::DAMAGE, FieldID::CRACK_STRAIN, FieldID::DISTENSION,
                FieldID::FLAWS_NTOT, FieldID::FLAWS_STRAIN_MIN,
                FieldID::FLAWS_STRAIN_MAX, FieldID::FLAWS_M0,
                FieldID::S_XX, FieldID::S_XY, FieldID::S_XZ,
                FieldID::S_YY, FieldID::S_YZ, FieldID::S_ZZ,
                FieldID::DRHODT, FieldID::DEDT}) {
            if (std::none_of(descriptors.begin(), descriptors.end(),
                    [&](const FieldDescriptor& d) { return d.field_id == static_cast<uint16_t>(required); }))
                throw std::invalid_argument("Incomplete SPH checkpoint: missing field " +
                                            std::to_string(static_cast<uint16_t>(required)));
        }
    }

    void save_output(bool checkpoint = false) {
        // Gather owned rows only.  Ghost rows are replicated communication
        // state and must never appear in a global snapshot.
        int64_t total_particles = 0;
        for (int g = 0; g < num_gpus_; ++g) {
            total_particles += gpu_states_[g].subdomain.num_local_particles;
        }
        if (total_particles <= 0 ||
            total_particles > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            std::cerr << "WARNING: Cannot write multi-GPU output with particle count "
                      << total_particles << std::endl;
            return;
        }
        const int32_t live_particle_count = static_cast<int32_t>(total_particles);
        const int32_t particle_count = live_particle_count + dead_rows_.size();

        // These are exactly the fields consumed by write_unified_binary for
        // ordinary DEM and SPH output records.  Keeping one correctly typed
        // gather eliminates the former float/double size mismatch and ensures
        // material IDs and SPH thermodynamic state are present.
        std::vector<double> h_pos_x(particle_count), h_pos_y(particle_count), h_pos_z(particle_count);
        std::vector<double> h_vel_x(particle_count), h_vel_y(particle_count), h_vel_z(particle_count);
        std::vector<double> h_force_x(particle_count), h_force_y(particle_count), h_force_z(particle_count);
        std::vector<double> h_omega_x(particle_count), h_omega_y(particle_count), h_omega_z(particle_count);
        std::vector<double> h_torque_x(particle_count), h_torque_y(particle_count), h_torque_z(particle_count);
        std::vector<double> h_energy(particle_count);
        std::vector<double> h_energy_cold(particle_count);
        std::vector<double> h_energy_thermal_peak(particle_count);
        std::vector<double> h_pressure_peak(particle_count);
        std::vector<double> h_fracture_area(particle_count);
        std::vector<double> h_fragment_length(particle_count);
        std::vector<double> h_damping_energy(particle_count);

        std::vector<float> h_radius(particle_count), h_mass(particle_count), h_moi(particle_count);
        std::vector<double> h_density(particle_count);
        std::vector<float> h_smoothing_length(particle_count);
        std::vector<float> h_pressure(particle_count), h_c_sound(particle_count), h_damage(particle_count);
        std::vector<int32_t> h_global_id(particle_count), h_material_id(particle_count), h_clump_id(particle_count);
        std::vector<float> h_clump_off_x(particle_count),h_clump_off_y(particle_count),h_clump_off_z(particle_count);
        std::vector<int64_t> h_winding_x(particle_count), h_winding_y(particle_count), h_winding_z(particle_count);

        std::vector<double> h_checkpoint_qw(checkpoint?particle_count:0),h_checkpoint_qx(checkpoint?particle_count:0),
            h_checkpoint_qy(checkpoint?particle_count:0),h_checkpoint_qz(checkpoint?particle_count:0);
        std::vector<int32_t> h_checkpoint_contacts(checkpoint?particle_count:0),h_checkpoint_wall_counts(checkpoint?particle_count:0);
        std::vector<ContactHistory> checkpoint_contact_blob;
        std::vector<std::vector<WallContactHistory>> checkpoint_wall_by_row;
        std::vector<DEMExactRow> checkpoint_exact_rows;
        std::vector<WallContactHistory> checkpoint_wall_rows;
        int checkpoint_contact_stride=0,checkpoint_wall_stride=0;
        int32_t checkpoint_total_contacts=0;
        std::vector<double> h_checkpoint_gx(checkpoint ? particle_count : 0), h_checkpoint_gy(checkpoint ? particle_count : 0), h_checkpoint_gz(checkpoint ? particle_count : 0);
        std::vector<double> h_checkpoint_rho_rate(checkpoint ? particle_count : 0);
        std::vector<double> h_checkpoint_dedt(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_xx(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_xy(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_xz(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_yy(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_yz(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_S_zz(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_crack_strain(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_distension(checkpoint ? particle_count : 0);
        std::vector<int32_t> h_checkpoint_flaws_ntot(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_flaws_strain_min(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_flaws_strain_max(checkpoint ? particle_count : 0);
        std::vector<float> h_checkpoint_flaws_m0(checkpoint ? particle_count : 0);

        auto gather_field = [&](auto& host, auto particle_member) {
            using Value = typename std::decay_t<decltype(host)>::value_type;
            int64_t offset = 0;
            for (int g = 0; g < num_gpus_; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                const int32_t n = gpu_states_[g].subdomain.num_local_particles;
                if (n > 0) {
                    const ParticleSoA& particles = gpu_states_[g].local_state.particles;
                    const Value* source = particles.*particle_member;
                    CUDA_CHECK(cudaMemcpy(host.data() + offset, source,
                                          static_cast<size_t>(n) * sizeof(Value),
                                          cudaMemcpyDeviceToHost));
                    offset += n;
                }
            }
        };

        gather_field(h_pos_x, &ParticleSoA::pos_x);
        gather_field(h_pos_y, &ParticleSoA::pos_y);
        gather_field(h_pos_z, &ParticleSoA::pos_z);
        gather_field(h_vel_x, &ParticleSoA::vel_x);
        gather_field(h_vel_y, &ParticleSoA::vel_y);
        gather_field(h_vel_z, &ParticleSoA::vel_z);
        gather_field(h_force_x, &ParticleSoA::force_x);
        gather_field(h_force_y, &ParticleSoA::force_y);
        gather_field(h_force_z, &ParticleSoA::force_z);
        gather_field(h_omega_x, &ParticleSoA::omega_x);
        gather_field(h_omega_y, &ParticleSoA::omega_y);
        gather_field(h_omega_z, &ParticleSoA::omega_z);
        gather_field(h_torque_x, &ParticleSoA::torque_x);
        gather_field(h_torque_y, &ParticleSoA::torque_y);
        gather_field(h_torque_z, &ParticleSoA::torque_z);
        gather_field(h_energy, &ParticleSoA::energy);
        gather_field(h_energy_cold, &ParticleSoA::energy_cold);
        gather_field(h_energy_thermal_peak, &ParticleSoA::energy_thermal_peak);
        gather_field(h_pressure_peak, &ParticleSoA::pressure_peak);
        gather_field(h_fracture_area, &ParticleSoA::fracture_area);
        gather_field(h_fragment_length, &ParticleSoA::fragment_length);
        gather_field(h_damping_energy, &ParticleSoA::damping_energy);

        gather_field(h_radius, &ParticleSoA::radius);
        gather_field(h_mass, &ParticleSoA::mass);
        gather_field(h_moi, &ParticleSoA::moment_of_inertia);
        gather_field(h_smoothing_length, &ParticleSoA::smoothing_length);
        gather_field(h_density, &ParticleSoA::density);
        gather_field(h_pressure, &ParticleSoA::pressure);
        gather_field(h_c_sound, &ParticleSoA::c_sound);
        gather_field(h_damage, &ParticleSoA::damage);
        gather_field(h_global_id, &ParticleSoA::global_id);
        gather_field(h_material_id, &ParticleSoA::material_id);
        gather_field(h_clump_id, &ParticleSoA::clump_id);
        gather_field(h_clump_off_x, &ParticleSoA::clump_off_x);
        gather_field(h_clump_off_y, &ParticleSoA::clump_off_y);
        gather_field(h_clump_off_z, &ParticleSoA::clump_off_z);
        gather_field(h_winding_x, &ParticleSoA::winding_x);
        gather_field(h_winding_y, &ParticleSoA::winding_y);
        gather_field(h_winding_z, &ParticleSoA::winding_z);

        if (checkpoint) {
            gather_field(h_checkpoint_qw,&ParticleSoA::quat_w);gather_field(h_checkpoint_qx,&ParticleSoA::quat_x);
            gather_field(h_checkpoint_qy,&ParticleSoA::quat_y);gather_field(h_checkpoint_qz,&ParticleSoA::quat_z);
            gather_field(h_checkpoint_contacts,&ParticleSoA::num_contacts);
            gather_field(h_checkpoint_wall_counts,&ParticleSoA::num_wall_contacts);
            if(!sph_mode_) {
                checkpoint_contact_stride=gpu_states_[0].local_state.contact_history.max_contacts_per_particle;
                checkpoint_wall_stride=gpu_states_[0].local_state.wall_contact_history.max_contacts_per_particle;
                const int64_t total=std::accumulate(h_checkpoint_contacts.begin(),h_checkpoint_contacts.end(),int64_t{0});
                if(total>INT32_MAX)throw std::runtime_error("Checkpoint active contact count exceeds int32");
                checkpoint_total_contacts=static_cast<int32_t>(total);
                if(total)checkpoint_contact_blob.resize(static_cast<size_t>(particle_count)*checkpoint_contact_stride);
                checkpoint_wall_by_row.resize(particle_count);
                size_t offset=0;
                for(int g=0;g<num_gpus_;++g){CUDA_CHECK(cudaSetDevice(g));const auto& local=gpu_states_[g].local_state;
                    const int n=gpu_states_[g].subdomain.num_local_particles;
                    if(local.contact_history.max_contacts_per_particle!=checkpoint_contact_stride||
                       local.wall_contact_history.max_contacts_per_particle!=checkpoint_wall_stride)
                        throw std::runtime_error("Checkpoint requires matching history strides on all ranks");
                    std::vector<int32_t> contact_offsets(n), wall_offsets(n);
                    if (n) {
                        CUDA_CHECK(cudaMemcpy(contact_offsets.data(), local.particles.contact_history_offset,
                            n*sizeof(int32_t), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(wall_offsets.data(), local.particles.wall_history_offset,
                            n*sizeof(int32_t), cudaMemcpyDeviceToHost));
                    }
                    std::vector<ContactHistory> contact_pool;
                    if (total && n) {
                        contact_pool.resize(local.contact_history.capacity);
                        CUDA_CHECK(cudaMemcpy(contact_pool.data(), local.contact_history.data,
                            contact_pool.size()*sizeof(ContactHistory), cudaMemcpyDeviceToHost));
                    }
                    for(int i=0;i<n;++i){
                        const int contact_count=h_checkpoint_contacts[offset+i];
                        const int count=h_checkpoint_wall_counts[offset+i];
                        if(contact_count<0||contact_count>checkpoint_contact_stride||
                           contact_offsets[i]<0||int64_t(contact_offsets[i])+checkpoint_contact_stride>local.contact_history.capacity)
                            throw std::runtime_error("Invalid particle contact row at checkpoint");
                        if(total)std::copy_n(contact_pool.data()+contact_offsets[i], checkpoint_contact_stride,
                            checkpoint_contact_blob.data()+(offset+i)*checkpoint_contact_stride);
                        if(count<0||count>checkpoint_wall_stride||wall_offsets[i]<0||
                           int64_t(wall_offsets[i])+checkpoint_wall_stride>local.wall_contact_history.capacity)
                            throw std::runtime_error("Invalid particle wall-history row at checkpoint");
                        auto& rows=checkpoint_wall_by_row[offset+i];rows.resize(count);
                        if(count)CUDA_CHECK(cudaMemcpy(rows.data(),local.wall_contact_history.data+wall_offsets[i],
                            count*sizeof(WallContactHistory),cudaMemcpyDeviceToHost));}
                    offset+=n;
                }
            }
            gather_field(h_checkpoint_gx, &ParticleSoA::gravity_force_x);
            gather_field(h_checkpoint_gy, &ParticleSoA::gravity_force_y);
            gather_field(h_checkpoint_gz, &ParticleSoA::gravity_force_z);
            gather_field(h_checkpoint_rho_rate, &ParticleSoA::rho_rate);
            gather_field(h_checkpoint_dedt, &ParticleSoA::dedt);
            gather_field(h_checkpoint_S_xx, &ParticleSoA::S_xx);
            gather_field(h_checkpoint_S_xy, &ParticleSoA::S_xy);
            gather_field(h_checkpoint_S_xz, &ParticleSoA::S_xz);
            gather_field(h_checkpoint_S_yy, &ParticleSoA::S_yy);
            gather_field(h_checkpoint_S_yz, &ParticleSoA::S_yz);
            gather_field(h_checkpoint_S_zz, &ParticleSoA::S_zz);
            gather_field(h_checkpoint_crack_strain, &ParticleSoA::crack_strain);
            gather_field(h_checkpoint_distension, &ParticleSoA::distension);
            gather_field(h_checkpoint_flaws_ntot, &ParticleSoA::flaws_ntot);
            gather_field(h_checkpoint_flaws_strain_min, &ParticleSoA::flaws_strain_min);
            gather_field(h_checkpoint_flaws_strain_max, &ParticleSoA::flaws_strain_max);
            gather_field(h_checkpoint_flaws_m0, &ParticleSoA::flaws_m0);
        }

        const bool respa_output = config_.acceleration.enable_self_gravity &&
            (sph_mode_ || config_.acceleration.gravity_update_rate > 1);
        if (respa_output && !checkpoint) {
            // Retained outer data is acceleration; compose a transient host
            // output view instead of mutating the fast contact-force arrays.
            std::vector<double> gx(live_particle_count), gy(live_particle_count),
                                gz(live_particle_count);
            int64_t offset = 0;
            for (int g = 0; g < num_gpus_; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                const int32_t n = gpu_states_[g].subdomain.num_local_particles;
                if (n > 0) {
                    const ParticleSoA& p = gpu_states_[g].local_state.particles;
                    CUDA_CHECK(cudaMemcpy(gx.data() + offset, p.gravity_force_x,
                                          n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(gy.data() + offset, p.gravity_force_y,
                                          n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(gz.data() + offset, p.gravity_force_z,
                                          n * sizeof(double), cudaMemcpyDeviceToHost));
                    offset += n;
                }
            }
            for (int32_t i = 0; i < live_particle_count; ++i) {
                const double m = static_cast<double>(h_mass[i]);
                h_force_x[i] += m * gx[i];
                h_force_y[i] += m * gy[i];
                h_force_z[i] += m * gz[i];
            }
        }

        if (config_.shear_box.enabled && !checkpoint) {
            // Hill acceleration belongs to the output force view; the native
            // kick history must retain only the perturbing/contact force.
            for (int32_t i = 0; i < live_particle_count; ++i) {
                double ax, ay, az;
                hill_acceleration(config_.shear_box, h_pos_x[i], h_pos_z[i],
                                  h_vel_x[i], h_vel_y[i], ax, ay, az);
                h_force_x[i] += static_cast<double>(h_mass[i]) * ax;
                h_force_y[i] += static_cast<double>(h_mass[i]) * ay;
                h_force_z[i] += static_cast<double>(h_mass[i]) * az;
            }
        }

        auto append_dead = [&](auto& host, const auto& dead) {
            std::copy(dead.begin(), dead.end(), host.begin() + live_particle_count);
        };
        append_dead(h_pos_x,dead_rows_.pos_x); append_dead(h_pos_y,dead_rows_.pos_y); append_dead(h_pos_z,dead_rows_.pos_z);
        append_dead(h_vel_x,dead_rows_.vel_x); append_dead(h_vel_y,dead_rows_.vel_y); append_dead(h_vel_z,dead_rows_.vel_z);
        append_dead(h_force_x,dead_rows_.force_x); append_dead(h_force_y,dead_rows_.force_y); append_dead(h_force_z,dead_rows_.force_z);
        append_dead(h_omega_x,dead_rows_.omega_x); append_dead(h_omega_y,dead_rows_.omega_y); append_dead(h_omega_z,dead_rows_.omega_z);
        append_dead(h_torque_x,dead_rows_.torque_x); append_dead(h_torque_y,dead_rows_.torque_y); append_dead(h_torque_z,dead_rows_.torque_z);
        append_dead(h_energy,dead_rows_.energy); append_dead(h_radius,dead_rows_.radius); append_dead(h_mass,dead_rows_.mass); append_dead(h_moi,dead_rows_.moi);
        append_dead(h_smoothing_length,dead_rows_.smoothing_length); append_dead(h_density,dead_rows_.density); append_dead(h_pressure,dead_rows_.pressure); append_dead(h_c_sound,dead_rows_.c_sound); append_dead(h_damage,dead_rows_.damage);
        append_dead(h_clump_off_x,dead_rows_.clump_off_x); append_dead(h_clump_off_y,dead_rows_.clump_off_y); append_dead(h_clump_off_z,dead_rows_.clump_off_z);
        append_dead(h_global_id,dead_rows_.global_id); append_dead(h_material_id,dead_rows_.material_id); append_dead(h_clump_id,dead_rows_.clump_id);
        append_dead(h_winding_x,dead_rows_.winding_x); append_dead(h_winding_y,dead_rows_.winding_y); append_dead(h_winding_z,dead_rows_.winding_z);

        // Stable global-ID order makes single-vs-multi output comparisons and
        // downstream conversion deterministic despite migration/rebalancing.
        std::vector<int32_t> indices(particle_count);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
            return h_global_id[a] < h_global_id[b];
        });

        auto reorder = [&](auto& vec) {
            auto temp = vec;
            for (int32_t i = 0; i < particle_count; ++i) {
                vec[i] = temp[indices[i]];
            }
        };

        reorder(h_pos_x); reorder(h_pos_y); reorder(h_pos_z);
        reorder(h_vel_x); reorder(h_vel_y); reorder(h_vel_z);
        reorder(h_force_x); reorder(h_force_y); reorder(h_force_z);
        reorder(h_omega_x); reorder(h_omega_y); reorder(h_omega_z);
        reorder(h_torque_x); reorder(h_torque_y); reorder(h_torque_z);
        reorder(h_energy);
        reorder(h_energy_cold);
        reorder(h_energy_thermal_peak);
        reorder(h_pressure_peak);
        reorder(h_fracture_area);
        reorder(h_fragment_length);
        reorder(h_damping_energy);

        reorder(h_radius); reorder(h_mass); reorder(h_moi);
        reorder(h_smoothing_length); reorder(h_density); reorder(h_pressure);
        reorder(h_c_sound); reorder(h_damage);
        reorder(h_global_id); reorder(h_material_id); reorder(h_clump_id);
        reorder(h_clump_off_x); reorder(h_clump_off_y); reorder(h_clump_off_z);
        reorder(h_winding_x); reorder(h_winding_y); reorder(h_winding_z);

        if (checkpoint) {
            reorder(h_checkpoint_qw);reorder(h_checkpoint_qx);reorder(h_checkpoint_qy);reorder(h_checkpoint_qz);
            reorder(h_checkpoint_contacts);reorder(h_checkpoint_wall_counts);
            if(!sph_mode_) {
                if(!checkpoint_contact_blob.empty()) {
                    auto original=std::move(checkpoint_contact_blob);
                    checkpoint_contact_blob.resize(original.size());
                    for(int i=0;i<particle_count;++i)std::copy_n(original.data()+static_cast<size_t>(indices[i])*checkpoint_contact_stride,
                        checkpoint_contact_stride,checkpoint_contact_blob.data()+static_cast<size_t>(i)*checkpoint_contact_stride);
                }
                checkpoint_exact_rows.resize(particle_count);
                for(int i=0;i<particle_count;++i){auto& row=checkpoint_exact_rows[i];row.gid=h_global_id[i];
                    row.wall_count=h_checkpoint_wall_counts[i];row.force[0]=h_force_x[i];row.force[1]=h_force_y[i];row.force[2]=h_force_z[i];
                    row.torque[0]=h_torque_x[i];row.torque[1]=h_torque_y[i];row.torque[2]=h_torque_z[i];
                    const auto& walls=checkpoint_wall_by_row[indices[i]];checkpoint_wall_rows.insert(checkpoint_wall_rows.end(),walls.begin(),walls.end());}
            }
            reorder(h_checkpoint_gx); reorder(h_checkpoint_gy); reorder(h_checkpoint_gz);
            reorder(h_checkpoint_rho_rate);
            reorder(h_checkpoint_dedt);
            reorder(h_checkpoint_S_xx);
            reorder(h_checkpoint_S_xy);
            reorder(h_checkpoint_S_xz);
            reorder(h_checkpoint_S_yy);
            reorder(h_checkpoint_S_yz);
            reorder(h_checkpoint_S_zz);
            reorder(h_checkpoint_crack_strain);
            reorder(h_checkpoint_distension);
            reorder(h_checkpoint_flaws_ntot);
            reorder(h_checkpoint_flaws_strain_min);
            reorder(h_checkpoint_flaws_strain_max);
            reorder(h_checkpoint_flaws_m0);
        }

        std::ostringstream oss;
        oss << config_.io.output_directory << "/output_step_"
            << std::setfill('0') << std::setw(10) << gpu_states_[0].local_state.current_step << ".bin";

        std::string filepath = checkpoint
            ? generate_checkpoint_path(config_, gpu_states_[0].local_state.current_step)
            : oss.str();

        std::unique_ptr<CheckpointTransaction> transaction;
        if (checkpoint) {
            if (dead_rows_.size() != 0)
                throw std::runtime_error("Distributed checkpoints require all rows live");
            transaction.reset(new CheckpointTransaction(filepath));
            filepath = transaction->staging_path();
        }
        std::filesystem::path output_dir = std::filesystem::path(filepath).parent_path();
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        // The unified writer operates on a device SoA.  Materialize the sorted
        // global snapshot on GPU 0, use the same v2 schema as single-GPU output,
        // then immediately release this temporary state.
        CUDA_CHECK(cudaSetDevice(0));
        SimulationState output_state;
        allocate_particle_soa(output_state.particles, particle_count);
        output_state.particles.num_particles = particle_count;
        output_state.particles.num_local = particle_count;
        output_state.particles.num_ghost = 0;
        output_state.particles.input_has_clump_offset=distributed_clumps_.enabled();
        output_state.particles.input_has_winding=config_.shear_box.enabled;
        output_state.current_step = gpu_states_[0].local_state.current_step;
        output_state.current_time = gpu_states_[0].local_state.current_time;
        // DEM checkpoints preserve the base step even after a terminal shortened step.
        output_state.dt = sph_mode_ ? gpu_states_[0].local_state.dt : dt_;

        auto upload_field = [&](auto particle_member, const auto& host) {
            using Value = typename std::decay_t<decltype(host)>::value_type;
            Value* target = output_state.particles.*particle_member;
            CUDA_CHECK(cudaMemcpy(target, host.data(),
                                  static_cast<size_t>(particle_count) * sizeof(Value),
                                  cudaMemcpyHostToDevice));
        };
        upload_field(&ParticleSoA::pos_x, h_pos_x);
        upload_field(&ParticleSoA::pos_y, h_pos_y);
        upload_field(&ParticleSoA::pos_z, h_pos_z);
        upload_field(&ParticleSoA::vel_x, h_vel_x);
        upload_field(&ParticleSoA::vel_y, h_vel_y);
        upload_field(&ParticleSoA::vel_z, h_vel_z);
        upload_field(&ParticleSoA::force_x, h_force_x);
        upload_field(&ParticleSoA::force_y, h_force_y);
        upload_field(&ParticleSoA::force_z, h_force_z);
        upload_field(&ParticleSoA::omega_x, h_omega_x);
        upload_field(&ParticleSoA::omega_y, h_omega_y);
        upload_field(&ParticleSoA::omega_z, h_omega_z);
        upload_field(&ParticleSoA::torque_x, h_torque_x);
        upload_field(&ParticleSoA::torque_y, h_torque_y);
        upload_field(&ParticleSoA::torque_z, h_torque_z);
        upload_field(&ParticleSoA::energy, h_energy);
        upload_field(&ParticleSoA::energy_cold, h_energy_cold);
        upload_field(&ParticleSoA::energy_thermal_peak, h_energy_thermal_peak);
        upload_field(&ParticleSoA::pressure_peak, h_pressure_peak);
        upload_field(&ParticleSoA::fracture_area, h_fracture_area);
        upload_field(&ParticleSoA::fragment_length, h_fragment_length);
        upload_field(&ParticleSoA::damping_energy, h_damping_energy);

        upload_field(&ParticleSoA::radius, h_radius);
        upload_field(&ParticleSoA::mass, h_mass);
        upload_field(&ParticleSoA::moment_of_inertia, h_moi);
        upload_field(&ParticleSoA::smoothing_length, h_smoothing_length);
        upload_field(&ParticleSoA::density, h_density);
        upload_field(&ParticleSoA::pressure, h_pressure);
        upload_field(&ParticleSoA::c_sound, h_c_sound);
        upload_field(&ParticleSoA::damage, h_damage);
        upload_field(&ParticleSoA::global_id, h_global_id);
        upload_field(&ParticleSoA::material_id, h_material_id);
        upload_field(&ParticleSoA::clump_id, h_clump_id);
        upload_field(&ParticleSoA::clump_off_x, h_clump_off_x);
        upload_field(&ParticleSoA::clump_off_y, h_clump_off_y);
        upload_field(&ParticleSoA::clump_off_z, h_clump_off_z);
        upload_field(&ParticleSoA::winding_x, h_winding_x);
        upload_field(&ParticleSoA::winding_y, h_winding_y);
        upload_field(&ParticleSoA::winding_z, h_winding_z);

        if (checkpoint) {
            upload_field(&ParticleSoA::quat_w,h_checkpoint_qw);upload_field(&ParticleSoA::quat_x,h_checkpoint_qx);
            upload_field(&ParticleSoA::quat_y,h_checkpoint_qy);upload_field(&ParticleSoA::quat_z,h_checkpoint_qz);
            upload_field(&ParticleSoA::num_contacts,h_checkpoint_contacts);
            upload_field(&ParticleSoA::num_wall_contacts,h_checkpoint_wall_counts);
            upload_field(&ParticleSoA::gravity_force_x, h_checkpoint_gx);
            upload_field(&ParticleSoA::gravity_force_y, h_checkpoint_gy);
            upload_field(&ParticleSoA::gravity_force_z, h_checkpoint_gz);
            upload_field(&ParticleSoA::rho_rate, h_checkpoint_rho_rate);
            upload_field(&ParticleSoA::dedt, h_checkpoint_dedt);
            upload_field(&ParticleSoA::S_xx, h_checkpoint_S_xx);
            upload_field(&ParticleSoA::S_xy, h_checkpoint_S_xy);
            upload_field(&ParticleSoA::S_xz, h_checkpoint_S_xz);
            upload_field(&ParticleSoA::S_yy, h_checkpoint_S_yy);
            upload_field(&ParticleSoA::S_yz, h_checkpoint_S_yz);
            upload_field(&ParticleSoA::S_zz, h_checkpoint_S_zz);
            upload_field(&ParticleSoA::crack_strain, h_checkpoint_crack_strain);
            upload_field(&ParticleSoA::distension, h_checkpoint_distension);
            upload_field(&ParticleSoA::flaws_ntot, h_checkpoint_flaws_ntot);
            upload_field(&ParticleSoA::flaws_strain_min, h_checkpoint_flaws_strain_min);
            upload_field(&ParticleSoA::flaws_strain_max, h_checkpoint_flaws_strain_max);
            upload_field(&ParticleSoA::flaws_m0, h_checkpoint_flaws_m0);
        }

        if (!write_unified_binary(filepath, output_state.particles,
                                  output_state, config_, checkpoint,
                                  checkpoint_contact_blob.empty()?nullptr:&checkpoint_contact_blob,
                                  checkpoint_contact_stride,checkpoint_total_contacts)) {
            free_particle_soa(output_state.particles);
            throw std::runtime_error("Failed writing multi-GPU snapshot: " + filepath);
        }
        if (checkpoint) {
            write_sph_stage_checkpoint(filepath, output_state, config_);
            write_respa_checkpoint_sidecar(filepath, output_state, config_);
            write_wall_history_checkpoint(filepath, output_state);
            std::vector<int> yorp_devices(num_gpus_);
            std::iota(yorp_devices.begin(), yorp_devices.end(), 0);
            std::vector<cudaStream_t> yorp_streams;
            for (const auto& rank : gpu_states_) yorp_streams.push_back(rank.compute_stream);
            write_yorp_runtime_checkpoint(filepath, output_state, config_, yorp_devices, yorp_streams, next_yorp_checks_);
            save_rcb_ownership_checkpoint(filepath, output_state);
            save_common_depth_checkpoint(filepath, output_state);
            if(!sph_mode_) {
                write_shear_checkpoint_sidecar(filepath,output_state,config_);
                save_dem_clump_sidecar(filepath,output_state,checkpoint_exact_rows,checkpoint_wall_rows,
                                      checkpoint_contact_stride,checkpoint_wall_stride);
            }
            transaction->commit();
        }
        CUDA_CHECK(cudaSetDevice(0));
        free_particle_soa(output_state.particles);
        if (checkpoint) {
            last_checkpoint_step_ = gpu_states_[0].local_state.current_step;
            checkpoint_cadence_step_ = last_checkpoint_step_;
            std::cout << "Checkpoint saved: " << filepath << std::endl;
            return;
        }

        // =====================================================================
        // Optional gravity-force sidecar (for LET single-vs-multi A/B validation)
        // =====================================================================
        // Writes per-particle gravitational force (gravity_force_x/y/z, double,
        // contact-free) to a sibling file "<base>.gravity_force" when
        // io.output.fields includes "gravity_force". This isolates the gravity
        // field from total force (which mixes in DEM contact) so the LET
        // approximation error can be compared directly.
        // =====================================================================
        if (config_.io.output_gravity_force) {
            std::vector<double> h_gforce_x(particle_count), h_gforce_y(particle_count), h_gforce_z(particle_count);

            int64_t off = 0;
            for (int g = 0; g < num_gpus_; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                int32_t n = gpu_states_[g].subdomain.num_local_particles;
                if (n > 0) {
                    const auto& particles = gpu_states_[g].local_state.particles;
                    CUDA_CHECK(cudaMemcpy(h_gforce_x.data() + off, particles.gravity_force_x, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_gforce_y.data() + off, particles.gravity_force_y, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_gforce_z.data() + off, particles.gravity_force_z, n * sizeof(double), cudaMemcpyDeviceToHost));
                    off += n;
                }
            }
            // Dead rows have no distributed gravity state by definition.
            std::fill(h_gforce_x.begin() + live_particle_count, h_gforce_x.end(), 0.0);
            std::fill(h_gforce_y.begin() + live_particle_count, h_gforce_y.end(), 0.0);
            std::fill(h_gforce_z.begin() + live_particle_count, h_gforce_z.end(), 0.0);

            // h_mass is already sorted by permanent ID. Put the gathered
            // gravity cache in that same order before converting its units.
            reorder(h_gforce_x); reorder(h_gforce_y); reorder(h_gforce_z);

            if (respa_output) {
                // The sidecar ABI is force units even though the retained
                // RESPA cache is acceleration. Include every sorted row:
                // dead rows retain zero force wherever their IDs place them.
                for (int32_t i = 0; i < particle_count; ++i) {
                    const double m = static_cast<double>(h_mass[i]);
                    h_gforce_x[i] *= m;
                    h_gforce_y[i] *= m;
                    h_gforce_z[i] *= m;
                }
            }

            std::string gpath = filepath + ".gravity_force";
            std::ofstream gfile(gpath, std::ios::binary);
            if (gfile.is_open()) {
                // The validation-only gravity sidecar retains its compact
                // legacy layout; the main particle output above is unified v3.
                const char sidecar_magic[8] = "DEMOUT\0";
                const uint32_t sidecar_version = 1;
                gfile.write(sidecar_magic, 8);
                gfile.write(reinterpret_cast<const char*>(&sidecar_version), sizeof(sidecar_version));
                gfile.write(reinterpret_cast<const char*>(&particle_count), sizeof(particle_count));
                gfile.write(reinterpret_cast<const char*>(&gpu_states_[0].local_state.current_step), sizeof(int64_t));
                gfile.write(reinterpret_cast<const char*>(&gpu_states_[0].local_state.current_time), sizeof(double));
                gfile.write(reinterpret_cast<const char*>(&gpu_states_[0].local_state.dt), sizeof(double));
                gfile.write(reinterpret_cast<char*>(h_gforce_x.data()), particle_count * sizeof(double));
                gfile.write(reinterpret_cast<char*>(h_gforce_y.data()), particle_count * sizeof(double));
                gfile.write(reinterpret_cast<char*>(h_gforce_z.data()), particle_count * sizeof(double));
                gfile.close();
            } else {
                std::cerr << "WARNING: Cannot create gravity-force sidecar: " << gpath << std::endl;
            }
        }
        if (distributed_clumps_.enabled())
            distributed_clumps_.save(config_.io.output_directory,gpu_states_[0].local_state.current_step,
                                     gpu_states_[0].local_state.current_time);
        last_output_step_ = gpu_states_[0].local_state.current_step;
    }

    void shutdown() {
        if (!initialized_) return;

        std::cout << "\nShutting down multi-GPU simulation..." << std::endl;

        clump_depth_census_.shutdown();
        distributed_clumps_.shutdown();
        distributed_yorp_.shutdown();
        AccelerationManager::shutdown();

        for (int g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            if (config_.shear_box.enabled) ShearBoxManager::shutdown();
            communicators_[g].shutdown();
            let_comms_[g].shutdown();
            NeighborSearch::free_contact_buffer(contacts_[g]);
            NeighborSearch::shutdown(gpu_states_[g].local_state);
            if (g < static_cast<int>(d_sph_hydro_owned_offsets_.size())) {
                cudaFree(d_sph_hydro_owned_offsets_[g]);
                d_sph_hydro_owned_offsets_[g] = nullptr;
            }
            if (sph_mode_) {
                shutdown_sph_timestep();
                shutdown_sph_strength_workspace();
                shutdown_sph_porosity();
            }
            if (g < static_cast<int>(sph_damage_workspaces_.size())) {
                if (sph_damage_workspaces_[g].qdot) {
                    cudaFree(sph_damage_workspaces_[g].qdot);
                    sph_damage_workspaces_[g].qdot = nullptr;
                }
                if (sph_damage_workspaces_[g].qcap) {
                    cudaFree(sph_damage_workspaces_[g].qcap);
                    sph_damage_workspaces_[g].qcap = nullptr;
                }
            }
            free_contact_history(gpu_states_[g].local_state.contact_history);
            free_wall_contact_history(gpu_states_[g].local_state.wall_contact_history);
            free_particle_soa(gpu_states_[g].local_state.particles);

            // Phase 4: Free pinned memory + events
            if (g < static_cast<int>(h_pinned_num_contacts_.size()) && h_pinned_num_contacts_[g]) {
                cudaFreeHost(h_pinned_num_contacts_[g]);
                h_pinned_num_contacts_[g] = nullptr;
            }
            if (g < static_cast<int>(contact_count_events_.size())) {
                cudaEventDestroy(contact_count_events_[g]);
            }
        }
        h_pinned_num_contacts_.clear();
        for (auto* pinned : h_pinned_sph_owned_degrees_) {
            if (pinned) cudaFreeHost(pinned);
        }
        h_pinned_sph_owned_degrees_.clear();
        d_sph_hydro_owned_offsets_.clear();
        sph_hydro_offset_capacities_.clear();
        sph_hydro_offsets_valid_.clear();
        contact_count_events_.clear();
        contact_count_ready_.clear();

        balancer_.shutdown();
        for (int32_t g = 0; g < num_gpus_; ++g) {
            cudaSetDevice(g);
            for (int32_t slot = 0; slot < kTimingEventSlots; ++slot) {
                size_t index = static_cast<size_t>(g) * kTimingEventSlots + slot;
                cudaEventDestroy(step_start_events_[index]);
                cudaEventDestroy(step_stop_events_[index]);
                for (uint8_t raw = 0;
                     raw < static_cast<uint8_t>(CalibrationSegment::Count); ++raw) {
                    const auto segment = static_cast<CalibrationSegment>(raw);
                    const size_t calibration_index =
                        calibration_event_index(g, slot, segment);
                    cudaEventDestroy(calibration_start_events_[calibration_index]);
                    cudaEventDestroy(calibration_stop_events_[calibration_index]);
                }
            }
            cudaEventDestroy(migration_transfer_events_[g]);
            cudaEventDestroy(halo_transfer_events_[g]);
        }
        step_start_events_.clear();
        step_stop_events_.clear();
        migration_transfer_events_.clear();
        halo_transfer_events_.clear();
        calibration_start_events_.clear();
        calibration_stop_events_.clear();
        latest_contact_counts_.clear();
        sph_damage_workspaces_.clear();

        morton_root_.shutdown();
        morton_balance_.shutdown();
        for(int gpu=0;gpu<num_gpus_;++gpu){CUDA_CHECK(cudaSetDevice(gpu));morton_work_collectors_[gpu].shutdown();}
        delete[] let_comms_;
        let_comms_ = nullptr;

        phase_coordinator_.shutdown();
        DomainManager::shutdown(gpu_states_);

        initialized_ = false;
        std::cout << "Shutdown complete." << std::endl;
    }
};

} // namespace cudadem

// =============================================================================
// Main Entry Point for Multi-GPU
// =============================================================================

int main(int argc, char* argv[]) {
    cudadem::CooperativeStopSignals cooperative_stop_signals;
    // See the single-GPU entry point: every module must be loaded before any
    // per-GPU persistent scheduler becomes resident.
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);

    std::string config_path = "config.yaml";

    if (argc > 1) {
        config_path = argv[1];
    }

    std::ifstream config_check(config_path);
    if (!config_check.good()) {
        std::cerr << "ERROR: Configuration file not found: " << config_path << std::endl;
        std::cerr << "Usage: " << argv[0] << " [config.yaml]" << std::endl;
        return 1;
    }
    config_check.close();

    // Check for multiple GPUs
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "Detected " << device_count << " CUDA device(s)" << std::endl;

    if (device_count < 1) {
        std::cerr << "WARNING: Multi-GPU mode requires at least 1 GPU." << std::endl;
        std::cerr << "Falling back to single-GPU mode." << std::endl;
        // Could redirect to single-GPU main here
        return 1;
    }

    // Set OpenMP thread count
    omp_set_num_threads(device_count);

    cudadem::MultiGPUSimulationRunner sim;

    try {
        if (!sim.initialize(config_path)) {
            std::cerr << "ERROR: Failed to initialize multi-GPU simulation" << std::endl;
            return 1;
        }
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Simulation failed - " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

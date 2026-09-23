#ifndef CUDADEM_LOAD_OR_RESUME_H
#define CUDADEM_LOAD_OR_RESUME_H

#include "command_line_interface.h"
#include "system_initialize.h"
#include "binary_format.h"
#include "binary_reader.h"
#include "../acceleration/external_bodies.h"
#include "../neighbor/neighbor_api.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <vector>
#include <iostream>

namespace cudadem {

// Sidecars must be restored from the exact checkpoint selected by the loader;
// this also makes RESUME_SPECIFIC independent of newer files in the directory.
inline std::string& loaded_checkpoint_path_storage() { static std::string path; return path; }
inline const std::string& loaded_checkpoint_path() { return loaded_checkpoint_path_storage(); }

// =============================================================================
// Forward Declarations for Initialization Kernels (defined in sphere_dynamic.cu)
// =============================================================================
extern void launch_init_particle_mass_kernel(float* mass, float* moi, float* support_extent, const float* radius,
                                             const int32_t* material_id, const MaterialRecord* materials,
                                             const DemInteractionProperties* interactions,
                                             uint32_t material_count, int32_t num_particles, int block_size);
extern void launch_init_cohesive_strength_kernel(float* strength, float base_strength,
                                                  int32_t num_particles, int block_size);
extern void launch_init_contact_offsets_kernel(int32_t* offsets, int32_t max_contacts,
                                                int32_t num_particles, int block_size);

// =============================================================================
// Config Hash for Change Detection
// =============================================================================

// Simple hash function for config validation
// Detects if critical parameters changed between runs
inline uint64_t compute_config_hash(const SimConfig& config) {
    uint64_t hash = 0;

    // Hash critical parameters that would invalidate a checkpoint
    auto hash_combine = [&hash](uint64_t value) {
        hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    };

    // Simulation type
    hash_combine(static_cast<uint64_t>(config.sim_type));
    hash_combine(static_cast<uint64_t>(config.coord_system));
    hash_combine(static_cast<uint64_t>(config.physics));
    hash_combine(static_cast<uint64_t>(config.dem_damping_model));

    // Wall count (structural change)
    hash_combine(config.num_walls);
    hash_combine(config.material_database.fingerprint);

    if (config.shear_box.enabled) {
        auto hash_double = [&hash_combine](double value) {
            uint64_t bits = 0;
            if (value == 0.0) value = 0.0;  // normalize signed zero
            std::memcpy(&bits, &value, sizeof(bits));
            hash_combine(bits);
        };
        hash_combine(UINT64_C(0x5348454152424f58)); // "SHEARBOX"
        hash_double(config.shear_box.omega);
        hash_double(config.shear_box.shear_parameter);
        hash_double(config.shear_box.vertical_frequency);
        hash_double(config.shear_box.time_origin);
        // The Hill terms depend on absolute x/z, not only on box lengths.
        // Hash every bound so moving an otherwise equal-sized box cannot
        // reinterpret canonical checkpoint coordinates under a new tidal
        // origin (and so the fixed gravity envelope remains identical).
        hash_double(static_cast<double>(config.domain.box_min.x));
        hash_double(static_cast<double>(config.domain.box_min.y));
        hash_double(static_cast<double>(config.domain.box_min.z));
        hash_double(static_cast<double>(config.domain.box_max.x));
        hash_double(static_cast<double>(config.domain.box_max.y));
        hash_double(static_cast<double>(config.domain.box_max.z));
        hash_combine(static_cast<uint64_t>(config.shear_box.gravity_layers_x));
        hash_combine(static_cast<uint64_t>(config.shear_box.gravity_layers_y));
        hash_double(config.shear_box.gravity_cutoff_radius);
    }

    return hash;
}

// =============================================================================
// Data Type Validation Functions
// =============================================================================

struct ValidationResult {
    bool success;
    std::string error_message;

    ValidationResult() : success(true) {}
    ValidationResult(bool s, const std::string& msg = "") : success(s), error_message(msg) {}
};

// Validate particle binary file data types against config
inline ValidationResult validate_particle_file(const std::string& filepath,
                                                const SimConfig& config) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return ValidationResult(false, "Cannot open file: " + filepath);
    }

    UnifiedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.close();

    // Check magic number
    if (memcmp(header.magic, MAGIC, 8) != 0) {
        return ValidationResult(false,
            "Invalid file format: magic number mismatch. "
            "Expected GRANIT unified v3 input.");
    }

    // Check version
    if (header.version != FORMAT_VERSION || header.header_size != HEADER_SIZE) {
        return ValidationResult(false,
            "Unsupported unified particle version/header.");
    }
    const uint8_t expected_physics = static_cast<uint8_t>(config.physics == SimulationPhysics::SPH ? BinaryPhysics::SPH : BinaryPhysics::DEM);
    if (header.physics != expected_physics || header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT)) {
        return ValidationResult(false, "Unified particle file physics or record kind does not match fresh-start input.");
    }

    // Check particle count is reasonable
    if (header.num_particles == 0) {
        return ValidationResult(false, "File contains zero particles");
    }

    if (header.num_particles > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ValidationResult(false,
            "Particle count exceeds sanity limit: " + std::to_string(header.num_particles));
    }

    return ValidationResult(true);
}

// Validate checkpoint file
inline ValidationResult validate_checkpoint_file(const std::string& filepath,
                                                  const SimConfig& config) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return ValidationResult(false, "Cannot open checkpoint: " + filepath);
    }

    UnifiedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.close();

    // Check magic
    if (memcmp(header.magic, MAGIC, 8) != 0) {
        return ValidationResult(false, "Invalid checkpoint file format");
    }

    // Check version
    if (header.version != FORMAT_VERSION || header.header_size != HEADER_SIZE ||
        header.record_kind != static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT)) {
        return ValidationResult(false,
            "Unsupported unified checkpoint version/header/kind.");
    }

    // Validate config hash (warn but don't fail)
    if (header.material_database_fingerprint != config.material_database.fingerprint) {
        return ValidationResult(false, "Checkpoint material database fingerprint mismatch.");
    }
    if (header.clump_metadata_fingerprint != config.clump.clump_metadata_fingerprint) {
        return ValidationResult(false, "Checkpoint clump metadata fingerprint mismatch.");
    }

    return ValidationResult(true);
}

// =============================================================================
// Checkpoint Generation Path
// =============================================================================

inline std::string generate_checkpoint_path(const SimConfig& config, int64_t step) {
    std::ostringstream oss;
    oss << config.io.checkpoint_path << "/checkpoint_step_"
        << std::setfill('0') << std::setw(10) << step << ".bin";
    return oss.str();
}

struct RespaSynchronizedCheckpointHeader {
    char magic[8];
    uint32_t version;
    int64_t step;
    double time;
};

// SPH stage physics is part of a checkpoint's continuation contract.  Store
// explicit values rather than struct padding, and bind them to its closed time.
inline bool sph_stage_checkpoint_requested(const SimConfig& config) {
    return config.physics == SimulationPhysics::SPH && (config.sph.stage_count > 0 ||
        config.sph.track_history || config.sph.fixed_part_count > 0 || config.sph.symmetry_enabled);
}

inline std::string sph_stage_checkpoint_signature(const SimConfig& config, int64_t step, double time) {
    std::ostringstream out;
    out << std::setprecision(17) << "GRANIT_SPH_STAGE_V1 " << step << ' ' << time;
    out << ' ' << config.sph.stage_count << ' ' << config.sph.track_history;
    out << ' ' << config.sph.relaxation_disable_strength << ' ' << config.sph.relaxation_disable_damage;
    for (int i=0; i<config.sph.stage_count; ++i) {
        const auto& s=config.sph.stages[i];
        out << ' ' << static_cast<int>(s.mode) << ' ' << s.end_time << ' ' << s.damping_rate
            << ' ' << s.evolve_energy << ' ' << s.evolve_damage << ' ' << s.evolve_porosity
            << ' ' << s.use_strength << ' ' << s.elastic_eos;
    }
    out << ' ' << config.sph.fixed_part_count;
    for (int i=0; i<config.sph.fixed_part_count; ++i) out << ' ' << config.sph.fixed_parts[i];
    out << ' ' << config.sph.symmetry_enabled;
    if (config.sph.symmetry_enabled)
        out << ' ' << config.sph.symmetry_normal[0] << ' ' << config.sph.symmetry_normal[1]
            << ' ' << config.sph.symmetry_normal[2] << ' ' << config.sph.symmetry_offset;
    return out.str();
}

inline bool valid_sph_stage_checkpoint(const std::string& path, const SimConfig& config,
    int64_t step, double time) {
    const std::string sidecar=path+".sph_stage";
    if (!sph_stage_checkpoint_requested(config) && !std::filesystem::exists(sidecar)) return true;
    if (!sph_stage_checkpoint_requested(config)) return false;
    std::ifstream in(sidecar); std::string signature;
    return static_cast<bool>(std::getline(in, signature)) &&
        signature == sph_stage_checkpoint_signature(config, step, time);
}

inline void write_sph_stage_checkpoint(const std::string& path, const SimulationState& state,
    const SimConfig& config) {
    if (!sph_stage_checkpoint_requested(config)) return;
    const std::string sidecar=path+".sph_stage";
    const std::string temporary=sidecar+".tmp";
    std::ofstream out(temporary);
    out << sph_stage_checkpoint_signature(config, state.current_step, state.current_time) << '\n';
    out.close();
    if (!out) throw std::runtime_error("Cannot write SPH stage checkpoint sidecar");
    std::filesystem::rename(temporary, sidecar);
}

inline bool respa_checkpoint_requested(const SimConfig& config) {
    return config.acceleration.enable_self_gravity &&
        (config.acceleration.gravity_update_rate > 1 ||
         config.acceleration.gravity_macro_dt > 0.0);
}

inline std::string respa_checkpoint_sidecar_path(const std::string& checkpoint_path) {
    return checkpoint_path + ".respa_sync";
}

inline void write_respa_checkpoint_sidecar(
    const std::string& checkpoint_path, const SimulationState& state,
    const SimConfig& config) {
    if (!respa_checkpoint_requested(config)) return;
    const RespaSynchronizedCheckpointHeader header{
        {'R','S','P','A','S','Y','N','C'}, 1, state.current_step, state.current_time};
    const std::string path = respa_checkpoint_sidecar_path(checkpoint_path);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot write RESPA synchronization sidecar: " + path);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) throw std::runtime_error("Failed writing RESPA synchronization sidecar: " + path);
}

inline bool valid_respa_checkpoint_sidecar(
    const std::string& checkpoint_path, int64_t expected_step,
    double expected_time) {
    RespaSynchronizedCheckpointHeader header{};
    const std::string path = respa_checkpoint_sidecar_path(checkpoint_path);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    const char magic[8] = {'R','S','P','A','S','Y','N','C'};
    return file && std::memcmp(header.magic, magic, sizeof(magic)) == 0 &&
        header.version == 1 && header.step == expected_step &&
        header.time == expected_time;
}

// A shear checkpoint is resumable only at a closed kick--exact-Hill--kick
// boundary and only under the identical physical image convention.
struct ShearSynchronizedCheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t integrator_version;
    int64_t step;
    double time;
    uint64_t config_hash;
    uint8_t closed_kick;
    uint8_t reserved[7];
};

inline std::string shear_checkpoint_sidecar_path(const std::string& checkpoint_path) {
    return checkpoint_path + ".shear_box";
}

inline bool adaptive_fmm_depth_checkpoint_requested(const SimConfig& config) {
    return config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM &&
        config.acceleration.unified_fmm.adaptive_depth.enabled;
}

inline std::string adaptive_fmm_depth_checkpoint_sidecar_path(
    const std::string& checkpoint_path) {
    return checkpoint_path + ".adaptive_fmm_depth";
}

// Kept separate from the payload so discovery can reject an obsolete v1
// sidecar before it is selected as a resumable checkpoint.
struct AdaptiveFMMDepthCheckpointSidecarHeader {
    char magic[8];
    uint32_t version;
    uint32_t payload_size;
    int64_t step;
    double time;
    uint64_t config_hash;
};

// The controller state is written by SimulationRunner after the particle
// checkpoint and the existing integrator sidecars.  Treat the pair as
// publish-on-sidecar: an interrupted write must never make the preceding
// particle binary resumable under adaptive depth.
inline bool has_adaptive_fmm_depth_checkpoint_sidecar(
    const std::string& checkpoint_path) {
    const std::filesystem::path sidecar(
        adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path));
    std::ifstream file(sidecar, std::ios::binary);
    AdaptiveFMMDepthCheckpointSidecarHeader header{};
    const char expected_magic[8] = {'A','F','M','M','D','P','T','H'};
    return file.read(reinterpret_cast<char*>(&header), sizeof(header)) &&
        std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) == 0 &&
        header.version == 2 && header.payload_size > 0;
}

inline uint32_t adaptive_fmm_depth_checkpoint_sidecar_version(
    const std::string& checkpoint_path) {
    std::ifstream file(adaptive_fmm_depth_checkpoint_sidecar_path(checkpoint_path),
                       std::ios::binary);
    AdaptiveFMMDepthCheckpointSidecarHeader header{};
    const char expected_magic[8] = {'A','F','M','M','D','P','T','H'};
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0)
        return 0;
    return header.version;
}

inline void write_shear_checkpoint_sidecar(
    const std::string& checkpoint_path, const SimulationState& state,
    const SimConfig& config) {
    if (!config.shear_box.enabled) return;
    const ShearSynchronizedCheckpointHeader header{
        {'S','H','E','A','R','S','Y','N'}, 1, 1, state.current_step,
        state.current_time, compute_config_hash(config), 1, {0,0,0,0,0,0,0}};
    const std::string path = shear_checkpoint_sidecar_path(checkpoint_path);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot write shear checkpoint sidecar: " + path);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) throw std::runtime_error("Failed writing shear checkpoint sidecar: " + path);
}

inline bool valid_shear_checkpoint_sidecar(
    const std::string& checkpoint_path, int64_t expected_step,
    double expected_time, const SimConfig& config) {
    if (!config.shear_box.enabled) return true;
    ShearSynchronizedCheckpointHeader header{};
    const std::string path = shear_checkpoint_sidecar_path(checkpoint_path);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    const char magic[8] = {'S','H','E','A','R','S','Y','N'};
    return file && std::memcmp(header.magic, magic, sizeof(magic)) == 0 &&
        header.version == 1 && header.integrator_version == 1 &&
        header.step == expected_step && header.time == expected_time &&
        header.config_hash == compute_config_hash(config) &&
        header.closed_kick == 1;
}

inline bool shear_checkpoint_mode_matches(
    const std::string& checkpoint_path, int64_t expected_step,
    double expected_time, const SimConfig& config) {
    std::error_code ec;
    const bool has_sidecar = std::filesystem::is_regular_file(
        shear_checkpoint_sidecar_path(checkpoint_path), ec) && !ec;
    if (!config.shear_box.enabled) return !has_sidecar;
    return has_sidecar && valid_shear_checkpoint_sidecar(
        checkpoint_path, expected_step, expected_time, config);
}

inline std::string get_latest_checkpoint_path(const SimConfig& config) {
    std::string checkpoint_dir = config.io.checkpoint_path;

    if (!std::filesystem::exists(checkpoint_dir)) {
        return "";
    }

    std::string latest_path;
    int64_t latest_step = -1;
    bool found_unsynchronized_respa_checkpoint = false;
    bool found_invalid_sph_checkpoint = false;
    bool found_invalid_shear_checkpoint = false;
    bool found_incomplete_external_bodies_checkpoint = false;
    bool found_incomplete_adaptive_fmm_depth_checkpoint = false;
    int64_t newest_incomplete_adaptive_step = -1;
    int64_t newest_legacy_adaptive_step = -1;

    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            constexpr const char* prefix = "checkpoint_step_";
            constexpr const char* suffix = ".bin";
            const size_t prefix_size = std::strlen(prefix);
            const size_t suffix_size = std::strlen(suffix);
            if (filename.size() > prefix_size + suffix_size &&
                filename.compare(0, prefix_size, prefix) == 0 &&
                filename.compare(filename.size() - suffix_size, suffix_size, suffix) == 0) {
                // Adaptive checkpoints are published as a required pair.  If
                // a process stopped between the binary and sidecar writes,
                // skip that orphan and resume the newest complete pair.
                if (config.acceleration.yorp.adaptive) {
                    const std::filesystem::path sidecar(entry.path().string() + ".adaptive_yorp");
                    std::error_code ec;
                    if (!std::filesystem::is_regular_file(sidecar, ec) || ec ||
                        std::filesystem::file_size(sidecar, ec) == 0 || ec) {
                        continue;
                    }
                }
                if (adaptive_fmm_depth_checkpoint_requested(config) &&
                    !has_adaptive_fmm_depth_checkpoint_sidecar(entry.path().string())) {
                    found_incomplete_adaptive_fmm_depth_checkpoint = true;
                    try {
                        const int64_t step = static_cast<int64_t>(std::stoll(filename.substr(
                            prefix_size, filename.size() - prefix_size - suffix_size)));
                        newest_incomplete_adaptive_step = std::max(
                            newest_incomplete_adaptive_step, step);
                        if (adaptive_fmm_depth_checkpoint_sidecar_version(
                                entry.path().string()) == 1) {
                            newest_legacy_adaptive_step = std::max(
                                newest_legacy_adaptive_step, step);
                        }
                    } catch (...) {}
                    continue;
                }
                if (respa_checkpoint_requested(config)) {
                    std::ifstream checkpoint(entry.path(), std::ios::binary);
                    UnifiedHeader header{};
                    checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
                    if (!checkpoint || !valid_respa_checkpoint_sidecar(
                            entry.path().string(), header.current_step,
                            header.current_time)) {
                        found_unsynchronized_respa_checkpoint = true;
                        continue;
                    }
                }
                if (config.shear_box.enabled || std::filesystem::exists(
                        shear_checkpoint_sidecar_path(entry.path().string()))) {
                    std::ifstream checkpoint(entry.path(), std::ios::binary);
                    UnifiedHeader header{};
                    checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
                    if (!checkpoint || !shear_checkpoint_mode_matches(
                            entry.path().string(), header.current_step,
                            header.current_time, config)) {
                        found_invalid_shear_checkpoint = true;
                        continue;
                    }
                }
                if (config.acceleration.external_bodies.enabled) {
                    std::ifstream checkpoint(entry.path(), std::ios::binary); UnifiedHeader header{};
                    checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
                    if (!checkpoint || !ExternalBodyManager::checkpoint_sidecar_matches(entry.path().string(), header.current_step, header.current_time, config)) { found_incomplete_external_bodies_checkpoint=true; continue; }
                }
                {
                    std::ifstream checkpoint(entry.path(), std::ios::binary); UnifiedHeader header{};
                    checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
                    if (!checkpoint || !valid_sph_stage_checkpoint(entry.path().string(), config,
                            header.current_step, header.current_time)) {
                        found_invalid_sph_checkpoint = true;
                        continue;
                    }
                }
                // Extract step number
                const size_t end = filename.size() - suffix_size;
                std::string step_str = filename.substr(prefix_size, end - prefix_size);

                try {
                    int64_t step = std::stoll(step_str);
                    if (step > latest_step) {
                        latest_step = step;
                        latest_path = entry.path().string();
                    }
                } catch (...) {
                    continue;
                }
            }
        }
    }

    if (latest_path.empty() && found_invalid_sph_checkpoint)
        throw std::runtime_error("No matching SPH stage checkpoint found; choose force_fresh only for a new run");
    if (latest_path.empty() && found_unsynchronized_respa_checkpoint) {
        throw std::runtime_error(
            "Only legacy or incomplete RESPA checkpoints were found; they were saved at "
            "an unknown kick phase and cannot be resumed safely");
    }
    if (latest_path.empty() && found_invalid_shear_checkpoint) {
        throw std::runtime_error(
            "Only incomplete or configuration-mismatched shear checkpoints were found; "
            "a versioned closed-kick shear sidecar is required");
    }
    if (latest_path.empty() && found_incomplete_external_bodies_checkpoint) throw std::runtime_error("Only incomplete external-body checkpoints were found; matching lifecycle ledger sidecar is required");
    if (found_incomplete_adaptive_fmm_depth_checkpoint &&
        newest_incomplete_adaptive_step > latest_step) {
        if (newest_legacy_adaptive_step == newest_incomplete_adaptive_step) {
            throw std::runtime_error(
                "The newest adaptive FMM depth checkpoint uses unsupported version 1; "
                "no converter is available");
        }
        throw std::runtime_error(
            "The newest adaptive FMM depth checkpoint is incomplete; refusing to "
            "silently roll back to an older controller state");
    }
    if (latest_path.empty() && found_incomplete_adaptive_fmm_depth_checkpoint) {
        throw std::runtime_error(
            "Only incomplete adaptive FMM depth checkpoints were found; a versioned "
            "controller sidecar matching the particle checkpoint is required");
    }

    return latest_path;
}

// =============================================================================
// Save Checkpoint
// =============================================================================

inline void save_checkpoint(const SimulationState& state,
                           const SimConfig& config) {
    // Create checkpoint directory if needed
    std::string checkpoint_dir = config.io.checkpoint_path;
    if (!std::filesystem::exists(checkpoint_dir)) {
        std::filesystem::create_directories(checkpoint_dir);
    }

    std::string filepath = generate_checkpoint_path(config, state.current_step);

    int32_t num_particles = state.particles.num_particles;

    // Count total contacts from GPU
    std::vector<int32_t> h_num_contacts(num_particles);
    CUDA_CHECK(cudaMemcpy(h_num_contacts.data(), state.particles.num_contacts,
                          num_particles * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int32_t total_contacts = 0;
    for (int32_t i = 0; i < num_particles; ++i) {
        total_contacts += h_num_contacts[i];
    }

    int32_t max_contacts = state.contact_history.max_contacts_per_particle;

    // Read contact history from GPU if any
    std::vector<ContactHistory> h_contact_history;
    if (total_contacts > 0) {
        int32_t history_size = num_particles * max_contacts;
        h_contact_history.resize(history_size);
        CUDA_CHECK(cudaMemcpy(h_contact_history.data(), state.contact_history.data,
                              history_size * sizeof(ContactHistory), cudaMemcpyDeviceToHost));
    }

    // Write unified binary checkpoint
    if (!write_unified_binary(filepath, state.particles, state, config, true,
                              total_contacts > 0 ? &h_contact_history : nullptr,
                              max_contacts, total_contacts)) {
        throw std::runtime_error("Failed to write unified binary checkpoint: " + filepath);
    }
    write_sph_stage_checkpoint(filepath, state, config);
    write_respa_checkpoint_sidecar(filepath, state, config);
    write_shear_checkpoint_sidecar(filepath, state, config);

    std::cout << "Checkpoint saved: " << filepath << std::endl;
    std::cout << "  Step: " << state.current_step
              << ", Time: " << state.current_time << " s"
              << ", Particles: " << num_particles << std::endl;
}

// =============================================================================
// Load Checkpoint / Resume Simulation
// =============================================================================

inline SimulationState load_checkpoint(const std::string& filepath,
                                       const SimConfig& config) {
    SimulationState state;

    if (adaptive_fmm_depth_checkpoint_requested(config) &&
        !has_adaptive_fmm_depth_checkpoint_sidecar(filepath)) {
        throw std::runtime_error(
            "Adaptive FMM depth resume requires checkpoint sidecar: " +
            adaptive_fmm_depth_checkpoint_sidecar_path(filepath));
    }

    if (respa_checkpoint_requested(config)) {
        std::ifstream checkpoint(filepath, std::ios::binary);
        UnifiedHeader header{};
        checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!checkpoint || !valid_respa_checkpoint_sidecar(
                filepath, header.current_step, header.current_time)) {
            throw std::runtime_error(
                "Legacy or incomplete RESPA checkpoint is not synchronized; restart from "
                "an ordinary particle output or a checkpoint written by the corrected integrator");
        }
    }
    if (config.shear_box.enabled ||
        std::filesystem::exists(shear_checkpoint_sidecar_path(filepath))) {
        std::ifstream checkpoint(filepath, std::ios::binary);
        UnifiedHeader header{};
        checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!checkpoint || !shear_checkpoint_mode_matches(
                filepath, header.current_step, header.current_time, config)) {
            throw std::runtime_error(
                "Checkpoint shear mode does not match, its closed-kick sidecar is "
                "missing, or its physical shear/image configuration changed");
        }
    }
    if (config.acceleration.external_bodies.enabled) {
        std::ifstream checkpoint(filepath, std::ios::binary); UnifiedHeader header{};
        checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!checkpoint || !ExternalBodyManager::checkpoint_sidecar_matches(filepath, header.current_step, header.current_time, config))
            throw std::runtime_error("External-body resume requires a complete matching lifecycle ledger sidecar");
    }

    {
        std::ifstream checkpoint(filepath, std::ios::binary); UnifiedHeader header{};
        checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!checkpoint || !valid_sph_stage_checkpoint(filepath, config, header.current_step, header.current_time))
            throw std::runtime_error("SPH checkpoint stage/history/boundary settings changed or .sph_stage sidecar is missing");
    }

    // Read unified binary (handles header, descriptors, Tier 1+2, contact blob)
    if (!read_unified_binary(filepath, state.particles, state, config, 0)) {
        throw std::runtime_error("Failed to read unified binary checkpoint: " + filepath);
    }

    int32_t num_particles = state.particles.num_particles;

    // If no contact blob was in the file, allocate contact history buffers
    if (state.contact_history.data == nullptr) {
        allocate_contact_history(state.contact_history, num_particles, 32);
    }

    std::cout << "Loading checkpoint: " << filepath << std::endl;
    loaded_checkpoint_path_storage() = filepath;
    std::cout << "  Step: " << state.current_step
              << ", Time: " << state.current_time << " s"
              << ", Particles: " << num_particles << std::endl;

    // Re-upload immutable material tables and derive quantities from the
    // checkpoint's preserved material identifiers.
    const MaterialDatabase material_database = load_material_database(
        config.material_database.path, config.dem_damping_model);
    state.material_database = upload_material_database(material_database, config.gpu.device_id);
    int block_size = config.gpu.block_size;

    if (config.physics == SimulationPhysics::DEM) {
        launch_init_particle_mass_kernel(
            state.particles.mass,
            state.particles.moment_of_inertia,
            state.particles.support_extent,
            state.particles.radius,
            state.particles.material_id,
            state.material_database.materials,
            state.material_database.dem_interactions,
            state.material_database.material_count,
            num_particles,
            block_size
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Cohesion is an immutable pair property.  The legacy per-particle array
    // remains allocated for checkpoint layout stability but is not a source
    // of physical parameters.
    CUDA_CHECK(cudaMemset(state.particles.cohesive_strength, 0,
                          state.particles.capacity * sizeof(float)));

    launch_init_contact_offsets_kernel(
        state.particles.contact_history_offset,
        state.contact_history.max_contacts_per_particle,
        num_particles,
        block_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Initialize walls
    initialize_walls(state.walls, config);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Checkpoint loaded successfully" << std::endl;

    return state;
}

// =============================================================================
// Load or Resume Orchestrator
// =============================================================================

enum class LoadMode {
    FRESH_START,     // Start from initial particle file
    RESUME_LATEST,   // Resume from latest checkpoint
    RESUME_SPECIFIC  // Resume from specific checkpoint file
};

struct LoadOptions {
    LoadMode mode;
    std::string checkpoint_path;  // For RESUME_SPECIFIC mode
    bool force_fresh;             // Ignore existing checkpoints

    LoadOptions()
        : mode(LoadMode::FRESH_START), force_fresh(false) {}
};

// Main function to initialize or resume simulation
inline SimulationState load_or_resume(const SimConfig& config,
                                       const LoadOptions& options = LoadOptions()) {
    loaded_checkpoint_path_storage().clear();
    std::cout << "\n========== Load / Resume ==========" << std::endl;

    // Set GPU device
    CUDA_CHECK(cudaSetDevice(config.gpu.device_id));

    // Copy config to GPU
    copy_config_to_device(config);

    // Determine load mode
    LoadMode effective_mode = options.mode;
    const bool force_fresh = options.force_fresh || config.io.checkpoint_force_fresh;

    if (!force_fresh && effective_mode == LoadMode::FRESH_START) {
        // Check if there's an existing checkpoint
        std::string latest = get_latest_checkpoint_path(config);
        if (!latest.empty()) {
            if (adaptive_fmm_depth_checkpoint_requested(config) &&
                config.acceleration.unified_fmm.adaptive_depth.tree_depth_was_omitted) {
                throw std::runtime_error(
                    "unified_fmm.tree_depth was omitted (fresh-start auto only); "
                    "resuming requires an explicit tree_depth: auto or integer, or "
                    "io.checkpoint.force_fresh: true");
            }
            std::cout << "Found existing checkpoint: " << latest << std::endl;
            std::cout << "Use force_fresh=true to ignore checkpoints" << std::endl;
            effective_mode = LoadMode::RESUME_LATEST;
        }
    }

    SimulationState state;

    switch (effective_mode) {
        case LoadMode::FRESH_START: {
            std::cout << "Starting fresh simulation" << std::endl;

            // Validate input file
            std::string particle_path = config.io.input_particle_path;
            if (particle_path[0] == '\0') {
                throw std::runtime_error("No particle input file specified");
            }

            if (config.io.validate_input_types) {
                ValidationResult validation = validate_particle_file(particle_path, config);
                if (!validation.success) {
                    throw std::runtime_error("Particle file validation failed: " +
                                           validation.error_message);
                }
                std::cout << "Particle file validation: PASSED" << std::endl;
            }

            // Use standard initialization
            state = initialize_system(config);
            break;
        }

        case LoadMode::RESUME_LATEST: {
            std::string latest = get_latest_checkpoint_path(config);
            if (latest.empty()) {
                throw std::runtime_error("No checkpoint found to resume from");
            }
            std::cout << "Resuming from latest checkpoint: " << latest << std::endl;
            if (!adaptive_fmm_depth_checkpoint_requested(config) &&
                adaptive_fmm_depth_checkpoint_sidecar_version(latest) != 0) {
                std::cerr << "WARNING: resuming the base particle checkpoint at fixed "
                          << "BFFMM depth; adaptive controller state is ignored\n";
            }
            state = load_checkpoint(latest, config);
            break;
        }

        case LoadMode::RESUME_SPECIFIC: {
            if (options.checkpoint_path.empty()) {
                throw std::runtime_error("No checkpoint path specified for RESUME_SPECIFIC mode");
            }
            std::cout << "Resuming from specific checkpoint: " << options.checkpoint_path << std::endl;
            if (!adaptive_fmm_depth_checkpoint_requested(config) &&
                adaptive_fmm_depth_checkpoint_sidecar_version(options.checkpoint_path) != 0) {
                std::cerr << "WARNING: resuming the base particle checkpoint at fixed "
                          << "BFFMM depth; adaptive controller state is ignored\n";
            }
            state = load_checkpoint(options.checkpoint_path, config);
            break;
        }
    }

    std::cout << "========== Load Complete ==========" << std::endl;
    return state;
}

// =============================================================================
// Cleanup Old Checkpoints
// =============================================================================

// Keep only the N most recent checkpoints
inline void cleanup_old_checkpoints(const SimConfig& config, int keep_count = 5) {
    std::string checkpoint_dir = config.io.checkpoint_path;

    if (!std::filesystem::exists(checkpoint_dir)) {
        return;
    }

    // Collect all checkpoint files with their step numbers
    std::vector<std::pair<int64_t, std::string>> checkpoints;

    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            constexpr const char* prefix = "checkpoint_step_";
            constexpr const char* suffix = ".bin";
            const size_t start = std::strlen(prefix);
            const size_t suffix_size = std::strlen(suffix);
            if (filename.size() > start + suffix_size &&
                filename.compare(0, start, prefix) == 0 &&
                filename.compare(filename.size() - suffix_size, suffix_size, suffix) == 0) {
                const size_t end = filename.size() - suffix_size;
                try {
                    int64_t step = std::stoll(filename.substr(start, end - start));
                    // Incomplete publish-on-sidecar checkpoints are not
                    // recovery points and therefore must not evict an older
                    // complete pair from the retention set.
                    if (adaptive_fmm_depth_checkpoint_requested(config) &&
                        !has_adaptive_fmm_depth_checkpoint_sidecar(
                            entry.path().string())) {
                        continue;
                    }
                    if (config.acceleration.yorp.adaptive) {
                        const std::filesystem::path sidecar(
                            entry.path().string() + ".adaptive_yorp");
                        std::error_code ec;
                        if (!std::filesystem::is_regular_file(sidecar, ec) || ec ||
                            std::filesystem::file_size(sidecar, ec) == 0 || ec) {
                            continue;
                        }
                    }
                    if (config.acceleration.external_bodies.enabled) {
                        std::ifstream checkpoint(entry.path(), std::ios::binary); UnifiedHeader header{};
                        checkpoint.read(reinterpret_cast<char*>(&header), sizeof(header));
                        if (!checkpoint || !ExternalBodyManager::checkpoint_sidecar_matches(
                                entry.path().string(), header.current_step,
                                header.current_time, config)) {
                            continue;
                        }
                    }
                    checkpoints.push_back({step, entry.path().string()});
                } catch (...) {
                    continue;
                }
            }
        }
    }

    // Sort by step (descending)
    std::sort(checkpoints.begin(), checkpoints.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Remove old checkpoints beyond keep_count
    for (size_t i = keep_count; i < checkpoints.size(); ++i) {
        const std::string& path = checkpoints[i].second;
        std::filesystem::remove(path + ".sph_stage");
        std::filesystem::remove(path);
        std::error_code ignored;
        std::filesystem::remove(path + ".respa_sync", ignored);
        std::filesystem::remove(path + ".adaptive_yorp", ignored);
        std::filesystem::remove(path + ".adaptive_fmm_depth", ignored);
        std::filesystem::remove(path + ".adaptive_fmm_depth.tmp", ignored);
        std::filesystem::remove(path + ".external_bodies", ignored);
        std::filesystem::remove(path + ".external_bodies.tmp", ignored);
        std::filesystem::remove(path + ".shear_box", ignored);
        std::cout << "Removed old checkpoint and sidecars: " << path << std::endl;
    }
}

// =============================================================================
// Utility: Check if simulation should checkpoint
// =============================================================================

inline bool should_checkpoint(const SimulationState& state, const SimConfig& config) {
    if (!config.io.checkpoint_enabled) {
        return false;
    }
    return (state.current_step % config.io.checkpoint_interval) == 0;
}

} // namespace cudadem

#endif // CUDADEM_LOAD_OR_RESUME_H

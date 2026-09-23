// =============================================================================
// Unified Binary Reader Implementation
// =============================================================================
// Host-only implementation — reads the unified GRANIT binary format.
// =============================================================================

#include "binary_reader.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace cudadem {

namespace {
bool field_layout(FieldID id, DType* dtype, uint8_t* count) {
    switch (id) {
        case FieldID::MATERIAL_ID: case FieldID::NUM_CONTACTS: case FieldID::CLUMP_ID:
        case FieldID::FLAWS_NTOT:
            *dtype=DType::I32; *count=1; return true;
        case FieldID::LIFECYCLE_STATE: *dtype=DType::U8; *count=1; return true;
        case FieldID::CLUMP_OFFSET: *dtype=DType::F32; *count=3; return true;
        case FieldID::WINDING_X: case FieldID::WINDING_Y: case FieldID::WINDING_Z:
            *dtype=DType::I64; *count=1; return true;
        case FieldID::GRAVITY_ACCELERATION:
        case FieldID::OMEGA: *dtype=DType::F64; *count=3; return true;
        case FieldID::QUATERNION: *dtype=DType::F64; *count=4; return true;
        case FieldID::DENSITY: case FieldID::DRHODT:
        case FieldID::ENERGY_COLD:
        case FieldID::ENERGY_THERMAL_PEAK:
        case FieldID::PRESSURE_PEAK:
        case FieldID::FRACTURE_AREA:
        case FieldID::FRAGMENT_LENGTH:
        case FieldID::DAMPING_ENERGY:
        case FieldID::ENERGY: case FieldID::DEDT: *dtype=DType::F64; *count=1; return true;
        case FieldID::FORCE: case FieldID::TORQUE: *dtype=DType::F32; *count=3; return true;
        case FieldID::PRESSURE: case FieldID::C_SOUND:
        case FieldID::SMOOTHING_LENGTH: case FieldID::RADIUS: case FieldID::MASS:
        case FieldID::MOMENT_OF_INERTIA: case FieldID::DAMAGE: case FieldID::CRACK_STRAIN:
        case FieldID::FLAWS_STRAIN_MIN: case FieldID::FLAWS_STRAIN_MAX: case FieldID::FLAWS_M0:
        case FieldID::S_XX: case FieldID::S_YY: case FieldID::S_ZZ: case FieldID::S_XY:
        case FieldID::S_XZ: case FieldID::S_YZ: case FieldID::DISTENSION:
            *dtype=DType::F32; *count=1; return true;
        default: return false;
    }
}
bool has_field(const std::vector<FieldDescriptor>& descriptors, FieldID id) {
    for (const auto& d : descriptors) if (d.field_id == static_cast<uint16_t>(id)) return true;
    return false;
}
bool allowed_input_field(FieldID id, bool is_sph) {
    if (is_sph) {
        return id == FieldID::MATERIAL_ID || id == FieldID::CLUMP_ID || id == FieldID::MASS ||
               id == FieldID::DENSITY || id == FieldID::SMOOTHING_LENGTH || id == FieldID::ENERGY ||
               id == FieldID::DAMAGE;
    }
    return id == FieldID::MATERIAL_ID || id == FieldID::CLUMP_ID || id == FieldID::RADIUS ||
           id == FieldID::OMEGA || id == FieldID::WINDING_X ||
           id == FieldID::WINDING_Y || id == FieldID::WINDING_Z;
}
} // namespace

bool read_unified_binary(
    const std::string& path,
    ParticleSoA& particles,
    SimulationState& state,
    const SimConfig& config,
    int32_t extra_capacity)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open unified binary file: " << path << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // --- Read header ---
    UnifiedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        std::cerr << "ERROR: Failed to read header from: " << path << std::endl;
        return false;
    }

    // Validate magic
    if (std::memcmp(header.magic, MAGIC, 8) != 0) {
        std::cerr << "ERROR: Invalid magic in unified binary file: " << path << std::endl;
        std::cerr << "  Expected: GRANIT, got: "
                  << std::string(header.magic, 8) << std::endl;
        return false;
    }

    if (header.version != FORMAT_VERSION || header.header_size != HEADER_SIZE) {
        std::cerr << "ERROR: Unsupported unified binary version "
                  << header.version << " (expected " << FORMAT_VERSION << ")" << std::endl;
        return false;
    }
    if (header.num_particles == 0 ||
        header.num_particles > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        std::cerr << "ERROR: Particle count is zero or exceeds runtime int32 limit" << std::endl;
        return false;
    }
    const BinaryPhysics expected_physics = config.physics == SimulationPhysics::SPH ? BinaryPhysics::SPH : BinaryPhysics::DEM;
    if (header.physics != static_cast<uint8_t>(expected_physics) ||
        (header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT) &&
         header.record_kind != static_cast<uint8_t>(BinaryRecordKind::OUTPUT) &&
         header.record_kind != static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT))) {
        std::cerr << "ERROR: Binary physics mode or record kind is invalid for this simulation" << std::endl;
        return false;
    }
    if (header.current_step < 0 || !std::isfinite(header.current_time) ||
        !std::isfinite(header.dt) || header.dt < 0.0 ||
        (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::INPUT) &&
         (header.current_step != 0 || header.current_time != 0.0 || header.dt != 0.0 ||
          header.max_contacts_per_particle != 0 || header.total_contacts != 0)) ||
        (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::OUTPUT) &&
         (header.max_contacts_per_particle != 0 || header.total_contacts != 0))) {
        std::cerr << "ERROR: Invalid record metadata for unified binary kind" << std::endl;
        return false;
    }
    if (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
        header.material_database_fingerprint != config.material_database.fingerprint) {
        std::cerr << "ERROR: Checkpoint material database fingerprint mismatch" << std::endl;
        return false;
    }
    if (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
        header.clump_metadata_fingerprint != config.clump.clump_metadata_fingerprint) {
        std::cerr << "ERROR: Checkpoint clump metadata fingerprint mismatch" << std::endl;
        return false;
    }

    int32_t N = static_cast<int32_t>(header.num_particles);
    uint16_t num_descriptors = header.num_descriptors;

    std::cout << "Reading unified binary: " << path << std::endl;
    std::cout << "  Particles: " << N
              << ", Descriptors: " << num_descriptors
              << ", Version: " << header.version << std::endl;

    // --- Read descriptor table ---
    std::vector<FieldDescriptor> descriptors(num_descriptors);
    if (num_descriptors > 0) {
        file.read(reinterpret_cast<char*>(descriptors.data()),
                  num_descriptors * sizeof(FieldDescriptor));
        if (!file) {
            std::cerr << "ERROR: Failed to read descriptor table from: " << path << std::endl;
            return false;
        }
    }
    const uint64_t tier1_size = static_cast<uint64_t>(N) * (6ULL * sizeof(double) + sizeof(int32_t));
    const uint64_t tier2_start = HEADER_SIZE + static_cast<uint64_t>(num_descriptors) * sizeof(FieldDescriptor) + tier1_size;
    if (tier2_start > static_cast<uint64_t>(file_size)) {
        std::cerr << "ERROR: Truncated descriptor table or Tier 1 data" << std::endl;
        return false;
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::set<uint16_t> field_ids;
    uint64_t tier2_end = 0;
    for (const auto& desc : descriptors) {
        DType expected_dtype; uint8_t expected_count;
        const FieldID field_id = static_cast<FieldID>(desc.field_id);
        const bool legacy_f32_density =
            (field_id == FieldID::DENSITY || field_id == FieldID::DRHODT) &&
            desc.dtype == static_cast<uint8_t>(DType::F32);
        if (desc.reserved != 0 || !field_ids.insert(desc.field_id).second ||
            !field_layout(field_id, &expected_dtype, &expected_count) ||
            (desc.dtype != static_cast<uint8_t>(expected_dtype) && !legacy_f32_density && !(field_id == FieldID::FORCE &&
                desc.dtype == static_cast<uint8_t>(DType::F64))) ||
            desc.count != expected_count) {
            std::cerr << "ERROR: Invalid, duplicate, or unsupported field descriptor" << std::endl;
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(N) * desc.count * dtype_size(static_cast<DType>(desc.dtype));
        if (desc.byte_offset > UINT64_MAX - bytes || tier2_start > UINT64_MAX - desc.byte_offset - bytes ||
            tier2_start + desc.byte_offset + bytes > static_cast<uint64_t>(file_size)) {
            std::cerr << "ERROR: Tier 2 descriptor out of bounds" << std::endl;
            return false;
        }
        ranges.emplace_back(desc.byte_offset, desc.byte_offset + bytes);
        tier2_end = std::max(tier2_end, desc.byte_offset + bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) if (ranges[i - 1].second > ranges[i].first) {
        std::cerr << "ERROR: Overlapping Tier 2 descriptors" << std::endl; return false;
    }
    const bool is_sph = expected_physics == BinaryPhysics::SPH;
    if (is_sph && config.acceleration.enable_self_gravity &&
        header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
        !has_field(descriptors, FieldID::GRAVITY_ACCELERATION)) {
        std::cerr << "ERROR: SPH gravity checkpoint requires the cached slow acceleration field" << std::endl;
        return false;
    }
    const MaterialDatabase configured_materials = load_material_database(
        config.material_database.path, config.dem_damping_model);
    if (!has_field(descriptors, FieldID::MATERIAL_ID) ||
        (!is_sph && !has_field(descriptors, FieldID::RADIUS)) ||
        (is_sph && (!has_field(descriptors, FieldID::MASS) || !has_field(descriptors, FieldID::DENSITY)))) {
        std::cerr << "ERROR: Required material/physics fields are missing" << std::endl;
        return false;
    }
    if (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::INPUT)) {
        for (const auto& descriptor : descriptors) {
            if (!allowed_input_field(static_cast<FieldID>(descriptor.field_id), is_sph)) {
                std::cerr << "ERROR: Unsupported field in fresh "
                          << (is_sph ? "SPH" : "DEM") << " input record" << std::endl;
                return false;
            }
        }
    }
    if (header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT) &&
        (!has_field(descriptors, FieldID::CLUMP_ID) ||
         (config.acceleration.external_bodies.enabled && !has_field(descriptors, FieldID::LIFECYCLE_STATE)))) {
        std::cerr << "ERROR: Unified v3 output/checkpoint requires CLUMP_ID and, for external bodies, LIFECYCLE_STATE" << std::endl;
        return false;
    }
    if (has_field(descriptors, FieldID::CLUMP_OFFSET) &&
        (is_sph || header.record_kind != static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT))) {
        std::cerr << "ERROR: CLUMP_OFFSET is valid only in DEM checkpoints" << std::endl;
        return false;
    }
    if (!is_sph && header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
        !has_field(descriptors, FieldID::CLUMP_OFFSET)) {
        std::cerr << "ERROR: DEM checkpoints require CLUMP_OFFSET" << std::endl;
        return false;
    }
    const bool has_flaws_ntot = has_field(descriptors, FieldID::FLAWS_NTOT);
    const bool has_flaws_strain_min = has_field(descriptors, FieldID::FLAWS_STRAIN_MIN);
    const bool has_flaws_strain_max = has_field(descriptors, FieldID::FLAWS_STRAIN_MAX);
    const bool has_flaws_m0 = has_field(descriptors, FieldID::FLAWS_M0);
    const bool any_flaw_metadata = has_flaws_ntot || has_flaws_strain_min ||
                                   has_flaws_strain_max || has_flaws_m0;
    const bool all_flaw_metadata = has_flaws_ntot && has_flaws_strain_min &&
                                   has_flaws_strain_max && has_flaws_m0;
    if (any_flaw_metadata &&
        (!all_flaw_metadata || !is_sph ||
         header.record_kind != static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT))) {
        std::cerr << "ERROR: Grady-Kipp flaw metadata must be a complete SPH checkpoint set" << std::endl;
        return false;
    }
    const bool has_wx = has_field(descriptors, FieldID::WINDING_X);
    const bool has_wy = has_field(descriptors, FieldID::WINDING_Y);
    const bool has_wz = has_field(descriptors, FieldID::WINDING_Z);
    if ((has_wx || has_wy || has_wz) && !(has_wx && has_wy && has_wz)) {
        std::cerr << "ERROR: WINDING_X/Y/Z must be present as a complete set" << std::endl;
        return false;
    }
    if (config.shear_box.enabled &&
        header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT) && !has_wx) {
        std::cerr << "ERROR: Shear-box output/checkpoint records require WINDING_X/Y/Z" << std::endl;
        return false;
    }
    const uint64_t payload_end = tier2_start + tier2_end;
    const uint64_t contact_blob_size = static_cast<uint64_t>(N) * header.max_contacts_per_particle * sizeof(ContactHistory);
    if (header.total_contacts > static_cast<uint64_t>(N) * header.max_contacts_per_particle ||
        (header.record_kind != static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) && payload_end != static_cast<uint64_t>(file_size)) ||
        (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
         ((header.total_contacts == 0 && payload_end != static_cast<uint64_t>(file_size)) ||
          (header.total_contacts > 0 && (contact_blob_size > UINT64_MAX - payload_end ||
                                         payload_end + contact_blob_size != static_cast<uint64_t>(file_size)))))) {
        std::cerr << "ERROR: Invalid trailing data or contact metadata" << std::endl;
        return false;
    }

    // --- Set metadata ---
    state.current_step = header.current_step;
    state.current_time = header.current_time;
    state.dt = header.dt;

    // --- Allocate GPU memory ---
    int32_t total_capacity = N + extra_capacity;
    allocate_particle_soa(particles, total_capacity);
    particles.num_particles = N;

    // --- Tier 1: Read fixed fields from file into host, then copy to GPU ---
    {
        std::vector<double> buf_d(N);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 position" << std::endl; return false; }
        cudaMemcpy(particles.pos_x, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 position" << std::endl; return false; }
        cudaMemcpy(particles.pos_y, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 position" << std::endl; return false; }
        cudaMemcpy(particles.pos_z, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 velocity" << std::endl; return false; }
        cudaMemcpy(particles.vel_x, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 velocity" << std::endl; return false; }
        cudaMemcpy(particles.vel_y, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
        for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: Non-finite Tier 1 velocity" << std::endl; return false; }
        cudaMemcpy(particles.vel_z, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    }

    {
        std::vector<int32_t> buf_i(N);
        file.read(reinterpret_cast<char*>(buf_i.data()), N * sizeof(int32_t));
        std::unordered_set<int32_t> ids;
        ids.reserve(buf_i.size());
        for (const int32_t id : buf_i) {
            if (id < 0 || !ids.insert(id).second) {
                std::cerr << "ERROR: global_id values must be unique and nonnegative" << std::endl;
                return false;
            }
        }
        cudaMemcpy(particles.global_id, buf_i.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    if (!file) {
        std::cerr << "ERROR: Failed to read Tier 1 data from: " << path << std::endl;
        return false;
    }

    // --- Tier 2: Read descriptor-driven fields ---
    particles.input_has_omega = false;
    particles.input_has_smoothing_length = has_field(descriptors, FieldID::SMOOTHING_LENGTH);
    particles.input_has_clump_offset = false;
    particles.input_has_winding = has_wx;
    particles.input_has_damage_flaw_metadata =
        is_sph && header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
        all_flaw_metadata;
    particles.source_record_kind = header.record_kind;
    for (const auto& desc : descriptors) {
        FieldID fid = static_cast<FieldID>(desc.field_id);
        uint8_t  cnt = desc.count;
        size_t   elem_size = dtype_size(static_cast<DType>(desc.dtype));
        file.seekg(static_cast<std::streamoff>(tier2_start + desc.byte_offset), std::ios::beg);

        switch (fid) {
            case FieldID::MATERIAL_ID: {
                std::vector<int32_t> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(int32_t));
                for (const int32_t id : buf) {
                    if (id < 0 || static_cast<uint32_t>(id) >= config.material_database.material_count) {
                        std::cerr << "ERROR: Particle material ID is outside the configured dense material database" << std::endl;
                        return false;
                    }
                    const MaterialRecord& material = configured_materials.materials[static_cast<size_t>(id)];
                    if ((is_sph && !material.has_sph) || (!is_sph && !material.has_dem)) {
                        std::cerr << "ERROR: Particle material does not define the active physics block" << std::endl;
                        return false;
                    }
                }
                cudaMemcpy(particles.material_id, buf.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::CLUMP_ID: {
                std::vector<int32_t> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(int32_t));
                std::set<int32_t> positive_ids;
                int32_t max_clump_id = 0;
                for (const int32_t id : buf) {
                    if (id < -1) {
                        std::cerr << "ERROR: Invalid clump_id for active physics" << std::endl;
                        return false;
                    }
                    if (id > 0) {
                        positive_ids.insert(id);
                        max_clump_id = std::max(max_clump_id, id);
                    }
                }
                if (static_cast<int32_t>(positive_ids.size()) != max_clump_id) {
                    std::cerr << "ERROR: Positive clump_id values must be contiguous 1..K" << std::endl;
                    return false;
                }
                // In SPH this field is the exact SPHSOL Part ID.  A positive
                // maximum therefore requires every active Part from 0..K;
                // unlike DEM, zero is not merely an ungrouped-particle value.
                if (is_sph && max_clump_id > 0 &&
                    std::find(buf.begin(), buf.end(), 0) == buf.end()) {
                    std::cerr << "ERROR: SPH Part IDs must be dense 0..K" << std::endl;
                    return false;
                }
                cudaMemcpy(particles.clump_id, buf.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::LIFECYCLE_STATE: {
                std::vector<uint8_t> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N);
                for (uint8_t value : buf) if (value > 1) { std::cerr << "ERROR: invalid lifecycle_state" << std::endl; return false; }
                cudaMemcpy(particles.lifecycle_state, buf.data(), N, cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::CLUMP_OFFSET: {
                particles.input_has_clump_offset = true;
                std::vector<float> buf(N);
                float* dst_ptrs[3] = {particles.clump_off_x, particles.clump_off_y, particles.clump_off_z};
                for (int c = 0; c < 3; ++c) {
                    file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                    for (const float value : buf) if (!std::isfinite(value)) {
                        std::cerr << "ERROR: clump_offset must be finite" << std::endl; return false;
                    }
                    cudaMemcpy(dst_ptrs[c], buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                }
                break;
            }
            case FieldID::WINDING_X:
            case FieldID::WINDING_Y:
            case FieldID::WINDING_Z: {
                std::vector<int64_t> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), static_cast<size_t>(N) * sizeof(int64_t));
                if (fid == FieldID::WINDING_Z && config.shear_box.enabled &&
                    std::any_of(buf.begin(),buf.end(),[](int64_t value){ return value != 0; })) {
                    std::cerr << "ERROR: WINDING_Z must be zero in supported shear-box mode" << std::endl;
                    return false;
                }
                int64_t* dst = fid == FieldID::WINDING_X ? particles.winding_x :
                               fid == FieldID::WINDING_Y ? particles.winding_y : particles.winding_z;
                cudaMemcpy(dst, buf.data(), static_cast<size_t>(N) * sizeof(int64_t), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::RADIUS: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value > 0.0f)) { std::cerr << "ERROR: DEM radius must be finite and positive" << std::endl; return false; }
                cudaMemcpy(particles.radius, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::OMEGA: {
                particles.input_has_omega = true;
                std::vector<double> buf(N);
                for (int c = 0; c < cnt; ++c) {
                    file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                    for (double value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: omega must be finite" << std::endl; return false; }
                    double* dst = (c == 0) ? particles.omega_x :
                                  (c == 1) ? particles.omega_y : particles.omega_z;
                    cudaMemcpy(dst, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                }
                break;
            }
            case FieldID::MASS: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value > 0.0f)) { std::cerr << "ERROR: SPH mass must be finite and positive" << std::endl; return false; }
                cudaMemcpy(particles.mass, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::MOMENT_OF_INERTIA: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value > 0.0f)) { std::cerr << "ERROR: moment of inertia must be finite and positive" << std::endl; return false; }
                cudaMemcpy(particles.moment_of_inertia, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::GRAVITY_ACCELERATION:
            case FieldID::FORCE: {
                const bool gravity = static_cast<FieldID>(desc.field_id) == FieldID::GRAVITY_ACCELERATION;
                double* dst[3] = {gravity ? particles.gravity_force_x : particles.force_x,
                                  gravity ? particles.gravity_force_y : particles.force_y,
                                  gravity ? particles.gravity_force_z : particles.force_z};
                for (int c = 0; c < cnt; ++c) {
                    std::vector<double> values(N);
                    if (desc.dtype == static_cast<uint8_t>(DType::F64))
                        file.read(reinterpret_cast<char*>(values.data()), N * sizeof(double));
                    else {
                        std::vector<float> legacy(N);
                        file.read(reinterpret_cast<char*>(legacy.data()), N * sizeof(float));
                        std::copy(legacy.begin(), legacy.end(), values.begin());
                    }
                    for (double value : values) if (!std::isfinite(value)) {
                        std::cerr << "ERROR: force/acceleration must be finite" << std::endl; return false;
                    }
                    CUDA_CHECK(cudaMemcpy(dst[c], values.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                }
                break;
            }
            case FieldID::TORQUE: {
                std::vector<float> buf(N);
                double* dst_ptrs[3] = { particles.torque_x, particles.torque_y, particles.torque_z };
                for (int c = 0; c < cnt; ++c) {
                    file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                    for (float value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: torque must be finite" << std::endl; return false; }
                    std::vector<double> buf_d(N);
                    for (int32_t i = 0; i < N; ++i) buf_d[i] = static_cast<double>(buf[i]);
                    cudaMemcpy(dst_ptrs[c], buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                }
                break;
            }
            case FieldID::QUATERNION: {
                std::vector<double> buf(N);
                double* dst_ptrs[4] = { particles.quat_w, particles.quat_x, particles.quat_y, particles.quat_z };
                for (int c = 0; c < cnt; ++c) {
                    file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                    for (double value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: quaternion must be finite" << std::endl; return false; }
                    cudaMemcpy(dst_ptrs[c], buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                }
                break;
            }
            case FieldID::NUM_CONTACTS: {
                std::vector<int32_t> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(int32_t));
                uint64_t contact_sum = 0;
                for (int32_t value : buf) {
                    if (value < 0 || static_cast<uint32_t>(value) > header.max_contacts_per_particle) {
                        std::cerr << "ERROR: num_contacts is outside the declared checkpoint stride" << std::endl;
                        return false;
                    }
                    contact_sum += static_cast<uint32_t>(value);
                }
                if (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) &&
                    contact_sum != header.total_contacts) {
                    std::cerr << "ERROR: num_contacts does not match checkpoint total_contacts" << std::endl;
                    return false;
                }
                cudaMemcpy(particles.num_contacts, buf.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::SMOOTHING_LENGTH: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value > 0.0f)) { std::cerr << "ERROR: smoothing length must be finite and positive" << std::endl; return false; }
                cudaMemcpy(particles.smoothing_length, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::DENSITY: {
                std::vector<double> buf_d(N);
                if (desc.dtype == static_cast<uint8_t>(DType::F32)) {
                    std::vector<float> buf_f(N);
                    file.read(reinterpret_cast<char*>(buf_f.data()), N * sizeof(float));
                    for (int32_t k = 0; k < N; ++k) buf_d[k] = static_cast<double>(buf_f[k]);
                } else {
                    file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
                }
                for (double value : buf_d) if (!(std::isfinite(value) && value > 0.0)) { std::cerr << "ERROR: SPH density must be finite and positive" << std::endl; return false; }
                cudaMemcpy(particles.density, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::PRESSURE: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: pressure must be finite" << std::endl; return false; }
                cudaMemcpy(particles.pressure, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::C_SOUND: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 0.0f)) { std::cerr << "ERROR: sound speed must be finite and nonnegative" << std::endl; return false; }
                cudaMemcpy(particles.c_sound, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::ENERGY: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: energy must be finite" << std::endl; return false; }
                cudaMemcpy(particles.energy, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::DAMAGE: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 0.0f && value <= 1.0f)) { std::cerr << "ERROR: damage must be finite and in [0,1]" << std::endl; return false; }
                cudaMemcpy(particles.damage, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::CRACK_STRAIN: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: crack strain must be finite" << std::endl; return false; }
                cudaMemcpy(particles.crack_strain, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::FLAWS_NTOT: {
                std::vector<int32_t> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(int32_t));
                for (int32_t value : buf) if (value < 0) { std::cerr << "ERROR: flaw count must be nonnegative" << std::endl; return false; }
                cudaMemcpy(particles.flaws_ntot, buf.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::FLAWS_STRAIN_MIN: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 0.0f)) { std::cerr << "ERROR: flaw minimum strain must be finite and nonnegative" << std::endl; return false; }
                cudaMemcpy(particles.flaws_strain_min, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::FLAWS_STRAIN_MAX: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 0.0f)) { std::cerr << "ERROR: flaw maximum strain must be finite and nonnegative" << std::endl; return false; }
                cudaMemcpy(particles.flaws_strain_max, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::FLAWS_M0: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 0.0f)) { std::cerr << "ERROR: flaw m0 must be finite and nonnegative" << std::endl; return false; }
                cudaMemcpy(particles.flaws_m0, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::S_XX: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_xx, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::S_YY: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_yy, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::S_ZZ: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_zz, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::S_XY: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_xy, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::S_XZ: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_xz, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::S_YZ: { std::vector<float> buf(N); file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float)); for (float v : buf) if (!std::isfinite(v)) { std::cerr << "ERROR: stress must be finite" << std::endl; return false; } cudaMemcpy(particles.S_yz, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice); break; }
            case FieldID::DISTENSION: {
                std::vector<float> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(float));
                for (float value : buf) if (!(std::isfinite(value) && value >= 1.0f)) { std::cerr << "ERROR: distension must be finite and at least one" << std::endl; return false; }
                cudaMemcpy(particles.distension, buf.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::DRHODT: {
                std::vector<double> buf_d(N);
                if (desc.dtype == static_cast<uint8_t>(DType::F32)) {
                    std::vector<float> buf_f(N);
                    file.read(reinterpret_cast<char*>(buf_f.data()), N * sizeof(float));
                    for (int32_t k = 0; k < N; ++k) buf_d[k] = static_cast<double>(buf_f[k]);
                } else {
                    file.read(reinterpret_cast<char*>(buf_d.data()), N * sizeof(double));
                }
                for (double value : buf_d) if (!std::isfinite(value)) { std::cerr << "ERROR: density rate must be finite" << std::endl; return false; }
                cudaMemcpy(particles.rho_rate, buf_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                break;
            }
            case FieldID::ENERGY_COLD: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value)) {
                    std::cerr << "ERROR: invalid SPH energy_cold history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.energy_cold, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::ENERGY_THERMAL_PEAK: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value) || value < 0.0) {
                    std::cerr << "ERROR: invalid SPH energy_thermal_peak history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.energy_thermal_peak, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::PRESSURE_PEAK: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value) || value < 0.0) {
                    std::cerr << "ERROR: invalid SPH pressure_peak history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.pressure_peak, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::FRACTURE_AREA: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value) || value < 0.0) {
                    std::cerr << "ERROR: invalid SPH fracture_area history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.fracture_area, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::FRAGMENT_LENGTH: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value) || value < 0.0) {
                    std::cerr << "ERROR: invalid SPH fragment_length history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.fragment_length, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::DAMPING_ENERGY: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value) || value < 0.0) {
                    std::cerr << "ERROR: invalid SPH damping_energy history" << std::endl; return false;
                }
                CUDA_CHECK(cudaMemcpy(particles.damping_energy, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice));
                break;
            }
            case FieldID::DEDT: {
                std::vector<double> buf(N);
                file.read(reinterpret_cast<char*>(buf.data()), N * sizeof(double));
                for (double value : buf) if (!std::isfinite(value)) { std::cerr << "ERROR: energy rate must be finite" << std::endl; return false; }
                cudaMemcpy(particles.dedt, buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
                break;
            }
            default: return false; // descriptor validation above makes this unreachable
        }

        if (!file) {
            std::cerr << "ERROR: Failed to read field 0x"
                      << std::hex << desc.field_id << std::dec
                      << " from: " << path << std::endl;
            return false;
        }
    }

    // CLUMP_ID is intentionally optional only for a fresh v3 input.  Such
    // files represent ordinary live particles by default, not dead particles.
    if (!has_field(descriptors, FieldID::CLUMP_ID)) {
        std::vector<int32_t> normal_ids(N, 0);
        cudaMemcpy(particles.clump_id, normal_ids.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    if (!has_field(descriptors, FieldID::LIFECYCLE_STATE)) {
        CUDA_CHECK(cudaMemset(particles.lifecycle_state, 0, N * sizeof(uint8_t)));
    }

    if (is_sph &&
        header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) && config.sph.track_history) {
        for (FieldID field : {FieldID::ENERGY_COLD, FieldID::ENERGY_THERMAL_PEAK, FieldID::PRESSURE_PEAK, FieldID::FRACTURE_AREA, FieldID::FRAGMENT_LENGTH, FieldID::DAMPING_ENERGY}) {
            if (!has_field(descriptors, field)) {
                std::cerr << "ERROR: track_history resume requires a checkpoint containing SPH history; "
                             "legacy history cannot be reconstructed" << std::endl;
                return false;
            }
        }
    }

    // --- Contact history blob (checkpoint only) ---
    if (header.record_kind == static_cast<uint8_t>(BinaryRecordKind::CHECKPOINT) && header.total_contacts > 0) {
        int32_t max_contacts = header.max_contacts_per_particle;
        allocate_contact_history(state.contact_history, N, max_contacts);

        int32_t history_size = N * max_contacts;
        std::vector<ContactHistory> h_contact_history(history_size);
        file.seekg(static_cast<std::streamoff>(payload_end), std::ios::beg);
        file.read(reinterpret_cast<char*>(h_contact_history.data()),
                  history_size * sizeof(ContactHistory));
        if (!file) {
            std::cerr << "ERROR: Failed to read contact history from: " << path << std::endl;
            return false;
        }
        cudaMemcpy(state.contact_history.data, h_contact_history.data(),
                   history_size * sizeof(ContactHistory), cudaMemcpyHostToDevice);
        std::cout << "  Contact history: " << header.total_contacts
                  << " contacts (capacity: " << history_size << ")" << std::endl;
    }

    file.close();
    std::cout << "Unified binary loaded: " << N << " particles" << std::endl;
    return true;
}

} // namespace cudadem

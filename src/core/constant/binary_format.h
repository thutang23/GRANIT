// =============================================================================
// Unified Binary Format — Header, Field Descriptors, Reader & Writer
// =============================================================================
// Replaces CUDADEM, DEMOUT, and DEMCKPT with a single self-describing format.
//
// Layout:
//   [Header: 72 bytes]
//   [Descriptor table: num_descriptors × 16 bytes]
//   [Tier 1 — fixed, always present]:
//     pos_x[N] double, pos_y[N] double, pos_z[N] double,
//     vel_x[N] double, vel_y[N] double, vel_z[N] double,
//     global_id[N] int32_t
//   [Tier 2 — descriptor-driven SoA arrays]
//   [Contact history blob — if has_contact_blob=1]
// =============================================================================

#ifndef CUDADEM_BINARY_FORMAT_H
#define CUDADEM_BINARY_FORMAT_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// The writer function references ParticleSoA, SimulationState, SimConfig.
// These are defined in the headers below — safe to include (no circular deps).
#include "command_line_interface.h"
#include "system_initialize.h"

namespace cudadem {

// =============================================================================
// Magic & Version Constants
// =============================================================================
constexpr char MAGIC[8] = {'G','R','A','N','I','T',0,0};
constexpr uint16_t FORMAT_VERSION = 3;
constexpr uint16_t HEADER_SIZE = 72;

// =============================================================================
// DType Enum — data type tags for field descriptors
// =============================================================================
enum class DType : uint8_t {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I16 = 3,
    U8  = 4,
    BOOL = 5,
    I64 = 6
};

enum class BinaryPhysics : uint8_t { DEM = 1, SPH = 2 };
enum class BinaryRecordKind : uint8_t { INPUT = 1, OUTPUT = 2, CHECKPOINT = 3 };

// Get the byte size of a DType
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F64:  return 8;
        case DType::I32:  return 4;
        case DType::I16:  return 2;
        case DType::U8:   return 1;
        case DType::BOOL: return 1;
        case DType::I64:  return 8;
        default:          return 0;
    }
}

// =============================================================================
// FieldID Enum — registry of known fields
// =============================================================================
enum class FieldID : uint16_t {
    // SPH fields
    // Fresh inputs may carry legacy f32 density; runtime output/checkpoints
    // write f64 so weak compression survives restart and comparison.
    DENSITY          = 0x0001,  // f32 input or f64 runtime, 1
    PRESSURE         = 0x0002,  // f32, 1
    C_SOUND          = 0x0003,  // f32, 1
    SMOOTHING_LENGTH = 0x0004,  // f32, 1
    ENERGY           = 0x0005,  // f64, 1
    MATERIAL_ID       = 0x0006,  // i32, 1 (required in every v3 record)

    // DEM fields
    OMEGA            = 0x0010,  // f64, 3
    RADIUS           = 0x0011,  // f32, 1
    MASS             = 0x0012,  // f32, 1
    MOMENT_OF_INERTIA = 0x0013, // f32, 1
    FORCE            = 0x0014,  // f32, 3
    TORQUE           = 0x0015,  // f32, 3
    QUATERNION       = 0x0016,  // f64, 4 (checkpoint only)
    NUM_CONTACTS     = 0x0017,  // i32, 1 (checkpoint only)
    CLUMP_ID         = 0x0018,  // i32, 1 (optional input; mandatory output/checkpoint)
    CLUMP_OFFSET     = 0x0019,  // f32, 3 (DEM checkpoint only, body frame)
    WINDING_X        = 0x001A,  // i64, 1 (canonical periodic image)
    WINDING_Y        = 0x001B,  // i64, 1
    WINDING_Z        = 0x001C,  // i64, 1 (reserved in current shear mode)
    LIFECYCLE_STATE  = 0x001D,  // u8, 1 (0 live, 1 externally absorbed)

    // SPH material state (checkpoint only)
    DAMAGE           = 0x0020,  // f32, 1
    CRACK_STRAIN     = 0x0021,  // f32, 1
    // Compact deterministic Grady-Kipp flaw realization (SPH checkpoint only).
    FLAWS_NTOT       = 0x0022,  // i32, 1
    FLAWS_STRAIN_MIN = 0x0023,  // f32, 1
    FLAWS_STRAIN_MAX = 0x0024,  // f32, 1
    FLAWS_M0         = 0x0025,  // f32, 1

    // Deviatoric stress tensor (checkpoint only)
    S_XX = 0x0030,  S_YY = 0x0031,  S_ZZ = 0x0032,
    S_XY = 0x0033,  S_XZ = 0x0034,  S_YZ = 0x0035,

    // P-alpha porosity (checkpoint only)
    DISTENSION       = 0x0040,  // f32, 1
    DRHODT           = 0x0041,  // f32 legacy or f64 runtime, 1
    DEDT             = 0x0042,  // f64, 1

    // Optional SPH physical history (output and checkpoint, f64, 1)
    ENERGY_COLD          = 0x0050,
    ENERGY_THERMAL_PEAK  = 0x0051,
    PRESSURE_PEAK        = 0x0052,
    FRACTURE_AREA        = 0x0053,
    FRAGMENT_LENGTH      = 0x0054,
    DAMPING_ENERGY       = 0x0055,

};

// =============================================================================
// Unified Header (72 bytes, fixed, packed)
// =============================================================================
#pragma pack(push, 1)
struct UnifiedHeader {
    char     magic[8];                  // "GRANIT\0\0"
    uint16_t version;                   // 3
    uint16_t header_size;               // 72
    uint64_t num_particles;
    int64_t  current_step;              // 0 for input files
    double   current_time;              // 0.0 for input files
    double   dt;                        // 0.0 for input files
    uint16_t num_descriptors;
    uint8_t  physics;                   // BinaryPhysics
    uint8_t  record_kind;               // BinaryRecordKind
    uint32_t max_contacts_per_particle;
    uint32_t total_contacts;
    uint64_t material_database_fingerprint;
    // Normalized clump metadata (instance IDs, inertias, multipoles, etc.).
    // Zero denotes an unbound fresh input with no configured clump metadata.
    uint64_t clump_metadata_fingerprint;

    UnifiedHeader()
        : version(FORMAT_VERSION), header_size(HEADER_SIZE),
          num_particles(0), current_step(0), current_time(0.0), dt(0.0),
          num_descriptors(0), physics(static_cast<uint8_t>(BinaryPhysics::DEM)),
          record_kind(static_cast<uint8_t>(BinaryRecordKind::INPUT)),
          max_contacts_per_particle(0), total_contacts(0), material_database_fingerprint(0),
          clump_metadata_fingerprint(0)
    {
        std::memcpy(magic, MAGIC, 8);
    }
};
#pragma pack(pop)

static_assert(sizeof(UnifiedHeader) == 72, "UnifiedHeader must be exactly 72 bytes");
static_assert(sizeof(ContactHistory) == 48,
              "Unified v3 checkpoint contact records require a stable 48-byte ContactHistory layout");

// =============================================================================
// Field Descriptor (16 bytes each, packed)
// =============================================================================
#pragma pack(push, 1)
struct FieldDescriptor {
    uint16_t field_id;      // FieldID enum value
    uint8_t  dtype;         // DType enum value
    uint8_t  count;         // component count (1=scalar, 3=vector, 4=quat)
    uint32_t reserved;      // must be zero
    uint64_t byte_offset;   // Tier 2-relative offset

    FieldDescriptor()
        : field_id(0), dtype(0), count(0), reserved(0), byte_offset(0) {}

    FieldDescriptor(FieldID fid, DType dt, uint8_t cnt, uint64_t off)
        : field_id(static_cast<uint16_t>(fid)),
          dtype(static_cast<uint8_t>(dt)),
          count(cnt), reserved(0), byte_offset(off) {}
};
#pragma pack(pop)

static_assert(sizeof(FieldDescriptor) == 16, "FieldDescriptor must be exactly 16 bytes");

// =============================================================================
// compute_descriptor_offsets
// =============================================================================
// Fills byte_offset for each descriptor and returns total Tier 2 data size.
inline uint64_t compute_descriptor_offsets(std::vector<FieldDescriptor>& descriptors,
                                           uint64_t num_particles) {
    uint64_t offset = 0;
    for (auto& desc : descriptors) {
        desc.byte_offset = offset;
        const uint64_t bytes = num_particles * static_cast<uint64_t>(desc.count) *
                               static_cast<uint64_t>(dtype_size(static_cast<DType>(desc.dtype)));
        if (bytes > UINT64_MAX - offset) throw std::overflow_error("Tier 2 binary offset overflow");
        offset += bytes;
    }
    return offset;
}

// =============================================================================
// Helper: write a GPU double array to file (direct — f64 in file)
// =============================================================================
inline void write_double_array(std::ofstream& file, const double* d_ptr, int32_t N) {
    std::vector<double> buf(N);
    cudaMemcpy(buf.data(), d_ptr, N * sizeof(double), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(buf.data()), N * sizeof(double));
}

// =============================================================================
// Helper: write a GPU double array as float to file
// =============================================================================
inline void write_double_as_float(std::ofstream& file, const double* d_ptr, int32_t N) {
    std::vector<double> buf_d(N);
    cudaMemcpy(buf_d.data(), d_ptr, N * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<float> buf_f(N);
    for (int32_t i = 0; i < N; ++i) buf_f[i] = static_cast<float>(buf_d[i]);
    file.write(reinterpret_cast<const char*>(buf_f.data()), N * sizeof(float));
}

// =============================================================================
// Helper: write a GPU float array to file
// =============================================================================
inline void write_float_array(std::ofstream& file, const float* d_ptr, int32_t N) {
    std::vector<float> buf(N);
    cudaMemcpy(buf.data(), d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(buf.data()), N * sizeof(float));
}

// =============================================================================
// Helper: write a GPU int32 array to file
// =============================================================================
inline void write_int32_array(std::ofstream& file, const int32_t* d_ptr, int32_t N) {
    std::vector<int32_t> buf(N);
    cudaMemcpy(buf.data(), d_ptr, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(buf.data()), N * sizeof(int32_t));
}
inline void write_uint8_array(std::ofstream& file, const uint8_t* d_ptr, int32_t N) {
    std::vector<uint8_t> buf(N); cudaMemcpy(buf.data(), d_ptr, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(buf.data()), N * sizeof(uint8_t));
}

inline void write_int64_array(std::ofstream& file, const int64_t* d_ptr, int32_t N) {
    std::vector<int64_t> buf(N);
    cudaMemcpy(buf.data(), d_ptr, static_cast<size_t>(N) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(buf.data()), static_cast<size_t>(N) * sizeof(int64_t));
}

inline void write_float3_array(std::ofstream& file, const float* x, const float* y,
                               const float* z, int32_t N) {
    write_float_array(file, x, N); write_float_array(file, y, N); write_float_array(file, z, N);
}

// =============================================================================
// write_unified_binary — writes SoA particle data in unified format
// =============================================================================
// is_checkpoint=true: includes quat, num_contacts, contact blob
// is_checkpoint=false: writes DEM or SPH fields based on config.sph.enabled
inline bool write_unified_binary(
    const std::string& path,
    const ParticleSoA& particles,
    const SimulationState& state,
    const SimConfig& config,
    bool is_checkpoint = false,
    const std::vector<ContactHistory>* contact_data = nullptr,
    int32_t max_contacts_per_particle = 0,
    int32_t total_contacts = 0,
    const double* output_force_x = nullptr,
    const double* output_force_y = nullptr,
    const double* output_force_z = nullptr)
{
    int32_t N = particles.num_particles;
    bool is_sph = config.physics == SimulationPhysics::SPH;

    const int64_t contact_capacity =
        static_cast<int64_t>(N) * max_contacts_per_particle;
    if (N <= 0 || max_contacts_per_particle < 0 || total_contacts < 0 ||
        total_contacts > contact_capacity) {
        std::cerr << "ERROR: Invalid unified binary particle/contact counts" << std::endl;
        return false;
    }
    if (is_checkpoint && total_contacts > 0 &&
        (contact_data == nullptr ||
         contact_data->size() != static_cast<size_t>(N) * max_contacts_per_particle)) {
        std::cerr << "ERROR: Checkpoint contact blob must contain N * stride records" << std::endl;
        return false;
    }

    // --- Build descriptor list ---
    std::vector<FieldDescriptor> descriptors;
    descriptors.emplace_back(FieldID::MATERIAL_ID, DType::I32, 1, 0);
    // Runtime outputs and checkpoints preserve membership even for all-normal
    // systems; fresh v3 inputs may omit it and default to zero on read.
    descriptors.emplace_back(FieldID::CLUMP_ID, DType::I32, 1, 0);
    descriptors.emplace_back(FieldID::LIFECYCLE_STATE, DType::U8, 1, 0);
    if (config.shear_box.enabled || particles.input_has_winding) {
        descriptors.emplace_back(FieldID::WINDING_X, DType::I64, 1, 0);
        descriptors.emplace_back(FieldID::WINDING_Y, DType::I64, 1, 0);
        descriptors.emplace_back(FieldID::WINDING_Z, DType::I64, 1, 0);
    }

    if (is_sph && config.sph.track_history) {
        descriptors.emplace_back(FieldID::ENERGY_COLD, DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::ENERGY_THERMAL_PEAK, DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::PRESSURE_PEAK, DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::FRACTURE_AREA, DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::FRAGMENT_LENGTH, DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::DAMPING_ENERGY, DType::F64, 1, 0);
    }

    if (is_checkpoint) {
        if (is_sph) {
            // A SPH checkpoint is self-contained: all evolving thermodynamic
            // and constitutive state needed by a resumed step is present.
            descriptors.emplace_back(FieldID::MASS,              DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::DENSITY,           DType::F64, 1, 0);
            descriptors.emplace_back(FieldID::SMOOTHING_LENGTH,  DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::PRESSURE,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::C_SOUND,           DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::ENERGY,            DType::F64, 1, 0);
            descriptors.emplace_back(FieldID::DAMAGE,            DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::FORCE,             DType::F32, 3, 0);
            descriptors.emplace_back(FieldID::CRACK_STRAIN,  DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::FLAWS_NTOT,    DType::I32, 1, 0);
            descriptors.emplace_back(FieldID::FLAWS_STRAIN_MIN, DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::FLAWS_STRAIN_MAX, DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::FLAWS_M0,      DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_XX,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_YY,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_ZZ,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_XY,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_XZ,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::S_YZ,          DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::DISTENSION,    DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::DRHODT,        DType::F64, 1, 0);
            descriptors.emplace_back(FieldID::DEDT,          DType::F64, 1, 0);
        } else {
            descriptors.emplace_back(FieldID::RADIUS,            DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::OMEGA,             DType::F64, 3, 0);
            descriptors.emplace_back(FieldID::MASS,              DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::MOMENT_OF_INERTIA, DType::F32, 1, 0);
            descriptors.emplace_back(FieldID::FORCE,             DType::F32, 3, 0);
            descriptors.emplace_back(FieldID::TORQUE,            DType::F32, 3, 0);
            descriptors.emplace_back(FieldID::QUATERNION,        DType::F64, 4, 0);
            descriptors.emplace_back(FieldID::NUM_CONTACTS,      DType::I32, 1, 0);
            descriptors.emplace_back(FieldID::CLUMP_OFFSET,      DType::F32, 3, 0);
        }
    } else if (is_sph) {
        descriptors.emplace_back(FieldID::SMOOTHING_LENGTH,  DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::DENSITY,           DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::MASS,              DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::PRESSURE,          DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::C_SOUND,           DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::ENERGY,            DType::F64, 1, 0);
        descriptors.emplace_back(FieldID::DAMAGE,            DType::F32, 1, 0);
    } else {
        // DEM output
        descriptors.emplace_back(FieldID::RADIUS,            DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::OMEGA,             DType::F64, 3, 0);
        descriptors.emplace_back(FieldID::MASS,              DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::MOMENT_OF_INERTIA, DType::F32, 1, 0);
        descriptors.emplace_back(FieldID::FORCE,             DType::F32, 3, 0);
        descriptors.emplace_back(FieldID::TORQUE,            DType::F32, 3, 0);
    }

    compute_descriptor_offsets(descriptors, static_cast<uint64_t>(N));

    // --- Open file ---
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot create unified binary file: " << path << std::endl;
        return false;
    }

    // --- Prepare header ---
    UnifiedHeader header;
    header.num_particles = static_cast<uint64_t>(N);
    header.current_step = state.current_step;
    header.current_time = state.current_time;
    header.dt = state.dt;
    header.num_descriptors = static_cast<uint16_t>(descriptors.size());

    header.physics = static_cast<uint8_t>(is_sph ? BinaryPhysics::SPH : BinaryPhysics::DEM);
    header.record_kind = static_cast<uint8_t>(is_checkpoint ? BinaryRecordKind::CHECKPOINT : BinaryRecordKind::OUTPUT);
    header.material_database_fingerprint = config.material_database.fingerprint;
    header.clump_metadata_fingerprint = config.clump.clump_metadata_fingerprint;
    header.max_contacts_per_particle = static_cast<uint32_t>(max_contacts_per_particle);
    header.total_contacts = static_cast<uint32_t>(total_contacts);

    // --- Write header ---
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // --- Write descriptor table ---
    file.write(reinterpret_cast<const char*>(descriptors.data()),
               descriptors.size() * sizeof(FieldDescriptor));

    // --- Tier 1: pos (f64), vel (f64), global_id (i32) ---
    write_double_array(file, particles.pos_x, N);
    write_double_array(file, particles.pos_y, N);
    write_double_array(file, particles.pos_z, N);
    write_double_array(file, particles.vel_x, N);
    write_double_array(file, particles.vel_y, N);
    write_double_array(file, particles.vel_z, N);
    write_int32_array(file, particles.global_id, N);

    // --- Tier 2: descriptor-driven fields ---
    for (const auto& desc : descriptors) {
        FieldID fid = static_cast<FieldID>(desc.field_id);
        switch (fid) {
            case FieldID::MATERIAL_ID:
                write_int32_array(file, particles.material_id, N);
                break;
            case FieldID::CLUMP_ID:
                write_int32_array(file, particles.clump_id, N);
                break;
            case FieldID::LIFECYCLE_STATE:
                write_uint8_array(file, particles.lifecycle_state, N);
                break;
            case FieldID::CLUMP_OFFSET:
                write_float3_array(file, particles.clump_off_x, particles.clump_off_y,
                                   particles.clump_off_z, N);
                break;
            case FieldID::WINDING_X: write_int64_array(file, particles.winding_x, N); break;
            case FieldID::WINDING_Y: write_int64_array(file, particles.winding_y, N); break;
            case FieldID::WINDING_Z: write_int64_array(file, particles.winding_z, N); break;
            case FieldID::RADIUS:
                write_float_array(file, particles.radius, N);
                break;
            case FieldID::OMEGA:
                write_double_array(file, particles.omega_x, N);
                write_double_array(file, particles.omega_y, N);
                write_double_array(file, particles.omega_z, N);
                break;
            case FieldID::MASS:
                write_float_array(file, particles.mass, N);
                break;
            case FieldID::MOMENT_OF_INERTIA:
                write_float_array(file, particles.moment_of_inertia, N);
                break;
            case FieldID::FORCE:
                write_double_as_float(file, output_force_x ? output_force_x : particles.force_x, N);
                write_double_as_float(file, output_force_y ? output_force_y : particles.force_y, N);
                write_double_as_float(file, output_force_z ? output_force_z : particles.force_z, N);
                break;
            case FieldID::TORQUE:
                write_double_as_float(file, particles.torque_x, N);
                write_double_as_float(file, particles.torque_y, N);
                write_double_as_float(file, particles.torque_z, N);
                break;
            case FieldID::QUATERNION:
                write_double_array(file, particles.quat_w, N);
                write_double_array(file, particles.quat_x, N);
                write_double_array(file, particles.quat_y, N);
                write_double_array(file, particles.quat_z, N);
                break;
            case FieldID::NUM_CONTACTS:
                write_int32_array(file, particles.num_contacts, N);
                break;
            case FieldID::SMOOTHING_LENGTH:
                write_float_array(file, particles.smoothing_length, N);
                break;
            case FieldID::DENSITY:
                write_double_array(file, particles.density, N);
                break;
            case FieldID::PRESSURE:
                write_float_array(file, particles.pressure, N);
                break;
            case FieldID::C_SOUND:
                write_float_array(file, particles.c_sound, N);
                break;
            case FieldID::ENERGY:
                write_double_array(file, particles.energy, N);
                break;
            case FieldID::DAMAGE:
                write_float_array(file, particles.damage, N);
                break;
            case FieldID::CRACK_STRAIN:
                write_float_array(file, particles.crack_strain, N);
                break;
            case FieldID::FLAWS_NTOT:
                write_int32_array(file, particles.flaws_ntot, N);
                break;
            case FieldID::FLAWS_STRAIN_MIN:
                write_float_array(file, particles.flaws_strain_min, N);
                break;
            case FieldID::FLAWS_STRAIN_MAX:
                write_float_array(file, particles.flaws_strain_max, N);
                break;
            case FieldID::FLAWS_M0:
                write_float_array(file, particles.flaws_m0, N);
                break;
            case FieldID::S_XX: write_float_array(file, particles.S_xx, N); break;
            case FieldID::S_YY: write_float_array(file, particles.S_yy, N); break;
            case FieldID::S_ZZ: write_float_array(file, particles.S_zz, N); break;
            case FieldID::S_XY: write_float_array(file, particles.S_xy, N); break;
            case FieldID::S_XZ: write_float_array(file, particles.S_xz, N); break;
            case FieldID::S_YZ: write_float_array(file, particles.S_yz, N); break;
            case FieldID::DISTENSION:
                write_float_array(file, particles.distension, N);
                break;
            case FieldID::DRHODT:
                write_double_array(file, particles.rho_rate, N);
                break;
            case FieldID::ENERGY_COLD: write_double_array(file, particles.energy_cold, N); break;
            case FieldID::ENERGY_THERMAL_PEAK: write_double_array(file, particles.energy_thermal_peak, N); break;
            case FieldID::PRESSURE_PEAK: write_double_array(file, particles.pressure_peak, N); break;
            case FieldID::FRACTURE_AREA: write_double_array(file, particles.fracture_area, N); break;
            case FieldID::FRAGMENT_LENGTH: write_double_array(file, particles.fragment_length, N); break;
            case FieldID::DAMPING_ENERGY: write_double_array(file, particles.damping_energy, N); break;
            case FieldID::DEDT:
                write_double_array(file, particles.dedt, N);
                break;
            default:
                std::cerr << "WARNING: Unknown field ID 0x"
                          << std::hex << desc.field_id << std::dec
                          << " in unified binary writer, skipping" << std::endl;
                break;
        }
    }

    // --- Contact history blob (checkpoint only) ---
    if (is_checkpoint && contact_data != nullptr && total_contacts > 0) {
        file.write(reinterpret_cast<const char*>(contact_data->data()),
                   contact_data->size() * sizeof(ContactHistory));
    }

    file.close();
    return true;
}

} // namespace cudadem

#endif // CUDADEM_BINARY_FORMAT_H

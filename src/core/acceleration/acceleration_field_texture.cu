// =============================================================================
// Acceleration Field Texture Manager for DEM Simulation
// =============================================================================
// Manages pre-calculated acceleration fields with:
// - Double buffering for time-varying fields (T_i and T_{i+1} in VRAM)
// - Async prefetching via CUDA streams (PCIe transfer overlapped with compute)
// - Support for 4 field types:
//   1. Time constant, space constant (e.g., uniform gravity)
//   2. Time constant, space varying (e.g., polyhedral gravity)
//   3. Time varying, space constant (e.g., seismic acceleration)
//   4. Time varying, space varying (e.g., asteroid shake gravity)
//
// Data Loading Strategy:
// - Distance data: Load all at startup (LoadMethod::ONCE)
// - Asteroid gravity: Load 100 frames every 100*dt_field (~1s simulation time)
// - Seismic data: Load 1000 frames every 1000*dt_field (~10s simulation time)
//
// Interpolation:
// - Spatial: Linear or Cubic (configurable per field via YAML)
// - Temporal: Linear interpolation between bracketing frames
//
// Reference: Standard N-body dynamics with pre-calculated fields
// =============================================================================

#include "acceleration.h"
#include "../particle_model/particle.h"
#include "../math/linear_interpolation.cuh"
#include "../math/cubic_interpolation.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <cstring>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cudadem {

// =============================================================================
// Acceleration Field Type Classification
// =============================================================================

enum class AccelFieldType : uint8_t {
    CONSTANT_CONSTANT = 0,   // Time constant, space constant
    CONSTANT_VARYING = 1,    // Time constant, space varying
    VARYING_CONSTANT = 2,    // Time varying, space constant
    VARYING_VARYING = 3      // Time varying, space varying
};

// Determine field type from configuration
inline AccelFieldType classify_field(const AccelFieldConfig& config) {
    if (!config.time_varying && !config.space_varying) {
        return AccelFieldType::CONSTANT_CONSTANT;
    } else if (!config.time_varying && config.space_varying) {
        return AccelFieldType::CONSTANT_VARYING;
    } else if (config.time_varying && !config.space_varying) {
        return AccelFieldType::VARYING_CONSTANT;
    } else {
        return AccelFieldType::VARYING_VARYING;
    }
}

// =============================================================================
// Field Data Binary File Format
// =============================================================================
// Header structure for acceleration field binary files

struct FieldBinaryHeader {
    char magic[8];                   // "ACCELFD\0"
    uint32_t version;                // File format version
    uint32_t num_frames;             // Number of time frames (1 for static)

    // Grid dimensions (0 for space-constant)
    int32_t dim_x, dim_y, dim_z;

    // Grid origin and spacing
    float origin_x, origin_y, origin_z;
    float spacing_x, spacing_y, spacing_z;

    // Time info
    float time_start;
    float time_interval;

    // Flags
    uint8_t is_time_varying;
    uint8_t is_space_varying;
    uint8_t interp_method;           // 0=none, 1=linear, 2=cubic
    uint8_t reserved[5];

    FieldBinaryHeader()
        : version(1), num_frames(1),
          dim_x(0), dim_y(0), dim_z(0),
          origin_x(0), origin_y(0), origin_z(0),
          spacing_x(1), spacing_y(1), spacing_z(1),
          time_start(0), time_interval(0.01f),
          is_time_varying(0), is_space_varying(0), interp_method(1) {
        memcpy(magic, "ACCELFD", 8);
        memset(reserved, 0, sizeof(reserved));
    }
};

// ACCELFD is an on-disk little-endian protocol.  Do not deserialize it by
// reading a native C++ struct: that would make padding and byte order part of
// the scientific-file contract.  Version 2 spatial payloads are time-major,
// component-planar [frame][component][z][y][x] float32.  The x index is
// contiguous and therefore matches CUDA's three component texture uploads.
constexpr size_t ACCELFD_HEADER_SIZE = 68;
constexpr uint32_t ACCELFD_V2 = 2;

static uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static int32_t read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(read_u32_le(p));
}

static float read_f32_le(const uint8_t* p) {
    const uint32_t bits = read_u32_le(p);
    float value;
    static_assert(sizeof(value) == sizeof(bits), "ACCELFD requires IEEE float32");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static FieldBinaryHeader read_accelfd_header(std::ifstream& file, const char* path) {
    static_assert(sizeof(FieldBinaryHeader) == ACCELFD_HEADER_SIZE,
                  "ACCELFD header layout must remain 68 bytes");
    std::array<uint8_t, ACCELFD_HEADER_SIZE> raw{};
    file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (file.gcount() != static_cast<std::streamsize>(raw.size())) {
        throw std::runtime_error(std::string("ACCELFD header is truncated: ") + path);
    }
    if (std::memcmp(raw.data(), "ACCELFD\0", 8) != 0) {
        throw std::runtime_error(std::string("invalid ACCELFD magic: ") + path);
    }
    FieldBinaryHeader header;
    header.version = read_u32_le(raw.data() + 8);
    header.num_frames = read_u32_le(raw.data() + 12);
    header.dim_x = read_i32_le(raw.data() + 16);
    header.dim_y = read_i32_le(raw.data() + 20);
    header.dim_z = read_i32_le(raw.data() + 24);
    header.origin_x = read_f32_le(raw.data() + 28);
    header.origin_y = read_f32_le(raw.data() + 32);
    header.origin_z = read_f32_le(raw.data() + 36);
    header.spacing_x = read_f32_le(raw.data() + 40);
    header.spacing_y = read_f32_le(raw.data() + 44);
    header.spacing_z = read_f32_le(raw.data() + 48);
    header.time_start = read_f32_le(raw.data() + 52);
    header.time_interval = read_f32_le(raw.data() + 56);
    header.is_time_varying = raw[60];
    header.is_space_varying = raw[61];
    header.interp_method = raw[62];
    std::memcpy(header.reserved, raw.data() + 63, sizeof(header.reserved));
    return header;
}

static void require_accelfd_static_header(
    const FieldBinaryHeader& header, bool expect_space_varying, const char* path
) {
    for (uint8_t value : header.reserved) {
        if (value != 0) {
            throw std::runtime_error(std::string("ACCELFD reserved header bytes must be zero: ") + path);
        }
    }
    const bool v1_uniform = header.version == 1 && !expect_space_varying;
    if (header.version != ACCELFD_V2 && !v1_uniform) {
        throw std::runtime_error(expect_space_varying
            ? std::string("ACCELFD v1 spatial fields are ambiguous; regenerate as v2: ") + path
            : std::string("unsupported ACCELFD version: ") + path);
    }
    if (header.num_frames != 1 || header.is_time_varying != 0) {
        throw std::runtime_error(std::string("only static ACCELFD fields are supported: ") + path);
    }
    const bool legacy_uniform_no_interp = v1_uniform && header.interp_method == 0;
    if (header.is_space_varying != static_cast<uint8_t>(expect_space_varying) ||
        (header.interp_method != 1 && !legacy_uniform_no_interp)) {
        throw std::runtime_error(std::string("ACCELFD type/interpolation does not match static linear configuration: ") + path);
    }
    const float metadata[] = {header.origin_x, header.origin_y, header.origin_z,
                              header.spacing_x, header.spacing_y, header.spacing_z,
                              header.time_start, header.time_interval};
    for (float value : metadata) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("ACCELFD header contains non-finite metadata: ") + path);
        }
    }
    if (expect_space_varying) {
        if (header.dim_x < 2 || header.dim_y < 2 || header.dim_z < 2 ||
            header.spacing_x <= 0.0f || header.spacing_y <= 0.0f || header.spacing_z <= 0.0f) {
            throw std::runtime_error(std::string("ACCELFD spatial grid needs dimensions >=2 and positive spacing: ") + path);
        }
    } else if (header.dim_x != 0 || header.dim_y != 0 || header.dim_z != 0) {
        // Legacy v1 uniform files produced by the hopper generator carry an
        // unused 1x1x1 grid.  Retain only that known compatibility case.
        if (header.version != 1 || header.dim_x != 1 || header.dim_y != 1 || header.dim_z != 1) {
            throw std::runtime_error(std::string("ACCELFD uniform field must have zero grid dimensions: ") + path);
        }
    }
}

static void require_accelfd_exact_length(std::ifstream& file, size_t payload_bytes, const char* path) {
    file.seekg(0, std::ios::end);
    const std::streamoff length = file.tellg();
    const std::streamoff expected = static_cast<std::streamoff>(ACCELFD_HEADER_SIZE + payload_bytes);
    if (length != expected) {
        throw std::runtime_error(std::string("ACCELFD exact length mismatch for ") + path +
                                 ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(length));
    }
    file.seekg(ACCELFD_HEADER_SIZE, std::ios::beg);
}

// =============================================================================
// Prefetch Request Structure
// =============================================================================

struct PrefetchRequest {
    int32_t field_id;                // Which field
    int32_t start_frame;             // First frame to load
    int32_t num_frames;              // Number of frames to load
    bool in_progress;                // Is transfer ongoing
    cudaEvent_t completion_event;    // Event to check completion

    PrefetchRequest()
        : field_id(-1), start_frame(0), num_frames(0),
          in_progress(false), completion_event(nullptr) {}
};

// =============================================================================
// Time-Varying Uniform Field Buffer (Space Constant)
// =============================================================================
// For seismic acceleration: time-varying but same everywhere

struct UniformFieldBuffer {
    // Host staging buffer for batch loading
    std::vector<float> h_buffer_x;
    std::vector<float> h_buffer_y;
    std::vector<float> h_buffer_z;
    std::vector<float> h_times;

    // Device pinned memory for async transfer
    float* d_pinned_x;
    float* d_pinned_y;
    float* d_pinned_z;
    float* d_pinned_times;

    // Current loaded range
    int32_t loaded_start_frame;
    int32_t loaded_num_frames;

    // File info
    int32_t total_frames;
    float time_start;
    float time_interval;

    // Batch size (frames to load at once)
    int32_t batch_size;

    UniformFieldBuffer()
        : d_pinned_x(nullptr), d_pinned_y(nullptr),
          d_pinned_z(nullptr), d_pinned_times(nullptr),
          loaded_start_frame(-1), loaded_num_frames(0),
          total_frames(0), time_start(0), time_interval(0.01f),
          batch_size(1000) {}

    void allocate(int32_t batch) {
        batch_size = batch;

        h_buffer_x.resize(batch);
        h_buffer_y.resize(batch);
        h_buffer_z.resize(batch);
        h_times.resize(batch);

        // Allocate pinned memory for async transfer
        size_t size = batch * sizeof(float);
        cudaMallocHost(&d_pinned_x, size);
        cudaMallocHost(&d_pinned_y, size);
        cudaMallocHost(&d_pinned_z, size);
        cudaMallocHost(&d_pinned_times, size);
    }

    void free() {
        if (d_pinned_x) cudaFreeHost(d_pinned_x);
        if (d_pinned_y) cudaFreeHost(d_pinned_y);
        if (d_pinned_z) cudaFreeHost(d_pinned_z);
        if (d_pinned_times) cudaFreeHost(d_pinned_times);
        d_pinned_x = nullptr;
        d_pinned_y = nullptr;
        d_pinned_z = nullptr;
        d_pinned_times = nullptr;
        h_buffer_x.clear();
        h_buffer_y.clear();
        h_buffer_z.clear();
        h_times.clear();
    }

    // Interpolate acceleration at given time
    __host__ void interpolate_at_time(
        float query_time,
        float& ax, float& ay, float& az
    ) const {
        if (loaded_num_frames <= 0) {
            ax = ay = az = 0.0f;
            return;
        }

        // Find bracketing frames
        float rel_time = (query_time - time_start) / time_interval;
        int frame_lo = static_cast<int>(rel_time) - loaded_start_frame;
        int frame_hi = frame_lo + 1;

        // Clamp to loaded range
        frame_lo = std::max(0, std::min(frame_lo, loaded_num_frames - 1));
        frame_hi = std::max(0, std::min(frame_hi, loaded_num_frames - 1));

        if (frame_lo == frame_hi) {
            ax = h_buffer_x[frame_lo];
            ay = h_buffer_y[frame_lo];
            az = h_buffer_z[frame_lo];
            return;
        }

        // Linear interpolation
        float t0 = time_start + (loaded_start_frame + frame_lo) * time_interval;
        float t1 = time_start + (loaded_start_frame + frame_hi) * time_interval;
        float alpha = (query_time - t0) / (t1 - t0);
        alpha = std::max(0.0f, std::min(1.0f, alpha));

        ax = h_buffer_x[frame_lo] + alpha * (h_buffer_x[frame_hi] - h_buffer_x[frame_lo]);
        ay = h_buffer_y[frame_lo] + alpha * (h_buffer_y[frame_hi] - h_buffer_y[frame_lo]);
        az = h_buffer_z[frame_lo] + alpha * (h_buffer_z[frame_hi] - h_buffer_z[frame_lo]);
    }
};

// =============================================================================
// GPU Constant Memory for Active Fields
// =============================================================================

// Maximum number of fields
constexpr int MAX_ACTIVE_FIELDS = 4;

// Constant field (time constant, space constant)
struct ConstantFieldGPU {
    float accel_x, accel_y, accel_z;
    bool enabled;
};

// Time-varying uniform field parameters (for kernel access)
struct UniformTimeVaryingGPU {
    float accel_x, accel_y, accel_z;  // Current interpolated value
    bool enabled;
};

__constant__ ConstantFieldGPU dev_constant_fields[MAX_ACTIVE_FIELDS];
__constant__ UniformTimeVaryingGPU dev_uniform_time_varying[MAX_ACTIVE_FIELDS];
__constant__ InterpolatorParams3D dev_space_varying_params[MAX_ACTIVE_FIELDS];
__constant__ InterpolatorParams4D dev_spacetime_varying_params[MAX_ACTIVE_FIELDS];

// Field enable flags
__constant__ uint8_t dev_num_constant_fields;
__constant__ uint8_t dev_num_uniform_time_varying;
__constant__ uint8_t dev_num_space_varying;
__constant__ uint8_t dev_num_spacetime_varying;

// Texture address mode is clamp for hardware interpolation, but using a
// clamped acceleration outside an approved scientific grid is invalid.  Trap
// the CUDA context rather than silently applying boundary acceleration.
__device__ __forceinline__ void fail_accelfd_outside_grid() {
    asm volatile("trap;");
}

// =============================================================================
// GPU Kernels for Applying Acceleration Fields
// =============================================================================

// Type 1: Constant field (time constant, space constant)
__global__ void apply_constant_field_kernel(
    // Particle positions (unused but for consistency)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle mass
    const float* __restrict__ mass,
    // Output forces
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float m = mass[idx];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Sum all constant fields
    for (int f = 0; f < dev_num_constant_fields; ++f) {
        if (dev_constant_fields[f].enabled) {
            fx += m * dev_constant_fields[f].accel_x;
            fy += m * dev_constant_fields[f].accel_y;
            fz += m * dev_constant_fields[f].accel_z;
        }
    }

    atomic_add_force(force_x, force_y, force_z, fx, fy, fz, idx);
}

// Type 2: Space-varying, time-constant field
template<bool UseCubicInterp>
__global__ void apply_space_varying_field_kernel(
    // Particle positions
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle mass
    const float* __restrict__ mass,
    // Output forces
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    // Field index
    int field_idx,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);
    float m = mass[idx];

    // Get field parameters
    InterpolatorParams3D params = dev_space_varying_params[field_idx];

    const float max_x = params.origin_x + (params.dim_x - 1) / params.inv_spacing_x;
    const float max_y = params.origin_y + (params.dim_y - 1) / params.inv_spacing_y;
    const float max_z = params.origin_z + (params.dim_z - 1) / params.inv_spacing_z;
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz) ||
        px < params.origin_x || px > max_x ||
        py < params.origin_y || py > max_y ||
        pz < params.origin_z || pz > max_z) {
        fail_accelfd_outside_grid();
        return;
    }

    // Interpolate acceleration
    float3 accel;
    if constexpr (UseCubicInterp) {
        // Cast to cubic params (same layout for 3D)
        CubicInterpolatorParams3D cubic_params;
        cubic_params.tex_x = params.tex_x;
        cubic_params.tex_y = params.tex_y;
        cubic_params.tex_z = params.tex_z;
        cubic_params.origin_x = params.origin_x;
        cubic_params.origin_y = params.origin_y;
        cubic_params.origin_z = params.origin_z;
        cubic_params.inv_spacing_x = params.inv_spacing_x;
        cubic_params.inv_spacing_y = params.inv_spacing_y;
        cubic_params.inv_spacing_z = params.inv_spacing_z;
        cubic_params.dim_x = params.dim_x;
        cubic_params.dim_y = params.dim_y;
        cubic_params.dim_z = params.dim_z;
        accel = interp_cubic_vector_field_3d(cubic_params, px, py, pz);
    } else {
        accel = interp_vector_field_3d(params, px, py, pz);
    }

    atomic_add_force(force_x, force_y, force_z, m * accel.x, m * accel.y, m * accel.z, idx);
}

// Type 3: Time-varying, space-constant field
__global__ void apply_uniform_time_varying_kernel(
    const float* __restrict__ mass,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float m = mass[idx];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Sum all time-varying uniform fields
    for (int f = 0; f < dev_num_uniform_time_varying; ++f) {
        if (dev_uniform_time_varying[f].enabled) {
            fx += m * dev_uniform_time_varying[f].accel_x;
            fy += m * dev_uniform_time_varying[f].accel_y;
            fz += m * dev_uniform_time_varying[f].accel_z;
        }
    }

    atomic_add_force(force_x, force_y, force_z, fx, fy, fz, idx);
}

// Type 4: Time-varying, space-varying field (full 4D interpolation)
template<bool UseCubicInterp>
__global__ void apply_spacetime_varying_field_kernel(
    // Particle positions
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle mass
    const float* __restrict__ mass,
    // Output forces
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    // Field index
    int field_idx,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);
    float m = mass[idx];

    // Get field parameters
    InterpolatorParams4D params = dev_spacetime_varying_params[field_idx];

    // Interpolate acceleration with time
    float3 accel;
    if constexpr (UseCubicInterp) {
        CubicInterpolatorParams4D cubic_params;
        cubic_params.tex_x_0 = params.tex_x_0;
        cubic_params.tex_x_1 = params.tex_x_1;
        cubic_params.tex_y_0 = params.tex_y_0;
        cubic_params.tex_y_1 = params.tex_y_1;
        cubic_params.tex_z_0 = params.tex_z_0;
        cubic_params.tex_z_1 = params.tex_z_1;
        cubic_params.origin_x = params.origin_x;
        cubic_params.origin_y = params.origin_y;
        cubic_params.origin_z = params.origin_z;
        cubic_params.inv_spacing_x = params.inv_spacing_x;
        cubic_params.inv_spacing_y = params.inv_spacing_y;
        cubic_params.inv_spacing_z = params.inv_spacing_z;
        cubic_params.dim_x = params.dim_x;
        cubic_params.dim_y = params.dim_y;
        cubic_params.dim_z = params.dim_z;
        cubic_params.time_alpha = params.time_alpha;
        accel = interp_cubic_vector_field_4d(cubic_params, px, py, pz);
    } else {
        accel = interp_vector_field_4d(params, px, py, pz);
    }

    atomic_add_force(force_x, force_y, force_z, m * accel.x, m * accel.y, m * accel.z, idx);
}

// =============================================================================
// Acceleration Field Manager
// =============================================================================

// Device buffer for gravity queries
static float* d_query_result = nullptr;

__global__ void query_constant_field_kernel(float* res) {
    if (threadIdx.x != 0) return;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int f = 0; f < dev_num_constant_fields; ++f) {
        if (dev_constant_fields[f].enabled) {
            gx += dev_constant_fields[f].accel_x;
            gy += dev_constant_fields[f].accel_y;
            gz += dev_constant_fields[f].accel_z;
        }
    }
    atomicAdd(&res[0], gx);
    atomicAdd(&res[1], gy);
    atomicAdd(&res[2], gz);
}

template<bool UseCubicInterp>
__global__ void query_space_varying_field_kernel(float px, float py, float pz, int field_idx, float* res) {
    if (threadIdx.x != 0) return;
    InterpolatorParams3D params = dev_space_varying_params[field_idx];
    float3 accel;
    if constexpr (UseCubicInterp) {
        CubicInterpolatorParams3D cubic_params;
        cubic_params.tex_x = params.tex_x;
        cubic_params.tex_y = params.tex_y;
        cubic_params.tex_z = params.tex_z;
        cubic_params.origin_x = params.origin_x;
        cubic_params.origin_y = params.origin_y;
        cubic_params.origin_z = params.origin_z;
        cubic_params.inv_spacing_x = params.inv_spacing_x;
        cubic_params.inv_spacing_y = params.inv_spacing_y;
        cubic_params.inv_spacing_z = params.inv_spacing_z;
        cubic_params.dim_x = params.dim_x;
        cubic_params.dim_y = params.dim_y;
        cubic_params.dim_z = params.dim_z;
        accel = interp_cubic_vector_field_3d(cubic_params, px, py, pz);
    } else {
        accel = interp_vector_field_3d(params, px, py, pz);
    }
    atomicAdd(&res[0], accel.x);
    atomicAdd(&res[1], accel.y);
    atomicAdd(&res[2], accel.z);
}

__global__ void query_uniform_time_varying_kernel(float* res) {
    if (threadIdx.x != 0) return;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int f = 0; f < dev_num_uniform_time_varying; ++f) {
        if (dev_uniform_time_varying[f].enabled) {
            gx += dev_uniform_time_varying[f].accel_x;
            gy += dev_uniform_time_varying[f].accel_y;
            gz += dev_uniform_time_varying[f].accel_z;
        }
    }
    atomicAdd(&res[0], gx);
    atomicAdd(&res[1], gy);
    atomicAdd(&res[2], gz);
}

template<bool UseCubicInterp>
__global__ void query_spacetime_varying_field_kernel(float px, float py, float pz, int field_idx, float* res) {
    if (threadIdx.x != 0) return;
    InterpolatorParams4D params = dev_spacetime_varying_params[field_idx];
    float3 accel;
    if constexpr (UseCubicInterp) {
        CubicInterpolatorParams4D cubic_params;
        cubic_params.tex_x_0 = params.tex_x_0;
        cubic_params.tex_x_1 = params.tex_x_1;
        cubic_params.tex_y_0 = params.tex_y_0;
        cubic_params.tex_y_1 = params.tex_y_1;
        cubic_params.tex_z_0 = params.tex_z_0;
        cubic_params.tex_z_1 = params.tex_z_1;
        cubic_params.origin_x = params.origin_x;
        cubic_params.origin_y = params.origin_y;
        cubic_params.origin_z = params.origin_z;
        cubic_params.inv_spacing_x = params.inv_spacing_x;
        cubic_params.inv_spacing_y = params.inv_spacing_y;
        cubic_params.inv_spacing_z = params.inv_spacing_z;
        cubic_params.dim_x = params.dim_x;
        cubic_params.dim_y = params.dim_y;
        cubic_params.dim_z = params.dim_z;
        cubic_params.time_alpha = params.time_alpha;
        accel = interp_cubic_vector_field_4d(cubic_params, px, py, pz);
    } else {
        accel = interp_vector_field_4d(params, px, py, pz);
    }
    atomicAdd(&res[0], accel.x);
    atomicAdd(&res[1], accel.y);
    atomicAdd(&res[2], accel.z);
}

class AccelerationFieldManager {
public:
    // Field storage by type
    std::vector<ConstantFieldGPU> constant_fields;
    std::vector<UniformFieldBuffer> uniform_time_varying_fields;
    std::vector<VectorField3D> space_varying_fields;
    std::vector<VectorField4D> spacetime_varying_fields;

    // Cubic variants for space-varying fields
    std::vector<CubicVectorField3D> space_varying_cubic;
    std::vector<CubicVectorField4D> spacetime_varying_cubic;

    // Interpolation method per field
    std::vector<InterpolationMethod> space_varying_interp;
    std::vector<InterpolationMethod> spacetime_varying_interp;

    // Field configurations
    std::vector<AccelFieldConfig> field_configs;

    // CUDA streams for prefetching
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    // Prefetch state
    std::vector<PrefetchRequest> pending_prefetches;

    // File handles for streaming
    std::vector<std::ifstream> field_files;
    std::vector<FieldBinaryHeader> field_headers;

    // Initialization state
    bool initialized;

    AccelerationFieldManager()
        : compute_stream(0), transfer_stream(0), initialized(false) {}

    ~AccelerationFieldManager() {
        shutdown();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    void initialize(const SimConfig& config) {
        if (initialized) {
            std::cerr << "WARNING: AccelerationFieldManager already initialized" << std::endl;
            shutdown();
        }

        // Create CUDA streams
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);

        // Process each configured field
        for (int i = 0; i < config.acceleration.num_accel_fields; ++i) {
            const AccelFieldConfig& fc = config.acceleration.fields[i];
            if (!fc.enabled) continue;

            if (fc.time_varying) {
                throw std::runtime_error(std::string("time-varying ACCELFD is not implemented; reject field '") + fc.name + "'");
            }
            if (fc.space_varying && fc.space_interp != InterpolationMethod::LINEAR) {
                throw std::runtime_error(std::string("only linear ACCELFD interpolation is supported; reject field '") + fc.name + "'");
            }

            field_configs.push_back(fc);

            AccelFieldType type = classify_field(fc);

            switch (type) {
                case AccelFieldType::CONSTANT_CONSTANT:
                    initialize_constant_field(fc);
                    break;
                case AccelFieldType::CONSTANT_VARYING:
                    initialize_space_varying_field(fc);
                    break;
                case AccelFieldType::VARYING_CONSTANT:
                    initialize_uniform_time_varying_field(fc);
                    break;
                case AccelFieldType::VARYING_VARYING:
                    initialize_spacetime_varying_field(fc);
                    break;
            }
        }

        // Update GPU constant memory counts
        uint8_t n_const = static_cast<uint8_t>(constant_fields.size());
        uint8_t n_unif = static_cast<uint8_t>(uniform_time_varying_fields.size());
        uint8_t n_space = static_cast<uint8_t>(space_varying_fields.size() +
                                                space_varying_cubic.size());
        uint8_t n_spacetime = static_cast<uint8_t>(spacetime_varying_fields.size() +
                                                    spacetime_varying_cubic.size());

        cudaMemcpyToSymbol(dev_num_constant_fields, &n_const, sizeof(uint8_t));
        cudaMemcpyToSymbol(dev_num_uniform_time_varying, &n_unif, sizeof(uint8_t));
        cudaMemcpyToSymbol(dev_num_space_varying, &n_space, sizeof(uint8_t));
        cudaMemcpyToSymbol(dev_num_spacetime_varying, &n_spacetime, sizeof(uint8_t));

        // Copy constant fields to GPU
        if (!constant_fields.empty()) {
            cudaMemcpyToSymbol(dev_constant_fields, constant_fields.data(),
                               constant_fields.size() * sizeof(ConstantFieldGPU));
        }

        initialized = true;

        print_field_summary();
    }

    // =========================================================================
    // Field Initialization by Type
    // =========================================================================

    void initialize_constant_field(const AccelFieldConfig& fc) {
        // For constant fields, acceleration is read from binary file with header
        // The binary file format uses ACCELFD header (68 bytes) followed by data

        ConstantFieldGPU field;
        field.enabled = true;

        // Try to read from binary file
        if (fc.data_path[0] != '\0') {
            std::ifstream file(fc.data_path, std::ios::binary);
            if (file.is_open()) {
                const FieldBinaryHeader header = read_accelfd_header(file, fc.data_path);
                require_accelfd_static_header(header, false, fc.data_path);
                require_accelfd_exact_length(file, 3 * sizeof(float), fc.data_path);

                // Read acceleration data (3 floats)
                float vals[3];
                file.read(reinterpret_cast<char*>(vals), 3 * sizeof(float));
                if (!file || !std::isfinite(vals[0]) || !std::isfinite(vals[1]) || !std::isfinite(vals[2])) {
                    throw std::runtime_error(std::string("invalid ACCELFD uniform payload: ") + fc.data_path);
                }
                field.accel_x = vals[0];
                field.accel_y = vals[1];
                field.accel_z = vals[2];
                file.close();
            } else {
                throw std::runtime_error(std::string("cannot open ACCELFD uniform field: ") + fc.data_path);
            }
        } else {
            throw std::runtime_error(std::string("ACCELFD uniform field requires data_path: ") + fc.name);
        }

        constant_fields.push_back(field);

        std::cout << "Initialized constant field '" << fc.name
                  << "': (" << field.accel_x << ", " << field.accel_y
                  << ", " << field.accel_z << ") m/s²" << std::endl;
    }

    void initialize_space_varying_field(const AccelFieldConfig& fc) {
        // Load 3D field from binary file
        std::ifstream file(fc.data_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("cannot open ACCELFD spatial field: ") + fc.data_path);
        }

        const FieldBinaryHeader header = read_accelfd_header(file, fc.data_path);
        require_accelfd_static_header(header, true, fc.data_path);

        const size_t dim_x = static_cast<size_t>(header.dim_x);
        const size_t dim_y = static_cast<size_t>(header.dim_y);
        const size_t dim_z = static_cast<size_t>(header.dim_z);
        if (dim_x > std::numeric_limits<size_t>::max() / dim_y ||
            dim_x * dim_y > std::numeric_limits<size_t>::max() / dim_z) {
            throw std::runtime_error(std::string("ACCELFD grid dimensions overflow: ") + fc.data_path);
        }
        const size_t volume_size = dim_x * dim_y * dim_z;
        if (volume_size > std::numeric_limits<size_t>::max() / (3 * sizeof(float))) {
            throw std::runtime_error(std::string("ACCELFD grid is too large: ") + fc.data_path);
        }
        require_accelfd_exact_length(file, volume_size * 3 * sizeof(float), fc.data_path);
        std::vector<float> data_x(volume_size), data_y(volume_size), data_z(volume_size);
        file.read(reinterpret_cast<char*>(data_x.data()), volume_size * sizeof(float));
        file.read(reinterpret_cast<char*>(data_y.data()), volume_size * sizeof(float));
        file.read(reinterpret_cast<char*>(data_z.data()), volume_size * sizeof(float));
        if (!file) {
            throw std::runtime_error(std::string("truncated ACCELFD spatial payload: ") + fc.data_path);
        }
        for (size_t i = 0; i < volume_size; ++i) {
            if (!std::isfinite(data_x[i]) || !std::isfinite(data_y[i]) || !std::isfinite(data_z[i])) {
                throw std::runtime_error(std::string("non-finite ACCELFD spatial payload: ") + fc.data_path);
            }
        }
        file.close();

        // Initialize appropriate interpolator
        if (fc.space_interp == InterpolationMethod::CUBIC) {
            CubicVectorField3D field;
            field.initialize(
                data_x.data(), data_y.data(), data_z.data(),
                header.dim_x, header.dim_y, header.dim_z,
                header.origin_x, header.origin_y, header.origin_z,
                header.spacing_x, header.spacing_y, header.spacing_z
            );
            space_varying_cubic.push_back(std::move(field));
            space_varying_interp.push_back(InterpolationMethod::CUBIC);
        } else {
            VectorField3D field;
            field.initialize(
                data_x.data(), data_y.data(), data_z.data(),
                header.dim_x, header.dim_y, header.dim_z,
                header.origin_x, header.origin_y, header.origin_z,
                header.spacing_x, header.spacing_y, header.spacing_z
            );
            space_varying_fields.push_back(std::move(field));
            space_varying_interp.push_back(InterpolationMethod::LINEAR);
        }

        std::cout << "Initialized space-varying field '" << fc.name
                  << "': " << header.dim_x << "x" << header.dim_y << "x" << header.dim_z
                  << " (" << (fc.space_interp == InterpolationMethod::CUBIC ? "cubic" : "linear")
                  << ")" << std::endl;
    }

    void initialize_uniform_time_varying_field(const AccelFieldConfig& fc) {
        // Open file for streaming
        std::ifstream file(fc.data_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open time-varying field: "
                      << fc.data_path << std::endl;
            return;
        }

        // Read header
        FieldBinaryHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        // Create buffer
        UniformFieldBuffer buffer;
        buffer.total_frames = header.num_frames;
        buffer.time_start = header.time_start;
        buffer.time_interval = header.time_interval;
        buffer.batch_size = fc.batch_size;
        buffer.allocate(fc.batch_size);

        // Store file handle and header for streaming
        field_files.push_back(std::move(file));
        field_headers.push_back(header);

        uniform_time_varying_fields.push_back(std::move(buffer));

        std::cout << "Initialized uniform time-varying field '" << fc.name
                  << "': " << header.num_frames << " frames, batch=" << fc.batch_size
                  << std::endl;
    }

    void initialize_spacetime_varying_field(const AccelFieldConfig& fc) {
        // Open file for streaming
        std::ifstream file(fc.data_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open spacetime-varying field: "
                      << fc.data_path << std::endl;
            return;
        }

        // Read header
        FieldBinaryHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        field_files.push_back(std::move(file));
        field_headers.push_back(header);

        // Initialize 4D interpolator (double-buffered)
        if (fc.space_interp == InterpolationMethod::CUBIC) {
            CubicVectorField4D field;
            field.initialize(
                header.dim_x, header.dim_y, header.dim_z,
                header.origin_x, header.origin_y, header.origin_z,
                header.spacing_x, header.spacing_y, header.spacing_z,
                header.time_start, header.time_interval, header.num_frames,
                true  // extrapolate
            );
            spacetime_varying_cubic.push_back(std::move(field));
            spacetime_varying_interp.push_back(InterpolationMethod::CUBIC);
        } else {
            VectorField4D field;
            field.initialize(
                header.dim_x, header.dim_y, header.dim_z,
                header.origin_x, header.origin_y, header.origin_z,
                header.spacing_x, header.spacing_y, header.spacing_z,
                header.time_start, header.time_interval, header.num_frames
            );
            spacetime_varying_fields.push_back(std::move(field));
            spacetime_varying_interp.push_back(InterpolationMethod::LINEAR);
        }

        std::cout << "Initialized spacetime-varying field '" << fc.name
                  << "': " << header.dim_x << "x" << header.dim_y << "x" << header.dim_z
                  << " x " << header.num_frames << " frames" << std::endl;
    }

    // =========================================================================
    // Prefetching Logic
    // =========================================================================

    // Check if prefetch is needed and initiate async transfer
    void update_prefetch(double current_time, double dt) {
        // Check each time-varying field
        for (size_t i = 0; i < uniform_time_varying_fields.size(); ++i) {
            update_uniform_field_prefetch(i, current_time);
        }

        for (size_t i = 0; i < spacetime_varying_fields.size(); ++i) {
            update_spacetime_field_prefetch(i, current_time, false);
        }

        for (size_t i = 0; i < spacetime_varying_cubic.size(); ++i) {
            update_spacetime_field_prefetch(i, current_time, true);
        }
    }

    void update_uniform_field_prefetch(size_t field_idx, double current_time) {
        UniformFieldBuffer& buf = uniform_time_varying_fields[field_idx];

        // Calculate which batch we need
        float rel_time = (static_cast<float>(current_time) - buf.time_start) / buf.time_interval;
        int needed_frame = static_cast<int>(rel_time);

        // Check if we need to load a new batch
        bool need_load = false;
        if (buf.loaded_start_frame < 0) {
            need_load = true;  // First load
        } else {
            // Check if approaching end of loaded range
            int frames_ahead = (buf.loaded_start_frame + buf.loaded_num_frames) - needed_frame;
            if (frames_ahead < buf.batch_size / 4) {
                need_load = true;  // Load next batch
            }
        }

        if (need_load) {
            int start_frame = std::max(0, needed_frame - buf.batch_size / 4);
            load_uniform_field_batch(field_idx, start_frame);
        }
    }

    void load_uniform_field_batch(size_t field_idx, int start_frame) {
        UniformFieldBuffer& buf = uniform_time_varying_fields[field_idx];

        // Clamp to valid range
        start_frame = std::max(0, std::min(start_frame, buf.total_frames - buf.batch_size));
        int num_to_load = std::min(buf.batch_size, buf.total_frames - start_frame);

        // Find corresponding file
        size_t file_idx = field_idx;  // Assuming 1:1 mapping
        if (file_idx >= field_files.size()) return;

        std::ifstream& file = field_files[file_idx];
        if (!file.is_open()) return;

        // Seek to data position
        size_t data_offset = sizeof(FieldBinaryHeader) + start_frame * 3 * sizeof(float);
        file.seekg(data_offset);

        // Read data (x, y, z interleaved per frame)
        for (int f = 0; f < num_to_load; ++f) {
            float vals[3];
            file.read(reinterpret_cast<char*>(vals), 3 * sizeof(float));
            buf.h_buffer_x[f] = vals[0];
            buf.h_buffer_y[f] = vals[1];
            buf.h_buffer_z[f] = vals[2];
            buf.h_times[f] = buf.time_start + (start_frame + f) * buf.time_interval;
        }

        buf.loaded_start_frame = start_frame;
        buf.loaded_num_frames = num_to_load;
    }

    void update_spacetime_field_prefetch(size_t field_idx, double current_time, bool is_cubic) {
        // For 4D fields, use the built-in double-buffer management in VectorField4D
        // The prepare_for_time method handles frame loading

        // Get the appropriate field
        if (is_cubic) {
            if (field_idx >= spacetime_varying_cubic.size()) return;
            // CubicVectorField4D doesn't have prepare_for_time, handle manually
        } else {
            if (field_idx >= spacetime_varying_fields.size()) return;
            // VectorField4D's LinearInterpolator4D has prepare_for_time
        }

        // For now, manual frame loading is deferred to compute_all
        // The actual implementation would use the transfer stream for async loading
    }

    // =========================================================================
    // Compute All Field Accelerations
    // =========================================================================

    void compute_all(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream
    ) {
        if (!initialized) return;

        int32_t num_particles = state.particles.num_particles;
        if (num_particles == 0) return;

        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        // Update prefetch status
        update_prefetch(current_time, state.dt);

        // Type 1: Constant fields
        if (!constant_fields.empty()) {
            apply_constant_field_kernel<<<num_blocks, block_size, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                num_particles
            );
        }

        // Type 3: Time-varying, space-constant (uniform) fields
        if (!uniform_time_varying_fields.empty()) {
            // Update GPU constant memory with interpolated values
            std::vector<UniformTimeVaryingGPU> gpu_uniforms(uniform_time_varying_fields.size());
            for (size_t i = 0; i < uniform_time_varying_fields.size(); ++i) {
                uniform_time_varying_fields[i].interpolate_at_time(
                    static_cast<float>(current_time),
                    gpu_uniforms[i].accel_x,
                    gpu_uniforms[i].accel_y,
                    gpu_uniforms[i].accel_z
                );
                gpu_uniforms[i].enabled = true;
            }

            cudaMemcpyToSymbolAsync(dev_uniform_time_varying, gpu_uniforms.data(),
                                    gpu_uniforms.size() * sizeof(UniformTimeVaryingGPU),
                                    0, cudaMemcpyHostToDevice, stream);

            apply_uniform_time_varying_kernel<<<num_blocks, block_size, 0, stream>>>(
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                num_particles
            );
        }

        // Type 2: Space-varying, time-constant fields (linear)
        for (size_t i = 0; i < space_varying_fields.size(); ++i) {
            InterpolatorParams3D params = get_interp_params(space_varying_fields[i]);
            cudaMemcpyToSymbolAsync(dev_space_varying_params, &params,
                                    sizeof(InterpolatorParams3D), i * sizeof(InterpolatorParams3D),
                                    cudaMemcpyHostToDevice, stream);

            apply_space_varying_field_kernel<false><<<num_blocks, block_size, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                static_cast<int>(i),
                num_particles
            );
        }

        // Type 2: Space-varying, time-constant fields (cubic)
        for (size_t i = 0; i < space_varying_cubic.size(); ++i) {
            CubicInterpolatorParams3D params = get_cubic_params(space_varying_cubic[i]);
            // Reinterpret as InterpolatorParams3D for storage (same layout)
            cudaMemcpyToSymbolAsync(dev_space_varying_params, &params,
                                    sizeof(CubicInterpolatorParams3D),
                                    (space_varying_fields.size() + i) * sizeof(InterpolatorParams3D),
                                    cudaMemcpyHostToDevice, stream);

            apply_space_varying_field_kernel<true><<<num_blocks, block_size, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                static_cast<int>(space_varying_fields.size() + i),
                num_particles
            );
        }

        // Type 4: Spacetime-varying fields (linear)
        for (size_t i = 0; i < spacetime_varying_fields.size(); ++i) {
            InterpolatorParams4D params = get_interp_params(spacetime_varying_fields[i], current_time);
            cudaMemcpyToSymbolAsync(dev_spacetime_varying_params, &params,
                                    sizeof(InterpolatorParams4D), i * sizeof(InterpolatorParams4D),
                                    cudaMemcpyHostToDevice, stream);

            apply_spacetime_varying_field_kernel<false><<<num_blocks, block_size, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                static_cast<int>(i),
                num_particles
            );
        }

        // Type 4: Spacetime-varying fields (cubic)
        for (size_t i = 0; i < spacetime_varying_cubic.size(); ++i) {
            CubicInterpolatorParams4D params = get_cubic_params(spacetime_varying_cubic[i], current_time);
            // Store in constant memory
            cudaMemcpyToSymbolAsync(dev_spacetime_varying_params, &params,
                                    sizeof(CubicInterpolatorParams4D),
                                    (spacetime_varying_fields.size() + i) * sizeof(InterpolatorParams4D),
                                    cudaMemcpyHostToDevice, stream);

            apply_spacetime_varying_field_kernel<true><<<num_blocks, block_size, 0, stream>>>(
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                static_cast<int>(spacetime_varying_fields.size() + i),
                num_particles
            );
        }
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    void shutdown() {
        // Destroy streams
        if (transfer_stream) cudaStreamDestroy(transfer_stream);
        if (compute_stream) cudaStreamDestroy(compute_stream);
        transfer_stream = 0;
        compute_stream = 0;

        // Free uniform buffers
        for (auto& buf : uniform_time_varying_fields) {
            buf.free();
        }
        uniform_time_varying_fields.clear();

        if (d_query_result) {
            cudaFree(d_query_result);
            d_query_result = nullptr;
        }

        // Destroy interpolators
        for (auto& field : space_varying_fields) {
            field.destroy();
        }
        space_varying_fields.clear();

        for (auto& field : spacetime_varying_fields) {
            field.destroy();
        }
        spacetime_varying_fields.clear();

        for (auto& field : space_varying_cubic) {
            field.destroy();
        }
        space_varying_cubic.clear();

        for (auto& field : spacetime_varying_cubic) {
            field.destroy();
        }
        spacetime_varying_cubic.clear();

        // Close files
        for (auto& file : field_files) {
            if (file.is_open()) file.close();
        }
        field_files.clear();
        field_headers.clear();

        constant_fields.clear();
        field_configs.clear();
        space_varying_interp.clear();
        spacetime_varying_interp.clear();

        initialized = false;
    }

    // =========================================================================
    // Debug/Status
    // =========================================================================

    void print_field_summary() const {
        std::cout << "\n=== Acceleration Field Summary ===" << std::endl;
        std::cout << "Constant fields:           " << constant_fields.size() << std::endl;
        std::cout << "Uniform time-varying:      " << uniform_time_varying_fields.size() << std::endl;
        std::cout << "Space-varying (linear):    " << space_varying_fields.size() << std::endl;
        std::cout << "Space-varying (cubic):     " << space_varying_cubic.size() << std::endl;
        std::cout << "Spacetime-varying (linear):" << spacetime_varying_fields.size() << std::endl;
        std::cout << "Spacetime-varying (cubic): " << spacetime_varying_cubic.size() << std::endl;
        std::cout << "==================================" << std::endl;
    }
};

// =============================================================================
// Static Instance and Integration with AccelerationManager
// =============================================================================

static AccelerationFieldManager g_field_manager;
static bool g_fields_initialized = false;

void AccelerationManager::compute_field_acceleration(
    SimulationState& state,
    const SimConfig& config,
    double current_time,
    int field_id,
    cudaStream_t stream
) {
    // Initialize if not already done
    if (!g_fields_initialized) {
        g_field_manager.initialize(config);
        g_fields_initialized = true;
    }

    // The caller invokes this method once per configured field.  This manager
    // intentionally applies *all* fields together, so only the first enabled
    // caller may dispatch it; otherwise N configured fields were added N times.
    int first_enabled = -1;
    for (int i = 0; i < config.acceleration.num_accel_fields; ++i) {
        if (config.acceleration.fields[i].enabled) {
            first_enabled = i;
            break;
        }
    }
    if (field_id != first_enabled) return;
    g_field_manager.compute_all(state, config, current_time, stream);
}

void AccelerationManager::query_gravity_at_position(
    double x, double y, double z,
    float& gx, float& gy, float& gz
) {
    if (!g_fields_initialized) {
        gx = 0.0f; gy = 0.0f; gz = 0.0f;
        return;
    }

    if (!d_query_result) {
        cudaMalloc(&d_query_result, 3 * sizeof(float));
    }

    float h_res[3] = {0.0f, 0.0f, 0.0f};
    cudaMemcpy(d_query_result, h_res, 3 * sizeof(float), cudaMemcpyHostToDevice);

    float px = static_cast<float>(x);
    float py = static_cast<float>(y);
    float pz = static_cast<float>(z);

    // Type 1
    if (!g_field_manager.constant_fields.empty()) {
        query_constant_field_kernel<<<1, 1>>>(d_query_result);
    }

    // Type 3
    if (!g_field_manager.uniform_time_varying_fields.empty()) {
        query_uniform_time_varying_kernel<<<1, 1>>>(d_query_result);
    }

    // Type 2 (linear)
    for (size_t i = 0; i < g_field_manager.space_varying_fields.size(); ++i) {
        query_space_varying_field_kernel<false><<<1, 1>>>(px, py, pz, static_cast<int>(i), d_query_result);
    }

    // Type 2 (cubic)
    for (size_t i = 0; i < g_field_manager.space_varying_cubic.size(); ++i) {
        query_space_varying_field_kernel<true><<<1, 1>>>(px, py, pz, static_cast<int>(g_field_manager.space_varying_fields.size() + i), d_query_result);
    }

    // Type 4 (linear)
    for (size_t i = 0; i < g_field_manager.spacetime_varying_fields.size(); ++i) {
        query_spacetime_varying_field_kernel<false><<<1, 1>>>(px, py, pz, static_cast<int>(i), d_query_result);
    }

    // Type 4 (cubic)
    for (size_t i = 0; i < g_field_manager.spacetime_varying_cubic.size(); ++i) {
        query_spacetime_varying_field_kernel<true><<<1, 1>>>(px, py, pz, static_cast<int>(g_field_manager.spacetime_varying_fields.size() + i), d_query_result);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_query_result, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    gx = h_res[0];
    gy = h_res[1];
    gz = h_res[2];
}

// =============================================================================
// Field File Format Documentation (for reference)
// =============================================================================
/*
 Binary file format for acceleration fields:

 Header (FieldBinaryHeader):
   - magic[8]: "ACCELFD\0"
   - version: uint32
   - num_frames: uint32 (1 for static fields)
   - dim_x, dim_y, dim_z: int32 (grid dimensions, 0 for uniform)
   - origin_x, origin_y, origin_z: float
   - spacing_x, spacing_y, spacing_z: float
   - time_start: float
   - time_interval: float
   - is_time_varying: uint8
   - is_space_varying: uint8
   - interp_method: uint8
   - reserved[5]: uint8

 Data section:
   For space-constant fields (dim_x=0):
     - For each frame: float[3] (ax, ay, az)

   For v2 space-varying fields:
     - one static frame only
     - float32[frame][component][z][y][x]
     - x is contiguous; ax/ay/az planes upload directly to CUDA textures.

 Example loading patterns:
   - Distance data (LoadMethod::ONCE): Load all frames at startup
   - Asteroid gravity (batch=100): Every 100*dt_field, load 100 frames
   - Seismic data (batch=1000): Every 1000*dt_field, load 1000 frames
*/

} // namespace cudadem

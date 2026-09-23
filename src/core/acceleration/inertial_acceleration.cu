// =============================================================================
// Inertial Acceleration for DEM Simulation
// =============================================================================
// Implements non-inertial reference frame accelerations for rotating body:
// - Euler acceleration:       -ω̇ × r
// - Coriolis acceleration:    -2ω × v
// - Centrifugal acceleration: -ω × (ω × r)
//
// Total fictitious acceleration (rotating-frame correction):
//   a_fict = -ω̇ × r - 2ω × v - ω × (ω × r)
//
// CRITICAL COORDINATE TRANSFORM:
// - r must be relative to Apophis center of mass, NOT simulation box origin
// - r = surface_origin + particle_position
// - surface_origin is passed from DomainConfig.surface_origin
//
// Data Loading Strategy:
// - Rotation data (ω, ω̇) pre-calculated from SPICE propagation
// - Data interval: 60s (configurable via YAML)
// - Load all data at startup (LoadMethod::ONCE) for short simulations
// - Or batch load for long simulations
// - Linear interpolation in time to Verlet timestep
//
// Reference: Equation (A4) in asteroid dynamics literature
// =============================================================================

#include "acceleration.h"
#include "../particle_model/particle.h"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cudadem {

// =============================================================================
// Binary File Format for Rotation Data
// =============================================================================

struct RotationBinaryHeader {
    char magic[8];                   // "ROTATION"
    uint32_t version;                // File format version
    uint32_t num_frames;             // Number of time frames
    float time_start;                // Start time (seconds from epoch)
    float time_interval;             // Time between frames (typically 60s)
    uint8_t reserved[12];            // Future use

    RotationBinaryHeader()
        : version(1), num_frames(0), time_start(0), time_interval(60.0f) {
        memcpy(magic, "ROTATION", 8);
        memset(reserved, 0, sizeof(reserved));
    }
};

// Single frame of rotation data
struct RotationFrame {
    float omega_x, omega_y, omega_z;           // Angular velocity (rad/s)
    float omega_dot_x, omega_dot_y, omega_dot_z; // Angular acceleration (rad/s²)
};
static_assert(sizeof(RotationBinaryHeader) == 36, "ROTATION v1 must remain a 36-byte little-endian record");
static_assert(sizeof(RotationFrame) == 24, "ROTATION v1 frame size changed");

// =============================================================================
// Rotation Data Buffer (Host-side storage with async loading)
// =============================================================================

struct RotationDataBuffer {
    // Host staging buffers
    std::vector<float> h_times;
    std::vector<float> h_omega_x, h_omega_y, h_omega_z;
    std::vector<float> h_omega_dot_x, h_omega_dot_y, h_omega_dot_z;

    // Current interpolated values (uploaded to GPU constant memory)
    float current_omega_x, current_omega_y, current_omega_z;
    float current_omega_dot_x, current_omega_dot_y, current_omega_dot_z;

    // Loaded range
    int32_t loaded_start_frame;
    int32_t loaded_num_frames;

    // File info
    int32_t total_frames;
    float time_start;
    float time_interval;

    // Batch size
    int32_t batch_size;

    RotationDataBuffer()
        : current_omega_x(0), current_omega_y(0), current_omega_z(0),
          current_omega_dot_x(0), current_omega_dot_y(0), current_omega_dot_z(0),
          loaded_start_frame(-1), loaded_num_frames(0),
          total_frames(0), time_start(0), time_interval(60.0f),
          batch_size(1000) {}

    void allocate(int32_t batch) {
        batch_size = batch;
        h_times.resize(batch);
        h_omega_x.resize(batch);
        h_omega_y.resize(batch);
        h_omega_z.resize(batch);
        h_omega_dot_x.resize(batch);
        h_omega_dot_y.resize(batch);
        h_omega_dot_z.resize(batch);
    }

    void clear() {
        h_times.clear();
        h_omega_x.clear();
        h_omega_y.clear();
        h_omega_z.clear();
        h_omega_dot_x.clear();
        h_omega_dot_y.clear();
        h_omega_dot_z.clear();
        loaded_start_frame = -1;
        loaded_num_frames = 0;
    }

    // Interpolate rotation data at given time
    void interpolate_at_time(float query_time) {
        if (loaded_num_frames <= 0) {
            throw std::runtime_error("ROTATION data were not loaded");
        }
        const float first = time_start;
        const float last = time_start + (total_frames - 1) * time_interval;
        if (!std::isfinite(query_time) || query_time < first || query_time > last)
            throw std::runtime_error("ROTATION data do not cover the requested simulation time");

        // Find bracketing frames
        float rel_time = (query_time - time_start) / time_interval;
        int frame_lo = static_cast<int>(rel_time) - loaded_start_frame;
        int frame_hi = frame_lo + 1;

        // Clamp to loaded range
        frame_lo = std::max(0, std::min(frame_lo, loaded_num_frames - 1));
        frame_hi = std::max(0, std::min(frame_hi, loaded_num_frames - 1));

        if (frame_lo == frame_hi) {
            // No interpolation needed
            current_omega_x = h_omega_x[frame_lo];
            current_omega_y = h_omega_y[frame_lo];
            current_omega_z = h_omega_z[frame_lo];
            current_omega_dot_x = h_omega_dot_x[frame_lo];
            current_omega_dot_y = h_omega_dot_y[frame_lo];
            current_omega_dot_z = h_omega_dot_z[frame_lo];
            return;
        }

        // Linear interpolation
        float t0 = time_start + (loaded_start_frame + frame_lo) * time_interval;
        float t1 = time_start + (loaded_start_frame + frame_hi) * time_interval;
        float alpha = (query_time - t0) / (t1 - t0);
        alpha = std::max(0.0f, std::min(1.0f, alpha));

        current_omega_x = h_omega_x[frame_lo] + alpha * (h_omega_x[frame_hi] - h_omega_x[frame_lo]);
        current_omega_y = h_omega_y[frame_lo] + alpha * (h_omega_y[frame_hi] - h_omega_y[frame_lo]);
        current_omega_z = h_omega_z[frame_lo] + alpha * (h_omega_z[frame_hi] - h_omega_z[frame_lo]);
        current_omega_dot_x = h_omega_dot_x[frame_lo] + alpha * (h_omega_dot_x[frame_hi] - h_omega_dot_x[frame_lo]);
        current_omega_dot_y = h_omega_dot_y[frame_lo] + alpha * (h_omega_dot_y[frame_hi] - h_omega_dot_y[frame_lo]);
        current_omega_dot_z = h_omega_dot_z[frame_lo] + alpha * (h_omega_dot_z[frame_hi] - h_omega_dot_z[frame_lo]);
    }
};

// =============================================================================
// GPU Constant Memory for Inertial Parameters
// =============================================================================

// Current rotation state (updated each timestep via interpolation)
struct InertialParamsGPU {
    // Angular velocity (rad/s)
    float omega_x, omega_y, omega_z;

    // Angular acceleration (rad/s²)
    float omega_dot_x, omega_dot_y, omega_dot_z;

    // Patch offset from Apophis center of mass (meters)
    // This is DomainConfig.surface_origin
    double offset_x, offset_y, offset_z;

    // Enable flag
    bool enabled;
};

__constant__ InertialParamsGPU dev_inertial_params;

// =============================================================================
// Inertial Acceleration Kernel
// =============================================================================
// Computes: a_fict = -ω̇ × r - 2ω × v - ω × (ω × r)
// Where:
//   r = offset + particle_position (relative to Apophis COM)
//   v = particle_velocity (ṙ in body-fixed frame)
//   ω = angular velocity
//   ω̇ = angular acceleration
// =============================================================================

__global__ void compute_inertial_acceleration_kernel(
    // Particle positions (double precision)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle velocities
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
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
    if (!dev_inertial_params.enabled) return;

    // Load particle position and velocity
    // Position relative to simulation box origin (surface-fixed)
    float px = static_cast<float>(pos_x[idx]);
    float py = static_cast<float>(pos_y[idx]);
    float pz = static_cast<float>(pos_z[idx]);

    double vx = vel_x[idx];
    double vy = vel_y[idx];
    double vz = vel_z[idx];

    float m = mass[idx];

    // CRITICAL: Transform to position relative to Apophis center of mass
    // r = offset + particle_position
    float rx = static_cast<float>(dev_inertial_params.offset_x) + px;
    float ry = static_cast<float>(dev_inertial_params.offset_y) + py;
    float rz = static_cast<float>(dev_inertial_params.offset_z) + pz;

    // Load rotation parameters
    float wx = dev_inertial_params.omega_x;
    float wy = dev_inertial_params.omega_y;
    float wz = dev_inertial_params.omega_z;

    float wdx = dev_inertial_params.omega_dot_x;
    float wdy = dev_inertial_params.omega_dot_y;
    float wdz = dev_inertial_params.omega_dot_z;

    // -------------------------------------------------------------------------
    // 1. Euler acceleration: ω̇ × r
    // -------------------------------------------------------------------------
    // Cast to double for calculation
    double wdy_d = static_cast<double>(wdy);
    double wdx_d = static_cast<double>(wdx);
    double wdz_d = static_cast<double>(wdz);
    double rz_d = static_cast<double>(rz);
    double ry_d = static_cast<double>(ry);
    double rx_d = static_cast<double>(rx);
    
    double euler_x = wdy_d * rz_d - wdz_d * ry_d;
    double euler_y = wdz_d * rx_d - wdx_d * rz_d;
    double euler_z = wdx_d * ry_d - wdy_d * rx_d;

    // -------------------------------------------------------------------------
    // 2. Coriolis acceleration: 2ω × v
    // -------------------------------------------------------------------------
    double wx_d = static_cast<double>(wx);
    double wy_d = static_cast<double>(wy);
    double wz_d = static_cast<double>(wz);
    double vx_d = vx;
    double vy_d = vy;
    double vz_d = vz;

    double coriolis_x = 2.0 * (wy_d * vz_d - wz_d * vy_d);
    double coriolis_y = 2.0 * (wz_d * vx_d - wx_d * vz_d);
    double coriolis_z = 2.0 * (wx_d * vy_d - wy_d * vx_d);

    // -------------------------------------------------------------------------
    // 3. Centrifugal acceleration: ω × (ω × r)
    // -------------------------------------------------------------------------
    double omega_cross_r_x = wy_d * rz_d - wz_d * ry_d;
    double omega_cross_r_y = wz_d * rx_d - wx_d * rz_d;
    double omega_cross_r_z = wx_d * ry_d - wy_d * rx_d;

    double centrifugal_x = wy_d * omega_cross_r_z - wz_d * omega_cross_r_y;
    double centrifugal_y = wz_d * omega_cross_r_x - wx_d * omega_cross_r_z;
    double centrifugal_z = wx_d * omega_cross_r_y - wy_d * omega_cross_r_x;

    // -------------------------------------------------------------------------
    // Total fictitious acceleration (negative: -ω̇×r - 2ω×v - ω×(ω×r))
    // -------------------------------------------------------------------------
    double ax = -(euler_x + coriolis_x + centrifugal_x);
    double ay = -(euler_y + coriolis_y + centrifugal_y);
    double az = -(euler_z + coriolis_z + centrifugal_z);

    // Convert to force and add to existing forces
    atomic_add_force(force_x, force_y, force_z, m * ax, m * ay, m * az, idx);
}

// =============================================================================
// Inertial Acceleration Manager
// =============================================================================

class InertialAccelerationManager {
public:
    RotationDataBuffer rotation_buffer;
    std::ifstream data_file;
    RotationBinaryHeader header;

    // Configuration
    bool enabled;
    LoadMethod load_method;

    // Constant omega mode (for rotating frame tidal simulations)
    bool use_constant_omega;
    double constant_omega_x, constant_omega_y, constant_omega_z;

    // Patch offset (from DomainConfig)
    double offset_x, offset_y, offset_z;

    // State
    bool initialized;
    bool data_loaded;

    InertialAccelerationManager()
        : enabled(false), load_method(LoadMethod::ONCE),
          use_constant_omega(false),
          constant_omega_x(0), constant_omega_y(0), constant_omega_z(0),
          offset_x(0), offset_y(0), offset_z(0),
          initialized(false), data_loaded(false) {}

    ~InertialAccelerationManager() {
        shutdown();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    bool initialize(const SimConfig& config) {
        if (initialized) {
            shutdown();
        }

        enabled = config.acceleration.enable_inertial;
        if (!enabled) {
            std::cout << "InertialAccelerationManager: Disabled" << std::endl;
            initialized = true;
            return true;
        }

        // Check for constant-omega mode (rotating frame)
        use_constant_omega = config.acceleration.inertial_constant_omega;

        // Store patch offset.
        // SURFACE mode: particle positions are relative to patch origin,
        //   so offset = surface_origin (position of patch in body-fixed frame).
        // MASS_CENTER / ROTATING mode: particle positions are already
        //   relative to COM, so offset = 0.
        if (config.coord_system == CoordinateSystem::SURFACE) {
            offset_x = config.domain.surface_origin.x;
            offset_y = config.domain.surface_origin.y;
            offset_z = config.domain.surface_origin.z;
        } else {
            offset_x = 0.0;
            offset_y = 0.0;
            offset_z = 0.0;
        }

        // -----------------------------------------------------------------
        // Constant-omega path: no binary file needed
        // -----------------------------------------------------------------
        if (use_constant_omega) {
            constant_omega_x = config.rotating_frame.omega.x;
            constant_omega_y = config.rotating_frame.omega.y;
            constant_omega_z = config.rotating_frame.omega.z;

            initialized = true;
            data_loaded = true;  // no file to load

            double omega_mag = std::sqrt(constant_omega_x * constant_omega_x +
                                         constant_omega_y * constant_omega_y +
                                         constant_omega_z * constant_omega_z);
            std::cout << "InertialAccelerationManager initialized (constant omega):" << std::endl;
            std::cout << "  Omega: (" << constant_omega_x << ", " << constant_omega_y
                      << ", " << constant_omega_z << ") rad/s" << std::endl;
            if (omega_mag > 1e-10) {
                std::cout << "  Rotation period: " << (2.0 * M_PI / omega_mag / 3600.0) << " h" << std::endl;
            }
            std::cout << "  Euler term: 0 (omega_dot = 0)" << std::endl;
            return true;
        }

        // -----------------------------------------------------------------
        // Time-varying omega path: read binary rotation data file
        // -----------------------------------------------------------------

        // Open data file
        const char* path = config.acceleration.inertial_data_path;
        if (path[0] == '\0') {
            std::cerr << "ERROR: No inertial data path specified" << std::endl;
            return false;
        }

        data_file.open(path, std::ios::binary);
        if (!data_file.is_open()) {
            std::cerr << "ERROR: Cannot open inertial data file: " << path << std::endl;
            return false;
        }

        // Read header
        data_file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!data_file || memcmp(header.magic, "ROTATION", 8) != 0 || header.version != 1 ||
            header.num_frames == 0 || header.num_frames > 100000000 ||
            !std::isfinite(header.time_start) || !std::isfinite(header.time_interval) ||
            header.time_interval <= 0.0f) {
            std::cerr << "ERROR: Invalid rotation data file format" << std::endl;
            data_file.close();
            return false;
        }
        data_file.seekg(0, std::ios::end);
        const std::streamoff expected = static_cast<std::streamoff>(sizeof(RotationBinaryHeader)) +
            static_cast<std::streamoff>(header.num_frames) * sizeof(RotationFrame);
        if (data_file.tellg() != expected) {
            std::cerr << "ERROR: ROTATION v1 file has a truncated payload or trailing bytes" << std::endl;
            data_file.close();
            return false;
        }
        data_file.seekg(sizeof(RotationBinaryHeader), std::ios::beg);

        // Initialize buffer
        rotation_buffer.total_frames = header.num_frames;
        rotation_buffer.time_start = header.time_start;
        rotation_buffer.time_interval = header.time_interval;

        // Determine batch size based on load method
        if (load_method == LoadMethod::ONCE) {
            // Load all data at startup
            rotation_buffer.allocate(header.num_frames);
            load_all_data();
        } else {
            // Batch loading
            int32_t batch = static_cast<int32_t>(
                config.acceleration.inertial_time_interval / header.time_interval);
            batch = std::max(100, batch);  // At least 100 frames
            rotation_buffer.allocate(batch);
        }

        initialized = true;

        std::cout << "InertialAccelerationManager initialized:" << std::endl;
        std::cout << "  Data frames: " << header.num_frames << std::endl;
        std::cout << "  Time interval: " << header.time_interval << " s" << std::endl;
        std::cout << "  Patch offset: (" << offset_x << ", " << offset_y
                  << ", " << offset_z << ") m" << std::endl;
        std::cout << "  Load method: " << (load_method == LoadMethod::ONCE ? "ONCE" : "SEQUENTIAL")
                  << std::endl;

        return true;
    }

    // =========================================================================
    // Data Loading
    // =========================================================================

    void load_all_data() {
        if (data_loaded) return;

        int32_t num_frames = header.num_frames;

        // Seek to data section (after header)
        data_file.seekg(sizeof(RotationBinaryHeader));

        // Read all frames
        for (int32_t i = 0; i < num_frames; ++i) {
            RotationFrame frame;
            data_file.read(reinterpret_cast<char*>(&frame), sizeof(RotationFrame));
            if (!data_file || !std::isfinite(frame.omega_x) || !std::isfinite(frame.omega_y) ||
                !std::isfinite(frame.omega_z) || !std::isfinite(frame.omega_dot_x) ||
                !std::isfinite(frame.omega_dot_y) || !std::isfinite(frame.omega_dot_z))
                throw std::runtime_error("ROTATION v1 frame is truncated or non-finite");

            rotation_buffer.h_times[i] = header.time_start + i * header.time_interval;
            rotation_buffer.h_omega_x[i] = frame.omega_x;
            rotation_buffer.h_omega_y[i] = frame.omega_y;
            rotation_buffer.h_omega_z[i] = frame.omega_z;
            rotation_buffer.h_omega_dot_x[i] = frame.omega_dot_x;
            rotation_buffer.h_omega_dot_y[i] = frame.omega_dot_y;
            rotation_buffer.h_omega_dot_z[i] = frame.omega_dot_z;
        }

        rotation_buffer.loaded_start_frame = 0;
        rotation_buffer.loaded_num_frames = num_frames;
        data_loaded = true;

        std::cout << "Loaded " << num_frames << " rotation frames" << std::endl;
    }

    void load_batch(int start_frame) {
        // Clamp to valid range
        start_frame = std::max(0, std::min(start_frame,
                               rotation_buffer.total_frames - rotation_buffer.batch_size));
        int num_to_load = std::min(rotation_buffer.batch_size,
                                   rotation_buffer.total_frames - start_frame);

        // Seek to position
        size_t data_offset = sizeof(RotationBinaryHeader) +
                             start_frame * sizeof(RotationFrame);
        data_file.seekg(data_offset);

        // Read frames
        for (int i = 0; i < num_to_load; ++i) {
            RotationFrame frame;
            data_file.read(reinterpret_cast<char*>(&frame), sizeof(RotationFrame));
            if (!data_file || !std::isfinite(frame.omega_x) || !std::isfinite(frame.omega_y) ||
                !std::isfinite(frame.omega_z) || !std::isfinite(frame.omega_dot_x) ||
                !std::isfinite(frame.omega_dot_y) || !std::isfinite(frame.omega_dot_z))
                throw std::runtime_error("ROTATION v1 frame is truncated or non-finite");

            rotation_buffer.h_times[i] = header.time_start +
                                         (start_frame + i) * header.time_interval;
            rotation_buffer.h_omega_x[i] = frame.omega_x;
            rotation_buffer.h_omega_y[i] = frame.omega_y;
            rotation_buffer.h_omega_z[i] = frame.omega_z;
            rotation_buffer.h_omega_dot_x[i] = frame.omega_dot_x;
            rotation_buffer.h_omega_dot_y[i] = frame.omega_dot_y;
            rotation_buffer.h_omega_dot_z[i] = frame.omega_dot_z;
        }

        rotation_buffer.loaded_start_frame = start_frame;
        rotation_buffer.loaded_num_frames = num_to_load;
    }

    void update_prefetch(double current_time) {
        if (load_method == LoadMethod::ONCE) return;
        if (!data_file.is_open()) return;

        // Calculate which frame we need
        float rel_time = (static_cast<float>(current_time) - rotation_buffer.time_start) /
                         rotation_buffer.time_interval;
        int needed_frame = static_cast<int>(rel_time);

        // Check if we need to load a new batch
        bool need_load = false;
        if (rotation_buffer.loaded_start_frame < 0) {
            need_load = true;
        } else {
            int frames_ahead = (rotation_buffer.loaded_start_frame +
                               rotation_buffer.loaded_num_frames) - needed_frame;
            if (frames_ahead < rotation_buffer.batch_size / 4) {
                need_load = true;
            }
        }

        if (need_load) {
            int start_frame = std::max(0, needed_frame - rotation_buffer.batch_size / 4);
            load_batch(start_frame);
        }
    }

    // =========================================================================
    // Compute
    // =========================================================================

    void compute(
        SimulationState& state,
        const SimConfig& config,
        double current_time,
        cudaStream_t stream
    ) {
        if (!initialized || !enabled) return;

        int32_t num_particles = state.particles.num_particles;
        if (num_particles == 0) return;

        // Prepare GPU parameters
        InertialParamsGPU params;

        if (use_constant_omega) {
            // Constant-omega mode: use stored omega, omega_dot = 0
            params.omega_x = static_cast<float>(constant_omega_x);
            params.omega_y = static_cast<float>(constant_omega_y);
            params.omega_z = static_cast<float>(constant_omega_z);
            params.omega_dot_x = 0.0f;
            params.omega_dot_y = 0.0f;
            params.omega_dot_z = 0.0f;
        } else {
            // Time-varying mode: interpolate from binary data
            update_prefetch(current_time);
            rotation_buffer.interpolate_at_time(static_cast<float>(current_time));

            params.omega_x = rotation_buffer.current_omega_x;
            params.omega_y = rotation_buffer.current_omega_y;
            params.omega_z = rotation_buffer.current_omega_z;
            params.omega_dot_x = rotation_buffer.current_omega_dot_x;
            params.omega_dot_y = rotation_buffer.current_omega_dot_y;
            params.omega_dot_z = rotation_buffer.current_omega_dot_z;
        }
        params.offset_x = offset_x;
        params.offset_y = offset_y;
        params.offset_z = offset_z;
        params.enabled = true;

        // Upload to constant memory
        cudaMemcpyToSymbolAsync(dev_inertial_params, &params,
                                sizeof(InertialParamsGPU), 0,
                                cudaMemcpyHostToDevice, stream);

        // Launch kernel
        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        compute_inertial_acceleration_kernel<<<num_blocks, block_size, 0, stream>>>(
            state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
            state.particles.vel_x, state.particles.vel_y, state.particles.vel_z,
            state.particles.mass,
            state.particles.force_x, state.particles.force_y, state.particles.force_z,
            num_particles
        );
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    void shutdown() {
        if (data_file.is_open()) {
            data_file.close();
        }
        rotation_buffer.clear();
        enabled = false;
        initialized = false;
        data_loaded = false;
    }
};

// =============================================================================
// Static Instance
// =============================================================================

static InertialAccelerationManager g_inertial_manager;

// =============================================================================
// AccelerationManager Integration
// =============================================================================

void AccelerationManager::compute_inertial(
    SimulationState& state,
    const SimConfig& config,
    double current_time,
    cudaStream_t stream
) {
    // Initialize on first call
    static bool inertial_initialized = false;
    if (!inertial_initialized) {
        if (!g_inertial_manager.initialize(config))
            throw std::runtime_error("failed to initialize ROTATION/inertial acceleration");
        inertial_initialized = true;
    }

    g_inertial_manager.compute(state, config, current_time, stream);
}

// =============================================================================
// Utility: Generate Rotation Data File (for testing/preprocessing)
// =============================================================================
// This function can be called from Python via pybind11 or from C++ preprocessing
// to convert SPICE ephemeris to binary format

#ifndef __CUDACC__

inline void write_rotation_data_binary(
    const std::string& output_path,
    const std::vector<float>& times,
    const std::vector<float>& omega_x,
    const std::vector<float>& omega_y,
    const std::vector<float>& omega_z,
    const std::vector<float>& omega_dot_x,
    const std::vector<float>& omega_dot_y,
    const std::vector<float>& omega_dot_z
) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create rotation data file: " + output_path);
    }

    int32_t num_frames = static_cast<int32_t>(times.size());
    float time_interval = (num_frames > 1) ? (times[1] - times[0]) : 60.0f;

    RotationBinaryHeader header;
    header.num_frames = num_frames;
    header.time_start = times[0];
    header.time_interval = time_interval;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (int32_t i = 0; i < num_frames; ++i) {
        RotationFrame frame;
        frame.omega_x = omega_x[i];
        frame.omega_y = omega_y[i];
        frame.omega_z = omega_z[i];
        frame.omega_dot_x = omega_dot_x[i];
        frame.omega_dot_y = omega_dot_y[i];
        frame.omega_dot_z = omega_dot_z[i];
        file.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
    }

    file.close();
    std::cout << "Wrote " << num_frames << " frames to " << output_path << std::endl;
}

#endif // __CUDACC__

} // namespace cudadem

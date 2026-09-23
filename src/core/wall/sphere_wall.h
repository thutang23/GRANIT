#ifndef CUDADEM_SPHERE_WALL_H
#define CUDADEM_SPHERE_WALL_H

// =============================================================================
// Sphere Wall — Prescribed-trajectory massive sphere with one-way contact coupling
// =============================================================================
// A sphere wall exerts contact forces (same Hertz-Mindlin formula as particle-particle)
// and point-mass gravity on all DEM particles, but does not receive back-reaction.
// Wall state is prescribed by the shared RIGIDPS2 rigid-pose trajectory:
// f64 time + world translation + body-to-world quaternion (wxyz), with an
// explicit reference origin and fatal-outside-coverage policy by default.
//
// Design:
// - SphereWallState in __constant__ memory (single wall per simulation)
// - Host-side trajectory interpolation + cudaMemcpyToSymbol each timestep
// - Contact force kernel reuses compute_contact_forces() from particle.h
// - Gravity kernel adds G*M/r² acceleration to each particle
// - WallContactHistory for tangential/rolling/twisting spring memory
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../wall/wall.h"
#include "../particle_model/particle.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstdint>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace cudadem {

// Forward declaration
struct SimulationState;

// =============================================================================
// SphereWallState — GPU constant memory struct
// =============================================================================
struct SphereWallState {
    bool enabled;
    int32_t material_id;
    
    // Rigid body properties (constant — from config)
    float radius;                    // Wall sphere radius (m)
    float mass;                      // Wall mass (kg)
    float moment_of_inertia;         // I = 2/5 * m * r^2 (kg*m^2)
    float cohesive_strength;         // Cohesion parameter (Pa)
    
    // Mutable state (updated each timestep from trajectory interpolation)
    double pos_x, pos_y, pos_z;      // Position (world frame, m)
    double vel_x, vel_y, vel_z;      // Linear velocity (m/s)
    double omega_x, omega_y, omega_z;// Angular velocity (rad/s)
    
    __host__ __device__ SphereWallState()
        : enabled(false), material_id(-1), radius(1.0f), mass(1.0f),
          moment_of_inertia(0.4f), cohesive_strength(0.0f),
          pos_x(0), pos_y(0), pos_z(0),
          vel_x(0), vel_y(0), vel_z(0),
          omega_x(0), omega_y(0), omega_z(0) {}
};

// =============================================================================
// SphereWallTrajectoryData — Host + device trajectory storage
// =============================================================================
struct SphereWallTrajectoryData {
    std::vector<double> times;
    std::vector<double> pos_x, pos_y, pos_z;
    std::vector<double> vel_x, vel_y, vel_z;
    std::vector<double> omega_x, omega_y, omega_z;
    int32_t num_frames;
    bool hold_out_of_range;
    
    // Device arrays for binary-search interpolation
    float*  d_times;
    double* d_pos_x, *d_pos_y, *d_pos_z;
    double* d_vel_x, *d_vel_y, *d_vel_z;
    double* d_omega_x, *d_omega_y, *d_omega_z;
    
    SphereWallTrajectoryData()
        : num_frames(0), hold_out_of_range(false),
          d_times(nullptr),
          d_pos_x(nullptr), d_pos_y(nullptr), d_pos_z(nullptr),
          d_vel_x(nullptr), d_vel_y(nullptr), d_vel_z(nullptr),
          d_omega_x(nullptr), d_omega_y(nullptr), d_omega_z(nullptr) {}
    
    void free_device() {
        if (d_times) cudaFree(d_times);
        if (d_pos_x) cudaFree(d_pos_x);
        if (d_pos_y) cudaFree(d_pos_y);
        if (d_pos_z) cudaFree(d_pos_z);
        if (d_vel_x) cudaFree(d_vel_x);
        if (d_vel_y) cudaFree(d_vel_y);
        if (d_vel_z) cudaFree(d_vel_z);
        if (d_omega_x) cudaFree(d_omega_x);
        if (d_omega_y) cudaFree(d_omega_y);
        if (d_omega_z) cudaFree(d_omega_z);
        d_times = nullptr;
        d_pos_x = nullptr; d_pos_y = nullptr; d_pos_z = nullptr;
        d_vel_x = nullptr; d_vel_y = nullptr; d_vel_z = nullptr;
        d_omega_x = nullptr; d_omega_y = nullptr; d_omega_z = nullptr;
    }
    
    void copy_to_device() {
        if (num_frames <= 0) return;
        size_t float_size = num_frames * sizeof(float);
        size_t double_size = num_frames * sizeof(double);
        cudaMalloc(&d_times, float_size);
        cudaMalloc(&d_pos_x, double_size);
        cudaMalloc(&d_pos_y, double_size);
        cudaMalloc(&d_pos_z, double_size);
        cudaMalloc(&d_vel_x, double_size);
        cudaMalloc(&d_vel_y, double_size);
        cudaMalloc(&d_vel_z, double_size);
        cudaMalloc(&d_omega_x, double_size);
        cudaMalloc(&d_omega_y, double_size);
        cudaMalloc(&d_omega_z, double_size);
        std::vector<float> time_f32(times.begin(), times.end());
        cudaMemcpy(d_times, time_f32.data(), float_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_x, pos_x.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, pos_y.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_z, pos_z.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_x, vel_x.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_y, vel_y.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_z, vel_z.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_omega_x, omega_x.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_omega_y, omega_y.data(), double_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_omega_z, omega_z.data(), double_size, cudaMemcpyHostToDevice);
    }
    
    // Load shared RIGIDPS2 trajectory.  Legacy velocity-only files are
    // deliberately rejected: their initial pose and motion semantics are
    // ambiguous (rigid motion versus a conveyor surface velocity).
    bool load_from_file(const char* filepath) {
        if (!filepath || filepath[0] == '\0') return false;
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open sphere wall trajectory file: " << filepath << std::endl;
            return false;
        }
        char magic[8]; uint32_t version = 0, n_frames = 0, policy = 0, reserved = 0;
        double reference_origin[3];
        file.read(magic, sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&n_frames), sizeof(n_frames));
        file.read(reinterpret_cast<char*>(reference_origin), sizeof(reference_origin));
        file.read(reinterpret_cast<char*>(&policy), sizeof(policy));
        file.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
        if (!file || std::memcmp(magic, "RIGIDPS2", 8) != 0 || version != 2 ||
            n_frames == 0 || n_frames > 1000000 || policy > 1 || reserved != 0 ||
            !std::isfinite(reference_origin[0]) || !std::isfinite(reference_origin[1]) || !std::isfinite(reference_origin[2])) {
            std::cerr << "ERROR: sphere wall trajectory must be valid RIGIDPS2" << std::endl;
            return false;
        }
        file.seekg(0, std::ios::end);
        constexpr std::streamoff header_size = 48, record_size = 64;
        if (file.tellg() != header_size + static_cast<std::streamoff>(n_frames) * record_size) {
            std::cerr << "ERROR: RIGIDPS2 trajectory has a truncated payload or trailing bytes" << std::endl;
            return false;
        }
        file.seekg(header_size, std::ios::beg);
        num_frames = n_frames;
        hold_out_of_range = policy == 1;
        times.resize(num_frames);
        pos_x.resize(num_frames); pos_y.resize(num_frames); pos_z.resize(num_frames);
        vel_x.resize(num_frames); vel_y.resize(num_frames); vel_z.resize(num_frames);
        omega_x.resize(num_frames); omega_y.resize(num_frames); omega_z.resize(num_frames);
        std::vector<std::array<double, 4>> q(num_frames);
        for (int32_t i = 0; i < num_frames; ++i) {
            double record[8]; file.read(reinterpret_cast<char*>(record), sizeof(record));
            if (!file || !std::isfinite(record[0]) || !std::isfinite(record[1]) || !std::isfinite(record[2]) || !std::isfinite(record[3]) ||
                !std::isfinite(record[4]) || !std::isfinite(record[5]) || !std::isfinite(record[6]) || !std::isfinite(record[7])) {
                std::cerr << "ERROR: RIGIDPS2 contains a truncated or non-finite pose" << std::endl;
                num_frames = 0; return false;
            }
            times[i] = record[0]; pos_x[i] = record[1]; pos_y[i] = record[2]; pos_z[i] = record[3];
            q[i] = {record[4], record[5], record[6], record[7]};
            const double norm = std::sqrt(q[i][0]*q[i][0] + q[i][1]*q[i][1] + q[i][2]*q[i][2] + q[i][3]*q[i][3]);
            if (norm <= 1e-12 || (i && times[i] <= times[i-1])) {
                std::cerr << "ERROR: RIGIDPS2 times/quaternions are invalid" << std::endl;
                num_frames = 0; return false;
            }
            for (double& value : q[i]) value /= norm;
        }
        for (int32_t i = 0; i < num_frames; ++i) {
            const int32_t lo = i == num_frames - 1 ? std::max(0, i - 1) : i;
            const int32_t hi = i == num_frames - 1 ? i : i + 1;
            const double dt = hi == lo ? 1.0 : times[hi] - times[lo];
            vel_x[i] = (pos_x[hi] - pos_x[lo]) / dt; vel_y[i] = (pos_y[hi] - pos_y[lo]) / dt; vel_z[i] = (pos_z[hi] - pos_z[lo]) / dt;
            auto qa = q[lo], qb = q[hi]; double dot = qa[0]*qb[0]+qa[1]*qb[1]+qa[2]*qb[2]+qa[3]*qb[3]; if (dot < 0) for (double& v : qb) v = -v;
            const double rw = qb[0]*qa[0]+qb[1]*qa[1]+qb[2]*qa[2]+qb[3]*qa[3];
            const double rx = -qb[0]*qa[1]+qb[1]*qa[0]-qb[2]*qa[3]+qb[3]*qa[2];
            const double ry = -qb[0]*qa[2]+qb[1]*qa[3]+qb[2]*qa[0]-qb[3]*qa[1];
            const double rz = -qb[0]*qa[3]-qb[1]*qa[2]+qb[2]*qa[1]+qb[3]*qa[0];
            const double s = std::sqrt(rx*rx+ry*ry+rz*rz), angle = 2.0*std::atan2(s, std::max(rw, 0.0));
            omega_x[i] = s < 1e-14 ? 0.0 : rx / s * angle / dt; omega_y[i] = s < 1e-14 ? 0.0 : ry / s * angle / dt; omega_z[i] = s < 1e-14 ? 0.0 : rz / s * angle / dt;
        }
        if (!file) {
            std::cerr << "ERROR: Failed to read RIGIDPS2 trajectory: " << filepath << std::endl;
            num_frames = 0;
            return false;
        }
        std::cout << "Loaded " << num_frames << " RIGIDPS2 sphere wall trajectory frames from " << filepath << std::endl;
        return true;
    }
};

// =============================================================================
// SphereWallManager — Static orchestration class
// =============================================================================
class SphereWallManager {
private:
    static SphereWallTrajectoryData trajectory_data_;
    static bool initialized_;
    static SphereWallState host_state_;  // Cached host-side state
    
public:
    static void initialize(const SimConfig& config, int32_t num_particles);
    
    // Interpolate wall state at current_time and copy to __constant__ memory
    static void update_state(double current_time);
    
    // Launch contact force kernel
    static void compute_contact_forces(
        SimulationState& state, const SimConfig& config,
        float dt, cudaStream_t stream = 0);
    
    // Launch gravity kernel
    static void compute_gravity(
        SimulationState& state, const SimConfig& config,
        cudaStream_t stream = 0);
    
    // Copy __constant__ memory to current device (multi-GPU)
    static void copy_to_current_device();
    
    // Free trajectory device arrays, reset state
    static void shutdown();
    
    // Query
    static bool is_enabled();
    static const SphereWallState& get_host_state();
};

} // namespace cudadem

#endif // CUDADEM_SPHERE_WALL_H

#ifndef CUDADEM_WALL_H
#define CUDADEM_WALL_H

// =============================================================================
// Unified Wall Interface for DEM Simulation
// =============================================================================
// This header provides a unified interface for all wall types:
// - Plain walls (static and moving infinite planes)
// - Periodic boundaries (coordinate wrapping)
// - Future: Mesh walls, cylindrical walls, etc.
//
// Design Philosophy:
// - Template Policy Composition for compile-time wall type selection
// - Runtime Configuration via SimConfig for wall parameters
// - Common structures shared across all wall types
// - Single WallManager class for orchestration
//
// Usage:
//   WallManager::initialize(state, config);
//   WallManager::compute_forces(state, config, dt, current_time, stream);
//   WallManager::apply_periodic_boundaries(state, config, stream);
// =============================================================================

#include "../constant/command_line_interface.h"
// Note: Do NOT include system_initialize.h here to avoid circular dependency
// system_initialize.h includes wall.h
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>

namespace cudadem {

// Forward declarations (defined in system_initialize.h)
// Must be inside namespace cudadem since SimulationState is in this namespace
struct SimulationState;

// Material-aware plain/moving/finite-plane implementation.  These wrappers
// keep the dispatcher private to plain_wall.cu while WallManager owns the
// common wall lifecycle.
void plain_wall_initialize(const SimConfig& config);
void plain_wall_copy_to_current_device();
void plain_wall_compute_forces(const SimulationState& state, const SimConfig& config,
                               float dt, float current_time, cudaStream_t stream = 0);

// =============================================================================
// Wall Type Traits (Compile-Time Properties)
// =============================================================================
// Use these traits to query wall type properties at compile time
// =============================================================================

template<WallType Type>
struct WallTypeTraits;

template<>
struct WallTypeTraits<WallType::PLAIN> {
    static constexpr bool has_contact_force = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move = false;
    static constexpr const char* name = "Plain";
};

template<>
struct WallTypeTraits<WallType::MOVING> {
    static constexpr bool has_contact_force = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move = true;
    static constexpr const char* name = "Moving";
};

template<>
struct WallTypeTraits<WallType::PERIODICAL> {
    static constexpr bool has_contact_force = false;
    static constexpr bool has_periodic_wrapping = true;
    static constexpr bool can_move = false;
    static constexpr const char* name = "Periodic";
};

template<>
struct WallTypeTraits<WallType::MESH> {
    static constexpr bool has_contact_force = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move = true;
    static constexpr const char* name = "Mesh";
};

template<>
struct WallTypeTraits<WallType::FREE> {
    static constexpr bool has_contact_force = false;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move = false;
    static constexpr const char* name = "Free";
};

template<>
struct WallTypeTraits<WallType::BONDED_RECT> {
    static constexpr bool has_contact_force     = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move              = true;
    static constexpr const char* name           = "BondedRect";
};

template<>
struct WallTypeTraits<WallType::FINITE_PLAIN> {
    static constexpr bool has_contact_force     = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move              = false;
    static constexpr const char* name           = "FinitePlain";
};

template<>
struct WallTypeTraits<WallType::FUNNEL> {
    static constexpr bool has_contact_force     = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move              = false;
    static constexpr const char* name           = "Funnel";
};

template<>
struct WallTypeTraits<WallType::SPHERE> {
    static constexpr bool has_contact_force     = true;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move              = true;
    static constexpr const char* name           = "Sphere";
};

template<>
struct WallTypeTraits<WallType::NONE> {
    static constexpr bool has_contact_force = false;
    static constexpr bool has_periodic_wrapping = false;
    static constexpr bool can_move = false;
    static constexpr const char* name = "None";
};

// =============================================================================
// Wall Geometry Structures (GPU-friendly)
// =============================================================================
// Union-based structure for different wall geometries
// =============================================================================

// Plane wall geometry: point + normal
struct PlaneGeometry {
    float point_x, point_y, point_z;     // Point on plane
    float normal_x, normal_y, normal_z;  // Unit normal (into domain)
    double velocity_x, velocity_y, velocity_z;  // Wall velocity
    float d;                              // Plane equation: n·x + d = 0

    __host__ __device__ PlaneGeometry()
        : point_x(0), point_y(0), point_z(0),
          normal_x(0), normal_y(1), normal_z(0),
          velocity_x(0), velocity_y(0), velocity_z(0),
          d(0) {}

    // Compute d from point and normal
    __host__ __device__ void compute_d() {
        d = -(normal_x * point_x + normal_y * point_y + normal_z * point_z);
    }
};

// Periodic boundary geometry: axis + bounds
struct PeriodicGeometry {
    uint8_t axis;                        // 0=X, 1=Y, 2=Z
    float bound_min, bound_max;          // Domain bounds
    float length;                        // box_max - box_min
    float half_length;                   // For minimum image convention

    __host__ __device__ PeriodicGeometry()
        : axis(0), bound_min(0), bound_max(0), length(0), half_length(0) {}

    __host__ __device__ void compute_derived() {
        length = bound_max - bound_min;
        half_length = length * 0.5f;
    }
};

// Cylinder wall geometry (future)
struct CylinderGeometry {
    float center_x, center_y, center_z;  // Center point
    float axis_x, axis_y, axis_z;        // Unit axis direction
    float radius;                         // Cylinder radius
    float length;                         // Cylinder length (0 = infinite)

    __host__ __device__ CylinderGeometry()
        : center_x(0), center_y(0), center_z(0),
          axis_x(0), axis_y(0), axis_z(1),
          radius(1), length(0) {}
};

// Funnel wall geometry: conical frustum (interior contact)
struct FunnelGeometry {
    float point_x, point_y, point_z;     // Base point of funnel
    float axis_x, axis_y, axis_z;        // Unit axis direction (base→top)
    float radius_bottom;                  // Radius at base end
    float radius_top;                     // Radius at top end
    float length;                         // Height of frustum along axis

    __host__ __device__ FunnelGeometry()
        : point_x(0), point_y(0), point_z(0),
          axis_x(0), axis_y(0), axis_z(1),
          radius_bottom(1), radius_top(0.5f), length(1) {}
};

// Unified wall data structure (GPU)
struct WallData {
    int32_t id;
    int32_t material_id;
    WallType type;
    bool enabled;
    bool is_smooth;                      // Frictionless wall
    bool is_finite;                      // Finite-area wall (shelf)

    // Finite wall rectangular bounds (only used when is_finite == true)
    // NOTE: Currently assumes wall normal is parallel to Z-axis.
    // For arbitrary orientations, a rotation matrix would be needed.
    float half_extent_x;                // Half-length in local X direction
    float half_extent_y;                // Half-length in local Y direction

    // Velocity (for moving walls)
    double velocity_x, velocity_y, velocity_z;

    // Geometry union
    union {
        PlaneGeometry plane;
        PeriodicGeometry periodic;
        CylinderGeometry cylinder;
        FunnelGeometry funnel;
    };

    // Moving wall velocity data pointers (for time-varying velocity)
    const float* d_velocity_times;
    const double* d_velocity_x;
    const double* d_velocity_y;
    const double* d_velocity_z;
    int32_t num_velocity_frames;

    __host__ __device__ WallData()
        : id(-1), material_id(-1), type(WallType::NONE), enabled(false), is_smooth(false),
          is_finite(false), half_extent_x(1e10f), half_extent_y(1e10f),
          velocity_x(0), velocity_y(0), velocity_z(0),
          d_velocity_times(nullptr), d_velocity_x(nullptr),
          d_velocity_y(nullptr), d_velocity_z(nullptr),
          num_velocity_frames(0) {
        plane = PlaneGeometry();
    }
};

// =============================================================================
// Wall Contact History (for tangential/rolling/twisting springs)
// =============================================================================
// Each particle maintains contact history with walls it touches
// =============================================================================

struct WallContactHistory {
    int32_t wall_id;                     // Wall identifier (-1 if inactive)

    // Tangential spring displacement
    float delta_t_x, delta_t_y, delta_t_z;

    // Rolling history
    float delta_r_x, delta_r_y, delta_r_z;

    // Twisting history
    float delta_tw_x, delta_tw_y, delta_tw_z;

    // State flags
    bool is_active;
    bool is_slipping;
    bool is_rolling;
    bool is_twisting;

    __host__ __device__ WallContactHistory()
        : wall_id(-1),
          delta_t_x(0), delta_t_y(0), delta_t_z(0),
          delta_r_x(0), delta_r_y(0), delta_r_z(0),
          delta_tw_x(0), delta_tw_y(0), delta_tw_z(0),
          is_active(false), is_slipping(false),
          is_rolling(false), is_twisting(false) {}
};

// Buffer for wall contact history
struct WallContactHistoryBuffer {
    WallContactHistory* data;
    int32_t capacity;
    int32_t max_contacts_per_particle;

    WallContactHistoryBuffer()
        : data(nullptr), capacity(0), max_contacts_per_particle(8) {}
};

// =============================================================================
// Periodic Boundary Configuration (Runtime)
// =============================================================================
// Extracted from SimConfig walls for efficient GPU access
// =============================================================================

struct PeriodicBoundaryConfig {
    // Enable flags per axis
    bool periodic_x;
    bool periodic_y;
    bool periodic_z;

    // Domain bounds per axis
    float box_min_x, box_max_x;
    float box_min_y, box_max_y;
    float box_min_z, box_max_z;

    // Domain lengths (computed from bounds)
    float box_length_x;
    float box_length_y;
    float box_length_z;

    // Half-lengths (for minimum image convention)
    float half_length_x;
    float half_length_y;
    float half_length_z;

    __host__ __device__ PeriodicBoundaryConfig()
        : periodic_x(false), periodic_y(false), periodic_z(false),
          box_min_x(0), box_max_x(0),
          box_min_y(0), box_max_y(0),
          box_min_z(0), box_max_z(0),
          box_length_x(0), box_length_y(0), box_length_z(0),
          half_length_x(0), half_length_y(0), half_length_z(0) {}

    // Check if any periodic boundary is enabled
    __host__ __device__ __forceinline__
    bool is_enabled() const {
        return periodic_x || periodic_y || periodic_z;
    }

    // Compute derived quantities
    __host__ void compute_derived() {
        box_length_x = box_max_x - box_min_x;
        box_length_y = box_max_y - box_min_y;
        box_length_z = box_max_z - box_min_z;
        half_length_x = box_length_x * 0.5f;
        half_length_y = box_length_y * 0.5f;
        half_length_z = box_length_z * 0.5f;
    }
};

// =============================================================================
// Wall Force Result (returned by contact force calculation)
// =============================================================================

struct WallForceResult {
    float force_x, force_y, force_z;
    float torque_x, torque_y, torque_z;

    // Updated history values
    float H_t_x, H_t_y, H_t_z;
    float H_r_x, H_r_y, H_r_z;
    float H_tw_x, H_tw_y, H_tw_z;

    // State flags
    bool is_contact;
    bool is_slipping;
    bool is_rolling;
    bool is_twisting;

    // Energy
    float energy_elastic;

    __device__ WallForceResult()
        : force_x(0), force_y(0), force_z(0),
          torque_x(0), torque_y(0), torque_z(0),
          H_t_x(0), H_t_y(0), H_t_z(0),
          H_r_x(0), H_r_y(0), H_r_z(0),
          H_tw_x(0), H_tw_y(0), H_tw_z(0),
          is_contact(false), is_slipping(false),
          is_rolling(false), is_twisting(false),
          energy_elastic(0) {}
};

// =============================================================================
// GPU Constant Memory Declarations
// =============================================================================
// Constants for wall system
// =============================================================================

constexpr int MAX_PLAIN_WALLS = 20;   // Maximum number of plain/moving walls
constexpr int SPHERE_WALL_ID = 100;     // Unique wall_id for sphere wall in WallContactHistory
constexpr int MAX_PERIODIC_AXES = 3;  // Maximum number of periodic axes

// Forward declarations of constant memory (defined in .cu files)
// extern __constant__ WallData dev_plain_walls[MAX_PLAIN_WALLS];
// extern __constant__ int32_t dev_num_plain_walls;
// extern __constant__ PeriodicBoundaryConfig dev_pbc_config;

// =============================================================================
// Device Functions: Minimum Image Convention
// =============================================================================
// For periodic boundaries, compute shortest distance
// =============================================================================

__device__ __forceinline__
float minimum_image(float dx, float box_length, float half_length) {
    if (dx > half_length) {
        dx -= box_length;
    } else if (dx < -half_length) {
        dx += box_length;
    }
    return dx;
}

__device__ __forceinline__
void minimum_image_distance(
    float& dx, float& dy, float& dz,
    const PeriodicBoundaryConfig& pbc
) {
    if (pbc.periodic_x && pbc.box_length_x > 0.0f) {
        dx = minimum_image(dx, pbc.box_length_x, pbc.half_length_x);
    }
    if (pbc.periodic_y && pbc.box_length_y > 0.0f) {
        dy = minimum_image(dy, pbc.box_length_y, pbc.half_length_y);
    }
    if (pbc.periodic_z && pbc.box_length_z > 0.0f) {
        dz = minimum_image(dz, pbc.box_length_z, pbc.half_length_z);
    }
}

// =============================================================================
// Device Functions: Coordinate Wrapping
// =============================================================================
// Wrap coordinates back into periodic domain
// =============================================================================

__device__ __forceinline__
float wrap_coordinate(float x, float box_min, float box_max, float box_length) {
    if (x >= box_min && x <= box_max) return x;

    float x_rel = x - box_min;
    float n = floorf(x_rel / box_length);
    return x - n * box_length;
}

__device__ __forceinline__
double wrap_coordinate_double(double x, double box_min, double box_max, double box_length) {
    if (x >= box_min && x <= box_max) return x;

    double x_rel = x - box_min;
    double n = floor(x_rel / box_length);
    return x - n * box_length;
}

__device__ __forceinline__
void wrap_position_double(
    double& x, double& y, double& z,
    const PeriodicBoundaryConfig& pbc
) {
    if (pbc.periodic_x && pbc.box_length_x > 0.0f) {
        x = wrap_coordinate_double(x,
            static_cast<double>(pbc.box_min_x),
            static_cast<double>(pbc.box_max_x),
            static_cast<double>(pbc.box_length_x));
    }
    if (pbc.periodic_y && pbc.box_length_y > 0.0f) {
        y = wrap_coordinate_double(y,
            static_cast<double>(pbc.box_min_y),
            static_cast<double>(pbc.box_max_y),
            static_cast<double>(pbc.box_length_y));
    }
    if (pbc.periodic_z && pbc.box_length_z > 0.0f) {
        z = wrap_coordinate_double(z,
            static_cast<double>(pbc.box_min_z),
            static_cast<double>(pbc.box_max_z),
            static_cast<double>(pbc.box_length_z));
    }
}

// =============================================================================
// Device Functions: Plane Geometry
// =============================================================================

// Compute signed distance from point to plane
// Positive = point is on positive side (outside wall)
// Negative = point is inside wall (overlap)
__device__ __forceinline__
float plane_signed_distance(
    float px, float py, float pz,           // Particle position
    float nx, float ny, float nz,           // Wall normal (unit vector)
    float point_x, float point_y, float point_z  // Point on wall
) {
    float rx = px - point_x;
    float ry = py - point_y;
    float rz = pz - point_z;
    return rx * nx + ry * ny + rz * nz;
}

// Compute overlap for particle-plane contact
// delta_n > 0 means contact
__device__ __forceinline__
float compute_plane_overlap(float signed_distance, float particle_radius) {
    return particle_radius - signed_distance;
}

// =============================================================================
// Device Functions: Finite Wall Area Check
// =============================================================================

// Check if particle surface overlaps with finite wall rectangular area
// Accounts for particle radius so edge particles with surface over the shelf are detected
// Assumes wall is axis-aligned (normal parallel to Z-axis)
// NOTE: Prefer compute_finite_wall_contact() for force calculation — this function
//       only checks area overlap and does not compute correct edge/corner geometry.
__device__ __forceinline__
bool is_in_finite_wall_area(
    float px, float py,
    float wall_cx, float wall_cy,
    float half_extent_x, float half_extent_y,
    float particle_radius = 0.0f
) {
    float dx = px - wall_cx;
    float dy = py - wall_cy;
    return (fabsf(dx) < half_extent_x + particle_radius) &&
           (fabsf(dy) < half_extent_y + particle_radius);
}

// =============================================================================
// Device Functions: Finite Wall Closest-Point Contact
// =============================================================================
// Closest-point-on-rectangle contact: clamp particle XY to rectangle bounds,
// compute 3D gap vector for correct distance and normal at edges/corners.
// Assumes wall normal is parallel to Z-axis.
// =============================================================================

struct FiniteWallContactResult {
    float delta_n;
    float normal_x, normal_y, normal_z;
    float dist;
};

__device__ __forceinline__
FiniteWallContactResult compute_finite_wall_contact(
    float px, float py, float pz,
    float R,
    float wall_px, float wall_py, float wall_pz,
    float wall_nx, float wall_ny, float wall_nz,
    float half_ext_x, float half_ext_y
) {
    FiniteWallContactResult res;
    res.delta_n = -1.0f;
    res.normal_x = 0.0f; res.normal_y = 0.0f; res.normal_z = 0.0f;
    res.dist = 0.0f;

    float dx = px - wall_px;
    float dy = py - wall_py;
    float dz = pz - wall_pz;

    float cx = fmaxf(-half_ext_x, fminf(dx, half_ext_x));
    float cy = fmaxf(-half_ext_y, fminf(dy, half_ext_y));

    float gx = dx - cx;
    float gy = dy - cy;
    float gz = dz;

    float dist_sq = gx * gx + gy * gy + gz * gz;
    float dist = sqrtf(dist_sq);

    float delta_n = R - dist;
    if (delta_n <= 0.0f) return res;

    if (dist > 1e-10f) {
        float inv_dist = 1.0f / dist;
        res.normal_x = gx * inv_dist;
        res.normal_y = gy * inv_dist;
        res.normal_z = gz * inv_dist;
    } else {
        res.normal_x = wall_nx;
        res.normal_y = wall_ny;
        res.normal_z = wall_nz;
    }

    res.delta_n = delta_n;
    res.dist = (dist > 0.0f) ? dist : 0.0f;
    return res;
}

// Same contact calculation for an arbitrarily oriented rigid rectangle.  The
// supplied tangents must be orthonormal and perpendicular to the normal.
__device__ __forceinline__
FiniteWallContactResult compute_oriented_finite_wall_contact(
    float px, float py, float pz, float R,
    float wall_px, float wall_py, float wall_pz,
    float nx, float ny, float nz,
    float ux, float uy, float uz, float vx, float vy, float vz,
    float half_ext_x, float half_ext_y
) {
    const float dx=px-wall_px, dy=py-wall_py, dz=pz-wall_pz;
    const float u=dx*ux+dy*uy+dz*uz, v=dx*vx+dy*vy+dz*vz, n=dx*nx+dy*ny+dz*nz;
    const float cu=fmaxf(-half_ext_x,fminf(u,half_ext_x));
    const float cv=fmaxf(-half_ext_y,fminf(v,half_ext_y));
    const float gx=(u-cu)*ux+(v-cv)*vx+n*nx;
    const float gy=(u-cu)*uy+(v-cv)*vy+n*ny;
    const float gz=(u-cu)*uz+(v-cv)*vz+n*nz;
    FiniteWallContactResult res{}; const float dist=sqrtf(gx*gx+gy*gy+gz*gz);
    res.delta_n=R-dist; res.dist=dist;
    if(res.delta_n<=0.0f) { res.delta_n=-1.0f; return res; }
    if(dist>1e-10f) { const float inv=1.0f/dist; res.normal_x=gx*inv; res.normal_y=gy*inv; res.normal_z=gz*inv; }
    else { res.normal_x=nx; res.normal_y=ny; res.normal_z=nz; }
    return res;
}

// =============================================================================
// Device Functions: Wall Velocity Interpolation
// =============================================================================

__device__ __forceinline__
void interpolate_wall_velocity(
    double& vx, double& vy, double& vz,
    float time,
    const float* __restrict__ d_times,
    const double* __restrict__ d_vx,
    const double* __restrict__ d_vy,
    const double* __restrict__ d_vz,
    int32_t num_frames
) {
    if (num_frames <= 0) {
        vx = 0.0f; vy = 0.0f; vz = 0.0f;
        return;
    }

    if (num_frames == 1 || time <= d_times[0]) {
        vx = d_vx[0];
        vy = d_vy[0];
        vz = d_vz[0];
        return;
    }

    if (time >= d_times[num_frames - 1]) {
        vx = d_vx[num_frames - 1];
        vy = d_vy[num_frames - 1];
        vz = d_vz[num_frames - 1];
        return;
    }

    // Binary search for time interval
    int lo = 0, hi = num_frames - 1;
    while (lo < hi - 1) {
        int mid = (lo + hi) / 2;
        if (d_times[mid] <= time) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation
    float t0 = d_times[lo];
    float t1 = d_times[hi];
    float alpha = (time - t0) / (t1 - t0);

    vx = d_vx[lo] + alpha * (d_vx[hi] - d_vx[lo]);
    vy = d_vy[lo] + alpha * (d_vy[hi] - d_vy[lo]);
    vz = d_vz[lo] + alpha * (d_vz[hi] - d_vz[lo]);
}

// =============================================================================
// Host Functions: Configuration Extraction
// =============================================================================

// Extract periodic boundary configuration from SimConfig
inline PeriodicBoundaryConfig extract_pbc_config(const SimConfig& config) {
    PeriodicBoundaryConfig pbc;

    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wall = config.walls[i];

        if (wall.type == WallType::PERIODICAL && wall.enabled) {
            switch (wall.axis) {
                case 0:  // X axis
                    pbc.periodic_x = true;
                    pbc.box_min_x = wall.bound_min;
                    pbc.box_max_x = wall.bound_max;
                    break;
                case 1:  // Y axis
                    pbc.periodic_y = true;
                    pbc.box_min_y = wall.bound_min;
                    pbc.box_max_y = wall.bound_max;
                    break;
                case 2:  // Z axis
                    pbc.periodic_z = true;
                    pbc.box_min_z = wall.bound_min;
                    pbc.box_max_z = wall.bound_max;
                    break;
            }
        }
    }

    pbc.compute_derived();
    return pbc;
}

// Convert WallConfig to WallData for GPU
inline WallData wall_config_to_data(const WallConfig& wc) {
    WallData wd;
    wd.id = wc.id;
    wd.material_id = wc.material_id;
    wd.type = wc.type;
    wd.enabled = wc.enabled;
    wd.is_smooth = false;  // Could add to WallConfig

    wd.velocity_x = wc.velocity.x;
    wd.velocity_y = wc.velocity.y;
    wd.velocity_z = wc.velocity.z;

    if (wc.type == WallType::PLAIN || wc.type == WallType::MOVING || wc.type == WallType::FINITE_PLAIN) {
        wd.plane.point_x = static_cast<float>(wc.position.x);
        wd.plane.point_y = static_cast<float>(wc.position.y);
        wd.plane.point_z = static_cast<float>(wc.position.z);

        // Normalize normal vector
        float nx = static_cast<float>(wc.normal.x);
        float ny = static_cast<float>(wc.normal.y);
        float nz = static_cast<float>(wc.normal.z);
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 1e-10f) {
            wd.plane.normal_x = nx / len;
            wd.plane.normal_y = ny / len;
            wd.plane.normal_z = nz / len;
        } else {
            wd.plane.normal_x = 0.0f;
            wd.plane.normal_y = 1.0f;
            wd.plane.normal_z = 0.0f;
        }
        wd.plane.compute_d();

        // Finite wall bounds
        wd.is_finite = (wc.type == WallType::FINITE_PLAIN);
        if (wd.is_finite) {
            wd.half_extent_x = wc.half_extents[0];
            wd.half_extent_y = wc.half_extents[1];
        }
    } else if (wc.type == WallType::PERIODICAL) {
        wd.periodic.axis = wc.axis;
        wd.periodic.bound_min = wc.bound_min;
        wd.periodic.bound_max = wc.bound_max;
        wd.periodic.compute_derived();
    } else if (wc.type == WallType::FUNNEL) {
        wd.funnel.point_x = static_cast<float>(wc.position.x);
        wd.funnel.point_y = static_cast<float>(wc.position.y);
        wd.funnel.point_z = static_cast<float>(wc.position.z);
        float ax = static_cast<float>(wc.normal.x);
        float ay = static_cast<float>(wc.normal.y);
        float az = static_cast<float>(wc.normal.z);
        float alen = sqrtf(ax * ax + ay * ay + az * az);
        if (alen > 1e-10f) {
            wd.funnel.axis_x = ax / alen;
            wd.funnel.axis_y = ay / alen;
            wd.funnel.axis_z = az / alen;
        } else {
            wd.funnel.axis_x = 0.0f;
            wd.funnel.axis_y = 0.0f;
            wd.funnel.axis_z = 1.0f;
        }
        wd.funnel.radius_bottom = wc.funnel_radius_bottom;
        wd.funnel.radius_top = wc.funnel_radius_top;
        wd.funnel.length = wc.funnel_length;
    }

    return wd;
}

// =============================================================================
// Host Functions: Memory Allocation
// =============================================================================

inline void allocate_wall_contact_history(
    WallContactHistoryBuffer& wch,
    int32_t num_particles,
    int32_t max_contacts_per_particle = 8
) {
    wch.max_contacts_per_particle = max_contacts_per_particle;
    wch.capacity = num_particles * max_contacts_per_particle;

    cudaMalloc(&wch.data, wch.capacity * sizeof(WallContactHistory));
    cudaMemset(wch.data, 0, wch.capacity * sizeof(WallContactHistory));

    // Initialize wall_id to -1 (inactive)
    std::vector<WallContactHistory> h_init(wch.capacity);
    for (auto& h : h_init) {
        h.wall_id = -1;
        h.is_active = false;
    }
    cudaMemcpy(wch.data, h_init.data(),
               wch.capacity * sizeof(WallContactHistory),
               cudaMemcpyHostToDevice);

    std::cout << "Allocated wall contact history: " << wch.capacity
              << " slots (" << max_contacts_per_particle << " per particle)"
              << std::endl;
}

inline void free_wall_contact_history(WallContactHistoryBuffer& wch) {
    if (wch.data) {
        cudaFree(wch.data);
    }
    wch = WallContactHistoryBuffer();
}

// =============================================================================
// Wall Statistics
// =============================================================================

struct WallStatistics {
    int32_t num_plain_walls;
    int32_t num_moving_walls;
    int32_t num_finite_plain_walls;
    int32_t num_funnel_walls;
    int32_t num_sphere_walls;
    int32_t num_periodic_axes;
    int32_t total_wall_contacts;

    WallStatistics()
        : num_plain_walls(0), num_moving_walls(0), num_finite_plain_walls(0),
          num_funnel_walls(0),
          num_sphere_walls(0),
          num_periodic_axes(0), total_wall_contacts(0) {}
};

inline WallStatistics compute_wall_statistics(const SimConfig& config) {
    WallStatistics stats;

    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];
        if (!wc.enabled) continue;

        switch (wc.type) {
            case WallType::PLAIN:
                stats.num_plain_walls++;
                break;
            case WallType::MOVING:
                stats.num_moving_walls++;
                break;
            case WallType::FINITE_PLAIN:
                stats.num_finite_plain_walls++;
                break;
            case WallType::PERIODICAL:
                stats.num_periodic_axes++;
                break;
            case WallType::FUNNEL:
                stats.num_funnel_walls++;
                break;
            case WallType::SPHERE:
                stats.num_sphere_walls++;
                break;
            default:
                break;
        }
    }

    return stats;
}

// =============================================================================
// Verification Functions (Host-only utilities)
// =============================================================================

inline void print_wall_summary(const SimConfig& config) {
    WallStatistics stats = compute_wall_statistics(config);

    std::cout << "=== Wall Summary ===" << std::endl;
    std::cout << "Plain walls:     " << stats.num_plain_walls << std::endl;
    std::cout << "Moving walls:    " << stats.num_moving_walls << std::endl;
    std::cout << "Finite walls:    " << stats.num_finite_plain_walls << std::endl;
    std::cout << "Periodic axes:   " << stats.num_periodic_axes << std::endl;
    std::cout << "Funnel walls:    " << stats.num_funnel_walls << std::endl;
    std::cout << "Sphere walls:    " << stats.num_sphere_walls << std::endl;
    std::cout << "===================" << std::endl;
}

inline void print_pbc_config(const PeriodicBoundaryConfig& pbc) {
    std::cout << "=== Periodic Boundary Config ===" << std::endl;

    std::cout << "Periodic X: " << (pbc.periodic_x ? "Yes" : "No");
    if (pbc.periodic_x) {
        std::cout << " [" << pbc.box_min_x << ", " << pbc.box_max_x << "]"
                  << " L=" << pbc.box_length_x;
    }
    std::cout << std::endl;

    std::cout << "Periodic Y: " << (pbc.periodic_y ? "Yes" : "No");
    if (pbc.periodic_y) {
        std::cout << " [" << pbc.box_min_y << ", " << pbc.box_max_y << "]"
                  << " L=" << pbc.box_length_y;
    }
    std::cout << std::endl;

    std::cout << "Periodic Z: " << (pbc.periodic_z ? "Yes" : "No");
    if (pbc.periodic_z) {
        std::cout << " [" << pbc.box_min_z << ", " << pbc.box_max_z << "]"
                  << " L=" << pbc.box_length_z;
    }
    std::cout << std::endl;
    std::cout << "================================" << std::endl;
}

inline bool validate_pbc_config(
    const PeriodicBoundaryConfig& pbc,
    float max_particle_radius
) {
    bool valid = true;

    if (pbc.periodic_x && pbc.box_length_x < 4.0f * max_particle_radius) {
        std::cerr << "WARNING: Periodic X domain too small for particle size" << std::endl;
        valid = false;
    }
    if (pbc.periodic_y && pbc.box_length_y < 4.0f * max_particle_radius) {
        std::cerr << "WARNING: Periodic Y domain too small for particle size" << std::endl;
        valid = false;
    }
    if (pbc.periodic_z && pbc.box_length_z < 4.0f * max_particle_radius) {
        std::cerr << "WARNING: Periodic Z domain too small for particle size" << std::endl;
        valid = false;
    }

    if (pbc.periodic_x && pbc.box_min_x >= pbc.box_max_x) {
        std::cerr << "ERROR: Periodic X bounds are invalid" << std::endl;
        valid = false;
    }
    if (pbc.periodic_y && pbc.box_min_y >= pbc.box_max_y) {
        std::cerr << "ERROR: Periodic Y bounds are invalid" << std::endl;
        valid = false;
    }
    if (pbc.periodic_z && pbc.box_min_z >= pbc.box_max_z) {
        std::cerr << "ERROR: Periodic Z bounds are invalid" << std::endl;
        valid = false;
    }

    return valid;
}

inline void print_walls_detail(const SimConfig& config) {
    std::cout << "=== Wall Details ===" << std::endl;

    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];

        std::cout << "Wall " << wc.id << ": ";

        switch (wc.type) {
            case WallType::PLAIN:
                std::cout << "PLAIN";
                break;
            case WallType::MOVING:
                std::cout << "MOVING";
                break;
            case WallType::FINITE_PLAIN:
                std::cout << "FINITE_PLAIN";
                break;
            case WallType::PERIODICAL:
                std::cout << "PERIODIC";
                break;
            case WallType::FUNNEL:
                std::cout << "FUNNEL";
                break;
            case WallType::MESH:
                std::cout << "MESH";
                break;
            case WallType::FREE:
                std::cout << "FREE";
                break;
            case WallType::SPHERE:
                std::cout << "SPHERE";
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }

        std::cout << (wc.enabled ? " [enabled]" : " [disabled]") << std::endl;

        if (wc.type == WallType::PLAIN || wc.type == WallType::MOVING) {
            std::cout << "  Point:  (" << wc.position.x << ", "
                      << wc.position.y << ", " << wc.position.z << ")" << std::endl;
            std::cout << "  Normal: (" << wc.normal.x << ", "
                      << wc.normal.y << ", " << wc.normal.z << ")" << std::endl;
            if (wc.type == WallType::MOVING) {
                std::cout << "  Velocity: (" << wc.velocity.x << ", "
                          << wc.velocity.y << ", " << wc.velocity.z << ")" << std::endl;
            }
        } else if (wc.type == WallType::FINITE_PLAIN) {
            std::cout << "  Point:  (" << wc.position.x << ", "
                      << wc.position.y << ", " << wc.position.z << ")" << std::endl;
            std::cout << "  Normal: (" << wc.normal.x << ", "
                      << wc.normal.y << ", " << wc.normal.z << ")" << std::endl;
            std::cout << "  Half-extents: (" << wc.half_extents[0] << ", "
                      << wc.half_extents[1] << ")" << std::endl;
        } else if (wc.type == WallType::PERIODICAL) {
            std::cout << "  Axis: " << (int)wc.axis
                      << " [" << wc.bound_min << ", " << wc.bound_max << "]" << std::endl;
        } else if (wc.type == WallType::FUNNEL) {
            std::cout << "  Point:  (" << wc.position.x << ", "
                      << wc.position.y << ", " << wc.position.z << ")" << std::endl;
            std::cout << "  Axis:   (" << wc.normal.x << ", "
                      << wc.normal.y << ", " << wc.normal.z << ")" << std::endl;
            std::cout << "  Radius bottom: " << wc.funnel_radius_bottom
                      << "  top: " << wc.funnel_radius_top
                      << "  length: " << wc.funnel_length << std::endl;
        } else if (wc.type == WallType::SPHERE) {
            std::cout << "  Radius: " << wc.sphere_radius
                      << "  Mass: " << wc.sphere_mass
                      << "  Trajectory: " << wc.sphere_trajectory_path << std::endl;
        }
    }

    std::cout << "====================" << std::endl;
}

// =============================================================================
// Forward Declarations for Implementation Files
// =============================================================================
// These are implemented in plain_wall.cu and periodical_boundary.cu
// =============================================================================

// From plain_wall.cu
class PlainWallDispatcher;

// From periodical_boundary.cu
class PeriodicBoundaryDispatcher;

// From trimesh_wall.cu
class TriMeshWallDispatcher;

// From sphere_wall.cu
class SphereWallManager;

// From funnel_wall.cu
void copy_funnel_to_current_device();

// =============================================================================
// Unified Wall Manager
// =============================================================================
// Orchestrates all wall types through a single interface
// =============================================================================

class WallManager {
public:
    // Moving wall velocity data storage
    struct MovingWallVelocityData {
        std::vector<float> times;
        std::vector<double> velocities_x;
        std::vector<double> velocities_y;
        std::vector<double> velocities_z;
        int32_t num_frames;

        float* d_times;
        double* d_vx;
        double* d_vy;
        double* d_vz;

        MovingWallVelocityData()
            : num_frames(0), d_times(nullptr),
              d_vx(nullptr), d_vy(nullptr), d_vz(nullptr) {}

        ~MovingWallVelocityData() {
            free_device();
        }

        void free_device() {
            if (d_times) cudaFree(d_times);
            if (d_vx) cudaFree(d_vx);
            if (d_vy) cudaFree(d_vy);
            if (d_vz) cudaFree(d_vz);
            d_times = nullptr;
            d_vx = nullptr;
            d_vy = nullptr;
            d_vz = nullptr;
        }

        void copy_to_device() {
            if (num_frames <= 0) return;

            size_t size = num_frames * sizeof(float);
            cudaMalloc(&d_times, size);
            cudaMalloc(&d_vx, size);
            cudaMalloc(&d_vy, size);
            cudaMalloc(&d_vz, size);

            cudaMemcpy(d_times, times.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vx, velocities_x.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vy, velocities_y.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vz, velocities_z.data(), size, cudaMemcpyHostToDevice);
        }

        // Load velocity data from binary file
        bool load_from_file(const char* filepath, WallType type) {
            if (!filepath || filepath[0] == '\0') return false;

            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Warning: Could not open velocity data file: " << filepath << std::endl;
                return false;
            }

            // Read number of frames
            int32_t n_frames;
            file.read(reinterpret_cast<char*>(&n_frames), sizeof(int32_t));
            if (n_frames <= 0 || n_frames > 100000) {
                std::cerr << "Warning: Invalid number of frames in velocity file: " << n_frames << std::endl;
                return false;
            }

            num_frames = n_frames;
            times.resize(num_frames);
            velocities_x.resize(num_frames);
            velocities_y.resize(num_frames);
            velocities_z.resize(num_frames);

            // Read time values
            file.read(reinterpret_cast<char*>(times.data()), num_frames * sizeof(float));

            // Read velocity components
            file.read(reinterpret_cast<char*>(velocities_x.data()), num_frames * sizeof(double));
            file.read(reinterpret_cast<char*>(velocities_y.data()), num_frames * sizeof(double));
            file.read(reinterpret_cast<char*>(velocities_z.data()), num_frames * sizeof(double));

            if (!file) {
                std::cerr << "Warning: Failed to read velocity data from file: " << filepath << std::endl;
                num_frames = 0;
                return false;
            }

            std::cout << "Loaded " << num_frames << " velocity frames from " << filepath << std::endl;
            return true;
        }
    };

private:
    static std::vector<MovingWallVelocityData> velocity_data_storage;
    static PeriodicBoundaryConfig pbc_config;
    static bool initialized;

public:
    // Initialize all wall subsystems
    static void initialize(
        SimulationState& state,
        const SimConfig& config
    );

    // Compute all wall contact forces
    static void compute_forces(
        SimulationState& state,
        const SimConfig& config,
        float dt,
        float current_time,
        cudaStream_t stream = 0
    );

    // Apply periodic boundary conditions (position wrapping)
    static void apply_periodic_boundaries(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    );

    // Cleanup wall contact history (remove inactive contacts)
    static void cleanup_contact_history(
        SimulationState& state,
        const SimConfig& config,
        cudaStream_t stream = 0
    );

    // Copy all wall-related __constant__ memory to the current CUDA device.
    // Must be called after cudaSetDevice() in multi-GPU per-device initialization
    // so each GPU has its own copy of wall data, PBC config, etc.
    // __constant__ memory is device-local and NOT shared via P2P.
    static void copy_to_current_device();

    // Get periodic boundary configuration
    static const PeriodicBoundaryConfig& get_pbc_config() {
        return pbc_config;
    }

    // Check if periodic boundaries are enabled
    static bool has_periodic_boundaries() {
        return pbc_config.is_enabled();
    }

    // Shutdown and free resources
    static void shutdown();
};

} // namespace cudadem

#endif // CUDADEM_WALL_H

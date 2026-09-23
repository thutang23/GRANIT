#ifndef CUDADEM_LINEAR_INTERPOLATION_CUH
#define CUDADEM_LINEAR_INTERPOLATION_CUH

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace cudadem {

// =============================================================================
// Linear ND Interpolator (GPU Version)
// =============================================================================
// Similar to scipy.interpolate.LinearNDInterpolator but for regular grids
// Uses CUDA texture hardware for efficient trilinear interpolation
// =============================================================================

// -----------------------------------------------------------------------------
// Boundary Handling Modes
// -----------------------------------------------------------------------------
enum class BoundaryMode : uint8_t {
    CLAMP = 0,           // Clamp coordinates to valid range
    WRAP = 1,            // Periodic wrap-around
    MIRROR = 2,          // Mirror at boundaries
    EXTRAPOLATE = 3      // Linear extrapolation (for time dimension)
};

// -----------------------------------------------------------------------------
// 3D Texture Interpolator (for space-only fields)
// -----------------------------------------------------------------------------
// Uses cudaTextureObject_t (modern API) for hardware-accelerated interpolation
// Data layout: [z][y][x] in memory, texture fetches with (x, y, z) coordinates
// -----------------------------------------------------------------------------
class LinearInterpolator3D {
public:
    // Grid parameters
    float origin_x, origin_y, origin_z;    // Grid origin
    float spacing_x, spacing_y, spacing_z; // Grid spacing
    int32_t dim_x, dim_y, dim_z;           // Grid dimensions

    // CUDA resources
    cudaArray_t cuda_array;
    cudaTextureObject_t tex_obj;

    // Boundary mode
    BoundaryMode boundary_mode;

    // Constructor
    LinearInterpolator3D()
        : origin_x(0), origin_y(0), origin_z(0),
          spacing_x(1), spacing_y(1), spacing_z(1),
          dim_x(0), dim_y(0), dim_z(0),
          cuda_array(nullptr), tex_obj(0),
          boundary_mode(BoundaryMode::CLAMP) {}

    ~LinearInterpolator3D() {
        destroy();
    }

    // Move constructor - transfer ownership of CUDA resources
    LinearInterpolator3D(LinearInterpolator3D&& other) noexcept
        : origin_x(other.origin_x), origin_y(other.origin_y), origin_z(other.origin_z),
          spacing_x(other.spacing_x), spacing_y(other.spacing_y), spacing_z(other.spacing_z),
          dim_x(other.dim_x), dim_y(other.dim_y), dim_z(other.dim_z),
          cuda_array(other.cuda_array), tex_obj(other.tex_obj),
          boundary_mode(other.boundary_mode) {
        // Clear source to prevent double-free
        other.cuda_array = nullptr;
        other.tex_obj = 0;
    }

    // Move assignment operator
    LinearInterpolator3D& operator=(LinearInterpolator3D&& other) noexcept {
        if (this != &other) {
            destroy();  // Clean up existing resources

            origin_x = other.origin_x;
            origin_y = other.origin_y;
            origin_z = other.origin_z;
            spacing_x = other.spacing_x;
            spacing_y = other.spacing_y;
            spacing_z = other.spacing_z;
            dim_x = other.dim_x;
            dim_y = other.dim_y;
            dim_z = other.dim_z;
            cuda_array = other.cuda_array;
            tex_obj = other.tex_obj;
            boundary_mode = other.boundary_mode;

            // Clear source to prevent double-free
            other.cuda_array = nullptr;
            other.tex_obj = 0;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment (owns CUDA resources)
    LinearInterpolator3D(const LinearInterpolator3D&) = delete;
    LinearInterpolator3D& operator=(const LinearInterpolator3D&) = delete;

    void destroy() {
        if (tex_obj != 0) {
            cudaDestroyTextureObject(tex_obj);
            tex_obj = 0;
        }
        if (cuda_array != nullptr) {
            cudaFreeArray(cuda_array);
            cuda_array = nullptr;
        }
    }

    // Initialize from host data
    void initialize(const float* host_data,
                   int32_t nx, int32_t ny, int32_t nz,
                   float ox, float oy, float oz,
                   float sx, float sy, float sz,
                   BoundaryMode mode = BoundaryMode::CLAMP) {
        destroy();

        dim_x = nx; dim_y = ny; dim_z = nz;
        origin_x = ox; origin_y = oy; origin_z = oz;
        spacing_x = sx; spacing_y = sy; spacing_z = sz;
        boundary_mode = mode;

        // Create 3D array
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(nx, ny, nz);

        cudaError_t err = cudaMalloc3DArray(&cuda_array, &channel_desc, extent);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate 3D CUDA array");
        }

        // Copy data to 3D array
        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(
            const_cast<float*>(host_data),
            nx * sizeof(float), nx, ny
        );
        params.dstArray = cuda_array;
        params.extent = extent;
        params.kind = cudaMemcpyHostToDevice;

        err = cudaMemcpy3D(&params);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to 3D CUDA array");
        }

        // Create texture object
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = get_address_mode(mode);
        tex_desc.addressMode[1] = get_address_mode(mode);
        tex_desc.addressMode[2] = get_address_mode(mode);
        tex_desc.filterMode = cudaFilterModeLinear;  // Hardware trilinear
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;  // Use unnormalized (absolute) coordinates

        err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create texture object");
        }

        std::cout << "LinearInterpolator3D initialized: "
                  << nx << "x" << ny << "x" << nz << " grid" << std::endl;
    }

private:
    static cudaTextureAddressMode get_address_mode(BoundaryMode mode) {
        switch (mode) {
            case BoundaryMode::WRAP: return cudaAddressModeWrap;
            case BoundaryMode::MIRROR: return cudaAddressModeMirror;
            case BoundaryMode::CLAMP:
            default: return cudaAddressModeClamp;
        }
    }
};

// -----------------------------------------------------------------------------
// Device Functions for 3D Interpolation
// -----------------------------------------------------------------------------

// Convert physical coordinates to texture coordinates
__device__ __forceinline__
float3 physical_to_texture_3d(
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    // Add 0.5 offset because texture coordinates are cell-centered
    return make_float3(
        (px - origin_x) * inv_spacing_x + 0.5f,
        (py - origin_y) * inv_spacing_y + 0.5f,
        (pz - origin_z) * inv_spacing_z + 0.5f
    );
}

// Interpolate at a single point (3D)
__device__ __forceinline__
float linear_interp_3d(
    cudaTextureObject_t tex,
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    float3 tc = physical_to_texture_3d(
        px, py, pz,
        origin_x, origin_y, origin_z,
        inv_spacing_x, inv_spacing_y, inv_spacing_z
    );
    return tex3D<float>(tex, tc.x, tc.y, tc.z);
}

// Interpolate 3-component vector at a single point (3D)
// Data layout: 3 separate textures for x, y, z components
__device__ __forceinline__
float3 linear_interp_3d_vec3(
    cudaTextureObject_t tex_x,
    cudaTextureObject_t tex_y,
    cudaTextureObject_t tex_z,
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    float3 tc = physical_to_texture_3d(
        px, py, pz,
        origin_x, origin_y, origin_z,
        inv_spacing_x, inv_spacing_y, inv_spacing_z
    );
    return make_float3(
        tex3D<float>(tex_x, tc.x, tc.y, tc.z),
        tex3D<float>(tex_y, tc.x, tc.y, tc.z),
        tex3D<float>(tex_z, tc.x, tc.y, tc.z)
    );
}

// =============================================================================
// 4D Interpolation (3D Space + 1D Time)
// =============================================================================
// For time-varying acceleration fields
// Uses double buffering: two 3D textures for consecutive time frames
// Time interpolation is done manually (linear) between the two frames
// =============================================================================

// Time boundary handling result
enum class TimeBoundaryResult : uint8_t {
    IN_RANGE = 0,           // Within data range
    BEFORE_START = 1,       // Before first frame
    AFTER_END = 2,          // After last frame
    EXTRAPOLATED = 3        // Extrapolated beyond data
};

// Double-buffered 4D interpolator
// Maintains two time frames and interpolates between them
class LinearInterpolator4D {
public:
    // Grid parameters (spatial)
    float origin_x, origin_y, origin_z;
    float spacing_x, spacing_y, spacing_z;
    int32_t dim_x, dim_y, dim_z;

    // Time parameters
    float time_start;                    // First frame time
    float time_interval;                 // Time between frames
    int32_t num_frames;                  // Total number of frames

    // Double buffer: two time frames
    cudaArray_t cuda_array_0;            // Frame at time t0
    cudaArray_t cuda_array_1;            // Frame at time t1
    cudaTextureObject_t tex_obj_0;
    cudaTextureObject_t tex_obj_1;

    // Current buffer state
    int32_t frame_index_0;               // Frame index in buffer 0
    int32_t frame_index_1;               // Frame index in buffer 1
    float time_0;                        // Time of frame 0
    float time_1;                        // Time of frame 1

    // Boundary handling
    BoundaryMode spatial_boundary;
    BoundaryMode time_boundary;

    LinearInterpolator4D()
        : origin_x(0), origin_y(0), origin_z(0),
          spacing_x(1), spacing_y(1), spacing_z(1),
          dim_x(0), dim_y(0), dim_z(0),
          time_start(0), time_interval(1), num_frames(0),
          cuda_array_0(nullptr), cuda_array_1(nullptr),
          tex_obj_0(0), tex_obj_1(0),
          frame_index_0(-1), frame_index_1(-1),
          time_0(0), time_1(0),
          spatial_boundary(BoundaryMode::CLAMP),
          time_boundary(BoundaryMode::EXTRAPOLATE) {}

    ~LinearInterpolator4D() {
        destroy();
    }

    // Move constructor - transfer ownership of CUDA resources
    LinearInterpolator4D(LinearInterpolator4D&& other) noexcept
        : origin_x(other.origin_x), origin_y(other.origin_y), origin_z(other.origin_z),
          spacing_x(other.spacing_x), spacing_y(other.spacing_y), spacing_z(other.spacing_z),
          dim_x(other.dim_x), dim_y(other.dim_y), dim_z(other.dim_z),
          time_start(other.time_start), time_interval(other.time_interval), num_frames(other.num_frames),
          cuda_array_0(other.cuda_array_0), cuda_array_1(other.cuda_array_1),
          tex_obj_0(other.tex_obj_0), tex_obj_1(other.tex_obj_1),
          frame_index_0(other.frame_index_0), frame_index_1(other.frame_index_1),
          time_0(other.time_0), time_1(other.time_1),
          spatial_boundary(other.spatial_boundary),
          time_boundary(other.time_boundary) {
        // Clear source to prevent double-free
        other.cuda_array_0 = nullptr;
        other.cuda_array_1 = nullptr;
        other.tex_obj_0 = 0;
        other.tex_obj_1 = 0;
    }

    // Move assignment operator
    LinearInterpolator4D& operator=(LinearInterpolator4D&& other) noexcept {
        if (this != &other) {
            destroy();  // Clean up existing resources

            origin_x = other.origin_x;
            origin_y = other.origin_y;
            origin_z = other.origin_z;
            spacing_x = other.spacing_x;
            spacing_y = other.spacing_y;
            spacing_z = other.spacing_z;
            dim_x = other.dim_x;
            dim_y = other.dim_y;
            dim_z = other.dim_z;
            time_start = other.time_start;
            time_interval = other.time_interval;
            num_frames = other.num_frames;
            cuda_array_0 = other.cuda_array_0;
            cuda_array_1 = other.cuda_array_1;
            tex_obj_0 = other.tex_obj_0;
            tex_obj_1 = other.tex_obj_1;
            frame_index_0 = other.frame_index_0;
            frame_index_1 = other.frame_index_1;
            time_0 = other.time_0;
            time_1 = other.time_1;
            spatial_boundary = other.spatial_boundary;
            time_boundary = other.time_boundary;

            // Clear source to prevent double-free
            other.cuda_array_0 = nullptr;
            other.cuda_array_1 = nullptr;
            other.tex_obj_0 = 0;
            other.tex_obj_1 = 0;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment (owns CUDA resources)
    LinearInterpolator4D(const LinearInterpolator4D&) = delete;
    LinearInterpolator4D& operator=(const LinearInterpolator4D&) = delete;

    void destroy() {
        if (tex_obj_0 != 0) cudaDestroyTextureObject(tex_obj_0);
        if (tex_obj_1 != 0) cudaDestroyTextureObject(tex_obj_1);
        if (cuda_array_0 != nullptr) cudaFreeArray(cuda_array_0);
        if (cuda_array_1 != nullptr) cudaFreeArray(cuda_array_1);
        tex_obj_0 = 0; tex_obj_1 = 0;
        cuda_array_0 = nullptr; cuda_array_1 = nullptr;
    }

    // Initialize with grid parameters (allocate buffers)
    void initialize(
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz,
        float t_start, float t_interval, int32_t n_frames,
        BoundaryMode spatial_mode = BoundaryMode::CLAMP,
        BoundaryMode time_mode = BoundaryMode::EXTRAPOLATE
    ) {
        destroy();

        dim_x = nx; dim_y = ny; dim_z = nz;
        origin_x = ox; origin_y = oy; origin_z = oz;
        spacing_x = sx; spacing_y = sy; spacing_z = sz;
        time_start = t_start;
        time_interval = t_interval;
        num_frames = n_frames;
        spatial_boundary = spatial_mode;
        time_boundary = time_mode;

        // Allocate two 3D arrays for double buffering
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(nx, ny, nz);

        cudaMalloc3DArray(&cuda_array_0, &channel_desc, extent);
        cudaMalloc3DArray(&cuda_array_1, &channel_desc, extent);

        // Create texture objects
        create_texture_object(&tex_obj_0, cuda_array_0, spatial_mode);
        create_texture_object(&tex_obj_1, cuda_array_1, spatial_mode);

        frame_index_0 = -1;
        frame_index_1 = -1;

        std::cout << "LinearInterpolator4D initialized: "
                  << nx << "x" << ny << "x" << nz << " x " << n_frames << " frames"
                  << std::endl;
    }

    // Load a specific time frame into one of the buffers
    // Returns which buffer (0 or 1) was used
    int load_frame(const float* host_data, int32_t frame_index) {
        if (frame_index < 0 || frame_index >= num_frames) {
            throw std::runtime_error("Frame index out of range");
        }

        // Decide which buffer to use
        int buffer_to_use;
        if (frame_index_0 < 0) {
            buffer_to_use = 0;
        } else if (frame_index_1 < 0) {
            buffer_to_use = 1;
        } else if (frame_index_0 < frame_index_1) {
            buffer_to_use = 0;  // Replace older frame
        } else {
            buffer_to_use = 1;
        }

        // Copy data to the selected buffer
        cudaArray_t target_array = (buffer_to_use == 0) ? cuda_array_0 : cuda_array_1;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(
            const_cast<float*>(host_data),
            dim_x * sizeof(float), dim_x, dim_y
        );
        params.dstArray = target_array;
        params.extent = make_cudaExtent(dim_x, dim_y, dim_z);
        params.kind = cudaMemcpyHostToDevice;

        cudaMemcpy3D(&params);

        // Update buffer state
        float frame_time = time_start + frame_index * time_interval;
        if (buffer_to_use == 0) {
            frame_index_0 = frame_index;
            time_0 = frame_time;
        } else {
            frame_index_1 = frame_index;
            time_1 = frame_time;
        }

        return buffer_to_use;
    }

    // Ensure the correct frames are loaded for a given time
    // Returns TimeBoundaryResult indicating if time is in range
    TimeBoundaryResult prepare_for_time(
        double query_time,
        const std::function<void(int32_t, float*)>& frame_loader  // Function to load a frame
    ) {
        float t = static_cast<float>(query_time);
        float end_time = time_start + (num_frames - 1) * time_interval;

        // Check bounds
        if (t < time_start) {
            if (time_boundary == BoundaryMode::CLAMP) {
                // Load first frame into both buffers
                if (frame_index_0 != 0 || frame_index_1 != 0) {
                    // Need to reload
                    std::vector<float> frame_data(dim_x * dim_y * dim_z);
                    frame_loader(0, frame_data.data());
                    load_frame(frame_data.data(), 0);
                    load_frame(frame_data.data(), 0);  // Same frame in both buffers
                }
                return TimeBoundaryResult::BEFORE_START;
            }
            return TimeBoundaryResult::BEFORE_START;
        }

        if (t >= end_time) {
            if (time_boundary == BoundaryMode::CLAMP) {
                // Load last frame into both buffers
                int last_frame = num_frames - 1;
                if (frame_index_0 != last_frame || frame_index_1 != last_frame) {
                    std::vector<float> frame_data(dim_x * dim_y * dim_z);
                    frame_loader(last_frame, frame_data.data());
                    load_frame(frame_data.data(), last_frame);
                }
                return TimeBoundaryResult::AFTER_END;
            } else if (time_boundary == BoundaryMode::EXTRAPOLATE) {
                // Keep the last two frames for extrapolation
                int last_frame = num_frames - 1;
                int prev_frame = num_frames - 2;
                ensure_frames_loaded(prev_frame, last_frame, frame_loader);
                return TimeBoundaryResult::EXTRAPOLATED;
            }
            return TimeBoundaryResult::AFTER_END;
        }

        // Calculate which frames bracket the query time
        float relative_time = (t - time_start) / time_interval;
        int32_t frame_lo = static_cast<int32_t>(relative_time);
        int32_t frame_hi = frame_lo + 1;

        if (frame_hi >= num_frames) {
            frame_hi = num_frames - 1;
        }

        ensure_frames_loaded(frame_lo, frame_hi, frame_loader);

        return TimeBoundaryResult::IN_RANGE;
    }

    // Get interpolation parameters for current time
    // Returns: alpha (interpolation factor between frame_0 and frame_1)
    float get_time_interpolation_factor(double query_time) const {
        float t = static_cast<float>(query_time);

        // Handle clamping at boundaries
        if (t <= time_0) return 0.0f;
        if (t >= time_1) return 1.0f;

        // Linear interpolation factor
        return (t - time_0) / (time_1 - time_0);
    }

private:
    void create_texture_object(
        cudaTextureObject_t* tex_obj,
        cudaArray_t cuda_array,
        BoundaryMode mode
    ) {
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        cudaTextureAddressMode addr_mode;
        switch (mode) {
            case BoundaryMode::WRAP: addr_mode = cudaAddressModeWrap; break;
            case BoundaryMode::MIRROR: addr_mode = cudaAddressModeMirror; break;
            default: addr_mode = cudaAddressModeClamp; break;
        }
        tex_desc.addressMode[0] = addr_mode;
        tex_desc.addressMode[1] = addr_mode;
        tex_desc.addressMode[2] = addr_mode;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, nullptr);
    }

    void ensure_frames_loaded(
        int32_t frame_lo, int32_t frame_hi,
        const std::function<void(int32_t, float*)>& frame_loader
    ) {
        // Check if we already have the right frames
        bool has_lo = (frame_index_0 == frame_lo || frame_index_1 == frame_lo);
        bool has_hi = (frame_index_0 == frame_hi || frame_index_1 == frame_hi);

        std::vector<float> frame_data(dim_x * dim_y * dim_z);

        if (!has_lo) {
            frame_loader(frame_lo, frame_data.data());
            load_frame(frame_data.data(), frame_lo);
        }

        if (!has_hi && frame_lo != frame_hi) {
            frame_loader(frame_hi, frame_data.data());
            load_frame(frame_data.data(), frame_hi);
        }

        // Ensure frame_0 is the earlier frame
        if (frame_index_1 < frame_index_0) {
            std::swap(frame_index_0, frame_index_1);
            std::swap(time_0, time_1);
            std::swap(cuda_array_0, cuda_array_1);
            std::swap(tex_obj_0, tex_obj_1);
        }
    }
};

// -----------------------------------------------------------------------------
// Device Functions for 4D Interpolation (Space + Time)
// -----------------------------------------------------------------------------

// Interpolate at a single point with time interpolation between two frames
__device__ __forceinline__
float linear_interp_4d(
    cudaTextureObject_t tex_0,    // Earlier time frame
    cudaTextureObject_t tex_1,    // Later time frame
    float alpha,                   // Time interpolation factor [0, 1]
    float px, float py, float pz,  // Physical coordinates
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    float3 tc = physical_to_texture_3d(
        px, py, pz,
        origin_x, origin_y, origin_z,
        inv_spacing_x, inv_spacing_y, inv_spacing_z
    );

    float val_0 = tex3D<float>(tex_0, tc.x, tc.y, tc.z);
    float val_1 = tex3D<float>(tex_1, tc.x, tc.y, tc.z);

    // Linear interpolation in time
    return val_0 + alpha * (val_1 - val_0);
}

// Interpolate 3-component vector with time interpolation
__device__ __forceinline__
float3 linear_interp_4d_vec3(
    cudaTextureObject_t tex_x_0, cudaTextureObject_t tex_x_1,
    cudaTextureObject_t tex_y_0, cudaTextureObject_t tex_y_1,
    cudaTextureObject_t tex_z_0, cudaTextureObject_t tex_z_1,
    float alpha,
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    float3 tc = physical_to_texture_3d(
        px, py, pz,
        origin_x, origin_y, origin_z,
        inv_spacing_x, inv_spacing_y, inv_spacing_z
    );

    float vx_0 = tex3D<float>(tex_x_0, tc.x, tc.y, tc.z);
    float vx_1 = tex3D<float>(tex_x_1, tc.x, tc.y, tc.z);
    float vy_0 = tex3D<float>(tex_y_0, tc.x, tc.y, tc.z);
    float vy_1 = tex3D<float>(tex_y_1, tc.x, tc.y, tc.z);
    float vz_0 = tex3D<float>(tex_z_0, tc.x, tc.y, tc.z);
    float vz_1 = tex3D<float>(tex_z_1, tc.x, tc.y, tc.z);

    return make_float3(
        vx_0 + alpha * (vx_1 - vx_0),
        vy_0 + alpha * (vy_1 - vy_0),
        vz_0 + alpha * (vz_1 - vz_0)
    );
}

// =============================================================================
// Vector Field Interpolators (for 3-component acceleration fields)
// =============================================================================

// 3D Vector field with 3 components (ax, ay, az)
struct VectorField3D {
    LinearInterpolator3D interp_x;
    LinearInterpolator3D interp_y;
    LinearInterpolator3D interp_z;

    // Cached inverse spacing for performance
    float inv_spacing_x, inv_spacing_y, inv_spacing_z;

    void initialize(
        const float* data_x, const float* data_y, const float* data_z,
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz,
        BoundaryMode mode = BoundaryMode::CLAMP
    ) {
        interp_x.initialize(data_x, nx, ny, nz, ox, oy, oz, sx, sy, sz, mode);
        interp_y.initialize(data_y, nx, ny, nz, ox, oy, oz, sx, sy, sz, mode);
        interp_z.initialize(data_z, nx, ny, nz, ox, oy, oz, sx, sy, sz, mode);

        inv_spacing_x = 1.0f / sx;
        inv_spacing_y = 1.0f / sy;
        inv_spacing_z = 1.0f / sz;
    }

    void destroy() {
        interp_x.destroy();
        interp_y.destroy();
        interp_z.destroy();
    }
};

// 4D Vector field (time-varying)
struct VectorField4D {
    LinearInterpolator4D interp_x;
    LinearInterpolator4D interp_y;
    LinearInterpolator4D interp_z;

    float inv_spacing_x, inv_spacing_y, inv_spacing_z;

    void initialize(
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz,
        float t_start, float t_interval, int32_t n_frames,
        BoundaryMode spatial_mode = BoundaryMode::CLAMP,
        BoundaryMode time_mode = BoundaryMode::EXTRAPOLATE
    ) {
        interp_x.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, spatial_mode, time_mode);
        interp_y.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, spatial_mode, time_mode);
        interp_z.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, spatial_mode, time_mode);

        inv_spacing_x = 1.0f / sx;
        inv_spacing_y = 1.0f / sy;
        inv_spacing_z = 1.0f / sz;
    }

    void destroy() {
        interp_x.destroy();
        interp_y.destroy();
        interp_z.destroy();
    }

    // Get time interpolation factor
    float get_time_alpha(double query_time) const {
        return interp_x.get_time_interpolation_factor(query_time);
    }
};

// =============================================================================
// GPU Constant Memory for Interpolator Parameters
// =============================================================================
// Store frequently used interpolator parameters in constant memory
// for faster access in kernels
// =============================================================================

struct InterpolatorParams3D {
    cudaTextureObject_t tex_x;
    cudaTextureObject_t tex_y;
    cudaTextureObject_t tex_z;
    float origin_x, origin_y, origin_z;
    float inv_spacing_x, inv_spacing_y, inv_spacing_z;
    int32_t dim_x, dim_y, dim_z;
};

struct InterpolatorParams4D {
    // Frame 0 (earlier time)
    cudaTextureObject_t tex_x_0;
    cudaTextureObject_t tex_y_0;
    cudaTextureObject_t tex_z_0;
    // Frame 1 (later time)
    cudaTextureObject_t tex_x_1;
    cudaTextureObject_t tex_y_1;
    cudaTextureObject_t tex_z_1;
    // Grid parameters
    float origin_x, origin_y, origin_z;
    float inv_spacing_x, inv_spacing_y, inv_spacing_z;
    int32_t dim_x, dim_y, dim_z;
    // Time interpolation
    float time_alpha;
};

// Helper to extract params from VectorField3D
inline InterpolatorParams3D get_interp_params(const VectorField3D& field) {
    InterpolatorParams3D params;
    params.tex_x = field.interp_x.tex_obj;
    params.tex_y = field.interp_y.tex_obj;
    params.tex_z = field.interp_z.tex_obj;
    params.origin_x = field.interp_x.origin_x;
    params.origin_y = field.interp_x.origin_y;
    params.origin_z = field.interp_x.origin_z;
    params.inv_spacing_x = field.inv_spacing_x;
    params.inv_spacing_y = field.inv_spacing_y;
    params.inv_spacing_z = field.inv_spacing_z;
    params.dim_x = field.interp_x.dim_x;
    params.dim_y = field.interp_x.dim_y;
    params.dim_z = field.interp_x.dim_z;
    return params;
}

// Helper to extract params from VectorField4D
inline InterpolatorParams4D get_interp_params(const VectorField4D& field, double query_time) {
    InterpolatorParams4D params;
    params.tex_x_0 = field.interp_x.tex_obj_0;
    params.tex_y_0 = field.interp_y.tex_obj_0;
    params.tex_z_0 = field.interp_z.tex_obj_0;
    params.tex_x_1 = field.interp_x.tex_obj_1;
    params.tex_y_1 = field.interp_y.tex_obj_1;
    params.tex_z_1 = field.interp_z.tex_obj_1;
    params.origin_x = field.interp_x.origin_x;
    params.origin_y = field.interp_x.origin_y;
    params.origin_z = field.interp_x.origin_z;
    params.inv_spacing_x = field.inv_spacing_x;
    params.inv_spacing_y = field.inv_spacing_y;
    params.inv_spacing_z = field.inv_spacing_z;
    params.dim_x = field.interp_x.dim_x;
    params.dim_y = field.interp_x.dim_y;
    params.dim_z = field.interp_x.dim_z;
    params.time_alpha = field.get_time_alpha(query_time);
    return params;
}

// =============================================================================
// Kernel-callable interpolation functions
// =============================================================================

// Interpolate 3D vector field at particle position
__device__ __forceinline__
float3 interp_vector_field_3d(
    const InterpolatorParams3D& params,
    float px, float py, float pz
) {
    return linear_interp_3d_vec3(
        params.tex_x, params.tex_y, params.tex_z,
        px, py, pz,
        params.origin_x, params.origin_y, params.origin_z,
        params.inv_spacing_x, params.inv_spacing_y, params.inv_spacing_z
    );
}

// Interpolate 4D (time-varying) vector field at particle position
__device__ __forceinline__
float3 interp_vector_field_4d(
    const InterpolatorParams4D& params,
    float px, float py, float pz
) {
    return linear_interp_4d_vec3(
        params.tex_x_0, params.tex_x_1,
        params.tex_y_0, params.tex_y_1,
        params.tex_z_0, params.tex_z_1,
        params.time_alpha,
        px, py, pz,
        params.origin_x, params.origin_y, params.origin_z,
        params.inv_spacing_x, params.inv_spacing_y, params.inv_spacing_z
    );
}

} // namespace cudadem

#endif // CUDADEM_LINEAR_INTERPOLATION_CUH

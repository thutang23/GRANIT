#ifndef CUDADEM_CUBIC_INTERPOLATION_CUH
#define CUDADEM_CUBIC_INTERPOLATION_CUH

// =============================================================================
// Cubic B-Spline Interpolation for CudaDEM
// =============================================================================
// Uses CubicInterpolationCUDA library for high-accuracy 3D interpolation
// Provides C² continuous interpolation with gradient computation
//
// References:
// - Daniel Ruijters and Philippe Thévenaz, "GPU Prefilter for Accurate Cubic
//   B-Spline Interpolation", The Computer Journal, 2012
// - Daniel Ruijters et al., "Efficient GPU-Based Texture Interpolation using
//   Uniform B-Splines", Journal of Graphics Tools, 2008
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

namespace cudadem {

// Forward declarations from linear_interpolation.cuh
enum class BoundaryMode : uint8_t;

// =============================================================================
// Cubic B-Spline Kernel Functions (from CubicInterpolationCUDA)
// =============================================================================

// B-spline basis function (β³)
__device__ __forceinline__
float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) {
        return (2.0f / 3.0f) - 0.5f * t * t * (2.0f - t);
    } else if (t < 2.0f) {
        float tmp = 2.0f - t;
        return (1.0f / 6.0f) * tmp * tmp * tmp;
    }
    return 0.0f;
}

// Compute B-spline weights for cubic interpolation
__device__ __forceinline__
void bspline_weights(float fraction, float& w0, float& w1, float& w2, float& w3) {
    float one_frac = 1.0f - fraction;
    float squared = fraction * fraction;
    float one_squared = one_frac * one_frac;

    w0 = (1.0f / 6.0f) * one_squared * one_frac;
    w1 = (2.0f / 3.0f) - 0.5f * squared * (2.0f - fraction);
    w2 = (2.0f / 3.0f) - 0.5f * one_squared * (2.0f - one_frac);
    w3 = (1.0f / 6.0f) * squared * fraction;
}

// Compute B-spline weights for first derivative
__device__ __forceinline__
void bspline_weights_1st_derivative(float fraction, float& w0, float& w1, float& w2, float& w3) {
    float one_frac = 1.0f - fraction;
    float squared = fraction * fraction;
    float one_squared = one_frac * one_frac;

    w0 = -0.5f * one_squared;
    w1 = 0.5f * (3.0f * squared - 4.0f * fraction);
    w2 = 0.5f * (-3.0f * one_squared + 4.0f * one_frac);
    w3 = 0.5f * squared;
}

// =============================================================================
// Prefiltering for Cubic B-Spline Coefficients
// =============================================================================
// Converts sample values to B-spline coefficients via recursive filtering
// CRITICAL: Must be called before interpolation for accurate results
// =============================================================================

// Pole for cubic B-spline (α = -2 + √3)
constexpr float BSPLINE_POLE = -0.2679491924311227f;

// Causal filter kernel
__global__ void causal_filter_x(
    float* volume,
    int pitch_floats,      // Pitch in number of floats
    int width, int height, int depth
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || z >= depth) return;

    float* row = volume + z * height * pitch_floats + y * pitch_floats;

    // Forward pass (causal)
    float pole = BSPLINE_POLE;
    float c_prev = row[0];
    row[0] = c_prev / (1.0f - pole);

    for (int x = 1; x < width; ++x) {
        row[x] = row[x] + pole * row[x - 1];
    }
}

__global__ void anticausal_filter_x(
    float* volume,
    int pitch_floats,
    int width, int height, int depth
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || z >= depth) return;

    float* row = volume + z * height * pitch_floats + y * pitch_floats;

    // Backward pass (anticausal)
    float pole = BSPLINE_POLE;
    row[width - 1] = (pole / (pole * pole - 1.0f)) *
                     (pole * row[width - 1] + row[width - 1]);

    for (int x = width - 2; x >= 0; --x) {
        row[x] = pole * (row[x + 1] - row[x]);
    }
}

__global__ void causal_filter_y(
    float* volume,
    int pitch_floats,
    int width, int height, int depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || z >= depth) return;

    float pole = BSPLINE_POLE;
    int slice_offset = z * height * pitch_floats;

    // Get initial value
    float* col = volume + slice_offset + x;
    float c_prev = col[0];
    col[0] = c_prev / (1.0f - pole);

    for (int y = 1; y < height; ++y) {
        int idx = y * pitch_floats;
        col[idx] = col[idx] + pole * col[(y - 1) * pitch_floats];
    }
}

__global__ void anticausal_filter_y(
    float* volume,
    int pitch_floats,
    int width, int height, int depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || z >= depth) return;

    float pole = BSPLINE_POLE;
    int slice_offset = z * height * pitch_floats;
    float* col = volume + slice_offset + x;

    int last = (height - 1) * pitch_floats;
    col[last] = (pole / (pole * pole - 1.0f)) *
                (pole * col[last] + col[last]);

    for (int y = height - 2; y >= 0; --y) {
        int idx = y * pitch_floats;
        col[idx] = pole * (col[(y + 1) * pitch_floats] - col[idx]);
    }
}

__global__ void causal_filter_z(
    float* volume,
    int pitch_floats,
    int width, int height, int depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float pole = BSPLINE_POLE;
    int row_offset = y * pitch_floats + x;
    int slice_size = height * pitch_floats;

    float c_prev = volume[row_offset];
    volume[row_offset] = c_prev / (1.0f - pole);

    for (int z = 1; z < depth; ++z) {
        int idx = z * slice_size + row_offset;
        volume[idx] = volume[idx] + pole * volume[(z - 1) * slice_size + row_offset];
    }
}

__global__ void anticausal_filter_z(
    float* volume,
    int pitch_floats,
    int width, int height, int depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float pole = BSPLINE_POLE;
    int row_offset = y * pitch_floats + x;
    int slice_size = height * pitch_floats;

    int last_z = (depth - 1) * slice_size + row_offset;
    volume[last_z] = (pole / (pole * pole - 1.0f)) *
                     (pole * volume[last_z] + volume[last_z]);

    for (int z = depth - 2; z >= 0; --z) {
        int idx = z * slice_size + row_offset;
        volume[idx] = pole * (volume[(z + 1) * slice_size + row_offset] - volume[idx]);
    }
}

// Host function to apply 3D prefiltering
inline void apply_cubic_prefilter_3d(
    float* d_volume,
    int pitch_floats,
    int width, int height, int depth,
    int block_size = 16
) {
    dim3 block(block_size, block_size);

    // X direction
    dim3 grid_x((height + block.x - 1) / block.x,
                (depth + block.y - 1) / block.y);
    causal_filter_x<<<grid_x, block>>>(d_volume, pitch_floats, width, height, depth);
    anticausal_filter_x<<<grid_x, block>>>(d_volume, pitch_floats, width, height, depth);

    // Y direction
    dim3 grid_y((width + block.x - 1) / block.x,
                (depth + block.y - 1) / block.y);
    causal_filter_y<<<grid_y, block>>>(d_volume, pitch_floats, width, height, depth);
    anticausal_filter_y<<<grid_y, block>>>(d_volume, pitch_floats, width, height, depth);

    // Z direction
    dim3 grid_z((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);
    causal_filter_z<<<grid_z, block>>>(d_volume, pitch_floats, width, height, depth);
    anticausal_filter_z<<<grid_z, block>>>(d_volume, pitch_floats, width, height, depth);

    cudaDeviceSynchronize();
}

// =============================================================================
// Cubic Interpolation using 8 Trilinear Lookups (Fast Method)
// =============================================================================
// Uses the key insight that cubic B-spline can be computed as weighted sum
// of 8 trilinear lookups, leveraging hardware texture filtering
// =============================================================================

// Compute optimized texture lookup positions and weights for fast cubic
__device__ __forceinline__
void cubic_bspline_fast_params(
    float fraction,
    float& g0, float& g1,      // Weights for texture fetches
    float& h0, float& h1       // Offset adjustments
) {
    // B-spline weights
    float w0, w1, w2, w3;
    bspline_weights(fraction, w0, w1, w2, w3);

    // Combine weights for texture lookups
    g0 = w0 + w1;
    g1 = w2 + w3;

    // Compute offsets (relative to floor index)
    h0 = (w1 / g0) - 1.0f;      // First lookup offset
    h1 = (w3 / g1) + 1.0f;      // Second lookup offset
}

// Fast cubic interpolation using 8 trilinear texture lookups
__device__ __forceinline__
float cubic_interp_3d_fast(
    cudaTextureObject_t tex,
    float x, float y, float z,
    int dim_x, int dim_y, int dim_z
) {
    // Transform to grid-aligned coordinates
    float3 coord = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
    float3 index = make_float3(floorf(coord.x), floorf(coord.y), floorf(coord.z));
    float3 fraction = make_float3(coord.x - index.x, coord.y - index.y, coord.z - index.z);

    // Compute fast cubic parameters for each dimension
    float g0x, g1x, h0x, h1x;
    float g0y, g1y, h0y, h1y;
    float g0z, g1z, h0z, h1z;

    cubic_bspline_fast_params(fraction.x, g0x, g1x, h0x, h1x);
    cubic_bspline_fast_params(fraction.y, g0y, g1y, h0y, h1y);
    cubic_bspline_fast_params(fraction.z, g0z, g1z, h0z, h1z);

    // Compute texture coordinates (add 0.5 for cell center)
    float x0 = index.x + h0x + 0.5f;
    float x1 = index.x + h1x + 0.5f;
    float y0 = index.y + h0y + 0.5f;
    float y1 = index.y + h1y + 0.5f;
    float z0 = index.z + h0z + 0.5f;
    float z1 = index.z + h1z + 0.5f;

    // 8 trilinear texture lookups
    float tex000 = tex3D<float>(tex, x0, y0, z0);
    float tex100 = tex3D<float>(tex, x1, y0, z0);
    float tex010 = tex3D<float>(tex, x0, y1, z0);
    float tex110 = tex3D<float>(tex, x1, y1, z0);
    float tex001 = tex3D<float>(tex, x0, y0, z1);
    float tex101 = tex3D<float>(tex, x1, y0, z1);
    float tex011 = tex3D<float>(tex, x0, y1, z1);
    float tex111 = tex3D<float>(tex, x1, y1, z1);

    // Combine with weights
    float result_z0 = g0y * (g0x * tex000 + g1x * tex100) +
                      g1y * (g0x * tex010 + g1x * tex110);
    float result_z1 = g0y * (g0x * tex001 + g1x * tex101) +
                      g1y * (g0x * tex011 + g1x * tex111);

    return g0z * result_z0 + g1z * result_z1;
}

// =============================================================================
// Simple Cubic Interpolation (Reference Implementation)
// =============================================================================
// Uses 64 nearest-neighbor lookups - slower but useful for validation
// =============================================================================

__device__ __forceinline__
float cubic_interp_3d_simple(
    cudaTextureObject_t tex,
    float x, float y, float z,
    int dim_x, int dim_y, int dim_z
) {
    // Transform to grid-aligned coordinates
    float3 coord = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
    float3 index = make_float3(floorf(coord.x), floorf(coord.y), floorf(coord.z));
    float3 fraction = make_float3(coord.x - index.x, coord.y - index.y, coord.z - index.z);

    // B-spline weights for each dimension
    float wx[4], wy[4], wz[4];
    bspline_weights(fraction.x, wx[0], wx[1], wx[2], wx[3]);
    bspline_weights(fraction.y, wy[0], wy[1], wy[2], wy[3]);
    bspline_weights(fraction.z, wz[0], wz[1], wz[2], wz[3]);

    // Sum over 4x4x4 neighborhood
    float result = 0.0f;
    for (int iz = -1; iz <= 2; ++iz) {
        float w_z = wz[iz + 1];
        float zc = index.z + iz + 0.5f;

        for (int iy = -1; iy <= 2; ++iy) {
            float w_yz = w_z * wy[iy + 1];
            float yc = index.y + iy + 0.5f;

            for (int ix = -1; ix <= 2; ++ix) {
                float w_xyz = w_yz * wx[ix + 1];
                float xc = index.x + ix + 0.5f;

                result += w_xyz * tex3D<float>(tex, xc, yc, zc);
            }
        }
    }

    return result;
}

// =============================================================================
// Cubic Interpolation with Gradient (First Derivative)
// =============================================================================

__device__ __forceinline__
float4 cubic_interp_3d_with_gradient(
    cudaTextureObject_t tex,
    float x, float y, float z,
    int dim_x, int dim_y, int dim_z
) {
    // Transform coordinates
    float3 coord = make_float3(x - 0.5f, y - 0.5f, z - 0.5f);
    float3 index = make_float3(floorf(coord.x), floorf(coord.y), floorf(coord.z));
    float3 fraction = make_float3(coord.x - index.x, coord.y - index.y, coord.z - index.z);

    // B-spline weights
    float wx[4], wy[4], wz[4];
    float dwx[4], dwy[4], dwz[4];  // Derivative weights

    bspline_weights(fraction.x, wx[0], wx[1], wx[2], wx[3]);
    bspline_weights(fraction.y, wy[0], wy[1], wy[2], wy[3]);
    bspline_weights(fraction.z, wz[0], wz[1], wz[2], wz[3]);

    bspline_weights_1st_derivative(fraction.x, dwx[0], dwx[1], dwx[2], dwx[3]);
    bspline_weights_1st_derivative(fraction.y, dwy[0], dwy[1], dwy[2], dwy[3]);
    bspline_weights_1st_derivative(fraction.z, dwz[0], dwz[1], dwz[2], dwz[3]);

    float value = 0.0f;
    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

    for (int iz = -1; iz <= 2; ++iz) {
        float zc = index.z + iz + 0.5f;

        for (int iy = -1; iy <= 2; ++iy) {
            float yc = index.y + iy + 0.5f;

            for (int ix = -1; ix <= 2; ++ix) {
                float xc = index.x + ix + 0.5f;
                float sample = tex3D<float>(tex, xc, yc, zc);

                float w_x = wx[ix + 1];
                float w_y = wy[iy + 1];
                float w_z = wz[iz + 1];

                value += sample * w_x * w_y * w_z;
                grad_x += sample * dwx[ix + 1] * w_y * w_z;
                grad_y += sample * w_x * dwy[iy + 1] * w_z;
                grad_z += sample * w_x * w_y * dwz[iz + 1];
            }
        }
    }

    return make_float4(value, grad_x, grad_y, grad_z);
}

// =============================================================================
// Cubic Interpolator Class (3D Space Only)
// =============================================================================

class CubicInterpolator3D {
public:
    // Grid parameters
    float origin_x, origin_y, origin_z;
    float spacing_x, spacing_y, spacing_z;
    int32_t dim_x, dim_y, dim_z;

    // Device memory for prefiltered coefficients
    float* d_coefficients;
    int pitch_floats;

    // Texture for interpolation
    cudaArray_t cuda_array;
    cudaTextureObject_t tex_obj;

    CubicInterpolator3D()
        : origin_x(0), origin_y(0), origin_z(0),
          spacing_x(1), spacing_y(1), spacing_z(1),
          dim_x(0), dim_y(0), dim_z(0),
          d_coefficients(nullptr), pitch_floats(0),
          cuda_array(nullptr), tex_obj(0) {}

    ~CubicInterpolator3D() {
        destroy();
    }

    void destroy() {
        if (tex_obj != 0) cudaDestroyTextureObject(tex_obj);
        if (cuda_array != nullptr) cudaFreeArray(cuda_array);
        if (d_coefficients != nullptr) cudaFree(d_coefficients);
        tex_obj = 0;
        cuda_array = nullptr;
        d_coefficients = nullptr;
    }

    // Initialize from host data (performs prefiltering)
    void initialize(
        const float* host_data,
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz
    ) {
        destroy();

        dim_x = nx; dim_y = ny; dim_z = nz;
        origin_x = ox; origin_y = oy; origin_z = oz;
        spacing_x = sx; spacing_y = sy; spacing_z = sz;

        size_t volume_size = nx * ny * nz * sizeof(float);

        // Allocate device memory for coefficients
        // Use pitched allocation for better memory access
        pitch_floats = nx;  // Simple linear layout
        cudaMalloc(&d_coefficients, volume_size);

        // Copy data to device
        cudaMemcpy(d_coefficients, host_data, volume_size, cudaMemcpyHostToDevice);

        // Apply B-spline prefiltering (in-place)
        apply_cubic_prefilter_3d(d_coefficients, pitch_floats, nx, ny, nz);

        // Create 3D texture from prefiltered coefficients
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(nx, ny, nz);
        cudaMalloc3DArray(&cuda_array, &channel_desc, extent);

        // Copy prefiltered data to texture array
        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(d_coefficients, nx * sizeof(float), nx, ny);
        params.dstArray = cuda_array;
        params.extent = extent;
        params.kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&params);

        // Create texture object
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;  // For fast cubic method
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);

        // Free temporary coefficient buffer (data is now in texture)
        cudaFree(d_coefficients);
        d_coefficients = nullptr;

        std::cout << "CubicInterpolator3D initialized: "
                  << nx << "x" << ny << "x" << nz << " grid (prefiltered)"
                  << std::endl;
    }
};

// =============================================================================
// Cubic Interpolator 4D (3D Space + Time)
// =============================================================================
// Uses double buffering for time-varying fields
// Time interpolation is linear between two prefiltered 3D frames
// =============================================================================

class CubicInterpolator4D {
public:
    // Grid parameters (spatial)
    float origin_x, origin_y, origin_z;
    float spacing_x, spacing_y, spacing_z;
    int32_t dim_x, dim_y, dim_z;

    // Time parameters
    float time_start;
    float time_interval;
    int32_t num_frames;

    // Double buffer for two time frames
    cudaArray_t cuda_array_0;
    cudaArray_t cuda_array_1;
    cudaTextureObject_t tex_obj_0;
    cudaTextureObject_t tex_obj_1;

    // Temporary buffer for prefiltering
    float* d_temp_buffer;

    // Current buffer state
    int32_t frame_index_0;
    int32_t frame_index_1;
    float time_0;
    float time_1;

    // Boundary handling for time
    bool extrapolate_time;

    CubicInterpolator4D()
        : origin_x(0), origin_y(0), origin_z(0),
          spacing_x(1), spacing_y(1), spacing_z(1),
          dim_x(0), dim_y(0), dim_z(0),
          time_start(0), time_interval(1), num_frames(0),
          cuda_array_0(nullptr), cuda_array_1(nullptr),
          tex_obj_0(0), tex_obj_1(0),
          d_temp_buffer(nullptr),
          frame_index_0(-1), frame_index_1(-1),
          time_0(0), time_1(0),
          extrapolate_time(true) {}

    ~CubicInterpolator4D() {
        destroy();
    }

    void destroy() {
        if (tex_obj_0 != 0) cudaDestroyTextureObject(tex_obj_0);
        if (tex_obj_1 != 0) cudaDestroyTextureObject(tex_obj_1);
        if (cuda_array_0 != nullptr) cudaFreeArray(cuda_array_0);
        if (cuda_array_1 != nullptr) cudaFreeArray(cuda_array_1);
        if (d_temp_buffer != nullptr) cudaFree(d_temp_buffer);
        tex_obj_0 = 0; tex_obj_1 = 0;
        cuda_array_0 = nullptr; cuda_array_1 = nullptr;
        d_temp_buffer = nullptr;
    }

    void initialize(
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz,
        float t_start, float t_interval, int32_t n_frames,
        bool extrapolate = true
    ) {
        destroy();

        dim_x = nx; dim_y = ny; dim_z = nz;
        origin_x = ox; origin_y = oy; origin_z = oz;
        spacing_x = sx; spacing_y = sy; spacing_z = sz;
        time_start = t_start;
        time_interval = t_interval;
        num_frames = n_frames;
        extrapolate_time = extrapolate;

        size_t volume_size = nx * ny * nz * sizeof(float);

        // Allocate temporary buffer for prefiltering
        cudaMalloc(&d_temp_buffer, volume_size);

        // Allocate two 3D texture arrays
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(nx, ny, nz);
        cudaMalloc3DArray(&cuda_array_0, &channel_desc, extent);
        cudaMalloc3DArray(&cuda_array_1, &channel_desc, extent);

        // Create texture objects
        create_texture_object(&tex_obj_0, cuda_array_0);
        create_texture_object(&tex_obj_1, cuda_array_1);

        frame_index_0 = -1;
        frame_index_1 = -1;

        std::cout << "CubicInterpolator4D initialized: "
                  << nx << "x" << ny << "x" << nz << " x " << n_frames << " frames"
                  << std::endl;
    }

    // Load and prefilter a frame into one of the buffers
    int load_frame(const float* host_data, int32_t frame_index) {
        if (frame_index < 0 || frame_index >= num_frames) {
            throw std::runtime_error("Frame index out of range");
        }

        size_t volume_size = dim_x * dim_y * dim_z * sizeof(float);

        // Copy to device temp buffer
        cudaMemcpy(d_temp_buffer, host_data, volume_size, cudaMemcpyHostToDevice);

        // Apply prefiltering
        apply_cubic_prefilter_3d(d_temp_buffer, dim_x, dim_x, dim_y, dim_z);

        // Decide which buffer to use
        int buffer_to_use;
        if (frame_index_0 < 0) {
            buffer_to_use = 0;
        } else if (frame_index_1 < 0) {
            buffer_to_use = 1;
        } else if (frame_index_0 < frame_index_1) {
            buffer_to_use = 0;
        } else {
            buffer_to_use = 1;
        }

        // Copy prefiltered data to texture array
        cudaArray_t target_array = (buffer_to_use == 0) ? cuda_array_0 : cuda_array_1;

        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(d_temp_buffer, dim_x * sizeof(float), dim_x, dim_y);
        params.dstArray = target_array;
        params.extent = make_cudaExtent(dim_x, dim_y, dim_z);
        params.kind = cudaMemcpyDeviceToDevice;
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

    // Get time interpolation factor (with proper boundary handling)
    float get_time_alpha(double query_time) const {
        float t = static_cast<float>(query_time);
        float end_time = time_start + (num_frames - 1) * time_interval;

        // Handle boundaries
        if (t <= time_start) {
            return 0.0f;  // Use first frame
        }
        if (t >= end_time) {
            if (extrapolate_time) {
                // Linear extrapolation using last two frames
                // Return alpha > 1.0 for extrapolation beyond last frame
                return (t - time_0) / (time_1 - time_0);
            }
            return 1.0f;  // Clamp to last frame
        }

        // Normal interpolation
        if (t <= time_0) return 0.0f;
        if (t >= time_1) return 1.0f;
        return (t - time_0) / (time_1 - time_0);
    }

private:
    void create_texture_object(cudaTextureObject_t* tex_obj, cudaArray_t cuda_array) {
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, nullptr);
    }
};

// =============================================================================
// Vector Field Interpolators (3-component acceleration fields)
// =============================================================================

struct CubicVectorField3D {
    CubicInterpolator3D interp_x;
    CubicInterpolator3D interp_y;
    CubicInterpolator3D interp_z;

    float inv_spacing_x, inv_spacing_y, inv_spacing_z;

    void initialize(
        const float* data_x, const float* data_y, const float* data_z,
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz
    ) {
        interp_x.initialize(data_x, nx, ny, nz, ox, oy, oz, sx, sy, sz);
        interp_y.initialize(data_y, nx, ny, nz, ox, oy, oz, sx, sy, sz);
        interp_z.initialize(data_z, nx, ny, nz, ox, oy, oz, sx, sy, sz);

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

struct CubicVectorField4D {
    CubicInterpolator4D interp_x;
    CubicInterpolator4D interp_y;
    CubicInterpolator4D interp_z;

    float inv_spacing_x, inv_spacing_y, inv_spacing_z;

    void initialize(
        int32_t nx, int32_t ny, int32_t nz,
        float ox, float oy, float oz,
        float sx, float sy, float sz,
        float t_start, float t_interval, int32_t n_frames,
        bool extrapolate = true
    ) {
        interp_x.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, extrapolate);
        interp_y.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, extrapolate);
        interp_z.initialize(nx, ny, nz, ox, oy, oz, sx, sy, sz,
                           t_start, t_interval, n_frames, extrapolate);

        inv_spacing_x = 1.0f / sx;
        inv_spacing_y = 1.0f / sy;
        inv_spacing_z = 1.0f / sz;
    }

    void destroy() {
        interp_x.destroy();
        interp_y.destroy();
        interp_z.destroy();
    }

    float get_time_alpha(double query_time) const {
        return interp_x.get_time_alpha(query_time);
    }
};

// =============================================================================
// GPU Constant Memory Params for Kernel Access
// =============================================================================

struct CubicInterpolatorParams3D {
    cudaTextureObject_t tex_x;
    cudaTextureObject_t tex_y;
    cudaTextureObject_t tex_z;
    float origin_x, origin_y, origin_z;
    float inv_spacing_x, inv_spacing_y, inv_spacing_z;
    int32_t dim_x, dim_y, dim_z;
};

struct CubicInterpolatorParams4D {
    cudaTextureObject_t tex_x_0, tex_x_1;
    cudaTextureObject_t tex_y_0, tex_y_1;
    cudaTextureObject_t tex_z_0, tex_z_1;
    float origin_x, origin_y, origin_z;
    float inv_spacing_x, inv_spacing_y, inv_spacing_z;
    int32_t dim_x, dim_y, dim_z;
    float time_alpha;
};

inline CubicInterpolatorParams3D get_cubic_params(const CubicVectorField3D& field) {
    CubicInterpolatorParams3D params;
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

inline CubicInterpolatorParams4D get_cubic_params(const CubicVectorField4D& field, double query_time) {
    CubicInterpolatorParams4D params;
    params.tex_x_0 = field.interp_x.tex_obj_0;
    params.tex_x_1 = field.interp_x.tex_obj_1;
    params.tex_y_0 = field.interp_y.tex_obj_0;
    params.tex_y_1 = field.interp_y.tex_obj_1;
    params.tex_z_0 = field.interp_z.tex_obj_0;
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
// Kernel-callable Interpolation Functions
// =============================================================================

// Convert physical coordinates to texture coordinates
__device__ __forceinline__
float3 physical_to_texture_cubic(
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float inv_spacing_x, float inv_spacing_y, float inv_spacing_z
) {
    // Add 0.5 offset for cell-centered coordinates
    return make_float3(
        (px - origin_x) * inv_spacing_x + 0.5f,
        (py - origin_y) * inv_spacing_y + 0.5f,
        (pz - origin_z) * inv_spacing_z + 0.5f
    );
}

// Interpolate 3D cubic vector field
__device__ __forceinline__
float3 interp_cubic_vector_field_3d(
    const CubicInterpolatorParams3D& params,
    float px, float py, float pz
) {
    float3 tc = physical_to_texture_cubic(
        px, py, pz,
        params.origin_x, params.origin_y, params.origin_z,
        params.inv_spacing_x, params.inv_spacing_y, params.inv_spacing_z
    );

    return make_float3(
        cubic_interp_3d_fast(params.tex_x, tc.x, tc.y, tc.z,
                             params.dim_x, params.dim_y, params.dim_z),
        cubic_interp_3d_fast(params.tex_y, tc.x, tc.y, tc.z,
                             params.dim_x, params.dim_y, params.dim_z),
        cubic_interp_3d_fast(params.tex_z, tc.x, tc.y, tc.z,
                             params.dim_x, params.dim_y, params.dim_z)
    );
}

// Interpolate 4D (time-varying) cubic vector field
__device__ __forceinline__
float3 interp_cubic_vector_field_4d(
    const CubicInterpolatorParams4D& params,
    float px, float py, float pz
) {
    float3 tc = physical_to_texture_cubic(
        px, py, pz,
        params.origin_x, params.origin_y, params.origin_z,
        params.inv_spacing_x, params.inv_spacing_y, params.inv_spacing_z
    );

    // Interpolate each component at both time frames
    float vx_0 = cubic_interp_3d_fast(params.tex_x_0, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);
    float vx_1 = cubic_interp_3d_fast(params.tex_x_1, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);

    float vy_0 = cubic_interp_3d_fast(params.tex_y_0, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);
    float vy_1 = cubic_interp_3d_fast(params.tex_y_1, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);

    float vz_0 = cubic_interp_3d_fast(params.tex_z_0, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);
    float vz_1 = cubic_interp_3d_fast(params.tex_z_1, tc.x, tc.y, tc.z,
                                       params.dim_x, params.dim_y, params.dim_z);

    // Linear interpolation in time (or extrapolation if alpha > 1 or < 0)
    float alpha = params.time_alpha;
    return make_float3(
        vx_0 + alpha * (vx_1 - vx_0),
        vy_0 + alpha * (vy_1 - vy_0),
        vz_0 + alpha * (vz_1 - vz_0)
    );
}

} // namespace cudadem

#endif // CUDADEM_CUBIC_INTERPOLATION_CUH

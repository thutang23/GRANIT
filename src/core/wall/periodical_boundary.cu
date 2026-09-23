// =============================================================================
// Periodic Boundary Conditions for DEM Simulation
// =============================================================================
// Implements periodic boundaries with:
// - Position wrapping after integration
// - Minimum image convention for distance calculation
// - Periodic neighbor search extension
//
// Reference: lib/pbc.f90, lib/forcePBC.f90
// =============================================================================

#include "wall.h"
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstdint>

namespace cudadem {

// =============================================================================
// GPU Constant Memory
// =============================================================================
// Use raw storage to avoid dynamic initialization (CUDA limitation)
// This is separate from wall_manager.cu's dev_pbc_config_storage - the PBC
// dispatcher copies the config here independently.
// =============================================================================

__constant__ unsigned char dev_pbc_local_storage[sizeof(PeriodicBoundaryConfig)];

// Accessor macro
#define dev_pbc_local (*reinterpret_cast<PeriodicBoundaryConfig*>(const_cast<unsigned char*>(dev_pbc_local_storage)))

// Copy PBC config to GPU constant memory
inline void copy_pbc_config_to_device_local(const PeriodicBoundaryConfig& pbc) {
    cudaMemcpyToSymbol(dev_pbc_local_storage, &pbc, sizeof(PeriodicBoundaryConfig));
}

// =============================================================================
// Additional Device Functions (not in wall.h)
// =============================================================================

// Wrap 3D position vector (float)
__device__ __forceinline__
void wrap_position_float(
    float& x, float& y, float& z,
    const PeriodicBoundaryConfig& pbc
) {
    if (pbc.periodic_x && pbc.box_length_x > 0.0f) {
        x = wrap_coordinate(x, pbc.box_min_x, pbc.box_max_x, pbc.box_length_x);
    }
    if (pbc.periodic_y && pbc.box_length_y > 0.0f) {
        y = wrap_coordinate(y, pbc.box_min_y, pbc.box_max_y, pbc.box_length_y);
    }
    if (pbc.periodic_z && pbc.box_length_z > 0.0f) {
        z = wrap_coordinate(z, pbc.box_min_z, pbc.box_max_z, pbc.box_length_z);
    }
}

// Apply minimum image convention to 3D distance vector with image tracking
__device__ __forceinline__
void minimum_image_distance_with_images(
    float& dx, float& dy, float& dz,
    int8_t& image_x, int8_t& image_y, int8_t& image_z,
    const PeriodicBoundaryConfig& pbc
) {
    image_x = 0;
    image_y = 0;
    image_z = 0;

    if (pbc.periodic_x && pbc.box_length_x > 0.0f) {
        if (dx > pbc.half_length_x) {
            dx -= pbc.box_length_x;
            image_x = -1;
        } else if (dx < -pbc.half_length_x) {
            dx += pbc.box_length_x;
            image_x = +1;
        }
    }

    if (pbc.periodic_y && pbc.box_length_y > 0.0f) {
        if (dy > pbc.half_length_y) {
            dy -= pbc.box_length_y;
            image_y = -1;
        } else if (dy < -pbc.half_length_y) {
            dy += pbc.box_length_y;
            image_y = +1;
        }
    }

    if (pbc.periodic_z && pbc.box_length_z > 0.0f) {
        if (dz > pbc.half_length_z) {
            dz -= pbc.box_length_z;
            image_z = -1;
        } else if (dz < -pbc.half_length_z) {
            dz += pbc.box_length_z;
            image_z = +1;
        }
    }
}

// Compute distance vector with minimum image convention (from i to j)
__device__ __forceinline__
void compute_distance_pbc(
    float& dx, float& dy, float& dz,
    float xi, float yi, float zi,
    float xj, float yj, float zj,
    const PeriodicBoundaryConfig& pbc
) {
    dx = xj - xi;
    dy = yj - yi;
    dz = zj - zi;

    minimum_image_distance(dx, dy, dz, pbc);
}

// Double precision version for position arrays
__device__ __forceinline__
void compute_distance_pbc_double(
    float& dx, float& dy, float& dz,
    double xi, double yi, double zi,
    double xj, double yj, double zj,
    const PeriodicBoundaryConfig& pbc
) {
    dx = static_cast<float>(xj - xi);
    dy = static_cast<float>(yj - yi);
    dz = static_cast<float>(zj - zi);

    minimum_image_distance(dx, dy, dz, pbc);
}

// =============================================================================
// Kernels: Position Wrapping
// =============================================================================

static __global__ void wrap_positions_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Read from constant memory
    PeriodicBoundaryConfig pbc = dev_pbc_local;

    if (!pbc.is_enabled()) return;

    double x = pos_x[idx];
    double y = pos_y[idx];
    double z = pos_z[idx];

    wrap_position_double(x, y, z, pbc);

    pos_x[idx] = x;
    pos_y[idx] = y;
    pos_z[idx] = z;
}

// Float version for temporary arrays
__global__ void wrap_positions_float_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    PeriodicBoundaryConfig pbc = dev_pbc_local;

    if (!pbc.is_enabled()) return;

    float x = pos_x[idx];
    float y = pos_y[idx];
    float z = pos_z[idx];

    wrap_position_float(x, y, z, pbc);

    pos_x[idx] = x;
    pos_y[idx] = y;
    pos_z[idx] = z;
}

// =============================================================================
// Periodic Neighbor Search Extension
// =============================================================================

// Compute cell index with periodic wrapping
__device__ __forceinline__
int periodic_cell_index(
    int cell_idx, int num_cells, bool is_periodic
) {
    if (!is_periodic) {
        // Non-periodic: clamp to valid range
        if (cell_idx < 0) return 0;
        if (cell_idx >= num_cells) return num_cells - 1;
        return cell_idx;
    }

    // Periodic: wrap around
    if (cell_idx < 0) {
        cell_idx += num_cells;
    } else if (cell_idx >= num_cells) {
        cell_idx -= num_cells;
    }
    return cell_idx;
}

// Get linear cell index with periodic wrapping
__device__ __forceinline__
int get_periodic_cell_linear_index(
    int cx, int cy, int cz,
    int cells_x, int cells_y, int cells_z,
    const PeriodicBoundaryConfig& pbc
) {
    cx = periodic_cell_index(cx, cells_x, pbc.periodic_x);
    cy = periodic_cell_index(cy, cells_y, pbc.periodic_y);
    cz = periodic_cell_index(cz, cells_z, pbc.periodic_z);

    return cx + cy * cells_x + cz * cells_x * cells_y;
}

// Check if neighbor cell search should include periodic image
__device__ __forceinline__
void get_periodic_offset(
    int dx, int dy, int dz,
    int cell_x, int cell_y, int cell_z,
    int cells_x, int cells_y, int cells_z,
    const PeriodicBoundaryConfig& pbc,
    float& offset_x, float& offset_y, float& offset_z
) {
    offset_x = 0.0f;
    offset_y = 0.0f;
    offset_z = 0.0f;

    if (pbc.periodic_x) {
        int neighbor_x = cell_x + dx;
        if (neighbor_x < 0) {
            offset_x = pbc.box_length_x;
        } else if (neighbor_x >= cells_x) {
            offset_x = -pbc.box_length_x;
        }
    }

    if (pbc.periodic_y) {
        int neighbor_y = cell_y + dy;
        if (neighbor_y < 0) {
            offset_y = pbc.box_length_y;
        } else if (neighbor_y >= cells_y) {
            offset_y = -pbc.box_length_y;
        }
    }

    if (pbc.periodic_z) {
        int neighbor_z = cell_z + dz;
        if (neighbor_z < 0) {
            offset_z = pbc.box_length_z;
        } else if (neighbor_z >= cells_z) {
            offset_z = -pbc.box_length_z;
        }
    }
}

// =============================================================================
// Periodic Contact Force Extension
// =============================================================================

// Structure to hold periodic contact information
struct PeriodicContactInfo {
    float offset_x, offset_y, offset_z;  // Position offset for periodic image
    bool is_periodic;                     // Whether this is a periodic image contact
};

// Compute contact distance with periodic handling
__device__ __forceinline__
float compute_contact_distance_pbc(
    float& nx, float& ny, float& nz,
    float& dist,
    float xi, float yi, float zi,
    float xj, float yj, float zj,
    const PeriodicBoundaryConfig& pbc
) {
    float dx = xj - xi;
    float dy = yj - yi;
    float dz = zj - zi;

    minimum_image_distance(dx, dy, dz, pbc);

    float dist_sq = dx * dx + dy * dy + dz * dz;
    dist = sqrtf(dist_sq);

    if (dist > 1e-10f) {
        float inv_dist = 1.0f / dist;
        nx = dx * inv_dist;
        ny = dy * inv_dist;
        nz = dz * inv_dist;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }

    return dist_sq;
}

// =============================================================================
// Host Dispatch Function
// =============================================================================

class PeriodicBoundaryDispatcher {
public:
    static void wrap_positions(
        double* d_pos_x, double* d_pos_y, double* d_pos_z,
        int32_t num_particles,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        PeriodicBoundaryConfig pbc = extract_pbc_config(config);

        if (!pbc.is_enabled()) return;

        copy_pbc_config_to_device_local(pbc);

        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        LAUNCH_KERNEL(wrap_positions_kernel, num_blocks, block_size, 0, stream,
            d_pos_x, d_pos_y, d_pos_z, num_particles
        );
    }

    static void wrap_positions_float(
        float* d_pos_x, float* d_pos_y, float* d_pos_z,
        int32_t num_particles,
        const SimConfig& config,
        cudaStream_t stream = 0
    ) {
        PeriodicBoundaryConfig pbc = extract_pbc_config(config);

        if (!pbc.is_enabled()) return;

        copy_pbc_config_to_device_local(pbc);

        int block_size = config.gpu.block_size;
        int num_blocks = (num_particles + block_size - 1) / block_size;

        LAUNCH_KERNEL(wrap_positions_float_kernel, num_blocks, block_size, 0, stream,
            d_pos_x, d_pos_y, d_pos_z, num_particles
        );
    }

    static PeriodicBoundaryConfig get_config(const SimConfig& config) {
        return extract_pbc_config(config);
    }

    static void initialize(const SimConfig& config) {
        PeriodicBoundaryConfig pbc = extract_pbc_config(config);
        copy_pbc_config_to_device_local(pbc);
    }
};

// =============================================================================
// Template Policy for Periodic Boundaries
// =============================================================================

template<bool EnableX, bool EnableY, bool EnableZ>
struct PeriodicPolicy {
    static constexpr bool periodic_x = EnableX;
    static constexpr bool periodic_y = EnableY;
    static constexpr bool periodic_z = EnableZ;
    static constexpr bool any_periodic = EnableX || EnableY || EnableZ;
};

using NonPeriodicPolicy = PeriodicPolicy<false, false, false>;
using PeriodicXPolicy = PeriodicPolicy<true, false, false>;
using PeriodicYPolicy = PeriodicPolicy<false, true, false>;
using PeriodicXYPolicy = PeriodicPolicy<true, true, false>;
using FullyPeriodicPolicy = PeriodicPolicy<true, true, true>;

// Template-based minimum image (compile-time optimization)
template<typename Policy>
__device__ __forceinline__
void minimum_image_distance_policy(
    float& dx, float& dy, float& dz,
    float box_length_x, float half_length_x,
    float box_length_y, float half_length_y,
    float box_length_z, float half_length_z
) {
    if constexpr (Policy::periodic_x) {
        dx = minimum_image(dx, box_length_x, half_length_x);
    }
    if constexpr (Policy::periodic_y) {
        dy = minimum_image(dy, box_length_y, half_length_y);
    }
    if constexpr (Policy::periodic_z) {
        dz = minimum_image(dz, box_length_z, half_length_z);
    }
}

// =============================================================================
// Integration with Simulation Step
// =============================================================================

inline void apply_periodic_boundaries(
    double* d_pos_x, double* d_pos_y, double* d_pos_z,
    int32_t num_particles,
    const SimConfig& config,
    cudaStream_t stream = 0
) {
    PeriodicBoundaryDispatcher::wrap_positions(
        d_pos_x, d_pos_y, d_pos_z,
        num_particles, config, stream
    );
}

} // namespace cudadem

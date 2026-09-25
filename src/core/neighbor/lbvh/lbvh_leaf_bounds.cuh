#pragma once
// Included inside cudadem::lbvh. Shared by the solver and GPU bounds regression.

// 1. Compute leaf AABBs from spheres
__global__ void lbvh_compute_aabbs_kernel(const double *__restrict__ pos_x,
                                          const double *__restrict__ pos_y,
                                          const double *__restrict__ pos_z,
                                          const float *__restrict__ radius,
                                          LBVHAABB *__restrict__ aabbs,
                                          int32_t num_particles,
                                          int32_t internal_offset,
                                          float extent_scale) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  float px = static_cast<float>(pos_x[idx]);
  float py = static_cast<float>(pos_y[idx]);
  float pz = static_cast<float>(pos_z[idx]);
  float r = radius[idx] * extent_scale;
  if (extent_scale != 1.0f)
    r = nextafterf(r * (1.0f + 8.0f * FLT_EPSILON)
                   + 2.0f * sqrtf(FLT_MIN), CUDART_INF_F);

  LBVHAABB box;
  if (extent_scale != 1.0f) {
    box.lower = make_float4(nextafterf(px - r, -CUDART_INF_F),
                            nextafterf(py - r, -CUDART_INF_F),
                            nextafterf(pz - r, -CUDART_INF_F), 0.0f);
    box.upper = make_float4(nextafterf(px + r, CUDART_INF_F),
                            nextafterf(py + r, CUDART_INF_F),
                            nextafterf(pz + r, CUDART_INF_F), 0.0f);
  } else {
    box.lower = make_float4(px - r, py - r, pz - r, 0.0f);
    box.upper = make_float4(px + r, py + r, pz + r, 0.0f);
  }

  // Leaves are stored at indices [internal_offset, 2N-2]
  aabbs[internal_offset + idx] = box;
}

// Fast-path: refresh leaf AABBs with skin inflation
// NOTE: Leaf AABBs are stored in Morton-sorted order, so we must use sorted_indices
// to look up the correct particle position for each leaf.
__global__ void lbvh_refresh_leaf_aabbs_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ radius,
    const uint32_t* __restrict__ sorted_indices,
    LBVHAABB*     __restrict__ aabbs,
    int32_t num_particles,
    int32_t internal_offset,
    float   skin_half,
    float   extent_scale)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // idx is the sorted position; sorted_indices[idx] gives the original particle index
    uint32_t orig_idx = sorted_indices[idx];

    float px = static_cast<float>(pos_x[orig_idx]);
    float py = static_cast<float>(pos_y[orig_idx]);
    float pz = static_cast<float>(pos_z[orig_idx]);
    float r  = radius[orig_idx] * extent_scale + skin_half;
    if (extent_scale != 1.0f)
      r = nextafterf(r * (1.0f + 8.0f * FLT_EPSILON)
                     + 2.0f * sqrtf(FLT_MIN), CUDART_INF_F);

    LBVHAABB box;
    if (extent_scale != 1.0f) {
      box.lower = make_float4(nextafterf(px - r, -CUDART_INF_F),
                              nextafterf(py - r, -CUDART_INF_F),
                              nextafterf(pz - r, -CUDART_INF_F), 0.0f);
      box.upper = make_float4(nextafterf(px + r, CUDART_INF_F),
                              nextafterf(py + r, CUDART_INF_F),
                              nextafterf(pz + r, CUDART_INF_F), 0.0f);
    } else {
      box.lower = make_float4(px - r, py - r, pz - r, 0.0f);
      box.upper = make_float4(px + r, py + r, pz + r, 0.0f);
    }

    aabbs[internal_offset + idx] = box;
}

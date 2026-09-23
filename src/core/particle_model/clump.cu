// =============================================================================
// Clump (Multi-Sphere Rigid Body) — CUDA Kernels and Host Functions
// =============================================================================

#include "clump.h"
#include "clump_rigid_math.cuh"
#include "cuda_utils.h"
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace cudadem {

// =============================================================================
// Allocation / Free
// =============================================================================

ClumpStateSoA allocate_clump_state(int32_t max_clumps) {
    ClumpStateSoA cs;
    cs.capacity   = max_clumps;
    cs.num_clumps = 0;

    auto alloc_d = [](double** p, int32_t n) {
        CUDA_CHECK(cudaMalloc(p, n * sizeof(double)));
    };

    alloc_d(&cs.com_x,  max_clumps); alloc_d(&cs.com_y,  max_clumps); alloc_d(&cs.com_z,  max_clumps);
    alloc_d(&cs.vel_x,  max_clumps); alloc_d(&cs.vel_y,  max_clumps); alloc_d(&cs.vel_z,  max_clumps);
    alloc_d(&cs.omega_x, max_clumps); alloc_d(&cs.omega_y, max_clumps); alloc_d(&cs.omega_z, max_clumps);
    alloc_d(&cs.quat_w, max_clumps); alloc_d(&cs.quat_x, max_clumps);
    alloc_d(&cs.quat_y, max_clumps); alloc_d(&cs.quat_z, max_clumps);
    alloc_d(&cs.mass, max_clumps);
    alloc_d(&cs.inertia_xx, max_clumps); alloc_d(&cs.inertia_xy, max_clumps); alloc_d(&cs.inertia_xz, max_clumps);
    alloc_d(&cs.inertia_yy, max_clumps); alloc_d(&cs.inertia_yz, max_clumps); alloc_d(&cs.inertia_zz, max_clumps);
    for (double*& q : cs.multipole_p3) alloc_d(&q, max_clumps);
    alloc_d(&cs.net_force_x,  max_clumps); alloc_d(&cs.net_force_y,  max_clumps); alloc_d(&cs.net_force_z,  max_clumps);
    alloc_d(&cs.net_torque_x, max_clumps); alloc_d(&cs.net_torque_y, max_clumps); alloc_d(&cs.net_torque_z, max_clumps);
    alloc_d(&cs.gravity_force_x,max_clumps); alloc_d(&cs.gravity_force_y,max_clumps); alloc_d(&cs.gravity_force_z,max_clumps);
    alloc_d(&cs.gravity_torque_x,max_clumps); alloc_d(&cs.gravity_torque_y,max_clumps); alloc_d(&cs.gravity_torque_z,max_clumps);
    return cs;
}

ClumpSegmentOffsets allocate_clump_segments(int32_t max_clumps) {
    ClumpSegmentOffsets s;
    s.num_segments = 0;
    CUDA_CHECK(cudaMalloc(&s.seg_begin, max_clumps * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&s.seg_end,   max_clumps * sizeof(int32_t)));
    return s;
}

ClumpStagingBuffer allocate_clump_staging(int32_t max_sub_spheres) {
    ClumpStagingBuffer b;
    b.capacity = max_sub_spheres;
    auto alloc_d = [](double** p, int32_t n) {
        CUDA_CHECK(cudaMalloc(p, n * sizeof(double)));
    };
    alloc_d(&b.fx, max_sub_spheres); alloc_d(&b.fy, max_sub_spheres); alloc_d(&b.fz, max_sub_spheres);
    alloc_d(&b.tx, max_sub_spheres); alloc_d(&b.ty, max_sub_spheres); alloc_d(&b.tz, max_sub_spheres);
    return b;
}

void free_clump_state(ClumpStateSoA& cs) {
    CUDA_CHECK(cudaFree(cs.com_x));  CUDA_CHECK(cudaFree(cs.com_y));  CUDA_CHECK(cudaFree(cs.com_z));
    CUDA_CHECK(cudaFree(cs.vel_x));  CUDA_CHECK(cudaFree(cs.vel_y));  CUDA_CHECK(cudaFree(cs.vel_z));
    CUDA_CHECK(cudaFree(cs.omega_x)); CUDA_CHECK(cudaFree(cs.omega_y)); CUDA_CHECK(cudaFree(cs.omega_z));
    CUDA_CHECK(cudaFree(cs.quat_w));  CUDA_CHECK(cudaFree(cs.quat_x));  CUDA_CHECK(cudaFree(cs.quat_y));  CUDA_CHECK(cudaFree(cs.quat_z));
    CUDA_CHECK(cudaFree(cs.mass));
    CUDA_CHECK(cudaFree(cs.inertia_xx)); CUDA_CHECK(cudaFree(cs.inertia_xy)); CUDA_CHECK(cudaFree(cs.inertia_xz));
    CUDA_CHECK(cudaFree(cs.inertia_yy)); CUDA_CHECK(cudaFree(cs.inertia_yz)); CUDA_CHECK(cudaFree(cs.inertia_zz));
    for (double*& q : cs.multipole_p3) CUDA_CHECK(cudaFree(q));
    CUDA_CHECK(cudaFree(cs.net_force_x));  CUDA_CHECK(cudaFree(cs.net_force_y));  CUDA_CHECK(cudaFree(cs.net_force_z));
    CUDA_CHECK(cudaFree(cs.net_torque_x)); CUDA_CHECK(cudaFree(cs.net_torque_y)); CUDA_CHECK(cudaFree(cs.net_torque_z));
    CUDA_CHECK(cudaFree(cs.gravity_force_x)); CUDA_CHECK(cudaFree(cs.gravity_force_y)); CUDA_CHECK(cudaFree(cs.gravity_force_z));
    CUDA_CHECK(cudaFree(cs.gravity_torque_x)); CUDA_CHECK(cudaFree(cs.gravity_torque_y)); CUDA_CHECK(cudaFree(cs.gravity_torque_z));
}

void free_clump_segments(ClumpSegmentOffsets& s) {
    CUDA_CHECK(cudaFree(s.seg_begin));
    CUDA_CHECK(cudaFree(s.seg_end));
}

void free_clump_staging(ClumpStagingBuffer& b) {
    CUDA_CHECK(cudaFree(b.fx)); CUDA_CHECK(cudaFree(b.fy)); CUDA_CHECK(cudaFree(b.fz));
    CUDA_CHECK(cudaFree(b.tx)); CUDA_CHECK(cudaFree(b.ty)); CUDA_CHECK(cudaFree(b.tz));
    if (b.cub_temp_storage) CUDA_CHECK(cudaFree(b.cub_temp_storage));
}

// =============================================================================
// Zero Accumulators
// =============================================================================
__global__ void clump_zero_accumulators_kernel(
    double* net_fx, double* net_fy, double* net_fz,
    double* net_tx, double* net_ty, double* net_tz,
    int32_t num_clumps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clumps) return;
    net_fx[idx] = 0.0; net_fy[idx] = 0.0; net_fz[idx] = 0.0;
    net_tx[idx] = 0.0; net_ty[idx] = 0.0; net_tz[idx] = 0.0;
}

// =============================================================================
// Pre-Aggregate: torque arm τ_i = (pos_i - com_cid) × force_i, write to staging
// =============================================================================
__global__ void clump_pre_aggregate_kernel(
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ force_z,
    const double* __restrict__ torque_x,
    const double* __restrict__ torque_y,
    const double* __restrict__ torque_z,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int32_t* __restrict__ clump_id,
    const double* __restrict__ com_x,
    const double* __restrict__ com_y,
    const double* __restrict__ com_z,
    double* __restrict__ stg_fx, double* __restrict__ stg_fy, double* __restrict__ stg_fz,
    double* __restrict__ stg_tx, double* __restrict__ stg_ty, double* __restrict__ stg_tz,
    int32_t num_sub_spheres
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sub_spheres) return;

    int32_t cid = clump_id[idx];
    if (cid <= 0) return;
    --cid; // persistent IDs are 1..K; state arrays are zero-based

    double fx = force_x[idx];
    double fy = force_y[idx];
    double fz = force_z[idx];

    // Torque arm: r = pos - com (world frame)
    double rx = pos_x[idx] - com_x[cid];
    double ry = pos_y[idx] - com_y[cid];
    double rz = pos_z[idx] - com_z[cid];

    // τ = r × F
    stg_fx[idx] = fx;
    stg_fy[idx] = fy;
    stg_fz[idx] = fz;
    stg_tx[idx] = ry * fz - rz * fy + torque_x[idx];
    stg_ty[idx] = rz * fx - rx * fz + torque_y[idx];
    stg_tz[idx] = rx * fy - ry * fx + torque_z[idx];
}

// =============================================================================
// Quaternion utilities (device, used by sync and verlet kernels)
// q rotates body → world: v_world = q ⊗ (0,v) ⊗ q*
// Efficient Rodrigues formula: v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))
// =============================================================================
__device__ __forceinline__
void quat_rotate_d(double qw, double qx, double qy, double qz,
                   double vx, double vy, double vz,
                   double& ox, double& oy, double& oz)
{
    double tx = 2.0 * (qy * vz - qz * vy);
    double ty = 2.0 * (qz * vx - qx * vz);
    double tz = 2.0 * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

// Conjugate quaternion: rotates world → body
__device__ __forceinline__
void quat_rotate_inv_d(double qw, double qx, double qy, double qz,
                       double vx, double vy, double vz,
                       double& ox, double& oy, double& oz)
{
    quat_rotate_d(qw, -qx, -qy, -qz, vx, vy, vz, ox, oy, oz);
}

__global__ void clump_apply_gravity_impulse_kernel(
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    const double* quat_w, const double* quat_x, const double* quat_y, const double* quat_z,
    const double* gravity_force_x, const double* gravity_force_y, const double* gravity_force_z,
    const double* gravity_torque_x, const double* gravity_torque_y, const double* gravity_torque_z,
    const double* mass,
    const double* ixx, const double* ixy, const double* ixz,
    const double* iyy, const double* iyz, const double* izz,
    double impulse_dt, int32_t num_clumps) {
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=num_clumps)return;
    const double inv_mass=1.0/mass[idx];
    vel_x[idx]+=impulse_dt*gravity_force_x[idx]*inv_mass;
    vel_y[idx]+=impulse_dt*gravity_force_y[idx]*inv_mass;
    vel_z[idx]+=impulse_dt*gravity_force_z[idx]*inv_mass;
    double tx,ty,tz;
    quat_rotate_inv_d(quat_w[idx],quat_x[idx],quat_y[idx],quat_z[idx],
                      gravity_torque_x[idx],gravity_torque_y[idx],gravity_torque_z[idx],tx,ty,tz);
    // A RESPA potential kick changes angular momentum only through the cached
    // gravity torque.  Torque-free Euler precession is already integrated by
    // every inner clump step and must not be applied again at the slow level.
    ClumpRigidVec3 alpha{};
    if(clump_tensor_torque_alpha(ixx[idx],ixy[idx],ixz[idx],iyy[idx],iyz[idx],izz[idx],
                                 {tx,ty,tz},alpha)) {
        omega_x[idx]+=impulse_dt*alpha.x;
        omega_y[idx]+=impulse_dt*alpha.y;
        omega_z[idx]+=impulse_dt*alpha.z;
    }
}

// =============================================================================
// Verlet Step 1: half-kick, drift CoM, and rotate body
// =============================================================================
__global__ void clump_verlet_step1_kernel(
    double* com_x, double* com_y, double* com_z,
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    const double* net_force_x, const double* net_force_y, const double* net_force_z,
    const double* net_torque_x, const double* net_torque_y, const double* net_torque_z,
    double* quat_w, double* quat_x, double* quat_y, double* quat_z,
    const double* mass, const double* ixx, const double* ixy, const double* ixz,
    const double* iyy, const double* iyz, const double* izz,
    double dt, int32_t num_clumps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clumps) return;

    double half_dt = 0.5 * dt;
    double inv_m   = 1.0 / mass[idx];

    // Translational half-kick
    vel_x[idx] += half_dt * net_force_x[idx] * inv_m;
    vel_y[idx] += half_dt * net_force_y[idx] * inv_m;
    vel_z[idx] += half_dt * net_force_z[idx] * inv_m;

    // Rotate world torque → body frame
    double tx_b, ty_b, tz_b;
    quat_rotate_inv_d(quat_w[idx], quat_x[idx], quat_y[idx], quat_z[idx],
                      net_torque_x[idx], net_torque_y[idx], net_torque_z[idx],
                      tx_b, ty_b, tz_b);

    ClumpRigidVec3 alpha{};
    if (!clump_full_tensor_alpha(
            ixx[idx],ixy[idx],ixz[idx],iyy[idx],iyz[idx],izz[idx],
            {omega_x[idx],omega_y[idx],omega_z[idx]}, {tx_b,ty_b,tz_b},alpha)) return;
    omega_x[idx] += half_dt * alpha.x;
    omega_y[idx] += half_dt * alpha.y;
    omega_z[idx] += half_dt * alpha.z;

    com_x[idx] += vel_x[idx] * dt;
    com_y[idx] += vel_y[idx] * dt;
    com_z[idx] += vel_z[idx] * dt;
    double qw=quat_w[idx],qx=quat_x[idx],qy=quat_y[idx],qz=quat_z[idx];
    clump_exp_quaternion(qw,qx,qy,qz,
                         {omega_x[idx],omega_y[idx],omega_z[idx]},dt);
    quat_w[idx]=qw;quat_x[idx]=qx;quat_y[idx]=qy;quat_z[idx]=qz;
}

// =============================================================================
// Verlet Step 2: second half-kick
// =============================================================================
__global__ void clump_verlet_step2_kernel(
    double* com_x, double* com_y, double* com_z,
    double* vel_x, double* vel_y, double* vel_z,
    double* omega_x, double* omega_y, double* omega_z,
    double* quat_w, double* quat_x, double* quat_y, double* quat_z,
    const double* net_force_x, const double* net_force_y, const double* net_force_z,
    const double* net_torque_x, const double* net_torque_y, const double* net_torque_z,
    const double* mass, const double* ixx, const double* ixy, const double* ixz,
    const double* iyy, const double* iyz, const double* izz,
    double dt, int32_t num_clumps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clumps) return;

    double half_dt = 0.5 * dt;

    // Translational second half-kick
    double inv_m = 1.0 / mass[idx];
    vel_x[idx] += half_dt * net_force_x[idx] * inv_m;
    vel_y[idx] += half_dt * net_force_y[idx] * inv_m;
    vel_z[idx] += half_dt * net_force_z[idx] * inv_m;

    // Rotational second half-kick (use updated quaternion for τ_body)
    double tx_b, ty_b, tz_b;
    quat_rotate_inv_d(quat_w[idx], quat_x[idx], quat_y[idx], quat_z[idx],
                      net_torque_x[idx], net_torque_y[idx], net_torque_z[idx],
                      tx_b, ty_b, tz_b);

    ClumpRigidVec3 alpha{};
    if (clump_full_tensor_alpha(
            ixx[idx],ixy[idx],ixz[idx],iyy[idx],iyz[idx],izz[idx],
            {omega_x[idx],omega_y[idx],omega_z[idx]}, {tx_b,ty_b,tz_b},alpha)) {
        omega_x[idx]+=half_dt*alpha.x;
        omega_y[idx]+=half_dt*alpha.y;
        omega_z[idx]+=half_dt*alpha.z;
    }
}

// =============================================================================
// Sync Positions: pos_i = com_{cid} + R(quat_{cid}) * body_offset_i
// =============================================================================
__global__ void clump_sync_positions_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const float* __restrict__ off_x,
    const float* __restrict__ off_y,
    const float* __restrict__ off_z,
    const int32_t* __restrict__ clump_id,
    const double* __restrict__ com_x,
    const double* __restrict__ com_y,
    const double* __restrict__ com_z,
    const double* __restrict__ quat_w,
    const double* __restrict__ quat_x,
    const double* __restrict__ quat_y,
    const double* __restrict__ quat_z,
    int32_t num_sub_spheres
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sub_spheres) return;

    int32_t cid = clump_id[idx];
    if (cid <= 0) return;
    --cid;

    double ox, oy, oz;
    quat_rotate_d(quat_w[cid], quat_x[cid], quat_y[cid], quat_z[cid],
                  (double)off_x[idx], (double)off_y[idx], (double)off_z[idx],
                  ox, oy, oz);

    pos_x[idx] = com_x[cid] + ox;
    pos_y[idx] = com_y[cid] + oy;
    pos_z[idx] = com_z[cid] + oz;
}

__global__ void clump_sync_kinematics_kernel(
    double* px, double* py, double* pz, double* vx, double* vy, double* vz,
    double* ox, double* oy, double* oz, double* pqw, double* pqx, double* pqy, double* pqz,
    const float* bx, const float* by, const float* bz, const int32_t* ids,
    const double* cx, const double* cy, const double* cz, const double* cvx, const double* cvy, const double* cvz, const double* bwx, const double* bwy, const double* bwz,
    const double* qw, const double* qx, const double* qy, const double* qz, int32_t n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n || ids[i]<=0) return; int c=ids[i]-1;
    double rx,ry,rz, wx,wy,wz; quat_rotate_d(qw[c],qx[c],qy[c],qz[c],bx[i],by[i],bz[i],rx,ry,rz);
    quat_rotate_d(qw[c],qx[c],qy[c],qz[c],bwx[c],bwy[c],bwz[c],wx,wy,wz);
    px[i]=cx[c]+rx; py[i]=cy[c]+ry; pz[i]=cz[c]+rz;
    vx[i]=cvx[c] + (wy*rz-wz*ry); vy[i]=cvy[c] + (wz*rx-wx*rz); vz[i]=cvz[c] + (wx*ry-wy*rx);
    // Member omega is world-frame and member quaternion is the rigid orientation.
    ox[i]=wx; oy[i]=wy; oz[i]=wz; pqw[i]=qw[c]; pqx[i]=qx[c]; pqy[i]=qy[c]; pqz[i]=qz[c];
}

__global__ void clump_wrap_com_kernel(
    double* x,double* y,double* z,int32_t n,bool px,bool py,bool pz,
    double minx,double miny,double minz,double lx,double ly,double lz) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(px&&lx>0)x[i]-=floor((x[i]-minx)/lx)*lx;
    if(py&&ly>0)y[i]-=floor((y[i]-miny)/ly)*ly;
    if(pz&&lz>0)z[i]-=floor((z[i]-minz)/lz)*lz;
}

// =============================================================================
// initialize_clump_cub_temp: pre-allocate CUB temp storage at init time.
// Call after initialize_clumps so seg offsets and staging fx pointer are valid.
// =============================================================================
void initialize_clump_cub_temp(
    ClumpStagingBuffer& staging,
    const ClumpStateSoA& cs,
    const ClumpSegmentOffsets& seg
) {
    if (cs.num_clumps == 0) return;

    // Query the temp size needed for one channel
    staging.cub_temp_storage = nullptr;
    staging.cub_temp_bytes   = 0;

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        staging.cub_temp_storage, staging.cub_temp_bytes,
        staging.fx, cs.net_force_x,
        seg.num_segments, seg.seg_begin, seg.seg_end,
        (cudaStream_t)0
    ));

    CUDA_CHECK(cudaMalloc(&staging.cub_temp_storage, staging.cub_temp_bytes));
}

// =============================================================================
// CUB Segmented Reduce: aggregate staging buffer → net_force/torque per clump
// Uses the pre-allocated persistent temp buffer from ClumpStagingBuffer.
// =============================================================================
void clump_aggregate_forces(
    const ClumpStagingBuffer& staging,
    ClumpStateSoA& cs,
    const ClumpSegmentOffsets& seg,
    int32_t num_sub_spheres,
    cudaStream_t stream
) {
    if (cs.num_clumps == 0 || num_sub_spheres == 0) return;

    void*  d_temp     = staging.cub_temp_storage;
    size_t temp_bytes = staging.cub_temp_bytes;

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.fx, cs.net_force_x,  seg.num_segments, seg.seg_begin, seg.seg_end, stream));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.fy, cs.net_force_y,  seg.num_segments, seg.seg_begin, seg.seg_end, stream));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.fz, cs.net_force_z,  seg.num_segments, seg.seg_begin, seg.seg_end, stream));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.tx, cs.net_torque_x, seg.num_segments, seg.seg_begin, seg.seg_end, stream));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.ty, cs.net_torque_y, seg.num_segments, seg.seg_begin, seg.seg_end, stream));
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, staging.tz, cs.net_torque_z, seg.num_segments, seg.seg_begin, seg.seg_end, stream));
}

// =============================================================================
// Host API — state retrieval for visualization
// Copies GPU clump state data to host memory for visualization output.
// Caller must free returned buffers with free_clump_host_state().
// =============================================================================
int32_t clump_get_states(
    ClumpStateSoA& host_states,
    int32_t** template_indices,
    const ClumpStateSoA& device_states
) {
    int32_t num_clumps = device_states.num_clumps;
    if (num_clumps == 0) {
        host_states.num_clumps = 0;
        return 0;
    }

    // Set capacity and num_clumps on host
    host_states.capacity   = device_states.capacity;
    host_states.num_clumps = num_clumps;

    // Allocate host buffers
    auto alloc_d = [](double** p, int32_t n) {
        CUDA_CHECK(cudaMallocHost(p, n * sizeof(double)));
    };

    alloc_d(&host_states.com_x,  num_clumps); alloc_d(&host_states.com_y,  num_clumps); alloc_d(&host_states.com_z,  num_clumps);
    alloc_d(&host_states.vel_x,  num_clumps); alloc_d(&host_states.vel_y,  num_clumps); alloc_d(&host_states.vel_z,  num_clumps);
    alloc_d(&host_states.omega_x, num_clumps); alloc_d(&host_states.omega_y, num_clumps); alloc_d(&host_states.omega_z, num_clumps);
    alloc_d(&host_states.quat_w, num_clumps); alloc_d(&host_states.quat_x, num_clumps);
    alloc_d(&host_states.quat_y, num_clumps); alloc_d(&host_states.quat_z, num_clumps);
    alloc_d(&host_states.mass, num_clumps);
    alloc_d(&host_states.inertia_xx, num_clumps); alloc_d(&host_states.inertia_xy, num_clumps); alloc_d(&host_states.inertia_xz, num_clumps);
    alloc_d(&host_states.inertia_yy, num_clumps); alloc_d(&host_states.inertia_yz, num_clumps); alloc_d(&host_states.inertia_zz, num_clumps);
    for (double*& q : host_states.multipole_p3) alloc_d(&q, num_clumps);
    alloc_d(&host_states.net_force_x,  num_clumps); alloc_d(&host_states.net_force_y,  num_clumps); alloc_d(&host_states.net_force_z,  num_clumps);
    alloc_d(&host_states.net_torque_x, num_clumps); alloc_d(&host_states.net_torque_y, num_clumps); alloc_d(&host_states.net_torque_z, num_clumps);

    // Allocate template_indices if requested
    if (template_indices != nullptr) {
        CUDA_CHECK(cudaMallocHost(template_indices, num_clumps * sizeof(int32_t)));
    }

    // Copy all fields from device to host
    auto copy_d = [](double* d_src, double* d_dst, int32_t n) {
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, n * sizeof(double), cudaMemcpyDeviceToHost));
    };

    copy_d(device_states.com_x,  host_states.com_x,  num_clumps);
    copy_d(device_states.com_y,  host_states.com_y,  num_clumps);
    copy_d(device_states.com_z,  host_states.com_z,  num_clumps);
    copy_d(device_states.vel_x,  host_states.vel_x,  num_clumps);
    copy_d(device_states.vel_y,  host_states.vel_y,  num_clumps);
    copy_d(device_states.vel_z,  host_states.vel_z,  num_clumps);
    copy_d(device_states.omega_x, host_states.omega_x, num_clumps);
    copy_d(device_states.omega_y, host_states.omega_y, num_clumps);
    copy_d(device_states.omega_z, host_states.omega_z, num_clumps);
    copy_d(device_states.quat_w, host_states.quat_w, num_clumps);
    copy_d(device_states.quat_x, host_states.quat_x, num_clumps);
    copy_d(device_states.quat_y, host_states.quat_y, num_clumps);
    copy_d(device_states.quat_z, host_states.quat_z, num_clumps);
    copy_d(device_states.mass, host_states.mass, num_clumps);
    copy_d(device_states.inertia_xx, host_states.inertia_xx, num_clumps); copy_d(device_states.inertia_xy, host_states.inertia_xy, num_clumps); copy_d(device_states.inertia_xz, host_states.inertia_xz, num_clumps);
    copy_d(device_states.inertia_yy, host_states.inertia_yy, num_clumps); copy_d(device_states.inertia_yz, host_states.inertia_yz, num_clumps); copy_d(device_states.inertia_zz, host_states.inertia_zz, num_clumps);
    for (int q = 0; q < 19; ++q) copy_d(device_states.multipole_p3[q], host_states.multipole_p3[q], num_clumps);
    copy_d(device_states.net_force_x,  host_states.net_force_x,  num_clumps);
    copy_d(device_states.net_force_y,  host_states.net_force_y,  num_clumps);
    copy_d(device_states.net_force_z,  host_states.net_force_z,  num_clumps);
    copy_d(device_states.net_torque_x, host_states.net_torque_x, num_clumps);
    copy_d(device_states.net_torque_y, host_states.net_torque_y, num_clumps);
    copy_d(device_states.net_torque_z, host_states.net_torque_z, num_clumps);

    // Fill template_indices with sequential indices
    if (template_indices != nullptr && *template_indices != nullptr) {
        for (int32_t i = 0; i < num_clumps; ++i) {
            (*template_indices)[i] = i;
        }
    }

    return num_clumps;
}

// =============================================================================
// Free host-side clump state buffers allocated by clump_get_states
// =============================================================================
void free_clump_host_state(ClumpStateSoA& host_states, int32_t* template_indices) {
    // Free double arrays
    if (host_states.com_x)  CUDA_CHECK(cudaFreeHost(host_states.com_x));
    if (host_states.com_y)  CUDA_CHECK(cudaFreeHost(host_states.com_y));
    if (host_states.com_z)  CUDA_CHECK(cudaFreeHost(host_states.com_z));
    if (host_states.vel_x)  CUDA_CHECK(cudaFreeHost(host_states.vel_x));
    if (host_states.vel_y)  CUDA_CHECK(cudaFreeHost(host_states.vel_y));
    if (host_states.vel_z)  CUDA_CHECK(cudaFreeHost(host_states.vel_z));
    if (host_states.omega_x) CUDA_CHECK(cudaFreeHost(host_states.omega_x));
    if (host_states.omega_y) CUDA_CHECK(cudaFreeHost(host_states.omega_y));
    if (host_states.omega_z) CUDA_CHECK(cudaFreeHost(host_states.omega_z));
    if (host_states.quat_w) CUDA_CHECK(cudaFreeHost(host_states.quat_w));
    if (host_states.quat_x) CUDA_CHECK(cudaFreeHost(host_states.quat_x));
    if (host_states.quat_y) CUDA_CHECK(cudaFreeHost(host_states.quat_y));
    if (host_states.quat_z) CUDA_CHECK(cudaFreeHost(host_states.quat_z));
    if (host_states.net_force_x)  CUDA_CHECK(cudaFreeHost(host_states.net_force_x));
    if (host_states.net_force_y)  CUDA_CHECK(cudaFreeHost(host_states.net_force_y));
    if (host_states.net_force_z)  CUDA_CHECK(cudaFreeHost(host_states.net_force_z));
    if (host_states.net_torque_x) CUDA_CHECK(cudaFreeHost(host_states.net_torque_x));
    if (host_states.net_torque_y) CUDA_CHECK(cudaFreeHost(host_states.net_torque_y));
    if (host_states.net_torque_z) CUDA_CHECK(cudaFreeHost(host_states.net_torque_z));

    if (host_states.mass)      CUDA_CHECK(cudaFreeHost(host_states.mass));
    if (host_states.inertia_xx) CUDA_CHECK(cudaFreeHost(host_states.inertia_xx)); if (host_states.inertia_xy) CUDA_CHECK(cudaFreeHost(host_states.inertia_xy)); if (host_states.inertia_xz) CUDA_CHECK(cudaFreeHost(host_states.inertia_xz));
    if (host_states.inertia_yy) CUDA_CHECK(cudaFreeHost(host_states.inertia_yy)); if (host_states.inertia_yz) CUDA_CHECK(cudaFreeHost(host_states.inertia_yz)); if (host_states.inertia_zz) CUDA_CHECK(cudaFreeHost(host_states.inertia_zz));
    for (double*& q : host_states.multipole_p3) if (q) CUDA_CHECK(cudaFreeHost(q));

    // Free template_indices
    if (template_indices) CUDA_CHECK(cudaFreeHost(template_indices));

    // Reset pointers
    host_states = ClumpStateSoA();
}

} // namespace cudadem

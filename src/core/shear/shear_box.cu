#include "shear_box.h"

#include "../constant/system_initialize.h"
#include "../wall/wall.h"
#include "../math/verlet_integration.cuh"
#include "../acceleration/device_state_array.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cudadem {
namespace {

DeviceStateArray<int*> wrap_errors;

int*& current_wrap_error() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return wrap_errors[device];
}

__global__ void convert_peculiar_velocity_kernel(
    const double* x, double* vy, double q_omega, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vy[i] -= q_omega * x[i];
}

__global__ void shear_drift_step1_kernel(
    double* x, double* y, double* z,
    double* vx, double* vy, double* vz,
    double* wx, double* wy, double* wz,
    double* qw, double* qx, double* qy, double* qz,
    double* fx, double* fy, double* fz,
    double* tx, double* ty, double* tz,
    const float* mass, const float* moi,
    double omega, double q, double nu, double dt, int32_t n,const int32_t* clump_ids) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (clump_ids && clump_ids[i]>0)) return;
    const double half_dt = 0.5 * dt;
    const double inv_m = 1.0 / static_cast<double>(mass[i]);
    const double inv_I = 1.0 / static_cast<double>(moi[i]);

    ShearDriftState s{x[i], y[i], z[i],
                      vx[i] + half_dt * fx[i] * inv_m,
                      vy[i] + half_dt * fy[i] * inv_m,
                      vz[i] + half_dt * fz[i] * inv_m};
    s = exact_hill_drift(s, omega, q, nu, dt);
    x[i] = s.x; y[i] = s.y; z[i] = s.z;
    vx[i] = s.vx; vy[i] = s.vy; vz[i] = s.vz;

    const double wxh = wx[i] + half_dt * tx[i] * inv_I;
    const double wyh = wy[i] + half_dt * ty[i] * inv_I;
    const double wzh = wz[i] + half_dt * tz[i] * inv_I;
    wx[i] = wxh; wy[i] = wyh; wz[i] = wzh;

    const double wmag = sqrt(wxh*wxh + wyh*wyh + wzh*wzh);
    double ew, ex, ey, ez;
    if (wmag > 1.0e-10) {
        const double angle = wmag * half_dt;
        const double scale = sin(angle) / wmag;
        ew = cos(angle); ex = scale*wxh; ey = scale*wyh; ez = scale*wzh;
    } else {
        ew = 1.0; ex = half_dt*wxh; ey = half_dt*wyh; ez = half_dt*wzh;
    }
    const double ow=qw[i], ox=qx[i], oy=qy[i], oz=qz[i];
    double nw=ew*ow-ex*ox-ey*oy-ez*oz;
    double nx=ew*ox+ex*ow+ey*oz-ez*oy;
    double ny=ew*oy-ex*oz+ey*ow+ez*ox;
    double nz=ew*oz+ex*oy-ey*ox+ez*ow;
    const double norm=sqrt(nw*nw+nx*nx+ny*ny+nz*nz);
    if (norm > 1.0e-10) {
        const double inv=1.0/norm; nw*=inv; nx*=inv; ny*=inv; nz*=inv;
    } else { nw=1.0; nx=ny=nz=0.0; }
    qw[i]=nw; qx[i]=nx; qy[i]=ny; qz[i]=nz;

    fx[i]=fy[i]=fz[i]=0.0;
    tx[i]=ty[i]=tz[i]=0.0;
}

__device__ bool checked_add_i64(int64_t a, int64_t b, int64_t& out) {
    if ((b > 0 && a > INT64_MAX - b) ||
        (b < 0 && a < INT64_MIN - b)) return false;
    out = a + b;
    return true;
}

__device__ bool checked_mul_i64(int64_t a, int64_t b, int64_t& out) {
    if (a == 0 || b == 0) { out = 0; return true; }
    if ((a == -1 && b == INT64_MIN) || (b == -1 && a == INT64_MIN)) return false;
    if (a > 0) {
        if ((b > 0 && a > INT64_MAX / b) ||
            (b < 0 && b < INT64_MIN / a)) return false;
    } else {
        if ((b > 0 && a < INT64_MIN / b) ||
            (b < 0 && a < INT64_MAX / b)) return false;
    }
    out = a * b;
    return true;
}

__device__ bool floor_i64(double value, int64_t& out) {
    const double k = floor(value);
    if (!isfinite(k) || k < -9223372036854775808.0 ||
        k >= 9223372036854775808.0) return false;
    out = static_cast<int64_t>(k);
    return true;
}

__global__ void shear_wrap_kernel(
    double* x, double* y, double* vy,
    int64_t* wx, int64_t* wy,
    double xmin, double ymin, double Lx, double Ly,
    double delta, double gamma, int64_t cycle,
    int* error, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t kx, ky;
    if (!floor_i64((x[i] - xmin) / Lx, kx)) { atomicExch(error, 1); return; }
    const double xn = x[i] - static_cast<double>(kx) * Lx;
    const double ys = y[i] + static_cast<double>(kx) * delta;
    if (!floor_i64((ys - ymin) / Ly, ky)) { atomicExch(error, 1); return; }
    const double yn = ys - static_cast<double>(ky) * Ly;

    int64_t cycle_term, y_increment, new_wx, new_wy;
    if (!checked_mul_i64(kx, cycle, cycle_term) ||
        !checked_add_i64(ky, cycle_term, y_increment) ||
        !checked_add_i64(wx[i], kx, new_wx) ||
        !checked_add_i64(wy[i], y_increment, new_wy)) {
        atomicExch(error, 2); return;
    }
    x[i] = xn;
    y[i] = yn;
    vy[i] += static_cast<double>(kx) * gamma;
    wx[i] = new_wx;
    wy[i] = new_wy;
}

__global__ void contact_epoch_kernel(
    ContactHistory* history, const int32_t* offsets, const int32_t* counts,
    int32_t stride, bool finish, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ContactHistory* rows = history + offsets[i];
    const int32_t count = min(counts[i], stride);
    for (int32_t k=0; k<count; ++k) {
        if (finish) {
            if (!rows[k].seen_this_step) rows[k].is_active = false;
        } else {
            rows[k].seen_this_step = false;
        }
    }
}

__global__ void compose_hill_force_kernel(
    const double* x, const double* z, const double* vx, const double* vy,
    const float* mass, const double* sx, const double* sy, const double* sz,
    double* dx, double* dy, double* dz, ShearBoxConfig shear, int32_t n) {
    const int32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    double ax,ay,az;
    hill_acceleration(shear,x[i],z[i],vx[i],vy[i],ax,ay,az);
    const double m=static_cast<double>(mass[i]);
    dx[i]=sx[i]+m*ax; dy[i]=sy[i]+m*ay; dz[i]=sz[i]+m*az;
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

} // namespace

void ShearBoxManager::initialize(SimulationState& state, const SimConfig& config,
                                 cudaStream_t stream) {
    if (!config.shear_box.enabled) return;
    int*& d_wrap_error = current_wrap_error();
    if (!d_wrap_error) CUDA_CHECK(cudaMalloc(&d_wrap_error, sizeof(int)));
    const int32_t n=state.particles.num_local;
    if (n > 0 && state.particles.source_record_kind == 1 &&
        config.shear_box.input_velocity == ShearInputVelocity::PECULIAR) {
        const int threads=256, blocks=(n+threads-1)/threads;
        convert_peculiar_velocity_kernel<<<blocks,threads,0,stream>>>(
            state.particles.pos_x,state.particles.vel_y,
            config.shear_box.shear_parameter*config.shear_box.omega,n);
        check_launch();
    }
}

void ShearBoxManager::shutdown() {
    int*& d_wrap_error = current_wrap_error();
    if (d_wrap_error) { cudaFree(d_wrap_error); d_wrap_error=nullptr; }
}

ShearPhase ShearBoxManager::validate_step(
    const SimulationState& state, const SimConfig& config,
    const PeriodicBoundaryConfig& pbc) {
    if (!config.shear_box.enabled) return ShearPhase();
    if (!(state.dt > 0.0) || !std::isfinite(state.dt) ||
        state.current_time + state.dt == state.current_time)
        throw std::runtime_error("shear step cannot advance floating-point time");
    return compute_shear_phase_checked(config.shear_box,pbc.box_length_x,
                                       pbc.box_length_y,state.current_time+state.dt);
}

void ShearBoxManager::drift_step1(SimulationState& state,const SimConfig& config,
                                  double dt,int32_t n,cudaStream_t stream,const int32_t* clump_ids) {
    if (n<=0) return;
    const int threads=config.gpu.block_size, blocks=(n+threads-1)/threads;
    ParticleSoA& p=state.particles;
    shear_drift_step1_kernel<<<blocks,threads,0,stream>>>(
        p.pos_x,p.pos_y,p.pos_z,p.vel_x,p.vel_y,p.vel_z,
        p.omega_x,p.omega_y,p.omega_z,p.quat_w,p.quat_x,p.quat_y,p.quat_z,
        p.force_x,p.force_y,p.force_z,p.torque_x,p.torque_y,p.torque_z,
        p.mass,p.moment_of_inertia,config.shear_box.omega,
        config.shear_box.shear_parameter,config.shear_box.vertical_frequency,dt,n,clump_ids);
    check_launch();
}

ShearPhase ShearBoxManager::wrap(SimulationState& state,const SimConfig& config,
                                 const PeriodicBoundaryConfig& pbc,double time,
                                 int32_t n,cudaStream_t stream) {
    const ShearPhase phase=compute_shear_phase_checked(
        config.shear_box,pbc.box_length_x,pbc.box_length_y,time);
    if (n<=0) return phase;
    int*& d_wrap_error = current_wrap_error();
    if (!d_wrap_error) CUDA_CHECK(cudaMalloc(&d_wrap_error,sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_wrap_error,0,sizeof(int),stream));
    const int threads=config.gpu.block_size, blocks=(n+threads-1)/threads;
    ParticleSoA& p=state.particles;
    shear_wrap_kernel<<<blocks,threads,0,stream>>>(
        p.pos_x,p.pos_y,p.vel_y,p.winding_x,p.winding_y,
        pbc.box_min_x,pbc.box_min_y,pbc.box_length_x,pbc.box_length_y,
        phase.reduced_shift,phase.shear_velocity,phase.cycle,d_wrap_error,n);
    check_launch();
    int error=0;
    CUDA_CHECK(cudaMemcpyAsync(&error,d_wrap_error,sizeof(int),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (error) throw std::overflow_error(error==1
        ? "shear boundary wrap count exceeds int64"
        : "shear winding arithmetic exceeds int64");
    return phase;
}

void ShearBoxManager::kick_step2(SimulationState& state,const SimConfig& config,
                                 double dt,int32_t n,cudaStream_t stream,const int32_t* clump_ids) {
    if (n<=0) return;
    const int threads=config.gpu.block_size, blocks=(n+threads-1)/threads;
    ParticleSoA& p=state.particles;
    verlet_step2_fused_kernel<<<blocks,threads,0,stream>>>(
        p.vel_x,p.vel_y,p.vel_z,p.omega_x,p.omega_y,p.omega_z,
        p.force_x,p.force_y,p.force_z,p.torque_x,p.torque_y,p.torque_z,
        p.mass,p.moment_of_inertia,dt,n,clump_ids);
    check_launch();
}

void ShearBoxManager::begin_contact_epoch(SimulationState& state,int32_t n,
                                          cudaStream_t stream) {
    if (n<=0) return;
    const int threads=256,blocks=(n+threads-1)/threads;
    contact_epoch_kernel<<<blocks,threads,0,stream>>>(state.contact_history.data,
        state.particles.contact_history_offset,state.particles.num_contacts,
        state.contact_history.max_contacts_per_particle,false,n);
    check_launch();
}

void ShearBoxManager::end_contact_epoch(SimulationState& state,int32_t n,
                                        cudaStream_t stream) {
    if (n<=0) return;
    const int threads=256,blocks=(n+threads-1)/threads;
    contact_epoch_kernel<<<blocks,threads,0,stream>>>(state.contact_history.data,
        state.particles.contact_history_offset,state.particles.num_contacts,
        state.contact_history.max_contacts_per_particle,true,n);
    check_launch();
}

void ShearBoxManager::compose_output_force(
    const SimulationState& state,const SimConfig& config,
    const double* sx,const double* sy,const double* sz,
    double* dx,double* dy,double* dz,int32_t n,cudaStream_t stream) {
    if (n<=0) return;
    const int threads=config.gpu.block_size,blocks=(n+threads-1)/threads;
    const ParticleSoA& p=state.particles;
    compose_hill_force_kernel<<<blocks,threads,0,stream>>>(
        p.pos_x,p.pos_z,p.vel_x,p.vel_y,p.mass,sx,sy,sz,dx,dy,dz,
        config.shear_box,n);
    check_launch();
}

ShearDiagnostics ShearBoxManager::compute_diagnostics(
    const SimulationState& state,const SimConfig& config,int32_t n,
    cudaStream_t stream) {
    ShearDiagnostics out;
    if (!config.shear_box.enabled || n<=0) return out;
    std::vector<double> x(n),z(n),vx(n),vy(n),vz(n);
    std::vector<float> mass(n);
#define COPY_DIAG(dst,src,type) CUDA_CHECK(cudaMemcpyAsync(dst.data(),src,static_cast<size_t>(n)*sizeof(type),cudaMemcpyDeviceToHost,stream))
    COPY_DIAG(x,state.particles.pos_x,double); COPY_DIAG(z,state.particles.pos_z,double);
    COPY_DIAG(vx,state.particles.vel_x,double); COPY_DIAG(vy,state.particles.vel_y,double);
    COPY_DIAG(vz,state.particles.vel_z,double); COPY_DIAG(mass,state.particles.mass,float);
#undef COPY_DIAG
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double omega=config.shear_box.omega,q=config.shear_box.shear_parameter;
    const double nu=config.shear_box.vertical_frequency;
    double peculiar=0.0,jacobi=0.0;
    for(int32_t i=0;i<n;++i){
        const double m=mass[i], px=x[i], pz=z[i], ux=vx[i];
        const double total_y=vy[i], uz=vz[i], peculiar_y=total_y+q*omega*px;
        peculiar += 0.5*m*(ux*ux+peculiar_y*peculiar_y+uz*uz);
        jacobi += 0.5*m*(ux*ux+total_y*total_y+uz*uz)
                 -m*q*omega*omega*px*px+0.5*m*nu*nu*pz*pz;
    }
    out.peculiar_kinetic_energy=static_cast<double>(peculiar);
    out.hill_jacobi_invariant=static_cast<double>(jacobi);
    return out;
}

} // namespace cudadem

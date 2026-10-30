#ifndef CUDADEM_YORP_CUH
#define CUDADEM_YORP_CUH

#include "../constant/system_initialize.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace cudadem {

// These values are persisted in the sidecar and intentionally form a stable
// public interface for the output writer.
enum class AdaptiveYorpState : int32_t { RELAXING = 0, ACCELERATING = 1,
    PROVISIONAL_PAUSE = 2, CEILING_RELAX = 3, COMPLETE = 4, FATAL = 5 };
enum class AdaptiveYorpTransition : int32_t { NONE = 0, INITIAL_BASELINE = 1,
    PROVISIONAL_PAUSE = 2, PAUSE_CONFIRMED = 3, PROVISIONAL_CLEARED = 4,
    RESUMED = 5, CEILING_REACHED = 6, COMPLETE = 7, INVALID_METRICS = 8 };

// The host obtains this only at an already-synchronizing output/checkpoint
// boundary.  Normal check intervals do not copy any data to the host.
struct AdaptiveYorpDiagnostic {
    int32_t version = 1;
    int32_t state = static_cast<int32_t>(AdaptiveYorpState::ACCELERATING);
    int32_t transition_reason = static_cast<int32_t>(AdaptiveYorpTransition::NONE);
    int32_t confirmed_cycles = 0;
    int32_t pause_checks = 0;
    int32_t resume_checks = 0;
    int32_t fatal = 0;
    double time = 0.0;
    double transition_time = 0.0;
    double omega_x = 0.0, omega_y = 0.0, omega_z = 0.0;
    double specific_energy_j_per_kg = 0.0;
    double current_baseline_axis_ratio = 1.0;
    double effective_epsilon = 0.0;
};

// Checkpoint sidecars persist precisely the hysteresis state that affects all
// subsequent torque decisions.  Do not silently accept a missing sidecar.
struct AdaptiveYorpCheckpoint {
    uint32_t version = 1;
    uint32_t reserved = 0;
    AdaptiveYorpDiagnostic diagnostic{};
    double baseline_axis_ratio = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    int32_t baseline_recorded = 0;
    int32_t reserved2 = 0;
    double next_check_time = 0.0;
};

struct YorpReduction {
    double mass, rx, ry, rz, px, py, pz;
    double ixx, iyy, izz, ixy, ixz, iyz;
    double lx, ly, lz, energy, self_inertia;
    int invalid;
};

struct YorpReductionSum {
    __host__ __device__ YorpReduction operator()(const YorpReduction& a,
                                                  const YorpReduction& b) const {
        return {a.mass+b.mass, a.rx+b.rx, a.ry+b.ry, a.rz+b.rz,
                a.px+b.px, a.py+b.py, a.pz+b.pz,
                a.ixx+b.ixx, a.iyy+b.iyy, a.izz+b.izz,
                a.ixy+b.ixy, a.ixz+b.ixz, a.iyz+b.iyz,
                a.lx+b.lx, a.ly+b.ly, a.lz+b.lz, a.energy+b.energy, a.self_inertia+b.self_inertia,
                a.invalid+b.invalid};
    }
};

struct YorpDeviceState {
    AdaptiveYorpDiagnostic diagnostic;
    double baseline_axis_ratio;
    double com_x, com_y, com_z;
    int baseline_recorded;
};

// Complete closed-boundary runtime, including legacy prescribed-spin modes.
// Reduction buffers and the next-increment scratch value are reconstructed.
struct YorpRuntimeCheckpoint {
    uint32_t version = 1, has_device_state = 0;
    uint32_t have_last_diagnostic = 0, reserved = 0;
    double next_time = 0.0, next_check_time = 0.0, accumulated_omega = 0.0;
    int64_t step_counter = 0;
    double mode3_start_time = -1.0, mode3_omega_init = 0.0;
    AdaptiveYorpDiagnostic last_diagnostic{};
    YorpDeviceState device_state{};
};

__device__ inline bool yorp_finite(double x) { return isfinite(x); }

__host__ __device__ inline double yorp_effective_epsilon(
    double epsilon0, double reduction_fraction, double minimum_fraction,
    int32_t confirmed_cycles) {
    const double retained = pow(fmax(0.0, 1.0 - reduction_fraction),
                                static_cast<double>(confirmed_cycles));
    return epsilon0 * fmax(minimum_fraction, retained);
}

__global__ static void yorp_metric_contributions_kernel(
    const double* x, const double* y, const double* z,
    const double* vx, const double* vy, const double* vz,
    const double* ox, const double* oy, const double* oz,
    const float* mass, const float* inertia, const int32_t* clump_id,
    double frame_x, double frame_y, double frame_z,
    YorpReduction* out, int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double m = static_cast<double>(mass[i]);
    const double r0=x[i], r1=y[i], r2=z[i];
    // v_inertial = v_rotating + Omega_frame x r.
    const double u0=vx[i] + frame_y*r2 - frame_z*r1;
    const double u1=vy[i] + frame_z*r0 - frame_x*r2;
    const double u2=vz[i] + frame_x*r1 - frame_y*r0;
    const double w0=ox[i]+frame_x, w1=oy[i]+frame_y, w2=oz[i]+frame_z;
    const double self=static_cast<double>(inertia[i]);
    const bool bad = !(m > 0.0) || !(self >= 0.0) || clump_id[i] > 0 ||
        !yorp_finite(r0) || !yorp_finite(r1) || !yorp_finite(r2) ||
        !yorp_finite(u0) || !yorp_finite(u1) || !yorp_finite(u2) ||
        !yorp_finite(w0) || !yorp_finite(w1) || !yorp_finite(w2);
    if (bad) { out[i] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}; return; }
    out[i] = {m, m*r0, m*r1, m*r2, m*u0, m*u1, m*u2,
              m*(r1*r1+r2*r2)+self, m*(r0*r0+r2*r2)+self,
              m*(r0*r0+r1*r1)+self, -m*r0*r1, -m*r0*r2, -m*r1*r2,
              m*(r1*u2-r2*u1)+self*w0, m*(r2*u0-r0*u2)+self*w1,
              m*(r0*u1-r1*u0)+self*w2,
              0.5*m*(u0*u0+u1*u1+u2*u2)+0.5*self*(w0*w0+w1*w1+w2*w2), self, 0};
}

__device__ inline void yorp_eigenvalues(double a, double d, double f,
                                          double b, double c, double e,
                                          double& l1, double& l2, double& l3) {
    const double q=(a+d+f)/3.0;
    const double p2=(a-q)*(a-q)+(d-q)*(d-q)+(f-q)*(f-q)+2.0*(b*b+c*c+e*e);
    const double p=sqrt(fmax(0.0,p2/6.0));
    if (!(p > 0.0) || !isfinite(p)) { l1=l2=l3=q; return; }
    const double A00=(a-q)/p,A01=b/p,A02=c/p,A11=(d-q)/p,A12=e/p,A22=(f-q)/p;
    const double det=A00*(A11*A22-A12*A12)-A01*(A01*A22-A12*A02)+A02*(A01*A12-A11*A02);
    const double phi=acos(fmax(-1.0,fmin(1.0,det*0.5)))/3.0;
    l1=q+2*p*cos(phi); l3=q+2*p*cos(phi+2.0943951023931954923); l2=3*q-l1-l3;
    if(l1<l2){double t=l1;l1=l2;l2=t;} if(l2<l3){double t=l2;l2=l3;l3=t;} if(l1<l2){double t=l1;l1=l2;l2=t;}
}

__global__ static void yorp_evaluate_kernel(YorpDeviceState* s, const YorpReduction* r,
    double now, double pause_energy, double resume_energy, int pause_need, int resume_need,
    double axis_drop, double epsilon0, double epsilon_drop, double minimum_fraction,
    double max_omega) {
    if (threadIdx.x || blockIdx.x) return;
    AdaptiveYorpDiagnostic& d=s->diagnostic;
    if (d.state == static_cast<int>(AdaptiveYorpState::FATAL) || d.state == static_cast<int>(AdaptiveYorpState::COMPLETE)) return;
    if (r->invalid || !(r->mass > 0.0) || !yorp_finite(r->energy)) {
        d.state=static_cast<int>(AdaptiveYorpState::FATAL); d.fatal=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INVALID_METRICS); d.transition_time=now; d.time=now; return;
    }
    const double M=r->mass, cx=r->rx/M, cy=r->ry/M, cz=r->rz/M;
    const double v0=r->px/M,v1=r->py/M,v2=r->pz/M;
    // Shift origin moments to mass CoM.  This is the exact full 3-D inertia.
    const double ixx=r->ixx-M*(cy*cy+cz*cz), iyy=r->iyy-M*(cx*cx+cz*cz), izz=r->izz-M*(cx*cx+cy*cy);
    const double ixy=r->ixy+M*cx*cy, ixz=r->ixz+M*cx*cz, iyz=r->iyz+M*cy*cz;
    const double lx=r->lx-(cy*r->pz-cz*r->py), ly=r->ly-(cz*r->px-cx*r->pz), lz=r->lz-(cx*r->py-cy*r->px);
    const double det=ixx*(iyy*izz-iyz*iyz)-ixy*(ixy*izz-iyz*ixz)+ixz*(ixy*iyz-iyy*ixz);
    const double scale=fmax(fmax(fabs(ixx),fabs(iyy)),fabs(izz));
    if (!(scale > 0.0) || !yorp_finite(det) || fabs(det) <= 1e-18*scale*scale*scale) {
        d.state=static_cast<int>(AdaptiveYorpState::FATAL); d.fatal=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INVALID_METRICS); d.transition_time=now; d.time=now; return;
    }
    const double wx=((iyy*izz-iyz*iyz)*lx+(ixz*iyz-ixy*izz)*ly+(ixy*iyz-ixz*iyy)*lz)/det;
    const double wy=((ixz*iyz-ixy*izz)*lx+(ixx*izz-ixz*ixz)*ly+(ixy*ixz-ixx*iyz)*lz)/det;
    const double wz=((ixy*iyz-ixz*iyy)*lx+(ixy*ixz-ixx*iyz)*ly+(ixx*iyy-ixy*ixy)*lz)/det;
    const double rigid=0.5*(lx*wx+ly*wy+lz*wz);
    const double residual=r->energy-0.5*M*(v0*v0+v1*v1+v2*v2)-rigid;
    const double residual_tolerance=1e-11*fmax(fmax(r->energy, rigid), 1.0);
    if (residual < -residual_tolerance) {
        d.state=static_cast<int>(AdaptiveYorpState::FATAL); d.fatal=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INVALID_METRICS); d.transition_time=now; d.time=now; return;
    }
    const double nonrigid=fmax(0.0, residual);
    // Report shape uses the mass-weighted RMS extents, i.e. eigenvalues of
    // the CoM covariance, not principal moments of inertia.  Subtract the
    // sphere self-inertia before recovering the orbital second moment.
    const double cxx=0.5*(iyy+izz-ixx-r->self_inertia)/M;
    const double cyy=0.5*(ixx+izz-iyy-r->self_inertia)/M;
    const double czz=0.5*(ixx+iyy-izz-r->self_inertia)/M;
    const double cxy=-ixy/M, cxz=-ixz/M, cyz=-iyz/M;
    double l1,l2,l3; yorp_eigenvalues(cxx,cyy,czz,cxy,cxz,cyz,l1,l2,l3);
    const double minor2=ixx*iyy-ixy*ixy;
    if (!(ixx > 0.0) || !(minor2 > 1e-18*scale*scale) || !(det > 1e-18*scale*scale*scale) ||
        !yorp_finite(wx) || !yorp_finite(wy) || !yorp_finite(wz)) {
        d.state=static_cast<int>(AdaptiveYorpState::FATAL); d.fatal=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INVALID_METRICS); d.transition_time=now; d.time=now; return;
    }
    const double covariance_scale=fmax(fmax(fabs(cxx),fabs(cyy)),fabs(czz));
    if (!(l1 > 0.0) || l3 < -1e-12*fmax(covariance_scale, 1.0)) {
        d.state=static_cast<int>(AdaptiveYorpState::FATAL); d.fatal=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INVALID_METRICS); d.transition_time=now; d.time=now; return;
    }
    const double axis=sqrt(fmax(0.0,l3/l1));
    if (!s->baseline_recorded) { s->baseline_axis_ratio=axis; s->baseline_recorded=1;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::INITIAL_BASELINE); d.transition_time=now; }
    s->com_x=cx; s->com_y=cy; s->com_z=cz;
    d.time=now; d.omega_x=wx; d.omega_y=wy; d.omega_z=wz;
    d.specific_energy_j_per_kg=nonrigid/M;
    d.current_baseline_axis_ratio=(s->baseline_axis_ratio > 0.0) ? axis/s->baseline_axis_ratio : 0.0;
    if (wz >= max_omega && d.state != static_cast<int>(AdaptiveYorpState::CEILING_RELAX)) {
        d.state=static_cast<int>(AdaptiveYorpState::CEILING_RELAX); d.resume_checks=0;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::CEILING_REACHED); d.transition_time=now;
    }
    const double drop=(s->baseline_axis_ratio>0.0) ? (s->baseline_axis_ratio-axis)/s->baseline_axis_ratio : 0.0;
    const bool breach=d.specific_energy_j_per_kg >= pause_energy || drop >= axis_drop;
    if (d.state == static_cast<int>(AdaptiveYorpState::ACCELERATING)) {
        if (breach) { d.state=static_cast<int>(AdaptiveYorpState::PROVISIONAL_PAUSE); d.pause_checks=1;
            d.transition_reason=static_cast<int>(AdaptiveYorpTransition::PROVISIONAL_PAUSE); d.transition_time=now; }
    } else if (d.state == static_cast<int>(AdaptiveYorpState::PROVISIONAL_PAUSE)) {
        if (breach) { ++d.pause_checks; if (d.pause_checks >= pause_need) { d.state=static_cast<int>(AdaptiveYorpState::RELAXING); ++d.confirmed_cycles; d.resume_checks=0;
                d.transition_reason=static_cast<int>(AdaptiveYorpTransition::PAUSE_CONFIRMED); d.transition_time=now; } }
        else { d.state=static_cast<int>(AdaptiveYorpState::ACCELERATING); d.pause_checks=0;
            d.transition_reason=static_cast<int>(AdaptiveYorpTransition::PROVISIONAL_CLEARED); d.transition_time=now; }
    } else if (d.state == static_cast<int>(AdaptiveYorpState::RELAXING) || d.state == static_cast<int>(AdaptiveYorpState::CEILING_RELAX)) {
        if (d.specific_energy_j_per_kg <= resume_energy) {
            ++d.resume_checks;
            if (d.resume_checks >= resume_need) {
                if (d.state == static_cast<int>(AdaptiveYorpState::CEILING_RELAX)) { d.state=static_cast<int>(AdaptiveYorpState::COMPLETE); d.transition_reason=static_cast<int>(AdaptiveYorpTransition::COMPLETE); }
                else { d.state=static_cast<int>(AdaptiveYorpState::ACCELERATING); s->baseline_axis_ratio=axis; d.pause_checks=0; d.transition_reason=static_cast<int>(AdaptiveYorpTransition::RESUMED); }
                d.transition_time=now;
            }
        } else d.resume_checks=0;
    }
    d.effective_epsilon=yorp_effective_epsilon(
        epsilon0, epsilon_drop, minimum_fraction, d.confirmed_cycles);
}

__global__ static void yorp_prepare_increment_kernel(YorpDeviceState* state, double* increment,
    double epsilon0, double epsilon_drop, double minimum_fraction, double max_omega,
    double now, double dt) {
    if (threadIdx.x || blockIdx.x) return;
    AdaptiveYorpDiagnostic& d=state->diagnostic;
    if (d.state != static_cast<int>(AdaptiveYorpState::ACCELERATING) || d.fatal) { *increment=0.0; return; }
    const double effective=yorp_effective_epsilon(
        epsilon0, epsilon_drop, minimum_fraction, d.confirmed_cycles);
    const double requested=fmax(0.0,effective*dt);
    // The ceiling constrains only our positive injection.  A naturally
    // overspun aggregate is never braked by this controller.
    const double budget=fmax(0.0,max_omega-d.omega_z);
    *increment=fmin(requested,budget);
    // Keep the budget current between metric observations.  This prevents a
    // stale fitted omega from spending the same remaining headroom every step.
    if (*increment > 0.0) {
        d.omega_z += *increment;
        if (d.omega_z >= max_omega) {
            d.state=static_cast<int>(AdaptiveYorpState::CEILING_RELAX);
            d.resume_checks=0;
            d.transition_reason=static_cast<int>(AdaptiveYorpTransition::CEILING_REACHED);
            d.transition_time=now;
        }
    } else if (budget <= 0.0) {
        d.state=static_cast<int>(AdaptiveYorpState::CEILING_RELAX);
        d.resume_checks=0;
        d.transition_reason=static_cast<int>(AdaptiveYorpTransition::CEILING_REACHED);
        d.transition_time=now;
    }
}

__global__ static void yorp_apply_increment_kernel(double* vx,double* vy,double* omega_z,
    const double* x,const double* y,const double* increment,const YorpDeviceState* state,int32_t n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; const double q=*increment;
    if(q==0.0)return; omega_z[i]+=q; vx[i]-=q*(y[i]-state->com_y); vy[i]+=q*(x[i]-state->com_x);
}

// The distributed coordinator writes one local reduction per logical GPU in
// GPU-index order, then launches this single-thread kernel on GPU 0.  The
// serial order is intentional: it gives reductions a stable order independent
// of host-thread arrival and of the domain partitioning.
__global__ static void yorp_sum_reductions_kernel(const YorpReduction* inputs,
                                                   int32_t count,
                                                   YorpReduction* output) {
    if (threadIdx.x || blockIdx.x) return;
    YorpReduction total{};
    for (int32_t i = 0; i < count; ++i) total = YorpReductionSum{}(total, inputs[i]);
    *output = total;
}

class YorpManager {
    YorpDeviceState* d_state_=nullptr;
    YorpReduction* d_partial_=nullptr;
    YorpReduction* d_reduced_=nullptr;
    double* d_increment_=nullptr;
    void* d_temp_=nullptr; size_t temp_bytes_=0; int32_t capacity_=0;
    int device_=-1;
    bool initialized_=false; double next_time_=0.0; double next_check_time_=0.0;
    double accumulated_omega_=0.0; int64_t step_counter_=0;
    double mode3_start_time_=-1.0, mode3_omega_init_=0.0;
    AdaptiveYorpDiagnostic last_diagnostic_{}; bool have_last_diagnostic_=false;
    void ensure_state();
    void ensure(int32_t n, cudaStream_t stream=0);
    void periodic_check(ParticleSoA&,int32_t,double,const SimConfig&,int,cudaStream_t);
public:
    YorpManager()=default; ~YorpManager(){shutdown();}
    void initialize(const SimConfig&); void update(ParticleSoA&,double,double,const SimConfig&,int=256,cudaStream_t=0);
    // General YORP path for a domain-owned prefix.  This covers the legacy
    // modes as well as adaptive operation; `update()` remains the single-GPU
    // compatibility wrapper.
    void update_owned(ParticleSoA&, int32_t owned_count, double, double,
                      const SimConfig&, int=256, cudaStream_t=0);
    // Call after Verlet step 2, and once at t=0 after initialisation.  This
    // queues metrics/state transitions but does not synchronize the stream.
    void observe_adaptive(ParticleSoA&,double,const SimConfig&,int=256,cudaStream_t=0); void shutdown();
    bool adaptive_enabled() const { return initialized_ && d_state_ != nullptr; }
    bool consume_adaptive_diagnostic(AdaptiveYorpDiagnostic*, cudaStream_t=0) const;
    bool adaptive_should_terminate(cudaStream_t=0) const;
    bool get_adaptive_checkpoint(AdaptiveYorpCheckpoint*, cudaStream_t=0) const;
    bool restore_adaptive_checkpoint(const AdaptiveYorpCheckpoint&, cudaStream_t=0);
    bool get_runtime_checkpoint(YorpRuntimeCheckpoint*, cudaStream_t=0) const;
    bool restore_runtime_checkpoint(const YorpRuntimeCheckpoint&, cudaStream_t=0);
    double get_accumulated_omega() const { return accumulated_omega_; }

    // These are deliberately count-based: multi-GPU callers must pass their
    // owned prefix, never ParticleSoA::num_particles after halo exchange.
    void enqueue_owned_reduction(ParticleSoA&, int32_t owned_count,
                                 const SimConfig&, int=256, cudaStream_t=0);
    void evaluate_adaptive_reduction(const YorpReduction* device_reduction,
                                     double now, const SimConfig&, cudaStream_t=0);
    void apply_adaptive_increment_owned(ParticleSoA&, int32_t owned_count,
                                        double now, double dt, const SimConfig&,
                                        int=256, cudaStream_t=0);

    // The distributed coordinator only moves these compact device records.
    // It never touches particle data or manager-owned allocation internals.
    YorpReduction* device_reduction() const { return d_reduced_; }
    YorpDeviceState* device_state() const { return d_state_; }
    int device_ordinal() const { return device_; }
};

inline void YorpManager::ensure_state() {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (device_ < 0) device_ = current_device;
    // A manager has device-affine allocations.  Catch accidental reuse from a
    // different persistent multi-GPU worker before CUDA silently corrupts the
    // intended ownership model.
    if (device_ != current_device) {
        std::fprintf(stderr, "YorpManager used from CUDA device %d after binding to device %d\n",
                     current_device, device_);
        std::abort();
    }
    if (!d_state_) { cudaMalloc(&d_state_,sizeof(YorpDeviceState)); YorpDeviceState zero{}; zero.diagnostic.state=static_cast<int>(AdaptiveYorpState::ACCELERATING); cudaMemcpy(d_state_,&zero,sizeof(zero),cudaMemcpyHostToDevice); cudaMalloc(&d_increment_,sizeof(double)); }
}
inline void YorpManager::ensure(int32_t n, cudaStream_t stream) {
    ensure_state();
    // CUB does not accept a zero-item reduction.  An empty domain contributes
    // the additive identity, allowing a temporarily empty RCB partition.
    if (n <= 0) {
        if (!d_reduced_) cudaMalloc(&d_reduced_, sizeof(YorpReduction));
        return;
    }
    if(n<=capacity_)return;
    // A distributed producer may reach this resize after its coordinator has
    // enqueued (but not yet completed) the prior peer-consumption dependency.
    // Drain that stream before replacing the source allocation.  This occurs
    // only when the owned high-water mark grows, never on the steady path.
    if (d_reduced_) CUDA_CHECK(cudaStreamSynchronize(stream));
    if(d_partial_)cudaFree(d_partial_); d_partial_=nullptr; if(d_reduced_)cudaFree(d_reduced_); d_reduced_=nullptr; if(d_temp_)cudaFree(d_temp_); d_temp_=nullptr; temp_bytes_=0;
    cudaMalloc(&d_partial_,static_cast<size_t>(n)*sizeof(YorpReduction)); cudaMalloc(&d_reduced_,sizeof(YorpReduction));
    YorpReduction identity{}; cub::DeviceReduce::Reduce(nullptr,temp_bytes_,d_partial_,d_reduced_,n,YorpReductionSum{},identity);
    cudaMalloc(&d_temp_,temp_bytes_); capacity_=n;
}
inline void YorpManager::shutdown(){ if(device_>=0) cudaSetDevice(device_); if(d_state_)cudaFree(d_state_);if(d_partial_)cudaFree(d_partial_);if(d_reduced_)cudaFree(d_reduced_);if(d_increment_)cudaFree(d_increment_);if(d_temp_)cudaFree(d_temp_);d_state_=nullptr;d_partial_=nullptr;d_reduced_=nullptr;d_increment_=nullptr;d_temp_=nullptr;capacity_=0;temp_bytes_=0;device_=-1;initialized_=false; }
inline void YorpManager::initialize(const SimConfig& c){ if(!c.acceleration.yorp.enabled)return; initialized_=true;next_time_=c.acceleration.yorp.start_time;next_check_time_=c.acceleration.yorp.start_time;accumulated_omega_=c.acceleration.yorp.omega_init;step_counter_=0;mode3_start_time_=-1.0;mode3_omega_init_=accumulated_omega_; }
inline void YorpManager::enqueue_owned_reduction(ParticleSoA& p,int32_t owned_count,const SimConfig& c,int block,cudaStream_t stream){
    const int32_t count = owned_count > 0 ? owned_count : 0;
    ensure(count,stream);
    if (count == 0) {
        // Keep the identity write behind the coordinator's prior-consumption
        // dependency and before this epoch's ready event on the same stream.
        CUDA_CHECK(cudaMemsetAsync(d_reduced_, 0, sizeof(YorpReduction), stream));
        return;
    }
    const int nb=(count+block-1)/block; const double fx=c.rotating_frame.enabled?c.rotating_frame.omega.x:0.0,fy=c.rotating_frame.enabled?c.rotating_frame.omega.y:0.0,fz=c.rotating_frame.enabled?c.rotating_frame.omega.z:0.0;
    yorp_metric_contributions_kernel<<<nb,block,0,stream>>>(p.pos_x,p.pos_y,p.pos_z,p.vel_x,p.vel_y,p.vel_z,p.omega_x,p.omega_y,p.omega_z,p.mass,p.moment_of_inertia,p.clump_id,fx,fy,fz,d_partial_,count); CUDA_CHECK(cudaGetLastError());
    YorpReduction identity{}; cub::DeviceReduce::Reduce(d_temp_,temp_bytes_,d_partial_,d_reduced_,count,YorpReductionSum{},identity,stream);
}
inline void YorpManager::evaluate_adaptive_reduction(const YorpReduction* reduction,double now,const SimConfig& c,cudaStream_t stream){
    // Do not resize or initialize d_reduced_ here: it may be the exact
    // device pointer the distributed aggregate just populated on this stream.
    ensure_state(); const auto& y=c.acceleration.yorp;
    yorp_evaluate_kernel<<<1,1,0,stream>>>(d_state_,reduction,now,y.pause_specific_energy_j_per_kg,y.resume_specific_energy_j_per_kg,y.pause_confirmation_checks,y.resume_confirmation_checks,y.axis_ratio_drop_fraction,y.epsilon,y.epsilon_reduction_fraction_per_cycle,y.minimum_epsilon_fraction,y.max_omega_rad_s); CUDA_CHECK(cudaGetLastError());
}
inline void YorpManager::periodic_check(ParticleSoA& p,int32_t count,double now,const SimConfig& c,int block,cudaStream_t stream){ enqueue_owned_reduction(p,count,c,block,stream); evaluate_adaptive_reduction(d_reduced_,now,c,stream); }
inline void YorpManager::observe_adaptive(ParticleSoA& p,double now,const SimConfig& c,int block,cudaStream_t stream) { if(!initialized_ || !c.acceleration.yorp.adaptive || now<c.acceleration.yorp.start_time)return; if(now+1e-12>=next_check_time_){periodic_check(p,p.num_particles,now,c,block,stream);do{next_check_time_+=c.acceleration.yorp.check_interval_seconds;}while(next_check_time_<=now);} }
inline void YorpManager::apply_adaptive_increment_owned(ParticleSoA& p,int32_t owned_count,double now,double dt,const SimConfig& c,int block,cudaStream_t stream){
    if(!initialized_ || !c.acceleration.yorp.enabled ||
       !c.acceleration.yorp.adaptive ||
       now<c.acceleration.yorp.start_time ||
       now>=c.acceleration.yorp.stop_time) return;
    const int32_t count=owned_count>0?owned_count:0; ensure(count,stream);
    const auto& y=c.acceleration.yorp;
    yorp_prepare_increment_kernel<<<1,1,0,stream>>>(d_state_,d_increment_,y.epsilon,y.epsilon_reduction_fraction_per_cycle,y.minimum_epsilon_fraction,y.max_omega_rad_s,now,dt); CUDA_CHECK(cudaGetLastError());
    if (count == 0) return;
    const int nb=(count+block-1)/block;
    yorp_apply_increment_kernel<<<nb,block,0,stream>>>(p.vel_x,p.vel_y,p.omega_z,p.pos_x,p.pos_y,d_increment_,d_state_,count); CUDA_CHECK(cudaGetLastError());
}
inline void YorpManager::update(ParticleSoA& p,double now,double dt,const SimConfig& c,int block,cudaStream_t stream){ update_owned(p,p.num_particles,now,dt,c,block,stream); }
inline void YorpManager::update_owned(ParticleSoA& p,int32_t owned_count,double now,double dt,const SimConfig& c,int block,cudaStream_t stream){ if(!initialized_||!c.acceleration.yorp.enabled||now<c.acceleration.yorp.start_time||now>=c.acceleration.yorp.stop_time)return; ++step_counter_; const int32_t count=owned_count>0?owned_count:0; const auto& y=c.acceleration.yorp;
    if(y.adaptive){ apply_adaptive_increment_owned(p,count,now,dt,c,block,stream); return; }
    double q=0; bool apply=false; if(y.mode==1&&now>=next_time_){q=y.increment;next_time_+=y.dt_check;apply=true;} else if(y.mode==2){q=y.epsilon*dt;apply=true;} else if(y.mode==3){ if(mode3_start_time_<0)mode3_start_time_=now; const double elapsed=now-mode3_start_time_; double period=elapsed<=y.fast_spinup_time?y.T_start_period+(y.T_inter_period-y.T_start_period)*(elapsed/y.fast_spinup_time):elapsed<=y.fast_spinup_time+y.slow_spinup_time?y.T_inter_period+(y.T_final_period-y.T_inter_period)*((elapsed-y.fast_spinup_time)/y.slow_spinup_time):y.T_final_period; const double target=2.0*M_PI/(period*3600.0);q=target-accumulated_omega_;apply=true;} if(apply&&fabs(q)>1e-20){accumulated_omega_+=q;ensure(count,stream);cudaMemcpyAsync(d_increment_,&q,sizeof(q),cudaMemcpyHostToDevice,stream);if(count==0)return;const int nb=(count+block-1)/block;yorp_apply_increment_kernel<<<nb,block,0,stream>>>(p.vel_x,p.vel_y,p.omega_z,p.pos_x,p.pos_y,d_increment_,d_state_,count);CUDA_CHECK(cudaGetLastError());}}
inline bool YorpManager::consume_adaptive_diagnostic(AdaptiveYorpDiagnostic* out,cudaStream_t stream) const { if(!d_state_||!out)return false; CUDA_CHECK(cudaMemcpyAsync(out,&d_state_->diagnostic,sizeof(*out),cudaMemcpyDeviceToHost,stream)); CUDA_CHECK(cudaStreamSynchronize(stream)); const_cast<YorpManager*>(this)->last_diagnostic_=*out; const_cast<YorpManager*>(this)->have_last_diagnostic_=true; return true; }
inline bool YorpManager::adaptive_should_terminate(cudaStream_t) const { return have_last_diagnostic_ && (last_diagnostic_.state==static_cast<int>(AdaptiveYorpState::COMPLETE)||last_diagnostic_.state==static_cast<int>(AdaptiveYorpState::FATAL)); }
inline bool YorpManager::get_adaptive_checkpoint(AdaptiveYorpCheckpoint* out,cudaStream_t stream) const { if(!d_state_||!out)return false; YorpDeviceState h{}; CUDA_CHECK(cudaMemcpyAsync(&h,d_state_,sizeof(h),cudaMemcpyDeviceToHost,stream)); CUDA_CHECK(cudaStreamSynchronize(stream)); const_cast<YorpManager*>(this)->last_diagnostic_=h.diagnostic;const_cast<YorpManager*>(this)->have_last_diagnostic_=true; out->diagnostic=h.diagnostic;out->baseline_axis_ratio=h.baseline_axis_ratio;out->com_x=h.com_x;out->com_y=h.com_y;out->com_z=h.com_z;out->baseline_recorded=h.baseline_recorded;out->next_check_time=next_check_time_;return true; }
inline bool YorpManager::restore_adaptive_checkpoint(const AdaptiveYorpCheckpoint& in,cudaStream_t stream){if(in.version!=1)return false;ensure(1,stream);YorpDeviceState h{};h.diagnostic=in.diagnostic;h.baseline_axis_ratio=in.baseline_axis_ratio;h.com_x=in.com_x;h.com_y=in.com_y;h.com_z=in.com_z;h.baseline_recorded=in.baseline_recorded;next_check_time_=in.next_check_time;CUDA_CHECK(cudaMemcpyAsync(d_state_,&h,sizeof(h),cudaMemcpyHostToDevice,stream));return true;}

inline bool YorpManager::get_runtime_checkpoint(YorpRuntimeCheckpoint* out, cudaStream_t stream) const {
    if (!initialized_ || !out) return false;
    *out = YorpRuntimeCheckpoint{};
    out->next_time=next_time_; out->next_check_time=next_check_time_;
    out->accumulated_omega=accumulated_omega_; out->step_counter=step_counter_;
    out->mode3_start_time=mode3_start_time_; out->mode3_omega_init=mode3_omega_init_;
    out->last_diagnostic=last_diagnostic_; out->have_last_diagnostic=have_last_diagnostic_;
    out->has_device_state=d_state_ != nullptr;
    if (d_state_) {
        CUDA_CHECK(cudaMemcpyAsync(&out->device_state, d_state_, sizeof(YorpDeviceState), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return true;
}
inline bool YorpManager::restore_runtime_checkpoint(const YorpRuntimeCheckpoint& in, cudaStream_t stream) {
    if (!initialized_ || in.version != 1 || in.has_device_state > 1 || in.have_last_diagnostic > 1 ||
        in.step_counter < 0 || !std::isfinite(in.next_time) || !std::isfinite(in.next_check_time) ||
        !std::isfinite(in.accumulated_omega) || !std::isfinite(in.mode3_start_time) || !std::isfinite(in.mode3_omega_init))
        return false;
    next_time_=in.next_time; next_check_time_=in.next_check_time;
    accumulated_omega_=in.accumulated_omega; step_counter_=in.step_counter;
    mode3_start_time_=in.mode3_start_time; mode3_omega_init_=in.mode3_omega_init;
    last_diagnostic_=in.last_diagnostic; have_last_diagnostic_=in.have_last_diagnostic;
    if (in.has_device_state) {
        ensure_state();
        CUDA_CHECK(cudaMemcpyAsync(d_state_, &in.device_state, sizeof(YorpDeviceState), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else if (d_state_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_state_)); d_state_=nullptr;
        CUDA_CHECK(cudaFree(d_increment_)); d_increment_=nullptr;
    }
    return true;
}

} // namespace cudadem
#endif

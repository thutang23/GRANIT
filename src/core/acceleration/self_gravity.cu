// =============================================================================
// Self-Gravity Calculation for DEM Simulation
// =============================================================================
// Implements particle-particle gravitational interaction using direct N² summation.
//
// For N=4600 particles, N² = ~21 million pairs, which is efficient on GPU.
// For larger N (>50000), consider Barnes-Hut tree or FMM.
//
// Key Features:
// - Direct N² O(N²) summation (suitable for N < 50000)
// - Cubic-spline gravity softening (matches SPHSOL KernelSoften, kdtree.cc)
// - Support for periodic boundaries via minimum image convention
// - Toggle via config.acceleration.enable_self_gravity
//
// Physics:
// F_ij = G * m_i * m_j * phi'(r, h_soft) / r * r_vec
//
// Where:
//   r = |r_ij|  — distance between particles
//   h_soft = 0.5 * (h_i + h_j) by default, or h_i in the explicit
//            SPHSOL comparison mode
//   phi'(r, h) — cubic spline gravity softening kernel
//
// For h_soft <= 0 (DEM particles with no smoothing length):
//   Falls back to exact Newtonian 1/r² (DEM-safe).
// For r >= 2*h_soft:
//   Exact Newtonian 1/r².
// For r < 2*h_soft:
//   Smoothly softened (Price & Monaghan 2007, Hernquist & Katz 1989).
//
// Reference: SPHSOL lib/SPHSOL/src/kernel.cc:93-122 (KernelSoften)
//   and lib/SPHSOL/src/kdtree.cc:CalcSelfGravityRecursive:324,340,366,376.
// The old max(r, h_ij) formula (solver.cc:1051, commented out in SPHSOL)
// has a discontinuous 2nd derivative at r=h and was replaced to fix
// energy non-conservation in SPH hex-packed relaxation.
// =============================================================================

#include "acceleration.h"
#include "../particle_model/particle.h"
#include "../wall/wall.h"  // For periodic boundary config
#include "../sph/sph_gravity_kernel.cuh"  // gravity_softening_force_factor_double
#include "../sph/sph_gravity_correction.h"
#include "../sph/sph_physics.h"
#include "../sph/sph_neighbor_query.cuh"
#include "../particle_model/clump.h"
#include "body_fitted_fmm/clump_p3_gravity.cuh"
#include "body_fitted_fmm/clump_gravity_body_view.cuh"
#include "shear_gravity_images.cuh"
#ifdef CUDADEM_HAVE_LET
#include "../../domain/let_communicator.h"
#endif
#ifdef CUDADEM_CPU_MOCK
#include "cuda_utils.h"
#else
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif
#include <cstdint>
#include <cmath>
#include <iostream>

namespace cudadem {

__global__ void sph_gravity_softening_correction_kernel(
    ParticleSoA p, NeighborData neighbors, double G, double epsilon, int count, bool outer) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // h may have evolved since the hydro adjacency was built. The point BVH
    // remains valid at these positions, but its cached support list may not.
    neighbors.sph_adjacency_valid = false;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for_each_sph_neighbor(i, static_cast<float>(p.pos_x[i]),
        static_cast<float>(p.pos_y[i]), static_cast<float>(p.pos_z[i]),
        p.smoothing_length[i], neighbors, p.pos_x, p.pos_y, p.pos_z,
        p.smoothing_length, 2.0f,
        [&](int j, float, float, float, float) {
            const double dx = p.pos_x[j] - p.pos_x[i];
            const double dy = p.pos_y[j] - p.pos_y[i];
            const double dz = p.pos_z[j] - p.pos_z[i];
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double h = 0.5 * (static_cast<double>(p.smoothing_length[i]) +
                                    p.smoothing_length[j]);
            if (!(r2 > 0.0) || r2 >= 4.0*h*h) return;
            const double r = sqrt(r2);
            const double inverse = 1.0 / sqrt(fmax(r2, epsilon*epsilon));
            const double baseline = inverse * inverse * inverse;
            const double difference = gravity_softening_force_factor_double(r, h) - baseline;
            const double force = G * (outer ? 1.0 : static_cast<double>(p.mass[i])) * p.mass[j] * difference;
            fx += force*dx; fy += force*dy; fz += force*dz;
        });
    if (outer) {
        p.gravity_force_x[i] += fx; p.gravity_force_y[i] += fy; p.gravity_force_z[i] += fz;
    } else {
        p.force_x[i] += fx; p.force_y[i] += fy; p.force_z[i] += fz;
    }
}

void add_sph_gravity_softening_correction(
    SimulationState& state, const SimConfig& config, cudaStream_t stream, bool outer) {
    if (config.physics != SimulationPhysics::SPH) return;
    const int n = state.particles.num_local;
    if (!n) return;
    if (compute_sph_h_min(state, n, stream) < config.sph.hsml_min)
        throw std::runtime_error("SPH FMM requires imported smoothing lengths >= hsml_min");
    const double epsilon = sph_gravity_regularization_length(config, stream);
    // Subtracting large baseline forces can erase the softened force even in
    // FP64. Reject a poorly scaled baseline instead of silently losing it.
    if (!(epsilon > 0.0) || compute_sph_h_max(state, n, stream) > 128.0 * epsilon)
        throw std::runtime_error("SPH FMM compact correction requires h_max/epsilon <= 128; "
                                 "use physical h bounds and a shallower FMM depth");
    sph_gravity_softening_correction_kernel<<<(n+255)/256,256,0,stream>>>(
        state.particles, state.neighbor,
        config.acceleration.unified_fmm.gravitational_constant, epsilon, n, outer);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// GPU Constant Memory for Self-Gravity Configuration
// =============================================================================
// Use raw storage to avoid dynamic initialization (CUDA limitation)
// =============================================================================

__constant__ unsigned char dev_self_gravity_config_storage[sizeof(SelfGravityConfig)];
__constant__ float dev_gravitational_constant;

// Accessor macro
#define dev_self_gravity_config (*reinterpret_cast<SelfGravityConfig*>(const_cast<unsigned char*>(dev_self_gravity_config_storage)))

// =============================================================================
// Static Member Initialization
// =============================================================================

bool AccelerationManager::initialized = false;
SelfGravityConfig AccelerationManager::self_gravity_config;
DeviceStateArray<YorpManager> AccelerationManager::yorp_managers_ = {};
DeviceStateArray<bool> AccelerationManager::fmm_initialized_ = {};
static DeviceStateArray<double*> shear_direct_diagnostic_ = {};
ClumpStateSoA* AccelerationManager::clump_runtime_state_ = nullptr;
const ClumpSegmentOffsets* AccelerationManager::clump_runtime_segments_ = nullptr;

YorpManager& AccelerationManager::current_yorp_manager() {
    int gpu = 0;
    CUDA_CHECK(cudaGetDevice(&gpu));
    if (gpu < 0 || gpu >= yorp_managers_.size()) {
        throw std::runtime_error("current CUDA device exceeds YORP manager capacity");
    }
    return yorp_managers_[static_cast<size_t>(gpu)];
}

YorpManager* AccelerationManager::yorp_manager_for_gpu(int32_t gpu) {
    if (gpu < 0 || gpu >= yorp_managers_.size()) return nullptr;
    return &yorp_managers_[static_cast<size_t>(gpu)];
}

void AccelerationManager::set_clump_runtime(
    ClumpStateSoA* state, const ClumpSegmentOffsets* segments) {
    clump_runtime_state_ = state;
    clump_runtime_segments_ = segments;
}

// Gravity bodies are never materialized in the primary ParticleSoA.  The
// compact BFFMM shadow view contains [0,normal_count) live normal particles
// followed by one CoM body per clump.  Thus Karras can never split a rigid
// clump or accidentally include a member/dead particle.
__device__ __forceinline__ body_fitted_fmm::ClumpP3 load_gravity_body_p3(
    int body, int normal_begin, int normal_count, const ParticleSoA& p,
    const ClumpStateSoA& cs, body_fitted_fmm::ClumpVec3& pos, bool& clump) {
    body_fitted_fmm::ClumpP3 out;
    if (body < normal_count) {
        const int i = normal_begin + body;
        pos = {p.pos_x[i], p.pos_y[i], p.pos_z[i]}; out.c[0] = p.mass[i]; clump = false;
    } else {
        const int c = body - normal_count;
        body_fitted_fmm::ClumpP3 body_p3;
        for (int q = 0; q < body_fitted_fmm::CLUMP_P3_COEFFS; ++q)
            body_p3.c[q] = cs.multipole_p3[q][c];
        // Mass is canonical state, even when old/imported p3 metadata omitted it.
        body_p3.c[0] = cs.mass[c];
        out = body_fitted_fmm::clump_p3_rotate_body_to_world(
            body_p3, cs.quat_w[c], cs.quat_x[c], cs.quat_y[c], cs.quat_z[c]);
        pos = {cs.com_x[c], cs.com_y[c], cs.com_z[c]}; clump = true;
    }
    return out;
}

template<bool Outer>
__global__ void compute_clump_body_gravity_direct_kernel(
    ParticleSoA p, ClumpStateSoA cs, int normal_begin, int normal_count,
    int num_clumps, double G, double softening,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double box_length_x, double box_length_y, double box_length_z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nbody = normal_count + num_clumps;
    if (i >= nbody) return;
    body_fitted_fmm::ClumpVec3 xi{}; bool i_clump = false;
    const auto pi = load_gravity_body_p3(i, normal_begin, normal_count, p, cs, xi, i_clump);
    // Each unordered pair is evaluated exactly once.  The returned opposite
    // force is routed verbatim, so pair momentum cancellation does not depend
    // on two separately rounded target-side evaluations.
    for (int j = i + 1; j < nbody; ++j) {
        body_fitted_fmm::ClumpVec3 xj{}; bool j_clump = false;
        const auto pj = load_gravity_body_p3(j, normal_begin, normal_count, p, cs, xj, j_clump);
        body_fitted_fmm::ClumpVec3 separation{xj.x-xi.x, xj.y-xi.y, xj.z-xi.z};
        if (periodic_x && box_length_x > 0.0) {
            const double half = 0.5 * box_length_x;
            if (separation.x > half) separation.x -= box_length_x;
            else if (separation.x < -half) separation.x += box_length_x;
        }
        if (periodic_y && box_length_y > 0.0) {
            const double half = 0.5 * box_length_y;
            if (separation.y > half) separation.y -= box_length_y;
            else if (separation.y < -half) separation.y += box_length_y;
        }
        if (periodic_z && box_length_z > 0.0) {
            const double half = 0.5 * box_length_z;
            if (separation.z > half) separation.z -= box_length_z;
            else if (separation.z < -half) separation.z += box_length_z;
        }
        body_fitted_fmm::ClumpVec3 fij{}, fji{};
        body_fitted_fmm::clump_p3_pair_force(pi, pj, separation, G, softening, fij, fji);
        if (i_clump) {
            const int c=i-normal_count;
            double *fx=Outer?cs.gravity_force_x:cs.net_force_x,*fy=Outer?cs.gravity_force_y:cs.net_force_y,*fz=Outer?cs.gravity_force_z:cs.net_force_z;
            double *tx=Outer?cs.gravity_torque_x:cs.net_torque_x,*ty=Outer?cs.gravity_torque_y:cs.net_torque_y,*tz=Outer?cs.gravity_torque_z:cs.net_torque_z;
            atomicAdd(&fx[c],fij.x); atomicAdd(&fy[c],fij.y); atomicAdd(&fz[c],fij.z);
            const auto t=body_fitted_fmm::clump_p3_pair_torque(pi,pj,separation,G,softening,true);
            atomicAdd(&tx[c],t.x); atomicAdd(&ty[c],t.y); atomicAdd(&tz[c],t.z);
        }
        else { const int particle=normal_begin+i;
        if constexpr (Outer) {
            const double inv_m = p.mass[particle] > 0.0f ? 1.0 / static_cast<double>(p.mass[particle]) : 0.0;
            atomicAdd(&p.gravity_force_x[particle], fij.x * inv_m); atomicAdd(&p.gravity_force_y[particle], fij.y * inv_m); atomicAdd(&p.gravity_force_z[particle], fij.z * inv_m);
        } else {
            atomicAdd(&p.force_x[particle], fij.x); atomicAdd(&p.force_y[particle], fij.y); atomicAdd(&p.force_z[particle], fij.z); }}
        if (j_clump) {
            const int c=j-normal_count;
            double *fx=Outer?cs.gravity_force_x:cs.net_force_x,*fy=Outer?cs.gravity_force_y:cs.net_force_y,*fz=Outer?cs.gravity_force_z:cs.net_force_z;
            double *tx=Outer?cs.gravity_torque_x:cs.net_torque_x,*ty=Outer?cs.gravity_torque_y:cs.net_torque_y,*tz=Outer?cs.gravity_torque_z:cs.net_torque_z;
            atomicAdd(&fx[c],fji.x); atomicAdd(&fy[c],fji.y); atomicAdd(&fz[c],fji.z);
            const auto t=body_fitted_fmm::clump_p3_pair_torque(pi,pj,separation,G,softening,false);
            atomicAdd(&tx[c],t.x); atomicAdd(&ty[c],t.y); atomicAdd(&tz[c],t.z);
        }
        else { const int particle=normal_begin+j;
        if constexpr (Outer) { const double inv_m=p.mass[particle]>0.0f?1.0/static_cast<double>(p.mass[particle]):0.0; atomicAdd(&p.gravity_force_x[particle],fji.x*inv_m); atomicAdd(&p.gravity_force_y[particle],fji.y*inv_m); atomicAdd(&p.gravity_force_z[particle],fji.z*inv_m); }
        else { atomicAdd(&p.force_x[particle],fji.x); atomicAdd(&p.force_y[particle],fji.y); atomicAdd(&p.force_z[particle],fji.z); }}
    }
}

static void compute_registered_clump_direct_gravity(
    SimulationState& state, ClumpStateSoA& cs, const ClumpSegmentOffsets& seg,
    const SimConfig& config, bool outer, cudaStream_t stream) {
    const int normal_count = seg.normal_end - seg.normal_begin;
    const int nbody = normal_count + cs.num_clumps;
    if (nbody < 2) return;
    const double G = config.acceleration.unified_fmm.gravitational_constant;
    const double eps = config.particle.radius_min * config.acceleration.unified_fmm.softening_factor;
    const PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();
    const int block = 128, grid = (nbody + block - 1) / block;
    if (outer)
        compute_clump_body_gravity_direct_kernel<true><<<grid, block, 0, stream>>>(
            state.particles, cs, seg.normal_begin, normal_count, cs.num_clumps, G, eps,
            pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
            pbc.box_length_x, pbc.box_length_y, pbc.box_length_z);
    else
        compute_clump_body_gravity_direct_kernel<false><<<grid, block, 0, stream>>>(
            state.particles, cs, seg.normal_begin, normal_count, cs.num_clumps, G, eps,
            pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
            pbc.box_length_x, pbc.box_length_y, pbc.box_length_z);
    CUDA_CHECK(cudaGetLastError());
}

// Build the compact gravity-body view used by the clump-aware BFFMM path.
// Clump members and dead rows are never copied into this view.  Clump masses
// are zero in `fmm_mass`: their interactions are evaluated by the conservative
// p3 cross pass, while their physical mass remains available in view.mass for
// the mandatory global projection.
__global__ void build_clump_gravity_body_view_kernel(
    ParticleSoA p, ClumpStateSoA cs, int normal_begin, int normal_count,
    body_fitted_fmm::GravityBodyView view, float* fmm_mass) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= view.count) return;
    if (body < normal_count) {
        const int particle = normal_begin + body;
        view.pos_x[body] = p.pos_x[particle];
        view.pos_y[body] = p.pos_y[particle];
        view.pos_z[body] = p.pos_z[particle];
        view.mass[body] = static_cast<double>(p.mass[particle]);
        view.source_index[body] = particle;
        view.kind[body] = body_fitted_fmm::GravityBodyKind::NormalParticle;
        fmm_mass[body] = p.mass[particle];
    } else {
        const int clump = body - normal_count;
        view.pos_x[body] = cs.com_x[clump];
        view.pos_y[body] = cs.com_y[clump];
        view.pos_z[body] = cs.com_z[clump];
        view.mass[body] = cs.mass[clump];
        view.source_index[body] = clump;
        view.kind[body] = body_fitted_fmm::GravityBodyKind::ClumpCoM;
        fmm_mass[body] = 0.0f;
    }
}

__device__ __forceinline__ body_fitted_fmm::ClumpP3 load_view_body_p3(
    int body, const body_fitted_fmm::GravityBodyView& view,
    const ParticleSoA& p, const ClumpStateSoA& cs) {
    body_fitted_fmm::ClumpP3 out;
    if (view.kind[body] == body_fitted_fmm::GravityBodyKind::NormalParticle) {
        out.c[0] = static_cast<double>(p.mass[view.source_index[body]]);
    } else {
        const int clump = view.source_index[body];
        body_fitted_fmm::ClumpP3 body_p3;
        for (int q = 0; q < body_fitted_fmm::CLUMP_P3_COEFFS; ++q)
            body_p3.c[q] = cs.multipole_p3[q][clump];
        body_p3.c[0] = cs.mass[clump];
        out = body_fitted_fmm::clump_p3_rotate_body_to_world(
            body_p3, cs.quat_w[clump], cs.quat_x[clump],
            cs.quat_y[clump], cs.quat_z[clump]);
    }
    return out;
}

// Exact conservative cross pass for every body pair involving a clump.
// Normal-normal pairs remain in BFFMM.  Evaluating each unordered cross pair
// once gives bitwise-opposite pair forces before the global projection.
__global__ void compute_clump_bffmm_cross_kernel(
    ParticleSoA p, ClumpStateSoA cs,
    body_fitted_fmm::GravityBodyView view,
    double* body_fx, double* body_fy, double* body_fz,
    double* body_tx, double* body_ty, double* body_tz,
    double G, double softening,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double box_length_x, double box_length_y, double box_length_z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.count) return;
    const bool i_clump = view.kind[i] == body_fitted_fmm::GravityBodyKind::ClumpCoM;
    const auto pi = load_view_body_p3(i, view, p, cs);
    const body_fitted_fmm::ClumpVec3 xi{view.pos_x[i], view.pos_y[i], view.pos_z[i]};
    for (int j = i + 1; j < view.count; ++j) {
        const bool j_clump = view.kind[j] == body_fitted_fmm::GravityBodyKind::ClumpCoM;
        if (!i_clump && !j_clump) continue;
        const auto pj = load_view_body_p3(j, view, p, cs);
        body_fitted_fmm::ClumpVec3 r{
            view.pos_x[j] - xi.x, view.pos_y[j] - xi.y, view.pos_z[j] - xi.z};
        if (periodic_x && box_length_x > 0.0) {
            const double half = 0.5 * box_length_x;
            if (r.x > half) r.x -= box_length_x; else if (r.x < -half) r.x += box_length_x;
        }
        if (periodic_y && box_length_y > 0.0) {
            const double half = 0.5 * box_length_y;
            if (r.y > half) r.y -= box_length_y; else if (r.y < -half) r.y += box_length_y;
        }
        if (periodic_z && box_length_z > 0.0) {
            const double half = 0.5 * box_length_z;
            if (r.z > half) r.z -= box_length_z; else if (r.z < -half) r.z += box_length_z;
        }
        body_fitted_fmm::ClumpVec3 fi{}, fj{};
        body_fitted_fmm::clump_p3_pair_force(pi, pj, r, G, softening, fi, fj);
        atomicAdd(&body_fx[i], fi.x); atomicAdd(&body_fy[i], fi.y); atomicAdd(&body_fz[i], fi.z);
        atomicAdd(&body_fx[j], fj.x); atomicAdd(&body_fy[j], fj.y); atomicAdd(&body_fz[j], fj.z);
        if (i_clump) {
            const auto t = body_fitted_fmm::clump_p3_pair_torque(pi, pj, r, G, softening, true);
            atomicAdd(&body_tx[i], t.x); atomicAdd(&body_ty[i], t.y); atomicAdd(&body_tz[i], t.z);
        }
        if (j_clump) {
            const auto t = body_fitted_fmm::clump_p3_pair_torque(pi, pj, r, G, softening, false);
            atomicAdd(&body_tx[j], t.x); atomicAdd(&body_ty[j], t.y); atomicAdd(&body_tz[j], t.z);
        }
    }
}

constexpr int CLUMP_GP_SCALARS = 16;

__global__ void reduce_clump_body_projection_kernel(
    body_fitted_fmm::GravityBodyView view,
    const double* fx, const double* fy, const double* fz,
    const double* tx, const double* ty, const double* tz,
    double* scalars) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.count) return;
    const double x=view.pos_x[i], y=view.pos_y[i], z=view.pos_z[i], m=view.mass[i];
    const double bx=fx[i], by=fy[i], bz=fz[i];
    atomicAdd(&scalars[0],bx); atomicAdd(&scalars[1],by); atomicAdd(&scalars[2],bz);
    atomicAdd(&scalars[3],y*bz-z*by+tx[i]);
    atomicAdd(&scalars[4],z*bx-x*bz+ty[i]);
    atomicAdd(&scalars[5],x*by-y*bx+tz[i]);
    atomicAdd(&scalars[6],m); atomicAdd(&scalars[7],m*x);
    atomicAdd(&scalars[8],m*y); atomicAdd(&scalars[9],m*z);
    const double r2=x*x+y*y+z*z;
    atomicAdd(&scalars[10],m*(r2-x*x)); atomicAdd(&scalars[11],m*(r2-y*y));
    atomicAdd(&scalars[12],m*(r2-z*z)); atomicAdd(&scalars[13],-m*x*y);
    atomicAdd(&scalars[14],-m*x*z); atomicAdd(&scalars[15],-m*y*z);
}

__global__ void apply_clump_body_projection_kernel(
    body_fitted_fmm::GravityBodyView view,
    double* fx, double* fy, double* fz,
    double Fex, double Fey, double Fez,
    double ax, double ay, double az,
    double cmx, double cmy, double cmz, double inv_mass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.count) return;
    const double m=view.mass[i], rx=view.pos_x[i]-cmx, ry=view.pos_y[i]-cmy, rz=view.pos_z[i]-cmz;
    fx[i] -= m*inv_mass*Fex + m*(ay*rz-az*ry);
    fy[i] -= m*inv_mass*Fey + m*(az*rx-ax*rz);
    fz[i] -= m*inv_mass*Fez + m*(ax*ry-ay*rx);
}

template<bool Outer>
__global__ void scatter_clump_gravity_body_kernel(
    body_fitted_fmm::GravityBodyView view, ParticleSoA p, ClumpStateSoA cs,
    const double* fx, const double* fy, const double* fz,
    const double* tx, const double* ty, const double* tz) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= view.count) return;
    const int source = view.source_index[body];
    if (view.kind[body] == body_fitted_fmm::GravityBodyKind::NormalParticle) {
        if constexpr (Outer) {
            const double inv_mass=view.mass[body]>0.0?1.0/view.mass[body]:0.0;
            atomicAdd(&p.gravity_force_x[source],fx[body]*inv_mass);
            atomicAdd(&p.gravity_force_y[source],fy[body]*inv_mass);
            atomicAdd(&p.gravity_force_z[source],fz[body]*inv_mass);
        } else {
            atomicAdd(&p.force_x[source],fx[body]); atomicAdd(&p.force_y[source],fy[body]);
            atomicAdd(&p.force_z[source],fz[body]);
        }
    } else {
        double *out_fx=Outer?cs.gravity_force_x:cs.net_force_x,*out_fy=Outer?cs.gravity_force_y:cs.net_force_y,*out_fz=Outer?cs.gravity_force_z:cs.net_force_z;
        double *out_tx=Outer?cs.gravity_torque_x:cs.net_torque_x,*out_ty=Outer?cs.gravity_torque_y:cs.net_torque_y,*out_tz=Outer?cs.gravity_torque_z:cs.net_torque_z;
        atomicAdd(&out_fx[source],fx[body]); atomicAdd(&out_fy[source],fy[body]);
        atomicAdd(&out_fz[source],fz[body]); atomicAdd(&out_tx[source],tx[body]);
        atomicAdd(&out_ty[source],ty[body]); atomicAdd(&out_tz[source],tz[body]);
    }
}

// =============================================================================
// Self-Gravity Kernel: Direct N² Summation
// =============================================================================

// Each thread computes gravitational acceleration on one particle from all others
//
// Design:
// - Thread per particle (each thread sums contributions from all other particles)
// - Uses shared memory tiling for improved memory access patterns
// - Supports periodic boundaries via minimum image convention
// =============================================================================

// Simple N² kernel without shared memory tiling (suitable for N < 10000)
template<bool UsePeriodic>
__global__ void compute_self_gravity_simple_kernel(
    // Particle positions (double precision for accuracy)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle masses
    const float* __restrict__ mass,
    // SPH smoothing length (for SPHSOL-style gravity softening)
    const float* __restrict__ smoothing_length,
    // Output: forces (atomic add)
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    // Periodic boundary config
    float box_length_x, float box_length_y, float box_length_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    // Parameters
    double G,                    // Gravitational constant (double)
    bool query_particle_softening,
    int32_t num_local_particles,
    int32_t num_total_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local_particles) return;

    // Load particle i position (double)
    double xi = pos_x[i];
    double yi = pos_y[i];
    double zi = pos_z[i];
    double mi = static_cast<double>(mass[i]);
    float hi = smoothing_length[i];  // SPH smoothing length

    // Accumulate acceleration (double)
    double ax = 0.0, ay = 0.0, az = 0.0;

    // Loop over all other particles (including ghosts)
    for (int j = 0; j < num_total_particles; ++j) {
        if (j == i) continue;

        // Load particle j position
        double xj = pos_x[j];
        double yj = pos_y[j];
        double zj = pos_z[j];
        double mj = static_cast<double>(mass[j]);
        float hj = smoothing_length[j];

        // Distance vector (from i to j)
        double dx = xj - xi;
        double dy = yj - yi;
        double dz = zj - zi;

        // Apply minimum image convention for periodic boundaries
        if constexpr (UsePeriodic) {
            if (periodic_x && box_length_x > 0.0f) {
                double half = static_cast<double>(box_length_x) * 0.5;
                if (dx > half) dx -= static_cast<double>(box_length_x);
                else if (dx < -half) dx += static_cast<double>(box_length_x);
            }
            if (periodic_y && box_length_y > 0.0f) {
                double half = static_cast<double>(box_length_y) * 0.5;
                if (dy > half) dy -= static_cast<double>(box_length_y);
                else if (dy < -half) dy += static_cast<double>(box_length_y);
            }
            if (periodic_z && box_length_z > 0.0f) {
                double half = static_cast<double>(box_length_z) * 0.5;
                if (dz > half) dz -= static_cast<double>(box_length_z);
                else if (dz < -half) dz += static_cast<double>(box_length_z);
            }
        }

        // Cubic-spline gravity softening (matches SPHSOL KernelSoften).
        // For h_soft <= 0: falls back to exact 1/r^3 Newtonian (DEM-safe).
        // For r >= 2*h_soft: exact 1/r^3 Newtonian.
        // For r < 2*h_soft: smoothly softened (Price & Monaghan 2007).
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = sqrt(r2);
        double h_soft = select_sph_gravity_softening_length(
            static_cast<double>(hi), static_cast<double>(hj),
            query_particle_softening);
        double ff = gravity_softening_force_factor_double(r, h_soft);

        // Gravitational acceleration: a = G * m_j * ff * r_vec
        double factor = G * mj * ff;

        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }

    // Convert acceleration to force: F = m * a
    double fx = mi * ax;
    double fy = mi * ay;
    double fz = mi * az;

    // Atomic add to force arrays
    atomic_add_force(force_x, force_y, force_z, fx, fy, fz, i);
}

// =============================================================================
// Outer-step Gravity Kernel: Direct N² (RESPA variant)
// =============================================================================
// Same accumulation body as compute_self_gravity_simple_kernel, but:
// - Outputs gravitational ACCELERATION (m/s²) — no mass multiplication
// - Per-thread direct store (no atomic-add) — each thread owns grav_acc[i]
// - Writes to grav_acc[] instead of force[]
// =============================================================================

template<bool UsePeriodic>
__global__ void compute_outer_gravity_simple_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ mass,
    const float*  __restrict__ smoothing_length,
    double* __restrict__ grav_acc_x,
    double* __restrict__ grav_acc_y,
    double* __restrict__ grav_acc_z,
    float box_length_x, float box_length_y, float box_length_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double G,
    bool query_particle_softening,
    int32_t num_local_particles,
    int32_t num_total_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local_particles) return;

    double xi = pos_x[i];
    double yi = pos_y[i];
    double zi = pos_z[i];
    float hi = smoothing_length[i];

    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int j = 0; j < num_total_particles; ++j) {
        if (j == i) continue;

        double xj = pos_x[j];
        double yj = pos_y[j];
        double zj = pos_z[j];
        double mj = static_cast<double>(mass[j]);
        float hj = smoothing_length[j];

        double dx = xj - xi;
        double dy = yj - yi;
        double dz = zj - zi;

        if constexpr (UsePeriodic) {
            if (periodic_x && box_length_x > 0.0f) {
                double half = static_cast<double>(box_length_x) * 0.5;
                if (dx > half) dx -= static_cast<double>(box_length_x);
                else if (dx < -half) dx += static_cast<double>(box_length_x);
            }
            if (periodic_y && box_length_y > 0.0f) {
                double half = static_cast<double>(box_length_y) * 0.5;
                if (dy > half) dy -= static_cast<double>(box_length_y);
                else if (dy < -half) dy += static_cast<double>(box_length_y);
            }
            if (periodic_z && box_length_z > 0.0f) {
                double half = static_cast<double>(box_length_z) * 0.5;
                if (dz > half) dz -= static_cast<double>(box_length_z);
                else if (dz < -half) dz += static_cast<double>(box_length_z);
            }
        }

        // Cubic-spline gravity softening (matches SPHSOL KernelSoften).
        // For h_soft <= 0: falls back to exact 1/r^3 Newtonian (DEM-safe).
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = sqrt(r2);
        double h_soft = select_sph_gravity_softening_length(
            static_cast<double>(hi), static_cast<double>(hj),
            query_particle_softening);
        double ff = gravity_softening_force_factor_double(r, h_soft);

        double factor = G * mj * ff;

        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }

    grav_acc_x[i] = ax;
    grav_acc_y[i] = ay;
    grav_acc_z[i] = az;
}

// =============================================================================
// Tiled N² Kernel with Shared Memory (Optimized)
// =============================================================================
// Features:
// - Shared memory tiling with float4 (reduces bank conflicts)
// - Fast inverse square root (rsqrtf)
// - Aggressive loop unrolling (hides ALU latency)
// - Template-based periodic boundary logic (removes branching)
// =============================================================================

constexpr int TILE_SIZE = 128;

// Finite physical-image summation for a shearing box.  This deliberately does
// not apply an x/y minimum image: every configured lattice image is a distinct
// source.  The central source is slot zero, so only (slot=0,j=i) is self.
template<bool Outer>
__global__ void compute_shear_image_gravity_tiled_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float* __restrict__ mass,
    const float* __restrict__ smoothing_length,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    double* __restrict__ out_z,
    ShearGravityImages images,
    bool periodic_z,
    double length_z,
    double G,
    int32_t num_local_particles,
    int32_t num_total_particles,
    double* diagnostic)
{
    __shared__ double s_pos_x[TILE_SIZE];
    __shared__ double s_pos_y[TILE_SIZE];
    __shared__ double s_pos_z[TILE_SIZE];
    __shared__ double s_mass[TILE_SIZE];
    __shared__ float s_h[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < num_local_particles;
    double xi = 0.0, yi = 0.0, zi = 0.0, mi = 0.0;
    float hi = 0.0f;
    if (active) {
        xi = pos_x[i]; yi = pos_y[i]; zi = pos_z[i];
        mi = static_cast<double>(mass[i]);
        hi = smoothing_length[i];
    }

    double ax = 0.0, ay = 0.0, az = 0.0;
    const double cutoff_sq = images.cutoff_radius > 0.0
        ? images.cutoff_radius * images.cutoff_radius : 0.0;
    const int tiles = (num_total_particles + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < tiles; ++tile) {
        const int j = tile * TILE_SIZE + threadIdx.x;
        if (j < num_total_particles) {
            s_pos_x[threadIdx.x] = pos_x[j];
            s_pos_y[threadIdx.x] = pos_y[j];
            s_pos_z[threadIdx.x] = pos_z[j];
            s_mass[threadIdx.x] = static_cast<double>(mass[j]);
            s_h[threadIdx.x] = smoothing_length[j];
        } else {
            s_pos_x[threadIdx.x] = 0.0;
            s_pos_y[threadIdx.x] = 0.0;
            s_pos_z[threadIdx.x] = 0.0;
            s_mass[threadIdx.x] = 0.0;
            s_h[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (active) {
            for (int slot = 0; slot < images.image_count; ++slot) {
                int32_t nx = 0, ny = 0;
                shear_gravity_image_indices(
                    slot, images.layers_x, images.layers_y, nx, ny);
                const double shift_x = static_cast<double>(nx) * images.length_x;
                const double shift_y = static_cast<double>(ny) * images.length_y;
                #pragma unroll 128
                for (int k = 0; k < TILE_SIZE; ++k) {
                    const int source = tile * TILE_SIZE + k;
                    if (source >= num_total_particles) break;
                    if (slot == 0 && source == i) continue;
                    double dx = s_pos_x[k] + shift_x - xi;
                    double dy = shear_gravity_canonical_y(s_pos_y[k], nx, images) + shift_y - yi;
                    double dz = s_pos_z[k] - zi;
                    if (periodic_z && length_z > 0.0)
                        dz -= nearbyint(dz / length_z) * length_z;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (cutoff_sq > 0.0 && r2 > cutoff_sq) continue;
                    const double r = sqrt(r2);
                    const double hij = 0.5 *
                        (static_cast<double>(hi) + static_cast<double>(s_h[k]));
                    const double factor = s_mass[k] *
                        gravity_softening_force_factor_double(r, hij);
                    ax += factor * dx;
                    ay += factor * dy;
                    az += factor * dz;
                }
            }
        }
        __syncthreads();
    }

    if (!active) return;
    ax *= G; ay *= G; az *= G;
    const double fx = mi * ax, fy = mi * ay, fz = mi * az;
    if (diagnostic) {
        atomicAdd(diagnostic + 0, fx);
        atomicAdd(diagnostic + 1, fy);
        atomicAdd(diagnostic + 2, fz);
        atomicAdd(diagnostic + 3, sqrt(fx * fx + fy * fy + fz * fz));
    }
    if constexpr (Outer) {
        out_x[i] = ax; out_y[i] = ay; out_z[i] = az;
    } else {
        atomic_add_force(out_x, out_y, out_z, fx, fy, fz, i);
    }
}

__global__ void shear_direct_force_balance_warn_kernel(
    const double* diagnostic, double threshold)
{
    if (blockIdx.x || threadIdx.x) return;
    const double numerator = sqrt(diagnostic[0] * diagnostic[0] +
                                  diagnostic[1] * diagnostic[1] +
                                  diagnostic[2] * diagnostic[2]);
    const double epsilon = diagnostic[3] > 0.0
        ? numerator / diagnostic[3] : 0.0;
    if (epsilon > threshold)
        printf("WARNING: finite shear-image direct gravity force imbalance "
               "epsilon_F=%.9e exceeds %.9e (diagnostic only; force unchanged)\n",
               epsilon, threshold);
}

static double* prepare_shear_direct_diagnostic(double threshold,
                                               cudaStream_t stream)
{
    if (!(threshold > 0.0)) return nullptr;
    int gpu = 0;
    CUDA_CHECK(cudaGetDevice(&gpu));
    if (!shear_direct_diagnostic_[gpu])
        CUDA_CHECK(cudaMalloc(&shear_direct_diagnostic_[gpu], 4 * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(
        shear_direct_diagnostic_[gpu], 0, 4 * sizeof(double), stream));
    return shear_direct_diagnostic_[gpu];
}

static void finish_shear_direct_diagnostic(double* diagnostic,
                                           double threshold,
                                           cudaStream_t stream)
{
    if (!diagnostic) return;
    shear_direct_force_balance_warn_kernel<<<1, 1, 0, stream>>>(
        diagnostic, threshold);
    CUDA_CHECK(cudaGetLastError());
}

template<bool UsePeriodic>
__global__ void compute_self_gravity_tiled_kernel(
    // Particle positions (double precision)
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    // Particle masses
    const float* __restrict__ mass,
    // SPH smoothing length
    const float* __restrict__ smoothing_length,
    // Output: forces (atomic add)
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    // Periodic boundary config
    float box_length_x, float box_length_y, float box_length_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    // Parameters
    double G,
    bool query_particle_softening,
    int32_t num_local_particles,
    int32_t num_total_particles
) {
    // Shared memory for tile
    // Using simple arrays to avoid alignment issues/custom struct complexity in shared mem
    __shared__ double s_pos_x[TILE_SIZE];
    __shared__ double s_pos_y[TILE_SIZE];
    __shared__ double s_pos_z[TILE_SIZE];
    __shared__ double s_mass[TILE_SIZE];
    __shared__ float  s_sl[TILE_SIZE];     // smoothing length

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load particle i data (double)
    double ix = 0.0, iy = 0.0, iz = 0.0, im = 0.0;
    float  ih = 0.0f;
    bool valid_i = (i < num_local_particles);

    if (valid_i) {
        ix = pos_x[i];
        iy = pos_y[i];
        iz = pos_z[i];
        im = static_cast<double>(mass[i]);
        ih = smoothing_length[i];
    }

    // Accumulate acceleration
    double ax = 0.0, ay = 0.0, az = 0.0;

    // Calculate number of tiles (based on TOTAL particles)
    int num_tiles = (num_total_particles + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over all tiles
    for (int tile = 0; tile < num_tiles; ++tile) {
        // 1. Cooperative load into shared memory
        int j = tile * TILE_SIZE + threadIdx.x;

        if (j < num_total_particles) {
            s_pos_x[threadIdx.x] = pos_x[j];
            s_pos_y[threadIdx.x] = pos_y[j];
            s_pos_z[threadIdx.x] = pos_z[j];
            s_mass [threadIdx.x] = static_cast<double>(mass[j]);
            s_sl   [threadIdx.x] = smoothing_length[j];
        } else {
            // Padding
            s_pos_x[threadIdx.x] = 0.0;
            s_pos_y[threadIdx.x] = 0.0;
            s_pos_z[threadIdx.x] = 0.0;
            s_mass [threadIdx.x] = 0.0;
            s_sl   [threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 2. Compute interactions
        if (valid_i) {
            // Unroll loop for instruction level parallelism
            #pragma unroll 128
            for (int k = 0; k < TILE_SIZE; ++k) {
                // Fetch cached body j
                double jx = s_pos_x[k];
                double jy = s_pos_y[k];
                double jz = s_pos_z[k];
                double jm = s_mass[k];
                float  jh = s_sl[k];

                // Vector from i to j
                double dx = jx - ix;
                double dy = jy - iy;
                double dz = jz - iz;

                // Apply minimum image convention
                if constexpr (UsePeriodic) {
                    if (periodic_x && box_length_x > 0.0f) {
                        double half = static_cast<double>(box_length_x) * 0.5;
                        if (dx > half) dx -= static_cast<double>(box_length_x);
                        else if (dx < -half) dx += static_cast<double>(box_length_x);
                    }
                    if (periodic_y && box_length_y > 0.0f) {
                        double half = static_cast<double>(box_length_y) * 0.5;
                        if (dy > half) dy -= static_cast<double>(box_length_y);
                        else if (dy < -half) dy += static_cast<double>(box_length_y);
                    }
                    if (periodic_z && box_length_z > 0.0f) {
                        double half = static_cast<double>(box_length_z) * 0.5;
                        if (dz > half) dz -= static_cast<double>(box_length_z);
                        else if (dz < -half) dz += static_cast<double>(box_length_z);
                    }
                }

                // Cubic-spline gravity softening (matches SPHSOL KernelSoften).
                // For h_soft <= 0: falls back to exact 1/r^3 Newtonian (DEM-safe).
                double r2 = dx*dx + dy*dy + dz*dz;
                double r = sqrt(r2);
                double h_soft = select_sph_gravity_softening_length(
                    static_cast<double>(ih), static_cast<double>(jh),
                    query_particle_softening);
                double ff = gravity_softening_force_factor_double(r, h_soft);

                // Interaction strength: s = m_j * ff  (ff = phi'(r)/r)
                double s = jm * ff;

                // Accumulate acceleration: a += s * r_vec
                ax += dx * s;
                ay += dy * s;
                az += dz * s;
            }
        }

        __syncthreads();
    }

    // 3. Write output
    if (valid_i) {
        // Multiply by G
        ax *= G;
        ay *= G;
        az *= G;

        // Force = m_i * a
        atomic_add_force(force_x, force_y, force_z, im * ax, im * ay, im * az, i);
    }
}

// =============================================================================
// Outer-step Gravity Kernel: Tiled with Shared Memory (RESPA variant)
// =============================================================================

template<bool UsePeriodic>
__global__ void compute_outer_gravity_tiled_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ mass,
    const float*  __restrict__ smoothing_length,
    double* __restrict__ grav_acc_x,
    double* __restrict__ grav_acc_y,
    double* __restrict__ grav_acc_z,
    float box_length_x, float box_length_y, float box_length_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double G,
    bool query_particle_softening,
    int32_t num_local_particles,
    int32_t num_total_particles
) {
    __shared__ double s_pos_x[TILE_SIZE];
    __shared__ double s_pos_y[TILE_SIZE];
    __shared__ double s_pos_z[TILE_SIZE];
    __shared__ double s_mass[TILE_SIZE];
    __shared__ float  s_sl[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double ix = 0.0, iy = 0.0, iz = 0.0;
    float  ih = 0.0f;
    bool valid_i = (i < num_local_particles);

    if (valid_i) {
        ix = pos_x[i];
        iy = pos_y[i];
        iz = pos_z[i];
        ih = smoothing_length[i];
    }

    double ax = 0.0, ay = 0.0, az = 0.0;

    int num_tiles = (num_total_particles + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int j = tile * TILE_SIZE + threadIdx.x;

        if (j < num_total_particles) {
            s_pos_x[threadIdx.x] = pos_x[j];
            s_pos_y[threadIdx.x] = pos_y[j];
            s_pos_z[threadIdx.x] = pos_z[j];
            s_mass [threadIdx.x] = static_cast<double>(mass[j]);
            s_sl   [threadIdx.x] = smoothing_length[j];
        } else {
            s_pos_x[threadIdx.x] = 0.0;
            s_pos_y[threadIdx.x] = 0.0;
            s_pos_z[threadIdx.x] = 0.0;
            s_mass [threadIdx.x] = 0.0;
            s_sl   [threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (valid_i) {
            #pragma unroll 128
            for (int k = 0; k < TILE_SIZE; ++k) {
                double jx = s_pos_x[k];
                double jy = s_pos_y[k];
                double jz = s_pos_z[k];
                double jm = s_mass[k];
                float  jh = s_sl[k];

                double dx = jx - ix;
                double dy = jy - iy;
                double dz = jz - iz;

                if constexpr (UsePeriodic) {
                    if (periodic_x && box_length_x > 0.0f) {
                        double half = static_cast<double>(box_length_x) * 0.5;
                        if (dx > half) dx -= static_cast<double>(box_length_x);
                        else if (dx < -half) dx += static_cast<double>(box_length_x);
                    }
                    if (periodic_y && box_length_y > 0.0f) {
                        double half = static_cast<double>(box_length_y) * 0.5;
                        if (dy > half) dy -= static_cast<double>(box_length_y);
                        else if (dy < -half) dy += static_cast<double>(box_length_y);
                    }
                    if (periodic_z && box_length_z > 0.0f) {
                        double half = static_cast<double>(box_length_z) * 0.5;
                        if (dz > half) dz -= static_cast<double>(box_length_z);
                        else if (dz < -half) dz += static_cast<double>(box_length_z);
                    }
                }

                // Cubic-spline gravity softening (matches SPHSOL KernelSoften).
                // For h_soft <= 0: falls back to exact 1/r^3 Newtonian (DEM-safe).
                double r2 = dx*dx + dy*dy + dz*dz;
                double r = sqrt(r2);
                double h_soft = select_sph_gravity_softening_length(
                    static_cast<double>(ih), static_cast<double>(jh),
                    query_particle_softening);
                double ff = gravity_softening_force_factor_double(r, h_soft);

                double s = jm * ff;

                ax += dx * s;
                ay += dy * s;
                az += dz * s;
            }
        }

        __syncthreads();
    }

    if (valid_i) {
        ax *= G;
        ay *= G;
        az *= G;

        grav_acc_x[i] = ax;
        grav_acc_y[i] = ay;
        grav_acc_z[i] = az;
    }
}

// =============================================================================
// Self-Gravity Dispatcher
// =============================================================================

class SelfGravityDispatcher {
public:
    static void compute(
        SimulationState& state,
        const SimConfig& config,
        const PeriodicBoundaryConfig& pbc,
        double evaluation_time,
        cudaStream_t stream = 0
    ) {
        int32_t num_particles = state.particles.num_particles;
        int32_t num_local = state.particles.num_local;
        if (num_local == 0) return;

        int block_size = config.gpu.block_size;
        int num_blocks = (num_local + block_size - 1) / block_size;

        const double G = config.acceleration.direct_gravity.gravitational_constant;
        const bool query_particle_softening =
            config.acceleration.direct_gravity.sph_softening_mode ==
            DirectSphSofteningMode::QUERY_PARTICLE;

        const ShearGravityImages shear_images =
            make_shear_gravity_images(config, pbc, evaluation_time);
        if (shear_images.enabled) {
            double* diagnostic = prepare_shear_direct_diagnostic(
                config.shear_box.force_balance_warn_threshold, stream);
            const int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
            LAUNCH_KERNEL(compute_shear_image_gravity_tiled_kernel<false>,
                tiled_blocks, TILE_SIZE, 0, stream,
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass, state.particles.smoothing_length,
                state.particles.force_x, state.particles.force_y, state.particles.force_z,
                shear_images, pbc.periodic_z,
                static_cast<double>(pbc.box_length_z), G, num_local, num_particles,
                diagnostic);
            finish_shear_direct_diagnostic(diagnostic,
                config.shear_box.force_balance_warn_threshold, stream);
            return;
        }

        // Check if periodic boundaries are enabled
        bool use_periodic = pbc.is_enabled();

        // Select kernel based on periodic boundaries
        if (use_periodic) {
            // Use tiled kernel for better performance
            #ifdef CUDADEM_CPU_MOCK
            if (false) { // Force simple kernel in mock mode
            #else
            if (num_particles > 1000) {
            #endif
                int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
                LAUNCH_KERNEL(compute_self_gravity_tiled_kernel<true>, tiled_blocks, TILE_SIZE, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.force_x, state.particles.force_y, state.particles.force_z,
                    pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                    pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                    G, query_particle_softening, num_local, num_particles
                );
            } else {
                LAUNCH_KERNEL(compute_self_gravity_simple_kernel<true>, num_blocks, block_size, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.force_x, state.particles.force_y, state.particles.force_z,
                    pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                    pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                    G, query_particle_softening, num_local, num_particles
                );
            }
        } else {
            // Non-periodic case
            #ifdef CUDADEM_CPU_MOCK
            if (false) { // Force simple kernel in mock mode
            #else
            if (num_particles > 1000) {
            #endif
                int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
                LAUNCH_KERNEL(compute_self_gravity_tiled_kernel<false>, tiled_blocks, TILE_SIZE, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.force_x, state.particles.force_y, state.particles.force_z,
                    0.0f, 0.0f, 0.0f, false, false, false,
                    G, query_particle_softening, num_local, num_particles
                );
            } else {
                LAUNCH_KERNEL(compute_self_gravity_simple_kernel<false>, num_blocks, block_size, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.force_x, state.particles.force_y, state.particles.force_z,
                    0.0f, 0.0f, 0.0f, false, false, false,
                    G, query_particle_softening, num_local, num_particles
                );
            }
        }
    }
};

// =============================================================================
// Outer-step Gravity Dispatcher (RESPA variant)
// =============================================================================
// Dispatches outer-gravity kernels that write acceleration (not force) to
// gravity_force_x/y/z. Uses same kernel-selection logic as SelfGravityDispatcher.
// =============================================================================

class OuterGravityDispatcher {
public:
    static void compute_outer(
        SimulationState& state,
        const SimConfig& config,
        const PeriodicBoundaryConfig& pbc,
        double evaluation_time,
        cudaStream_t stream = 0
    ) {
        int32_t num_particles = state.particles.num_particles;
        int32_t num_local = state.particles.num_local;
        if (num_local == 0) return;

        int block_size = config.gpu.block_size;
        int num_blocks = (num_local + block_size - 1) / block_size;

        const double G = config.acceleration.direct_gravity.gravitational_constant;
        const bool query_particle_softening =
            config.acceleration.direct_gravity.sph_softening_mode ==
            DirectSphSofteningMode::QUERY_PARTICLE;

        const ShearGravityImages shear_images =
            make_shear_gravity_images(config, pbc, evaluation_time);
        if (shear_images.enabled) {
            double* diagnostic = prepare_shear_direct_diagnostic(
                config.shear_box.force_balance_warn_threshold, stream);
            const int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
            LAUNCH_KERNEL(compute_shear_image_gravity_tiled_kernel<true>,
                tiled_blocks, TILE_SIZE, 0, stream,
                state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                state.particles.mass, state.particles.smoothing_length,
                state.particles.gravity_force_x, state.particles.gravity_force_y,
                state.particles.gravity_force_z, shear_images, pbc.periodic_z,
                static_cast<double>(pbc.box_length_z), G, num_local, num_particles,
                diagnostic);
            finish_shear_direct_diagnostic(diagnostic,
                config.shear_box.force_balance_warn_threshold, stream);
            return;
        }

        bool use_periodic = pbc.is_enabled();

        if (use_periodic) {
            #ifdef CUDADEM_CPU_MOCK
            if (false) {
            #else
            if (num_particles > 1000) {
            #endif
                int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
                LAUNCH_KERNEL(compute_outer_gravity_tiled_kernel<true>, tiled_blocks, TILE_SIZE, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
                    pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                    pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                    G, query_particle_softening, num_local, num_particles
                );
            } else {
                LAUNCH_KERNEL(compute_outer_gravity_simple_kernel<true>, num_blocks, block_size, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
                    pbc.box_length_x, pbc.box_length_y, pbc.box_length_z,
                    pbc.periodic_x, pbc.periodic_y, pbc.periodic_z,
                    G, query_particle_softening, num_local, num_particles
                );
            }
        } else {
            #ifdef CUDADEM_CPU_MOCK
            if (false) {
            #else
            if (num_particles > 1000) {
            #endif
                int tiled_blocks = (num_local + TILE_SIZE - 1) / TILE_SIZE;
                LAUNCH_KERNEL(compute_outer_gravity_tiled_kernel<false>, tiled_blocks, TILE_SIZE, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
                    0.0f, 0.0f, 0.0f, false, false, false,
                    G, query_particle_softening, num_local, num_particles
                );
            } else {
                LAUNCH_KERNEL(compute_outer_gravity_simple_kernel<false>, num_blocks, block_size, 0, stream,
                    state.particles.pos_x, state.particles.pos_y, state.particles.pos_z,
                    state.particles.mass,
                    state.particles.smoothing_length,
                    state.particles.gravity_force_x, state.particles.gravity_force_y, state.particles.gravity_force_z,
                    0.0f, 0.0f, 0.0f, false, false, false,
                    G, query_particle_softening, num_local, num_particles
                );
            }
        }
    }
};

#include "body_fitted_fmm/body_fitted_fmm_manager.h"

namespace {

AdaptiveFMMDepthControllerState export_adaptive_fmm_depth_state(
    const body_fitted_fmm::AdaptiveFMMDepthControllerState& source) {
    AdaptiveFMMDepthControllerState output{};
    output.selected_depth = source.selected_depth;
    output.candidate_depth = source.candidate_depth;
    output.launch_depth = source.launch_depth;
    output.search_direction = source.search_direction;
    output.trial_phase = source.trial_phase;
    output.trial_warmups = source.trial_warmups;
    output.trial_sample_count = source.trial_sample_count;
    output.confirmation_count = source.confirmation_count;
    output.evaluations_since_trial = source.evaluations_since_trial;
    output.effective_max_depth = source.effective_max_depth;
    output.node_high_water = source.node_high_water;
    output.capacity_fallback_count = source.capacity_fallback_count;
    output.evaluation_counter = source.evaluation_counter;
    output.selected_time_ns = source.selected_time_ns;
    output.selected_sample_count = source.selected_sample_count;
    output.candidate_time_ns = source.candidate_time_ns;
    output.last_elapsed_ns = source.last_elapsed_ns;
    output.last_relative_improvement = source.last_relative_improvement;
    return output;
}

body_fitted_fmm::AdaptiveFMMDepthControllerState import_adaptive_fmm_depth_state(
    const AdaptiveFMMDepthControllerState& source) {
    body_fitted_fmm::AdaptiveFMMDepthControllerState output{};
    output.selected_depth = source.selected_depth;
    output.candidate_depth = source.candidate_depth;
    output.launch_depth = source.launch_depth;
    output.search_direction = source.search_direction;
    output.trial_phase = source.trial_phase;
    output.trial_warmups = source.trial_warmups;
    output.trial_sample_count = source.trial_sample_count;
    output.confirmation_count = source.confirmation_count;
    output.evaluations_since_trial = source.evaluations_since_trial;
    output.effective_max_depth = source.effective_max_depth;
    output.node_high_water = source.node_high_water;
    output.capacity_fallback_count = source.capacity_fallback_count;
    output.evaluation_counter = source.evaluation_counter;
    output.selected_time_ns = source.selected_time_ns;
    output.selected_sample_count = source.selected_sample_count;
    output.candidate_time_ns = source.candidate_time_ns;
    output.last_elapsed_ns = source.last_elapsed_ns;
    output.last_relative_improvement = source.last_relative_improvement;
    return output;
}

AdaptiveFMMDepthDiagnostic export_adaptive_fmm_depth_diagnostic(
    const body_fitted_fmm::AdaptiveFMMDepthDiagnostic& source) {
    AdaptiveFMMDepthDiagnostic output{};
    output.sequence = source.sequence;
    output.evaluation_counter = source.evaluation_counter;
    output.evaluation_time = source.evaluation_time;
    output.tree_dimension = source.tree_dimension;
    output.selected_depth = source.selected_depth;
    output.candidate_depth = source.candidate_depth;
    output.search_direction = source.search_direction;
    output.trial_phase = source.trial_phase;
    output.elapsed_ms = source.elapsed_ms;
    output.relative_improvement = source.relative_improvement;
    output.node_count = source.node_count;
    output.node_capacity = source.node_capacity;
    output.effective_max_depth = source.effective_max_depth;
    output.transition_reason = static_cast<AdaptiveFMMDepthTransition>(
        static_cast<int32_t>(source.transition_reason));
    return output;
}

struct ClumpGravityBodyBuffers {
    ::body_fitted_fmm::GravityBodyView view{};
    float* fmm_mass = nullptr;
    double *fx=nullptr,*fy=nullptr,*fz=nullptr,*tx=nullptr,*ty=nullptr,*tz=nullptr;
    double* projection_scalars = nullptr;

    explicit ClumpGravityBodyBuffers(int normal_count, int clump_count) {
        view.normal_count=normal_count; view.clump_count=clump_count;
        view.count=normal_count+clump_count;
        const size_t dbytes=static_cast<size_t>(view.count)*sizeof(double);
        CUDA_CHECK(cudaMalloc(&view.pos_x,dbytes)); CUDA_CHECK(cudaMalloc(&view.pos_y,dbytes));
        CUDA_CHECK(cudaMalloc(&view.pos_z,dbytes)); CUDA_CHECK(cudaMalloc(&view.mass,dbytes));
        CUDA_CHECK(cudaMalloc(&view.source_index,static_cast<size_t>(view.count)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&view.kind,static_cast<size_t>(view.count)*sizeof(::body_fitted_fmm::GravityBodyKind)));
        CUDA_CHECK(cudaMalloc(&fmm_mass,static_cast<size_t>(view.count)*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&fx,dbytes)); CUDA_CHECK(cudaMalloc(&fy,dbytes)); CUDA_CHECK(cudaMalloc(&fz,dbytes));
        CUDA_CHECK(cudaMalloc(&tx,dbytes)); CUDA_CHECK(cudaMalloc(&ty,dbytes)); CUDA_CHECK(cudaMalloc(&tz,dbytes));
        CUDA_CHECK(cudaMalloc(&projection_scalars,CLUMP_GP_SCALARS*sizeof(double)));
    }
    ~ClumpGravityBodyBuffers() {
        cudaFree(view.pos_x); cudaFree(view.pos_y); cudaFree(view.pos_z); cudaFree(view.mass);
        cudaFree(view.source_index); cudaFree(view.kind); cudaFree(fmm_mass);
        cudaFree(fx); cudaFree(fy); cudaFree(fz); cudaFree(tx); cudaFree(ty); cudaFree(tz);
        cudaFree(projection_scalars);
    }
    ClumpGravityBodyBuffers(const ClumpGravityBodyBuffers&)=delete;
    ClumpGravityBodyBuffers& operator=(const ClumpGravityBodyBuffers&)=delete;
};

bool invert_projection_tensor(const double A[9], double inverse[9]) {
    const double a=A[0],b=A[1],c=A[2],d=A[3],e=A[4],f=A[5],g=A[6],h=A[7],i=A[8];
    const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if (std::fabs(det)<1.e-40) return false;
    const double s=1.0/det;
    inverse[0]=(e*i-f*h)*s; inverse[1]=(c*h-b*i)*s; inverse[2]=(b*f-c*e)*s;
    inverse[3]=(f*g-d*i)*s; inverse[4]=(a*i-c*g)*s; inverse[5]=(c*d-a*f)*s;
    inverse[6]=(d*h-e*g)*s; inverse[7]=(b*g-a*h)*s; inverse[8]=(a*e-b*d)*s;
    return true;
}

void project_clump_gravity_bodies(ClumpGravityBodyBuffers& b, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(b.projection_scalars,0,CLUMP_GP_SCALARS*sizeof(double),stream));
    const int block=256,grid=(b.view.count+block-1)/block;
    reduce_clump_body_projection_kernel<<<grid,block,0,stream>>>(
        b.view,b.fx,b.fy,b.fz,b.tx,b.ty,b.tz,b.projection_scalars);
    CUDA_CHECK(cudaGetLastError());
    double h[CLUMP_GP_SCALARS];
    CUDA_CHECK(cudaMemcpyAsync(h,b.projection_scalars,sizeof(h),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double M=h[6]; if(M<1.e-30) return;
    const double invM=1.0/M,cmx=h[7]*invM,cmy=h[8]*invM,cmz=h[9]*invM;
    const double Fx=h[0],Fy=h[1],Fz=h[2];
    const double tx=h[3]-(cmy*Fz-cmz*Fy),ty=h[4]-(cmz*Fx-cmx*Fz),tz=h[5]-(cmx*Fy-cmy*Fx);
    const double cm2=cmx*cmx+cmy*cmy+cmz*cmz;
    const double I[9]={
        h[10]-M*(cm2-cmx*cmx),h[13]+M*cmx*cmy,h[14]+M*cmx*cmz,
        h[13]+M*cmx*cmy,h[11]-M*(cm2-cmy*cmy),h[15]+M*cmy*cmz,
        h[14]+M*cmx*cmz,h[15]+M*cmy*cmz,h[12]-M*(cm2-cmz*cmz)};
    double Ii[9],ax=0,ay=0,az=0;
    if(invert_projection_tensor(I,Ii)) {
        ax=Ii[0]*tx+Ii[1]*ty+Ii[2]*tz;
        ay=Ii[3]*tx+Ii[4]*ty+Ii[5]*tz;
        az=Ii[6]*tx+Ii[7]*ty+Ii[8]*tz;
    }
    apply_clump_body_projection_kernel<<<grid,block,0,stream>>>(
        b.view,b.fx,b.fy,b.fz,Fx,Fy,Fz,ax,ay,az,cmx,cmy,cmz,invM);
    CUDA_CHECK(cudaGetLastError());
}

void compute_registered_clump_bffmm_gravity(
    SimulationState& state, ClumpStateSoA& cs, const ClumpSegmentOffsets& seg,
    const SimConfig& config, const PeriodicBoundaryConfig& pbc, bool outer,
    double evaluation_time, cudaStream_t stream) {
    const int normal_count=seg.normal_end-seg.normal_begin;
    ClumpGravityBodyBuffers b(normal_count,cs.num_clumps);
    if(!::body_fitted_fmm::gravity_body_view_valid(b.view))
        throw std::runtime_error("invalid compact clump gravity body view");
    const int block=128,grid=(b.view.count+block-1)/block;
    build_clump_gravity_body_view_kernel<<<grid,block,0,stream>>>(
        state.particles,cs,seg.normal_begin,normal_count,b.view,b.fmm_mass);
    CUDA_CHECK(cudaGetLastError());
    const size_t bytes=static_cast<size_t>(b.view.count)*sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(b.fx,0,bytes,stream)); CUDA_CHECK(cudaMemsetAsync(b.fy,0,bytes,stream));
    CUDA_CHECK(cudaMemsetAsync(b.fz,0,bytes,stream)); CUDA_CHECK(cudaMemsetAsync(b.tx,0,bytes,stream));
    CUDA_CHECK(cudaMemsetAsync(b.ty,0,bytes,stream)); CUDA_CHECK(cudaMemsetAsync(b.tz,0,bytes,stream));

    SimulationState body_state;
    body_state.current_step=state.current_step; body_state.current_time=state.current_time; body_state.dt=state.dt;
    body_state.particles.num_particles=b.view.count; body_state.particles.num_local=b.view.count;
    body_state.particles.capacity=b.view.count;
    body_state.particles.pos_x=b.view.pos_x; body_state.particles.pos_y=b.view.pos_y; body_state.particles.pos_z=b.view.pos_z;
    body_state.particles.mass=b.fmm_mass;
    body_state.particles.force_x=b.fx; body_state.particles.force_y=b.fy; body_state.particles.force_z=b.fz;

    // Normal-normal gravity follows the configured BFFMM hierarchy. Clump
    // virtual bodies have zero FMM source/target mass, so no point-monopole
    // interaction can duplicate the p3 cross pass below.
    SimConfig fmm_config=config;
    fmm_config.acceleration.unified_fmm.enable_global_projection=false;
    body_fitted_fmm::UnifiedFMMDispatcher::compute(
        body_state,fmm_config,body_state.neighbor,pbc,evaluation_time,stream,true);

    const double G=config.acceleration.unified_fmm.gravitational_constant;
    const double eps=config.particle.radius_min*config.acceleration.unified_fmm.softening_factor;
    compute_clump_bffmm_cross_kernel<<<grid,block,0,stream>>>(
        state.particles,cs,b.view,b.fx,b.fy,b.fz,b.tx,b.ty,b.tz,G,eps,
        pbc.periodic_x,pbc.periodic_y,pbc.periodic_z,
        pbc.box_length_x,pbc.box_length_y,pbc.box_length_z);
    CUDA_CHECK(cudaGetLastError());
    project_clump_gravity_bodies(b,stream);
    if(outer)
        scatter_clump_gravity_body_kernel<true><<<grid,block,0,stream>>>(
            b.view,state.particles,cs,b.fx,b.fy,b.fz,b.tx,b.ty,b.tz);
    else
        scatter_clump_gravity_body_kernel<false><<<grid,block,0,stream>>>(
            b.view,state.particles,cs,b.fx,b.fy,b.fz,b.tx,b.ty,b.tz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace

// =============================================================================
// AccelerationManager Implementation
// =============================================================================

void AccelerationManager::initialize(const SimConfig& config) {
    if (initialized) {
        std::cerr << "WARNING: AccelerationManager already initialized, reinitializing..." << std::endl;
        shutdown();
    }

    // Extract self-gravity config
    self_gravity_config.enabled = config.acceleration.enable_self_gravity;
    self_gravity_config.softening = config.particle.radius_min * DEFAULT_SOFTENING_FACTOR;
    self_gravity_config.respect_periodic = true;
    // gravity_method now lives in AccelerationConfig; SelfGravityConfig only holds
    // softening and toggle flags for the direct N² kernel.
    // Copy to constant memory
    cudaMemcpyToSymbol(dev_self_gravity_config_storage, &self_gravity_config, sizeof(SelfGravityConfig));

    float G = static_cast<float>(
        config.acceleration.direct_gravity.gravitational_constant);
    cudaMemcpyToSymbol(dev_gravitational_constant, &G, sizeof(float));

    // Initialization is host-only; device allocations remain lazy on the
    // owning GPU thread.  Give every device-local manager identical cadence
    // and controller configuration.
    for (YorpManager& manager : yorp_managers_) manager.initialize(config);

    initialized = true;

    // Print configuration
    std::cout << "AccelerationManager initialized:" << std::endl;
    std::cout << "  Self-gravity: " << (self_gravity_config.enabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Softening: " << self_gravity_config.softening << " m" << std::endl;
    std::cout << "  Inertial: " << (config.acceleration.enable_inertial ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Tidal: " << (config.acceleration.enable_tidal ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  YORP: " << (config.acceleration.yorp.enabled ? "Enabled" : "Disabled") << std::endl;
}

void AccelerationManager::compute_all(
    SimulationState& state,
    const SimConfig& config,
    double current_time,
    cudaStream_t stream
) {
    if (!initialized) {
        std::cerr << "ERROR: AccelerationManager not initialized" << std::endl;
        return;
    }

    // Self-gravity
    if (config.acceleration.enable_self_gravity) {
        compute_self_gravity(state, config, current_time, stream);
    }

    // Inertial acceleration (to be implemented in inertial_acceleration.cu)
    if (config.acceleration.enable_inertial) {
        compute_inertial(state, config, current_time, stream);
    }

    // Tidal acceleration (to be implemented in tidal_acceleration.cu)
    if (config.acceleration.enable_tidal) {
        compute_tidal(state, config, current_time, stream);
    }

    // Pre-calculated acceleration fields
    for (int32_t i = 0; i < config.acceleration.num_accel_fields; ++i) {
        if (config.acceleration.fields[i].enabled) {
            compute_field_acceleration(state, config, current_time, i, stream);
        }
    }

    // YORP effect (incremental rotation updates)
    // Template policy: only called when enabled in config
    if (config.acceleration.yorp.enabled) {
        current_yorp_manager().update(
            state.particles, current_time, state.dt, config,
            config.gpu.block_size, stream);
    }
}

void AccelerationManager::compute_all_without_gravity(
    SimulationState& state,
    const SimConfig& config,
    double current_time,
    cudaStream_t stream,
    bool include_incremental_effects
) {
    if (!initialized) {
        std::cerr << "ERROR: AccelerationManager not initialized" << std::endl;
        return;
    }

    // Self-gravity is SKIPPED — handled by outer step in RESPA

    // Inertial acceleration
    if (config.acceleration.enable_inertial) {
        compute_inertial(state, config, current_time, stream);
    }

    // Tidal acceleration
    if (config.acceleration.enable_tidal) {
        compute_tidal(state, config, current_time, stream);
    }

    // Pre-calculated acceleration fields
    for (int32_t i = 0; i < config.acceleration.num_accel_fields; ++i) {
        if (config.acceleration.fields[i].enabled) {
            compute_field_acceleration(state, config, current_time, i, stream);
        }
    }

    // YORP effect
    if (include_incremental_effects && config.acceleration.yorp.enabled) {
        current_yorp_manager().update(
            state.particles, current_time, state.dt, config,
            config.gpu.block_size, stream);
    }
}

void AccelerationManager::observe_adaptive_yorp(
    SimulationState& state, const SimConfig& config, double current_time,
    cudaStream_t stream
) {
    if (!initialized || !config.acceleration.yorp.adaptive) return;
    current_yorp_manager().observe_adaptive(
        state.particles, current_time, config, config.gpu.block_size, stream);
}

bool AccelerationManager::consume_adaptive_yorp_diagnostic(
    AdaptiveYorpDiagnostic* diagnostic, cudaStream_t stream
) { return current_yorp_manager().consume_adaptive_diagnostic(diagnostic, stream); }

bool AccelerationManager::adaptive_yorp_should_terminate(cudaStream_t stream) {
    return current_yorp_manager().adaptive_should_terminate(stream);
}

bool AccelerationManager::get_adaptive_yorp_checkpoint(
    AdaptiveYorpCheckpoint* checkpoint, cudaStream_t stream
) { return current_yorp_manager().get_adaptive_checkpoint(checkpoint, stream); }

bool AccelerationManager::restore_adaptive_yorp_checkpoint(
    const AdaptiveYorpCheckpoint& checkpoint, cudaStream_t stream
) { return current_yorp_manager().restore_adaptive_checkpoint(checkpoint, stream); }

bool AccelerationManager::get_adaptive_fmm_depth_checkpoint(
    AdaptiveFMMDepthCheckpoint* checkpoint, cudaStream_t stream
) {
    if (!checkpoint) return false;
    body_fitted_fmm::AdaptiveFMMDepthCheckpoint runtime_checkpoint{};
    if (!body_fitted_fmm::UnifiedFMMDispatcher::
            get_adaptive_fmm_depth_checkpoint(&runtime_checkpoint, stream)) {
        return false;
    }
    checkpoint->version = 2;
    checkpoint->tree_dimension = runtime_checkpoint.tree_dimension;
    checkpoint->controller = export_adaptive_fmm_depth_state(
        runtime_checkpoint.controller);
    checkpoint->producer_sequence = runtime_checkpoint.diagnostic_sequence;
    checkpoint->persisted_csv_sequence = runtime_checkpoint.persisted_csv_sequence;
    return true;
}

bool AccelerationManager::restore_adaptive_fmm_depth_checkpoint(
    const AdaptiveFMMDepthCheckpoint& checkpoint, cudaStream_t stream
) {
    body_fitted_fmm::AdaptiveFMMDepthCheckpoint runtime_checkpoint{};
    if (checkpoint.version != 2 ||
        (checkpoint.tree_dimension != 2 && checkpoint.tree_dimension != 3)) return false;
    runtime_checkpoint.version = 2;
    runtime_checkpoint.tree_dimension = checkpoint.tree_dimension;
    runtime_checkpoint.controller =
        import_adaptive_fmm_depth_state(checkpoint.controller);
    runtime_checkpoint.diagnostic_sequence = checkpoint.producer_sequence;
    runtime_checkpoint.persisted_csv_sequence = checkpoint.persisted_csv_sequence;
    return body_fitted_fmm::UnifiedFMMDispatcher::
        restore_adaptive_fmm_depth_checkpoint(runtime_checkpoint, stream);
}

void AccelerationManager::consume_adaptive_fmm_depth_diagnostics(
    std::vector<AdaptiveFMMDepthDiagnostic>& diagnostics,
    uint64_t* dropped, cudaStream_t stream
) {
    std::vector<body_fitted_fmm::AdaptiveFMMDepthDiagnostic> runtime_diagnostics;
    body_fitted_fmm::UnifiedFMMDispatcher::
        consume_adaptive_fmm_depth_diagnostics(
            runtime_diagnostics, dropped, stream);
    diagnostics.reserve(diagnostics.size() + runtime_diagnostics.size());
    for (const auto& diagnostic : runtime_diagnostics) {
        diagnostics.push_back(export_adaptive_fmm_depth_diagnostic(diagnostic));
    }
}

bool AccelerationManager::service_adaptive_fmm_capacity(
    SimulationState& state, const SimConfig& config, cudaStream_t stream)
{
    return body_fitted_fmm::UnifiedFMMDispatcher::
        service_adaptive_fmm_capacity(state, config, stream);
}

void AccelerationManager::throw_if_fmm_terminal_error() {
    body_fitted_fmm::UnifiedFMMDispatcher::throw_if_terminal_error();
}

void AccelerationManager::compute_self_gravity(
    SimulationState& state,
    const SimConfig& config,
    double evaluation_time,
    cudaStream_t stream
) {
    if (!self_gravity_config.enabled) return;

    // Extract periodic boundary config from WallManager
    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    if (clump_runtime_state_ && clump_runtime_segments_ &&
        clump_runtime_state_->num_clumps > 0) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument(
                "adaptive unified_fmm tree depth is not supported for clump gravity");
        if (config.acceleration.gravity_method == SelfGravityMethod::DIRECT) {
            compute_registered_clump_direct_gravity(
                state,*clump_runtime_state_,*clump_runtime_segments_,config,false,stream);
        } else {
            const int body_count=(clump_runtime_segments_->normal_end-clump_runtime_segments_->normal_begin)
                               +clump_runtime_state_->num_clumps;
            int g=0; cudaGetDevice(&g);
            if(!fmm_initialized_[g]) {
                body_fitted_fmm::UnifiedFMMDispatcher::initialize(config,body_count);
                fmm_initialized_[g]=true;
            }
            compute_registered_clump_bffmm_gravity(
                state,*clump_runtime_state_,*clump_runtime_segments_,config,pbc,false,evaluation_time,stream);
        }
        return;
    }

    switch (config.acceleration.gravity_method) {
        case SelfGravityMethod::BFFMM: {
            int g = 0; cudaGetDevice(&g);
            if (!fmm_initialized_[g]) {
                if (!config.acceleration.unified_fmm.adaptive_depth.enabled)
                    body_fitted_fmm::UnifiedFMMDispatcher::initialize(
                        config, state.particles.num_particles);
                fmm_initialized_[g] = true;
            }
            body_fitted_fmm::UnifiedFMMDispatcher::compute(
                state, config, state.neighbor, pbc, evaluation_time, stream,
                /*synchronize_at_end=*/false);
            return;
        }
        case SelfGravityMethod::DIRECT:
            SelfGravityDispatcher::compute(state, config, pbc, evaluation_time, stream);
            return;
    }
}

void AccelerationManager::compute_outer_gravity(
    SimulationState& state,
    const SimConfig& config,
    double evaluation_time,
    cudaStream_t stream
) {
    if (!self_gravity_config.enabled) return;

    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    if (clump_runtime_state_ && clump_runtime_segments_ &&
        clump_runtime_state_->num_clumps > 0) {
        if (config.acceleration.unified_fmm.adaptive_depth.enabled)
            throw std::invalid_argument(
                "adaptive unified_fmm tree depth is not supported for clump gravity");
        const int normal_begin=clump_runtime_segments_->normal_begin;
        const int normal_count=clump_runtime_segments_->normal_end-normal_begin;
        const size_t normal_bytes=static_cast<size_t>(normal_count)*sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_x+normal_begin,0,normal_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_y+normal_begin,0,normal_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(state.particles.gravity_force_z+normal_begin,0,normal_bytes,stream));
        const size_t clump_bytes=static_cast<size_t>(clump_runtime_state_->num_clumps)*sizeof(double);
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_force_x,0,clump_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_force_y,0,clump_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_force_z,0,clump_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_torque_x,0,clump_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_torque_y,0,clump_bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(clump_runtime_state_->gravity_torque_z,0,clump_bytes,stream));
        if(config.acceleration.gravity_method==SelfGravityMethod::BFFMM) {
            const int body_count=normal_count+clump_runtime_state_->num_clumps;
            int g=0;cudaGetDevice(&g);
            if(!fmm_initialized_[g]) {
                body_fitted_fmm::UnifiedFMMDispatcher::initialize(config,body_count);
                fmm_initialized_[g]=true;
            }
            compute_registered_clump_bffmm_gravity(
                state,*clump_runtime_state_,*clump_runtime_segments_,config,pbc,true,evaluation_time,stream);
        } else {
            compute_registered_clump_direct_gravity(
                state,*clump_runtime_state_,*clump_runtime_segments_,config,true,stream);
        }
        return;
    }

    switch (config.acceleration.gravity_method) {
        case SelfGravityMethod::BFFMM: {
            int g = 0; cudaGetDevice(&g);
            if (!fmm_initialized_[g]) {
                if (!config.acceleration.unified_fmm.adaptive_depth.enabled)
                    body_fitted_fmm::UnifiedFMMDispatcher::initialize(
                        config, state.particles.num_particles);
                fmm_initialized_[g] = true;
            }
            body_fitted_fmm::UnifiedFMMDispatcher::compute_outer(state, config, state.neighbor, pbc, evaluation_time, stream);
            return;
        }
        case SelfGravityMethod::DIRECT:
            OuterGravityDispatcher::compute_outer(state, config, pbc, evaluation_time, stream);
            return;
    }
}

#ifdef CUDADEM_HAVE_LET

void AccelerationManager::compute_self_gravity_multigpu(
    SimulationState& state,
    const SimConfig& config,
    double evaluation_time,
    uint64_t gravity_evaluation_epoch,
    class LETCommunicator* let_comm,
    cudaStream_t stream
) {
    if (!self_gravity_config.enabled || !let_comm) return;

    PeriodicBoundaryConfig pbc = WallManager::get_pbc_config();

    switch (config.acceleration.gravity_method) {
        case SelfGravityMethod::BFFMM: {
            int g = 0;
            cudaGetDevice(&g);
            if (!fmm_initialized_[g]) {
                // The dispatcher owns initialization for both fixed and
                // adaptive depth. A force-only bootstrap may already have
                // initialized it without visiting this wrapper. Reinitializing
                // here would destroy that live distributed partition/state.
                fmm_initialized_[g] = true;
            }
            body_fitted_fmm::UnifiedFMMDispatcher::compute_multigpu(
                state, config, state.neighbor, pbc, *let_comm,
                gravity_evaluation_epoch, evaluation_time, stream);
            return;
        }
        case SelfGravityMethod::DIRECT:
            SelfGravityDispatcher::compute(
                state, config, pbc, state.current_time, stream);
            return;
    }
}

void AccelerationManager::compute_all_multigpu(
    SimulationState& state,
    const SimConfig& config,
    double current_time,
    uint64_t gravity_evaluation_epoch,
    class LETCommunicator* let_comm,
    cudaStream_t stream
) {
    if (!initialized) { std::cerr << "ERROR: AccelerationManager not initialized" << std::endl; return; }
    if (config.acceleration.enable_self_gravity)
        compute_self_gravity_multigpu(
            state, config, current_time, gravity_evaluation_epoch,
            let_comm, stream);
    if (config.acceleration.enable_inertial)
        compute_inertial(state, config, current_time, stream);
    if (config.acceleration.enable_tidal)
        compute_tidal(state, config, current_time, stream);
    for (int32_t i = 0; i < config.acceleration.num_accel_fields; ++i)
        if (config.acceleration.fields[i].enabled)
            compute_field_acceleration(state, config, current_time, i, stream);
    if (config.acceleration.yorp.enabled &&
        !config.acceleration.yorp.adaptive) {
        current_yorp_manager().update_owned(
            state.particles, state.particles.num_local,
            current_time, state.dt, config, config.gpu.block_size, stream);
    }
}

#endif // CUDADEM_HAVE_LET

// NOTE: compute_inertial() implemented in inertial_acceleration.cu
// NOTE: compute_tidal() implemented in tidal_acceleration.cu
// NOTE: compute_field_acceleration() implemented in acceleration_field_texture.cu

void AccelerationManager::shutdown() {
    body_fitted_fmm::UnifiedFMMDispatcher::shutdown_all();
    int saved_gpu = 0;
    cudaGetDevice(&saved_gpu);
    for (int gpu = 0; gpu < shear_direct_diagnostic_.size(); ++gpu) {
        if (!shear_direct_diagnostic_[gpu]) continue;
        cudaSetDevice(gpu);
        cudaFree(shear_direct_diagnostic_[gpu]);
        shear_direct_diagnostic_[gpu] = nullptr;
    }
    cudaSetDevice(saved_gpu);
    fmm_initialized_.fill(false);
    // Each manager remembers its owning device.  Unbound slots require no
    // CUDA context, which also keeps shutdown valid on hosts with < 8 GPUs.
    for (YorpManager& manager : yorp_managers_) manager.shutdown();
    cudaSetDevice(saved_gpu);
    clump_runtime_state_ = nullptr;
    clump_runtime_segments_ = nullptr;
    
    self_gravity_config = SelfGravityConfig();
    initialized = false;
}

// =============================================================================
// Verification Functions (Host-only)
// =============================================================================

#ifndef __CUDACC__

// Compute total gravitational potential energy (host-side verification)
inline double compute_total_gravitational_potential(
    const SimulationState& state,
    const SimConfig& config
) {
    int32_t N = state.particles.num_particles;
    double total_potential = 0.0;

    // Copy particle data to host
    std::vector<double> h_pos_x(N), h_pos_y(N), h_pos_z(N);
    std::vector<float> h_mass(N);
    std::vector<float> h_smoothing_length(N);

    cudaMemcpy(h_pos_x.data(), state.particles.pos_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_y.data(), state.particles.pos_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_z.data(), state.particles.pos_z, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass.data(), state.particles.mass, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smoothing_length.data(), state.particles.smoothing_length, N * sizeof(float), cudaMemcpyDeviceToHost);

    // O(N²) summation on host, using cubic-spline softening potential
    // (consistent with the device kernel's force softening)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = h_pos_x[j] - h_pos_x[i];
            double dy = h_pos_y[j] - h_pos_y[i];
            double dz = h_pos_z[j] - h_pos_z[i];

            double r2 = dx * dx + dy * dy + dz * dz;
            double r = sqrt(r2);
            double h_ij = 0.5 * (static_cast<double>(h_smoothing_length[i]) +
                                 static_cast<double>(h_smoothing_length[j]));
            double phi = gravity_softening_potential(static_cast<float>(r),
                                                     static_cast<float>(h_ij));

            total_potential += GRAVITATIONAL_CONSTANT * h_mass[i] * h_mass[j] * phi;
        }
    }

    return total_potential;
}

// Print self-gravity statistics
inline void print_self_gravity_stats(
    const SimulationState& state,
    const SimConfig& config
) {
    int32_t N = state.particles.num_particles;
    double N_sq = static_cast<double>(N) * static_cast<double>(N);

    std::cout << "=== Self-Gravity Statistics ===" << std::endl;
    std::cout << "Particles: " << N << std::endl;
    std::cout << "Pairs: " << N * (N - 1) / 2 << std::endl;
    std::cout << "Interactions/step: " << N_sq << std::endl;

    // Estimate computation time (very rough)
    // Assuming ~100 GFLOPS on modern GPU, ~10 FLOPs per interaction
    double flops_per_step = N_sq * 10.0;
    double estimated_time_ms = flops_per_step / (100.0e9) * 1000.0;
    std::cout << "Estimated time/step: " << estimated_time_ms << " ms" << std::endl;
    std::cout << "===============================" << std::endl;
}

#endif // __CUDACC__

} // namespace cudadem

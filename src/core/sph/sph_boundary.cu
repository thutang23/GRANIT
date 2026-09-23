#include "sph_boundary.h"
#include "sph_boundary_math.cuh"
#include "cuda_utils.h"
#include <stdexcept>
#include <vector>

namespace cudadem {

struct SphFixedState {
    bool fixed;
    double x, y, z, rho, energy, cold;
    float h, xx, xy, xz, yy, yz, zz, damage, alpha;
};

__global__ void capture_sph_fixed_kernel(ParticleSoA p, SphFixedState* saved, SPHConfig config) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= p.num_particles) return;
    bool fixed = false;
    for (int j=0; j<config.fixed_part_count; ++j) fixed |= p.clump_id[i] == config.fixed_parts[j];
    saved[i] = {fixed, p.pos_x[i], p.pos_y[i], p.pos_z[i], p.density[i], p.energy[i], p.energy_cold[i],
        p.smoothing_length[i], p.S_xx[i], p.S_xy[i], p.S_xz[i], p.S_yy[i], p.S_yz[i], p.S_zz[i],
        p.damage[i], p.distension[i]};
}

__global__ void restore_sph_fixed_kernel(ParticleSoA p, const SphFixedState* saved) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= p.num_particles || !saved[i].fixed) return;
    const auto& s = saved[i];
    p.pos_x[i]=s.x; p.pos_y[i]=s.y; p.pos_z[i]=s.z;
    p.vel_x[i]=p.vel_y[i]=p.vel_z[i]=0.0;
    p.density[i]=s.rho; p.energy[i]=s.energy; p.energy_cold[i]=s.cold;
    p.smoothing_length[i]=s.h; p.damage[i]=s.damage; p.distension[i]=s.alpha;
    p.S_xx[i]=s.xx; p.S_xy[i]=s.xy; p.S_xz[i]=s.xz;
    p.S_yy[i]=s.yy; p.S_yz[i]=s.yz; p.S_zz[i]=s.zz;
    p.rho_rate[i]=p.dedt[i]=0.0;
    // Force is retained as the support reaction, not applied to this fixed row.
}

__global__ void reflect_sph_domain_kernel(ParticleSoA p, SPHConfig config) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= p.num_particles) return;
    const double nx=config.symmetry_normal[0], ny=config.symmetry_normal[1], nz=config.symmetry_normal[2];
    if (nx*p.pos_x[i]+ny*p.pos_y[i]+nz*p.pos_z[i]+config.symmetry_offset >= 0.0) return;
    sph_reflect_vector(p.pos_x[i],p.pos_y[i],p.pos_z[i],nx,ny,nz,config.symmetry_offset);
    sph_reflect_vector(p.vel_x[i],p.vel_y[i],p.vel_z[i],nx,ny,nz);
    sph_reflect_stress(p.S_xx[i],p.S_xy[i],p.S_xz[i],p.S_yy[i],p.S_yz[i],p.S_zz[i],nx,ny,nz);
}

__global__ void update_sph_mirrors_kernel(ParticleSoA p, SPHConfig config) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= p.num_particles) return;
    const int j = i+p.num_particles;
#define COPY_MIRROR(field) p.field[j] = p.field[i]
    COPY_MIRROR(pos_x); COPY_MIRROR(pos_y); COPY_MIRROR(pos_z);
    COPY_MIRROR(vel_x); COPY_MIRROR(vel_y); COPY_MIRROR(vel_z);
    COPY_MIRROR(mass); COPY_MIRROR(density); COPY_MIRROR(pressure); COPY_MIRROR(c_sound);
    COPY_MIRROR(smoothing_length); COPY_MIRROR(support_extent); COPY_MIRROR(energy);
    COPY_MIRROR(S_xx); COPY_MIRROR(S_xy); COPY_MIRROR(S_xz);
    COPY_MIRROR(S_yy); COPY_MIRROR(S_yz); COPY_MIRROR(S_zz);
    COPY_MIRROR(damage); COPY_MIRROR(distension); COPY_MIRROR(material_id); COPY_MIRROR(clump_id);
    COPY_MIRROR(L_00); COPY_MIRROR(L_01); COPY_MIRROR(L_02);
    COPY_MIRROR(L_10); COPY_MIRROR(L_11); COPY_MIRROR(L_12);
    COPY_MIRROR(L_20); COPY_MIRROR(L_21); COPY_MIRROR(L_22);
#undef COPY_MIRROR
    p.global_id[j] = -i-1;
    const double nx=config.symmetry_normal[0], ny=config.symmetry_normal[1], nz=config.symmetry_normal[2];
    sph_reflect_vector(p.pos_x[j],p.pos_y[j],p.pos_z[j],nx,ny,nz,config.symmetry_offset);
    sph_reflect_vector(p.vel_x[j],p.vel_y[j],p.vel_z[j],nx,ny,nz);
    sph_reflect_stress(p.S_xx[j],p.S_xy[j],p.S_xz[j],p.S_yy[j],p.S_yz[j],p.S_zz[j],nx,ny,nz);
}

void initialize_sph_boundaries(SimulationState& state, const SimConfig& config,
    SphBoundaryWorkspace& workspace, cudaStream_t stream) {
    const int n=state.particles.num_particles, bs=config.gpu.block_size;
    if (n <= 0) return;
    if (config.sph.symmetry_enabled) {
        std::vector<double> x(n), y(n), z(n);
        CUDA_CHECK(cudaMemcpy(x.data(), state.particles.pos_x, n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y.data(), state.particles.pos_y, n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z.data(), state.particles.pos_z, n*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i=0; i<n; ++i)
            if (!(config.sph.symmetry_normal[0]*x[i] + config.sph.symmetry_normal[1]*y[i] +
                  config.sph.symmetry_normal[2]*z[i] + config.sph.symmetry_offset >= 0.0))
                throw std::runtime_error("SPH symmetry input must lie in the nonnegative half-space");
    }
    if (config.sph.fixed_part_count > 0) {
        std::vector<int32_t> parts(n);
        CUDA_CHECK(cudaMemcpy(parts.data(), state.particles.clump_id, n*sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int j=0; j<config.sph.fixed_part_count; ++j) {
            bool found=false;
            for (int id : parts) found |= id == config.sph.fixed_parts[j];
            if (!found) throw std::runtime_error("SPH fixed_parts references an absent Part ID");
        }
        CUDA_CHECK(cudaMalloc(&workspace.fixed, n*sizeof(SphFixedState)));
        LAUNCH_KERNEL(capture_sph_fixed_kernel, (n+bs-1)/bs, bs, 0, stream, state.particles, workspace.fixed, config.sph);
        restore_sph_fixed(state, config, workspace, stream);
    }
}

void restore_sph_fixed(SimulationState& state, const SimConfig& config,
    const SphBoundaryWorkspace& workspace, cudaStream_t stream) {
    const int n=state.particles.num_particles, bs=config.gpu.block_size;
    if (!workspace.fixed || n <= 0) return;
    LAUNCH_KERNEL(restore_sph_fixed_kernel, (n+bs-1)/bs, bs, 0, stream, state.particles, workspace.fixed);
}

void reflect_sph_domain(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    const int n=state.particles.num_particles, bs=config.gpu.block_size;
    if (!config.sph.symmetry_enabled || n <= 0) return;
    LAUNCH_KERNEL(reflect_sph_domain_kernel, (n+bs-1)/bs, bs, 0, stream, state.particles, config.sph);
}

void update_sph_mirrors(SimulationState& state, const SimConfig& config, cudaStream_t stream) {
    const int n=state.particles.num_particles, bs=config.gpu.block_size;
    if (!config.sph.symmetry_enabled || n <= 0) return;
    LAUNCH_KERNEL(update_sph_mirrors_kernel, (n+bs-1)/bs, bs, 0, stream, state.particles, config.sph);
}

void shutdown_sph_boundaries(SphBoundaryWorkspace& workspace) {
    if (workspace.fixed) cudaFree(workspace.fixed);
    workspace.fixed=nullptr;
}

} // namespace cudadem

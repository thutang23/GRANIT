// =============================================================================
// System Initialization - Host-Only Implementation
// =============================================================================
// These functions use CUDA runtime API and are host-only.
// Compiled separately from CUDA device code.
// =============================================================================

#include "system_initialize.h"
#include "particle_clump_order.h"
#include "command_line_interface.h"
#include "binary_reader.h"
#include "../wall/wall.h"
#include "../neighbor/neighbor_api.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <algorithm>

namespace cudadem {

// =============================================================================
// Memory Allocation Functions
// =============================================================================

void allocate_particle_soa(ParticleSoA& p, int32_t num_particles) {
    p.num_particles = num_particles;
    p.num_local = num_particles; // Initially all particles are local
    p.num_ghost = 0;
    p.capacity = num_particles;

    size_t size_d = num_particles * sizeof(double);
    size_t size_f = num_particles * sizeof(float);
    size_t size_i = num_particles * sizeof(int32_t);
    size_t size_l = num_particles * sizeof(int64_t);

    // Position (double)
    CUDA_CHECK(cudaMalloc(&p.pos_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.pos_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.pos_z, size_d));

    // Last Position (double)
    CUDA_CHECK(cudaMalloc(&p.last_pos_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.last_pos_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.last_pos_z, size_d));

    // Velocity (double)
    CUDA_CHECK(cudaMalloc(&p.vel_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.vel_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.vel_z, size_d));

    CUDA_CHECK(cudaMalloc(&p.winding_x, size_l));
    CUDA_CHECK(cudaMalloc(&p.winding_y, size_l));
    CUDA_CHECK(cudaMalloc(&p.winding_z, size_l));

    // Force (double)
    CUDA_CHECK(cudaMalloc(&p.force_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.force_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.force_z, size_d));

    // RESPA outer-loop gravity acceleration (double)
    CUDA_CHECK(cudaMalloc(&p.gravity_force_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.gravity_force_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.gravity_force_z, size_d));

    // Acceleration aliases (point to force arrays)
    p.acc_x = p.force_x;
    p.acc_y = p.force_y;
    p.acc_z = p.force_z;

    // Angular velocity (double)
    CUDA_CHECK(cudaMalloc(&p.omega_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.omega_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.omega_z, size_d));

    // Torque (double)
    CUDA_CHECK(cudaMalloc(&p.torque_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.torque_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.torque_z, size_d));

    // Angular acceleration aliases (point to torque arrays)
    p.alpha_x = p.torque_x;
    p.alpha_y = p.torque_y;
    p.alpha_z = p.torque_z;

    // Quaternion (double)
    CUDA_CHECK(cudaMalloc(&p.quat_w, size_d));
    CUDA_CHECK(cudaMalloc(&p.quat_x, size_d));
    CUDA_CHECK(cudaMalloc(&p.quat_y, size_d));
    CUDA_CHECK(cudaMalloc(&p.quat_z, size_d));

    // Properties (float)
    CUDA_CHECK(cudaMalloc(&p.radius, size_f));
    CUDA_CHECK(cudaMalloc(&p.mass, size_f));
    CUDA_CHECK(cudaMalloc(&p.moment_of_inertia, size_f));
    CUDA_CHECK(cudaMalloc(&p.support_extent, size_f));  // Interaction support extent
    CUDA_CHECK(cudaMalloc(&p.cohesive_strength, size_f));
    // SPH Properties (float/double). Density and rho_rate stay FP64 so weak
    // hydrostatic compression is not lost when integrated around rho0.
    CUDA_CHECK(cudaMalloc(&p.density, size_d));
    CUDA_CHECK(cudaMalloc(&p.pressure, size_f));
    CUDA_CHECK(cudaMalloc(&p.viscosity, size_f));
    CUDA_CHECK(cudaMalloc(&p.c_sound, size_f));
    CUDA_CHECK(cudaMalloc(&p.drhodt, size_f));
    CUDA_CHECK(cudaMalloc(&p.rho_rate, size_d));  // Density rate of change (continuous density mode)
    CUDA_CHECK(cudaMalloc(&p.smoothing_length, size_f));
    CUDA_CHECK(cudaMalloc(&p.energy, size_d));
    CUDA_CHECK(cudaMalloc(&p.energy_cold, size_d));
    CUDA_CHECK(cudaMemset(p.energy_cold, 0, size_d));
    CUDA_CHECK(cudaMalloc(&p.energy_thermal_peak, size_d));
    CUDA_CHECK(cudaMemset(p.energy_thermal_peak, 0, size_d));
    CUDA_CHECK(cudaMalloc(&p.pressure_peak, size_d));
    CUDA_CHECK(cudaMemset(p.pressure_peak, 0, size_d));
    CUDA_CHECK(cudaMalloc(&p.fracture_area, size_d));
    CUDA_CHECK(cudaMemset(p.fracture_area, 0, size_d));
    CUDA_CHECK(cudaMalloc(&p.fragment_length, size_d));
    CUDA_CHECK(cudaMemset(p.fragment_length, 0, size_d));
    CUDA_CHECK(cudaMalloc(&p.damping_energy, size_d));
    CUDA_CHECK(cudaMemset(p.damping_energy, 0, size_d));

    CUDA_CHECK(cudaMalloc(&p.dedt, size_d));

    // Zero-init SPH scalars that are otherwise left as cudaMalloc garbage.
    // For DEM runs these are never loaded from the .bin (no SPH descriptors)
    // and never set by SPH init, so without this they hold garbage. Critical
    // for method:"direct" self-gravity, which softens via smoothing_length:
    // h<=0 yields pure 1/r^2 Newtonian (matches DEMBody); garbage h corrupts it.
    CUDA_CHECK(cudaMemset(p.pressure,        0, size_f));
    CUDA_CHECK(cudaMemset(p.viscosity,       0, size_f));
    CUDA_CHECK(cudaMemset(p.c_sound,         0, size_f));
    CUDA_CHECK(cudaMemset(p.drhodt,           0, size_f));
    CUDA_CHECK(cudaMemset(p.rho_rate,        0, size_d));
    CUDA_CHECK(cudaMemset(p.smoothing_length, 0, size_f));
    CUDA_CHECK(cudaMemset(p.energy,          0, size_d));
    CUDA_CHECK(cudaMemset(p.dedt,            0, size_d));

    // SPH material state - deviatoric stress tensor (6 symmetric SoA components)
    CUDA_CHECK(cudaMalloc(&p.S_xx, size_f));
    CUDA_CHECK(cudaMalloc(&p.S_xy, size_f));
    CUDA_CHECK(cudaMalloc(&p.S_xz, size_f));
    CUDA_CHECK(cudaMalloc(&p.S_yy, size_f));
    CUDA_CHECK(cudaMalloc(&p.S_yz, size_f));
    CUDA_CHECK(cudaMalloc(&p.S_zz, size_f));
    // Grady-Kipp damage scalars
    CUDA_CHECK(cudaMalloc(&p.damage, size_f));
    CUDA_CHECK(cudaMalloc(&p.crack_strain, size_f));
    CUDA_CHECK(cudaMalloc(&p.flaws_ntot, size_i));
    CUDA_CHECK(cudaMalloc(&p.flaws_strain_min, size_f));
    CUDA_CHECK(cudaMalloc(&p.flaws_strain_max, size_f));
    CUDA_CHECK(cudaMalloc(&p.flaws_m0, size_f));
    // P-alpha porosity distension
    CUDA_CHECK(cudaMalloc(&p.distension, size_f));
    // CSPH kernel gradient correction matrix (3x3 row-major SoA)
    CUDA_CHECK(cudaMalloc(&p.L_00, size_f));  CUDA_CHECK(cudaMalloc(&p.L_01, size_f));  CUDA_CHECK(cudaMalloc(&p.L_02, size_f));
    CUDA_CHECK(cudaMalloc(&p.L_10, size_f));  CUDA_CHECK(cudaMalloc(&p.L_11, size_f));  CUDA_CHECK(cudaMalloc(&p.L_12, size_f));
    CUDA_CHECK(cudaMalloc(&p.L_20, size_f));  CUDA_CHECK(cudaMalloc(&p.L_21, size_f));  CUDA_CHECK(cudaMalloc(&p.L_22, size_f));

    // Contact tracking (int32)
    CUDA_CHECK(cudaMalloc(&p.contact_history_offset, size_i));
    CUDA_CHECK(cudaMalloc(&p.num_contacts, size_i));

    // Wall contact tracking (int32)
    CUDA_CHECK(cudaMalloc(&p.wall_history_offset, size_i));
    CUDA_CHECK(cudaMalloc(&p.num_wall_contacts, size_i));

    // Neighbor (int32)
    CUDA_CHECK(cudaMalloc(&p.cell_id, size_i));
    CUDA_CHECK(cudaMalloc(&p.sorted_index, size_i));

    // Multi-GPU (int32)
    CUDA_CHECK(cudaMalloc(&p.global_id, size_i));
    CUDA_CHECK(cudaMalloc(&p.material_id, size_i));
    CUDA_CHECK(cudaMalloc(&p.lifecycle_state, num_particles * sizeof(uint8_t)));

    // Clump fields
    CUDA_CHECK(cudaMalloc(&p.clump_id,    num_particles * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&p.clump_off_x, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.clump_off_y, num_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.clump_off_z, num_particles * sizeof(float)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(p.vel_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.vel_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.vel_z, 0, size_d));
    CUDA_CHECK(cudaMemset(p.winding_x, 0, size_l));
    CUDA_CHECK(cudaMemset(p.winding_y, 0, size_l));
    CUDA_CHECK(cudaMemset(p.winding_z, 0, size_l));
    CUDA_CHECK(cudaMemset(p.force_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.force_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.force_z, 0, size_d));
    CUDA_CHECK(cudaMemset(p.gravity_force_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.gravity_force_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.gravity_force_z, 0, size_d));
    CUDA_CHECK(cudaMemset(p.omega_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.omega_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.omega_z, 0, size_d));
    CUDA_CHECK(cudaMemset(p.torque_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.torque_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.torque_z, 0, size_d));
    CUDA_CHECK(cudaMemset(p.num_contacts, 0, size_i));
    CUDA_CHECK(cudaMemset(p.num_wall_contacts, 0, size_i));
    CUDA_CHECK(cudaMemset(p.material_id, 0xFF, size_i));
    CUDA_CHECK(cudaMemset(p.lifecycle_state, 0, num_particles * sizeof(uint8_t)));
    // Zero-init SPH material state arrays
    CUDA_CHECK(cudaMemset(p.S_xx, 0, size_f));  CUDA_CHECK(cudaMemset(p.S_xy, 0, size_f));  CUDA_CHECK(cudaMemset(p.S_xz, 0, size_f));
    CUDA_CHECK(cudaMemset(p.S_yy, 0, size_f));  CUDA_CHECK(cudaMemset(p.S_yz, 0, size_f));  CUDA_CHECK(cudaMemset(p.S_zz, 0, size_f));
    CUDA_CHECK(cudaMemset(p.damage, 0, size_f));
    CUDA_CHECK(cudaMemset(p.crack_strain, 0, size_f));
    CUDA_CHECK(cudaMemset(p.flaws_ntot, 0, size_i));
    CUDA_CHECK(cudaMemset(p.flaws_strain_min, 0, size_f));
    CUDA_CHECK(cudaMemset(p.flaws_strain_max, 0, size_f));
    CUDA_CHECK(cudaMemset(p.flaws_m0, 0, size_f));
    CUDA_CHECK(cudaMemset(p.distension, 0, size_f)); // Will be overwritten to alpha_initial by init kernel
    CUDA_CHECK(cudaMemset(p.L_00, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_01, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_02, 0, size_f));
    CUDA_CHECK(cudaMemset(p.L_10, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_11, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_12, 0, size_f));
    CUDA_CHECK(cudaMemset(p.L_20, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_21, 0, size_f));  CUDA_CHECK(cudaMemset(p.L_22, 0, size_f));

    // Initialize quaternions to identity (w=1, x=y=z=0)
    std::vector<double> ones(num_particles, 1.0);
    CUDA_CHECK(cudaMemcpy(p.quat_w, ones.data(), size_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(p.quat_x, 0, size_d));
    CUDA_CHECK(cudaMemset(p.quat_y, 0, size_d));
    CUDA_CHECK(cudaMemset(p.quat_z, 0, size_d));

    // Unified-v3 semantics: 0 is an ordinary live particle.  The -1 value is
    // reserved for explicitly dead rows and must never be invented by an
    // allocator or a missing optional input field.
    CUDA_CHECK(cudaMemset(p.clump_id,    0,    num_particles * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(p.clump_off_x, 0,    num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(p.clump_off_y, 0,    num_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(p.clump_off_z, 0,    num_particles * sizeof(float)));

    std::cout << "Allocated particle SoA for " << num_particles << " particles" << std::endl;
}

void resize_particle_soa_preserving(ParticleSoA& p, int32_t new_capacity) {
    if (new_capacity <= p.capacity) return;
    if (new_capacity <= 0 || p.num_particles < 0 || p.num_particles > p.capacity) {
        throw std::runtime_error("Invalid ParticleSoA resize request");
    }
    const int32_t count = p.num_particles;
    ParticleSoA replacement;
    try {
        allocate_particle_soa(replacement, new_capacity);
#define COPY_PARTICLE_FIELD(name, type) \
        CUDA_CHECK(cudaMemcpy(replacement.name, p.name, static_cast<size_t>(count) * sizeof(type), cudaMemcpyDeviceToDevice))
        COPY_PARTICLE_FIELD(pos_x, double); COPY_PARTICLE_FIELD(pos_y, double); COPY_PARTICLE_FIELD(pos_z, double);
        COPY_PARTICLE_FIELD(last_pos_x, double); COPY_PARTICLE_FIELD(last_pos_y, double); COPY_PARTICLE_FIELD(last_pos_z, double);
        COPY_PARTICLE_FIELD(vel_x, double); COPY_PARTICLE_FIELD(vel_y, double); COPY_PARTICLE_FIELD(vel_z, double);
        COPY_PARTICLE_FIELD(winding_x, int64_t); COPY_PARTICLE_FIELD(winding_y, int64_t); COPY_PARTICLE_FIELD(winding_z, int64_t);
        COPY_PARTICLE_FIELD(force_x, double); COPY_PARTICLE_FIELD(force_y, double); COPY_PARTICLE_FIELD(force_z, double);
        COPY_PARTICLE_FIELD(gravity_force_x, double); COPY_PARTICLE_FIELD(gravity_force_y, double); COPY_PARTICLE_FIELD(gravity_force_z, double);
        COPY_PARTICLE_FIELD(omega_x, double); COPY_PARTICLE_FIELD(omega_y, double); COPY_PARTICLE_FIELD(omega_z, double);
        COPY_PARTICLE_FIELD(torque_x, double); COPY_PARTICLE_FIELD(torque_y, double); COPY_PARTICLE_FIELD(torque_z, double);
        COPY_PARTICLE_FIELD(quat_w, double); COPY_PARTICLE_FIELD(quat_x, double); COPY_PARTICLE_FIELD(quat_y, double); COPY_PARTICLE_FIELD(quat_z, double);
        COPY_PARTICLE_FIELD(radius, float); COPY_PARTICLE_FIELD(mass, float); COPY_PARTICLE_FIELD(moment_of_inertia, float);
        COPY_PARTICLE_FIELD(support_extent, float); COPY_PARTICLE_FIELD(cohesive_strength, float);
        COPY_PARTICLE_FIELD(density, double); COPY_PARTICLE_FIELD(pressure, float); COPY_PARTICLE_FIELD(viscosity, float); COPY_PARTICLE_FIELD(c_sound, float);
        COPY_PARTICLE_FIELD(drhodt, float); COPY_PARTICLE_FIELD(rho_rate, double); COPY_PARTICLE_FIELD(smoothing_length, float);
        COPY_PARTICLE_FIELD(energy, double); COPY_PARTICLE_FIELD(energy_cold, double); COPY_PARTICLE_FIELD(energy_thermal_peak, double); COPY_PARTICLE_FIELD(pressure_peak, double); COPY_PARTICLE_FIELD(fracture_area, double); COPY_PARTICLE_FIELD(fragment_length, double); COPY_PARTICLE_FIELD(damping_energy, double);  COPY_PARTICLE_FIELD(dedt, double);
        COPY_PARTICLE_FIELD(S_xx, float); COPY_PARTICLE_FIELD(S_xy, float); COPY_PARTICLE_FIELD(S_xz, float); COPY_PARTICLE_FIELD(S_yy, float); COPY_PARTICLE_FIELD(S_yz, float); COPY_PARTICLE_FIELD(S_zz, float);
        COPY_PARTICLE_FIELD(damage, float); COPY_PARTICLE_FIELD(crack_strain, float); COPY_PARTICLE_FIELD(flaws_ntot, int32_t);
        COPY_PARTICLE_FIELD(flaws_strain_min, float); COPY_PARTICLE_FIELD(flaws_strain_max, float); COPY_PARTICLE_FIELD(flaws_m0, float); COPY_PARTICLE_FIELD(distension, float);
        COPY_PARTICLE_FIELD(L_00, float); COPY_PARTICLE_FIELD(L_01, float); COPY_PARTICLE_FIELD(L_02, float); COPY_PARTICLE_FIELD(L_10, float); COPY_PARTICLE_FIELD(L_11, float); COPY_PARTICLE_FIELD(L_12, float); COPY_PARTICLE_FIELD(L_20, float); COPY_PARTICLE_FIELD(L_21, float); COPY_PARTICLE_FIELD(L_22, float);
        COPY_PARTICLE_FIELD(contact_history_offset, int32_t); COPY_PARTICLE_FIELD(num_contacts, int32_t);
        COPY_PARTICLE_FIELD(wall_history_offset, int32_t); COPY_PARTICLE_FIELD(num_wall_contacts, int32_t);
        COPY_PARTICLE_FIELD(cell_id, int32_t); COPY_PARTICLE_FIELD(sorted_index, int32_t); COPY_PARTICLE_FIELD(global_id, int32_t); COPY_PARTICLE_FIELD(material_id, int32_t); COPY_PARTICLE_FIELD(lifecycle_state, uint8_t);
        COPY_PARTICLE_FIELD(clump_id, int32_t); COPY_PARTICLE_FIELD(clump_off_x, float); COPY_PARTICLE_FIELD(clump_off_y, float); COPY_PARTICLE_FIELD(clump_off_z, float);
#undef COPY_PARTICLE_FIELD
    } catch (const std::exception& e) {
        free_particle_soa(replacement);
        std::ostringstream msg;
        msg << "FATAL core ParticleSoA resize OOM/failure: old_capacity=" << p.capacity
            << ", requested_capacity=" << new_capacity << ", active_particles=" << count
            << ", cuda_error=" << e.what();
        throw std::runtime_error(msg.str());
    }
    replacement.num_particles = p.num_particles;
    replacement.num_local = p.num_local;
    replacement.num_ghost = p.num_ghost;
    replacement.input_has_omega = p.input_has_omega;
    replacement.input_has_smoothing_length = p.input_has_smoothing_length;
    replacement.input_has_clump_offset = p.input_has_clump_offset;
    replacement.input_has_winding = p.input_has_winding;
    replacement.input_has_damage_flaw_metadata = p.input_has_damage_flaw_metadata;
    replacement.source_record_kind = p.source_record_kind;
    free_particle_soa(p);
    p = replacement;
}

void canonicalize_particle_clump_order(
    ParticleSoA& p, ContactHistoryBuffer* contact_history,
    bool preserve_active_order) {
    const int32_t n=p.num_particles; if(n<=1) return;
    std::vector<int32_t> id(n);
    std::vector<uint8_t> life(n);
    CUDA_CHECK(cudaMemcpy(id.data(),p.clump_id,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(life.data(),p.lifecycle_state,n*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    const std::vector<int32_t> order=particle_clump_order(id,life,preserve_active_order);
#define PERMUTE_PARTICLE_FIELD(name,type) do { std::vector<type> a(n), b(n); CUDA_CHECK(cudaMemcpy(a.data(),p.name,n*sizeof(type),cudaMemcpyDeviceToHost)); for(int i=0;i<n;++i)b[i]=a[order[i]]; CUDA_CHECK(cudaMemcpy(p.name,b.data(),n*sizeof(type),cudaMemcpyHostToDevice)); } while(0)
    PERMUTE_PARTICLE_FIELD(pos_x,double);PERMUTE_PARTICLE_FIELD(pos_y,double);PERMUTE_PARTICLE_FIELD(pos_z,double);PERMUTE_PARTICLE_FIELD(last_pos_x,double);PERMUTE_PARTICLE_FIELD(last_pos_y,double);PERMUTE_PARTICLE_FIELD(last_pos_z,double);
    PERMUTE_PARTICLE_FIELD(vel_x,double);PERMUTE_PARTICLE_FIELD(vel_y,double);PERMUTE_PARTICLE_FIELD(vel_z,double);PERMUTE_PARTICLE_FIELD(force_x,double);PERMUTE_PARTICLE_FIELD(force_y,double);PERMUTE_PARTICLE_FIELD(force_z,double);PERMUTE_PARTICLE_FIELD(gravity_force_x,double);PERMUTE_PARTICLE_FIELD(gravity_force_y,double);PERMUTE_PARTICLE_FIELD(gravity_force_z,double);PERMUTE_PARTICLE_FIELD(omega_x,double);PERMUTE_PARTICLE_FIELD(omega_y,double);PERMUTE_PARTICLE_FIELD(omega_z,double);PERMUTE_PARTICLE_FIELD(torque_x,double);PERMUTE_PARTICLE_FIELD(torque_y,double);PERMUTE_PARTICLE_FIELD(torque_z,double);PERMUTE_PARTICLE_FIELD(quat_w,double);PERMUTE_PARTICLE_FIELD(quat_x,double);PERMUTE_PARTICLE_FIELD(quat_y,double);PERMUTE_PARTICLE_FIELD(quat_z,double);PERMUTE_PARTICLE_FIELD(energy,double);PERMUTE_PARTICLE_FIELD(energy_cold,double);PERMUTE_PARTICLE_FIELD(energy_thermal_peak,double);PERMUTE_PARTICLE_FIELD(pressure_peak,double);PERMUTE_PARTICLE_FIELD(fracture_area,double);PERMUTE_PARTICLE_FIELD(fragment_length,double);PERMUTE_PARTICLE_FIELD(damping_energy,double);PERMUTE_PARTICLE_FIELD(dedt,double);
    PERMUTE_PARTICLE_FIELD(winding_x,int64_t);PERMUTE_PARTICLE_FIELD(winding_y,int64_t);PERMUTE_PARTICLE_FIELD(winding_z,int64_t);
    PERMUTE_PARTICLE_FIELD(radius,float);PERMUTE_PARTICLE_FIELD(mass,float);PERMUTE_PARTICLE_FIELD(moment_of_inertia,float);PERMUTE_PARTICLE_FIELD(support_extent,float);PERMUTE_PARTICLE_FIELD(cohesive_strength,float);PERMUTE_PARTICLE_FIELD(density,double);PERMUTE_PARTICLE_FIELD(pressure,float);PERMUTE_PARTICLE_FIELD(viscosity,float);PERMUTE_PARTICLE_FIELD(c_sound,float);PERMUTE_PARTICLE_FIELD(drhodt,float);PERMUTE_PARTICLE_FIELD(rho_rate,double);PERMUTE_PARTICLE_FIELD(smoothing_length,float);PERMUTE_PARTICLE_FIELD(S_xx,float);PERMUTE_PARTICLE_FIELD(S_xy,float);PERMUTE_PARTICLE_FIELD(S_xz,float);PERMUTE_PARTICLE_FIELD(S_yy,float);PERMUTE_PARTICLE_FIELD(S_yz,float);PERMUTE_PARTICLE_FIELD(S_zz,float);PERMUTE_PARTICLE_FIELD(damage,float);PERMUTE_PARTICLE_FIELD(crack_strain,float);PERMUTE_PARTICLE_FIELD(flaws_strain_min,float);PERMUTE_PARTICLE_FIELD(flaws_strain_max,float);PERMUTE_PARTICLE_FIELD(flaws_m0,float);PERMUTE_PARTICLE_FIELD(distension,float);PERMUTE_PARTICLE_FIELD(L_00,float);PERMUTE_PARTICLE_FIELD(L_01,float);PERMUTE_PARTICLE_FIELD(L_02,float);PERMUTE_PARTICLE_FIELD(L_10,float);PERMUTE_PARTICLE_FIELD(L_11,float);PERMUTE_PARTICLE_FIELD(L_12,float);PERMUTE_PARTICLE_FIELD(L_20,float);PERMUTE_PARTICLE_FIELD(L_21,float);PERMUTE_PARTICLE_FIELD(L_22,float);PERMUTE_PARTICLE_FIELD(clump_off_x,float);PERMUTE_PARTICLE_FIELD(clump_off_y,float);PERMUTE_PARTICLE_FIELD(clump_off_z,float);
    PERMUTE_PARTICLE_FIELD(contact_history_offset,int32_t);PERMUTE_PARTICLE_FIELD(num_contacts,int32_t);PERMUTE_PARTICLE_FIELD(wall_history_offset,int32_t);PERMUTE_PARTICLE_FIELD(num_wall_contacts,int32_t);PERMUTE_PARTICLE_FIELD(cell_id,int32_t);PERMUTE_PARTICLE_FIELD(sorted_index,int32_t);PERMUTE_PARTICLE_FIELD(global_id,int32_t);PERMUTE_PARTICLE_FIELD(material_id,int32_t);PERMUTE_PARTICLE_FIELD(lifecycle_state,uint8_t);PERMUTE_PARTICLE_FIELD(clump_id,int32_t);PERMUTE_PARTICLE_FIELD(flaws_ntot,int32_t);
#undef PERMUTE_PARTICLE_FIELD

    if (contact_history && contact_history->data) {
        const int32_t stride = contact_history->max_contacts_per_particle;
        const int64_t required = static_cast<int64_t>(n) * stride;
        if (stride <= 0 || required > contact_history->capacity) {
            throw std::runtime_error("contact history is too small for particle canonicalization");
        }
        std::vector<ContactHistory> before(static_cast<size_t>(required));
        std::vector<ContactHistory> after(static_cast<size_t>(required));
        CUDA_CHECK(cudaMemcpy(before.data(), contact_history->data,
                              static_cast<size_t>(required) * sizeof(ContactHistory),
                              cudaMemcpyDeviceToHost));
        for (int32_t i = 0; i < n; ++i) {
            std::copy_n(before.data() + static_cast<size_t>(order[i]) * stride,
                        stride,
                        after.data() + static_cast<size_t>(i) * stride);
        }
        CUDA_CHECK(cudaMemcpy(contact_history->data, after.data(),
                              static_cast<size_t>(required) * sizeof(ContactHistory),
                              cudaMemcpyHostToDevice));
        std::vector<int32_t> offsets(n);
        for (int32_t i = 0; i < n; ++i) offsets[i] = i * stride;
        CUDA_CHECK(cudaMemcpy(p.contact_history_offset, offsets.data(),
                              static_cast<size_t>(n) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
    }
}

void sanitize_inactive_sph_tail_for_serialization(
    ParticleSoA& p, int32_t live_particles, int32_t total_particles,
    float fallback_smoothing_length) {
    if (live_particles < 0 || total_particles < live_particles ||
        total_particles > p.capacity ||
        !(std::isfinite(fallback_smoothing_length) &&
          fallback_smoothing_length > 0.0f)) {
        throw std::runtime_error("invalid SPH inactive-tail serialization range");
    }
    const int32_t tail_count = total_particles - live_particles;
    if (tail_count == 0) return;

    std::vector<float> smoothing_length(tail_count);
    std::vector<float> distension(tail_count);
    CUDA_CHECK(cudaMemcpy(smoothing_length.data(), p.smoothing_length + live_particles,
                          static_cast<size_t>(tail_count) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(distension.data(), p.distension + live_particles,
                          static_cast<size_t>(tail_count) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const auto original_smoothing_length = smoothing_length;
    const auto original_distension = distension;
    sanitize_inactive_sph_tail_values(smoothing_length.data(), distension.data(),
                                      tail_count, fallback_smoothing_length);
    if (smoothing_length != original_smoothing_length) {
        CUDA_CHECK(cudaMemcpy(p.smoothing_length + live_particles,
                              smoothing_length.data(),
                              static_cast<size_t>(tail_count) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (distension != original_distension) {
        CUDA_CHECK(cudaMemcpy(p.distension + live_particles, distension.data(),
                              static_cast<size_t>(tail_count) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void allocate_contact_history(ContactHistoryBuffer& ch,
                              int32_t num_particles,
                              int32_t max_contacts_per_particle) {
    ch.max_contacts_per_particle = max_contacts_per_particle;
    ch.capacity = num_particles * max_contacts_per_particle;

    CUDA_CHECK(cudaMalloc(&ch.data, ch.capacity * sizeof(ContactHistory)));

    // IMPORTANT: Cannot use memset(0xFF) because ContactHistory contains mixed types:
    // - int32_t fields: 0xFF -> -1 (correct for invalid marker)
    // - bool is_active: 0xFF -> true (WRONG! should be false)
    // - float fields: 0xFF -> NaN (WRONG! should be 0.0f)
    //
    // Solution: Initialize on host with proper values, then copy to device
    std::vector<ContactHistory> h_init(ch.capacity);
    for (auto& h : h_init) {
        h.partner_global_id = -1;
        h.partner_local_id = -1;
        h.is_active = false;
        h.seen_this_step = false;
        h.delta_t_x = 0.0f;
        h.delta_t_y = 0.0f;
        h.delta_t_z = 0.0f;
        h.delta_r_x = 0.0f;
        h.delta_r_y = 0.0f;
        h.delta_r_z = 0.0f;
        h.delta_tw_x = 0.0f;
        h.delta_tw_y = 0.0f;
        h.delta_tw_z = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(ch.data, h_init.data(),
                          ch.capacity * sizeof(ContactHistory),
                          cudaMemcpyHostToDevice));

    std::cout << "Allocated contact history buffer: " << ch.capacity << " slots" << std::endl;
}

void resize_contact_history_preserving(ContactHistoryBuffer& ch, int32_t num_particles) {
    if (!ch.data) return;
    if (num_particles <= 0 || ch.max_contacts_per_particle <= 0) {
        throw std::runtime_error("Invalid contact-history resize request");
    }
    const int64_t requested64 = static_cast<int64_t>(num_particles) * ch.max_contacts_per_particle;
    if (requested64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("FATAL contact-history resize overflow");
    }
    const int32_t requested = static_cast<int32_t>(requested64);
    if (requested <= ch.capacity) return;
    ContactHistoryBuffer replacement;
    try {
        allocate_contact_history(replacement, num_particles, ch.max_contacts_per_particle);
        CUDA_CHECK(cudaMemcpy(replacement.data, ch.data,
                              static_cast<size_t>(ch.capacity) * sizeof(ContactHistory),
                              cudaMemcpyDeviceToDevice));
    } catch (const std::exception& e) {
        free_contact_history(replacement);
        throw std::runtime_error(std::string("FATAL contact-history resize OOM: old_capacity=")
            + std::to_string(ch.capacity) + ", requested_capacity=" + std::to_string(requested)
            + ", cuda_error=" + e.what());
    }
    free_contact_history(ch);
    ch = replacement;
}

void resize_wall_contact_history_preserving(WallContactHistoryBuffer& wch, int32_t num_particles) {
    if (!wch.data) return;
    if (num_particles <= 0 || wch.max_contacts_per_particle <= 0) {
        throw std::runtime_error("Invalid wall-contact-history resize request");
    }
    const int64_t requested64 = static_cast<int64_t>(num_particles) * wch.max_contacts_per_particle;
    if (requested64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("FATAL wall-contact-history resize overflow");
    }
    const int32_t requested = static_cast<int32_t>(requested64);
    if (requested <= wch.capacity) return;
    WallContactHistoryBuffer replacement;
    try {
        allocate_wall_contact_history(replacement, num_particles, wch.max_contacts_per_particle);
        CUDA_CHECK(cudaMemcpy(replacement.data, wch.data,
                              static_cast<size_t>(wch.capacity) * sizeof(WallContactHistory),
                              cudaMemcpyDeviceToDevice));
    } catch (const std::exception& e) {
        free_wall_contact_history(replacement);
        throw std::runtime_error(std::string("FATAL wall-contact-history resize OOM: old_capacity=")
            + std::to_string(wch.capacity) + ", requested_capacity=" + std::to_string(requested)
            + ", cuda_error=" + e.what());
    }
    free_wall_contact_history(wch);
    wch = replacement;
}

void allocate_wall_soa(WallSoA& w, int32_t num_walls) {
    if (num_walls == 0) {
        w.num_walls = 0;
        return;
    }

    w.num_walls = num_walls;

    size_t size_d = num_walls * sizeof(double);
    size_t size_f = num_walls * sizeof(float);
    size_t size_i = num_walls * sizeof(int32_t);
    size_t size_b = num_walls * sizeof(bool);
    size_t size_u = num_walls * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&w.id, size_i));
    CUDA_CHECK(cudaMalloc(&w.type, size_u));
    CUDA_CHECK(cudaMalloc(&w.enabled, size_b));

    CUDA_CHECK(cudaMalloc(&w.pos_x, size_d));
    CUDA_CHECK(cudaMalloc(&w.pos_y, size_d));
    CUDA_CHECK(cudaMalloc(&w.pos_z, size_d));
    CUDA_CHECK(cudaMalloc(&w.normal_x, size_d));
    CUDA_CHECK(cudaMalloc(&w.normal_y, size_d));
    CUDA_CHECK(cudaMalloc(&w.normal_z, size_d));

    CUDA_CHECK(cudaMalloc(&w.vel_x, size_d));
    CUDA_CHECK(cudaMalloc(&w.vel_y, size_d));
    CUDA_CHECK(cudaMalloc(&w.vel_z, size_d));

    CUDA_CHECK(cudaMalloc(&w.axis, size_u));
    CUDA_CHECK(cudaMalloc(&w.bound_min, size_f));
    CUDA_CHECK(cudaMalloc(&w.bound_max, size_f));

    std::cout << "Allocated wall SoA for " << num_walls << " walls" << std::endl;
}

// =============================================================================
// Free Memory Functions
// =============================================================================

void free_particle_soa(ParticleSoA& p) {
    if (p.pos_x) cudaFree(p.pos_x);
    if (p.pos_y) cudaFree(p.pos_y);
    if (p.pos_z) cudaFree(p.pos_z);
    if (p.last_pos_x) cudaFree(p.last_pos_x);
    if (p.last_pos_y) cudaFree(p.last_pos_y);
    if (p.last_pos_z) cudaFree(p.last_pos_z);
    if (p.vel_x) cudaFree(p.vel_x);
    if (p.vel_y) cudaFree(p.vel_y);
    if (p.vel_z) cudaFree(p.vel_z);
    if (p.winding_x) cudaFree(p.winding_x);
    if (p.winding_y) cudaFree(p.winding_y);
    if (p.winding_z) cudaFree(p.winding_z);
    if (p.force_x) cudaFree(p.force_x);
    if (p.force_y) cudaFree(p.force_y);
    if (p.force_z) cudaFree(p.force_z);
    if (p.gravity_force_x) cudaFree(p.gravity_force_x);
    if (p.gravity_force_y) cudaFree(p.gravity_force_y);
    if (p.gravity_force_z) cudaFree(p.gravity_force_z);
    if (p.omega_x) cudaFree(p.omega_x);
    if (p.omega_y) cudaFree(p.omega_y);
    if (p.omega_z) cudaFree(p.omega_z);
    if (p.torque_x) cudaFree(p.torque_x);
    if (p.torque_y) cudaFree(p.torque_y);
    if (p.torque_z) cudaFree(p.torque_z);
    if (p.quat_w) cudaFree(p.quat_w);
    if (p.quat_x) cudaFree(p.quat_x);
    if (p.quat_y) cudaFree(p.quat_y);
    if (p.quat_z) cudaFree(p.quat_z);
    if (p.radius) cudaFree(p.radius);
    if (p.mass) cudaFree(p.mass);
    if (p.moment_of_inertia) cudaFree(p.moment_of_inertia);
    if (p.support_extent) cudaFree(p.support_extent);
    if (p.density) cudaFree(p.density);
    if (p.pressure) cudaFree(p.pressure);
    if (p.viscosity) cudaFree(p.viscosity);
    if (p.c_sound) cudaFree(p.c_sound);
    if (p.drhodt) cudaFree(p.drhodt);
if (p.rho_rate) cudaFree(p.rho_rate);
    if (p.smoothing_length) cudaFree(p.smoothing_length);
    if (p.energy) cudaFree(p.energy);
    if (p.energy_cold) cudaFree(p.energy_cold);
    if (p.energy_thermal_peak) cudaFree(p.energy_thermal_peak);
    if (p.pressure_peak) cudaFree(p.pressure_peak);
    if (p.fracture_area) cudaFree(p.fracture_area);
    if (p.fragment_length) cudaFree(p.fragment_length);
    if (p.damping_energy) cudaFree(p.damping_energy);

    if (p.dedt) cudaFree(p.dedt);
    if (p.cohesive_strength) cudaFree(p.cohesive_strength);
    if (p.S_xx) cudaFree(p.S_xx);    if (p.S_xy) cudaFree(p.S_xy);    if (p.S_xz) cudaFree(p.S_xz);
    if (p.S_yy) cudaFree(p.S_yy);    if (p.S_yz) cudaFree(p.S_yz);    if (p.S_zz) cudaFree(p.S_zz);
    if (p.damage) cudaFree(p.damage);
    if (p.crack_strain) cudaFree(p.crack_strain);
    if (p.flaws_ntot) cudaFree(p.flaws_ntot);
    if (p.flaws_strain_min) cudaFree(p.flaws_strain_min);
    if (p.flaws_strain_max) cudaFree(p.flaws_strain_max);
    if (p.flaws_m0) cudaFree(p.flaws_m0);
    if (p.distension) cudaFree(p.distension);
    if (p.L_00) cudaFree(p.L_00);  if (p.L_01) cudaFree(p.L_01);  if (p.L_02) cudaFree(p.L_02);
    if (p.L_10) cudaFree(p.L_10);  if (p.L_11) cudaFree(p.L_11);  if (p.L_12) cudaFree(p.L_12);
    if (p.L_20) cudaFree(p.L_20);  if (p.L_21) cudaFree(p.L_21);  if (p.L_22) cudaFree(p.L_22);
    if (p.contact_history_offset) cudaFree(p.contact_history_offset);
    if (p.num_contacts) cudaFree(p.num_contacts);
    if (p.wall_history_offset) cudaFree(p.wall_history_offset);
    if (p.num_wall_contacts) cudaFree(p.num_wall_contacts);
    if (p.cell_id) cudaFree(p.cell_id);
    if (p.sorted_index) cudaFree(p.sorted_index);
    if (p.global_id) cudaFree(p.global_id);
    if (p.material_id) cudaFree(p.material_id);
    if (p.lifecycle_state) cudaFree(p.lifecycle_state);
    if (p.clump_id)    cudaFree(p.clump_id);
    if (p.clump_off_x) cudaFree(p.clump_off_x);
    if (p.clump_off_y) cudaFree(p.clump_off_y);
    if (p.clump_off_z) cudaFree(p.clump_off_z);
    p = ParticleSoA();
}

void free_contact_history(ContactHistoryBuffer& ch) {
    if (ch.data) cudaFree(ch.data);
    ch = ContactHistoryBuffer();
}

void free_wall_soa(WallSoA& w) {
    if (w.id) cudaFree(w.id);
    if (w.type) cudaFree(w.type);
    if (w.enabled) cudaFree(w.enabled);
    if (w.pos_x) cudaFree(w.pos_x);
    if (w.pos_y) cudaFree(w.pos_y);
    if (w.pos_z) cudaFree(w.pos_z);
    if (w.normal_x) cudaFree(w.normal_x);
    if (w.normal_y) cudaFree(w.normal_y);
    if (w.normal_z) cudaFree(w.normal_z);
    if (w.vel_x) cudaFree(w.vel_x);
    if (w.vel_y) cudaFree(w.vel_y);
    if (w.vel_z) cudaFree(w.vel_z);
    if (w.axis) cudaFree(w.axis);
    if (w.bound_min) cudaFree(w.bound_min);
    if (w.bound_max) cudaFree(w.bound_max);
    w = WallSoA();
}

void free_simulation_state(SimulationState& state) {
    release_material_database(&state.material_database);
    free_particle_soa(state.particles);
    free_contact_history(state.contact_history);
    free_wall_contact_history(state.wall_contact_history);
    free_wall_soa(state.walls);
    NeighborSearch::shutdown(state);
}

int32_t compute_total_clump_sub_spheres(const SimConfig& config) {
    (void)config;
    // Unified v3 stores every member sphere in the input. YAML never creates
    // particle rows, so no extra ParticleSoA capacity is required.
    return 0;
}

int32_t load_particles_from_binary(ParticleSoA& particles,
                                   const std::string& filepath,
                                   const SimConfig& config,
                                   int32_t extra_capacity) {
    std::ifstream input(filepath, std::ios::binary);
    UnifiedHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT)) {
        throw std::runtime_error(
            "Fresh particle input must be a unified v3 INPUT record: " + filepath);
    }
    SimulationState temp_state;
    if (!read_unified_binary(filepath, particles, temp_state, config, extra_capacity)) {
        throw std::runtime_error("Failed to load particles from unified binary: " + filepath);
    }

    int32_t num_particles = particles.num_particles;
    std::cout << "Loaded " << num_particles << " particles from " << filepath << std::endl;
    return num_particles;
}

// =============================================================================
// Wall Initialization
// =============================================================================

void initialize_walls(WallSoA& walls, const SimConfig& config) {
    int32_t num_walls = config.num_walls;
    if (num_walls == 0) return;

    allocate_wall_soa(walls, num_walls);

    // Prepare host arrays
    std::vector<int32_t> h_id(num_walls);
    std::vector<uint8_t> h_type(num_walls);
    std::vector<uint8_t> h_enabled(num_walls);
    std::vector<double> h_pos_x(num_walls), h_pos_y(num_walls), h_pos_z(num_walls);
    std::vector<double> h_normal_x(num_walls), h_normal_y(num_walls), h_normal_z(num_walls);
    std::vector<double> h_vel_x(num_walls), h_vel_y(num_walls), h_vel_z(num_walls);
    std::vector<uint8_t> h_axis(num_walls);
    std::vector<float> h_bound_min(num_walls), h_bound_max(num_walls);

    for (int i = 0; i < num_walls; ++i) {
        const WallConfig& wc = config.walls[i];
        h_id[i] = wc.id;
        h_type[i] = static_cast<uint8_t>(wc.type);
        h_enabled[i] = wc.enabled ? 1 : 0;

        h_pos_x[i] = wc.position.x;
        h_pos_y[i] = wc.position.y;
        h_pos_z[i] = wc.position.z;

        h_normal_x[i] = wc.normal.x;
        h_normal_y[i] = wc.normal.y;
        h_normal_z[i] = wc.normal.z;

        h_vel_x[i] = wc.velocity.x;
        h_vel_y[i] = wc.velocity.y;
        h_vel_z[i] = wc.velocity.z;

        h_axis[i] = wc.axis;
        h_bound_min[i] = wc.bound_min;
        h_bound_max[i] = wc.bound_max;
    }

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(walls.id, h_id.data(), num_walls * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.type, h_type.data(), num_walls * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.enabled, h_enabled.data(), num_walls * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(walls.pos_x, h_pos_x.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.pos_y, h_pos_y.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.pos_z, h_pos_z.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(walls.normal_x, h_normal_x.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.normal_y, h_normal_y.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.normal_z, h_normal_z.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(walls.vel_x, h_vel_x.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.vel_y, h_vel_y.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.vel_z, h_vel_z.data(), num_walls * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(walls.axis, h_axis.data(), num_walls * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.bound_min, h_bound_min.data(), num_walls * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(walls.bound_max, h_bound_max.data(), num_walls * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Initialized " << num_walls << " walls" << std::endl;
}

// =============================================================================
// Timestep Calculation
// =============================================================================

std::pair<float, float> derive_particle_radius_bounds(const ParticleSoA& particles,
                                                       int32_t num_particles) {
    if (num_particles <= 0) throw std::runtime_error("DEM input has no particles");
    std::vector<float> radius(static_cast<size_t>(num_particles));
    CUDA_CHECK(cudaMemcpy(radius.data(), particles.radius,
                          radius.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto bounds = std::minmax_element(radius.begin(), radius.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second) || *bounds.first <= 0.0f)
        throw std::runtime_error("DEM input radii must be finite and positive");
    return {*bounds.first, *bounds.second};
}

double calculate_critical_timestep(const SimConfig& config,
                                   const ParticleSoA& particles,
                                   const MaterialDatabase& database,
                                   int32_t num_particles) {
    if (config.physics != SimulationPhysics::DEM) return config.sph.dt_max;
    if (!(config.time.dt_factor > 0.0f) || !std::isfinite(config.time.dt_factor))
        throw std::runtime_error("simulation.time.dt_factor must be finite and positive");
    std::vector<float> radius(static_cast<size_t>(num_particles));
    std::vector<float> mass(static_cast<size_t>(num_particles));
    std::vector<int32_t> material_id(static_cast<size_t>(num_particles));
    CUDA_CHECK(cudaMemcpy(radius.data(), particles.radius, radius.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass.data(), particles.mass, mass.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(material_id.data(), particles.material_id, material_id.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    const size_t count = database.materials.size();
    std::vector<float> min_radius(count, std::numeric_limits<float>::infinity());
    std::vector<float> min_mass(count, std::numeric_limits<float>::infinity());
    double dt_crit = std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < num_particles; ++i) {
        const int32_t id = material_id[static_cast<size_t>(i)];
        if (id < 0 || static_cast<size_t>(id) >= count || !database.materials[id].has_dem)
            throw std::runtime_error("DEM timestep encountered invalid material_id");
        const auto& dem = database.materials[id].dem;
        const double rayleigh = M_PI * static_cast<double>(radius[i]) /
            (0.163 * dem.poisson_ratio + 0.877) * std::sqrt(dem.density / dem.youngs_modulus);
        dt_crit = std::min(dt_crit, rayleigh);
        min_radius[id] = std::min(min_radius[id], radius[i]);
        min_mass[id] = std::min(min_mass[id], mass[i]);
    }

    auto contact_limit = [&](uint32_t a, uint32_t b, float r_a, float r_b,
                             float m_a, float m_b) {
        const auto& da = database.materials[a].dem;
        const auto& db = database.materials[b].dem;
        const double E_star = 1.0 /
            ((1.0 - da.poisson_ratio * da.poisson_ratio) / da.youngs_modulus +
             (1.0 - db.poisson_ratio * db.poisson_ratio) / db.youngs_modulus);
        const double R_eff = static_cast<double>(r_a) * r_b / (r_a + r_b);
        const double M_eff = static_cast<double>(m_a) * m_b / (m_a + m_b);
        const double stiffness = (4.0 / 3.0) * E_star * R_eff;
        if (stiffness > 0.0 && M_eff > 0.0)
            dt_crit = std::min(dt_crit, M_PI * std::sqrt(M_eff / stiffness));
    };

    // Every material combination actually present in the input.
    for (uint32_t a = 0; a < count; ++a) {
        if (!std::isfinite(min_radius[a])) continue;
        for (uint32_t b = a; b < count; ++b) {
            if (!std::isfinite(min_radius[b])) continue;
            contact_limit(a, b, min_radius[a], min_radius[b], min_mass[a], min_mass[b]);
        }
    }

    // Particle-wall limits use the same symmetric elastic reduction. Static
    // walls have infinite mass; dynamic sphere/bonded bodies retain configured
    // mass in the effective-mass reduction.
    for (uint32_t a = 0; a < count; ++a) {
        if (!std::isfinite(min_radius[a])) continue;
        for (int32_t w = 0; w < config.num_walls; ++w) {
            const WallConfig& wall = config.walls[w];
            if (!wall.enabled || wall.material_id < 0 ||
                static_cast<size_t>(wall.material_id) >= count) continue;
            float wall_radius = min_radius[a];
            float wall_mass = std::numeric_limits<float>::infinity();
            if (wall.type == WallType::SPHERE && wall.sphere_radius > 0.0f) {
                wall_radius = wall.sphere_radius;
                if (wall.sphere_mass > 0.0f) wall_mass = wall.sphere_mass;
            } else if (wall.type == WallType::BONDED_RECT && wall.bonded_mass > 0.0f) {
                wall_mass = wall.bonded_mass;
            }
            if (std::isfinite(wall_mass)) {
                contact_limit(a, static_cast<uint32_t>(wall.material_id),
                              min_radius[a], wall_radius, min_mass[a], wall_mass);
            } else {
                // Infinite wall: encode a very large second mass without
                // risking infinity/infinity in the reduction.
                contact_limit(a, static_cast<uint32_t>(wall.material_id),
                              min_radius[a], wall_radius, min_mass[a],
                              std::numeric_limits<float>::max());
            }
        }
    }

    if (!std::isfinite(dt_crit) || !(dt_crit > 0.0))
        throw std::runtime_error("Unable to derive a positive DEM critical timestep");
    const double contact_dt = config.time.dt_factor * dt_crit;
    double orbital_dt = std::numeric_limits<double>::infinity();
    if (config.shear_box.enabled) {
        const double omega = std::fabs(config.shear_box.omega);
        const double q = config.shear_box.shear_parameter;
        const double kappa_or_growth = std::sqrt(std::fabs(2.0 * (2.0 - q))) * omega;
        const double fastest = std::max({kappa_or_growth,
            std::fabs(config.shear_box.vertical_frequency), std::fabs(q * config.shear_box.omega)});
        if (fastest > 0.0)
            orbital_dt = config.shear_box.orbital_dt_factor / fastest;
    }
    const double explicit_dt = config.time.dt_upper_bound > 0.0
        ? config.time.dt_upper_bound : std::numeric_limits<double>::infinity();
    const double dt = std::min({contact_dt, orbital_dt, explicit_dt});

    std::cout << "Critical timestep: " << dt_crit << " s" << std::endl;
    std::cout << "Contact timestep candidate: " << contact_dt << " s" << std::endl;
    if (std::isfinite(orbital_dt))
        std::cout << "Orbital timestep candidate: " << orbital_dt << " s" << std::endl;
    if (std::isfinite(explicit_dt))
        std::cout << "Explicit timestep ceiling: " << explicit_dt << " s" << std::endl;
    std::cout << "Actual timestep: " << dt << " s" << std::endl;

    return dt;
}

// =============================================================================
// Main System Initialization Function
// =============================================================================

// Forward declaration for kernel launchers (defined in .cu files)
extern void launch_init_particle_mass_kernel(float* mass, float* moi, float* support_extent, const float* radius,
                                             const int32_t* material_id, const MaterialRecord* materials,
                                             const DemInteractionProperties* interactions,
                                             uint32_t material_count, int32_t num_particles, int block_size);
extern void launch_init_cohesive_strength_kernel(float* strength, float base_strength,
                                                  int32_t num_particles, int block_size);
extern void launch_init_contact_offsets_kernel(int32_t* offsets, int32_t max_contacts,
                                               int32_t num_particles, int block_size);

SimulationState initialize_system(const SimConfig& config) {
    SimulationState state;

    std::cout << "\n========== System Initialization ==========" << std::endl;

    // 1. Set GPU device
    CUDA_CHECK(cudaSetDevice(config.gpu.device_id));
    std::cout << "Using GPU device: " << config.gpu.device_id << std::endl;

    // 2. Copy config to GPU constant memory
    copy_config_to_device(config);
    std::cout << "Configuration loaded to GPU constant memory" << std::endl;

    // 3. Load particles from binary file
    std::string particle_path = config.io.input_particle_path;
    if (particle_path[0] == '\0') {
        throw std::runtime_error("No particle input file specified in config");
    }

    // Pre-compute clump sub-sphere count for capacity allocation
    int32_t clump_sub_spheres = compute_total_clump_sub_spheres(config);
    int32_t num_particles = load_particles_from_binary(state.particles, particle_path, config, clump_sub_spheres);
    std::cout << "Particle capacity: " << state.particles.capacity
              << " (file: " << num_particles << ", clump buffer: " << clump_sub_spheres << ")" << std::endl;

    // 4. Upload the immutable, runtime-sized material tables and derive mass
    // and inertia from each particle's required material id.
    const MaterialDatabase material_database = load_material_database(
        config.material_database.path, config.dem_damping_model);
    state.material_database = upload_material_database(material_database, config.gpu.device_id);
    int block_size = config.gpu.block_size;

    if (config.physics == SimulationPhysics::DEM) {
        std::cout << "Initializing particle mass and moment of inertia..." << std::endl;
        launch_init_particle_mass_kernel(
            state.particles.mass,
            state.particles.moment_of_inertia,
            state.particles.support_extent,
            state.particles.radius,
            state.particles.material_id,
            state.material_database.materials,
            state.material_database.dem_interactions,
            state.material_database.material_count,
            num_particles,
            block_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cohesion is read from the dense material-pair table.  Keep the legacy
    // particle array zeroed solely for binary/checkpoint layout stability.
    CUDA_CHECK(cudaMemset(state.particles.cohesive_strength, 0,
                          state.particles.capacity * sizeof(float)));

    // 5. Allocate and initialize contact history
    allocate_contact_history(state.contact_history, num_particles);

    // Initialize contact history offsets
    int32_t max_contacts = state.contact_history.max_contacts_per_particle;
    std::cout << "Initializing contact history offsets..." << std::endl;
    launch_init_contact_offsets_kernel(
        state.particles.contact_history_offset,
        max_contacts,
        num_particles,
        block_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Initialize walls
    initialize_walls(state.walls, config);
    // WallManager::initialize(state, config);  // Called separately in main.cu
    // NeighborSearch::initialize(state, config);  // Called separately in main.cu

    // 7. Calculate timestep
    state.dt = calculate_critical_timestep(config, state.particles,
                                           material_database, num_particles);
    state.current_time = config.time.start_time;
    state.current_step = 0;

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "========== Initialization Complete ==========" << std::endl;
    std::cout << "Particles: " << num_particles << std::endl;
    std::cout << "Walls: " << config.num_walls << std::endl;
    std::cout << "Timestep: " << state.dt << " s" << std::endl;
    std::cout << "=============================================" << std::endl;

    return state;
}

} // namespace cudadem

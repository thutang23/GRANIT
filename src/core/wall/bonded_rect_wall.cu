// =============================================================================
// Bonded Rectangular Prism Wall — CUDA Implementation
// =============================================================================
// Contact kernel:    per-particle, box closest-point → Hertz-Mindlin force
// Integration kernel: single-thread, Euler's equations + Rodrigues rotation
//
// Rotation matrix convention (rot[9], row-major, body-to-world):
//   v_world[i] = sum_j rot[i*3+j] * v_body[j]
//   v_body[i]  = sum_j rot[j*3+i] * v_world[j]   (transpose)
// =============================================================================

#include "../constant/command_line_interface.h"
#include "../constant/system_initialize.h"
#include "../particle_model/particle.h"
#include "bonded_rect_wall.h"
#include "wall_contact_force.cuh"
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cudadem {

// =============================================================================
// Static Device Pointers (module-level, initialized once)
// =============================================================================

static BondedRectWallState* d_bonded_states    = nullptr;
static BondedRectReaction*  d_bonded_reactions  = nullptr;
static int32_t              s_num_bonded_walls  = 0;
static bool                 s_bonded_initialized = false;

// =============================================================================
// BoxContactResult — output of compute_box_contact
// =============================================================================

struct BoxContactResult {
    float delta_n;                            // Penetration depth (>0 = contact)
    float normal_x, normal_y, normal_z;       // World frame, points FROM wall surface INTO domain (unit)
    float dist;                               // Sphere-center to contact-point distance
    float contact_px, contact_py, contact_pz; // Contact point (world frame)
    float wall_vx, wall_vy, wall_vz;          // Wall velocity at contact point (world frame)
};

// =============================================================================
// Device function: box closest-point sphere contact test
// =============================================================================
// Inputs:
//   sphere center (world frame), sphere radius
//   BondedRectWallState (contains pos, rot, half-extents, vel, omega_body)
// Returns:
//   BoxContactResult with delta_n set to <=0 if no contact
//
// Normal convention (matches plain_wall.cu):
//   - Normal points FROM wall surface INTO the domain (away from wall)
//   - For bottom face contact: normal points UP (+Z in body frame)
//   - For top face contact: normal points DOWN (-Z in body frame)
//   - This ensures force on particle pushes it AWAY from wall
// =============================================================================

__device__ __forceinline__
BoxContactResult compute_box_contact(
    double sphere_x, double sphere_y, double sphere_z,
    float  R_sphere,
    const BondedRectWallState& wall
) {
    BoxContactResult res;
    res.delta_n = -1.0f;

    // Vector from wall CoM to sphere center (world frame)
    float dx = (float)(sphere_x - wall.pos_x);
    float dy = (float)(sphere_y - wall.pos_y);
    float dz = (float)(sphere_z - wall.pos_z);

    // Transform to body frame: p_local = R^T * d
    // R is row-major body-to-world; R^T[i][j] = R[j][i] = rot[j*3+i]
    float px = wall.rot[0]*dx + wall.rot[3]*dy + wall.rot[6]*dz;
    float py = wall.rot[1]*dx + wall.rot[4]*dy + wall.rot[7]*dz;
    float pz = wall.rot[2]*dx + wall.rot[5]*dy + wall.rot[8]*dz;

    // Closest point on box surface in body frame (clamp to interior, then project)
    float cx = fmaxf(-wall.half_x, fminf(px, wall.half_x));
    float cy = fmaxf(-wall.half_y, fminf(py, wall.half_y));
    float cz = fmaxf(-wall.half_z, fminf(pz, wall.half_z));

    // Gap: from closest-box-point to sphere center (body frame)
    float gx = px - cx;
    float gy = py - cy;
    float gz = pz - cz;
    float dist_sq = gx*gx + gy*gy + gz*gz;

    float normal_lx, normal_ly, normal_lz, dist;

    if (dist_sq > 1e-12f) {
        // Sphere center is OUTSIDE the box
        dist = sqrtf(dist_sq);
        float inv_dist = 1.0f / dist;
        // Normal points FROM box surface TOWARD sphere center (outward from wall)
        // This matches plain_wall convention: normal points into the domain
        // FIX: Negate the gap direction so normal points AWAY from wall surface
        normal_lx = -gx * inv_dist;
        normal_ly = -gy * inv_dist;
        normal_lz = -gz * inv_dist;
    } else {
        // Sphere center is INSIDE the box — find nearest face
        // Normal should point outward from wall surface (toward sphere center direction)
        float dx_face = wall.half_x - fabsf(px);
        float dy_face = wall.half_y - fabsf(py);
        float dz_face = wall.half_z - fabsf(pz);
        if (dx_face <= dy_face && dx_face <= dz_face) {
            // Nearest face is X face; normal points toward sphere center
            // If px > 0, sphere is on +X side, normal should point -X (into wall)
            // But we want normal pointing AWAY from wall surface, so flip
            normal_lx = (px >= 0.0f) ? -1.0f : 1.0f;
            normal_ly = 0.0f;
            normal_lz = 0.0f;
            dist = -dx_face;   // negative: inside
        } else if (dy_face <= dz_face) {
            normal_lx = 0.0f;
            normal_ly = (py >= 0.0f) ? -1.0f : 1.0f;
            normal_lz = 0.0f;
            dist = -dy_face;
        } else {
            normal_lx = 0.0f;
            normal_ly = 0.0f;
            normal_lz = (pz >= 0.0f) ? -1.0f : 1.0f;
            dist = -dz_face;
        }
    }

    res.delta_n = R_sphere - dist;
    if (res.delta_n <= 0.0f) return res;   // no contact

    // Transform contact normal to world frame: n_world = R * n_local
    res.normal_x = wall.rot[0]*normal_lx + wall.rot[1]*normal_ly + wall.rot[2]*normal_lz;
    res.normal_y = wall.rot[3]*normal_lx + wall.rot[4]*normal_ly + wall.rot[5]*normal_lz;
    res.normal_z = wall.rot[6]*normal_lx + wall.rot[7]*normal_ly + wall.rot[8]*normal_lz;
    res.dist = (dist > 0.0f) ? dist : 0.0f;

    // Contact point in world frame: pos_wall + R * c_local
    res.contact_px = (float)wall.pos_x + wall.rot[0]*cx + wall.rot[1]*cy + wall.rot[2]*cz;
    res.contact_py = (float)wall.pos_y + wall.rot[3]*cx + wall.rot[4]*cy + wall.rot[5]*cz;
    res.contact_pz = (float)wall.pos_z + wall.rot[6]*cx + wall.rot[7]*cy + wall.rot[8]*cz;

    // Wall velocity at contact point: v_com + R*(omega_body × c_local)
    // Cross product in body frame: omega_body × c_local
    float cross_x = wall.omega_y * cz  - wall.omega_z * cy;
    float cross_y = wall.omega_z * cx  - wall.omega_x * cz;
    float cross_z = wall.omega_x * cy  - wall.omega_y * cx;
    // Rotate to world frame
    float v_rot_x = wall.rot[0]*cross_x + wall.rot[1]*cross_y + wall.rot[2]*cross_z;
    float v_rot_y = wall.rot[3]*cross_x + wall.rot[4]*cross_y + wall.rot[5]*cross_z;
    float v_rot_z = wall.rot[6]*cross_x + wall.rot[7]*cross_y + wall.rot[8]*cross_z;
    res.wall_vx = (float)wall.vel_x + v_rot_x;
    res.wall_vy = (float)wall.vel_y + v_rot_y;
    res.wall_vz = (float)wall.vel_z + v_rot_z;

    return res;
}

// =============================================================================
// Contact Force Kernel
// =============================================================================
// One thread per particle. Loops over all enabled bonded rect walls.
// Uses calculate_wall_force_impl from wall_contact_force.cuh.
// Reaction forces accumulated into d_bonded_reactions via double atomicAdd.
// =============================================================================

__global__ void compute_bonded_rect_forces_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const double* __restrict__ omega_x,
    const double* __restrict__ omega_y,
    const double* __restrict__ omega_z,
    const float*  __restrict__ radius,
    const float*  __restrict__ mass,
    const float*  __restrict__ moment_of_inertia,
    const int32_t* __restrict__ material_id,
    const MaterialRecord* __restrict__ materials,
    const DemInteractionProperties* __restrict__ interactions,
    uint32_t material_count,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ force_z,
    double* __restrict__ torque_x,
    double* __restrict__ torque_y,
    double* __restrict__ torque_z,
    WallContactHistory* __restrict__ wall_history,
    const int32_t* __restrict__ wall_history_offset,
    int32_t* __restrict__ num_wall_contacts,
    int32_t max_contacts_per_particle,
    const BondedRectWallState* __restrict__ wall_states,
    int32_t num_walls,
    BondedRectReaction* reactions,
    bool use_visco,
    float dt,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    double pi_x = pos_x[idx], pi_y = pos_y[idx], pi_z = pos_z[idx];
    double vi_x = vel_x[idx], vi_y = vel_y[idx], vi_z = vel_z[idx];
    double wi_x = omega_x[idx], wi_y = omega_y[idx], wi_z = omega_z[idx];
    float  R_i = radius[idx], m_i = mass[idx], I_i = moment_of_inertia[idx];
    const int32_t particle_material_id = material_id[idx];

    int32_t hist_offset  = wall_history_offset[idx];
    WallContactHistory* my_history    = wall_history + hist_offset;
    int32_t my_num_contacts = num_wall_contacts[idx];

    double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
    double tx_acc = 0.0, ty_acc = 0.0, tz_acc = 0.0;
    int32_t new_contact_count = 0;

    for (int32_t w = 0; w < num_walls; ++w) {
        const BondedRectWallState& wall = wall_states[w];
        if (!wall.enabled) continue;

        float E_star, G_star;
        DemInteractionProperties pair{};
        if (!resolve_wall_material_pair(particle_material_id, wall.material_id,
                                        materials, interactions, material_count,
                                        E_star, G_star, pair)) continue;

        BoxContactResult cr = compute_box_contact(pi_x, pi_y, pi_z, R_i, wall);

        // Clear inactive history slot when no contact
        if (cr.delta_n <= 0.0f) {
            WallForceResult cohesion;
            add_wall_cohesion(cohesion, cr.delta_n,
                              cr.normal_x, cr.normal_y, cr.normal_z, R_i, pair);
            fx_acc += cohesion.force_x;
            fy_acc += cohesion.force_y;
            fz_acc += cohesion.force_z;
            for (int32_t k = 0; k < my_num_contacts; ++k) {
                if (my_history[k].wall_id == wall.id) {
                    my_history[k].is_active = false;
                    break;
                }
            }
            continue;
        }

        // Find or create contact history slot
        int slot = -1;
        float H_t_x = 0.f, H_t_y = 0.f, H_t_z = 0.f;
        float H_r_x = 0.f, H_r_y = 0.f, H_r_z = 0.f;
        float H_tw_x = 0.f, H_tw_y = 0.f, H_tw_z = 0.f;
        for (int32_t k = 0; k < my_num_contacts; ++k) {
            if (my_history[k].wall_id == wall.id && my_history[k].is_active) {
                slot = k;
                H_t_x  = my_history[k].delta_t_x;  H_t_y  = my_history[k].delta_t_y;
                H_t_z  = my_history[k].delta_t_z;
                H_r_x  = my_history[k].delta_r_x;  H_r_y  = my_history[k].delta_r_y;
                H_r_z  = my_history[k].delta_r_z;
                H_tw_x = my_history[k].delta_tw_x; H_tw_y = my_history[k].delta_tw_y;
                H_tw_z = my_history[k].delta_tw_z;
                break;
            }
        }

        // Ap = distance from sphere center to contact point
        float Ap = (cr.dist > 1e-10f) ? cr.dist : R_i;

        WallForceResult result = calculate_wall_force_impl<FullFrictionPolicy>(
            cr.delta_n, Ap,
            vi_x, vi_y, vi_z,
            wi_x, wi_y, wi_z,
            R_i, m_i, I_i,
            cr.normal_x, cr.normal_y, cr.normal_z,
            (double)cr.wall_vx, (double)cr.wall_vy, (double)cr.wall_vz,
            H_t_x, H_t_y, H_t_z,
            H_r_x, H_r_y, H_r_z,
            H_tw_x, H_tw_y, H_tw_z,
            E_star, G_star, pair.restitution,
            pair.friction_static, pair.friction_rolling, pair.friction_twisting, pair.shape_parameter,
            use_visco, pair.viscous_coefficient, dt
        );
        add_wall_cohesion(result, cr.delta_n,
                          cr.normal_x, cr.normal_y, cr.normal_z, R_i, pair);

        fx_acc += result.force_x;
        fy_acc += result.force_y;
        fz_acc += result.force_z;
        tx_acc += result.torque_x;
        ty_acc += result.torque_y;
        tz_acc += result.torque_z;

        // Newton's 3rd law reaction on wall
        // Torque: r × F_reaction, where r = contact_pt - wall_CoM (world frame)
        double rx = (double)cr.contact_px - wall.pos_x;
        double ry = (double)cr.contact_py - wall.pos_y;
        double rz = (double)cr.contact_pz - wall.pos_z;
        double rfx = -(double)result.force_x;
        double rfy = -(double)result.force_y;
        double rfz = -(double)result.force_z;
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600)
        atomicAdd(&reactions[w].force_x, rfx);
        atomicAdd(&reactions[w].force_y, rfy);
        atomicAdd(&reactions[w].force_z, rfz);
        atomicAdd(&reactions[w].torque_x, ry*rfz - rz*rfy);
        atomicAdd(&reactions[w].torque_y, rz*rfx - rx*rfz);
        atomicAdd(&reactions[w].torque_z, rx*rfy - ry*rfx);
#else
        // Requires sm_60+ for double atomicAdd; this target enforces sm_60+
#error "BondedRectWall requires CUDA compute capability >= 6.0 for double atomicAdd"
#endif

        // Write contact history
        if (slot < 0 && (my_num_contacts + new_contact_count) < max_contacts_per_particle) {
            slot = my_num_contacts + new_contact_count;
            new_contact_count++;
        }
        if (slot >= 0 && slot < max_contacts_per_particle) {
            my_history[slot].wall_id    = wall.id;
            my_history[slot].delta_t_x  = result.H_t_x;
            my_history[slot].delta_t_y  = result.H_t_y;
            my_history[slot].delta_t_z  = result.H_t_z;
            my_history[slot].delta_r_x  = result.H_r_x;
            my_history[slot].delta_r_y  = result.H_r_y;
            my_history[slot].delta_r_z  = result.H_r_z;
            my_history[slot].delta_tw_x = result.H_tw_x;
            my_history[slot].delta_tw_y = result.H_tw_y;
            my_history[slot].delta_tw_z = result.H_tw_z;
            my_history[slot].is_active   = true;
            my_history[slot].is_slipping = result.is_slipping;
            my_history[slot].is_rolling  = result.is_rolling;
            my_history[slot].is_twisting = result.is_twisting;
        }
    }

    atomic_add_force( force_x,  force_y,  force_z,  fx_acc, fy_acc, fz_acc, idx);
    atomic_add_torque(torque_x, torque_y, torque_z, tx_acc, ty_acc, tz_acc, idx);
    num_wall_contacts[idx] = my_num_contacts + new_contact_count;
}

// =============================================================================
// Integration Kernel
// =============================================================================
// One thread per wall (gridDim=1, blockDim=MAX_BONDED_RECT_WALLS).
// Reads reaction forces/torques, updates 6-DOF state via:
//   - Euler's equations (body frame angular acceleration)
//   - Rodrigues incremental rotation (exact, preserves orthogonality)
//   - Symplectic Euler for position and velocity
// =============================================================================

__global__ void integrate_bonded_rect_kernel(
    BondedRectWallState* __restrict__ states,
    const BondedRectReaction* __restrict__ reactions,
    int32_t num_walls,
    float dt,
    float gx, float gy, float gz  // gravity (world frame, m/s²)
) {
    int w = (int)threadIdx.x;
    if (w >= num_walls) return;

    BondedRectWallState& s  = states[w];
    const BondedRectReaction& r = reactions[w];
    if (!s.enabled) return;

    // --- Linear dynamics (world frame) ---
    double ax = r.force_x / (double)s.mass + (double)gx;
    double ay = r.force_y / (double)s.mass + (double)gy;
    double az = r.force_z / (double)s.mass + (double)gz;

    s.vel_x += ax * dt;
    s.vel_y += ay * dt;
    s.vel_z += az * dt;
    s.pos_x += s.vel_x * dt;
    s.pos_y += s.vel_y * dt;
    s.pos_z += s.vel_z * dt;

    // --- Angular dynamics (body frame) ---
    // Transform reaction torque to body frame: tau_body = R^T * tau_world
    // R^T[i][j] = rot[j*3+i]
    float tau_bx = s.rot[0]*(float)r.torque_x + s.rot[3]*(float)r.torque_y + s.rot[6]*(float)r.torque_z;
    float tau_by = s.rot[1]*(float)r.torque_x + s.rot[4]*(float)r.torque_y + s.rot[7]*(float)r.torque_z;
    float tau_bz = s.rot[2]*(float)r.torque_x + s.rot[5]*(float)r.torque_y + s.rot[8]*(float)r.torque_z;

    // Euler's equations: I * alpha = tau - omega × (I * omega)
    float alpha_x = (tau_bx - (s.inertia_z - s.inertia_y)*s.omega_y*s.omega_z) / s.inertia_x;
    float alpha_y = (tau_by - (s.inertia_x - s.inertia_z)*s.omega_z*s.omega_x) / s.inertia_y;
    float alpha_z = (tau_bz - (s.inertia_y - s.inertia_x)*s.omega_x*s.omega_y) / s.inertia_z;

    s.omega_x += alpha_x * dt;
    s.omega_y += alpha_y * dt;
    s.omega_z += alpha_z * dt;

    // --- Rotation matrix update via Rodrigues ---
    float omega_mag = sqrtf(s.omega_x*s.omega_x + s.omega_y*s.omega_y + s.omega_z*s.omega_z);
    float angle = omega_mag * dt;

    if (angle > 1e-12f) {
        float inv_omega = 1.0f / omega_mag;
        float ax_r = s.omega_x * inv_omega;   // unit rotation axis (body frame)
        float ay_r = s.omega_y * inv_omega;
        float az_r = s.omega_z * inv_omega;

        float c = cosf(angle);
        float sv = sinf(angle);
        float t = 1.0f - c;

        // Rodrigues: dR = t*(a⊗a) + c*I + s*[a]×
        // Row-major, body-frame incremental rotation
        float dR[9];
        dR[0] = t*ax_r*ax_r + c;          dR[1] = t*ax_r*ay_r - sv*az_r; dR[2] = t*ax_r*az_r + sv*ay_r;
        dR[3] = t*ay_r*ax_r + sv*az_r;    dR[4] = t*ay_r*ay_r + c;       dR[5] = t*ay_r*az_r - sv*ax_r;
        dR[6] = t*az_r*ax_r - sv*ay_r;    dR[7] = t*az_r*ay_r + sv*ax_r; dR[8] = t*az_r*az_r + c;

        // R_new = R_old * dR  (apply body-frame rotation)
        float new_rot[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) sum += s.rot[i*3+k] * dR[k*3+j];
                new_rot[i*3+j] = sum;
            }
        }
        for (int i = 0; i < 9; ++i) s.rot[i] = new_rot[i];
    }
}

// =============================================================================
// Host Helper: ZYX Euler angles (degrees) → 3×3 rotation matrix (row-major)
// Convention: R = R_z(yaw) * R_y(pitch) * R_x(roll), body-to-world
// =============================================================================

static void euler_zyx_to_rot(float roll_deg, float pitch_deg, float yaw_deg, float rot[9]) {
    const float deg2rad = 3.14159265358979f / 180.0f;
    float cr = cosf(roll_deg  * deg2rad), sr = sinf(roll_deg  * deg2rad);
    float cp = cosf(pitch_deg * deg2rad), sp = sinf(pitch_deg * deg2rad);
    float cy = cosf(yaw_deg   * deg2rad), sy = sinf(yaw_deg   * deg2rad);

    // R = Rz * Ry * Rx (row-major)
    rot[0] =  cy*cp;   rot[1] = cy*sp*sr - sy*cr;  rot[2] = cy*sp*cr + sy*sr;
    rot[3] =  sy*cp;   rot[4] = sy*sp*sr + cy*cr;  rot[5] = sy*sp*cr - cy*sr;
    rot[6] = -sp;      rot[7] = cp*sr;              rot[8] = cp*cr;
}

// =============================================================================
// Host Functions
// =============================================================================

bool bonded_rect_wall_is_initialized() {
    return s_bonded_initialized;
}

int32_t bonded_rect_wall_get_states(
    BondedRectWallState* host_states,
    BondedRectReaction* host_reactions
) {
    if (!s_bonded_initialized || s_num_bonded_walls == 0) return 0;
    
    if (host_states) {
        CUDA_CHECK(cudaMemcpy(host_states, d_bonded_states,
                   s_num_bonded_walls * sizeof(BondedRectWallState),
                   cudaMemcpyDeviceToHost));
    }
    if (host_reactions) {
        CUDA_CHECK(cudaMemcpy(host_reactions, d_bonded_reactions,
                   s_num_bonded_walls * sizeof(BondedRectReaction),
                   cudaMemcpyDeviceToHost));
    }
    return s_num_bonded_walls;
}

void bonded_rect_wall_restore_states(
    const BondedRectWallState* host_states,
    const BondedRectReaction* host_reactions,
    int32_t count
) {
    if (count != bonded_rect_wall_get_states(nullptr, nullptr) ||
        count < 0 || count > MAX_BONDED_RECT_WALLS ||
        (count && (!host_states || !host_reactions)))
        throw std::runtime_error("Bonded wall checkpoint count does not match initialized walls");
    if (!count) return;
    std::vector<BondedRectWallState> configured(count);
    bonded_rect_wall_get_states(configured.data(), nullptr);
    for (int32_t i = 0; i < count; ++i) {
        const auto& a = configured[i];
        const auto& b = host_states[i];
        if (a.id != b.id || a.material_id != b.material_id || a.enabled != b.enabled ||
            a.half_x != b.half_x || a.half_y != b.half_y || a.half_z != b.half_z ||
            a.mass != b.mass || a.inertia_x != b.inertia_x ||
            a.inertia_y != b.inertia_y || a.inertia_z != b.inertia_z)
            throw std::runtime_error("Bonded wall checkpoint geometry/material/inertia mismatch");
        for (double value : {b.pos_x, b.pos_y, b.pos_z, b.vel_x, b.vel_y, b.vel_z,
                             double(b.omega_x), double(b.omega_y), double(b.omega_z)})
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite bonded wall checkpoint state");
        for (float value : b.rot)
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite bonded wall checkpoint rotation");
        const auto& r = host_reactions[i];
        for (double value : {r.force_x, r.force_y, r.force_z,
                             r.torque_x, r.torque_y, r.torque_z})
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite bonded wall checkpoint reaction");
    }
    CUDA_CHECK(cudaMemcpy(d_bonded_states, host_states,
                         count * sizeof(BondedRectWallState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bonded_reactions, host_reactions,
                         count * sizeof(BondedRectReaction), cudaMemcpyHostToDevice));
}

void bonded_rect_wall_initialize(
    const SimConfig& config,
    int32_t /*num_particles*/
) {
    // Collect BONDED_RECT walls from config
    std::vector<BondedRectWallState> host_states;
    for (int32_t i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];
        if (wc.type != WallType::BONDED_RECT || !wc.enabled) continue;

        if ((int)host_states.size() >= MAX_BONDED_RECT_WALLS) {
            std::cerr << "WARNING: More than " << MAX_BONDED_RECT_WALLS
                      << " bonded_rect walls — truncating." << std::endl;
            break;
        }

        BondedRectWallState st;
        st.id      = wc.id;
        st.material_id = wc.material_id;
        st.enabled = true;

        st.half_x = wc.half_extents[0];
        st.half_y = wc.half_extents[1];
        st.half_z = wc.half_extents[2];

        st.mass       = wc.bonded_mass;
        st.inertia_x  = wc.inertia_body[0];
        st.inertia_y  = wc.inertia_body[1];
        st.inertia_z  = wc.inertia_body[2];

        st.pos_x = wc.position.x;
        st.pos_y = wc.position.y;
        st.pos_z = wc.position.z;

        st.vel_x = wc.velocity.x;
        st.vel_y = wc.velocity.y;
        st.vel_z = wc.velocity.z;

        st.omega_x = (float)wc.initial_omega_body.x;
        st.omega_y = (float)wc.initial_omega_body.y;
        st.omega_z = (float)wc.initial_omega_body.z;

        // euler_zyx_deg: [Z-yaw, Y-pitch, X-roll] in degrees
        euler_zyx_to_rot(
            (float)wc.euler_zyx_deg.z,   // roll  (X)
            (float)wc.euler_zyx_deg.y,   // pitch (Y)
            (float)wc.euler_zyx_deg.x,   // yaw   (Z)
            st.rot
        );

        host_states.push_back(st);
    }

    s_num_bonded_walls = (int32_t)host_states.size();
    if (s_num_bonded_walls == 0) {
        std::cout << "BondedRectWall: no enabled walls found, skipping." << std::endl;
        return;
    }

    // Allocate device arrays
    cudaMalloc(&d_bonded_states,    s_num_bonded_walls * sizeof(BondedRectWallState));
    cudaMalloc(&d_bonded_reactions, s_num_bonded_walls * sizeof(BondedRectReaction));

    cudaMemcpy(d_bonded_states, host_states.data(),
               s_num_bonded_walls * sizeof(BondedRectWallState),
               cudaMemcpyHostToDevice);
    cudaMemset(d_bonded_reactions, 0,
               s_num_bonded_walls * sizeof(BondedRectReaction));

    s_bonded_initialized = true;
    std::cout << "BondedRectWall: initialized " << s_num_bonded_walls << " wall(s)." << std::endl;

    for (const auto& st : host_states) {
        std::cout << "  id=" << st.id
                  << " half_extents=[" << st.half_x << "," << st.half_y << "," << st.half_z
                  << "] mass=" << st.mass
                  << " bonded_rect_wall" << std::endl;
    }
}

void bonded_rect_wall_compute_forces(
    SimulationState& state,
    const SimConfig& config,
    float dt,
    float /*current_time*/,
    cudaStream_t stream
) {
    if (!s_bonded_initialized || s_num_bonded_walls == 0) return;

    // Zero reaction buffers
    cudaMemsetAsync(d_bonded_reactions, 0,
                    s_num_bonded_walls * sizeof(BondedRectReaction), stream);

    int32_t num_particles = state.particles.num_particles;
    if (num_particles == 0) return;

    int block_size = config.gpu.block_size;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    bool use_visco = config.dem_damping_model == DemDampingModel::VISCOUS;

    LAUNCH_KERNEL(compute_bonded_rect_forces_kernel, num_blocks, block_size, 0, stream,
        state.particles.pos_x,   state.particles.pos_y,   state.particles.pos_z,
        state.particles.vel_x,   state.particles.vel_y,   state.particles.vel_z,
        state.particles.omega_x, state.particles.omega_y, state.particles.omega_z,
        state.particles.radius,  state.particles.mass,    state.particles.moment_of_inertia,
        state.particles.material_id, state.material_database.materials,
        state.material_database.dem_interactions, state.material_database.material_count,
        state.particles.force_x, state.particles.force_y, state.particles.force_z,
        state.particles.torque_x,state.particles.torque_y,state.particles.torque_z,
        state.wall_contact_history.data,
        state.particles.wall_history_offset,
        state.particles.num_wall_contacts,
        state.wall_contact_history.max_contacts_per_particle,
        d_bonded_states,
        s_num_bonded_walls,
        d_bonded_reactions,
        use_visco, dt,
        num_particles
    );
}

void bonded_rect_wall_integrate(
    float dt,
    float gx, float gy, float gz,
    cudaStream_t stream
) {
    if (!s_bonded_initialized || s_num_bonded_walls == 0) return;

    // One thread per wall; at most MAX_BONDED_RECT_WALLS threads in a single block
    integrate_bonded_rect_kernel<<<1, MAX_BONDED_RECT_WALLS, 0, stream>>>(
        d_bonded_states, d_bonded_reactions,
        s_num_bonded_walls, dt, gx, gy, gz
    );
}

void bonded_rect_wall_shutdown() {
    if (d_bonded_states)    { cudaFree(d_bonded_states);    d_bonded_states    = nullptr; }
    if (d_bonded_reactions) { cudaFree(d_bonded_reactions); d_bonded_reactions = nullptr; }
    s_num_bonded_walls  = 0;
    s_bonded_initialized = false;
}

} // namespace cudadem

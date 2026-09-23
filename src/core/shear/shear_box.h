#ifndef GRANIT_SHEAR_BOX_H
#define GRANIT_SHEAR_BOX_H

#include "../constant/command_line_interface.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cudadem {

struct ParticleSoA;
struct SimulationState;
struct PeriodicBoundaryConfig;

// The shearing-periodic lattice is represented by a small, centered shift.
// full_shift = reduced_shift + cycle * Ly, with
// reduced_shift in [-Ly/2, Ly/2).  Keeping the phase centered bounds all
// contact/image transforms even in runs spanning many shear periods.
struct ShearPhase {
    double reduced_shift;
    double shear_velocity;
    int64_t cycle;

    __host__ __device__ ShearPhase()
        : reduced_shift(0.0), shear_velocity(0.0), cycle(0) {}
};

struct ShearImageTransform {
    double x;
    double y;
    double z;
    double velocity_y;
};

struct ShearDriftState {
    double x, y, z;
    double vx, vy, vz;
};

struct ShearDiagnostics {
    double peculiar_kinetic_energy;
    double hill_jacobi_invariant;

    ShearDiagnostics()
        : peculiar_kinetic_energy(0.0), hill_jacobi_invariant(0.0) {}
};

// Defined in a host-only C++ translation unit so long double is never lowered
// to device double by nvcc.
ShearPhase compute_shear_phase_checked(
    const ShearBoxConfig& shear, double Lx, double Ly, double time);

__host__ __device__ __forceinline__
ShearImageTransform shear_partner_image(
    int image_x, int image_y, int image_z,
    double box_length_x, double box_length_y, double box_length_z,
    const ShearPhase& phase) {
    ShearImageTransform out;
    out.x = static_cast<double>(image_x) * box_length_x;
    out.y = static_cast<double>(image_y) * box_length_y
          - static_cast<double>(image_x) * phase.reduced_shift;
    out.z = static_cast<double>(image_z) * box_length_z;
    out.velocity_y = -static_cast<double>(image_x) * phase.shear_velocity;
    return out;
}

// Exact flow of the homogeneous Hill equations for arbitrary q.  The series
// branch is shared by q==2 and nearby values, avoiding cancellation in
// (1-cos(kappa*h))/kappa^2.  For q>2 the same analytic continuation produces
// the physically hyperbolic map.
__host__ __device__ __forceinline__
ShearDriftState exact_hill_drift(
    ShearDriftState s, double omega, double q,
    double vertical_frequency, double h) {
    const double K = 2.0 * (2.0 - q) * omega * omega;
    const double h2 = h * h;
    const double kh2 = K * h2;
    double c, sn, g, r;
    if (fabs(kh2) < 1.0e-8) {
        const double h3 = h2 * h;
        const double h4 = h2 * h2;
        const double h5 = h4 * h;
        const double h6 = h3 * h3;
        const double k2 = K * K;
        c  = 1.0 - 0.5 * K * h2 + (k2 * h4) / 24.0;
        sn = h - (K * h3) / 6.0 + (k2 * h5) / 120.0;
        g  = 0.5 * h2 - (K * h4) / 24.0 + (k2 * h6) / 720.0;
        r  = h3 / 6.0 - (K * h5) / 120.0;
    } else if (K > 0.0) {
        const double kappa = sqrt(K);
        c = cos(kappa * h);
        sn = sin(kappa * h) / kappa;
        g = (1.0 - c) / K;
        r = (h - sn) / K;
    } else {
        const double growth = sqrt(-K);
        c = cosh(growth * h);
        sn = sinh(growth * h) / growth;
        g = (c - 1.0) / (-K);
        r = (sn - h) / (-K);
    }

    const double C = s.vy + 2.0 * omega * s.x;
    const double A = 2.0 * omega * C;
    const double integral_x = sn * s.x + g * s.vx + A * r;
    const double x1 = c * s.x + sn * s.vx + A * g;
    const double vx1 = -K * sn * s.x + c * s.vx + A * sn;
    const double y1 = s.y + C * h - 2.0 * omega * integral_x;
    const double vy1 = C - 2.0 * omega * x1;

    double z1, vz1;
    const double theta = vertical_frequency * h;
    if (vertical_frequency == 0.0) {
        z1 = s.z + h * s.vz;
        vz1 = s.vz;
    } else if (fabs(theta) < 1.0e-4) {
        const double theta2 = theta * theta;
        const double cv = 1.0 - 0.5 * theta2 + theta2 * theta2 / 24.0;
        const double sv_over_nu = h * (1.0 - theta2 / 6.0
                                      + theta2 * theta2 / 120.0);
        z1 = cv * s.z + sv_over_nu * s.vz;
        vz1 = -vertical_frequency * vertical_frequency * sv_over_nu * s.z
              + cv * s.vz;
    } else {
        const double cv = cos(theta);
        const double sv = sin(theta);
        z1 = cv * s.z + (sv / vertical_frequency) * s.vz;
        vz1 = -vertical_frequency * sv * s.z + cv * s.vz;
    }

    s.x = x1; s.y = y1; s.z = z1;
    s.vx = vx1; s.vy = vy1; s.vz = vz1;
    return s;
}

__host__ __device__ __forceinline__
void hill_acceleration(const ShearBoxConfig& shear,
                       double x, double z, double vx, double vy,
                       double& ax, double& ay, double& az) {
    const double omega = shear.omega;
    ax = 2.0 * omega * vy
       + 2.0 * shear.shear_parameter * omega * omega * x;
    ay = -2.0 * omega * vx;
    az = -shear.vertical_frequency * shear.vertical_frequency * z;
}

class ShearBoxManager {
public:
    static void initialize(SimulationState& state, const SimConfig& config,
                           cudaStream_t stream = 0);
    // Release only the current CUDA device's boundary workspace.
    static void shutdown();

    // Fail-closed checks performed before every drift.
    static ShearPhase validate_step(const SimulationState& state,
                                    const SimConfig& config,
                                    const PeriodicBoundaryConfig& pbc);

    // K_non-Hill(h/2) E_Hill(h), plus the unchanged rotational half-kick and
    // quaternion drift.  Retained force/torque arrays are cleared.
    static void drift_step1(SimulationState& state, const SimConfig& config,
                            double dt, int32_t num_particles,
                            cudaStream_t stream = 0);

    // Canonical half-open x/y wrap at the endpoint phase, with checked int64
    // winding updates and the radial-image velocity jump.
    static ShearPhase wrap(SimulationState& state, const SimConfig& config,
                           const PeriodicBoundaryConfig& pbc,
                           double evaluation_time, int32_t num_particles,
                           cudaStream_t stream = 0);

    // Closing non-Hill half kick; rotational Verlet step 2 is unchanged.
    static void kick_step2(SimulationState& state, const SimConfig& config,
                           double dt, int32_t num_particles,
                           cudaStream_t stream = 0);

    // Contact epochs preserve spring memory while distinguishing a missed
    // image/contact from an active one.  begin clears seen_this_step; the
    // contact pass marks matching rows; end deactivates rows not seen.
    static void begin_contact_epoch(SimulationState& state,
                                    int32_t num_particles,
                                    cudaStream_t stream = 0);
    static void end_contact_epoch(SimulationState& state,
                                  int32_t num_particles,
                                  cudaStream_t stream = 0);

    // Output-only force composition.  Source and destination may alias.
    static void compose_output_force(const SimulationState& state,
                                     const SimConfig& config,
                                     const double* source_x,
                                     const double* source_y,
                                     const double* source_z,
                                     double* destination_x,
                                     double* destination_y,
                                     double* destination_z,
                                     int32_t num_particles,
                                     cudaStream_t stream = 0);

    static ShearDiagnostics compute_diagnostics(const SimulationState& state,
                                                const SimConfig& config,
                                                int32_t num_particles,
                                                cudaStream_t stream = 0);
};

} // namespace cudadem

#endif

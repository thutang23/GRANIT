#ifndef CUDADEM_SPH_POROSITY_MATH_CUH
#define CUDADEM_SPH_POROSITY_MATH_CUH

#include <cuda_runtime.h>

#include <cmath>

namespace cudadem {

// Solid Tillotson EOS value and derivatives at fixed solid density / energy.
// Keeping this beside the P-alpha rate equation makes the latter testable and
// prevents it from silently drifting from the EOS used to close pressure.
struct SphTillotsonSolidDerivatives {
    double pressure;
    double dp_drho;
    double dp_de;
};

__host__ __device__ inline SphTillotsonSolidDerivatives
sph_tillotson_cold_derivatives(
    double rho, double rho0, double energy, double mu,
    double a, double b, double A, double B,
    double inv_omg, double inv_omg2,
    double domega_drho, double domega_de) {
    const double coefficient = a + b * inv_omg;
    const double pressure = coefficient * rho * energy + mu * (A + mu * B);
    return {
        pressure,
        coefficient * energy - b * rho * energy * inv_omg2 * domega_drho +
            (A + 2.0 * B * mu) / rho0,
        coefficient * rho - b * rho * energy * inv_omg2 * domega_de,
    };
}

__host__ __device__ inline SphTillotsonSolidDerivatives
sph_tillotson_hot_derivatives(
    double rho, double rho0, double energy, double mu, double nu, double eta2,
    double a, double b, double A, double alpha_eos, double beta_eos,
    double inv_omg, double inv_omg2,
    double domega_drho, double domega_de) {
    const double evaporation = exp(-beta_eos * nu);
    const double expansion = exp(-alpha_eos * nu * nu);
    const double thermal = b * rho * energy * inv_omg;
    const double cold_term = A * mu * evaporation;
    const double bracket = thermal + cold_term;
    const double dnu_drho = -1.0 / (rho0 * eta2);
    const double dthermal_drho = b * energy * inv_omg -
        b * rho * energy * inv_omg2 * domega_drho;
    const double dcold_drho = A * evaporation *
        (1.0 / rho0 - beta_eos * mu * dnu_drho);
    const double dexpansion_drho = expansion *
        (-2.0 * alpha_eos * nu * dnu_drho);
    return {
        a * rho * energy + bracket * expansion,
        a * energy + (dthermal_drho + dcold_drho) * expansion +
            bracket * dexpansion_drho,
        a * rho + (b * rho * inv_omg -
            b * rho * energy * inv_omg2 * domega_de) * expansion,
    };
}

__host__ __device__ inline SphTillotsonSolidDerivatives
sph_tillotson_solid_pressure_derivatives(
    double rho, double rho0, double energy,
    double a, double b, double A, double B,
    double alpha_eos, double beta_eos,
    double E0, double Eiv, double Ecv) {
    // The material parser requires positive finite rho0/E0 and ordered
    // vaporization energies.  Keep a finite zero result for defensive callers
    // rather than emitting an undefined divide in an already-invalid state.
    if (!(rho > 0.0) || !(rho0 > 0.0) || !(E0 > 0.0) ||
        !isfinite(rho) || !isfinite(rho0) || !isfinite(energy)) {
        return {0.0, 0.0, 0.0};
    }

    // Form small compression directly so the shared helper preserves the
    // EOS's FP64 hydrostatic signal near rho == rho0.
    const double mu = (rho - rho0) / rho0;
    const double eta = 1.0 + mu;
    const double nu = 1.0 / eta - 1.0;
    const double eta2 = eta * eta;
    const double omg = 1.0 + energy / (E0 * eta2);
    if (!(omg > 0.0) || !isfinite(omg)) return {0.0, 0.0, 0.0};

    const double inv_omg = 1.0 / omg;
    const double inv_omg2 = inv_omg * inv_omg;
    const double domega_drho = -2.0 * energy / (E0 * rho0 * eta2 * eta);
    const double domega_de = 1.0 / (E0 * eta2);

    if (eta < 0.8 && energy < Eiv) return {0.0, 0.0, 0.0};
    if (eta >= 1.0 || energy <= Eiv) {
        return sph_tillotson_cold_derivatives(
            rho, rho0, energy, mu, a, b, A, B,
            inv_omg, inv_omg2, domega_drho, domega_de);
    }
    if (energy >= Ecv) {
        return sph_tillotson_hot_derivatives(
            rho, rho0, energy, mu, nu, eta2, a, b, A, alpha_eos, beta_eos,
            inv_omg, inv_omg2, domega_drho, domega_de);
    }

    const SphTillotsonSolidDerivatives pc = sph_tillotson_cold_derivatives(
        rho, rho0, energy, mu, a, b, A, B,
        inv_omg, inv_omg2, domega_drho, domega_de);
    const SphTillotsonSolidDerivatives pe = sph_tillotson_hot_derivatives(
        rho, rho0, energy, mu, nu, eta2, a, b, A, alpha_eos, beta_eos,
        inv_omg, inv_omg2, domega_drho, domega_de);
    const double interval = Ecv - Eiv;
    if (!(interval > 0.0)) return pc;
    const double hot_weight = energy - Eiv;
    const double cold_weight = Ecv - energy;
    return {
        (pe.pressure * hot_weight + pc.pressure * cold_weight) / interval,
        (pe.dp_drho * hot_weight + pc.dp_drho * cold_weight) / interval,
        (pe.dp_de * hot_weight + pe.pressure + pc.dp_de * cold_weight - pc.pressure) / interval,
    };
}

// P-alpha closure for P = P_solid(alpha*rho, e) / alpha and alpha = alpha(P).
// `dalpha_dP` is the crush-curve derivative with respect to porous pressure.
// The returned dpdt is likewise the porous-pressure rate used by SPHSOL dt7.
struct SphPorosityRate {
    double dalpha_dt;
    double dpdt;
};

__host__ __device__ inline SphPorosityRate sph_palpha_rate(
    double rho, double alpha, double pressure,
    double rhodot, double dedt, double dalpha_dP,
    const SphTillotsonSolidDerivatives& solid) {
    const double numerator = dedt * solid.dp_de +
        rhodot * alpha * solid.dp_drho;
    const double denominator = alpha + dalpha_dP *
        (pressure - rho * solid.dp_drho);
    if (!(fabs(denominator) > 1.0e-12) || !isfinite(numerator) ||
        !isfinite(denominator)) {
        return {0.0, 0.0};
    }
    const double dpdt = numerator / denominator;
    const double dalpha_dt = dalpha_dP * dpdt;
    if (!isfinite(dpdt) || !isfinite(dalpha_dt)) return {0.0, 0.0};
    return {dalpha_dt, dpdt};
}

// The historical compacted state defines the reload threshold.  Below that
// pressure alpha stays fixed; unloading cannot recreate crushed pore volume.
__host__ __device__ inline double sph_palpha_crush_pressure(
    double alpha, double alpha0, double pe, double ps, double exponent) {
    if (!(alpha0 > 1.0) || alpha <= 1.0) return ps;
    const double fraction = fmin(1.0, fmax(0.0, (alpha - 1.0) / (alpha0 - 1.0)));
    return ps - (ps - pe) * pow(fraction, 1.0 / exponent);
}

__host__ __device__ inline double sph_palpha_commit(
    double alpha, double alpha0, double pressure, double pe, double ps,
    double exponent, double rate, double dt) {
    const double old = fmax(1.0, alpha);
    if (pressure >= ps) return 1.0;
    const double reload = sph_palpha_crush_pressure(old, alpha0, pe, ps, exponent);
    if (pressure < reload || pressure <= pe) return old;
    const double curve = 1.0 + (alpha0 - 1.0) * pow((ps-pressure)/(ps-pe), exponent);
    const double trial = old + fmin(0.0, rate) * fmax(0.0, dt);
    return fmax(1.0, fmin(old, fmax(trial, curve)));
}

} // namespace cudadem

#endif // CUDADEM_SPH_POROSITY_MATH_CUH

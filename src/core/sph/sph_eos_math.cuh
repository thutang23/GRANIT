#ifndef CUDADEM_SPH_EOS_MATH_CUH
#define CUDADEM_SPH_EOS_MATH_CUH

#include "src/core/material/material_database.h"
#include "sph_porosity_math.cuh"

namespace cudadem {

__host__ __device__ inline double sph_bulk_modulus(const MaterialRecord& material) {
    return material.sph_eos.bulk_modulus > 0.0f
        ? material.sph_eos.bulk_modulus : material.sph_tillotson.A;
}

// Optional pressure-free release is confined to parser-validated nonporous
// fluids. Keep it in the shared EOS closure, including stage elastic overrides.
__host__ __device__ inline bool sph_liquid_release(
    const MaterialRecord& material, double rho, double rho0, double energy,
    double pressure, bool force_elastic = false) {
    return material.sph_tillotson.negative_pressure == SphNegativePressure::ZERO &&
        (pressure < 0.0 || (!force_elastic &&
         material.sph_eos.model == SphEosModel::TILLOTSON &&
         rho / rho0 < 0.8 && energy < material.sph_tillotson.Eiv));
}

__host__ __device__ inline SphTillotsonSolidDerivatives sph_solid_eos_derivatives(
    const MaterialRecord& material, double rho, double rho0, double energy,
    bool force_elastic = false) {
    const auto& eos = material.sph_tillotson;
    if (force_elastic || material.sph_eos.model == SphEosModel::ELASTIC) {
        const double K = sph_bulk_modulus(material);
        const double pressure = K * (rho - rho0) / rho0;
        if (sph_liquid_release(material, rho, rho0, energy, pressure, true))
            return {0.0, 0.0, 0.0};
        return {pressure, K / rho0, 0.0};
    }
    if (material.sph_eos.model == SphEosModel::SIMPLIFIED_TILLOTSON) {
        const double c = material.sph_eos.simple_c;
        return {c * rho * energy + eos.A * (rho - rho0) / rho0,
                c * energy + eos.A / rho0, c * rho};
    }
    const auto solid = sph_tillotson_solid_pressure_derivatives(rho, rho0, energy,
        eos.a, eos.b, eos.A, eos.B, eos.alpha, eos.beta, eos.E0, eos.Eiv, eos.Ecv);
    if (sph_liquid_release(material, rho, rho0, energy, solid.pressure))
        return {0.0, 0.0, 0.0};
    return solid;
}

struct SphEosResult {
    double pressure;
    double c_sound;
};

// The Tillotson branch preserves GRANIT's existing solid-pressure sound speed.
// Alternative EOS branches use their own thermodynamic pressure derivatives.
__host__ __device__ inline SphEosResult sph_eos_state(
    const MaterialRecord& material, double density, double energy,
    double damage, double distension, bool force_elastic = false) {
    const SphTillotsonProperties& eos = material.sph_tillotson;
    const SphPorosityProperties& porosity = material.sph_porosity;
    const double alpha = porosity.model == SphPorosityModel::P_ALPHA ? distension : 1.0;
    const double rho = density * alpha;
    const double rho0 = static_cast<double>(eos.reference_density) *
        (porosity.model == SphPorosityModel::P_ALPHA ? porosity.alpha_initial : 1.0);
    const double e = energy;
    const double D = material.sph_damage.model == SphDamageModel::GRADY_KIPP ? damage : 0.0;

    if (force_elastic || material.sph_eos.model != SphEosModel::TILLOTSON) {
        if ((force_elastic || material.sph_eos.model == SphEosModel::ELASTIC) &&
            sph_liquid_release(material, rho, rho0, e,
                sph_bulk_modulus(material) * (rho-rho0)/rho0, true))
            return {0.0, 1.0};
        const auto solid = sph_solid_eos_derivatives(material, rho, rho0, e, force_elastic);
        const double p = solid.pressure < 0.0 ? solid.pressure * (1.0 - D) : solid.pressure;
        // Retain SPHSOL's closures for the two optional simple EOS models.
        // P-alpha still uses the analytical derivatives above for alpha_dot.
        const double c2 = force_elastic || material.sph_eos.model == SphEosModel::ELASTIC
            ? sph_bulk_modulus(material) / rho
            : static_cast<double>(eos.A) / rho;
        return {p / alpha, sqrt(fmax(c2, sph_bulk_modulus(material) / rho0))};
    }

    const double a = eos.a, b = eos.b, A = eos.A, B = eos.B;
    const double alpha_eos = eos.alpha, beta_eos = eos.beta;
    const double E0 = eos.E0, Eiv = eos.Eiv, Ecv = eos.Ecv;

    // Form the small compression directly.  rho/rho0 - 1 loses most of the
    // hydrostatic signal even when rho itself has already been promoted.
    const double mu = (rho - rho0) / rho0;
    const double eta = 1.0 + mu;
    const double nu = 1.0 / eta - 1.0;
    const double omg = e / (E0 * eta * eta) + 1.0;

    // Keep the value used to close the EOS identical to the value whose full
    // derivatives drive P-alpha rates.
    double p = sph_tillotson_solid_pressure_derivatives(
        rho, rho0, e, a, b, A, B, alpha_eos, beta_eos, E0, Eiv, Ecv).pressure;

    if (sph_liquid_release(material, rho, rho0, e, p))
        return {0.0, 1.0};

    // Apply damage to negative pressure
    if (p < 0.0) {
        p = p * (1.0 - D);
    }

    // Porous pressure
    const double p_final = p / alpha;


    // Full Tillotson sound speed (matches SPHSOL physics.cc:48-84)
    // Three branches with full c_pressure2 analytical derivatives,
    // plus K/rho0 floor (physics.cc:128)
    double c_pressure2 = 0.0;

    // Density limit for cold expanded state
    if (eta < 0.8 && e < Eiv) {
        c_pressure2 = 0.0;
    } else if (eta >= 1.0 || e <= Eiv) {
        // Cold compressed / cold expanded branch (physics.cc:48-49)
        // These are solid-EOS derivatives: p and rho are both solid values.
        // Dividing p by alpha before this calculation spuriously softened
        // porous material a second time.
        c_pressure2 = (1.0 + a + b / omg) * p / rho
                    + (2.0 * e - p / rho) * b * (omg - 1.0) / (omg * omg)
                    + (A + B * (eta * eta - 1.0)) / rho;
    } else if (e >= Ecv) {
        // Hot expanded branch (physics.cc:60-63)
        double exp_alpha_nu2 = exp(-alpha_eos * nu * nu);
        c_pressure2 = (1.0 + a + b / omg * exp_alpha_nu2) * p / rho
                    + exp_alpha_nu2 * (
                        ((2.0 * e - p / rho) / (E0 * rho)
                      + 2.0 * alpha_eos * nu * omg / rho0) * b * rho * e / (omg * omg * eta * eta)
                      + (1.0 + mu / (eta * eta) * (beta_eos + 2.0 * alpha_eos * nu - eta))
                          * A / rho0 * exp(-beta_eos * nu)
                    );
    } else {
        // Mixed branch: Eiv < e < Ecv (physics.cc:79-85)
        // Interpolate c² between cold (cc2) and hot (ce2) branches
        double cc2 = (1.0 + a + b / omg) * p / rho
                   + (2.0 * e - p / rho) * b * (omg - 1.0) / (omg * omg)
                   + (A + B * (eta * eta - 1.0)) / rho;

        double exp_alpha_nu2 = exp(-alpha_eos * nu * nu);
        double ce2 = (1.0 + a + b / omg * exp_alpha_nu2) * p / rho
                  + exp_alpha_nu2 * (
                      ((2.0 * e - p / rho) / (E0 * rho)
                    + 2.0 * alpha_eos * nu * omg / rho0) * b * rho * e / (omg * omg * eta * eta)
                    + (1.0 + mu / (eta * eta) * (beta_eos + 2.0 * alpha_eos * nu - eta))
                        * A / rho0 * exp(-beta_eos * nu)
                  );

        c_pressure2 = (ce2 * (e - Eiv) + cc2 * (Ecv - e)) / (Ecv - Eiv);
    }

    // Sound speed floor: c² >= K/rho0.  Legacy materials use K = A.
    const double c_floor = sph_bulk_modulus(material) / rho0;
    c_pressure2 = fmax(c_pressure2, c_floor);

    return {p_final, sqrt(c_pressure2)};
}

} // namespace cudadem

#endif // CUDADEM_SPH_EOS_MATH_CUH

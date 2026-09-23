#ifndef CUDADEM_SPH_DAMAGE_MATH_CUH
#define CUDADEM_SPH_DAMAGE_MATH_CUH

#include <cmath>

namespace cudadem {

struct SphDamageRate {
    float crack_strain;
    float qdot;
    float qcap;
    float nact;
};

struct SphFlawInterval {
    float strain_min;
    float strain_max;
    float m0;
};

// Build the compact Weibull interval in double precision.  Asteroid-scale
// k*V products routinely exceed FLT_MAX (for the basalt pilot, ~1e47), even
// though the resulting strain thresholds are ordinary positive floats.
__host__ __device__ inline SphFlawInterval sph_damage_flaw_interval(
    double k, float m, double body_volume, int ntot,
    double first_order, double last_order, float fallback_min_strain) {
    const float fallback = isfinite(fallback_min_strain) && fallback_min_strain > 0.0f
        ? fallback_min_strain : 1.0e-6f;
    SphFlawInterval out{fallback, fallback, 1.0f};
    if (!(k > 0.0f) || !(m > 0.0f) || !(body_volume > 0.0) || ntot <= 0 ||
        !isfinite(k) || !isfinite(m) || !isfinite(body_volume) ||
        !(first_order > 0.0) || !(last_order > 0.0) ||
        !isfinite(first_order) || !isfinite(last_order)) return out;
    // Form logarithms separately: k*V may overflow even binary64 for an
    // otherwise valid material, while the resulting strain remains ordinary.
    const double log_scale = -log(k) - log(body_volume);
    const double raw_min = exp((log_scale + log(fmin(first_order, last_order))) / m);
    const double raw_max = fmin(
        exp((log_scale + log(fmax(first_order, last_order))) / m),
        raw_min * static_cast<double>(ntot));
    if (!(raw_min > 0.0) || !isfinite(raw_min)) return out;
    out.strain_min = static_cast<float>(raw_min);
    if (!(out.strain_min > 0.0f) || !isfinite(out.strain_min)) out.strain_min = fallback;
    out.strain_max = raw_max >= static_cast<double>(out.strain_min) && isfinite(raw_max)
        ? static_cast<float>(raw_max) : out.strain_min;
    if (!(out.strain_max >= out.strain_min) || !isfinite(out.strain_max))
        out.strain_max = out.strain_min;
    if (out.strain_max > out.strain_min && ntot > 1) {
        const double fitted = log(static_cast<double>(ntot)) /
                              log(static_cast<double>(out.strain_max) /
                                  static_cast<double>(out.strain_min));
        if (isfinite(fitted)) out.m0 = fmaxf(static_cast<float>(fitted), 1.0f);
    }
    return out;
}

// Scale before forming invariants.  This symmetric 3x3 eigensolver remains
// finite for diagonal, hydrostatic, and degenerate matrices.
__host__ __device__ inline float sph_damage_lambda_max(
    float a00, float a01, float a02, float a11, float a12, float a22) {
    const float scale = fmaxf(fmaxf(fabsf(a00), fabsf(a01)),
                        fmaxf(fmaxf(fabsf(a02), fabsf(a11)), fmaxf(fabsf(a12), fabsf(a22))));
    if (!(scale > 0.0f) || !isfinite(scale)) return 0.0f;
    if (a01 == 0.0f && a02 == 0.0f && a12 == 0.0f) {
        const float value = fmaxf(a00, fmaxf(a11, a22));
        return isfinite(value) ? value : 0.0f;
    }
    a00 /= scale; a01 /= scale; a02 /= scale;
    a11 /= scale; a12 /= scale; a22 /= scale;
    const float q = (a00 + a11 + a22) / 3.0f;
    const float b00 = a00-q, b11 = a11-q, b22 = a22-q;
    const float p2 = (b00*b00 + b11*b11 + b22*b22 +
        2.0f*(a01*a01 + a02*a02 + a12*a12)) / 6.0f;
    if (!(p2 > 1.0e-30f) || !isfinite(p2)) {
        const float value = q * scale;
        return isfinite(value) ? value : 0.0f;
    }
    const float p = sqrtf(p2);
    const float c00=b00/p, c01=a01/p, c02=a02/p, c11=b11/p, c12=a12/p, c22=b22/p;
    const float det = c00*(c11*c22-c12*c12) - c01*(c01*c22-c12*c02) + c02*(c01*c12-c11*c02);
    const float phi = acosf(fminf(1.0f, fmaxf(-1.0f, 0.5f*det))) / 3.0f;
    constexpr float two_pi_over_three = 2.0943951023931954923f;
    const float value = fmaxf(q + 2.0f*p*cosf(phi),
        fmaxf(q + 2.0f*p*cosf(phi + two_pi_over_three),
              q + 2.0f*p*cosf(phi + 2.0f*two_pi_over_three))) * scale;
    return isfinite(value) ? value : 0.0f;
}

// q=cbrt(D). Activation is an integer floor estimate, then q is limited by
// the active-flaw cap.  SPHSOL grows cracks across the cubic-spline support
// diameter (2 h), using the undamaged elastic wave speed at rho0.
__host__ __device__ inline SphDamageRate sph_damage_rate(
    float lambda_max, float Dold, float K, float mu, float reference_density, float alpha,
    float smoothing_length, int ntot, float strain_min, float activation_floor,
    float strain_max, float m0, float crack_speed_fraction,
    float elastic_modulus = 0.0f, bool continuous_activation = false) {
    SphDamageRate out{0.0f, 0.0f, 0.0f, 0};
    if (!(K > 0.0f) || !(mu > 0.0f) || !(reference_density > 0.0f) || !(smoothing_length > 0.0f) ||
        !(alpha >= 1.0f) || !isfinite(lambda_max)) return out;
    const float D = fminf(1.0f, fmaxf(0.0f, Dold));
    const float E_undamaged = elastic_modulus > 0.0f
        ? elastic_modulus : 9.0f*K*mu/(3.0f*K+mu);
    out.crack_strain = lambda_max > 0.0f && E_undamaged > 0.0f && D < 1.0f
        ? lambda_max*alpha/((1.0f-D)*E_undamaged) : 0.0f;
    if (!isfinite(out.crack_strain)) { out.crack_strain = 0.0f; return out; }
    if (ntot <= 0 || !(activation_floor > 0.0f) || out.crack_strain < activation_floor) return out;
    if (!(strain_max > strain_min) || out.crack_strain >= strain_max) out.nact = ntot;
    else {
        const float estimate = powf(out.crack_strain/strain_min, fmaxf(m0, 1.0f));
        out.nact = isfinite(estimate)
            ? (continuous_activation ? estimate : floorf(estimate)) : ntot;
        out.nact = out.nact < 1 ? 1 : (out.nact > ntot ? ntot : out.nact);
    }
    const float cg = crack_speed_fraction > 0.0f
        ? crack_speed_fraction*sqrtf((K + (4.0f/3.0f)*mu)/(reference_density*alpha)) : 0.0f;
    if (!(cg > 0.0f) || !isfinite(cg)) { out.nact = 0; return out; }
    out.qdot = static_cast<float>(out.nact)*cg/(2.0f*smoothing_length);
    out.qcap = powf(static_cast<float>(out.nact)/static_cast<float>(ntot), 1.0f/9.0f);
    return out;
}

} // namespace cudadem

#endif // CUDADEM_SPH_DAMAGE_MATH_CUH

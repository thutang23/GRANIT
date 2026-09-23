#ifndef CUDADEM_SPH_HISTORY_MATH_CUH
#define CUDADEM_SPH_HISTORY_MATH_CUH

#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

__host__ __device__ inline double sph_fracture_area_rate(
    double strain, double q, double m, double k, double crack_speed) {
    if (!(strain > 0.0 && q > 0.0 && m > 0.0 && k > 0.0 && crack_speed > 0.0)) return 0.0;
    const double coefficient = 8.0 * 3.141592653589793 * crack_speed * crack_speed *
        crack_speed * k / ((m+1.0)*(m+2.0)*(m+3.0));
    return (m+2.0)*(m+3.0)/(2.0*crack_speed) *
        pow(coefficient, 2.0/3.0) * pow(strain, 2.0*m/3.0) * q;
}

__host__ __device__ inline double sph_fragment_length(double area, double m) {
    // Zero denotes an unmeasured/undamaged length, never an infinite size.
    return area > 0.0 ? 3.0*(m+3.0)/((m+2.0)*area) : 0.0;
}

} // namespace cudadem

#endif // CUDADEM_SPH_HISTORY_MATH_CUH

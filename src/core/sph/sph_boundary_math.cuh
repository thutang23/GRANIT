#ifndef CUDADEM_SPH_BOUNDARY_MATH_CUH
#define CUDADEM_SPH_BOUNDARY_MATH_CUH

#include <cuda_runtime.h>

namespace cudadem {

__host__ __device__ inline void sph_reflect_vector(double& x, double& y, double& z,
    double nx, double ny, double nz, double offset = 0.0) {
    const double distance = nx*x + ny*y + nz*z + offset;
    x -= 2.0*distance*nx; y -= 2.0*distance*ny; z -= 2.0*distance*nz;
}

// Tensor reflection R S R^T works for arbitrary plane normals, including
// shear components; merely changing the diagonal is not a symmetry condition.
__host__ __device__ inline void sph_reflect_stress(
    float& xx, float& xy, float& xz, float& yy, float& yz, float& zz,
    double nx, double ny, double nz) {
    const double n[3] = {nx, ny, nz};
    const double S[3][3] = {{xx,xy,xz},{xy,yy,yz},{xz,yz,zz}};
    double R[3][3], result[3][3] = {};
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
        R[a][b] = (a == b ? 1.0 : 0.0) - 2.0*n[a]*n[b];
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
        for (int c=0; c<3; ++c) for (int d=0; d<3; ++d)
            result[a][b] += R[a][c]*S[c][d]*R[b][d];
    xx=result[0][0]; xy=result[0][1]; xz=result[0][2];
    yy=result[1][1]; yz=result[1][2]; zz=result[2][2];
}

} // namespace cudadem

#endif // CUDADEM_SPH_BOUNDARY_MATH_CUH

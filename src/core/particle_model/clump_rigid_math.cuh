#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace cudadem {

struct ClumpRigidVec3 {
    double x, y, z;
};

// Full body-frame Euler equation:
//   I alpha = torque - omega x (I omega)
// The six inputs are the independent entries of a symmetric inertia tensor.
__host__ __device__ inline bool clump_full_tensor_alpha(
    double ixx, double ixy, double ixz,
    double iyy, double iyz, double izz,
    ClumpRigidVec3 omega, ClumpRigidVec3 torque,
    ClumpRigidVec3& alpha) {
    const double lx=ixx*omega.x+ixy*omega.y+ixz*omega.z;
    const double ly=ixy*omega.x+iyy*omega.y+iyz*omega.z;
    const double lz=ixz*omega.x+iyz*omega.y+izz*omega.z;
    const double rx=torque.x-(omega.y*lz-omega.z*ly);
    const double ry=torque.y-(omega.z*lx-omega.x*lz);
    const double rz=torque.z-(omega.x*ly-omega.y*lx);
    const double A=iyy*izz-iyz*iyz;
    const double B=ixz*iyz-ixy*izz;
    const double C=ixy*iyz-ixz*iyy;
    const double D=ixx*izz-ixz*ixz;
    const double E=ixy*ixz-ixx*iyz;
    const double F=ixx*iyy-ixy*ixy;
    const double det=ixx*A+ixy*B+ixz*C;
    if (!(fabs(det) > 1.e-30)) {
        alpha={0.0,0.0,0.0};
        return false;
    }
    alpha={(A*rx+B*ry+C*rz)/det,
           (B*rx+D*ry+E*rz)/det,
           (C*rx+E*ry+F*rz)/det};
    return true;
}

// Potential-force kicks change angular momentum through torque alone. This
// inverse-inertia helper deliberately excludes the torque-free Euler term.
__host__ __device__ inline bool clump_tensor_torque_alpha(
    double ixx, double ixy, double ixz,
    double iyy, double iyz, double izz,
    ClumpRigidVec3 torque, ClumpRigidVec3& alpha) {
    return clump_full_tensor_alpha(
        ixx, ixy, ixz, iyy, iyz, izz,
        {0.0, 0.0, 0.0}, torque, alpha);
}

// Exact exponential-map orientation update for constant body-frame omega over
// dt. q rotates body to world, so the increment multiplies on the right.
__host__ __device__ inline void clump_exp_quaternion(
    double& qw, double& qx, double& qy, double& qz,
    ClumpRigidVec3 omega, double dt) {
    const double norm=sqrt(omega.x*omega.x+omega.y*omega.y+omega.z*omega.z);
    const double half=0.5*dt*norm;
    const double scale=norm<1.e-12?0.5*dt:sin(half)/norm;
    const double dw=cos(half),dx=scale*omega.x,dy=scale*omega.y,dz=scale*omega.z;
    const double nw=qw*dw-qx*dx-qy*dy-qz*dz;
    const double nx=qw*dx+qx*dw+qy*dz-qz*dy;
    const double ny=qw*dy-qx*dz+qy*dw+qz*dx;
    const double nz=qw*dz+qx*dy-qy*dx+qz*dw;
    const double inverse_norm=1.0/sqrt(nw*nw+nx*nx+ny*ny+nz*nz);
    qw=nw*inverse_norm;qx=nx*inverse_norm;qy=ny*inverse_norm;qz=nz*inverse_norm;
}

} // namespace cudadem

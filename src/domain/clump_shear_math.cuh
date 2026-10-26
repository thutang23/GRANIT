#pragma once

#include "../core/shear/shear_box.h"
#include "../core/particle_model/clump_rigid_math.cuh"
#include <climits>

namespace cudadem {

// The continuum second moment includes every sphere's intrinsic inertia.
// Summing Hill forces only at member centers would omit that contribution.
__host__ __device__ inline ClumpRigidVec3 clump_hill_torque(
    const double* inertia,const double* quat,ClumpRigidVec3 omega_world,
    const ShearBoxConfig& shear) {
    const double w=quat[0],x=quat[1],y=quat[2],z=quat[3];
    const double R[3][3]={{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)},
                          {2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)},
                          {2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}};
    const double I[3][3]={{inertia[0],inertia[1],inertia[2]},
                          {inertia[1],inertia[3],inertia[4]},
                          {inertia[2],inertia[4],inertia[5]}};
    double world[3][3]{};
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)world[a][b]+=R[a][i]*I[i][j]*R[b][j];
    const double half_trace=.5*(world[0][0]+world[1][1]+world[2][2]);
    const double sxz=-world[0][2],syz=-world[1][2],sxy=-world[0][1],szz=half_trace-world[2][2];
    const double ax=2*shear.shear_parameter*shear.omega*shear.omega;
    const double az=-shear.vertical_frequency*shear.vertical_frequency;
    const ClumpRigidVec3 moment{sxz*shear.omega,syz*shear.omega,szz*shear.omega};
    return {az*syz+2*(omega_world.y*moment.z-omega_world.z*moment.y),
            (ax-az)*sxz+2*(omega_world.z*moment.x-omega_world.x*moment.z),
            -ax*sxy+2*(omega_world.x*moment.y-omega_world.y*moment.x)};
}

__host__ __device__ inline bool clump_checked_add(int64_t a,int64_t b,int64_t& result) {
    if((b>0&&a>INT64_MAX-b)||(b<0&&a<INT64_MIN-b))return false;
    result=a+b;return true;
}
__host__ __device__ inline bool clump_checked_mul(int64_t a,int64_t b,int64_t& result) {
    if(a==0||b==0){result=0;return true;}
    if((a==-1&&b==INT64_MIN)||(b==-1&&a==INT64_MIN))return false;
    if(a>0){if((b>0&&a>INT64_MAX/b)||(b<0&&b<INT64_MIN/a))return false;}
    else if((b>0&&a<INT64_MIN/b)||(b<0&&a<INT64_MAX/b))return false;
    result=a*b;return true;
}
// Canonical shearing representative and absolute winding; repeated member
// synchronization is idempotent because it starts from body winding each time.
__host__ __device__ inline bool clump_shear_wrap(
    double& x,double& y,double& vy,int64_t& wx,int64_t& wy,
    double xmin,double ymin,double lx,double ly,const ShearPhase& phase) {
    const double fx=floor((x-xmin)/lx);
    if(!(fx>=-9223372036854775808.0&&fx<9223372036854775808.0))return false;
    const int64_t kx=static_cast<int64_t>(fx);
    const double shifted=y+static_cast<double>(kx)*phase.reduced_shift;
    const double fy=floor((shifted-ymin)/ly);
    if(!(fy>=-9223372036854775808.0&&fy<9223372036854775808.0))return false;
    const int64_t ky=static_cast<int64_t>(fy);
    int64_t cycle,increment,new_x,new_y;
    if(!clump_checked_mul(kx,phase.cycle,cycle)||!clump_checked_add(ky,cycle,increment)||
       !clump_checked_add(wx,kx,new_x)||!clump_checked_add(wy,increment,new_y))return false;
    x-=static_cast<double>(kx)*lx;y=shifted-static_cast<double>(ky)*ly;
    vy+=static_cast<double>(kx)*phase.shear_velocity;wx=new_x;wy=new_y;return true;
}
} // namespace cudadem

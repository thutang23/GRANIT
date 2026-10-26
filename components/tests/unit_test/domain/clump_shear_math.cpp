#include "src/domain/clump_shear_math.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using cudadem::ClumpRigidVec3;
static ClumpRigidVec3 cross(ClumpRigidVec3 a,ClumpRigidVec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
static void near(double a,double b,double tolerance=1.e-12) {
    if(std::fabs(a-b)>tolerance*(1+std::fabs(b)))throw std::runtime_error("clump shear analytic check failed");
}
static ClumpRigidVec3 rotate(const double* q,ClumpRigidVec3 v) {
    const ClumpRigidVec3 r{q[1],q[2],q[3]};auto t=cross(r,v);t={2*t.x,2*t.y,2*t.z};const auto u=cross(r,t);
    return {v.x+q[0]*t.x+u.x,v.y+q[0]*t.y+u.y,v.z+q[0]*t.z+u.z};
}
int main() {
    cudadem::ShearBoxConfig shear;shear.enabled=true;shear.omega=.37;shear.shear_parameter=1.5;shear.vertical_frequency=.23;
    struct Point {ClumpRigidVec3 r;double mass;};
    std::vector<Point> points;
    // Six-point quadrature of each sphere's second moment. These terms are
    // absent from the incorrect member-center-only Hill-torque reduction.
    const std::array<ClumpRigidVec3,3> centers{{{.13,-.07,.05},{-.11,.17,.03},{.02,-.10,-.08}}};
    for(size_t sphere=0;sphere<centers.size();++sphere){const double mass=1.2+sphere*.3,radius=.04+sphere*.01;
        const double intrinsic=.4*mass*radius*radius;
        const double arm=std::sqrt(1.5*intrinsic/mass);
        for(int axis=0;axis<3;++axis)for(double sign:{-1.,1.}){auto r=centers[sphere];
            if(axis==0)r.x+=sign*arm;if(axis==1)r.y+=sign*arm;if(axis==2)r.z+=sign*arm;
            points.push_back({r,mass/6});}}
    double inertia[6]{};
    for(const auto& point:points){const auto r=point.r;const double m=point.mass;
        inertia[0]+=m*(r.y*r.y+r.z*r.z);inertia[1]-=m*r.x*r.y;inertia[2]-=m*r.x*r.z;
        inertia[3]+=m*(r.x*r.x+r.z*r.z);inertia[4]-=m*r.y*r.z;inertia[5]+=m*(r.x*r.x+r.y*r.y);}
    const double angle=.63,norm=std::sqrt(14.);const double q[4]={std::cos(angle/2),std::sin(angle/2)/norm,2*std::sin(angle/2)/norm,3*std::sin(angle/2)/norm};
    const ClumpRigidVec3 spin{.7,-.2,.9},orbital{0,0,shear.omega};
    ClumpRigidVec3 direct{};
    for(const auto& point:points){const auto r=rotate(q,point.r);const auto velocity=cross(spin,r);const auto coriolis=cross(orbital,velocity);
        const ClumpRigidVec3 acceleration{2*shear.shear_parameter*shear.omega*shear.omega*r.x-2*coriolis.x,
            -2*coriolis.y,-shear.vertical_frequency*shear.vertical_frequency*r.z-2*coriolis.z};
        const auto torque=cross(r,acceleration);direct.x+=point.mass*torque.x;direct.y+=point.mass*torque.y;direct.z+=point.mass*torque.z;}
    const auto analytic=cudadem::clump_hill_torque(inertia,q,spin,shear);
    near(analytic.x,direct.x);near(analytic.y,direct.y);near(analytic.z,direct.z);
    const double sphere_inertia[6]={2,0,0,2,0,2},identity[4]={1,0,0,0};
    const auto precession=cudadem::clump_hill_torque(sphere_inertia,identity,{.7,0,0},shear);
    near(precession.x,0);near(precession.y,-2*shear.omega*.7);near(precession.z,0);

    cudadem::ShearPhase phase;phase.reduced_shift=.41;phase.shear_velocity=.08;phase.cycle=3;
    double x=1.037,y=.21,vy=.5;int64_t wx=7,wy=-5;
    const double original_x=x+2*wx,original_y=y+2*wy-wx*(phase.reduced_shift+2*phase.cycle),original_vy=vy-wx*phase.shear_velocity;
    if(!cudadem::clump_shear_wrap(x,y,vy,wx,wy,-1,-1,2,2,phase))throw std::runtime_error("clump wrap unexpectedly failed");
    near(x+2*wx,original_x);near(y+2*wy-wx*(phase.reduced_shift+2*phase.cycle),original_y);near(vy-wx*phase.shear_velocity,original_vy);
    if(!(x>=-1&&x<1&&y>=-1&&y<1))throw std::runtime_error("clump wrap left physical box");
    double overflow_x=1.1,overflow_y=0,overflow_vy=0;int64_t overflow_wx=INT64_MAX,overflow_wy=0;
    if(cudadem::clump_shear_wrap(overflow_x,overflow_y,overflow_vy,overflow_wx,overflow_wy,-1,-1,2,2,phase))
        throw std::runtime_error("clump winding overflow was accepted");
    std::cout<<"clump shear full-inertia torque, spherical precession, lattice winding and overflow checks passed\n";
}

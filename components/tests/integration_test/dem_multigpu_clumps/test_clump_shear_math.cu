// Host-only test of the actual production __host__ __device__ shear helpers.
// Build with nvcc; this executable never initializes CUDA or launches kernels.
#include "../../../../src/domain/clump_shear_math.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace cudadem;
using V = std::array<double,3>;
static V cross(V a,V b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static V rotate(const double* q,V p) {
    const V u{q[1],q[2],q[3]};auto t=cross(u,p);for(auto& x:t)x*=2;
    const auto r=cross(u,t);for(int a=0;a<3;++a)p[a]+=q[0]*t[a]+r[a];return p;
}
static double norm(V v) {return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
static void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

int main() {
    try {
        double worst=0, intrinsic_signal=0;
        const V centers[4]={{-.1,-.2,-.3},{-.1,.2,.3},{.1,-.2,.3},{.1,.2,-.3}};
        for(int test=0;test<24;++test) {
            double q[4]={1.,.013*test,-.017*(test%5),.019*(test%7)};
            double qnorm=0;for(double x:q)qnorm+=x*x;for(auto& x:q)x/=std::sqrt(qnorm);
            const V omega_body{.2+.07*test,-.8+.03*test,1.1-.02*test};
            const V w=rotate(q,omega_body);
            ShearBoxConfig shear{};shear.enabled=true;shear.omega=.1+.003*test;
            shear.shear_parameter=1.1+.02*test;shear.vertical_frequency=.07+.001*test;
            const double radius=.03+.001*test;
            double I[6]={};
            V quadrature{},center_torque{};
            auto field=[&](V r) {
                const auto spin=cross(w,r);
                return V{2*shear.omega*spin[1]+2*shear.shear_parameter*shear.omega*shear.omega*r[0],
                         -2*shear.omega*spin[0],-shear.vertical_frequency*shear.vertical_frequency*r[2]};
            };
            for(const auto& c:centers) {
                const double self=.4*radius*radius;
                I[0]+=self+c[1]*c[1]+c[2]*c[2];I[1]-=c[0]*c[1];I[2]-=c[0]*c[2];
                I[3]+=self+c[0]*c[0]+c[2]*c[2];I[4]-=c[1]*c[2];I[5]+=self+c[0]*c[0]+c[1]*c[1];
                const auto rc=rotate(q,c);const auto tc=cross(rc,field(rc));
                for(int a=0;a<3;++a)center_torque[a]+=tc[a];
                // Six equal masses reproduce the sphere's full second moment:
                // variance R²/5 per axis, hence intrinsic inertia2mR²/5.
                for(int axis=0;axis<3;++axis)for(double sign:{-1.,1.}) {
                    auto point=c;point[axis]+=sign*radius*std::sqrt(3./5.);
                    const auto r=rotate(q,point);const auto torque=cross(r,field(r));
                    for(int a=0;a<3;++a)quadrature[a]+=torque[a]/6.;
                }
            }
            const auto actual=clump_hill_torque(I,q,{w[0],w[1],w[2]},shear);
            const V delta{actual.x-quadrature[0],actual.y-quadrature[1],actual.z-quadrature[2]};
            worst=std::max(worst,norm(delta));
            intrinsic_signal=std::max(intrinsic_signal,norm({quadrature[0]-center_torque[0],quadrature[1]-center_torque[1],quadrature[2]-center_torque[2]}));
            require(norm(delta)<1e-12,"full-inertia Hill torque differs from continuum quadrature");
        }
        require(intrinsic_signal>1e-6,"fixture failed to exercise intrinsic sphere torque");
        ShearPhase phase{};phase.reduced_shift=.3;phase.shear_velocity=.4;phase.cycle=2;
        double x=1.005,y=.2,vy=.7;int64_t wx=0,wy=0;
        require(clump_shear_wrap(x,y,vy,wx,wy,-1,-1,2,2,phase),"seam wrap failed");
        require(std::fabs(x+.995)<1e-14&&std::fabs(y-.5)<1e-14&&std::fabs(vy-1.1)<1e-14&&wx==1&&wy==2,"wrong seam image or winding transform");
        const auto saved=std::array<double,3>{x,y,vy};
        require(clump_shear_wrap(x,y,vy,wx,wy,-1,-1,2,2,phase),"canonical rewrap failed");
        require(saved==std::array<double,3>{x,y,vy}&&wx==1&&wy==2,"wrapping is not idempotent");
        x=1.005;y=.2;vy=.7;wx=INT64_MAX;wy=0;
        require(!clump_shear_wrap(x,y,vy,wx,wy,-1,-1,2,2,phase),"winding overflow was accepted");
        std::cout<<"{\"pass\":true,\"hill_cases\":24,\"max_torque_error\":"<<worst
                 <<",\"intrinsic_torque_signal\":"<<intrinsic_signal<<",\"wrap_checks\":3}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}

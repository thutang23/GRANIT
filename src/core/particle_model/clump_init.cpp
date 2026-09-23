// =============================================================================
// Unified-v3 imported-member clump initialization
// =============================================================================

#include "clump.h"
#include "cuda_utils.h"
#include "../acceleration/body_fitted_fmm/clump_p3_gravity.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cudadem {
namespace {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

Vec3 rotate(const Quat& q, const Vec3& v) {
    const double tx=2.0*(q[2]*v[2]-q[3]*v[1]);
    const double ty=2.0*(q[3]*v[0]-q[1]*v[2]);
    const double tz=2.0*(q[1]*v[1]-q[2]*v[0]);
    return {v[0]+q[0]*tx+(q[2]*tz-q[3]*ty),
            v[1]+q[0]*ty+(q[3]*tx-q[1]*tz),
            v[2]+q[0]*tz+(q[1]*ty-q[2]*tx)};
}

Vec3 inverse_rotate(const Quat& q, const Vec3& v) {
    return rotate({q[0],-q[1],-q[2],-q[3]}, v);
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

const ClumpTemplate* find_template(const ClumpConfig& cfg, const char* name) {
    if (!name || !name[0]) return nullptr;
    for (int32_t i=0;i<cfg.num_templates;++i)
        if (std::strcmp(cfg.templates[i].name,name)==0) return &cfg.templates[i];
    return nullptr;
}

// Minimum-norm solve for a symmetric positive-semidefinite 3x3 matrix using
// Jacobi eigenvectors. Eigenvalues below a scale-aware threshold are treated
// as unconstrained spin and therefore mapped to zero.
Vec3 minimum_norm_solve(double A[3][3], Vec3 b) {
    double V[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(int sweep=0;sweep<16;++sweep) {
        int p=0,q=1;
        if(std::fabs(A[0][2])>std::fabs(A[p][q])){p=0;q=2;}
        if(std::fabs(A[1][2])>std::fabs(A[p][q])){p=1;q=2;}
        if(std::fabs(A[p][q]) <= 64*std::numeric_limits<double>::epsilon() *
            std::max({std::fabs(A[0][0]),std::fabs(A[1][1]),std::fabs(A[2][2]),1.0})) break;
        const double tau=(A[q][q]-A[p][p])/(2*A[p][q]);
        const double t=(tau>=0?1.0:-1.0)/(std::fabs(tau)+std::sqrt(1+tau*tau));
        const double c=1/std::sqrt(1+t*t), s=t*c;
        const double app=A[p][p], aqq=A[q][q], apq=A[p][q];
        A[p][p]=app-t*apq; A[q][q]=aqq+t*apq; A[p][q]=A[q][p]=0;
        for(int r=0;r<3;++r) if(r!=p&&r!=q) {
            const double arp=A[r][p], arq=A[r][q];
            A[r][p]=A[p][r]=c*arp-s*arq; A[r][q]=A[q][r]=s*arp+c*arq;
        }
        for(int r=0;r<3;++r) { const double vrp=V[r][p],vrq=V[r][q]; V[r][p]=c*vrp-s*vrq;V[r][q]=s*vrp+c*vrq; }
    }
    const double max_eval=std::max({std::fabs(A[0][0]),std::fabs(A[1][1]),std::fabs(A[2][2]),1.0});
    const double tol=256*std::numeric_limits<double>::epsilon()*max_eval;
    Vec3 x{0,0,0};
    for(int k=0;k<3;++k) if(A[k][k]>tol) {
        double projection=0; for(int r=0;r<3;++r) projection+=V[r][k]*b[r];
        projection/=A[k][k]; for(int r=0;r<3;++r)x[r]+=V[r][k]*projection;
    }
    return x;
}

void derive_p3(double out[19], double mass, const Vec3& r) {
    const double x=r[0],y=r[1],z=r[2], x2=x*x,y2=y*y,z2=z*z, r2=x2+y2+z2;
    out[0]+=mass; out[1]+=mass*x; out[2]+=mass*y; out[3]+=mass*z;
    out[4]+=3*mass*(x2-y2); out[5]+=3*mass*(y2-z2);
    out[6]+=3*mass*x*y; out[7]+=3*mass*x*z; out[8]+=3*mass*y*z;
    out[9]+=mass*(5*x*x2-3*x*r2); out[10]+=mass*(5*y*y2-3*y*r2);
    out[11]+=mass*(5*z*z2-3*z*r2); out[12]+=mass*(5*x2*y-y*r2);
    out[13]+=mass*(5*x2*z-z*r2); out[14]+=mass*(5*y2*z-z*r2);
    out[15]+=5*mass*x*y*z; out[16]+=mass*(5*x*y2-x*r2);
    out[17]+=mass*(5*x*z2-x*r2); out[18]+=mass*(5*y*z2-y*r2);
}

} // namespace

void initialize_clumps(const SimConfig& config,
                       const DeviceMaterialDatabase&,
                       ParticleSoA& particles,
                       ClumpStateSoA& cs,
                       ClumpSegmentOffsets& seg) {
    const int32_t n=particles.num_particles;
    canonicalize_particle_clump_order(particles);
    std::vector<int32_t> ids(n); std::vector<uint8_t> life(n);
    if(n) CUDA_CHECK(cudaMemcpy(ids.data(),particles.clump_id,n*sizeof(int32_t),cudaMemcpyDeviceToHost));
    if(n) CUDA_CHECK(cudaMemcpy(life.data(),particles.lifecycle_state,n*sizeof(uint8_t),cudaMemcpyDeviceToHost));
    int32_t normal_end=0,max_id=0,member_end=0;
    while(normal_end<n&&ids[normal_end]==0&&life[normal_end]==0)++normal_end;
    member_end=normal_end;
    while(member_end<n&&ids[member_end]>0&&life[member_end]==0){max_id=std::max(max_id,ids[member_end]);++member_end;}
    for(int32_t i=member_end;i<n;++i) if(ids[i]!=-1 && life[i]==0) throw std::runtime_error("internal clump ordering failure");
    seg.normal_begin=0;seg.normal_end=normal_end;seg.member_begin=normal_end;seg.member_end=member_end;seg.dead_begin=member_end;seg.dead_end=n;

    if(max_id==0) {
        if(config.clump.num_instances!=0) throw std::runtime_error("clumps.instances is present but the input has no positive clump IDs");
        return;
    }
    if(config.physics!=SimulationPhysics::DEM) throw std::runtime_error("SPH inputs may contain only clump_id -1 or 0");
    std::vector<int32_t> count(max_id+1,0);
    for(int32_t i=normal_end;i<member_end;++i)++count[ids[i]];
    for(int32_t id=1;id<=max_id;++id)if(count[id]==0)throw std::runtime_error("positive clump IDs must be contiguous 1..K");
    if(config.clump.num_instances!=max_id)throw std::runtime_error("clumps.instances must contain exactly one instance for every positive ID");
    if(cs.capacity<max_id)throw std::runtime_error("clump state capacity is too small");

    std::vector<double> px(n),py(n),pz(n),vx(n),vy(n),vz(n),wx(n),wy(n),wz(n),pqw(n),pqx(n),pqy(n),pqz(n);
    std::vector<double> fx(n,0),fy(n,0),fz(n,0),tx(n,0),ty(n,0),tz(n,0);
    std::vector<float> masses(n),radii(n),offx(n,0),offy(n,0),offz(n,0);
    auto d2h=[&](auto* d,const auto* s){if(n)CUDA_CHECK(cudaMemcpy(d,s,n*sizeof(*d),cudaMemcpyDeviceToHost));};
    d2h(px.data(),particles.pos_x);d2h(py.data(),particles.pos_y);d2h(pz.data(),particles.pos_z);
    d2h(vx.data(),particles.vel_x);d2h(vy.data(),particles.vel_y);d2h(vz.data(),particles.vel_z);
    d2h(wx.data(),particles.omega_x);d2h(wy.data(),particles.omega_y);d2h(wz.data(),particles.omega_z);
    d2h(pqw.data(),particles.quat_w);d2h(pqx.data(),particles.quat_x);d2h(pqy.data(),particles.quat_y);d2h(pqz.data(),particles.quat_z);
    d2h(masses.data(),particles.mass);d2h(radii.data(),particles.radius);
    const bool checkpoint=particles.input_has_clump_offset;
    if(checkpoint){d2h(offx.data(),particles.clump_off_x);d2h(offy.data(),particles.clump_off_y);d2h(offz.data(),particles.clump_off_z);d2h(fx.data(),particles.force_x);d2h(fy.data(),particles.force_y);d2h(fz.data(),particles.force_z);d2h(tx.data(),particles.torque_x);d2h(ty.data(),particles.torque_y);d2h(tz.data(),particles.torque_z);}

    std::vector<double> M(max_id,0),cx(max_id,0),cy(max_id,0),cz(max_id,0),cvx(max_id,0),cvy(max_id,0),cvz(max_id,0);
    for(int32_t i=normal_end;i<member_end;++i){const int c=ids[i]-1; const double mass=masses[i]; if(!(mass>0)||!std::isfinite(mass))throw std::runtime_error("clump member mass must be finite and positive"); M[c]+=mass;cx[c]+=mass*px[i];cy[c]+=mass*py[i];cz[c]+=mass*pz[i];cvx[c]+=mass*vx[i];cvy[c]+=mass*vy[i];cvz[c]+=mass*vz[i];}
    for(int c=0;c<max_id;++c){cx[c]/=M[c];cy[c]/=M[c];cz[c]/=M[c];cvx[c]/=M[c];cvy[c]/=M[c];cvz[c]/=M[c];}

    std::vector<double> ixx(max_id,0),iyy(max_id,0),izz(max_id,0),ixy(max_id,0),ixz(max_id,0),iyz(max_id,0),owx(max_id,0),owy(max_id,0),owz(max_id,0);
    std::vector<double> nfx(max_id,0),nfy(max_id,0),nfz(max_id,0),ntx(max_id,0),nty(max_id,0),ntz(max_id,0);
    std::vector<double> qw(max_id,1),qx(max_id,0),qy(max_id,0),qz(max_id,0);
    std::array<std::vector<double>,19> p3;for(auto& v:p3)v.assign(max_id,0);
    std::vector<int32_t> begin(max_id),end(max_id);int cursor=normal_end;
    for(int c=0;c<max_id;++c){begin[c]=cursor;cursor+=count[c+1];end[c]=cursor;const auto& inst=config.clump.instances[c];if(inst.id!=c+1)throw std::runtime_error("clump instance IDs do not match input IDs");
        const ClumpTemplate* tmpl=find_template(config.clump,inst.template_name);
        Quat q=checkpoint ? Quat{pqw[begin[c]],pqx[begin[c]],pqy[begin[c]],pqz[begin[c]]}
                          : Quat{inst.orientation[0],inst.orientation[1],inst.orientation[2],inst.orientation[3]};
        if(checkpoint){
            double qn=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
            if(!(qn>0))throw std::runtime_error("checkpoint clump quaternion is invalid");for(double& x:q)x/=qn;
            Vec3 recovered{0,0,0};
            for(int i=begin[c];i<end[c];++i){Quat qi={pqw[i],pqx[i],pqy[i],pqz[i]};double dot=0;for(int k=0;k<4;++k)dot+=q[k]*qi[k];if(std::fabs(std::fabs(dot)-1.0)>1e-10)throw std::runtime_error("checkpoint member quaternions disagree");Vec3 rw=rotate(q,{offx[i],offy[i],offz[i]});recovered[0]+=masses[i]*(px[i]-rw[0]);recovered[1]+=masses[i]*(py[i]-rw[1]);recovered[2]+=masses[i]*(pz[i]-rw[2]);}
            cx[c]=recovered[0]/M[c];cy[c]=recovered[1]/M[c];cz[c]=recovered[2]/M[c];
        }
        qw[c]=q[0];qx[c]=q[1];qy[c]=q[2];qz[c]=q[3];
        double extent=0;
        if(inst.has_com&&!checkpoint){const double delta=std::hypot(std::hypot(inst.com[0]-cx[c],inst.com[1]-cy[c]),inst.com[2]-cz[c]);for(int i=begin[c];i<end[c];++i)extent=std::max(extent,std::hypot(std::hypot(px[i]-cx[c],py[i]-cy[c]),pz[i]-cz[c])+radii[i]);const double tol=1e-10*std::max(extent,1.0);if(delta>tol&&(!tmpl||!tmpl->has_inertia||!tmpl->has_multipole))throw std::runtime_error("materially different CoM override requires inertia and multipole overrides");cx[c]=inst.com[0];cy[c]=inst.com[1];cz[c]=inst.com[2];}
        double local_p3[19]={};double A[3][3]={{0,0,0},{0,0,0},{0,0,0}};Vec3 rhs{0,0,0};
        for(int i=begin[c];i<end[c];++i){const double mass=masses[i];Vec3 rw={px[i]-cx[c],py[i]-cy[c],pz[i]-cz[c]};Vec3 rb=checkpoint?Vec3{offx[i],offy[i],offz[i]}:inverse_rotate(q,rw);offx[i]=static_cast<float>(rb[0]);offy[i]=static_cast<float>(rb[1]);offz[i]=static_cast<float>(rb[2]);extent=std::max(extent,std::sqrt(rb[0]*rb[0]+rb[1]*rb[1]+rb[2]*rb[2])+radii[i]);const double self=.4*mass*radii[i]*radii[i];ixx[c]+=self+mass*(rb[1]*rb[1]+rb[2]*rb[2]);iyy[c]+=self+mass*(rb[0]*rb[0]+rb[2]*rb[2]);izz[c]+=self+mass*(rb[0]*rb[0]+rb[1]*rb[1]);ixy[c]-=mass*rb[0]*rb[1];ixz[c]-=mass*rb[0]*rb[2];iyz[c]-=mass*rb[1]*rb[2];derive_p3(local_p3,mass,rb);
            if(checkpoint){nfx[c]+=fx[i];nfy[c]+=fy[i];nfz[c]+=fz[i];ntx[c]+=rw[1]*fz[i]-rw[2]*fy[i]+tx[i];nty[c]+=rw[2]*fx[i]-rw[0]*fz[i]+ty[i];ntz[c]+=rw[0]*fy[i]-rw[1]*fx[i]+tz[i];}
            A[0][0]+=mass*(rw[1]*rw[1]+rw[2]*rw[2]);A[1][1]+=mass*(rw[0]*rw[0]+rw[2]*rw[2]);A[2][2]+=mass*(rw[0]*rw[0]+rw[1]*rw[1]);A[0][1]-=mass*rw[0]*rw[1];A[0][2]-=mass*rw[0]*rw[2];A[1][2]-=mass*rw[1]*rw[2];Vec3 dv={vx[i]-cvx[c],vy[i]-cvy[c],vz[i]-cvz[c]};Vec3 cr=cross(rw,dv);for(int k=0;k<3;++k)rhs[k]+=mass*cr[k];}
        A[1][0]=A[0][1];A[2][0]=A[0][2];A[2][1]=A[1][2];
        Vec3 omega_world;
        if(particles.input_has_omega){omega_world={wx[begin[c]],wy[begin[c]],wz[begin[c]]};const double tol=1e-10*std::max({std::fabs(omega_world[0]),std::fabs(omega_world[1]),std::fabs(omega_world[2]),1.0});for(int i=begin[c];i<end[c];++i)if(std::fabs(wx[i]-omega_world[0])>tol||std::fabs(wy[i]-omega_world[1])>tol||std::fabs(wz[i]-omega_world[2])>tol)throw std::runtime_error("all members of a clump must store the same world omega");Vec3 recovered_v{0,0,0};for(int i=begin[c];i<end[c];++i){Vec3 rw={px[i]-cx[c],py[i]-cy[c],pz[i]-cz[c]},spin=cross(omega_world,rw);recovered_v[0]+=masses[i]*(vx[i]-spin[0]);recovered_v[1]+=masses[i]*(vy[i]-spin[1]);recovered_v[2]+=masses[i]*(vz[i]-spin[2]);}cvx[c]=recovered_v[0]/M[c];cvy[c]=recovered_v[1]/M[c];cvz[c]=recovered_v[2]/M[c];const double vtol=1e-10*std::max({std::fabs(cvx[c]),std::fabs(cvy[c]),std::fabs(cvz[c]),extent*std::fabs(omega_world[0]),extent*std::fabs(omega_world[1]),extent*std::fabs(omega_world[2]),1.0});for(int i=begin[c];i<end[c];++i){Vec3 rw={px[i]-cx[c],py[i]-cy[c],pz[i]-cz[c]},spin=cross(omega_world,rw);if(std::fabs(vx[i]-cvx[c]-spin[0])>vtol||std::fabs(vy[i]-cvy[c]-spin[1])>vtol||std::fabs(vz[i]-cvz[c]-spin[2])>vtol)throw std::runtime_error("member velocities do not match the supplied rigid omega");}}
        else omega_world=minimum_norm_solve(A,rhs);
        Vec3 omega_body=inverse_rotate(q,omega_world);owx[c]=omega_body[0];owy[c]=omega_body[1];owz[c]=omega_body[2];
        if(tmpl&&tmpl->has_inertia){ixx[c]=tmpl->inertia[0];iyy[c]=tmpl->inertia[1];izz[c]=tmpl->inertia[2];ixy[c]=tmpl->inertia[3];ixz[c]=tmpl->inertia[4];iyz[c]=tmpl->inertia[5];}
        if(tmpl&&tmpl->has_multipole){const double mtol=1e-10*std::max(M[c],1.0),dtol=1e-10*std::max(M[c]*std::max(extent,1.0),1.0);if(std::fabs(tmpl->multipole_p3[0]-M[c])>mtol)throw std::runtime_error("clump multipole monopole must equal derived mass");for(int k=1;k<=3;++k)if(std::fabs(tmpl->multipole_p3[k])>dtol)throw std::runtime_error("clump multipole dipole must vanish about its CoM");for(int k=0;k<19;++k)p3[k][c]=tmpl->multipole_p3[k];}else for(int k=0;k<19;++k)p3[k][c]=local_p3[k];
    }

    // A velocity-Verlet restart must retain the force used by the next first
    // half-kick. Member checkpoint forces reconstruct contact/wall/external
    // wrench above; clump self-gravity is body-level and therefore has to be
    // re-evaluated from the same conservative p3 energy here.
    if(checkpoint && config.acceleration.enable_self_gravity &&
       config.acceleration.gravity_update_rate <= 1) {
        using namespace body_fitted_fmm;
        bool periodic[3]={false,false,false}; double length[3]={0,0,0};
        for(int w=0;w<config.num_walls;++w) if(config.walls[w].enabled&&config.walls[w].type==WallType::PERIODICAL&&config.walls[w].axis<3){const int a=config.walls[w].axis;periodic[a]=true;length[a]=config.walls[w].bound_max-config.walls[w].bound_min;}
        auto minimum_image=[&](ClumpVec3 r){if(periodic[0]&&length[0]>0){const double h=.5*length[0];if(r.x>h)r.x-=length[0];else if(r.x<-h)r.x+=length[0];}if(periodic[1]&&length[1]>0){const double h=.5*length[1];if(r.y>h)r.y-=length[1];else if(r.y<-h)r.y+=length[1];}if(periodic[2]&&length[2]>0){const double h=.5*length[2];if(r.z>h)r.z-=length[2];else if(r.z<-h)r.z+=length[2];}return r;};
        const double G=config.acceleration.unified_fmm.gravitational_constant;
        float min_radius=std::numeric_limits<float>::infinity();for(int i=0;i<member_end;++i)min_radius=std::min(min_radius,radii[i]);
        const double eps=static_cast<double>(min_radius)*config.acceleration.unified_fmm.softening_factor;
        std::vector<ClumpP3> world(max_id);
        for(int c=0;c<max_id;++c){ClumpP3 body;for(int k=0;k<19;++k)body.c[k]=p3[k][c];body.c[0]=M[c];world[c]=clump_p3_rotate_body_to_world(body,qw[c],qx[c],qy[c],qz[c]);}
        for(int c=0;c<max_id;++c){
            const ClumpVec3 xc{cx[c],cy[c],cz[c]};
            for(int i=0;i<normal_end;++i){ClumpP3 point;point.c[0]=masses[i];const ClumpVec3 r=minimum_image({px[i]-xc.x,py[i]-xc.y,pz[i]-xc.z});ClumpVec3 fc{},fp{};clump_p3_pair_force(world[c],point,r,G,eps,fc,fp);const ClumpVec3 tc=clump_p3_pair_torque(world[c],point,r,G,eps,true);nfx[c]+=fc.x;nfy[c]+=fc.y;nfz[c]+=fc.z;ntx[c]+=tc.x;nty[c]+=tc.y;ntz[c]+=tc.z;}
            for(int d=c+1;d<max_id;++d){const ClumpVec3 r=minimum_image({cx[d]-xc.x,cy[d]-xc.y,cz[d]-xc.z});ClumpVec3 fc{},fd{};clump_p3_pair_force(world[c],world[d],r,G,eps,fc,fd);const ClumpVec3 tc=clump_p3_pair_torque(world[c],world[d],r,G,eps,true),td=clump_p3_pair_torque(world[c],world[d],r,G,eps,false);nfx[c]+=fc.x;nfy[c]+=fc.y;nfz[c]+=fc.z;ntx[c]+=tc.x;nty[c]+=tc.y;ntz[c]+=tc.z;nfx[d]+=fd.x;nfy[d]+=fd.y;nfz[d]+=fd.z;ntx[d]+=td.x;nty[d]+=td.y;ntz[d]+=td.z;}
        }
    }

    auto h2d=[&](double* d,const std::vector<double>& v){CUDA_CHECK(cudaMemcpy(d,v.data(),max_id*sizeof(double),cudaMemcpyHostToDevice));};
    h2d(cs.mass,M);h2d(cs.com_x,cx);h2d(cs.com_y,cy);h2d(cs.com_z,cz);h2d(cs.vel_x,cvx);h2d(cs.vel_y,cvy);h2d(cs.vel_z,cvz);h2d(cs.omega_x,owx);h2d(cs.omega_y,owy);h2d(cs.omega_z,owz);h2d(cs.quat_w,qw);h2d(cs.quat_x,qx);h2d(cs.quat_y,qy);h2d(cs.quat_z,qz);h2d(cs.inertia_xx,ixx);h2d(cs.inertia_yy,iyy);h2d(cs.inertia_zz,izz);h2d(cs.inertia_xy,ixy);h2d(cs.inertia_xz,ixz);h2d(cs.inertia_yz,iyz);for(int k=0;k<19;++k)h2d(cs.multipole_p3[k],p3[k]);h2d(cs.net_force_x,nfx);h2d(cs.net_force_y,nfy);h2d(cs.net_force_z,nfz);h2d(cs.net_torque_x,ntx);h2d(cs.net_torque_y,nty);h2d(cs.net_torque_z,ntz);
    const size_t gravity_bytes=static_cast<size_t>(max_id)*sizeof(double);
    CUDA_CHECK(cudaMemset(cs.gravity_force_x,0,gravity_bytes));CUDA_CHECK(cudaMemset(cs.gravity_force_y,0,gravity_bytes));CUDA_CHECK(cudaMemset(cs.gravity_force_z,0,gravity_bytes));
    CUDA_CHECK(cudaMemset(cs.gravity_torque_x,0,gravity_bytes));CUDA_CHECK(cudaMemset(cs.gravity_torque_y,0,gravity_bytes));CUDA_CHECK(cudaMemset(cs.gravity_torque_z,0,gravity_bytes));
    CUDA_CHECK(cudaMemcpy(particles.clump_off_x,offx.data(),n*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(particles.clump_off_y,offy.data(),n*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(particles.clump_off_z,offz.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    for(int c=0;c<max_id;++c){begin[c]-=normal_end;end[c]-=normal_end;}
    CUDA_CHECK(cudaMemcpy(seg.seg_begin,begin.data(),max_id*sizeof(int32_t),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(seg.seg_end,end.data(),max_id*sizeof(int32_t),cudaMemcpyHostToDevice));seg.num_segments=max_id;cs.num_clumps=max_id;
}

} // namespace cudadem

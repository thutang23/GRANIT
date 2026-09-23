// =============================================================================
// trace_free_p3_v1 clump gravity primitives
// =============================================================================
// The on-disk clump p=3 representation deliberately uses the same Cartesian
// ordering as the p=4 FMM expansion.  Keeping this conversion here prevents a
// second, subtly incompatible p=3 convention in the direct and FMM paths.
#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace body_fitted_fmm {

constexpr int CLUMP_P3_COEFFS = 19;
constexpr int CLUMP_P4_FMM_ACTIVE_COEFFS = 28;
// p=3 entries in the canonical FMM p=4 expansion.  The final three are the
// supplementary octupole components required by the Cartesian shift algebra.
// Compact trace_free_p3_v1 index 0..15 maps directly to FMM slots 0..15;
// compact indices 16..18 map to the octupole supplementary slots 25..27.
__host__ __device__ constexpr int clump_p3_to_p4_slot(int compact) {
    return compact < 16 ? compact : compact + 9;
}

struct ClumpP3 {
    double c[CLUMP_P3_COEFFS];
    __host__ __device__ ClumpP3() : c{} {}
};

struct ClumpVec3 { double x, y, z; };

__host__ __device__ inline ClumpVec3 cp3_add(ClumpVec3 a, ClumpVec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__host__ __device__ inline ClumpVec3 cp3_scale(ClumpVec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
__host__ __device__ inline double cp3_dot(ClumpVec3 a, ClumpVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ inline ClumpVec3 cp3_cross(ClumpVec3 a, ClumpVec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

__host__ __device__ inline void cp3_set_sym3(double T[3][3][3], int i, int j, int k, double v) {
    T[i][j][k]=T[i][k][j]=T[j][i][k]=T[j][k][i]=T[k][i][j]=T[k][j][i]=v;
}

__host__ __device__ inline double cp3_qrr(const ClumpP3& p, ClumpVec3 r) {
    const double qxx=(2*p.c[4]+p.c[5])/9, qyy=(-p.c[4]+p.c[5])/9, qzz=-qxx-qyy;
    return qxx*r.x*r.x+qyy*r.y*r.y+qzz*r.z*r.z+2*(p.c[6]/3*r.x*r.y+p.c[7]/3*r.x*r.z+p.c[8]/3*r.y*r.z);
}

__host__ __device__ inline double cp3_trrr(const ClumpP3& p, ClumpVec3 r) {
    const double x=r.x, y=r.y, z=r.z;
    return (p.c[9]*x*x*x+p.c[10]*y*y*y+p.c[11]*z*z*z + 3*(p.c[12]*x*x*y+p.c[13]*x*x*z+p.c[16]*x*y*y+p.c[17]*x*z*z+p.c[14]*y*y*z+p.c[18]*y*z*z)+6*p.c[15]*x*y*z)/5.0;
}

__host__ __device__ inline ClumpVec3 cp3_qrr_grad(const ClumpP3& p, ClumpVec3 r) {
    const double qxx=(2*p.c[4]+p.c[5])/9, qyy=(-p.c[4]+p.c[5])/9, qzz=-qxx-qyy;
    return {2*(qxx*r.x+p.c[6]/3*r.y+p.c[7]/3*r.z),
            2*(p.c[6]/3*r.x+qyy*r.y+p.c[8]/3*r.z),
            2*(p.c[7]/3*r.x+p.c[8]/3*r.y+qzz*r.z)};
}

__host__ __device__ inline ClumpVec3 cp3_trrr_grad(const ClumpP3& p, ClumpVec3 r) {
    const double x=r.x, y=r.y, z=r.z;
    return {(3*p.c[9]*x*x + 3*(2*p.c[12]*x*y+2*p.c[13]*x*z+p.c[16]*y*y+p.c[17]*z*z)+6*p.c[15]*y*z)/5.0,
            (3*p.c[10]*y*y + 3*(p.c[12]*x*x+2*p.c[16]*x*y+2*p.c[14]*y*z+p.c[18]*z*z)+6*p.c[15]*x*z)/5.0,
            (3*p.c[11]*z*z + 3*(p.c[13]*x*x+2*p.c[17]*x*z+p.c[14]*y*y+2*p.c[18]*y*z)+6*p.c[15]*x*y)/5.0};
}

__host__ __device__ inline void clump_p3_to_fmm_p4(const ClumpP3& p3, double p4[CLUMP_P4_FMM_ACTIVE_COEFFS]) {
    for (int i = 0; i < CLUMP_P4_FMM_ACTIVE_COEFFS; ++i) p4[i] = 0.0;
    for (int i = 0; i < CLUMP_P3_COEFFS; ++i) p4[clump_p3_to_p4_slot(i)] = p3.c[i];
}

// Add a p=3 clump source to an existing p=4 node expansion.  `p3` is already
// expressed about `source_center`; translation to a node centre remains the
// normal M2M operation, so this function is intentionally injection-only.
__host__ __device__ inline void clump_p3_inject_fmm_p4(const ClumpP3& p3, double p4[CLUMP_P4_FMM_ACTIVE_COEFFS]) {
    for (int i = 0; i < CLUMP_P3_COEFFS; ++i) p4[clump_p3_to_p4_slot(i)] += p3.c[i];
}

__host__ __device__ inline void cp3_quat_matrix(double w, double x, double y, double z, double R[3][3]) {
    const double n = w*w+x*x+y*y+z*z;
    if (n <= 0.0) { R[0][0]=R[1][1]=R[2][2]=1.0; R[0][1]=R[0][2]=R[1][0]=R[1][2]=R[2][0]=R[2][1]=0.0; return; }
    const double s = 2.0/n;
    R[0][0]=1.0-s*(y*y+z*z); R[0][1]=s*(x*y-z*w);     R[0][2]=s*(x*z+y*w);
    R[1][0]=s*(x*y+z*w);     R[1][1]=1.0-s*(x*x+z*z); R[1][2]=s*(y*z-x*w);
    R[2][0]=s*(x*z-y*w);     R[2][1]=s*(y*z+x*w);     R[2][2]=1.0-s*(x*x+y*y);
}

// Rotate the mass, dipole, STF quadrupole, and STF rank-3 tensor from body to
// world coordinates.  All ten symmetric rank-3 entries are rotated, including
// the three supplementary storage entries, which makes overridden p3 metadata
// round-trip without silently discarding information.
__host__ __device__ inline ClumpP3 clump_p3_rotate_body_to_world(const ClumpP3& in, double qw, double qx, double qy, double qz) {
    double R[3][3]; cp3_quat_matrix(qw,qx,qy,qz,R); ClumpP3 o;
    o.c[0]=in.c[0];
    for(int i=0;i<3;++i) o.c[1+i]=R[i][0]*in.c[1]+R[i][1]*in.c[2]+R[i][2]*in.c[3];
    double Q[3][3] = {};
    Q[0][0]=(2.0*in.c[4]+in.c[5])/9.0; Q[1][1]=(-in.c[4]+in.c[5])/9.0; Q[2][2]=-Q[0][0]-Q[1][1];
    Q[0][1]=Q[1][0]=in.c[6]/3.0; Q[0][2]=Q[2][0]=in.c[7]/3.0; Q[1][2]=Q[2][1]=in.c[8]/3.0;
    double W[3][3] = {};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int a=0;a<3;++a) for(int b=0;b<3;++b) W[i][j]+=R[i][a]*Q[a][b]*R[j][b];
    o.c[4]=3.0*(W[0][0]-W[1][1]); o.c[5]=3.0*(W[1][1]-W[2][2]); o.c[6]=3.0*W[0][1]; o.c[7]=3.0*W[0][2]; o.c[8]=3.0*W[1][2];
    double T[3][3][3] = {};
    cp3_set_sym3(T,0,0,0,in.c[9]/5.0); cp3_set_sym3(T,1,1,1,in.c[10]/5.0); cp3_set_sym3(T,2,2,2,in.c[11]/5.0);
    cp3_set_sym3(T,0,0,1,in.c[12]/5.0); cp3_set_sym3(T,0,0,2,in.c[13]/5.0); cp3_set_sym3(T,1,1,2,in.c[14]/5.0); cp3_set_sym3(T,0,1,2,in.c[15]/5.0);
    cp3_set_sym3(T,0,1,1,in.c[16]/5.0); cp3_set_sym3(T,0,2,2,in.c[17]/5.0); cp3_set_sym3(T,1,2,2,in.c[18]/5.0);
    double U[3][3][3] = {};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) for(int a=0;a<3;++a) for(int b=0;b<3;++b) for(int d=0;d<3;++d) U[i][j][k]+=R[i][a]*R[j][b]*R[k][d]*T[a][b][d];
    o.c[9]=5*U[0][0][0]; o.c[10]=5*U[1][1][1]; o.c[11]=5*U[2][2][2]; o.c[12]=5*U[0][0][1]; o.c[13]=5*U[0][0][2]; o.c[14]=5*U[1][1][2]; o.c[15]=5*U[0][1][2]; o.c[16]=5*U[0][1][1]; o.c[17]=5*U[0][2][2]; o.c[18]=5*U[1][2][2];
    return o;
}

// Analytic Cartesian derivatives of the Plummer Green function.  This small
// polynomial recurrence evaluates every derivative needed through order seven
// without maintaining an error-prone table of p3-p3 cross terms.
struct Cp3GreenTerm { double coeff; int x, y, z, power; };
__host__ __device__ inline double cp3_green_derivative(ClumpVec3 r,double eps,int nx,int ny,int nz) {
    Cp3GreenTerm cur[128],nxt[128]; cur[0]={1.0,0,0,0,1}; int count=1, order=nx+ny+nz;
    for(int q=0;q<order;++q) { int axis=q<nx?0:(q<nx+ny?1:2),out=0; for(int t=0;t<count;++t) { Cp3GreenTerm v=cur[t]; int e=axis==0?v.x:(axis==1?v.y:v.z); if(e){Cp3GreenTerm a=v;a.coeff*=e;if(axis==0)--a.x;else if(axis==1)--a.y;else --a.z;nxt[out++]=a;} Cp3GreenTerm b=v;b.coeff*=-v.power;b.power+=2;if(axis==0)++b.x;else if(axis==1)++b.y;else ++b.z;nxt[out++]=b; } count=out;for(int t=0;t<count;++t)cur[t]=nxt[t]; }
    const double s=sqrt(cp3_dot(r,r)+eps*eps); double out=0.0; for(int t=0;t<count;++t){Cp3GreenTerm v=cur[t];double m=1,d=1;for(int i=0;i<v.x;++i)m*=r.x;for(int i=0;i<v.y;++i)m*=r.y;for(int i=0;i<v.z;++i)m*=r.z;for(int i=0;i<v.power;++i)d*=s;out+=v.coeff*m/d;} return out;
}
__host__ __device__ inline double cp3_tensor(const ClumpP3& p,int rank,int code) {
    if(rank==0)return p.c[0];int i=code%3;if(rank==1)return p.c[1+i];int j=(code/3)%3;
    if(rank==2){double qxx=(2*p.c[4]+p.c[5])/9,qyy=(-p.c[4]+p.c[5])/9,qzz=-qxx-qyy;if(i==j)return i==0?qxx:(i==1?qyy:qzz);return(i+j==1?p.c[6]:(i+j==2?p.c[7]:p.c[8]))/3;}
    int k=(code/9)%3,n0=(i==0)+(j==0)+(k==0),n1=(i==1)+(j==1)+(k==1),n2=3-n0-n1;if(n0==3)return p.c[9]/5;if(n1==3)return p.c[10]/5;if(n2==3)return p.c[11]/5;if(n0==2&&n1==1)return p.c[12]/5;if(n0==2&&n2==1)return p.c[13]/5;if(n1==2&&n2==1)return p.c[14]/5;if(n0==1&&n1==1&&n2==1)return p.c[15]/5;if(n0==1&&n1==2)return p.c[16]/5;if(n0==1&&n2==2)return p.c[17]/5;return p.c[18]/5;
}
__host__ __device__ inline int cp3_fact(int n){return n<2?1:(n==2?2:6);}
__host__ __device__ inline double cp3_full_energy(const ClumpP3&a,const ClumpP3&b,ClumpVec3 r,double G,double eps,int grad_axis) {
    double sum=0;for(int na=0;na<=3;++na)for(int nb=0;nb<=3;++nb){int ca=1,cb=1;for(int q=0;q<na;++q)ca*=3;for(int q=0;q<nb;++q)cb*=3;double scale=(na&1?-1.0:1.0)/(cp3_fact(na)*cp3_fact(nb));for(int ia=0;ia<ca;++ia)for(int ib=0;ib<cb;++ib){int n[3]={0,0,0},x=ia,y=ib;for(int q=0;q<na;++q){++n[x%3];x/=3;}for(int q=0;q<nb;++q){++n[y%3];y/=3;}if(grad_axis>=0)++n[grad_axis];sum+=scale*cp3_tensor(a,na,ia)*cp3_tensor(b,nb,ib)*cp3_green_derivative(r,eps,n[0],n[1],n[2]);}}return -G*sum;
}
// Full p0..p3 by p0..p3 scalar energy; force invokes derivatives through p7.
__host__ __device__ inline double clump_p3_pair_energy(const ClumpP3&a,const ClumpP3&b,ClumpVec3 r,double G,double eps){return cp3_full_energy(a,b,r,G,eps,-1);}
__host__ __device__ inline void clump_p3_pair_force(const ClumpP3&a,const ClumpP3&b,ClumpVec3 r,double G,double eps,ClumpVec3&fa,ClumpVec3&fb){fa={cp3_full_energy(a,b,r,G,eps,0),cp3_full_energy(a,b,r,G,eps,1),cp3_full_energy(a,b,r,G,eps,2)};fb=cp3_scale(fa,-1.0);}

// Orientation derivative of the same scalar energy.  It is intentionally
// evaluated from a centred infinitesimal rigid rotation so overrides using all
// 19 stored components retain an energy-consistent tidal torque.
__host__ __device__ inline ClumpVec3 clump_p3_pair_torque(const ClumpP3& a, const ClumpP3& b, ClumpVec3 r, double G, double eps, bool on_a) {
    const double h=1.e-6; ClumpVec3 tau{};
    for (int axis=0; axis<3; ++axis) {
        const double s=sin(0.5*h), c=cos(0.5*h);
        ClumpP3 plus = on_a ? clump_p3_rotate_body_to_world(a,c,axis==0?s:0.0,axis==1?s:0.0,axis==2?s:0.0) : a;
        ClumpP3 p_b = on_a ? b : clump_p3_rotate_body_to_world(b,c,axis==0?s:0.0,axis==1?s:0.0,axis==2?s:0.0);
        ClumpP3 minus = on_a ? clump_p3_rotate_body_to_world(a,c,axis==0?-s:0.0,axis==1?-s:0.0,axis==2?-s:0.0) : a;
        ClumpP3 m_b = on_a ? b : clump_p3_rotate_body_to_world(b,c,axis==0?-s:0.0,axis==1?-s:0.0,axis==2?-s:0.0);
        const double d=-(clump_p3_pair_energy(plus,p_b,r,G,eps)-clump_p3_pair_energy(minus,m_b,r,G,eps))/(2*h);
        if(axis==0)tau.x=d; else if(axis==1)tau.y=d; else tau.z=d;
    }
    return tau;
}

} // namespace body_fitted_fmm

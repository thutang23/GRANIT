// Independent member gravity oracle + fixed-state public API benchmark.
// The trace-free metadata below exists ONLY to run unchanged against the old
// API implementation. No production multipole/force primitive enters oracle().
#include "../../../../src/core/acceleration/acceleration.h"
#include "../../../../src/core/particle_model/clump.h"
#include "../../../../src/core/wall/wall.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <cuda.h>
#include <cufft.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include "../../../../src/domain/rcb_work_model.h"
// Match the production manager's inclusion contract. Its namespace block and
// component types are intentionally nested inside cudadem in manager.cu.
// All transitive system/STL headers and the self-namespaced domain header must
// be included globally first.
namespace cudadem {
#include "../../../../src/core/acceleration/body_fitted_fmm/body_fitted_fmm_manager.h"
}
using namespace cudadem;
namespace {
using Clock=std::chrono::steady_clock;
struct V { long double x=0,y=0,z=0; };
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(V a,long double b){return {a.x*b,a.y*b,a.z*b};}
long double norm(V a){return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> void upload(T* p,const std::vector<T>& v){if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> std::vector<T> download(const T* p,int n){std::vector<T> v(n);if(n)ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct Row {V p; float mass=1;int id=0;bool dead=false;};
struct Geometry {std::string name;std::vector<Row> rows;int normals=0,clumps=0,live=0;bool near=false;};
struct Options {std::string name="all",method="both",json;int size=128,clumps=2,depth=5,warmup=3,repeats=10,threshold_leaf=-1,threshold_internal=-1;bool benchmark=false,verify=true,baseline=false,project=false,plain_control=false,worst_body=false,implementation_check=false,clump_diagnostics=false;};
bool periodic_case(const std::string& name){return name=="periodic"||name=="differing_g_periodic";}
bool direct_fallback_case(const std::string& name){return name=="threshold_above"||name=="differing_g_fallback";}
bool differing_g_case(const std::string& name){return name=="differing_g"||name=="differing_g_fallback"||name=="differing_g_periodic";}
V transform(V p,int kind){if(kind==1)return p+V{7.125L,-3.75L,2.5L};if(kind==2){const long double c=std::cos(.71L),s=std::sin(.71L);return {c*p.x-s*p.y,s*p.x+c*p.y,p.z};}return p;}
void add_clump(Geometry& g,V center,double scale,int transform_kind=0){
    ++g.clumps;
    const std::array<V,4> offsets={V{-.78L,-.19L,.11L},V{.43L,-.31L,-.23L},V{.08L,.57L,.37L},V{.34L,.12L,-.46L}};
    const float masses[]={.625f,1.375f,.875f,1.125f};
    for(int i=0;i<4;++i)g.rows.push_back({transform(center+offsets[i]*scale,transform_kind),masses[i],g.clumps,false});
}
Geometry make_geometry(const std::string& name,int size,int clumps,int depth){
    if(differing_g_case(name)){
        const std::string geometry_name=name=="differing_g_fallback"?"threshold_above":(name=="differing_g_periodic"?"periodic":"small");
        auto g=make_geometry(geometry_name,size,clumps,depth);g.name=name;return g;
    }
    Geometry g;g.name=name;
    if(name=="tidal"){
        const double h=(32.+2.*(32.*.01))/4.;const V target={-16.-32.*.01+h*.5,-16.-32.*.01+h*.5,-16.-32.*.01+h*.5};
        g.rows.push_back({{-16,-16,-16},0.f,0});g.rows.push_back({{16,16,16},0.f,0});
        g.rows.push_back({target+V{2*h,h,0},1.f,0});g.normals=3;
        add_clump(g,{},.001*h);V mean;long double total=0;for(int i=g.normals;i<(int)g.rows.size();++i){mean=mean+g.rows[i].p*g.rows[i].mass;total+=g.rows[i].mass;}mean=mean*(1/total);
        const V com=target+V{.49*h,.245*h,0};for(int i=g.normals;i<(int)g.rows.size();++i)g.rows[i].p=g.rows[i].p-mean+com;
        g.live=g.rows.size();g.near=false;for(auto& row:g.rows)row.p={(double)row.p.x,(double)row.p.y,(double)row.p.z};return g;
    }
    const bool normal_only=name=="normals";
    const bool populated=name=="fmm"||name=="benchmark"||name=="large";
    const int kind=name=="translated"?1:(name=="rotated"?2:0);
    if(name=="periodic"){
        g.rows.push_back({{-.92L,.11L,.04L},1.25f,0});
        g.rows.push_back({{.88L,-.14L,.22L},.75f,0});
        g.rows.push_back({{.12L,.39L,-.30L},1.625f,0});
    }else if(name!="internal"){
        // Symmetric finite-mass anchors make the dynamic tree root and its
        // midpoint deterministic. The tiny clump crosses midpoint faces,
        // edges, or corners but its diameter remains well below leaf h.
        g.rows.push_back({transform({-16,-16,-16},kind),.0625f,0});
        g.rows.push_back({transform({16,16,16},kind),.0625f,0});
        if(populated){
            if(size<2+4*clumps)throw std::runtime_error("--size must include two anchors and four members per --clumps");
            const int n=size-2-4*clumps;
            for(int i=0;i<n;++i){
                // Irrational lattice avoids coincident sites and exact symmetry.
                const long double x=std::fmod((i+1)*.7548776662466927L,1.L)*24-12;
                const long double y=std::fmod((i+1)*.5698402909980532L,1.L)*24-12;
                const long double z=std::fmod((i+1)*.438579021L,1.L)*24-12;
                g.rows.push_back({{x,y,z},float(.5+(i%13)*.0625),0});
            }
        }else{
            g.rows.push_back({transform({.063L,.041L,.083L},kind),1.25f,0});
            g.rows.push_back({transform({-2.7L,.9L,1.8L},kind),.75f,0});
            g.rows.push_back({transform({4.1L,-3.2L,.7L},kind),1.625f,0});
        }
    }
    g.normals=g.rows.size();
    if(!normal_only){
        V center={.21L,.23L,.19L};
        if(name=="face")center={0,.23L,.19L};
        if(name=="edge")center={0,0,.19L};
        if(name=="corner")center={0,0,0};
        if(name=="internal")center={0,0,0};
        if(name=="periodic")center={.998L,.13L,-.20L};
        double scale=name=="large"?.9:.006;
        if(name=="threshold_below"||name=="threshold_above"){
            // The anchors fix the actual dynamic extent at 32 and the root at
            // 32+2*(32*.01), independently of these bounded four-member bodies.
            // Set R about the mass COM, including the uploaded sphere radius.
            Geometry unit;add_clump(unit,{},1.);V com;long double mass=0;
            for(const auto& row:unit.rows){com=com+row.p*row.mass;mass+=row.mass;}com=com*(1/mass);
            long double radius=0;for(const auto& row:unit.rows)radius=std::max(radius,norm(row.p-com));
            const double h=std::ldexp(32.+2.*(32.*.01),-depth);
            const double side=name=="threshold_below"?1.-1.e-6:1.+1.e-6;
            scale=(.5*h*side-(double).001f)/(double)radius;
            if(!(scale>0))throw std::runtime_error("threshold case sphere radius exceeds half leaf width");
            center={0,0,0};
        }
        add_clump(g,center,scale,kind);
        if(name!="internal"&&(!populated||clumps>1))add_clump(g,name=="periodic"?V{-.93L,-.28L,.34L}:V{-1.3L,-.7L,1.1L},scale*.83,kind);
        if(populated)for(int c=2;c<clumps;++c){
            const V center={std::fmod((c+1)*.6180339887498948L,1.L)*16-8,std::fmod((c+1)*.414213562373095L,1.L)*16-8,std::fmod((c+1)*.732050807568877L,1.L)*16-8};
            add_clump(g,center,scale*(.7+.1*(c%4)));
        }
    }
    g.live=g.rows.size();
    if(name=="dead"){
        g.rows.push_back({{.22L,.23L,.19L},1000,-1,true});
        // A canonical dead tail also includes a zero-ID absorbed row.
        g.rows.push_back({{.20L,.23L,.19L},1000,0,true});
    }
    // Match the actual uploaded FP64 coordinates in the CPU oracle.
    for(auto& row:g.rows)row.p={(double)row.p.x,(double)row.p.y,(double)row.p.z};
    g.near=!populated;
    return g;
}
// Independent Cartesian moments used only to populate baseline metadata.
void metadata(std::array<std::vector<double>,19>& a,int c,double mass,V r){
    const double x=r.x,y=r.y,z=r.z,x2=x*x,y2=y*y,z2=z*z,r2=x2+y2+z2;
    const double q[]={mass,mass*x,mass*y,mass*z,3*mass*(x2-y2),3*mass*(y2-z2),3*mass*x*y,3*mass*x*z,3*mass*y*z,
        mass*(5*x*x2-3*x*r2),mass*(5*y*y2-3*y*r2),mass*(5*z*z2-3*z*r2),mass*(5*x2*y-y*r2),mass*(5*x2*z-z*r2),mass*(5*y2*z-z*r2),5*mass*x*y*z,mass*(5*x*y2-x*r2),mass*(5*x*z2-x*r2),mass*(5*y*z2-y*r2)};
    for(int k=0;k<19;++k)a[k][c]+=q[k];
}
struct DeviceCase {
    SimulationState state;ClumpStateSoA cs;ClumpSegmentOffsets segments;
    std::vector<V> com;std::vector<double> masses;
    explicit DeviceCase(const Geometry& g){
        const int n=g.rows.size(),k=g.clumps;
        allocate_particle_soa(state.particles,n);state.particles.num_local=n;state.particles.num_ghost=0;
        auto& p=state.particles;
        std::vector<double>x(n),y(n),z(n);std::vector<float>m(n),r(n,.001f),ox(n),oy(n),oz(n),h(n,0.f);std::vector<int32_t>id(n);std::vector<uint8_t>dead(n);
        com.assign(k,{});masses.assign(k,0);
        for(const auto& row:g.rows)if(row.id>0&&!row.dead){com[row.id-1]=com[row.id-1]+row.p*row.mass;masses[row.id-1]+=row.mass;}
        for(int c=0;c<k;++c){com[c]=com[c]*(1/masses[c]);com[c]={(double)com[c].x,(double)com[c].y,(double)com[c].z};}
        std::array<std::vector<double>,19>a;for(auto& v:a)v.assign(k,0);
        for(int i=0;i<n;++i){const auto& row=g.rows[i];x[i]=row.p.x;y[i]=row.p.y;z[i]=row.p.z;m[i]=row.mass;id[i]=row.id;dead[i]=row.dead;
            if(row.id>0){const V arm=row.p-com[row.id-1];ox[i]=arm.x;oy[i]=arm.y;oz[i]=arm.z;metadata(a,row.id-1,row.mass,arm);}}
        upload(p.pos_x,x);upload(p.pos_y,y);upload(p.pos_z,z);upload(p.last_pos_x,x);upload(p.last_pos_y,y);upload(p.last_pos_z,z);
        upload(p.mass,m);upload(p.radius,r);upload(p.support_extent,r);upload(p.smoothing_length,h);upload(p.clump_id,id);upload(p.lifecycle_state,dead);
        upload(p.clump_off_x,ox);upload(p.clump_off_y,oy);upload(p.clump_off_z,oz);
        if(k){
            cs=allocate_clump_state(k);cs.num_clumps=k;segments=allocate_clump_segments(k);segments.num_segments=k;
            segments.normal_begin=0;segments.normal_end=g.normals;segments.member_begin=g.normals;segments.member_end=g.live;segments.dead_begin=g.live;segments.dead_end=n;
            std::vector<int32_t>b(k),e(k);std::vector<double>cx(k),cy(k),cz(k),zero(k,0),one(k,1);
            for(int c=0;c<k;++c){b[c]=4*c;e[c]=b[c]+4;cx[c]=com[c].x;cy[c]=com[c].y;cz[c]=com[c].z;}
            upload(segments.seg_begin,b);upload(segments.seg_end,e);upload(cs.com_x,cx);upload(cs.com_y,cy);upload(cs.com_z,cz);upload(cs.mass,masses);
            upload(cs.quat_w,one);upload(cs.quat_x,zero);upload(cs.quat_y,zero);upload(cs.quat_z,zero);
            upload(cs.inertia_xx,one);upload(cs.inertia_yy,one);upload(cs.inertia_zz,one);upload(cs.inertia_xy,zero);upload(cs.inertia_xz,zero);upload(cs.inertia_yz,zero);
            for(int q=0;q<19;++q)upload(cs.multipole_p3[q],a[q]);
        }
    }
    void clear(){auto& p=state.particles;for(double* v:{p.force_x,p.force_y,p.force_z,p.torque_x,p.torque_y,p.torque_z,p.gravity_force_x,p.gravity_force_y,p.gravity_force_z})ck(cudaMemset(v,0,p.num_particles*sizeof(double)));
        if(cs.num_clumps)for(double* v:{cs.net_force_x,cs.net_force_y,cs.net_force_z,cs.net_torque_x,cs.net_torque_y,cs.net_torque_z,cs.gravity_force_x,cs.gravity_force_y,cs.gravity_force_z,cs.gravity_torque_x,cs.gravity_torque_y,cs.gravity_torque_z})ck(cudaMemset(v,0,cs.num_clumps*sizeof(double)));}
    ~DeviceCase(){AccelerationManager::shutdown();WallManager::shutdown();if(cs.capacity)free_clump_state(cs);if(segments.seg_begin)free_clump_segments(segments);free_particle_soa(state.particles);}
};
struct Result {std::vector<V> f,t;std::vector<long double> fs,ts;};
Result oracle(const Geometry& g,const DeviceCase& d,long double eps,long double G=1){
    Result out;const int n=g.normals+g.clumps;out.f.resize(n);out.t.resize(n);out.fs.resize(n);out.ts.resize(n);
    for(int i=0;i<g.live;++i){const auto& a=g.rows[i];V f;
        for(int j=0;j<g.live;++j){const auto& b=g.rows[j];if(i==j||(a.id>0&&a.id==b.id))continue;
            V r=b.p-a.p;
            // Independent floor-based reduction; no half-box ties occur in
            // this geometry. The production kernel uses nearbyint instead.
            if(periodic_case(g.name))r.x-=2.L*std::floor((r.x+1.L)/2.L);
            const long double r2=r.x*r.x+r.y*r.y+r.z*r.z+eps*eps;
            f=f+r*(G*(long double)a.mass*b.mass/(r2*std::sqrt(r2)));}
        const int body=a.id>0?g.normals+a.id-1:i;
        const V arm=a.id>0?a.p-d.com[a.id-1]:V{};
        out.f[body]=out.f[body]+f;out.t[body]=out.t[body]+cross(arm,f);out.fs[body]+=norm(f);out.ts[body]+=norm(arm)*norm(f);
    }return out;
}
std::vector<V> member_oracle(const Geometry& g,long double eps,long double G=1){
    std::vector<V> forces(g.live);
    for(int i=0;i<g.live;++i)for(int j=0;j<g.live;++j){
        const auto& a=g.rows[i];const auto& b=g.rows[j];if(i==j||(a.id>0&&a.id==b.id))continue;
        V r=b.p-a.p;if(periodic_case(g.name))r.x-=2.L*std::floor((r.x+1.L)/2.L);
        const long double r2=r.x*r.x+r.y*r.y+r.z*r.z+eps*eps;
        forces[i]=forces[i]+r*(G*(long double)a.mass*b.mass/(r2*std::sqrt(r2)));
    }
    return forces;
}
std::vector<V> particle_force_rows(const Geometry& g,const DeviceCase& d,bool outer){
    const auto& p=d.state.particles;
    const auto x=download(outer?p.gravity_force_x:p.force_x,g.live),y=download(outer?p.gravity_force_y:p.force_y,g.live),z=download(outer?p.gravity_force_z:p.force_z,g.live);
    std::vector<V> result(g.live);for(int i=0;i<g.live;++i){const long double mass=outer?g.rows[i].mass:1.L;result[i]={x[i]*mass,y[i]*mass,z[i]*mass};}return result;
}
Result snapshot(const Geometry& g,const DeviceCase& d,bool outer){
    Result out;const int n=g.normals+g.clumps;out.f.resize(n);out.t.resize(n);
    const auto& p=d.state.particles;const auto& c=d.cs;
    auto x=download(outer?p.gravity_force_x:p.force_x,(int)g.rows.size()),y=download(outer?p.gravity_force_y:p.force_y,(int)g.rows.size()),z=download(outer?p.gravity_force_z:p.force_z,(int)g.rows.size());
    for(int i=0;i<g.normals;++i){const double m=outer?g.rows[i].mass:1.;out.f[i]={x[i]*m,y[i]*m,z[i]*m};}
    if(g.clumps){auto cx=download(outer?c.gravity_force_x:c.net_force_x,g.clumps),cy=download(outer?c.gravity_force_y:c.net_force_y,g.clumps),cz=download(outer?c.gravity_force_z:c.net_force_z,g.clumps);
        auto tx=download(outer?c.gravity_torque_x:c.net_torque_x,g.clumps),ty=download(outer?c.gravity_torque_y:c.net_torque_y,g.clumps),tz=download(outer?c.gravity_torque_z:c.net_torque_z,g.clumps);
        for(int i=0;i<g.clumps;++i){out.f[g.normals+i]={cx[i],cy[i],cz[i]};out.t[g.normals+i]={tx[i],ty[i],tz[i]};}}
    return out;
}
// Aggregate the unfiltered plain FMM's actual member rows on the host.
// Long-double sums reduce host cancellation error, but cannot undo rounding
// from the large internal clump forces already present in each GPU row.
Result aggregate_member_rows(const Geometry& g,const DeviceCase& d){
    Result out;out.f.resize(g.normals+g.clumps);out.t.resize(g.normals+g.clumps);
    const auto& p=d.state.particles;const auto x=download(p.force_x,g.live),y=download(p.force_y,g.live),z=download(p.force_z,g.live);
    for(int i=0;i<g.live;++i){const auto& row=g.rows[i];const int b=row.id>0?g.normals+row.id-1:i;const V f={x[i],y[i],z[i]};
        const V arm=row.id>0?row.p-d.com[row.id-1]:V{};out.f[b]=out.f[b]+f;out.t[b]=out.t[b]+cross(arm,f);}
    return out;
}
struct WorstBody {int index=-1;V reference,actual,position;long double normalized=0,absolute=0;};
WorstBody worst_force(const Result& value,const Result& reference,const Geometry& g,const DeviceCase& d){
    WorstBody worst;
    for(size_t i=0;i<value.f.size();++i){const long double absolute=norm(value.f[i]-reference.f[i]);const long double scale=reference.fs.empty()?norm(reference.f[i]):reference.fs[i];const long double relative=absolute/std::max(scale,1.e-28L);
        if(worst.index<0||relative>worst.normalized){worst.index=i;worst.reference=reference.f[i];worst.actual=value.f[i];worst.position=i<(size_t)g.normals?g.rows[i].p:d.com[i-g.normals];worst.normalized=relative;worst.absolute=absolute;}}
    return worst;
}
struct Metrics {long double force_p50=0,force_p90=0,force_p99=0,force_rms=0,force_max_body=0,force_max_clump=0,torque_max_clump=0,force=0,torque=0,resultant_torque=0,force_abs=0,torque_abs=0,balance=0,angular_balance=0;bool finite=true;};
Metrics compare(const Result& value,const Result& ref,const Geometry& g,const DeviceCase& d){
    Metrics m;std::vector<long double> normalized_force_errors;normalized_force_errors.reserve(value.f.size());long double force_scale=0,moment_scale=0,total_force_error=0,total_torque_error=0,torque_resultant=0,angular_scale=0;V force,angular;
    for(size_t i=0;i<value.f.size();++i){const long double fe=norm(value.f[i]-ref.f[i]),te=norm(value.t[i]-ref.t[i]);
        m.finite=m.finite&&std::isfinite(fe)&&std::isfinite(te);m.force_abs=std::max(m.force_abs,fe);m.torque_abs=std::max(m.torque_abs,te);
        const long double fs=ref.fs.empty()?norm(ref.f[i]):ref.fs[i],ts=ref.ts.empty()?norm(ref.t[i]):ref.ts[i];
        const long double fn=fe/std::max(fs,1.e-28L),tn=te/std::max(ts,1.e-28L);
        normalized_force_errors.push_back(fn);m.force_rms+=fn*fn;m.force_max_body=std::max(m.force_max_body,fn);
        if(i>=(size_t)g.normals){m.force_max_clump=std::max(m.force_max_clump,fn);m.torque_max_clump=std::max(m.torque_max_clump,tn);}
        force_scale+=fs;moment_scale+=ts;total_force_error+=fe;total_torque_error+=te;torque_resultant+=norm(ref.t[i]);
        force=force+value.f[i];const V pos=i<(size_t)g.normals?g.rows[i].p:d.com[i-g.normals];angular=angular+cross(pos,value.f[i])+value.t[i];angular_scale+=norm(pos)*norm(ref.f[i])+norm(ref.t[i]);}
    // Deterministic empirical nearest-rank quantiles: sort ascending and use
    // rank ceil(percent*N/100), one-based, without interpolation. Integer rank
    // arithmetic avoids floating-point ambiguity at exact percentile ranks.
    std::sort(normalized_force_errors.begin(),normalized_force_errors.end());
    if(!normalized_force_errors.empty()){auto quantile=[&](size_t percent){const size_t rank=(percent*normalized_force_errors.size()+99)/100;return normalized_force_errors[rank-1];};m.force_p50=quantile(50);m.force_p90=quantile(90);m.force_p99=quantile(99);}
    const long double floor=1.e-28L;m.force=total_force_error/std::max(force_scale,floor);m.torque=total_torque_error/std::max(moment_scale,floor);m.resultant_torque=total_torque_error/std::max(torque_resultant,floor);m.balance=norm(force)/std::max(force_scale,floor);m.angular_balance=norm(angular)/std::max(angular_scale,floor);m.force_rms=std::sqrt(m.force_rms/std::max((size_t)1,value.f.size()));return m;
}
struct Workload {
    bool valid=false,metadata_read=false,all_clumps_small=true;
    int view_particles=0,num_nodes=0,depth=0,tasks[4]={};
    double domain_min[3]={};double root_size=0,leaf_width=0,independent_root_size=0,independent_leaf_width=0,max_diameter_to_leaf=0;
};
Workload workload(const Geometry& g,const DeviceCase& d,int requested_depth){
    Workload out;
    const auto view=::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::get_workload_view();
    out.valid=view.valid;out.view_particles=view.num_particles;
    if(view.valid&&view.metadata){
        ::cudadem::body_fitted_fmm::PersistentTreeMetadata metadata;
        ck(cudaMemcpy(&metadata,view.metadata,sizeof(metadata),cudaMemcpyDeviceToHost));
        out.metadata_read=true;out.num_nodes=metadata.num_nodes;out.depth=metadata.evaluation_depth;
        out.domain_min[0]=metadata.domain_min_x;out.domain_min[1]=metadata.domain_min_y;out.domain_min[2]=metadata.domain_min_z;out.root_size=metadata.root_cell_size;out.leaf_width=std::ldexp(metadata.root_cell_size,-metadata.evaluation_depth);
        for(int k=0;k<4;++k)out.tasks[k]=metadata.task_counts[k];
    }
    // Independent copy of the root geometry calculation, over actual live
    // uploaded coordinates. This classification does not infer dispatch mode.
    double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
    for(int i=0;i<g.live;++i){const double p[3]={(double)g.rows[i].p.x,(double)g.rows[i].p.y,(double)g.rows[i].p.z};for(int k=0;k<3;++k){lo[k]=std::min(lo[k],p[k]);hi[k]=std::max(hi[k],p[k]);}}
    double extent=std::max({hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]});if(extent<1.e-30)extent=1.;
    out.independent_root_size=extent+2.*(extent*.01);out.independent_leaf_width=std::ldexp(out.independent_root_size,-requested_depth);
    std::vector<double> radii(g.clumps,0.);
    for(int i=g.normals;i<g.live;++i){const auto& row=g.rows[i];const auto center=d.com[row.id-1];const double dx=(double)row.p.x-(double)center.x,dy=(double)row.p.y-(double)center.y,dz=(double)row.p.z-(double)center.z;
        radii[row.id-1]=std::max(radii[row.id-1],std::sqrt(dx*dx+dy*dy+dz*dz)+(double).001f);}
    for(double radius:radii){out.max_diameter_to_leaf=std::max(out.max_diameter_to_leaf,2.*radius/out.independent_leaf_width);if(!(2.*radius<out.independent_leaf_width*(1.-64.*std::numeric_limits<double>::epsilon())))out.all_clumps_small=false;}
    return out;
}
using Cell=std::array<int,3>;
Cell member_cell(V p,const Workload& work){return {(int)std::floor(((double)p.x-work.domain_min[0])/work.leaf_width),(int)std::floor(((double)p.y-work.domain_min[1])/work.leaf_width),(int)std::floor(((double)p.z-work.domain_min[2])/work.leaf_width)};}
struct ClumpDiagnostics {bool checked=false,workload_matches=false;std::vector<V> reference,actual;Result aggregate;Workload work;};
#ifdef GRANIT_CLUMP_MEMBER_TEST
ClumpDiagnostics collect_clump_diagnostics(const Geometry& g,DeviceCase& d,const SimConfig& config,long double eps,int depth){
    ClumpDiagnostics out;out.checked=true;SimConfig raw=config;raw.acceleration.unified_fmm.enable_global_projection=false;
    SimulationState members=d.state;members.particles.num_particles=g.live;members.particles.num_local=g.live;members.particles.num_ghost=0;
    d.clear();::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute(members,raw,members.neighbor,WallManager::get_pbc_config(),0,0,true,members.particles.clump_id);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    out.reference=member_oracle(g,eps,config.acceleration.unified_fmm.gravitational_constant);out.actual=particle_force_rows(g,d,false);out.aggregate=aggregate_member_rows(g,d);out.work=workload(g,d,depth);
    return out;
}
#endif
struct TidalDiagnostic {bool checked=false,geometry_gate=false,source_at_cell_center=false,same_target_leaf=false;int distinct_leaves=0;V target_center,source_position,desired_com,p4_force,p4_torque,production_force,production_torque,direct_force,direct_torque;long double force_scale=0,moment_scale=0,h=0;std::vector<V> p4_member_force;};
V tidal_quartic_gradient(V R,V r){
    const long double a=R.x*r.x+R.y*r.y+R.z*r.z,b=r.x*r.x+r.y*r.y+r.z*r.z,D=R.x*R.x+R.y*R.y+R.z*R.z;
    const long double d3=D*std::sqrt(D),d5=d3*D,d7=d5*D,d9=d7*D;
    return R*(1/d3)+(R*(3*a)-r*D)*(1/d5)+(R*(15*a*a-3*D*b)-r*(6*D*a))*(1/(2*d7))+(R*(140*a*a*a-60*D*a*b)+r*(-60*D*a*a+12*D*D*b))*(1/(8*d9));
}
TidalDiagnostic tidal_diagnostic(const Geometry& g,const DeviceCase& d,const Workload& work,const Result& candidate,const Result& direct){
    TidalDiagnostic out;out.checked=true;
    const long double expected_h=(32.L+2.L*(32.L*.01L))/4.L;
    out.h=work.metadata_read?work.leaf_width:expected_h;
    out.target_center=work.metadata_read?V{work.domain_min[0]+out.h*.5L,work.domain_min[1]+out.h*.5L,work.domain_min[2]+out.h*.5L}:V{-16.32L+out.h*.5L,-16.32L+out.h*.5L,-16.32L+out.h*.5L};
    out.source_position=g.rows[2].p;out.desired_com=out.target_center+V{.49L*out.h,.245L*out.h,0};
    std::set<Cell> leaves;if(work.metadata_read)for(int i=g.normals;i<g.live;++i)leaves.insert(member_cell(g.rows[i].p,work));
    out.distinct_leaves=leaves.size();out.same_target_leaf=leaves.size()==1&&*leaves.begin()==Cell{0,0,0};
    const long double tolerance=1.e-12L;
    out.source_at_cell_center=norm(out.source_position-(out.target_center+V{2*out.h,out.h,0}))<tolerance;
    out.geometry_gate=work.valid&&work.metadata_read&&work.depth==2&&(work.tasks[0]>0||work.tasks[1]>0)&&out.same_target_leaf&&out.source_at_cell_center&&std::fabs(out.h-expected_h)<tolerance;
    if(work.metadata_read)for(int k=0;k<3;++k)out.geometry_gate=out.geometry_gate&&std::fabs(work.domain_min[k]+16.32L)<tolerance;
    out.p4_member_force.resize(g.live);const V R=out.source_position-out.target_center;
    for(int i=g.normals;i<g.live;++i){const V f=tidal_quartic_gradient(R,g.rows[i].p-out.target_center)*((long double)g.rows[i].mass*g.rows[2].mass);out.p4_member_force[i]=f;out.p4_force=out.p4_force+f;out.p4_torque=out.p4_torque+cross(g.rows[i].p-d.com[0],f);}
    out.production_force=candidate.f[g.normals];out.production_torque=candidate.t[g.normals];out.direct_force=direct.f[g.normals];out.direct_torque=direct.t[g.normals];out.force_scale=direct.fs[g.normals];out.moment_scale=direct.ts[g.normals];return out;
}
long double implementation_tolerance(){
#ifdef FMM_P2P_LOCAL_FP32
    return 1.e-6L;
#else
    return 1.e-10L;
#endif
}
bool same_workload(const Workload& a,const Workload& b){
    if(!a.valid||!b.valid||!a.metadata_read||!b.metadata_read||a.root_size!=b.root_size||a.depth!=b.depth||a.num_nodes!=b.num_nodes||a.view_particles!=b.view_particles)return false;
    for(int k=0;k<3;++k)if(a.domain_min[k]!=b.domain_min[k])return false;
    for(int k=0;k<4;++k)if(a.tasks[k]!=b.tasks[k])return false;
    return true;
}
struct CancellationFreeControl {
    bool checked=false,workloads_match=true,masses_restored=true,gate_pass=false;
    Metrics accuracy,difference;std::vector<Workload> runs;
};
CancellationFreeControl cancellation_free_control(const Geometry& g,DeviceCase& d,const SimConfig& config,const Workload& candidate_work,const Result& candidate,const Result& expected,int depth){
    CancellationFreeControl out;out.checked=true;
    const auto original=download(d.state.particles.mass,(int)g.rows.size());
    // Restore masses during exception unwinding too; every subsequent public
    // gravity call and every benchmark must see the original physical state.
    struct MassRestore {float* p;const std::vector<float>& original;bool active=true;~MassRestore(){if(active)cudaMemcpy(p,original.data(),original.size()*sizeof(float),cudaMemcpyHostToDevice);}} restore{d.state.particles.mass,original};
    SimConfig plain=config;plain.acceleration.unified_fmm.enable_global_projection=false;
    Result control;control.f.resize(g.normals+g.clumps);control.t.resize(g.normals+g.clumps);
    auto outer=[&](){d.clear();::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute_outer(d.state,plain,d.state.neighbor,WallManager::get_pbc_config(),0,0);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
        const auto current=workload(g,d,depth);out.workloads_match=out.workloads_match&&same_workload(candidate_work,current);out.runs.push_back(current);return particle_force_rows(g,d,true);};
    // Ordinary normal targets use INNER with all source masses retained,
    // matching the candidate's target-mass multiplication and FP32 rounding.
    // Clump targets below still use OUTER with zero own-source masses, then
    // convert acceleration using the immutable original target masses.
    d.clear();::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute(d.state,plain,d.state.neighbor,WallManager::get_pbc_config(),0,0,true);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    const auto full_work=workload(g,d,depth);out.workloads_match=out.workloads_match&&same_workload(candidate_work,full_work);out.runs.push_back(full_work);
    const auto full=particle_force_rows(g,d,false);for(int i=0;i<g.normals;++i)control.f[i]=full[i];
    for(int c=0;c<g.clumps;++c){
        auto sources=original;for(int i=g.normals;i<g.live;++i)if(g.rows[i].id==c+1)sources[i]=0.f;upload(d.state.particles.mass,sources);
        const auto external=outer();const int b=g.normals+c;
        for(int i=g.normals;i<g.live;++i)if(g.rows[i].id==c+1){control.f[b]=control.f[b]+external[i];control.t[b]=control.t[b]+cross(g.rows[i].p-d.com[c],external[i]);}
    }
    upload(d.state.particles.mass,original);restore.active=false;
    const auto restored=download(d.state.particles.mass,(int)g.rows.size());out.masses_restored=std::memcmp(original.data(),restored.data(),original.size()*sizeof(float))==0;
    out.accuracy=compare(control,expected,g,d);Result scaled_candidate=candidate;scaled_candidate.fs=expected.fs;scaled_candidate.ts=expected.ts;out.difference=compare(control,scaled_candidate,g,d);
    const long double tolerance=implementation_tolerance();out.gate_pass=out.workloads_match&&out.masses_restored&&out.accuracy.finite&&out.difference.finite&&out.difference.force<=tolerance&&out.difference.torque<=tolerance&&out.difference.force_max_body<=tolerance&&out.difference.torque_max_clump<=tolerance;
    return out;
}
struct MemberGate {bool checked=false,finite=true,gate_pass=false;long double inner_max=0,outer_max=0,equivalence_max=0,inner_abs=0,outer_abs=0;int worst_inner=-1,worst_outer=-1;};
#ifdef GRANIT_CLUMP_MEMBER_TEST
MemberGate filtered_member_gate(const Geometry& g,DeviceCase& d,const SimConfig& config,long double epsilon){
    MemberGate out;out.checked=true;SimConfig raw=config;raw.acceleration.unified_fmm.enable_global_projection=false;
    SimulationState members=d.state;members.particles.num_particles=g.live;members.particles.num_local=g.live;members.particles.num_ghost=0;
    d.clear();::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute(members,raw,members.neighbor,WallManager::get_pbc_config(),0,0,true,members.particles.clump_id);ck(cudaGetLastError());ck(cudaDeviceSynchronize());const auto inner=particle_force_rows(g,d,false);
    d.clear();::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute_outer(members,raw,members.neighbor,WallManager::get_pbc_config(),0,0,members.particles.clump_id);ck(cudaGetLastError());ck(cudaDeviceSynchronize());const auto outer=particle_force_rows(g,d,true);
    const auto reference=member_oracle(g,epsilon,config.acceleration.unified_fmm.gravitational_constant);
    for(int i=0;i<g.live;++i){const long double scale=std::max(norm(reference[i]),1.e-28L),ie=norm(inner[i]-reference[i]),oe=norm(outer[i]-reference[i]);
        out.finite=out.finite&&std::isfinite(ie)&&std::isfinite(oe);if(out.worst_inner<0||ie/scale>out.inner_max){out.inner_max=ie/scale;out.worst_inner=i;}if(out.worst_outer<0||oe/scale>out.outer_max){out.outer_max=oe/scale;out.worst_outer=i;}
        out.inner_abs=std::max(out.inner_abs,ie);out.outer_abs=std::max(out.outer_abs,oe);out.equivalence_max=std::max(out.equivalence_max,norm(inner[i]-outer[i])/scale);}
    const long double tolerance=implementation_tolerance();out.gate_pass=out.finite&&out.inner_max<=tolerance&&out.outer_max<=tolerance&&out.equivalence_max<=tolerance;return out;
}
#endif
void workload_json(std::ostream& s,const Workload& w){
    s<<"{\"valid\":"<<(w.valid?"true":"false")<<",\"metadata_read\":"<<(w.metadata_read?"true":"false")<<",\"view_num_particles\":"<<w.view_particles<<",\"num_nodes\":";
    if(w.metadata_read)s<<w.num_nodes;else s<<"null";
    s<<",\"evaluation_depth\":";if(w.metadata_read)s<<w.depth;else s<<"null";
    s<<",\"task_counts\":";if(w.metadata_read)s<<'['<<w.tasks[0]<<','<<w.tasks[1]<<','<<w.tasks[2]<<','<<w.tasks[3]<<']';else s<<"null";
    s<<",\"domain_min\":";if(w.metadata_read)s<<'['<<w.domain_min[0]<<','<<w.domain_min[1]<<','<<w.domain_min[2]<<']';else s<<"null";
    s<<",\"root_size\":";if(w.metadata_read)s<<w.root_size;else s<<"null";
    s<<",\"leaf_width\":";if(w.metadata_read)s<<w.leaf_width;else s<<"null";
    s<<",\"independent_root_size\":"<<w.independent_root_size<<",\"independent_leaf_width\":"<<w.independent_leaf_width<<",\"max_clump_diameter_to_leaf_width\":"<<w.max_diameter_to_leaf<<",\"all_clumps_small\":"<<(w.all_clumps_small?"true":"false")<<'}';
}
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])*.5;}
void number(std::ostream& s,long double x){if(std::isfinite(x))s<<(double)x;else s<<"null";}
void vector_json(std::ostream& s,V p){s<<'[';number(s,p.x);s<<',';number(s,p.y);s<<',';number(s,p.z);s<<']';}
void worst_json(std::ostream& s,const WorstBody& w,const Geometry& g){
    if(w.index<0){s<<"null";return;}
    s<<"{\"body_index\":"<<w.index<<",\"normal_particle_index\":";if(w.index<g.normals)s<<w.index;else s<<"null";
    s<<",\"clump_id\":"<<(w.index<g.normals?0:w.index-g.normals+1)<<",\"position\":";vector_json(s,w.position);s<<",\"reference_force\":";vector_json(s,w.reference);s<<",\"actual_force\":";vector_json(s,w.actual);
    s<<",\"absolute_error\":";number(s,w.absolute);s<<",\"normalized_error\":";number(s,w.normalized);s<<'}';
}
void metric_json(std::ostream& s,const Metrics& m){s<<"{\"force_body_quantile_method\":\"nearest_rank\",\"force_body_p50_normalized\":";number(s,m.force_p50);s<<",\"force_body_p90_normalized\":";number(s,m.force_p90);s<<",\"force_body_p99_normalized\":";number(s,m.force_p99);s<<",\"force_body_rms_normalized\":";number(s,m.force_rms);s<<",\"force_body_max_normalized\":";number(s,m.force_max_body);s<<",\"force_clump_max_normalized\":";number(s,m.force_max_clump);s<<",\"torque_clump_max_moment_normalized\":";number(s,m.torque_max_clump);s<<",\"force_normalized\":";number(s,m.force);s<<",\"torque_moment_normalized\":";number(s,m.torque);s<<",\"torque_resultant_relative\":";number(s,m.resultant_torque);s<<",\"force_max_abs\":";number(s,m.force_abs);s<<",\"torque_max_abs\":";number(s,m.torque_abs);s<<",\"net_force_normalized\":";number(s,m.balance);s<<",\"net_angular_momentum_rate_normalized\":";number(s,m.angular_balance);s<<",\"finite\":"<<(m.finite?"true":"false")<<"}";}
void cell_json(std::ostream& s,Cell cell){s<<'['<<cell[0]<<','<<cell[1]<<','<<cell[2]<<']';}
void wrench_error_json(std::ostream& s,V actual_f,V actual_t,V reference_f,V reference_t,long double force_scale,long double moment_scale){
    s<<"{\"force_error_vector\":";vector_json(s,actual_f-reference_f);s<<",\"torque_error_vector\":";vector_json(s,actual_t-reference_t);
    s<<",\"force_normalized\":";number(s,norm(actual_f-reference_f)/std::max(force_scale,1.e-28L));s<<",\"force_resultant_relative\":";number(s,norm(actual_f-reference_f)/std::max(norm(reference_f),1.e-28L));s<<",\"torque_moment_normalized\":";number(s,norm(actual_t-reference_t)/std::max(moment_scale,1.e-28L));
    s<<",\"torque_resultant_relative\":";number(s,norm(actual_t-reference_t)/std::max(norm(reference_t),1.e-28L));s<<'}';
}
void clump_diagnostics_json(std::ostream& s,const ClumpDiagnostics& diagnostic,const Geometry& g,const DeviceCase& d,const Result& candidate,const Result& direct,const TidalDiagnostic& tidal){
    s<<"{\"checked\":"<<(diagnostic.checked?"true":"false");if(!diagnostic.checked){s<<'}';return;}
    s<<",\"workload_matches_candidate\":"<<(diagnostic.workload_matches?"true":"false")<<",\"clumps\":[";
    for(int c=0;c<g.clumps;++c){if(c)s<<',';const int b=g.normals+c;std::set<Cell> cells;long double radius=0;
        for(int i=g.normals;i<g.live;++i)if(g.rows[i].id==c+1){cells.insert(member_cell(g.rows[i].p,diagnostic.work));radius=std::max(radius,norm(g.rows[i].p-d.com[c])+(long double).001f);}
        s<<"{\"clump_id\":"<<c+1<<",\"com\":";vector_json(s,d.com[c]);s<<",\"bounding_radius\":";number(s,radius);s<<",\"distinct_leaf_count\":"<<cells.size()<<",\"force_normalization_scale\":";number(s,direct.fs[b]);s<<",\"moment_normalization_scale\":";number(s,direct.ts[b]);
        s<<",\"reference_force\":";vector_json(s,direct.f[b]);s<<",\"actual_force\":";vector_json(s,candidate.f[b]);s<<",\"reference_torque\":";vector_json(s,direct.t[b]);s<<",\"actual_torque\":";vector_json(s,candidate.t[b]);s<<",\"production_error\":";wrench_error_json(s,candidate.f[b],candidate.t[b],direct.f[b],direct.t[b],direct.fs[b],direct.ts[b]);
        s<<",\"force_from_filtered_members\":";vector_json(s,diagnostic.aggregate.f[b]);s<<",\"torque_from_filtered_members\":";vector_json(s,diagnostic.aggregate.t[b]);s<<",\"member_aggregation_minus_production\":";wrench_error_json(s,diagnostic.aggregate.f[b],diagnostic.aggregate.t[b],candidate.f[b],candidate.t[b],direct.fs[b],direct.ts[b]);
        s<<",\"members\":[";bool first=true;for(int i=g.normals;i<g.live;++i)if(g.rows[i].id==c+1){if(!first)s<<',';first=false;s<<"{\"particle_index\":"<<i<<",\"mass\":"<<g.rows[i].mass<<",\"sphere_radius\":"<<(double).001f<<",\"position\":";vector_json(s,g.rows[i].p);s<<",\"cell\":";cell_json(s,member_cell(g.rows[i].p,diagnostic.work));s<<",\"reference_force\":";vector_json(s,diagnostic.reference[i]);s<<",\"actual_force\":";vector_json(s,diagnostic.actual[i]);s<<",\"force_error\":";vector_json(s,diagnostic.actual[i]-diagnostic.reference[i]);s<<",\"force_normalized_error\":";number(s,norm(diagnostic.actual[i]-diagnostic.reference[i])/std::max(norm(diagnostic.reference[i]),1.e-28L));if(tidal.checked){s<<",\"analytic_p4_force\":";vector_json(s,tidal.p4_member_force[i]);}s<<'}';}s<<"]}";
    }s<<"]}";
}
void tidal_json(std::ostream& s,const TidalDiagnostic& t){
    s<<"{\"geometry_gate_pass\":"<<(t.geometry_gate?"true":"false")<<",\"distinct_member_leaves\":"<<t.distinct_leaves<<",\"all_members_in_target_leaf_zero\":"<<(t.same_target_leaf?"true":"false")<<",\"source_at_cell_center\":"<<(t.source_at_cell_center?"true":"false")<<",\"h\":";number(s,t.h);s<<",\"target_leaf_center\":";vector_json(s,t.target_center);s<<",\"source_position\":";vector_json(s,t.source_position);s<<",\"desired_com\":";vector_json(s,t.desired_com);s<<",\"force_normalization_scale\":";number(s,t.force_scale);s<<",\"moment_normalization_scale\":";number(s,t.moment_scale);
    s<<",\"analytic_p4_force\":";vector_json(s,t.p4_force);s<<",\"analytic_p4_torque\":";vector_json(s,t.p4_torque);s<<",\"production_force\":";vector_json(s,t.production_force);s<<",\"production_torque\":";vector_json(s,t.production_torque);s<<",\"direct_force\":";vector_json(s,t.direct_force);s<<",\"direct_torque\":";vector_json(s,t.direct_torque);
    s<<",\"analytic_p4_minus_production\":";wrench_error_json(s,t.p4_force,t.p4_torque,t.production_force,t.production_torque,t.force_scale,t.moment_scale);s<<",\"analytic_p4_minus_direct\":";wrench_error_json(s,t.p4_force,t.p4_torque,t.direct_force,t.direct_torque,t.force_scale,t.moment_scale);s<<",\"production_minus_direct\":";wrench_error_json(s,t.production_force,t.production_torque,t.direct_force,t.direct_torque,t.force_scale,t.moment_scale);s<<'}';
}
bool run(const Geometry& g,const Options& o,SelfGravityMethod method,std::ostream& json){
    const bool fmm=method==SelfGravityMethod::BFFMM;
    double equivalence_force_tolerance=1.e-9;
    const char* equivalence_tolerance_reason="same_precision_force_comparison";
#ifdef FMM_P2P_LOCAL_FP32
    if(fmm&&g.near&&g.clumps==0){
        // Ordinary inner P2P multiplies by target mass in float before its
        // float sum. Ordinary OUTER sums acceleration in float, followed by
        // host FP64 mass conversion. Registered clumps reuse the inner buffer.
        equivalence_force_tolerance=1.e-6;
        equivalence_tolerance_reason="local_fp32_unclumped_inner_mass_product_vs_outer_acceleration_conversion";
    }
#endif
    SimConfig config;config.acceleration.enable_self_gravity=true;config.acceleration.gravity_method=method;config.acceleration.direct_gravity.gravitational_constant=1;
    auto& f=config.acceleration.unified_fmm;f.gravitational_constant=1;f.tree_depth=o.depth;f.adaptive_depth.enabled=false;f.adaptive_depth.initial_depth_auto=false;f.adaptive_depth.tree_depth_was_omitted=false;f.enable_global_projection=o.project;f.verbose=false;
    if(differing_g_case(g.name)){config.acceleration.direct_gravity.gravitational_constant=2.;f.gravitational_constant=3.;}
    const double oracle_G=fmm?f.gravitational_constant:config.acceleration.direct_gravity.gravitational_constant;
    // Reproduce the float configuration product before promoting epsilon.
    config.particle.radius_min=.001f;f.softening_factor=.001f;f.direct_threshold_leaf=g.near?100000000:0;f.direct_threshold_internal=g.near?100000000:0;
    if(!g.near){if(o.threshold_leaf>=0)f.direct_threshold_leaf=o.threshold_leaf;if(o.threshold_internal>=0)f.direct_threshold_internal=o.threshold_internal;}
    if(g.name=="tidal"){f.softening_factor=0.f;f.direct_threshold_leaf=0;f.direct_threshold_internal=0;f.enable_global_projection=false;}
    // Force a finite far interaction population for the dedicated FMM cases.
    f.theta_far=.8f;
    if(periodic_case(g.name)){config.num_walls=1;auto& wall=config.walls[0];wall.enabled=true;wall.type=WallType::PERIODICAL;wall.axis=0;wall.bound_min=-1.f;wall.bound_max=1.f;}
    DeviceCase d(g);d.clear();const auto start=Clock::now();
    WallManager::initialize(d.state,config);AccelerationManager::initialize(config);AccelerationManager::set_clump_runtime(g.clumps?&d.cs:nullptr,g.clumps?&d.segments:nullptr);ck(cudaDeviceSynchronize());
    const double init_ms=ms(start,Clock::now());
    auto compute=[&](bool outer){if(outer)AccelerationManager::compute_outer_gravity(d.state,config,0);else AccelerationManager::compute_self_gravity(d.state,config,0);ck(cudaGetLastError());};
    const auto first=Clock::now();compute(false);ck(cudaDeviceSynchronize());const double first_ms=ms(first,Clock::now());
    const auto work=workload(g,d,o.depth);bool structure_pass=true,structure_checked=false;
    ClumpDiagnostics clump_details;TidalDiagnostic tidal;Result diagnostic_candidate,diagnostic_reference;
    CancellationFreeControl independent_control;MemberGate member_gate;bool implementation_pass=false,numeric_gate_pass=false;
    Metrics inner,outer,equivalence,control_accuracy,control_difference;Workload control_work;bool control_checked=false;long double max_normal_control_difference=0;WorstBody worst_inner,worst_outer,worst_control;bool pass=true;double dead_max_abs=0;
    bool helper_checked=false,normal_force_unchanged=true;long double helper_force_error=0,helper_torque_error=0;
    auto dead_residual=[&](bool outer){double worst=0;const auto& p=d.state.particles;
        for(const double* field:{outer?p.gravity_force_x:p.force_x,outer?p.gravity_force_y:p.force_y,outer?p.gravity_force_z:p.force_z}){const auto values=download(field,(int)g.rows.size());for(size_t i=g.live;i<g.rows.size();++i)worst=std::max(worst,std::fabs(values[i]));}return worst;};
    const bool reference=o.verify&&g.live<=8192;
    if(o.verify&&!reference)throw std::runtime_error("reference limited to 8192 live rows; use --no-verify for larger benchmark");
    if(reference){const long double eps=!fmm&&!g.clumps?0.L:(double)(config.particle.radius_min*f.softening_factor);
        const auto expected=oracle(g,d,eps,oracle_G);const auto inside=snapshot(g,d,false);inner=compare(inside,expected,g,d);dead_max_abs=dead_residual(false);
        if(o.worst_body||o.plain_control)worst_inner=worst_force(inside,expected,g,d);
        if(o.plain_control&&fmm&&(g.name=="fmm"||g.name=="benchmark")){
            if(o.project||!work.all_clumps_small)throw std::runtime_error("plain FMM control requires raw nonperiodic all-small member geometry");
            control_checked=true;d.clear();SimConfig plain_config=config;plain_config.acceleration.unified_fmm.enable_global_projection=false;
            // Omitting the optional membership argument invokes its nullptr
            // default in the new API and remains compatible with the old API.
            ::cudadem::body_fitted_fmm::UnifiedFMMDispatcher::compute(d.state,plain_config,d.state.neighbor,WallManager::get_pbc_config(),0,0,true);
            ck(cudaGetLastError());ck(cudaDeviceSynchronize());
            const auto control=aggregate_member_rows(g,d);control_accuracy=compare(control,expected,g,d);
            Result candidate_scaled=inside;candidate_scaled.fs=expected.fs;candidate_scaled.ts=expected.ts;
            control_difference=compare(control,candidate_scaled,g,d);worst_control=worst_force(control,expected,g,d);control_work=workload(g,d,o.depth);
            for(int i=0;i<g.normals;++i)max_normal_control_difference=std::max(max_normal_control_difference,norm(control.f[i]-inside.f[i]));
            AccelerationManager::set_clump_runtime(g.clumps?&d.cs:nullptr,g.clumps?&d.segments:nullptr);
        }
        if(o.implementation_check&&fmm&&(g.name=="fmm"||g.name=="benchmark")){
            if(o.project||!work.all_clumps_small)throw std::runtime_error("implementation control requires raw all-small nonperiodic geometry");
            independent_control=cancellation_free_control(g,d,config,work,inside,expected,o.depth);
        }
#ifdef GRANIT_CLUMP_MEMBER_TEST
        if((o.clump_diagnostics||g.name=="tidal")&&fmm&&g.clumps>0&&work.all_clumps_small&&work.valid&&!periodic_case(g.name)&&!o.project){clump_details=collect_clump_diagnostics(g,d,config,eps,o.depth);clump_details.workload_matches=same_workload(work,clump_details.work);diagnostic_candidate=inside;diagnostic_reference=expected;}
#endif
        if(g.name=="tidal")tidal=tidal_diagnostic(g,d,work,inside,expected);
        d.clear();compute(true);ck(cudaDeviceSynchronize());const auto outside=snapshot(g,d,true);outer=compare(outside,expected,g,d);if(o.worst_body||o.plain_control)worst_outer=worst_force(outside,expected,g,d);equivalence=compare(outside,inside,g,d);dead_max_abs=std::max(dead_max_abs,dead_residual(true));
        // FP64 direct member near sums have a strict roundoff gate. FMM p=4
        // tolerances are fixed before running, raw and projected reported apart.
        double fb=1.e-10,tb=1.e-10;
        if(fmm&&!periodic_case(g.name)&&!direct_fallback_case(g.name)){fb=g.near?1.e-10:1.e-3;tb=g.near?1.e-10:3.e-3;
#ifdef FMM_P2P_LOCAL_FP32
            if(g.near){fb=1.e-6;tb=1.e-6;}
#endif
        }
        pass=dead_max_abs==0&&inner.finite&&outer.finite&&equivalence.finite&&inner.force<=fb&&outer.force<=fb&&inner.torque<=tb&&outer.torque<=tb&&inner.force_max_body<=fb&&outer.force_max_body<=fb&&inner.torque_max_clump<=tb&&outer.torque_max_clump<=tb&&equivalence.force<=equivalence_force_tolerance&&equivalence.torque_abs<=1.e-9;
#ifdef GRANIT_CLUMP_MEMBER_TEST
        if(o.implementation_check&&fmm&&g.near&&g.clumps>0&&work.all_clumps_small&&work.valid&&!periodic_case(g.name)&&!o.project)member_gate=filtered_member_gate(g,d,config,eps);
        if(g.name=="checkpoint"){
            helper_checked=true;
            // Emulate a checkpoint's retained normal forces and reconstructed
            // fast clump wrench. Restore must add gravity exactly once.
            const int n=g.rows.size(),k=g.clumps;
            std::array<std::vector<double>,3> retained;
            for(int axis=0;axis<3;++axis){retained[axis].resize(n);for(int i=0;i<n;++i)retained[axis][i]=(axis+1)*16.+i*.125;}
            retained[0][0]=-0.;
            upload(d.state.particles.force_x,retained[0]);upload(d.state.particles.force_y,retained[1]);upload(d.state.particles.force_z,retained[2]);
            std::array<std::vector<double>,6> fast;
            for(int axis=0;axis<6;++axis){fast[axis].resize(k);for(int c=0;c<k;++c)fast[axis][c]=(axis+1)*3.+c*.25;}
            upload(d.cs.net_force_x,fast[0]);upload(d.cs.net_force_y,fast[1]);upload(d.cs.net_force_z,fast[2]);
            upload(d.cs.net_torque_x,fast[3]);upload(d.cs.net_torque_y,fast[4]);upload(d.cs.net_torque_z,fast[5]);
            AccelerationManager::restore_checkpoint_clump_gravity(d.state,config,0);ck(cudaDeviceSynchronize());
            const auto restored=snapshot(g,d,false);
            const std::array<std::vector<double>,3> after={download(d.state.particles.force_x,n),download(d.state.particles.force_y,n),download(d.state.particles.force_z,n)};
            for(int axis=0;axis<3;++axis)normal_force_unchanged=normal_force_unchanged&&std::memcmp(retained[axis].data(),after[axis].data(),g.normals*sizeof(double))==0;
            for(int c=0;c<k;++c){const int b=g.normals+c;const V ff={fast[0][c],fast[1][c],fast[2][c]},ft={fast[3][c],fast[4][c],fast[5][c]};
                helper_force_error=std::max(helper_force_error,norm(restored.f[b]-ff-inside.f[b]));helper_torque_error=std::max(helper_torque_error,norm(restored.t[b]-ft-inside.t[b]));}
            pass=pass&&normal_force_unchanged&&helper_force_error<=1.e-10L&&helper_torque_error<=1.e-10L;
        }
#endif
    }
    if(reference)numeric_gate_pass=pass;
    if(reference&&!o.baseline&&fmm){
        const bool expect_small_route=g.clumps>0&&g.name!="internal"&&!periodic_case(g.name)&&g.name!="large"&&!direct_fallback_case(g.name);
        structure_checked=expect_small_route||g.name=="fmm"||g.name=="benchmark"||direct_fallback_case(g.name);
        if(expect_small_route)structure_pass=work.all_clumps_small&&work.valid&&work.metadata_read&&work.view_particles==g.live&&work.num_nodes>0;
        if(g.name=="fmm"||g.name=="benchmark")structure_pass=structure_pass&&g.live==o.size&&g.clumps==o.clumps&&work.valid&&work.metadata_read&&(work.tasks[0]>0||work.tasks[1]>0);
        // A fresh stage-1 manager must publish no FMM view when any rigid
        // diameter is outside the certified small-clump range.
        if(direct_fallback_case(g.name))structure_pass=!work.all_clumps_small&&!work.valid&&!work.metadata_read;
        if(g.name=="tidal")structure_pass=structure_pass&&tidal.geometry_gate;
        pass=pass&&structure_pass;
    }
    if(o.implementation_check){
        implementation_pass=pass;
        if(fmm&&(g.name=="fmm"||g.name=="benchmark"))implementation_pass=independent_control.checked&&independent_control.gate_pass&&structure_pass&&inner.finite&&outer.finite&&equivalence.finite&&dead_max_abs==0&&equivalence.force<=equivalence_force_tolerance&&equivalence.torque_abs<=1.e-9;
        if(member_gate.checked)implementation_pass=implementation_pass&&member_gate.gate_pass;
        if(!numeric_gate_pass)std::cerr<<"[Accuracy target FAILED] case="<<g.name<<" method="<<(fmm?"bffmm":"direct")<<"; original gate_pass remains false; implementation_gate_pass="<<(implementation_pass?"true":"false")<<'\n';
    }
    std::vector<double> wall_samples,event_samples;
    if(o.benchmark){
        for(int i=0;i<o.warmup;++i){d.clear();compute(false);}ck(cudaDeviceSynchronize());
        cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
        for(int i=0;i<o.repeats;++i){d.clear();ck(cudaDeviceSynchronize());const auto before=Clock::now();ck(cudaEventRecord(begin));compute(false);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float elapsed;ck(cudaEventElapsedTime(&elapsed,begin,end));event_samples.push_back(elapsed);wall_samples.push_back(ms(before,Clock::now()));}
        ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
    }
    json<<std::setprecision(17)<<"{\"case\":\""<<g.name<<"\",\"method\":\""<<(fmm?"bffmm":"direct")<<"\",\"live_rows\":"<<g.live<<",\"normal_rows\":"<<g.normals<<",\"clumps\":"<<g.clumps<<",\"dead_rows\":"<<g.rows.size()-g.live<<",\"direct_gravitational_constant\":"<<config.acceleration.direct_gravity.gravitational_constant<<",\"fmm_gravitational_constant\":"<<f.gravitational_constant<<",\"oracle_gravitational_constant\":"<<oracle_G<<",\"direct_threshold_leaf\":"<<f.direct_threshold_leaf<<",\"direct_threshold_internal\":"<<f.direct_threshold_internal<<",\"depth\":"<<o.depth<<",\"project_requested\":"<<(o.project?"true":"false")<<",\"baseline_observation\":"<<(o.baseline?"true":"false")<<",\"reference_verified\":"<<(reference?"true":"false")<<",\"gate_pass\":"<<(reference?(pass?"true":"false"):"null")<<",\"checkpoint_helper_checked\":"<<(helper_checked?"true":"false")<<",\"retained_normal_force_bytes_unchanged\":"<<(normal_force_unchanged?"true":"false")<<",\"checkpoint_helper_force_max_abs\":"<<(double)helper_force_error<<",\"checkpoint_helper_torque_max_abs\":"<<(double)helper_torque_error<<",\"dead_output_max_abs\":"<<dead_max_abs<<",\"initialize_host_ms\":"<<init_ms<<",\"first_public_api_host_ms\":"<<first_ms<<",\"inner_outer_force_tolerance\":"<<equivalence_force_tolerance<<",\"inner_outer_force_tolerance_reason\":\""<<equivalence_tolerance_reason<<"\"";
    json<<",\"workload_structure_gate_applied\":"<<(structure_checked?"true":"false")<<",\"workload_structure_gate_pass\":"<<(structure_checked?(structure_pass?"true":"false"):"null")<<",\"workload\":";workload_json(json,work);
    if(reference){json<<",\"inner\":";metric_json(json,inner);json<<",\"outer\":";metric_json(json,outer);json<<",\"inner_outer_difference\":";metric_json(json,equivalence);}
    if(o.worst_body||o.plain_control){json<<",\"worst_inner_force\":";worst_json(json,worst_inner,g);json<<",\"worst_outer_force\":";worst_json(json,worst_outer,g);}
    json<<",\"plain_fmm_control_checked\":"<<(control_checked?"true":"false");
    if(control_checked){json<<",\"plain_fmm_control_accuracy\":";metric_json(json,control_accuracy);json<<",\"plain_fmm_control_minus_candidate\":";metric_json(json,control_difference);json<<",\"plain_fmm_control_max_normal_force_abs_difference\":";number(json,max_normal_control_difference);json<<",\"plain_fmm_control_worst_force\":";worst_json(json,worst_control,g);json<<",\"plain_fmm_control_workload\":";workload_json(json,control_work);}
    if(o.clump_diagnostics||g.name=="tidal"){json<<",\"clump_diagnostics\":";clump_diagnostics_json(json,clump_details,g,d,diagnostic_candidate,diagnostic_reference,tidal);}
    if(tidal.checked){json<<",\"tidal\":";tidal_json(json,tidal);}
    json<<",\"implementation_check\":"<<(o.implementation_check?"true":"false")<<",\"implementation_gate_pass\":"<<(o.implementation_check?(implementation_pass?"true":"false"):"null")<<",\"original_accuracy_gate_pass\":"<<(reference?(numeric_gate_pass?"true":"false"):"null")<<",\"original_accuracy_target_failed\":"<<(reference?(!numeric_gate_pass?"true":"false"):"null");
    if(o.implementation_check){json<<",\"implementation_comparison_tolerance\":"<<(double)implementation_tolerance()<<",\"cancellation_free_control\":{\"checked\":"<<(independent_control.checked?"true":"false");
        if(independent_control.checked){json<<",\"normal_control_api\":\"inner\",\"clump_control_api\":\"outer_original_target_mass\",\"gate_pass\":"<<(independent_control.gate_pass?"true":"false")<<",\"workloads_match\":"<<(independent_control.workloads_match?"true":"false")<<",\"masses_restored\":"<<(independent_control.masses_restored?"true":"false")<<",\"accuracy\":";metric_json(json,independent_control.accuracy);json<<",\"minus_candidate\":";metric_json(json,independent_control.difference);json<<",\"workloads\":[";for(size_t i=0;i<independent_control.runs.size();++i){if(i)json<<',';workload_json(json,independent_control.runs[i]);}json<<']';}json<<'}';
        json<<",\"filtered_member_gate\":{\"checked\":"<<(member_gate.checked?"true":"false");if(member_gate.checked){json<<",\"gate_pass\":"<<(member_gate.gate_pass?"true":"false")<<",\"finite\":"<<(member_gate.finite?"true":"false")<<",\"inner_max_normalized\":";number(json,member_gate.inner_max);json<<",\"outer_max_normalized\":";number(json,member_gate.outer_max);json<<",\"inner_outer_max_normalized\":";number(json,member_gate.equivalence_max);json<<",\"inner_max_abs\":";number(json,member_gate.inner_abs);json<<",\"outer_max_abs\":";number(json,member_gate.outer_abs);json<<",\"worst_inner_particle\":"<<member_gate.worst_inner<<",\"worst_outer_particle\":"<<member_gate.worst_outer;}json<<'}';}
    if(o.benchmark){json<<",\"warmup\":"<<o.warmup<<",\"repeats\":"<<o.repeats<<",\"host_median_ms\":"<<median(wall_samples)<<",\"cuda_event_median_ms\":"<<median(event_samples)<<",\"host_samples_ms\":[";for(size_t i=0;i<wall_samples.size();++i){if(i)json<<',';json<<wall_samples[i];}json<<"],\"cuda_event_samples_ms\":[";for(size_t i=0;i<event_samples.size();++i){if(i)json<<',';json<<event_samples[i];}json<<']';}
    json<<"}\n";json.flush();return o.implementation_check?implementation_pass:(pass||o.baseline||!reference);
}
Options parse(int argc,char** argv){Options o;for(int i=1;i<argc;++i){const std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[i]);};
    if(a=="--case")o.name=value();else if(a=="--method")o.method=value();else if(a=="--size")o.size=std::stoi(value());else if(a=="--clumps")o.clumps=std::stoi(value());else if(a=="--depth")o.depth=std::stoi(value());else if(a=="--direct-threshold-leaf")o.threshold_leaf=std::stoi(value());else if(a=="--direct-threshold-internal")o.threshold_internal=std::stoi(value());else if(a=="--warmup")o.warmup=std::stoi(value());else if(a=="--repeats")o.repeats=std::stoi(value());else if(a=="--json")o.json=value();else if(a=="--benchmark")o.benchmark=true;else if(a=="--baseline")o.baseline=true;else if(a=="--no-verify")o.verify=false;else if(a=="--project")o.project=true;else if(a=="--plain-fmm-control")o.plain_control=true;else if(a=="--worst-body")o.worst_body=true;else if(a=="--implementation-check")o.implementation_check=true;else if(a=="--clump-diagnostics")o.clump_diagnostics=true;else if(a=="--help"){std::cout<<"--case all|small|internal|face|edge|corner|translated|rotated|normals|dead|periodic|fmm|large|benchmark|threshold_below|threshold_above|checkpoint|tidal|differing_g|differing_g_fallback|differing_g_periodic --method direct|bffmm|both --depth 5 --benchmark --size 128 --clumps 2 --warmup 3 --repeats 10 --json results.jsonl --baseline --project --no-verify --plain-fmm-control --worst-body --implementation-check --clump-diagnostics --direct-threshold-leaf 0 --direct-threshold-internal 0\n";std::exit(0);}else throw std::runtime_error("unknown option "+a);}
    if(o.name=="tidal"){if(o.project||!o.verify)throw std::runtime_error("tidal requires verified raw projectionfalse");o.depth=2;}
    if(o.clump_diagnostics&&(!o.verify||o.project||o.method=="direct"))throw std::runtime_error("--clump-diagnostics requires verified raw all-small FMM cases");
#ifndef GRANIT_CLUMP_MEMBER_TEST
    if(o.clump_diagnostics)throw std::runtime_error("--clump-diagnostics requires the active member-test build");
#endif
    if(o.implementation_check&&(!o.verify||o.project))throw std::runtime_error("--implementation-check initially requires verified raw cases");
    if(o.plain_control&&(!o.verify||o.project||o.method=="direct"||(o.name!="all"&&o.name!="fmm"&&o.name!="benchmark")))throw std::runtime_error("--plain-fmm-control requires verified raw FMM all/fmm/benchmark case");
    if(o.threshold_leaf < -1 || o.threshold_internal < -1)throw std::runtime_error("direct thresholds must be nonnegative");
    if(o.depth<2||o.depth>12||o.size<8||o.clumps<1||o.repeats<1||o.warmup<0)throw std::runtime_error("invalid depth, size, or timing count");if(o.method!="both"&&o.method!="direct"&&o.method!="bffmm")throw std::runtime_error("invalid method");return o;
}
}
int main(int argc,char** argv){try{const auto o=parse(argc,argv);int count=0;const auto status=cudaGetDeviceCount(&count);if(status!=cudaSuccess||count==0){std::cerr<<"SKIP: CUDA device unavailable\n";return 77;}
    std::ofstream file;if(!o.json.empty()){file.open(o.json);if(!file)throw std::runtime_error("cannot write "+o.json);}std::ostream& out=o.json.empty()?std::cout:file;
    std::vector<std::string> cases=o.name=="all"?std::vector<std::string>{"small","internal","face","edge","corner","translated","rotated","normals","dead","periodic","threshold_below","threshold_above","fmm"}:std::vector<std::string>{o.name};
#ifdef GRANIT_CLUMP_MEMBER_TEST
    if(o.name=="all")cases.push_back("checkpoint");
#else
    if(o.name=="checkpoint")throw std::runtime_error("checkpoint helper case requires GRANIT_CLUMP_MEMBER_TEST");
#endif
    const std::vector<std::string> known={"small","internal","face","edge","corner","translated","rotated","normals","dead","periodic","fmm","large","benchmark","threshold_below","threshold_above","checkpoint","tidal","differing_g","differing_g_fallback","differing_g_periodic"};bool pass=true;
    for(const auto& name:cases){if(std::find(known.begin(),known.end(),name)==known.end())throw std::runtime_error("unknown case "+name);const auto geometry=make_geometry(name,o.size,o.clumps,o.depth);
        // Zero smoothing length makes normals-only DIRECT exactly Newtonian;
        // registered clumps and BFFMM use the configured Plummer epsilon.
        if(o.method!="bffmm")pass=run(geometry,o,SelfGravityMethod::DIRECT,out)&&pass;
        if(o.method!="direct")pass=run(geometry,o,SelfGravityMethod::BFFMM,out)&&pass;}
    return pass?0:1;
}catch(const std::exception& e){std::cerr<<"clump_gravity: "<<e.what()<<'\n';return 2;}}

#include "external_bodies.h"

#include "acceleration.h"
#include "cuda_utils.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace cudadem {
namespace {
__constant__ ExternalBodyState dev_external_body_state;

// The sidecar is deliberately self-contained: it binds the physics inputs
// and preserves every immutable loss event, so resume neither loses nor
// duplicates records.
struct ExternalBodyCheckpointHeader {
    char magic[8]; uint32_t version; int64_t step; double time; uint64_t config_hash;
    int64_t initial_live_particles; double initial_live_mass;
    int64_t absorbed_particles; double absorbed_mass;
    double px,py,pz,lx,ly,lz,energy; uint64_t event_count;
};
ExternalBodyLedger host_ledger{};
std::vector<ExternalBodyEvent> host_events;
bool initialized=false; int32_t* d_absorbed_count=nullptr;
ExternalBodyEvent* d_absorbed_events=nullptr; int32_t event_capacity=0;
size_t published_event_count=0; uint64_t host_gravity_evaluations=0;

__global__ void add_external_body_forces_kernel(const double* x,const double* y,const double* z,
 const float* mass,double* fx,double* fy,double* fz,int32_t n) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; const auto b=dev_external_body_state; if(!b.enabled)return;
 const double px=x[i],py=y[i],pz=z[i]; double ax=0,ay=0,az=0;
 if(b.saturn_enabled){const double dx=b.saturn_x-px,dy=b.saturn_y-py,dz=b.saturn_z-pz,r2=dx*dx+dy*dy+dz*dz;if(r2>0){const double s=GRAVITATIONAL_CONSTANT*b.saturn_mass/(r2*sqrt(r2));ax+=s*dx;ay+=s*dy;az+=s*dz;}}
 if(b.satellite_enabled){const double dx=b.satellite_x-px,dy=b.satellite_y-py,dz=b.satellite_z-pz,r2=dx*dx+dy*dy+dz*dz;if(r2>0){const double s=GRAVITATIONAL_CONSTANT*b.satellite_mass/(r2*sqrt(r2));ax+=s*dx;ay+=s*dy;az+=s*dz;}}
 const double m=mass[i]; atomicAdd(&fx[i],m*ax);atomicAdd(&fy[i],m*ay);atomicAdd(&fz[i],m*az);
}

__global__ void mark_saturn_absorption_kernel(const double* x,const double* y,const double* z,
 const double* vx,const double* vy,const double* vz,const float* mass,const int32_t* gid,
 uint8_t* lifecycle,int32_t n,int64_t step,double t0,double dt,int32_t* count,ExternalBodyEvent* events) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; const auto b=dev_external_body_state;
 if(!b.enabled||!b.saturn_enabled||!(b.saturn_absorb_radius>0)||lifecycle[i]!=0)return;
 const double x1=x[i],y1=y[i],z1=z[i], ux=vx[i],uy=vy[i],uz=vz[i];
 const double x0=x1-ux*dt,y0=y1-uy*dt,z0=z1-uz*dt;
 const double rx=x0-b.saturn_x,ry=y0-b.saturn_y,rz=z0-b.saturn_z,qx=x1-x0,qy=y1-y0,qz=z1-z0;
 const double a=qx*qx+qy*qy+qz*qz, bb=2*(rx*qx+ry*qy+rz*qz), c=rx*rx+ry*ry+rz*rz-b.saturn_absorb_radius*b.saturn_absorb_radius;
 double lambda=0; if(c>0){if(!(a>0))return;const double disc=bb*bb-4*a*c;if(disc<0)return;lambda=(-bb-sqrt(disc))/(2*a);if(lambda<0||lambda>1)return;} // first entry root; catches enter-and-exit drifts
 const double cx=x0+lambda*qx,cy=y0+lambda*qy,cz=z0+lambda*qz, crx=cx-b.saturn_x,cry=cy-b.saturn_y,crz=cz-b.saturn_z,m=mass[i];
 lifecycle[i]=1; const int slot=atomicAdd(count,1); const double px=m*ux,py=m*uy,pz=m*uz, rr=sqrt(crx*crx+cry*cry+crz*crz);
 events[slot]={gid[i],1,step,t0+lambda*dt,cx,cy,cz,ux,uy,uz,m,px,py,pz,cry*pz-crz*py,crz*px-crx*pz,crx*py-cry*px,m*(0.5*(ux*ux+uy*uy+uz*uz)-GRAVITATIONAL_CONSTANT*b.saturn_mass/rr)};
}

std::string sidecar_path(const std::string& p){return p+".external_bodies";}
uint64_t external_body_config_hash(const SimConfig& config) {
 const auto& c=config.acceleration.external_bodies; uint64_t h=UINT64_C(1469598103934665603); auto mix=[&](const void*d,size_t n){auto p=static_cast<const unsigned char*>(d);for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}}; auto md=[&](double v){if(v==0)v=0;mix(&v,sizeof v);};
 mix(&c.enabled,sizeof c.enabled);mix(&c.saturn_enabled,sizeof c.saturn_enabled);md(c.saturn_position.x);md(c.saturn_position.y);md(c.saturn_position.z);md(c.saturn_mass);md(c.saturn_absorb_radius);mix(&c.satellite_enabled,sizeof c.satellite_enabled);md(c.satellite_mass);md(c.satellite_orbit_radius);md(c.satellite_phase);mix(&c.satellite_prograde,sizeof c.satellite_prograde);md(config.time.start_time);return h;
}
}

ExternalBodyState evaluate_external_body_state(const SimConfig& config,double time){const auto&c=config.acceleration.external_bodies;ExternalBodyState s{};s.enabled=c.enabled;s.saturn_enabled=c.enabled&&c.saturn_enabled;s.saturn_x=c.saturn_position.x;s.saturn_y=c.saturn_position.y;s.saturn_z=c.saturn_position.z;s.saturn_mass=c.saturn_mass;s.saturn_absorb_radius=c.saturn_absorb_radius;s.satellite_enabled=c.enabled&&c.satellite_enabled;s.satellite_mass=c.satellite_mass;if(s.satellite_enabled){const double w=sqrt(GRAVITATIONAL_CONSTANT*c.saturn_mass/(c.satellite_orbit_radius*c.satellite_orbit_radius*c.satellite_orbit_radius));const double phase=c.satellite_phase+(c.satellite_prograde?1.:-1.)*w*(time-config.time.start_time);s.satellite_x=s.saturn_x+c.satellite_orbit_radius*cos(phase);s.satellite_y=s.saturn_y+c.satellite_orbit_radius*sin(phase);s.satellite_z=s.saturn_z;}return s;}

void ExternalBodyManager::initialize(const SimConfig& c,int32_t n,double m){shutdown();if(!c.acceleration.external_bodies.enabled)return;if(n<0||!(m>=0)||!std::isfinite(m))throw std::runtime_error("invalid initial external-body ledger");host_ledger={n,m,0,0,0,0,0,0,0,0,0};event_capacity=n;CUDA_CHECK(cudaMalloc(&d_absorbed_count,sizeof(int32_t)));if(n)CUDA_CHECK(cudaMalloc(&d_absorbed_events,static_cast<size_t>(n)*sizeof(ExternalBodyEvent)));initialized=true;const auto s=evaluate_external_body_state(c,c.time.start_time);CUDA_CHECK(cudaMemcpyToSymbol(dev_external_body_state,&s,sizeof s));}
void ExternalBodyManager::shutdown(){if(d_absorbed_count)cudaFree(d_absorbed_count);if(d_absorbed_events)cudaFree(d_absorbed_events);d_absorbed_count=nullptr;d_absorbed_events=nullptr;event_capacity=0;host_ledger={};host_events.clear();published_event_count=0;host_gravity_evaluations=0;initialized=false;}
void ExternalBodyManager::add_forces(SimulationState& st,const SimConfig& c,double time,int32_t n,cudaStream_t stream){if(!initialized||n<=0)return;const auto b=evaluate_external_body_state(c,time);CUDA_CHECK(cudaMemcpyToSymbolAsync(dev_external_body_state,&b,sizeof b,0,cudaMemcpyHostToDevice,stream));const int block=c.gpu.block_size,grid=(n+block-1)/block;LAUNCH_KERNEL(add_external_body_forces_kernel,grid,block,0,stream,st.particles.pos_x,st.particles.pos_y,st.particles.pos_z,st.particles.mass,st.particles.force_x,st.particles.force_y,st.particles.force_z,n);++host_gravity_evaluations;}
ExternalBodyAbsorption ExternalBodyManager::mark_absorbed(SimulationState& st,const SimConfig& c,int32_t n,int64_t step,double t0,double dt,cudaStream_t stream){if(!initialized||n<=0||!(c.acceleration.external_bodies.saturn_absorb_radius>0))return {0,0};CUDA_CHECK(cudaMemsetAsync(d_absorbed_count,0,sizeof(int32_t),stream));const int block=c.gpu.block_size,grid=(n+block-1)/block;LAUNCH_KERNEL(mark_saturn_absorption_kernel,grid,block,0,stream,st.particles.pos_x,st.particles.pos_y,st.particles.pos_z,st.particles.vel_x,st.particles.vel_y,st.particles.vel_z,st.particles.mass,st.particles.global_id,st.particles.lifecycle_state,n,step,t0,dt,d_absorbed_count,d_absorbed_events);ExternalBodyAbsorption delta{};CUDA_CHECK(cudaMemcpyAsync(&delta.count,d_absorbed_count,sizeof delta.count,cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));if(delta.count<0||delta.count>event_capacity)throw std::runtime_error("invalid Saturn absorption event count");std::vector<ExternalBodyEvent> recent(delta.count);if(delta.count)CUDA_CHECK(cudaMemcpy(recent.data(),d_absorbed_events,recent.size()*sizeof(ExternalBodyEvent),cudaMemcpyDeviceToHost));for(const auto&e:recent){if(!(e.mass>0)||!std::isfinite(e.orbital_energy))throw std::runtime_error("invalid Saturn absorption event");delta.mass+=e.mass;host_ledger.absorbed_particles++;host_ledger.absorbed_mass+=e.mass;host_ledger.absorbed_momentum_x+=e.momentum_x;host_ledger.absorbed_momentum_y+=e.momentum_y;host_ledger.absorbed_momentum_z+=e.momentum_z;host_ledger.absorbed_angular_momentum_x+=e.angular_momentum_x;host_ledger.absorbed_angular_momentum_y+=e.angular_momentum_y;host_ledger.absorbed_angular_momentum_z+=e.angular_momentum_z;host_ledger.absorbed_orbital_energy+=e.orbital_energy;host_events.push_back(e);}return delta;}
void ExternalBodyManager::write_checkpoint_sidecar(const std::string&p,const SimulationState&st,const SimConfig&c){if(!initialized)return;ExternalBodyCheckpointHeader h{{'E','X','T','B','O','D','Y','\0'},2,st.current_step,st.current_time,external_body_config_hash(c),host_ledger.initial_live_particles,host_ledger.initial_live_mass,host_ledger.absorbed_particles,host_ledger.absorbed_mass,host_ledger.absorbed_momentum_x,host_ledger.absorbed_momentum_y,host_ledger.absorbed_momentum_z,host_ledger.absorbed_angular_momentum_x,host_ledger.absorbed_angular_momentum_y,host_ledger.absorbed_angular_momentum_z,host_ledger.absorbed_orbital_energy,host_events.size()};const std::string final=sidecar_path(p),tmp=final+".tmp";std::ofstream f(tmp,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write external-bodies checkpoint sidecar");f.write(reinterpret_cast<const char*>(&h),sizeof h);if(!host_events.empty())f.write(reinterpret_cast<const char*>(host_events.data()),host_events.size()*sizeof(ExternalBodyEvent));f.close();if(!f)throw std::runtime_error("failed writing external-bodies checkpoint sidecar");if(std::rename(tmp.c_str(),final.c_str())!=0)throw std::runtime_error("cannot atomically publish external-bodies checkpoint sidecar");}
void ExternalBodyManager::restore_checkpoint_sidecar(const std::string&p,const SimulationState&st,const SimConfig&c){if(!initialized)return;ExternalBodyCheckpointHeader h{};std::ifstream f(sidecar_path(p),std::ios::binary);const char magic[8]={'E','X','T','B','O','D','Y','\0'};if(!f.read(reinterpret_cast<char*>(&h),sizeof h)||memcmp(h.magic,magic,sizeof magic)||h.version!=2||h.step!=st.current_step||h.time!=st.current_time||h.config_hash!=external_body_config_hash(c)||h.initial_live_particles<0||h.absorbed_particles<0||h.absorbed_particles>h.initial_live_particles||h.event_count!=static_cast<uint64_t>(h.absorbed_particles)||!std::isfinite(h.initial_live_mass)||!std::isfinite(h.absorbed_mass)||h.initial_live_mass<0||h.absorbed_mass<0||h.absorbed_mass>h.initial_live_mass)throw std::runtime_error("external-bodies resume requires matching event-ledger sidecar");host_events.resize(h.event_count);if(h.event_count&&!f.read(reinterpret_cast<char*>(host_events.data()),host_events.size()*sizeof(ExternalBodyEvent)))throw std::runtime_error("external-bodies checkpoint ledger is truncated");host_ledger={h.initial_live_particles,h.initial_live_mass,h.absorbed_particles,h.absorbed_mass,h.px,h.py,h.pz,h.lx,h.ly,h.lz,h.energy};published_event_count=0;}
bool ExternalBodyManager::checkpoint_sidecar_matches(const std::string&p,int64_t step,double time,const SimConfig&c){ExternalBodyCheckpointHeader h{};std::ifstream f(sidecar_path(p),std::ios::binary|std::ios::ate);const char magic[8]={'E','X','T','B','O','D','Y','\0'};if(!f)return false;const std::streamoff bytes=f.tellg();if(bytes<static_cast<std::streamoff>(sizeof h))return false;f.seekg(0);if(!f.read(reinterpret_cast<char*>(&h),sizeof h))return false;const uint64_t max_events=(std::numeric_limits<uint64_t>::max()-sizeof h)/sizeof(ExternalBodyEvent);if(h.event_count>max_events||bytes!=static_cast<std::streamoff>(sizeof h+h.event_count*sizeof(ExternalBodyEvent)))return false;const bool finite=std::isfinite(h.initial_live_mass)&&std::isfinite(h.absorbed_mass)&&std::isfinite(h.px)&&std::isfinite(h.py)&&std::isfinite(h.pz)&&std::isfinite(h.lx)&&std::isfinite(h.ly)&&std::isfinite(h.lz)&&std::isfinite(h.energy);return !memcmp(h.magic,magic,sizeof magic)&&h.version==2&&h.step==step&&h.time==time&&h.config_hash==external_body_config_hash(c)&&h.event_count==static_cast<uint64_t>(h.absorbed_particles)&&h.initial_live_particles>=0&&h.absorbed_particles>=0&&h.initial_live_particles>=h.absorbed_particles&&h.initial_live_mass>=0&&h.absorbed_mass>=0&&h.absorbed_mass<=h.initial_live_mass&&finite;}
bool ExternalBodyManager::is_enabled(){return initialized;} ExternalBodyLedger ExternalBodyManager::ledger(){return host_ledger;} const std::vector<ExternalBodyEvent>& ExternalBodyManager::events(){return host_events;} uint64_t ExternalBodyManager::gravity_evaluations(){return host_gravity_evaluations;}
void ExternalBodyManager::write_event_receipts(const std::string& dir){if(!initialized||dir.empty()||published_event_count==host_events.size())return;std::ofstream f(dir+"/external_body_events.jsonl",std::ios::app);if(!f)throw std::runtime_error("cannot append external-body event ledger");for(;published_event_count<host_events.size();++published_event_count){const auto&e=host_events[published_event_count];f<<std::setprecision(17)<<"{\"schema_version\":\"granit-external-event-v1\",\"particle_global_id\":"<<e.particle_global_id<<",\"body_id\":1,\"reason\":\"saturn_center_crossing\",\"step\":"<<e.step<<",\"time_seconds\":"<<e.time<<",\"position_m\":["<<e.position_x<<","<<e.position_y<<","<<e.position_z<<"],\"velocity_m_s\":["<<e.velocity_x<<","<<e.velocity_y<<","<<e.velocity_z<<"],\"mass_kg\":"<<e.mass<<",\"momentum_kg_m_s\":["<<e.momentum_x<<","<<e.momentum_y<<","<<e.momentum_z<<"],\"angular_momentum_kg_m2_s\":["<<e.angular_momentum_x<<","<<e.angular_momentum_y<<","<<e.angular_momentum_z<<"],\"orbital_energy_j\":"<<e.orbital_energy<<",\"energy_convention\":\"m*(0.5*|v_drift_half|^2-G*M_saturn/|r_cross|)\"}\n";}if(!f)throw std::runtime_error("failed writing external-body event ledger");}
} // namespace cudadem

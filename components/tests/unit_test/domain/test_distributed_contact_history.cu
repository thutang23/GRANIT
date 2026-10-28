// Tests the same CUDA lifecycle and force kernels included by main_multigpu.cu.
#include "src/domain/distributed_contact_history.cuh"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace cudadem;
namespace {
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void sync(){CUDA_CHECK(cudaDeviceSynchronize());}
template<class T> T* alloc(int n){T* p;CUDA_CHECK(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;++i)p[i]=T{};return p;}
constexpr int N=40,S=16;
struct Fixture {
 SimulationState s; ContactHistoryWorkspace workspace; ContactPair* pairs=alloc<ContactPair>(N); MaterialRecord* material=alloc<MaterialRecord>(1); DemInteractionProperties* interaction=alloc<DemInteractionProperties>(1);
 std::vector<void*> allocations;
 template<class T> void field(T*& p){p=alloc<T>(N);allocations.push_back(p);}
 Fixture(){auto& p=s.particles;p.num_particles=N;p.num_local=N;p.capacity=N;
 #define F(x) field(p.x)
 F(pos_x);F(pos_y);F(pos_z);F(vel_x);F(vel_y);F(vel_z);F(omega_x);F(omega_y);F(omega_z);F(radius);F(mass);F(moment_of_inertia);F(material_id);F(global_id);F(force_x);F(force_y);F(force_z);F(torque_x);F(torque_y);F(torque_z);F(contact_history_offset);F(num_contacts);
 #undef F
 s.contact_history.data=alloc<ContactHistory>(N*S);s.contact_history.max_contacts_per_particle=S;s.contact_history.capacity=N*S;
 for(int i=0;i<N;++i){p.global_id[i]=i;p.contact_history_offset[i]=i*S;p.radius[i]=1;p.mass[i]=1;p.moment_of_inertia[i]=.4;p.pos_x[i]=1.9;}
 p.pos_x[0]=0;p.vel_y[0]=.01;p.omega_x[0]=.02;p.omega_z[0]=.03;
 material[0].has_dem=1;material[0].dem={1000,1000000,.25};interaction[0].restitution=.8;interaction[0].friction_static=.4;interaction[0].friction_rolling=.03;interaction[0].friction_twisting=.02;interaction[0].shape_parameter=1;interaction[0].cohesion_cutoff_factor=1;
 }
 ~Fixture(){sync();for(auto p:allocations)cudaFree(p);cudaFree(pairs);cudaFree(material);cudaFree(interaction);cudaFree(s.contact_history.data);}
 ContactHistoryOverflow error(){ContactHistoryOverflow e;CUDA_CHECK(cudaMemcpy(&e,workspace.overflow,sizeof(e),cudaMemcpyDeviceToHost));return e;}
 bool shear=false;
 void prepare(int n){prepare_distributed_contact_history(pairs,n,s,N,workspace,0,shear,2,2,.15,true);sync();}
 template<bool flip> void force(int n,float dt){auto& p=s.particles;compute_contact_forces_multigpu_kernel<flip><<<1,64>>>(pairs,n,p.pos_x,p.pos_y,p.pos_z,p.vel_x,p.vel_y,p.vel_z,p.omega_x,p.omega_y,p.omega_z,p.radius,p.mass,p.moment_of_inertia,p.material_id,material,interaction,1,p.global_id,p.force_x,p.force_y,p.force_z,p.torque_x,p.torque_y,p.torque_z,s.contact_history.data,p.contact_history_offset,p.num_contacts,S,workspace.lookup_counts,workspace.overflow,false,dt,N,shear,2,2,.15,0,true);sync();}
 void step(int n,float dt=1e-4){auto& p=s.particles;for(int i=0;i<N;++i)p.force_x[i]=p.force_y[i]=p.force_z[i]=p.torque_x[i]=p.torque_y[i]=p.torque_z[i]=0;prepare(n);force<false>(n,dt);force<true>(n,dt);}
 ContactHistory& row(int i,int k=0){return s.contact_history.data[i*S+k];}
};
std::array<float,9> springs(const ContactHistory& h){return {h.delta_t_x,h.delta_t_y,h.delta_t_z,h.delta_r_x,h.delta_r_y,h.delta_r_z,h.delta_tw_x,h.delta_tw_y,h.delta_tw_z};}
void pair_checks(Fixture& f,int j){auto a=springs(f.row(0)),b=springs(f.row(j));for(int k=0;k<9;++k)require(a[k]==-b[k],"reciprocal nine-component history mismatch");auto& p=f.s.particles;
 double residual=0;std::array<double,3> torque{};for(int k=0;k<N;++k){residual+=p.force_y[k];torque[0]+=p.torque_x[k]+p.pos_y[k]*p.force_z[k]-p.pos_z[k]*p.force_y[k];torque[1]+=p.torque_y[k]+p.pos_z[k]*p.force_x[k]-p.pos_x[k]*p.force_z[k];torque[2]+=p.torque_z[k]+p.pos_x[k]*p.force_y[k]-p.pos_y[k]*p.force_x[k];}
 require(std::abs(residual)<1e-12,"internal pair force imbalance");require(std::sqrt(torque[0]*torque[0]+torque[1]*torque[1]+torque[2]*torque[2])<2e-7,"orbital plus spin angular momentum derivative imbalance");require(p.force_x[0]==-p.force_x[j]&&p.force_z[0]==-p.force_z[j],"force components imbalance");
}
void births(){Fixture f;for(int encounter=0;encounter<64;++encounter){int j=1+encounter%32;f.pairs[0]=ContactPair(0,j);f.step(1);require(f.s.particles.num_contacts[0]==1,"retired rows not reclaimed");require(f.row(0).partner_global_id==j,"new partner missing");pair_checks(f,j);f.step(1);pair_checks(f,j);f.step(0);require(f.s.particles.num_contacts[0]==0,"death not reclaimed");}require(!f.error().flag,"cumulative births falsely overflowed");}
void seeded(){Fixture f;auto& p=f.s.particles;p.num_contacts[0]=S;for(int k=0;k<S;++k){f.row(0,k).partner_global_id=k+1;f.row(0,k).is_active=false;}
 f.pairs[0]=ContactPair(0,20);f.step(1);require(p.num_contacts[0]==1,"16 inactive checkpoint rows block new birth");pair_checks(f,20);
 // Mixed row keeps all nine spring values byte-for-byte through preparation.
 p.num_contacts[0]=S;for(int k=1;k<S;++k){f.row(0,k)=ContactHistory{};f.row(0,k).partner_global_id=k;}
 auto retained=f.row(0);f.row(0)=ContactHistory{};f.row(0,7)=retained;auto expected=springs(retained);f.prepare(1);require(p.num_contacts[0]==1&&springs(f.row(0))==expected,"compaction changed retained spring");
 // All sixteen formerly active contacts die at this exact endpoint.
 for(int k=0;k<S;++k){f.row(0,k)=ContactHistory{};f.row(0,k).is_active=true;f.row(0,k).partner_global_id=k+1;}p.num_contacts[0]=S;f.pairs[0]=ContactPair(0,30);f.step(1);require(p.num_contacts[0]==1&&f.row(0).partner_global_id==30,"same-endpoint deaths fail to free capacity");require(!f.error().flag,"same-endpoint deaths falsely overflow");pair_checks(f,30);
}
void geometry(){
 for(int mode=0;mode<5;++mode){Fixture f;auto& p=f.s.particles;f.pairs[0]=ContactPair(0,1);
 if(mode<3){float x=mode==0?2.f:(mode==1?std::nextafter(2.f,0.f):std::nextafter(2.f,3.f));p.pos_x[1]=x;}
 else{f.shear=true;p.radius[0]=p.radius[1]=.1f;p.pos_x[0]=-.95;p.pos_x[1]=.95;p.pos_y[1]=-.15;f.pairs[0]=ContactPair(0,1,-1,0,0);if(mode==4){p.global_id[0]=10;p.global_id[1]=2;}}
 const bool contact=mode==1||mode>=3;f.row(0).is_active=true;f.row(0).partner_global_id=p.global_id[1];p.num_contacts[0]=1;f.prepare(1);require(p.num_contacts[0]==int(contact),"FP32/shear preparation disagrees with expected overlap");f.force<false>(1,1e-4);f.force<true>(1,1e-4);require(p.num_contacts[0]==int(contact)&&p.num_contacts[1]==int(contact),"FP32/shear preparation and production force disagree");}
}
void overflow(){Fixture f;for(int k=0;k<17;++k)f.pairs[k]=ContactPair(0,k+1);f.step(17);require(f.error().flag,"17 simultaneous contacts silently accepted");require(f.s.particles.num_contacts[0]<=S,"overflow count exceeds row allocation");}
}
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices){std::cout<<"SKIP no CUDA device\n";return 77;}try{births();seeded();geometry();overflow();std::cout<<"PASS distributed contact history: 64 births, repeated partners, exact nine springs, conservation, inactive/mixed/same-endpoint rows, overflow\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

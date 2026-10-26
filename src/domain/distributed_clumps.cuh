#pragma once

#include "domain_manager.h"
#include "peer_transfer.h"
#include "clump_shear_math.cuh"
#include "../core/wall/wall.h"
#include "../core/neighbor/neighbor_types.h"
#include "../core/particle_model/clump.h"
#include "../core/particle_model/clump_rigid_math.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace cudadem {

// Authoritative rows exist only on a body's fixed owner. A participant holds
// just kinematics and a partial wrench for bodies with locally owned members.
struct DistributedClumpPose {
    double com[3], vel[3], omega[3], quat[4];
    int64_t winding[3];
};
struct DistributedClumpWrench { double f[3], t[3]; };
struct DistributedClumpBody {
    DistributedClumpPose pose;
    DistributedClumpWrench wrench;
    double mass, inertia[6], p3[19]; // xx,xy,xz,yy,yz,zz
    int32_t cid, owner;
};

__host__ __device__ inline ClumpRigidVec3 distributed_clump_rotate(
    const double* q, ClumpRigidVec3 v, bool inverse=false) {
    const double sign=inverse?-1.0:1.0;
    const double x=sign*q[1],y=sign*q[2],z=sign*q[3];
    const double tx=2*(y*v.z-z*v.y),ty=2*(z*v.x-x*v.z),tz=2*(x*v.y-y*v.x);
    return {v.x+q[0]*tx+y*tz-z*ty,v.y+q[0]*ty+z*tx-x*tz,v.z+q[0]*tz+x*ty-y*tx};
}
static __global__ void distributed_clump_integrate(
    DistributedClumpBody* bodies,int n,double dt,bool drift,
    ShearBoxConfig shear,PeriodicBoundaryConfig pbc,ShearPhase phase,int* error) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& b=bodies[i];auto& p=b.pose;
    for(int a=0;a<3;++a)p.vel[a]+=0.5*dt*b.wrench.f[a]/b.mass;
    ClumpRigidVec3 effective{b.wrench.t[0],b.wrench.t[1],b.wrench.t[2]};
    if(shear.enabled){const auto omega=distributed_clump_rotate(p.quat,{p.omega[0],p.omega[1],p.omega[2]});
        const auto hill=clump_hill_torque(b.inertia,p.quat,omega,shear);
        effective.x+=hill.x;effective.y+=hill.y;effective.z+=hill.z;}
    const auto torque=distributed_clump_rotate(p.quat,effective,true);
    ClumpRigidVec3 alpha{};
    if(clump_full_tensor_alpha(b.inertia[0],b.inertia[1],b.inertia[2],b.inertia[3],b.inertia[4],b.inertia[5],
       {p.omega[0],p.omega[1],p.omega[2]},torque,alpha)) {
        p.omega[0]+=0.5*dt*alpha.x;p.omega[1]+=0.5*dt*alpha.y;p.omega[2]+=0.5*dt*alpha.z;
    }
    if(drift){
        if(shear.enabled){auto state=exact_hill_drift({p.com[0],p.com[1],p.com[2],p.vel[0],p.vel[1],p.vel[2]},
                shear.omega,shear.shear_parameter,shear.vertical_frequency,dt);
            p.com[0]=state.x;p.com[1]=state.y;p.com[2]=state.z;
            p.vel[0]=state.vx;p.vel[1]=state.vy;p.vel[2]=state.vz;
            if(!clump_shear_wrap(p.com[0],p.com[1],p.vel[1],p.winding[0],p.winding[1],
                pbc.box_min_x,pbc.box_min_y,pbc.box_length_x,pbc.box_length_y,phase))atomicExch(error,1);
        }else for(int a=0;a<3;++a)p.com[a]+=dt*p.vel[a];
        clump_exp_quaternion(p.quat[0],p.quat[1],p.quat[2],p.quat[3],{p.omega[0],p.omega[1],p.omega[2]},dt);
    }
}
static __global__ void distributed_clump_pack_pose(
    const DistributedClumpBody* bodies,const int* index,DistributedClumpPose* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=bodies[index[i]].pose;
}
static __global__ void distributed_clump_scatter_pose(
    const DistributedClumpPose* in,const int* index,DistributedClumpPose* replicas,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)replicas[index[i]]=in[i];
}
static __global__ void distributed_clump_copy_local_pose(
    const DistributedClumpBody* bodies,const int* own,const int* part,DistributedClumpPose* replicas,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)replicas[part[i]]=bodies[own[i]].pose;
}
static __global__ void distributed_clump_sync_members(
    ParticleSoA p,int n,const int* lookup,const DistributedClumpPose* rows,
    ShearBoxConfig shear,PeriodicBoundaryConfig pbc,ShearPhase phase,int* error) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||p.clump_id[i]<=0)return;
    const auto& b=rows[lookup[p.clump_id[i]]];
    const auto r=distributed_clump_rotate(b.quat,{p.clump_off_x[i],p.clump_off_y[i],p.clump_off_z[i]});
    const auto w=distributed_clump_rotate(b.quat,{b.omega[0],b.omega[1],b.omega[2]});
    p.pos_x[i]=b.com[0]+r.x;p.pos_y[i]=b.com[1]+r.y;p.pos_z[i]=b.com[2]+r.z;
    p.vel_x[i]=b.vel[0]+w.y*r.z-w.z*r.y;
    p.vel_y[i]=b.vel[1]+w.z*r.x-w.x*r.z;
    p.vel_z[i]=b.vel[2]+w.x*r.y-w.y*r.x;
    if(shear.enabled){p.winding_x[i]=b.winding[0];p.winding_y[i]=b.winding[1];p.winding_z[i]=b.winding[2];
        if(!clump_shear_wrap(p.pos_x[i],p.pos_y[i],p.vel_y[i],p.winding_x[i],p.winding_y[i],
            pbc.box_min_x,pbc.box_min_y,pbc.box_length_x,pbc.box_length_y,phase))atomicExch(error,1);}
    p.omega_x[i]=w.x;p.omega_y[i]=w.y;p.omega_z[i]=w.z;
    p.quat_w[i]=b.quat[0];p.quat_x[i]=b.quat[1];p.quat_y[i]=b.quat[2];p.quat_z[i]=b.quat[3];
}
static __global__ void distributed_clump_reduce_members(
    ParticleSoA p,int n,const int* lookup,const DistributedClumpPose* rows,DistributedClumpWrench* out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||p.clump_id[i]<=0)return;
    const int slot=lookup[p.clump_id[i]];const auto& b=rows[slot];auto& w=out[slot];
    const auto arm=distributed_clump_rotate(b.quat,{p.clump_off_x[i],p.clump_off_y[i],p.clump_off_z[i]});
    const double rx=arm.x,ry=arm.y,rz=arm.z;
    const double fx=p.force_x[i],fy=p.force_y[i],fz=p.force_z[i];
    atomicAdd(&w.f[0],fx);atomicAdd(&w.f[1],fy);atomicAdd(&w.f[2],fz);
    atomicAdd(&w.t[0],ry*fz-rz*fy+p.torque_x[i]);
    atomicAdd(&w.t[1],rz*fx-rx*fz+p.torque_y[i]);
    atomicAdd(&w.t[2],rx*fy-ry*fx+p.torque_z[i]);
}
static __global__ void distributed_clump_zero_owned(DistributedClumpBody* b,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i].wrench={};
}
static __global__ void distributed_clump_pack_wrench(
    const DistributedClumpWrench* in,const int* index,DistributedClumpWrench* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=in[index[i]];
}
static __global__ void distributed_clump_add_wrench(
    const DistributedClumpWrench* in,const int* own,DistributedClumpBody* bodies,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& w=bodies[own[i]].wrench;
    for(int a=0;a<3;++a){w.f[a]+=in[i].f[a];w.t[a]+=in[i].t[a];}
}
static __global__ void distributed_clump_add_local_wrench(
    const DistributedClumpWrench* in,const int* part,const int* own,DistributedClumpBody* bodies,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& v=in[part[i]];auto& w=bodies[own[i]].wrench;
    for(int a=0;a<3;++a){w.f[a]+=v.f[a];w.t[a]+=v.t[a];}
}

// Opt-in regression evidence counts physical inter-body contacts that use
// halo members, independently of whether the owning body spans ranks.
static __global__ void distributed_clump_count_remote_contacts(
    ParticleSoA p,const ContactPair* pairs,int n,int local,unsigned long long* out,
    ShearBoxConfig shear,PeriodicBoundaryConfig pbc,ShearPhase phase) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int a=pairs[i].i,b=pairs[i].j;
    if(a<0||b<0||((a<local)==(b<local)))return;
    if(p.clump_id[a]<=0||p.clump_id[b]<=0||p.clump_id[a]==p.clump_id[b])return;
    const auto image=shear_partner_image(pairs[i].image_x,pairs[i].image_y,pairs[i].image_z,
        pbc.box_length_x,pbc.box_length_y,pbc.box_length_z,phase);
    const double x=p.pos_x[a]-p.pos_x[b]-(shear.enabled?image.x:0),
        y=p.pos_y[a]-p.pos_y[b]-(shear.enabled?image.y:0),z=p.pos_z[a]-p.pos_z[b];
    const double radius=double(p.radius[a])+p.radius[b];
    if(x*x+y*y+z*z<radius*radius)atomicAdd(out,1ULL);
}

class DistributedClumps {
    struct Rank {
        DistributedClumpBody* owned=nullptr;int n_owned=0;
        DistributedClumpPose* replica=nullptr;DistributedClumpWrench* partial=nullptr;
        int* lookup=nullptr;std::vector<int> cids;
        cudaEvent_t ready=nullptr;cudaStream_t stream=nullptr;
        unsigned long long* contact_counter=nullptr;int* shear_error=nullptr;int member_count=0;
    };
    // One sparse relation per owner -> participant. Indices are allocated on
    // the GPU that consumes them; the two message directions carry no CID.
    struct Edge {
        int n=0;int* own=nullptr;int* part=nullptr;
        DistributedClumpPose *pose_send=nullptr,*pose_recv=nullptr;
        DistributedClumpWrench *wrench_send=nullptr,*wrench_recv=nullptr;
    };
    std::vector<Rank> ranks_;std::vector<Edge> edges_;
    std::vector<int> owner_,owner_slot_,changed_;
    std::vector<std::string> rank_errors_;
    std::string collective_error_;
    bool rebuilt_=false,diagnostics_=false;uint64_t route_generation_=0;
    int k_=0;double max_diameter_=0;
    ShearBoxConfig shear_;
    PeriodicBoundaryConfig pbc_;
    ShearPhase phase_at(double time) const {
        return shear_.enabled?compute_shear_phase_checked(shear_,pbc_.box_length_x,pbc_.box_length_y,time):ShearPhase{};
    }
    static int blocks(int n){return (n+255)/256;}
    Edge& edge(int o,int p){return edges_[o*ranks_.size()+p];}
    void agree_errors() {
        #pragma omp barrier
        #pragma omp single
        {
            collective_error_.clear();
            for (size_t rank=0; rank<rank_errors_.size(); ++rank) {
                if (!rank_errors_[rank].empty()) {
                    collective_error_="distributed clumps rank "+std::to_string(rank)+": "+rank_errors_[rank];
                    break;
                }
            }
        }
        #pragma omp barrier
        if (!collective_error_.empty()) throw std::runtime_error(collective_error_);
    }
    template<class Function> void local_phase(int rank,Function work) {
        rank_errors_[rank].clear();
        try { work(); }
        catch (const std::exception& error) { rank_errors_[rank]=error.what(); }
        catch (...) { rank_errors_[rank]="unknown clump communication error"; }
    }
    template<class T> static void allocate(T*& p,size_t n){if(n)CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));}
    template<class T> static void upload(T* p,const std::vector<T>& h){if(!h.empty())CUDA_CHECK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
    bool read_participants(int g,const ParticleSoA& p,int n,cudaStream_t stream) {
        auto& r=ranks_[g];std::vector<int> ids(n);
        if(n)CUDA_CHECK(cudaMemcpyAsync(ids.data(),p.clump_id,n*sizeof(int),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ids.erase(std::remove_if(ids.begin(),ids.end(),[](int cid){return cid<=0;}),ids.end());
        r.member_count=ids.size();
        std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        for(int cid:ids)if(cid>k_)throw std::runtime_error("distributed clump ID exceeds initialized body count");
        const bool changed=ids!=r.cids;
        r.cids=std::move(ids);return changed;
    }
    void free_routes() {
        const int n=ranks_.size();
        for(int o=0;o<n;++o)for(int p=0;p<n;++p){auto& e=edge(o,p);
            CUDA_CHECK(cudaSetDevice(o));cudaFree(e.own);cudaFree(e.pose_send);cudaFree(e.wrench_recv);
            CUDA_CHECK(cudaSetDevice(p));cudaFree(e.part);cudaFree(e.pose_recv);cudaFree(e.wrench_send);e=Edge{};
        }
        for(int g=0;g<n;++g){CUDA_CHECK(cudaSetDevice(g));auto& r=ranks_[g];
            cudaFree(r.replica);cudaFree(r.partial);r.replica=nullptr;r.partial=nullptr;
        }
    }
    void build_routes() {
        free_routes();const int n=ranks_.size();
        for(int p=0;p<n;++p){auto& r=ranks_[p];CUDA_CHECK(cudaSetDevice(p));
            allocate(r.replica,r.cids.size());allocate(r.partial,r.cids.size());
            std::vector<int> lookup(k_+1,-1);
            for(size_t s=0;s<r.cids.size();++s)lookup[r.cids[s]]=s;
            upload(r.lookup,lookup);
            for(int o=0;o<n;++o){std::vector<int> own,part;
                for(size_t s=0;s<r.cids.size();++s){const int c=r.cids[s]-1;if(owner_[c]==o){own.push_back(owner_slot_[c]);part.push_back(s);}}
                auto& e=edge(o,p);e.n=own.size();
                CUDA_CHECK(cudaSetDevice(o));allocate(e.own,e.n);upload(e.own,own);
                CUDA_CHECK(cudaSetDevice(p));allocate(e.part,e.n);upload(e.part,part);
                if(o!=p){allocate(e.pose_recv,e.n);allocate(e.wrench_send,e.n);
                    CUDA_CHECK(cudaSetDevice(o));allocate(e.pose_send,e.n);allocate(e.wrench_recv,e.n);}
            }
        }
        ++route_generation_;
        for(int g=0;g<n;++g){int remote=0;
            for(int cid:ranks_[g].cids)remote+=owner_[cid-1]!=g;
            std::cout<<"[clumps-routes] generation="<<route_generation_<<" rank="<<g
                     <<" owner_bodies="<<ranks_[g].n_owned<<" participant_bodies="<<ranks_[g].cids.size()
                     <<" remote_bodies="<<remote<<" owned_members="<<ranks_[g].member_count<<'\n';
        }
    }
public:
    DistributedClumps()=default;
    DistributedClumps(const DistributedClumps&)=delete;
    DistributedClumps& operator=(const DistributedClumps&)=delete;
    ~DistributedClumps(){shutdown();}
    bool enabled()const{return k_>0;}
    double max_diameter()const{return max_diameter_;}
    // Initialization alone stages global body properties. No body state or
    // wrench visits the CPU during an ordinary timestep.
    void initialize(const ClumpStateSoA& cs,const ParticleSoA& all,
                    const std::vector<GPUDomainState>& states,const DomainParamsGPU& domain,
                    const std::vector<DistributedClumpBody>* restored=nullptr,
                    const SimConfig* config=nullptr,double time=0,bool preserve_owner=false,
                    const PeriodicBoundaryConfig* active_pbc=nullptr) {
        if(config){
            shear_=config->shear_box;
            if(active_pbc)pbc_=*active_pbc;
            else if(shear_.enabled)
                throw std::invalid_argument("Distributed shear clumps require the active wall-manager boundary configuration");
            else pbc_=extract_pbc_config(*config);
        }
        k_=restored?static_cast<int>(restored->size()):cs.num_clumps;if(!k_)return;
        const char* debug=std::getenv("GRANIT_CLUMP_DIAGNOSTICS");diagnostics_=debug&&std::string(debug)=="1";
        const int n=states.size();ranks_.resize(n);edges_.resize(n*n);changed_.assign(n,0);rank_errors_.resize(n);
        owner_.resize(k_);owner_slot_.resize(k_);
        ClumpStateSoA h;
        if(!restored)clump_get_states(h,nullptr,cs);
        std::vector<std::vector<DistributedClumpBody>> bodies(n);
        for(int c=0;c<k_;++c){DistributedClumpBody b{};b.cid=c+1;
            if(restored)b=(*restored)[c];
            else {
            b.pose={{h.com_x[c],h.com_y[c],h.com_z[c]},{h.vel_x[c],h.vel_y[c],h.vel_z[c]},
                    {h.omega_x[c],h.omega_y[c],h.omega_z[c]},{h.quat_w[c],h.quat_x[c],h.quat_y[c],h.quat_z[c]}};
            b.mass=h.mass[c];b.inertia[0]=h.inertia_xx[c];b.inertia[1]=h.inertia_xy[c];b.inertia[2]=h.inertia_xz[c];
            b.inertia[3]=h.inertia_yy[c];b.inertia[4]=h.inertia_yz[c];b.inertia[5]=h.inertia_zz[c];
            for(int q=0;q<19;++q)b.p3[q]=h.multipole_p3[q][c];
            b.wrench={{h.net_force_x[c],h.net_force_y[c],h.net_force_z[c]}, {h.net_torque_x[c],h.net_torque_y[c],h.net_torque_z[c]}};
            }
            if(shear_.enabled&&!restored){const auto phase=phase_at(time);
                if(!clump_shear_wrap(b.pose.com[0],b.pose.com[1],b.pose.vel[1],b.pose.winding[0],b.pose.winding[1],
                    pbc_.box_min_x,pbc_.box_min_y,pbc_.box_length_x,pbc_.box_length_y,phase))
                    throw std::runtime_error("Initial clump shear winding exceeds int64");}
            const int g=restored&&preserve_owner?b.owner:DomainManager::position_to_gpu(b.pose.com[0],b.pose.com[1],b.pose.com[2],domain,0);
            if(g<0||g>=n)throw std::runtime_error("Invalid authoritative owner in clump checkpoint");
            b.owner=g;
            owner_[c]=g;owner_slot_[c]=bodies[g].size();bodies[g].push_back(b);
        }
        if(!restored)free_clump_host_state(h,nullptr);
        const int count=all.num_local;std::vector<int> ids(count);std::vector<float> x(count),y(count),z(count),radius(count);
        if(count){CUDA_CHECK(cudaMemcpy(ids.data(),all.clump_id,count*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(x.data(),all.clump_off_x,count*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y.data(),all.clump_off_y,count*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(z.data(),all.clump_off_z,count*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(radius.data(),all.radius,count*sizeof(float),cudaMemcpyDeviceToHost));}
        for(int i=0;i<count;++i)if(ids[i]>0)max_diameter_=std::max(max_diameter_,2*(std::sqrt(double(x[i])*x[i]+double(y[i])*y[i]+double(z[i])*z[i])+radius[i]));
        for(int g=0;g<n;++g){CUDA_CHECK(cudaSetDevice(g));auto& r=ranks_[g];r.stream=states[g].compute_stream;r.n_owned=bodies[g].size();
            allocate(r.owned,r.n_owned);upload(r.owned,bodies[g]);allocate(r.lookup,k_+1);
            CUDA_CHECK(cudaEventCreateWithFlags(&r.ready,cudaEventDisableTiming));
            if(diagnostics_)allocate(r.contact_counter,1);
            allocate(r.shear_error,1);CUDA_CHECK(cudaMemset(r.shear_error,0,sizeof(int)));
            read_participants(g,states[g].local_state.particles,states[g].subdomain.num_local_particles,r.stream);
        }
        build_routes();
        // Initial replicas can be populated directly while the one-time host
        // initialization rows are available, avoiding a collective here.
        for(int g=0;g<n;++g){CUDA_CHECK(cudaSetDevice(g));std::vector<DistributedClumpPose> poses;
            for(int cid:ranks_[g].cids)poses.push_back(bodies[owner_[cid-1]][owner_slot_[cid-1]].pose);
            upload(ranks_[g].replica,poses);
        }
        CUDA_CHECK(cudaSetDevice(0));
        std::cout<<"[clumps] distributed bodies="<<k_<<" fixed owners=initial-COM max_diameter="<<max_diameter_<<'\n';
    }
    void validate_leaf_width(double root_size,int depth)const {
        const double width=std::ldexp(root_size,-depth);
        if(!(std::isfinite(width)&&max_diameter_<width*(1-1.e-6))) {
            std::ostringstream error;error<<"distributed small-clump diameter must satisfy 2R < finest leaf width with margin: diameter="<<max_diameter_<<" leaf_width="<<width;
            throw std::runtime_error(error.str());
        }
    }
    // All persistent GPU workers call these collective methods in the same
    // order. Barriers protect host route generations; CUDA events protect the
    // device messages. PeerTransfer selects NVLink/P2P or pinned host staging.
    void refresh(int rank,const ParticleSoA& particles,int count,
                 bool membership_changed,cudaStream_t stream,double time=0) {
        local_phase(rank,[&] {
            changed_[rank]=0;
            if (!membership_changed) return;
            const int previous_members=ranks_[rank].member_count;
            changed_[rank]=read_participants(rank,particles,count,stream)?1:0;
            if (diagnostics_) {
                #pragma omp critical(clump_diagnostics)
                std::cout<<"[clumps-migration] rank="<<rank<<" owned_particles="<<count
                         <<" previous_members="<<previous_members<<" owned_members="<<ranks_[rank].member_count
                         <<" participant_set_changed="<<changed_[rank]<<'\n';
            }
        });
        agree_errors();
        #pragma omp single
        {
            local_phase(rank,[&] {
                rebuilt_=std::any_of(changed_.begin(),changed_.end(),[](int value){return value!=0;});
                if (rebuilt_) {
                    for (int gpu=0;gpu<int(ranks_.size());++gpu) {
                        CUDA_CHECK(cudaSetDevice(gpu));
                        CUDA_CHECK(cudaStreamSynchronize(ranks_[gpu].stream));
                    }
                    build_routes();
                }
            });
        }
        agree_errors();
        CUDA_CHECK(cudaSetDevice(rank));
        if (rebuilt_) sync(rank,particles,count,stream,time);
    }

    void integrate(int rank,double dt,bool drift,cudaStream_t stream,double time=0) {
        auto& gpu=ranks_[rank];
        if (gpu.n_owned)
            distributed_clump_integrate<<<blocks(gpu.n_owned),256,0,stream>>>(
                gpu.owned,gpu.n_owned,dt,drift,shear_,pbc_,phase_at(time),gpu.shear_error);
    }

    void sync(int rank,ParticleSoA particles,int count,cudaStream_t stream,double time=0) {
        auto& gpu=ranks_[rank];
        const int gpu_count=ranks_.size();
        local_phase(rank,[&] {
            auto& local=edge(rank,rank);
            if (local.n)
                distributed_clump_copy_local_pose<<<blocks(local.n),256,0,stream>>>(
                    gpu.owned,local.own,local.part,gpu.replica,local.n);
            for (int participant=0;participant<gpu_count;++participant) {
                auto& route=edge(rank,participant);
                if (participant!=rank && route.n)
                    distributed_clump_pack_pose<<<blocks(route.n),256,0,stream>>>(
                        gpu.owned,route.own,route.pose_send,route.n);
            }
            CUDA_CHECK(cudaEventRecord(gpu.ready,stream));
        });
        agree_errors();
        local_phase(rank,[&] {
            for (int owner=0;owner<gpu_count;++owner) {
                auto& route=edge(owner,rank);
                if (owner==rank || !route.n) continue;
                CUDA_CHECK(cudaStreamWaitEvent(stream,ranks_[owner].ready,0));
                copy_peer_payload(route.pose_recv,rank,route.pose_send,owner,
                                  route.n*sizeof(DistributedClumpPose),stream);
                distributed_clump_scatter_pose<<<blocks(route.n),256,0,stream>>>(
                    route.pose_recv,route.part,gpu.replica,route.n);
            }
            if (count)
                distributed_clump_sync_members<<<blocks(count),256,0,stream>>>(
                    particles,count,gpu.lookup,gpu.replica,shear_,pbc_,phase_at(time),gpu.shear_error);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int error=0;CUDA_CHECK(cudaMemcpy(&error,gpu.shear_error,sizeof(int),cudaMemcpyDeviceToHost));
            if(error)throw std::overflow_error("Clump/member shear winding exceeds int64");
        });
        // Both the direct and staged paths finish reading every send buffer
        // before the next event/message generation may overwrite it.
        agree_errors();
    }

    void aggregate(int rank,ParticleSoA particles,int count,cudaStream_t stream) {
        auto& gpu=ranks_[rank];
        const int gpu_count=ranks_.size();
        local_phase(rank,[&] {
            if (!gpu.cids.empty())
                CUDA_CHECK(cudaMemsetAsync(gpu.partial,0,
                    gpu.cids.size()*sizeof(DistributedClumpWrench),stream));
            if (count)
                distributed_clump_reduce_members<<<blocks(count),256,0,stream>>>(
                    particles,count,gpu.lookup,gpu.replica,gpu.partial);
            for (int owner=0;owner<gpu_count;++owner) {
                auto& route=edge(owner,rank);
                if (owner!=rank && route.n)
                    distributed_clump_pack_wrench<<<blocks(route.n),256,0,stream>>>(
                        gpu.partial,route.part,route.wrench_send,route.n);
            }
            CUDA_CHECK(cudaEventRecord(gpu.ready,stream));
        });
        agree_errors();
        local_phase(rank,[&] {
            if (gpu.n_owned)
                distributed_clump_zero_owned<<<blocks(gpu.n_owned),256,0,stream>>>(gpu.owned,gpu.n_owned);
            auto& local=edge(rank,rank);
            if (local.n)
                distributed_clump_add_local_wrench<<<blocks(local.n),256,0,stream>>>(
                    gpu.partial,local.part,local.own,gpu.owned,local.n);
            // Rank order fixes the order of inter-rank additions. Each route
            // has exactly one entry for a body, so these additions need no
            // device atomics beyond the member-to-participant local sum.
            for (int participant=0;participant<gpu_count;++participant) {
                auto& route=edge(rank,participant);
                if (participant==rank || !route.n) continue;
                CUDA_CHECK(cudaStreamWaitEvent(stream,ranks_[participant].ready,0));
                copy_peer_payload(route.wrench_recv,rank,route.wrench_send,participant,
                                  route.n*sizeof(DistributedClumpWrench),stream);
                distributed_clump_add_wrench<<<blocks(route.n),256,0,stream>>>(
                    route.wrench_recv,route.own,gpu.owned,route.n);
            }
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        agree_errors();
    }

    void diagnose_contacts(int g,ParticleSoA p,const ContactPair* pairs,int count,int local,
                           int64_t step,cudaStream_t stream,double time=0) {
        if(!diagnostics_||!count)return;auto* counter=ranks_[g].contact_counter;
        CUDA_CHECK(cudaMemsetAsync(counter,0,sizeof(unsigned long long),stream));
        distributed_clump_count_remote_contacts<<<blocks(count),256,0,stream>>>(p,pairs,count,local,counter,shear_,pbc_,phase_at(time));
        unsigned long long contacts=0;
        CUDA_CHECK(cudaMemcpyAsync(&contacts,counter,sizeof(contacts),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(contacts){
            #pragma omp critical(clump_diagnostics)
            std::cout<<"[clumps-contact] rank="<<g<<" step="<<step
                     <<" remote_interclump_overlap_pairs="<<contacts<<'\n';
        }
    }
    std::vector<DistributedClumpBody> snapshot() const {
        std::vector<DistributedClumpBody> all(k_);
        for(int g=0;g<int(ranks_.size());++g){CUDA_CHECK(cudaSetDevice(g));const auto& r=ranks_[g];std::vector<DistributedClumpBody> rows(r.n_owned);
            if(r.n_owned)CUDA_CHECK(cudaMemcpy(rows.data(),r.owned,r.n_owned*sizeof(DistributedClumpBody),cudaMemcpyDeviceToHost));
            for(const auto& b:rows)all[b.cid-1]=b;}
        return all;
    }
    void save(const std::string& directory,int64_t step,double time) {
        if(!enabled())return;
        auto all=snapshot();
        if(shear_.enabled)for(auto& body:all){auto& pose=body.pose;double ax,ay,az;
            hill_acceleration(shear_,pose.com[0],pose.com[2],pose.vel[0],pose.vel[1],ax,ay,az);
            body.wrench.f[0]+=body.mass*ax;body.wrench.f[1]+=body.mass*ay;body.wrench.f[2]+=body.mass*az;
            const auto omega=distributed_clump_rotate(pose.quat,{pose.omega[0],pose.omega[1],pose.omega[2]});
            const auto torque=clump_hill_torque(body.inertia,pose.quat,omega,shear_);
            body.wrench.t[0]+=torque.x;body.wrench.t[1]+=torque.y;body.wrench.t[2]+=torque.z;}

        std::ostringstream path;path<<directory<<"/clump_states_step_"<<std::setfill('0')<<std::setw(10)<<step<<".bin";
        std::ofstream file(path.str(),std::ios::binary);if(!file)throw std::runtime_error("cannot write distributed clump states: "+path.str());
        const char magic[8]={'C','L','U','M','P',0,0,0};const uint32_t version=2;
        file.write(magic,8);file.write(reinterpret_cast<const char*>(&version),4);file.write(reinterpret_cast<const char*>(&k_),4);
        file.write(reinterpret_cast<const char*>(&step),8);file.write(reinterpret_cast<const char*>(&time),8);
        for(const auto& b:all){file.write(reinterpret_cast<const char*>(&b.cid),4);
            file.write(reinterpret_cast<const char*>(b.pose.com),24);file.write(reinterpret_cast<const char*>(b.pose.vel),24);
            file.write(reinterpret_cast<const char*>(b.pose.quat),32);file.write(reinterpret_cast<const char*>(b.pose.omega),24);
            file.write(reinterpret_cast<const char*>(&b.mass),8);
            const double inertia[6]={b.inertia[0],b.inertia[3],b.inertia[5],b.inertia[1],b.inertia[2],b.inertia[4]};
            file.write(reinterpret_cast<const char*>(inertia),48);file.write(reinterpret_cast<const char*>(b.p3),19*8);
            file.write(reinterpret_cast<const char*>(&b.wrench),48);}
        if(!file)throw std::runtime_error("failed writing distributed clump states: "+path.str());
    }
    void shutdown()noexcept {
        if(ranks_.empty())return;int previous=0;cudaGetDevice(&previous);
        try{for(int g=0;g<int(ranks_.size());++g){cudaSetDevice(g);cudaStreamSynchronize(ranks_[g].stream);}free_routes();}catch(...){}
        for(int g=0;g<int(ranks_.size());++g){cudaSetDevice(g);cudaFree(ranks_[g].owned);cudaFree(ranks_[g].lookup);cudaFree(ranks_[g].contact_counter);cudaFree(ranks_[g].shear_error);if(ranks_[g].ready)cudaEventDestroy(ranks_[g].ready);}
        ranks_.clear();edges_.clear();k_=0;cudaSetDevice(previous);
    }
};
} // namespace cudadem

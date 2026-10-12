#!/usr/bin/env python3
"""Host-replay actual compact LET kernels against a frozen dense reference.

Compiles extracted production functions, changing only CUDA execution plumbing.
The fixture supplies coordinate lookup, node storage and serialized atomics.
It checks exact route bytes, particle-needed flags, counts, shared-moment masks,
and ordered far/direct references. It does not prove GPU race freedom or forces.
"""
from pathlib import Path
import argparse
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=ROOT)
parser.add_argument("--emit", type=Path, help="Also save generated replay C++")
args = parser.parse_args()
SRC = args.source / "src/core/acceleration/body_fitted_fmm"


def function(source, name):
    match = re.search(r"\b" + name + r"\(", source)
    if match is None:
        raise ValueError("Production function missing: " + name)
    start = source.rfind("\n", 0, match.start()) + 1
    previous = source.rfind("\n", 0, start - 1) + 1
    if source[previous:start].lstrip().startswith("template"):
        start = previous
    brace = source.index("{", match.end())
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise ValueError("Unclosed production function: " + name)


prelude = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
using cudaStream_t=void*;
struct Dim {int x=0;};
struct double4 {double x=0,y=0,z=0,w=0;};
inline double4 make_double4(double x,double y,double z,double w){return {x,y,z,w};}
namespace cudadem {struct ShearSourceRouting {};}
Dim blockIdx,threadIdx,blockDim{1};
template<class T> T atomicExch(T* p,T x){T old=*p;*p=x;return old;}
template<class T> T atomicOr(T* p,T x){T old=*p;*p|=x;return old;}
template<class T> T atomicAdd(T* p,T x){T old=*p;*p+=x;return old;}
inline int __ffs(int x){return __builtin_ffs(x);}
inline void __trap(){throw std::runtime_error("production trap");}
template<class F> void launch(int n,F f){for(int i=0;i<n;++i){blockIdx.x=i;f();}}
'''

arena = (SRC / "grouped_interaction_arena.cuh").read_text().replace("#include <cuda_runtime.h>", "")
arena = arena.replace("#pragma once", "")
traits = (SRC / "grouped_stencil_traits.cuh").read_text()
traits = traits.replace("#pragma once", "")
traits = re.sub(r'^#include[^\n]*\n', '', traits, flags=re.M)
traits = re.sub(r'\b(GROUPED_D[23]_(?:CROSS|ALL_FAR|MIXED))\b', r'HOST_\1', traits)
types = r'''
namespace body_fitted_fmm {
struct Double3 {double x=0,y=0,z=0;};
struct UnifiedNode {
    Double3 center{}; uint64_t compact_morton=0; int level=0;
    uint32_t count=0,global_count=0,target_count=0,flags=0;
    uint8_t childMask=0; uint32_t childBase=0;
};
constexpr uint32_t UNIFIED_NODE_FLAG_IS_LEAF=1;
struct FMMCellKey {uint64_t morton=0;int level=0;};
struct FMMSourceParticle {double x,y,z;float mass;int original;};
struct FMMSourceReply {uint32_t begin,count,global_count;float coeffs[32];};
struct PersistentTreeContext {
    struct {UnifiedNode* unified_nodes=nullptr;const int* children_compact=nullptr;} dd;
    struct {GroupedTargetInteractions* d_targets=nullptr;} arena;
    struct {int num_nodes=0;double root_cell_size=1;} metadata;
    struct {double cutoff_radius=0;} command;
    std::map<std::pair<int,uint64_t>,int> lookup;
};
template<int D> int get_child_idx(const UnifiedNode* nodes,const int* children,int parent,int slot) {
    const auto& n=nodes[parent];
    if(!(n.childMask&(1u<<slot)))return -1;
    return children[n.childBase+__builtin_popcount(unsigned(n.childMask)&((1u<<slot)-1u))];
}
template<int D> uint64_t morton_encode(uint32_t x,uint32_t y,uint32_t z) {
    uint64_t key=0;uint32_t xyz[3]={x,y,z};
    for(int bit=0;bit<20;++bit)for(int axis=0;axis<D;++axis)
        key|=uint64_t((xyz[axis]>>bit)&1u)<<(D*bit+axis);
    return key;
}
template<int D> void morton_decode(uint64_t code,int level,uint32_t& x,uint32_t& y,uint32_t& z) {
    uint32_t xyz[3]{};
    for(int bit=0;bit<level;++bit)for(int axis=0;axis<D;++axis)
        xyz[axis]|=((code>>(D*bit+axis))&1u)<<bit;
    x=xyz[0];y=xyz[1];z=xyz[2];
}
template<int D> int persistent_coordinate_lookup(const PersistentTreeContext* ctx,int level,int x,int y,int z) {
    if(x<0||y<0||z<0||x>=(1<<level)||y>=(1<<level)||(D==3&&z>=(1<<level)))return -1;
    auto found=ctx->lookup.find({level,morton_encode<D>(x,y,z)});
    return found==ctx->lookup.end()?-1:found->second;
}
inline uint64_t grouped_direct_threshold(int,bool leaf){return leaf?16:64;}
enum class CutoffRelation : uint8_t {Inside,Outside,Straddles};
struct FMMRemoteRef {int peer=-1,node=-1;};
struct FMMRemoteView {
    const UnifiedNode* nodes=nullptr;const int* children=nullptr;
    const GroupedTargetInteractions* far_routes=nullptr;
    const GroupedTargetInteractions* direct_routes=nullptr;
    int count=0;const int* route_index=nullptr;
};
}
'''
gpu = (SRC / "distributed_fmm_gpu.cuh").read_text()
plan = (SRC / "remote_interaction_plan.cuh").read_text()
view = (SRC / "remote_interaction_lists.cuh").read_text()
cutoff = (SRC / "persistent_tree_traversal.cu").read_text()
helpers = "\n".join([function(gpu, "gpu_cell_code"), function(gpu, "fmm_halo_lookup"),
                      function(cutoff, "persistent_cutoff_relation"), function(view, "fmm_remote_route_row")])
names = ["fmm_halo_initialize_routes_kernel", "fmm_halo_count_routes_kernel",
         "fmm_halo_inverse_routes_kernel", "fmm_halo_moment_routes_kernel"]
current = "\n".join(function(gpu, name) for name in names)
current += "\n" + "\n".join(function(plan, name) for name in
                             ("fmm_remote_far_ref", "fmm_remote_direct_row", "fmm_remote_has_far"))
current += "\n" + function((SRC / "distributed_fmm.cuh").read_text(), "fmm_owned_node_work_kernel")
reference = (HERE / "dense_halo_routes_reference.inc").read_text()

cases = r'''
using namespace body_fitted_fmm;
static uint64_t checked_routes=0,checked_refs=0;
static void require(bool value,const char* why) {
    if(!value)throw std::runtime_error(why);
}
static bool same(FMMRemoteRef a,FMMRemoteRef b){return a.peer==b.peer&&a.node==b.node;}
static bool vacant(const GroupedTargetInteractions& r) {
    for(auto n:r.cross_node)if(n!=GROUPED_ROUTE_INVALID_NODE)return false;
    for(auto n:r.block_parent)if(n!=GROUPED_ROUTE_INVALID_NODE)return false;
    return true;
}
template<int D> UnifiedNode node(uint64_t code,int level,int count) {
    UnifiedNode n{};n.level=level;n.compact_morton=code;
    n.count=n.global_count=count;n.target_count=1;
    uint32_t x=0,y=0,z=0;morton_decode<D>(code,level,x,y,z);
    double h=std::ldexp(1.,-level);n.center={(x+.5)*h,(y+.5)*h,(z+.5)*h};
    if(level==3)n.flags=UNIFIED_NODE_FLAG_IS_LEAF;
    return n;
}
struct Peer {
    std::vector<UnifiedNode> nodes;std::vector<int> children;
    std::vector<FMMCellKey> keys;
    std::vector<GroupedTargetInteractions> dense,compact,dense_moment,compact_moment;
    std::vector<int> dense_needed,compact_needed,map,ids,slots,owners;
};
template<int D> void one_case(int seed,int ranks,bool empty_targets,bool empty_sources,
                             bool disable_fallback,double cutoff,int population) {
    using T=GroupedStencilTraits<D>;std::mt19937 rng(seed);
    PersistentTreeContext ctx{};ctx.command.cutoff_radius=cutoff;
    std::vector<UnifiedNode> local;
    for(int level=1;level<=3;++level)for(uint64_t code=0;code<(1ull<<(D*level));++code) {
        const int id=local.size();local.push_back(node<D>(code,level,population));
        local.back().target_count=empty_targets?0:1;
        // Keep parent chains complete but leave sparse holes in the leaf set.
        if(level==3 && (rng()%5==0))local.back().target_count=0;
        ctx.lookup[{level,code}]=id;
    }
    std::vector<GroupedTargetInteractions> native(local.size());
    for(size_t i=0;i<local.size();++i)native[i].target_parent=local[i].level>1?
        ctx.lookup.at({local[i].level-1,local[i].compact_morton>>D}):-1;
    ctx.dd.unified_nodes=local.data();ctx.arena.d_targets=native.data();ctx.metadata.num_nodes=local.size();
    std::vector<Peer> peers(ranks);
    std::vector<FMMRemoteView> dense_views(ranks),compact_views(ranks);
    for(int p=0;p<ranks;++p) {
        auto& a=peers[p];
        if(!empty_sources)for(int level=1;level<=3;++level)
            for(uint64_t code=0;code<(1ull<<(D*level));++code)
                if((rng()%3)!=0) {a.keys.push_back({code,level});a.nodes.push_back(node<D>(code,level,1+rng()%population));}
        const int n=a.nodes.size();
        for(int s=0;s<n;++s) {
            auto& source=a.nodes[s];source.childBase=a.children.size();
            for(int c=0;c<T::CHILD_COUNT;++c) {
                int child=fmm_halo_lookup<D>(a.keys.data(),n,{(source.compact_morton<<D)|uint64_t(c),source.level+1});
                if(child>=0){source.childMask|=uint8_t(1u<<c);a.children.push_back(child);}
            }
        }
        a.dense.resize(local.size());a.dense_needed.resize(n);
        launch(std::max(int(local.size()),n),[&]{dense_fmm_halo_initialize_routes_kernel(&ctx,local.size(),a.dense.data(),a.dense_needed.data(),n);});
        launch(n*T::CHILD_COUNT,[&]{dense_fmm_halo_inverse_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.keys.data(),n,3,disable_fallback,a.dense.data(),a.dense_needed.data());});
        a.map.assign(local.size(),0);
        launch(n*T::CHILD_COUNT,[&]{fmm_halo_inverse_routes_kernel<D,true>(&ctx,a.nodes.data(),a.children.data(),a.keys.data(),n,3,disable_fallback,nullptr,nullptr,a.map.data());});
        for(size_t i=0;i<a.map.size();++i)if(a.map[i])a.ids.push_back(i);
        require(std::is_sorted(a.ids.begin(),a.ids.end()),"selected IDs not ascending");
        a.map.assign(local.size(),-1);a.compact.resize(a.ids.size());a.compact_needed.resize(n);
        launch(std::max(int(a.ids.size()),n),[&]{fmm_halo_initialize_routes_kernel(&ctx,a.ids.size(),a.compact.data(),a.compact_needed.data(),n,a.ids.data(),a.map.data());});
        launch(n*T::CHILD_COUNT,[&]{fmm_halo_inverse_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.keys.data(),n,3,disable_fallback,a.compact.data(),a.compact_needed.data(),a.map.data());});
        require(a.dense_needed==a.compact_needed,"particle-needed flags differ");
        for(size_t i=0;i<local.size();++i) {
            if(a.map[i]<0)require(vacant(a.dense[i]),"mark pass lost dense route");
            else {require(std::memcmp(&a.dense[i],&a.compact[a.map[i]],sizeof(GroupedTargetInteractions))==0,"route bytes differ");++checked_routes;}
        }
        int old_counts[4]{},new_counts[4]{};
        launch(local.size(),[&]{dense_fmm_halo_count_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.dense.data(),local.size(),old_counts,false);});
        launch(a.ids.size(),[&]{fmm_halo_count_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.compact.data(),a.ids.size(),new_counts,false,a.ids.data());});
        require(std::equal(old_counts,old_counts+4,new_counts),"route counts differ");
        a.slots.assign(size_t(n)*(ranks+1),-1);a.owners.resize(n);
        for(int i=0;i<n;++i) {
            auto code=gpu_cell_code<D>(a.keys[i]);a.owners[i]=code%ranks;
            // Shared logical cells: some overlay the native node, others have one peer owner.
            if(code%4==0)a.slots[size_t(i)*(ranks+1)]=0;
        }
        a.dense_moment.resize(local.size());a.compact_moment.resize(a.ids.size());
        launch(local.size(),[&]{dense_fmm_halo_moment_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.dense.data(),a.dense_moment.data(),local.size(),p,ranks,a.slots.data(),a.owners.data());});
        launch(a.ids.size(),[&]{fmm_halo_moment_routes_kernel<D>(&ctx,a.nodes.data(),a.children.data(),a.compact.data(),a.compact_moment.data(),a.ids.size(),p,ranks,a.slots.data(),a.owners.data(),a.ids.data());});
        for(size_t row=0;row<a.ids.size();++row)
            require(std::memcmp(&a.dense_moment[a.ids[row]],&a.compact_moment[row],sizeof(GroupedTargetInteractions))==0,"moment ownership masks differ");
        // The partition/depth controller consumes these work estimates. Empty
        // compact planes must contribute zero rather than native routes.
        std::vector<double4> old_work(local.size()),new_work(local.size());
        if(n)for(bool foreign:{false,true}) {
            launch(local.size(),[&]{dense_fmm_owned_node_work_kernel<D>(&ctx,local.size(),0,0,old_work.data(),{},0,
                a.nodes.data(),a.children.data(),a.dense_moment.data(),foreign);});
            launch(local.size(),[&]{fmm_owned_node_work_kernel<D>(&ctx,local.size(),0,0,new_work.data(),{},0,
                a.nodes.data(),a.children.data(),a.compact_moment.data(),foreign,a.map.data());});
            require(std::memcmp(old_work.data(),new_work.data(),old_work.size()*sizeof(double4))==0,"owned-node work differs");
        }
        dense_views[p]={a.nodes.data(),a.children.data(),a.dense_moment.data(),a.dense.data(),n,nullptr};
        compact_views[p]={a.nodes.data(),a.children.data(),a.compact_moment.data(),a.compact.data(),n,a.map.data()};
    }
    for(size_t target=0;target<local.size();++target) {
        if(!local[target].target_count || local[target].level<2)continue;
        require(dense_fmm_remote_has_far<D>(target,local[target],dense_views.data(),ranks)==
                fmm_remote_has_far<D>(target,local[target],compact_views.data(),ranks),"far selection differs");
        for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
            int old_error=0,new_error=0;
            auto a=dense_fmm_remote_far_ref<D>(target,slot,local.data(),dense_views.data(),ranks,&old_error);
            auto b=fmm_remote_far_ref<D>(target,slot,local.data(),compact_views.data(),ranks,&new_error);
            require(same(a,b)&&old_error==new_error,"ordered far refs or duplicate-owner flags differ");++checked_refs;
        }
        if(local[target].flags&UNIFIED_NODE_FLAG_IS_LEAF) {
            auto a=dense_fmm_remote_direct_row<D,false>(target,&ctx,dense_views.data(),ranks,nullptr);
            auto b=fmm_remote_direct_row<D,false>(target,&ctx,compact_views.data(),ranks,nullptr);
            require(a==b,"direct counts differ");std::vector<FMMRemoteRef> x(a),y(b);
            dense_fmm_remote_direct_row<D,true>(target,&ctx,dense_views.data(),ranks,x.data());
            fmm_remote_direct_row<D,true>(target,&ctx,compact_views.data(),ranks,y.data());
            require(std::equal(x.begin(),x.end(),y.begin(),same),"ordered direct refs differ");checked_refs+=a;
        }
    }
}
template<int D> void missing_leaf_row() {
    // A finite-cutoff direct route on an ancestor must survive an absent leaf row.
    UnifiedNode nodes[2]{};nodes[0].level=2;nodes[1].level=3;
    nodes[0].target_count=nodes[1].target_count=1;
    GroupedTargetInteractions native[2]{};native[0].target_parent=-1;native[1].target_parent=0;
    PersistentTreeContext ctx{};ctx.dd.unified_nodes=nodes;ctx.arena.d_targets=native;ctx.metadata.num_nodes=2;
    GroupedTargetInteractions dense[2]{},compact[1]{};int flags[1]{};
    launch(2,[&]{dense_fmm_halo_initialize_routes_kernel(&ctx,2,dense,flags,0);});
    grouped_set_cross_route(dense[0],0,{0,1});compact[0]=dense[0];int map[2]={0,-1};
    FMMRemoteView a[4]{},b[4]{};
    for(int p=0;p<4;++p){a[p]={nodes,nullptr,dense,dense,1,nullptr};b[p]={nodes,nullptr,compact,compact,1,map};}
    std::vector<FMMRemoteRef> x(4),y(4);
    require(dense_fmm_remote_direct_row<D,true>(1,&ctx,a,4,x.data())==4,"dense ancestor fixture invalid");
    require(fmm_remote_direct_row<D,true>(1,&ctx,b,4,y.data())==4,"absent leaf lost direct ancestor");
    require(std::equal(x.begin(),x.end(),y.begin(),same),"ancestor direct peer order changed");
    for(int p=0;p<4;++p)require(y[p].peer==p && y[p].node==0,"explicit ancestor order failed");
}
template<int D> void run() {
    int cases=0;
    for(int ranks:{1,2,4})for(bool fallback:{false,true})for(double cutoff:{0.,.08,.35,2.}) {
        one_case<D>(982421+cases,ranks,false,false,!fallback,cutoff,fallback?3:100);++cases;
    }
    one_case<D>(42,4,true,false,true,0,100);++cases;
    one_case<D>(43,4,false,true,true,0,100);++cases;
    missing_leaf_row<D>();++cases;
    std::printf("D%d: %d compact/dense fixture cases PASS\n",D,cases);
}
int main(){run<2>();run<3>();std::printf("Exact route rows=%llu ordered references=%llu PASS\n",
    (unsigned long long)checked_routes,(unsigned long long)checked_refs);}
'''
generated = (prelude + arena + types + traits + "\nnamespace body_fitted_fmm {\n" +
             helpers + "\n" + reference + "\n" + current + "\n}\n" + cases)
if args.emit:
    args.emit.write_text(generated)
with tempfile.TemporaryDirectory(prefix="granit-compact-routes-") as tmp:
    cpp, binary = Path(tmp) / "replay.cpp", Path(tmp) / "replay"
    cpp.write_text(generated)
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unknown-pragmas",
                    "-I", str(SRC), str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

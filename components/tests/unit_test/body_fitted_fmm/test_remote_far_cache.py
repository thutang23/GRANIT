#!/usr/bin/env python3
"""Replay production FAR-cache kernels on CPU against the frozen dense resolver.

Serialized CUDA plumbing tests exact schedule semantics, not GPU race freedom.
Arbitrary node permutations exercise the cache's supplied remap; the production
binary-search remap is separately tested with its required sorted-key inputs.
"""
from pathlib import Path
import argparse
import ast
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT)
parser.add_argument('--emit', type=Path)
args = parser.parse_args()
SRC = args.source / 'src/core/acceleration/body_fitted_fmm'

# Reuse only literal fixture declarations: importing that test would run it.
fixture = ast.parse((HERE / 'test_compact_halo_routes.py').read_text())
literals = {n.targets[0].id: ast.literal_eval(n.value) for n in fixture.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('prelude', 'types')}
prelude = literals['prelude'] + '\ntemplate<class T> T atomicCAS(T* p,T a,T b){T old=*p;if(old==a)*p=b;return old;}\n'
types = literals['types']


def function(source, name):
    match = re.search(r'\b' + name + r'\(', source)
    if match is None:
        raise ValueError('Missing production function: ' + name)
    start = source.rfind('\n', 0, match.start()) + 1
    previous = source.rfind('\n', 0, start - 1) + 1
    if source[previous:start].lstrip().startswith('template'):
        start = previous
    brace = source.index('{', match.end())
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise ValueError('Unclosed function: ' + name)


arena = (SRC / 'grouped_interaction_arena.cuh').read_text().replace('#include <cuda_runtime.h>', '').replace('#pragma once', '')
traits = (SRC / 'grouped_stencil_traits.cuh').read_text().replace('#pragma once', '')
traits = re.sub(r'^#include[^\n]*\n', '', traits, flags=re.M)
traits = re.sub(r'\b(GROUPED_D[23]_(?:CROSS|ALL_FAR|MIXED))\b', r'HOST_\1', traits)
gpu = (SRC / 'distributed_fmm_gpu.cuh').read_text()
plan = (SRC / 'remote_interaction_plan.cuh').read_text()
view = (SRC / 'remote_interaction_lists.cuh').read_text()
cache_types = view[view.index('struct FMMFarCachePeerView'):view.index('struct FMMRemoteM2LList')]
helpers = '\n'.join([function(gpu, 'gpu_cell_code'), function(gpu, 'fmm_halo_lookup'),
                     function(view, 'fmm_remote_route_row')])
names = ('fmm_remote_far_ref', 'fmm_far_cache_find_row', 'fmm_far_cache_same_source',
         'fmm_far_cache_target_matches', 'fmm_far_cache_source_remap_kernel',
         'fmm_remote_far_cached_sources_kernel', 'fmm_far_cache_target_keys_kernel',
         'fmm_far_cache_snapshot_targets_kernel', 'fmm_far_cache_snapshot_children_kernel')
current = '\n'.join(function(plan, n) for n in names)
oracle = function((HERE / 'dense_halo_routes_reference.inc').read_text(), 'dense_fmm_remote_far_ref')

cases = r'''
using namespace body_fitted_fmm;
static uint64_t checked_refs=0,checked_rows=0,reused_rows=0,changed_rows=0;
static void require(bool x,const char* reason){if(!x)throw std::runtime_error(reason);}
static GroupedTargetInteractions blank(){
    GroupedTargetInteractions r{};
    for(auto& x:r.cross_node)x=GROUPED_ROUTE_INVALID_NODE;
    for(auto& x:r.block_parent)x=GROUPED_ROUTE_INVALID_NODE;
    return r;
}
struct Peer {
    std::vector<UnifiedNode> nodes;
    std::vector<int> children,ids,map,remap;
    std::vector<uint8_t> masks;
    std::vector<GroupedTargetInteractions> routes,dense;
    FMMRemoteView view(bool dense_view=false) const {
        return {nodes.data(),children.data(),dense_view?dense.data():routes.data(),nullptr,
                int(nodes.size()),dense_view?nullptr:map.data()};
    }
};
template<int D> void sources(Peer& p,std::vector<int> order){
    constexpr int C=1<<D;
    p.ids=order;p.nodes.resize(order.size());
    std::map<int,int> index;
    for(int i=0;i<int(order.size());++i)index[order[i]]=i;
    for(int i=0;i<int(order.size());++i){
        const int id=order[i],g=id/(C+1),at=id%(C+1);auto& n=p.nodes[i];
        n.level=at?3:2;n.compact_morton=at?(uint64_t(g)<<D)|(at-1):g;
        n.childBase=p.children.size();n.childMask=0;
        if(!at)for(int c=0;c<C;++c){auto it=index.find(g*(C+1)+c+1);
            if(it!=index.end()){n.childMask|=1u<<c;p.children.push_back(it->second);}}
    }
}
template<int D> void one_case(int seed,int ranks,int mutation){
    using T=GroupedStencilTraits<D>;constexpr int C=T::CHILD_COUNT;
    constexpr int S=T::CROSS_COUNT+T::BLOCK_COUNT*C;
    std::mt19937 rng(seed);const int oldn=24;
    std::vector<int> old_target_order(oldn);for(int i=0;i<oldn;++i)old_target_order[i]=i;
    std::shuffle(old_target_order.begin(),old_target_order.end(),rng);
    std::vector<UnifiedNode> oldnodes(oldn);
    std::vector<int> oldtargets(oldn);
    for(int i=0;i<oldn;++i){oldnodes[i].level=3;oldnodes[i].compact_morton=old_target_order[i];oldtargets[i]=i;}
    std::shuffle(oldtargets.begin(),oldtargets.end(),rng); // old FAR row ordering independent of native IDs
    std::vector<int> current_order=old_target_order;
    if(mutation==8){current_order.erase(current_order.begin()+3);current_order.push_back(47);} // target remove/insert
    std::shuffle(current_order.begin(),current_order.end(),rng);
    const int n=current_order.size();std::vector<UnifiedNode> nodes(n);std::vector<int> targets(n);
    for(int i=0;i<n;++i){nodes[i].level=3;nodes[i].compact_morton=current_order[i];targets[i]=i;}
    std::shuffle(targets.begin(),targets.end(),rng);
    std::vector<Peer> old(ranks),now(ranks);
    std::vector<int> identities(5*(C+1));for(int i=0;i<int(identities.size());++i)identities[i]=i;
    for(int peer=0;peer<ranks;++peer){
        auto a=identities,b=identities;std::shuffle(a.begin(),a.end(),rng);std::shuffle(b.begin(),b.end(),rng);
        if(mutation==5)b.erase(std::find(b.begin(),b.end(),1)); // remove child; parent survives
        if(mutation==6)b.erase(std::find(b.begin(),b.end(),0)); // referenced source deletion
        if(mutation==7)b.push_back(5*(C+1)); // unrelated source insertion
        if(mutation==10 && peer==ranks-1)b.clear(); // empty peer transition
        sources<D>(old[peer],a);sources<D>(now[peer],b);
        old[peer].remap.assign(a.size(),-1);
        for(int i=0;i<int(a.size());++i){auto it=std::find(b.begin(),b.end(),a[i]);
            if(it!=b.end())old[peer].remap[i]=it-b.begin();}
        old[peer].masks.resize(a.size());
        launch(a.size(),[&]{fmm_far_cache_snapshot_children_kernel(old[peer].nodes.data(),a.size(),old[peer].masks.data());});
        old[peer].map.assign(oldn,-1);now[peer].map.assign(n,-1);
        old[peer].dense.assign(oldn,blank());now[peer].dense.assign(n,blank());
    }
    auto original_route=[&](int key,int peer){
        auto r=blank();
        for(int c=0;c<T::CROSS_COUNT;++c)if((key+c)%ranks==peer){
            int identity=((key+c)%5)*(C+1);
            r.cross_node[c]=std::find(old[peer].ids.begin(),old[peer].ids.end(),identity)-old[peer].ids.begin();
            if((key+c)%4==0)r.cross_direct_bits|=1u<<c;}
        for(int b=0;b<T::BLOCK_COUNT;++b)if((key+b)%ranks==peer){
            int identity=((key+b)%5)*(C+1);
            r.block_parent[b]=std::find(old[peer].ids.begin(),old[peer].ids.end(),identity)-old[peer].ids.begin();
            r.block_direct_mask[b]=uint8_t((key+b)*13);r.block_omit_mask[b]=uint8_t((key+b)*7);}
        return r;
    };
    for(int peer=0;peer<ranks;++peer){
        for(int t=0;t<oldn;++t){int key=oldnodes[t].compact_morton;
            if((key+peer)%7==0)continue;
            old[peer].map[t]=old[peer].routes.size();auto r=original_route(key,peer);
            old[peer].routes.push_back(r);old[peer].dense[t]=r;}
        for(int t=0;t<n;++t){int key=nodes[t].compact_morton;
            bool present=(key+peer)%7!=0;
            if(mutation==9 && key==2)present=!present;
            if(!present || now[peer].nodes.empty())continue;
            auto r=original_route(key,peer);
            auto map_source=[&](uint32_t x){return x==GROUPED_ROUTE_INVALID_NODE || old[peer].remap[x]<0?
                GROUPED_ROUTE_INVALID_NODE:uint32_t(old[peer].remap[x]);};
            for(auto& x:r.cross_node)x=map_source(x);
            for(auto& x:r.block_parent)x=map_source(x);
            if(key==1){
                if(mutation==1)r.cross_direct_bits^=1u;
                if(mutation==2)r.block_direct_mask[0]^=1u;
                if(mutation==3)r.block_omit_mask[0]^=1u;
                if(mutation==4){ // source ownership moves, keeping canonical source identities
                    r=blank();if(peer==1%ranks){int id=0;
                        r.cross_node[0]=std::find(now[peer].ids.begin(),now[peer].ids.end(),id)-now[peer].ids.begin();}}
            }
            now[peer].map[t]=now[peer].routes.size();now[peer].routes.push_back(r);now[peer].dense[t]=r;
        }
    }
    std::vector<FMMRemoteView> oldviews,views,denseviews;
    std::vector<FMMFarCachePeerView> cache;
    for(int p=0;p<ranks;++p){oldviews.push_back(old[p].view());views.push_back(now[p].view());denseviews.push_back(now[p].view(true));
        cache.push_back({old[p].remap.data(),old[p].masks.data(),int(old[p].nodes.size())});}
    std::vector<FMMRemoteRef> before(size_t(S)*oldn),actual(size_t(S)*n);
    int error=0;
    for(int row=0;row<oldn;++row)for(int s=0;s<S;++s)
        before[size_t(s)*oldn+row]=fmm_remote_far_ref<D>(oldtargets[row],s,oldnodes.data(),oldviews.data(),ranks,&error);
    require(error==0,"fixture has duplicate old owners");
    std::vector<FMMFarCacheTargetPeer> snapshots(size_t(oldn)*ranks);
    launch(oldn*ranks,[&]{fmm_far_cache_snapshot_targets_kernel(oldtargets.data(),oldn,oldviews.data(),ranks,snapshots.data());});
    std::vector<uint64_t> keys(oldn);std::vector<int> rows(oldn);
    launch(oldn,[&]{fmm_far_cache_target_keys_kernel<D>(oldtargets.data(),oldn,oldnodes.data(),keys.data(),rows.data());});
    std::vector<std::pair<uint64_t,int>> sorted;
    for(int i=0;i<oldn;++i)sorted.push_back({keys[i],rows[i]});std::sort(sorted.begin(),sorted.end());
    for(int i=0;i<oldn;++i){keys[i]=sorted[i].first;rows[i]=sorted[i].second;}
    int stats[2]{};const int eligible=mutation==11?0:oldn; // full invalidation forces rebuild
    launch(n,[&]{fmm_remote_far_cached_sources_kernel<D>(targets.data(),n,nodes.data(),views.data(),ranks,actual.data(),&error,
        keys.data(),rows.data(),eligible,cache.data(),snapshots.data(),before.data(),true,stats);});
    require(error==0,"production fresh-resolution oracle failed");
    require(stats[0]+stats[1]==n,"cache row accounting");
    if(mutation==0 || mutation==7)require(stats[0]==n,"unchanged canonical rows did not reuse");
    if(mutation==11)require(stats[0]==0,"ineligible geometry reused rows");
    if(mutation==8)require(stats[1]>=1,"new target reused old row");
    if(mutation>=1&&mutation<=6)require(stats[1]>=1,"route/source mutation failed to invalidate");
    reused_rows+=stats[0];changed_rows+=stats[1];checked_rows+=n;
    for(int row=0;row<n;++row)for(int s=0;s<S;++s){
        auto expected=dense_fmm_remote_far_ref<D>(targets[row],s,nodes.data(),denseviews.data(),ranks,&error);
        auto a=actual[size_t(s)*n+row];require(a.peer==expected.peer&&a.node==expected.node,"cached/frozen dense reference mismatch");
        ++checked_refs;}
    require(error==0,"fixture has duplicate new owners");
    // Negative test demonstrates that the runtime oracle detects corruption in a reused reference.
    if(mutation==0 && seed==1 && ranks==1){bool poisoned=false;
        for(auto& x:before)if(x.node>=0){x.peer=(x.peer+1)%ranks;x.node=0;poisoned=true;break;}
        if(poisoned){int bad=0;launch(n,[&]{fmm_remote_far_cached_sources_kernel<D>(targets.data(),n,nodes.data(),views.data(),ranks,
            actual.data(),&bad,keys.data(),rows.data(),oldn,cache.data(),snapshots.data(),before.data(),true,nullptr);});
            require(bad==2,"runtime oracle did not detect poisoned cache");}}
}
template<int D> void remap_test(){
    std::vector<FMMCellKey> old{{0,2},{1,2},{3,2},{7,2},{0,3},{6,3}};
    std::vector<FMMCellKey> now{{0,2},{2,2},{3,2},{7,2},{0,3},{5,3},{6,3}};
    auto less=[](FMMCellKey a,FMMCellKey b){return gpu_cell_code<D>(a)<gpu_cell_code<D>(b);};
    std::sort(old.begin(),old.end(),less);std::sort(now.begin(),now.end(),less);
    std::vector<int> remap(old.size());
    launch(old.size(),[&]{fmm_far_cache_source_remap_kernel<D>(old.data(),old.size(),now.data(),now.size(),remap.data());});
    for(int i=0;i<int(old.size());++i){int expected=-1;for(int j=0;j<int(now.size());++j)
        if(old[i].level==now[j].level&&old[i].morton==now[j].morton)expected=j;
        require(remap[i]==expected,"canonical remap lookup differs from linear oracle");}
    launch(old.size(),[&]{fmm_far_cache_source_remap_kernel<D>(old.data(),old.size(),nullptr,0,remap.data());});
    for(auto x:remap)require(x==-1,"empty peer remap");
    FMMFarCachePeerView empty{};
    require(fmm_far_cache_same_source(GROUPED_ROUTE_INVALID_NODE,GROUPED_ROUTE_INVALID_NODE,empty),"invalid slots differ");
    require(!fmm_far_cache_same_source(0,0,empty),"empty peer source accepted");
    require(fmm_far_cache_find_row(1,nullptr,nullptr,0)==-1,"empty target lookup");
}
int main(){
    for(int seed=1;seed<=5;++seed)for(int ranks:{1,2,4})for(int mutation=0;mutation<12;++mutation){
        one_case<2>(seed,ranks,mutation);one_case<3>(seed,ranks,mutation);}
    remap_test<2>();remap_test<3>();
    std::printf("FAR cache replay PASS rows=%llu refs=%llu reused=%llu rebuilt=%llu (including invalid slots)\n",
        (unsigned long long)checked_rows,(unsigned long long)checked_refs,
        (unsigned long long)reused_rows,(unsigned long long)changed_rows);
}
'''

cpp = prelude + arena + types + traits + '\nnamespace body_fitted_fmm {\n' + cache_types + helpers + current + oracle + '\n}\n' + cases
geometry = r'''
#include "src/core/acceleration/body_fitted_fmm/fmm_halo_geometry.h"
#include <cstdio>
#include <stdexcept>
namespace body_fitted_fmm {
struct FMMRemoteLists {
    bool cache_valid=true;int cache_dimension=3;
    std::vector<int> cache_peers{0,0};
    double cache_cutoff=0,cache_length_x=16,cache_length_y=16;
    int cache_layers_x=1,cache_layers_y=1;
    FMMHaloGeometryKey cache_geometry;
};
struct DistributedFMMState {
    int ranks=2;
    struct {double cutoff=0;int depth=5;} gpu_plan;
    cudadem::MortonGrid grid{0,0,0,16,3,5,1},ownership_grid=grid;
    std::vector<uint64_t> cuts{0,grid.end_key()/2,grid.end_key()};
    cudadem::ShearSourceRouting routing{};bool local_only=false;
};
''' + function(plan, 'fmm_far_cache_geometry_matches') + r'''
}
using namespace body_fitted_fmm;
static void check(bool ok){if(!ok)throw std::runtime_error("FAR cache geometry invalidation failed");}
int main(){
    DistributedFMMState s;s.routing.enabled=true;s.routing.layers_x=s.routing.layers_y=1;
    s.routing.length_x=s.routing.length_y=16;s.routing.max_x=s.routing.max_y=16;
    FMMRemoteLists p;p.cache_geometry.capture(s.grid,s.ownership_grid,s.cuts,s.routing,s.local_only,0,s.gpu_plan.depth);
    check(fmm_far_cache_geometry_matches<3>(p,s,0));
    auto changed=s;changed.routing.shift+=.5;changed.grid.generation++;
    check(fmm_far_cache_geometry_matches<3>(p,changed,0)); // live phase and topology generation are not geometry
    changed=s;changed.ranks=3;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.gpu_plan.cutoff=.1;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.gpu_plan.depth++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.layers_x++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.layers_y++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.length_x++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.length_y++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.enabled=false;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.routing.max_x--;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.grid.x++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.grid.size*=2;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.ownership_grid.y++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.cuts[1]++;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    changed=s;changed.local_only=true;check(!fmm_far_cache_geometry_matches<3>(p,changed,0));
    check(!fmm_far_cache_geometry_matches<2>(p,s,0));
    p.cache_valid=false;check(!fmm_far_cache_geometry_matches<3>(p,s,0));p.cache_valid=true;
    s.local_only=true;p.cache_geometry.capture(s.grid,s.ownership_grid,s.cuts,s.routing,true,0,s.gpu_plan.depth);
    check(fmm_far_cache_geometry_matches<3>(p,s,0));
    check(!fmm_far_cache_geometry_matches<3>(p,s,1));
    std::puts("FAR cache geometry/config invalidation PASS");
}
'''
if args.emit:
    args.emit.write_text(cpp)
with tempfile.TemporaryDirectory(prefix='granit-far-cache-') as tmp:
    source, binary = Path(tmp) / 'replay.cpp', Path(tmp) / 'replay'
    source.write_text(cpp)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(SRC), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    source.write_text(geometry)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(args.source / 'src'), '-I', str(args.source),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

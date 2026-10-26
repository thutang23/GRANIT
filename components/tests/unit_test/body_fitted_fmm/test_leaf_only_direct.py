#!/usr/bin/env python3
"""Host replay of local P2P route order against a frozen general walker.

The route replay replaces the source-warp callback to record visitation order.
A separate scalar replay checks the actual source helper's clump filters for
one-source tiles. Neither replay checks GPU warp synchronization or arithmetic.
"""
from pathlib import Path
import argparse
import ast
import hashlib
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


fixture = ast.parse((HERE / 'test_compact_halo_routes.py').read_text())
literals = {n.targets[0].id: ast.literal_eval(n.value) for n in fixture.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('prelude', 'types')}
prelude = literals['prelude'] + '\n#include <limits>\n#include <type_traits>\nDim gridDim{1};\nnamespace fmm {constexpr int FMM_MAX_LEVELS=32;}\n'
types = literals['types'].replace('uint32_t count=0,', 'uint32_t begin=0,count=0,')
types = types.replace('constexpr uint32_t UNIFIED_NODE_FLAG_IS_LEAF=1;',
                      'constexpr uint32_t UNIFIED_NODE_FLAG_IS_LEAF=1; constexpr uint32_t UNIFIED_NODE_FLAG_VALID=2;')
types = types.replace('struct PersistentTreeContext {',
    'struct PersistentTreeMetadata {int num_nodes=0,evaluation_depth=0,active_tree=3,num_unique_cells=0;double cutoff_radius=0,root_cell_size=1;};\nstruct PersistentTreeContext {\nint tree_depth=0;')
types = types.replace('const int* children_compact=nullptr;',
    'const int* children_compact=nullptr; const int *level_list=nullptr,*level_start=nullptr,*level_count=nullptr;')
types = types.replace('struct {int num_nodes=0;double root_cell_size=1;} metadata;', 'PersistentTreeMetadata metadata;')
types = types.replace('struct {double cutoff_radius=0;} command;', 'struct {double cutoff_radius=0;int disable_direct_fallback=1;} command;')
arena = (SRC / 'grouped_interaction_arena.cuh').read_text().replace('#include <cuda_runtime.h>', '').replace('#pragma once', '')
traits = (SRC / 'grouped_stencil_traits.cuh').read_text().replace('#pragma once', '')
traits = re.sub(r'^#include[^\n]*\n', '', traits, flags=re.M)
traits = re.sub(r'\b(GROUPED_D[23]_(?:CROSS|ALL_FAR|MIXED))\b', r'HOST_\1', traits)
p2p = (SRC / 'p2p_unified.cu').read_text()
reference = (HERE / 'general_leaf_routes_reference.inc').read_text()
callback = r'''
using p2p_compute_t=float;
static std::vector<int> source_visits,particle_visits;
static const UnifiedNode* callback_nodes=nullptr;
static const int* callback_order=nullptr;
template<bool OUTER,class... Rest> void grouped_p2p_source_warp(int source,bool,unsigned,int orig_i,Rest&&...){
    source_visits.push_back(source);const auto& n=callback_nodes[source];
    for(uint32_t i=0;i<n.count;++i){int p=callback_order[n.begin+i];if(p!=orig_i)particle_visits.push_back(p);}
}
'''

cases = r'''
using namespace body_fitted_fmm;
static uint64_t checked_leaves=0,checked_sources=0,checked_particles=0;
static void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
static GroupedTargetInteractions blank(){auto r=GroupedTargetInteractions{};
    for(auto& n:r.cross_node)n=GROUPED_ROUTE_INVALID_NODE;
    for(auto& n:r.block_parent)n=GROUPED_ROUTE_INVALID_NODE;r.target_parent=-1;return r;}
template<int D,bool OUTER=false,class Compute=float> void one_case(int seed,bool general){
    using T=GroupedStencilTraits<D>;constexpr int C=T::CHILD_COUNT,depth=4;
    std::mt19937 rng(seed);
    std::map<std::pair<int,uint64_t>,int> canonical;
    std::vector<std::pair<int,uint64_t>> identities;
    for(uint64_t leaf=0;leaf<96;++leaf){if(rng()%5==0)continue;
        for(int level=1;level<=depth;++level)canonical[{level,leaf>>(D*(depth-level))}]=0;}
    for(auto x:canonical)identities.push_back(x.first);
    std::shuffle(identities.begin(),identities.end(),rng);
    std::vector<UnifiedNode> nodes(identities.size());
    for(int i=0;i<int(identities.size());++i){canonical[identities[i]]=i;auto& n=nodes[i];
        n.level=identities[i].first;n.compact_morton=identities[i].second;
        n.flags=UNIFIED_NODE_FLAG_VALID|(n.level==depth?UNIFIED_NODE_FLAG_IS_LEAF:0);
        n.target_count=1;
        if(n.level==depth && n.compact_morton%6==0)n.target_count=0; // image-only destination
        if(n.level==depth && n.compact_morton%11==0)n.flags&=~UNIFIED_NODE_FLAG_VALID;
    }
    std::vector<UnifiedNode> sources(6*(C+1));std::vector<int> children,order;
    for(int g=0;g<6;++g){auto& parent=sources[g*(C+1)];parent.childBase=children.size();
        for(int c=0;c<C;++c)if(rng()%5!=0){int id=g*(C+1)+c+1;parent.childMask|=1u<<c;children.push_back(id);}
        for(int at=0;at<=C;++at){auto& n=sources[g*(C+1)+at];n.begin=order.size();n.count=1+rng()%7;
            for(uint32_t i=0;i<n.count;++i)order.push_back(1000+order.size());}}
    std::vector<GroupedTargetInteractions> routes(nodes.size(),blank());
    for(int t=0;t<int(nodes.size());++t){auto& r=routes[t];auto n=nodes[t];
        if(n.level>=2)r.target_parent=canonical.at({n.level-1,n.compact_morton>>D});
        for(int c=0;c<T::CROSS_COUNT;++c){if(rng()%4==0)continue;r.cross_node[c]=(rng()%6)*(C+1);
            if(general&&rng()%3==0)r.cross_direct_bits|=1u<<c;}
        for(int b=0;b<T::BLOCK_COUNT;++b){if(rng()%4==0)continue;r.block_parent[b]=(rng()%6)*(C+1);
            auto mask=general?uint8_t(rng()):uint8_t(T::block(grouped_target_child<D>(n.compact_morton),b).near_mask);
            r.block_direct_mask[b]=(n.level==depth||general)?mask:0;}
    }
    GroupedInteractionArena arena{};arena.d_targets=routes.data();callback_nodes=sources.data();callback_order=order.data();
    std::vector<int> oldset,levelset;
    for(int i=0;i<int(nodes.size());++i){auto n=nodes[i];
        if((n.flags&UNIFIED_NODE_FLAG_VALID)&&(n.flags&UNIFIED_NODE_FLAG_IS_LEAF)&&n.level>=2&&n.target_count)oldset.push_back(i);
        if(n.level==depth)levelset.push_back(i);}
    std::shuffle(levelset.begin(),levelset.end(),rng);
    std::vector<int> level_list,level_start(depth+1),level_count(depth+1);
    for(int level=0;level<=depth;++level){level_start[level]=level_list.size();
        for(int i=0;i<int(nodes.size());++i)if(nodes[i].level==level)level_list.push_back(i);
        level_count[level]=level_list.size()-level_start[level];
        std::shuffle(level_list.begin()+level_start[level],level_list.end(),rng);}
    PersistentTreeContext ctx{};ctx.tree_depth=depth;ctx.dd.level_list=level_list.data();
    ctx.dd.level_start=level_start.data();ctx.dd.level_count=level_count.data();
    PersistentTreeMetadata metadata{};metadata.num_nodes=nodes.size();metadata.evaluation_depth=depth;
    metadata.active_tree=D;metadata.num_unique_cells=levelset.size();ctx.metadata=metadata;
    ctx.dd.unified_nodes=nodes.data();ctx.arena.d_targets=routes.data();
    require(grouped_p2p_leaf_work_count<false>(&metadata,&ctx,nodes.size())==int(nodes.size()),"general work count changed");
    require(grouped_p2p_leaf_work_count<true>(&metadata,&ctx,nodes.size())==int(levelset.size()),"compact leaf count changed");
    std::vector<int> filtered;
    for(int work=0;work<int(nodes.size());++work)
        require(grouped_p2p_leaf_work_node<false>(work,&metadata,&ctx)==work,"general native traversal changed");
    for(int work=0;work<int(levelset.size());++work){int id=grouped_p2p_leaf_work_node<true>(work,&metadata,&ctx);
        if(grouped_p2p_leaf_eligible(nodes[id]))filtered.push_back(id);}
    auto sorted=filtered;std::sort(sorted.begin(),sorted.end());require(sorted==oldset,"deepest level list lost or duplicated targets");
    for(int leaf:filtered){Compute ax=0,ay=0,az=0;
        auto run=[&](bool old){source_visits.clear();particle_visits.clear();
            if(old)frozen_grouped_p2p_leaf_routes<D,OUTER,Compute>(leaf,nodes.size(),true,1,-1,0,0,0,Compute(1),ax,ay,az,arena,
                nodes.data(),sources.data(),children.data(),nullptr,nullptr,nullptr,nullptr,order.data(),0,nullptr,nullptr,nullptr,nullptr,Compute(1),Compute(0),general?1.:0.);
            else if(general)grouped_p2p_leaf_routes<D,OUTER,Compute,false>(leaf,nodes.size(),true,1,-1,0,0,0,Compute(1),ax,ay,az,arena,
                nodes.data(),sources.data(),children.data(),nullptr,nullptr,nullptr,nullptr,order.data(),0,nullptr,nullptr,nullptr,nullptr,Compute(1),Compute(0),1.);
            else grouped_p2p_leaf_routes<D,OUTER,Compute,true>(leaf,nodes.size(),true,1,-1,0,0,0,Compute(1),ax,ay,az,arena,
                nodes.data(),sources.data(),children.data(),nullptr,nullptr,nullptr,nullptr,order.data(),0,nullptr,nullptr,nullptr,nullptr,Compute(1),Compute(0),0.);
            return std::make_pair(source_visits,particle_visits);};
        auto expected=run(true),actual=run(false);require(expected==actual,"P2P source/particle order changed");
        checked_leaves++;checked_sources+=actual.first.size();checked_particles+=actual.second.size();
    }
    if(!general)for(int t=0;t<int(nodes.size());++t){auto n=nodes[t];auto r=routes[t];
        require(grouped_p2p_leaf_only_routes_valid<D>(n,r),"valid leaf-only stencil rejected");
        if(!(n.flags&UNIFIED_NODE_FLAG_VALID)||!n.target_count||n.level<2)continue;
        r.cross_direct_bits|=1u;require(!grouped_p2p_leaf_only_routes_valid<D>(n,r),"cross direct corruption accepted");
        r=routes[t];if(n.level<depth){r.block_direct_mask[0]|=1u;
            require(!grouped_p2p_leaf_only_routes_valid<D>(n,r),"ancestor direct corruption accepted");}
        else for(int b=0;b<T::BLOCK_COUNT;++b){auto extra=uint8_t(((1u<<C)-1u)&~T::block(grouped_target_child<D>(n.compact_morton),b).near_mask);
            if(extra){r=routes[t];r.block_direct_mask[b]|=extra;
                require(!grouped_p2p_leaf_only_routes_valid<D>(n,r),"leaf far-direct corruption accepted");break;}}
    }
    if(!general){blockIdx.x=threadIdx.x=0;blockDim.x=gridDim.x=1;
        fmm_verify_leaf_only_kernel(&ctx);
        if(seed==1 && !OUTER && std::is_same_v<Compute,float>){auto rejects=[&]{bool trapped=false;try{fmm_verify_leaf_only_kernel(&ctx);}catch(const std::runtime_error&){trapped=true;}
            require(trapped,"runtime invariant oracle accepted corrupted input");};
            int leaf=filtered.front();routes[leaf].cross_direct_bits|=1u;rejects();routes[leaf].cross_direct_bits=0;
            ctx.command.disable_direct_fallback=0;rejects();ctx.command.disable_direct_fallback=1;
            ctx.command.cutoff_radius=.1;rejects();ctx.command.cutoff_radius=0;
            ctx.metadata.cutoff_radius=.1;rejects();ctx.metadata.cutoff_radius=0;
            ctx.metadata.num_unique_cells++;rejects();ctx.metadata.num_unique_cells--;
            int ancestor=routes[leaf].target_parent;nodes[ancestor].flags|=UNIFIED_NODE_FLAG_IS_LEAF;rejects();
            nodes[ancestor].flags&=~UNIFIED_NODE_FLAG_IS_LEAF;
            ctx.metadata.evaluation_depth=1;rejects();ctx.metadata.evaluation_depth=depth;
        }
    }
}
int main(){
    require(grouped_p2p_leaf_only_allowed(1,0.),"zero-cutoff fixed stencil rejected");
    require(grouped_p2p_leaf_only_allowed(1,-1.),"negative-cutoff fixed stencil rejected");
    require(!grouped_p2p_leaf_only_allowed(0,0.),"fallback-enabled specialization accepted");
    require(!grouped_p2p_leaf_only_allowed(1,.1),"finite-cutoff specialization accepted");
    require(!grouped_p2p_leaf_only_allowed(1,std::numeric_limits<double>::quiet_NaN()),"NaN cutoff specialization accepted");
    for(int seed=1;seed<=20;++seed)for(bool general:{false,true}){
        one_case<2,false,float>(seed,general);one_case<3,false,float>(seed,general);
        one_case<2,true,float>(seed,general);one_case<3,true,float>(seed,general);
        one_case<2,false,double>(seed,general);one_case<3,false,double>(seed,general);
        one_case<2,true,double>(seed,general);one_case<3,true,double>(seed,general);}
    std::printf("Local P2P ordered replay PASS leaves=%llu sources=%llu particles=%llu\n",
        (unsigned long long)checked_leaves,(unsigned long long)checked_sources,(unsigned long long)checked_particles);}
'''

current = function(p2p, 'grouped_p2p_leaf_routes')
if 'LeafOnly' not in current:
    raise SystemExit('Stage 3 LeafOnly production specialization is not present yet; harness prepared, not run.')
helper_names = ('grouped_p2p_leaf_only_allowed', 'grouped_p2p_leaf_eligible', 'grouped_p2p_leaf_work_count',
                'grouped_p2p_leaf_work_node', 'grouped_p2p_leaf_only_routes_valid', 'fmm_verify_leaf_only_kernel')
helpers = '\n'.join(function(p2p, name) for name in helper_names)
source_helper = function(p2p, 'grouped_p2p_source_warp')
# Frozen main's arithmetic and visitation order remain byte-for-byte pinned.
# Normalize ONLY the explicit local and remote clump-filter additions; requiring
# each exact fragment once prevents this exception from hiding other changes.
normalized_helper = source_helper
clump_additions = (
    ('''                const int target_clump = __shfl_sync(0xffffffffu, remote_target_clump_id, target_lane);
''', ''),
    ('''Compute G, Compute softening_sq, double cutoff_sq,
    const int* gravity_clump_ids = nullptr,
    int remote_target_clump_id = 0, const int* remote_source_clump_ids = nullptr,
    double clump_exclusion_distance_sq = 0.0)''', 'Compute G, Compute softening_sq, double cutoff_sq)'),
    ('''                    const bool same_clump =
                        (gravity_clump_ids && gravity_clump_ids[target_orig] > 0 &&
                         gravity_clump_ids[target_orig] == gravity_clump_ids[orig_j]) ||
                        (target_clump > 0 && remote_source_clump_ids &&
                         target_clump == remote_source_clump_ids[orig_j]);
''', ''),
    ('''                        const double distance_sq = rdx*rdx+rdy*rdy+rdz*rdz;
                        // In a shearing lattice only the geometrically adjacent
                        // copy is internal. Other replicas of this clump remain
                        // physical sources. The host checks D<h and 2D<min(Lx,Ly).
                        contributes = !(same_clump && (clump_exclusion_distance_sq <= 0.0 ||
                            distance_sq < clump_exclusion_distance_sq)) &&
                            !(cutoff_sq > 0.0 && distance_sq > cutoff_sq);''', '''                        contributes = !(cutoff_sq > 0.0 &&
                            rdx * rdx + rdy * rdy + rdz * rdz > cutoff_sq);'''),
    ('''                const bool same_clump =
                    (gravity_clump_ids && gravity_clump_ids[orig_i] > 0 &&
                     gravity_clump_ids[orig_i] == gravity_clump_ids[orig_j]) ||
                    (remote_target_clump_id > 0 && remote_source_clump_ids &&
                     remote_target_clump_id == remote_source_clump_ids[orig_j]);
''', ''),
    ('''                const double distance_sq = rdx*rdx+rdy*rdy+rdz*rdz;
                if ((same_clump && (clump_exclusion_distance_sq <= 0.0 ||
                     distance_sq < clump_exclusion_distance_sq)) ||
                    (cutoff_sq > 0.0 && distance_sq > cutoff_sq)) continue;''', '''                if (cutoff_sq > 0.0 &&
                    rdx * rdx + rdy * rdy + rdz * rdz > cutoff_sq)
                    continue;'''),
)
for addition, original in clump_additions:
    assert normalized_helper.count(addition) == 1, 'Documented clump-filter addition changed'
    normalized_helper = normalized_helper.replace(addition, original)
assert hashlib.sha256(normalized_helper.encode()).hexdigest() == \
    '879bd6434401a6fd755c26bc180f6093a43b3c8b3cd93b26193abdc6701f4092', 'Particle arithmetic/source callback changed from frozen baseline'
cpp = prelude + arena + types + traits + '\nnamespace body_fitted_fmm {\n' + callback + helpers + current + reference + '\n}\n' + cases

# Execute production local and remote filter expressions in both occupancy paths.
# One source in lane zero makes scalar shuffle/ballot shims sufficient; the
# force primitive is a marker, so this verifies filtering and mass placement,
# while the frozen hash above still pins the real arithmetic callback/order.
source_shims = r'''
using namespace body_fitted_fmm;
using std::min;
inline int __popc(unsigned value){return __builtin_popcount(value);}
inline void __syncwarp(){}
template<class T> T __shfl_sync(unsigned,T value,int){return value;}
inline unsigned __ballot_sync(unsigned,bool value){return value?1u:0u;}
using p2p_compute_t=float;
static int primitive_calls=0;
template<class T> void compute_gravitational_acceleration_from_delta_unified(
    T,T,T,T,T,T,T& x,T& y,T& z){++primitive_calls;x=T(2);y=T(3);z=T(5);}
'''
source_cases = r'''
template<bool OUTER,class Compute> void check_source_filters(){
    UnifiedNode node{};node.begin=0;node.count=1;
    int order[1]={7},ids[8]{};double pos[8]{},sx[32]{},sy[32]{},sz[32]{};
    float mass[8]{};mass[7]=1;Compute sm[32]{};
    for(unsigned mask:{1u,31u})for(int scenario=0;scenario<6;++scenario){
        // IDs are canonical (target 2, source 7), unrelated to tile lane 0.
        std::fill(ids,ids+8,0);ids[2]=ids[7]=11;order[0]=7;
        const int* filter=ids;bool included=true;
        if(scenario==0)filter=nullptr;                 // explicit null API
        if(scenario==1)included=false;                // same positive clump
        if(scenario==2)ids[7]=12;                     // different clumps
        if(scenario==3)ids[2]=ids[7]=0;               // ordinary zero IDs
        if(scenario==4){order[0]=2;filter=nullptr;included=false;}    // exact self, no filter
        if(scenario==5)filter=nullptr;                // omitted default API
        Compute x=0,y=0,z=0;primitive_calls=0;
        if(scenario==5)grouped_p2p_source_warp<OUTER,Compute>(0,true,mask,2,
            0.,0.,0.,Compute(2),x,y,z,&node,pos,pos,pos,mass,order,0,
            sx,sy,sz,sm,Compute(1),Compute(0),0.);
        else grouped_p2p_source_warp<OUTER,Compute>(0,true,mask,2,
            0.,0.,0.,Compute(2),x,y,z,&node,pos,pos,pos,mass,order,0,
            sx,sy,sz,sm,Compute(1),Compute(0),0.,filter);
        const Compute multiplier=included?Compute(OUTER?1:2):Compute(0);
        if(primitive_calls!=int(included)||x!=2*multiplier||y!=3*multiplier||z!=5*multiplier)
            throw std::runtime_error("production source helper clump/null/self filter regression");
    }
}
template<bool OUTER,class Compute> void check_remote_source_filters(){
    UnifiedNode node{};node.begin=0;node.count=1;
    int order[1]={7},source_ids[8]{};double pos[8]{},sx[32]{},sy[32]{},sz[32]{};
    float mass[8]{};mass[7]=1;Compute sm[32]{};
    for(unsigned mask:{1u,31u})for(int scenario=0;scenario<6;++scenario){
        int target_clump=11;source_ids[7]=11;const int* source_filter=source_ids;
        bool included=true;
        if(scenario==0)source_filter=nullptr;
        if(scenario==1)included=false;
        if(scenario==2)source_ids[7]=12;
        if(scenario==3)target_clump=source_ids[7]=0;
        if(scenario==4)target_clump=0;
        if(scenario==5)source_ids[7]=0;
        Compute x=0,y=0,z=0;primitive_calls=0;
        // Remote target indices deliberately have no meaning in the source arrays.
        grouped_p2p_source_warp<OUTER,Compute>(0,true,mask,-1,
            0.,0.,0.,Compute(2),x,y,z,&node,pos,pos,pos,mass,order,0,
            sx,sy,sz,sm,Compute(1),Compute(0),0.,nullptr,target_clump,source_filter);
        const Compute multiplier=included?Compute(OUTER?1:2):Compute(0);
        if(primitive_calls!=int(included)||x!=2*multiplier||y!=3*multiplier||z!=5*multiplier)
            throw std::runtime_error("production remote clump/null filter or force units regression");
    }
}

template<bool OUTER,class Compute> void check_shear_source_filters(){
    UnifiedNode node{};node.begin=0;node.count=1;
    int order[1]={7},ids[8]{};ids[2]=ids[7]=11;
    double px[8]{},zero[8]{},sx[32]{},sy[32]{},sz[32]{};
    float mass[8]{};mass[7]=1;Compute sm[32]{};
    for(unsigned mask:{1u,31u})for(bool remote:{false,true})
    for(double separation:{0.,0.05,0.1,1.}) {
        px[7]=separation;Compute x=0,y=0,z=0;primitive_calls=0;
        grouped_p2p_source_warp<OUTER,Compute>(0,true,mask,remote?-1:2,
            0.,0.,0.,Compute(2),x,y,z,&node,px,zero,zero,mass,order,0,
            sx,sy,sz,sm,Compute(1),Compute(0),0.,remote?nullptr:ids,
            remote?11:0,remote?ids:nullptr,0.01);
        // D=0.1; only the close copy is internal. The exact boundary and
        // distant replica are retained (the full body envelope has a margin).
        const bool included=separation>=0.1;
        const Compute multiplier=included?Compute(OUTER?1:2):Compute(0);
        if(primitive_calls!=int(included)||x!=2*multiplier||y!=3*multiplier||z!=5*multiplier)
            throw std::runtime_error("shear clump image identity regression");
    }
}
int main(){check_source_filters<false,float>();check_source_filters<true,float>();
    check_source_filters<false,double>();check_source_filters<true,double>();
    check_remote_source_filters<false,float>();check_remote_source_filters<true,float>();
    check_remote_source_filters<false,double>();check_remote_source_filters<true,double>();
    check_shear_source_filters<false,float>();check_shear_source_filters<true,float>();
    check_shear_source_filters<false,double>();check_shear_source_filters<true,double>();
    std::puts("Production P2P source filters PASS cases=160 (local/remote, sparse/dense, force/acceleration, float/double)");}
'''
source_cpp = prelude + arena + types + source_shims + source_helper + source_cases
if args.emit:
    args.emit.write_text(cpp)
with tempfile.TemporaryDirectory(prefix='granit-leaf-only-') as tmp:
    source, binary = Path(tmp) / 'replay.cpp', Path(tmp) / 'replay'
    source.write_text(cpp)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(SRC), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    source.write_text(source_cpp)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(SRC), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

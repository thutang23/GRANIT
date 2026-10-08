#!/usr/bin/env python3
"""Compile the actual optimized predicate as host C++ and compare slot oracle.

No GPU or solver build is required. The only source transformation changes the
device qualifier and names the fixture's host child/stencil accessors.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
default_source = next((root for parent in (HERE, *HERE.parents)
                       for root in (parent, parent / 'source')
                       if (root / 'src/core/acceleration/body_fitted_fmm/remote_interaction_plan.cuh').is_file()), None)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=default_source,
                    help='GRANIT source root to verify')
args = parser.parse_args()
if args.source is None:
    parser.error('no source tree found; pass --source /path/to/GRANIT')
SRC = args.source / "src/core/acceleration/body_fitted_fmm"
source = (SRC / "remote_interaction_plan.cuh").read_text()
start = source.index("template<int D>\n__device__ bool fmm_remote_has_far(")
open_brace = source.index("{", start)
depth = 0
for end in range(open_brace, len(source)):
    depth += (source[end] == "{") - (source[end] == "}")
    if depth == 0:
        break
predicate = source[start:end + 1].replace("__device__ bool", "bool")
predicate = predicate.replace("GroupedStencilTraits", "FixtureTraits")
predicate = predicate.replace("grouped_child_node", "fixture_child_node")

fixture = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "grouped_stencil_tables.cuh"
using namespace body_fitted_fmm;

struct UnifiedNode { uint64_t compact_morton=0; uint8_t childMask=0; int childBase=0; };
struct GroupedSourceRoute { uint32_t node=UINT32_MAX; uint8_t direct_mask=0; };
struct GroupedTargetInteractions {
    GroupedSourceRoute cross[16],block[17]; uint8_t omit[17]{};
};
struct FMMRemoteView {
    const UnifiedNode* nodes=nullptr; const int* children=nullptr;
    const GroupedTargetInteractions* far_routes=nullptr; int count=0;
};
// This predicate fixture retains the supported dense-view convention.
// Compact mappings and absent rows are covered by test_compact_halo_routes.py.
inline int fmm_remote_route_row(const FMMRemoteView&,int target){return target;}
inline GroupedSourceRoute grouped_cross_route(const GroupedTargetInteractions& x,int i){return x.cross[i];}
inline GroupedSourceRoute grouped_block_route(const GroupedTargetInteractions& x,int i){return x.block[i];}
inline bool grouped_route_valid(GroupedSourceRoute x){return x.node!=UINT32_MAX;}
inline int grouped_route_node(GroupedSourceRoute x){return int(x.node);}
inline uint8_t grouped_route_mask(GroupedSourceRoute x){return x.direct_mask;}
inline uint8_t grouped_block_omit_mask(const GroupedTargetInteractions& x,int i){return x.omit[i];}
inline int __ffs(int x){return x ? __builtin_ffs(x) : 0;}
template<int D> struct FixtureTraits;
template<> struct FixtureTraits<2> {
    static constexpr int CROSS_COUNT=GROUPED_D2_CROSS_COUNT;
    static constexpr int CHILD_COUNT=GROUPED_D2_CHILD_COUNT;
    static constexpr int BLOCK_COUNT=GROUPED_D2_BLOCK_COUNT;
    static auto block(int q,int b) {return b<GROUPED_D2_ALL_FAR_COUNT ?
        HOST_GROUPED_D2_ALL_FAR[q][b] : HOST_GROUPED_D2_MIXED[q][b-GROUPED_D2_ALL_FAR_COUNT];}
};
template<> struct FixtureTraits<3> {
    static constexpr int CROSS_COUNT=GROUPED_D3_CROSS_COUNT;
    static constexpr int CHILD_COUNT=GROUPED_D3_CHILD_COUNT;
    static constexpr int BLOCK_COUNT=GROUPED_D3_BLOCK_COUNT;
    static auto block(int q,int b) {return b<GROUPED_D3_ALL_FAR_COUNT ?
        HOST_GROUPED_D3_ALL_FAR[q][b] : HOST_GROUPED_D3_MIXED[q][b-GROUPED_D3_ALL_FAR_COUNT];}
};
template<int D> int grouped_target_child(uint64_t morton) {
    int slot=0;for(int axis=0;axis<D;++axis)
        slot|=((morton>>(D-1-axis))&1u)<<axis;
    return slot;
}
template<int D> int fixture_child_node(const UnifiedNode* nodes,const int* children,int parent,int generated) {
    int slot=0;for(int axis=0;axis<D;++axis)
        slot|=((generated>>(D-1-axis))&1u)<<axis;
    const auto& p=nodes[parent];
    if(!(p.childMask&(1u<<slot)))return -1;
    return children[p.childBase+__builtin_popcount(unsigned(p.childMask)&((1u<<slot)-1u))];
}
'''
reference = r'''
template<int D> bool old_slot_oracle(int target,const UnifiedNode& node,
    const FMMRemoteView* peers,int ranks) {
    using T=FixtureTraits<D>;
    const int q=grouped_target_child<D>(node.compact_morton);
    for(int slot=0;slot<T::CROSS_COUNT+T::BLOCK_COUNT*T::CHILD_COUNT;++slot) {
        for(int peer=0;peer<ranks;++peer) {
            const auto& view=peers[peer];if(!view.count)continue;
            const auto& work=view.far_routes[target];int source=-1;
            if(slot<T::CROSS_COUNT) {
                const auto route=grouped_cross_route(work,slot);
                if(grouped_route_valid(route) && !(grouped_route_mask(route)&1u))
                    source=grouped_route_node(route);
            } else {
                int at=slot-T::CROSS_COUNT,b=at/T::CHILD_COUNT,c=at%T::CHILD_COUNT;
                const auto route=grouped_block_route(work,b);const uint8_t bit=uint8_t(1u<<c);
                if(grouped_route_valid(route) && (T::block(q,b).far_mask&bit) &&
                    !(grouped_route_mask(route)&bit) && !(grouped_block_omit_mask(work,b)&bit))
                    source=fixture_child_node<D>(view.nodes,view.children,grouped_route_node(route),c);
            }
            if(source>=0)return true;
        }
    }
    return false;
}
template<int D> void run() {
    using T=FixtureTraits<D>;
    std::mt19937 rng(982421+D);
    UnifiedNode target{};target.compact_morton=0;
    std::vector<UnifiedNode> nodes(2);
    std::vector<int> children(T::CHILD_COUNT);
    for(int i=0;i<T::CHILD_COUNT;++i)children[i]=1;
    GroupedTargetInteractions work[4]{};
    FMMRemoteView views[4]{};
    for(int r=0;r<4;++r)views[r]={nodes.data(),children.data(),&work[r],1};
    int cases=0;
    auto check=[&](int ranks) {
        bool got=fmm_remote_has_far<D>(0,target,views,ranks);
        bool expected=old_slot_oracle<D>(0,target,views,ranks);
        if(got!=expected) {std::fprintf(stderr,"D%d case=%d ranks=%d got=%d want=%d\n",
            D,cases,ranks,got,expected);std::abort();}
        ++cases;
    };
    // Empty peers and invalid routes.
    for(int ranks:{1,2,4}) {for(int r=0;r<4;++r)views[r].count=0;check(ranks);}
    for(int q=0;q<T::CHILD_COUNT;++q) {
        target.compact_morton=q;
        for(int iteration=0;iteration<300;++iteration) {
            for(int r=0;r<4;++r) {
                views[r].count=(rng()%5!=0);
                for(int c=0;c<T::CROSS_COUNT;++c)
                    work[r].cross[c]={rng()%4 ? 0u : UINT32_MAX,uint8_t(rng()%2)};
                for(int b=0;b<T::BLOCK_COUNT;++b) {
                    work[r].block[b]={rng()%5 ? 0u : UINT32_MAX,uint8_t(rng()%256)};
                    work[r].omit[b]=uint8_t(rng()%256);
                }
            }
            nodes[0].childMask=uint8_t(rng()&((1u<<T::CHILD_COUNT)-1u));
            for(int ranks:{1,2,4})check(ranks);
        }
    }
    // Explicit cross-only, all-near, fully omitted, and sparse-child cases.
    for(auto& x:work)for(auto& c:x.cross)c={UINT32_MAX,0};
    for(auto& x:work)for(auto& b:x.block)b={UINT32_MAX,0};
    for(auto& x:work)for(auto& mask:x.omit)mask=0;
    for(auto& v:views)v.count=1;
    target.compact_morton=0;nodes[0].childMask=0;
    work[3].cross[0]={0,0};check(4);
    work[3].cross[0]={0,1};check(4);
    int far_block=-1,far_child=-1;
    for(int b=0;b<T::BLOCK_COUNT && far_block<0;++b)
        for(int c=0;c<T::CHILD_COUNT;++c)
            if(T::block(0,b).far_mask&(1u<<c)){far_block=b;far_child=c;break;}
    assert(far_block>=0);
    work[3].block[far_block]={0,0};check(4); // no child exists
    int slot=0;for(int axis=0;axis<D;++axis)
        slot|=((far_child>>(D-1-axis))&1u)<<axis;
    nodes[0].childMask=uint8_t(1u<<slot);check(4);
    work[3].omit[far_block]=uint8_t(1u<<far_child);check(4);
    work[3].omit[far_block]=0;work[3].block[far_block].direct_mask=uint8_t(1u<<far_child);check(4);
    // Force every block-mask path to execute without a cross-route early exit.
    // The truth value is independently known for exactly one present child.
    for(int q=0;q<T::CHILD_COUNT;++q) {
        target.compact_morton=q;
        for(int b=0;b<T::BLOCK_COUNT;++b)for(int c=0;c<T::CHILD_COUNT;++c) {
            for(auto& x:work) {
                for(auto& route:x.cross)route={UINT32_MAX,0};
                for(auto& route:x.block)route={UINT32_MAX,0};
                for(auto& mask:x.omit)mask=0;
            }
            int child_slot=0;
            for(int axis=0;axis<D;++axis)
                child_slot|=((c>>(D-1-axis))&1u)<<axis;
            for(int present=0;present<2;++present)
                for(int direct=0;direct<2;++direct)
                    for(int omitted=0;omitted<2;++omitted) {
                        nodes[0].childMask=present ? uint8_t(1u<<child_slot) : 0;
                        work[3].block[b]={0,uint8_t(direct ? (1u<<c) : 0)};
                        work[3].omit[b]=uint8_t(omitted ? (1u<<c) : 0);
                        const int mapped_q=grouped_target_child<D>(target.compact_morton);
                        bool want=present && !direct && !omitted &&
                                  (T::block(mapped_q,b).far_mask&(1u<<c));
                        assert(fmm_remote_has_far<D>(0,target,views,4)==want);
                        check(4);
                    }
        }
    }
    std::printf("D%d: %d host predicate/oracle cases passed\n",D,cases);
}
int main(){run<2>();run<3>();}
'''

with tempfile.TemporaryDirectory(prefix="granit-far-predicate-") as tmp:
    test_cpp = Path(tmp) / "predicate.cpp"
    executable = Path(tmp) / "predicate"
    test_cpp.write_text(fixture + "\n" + predicate + "\n" + reference)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(SRC), str(test_cpp),
                    "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)

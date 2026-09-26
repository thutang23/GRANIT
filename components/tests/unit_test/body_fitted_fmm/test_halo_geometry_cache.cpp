#include "src/core/acceleration/body_fitted_fmm/fmm_halo_geometry.h"

#include <iostream>
#include <stdexcept>

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error("halo geometry cache check failed: " #condition); \
} while (false)

using namespace body_fitted_fmm;

static bool same_boxes(const std::vector<FMMHaloBox>& a,const std::vector<FMMHaloBox>& b) {
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i) {
        if(a[i].level!=b[i].level || a[i].owners!=b[i].owners)return false;
        for(int axis=0;axis<3;++axis)
            if(a[i].lo[axis]!=b[i].lo[axis] || a[i].hi[axis]!=b[i].hi[axis])return false;
    }
    return true;
}

int main() {
    cudadem::MortonGrid gravity{0,0,0,16,2,5,1};
    cudadem::MortonGrid ownership=gravity;
    const uint64_t end=ownership.end_key();
    std::vector<uint64_t> cuts{0,end/2,end};
    cudadem::ShearSourceRouting routing{};
    routing.min_x=0;routing.max_x=16;routing.min_y=0;routing.max_y=16;
    FMMHaloGeometryKey key;
    CHECK(!key.matches(gravity,ownership,cuts,routing,false,0,5));
    const auto boxes=fmm_make_halo_geometry(gravity,ownership,cuts,routing,false,0,5);
    CHECK(!boxes.empty());
    key.capture(gravity,ownership,cuts,routing,false,0,5);
    CHECK(key.matches(gravity,ownership,cuts,routing,false,0,5));
    CHECK(same_boxes(boxes,fmm_make_halo_geometry(gravity,ownership,cuts,routing,false,0,5)));

    // Occupied-node topology and grid generation are absent from the box math.
    gravity.generation++;ownership.generation++;
    CHECK(key.matches(gravity,ownership,cuts,routing,false,0,5));
    CHECK(same_boxes(boxes,fmm_make_halo_geometry(gravity,ownership,cuts,routing,false,0,5)));
    CHECK(!key.matches(gravity,ownership,cuts,routing,false,0,6));
    auto moved=gravity;moved.x+=0.25;
    CHECK(!key.matches(moved,ownership,cuts,routing,false,0,5));
    moved=gravity;moved.size*=2;
    CHECK(!key.matches(moved,ownership,cuts,routing,false,0,5));
    moved=gravity;moved.dimension=3;
    CHECK(!key.matches(moved,ownership,cuts,routing,false,0,5));
    auto moved_owner=ownership;moved_owner.y+=0.25;
    CHECK(!key.matches(gravity,moved_owner,cuts,routing,false,0,5));
    moved_owner=ownership;moved_owner.size*=2;
    CHECK(!key.matches(gravity,moved_owner,cuts,routing,false,0,5));
    auto changed_cuts=cuts;changed_cuts[1]+=1;
    CHECK(!key.matches(gravity,ownership,changed_cuts,routing,false,0,5));
    const std::vector<uint64_t> empty_owner_cuts{0,end/2,end/2,end};
    CHECK(!key.matches(gravity,ownership,empty_owner_cuts,routing,false,0,5));
    CHECK(!fmm_make_halo_geometry(gravity,ownership,empty_owner_cuts,routing,false,0,5).empty());
    auto clipped=routing;clipped.enabled=true;
    CHECK(!key.matches(gravity,ownership,cuts,clipped,false,0,5));
    clipped.enabled=false;clipped.max_x=15;
    CHECK(key.matches(gravity,ownership,cuts,clipped,false,0,5));
    clipped.enabled=true;
    key.capture(gravity,ownership,cuts,clipped,false,0,5);
    CHECK(key.matches(gravity,ownership,cuts,clipped,false,0,5));
    const auto clipped_boxes=fmm_make_halo_geometry(gravity,ownership,cuts,clipped,false,0,5);
    clipped.shift+=0.125;
    CHECK(key.matches(gravity,ownership,cuts,clipped,false,0,5));
    CHECK(same_boxes(clipped_boxes,fmm_make_halo_geometry(gravity,ownership,cuts,clipped,false,0,5)));
    clipped.max_x=14;
    CHECK(!key.matches(gravity,ownership,cuts,clipped,false,0,5));

    // The local-image cover uses its rank and physical bounds, not the cuts.
    key.capture(gravity,ownership,cuts,routing,true,0,5);
    CHECK(key.matches(gravity,moved_owner,changed_cuts,routing,true,0,5));
    CHECK(!key.matches(gravity,ownership,cuts,routing,true,1,5));
    CHECK(!key.matches(gravity,ownership,cuts,clipped,true,0,5));
    const auto local=fmm_make_halo_geometry(gravity,ownership,cuts,routing,true,0,5);
    CHECK(!local.empty());
    CHECK(same_boxes(local,fmm_make_halo_geometry(gravity,moved_owner,changed_cuts,routing,true,0,5)));

    auto gravity3=gravity;gravity3.dimension=3;gravity3.z=-8;
    auto ownership3=gravity3;
    const std::vector<uint64_t> cuts3{0,ownership3.end_key()/2,ownership3.end_key()};
    key.capture(gravity3,ownership3,cuts3,routing,false,0,4);
    const auto boxes3=fmm_make_halo_geometry(gravity3,ownership3,cuts3,routing,false,0,4);
    CHECK(!boxes3.empty());
    CHECK(key.matches(gravity3,ownership3,cuts3,routing,false,0,4));
    gravity3.z+=0.5;
    CHECK(!key.matches(gravity3,ownership3,cuts3,routing,false,0,4));
    std::cout<<"FMM halo geometry cache PASS\n";
}

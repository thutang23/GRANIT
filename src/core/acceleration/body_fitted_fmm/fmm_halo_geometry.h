#pragma once

#include "domain/morton_partition.h"
#include "grouped_stencil_tables.cuh"
#include <array>
#include <set>

namespace body_fitted_fmm {

// Half-open integer parent-cell boxes. Geometry covers ownership, not occupied
// Karras nodes. Payload selection subsequently visits only occupied sources.
struct FMMHaloBox {
    int lo[3], hi[3];
    int level;
    uint32_t owners;
};

inline std::vector<std::array<int,3>> fmm_halo_parent_offsets(int dimension) {
    std::set<std::array<int,3>> offsets;
    auto add=[&](const auto& entry) {
        offsets.insert({entry.parent[0],entry.parent[1],entry.parent[2]});
    };
    if (dimension==2) {
        for (int q=0;q<GROUPED_D2_CHILD_COUNT;++q) {
            for (const auto& e:HOST_GROUPED_D2_CROSS[q]) add(e);
            for (const auto& e:HOST_GROUPED_D2_MIXED[q]) add(e);
        }
    } else {
        for (int q=0;q<GROUPED_D3_CHILD_COUNT;++q) {
            for (const auto& e:HOST_GROUPED_D3_CROSS[q]) add(e);
            for (const auto& e:HOST_GROUPED_D3_ALL_FAR[q]) add(e);
            for (const auto& e:HOST_GROUPED_D3_MIXED[q]) add(e);
        }
    }
    return {offsets.begin(),offsets.end()};
}

inline std::vector<FMMHaloBox> fmm_halo_boxes(
    const cudadem::MortonGrid& gravity, int depth,
    const std::vector<cudadem::MortonCoverCell>& cover)
{
    if ((gravity.dimension!=2 && gravity.dimension!=3) || depth<2 || depth>20)
        throw std::invalid_argument("invalid FMM halo grid/depth");
    const auto offsets=fmm_halo_parent_offsets(gravity.dimension);
    std::vector<FMMHaloBox> boxes;
    const double origin[3]={gravity.x,gravity.y,gravity.z};
    for (const auto& cell:cover) {
        if (cell.owner<0 || cell.owner>=32)
            throw std::invalid_argument("FMM halo supports at most 32 owners");
        for (int level=1;level<depth;++level) {
            const int side=1<<level;
            const double scale=std::ldexp(1.0,level)/gravity.size;
            int lo[3]{},hi[3]{1,1,1};
            bool empty=false;
            for (int a=0;a<gravity.dimension;++a) {
                // Outward rounding includes cells touching an ownership face.
                // This is bounded overfetch; exact inverse routes decide use.
                const double low=std::isfinite(cell.lo[a])
                    ? std::nextafter((cell.lo[a]-origin[a])*scale,-INFINITY):0;
                const double high=std::isfinite(cell.hi[a])
                    ? std::nextafter((cell.hi[a]-origin[a])*scale,INFINITY):side;
                lo[a]=int(std::max(0.0,std::min(double(side),std::floor(low))));
                hi[a]=int(std::max(0.0,std::min(double(side),std::ceil(high))));
                empty|=lo[a]>=hi[a];
            }
            if (empty) continue;
            for (const auto& delta:offsets) {
                FMMHaloBox b{{},{},level,uint32_t{1}<<cell.owner};
                for (int a=0;a<3;++a) {
                    b.lo[a]=std::max(0,lo[a]+delta[a]);
                    b.hi[a]=std::min(a<gravity.dimension?side:1,hi[a]+delta[a]);
                }
                if (b.lo[0]<b.hi[0] && b.lo[1]<b.hi[1] && b.lo[2]<b.hi[2])
                    boxes.push_back(b);
            }
        }
    }
    // Merge adjacent/overlapping intervals with identical other axes. Sorting
    // avoids the quadratic global pair search for fragmented Morton covers.
    size_t previous=0;
    do {
        previous=boxes.size();
        for(int axis=0;axis<gravity.dimension;++axis) {
            const auto less=[axis](const FMMHaloBox& a,const FMMHaloBox& b) {
                if(a.level!=b.level)return a.level<b.level;
                if(a.owners!=b.owners)return a.owners<b.owners;
                for(int k=0;k<3;++k)if(k!=axis) {
                    if(a.lo[k]!=b.lo[k])return a.lo[k]<b.lo[k];
                    if(a.hi[k]!=b.hi[k])return a.hi[k]<b.hi[k];
                }
                return a.lo[axis]!=b.lo[axis]?a.lo[axis]<b.lo[axis]:a.hi[axis]<b.hi[axis];
            };
            std::sort(boxes.begin(),boxes.end(),less);
            size_t out=0;
            for(const auto b:boxes) {
                bool merge=out>0;
                if(merge) {
                    const auto& a=boxes[out-1];
                    merge=a.level==b.level && a.owners==b.owners && b.lo[axis]<=a.hi[axis];
                    for(int k=0;k<3;++k)if(k!=axis)merge&=a.lo[k]==b.lo[k] && a.hi[k]==b.hi[k];
                }
                if(merge)boxes[out-1].hi[axis]=std::max(boxes[out-1].hi[axis],b.hi[axis]);
                else boxes[out++]=b;
            }
            boxes.resize(out);
        }
    } while(boxes.size()<previous);
    return boxes;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline uint32_t fmm_halo_owners(const FMMHaloBox* boxes,int count,
    int level,uint32_t x,uint32_t y,uint32_t z) {
    uint32_t owners=0;
    int lo=0,hi=count;
    while(lo<hi) {const int mid=lo+(hi-lo)/2;if(boxes[mid].level<level-1)lo=mid+1;else hi=mid;}
    for(int i=lo;i<count && boxes[i].level<=level;++i) {
        const auto& b=boxes[i];
        const int shift=b.level==level?0:1;
        const int p[3]={int(x>>shift),int(y>>shift),int(z>>shift)};
        if(p[0]>=b.lo[0] && p[0]<b.hi[0] && p[1]>=b.lo[1] && p[1]<b.hi[1] &&
           p[2]>=b.lo[2] && p[2]<b.hi[2])owners|=b.owners;
    }
    return owners;
}

} // namespace body_fitted_fmm

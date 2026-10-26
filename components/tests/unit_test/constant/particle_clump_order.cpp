#include "src/core/constant/particle_clump_order.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error("particle order check failed: " #condition); \
} while (false)

// The production implementation before the startup scalability correction.
// Keep its independently expressed grouping loops for exact differential tests.
static std::vector<int32_t> previous_order(
    const std::vector<int32_t>& ids, const std::vector<uint8_t>& life, bool preserve) {
    int32_t maximum=0;
    for (int32_t id:ids) {
        if (id < -1) throw std::runtime_error("invalid clump_id");
        maximum=std::max(maximum,id);
    }
    for (uint8_t state:life)
        if (state > 1) throw std::runtime_error("invalid lifecycle_state");
    std::vector<int32_t> result;
    if (preserve) {
        for (size_t row=0;row<ids.size();++row)
            if (ids[row]>=0 && life[row]==0) result.push_back(row);
    } else {
        for (size_t row=0;row<ids.size();++row)
            if (ids[row]==0 && life[row]==0) result.push_back(row);
        for (int32_t cid=1;cid<=maximum;++cid)
            for (size_t row=0;row<ids.size();++row)
                if (ids[row]==cid && life[row]==0) result.push_back(row);
    }
    for (size_t row=0;row<ids.size();++row)
        if (life[row]!=0 || ids[row]==-1) result.push_back(row);
    return result;
}

static void differential(const std::vector<int32_t>& ids,const std::vector<uint8_t>& life) {
    for (bool preserve:{false,true}) {
        std::string old_error,new_error;
        std::vector<int32_t> expected,actual;
        try { expected=previous_order(ids,life,preserve); }
        catch (const std::runtime_error& error) { old_error=error.what(); }
        try { actual=cudadem::particle_clump_order(ids,life,preserve); }
        catch (const std::runtime_error& error) { new_error=error.what(); }
        CHECK(old_error==new_error);
        CHECK(expected==actual);
    }
}

int main() {
    differential({},{});
    differential({0},{0});
    differential({3,0,2,3,-1,0,1,2,0,4},{0,0,0,0,0,1,0,1,0,1});
    differential({-1,4,0,-1,3},{0,1,1,1,1});
    differential({0,0,0,0},{0,1,0,1});
    differential({7,7,1,1,7,1},{0,0,0,0,0,0});
    differential({-2,0,3},{1,0,0});
    differential({-1,0,3},{0,2,0});
    // Invalid CID takes precedence over lifecycle errors, as before.
    differential({-1,-2,3},{0,1,2});

    std::mt19937 random(0xC1A09u);
    for (int sample=0;sample<3000;++sample) {
        const size_t count=random()%257;
        std::vector<int32_t> ids(count);
        std::vector<uint8_t> life(count);
        for (size_t row=0;row<count;++row) {
            ids[row]=static_cast<int32_t>(random()%82)-1;
            life[row]=random()%2;
        }
        differential(ids,life);
    }
    // Sparse extreme IDs must neither allocate max(CID) entries nor iterate
    // over every absent CID. Stability within equal IDs and the dead tail is
    // specified directly because the old quadratic reference cannot run here.
    const int32_t large=std::numeric_limits<int32_t>::max();
    const std::vector<int32_t> sparse{large,0,5,large,-1,5,large,0};
    const std::vector<uint8_t> life{0,0,0,0,0,0,1,1};
    CHECK(cudadem::particle_clump_order(sparse,life,false)==
          (std::vector<int32_t>{1,2,5,0,3,4,6,7}));
    CHECK(cudadem::particle_clump_order(sparse,life,true)==
          (std::vector<int32_t>{0,1,2,3,5,4,6,7}));

    // The actual million-member / 250k-body startup size that exposed the
    // old 250-billion-comparison loop. Reverse the body groups to exercise
    // sorting; each group's four original member rows must remain stable.
    constexpr int32_t count=1000000;
    std::vector<int32_t> ids(count);
    std::vector<uint8_t> live(count,0);
    for (int32_t row=0;row<count;++row) ids[row]=count/4-row/4;
    const auto begin=std::chrono::steady_clock::now();
    const auto order=cudadem::particle_clump_order(ids,live,false);
    CHECK(order.size()==count);
    for (int32_t row=0;row<count;++row)
        CHECK(order[row]==count-4-4*(row/4)+row%4);
    const auto sorted=cudadem::particle_clump_order(ids,live,true);
    for (int32_t row=0;row<count;++row) CHECK(sorted[row]==row);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"particle clump order: 3000 randomized differential cases, sparse INT32_MAX IDs, "
             <<count<<" members passed; million-row sort+preserve seconds="<<seconds<<'\n';
}

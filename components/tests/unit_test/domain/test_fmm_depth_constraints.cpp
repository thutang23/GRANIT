#include "src/domain/fmm_depth_constraints.h"
#include "src/domain/distributed_fmm_depth_controller.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
using namespace cudadem;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("hard depth check failed: " #x); } while(false)

static FMMLeafOccupancySummary summarize(const std::vector<uint64_t>& keys,size_t begin,size_t end) {
    FMMLeafOccupancySummary out{};out.source_count=end-begin;
    if (begin==end) return out;
    out.first_key=keys[begin];out.last_key=keys[end-1];
    for (size_t row=begin;row<end;) {
        size_t next=row+1;while(next<end&&keys[next]==keys[row]) ++next;
        const uint64_t count=next-row;
        out.maximum_count=std::max(out.maximum_count,count);
        if(row==begin)out.first_count=count;
        if(next==end)out.last_count=count;
        row=next;
    }
    return out;
}
static std::string snapshot(const DistributedFMMDepthController& controller) {
    std::ostringstream out(std::ios::binary);controller.save_checkpoint(out);return out.str();
}
template<class T> static void write(std::ostream& out,const T& value) {out.write(reinterpret_cast<const char*>(&value),sizeof(value));}
static std::string legacy_checkpoint() {
    std::ostringstream out(std::ios::binary);
    write(out,int32_t{1});write(out,true);write(out,false);
    write(out,uint64_t{0});write(out,uint64_t{0});write(out,uint64_t{1});write(out,uint64_t{0});
    write(out,int32_t{-1});write(out,int32_t{0});write(out,uint8_t{0});write(out,int32_t{0});
    for(int i=0;i<3;++i)write(out,uint64_t{0});
    struct Aggregate {double maximum,mean;uint64_t bytes;} aggregate{};
    write(out,aggregate);write(out,aggregate);write(out,uint64_t{1});write(out,uint64_t{0});
    struct LegacyState {int32_t selected,candidate,maximum,direction,confirmation,evaluations;int64_t count;} state{3,3,8,1,0,0,0};
    static_assert(sizeof(state)==32,"legacy state layout");write(out,state);write(out,uint8_t{1});write(out,uint8_t{0});
    for(int slot=0;slot<DistributedFMMDepthController::kMailboxSlots;++slot) {
        write(out,uint64_t{0});write(out,DistributedFMMDepthEpochTiming{});write(out,uint8_t{0});
    }
    write(out,uint32_t{0});return out.str();
}

int main() {
    // A leaf shared through empty ranks must use SUM, not max(local count).
    FMMLeafOccupancySummary a{7,7,100,100,100,100},b{7,7,101,101,101,101};
    CHECK(fmm_global_maximum_leaf_count({a,{},b})==201);
    std::mt19937 rng(78123);
    for(int sample=0;sample<2000;++sample) {
        std::vector<uint64_t> keys(rng()%700);
        for(auto& key:keys)key=rng()%30;
        std::sort(keys.begin(),keys.end());std::map<uint64_t,uint64_t> oracle;
        for(auto key:keys)++oracle[key];
        uint64_t maximum=0;
        for(auto entry:oracle)maximum=std::max(maximum,entry.second);
        std::vector<size_t> cuts{0,keys.size()};
        for(int rank=0;rank<7;++rank)cuts.push_back(keys.empty()?0:rng()%(keys.size()+1));
        std::sort(cuts.begin(),cuts.end());std::vector<FMMLeafOccupancySummary> shards;
        for(size_t rank=1;rank<cuts.size();++rank)shards.push_back(summarize(keys,cuts[rank-1],cuts[rank]));
        CHECK(fmm_global_maximum_leaf_count(shards)==maximum);
    }
    FMMLeafCounts counts{};
    for(int depth=0;depth<=20;++depth)counts[depth]=depth<5?256:(depth==5?128:64);
    const auto range=fmm_feasible_depth_range(counts,2,8,8,1./128,1.);
    CHECK(range.minimum_depth==5&&range.maximum_depth==6);
    CHECK(fmm_feasible_depth_range(counts,2,8,5,1./128,1.).maximum_depth==5);
    bool rejected=false;
    try {fmm_feasible_depth_range(counts,2,8,8,.6,1.);}catch(const std::runtime_error& error){rejected=std::string(error.what()).find("no feasible FMM depth")!=std::string::npos;}
    CHECK(rejected);
    rejected=false;try{fmm_feasible_depth_range(counts,2,8,4,0.,1.);}catch(const std::runtime_error&){rejected=true;}CHECK(rejected);
    auto plateau=fmm_identity_effective_depths();
    for(int depth=2;depth<=8;++depth)plateau[depth]=std::max(5,depth+1);
    const auto mapped=fmm_feasible_depth_range(counts,2,8,8,1./128,1.,plateau);
    CHECK(mapped.minimum_depth==2&&mapped.maximum_depth==5);
    const double boundary=std::ldexp(1.,-6)*(1.-kFMMClumpDiameterMargin);
    CHECK(fmm_feasible_depth_range(counts,2,8,8,boundary,1.).maximum_depth==5);
    // Canonical Y starts halfway through leaves at the alignment level.
    // Two 65-source slab boundaries merge to 130 once logical Y images exist.
    MortonGrid root{-2.,-2.,-2.,4.,3,2,0};
    for(int depth:{2,3}) {
        std::vector<uint64_t> all,stored;
        for(int repeat=0;repeat<65;++repeat)for(double y:{-.49,.49}) {
            uint64_t key=0;CHECK(fmm_constraint_source_key(root,0.,y,0.,key));stored.push_back(key>>(3*(20-depth)));
            for(int ny=-1;ny<=1;++ny) {CHECK(fmm_constraint_source_key(root,0.,y+ny,0.,key));all.push_back(key>>(3*(20-depth)));}
        }
        std::sort(stored.begin(),stored.end());std::sort(all.begin(),all.end());
        CHECK(summarize(stored,0,stored.size()).maximum_count==65);
        CHECK(summarize(all,0,all.size()).maximum_count==uint64_t(depth==2?130:65));
    }
    uint64_t key=0;
    CHECK(!fmm_constraint_source_key(root,2.,0.,0.,key));
    CHECK(!fmm_constraint_source_key(root,std::numeric_limits<double>::quiet_NaN(),0.,0.,key));
    // The nearest world-coordinate neighbor can round to fx==1 during
    // subtraction, exactly as the production Karras fixed-root check does.
    CHECK(!fmm_constraint_source_key(root,std::nextafter(2.,-2.),0.,0.,key));
    CHECK(fmm_constraint_source_key(root,std::nextafter(std::nextafter(2.,-2.),-2.),0.,0.,key));

    AdaptiveDepthConfig config;config.enabled=true;config.min_depth=2;config.max_depth=8;
    config.trial_interval_evaluations=1;config.trial_warmup_evaluations=0;config.trial_samples=1;config.confirmation_trials=1;
    DistributedFMMDepthController controller;CHECK(controller.initialize(config,1,{3}));
    uint64_t epoch=0;
    while(!controller.trial_active()&&epoch<10) {
        auto command=controller.command_for_gpu(0);DistributedFMMDepthEpochTiming timing{};
        timing.epoch=++epoch;timing.launch_depth=command.launch_depth;timing.command_trial_id=command.trial_id;
        timing.elapsed_ns=1000;timing.local_owned_particles=300;CHECK(controller.publish_timing(0,timing));controller.process_ready_epochs();
    }
    CHECK(controller.trial_active());controller.set_effective_depth_range(0,5,6,epoch);
    CHECK(!controller.trial_active()&&controller.command_for_gpu(0).launch_depth==5);
    CHECK(controller.state_for_gpu(0).effective_min_depth==5);
    DistributedFMMDepthEpochTiming stale{};stale.epoch=epoch;CHECK(!controller.publish_timing(0,stale));
    for(int step=0;step<20;++step) {
        auto command=controller.command_for_gpu(0);CHECK(command.launch_depth>=5&&command.launch_depth<=6);
        DistributedFMMDepthEpochTiming timing{};timing.epoch=++epoch;timing.launch_depth=command.launch_depth;
        timing.command_trial_id=command.trial_id;timing.elapsed_ns=1000;timing.local_owned_particles=300;
        CHECK(controller.publish_timing(0,timing));controller.process_ready_epochs();
    }
    controller.set_effective_depth_range(0,6,6,epoch);
    CHECK(controller.command_for_gpu(0).launch_depth==6);
    const auto bytes=snapshot(controller);DistributedFMMDepthController restored;
    std::istringstream input(bytes,std::ios::binary);restored.restore_checkpoint(input,config);
    CHECK(snapshot(restored)==bytes&&restored.state_for_gpu(0).effective_min_depth==6);
    DistributedFMMDepthController legacy;std::istringstream old(legacy_checkpoint(),std::ios::binary);legacy.restore_checkpoint(old,config);
    CHECK(legacy.state_for_gpu(0).effective_min_depth==2&&legacy.command_for_gpu(0).launch_depth==3);
    legacy.set_effective_depth_range(0,4,7,0);CHECK(legacy.command_for_gpu(0).launch_depth==4);
    std::cout<<"FMM hard-depth constraints PASS: exact global occupancy, logical shear images, diameter/memory intersection, bounded trials, v1/v2 checkpoint replay\n";
}

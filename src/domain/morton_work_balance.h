#pragma once
#include "morton_partition.h"
#include <numeric>

namespace cudadem {
struct MortonWorkLeaf {
    uint64_t key=0;
    double seconds=0;
    uint64_t migration_bytes=0;
};

// Workers retain their own leaf prefix sums. Only rank totals, P-1 cut keys,
// and P-by-P predicted load/byte scalars cross the collective boundaries.
class MortonWorkBalance {
    struct Rank {
        std::vector<MortonWorkLeaf> leaves;
        std::vector<double> prefix;
        std::vector<double> proposed_seconds;
        std::vector<uint64_t> moved_bytes;
        double total=0;
    };
    std::vector<Rank> ranks_;
    std::vector<double> rank_prefix_;
    std::vector<uint64_t> old_cuts_, proposed_cuts_;
    MortonGrid grid_{};
public:
    double current_max=0, proposed_max=0, migration_seconds=0;
    void initialize(int count){ranks_.resize(count);rank_prefix_.resize(count+1);}
    void begin(const MortonGrid& grid,const uint64_t* cuts){
        grid_=grid;old_cuts_.assign(cuts,cuts+ranks_.size()+1);proposed_cuts_=old_cuts_;
    }
    void set_local(int rank,std::vector<MortonWorkLeaf> leaves){
        auto& r=ranks_.at(rank);
        std::sort(leaves.begin(),leaves.end(),[](const auto&a,const auto&b){return a.key<b.key;});
        r.leaves.clear();r.total=0;
        for(const auto& leaf:leaves){
            if(leaf.key<old_cuts_[rank]||leaf.key>=old_cuts_[rank+1]||grid_.leaf_begin(leaf.key)!=leaf.key||
                !std::isfinite(leaf.seconds)||leaf.seconds<0)throw std::runtime_error("invalid local Morton work record");
            if(!r.leaves.empty()&&r.leaves.back().key==leaf.key){r.leaves.back().seconds+=leaf.seconds;r.leaves.back().migration_bytes+=leaf.migration_bytes;}
            else r.leaves.push_back(leaf);
            r.total+=leaf.seconds;
        }
        r.prefix.assign(1,0);
        for(const auto& leaf:r.leaves)r.prefix.push_back(r.prefix.back()+leaf.seconds);
        r.proposed_seconds.assign(ranks_.size(),0);r.moved_bytes.assign(ranks_.size(),0);
    }
    void sum_rank_totals(){
        rank_prefix_[0]=0;current_max=0;
        for(size_t r=0;r<ranks_.size();++r){rank_prefix_[r+1]=rank_prefix_[r]+ranks_[r].total;current_max=std::max(current_max,ranks_[r].total);}
        if(!std::isfinite(rank_prefix_.back()))throw std::overflow_error("Morton work sum overflow");
    }
    void choose_local_cuts(int rank){
        const auto& r=ranks_.at(rank);if(r.total<=0)return;
        for(size_t cut=1;cut<ranks_.size();++cut){
            const double target=rank_prefix_.back()*static_cast<double>(cut)/ranks_.size();
            if(target<rank_prefix_[rank]||target>=rank_prefix_[rank+1])continue;
            const double local=target-rank_prefix_[rank];
            auto it=std::upper_bound(r.prefix.begin(),r.prefix.end(),local);
            size_t i=std::min<size_t>(r.leaves.size()-1,it-r.prefix.begin()-1);
            if(local>=r.prefix[i]+r.leaves[i].seconds*0.5)++i;
            proposed_cuts_[cut]=i<r.leaves.size()?r.leaves[i].key:old_cuts_[rank+1];
        }
    }
    void predict_local(int rank){
        auto& r=ranks_.at(rank);
        for(const auto& leaf:r.leaves){const int owner=morton_owner(leaf.key,proposed_cuts_.data(),ranks_.size());
            r.proposed_seconds[owner]+=leaf.seconds;
            if(owner!=rank)r.moved_bytes[owner]+=leaf.migration_bytes;
        }
    }
    bool accept(double minimum_fraction,int amortization_steps,double seconds_per_byte){
        if(!std::is_sorted(proposed_cuts_.begin(),proposed_cuts_.end()))throw std::runtime_error("nonmonotonic distributed Morton cuts");
        if(current_max<=0||proposed_cuts_==old_cuts_)return false;
        std::vector<double> loads(ranks_.size(),0),transfer(ranks_.size(),0);
        for(size_t from=0;from<ranks_.size();++from)for(size_t to=0;to<ranks_.size();++to){
            loads[to]+=ranks_[from].proposed_seconds[to];
            const double cost=ranks_[from].moved_bytes[to]*seconds_per_byte;
            transfer[from]+=cost;transfer[to]+=cost;
        }
        proposed_max=*std::max_element(loads.begin(),loads.end());
        migration_seconds=*std::max_element(transfer.begin(),transfer.end());
        return proposed_max<current_max*(1.0-minimum_fraction) &&
            (current_max-proposed_max)*std::max(1,amortization_steps)>migration_seconds;
    }
    const std::vector<uint64_t>& cuts()const{return proposed_cuts_;}
};
} // namespace cudadem

#ifndef CUDADEM_RCB_BALANCER_H
#define CUDADEM_RCB_BALANCER_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "rcb_math.h"

namespace body_fitted_fmm {
struct UnifiedNode; struct M2LTask; struct PairTask;
struct GroupedTargetInteractions; struct PersistentTreeMetadata;
}

namespace cudadem {

// Weight policy: per-particle FMM neighbor term. SP1 uses ZeroFmmWeight.
struct ZeroFmmWeight {
    __device__ float operator()(int32_t /*i*/) const { return 0.0f; }
};

// Compute Wi = alpha + beta*num_contacts[i] + gamma*fmm(i) for local particles.
template <class FmmWeightPolicy>
__global__ void compute_weights_kernel(
    const int32_t* __restrict__ num_contacts,
    int32_t n, float alpha, float beta, float gamma,
    FmmWeightPolicy fmm, float* __restrict__ out_weight);

// Histogram particle weights along `axis` into `bins` over [lo,hi], counting only
// particles whose position lies inside [box_min, box_max).
__global__ void axis_histogram_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const float*  __restrict__ weight,
    int32_t n, int axis, int num_bins,
    double lo, double hi,
    double box_min_x, double box_min_y, double box_min_z,
    double box_max_x, double box_max_y, double box_max_z,
    unsigned long long* __restrict__ bins);  // fixed-point accumulation

struct GPUDomainState;     // fwd (defined in domain_manager.h)
struct DomainParamsGPU;    // fwd
struct ContactPair;
constexpr int32_t RCB_MAX_PEERS = 8;

struct AdaptiveRCBCoefficients {
    float base = 0.0f, contact_pair = 0.0f, m2l = 0.0f, p2p = 0.0f;
    float let_source_node = 0.0f, let_source_particle = 0.0f, let_receiver = 0.0f, halo_lifecycle = 0.0f;
};
struct FMMWorkloadViewInput {
    const body_fitted_fmm::UnifiedNode* nodes = nullptr; const int* sorted_to_original = nullptr;
    const body_fitted_fmm::M2LTask *m2l_same = nullptr, *m2l_cross = nullptr;
    const body_fitted_fmm::PairTask *p2p_regular = nullptr, *p2p_self = nullptr;
    int32_t num_nodes = 0, m2l_same_count = 0, m2l_cross_count = 0, p2p_regular_count = 0, p2p_self_count = 0;
    const body_fitted_fmm::GroupedTargetInteractions* grouped_targets = nullptr;
    const int* children_compact = nullptr;
    const body_fitted_fmm::PersistentTreeMetadata* metadata = nullptr;
    int32_t node_capacity = 0;
};
struct AdaptiveRCBInput {
    int32_t owned_count = 0;
    const ContactPair* contact_pairs = nullptr; int32_t contact_pair_count = 0;
    const uint32_t* halo_destination_mask = nullptr; // one bit per peer, indexed only by owned particle
    float halo_peer_cost[RCB_MAX_PEERS]{}; // already link-class calibrated lifecycle costs
    FMMWorkloadViewInput fmm{};
    // Indexed by source node: x=accepted node records, y=leaf-particle records.
    const uint2* let_source_work = nullptr; int32_t let_source_work_count = 0;
    float receiver_uniform_units = 0.0f;
    AdaptiveRCBCoefficients coefficients{};
};
struct RCBProposal {
    bool valid = false, eligible = false; RcbBox boxes[RCB_MAX_PEERS]{};
    float current_tmax_over_avg = 0.0f, proposed_tmax_over_avg = 0.0f;
};
void build_adaptive_rcb_weights(const AdaptiveRCBInput& input, float* weights,
                                float* sorted_difference, int32_t* projection_status,
                                cudaStream_t stream);
__global__ void scale_weights_kernel(float* weights, int n, float scale);

class RCBBalancer {
public:
    void initialize(int32_t num_gpus, const std::vector<int32_t>& particle_capacities,
                    float alpha, float beta, float gamma, int32_t num_bins);
    void shutdown();

    // Computes N disjoint AABBs into params.gpu_box_* using distributed histograms.
    // Does NOT move particle data; the caller's migration pass redistributes.
    void compute_splits(std::vector<GPUDomainState>& gpu_states,
                        DomainParamsGPU& params);
    // Computes candidate boxes only.  Caller applies them only after migration readiness.
    RCBProposal compute_adaptive_proposal(std::vector<GPUDomainState>& gpu_states,
        const DomainParamsGPU& params, const std::vector<AdaptiveRCBInput>& inputs,
        float min_predicted_improvement);
    static bool meets_predicted_improvement(float current_ratio, float proposed_ratio, float minimum);

private:
    int32_t num_gpus_ = 0;
    int32_t num_bins_ = 1024;
    float alpha_ = 1.0f, beta_ = 0.1f, gamma_ = 0.0f;

    // Per-GPU device scratch.
    std::vector<float*> d_weight_;                 // [num_gpus] weight per particle
    std::vector<int32_t> weight_capacity_;         // independent per-GPU high-water marks
    std::vector<unsigned long long*> d_bins_;      // [num_gpus] histogram bins
    std::vector<unsigned long long*> h_bins_;      // [num_gpus] pinned host copy
    std::vector<float*> d_projected_diff_;         // [num_gpus] N_local+1 grow-only projection arena
    std::vector<int32_t> projected_diff_capacity_;
    std::vector<int32_t*> d_projection_status_;    // [num_gpus] first invalid borrowed range/mapping
};

} // namespace cudadem
#endif // CUDADEM_RCB_BALANCER_H

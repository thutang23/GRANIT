#include "rcb_balancer.h"
#include "domain_manager.h"   // GPUDomainState, DomainParamsGPU, CUDA_CHECK, MAX_MULTI_GPUS
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace cudadem {

void RCBBalancer::initialize(int32_t num_gpus,
                             const std::vector<int32_t>& particle_capacities,
                             float alpha, float beta, float gamma, int32_t num_bins) {
    if (static_cast<int32_t>(particle_capacities.size()) != num_gpus) {
        throw std::runtime_error("RCB particle capacity count does not match GPU count");
    }
    num_gpus_ = num_gpus;
    num_bins_ = num_bins;
    alpha_ = alpha; beta_ = beta; gamma_ = gamma;
    d_weight_.assign(num_gpus, nullptr);
    weight_capacity_ = particle_capacities;
    d_bins_.assign(num_gpus, nullptr);
    h_bins_.assign(num_gpus, nullptr);
    d_projected_diff_.assign(num_gpus, nullptr);
    projected_diff_capacity_.assign(num_gpus, 0);
    d_projection_status_.assign(num_gpus, nullptr);
    for (int g = 0; g < num_gpus; ++g) {
        if (weight_capacity_[g] <= 0) {
            throw std::runtime_error("RCB particle capacity must be positive for GPU "
                                     + std::to_string(g));
        }
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&d_weight_[g],
                              static_cast<size_t>(weight_capacity_[g]) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bins_[g], num_bins * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMallocHost(&h_bins_[g], num_bins * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_projection_status_[g], sizeof(int32_t)));
    }
}

void RCBBalancer::shutdown() {
    for (int g = 0; g < num_gpus_; ++g) {
        cudaSetDevice(g);
        if (d_weight_[g]) cudaFree(d_weight_[g]);
        if (d_bins_[g]) cudaFree(d_bins_[g]);
        if (h_bins_[g]) cudaFreeHost(h_bins_[g]);
        if (d_projected_diff_[g]) cudaFree(d_projected_diff_[g]);
        if (d_projection_status_[g]) cudaFree(d_projection_status_[g]);
    }
    d_weight_.clear(); d_bins_.clear(); h_bins_.clear();
    weight_capacity_.clear();
    d_projected_diff_.clear(); projected_diff_capacity_.clear();
    d_projection_status_.clear();
}

bool RCBBalancer::meets_predicted_improvement(float current_ratio, float proposed_ratio, float minimum) {
    return std::isfinite(current_ratio) && std::isfinite(proposed_ratio) && current_ratio > 0.0f &&
           proposed_ratio > 0.0f && minimum >= 0.0f &&
           (current_ratio - proposed_ratio) / current_ratio >= minimum;
}

RCBProposal RCBBalancer::compute_adaptive_proposal(std::vector<GPUDomainState>& gpu_states,
    const DomainParamsGPU& params, const std::vector<AdaptiveRCBInput>& inputs, float min_predicted_improvement) {
    RCBProposal proposal;
    for (int32_t g = 0; g < num_gpus_ && g < RCB_MAX_PEERS; ++g) {
        proposal.boxes[g] = {{params.gpu_box_min_x[g], params.gpu_box_min_y[g],
                              params.gpu_box_min_z[g]},
                             {params.gpu_box_max_x[g], params.gpu_box_max_y[g],
                              params.gpu_box_max_z[g]}};
    }
    if (num_gpus_ <= 1 || num_gpus_ > RCB_MAX_PEERS || num_bins_ <= 1 ||
        static_cast<int32_t>(gpu_states.size()) != num_gpus_ ||
        static_cast<int32_t>(inputs.size()) != num_gpus_ ||
        !std::isfinite(min_predicted_improvement) || min_predicted_improvement < 0.0f) {
        return proposal;
    }

    const RcbBox root{{params.domain_min_x, params.domain_min_y, params.domain_min_z},
                      {params.domain_max_x, params.domain_max_y, params.domain_max_z}};
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(root.min[axis]) || !std::isfinite(root.max[axis]) ||
            !(root.min[axis] < root.max[axis])) return proposal;
    }

    auto valid_input = [](const AdaptiveRCBInput& input) {
        const float coefficients[] = {
            input.coefficients.base, input.coefficients.contact_pair,
            input.coefficients.m2l, input.coefficients.p2p,
            input.coefficients.let_source_node,
            input.coefficients.let_source_particle,
            input.coefficients.let_receiver,
            input.coefficients.halo_lifecycle
        };
        if (!(input.coefficients.base > 0.0f)) return false;
        for (float value : coefficients) {
            if (!std::isfinite(value) || value < 0.0f) return false;
        }
        if (!std::isfinite(input.receiver_uniform_units) ||
            input.receiver_uniform_units < 0.0f || input.contact_pair_count < 0 ||
            input.fmm.num_nodes < 0 || input.fmm.m2l_same_count < 0 ||
            input.fmm.m2l_cross_count < 0 || input.fmm.p2p_regular_count < 0 ||
            input.fmm.p2p_self_count < 0 || input.let_source_work_count < 0) return false;
        if (input.fmm.node_capacity < 0) return false;
        for (float cost : input.halo_peer_cost) {
            if (!std::isfinite(cost) || cost < 0.0f) return false;
        }
        if (input.contact_pair_count > 0 && !input.contact_pairs) return false;
        if (input.fmm.m2l_same_count > 0 && !input.fmm.m2l_same) return false;
        if (input.fmm.m2l_cross_count > 0 && !input.fmm.m2l_cross) return false;
        if (input.fmm.p2p_regular_count > 0 && !input.fmm.p2p_regular) return false;
        if (input.fmm.p2p_self_count > 0 && !input.fmm.p2p_self) return false;
        if (input.fmm.grouped_targets &&
            (!input.fmm.metadata || !input.fmm.children_compact ||
             input.fmm.node_capacity <= 0)) return false;
        const bool has_spatial_fmm = input.fmm.m2l_same_count > 0 ||
            input.fmm.m2l_cross_count > 0 || input.fmm.p2p_regular_count > 0 ||
            input.fmm.p2p_self_count > 0 || input.fmm.grouped_targets ||
            input.let_source_work_count > 0;
        if (has_spatial_fmm && (!input.fmm.nodes || !input.fmm.sorted_to_original ||
                                input.fmm.num_nodes <= 0)) return false;
        if (input.let_source_work_count > 0 && (!input.let_source_work ||
            input.let_source_work_count > input.fmm.num_nodes)) return false;
        return true;
    };

    std::vector<std::vector<float>> host_weights(static_cast<size_t>(num_gpus_));
    int64_t global_n = 0;
    for (int32_t g = 0; g < num_gpus_; ++g) {
        const AdaptiveRCBInput& input = inputs[g];
        if (input.owned_count < 0 ||
            input.owned_count != gpu_states[g].subdomain.num_local_particles ||
            !valid_input(input)) return proposal;
        global_n += input.owned_count;
        if (input.owned_count == 0) continue;

        CUDA_CHECK(cudaSetDevice(g));
        if (input.owned_count > weight_capacity_[g]) {
            const int64_t grown = static_cast<int64_t>(weight_capacity_[g])
                                + (static_cast<int64_t>(weight_capacity_[g]) + 1) / 2;
            const int64_t target = std::max<int64_t>(input.owned_count, grown);
            if (target > std::numeric_limits<int32_t>::max()) return proposal;
            float* replacement = nullptr;
            CUDA_CHECK(cudaMalloc(&replacement, static_cast<size_t>(target) * sizeof(float)));
            CUDA_CHECK(cudaFree(d_weight_[g]));
            d_weight_[g] = replacement;
            weight_capacity_[g] = static_cast<int32_t>(target);
        }
        const int32_t required = input.owned_count + 1;
        if (required > projected_diff_capacity_[g]) {
            const int64_t grown = static_cast<int64_t>(projected_diff_capacity_[g])
                                + (static_cast<int64_t>(projected_diff_capacity_[g]) + 1) / 2;
            const int64_t target = std::max<int64_t>(required, grown);
            if (target > std::numeric_limits<int32_t>::max()) return proposal;
            float* replacement = nullptr;
            CUDA_CHECK(cudaMalloc(&replacement, static_cast<size_t>(target) * sizeof(float)));
            if (d_projected_diff_[g]) CUDA_CHECK(cudaFree(d_projected_diff_[g]));
            d_projected_diff_[g] = replacement;
            projected_diff_capacity_[g] = static_cast<int32_t>(target);
        }
        CUDA_CHECK(cudaMemsetAsync(d_projection_status_[g], 0, sizeof(int32_t),
                                   gpu_states[g].compute_stream));
        build_adaptive_rcb_weights(input, d_weight_[g], d_projected_diff_[g],
                                   d_projection_status_[g], gpu_states[g].compute_stream);
        CUDA_CHECK(cudaGetLastError());
    }
    if (global_n <= 0) return proposal;

    double global_sum = 0.0;
    for (int32_t g = 0; g < num_gpus_; ++g) {
        const int32_t n = inputs[g].owned_count;
        if (n == 0) continue;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(gpu_states[g].compute_stream));
        int32_t projection_status = 0;
        CUDA_CHECK(cudaMemcpy(&projection_status, d_projection_status_[g],
                              sizeof(projection_status), cudaMemcpyDeviceToHost));
        if (projection_status != 0) return proposal;
        host_weights[g].resize(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(host_weights[g].data(), d_weight_[g],
                              static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (float weight : host_weights[g]) {
            if (!std::isfinite(weight) || weight < 0.0f) return proposal;
            global_sum += weight;
        }
    }
    if (!std::isfinite(global_sum) || !(global_sum > 0.0)) return proposal;
    const double scale_double = static_cast<double>(global_n) / global_sum;
    if (!std::isfinite(scale_double) || !(scale_double > 0.0) ||
        scale_double > std::numeric_limits<float>::max()) return proposal;
    const float scale = static_cast<float>(scale_double);
    constexpr double max_fixed_weight =
        static_cast<double>(std::numeric_limits<unsigned long long>::max()) / 1024.0;
    for (int32_t g = 0; g < num_gpus_; ++g) {
        const int32_t n = inputs[g].owned_count;
        if (n == 0) continue;
        for (float& weight : host_weights[g]) {
            const double normalized = static_cast<double>(weight) * scale_double;
            if (!std::isfinite(normalized) || normalized < 0.0 ||
                normalized > max_fixed_weight) return proposal;
            weight = static_cast<float>(normalized);
        }
        CUDA_CHECK(cudaSetDevice(g));
        scale_weights_kernel<<<(n + 255) / 256, 256, 0,
                               gpu_states[g].compute_stream>>>(d_weight_[g], n, scale);
        CUDA_CHECK(cudaGetLastError());
    }

    bool histogram_valid = true;
    auto histogram = [&](const RcbBox& box, int axis, int nbins,
                         double lo, double hi, double* out) {
        std::fill(out, out + nbins, 0.0);
        std::vector<unsigned long long> accumulated(static_cast<size_t>(nbins), 0ull);
        for (int32_t g = 0; g < num_gpus_; ++g) {
            const int32_t n = inputs[g].owned_count;
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemsetAsync(d_bins_[g], 0,
                                       static_cast<size_t>(nbins) * sizeof(*d_bins_[g]),
                                       gpu_states[g].compute_stream));
            if (n > 0) {
                axis_histogram_kernel<<<(n + 255) / 256, 256, 0,
                    gpu_states[g].compute_stream>>>(
                    gpu_states[g].local_state.particles.pos_x,
                    gpu_states[g].local_state.particles.pos_y,
                    gpu_states[g].local_state.particles.pos_z,
                    d_weight_[g], n, axis, nbins, lo, hi,
                    box.min[0], box.min[1], box.min[2],
                    box.max[0], box.max[1], box.max[2], d_bins_[g]);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaMemcpyAsync(h_bins_[g], d_bins_[g],
                                       static_cast<size_t>(nbins) * sizeof(*h_bins_[g]),
                                       cudaMemcpyDeviceToHost, gpu_states[g].compute_stream));
        }
        for (int32_t g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(gpu_states[g].compute_stream));
            for (int32_t bin = 0; bin < nbins; ++bin) {
                if (std::numeric_limits<unsigned long long>::max() - accumulated[bin]
                    < h_bins_[g][bin]) {
                    histogram_valid = false;
                    continue;
                }
                accumulated[bin] += h_bins_[g][bin];
            }
        }
        for (int32_t bin = 0; bin < nbins; ++bin)
            out[bin] = static_cast<double>(accumulated[bin]) / 1024.0;
    };

    RcbBox candidate[RCB_MAX_PEERS]{};
    rcb_compute_boxes(root, num_gpus_, num_bins_, 1, histogram, candidate);
    if (!histogram_valid) return proposal;
    for (int32_t g = 0; g < num_gpus_; ++g) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(candidate[g].min[axis]) ||
                !std::isfinite(candidate[g].max[axis]) ||
                candidate[g].min[axis] < root.min[axis] ||
                candidate[g].max[axis] > root.max[axis] ||
                !(candidate[g].min[axis] < candidate[g].max[axis])) return proposal;
        }
    }

    auto inside = [](double x, double y, double z, const RcbBox& box) {
        return x >= box.min[0] && x < box.max[0] &&
               y >= box.min[1] && y < box.max[1] &&
               z >= box.min[2] && z < box.max[2];
    };
    std::vector<double> current_load(static_cast<size_t>(num_gpus_), 0.0);
    std::vector<double> candidate_load(static_cast<size_t>(num_gpus_), 0.0);
    for (int32_t g = 0; g < num_gpus_; ++g) {
        const int32_t n = inputs[g].owned_count;
        if (n == 0) continue;
        std::vector<double> x(static_cast<size_t>(n));
        std::vector<double> y(static_cast<size_t>(n));
        std::vector<double> z(static_cast<size_t>(n));
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(x.data(), gpu_states[g].local_state.particles.pos_x,
                              static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y.data(), gpu_states[g].local_state.particles.pos_y,
                              static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z.data(), gpu_states[g].local_state.particles.pos_z,
                              static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
        for (int32_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
                return proposal;
            int32_t candidate_owner = -1;
            for (int32_t box_index = 0; box_index < num_gpus_; ++box_index) {
                if (inside(x[i], y[i], z[i], candidate[box_index])) {
                    if (candidate_owner >= 0) return proposal;
                    candidate_owner = box_index;
                }
            }
            // The current owner is the rank that physically owns this entry.
            // Immediately after an accepted cut, migration may not have run yet,
            // so requiring every entry to already lie in its new current box
            // would reject a valid follow-up proposal during that transition.
            if (candidate_owner < 0) return proposal;
            current_load[g] += host_weights[g][i];
            candidate_load[candidate_owner] += host_weights[g][i];
        }
    }
    auto ratio = [&](const std::vector<double>& loads) {
        double sum = 0.0;
        double maximum = 0.0;
        for (double load : loads) {
            if (!std::isfinite(load) || load < 0.0) return 0.0f;
            sum += load;
            maximum = std::max(maximum, load);
        }
        return std::isfinite(sum) && sum > 0.0
            ? static_cast<float>(maximum * num_gpus_ / sum) : 0.0f;
    };
    proposal.current_tmax_over_avg = ratio(current_load);
    proposal.proposed_tmax_over_avg = ratio(candidate_load);
    for (int32_t g = 0; g < num_gpus_; ++g) proposal.boxes[g] = candidate[g];
    proposal.valid = true;
    proposal.eligible = meets_predicted_improvement(proposal.current_tmax_over_avg,
        proposal.proposed_tmax_over_avg, min_predicted_improvement);
    return proposal;
}

void RCBBalancer::compute_splits(std::vector<GPUDomainState>& gpu_states,
                                 DomainParamsGPU& params) {
    // 1) Per-GPU weights (computed once; reused across all histogram queries).
    for (int g = 0; g < num_gpus_; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        int32_t n = gpu_states[g].subdomain.num_local_particles;
        if (n <= 0) continue;
        if (n > weight_capacity_[g]) {
            const int32_t old_capacity = weight_capacity_[g];
            const int64_t grown = static_cast<int64_t>(old_capacity)
                                + (static_cast<int64_t>(old_capacity) + 1) / 2;
            const int64_t target = std::max<int64_t>(n, grown);
            if (target > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("RCB weight capacity exceeds int32 indexing on GPU "
                                         + std::to_string(g));
            }
            float* replacement = nullptr;
            const cudaError_t status = cudaMalloc(
                &replacement, static_cast<size_t>(target) * sizeof(float));
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    "FATAL RCB weight resize on GPU " + std::to_string(g)
                    + ": old_capacity=" + std::to_string(old_capacity)
                    + ", required_capacity=" + std::to_string(n)
                    + ", requested_capacity=" + std::to_string(target)
                    + ", cuda_error=" + cudaGetErrorString(status));
            }
            CUDA_CHECK(cudaFree(d_weight_[g]));
            d_weight_[g] = replacement;
            weight_capacity_[g] = static_cast<int32_t>(target);
            std::printf("[memory] GPU %d RCB grow arena=weight old_capacity=%d "
                        "required=%d new_capacity=%d bytes=%zu\n",
                        g, old_capacity, n, weight_capacity_[g],
                        static_cast<size_t>(weight_capacity_[g]) * sizeof(float));
        }
        int bs = 256, nb = (n + bs - 1) / bs;
        compute_weights_kernel<ZeroFmmWeight><<<nb, bs, 0, gpu_states[g].compute_stream>>>(
            gpu_states[g].local_state.particles.num_contacts,
            n, alpha_, beta_, gamma_, ZeroFmmWeight{}, d_weight_[g]);
    }
    for (int g = 0; g < num_gpus_; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(gpu_states[g].compute_stream));
    }

    // 2) Root box = global domain bounds.
    RcbBox root;
    root.min[0] = params.domain_min_x; root.max[0] = params.domain_max_x;
    root.min[1] = params.domain_min_y; root.max[1] = params.domain_max_y;
    root.min[2] = params.domain_min_z; root.max[2] = params.domain_max_z;

    // 3) HistogramFn: distributed histogram reduced across all GPUs (fixed-point ints),
    //    converted to double for the pure driver.
    const int num_bins = num_bins_;
    HistogramFn histogram =
        [&](const RcbBox& box, int axis, int nbins, double lo, double hi, double* out) {
        for (int i = 0; i < nbins; ++i) out[i] = 0.0;
        std::vector<unsigned long long> acc(nbins, 0ull);
        for (int g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            int32_t n = gpu_states[g].subdomain.num_local_particles;
            CUDA_CHECK(cudaMemsetAsync(d_bins_[g], 0, nbins * sizeof(unsigned long long),
                                       gpu_states[g].compute_stream));
            if (n > 0) {
                int bs = 256, nb = (n + bs - 1) / bs;
                axis_histogram_kernel<<<nb, bs, 0, gpu_states[g].compute_stream>>>(
                    gpu_states[g].local_state.particles.pos_x,
                    gpu_states[g].local_state.particles.pos_y,
                    gpu_states[g].local_state.particles.pos_z,
                    d_weight_[g], n, axis, nbins, lo, hi,
                    box.min[0], box.min[1], box.min[2],
                    box.max[0], box.max[1], box.max[2],
                    d_bins_[g]);
            }
            CUDA_CHECK(cudaMemcpyAsync(h_bins_[g], d_bins_[g],
                                       nbins * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost,
                                       gpu_states[g].compute_stream));
        }
        for (int g = 0; g < num_gpus_; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(gpu_states[g].compute_stream));
            for (int i = 0; i < nbins; ++i) acc[i] += h_bins_[g][i];
        }
        for (int i = 0; i < nbins; ++i) out[i] = static_cast<double>(acc[i]) / 1024.0;
    };

    // 4) Run the pure recursive driver.
    RcbBox boxes[MAX_MULTI_GPUS];
    rcb_compute_boxes(root, num_gpus_, num_bins, /*refine_passes=*/1, histogram, boxes);

    // 5) Write boxes back into params (ownership convention: min inclusive, max exclusive).
    for (int g = 0; g < num_gpus_; ++g) {
        params.gpu_box_min_x[g] = boxes[g].min[0];
        params.gpu_box_min_y[g] = boxes[g].min[1];
        params.gpu_box_min_z[g] = boxes[g].min[2];
        params.gpu_box_max_x[g] = boxes[g].max[0];
        params.gpu_box_max_y[g] = boxes[g].max[1];
        params.gpu_box_max_z[g] = boxes[g].max[2];
    }
}

} // namespace cudadem

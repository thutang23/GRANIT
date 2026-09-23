#ifndef CUDADEM_RCB_MATH_H
#define CUDADEM_RCB_MATH_H

// =============================================================================
// Pure host RCB math (NO CUDA). Unit-testable with plain g++.
// =============================================================================
#include <cstdint>
#include <functional>

namespace cudadem {

struct RcbBox {
    double min[3];
    double max[3];
};

// Index (0,1,2) of the axis with the largest extent.
inline int longest_axis(const RcbBox& b) {
    double ex = b.max[0] - b.min[0];
    double ey = b.max[1] - b.min[1];
    double ez = b.max[2] - b.min[2];
    if (ex >= ey && ex >= ez) return 0;
    if (ey >= ex && ey >= ez) return 1;
    return 2;
}

struct SplitRatio {
    int    left_k;
    int    right_k;
    double left_fraction;  // fraction of total weight the left subgroup should hold
};

// Split a group of k (>= 2) GPUs into ceil(k/2) | floor(k/2); left weight fraction
// is proportional to the GPU counts.
inline SplitRatio proportional_split(int k) {
    int left_k = (k + 1) / 2;   // ceil
    int right_k = k - left_k;   // floor
    SplitRatio s;
    s.left_k = left_k;
    s.right_k = right_k;
    s.left_fraction = static_cast<double>(left_k) / static_cast<double>(k);
    return s;
}

// Coordinate in [lo,hi] where cumulative weight first reaches target_fraction*total.
// Linear interpolation within the crossing bin. Zero total -> geometric split.
inline double weighted_median_from_histogram(const double* bins, int num_bins,
                                             double lo, double hi,
                                             double target_fraction) {
    double total = 0.0;
    for (int i = 0; i < num_bins; ++i) total += bins[i];

    const double span = hi - lo;
    if (total <= 0.0 || span <= 0.0) {
        return lo + target_fraction * span;  // geometric fallback
    }

    const double target = target_fraction * total;
    const double bin_w = span / static_cast<double>(num_bins);
    double cum = 0.0;
    for (int i = 0; i < num_bins; ++i) {
        double next = cum + bins[i];
        if (next >= target) {
            // Linear interpolation inside bin i: how far through this bin's weight.
            double frac_in_bin = (bins[i] > 0.0) ? (target - cum) / bins[i] : 0.0;
            return lo + (static_cast<double>(i) + frac_in_bin) * bin_w;
        }
        cum = next;
    }
    return hi;  // numerical guard
}

// Provides a weighted histogram of particles within `box` along `axis` into
// bins_out[0..num_bins) spanning [lo,hi]. Pure interface: tests supply synthetic
// data; production supplies GPU histograms reduced across GPUs.
using HistogramFn = std::function<void(const RcbBox& box, int axis, int num_bins,
                                       double lo, double hi, double* bins_out)>;

namespace detail {

// Find the weighted-split plane along `axis` within `box`. Each refine pass narrows the
// search to the crossing bin and re-histograms it, carrying the absolute weight below the
// window (`cum_below`) so the target stays global. Pass 0 == single-histogram median
// (equivalent to weighted_median_from_histogram, which is tested independently in Task 3).
inline double rcb_find_plane(const RcbBox& box, int axis, int num_bins,
                             int refine_passes, double left_fraction,
                             const HistogramFn& histogram) {
    double lo = box.min[axis], hi = box.max[axis];
    if (hi - lo <= 0.0) return lo;
    int nb = (num_bins > 4096) ? 4096 : num_bins;
    double bins[4096];

    // Total weight over the full box (from the first, full-range histogram).
    histogram(box, axis, nb, lo, hi, bins);
    double total = 0.0;
    for (int i = 0; i < nb; ++i) total += bins[i];
    if (total <= 0.0) return lo + left_fraction * (hi - lo);  // geometric fallback

    const double target = left_fraction * total;  // absolute global weight target
    double win_lo = lo, win_hi = hi;
    double cum_below = 0.0;                        // weight strictly below win_lo
    double plane = lo + left_fraction * (hi - lo);

    for (int pass = 0; ; ++pass) {
        if (pass > 0) histogram(box, axis, nb, win_lo, win_hi, bins);
        double bw = (win_hi - win_lo) / static_cast<double>(nb);
        double cum = cum_below;
        int cross = nb - 1;
        for (int i = 0; i < nb; ++i) {
            double next = cum + bins[i];
            if (next >= target) { cross = i; break; }
            cum = next;
        }
        double bin_weight = bins[cross];
        double frac_in = (bin_weight > 0.0) ? (target - cum) / bin_weight : 0.0;
        plane = win_lo + (static_cast<double>(cross) + frac_in) * bw;

        if (pass >= refine_passes) break;
        // Narrow to the crossing bin for the next pass.
        cum_below = cum;
        win_lo = win_lo + static_cast<double>(cross) * bw;
        win_hi = win_lo + bw;
        if (win_hi - win_lo <= 0.0) break;
    }
    return plane;
}

inline void rcb_recurse(const RcbBox& box, int g0, int k, int num_bins,
                        int refine_passes, const HistogramFn& histogram,
                        RcbBox* out_boxes) {
    if (k <= 1) { out_boxes[g0] = box; return; }

    int axis = longest_axis(box);
    SplitRatio s = proportional_split(k);
    double plane = rcb_find_plane(box, axis, num_bins, refine_passes,
                                  s.left_fraction, histogram);

    // Guard against degenerate planes (keep boxes non-empty).
    double lo = box.min[axis], hi = box.max[axis];
    double eps = (hi - lo) * 1e-9;
    if (plane <= lo + eps) plane = lo + eps;
    if (plane >= hi - eps) plane = hi - eps;

    RcbBox left = box, right = box;
    left.max[axis] = plane;
    right.min[axis] = plane;

    rcb_recurse(left,  g0,           s.left_k,  num_bins, refine_passes, histogram, out_boxes);
    rcb_recurse(right, g0 + s.left_k, s.right_k, num_bins, refine_passes, histogram, out_boxes);
}

} // namespace detail

// Fill out_boxes[0..num_gpus) with disjoint AABBs covering `root`, balanced by weight.
// refine_passes: extra narrowed histogram passes per split (>=0) for median accuracy.
inline void rcb_compute_boxes(const RcbBox& root, int num_gpus, int num_bins,
                              int refine_passes, const HistogramFn& histogram,
                              RcbBox* out_boxes) {
    if (num_gpus <= 0) return;
    if (num_gpus == 1) { out_boxes[0] = root; return; }
    detail::rcb_recurse(root, 0, num_gpus, num_bins, refine_passes, histogram, out_boxes);
}

} // namespace cudadem
#endif // CUDADEM_RCB_MATH_H

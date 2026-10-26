#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cudadem {

// Return the stable normal -> ascending positive CID -> inactive permutation.
// SPH keeps its active input order. Storage depends on row count, never on the
// largest CID, and construction costs O(N log N) rather than O(N * max(CID)).
inline std::vector<int32_t> particle_clump_order(
    const std::vector<int32_t>& ids, const std::vector<uint8_t>& lifecycle,
    bool preserve_active_order) {
    if (ids.size() != lifecycle.size())
        throw std::runtime_error("invalid clump partition");
    for (int32_t id : ids)
        if (id < -1) throw std::runtime_error("invalid clump_id");
    for (uint8_t state : lifecycle)
        if (state > 1) throw std::runtime_error("invalid lifecycle_state");

    std::vector<int32_t> order;
    order.reserve(ids.size());
    for (size_t row = 0; row < ids.size(); ++row)
        if (ids[row] >= 0 && lifecycle[row] == 0)
            order.push_back(static_cast<int32_t>(row));

    if (!preserve_active_order) {
        const auto earlier_clump = [&](int32_t lhs, int32_t rhs) {
            return ids[lhs] < ids[rhs];
        };
        // Most production inputs are already canonical. Equal-CID rows retain
        // their input order in both the sorted and already-sorted cases.
        if (!std::is_sorted(order.begin(), order.end(), earlier_clump))
            std::stable_sort(order.begin(), order.end(), earlier_clump);
    }
    for (size_t row = 0; row < ids.size(); ++row)
        if (lifecycle[row] != 0 || ids[row] == -1)
            order.push_back(static_cast<int32_t>(row));
    return order;
}

} // namespace cudadem

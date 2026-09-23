#pragma once

namespace body_fitted_fmm {

template<int D>
struct TreeTraits;

template<>
struct TreeTraits<3> {
    static constexpr int BITS_PER_LEVEL = 3;
    static constexpr int CHILDREN_PER_NODE = 8;
    static constexpr int MAX_LEVELS = 21;  // levels 0..20 (float-center policy)
    static constexpr int MORTON_BITS = 63;
};

template<>
struct TreeTraits<2> {
    static constexpr int BITS_PER_LEVEL = 2;
    static constexpr int CHILDREN_PER_NODE = 4;
    static constexpr int MAX_LEVELS = 21;  // levels 0..20 (float-center policy)
    static constexpr int MORTON_BITS = 62;
};

}  // namespace body_fitted_fmm

#pragma once

#include "rt_backend.h"
#include "../neighbor_types.h"

namespace cudadem { namespace rt {

struct LaunchParams {
    SearchInput input;
    uint64_t traversable;
    ContactPair* pairs;
    unsigned long long* count;
    int32_t capacity;
};

} }

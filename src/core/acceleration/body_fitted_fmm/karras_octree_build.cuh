#pragma once

#include <cuda_runtime.h>

#include "../fmm/fmm_types.h"

namespace body_fitted_fmm {

struct PersistentTreeRuntime;

// Use the same CUB selection instantiation for allocation and memory preflight.
// This only queries storage; it does not allocate or execute GPU work.
size_t karras_unique_selection_scratch_bytes(int particle_capacity);

// Allocate/free the address-stable workspace captured by the production
// Karras tree graphs.  These functions are initialization/teardown only.
void initialize_karras_tree_workspace(PersistentTreeRuntime& runtime,
                                      fmm::FMMDeviceData& dd,
                                      int tree_dimension);
void shutdown_karras_tree_workspace(PersistentTreeRuntime& runtime);

// Append the complete fixed-depth Karras tree build to an already-capturing
// stream.  The caller owns the surrounding scheduler/stencil graph.
template<int D>
void capture_karras_tree_stages(PersistentTreeRuntime& runtime,
                                cudaStream_t stream);

}  // namespace body_fitted_fmm

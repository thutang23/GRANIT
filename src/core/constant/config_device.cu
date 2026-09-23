#include "command_line_interface.h"

namespace cudadem {

// =============================================================================
// GPU Constant Memory Definition
// =============================================================================
// This is the single source of truth for simulation configuration on the GPU.
// All kernels access this via the extern declaration in command_line_interface.h
// =============================================================================
// Use raw storage to avoid dynamic initialization error in CUDA.
// The actual data is copied via cudaMemcpyToSymbol at runtime.
__constant__ unsigned char dev_config_storage[sizeof(SimConfig)];

} // namespace cudadem

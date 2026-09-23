#pragma once
#include <cuda_runtime.h>

namespace cudadem {
struct SimulationState;
struct SimConfig;

// Add the compact-support difference between pair-mean cubic-spline gravity
// and a finite compact baseline. FMM supplies its Newtonian far field and
// regularized P2P; local/remote SPH neighbors supply the softening correction.
// Same compact baseline for P2P and its SPH correction; bound by the current
// tree's leaf width, not a static input box or requested adaptive depth.
double sph_gravity_regularization_length(const SimConfig& config, cudaStream_t stream);
void add_sph_gravity_softening_correction(
    SimulationState& state, const SimConfig& config, cudaStream_t stream, bool outer = false);
}

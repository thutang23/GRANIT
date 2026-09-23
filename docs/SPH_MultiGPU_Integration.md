# SPH single and multi-GPU integration

This integrates the earlier distributed SPH capabilities into production export `beb3d5150e177e2c387e85f02a8a6bcf7067f37c`. The updated EOS, stage controls, irreversible P-alpha law, continuous activation option, and six history fields remain in use.

## Corrections

Owned density/EOS state is exchanged before constitutive rates, and updated pressure/stress is exchanged before pair forces. SPH gravity uses Morton ownership, the updated local/remote FMM routing, and a compact correction for pair-mean smoothing lengths. Checkpoints retain complete SPH state and new history arrays, with the existing stage sidecar contract.

Both drivers now land at the same terminal time. Mid-stage resume preserves closed forces and rates; an internal stage boundary rebuilds them under the next stage. `rebootstrap_forces: true` still explicitly rebuilds at other times. Both drivers write a terminal checkpoint when checkpoint output is enabled, including a load/save with no steps.

Wide Weibull coefficients use double storage and logarithmic interval construction. Default material fingerprints remain compatible. The optional Tillotson `negative_pressure: zero` setting is allowed only with strength, damage, and porosity disabled; it supplies pressure-free release with a 1 m/s sound-speed floor. Stage-forced elastic tension follows the same policy. This remains an approximate liquid release treatment.

## Build and supported scope

Use CUDA 12.1 explicitly:

```bash
cmake -S . -B build-sph \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DFMM_P2P_LOCAL_FP32=OFF -DFMM_M2L_MODE=FP64 \
  -DGRANIT_ENABLE_OPTIX=OFF
cmake --build build-sph --target granit_sph granit_multigpu -j12
```

Distributed SPH gravity requires isolated geometry, fixed FMM depth, CUDA neighbors, ordinary gravity cadence, `multigpu.morton_ownership: true`, cutoff at least two, and halo factor at least one. Smoothing bounds must be finite and ordered; imported lengths must meet the configured lower bound. The runtime rejects a compact correction scale with `h_max/epsilon > 128`. Single-GPU fixed and symmetry boundaries remain single-GPU features. Periodic SPH and adaptive FMM depth are outside this distributed SPH configuration.

## Verification

See `components/tests/integration_test/sph_multigpu_validation/README.md` for the retained suite and fixed gates. All 95 checks passed in Slurm job 44685 on tested source commit `77e6071`. The largest direct-gravity relative RMS error was 0.198568%, below the fixed 1% gate. Detailed results and binary hashes are recorded in the delivery receipt. The cases compare IDs and checkpoint metadata exactly, floating fields within fixed reference-based tolerances, and approximate gravity against direct spline gravity separately. They exercise all six histories with nonzero values across the suite.

The host comparison evaluates 1,260 states across three EOS models and six material fingerprint variants against the unchanged export. Material-schema, wide-Weibull, and liquid-release host checks are retained as well.

These are short deterministic 216-particle integration cases on L40/SM89, with fixed timestep bounds and fractional stage/terminal boundaries. They do not establish physical convergence, long-duration conservation, adaptive-timestep equivalence, or performance scaling. The accepted five-hour asteroid result has not been rerun with this integration.

## Export provenance

When this integration is applied to the production export, `SOURCE_MANIFEST.json` remains the immutable upstream manifest. `SPH_INTEGRATION_MANIFEST.json` lists the replacement/additional file hashes, tested source revision, validation report, and original-export backup. The overlaid directory must not be described as pristine upstream. Development commits and the complete export baseline are retained in the isolated integration repository.

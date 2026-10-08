# Depth-8 LET and local P2P optimizations

Measured on 2026-10-08 using the supplied two-million-particle shear-box state,
all three sequential changes reduced the native compute-window time from
32.25 to 24.23 ms per DEM step: 24.87% less time, or 1.331 times the throughput.
These are short-run measurements of this workload, not a general scaling claim.

## Changes

| Stage | Commit | Behavior |
| --- | --- | --- |
| 1 | c7f153e | Store both remote route planes only for inverse-stencil boundary targets. A dense integer map preserves constant-time lookup and existing source ordering. |
| 2 | 21ed5f1 | Reuse individual far-reference rows after exact validation of target/source cell identities, direct/omit masks, and child occupancy. Remap source indexes to current buffers. |
| 3 | 94c8a9e | Use the existing deepest-level leaf list and omit cross/ancestor direct traversal when the submitted command disables population fallback and has no finite cutoff. |

Stage 2 still constructs current routes and moment ownership to validate reuse;
it does not cache coefficients, populations, particle ranges, or the direct list.
Changed rows use the original resolver. Geometry and routing changes invalidate
the cache. Stage 3 preserves block, child, and particle visitation order and
retains the general path for other routing policies.

## Controlled measurements

All stages use the same input, materials, shear phase, physical parameters,
fixed base depth 8, partition depth 6, fixed ownership cuts, and four A100-SXM4
40 GB GPUs. Transport remains the existing two NVLink pairs plus safe host
staging across pairs. The supplied OUTPUT was converted to a fresh INPUT;
contact history is reinitialized equally in every run, so this is not a full
checkpoint continuation.

Each run advances 128 steps of 0.4 s. The reported metric is
1000 times compute_window_wall_seconds divided by 64, using steps 32 through
96. It is the existing synchronized native compute window, excluding startup
and output work outside that window; it is not full-process wall time. Clean
runs disable diagnostics and verification oracles. Separate runs collect phase
timings and check invariants.

| Version | Clean runs (ms/step) | Mean (ms/step) | Incremental reduction |
| --- | --- | ---: | ---: |
| Baseline | 32.176664, 32.178540, 32.388426 | 32.247877 | — |
| Compact boundary routes | 27.897728, 27.754200 | 27.825964 | 13.71% |
| Add validated far-row cache | 26.907418, 26.640287 | 26.773852 | 3.78% |
| Add leaf-only local P2P | 23.775382, 24.557676, 24.348207 | 24.227088 | 9.51% |

The third baseline/final pair was a late control for timing variability. All
valid clean runs are included. Incremental percentages must not be added.
Repeated runs share one initial state and do not constitute independent physical
realizations or a confidence interval.

Phase medians support the intended mechanism: LET decreases from approximately
11–12 ms per GPU to 6–7 ms, and near-field work decreases from approximately
8–9 ms to 6 ms. Far-field work remains approximately 6 ms. Separate phase maxima
must not be summed to infer a critical path. The far-row cache reused 99.29% of
rows during the measured window of its dedicated verification run.

The baseline and candidates were rebuilt with CUDA 13.4.92, GCC 15.2, Release,
SM80, TF32_TC M2L, FP32 local P2P, and profiling enabled. Commit 5f6c322 adds
missing standard/Thrust includes equally used by the baseline and candidates.
The shared pinned yaml-cpp dependency needed an explicit cstdint include with
this compiler, and CMake required CMAKE_POLICY_VERSION_MINIMUM=3.5. The local
toolchain receipt records these compatibility fixes.

## Verification

- All three stages produce bitwise-identical gravity at the original state and
  at a second identical moved state reconstructed from baseline step 128.
- The full moving four-GPU cache oracle checks every stored reference against
  fresh resolution. The final cache run checked 558,068,000 references over
  516 rank-epochs without mismatch. The combined final-stage run also executes
  the leaf-only routing invariant check on all 516 rank evaluations.
- CPU tests cover compact/dense route equivalence, exact cache invalidation and
  remapping, and P2P ordering across 2D/3D, float/double, and force/acceleration
  variants. Existing geometry and transport host tests pass.
- Every stage passes a bitwise initial/final comparison for the small
  single-GPU image case with finite cutoff. The final solver logs leaf_only=0
  there, exercising the general path.

The unchanged baseline itself is not bitwise deterministic over the full
128-step trajectory: two baseline runs differ by at most 1.89e-9 m in position
and approximately 1.21e-8 relative L2 in final gravity. Strict trajectory reports
remain marked FAIL. Separate numerical gates use bounds declared before
inspecting candidate numerical differences, calibrated from that baseline pair;
all three candidates pass. These bounds are regression checks, not independent
force-accuracy or long-term convergence guarantees. Exact fixed-state gravity
and the reference/invariant oracles provide separate checks of the changes.

Sparse multi-GPU smoke coverage remains incomplete: the unchanged baseline
times out or reports a CUDA invalid argument on the prepared small 2D/3D cases.
Those failures are retained and are not counted as candidate passes. GPU SPH
and long physical trajectories were not exercised by this DEM campaign.

## Evidence and reproduction

The local campaign directory is benchmarks/shear_let_optimization_20261008.
Raw particle outputs and binaries are retained locally, not committed. Its
[final audit](../benchmarks/shear_let_optimization_20261008/campaign_audit_final.json)
binds configurations, input/material hashes, executable hashes, timing windows,
and validation reports. The directory also contains per-attempt logs, receipts,
telemetry, frozen executables, phase_summary.json, and toolchain_note.json.

The maintained CPU checks can be run from the repository root:

    uv run --no-project python components/tests/unit_test/body_fitted_fmm/test_compact_halo_routes.py
    uv run --no-project python components/tests/unit_test/body_fitted_fmm/test_remote_far_cache.py
    uv run --no-project python components/tests/unit_test/body_fitted_fmm/test_leaf_only_direct.py

For an authorized GPU verification run, GRANIT_FMM_VERIFY_FAR_CACHE checks
cached references and GRANIT_FMM_VERIFY_LEAF_ONLY checks the specialized routing
invariants. Leave both unset for native timing. The local profile_attempt.py
records these settings and accepts the saved depth_8.yaml and a unique attempt
name; its full commands are preserved in each receipt.

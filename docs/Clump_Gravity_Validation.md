# Member-based clump gravity: staged validation

Campaign: 2026-10-25, starting from `51af3606fac9bc54bb87a2345642a14a615b9e16`.
Evidence is stored in `runs/clump_gravity_campaign`, a symlink to
`/home/user/data/granit-clump-gravity-20261025-b8e1`. Builds, detailed JSON results,
logs, and reference PDFs stay outside Git. This report records measured behavior;
the tests do not establish physical convergence of a DEM experiment.

## Stage 1: small clumps

Gravity is evaluated at each live constituent position. Same-clump source/target
pairs are excluded. The rigid-body outputs are
`F = sum(F_i)` and `tau = sum((x_i - COM) cross F_i)`. Member gravity uses scratch
arrays so contact-force collection cannot add it twice. Both ordinary kicks and
RESPA outer kicks use these outputs; checkpoint restoration adds the reconstructed
gravity wrench once while preserving retained normal-particle force bytes.

The small-clump FMM route uses the existing fixed-depth Karras tree containing
ordinary particles and all small-clump constituents. The sufficient condition is
`2 R < leaf_width`, with a floating-point margin and
`R = max(|x_i - COM| + sphere_radius_i)`. It permits crossing a leaf face, edge,
or corner: all members of that clump remain near neighbors at every level, so
the near P2P exclusion removes all internal gravity. It does not require all
members to occupy one leaf. In this stage, any large clump selects member-direct
gravity for the population. Periodic and 2D registered-clump cases also use that
fallback. Adaptive clump gravity remains unsupported.

Clump multipole overrides are no longer an independent gravity source. Supplied
coefficients must agree with member-derived metadata. Checkpoint comparisons
allow the documented FP32 body-offset quantization. Global projection is applied
only when requested for nonperiodic FMM; DIRECT and periodic fallback are raw.

### Verification and error interpretation

The independent CPU oracle uses long-double pairwise Plummer forces, excludes
same-clump pairs, and sums both force and lever-arm torque. Test coordinates,
masses, and COM are the values actually uploaded to the GPU. The registered
DIRECT route is compared against this same force law. Existing unclumped DIRECT
retains its separate smoothing-length-based gravity law.
Registered DIRECT selects `direct.gravitational_constant`; FMM and its direct
fallback select `unified_fmm.gravitational_constant`. Registered clump Plummer
softening for both methods uses `radius_min * unified_fmm.softening_factor`,
because there is no separate DEM DIRECT softening parameter.

For body `b`, the force denominator is `S_F,b = sum_i |F_i,direct|`; for a normal
particle it is simply its direct force magnitude. The torque moment denominator
is `S_tau,b = sum_i |x_i - COM| |F_i,direct|`. Resultant-relative torque error uses
`|tau_direct|` and is reported separately because cancellation can make it much
larger. Percentiles below cover normal particles and clump resultant forces,
not individual member forces.

* DIRECT and near-only FP64 fixtures agree with the oracle at about `1e-16`;
  the production FP32 P2P build passes the predeclared `1e-6` near tolerance.
* Tests include internal-force exclusion, leaf face/edge/corner crossings,
  translation, rotation, dead rows, normals only, the diameter threshold, periodic
  minimum images, both public gravity APIs, and checkpoint wrench restoration.
  Actual member rows are checked in both sparse and regular grouped-P2P branches.
  Additional small, large-fallback, and periodic fixtures set DIRECT G=2 and
  FMM G=3 to check method-specific configuration dispatch.
* A cancellation-free control uses the ordinary FMM with identical geometry and
  tree/task metadata. It zeros only the target clump's source masses, evaluates
  its member accelerations, and multiplies the original target masses before
  aggregation. This agrees with the new route to FP64 roundoff (and within
  `1e-6` for the production precision), including populated M2L workloads.
* The production solver's uninterrupted versus resumed four-step trajectories
  pass for DIRECT/FMM and gravity update rates 1/2. The largest reported velocity
  difference is `1.45e-11`; the largest angular-velocity difference is `5.92e-11`.
* `config_defaults`, `sph_respa`, `g_table`, and `clump_gravity_contract` pass.
  The contract explicitly separates implementation equivalence from the original
  numerical accuracy target. Failed accuracy targets remain failed in JSON.

The initially chosen populated-FMM accuracy targets (`1e-3` maximum normalized
force and `3e-3` maximum moment-normalized torque) were **not met**. They have not
been relaxed. Diagnosis establishes that the discrepancy is inherited from the
existing order-four FMM operator, rather than the member reduction. Continuation
to stage 2 was authorized after this distinction was established.

Raw FP64, depth 5, projection off, forced FMM thresholds 0/0:

| Members + normals | Clumps | Body force RMS | Worst body force | Worst clump force | Worst clump torque / moment scale |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 0.7154% | 2.4297% | 0.7055% | 0.004409% |
| 512 | 16 | 0.4381% | 2.2700% | 0.8618% | 0.417567% |
| 2048 | 64 | 0.3286% | 8.1567% | 2.9294% | 0.105213% |
| 8192 | 256 | 0.1718% | 2.8679% | 1.4781% | 0.111754% |

With production thresholds 16/64 at 2048 rows and 64 clumps, force RMS is
**0.1134%**, P90 **0.1608%**, P99 **0.3753%**, and maximum **1.3571%**.
The worst moment-normalized torque error is **0.01883%**. That clump spans two
leaves and its resultant-relative torque error is **51.04%**. For the forced-FMM
512-row fixture, the corresponding two-leaf clump has **191.63%** resultant-relative
torque error. These are significant limitations for rotational dynamics, despite
small moment-normalized errors. Its member-to-body reduction still agrees to
roundoff. Piecewise local expansions and cancellation make force accuracy an
insufficient proxy for torque accuracy.

The controlled `tidal` fixture isolates local Taylor truncation: one source is at
a cell center with offset `(2h,h,0)` from the target cell center; a tiny asymmetric
clump lies near the target face at `(0.49h,0.245h,0)`. All its members occupy one
leaf. An independent analytic gradient of the quartic potential predicts
**1.4484078%** force error and **9.8123005%** resultant torque error. Production
gives **1.4484091%** and **9.8123013%**, respectively. Analytic-versus-production
normalized force difference is `1.39e-8`, consistent with the stored float G-table
coefficients even when arithmetic is FP64. Thus percent-level error exists without
leaf splitting or clump multipole approximation.

[Dehnen (2014), sections 3.1 and 4, Figure 4](https://arxiv.org/abs/1405.2255)
shows substantial geometry-dependent scatter in single-interaction FMM errors.
[Springel et al. (2021), sections 2.4 and 3.2, Figure 10](https://arxiv.org/abs/2010.03567)
compares force-error distributions for several orders and opening criteria.
Neither paper promises 1% error from `p=4, theta=0.8`. Their acceptance definitions,
cell centers, particle populations, and error statistics differ from this test;
GADGET-4's FMM geometric criterion uses the sum of cell side lengths. The present
analytic fixture, rather than a numerical comparison of unlike published curves,
identifies the truncation mechanism.

Projection enforces the configured global force/angular-momentum balance, but
redistributes residual force. In the production-precision near fixtures, weak
normal-particle forces can acquire relative errors up to `8.47e-5` (aggregate
normalized error about `4.6e-8`). Raw and projected results are retained separately;
projection is not an accuracy certificate.

### Timings

Device: NVIDIA A100-SXM4-40GB, UUID
`GPU-323456b9-85e7-098d-af7d-a3628c20264b`. CUDA 13.4, architecture 80, release
`-O3 --use_fast_math -lineinfo`. These are warmed, fixed-state public gravity API
host medians, including tree construction, force evaluation, body reduction, and
requested projection; initialization is separate. They are not full-step timings.

The old baseline and new implementation use identical FP64 build modes and
projection enabled, 3 warmups and 20 measurements:

| Rows / clumps | Old clump-p3 path | Small-member path |
|---:|---:|---:|
| 128 / 2 | 3539.998 ms | 0.825 ms |
| 512 / 16 | 5040.998 ms | 1.111 ms |

These are **not accuracy-matched comparisons**: the old p3 path's worst clump
force errors were about `5.27e-5` and `2.57e-4`, versus about `0.00705` and
`0.00846` for the new projected path. The old 2048-row benchmark timed out after
180 seconds without a summary; this censors the whole job, not an individual-call
lower bound. The old 8192-row benchmark was not run.

New member DIRECT versus member FMM, raw FP64, 3 warmups and 10 measurements:

| Rows / clumps | DIRECT | FMM | Direct-reference verification |
|---:|---:|---:|---|
| 2048 / 64 | 1.325 ms | 2.933 ms | Yes |
| 8192 / 256 | 5.172 ms | 5.413 ms | Yes |
| 32768 / 2048 | 17.018 ms | 10.663 ms | No, timing only |

The small fixtures favor direct evaluation. The measured crossover is specific to
these populations, thresholds, build, and GPU.

### Reproduction and evidence

Build targets: `granit_clump_gravity_test` and `granit_dem`. Campaign CMake options:

```text
-DCMAKE_CUDA_ARCHITECTURES=80 -DGRANIT_BUILD_MULTIGPU=OFF -DBUILD_TESTING=ON
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DFMM_M2L_MODE=FP64 -DFMM_P2P_LOCAL_FP32=OFF
-DCMAKE_CXX_FLAGS=-include cstdint
```

The production-precision build uses `TF32_TC` and `FMM_P2P_LOCAL_FP32=ON`.
The existing yaml-cpp checkout is supplied through `FETCHCONTENT_SOURCE_DIR_YAML-CPP`.
The `cstdint` flag and explicit Thrust tuple/standard algorithm includes resolve
compiler compatibility with GCC 15; the baseline uses the same applicable fixes.

```bash
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case all --method both --implementation-check --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case fmm --method bffmm --size 2048 --clumps 64 \
  --direct-threshold-leaf 16 --direct-threshold-internal 64 \
  --clump-diagnostics --implementation-check --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case tidal --method both --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 python3 components/tests/unit_test/body_fitted_fmm/clump_restart.py \
  --executable <build>/bin/granit_dem --output-dir <new-directory>
```

The tidal command exits nonzero because it intentionally retains the failed
original numerical target. Its analytic diagnostic is recorded in the JSON.
`stage1/final-checks-v2/commands.json` records precision-matrix commands and binary
hashes. Other evidence: `diagnostic-512.jsonl`, `diagnostic-default-2048.jsonl`,
`tidal-fp64.jsonl`, `restart-small/results.json`, `ctest-final.log`,
`benchmarks-fp64/`, and `scale-fp64/`. Earlier failed diagnostic attempts are retained
with their original results rather than overwritten as passes.

## Stage 2: aligned large-clump trees

Pending implementation and separate validation after the stage-1 checkpoint.

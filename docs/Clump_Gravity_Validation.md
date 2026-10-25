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

Implemented after stage 1 was validated, benchmarked, and committed as
`6cd592347adc86ad85f492759169c331e18ef8ab`.

### Tree ownership and evaluation

The green Karras tree contains ordinary particles and small-clump members. Each
large clump has its own blue Karras tree. All components use the same current
root origin, root width, depth, and aligned cell centers; their occupied cells
may overlap. Each component computes its own P2M/M2M coefficients.

A union tree supplies the common source topology. Component coefficients are
added into matching `(level, Morton key)` cells at every level, including their
shared coarse parents. There is no second upward pass over already-merged
coefficients. For a blue target, this sum includes only other components; its own
coefficients are never added. This avoids subtracting a large internal field from
a small external field. Green targets use all source components. Near P2P checks
canonical particle indices and excludes same-clump pairs.

Target locals and downward propagation remain private to each component. Source
node/child indices belong to the union tree; target ancestry belongs to the
component tree. The result is scattered once to each canonical member row, then
reduced as `sum(F_i)` and `sum((x_i-COM) cross F_i)`. A block cooperatively reduces
each clump when the aligned path is active. The all-small route continues using
the stage-1 tree and reduction.

The pool caches capacity per GPU and shares one immutable G table across aligned
components. Allocation preflight reports an estimated reserve and fails explicitly
on its memory budget; it does not silently select direct summation. Periodic and
2D registered-clump cases retain direct evaluation. The aligned route requires
fixed-depth, single-GPU, nonperiodic 3D DEM. Adaptive depth remains unsupported.

### Verification and precision limits

The FP64 standard and expanded implementation contracts pass. Tests cover an
isolated large clump, mixed green/blue populations, overlapping large clumps,
shared occupied cells, unequal component sizes, dead rows, partial P2P tiles,
near-only interactions, populated M2L, depths 2 and 12, and both gravity APIs.
A same-process fixture changes the number of large clumps through **2 → 1 → 0 → 2**,
checking both pool reuse and the return from the ordinary dispatcher. Ordinary
all-small, differing-G, periodic, and checkpoint-helper regressions also pass.

For populated far-field fixtures with thresholds 0/0, the cancellation-free
ordinary-FMM control uses the identical union geometry and zeros only the target
clump's source masses. Canonical member forces agree within about `5e-15` in the
expanded FP64 fixtures, and body force/torque reductions agree at roundoff.
Same-clump source coefficients, cell alignment, remapping, actual interaction
routes, and member output counts are checked independently. This verifies the
aligned decomposition against the existing FMM operator, while retaining its
approximation error relative to direct gravity.

Production uninterrupted/resumed trajectories pass for DIRECT/FMM and update
rates 1/2 with large clumps. A separate no-green case (two clumps, eight members,
zero ordinary particles) also passes all four combinations. A traced production
run confirms `mode=aligned_components`. These short restart tests establish
restart consistency, not long-time rotational accuracy.

Both standard CTest contracts pass in FP64 and the default `TF32_TC`/FP32 build.
The expanded default-precision suite has **retained failures** at the original
`1e-6` near/member-comparison tolerance:

* Near-only two-clump/1024-members-each: worst member error `2.65610e-6`;
  clump force and moment-normalized torque errors are `3.63e-7` and `4.30e-7`.
  An ordinary INNER P2P control produces **bit-identical member forces** and the
  same CPU-reference error. Thus this failure comes from inherited FP32 P2P
  arithmetic. Switching only P2P to FP64 reduces the worst member error to `2.96e-15`.
* Far-field canonical-member comparisons reach `1.02e-6`, `1.30e-6`, and
  `1.96e-6` in the expanded many/overlap/shared-cell cases. With TF32 M2L retained
  and only P2P changed to FP64, these differences fall to `2.27e-7`, `2.51e-7`, and
  `1.87e-7`. P2P arithmetic therefore contributes most of the discrepancy, but
  this experiment does not isolate the remaining residual. The diagnostic
  FP64-P2P build fails its stricter `1e-10` comparison gate for those far cases.
* The 256-member reclassification fixture also retains a roughly `1.12e-6`
  member error in its large-clump phases. All corresponding structure checks pass.

No threshold was relaxed and the production precision defaults were unchanged.
Original CPU-reference FMM accuracy targets also remain failed where reported.
The expanded depth-12 mixed fixture reaches **12.52% worst individual member
force error** inside the large clump, while matching the ordinary FMM control
within `1.96e-16`. Its body-force maximum is **2.53%**, body-force RMS is
**0.821%**, and worst clump resultant force error is **0.425%** under the
normalization above. This fixture uses forced-M2L thresholds 0/0. The earlier
report incorrectly labeled the member maximum as a body/member maximum. The
percent-level benchmark errors below are geometry-specific, not a uniform bound
over the validation matrix.

Compute Sanitizer memcheck reports **zero errors** for 33-member, three-clump
overlap tests in both precisions, and FP64 reclassification. Checks use padding
256 and error exit code 99, with no kernel filtering. `--report-api-errors explicit`
retains user-facing CUDA API errors while avoiding the extended internal logging
mode: the frozen stage-1 executable reports three CUB kernel-registration messages
in extended mode, preserved in its original log. These checks cover device memory
access, not every race or synchronization hazard.

### Matched stage-2 benchmarks

Same A100 and FP64 build settings as above; depth 5, thresholds 0/0, raw forces,
3 warmups and 20 samples. Each pair uses the same final harness and geometry.
The stage-1 comparator links the frozen stage-1 production objects, which use
member-direct fallback whenever a large clump is present. Stage 2 uses the
aligned trees. Thus these timings compare different accuracy/cost choices.
The large-clump benchmark places small-clump centers on a bounded grid so growing
their count does not change the intended large/small classification.

| Total rows | Small clumps | Large clumps × members | Stage 1 direct fallback | Stage 2 aligned | Direct reference |
|---:|---:|---:|---:|---:|---|
| 512 | 8 | 1 × 64 | 0.386 ms | 2.972 ms | Yes |
| 2048 | 64 | 1 × 256 | 1.145 ms | 3.772 ms | Yes |
| 8192 | 256 | 1 × 1024 | 4.404 ms | 6.248 ms | Yes |
| 32768 | 2048 | 1 × 1024 | 17.479 ms | 12.782 ms | No, timing only |
| 131072 | 8192 | 1 × 1024 | 181.305 ms | 16.811 ms | No, timing only |
| 32768 | 2048 | 2 × 1024 | 17.495 ms | 14.025 ms | No, timing only |
| 32768 | 2048 | 8 × 256 | 17.200 ms | 21.123 ms | No, timing only |

With one large clump, the measured speedup is **1.37× at 32768 rows** and **10.78×
at 131072 rows**; smaller fixtures favor direct evaluation. Eight large clumps
are slower here. Target branches run sequentially, rebuilding their source sums
and auditing exclusion. For `L` large targets, `K` components, and `U` union nodes,
the always-on full-coefficient exclusion audit costs `O(L (K-1) U)`; this grows
quadratically with large-clump count when `K` grows with `L`. Cross-P2P still
visits own-clump near candidates before filtering their IDs. These costs are
included in the measurements and favor the intended rare-large-clump regime.
The two-versus-eight comparison has equal large-member totals but different
spatial distributions; it is not an isolated measurement of clump-count scaling.
With projection enabled, the 32768-row/one-large pair is 18.151 ms versus 13.470 ms.

For the verified 512/2048/8192-row aligned benchmarks, respectively:

| Rows | Body force RMS | Worst body force | Worst clump force | Worst clump torque / moment scale |
|---:|---:|---:|---:|---:|
| 512 | 0.3998% | 1.8015% | 0.6193% | 0.1035% |
| 2048 | 0.2405% | 1.5107% | 0.4904% | 0.06784% |
| 8192 | 0.2342% | 1.7281% | 0.9154% | 0.2108% |

The corresponding stage-1 direct forces agree with the CPU oracle within
`8e-15`. With production thresholds 16/64 at 2048 rows, aligned FP64 gives
1.1667% worst body force, 0.4238% worst clump force, and 0.03509%
worst moment-normalized torque error. The stage-1 discussion of cancellation and
resultant torque remains applicable; these fixtures do not guarantee accuracy
for arbitrary clump geometry.

All-small paired medians preserve the existing path. Here every clump is small,
but only 12.5% (2048/8192 rows) or 25% (32768 rows) of particles belong to clumps;
the rest are ordinary particles. See the [clump-only follow-up](Clump_Only_Gravity_Benchmark.md)
for a spherical aggregate containing only clump constituents, and for the
corrected depth-12 error distributions.

| Rows / small clumps | Frozen stage 1 | Stage 2 |
|---:|---:|---:|
| 2048 / 64 | 2.295 ms | 2.311 ms |
| 8192 / 256 | 4.197 ms | 4.263 ms |
| 32768 / 2048 | 10.700 ms | 10.684 ms |

The first two pairs have identical reported force/torque errors; the final pair
is timing-only with matching populated tree/task metadata. Observed timing changes
range from -0.15% to +1.56%, with no large regression in these cases.

The reported aligned-pool reserve is an **estimate, not measured peak VRAM**:
about 334 MiB for the 32768-row/one-large case, 400 MiB at 131072 rows, and
819 MiB for the eight-large case. The generic estimator includes per-context
64 MiB headroom, a legacy-tree allowance, and arrays unused by the source-only
union, plus a shared 32 MiB allowance. It excludes other solver storage and uses
heuristic workspace estimates. Preflight adds the previous estimate back to free
memory, so admission is a heuristic screen; checked CUDA allocations determine
whether the request actually fits. An all-small evaluation retains a previously
allocated aligned pool until shutdown. Aligned first calls in these benchmarks take approximately 53–99 ms; warmed timings should not
be substituted for initialization cost.

### Reproduction and evidence

```bash
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case aligned_all --method both --implementation-check --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case aligned_many --members-per-large 1024 --large-clumps 2 \
  --method both --implementation-check --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case aligned_benchmark --size 32768 --small-clumps 2048 \
  --large-clumps 1 --members-per-large 1024 --method bffmm \
  --benchmark --warmup 3 --repeats 20 --no-verify --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 python3 components/tests/unit_test/body_fitted_fmm/clump_restart.py \
  --executable <build>/bin/granit_dem --output-dir <new-directory> --large
```

Evidence under `stage2/`: `stable-full-fp64.jsonl`, `validation-fp64/`,
`validation-default/`, `precision-near-1024.jsonl`, `final-checks/`,
`restart-large/`, `restart-no-green/`, and `benchmarks-fp64/`. Commands and binary
hashes accompany the matrices and benchmarks. `stage1-comparator-v2/` records the
common-harness build and hashes of every frozen production object.
`final-checks/` supersedes the earlier two 2048-row benchmark diagnostics after
the bounded-grid fixture correction. Earlier failures and build diagnostics remain
preserved. `manifest.json` records final source/binary hashes and report status;
`scripts/` contains campaign reproduction drivers.

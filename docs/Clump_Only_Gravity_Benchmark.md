# Clump-only gravity benchmarks and depth-12 diagnosis

Follow-up to the 2026-10-25 staged implementation. Production code is unchanged
from `b9179d28f88da79104cac93a6202b25392ff7073`; this follow-up adds test geometry,
force distributions, an independent analytic replay, and reporting corrections.
Evidence is local under `runs/clump_gravity_campaign/followup-clump-only/`.

## What the previous benchmark measured

The prior term “all-small” meant **every clump was small**, not every particle
belonged to a clump. Its 2048/64 and 8192/256 rows/clumps cases had 12.5% member
rows; the 32768/2048 case had 25%. The rest were ordinary particles, including
two anchors. Those cases tested the mixed-population path and its preservation
between stages 1 and 2. They did not represent a clump-only aggregate.

## New clump-only fixture

Every particle belongs to a four-member rigid clump. There are **zero ordinary,
anchor, dead, or padding rows**. Clump centers occupy the nearest cubic-lattice
sites to the origin, forming a roughly spherical cloud. Site spacing is 0.30.
Each clump has four equal-mass spheres at rotated tetrahedral offsets
`(a,a,a), (a,-a,-a), (-a,a,-a), (-a,-a,a)`, with `a=0.04` and sphere radius 0.06.
Rotations and small variations of total clump mass are deterministic. Adjacent
member spheres within a clump overlap intentionally: center distance is
`2 sqrt(2) a = 0.11314 < 0.12`. The clump envelope radius is
`sqrt(3) a + r = 0.12928`, so distinct clumps have nonoverlapping envelope spheres
at center spacing 0.30. This is an unrelaxed gravity benchmark; no contact,
settling, time integration, or equilibrium criterion is being timed.

The fixture root follows actual member positions. Every run checks zero normal
rows, exactly four members per clump, and every radius-inclusive diameter below
the member-derived leaf width. Current FMM runs additionally require the
matching member tree and nonzero M2L work; DIRECT and old p3 have separate paths. Depth is selected for each size to keep
clumps small; the reported diameter/leaf-width ratios span 0.4675–0.7802.
A fingerprint of uploaded positions, masses, IDs, radii, and lifecycle values
matches across baseline/current, direct/FMM, and precision comparisons.

## Performance

NVIDIA A100-SXM4-40GB, GPU UUID `GPU-323456b9-85e7-098d-af7d-a3628c20264b`,
CUDA 13.4, SM80, release build. Times below are warmed host medians of the public
gravity call, including tree construction, classification, force evaluation,
body reduction, and requested projection. Native event samples are also saved.
Current methods use 3 warmups and 20 samples. Each FP64 direct/FMM pair runs in
the same process on identical fixed state. FMM uses production thresholds 16/64
and projection enabled; DIRECT retains its exact, unprojected member force law.
These methods have different approximation errors, reported below.

| Clumps | Constituent spheres | Depth | Member DIRECT (ms) | Member FMM (ms) | DIRECT/FMM | Full direct oracle |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 128 | 2 | 0.116 | 0.697 | 0.17× | Yes |
| 128 | 512 | 2 | 0.285 | 0.629 | 0.45× | Yes |
| 512 | 2,048 | 3 | 1.033 | 1.143 | 0.90× | Yes |
| 2,048 | 8,192 | 3 | 5.203 | 1.630 | 3.19× | Yes |
| 8,192 | 32,768 | 4 | 17.520 | 3.563 | 4.92× | Timing only |
| 32,768 | 131,072 | 5 | 197.564 | 20.772 | 9.51× | Timing only |

At 8192 spheres, member FMM is 3.19× faster than member DIRECT; at 131072
spheres, the measured ratio is 9.51×. Small cases favor DIRECT. These are matched
comparisons at each size; the changing depths make this a workload family,
rather than a fixed-depth asymptotic scaling study. The largest two cases are
timing-only, with population/classification/actual-M2L checks.

The identical new harness also links against the unchanged pre-stage-1 production
objects (`51af3606fac9bc54bb87a2345642a14a615b9e16`). Its old clump-p3 cross pass
uses all clump pairs; clump virtual bodies carry zero mass in the ordinary FMM.
With projection enabled in both implementations:

| Clumps / spheres | Original clump-p3 | Current member FMM |
|---:|---:|---:|
| 32 / 128 | 5905.124 ms | 0.697 ms |
| 128 / 512 | 24996.101 ms | 0.629 ms |

The old path uses 1 warmup and 3 samples. The 128-clump process took more than
90 seconds including initialization/reference/API checks, so the predeclared
budget rule skipped its 512-clump extension. These large differences characterize
the existing implementations, not an inherent performance ratio between p3 and
p4 methods. Their projected body-force RMS errors, under the original
`sum |member reference force|` normalization, are respectively 0.3588%/0.1800%
for the old path and 0.2318%/0.1032% for the new path.

Supplementary default `TF32_TC`/FP32 timings (DIRECT uses the same member-direct
law; the default-build direct and FMM runs are separate processes):

| Clumps | Member DIRECT (ms) | Member FMM (ms) | DIRECT/FMM |
|---:|---:|---:|---:|
| 512 | 1.328 | 2.341 | 0.57× |
| 2,048 | 5.210 | 3.288 | 1.58× |
| 8,192 | 17.510 | 5.064 | 3.46× |
| 32,768 | 197.480 | 8.028 | 24.60× |

The default precision is slower than FP64 in the small fixtures and faster in
the largest fixture in this campaign. No general precision-mode speed ordering
is inferred from these measurements.

## Force distributions in the clump-only fixture

The independent CPU reference sums long-double Plummer pair forces at the actual
uploaded coordinates, skipping `i==j` and pairs with the same positive clump ID.
It then computes clump force and torque as `sum f_i` and `sum (r_i cross f_i)`.
The full direct-oracle range is 128–8192 spheres. DIRECT body-force errors are at
roundoff. FMM still fails the original strict maximum-error gate where recorded;
those failures have not been renamed as passes.

For the distribution tables here, use one conventional normalization:
`e = |F - F_direct| / max(|F_direct|, 1e-28)`, with an arithmetic mean and
`RMS = sqrt(mean(e^2))`. For members, `F` is the external force on that member;
for clump resultants, `F` is the summed body force. The raw FMM table below has
projection disabled to expose force accuracy before the global correction.

| Clumps | Member mean | Member RMS | Member max | Clump resultant mean | Clump resultant RMS | Clump resultant max |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.2089% | 0.6062% | 4.8703% | 0.1076% | 0.2377% | 1.2312% |
| 128 | 0.0959% | 0.2519% | 2.8842% | 0.0679% | 0.1022% | 0.8604% |
| 512 | 0.1241% | 0.6510% | 19.3631% | 0.0850% | 0.1518% | 2.6138% |
| 2,048 | 0.0759% | 0.1791% | 7.1752% | 0.0630% | 0.0874% | 0.5910% |

The 512-clump member maximum is a near-cancellation case at the aggregate center:
member 1 has `|F_direct|=18.0872`, while the sum of magnitudes of its external pair
forces is 1790.2656 (conditioning ratio 98.98). Its absolute force error is 3.50224,
or 0.1956% of that pair-force scale, but 19.3631% of the small resultant. An
independent analytic p4 replay predicts 19.3630953%, versus measured 19.3630990%.
All 6144 ordered intra-clump pairs in this fixture independently classify as
near/P2P; the external-only analytic sum matches production within `4.14e-8`
relative to the resultant. This identifies truncation amplified by cancellation,
with self-interaction excluded.

The torque results are retained as well: aggregate resultant-relative torque
error is 2.04–5.33% across these four raw fixtures, and the maximum
moment-normalized clump torque error reaches 4.41%. Force-error statistics alone
do not validate rotational dynamics.

For another common RMS definition, global relative L2 error
`sqrt(sum |delta F|^2 / sum |F_direct|^2)` is 0.09806% for the 512-clump member
field and 0.07820% for its clump resultants. This differs from averaging each
member's relative error equally.

## Correcting and explaining the depth-12 result

The original fixture has 118 live rows: 50 ordinary particles (including two
finite-mass anchors), one 4-member small clump, and one 64-member large clump.
It is a deliberately sparse correctness fixture, not a rubble-pile benchmark.
It uses FP64, depth 12, `theta=0.8`, and **forced-M2L thresholds 0/0**.

The reported **12.52% is a member-force maximum**, at canonical row 76 inside the
large clump. The earlier summary incorrectly combined this with body-force
statistics. Under the conventional normalization defined above:

| Population | Count | Mean | RMS | Maximum |
|---|---:|---:|---:|---:|
| All particle rows | 118 | 0.9586% | 1.8065% | 12.5226% |
| Large-clump members | 64 | 1.2633% | 2.3386% | 12.5226% |
| Ordinary particles | 50 | 0.6256% | 0.8343% | 2.5298% |
| Physical bodies (normals + clump resultants) | 52 | 0.6361% | 0.8466% | 2.5298% |
| Clump resultants only | 2 | 0.8990% | 1.1100% | 1.5500% |

The older 0.821% body RMS and 0.425% maximum clump force use the different
`sum |member reference force|` denominator for clumps. They are valid for that
scale but should not be mixed with conventional resultant-relative statistics.
The worst physical-body force belongs to the positive anchor at `(16,16,16)`;
the member outlier belongs to the large clump.

Self-interaction is excluded by three independent checks: the direct oracle
omits same-clump pairs; blue targets omit their entire own component from source
coefficients; and isolated large-clump forces are exactly zero. A source-mass-zero
ordinary-FMM control also matches the constituent field to about `2e-16`.

A separate CPU implementation reconstructs accepted source/target cells from the
stencil and differentiates the bivariate Newtonian potential through source
order 4 and local-potential order 4. It uses no production G matrices or basis
functions. For the actual uploaded row 76 it predicts **12.5225682%**, compared
with **12.5225688%** from the GPU; the force-vector difference is `6.23e-9` relative
to the exact external force. Direct summation in that replay matches the saved
long-double reference within `1.52e-16`.

The dominant erroneous contribution is the *other, small clump* (rows 50–53),
accepted at target level 4/source level 3. Its cell circumsphere ratio is 0.7924,
just below theta 0.8. Source expansion truncation dominates: keeping source p4
alone predicts 10.78% error, local-potential p4 alone 2.85%, and both 12.52%.
Cancellation is mild here (pair-force scale/resultant = 1.37). The same physical
geometry at depths 5, 8, and 12 retains the same outlier because that interaction
was already accepted at a coarse level. Zero softening leaves the result
unchanged to about `1e-13` in relative-error units.

With production thresholds **16/64**, this particular 118-row fixture converts
all interactions to P2P: zero M2L routes and maximum member error `1.70e-15`.
Its command returns nonzero solely because the original fixture requires a
populated far workload; the failed fixture condition is retained in JSON.
The new clump-only performance cases have nonzero M2L at production thresholds
and retain the error tails shown above.

## Reproduction and validation

```bash
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case clump_only --clumps 2048 --size 8192 --depth 3 --method both \
  --direct-threshold-leaf 16 --direct-threshold-internal 64 \
  --project --benchmark --warmup 3 --repeats 20 --json <new-results.jsonl>
CUDA_VISIBLE_DEVICES=1 <build>/bin/granit_clump_gravity_test \
  --case aligned_mixed --members-per-large 64 --depth 12 --geometry-depth 12 \
  --method bffmm --implementation-check --force-diagnostics --worst-body \
  --json <new-depth12.jsonl>
python3 components/tests/unit_test/body_fitted_fmm/clump_pair_taylor.py \
  --input <new-depth12.jsonl> --target 76 --output <new-replay.json>
```

`--force-diagnostics` is restricted to verified raw runs. It saves actual aligned
member forces before controls; all-small member forces come from an explicitly
labeled repeat of the same filtered dispatcher, with matching tree/workload
metadata. DIRECT exposes body outputs but no public registered member scratch,
so its member actual-force entries are null. The analytic replay supports
forced-0/0 aligned fields and ordinary-tree occupancy cutoffs; it explicitly
rejects aligned nonzero-cutoff replay without component occupancy information.

Both standard CTest contracts pass again in FP64 and default precision. Independent
review recomputes distributions from force vectors and confirms the geometry,
normalizations, self-exclusion, and analytic attribution. Original failed accuracy
gates remain in the records. No production gravity code was modified.

Evidence folders: `diagnostics/`, `frozen-depth12/`, `analytic-depth12/`,
`benchmarks/`, `final-checks/`, and `comparator-baseline/`. Each run group records
commands and binary hashes. `plan.json` records the experiment choices;
`manifest.json` pins final source/binary identities and summaries. Reproduction
drivers are retained in `scripts/`. Build products and force dumps remain in the
linked data directory, outside Git.

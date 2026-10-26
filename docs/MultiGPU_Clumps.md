# Distributed small-clump DEM

`granit_multigpu` supports DEM runs containing ordinary spheres, rigid
clumps, or only rigid-clump members. Members retain spatial ownership and may
span GPUs or migrate independently. Small clumps support fixed or adaptive
common-depth 3D BFFMM gravity, isolated or shear-box boundaries, and exact
checkpoint/restart within the contract below.

## Configuration contract

Use the existing unified-v3 member input and contiguous positive clump IDs
(`1..K`), with corresponding `clumps.instances`. Ordinary live spheres use ID
zero. Select LBVH neighbors and Morton ownership.
With self-gravity enabled, set:

```yaml
accelerations:
  self_gravity:
    enabled: true
    method: unified_fmm
    gravity_update_rate: 1
    gravity_macro_dt: 0
    unified_fmm:
      tree_dimension: 3
      tree_depth: 4
      adaptive_depth: {enabled: false}
      theta_far: 0.8
      direct_threshold_leaf: 0
      direct_threshold_internal: 0
      enable_global_projection: false
```

Check the maintained test configurations for the complete schema. The example
selects fixed depth four. Adaptive runs may use `tree_depth: auto` or an integer
seed with an enabled `adaptive_depth` block; configured depths lie in `[2,20]`
and the effective source-tree depth must also be at most 20.
Every body must satisfy `D < h_leaf * (1 - 1e-6)`, where `D = 2R` is its full
bounding diameter, including member radii. `h_leaf` uses the actual gravity
source root and effective launch depth. The runner checks the current geometry
before every gravity evaluation, including bootstrap, trials, root changes,
and restart. Members may cross adjacent leaf cells and GPU boundaries.

Adaptive clump runs also enforce at most **128 actual global source particles
per leaf**, counting all live members and ordinary spheres. This is a maximum
over globally combined leaf counts, not an average or a per-rank limit. In a
shear box it includes every logical finite gravity-image replica, including
the Y images evaluated through virtual translations. A GPU Morton-key census
combines split leaves across ranks without gathering all particles to the host.
The occupancy lower depth bound is intersected with the diameter upper bound,
configured range, and common memory ceiling. An empty intersection terminates
with a descriptive error. The controller clamps both selected and trial depths
to this interval and invalidates stale comparisons when it changes.

The direct thresholds compare source-count × target-count against the cost
threshold for converting otherwise-far interactions to direct particle work.
Zero disables that conversion; required near-field P2P remains active. The
distributed clump path requires explicit zero values, including one GPU
through `granit_multigpu`.

Unsupported configurations fail explicitly: large-clump or 2D gravity,
gravity subcycling, global gravity projection, non-Morton ownership,
DIRECT/CLL neighbor search, ordinary periodic boundaries outside a shear box,
YORP, global damping, acceleration limiting, and rotating/inertial frames
outside the supported shear-box formulation.
Contact-only clumps do not require the gravity leaf-width condition.

Shear clumps require `2D < min(Lx,Ly) * (1 - 1e-6)` for a unique internal
periodic image, including contact-only runs. Their FMM gravity uses the
configured finite sheared image patch and its effective depth; it is not an
infinite periodic/Ewald sum. The leaf condition uses this image-envelope root,
not the physical XY ownership grid. See [shear-box configuration](shear_box.md).

## Runtime behavior

Each body has one authoritative state on the GPU owning its initial center
of mass. Owners integrate translation, quaternion orientation, and body-frame
angular velocity with the full inertia tensor. Members receive the rigid pose
and velocity before force evaluation and after the final kick. Ordinary
spheres continue through the existing particle integrator.

In a shear box, body centers follow the exact Hill drift. Angular integration
includes the full-tensor gyroscopic and Hill torques, including the intrinsic
second moments of member spheres. Rigid member reconstruction preserves body
offsets across sliding boundaries and tracks signed integer winding counters.
Force reduction uses the unwrapped rigid moment arm.

Each participant GPU reduces its owned members' force and torque, including
intrinsic member torque and the moment arm about the body center. Remote
participants send compact partial wrenches to the owner; owners send compact
poses back. Ghost members never contribute a second wrench. Bodies entirely
on their owner need no inter-GPU body message. Routing is rebuilt when migration
changes a rank's participant-body set. This version uses stream completion and
host barriers to protect message lifetimes.

In clump scenes, both directed contact passes evaluate the pair in stable
global-ID order and select the corresponding force and torque. Each member
still owns its migrated friction history. Reversing the evaluation converts
the three vector histories to and from canonical signs. This prevents FP32
rounding differences from independently recomputing the opposite contact arm.

Gravity uses the existing common spatial FMM tree. Local and remote P2P omit
pairs only when both particles share the same **positive** clump ID. In shear
gravity, this exclusion selects the body's unique internal image; other finite
replicas of that body remain gravitational sources. The small-body condition
keeps all intrabody pairs within near routes; distant
interbody interactions retain M2L. Remote near-particle traffic adds a separate
four-byte clump ID per source when clumps are present. Unclumped scenes keep
their original particle wire format and avoid this extra payload.

Authoritative body storage is distributed. Participant poses and wrenches
scale with body/rank participation; each GPU also stores a dense four-byte
body-ID lookup. Owners stay fixed during a run, so sustained bulk motion can
increase remote body traffic. Route rebuilding still reads member
IDs to the host on migration events. These are explicit optimization targets.

Outputs preserve member IDs and body offsets and write the established v2
`clump_states_step_*.bin` body snapshots. Set `GRANIT_CLUMP_DIAGNOSTICS=1` for
participant counts, member migration, and remote interbody contact evidence.
Leave diagnostics disabled for performance measurements.
The `[clumps-depth]` record reports the base/effective depth, evaluation time,
source root, full diameter, and leaf width; adaptive records also report the
global maximum leaf population, physical/stored/logical source counts, and
feasible depth interval. Enable diagnostics to emit it at every evaluation.

## Checkpoint and restart

Distributed clump checkpoints require the unified-v3 checkpoint and its matching
`.distributed_clumps` sidecar. The sidecar preserves authoritative body state,
retained forces and torques, winding, exact member force/torque rows, wall
histories, and ownership metadata; contact histories follow global particle IDs.
It is bound to the checkpoint bytes and clump metadata. Missing, corrupt, or
mismatched sidecars fail explicitly. An ordinary member output snapshot alone
does not contain this exact continuation state.

Fixed-rank restart restores ownership cuts and body owners. Fixed-depth restart
may change the GPU count: particles are repartitioned and body owners are
reassigned by current center of mass. Adaptive restart requires the same rank
count, backend, and compatible configuration, and also requires the common
`.adaptive_fmm_depth` sidecar containing controller and ownership state. Keep
both sidecars beside the checkpoint. Shear restart preserves phase and winding;
`shear_box.time_origin` must remain fixed.

## Reproducible checks

The standard-library drivers in
[`dem_multigpu_clumps`](../components/tests/integration_test/dem_multigpu_clumps/README.md)
generate controlled fixtures, compare body/member state, check independent
near-field gravity, exercise unsupported settings, and collect native full-step
timing with exact build/input identities. Large artifacts belong under
`~/data/`, linked into `runs/`.

Standalone single-GPU and common-root distributed FMM may choose different
roots and therefore different approximations. Mechanical SG/distributed
comparisons disable gravity; distributed rank scaling uses the distributed
one-GPU result as its same-root reference. The near-field fixture additionally
checks direct member and body force/torque, including same-clump exclusion.

The million-member throughput fixture has 250,000 four-member bodies. Its
paired sphere fixture uses identical geometry and materials, with rigid
membership removed. This measures workload cost; it changes internal gravity
exclusions and rigid constraints. Sparse and intentionally preloaded contact
fixtures are reported separately. Finite outputs and a short timing window do
not establish long-time physical convergence or general FMM force accuracy.

## Initial verification, 25–26 October 2026 (pre-extension build)

The evidence in this section and the following scaling section predates the
adaptive-depth, checkpoint/restart, and shear-clump extensions. It describes the
initial fixed-depth isolated implementation and its frozen binaries.

The source baseline is `7b8e42bc7ad92375cc3e117ec56b3876271aef91`, plus the
distributed-clump changes. CUDA 13.4 builds target A100 (`sm_80`), using both
FP64 M2L/P2P and the default TF32 M2L/FP32 local-P2P modes. Raw evidence is
linked at `runs/multigpu_clumps_implementation` and stored under
`~/data/granit-multigpu-clumps-20261025-a94e`.

The first million-member attempt exposed a pre-existing startup loop with
`N * max(clump_id)` comparisons and timed out before GPU stepping. Stable
ordering now takes O(N log N), or O(N) for already ordered inputs, while using
O(N) index storage. Particle-field and contact-history permutation are
unchanged. A host test compares the actual helper against the old algorithm
on 3,000 randomized cases, checks extreme IDs and lifecycle handling, and
orders one million rows. It is registered as CTest `particle_clump_order`.
The final builds also repeat four-GPU mixed-body mechanics and independent
FP64 gravity after this startup-only change.

| Check | Result |
| --- | --- |
| Mechanical SG parity, five fixtures × 1/2/4 GPUs | 15/15 pass |
| Unsupported controls, including oversized bodies on 2/4 GPUs | 20/20 normal error exits with expected messages |
| Integrated gravity, frictional contact, and actual member migration | 1/2/4 GPU parity passes; 2/4 GPU remote contact and migration confirmed |
| Populated common-root FMM, default precision | 1/2/4 GPU parity passes |
| Independent near-gravity, FP64, 2/4 GPUs | Member force and body force/torque pass |
| Independent near-gravity, default precision, 2/4 GPUs | Force passes; relative body-torque gate fails |
| Harness CPU checks | 7/7 pass |
| Two-GPU integrated-contact padded memcheck, API reporting disabled | 0 memory-access errors; trajectory and migration checks pass |

The default-precision symmetric near-field fixture has body-torque relative
L2 error about `2.38e-4`, exceeding the unchanged `1e-4` gate. Absolute L2
error is about `7.8e-21`, against a reference norm `3.26e-17`; torque
cancellation factors are approximately 8,600–9,500. Member-force error is
about `1.1e-7`. This is a retained precision limitation, not a passing torque
test. The campaign summary therefore remains `pass: false` for the full
accuracy matrix. FP64 passes the same fixture.

The API-inclusive sanitizer run completed the physical trajectory but returned
19 API diagnostics: 17 duplicate CUB kernel registrations and two handled
`cudaErrorPeerAccessAlreadyEnabled` statuses. It reported no memory-access
violations. The separate memory-only run explicitly disables API reporting;
neither result is presented as an entirely clean API-inclusive sanitizer run.
See NVIDIA's [API reporting modes](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#reporting-modes)
for this distinction. These checks do not cover global-memory races or all
possible workloads.

## Initial scaling results (pre-extension build)

All 27 final native runs passed their endpoint/readiness checks: sparse clumps,
matched sparse independent spheres, and dense-contact clumps at 10,000,
100,000, and 1,000,000 constituents on 1/2/4 GPUs. The million-member clump
population contains 250,000 four-member bodies and zero ordinary spheres.

For the sparse million-member case, fixed depth six, default precision:

| GPUs | Clumps, ms/step | Independent spheres, ms/step | Clump overhead | Clump speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 69.745 | 61.403 | 13.6% | 1.00× |
| 2 | 31.805 | 28.679 | 10.9% | 2.19× |
| 4 | 22.733 | 20.535 | 10.7% | 3.07× |

Each measurement is one native 16-step compute window after eight warmup
steps, excluding initialization and output. The endpoint is 0.00024 seconds.
All particle fields, IDs and memberships and all body fields/quaternions are
checked; 1,024 bodies are sampled for rigid geometry, velocity and spin.
Sampled maximum device usage was 5.42 GiB with one GPU and 3.51 GiB on the most
loaded device with four GPUs, including initialization. These are sampled total
device values, not exact allocator peaks.

The machine has four A100-SXM4-40GB GPUs: 0–1 and 2–3 have NV12 links, with
cross-pair SYS traffic using host staging. The dense million-member contact
case also passes on 1/2/4 GPUs and confirms remote interbody overlap pairs;
it uses depth five and enabled diagnostics, so its timing is a separate health
measurement. Actual member migration is established by the small integrated
fixture, not by these short large runs.

The final campaign is `benchmark-default-04`. Its `scaling-summary.json` and
the campaign-root `REPORT.md` preserve the complete table, launch identities,
memory samples, failed preliminary attempts, and accuracy limitations. The
full-step comparisons are workload-specific samples, not statistical speedup
guarantees or physical equivalence between spheres and rigid bodies.

A separate paired Nsight profile at 100,000 constituents on four GPUs found
similar domain and search costs. Maximum-rank mean host ranges per measured
step were 0.384/0.415 ms for migration, 0.416/0.456 ms for halo exchange, and
0.423/0.422 ms for neighbor search (clumps/spheres). Recorded peer-copy bytes
increased by 1.1%. The additional rigid-body kernels used about 0.087 ms of
visible GPU time per device per step; launch calls rose from 94.25 to 107.75
and stream-synchronization calls from 32.81 to 41.25 per rank per step.
These observations identify body work and synchronization as optimization
targets without showing a large domain/contact-search penalty in this fixture.
The profile has no active contact-force pairs, uses instrumentation, and is not
a decomposition of the million-member native wall time. Host ranges, API wait
durations and overlapping device work must not be added together. Raw traces,
SQLite, CSV reports and window definitions are in `profile-default-100k4`.

## Restart, adaptive depth, and shear extensions, 26 October 2026

Extension evidence is in `runs/multigpu_clumps_implementation/extensions-20261026`.
The extension report records build identities separately from the initial
fixed-depth campaign above. FP64 tests cover 1/2/4 GPUs; representative default
precision combined tests cover 2/4 GPUs.

| Check | Result |
| --- | --- |
| Isolated contact/gravity restart, including exact active friction histories, zero-step restoration, and missing/corrupt sidecars | 28/28 pass |
| Exact global adaptive occupancy, fixed-depth controls, and infeasible occupancy/diameter bounds | 8/8 pass |
| Remaining unsupported configurations on the final build | 17/17 explicit error exits with expected diagnostics |
| Combined shear contact + adaptive depth + restart, isolated root growth, fixed-depth 2-to-4 GPU continuation, and adaptive rank-change rejection | FP64 44/44; default precision 35/35 pass |
| Independent finite-image gravity for a seam-split body, including member force and body force/torque | FP64 1/2/4 GPUs and default precision 4 GPUs pass |
| Full 40-step two-GPU shear/contact/adaptive padded memcheck | 0 memory-access errors; all 41 output/occupancy audits pass; API reporting disabled |
| Focused host tests and Python harness tests | 5/5 and 21/21 pass |

The seam-split gravity control retains the finite nine-image patch and far M2L
interactions. Its maximum relative L2 errors across tested modes are about
`8.98e-6` for member force, `5.00e-6` for body force, and `4.85e-5` for body
torque, within the unchanged `1e-4` gates. Eight directed internal member
pairs use a noncentral periodic copy. Contact/restart fixtures separately
exercise actual winding changes and nonzero friction history.

One far-field accuracy case remains **failed**. With all 544 retained
target/source-image pairs outside the near stencil, FP64 member-force relative
L2 error is `1.56e-2`, body-force error `1.03e-2`, and body-torque error `1.58`.
A source-identical unclumped control, after subtracting exact internal-body
forces, matches the clump result to `1.97e-14` relative L2 and reproduces the
same far-error pattern. This isolates the error to the common FMM path in
that fixture, consistent with its coarse fourth-order approximation. It does
not pass the force/torque accuracy gates. Neither the occupancy constraint nor
FP64 arithmetic guarantees adequate FMM truncation accuracy.

The final default build also passes four bounded all-clump readiness runs on
four A100-SXM4-40GB GPUs. Each uses 24 steps with eight warmup steps and a
16-step native timing window, with diagnostics disabled after the initial
depth record:

| Constituents | Gravity geometry | Actual depth | Initial maximum sources/leaf | ms/step |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | Isolated | 4 | 32 | 7.175 |
| 100,000 | Shear, nine finite images | 6 | 63 | 12.597 |
| 1,000,000 | Isolated | 5 | 32 | 13.892 |
| 1,000,000 | Shear, nine finite images | 7 | 63 | 37.471 |

The million-member runs contain 250,000 rigid bodies and no ordinary spheres.
The shear census counts nine million logical sources; only three million
X-column sources are stored by the FMM. Every stored particle/body field is
checked for finiteness, IDs and memberships are checked, every body quaternion
is checked, and 1,024 bodies are sampled for rigid pose/velocity/spin. The
largest sampled total device usage is about 3.51 GiB. These short runs establish
execution and rigid-state readiness, not large-case force accuracy or long-time
physical convergence. Different selected depths and shear image populations
prevent treating this table as a speedup comparison against the earlier fixed
depth-six campaign.

The recorded final source edit changes only `GRANIT_CLUMP_DIAGNOSTICS=0`
logging relative to the accepted combined/accuracy binaries. Final-build
memcheck, four-GPU independent gravity checks in both precision modes, and
the four large runs cover the final binaries. The pre-edit extension source
was not separately archived; the extension report identifies this lineage
limit. Earlier failed attempts, the far-field accuracy failure, tested binaries,
and reproducible launch records are retained.

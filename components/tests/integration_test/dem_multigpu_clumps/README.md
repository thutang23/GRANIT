# Small-clump distributed DEM validation

`validate.py` uses only Python's standard library. It generates DEM unified-v3
inputs and JSON configurations (valid YAML), invokes existing executables with
bounded timeouts, and checks physical output. It does not build GRANIT.

Store generated campaigns under `~/data/` and link that directory into the
checkout when convenient. Each generate/run/reject destination must be new;
accepted evidence is never overwritten. Commands below use example paths;
replace the executable paths with the exact build being investigated.

```bash
python3 -m unittest discover -s components/tests/integration_test/dem_multigpu_clumps -p test_validation.py -v

python3 components/tests/integration_test/dem_multigpu_clumps/validate.py run \
  --single /absolute/path/granit_dem --multi /absolute/path/granit_multigpu \
  --work-dir "$HOME/data/clumps-validation-NEW" --ranks 1 2 4 --timeout 90
```

The matrix uses the same material, input bytes, timestep and physics for each
case, fixed 3D FMM depth 4, `theta_far: 0.8`, both direct thresholds **0**, no
global projection, gravity update rate 1, isolated nonperiodic boundaries,
and no adaptive depth, YORP or checkpoints. Mechanical fixtures disable gravity
for controlled SG comparison; near-gravity and FMM fixtures enable it.

| Fixture | Intended evidence |
| --- | --- |
| `all-clump` | 40 members, ten bodies, exactly zero normal particles; spin, small bodies across leaf cuts |
| `contact-migration` | Same all-clump population, two distinct bodies in external frictional contact; spin/torque, common translation across x/z ownership cuts |
| `mixed` | Contact fixture plus three normal spheres |
| `empty-rank` | Four coincident members of one spinning rigid body, intentional internal overlap, contact-only; a single occupied leaf leaves empty particle ranks at 2/4 GPUs |
| `near-gravity` | Two nearby compact four-member bodies, two tiny-mass remote normal anchors defining the root; independent Newton body force/torque reference |
| `populated-fmm` | 216 bodies/864 members on a 6³ lattice; populated tree, fixed-state initialization and short evolution |
| `inferred-spin` | Contact/migration geometry and rigid rotational member velocities, with the input OMEGA field omitted; checks inferred spin bootstrap against SG |
| `integrated-contact` | Contact/migration plus FMM gravity; distributed one-GPU reference controls the common root |
| `oversized` | Deliberately violates the strict small-body leaf-width bound |

Contact tetrahedra have radius .01 and offsets with components ±.008; their
internal pair distances are .0226274, so no internal overlap is converted to a
spurious contact in the optional sphere baseline. The two contacting body
centers are separated by .026; nearest external member distance is .018868,
giving intentional .001132 overlap. Static/rolling/twisting friction are
.4/.03/.02. Tangential motion and opposite initial spin make nonzero body
torque a required activation gate. The near-gravity anchors use radius 1e-5
and a separate low stiffness material to preserve the explicit 1e-5 timestep;
their force is negligible, but their stored mass remains positive.
The contacting bodies start at z=-.003 with common velocity (20, 0, 40),
in addition to opposite tangential y velocities. Over .0004 s, the negative-z
members cross zero while relative contact motion remains unchanged.

Each saved member frame must have the expected count, all canonical IDs exactly
once, unchanged positive clump membership, finite state, positive radius/mass,
and live lifecycle state. Every intrabody pair distance must stay within 2e-8
of its input value, and member velocity must satisfy `v = V + omega × (x-COM)`
within 2e-8. Every completed-step body snapshot must contain all body IDs,
finite position/velocity/spin/inertia/multipoles/wrench, positive mass, and a
unit quaternion within 1e-10. The established SG runner omits a step-zero body
file; the initial member frame is still required and verified.

SG versus distributed member-state parity uses atol 2e-8 and rtol 2e-6; member
omega uses atol 2e-7 and rtol 2e-5. Body state uses atol 2e-8/rtol 2e-6; body
force/torque uses atol 1e-12/rtol 2e-5. Reports retain maximum tolerance
fractions. Raw member force fields are not compared because SG assigns clump
gravity to body wrench while the distributed path reduces physical member
forces. The rate-1 gravity cache sidecar is not a physical oracle.

The independent near-gravity reference sums all Newton interactions except a
particle with itself and pairs sharing the same **positive** clump ID, then
reduces member forces and moment arms to body force and torque. At the first
available body snapshot, force and torque each require relative L2 error <=
1e-4 and maximum error <=5e-4 of the corresponding reference RMS. The distributed
runner also requires its physical per-member FORCE to meet those gates against
the independent member reference. This catches self-clump contributions even
when their net body force and torque cancel. Both nearby clumps straddle y/z
leaf boundaries. No force or torque projection is applied.

For `populated-fmm` and `integrated-contact`, the parity reference is **distributed one GPU**: its common
Morton root is shared with distributed two/four GPUs. Standalone SG can use a
different dynamic root and therefore a different accepted FMM approximation.
The SG trajectory remains separately verified, without claiming its FMM tree
is identical. Mechanical SG comparisons use gravity-free `all-clump`,
`contact-migration`, `mixed`, `inferred-spin`, and `empty-rank` fixtures.
Keep `--ranks 1 2 4` in that order.

The runner enables `GRANIT_CLUMP_DIAGNOSTICS=1`. Verification preserves
`[clumps-routes]`, `[clumps-migration]` and `[clumps-contact]` measurements.
`x=0` crossing establishes geometric motion; **positive actual member-count
changes and remote interclump overlap diagnostics** establish that migration
and remote contact were exercised. These are required acceptance gates for
`integrated-contact`. Gravity-free mechanical fixtures use a different spatial
partition, so their acceptance requires SG numerical parity while their actual
cross-rank coverage is recorded separately. An empty rank requires a rank with no owned
particles. Reports leave absent diagnostics explicit. Contact history is not
serialized by normal outputs; trajectory agreement checks its dynamical
effect. For a claim specifically about tangential-history migration, require
positive contact and migration evidence in the same run.

`run_campaign.py` runs separate mechanical, gravity and rejection stages with
already-built FP64/default binaries and accepted SG reference directories.
`--reuse-runs` verifies matching binary/config identities and reuses completed
raw outputs, writing new receipts without overwriting prior evidence. Gravity
accuracy failures are retained while the remaining accuracy cases finish;
the stage returns nonzero and its report remains `pass: false`.

The direct oracle reports absolute torque error, reference norms and the
cancellation condition `sum(norm(r × f)) / norm(sum(r × f))` per body.
These measurements explain a failed relative gate without converting it to a
pass. Keep the original symmetric fixture and its original gates when comparing
precision modes.

Rejected configurations are tested with a timeout, a normal positive nonzero exit, and an
expected diagnostic substring. A timeout is a failure, never a clean rejection.
Termination by a signal is also a failed clean-exit check.
The generic JSON overlay supports checking every unsupported control without
editing the generator:

```bash
python3 components/tests/integration_test/dem_multigpu_clumps/validate.py reject \
  --binary /absolute/path/granit_multigpu \
  --work-dir "$HOME/data/clumps-reject-oversized-NEW" \
  --fixture oversized --ranks 4 --expect 'distributed small-clump diameter'

python3 components/tests/integration_test/dem_multigpu_clumps/validate.py reject \
  --binary /absolute/path/granit_multigpu \
  --work-dir "$HOME/data/clumps-reject-projection-NEW" --ranks 4 \
  --overlay '{"accelerations":{"self_gravity":{"unified_fmm":{"enable_global_projection":true}}}}' \
  --expect 'Distributed DEM clumps:'
```

Also reject gravity rate >1, nonzero direct thresholds, 2D trees, non-Morton
ownership, ordinary periodic boundaries outside a shear box, and unsupported
YORP/frame controls, using the specific guard text where practical. Adaptive
depth, shear clumps, and checkpoint/restart are supported by the extension
contract and tested below; their rejection in the initial campaign is
historical evidence rather than a current requirement.
Retain each `launch.json`, configuration, log and `rejection.json`.

Generation and artifact checking can run without a GPU:

```bash
python3 components/tests/integration_test/dem_multigpu_clumps/validate.py generate \
  --fixture throughput --clumps 250000 --ranks 4 --steps 4 \
  --output-dir "$HOME/data/clumps-1M-generate-only-NEW"

python3 components/tests/integration_test/dem_multigpu_clumps/validate.py verify \
  --work-dir /absolute/path/near-gravity-gpu2 --direct-gravity
```

The throughput generator streams arrays into a 1M-member binary and writes
250k explicit contiguous instance IDs. Add `--spheres` only to `throughput`
for the identical nonoverlapping geometry with zero clump IDs. That is a
throughput comparison, not a physical rigid-body equivalence test. Benchmark
distributed 1/2/4 GPUs with fixed input, depth, thresholds, duration and output
policy, recording build hashes and native solver timing. Keep correctness
diagnostics disabled for timing. Generate-only does not authorize a large GPU
run, and a short timing run does not establish physical convergence.

## Restart, shearing-box and constrained-depth extension

`extension_validation.py`, `test_extension_validation.py` and
`test_clump_shear_math.cu` form a separate extension suite. They do not change
the original fixtures or their acceptance receipts. Keep new generated data
under a distinct `extensions-20261026/` attempt beneath the campaign in
`~/data/`; preserve all earlier successful and failed evidence.

CPU-only preparation and tests:

```bash
python3 -m unittest discover -s components/tests/integration_test/dem_multigpu_clumps -p test_extension_validation.py -v
python3 components/tests/integration_test/dem_multigpu_clumps/extension_validation.py prepare \
  --fixture shear-contact --ranks 2 \
  --work-dir "$HOME/data/clumps-extensions-NEW/shear-input"
```

The optional production-helper test is a host-only executable compiled with
the configured CUDA compiler and repository/yaml include paths. It calls the
actual `clump_hill_torque` and `clump_shear_wrap` helpers without initializing
CUDA or launching a kernel. Twenty-four full-tensor quaternion/spin cases use
independent six-point-per-sphere continuum quadrature, including intrinsic
sphere inertia; wrapping tests cover radial velocity transforms, absolute
winding, idempotence and checked overflow. Record the exact compiler command
and result outside the checkout.

After the requested build and GPU lane are available, each solver stage is an
explicit command against an existing executable:

```bash
python3 components/tests/integration_test/dem_multigpu_clumps/extension_validation.py run \
  --binary /absolute/path/granit_multigpu \
  --work-dir "$HOME/data/clumps-extensions-NEW/restart" --phase restart --ranks 1 2 4
```

Use `--phase shear` or `--phase adaptive` for the other stages. All subprocesses
are bounded, inputs/configurations/executables are hashed, and the campaign
stops at its first failed gate. The initial implementation is tested with
same-rank restart; changed-rank adaptive checkpoint compatibility must be
reported separately rather than inferred.

Restart checks compare uninterrupted versus split/resumed contact and gravity
trajectories, completed-step body wrench, checkpoint member state and exact
semantic contact history. The 48-byte contact ABI is parsed explicitly: active
directed histories are keyed by owner/partner global IDs; both semantic flags
and all nine FP32 spring components must match exactly. Cached local IDs,
inactive unused slots and the two ABI padding bytes are excluded. Zero-active
checkpoints may omit the contact payload or store a complete inactive stride;
partial/truncated payloads fail. Nonzero spring history must be exercised in
the contact fixtures. Zero-step restart and missing/corrupt mandatory
`.distributed_clumps` sidecars have separate checks.

Periodic extension means the existing **shearing-box finite-image FMM gravity
model**. It is not an ordinary periodic contact-only test or an infinite-sheet
gravity calculation. The fixture uses two spinning four-member bodies across
the radial seam, with intentional interbody overlap, common radial motion,
finite `(layers_x,layers_y)=(1,1)`, zero gravity cutoff and fixed shear phase
origin. Output checks use the shearing lattice to recover coherent member
offsets/velocities, retain canonical IDs and winding, verify spin/torque, and
require a real winding change in the full seam-crossing run. The same fixture
is split and resumed across the seam.

The independent gravity reference sums each configured physical image using
the model's canonical sheared-Y coordinate. It independently locates the
nearest internal rigid copy using a search wider than the finite image window,
then excludes only that internal copy when present. Nonlocal replicas sharing
the same positive clump ID remain physical sources. This avoids copying the
production diameter-based near-filter into the oracle. Pure gravitational
member force comes from the gravity sidecar; the contact-free body comparison
subtracts analytic full-step Hill force and full-inertia Hill torque from the
body output. Force/torque accuracy retains the established relative L2 `1e-4`
and maximum/RMS `5e-4` gates, with absolute norms and cancellation evidence
reported when relevant.

Adaptive validation independently bins globally canonical source positions
under every recorded common root/effective depth. For shear it explicitly
enumerates the full configured logical X/Y image set, independently of the
production stored-X/virtual-Y optimization. Every evaluation must satisfy both
global maximum leaf occupancy `<=128` and radius-inclusive full clump diameter
`<leaf_width*(1-1e-6)`. Diagnostic roots/depths/counts must agree with the saved
physical frames. The CPU tests include two artificial 80-particle shards whose
local maxima are legal but global same-leaf count is160, and a case with no
jointly valid occupancy/diameter depth. Runtime fixtures check refinement,
same-chosen-depth fixed/adaptive parity and explicit clean rejection of a
too-coarse range or diameter/occupancy conflict. Root-growth coverage requires
observed root changes; fixture preparation alone does not establish it.

The separate `--phase combined` adds concurrent shear+FMM+adaptive-depth+
checkpoint coverage. `shear-adaptive-contact` uses base-depth range2..6 with
trial interval2 evaluations, zero warmup, one sample and one confirmation;
every actual image-envelope depth is independently audited. The phase also
checks isolated adaptive restart, an outward moving anchor that must produce
an observed root enlargement, fixed-depth two-to-four-GPU continuation, and
explicit rejection of an adaptive checkpoint on a changed GPU count.

```bash
python3 components/tests/integration_test/dem_multigpu_clumps/extension_validation.py run \
  --binary /absolute/path/granit_multigpu \
  --work-dir "$HOME/data/clumps-extensions-NEW/combined" --phase combined --ranks 1 2 4
```

`--reference-campaign /absolute/accepted/restart-stage` reuses its
`restart-gravity-gpu2-full` for the fixed two-to-four test only when the binary
SHA matches. Omit this option to create a new reference. Restart always
uses scheduled checkpoint20 from a full40-step run; zero-step roundtrip uses
the checkpoint's actual stored time, avoiding an extra clipped residual step.

`--phase shear-control` runs the additional `shear-near-gravity` case, retaining
the original `shear-gravity` test and its failure receipts. The new case uses
contact-free anisotropic bodies of radius .001, with one body's actual member
representatives split across the radial seam. Base depth2 places the nearest
interbody image in the P2P neighborhood; distant configured images continue
through M2L. The same independent force/torque gates apply. The suite requires
both split-body winding and noncentral internal-copy exclusions to be observed.

For a source-identical causality check, prepare normal spheres from the actual
accepted clump bootstrap snapshot rather than the earlier input coordinates:

```bash
python3 components/tests/integration_test/dem_multigpu_clumps/extension_validation.py prepare-sphere-control \
  --clump-run /absolute/original/shear-gravity-run \
  --work-dir "$HOME/data/clumps-extensions-NEW/sphere-causality"
```

After an explicitly authorized one-GPU solver run using that new config,
`analyze-sphere-control --clump-run ORIGINAL --sphere-run CONTROL` subtracts
the exact direct internal forces between members of each original body from
the sphere gravity sidecar. It requires identical bootstrap positions,
velocities, radii, masses, materials and finite-image/root controls, then
compares corrected forces and the complete far-error residual pattern. A
matching residual attributes that measured error to the shared unclumped FMM
path; it does not turn the original accuracy failure into a pass.

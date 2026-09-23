# GRANIT YAML configuration

The maintained templates are:

- [`dem_example.yaml`](../components/config/dem_example.yaml)
- [`sph_example.yaml`](../components/config/sph_example.yaml)
- [`materials_example.yaml`](../components/config/materials_example.yaml)

Planetary-ring local simulations have additional frame and boundary semantics;
see [Planetary-ring shearing box](shear_box.md).

Copy the relevant simulation template and point it at a version 3 particle
binary. Material and interface physics belong only in the external material
database; see [Material database](material_database.md).

## Required mode and inputs

```yaml
material_database: materials.yaml

simulation:
  physics: dem                 # dem or sph; hybrid is not supported
  time:
    start: 0.0
    stop: 1.0
    output_step: 0.1

io:
  input:
    particle_file: data/particles_v3.bin
```

`simulation.physics` is the sole physics selector. Both modes use
`io.input.particle_file`; SPH has no CSV runtime path or separate input-format
switch. The binary's physics tag must match the YAML mode and every particle
must carry a valid `material_id`.

Do not add a `particle:` block. DEM radius comes from the binary and density,
elasticity, damping, friction, shape, and cohesion come from the material
database. SPH mass and current density come from the binary; constitutive
parameters come from the database.

DEM additionally chooses one damping family for the whole run:

```yaml
dem:
  damping_model: restitution   # or viscous
```

All database interaction records must use the corresponding coefficient.

## SPH numerical controls

The `sph_parameters` block contains numerical controls only:

```yaml
sph_parameters:
  viscosity: 1.0
  smoothing_length: 0.02
  num_neighbors: 64
  density_update_mode: 1
  enable_shepard_filter: true
  use_kernel_correction: true
  use_artificial_stress: true
  artstr_epsilon: 0.3
  artstr_n: 4.0
  gravity_softening_len: 0.01
  cfl_number: 0.3
  dt_min: 1.0e-8
  dt_max: 1.0e-4
  kernel_cutoff_factor: 2.0
  hsml_min: 0.005
  hsml_max: 0.05
  damage_seed: 12345678  # deterministic Grady-Kipp flaw realization
```

`damage_seed` is the global deterministic seed used to realize Grady-Kipp
flaws; record it with run provenance. EOS, reference density, strength, damage,
and porosity keys are rejected here.

## Walls and clumps

Every contact-capable wall has a required material ID:

```yaml
walls:
  - id: 1
    type: plain
    material_id: 0
    position: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
```

The rule applies to plain, finite plain, moving, mesh, bonded rectangular,
funnel, and sphere walls. Dynamic wall mass/inertia remains wall configuration;
the material ID selects contact constitutive properties.

Unified-v3 inputs store every member sphere and its material directly. YAML
contains only reusable rigid-property overrides and one instance per positive,
contiguous `clump_id` (`1..K`); it never creates spheres:

```yaml
clumps:
  templates:
    - name: aggregate
      inertia: {xx: 1.0, yy: 1.1, zz: 1.2, xy: 0.0, xz: 0.0, yz: 0.0}
      multipole:
        basis: trace_free_p3_v1
        values: [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  instances:
    - id: 1
      template: aggregate
      com: [0.0, 0.0, 0.0]
      orientation: [1, 0, 0, 0]
```

`clump_id: 0` is an ordinary live particle and `clump_id: -1` is a dead row
excluded from all later calculations. A materially different CoM override
requires both inertia and multipole overrides. The multipole monopole must
equal the derived member mass and its three dipole coefficients must be zero.
The six inertia values and all multipole coefficients are body-frame values.

For self-gravity, the compact body list contains ordinary live particles plus
one CoM entry per clump; member and dead rows never enter Karras. DIRECT uses
the conservative p0..p3 pair energy for every body pair. BFFMM retains its
hierarchy for normal-normal interactions and evaluates every pair involving a
clump with that same p3 conservative kernel, including body torque. Clump
BFFMM runs therefore require `unified_fmm.enable_global_projection: true`
(the default). A `gravity_update_rate` greater than one uses RESPA bracketing
impulses and a separate cached clump gravity wrench.

For SPH, the same unified-v3 storage field has a different meaning: it is the
exact SPHSOL Part ID. Active IDs are dense `0..K`; `-1` is inactive/dead. A
fresh SPH input that omits the field defaults every particle to Part `0`.
This does not create or configure a DEM clump.

RESPA treats self-gravity as the slow force level and contact, wall, and other
enabled accelerations as the fast level. Select the outer interval with either
`gravity_update_rate` (an integer number of inner DEM steps) or
`gravity_macro_dt` (a maximum interval in seconds, rounded down to an integer
number of inner steps), but not both:

```yaml
accelerations:
  self_gravity:
    enabled: true
    method: direct
    gravity_update_rate: 2
```

Every force level is evaluated before the first drift. Output, checkpoint, and
termination boundaries shorten the current macro-step so saved states always
include the closing slow-force kick. New RESPA checkpoints carry a
`.respa_sync` sidecar; older RESPA checkpoints are rejected because their kick
phase is unknowable. Ordinary particle inputs and outputs remain valid restart
sources.

`tree_rebuild_rate` is not a RESPA alias and is rejected. BFFMM rebuilds its
topology on every gravity evaluation; its former topology-cadence controls are
obsolete. Cundall local
damping and acceleration limiting are nonlinear in the total force and are
therefore rejected with RESPA; separable viscous damping is supported with a
symmetric split. Always demonstrate convergence in bulk observables as the
outer interval is reduced. A slowly varying gravity field alone does not make a
large outer step safe when gravity is balanced by a frictional contact network.

## Adaptive BFFMM tree depth

New non-shear BFFMM simulations can let the runtime select and time neighboring
tree depths. Estimating a suitable fixed depth is difficult and the best depth
changes with particle count and spatial occupancy:

```yaml
accelerations:
  self_gravity:
    enabled: true
    method: unified_fmm
    gravity_update_rate: 2
    unified_fmm:
      tree_dimension: 3       # required: 2 (X-Y quadtree) or 3 (octree)
      tree_depth: auto
      adaptive_depth:
        # enabled defaults to true when this block is present
        min_depth: 6
        max_depth: 12
        trial_interval_evaluations: 32
        trial_warmup_evaluations: 1
        trial_samples: 4
        min_improvement_fraction: 0.02
        confirmation_trials: 2
        capacity_headroom: 2.0
        capacity_growth_trigger: 0.75
        multigpu:
          local_count_reset_fraction: 0.10
          secondary_objective: mean_epoch_time
          trial_order: round_robin
```

The startup sparse-tree probe chooses the initial depth when `tree_depth: auto`;
the single timing controller then refines it. An explicit integer can instead
seed that controller, but it must lie in the configured inclusive range. All
tree depths must be in `[2, 20]`. The standard
adaptive ceiling is 12; values through 20 are supported as explicit overrides
when memory permits.

The trial interval and sample counts are measured in completed gravity
evaluations, not inner DEM steps. Candidate forces are used by the integration
step, so trials do not run a duplicate shadow gravity evaluation.

Startup first builds a lightweight Morton-prefix occupancy model, then reserves
sparse node storage for the chosen depth and one deeper neighbor with the
configured headroom. It does not allocate the strict `max_depth` tree up front.
If the sparse high-water mark reaches `capacity_growth_trigger`, storage growth
and CUDA-graph recapture are serviced only at an existing output, checkpoint,
or final boundary. Until such a boundary, a capacity-limited evaluation falls
back to the deepest fitting candidate; exhaustion at `min_depth` is a controlled
error. Startup also preflights the eventual maximum-depth allocation and reports
the required/free memory, component breakdown, and largest estimated feasible
depth before launching the adaptive pipeline.

On the single-GPU runner, accepted transitions are appended to
`adaptive_fmm_depth.csv` in the output directory. Adaptive checkpoints add an
atomic `.adaptive_fmm_depth` sidecar
containing version 2's tree dimension, one controller state, producer sequence,
and persisted CSV watermark. The sidecar is required when resuming an adaptive
run; version-1 adaptive sidecars are rejected rather than converted. The CSV is
rewound to the checkpoint watermark before new rows are appended. A diagnostic
sequence gap is fatal: telemetry is lossless and never reports normal dropped
records.

`unified_fmm.tree_dimension` is required and must be `2` or `3`. The legacy
`force_tree_type: 2|3` spelling is accepted with a warning for one release;
`0`, conflicting values, and omitted dimensions are errors. If `tree_depth` is
omitted it enables adaptive depth and starts automatically on a fresh launch.
Automatic checkpoint resume requires explicit `tree_depth: auto` or an integer.
An explicit integer
without an `adaptive_depth` block selects fixed depth; with an enabled block it
is an adaptive seed. `enabled: false` requires an explicit integer, and rejects
`auto` or an omitted depth. The old topology keys `rebuild_threshold`,
`enable_topology_freeze`, `update_rate`, `full_rebuild_rate`,
`tree_check_interval`, `min_residence_checks`, `consecutive_pass_for_downgrade`,
and `shape_ratio_quadtree` are rejected.

To deliberately ignore an existing checkpoint and begin from the particle
input, set `io.checkpoint.force_fresh: true`. Otherwise an adaptive resume
requires its matching version-2 sidecar; a fixed-depth configuration may resume
the base particle checkpoint and emits a warning that the adaptive controller
state is ignored.

With Morton ownership, the multi-GPU runner uses one common computation depth
across ranks. The host controller scores maximum rank gravity time, mean rank
time, and aggregate LET bytes, and stages the same depth on every rank. The
minimum per-rank memory ceiling constrains all trials; ownership
`partition_depth` remains separate. This path supports fresh shear-box inputs,
including one visible GPU through `granit_multigpu`; the finite-image envelope
adds levels to the selected base depth. Adaptive shear is rejected by the
standalone `granit_dem` runner. See [shear-box configuration](shear_box.md).

The legacy non-Morton route uses one independently selected depth per GPU. GPU 0 runs
a nonblocking host controller over complete gravity epochs. Trials change one GPU at
a time in round-robin A/B/B/A order and score global maximum epoch time first,
global mean second, and aggregate LET bytes third. The existing LET wire format
is self-describing, so different sender depths require no tree or depth-map
exchange. A host-only worker rendezvous snapshots one cohort-consistent trial
command per step; it adds no CUDA synchronization or peer payload. Trial epochs
are excluded from every adaptive-RCB calibration component and trigger
decisions; accepted RCB changes, capacity service, YORP transitions, and a local
owned-count change larger than `local_count_reset_fraction` cancel the active
comparison. `secondary_objective` currently accepts `mean_epoch_time`, and
`trial_order` accepts `round_robin`.

Distributed timing and controller records are written to
`adaptive_fmm_depth_multigpu.csv` and
`adaptive_fmm_depth_multigpu_events.csv`. They include the actual launch depth,
trial ID, per-GPU whole-gravity and remote-evaluation times, owned count, LET
records/bytes, a configuration-file fingerprint, and the visible PCI bus layout.
The layout distinguishes local and cross-socket `CUDA_VISIBLE_DEVICES`
placements. Diagnostic history is lossless: an overwritten, dropped, or invalid
timing epoch is fatal rather than silently scored as a different A/B phase.
Recoverable device capacity fallback updates the owning
GPU's host-side effective ceiling before another candidate is issued. In the
event CSV, `cancel_reason` is `-1` unless `event` is `TrialCancelled`.

Distributed adaptive YORP performs one owned-only reduction per GPU at each
configured physical-time check. GPU 0 sums those compact records in logical-GPU
order, evaluates the adaptive state machine once, and broadcasts its state.
Each worker consumes that mirrored state directly; there is no OpenMP
check-boundary barrier. Before GPU 0 mutates the authoritative state again, its
stream waits for the directed peer-consumed acknowledgements from that broadcast.
An invalid physical metric produces the same FATAL state on every GPU, writes the
final common-step output, and terminates the run cleanly. CUDA/P2P coordination
failures remain fatal process errors.

Adaptive multi-GPU depth remains limited to fresh-start, unclumped DEM with
non-shear BFFMM. Clump runs fail before launching their gravity pipeline; use a
fixed integer `tree_depth` for those simulations.

## Common runtime sections

`domain`, `neighbor`, `accelerations`, `damping`, `io.output`, `io.checkpoint`,
`gpu`, and `multigpu` retain their established numerical/runtime meanings. The
examples intentionally show only a small useful subset. In particular:

```yaml
domain:
  box_min: [-1.0, -1.0, -1.0]
  box_max: [ 1.0,  1.0,  1.0]

neighbor:
  method: cll                  # cll, lbvh, or direct
  backend: auto                # auto, cuda, or rt; RT requires method: lbvh

gpu:
  device_id: 0
  block_size: 256

multigpu:
  enabled: true
  max_gpus: 2
```

Paths in the simulation file should be treated as deployment inputs. The
material database path is explicitly resolved relative to the simulation YAML.

`neighbor.backend: auto` selects RT for supported single-GPU, non-periodic DEM
and SPH LBVH runs when available and otherwise uses CUDA. `cuda` forces the CUDA
backend; `rt` requires RT and rejects unsupported configurations. The method
default remains `cll`. RT requires `multigpu.enabled: false`, no enabled
periodic wall, and `shear_box.enabled: false`.

The optimized single-GPU SPH LBVH path builds directed adjacency once and
reuses it across the density, density-rate, force, strength, correction, and
energy passes. With `neighbor.update_frequency: 1`, it rebuilds each step.
Larger cadences permit candidate reuse only while particle identities and
layout remain unchanged, displacement stays within the cache budget, and
smoothing lengths remain unchanged. Any smoothing-length change
conservatively invalidates the cached candidates. Multi-GPU SPH retains its
existing per-step CUDA rebuild and boundary restrictions. These controls do
not imply a measured SPH speedup. See [RT neighbor search](neighbor_rt.md) for
build options, supported modes, and the numerical contract.

Optional SPH stages, physical histories, fixed supports, symmetry, and restart compatibility are documented in [SPH physics stages](SPH_Physics_Stages.md).


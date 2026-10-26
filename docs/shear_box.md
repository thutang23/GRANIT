# Planetary-ring shearing box

GRANIT's shear box is a local Cartesian approximation to a small patch of a
planetary ring. The coordinates are `x` radially outward, `y` in the prograde
azimuthal direction, and `z` normal to the ring plane. The background flow is

```text
v_y = -q Omega x
```

with `q=3/2` for a Keplerian disk. The runtime adds Hill's radial tide,
Coriolis acceleration, and vertical harmonic gravity, and applies sliding
periodic boundaries in `x` and ordinary periodic boundaries in `y`. The `z`
direction is nonperiodic; optional horizontal physical walls can be configured
separately when a bounded vertical domain is wanted.
This is a local model: every horizontal box length must remain much smaller
than the orbital radius.

The method follows the local ring simulations introduced by
[Wisdom & Tremaine (1988)](https://doi.org/10.1086/114690). Useful descriptions
of the boundary construction and simulation practice are given by
[Salo (1995)](https://doi.org/10.1006/icar.1995.1057),
[Daisaka & Ida (1999)](https://arxiv.org/abs/astro-ph/9908057), and the
[Salo, Ohtsuki & Lewis review](https://cc.oulu.fi/~hsalo/CHAPTER17/chapter17_120416.pdf).

## Configuration

Specify either `omega` directly or a central mass and orbital radius:

```yaml
shear_box:
  enabled: true
  omega: 1.3143527e-4
  shear_parameter: 1.5
  vertical_frequency: 1.3143527e-4
  time_origin: 0.0
  input_velocity: total
  orbital_dt_factor: 0.05
  force_balance_warn_threshold: 1.0e-5
  gravity_images:
    layers_x: 1
    layers_y: 1
    cutoff_radius: 0.0
```

For finite-order FMM, use a looser diagnostic threshold such as `1.0e-3`;
the p=4 approximation is not pairwise antisymmetric to direct-summation
roundoff. The wrapped finite image window can also have nonzero net force in
an exact direct sum, because its outer boundary does not retain every opposite
pair. Separate this truncation effect from FMM error when interpreting the
warning. The Saturn FMM presets use this value.

The equivalent frequency-derived form is:

```yaml
shear_box:
  enabled: true
  central_body:
    mass: 5.6834e26
    orbital_radius: 1.30e8
    gravitational_constant: 6.67428e-11
```

`vertical_frequency` defaults to `omega`. `time_origin` is the zero-shear
phase and must remain fixed across restart. `input_velocity` is either `total`
or `peculiar`; outputs use total velocity. `orbital_dt_factor` bounds the time
step by an orbital scale, but it does not replace the stricter soft-contact
time-step constraint. Optional `simulation.time.dt` is another upper bound,
not a fixed step.

`gravity_images.layers_x=1` and `layers_y=1` evaluate the central patch plus
the surrounding eight sheared replicas. Two layers evaluate a 5 by 5 set (24
replicas). `cutoff_radius: 0.0` (or `null`) disables the optional physical
cutoff; negative values are rejected. These are
gravity replicas, not merely collision ghosts: increasing their count changes
the long-range force approximation. A finite replicated patch is not an Ewald
sum of an infinite sheet, so image-layer and box-size convergence must be
reported for quantitative work.

The official [REBOUND Saturn-ring example](https://rebound.hanno-rein.de/ipython_examples/SaturnsRings/)
uses two gravity-image layers and notes that additional layers smooth the force
at a boundary. Its hard-sphere time step must not be copied into a soft-sphere
GRANIT run. In a force/contact method, the Salo review recommends resolving a
contact duration with roughly 10--100 integration steps.

## Multi-GPU execution

The `granit_multigpu` executable accepts shear-box DEM inputs on GPUs
within one node. Add the following settings to an otherwise complete shear-box
configuration, retaining its materials, time step, particle input, and domain:

```yaml
multigpu:
  enabled: true
  max_gpus: 2
  morton_ownership: true
  partition_depth: 6

accelerations:
  self_gravity:
    enabled: true
    method: unified_fmm
    gravity_update_rate: 1
    unified_fmm:
      tree_dimension: 3
      tree_depth: auto
      adaptive_depth:
        enabled: true
        min_depth: 6
        max_depth: 12

neighbor:
  method: lbvh
  update_frequency: 1
```

For ordinary-sphere DEM shear boxes, `neighbor.method: cll` is also supported.
It rebuilds 2D XY cell bins at each contact evaluation using the current particle support
extent and shear phase, then checks full 3D distances. The contact search
requires `2*max_support+verlet_skin` to be less than half each planar box
length so every particle pair has a unique periodic image. The vertical box
extent does not constrain the bin size. `update_frequency` must remain `1`.

Rigid clumps use the distributed small-clump path with LBVH neighbors and Morton
ownership. With self-gravity, retain `tree_dimension: 3`,
`gravity_update_rate: 1`, `gravity_macro_dt: 0`,
`direct_threshold_leaf: 0`, `direct_threshold_internal: 0`, and
`enable_global_projection: false`. Both fixed and adaptive common depth are
supported. Body centers use the exact Hill drift; full-tensor angular dynamics
include gyroscopic and Hill torques. Rigid reconstruction and unwrapped moment
arms preserve bodies crossing sliding faces.

Let `D` be the maximum full clump bounding diameter, including sphere radii.
Require `2D < min(Lx,Ly) * (1 - 1e-6)` for a unique internal image. Gravity also
requires `D < h_leaf * (1 - 1e-6)`, using the finite-image source root and actual
effective FMM depth. Same-body internal pairs are excluded, while the body's
other finite replicas remain gravitational sources. Adaptive clump runs count
all logical X/Y gravity replicas and enforce at most 128 global source particles
per leaf before every evaluation. The allowed depth interval must satisfy this
occupancy limit, the diameter limit, configured bounds, and memory constraints;
an empty intersection is a normal descriptive error. These checks do not change
the finite-image gravity model described above.

Keep `shear_box.gravity_images.cutoff_radius: 0.0` when comparing the uncut multi-GPU gravity calculation. All compared runs must use
the same physical particles, image layers, and force parameters. A positive
cutoff changes the requested gravity calculation and cannot substitute for an
uncut comparison.

Before migration and contact queries, each device wraps its owned particles
at the current force-evaluation time. Halo classification intersects the
ownership tiles with the nine current contact-image transforms and sends each
physical particle at most once to each destination. The contact query applies
the selected position and radial-image velocity transform. Particle IDs,
contact history, and integer winding counters follow migration; initial
scatter and output gathering preserve winding as well.
Recipient tiles are clipped to the physical XY rectangle before applying the
image transforms; padding in the square Morton grid cannot create extra halos.

The optional `gravity_force` output sidecar contains pure gravitational force
in both ordinary and RESPA stepping; the latter converts its acceleration
cache using each particle's mass. The ordinary BFFMM shear path records the
gravity increment without adding another force evaluation. Sparse and dense
`TF32_TC` batches use the same input rounding, so ownership changes cannot
switch their operand precision.

The single-GPU solver's per-step force-balance warning remains a separate
diagnostic cost. The distributed path suppresses per-rank warnings, because
a rank's force sum is not the global force sum; it does not currently emit an
aggregate force-balance warning. Compare matched gravity-core timings as well
as full solver timings when assessing this diagnostic difference.

The tree uses a fixed image-envelope root. Observed vertical extents determine
the allocation bound, but particles leaving that root fail the containment
check; the root does not expand automatically in `z`. Contact ownership itself
has no vertical split. Distributed clump checkpoint/resume preserves body state,
member/contact histories, and winding through the checkpoint and its mandatory
`.distributed_clumps` sidecar. Fixed-depth restart can change rank count;
adaptive restart requires the same ranks/backend/configuration and the matching
`.adaptive_fmm_depth` sidecar as well. Preserve `time_origin` across restart.
See [distributed small-clump DEM](MultiGPU_Clumps.md) for the complete contract.
Shear checkpoint/resume for ordinary-sphere-only multi-GPU DEM, standalone
single-GPU shear clumps, SPH shear boxes, and incompatible forcing remain
unsupported.

Build with an installed CUDA compiler and select the devices available to the run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target granit_multigpu -j4
build/bin/granit_multigpu /absolute/path/to/config.yaml
```

## Scientific limits

The Toomre critical wavelength is `lambda_crit = 4 pi^2 G Sigma / Omega^2`. Quantitative ring studies require checks of box size, gravity-image layers, contact and gravity timesteps, FMM accuracy, duration, and initial-transient removal. Soft-sphere contact stiffness and overlap resolution require their own convergence checks; a hard-sphere timestep is not transferable to this solver.

## Continuous trajectories and winding counters

Wrapped snapshots alone cannot reliably reveal how many boundaries a particle
crossed between sparse outputs. Shear-box outputs and checkpoints therefore
store the all-or-none signed `int64` fields `winding_x`, `winding_y`, and
`winding_z` (binary IDs `0x001A`--`0x001C`). Fresh inputs may omit them and
start at zero.

For counters `(nx,ny,nz)` and box lengths `(Lx,Ly,Lz)`, reconstruct the selected
continuous image as

```text
x_u = x + nx Lx
y_u = y + ny Ly - q Omega (t - t0) nx Lx
z_u = z + nz Lz.
```

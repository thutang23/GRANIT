# SPH RESPA, adaptive depth and restart

Both SPH runners select a CFL timestep at a closed macro boundary. The default
`accelerations.self_gravity.gravity_update_rate: 2` holds that timestep for two
hydrodynamic steps. This cadence also applies without gravity. Set the rate to
1 for per-step CFL selection. With gravity enabled, its cached acceleration
applies half-impulses before and after the macro; only endpoint gravity is
reevaluated. Hydro, strength, energy, damage, porosity and smoothing lengths
still advance every inner step.

Output, stage transitions and the terminal time close a macro exactly. If an
event falls inside the proposed macro, all inner timesteps shorten equally.
Step-based checkpoint intervals publish at the first closed boundary at or
after the requested step interval. SPH relaxation damping brackets each hydro
step; gravity impulses bracket the entire macro. A stage change rebuilds forces
under the new stage before the next macro.

`gravity_macro_dt` is an alternative maximum macro duration. Each new CFL limit
chooses `ceil(gravity_macro_dt / CFL_dt)` equal steps, then clips the span to the
next event. It requires an omitted update rate or an explicit rate of 1.

When `io.output.scalar_interval` is positive, `sph_macros.csv` records macro
start step/time, inner count, held timestep, endpoint, CFL candidate and gravity
evaluation count. The single-GPU record additionally counts CFL evaluations.
The default scalar interval is zero, avoiding this per-macro I/O.

## Gravity and depth

Omitted or `auto` FMM depth enables the same timing controller used by DEM.
Depth timing advances on actual gravity evaluations. Single-GPU timing covers
tree through P2P/L2P; distributed timing covers the collective gravity epoch.

SPH uses its current pair-mean spline softening. The compact baseline and its
correction both run in FP64, automatically selected even when DEM P2P is built
with FP32. The correction joins slow gravity before global projection. A live
physical depth ceiling enforces `h_max / epsilon <= 128`, where
`epsilon = min(hsml_min, leaf_width/4)`. It combines with the memory ceiling.
If even the configured minimum depth is infeasible, the run fails rather than
using a numerically ill-conditioned correction. Imported `h` must be at least
`hsml_min`; choose physical h bounds for the input.

## Checkpoints

SPH checkpoints store FP64 fast force and a separate FP64 slow gravitational
acceleration (`GRAVITY_ACCELERATION`, field ID `0x001E`). Visualization fields
retain their existing layout. Normal restart preserves these arrays and the
constitutive histories/rates; it does not recompute a closed force state.
Explicit force rebootstrap and stage transitions still rebuild it.

The `.respa_sync` SPH version-2 marker binds the closed state to its cadence and
CFL configuration. Older SPH gravity checkpoints without the cached slow field
cannot resume this integrator. Adaptive checkpoints also require the matching
`.adaptive_fmm_depth` sidecar: version 2 for the single-GPU controller, backend
version 3 for the common distributed controller. Distributed restart preserves
its trial state, root/cuts, epoch, rank count and scalar load-balance calibration.
Unfinished wall-clock timing samples are discarded; subsequent timing-driven
depth/balancing choices need not be bitwise identical. Controller backends and
rank counts cannot be exchanged during adaptive restart.

Single-GPU diagnostic CSV segments retain their first sequence in
`adaptive_fmm_depth.csv.start`, so a continuation in a new output directory
can resume again while still rejecting missing rows. Keep that small file
beside an existing CSV; a completely new output directory starts a new segment.

A zero-step restart can write the same complete checkpoint without an extra
gravity evaluation. Checkpoint integrity does not establish timestep convergence
or long-duration energy conservation.

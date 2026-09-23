# GRANIT SPH physical stages

This extension retains GRANIT's pressure force, geometric pressure work, half-weight artificial-viscosity heating, strength evolution, material database, and fixed particle masses. It adds optional stage controls and physical histories. The default run without `stages` retains the original integration path, except for the irreversible P-alpha unloading correction.

The subsequent SPH multi-GPU integration passed 95 bounded checks on CUDA 12.1/SM89, including single-versus-1/2/4-GPU comparisons, stage/history continuation, and direct-force gravity references. This supersedes the earlier failed single-versus-two-GPU comparison. See [SPH_MultiGPU_Integration.md](SPH_MultiGPU_Integration.md) for source scope, fixed gates, reproduction, and remaining limits. These checks do not establish scientific convergence.

## Stage configuration

Add the following to an existing SPH configuration. Times are absolute simulation seconds, and the last end time must exactly equal `simulation.time.stop`. The numbers illustrate syntax; choose times and damping for the physical problem.

```yaml
simulation:
  physics: sph
  time: {start: 0.0, stop: 0.01, output_step: 0.001}

sph_parameters:
  density_update_mode: 1
  track_history: true
  stages:
    - mode: RELAX
      end_time: 0.002
      damping_rate: 1000.0
    - mode: SHOCK
      end_time: 0.008
    - mode: LATE
      end_time: 0.01
```

| Default | RELAX | SHOCK | LATE |
|---|---|---|---|
| EOS | Elastic | Material-selected | Elastic |
| Density, velocity, smoothing length | Evolve | Evolve | Evolve |
| Internal energy | Frozen | Evolve | Evolve |
| Deviatoric strength | Enabled | Enabled | Enabled |
| Damage and distension | Frozen | Evolve | Frozen |
| Shock peaks / cold energy / fracture history | Frozen | Update | Frozen |
| Velocity damping | Optional | Off | Off |

Each stage accepts `evolve_energy`, `evolve_damage`, `evolve_porosity`, `use_strength`, and `elastic_eos` overrides. History diagnostics remain SHOCK-only except for damping energy. A sequence can omit stages but must retain RELAX → SHOCK → LATE order, without repeats. Stage sequences require time-based termination. `track_history` defaults to true when stages are supplied, false otherwise; it requires continuous density (`density_update_mode: 1`). Legacy relaxation strength/damage disable flags override RELAX in a sequence, and retain their old whole-run meaning when no sequence is configured.

A step lands at each stage boundary. Forces are rebuilt under the new stage before its next drift. RELAX applies two half-step exponential velocity decays, with specific kinetic-energy removal accumulated separately. It does not convert removed kinetic energy to material heat and does not alter particle mass.

RELAX acts on all live, unconstrained particles in the supplied run. It does not automatically remove an impactor and later insert it. For target-only preparation, relax the target separately and construct the collision input from that result plus the impactor, then run SHOCK → LATE. Do not expect an impactor's initial velocity to survive a whole-system damping stage. Stage transitions do not reset positions, stress, density references, energy, damage, or porosity.

## Material controls

In a material's existing `sph` mapping, `eos.model` accepts:

- `tillotson`: the original complete parameter set and branch formulas.
- `simplified_tillotson`: `A` and `c`; solid pressure is `c*rho_s*e + A*(rho_s-rho_s0)/rho_s0`.
- `elastic`: `A`; solid pressure is `K*(rho_s-rho_s0)/rho_s0`.

Optional siblings of `eos`:

```yaml
bulk_modulus: 2.67e10
elastic_modulus: 5.31e10
continuous_activation: false
```

`bulk_modulus` defaults to Tillotson `A`. It controls the elastic EOS, sound-speed floor, and damage crack speed; it does not replace `A` in the full Tillotson pressure polynomial. `elastic_modulus` defaults to `9*K*G/(3*K+G)` and controls tensile strain in damage activation. Independently specified moduli are the user's constitutive choices and are not silently adjusted for isotropic consistency. `continuous_activation: true` retains a fractional active-flaw estimate; false preserves GRANIT's integer-floor estimate.

Porous EOS uses `rho_s=alpha*rho`, `rho_s0=alpha_initial*reference_density`, and `P=P_s/alpha`. Expanded tensile pressure retains the existing damage reduction. EOS pressure derivatives used in P-alpha follow the selected EOS, including a stage's elastic override. The full Tillotson sound-speed formula is retained; alternative EOS models use their thermodynamic derivatives plus the bulk-modulus floor.

P-alpha now retains the maximum achieved compaction during unloading. Reloading starts at the pressure corresponding to the historical alpha on the crush curve. Alpha cannot increase and cannot fall below one. This correction intentionally changes legacy porous unloading results.

## Physical-history fields

| Field | Meaning | Unit |
|---|---|---|
| `energy_cold` | Integrated cold-compression contribution | J/kg |
| `energy_thermal_peak` | Maximum of `energy-energy_cold`, lower bound zero | J/kg |
| `pressure_peak` | Maximum compressive pressure, lower bound zero | Pa |
| `fracture_area` | Crack-area density estimate | 1/m |
| `fragment_length` | Statistical characteristic fragment length | m |
| `damping_energy` | Cumulative specific kinetic energy removed by RELAX damping | J/kg |

Cold energy integrates `P_cold/rho^2 * rho_rate`; `P_cold` evaluates the material EOS at the cold-energy state and current distension, including the ordinary tensile-damage reduction used by SPHSol. It is a diagnostic partition and is not added to the evolved internal energy. This is not a temperature model or latent-heat calculation. Fracture area follows the Grady–Kipp diagnostic growth expression; fragment length is `3*(m+3)/((m+2)*area)`. Zero denotes unavailable/no measured fragmentation when area is zero, rather than infinity. These estimates do not identify actual connected fragments or replace fragment clustering.

The six f64 fields use descriptor IDs `0x0050`–`0x0055`. They are carried through allocation, reordering, migration, output/checkpoint, binary readers/writers. Arrays add 48 bytes per allocated particle even when tracking is off; history descriptors are emitted only when tracking is enabled. The existing v3 descriptor format is retained.

## Boundaries

```yaml
sph_parameters:
  fixed_parts: [2]
  symmetry_plane:
    normal: [1.0, 0.0, 0.0]
    offset: 0.0
neighbor:
  method: lbvh
```

Both features are optional and currently single-GPU only. `fixed_parts` contains up to 16 distinct nonnegative SPH Part IDs from `clump_id`, not material IDs. The selected particles retain position, density, internal/cold energy, smoothing length, deviatoric stress, damage and distension; velocity is zero. They still supply SPH neighbors and support reactions. This is a fully fixed support, not a prescribed-motion or per-axis velocity condition.

One symmetry plane is supported. Its equation is `normal·x + offset = 0`; coefficients are normalized together. Input real particles must lie in the nonnegative half-space. Mirrored position, velocity and full stress tensor are generated in auxiliary rows; crossing real particles are reflected. Mirror rows are excluded from integration, serialization and particle mass totals. The implementation requires LBVH and live-only input, doubles particle capacity, and rejects self-gravity, periodic/shearing boundaries and multi-GPU mode. External loads must also respect the chosen physical symmetry. Multiple intersecting planes and image gravity are not implemented.

## Restart

Keep each extended checkpoint `.bin` together with its `.sph_stage` sidecar and any original GRANIT sidecars. The stage sidecar binds step, time, stage rules, history switch, fixed parts and plane coefficients. Changed/missing stage settings are rejected; an automatic search with no compatible checkpoint also fails instead of silently starting a new run. Normal outputs are not a substitute for a state-complete checkpoint.

Ordinary mid-stage resume preserves stored forces and rates. At an internal stage boundary, forces are rebuilt for the next stage; `rebootstrap_forces: true` explicitly requests a rebuild at other times. An end-of-schedule load/save preserves the closed checkpoint without advancing history.

Legacy checkpoints remain readable in legacy mode. History cannot be reconstructed from a legacy checkpoint; enabling tracked history during such a resume is rejected. Adding a new stage sequence to an old checkpoint is a new physical run, not a transparent continuation. Changing the scheduled stop time is also a configuration change and requires deliberate new-run preparation.

Numerical and physical validation depends on the selected material laws, initial conditions, resolution, and timestep. The history estimates do not establish convergence or identify connected physical fragments.


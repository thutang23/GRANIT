# Material database

DEM and SPH particle files contain immutable material IDs; constitutive data
lives in one YAML database selected by the simulation's top-level
`material_database` path. Relative paths are resolved from the simulation YAML.

See [`components/config/materials_example.yaml`](../components/config/materials_example.yaml)
for a complete two-material database.

## Identity rules

- `version` must be `1`.
- IDs are dense and contiguous: `0..N-1`.
- Existing IDs are immutable. Add a new material by appending a new ID; do not
  renumber or silently change an ID used by checkpoints.
- A material may define `dem`, `sph`, or both. Every material used by the active
  physics mode must define that mode's block.
- The parser is strict: unknown keys, missing model tags or parameters,
  non-finite values, and invalid ranges are fatal.

The normalized, expanded database is fingerprinted. Outputs and checkpoints
record that fingerprint; a checkpoint/database mismatch is fatal.

## DEM materials and interactions

The intrinsic DEM block contains only:

```yaml
dem:
  density: 2700.0
  youngs_modulus: 5.31e10
  poisson_ratio: 0.17
```

Interface properties are explicit unordered pairs in `dem_interactions`. Every
pair of DEM-capable materials, including diagonal pairs, must occur exactly
once with ascending IDs. The loader expands these records into a full symmetric
`N × N` table on every GPU.

```yaml
- materials: [0, 1]
  restitution: 0.45
  friction:
    static: 0.6
    rolling: 0.1
    twisting: 0.08
  shape_parameter: 0.7
  cohesion:
    model: scheeres
    strength: 1000.0
    cutoff_factor: 1.1
```

The simulation chooses one damping family for all pairs:

```yaml
dem:
  damping_model: restitution   # every pair uses `restitution`
```

With `damping_model: viscous`, every pair must instead provide
`viscous_coefficient`. Mixing the two forms is rejected. Cohesion is tagged
`none` or `scheeres`; Scheeres records require strength and a pair-specific
cutoff factor of at least 1.0. A cutoff factor of exactly 1.0 applies full
cohesion only while particles are overlapping or exactly touching; values
above 1.0 extend cohesion across a finite separation with the Scheeres ramp.
Cohesion does not increase sliding, rolling, or twisting load limits.

For particles `i,j`, the runtime derives only the symmetric elastic reductions:

```text
E* = 1 / ((1-nu_i^2)/E_i + (1-nu_j^2)/E_j)
G_i = E_i / (2(1+nu_i))
G* = 1 / ((2-nu_i)/G_i + (2-nu_j)/G_j)
```

All other interface values are direct table lookups. Runtime particle mass and
spherical inertia are `4/3 pi r^3 rho` and `2/5 m r^2`. Walls also carry a
required `material_id` and use the same table; static walls have infinite
effective mass while dynamic wall bodies retain their configured mass/inertia.

## SPH materials

SPH has no pair-interaction table. Each particle uses its own material for EOS,
sound speed, strength, damage, and porosity, while pair forces keep the existing
symmetric kernel formulation and global numerical controls.

A full Tillotson material can be configured as follows:

```yaml
sph:
  reference_density: 2700.0
  eos:
    model: tillotson
    a: 0.5
    b: 1.5
    A: 2.67e10
    B: 2.67e10
    alpha: 5.0
    beta: 5.0
    E0: 4.87e8
    Eiv: 4.72e6
    Ecv: 1.82e7
  strength: {model: none}       # none | von_mises | lundborg
  damage: {model: none}         # none | grady_kipp
  porosity: {model: none}       # none | p_alpha
```

For `grady_kipp`, `k` and `m` are required. `crack_speed_fraction` is the
crack-growth speed factor used with the material's elastic wave speed; it
defaults to the pilot value `0.4` and must lie in `(0, 1]`. `min_strain` is an
optional numerical activation floor and defaults to `1e-6`.

Model-specific values are required when a model other than `none` is selected.
The EOS models are `tillotson`, `simplified_tillotson`, and `elastic`. See [SPH physics stages](SPH_Physics_Stages.md) for the alternative EOS parameters, moduli, continuous flaw activation, and irreversible P-alpha unloading rules.


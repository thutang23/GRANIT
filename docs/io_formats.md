# GRANIT particle binary v3

Unified binary version 3 is the only runtime particle format for DEM and SPH.
Input, output, and checkpoint records share the same self-describing, little-endian
SoA layout. Legacy v1 binaries and SPH CSV files are accepted only by the
migration tool.

## Layout

```text
72-byte header
16-byte descriptor × descriptor_count
Tier 1 fixed arrays
Tier 2 descriptor-driven arrays
optional fixed-stride contact-history blob (DEM checkpoint only)
```

Tier 1 always contains, in order:

```text
pos_x, pos_y, pos_z       float64[N]
vel_x, vel_y, vel_z       float64[N]
global_id                 int32[N]
```

`global_id` values must be unique. Every v3 record also has a Tier 2
`material_id` field. Material IDs are signed 32-bit values on disk but must be
nonnegative members of the configured dense material database.

## Header

Magic is the eight bytes `GRANIT\0\0`; `header_size` is 72 and `version` is 3.

| Offset | Size | Type | Meaning |
| ---: | ---: | --- | --- |
| 0 | 8 | bytes | magic |
| 8 | 2 | uint16 | version |
| 10 | 2 | uint16 | header size |
| 12 | 8 | uint64 | particle count |
| 20 | 8 | int64 | current step |
| 28 | 8 | float64 | current time |
| 36 | 8 | float64 | timestep |
| 44 | 2 | uint16 | descriptor count |
| 46 | 1 | uint8 | physics: 1 DEM, 2 SPH |
| 47 | 1 | uint8 | kind: 1 input, 2 output, 3 checkpoint |
| 48 | 4 | uint32 | contact-history stride |
| 52 | 4 | uint32 | active contact count |
| 56 | 8 | uint64 | normalized material-database fingerprint |
| 64 | 8 | uint64 | normalized clump-metadata fingerprint |

An unbound generated or migrated input may use fingerprint zero. Output and
checkpoint writers store the active database fingerprint. Restart rejects any
checkpoint whose material or clump-metadata fingerprint differs from the
configured simulation.

INPUT records require step, time, timestep, and both contact metadata values to
be zero. OUTPUT records also require zero contact metadata. Checkpoint contact
history is a fixed `N × stride` array of 48-byte records whenever the active
contact count is nonzero; truncated, oversized, or trailing blobs are rejected.

## Descriptor

Each 16-byte descriptor is `<uint16 field_id, uint8 dtype, uint8 count,
uint32 reserved, uint64 byte_offset>`. `byte_offset` is relative to the first
byte of Tier 2, not to the file. Components use SoA order, so a three-component
field is `x[N], y[N], z[N]`.

Dtype values are 0 float32, 1 float64, 2 int32, 3 int16, 4 uint8, 5 bool, and
6 int64. Readers reject unknown IDs or dtypes, duplicate fields, wrong
dtype/count combinations, overlapping ranges, out-of-bounds offsets, trailing
data, and truncation.

| ID | Field | Type × count |
| ---: | --- | --- |
| 0x0001 | density | f32 × 1 |
| 0x0002 | pressure | f32 × 1 |
| 0x0003 | c_sound | f32 × 1 |
| 0x0004 | smoothing_length | f32 × 1 |
| 0x0005 | energy | f64 × 1 |
| 0x0006 | material_id | i32 × 1 |
| 0x0010 | omega | f64 × 3 |
| 0x0011 | radius | f32 × 1 |
| 0x0012 | mass | f32 × 1 |
| 0x0013 | moment_of_inertia | f32 × 1 |
| 0x0014 | force | f32 × 3; SPH checkpoints use f64 × 3 fast force |
| 0x001E | gravity_acceleration | f64 × 3; SPH checkpoint slow acceleration |
| 0x0015 | torque | f32 × 3 |
| 0x0016 | quaternion | f64 × 4 |
| 0x0017 | num_contacts | i32 × 1 |
| 0x0018 | clump_id | i32 × 1 |
| 0x0019 | clump_offset | f32 × 3 |
| 0x001A | WINDING_X (`winding_x` in Python/HDF5) | i64 × 1 |
| 0x001B | WINDING_Y (`winding_y` in Python/HDF5) | i64 × 1 |
| 0x001C | WINDING_Z (`winding_z` in Python/HDF5) | i64 × 1 |
| 0x0020 | damage | f32 × 1 |
| 0x0021 | crack_strain | f32 × 1 |
| 0x0022 | flaws_ntot | i32 × 1 |
| 0x0023 | flaws_strain_min | f32 × 1 |
| 0x0024 | flaws_strain_max | f32 × 1 |
| 0x0025 | flaws_m0 | f32 × 1 |
| 0x0030–0x0035 | six symmetric stress components | f32 × 1 |
| 0x0040 | distension | f32 × 1 |
| 0x0041 | drhodt | f32 × 1 |
| 0x0042 | dedt | f64 × 1 |

`clump_offset` is required in every DEM checkpoint and stores the member's
body-frame offset from its clump CoM. It is never valid in fresh input, output,
or SPH records.

The three signed `winding_*` counters are an all-or-none set. A fresh input may
omit them, in which case the runtime initializes all counters to zero. Shear-box
outputs and checkpoints carry all three so post-processing can reconstruct a
continuous trajectory without guessing boundary crossings from sparsely sampled
wrapped coordinates.

## Required input fields

- DEM: Tier 1, `material_id`, and `radius`. `clump_id` is optional in a fresh
  input and defaults to `0`; it is mandatory in output and checkpoint records.
  Values are `-1` (dead), `0` (normal), or a positive clump ID. Runtime derives mass and spherical
  moment of inertia from radius and the material density. `omega` is the only
  optional fresh-input field.
- SPH: Tier 1, `material_id`, `mass`, and `density`. In SPH, the historically
  named `clump_id` field is the exact SPHSOL **Part ID**, not clump membership:
  active IDs must be dense `0..K`, and `-1` is inactive/dead. It is optional
  in a fresh input (omission defaults every particle to Part `0`) and mandatory
  in output and checkpoint records. Energy and damage default to zero and are
  the only optional fresh-input fields; smoothing length comes from the
  simulation config.

There is no runtime CSV input and no implicit material 0.

SPH checkpoints written by current GRANIT also carry the complete compact
Grady-Kipp flaw realization: `flaws_ntot`, `flaws_strain_min`,
`flaws_strain_max`, and `flaws_m0`. The four descriptors are all-or-none and
are valid only in SPH checkpoints. Legacy v3 checkpoints may omit all four;
readers expose that absence so the caller can regenerate them from its recorded
`sph_parameters.damage_seed`.

## Other binary formats

Acceleration textures (`ACCELFD\0`), tidal ephemerides (`EARTHPOS`), rotation
series (`ROTATION`), wall trajectories, meshes, and clump assemblies are not
particle inputs. Their strict DEM layouts, YAML contracts, units, coverage,
and migration rules are defined in
[`DEM_Auxiliary_Input_Formats.md`](DEM_Auxiliary_Input_Formats.md).

The optional SPH history descriptors are documented in [SPH physics stages](SPH_Physics_Stages.md). Input generation and legacy-format conversion tools are maintained separately from this source package.


SPH macro checkpoints preserve the two force levels separately; see [SPH RESPA](SPH_RESPA.md) for synchronized sidecars and restart compatibility.

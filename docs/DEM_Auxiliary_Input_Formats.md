# GRANIT DEM auxiliary input formats

This document defines the maintained auxiliary inputs for GRANIT DEM. All binary scalars are little-endian. Length, version, finiteness, time coverage, units, and coordinate frames are validation requirements. SPH may define different contracts in future work.

Use SI units unless stated otherwise. Record source frame, simulation frame, epoch/orientation transform, generating command, and SHA-256 in your run provenance record. A binary header does not replace that provenance.

## ACCELFD v2: static acceleration fields

Spatial fields use a time-major, component-planar, x-contiguous layout:

```text
float32 acceleration[frame][component][z][y][x]
NumPy shape: (1, nx, ny, nz, 3)
component: 0=ax, 1=ay, 2=az
```

The array is written in C order, so `x` varies fastest, then `y`, `z`, component, and frame. Each `(ax, ay, az)` component plane can be uploaded directly to its CUDA texture without a runtime transpose. This revision supports one static frame. Writers must convert their in-memory layout to the component-planar disk layout above.

### Header: 68 bytes

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | `ACCELFD\0` |
| 8 | `uint32` | version `2` |
| 12 | `uint32` | frame count; currently exactly `1` |
| 16,20,24 | `int32` | `nx, ny, nz`; each >=2 for spatial fields |
| 28,32,36 | `float32` | grid origin `(x,y,z)` in m |
| 40,44,48 | `float32` | positive spacing `(dx,dy,dz)` in m |
| 52 | `float32` | time start; zero for a static field |
| 56 | `float32` | time interval; zero for a static field |
| 60 | `uint8` | time-varying flag; currently `0` |
| 61 | `uint8` | space-varying flag |
| 62 | `uint8` | interpolation: `1` = linear |
| 63 | `uint8[5]` | reserved zero bytes |

The payload is exactly `12 * nx * ny * nz` bytes for a spatial field. A v2 uniform field uses zero dimensions, `space_varying=0`, and one `(ax,ay,az)` float32 vector. The runtime keeps only the known v1 static-uniform compatibility form; v1 spatial data are rejected and must be regenerated.

```yaml
accelerations:
  self_gravity:
    enabled: false        # true makes ACCELFD additive; approve this choice
  fields:
    - id: 1
      name: body_gravity
      enabled: true
      data_path: /absolute/campaign/input/body_gravity.accelfd
      load_method: once
      time_varying: false
      space_varying: true
      approved_position_bounds_m:
        min: [-500, -500, -500]  # approved full planned center envelope
        max: [ 500,  500,  500]
      interpolation:
        space_method: linear
        time_method: constant
```

Verify the particle centers against `approved_position_bounds_m` and that full envelope against `[origin, origin + spacing*(dims-1)]`. The approved envelope must conservatively cover the full planned motion, not only the initial state. Particles outside the grid cause a fatal device error. Boundary clamping would change the force and is not supported. Time-varying ACCELFD is not implemented in this revision.

Before generation, approve whether the field is world-fixed or body-fixed, the coordinate transform, add-versus-replace self-gravity, source mass/density normalization, grid spacing/padding, float32 error tolerance, and coverage of every expected particle position.

## ROTATION v1: rotating-frame angular kinematics

The 36-byte header is:

```text
<8sIIff12s
magic="ROTATION", version=1, frame_count,
time_start_s, uniform_time_interval_s, reserved=zeros
```

It is followed by `frame_count` records of six float32 values:

```text
omega_x, omega_y, omega_z, omega_dot_x, omega_dot_y, omega_dot_z
```

`omega` is rad/s and `omega_dot` is rad/s² in the documented simulation-frame axes. Linear interpolation is the only supported interpolation. The file must cover the complete simulation interval; the runtime does not clamp. `omega_dot` is mandatory and is never inferred without a separate approved derivation.

```yaml
accelerations:
  inertial:
    enabled: true
    constant_omega: false
    data_path: /absolute/campaign/input/frame.rotation
    load_method: once
    interpolation: {method: linear}
```

For an approved constant-angular-velocity rotating frame, use `simulation.rotating_frame.omega` with `constant_omega: true`; do not manufacture a ROTATION file.

## EARTHPOS v1: tidal perturber ephemeris

The 40-byte header is:

```text
<8sIIffd8s
magic="EARTHPOS", version=1, frame_count,
time_start_s, uniform_time_interval_s,
perturber_mass_kg, reserved=zeros
```

Each frame is four float64 values:

```text
perturber_position_x_m, y_m, z_m, stored_magnitude_m
```

The position is the perturber relative to the simulation origin in the approved simulation frame. The stored magnitude must match the vector. Linear interpolation is used, complete coverage is required, and out-of-range sampling is fatal. Header mass is authoritative; YAML must name the same finite positive mass.

```yaml
accelerations:
  tidal:
    enabled: true
    data_path: /absolute/campaign/input/perturber.earthpos
    load_method: once
    interpolation: {method: linear}
    perturber: {mass: 5.972e24}
```

Record the source ephemeris frame and epoch plus the full inertial-to-simulation rotation. A legacy `omega/phase` +Z rotation is not a general frame transform.

## RIGIDPS2: plain, finite, sphere, and mesh wall motion

RIGIDPS2 is the canonical file motion for contact-wall geometry. It stores poses, so geometry and contact velocity cannot diverge.

```text
x_world(t) = translation(t) + R(q_body_to_world(t)) * (b - reference_origin)
```

The 48-byte header is:

```text
<8sII3dII
magic="RIGIDPS2", version=2, pose_count,
reference_origin_x/y/z (float64),
out_of_range_policy (0=fatal, 1=hold), reserved=0
```

Each pose is eight float64 values:

```text
time_s, translation_x/y/z_m, quaternion_w/x/y/z
```

Times are strictly increasing. Quaternions are normalized body-to-world rotations in scalar-first order. Translation is linear and orientation uses shortest-arc SLERP. Segment linear and angular velocities are derived from adjacent poses. Fatal coverage is the default. Explicit hold freezes the endpoint pose and sets both velocities to zero.

```yaml
walls:
  - id: 1
    type: mesh              # plain, finite_plain, sphere also supported
    material_id: 0
    enabled: true
    mesh_path: /absolute/campaign/input/wall.obj
    motion_mode: rigid_pose
    trajectory_path: /absolute/campaign/input/wall.rigidps2
```

Mesh triangles and the BVH stay in the body frame; runtime contact queries transform between body and world. For a conveyor or another intentionally fixed geometry with moving surface velocity, use:

```yaml
motion_mode: surface_only
surface_velocity: [vx, vy, vz]  # world-frame m/s
```

Legacy `velocity_data_path`, `motion_data_path`, and ambiguous bare `velocity` trajectories are rejected. Convert them only after deciding whether the old intent was rigid geometry motion or surface-only motion.

## Clump inputs

Clump member particles use positive contiguous `clump_id` values in unified-v3 DEM inputs. Supply the matching runtime `clumps:` metadata described in the [YAML guide](YAML_Configuration_Guide.md). Particle preparation and assembly tools are maintained separately from this package.


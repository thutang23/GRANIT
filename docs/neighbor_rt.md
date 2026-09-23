# RT-core DEM and SPH neighbor search

GRANIT can use NVIDIA OptiX to accelerate the LBVH neighbor search for
single-GPU DEM and SPH runs with non-periodic boundaries and no shearing box.
The CUDA LBVH remains available on every CUDA GPU, including A100 and H100.
Multi-GPU runs, periodic images, and shear-box neighbor searches retain their
existing CUDA paths and restrictions.

## Selecting a backend

```yaml
simulation:
  physics: dem

neighbor:
  method: lbvh
  backend: auto
```

`neighbor.backend` accepts exactly these lowercase values:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use RT when the workload, build, driver, and device support it; otherwise use the existing CUDA backend. |
| `cuda` | Use the existing CUDA backend explicitly. |
| `rt` | Require RT. An unsupported workload or unavailable RT implementation produces an error. |

The backend option does not change `neighbor.method`. Its default remains
`cll`; select `method: lbvh` to use the RT implementation. `auto` and `cuda`
also remain valid with `cll` and `direct`.

An explicit `rt` request requires LBVH, `multigpu.enabled: false`,
`shear_box.enabled: false`, and no enabled periodic wall. Ordinary contact
walls remain supported through the existing wall contact code. Runtime also
checks the actual particle/ghost layout before enabling RT. A disabled
periodic wall does not prevent RT selection.

Hardware selection uses the OptiX RT-core capability, rather than assuming
that a CUDA compute capability implies the presence of RT cores. A100 and
H100 run the CUDA implementation under `auto`. The selected backend and any
fallback reason are reported at runtime.

## Building

`GRANIT_ENABLE_OPTIX` defaults to `OFF`. When OptiX is enabled and `OPTIX_INCLUDE_DIR` is unset,
CMake fetches NVIDIA's OptiX 8.0 headers at pinned commit
`f60c1e44f18426f426a2ed948f28515b3cf67b8a`. Supply an installed OptiX SDK's
include directory to use local headers instead:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGRANIT_ENABLE_OPTIX=ON \
  -DOPTIX_INCLUDE_DIR=/path/to/OptiX/include
cmake --build build -j
```

The header fetch needs network access on the first configuration. An invalid
explicit `OPTIX_INCLUDE_DIR` fails configuration with the missing header and
instructions to provide a valid directory or disable OptiX. The build does
not download the driver runtime or OWL. To build explicitly without OptiX or
any OptiX header fetch:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGRANIT_ENABLE_OPTIX=OFF
cmake --build build-cuda -j
```

Match CUDA code generation to the deployment GPU:

| GPU | `CMAKE_CUDA_ARCHITECTURES` | Neighbor backend with `auto` |
| --- | --- | --- |
| NVIDIA A100 | `80` | CUDA |
| GeForce RTX 4090 | `89` | RT when the optional implementation and driver are available |
| NVIDIA H100 | `90` | CUDA |

For one build targeting all three, use
`-DCMAKE_CUDA_ARCHITECTURES="80;89;90"` with a CUDA toolkit supporting those
targets. These architecture values are listed in NVIDIA's
[compute-capability table](https://developer.nvidia.com/cuda/gpus).

An RT-enabled executable loads the driver's OptiX runtime only when needed;
it does not acquire a required dynamic link to `libnvoptix`. The CUDA fallback
therefore works without an OptiX SDK or OptiX runtime installed on the target
machine. A usable NVIDIA CUDA driver is still required. `backend: rt` reports
an error when the RT runtime is unavailable.

## DEM algorithm and numerical contract

The implementation follows Mochi's sphere-to-ray reduction: place a short
ray at every particle center and traverse custom AABBs whose half-width is
twice the particle's effective search radius. RT cores perform traversal;
the custom intersection program filters candidates and the any-hit program
emits each accepted pair once. The effective radius is the existing
`support_extent` (including cohesive search range) plus half the active
Verlet skin, giving proxy half-width `2 * support_extent + skin`.

Each unordered pair is owned by the ray from the smaller effective-radius
particle, with particle index breaking equal-radius ties. If effective radii
are `rho_i <= rho_j`, then a candidate at distance
`d <= rho_i + rho_j` also satisfies `d <= 2*rho_j`, so the owner ray is covered
by the larger particle's proxy. Pair IDs are emitted in canonical order;
the order of different pairs in the output buffer is unspecified.

This ownership rule avoids an equality gap in the reference implementation's
reciprocal-coverage test. In that code, radii 1 and 2, center distance 2, and a
higher index on the smaller sphere can cause both directions to discard an
overlapping pair. GRANIT does not copy that predicate.

RT traversal uses conservatively expanded float bounds. Candidate acceptance
preserves the current CUDA LBVH's float distance/cutoff arithmetic, including
canonical index orientation, the strict `sqrtf` cutoff check, and the
`dist_sq > 1e-12` guard. The device code uses the same fast-math convention as
the CUDA implementation. This backend does not introduce a new physical
contact tolerance or change the DEM force arithmetic.

Build/refit and Verlet-list invalidation follow the existing LBVH update
frequency and displacement decisions. With `update_frequency: 1`, the
per-step rebuild path retains the existing treatment of search skin. Cached
neighbor-list reuse retains its existing rules.

Unlike the Mochi demonstration, which applies pair forces inside its OptiX
intersection program, GRANIT produces the existing pair representation and
continues through its contact ownership, history, and force kernels. Contact
forces, materials, and wall handling retain their existing implementation.

## SPH adjacency and cache lifecycle

For SPH, select `simulation.physics: sph` and `neighbor.method: lbvh` with
`backend: cuda`, `rt`, or `auto`. The optimized SPH path initially supports
one GPU with non-periodic boundaries and no shearing box. Multi-GPU SPH
continues to rebuild its CUDA spatial structure after halo exchange; its
existing CLL and periodic-boundary restrictions remain in effect.

CUDA and RT prepare a directed candidate list, stored as per-particle offsets
and neighbor indices. Each physical pair appears in both particles' rows;
the list may also include candidates within the active Verlet skin.
Density, density rate, forces, strength, kernel correction, and energy reuse
this adjacency during the same step, applying the current SPH support check
`distance < kernel_cutoff_factor * (h_i + h_j) / 2`. Self-density remains a
separate contribution. SPH does not use DEM contact history or the DEM
minimum-distance exclusion when deciding neighbor membership.

The neighbor manager controls full builds, topology refits, and candidate
reuse. A changed smoothing length conservatively invalidates cached SPH
candidates. Reuse also requires unchanged particle identity/layout and a
valid displacement budget; moving particles must not enter physical support
without appearing in the cached candidates. `update_frequency: 1` rebuilds
each step. A larger cadence permits reuse only while these checks remain
valid, and current geometry is re-evaluated by the physics kernels.

## Validation and performance

DEM correctness means agreement with the CUDA LBVH's unordered pair set for the
same particle data and search settings, plus agreement of integrated contact
results within their existing numerical tolerances. Boundary cases include
unequal radii, equal ownership thresholds, nearly coincident particles,
translated coordinates, and the cutoff boundary. Traversal and force
accumulation orders can differ, so bitwise equality of accumulated forces is
not the acceptance criterion.

Report bounds preparation, acceleration-structure build/refit, traversal and
pair production, and total neighbor-search time. Separate initialization and
warmup from steady-state samples. Include pair-buffer growth and any sorting
or conversion required by the measured path in end-to-end costs. A correct
backend with measured performance is the first milestone; a speedup depends
on particle count, radius distribution, motion, and cache reuse.

## References

- [Mochi paper, DOI 10.1145/3797905.3807836](https://doi.org/10.1145/3797905.3807836).
- [Mochi reference implementation, commit 59e7d3c320e1c8a145e8f25f43e2d9e722616b1d](https://github.com/MDurgaKeerthi/Mochi-DCD-on-RT/tree/59e7d3c320e1c8a145e8f25f43e2d9e722616b1d).
- [Reference bounds, ownership predicate, and fused contact response](https://github.com/MDurgaKeerthi/Mochi-DCD-on-RT/blob/59e7d3c320e1c8a145e8f25f43e2d9e722616b1d/src/s23-3Dparticlesim/deviceCode.cu#L14-L112).

GRANIT implements the reduction directly with OptiX. It does not vendor the
reference repository or its OWL wrapper.


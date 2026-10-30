# GRANIT: GPU-accelerated DEM and SPH

GRANIT simulates discrete particles with the discrete-element method (DEM)
and continuum materials with smoothed-particle hydrodynamics (SPH) on NVIDIA
GPUs. This package contains the production CUDA/C++ solvers, their build
dependencies, configuration references, and redistribution notices.

## Build

Use Linux, a CUDA-compatible C++17 compiler with OpenMP, CUDA Toolkit 12.x
(the original baseline is 12.1) or 13.2, and CMake 3.18 or newer. The multi-GPU executable also requires MPI development
headers and libraries. yaml-cpp is fetched from its official upstream
repository at a fixed revision. The first configuration
needs network access unless their sources are already cached.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
cmake --build build -j
```

Use a separate build directory when switching toolkits. For CUDA 13.2:

```bash
cmake -S . -B build-cuda13 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
cmake --build build-cuda13 -j
```

Default architectures include native code and PTX:

| Toolkit | Default `CMAKE_CUDA_ARCHITECTURES` | GPU support |
| --- | --- | --- |
| CUDA 12.x | `70;75;80;86;87;89;90` | Volta, Turing, Ampere (including Orin), Ada, Hopper |
| CUDA 13.2 | `75;80;86;87;89;90` | Turing, Ampere (including Orin), Ada, Hopper |

CUDA 13 removed Volta compilation support. Keep using a CUDA 12.x build for
Volta; the source tree and solver features are shared between both toolkits.
An explicit `CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` selection is preserved,
so it must be supported by the chosen toolkit. For a faster local build,
use `-DCMAKE_CUDA_ARCHITECTURES=80` on an A100 or `89` on an L40.
CUDA 12.1 cannot emit native Blackwell code; bundled PTX provides the
forward-compatibility route described by
[NVIDIA](https://docs.nvidia.com/cuda/archive/12.9.0/blackwell-compatibility-guide/index.html).
Blackwell hardware execution is not part of this checkout's local validation.
Executables are written to
`build/bin/`:

| Executable | Purpose |
| --- | --- |
| `granit_dem` | Single-GPU DEM |
| `granit_sph` | Single-GPU SPH |
| `granit_multigpu` | Supported multi-GPU DEM/SPH paths |

Set `-DGRANIT_BUILD_MULTIGPU=OFF` to omit the multi-GPU executable and MPI
dependency. Optional OptiX neighbor search is disabled by default. Enable it
with `-DGRANIT_ENABLE_OPTIX=ON`, supplying `-DOPTIX_INCLUDE_DIR=/path/to/include`
or allowing CMake to fetch the pinned NVIDIA headers. The backend requires
an appropriate NVIDIA driver.

The production precision defaults are `FMM_M2L_MODE=TF32_TC` and
`FMM_P2P_LOCAL_FP32=ON`. The available M2L modes are `FP64`, `FP32_SIMT_LINEAR`,
`FP32_SIMT_DUAL`, and `TF32_TC`; TF32 Tensor Core execution requires SM80 or
newer; SM70/75 use the FP32 SIMT fallback. Release is the default build type.
DEM P2P uses FP32 arithmetic with FP64 particle storage; multipole storage/scaling
and final accumulation remain FP64. SPH compact-softening P2P automatically
uses FP64 within the same binary to preserve cancellation accuracy. Changing precision options changes numerical
behavior; these defaults are not an accuracy or conservation guarantee.

For an offline configuration, pass existing dependency directories through
`FETCHCONTENT_SOURCE_DIR_YAML-CPP`.
OptiX can use `OPTIX_INCLUDE_DIR`. Dependency revisions and source hashes are
recorded in the historical export [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
The current Git tree records subsequent release changes; that manifest is not
a checksum inventory of the current tree.

## Configure and run

Pass a YAML configuration as the positional argument:

```bash
build/bin/granit_dem /path/to/dem.yaml
build/bin/granit_sph /path/to/sph.yaml
build/bin/granit_multigpu /path/to/multigpu.yaml
```

The files in [components/config](components/config) are configuration
references and require suitable particle inputs and paths. See the
[YAML guide](docs/YAML_Configuration_Guide.md),
[material database](docs/material_database.md),
[binary input/output formats](docs/io_formats.md), and
[SPH physics stages](docs/SPH_Physics_Stages.md).

Set the DEM timestep directly in seconds:

```yaml
simulation:
  physics: dem
  time: {start: 0.0, stop: 1.0, output_step: 0.1, dt: 1.0e-6}
```

`dt` selects the integration step; values above the computed contact or orbital
limit are rejected. SPH retains adaptive timesteps bounded by
`sph_parameters.dt_min` and `dt_max`, also in seconds. See the
[timestep and migration guide](docs/YAML_Configuration_Guide.md#timestep-in-seconds).

For Slurm pause or time-limit warnings, send `SIGUSR1` to request a checkpoint
at the next complete integration boundary and exit. See
[cooperative checkpoint stop](docs/Cooperative_Checkpoint_Stop.md) for signals,
atomic checkpoint storage, and resume instructions.

Self-gravity, prescribed inertial/tidal acceleration, and global damping are
disabled until requested. Enabling self-gravity selects unified FMM with
θ=0.8, global force/torque projection, adaptive depth, and RESPA with two inner
steps. SPH selects CFL once per macro and holds equal inner timesteps, including
pairs shortened together to land on output/stage/terminal times. Neighbor search
defaults to LBVH. The shear gravity cutoff defaults to zero (disabled); finite
shear replicas retain their projection restriction. See the
[defaults audit](docs/Configuration_Defaults.md) for the
full inventory, retained controls, removed keys, and exceptions.

Particle generation, analysis utilities, experiments, profiling campaigns, and
results remain in the development checkout. The release configuration and
G-table regression tests are available with `-DBUILD_TESTING=ON`, followed by
`cmake --build build -j` and `ctest --test-dir build --output-on-failure`.

The precomposed binary [FMM G table](data/fmm/README.md) is committed and
verified during configuration. CMake copies it into `build/share/granit/fmm/`
and installs it alongside the executables via `cmake --install build --prefix
/path/to/install`. Keep the `bin/` and `share/granit/` layout together when
moving a build or installation. Stencils/canonical tables remain in C++ headers;
normal builds need no table generation or Python.
Some source names and runtime banners retain the historical CudaDEM name.

## License

GRANIT contributions are distributed under the [BSD-3-Clause license](LICENSE).
Preserve the [third-party notices](THIRD_PARTY.md) and bundled license texts.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

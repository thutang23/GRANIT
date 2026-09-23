# GRANIT: GPU-accelerated DEM and SPH

GRANIT simulates discrete particles with the discrete-element method (DEM)
and continuum materials with smoothed-particle hydrodynamics (SPH) on NVIDIA
GPUs. This package contains the production CUDA/C++ solvers, their build
dependencies, configuration references, and redistribution notices.

## Build

Use Linux, a CUDA-compatible C++17 compiler with OpenMP, CUDA Toolkit 12.1,
and CMake 3.18 or newer. The multi-GPU executable also requires MPI development
headers and libraries. yaml-cpp and cuCollections are fetched from their
official upstream repositories at fixed revisions. The first configuration
needs network access unless their sources are already cached.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Architecture `89` targets Ada GPUs such as the L40; select the architecture
and installed CUDA compiler for your machine. Executables are written to
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

The production precision defaults are `FMM_M2L_MODE=FP64` and
`FMM_P2P_LOCAL_FP32=ON`. The available M2L modes are `FP64`, `FP32_SIMT_LINEAR`,
`FP32_SIMT_DUAL`, and `TF32_TC`; TF32 Tensor Core execution requires SM80 or
newer. Changing these options changes the numerical execution mode.

For an offline configuration, pass existing dependency directories through
`FETCHCONTENT_SOURCE_DIR_CUCOLLECTIONS` and `FETCHCONTENT_SOURCE_DIR_YAML-CPP`.
OptiX can use `OPTIX_INCLUDE_DIR`. Dependency revisions and source hashes are
recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

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

The DEM and SPH reference configurations explicitly disable self-gravity,
inertial acceleration, and tidal acceleration. They also specify valid inactive
FMM settings required by configuration validation. Enable and configure these
forces deliberately for a simulation that requires them.

Particle generation, analysis utilities, tests, experiments, profiling
campaigns, and results remain in the development checkout. This source
package does not include those tools or their datasets.

Generated FMM coefficient and stencil tables are included as C++ headers;
building the solver does not require Python or regenerating those tables.
Some source names and runtime banners retain the historical CudaDEM name.

## License

GRANIT contributions are distributed under the [BSD-3-Clause license](LICENSE).
Preserve the [third-party notices](THIRD_PARTY.md) and bundled license texts.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

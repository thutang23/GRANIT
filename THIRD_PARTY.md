# Third-party notices and scientific references

The project license covers GRANIT contributions. Copyright in third-party
material remains with its respective rights holders, and the notices below
must accompany redistribution where applicable.

## DEMBody wall implementations

The funnel and triangle-mesh wall code identifies ports from DEMBody's
`forceFunnelWalls.f90` and `forceTriMeshWalls.f90`. Related wall and periodic
boundary implementations retain their Fortran reference comments. The project
maintainer has confirmed permission to redistribute the GRANIT ports under
BSD-3-Clause. DEMBody itself is not included in this distribution; that
permission does not license or distribute the original DEMBody project.

The affected attribution includes `src/core/wall/funnel_wall.cu`,
`trimesh_wall.h`, `trimesh_wall.cu`, and `trimesh_loader.cpp`.

## CUDA Cubic B-Spline Interpolation

`src/core/math/cubic_interpolation.cuh` credits CubicInterpolationCUDA.
Copyright (c) 2008–2013 Danny Ruijters. The upstream BSD-style license and
scientific citation request are reproduced in
[licenses/CubicInterpolationCUDA.txt](licenses/CubicInterpolationCUDA.txt).

Upstream: [DannyRuijters/CubicInterpolationCUDA](https://github.com/DannyRuijters/CubicInterpolationCUDA).
License retrieved from the upstream repository on 2026-09-07.

## Build dependencies

Dependencies are installed separately or fetched into the build directory;
their source and binary distributions retain their own licenses.

- [yaml-cpp](https://github.com/jbeder/yaml-cpp): MIT.
- [cuCollections](https://github.com/NVIDIA/cuCollections): Apache-2.0;
  used only by the applicable CUDA-toolkit build path.
- [NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev): NVIDIA's
  accompanying license; this backend is optional and disabled by default.
- CUDA Toolkit components and the NVIDIA driver are distributed under their
  respective NVIDIA/component terms. OpenMP runtime terms depend on the compiler.
- [Eigen](https://libeigen.gitlab.io): MPL-2.0; used by development SPH reference tests; it is not required or bundled
  in this production package.

## Scientific references

The RT neighbor implementation cites the Mochi proxy-sphere reduction:
[Rethinking Collision Detection on GPU Ray Tracing Architecture](https://doi.org/10.1145/3797905.3807836).
SPH source and reference tests identify SPHSOL relations and numerical
conventions in their comments. Method and equation citations are retained;
a matching formula alone is not a claim that the corresponding external
software is bundled here or has been licensed by GRANIT.

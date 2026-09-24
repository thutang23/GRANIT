# Release-default validation, 2026-09-23

The requested defaults and configuration cleanup are implemented. Builds,
host regressions, coefficient reproduction, installation and short GPU runs
were checked. SPH RESPA/restart checks pass; **the separate two-GPU DEM
direct-force accuracy gate does not pass.** This
record does not certify universal force accuracy, energy conservation,
long-term integration stability, or performance.

## Build and host checks

- All three solvers (`granit_dem`, `granit_sph`, `granit_multigpu`) built with
  `/usr/local/cuda-12.1/bin/nvcc`, Release, SM89, TF32 M2L and FP32 P2P.
- A fresh configure without explicit precision/architecture/build-type flags
  selected TF32 M2L, FP32 P2P, `70;75;80;86;87;89;90` and Release.
  Tests were explicitly enabled for validation; OptiX stayed off.
- All three CTest regressions passed: configuration defaults/removed keys,
  SPH macro scheduling plus controller/calibrator continuation, and the G table.
  The SPH host test replays 40 controller epochs across live A/B trial states.
- SPH compact-softening P2P and its cancellation correction select FP64
  automatically in the production FP32-P2P build; DEM retains FP32 P2P.
- All 3,225,600 G-table entries passed reference bit preservation and independent
  TF32 rounding/padding checks. Missing/corrupt assets were rejected.
- Regenerating the binary from canonical headers produced identical bytes.
- `cmake --install` included the G table. Installed solver runs and a relocated
  standalone asset test found it from unrelated working directories.
- The constructor audit evaluates 333 members; required material/particle data
  and mode-dependent parser normalization are explained in the defaults guide.

Local build recipes used the existing pinned yaml-cpp source cache via
`FETCHCONTENT_SOURCE_DIR_YAML-CPP`. This host's MPI wrapper discovery needed
explicit `MPI_CXX_HEADER_DIR`, `MPI_CXX_ADDITIONAL_INCLUDE_DIRS`,
`MPI_CXX_LIB_NAMES=mpi_cxx;mpi`, `MPI_mpi_LIBRARY` and `MPI_mpi_cxx_LIBRARY`.
These are host setup overrides, not new GRANIT configuration options.

## DEM checks

The checked-in opt-in test is
[`gpu_smoke.py`](../components/tests/release_defaults/gpu_smoke.py). It uses
NumPy/PyYAML, requires a new output directory and two available GPUs, and
returns a failure status when any gate fails:

```bash
python3 components/tests/release_defaults/gpu_smoke.py \
  --build-dir /path/to/installed/granit \
  --output-dir /path/to/new/validation-directory --gpus 0,1
```

The final native audit used idle NVIDIA L40 GPUs 3 and 4. Each DEM case used the same 512
non-contacting particles, the public material database, eight `1e-5` s inner
steps, default G/softening/projection/RESPA=2, and output at closed boundaries.
Only the FMM depth and GPU count varied. The initial snapshot was compared to
a CPU softened direct sum. The fixed gate was relative L2 force error ≤0.002;
normalized net force and torque each had a ≤1e-6 gate.

| Case | Initial force relative L2 error | Net-force fraction | Net-torque fraction | Force accuracy |
| --- | ---: | ---: | ---: | --- |
| One GPU, automatic depth | 0.0001867970 (0.01868%) | 5.71e-17 | 2.58e-18 | Pass |
| One GPU, D8 | 0.0001867970 (0.01868%) | 9.14e-17 | 4.45e-18 | Pass |
| One GPU, D9 | 0.0001867970 (0.01868%) | 8.34e-17 | 2.22e-18 | Pass |
| Two GPUs, automatic depth | 0.0022199877 (0.22200%) | 3.66e-17 | 3.34e-18 | **Fail** |

All four DEM runs reached `8e-5` s with five finite output frames; the net-force
and net-torque projection checks passed. The small no-gravity SPH smoke used 125 particles, reached `2e-7` s with
four inner steps in two equal pairs, and produced three finite frames.
SPH gravity and restart have a separate matrix below.

The first attempt exposed a pre-existing output-ordering defect: single-GPU
RESPA wrote the initial snapshot before the outer-gravity bootstrap, producing
zero initial gravity sidecars. Moving initial output after that bootstrap fixed
it; the table above comes from the repaired, installed executables.

The two-GPU accuracy failure remains visible in `validation.json`, and the
smoke test exits 1. The threshold was not relaxed. This audit did not establish
whether the residual predates this release-default change, and does not
attribute it to TF32, the G-table packaging, or a particular distributed path.
A controlled baseline and distributed force investigation are required before
claiming that this gate passes. None of these short runs establish energy
conservation or timestep convergence.

Local final receipts are in `build-release-audit/smoke-release/`: exact
inputs/YAML, solver logs, snapshots, `validation.json`, and `provenance.json`
with binary/input/G-table SHA-256 hashes. The fresh install is
`build-release-audit/install-release/`.

## SPH RESPA, depth and restart checks

The opt-in test is
[`sph_respa_gpu.py`](../components/tests/release_defaults/sph_respa_gpu.py):

```bash
python3 components/tests/release_defaults/sph_respa_gpu.py \
  --build-dir build-release-audit \
  --output-dir /path/to/new/sph-validation-directory --gpus 3,4
```

The final matrix passed **72 checks in 50 process invocations**, including ten
expected rejections. It uses 216 perturbed SPH particles, the public material
database, L40 GPUs 3/4, and the single-GPU runner plus the multi-GPU runner with
one and two devices. It checks:

- K=1 and K=2 CFL cadence; equal inner timesteps; output, stage and terminal
  endpoints including `3.1e-7`, `4.7e-7`, `6.2e-7`, `9.3e-7`, `1.09e-6` s.
- Stage continuation, repeated same-directory continuation, preserved checkpoint
  cadence after resume, fixed/adaptive gravity continuation, and exact array
  preservation in zero-step checkpoint round trips, including fast/slow forces,
  density, stress, energy, damage, porosity and enabled histories.
- Default adaptive depth bounds, shortened trial schedules, and an empty-owner
  two-GPU case. The host replay separately checks full adaptive trial state.
- Changed RESPA cadence, truncated adaptive sidecars, missing diagnostic rows
  and infeasible smoothing bounds fail explicitly, including collective
  two-GPU rejection.
- Initial FMM gravity versus the direct SPH solver on near and spread inputs,
  using the default adaptive range 6–12. The unchanged gates are RMS error ≤1%,
  maximum particle error/reference RMS ≤5%, and normalized net force/torque
  ≤1e-6.

| Gravity fixture | Single runner | Multi runner, 1 GPU | Multi runner, 2 GPUs |
| --- | ---: | ---: | ---: |
| Near, relative RMS error | 0.03912% | 5.76e-14% | 4.97e-14% |
| Spread, relative RMS error | 0.03694% | 0.02687% | 0.25857% |

The largest maximum-particle error/reference RMS was 0.51082%. Across these
force fixtures the largest net-force fraction was `1.39e-16`, and net torque
was `9.85e-18`. Continuation uses the existing replay field tolerances
(`rtol=2e-5` plus field floors/32 float-roundoff units); zero-step round trips
add exact equality for every stored array. The near multi-GPU cases can route
entirely through direct work; their tiny residuals are not general FMM accuracy.

Receipts are in `build-release-audit/sph-respa-validated/`: all inputs,
YAML, logs, checkpoints and `report.json` with commands, binary/config hashes
and per-field verdicts. The fresh native build log is
`/tmp/granit-sph-build-cadence.log`; installation is recorded in
`/tmp/granit-release-install.log`. Generated runs/builds are excluded
from the commit.

These bounded checks do not establish long-duration energy conservation,
timestep convergence, four-GPU correctness for this revision, cross-partition
bitwise agreement, or a performance improvement. Timing-driven adaptive
choices after restart may diverge as subsequent measured timings change.

Two additional native runs in `build-release-audit/sph-respa-extra/` passed:
one-device non-Morton execution with checkpointing disabled, and a two-device
`gravity_macro_dt` run with three equal steps per macro and exact output/stop
times. Their report records binary/config hashes.


## CUDA compatibility checks

A fresh default-architecture build in `build-release-portable` compiled all
three solvers with `/usr/local/cuda-12.1/bin/nvcc`; an incremental build included
the final restart fixes. All three host tests passed. `cuobjdump --all-fatbin`
confirmed native code **and** PTX for `70,75,80,86,87,89,90` in each executable.
The binary hashes and inspection results are in
`build-release-portable/architecture-report.json`; build logs are
`/tmp/granit-portable-build.log` and `/tmp/granit-portable-final.log`.

The portable SPH force matrix passed 14 checks in eight invocations on L40
GPUs 3/4 (`build-release-portable/sph-force/report.json`). The portable DEM
smoke reproduced the native force errors and the same two-GPU gate failure
(`build-release-portable/smoke/validation.json`).

The native SM89 build was also run with `CUDA_FORCE_PTX_JIT=1` on L40 GPU 0.
All six checks in four direct/FMM invocations passed for both SPH force
fixtures, with the same errors as native execution. Receipts are in
`build-release-audit/sph-ptx-jit-long/`. The normal 180-second process timeout
expired during cold driver compilation; repeating with `--timeout 900`
completed. Accuracy gates were unchanged. This checks PTX execution on the
available Ada hardware; it does not validate Blackwell or other physical GPU
families.

## CUDA 12.1 / 13.2 compatibility, 2026-09-24

The shared source now builds with both CUDA 12.1.105 and CUDA 13.2.86.
Compatibility changes replace the removed CUB min/max functors with the same
comparison operations, replace the CUB counting iterator with Thrust's iterator,
and remove two redundant legacy texture-header includes. The texture-object
implementation, solver algorithms and precision defaults remain intact.

All three solvers were built in Release for SM80, with MPI, pinned OptiX 8.0
headers and `BUILD_TESTING=ON`. The host was Ubuntu 24.04 with four NVIDIA A100
SM80 GPUs and driver 595.91.07. CUDA 13.2 used GCC 13.3; the isolated CUDA 12.1
compiler used GCC 12.4 as its CUDA host compiler. CMake 3.31.10 drove the builds.
CUDA 12.1 components were extracted under `/tmp` from NVIDIA's redistribution
archives after checking their published SHA-256 values; the system toolkit was
not replaced.

| Check | CUDA 12.1 | CUDA 13.2 |
| --- | --- | --- |
| DEM, SPH and multi-GPU executables, including OptiX PTX | Pass | Pass |
| Existing CTest regressions | 3/3 pass | 3/3 pass |
| Full SPH RESPA / gravity / restart matrix | 72 checks, 50 invocations pass | 72 checks, 50 invocations pass |
| Cached SPH LBVH versus direct, both density modes | Pass | Pass |
| Overlapping DEM contacts: direct, LBVH and CLL | Pass | Pass |
| DEM single-GPU smoke and force gates | Pass | Pass |
| DEM two-GPU force gate | Existing failure reproduced | Same failure reproduced |

The contact fixture used 12 particles with three separated overlapping pairs,
empty owners between pairs, and four steps. LBVH and CLL matched direct-neighbor
velocities; all six isolated particles remained stationary. This exercises the
owner-offset suffix scan that the non-contacting release smoke does not cover.
The cached SPH replay used the existing `replay_sph_backends.py --skip-rt` checks.

Both toolkits produced the same two-GPU DEM relative L2 error,
`0.0022199877461741265` (0.2219988%), above the unchanged 0.2% gate. All net-force
and net-torque gates passed. The smoke command therefore exits 1 on both builds;
this compatibility change does not resolve or hide the previously recorded
accuracy failure.

Fresh CMake configurations and reconfiguration verified these defaults:

- CUDA 12.1: `70;75;80;86;87;89;90`; OptiX PTX uses `compute_70`.
- CUDA 13.2: `75;80;86;87;89;90`; OptiX PTX uses `compute_75`.
- Explicit `CMAKE_CUDA_ARCHITECTURES=80` wins over `CUDAARCHS=75`, and an
  environment-only `CUDAARCHS=75` is preserved, under both toolkits.
- A CUDA 13.2 toolchain-file selection of `86` is preserved. Explicit `70`
  fails with the compiler's unsupported-architecture error rather than being
  silently removed.
- The final CUDA 13.2 defaults also configure with the declared minimum CMake
  3.18.4. Final reconfiguration of both SM80 builds reports no work remaining.

These checks compiled the full solvers for SM80; the wider portable lists were
configuration checks, not full multi-architecture solver builds. Volta, Turing,
Ada, Hopper and Blackwell hardware execution was not tested in this audit.
The OptiX-enabled executables correctly selected CUDA fallback on A100 because
it has no RT cores. OptiX RT execution and its runtime PTX compilation still need
an RT-capable GPU. Other CUDA 12.x minor releases were not tested here.

Local, uncommitted receipts are under `/tmp/granit-build-{12.1,13.2}`,
`/tmp/granit-smoke-{12.1,13.2}/validation.json`,
`/tmp/granit-sph-respa-{12.1,13.2}/report.json`,
`/tmp/granit-sph-cache-{12.1,13.2}/result.json`, and
`/tmp/granit-contacts-{12.1,13.2}/report.json`.
The configuration matrix is `/tmp/granit-architecture-validation.json`;
build/configure logs and the isolated build environment are also under `/tmp`.

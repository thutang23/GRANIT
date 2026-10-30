# DEM timestep seconds regression

This bounded CUDA integration regression uses the existing tiny overlapping
rigid-clump fixture. It chooses an exactly representable seconds-valued timestep
above the legacy 0.2 critical-step selection and below the physical limit. It
checks exact checkpoint timestep and endpoint, active contact-history restart
parity against the uninterrupted run, and rejection of invalid or conflicting
configuration, physical/orbital excess and an explicit resume mismatch.

Run after building both solver targets:

```sh
CUDA_VISIBLE_DEVICES=0,1 uv run --with numpy --with pyyaml \
  components/tests/integration_test/timestep_seconds/validate.py \
  --single build/bin/granit_dem \
  --multi build/bin/granit_multigpu \
  --work /path/to/new-evidence-directory
```

Choose actual binary names from the build. The work directory must not exist;
keep the evidence symlink on `~/data/`. The runner invokes the single solver and
one- and two-GPU distributed solvers, writes every command/config/log, and
persists `report.json` even on failure. It launches no build or production study.
These structural and restart checks do not establish physical convergence.

## Validation on 2026-10-30

CUDA 13.4, SM80 Release builds on A100 GPUs passed 42 bounded solver launches:
18 successful launches and 24 intentional rejections with the expected error
and no new checkpoint. All three execution modes selected exactly
`dt = 0.00048828125 s`, above the old automatic step (about `0.000216498 s`)
and below the fixture's critical estimate (about `0.00108249 s`). Active contact
histories and the continued state matched the uninterrupted controls exactly.
An 8.5-step terminal time preserved the base timestep and passed a zero-step
checkpoint roundtrip. Driver queries confirmed every test solver PID had exited.

The same final build passed seven focused CTest checks and the full cooperative
stop suite: 23 native signal/stop/resume cases plus one unsupported-configuration
rejection. This does not exercise a live Slurm controller or production-size
checkpoint latency.

Local evidence is in `checkpoint-stop-evidence/timestep-seconds-final-v4/report.json`,
`timestep-signal-{dem,sph,multi}/report.json`, and `timestep-ctest.log`.
Reports retain executable/script hashes, commands, configuration hashes and logs.

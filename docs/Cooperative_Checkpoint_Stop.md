# Cooperative checkpoint stop

`granit_dem`, `granit_sph`, and supported `granit_multigpu` configurations handle
`SIGUSR1`, `SIGTERM`, and `SIGINT` as checkpoint-and-exit requests. A signal
handler only sets a lock-free flag. The solver finishes its current integration
step, synchronizes device work, commits a complete checkpoint, prints a
`[cooperative-stop]` receipt, and exits with status 0. Checkpoint failures produce
a nonzero exit status and no successful stop receipt. Normal process exit frees
the CUDA contexts and GPU allocations.

The request forces a checkpoint even when periodic checkpoints are disabled or
the current step is outside `io.checkpoint.interval`. The destination is
`io.checkpoint.path`; when omitted, it defaults to `output_directory/checkpoints`
or `./checkpoints` if the output directory is also empty. Repeated signals keep
the request set; they do not interrupt
a checkpoint write. `SIGKILL` and `SIGSTOP` cannot request a checkpoint.

With RESPA, a complete integration step means the **outer step**: all remaining
inner steps, refreshed slow force, and closing kick finish before publication.
The stop latency is therefore the remaining outer-step work plus device
synchronization and checkpoint I/O. It is independent of the periodic checkpoint
interval. Multi-GPU workers agree on the stop at a collective closed boundary;
one worker cannot leave while another waits at a barrier.

## Request and resume

For a directly launched solver:

```bash
kill -USR1 SOLVER_PID
```

Start the later job with the same physical configuration, material database,
clump metadata, and prescribed input trajectories. Select the committed checkpoint
explicitly:

```yaml
io:
  checkpoint:
    enabled: true
    interval: 10000
    path: /path/to/checkpoints
    force_fresh: false
    resume_from: /path/to/checkpoints/checkpoint_step_0000000123.bin
```

`granit_dem` also retains its automatic latest-checkpoint selection. Explicit
`resume_from` avoids accidentally selecting a later checkpoint. SPH and the
multi-GPU executable use the explicit path. Keep `rebootstrap_forces` disabled
for an ordinary continuation; it intentionally changes the force initialization
contract.

## Slurm

Run GRANIT as a job step, and let the batch script finish when the solver exits:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --signal=USR1@120
set -euo pipefail
srun /path/to/granit_dem /path/to/config.yaml
```

Request a cooperative pause manually with:

```bash
scancel --signal=USR1 JOB_ID
```

Slurm sends this signal to the job steps. The batch script must finish afterward
to release the allocation. This feature exits the solver; a later job must be
submitted to resume it. It does not automatically requeue or suspend the job.
Choose a warning interval long enough for the largest outer step and checkpoint
write on the actual storage system. An ordinary Slurm cancellation can escalate
to `SIGKILL` after the site's `KillWait` interval.

See the official [scancel documentation](https://slurm.schedmd.com/scancel.html)
and [sbatch signal option](https://slurm.schedmd.com/sbatch.html#OPT_signal).

## Atomic checkpoint layout

Each write creates a unique hidden generation in the checkpoint directory:

```text
checkpoints/
  checkpoint_step_0000000123.bin -> .checkpoint-generation-ABC123/checkpoint.bin
  .checkpoint-generation-ABC123/
    checkpoint.bin
    checkpoint.bin.wall_history
    checkpoint.bin.respa_sync        # when applicable
    ...                             # mode-specific restart state
    COMPLETE
```

The writer closes and checks every payload, flushes files and the generation
directory with `fsync`, and atomically renames a relative symlink to publish the
generation. The `COMPLETE` manifest binds all members by size and checksum.
The parent directory is then synced. A failure before publication leaves the
previous checkpoint untouched, including a previous write of the same step.
If the final directory sync fails after publication, the solver reports uncertain
durability and exits unsuccessfully without deleting the published generation.

Readers resolve the `.bin` alias **once**, verify the entire manifest, and read
all sidecars next to that resolved payload. Binary-only readers can still open
the visible `.bin` path. Tools accessing sidecars must resolve the alias first:

```python
payload = checkpoint_path.resolve(strict=True)
respa_sidecar = payload.with_name(payload.name + ".respa_sync")
```

Copy the entire checkpoint directory with hidden directories and symlinks
preserved, or copy a resolved payload together with all its sidecars and
manifest. Copying only the visible symlink or dereferencing only the binary loses
restart state. Superseded and interrupted generations remain available for
readers that already resolved them. Remove unreferenced generations only when
no writer or reader is using the directory; checkpoint retention removes visible
aliases but does not garbage-collect hidden generations.

Legacy regular v3 records remain readable. A restart requiring newly preserved
clump, bonded-wall, or YORP state rejects a legacy checkpoint that lacks that
state; the loader cannot reconstruct omitted physical memory.

## Saved state and verification

New DEM checkpoints preserve particle state, double-precision force/torque and
slow gravity caches, contact springs in particle order, and wall-contact springs
with their actual indexing. Single-GPU clumps retain authoritative body state,
and dynamic bonded walls retain pose, velocity, spin and reactions. SPH retains
its constitutive state, rates and slow gravity cache. Applicable integrator,
shear, adaptive-controller, and external-body sidecars are part of the same
atomic generation.

Distributed restart preserves the GPU count and ownership geometry. Multi-GPU
runs reject inactive input tails, dynamic bonded walls, and SPH shearing boxes
at initialization because those paths do not provide a complete supported
distributed checkpoint. A checkpoint request never labels incomplete state as
successful. Self-gravity with more than one GPU retains the existing requirement
for Morton ownership; the non-Morton adaptive restart path is tested with one GPU.

The focused filesystem and GPU payload tests are registered with CTest:

```bash
cmake --build build --target granit_checkpoint_transaction_test granit_checkpoint_payload_test granit_bonded_wall_checkpoint_test
ctest --test-dir build -R '^(checkpoint_(transaction|payload)|bonded_wall_checkpoint|cooperative_stop_harness)$' --output-on-failure
```

The signal regression creates small fresh fixtures, sends real OS signals after
observed progress, and checks off-cadence and disabled-periodic checkpoints,
closed RESPA boundaries, exact zero-step state, and resumed/uninterrupted
continuation. It requires CUDA, NumPy and PyYAML:

```bash
uv run --with numpy --with pyyaml \
  components/tests/integration_test/cooperative_stop/validate.py \
  --dem-binary build/bin/granit_dem --sph-binary build/bin/granit_sph \
  --multi-binary build/bin/granit_multigpu --ranks 1 2 \
  --work /path/to/new-test-directory --gpus 0,1
```

Add `--variants contact yorp1 yorp3 yorp-adaptive nonmorton-adaptive morton-adaptive`
to include active YORP and adaptive FMM controller cases. The two-GPU non-Morton
case checks the configuration rejection; it makes no runtime or restart claim.

These checks exercise checkpoint infrastructure and small restart fixtures;
they do not establish production-scale timing or scientific convergence.

### Validation on 2026-10-30

All three executables built with CUDA 13.4, SM80, Release, `FMM_M2L_MODE=TF32_TC`
and `FMM_P2P_LOCAL_FP32=ON`. Verification used A100 GPUs and fresh synthetic
inputs; existing GPU jobs were left running.

- Six focused CTest checks passed: transaction publication/failure, GPU payload,
  bonded-wall continuation, the Python harness, SPH RESPA, and FMM depth constraints.
- The harness's 14 host tests and 21 existing clump validation tests passed.
- Twenty-three native signal/stop/resume cases passed: seven single-GPU DEM/SPH
  cases and sixteen cases through the multi-GPU executable on one/two GPUs.
  These include active contact memory, closed RESPA state, omitted checkpoint
  paths with periodic writes disabled, YORP modes 1/3/adaptive, one-GPU non-Morton
  adaptive FMM, and two-GPU Morton adaptive FMM.
- Zero-step comparisons preserved physical fields and applicable controller
  state. The adaptive sidecar comparison excludes only the validated boolean
  `ownership.restore_morton_partition`, which changes when initializing a restart.
  Continuation comparisons passed the fixture's fixed numerical gates.
- The non-Morton capacity-growth case retained contiguous diagnostic records;
  a separate negative control rejects diagnostic sequence gaps.
- Driver queries found no remaining GPU contexts belonging to the test solvers.
  An additional case verified the two-GPU non-Morton configuration rejection.

Local evidence is linked at `checkpoint-stop-evidence/validation_summary.json`,
with the final per-case reports in `signal-single-final/report.json` and
`signal-multi-final/report.json`. It includes source/binary hashes, commands,
configuration hashes, comparisons, logs, and GPU-release checks. This validates
native OS-signal handling; a live Slurm controller/preemption test and
production-size checkpoint latency were not measured.

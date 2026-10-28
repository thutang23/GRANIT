# Distributed DEM contact-history repair

`granit_multigpu` now reclaims histories for ended contacts before allocating
records for new contacts. This includes a full row whose old contacts end during
the same force evaluation that new contacts begin. Previously, retired records
could exhaust the 16-slot row and silently discard one endpoint's spring memory,
breaking reciprocal forces. More simultaneous contacts than the configured
stride now cause a collective error before integration, output, or checkpoint.

## Implementation

The production force kernel and lifecycle helpers live in
`src/domain/distributed_contact_history.cuh`, shared with the CUDA regression.
Each force evaluation clears the old seen flags, marks current overlaps in both
owner directions, and stably compacts each local row. The overlap calculation
uses the force kernel's FP32 positions, canonical order and shear-image
transformation. It copies retained tangential, rolling and twisting springs
without integrating them.

The force passes search an immutable retained prefix. New records are appended
using a capacity-bounded atomic reservation, so other pair threads never search
partially initialized records. A failed reservation records the rank, owner and
partner global IDs, stride, and a lower bound on required capacity. All GPU
workers synchronize and take the same exception path; shutdown frees the
workspace on its allocation device. The contact law and checkpoint schema are
unchanged.

## Validation on 2026-10-28

Base commit: `055f28d6661e9895ea074f6e6c6955181e07c653`.
CUDA 13.4, NVIDIA A100-SXM4-40GB. Both FP64 FMM and default TF32/FP32
builds were compiled. These contact fixtures disable gravity.

| Check | Result |
| --- | --- |
| Frozen contact-law CPU reproduction | After 16 retired contacts, net pair force was 0.06464973 N; retaining both histories gave zero. This is a host translation, not a CUDA trajectory. |
| Production CUDA kernel | Passed 64 births, repeated partners, nine reciprocal spring components, force/torque balance, inactive and mixed full rows, same-evaluation deaths/births, FP32 overlap boundaries, shear images, and 17-contact overflow. |
| Fresh churn, 1/2 GPUs, both builds | Passed 21 distinct partners, 42 active directed records checked per run, and exactly reciprocal spring vectors. |
| Integrated conservation | Maximum absolute linear momentum drift 1.53e-16 kg m/s; angular momentum drift 7.77e-15 kg m²/s. Fixture gate: 1e-8 in each quantity's SI units. |
| Churn restart | Split at step 710 with two active directed histories and nonzero spring memory. Final meaningful state matched the uninterrupted run exactly. |
| Multi-member clump restart | Exact semantic comparison retained eight nonzero directed histories; trajectory and checkpoint comparisons passed. |
| Cross-rank transport | Logs explicitly show remote contact on both ranks and member counts changing 20→16 / 20→24. The churn fixture itself does not migrate. |
| Genuine overflow, 1/2 GPUs | Exit 1, capacity diagnostic, and no output or checkpoint files. |
| Frozen negative control | Both frozen precision builds fail the churn reciprocal-history audit at step 1010. |
| Unexhausted matched control | Fresh short trajectories match frozen binaries exactly on all checked numerical fields. |
| Harness checks | Six CPU tests pass; registered CUDA CTest passes. |

Restart comparisons ignore cached local partner indices, inactive unused slots,
and ABI padding. Kinematic gates are `atol=2e-8, rtol=2e-6`; force/torque gates
are `atol=2e-7, rtol=2e-5`. The measured churn-restart and matched-control gate
fractions were zero. The CUDA fixed-position total orbital-plus-spin torque
residual gate is `2e-7`; it is separate from integrated momentum evidence.

## Memory checks

The standalone production-kernel test passes Compute Sanitizer memcheck with
256-byte padding and full leak checking: zero errors, zero bytes leaked.
The full two-GPU executable reports 21 duplicate CUB kernel registrations at
startup under the default extended API-reporting mode. The frozen executable
reports exactly the same 21 messages. These raw logs are retained; they are
not classified as clean runs.

With `--report-api-errors explicit`, both normal completion and collective
overflow pass with zero errors and zero leaks; their process exit codes are 0
and 1 respectively. This mode checks explicit user CUDA API calls while leaving
internal runtime logging out of the API report, as described in the
[NVIDIA reporting-mode documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#reporting-modes).
No invalid memory access or leak was reported in either reporting mode.

## Reproduction and evidence

Build the solver and test using the repository's usual CMake configuration:

```bash
cmake --build /absolute/build --target granit_multigpu granit_distributed_contact_history_test -j8
ctest --test-dir /absolute/build -R '^distributed_contact_history$' --output-on-failure
python3 -m unittest discover -s components/tests/integration_test/dem_contact_history -p 'test*.py' -v
CUDA_VISIBLE_DEVICES=2,3 python3 components/tests/integration_test/dem_contact_history/validate.py \
  --binary /absolute/build/bin/granit_multigpu \
  --baseline /absolute/frozen/bin/granit_multigpu \
  --work "$HOME/data/contact-history-regression-NEW" --ranks 1 2
```

The work directory must be new. `--baseline` is optional. The report preserves
launch commands, binary/configuration hashes, physical checks, and restart
comparisons. Actual runs and frozen source/binary identities are under
`runs/contact_history_fix_20261028`, a local link to
`/home/user/data/granit-contact-history-fix-20261028-a94e`:

- `integration-01/report.json`: FP64 results.
- `integration-default/report.json`: default-precision results with explicit
  transport and overflow-artifact gates.
- `final-audit.json`: replay of those gates against the earlier FP64 run.
- `memcheck-kernel.log`, `memcheck-integration/`, `memcheck-controls/`, and
  `memcheck-explicit/`: raw and scoped sanitizer evidence.
- `frozen/identity.json`: source and binary SHA-256 identities.

This repair has not been installed into the DEM steward or used to resume
either affected scientific run. It cannot restore lost spring history or undo
altered motion in existing checkpoints. The fixtures establish the repaired
lifecycle and bounded conservation/restart behavior; they do not establish
long-duration Saturn accuracy or production performance overhead.

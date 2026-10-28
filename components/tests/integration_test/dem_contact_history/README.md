These regressions exercise the production distributed history lifecycle and force
kernel, and fresh native `granit_multigpu` trajectories. They never read scientific
campaign checkpoints or interact with the DEM controller.

See `docs/Distributed_Contact_History_Repair.md` for the measured repair campaign
and its retained sanitizer and scientific limits.

The CUDA unit `test_distributed_contact_history` includes the same production
header as `main_multigpu.cu`. It checks 64 births with 32 distinct and repeated
partners, all nine reciprocal spring components, exact retained spring values
through preparation, pair force and total orbital plus spin torque, inactive and
mixed full rows, simultaneous deaths and births, and genuine 17-neighbor overflow.
The force-state checks use fixed positions; integrated conservation belongs to
trajectory evidence, rather than being inferred from a kernel test.

The integration driver prepares a free moving radius .03 sphere and 21 separated
radius .01 spheres, each represented by a separate one-member rigid clump. Low
stiffness limits deflection; observed histories must prove more than 16 distinct
partners. Every saved active directed record must have an exactly opposite
nine-component reciprocal spring. The fixture gate fails if insufficient contacts
actually occur. A center sphere with 17 overlapping neighbors tests collective
exit code 1 on one and two GPUs. Existing clump contact/migration fixtures check
rigid ownership transport and exact semantic midpoint restart. An optional frozen
binary supplies a matched fresh short-trajectory control.

```bash
python3 -m unittest discover -s components/tests/integration_test/dem_contact_history -p test_validation.py -v
python3 components/tests/integration_test/dem_contact_history/validate.py \
  --binary /absolute/path/granit_multigpu --baseline /absolute/path/frozen/granit_multigpu \
  --work "$HOME/data/granit-contact-history-fix-20261028-a94e/campaign-NEW" --ranks 1 2
```

All destinations must be new. Reports retain exact launch commands, binary and
configuration hashes, history checks and restart comparisons. Actual rank
migration must be confirmed from ownership diagnostics; crossing a spatial cut
alone is not proof. Passing bounded fixtures does not restore missing physical
memory or corrupted velocities in historical scientific checkpoints.

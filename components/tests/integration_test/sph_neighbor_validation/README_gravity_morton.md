# SPH gravity target-order regression

`test_sph_gravity_morton.cu` calls the production
`sph_gravity_softening_correction_kernel` linked from the candidate solver's
`self_gravity.cu.o`. Build it with:

```sh
python3 components/tests/integration_test/sph_neighbor_validation/build_sph_gravity_morton_test.py \
  --build /path/to/candidate-build --out /path/to/test-output
/path/to/test-output/test_sph_gravity_morton
```

The relink replaces only the solver's main object and records source, gravity
object, and binary hashes in `build_receipt.json`. Run the binary on a CUDA GPU.
It compares canonical and Morton target scheduling bit for bit, then checks an
all-pairs CPU neighbor enumeration. The CPU reference intentionally calls the
same cubic-spline softening-law helper, so it independently tests target
coverage and cutoff selection, not the softening formula itself.

Cases include variable smoothing lengths and masses, interleaved owned and
ghost ranks, a 270-leaf permutation spanning three CUDA blocks, both inner
force and outer acceleration accumulation, ghost and inactive-buffer sentinels,
zero owned with live ghosts, one particle, and an empty no-launch domain.
The tree is hand constructed to isolate target scheduling; this binary calls
the kernel directly and does not test the host wrapper's Morton-selection
guard. The maintained SPH validation suite covers the wrapper in normal
solver execution.

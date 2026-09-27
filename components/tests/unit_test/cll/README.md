# Shear CLL contact fixture

Build `granit_cll_shear_contacts_test` with `BUILD_TESTING=ON` and
`CMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc`. Run the executable on a
CUDA node or use `ctest -R '^cll_shear_contacts$' --output-on-failure`.
The executable returns 77 when no CUDA device is available.

The fixture calls the production `NeighborSearch` CLL and LBVH paths. For each
shear case it compares sorted pair **and image** identities with a host
enumeration of all nine Lees–Edwards images. Cases cover noncommensurate box
and cell widths, X/Y/corner seams, positive and negative remapped phase,
different support and physical radius, Z rejection, same-clump rejection,
local/ghost ownership, seeded random states, output overflow, narrow grids,
growth while initialized, and concurrent per-device state on up to four GPUs.
The ordinary CLL smoke case checks the existing non-shear dispatch.

The oracle checks candidate geometry at the evaluation instant with an update
frequency of one and no Verlet shell. This fixture validates contact search;
the campaign's full solver comparison validates force and trajectory behavior.
For memory safety, run `compute-sanitizer --tool memcheck --error-exitcode 99`
against the same executable on the CUDA node.

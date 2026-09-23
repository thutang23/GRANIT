# Precomposed FMM G table

`g_table_v1.bin` contains the full 48-orientation coefficient table for the
bundled θ=0.8 stencils. It includes the four appended planar gap +1 operators.
It is generated from `g_tables.h` and `planar_gap1_operator_table.h` by the same
two double-precision matrix products previously executed at solver startup.
Coefficients are stored once as unrounded FP32. The loader retains that layout
for reference modes, or pads each matrix to 32×32 and rounds to nearest-even
TF32 for `TF32_TC`; FP64 expansion storage/scaling is unchanged.

Reproduce from the repository root (a host C++17 compiler suffices):

```bash
c++ -O2 -ffp-contract=off -std=c++17 -I. tools/generate_g_table.cpp -o /tmp/granit-g-table
/tmp/granit-g-table /tmp/g_table_v1.bin
cmp data/fmm/g_table_v1.bin /tmp/g_table_v1.bin
```

Size: 12,902,448 bytes. SHA-256:
`bb1ae2ea6eb1ec8d52057e65dc11f24832c1bba3b72e2fc9b2a8804cc28e72d5`.
CMake checks this hash. Runtime checks magic, layout, size, payload checksum,
and finite coefficients, and fails if the asset is missing or corrupt.

All integers/floats use little-endian byte order. Header:

| Byte offset | Type | Value |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `GRANITG1` |
| 8 | uint32 | Version 1 |
| 12, 16, 20 | uint32 each | 28 rows, 32 columns, 48 orientations |
| 24, 28, 32 | uint32 each | 21 gap-0, 29 gap-+1, 25 gap-−1 canonical operators |
| 36 | uint32 | 3,225,600 floats |
| 40 | uint64 | FNV-1a checksum of payload bytes |
| 48 | IEEE-754 float32 array | Gap, canonical ID, generator orientation, row, column |

The build and install layouts both resolve `bin/../share/granit/fmm/g_table_v1.bin`
from the executable (`/proc/self/exe` on Linux), independently of the current
working directory and YAML location. Relocate the complete bin/share layout.

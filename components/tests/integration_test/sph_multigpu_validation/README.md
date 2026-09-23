# SPH single and distributed consistency

From the source root, configure and build the two executables:

```bash
cmake -S . -B build-sph \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DFMM_P2P_LOCAL_FP32=OFF -DFMM_M2L_MODE=FP64 \
  -DGRANIT_ENABLE_OPTIX=OFF
cmake --build build-sph --target granit_sph granit_multigpu -j12
```

Inside an allocation exposing four CUDA devices, run:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 components/tests/integration_test/sph_multigpu_validation/run_validation.py \
  --work-dir /absolute/path/to/new-results \
  --single "$PWD/build-sph/bin/granit_sph" \
  --multi "$PWD/build-sph/bin/granit_multigpu"
```

Python requires NumPy and PyYAML. The output directory must not already exist; every input, configuration, log, checkpoint, and report is retained. The site-specific Slurm example records GPU and source receipts and caps the allocation at one hour. It does not submit itself.

Hydrodynamic field comparisons use the fixed gates in `replay_sph_backends.py`. Six history fields are also compared when present. Gravity approximation gates are separate and fixed before execution. The suite stops at the first unexpected failure; expected rejection cases require both a nonzero exit and the specified diagnostic.

The host EOS preservation probe can compare this source with a preserved upstream export using `check_eos_preservation.py`; the build argument expects the pinned dependency layout used for this integration. `test_material_database.cpp` and `test_sph_release.cpp` can be linked with the production material parser, pinned yaml-cpp, and CUDA runtime. These are host numerical checks and do not initialize CUDA devices.

This suite tests bounded numerical consistency and continuation. It does not establish physical convergence, long-duration conservation, or performance scaling. Distributed fixed/symmetry boundaries, periodic SPH, adaptive FMM depth, and macro gravity cadence remain outside the supported configuration.

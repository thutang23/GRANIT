# Peer transfer integrity check

On Linux, CUDA reporting peer access does not establish that a PCIe copy is
safe under translated IOMMU mappings. GRANIT reads each visible GPU's PCI
identity and IOMMU group type once. Generic PCIe routes with translated or
unreadable IOMMU mode use pinned host staging automatically. Routes proven by
NVML to use NVLink or NVSwitch keep direct peer copies. An absent IOMMU group
or an identity group retains CUDA's peer-copy behavior.

Configure and run the opt-in test from the repository root:

```sh
cmake -S . -B <build-dir> -DBUILD_TESTING=ON
cmake --build <build-dir> --target granit_peer_transfer_gpu_test
<build-dir>/bin/granit_peer_transfer_gpu_test
GRANIT_FORCE_HOST_STAGING=1 <build-dir>/bin/granit_peer_transfer_gpu_test
```

The test requires at least two visible GPUs and checks every ordered pair
among up to four GPUs. It tests four payload sizes and repeated copies on
alternating source and destination streams, comparing every 32-bit word.
Its default `auto` mode calls the production transfer helper. The standalone
test does not register NVML fabric evidence, so translated-IOMMU routes use
staging even when a full GRANIT run proves NVLink. The full application logs
its selected route once per pair. `bad_words` or `total_bad_words` above zero,
or a nonzero exit status, indicates a failed integrity check. This workflow
does not require changing OS IOMMU settings.

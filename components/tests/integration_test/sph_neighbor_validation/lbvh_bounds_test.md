# SPH LBVH bounds regression

From the repository root on this workspace, compile and run with:

```sh
nvcc -arch=sm_80 -std=c++17 -O3 --use_fast_math -I. \
  -I/tmp/granit-build-13.2/_deps/yaml-cpp-src/include \
  components/tests/integration_test/sph_neighbor_validation/test_lbvh_pair_bounds.cu \
  -o /tmp/granit-test-lbvh-pair-bounds
/tmp/granit-test-lbvh-pair-bounds

nvcc -arch=sm_80 -std=c++17 -O3 --use_fast_math -I. \
  -I/tmp/granit-build-13.2/_deps/yaml-cpp-src/include \
  components/tests/integration_test/sph_neighbor_validation/test_lbvh_pair_ownership.cu \
  -o /tmp/granit-test-lbvh-pair-ownership
/tmp/granit-test-lbvh-pair-ownership
```

These harnesses compile the same leaf build/refit and SPH pair query used by the solver; the bounds harness also compiles the LBVH gravity fallback. The bounds harness constructs a small valid tree topology by hand and compares GPU pair sets with an independent GPU brute-force predicate after initial build and refits. Cases cover 5:1 smoothing lengths, cutoff and skin boundaries, fast-math underflow, and a double-precision gravity pair whose float centers coincide. The gravity case uses tree support factor 3 and query factor 2. This test does not exercise the production Morton sort or tree topology builder; the solver replay covers full simulation behavior.

The ownership harness uses 96 dense particles with Morton order shuffled across owned and ghost indices. It compares exact GPU predicate pair sets for all-owned, mixed, single-owned, and zero-owned layouts. It also fills a seven-pair buffer, checks the complete count and untouched guard, then retries with full capacity and checks the pair set.

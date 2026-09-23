#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef CUDADEM_CPU_MOCK
    #include "cuda_mock.h"
    #include "thrust_mock.h"

    // Macro to launch kernels on CPU
    // Usage: LAUNCH_KERNEL(MyKernel, grid, block, shmem, stream, arg1, arg2...)
    #define LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
        cpu_launch_kernel(kernel, grid, block, __VA_ARGS__)

    // Mock Check Macro
    #define CUDA_CHECK(call) call

#else
    #include <cuda_runtime.h>

    // Standard CUDA Launch
    #define LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
        kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)

    // Error checking macro (keep your existing one or use this)
    #ifndef CUDA_CHECK
    #define CUDA_CHECK(call) { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    }
    #endif
#endif

#endif // CUDA_UTILS_H

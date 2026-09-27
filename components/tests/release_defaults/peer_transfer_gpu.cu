// Opt-in GPU integrity regression for the production peer-transfer helper.
// Build with CMake target granit_peer_transfer_gpu_test and run the executable
// with at least two visible GPUs. The default mode uses automatic route policy;
// GRANIT_FORCE_HOST_STAGING=1 also exercises the explicit staging override.
#include "src/domain/peer_transfer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

__global__ void fill(uint32_t* values, size_t count, uint32_t seed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        values[i] = (uint32_t(i) * 0x9e3779b9u) ^ seed ^
                    (uint32_t(i >> 16) * 0x85ebca6bu);
}

int main() {
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "device count");
    if (visible < 2) {
        std::fprintf(stderr, "peer transfer test needs at least two visible GPUs; found %d\n", visible);
        return 2;
    }
    const int tested = std::min(visible, 4);
    const char* forced = std::getenv("GRANIT_FORCE_HOST_STAGING");
    const char* mode = forced && forced[0] == '1' && forced[1] == '\0' ? "staged" : "auto";
    std::printf("{\"mode\":\"%s\",\"visible_devices\":%d,\"tested_devices\":%d}\n",
                mode, visible, tested);
    const size_t sizes[] = {440, 4096, 1048576, 4194304};
    unsigned long long total_bad = 0;
    for (int src = 0; src < tested; ++src) {
        for (int dst = 0; dst < tested; ++dst) {
            if (src == dst) continue;
            int accessible = 0;
            check(cudaDeviceCanAccessPeer(&accessible, dst, src), "peer access query");
            for (size_t bytes : sizes) {
                const size_t words = bytes / sizeof(uint32_t);
                check(cudaSetDevice(src), "select source");
                uint32_t* source = nullptr;
                cudaStream_t source_streams[2] = {};
                cudaEvent_t ready[2] = {};
                check(cudaMalloc(&source, bytes), "allocate source");
                for (int lane = 0; lane < 2; ++lane) {
                    check(cudaStreamCreateWithFlags(&source_streams[lane], cudaStreamNonBlocking),
                          "source stream");
                    check(cudaEventCreateWithFlags(&ready[lane], cudaEventDisableTiming),
                          "source ready event");
                }
                check(cudaSetDevice(dst), "select destination");
                uint32_t* destination = nullptr;
                cudaStream_t destination_streams[2] = {};
                check(cudaMalloc(&destination, bytes), "allocate destination");
                for (cudaStream_t& stream : destination_streams)
                    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                          "destination stream");

                std::vector<uint32_t> observed(words);
                unsigned long long bad = 0;
                long long first_bad = -1;
                for (int repeat = 0; repeat < 4; ++repeat) {
                    const int lane = repeat & 1;
                    const uint32_t seed = 0xa5c31e67u ^ (uint32_t(src) << 24) ^
                                          (uint32_t(dst) << 16) ^ uint32_t(repeat * 0x12345u);
                    check(cudaSetDevice(src), "select fill source");
                    fill<<<unsigned((words + 255) / 256), 256, 0, source_streams[lane]>>>(
                        source, words, seed);
                    check(cudaGetLastError(), "fill launch");
                    check(cudaEventRecord(ready[lane], source_streams[lane]), "record source ready");
                    check(cudaSetDevice(dst), "select copy destination");
                    check(cudaStreamWaitEvent(destination_streams[lane], ready[lane], 0),
                          "wait source ready");
                    try {
                        cudadem::copy_peer_payload(destination, dst, source, src, bytes,
                                                   destination_streams[lane]);
                    } catch (const std::exception& error) {
                        std::fprintf(stderr, "peer transfer %d -> %d: %s\n",
                                     src, dst, error.what());
                        return 2;
                    }
                    check(cudaMemcpyAsync(observed.data(), destination, bytes,
                                          cudaMemcpyDeviceToHost, destination_streams[lane]),
                          "copy observed");
                    check(cudaStreamSynchronize(destination_streams[lane]), "wait destination");
                    for (size_t i = 0; i < words; ++i) {
                        const uint32_t expected = (uint32_t(i) * 0x9e3779b9u) ^ seed ^
                                                  (uint32_t(i >> 16) * 0x85ebca6bu);
                        if (observed[i] != expected) {
                            ++bad;
                            if (first_bad < 0) first_bad = static_cast<long long>(i);
                        }
                    }
                }
                total_bad += bad;
                std::printf("{\"mode\":\"%s\",\"src\":%d,\"dst\":%d,\"peer_access\":%d,\"bytes\":%zu,\"repeats\":4,\"bad_words\":%llu,\"first_bad_word\":%lld}\n",
                            mode, src, dst, accessible, bytes, bad, first_bad);
                std::fflush(stdout);

                check(cudaSetDevice(dst), "cleanup destination");
                for (cudaStream_t stream : destination_streams)
                    check(cudaStreamDestroy(stream), "destroy destination stream");
                check(cudaFree(destination), "free destination");
                check(cudaSetDevice(src), "cleanup source");
                for (int lane = 0; lane < 2; ++lane) {
                    check(cudaEventDestroy(ready[lane]), "destroy ready event");
                    check(cudaStreamDestroy(source_streams[lane]), "destroy source stream");
                }
                check(cudaFree(source), "free source");
            }
        }
    }
    std::printf("{\"mode\":\"%s\",\"total_bad_words\":%llu}\n", mode, total_bad);
    return total_bad ? 1 : 0;
}

#include "src/domain/peer_transfer.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

void check(cudaError_t error, const char* operation) {
    cudadem::peer_transfer_check(error, operation);
}

void copy_case(int destination_device, int source_device, int value) {
    constexpr size_t count = 16;
    std::array<int, count> input{};
    std::array<int, count> output{};
    for (size_t i = 0; i < count; ++i) input[i] = value + static_cast<int>(i);

    // The prepare helper must restore a worker's previous selected device.
    check(cudaSetDevice(source_device), "select test source");
    cudadem::prepare_peer_payload_route(destination_device, source_device);
    cudadem::prepare_peer_payload_route(source_device, destination_device);
    int selected = -1;
    check(cudaGetDevice(&selected), "read selected device");
    if (selected != source_device)
        throw std::runtime_error("peer route preparation changed current device");

    int* source = nullptr;
    int* destination = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaMalloc(&source, sizeof(input)), "allocate source");
    check(cudaMemcpy(source, input.data(), sizeof(input), cudaMemcpyHostToDevice),
          "upload source");
    check(cudaSetDevice(destination_device), "select test destination");
    check(cudaMalloc(&destination, sizeof(output)), "allocate destination");
    check(cudaStreamCreate(&stream), "create destination stream");
    cudadem::copy_peer_payload(destination, destination_device, source,
                               source_device, sizeof(input), stream);
    // A second transfer must reuse the prepared route and staging resources.
    cudadem::copy_peer_payload(destination, destination_device, source,
                               source_device, sizeof(input), stream);
    check(cudaStreamSynchronize(stream), "complete peer copy");
    check(cudaMemcpy(output.data(), destination, sizeof(output),
                     cudaMemcpyDeviceToHost), "download destination");
    if (output != input)
        throw std::runtime_error("peer payload does not match source");

    std::array<int, count> reverse_input{};
    for (size_t i = 0; i < count; ++i)
        reverse_input[i] = value + 1000 + static_cast<int>(i);
    check(cudaMemcpy(destination, reverse_input.data(), sizeof(reverse_input),
                     cudaMemcpyHostToDevice), "upload reverse source");
    check(cudaSetDevice(source_device), "select reverse destination");
    cudaStream_t reverse_stream = nullptr;
    check(cudaStreamCreate(&reverse_stream), "create reverse stream");
    cudadem::copy_peer_payload(source, source_device, destination,
                               destination_device, sizeof(reverse_input),
                               reverse_stream);
    check(cudaStreamSynchronize(reverse_stream), "complete reverse peer copy");
    output.fill(0);
    check(cudaMemcpy(output.data(), source, sizeof(output),
                     cudaMemcpyDeviceToHost), "download reverse destination");
    if (output != reverse_input)
        throw std::runtime_error("reverse peer payload does not match source");

    check(cudaStreamDestroy(reverse_stream), "destroy reverse stream");
    check(cudaSetDevice(destination_device), "select forward destination");
    check(cudaStreamDestroy(stream), "destroy destination stream");
    check(cudaFree(destination), "release destination");
    check(cudaSetDevice(source_device), "restore test source");
    check(cudaFree(source), "release source");
}

} // namespace

int main() {
    int devices = 0;
    const cudaError_t count_result = cudaGetDeviceCount(&devices);
    if (count_result == cudaErrorNoDevice ||
        count_result == cudaErrorInsufficientDriver ||
        count_result == cudaErrorInitializationError) {
        std::cout << "SKIP: CUDA devices unavailable on this node\n";
        return 77;
    }
    check(count_result, "count CUDA devices");
    if (devices < 2) {
        std::cout << "SKIP: peer transfer test requires two CUDA devices\n";
        return 77;
    }

    std::exception_ptr first_failure;
    std::exception_ptr second_failure;
    std::thread first([&] {
        try { copy_case(0, 1, 100); }
        catch (...) { first_failure = std::current_exception(); }
    });
    std::thread second([&] {
        try { copy_case(1, 0, 200); }
        catch (...) { second_failure = std::current_exception(); }
    });
    first.join();
    second.join();
    if (first_failure) std::rethrow_exception(first_failure);
    if (second_failure) std::rethrow_exception(second_failure);

    setenv("GRANIT_FORCE_HOST_STAGING", "1", 1);
    std::exception_ptr staging_failure;
    std::thread staged([&] {
        try { copy_case(0, 1, 300); }
        catch (...) { staging_failure = std::current_exception(); }
    });
    staged.join();
    unsetenv("GRANIT_FORCE_HOST_STAGING");
    if (staging_failure) std::rethrow_exception(staging_failure);
    std::cout << "peer transfer direct and staged routes passed\n";
}

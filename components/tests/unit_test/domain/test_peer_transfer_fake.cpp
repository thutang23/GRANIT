#include "src/domain/peer_transfer.h"

#include <array>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using cudadem::PeerTransfer;

#define REQUIRE(condition) do { \
    if (!(condition)) throw std::runtime_error( \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
        ": failed: " #condition); \
} while (false)

static void reset_route_policy() {
    auto& policy = cudadem::peer_transport_detail::state();
    std::lock_guard<std::mutex> guard(policy.mutex);
    policy.pci_ids.clear();
    policy.iommu_modes.clear();
    policy.fabrics.clear();
    policy.logged_routes.clear();
    // Fake PCI identities do not exist in /sys. Supply deterministic modes
    // instead of depending on the host running these unit tests.
    for (int device = 0; device < 6; ++device) {
        char id[32];
        REQUIRE(cudaDeviceGetPCIBusId(id, sizeof(id), device) == cudaSuccess);
        policy.iommu_modes[id] = cudadem::IommuMode::Absent;
    }
}

static void test_prepared_direct_route() {
    fake_cuda::reset();
    reset_route_policy();
    cudaSetDevice(3);
    PeerTransfer transfer;
    transfer.prepare(0, 1);
    transfer.prepare(0, 1);
    int selected = -1;
    cudaGetDevice(&selected);
    REQUIRE(selected == 3);
    REQUIRE(fake_cuda::can_access_calls == 1);
    REQUIRE(fake_cuda::enable_calls == 1);

    const std::array<int, 4> source{{2, 3, 5, 7}};
    std::array<int, 4> destination{};
    transfer.copy(destination.data(), 0, source.data(), 1,
                  sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::peer_copies == 1);
    REQUIRE(fake_cuda::can_access_calls == 1);
    REQUIRE(fake_cuda::enable_calls == 1);
}

static void test_forced_staging_and_override_change() {
    fake_cuda::reset();
    reset_route_policy();
    setenv("GRANIT_FORCE_HOST_STAGING", "1", 1);
    cudaSetDevice(4);
    PeerTransfer transfer;
    transfer.prepare(0, 1);
    int selected = -1;
    cudaGetDevice(&selected);
    REQUIRE(selected == 4);
    REQUIRE(fake_cuda::can_access_calls == 1);
    REQUIRE(fake_cuda::enable_calls == 0);

    const std::array<int, 3> source{{11, 13, 17}};
    std::array<int, 3> destination{};
    transfer.copy(destination.data(), 0, source.data(), 1,
                  sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::staged_d2h == 1);
    REQUIRE(fake_cuda::staged_h2d == 1);
    REQUIRE(fake_cuda::peer_copies == 0);

    unsetenv("GRANIT_FORCE_HOST_STAGING");
    destination.fill(0);
    transfer.copy(destination.data(), 0, source.data(), 1,
                  sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::peer_copies == 1);
    REQUIRE(fake_cuda::enable_calls == 1);
    REQUIRE(fake_cuda::can_access_calls == 1);
}

static void test_translated_route_then_fabric_registration() {
    fake_cuda::reset();
    reset_route_policy();
    auto& policy = cudadem::peer_transport_detail::state();
    {
        std::lock_guard<std::mutex> guard(policy.mutex);
        policy.iommu_modes["0000:00:00.0"] = cudadem::IommuMode::Translated;
    }
    PeerTransfer transfer;
    cudaSetDevice(3);
    transfer.prepare(0, 1);
    int selected = -1;
    cudaGetDevice(&selected);
    REQUIRE(selected == 3);
    REQUIRE(fake_cuda::can_access_calls == 1);
    REQUIRE(fake_cuda::enable_calls == 0);

    const int source = 43;
    int destination = 0;
    transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::staged_d2h == 1 && fake_cuda::peer_copies == 0);

    // Topology discovery may finish after route preparation. Its fabric
    // evidence must take effect without re-querying CUDA accessibility.
    cudadem::register_peer_link_class("0000:00:00.0", "0000:01:00.0",
                                      cudadem::LinkClass::DirectNvLink);
    destination = 0;
    transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::peer_copies == 1 && fake_cuda::enable_calls == 1);

    setenv("GRANIT_FORCE_HOST_STAGING", "1", 1);
    destination = 0;
    transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::staged_d2h == 2 && fake_cuda::staged_h2d == 2);
    unsetenv("GRANIT_FORCE_HOST_STAGING");
    destination = 0;
    transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::peer_copies == 2);
    REQUIRE(fake_cuda::can_access_calls == 1 && fake_cuda::enable_calls == 1);
}

static void test_mapping_fallback(cudaError_t status) {
    fake_cuda::reset();
    reset_route_policy();
    fake_cuda::enable_result = status;
    PeerTransfer transfer;
    transfer.prepare(0, 1);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    transfer.prepare(0, 1);

    const std::array<int, 2> source{{19, 23}};
    std::array<int, 2> destination{};
    transfer.copy(destination.data(), 0, source.data(), 1,
                  sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::enable_calls == 1);
    REQUIRE(fake_cuda::can_access_calls == 1);
    REQUIRE(fake_cuda::peer_copies == 0);
    REQUIRE(fake_cuda::staged_d2h == 1);
    REQUIRE(fake_cuda::staged_h2d == 1);
}

static void test_already_enabled_and_unavailable() {
    fake_cuda::reset();
    reset_route_policy();
    fake_cuda::enable_result = cudaErrorPeerAccessAlreadyEnabled;
    {
        PeerTransfer transfer;
        transfer.prepare(0, 1);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const int source = 29;
        int destination = 0;
        transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
        REQUIRE(destination == source && fake_cuda::peer_copies == 1);
    }

    fake_cuda::reset();
    reset_route_policy();
    fake_cuda::accessible = 0;
    PeerTransfer transfer;
    transfer.prepare(0, 1);
    const int source = 31;
    int destination = 0;
    transfer.copy(&destination, 0, &source, 1, sizeof(source), nullptr);
    REQUIRE(destination == source);
    REQUIRE(fake_cuda::enable_calls == 0 && fake_cuda::staged_d2h == 1);
}

static void test_prepare_restores_device_on_error() {
    fake_cuda::reset();
    reset_route_policy();
    fake_cuda::enable_result = cudaErrorUnknown;
    cudaSetDevice(5);
    PeerTransfer transfer;
    bool threw = false;
    try { transfer.prepare(0, 1); }
    catch (const std::runtime_error&) { threw = true; }
    int selected = -1;
    cudaGetDevice(&selected);
    REQUIRE(threw && selected == 5);
}

static void test_function_helpers_share_worker_local_cache() {
    fake_cuda::reset();
    reset_route_policy();
    auto worker = [] {
        cudaSetDevice(2);
        cudadem::prepare_peer_payload_route(0, 1);
        int selected = -1;
        cudaGetDevice(&selected);
        REQUIRE(selected == 2);
        const int source = 37;
        int destination = 0;
        cudadem::copy_peer_payload(&destination, 0, &source, 1,
                                   sizeof(source), nullptr);
        REQUIRE(destination == source);
    };
    std::thread first(worker);
    first.join();
    std::thread second(worker);
    second.join();
    REQUIRE(fake_cuda::can_access_calls == 2);
    REQUIRE(fake_cuda::enable_calls == 2);
    REQUIRE(fake_cuda::peer_copies == 2);
}

static void test_own_and_reverse_routes_are_cached() {
    fake_cuda::reset();
    reset_route_policy();
    std::thread worker([] {
        cudaSetDevice(0);
        // First pass: this worker owns destination 0. Second pass: its
        // source GPU 0 sends direct flags into remote destinations.
        cudadem::prepare_peer_payload_route(0, 1);
        cudadem::prepare_peer_payload_route(0, 2);
        cudadem::prepare_peer_payload_route(1, 0);
        cudadem::prepare_peer_payload_route(2, 0);
        int selected = -1;
        cudaGetDevice(&selected);
        REQUIRE(selected == 0);
        REQUIRE(fake_cuda::can_access_calls == 4);
        REQUIRE(fake_cuda::enable_calls == 4);

        const int source = 41;
        int destination = 0;
        cudadem::copy_peer_payload(&destination, 0, &source, 1,
                                   sizeof(source), nullptr);
        REQUIRE(destination == source);
        destination = 0;
        cudadem::copy_peer_payload(&destination, 1, &source, 0,
                                   sizeof(source), nullptr);
        REQUIRE(destination == source);
        destination = 0;
        cudadem::copy_peer_payload(&destination, 2, &source, 0,
                                   sizeof(source), nullptr);
        REQUIRE(destination == source);
        REQUIRE(fake_cuda::can_access_calls == 4);
        REQUIRE(fake_cuda::enable_calls == 4);
        REQUIRE(fake_cuda::peer_copies == 3);
    });
    worker.join();
}

int main() {
    test_prepared_direct_route();
    test_forced_staging_and_override_change();
    test_translated_route_then_fabric_registration();
    test_mapping_fallback(cudaErrorTooManyPeers);
    test_mapping_fallback(cudaErrorPeerAccessUnsupported);
    test_already_enabled_and_unavailable();
    test_prepare_restores_device_on_error();
    test_function_helpers_share_worker_local_cache();
    test_own_and_reverse_routes_are_cached();
}

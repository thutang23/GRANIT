#include "../../../../src/core/neighbor/neighbor_api.h"
#include "../../../../src/core/wall/wall.h"
#include "../../../../src/core/shear/shear_box.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace cudadem;

namespace {

struct Point { double x, y, z; float support; int32_t clump = 0; float radius = -1; };
using Key = std::tuple<int32_t, int32_t, int, int, int>;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
Key key(const ContactPair& p) {
    return {p.i, p.j, p.image_x, p.image_y, p.image_z};
}

std::vector<Key> oracle(const std::vector<Point>& points, int local,
                        const SimConfig& config, double time) {
    const double lx = config.domain.box_max.x - config.domain.box_min.x;
    const double ly = config.domain.box_max.y - config.domain.box_min.y;
    // Derive the phase independently from the image helper used by the device.
    const long double full = static_cast<long double>(config.shear_box.shear_parameter) *
        config.shear_box.omega * lx * (time - config.shear_box.time_origin);
    // The canonical centered interval is [-Ly/2, Ly/2), including the
    // positive half-cycle tie, rather than std::remainder's ties-to-even rule.
    const long double cycle = std::floor(full / ly + 0.5L);
    const double shift = static_cast<double>(full - cycle * ly);
    std::vector<Key> result;
    for (int i = 0; i < local; ++i) for (int j = i + 1; j < static_cast<int>(points.size()); ++j) {
        if (points[i].clump > 0 && points[i].clump == points[j].clump) continue;
        const double limit = points[i].support + points[j].support;
        for (int ix = -1; ix <= 1; ++ix) for (int iy = -1; iy <= 1; ++iy) {
            const double dx = points[j].x + ix * lx - points[i].x;
            const double dy = points[j].y + iy * ly - ix * shift - points[i].y;
            const double dz = points[j].z - points[i].z;
            const double distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 > 1e-12 && distance2 < limit * limit)
                result.emplace_back(i, j, ix, iy, 0);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

template <typename T> void copy(T* destination, const std::vector<T>& source) {
    check(cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

void load(ParticleSoA& particles, const std::vector<Point>& points, int local) {
    const int n = static_cast<int>(points.size());
    allocate_particle_soa(particles, n);
    particles.num_local = local;
    particles.num_ghost = n - local;
    std::vector<double> x, y, z;
    std::vector<float> r, support;
    std::vector<int32_t> clump;
    for (const auto& p : points) {
        x.push_back(p.x); y.push_back(p.y); z.push_back(p.z);
        support.push_back(p.support);
        r.push_back(p.radius > 0 ? p.radius : p.support);
        clump.push_back(p.clump);
    }
    copy(particles.pos_x, x); copy(particles.pos_y, y); copy(particles.pos_z, z);
    copy(particles.last_pos_x, x); copy(particles.last_pos_y, y); copy(particles.last_pos_z, z);
    copy(particles.radius, r); copy(particles.support_extent, support);
    copy(particles.clump_id, clump);
}

std::vector<Key> query(SimulationState& state, const SimConfig& config,
                       ContactListBuffer& contacts, double time) {
    NeighborSearch::find_contacts(state, config, contacts, time);
    check(cudaDeviceSynchronize());
    int count = NeighborSearch::get_contact_count(contacts);
    require(count >= 0 && count <= contacts.max_contacts, "contact buffer was not grown safely");
    std::vector<ContactPair> raw(count);
    if (count) check(cudaMemcpy(raw.data(), contacts.pairs,
                                count * sizeof(ContactPair), cudaMemcpyDeviceToHost));
    std::vector<Key> result;
    for (const auto& p : raw) result.push_back(key(p));
    std::sort(result.begin(), result.end());
    require(std::adjacent_find(result.begin(), result.end()) == result.end(),
            "duplicate pair and image identity");
    return result;
}

std::vector<Key> run_backend(SimulationState& state, SimConfig config,
                             NeighborMethod method, double time, int initial_capacity) {
    config.neighbor.method = method;
    ContactListBuffer contacts;
    NeighborSearch::allocate_contact_buffer(contacts, initial_capacity);
    NeighborSearch::initialize(config, state);
    auto result = query(state, config, contacts, time);
    NeighborSearch::shutdown(state);
    NeighborSearch::free_contact_buffer(contacts);
    return result;
}

void compare(const char* name, const std::vector<Point>& points, int local,
             double lx, double ly, double time, int initial_capacity = 2,
             bool expect_contact = true) {
    SimConfig config;
    config.shear_box.enabled = true;
    config.shear_box.omega = 1.25;
    config.shear_box.shear_parameter = 1.5;
    config.shear_box.time_origin = 0.17;
    config.domain.box_min = Vec3f(0, 0, -1);
    config.domain.box_max = Vec3f(lx, ly, 1);
    config.domain.auto_bounds = false;
    config.particle.radius_max = 0.34f;
    config.neighbor.update_frequency = 1;
    config.neighbor.verlet_buffer_enabled = false;
    config.neighbor.cell_size_factor = 1.1f;
    config.gpu.block_size = 128;

    const auto expected = oracle(points, local, config, time);
    require(expected.empty() != expect_contact,
            std::string(name) + ": fixture contact expectation is wrong");
    SimulationState state;
    load(state.particles, points, local);
    WallManager::initialize(state, config);
    const auto cll = run_backend(state, config, NeighborMethod::CELL_LINKED_LIST,
                                 time, initial_capacity);
    const auto lbvh = run_backend(state, config, NeighborMethod::LBVH,
                                  time, initial_capacity);
    WallManager::shutdown();
    free_wall_contact_history(state.wall_contact_history);
    free_particle_soa(state.particles);
    if (cll != expected || lbvh != expected) {
        std::fprintf(stderr, "%s: oracle=%zu CLL=%zu LBVH=%zu\n", name,
                     expected.size(), cll.size(), lbvh.size());
        for (const auto& k : expected) if (!std::binary_search(cll.begin(), cll.end(), k))
            std::fprintf(stderr, "  CLL missing (%d,%d,%d,%d,%d)\n",
                         std::get<0>(k), std::get<1>(k), std::get<2>(k),
                         std::get<3>(k), std::get<4>(k));
        throw std::runtime_error(std::string(name) + ": pair/image mismatch");
    }
    std::printf("PASS %-22s %zu pairs\n", name, expected.size());
}

void ordinary_smoke() {
    SimConfig config;
    config.neighbor.method = NeighborMethod::CELL_LINKED_LIST;
    config.neighbor.update_frequency = 1;
    config.neighbor.verlet_buffer_enabled = false;
    config.particle.radius_max = .34f;
    config.domain.box_min = Vec3f(0, 0, -1);
    config.domain.box_max = Vec3f(7.3f, 8.7f, 1);
    SimulationState state;
    load(state.particles, {{2, 2, 0, .34f}, {2.2, 2, 0, .34f},
                           {5, 5, 0, .34f}}, 3);
    const auto result = run_backend(state, config, NeighborMethod::CELL_LINKED_LIST, 0, 4);
    free_particle_soa(state.particles);
    require(result.size() == 1, "ordinary CLL regression produced wrong pair count");
    std::puts("PASS ordinary CLL smoke");
}

void growth_without_reinit() {
    SimConfig config;
    config.shear_box.enabled = true;
    config.shear_box.omega = 1.25;
    config.shear_box.shear_parameter = 1.5;
    config.domain.box_min = Vec3f(0, 0, -1);
    config.domain.box_max = Vec3f(7.3f, 8.7f, 1);
    config.particle.radius_max = .34f;
    config.neighbor.method = NeighborMethod::CELL_LINKED_LIST;
    config.neighbor.update_frequency = 1;
    config.neighbor.verlet_buffer_enabled = false;
    SimulationState state;
    const std::vector<Point> small{{2, 2, 0, .34f}, {2.2, 2, 0, .34f}};
    load(state.particles, small, 2);
    WallManager::initialize(state, config);
    ContactListBuffer contacts;
    NeighborSearch::allocate_contact_buffer(contacts, 1);
    NeighborSearch::initialize(config, state);
    require(query(state, config, contacts, 0) == oracle(small, 2, config, 0),
            "CLL initial query failed");
    free_particle_soa(state.particles);
    std::vector<Point> large;
    large.push_back({2, 2, 0, .34f});
    large.push_back({2.2, 2, 0, .34f});
    for (int i = 2; i < 1100; ++i)
        large.push_back({5.0 + 2.0 * (i % 40), 5.0 + 2.0 * (i / 40),
                         .01 * (i % 3), .34f});
    // Grow the box too: a sparse 1100-particle state keeps this test cheap
    // under compute-sanitizer while exceeding the initial 1024-slot workspace.
    config.domain.box_max = Vec3f(100, 100, 1);
    WallManager::shutdown();
    free_wall_contact_history(state.wall_contact_history);
    load(state.particles, large, 823);
    WallManager::initialize(state, config);
    require(query(state, config, contacts, 0) == oracle(large, 823, config, 0),
            "CLL capacity growth failed");
    large[0].support = .05f;
    large[1].support = .05f;
    check(cudaMemcpy(state.particles.support_extent, &large[0].support,
                     sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(state.particles.support_extent + 1, &large[1].support,
                     sizeof(float), cudaMemcpyHostToDevice));
    require(query(state, config, contacts, 0) == oracle(large, 823, config, 0),
            "CLL did not update support at unchanged capacity");
    NeighborSearch::shutdown(state);
    NeighborSearch::free_contact_buffer(contacts);
    WallManager::shutdown();
    free_wall_contact_history(state.wall_contact_history);
    free_particle_soa(state.particles);
    std::puts("PASS CLL growth without reinit");
}

void concurrent_devices() {
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    if (devices < 2) return;
    const int workers_count = std::min(devices, 4);
    SimConfig config;
    config.shear_box.enabled = true;
    config.shear_box.omega = 1.25;
    config.shear_box.shear_parameter = 1.5;
    config.domain.box_min = Vec3f(0, 0, -1);
    config.domain.box_max = Vec3f(7.3f, 8.7f, 1);
    config.particle.radius_max = .34f;
    config.neighbor.update_frequency = 1;
    config.neighbor.verlet_buffer_enabled = false;
    const std::vector<Point> points{{2, 2, 0, .34f}, {2.2, 2, 0, .34f},
                                    {2.35, 2, 0, .34f}, {6, 6, 0, .34f}};
    SimulationState wall_state;
    WallManager::initialize(wall_state, config);
    const auto expected = oracle(points, 3, config, 0);
    std::vector<std::exception_ptr> errors(workers_count);
    std::vector<std::thread> workers(workers_count);
    for (int gpu = 0; gpu < workers_count; ++gpu) {
        workers[gpu] = std::thread([&, gpu] {
            try {
                check(cudaSetDevice(gpu));
                WallManager::copy_to_current_device();
                SimulationState state;
                load(state.particles, points, 3);
                const auto result = run_backend(state, config,
                    NeighborMethod::CELL_LINKED_LIST, 0, 1);
                free_particle_soa(state.particles);
                require(result == expected, "per-GPU CLL result differs from oracle");
            } catch (...) { errors[gpu] = std::current_exception(); }
        });
    }
    for (auto& worker : workers) worker.join();
    check(cudaSetDevice(0));
    WallManager::shutdown();
    for (auto& error : errors) if (error) std::rethrow_exception(error);
    std::printf("PASS concurrent %d-GPU CLL\n", workers_count);
}

} // namespace

int main() {
    int devices = 0;
    const auto discovery = cudaGetDeviceCount(&devices);
    if (discovery == cudaErrorNoDevice || discovery == cudaErrorInsufficientDriver ||
        discovery == cudaErrorInitializationError || (discovery == cudaSuccess && devices == 0)) {
        std::fprintf(stderr, "SKIP shear CLL contacts: no usable CUDA device\n");
        return 77;
    }
    check(discovery);
    check(cudaSetDevice(0));
    try {
        // 7.3/8.7 are deliberately not multiples of the 0.748 m cell width.
        compare("x seam and y shift", {{0.08, 2.0, 0, .34f},
                                       {7.22, 2.03, 0, .34f}}, 2, 7.3, 8.7, .17);
        compare("y seam", {{3.1, .08, 0, .34f},
                           {3.12, 8.62, 0, .34f}}, 2, 7.3, 8.7, .17);
        compare("corner phase", {{.08, .08, 0, .34f},
                                {7.22, 8.62, 0, .34f}}, 2, 7.3, 8.7, .21);
        compare("phase remap positive", {{.08, 1.1, 0, .34f},
                                        {7.22, 4.5125, 0, .34f}}, 2, 7.3, 8.7, 4.37);
        compare("phase remap negative", {{.08, 4.0, 0, .34f},
                                        {7.22, .5875, 0, .34f}}, 2, 7.3, 8.7, -4.03);
        compare("local ghost ownership", {{2, 2, 0, .34f},
                                          {2.3, 2, 0, .34f},
                                          {2.1, 2.1, 0, .34f},
                                          {2.25, 2.1, 0, .34f}}, 2, 7.3, 8.7, .17);
        compare("z and same clump", {{3, 3, 0, .34f, 4},
                                     {3.1, 3, 0, .34f, 4},
                                     {3.2, 3, .85, .34f},
                                     {3.25, 3, .02, .34f}}, 4, 7.3, 8.7, .17);
        compare("support exceeds radius", {{3, 3, 0, .34f, 0, .10f},
                                           {3.55, 3, 0, .34f, 0, .10f}},
                2, 7.3, 8.7, .17);
        compare("distant empty", {{1, 1, 0, .34f}, {4, 5, 0, .34f}},
                2, 7.3, 8.7, .17, 2, false);
        compare("all ghost", {{2, 2, 0, .34f}, {2.2, 2, 0, .34f}},
                0, 7.3, 8.7, .17, 2, false);
        compare("wide cell axis", {{.08, 2, 0, .34f},
                                   {3235.92, 2, 0, .34f}},
                2, 3236.0, 8.7, .17);
        std::mt19937 generator(7531);
        std::uniform_real_distribution<double> xy(.02, 7.28);
        std::uniform_real_distribution<double> z(-.4, .4);
        std::vector<Point> random;
        for (int i = 0; i < 192; ++i)
            random.push_back({xy(generator), xy(generator) * 8.7 / 7.3,
                              z(generator), .19f + .03f * (i % 4)});
        compare("seeded random", random, 137, 7.3, 8.7, .42);
        std::vector<Point> dense;
        for (int i = 0; i < 48; ++i)
            dense.push_back({3.0 + .02 * (i % 8), 3.0 + .02 * (i / 8),
                             .01 * (i % 3), .34f});
        compare("contact overflow", dense, 37, 7.3, 8.7, .17, 1);
        compare("narrow grid", {{.02, .02, 0, .08f},
                                {.98, .98, 0, .08f},
                                {.04, .03, 0, .08f}}, 3, 1.0, 1.0, .17);
        ordinary_smoke();
        growth_without_reinit();
        concurrent_devices();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL shear CLL contacts: %s\n", e.what());
        return 1;
    }
}

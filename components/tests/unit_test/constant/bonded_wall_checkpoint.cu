#include "src/core/wall/bonded_wall_checkpoint.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace cudadem;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void same_state(const BondedRectWallState& a, const BondedRectWallState& b) {
    require(a.id == b.id && a.material_id == b.material_id && a.enabled == b.enabled,
            "wall identity changed");
    require(a.half_x == b.half_x && a.half_y == b.half_y && a.half_z == b.half_z &&
            a.mass == b.mass && a.inertia_x == b.inertia_x &&
            a.inertia_y == b.inertia_y && a.inertia_z == b.inertia_z,
            "wall fixed data changed");
    require(a.pos_x == b.pos_x && a.pos_y == b.pos_y && a.pos_z == b.pos_z &&
            a.vel_x == b.vel_x && a.vel_y == b.vel_y && a.vel_z == b.vel_z &&
            a.omega_x == b.omega_x && a.omega_y == b.omega_y && a.omega_z == b.omega_z,
            "wall evolving data changed");
    for (int i = 0; i < 9; ++i)
        require(a.rot[i] == b.rot[i], "wall rotation changed");
}

static void integrate(int steps) {
    for (int i = 0; i < steps; ++i)
        bonded_rect_wall_integrate(0.002f, 0.1f, -0.2f, -0.7f);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        SimConfig config;
        config.num_walls = 1;
        auto& wall = config.walls[0];
        wall.enabled = true; wall.type = WallType::BONDED_RECT;
        wall.id = 7; wall.material_id = 0;
        wall.half_extents[0] = 0.7f; wall.half_extents[1] = 0.9f; wall.half_extents[2] = 1.1f;
        wall.bonded_mass = 3.0f;
        wall.inertia_body[0] = 1.0f; wall.inertia_body[1] = 1.5f; wall.inertia_body[2] = 2.0f;
        wall.velocity.x = 0.3; wall.velocity.y = -0.2; wall.velocity.z = 0.1;
        wall.initial_omega_body.x = 0.2; wall.initial_omega_body.y = 0.3; wall.initial_omega_body.z = -0.4;
        bonded_rect_wall_initialize(config, 0);

        BondedRectWallState initial;
        BondedRectReaction reaction;
        require(bonded_rect_wall_get_states(&initial, &reaction) == 1, "wall did not initialize");
        reaction.force_x = 0.1; reaction.force_y = -0.3; reaction.force_z = 0.2;
        reaction.torque_x = 0.05; reaction.torque_y = -0.02; reaction.torque_z = 0.03;
        bonded_rect_wall_restore_states(&initial, &reaction, 1);
        integrate(5);
        BondedRectWallState saved;
        BondedRectReaction saved_reaction;
        bonded_rect_wall_get_states(&saved, &saved_reaction);
        require(saved.pos_x != initial.pos_x && saved.rot[0] != initial.rot[0],
                "body fixture did not evolve");

        SimulationState state;
        state.current_step = 5; state.current_time = 0.01;
        char directory[] = "/tmp/granit-bonded-roundtrip-XXXXXX";
        require(mkdtemp(directory), "tempdir failed");
        const std::string path = std::string(directory) + "/checkpoint.bin";
        save_bonded_wall_checkpoint(path, state);
        integrate(4);
        BondedRectWallState baseline;
        BondedRectReaction baseline_reaction;
        bonded_rect_wall_get_states(&baseline, &baseline_reaction);

        restore_bonded_wall_checkpoint(path, state);
        BondedRectWallState roundtrip;
        BondedRectReaction roundtrip_reaction;
        bonded_rect_wall_get_states(&roundtrip, &roundtrip_reaction);
        same_state(saved, roundtrip);
        require(std::memcmp(&saved_reaction, &roundtrip_reaction, sizeof(saved_reaction)) == 0,
                "reactions failed roundtrip");
        integrate(4);
        BondedRectWallState resumed;
        BondedRectReaction resumed_reaction;
        bonded_rect_wall_get_states(&resumed, &resumed_reaction);
        same_state(baseline, resumed);
        require(std::memcmp(&baseline_reaction, &resumed_reaction, sizeof(baseline_reaction)) == 0,
                "reactions changed in continuation");

        bool rejected = false;
        auto bad = saved;
        bad.mass += 1;
        try { bonded_rect_wall_restore_states(&bad, &saved_reaction, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "mismatched mass accepted");
        rejected = false;
        ++state.current_step;
        try { restore_bonded_wall_checkpoint(path, state); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "mismatched step accepted");
        bonded_rect_wall_shutdown();
        std::filesystem::remove_all(directory);
        std::cout << "bonded wall GPU checkpoint roundtrip and bit-exact continuation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

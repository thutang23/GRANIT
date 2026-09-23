#include "src/core/constant/command_line_interface.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <unistd.h>

using namespace cudadem;
namespace fs = std::filesystem;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto source = fs::absolute(argv[1]);
    auto temporary = fs::temp_directory_path() / ("granit-defaults-" + std::to_string(getpid()));
    fs::create_directories(temporary);
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{temporary};
    int cases = 0;
    auto base = [&]() {
        auto root = YAML::Load("simulation: {physics: dem}\ndem: {damping_model: restitution}\n");
        root["material_database"] = (source / "components/config/materials_example.yaml").string();
        return root;
    };
    auto parse = [&](const YAML::Node& root) {
        const auto path = temporary / "config.yaml";
        std::ofstream(path) << root;
        ++cases;
        return parse_yaml(path.string());
    };
    auto rejects = [&](const YAML::Node& root, const char* expected) {
        try { parse(root); }
        catch (const std::exception& error) {
            check(std::string(error.what()).find(expected) != std::string::npos, error.what());
            return;
        }
        throw std::runtime_error(std::string("unexpected acceptance: ") + expected);
    };
    try {
        const SimConfig raw;
        check(raw.acceleration.gravity_update_rate == 2, "raw RESPA default");
        check(raw.neighbor.method == NeighborMethod::LBVH, "raw LBVH default");
        check(raw.acceleration.unified_fmm.adaptive_depth.enabled, "raw adaptive default");
        auto root = base();
        auto config = parse(root);
        check(!config.acceleration.enable_self_gravity && !config.acceleration.enable_inertial &&
              !config.acceleration.enable_tidal && !config.damping.enabled, "inactive forces/damping");
        check(!config.acceleration.unified_fmm.adaptive_depth.enabled, "inactive controller");
        root["accelerations"]["self_gravity"]["enabled"] = true;
        auto gravity = YAML::Clone(root);
        config = parse(root);
        check(config.acceleration.gravity_method == SelfGravityMethod::BFFMM &&
              config.acceleration.gravity_update_rate == 2 &&
              config.acceleration.unified_fmm.enable_global_projection &&
              config.acceleration.unified_fmm.theta_far == 0.8f &&
              config.acceleration.unified_fmm.adaptive_depth.enabled &&
              config.acceleration.unified_fmm.adaptive_depth.initial_depth_auto, "DEM gravity defaults");
        root["accelerations"]["self_gravity"]["unified_fmm"]["tree_depth"] = 8;
        config = parse(root);
        check(!config.acceleration.unified_fmm.adaptive_depth.enabled &&
              config.acceleration.unified_fmm.tree_depth == 8, "integer fixed depth");
        root["accelerations"]["self_gravity"]["unified_fmm"]["adaptive_depth"]["enabled"] = true;
        config = parse(root);
        check(config.acceleration.unified_fmm.adaptive_depth.enabled &&
              !config.acceleration.unified_fmm.adaptive_depth.initial_depth_auto, "adaptive seed");
        root["accelerations"]["self_gravity"]["unified_fmm"]["tree_depth"] = "auto";
        check(parse(root).acceleration.unified_fmm.adaptive_depth.initial_depth_auto, "explicit auto");
        root["accelerations"]["self_gravity"]["unified_fmm"]["adaptive_depth"]["enabled"] = false;
        rejects(root, "requires");
        root = YAML::Clone(gravity);
        root["accelerations"]["self_gravity"]["unified_fmm"]["theta_far"] = 0.7;
        rejects(root, "theta_far must be 0.8");
        root = YAML::Clone(gravity);
        root["accelerations"]["self_gravity"]["gravity_macro_dt"] = 0.02;
        check(parse(root).acceleration.gravity_update_rate == 1, "macro dt replaces omitted rate");
        root["accelerations"]["self_gravity"]["gravity_update_rate"] = 2;
        rejects(root, "gravity_macro_dt requires");
        root = YAML::Clone(gravity);
        root["accelerations"]["self_gravity"]["gravity_update_rate"] = 0;
        rejects(root, "gravity_update_rate must be a positive integer");
        root = YAML::Clone(gravity);
        root["damping"]["enabled"] = true;
        rejects(root, "RESPA is incompatible");
        root["accelerations"]["self_gravity"]["gravity_update_rate"] = 1;
        check(parse(root).damping.enabled, "explicit local damping without RESPA");
        root = YAML::Clone(gravity);
        root["accelerations"]["acceleration_limit"]["enabled"] = true;
        rejects(root, "RESPA is incompatible");
        for (const char* key : {"node_overallocate_factor", "p2p_queue_initial_factor",
                "p2p_queue_resize_factor", "m2l_warps_per_block", "p2p_tile_t", "p2p_tile_s"}) {
            root = YAML::Clone(gravity);
            root["accelerations"]["self_gravity"]["unified_fmm"][key] = 4;
            rejects(root, "unsupported or obsolete");
        }
        for (const auto& pair : {std::pair{"gpu", "use_double_buffer"}, {"gpu", "texture_memory"},
                               {"neighbor", "verlet_skin_factor"}, {"neighbor", "typo"}, {"neighbor", "use_hash_table"}}) {
            root = base(); root[pair.first][pair.second] = true;
            rejects(root, "unsupported or obsolete");
        }
        root = base(); root["multigpu"]["rebalance_frequency"] = 100;
        rejects(root, "rebalance_frequency is obsolete");
        root = base(); root["accelerations"]["self_gravity"]["tree_rebuild_interval"] = 10;
        rejects(root, "unsupported or obsolete");
        root = base(); root["io"]["output"]["damage"] = true;
        rejects(root, "unsupported or obsolete");
        root = base(); root["io"]["output"]["fields"].push_back("flag");
        rejects(root, "standard v3 fields are always written");
        root = base(); root["simulation"]["physics"] = "sph";
        config = parse(root);
        check(config.acceleration.gravity_update_rate == 2 &&
              config.acceleration.unified_fmm.tree_depth == 0, "SPH defaults");
        root["accelerations"]["self_gravity"]["enabled"] = true;
        config = parse(root);
        check(config.acceleration.unified_fmm.adaptive_depth.enabled, "SPH adaptive FMM");
        check(config.shear_box.gravity_cutoff_radius == 0.0, "shear cutoff defaults off");
        root["accelerations"]["self_gravity"]["method"] = "direct";
        check(parse(root).acceleration.gravity_update_rate == 2, "SPH direct RESPA");
        root["accelerations"]["self_gravity"]["gravity_update_rate"] = 1;
        check(parse(root).acceleration.gravity_update_rate == 1, "SPH per-step control");
        for (const char* filename : {"dem_example.yaml", "sph_example.yaml"}) {
            ++cases;
            parse_yaml((source / "components/config" / filename).string());
        }
        std::cout << cases << " configuration cases passed\n";
    } catch (const std::exception& error) {
        std::cerr << "case " << cases << ": " << error.what() << '\n';
        return 1;
    }
}

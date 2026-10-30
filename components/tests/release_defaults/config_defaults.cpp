#include "src/core/constant/command_line_interface.h"
#include "src/core/constant/dem_timestep_selection.h"
#include <limits>
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
        const double infinity = std::numeric_limits<double>::infinity();
        check(select_dem_timestep(0.75, 0.2, 1.0, infinity) == 0.75,
              "explicit dt above legacy fraction remains exact");
        check(select_dem_timestep(0.0, 0.2, 1.0, infinity) == 0.2, "legacy default dt");
        check(select_dem_timestep(0.0, 0.3, 1.0, 0.1) == 0.1, "legacy orbital clipping");
        auto selection_rejects = [&](double dt, double factor, double critical, double orbital) {
            try { select_dem_timestep(dt, factor, critical, orbital); }
            catch (const std::runtime_error&) { return; }
            throw std::runtime_error("invalid timestep accepted");
        };
        selection_rejects(1.01, 0.2, 1.0, infinity);
        selection_rejects(0.5, 0.2, 1.0, 0.4);
        selection_rejects(-1.0, 0.2, 1.0, infinity);
        selection_rejects(infinity, 0.2, 1.0, infinity);
        selection_rejects(0.0, infinity, 1.0, infinity);
        validate_dem_checkpoint_timestep(0.5, 0.75, 0.0);
        validate_dem_checkpoint_timestep(0.5, 0.5, 0.5);
        for (const auto request : {0.25, 0.75}) {
            bool rejected = false;
            try { validate_dem_checkpoint_timestep(0.5, request, request); }
            catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "checkpoint explicit dt mismatch rejected");
        }
        const SimConfig raw;
        check(raw.acceleration.gravity_update_rate == 2, "raw RESPA default");
        check(raw.neighbor.method == NeighborMethod::LBVH, "raw LBVH default");
        check(raw.acceleration.unified_fmm.adaptive_depth.enabled, "raw adaptive default");
        auto root = base();
        auto config = parse(root);
        check(!config.acceleration.enable_self_gravity && !config.acceleration.enable_inertial &&
              !config.acceleration.enable_tidal && !config.damping.enabled, "inactive forces/damping");
        check(!config.acceleration.unified_fmm.adaptive_depth.enabled, "inactive controller");
        auto timestep = base();
        timestep["simulation"]["time"]["dt"] = 0.75;
        check(parse(timestep).time.dt == 0.75, "seconds timestep parsed exactly");
        timestep["simulation"]["time"]["dt_factor"] = 0.1;
        rejects(timestep, "mutually exclusive");
        for (double invalid : {0.0, -1.0, infinity, std::numeric_limits<double>::quiet_NaN()}) {
            timestep = base(); timestep["simulation"]["time"]["dt"] = invalid;
            rejects(timestep, "simulation.time.dt must be finite");
            timestep = base(); timestep["simulation"]["time"]["dt_factor"] = invalid;
            rejects(timestep, "simulation.time.dt_factor must be finite");
        }
        timestep = base(); timestep["simulation"]["time"]["dt_factor"] = 0.1;
        check(parse(timestep).time.dt_factor == 0.1 && parse(timestep).time.dt == 0.0,
              "legacy factor preserved");
        for (const char* key : {"dt", "dt_factor"}) {
            timestep = base(); timestep["simulation"]["physics"] = "sph";
            timestep["simulation"]["time"][key] = 0.1;
            rejects(timestep, "SPH uses sph_parameters.dt_min and sph_parameters.dt_max");
        }
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

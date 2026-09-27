// =============================================================================
// Host-Only Configuration Functions
// =============================================================================
// These functions use std::string, YAML, and other host-only features.
// Compiled separately from CUDA code to avoid nvcc parsing issues.
// =============================================================================

#include "command_line_interface.h"
#include "../../domain/domain_manager.h"  // MAX_MULTI_GPUS is the runner's hard limit.
#include "../acceleration/body_fitted_fmm/shear_fmm_capacity.h"
#include <yaml-cpp/yaml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <set>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

namespace cudadem {

namespace detail {

// Reject removed or misspelled performance controls instead of silently ignoring them.
void require_keys(const YAML::Node& node, const char* path,
                  std::initializer_list<const char*> allowed) {
    if (!node.IsMap()) throw std::runtime_error(std::string(path) + " must be a mapping");
    for (const auto& entry : node) {
        const auto key = entry.first.as<std::string>();
        if (std::none_of(allowed.begin(), allowed.end(),
                         [&](const char* candidate) { return key == candidate; }))
            throw std::runtime_error(std::string(path) + "." + key +
                                     " is unsupported or obsolete");
    }
}

// Safe string copy for fixed-size arrays
template<size_t N>
void safe_strcpy(char (&dest)[N], const std::string& src) {
    if (src.size() >= N)
        throw std::runtime_error("configuration string exceeds fixed runtime capacity");
    strncpy(dest, src.c_str(), N - 1);
    dest[N - 1] = '\0';
}

// Explicit instantiations for common sizes
template void safe_strcpy<MAX_NAME_LENGTH>(char (&)[MAX_NAME_LENGTH], const std::string&);
template void safe_strcpy<MAX_PATH_LENGTH>(char (&)[MAX_PATH_LENGTH], const std::string&);

inline void fingerprint_bytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= UINT64_C(1099511628211);
    }
}

inline void fingerprint_string(uint64_t& hash, const char* value) {
    const uint32_t size = static_cast<uint32_t>(std::strlen(value));
    fingerprint_bytes(hash, &size, sizeof(size));
    fingerprint_bytes(hash, value, size);
}

inline void fingerprint_double(uint64_t& hash, double value) {
    // Normalize signed zero so semantically identical YAML hashes identically.
    if (value == 0.0) value = 0.0;
    fingerprint_bytes(hash, &value, sizeof(value));
}

// Parse Vec3d from YAML
Vec3d parse_vec3d(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3)
        throw std::runtime_error("expected a three-component vector");
    Vec3d v;
    v.x = node[0].as<double>();
    v.y = node[1].as<double>();
    v.z = node[2].as<double>();
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        throw std::runtime_error("vector components must be finite");
    return v;
}

// Parse Vec3f from YAML
Vec3f parse_vec3f(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3)
        throw std::runtime_error("expected a three-component vector");
    Vec3f v;
    v.x = node[0].as<float>();
    v.y = node[1].as<float>();
    v.z = node[2].as<float>();
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        throw std::runtime_error("vector components must be finite");
    return v;
}

// Parse 3x3 matrix from YAML
Mat3d parse_mat3d(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3)
        throw std::runtime_error("expected a 3x3 matrix");
    Mat3d m;
    for (int i = 0; i < 3; ++i) {
        if (!node[i].IsSequence() || node[i].size() != 3)
            throw std::runtime_error("expected a 3x3 matrix");
        for (int j = 0; j < 3; ++j) {
            m.data[i*3 + j] = node[i][j].as<double>();
            if (!std::isfinite(m.data[i*3 + j]))
                throw std::runtime_error("matrix components must be finite");
        }
    }
    return m;
}

// Parse enumerations
SimulationType parse_sim_type(const std::string& s) {
    if (s == "asteroid_shake" || s == "asteroid") return SimulationType::ASTEROID_SHAKE;
    if (s == "surface_slope" || s == "surface") return SimulationType::SURFACE_SLOPE;
    throw std::runtime_error("unsupported simulation.type: " + s);
}

CoordinateSystem parse_coord_system(const std::string& s) {
    if (s == "mass_center" || s == "masscenter") return CoordinateSystem::MASS_CENTER;
    if (s == "rotating") return CoordinateSystem::ROTATING;
    if (s == "surface") return CoordinateSystem::SURFACE;
    throw std::runtime_error("unsupported simulation.coordinate_system: " + s);
}

ContactModel parse_contact_model(const std::string& s) {
    if (s == "visco" || s == "viscous") return ContactModel::VISCOUS;
    if (s == "restitution") return ContactModel::RESTITUTION;
    throw std::runtime_error("unsupported contact model: " + s);
}

SimulationPhysics parse_physics(const std::string& s) {
    if (s == "dem") return SimulationPhysics::DEM;
    if (s == "sph") return SimulationPhysics::SPH;
    throw std::runtime_error("simulation.physics must be 'dem' or 'sph'");
}

DemDampingModel parse_dem_damping_model(const std::string& s) {
    if (s == "restitution") return DemDampingModel::RESTITUTION;
    if (s == "viscous") return DemDampingModel::VISCOUS;
    throw std::runtime_error("dem.damping_model must be 'restitution' or 'viscous'");
}

WallType parse_wall_type(const std::string& s) {
    if (s == "plain") return WallType::PLAIN;
    if (s == "periodical" || s == "periodic") return WallType::PERIODICAL;
    if (s == "moving") return WallType::MOVING;
    if (s == "mesh") return WallType::MESH;
    if (s == "free") return WallType::FREE;
    if (s == "bonded_rect") return WallType::BONDED_RECT;
    if (s == "finite_plain" || s == "finiteplain") return WallType::FINITE_PLAIN;
    if (s == "funnel") return WallType::FUNNEL;
    if (s == "sphere") return WallType::SPHERE;
    throw std::runtime_error("unsupported wall type: " + s);
}

LoadMethod parse_load_method(const std::string& s) {
    if (s == "sequential") return LoadMethod::SEQUENTIAL;
    if (s == "once") return LoadMethod::ONCE;
    throw std::runtime_error("load_method must be 'once' or 'sequential'");
}

InterpolationMethod parse_interp_method(const std::string& s) {
    if (s == "linear") return InterpolationMethod::LINEAR;
    if (s == "cubic") return InterpolationMethod::CUBIC;
    if (s == "constant" || s == "none") return InterpolationMethod::NONE;
    throw std::runtime_error("interpolation method must be constant, linear, or cubic");
}

DampingType parse_damping_type(const std::string& s) {
    if (s == "viscous") return DampingType::VISCOUS;
    if (s == "local") return DampingType::LOCAL;
    if (s == "none") return DampingType::NONE;
    throw std::runtime_error("damping.type must be viscous, local, or none");
}

TerminationCondition parse_termination(const std::string& s) {
    if (s == "energy" || s == "energy_threshold") return TerminationCondition::ENERGY_THRESHOLD;
    if (s == "time") return TerminationCondition::TIME;
    if (s == "adaptive_yorp") return TerminationCondition::ADAPTIVE_YORP;
    throw std::runtime_error("simulation.termination.condition must be 'time', 'energy_threshold', or 'adaptive_yorp'");
}

} // namespace detail

// -----------------------------------------------------------------------------
// Parse YAML file into SimConfig
// -----------------------------------------------------------------------------
SimConfig parse_yaml(const std::string& yaml_path) {
    SimConfig config;

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load YAML file: " + yaml_path + "\n" + e.what());
    }

    detail::require_keys(root, "configuration", {"material_database", "simulation", "dem",
        "domain", "walls", "accelerations", "neighbor", "damping", "io", "gpu",
        "multigpu", "clumps", "sph_parameters", "shear_box", "particle"});
    if (!root["simulation"] || !root["simulation"]["physics"]) {
        throw std::runtime_error("simulation.physics is required");
    }
    config.physics = detail::parse_physics(root["simulation"]["physics"].as<std::string>());
    // simulation.physics is the sole mode selector; no secondary SPH enable
    // switch is accepted or inferred from the input filename.
    config.sph.enabled = (config.physics == SimulationPhysics::SPH);
    if (config.physics == SimulationPhysics::DEM) {
        const auto dem = root["dem"];
        if (!dem || !dem["damping_model"]) {
            throw std::runtime_error("dem.damping_model is required for DEM runs");
        }
        config.dem_damping_model = detail::parse_dem_damping_model(dem["damping_model"].as<std::string>());
    }
    // Material data is intentionally external to the simulation YAML, so it
    // can be fingerprinted and replicated unchanged to every participating GPU.
    if (!root["material_database"] || !root["material_database"].IsScalar()) {
        throw std::runtime_error("material_database is required and must be a path");
    }
    const std::filesystem::path config_directory =
        std::filesystem::absolute(std::filesystem::path(yaml_path)).parent_path();
    const std::filesystem::path material_path =
        (config_directory / root["material_database"].as<std::string>()).lexically_normal();
    const MaterialDatabase materials = load_material_database(material_path.string(), config.dem_damping_model);
    config.material_database = materials.info;
    if (root["particle"]) {
        throw std::runtime_error("particle is no longer supported; material properties belong in material_database and radii come from unified v3 input");
    }

    // --- Simulation section ---
    if (auto sim = root["simulation"]) {
        if (sim["name"]) {
            detail::safe_strcpy(config.name, sim["name"].as<std::string>());
        }
        if (sim["type"]) {
            config.sim_type = detail::parse_sim_type(sim["type"].as<std::string>());
        }
        if (sim["coordinate_system"]) {
            config.coord_system = detail::parse_coord_system(sim["coordinate_system"].as<std::string>());
        }

        // Rotating frame configuration
        if (auto rf = sim["rotating_frame"]) {
            config.rotating_frame.enabled = true;
            if (rf["omega"]) {
                auto om = rf["omega"];
                if (om.IsSequence() && om.size() >= 3) {
                    config.rotating_frame.omega.x = om[0].as<double>();
                    config.rotating_frame.omega.y = om[1].as<double>();
                    config.rotating_frame.omega.z = om[2].as<double>();
                }
            }
            if (rf["rotation_period_h"]) {
                config.rotating_frame.rotation_period_h = rf["rotation_period_h"].as<float>();
                // Auto-compute omega if not explicitly set
                if (!rf["omega"]) {
                    double period_s = config.rotating_frame.rotation_period_h * 3600.0;
                    double omega_mag = 2.0 * M_PI / period_s;
                    config.rotating_frame.omega.z = omega_mag;  // default: z-axis
                }
            }
            if (rf["settlement"])
                throw std::runtime_error("simulation.rotating_frame.settlement is unsupported; no runtime settlement controller exists");
        }

        // Time configuration
        if (auto time = sim["time"]) {
            if (time["start"]) {
                std::string start_str = time["start"].as<std::string>();
                bool parsed = false;
                try {
                    // Try parsing ISO 8601 format: "2029-04-13T18:40:00.0000"
                    std::tm tm = {};
                    std::istringstream iss(start_str);
                    iss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");

                    if (!iss.fail()) {
                        config.time.start_time = static_cast<double>(std::mktime(&tm));
                        // Handle fractional seconds
                        size_t dot_pos = start_str.find('.');
                        if (dot_pos != std::string::npos && dot_pos + 1 < start_str.length()) {
                            std::string frac_str = start_str.substr(dot_pos + 1);
                            try {
                                double frac = std::stod("0." + frac_str);
                                config.time.start_time += frac;
                            } catch (...) {}
                        }
                        parsed = true;
                    }
                } catch (...) {}

                if (!parsed) {
                    // Fall back to numeric parsing
                    try {
                        config.time.start_time = std::stod(start_str);
                    } catch (...) {
                        config.time.start_time = 0.0;
                    }
                }
            }
            if (time["stop"]) {
                std::string stop_str = time["stop"].as<std::string>();
                bool parsed = false;
                try {
                    // Try parsing ISO 8601 format: "2029-04-14T00:40:00.0000"
                    std::tm tm = {};
                    std::istringstream iss(stop_str);
                    iss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");

                    if (!iss.fail()) {
                        config.time.stop_time = static_cast<double>(std::mktime(&tm));
                        // Handle fractional seconds
                        size_t dot_pos = stop_str.find('.');
                        if (dot_pos != std::string::npos && dot_pos + 1 < stop_str.length()) {
                            std::string frac_str = stop_str.substr(dot_pos + 1);
                            try {
                                double frac = std::stod("0." + frac_str);
                                config.time.stop_time += frac;
                            } catch (...) {}
                        }
                        parsed = true;
                    }
                } catch (...) {}

                if (!parsed) {
                    // Fall back to numeric parsing
                    try {
                        config.time.stop_time = std::stod(stop_str);
                    } catch (...) {
                        config.time.stop_time = 0.0;
                    }
                }
            }
            if (time["output_step"]) config.time.output_step = time["output_step"].as<double>();
            if (time["dt_factor"]) config.time.dt_factor = time["dt_factor"].as<double>();
            if (time["dt"]) config.time.dt_upper_bound = time["dt"].as<double>();
        }

        // Termination
        if (auto term = sim["termination"]) {
            if (term["condition"]) {
                config.termination = detail::parse_termination(term["condition"].as<std::string>());
            }
            if (term["energy_threshold"]) {
                config.energy_threshold = term["energy_threshold"].as<double>();
            }
        }
    }

    // --- Local planetary-ring shearing sheet ---
    if (auto shear = root["shear_box"]) {
        if (!shear.IsMap()) {
            throw std::runtime_error("shear_box must be a mapping");
        }
        static const std::set<std::string> allowed_shear_keys = {
            "enabled", "omega", "central_body", "shear_parameter",
            "vertical_frequency", "time_origin", "input_velocity",
            "orbital_dt_factor", "gravity_images",
            "force_balance_warn_threshold"
        };
        for (const auto& entry : shear) {
            const std::string key = entry.first.as<std::string>();
            if (allowed_shear_keys.count(key) == 0) {
                throw std::runtime_error("unsupported shear_box key: " + key);
            }
        }

        if (shear["enabled"]) config.shear_box.enabled = shear["enabled"].as<bool>();
        if (shear["shear_parameter"])
            config.shear_box.shear_parameter = shear["shear_parameter"].as<double>();
        if (shear["time_origin"])
            config.shear_box.time_origin = shear["time_origin"].as<double>();
        if (shear["orbital_dt_factor"])
            config.shear_box.orbital_dt_factor = shear["orbital_dt_factor"].as<double>();
        if (shear["force_balance_warn_threshold"])
            config.shear_box.force_balance_warn_threshold =
                shear["force_balance_warn_threshold"].as<double>();

        const bool has_omega = static_cast<bool>(shear["omega"]);
        const bool has_central = static_cast<bool>(shear["central_body"]);
        if (has_omega && has_central) {
            throw std::runtime_error(
                "shear_box.omega and shear_box.central_body are mutually exclusive");
        }
        if (has_omega) config.shear_box.omega = shear["omega"].as<double>();
        if (has_central) {
            const auto central = shear["central_body"];
            if (!central.IsMap())
                throw std::runtime_error("shear_box.central_body must be a mapping");
            static const std::set<std::string> allowed_central_keys = {
                "mass", "orbital_radius", "gravitational_constant"
            };
            for (const auto& entry : central) {
                const std::string key = entry.first.as<std::string>();
                if (allowed_central_keys.count(key) == 0)
                    throw std::runtime_error(
                        "unsupported shear_box.central_body key: " + key);
            }
            for (const char* key : {"mass", "orbital_radius", "gravitational_constant"}) {
                if (!central[key])
                    throw std::runtime_error(
                        std::string("shear_box.central_body.") + key + " is required");
            }
            const double mass = central["mass"].as<double>();
            const double radius = central["orbital_radius"].as<double>();
            const double gravitational_constant =
                central["gravitational_constant"].as<double>();
            if (!(mass > 0.0) || !(radius > 0.0) ||
                !(gravitational_constant > 0.0) || !std::isfinite(mass) ||
                !std::isfinite(radius) || !std::isfinite(gravitational_constant)) {
                throw std::runtime_error(
                    "shear_box.central_body values must be finite and positive");
            }
            config.shear_box.omega =
                std::sqrt(gravitational_constant * mass / (radius * radius * radius));
        }

        if (shear["vertical_frequency"]) {
            config.shear_box.vertical_frequency =
                shear["vertical_frequency"].as<double>();
        } else {
            config.shear_box.vertical_frequency = config.shear_box.omega;
        }
        if (shear["input_velocity"]) {
            const std::string convention = shear["input_velocity"].as<std::string>();
            if (convention == "total") {
                config.shear_box.input_velocity = ShearInputVelocity::TOTAL;
            } else if (convention == "peculiar") {
                config.shear_box.input_velocity = ShearInputVelocity::PECULIAR;
            } else {
                throw std::runtime_error(
                    "shear_box.input_velocity must be 'total' or 'peculiar'");
            }
        }
        if (auto images = shear["gravity_images"]) {
            if (!images.IsMap())
                throw std::runtime_error("shear_box.gravity_images must be a mapping");
            static const std::set<std::string> allowed_image_keys = {
                "layers_x", "layers_y", "cutoff_radius"
            };
            for (const auto& entry : images) {
                const std::string key = entry.first.as<std::string>();
                if (allowed_image_keys.count(key) == 0)
                    throw std::runtime_error(
                        "unsupported shear_box.gravity_images key: " + key);
            }
            if (images["layers_x"])
                config.shear_box.gravity_layers_x = images["layers_x"].as<int32_t>();
            if (images["layers_y"])
                config.shear_box.gravity_layers_y = images["layers_y"].as<int32_t>();
            if (images["cutoff_radius"] && !images["cutoff_radius"].IsNull())
                config.shear_box.gravity_cutoff_radius =
                    images["cutoff_radius"].as<double>();
        }
    }

    // --- Domain section ---
    if (auto dom = root["domain"]) {
        if (auto ps = dom["patch_size"]) {
            if (ps["length"] && ps["width"] && ps["height"]) {
                config.domain.patch_size = Vec3f(
                    ps["length"].as<float>(),
                    ps["width"].as<float>(),
                    ps["height"].as<float>()
                );
            }
        }
        // Domain bounding box
        if (dom["box_min"]) {
            config.domain.box_min = detail::parse_vec3f(dom["box_min"]);
        }
        if (dom["box_max"]) {
            config.domain.box_max = detail::parse_vec3f(dom["box_max"]);
        }
        if (dom["bounds"]) {
            std::string bounds_str = dom["bounds"].as<std::string>();
            if (bounds_str == "auto") {
                config.domain.auto_bounds = true;
            }
        }
        if (auto so = dom["surface_origin"]) {
            if (so["position"]) {
                config.domain.surface_origin = detail::parse_vec3d(so["position"]);
            }
            if (so["rotation_matrix"]) {
                config.domain.rotation_matrix = detail::parse_mat3d(so["rotation_matrix"]);
            }
        }
    }

    // --- Walls section ---
    if (auto walls = root["walls"]) {
        if (!walls.IsSequence()) throw std::runtime_error("walls must be a sequence");
        if (walls.size() > MAX_WALLS) throw std::runtime_error("walls exceeds MAX_WALLS");
        config.num_walls = 0;
        for (size_t i = 0; i < walls.size() && i < MAX_WALLS; ++i) {
            auto w = walls[i];
            WallConfig& wc = config.walls[config.num_walls];

            if (w["id"]) wc.id = w["id"].as<int32_t>();
            else wc.id = static_cast<int32_t>(i + 1);
            if (!w["material_id"]) {
                throw std::runtime_error("walls[" + std::to_string(i) + "].material_id is required");
            }
            wc.material_id = w["material_id"].as<int32_t>();
            if (wc.material_id < 0 || static_cast<uint32_t>(wc.material_id) >= config.material_database.material_count) {
                throw std::runtime_error("walls[" + std::to_string(i) + "].material_id is out of range");
            }
            if (!materials.materials[static_cast<size_t>(wc.material_id)].has_dem) {
                throw std::runtime_error("walls[" + std::to_string(i) + "].material_id must reference a DEM-capable material");
            }

            if (w["type"]) wc.type = detail::parse_wall_type(w["type"].as<std::string>());
            if (w["enabled"]) wc.enabled = w["enabled"].as<bool>();
            else wc.enabled = true;

            if (w["position"]) wc.position = detail::parse_vec3d(w["position"]);
            if (w["normal"]) wc.normal = detail::parse_vec3d(w["normal"]);
            // ``velocity`` historically had two incompatible meanings: a
            // translated wall geometry or a conveyor surface speed.  Accept
            // neither implicitly; users must choose an explicit motion mode.
            if (w["velocity_data_path"].IsDefined() || w["motion_data_path"].IsDefined()) {
                throw std::runtime_error("legacy wall velocity/motion data files are ambiguous; use trajectory_path (RIGIDPS2)");
            }
            if (w["motion_mode"]) {
                const std::string mode = w["motion_mode"].as<std::string>();
                if (mode == "surface_only") wc.surface_only_motion = true;
                else if (mode == "rigid_pose") wc.surface_only_motion = false;
                else throw std::runtime_error("wall motion_mode must be rigid_pose or surface_only");
            }
            if (w["surface_velocity"]) {
                if (!wc.surface_only_motion) throw std::runtime_error("surface_velocity requires motion_mode: surface_only");
                wc.surface_velocity = detail::parse_vec3d(w["surface_velocity"]);
            }
            if (w["velocity"]) {
                if (!wc.surface_only_motion) throw std::runtime_error("velocity is ambiguous; use trajectory_path or motion_mode: surface_only with surface_velocity");
                wc.surface_velocity = detail::parse_vec3d(w["velocity"]);
            }
            if (w["trajectory_path"]) {
                if (wc.surface_only_motion) throw std::runtime_error("trajectory_path and surface_only motion cannot be combined");
                detail::safe_strcpy(wc.rigid_pose_path, w["trajectory_path"].as<std::string>());
            }

            if (w["axis"]) {
                std::string axis = w["axis"].as<std::string>();
                if (axis == "x") wc.axis = 0;
                else if (axis == "y") wc.axis = 1;
                else if (axis == "z") wc.axis = 2;
                else throw std::runtime_error("wall axis must be x, y, or z");
            }
            if (w["bounds"]) {
                auto b = w["bounds"];
                if (b.IsSequence() && b.size() >= 2) {
                    wc.bound_min = b[0].as<float>();
                    wc.bound_max = b[1].as<float>();
                }
            }

            // Mesh wall specific
            if (w["mesh_path"]) {
                detail::safe_strcpy(wc.mesh_path, w["mesh_path"].as<std::string>());
            }
            if (w["angular_velocity"]) {
                wc.angular_velocity = detail::parse_vec3d(w["angular_velocity"]);
            }

            // Bonded rectangular prism fields
            if (wc.type == WallType::BONDED_RECT) {
                if (w["half_extents"]) {
                    auto he = w["half_extents"];
                    if (he.IsSequence() && he.size() >= 3) {
                        wc.half_extents[0] = he[0].as<float>();
                        wc.half_extents[1] = he[1].as<float>();
                        wc.half_extents[2] = he[2].as<float>();
                    }
                }
                if (w["mass"])
                    wc.bonded_mass = w["mass"].as<float>();
                if (w["inertia_body"]) {
                    auto ib = w["inertia_body"];
                    if (ib.IsSequence() && ib.size() >= 3) {
                        wc.inertia_body[0] = ib[0].as<float>();
                        wc.inertia_body[1] = ib[1].as<float>();
                        wc.inertia_body[2] = ib[2].as<float>();
                    }
                }
                if (w["omega_body"])
                    wc.initial_omega_body = detail::parse_vec3d(w["omega_body"]);
                if (w["euler_zyx_deg"])
                    wc.euler_zyx_deg = detail::parse_vec3d(w["euler_zyx_deg"]);
            }

            // Finite plain wall fields
            if (wc.type == WallType::FINITE_PLAIN) {
                if (w["half_extents"]) {
                    auto he = w["half_extents"];
                    if (he.IsSequence() && he.size() >= 2) {
                        // Sequence format: half_extents: [0.1, 0.02]
                        wc.half_extents[0] = he[0].as<float>();
                        wc.half_extents[1] = he[1].as<float>();
                    } else if (he["x"] && he["y"]) {
                        // Mapping format: half_extents: {x: 0.1, y: 0.02}
                        wc.half_extents[0] = he["x"].as<float>();
                        wc.half_extents[1] = he["y"].as<float>();
                    }
                }
            }

            // Funnel wall fields
            if (wc.type == WallType::FUNNEL) {
                if (w["funnel_radius_bottom"])
                    wc.funnel_radius_bottom = w["funnel_radius_bottom"].as<float>();
                if (w["funnel_radius_top"])
                    wc.funnel_radius_top = w["funnel_radius_top"].as<float>();
                if (w["funnel_length"])
                    wc.funnel_length = w["funnel_length"].as<float>();
            }

            // Sphere wall fields
            if (wc.type == WallType::SPHERE) {
                if (w["sphere_radius"])
                    wc.sphere_radius = w["sphere_radius"].as<float>();
                if (w["sphere_mass"])
                    wc.sphere_mass = w["sphere_mass"].as<float>();
                if (w["sphere_cohesive_strength"])
                    throw std::runtime_error("sphere_cohesive_strength is unsupported; use material_database interactions");
                if (w["sphere_trajectory_path"]) {
                    detail::safe_strcpy(wc.sphere_trajectory_path,
                                       w["sphere_trajectory_path"].as<std::string>());
                }
                if (wc.rigid_pose_path[0] != '\0' && wc.sphere_trajectory_path[0] != '\0') {
                    throw std::runtime_error("sphere wall must use trajectory_path or sphere_trajectory_path, not both");
                }
                if (wc.rigid_pose_path[0] != '\0') {
                    detail::safe_strcpy(wc.sphere_trajectory_path, std::string(wc.rigid_pose_path));
                }
            }

            if (wc.rigid_pose_path[0] != '\0' &&
                !(wc.type == WallType::PLAIN || wc.type == WallType::MOVING ||
                  wc.type == WallType::FINITE_PLAIN || wc.type == WallType::MESH || wc.type == WallType::SPHERE)) {
                throw std::runtime_error("trajectory_path is only valid for plain, finite_plain, mesh, or sphere walls");
            }
            if (wc.type == WallType::SPHERE && wc.surface_only_motion) {
                throw std::runtime_error("sphere walls cannot use surface_only motion; use a RIGIDPS2 trajectory");
            }

            config.num_walls++;
        }
    }

    // --- Accelerations section ---
    if (auto acc = root["accelerations"]) {
        // Inertial
        if (auto iner = acc["inertial"]) {
            if (iner["enabled"]) config.acceleration.enable_inertial = iner["enabled"].as<bool>();
            if (iner["constant_omega"]) config.acceleration.inertial_constant_omega = iner["constant_omega"].as<bool>();
            if (iner["data_path"]) {
                detail::safe_strcpy(config.acceleration.inertial_data_path,
                                   iner["data_path"].as<std::string>());
            }
            if (iner["load_method"]) {
                config.acceleration.inertial_load_method =
                    detail::parse_load_method(iner["load_method"].as<std::string>());
            }
            if (iner["time_interval"]) {
                config.acceleration.inertial_time_interval = iner["time_interval"].as<float>();
            }
            if (auto interp = iner["interpolation"]) {
                if (interp["method"]) {
                    config.acceleration.inertial_interp =
                        detail::parse_interp_method(interp["method"].as<std::string>());
                }
            }
        }

        // Tidal
        if (auto tidal = acc["tidal"]) {
            if (tidal["enabled"]) config.acceleration.enable_tidal = tidal["enabled"].as<bool>();
            if (tidal["data_path"]) {
                detail::safe_strcpy(config.acceleration.tidal_data_path,
                                   tidal["data_path"].as<std::string>());
            }
            if (tidal["load_method"]) {
                config.acceleration.tidal_load_method =
                    detail::parse_load_method(tidal["load_method"].as<std::string>());
            }
            if (tidal["time_interval"]) {
                config.acceleration.tidal_time_interval = tidal["time_interval"].as<float>();
            }
            if (auto interp = tidal["interpolation"]) {
                if (interp["method"]) {
                    config.acceleration.tidal_interp =
                        detail::parse_interp_method(interp["method"].as<std::string>());
                }
            }
            if (auto pert = tidal["perturber"]) {
                if (pert["mass"]) config.acceleration.perturber_mass = pert["mass"].as<double>();
            }
        }

        // Prescribed, gravity-only massive bodies for inertial global-ring
        // simulations.  They deliberately do not reuse sphere walls because
        // their force path is unsoftened, double precision, and contact-free.
        if (auto bodies = acc["external_bodies"]) {
            if (!bodies.IsMap())
                throw std::runtime_error("accelerations.external_bodies must be a mapping");
            static const std::set<std::string> allowed_body_keys = {
                "enabled", "saturn", "satellite"};
            for (const auto& entry : bodies) {
                const std::string key = entry.first.as<std::string>();
                if (!allowed_body_keys.count(key))
                    throw std::runtime_error("unsupported accelerations.external_bodies key: " + key);
            }
            auto& external = config.acceleration.external_bodies;
            if (bodies["enabled"]) external.enabled = bodies["enabled"].as<bool>();
            if (auto saturn = bodies["saturn"]) {
                if (!saturn.IsMap())
                    throw std::runtime_error("accelerations.external_bodies.saturn must be a mapping");
                static const std::set<std::string> allowed_saturn_keys = {
                    "enabled", "position", "mass", "absorb_radius"};
                for (const auto& entry : saturn) {
                    const std::string key = entry.first.as<std::string>();
                    if (!allowed_saturn_keys.count(key))
                        throw std::runtime_error("unsupported external_bodies.saturn key: " + key);
                }
                if (saturn["enabled"]) external.saturn_enabled = saturn["enabled"].as<bool>();
                if (saturn["position"]) external.saturn_position = detail::parse_vec3d(saturn["position"]);
                if (saturn["mass"]) external.saturn_mass = saturn["mass"].as<double>();
                if (saturn["absorb_radius"])
                    external.saturn_absorb_radius = saturn["absorb_radius"].as<double>();
            }
            if (auto satellite = bodies["satellite"]) {
                if (!satellite.IsMap())
                    throw std::runtime_error("accelerations.external_bodies.satellite must be a mapping");
                static const std::set<std::string> allowed_satellite_keys = {
                    "enabled", "mass", "orbit_radius", "phase", "prograde"};
                for (const auto& entry : satellite) {
                    const std::string key = entry.first.as<std::string>();
                    if (!allowed_satellite_keys.count(key))
                        throw std::runtime_error("unsupported external_bodies.satellite key: " + key);
                }
                if (satellite["enabled"]) external.satellite_enabled = satellite["enabled"].as<bool>();
                if (satellite["mass"]) external.satellite_mass = satellite["mass"].as<double>();
                if (satellite["orbit_radius"])
                    external.satellite_orbit_radius = satellite["orbit_radius"].as<double>();
                if (satellite["phase"]) external.satellite_phase = satellite["phase"].as<double>();
                if (satellite["prograde"]) external.satellite_prograde = satellite["prograde"].as<bool>();
            }
        }

        // Self-gravity
        if (auto sg = acc["self_gravity"]) {
            detail::require_keys(sg, "accelerations.self_gravity",
                {"enabled", "method", "gravity_update_rate", "gravity_macro_dt",
                 "displacement_threshold", "direct", "unified_fmm", "tree_rebuild_rate"});
            if (sg["enabled"]) config.acceleration.enable_self_gravity = sg["enabled"].as<bool>();
            if (sg["method"]) {
                std::string m = sg["method"].as<std::string>();
                if (m == "direct") {
                    config.acceleration.gravity_method = SelfGravityMethod::DIRECT;
                } else if (m == "unified_fmm" || m == "fmm" || m == "lbvh_fmm") {
                    // "fmm"/"lbvh_fmm" are accepted as aliases for the unified
                    // body-fitted FMM (the sole runtime FMM) for backward config compat.
                    if (m == "fmm" || m == "lbvh_fmm") {
                        std::cerr << "NOTE: gravity_method '" << m
                                  << "' is deprecated; use 'unified_fmm'. Treating as unified_fmm." << std::endl;
                    }
                    config.acceleration.gravity_method = SelfGravityMethod::BFFMM;
                } else {
                    throw std::runtime_error("unsupported accelerations.self_gravity.method: " + m);
                }
            }
            if (sg["tree_rebuild_rate"]) {
                throw std::runtime_error(
                    "accelerations.self_gravity.tree_rebuild_rate is no longer supported: "
                    "it was ambiguously interpreted as a RESPA integration interval. Use "
                    "gravity_update_rate for RESPA; BFFMM rebuilds its topology on each "
                    "gravity evaluation");
            }
            if (sg["gravity_update_rate"]) config.acceleration.gravity_update_rate = sg["gravity_update_rate"].as<int>();
            if (sg["gravity_macro_dt"]) {
                config.acceleration.gravity_macro_dt = sg["gravity_macro_dt"].as<double>();
                if (config.acceleration.gravity_macro_dt > 0.0 && !sg["gravity_update_rate"])
                    config.acceleration.gravity_update_rate = 1;
            }
            if (sg["displacement_threshold"]) config.acceleration.displacement_threshold = sg["displacement_threshold"].as<float>();

            if (auto direct = sg["direct"]) {
                if (!direct.IsMap()) {
                    throw std::runtime_error(
                        "accelerations.self_gravity.direct must be a mapping");
                }
                static const std::set<std::string> allowed_direct_keys = {
                    "gravitational_constant", "sph_softening_mode"
                };
                for (const auto& entry : direct) {
                    const std::string key = entry.first.as<std::string>();
                    if (!allowed_direct_keys.count(key)) {
                        throw std::runtime_error(
                            "accelerations.self_gravity.direct." + key +
                            " is not supported");
                    }
                }
                auto& direct_config = config.acceleration.direct_gravity;
                if (direct["gravitational_constant"]) {
                    direct_config.gravitational_constant =
                        direct["gravitational_constant"].as<double>();
                }
                if (direct["sph_softening_mode"]) {
                    const std::string mode =
                        direct["sph_softening_mode"].as<std::string>();
                    if (mode == "symmetric_mean") {
                        direct_config.sph_softening_mode =
                            DirectSphSofteningMode::SYMMETRIC_MEAN;
                    } else if (mode == "query_particle") {
                        direct_config.sph_softening_mode =
                            DirectSphSofteningMode::QUERY_PARTICLE;
                    } else {
                        throw std::runtime_error(
                            "accelerations.self_gravity.direct.sph_softening_mode "
                            "must be symmetric_mean or query_particle");
                    }
                }
            }

            // Unified FMM parameters (under self_gravity section)
            if (auto unified_fmm = sg["unified_fmm"]) {
                detail::require_keys(unified_fmm, "unified_fmm",
                    {"theta_far", "softening_factor", "gravitational_constant", "tree_dimension",
                     "tree_depth", "adaptive_depth", "particle_overallocate_factor", "min_nodes_estimate",
                     "enable_global_projection", "verbose", "direct_threshold_leaf",
                     "direct_threshold_internal", "force_tree_type"});
                auto& c = config.acceleration.unified_fmm;
                if (unified_fmm["theta_far"])                    c.theta_far = unified_fmm["theta_far"].as<float>();
                if (unified_fmm["softening_factor"])             c.softening_factor = unified_fmm["softening_factor"].as<float>();
                if (unified_fmm["gravitational_constant"])       c.gravitational_constant = unified_fmm["gravitational_constant"].as<double>();

                c.adaptive_depth.enabled = false;

                // An explicit integer without an adaptive block preserves the
                // fixed-depth compatibility path. Omitted depth or `auto`
                // enables the selected-dimension adaptive controller.
                if (auto adaptive = unified_fmm["adaptive_depth"]) {
                    detail::require_keys(adaptive, "unified_fmm.adaptive_depth",
                        {"enabled", "min_depth", "max_depth", "trial_interval_evaluations",
                         "trial_warmup_evaluations", "trial_samples", "min_improvement_fraction",
                         "confirmation_trials", "capacity_headroom", "capacity_growth_trigger", "multigpu"});
                    c.adaptive_depth.enabled = true;
                    if (adaptive["enabled"])
                        c.adaptive_depth.enabled = adaptive["enabled"].as<bool>();
                    if (adaptive["min_depth"])
                        c.adaptive_depth.min_depth = adaptive["min_depth"].as<int>();
                    if (adaptive["max_depth"])
                        c.adaptive_depth.max_depth = adaptive["max_depth"].as<int>();
                    if (adaptive["trial_interval_evaluations"])
                        c.adaptive_depth.trial_interval_evaluations =
                            adaptive["trial_interval_evaluations"].as<int>();
                    if (adaptive["trial_warmup_evaluations"])
                        c.adaptive_depth.trial_warmup_evaluations =
                            adaptive["trial_warmup_evaluations"].as<int>();
                    if (adaptive["trial_samples"])
                        c.adaptive_depth.trial_samples = adaptive["trial_samples"].as<int>();
                    if (adaptive["min_improvement_fraction"])
                        c.adaptive_depth.min_improvement_fraction =
                            adaptive["min_improvement_fraction"].as<float>();
                    if (adaptive["confirmation_trials"])
                        c.adaptive_depth.confirmation_trials =
                            adaptive["confirmation_trials"].as<int>();
                    if (adaptive["capacity_headroom"])
                        c.adaptive_depth.capacity_headroom =
                            adaptive["capacity_headroom"].as<float>();
                    if (adaptive["capacity_growth_trigger"])
                        c.adaptive_depth.capacity_growth_trigger =
                            adaptive["capacity_growth_trigger"].as<float>();
                    if (auto distributed = adaptive["multigpu"]) {
                        detail::require_keys(distributed, "unified_fmm.adaptive_depth.multigpu",
                            {"local_count_reset_fraction", "secondary_objective", "trial_order"});
                        if (distributed["local_count_reset_fraction"]) {
                            c.adaptive_depth.multigpu.local_count_reset_fraction =
                                distributed["local_count_reset_fraction"].as<float>();
                        }
                        if (distributed["secondary_objective"]) {
                            const std::string value =
                                distributed["secondary_objective"].as<std::string>();
                            if (value != "mean_epoch_time") {
                                throw std::runtime_error(
                                    "unified_fmm.adaptive_depth.multigpu.secondary_objective "
                                    "must be 'mean_epoch_time'");
                            }
                            c.adaptive_depth.multigpu.secondary_objective =
                                AdaptiveDepthSecondaryObjective::MeanEpochTime;
                        }
                        if (distributed["trial_order"]) {
                            const std::string value =
                                distributed["trial_order"].as<std::string>();
                            if (value != "round_robin") {
                                throw std::runtime_error(
                                    "unified_fmm.adaptive_depth.multigpu.trial_order "
                                    "must be 'round_robin'");
                            }
                            c.adaptive_depth.multigpu.trial_order =
                                AdaptiveDepthTrialOrder::RoundRobin;
                        }
                    }
                }
                if (!unified_fmm["tree_depth"]) {
                    if (unified_fmm["adaptive_depth"] && !c.adaptive_depth.enabled) {
                        throw std::runtime_error(
                            "unified_fmm.tree_depth is required when adaptive_depth.enabled is false");
                    }
                    // Omitted depth is the public automatic-start spelling.
                    c.adaptive_depth.enabled = true;
                    c.adaptive_depth.initial_depth_auto = true;
                    c.adaptive_depth.tree_depth_was_omitted = true;
                    c.tree_depth = 0;
                } else {
                    const YAML::Node depth = unified_fmm["tree_depth"];
                    bool parsed_integer = false;
                    try {
                        c.tree_depth = depth.as<int>();
                        parsed_integer = true;
                    } catch (const YAML::Exception&) {
                        // `auto` has one stable spelling; all other non-integer
                        // values are configuration errors.
                    }
                    if (parsed_integer) {
                        c.adaptive_depth.initial_depth_auto = false;
                        c.adaptive_depth.tree_depth_was_omitted = false;
                    } else if (depth.IsScalar() &&
                               depth.as<std::string>() == "auto") {
                        if (!c.adaptive_depth.enabled) {
                            if (unified_fmm["adaptive_depth"]) {
                                throw std::runtime_error(
                                    "unified_fmm.tree_depth: auto requires "
                                    "unified_fmm.adaptive_depth.enabled: true");
                            }
                            c.adaptive_depth.enabled = true;
                        }
                        c.adaptive_depth.initial_depth_auto = true;
                        c.adaptive_depth.tree_depth_was_omitted = false;
                        c.tree_depth = 0;
                    } else {
                        throw std::runtime_error(
                            "unified_fmm.tree_depth must be an integer or 'auto'");
                    }
                }
                if (unified_fmm["particle_overallocate_factor"]) c.particle_overallocate_factor = unified_fmm["particle_overallocate_factor"].as<float>();
                if (unified_fmm["min_nodes_estimate"])           c.min_nodes_estimate = unified_fmm["min_nodes_estimate"].as<int>();
                if (unified_fmm["enable_global_projection"])     c.enable_global_projection = unified_fmm["enable_global_projection"].as<bool>();
                if (unified_fmm["verbose"])                      c.verbose = unified_fmm["verbose"].as<bool>();
                const bool has_dimension = static_cast<bool>(unified_fmm["tree_dimension"]);
                const bool has_legacy_dimension = static_cast<bool>(unified_fmm["force_tree_type"]);
                const int dimension = has_dimension
                    ? unified_fmm["tree_dimension"].as<int>()
                    : (has_legacy_dimension ? unified_fmm["force_tree_type"].as<int>() : c.tree_dimension);
                if (has_dimension && has_legacy_dimension &&
                    unified_fmm["force_tree_type"].as<int>() != dimension) {
                    throw std::runtime_error(
                        "unified_fmm.tree_dimension conflicts with deprecated force_tree_type");
                }
                if (dimension != 2 && dimension != 3) {
                    throw std::runtime_error(
                        "unified_fmm.tree_dimension must be 2 or 3 (auto/0 is not supported)");
                }
                if (has_legacy_dimension && !has_dimension) {
                    std::cerr << "WARNING: unified_fmm.force_tree_type is deprecated; "
                              << "use unified_fmm.tree_dimension.\n";
                }
                c.tree_dimension = dimension;
                if (unified_fmm["direct_threshold_leaf"])        c.direct_threshold_leaf = unified_fmm["direct_threshold_leaf"].as<int>();
                if (unified_fmm["direct_threshold_internal"])   c.direct_threshold_internal = unified_fmm["direct_threshold_internal"].as<int>();
            }
        }

        if (acc["gravity_update_rate"] || acc["gravity_macro_dt"] ||
            acc["tree_rebuild_interval"] || acc["displacement_threshold"])
            throw std::runtime_error("gravity controls belong under accelerations.self_gravity; tree_rebuild_interval was removed");

        // YORP
        if (auto yorp = acc["yorp"]) {
            if (yorp["enabled"]) config.acceleration.yorp.enabled = yorp["enabled"].as<bool>();
            if (yorp["mode"]) config.acceleration.yorp.mode = yorp["mode"].as<int32_t>();
            
            if (auto yt = yorp["time"]) {
                if (yt["start"]) config.acceleration.yorp.start_time = yt["start"].as<double>();
                if (yt["stop"]) config.acceleration.yorp.stop_time = yt["stop"].as<double>();
            }
            
            if (yorp["omega_init"]) config.acceleration.yorp.omega_init = yorp["omega_init"].as<double>();
            if (yorp["increment"]) config.acceleration.yorp.increment = yorp["increment"].as<double>();
            if (yorp["dt_check"]) config.acceleration.yorp.dt_check = yorp["dt_check"].as<double>();
            if (yorp["epsilon"]) config.acceleration.yorp.epsilon = yorp["epsilon"].as<double>();

            // Mode 3: prescribed spin period
            if (auto m3 = yorp["mode_3"]) {
                if (m3["T_start"]) config.acceleration.yorp.T_start_period = m3["T_start"].as<double>();
                if (m3["T_inter"]) config.acceleration.yorp.T_inter_period = m3["T_inter"].as<double>();
                if (m3["T_final"]) config.acceleration.yorp.T_final_period = m3["T_final"].as<double>();
                if (m3["fast_spinup_time"]) config.acceleration.yorp.fast_spinup_time = m3["fast_spinup_time"].as<double>();
                if (m3["slow_spinup_time"]) config.acceleration.yorp.slow_spinup_time = m3["slow_spinup_time"].as<double>();
            }

            // Adaptive YORP control
            if (auto adaptive = yorp["adaptive"]) {
                static const std::set<std::string> allowed_adaptive_keys = {
                    "enabled", "check_interval_seconds",
                    "pause_specific_energy_j_per_kg", "resume_specific_energy_j_per_kg",
                    "pause_confirmation_checks", "resume_confirmation_checks",
                    "axis_ratio_drop_fraction", "epsilon_reduction_fraction_per_cycle",
                    "minimum_epsilon_fraction", "max_omega_rad_s"
                };
                for (const auto& entry : adaptive) {
                    const std::string key = entry.first.as<std::string>();
                    if (allowed_adaptive_keys.count(key) == 0) {
                        throw std::runtime_error("unsupported accelerations.yorp.adaptive key: " + key);
                    }
                }
                if (adaptive["enabled"]) config.acceleration.yorp.adaptive = adaptive["enabled"].as<bool>();
                if (adaptive["check_interval_seconds"]) config.acceleration.yorp.check_interval_seconds = adaptive["check_interval_seconds"].as<double>();
                if (adaptive["pause_specific_energy_j_per_kg"]) config.acceleration.yorp.pause_specific_energy_j_per_kg = adaptive["pause_specific_energy_j_per_kg"].as<double>();
                if (adaptive["resume_specific_energy_j_per_kg"]) config.acceleration.yorp.resume_specific_energy_j_per_kg = adaptive["resume_specific_energy_j_per_kg"].as<double>();
                if (adaptive["pause_confirmation_checks"]) config.acceleration.yorp.pause_confirmation_checks = adaptive["pause_confirmation_checks"].as<int32_t>();
                if (adaptive["resume_confirmation_checks"]) config.acceleration.yorp.resume_confirmation_checks = adaptive["resume_confirmation_checks"].as<int32_t>();
                if (adaptive["axis_ratio_drop_fraction"]) config.acceleration.yorp.axis_ratio_drop_fraction = adaptive["axis_ratio_drop_fraction"].as<double>();
                if (adaptive["epsilon_reduction_fraction_per_cycle"]) config.acceleration.yorp.epsilon_reduction_fraction_per_cycle = adaptive["epsilon_reduction_fraction_per_cycle"].as<double>();
                if (adaptive["minimum_epsilon_fraction"]) config.acceleration.yorp.minimum_epsilon_fraction = adaptive["minimum_epsilon_fraction"].as<double>();
                if (adaptive["max_omega_rad_s"]) config.acceleration.yorp.max_omega_rad_s = adaptive["max_omega_rad_s"].as<double>();
                if (config.acceleration.yorp.adaptive) {
                    for (const char* required : {"enabled", "check_interval_seconds",
                            "pause_specific_energy_j_per_kg", "resume_specific_energy_j_per_kg",
                            "pause_confirmation_checks", "resume_confirmation_checks",
                            "axis_ratio_drop_fraction", "epsilon_reduction_fraction_per_cycle",
                            "minimum_epsilon_fraction", "max_omega_rad_s"}) {
                        if (!adaptive[required]) {
                            throw std::runtime_error(std::string("adaptive YORP requires accelerations.yorp.adaptive.") + required);
                        }
                    }
                }
            }
        }

        // Acceleration fields
        if (auto fields = acc["fields"]) {
            if (!fields.IsSequence()) throw std::runtime_error("accelerations.fields must be a sequence");
            if (fields.size() > MAX_ACCEL_FIELDS) throw std::runtime_error("accelerations.fields exceeds MAX_ACCEL_FIELDS");
            config.acceleration.num_accel_fields = 0;
            for (size_t i = 0; i < fields.size() && i < MAX_ACCEL_FIELDS; ++i) {
                auto f = fields[i];
                AccelFieldConfig& fc = config.acceleration.fields[config.acceleration.num_accel_fields];

                if (f["id"]) fc.id = f["id"].as<int32_t>();
                if (f["name"]) detail::safe_strcpy(fc.name, f["name"].as<std::string>());
                if (f["enabled"]) fc.enabled = f["enabled"].as<bool>();
                if (f["data_path"]) detail::safe_strcpy(fc.data_path, f["data_path"].as<std::string>());
                if (f["load_method"]) fc.load_method = detail::parse_load_method(f["load_method"].as<std::string>());
                if (f["time_interval"]) fc.time_interval = f["time_interval"].as<float>();
                if (f["batch_size"]) fc.batch_size = f["batch_size"].as<int32_t>();
                if (f["time_varying"]) fc.time_varying = f["time_varying"].as<bool>();
                if (f["space_varying"]) fc.space_varying = f["space_varying"].as<bool>();

                if (auto interp = f["interpolation"]) {
                    if (interp["time_method"]) {
                        fc.time_interp = detail::parse_interp_method(interp["time_method"].as<std::string>());
                    }
                    if (interp["space_method"]) {
                        fc.space_interp = detail::parse_interp_method(interp["space_method"].as<std::string>());
                    }
                }

                if (f["verify"])
                    throw std::runtime_error("accelerations.fields.verify is unsupported; it had no runtime effect");

                config.acceleration.num_accel_fields++;
            }
        }

        // Acceleration limiting for numerical stability
        if (auto limit = acc["acceleration_limit"]) {
            if (limit["enabled"]) config.acceleration.enable_accel_limit = limit["enabled"].as<bool>();
            if (limit["max_magnitude"]) config.acceleration.max_accel_magnitude = limit["max_magnitude"].as<float>();
        }
    }

    // --- Neighbor section ---
    if (auto neigh = root["neighbor"]) {
        detail::require_keys(neigh, "neighbor",
            {"method", "backend", "cell_size_factor", "update_frequency", "lbvh_skin_factor", "lbvh_rebuild_threshold", "verlet_buffer_enabled", "verlet_skin"});
        if (neigh["method"]) {
            std::string m = neigh["method"].as<std::string>();
            if (m == "lbvh" || m == "LBVH") {
                config.neighbor.method = NeighborMethod::LBVH;
            } else if (m == "direct" || m == "DIRECT") {
                config.neighbor.method = NeighborMethod::DIRECT;
            } else if (m == "cll" || m == "CLL" || m == "cell_linked_list") {
                config.neighbor.method = NeighborMethod::CELL_LINKED_LIST;
            } else {
                throw std::runtime_error("unsupported neighbor.method: " + m);
            }
        }
        if (neigh["backend"]) {
            const std::string backend = neigh["backend"].as<std::string>();
            if (backend == "auto") {
                config.neighbor.backend = NeighborBackend::AUTO;
            } else if (backend == "cuda") {
                config.neighbor.backend = NeighborBackend::CUDA;
            } else if (backend == "rt") {
                config.neighbor.backend = NeighborBackend::RT;
            } else {
                throw std::runtime_error("neighbor.backend must be 'auto', 'cuda', or 'rt'");
            }
        }
        if (neigh["cell_size_factor"]) {
            config.neighbor.cell_size_factor = neigh["cell_size_factor"].as<float>();
        }
        if (neigh["update_frequency"]) {
            config.neighbor.update_frequency = neigh["update_frequency"].as<int32_t>();
        }
        if (neigh["lbvh_skin_factor"]) {
            config.neighbor.lbvh_skin_factor = neigh["lbvh_skin_factor"].as<float>();
        }
        if (neigh["lbvh_rebuild_threshold"]) {
            config.neighbor.lbvh_rebuild_threshold = neigh["lbvh_rebuild_threshold"].as<float>();
        }
        if (neigh["verlet_buffer_enabled"]) {
            config.neighbor.verlet_buffer_enabled = neigh["verlet_buffer_enabled"].as<bool>();
        }
        if (neigh["verlet_skin"]) {
            config.neighbor.verlet_skin = neigh["verlet_skin"].as<float>();
        }
    }

    // --- Damping section ---
    if (auto damp = root["damping"]) {
        detail::require_keys(damp, "damping", {"enabled", "type", "coefficient"});
        if (damp["enabled"]) config.damping.enabled = damp["enabled"].as<bool>();
        if (damp["type"]) config.damping.type = detail::parse_damping_type(damp["type"].as<std::string>());
        if (damp["coefficient"]) config.damping.coefficient = damp["coefficient"].as<float>();
    }

    // --- IO section ---
    if (auto io = root["io"]) {
        if (auto inp = io["input"]) {
            if (inp["particle_file"]) {
                detail::safe_strcpy(config.io.input_particle_path, inp["particle_file"].as<std::string>());
            }
            if (inp["validate_types"]) {
                config.io.validate_input_types = inp["validate_types"].as<bool>();
            }
        }
        if (auto out = io["output"]) {
            detail::require_keys(out, "io.output",
                {"directory", "initial_state", "scalar_interval", "times", "timing", "timing_receipt_path", "fields"});
            if (out["directory"]) {
                detail::safe_strcpy(config.io.output_directory, out["directory"].as<std::string>());
            }
            if (out["initial_state"])
                config.io.output_initial_state = out["initial_state"].as<bool>();
            if (out["scalar_interval"])
                config.io.scalar_interval = out["scalar_interval"].as<int32_t>();
            if (out["times"]) {
                const auto times = out["times"];
                if (!times.IsSequence() || times.size() > 16) throw std::runtime_error("io.output.times must be a sequence of at most 16 values");
                config.io.output_schedule_count = static_cast<int32_t>(times.size());
                for (size_t i=0;i<times.size();++i) config.io.output_schedule_times[i]=times[i].as<double>();
            }
            if (out["timing"]) { const auto timing=out["timing"]; if(timing["warmup_steps"]) config.io.timing_warmup_steps=timing["warmup_steps"].as<int32_t>(); if(timing["measure_steps"]) config.io.timing_measure_steps=timing["measure_steps"].as<int32_t>(); }
            if (out["timing_receipt_path"])
                detail::safe_strcpy(config.io.timing_receipt_path,
                                   out["timing_receipt_path"].as<std::string>());

            // Standard binary fields are fixed; this list enables optional sidecars.
            if (auto fields = out["fields"]) {
                if (!fields.IsSequence()) throw std::runtime_error("io.output.fields must be a sequence");
                {
                    for (const auto& field : fields) {
                        std::string field_name = field.as<std::string>();
                        if (field_name == "gravity_force") {
                            config.io.output_gravity_force = true;
                        } else {
                            throw std::runtime_error("io.output.fields supports only gravity_force; standard v3 fields are always written");
                        }
                    }
                }
            }
        }
        if (auto ckpt = io["checkpoint"]) {
            if (ckpt["enabled"]) config.io.checkpoint_enabled = ckpt["enabled"].as<bool>();
            if (ckpt["force_fresh"]) config.io.checkpoint_force_fresh = ckpt["force_fresh"].as<bool>();
            if (ckpt["interval"]) config.io.checkpoint_interval = ckpt["interval"].as<int32_t>();
            if (ckpt["path"]) {
                detail::safe_strcpy(config.io.checkpoint_path, ckpt["path"].as<std::string>());
            }
            if (ckpt["resume_from"]) {
                detail::safe_strcpy(config.io.checkpoint_resume_from,
                                    ckpt["resume_from"].as<std::string>());
            }
            if (ckpt["rebootstrap_forces"])
                config.io.checkpoint_rebootstrap_forces =
                    ckpt["rebootstrap_forces"].as<bool>();
        }
        // Contact dump option
        if (io["dump_contacts"]) {
            config.io.dump_contacts = io["dump_contacts"].as<bool>();
        }
    }

    // --- GPU section ---
    if (auto gpu = root["gpu"]) {
        detail::require_keys(gpu, "gpu",
            {"device_id", "block_size"});
        if (gpu["device_id"]) config.gpu.device_id = gpu["device_id"].as<int32_t>();
        if (gpu["block_size"]) config.gpu.block_size = gpu["block_size"].as<int32_t>();
    }

    // --- Multi-GPU section ---
    if (auto mg = root["multigpu"]) {
        if (mg["rebalance_frequency"])
            throw std::runtime_error("multigpu.rebalance_frequency is obsolete; use rebalance_min_interval and rebalance_threshold");
        if (mg["enabled"]) config.multigpu.enabled = mg["enabled"].as<bool>();
        if (mg["max_gpus"]) config.multigpu.max_gpus = mg["max_gpus"].as<int32_t>();
        if (mg["morton_ownership"]) config.multigpu.morton_ownership = mg["morton_ownership"].as<bool>();
        if (mg["partition_depth"]) config.multigpu.partition_depth = mg["partition_depth"].as<int>();
        if (mg["halo_width_factor"]) config.multigpu.halo_width_factor = mg["halo_width_factor"].as<float>();
        if (mg["buffer_capacity_factor"]) config.multigpu.buffer_capacity_factor = mg["buffer_capacity_factor"].as<float>();
        if (mg["let"]) {
            if (mg["let"]["use_one_shot_push"]) {
                config.multigpu.let_use_one_shot_push = mg["let"]["use_one_shot_push"].as<bool>();
            }
            if (mg["let"]["shared_bucket_min_saved_bytes"]) {
                int64_t value = 0;
                try {
                    value = mg["let"]["shared_bucket_min_saved_bytes"].as<int64_t>();
                } catch (const YAML::Exception&) {
                    throw std::runtime_error(
                        "multigpu.let.shared_bucket_min_saved_bytes must be a nonnegative integer");
                }
                if (value < 0) {
                    throw std::runtime_error(
                        "multigpu.let.shared_bucket_min_saved_bytes must be a nonnegative integer");
                }
                config.multigpu.shared_bucket_min_saved_bytes =
                    static_cast<uint64_t>(value);
            }
            if (mg["let"]["max_shared_buckets"]) {
                int32_t value = 0;
                try {
                    value = mg["let"]["max_shared_buckets"].as<int32_t>();
                } catch (const YAML::Exception&) {
                    throw std::runtime_error(
                        "multigpu.let.max_shared_buckets must be an integer in [0, 8]");
                }
                if (value < 0 || value > 8) {
                    throw std::runtime_error(
                        "multigpu.let.max_shared_buckets must be in [0, 8]");
                }
                config.multigpu.max_shared_buckets = static_cast<uint8_t>(value);
            }
        }
        if (mg["rcb_alpha"]) config.multigpu.rcb_alpha = mg["rcb_alpha"].as<float>();
        if (mg["rcb_beta"]) config.multigpu.rcb_beta = mg["rcb_beta"].as<float>();
        if (mg["rcb_gamma"]) config.multigpu.rcb_gamma = mg["rcb_gamma"].as<float>();
        if (mg["rcb_cost_model"]) {
            const std::string model = mg["rcb_cost_model"].as<std::string>();
            if (!parse_rcb_cost_model(model, &config.multigpu.rcb_cost_model)) {
                throw std::runtime_error("multigpu.rcb_cost_model must be 'adaptive' or 'legacy'");
            }
        }
        if (mg["rcb_calibration_sample_interval"])
            config.multigpu.rcb_calibration_sample_interval = mg["rcb_calibration_sample_interval"].as<int32_t>();
        if (mg["rcb_calibration_half_life"])
            config.multigpu.rcb_calibration_half_life = mg["rcb_calibration_half_life"].as<float>();
        if (mg["rcb_min_predicted_improvement"])
            config.multigpu.rcb_min_predicted_improvement = mg["rcb_min_predicted_improvement"].as<float>();
        if (mg["rcb_calibration_verbose"])
            config.multigpu.rcb_calibration_verbose = mg["rcb_calibration_verbose"].as<bool>();
        if (mg["rcb_num_bins"]) config.multigpu.rcb_num_bins = mg["rcb_num_bins"].as<int32_t>();
        if (mg["rebalance_threshold"]) config.multigpu.rebalance_threshold = mg["rebalance_threshold"].as<float>();
        if (mg["rebalance_min_interval"]) config.multigpu.rebalance_min_interval = mg["rebalance_min_interval"].as<int32_t>();
    }

    // --- Clumps ---
    if (root["clumps"]) {
        auto c = root["clumps"];
        if (!c.IsMap()) throw std::runtime_error("clumps must be a mapping");
        for (const auto& entry : c) {
            const std::string key = entry.first.as<std::string>();
            if (key != "templates" && key != "instances") {
                throw std::runtime_error("clumps." + key + " is not supported by the imported-member schema");
            }
        }

        std::vector<ClumpTemplate> templates;
        std::set<std::string> template_names;
        if (c["templates"]) {
            if (!c["templates"].IsSequence()) throw std::runtime_error("clumps.templates must be a sequence");
            for (const auto& node : c["templates"]) {
                if (!node.IsMap() || !node["name"]) throw std::runtime_error("each clump template requires name");
                for (const auto& entry : node) {
                    const std::string key = entry.first.as<std::string>();
                    if (key != "name" && key != "inertia" && key != "multipole") {
                        throw std::runtime_error("unsupported clump template key: " + key);
                    }
                }
                ClumpTemplate value;
                const std::string name = node["name"].as<std::string>();
                if (name.empty() || !template_names.insert(name).second) {
                    throw std::runtime_error("clump template names must be non-empty and unique");
                }
                detail::safe_strcpy(value.name, name);
                if (node["inertia"]) {
                    const auto inertia = node["inertia"];
                    if (!inertia.IsMap()) throw std::runtime_error("clump template inertia must be a mapping");
                    constexpr const char* keys[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
                    for (int k = 0; k < 6; ++k) {
                        if (!inertia[keys[k]]) throw std::runtime_error("clump inertia requires xx, yy, zz, xy, xz, yz");
                        value.inertia[k] = inertia[keys[k]].as<double>();
                        if (!std::isfinite(value.inertia[k])) throw std::runtime_error("clump inertia values must be finite");
                    }
                    const double xx=value.inertia[0], yy=value.inertia[1], zz=value.inertia[2];
                    const double xy=value.inertia[3], xz=value.inertia[4], yz=value.inertia[5];
                    const double scale = std::max({std::fabs(xx), std::fabs(yy), std::fabs(zz), 1.0});
                    const double tol = 128.0 * std::numeric_limits<double>::epsilon() * scale;
                    const double det2 = xx*yy - xy*xy;
                    const double det3 = xx*(yy*zz-yz*yz) - xy*(xy*zz-yz*xz) + xz*(xy*yz-yy*xz);
                    if (!(xx > tol && det2 > tol*scale && det3 > tol*scale*scale)) {
                        throw std::runtime_error("clump inertia override must be symmetric positive definite");
                    }
                    value.has_inertia = true;
                }
                if (node["multipole"]) {
                    const auto multipole = node["multipole"];
                    if (!multipole.IsMap() || !multipole["basis"] || !multipole["values"]) {
                        throw std::runtime_error("clump multipole requires basis and values");
                    }
                    const std::string basis = multipole["basis"].as<std::string>();
                    if (basis != "trace_free_p3_v1") throw std::runtime_error("clump multipole basis must be trace_free_p3_v1");
                    const auto values = multipole["values"];
                    if (!values.IsSequence() || values.size() != 19) throw std::runtime_error("trace_free_p3_v1 requires exactly 19 values");
                    detail::safe_strcpy(value.multipole_basis, basis);
                    for (size_t k = 0; k < 19; ++k) {
                        value.multipole_p3[k] = values[k].as<double>();
                        if (!std::isfinite(value.multipole_p3[k])) throw std::runtime_error("clump multipole values must be finite");
                    }
                    value.has_multipole = true;
                }
                templates.push_back(value);
            }
        }
        std::sort(templates.begin(), templates.end(), [](const ClumpTemplate& a, const ClumpTemplate& b) {
            return std::strcmp(a.name, b.name) < 0;
        });

        std::vector<ClumpInstance> instances;
        if (c["instances"]) {
            if (!c["instances"].IsSequence()) throw std::runtime_error("clumps.instances must be a sequence");
            for (const auto& node : c["instances"]) {
                if (!node.IsMap() || !node["id"]) throw std::runtime_error("each clump instance requires id");
                for (const auto& entry : node) {
                    const std::string key = entry.first.as<std::string>();
                    if (key != "id" && key != "template" && key != "com" && key != "center_of_mass" && key != "orientation") {
                        throw std::runtime_error("unsupported clump instance key: " + key);
                    }
                }
                ClumpInstance value;
                value.id = node["id"].as<int32_t>();
                if (value.id <= 0) throw std::runtime_error("clump instance id must be positive");
                if (node["template"]) detail::safe_strcpy(value.template_name, node["template"].as<std::string>());
                const YAML::Node com = node["com"] ? node["com"] : node["center_of_mass"];
                if (com) {
                    if (!com.IsSequence() || com.size() != 3) throw std::runtime_error("clump instance com requires three values");
                    for (int k=0;k<3;++k) { value.com[k]=com[k].as<double>(); if(!std::isfinite(value.com[k])) throw std::runtime_error("clump CoM must be finite"); }
                    value.has_com = true;
                }
                if (node["orientation"]) {
                    const auto q = node["orientation"];
                    if (!q.IsSequence() || q.size() != 4) throw std::runtime_error("clump orientation must be quaternion [w,x,y,z]");
                    double norm2 = 0.0;
                    for (int k=0;k<4;++k) { value.orientation[k]=q[k].as<double>(); if(!std::isfinite(value.orientation[k])) throw std::runtime_error("clump orientation must be finite"); norm2 += value.orientation[k]*value.orientation[k]; }
                    const double norm = std::sqrt(norm2);
                    if (!(norm > 0.0) || std::fabs(norm-1.0) > 1.0e-10) throw std::runtime_error("clump orientation must be unit length within 1e-10");
                    for (double& component : value.orientation) component /= norm;
                    if (value.orientation[0] < 0.0) for (double& component : value.orientation) component = -component;
                    value.has_orientation = true;
                }
                instances.push_back(value);
            }
        }
        std::sort(instances.begin(), instances.end(), [](const ClumpInstance& a, const ClumpInstance& b) { return a.id < b.id; });
        for (size_t i=0;i<instances.size();++i) {
            if (instances[i].id != static_cast<int32_t>(i+1)) throw std::runtime_error("clumps.instances IDs must be exactly contiguous 1..K");
            const ClumpTemplate* selected = nullptr;
            if (instances[i].template_name[0]) {
                auto it = std::lower_bound(templates.begin(), templates.end(), instances[i].template_name,
                    [](const ClumpTemplate& a, const char* name) { return std::strcmp(a.name, name) < 0; });
                if (it == templates.end() || std::strcmp(it->name, instances[i].template_name) != 0) throw std::runtime_error("clump instance references unknown template");
                selected = &*it;
            }
            (void)selected; // CoM materiality is scale-dependent and validated after members load.
        }

        config.clump.num_templates = static_cast<int32_t>(templates.size());
        config.clump.num_instances = static_cast<int32_t>(instances.size());
        if (!templates.empty()) {
            config.clump.templates = new ClumpTemplate[templates.size()];
            std::copy(templates.begin(), templates.end(), config.clump.templates);
        }
        if (!instances.empty()) {
            config.clump.instances = new ClumpInstance[instances.size()];
            std::copy(instances.begin(), instances.end(), config.clump.instances);
        }

        if (!instances.empty()) {
            uint64_t fingerprint = UINT64_C(1469598103934665603);
            const uint32_t schema_version = 1;
            detail::fingerprint_bytes(fingerprint, &schema_version, sizeof(schema_version));
            const uint32_t nt = static_cast<uint32_t>(templates.size()), ni = static_cast<uint32_t>(instances.size());
            detail::fingerprint_bytes(fingerprint, &nt, sizeof(nt));
            for (const auto& value : templates) {
                detail::fingerprint_string(fingerprint, value.name);
                detail::fingerprint_bytes(fingerprint, &value.has_inertia, sizeof(value.has_inertia));
                if (value.has_inertia) for (double x : value.inertia) detail::fingerprint_double(fingerprint, x);
                detail::fingerprint_bytes(fingerprint, &value.has_multipole, sizeof(value.has_multipole));
                if (value.has_multipole) {
                    detail::fingerprint_string(fingerprint, value.multipole_basis);
                    for (double x : value.multipole_p3) detail::fingerprint_double(fingerprint, x);
                }
            }
            detail::fingerprint_bytes(fingerprint, &ni, sizeof(ni));
            for (const auto& value : instances) {
                detail::fingerprint_bytes(fingerprint, &value.id, sizeof(value.id));
                detail::fingerprint_string(fingerprint, value.template_name);
                detail::fingerprint_bytes(fingerprint, &value.has_com, sizeof(value.has_com));
                if (value.has_com) for (double x : value.com) detail::fingerprint_double(fingerprint, x);
                detail::fingerprint_bytes(fingerprint, &value.has_orientation, sizeof(value.has_orientation));
                if (value.has_orientation) for (double x : value.orientation) detail::fingerprint_double(fingerprint, x);
            }
            config.clump.clump_metadata_fingerprint = fingerprint == 0 ? 1 : fingerprint;
        }
    }

    // --- SPH Parameters section ---
    if (auto sph = root["sph_parameters"]) {
        // Constitutive parameters are deliberately forbidden here: they are
        // material properties and must come from material_database.  Retain
        // only numerical integration controls in this legacy-named section.
        static const std::set<std::string> allowed_sph_controls = {
            "viscosity", "smoothing_length", "num_neighbors",
            "use_kernel_correction", "use_artificial_stress", "artstr_epsilon", "artstr_n",
            "density_update_mode", "enable_shepard_filter", "gravity_softening_len",
            "cfl_number", "dt_min", "dt_max", "kernel_cutoff_factor", "hsml_min", "hsml_max",
            "relaxation", "damage_seed", "stages", "track_history", "fixed_parts", "symmetry_plane"
        };
        if (!sph.IsMap()) throw std::runtime_error("sph_parameters must be a mapping");
        for (const auto& entry : sph) {
            const std::string key = entry.first.as<std::string>();
            if (!allowed_sph_controls.count(key)) {
                throw std::runtime_error("sph_parameters." + key + " is not supported; constitutive properties belong in material_database");
            }
        }
        if (const auto fixed = sph["fixed_parts"]) {
            if (!fixed.IsSequence() || fixed.size() > 16)
                throw std::runtime_error("sph_parameters.fixed_parts requires up to 16 Part IDs");
            std::set<int32_t> ids;
            for (size_t i = 0; i < fixed.size(); ++i) {
                const int32_t id = fixed[i].as<int32_t>();
                if (id < 0 || !ids.insert(id).second)
                    throw std::runtime_error("fixed_parts must contain distinct nonnegative Part IDs");
                config.sph.fixed_parts[i] = id;
            }
            config.sph.fixed_part_count = static_cast<int32_t>(fixed.size());
        }
        if (const auto plane = sph["symmetry_plane"]) {
            if (!plane.IsMap()) throw std::runtime_error("symmetry_plane must be a mapping");
            for (const auto& entry : plane)
                if (entry.first.as<std::string>() != "normal" && entry.first.as<std::string>() != "offset")
                    throw std::runtime_error("unknown symmetry_plane key");
            const auto normal = plane["normal"];
            if (!normal.IsSequence() || normal.size() != 3)
                throw std::runtime_error("symmetry_plane.normal requires three components");
            double norm2 = 0.0;
            for (int i = 0; i < 3; ++i) {
                config.sph.symmetry_normal[i] = normal[i].as<double>();
                norm2 += config.sph.symmetry_normal[i]*config.sph.symmetry_normal[i];
            }
            const double offset = plane["offset"] ? plane["offset"].as<double>() : 0.0;
            if (!(norm2 > 0.0) || !std::isfinite(norm2) || !std::isfinite(offset))
                throw std::runtime_error("symmetry_plane requires a finite nonzero normal and finite offset");
            for (double& x : config.sph.symmetry_normal) x /= std::sqrt(norm2);
            config.sph.symmetry_offset = offset / std::sqrt(norm2);
            config.sph.symmetry_enabled = true;
        }
        if (sph["track_history"]) config.sph.track_history = sph["track_history"].as<bool>();
        if (const auto stages = sph["stages"]) {
            if (!stages.IsSequence() || stages.size() == 0 || stages.size() > 3)
                throw std::runtime_error("sph_parameters.stages requires one to three ordered stages");
            double previous_end = config.time.start_time;
            int previous_mode = -1;
            for (size_t i = 0; i < stages.size(); ++i) {
                const auto node = stages[i];
                if (!node.IsMap()) throw std::runtime_error("SPH stage must be a mapping");
                for (const auto& entry : node) {
                    const std::string key = entry.first.as<std::string>();
                    if (key != "mode" && key != "end_time" && key != "damping_rate" &&
                        key != "evolve_energy" && key != "evolve_damage" &&
                        key != "evolve_porosity" && key != "use_strength" && key != "elastic_eos")
                        throw std::runtime_error("unknown SPH stage key: " + key);
                }
                const std::string mode = node["mode"].as<std::string>();
                const int value = mode == "RELAX" ? 0 : mode == "SHOCK" ? 1 : mode == "LATE" ? 2 : -1;
                if (value < 0 || value <= previous_mode)
                    throw std::runtime_error("SPH stages must follow RELAX, SHOCK, LATE order without duplicates");
                SphStageParameters stage = sph_stage_defaults(static_cast<SphRunStage>(value));
                stage.end_time = node["end_time"].as<double>();
                if (!std::isfinite(stage.end_time) || stage.end_time <= previous_end)
                    throw std::runtime_error("SPH stage end_time must be finite and strictly increasing");
                if (node["damping_rate"]) stage.damping_rate = node["damping_rate"].as<double>();
                if (!std::isfinite(stage.damping_rate) || stage.damping_rate < 0.0 ||
                    (value != 0 && stage.damping_rate != 0.0))
                    throw std::runtime_error("SPH damping_rate must be nonnegative and is only valid in RELAX");
                if (node["evolve_energy"]) stage.evolve_energy = node["evolve_energy"].as<bool>();
                if (node["evolve_damage"]) stage.evolve_damage = node["evolve_damage"].as<bool>();
                if (node["evolve_porosity"]) stage.evolve_porosity = node["evolve_porosity"].as<bool>();
                if (node["use_strength"]) stage.use_strength = node["use_strength"].as<bool>();
                if (node["elastic_eos"]) stage.elastic_eos = node["elastic_eos"].as<bool>();
                config.sph.stages[i] = stage;
                previous_end = stage.end_time;
                previous_mode = value;
            }
            config.sph.stage_count = static_cast<int32_t>(stages.size());
            if (previous_end != config.time.stop_time)
                throw std::runtime_error("last SPH stage end_time must equal simulation.time.stop");
            if (!sph["track_history"]) config.sph.track_history = true;
        }
        // Basic SPH parameters
        if (sph["viscosity"]) config.sph.viscosity = sph["viscosity"].as<float>();
        if (sph["smoothing_length"]) config.sph.smoothing_length = sph["smoothing_length"].as<float>();
        if (sph["num_neighbors"]) config.sph.num_neighbors = sph["num_neighbors"].as<int32_t>();

        // Tillotson EOS parameters
        if (sph["tillotson_a"]) config.sph.tillotson_a = sph["tillotson_a"].as<float>();
        if (sph["tillotson_b"]) config.sph.tillotson_b = sph["tillotson_b"].as<float>();
        if (sph["tillotson_A"]) config.sph.tillotson_A = sph["tillotson_A"].as<float>();
        if (sph["tillotson_B"]) config.sph.tillotson_B = sph["tillotson_B"].as<float>();
        if (sph["tillotson_alpha"]) config.sph.tillotson_alpha = sph["tillotson_alpha"].as<float>();
        if (sph["tillotson_beta"]) config.sph.tillotson_beta = sph["tillotson_beta"].as<float>();
        if (sph["tillotson_E0"]) config.sph.tillotson_E0 = sph["tillotson_E0"].as<float>();
        if (sph["tillotson_Eiv"]) config.sph.tillotson_Eiv = sph["tillotson_Eiv"].as<float>();
        if (sph["tillotson_Ecv"]) config.sph.tillotson_Ecv = sph["tillotson_Ecv"].as<float>();

        // Strength model parameters
        if (sph["use_strength"]) config.sph.use_strength = sph["use_strength"].as<bool>();
        if (sph["shear_modulus"]) config.sph.shear_modulus = sph["shear_modulus"].as<float>();
        if (sph["yield_strength_Y0"]) config.sph.yield_strength_Y0 = sph["yield_strength_Y0"].as<float>();

        // Lundborg strength model
        if (sph["use_lundborg"]) config.sph.use_lundborg = sph["use_lundborg"].as<bool>();
        if (sph["lundborg_Yi0"]) config.sph.lundborg_Yi0 = sph["lundborg_Yi0"].as<float>();
        if (sph["lundborg_Yd0"]) config.sph.lundborg_Yd0 = sph["lundborg_Yd0"].as<float>();
        if (sph["lundborg_Ym"]) config.sph.lundborg_Ym = sph["lundborg_Ym"].as<float>();
        if (sph["lundborg_mui"]) config.sph.lundborg_mui = sph["lundborg_mui"].as<float>();
        if (sph["lundborg_mud"]) config.sph.lundborg_mud = sph["lundborg_mud"].as<float>();

        // Porosity model parameters
        if (sph["use_porosity"]) config.sph.use_porosity = sph["use_porosity"].as<bool>();
        if (sph["alpha_initial"]) config.sph.alpha_initial = sph["alpha_initial"].as<float>();
        if (sph["P_elastic_limit"]) config.sph.P_elastic_limit = sph["P_elastic_limit"].as<float>();
        if (sph["P_solid_crush"]) config.sph.P_solid_crush = sph["P_solid_crush"].as<float>();
        if (sph["crushing_exponent"]) config.sph.crushing_exponent = sph["crushing_exponent"].as<float>();

        // Match SPHSOL physics.cc:9-10: alpha0 is only meaningful when porosity is enabled.
        // SPHSOL: alpha0 = (mat.porosity == P_ALPHA) ? mat.p_alpha.alpha0 : 1.0
        // When use_porosity is false, force alpha_initial = 1.0 so that rho0 = rest_density
        // and the Tillotson EOS sees eta = rho / rest_density (not rho / (rest_density * 1.3)).
        if (!config.sph.use_porosity) {
            config.sph.alpha_initial = 1.0f;
        }

        // Kernel correction
        if (sph["use_kernel_correction"]) config.sph.use_kernel_correction = sph["use_kernel_correction"].as<bool>();

        // Artificial stress
        if (sph["use_artificial_stress"]) config.sph.use_artificial_stress = sph["use_artificial_stress"].as<bool>();
        if (sph["artstr_epsilon"]) config.sph.artstr_epsilon = sph["artstr_epsilon"].as<float>();
        if (sph["artstr_n"]) config.sph.artstr_n = sph["artstr_n"].as<float>();

        // Density update mode
        if (sph["density_update_mode"]) config.sph.density_update_mode = sph["density_update_mode"].as<int32_t>();
        if (sph["enable_shepard_filter"]) config.sph.enable_shepard_filter = sph["enable_shepard_filter"].as<bool>();
        if (sph["gravity_softening_len"]) config.sph.gravity_softening_len = sph["gravity_softening_len"].as<float>();

        // CFL timestep parameters
        if (sph["cfl_number"]) config.sph.cfl_number = sph["cfl_number"].as<float>();
        if (sph["dt_min"]) config.sph.dt_min = sph["dt_min"].as<float>();
        if (sph["dt_max"]) config.sph.dt_max = sph["dt_max"].as<float>();
        if (sph["kernel_cutoff_factor"]) config.sph.kernel_cutoff_factor = sph["kernel_cutoff_factor"].as<float>();
        if (sph["hsml_min"]) config.sph.hsml_min = sph["hsml_min"].as<float>();
        if (sph["hsml_max"]) config.sph.hsml_max = sph["hsml_max"].as<float>();
        if (sph["damage_seed"]) config.sph.damage_seed = sph["damage_seed"].as<uint64_t>();

        if (const auto relaxation = sph["relaxation"]) {
            if (!relaxation.IsMap())
                throw std::runtime_error("sph_parameters.relaxation must be a mapping");
            static const std::set<std::string> allowed_relaxation_controls = {
                "disable_strength", "disable_damage"
            };
            for (const auto& entry : relaxation) {
                const std::string key = entry.first.as<std::string>();
                if (!allowed_relaxation_controls.count(key))
                    throw std::runtime_error(
                        "sph_parameters.relaxation." + key + " is not supported");
            }
            if (relaxation["disable_strength"])
                config.sph.relaxation_disable_strength =
                    relaxation["disable_strength"].as<bool>();
            if (relaxation["disable_damage"])
                config.sph.relaxation_disable_damage =
                    relaxation["disable_damage"].as<bool>();
        }

    }

    // Inactive FMM defaults must not activate controllers in non-gravity runners.
    if (!config.acceleration.enable_self_gravity ||
        config.acceleration.gravity_method != SelfGravityMethod::BFFMM)
        config.acceleration.unified_fmm.adaptive_depth.enabled = false;

    // The compact-support gravity correction traverses the current CUDA BVH
    // after h evolves, rather than a cached hydro adjacency from before it.
    if (config.physics == SimulationPhysics::SPH &&
        config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
        if (!(std::isfinite(config.sph.hsml_min) && config.sph.hsml_min > 0 &&
              std::isfinite(config.sph.hsml_max) && config.sph.hsml_max >= config.sph.hsml_min))
            throw std::runtime_error("SPH BFFMM requires finite positive ordered h bounds");
        if (config.neighbor.backend == NeighborBackend::RT)
            throw std::runtime_error("SPH BFFMM gravity currently requires neighbor.backend: cuda");
        if (!(std::isfinite(config.sph.kernel_cutoff_factor) && config.sph.kernel_cutoff_factor >= 2.0f) ||
            (config.multigpu.enabled && !(std::isfinite(config.multigpu.halo_width_factor) &&
                                         config.multigpu.halo_width_factor >= 1.0f)))
            throw std::runtime_error("SPH BFFMM requires kernel_cutoff_factor >= 2 and multi-GPU halo_width_factor >= 1");
        config.neighbor.backend = NeighborBackend::CUDA;
    }

    // AUTO remains valid for all modes; unsupported RT workloads use their
    // existing CUDA path.  An explicit RT request must never silently fall back.
    if (config.neighbor.backend == NeighborBackend::RT) {
        if (config.neighbor.method != NeighborMethod::LBVH)
            throw std::runtime_error("neighbor.backend: rt requires neighbor.method: lbvh");
        if (config.multigpu.enabled)
            throw std::runtime_error("neighbor.backend: rt currently supports one GPU only");
        if (config.shear_box.enabled)
            throw std::runtime_error("neighbor.backend: rt does not support shear_box");
        for (int32_t i = 0; i < config.num_walls; ++i) {
            if (config.walls[i].enabled && config.walls[i].type == WallType::PERIODICAL)
                throw std::runtime_error("neighbor.backend: rt requires non-periodic boundaries");
        }
    }

    if (config.multigpu.enabled && config.multigpu.max_gpus > 1 &&
        config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM &&
        !config.multigpu.morton_ownership)
        throw std::runtime_error("Multi-GPU BFFMM requires multigpu.morton_ownership=true");

    if (config.shear_box.enabled) {
        const auto shear = root["shear_box"];
        if (!shear["omega"] && !shear["central_body"])
            throw std::runtime_error(
                "enabled shear_box requires either omega or central_body");
        if (!(config.shear_box.omega > 0.0) ||
            !std::isfinite(config.shear_box.omega))
            throw std::runtime_error("shear_box omega must be finite and > 0");
        if (!std::isfinite(config.shear_box.shear_parameter))
            throw std::runtime_error("shear_box.shear_parameter must be finite");
        if (!(config.shear_box.vertical_frequency >= 0.0) ||
            !std::isfinite(config.shear_box.vertical_frequency))
            throw std::runtime_error(
                "shear_box.vertical_frequency must be finite and >= 0");
        if (!std::isfinite(config.shear_box.time_origin))
            throw std::runtime_error("shear_box.time_origin must be finite");
        if (!(config.shear_box.orbital_dt_factor > 0.0) ||
            !std::isfinite(config.shear_box.orbital_dt_factor))
            throw std::runtime_error(
                "shear_box.orbital_dt_factor must be finite and > 0");
        if (config.shear_box.gravity_layers_x < 0 ||
            config.shear_box.gravity_layers_y < 0)
            throw std::runtime_error(
                "shear_box gravity image layers must be non-negative integers");
        if (!(config.shear_box.gravity_cutoff_radius >= 0.0) ||
            !std::isfinite(config.shear_box.gravity_cutoff_radius))
            throw std::runtime_error(
                "shear_box.gravity_images.cutoff_radius must be null or finite and >= 0");
        if (!(config.shear_box.force_balance_warn_threshold > 0.0) ||
            !std::isfinite(config.shear_box.force_balance_warn_threshold))
            throw std::runtime_error(
                "shear_box.force_balance_warn_threshold must be finite and > 0");
        if (config.shear_box.shear_parameter >= 2.0) {
            std::cerr << "WARNING: shear_box.shear_parameter >= 2 selects the "
                         "marginal/hyperbolic Hill map; verify that this unstable "
                         "local model is intentional" << std::endl;
        }
        if (config.physics != SimulationPhysics::DEM)
            throw std::runtime_error("shear_box currently supports DEM only");
        if (config.multigpu.enabled) {
            config.multigpu.morton_ownership = true;
            if (config.multigpu.max_gpus > 32)
                throw std::runtime_error("multi-GPU shear_box currently supports at most 32 GPU owners");
            if (config.acceleration.enable_self_gravity &&
                config.acceleration.gravity_method != SelfGravityMethod::BFFMM)
                throw std::runtime_error("multi-GPU shear_box gravity requires unified FMM");
            if (config.acceleration.enable_self_gravity &&
                config.acceleration.unified_fmm.tree_dimension != 3)
                throw std::runtime_error("multi-GPU shear_box requires a 3D gravity tree; ownership uses XY only");
        }
        if (config.clump.num_instances > 0)
            throw std::runtime_error("shear_box does not support clumps");
        if (config.neighbor.method != NeighborMethod::LBVH &&
            config.neighbor.method != NeighborMethod::CELL_LINKED_LIST)
            throw std::runtime_error("shear_box requires neighbor.method: lbvh or cll");
        if (config.neighbor.update_frequency != 1)
            throw std::runtime_error(
                "shear_box currently requires neighbor.update_frequency: 1");
        const auto domain = root["domain"];
        if (!domain || !domain.IsMap() || !domain["box_min"] ||
            !domain["box_max"] || config.domain.auto_bounds)
            throw std::runtime_error(
                "shear_box requires explicit fixed domain.box_min/box_max");
        const double lx = static_cast<double>(config.domain.box_max.x) -
                          static_cast<double>(config.domain.box_min.x);
        const double ly = static_cast<double>(config.domain.box_max.y) -
                          static_cast<double>(config.domain.box_min.y);
        const double lz = static_cast<double>(config.domain.box_max.z) -
                          static_cast<double>(config.domain.box_min.z);
        if (!(lx > 0.0) || !(ly > 0.0) || !(lz > 0.0) ||
            !std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz))
            throw std::runtime_error(
                "shear_box requires finite positive domain extents");
        if (config.coord_system == CoordinateSystem::ROTATING ||
            config.rotating_frame.enabled)
            throw std::runtime_error(
                "shear_box owns Coriolis and tidal dynamics; rotating_frame is incompatible");

        // Legacy modules default to enabled in historical configurations.  In
        // shear mode omission means disabled, while an explicit request fails.
        const auto accelerations = root["accelerations"];
        const auto inertial = accelerations ? accelerations["inertial"] : YAML::Node();
        const auto tidal = accelerations ? accelerations["tidal"] : YAML::Node();
        if (inertial && inertial["enabled"] && inertial["enabled"].as<bool>())
            throw std::runtime_error(
                "shear_box is incompatible with accelerations.inertial.enabled: true");
        if (tidal && tidal["enabled"] && tidal["enabled"].as<bool>())
            throw std::runtime_error(
                "shear_box is incompatible with accelerations.tidal.enabled: true");
        config.acceleration.enable_inertial = false;
        config.acceleration.enable_tidal = false;
        if (config.acceleration.yorp.enabled || config.acceleration.yorp.adaptive)
            throw std::runtime_error("shear_box is incompatible with YORP");
        if (config.acceleration.enable_accel_limit)
            throw std::runtime_error(
                "shear_box is incompatible with acceleration limiting");
        for (int32_t i = 0; i < config.acceleration.num_accel_fields; ++i) {
            const AccelFieldConfig& field = config.acceleration.fields[i];
            if (field.enabled && field.space_varying)
                throw std::runtime_error(
                    "shear_box acceleration fields must be spatially uniform");
        }

        const auto damping = root["damping"];
        if (damping && damping["enabled"] && damping["enabled"].as<bool>())
            throw std::runtime_error(
                "shear_box is incompatible with global viscous/local damping; "
                "contact damping remains available through the material model");
        config.damping.enabled = false;
        if (config.termination != TerminationCondition::TIME)
            throw std::runtime_error(
                "shear_box supports time-based termination only");

        for (int32_t i = 0; i < config.num_walls; ++i) {
            const WallConfig& wall = config.walls[i];
            if (!wall.enabled) continue;
            const bool horizontal_plane =
                (wall.type == WallType::PLAIN || wall.type == WallType::MOVING) &&
                std::isfinite(wall.normal.x) && std::isfinite(wall.normal.y) &&
                std::isfinite(wall.normal.z) && wall.normal.x == 0.0 &&
                wall.normal.y == 0.0 && wall.normal.z != 0.0;
            if (!horizontal_plane)
                throw std::runtime_error(
                    "shear_box supports only horizontal z-normal plane walls");
        }

        if (config.acceleration.enable_self_gravity &&
            config.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
            const auto sg = accelerations ? accelerations["self_gravity"] : YAML::Node();
            const auto ufmm = sg ? sg["unified_fmm"] : YAML::Node();
            if (ufmm && ufmm["enable_global_projection"] &&
                ufmm["enable_global_projection"].as<bool>())
                throw std::runtime_error(
                    "finite shear replica gravity is incompatible with global "
                    "force/torque projection");
            config.acceleration.unified_fmm.enable_global_projection = false;
        }
    }
    if (!std::isfinite(config.time.dt_upper_bound) ||
        config.time.dt_upper_bound < 0.0 ||
        (root["simulation"] && root["simulation"]["time"] &&
         root["simulation"]["time"]["dt"] &&
         !(config.time.dt_upper_bound > 0.0)))
        throw std::runtime_error("simulation.time.dt must be finite and > 0 when supplied");
    if (!std::isfinite(config.time.dt_factor) || !(config.time.dt_factor > 0.0))
        throw std::runtime_error("simulation.time.dt_factor must be finite and > 0");

    if (config.multigpu.buffer_capacity_factor < 1.0f) {
        throw std::runtime_error("multigpu.buffer_capacity_factor must be >= 1.0");
    }
    const bool morton_fmm = config.multigpu.morton_ownership &&
        config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM;
    if (config.multigpu.max_gpus <= 0)
        throw std::runtime_error("multigpu.max_gpus must be positive");
    if (!morton_fmm && config.multigpu.max_gpus > MAX_MULTI_GPUS)
        throw std::runtime_error("legacy multigpu.max_gpus must be <= " +
                                 std::to_string(MAX_MULTI_GPUS));
    if (config.multigpu.rcb_calibration_sample_interval <= 0) {
        throw std::runtime_error("multigpu.rcb_calibration_sample_interval must be > 0");
    }
    if (!(config.multigpu.rcb_calibration_half_life > 0.0f)) {
        throw std::runtime_error("multigpu.rcb_calibration_half_life must be > 0");
    }
    if (config.multigpu.rcb_min_predicted_improvement < 0.0f) {
        throw std::runtime_error("multigpu.rcb_min_predicted_improvement must be >= 0");
    }
    if (config.sph.stage_count > 0 || config.sph.track_history ||
        config.sph.fixed_part_count > 0 || config.sph.symmetry_enabled) {
        if (config.physics != SimulationPhysics::SPH)
            throw std::runtime_error("SPH stages/history/boundaries require simulation.physics: sph");
        if (config.sph.stage_count > 0 && config.termination != TerminationCondition::TIME)
            throw std::runtime_error("SPH stage sequences require time termination; choose explicit stage end times");
        if (config.sph.track_history && config.sph.density_update_mode != 1)
            throw std::runtime_error("SPH cold-energy history requires density_update_mode: 1");
        if (config.multigpu.enabled && (config.sph.fixed_part_count > 0 || config.sph.symmetry_enabled))
            throw std::runtime_error("SPH fixed_parts and symmetry_plane currently require single-GPU execution");
        if (config.sph.symmetry_enabled) {
            if (config.neighbor.method != NeighborMethod::LBVH)
                throw std::runtime_error("SPH symmetry_plane requires neighbor.method: lbvh");
            if (config.acceleration.enable_self_gravity || config.shear_box.enabled)
                throw std::runtime_error("SPH symmetry_plane does not support self-gravity or shear_box");
            for (int i = 0; i < config.num_walls; ++i)
                if (config.walls[i].enabled && config.walls[i].type == WallType::PERIODICAL)
                    throw std::runtime_error("SPH symmetry_plane cannot be combined with periodic boundaries");
        }
    }
    if (config.physics == SimulationPhysics::SPH && config.clump.num_instances > 0) {
        throw std::runtime_error("SPH configurations cannot define positive-ID clump instances");
    }
    if (config.acceleration.gravity_update_rate <= 0) {
        throw std::runtime_error("gravity_update_rate must be a positive integer");
    }
    if (config.physics == SimulationPhysics::SPH &&
        !(std::isfinite(config.sph.cfl_number) && config.sph.cfl_number > 0.0f &&
          config.sph.cfl_number <= 1.0f)) {
        throw std::runtime_error(
            "sph_parameters.cfl_number must be finite and in (0, 1]");
    }
    if (config.io.scalar_interval < 0) {
        throw std::runtime_error("io.output.scalar_interval must be >= 0");
    }
    if (config.io.output_schedule_count < 0 || config.io.output_schedule_count > 16 || config.io.timing_warmup_steps < 0 || config.io.timing_measure_steps < 0) throw std::runtime_error("invalid output schedule or timing window");
    for (int32_t i=0;i<config.io.output_schedule_count;++i) {
        const double value=config.io.output_schedule_times[i];
        if (!std::isfinite(value) || value < config.time.start_time || value > config.time.stop_time || (i && !(value>config.io.output_schedule_times[i-1]))) throw std::runtime_error("io.output.times must be finite, strictly increasing, and within simulation time");
    }
    {
        const ExternalBodiesConfig& bodies = config.acceleration.external_bodies;
        if (bodies.enabled) {
            if (config.physics != SimulationPhysics::DEM)
                throw std::runtime_error("external_bodies requires simulation.physics: dem");
            if (!bodies.saturn_enabled || !std::isfinite(bodies.saturn_mass) ||
                !(bodies.saturn_mass > 0.0) ||
                !std::isfinite(bodies.saturn_absorb_radius) ||
                bodies.saturn_absorb_radius < 0.0)
                throw std::runtime_error(
                    "external_bodies requires an enabled Saturn with finite mass > 0 "
                    "and absorb_radius >= 0");
            if (bodies.satellite_enabled &&
                (!std::isfinite(bodies.satellite_mass) || !(bodies.satellite_mass > 0.0) ||
                 !std::isfinite(bodies.satellite_orbit_radius) ||
                 !(bodies.satellite_orbit_radius > 0.0) ||
                 !std::isfinite(bodies.satellite_phase)))
                throw std::runtime_error(
                    "enabled external_bodies satellite requires finite mass and orbit_radius > 0");
            if (config.shear_box.enabled)
                throw std::runtime_error("external_bodies is incompatible with shear_box");
            if (config.multigpu.enabled)
                throw std::runtime_error("external_bodies is currently single-GPU only");
            if (config.clump.num_instances != 0)
                throw std::runtime_error("external_bodies does not support clump instances");
            for (int32_t i = 0; i < config.num_walls; ++i) {
                if (config.walls[i].enabled &&
                    config.walls[i].type == WallType::PERIODICAL)
                    throw std::runtime_error(
                        "external_bodies requires non-periodic boundaries");
            }
        }
    }
    if (!std::isfinite(config.acceleration.gravity_macro_dt) ||
        config.acceleration.gravity_macro_dt < 0.0) {
        throw std::runtime_error("gravity_macro_dt must be finite and >= 0 seconds");
    }
    if (config.acceleration.gravity_macro_dt > 0.0 &&
        config.acceleration.gravity_update_rate != 1) {
        throw std::runtime_error(
            "gravity_macro_dt requires gravity_update_rate: 1 or an omitted gravity_update_rate");
    }
    {
        const DirectGravityConfig& direct = config.acceleration.direct_gravity;
        if (!std::isfinite(direct.gravitational_constant) ||
            !(direct.gravitational_constant > 0.0)) {
            throw std::runtime_error(
                "accelerations.self_gravity.direct.gravitational_constant must be "
                "finite and > 0");
        }
        if (direct.sph_softening_mode ==
            DirectSphSofteningMode::QUERY_PARTICLE) {
            if (config.physics != SimulationPhysics::SPH ||
                !config.acceleration.enable_self_gravity ||
                config.acceleration.gravity_method != SelfGravityMethod::DIRECT) {
                throw std::runtime_error(
                    "direct query_particle SPH softening requires enabled direct "
                    "self-gravity in simulation.physics: sph");
            }
            if (config.shear_box.enabled) {
                throw std::runtime_error(
                    "direct query_particle SPH softening is not supported with shear_box");
            }
        }
    }
    {
        const UnifiedFMMConfig& fmm = config.acceleration.unified_fmm;
        const AdaptiveDepthConfig& adaptive = fmm.adaptive_depth;
        if (!std::isfinite(fmm.theta_far) || std::abs(fmm.theta_far - 0.8f) > 1.0e-6f)
            throw std::runtime_error("unified_fmm.theta_far must be 0.8 for the bundled G/stencil tables");
        if (fmm.tree_dimension != 2 && fmm.tree_dimension != 3) {
            throw std::runtime_error("unified_fmm.tree_dimension must be 2 or 3");
        }
        if (!adaptive.initial_depth_auto &&
            (fmm.tree_depth < 2 || fmm.tree_depth > 20)) {
            throw std::runtime_error("unified_fmm.tree_depth must be in [2, 20]");
        }
        if (adaptive.enabled && config.acceleration.enable_self_gravity &&
            config.acceleration.gravity_method == SelfGravityMethod::BFFMM) {
            if (adaptive.min_depth < 2 || adaptive.min_depth > 20 ||
                adaptive.max_depth < 2 || adaptive.max_depth > 20 ||
                adaptive.min_depth > adaptive.max_depth) {
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth min_depth/max_depth must define "
                    "an ordered range within [2, 20]");
            }
            if (!adaptive.initial_depth_auto &&
                (fmm.tree_depth < adaptive.min_depth ||
                 fmm.tree_depth > adaptive.max_depth)) {
                throw std::runtime_error(
                    "unified_fmm.tree_depth must lie inside the adaptive depth range");
            }
            if (adaptive.trial_interval_evaluations <= 0)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.trial_interval_evaluations must be > 0");
            if (adaptive.trial_warmup_evaluations < 0)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.trial_warmup_evaluations must be >= 0");
            if (adaptive.trial_samples <= 0)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.trial_samples must be > 0");
            if (!std::isfinite(adaptive.min_improvement_fraction) ||
                adaptive.min_improvement_fraction < 0.0f ||
                adaptive.min_improvement_fraction >= 1.0f)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.min_improvement_fraction must be finite and in [0, 1)");
            if (adaptive.confirmation_trials <= 0)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.confirmation_trials must be > 0");
            if (!std::isfinite(adaptive.capacity_headroom) ||
                adaptive.capacity_headroom < 1.0f)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.capacity_headroom must be finite and >= 1");
            if (!std::isfinite(adaptive.capacity_growth_trigger) ||
                adaptive.capacity_growth_trigger <= 0.0f ||
                adaptive.capacity_growth_trigger > 1.0f)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.capacity_growth_trigger must be finite and in (0, 1]");
            if (!std::isfinite(adaptive.multigpu.local_count_reset_fraction) ||
                adaptive.multigpu.local_count_reset_fraction <= 0.0f ||
                adaptive.multigpu.local_count_reset_fraction > 1.0f)
                throw std::runtime_error(
                    "unified_fmm.adaptive_depth.multigpu.local_count_reset_fraction "
                    "must be finite and in (0, 1]");
            if (config.shear_box.enabled) {
                if (!config.multigpu.enabled || !config.multigpu.morton_ownership)
                    throw std::runtime_error(
                        "adaptive FMM depth for shear_box requires the cudadem_multigpu common controller (multigpu.enabled: true)");
                const auto& box = config.domain;
                const int extra = shear_fmm_capacity::depth_offset(
                    double(box.box_max.x - box.box_min.x),
                    double(box.box_max.y - box.box_min.y),
                    double(box.box_max.z - box.box_min.z),
                    config.shear_box.gravity_layers_x, config.shear_box.gravity_layers_y);
                if (adaptive.max_depth + extra > 20)
                    throw std::runtime_error(
                        "adaptive shear_box max_depth plus image depth offset exceeds effective depth limit 20");
            }
        }
    }
    const bool respa_requested = config.acceleration.enable_self_gravity &&
        (config.acceleration.gravity_update_rate > 1 ||
         config.acceleration.gravity_macro_dt > 0.0);
    if (respa_requested && config.damping.enabled &&
        config.damping.type == DampingType::LOCAL) {
        throw std::runtime_error(
            "RESPA is incompatible with Cundall local damping because that nonlinear "
            "operator requires the unsplit total force; use viscous damping or "
            "gravity_update_rate: 1");
    }
    if (respa_requested && config.acceleration.enable_accel_limit) {
        throw std::runtime_error(
            "RESPA is incompatible with acceleration limiting because clamping the split "
            "force levels changes the physical operator; use gravity_update_rate: 1");
    }
    if (config.acceleration.yorp.adaptive) {
        const YorpConfig& y = config.acceleration.yorp;
        if (config.acceleration.yorp.stop_time <= 0.0) {
            config.acceleration.yorp.stop_time = config.time.stop_time;
        }
        if (!config.acceleration.yorp.enabled)
            throw std::runtime_error("adaptive YORP requires accelerations.yorp.enabled: true");
        if (config.physics != SimulationPhysics::DEM)
            throw std::runtime_error("adaptive YORP requires simulation.physics: dem");
        if (config.termination != TerminationCondition::ADAPTIVE_YORP)
            throw std::runtime_error("adaptive YORP requires simulation.termination.condition: adaptive_yorp");
        if (y.mode != 2)
            throw std::runtime_error("adaptive YORP requires accelerations.yorp.mode: 2");
        if (!(y.epsilon > 0.0) || !std::isfinite(y.epsilon))
            throw std::runtime_error("adaptive YORP requires finite accelerations.yorp.epsilon > 0");
        if (!std::isfinite(y.start_time) || !std::isfinite(y.stop_time) || !(y.stop_time > y.start_time))
            throw std::runtime_error("adaptive YORP requires finite stop time > start time");
        if (config.rotating_frame.enabled && config.acceleration.enable_inertial &&
            !config.acceleration.inertial_constant_omega)
            throw std::runtime_error("adaptive YORP does not support a time-varying rotating frame");
        if (!(y.check_interval_seconds > 0.0) || !std::isfinite(y.check_interval_seconds))
            throw std::runtime_error("adaptive YORP check_interval_seconds must be finite and > 0");
        if (!(y.pause_specific_energy_j_per_kg >= 0.0) || !std::isfinite(y.pause_specific_energy_j_per_kg))
            throw std::runtime_error("adaptive YORP pause_specific_energy_j_per_kg must be finite and >= 0");
        if (!(y.resume_specific_energy_j_per_kg >= 0.0) || !std::isfinite(y.resume_specific_energy_j_per_kg) ||
            y.resume_specific_energy_j_per_kg >= y.pause_specific_energy_j_per_kg)
            throw std::runtime_error("adaptive YORP resume_specific_energy_j_per_kg must be finite, >= 0, and < pause threshold");
        if (y.pause_confirmation_checks <= 0 || y.resume_confirmation_checks <= 0)
            throw std::runtime_error("adaptive YORP confirmation check counts must be > 0");
        if (!(y.axis_ratio_drop_fraction >= 0.0 && y.axis_ratio_drop_fraction <= 1.0) ||
            !std::isfinite(y.axis_ratio_drop_fraction))
            throw std::runtime_error("adaptive YORP axis_ratio_drop_fraction must be in [0, 1]");
        if (!(y.epsilon_reduction_fraction_per_cycle >= 0.0 && y.epsilon_reduction_fraction_per_cycle <= 1.0) ||
            !std::isfinite(y.epsilon_reduction_fraction_per_cycle))
            throw std::runtime_error("adaptive YORP epsilon_reduction_fraction_per_cycle must be in [0, 1]");
        if (!(y.minimum_epsilon_fraction > 0.0 && y.minimum_epsilon_fraction <= 1.0) ||
            !std::isfinite(y.minimum_epsilon_fraction))
            throw std::runtime_error("adaptive YORP minimum_epsilon_fraction must be in (0, 1]");
        if (!(y.max_omega_rad_s > 0.0) || !std::isfinite(y.max_omega_rad_s))
            throw std::runtime_error("adaptive YORP max_omega_rad_s must be finite and > 0");
    }
    if (config.clump.num_instances > 0 &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM &&
        config.acceleration.unified_fmm.adaptive_depth.enabled) {
        throw std::runtime_error(
            "adaptive unified_fmm tree depth does not support clump gravity");
    }
    if (config.clump.num_instances > 0 &&
        config.acceleration.gravity_method == SelfGravityMethod::BFFMM &&
        !config.acceleration.unified_fmm.enable_global_projection) {
        throw std::runtime_error("BFFMM clump runs require enable_global_projection: true");
    }

    if (morton_fmm) {
        const auto& fmm = config.acceleration.unified_fmm;
        const int minimum = fmm.adaptive_depth.enabled ? fmm.adaptive_depth.min_depth : std::max(2, fmm.tree_depth);
        if (config.multigpu.partition_depth == 0) config.multigpu.partition_depth = std::max(2, minimum);
        if (config.multigpu.partition_depth < 2 || config.multigpu.partition_depth > 20 ||
            config.multigpu.partition_depth > minimum)
            throw std::runtime_error("multigpu.partition_depth must be in [2,20] and <= minimum computation depth");
    }
    return config;
}

// -----------------------------------------------------------------------------
// Copy SimConfig to GPU constant memory
// -----------------------------------------------------------------------------
void copy_config_to_device(const SimConfig& host_config) {
    cudaError_t err = cudaMemcpyToSymbol(dev_config_storage, &host_config, sizeof(SimConfig));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy config to device: ") +
                                cudaGetErrorString(err));
    }
}

// -----------------------------------------------------------------------------
// Print configuration summary
// -----------------------------------------------------------------------------
void print_config(const SimConfig& config) {
    std::cout << "=== CudaDEM Configuration ===" << std::endl;
    std::cout << "Name: " << config.name << std::endl;
    std::cout << "Type: " << (config.sim_type == SimulationType::ASTEROID_SHAKE ?
                              "Asteroid Shake" : "Surface Slope") << std::endl;
    std::cout << "Coordinate: ";
    if (config.coord_system == CoordinateSystem::MASS_CENTER) {
        std::cout << "Mass Center";
    } else if (config.coord_system == CoordinateSystem::ROTATING) {
        std::cout << "Rotating";
    } else {
        std::cout << "Surface";
    }
    std::cout << std::endl;

    if (config.shear_box.enabled) {
        std::cout << "\n--- Shear Box ---" << std::endl;
        std::cout << "Omega: " << config.shear_box.omega << " rad/s" << std::endl;
        std::cout << "q: " << config.shear_box.shear_parameter
                  << ", vertical frequency: " << config.shear_box.vertical_frequency
                  << " rad/s" << std::endl;
        std::cout << "Input velocity: "
                  << (config.shear_box.input_velocity == ShearInputVelocity::TOTAL ?
                      "total" : "peculiar") << std::endl;
        std::cout << "Gravity image layers: "
                  << config.shear_box.gravity_layers_x << " x "
                  << config.shear_box.gravity_layers_y;
        if (config.shear_box.gravity_cutoff_radius > 0.0)
            std::cout << ", cutoff=" << config.shear_box.gravity_cutoff_radius << " m";
        else
            std::cout << ", no cutoff";
        std::cout << std::endl;
    }

    // Rotating frame info
    if (config.coord_system == CoordinateSystem::ROTATING || config.rotating_frame.enabled) {
        std::cout << "\n--- Rotating Frame ---" << std::endl;
        std::cout << "Omega: (" << config.rotating_frame.omega.x << ", "
                  << config.rotating_frame.omega.y << ", "
                  << config.rotating_frame.omega.z << ") rad/s" << std::endl;
        double omega_mag = std::sqrt(config.rotating_frame.omega.x * config.rotating_frame.omega.x +
                                     config.rotating_frame.omega.y * config.rotating_frame.omega.y +
                                     config.rotating_frame.omega.z * config.rotating_frame.omega.z);
        if (omega_mag > 1e-10) {
            std::cout << "Rotation period: " << (2.0 * M_PI / omega_mag / 3600.0) << " h" << std::endl;
        }

    }
    std::cout << "\n--- Physics and materials ---" << std::endl;
    std::cout << "Physics: " << (config.physics == SimulationPhysics::DEM ? "DEM" : "SPH") << std::endl;
    std::cout << "Particle file: " << config.io.input_particle_path << std::endl;
    std::cout << "Material database: " << config.material_database.path << std::endl;
    std::cout << "Material fingerprint: 0x" << std::hex
              << config.material_database.fingerprint << std::dec << std::endl;
    if (config.physics == SimulationPhysics::DEM) {
        std::cout << "Input-derived radius: [" << config.particle.radius_min << ", "
                  << config.particle.radius_max << "] m" << std::endl;
        std::cout << "Damping model: "
                  << (config.dem_damping_model == DemDampingModel::VISCOUS ?
                      "viscous" : "restitution") << std::endl;
    }

    std::cout << "\n--- Accelerations ---" << std::endl;
    std::cout << "Inertial: " << (config.acceleration.enable_inertial ? "Enabled" : "Disabled");
    if (config.acceleration.enable_inertial && config.acceleration.inertial_constant_omega) {
        std::cout << " (constant omega)";
    }
    std::cout << std::endl;
    std::cout << "Tidal: " << (config.acceleration.enable_tidal ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Self-gravity: " << (config.acceleration.enable_self_gravity ? "Enabled" : "Disabled");
    if (config.acceleration.enable_self_gravity) {
        switch (config.acceleration.gravity_method) {
            case SelfGravityMethod::BFFMM:
                std::cout << " (UnifiedFMM, θ=" << config.acceleration.unified_fmm.theta_far
                          << ", tree_dimension=" << config.acceleration.unified_fmm.tree_dimension
                          << ", gravity_respa=" << config.acceleration.gravity_update_rate;
                if (config.acceleration.unified_fmm.adaptive_depth.enabled) {
                    std::cout << ", tree_depth=";
                    if (config.acceleration.unified_fmm.adaptive_depth.initial_depth_auto)
                        std::cout << "auto";
                    else
                        std::cout << config.acceleration.unified_fmm.tree_depth;
                    std::cout << "["
                              << config.acceleration.unified_fmm.adaptive_depth.min_depth
                              << ".."
                              << config.acceleration.unified_fmm.adaptive_depth.max_depth
                              << "]";
                } else {
                    std::cout << ", tree_depth="
                              << config.acceleration.unified_fmm.tree_depth;
                }
                std::cout << ")";
                break;
            case SelfGravityMethod::DIRECT:
                std::cout << " (Direct N^2, G="
                          << config.acceleration.direct_gravity.gravitational_constant;
                if (config.physics == SimulationPhysics::SPH) {
                    std::cout << ", sph_softening="
                              << (config.acceleration.direct_gravity.sph_softening_mode ==
                                          DirectSphSofteningMode::QUERY_PARTICLE
                                      ? "query_particle" : "symmetric_mean");
                }
                std::cout << ")";
                break;
        }
    }
    std::cout << std::endl;
    std::cout << "Accel fields: " << config.acceleration.num_accel_fields << std::endl;
    std::cout << "Accel limit: " << (config.acceleration.enable_accel_limit ? "Enabled" : "Disabled");
    if (config.acceleration.enable_accel_limit) {
        std::cout << " (max=" << config.acceleration.max_accel_magnitude << " m/s²)";
    }
    std::cout << std::endl;

    std::cout << "\n--- Walls ---" << std::endl;
    std::cout << "Number of walls: " << config.num_walls << std::endl;
    for (int i = 0; i < config.num_walls; ++i) {
        const WallConfig& wc = config.walls[i];
        if (wc.type == WallType::BONDED_RECT) {
            std::cout << "  [bonded_rect] half_extents=["
                      << wc.half_extents[0] << "," << wc.half_extents[1]
                      << "," << wc.half_extents[2]
                      << "] mass=" << wc.bonded_mass
                      << " bonded_rect_wall" << std::endl;
        }
        if (wc.type == WallType::FUNNEL) {
            std::cout << "  [funnel] r_bot=" << wc.funnel_radius_bottom
                      << " r_top=" << wc.funnel_radius_top
                      << " length=" << wc.funnel_length
                      << " axis=(" << wc.normal.x << "," << wc.normal.y << "," << wc.normal.z << ")"
                      << std::endl;
        }
        if (wc.type == WallType::SPHERE) {
            std::cout << "  [sphere] radius=" << wc.sphere_radius
                      << " mass=" << wc.sphere_mass
                      << " trajectory=" << wc.sphere_trajectory_path << std::endl;
        }
    }

    std::cout << "\n--- Damping ---" << std::endl;
    std::cout << "Enabled: " << (config.damping.enabled ? "Yes" : "No") << std::endl;
    std::cout << "Type: " << (config.damping.type == DampingType::LOCAL ? "Cundall local"
                              : config.damping.type == DampingType::VISCOUS ? "Viscous" : "None") << std::endl;
    std::cout << "Coefficient: " << config.damping.coefficient << std::endl;

    std::cout << "==============================" << std::endl;
}

} // namespace cudadem

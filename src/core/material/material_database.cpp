#include "material_database.h"
#include <cstddef>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cudadem {
namespace {

[[noreturn]] void invalid(const std::string& where, const std::string& why) {
    throw std::runtime_error("Invalid material database at " + where + ": " + why);
}

void require_map(const YAML::Node& n, const std::string& where) {
    if (!n || !n.IsMap()) invalid(where, "must be a mapping");
}
void require_seq(const YAML::Node& n, const std::string& where) {
    if (!n || !n.IsSequence()) invalid(where, "must be a sequence");
}
void reject_unknown(const YAML::Node& n, std::initializer_list<const char*> keys,
                    const std::string& where) {
    require_map(n, where);
    std::unordered_set<std::string> allowed;
    for (const char* key : keys) allowed.emplace(key);
    for (const auto& entry : n) {
        if (!entry.first.IsScalar() || !allowed.count(entry.first.as<std::string>()))
            invalid(where, "unknown key '" + entry.first.as<std::string>() + "'");
    }
}
template <typename T>
T required(const YAML::Node& n, const char* key, const std::string& where) {
    if (!n[key]) invalid(where, "missing required key '" + std::string(key) + "'");
    try { return n[key].as<T>(); }
    catch (const YAML::Exception& e) { invalid(where + "." + key, e.what()); }
}
float finite_number(const YAML::Node& n, const char* key, const std::string& where,
                    bool positive = false, bool nonnegative = false) {
    const float value = required<float>(n, key, where);
    if (!std::isfinite(value)) invalid(where + "." + key, "must be finite");
    if (positive && !(value > 0.0f)) invalid(where + "." + key, "must be > 0");
    if (nonnegative && value < 0.0f) invalid(where + "." + key, "must be >= 0");
    return value;
}
uint32_t material_id(const YAML::Node& n, const char* key, const std::string& where) {
    const int64_t value = required<int64_t>(n, key, where);
    if (value < 0 || value > std::numeric_limits<uint32_t>::max())
        invalid(where + "." + key, "must be an unsigned 32-bit integer");
    return static_cast<uint32_t>(value);
}
void copy_name(char (&out)[MATERIAL_NAME_LENGTH], const std::string& value, const std::string& where) {
    if (value.empty() || value.size() >= MATERIAL_NAME_LENGTH)
        invalid(where, "must be 1..63 bytes");
    std::memset(out, 0, sizeof(out));
    std::memcpy(out, value.data(), value.size());
}

void parse_dem(const YAML::Node& n, MaterialRecord* out, const std::string& where) {
    reject_unknown(n, {"density", "youngs_modulus", "poisson_ratio"}, where);
    out->dem.density = finite_number(n, "density", where, true);
    out->dem.youngs_modulus = finite_number(n, "youngs_modulus", where, true);
    out->dem.poisson_ratio = finite_number(n, "poisson_ratio", where);
    if (!(out->dem.poisson_ratio >= 0.0f && out->dem.poisson_ratio < 0.5f))
        invalid(where + ".poisson_ratio", "must be in [0, 0.5)");
}

void parse_sph_strength(const YAML::Node& n, MaterialRecord* out, const std::string& where) {
    require_map(n, where);
    const std::string model = required<std::string>(n, "model", where);
    if (model == "none") {
        reject_unknown(n, {"model"}, where);
        out->sph_strength.model = SphStrengthModel::NONE;
        return;
    }
    if (model == "von_mises") {
        reject_unknown(n, {"model", "shear_modulus", "yield_strength_Y0"}, where);
        out->sph_strength.model = SphStrengthModel::VON_MISES;
        out->sph_strength.shear_modulus = finite_number(n, "shear_modulus", where, true);
        out->sph_strength.yield_strength_Y0 = finite_number(n, "yield_strength_Y0", where, false, true);
        return;
    }
    if (model == "lundborg") {
        reject_unknown(n, {"model", "shear_modulus", "Yi0", "Yd0", "Ym", "mui", "mud"}, where);
        out->sph_strength.model = SphStrengthModel::LUNDBORG;
        out->sph_strength.shear_modulus = finite_number(n, "shear_modulus", where, true);
        out->sph_strength.Yi0 = finite_number(n, "Yi0", where, false, true);
        out->sph_strength.Yd0 = finite_number(n, "Yd0", where, false, true);
        out->sph_strength.Ym = finite_number(n, "Ym", where, false, true);
        out->sph_strength.mui = finite_number(n, "mui", where, false, true);
        out->sph_strength.mud = finite_number(n, "mud", where, false, true);
        return;
    }
    invalid(where + ".model", "must be none, von_mises, or lundborg");
}
void parse_sph_damage(const YAML::Node& n, MaterialRecord* out, const std::string& where) {
    require_map(n, where);
    const std::string model = required<std::string>(n, "model", where);
    if (model == "none") {
        reject_unknown(n, {"model"}, where);
        out->sph_damage.model = SphDamageModel::NONE;
        return;
    }
    if (model != "grady_kipp") invalid(where + ".model", "must be none or grady_kipp");
    reject_unknown(n, {"model", "k", "m", "crack_speed_fraction", "min_strain"}, where);
    out->sph_damage.model = SphDamageModel::GRADY_KIPP;
    const double k = required<double>(n, "k", where);
    if (!(std::isfinite(k) && k > 0.0)) invalid(where + ".k", "must be finite and > 0");
    // Preserve legacy cards' exact binary32 value and checkpoint identity;
    // extend the range only when binary32 cannot represent a positive finite k.
    const float legacy_k = k <= std::numeric_limits<float>::max() ? static_cast<float>(k) : 0.0f;
    out->sph_damage.k = std::isfinite(legacy_k) && legacy_k > 0.0f ? legacy_k : k;
    out->sph_damage.m = finite_number(n, "m", where, true);
    // Crack growth is a fraction of the evolving particle sound speed.  Keep
    // the pilot value and a numerical strain floor explicit as safe defaults.
    out->sph_damage.crack_speed_fraction = 0.4f;
    out->sph_damage.min_strain = 1.0e-6f;
    if (n["crack_speed_fraction"])
        out->sph_damage.crack_speed_fraction =
            finite_number(n, "crack_speed_fraction", where, true);
    if (out->sph_damage.crack_speed_fraction > 1.0f)
        invalid(where + ".crack_speed_fraction", "must be in (0, 1]");
    if (n["min_strain"])
        out->sph_damage.min_strain = finite_number(n, "min_strain", where, false, true);
}
void parse_sph_porosity(const YAML::Node& n, MaterialRecord* out, const std::string& where) {
    require_map(n, where);
    const std::string model = required<std::string>(n, "model", where);
    if (model == "none") {
        reject_unknown(n, {"model"}, where);
        out->sph_porosity.model = SphPorosityModel::NONE;
        out->sph_porosity.alpha_initial = 1.0f;
        return;
    }
    if (model != "p_alpha") invalid(where + ".model", "must be none or p_alpha");
    reject_unknown(n, {"model", "alpha_initial", "P_elastic_limit", "P_solid_crush", "crushing_exponent"}, where);
    out->sph_porosity.model = SphPorosityModel::P_ALPHA;
    out->sph_porosity.alpha_initial = finite_number(n, "alpha_initial", where, true);
    if (out->sph_porosity.alpha_initial < 1.0f) invalid(where + ".alpha_initial", "must be >= 1");
    out->sph_porosity.P_elastic_limit = finite_number(n, "P_elastic_limit", where, false, true);
    out->sph_porosity.P_solid_crush = finite_number(n, "P_solid_crush", where, true);
    if (out->sph_porosity.P_solid_crush <= out->sph_porosity.P_elastic_limit)
        invalid(where + ".P_solid_crush", "must be > P_elastic_limit");
    out->sph_porosity.crushing_exponent = finite_number(n, "crushing_exponent", where, true);
}
void parse_sph(const YAML::Node& n, MaterialRecord* out, const std::string& where) {
    reject_unknown(n, {"reference_density", "eos", "strength", "damage", "porosity",
                       "bulk_modulus", "elastic_modulus", "continuous_activation"}, where);
    out->sph_tillotson.reference_density = finite_number(n, "reference_density", where, true);
    const auto eos = n["eos"];
    const std::string model = required<std::string>(eos, "model", where + ".eos");
    if (model == "tillotson") {
        reject_unknown(eos, {"model", "a", "b", "A", "B", "alpha", "beta", "E0", "Eiv", "Ecv", "negative_pressure"}, where + ".eos");
        out->sph_tillotson.a = finite_number(eos, "a", where + ".eos");
        out->sph_tillotson.b = finite_number(eos, "b", where + ".eos");
        out->sph_tillotson.A = finite_number(eos, "A", where + ".eos", true);
        out->sph_tillotson.B = finite_number(eos, "B", where + ".eos", true);
        out->sph_tillotson.alpha = finite_number(eos, "alpha", where + ".eos", true);
        out->sph_tillotson.beta = finite_number(eos, "beta", where + ".eos", true);
        out->sph_tillotson.E0 = finite_number(eos, "E0", where + ".eos", true);
        out->sph_tillotson.Eiv = finite_number(eos, "Eiv", where + ".eos", false, true);
        out->sph_tillotson.Ecv = finite_number(eos, "Ecv", where + ".eos", false, true);
        if (out->sph_tillotson.Ecv < out->sph_tillotson.Eiv)
            invalid(where + ".eos.Ecv", "must be >= Eiv");
    } else if (model == "simplified_tillotson" || model == "elastic") {
        if (model == "elastic") reject_unknown(eos, {"model", "A"}, where + ".eos");
        else reject_unknown(eos, {"model", "A", "c"}, where + ".eos");
        out->sph_eos.model = model == "elastic" ? SphEosModel::ELASTIC : SphEosModel::SIMPLIFIED_TILLOTSON;
        out->sph_tillotson.A = finite_number(eos, "A", where + ".eos", true);
        out->sph_eos.simple_c = model == "elastic" ? 0.0f : finite_number(eos, "c", where + ".eos", false, true);
    } else invalid(where + ".eos.model", "must be tillotson, simplified_tillotson or elastic");
    if (n["bulk_modulus"]) out->sph_eos.bulk_modulus = finite_number(n, "bulk_modulus", where, true);
    if (n["elastic_modulus"]) out->sph_eos.elastic_modulus = finite_number(n, "elastic_modulus", where, true);
    if (n["continuous_activation"]) out->sph_eos.continuous_activation = n["continuous_activation"].as<bool>();
    parse_sph_strength(n["strength"], out, where + ".strength");
    parse_sph_damage(n["damage"], out, where + ".damage");
    parse_sph_porosity(n["porosity"], out, where + ".porosity");
    if (eos["negative_pressure"]) {
        const auto policy = eos["negative_pressure"].as<std::string>();
        if (policy == "allow") out->sph_tillotson.negative_pressure = SphNegativePressure::ALLOW;
        else if (policy == "zero") out->sph_tillotson.negative_pressure = SphNegativePressure::ZERO;
        else invalid(where + ".eos.negative_pressure", "must be allow or zero");
    }
    if (out->sph_tillotson.negative_pressure == SphNegativePressure::ZERO &&
        (out->sph_strength.model != SphStrengthModel::NONE ||
         out->sph_damage.model != SphDamageModel::NONE ||
         out->sph_porosity.model != SphPorosityModel::NONE))
        invalid(where + ".eos.negative_pressure", "zero requires strength, damage and porosity models none");
}

DemInteractionProperties parse_interaction(const YAML::Node& n, const std::string& where, DemDampingModel damping_model) {
    if (damping_model == DemDampingModel::RESTITUTION)
        reject_unknown(n, {"materials", "restitution", "friction", "shape_parameter", "cohesion"}, where);
    else
        reject_unknown(n, {"materials", "viscous_coefficient", "friction", "shape_parameter", "cohesion"}, where);
    require_seq(n["materials"], where + ".materials");
    if (n["materials"].size() != 2) invalid(where + ".materials", "must have exactly two material ids");
    DemInteractionProperties result{};
    const int64_t i = n["materials"][0].as<int64_t>();
    const int64_t j = n["materials"][1].as<int64_t>();
    if (i < 0 || j < 0 || i > std::numeric_limits<uint32_t>::max() || j > std::numeric_limits<uint32_t>::max())
        invalid(where + ".materials", "ids must be unsigned 32-bit integers");
    result.material_i = static_cast<uint32_t>(i); result.material_j = static_cast<uint32_t>(j);
    if (result.material_i > result.material_j) invalid(where + ".materials", "must be an unordered pair in ascending id order");
    if (damping_model == DemDampingModel::RESTITUTION) {
        result.restitution = finite_number(n, "restitution", where);
        if (result.restitution < 0.0f || result.restitution > 1.0f)
            invalid(where + ".restitution", "must be in [0, 1]");
    } else result.viscous_coefficient = finite_number(n, "viscous_coefficient", where, false, true);
    const auto friction = n["friction"];
    reject_unknown(friction, {"static", "rolling", "twisting"}, where + ".friction");
    result.friction_static = finite_number(friction, "static", where + ".friction", false, true);
    result.friction_rolling = finite_number(friction, "rolling", where + ".friction", false, true);
    result.friction_twisting = finite_number(friction, "twisting", where + ".friction", false, true);
    result.shape_parameter = finite_number(n, "shape_parameter", where, true);
    const auto cohesion = n["cohesion"];
    const std::string cohesion_model = required<std::string>(cohesion, "model", where + ".cohesion");
    if (cohesion_model == "none") {
        reject_unknown(cohesion, {"model"}, where + ".cohesion"); result.cohesion_enabled = 0;
    } else if (cohesion_model == "scheeres") {
        reject_unknown(cohesion, {"model", "strength", "cutoff_factor"}, where + ".cohesion"); result.cohesion_enabled = 1;
        result.cohesion_strength = finite_number(cohesion, "strength", where + ".cohesion", false, true);
        result.cohesion_cutoff_factor = finite_number(cohesion, "cutoff_factor", where + ".cohesion", true);
        if (result.cohesion_cutoff_factor < 1.0f)
            invalid(where + ".cohesion.cutoff_factor", "must be >= 1");
    } else invalid(where + ".cohesion.model", "must be none or scheeres");
    return result;
}

uint64_t fnv1a(uint64_t hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}
} // namespace

const DemInteractionProperties& MaterialDatabase::interaction(uint32_t i, uint32_t j) const {
    if (i >= info.material_count || j >= info.material_count) throw std::out_of_range("material interaction id");
    return dem_interactions[static_cast<size_t>(i) * info.material_count + j];
}

uint64_t material_database_fingerprint(const MaterialDatabase& database) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = fnv1a(hash, &database.info.schema_version, sizeof(database.info.schema_version));
    // Freeze the historical fingerprint representation rather than hashing the
    // widened in-memory ABI. Old cards/checkpoints retain the same identity.
    struct LegacyDamage { SphDamageModel model; float k, m, crack, minimum; };
    struct LegacyTillotson { float reference_density, a, b, A, B, alpha, beta, E0, Eiv, Ecv; };
    struct LegacyMaterial {
        uint32_t id; char name[MATERIAL_NAME_LENGTH]; uint8_t has_dem, has_sph;
        DemMaterialProperties dem; LegacyTillotson tillotson;
        SphStrengthProperties strength; LegacyDamage damage; SphPorosityProperties porosity;
    };
    static_assert(sizeof(LegacyMaterial) == 196, "legacy material fingerprint ABI");
    bool extended = false;
    for (const auto& material : database.materials) {
        LegacyMaterial legacy;
        std::memset(&legacy, 0, sizeof(legacy));
        legacy.id = material.id;
        std::memcpy(legacy.name, material.name, sizeof(legacy.name));
        legacy.has_dem = material.has_dem; legacy.has_sph = material.has_sph;
        legacy.dem = material.dem;
        const auto& t = material.sph_tillotson;
        legacy.tillotson = {t.reference_density,t.a,t.b,t.A,t.B,t.alpha,t.beta,t.E0,t.Eiv,t.Ecv};
        legacy.strength = material.sph_strength; legacy.porosity = material.sph_porosity;
        const auto& d = material.sph_damage;
        const float old_k = d.k <= std::numeric_limits<float>::max() ? static_cast<float>(d.k) : 0.0f;
        const bool wide = !std::isfinite(old_k) || static_cast<double>(old_k) != d.k;
        legacy.damage = {d.model, wide ? 0.0f : old_k, d.m, d.crack_speed_fraction, d.min_strain};
        extended = extended || wide;
        hash = fnv1a(hash, &legacy, sizeof(legacy));
    }
    // Preserve legacy fingerprints unless the new constitutive options are used.
    for (const auto& material : database.materials) {
        const auto& eos = material.sph_eos;
        if (eos.model != SphEosModel::TILLOTSON || eos.bulk_modulus != 0.0f ||
            eos.elastic_modulus != 0.0f || eos.continuous_activation) {
            const uint64_t tag = UINT64_C(0x535048454f533031);
            hash = fnv1a(hash, &tag, sizeof(tag));
            hash = fnv1a(hash, &material.id, sizeof(material.id));
            hash = fnv1a(hash, &eos.model, sizeof(eos.model));
            hash = fnv1a(hash, &eos.simple_c, sizeof(eos.simple_c));
            hash = fnv1a(hash, &eos.bulk_modulus, sizeof(eos.bulk_modulus));
            hash = fnv1a(hash, &eos.elastic_modulus, sizeof(eos.elastic_modulus));
            hash = fnv1a(hash, &eos.continuous_activation, sizeof(eos.continuous_activation));
        }
    }
    hash = fnv1a(hash, database.dem_interactions.data(), database.dem_interactions.size() * sizeof(DemInteractionProperties));
    if (extended) {
        static constexpr char tag[] = "GRANIT/Weibull-k/f64/v1";
        hash = fnv1a(hash, tag, sizeof(tag));
        for (const auto& material : database.materials) {
            const double k = material.sph_damage.k;
            const float old_k = k <= std::numeric_limits<float>::max() ? static_cast<float>(k) : 0.0f;
            if (!std::isfinite(old_k) || static_cast<double>(old_k) != k) {
                hash = fnv1a(hash, &material.id, sizeof(material.id));
                hash = fnv1a(hash, &k, sizeof(k));
            }
        }
    }
    bool has_zero_tension = false;
    for (const auto& material : database.materials)
        has_zero_tension = has_zero_tension || material.sph_tillotson.negative_pressure == SphNegativePressure::ZERO;
    if (has_zero_tension) {
        static constexpr char tag[] = "GRANIT/Tillotson-negative-pressure/v1";
        hash = fnv1a(hash, tag, sizeof(tag));
        for (const auto& material : database.materials) {
            if (material.sph_tillotson.negative_pressure == SphNegativePressure::ZERO) {
                hash = fnv1a(hash, &material.id, sizeof(material.id));
                hash = fnv1a(hash, &material.sph_tillotson.negative_pressure, sizeof(SphNegativePressure));
            }
        }
    }
    return hash;
}

MaterialDatabase load_material_database(const std::string& database_path, DemDampingModel damping_model) {
    YAML::Node root;
    try { root = YAML::LoadFile(database_path); }
    catch (const YAML::Exception& e) { throw std::runtime_error("Failed to load material database '" + database_path + "': " + e.what()); }
    reject_unknown(root, {"version", "materials", "dem_interactions"}, "root");
    const uint32_t version = required<uint32_t>(root, "version", "root");
    if (version != MATERIAL_DATABASE_SCHEMA_VERSION) invalid("root.version", "only version 1 is supported");
    require_seq(root["materials"], "root.materials");
    if (root["materials"].size() == 0) invalid("root.materials", "must not be empty");
    MaterialDatabase database;
    database.info.schema_version = version;
    database.info.material_count = static_cast<uint32_t>(root["materials"].size());
    if (database.info.material_count > 65535) invalid("root.materials", "supports at most 65535 materials");
    database.materials.resize(database.info.material_count);
    std::set<uint32_t> ids;
    for (size_t index = 0; index < root["materials"].size(); ++index) {
        const auto n = root["materials"][index]; const std::string where = "root.materials[" + std::to_string(index) + "]";
        reject_unknown(n, {"id", "name", "dem", "sph"}, where);
        MaterialRecord material{};
        material.id = material_id(n, "id", where);
        copy_name(material.name, required<std::string>(n, "name", where), where + ".name");
        material.has_dem = n["dem"] ? 1 : 0;
        material.has_sph = n["sph"] ? 1 : 0;
        if (!material.has_dem && !material.has_sph) invalid(where, "must define dem and/or sph");
        if (material.has_dem) parse_dem(n["dem"], &material, where + ".dem");
        if (material.has_sph) parse_sph(n["sph"], &material, where + ".sph");
        if (!ids.insert(material.id).second) invalid(where + ".id", "duplicate id");
        if (material.id >= database.info.material_count) invalid(where + ".id", "ids must be dense in [0,N)");
        database.materials[material.id] = material;
    }
    for (uint32_t id = 0; id < database.info.material_count; ++id)
        if (!ids.count(id)) invalid("root.materials", "ids must be dense in [0,N)");
    require_seq(root["dem_interactions"], "root.dem_interactions");
    uint32_t dem_material_count = 0;
    for (const auto& material : database.materials) dem_material_count += material.has_dem;
    const size_t required_pairs = static_cast<size_t>(dem_material_count) * (dem_material_count + 1) / 2;
    if (root["dem_interactions"].size() != required_pairs)
        invalid("root.dem_interactions", "must contain one explicit entry for every unordered DEM material pair, including diagonals");
    database.dem_interactions.resize(static_cast<size_t>(database.info.material_count) * database.info.material_count);
    std::vector<bool> found(database.dem_interactions.size(), false);
    for (size_t index = 0; index < root["dem_interactions"].size(); ++index) {
        const auto interaction = parse_interaction(root["dem_interactions"][index], "root.dem_interactions[" + std::to_string(index) + "]", damping_model);
        if (interaction.material_i >= database.info.material_count || interaction.material_j >= database.info.material_count)
            invalid("root.dem_interactions[" + std::to_string(index) + "].materials", "references an unknown material id");
        if (!database.materials[interaction.material_i].has_dem || !database.materials[interaction.material_j].has_dem)
            invalid("root.dem_interactions[" + std::to_string(index) + "].materials", "both materials must define dem");
        const size_t offset = static_cast<size_t>(interaction.material_i) * database.info.material_count + interaction.material_j;
        if (found[offset]) invalid("root.dem_interactions[" + std::to_string(index) + "].pair", "duplicate pair");
        found[offset] = true; database.dem_interactions[offset] = interaction;
        const size_t reverse = static_cast<size_t>(interaction.material_j) * database.info.material_count + interaction.material_i;
        database.dem_interactions[reverse] = interaction;
        database.dem_interactions[reverse].material_i = interaction.material_j;
        database.dem_interactions[reverse].material_j = interaction.material_i;
        found[reverse] = true;
    }
    for (uint32_t i = 0; i < database.info.material_count; ++i) for (uint32_t j = 0; j < database.info.material_count; ++j) {
        if (database.materials[i].has_dem && database.materials[j].has_dem && !found[static_cast<size_t>(i) * database.info.material_count + j])
            invalid("root.dem_interactions", "interaction matrix is incomplete");
    }
    for (uint32_t i = 0; i < database.info.material_count; ++i) for (uint32_t j = 0; j < database.info.material_count; ++j) {
        const auto& a = database.interaction(i, j); const auto& b = database.interaction(j, i);
        if (a.restitution != b.restitution || a.viscous_coefficient != b.viscous_coefficient ||
            a.friction_static != b.friction_static || a.friction_rolling != b.friction_rolling ||
            a.friction_twisting != b.friction_twisting || a.shape_parameter != b.shape_parameter || a.cohesion_enabled != b.cohesion_enabled ||
            a.cohesion_cutoff_factor != b.cohesion_cutoff_factor || a.cohesion_strength != b.cohesion_strength)
            invalid("root.dem_interactions", "matrix entries must be symmetric");
    }
    std::memset(database.info.path, 0, sizeof(database.info.path));
    const auto absolute = std::filesystem::absolute(database_path).lexically_normal().string();
    if (absolute.size() >= sizeof(database.info.path)) invalid("root", "database path exceeds 255 bytes");
    std::memcpy(database.info.path, absolute.c_str(), absolute.size());
    database.info.fingerprint = material_database_fingerprint(database);
    return database;
}

DeviceMaterialDatabase upload_material_database(const MaterialDatabase& database, int device_id) {
    if (database.materials.empty()) throw std::runtime_error("cannot upload an empty material database");
    int previous_device = 0; cudaError_t error = cudaGetDevice(&previous_device);
    if (error != cudaSuccess) throw std::runtime_error("cudaGetDevice failed while uploading material database");
    error = cudaSetDevice(device_id);
    if (error != cudaSuccess) throw std::runtime_error("cudaSetDevice failed while uploading material database");
    DeviceMaterialDatabase result{}; result.device_id = device_id; result.material_count = database.info.material_count; result.fingerprint = database.info.fingerprint;
    error = cudaMalloc(&result.materials, database.materials.size() * sizeof(MaterialRecord));
    if (error == cudaSuccess && !database.dem_interactions.empty())
        error = cudaMalloc(&result.dem_interactions, database.dem_interactions.size() * sizeof(DemInteractionProperties));
    if (error == cudaSuccess) error = cudaMemcpy(result.materials, database.materials.data(), database.materials.size() * sizeof(MaterialRecord), cudaMemcpyHostToDevice);
    if (error == cudaSuccess && !database.dem_interactions.empty())
        error = cudaMemcpy(result.dem_interactions, database.dem_interactions.data(), database.dem_interactions.size() * sizeof(DemInteractionProperties), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { if (result.materials) cudaFree(result.materials); if (result.dem_interactions) cudaFree(result.dem_interactions); cudaSetDevice(previous_device); throw std::runtime_error("failed to upload material database: " + std::string(cudaGetErrorString(error))); }
    cudaSetDevice(previous_device); return result;
}

void release_material_database(DeviceMaterialDatabase* database) {
    if (!database) return;
    int previous_device = 0; if (cudaGetDevice(&previous_device) != cudaSuccess) return;
    if (database->device_id >= 0) cudaSetDevice(database->device_id);
    if (database->materials) cudaFree(database->materials);
    if (database->dem_interactions) cudaFree(database->dem_interactions);
    cudaSetDevice(previous_device); *database = DeviceMaterialDatabase{};
}

} // namespace cudadem

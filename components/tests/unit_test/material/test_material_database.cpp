// Focused validation of the strict v1 material database contract.
#include "src/core/material/material_database.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace cudadem;

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s (line %d)\\n", message, __LINE__); ++failures; } \
} while (false)

std::filesystem::path fixture_path(const char* label) {
    return std::filesystem::temp_directory_path() /
           (std::string("granit_material_") + label + "_" + std::to_string(::getpid()) + ".yaml");
}

void write_fixture(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
    out.close();
}

bool rejects(const std::filesystem::path& path, const std::string& yaml,
             DemDampingModel model) {
    write_fixture(path, yaml);
    try {
        (void)load_material_database(path.string(), model);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

float expected_e_star(float ea, float nua, float eb, float nub) {
    return 1.0f / ((1.0f - nua * nua) / ea + (1.0f - nub * nub) / eb);
}
float expected_g_star(float ea, float nua, float eb, float nub) {
    const float ga = ea / (2.0f * (1.0f + nua));
    const float gb = eb / (2.0f * (1.0f + nub));
    return 1.0f / ((2.0f - nua) / ga + (2.0f - nub) / gb);
}

const char* valid_restitution = R"YAML(
version: 1
materials:
  - id: 0
    name: A
    dem: {density: 1000.0, youngs_modulus: 1000000.0, poisson_ratio: 0.20}
  - id: 1
    name: B
    dem: {density: 2000.0, youngs_modulus: 4000000.0, poisson_ratio: 0.30}
dem_interactions:
  - materials: [0, 0]
    restitution: 0.10
    friction: {static: 0.11, rolling: 0.12, twisting: 0.13}
    shape_parameter: 0.14
    cohesion: {model: none}
  - materials: [0, 1]
    restitution: 0.20
    friction: {static: 0.21, rolling: 0.22, twisting: 0.23}
    shape_parameter: 0.24
    cohesion: {model: scheeres, strength: 7.0, cutoff_factor: 1.25}
  - materials: [1, 1]
    restitution: 0.30
    friction: {static: 0.31, rolling: 0.32, twisting: 0.33}
    shape_parameter: 0.34
    cohesion: {model: none}
)YAML";
}  // namespace

int main() {
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path();
    const auto legacy = load_material_database((root / "components/config/materials_example.yaml").string(),
                                               DemDampingModel::RESTITUTION);
    CHECK(legacy.info.fingerprint == UINT64_C(7561619062791092291),
          "widened Weibull storage must preserve legacy checkpoint fingerprints");
    const auto wide_path = fixture_path("wide_weibull");
    std::ifstream card_stream(root / "components/config/materials_example.yaml");
    std::string card((std::istreambuf_iterator<char>(card_stream)), std::istreambuf_iterator<char>());
    const auto k_start = card.find("k: ");
    CHECK(k_start != std::string::npos, "example must include a damage coefficient");
    if (k_start != std::string::npos) {
        card.replace(k_start, card.find('\n', k_start)-k_start, "k: 1.0e61");
        write_fixture(wide_path, card);
        const auto wide = load_material_database(wide_path.string(), DemDampingModel::RESTITUTION);
        bool retained = false;
        for (const auto& material : wide.materials) retained = retained || material.sph_damage.k == 1.0e61;
        CHECK(retained, "published Weibull coefficient must survive material parsing");
        std::filesystem::remove(wide_path);
    }
    auto extended = legacy;
    extended.materials[0].sph_damage.k = 1.0e61;
    const auto wide_fingerprint = material_database_fingerprint(extended);
    extended.materials[0].sph_damage.k = 2.0e61;
    CHECK(wide_fingerprint != legacy.info.fingerprint &&
          wide_fingerprint != material_database_fingerprint(extended),
          "wide coefficients must have distinct material fingerprints");
    auto liquid=legacy;
    liquid.materials[0].sph_tillotson.negative_pressure=SphNegativePressure::ZERO;
    CHECK(material_database_fingerprint(liquid)!=legacy.info.fingerprint,
          "zero-tension policy must change checkpoint material identity");
    const auto fluid_path=fixture_path("fluid_policy");
    const std::string fluid_card=R"YAML(
version: 1
materials:
- id: 0
  name: fluid
  sph:
    reference_density: 7019
    eos: {model: tillotson, a: 0.5, b: 1.5, A: 1.097e11, B: 1.05e11, alpha: 5, beta: 5, E0: 9.5e6, Eiv: 1.42e6, Ecv: 8.45e6, negative_pressure: zero}
    strength: {model: none}
    damage: {model: none}
    porosity: {model: none}
dem_interactions: []
)YAML";
    write_fixture(fluid_path,fluid_card);
    CHECK(load_material_database(fluid_path.string(),DemDampingModel::RESTITUTION).materials[0].sph_tillotson.negative_pressure==SphNegativePressure::ZERO,
          "strict material parser must accept explicit zero-tension fluid");
    std::string invalid_fluid=fluid_card;
    invalid_fluid.replace(invalid_fluid.find("strength: {model: none}"),23,
                         "strength: {model: von_mises, shear_modulus: 1e9, yield_strength_Y0: 1e6}");
    CHECK(rejects(fluid_path,invalid_fluid,DemDampingModel::RESTITUTION),
          "zero-tension liquid option must not silently alter a strength-bearing solid");
    std::filesystem::remove(fluid_path);
    const auto valid_path = fixture_path("valid");
    write_fixture(valid_path, valid_restitution);
    try {
        const MaterialDatabase db = load_material_database(valid_path.string(), DemDampingModel::RESTITUTION);
        CHECK(db.info.material_count == 2, "material IDs define dense two-entry table");
        CHECK(db.dem_interactions.size() == 4, "database stores full dense interaction table");
        const auto& aa = db.interaction(0, 0);
        const auto& ab = db.interaction(0, 1);
        const auto& ba = db.interaction(1, 0);
        const auto& bb = db.interaction(1, 1);
        CHECK(std::fabs(aa.restitution - 0.10f) < 1e-6f, "AA restitution retained");
        CHECK(std::fabs(ab.restitution - 0.20f) < 1e-6f && std::fabs(ba.restitution - 0.20f) < 1e-6f,
              "AB pair is symmetric");
        CHECK(std::fabs(bb.friction_static - 0.31f) < 1e-6f, "BB static friction retained");
        CHECK(std::fabs(ab.friction_rolling - 0.22f) < 1e-6f &&
              std::fabs(ab.friction_twisting - 0.23f) < 1e-6f,
              "pair-specific rolling and twisting friction retained");
        CHECK(ab.cohesion_enabled && std::fabs(ab.cohesion_cutoff_factor - 1.25f) < 1e-6f,
              "pair-specific Scheeres range retained");

        const auto& a = db.materials[0].dem;
        const auto& b = db.materials[1].dem;
        const float aa_e = expected_e_star(a.youngs_modulus, a.poisson_ratio, a.youngs_modulus, a.poisson_ratio);
        const float ab_e = expected_e_star(a.youngs_modulus, a.poisson_ratio, b.youngs_modulus, b.poisson_ratio);
        const float bb_e = expected_e_star(b.youngs_modulus, b.poisson_ratio, b.youngs_modulus, b.poisson_ratio);
        const float aa_g = expected_g_star(a.youngs_modulus, a.poisson_ratio, a.youngs_modulus, a.poisson_ratio);
        const float ab_g = expected_g_star(a.youngs_modulus, a.poisson_ratio, b.youngs_modulus, b.poisson_ratio);
        const float bb_g = expected_g_star(b.youngs_modulus, b.poisson_ratio, b.youngs_modulus, b.poisson_ratio);
        CHECK(aa_e > 0.0f && ab_e > 0.0f && bb_e > 0.0f && aa_g > 0.0f && ab_g > 0.0f && bb_g > 0.0f,
              "AA/AB/BB Hertz-Mindlin effective moduli are finite and positive");
        CHECK(std::fabs(expected_e_star(a.youngs_modulus, a.poisson_ratio, b.youngs_modulus, b.poisson_ratio) -
                        expected_e_star(b.youngs_modulus, b.poisson_ratio, a.youngs_modulus, a.poisson_ratio)) < 1e-6f,
              "effective Young modulus is symmetric");
        CHECK(std::fabs(expected_g_star(a.youngs_modulus, a.poisson_ratio, b.youngs_modulus, b.poisson_ratio) -
                        expected_g_star(b.youngs_modulus, b.poisson_ratio, a.youngs_modulus, a.poisson_ratio)) < 1e-6f,
              "effective shear modulus is symmetric");
        CHECK(db.info.fingerprint == material_database_fingerprint(db), "database fingerprint is deterministic");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected valid database failure: %s\\n", e.what());
        ++failures;
    }

    const auto invalid_path = fixture_path("invalid");
    std::string missing_pair = valid_restitution;
    const auto last = missing_pair.rfind("  - materials: [1, 1]");
    missing_pair.erase(last);
    CHECK(rejects(invalid_path, missing_pair, DemDampingModel::RESTITUTION),
          "all triangular DEM pairs including diagonals are required");
    std::string sparse_id = valid_restitution;
    const auto id_pos = sparse_id.find("id: 1");
    sparse_id.replace(id_pos, 5, "id: 2");
    CHECK(rejects(invalid_path, sparse_id, DemDampingModel::RESTITUTION), "material IDs must be dense");
    std::string wrong_damping_key = valid_restitution;
    const auto rest_pos = wrong_damping_key.find("restitution: 0.10");
    wrong_damping_key.replace(rest_pos, std::strlen("restitution: 0.10"), "viscous_coefficient: 0.10");
    CHECK(rejects(invalid_path, wrong_damping_key, DemDampingModel::RESTITUTION),
          "restitution mode rejects viscous-only interaction key");
    std::string missing_viscous = valid_restitution;
    CHECK(rejects(invalid_path, missing_viscous, DemDampingModel::VISCOUS),
          "viscous mode requires viscous coefficient for every pair");
    std::string zero_shape = valid_restitution;
    const auto shape_pos = zero_shape.find("shape_parameter: 0.14");
    zero_shape.replace(shape_pos, std::strlen("shape_parameter: 0.14"), "shape_parameter: 0.0");
    CHECK(rejects(invalid_path, zero_shape, DemDampingModel::RESTITUTION),
          "shape parameter must be strictly positive");
    std::string invalid_cutoff = valid_restitution;
    const auto cutoff_pos = invalid_cutoff.find("cutoff_factor: 1.25");
    invalid_cutoff.replace(cutoff_pos, std::strlen("cutoff_factor: 1.25"), "cutoff_factor: 0.99");
    CHECK(rejects(invalid_path, invalid_cutoff, DemDampingModel::RESTITUTION),
          "Scheeres cutoff factor must be at least one");
    std::string unused_cohesion_key = valid_restitution;
    const auto none_pos = unused_cohesion_key.find("cohesion: {model: none}");
    unused_cohesion_key.replace(none_pos, std::strlen("cohesion: {model: none}"),
                                "cohesion: {model: none, strength: 1.0}");
    CHECK(rejects(invalid_path, unused_cohesion_key, DemDampingModel::RESTITUTION),
          "inactive cohesion model rejects unused model-specific keys");
    std::string unknown = valid_restitution;
    unknown.replace(0, 0, "obsolete: true\\n");
    CHECK(rejects(invalid_path, unknown, DemDampingModel::RESTITUTION), "unknown schema keys are rejected");

    std::filesystem::remove(valid_path);
    std::filesystem::remove(invalid_path);
    if (failures == 0) {
        std::puts("ALL PASS");
        return 0;
    }
    std::fprintf(stderr, "%d material database checks failed\\n", failures);
    return 1;
}

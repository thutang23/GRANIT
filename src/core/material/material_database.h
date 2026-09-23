#ifndef CUDADEM_MATERIAL_DATABASE_H
#define CUDADEM_MATERIAL_DATABASE_H

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cudadem {

constexpr uint32_t MATERIAL_DATABASE_SCHEMA_VERSION = 1;
constexpr int MATERIAL_NAME_LENGTH = 64;

enum class SphNegativePressure : uint8_t { ALLOW = 0, ZERO = 1 };

enum class SphEosModel : uint8_t { TILLOTSON = 0, SIMPLIFIED_TILLOTSON = 1, ELASTIC = 2 };

enum class SphStrengthModel : uint8_t { NONE = 0, VON_MISES = 1, LUNDBORG = 2 };
enum class SphDamageModel : uint8_t { NONE = 0, GRADY_KIPP = 1 };
enum class SphPorosityModel : uint8_t { NONE = 0, P_ALPHA = 1 };
enum class DemDampingModel : uint8_t { RESTITUTION = 0, VISCOUS = 1 };

// POD records deliberately have no host containers: copies of these tables are
// passed directly to every GPU and may be safely consumed by device kernels.
struct DemMaterialProperties {
    float density;
    float youngs_modulus;
    float poisson_ratio;
};

struct SphTillotsonProperties {
    float reference_density;
    float a, b, A, B, alpha, beta, E0, Eiv, Ecv;
    SphNegativePressure negative_pressure;
};

struct SphStrengthProperties {
    SphStrengthModel model;
    float shear_modulus;
    float yield_strength_Y0;
    float Yi0, Yd0, Ym, mui, mud;
};

struct SphDamageProperties {
    SphDamageModel model;
    double k;  // SI Weibull coefficients may exceed binary32 range.
    float m, crack_speed_fraction, min_strain;
};

struct SphPorosityProperties {
    SphPorosityModel model;
    float alpha_initial, P_elastic_limit, P_solid_crush, crushing_exponent;
};

struct SphEosProperties {
    SphEosModel model;
    float simple_c;
    float bulk_modulus;          // Zero uses the Tillotson A parameter
    float elastic_modulus;       // Zero derives E from K and G
    bool continuous_activation;  // Optional continuous Grady-Kipp flaw estimate
};

struct MaterialRecord {
    uint32_t id;
    char name[MATERIAL_NAME_LENGTH];
    uint8_t has_dem;
    uint8_t has_sph;
    DemMaterialProperties dem;
    SphTillotsonProperties sph_tillotson;
    SphStrengthProperties sph_strength;
    SphDamageProperties sph_damage;
    SphPorosityProperties sph_porosity;
    SphEosProperties sph_eos;
};

struct DemInteractionProperties {
    uint32_t material_i;
    uint32_t material_j;
    float restitution;
    float viscous_coefficient;
    float friction_static;
    float friction_rolling;
    float friction_twisting;
    float shape_parameter;
    uint8_t cohesion_enabled; // Scheeres iff true
    float cohesion_cutoff_factor;
    float cohesion_strength;
};

struct MaterialDatabaseInfo {
    uint32_t schema_version = MATERIAL_DATABASE_SCHEMA_VERSION;
    uint32_t material_count = 0;
    uint64_t fingerprint = 0;
    char path[256] = {};
};

struct MaterialDatabase {
    MaterialDatabaseInfo info;
    std::vector<MaterialRecord> materials;
    // Dense row-major table: [i * material_count + j].
    std::vector<DemInteractionProperties> dem_interactions;
    const DemInteractionProperties& interaction(uint32_t i, uint32_t j) const;
};

struct DeviceMaterialDatabase {
    MaterialRecord* materials = nullptr;
    DemInteractionProperties* dem_interactions = nullptr;
    uint32_t material_count = 0;
    uint64_t fingerprint = 0;
    int device_id = -1;
};

// Loads and validates the versioned YAML database. All ids must form the dense
// range [0,N), and the interaction matrix must be explicit and symmetric.
MaterialDatabase load_material_database(const std::string& database_path,
                                        DemDampingModel damping_model);
uint64_t material_database_fingerprint(const MaterialDatabase& database);

// Each device owns a complete, independent copy. The caller must select no
// particular CUDA device beforehand; upload selects device_id temporarily.
DeviceMaterialDatabase upload_material_database(const MaterialDatabase& database, int device_id);
void release_material_database(DeviceMaterialDatabase* database);

} // namespace cudadem

#endif

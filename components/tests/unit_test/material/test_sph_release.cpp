#include <cstdio>
#include <cmath>
using std::isfinite;
#include "src/core/material/material_database.h"
#include "src/core/sph/sph_eos_math.cuh"
#include "src/core/sph/sph_damage_math.cuh"
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto db=cudadem::load_material_database(argv[1],cudadem::DemDampingModel::RESTITUTION);
    auto liquid=db.materials[0];
    liquid.sph_strength.model=cudadem::SphStrengthModel::NONE;
    liquid.sph_damage.model=cudadem::SphDamageModel::NONE;
    liquid.sph_porosity.model=cudadem::SphPorosityModel::NONE;
    auto allowed=liquid;
    liquid.sph_tillotson.negative_pressure=cudadem::SphNegativePressure::ZERO;
    int release=0,compressed=0;
    for (bool elastic : {false,true}) for (double ratio : {.5,.79,.81,.99,1.,1.2})
    for (double e : {0.,1e3,1e7,2e7}) {
        const double rho0=liquid.sph_tillotson.reference_density,rho=ratio*rho0;
        auto a=cudadem::sph_eos_state(allowed,rho,e,0,1,elastic);
        auto b=cudadem::sph_eos_state(liquid,rho,e,0,1,elastic);
        auto d=cudadem::sph_solid_eos_derivatives(liquid,rho,rho0,e,elastic);
        if (!(isfinite(b.pressure)&&isfinite(b.c_sound)&&b.pressure>=0)) return 3;
        if (a.pressure<0 || (!elastic && ratio<.8 && e<liquid.sph_tillotson.Eiv)) {
            if (b.pressure!=0 || b.c_sound!=1 || d.pressure!=0 || d.dp_drho!=0 || d.dp_de!=0) return 4;
            ++release;
        } else {
            if (a.pressure!=b.pressure || a.c_sound!=b.c_sound) return 5;
            ++compressed;
        }
    }
    // Independent long-double order-statistic expectation, with k*V beyond double.
    auto f=cudadem::sph_damage_flaw_interval(1e300,9.5f,1e40,64,1,64,1e-6f);
    const long double expected=std::pow(1.0L/(1e300L*1e40L),1.0L/9.5L);
    if (!(f.strain_min>0 && std::abs(f.strain_min/expected-1)<2e-5L && f.strain_max>=f.strain_min)) return 6;
    std::printf("PASS liquid release %d, preserved branches %d, wide Weibull overflow case\n",release,compressed);
}

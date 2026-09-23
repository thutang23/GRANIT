// Deterministic host probe: compile against the export and integration headers.
// Exact output equality verifies unchanged default EOS branches/fingerprints.
#include <cstdio>
#include <cmath>
using std::isfinite;
#include "src/core/material/material_database.h"
#include "src/core/sph/sph_eos_math.cuh"
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto db = cudadem::load_material_database(argv[1], cudadem::DemDampingModel::RESTITUTION);
    for (int model=0; model<3; ++model) for (int extended=0; extended<2; ++extended) {
        auto& m = db.materials[0];
        m.sph_eos.model=static_cast<cudadem::SphEosModel>(model);
        m.sph_eos.simple_c=2;
        m.sph_eos.bulk_modulus=extended ? 2.7e10f : 0;
        m.sph_eos.elastic_modulus=extended ? 5.4e10f : 0;
        m.sph_eos.continuous_activation=extended;
        std::printf("fingerprint %llu\n", (unsigned long long)cudadem::material_database_fingerprint(db));
        for (double rho_ratio : {.5,.79,.8,.99,1.,1.1,2.})
        for (double energy : {0.,1e3,4.72e6,1e7,2e7})
        for (float damage : {0.f,.5f,1.f}) for (bool elastic : {false,true}) {
            const double rho=rho_ratio*m.sph_tillotson.reference_density;
            auto p=cudadem::sph_eos_state(m,rho,energy,damage,1,elastic);
            auto d=cudadem::sph_solid_eos_derivatives(m,rho,m.sph_tillotson.reference_density,energy,elastic);
            std::printf("%a %a %a %a %a\n",double(p.pressure),double(p.c_sound),d.pressure,d.dp_drho,d.dp_de);
        }
    }
}

// =============================================================================
// SPH Configuration Printer Implementation
// =============================================================================

#include "sph_config_printer.h"

namespace cudadem {

void print_sph_config(const SimConfig& config) {
    std::cout << "\n=== SPH Configuration ===" << std::endl;
    std::cout << "Material database: " << config.material_database.path
              << " (" << config.material_database.material_count << " materials)" << std::endl;
    std::cout << "Input (unified v3): " << config.io.input_particle_path << std::endl;
    std::cout << "Viscosity:          " << config.sph.viscosity << std::endl;
    std::cout << "Smoothing length:   " << config.sph.smoothing_length << " m" << std::endl;
    std::cout << "Target neighbors:   " << config.sph.num_neighbors << std::endl;
    std::cout << "Kernel correction:  " << (config.sph.use_kernel_correction ? "enabled" : "disabled") << std::endl;
    std::cout << "Artificial stress:  " << (config.sph.use_artificial_stress ? "enabled" : "disabled") << std::endl;
    std::cout << "Relax strength:     " << (config.sph.relaxation_disable_strength ? "disabled" : "active") << std::endl;
    std::cout << "Relax damage:       " << (config.sph.relaxation_disable_damage ? "disabled" : "active") << std::endl;
    std::cout << "Damage flaw seed:   " << config.sph.damage_seed << std::endl;
    std::cout << "CFL range:          [" << config.sph.dt_min << ", " << config.sph.dt_max << "] s" << std::endl;
    std::cout << "Kernel cutoff:      " << config.sph.kernel_cutoff_factor << " h" << std::endl;
    if (config.acceleration.enable_self_gravity &&
        config.acceleration.gravity_method == SelfGravityMethod::DIRECT) {
        std::cout << "Gravity softening:  "
                  << (config.acceleration.direct_gravity.sph_softening_mode ==
                              DirectSphSofteningMode::QUERY_PARTICLE
                          ? "query-particle h_i (SPHSOL comparison)"
                          : "symmetric mean h_ij") << std::endl;
    }
    std::cout << "=============================" << std::endl;
}

} // namespace cudadem

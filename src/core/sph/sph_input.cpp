// =============================================================================
// SPH v3 Unified-Binary Input
// =============================================================================

#include "src/core/sph/sph_input.h"

#include "src/core/constant/binary_reader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cudadem {

int32_t load_sph_particles(ParticleSoA& particles, const SimConfig& config) {
    if (config.physics != SimulationPhysics::SPH) {
        throw std::runtime_error("load_sph_particles requires simulation.physics: sph");
    }
    const std::string path = config.io.input_particle_path;
    if (path.empty()) {
        throw std::runtime_error("io.input.particle_file is required for SPH");
    }

    std::ifstream input(path, std::ios::binary);
    UnifiedHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.record_kind != static_cast<uint8_t>(BinaryRecordKind::INPUT)) {
        throw std::runtime_error(
            "fresh SPH particle input must be a unified v3 INPUT record: " + path);
    }

    // The binary reader owns strict v3 validation, including the required
    // SPH fields and immutable dense material ids.  A temporary state is
    // sufficient here because SPH input records do not carry contact history.
    SimulationState input_state;
    if (!read_unified_binary(path, particles, input_state, config, 0)) {
        throw std::runtime_error("failed to load v3 SPH input: " + path);
    }
    std::cout << "Loaded " << particles.num_particles
              << " SPH particles from v3 unified binary: " << path << std::endl;
    return particles.num_particles;
}

} // namespace cudadem

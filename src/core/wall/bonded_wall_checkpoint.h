#ifndef GRANIT_BONDED_WALL_CHECKPOINT_H
#define GRANIT_BONDED_WALL_CHECKPOINT_H

#include "bonded_rect_wall.h"
#include "../constant/system_initialize.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudadem {

struct BondedWallCheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t count;
    int64_t step;
    double time;
    uint32_t state_size;
    uint32_t reaction_size;
};

// The enclosing checkpoint transaction binds this sidecar to its binary and
// checksums it. No device pointers are part of either saved wall structure.
inline void save_bonded_wall_checkpoint(const std::string& path,
                                        const SimulationState& state) {
    const int32_t count = bonded_rect_wall_get_states(nullptr, nullptr);
    std::vector<BondedRectWallState> states(count);
    std::vector<BondedRectReaction> reactions(count);
    bonded_rect_wall_get_states(states.data(), reactions.data());
    const BondedWallCheckpointHeader header{
        {'G','R','B','W','A','L','L','1'}, 1, static_cast<uint32_t>(count),
        state.current_step, state.current_time,
        sizeof(BondedRectWallState), sizeof(BondedRectReaction)};
    std::ofstream out(path + ".bonded_walls", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (count) {
        out.write(reinterpret_cast<const char*>(states.data()), states.size() * sizeof(states[0]));
        out.write(reinterpret_cast<const char*>(reactions.data()), reactions.size() * sizeof(reactions[0]));
    }
    out.close();
    if (!out) throw std::runtime_error("Cannot write bonded wall checkpoint: " + path);
}

inline void restore_bonded_wall_checkpoint(const std::string& path,
                                           const SimulationState& state) {
    const std::string sidecar = path + ".bonded_walls";
    if (!std::filesystem::exists(sidecar)) {
        if (bonded_rect_wall_get_states(nullptr, nullptr) != 0)
            throw std::runtime_error("Bonded wall restart requires checkpoint sidecar: " + sidecar);
        return; // Legacy checkpoints without dynamic bonded walls.
    }
    BondedWallCheckpointHeader header{};
    std::ifstream in(sidecar, std::ios::binary);
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "GRBWALL1", 8) != 0 || header.version != 1 ||
        header.count > MAX_BONDED_RECT_WALLS || header.step != state.current_step ||
        header.time != state.current_time || header.state_size != sizeof(BondedRectWallState) ||
        header.reaction_size != sizeof(BondedRectReaction))
        throw std::runtime_error("Invalid or mismatched bonded wall checkpoint: " + sidecar);
    std::vector<BondedRectWallState> states(header.count);
    std::vector<BondedRectReaction> reactions(header.count);
    if (header.count) {
        in.read(reinterpret_cast<char*>(states.data()), states.size() * sizeof(states[0]));
        in.read(reinterpret_cast<char*>(reactions.data()), reactions.size() * sizeof(reactions[0]));
    }
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Truncated or oversized bonded wall checkpoint: " + sidecar);
    bonded_rect_wall_restore_states(states.data(), reactions.data(), header.count);
}

} // namespace cudadem
#endif

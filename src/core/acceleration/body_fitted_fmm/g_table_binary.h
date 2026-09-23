#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

// Portable, little-endian, unrounded FP32 table. GPU layout conversion is kept
// separate so all M2L precision modes use the same versioned coefficient asset.
namespace granit_g_table {
inline constexpr std::array<uint32_t, 8> metadata = {
    1, 28, 32, 48, 21, 29, 25, 75 * 48 * 28 * 32
};
inline constexpr size_t header_bytes = 48;
inline constexpr int counts[3] = {21, 29, 25};
inline constexpr int gaps[3] = {0, 1, -1};
inline uint64_t checksum(const unsigned char* bytes, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
inline uint64_t little_endian(const unsigned char* p, int n) {
    uint64_t value = 0;
    for (int i = 0; i < n; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}
inline std::vector<float> load(const std::filesystem::path& path) {
    const auto expected_bytes = header_bytes + size_t(metadata.back()) * 4;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(expected_bytes))
        throw std::runtime_error("Missing or wrong-sized GRANIT G table: " + path.string());
    std::vector<unsigned char> bytes(expected_bytes);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!in || std::memcmp(bytes.data(), "GRANITG1", 8) != 0)
        throw std::runtime_error("Invalid GRANIT G-table header: " + path.string());
    for (size_t i = 0; i < metadata.size(); ++i)
        if (little_endian(bytes.data() + 8 + i * 4, 4) != metadata[i])
            throw std::runtime_error("Incompatible GRANIT G-table layout: " + path.string());
    if (checksum(bytes.data() + header_bytes, bytes.size() - header_bytes) !=
        little_endian(bytes.data() + 40, 8))
        throw std::runtime_error("Corrupt GRANIT G-table payload: " + path.string());
    std::vector<float> values(metadata.back());
    for (size_t i = 0; i < values.size(); ++i) {
        const auto bits = static_cast<uint32_t>(little_endian(bytes.data() + header_bytes + i * 4, 4));
        std::memcpy(&values[i], &bits, 4);
        if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite GRANIT G-table coefficient");
    }
    return values;
}
inline std::vector<float> segment(const std::vector<float>& values, int gap_index, bool tf32) {
    if (gap_index < 0 || gap_index >= 3 || values.size() != metadata.back())
        throw std::runtime_error("Invalid GRANIT G-table segment");
    const int rows = tf32 ? 32 : 28;
    size_t offset = 0;
    for (int i = 0; i < gap_index; ++i) offset += size_t(counts[i]) * 48 * 28 * 32;
    std::vector<float> result(size_t(counts[gap_index]) * 48 * rows * 32, 0.0f);
    for (int matrix = 0; matrix < counts[gap_index] * 48; ++matrix)
        for (int row = 0; row < 28; ++row)
            for (int col = 0; col < 28; ++col) {
                float value = values[offset + size_t(matrix) * 28 * 32 + row * 32 + col];
                if (tf32) {
                    uint32_t bits;
                    std::memcpy(&bits, &value, 4);
                    bits += 0x00000fffu + ((bits >> 13) & 1u);
                    bits &= 0xffffe000u;
                    std::memcpy(&value, &bits, 4);
                }
                result[size_t(matrix) * rows * 32 + row * 32 + col] = value;
            }
    return result;
}
// Both build and installed trees use bin/../share/granit/fmm. Resolve from the
// executable, never the working directory, so YAML may live anywhere.
inline std::filesystem::path installed_path() {
    return std::filesystem::read_symlink("/proc/self/exe").parent_path()
        / "../share/granit/fmm/g_table_v1.bin";
}
} // namespace granit_g_table

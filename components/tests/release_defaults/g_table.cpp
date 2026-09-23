#include "src/core/acceleration/body_fitted_fmm/g_table_binary.h"
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;
int main() {
    const auto path = granit_g_table::installed_path();
    const auto values = granit_g_table::load(path);
    size_t offset = 0;
    for (int gi = 0; gi < 3; ++gi) {
        const auto reference = granit_g_table::segment(values, gi, false);
        if (std::memcmp(reference.data(), values.data() + offset, reference.size() * 4)) return 1;
        const auto tensor = granit_g_table::segment(values, gi, true);
        for (size_t i = 0; i < tensor.size(); ++i) {
            const size_t matrix = i / 1024, row = (i % 1024) / 32, col = i % 32;
            if (row >= 28 || col >= 28) {
                if (tensor[i] != 0) return 2;
                continue;
            }
            uint32_t before, after;
            std::memcpy(&before, &reference[matrix * 896 + row * 32 + col], 4);
            std::memcpy(&after, &tensor[i], 4);
            const uint32_t remainder = before & 0x1fff;
            uint32_t expected = before & ~uint32_t(0x1fff);
            if (remainder > 4096 || (remainder == 4096 && ((before >> 13) & 1))) expected += 8192;
            if (after != expected) return 3;
        }
        offset += reference.size();
    }
    auto temporary = fs::temp_directory_path() / ("granit-table-" + std::to_string(getpid()));
    fs::create_directories(temporary);
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{temporary};
    const auto corrupt = temporary / "corrupt.bin";
    auto rejected = [&](const fs::path& input) {
        try { granit_g_table::load(input); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    if (!rejected(temporary / "missing.bin")) return 4;
    for (const size_t position : {size_t(0), size_t(8), size_t(40), size_t(80)}) {
        fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
        std::fstream out(corrupt, std::ios::in | std::ios::out | std::ios::binary);
        out.seekg(position); char value; out.get(value); out.seekp(position); out.put(value ^ 1); out.close();
        if (!rejected(corrupt)) return 5;
    }
    fs::resize_file(corrupt, 80);
    if (!rejected(corrupt)) return 6;
    std::cout << values.size() << " coefficients: reference bits, TF32 rounding/padding, asset lookup and corruption checks passed\n";
}

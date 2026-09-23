// Reproduce the shipped G table from the canonical headers. No CUDA/GPU required.
// Build: c++ -O2 -ffp-contract=off -std=c++17 -I. tools/generate_g_table.cpp -o /tmp/granit-g-table
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#define __device__
#include "src/core/acceleration/body_fitted_fmm/g_tables.h"
#include "src/core/acceleration/body_fitted_fmm/planar_gap1_operator_table.h"
#include "src/core/acceleration/body_fitted_fmm/g_table_binary.h"

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: generate_g_table OUTPUT.bin\n"; return 2; }
    using namespace body_fitted_fmm;
    constexpr int CANON_G_STRIDE = 28 * 32, G_STRIDE = CANON_G_STRIDE;
    constexpr int NUM_ORIENT = 48, NC = 28;
    // OID_GEN_TO_V2 — orientation ID mapping between generator conventions
    // (moved from the kernel; now used only at host init time)
    constexpr uint8_t OID_GEN_TO_V2[48] = {
        0,4,2,6,1,5,3,7, 8,12,9,13,10,14,11,15,
        16,18,20,22,17,19,21,23, 32,34,33,35,36,38,37,39,
        24,25,28,29,26,27,30,31, 40,41,42,43,44,45,46,47,
    };

    // Gaps supported in Phase 1
    const int STORAGE_ORDER[3] = {0, 1, -1};
    const int COUNTS[3] = {
        G_GAP0_COUNT,        // gap 0
        G_GAP1_FWD_COUNT + G_GAP1_PLANAR_COUNT,  // gap +1: IDs 0..28
        G_GAP1_REV_COUNT,    // gap -1
    };
    const float* GSRC_PTRS[3] = {
        &G_GAP0[0][0][0], &G_GAP1_FWD[0][0][0], &G_GAP1_REV[0][0][0],
    };


    std::vector<float> payload;
    for (int gi = 0; gi < 3; ++gi) {
        const int signed_gap = STORAGE_ORDER[gi], count = COUNTS[gi];
        const float* Gsrc = GSRC_PTRS[gi];
        std::vector<float> host_segment(size_t(count) * NUM_ORIENT * G_STRIDE, 0.0f);
        for (int cid = 0; cid < count; ++cid) {
            // The gap +1 planar operators are appended after the historic
            // contiguous 25-entry source table.  Select them by canonical ID
            // while preserving IDs 0..24 and the unchanged gap -1 segment.
            const float* Gsrc_cid = (signed_gap == 1 && cid >= G_GAP1_PLANAR_BASE)
                ? &G_GAP1_PLANAR[cid - G_GAP1_PLANAR_BASE][0][0]
                : Gsrc + (size_t)cid * CANON_G_STRIDE;
            for (int o_gen = 0; o_gen < NUM_ORIENT; ++o_gen) {
                uint8_t v2 = OID_GEN_TO_V2[o_gen];

                // Two-matmul COMPOSE (see Global Constraints):
                //   C[k][lane] = sum_tc Gsrc[cid][k][tc] * B_R[v2][lane][tc]
                //   G_full[sc][lane] = sum_k A_R[v2][k][sc] * C[k][lane]
                // intermediate in double for precision
                double C[NC][NC] = {};
                for (int k = 0; k < NC; ++k) {
                    for (int lane = 0; lane < NC; ++lane) {
                        double sum = 0.0;
                        for (int tc = 0; tc < NC; ++tc) {
                            sum += (double)Gsrc_cid[(size_t)k * 32 + tc]
                                 * (double)G_B_R[v2][lane][tc];
                        }
                        C[k][lane] = sum;
                    }
                }

                size_t orient_base = (size_t)cid * NUM_ORIENT + (size_t)o_gen;
                for (int sc = 0; sc < NC; ++sc) {
                    for (int lane = 0; lane < NC; ++lane) {
                        double sum = 0.0;
                        for (int k = 0; k < NC; ++k) {
                            sum += (double)G_A_R[v2][k][sc]
                                 * C[k][lane];
                        }
                        float stored = static_cast<float>(sum);

                        // Padded columns and (for TF32_TC) padded K rows remain
                        // zero from vector initialization.
                        host_segment[orient_base * G_STRIDE +
                                     (size_t)sc * 32 + lane] = stored;
                    }
                }
            }
        }


        payload.insert(payload.end(), host_segment.begin(), host_segment.end());
    }
    std::vector<unsigned char> bytes;
    for (float value : payload) {
        uint32_t bits; std::memcpy(&bits, &value, 4);
        for (int i = 0; i < 4; ++i) bytes.push_back((bits >> (8*i)) & 0xff);
    }
    std::ofstream out(argv[1], std::ios::binary);
    out.write("GRANITG1", 8);
    auto word = [&](uint64_t value, int n) {
        for (int i=0; i<n; ++i) out.put(static_cast<char>((value >> (8*i)) & 0xff));
    };
    for (uint32_t value : granit_g_table::metadata) word(value, 4);
    word(granit_g_table::checksum(bytes.data(), bytes.size()), 8);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!out) { std::cerr << "G-table write failed\n"; return 1; }
    std::cout << payload.size() << " coefficients written\n";
}

#pragma once

#include "p2p_topology.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

namespace cudadem {

enum class IommuMode { Absent, Identity, Translated, Unknown };

inline std::string normalize_pci_bus_id(const std::string& input) {
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    int consumed = 0;
    if (std::sscanf(input.c_str(), "%x:%x:%x.%x%n", &domain, &bus,
                    &device, &function, &consumed) != 4 ||
        consumed != static_cast<int>(input.size()) || domain > 0xffff ||
        bus > 0xff || device > 0x1f || function > 7) return {};
    char canonical[32];
    std::snprintf(canonical, sizeof(canonical), "%04x:%02x:%02x.%x",
                  domain, bus, device, function);
    return canonical;
}

inline IommuMode parse_iommu_type(std::string type) {
    const auto first = std::find_if_not(type.begin(), type.end(),
        [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(type.rbegin(), type.rend(),
        [](unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) return IommuMode::Unknown;
    type = std::string(first, last);
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return type == "identity" ? IommuMode::Identity : IommuMode::Translated;
}

inline bool should_stage_peer_copy(bool force_host_staging, LinkClass link,
                                   IommuMode destination, IommuMode source) {
    if (force_host_staging) return true;
    if (is_nvlink_fabric(link)) return false;
    return destination == IommuMode::Translated || source == IommuMode::Translated ||
           destination == IommuMode::Unknown || source == IommuMode::Unknown;
}

} // namespace cudadem

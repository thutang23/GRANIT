#ifndef CUDADEM_P2P_TOPOLOGY_H
#define CUDADEM_P2P_TOPOLOGY_H

namespace cudadem {

// The transport class is deliberately conservative.  CUDA P2P capability says
// nothing about the physical fabric, so an unverified route stays in the
// generic class even when native atomics are available.
enum class LinkClass {
    Self,
    Unavailable,
    GenericPcieP2P,
    DirectNvLink,
    NvSwitch
};

inline constexpr bool is_nvlink_fabric(const LinkClass link) {
    return link == LinkClass::DirectNvLink || link == LinkClass::NvSwitch;
}

// Kept pure so it can be unit-tested without CUDA or NVML.  The two proof
// inputs are derived from NVML only after all of their active-link evidence has
// been checked.
inline constexpr LinkClass classify_p2p_link(const bool is_self,
                                              const bool cuda_p2p_available,
                                              const bool direct_nvlink_proven,
                                              const bool nvswitch_path_proven) {
    if (is_self) return LinkClass::Self;
    if (!cuda_p2p_available) return LinkClass::Unavailable;
    if (direct_nvlink_proven) return LinkClass::DirectNvLink;
    if (nvswitch_path_proven) return LinkClass::NvSwitch;
    return LinkClass::GenericPcieP2P;
}

// Ownership changes cannot be inferred from a net local-count delta: an equal
// send/receive exchange still changes the locally owned particle set.
inline constexpr bool migration_membership_changed(const bool any_send,
                                                   const bool any_receive) {
    return any_send || any_receive;
}

} // namespace cudadem

#endif

#include "src/domain/peer_transport_policy.h"
#include <stdexcept>

using namespace cudadem;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)

int main() {
    CHECK(normalize_pci_bus_id("00000000:3B:00.0") == "0000:3b:00.0");
    CHECK(normalize_pci_bus_id("0000:3b:00.0") == "0000:3b:00.0");
    CHECK(normalize_pci_bus_id("00000000:B1:00.0") ==
          normalize_pci_bus_id("0000:b1:00.0"));
    CHECK(normalize_pci_bus_id("0000:3b:20.0").empty());
    CHECK(normalize_pci_bus_id("0000:3b:00.0 trailing").empty());

    CHECK(parse_iommu_type("identity\n") == IommuMode::Identity);
    CHECK(parse_iommu_type("  IDENTITY  ") == IommuMode::Identity);
    CHECK(parse_iommu_type("DMA-FQ\n") == IommuMode::Translated);
    CHECK(parse_iommu_type("DMA") == IommuMode::Translated);
    CHECK(parse_iommu_type("  ") == IommuMode::Unknown);

    const LinkClass pcie = LinkClass::GenericPcieP2P;
    const LinkClass nvlink = LinkClass::DirectNvLink;
    const LinkClass nvswitch = LinkClass::NvSwitch;
    CHECK(!should_stage_peer_copy(false, pcie, IommuMode::Absent, IommuMode::Absent));
    CHECK(!should_stage_peer_copy(false, pcie, IommuMode::Identity, IommuMode::Identity));
    CHECK(should_stage_peer_copy(false, pcie, IommuMode::Translated, IommuMode::Identity));
    CHECK(should_stage_peer_copy(false, pcie, IommuMode::Absent, IommuMode::Translated));
    CHECK(should_stage_peer_copy(false, pcie, IommuMode::Unknown, IommuMode::Absent));
    CHECK(should_stage_peer_copy(false, pcie, IommuMode::Absent, IommuMode::Unknown));
    CHECK(!should_stage_peer_copy(false, nvlink, IommuMode::Translated, IommuMode::Translated));
    CHECK(!should_stage_peer_copy(false, nvswitch, IommuMode::Unknown, IommuMode::Unknown));
    CHECK(should_stage_peer_copy(true, nvlink, IommuMode::Absent, IommuMode::Absent));
}

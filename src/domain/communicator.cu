#include "communicator.h"
#include "peer_transfer.h"
#include <cub/cub.cuh>
#include <iostream>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// This is deliberately opt-in: CUDA itself cannot identify whether a P2P path
// is NVLink, and environments without libnvidia-ml must retain P2P support.
#if defined(CUDADEM_HAS_NVML)
#include <nvml.h>
#endif

namespace cudadem {

namespace {

ParticlePackSource make_particle_pack_source(const GPUDomainState& gpu_state) {
    const auto& particles = gpu_state.local_state.particles;
    return {
        particles.pos_x, particles.pos_y, particles.pos_z,
        particles.last_pos_x, particles.last_pos_y, particles.last_pos_z,
        particles.vel_x, particles.vel_y, particles.vel_z,
        particles.omega_x, particles.omega_y, particles.omega_z,
        particles.quat_w, particles.quat_x, particles.quat_y, particles.quat_z,
        particles.radius, particles.mass, particles.moment_of_inertia,
        particles.cohesive_strength, particles.support_extent,
        particles.density, particles.pressure, particles.viscosity,
        particles.c_sound, particles.drhodt, particles.rho_rate,
        particles.smoothing_length, particles.energy, particles.dedt,
        particles.energy_cold, particles.energy_thermal_peak, particles.pressure_peak, particles.fracture_area, particles.fragment_length, particles.damping_energy,
        particles.S_xx, particles.S_xy, particles.S_xz,
        particles.S_yy, particles.S_yz, particles.S_zz,
        particles.damage, particles.crack_strain, particles.flaws_ntot,
        particles.flaws_strain_min, particles.flaws_strain_max, particles.flaws_m0,
        particles.distension,
        particles.L_00, particles.L_01, particles.L_02,
        particles.L_10, particles.L_11, particles.L_12,
        particles.L_20, particles.L_21, particles.L_22,
        particles.global_id, particles.material_id,
        particles.clump_id, particles.clump_off_x, particles.clump_off_y, particles.clump_off_z,
        particles.gravity_force_x, particles.gravity_force_y, particles.gravity_force_z,
        particles.winding_x, particles.winding_y, particles.winding_z
    };
}

#if defined(CUDADEM_HAS_NVML)

struct NvmlLinkEvidence {
    bool direct_to_peer = false;
    bool shared_switch_with_peer = false;
};

struct NvmlRemoteEndpoint {
    std::string pci_id;
    bool is_switch = false;
};

// NVML reports active physical NVLink ports.  A remote PCI endpoint that is a
// peer GPU proves a direct link; a common endpoint typed by NVML as SWITCH
// proves that both GPUs attach to the same NVSwitch. All failed/ambiguous
// probes are intentionally left unclassified.
std::vector<std::vector<NvmlLinkEvidence>> query_nvml_link_evidence(
    const std::vector<std::string>& cuda_pci_bus_ids) {
    const size_t count = cuda_pci_bus_ids.size();
    std::vector<std::vector<NvmlLinkEvidence>> evidence(
        count, std::vector<NvmlLinkEvidence>(count));
    std::vector<std::string> normalized_gpu_ids(count);
    for (size_t i = 0; i < count; ++i) {
        normalized_gpu_ids[i] = normalize_pci_bus_id(cuda_pci_bus_ids[i]);
        if (normalized_gpu_ids[i].empty()) return evidence;
    }
    if (nvmlInit_v2() != NVML_SUCCESS) return evidence;

    std::vector<nvmlDevice_t> devices(count, nullptr);
    for (size_t i = 0; i < count; ++i) {
        if (nvmlDeviceGetHandleByPciBusId_v2(cuda_pci_bus_ids[i].c_str(), &devices[i]) != NVML_SUCCESS) {
            nvmlShutdown();
            return evidence;
        }
    }

    std::vector<std::vector<NvmlRemoteEndpoint>> active_remote_endpoints(count);
    for (size_t i = 0; i < count; ++i) {
        for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
            nvmlEnableState_t state = NVML_FEATURE_DISABLED;
            if (nvmlDeviceGetNvLinkState(devices[i], link, &state) != NVML_SUCCESS ||
                state != NVML_FEATURE_ENABLED) continue;
            nvmlPciInfo_t remote_pci{};
            if (nvmlDeviceGetNvLinkRemotePciInfo_v2(devices[i], link, &remote_pci) != NVML_SUCCESS ||
                remote_pci.busId[0] == '\0') continue;
            const std::string endpoint = normalize_pci_bus_id(remote_pci.busId);
            if (endpoint.empty()) continue;
            nvmlIntNvLinkDeviceType_t remote_type = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
            const bool is_switch =
                nvmlDeviceGetNvLinkRemoteDeviceType(devices[i], link, &remote_type) == NVML_SUCCESS &&
                remote_type == NVML_NVLINK_DEVICE_TYPE_SWITCH;
            active_remote_endpoints[i].push_back({endpoint, is_switch});
        }
    }

    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < count; ++j) {
            if (i == j) continue;
            for (const NvmlRemoteEndpoint& endpoint : active_remote_endpoints[i]) {
                if (endpoint.pci_id == normalized_gpu_ids[j]) {
                    evidence[i][j].direct_to_peer = true;
                    break;
                }
            }
            if (evidence[i][j].direct_to_peer) continue;
            for (const NvmlRemoteEndpoint& endpoint : active_remote_endpoints[i]) {
                if (!endpoint.is_switch) continue;
                // A GPU PCI endpoint would have established direct evidence
                // above (or be an unproven route); it is not treated as a
                // switch endpoint.
                bool is_gpu_endpoint = false;
                for (const std::string& gpu_pci : normalized_gpu_ids) {
                    if (endpoint.pci_id == gpu_pci) { is_gpu_endpoint = true; break; }
                }
                if (is_gpu_endpoint) continue;
                for (const NvmlRemoteEndpoint& peer_endpoint : active_remote_endpoints[j]) {
                    if (peer_endpoint.is_switch && endpoint.pci_id == peer_endpoint.pci_id) {
                        evidence[i][j].shared_switch_with_peer = true;
                        break;
                    }
                }
                if (evidence[i][j].shared_switch_with_peer) break;
            }
        }
    }
    nvmlShutdown();
    return evidence;
}

#endif // CUDADEM_HAS_NVML

} // namespace

// =============================================================================
// Invert Flags Kernel (Phase 5: replaces thrust::transform for keep = 1 - migrate)
// =============================================================================

__global__ void invert_flags_kernel(
    const int32_t* __restrict__ is_migrating,
    int32_t* __restrict__ keep_flags,
    int32_t num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        keep_flags[idx] = 1 - is_migrating[idx];
    }
}

__global__ void collect_tail_fill_pairs_kernel(const int32_t* migrating, int32_t old_count, int32_t new_count,
                                               int32_t* holes, int32_t* sources, int32_t* hole_count, int32_t* source_count) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= old_count) return;
    if (i < new_count && migrating[i]) holes[atomicAdd(hole_count, 1)] = i;
    if (i >= new_count && !migrating[i]) sources[atomicAdd(source_count, 1)] = i;
}

__global__ void count_migrants_kernel(const int32_t* migrating, int32_t n, int32_t* count) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && migrating[i]) atomicAdd(count, 1);
}

__global__ void copy_tail_fill_pairs_kernel(const int32_t* holes, const int32_t* sources, int32_t pairs,
    ParticleSoA p, ContactHistory* contacts, WallContactHistory* walls, int32_t cs, int32_t ws) {
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= pairs) return;
    const int32_t d = holes[k], s = sources[k];
#define COPY_FIELD(x) p.x[d] = p.x[s]
    COPY_FIELD(pos_x); COPY_FIELD(pos_y); COPY_FIELD(pos_z);
    COPY_FIELD(last_pos_x); COPY_FIELD(last_pos_y); COPY_FIELD(last_pos_z);
    COPY_FIELD(winding_x); COPY_FIELD(winding_y); COPY_FIELD(winding_z);
    COPY_FIELD(vel_x); COPY_FIELD(vel_y); COPY_FIELD(vel_z);
    COPY_FIELD(force_x); COPY_FIELD(force_y); COPY_FIELD(force_z);
    COPY_FIELD(gravity_force_x); COPY_FIELD(gravity_force_y); COPY_FIELD(gravity_force_z);
    COPY_FIELD(acc_x); COPY_FIELD(acc_y); COPY_FIELD(acc_z);
    COPY_FIELD(omega_x); COPY_FIELD(omega_y); COPY_FIELD(omega_z);
    COPY_FIELD(torque_x); COPY_FIELD(torque_y); COPY_FIELD(torque_z);
    COPY_FIELD(alpha_x); COPY_FIELD(alpha_y); COPY_FIELD(alpha_z);
    COPY_FIELD(quat_w); COPY_FIELD(quat_x); COPY_FIELD(quat_y); COPY_FIELD(quat_z);
    COPY_FIELD(radius); COPY_FIELD(mass); COPY_FIELD(moment_of_inertia);
    COPY_FIELD(support_extent); COPY_FIELD(cohesive_strength);
    COPY_FIELD(density); COPY_FIELD(pressure); COPY_FIELD(viscosity); COPY_FIELD(c_sound);
    COPY_FIELD(drhodt); COPY_FIELD(rho_rate); COPY_FIELD(smoothing_length);
    COPY_FIELD(energy); COPY_FIELD(dedt);
    COPY_FIELD(energy_cold); COPY_FIELD(energy_thermal_peak); COPY_FIELD(pressure_peak); COPY_FIELD(fracture_area); COPY_FIELD(fragment_length); COPY_FIELD(damping_energy);
    COPY_FIELD(S_xx); COPY_FIELD(S_xy); COPY_FIELD(S_xz); COPY_FIELD(S_yy); COPY_FIELD(S_yz); COPY_FIELD(S_zz);
    COPY_FIELD(damage); COPY_FIELD(crack_strain); COPY_FIELD(flaws_ntot);
    COPY_FIELD(flaws_strain_min); COPY_FIELD(flaws_strain_max); COPY_FIELD(flaws_m0); COPY_FIELD(distension);
    COPY_FIELD(L_00); COPY_FIELD(L_01); COPY_FIELD(L_02); COPY_FIELD(L_10); COPY_FIELD(L_11); COPY_FIELD(L_12);
    COPY_FIELD(L_20); COPY_FIELD(L_21); COPY_FIELD(L_22);
    COPY_FIELD(num_contacts); COPY_FIELD(num_wall_contacts);
    COPY_FIELD(cell_id); COPY_FIELD(sorted_index); COPY_FIELD(global_id); COPY_FIELD(material_id);
    COPY_FIELD(clump_id); COPY_FIELD(clump_off_x); COPY_FIELD(clump_off_y); COPY_FIELD(clump_off_z);
#undef COPY_FIELD
    for (int32_t i = 0; i < cs; ++i) contacts[static_cast<size_t>(d) * cs + i] = contacts[static_cast<size_t>(s) * cs + i];
    for (int32_t i = 0; i < ws; ++i) walls[static_cast<size_t>(d) * ws + i] = walls[static_cast<size_t>(s) * ws + i];
    p.contact_history_offset[d] = d * cs; p.wall_history_offset[d] = d * ws;
}

// =============================================================================
// Communicator Implementation
// =============================================================================

void Communicator::initialize(int32_t device_id, int32_t num_gpus, int32_t max_particles,
                              int32_t contact_history_stride, int32_t wall_history_stride) {
    device_id_ = device_id;
    num_gpus_ = num_gpus;
    max_particles_ = max_particles;
    contact_history_stride_ = contact_history_stride;
    wall_history_stride_ = wall_history_stride;

    if (max_particles_ <= 0 || contact_history_stride_ <= 0 || wall_history_stride_ <= 0) {
        throw std::runtime_error("Migration capacity and history strides must be positive");
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    // Query P2P topology
    query_topology();

    // Allocate communication buffers for each peer GPU
    buffers_.resize(num_gpus);

    // Payload storage starts empty. It is grown as one aggregate arena after
    // classification has produced exact per-peer demand.
    send_particle_capacity_ = 0;

    for (int32_t peer = 0; peer < num_gpus; ++peer) {
        if (peer == device_id) continue;

        CommBuffer& buf = buffers_[peer];
        buf.particle_capacity = 0;
        buf.contact_history_stride = contact_history_stride_;
        buf.wall_history_stride = wall_history_stride_;
        buf.history_capacity = 0;
        buf.wall_history_capacity = 0;

        // Counts retain their old ownership; payload pointers become slices of
        // the aggregate arenas in layout_send_arena/layout_recv_arena.
        CUDA_CHECK(cudaMalloc(&buf.d_send_count, sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_recv_count, sizeof(int32_t)));

        // Initialize counts to zero
        CUDA_CHECK(cudaMemset(buf.d_send_count, 0, sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(buf.d_recv_count, 0, sizeof(int32_t)));
    }

    // Make the existing per-peer send buffers addressable by one
    // particle-parallel direct-scatter kernel.
    std::vector<PackedParticle*> h_send_particle_buffers(num_gpus, nullptr);
    std::vector<int32_t*> h_send_index_buffers(num_gpus, nullptr);
    for (int32_t peer = 0; peer < num_gpus; ++peer) {
        if (peer == device_id) continue;
        h_send_particle_buffers[peer] = buffers_[peer].d_send_particles;
        h_send_index_buffers[peer] = buffers_[peer].d_send_indices;
    }
    CUDA_CHECK(cudaMalloc(&d_send_particle_buffers_, num_gpus * sizeof(PackedParticle*)));
    CUDA_CHECK(cudaMalloc(&d_send_index_buffers_, num_gpus * sizeof(int32_t*)));
    CUDA_CHECK(cudaMemcpy(d_send_particle_buffers_, h_send_particle_buffers.data(),
                          num_gpus * sizeof(PackedParticle*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_send_index_buffers_, h_send_index_buffers.data(),
                          num_gpus * sizeof(int32_t*), cudaMemcpyHostToDevice));

    // Allocate halo identification buffers
    CUDA_CHECK(cudaMalloc(&d_halo_count_, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_halo_dest_mask_, static_cast<size_t>(max_particles) * ((num_gpus + 31) / 32) * sizeof(int32_t)));

    // Allocate migration buffers
    CUDA_CHECK(cudaMalloc(&d_migrate_indices_, max_particles * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_migrate_count_, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_migrate_dest_gpu_, max_particles * sizeof(int32_t)));

    // Allocate compaction buffers
    CUDA_CHECK(cudaMalloc(&d_keep_flags_, max_particles * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_new_indices_, max_particles * sizeof(int32_t)));

    // =========================================================================
    // Pre-allocate temporary buffers (Phase 1)
    // =========================================================================
    CUDA_CHECK(cudaMalloc(&d_gpu_box_min_x_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpu_box_min_y_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpu_box_min_z_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpu_box_max_x_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpu_box_max_y_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpu_box_max_z_, MAX_MULTI_GPUS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_history_overflow_status_, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_history_payload_counters_, 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_send_contact_history_counts_, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_send_wall_history_counts_, num_gpus * sizeof(int32_t)));

    // =========================================================================
    // Pinned host memory (Phase 2)
    // =========================================================================
    CUDA_CHECK(cudaMallocHost(&h_pinned_send_counts_, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_scan_result_, 2 * sizeof(int32_t)));
    memset(h_pinned_send_counts_, 0, num_gpus * sizeof(int32_t));
    memset(h_pinned_scan_result_, 0, 2 * sizeof(int32_t));

    // =========================================================================
    // CUB temp storage (Phase 5: pre-allocate for exclusive scan)
    // =========================================================================
    cub_scan_temp_bytes_ = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_scan_temp_bytes_,
                                   d_keep_flags_, d_new_indices_, max_particles);
    CUDA_CHECK(cudaMalloc(&d_cub_scan_temp_, cub_scan_temp_bytes_));

    // =========================================================================
    // Contiguous send counts (Phase 7: batch memset + memcpy)
    // =========================================================================
    CUDA_CHECK(cudaMalloc(&d_all_send_counts_, num_gpus * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_all_send_counts_, 0, num_gpus * sizeof(int32_t)));

    std::cout << "Communicator initialized for GPU " << device_id
              << " with " << num_gpus << " total GPUs" << std::endl;
}

void Communicator::query_topology() {
    topology_.can_access.resize(num_gpus_, std::vector<bool>(num_gpus_, false));
    topology_.is_nvlink.resize(num_gpus_, std::vector<bool>(num_gpus_, false));
    topology_.link_class.resize(num_gpus_, std::vector<LinkClass>(num_gpus_, LinkClass::Unavailable));
    topology_.nvlink_available = false;

    std::vector<std::string> cuda_pci_bus_ids(num_gpus_);
    for (int32_t i = 0; i < num_gpus_; ++i) {
        char bus_id[32] = {};
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), i) == cudaSuccess) {
            cuda_pci_bus_ids[i] = bus_id;
        }
    }

#if defined(CUDADEM_HAS_NVML)
    const auto nvml_evidence = query_nvml_link_evidence(cuda_pci_bus_ids);
#endif

    for (int32_t i = 0; i < num_gpus_; ++i) {
        for (int32_t j = 0; j < num_gpus_; ++j) {
            if (i == j) {
                topology_.can_access[i][j] = true;
                topology_.link_class[i][j] = LinkClass::Self;
                continue;
            }

            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
            topology_.can_access[i][j] = (can_access != 0);

            bool direct_nvlink_proven = false;
            bool nvswitch_path_proven = false;
#if defined(CUDADEM_HAS_NVML)
            direct_nvlink_proven = nvml_evidence[i][j].direct_to_peer;
            nvswitch_path_proven = nvml_evidence[i][j].shared_switch_with_peer;
#endif
            const LinkClass link = classify_p2p_link(false, can_access != 0,
                                                     direct_nvlink_proven,
                                                     nvswitch_path_proven);
            topology_.link_class[i][j] = link;
            register_peer_link_class(cuda_pci_bus_ids[i], cuda_pci_bus_ids[j], link);
            topology_.is_nvlink[i][j] = is_nvlink_fabric(link);
            topology_.nvlink_available = topology_.nvlink_available || is_nvlink_fabric(link);
        }
    }

    if (topology_.nvlink_available) {
        std::cout << "NVLink/NVSwitch connections verified by NVML" << std::endl;
    }
}

CommunicationAllocationTelemetry Communicator::allocation_telemetry() const {
    CommunicationAllocationTelemetry telemetry;
    telemetry.send_particle_bytes = static_cast<size_t>(send_arena_capacity_) * sizeof(PackedParticle);
    telemetry.recv_particle_bytes = static_cast<size_t>(recv_arena_capacity_) * sizeof(PackedParticle);
    telemetry.send_history_bytes = static_cast<size_t>(send_history_arena_capacity_) * sizeof(ContactHistoryPacked)
        + static_cast<size_t>(send_wall_history_arena_capacity_) * sizeof(WallContactHistoryPacked);
    telemetry.recv_history_bytes = static_cast<size_t>(recv_history_arena_capacity_) * sizeof(ContactHistoryPacked)
        + static_cast<size_t>(recv_wall_history_arena_capacity_) * sizeof(WallContactHistoryPacked);
    telemetry.send_index_bytes = static_cast<size_t>(send_arena_capacity_) * sizeof(int32_t);
    // Five particle-indexed int32 arrays: halo mask, migration indices and
    // destinations, keep flags, and compacted indices. Scalar/pointer/box
    // metadata and the CUB scan workspace are also live communicator memory.
    telemetry.workspace_bytes = static_cast<size_t>(max_particles_) * (4 + (num_gpus_ + 31) / 32) * sizeof(int32_t)
        + cub_scan_temp_bytes_
        + static_cast<size_t>(2 * num_gpus_) * sizeof(void*)
        + static_cast<size_t>(5 * num_gpus_ + 3) * sizeof(int32_t)
        + static_cast<size_t>(6 * MAX_MULTI_GPUS) * sizeof(double);
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        const CommBuffer& buffer = buffers_[peer];
        telemetry.active_send_peers += buffer.h_send_count > 0;
        telemetry.active_recv_peers += buffer.h_recv_count > 0;
    }
    return telemetry;
}

HaloTransferSummary Communicator::halo_transfer_summary() const {
    HaloTransferSummary summary;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        const CommBuffer& buffer = buffers_[peer];
        summary.sent_particles += buffer.h_send_count;
        summary.received_particles += buffer.h_recv_count;
        summary.active_send_peers += buffer.h_send_count > 0;
        summary.active_receive_peers += buffer.h_recv_count > 0;
    }
    summary.sent_bytes = static_cast<size_t>(summary.sent_particles) * sizeof(PackedParticle);
    summary.received_bytes = static_cast<size_t>(summary.received_particles) * sizeof(PackedParticle);
    return summary;
}

namespace {
int32_t grown_capacity(const int32_t required, const int32_t current, const char* arena, const int32_t gpu) {
    if (required < 0 || required > std::numeric_limits<int32_t>::max() / 3 * 2)
        throw std::runtime_error(std::string("Communication arena size overflow for ") + arena +
                                 " on GPU " + std::to_string(gpu));
    const int64_t grown = static_cast<int64_t>(current) + (static_cast<int64_t>(current) + 1) / 2;
    return static_cast<int32_t>(std::max<int64_t>(required, grown));
}

void log_communication_growth(const int32_t gpu, const char* arena,
                              const int32_t old_capacity,
                              const int32_t required,
                              const int32_t new_capacity,
                              const size_t element_bytes) {
    std::printf("[memory] GPU %d communication grow arena=%s old_capacity=%d "
                "required=%d new_capacity=%d bytes=%zu\n",
                gpu, arena, old_capacity, required, new_capacity,
                static_cast<size_t>(new_capacity) * element_bytes);
}
}

void Communicator::ensure_workspace_capacity(const int32_t required_capacity) {
    if (required_capacity <= max_particles_) return;
    const int32_t old_capacity = max_particles_;
    const int32_t new_capacity = grown_capacity(
        required_capacity, old_capacity, "particle workspace", device_id_);

    int32_t* new_halo_dest_mask = nullptr;
    int32_t* new_migrate_indices = nullptr;
    int32_t* new_migrate_dest = nullptr;
    int32_t* new_keep_flags = nullptr;
    int32_t* new_indices = nullptr;
    void* new_cub_scan_temp = nullptr;
    size_t new_cub_scan_temp_bytes = 0;
    try {
        CUDA_CHECK(cudaMalloc(&new_halo_dest_mask,
                              static_cast<size_t>(new_capacity) * ((num_gpus_ + 31) / 32) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&new_migrate_indices,
                              static_cast<size_t>(new_capacity) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&new_migrate_dest,
                              static_cast<size_t>(new_capacity) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&new_keep_flags,
                              static_cast<size_t>(new_capacity) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&new_indices,
                              static_cast<size_t>(new_capacity) * sizeof(int32_t)));
        cub::DeviceScan::ExclusiveSum(nullptr, new_cub_scan_temp_bytes,
                                      new_keep_flags, new_indices, new_capacity);
        CUDA_CHECK(cudaMalloc(&new_cub_scan_temp, new_cub_scan_temp_bytes));
        // Halo unpack can grow this workspace after classification. Preserve
        // its owned rows for the work collector at the end of the same epoch.
        const size_t mask_stride = static_cast<size_t>((num_gpus_ + 31) / 32);
        CUDA_CHECK(cudaMemset(new_halo_dest_mask, 0,
                              static_cast<size_t>(new_capacity) * mask_stride * sizeof(int32_t)));
        if (d_halo_dest_mask_ && old_capacity)
            CUDA_CHECK(cudaMemcpy(new_halo_dest_mask, d_halo_dest_mask_,
                                  static_cast<size_t>(old_capacity) * mask_stride * sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice));
    } catch (const std::exception& e) {
        cudaFree(new_halo_dest_mask); cudaFree(new_migrate_indices);
        cudaFree(new_migrate_dest); cudaFree(new_keep_flags);
        cudaFree(new_indices); cudaFree(new_cub_scan_temp);
        throw std::runtime_error(
            "FATAL communication workspace resize on GPU "
            + std::to_string(device_id_) + ": old_capacity="
            + std::to_string(old_capacity) + ", required_capacity="
            + std::to_string(required_capacity) + ", requested_capacity="
            + std::to_string(new_capacity) + ", cuda_error=" + e.what());
    }

    cudaFree(d_halo_dest_mask_); cudaFree(d_migrate_indices_);
    cudaFree(d_migrate_dest_gpu_); cudaFree(d_keep_flags_);
    cudaFree(d_new_indices_); cudaFree(d_cub_scan_temp_);
    d_halo_dest_mask_ = new_halo_dest_mask;
    d_migrate_indices_ = new_migrate_indices;
    d_migrate_dest_gpu_ = new_migrate_dest;
    d_keep_flags_ = new_keep_flags;
    d_new_indices_ = new_indices;
    d_cub_scan_temp_ = new_cub_scan_temp;
    cub_scan_temp_bytes_ = new_cub_scan_temp_bytes;
    max_particles_ = new_capacity;
    std::printf("[memory] GPU %d communication grow arena=particle-workspace "
                "old_capacity=%d required=%d new_capacity=%d bytes=%zu\n",
                device_id_, old_capacity, required_capacity, new_capacity,
                allocation_telemetry().workspace_bytes);
}

void Communicator::layout_send_arena(const std::vector<int32_t>& counts) {
    int64_t total = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) if (peer != device_id_) total += counts[peer];
    if (total > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("Communication send arena exceeds int32 indexing on GPU " + std::to_string(device_id_));
    const int32_t required = static_cast<int32_t>(total);
    if (required > send_arena_capacity_) {
        const int32_t old_capacity = send_arena_capacity_;
        const int32_t capacity = grown_capacity(required, send_arena_capacity_, "send", device_id_);
        cudaFree(d_send_particle_arena_); cudaFree(d_send_index_arena_);
        d_send_particle_arena_ = nullptr; d_send_index_arena_ = nullptr;
        cudaError_t status = cudaMalloc(&d_send_particle_arena_, static_cast<size_t>(capacity) * sizeof(PackedParticle));
        if (status != cudaSuccess) throw std::runtime_error("OOM growing send particle arena on GPU " + std::to_string(device_id_) + ": " + cudaGetErrorString(status));
        CUDA_CHECK(cudaMalloc(&d_send_index_arena_, static_cast<size_t>(capacity) * sizeof(int32_t)));
        send_arena_capacity_ = capacity;
        log_communication_growth(device_id_, "send-particle-index", old_capacity,
                                 required, capacity,
                                 sizeof(PackedParticle) + sizeof(int32_t));
    }
    int32_t offset = 0;
    std::vector<PackedParticle*> particles(num_gpus_, nullptr); std::vector<int32_t*> indices(num_gpus_, nullptr);
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        CommBuffer& b = buffers_[peer]; const int32_t count = counts[peer];
        b.particle_capacity = count;
        b.d_send_particles = count > 0 ? d_send_particle_arena_ + offset : nullptr;
        b.d_send_indices = count > 0 ? d_send_index_arena_ + offset : nullptr;
        particles[peer] = b.d_send_particles; indices[peer] = b.d_send_indices; offset += count;
    }
    send_particle_capacity_ = required;
    CUDA_CHECK(cudaMemcpy(d_send_particle_buffers_, particles.data(), num_gpus_ * sizeof(PackedParticle*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_send_index_buffers_, indices.data(), num_gpus_ * sizeof(int32_t*), cudaMemcpyHostToDevice));
}

void Communicator::layout_recv_arena(const std::vector<int32_t>& counts) {
    int64_t total = 0; for (int32_t peer = 0; peer < num_gpus_; ++peer) if (peer != device_id_) total += counts[peer];
    if (total > std::numeric_limits<int32_t>::max()) throw std::runtime_error("Communication receive arena exceeds int32 indexing on GPU " + std::to_string(device_id_));
    const int32_t required = static_cast<int32_t>(total);
    if (required > recv_arena_capacity_) {
        const int32_t old_capacity = recv_arena_capacity_;
        const int32_t capacity = grown_capacity(required, recv_arena_capacity_, "receive", device_id_);
        cudaFree(d_recv_particle_arena_);
        d_recv_particle_arena_ = nullptr;
        cudaError_t status = cudaMalloc(&d_recv_particle_arena_, static_cast<size_t>(capacity) * sizeof(PackedParticle));
        if (status != cudaSuccess) throw std::runtime_error("OOM growing receive particle arena on GPU " + std::to_string(device_id_) + ": " + cudaGetErrorString(status));
        recv_arena_capacity_ = capacity;
        log_communication_growth(device_id_, "receive-particle", old_capacity,
                                 required, capacity, sizeof(PackedParticle));
    }
    int32_t offset = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) if (peer != device_id_) {
        CommBuffer& b = buffers_[peer]; const int32_t count = counts[peer];
        b.d_recv_particles = count > 0 ? d_recv_particle_arena_ + offset : nullptr;
        offset += count;
    }
}

void Communicator::layout_send_history_arenas(const std::vector<int32_t>& contacts, const std::vector<int32_t>& walls) {
    int64_t cr = 0, wr = 0;
    for (int p = 0; p < num_gpus_; ++p) {
        if (p != device_id_) { cr += contacts[p]; wr += walls[p]; }
    }
    if (cr > INT32_MAX || wr > INT32_MAX) {
        throw std::runtime_error("Migration send history arena exceeds int32 indexing on GPU "
                                 + std::to_string(device_id_));
    }
    const int32_t c = static_cast<int32_t>(cr), w = static_cast<int32_t>(wr);
    if (c > send_history_arena_capacity_) {
        const int32_t old = send_history_arena_capacity_;
        const int32_t cap = grown_capacity(c, old, "send contact history", device_id_);
        cudaFree(d_send_history_arena_); d_send_history_arena_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_send_history_arena_, static_cast<size_t>(cap) * sizeof(ContactHistoryPacked)));
        send_history_arena_capacity_ = cap;
        log_communication_growth(device_id_, "send-contact-history", old, c, cap,
                                 sizeof(ContactHistoryPacked));
    }
    if (w > send_wall_history_arena_capacity_) {
        const int32_t old = send_wall_history_arena_capacity_;
        const int32_t cap = grown_capacity(w, old, "send wall history", device_id_);
        cudaFree(d_send_wall_history_arena_); d_send_wall_history_arena_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_send_wall_history_arena_, static_cast<size_t>(cap) * sizeof(WallContactHistoryPacked)));
        send_wall_history_arena_capacity_ = cap;
        log_communication_growth(device_id_, "send-wall-history", old, w, cap,
                                 sizeof(WallContactHistoryPacked));
    }
    int32_t co = 0, wo = 0;
    for (int p = 0; p < num_gpus_; ++p) {
        if (p == device_id_) continue;
        auto& b = buffers_[p];
        b.history_capacity = contacts[p]; b.wall_history_capacity = walls[p];
        b.d_send_history = contacts[p] > 0 ? d_send_history_arena_ + co : nullptr;
        b.d_send_wall_history = walls[p] > 0 ? d_send_wall_history_arena_ + wo : nullptr;
        co += contacts[p]; wo += walls[p];
    }
}

void Communicator::layout_recv_history_arenas(const std::vector<int32_t>& contacts, const std::vector<int32_t>& walls) {
    int64_t cr = 0, wr = 0;
    for (int p = 0; p < num_gpus_; ++p) {
        if (p != device_id_) { cr += contacts[p]; wr += walls[p]; }
    }
    if (cr > INT32_MAX || wr > INT32_MAX) {
        throw std::runtime_error("Migration receive history arena exceeds int32 indexing on GPU "
                                 + std::to_string(device_id_));
    }
    const int32_t c = static_cast<int32_t>(cr), w = static_cast<int32_t>(wr);
    if (c > recv_history_arena_capacity_) {
        const int32_t old = recv_history_arena_capacity_;
        const int32_t cap = grown_capacity(c, old, "receive contact history", device_id_);
        cudaFree(d_recv_history_arena_); d_recv_history_arena_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_recv_history_arena_, static_cast<size_t>(cap) * sizeof(ContactHistoryPacked)));
        recv_history_arena_capacity_ = cap;
        log_communication_growth(device_id_, "receive-contact-history", old, c, cap,
                                 sizeof(ContactHistoryPacked));
    }
    if (w > recv_wall_history_arena_capacity_) {
        const int32_t old = recv_wall_history_arena_capacity_;
        const int32_t cap = grown_capacity(w, old, "receive wall history", device_id_);
        cudaFree(d_recv_wall_history_arena_); d_recv_wall_history_arena_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_recv_wall_history_arena_, static_cast<size_t>(cap) * sizeof(WallContactHistoryPacked)));
        recv_wall_history_arena_capacity_ = cap;
        log_communication_growth(device_id_, "receive-wall-history", old, w, cap,
                                 sizeof(WallContactHistoryPacked));
    }
    int32_t co = 0, wo = 0;
    for (int p = 0; p < num_gpus_; ++p) {
        if (p == device_id_) continue;
        auto& b = buffers_[p];
        b.history_capacity = contacts[p]; b.wall_history_capacity = walls[p];
        b.d_recv_history = contacts[p] > 0 ? d_recv_history_arena_ + co : nullptr;
        b.d_recv_wall_history = walls[p] > 0 ? d_recv_wall_history_arena_ + wo : nullptr;
        co += contacts[p]; wo += walls[p];
    }
}

// =============================================================================
// Halo Exchange Implementation
// =============================================================================

void Communicator::identify_halo_particles(
    GPUDomainState& gpu_state,
    const DomainParamsGPU& params,
    float halo_width,
    cudaStream_t stream,
    bool use_support_extent,
    float interaction_extent_scale
) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    for (int32_t peer = 0; peer < num_gpus_; ++peer) if (peer != device_id_) {
        buffers_[peer].h_send_history_count = 0;
        buffers_[peer].h_send_wall_history_count = 0;
    }

    int32_t num_local = gpu_state.subdomain.num_local_particles;
    int32_t capacity = gpu_state.subdomain.capacity;

    if (num_local <= 0) return;

    // Safety check: ensure arrays are allocated and within bounds
    if (gpu_state.d_is_halo == nullptr) {
        fprintf(stderr, "ERROR: d_is_halo is NULL on GPU %d\n", device_id_);
        return;
    }
    if (num_local > capacity) {
        fprintf(stderr, "ERROR: num_local (%d) > capacity (%d) on GPU %d\n",
                num_local, capacity, device_id_);
        return;
    }

    // Reset halo flags and destination masks (full capacity for safety)
    CUDA_CHECK(cudaMemsetAsync(gpu_state.d_is_halo, 0, capacity * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_halo_dest_mask_, 0, static_cast<size_t>(capacity) * ((num_gpus_ + 31) / 32) * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_halo_count_, 0, sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));

    int block_size = 256;
    int num_blocks = (num_local + block_size - 1) / block_size;

    // Copy ORB box arrays to pre-allocated device buffers
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_x_, params.gpu_box_min_x,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_y_, params.gpu_box_min_y,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_z_, params.gpu_box_min_z,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_x_, params.gpu_box_max_x,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_y_, params.gpu_box_max_y,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_z_, params.gpu_box_max_z,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));

    identify_halo_particles_orb_kernel<<<num_blocks, block_size, 0, stream>>>(
        gpu_state.local_state.particles.pos_x,
        gpu_state.local_state.particles.pos_y,
        gpu_state.local_state.particles.pos_z,
        use_support_extent ? gpu_state.local_state.particles.support_extent
                           : gpu_state.local_state.particles.radius,
        interaction_extent_scale,
        gpu_state.d_is_halo,
        d_halo_dest_mask_,
        num_local,
        halo_width,
        device_id_,
        num_gpus_, d_all_send_counts_
    );
}

void Communicator::pack_halo_particles(
    const GPUDomainState& gpu_state,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    int32_t num_local = gpu_state.subdomain.num_local_particles;
    if (num_local <= 0) {
        CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
        for (int32_t peer = 0; peer < num_gpus_; ++peer) {
            if (peer != device_id_) {
                buffers_[peer].h_send_count = 0;
                buffers_[peer].h_send_history_count = 0;
                buffers_[peer].h_send_wall_history_count = 0;
            }
        }
        return;
    }

    int block_size = 256;
    int num_blocks = (num_local + block_size - 1) / block_size;

    // =========================================================================
    // Single count reset, one direct-scatter launch, and one count readback.
    // =========================================================================

    // identify_halo_particles() already produced exact per-peer counts on this
    // stream.  Read those counts before laying out the aggregate arena.  A
    // former count pass reused stale per-peer slices from migration; with
    // three or more peers a zero-length slice could be null while the aggregate
    // capacity was nonzero, causing the count pass to write through address 0.
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_send_counts_, d_all_send_counts_,
                                num_gpus_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

    // Single sync point for count readback
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy counts to CommBuffer h_send_count fields
    // Cap counts at buffer capacity to prevent overflow in pack kernel
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        int32_t count = h_pinned_send_counts_[peer];
        // Buffer overflow is a fatal error — silent truncation causes particle loss
        buffers_[peer].h_send_count = count;
    }

    std::vector<int32_t> counts(h_pinned_send_counts_, h_pinned_send_counts_ + num_gpus_);
    layout_send_arena(counts);
    CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
    direct_scatter_halo_particles_kernel<<<num_blocks, block_size, 0, stream>>>(
        gpu_state.d_is_halo, d_halo_dest_mask_, d_send_particle_buffers_, d_send_index_buffers_,
        d_all_send_counts_, send_particle_capacity_, make_particle_pack_source(gpu_state),
        num_local, device_id_, num_gpus_);

}

void Communicator::repack_halo_particles(
    const GPUDomainState& gpu_state, cudaStream_t stream) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    const auto source = make_particle_pack_source(gpu_state);
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        const auto& buffer = buffers_[peer];
        if (buffer.h_send_count > 0)
            repack_halo_particles_kernel<<<(buffer.h_send_count+255)/256,256,0,stream>>>(
                buffer.d_send_particles, buffer.d_send_indices,
                buffer.h_send_count, source, device_id_);
    }
    CUDA_CHECK(cudaGetLastError());
}

void Communicator::exchange_halo_counts(
    std::vector<GPUDomainState>& all_gpu_states
) {
    // All-to-all count exchange
    // Each GPU needs to know how many particles it will receive from each peer

    // This is done via host staging (simple but not optimal)
    // Production code would use NCCL or direct P2P count exchange

    for (int32_t src = 0; src < num_gpus_; ++src) {
        for (int32_t dst = 0; dst < num_gpus_; ++dst) {
            if (src == dst) continue;

            // Get send count from src's buffer for dst
            // and store in dst's receive count from src
            // This requires coordination between all GPUs
        }
    }
}

void Communicator::transfer_halos(
    std::vector<Communicator>& all_communicators,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    std::vector<int32_t> incoming(num_gpus_, 0);
    for (int32_t peer = 0; peer < num_gpus_; ++peer)
        if (peer != device_id_) {
            const CommBuffer& source = all_communicators[peer].buffers_[device_id_];
            incoming[peer] = source.h_send_count;
        }
    layout_recv_arena(incoming);

    // Receive halos from all peer GPUs
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;

        // Get peer's communicator to access their send buffer
        Communicator& peer_comm = all_communicators[peer];

        // Peer's buffer for sending TO us (indexed by our device_id)
        CommBuffer& peer_send_buf = peer_comm.buffers_[device_id_];

        // Our buffer for receiving FROM peer
        CommBuffer& my_recv_buf = buffers_[peer];

        // Reset recv count — prevents stale values from previous phases
        // (e.g., migration reading leftover halo counts or vice versa)
        my_recv_buf.h_recv_count = 0;

        if (peer_send_buf.h_send_count > 0) {
            size_t transfer_size = peer_send_buf.h_send_count * sizeof(PackedParticle);

            // P2P transfer: read from peer's send buffer, write to our receive buffer
            copy_peer_payload(
                my_recv_buf.d_recv_particles,   // Destination: our receive buffer (on device_id_)
                device_id_,                      // Destination device: this GPU
                peer_send_buf.d_send_particles, // Source: peer's send buffer (on peer)
                peer,                            // Source device: peer GPU
                transfer_size,
                stream
            );

            // Update our receive count from peer's send count
            my_recv_buf.h_recv_count = peer_send_buf.h_send_count;
        }
    }
}

void Communicator::unpack_ghost_particles(
    GPUDomainState& gpu_state,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    int32_t ghost_start = gpu_state.subdomain.num_local_particles;
    int32_t total_received = 0;
    int32_t capacity = gpu_state.subdomain.capacity;

    // First pass: count total ghosts to check capacity
    int32_t total_ghosts = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        total_ghosts += buffers_[peer].h_recv_count;
    }

    // Phase-boundary growth is owned by DomainManager so all SoA fields and
    // persistent fixed-stride history storage remain coherent.
    ensure_particle_capacity(gpu_state, ghost_start + total_ghosts, "halo receive");
    ensure_workspace_capacity(gpu_state.subdomain.capacity);
    capacity = gpu_state.subdomain.capacity;

    // Ghost overflow is a fatal error — silent drops cause particle loss
    if (ghost_start + total_ghosts > capacity) {
        throw std::runtime_error(
            "Ghost particle overflow on GPU " + std::to_string(device_id_)
            + ": need " + std::to_string(ghost_start + total_ghosts) + " slots but capacity is "
            + std::to_string(capacity) + " (local=" + std::to_string(ghost_start) + ")");
    }

    int block_size = 256;

    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;

        CommBuffer& buf = buffers_[peer];
        if (buf.h_recv_count <= 0) continue;

        // Check if this batch would overflow
        int32_t count_to_unpack = buf.h_recv_count;
        if (ghost_start + total_received + count_to_unpack > capacity) {
            count_to_unpack = capacity - (ghost_start + total_received);
            if (count_to_unpack <= 0) break;  // No more space
        }

        int num_blocks = (count_to_unpack + block_size - 1) / block_size;

        unpack_ghost_particles_kernel<<<num_blocks, block_size, 0, stream>>>(
            buf.d_recv_particles,
            count_to_unpack,
            ghost_start + total_received,
            gpu_state.local_state.particles
        );

        total_received += count_to_unpack;
    }

    gpu_state.subdomain.num_ghost_particles = total_received;
    gpu_state.subdomain.ghost_start_index = ghost_start;

    // CRITICAL FIX: Update ParticleSoA counts to reflect local + ghost total.
    // Without this, state.particles.num_particles = local-only count after
    // migration, causing neighbor search (LBVH/CLL) to exclude ghost particles
    // from BVH construction → illegal memory access from stale internal state.
    gpu_state.local_state.particles.num_local = gpu_state.subdomain.num_local_particles;
    gpu_state.local_state.particles.num_ghost = total_received;
    gpu_state.local_state.particles.num_particles =
        gpu_state.subdomain.num_local_particles + total_received;

    // Zero auxiliary fields for ghost particles (force, torque, contact history).
    // Ghost particles only need pos/vel/radius for contact detection;
    // force/torque are accumulated by the contact kernel (ghost skip at owner_idx >= num_local).
    // contact_history_offset/num_contacts must be zeroed to prevent stale offsets
    // from causing out-of-bounds reads in compact_contact_history_kernel.
    if (total_received > 0) {
        size_t ghost_bytes_d = total_received * sizeof(double);
        size_t ghost_bytes_i = total_received * sizeof(int32_t);

        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.force_x + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.force_y + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.force_z + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.torque_x + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.torque_y + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.torque_z + ghost_start, 0, ghost_bytes_d, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.contact_history_offset + ghost_start, 0, ghost_bytes_i, stream));
        CUDA_CHECK(cudaMemsetAsync(
            gpu_state.local_state.particles.num_contacts + ghost_start, 0, ghost_bytes_i, stream));

        // Wall contact history fields for ghosts
        if (gpu_state.local_state.particles.wall_history_offset) {
            CUDA_CHECK(cudaMemsetAsync(
                gpu_state.local_state.particles.wall_history_offset + ghost_start, 0, ghost_bytes_i, stream));
        }
        if (gpu_state.local_state.particles.num_wall_contacts) {
            CUDA_CHECK(cudaMemsetAsync(
                gpu_state.local_state.particles.num_wall_contacts + ghost_start, 0, ghost_bytes_i, stream));
        }
    }
}

void Communicator::exchange_halos(
    std::vector<GPUDomainState>& all_gpu_states,
    std::vector<Communicator>& all_communicators,
    const DomainParamsGPU& params,
    float halo_width
) {
    GPUDomainState& my_state = all_gpu_states[device_id_];

    // Step 1: Identify halo particles
    identify_halo_particles(my_state, params, halo_width, my_state.compute_stream);

    // Step 2: Pack halo particles
    pack_halo_particles(my_state, my_state.compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(my_state.compute_stream));

    // Step 3: Exchange counts (requires barrier in caller)
    exchange_halo_counts(all_gpu_states);

    // Step 4: Transfer halos via P2P (using all communicators for peer buffer access)
    transfer_halos(all_communicators, my_state.comm_stream);

    CUDA_CHECK(cudaStreamSynchronize(my_state.comm_stream));

    // Step 5: Unpack ghost particles
    unpack_ghost_particles(my_state, my_state.compute_stream);
}

// =============================================================================
// Particle Migration Implementation
// =============================================================================

void Communicator::identify_migrating_particles(
    GPUDomainState& gpu_state,
    const DomainParamsGPU& params,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    particle_membership_changed_ = false;

    int32_t num_local = gpu_state.subdomain.num_local_particles;
    int32_t capacity = gpu_state.subdomain.capacity;

    if (num_local <= 0) return;

    // Safety check: ensure arrays are allocated and within bounds
    if (gpu_state.d_is_migrating == nullptr) {
        fprintf(stderr, "ERROR: d_is_migrating is NULL on GPU %d\n", device_id_);
        return;
    }
    if (num_local > capacity) {
        fprintf(stderr, "ERROR: num_local (%d) > capacity (%d) on GPU %d\n",
                num_local, capacity, device_id_);
        return;
    }

    // Reset migrate flags (full capacity for safety)
    CUDA_CHECK(cudaMemsetAsync(gpu_state.d_is_migrating, 0, capacity * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_migrate_count_, 0, sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));

    int block_size = 256;
    int num_blocks = (num_local + block_size - 1) / block_size;

    // Copy ORB box arrays to pre-allocated device buffers
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_x_, params.gpu_box_min_x,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_y_, params.gpu_box_min_y,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_min_z_, params.gpu_box_min_z,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_x_, params.gpu_box_max_x,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_y_, params.gpu_box_max_y,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gpu_box_max_z_, params.gpu_box_max_z,
                                MAX_MULTI_GPUS * sizeof(double), cudaMemcpyHostToDevice, stream));

    identify_migrating_particles_orb_kernel<<<num_blocks, block_size, 0, stream>>>(
        gpu_state.local_state.particles.pos_x,
        gpu_state.local_state.particles.pos_y,
        gpu_state.local_state.particles.pos_z,
        gpu_state.d_is_migrating,
        d_migrate_dest_gpu_,
        num_local,
        num_gpus_,
        device_id_,
        d_all_send_counts_
    );
}

void Communicator::pack_migrating_particles(
    GPUDomainState& gpu_state,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    int32_t num_local = gpu_state.subdomain.num_local_particles;
    if (num_local <= 0) {
        CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
        for (int32_t peer = 0; peer < num_gpus_; ++peer) {
            if (peer != device_id_) {
                buffers_[peer].h_send_count = 0;
                buffers_[peer].h_send_history_count = 0;
                buffers_[peer].h_send_wall_history_count = 0;
            }
        }
        return;
    }

    int block_size = 256;
    int num_blocks = (num_local + block_size - 1) / block_size;

    // =========================================================================
    // Single count reset, one direct-scatter launch, and one count readback.
    // =========================================================================

    // identify_migrating_particles() already counted every destination on the
    // same stream.  Size the arena from that classification, then perform the
    // sole payload scatter below.
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_send_counts_, d_all_send_counts_,
                                num_gpus_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

    // Single sync point for count readback
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy counts to CommBuffer h_send_count fields
    // Cap counts at buffer capacity to prevent overflow in pack kernel
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        int32_t count = h_pinned_send_counts_[peer];
        // Buffer overflow is a fatal error — silent truncation causes particle loss
        buffers_[peer].h_send_count = count;
        particle_membership_changed_ = migration_membership_changed(
            particle_membership_changed_, count > 0);
    }

    std::vector<int32_t> counts(h_pinned_send_counts_, h_pinned_send_counts_ + num_gpus_);
    layout_send_arena(counts);
    CUDA_CHECK(cudaMemsetAsync(d_all_send_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
    direct_scatter_migrating_particles_kernel<<<num_blocks, block_size, 0, stream>>>(
        gpu_state.d_is_migrating, d_migrate_dest_gpu_, d_send_particle_buffers_, d_send_index_buffers_,
        d_all_send_counts_, send_particle_capacity_, make_particle_pack_source(gpu_state),
        num_local, device_id_, num_gpus_);

    // Count active entries for every packed peer before allocating transit
    // history. Persistent storage remains fixed-stride; only the wire arena is
    // compact.
    CUDA_CHECK(cudaMemsetAsync(d_send_contact_history_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_send_wall_history_counts_, 0, num_gpus_ * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_history_overflow_status_, 0, sizeof(int32_t), stream));
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        CommBuffer& buf = buffers_[peer];
        if (buf.h_send_count <= 0) continue;
        const int pack_blocks = (buf.h_send_count + block_size - 1) / block_size;
        count_active_migration_history_kernel<<<pack_blocks, block_size, 0, stream>>>(buf.d_send_indices, buf.h_send_count,
            gpu_state.local_state.contact_history.data, gpu_state.local_state.wall_contact_history.data,
            gpu_state.local_state.particles.contact_history_offset, gpu_state.local_state.particles.wall_history_offset,
            gpu_state.local_state.particles.num_contacts, gpu_state.local_state.particles.num_wall_contacts,
            contact_history_stride_, wall_history_stride_, d_send_contact_history_counts_ + peer,
            d_send_wall_history_counts_ + peer, d_history_overflow_status_);
    }
    std::vector<int32_t> contact_counts(num_gpus_), wall_counts(num_gpus_);
    CUDA_CHECK(cudaMemcpyAsync(contact_counts.data(), d_send_contact_history_counts_, num_gpus_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(wall_counts.data(), d_send_wall_history_counts_, num_gpus_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_scan_result_, d_history_overflow_status_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_pinned_scan_result_[0]) {
        const char* kind = h_pinned_scan_result_[0] == 1 ? "contact" : "wall-contact";
        throw std::runtime_error(
            "Invalid " + std::string(kind) + " migration history count on GPU "
            + std::to_string(device_id_) + " (configured strides: contact="
            + std::to_string(contact_history_stride_) + ", wall-contact="
            + std::to_string(wall_history_stride_) + ")");
    }
    layout_send_history_arenas(contact_counts, wall_counts);
    for (int32_t peer=0; peer<num_gpus_; ++peer) if(peer!=device_id_) {
        CommBuffer& buf=buffers_[peer]; buf.h_send_history_count=contact_counts[peer]; buf.h_send_wall_history_count=wall_counts[peer];
        if (!buf.h_send_count) continue;
        CUDA_CHECK(cudaMemsetAsync(d_history_payload_counters_, 0, 2*sizeof(int32_t), stream));
        const int blocks=(buf.h_send_count+block_size-1)/block_size;
        pack_active_migration_history_kernel<<<blocks,block_size,0,stream>>>(buf.d_send_history,buf.d_send_wall_history,buf.d_send_particles,
            gpu_state.local_state.contact_history.data,gpu_state.local_state.wall_contact_history.data,
            gpu_state.local_state.particles.contact_history_offset,gpu_state.local_state.particles.wall_history_offset,
            gpu_state.local_state.particles.num_contacts,gpu_state.local_state.particles.num_wall_contacts,buf.h_send_count,buf.d_send_indices,
            contact_history_stride_,wall_history_stride_,d_history_payload_counters_,d_history_payload_counters_+1);
    }
}

void Communicator::transfer_migrants(
    std::vector<Communicator>& all_communicators,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    std::vector<int32_t> incoming(num_gpus_, 0);
    std::vector<int32_t> incoming_contacts(num_gpus_, 0);
    std::vector<int32_t> incoming_walls(num_gpus_, 0);
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;
        const CommBuffer& source = all_communicators[peer].buffers_[device_id_];
        incoming[peer] = source.h_send_count;
        incoming_contacts[peer] = source.h_send_history_count;
        incoming_walls[peer] = source.h_send_wall_history_count;
    }
    layout_recv_arena(incoming);
    layout_recv_history_arenas(incoming_contacts, incoming_walls);

    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;

        // Get peer's communicator to access their send buffer
        Communicator& peer_comm = all_communicators[peer];

        // Peer's buffer for sending TO us (indexed by our device_id)
        CommBuffer& peer_send_buf = peer_comm.buffers_[device_id_];

        // Our buffer for receiving FROM peer
        CommBuffer& my_recv_buf = buffers_[peer];

        // Reset recv count — prevents stale values from previous phases
        // (e.g., halo counts leaking into migration unpack or vice versa)
        my_recv_buf.h_recv_count = 0;

        if (peer_send_buf.h_send_count <= 0) continue;

        // A receive changes ownership membership even if the number sent from
        // this GPU makes the final local count unchanged.
        particle_membership_changed_ = migration_membership_changed(
            particle_membership_changed_, true);

        if (peer_send_buf.contact_history_stride != my_recv_buf.contact_history_stride ||
            peer_send_buf.wall_history_stride != my_recv_buf.wall_history_stride) {
            throw std::runtime_error(
                "Incompatible migration payload from GPU " + std::to_string(peer)
                + " to GPU " + std::to_string(device_id_));
        }

        {
            size_t transfer_size = peer_send_buf.h_send_count * sizeof(PackedParticle);

            // P2P transfer: read from peer's send buffer, write to our receive buffer
            copy_peer_payload(
                my_recv_buf.d_recv_particles,   // Destination: our receive buffer (on device_id_)
                device_id_,                      // Destination device: this GPU
                peer_send_buf.d_send_particles, // Source: peer's send buffer (on peer)
                peer,                            // Source device: peer GPU
                transfer_size,
                stream
            );
            if (peer_send_buf.h_send_history_count > 0) {
                copy_peer_payload(
                    my_recv_buf.d_recv_history, device_id_, peer_send_buf.d_send_history, peer,
                    static_cast<size_t>(peer_send_buf.h_send_history_count)
                        * sizeof(ContactHistoryPacked), stream);
            }
            if (peer_send_buf.h_send_wall_history_count > 0) {
                copy_peer_payload(
                    my_recv_buf.d_recv_wall_history, device_id_, peer_send_buf.d_send_wall_history, peer,
                    static_cast<size_t>(peer_send_buf.h_send_wall_history_count)
                        * sizeof(WallContactHistoryPacked), stream);
            }

            // Update our receive count from peer's send count
            my_recv_buf.h_recv_count = peer_send_buf.h_send_count;
            my_recv_buf.h_recv_history_count = peer_send_buf.h_send_history_count;
            my_recv_buf.h_recv_wall_history_count = peer_send_buf.h_send_wall_history_count;
        }
    }

}

void Communicator::unpack_migrating_particles(
    GPUDomainState& gpu_state,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Append received migrants to local particle array
    int32_t insert_pos = gpu_state.subdomain.num_local_particles;
    int32_t total_received = 0;
    int32_t capacity = gpu_state.subdomain.capacity;

    int block_size = 256;
    int32_t total_expected = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer != device_id_) total_expected += buffers_[peer].h_recv_count;
    }
    ensure_particle_capacity(gpu_state, insert_pos + total_expected, "migration receive");
    ensure_workspace_capacity(gpu_state.subdomain.capacity);
    capacity = gpu_state.subdomain.capacity;
    if (insert_pos + total_expected > capacity) {
        throw std::runtime_error(
            "Migrant overflow on GPU " + std::to_string(device_id_)
            + " after survivor compaction: need " + std::to_string(insert_pos + total_expected)
            + " slots but capacity is " + std::to_string(capacity));
    }

    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer == device_id_) continue;

        CommBuffer& buf = buffers_[peer];
        if (buf.h_recv_count <= 0) continue;

        int32_t count_to_unpack = buf.h_recv_count;

        int num_blocks = (count_to_unpack + block_size - 1) / block_size;

        // Unpack particles (append after current local particles)
        unpack_ghost_particles_kernel<<<num_blocks, block_size, 0, stream>>>(
            buf.d_recv_particles,
            count_to_unpack,
            insert_pos + total_received,
            gpu_state.local_state.particles
        );

        unpack_contact_history_active_kernel<<<num_blocks, block_size, 0, stream>>>(
            buf.d_recv_history, buf.d_recv_particles, count_to_unpack, insert_pos + total_received,
            gpu_state.local_state.contact_history.data,
            gpu_state.local_state.particles.contact_history_offset,
            gpu_state.local_state.particles.num_contacts,
            contact_history_stride_);
        unpack_wall_history_active_kernel<<<num_blocks, block_size, 0, stream>>>(
            buf.d_recv_wall_history, buf.d_recv_particles, count_to_unpack, insert_pos + total_received,
            gpu_state.local_state.wall_contact_history.data,
            gpu_state.local_state.particles.wall_history_offset,
            gpu_state.local_state.particles.num_wall_contacts,
            wall_history_stride_);

        total_received += count_to_unpack;
    }

    // Compaction has already completed. Publish the appended owned prefix
    // before halo unpack can resize the SoA and preserve its live rows.
    gpu_state.subdomain.num_local_particles += total_received;
    auto& particles = gpu_state.local_state.particles;
    particles.num_local = gpu_state.subdomain.num_local_particles;
    particles.num_ghost = 0;
    particles.num_particles = particles.num_local;
}

void Communicator::compact_local_particles(
    GPUDomainState& gpu_state,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    // NOTE: We only compact the ORIGINAL local particles (before migrants were appended).
    // The is_migrating array was only set for original particles [0, num_local_before_migration).
    // New migrants were appended at indices >= num_local_before_migration and should be kept.
    int32_t num_original = gpu_state.subdomain.num_local_before_migration;
    if (num_original <= 0) {
        // Nothing to compact, just return
        return;
    }

    // Unordered in-place tail fill replaces the former scan plus full-size
    // particle/history replicas.  The migration flag is read only for the
    // original local range; received particles are appended afterwards.
    {
    const int block_size = 256;
    const int blocks = (num_original + block_size - 1) / block_size;
    CUDA_CHECK(cudaMemsetAsync(d_migrate_count_, 0, sizeof(int32_t), stream));
    count_migrants_kernel<<<blocks, block_size, 0, stream>>>(gpu_state.d_is_migrating, num_original, d_migrate_count_);
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_scan_result_, d_migrate_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int32_t new_count = num_original - h_pinned_scan_result_[0];
    CUDA_CHECK(cudaMemsetAsync(d_migrate_count_, 0, sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_halo_count_, 0, sizeof(int32_t), stream));
    collect_tail_fill_pairs_kernel<<<blocks, block_size, 0, stream>>>(gpu_state.d_is_migrating, num_original, new_count,
        d_migrate_indices_, d_new_indices_, d_migrate_count_, d_halo_count_);
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_scan_result_, d_migrate_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_scan_result_ + 1, d_halo_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_pinned_scan_result_[0] != h_pinned_scan_result_[1])
        throw std::runtime_error("Tail-fill hole/source count mismatch on GPU " + std::to_string(device_id_));
    const int pairs = h_pinned_scan_result_[0];
    if (pairs) copy_tail_fill_pairs_kernel<<<(pairs + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_migrate_indices_, d_new_indices_, pairs,
        gpu_state.local_state.particles,
        gpu_state.local_state.contact_history.data, gpu_state.local_state.wall_contact_history.data, contact_history_stride_, wall_history_stride_);
    gpu_state.subdomain.num_local_particles = new_count;
    gpu_state.local_state.particles.num_particles = new_count;
    gpu_state.local_state.particles.num_local = new_count;
    gpu_state.local_state.particles.num_ghost = 0;
    gpu_state.subdomain.num_ghost_particles = 0;
    return;
    }

}
void Communicator::migrate_particles(
    std::vector<GPUDomainState>& all_gpu_states,
    std::vector<Communicator>& all_communicators,
    const DomainParamsGPU& params
) {
    GPUDomainState& my_state = all_gpu_states[device_id_];

    int32_t initial_local = my_state.subdomain.num_local_particles;

    // Step 1: Identify migrating particles
    identify_migrating_particles(my_state, params, my_state.compute_stream);

    // Step 2: Pack migrating particles with history
    pack_migrating_particles(my_state, my_state.compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(my_state.compute_stream));

    // Count particles being sent out
    int32_t total_sending = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer != device_id_) {
            total_sending += buffers_[peer].h_send_count;
        }
    }

    // Step 3: Transfer (requires barrier in caller) - now with correct P2P access
    transfer_migrants(all_communicators, my_state.comm_stream);

    // This convenience path has no phase coordinator to establish a
    // cross-stream dependency.  Finish the receiver-pull copies before the
    // compute stream compacts survivors and reads the appended history rows.
    CUDA_CHECK(cudaStreamSynchronize(my_state.comm_stream));

    // Count particles received
    int32_t total_received = 0;
    for (int32_t peer = 0; peer < num_gpus_; ++peer) {
        if (peer != device_id_) {
            total_received += buffers_[peer].h_recv_count;
        }
    }

    // Save the original local count BEFORE unpacking migrants
    // This is needed for compact_local_particles to know which indices had is_migrating set
    my_state.subdomain.num_local_before_migration = my_state.subdomain.num_local_particles;

    // Step 4: Compact survivors and their history rows before appending
    // migrants, so no moved row aliases a still-live particle on later steps.
    compact_local_particles(my_state, my_state.compute_stream);

    // Step 5: Append received particles and their fixed-stride history rows.
    unpack_migrating_particles(my_state, my_state.compute_stream);

    int32_t after_unpack = my_state.subdomain.num_local_particles;

    CUDA_CHECK(cudaStreamSynchronize(my_state.compute_stream));

    int32_t final_local = my_state.subdomain.num_local_particles;

    // Debug: Print particle count changes every 100 steps
    static int debug_counter = 0;
    if (debug_counter++ % 100 == 0) {
        fprintf(stderr, "GPU %d migration: initial=%d, sending=%d, received=%d, after_unpack=%d, final=%d\n",
                device_id_, initial_local, total_sending, total_received, after_unpack, final_local);
    }
}

// =============================================================================
// Cleanup
// =============================================================================

void Communicator::shutdown() {
    if (device_id_ < 0) return;

    CUDA_CHECK(cudaSetDevice(device_id_));

    // Free device-side pointer tables before releasing the peer buffers they reference.
    if (d_send_particle_buffers_) { cudaFree(d_send_particle_buffers_); d_send_particle_buffers_ = nullptr; }
    if (d_send_index_buffers_) { cudaFree(d_send_index_buffers_); d_send_index_buffers_ = nullptr; }
    send_particle_capacity_ = 0;

    // Free communication buffers
    for (auto& buf : buffers_) {
        if (buf.d_send_count) cudaFree(buf.d_send_count);
        if (buf.d_recv_count) cudaFree(buf.d_recv_count);
    }
    if (d_send_particle_arena_) cudaFree(d_send_particle_arena_);
    if (d_send_history_arena_) cudaFree(d_send_history_arena_);
    if (d_send_wall_history_arena_) cudaFree(d_send_wall_history_arena_);
    if (d_send_index_arena_) cudaFree(d_send_index_arena_);
    if (d_recv_particle_arena_) cudaFree(d_recv_particle_arena_);
    if (d_recv_history_arena_) cudaFree(d_recv_history_arena_);
    if (d_recv_wall_history_arena_) cudaFree(d_recv_wall_history_arena_);
    d_send_particle_arena_ = nullptr; d_send_history_arena_ = nullptr;
    d_send_wall_history_arena_ = nullptr; d_send_index_arena_ = nullptr;
    d_recv_particle_arena_ = nullptr; d_recv_history_arena_ = nullptr;
    d_recv_wall_history_arena_ = nullptr; send_arena_capacity_ = recv_arena_capacity_ = 0;
    buffers_.clear();

    // Free halo buffers
    if (d_halo_count_) { cudaFree(d_halo_count_); d_halo_count_ = nullptr; }
    if (d_halo_dest_mask_) { cudaFree(d_halo_dest_mask_); d_halo_dest_mask_ = nullptr; }

    // Free migration buffers
    if (d_migrate_indices_) { cudaFree(d_migrate_indices_); d_migrate_indices_ = nullptr; }
    if (d_migrate_count_) { cudaFree(d_migrate_count_); d_migrate_count_ = nullptr; }
    if (d_migrate_dest_gpu_) { cudaFree(d_migrate_dest_gpu_); d_migrate_dest_gpu_ = nullptr; }

    // Free compaction buffers
    if (d_keep_flags_) { cudaFree(d_keep_flags_); d_keep_flags_ = nullptr; }
    if (d_new_indices_) { cudaFree(d_new_indices_); d_new_indices_ = nullptr; }

    // Free ORB staging and migration-history accounting buffers.
    if (d_gpu_box_min_x_) { cudaFree(d_gpu_box_min_x_); d_gpu_box_min_x_ = nullptr; }
    if (d_gpu_box_min_y_) { cudaFree(d_gpu_box_min_y_); d_gpu_box_min_y_ = nullptr; }
    if (d_gpu_box_min_z_) { cudaFree(d_gpu_box_min_z_); d_gpu_box_min_z_ = nullptr; }
    if (d_gpu_box_max_x_) { cudaFree(d_gpu_box_max_x_); d_gpu_box_max_x_ = nullptr; }
    if (d_gpu_box_max_y_) { cudaFree(d_gpu_box_max_y_); d_gpu_box_max_y_ = nullptr; }
    if (d_gpu_box_max_z_) { cudaFree(d_gpu_box_max_z_); d_gpu_box_max_z_ = nullptr; }
    if (d_history_overflow_status_) { cudaFree(d_history_overflow_status_); d_history_overflow_status_ = nullptr; }
    if (d_history_payload_counters_) { cudaFree(d_history_payload_counters_); d_history_payload_counters_ = nullptr; }
    if (d_send_contact_history_counts_) { cudaFree(d_send_contact_history_counts_); d_send_contact_history_counts_ = nullptr; }
    if (d_send_wall_history_counts_) { cudaFree(d_send_wall_history_counts_); d_send_wall_history_counts_ = nullptr; }

    // Free CUB temp storage (Phase 5)
    if (d_cub_scan_temp_) { cudaFree(d_cub_scan_temp_); d_cub_scan_temp_ = nullptr; }

    // Free contiguous send counts (Phase 7)
    if (d_all_send_counts_) { cudaFree(d_all_send_counts_); d_all_send_counts_ = nullptr; }

    // Free pinned host memory (Phase 2)
    if (h_pinned_send_counts_) { cudaFreeHost(h_pinned_send_counts_); h_pinned_send_counts_ = nullptr; }
    if (h_pinned_scan_result_) { cudaFreeHost(h_pinned_scan_result_); h_pinned_scan_result_ = nullptr; }

    device_id_ = -1;
    std::cout << "Communicator shutdown complete" << std::endl;
}

} // namespace cudadem

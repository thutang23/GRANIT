#include "src/core/constant/load_or_resume.h"
#include <cmath>
#include <iostream>

using namespace cudadem;
namespace fs = std::filesystem;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> static T* upload(const std::vector<T>& values, std::vector<void*>& allocations) {
    T* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    allocations.push_back(pointer);
    return pointer;
}
template<class T> static std::vector<T> download(const T* pointer, size_t count) {
    std::vector<T> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        char directory[] = "/tmp/granit-checkpoint-payload-XXXXXX";
        require(::mkdtemp(directory), "Cannot create test directory");
        const std::string alias = (fs::path(directory) / "checkpoint_step_0000000007.bin").string();
        std::vector<void*> allocations;
        SimulationState source;
        source.current_step = 7; source.current_time = 0.125; source.dt = 0.001;
        auto& p = source.particles;
        p.num_particles = 3; p.num_local = 3;
        const std::vector<double> precise{1.0 + std::ldexp(1.0, -40), -2.0 + std::ldexp(1.0, -41), 3.0 + std::ldexp(1.0, -42)};
        const auto d = [&] { return upload(precise, allocations); };
        const auto f = [&] { return upload(std::vector<float>(3, 1.0f), allocations); };
        const auto i = [&] { return upload(std::vector<int32_t>(3, 0), allocations); };
        p.pos_x=d();p.pos_y=d();p.pos_z=d();p.vel_x=d();p.vel_y=d();p.vel_z=d();
        p.global_id=upload(std::vector<int32_t>{0,1,2},allocations);
        p.material_id=i();p.clump_id=i();p.lifecycle_state=upload(std::vector<uint8_t>(3,0),allocations);
        p.radius=f();p.mass=f();p.moment_of_inertia=f();
        p.omega_x=d();p.omega_y=d();p.omega_z=d();
        p.force_x=d();p.force_y=d();p.force_z=d();p.torque_x=d();p.torque_y=d();p.torque_z=d();
        p.gravity_force_x=d();p.gravity_force_y=d();p.gravity_force_z=d();
        p.quat_w=d();p.quat_x=d();p.quat_y=d();p.quat_z=d();
        p.clump_off_x=f();p.clump_off_y=f();p.clump_off_z=f();p.num_contacts=i();
        const std::vector<int32_t> offsets{4,0,2}, counts{1,2,1};
        p.wall_history_offset=upload(offsets,allocations);p.num_wall_contacts=upload(counts,allocations);
        std::vector<WallContactHistory> history(6);
        for (int row=0;row<3;++row) for (int slot=0;slot<counts[row];++slot) {
            auto& spring=history[offsets[row]+slot];
            spring.wall_id=row+10;spring.is_active=true;spring.is_slipping=slot==1;
            spring.delta_t_x=0.125f*(row+1);spring.delta_r_y=-0.25f*(slot+1);spring.delta_tw_z=0.0625f;
        }
        source.wall_contact_history.data=upload(history,allocations);
        source.wall_contact_history.capacity=6;source.wall_contact_history.max_contacts_per_particle=2;
        CheckpointTransaction tx(alias);
        SimConfig config;
        config.io.checkpoint_path[0] = '\0';
        std::strncpy(config.io.output_directory, directory, sizeof(config.io.output_directory) - 1);
        require(fs::path(generate_checkpoint_path(config, 7)).parent_path() == fs::path(directory) / "checkpoints",
                "Forced checkpoint without an explicit path did not use the output directory");
        require(write_unified_binary(tx.staging_path(),p,source,config,true),"Checkpoint write failed");
        write_wall_history_checkpoint(tx.staging_path(),source);
        tx.commit();
        const std::string payload=resolve_checkpoint_path(alias);
        SimulationState restored;
        restored.current_step=source.current_step;restored.current_time=source.current_time;
        restored.particles.num_particles=3;
        restored.particles.wall_history_offset=i();restored.particles.num_wall_contacts=i();
        read_wall_history_checkpoint(payload,restored);
        allocations.push_back(restored.wall_contact_history.data);
        require(download(restored.particles.wall_history_offset,3)==offsets,"Wall offsets changed");
        require(download(restored.particles.num_wall_contacts,3)==counts,"Wall counts changed");
        const auto actual=download(restored.wall_contact_history.data,6);
        for (size_t slot=0;slot<history.size();++slot) {
            const auto& a=actual[slot];const auto& b=history[slot];
            require(a.wall_id==b.wall_id && a.delta_t_x==b.delta_t_x && a.delta_r_y==b.delta_r_y &&
                a.delta_tw_z==b.delta_tw_z && a.is_active==b.is_active && a.is_slipping==b.is_slipping,
                "Wall spring state changed");
        }
        // Independently decode the actual disk descriptors: f32 roundoff would
        // erase every low bit deliberately present in these fixture values.
        std::ifstream in(payload,std::ios::binary);
        UnifiedHeader header{};in.read(reinterpret_cast<char*>(&header),sizeof(header));
        std::vector<FieldDescriptor> descriptors(header.num_descriptors);
        in.read(reinterpret_cast<char*>(descriptors.data()),descriptors.size()*sizeof(FieldDescriptor));
        const uint64_t tier2=sizeof(header)+descriptors.size()*sizeof(FieldDescriptor)+3*(6*sizeof(double)+sizeof(int32_t));
        int checked=0;
        for(const auto& field:descriptors) if(field.field_id==static_cast<uint16_t>(FieldID::FORCE)||
            field.field_id==static_cast<uint16_t>(FieldID::TORQUE)||field.field_id==static_cast<uint16_t>(FieldID::GRAVITY_ACCELERATION)) {
            require(field.dtype==static_cast<uint8_t>(DType::F64)&&field.count==3,"Integrator cache lost f64 precision");
            in.seekg(tier2+field.byte_offset);
            for(int component=0;component<3;++component) {
                std::vector<double> values(3);in.read(reinterpret_cast<char*>(values.data()),3*sizeof(double));
                require(values==precise,"Integrator cache changed on disk");
            }
            ++checked;
        }
        require(checked==3,"Integrator cache descriptors missing");
        for(void* pointer:allocations) CUDA_CHECK(cudaFree(pointer));
        fs::remove_all(directory);
        std::cout << "checkpoint GPU wall history and f64 payload tests passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}

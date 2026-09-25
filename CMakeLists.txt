cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Load toolchain-file settings before deciding whether GPU targets were supplied.
project(GRANIT VERSION 1.0.0 LANGUAGES CXX)
set(_granit_default_cuda_architectures OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(_granit_default_cuda_architectures ON)
endif()
# Let CUDA initialize with its supported compiler default, then choose our
# portable targets once the compiler version is known.
enable_language(CUDA)

if(_granit_default_cuda_architectures)
    set(_granit_cuda_architectures "75;80;86;87;89;90")
    # CUDA 13 removed Volta code generation; keep it in CUDA 12 builds.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
        list(PREPEND _granit_cuda_architectures 70)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_granit_cuda_architectures}" CACHE STRING
        "CUDA architectures" FORCE)
    unset(_granit_cuda_architectures)
endif()
unset(_granit_default_cuda_architectures)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Verify and stage the runtime coefficient asset for build and installed binaries.
set(GRANIT_G_TABLE "${CMAKE_CURRENT_SOURCE_DIR}/data/fmm/g_table_v1.bin")
file(SHA256 "${GRANIT_G_TABLE}" GRANIT_G_TABLE_SHA256)
if(NOT GRANIT_G_TABLE_SHA256 STREQUAL "bb1ae2ea6eb1ec8d52057e65dc11f24832c1bba3b72e2fc9b2a8804cc28e72d5")
    message(FATAL_ERROR "Bundled FMM G-table checksum mismatch")
endif()
configure_file("${GRANIT_G_TABLE}" "${CMAKE_BINARY_DIR}/share/granit/fmm/g_table_v1.bin" COPYONLY)
install(FILES "${GRANIT_G_TABLE}" DESTINATION share/granit/fmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -lineinfo")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(BUILD_TESTING "Build host configuration and coefficient-asset tests" OFF)
option(GRANIT_BUILD_MULTIGPU "Build the multi-GPU DEM/SPH executable" ON)
option(ENABLE_PROFILING "Enable production CUDA profiling markers" OFF)
option(SPH_DEBUG "Enable SPH startup diagnostics" OFF)
option(FMM_P2P_LOCAL_FP32 "Use FP32 arithmetic inside body-fitted P2P kernels with FP64 storage" ON)
set(FMM_M2L_MODE "TF32_TC" CACHE STRING "M2L precision: FP64, FP32_SIMT_LINEAR, FP32_SIMT_DUAL, TF32_TC")
set_property(CACHE FMM_M2L_MODE PROPERTY STRINGS FP64 FP32_SIMT_LINEAR FP32_SIMT_DUAL TF32_TC)
if(FMM_M2L_MODE STREQUAL "FP64")
    set(FMM_M2L_MODE_DEFINE 0)
elseif(FMM_M2L_MODE STREQUAL "FP32_SIMT_LINEAR")
    set(FMM_M2L_MODE_DEFINE 1)
elseif(FMM_M2L_MODE STREQUAL "FP32_SIMT_DUAL")
    set(FMM_M2L_MODE_DEFINE 2)
elseif(FMM_M2L_MODE STREQUAL "TF32_TC")
    set(FMM_M2L_MODE_DEFINE 3)
else()
    message(FATAL_ERROR "Invalid FMM_M2L_MODE='${FMM_M2L_MODE}'")
endif()
if(BUILD_CPU_MOCK OR CUDADEM_CPU_MOCK)
    message(FATAL_ERROR "This production package requires CUDA; CPU mock tooling is not included")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
include(FetchContent)
include(cmake/GranitOptix.cmake)

# Use the exact dependency revisions present in the validated source build.
# FetchContent source overrides support builds from an existing dependency cache.
FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG f7320141120f720aecc4c32be25586e7da9eb978
    GIT_SUBMODULES "")
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

set(GRANIT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(GRANIT_CORE_SOURCES
    src/core/constant/config_device.cu
    src/core/constant/config_host.cpp
    src/core/material/material_database.cpp
    src/domain/rcb_work_model.cpp
    src/core/constant/system_initialize_host.cpp
    src/core/constant/binary_reader.cpp
    src/core/particle_model/sphere_dynamic.cu
    src/core/particle_model/clump.cu
    src/core/particle_model/clump_init.cpp
    src/core/shear/shear_box.cu
    src/core/shear/shear_phase.cpp
    src/core/neighbor/cll/cll_kernels.cu
    src/core/neighbor/lbvh/lbvh_kernels.cu
    src/core/neighbor/direct/direct_kernels.cu
    src/core/wall/plain_wall.cu
    src/core/wall/periodical_boundary.cu
    src/core/wall/wall_manager.cu
    src/core/wall/trimesh_wall.cu
    src/core/wall/trimesh_loader.cpp
    src/core/wall/bonded_rect_wall.cu
    src/core/wall/funnel_wall.cu
    src/core/wall/sphere_wall.cu
    src/core/acceleration/self_gravity.cu
    src/core/acceleration/external_bodies.cu
    src/core/acceleration/body_fitted_fmm/body_fitted_fmm_manager.cu
    src/core/acceleration/inertial_acceleration.cu
    src/core/acceleration/tidal_acceleration.cu
    src/core/acceleration/acceleration_field_texture.cu
    src/core/sph/sph_physics.cu
    src/core/sph/sph_strength.cu
    src/core/sph/sph_damage.cu
    src/core/sph/sph_porosity.cu
    src/core/sph/sph_boundary.cu
    src/core/sph/sph_history.cu
    src/core/sph/sph_energy.cu
    src/core/sph/sph_input.cpp
    src/core/sph/sph_timestep.cu
    src/core/sph/sph_config_printer.cpp
)
set(GRANIT_DOMAIN_SOURCES
    src/domain/domain_manager.cu
    src/core/shear/shear_phase.cpp
    src/domain/communicator.cu
    src/domain/packer_kernel.cu
    src/domain/let_communicator.cu
    src/domain/let_kernels_fmm.cu
    src/core/material/material_database.cpp
)
set(GRANIT_MULTIGPU_SOURCES
    src/domain/rcb_kernels.cu
    src/domain/rcb_balancer.cu
    src/domain/distributed_fmm_depth_controller.cpp
)

# Apply the same precision settings to all production solvers.
function(granit_solver target entrypoint)
    add_executable(${target} ${entrypoint} ${GRANIT_CORE_SOURCES})
    target_include_directories(${target} PRIVATE ${GRANIT_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_compile_definitions(${target} PRIVATE FMM_M2L_MODE=${FMM_M2L_MODE_DEFINE})
    if(FMM_P2P_LOCAL_FP32)
        target_compile_definitions(${target} PRIVATE FMM_P2P_LOCAL_FP32=1)
    endif()
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cuda_driver
        yaml-cpp OpenMP::OpenMP_CXX)
    if(ENABLE_PROFILING)
        target_compile_definitions(${target} PRIVATE CUDADEM_ENABLE_PROFILING)
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    granit_target_optix(${target})
endfunction()

granit_solver(granit_dem main.cu)
granit_solver(granit_sph main_sph.cu)
target_compile_definitions(granit_dem PRIVATE CUDADEM_VERSION="${PROJECT_VERSION}")
if(SPH_DEBUG)
    target_compile_definitions(granit_sph PRIVATE SPH_DEBUG)
endif()

set(GRANIT_INSTALL_TARGETS granit_dem granit_sph)
if(GRANIT_BUILD_MULTIGPU)
    find_package(MPI REQUIRED)
    find_path(NVML_INCLUDE_DIR nvml.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    find_library(NVML_LIBRARY nvidia-ml)
    add_library(granit_domain ${GRANIT_DOMAIN_SOURCES})
    target_include_directories(granit_domain PRIVATE ${GRANIT_INCLUDE_DIRS})
    target_link_libraries(granit_domain PRIVATE CUDA::cudart yaml-cpp MPI::MPI_CXX)
    if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
        target_include_directories(granit_domain PRIVATE ${NVML_INCLUDE_DIR})
        target_compile_definitions(granit_domain PRIVATE CUDADEM_HAS_NVML=1)
        target_link_libraries(granit_domain PRIVATE ${NVML_LIBRARY})
    endif()
    granit_solver(granit_multigpu main_multigpu.cu)
    target_sources(granit_multigpu PRIVATE ${GRANIT_MULTIGPU_SOURCES})
    target_compile_definitions(granit_multigpu PRIVATE CUDADEM_HAVE_LET)
    target_link_libraries(granit_multigpu PRIVATE granit_domain)
    list(APPEND GRANIT_INSTALL_TARGETS granit_multigpu)
endif()
install(TARGETS ${GRANIT_INSTALL_TARGETS} RUNTIME DESTINATION bin)
install(DIRECTORY components/config/ DESTINATION share/granit/config FILES_MATCHING PATTERN "*.yaml")
message(STATUS "GRANIT production package: M2L=${FMM_M2L_MODE}, local P2P FP32=${FMM_P2P_LOCAL_FP32}")

# Host-only release regressions use the production parser and need no GPU.
include(CTest)
if(BUILD_TESTING)
    add_executable(granit_config_defaults_test
        components/tests/release_defaults/config_defaults.cpp
        src/core/constant/config_host.cpp
        src/core/constant/config_device.cu
        src/core/material/material_database.cpp
        src/domain/rcb_work_model.cpp)
    target_include_directories(granit_config_defaults_test PRIVATE ${GRANIT_INCLUDE_DIRS})
    target_link_libraries(granit_config_defaults_test PRIVATE CUDA::cudart yaml-cpp)
    if(FMM_P2P_LOCAL_FP32)
        target_compile_definitions(granit_config_defaults_test PRIVATE FMM_P2P_LOCAL_FP32=1)
    endif()
    add_test(NAME config_defaults COMMAND granit_config_defaults_test "${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(granit_sph_respa_test components/tests/release_defaults/sph_respa.cpp
        src/domain/distributed_fmm_depth_controller.cpp src/domain/rcb_work_model.cpp)
    target_include_directories(granit_sph_respa_test PRIVATE ${GRANIT_INCLUDE_DIRS})
    target_link_libraries(granit_sph_respa_test PRIVATE CUDA::cudart yaml-cpp)
    add_test(NAME sph_respa COMMAND granit_sph_respa_test)
    add_executable(granit_peer_transport_policy_test
        components/tests/release_defaults/peer_transport_policy.cpp)
    target_include_directories(granit_peer_transport_policy_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME peer_transport_policy COMMAND granit_peer_transport_policy_test)
    # Opt-in GPU payload verification; ordinary CTest remains host-only.
    add_executable(granit_peer_transfer_gpu_test
        components/tests/release_defaults/peer_transfer_gpu.cu)
    target_include_directories(granit_peer_transfer_gpu_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(granit_peer_transfer_gpu_test PRIVATE CUDA::cudart)
    add_executable(granit_g_table_test components/tests/release_defaults/g_table.cpp)
    target_include_directories(granit_g_table_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME g_table COMMAND granit_g_table_test)
endif()

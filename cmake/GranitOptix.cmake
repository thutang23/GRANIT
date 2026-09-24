include_guard(GLOBAL)

option(GRANIT_ENABLE_OPTIX "Build the optional OptiX RT-core neighbor backend" OFF)
set(OPTIX_INCLUDE_DIR "" CACHE PATH "Directory containing local OptiX SDK headers; empty fetches pinned NVIDIA headers")

# Attach the host backend even in CUDA-only builds, where it supplies the
# unavailable-backend stub without including or loading any OptiX code.
function(granit_target_optix target)
    get_filename_component(_granit_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    target_sources(${target} PRIVATE "${_granit_root}/src/core/neighbor/rt/rt_backend.cu")
    if(NOT GRANIT_ENABLE_OPTIX)
        return()
    endif()

    get_property(_granit_optix_ready GLOBAL PROPERTY GRANIT_OPTIX_READY)
    if(NOT _granit_optix_ready)
        if(OPTIX_INCLUDE_DIR)
            get_filename_component(_optix_include "${OPTIX_INCLUDE_DIR}" ABSOLUTE)
        else()
            include(FetchContent)
            FetchContent_Declare(granit_optix_headers
                GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
                # NVIDIA optix-dev v8.0.0; requires an R535 or newer driver.
                GIT_TAG f60c1e44f18426f426a2ed948f28515b3cf67b8a
                GIT_SUBMODULES ""
            )
            FetchContent_MakeAvailable(granit_optix_headers)
            set(_optix_include "${granit_optix_headers_SOURCE_DIR}/include")
        endif()
        foreach(_header optix.h optix_stubs.h optix_function_table_definition.h)
            if(NOT EXISTS "${_optix_include}/${_header}")
                message(FATAL_ERROR
                    "Missing ${_header} in OPTIX_INCLUDE_DIR='${_optix_include}'. "
                    "Provide OptiX headers or configure GRANIT_ENABLE_OPTIX=OFF.")
            endif()
        endforeach()

        set(_generated_dir "${CMAKE_BINARY_DIR}/granit-optix")
        file(MAKE_DIRECTORY "${_generated_dir}")
        set(_ptx "${_generated_dir}/mochi.ptx")
        set(_ptx_header "${_generated_dir}/mochi_ptx.h")
        set(_host_compiler_args)
        if(CMAKE_CUDA_HOST_COMPILER)
            list(APPEND _host_compiler_args --compiler-bindir "${CMAKE_CUDA_HOST_COMPILER}")
        endif()
        set(_dependency_args)
        set(_depfile_args)
        # CMake 3.18 supports custom-command depfiles with Ninja; Makefile
        # generators gained support in 3.20. Keep explicit headers below for
        # older Makefile builds as well.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20" OR CMAKE_GENERATOR MATCHES "Ninja")
            list(APPEND _dependency_args --generate-dependencies-with-compile
                --dependency-output "${_ptx}.d" --dependency-target-name "${_ptx_header}")
            list(APPEND _depfile_args DEPFILE "${_ptx}.d")
        endif()
        file(GLOB _rt_headers CONFIGURE_DEPENDS
            "${_granit_root}/src/core/neighbor/rt/*.h"
            "${_granit_root}/src/core/neighbor/rt/*.cuh")
        set(_optix_compute_arch 70)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            set(_optix_compute_arch 75)
        endif()
        add_custom_command(
            OUTPUT "${_ptx_header}"
            BYPRODUCTS "${_ptx}"
            COMMAND "${CMAKE_CUDA_COMPILER}" --ptx --std=c++17
                --gpu-architecture=compute_${_optix_compute_arch} --fmad=true
                "$<$<CONFIG:Release>:--use_fast_math>"
                --expt-relaxed-constexpr --extended-lambda
                ${_host_compiler_args} ${_dependency_args}
                "-I${_optix_include}"
                "-I${_granit_root}" "-I${_granit_root}/src" "-I${_granit_root}/src/core"
                "$<$<BOOL:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>:-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>>"
                "${_granit_root}/src/core/neighbor/rt/rt_device.cu" -o "${_ptx}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${_ptx}" "-DOUTPUT=${_ptx_header}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EmbedOptixPtx.cmake"
            DEPENDS
                "${_granit_root}/src/core/neighbor/sph_pair.cuh"
                "${_granit_root}/src/core/neighbor/rt/rt_device.cu"
                ${_rt_headers}
                "${_granit_root}/src/core/neighbor/neighbor_types.h"
                "${_granit_root}/src/core/constant/command_line_interface.h"
                "${_granit_root}/src/core/material/material_database.h"
                "${_granit_root}/src/domain/rcb_work_model.h"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GranitOptix.cmake"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EmbedOptixPtx.cmake"
            ${_depfile_args}
            COMMENT "Compiling and embedding the Mochi OptiX neighbor programs"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(granit_optix_ptx DEPENDS "${_ptx_header}")
        set_property(GLOBAL PROPERTY GRANIT_OPTIX_READY TRUE)
        set_property(GLOBAL PROPERTY GRANIT_OPTIX_HEADERS "${_optix_include}")
        set_property(GLOBAL PROPERTY GRANIT_OPTIX_GENERATED_DIR "${_generated_dir}")
        message(STATUS "OptiX RT neighbor backend enabled: ${_optix_include}")
    endif()

    get_property(_optix_include GLOBAL PROPERTY GRANIT_OPTIX_HEADERS)
    get_property(_generated_dir GLOBAL PROPERTY GRANIT_OPTIX_GENERATED_DIR)
    add_dependencies(${target} granit_optix_ptx)
    target_compile_definitions(${target} PRIVATE GRANIT_ENABLE_OPTIX=1)
    target_include_directories(${target} SYSTEM PRIVATE "${_optix_include}")
    target_include_directories(${target} PRIVATE "${_generated_dir}")
    target_link_libraries(${target} PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
endfunction()

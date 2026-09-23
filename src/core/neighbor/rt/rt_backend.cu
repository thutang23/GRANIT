#include "rt_backend.h"
#include "../neighbor_types.h"
#include <stdexcept>

#if defined(GRANIT_ENABLE_OPTIX) && GRANIT_ENABLE_OPTIX
#include "rt_launch_params.h"
#include "mochi_ptx.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace cudadem { namespace rt {
namespace {
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("RT CUDA: ") + cudaGetErrorString(result));
}
void check_optix(OptixResult result, const char* operation, const char* log = "") {
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string("RT ") + operation + ": " +
                                 optixGetErrorName(result) + " " + log);
}

__global__ void proxy_bounds(SearchInput in, OptixAabb* bounds, int* invalid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in.count) return;
    const float x = float(in.pos_x[i]), y = float(in.pos_y[i]), z = float(in.pos_z[i]);
    // Match LBVH's rounded absolute coordinates. Expand the doubled support
    // conservatively for roundoff in its float distance/cutoff arithmetic,
    // then round every bound outward. Extra candidates are filtered later.
    double extent;
    float support;
    if (in.smoothing_length) {
        const float h = in.smoothing_length[i];
        // Radius-first ownership guarantees h_query <= h. Evaluate the largest
        // possible pair cutoff in exactly SPH's arithmetic order, making this
        // tight proxy conservative even when h+h rounds before multiplication.
        support = in.kernel_cutoff_factor * (0.5f * (h + h));
        extent = (double(support) + double(in.skin)) *
                 (1.0 + 8.0 * double(FLT_EPSILON));
        // SPH retains near-coincident pairs. With CUDA fast math, squared
        // distances below FLT_MIN can flush to zero, so cover that absolute
        // underflow region as well as ordinary relative roundoff.
        extent += 2.0 * sqrt(double(FLT_MIN));
        if (!(h >= 0.0f)) support = -1.0f;
    } else {
        support = in.support_extent[i];
        extent = (2.0 * double(in.support_extent[i]) + double(in.skin)) *
                 (1.0 + 8.0 * double(FLT_EPSILON));
    }
    OptixAabb b;
    b.minX = __double2float_rd(double(x) - extent);
    b.minY = __double2float_rd(double(y) - extent);
    b.minZ = __double2float_rd(double(z) - extent);
    b.maxX = __double2float_ru(double(x) + extent);
    b.maxY = __double2float_ru(double(y) + extent);
    b.maxZ = __double2float_ru(double(z) + extent);
    if (!(support >= 0.0f) || !isfinite(extent) ||
        !isfinite(b.minX) || !isfinite(b.minY) || !isfinite(b.minZ) ||
        !isfinite(b.maxX) || !isfinite(b.maxY) || !isfinite(b.maxZ)) {
        atomicExch(invalid, 1);
        b = {};
    }
    bounds[i] = b;
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void reserve(CUdeviceptr& ptr, size_t& capacity, size_t required) {
    if (required <= capacity) return;
    void* next = nullptr;
    check_cuda(cudaMalloc(&next, required));
    if (ptr) {
        const cudaError_t result = cudaFree(reinterpret_cast<void*>(ptr));
        if (result != cudaSuccess) {
            cudaFree(next);
            check_cuda(result);
        }
    }
    ptr = reinterpret_cast<CUdeviceptr>(next);
    capacity = required;
}
}

struct Backend::Impl {
    int device = -1;
    cudaStream_t stream = nullptr;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygen = nullptr, miss = nullptr, hit = nullptr;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr records = 0, launch_params = 0, aabbs = 0, scratch = 0, gas = 0;
    size_t records_size = 0, params_size = 0, aabbs_size = 0, scratch_size = 0, gas_size = 0;
    unsigned long long* count = nullptr;
    int* invalid = nullptr;
    LaunchParams params = {};
    bool built = false;
    int built_count = -1;

    ~Impl() {
        int previous = -1;
        cudaGetDevice(&previous);
        if (device >= 0) cudaSetDevice(device);
        // Shutdown may follow an explicit build without a subsequent query.
        cudaStreamSynchronize(stream);
        if (pipeline) optixPipelineDestroy(pipeline);
        if (hit) optixProgramGroupDestroy(hit);
        if (miss) optixProgramGroupDestroy(miss);
        if (raygen) optixProgramGroupDestroy(raygen);
        if (module) optixModuleDestroy(module);
        if (context) optixDeviceContextDestroy(context);
        for (CUdeviceptr p : {records, launch_params, aabbs, scratch, gas})
            if (p) cudaFree(reinterpret_cast<void*>(p));
        if (count) cudaFree(count);
        if (invalid) cudaFree(invalid);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
};

Backend* Backend::create(std::string& unavailable_reason) {
    static std::once_flag init_once;
    static OptixResult init_result = OPTIX_ERROR_UNKNOWN;
    std::call_once(init_once, [] { init_result = optixInit(); });
    if (init_result != OPTIX_SUCCESS) {
        // Error-name functions are themselves driver entry points. They are
        // unavailable if loading the function table failed.
        unavailable_reason = "OptiX driver unavailable (initialization code " +
                             std::to_string(int(init_result)) + ")";
        return nullptr;
    }
    check_cuda(cudaFree(nullptr));
    std::unique_ptr<Impl> p(new Impl);
    check_cuda(cudaGetDevice(&p->device));
    CUcontext cuda_context = nullptr;
    if (cuCtxGetCurrent(&cuda_context) != CUDA_SUCCESS || !cuda_context)
        throw std::runtime_error("RT cannot obtain the active CUDA context");
    OptixDeviceContextOptions context_options = {};
    const OptixResult context_result = optixDeviceContextCreate(cuda_context, &context_options, &p->context);
    if (context_result == OPTIX_ERROR_NOT_SUPPORTED) {
        unavailable_reason = "device has no supported OptiX RT device";
        return nullptr;
    }
    check_optix(context_result, "context creation");
    unsigned int rt_version = 0;
    check_optix(optixDeviceContextGetProperty(p->context, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                    &rt_version, sizeof(rt_version)), "RT-core capability query");
    if (!rt_version) {
        unavailable_reason = "device has no RT cores";
        return nullptr;
    }

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    char log[8192] = {};
    size_t log_size = sizeof(log);
    const OptixResult module_result = optixModuleCreate(p->context, &module_options, &pipeline_options,
                mochi_ptx, sizeof(mochi_ptx) - 1, log, &log_size, &p->module);
    check_optix(module_result, "module compilation", log);

    OptixProgramGroupOptions group_options = {};
    auto create_group = [&](OptixProgramGroupDesc& desc, OptixProgramGroup& group) {
        log_size = sizeof(log); log[0] = '\0';
        const OptixResult result = optixProgramGroupCreate(p->context, &desc, 1, &group_options,
                                                           log, &log_size, &group);
        check_optix(result, "program group creation", log);
    };
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = p->module;
    desc.raygen.entryFunctionName = "__raygen__neighbors";
    create_group(desc, p->raygen);
    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = p->module;
    desc.miss.entryFunctionName = "__miss__neighbors";
    create_group(desc, p->miss);
    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleIS = p->module;
    desc.hitgroup.entryFunctionNameIS = "__intersection__neighbors";
    desc.hitgroup.moduleAH = p->module;
    desc.hitgroup.entryFunctionNameAH = "__anyhit__neighbors";
    create_group(desc, p->hit);

    OptixProgramGroup groups[] = {p->raygen, p->miss, p->hit};
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log); log[0] = '\0';
    const OptixResult pipeline_result = optixPipelineCreate(p->context, &pipeline_options,
                    &link_options, groups, 3, log, &log_size, &p->pipeline);
    check_optix(pipeline_result, "pipeline creation", log);
    OptixStackSizes stack = {};
    for (auto group : groups)
        check_optix(optixUtilAccumulateStackSizes(group, &stack, p->pipeline), "stack sizing");
    unsigned int dc_traversal = 0, dc_state = 0, continuation = 0;
    check_optix(optixUtilComputeStackSizes(&stack, 1, 0, 0, &dc_traversal, &dc_state, &continuation),
                "stack computation");
    check_optix(optixPipelineSetStackSize(p->pipeline, dc_traversal, dc_state, continuation, 1),
                "stack setup");
    Record host_records[3] = {};
    for (int i = 0; i < 3; ++i)
        check_optix(optixSbtRecordPackHeader(groups[i], &host_records[i]), "SBT header");
    reserve(p->records, p->records_size, sizeof(host_records));
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(p->records), host_records, sizeof(host_records),
                           cudaMemcpyHostToDevice));
    p->sbt.raygenRecord = p->records;
    p->sbt.missRecordBase = p->records + sizeof(Record);
    p->sbt.missRecordStrideInBytes = sizeof(Record);
    p->sbt.missRecordCount = 1;
    p->sbt.hitgroupRecordBase = p->records + 2 * sizeof(Record);
    p->sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    p->sbt.hitgroupRecordCount = 1;
    reserve(p->launch_params, p->params_size, sizeof(LaunchParams));
    check_cuda(cudaMalloc(&p->count, sizeof(unsigned long long)));
    check_cuda(cudaMalloc(&p->invalid, sizeof(int)));
    unavailable_reason.clear();
    return new Backend(p.release());
}

Backend::~Backend() { delete impl_; }

void Backend::update(const SearchInput& input, bool rebuild, cudaStream_t stream) {
    auto& p = *impl_;
    p.stream = stream;
    if (input.count < 1 || !std::isfinite(input.skin) || input.skin < 0.0f)
        throw std::invalid_argument("RT requires positive particle count and finite nonnegative skin");
    if (input.smoothing_length &&
        (!(input.kernel_cutoff_factor > 0.0f) || !std::isfinite(input.kernel_cutoff_factor)))
        throw std::invalid_argument("RT SPH requires a finite positive kernel cutoff factor");
    p.params.input = input;
    reserve(p.aabbs, p.aabbs_size, size_t(input.count) * sizeof(OptixAabb));
    check_cuda(cudaMemsetAsync(p.invalid, 0, sizeof(int), stream));
    proxy_bounds<<<(input.count + 255) / 256, 256, 0, stream>>>(
        input, reinterpret_cast<OptixAabb*>(p.aabbs), p.invalid);
    check_cuda(cudaGetLastError());
    unsigned int flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &p.aabbs;
    build_input.customPrimitiveArray.numPrimitives = input.count;
    build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    build_input.customPrimitiveArray.flags = &flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    const bool full = rebuild || !p.built || p.built_count != input.count;
    options.operation = full ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;
    OptixAccelBufferSizes sizes = {};
    check_optix(optixAccelComputeMemoryUsage(p.context, &options, &build_input, 1, &sizes),
                "acceleration memory query");
    reserve(p.scratch, p.scratch_size, std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
    reserve(p.gas, p.gas_size, sizes.outputSizeInBytes);
    OptixTraversableHandle handle = p.params.traversable;
    check_optix(optixAccelBuild(p.context, stream, &options, &build_input, 1,
                p.scratch, p.scratch_size, p.gas, p.gas_size, &handle, nullptr, 0),
                full ? "GAS build" : "GAS update");
    p.params.traversable = handle;
    p.built = true;
    p.built_count = input.count;
}

void Backend::find_contacts(ContactListBuffer& contacts, cudaStream_t stream) {
    auto& p = *impl_;
    p.stream = stream;
    if (!p.built) throw std::logic_error("RT query before acceleration structure build");
    for (;;) {
        p.params.pairs = contacts.pairs;
        p.params.capacity = contacts.max_contacts;
        p.params.count = p.count;
        check_cuda(cudaMemsetAsync(p.count, 0, sizeof(unsigned long long), stream));
        check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(p.launch_params), &p.params,
                                   sizeof(p.params), cudaMemcpyHostToDevice, stream));
        check_optix(optixLaunch(p.pipeline, stream, p.launch_params, sizeof(LaunchParams),
                    &p.sbt, p.params.input.count, 1, 1), "neighbor launch");
        unsigned long long count = 0;
        int invalid = 0;
        check_cuda(cudaMemcpyAsync(&count, p.count, sizeof(count), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaMemcpyAsync(&invalid, p.invalid, sizeof(invalid), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
        if (invalid) throw std::runtime_error("RT proxy bounds require finite float coordinates and nonnegative finite supports");
        if (count > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max()))
            throw std::overflow_error("RT contact count exceeds int32 capacity");
        if (count > static_cast<unsigned long long>(contacts.max_contacts)) {
            const int64_t desired = std::max<int64_t>(count, 2LL * contacts.max_contacts);
            const int32_t capacity = static_cast<int32_t>(std::min<int64_t>(
                desired, std::numeric_limits<int32_t>::max()));
            ContactPair* next = nullptr;
            check_cuda(cudaMalloc(&next, size_t(capacity) * sizeof(ContactPair)));
            const cudaError_t result = cudaFree(contacts.pairs);
            if (result != cudaSuccess) { cudaFree(next); check_cuda(result); }
            contacts.pairs = next;
            contacts.max_contacts = capacity;
            continue; // No truncated list can reach force evaluation or cache.
        }
        contacts.h_num_contacts = static_cast<int32_t>(count);
        check_cuda(cudaMemcpyAsync(contacts.num_contacts, &contacts.h_num_contacts,
                                   sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        return;
    }
}

} }
#else
namespace cudadem { namespace rt {
Backend* Backend::create(std::string& reason) {
    reason = "OptiX support disabled at build time";
    return nullptr;
}
Backend::~Backend() = default;
void Backend::update(const SearchInput&, bool, cudaStream_t) {
    throw std::runtime_error("OptiX support disabled at build time");
}
void Backend::find_contacts(ContactListBuffer&, cudaStream_t) {
    throw std::runtime_error("OptiX support disabled at build time");
}
} }
#endif

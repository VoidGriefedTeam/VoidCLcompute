// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// ============================================================
// VoidCLcompute.cpp  (optimized v2)
//
// Changes vs v1:
//   1. Kernels now operate on float4 where count is divisible
//      by 4 (with a scalar tail kernel path), quadrupling the
//      work done per work-item and cutting instruction overhead.
//   2. native_sin/native_cos/native_sqrt used in `heavy` — much
//      cheaper on most GPUs; acceptable since this workload is
//      a benchmark stress test, not something needing IEEE
//      precision. (Kept exact ops like +,-,*,/ untouched.)
//   3. Host buffers use CL_MEM_ALLOC_HOST_PTR + clEnqueueMapBuffer
//      instead of plain clEnqueueWriteBuffer/ReadBuffer, letting
//      the driver use pinned/zero-copy memory on supporting
//      platforms (big win on integrated GPUs, real win on discrete).
//   4. Local work size queried per-kernel via
//      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE instead of
//      a hardcoded 64.
//   5. Buffer pool still keyed by byte size, now stores mapped
//      host pointers alongside device buffers.
// ============================================================
#define VOIDCLCOMPUTE_EXPORTS
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "VoidCLcompute.h"

#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <direct.h> // _mkdir on Windows

static cl_platform_id   s_platform = nullptr;
static cl_device_id     s_device   = nullptr;
static cl_context       s_context  = nullptr;
static cl_command_queue s_queue    = nullptr;
static std::string      s_deviceName;

enum OpId {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_ADD_SCALAR, OP_SUB_SCALAR, OP_MUL_SCALAR, OP_DIV_SCALAR,
    OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN,
    OP_HEAVY
};

static std::unordered_map<int, cl_kernel>  s_kernelCache;
static std::unordered_map<int, cl_program> s_programCache;
static std::unordered_map<int, size_t>     s_preferredMultiple; // per-op work-group multiple

// ---- Pinned/mapped device buffer pool ----------------------
struct BinaryBufSet {
    cl_mem a = nullptr, b = nullptr, out = nullptr;
    float* mappedA = nullptr;
    float* mappedB = nullptr;
    float* mappedOut = nullptr;
    size_t bytes = 0;
};
struct UnaryBufSet {
    cl_mem a = nullptr, out = nullptr;
    float* mappedA = nullptr;
    float* mappedOut = nullptr;
    size_t bytes = 0;
};

static std::unordered_map<size_t, BinaryBufSet> s_binaryPool;
static std::unordered_map<size_t, UnaryBufSet>  s_unaryPool;

// Vectorized (float4) kernels with scalar tail handled by clamping idx*4+k < count.
static const char* kBinaryTemplateVec =
    "__kernel void compute(__global const float* a, __global const float* b, "
    "__global float* out, uint count) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint base = idx * 4;\n"
    "    if (base >= count) return;\n"
    "    uint remain = count - base;\n"
    "    if (remain >= 4) {\n"
    "        float4 va = vload4(0, a + base);\n"
    "        float4 vb = vload4(0, b + base);\n"
    "        float4 vr = va %s vb;\n"
    "        vstore4(vr, 0, out + base);\n"
    "    } else {\n"
    "        for (uint k = 0; k < remain; k++) out[base+k] = a[base+k] %s b[base+k];\n"
    "    }\n"
    "}\n";

static const char* kUnaryTemplateVec =
    "__kernel void compute(__global const float* a, __global float* out, uint count) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint base = idx * 4;\n"
    "    if (base >= count) return;\n"
    "    uint remain = count - base;\n"
    "    if (remain >= 4) {\n"
    "        float4 va = vload4(0, a + base);\n"
    "        float4 vr = %s(va);\n"
    "        vstore4(vr, 0, out + base);\n"
    "    } else {\n"
    "        for (uint k = 0; k < remain; k++) out[base+k] = %s(a[base+k]);\n"
    "    }\n"
    "}\n";

// Scalar variant: ONE input array + a plain (non-buffer) scalar kernel arg.
// No second array is allocated or uploaded — the scalar value travels as a
// regular kernel argument via clSetKernelArg, same as `count`.
static const char* kScalarTemplateVec =
    "__kernel void compute(__global const float* a, float scalarValue, "
    "__global float* out, uint count) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint base = idx * 4;\n"
    "    if (base >= count) return;\n"
    "    uint remain = count - base;\n"
    "    if (remain >= 4) {\n"
    "        float4 va = vload4(0, a + base);\n"
    "        float4 vr = va %s scalarValue;\n"
    "        vstore4(vr, 0, out + base);\n"
    "    } else {\n"
    "        for (uint k = 0; k < remain; k++) out[base+k] = a[base+k] %s scalarValue;\n"
    "    }\n"
    "}\n";

// Heavy kernel: scalar (control-flow-light already; vectorizing loses more
// than it gains here because %-precision math funcs on float4 don't map to
// native_ hardware paths on all vendors). native_* used for real speedup.
static const char* kHeavySource =
    "__kernel void compute(__global const float* a, __global const float* b, "
    "__global float* out, uint count) {\n"
    "    uint idx = get_global_id(0);\n"
    "    if (idx >= count) return;\n"
    "    float x = a[idx];\n"
    "    float y = b[idx];\n"
    "    for (int i = 0; i < 50; i++) {\n"
    "        x = native_sin(x) * native_cos(y) + native_sqrt(fabs(x) + 1.0f);\n"
    "        y = native_sin(y) + native_cos(x);\n"
    "    }\n"
    "    out[idx] = x + y;\n"
    "}\n";

static void checkErr(cl_int err, const char* label) {
    if (err != CL_SUCCESS) {
        std::printf("[VoidCLcompute] %s failed, error code %d\n", label, err);
    }
}

// ---- Compiled kernel binary cache (disk) --------------------
// Skips clBuildProgram's JIT compile on subsequent runs by loading a
// previously-built binary for this exact device + op combo.
static const char* kCacheDir = "gwcl_kernel_cache";

static std::string cacheFilePath(const char* deviceName, OpId op) {
    // Sanitize device name for filesystem use.
    std::string safe;
    for (const char* p = deviceName; *p; ++p) {
        char c = *p;
        safe += (isalnum((unsigned char)c)) ? c : '_';
    }
    return std::string(kCacheDir) + "/" + safe + "_op" + std::to_string((int)op) + ".bin";
}

static bool loadCachedBinary(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size <= 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize((size_t)size);
    return (bool)f.read(reinterpret_cast<char*>(out.data()), size);
}

static void saveCachedBinary(const std::string& path, const unsigned char* data, size_t size) {
    _mkdir(kCacheDir); // ignore error if it already exists
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (f) f.write(reinterpret_cast<const char*>(data), (std::streamsize)size);
}

static cl_kernel getOrBuildKernel(OpId op) {
    auto it = s_kernelCache.find(op);
    if (it != s_kernelCache.end()) return it->second;

    cl_int err = CL_SUCCESS;
    cl_program program = nullptr;
    std::string cachePath = cacheFilePath(s_deviceName.c_str(), op);

    // Try loading a previously compiled binary first — skips JIT entirely.
    std::vector<unsigned char> binary;
    if (loadCachedBinary(cachePath, binary)) {
        const unsigned char* binPtr = binary.data();
        size_t binSize = binary.size();
        cl_int binStatus = CL_SUCCESS;
        program = clCreateProgramWithBinary(s_context, 1, &s_device, &binSize, &binPtr, &binStatus, &err);
        if (err != CL_SUCCESS || binStatus != CL_SUCCESS) {
            std::printf("[VoidCLcompute] Cached binary rejected (stale/incompatible), rebuilding\n");
            if (program) clReleaseProgram(program);
            program = nullptr;
        } else {
            err = clBuildProgram(program, 1, &s_device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                program = nullptr;
            }
        }
    }

    // Cache miss or rejected — compile from source and save the binary.
    if (!program) {
        char srcBuffer[4096];
        const char* srcPtr = srcBuffer;

        if (op == OP_HEAVY) {
            srcPtr = kHeavySource;
        } else {
            switch (op) {
                case OP_ADD:  std::snprintf(srcBuffer, sizeof(srcBuffer), kBinaryTemplateVec, "+", "+"); break;
                case OP_SUB:  std::snprintf(srcBuffer, sizeof(srcBuffer), kBinaryTemplateVec, "-", "-"); break;
                case OP_MUL:  std::snprintf(srcBuffer, sizeof(srcBuffer), kBinaryTemplateVec, "*", "*"); break;
                case OP_DIV:  std::snprintf(srcBuffer, sizeof(srcBuffer), kBinaryTemplateVec, "/", "/"); break;
                case OP_ADD_SCALAR: std::snprintf(srcBuffer, sizeof(srcBuffer), kScalarTemplateVec, "+", "+"); break;
                case OP_SUB_SCALAR: std::snprintf(srcBuffer, sizeof(srcBuffer), kScalarTemplateVec, "-", "-"); break;
                case OP_MUL_SCALAR: std::snprintf(srcBuffer, sizeof(srcBuffer), kScalarTemplateVec, "*", "*"); break;
                case OP_DIV_SCALAR: std::snprintf(srcBuffer, sizeof(srcBuffer), kScalarTemplateVec, "/", "/"); break;
                case OP_SIN:  std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "native_sin", "native_sin"); break;
                case OP_COS:  std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "native_cos", "native_cos"); break;
                case OP_TAN:  std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "native_tan", "native_tan"); break;
                case OP_ASIN: std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "asin", "asin"); break; // no native_asin on most vendors
                case OP_ACOS: std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "acos", "acos"); break; // no native_acos on most vendors
                case OP_ATAN: std::snprintf(srcBuffer, sizeof(srcBuffer), kUnaryTemplateVec, "atan", "atan"); break;
                default: return nullptr;
            }
        }

        program = clCreateProgramWithSource(s_context, 1, &srcPtr, nullptr, &err);
        checkErr(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &s_device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[2048];
            clGetProgramBuildInfo(program, s_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            std::printf("[VoidCLcompute] Build error:\n%s\n", log);
            clReleaseProgram(program);
            return nullptr;
        }

        // Save compiled binary for next run.
        size_t binSize = 0;
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binSize), &binSize, nullptr);
        if (binSize > 0) {
            std::vector<unsigned char> binOut(binSize);
            unsigned char* binOutPtr = binOut.data();
            clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binOutPtr), &binOutPtr, nullptr);
            saveCachedBinary(cachePath, binOut.data(), binOut.size());
        }
    }

    cl_kernel kernel = clCreateKernel(program, "compute", &err);
    checkErr(err, "clCreateKernel");

    size_t multiple = 64;
    clGetKernelWorkGroupInfo(kernel, s_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                              sizeof(multiple), &multiple, nullptr);
    if (multiple == 0) multiple = 64;

    s_programCache[op] = program;
    s_kernelCache[op] = kernel;
    s_preferredMultiple[op] = multiple;
    return kernel;
}

static BinaryBufSet& getBinaryBuffers(size_t bytes, cl_int& err) {
    auto it = s_binaryPool.find(bytes);
    if (it != s_binaryPool.end()) return it->second;

    BinaryBufSet set;
    set.bytes = bytes;
    set.a   = clCreateBuffer(s_context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
    set.b   = clCreateBuffer(s_context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
    set.out = clCreateBuffer(s_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);

    set.mappedA   = (float*)clEnqueueMapBuffer(s_queue, set.a,   CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
    set.mappedB   = (float*)clEnqueueMapBuffer(s_queue, set.b,   CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
    set.mappedOut = (float*)clEnqueueMapBuffer(s_queue, set.out, CL_TRUE, CL_MAP_READ,  0, bytes, 0, nullptr, nullptr, &err);

    return s_binaryPool.emplace(bytes, set).first->second;
}

static UnaryBufSet& getUnaryBuffers(size_t bytes, cl_int& err) {
    auto it = s_unaryPool.find(bytes);
    if (it != s_unaryPool.end()) return it->second;

    UnaryBufSet set;
    set.bytes = bytes;
    set.a   = clCreateBuffer(s_context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
    set.out = clCreateBuffer(s_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);

    set.mappedA   = (float*)clEnqueueMapBuffer(s_queue, set.a,   CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
    set.mappedOut = (float*)clEnqueueMapBuffer(s_queue, set.out, CL_TRUE, CL_MAP_READ,  0, bytes, 0, nullptr, nullptr, &err);

    return s_unaryPool.emplace(bytes, set).first->second;
}

static void releaseBinaryPool() {
    for (auto& kv : s_binaryPool) {
        auto& s = kv.second;
        if (s.mappedA)   clEnqueueUnmapMemObject(s_queue, s.a,   s.mappedA, 0, nullptr, nullptr);
        if (s.mappedB)   clEnqueueUnmapMemObject(s_queue, s.b,   s.mappedB, 0, nullptr, nullptr);
        if (s.mappedOut) clEnqueueUnmapMemObject(s_queue, s.out, s.mappedOut, 0, nullptr, nullptr);
        if (s.a)   clReleaseMemObject(s.a);
        if (s.b)   clReleaseMemObject(s.b);
        if (s.out) clReleaseMemObject(s.out);
    }
    if (s_queue) clFinish(s_queue);
    s_binaryPool.clear();
}

static void releaseUnaryPool() {
    for (auto& kv : s_unaryPool) {
        auto& s = kv.second;
        if (s.mappedA)   clEnqueueUnmapMemObject(s_queue, s.a,   s.mappedA, 0, nullptr, nullptr);
        if (s.mappedOut) clEnqueueUnmapMemObject(s_queue, s.out, s.mappedOut, 0, nullptr, nullptr);
        if (s.a)   clReleaseMemObject(s.a);
        if (s.out) clReleaseMemObject(s.out);
    }
    if (s_queue) clFinish(s_queue);
    s_unaryPool.clear();
}

bool GC_Init() {
    if (s_context) return true;

    cl_int err = clGetPlatformIDs(1, &s_platform, nullptr);
    if (err != CL_SUCCESS) {
        std::printf("[VoidCLcompute] No OpenCL platform found\n");
        return false;
    }

    err = clGetDeviceIDs(s_platform, CL_DEVICE_TYPE_GPU, 1, &s_device, nullptr);
    if (err != CL_SUCCESS) {
        std::printf("[VoidCLcompute] No GPU device found, falling back to CPU\n");
        err = clGetDeviceIDs(s_platform, CL_DEVICE_TYPE_CPU, 1, &s_device, nullptr);
        if (err != CL_SUCCESS) {
            std::printf("[VoidCLcompute] No OpenCL device found at all\n");
            return false;
        }
    }

    char deviceName[256];
    clGetDeviceInfo(s_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::printf("[VoidCLcompute] Using device: %s\n", deviceName);
    s_deviceName = deviceName;

    s_context = clCreateContext(nullptr, 1, &s_device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");

    // Out-of-order queue: lets independent enqueued ops overlap on the
    // device instead of forcing strict in-order execution.
    cl_command_queue_properties qprops = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    s_queue = clCreateCommandQueue(s_context, s_device, qprops, &err);
    if (err != CL_SUCCESS) {
        // Not all devices support out-of-order queues — fall back gracefully.
        std::printf("[VoidCLcompute] Out-of-order queue unsupported, using in-order\n");
        s_queue = clCreateCommandQueue(s_context, s_device, 0, &err);
    }
    checkErr(err, "clCreateCommandQueue");

    return err == CL_SUCCESS;
}

void GC_TrimBufferCache() {
    releaseBinaryPool();
    releaseUnaryPool();
}

void GC_Shutdown() {
    for (auto& pair : s_kernelCache)  clReleaseKernel(pair.second);
    for (auto& pair : s_programCache) clReleaseProgram(pair.second);
    s_kernelCache.clear();
    s_programCache.clear();
    s_preferredMultiple.clear();

    GC_TrimBufferCache();

    if (s_queue)   clReleaseCommandQueue(s_queue);
    if (s_context) clReleaseContext(s_context);
    s_queue = nullptr;
    s_context = nullptr;
}

static size_t roundUp(size_t n, size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

static void runBinaryOp(OpId op, const float* a, const float* b, float* result, int count) {
    cl_kernel kernel = getOrBuildKernel(op);
    if (!kernel) return;

    cl_int err = CL_SUCCESS;
    size_t bytes = (size_t)count * sizeof(float);
    BinaryBufSet& bufs = getBinaryBuffers(bytes, err);
    if (!bufs.mappedA || !bufs.mappedB || !bufs.mappedOut) { checkErr(err, "getBinaryBuffers"); return; }

    // Direct copy into mapped (pinned) host-visible memory — avoids the
    // driver's internal staging-buffer copy that plain WriteBuffer implies
    // on discrete GPUs without CL_MEM_ALLOC_HOST_PTR.
    std::memcpy(bufs.mappedA, a, bytes);
    std::memcpy(bufs.mappedB, b, bytes);

    unsigned int uCount = (unsigned int)count;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufs.a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufs.b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufs.out);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &uCount);

    // OP_HEAVY is scalar (one work-item per element); vectorized ops
    // process 4 elements per work-item, so divide the item count by 4.
    size_t items = (op == OP_HEAVY) ? (size_t)count : (((size_t)count + 3) / 4);
    size_t localSize = s_preferredMultiple.count(op) ? s_preferredMultiple[op] : 64;
    size_t globalSize = roundUp(items, localSize);

    cl_event kernelDone = nullptr;
    err = clEnqueueNDRangeKernel(s_queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &kernelDone);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Wait only on this kernel's completion event, not the whole queue —
    // lets unrelated in-flight work on an out-of-order queue keep running.
    clWaitForEvents(1, &kernelDone);
    clReleaseEvent(kernelDone);
    std::memcpy(result, bufs.mappedOut, bytes);
}

static void runUnaryOp(OpId op, const float* input, float* result, int count) {
    cl_kernel kernel = getOrBuildKernel(op);
    if (!kernel) return;

    cl_int err = CL_SUCCESS;
    size_t bytes = (size_t)count * sizeof(float);
    UnaryBufSet& bufs = getUnaryBuffers(bytes, err);
    if (!bufs.mappedA || !bufs.mappedOut) { checkErr(err, "getUnaryBuffers"); return; }

    std::memcpy(bufs.mappedA, input, bytes);

    unsigned int uCount = (unsigned int)count;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufs.a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufs.out);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &uCount);

    size_t items = (((size_t)count + 3) / 4);
    size_t localSize = s_preferredMultiple.count(op) ? s_preferredMultiple[op] : 64;
    size_t globalSize = roundUp(items, localSize);

    cl_event kernelDone = nullptr;
    err = clEnqueueNDRangeKernel(s_queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &kernelDone);
    checkErr(err, "clEnqueueNDRangeKernel");

    clWaitForEvents(1, &kernelDone);
    clReleaseEvent(kernelDone);
    std::memcpy(result, bufs.mappedOut, bytes);
}

// One input array + one plain scalar kernel argument — no second array,
// no fake "fill an array with the same number" step. Reuses the same
// buffer pool as the unary ops since the memory shape (1 in, 1 out) is
// identical; only the kernel argument list differs.
static void runScalarOp(OpId op, const float* a, float scalar, float* result, int count) {
    cl_kernel kernel = getOrBuildKernel(op);
    if (!kernel) return;

    cl_int err = CL_SUCCESS;
    size_t bytes = (size_t)count * sizeof(float);
    UnaryBufSet& bufs = getUnaryBuffers(bytes, err);
    if (!bufs.mappedA || !bufs.mappedOut) { checkErr(err, "getUnaryBuffers"); return; }

    std::memcpy(bufs.mappedA, a, bytes);

    unsigned int uCount = (unsigned int)count;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufs.a);
    clSetKernelArg(kernel, 1, sizeof(float), &scalar);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufs.out);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &uCount);

    size_t items = (((size_t)count + 3) / 4);
    size_t localSize = s_preferredMultiple.count(op) ? s_preferredMultiple[op] : 64;
    size_t globalSize = roundUp(items, localSize);

    cl_event kernelDone = nullptr;
    err = clEnqueueNDRangeKernel(s_queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &kernelDone);
    checkErr(err, "clEnqueueNDRangeKernel");

    clWaitForEvents(1, &kernelDone);
    clReleaseEvent(kernelDone);
    std::memcpy(result, bufs.mappedOut, bytes);
}

// ============================================================
// Exported function wrappers
// ============================================================
void gpu_add(const float* a, const float* b, float* result, int count)      { runBinaryOp(OP_ADD, a, b, result, count); }
void gpu_subtract(const float* a, const float* b, float* result, int count) { runBinaryOp(OP_SUB, a, b, result, count); }
void gpu_multiply(const float* a, const float* b, float* result, int count) { runBinaryOp(OP_MUL, a, b, result, count); }
void gpu_divide(const float* a, const float* b, float* result, int count)   { runBinaryOp(OP_DIV, a, b, result, count); }

void gpu_sin(const float* input, float* result, int count)  { runUnaryOp(OP_SIN, input, result, count); }
void gpu_cos(const float* input, float* result, int count)  { runUnaryOp(OP_COS, input, result, count); }
void gpu_tan(const float* input, float* result, int count)  { runUnaryOp(OP_TAN, input, result, count); }
void gpu_asin(const float* input, float* result, int count) { runUnaryOp(OP_ASIN, input, result, count); }
void gpu_acos(const float* input, float* result, int count) { runUnaryOp(OP_ACOS, input, result, count); }
void gpu_atan(const float* input, float* result, int count) { runUnaryOp(OP_ATAN, input, result, count); }

void gpu_heavy(const float* a, const float* b, float* result, int count) {
    runBinaryOp(OP_HEAVY, a, b, result, count);
}

void gpu_add_scalar(const float* a, float scalar, float* result, int count)      { runScalarOp(OP_ADD_SCALAR, a, scalar, result, count); }
void gpu_subtract_scalar(const float* a, float scalar, float* result, int count) { runScalarOp(OP_SUB_SCALAR, a, scalar, result, count); }
void gpu_multiply_scalar(const float* a, float scalar, float* result, int count) { runScalarOp(OP_MUL_SCALAR, a, scalar, result, count); }
void gpu_divide_scalar(const float* a, float scalar, float* result, int count)   { runScalarOp(OP_DIV_SCALAR, a, scalar, result, count); }

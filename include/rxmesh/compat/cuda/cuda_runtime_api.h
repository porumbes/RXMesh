#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __constant__
#define __constant__
#endif

#ifndef __grid_constant__
#define __grid_constant__
#endif

#ifndef __forceinline__
#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#else
#define __forceinline__ inline __attribute__((always_inline))
#endif
#endif

#ifndef __inline__
#define __inline__ inline
#endif

#ifndef __nv_is_extended_device_lambda_closure_type
#define __nv_is_extended_device_lambda_closure_type(X) true
#endif

#ifndef __nv_is_extended_host_device_lambda_closure_type
#define __nv_is_extended_host_device_lambda_closure_type(X) true
#endif

enum cudaError_t
{
    cudaSuccess                = 0,
    cudaErrorInvalidValue      = 1,
    cudaErrorMemoryAllocation  = 2
};

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

enum cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
};

enum cudaDataType_t
{
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_C_32F = 4,
    CUDA_C_64F = 5,
    CUDA_R_8I  = 3,
    CUDA_R_8U  = 8,
    CUDA_R_16I = 20,
    CUDA_R_16U = 21,
    CUDA_R_32I = 10,
    CUDA_R_32U = 12,
    CUDA_R_64I = 24,
    CUDA_R_64U = 25
};

struct cuComplex
{
    float x, y;
};

struct cuDoubleComplex
{
    double x, y;
};

using cudaStream_t = void*;

struct cudaEvent_i
{
    std::chrono::high_resolution_clock::time_point tp;
};
using cudaEvent_t = cudaEvent_i*;

struct dim3
{
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1,
                   unsigned int vy = 1,
                   unsigned int vz = 1)
        : x(vx), y(vy), z(vz)
    {
    }
};

struct cudaFuncAttributes
{
    size_t sharedSizeBytes         = 0;
    size_t constSizeBytes          = 0;
    size_t localSizeBytes          = 0;
    int    maxThreadsPerBlock      = 1024;
    int    numRegs                 = 32;
    int    ptxVersion              = 86;
    int    binaryVersion           = 86;
    int    cacheModeCA             = 0;
    int    maxDynamicSharedSizeBytes = 65536;
    int    preferredShmemCarveout  = 0;
};

struct cudaDeviceProp
{
    char   name[256]                 = "Host CPU / Unified Memory";
    int    major                     = 8;
    int    minor                     = 6;
    size_t totalGlobalMem            = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    int    multiProcessorCount       = 1;
    int    maxBlocksPerMultiProcessor = 16;
    int    ECCEnabled                = 0;
    int    clockRate                 = 1000000;
    int    memoryClockRate           = 1000000;
    int    memoryBusWidth            = 256;
    int    managedMemory             = 1;
    size_t sharedMemPerBlock         = 65536;
    size_t sharedMemPerBlockOptin    = 65536;
    int    maxThreadsPerMultiProcessor = 1024;
    int    maxThreadsPerBlock        = 1024;
    int    regsPerBlock              = 65536;
};

inline const char* cudaGetErrorString(cudaError_t err)
{
    return (err == cudaSuccess) ? "no error" : "cuda compat error";
}

inline cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize()
{
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t)
{
    return cudaSuccess;
}

inline cudaError_t cudaDeviceReset()
{
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }
    *devPtr = std::calloc(1, size == 0 ? 1 : size);
    return (*devPtr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFree(void* devPtr)
{
    std::free(devPtr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void*          dst,
                              const void*    src,
                              size_t         count,
                              cudaMemcpyKind /*kind*/)
{
    if (count > 0 && dst != src) {
        std::memcpy(dst, src, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void*          dst,
                                   const void*    src,
                                   size_t         count,
                                   cudaMemcpyKind kind,
                                   cudaStream_t /*stream*/ = nullptr)
{
    return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
    if (count > 0 && devPtr != nullptr) {
        std::memset(devPtr, value, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void*        devPtr,
                                   int          value,
                                   size_t       count,
                                   cudaStream_t /*stream*/ = nullptr)
{
    return cudaMemset(devPtr, value, count);
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    if (!event) {
        return cudaErrorInvalidValue;
    }
    *event = new cudaEvent_i();
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t  event,
                                   cudaStream_t /*stream*/ = nullptr)
{
    if (event) {
        event->tp = std::chrono::high_resolution_clock::now();
    }
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t /*event*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaEventElapsedTime(float*      ms,
                                        cudaEvent_t start,
                                        cudaEvent_t end)
{
    if (!ms || !start || !end) {
        return cudaErrorInvalidValue;
    }
    *ms = std::chrono::duration<float, std::milli>(end->tp - start->tp).count();
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int* count)
{
    if (!count) {
        return cudaErrorInvalidValue;
    }
    *count = 1;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device)
{
    if (!device) {
        return cudaErrorInvalidValue;
    }
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int /*device*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/)
{
    if (!prop) {
        return cudaErrorInvalidValue;
    }
    *prop = cudaDeviceProp{};
    return cudaSuccess;
}

inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                         const void* /*func*/)
{
    if (!attr) {
        return cudaErrorInvalidValue;
    }
    *attr = cudaFuncAttributes{};
    return cudaSuccess;
}

inline cudaError_t cudaFuncSetAttribute(const void* /*func*/,
                                        cudaFuncAttribute /*attr*/,
                                        int /*value*/)
{
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(size_t* free_mem, size_t* total_mem)
{
    if (free_mem) {
        *free_mem = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    if (total_mem) {
        *total_mem = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int* driverVersion)
{
    if (driverVersion) {
        *driverVersion = 0;
    }
    return cudaSuccess;
}

inline cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
    if (runtimeVersion) {
        *runtimeVersion = 0;
    }
    return cudaSuccess;
}

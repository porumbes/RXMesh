#pragma once

#ifndef RXMESH_LIBRARY_PROPERTY_TYPE_DEFINED
#define RXMESH_LIBRARY_PROPERTY_TYPE_DEFINED
enum libraryPropertyType_t
{
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL   = 2
};
#endif

enum cusolverStatus_t
{
    CUSOLVER_STATUS_SUCCESS                   = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED           = 1,
    CUSOLVER_STATUS_ALLOC_FAILED              = 2,
    CUSOLVER_STATUS_INVALID_VALUE             = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH             = 4,
    CUSOLVER_STATUS_EXECUTION_FAILED          = 5,
    CUSOLVER_STATUS_INTERNAL_ERROR            = 6,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 7
};

inline cusolverStatus_t cusolverGetProperty(libraryPropertyType_t, int* value)
{
    if (value) {
        *value = 0;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

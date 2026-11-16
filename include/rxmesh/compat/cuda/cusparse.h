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

enum cusparseStatus_t
{
    CUSPARSE_STATUS_SUCCESS = 0
};

inline const char* cusparseGetErrorString(cusparseStatus_t)
{
    return "CUSPARSE_STATUS_SUCCESS";
}

inline cusparseStatus_t cusparseGetProperty(libraryPropertyType_t, int* value)
{
    if (value) {
        *value = 0;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

enum cublasStatus_t
{
    CUBLAS_STATUS_SUCCESS = 0
};

inline const char* cublasGetStatusString(cublasStatus_t)
{
    return "CUBLAS_STATUS_SUCCESS";
}

inline cublasStatus_t cublasGetProperty(libraryPropertyType_t, int* value)
{
    if (value) {
        *value = 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

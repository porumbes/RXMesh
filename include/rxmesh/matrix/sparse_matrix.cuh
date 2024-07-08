#pragma once
#include "cusolverSp.h"
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/types.h"

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/gather.h"
#include "thrust/scatter.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "rxmesh/matrix/dense_matrix.cuh"

namespace rxmesh {

/**
 * @brief The enum class for choosing different solver types
 */
enum class Solver
{
    CHOL = 0,
    LU   = 1,
    QR   = 2
};

/**
 * @brief The enum class for choosing different reorder types
 * NONE for No Reordering Applied, SYMRCM for Symmetric Reverse Cuthill-McKee
 * permutation, SYMAMD for Symmetric Approximate Minimum Degree Algorithm based
 * on Quotient Graph, NSTDIS for Nested Dissection
 */
enum class Reorder
{
    NONE   = 0,
    SYMRCM = 1,
    SYMAMD = 2,
    NSTDIS = 3
};

static int reorder_to_int(const Reorder& reorder)
{
    switch (reorder) {
        case Reorder::NONE:
            return 0;
        case Reorder::SYMRCM:
            return 1;
        case Reorder::SYMAMD:
            return 2;
        case Reorder::NSTDIS:
            return 3;
        default: {
            RXMESH_ERROR("reorder_to_int() unknown input reorder");
            return 0;
        }
    }
}

namespace detail {
// this is the function for the CSR calculation
template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_prescan(const rxmesh::Context context,
                                          IndexT*               row_ptr)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids                                          = v_id.unpack();
        uint32_t patch_id                                     = ids.first;
        uint16_t local_id                                     = ids.second;
        row_ptr[context.vertex_prefix()[patch_id] + local_id] = iter.size() + 1;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_col_fill(const rxmesh::Context context,
                                           IndexT*               row_ptr,
                                           IndexT*               col_idx)
{
    using namespace rxmesh;

    auto col_fillin = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        col_idx[row_ptr[context.vertex_prefix()[patch_id] + local_id]] =
            context.vertex_prefix()[patch_id] + local_id;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            auto     s_ids      = iter[v].unpack();
            uint32_t s_patch_id = s_ids.first;
            uint16_t s_local_id = s_ids.second;
            col_idx[row_ptr[context.vertex_prefix()[patch_id] + local_id] + v +
                    1] = context.vertex_prefix()[s_patch_id] + s_local_id;
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, col_fillin);
}

// d_out[d_p[i]] = d_in[i]
template <typename T, typename IndexT = int>
void permute_scatter(IndexT* d_p, T* d_in, T* d_out, IndexT size)
{
    thrust::device_ptr<IndexT> t_p(d_p);
    thrust::device_ptr<T>      t_i(d_in);
    thrust::device_ptr<T>      t_o(d_out);

    thrust::scatter(thrust::device, t_i, t_i + size, t_p, t_o);
}

// d_out[i] = d_in[d_p[i]]
template <typename T, typename IndexT = int>
void permute_gather(IndexT* d_p, T* d_in, T* d_out, IndexT size)
{
    thrust::device_ptr<IndexT> t_p(d_p);
    thrust::device_ptr<T>      t_i(d_in);
    thrust::device_ptr<T>      t_o(d_out);

    thrust::gather(thrust::device, t_p, t_p + size, t_i, t_o);
}

}  // namespace detail


/**
 * @brief Device-only sparse matrix that represent the VV connectivity, i.e., it
 * is a square matrix with number of rows/cols is equal to number of vertices
 * and there is non-zero values at entry (i,j) only if the vertex i is connected
 * to vertex j. The sparse matrix is stored as a CSR matrix. The class also
 * provides implementation for matrix-vector multiplication and linear
 * solver�(using cuSolver and cuSparse as a back-end.
 */
template <typename T, typename IndexT = int>
struct SparseMatrix
{
    SparseMatrix(const RXMeshStatic& rx)
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_h_row_ptr(nullptr),
          m_h_col_idx(nullptr),
          m_h_val(nullptr),
          m_num_rows(0),
          m_num_cols(0),
          m_nnz(0),
          m_context(rx.get_context()),
          m_cusparse_handle(NULL),
          m_descr(NULL),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0),
          m_h_permute(nullptr),
          m_d_permute(nullptr),
          m_d_solver_row_ptr(nullptr),
          m_d_solver_col_idx(nullptr),
          m_d_solver_val(nullptr),
          m_use_reorder(false),
          m_reorder_allocated(false),
          m_d_cusparse_spmm_buffer(false),
          m_allocated(LOCATION_NONE)
    {
        using namespace rxmesh;
        constexpr uint32_t blockThreads = 256;

        IndexT num_patches  = rx.get_num_patches();
        IndexT num_vertices = rx.get_num_vertices();
        IndexT num_edges    = rx.get_num_edges();

        m_num_rows = num_vertices;
        m_num_cols = num_vertices;

        // row pointer allocation and init with prefix sum for CSR
        CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                              (num_vertices + 1) * sizeof(IndexT)));

        CUDA_ERROR(
            cudaMemset(m_d_row_ptr, 0, (num_vertices + 1) * sizeof(IndexT)));

        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::VV},
                              launch_box,
                              (void*)detail::sparse_mat_prescan<blockThreads>);

        detail::sparse_mat_prescan<blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(m_context, m_d_row_ptr);

        // prefix sum using CUB.
        void*  d_cub_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      num_vertices + 1);
        CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      num_vertices + 1);

        CUDA_ERROR(cudaFree(d_cub_temp_storage));

        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + num_vertices),
                              sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        // column index allocation and init
        CUDA_ERROR(cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));
        rx.prepare_launch_box({Op::VV},
                              launch_box,
                              (void*)detail::sparse_mat_col_fill<blockThreads>);

        detail::sparse_mat_col_fill<blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(
                m_context, m_d_row_ptr, m_d_col_idx);

        // allocate value ptr
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(T)));
        CUDA_ERROR(cudaMemset(m_d_val, 0, m_nnz * sizeof(T)));
        m_allocated = m_allocated | DEVICE;

        // create cusparse matrix
        CUSPARSE_ERROR(cusparseCreateMatDescr(&m_descr));
        CUSPARSE_ERROR(
            cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO));

        CUSPARSE_ERROR(cusparseCreateCsr(&m_spdescr,
                                         m_num_rows,
                                         m_num_cols,
                                         m_nnz,
                                         m_d_row_ptr,
                                         m_d_col_idx,
                                         m_d_val,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));

        CUSPARSE_ERROR(cusparseCreate(&m_cusparse_handle));
        CUSOLVER_ERROR(cusolverSpCreate(&m_cusolver_sphandle));


        // allocate the host
        m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
        m_h_row_ptr =
            static_cast<IndexT*>(malloc((m_num_rows + 1) * sizeof(IndexT)));
        m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

        CUDA_ERROR(cudaMemcpy(
            m_h_val, m_d_val, m_nnz * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_col_idx,
                              m_d_col_idx,
                              m_nnz * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_row_ptr,
                              m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        m_allocated = m_allocated | HOST;
    }

    /**
     * @brief set all entries in the matrix to ones on both host and device
     */
    __host__ void set_ones()
    {
        std::fill_n(m_h_val, m_nnz, 1);
        CUDA_ERROR(cudaMemcpy(
            m_d_val, m_h_val, m_nnz * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief set all entries in the matrix to zeros on both host and device
     */
    __host__ void set_zeros()
    {
        std::memset(m_h_val, 0, m_nnz * sizeof(T));

        CUDA_ERROR(cudaMemset(m_d_val, 0, m_nnz * sizeof(T)));
    }

    /**
     * @brief return number of rows
     */
    __device__ __host__ IndexT rows() const
    {
        return m_num_rows;
    }

    /**
     * @brief return number of cols
     */
    __device__ __host__ IndexT cols() const
    {
        return m_num_cols;
    }

    /**
     * @brief return number of non-zero values
     */
    __device__ __host__ IndexT non_zeros() const
    {
        return m_nnz;
    }

    /**
     * @brief access the matrix using VertexHandle
     */
    __device__ T& operator()(const VertexHandle& row_v,
                             const VertexHandle& col_v)
    {
        return m_d_val[get_val_idx(row_v, col_v)];
    }

    /**
     * @brief access the matrix using VertexHandle
     */
    __device__ T& operator()(const VertexHandle& row_v,
                             const VertexHandle& col_v) const
    {
        return m_d_val[get_val_idx(row_v, col_v)];
    }

    /**
     * @brief access the matrix using row and col index
     */
    __device__ T& operator()(const IndexT x, const IndexT y)
    {
        const IndexT start = m_d_row_ptr[x];
        const IndexT end   = m_d_row_ptr[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (m_d_col_idx[i] == y) {
                return m_d_val[i];
            }
        }
        assert(1 != 1);
    }

    /**
     * @brief access the matrix using row and col index
     */
    __device__ T& operator()(const IndexT x, const IndexT y) const
    {
        const IndexT start = m_d_row_ptr[x];
        const IndexT end   = m_d_row_ptr[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (m_d_col_idx[i] == y) {
                return m_d_val[i];
            }
        }
        assert(1 != 1);
    }

    /**
     * @brief return the row pointer of the CSR matrix
     * @return
     */
    __device__ __host__ const IndexT* row_ptr() const
    {
#ifdef __CUDA_ARCH__
        return m_d_row_ptr;
#else
        return m_h_row_ptr;
#endif
    }

    /**
     * @brief return the column index pointer of the CSR matrix
     * @return
     */
    __device__ __host__ const IndexT* col_idx() const
    {
#ifdef __CUDA_ARCH__
        return m_d_col_idx;
#else
        return m_h_col_idx;
#endif
    }

    /**
     * @brief access the value of (1D array) array that holds the nnz in the CSR
     * matrix
     */
    __device__ __host__ T& get_val_at(IndexT idx) const
    {
#ifdef __CUDA_ARCH__
        return m_d_val[idx];
#else
        return m_h_val[idx];
#endif
    }


    /**
     * @brief release all allocated memory
     */
    void release()
    {
        release(LOCATION_ALL);
        CUSPARSE_ERROR(cusparseDestroy(m_cusparse_handle));
        CUSPARSE_ERROR(cusparseDestroyMatDescr(m_descr));
        CUSOLVER_ERROR(cusolverSpDestroy(m_cusolver_sphandle));

        if (m_reorder_allocated) {
            GPU_FREE(m_d_solver_val);
            GPU_FREE(m_d_solver_row_ptr);
            GPU_FREE(m_d_solver_col_idx);
            GPU_FREE(m_d_permute);
            free(m_h_permute);
        }
        GPU_FREE(m_d_cusparse_spmm_buffer);
    }

    /**
     * @brief move the data between host an device
     */
    void move(locationT source, locationT target, cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "SparseMatrix::move() source ({}) and target ({}) "
                "are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "SparseMatrix::move() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source));
            return;
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_ERROR("SparseMatrix::move() target {} is not allocated!",
                         location_to_string(target));
            return;
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       m_h_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(m_h_val,
                                       m_d_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }
    }

    /**
     * @brief multiply the sparse matrix by a dense matrix. The function
     * performs the multiplication as
     * C = A*B
     * where A is the sparse matrix, B is a dense matrix, and the result is a
     * dense matrix C
     */
    __host__ void multiply_by_dense_matrix(rxmesh::DenseMatrix<T>& B_mat,
                                           rxmesh::DenseMatrix<T>& C_mat,
                                           cudaStream_t            stream = 0)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        // A_mat.create_cusparse_handle();
        cusparseSpMatDescr_t matA = m_spdescr;
        cusparseDnMatDescr_t matB = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC = C_mat.m_dendescr;

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        // allocate an external buffer if needed
        if (m_spmm_buffer_size == 0) {
            denmat_mul_buffer_size(B_mat, C_mat, stream);
            CUDA_ERROR(
                cudaMalloc(&m_d_cusparse_spmm_buffer, m_spmm_buffer_size));
        }


        // execute SpMM
        CUSPARSE_ERROR(cusparseSpMM(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    m_d_cusparse_spmm_buffer));
    }

    /**
     * @brief wrap up the cusparse api for sparse matrix array
     * multiplication.
     */
    void arr_mul(T* in_arr, T* rt_arr, cudaStream_t stream = 0)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        void*                buffer = NULL;
        cusparseDnVecDescr_t vecx   = NULL;
        cusparseDnVecDescr_t vecy   = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_num_cols, in_arr, CUDA_R_32F));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_num_rows, rt_arr, CUDA_R_32F));

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        if (m_spmv_buffer_size == 0) {
            RXMESH_WARN(
                "Sparse matrix - Array multiplication buffer size not "
                "initialized."
                "Calculate it now.");
            arr_mul_buffer_size(in_arr, rt_arr, stream);
        }

        CUDA_ERROR(cudaMalloc(&buffer, m_spmv_buffer_size));

        CUSPARSE_ERROR(cusparseSpMV(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    m_spdescr,
                                    vecx,
                                    &beta,
                                    vecy,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                    buffer));

        CUSPARSE_ERROR(cusparseDestroyDnVec(vecx));
        CUSPARSE_ERROR(cusparseDestroyDnVec(vecy));
        CUDA_ERROR(cudaFree(buffer));
    }

    /**
     * @brief do the sparse matrix dense matrix multiplication using sparse
     * matrix array multiplication in a column wise way
     */
    void spmat_denmat_mul_cw(rxmesh::DenseMatrix<T> B_mat,
                             rxmesh::DenseMatrix<T> C_mat)
    {
        for (int i = 0; i < B_mat.m_num_cols; ++i) {
            arr_mul(B_mat.col_data(i), C_mat.col_data(i));
        }
    }

    /*  ----- SOLVER -----  */

    /* --- HIGH LEVEL API --- */

    /**
     * @brief solve the Ax=b for x where x and b are all array
     */
    void spmat_linear_solve(T*              B_arr,
                            T*              X_arr,
                            rxmesh::Solver  solver,
                            rxmesh::Reorder reorder)
    {
        cusparse_linear_solver_wrapper(solver,
                                       reorder,
                                       m_cusolver_sphandle,
                                       m_descr,
                                       m_num_rows,
                                       m_num_cols,
                                       m_nnz,
                                       m_d_row_ptr,
                                       m_d_col_idx,
                                       m_d_val,
                                       B_arr,
                                       X_arr);
    }

    /**
     * @brief solve the AX=B for X where X and B are all dense matrix and we
     * would solve it in a column wise manner
     */
    void spmat_linear_solve(rxmesh::DenseMatrix<T> B_mat,
                            rxmesh::DenseMatrix<T> X_mat,
                            rxmesh::Solver         solver,
                            rxmesh::Reorder        reorder)
    {
        for (int i = 0; i < B_mat.cols(); ++i) {
            cusparse_linear_solver_wrapper(solver,
                                           reorder,
                                           m_cusolver_sphandle,
                                           m_descr,
                                           m_num_rows,
                                           m_num_cols,
                                           m_nnz,
                                           m_d_row_ptr,
                                           m_d_col_idx,
                                           m_d_val,
                                           B_mat.col_data(i),
                                           X_mat.col_data(i));
        }
    }

    /**
     * @brief wrap up the cusolver api for solving linear systems. This is a
     * lower level api
     */
    void cusparse_linear_solver_wrapper(const rxmesh::Solver  solver,
                                        const rxmesh::Reorder reorder,
                                        cusolverSpHandle_t    handle,
                                        cusparseMatDescr_t    descrA,
                                        int                   rowsA,
                                        int                   colsA,
                                        int                   nnzA,
                                        int*                  d_csrRowPtrA,
                                        int*                  d_csrColIndA,
                                        T*                    d_csrValA,
                                        T*                    d_b,
                                        T*                    d_x)
    {
        if constexpr ((!std::is_same_v<T, float>)&&(
                          !std::is_same_v<T, double>)) {
            RXMESH_ERROR(
                "Unsupported type for cusparse: {}"
                "Only float and double are supported",
                typeid(T).name());
        }

        double tol         = 1.e-12;
        int    singularity = 0; /* -1 if A is invertible under tol. */

        /* solve B*z = Q*b */
        if (solver == Solver::CHOL) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvchol(handle,
                                                     rowsA,
                                                     nnzA,
                                                     descrA,
                                                     d_csrValA,
                                                     d_csrRowPtrA,
                                                     d_csrColIndA,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvchol(handle,
                                                     rowsA,
                                                     nnzA,
                                                     descrA,
                                                     d_csrValA,
                                                     d_csrRowPtrA,
                                                     d_csrColIndA,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

        } else if (solver == Solver::QR) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvqr(handle,
                                                   rowsA,
                                                   nnzA,
                                                   descrA,
                                                   d_csrValA,
                                                   d_csrRowPtrA,
                                                   d_csrColIndA,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvqr(handle,
                                                   rowsA,
                                                   nnzA,
                                                   descrA,
                                                   d_csrValA,
                                                   d_csrRowPtrA,
                                                   d_csrColIndA,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }
        } else {
            RXMESH_ERROR(
                "Only Solver::CHOL and Solver::QR is supported, use CUDA 12.x "
                "for "
                "Solver::LU");
        }
        CUDA_ERROR(cudaDeviceSynchronize());

        if (0 <= singularity) {
            RXMESH_WARN(
                "WARNING: the matrix is singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    /* --- LOW LEVEL API --- */

    /**
     * @brief The lower level api of reordering. Specify the reordering type or
     * simply NONE for no reordering. This should be called at the beginning of
     * the solving process. Any other function call order would be undefined.
     * @param reorder: the reorder method applied.
     */
    void spmat_chol_reorder(rxmesh::Reorder reorder)
    {
        if (reorder == Reorder::NONE) {
            RXMESH_INFO("None reordering is specified",
                        "Continue without reordering");
            m_use_reorder = false;

            if (m_reorder_allocated) {
                GPU_FREE(m_d_solver_val);
                GPU_FREE(m_d_solver_row_ptr);
                GPU_FREE(m_d_solver_col_idx);
                GPU_FREE(m_d_permute);
                free(m_h_permute);
                m_reorder_allocated = false;
            }

            return;
        }

        /*check on host*/
        bool on_host = true;
        if ((HOST & m_allocated) != HOST) {
            move(DEVICE, HOST);
            on_host = false;
        }

        m_use_reorder = true;

        // allocate the purmutated csr
        m_reorder_allocated = true;
        CUDA_ERROR(cudaMalloc((void**)&m_d_solver_val, m_nnz * sizeof(T)));
        CUDA_ERROR(cudaMalloc((void**)&m_d_solver_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT)));
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_solver_col_idx, m_nnz * sizeof(IndexT)));

        m_h_permute = (IndexT*)malloc(m_num_rows * sizeof(IndexT));
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_permute, m_num_rows * sizeof(IndexT)));

        CUSOLVER_ERROR(cusolverSpCreate(&m_cusolver_sphandle));

        if (reorder == Reorder::SYMRCM) {
            CUSOLVER_ERROR(cusolverSpXcsrsymrcmHost(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_row_ptr,
                                                    m_h_col_idx,
                                                    m_h_permute));
        } else if (reorder == Reorder::SYMAMD) {
            CUSOLVER_ERROR(cusolverSpXcsrsymamdHost(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_row_ptr,
                                                    m_h_col_idx,
                                                    m_h_permute));
        } else if (reorder == Reorder::NSTDIS) {
            CUSOLVER_ERROR(cusolverSpXcsrmetisndHost(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_nnz,
                                                     m_descr,
                                                     m_h_row_ptr,
                                                     m_h_col_idx,
                                                     NULL,
                                                     m_h_permute));
        }

        CUDA_ERROR(cudaMemcpyAsync(m_d_permute,
                                   m_h_permute,
                                   m_num_rows * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // working space for permutation: B = A*Q*A^T
        // the permutation for matrix A which works only for the col and row
        // indices, the val will be done on device with the d/h_val_permute
        IndexT* h_val_permute =
            static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));
        IndexT* d_val_permute;
        CUDA_ERROR(cudaMalloc((void**)&d_val_permute, m_nnz * sizeof(IndexT)));

        size_t size_perm       = 0;
        void*  perm_buffer_cpu = NULL;

        CUSOLVER_ERROR(cusolverSpXcsrperm_bufferSizeHost(m_cusolver_sphandle,
                                                         m_num_rows,
                                                         m_num_cols,
                                                         m_nnz,
                                                         m_descr,
                                                         m_h_row_ptr,
                                                         m_h_col_idx,
                                                         m_h_permute,
                                                         m_h_permute,
                                                         &size_perm));

        perm_buffer_cpu = (void*)malloc(sizeof(char) * size_perm);

        for (int j = 0; j < m_nnz; j++) {
            h_val_permute[j] = j;
        }

        CUSOLVER_ERROR(cusolverSpXcsrpermHost(m_cusolver_sphandle,
                                              m_num_rows,
                                              m_num_cols,
                                              m_nnz,
                                              m_descr,
                                              m_h_row_ptr,
                                              m_h_col_idx,
                                              m_h_permute,
                                              m_h_permute,
                                              h_val_permute,
                                              perm_buffer_cpu));


        // do the permutation for val indices on host
        // T* tmp_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));

        // for (int j = 0; j < m_nnz; j++) {
        //     tmp_h_val[j] = m_h_val[j];
        // }
        // for (int j = 0; j < m_nnz; j++) {
        //     m_h_val[j] = tmp_h_val[h_val_permute[j]];
        // }

        // copy the purmutated csr from the host
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_val,
                                   m_h_val,
                                   m_nnz * sizeof(T),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_row_ptr,
                                   m_h_row_ptr,
                                   (m_num_rows + 1) * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_col_idx,
                                   m_h_col_idx,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // do the permutation for val indices on device
        CUDA_ERROR(cudaMemcpyAsync(d_val_permute,
                                   h_val_permute,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        detail::permute_gather(d_val_permute, m_d_val, m_d_solver_val, m_nnz);

        free(h_val_permute);
        GPU_FREE(d_val_permute);

        // restore the host data back to the original
        if (on_host) {
            move(DEVICE, HOST);
        } else {
            release(HOST);
        }
    }

    /**
     * @brief The lower level api of matrix analysis. Generating a member value
     * of type csrcholInfo_t for cucolver.
     */
    void spmat_chol_analysis()
    {
        if (!m_use_reorder) {
            m_d_solver_row_ptr = m_d_row_ptr;
            m_d_solver_col_idx = m_d_col_idx;
            m_d_solver_val     = m_d_val;
        }

        CUSOLVER_ERROR(cusolverSpCreateCsrcholInfo(&m_chol_info));
        m_internalDataInBytes = 0;
        m_workspaceInBytes    = 0;
        CUSOLVER_ERROR(cusolverSpXcsrcholAnalysis(m_cusolver_sphandle,
                                                  m_num_rows,
                                                  m_nnz,
                                                  m_descr,
                                                  m_d_solver_row_ptr,
                                                  m_d_solver_col_idx,
                                                  m_chol_info));
    }

    /**
     * @brief The lower level api of matrix factorization buffer calculation and
     * allocation. The buffer is a member variable.
     */
    void spmat_chol_buffer_alloc()
    {
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        CUDA_ERROR(cudaMalloc((void**)&m_chol_buffer, m_workspaceInBytes));
    }

    /**
     * @brief The lower level api of matrix factorization buffer release.
     */
    void spmat_chol_buffer_free()
    {
        CUDA_ERROR(cudaFree(m_chol_buffer));
    }

    /**
     * @brief The lower level api of matrix factorization and save the
     * factorization result in to the buffer.
     */
    void spmat_chol_factor()
    {
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholFactor(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_chol_buffer));
        }
        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholFactor(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_chol_buffer));
        }

        double tol = 1.0e-8;
        int    singularity;

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }
        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholZeroPivot(
                m_cusolver_sphandle, m_chol_info, tol, &singularity));
        }
        if (0 <= singularity) {
            RXMESH_WARN(
                "WARNING: the matrix is singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    /**
     * @brief The lower level api of solving the linear system after using
     * cholesky factorization. The format follows Ax=b to solve x, where A is
     * the sparse matrix, x and b are device array. As long as A doesn't change.
     * This function could be called for many different b and x.
     * @param d_b: device array of b
     * @param d_x: device array of x
     */
    void spmat_chol_solve(T* d_b, T* d_x)
    {

        T* d_solver_b;
        T* d_solver_x;

        if (m_use_reorder) {
            /* purmute b and x*/
            CUDA_ERROR(cudaMalloc((void**)&d_solver_b, m_num_rows * sizeof(T)));
            detail::permute_gather(m_d_permute, d_b, d_solver_b, m_num_rows);

            CUDA_ERROR(cudaMalloc((void**)&d_solver_x, m_num_cols * sizeof(T)));
            detail::permute_gather(m_d_permute, d_x, d_solver_x, m_num_rows);
        } else {
            d_solver_b = d_b;
            d_solver_x = d_x;
        }

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholSolve(m_cusolver_sphandle,
                                                   m_num_rows,
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_chol_buffer));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholSolve(m_cusolver_sphandle,
                                                   m_num_rows,
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_chol_buffer));
        }

        if (m_use_reorder) {
            detail::permute_scatter(m_d_permute, d_solver_x, d_x, m_num_rows);
            GPU_FREE(d_solver_b);
            GPU_FREE(d_solver_x);
        }
    }


   private:
    __device__ const IndexT get_val_idx(const VertexHandle& row_v,
                                        const VertexHandle& col_v) const
    {
        auto     r_ids      = row_v.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        auto     c_ids      = col_v.unpack();
        uint32_t c_patch_id = c_ids.first;
        uint16_t c_local_id = c_ids.second;

        uint32_t col_index = m_context.vertex_prefix()[c_patch_id] + c_local_id;
        uint32_t row_index = m_context.vertex_prefix()[r_patch_id] + r_local_id;

        const IndexT start = m_d_row_ptr[row_index];
        const IndexT end   = m_d_row_ptr[row_index + 1];

        for (IndexT i = start; i < end; ++i) {
            if (m_d_col_idx[i] == col_index) {
                return i;
            }
        }
        return 0;
    }


    void denmat_mul_buffer_size(rxmesh::DenseMatrix<T> B_mat,
                                rxmesh::DenseMatrix<T> C_mat,
                                cudaStream_t           stream = 0)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        cusparseSpMatDescr_t matA    = m_spdescr;
        cusparseDnMatDescr_t matB    = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC    = C_mat.m_dendescr;
        void*                dBuffer = NULL;

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        // allocate an external buffer if needed
        CUSPARSE_ERROR(cusparseSpMM_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               CUDA_R_32F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &m_spmm_buffer_size));
    }


    void arr_mul_buffer_size(T* in_arr, T* rt_arr, cudaStream_t stream = 0)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        cusparseDnVecDescr_t vecx = NULL;
        cusparseDnVecDescr_t vecy = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_num_cols, in_arr, CUDA_R_32F));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_num_rows, rt_arr, CUDA_R_32F));

        CUSPARSE_ERROR(cusparseSpMV_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               m_spdescr,
                                               vecx,
                                               &beta,
                                               vecy,
                                               CUDA_R_32F,
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               &m_spmv_buffer_size));
    }


    void release(locationT location)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            free(m_h_val);
            free(m_h_row_ptr);
            free(m_h_col_idx);
            m_h_val     = nullptr;
            m_h_row_ptr = nullptr;
            m_h_col_idx = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_val);
            GPU_FREE(m_d_row_ptr);
            GPU_FREE(m_d_col_idx);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    void allocate(locationT location)
    {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
            m_h_row_ptr =
                static_cast<IndexT*>(malloc((m_num_rows + 1) * sizeof(IndexT)));
            m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(T)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                                  (m_num_rows + 1) * sizeof(IndexT)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | DEVICE;
        }
    }

    const Context        m_context;
    cusparseHandle_t     m_cusparse_handle;
    cusolverSpHandle_t   m_cusolver_sphandle;
    cusparseSpMatDescr_t m_spdescr;
    cusparseMatDescr_t   m_descr;

    IndexT m_num_rows;
    IndexT m_num_cols;
    IndexT m_nnz;

    // device csr data
    IndexT* m_d_row_ptr;
    IndexT* m_d_col_idx;
    T*      m_d_val;

    // host csr data
    IndexT* m_h_row_ptr;
    IndexT* m_h_col_idx;
    T*      m_h_val;

    // susparse buffer
    size_t m_spmm_buffer_size;
    size_t m_spmv_buffer_size;

    // lower level API parameters
    csrcholInfo_t m_chol_info;
    size_t        m_internalDataInBytes;
    size_t        m_workspaceInBytes;
    void*         m_chol_buffer;

    // purmutation array
    IndexT* m_h_permute;
    IndexT* m_d_permute;

    // CSR matrix for solving only
    // equal to the original matrix if not permutated
    // only allocated as a new CSR matrix if permutated
    bool    m_reorder_allocated;
    IndexT* m_d_solver_row_ptr;
    IndexT* m_d_solver_col_idx;
    T*      m_d_solver_val;

    void* m_d_cusparse_spmm_buffer;

    // flags
    bool      m_use_reorder;
    locationT m_allocated;
};

}  // namespace rxmesh
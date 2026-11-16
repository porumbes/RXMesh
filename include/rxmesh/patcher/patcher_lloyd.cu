#include <assert.h>
#include <stdint.h>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"
#include "cuda_profiler_api.h"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patcher/patcher.h"
#include "rxmesh/patcher/patcher_kernel.cuh"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
namespace patcher {

void Patcher::execute_lloyd(std::vector<uint32_t>&       seeds,
                            const std::vector<uint32_t>& ff_offset,
                            const std::vector<uint32_t>& ff_values)
{
    uint32_t* d_face_patch            = nullptr;
    uint32_t* d_queue                 = nullptr;
    uint32_t* d_queue_ptr             = nullptr;
    uint32_t* d_ff_values             = nullptr;
    uint32_t* d_ff_offset             = nullptr;
    void*     d_cub_temp_storage_scan = nullptr;
    void*     d_cub_temp_storage_max  = nullptr;
    size_t    cub_scan_bytes          = 0;
    size_t    cub_max_bytes           = 0;
    uint32_t* d_seeds                 = nullptr;
    uint32_t* d_new_num_patches       = nullptr;
    uint32_t* d_max_patch_size        = nullptr;
    uint32_t* d_patches_offset        = nullptr;
    uint32_t* d_patches_size          = nullptr;
    uint32_t* d_patches_val           = nullptr;

    initialize_random_seeds(seeds, ff_offset, ff_values);
    allocate_device_memory(seeds,
                           ff_offset,
                           ff_values,
                           d_face_patch,
                           d_queue,
                           d_queue_ptr,
                           d_ff_values,
                           d_ff_offset,
                           d_cub_temp_storage_scan,
                           d_cub_temp_storage_max,
                           cub_scan_bytes,
                           cub_max_bytes,
                           d_seeds,
                           d_new_num_patches,
                           d_max_patch_size,
                           d_patches_offset,
                           d_patches_size,
                           d_patches_val);
    run_lloyd(d_face_patch,
              d_queue,
              d_queue_ptr,
              d_ff_values,
              d_ff_offset,
              d_cub_temp_storage_scan,
              d_cub_temp_storage_max,
              cub_scan_bytes,
              cub_max_bytes,
              d_seeds,
              d_new_num_patches,
              d_max_patch_size,
              d_patches_offset,
              d_patches_size,
              d_patches_val);

    GPU_FREE(d_face_patch);
    GPU_FREE(d_queue);
    GPU_FREE(d_queue_ptr);
    GPU_FREE(d_ff_values);
    GPU_FREE(d_ff_offset);
    GPU_FREE(d_cub_temp_storage_scan);
    GPU_FREE(d_cub_temp_storage_max);
    GPU_FREE(d_seeds);
    GPU_FREE(d_new_num_patches);
    GPU_FREE(d_max_patch_size);
    GPU_FREE(d_patches_offset);
    GPU_FREE(d_patches_size);
    GPU_FREE(d_patches_val);
}

void Patcher::allocate_device_memory(const std::vector<uint32_t>& seeds,
                                     const std::vector<uint32_t>& ff_offset,
                                     const std::vector<uint32_t>& ff_values,
                                     uint32_t*&                   d_face_patch,
                                     uint32_t*&                   d_queue,
                                     uint32_t*&                   d_queue_ptr,
                                     uint32_t*&                   d_ff_values,
                                     uint32_t*&                   d_ff_offset,
                                     void*&     d_cub_temp_storage_scan,
                                     void*&     d_cub_temp_storage_max,
                                     size_t&    cub_scan_bytes,
                                     size_t&    cub_max_bytes,
                                     uint32_t*& d_seeds,
                                     uint32_t*& d_new_num_patches,
                                     uint32_t*& d_max_patch_size,
                                     uint32_t*& d_patches_offset,
                                     uint32_t*& d_patches_size,
                                     uint32_t*& d_patches_val)
{
    CUDA_ERROR(
        cudaMalloc((void**)&d_ff_values, ff_values.size() * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMalloc((void**)&d_ff_offset, ff_offset.size() * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_ff_values,
                          ff_values.data(),
                          ff_values.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemcpy((void**)d_ff_offset,
                          ff_offset.data(),
                          ff_offset.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&d_face_patch,
                          m_num_top_simplices * sizeof(uint32_t)));

    CUDA_ERROR(
        cudaMalloc((void**)&d_seeds, m_max_num_patches * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_seeds,
                          seeds.data(),
                          m_num_patches * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    std::vector<uint32_t> h_queue_ptr{0, m_num_patches, m_num_patches};
    CUDA_ERROR(
        cudaMalloc((void**)&d_queue, m_num_top_simplices * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_queue_ptr, 3 * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_queue_ptr,
                          h_queue_ptr.data(),
                          3 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc((void**)&d_patches_offset,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patches_size,
                          m_max_num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patches_val,
                          m_num_top_simplices * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_max_patch_size, sizeof(uint32_t)));

    CUDA_ERROR(cudaMalloc((void**)&d_new_num_patches, sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy((void**)d_new_num_patches,
                          &m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    d_cub_temp_storage_scan = nullptr;
    d_cub_temp_storage_max  = nullptr;
    cub_scan_bytes          = 0;
    cub_max_bytes           = 0;
    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_scan_bytes,
                                    d_patches_size,
                                    d_patches_offset,
                                    m_max_num_patches);
    ::cub::DeviceReduce::Max(d_cub_temp_storage_max,
                             cub_max_bytes,
                             d_patches_size,
                             d_max_patch_size,
                             m_max_num_patches);
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_scan, cub_scan_bytes));
    CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage_max, cub_max_bytes));
}

void Patcher::run_lloyd(uint32_t* d_face_patch,
                        uint32_t* d_queue,
                        uint32_t* d_queue_ptr,
                        uint32_t* d_ff_values,
                        uint32_t* d_ff_offset,
                        void*     d_cub_temp_storage_scan,
                        void*     d_cub_temp_storage_max,
                        size_t    cub_scan_bytes,
                        size_t    cub_max_bytes,
                        uint32_t* d_seeds,
                        uint32_t* d_new_num_patches,
                        uint32_t* d_max_patch_size,
                        uint32_t* d_patches_offset,
                        uint32_t* d_patches_size,
                        uint32_t* d_patches_val)
{
    std::vector<uint32_t> h_queue_ptr{0, m_num_patches, m_num_patches};

    GPUTimer timer;
    timer.start();

    m_num_lloyd_run = 0;
    while (true) {
        ++m_num_lloyd_run;

        const uint32_t threads_s = 256;
        const uint32_t blocks_s  = DIVIDE_UP(m_num_patches, threads_s);
        const uint32_t threads_f = 256;
        const uint32_t blocks_f  = DIVIDE_UP(m_num_top_simplices, threads_f);

        if (m_num_lloyd_run % 5 == 0 && m_num_lloyd_run > 0) {
            uint32_t threshold = m_patch_size;

            CUDA_ERROR(cudaMemcpy(d_new_num_patches,
                                  &m_num_patches,
                                  sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            add_more_seeds<<<m_num_patches, 1>>>(m_num_patches,
                                                 d_new_num_patches,
                                                 d_seeds,
                                                 d_patches_offset,
                                                 d_patches_val,
                                                 threshold);

            CUDA_ERROR(cudaMemcpy(&m_num_patches,
                                  d_new_num_patches,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            if (m_num_patches >= m_max_num_patches) {
                RXMESH_ERROR(
                    "Patcher::run_lloyd() m_num_patches exceeds "
                    "m_max_num_patches");
                exit(EXIT_FAILURE);
            }
        }
        h_queue_ptr[0] = 0;
        h_queue_ptr[1] = m_num_patches;
        h_queue_ptr[2] = m_num_patches;
        CUDA_ERROR(cudaMemcpy(d_queue_ptr,
                              h_queue_ptr.data(),
                              3 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        rxmesh::memsett<<<blocks_f, threads_f>>>(
            d_face_patch, INVALID32, m_num_top_simplices);

        rxmesh::memcopy<<<blocks_s, threads_s>>>(
            d_queue, d_seeds, m_num_patches);

        rxmesh::memsett<<<blocks_s, threads_s>>>(
            d_patches_size, 0u, m_num_patches);

        write_initial_face_patch<<<blocks_s, threads_s>>>(
            m_num_patches, d_face_patch, d_seeds, d_patches_size);

        while (true) {
            cluster_seed_propagation<<<blocks_f, threads_f>>>(
                m_num_top_simplices,
                m_num_patches,
                d_queue_ptr,
                d_queue,
                d_face_patch,
                d_patches_size,
                d_ff_offset,
                d_ff_values);

            reset_queue_ptr<<<1, 1>>>(d_queue_ptr);

            CUDA_ERROR(cudaMemcpy(h_queue_ptr.data(),
                                  d_queue_ptr,
                                  sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            if (h_queue_ptr[0] >= m_num_top_simplices) {
                break;
            }
        }

        uint32_t max_patch_size =
            construct_patches_compressed_format(d_face_patch,
                                                d_cub_temp_storage_scan,
                                                d_cub_temp_storage_max,
                                                cub_scan_bytes,
                                                cub_max_bytes,
                                                d_max_patch_size,
                                                d_patches_offset,
                                                d_patches_size,
                                                d_patches_val);

        uint32_t threads_i   = 512;
        uint32_t shmem_bytes = max_patch_size * (sizeof(uint32_t));
        rxmesh::memsett<<<blocks_f, threads_f>>>(
            d_queue, INVALID32, m_num_top_simplices);
        interior<<<m_num_patches, threads_i, shmem_bytes>>>(m_num_patches,
                                                            d_patches_offset,
                                                            d_patches_val,
                                                            d_face_patch,
                                                            d_seeds,
                                                            d_ff_offset,
                                                            d_ff_values,
                                                            d_queue);

        if (max_patch_size <= m_patch_size) {
            shift<<<blocks_f, threads_f>>>(
                m_num_top_simplices, d_face_patch, d_patches_val);

            break;
        }
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    m_patching_time_ms = timer.elapsed_millis();

    m_num_seeds = m_num_patches;

    CUDA_ERROR(cudaMemcpy(m_top_simplex_patch.data(),
                          d_face_patch,
                          sizeof(uint32_t) * m_num_top_simplices,
                          cudaMemcpyDeviceToHost));
    m_patches_offset.resize(m_num_patches);
    CUDA_ERROR(cudaMemcpy(m_patches_offset.data(),
                          d_patches_offset,
                          sizeof(uint32_t) * m_num_patches,
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(m_patches_val.data(),
                          d_patches_val,
                          sizeof(uint32_t) * m_num_top_simplices,
                          cudaMemcpyDeviceToHost));
}

uint32_t Patcher::construct_patches_compressed_format(
    uint32_t* d_face_patch,
    void*     d_cub_temp_storage_scan,
    void*     d_cub_temp_storage_max,
    size_t    cub_scan_bytes,
    size_t    cub_max_bytes,
    uint32_t* d_max_patch_size,
    uint32_t* d_patches_offset,
    uint32_t* d_patches_size,
    uint32_t* d_patches_val)
{
    uint32_t       max_patch_size = 0;
    const uint32_t threads_s      = 256;
    const uint32_t blocks_s       = DIVIDE_UP(m_num_patches, threads_s);
    const uint32_t threads_f      = 256;
    const uint32_t blocks_f       = DIVIDE_UP(m_num_top_simplices, threads_f);

    max_patch_size = 0;
    ::cub::DeviceReduce::Max(d_cub_temp_storage_max,
                             cub_max_bytes,
                             d_patches_size,
                             d_max_patch_size,
                             m_num_patches);
    CUDA_ERROR(cudaMemcpy(&max_patch_size,
                          d_max_patch_size,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    ::cub::DeviceScan::InclusiveSum(d_cub_temp_storage_scan,
                                    cub_scan_bytes,
                                    d_patches_size,
                                    d_patches_offset,
                                    m_num_patches);
    rxmesh::memsett<<<blocks_s, threads_s>>>(d_patches_size, 0u, m_num_patches);

    construct_patches_compressed<<<blocks_f, threads_f>>>(m_num_top_simplices,
                                                          d_face_patch,
                                                          m_num_patches,
                                                          d_patches_offset,
                                                          d_patches_size,
                                                          d_patches_val);

    return max_patch_size;
}

}  // namespace patcher
}  // namespace rxmesh

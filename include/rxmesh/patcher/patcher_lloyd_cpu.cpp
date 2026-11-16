#include <algorithm>
#include <numeric>
#include <vector>

#include "rxmesh/patcher/patcher.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"

namespace rxmesh {
namespace patcher {

void Patcher::execute_lloyd(std::vector<uint32_t>&       seeds,
                            const std::vector<uint32_t>& ff_offset,
                            const std::vector<uint32_t>& ff_values)
{
    initialize_random_seeds(seeds, ff_offset, ff_values);
    seeds.resize(m_max_num_patches, INVALID32);

    std::vector<uint32_t> face_patch(m_num_top_simplices, INVALID32);
    std::vector<uint32_t> queue(m_num_top_simplices, INVALID32);
    std::vector<uint32_t> patches_offset(m_max_num_patches, 0);
    std::vector<uint32_t> patches_size(m_max_num_patches, 0);
    std::vector<uint32_t> patches_val(m_num_top_simplices, 0);
    std::vector<uint32_t> interior_visited(m_num_top_simplices, INVALID32);
    std::vector<uint32_t> s_queue;

    CPUTimer timer;
    timer.start();

    m_num_lloyd_run = 0;
    while (true) {
        ++m_num_lloyd_run;

        if (m_num_lloyd_run % 5 == 0 && m_num_lloyd_run > 0) {
            const uint32_t old_num_patches = m_num_patches;
            const uint32_t threshold       = m_patch_size;
            for (uint32_t p = 0; p < old_num_patches; ++p) {
                const uint32_t p_start = (p == 0) ? 0 : patches_offset[p - 1];
                const uint32_t p_end   = patches_offset[p];
                const uint32_t p_size  = p_end - p_start;
                if (p_size >= threshold) {
                    bool added = false;
                    for (uint32_t f = p_start; f < p_end; ++f) {
                        uint32_t face = patches_val[f];
                        if (face & 1u) {
                            seeds[m_num_patches++] = face >> 1;
                            added                  = true;
                            break;
                        }
                    }
                    if (!added) {
                        seeds[m_num_patches++] = patches_val[p_start] >> 1;
                    }
                }
            }
            if (m_num_patches >= m_max_num_patches) {
                RXMESH_ERROR(
                    "Patcher::execute_lloyd() m_num_patches exceeds "
                    "m_max_num_patches");
                exit(EXIT_FAILURE);
            }
        }

        std::fill(face_patch.begin(), face_patch.end(), INVALID32);
        std::fill(
            patches_size.begin(), patches_size.begin() + m_num_patches, 0u);

        for (uint32_t p = 0; p < m_num_patches; ++p) {
            uint32_t seed      = seeds[p];
            queue[p]           = seed;
            face_patch[seed]   = p << 1;
            patches_size[p]    = 1;
        }

        uint32_t queue_start = 0;
        uint32_t queue_end   = m_num_patches;
        uint32_t next_end    = m_num_patches;

        while (queue_start < m_num_top_simplices) {
            for (uint32_t q = queue_start; q < queue_end; ++q) {
                uint32_t face_id     = queue[q];
                uint32_t p           = face_patch[face_id] >> 1;
                uint32_t is_boundary = 0;

                for (uint32_t idx = ff_offset[face_id];
                     idx < ff_offset[face_id + 1];
                     ++idx) {
                    uint32_t n_face = ff_values[idx];
                    if (face_patch[n_face] == INVALID32) {
                        face_patch[n_face] = p << 1;
                        queue[next_end++]  = n_face;
                        patches_size[p]++;
                    } else if ((face_patch[n_face] >> 1) != p) {
                        is_boundary = 1u;
                    }
                }
                face_patch[face_id] = (p << 1) | is_boundary;
            }
            queue_start = queue_end;
            queue_end   = next_end;
        }

        uint32_t max_patch_size = 0;
        uint32_t running_offset = 0;
        for (uint32_t p = 0; p < m_num_patches; ++p) {
            max_patch_size    = std::max(max_patch_size, patches_size[p]);
            running_offset   += patches_size[p];
            patches_offset[p] = running_offset;
            patches_size[p]   = 0;
        }

        for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
            uint32_t p           = face_patch[f] >> 1;
            uint32_t is_boundary = face_patch[f] & 1u;
            uint32_t pos         = patches_size[p]++;
            if (p != 0) {
                pos += patches_offset[p - 1];
            }
            patches_val[pos] = (f << 1) | is_boundary;
        }

        std::fill(interior_visited.begin(), interior_visited.end(), INVALID32);
        s_queue.resize(max_patch_size);

        for (uint32_t p = 0; p < m_num_patches; ++p) {
            const uint32_t p_start = (p == 0) ? 0 : patches_offset[p - 1];
            const uint32_t p_end   = patches_offset[p];
            const uint32_t p_size  = p_end - p_start;

            uint32_t s_queue_size = 0;
            for (uint32_t i = 0; i < p_size; ++i) {
                uint32_t packed = patches_val[p_start + i];
                if (packed & 1u) {
                    uint32_t f          = packed >> 1;
                    s_queue[s_queue_size++] = f;
                    interior_visited[f] = 0;
                }
            }

            if (s_queue_size > 0) {
                uint32_t q_start = 0;
                uint32_t q_end   = 0;
                while (true) {
                    q_start = q_end;
                    q_end   = s_queue_size;
                    if (q_end == p_size) {
                        break;
                    }
                    for (uint32_t i = q_start; i < q_end; ++i) {
                        uint32_t f = s_queue[i];
                        for (uint32_t idx = ff_offset[f];
                             idx < ff_offset[f + 1];
                             ++idx) {
                            uint32_t nf = ff_values[idx];
                            if ((face_patch[nf] >> 1) == p &&
                                interior_visited[nf] == INVALID32) {
                                interior_visited[nf]    = p;
                                s_queue[s_queue_size++] = nf;
                            }
                        }
                    }
                }
                if (q_start != 0) {
                    seeds[p] = s_queue[q_start];
                }
            }
        }

        if (max_patch_size <= m_patch_size) {
            for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
                m_top_simplex_patch[f] = face_patch[f] >> 1;
                m_patches_val[f]       = patches_val[f] >> 1;
            }
            m_patches_offset.assign(
                patches_offset.begin(), patches_offset.begin() + m_num_patches);
            break;
        }
    }

    timer.stop();
    m_patching_time_ms = timer.elapsed_millis();
    m_num_seeds        = m_num_patches;
}

}  // namespace patcher
}  // namespace rxmesh

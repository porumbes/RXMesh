#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>

#include "rxmesh/patcher/patcher.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/util.h"

#include "metis.h"

namespace rxmesh {

namespace patcher {

Patcher::Patcher(std::string                                      filename,
                 MeshKind                                         mesh_kind,
                 const std::vector<uint32_t>&                     ff_offset,
                 const std::vector<uint32_t>&                     ff_values,
                 const std::vector<std::vector<uint32_t>>&        simplices,
                 const std::vector<std::array<uint32_t, 4>>&      tf,
                 const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                          uint32_t,
                                          detail::edge_key_hash>& edges_map,
                 const uint32_t                                   num_vertices,
                 const uint32_t                                   num_edges,
                 const uint32_t                                   num_faces)
    : m_mesh_kind(mesh_kind),
      m_patch_size(0),
      m_num_patches(0),
      m_num_vertices(num_vertices),
      m_num_edges(num_edges),
      m_num_faces(num_faces),
      m_num_top_simplices(static_cast<uint32_t>(simplices.size())),
      m_num_seeds(0),
      m_max_num_patches(0),
      m_num_components(0),
      m_num_lloyd_run(0),
      m_patching_time_ms(0.0)
{
    RXMESH_TRACE("Patcher: Reading {}", filename);

    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        RXMESH_ERROR("Patcher: Failed to open {}", filename);
        exit(EXIT_FAILURE);
    }

    cereal::PortableBinaryInputArchive archive(is);

    uint32_t magic = 0;
    archive(magic);
    if (magic != archive_magic) {
        RXMESH_ERROR("Patcher: {} uses an unsupported patcher archive format",
                     filename);
        exit(EXIT_FAILURE);
    }

    MeshKind archive_mesh_kind = MeshKind::Triangle;
    archive(archive_mesh_kind);
    if (archive_mesh_kind != mesh_kind) {
        RXMESH_ERROR("Patcher: {} was saved for a different mesh kind",
                     filename);
        exit(EXIT_FAILURE);
    }

    m_mesh_kind = archive_mesh_kind;
    archive(m_patch_size, m_top_simplex_patch);

    if (m_top_simplex_patch.size() != m_num_top_simplices) {
        RXMESH_ERROR(
            "Patcher: {} contains {} top simplices but the input has {}",
            filename,
            m_top_simplex_patch.size(),
            m_num_top_simplices);
        exit(EXIT_FAILURE);
    }

    uint32_t max_patch = 0;
    for (uint32_t p : m_top_simplex_patch) {
        if (p == INVALID32 || p >= m_num_top_simplices) {
            RXMESH_ERROR("Patcher: {} contains an invalid patch ID", filename);
            exit(EXIT_FAILURE);
        }
        max_patch = std::max(max_patch, p);
    }

    m_num_patches = max_patch + 1;
    std::vector<bool> patch_found(m_num_patches, false);
    for (uint32_t p : m_top_simplex_patch) {
        patch_found[p] = true;
    }
    if (std::find(patch_found.begin(), patch_found.end(), false) !=
        patch_found.end()) {
        RXMESH_ERROR("Patcher: {} contains an empty patch", filename);
        exit(EXIT_FAILURE);
    }

    m_num_seeds        = m_num_patches;
    m_max_num_patches  = m_num_patches;
    m_num_lloyd_run    = 0;
    m_patching_time_ms = 0.0;

    std::vector<uint32_t> seeds;
    allocate_memory(seeds);
    compute_inital_compressed_patches();

    std::vector<std::vector<uint32_t>> components;
    get_multi_components(components, ff_offset, ff_values);
    m_num_components = static_cast<uint32_t>(components.size());

    extract_ribbons(simplices);
    assign_patch(simplices, tf, edges_map);
    print_statistics();
}

Patcher::Patcher(MeshKind                                         mesh_kind,
                 uint32_t                                         patch_size,
                 const std::vector<uint32_t>&                     ff_offset,
                 const std::vector<uint32_t>&                     ff_values,
                 const std::vector<std::vector<uint32_t>>&        simplices,
                 const std::vector<std::array<uint32_t, 4>>&      tf,
                 const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                          uint32_t,
                                          detail::edge_key_hash>& edges_map,
                 const uint32_t                                   num_vertices,
                 const uint32_t                                   num_edges,
                 const uint32_t                                   num_faces,
                 bool                                             use_metis)
    : m_mesh_kind(mesh_kind),
      m_patch_size(patch_size),
      m_num_patches(0),
      m_num_vertices(num_vertices),
      m_num_edges(num_edges),
      m_num_faces(num_faces),
      m_num_top_simplices(static_cast<uint32_t>(simplices.size())),
      m_num_seeds(0),
      m_max_num_patches(0),
      m_num_components(0),
      m_num_lloyd_run(0),
      m_patching_time_ms(0.0)
{
    m_num_patches = m_num_top_simplices / m_patch_size +
                    ((m_num_top_simplices % m_patch_size) ? 1 : 0);

    m_max_num_patches = 5 * m_num_patches;
    m_num_seeds       = m_num_patches;

    std::vector<uint32_t> seeds;

    allocate_memory(seeds);

    if (m_num_patches <= 1) {
        m_patches_offset[0] = m_num_top_simplices;
        m_num_seeds         = 1;
        m_num_components    = 1;
        m_num_lloyd_run     = 0;
        for (uint32_t i = 0; i < m_num_top_simplices; ++i) {
            m_top_simplex_patch[i] = 0;
            m_patches_val[i]       = i;
        }
    } else {
        if (use_metis) {
            metis_kway(ff_offset, ff_values);
        } else {
            execute_lloyd(seeds, ff_offset, ff_values);
        }

        extract_ribbons(simplices);
    }

    assign_patch(simplices, tf, edges_map);

    print_statistics();
}

void Patcher::save(std::string filename)
{
    std::ofstream ss(filename, std::ios::binary);
    if (!ss.is_open()) {
        RXMESH_ERROR("Patcher: Failed to open {}", filename);
        exit(EXIT_FAILURE);
    }

    cereal::PortableBinaryOutputArchive archive(ss);
    uint32_t                            magic = archive_magic;
    archive(magic, m_mesh_kind, m_patch_size, m_top_simplex_patch);
}

Patcher::~Patcher()
{
}

void Patcher::allocate_memory(std::vector<uint32_t>& seeds)
{
    seeds.reserve(m_num_seeds);

    if (m_top_simplex_patch.empty()) {
        m_top_simplex_patch.resize(m_num_top_simplices, INVALID32);
    }

    if (m_mesh_kind == MeshKind::Tet) {
        m_face_patch.resize(m_num_faces, INVALID32);
    }

    m_vertex_patch.resize(m_num_vertices, INVALID32);
    m_edge_patch.resize(m_num_edges, INVALID32);

    m_patches_val.resize(m_num_top_simplices);
    m_patches_offset.resize(m_max_num_patches);

    m_ribbon_ext_offset.resize(m_max_num_patches, 0);
    m_ribbon_ext_val.clear();
    m_ribbon_ext_val.reserve(m_num_top_simplices);
}

void Patcher::calc_edge_cut(const std::vector<std::vector<uint32_t>>& simplices,
                            const std::vector<uint32_t>&              ff_offset,
                            const std::vector<uint32_t>&              ff_values)
{
    // given a graph where nodes represents faces in the mesh and two nodes
    // are connected in this graph if two faces share an edge, we calculate
    // the edge cut fo such a graph
    uint32_t face_edge_cut = 0;
    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        for (uint32_t i = ff_offset[f]; i < ff_offset[f + 1]; ++i) {
            uint32_t n = ff_values[i];
            if (f < n && m_top_simplex_patch[f] != m_top_simplex_patch[n]) {
                face_edge_cut++;
            }
        }
    }

    uint32_t vertex_edge_cut = 0;

    using EdgeMapT = std::unordered_map<std::pair<uint32_t, uint32_t>,
                                        uint32_t,
                                        detail::edge_key_hash>;

    EdgeMapT edges_map;
    uint32_t num_edges = 0;

    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        for (uint32_t i = 0; i < simplices[f].size(); ++i) {

            uint32_t v0 = simplices[f][i];
            uint32_t v1 = simplices[f][(i + 1) % simplices[f].size()];

            std::pair<uint32_t, uint32_t> edge = detail::edge_key(v0, v1);

            auto e_iter = edges_map.find(edge);

            if (e_iter == edges_map.end()) {
                uint32_t edge_id = num_edges++;
                edges_map.insert(std::make_pair(edge, edge_id));

                if (m_vertex_patch[v0] != m_vertex_patch[v1]) {
                    vertex_edge_cut++;
                }
            }
        }
    }

    RXMESH_INFO("Patcher: (Face) Edge Cut = {}, (Vertex) Edge Cut = {} ",
                face_edge_cut,
                vertex_edge_cut);
}

void Patcher::print_statistics()
{
    RXMESH_INFO("Patcher: num_patches = {}", m_num_patches);
    RXMESH_INFO("Patcher: patches_size = {}", m_patch_size);
    RXMESH_INFO("Patcher: num_components = {}", m_num_components);

    // patching time
    RXMESH_INFO("Patcher: Num lloyd run = {}", m_num_lloyd_run);
    RXMESH_INFO(
        "Patcher: Parallel patches construction time = {} (ms) and {} "
        "(ms/lloyd_run)",
        m_patching_time_ms,
        ((m_num_lloyd_run == 0) ? 0 :
                                  m_patching_time_ms / float(m_num_lloyd_run)));

    // max-min patch size
    uint32_t max_patch_size(0), min_patch_size(m_num_top_simplices),
        avg_patch_size(0);
    get_max_min_avg_patch_size(min_patch_size, max_patch_size, avg_patch_size);
    RXMESH_INFO(
        "Patcher: max_patch_size= {}, min_patch_size= {}, avg_patch_size= {}",
        max_patch_size,
        min_patch_size,
        avg_patch_size);

    RXMESH_INFO("Patcher: number external ribbon elements = {} ({:02.2f}%)",
                get_num_ext_ribbon_faces(),
                get_ribbon_overhead());
}

void Patcher::initialize_random_seeds(std::vector<uint32_t>&       seeds,
                                      const std::vector<uint32_t>& ff_offset,
                                      const std::vector<uint32_t>& ff_values)
{
    std::vector<std::vector<uint32_t>> components;
    get_multi_components(components, ff_offset, ff_values);

    m_num_components = components.size();
    if (m_num_components == 1) {
        initialize_random_seeds_single_component(seeds);
    } else {
        if (m_num_seeds <= m_num_components) {
            m_num_seeds = m_num_components;
            for (auto& comp : components) {
                generate_random_seed_from_component(seeds, comp, 1);
            }
        } else {
            uint32_t num_remaining_seeds      = m_num_seeds - m_num_components;
            uint32_t num_extra_seeds_inserted = 0;

            std::vector<size_t> component_order(components.size());
            fill_with_sequential_numbers(component_order.data(),
                                         component_order.size());
            std::sort(component_order.begin(),
                      component_order.end(),
                      [&components](const size_t& a, const size_t& b) {
                          return components[a].size() > components[b].size();
                      });

            for (size_t c = 0; c < component_order.size(); ++c) {

                std::vector<uint32_t>& comp = components[component_order[c]];

                uint32_t size   = comp.size();
                float    weight = static_cast<float>(size) /
                               static_cast<float>(m_num_top_simplices);
                uint32_t component_num_seeds = static_cast<uint32_t>(std::ceil(
                    weight * static_cast<float>(num_remaining_seeds)));

                num_extra_seeds_inserted += component_num_seeds;
                if (num_extra_seeds_inserted > num_remaining_seeds) {
                    if (num_extra_seeds_inserted - num_remaining_seeds >
                        component_num_seeds) {
                        component_num_seeds = 0;
                    } else {
                        component_num_seeds -=
                            (num_extra_seeds_inserted - num_remaining_seeds);
                    }
                }

                component_num_seeds += 1;
                generate_random_seed_from_component(
                    seeds, comp, component_num_seeds);
            }
        }
    }

    assert(m_num_patches == seeds.size());
}

void Patcher::initialize_random_seeds_single_component(
    std::vector<uint32_t>& seeds)
{
    std::vector<uint32_t> rand_num(m_num_top_simplices);
    fill_with_sequential_numbers(rand_num.data(), rand_num.size());
    random_shuffle(rand_num.data(), rand_num.size());
    seeds.resize(m_num_seeds);
    std::memcpy(seeds.data(), rand_num.data(), m_num_seeds * sizeof(uint32_t));
}

void Patcher::generate_random_seed_from_component(
    std::vector<uint32_t>& seeds,
    std::vector<uint32_t>& component,
    const uint32_t         num_seeds)
{
    uint32_t num_seeds_before = seeds.size();
    if (num_seeds < 1) {
        RXMESH_ERROR(
            "Patcher::generate_random_seed_in_component() num_seeds should be "
            "larger than 1");
    }

    random_shuffle(component.data(), component.size());
    seeds.resize(num_seeds_before + num_seeds);
    std::memcpy(seeds.data() + num_seeds_before,
                component.data(),
                num_seeds * sizeof(uint32_t));
}

void Patcher::get_multi_components(
    std::vector<std::vector<uint32_t>>& components,
    const std::vector<uint32_t>&        ff_offset,
    const std::vector<uint32_t>&        ff_values)
{
    std::vector<bool> visited(m_num_top_simplices, false);
    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        if (!visited[f]) {
            std::vector<uint32_t> current_component;
            current_component.reserve(static_cast<uint32_t>(
                static_cast<double>(m_num_top_simplices) / 10.0));

            current_component.push_back(f);
            visited[f] = true;

            std::queue<uint32_t> face_queue;
            face_queue.push(f);
            while (!face_queue.empty()) {
                uint32_t face = face_queue.front();
                face_queue.pop();
                uint32_t start = ff_offset[face];
                uint32_t end   = ff_offset[face + 1];
                for (uint32_t f = start; f < end; ++f) {
                    uint32_t n_face = ff_values[f];
                    if (!visited[n_face]) {
                        current_component.push_back(n_face);
                        face_queue.push(n_face);
                        visited[n_face] = true;
                    }
                }
            }

            components.push_back(current_component);
        }
    }
}

void Patcher::bfs(const std::vector<uint32_t>& ff_offset,
                  const std::vector<uint32_t>& ff_values)
{
    std::vector<uint32_t> bfs_patch_id(m_num_patches);

    std::vector<std::vector<uint32_t>> patch_neighbour;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        std::vector<uint32_t> np;
        for (uint32_t f = (p == 0) ? 0 : m_patches_offset[p - 1];
             f < m_patches_offset[p];
             ++f) {
            uint32_t face = m_patches_val[f];
            for (uint32_t n = ff_offset[face]; n < ff_offset[face + 1]; ++n) {
                uint32_t n_face  = ff_values[n];
                uint32_t n_patch = m_top_simplex_patch[n_face];
                if (n_patch != p) {
                    if (find_index(n_patch, np) ==
                        std::numeric_limits<uint32_t>::max()) {
                        np.push_back(n_patch);
                    }
                }
            }
        }
        patch_neighbour.push_back(np);
    }
    std::vector<uint32_t> qu(1, 0);
    qu.reserve(m_num_patches);
    for (uint32_t p = 0; p < qu.size(); p++) {
        uint32_t patch      = qu[p];
        bfs_patch_id[patch] = p;
        for (uint32_t i = 0; i < patch_neighbour[patch].size(); i++) {
            uint32_t pn = patch_neighbour[patch][i];
            if (find_index(pn, qu) == std::numeric_limits<uint32_t>::max()) {
                qu.push_back(pn);
            }
        }
    }
    std::fill(m_patches_offset.begin(), m_patches_offset.end(), 0);
    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        m_top_simplex_patch[f] = bfs_patch_id[m_top_simplex_patch[f]];
        m_patches_offset[m_top_simplex_patch[f]]++;
    }
    uint32_t acc = 0;
    for (uint32_t p = 0; p < m_num_patches; ++p) {
        acc += m_patches_offset[p];
        m_patches_offset[p] = acc;
    }
    std::vector<uint32_t> temp_offset(m_num_patches, 0);
    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        uint32_t p     = m_top_simplex_patch[f];
        uint32_t start = (p == 0) ? p : m_patches_offset[p - 1];
        m_patches_val[start + temp_offset[p]] = f;
        temp_offset[p]++;
    }
}

void Patcher::extract_ribbons(
    const std::vector<std::vector<uint32_t>>& simplices)
{
    std::vector<std::vector<uint32_t>> vertex_simplices(m_num_vertices);

    for (uint32_t s = 0; s < m_num_top_simplices; ++s) {
        for (uint32_t v : simplices[s]) {
            vertex_simplices[v].push_back(s);
        }
    }

    std::vector<uint32_t> ribbon;
    ribbon.reserve(m_num_top_simplices);
    m_ribbon_ext_val.clear();

    for (uint32_t p = 0; p < m_num_patches; ++p) {
        const uint32_t p_start = (p == 0) ? 0 : m_patches_offset[p - 1];
        const uint32_t p_end   = m_patches_offset[p];

        ribbon.clear();

        for (uint32_t s = p_start; s < p_end; ++s) {
            const uint32_t simplex = m_patches_val[s];

            for (uint32_t vertex : simplices[simplex]) {
                for (uint32_t neighbor : vertex_simplices[vertex]) {
                    if (get_top_simplex_patch_id(neighbor) != p &&
                        std::find(ribbon.begin(), ribbon.end(), neighbor) ==
                            ribbon.end()) {
                        ribbon.push_back(neighbor);
                    }
                }
            }
        }

        m_ribbon_ext_val.insert(
            m_ribbon_ext_val.end(), ribbon.begin(), ribbon.end());
        m_ribbon_ext_offset[p] = static_cast<uint32_t>(m_ribbon_ext_val.size());
    }
}

void Patcher::assign_patch(
    const std::vector<std::vector<uint32_t>>&        simplices,
    const std::vector<std::array<uint32_t, 4>>&      tf,
    const std::unordered_map<std::pair<uint32_t, uint32_t>,
                             uint32_t,
                             detail::edge_key_hash>& edges_map)
{
    for (uint32_t cur_p = 0; cur_p < m_num_patches; ++cur_p) {
        const uint32_t p_start = (cur_p == 0) ? 0 : m_patches_offset[cur_p - 1];
        const uint32_t p_end   = m_patches_offset[cur_p];

        for (uint32_t s = p_start; s < p_end; ++s) {
            const uint32_t simplex = m_patches_val[s];

            if (m_mesh_kind == MeshKind::Triangle) {
                uint32_t v1 = simplices[simplex].back();
                for (uint32_t v0 : simplices[simplex]) {
                    const auto     edge    = detail::edge_key(v0, v1);
                    const uint32_t edge_id = edges_map.at(edge);

                    if (m_vertex_patch[v0] == INVALID32) {
                        m_vertex_patch[v0] = cur_p;
                    }
                    if (m_edge_patch[edge_id] == INVALID32) {
                        m_edge_patch[edge_id] = cur_p;
                    }
                    v1 = v0;
                }
            } else {
                const auto& tet = simplices[simplex];

                for (uint32_t vertex : tet) {
                    if (m_vertex_patch[vertex] == INVALID32) {
                        m_vertex_patch[vertex] = cur_p;
                    }
                }

                for (const auto& edge : tet_edges()) {
                    const auto key =
                        detail::edge_key(tet[edge[0]], tet[edge[1]]);
                    const uint32_t edge_id = edges_map.at(key);
                    if (m_edge_patch[edge_id] == INVALID32) {
                        m_edge_patch[edge_id] = cur_p;
                    }
                }

                for (uint32_t packed_face : tf[simplex]) {
                    const uint32_t face_id = packed_face >> 1;
                    if (m_face_patch[face_id] == INVALID32) {
                        m_face_patch[face_id] = cur_p;
                    }
                }
            }
        }
    }
}

void Patcher::metis_kway(const std::vector<uint32_t>& ff_offset,
                         const std::vector<uint32_t>& ff_values)
{
    std::vector<idx_t> xadj(ff_offset.size());
    std::vector<idx_t> adjncy(ff_values.size());

    for (uint32_t i = 0; i < ff_offset.size(); ++i) {
        xadj[i] = ff_offset[i];
    }

    for (uint32_t i = 0; i < ff_values.size(); ++i) {
        adjncy[i] = ff_values[i];
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] =
        METIS_OBJTYPE_VOL;  // Total communication volume minimization.
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = METIS_DBG_TIME;

    // number of vertices in the graph
    idx_t              nvtxs  = m_num_top_simplices;
    idx_t              ncon   = 1;
    idx_t*             vwgt   = NULL;
    idx_t*             vsize  = NULL;
    idx_t*             adjwgt = NULL;
    idx_t              nparts = DIVIDE_UP(m_num_top_simplices, m_patch_size);
    real_t*            tpwgts = NULL;
    real_t*            ubvec  = NULL;
    idx_t              objval = 0;
    std::vector<idx_t> part(nvtxs, 0);

    CPUTimer timer;
    timer.start();

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           xadj.data(),
                                           adjncy.data(),
                                           vwgt,
                                           vsize,
                                           adjwgt,
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           part.data());
    timer.stop();
    m_patching_time_ms = timer.elapsed_millis();

    if (metis_status == METIS_ERROR_INPUT) {
        RXMESH_ERROR("METIS ERROR INPUT");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR_MEMORY) {
        RXMESH_ERROR("\n METIS ERROR MEMORY \n");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR) {
        RXMESH_ERROR("\n METIS ERROR\n");
        exit(EXIT_FAILURE);
    }

    m_num_patches = nparts;

    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        m_top_simplex_patch[f] = part[f];
    }

    compute_inital_compressed_patches();
}

void Patcher::compute_inital_compressed_patches()
{
    m_patches_offset.resize(m_num_patches, 0);

    std::vector<uint32_t> patches_size(m_num_patches, 0);
    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        patches_size[m_top_simplex_patch[f]]++;
    }

    std::inclusive_scan(
        patches_size.begin(), patches_size.end(), m_patches_offset.begin());

    if (m_patches_offset.back() != m_num_top_simplices) {
        RXMESH_ERROR(
            "Patcher::compute_inital_compressed_patches()  Error with creating "
            "patch graph");
        exit(EXIT_FAILURE);
    }

    std::fill(patches_size.begin(), patches_size.end(), 0);

    for (uint32_t f = 0; f < m_num_top_simplices; ++f) {
        int p = m_top_simplex_patch[f];

        uint32_t id = (p == 0) ? 0 : m_patches_offset[p - 1];

        id += patches_size[p]++;

        m_patches_val[id] = f;
    }
}

}  // namespace patcher
}  // namespace rxmesh

#include "gtest/gtest.h"

#include <cstdio>
#include <filesystem>
#include <set>
#include <vector>

#include "rxmesh/rxmesh.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/msh_io.h"

namespace {

using namespace rxmesh;

class HostRXMesh : public RXMesh
{
   public:
    explicit HostRXMesh(const std::vector<std::vector<uint32_t>>& simplices,
                        const std::string& patcher_file = "",
                        const uint32_t     patch_size   = 512)
        : RXMesh(patch_size)
    {
        if (!simplices.empty() && simplices[0].size() == 4) {
            m_is_tet_mesh = true;
        }
        init(simplices, patcher_file);
    }

    const PatchInfo* get_device_patches_info() const
    {
        return m_d_patches_info;
    }

    void validate_host_and_slab_topology() const
    {
        uint32_t owned_v = 0;
        uint32_t owned_e = 0;
        uint32_t owned_f = 0;
        uint32_t owned_t = 0;

        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            const PatchInfo& pi = get_patch(p);
            EXPECT_EQ(pi.patch_id, p);
            EXPECT_LT(pi.color, get_num_colors());

            PatchInfo d_pi;
            CUDA_ERROR(cudaMemcpy(&d_pi,
                                  m_d_patches_info + p,
                                  sizeof(PatchInfo),
                                  cudaMemcpyDeviceToHost));
            EXPECT_EQ(d_pi.patch_id, p);
            EXPECT_EQ(d_pi.color, pi.color);
            EXPECT_EQ(d_pi.vertices_capacity, pi.vertices_capacity);
            EXPECT_EQ(d_pi.edges_capacity, pi.edges_capacity);
            EXPECT_EQ(d_pi.faces_capacity, pi.faces_capacity);
            EXPECT_EQ(d_pi.tets_capacity, pi.tets_capacity);

            uint16_t d_num_v = 0, d_num_e = 0, d_num_f = 0;
            CUDA_ERROR(cudaMemcpy(&d_num_v,
                                  d_pi.num_vertices,
                                  sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(&d_num_e,
                                  d_pi.num_edges,
                                  sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(&d_num_f,
                                  d_pi.num_faces,
                                  sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost));
            EXPECT_EQ(d_num_v, pi.num_vertices[0]);
            EXPECT_EQ(d_num_e, pi.num_edges[0]);
            EXPECT_EQ(d_num_f, pi.num_faces[0]);

            const uint16_t num_v = pi.num_vertices[0];
            const uint16_t num_e = pi.num_edges[0];
            const uint16_t num_f = pi.num_faces[0];
            const uint16_t num_t = m_is_tet_mesh ? pi.num_tets[0] : 0;

            EXPECT_LE(num_v, pi.vertices_capacity);
            EXPECT_LE(num_e, pi.edges_capacity);
            EXPECT_LE(num_f, pi.faces_capacity);
            if (m_is_tet_mesh) {
                EXPECT_LE(num_t, pi.tets_capacity);
            }

            // 1. Validate vertices and ribbon LPHashTable lookup
            for (uint16_t v = 0; v < num_v; ++v) {
                const LocalVertexT vl(v);
                if (pi.is_deleted(vl)) {
                    continue;
                }
                if (pi.is_owned(vl)) {
                    ++owned_v;
                } else {
                    const VertexHandle owner_vh = pi.find<VertexHandle>(v);
                    ASSERT_TRUE(owner_vh.is_valid());
                    EXPECT_NE(owner_vh.patch_id(), p);
                    ASSERT_LT(owner_vh.patch_id(), get_num_patches());
                    const PatchInfo& owner_pi = get_patch(owner_vh.patch_id());
                    const LocalVertexT owner_vl(owner_vh.local_id());
                    EXPECT_FALSE(owner_pi.is_deleted(owner_vl));
                    EXPECT_TRUE(owner_pi.is_owned(owner_vl));
                }
            }

            // 2. Validate edges (uniqueness of endpoints + ribbon lookup)
            for (uint16_t e = 0; e < num_e; ++e) {
                const LocalEdgeT el(e);
                if (pi.is_deleted(el)) {
                    continue;
                }
                const uint16_t v0 = pi.ev[2 * e + 0].id;
                const uint16_t v1 = pi.ev[2 * e + 1].id;
                EXPECT_LT(v0, num_v);
                EXPECT_LT(v1, num_v);
                EXPECT_NE(v0, v1);

                if (pi.is_owned(el)) {
                    ++owned_e;
                } else {
                    const EdgeHandle owner_eh = pi.find<EdgeHandle>(e);
                    ASSERT_TRUE(owner_eh.is_valid());
                    EXPECT_NE(owner_eh.patch_id(), p);
                    ASSERT_LT(owner_eh.patch_id(), get_num_patches());
                    const PatchInfo& owner_pi = get_patch(owner_eh.patch_id());
                    const LocalEdgeT owner_el(owner_eh.local_id());
                    EXPECT_FALSE(owner_pi.is_deleted(owner_el));
                    EXPECT_TRUE(owner_pi.is_owned(owner_el));

                    // Verify ribbon edge endpoints resolve to the exact same
                    // global owner vertices as in the owner patch
                    const VertexHandle p_v0 = pi.find<VertexHandle>(v0);
                    const VertexHandle p_v1 = pi.find<VertexHandle>(v1);
                    const uint16_t     ov0 =
                        owner_pi.ev[2 * owner_el.id + 0].id;
                    const uint16_t     ov1 =
                        owner_pi.ev[2 * owner_el.id + 1].id;
                    const VertexHandle o_v0 = owner_pi.find<VertexHandle>(ov0);
                    const VertexHandle o_v1 = owner_pi.find<VertexHandle>(ov1);
                    EXPECT_EQ(p_v0, o_v0);
                    EXPECT_EQ(p_v1, o_v1);
                }
            }

            // 3. Validate faces (3 unique edges, 3 unique vertices, ribbon)
            for (uint16_t f = 0; f < num_f; ++f) {
                const LocalFaceT fl(f);
                if (pi.is_deleted(fl)) {
                    continue;
                }
                const uint16_t e0 = pi.fe[3 * f + 0].id >> 1;
                const uint16_t e1 = pi.fe[3 * f + 1].id >> 1;
                const uint16_t e2 = pi.fe[3 * f + 2].id >> 1;
                EXPECT_LT(e0, num_e);
                EXPECT_LT(e1, num_e);
                EXPECT_LT(e2, num_e);
                EXPECT_NE(e0, e1);
                EXPECT_NE(e1, e2);
                EXPECT_NE(e2, e0);

                if (pi.is_owned(fl)) {
                    ++owned_f;
                } else {
                    const FaceHandle owner_fh = pi.find<FaceHandle>(f);
                    ASSERT_TRUE(owner_fh.is_valid());
                    EXPECT_NE(owner_fh.patch_id(), p);
                    ASSERT_LT(owner_fh.patch_id(), get_num_patches());
                    const PatchInfo& owner_pi = get_patch(owner_fh.patch_id());
                    const LocalFaceT owner_fl(owner_fh.local_id());
                    EXPECT_FALSE(owner_pi.is_deleted(owner_fl));
                    EXPECT_TRUE(owner_pi.is_owned(owner_fl));
                }
            }

            // 4. Validate tets if tetrahedral mesh
            for (uint16_t t = 0; t < num_t; ++t) {
                const LocalTetT tl(t);
                if (pi.is_deleted(tl)) {
                    continue;
                }
                if (pi.is_owned(tl)) {
                    ++owned_t;
                } else {
                    const TetHandle owner_th = pi.find<TetHandle>(t);
                    ASSERT_TRUE(owner_th.is_valid());
                    EXPECT_NE(owner_th.patch_id(), p);
                    ASSERT_LT(owner_th.patch_id(), get_num_patches());
                    const PatchInfo& owner_pi = get_patch(owner_th.patch_id());
                    const LocalTetT  owner_tl(owner_th.local_id());
                    EXPECT_FALSE(owner_pi.is_deleted(owner_tl));
                    EXPECT_TRUE(owner_pi.is_owned(owner_tl));
                }
            }
        }

        EXPECT_EQ(owned_v, get_num_vertices());
        EXPECT_EQ(owned_e, get_num_edges());
        EXPECT_EQ(owned_f, get_num_faces());
        if (m_is_tet_mesh) {
            EXPECT_EQ(owned_t, get_num_tets());
        }
    }
};

}  // namespace

TEST(RXMeshSlice1, TriangleMeshInitAndTopologyValidation)
{
    std::vector<std::vector<rx_coord_t>> verts;
    std::vector<std::vector<uint32_t>>   fv;
    ASSERT_TRUE(import_obj(STRINGIFY(INPUT_DIR) "dragon.obj", verts, fv));

    HostRXMesh rx(fv);
    EXPECT_EQ(rx.get_num_vertices(), 10000u);
    EXPECT_EQ(rx.get_num_faces(), 20000u);
    EXPECT_EQ(rx.get_num_edges(), 30000u);
    EXPECT_GT(rx.get_num_patches(), 1u);
    EXPECT_GT(rx.get_num_colors(), 1u);

    rx.validate_host_and_slab_topology();

    // Test Patcher save & load round-trip reproducibility
    const std::string patch_archive =
        STRINGIFY(OUTPUT_DIR) "slice1_dragon_patches.bin";
    rx.save(patch_archive);

    HostRXMesh rx_reloaded(fv, patch_archive);
    EXPECT_EQ(rx_reloaded.get_num_patches(), rx.get_num_patches());
    EXPECT_EQ(rx_reloaded.get_num_vertices(), rx.get_num_vertices());
    EXPECT_EQ(rx_reloaded.get_num_edges(), rx.get_num_edges());
    EXPECT_EQ(rx_reloaded.get_num_faces(), rx.get_num_faces());
    rx_reloaded.validate_host_and_slab_topology();

    std::filesystem::remove(patch_archive);
}

TEST(RXMeshSlice1, BoundaryMeshInitAndValidation)
{
    std::vector<std::vector<rx_coord_t>> verts;
    std::vector<std::vector<uint32_t>>   fv;
    ASSERT_TRUE(import_obj(STRINGIFY(INPUT_DIR) "bunnyhead.obj", verts, fv));

    HostRXMesh rx(fv);
    EXPECT_EQ(rx.get_num_vertices(), 1550u);
    EXPECT_EQ(rx.get_num_faces(), 3000u);
    EXPECT_EQ(rx.get_num_edges(), 4549u);
    EXPECT_FALSE(rx.is_closed());
    EXPECT_TRUE(rx.is_edge_manifold());

    rx.validate_host_and_slab_topology();
}

TEST(RXMeshSlice1, TetMeshInitAndValidation)
{
    std::vector<std::vector<rx_coord_t>> verts;
    std::vector<std::vector<uint32_t>>   tv;
    ASSERT_EQ(load_msh(STRINGIFY(INPUT_DIR) "car.msh", verts, tv),
              MeshKind::Tet);

    HostRXMesh rx(tv);
    EXPECT_GT(rx.get_num_vertices(), 0u);
    EXPECT_GT(rx.get_num_edges(), 0u);
    EXPECT_GT(rx.get_num_faces(), 0u);
    EXPECT_EQ(rx.get_num_tets(), static_cast<uint32_t>(tv.size()));

    rx.validate_host_and_slab_topology();
}

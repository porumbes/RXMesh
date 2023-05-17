#pragma once
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

class RXMeshDynamic : public RXMeshStatic
{
   public:
    RXMeshDynamic(const RXMeshDynamic&) = delete;

    /**
     * @brief Constructor using path to obj file
     * @param file_path path to an obj file
     * @param quite run in quite mode
     */
    RXMeshDynamic(const std::string file_path,
                  const bool        quite        = false,
                  const std::string patcher_file = "")
        : RXMeshStatic(file_path, quite, patcher_file)
    {
    }

    /**
     * @brief Constructor using triangles and vertices
     * @param fv Face incident vertices as read from an obj file
     * @param quite run in quite mode
     */
    RXMeshDynamic(std::vector<std::vector<uint32_t>>& fv,
                  const bool                          quite        = false,
                  const std::string                   patcher_file = "")
        : RXMeshStatic(fv, quite, patcher_file)
    {
    }

    /**
     * @brief save/seralize the patcher info to a file
     * @param filename
     */
    virtual void save(std::string filename) override;

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for a kernel that may use dynamic and query operations
     * @param op List of query operations done inside the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const
    {

        launch_box.blocks = this->m_num_patches;

        size_t static_shmem = 0;
        for (auto o : op) {
            static_shmem =
                std::max(static_shmem,
                         this->calc_shared_memory<blockThreads>(o, oriented));
        }

        uint16_t vertex_cap = static_cast<uint16_t>(
            this->m_capacity_factor *
            static_cast<float>(this->m_max_vertices_per_patch));

        uint16_t edge_cap = static_cast<uint16_t>(
            this->m_capacity_factor *
            static_cast<float>(this->m_max_edges_per_patch));

        uint16_t face_cap = static_cast<uint16_t>(
            this->m_capacity_factor *
            static_cast<float>(this->m_max_faces_per_patch));

        // To load EV and FE
        size_t dyn_shmem = 3 * face_cap * sizeof(uint16_t) +
                           2 * edge_cap * sizeof(uint16_t) +
                           2 * ShmemAllocator::default_alignment;

        // cavity ID
        dyn_shmem += std::max(
            vertex_cap * sizeof(uint16_t),
            max_lp_hashtable_capacity<LocalVertexT>() * sizeof(LPPair));
        dyn_shmem +=
            std::max(edge_cap * sizeof(uint16_t),
                     max_lp_hashtable_capacity<LocalEdgeT>() * sizeof(LPPair));
        dyn_shmem +=
            std::max(face_cap * sizeof(uint16_t),
                     max_lp_hashtable_capacity<LocalFaceT>() * sizeof(LPPair));

        dyn_shmem += 3 * ShmemAllocator::default_alignment;

        // cavity loop
        dyn_shmem += this->m_max_edges_per_patch * sizeof(uint16_t) +
                     ShmemAllocator::default_alignment;

        // store number of cavities and patches to lock
        dyn_shmem += 3 * sizeof(int) + ShmemAllocator::default_alignment;


        // store cavity size (assume number of cavities is half the patch size)
        dyn_shmem += (this->m_max_faces_per_patch / 2) * sizeof(int) +
                     ShmemAllocator::default_alignment;

        // active, owned, migrate(for vertices only), src bitmask (for vertices
        // and edges only), src connect (for vertices and edges only), ownership
        // owned_cavity_bdry (for vertices only), ribbonize (for vertices only)
        // added_to_lp, in_cavity
        dyn_shmem += 10 * detail::mask_num_bytes(vertex_cap) +
                     10 * ShmemAllocator::default_alignment;
        dyn_shmem += 7 * detail::mask_num_bytes(edge_cap) +
                     7 * ShmemAllocator::default_alignment;
        dyn_shmem += 5 * detail::mask_num_bytes(face_cap) +
                     5 * ShmemAllocator::default_alignment;

        //patch stash 
        dyn_shmem += PatchStash::stash_size * sizeof(uint32_t);

        if (!this->m_quite) {
            RXMESH_TRACE(
                "RXMeshDynamic::calc_shared_memory() launching {} blocks with "
                "{} threads on the device",
                launch_box.blocks,
                blockThreads);
        }

        // since we are either doing static query or dynamic changes,
        // shared memory is the max of both
        launch_box.smem_bytes_dyn = std::max(dyn_shmem, static_shmem);

        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            blockThreads,
                            kernel);
    }

    virtual ~RXMeshDynamic() = default;

    /**
     * @brief check if there is remaining patches not processed yet
     */
    bool is_queue_empty(cudaStream_t stream = NULL)
    {
        return this->m_rxmesh_context.m_patch_scheduler.is_empty(stream);
    }


    /**
     * @brief reset the patches for a another kernel. This needs only to be
     * called where more than one kernel is called. For a single kernel, the
     * queue is initialized during the construction so the user does not to call
     * this
     */
    void reset_queue()
    {
        this->m_rxmesh_context.m_patch_scheduler.refill();
    }

    /**
     * @brief Validate the topology information stored in RXMesh. All checks are
     * done on the information stored on the GPU memory and thus all checks are
     * done on the GPU
     * @return true in case all information stored are valid
     */
    bool validate();

    /**
     * @brief cleanup after topology changes by removing surplus elements
     * and make sure that hashtable store owner patches
     */
    void cleanup();

    /**
     * @brief slice a patch if the number of faces in the patch is greater
     * than a threshold
     */
    void slice_patches(const uint32_t                num_faces_threshold,
                       rxmesh::FaceAttribute<int>&   f_attr,
                       rxmesh::EdgeAttribute<int>&   e_attr,
                       rxmesh::VertexAttribute<int>& v_attr);

    void copy_patch_debug(const uint32_t                  pid,
                          rxmesh::VertexAttribute<float>& coords);

    /**
     * @brief update the host side. Use this function to update the host side
     * after performing (dynamic) updates on the GPU. This function may
     * re-allocates the host side memory buffers in case it is not enough (e.g.,
     * after performing mesh refinement on the GPU)
     */
    void update_host();

    /**
     * @brief update polyscope after performing dynamic changes. This function
     * is supposed to be called after a call to update_host since polyscope
     * reads information from the host side of RXMesh which include the topology
     * (stored in RXMesh/RXMeshStatic/RXMeshDynamic) and the input vertex
     * coordinates as well. Thus, a call to `move(DEVICE, HOST)` should be done
     * to RXMesh-stored vertex coordinates before calling this function.
     */
    void update_polyscope();
};
}  // namespace rxmesh
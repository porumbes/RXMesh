# RXMesh Agent Guidelines (`AGENTS.md`)

RXMesh is a C++17 / CUDA library for high-performance triangle (and tetrahedral) mesh processing on the GPU, featuring patch-based static and dynamic mesh representations, tightly coupled sparse/dense matrix solvers (`cuSolver`, `cuSparse`, `cuBLAS`, `cuDSS`), and GPU-accelerated Automatic Differentiation (AD).

---

## 1. Repository Layout

- `include/rxmesh/`: Core library headers and compiled `.cu`/`.cpp` sources (all in namespace `rxmesh`).
  - **Core Mesh Hierarchy**:
    - `rxmesh.h` / `rxmesh.cpp`: Base `RXMesh` class managing patch decomposition, topology, and device context (`Context`). Not instantiated directly by users.
    - `rxmesh_static.h` / `rxmesh_static.cu` / `rxmesh_static.inl`: `RXMeshStatic` for static mesh queries (`for_each`, `run_query_kernel`), attribute management, and Polyscope registration.
    - `rxmesh_dynamic.h` / `rxmesh_dynamic.cu`: `RXMeshDynamic` extending `RXMeshStatic` for dynamic topology modifications (edge split/flip/collapse, cavity operations, patch slicing).
  - **Handles, Attributes & Reductions**:
    - `handle.h`: Element handles (`VertexHandle`, `EdgeHandle`, `FaceHandle`, `TetHandle`) encoding `(patch_id, local_id)`.
    - `attribute.h` / `attribute.cu`: `Attribute<T, HandleT>` and `VertexAttribute`/`EdgeAttribute`/`FaceAttribute`/`TetAttribute` containers across `HOST` and `DEVICE`.
    - `reduce_handle.h` / `reduce_handle.cu`: `ReduceHandle` for GPU reductions (dot products, norms, min/max) over mesh attributes.
  - **Queries & Kernels**:
    - `query.h` / `query.cu` / `query.inl`: Host/device query orchestration for topological queries (`Op::VV`, `Op::VE`, `Op::VF`, `Op::FV`, `Op::FE`, `Op::FF`, `Op::EV`, `Op::EE`, `Op::EF`, `Op::EVDiamond`, and tet variants).
    - `kernels/`: Device-side query dispatchers (`query_dispatcher.cuh`, `rxmesh_queries.cuh`), shared-memory allocators (`shmem_allocator.cuh`), iterators, and collective operations.
    - `cavity_manager.cuh` / `cavity_manager_impl.cuh`: Device-side cavity creation and reconnection for dynamic remeshing.
  - **Patching**:
    - `patcher/`: Mesh partitioning into small patches (`PatchInfo`, `PatchStash`, `LPHashTable`) that fit in GPU shared memory.
  - **Linear Algebra (`matrix/`)**:
    - `sparse_matrix.h`, `dense_matrix.h`: Mesh-coupled GPU matrices.
    - Direct & iterative solvers: `cholesky_solver.h`, `cudss_cholesky_solver.h`, `lu_solver.h`, `qr_solver.h`, `cg_solver.h`, `pcg_solver.h`, `cg_mat_free_solver.h`, `gmg_solver.h` (geometric multigrid).
  - **Automatic Differentiation (`diff/`)**:
    - `scalar.h`: Forward/reverse-mode active scalar type (`Scalar<T, k, WithHessian>`) for local per-element energy differentiation.
    - `diff_scalar_problem.h`, `diff_vector_problem.h`: Global energy assembly (gradients and sparse Hessians/Jacobians).
    - Nonlinear optimizers: `newton_solver.h`, `gauss_newton_solver.h`, `lbfgs_solver.h`, `gradient_descent.h`.
  - **Utilities (`util/`)**:
    - `macros.h` (error checking macros `CUDA_ERROR`, `CUSPARSE_ERROR`, `CUSOLVER_ERROR`, `CUBLAS_ERROR`, `myAssert`), `log.h` (`RXMESH_INFO`, `RXMESH_WARN`, `RXMESH_ERROR`), `timer.h`, `report.h`.
- `tests/`: Unit test suites (`RXMesh_test` and `Polyscope_test`) built with GoogleTest.
- `apps/`: Standalone geometry processing, simulation, and remeshing applications (e.g., `ARAP`, `DiffARAP`, `Heat`, `MCF`, `NeoHookean`, `Parameterization`, `Remesh`, `SECPriority`, `SurfaceTracking`, `TetDeformation`).
- `cmake/`: CMake modules (`RXMeshTarget.cmake`, `RXMeshApp.cmake`, `ThirdParty.cmake`, `CPM.cmake`).
- `input/` & `output/`: Default mesh input directory (`INPUT_DIR` compile definition) and output directory (`OUTPUT_DIR` compile definition).

---

## 2. Build & Configuration

- **Minimum Requirements**: CMake >= 3.25, C++17 and CUDA 17 compiler (`nvcc` with separable compilation `-rdc=true`, extended lambdas `--expt-extended-lambda`, and `--expt-relaxed-constexpr`).
- **CUDA Architectures**: Device LTO (`rxmesh_enable_cuda_device_lto`) requires numeric `CMAKE_CUDA_ARCHITECTURES` (or `native` when an NVIDIA GPU is present on the build host). When configuring without a local NVIDIA GPU or cross-compiling, explicitly pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` (e.g., `89`).
- **Key CMake Options**:
  - `RX_USE_POLYSCOPE` (default `ON`): Enable Polyscope visualization (`USE_POLYSCOPE`).
  - `RX_BUILD_TESTS` (default `ON`): Build unit tests in `tests/`.
  - `RX_BUILD_APPS` (default `ON`): Build applications in `apps/`.
  - `RX_USE_CUDSS` (default `OFF`): Enable NVIDIA cuDSS direct solver support (`USE_CUDSS`).
  - `RX_USE_SUITESPARSE` (default `OFF`): Enable SuiteSparse for benchmarks (`USE_SUITESPARSE`).
  - `RX_USE_DOUBLE` (default `OFF`): Use `double` instead of `float` for `rx_coord_t` (`RXMESH_USE_DOUBLE`).
- **Standard Configure & Build Commands**:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build build -j
  ```
  All compiled binaries are emitted to `build/bin/`.
- **Adding New Applications or Tests**:
  Use `rxmesh_add_app(<TargetName> SOURCES ... [LIBS ...] [FOLDER ...])` defined in `cmake/RXMeshApp.cmake`, which automatically configures `CUDA_SEPARABLE_COMPILATION`, device LTO, and links against `RXMesh`.

---

## 3. Running Tests

- Run all tests via CTest from the build directory:
  ```bash
  ctest --test-dir build --output-on-failure
  ```
- Run specific unit tests directly via the GoogleTest binary in `build/bin/RXMesh_test`:
  ```bash
  ./build/bin/RXMesh_test --gtest_filter="RXMeshStatic.*" -d 0
  ```
  `RXMesh_test` accepts CLI11 flags `-i,--input <mesh.obj>`, `-d,--device_id <id>`, and `-o,--output <dir>` alongside standard `--gtest_*` flags.

---

## 4. Coding Style & Conventions

- **Formatting**: Enforced via `.clang-format` (Chromium base, 4-space indentation, no tabs, **80-character column limit**, custom brace wrapping where classes/structs/enums/functions place opening braces on a new line while control statements keep braces on the same line, `AlwaysBreakTemplateDeclarations: true`).
- **Error Handling & Logging**:
  - Wrap all CUDA runtime and library calls in `CUDA_ERROR(...)`, `CUSPARSE_ERROR(...)`, `CUSOLVER_ERROR(...)`, `CUBLAS_ERROR(...)`, or `CUDSS_ERROR(...)` from `rxmesh/util/macros.h`.
  - Use `GPU_FREE(ptr)` for safe device memory deallocation.
  - Use `myAssert(cond)` inside device/host code and `RXMESH_INFO(...)` / `RXMESH_WARN(...)` / `RXMESH_ERROR(...)` for logging instead of raw `std::cout`.
- **CUDA & Kernel Launching**:
  - Initialize the runtime via `rx_init(device_id)` before creating `RXMeshStatic` or `RXMeshDynamic` instances.
  - Prepare kernel launch parameters (threads, blocks, dynamic shared memory size) on the host using `LaunchBox<blockThreads>` (`rxmesh.prepare_launch_box(...)`) before launching device kernels that perform mesh queries or cavity operations.
  - Prefer `rxmesh.for_each_vertex` / `for_each_edge` / `for_each_face` with `__device__` lambdas for simple element-wise operations, and `Context` + `Query<blockThreads>` inside `__global__` kernels for neighborhood queries.


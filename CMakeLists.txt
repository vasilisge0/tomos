cmake_minimum_required(VERSION 3.13.11)
set(CMAKE_POLICY_DEFAULT_CMP 135 NEW)
set(CMAKE_CXX_STANDARD 20)
project(project_tomos LANGUAGES CXX C CUDA VERSION 1.0)

# NOT USED. TOMOS_LINK_FLAGS and TOMOS_INC (below) are each only ever set()
# or appended to themselves (see the re-set at the CUDA block further down
# and the self-appends to both variables) -- neither is ever passed to
# target_link_libraries/target_link_options/target_include_directories/
# include_directories anywhere in this file. Every real target (tomos_core,
# tomos_cpu, tomos_cuda) gets its actual include dirs and link flags from
# separate, directly-written target_include_directories(...)/
# target_link_libraries(...) calls instead. Kept for now, documented rather
# than removed.
set(TOMOS_LINK_FLAGS
    -Wl,-rpath=${PROJECT_BINARY_DIR}
    -Wl,-rpath=${PROJECT_BINARY_DIR}/kernels/cpu
    -Wl,-rpath=${PROJECT_BINARY_DIR}/kernels/cuda
    -L${PROJECT_BINARY_DIR}
    -ltomos_cpu
    -ltomos_core
)

set(TOMOS_INC
    ${CMAKE_SOURCE_DIR}/include/

    # ${CMAKE_SOURCE_DIR}/io/
    # ${CMAKE_SOURCE_DIR}/include/core/precond
    # ${CMAKE_SOURCE_DIR}/tests/utils
    # ${CMAKE_SOURCE_DIR}/utils
    # ${CMAKE_SOURCE_DIR}/src/core/context
    # ${CMAKE_SOURCE_DIR}/utils
    # ${CMAKE_SOURCE_DIR}/include/core/precisions
    # ${CMAKE_SOURCE_DIR}/include/core/matrix
    # ${CMAKE_SOURCE_DIR}/include/core/memory
    # ${CMAKE_SOURCE_DIR}/include/core/error
    # ${CMAKE_SOURCE_DIR}/include/core/object_ownership
    # ${CMAKE_SOURCE_DIR}/include/core/mixins
    # ${CMAKE_SOURCE_DIR}/include/core/device
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/memory
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/matrix
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/utils
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/memory
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/device
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/solver
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/matrix
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/blas
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/sketch
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/solver
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/sketch
    # ${CMAKE_SOURCE_DIR}/include/core/sketch
    # ${CMAKE_SOURCE_DIR}/include/core/solver
    # ${CMAKE_SOURCE_DIR}/include/core/blas
    # ${CMAKE_SOURCE_DIR}/src/kernels/cuda/tests
    # ${CMAKE_SOURCE_DIR}/dependencies/blas
    # ${CMAKE_SOURCE_DIR}/src/kernels/cuda/tests/matrix
    # ${CMAKE_SOURCE_DIR}/src/kernels/cuda/tests/
    # ${CMAKE_SOURCE_DIR}/include/core/logger
    # ${CMAKE_SOURCE_DIR}/include/core/sketch
    # ${CMAKE_SOURCE_DIR}/include/core/matrix/matrix_dispatch.hpp
    # ${CMAKE_SOURCE_DIR}/io
)

add_compile_definitions(${BLAS})

# USE_CUDA must be defined identically for every target, not just
# tomos_core/tomos_cpu -- context.hpp's Context holds a
# std::variant<CpuContext, CudaContext>, and CudaContext's members are
# themselves guarded by #if defined(USE_CUDA). A target compiled without
# this macro sees a smaller/different Context layout than the one used
# inside libtomos_core.so, so any code path that constructs or writes a
# Context (e.g. Context::Context(Device)) overruns the caller's object and
# corrupts adjacent memory -- confirmed via gdb as the cause of
# heap/stack corruption crashes in test_dense/test_sparse/test_countsketch/
# test_gaussian_sketch (only test_lsqr, which had its own one-off
# target_compile_definitions(test_lsqr PRIVATE USE_CUDA=1), was unaffected).
# Defining it globally, before any target is declared, keeps every TU in
# the build consistent.
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    add_compile_definitions(USE_CUDA=1)
endif()

if("${DEBUG}" STREQUAL "BUILD_WITH_DEBUG_INFO")
    set(CMAKE_CXX_FLAGS
        "-Wall -std=c++17           \
        -DPRINT_SOLVER_STATE=1      \
        -DCHOLMOD_HAS_CUDA          \
        -DSUITESPARSE_USE_CUDA=ON   \
        -DCHOLMOD_USE_CUDA=ON       \
        -DENABLE_CHOLMOD            \
        -fPIC                       \
        -g3                         \
        -DSPQR_USE_CUDA"
    )
else()
    set(CMAKE_CXX_FLAGS
        "-Wall                      \
        -std=c++17                  \
        -DPRINT_SOLVER_STATE=1      \
        -DCHOLMOD_HAS_CUDA          \
        -DSUITESPARSE_USE_CUDA=ON   \
        -DCHOLMOD_USE_CUDA=ON       \
        -fPIC                       \
        -DENABLE_CHOLMOD            \
        -DSPQR_USE_CUDA")
endif()

# --- tomos_core
add_library(tomos_core SHARED
    src/core/context/context.cpp
    src/core/memory/malloc.cpp
    src/core/memory/dense_storage.cpp
    src/core/memory/sparse_storage.cpp
    src/core/memory/set_values.cpp
    src/core/matrix/matrix.cpp

    src/core/sketch/countsketch.cpp
    src/core/sketch/sketch.cpp
    src/core/blas/blas.cpp
    src/core/precond/sketch_qr.cpp
    src/core/workspace/workspace.cpp
    src/kernels/cpu/dependencies/blas.cpp
    src/kernels/cpu/dependencies/lapack.cpp

    src/dispatch/dense_registries.cpp
    src/dispatch/sparse_registries.cpp
    src/dispatch/precond_registries.cpp
    src/dispatch/lsqr_registries.cpp
    src/dispatch/storage_dispatch.cpp
    src/dispatch/sketch_registries.cpp
    src/dispatch/sketch_dispatch.cpp
    src/dispatch/gaussian_sketch_registries.cpp
    src/dispatch/gaussian_sketch_dispatch.cpp
    src/dispatch/fp32_16_registries.cpp
    src/io/mmio.c
    src/io/io.cpp
    src/core/solver/lsqr.cpp
)
target_include_directories(tomos_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/src/core/context
    ${CMAKE_SOURCE_DIR}/src/core/memory
    ${CMAKE_SOURCE_DIR}/src/core/blas
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/context
    ${CMAKE_SOURCE_DIR}/src/macros
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/memory
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/sketch
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/constants
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/memory
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/sketch
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/precond
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/blas
    ${CMAKE_SOURCE_DIR}/src/core/matrix
    ${CMAKE_SOURCE_DIR}/src/core/solver
    ${CMAKE_SOURCE_DIR}/src/dispatch
    ${CMAKE_SOURCE_DIR}/src/io
)

target_compile_options(tomos_core PUBLIC -g)
set_property(TARGET tomos_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET tomos_core PROPERTY CUDA_STANDARD 17)
set_target_properties(tomos_core PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tomos_core PRIVATE
    tomos_cpu
    -std=c++17
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/io
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/cuda
    -L${CMAKE_INSTALL_PREFIX}/io
    -Wl,--allow-shlib-undefined

    # -lmmio
)

# --- tomos_cpu
add_library(tomos_cpu SHARED
    src/core/memory/malloc.cpp
    src/kernels/cpu/memory/copy_cpu_kernels.cpp
    src/kernels/cpu/matrix/matrix_cpu_kernels.cpp
    src/kernels/cpu/matrix/mm_cpu_kernels.cpp
    src/kernels/cpu/memory/set_values_cpu_kernels.cpp
    src/kernels/cpu/matrix/csr_kernels.cpp
    src/kernels/cpu/matrix/dense_kernels.cpp
    src/kernels/cpu/matrix/fp32_16_kernels.cpp
    src/kernels/cpu/sketch/countsketch_kernels.cpp
    src/kernels/cpu/sketch/gaussian_sketch_kernels.cpp)

add_dependencies(tomos_core tomos_cpu)
target_include_directories(tomos_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/src/macros
    ${CMAKE_SOURCE_DIR}/src/core/context
    ${CMAKE_SOURCE_DIR}/src/core/memory
    ${CMAKE_SOURCE_DIR}/src/core/blas
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/context
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/memory
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/memory
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/constants
    ${CMAKE_SOURCE_DIR}/src/core/context
    ${CMAKE_SOURCE_DIR}/src/core/matrix
    ${CMAKE_SOURCE_DIR}/src/dispatch
    ${CMAKE_SOURCE_DIR}/src/io
)

target_compile_options(tomos_cpu PUBLIC -g)
set_property(TARGET tomos_cpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET tomos_cpu PROPERTY CUDA_STANDARD 17)
set_target_properties(tomos_cpu PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tomos_cpu PRIVATE
    -std=c++17
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/io
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/cuda
    -L${CMAKE_INSTALL_PREFIX}/io

    # -lmmio
)

if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    target_link_libraries(tomos_cpu PRIVATE
        -L${CUDA_LIB}
        -Wl,-rpath=${CUDA_LIB}
        -lcudart
    )
endif()

# Re-set here, still unused -- see the NOT USED note at this file's top.
set(TOMOS_LINK_FLAGS
    -Wl,-rpath=${PROJECT_BINARY_DIR}
    -Wl,-rpath=${PROJECT_BINARY_DIR}/src/kernels/cpu
    -ltomos_cpu

    # -ltomos_core
)

# Selects BLAS library.
if("${BLAS}" STREQUAL "USE_OPENBLAS")
    target_include_directories(tomos_cpu PUBLIC
        ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
        ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
        ${CMAKE_INSTALL_PREFIX}/generated
    )
    target_link_libraries(tomos_cpu PRIVATE
        -L${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib
        -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib
        -lopenblas
    )
    target_include_directories(tomos_core PUBLIC
        ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
        ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
        ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/lapack-netlib/LAPACKE/include
        ${CMAKE_INSTALL_PREFIX}/generated
        ${CMAKE_INSTALL_PREFIX}
    )
    target_link_libraries(tomos_core PRIVATE
        -shared
        -L${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib -lopenblas
        -lgfortran
    )

    # tomos_cpu/tomos_core link OpenBLAS via raw -L/-l flags above (not a CMake
    # target), so CMake can't infer this ordering on its own -- without it, a
    # fully parallel from-scratch build can link against libopenblas.so before
    # it exists.
    add_dependencies(tomos_cpu openblas_shared)
    add_dependencies(tomos_core openblas_shared)
endif()

# Selects GPU backend.
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    add_dependencies(tomos_core tomos_cuda)
    add_subdirectory(src/kernels/cuda)

    set(TOMOS_LINK_FLAGS
        ${TOMOS_LINK_FLAGS}
        -L${CUDA_LIB}
        -Wl,-rpath=${PROJECT_BINARY_DIR}/src/cuda
        -Wl,-rpath=${PROJECT_BINARY_DIR}
        -L${PROJECT_BINARY_DIR}/src/cuda
        -L${PROJECT_BINARY_DIR}
        -ltomos_cuda
        -lcublas
        -lcusparse
        -lcudart
        -lcurand
        -lcudadevrt
        -lcublasLt
        -lcusolver
    )

    set(TOMOS_INC "${TOMOS_INC} ${CUDA_INC}")

    # tomos_core/tomos_cpu are compiled as plain C++ (not nvcc), but
    # context.hpp/lsqr.cpp etc. include cublas_v2.h/cusparse.h directly, so
    # they need the CUDA toolkit's include directory too.
    target_include_directories(tomos_core PUBLIC ${CUDA_INC})
    target_include_directories(tomos_cpu PUBLIC ${CUDA_INC})

    target_link_libraries(tomos_core PRIVATE
        tomos_cuda
        -Wl,-rpath=${PROJECT_BINARY_DIR}
        -L${PROJECT_BINARY_DIR}/kernels/cuda
        -L${CUDA_LIB}
        -lcublas
        -lcusparse
        -lcudart
        -lcurand
        -lcudadevrt
        -lcublasLt
        -lcusolver)

    # -ltomos_cuda
endif()

add_subdirectory(test)
add_subdirectory(benchmarks)
add_subdirectory(dependencies)

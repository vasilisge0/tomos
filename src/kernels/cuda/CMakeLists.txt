set(TOMOS_CUDA_SOURCES
    matrix/mv_cuda_kernels.cu
    matrix/mm_cuda_kernels.cu
    matrix/fp32_16_cuda_kernels.cu
    matrix/fp32_bf16_cuda_kernels.cu
    dependencies/blas_kernels.cu
    memory/copy_cuda_kernels.cu
    blas/blas_cuda_kernels.cu
    blas/lsqr_blas1_cuda_kernels.cu
    sketch/countsketch_cuda_kernels.cu
    sketch/gaussian_sketch_cuda_kernels.cu
    precond/sketch_qr_cuda_kernels.cu

    # sketch/countsketch_cuda_kernels.cu
    memory/set_values_cuda_kernels.cu
    # matrix/cdense_kernels.cu
    # matrix/dense_kernels.cu
    # matrix/csr_kernels.cu
    # matrix/dense_64_16_kernels.cu
    # matrix/dense_32_16_kernels.cu
    # sketch/countsketch_kernels.cu
    # utils/print_mtx_kernels.cu
    # blas/cuda_blas.cu
    # solver/jacobi_kernels.cu
    # tests/initialize_raw_array.cu
    # tests/matrix/test_dense_32_16_kernels.cu
)

add_library(tomos_cuda SHARED ${TOMOS_CUDA_SOURCES})

# add_dependencies(tomos_cuda tomos_core)
set_target_properties(tomos_cuda PROPERTIES LINKER_LANGUAGE CUDA)
set_property(TARGET tomos_cuda PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
target_compile_definitions(tomos_cuda PRIVATE USE_CUDA=1)

# @FIX: hardcodes compute architecture
if("${DEBUG}" STREQUAL "BUILD_WITH_DEBUG_INFO")
    if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "75")
        target_compile_options(tomos_cuda PRIVATE -fPIC -arch=compute_75 -g3 -G)
    elseif("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "70")
        target_compile_options(tomos_cuda PRIVATE -fPIC -arch=compute_70 -g3 -G)
    elseif("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "80")
        target_compile_options(tomos_cuda PRIVATE -fPIC -gencode arch=compute_80,code=sm_80 -g3 -G)
    endif()
else()
    if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "75")
        target_compile_options(tomos_cuda PRIVATE -fPIC -arch=compute_75 -O3)
    elseif("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "70")
        target_compile_options(tomos_cuda PRIVATE -fPIC -arch=compute_70 -O3)
    elseif("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "80")
        target_compile_options(tomos_cuda PRIVATE -fPIC -gencode arch=compute_80,code=sm_80 --ptxas-options=-v -lineinfo -O3 -Xptxas -allow-expensive-optimizations=true)
    endif()
endif()

target_include_directories(tomos_cuda PUBLIC
    .
    ../
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/src/core/context
    ${CMAKE_SOURCE_DIR}/src/core/memory
    ${CMAKE_SOURCE_DIR}/src/core/blas
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/context
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/precond
    ${CMAKE_SOURCE_DIR}/src/macros
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/constants
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/memory
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/memory
    ${CMAKE_SOURCE_DIR}/src/core/matrix
    ${CMAKE_SOURCE_DIR}/src/core/detail

    # ${CMAKE_SOURCE_DIR}/include/core/solver
    # ${CMAKE_SOURCE_DIR}/include/core/precisions
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/solver
    # ${CMAKE_SOURCE_DIR}/include/kernels/cpu/blas
    # ${CMAKE_SOURCE_DIR}/dependencies/blas
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/memory
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/solver
    # ${CMAKE_SOURCE_DIR}/include/core/memory
    # ${CMAKE_SOURCE_DIR}/src/core/context
    # ${CMAKE_SOURCE_DIR}/include/core/error
    # ${CMAKE_SOURCE_DIR}/include/core/device
    # ${CMAKE_SOURCE_DIR}/include/core/matrix
    # ${CMAKE_SOURCE_DIR}/include/core/mixins
    # ${CMAKE_SOURCE_DIR}/include/core/blas
    # ${CMAKE_SOURCE_DIR}/include/core/object_ownership
    # ${CMAKE_SOURCE_DIR}/include/core/device
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/tests
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/utils
    # ${CMAKE_SOURCE_DIR}/src/kernels/cuda/tests/matrix
    # ${CMAKE_SOURCE_DIR}/src/kernels/cuda/tests/
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/matrix
    # ${CMAKE_SOURCE_DIR}/include/core/sketch
    # ${CMAKE_SOURCE_DIR}/include/kernels/cuda/blas
)

target_link_libraries(tomos_cuda PRIVATE
    -L${CUDA_LIB}
    -lcusparse
    -lcudart
    -lcublas
    -lcurand
    -lcudadevrt
    -lcublasLt
    -lcusolver
    -L${CMAKE_INSTALL_PREFIX}
    -L${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}
    -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/lib
    -L${PROJECT_BINARY_DIR}
    -Wl,-rpath=${PROJECT_BINARY_DIR}
    ${CMAKE_CUDA_FLAGS}
)

if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)

    add_executable(test_dense_cuda_kernels test_dense_cuda_kernels.cu)
    set_property(TARGET test_dense_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_dense_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    )
    target_link_libraries(test_dense_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cublas
        CUDA::cudart
        GTest::gtest_main
    )

    add_executable(test_mm_cuda_kernels test_mm_cuda_kernels.cu)
    set_property(TARGET test_mm_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_mm_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    )
    target_link_libraries(test_mm_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cublas
        CUDA::cusparse
        CUDA::cudart
        GTest::gtest_main
    )

    add_executable(test_csr_cuda_kernels test_csr_cuda_kernels.cu)
    set_property(TARGET test_csr_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_csr_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    )
    target_link_libraries(test_csr_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cusparse
        CUDA::cudart
        GTest::gtest_main
    )

    add_executable(test_fp32_16_cuda_kernels test_fp32_16_cuda_kernels.cu)
    set_property(TARGET test_fp32_16_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_fp32_16_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    )
    target_link_libraries(test_fp32_16_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cudart
        GTest::gtest_main
    )
endif()

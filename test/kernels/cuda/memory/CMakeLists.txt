if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)

    add_executable(test_copy_cuda_kernels test_copy_cuda_kernels.cu)
    set_property(TARGET test_copy_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_copy_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/memory
    )
    target_link_libraries(test_copy_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cudart
        GTest::gtest_main
    )

    add_executable(test_set_values_cuda_kernels test_set_values_cuda_kernels.cu)
    set_property(TARGET test_set_values_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_set_values_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/memory
    )
    target_link_libraries(test_set_values_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cudart
        GTest::gtest_main
    )
endif()

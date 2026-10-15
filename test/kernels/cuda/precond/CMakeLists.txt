if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)

    add_executable(test_sketch_qr_cuda_kernels test_sketch_qr_cuda_kernels.cu)
    set_property(TARGET test_sketch_qr_cuda_kernels PROPERTY CUDA_STANDARD 17)
    target_include_directories(test_sketch_qr_cuda_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels/cuda/precond
    )
    target_link_libraries(test_sketch_qr_cuda_kernels
        tomos_core
        tomos_cpu
        tomos_cuda
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
        GTest::gtest_main
    )
endif()

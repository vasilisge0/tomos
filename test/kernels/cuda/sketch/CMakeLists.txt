find_package(CUDAToolkit REQUIRED)

add_executable(test_countsketch_cuda_kernels test_countsketch_cuda_kernels.cu)
set_property(TARGET test_countsketch_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(test_countsketch_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/sketch
)
target_link_directories(test_countsketch_cuda_kernels PRIVATE
    ${PROJECT_BINARY_DIR}/src/kernels/cuda
)
target_link_libraries(test_countsketch_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cudart
    GTest::gtest_main
)

add_executable(test_gaussian_sketch_cuda_kernels test_gaussian_sketch_cuda_kernels.cu)
set_property(TARGET test_gaussian_sketch_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(test_gaussian_sketch_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/sketch
)
target_link_directories(test_gaussian_sketch_cuda_kernels PRIVATE
    ${PROJECT_BINARY_DIR}/src/kernels/cuda
)
target_link_libraries(test_gaussian_sketch_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cudart
    GTest::gtest_main
)
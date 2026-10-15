find_package(CUDAToolkit REQUIRED)

add_executable(bench_lsqr_cuda bench_lsqr_cuda.cu)
set_property(TARGET bench_lsqr_cuda PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_lsqr_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
)
target_link_libraries(bench_lsqr_cuda
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cudart
    benchmark::benchmark
)

add_executable(lsqr_precision_runner lsqr_precision_runner.cu)
set_property(TARGET lsqr_precision_runner PROPERTY CUDA_STANDARD 17)
target_include_directories(lsqr_precision_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/precond
)
target_link_libraries(lsqr_precision_runner
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cudart
)

add_executable(lsqr_gemv_isolated lsqr_gemv_isolated.cu)
set_property(TARGET lsqr_gemv_isolated PROPERTY CUDA_STANDARD 17)
target_link_libraries(lsqr_gemv_isolated
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cudart
)

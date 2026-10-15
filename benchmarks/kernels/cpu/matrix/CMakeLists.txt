add_executable(bench_mv_cpu_kernels bench_mv_cpu_kernels.cpp)
target_include_directories(bench_mv_cpu_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(bench_mv_cpu_kernels
    tomos_core
    tomos_cpu
    openblas
    benchmark::benchmark
)

add_executable(bench_mm_cpu_kernels bench_mm_cpu_kernels.cpp)
target_include_directories(bench_mm_cpu_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(bench_mm_cpu_kernels
    tomos_core
    tomos_cpu
    openblas
    benchmark::benchmark
)

add_executable(bench_fp32_16_cpu_kernels bench_fp32_16_cpu_kernels.cpp)
target_include_directories(bench_fp32_16_cpu_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(bench_fp32_16_cpu_kernels
    tomos_core
    tomos_cpu
    openblas
    benchmark::benchmark
)

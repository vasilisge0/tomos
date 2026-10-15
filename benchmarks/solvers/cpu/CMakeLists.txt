add_executable(bench_lsqr bench_lsqr.cpp)
target_include_directories(bench_lsqr PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/core/precond
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(bench_lsqr
    tomos_core
    tomos_cpu
    openblas
    benchmark::benchmark
)

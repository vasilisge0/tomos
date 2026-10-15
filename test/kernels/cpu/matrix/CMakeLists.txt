set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

add_executable(test_dense_cpu_kernels test_dense_cpu_kernels.cpp)
target_include_directories(test_dense_cpu_kernels PRIVATE
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(test_dense_cpu_kernels
    ${TOMOS_TEST_LIBS}
    openblas
    GTest::gtest_main
)

add_executable(test_mm_cpu_kernels test_mm_cpu_kernels.cpp)
target_include_directories(test_mm_cpu_kernels PRIVATE
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(test_mm_cpu_kernels
    ${TOMOS_TEST_LIBS}
    openblas
    GTest::gtest_main
)

add_executable(test_csr_cpu_kernels test_csr_cpu_kernels.cpp)
target_include_directories(test_csr_cpu_kernels PRIVATE
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
)
target_link_libraries(test_csr_cpu_kernels
    ${TOMOS_TEST_LIBS}
    openblas
    GTest::gtest_main
)

add_executable(test_fp32_16_cpu_kernels test_fp32_16_cpu_kernels.cpp)
target_link_libraries(test_fp32_16_cpu_kernels
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)
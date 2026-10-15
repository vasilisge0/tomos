set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

add_executable(test_copy_cpu_kernels test_copy_cpu_kernels.cpp)
target_link_libraries(test_copy_cpu_kernels
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)

add_executable(test_set_values_cpu_kernels test_set_values_cpu_kernels.cpp)
target_link_libraries(test_set_values_cpu_kernels
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)

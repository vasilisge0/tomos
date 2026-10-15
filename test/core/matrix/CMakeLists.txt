set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

# test_matrix
add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_dense
add_executable(test_dense test_dense.cpp)
target_link_libraries(test_dense ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_sparse
add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_fp32_16
add_executable(test_fp32_16 test_fp32_16.cpp)
target_link_libraries(test_fp32_16 ${TOMOS_TEST_LIBS} GTest::gtest_main)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    target_include_directories(test_fp32_16 PRIVATE ${CUDA_INC})
    target_link_libraries(test_fp32_16 -L${CUDA_LIB} -lcudart)
endif()
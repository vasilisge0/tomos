set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

# test_dense_registries
add_executable(test_dense_registries test_dense_registries.cpp)
target_link_libraries(test_dense_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_sparse_registries
add_executable(test_sparse_registries test_sparse_registries.cpp)
target_link_libraries(test_sparse_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_fp32_16_registries
add_executable(test_fp32_16_registries test_fp32_16_registries.cpp)
target_link_libraries(test_fp32_16_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_lsqr_registries
add_executable(test_lsqr_registries test_lsqr_registries.cpp)
target_link_libraries(test_lsqr_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_sketch_registries
add_executable(test_sketch_registries test_sketch_registries.cpp)
target_link_libraries(test_sketch_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_gaussian_sketch_registries
add_executable(test_gaussian_sketch_registries test_gaussian_sketch_registries.cpp)
target_link_libraries(test_gaussian_sketch_registries ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_storage_dispatch
add_executable(test_storage_dispatch test_storage_dispatch.cpp)
target_link_libraries(test_storage_dispatch ${TOMOS_TEST_LIBS} GTest::gtest_main)

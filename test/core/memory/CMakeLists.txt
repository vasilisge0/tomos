set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

# test_copy
add_executable(test_copy test_copy.cpp)
target_link_libraries(test_copy ${TOMOS_TEST_LIBS} GTest::gtest_main)

# test_malloc
add_executable(test_malloc test_malloc.cpp)
target_link_libraries(test_malloc ${TOMOS_TEST_LIBS} GTest::gtest_main)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    target_include_directories(test_malloc PRIVATE ${CUDA_INC})
    target_link_libraries(test_malloc -L${CUDA_LIB} -lcudart)
endif()

# test_set_values
add_executable(test_set_values test_set_values.cpp)
target_link_libraries(test_set_values ${TOMOS_TEST_LIBS} GTest::gtest_main)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    target_include_directories(test_set_values PRIVATE ${CUDA_INC})
    target_link_libraries(test_set_values -L${CUDA_LIB} -lcudart)
endif()
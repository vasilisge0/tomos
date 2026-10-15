set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

add_executable(test_countsketch test_countsketch.cpp)
target_link_libraries(test_countsketch
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)

add_executable(test_gaussian_sketch test_gaussian_sketch.cpp)
target_link_libraries(test_gaussian_sketch
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)
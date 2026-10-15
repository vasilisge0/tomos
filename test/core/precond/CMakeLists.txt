set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

add_executable(test_sketch_qr test_sketch_qr.cpp)
target_include_directories(test_sketch_qr PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/precond
)
target_link_libraries(test_sketch_qr
    ${TOMOS_TEST_LIBS}
    GTest::gtest_main
)

set(TOMOS_TEST_LIBS tomos_core tomos_cpu)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    list(APPEND TOMOS_TEST_LIBS tomos_cuda)
endif()

add_executable(test_lsqr test_lsqr.cpp)
target_include_directories(test_lsqr PRIVATE
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-src/
    ${CMAKE_INSTALL_PREFIX}/_deps/openblas-build/
    ${CMAKE_SOURCE_DIR}/src/core/solver
    ${CMAKE_SOURCE_DIR}/src/core/precond
)
target_link_libraries(test_lsqr
    ${TOMOS_TEST_LIBS}
    openblas
    GTest::gtest_main
)
if("${DEVICE_BACKEND}" STREQUAL "CUDA")
    target_include_directories(test_lsqr PRIVATE ${CUDA_INC})
    target_link_libraries(test_lsqr -L${CUDA_LIB} -lcudart)
endif()
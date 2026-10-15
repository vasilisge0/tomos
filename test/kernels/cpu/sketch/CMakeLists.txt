add_executable(test_countsketch_cpu_kernels test_countsketch_cpu_kernels.cpp)
target_include_directories(test_countsketch_cpu_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/sketch
)
target_link_libraries(test_countsketch_cpu_kernels
    tomos_core
    tomos_cpu
    GTest::gtest_main
)

add_executable(test_gaussian_sketch_cpu_kernels test_gaussian_sketch_cpu_kernels.cpp)
target_include_directories(test_gaussian_sketch_cpu_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/sketch
)
target_link_libraries(test_gaussian_sketch_cpu_kernels
    tomos_core
    tomos_cpu
    GTest::gtest_main
)
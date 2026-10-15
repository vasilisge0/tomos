#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "dimensions.hpp"
#include "set_values_cpu_kernels.hpp"

namespace tomos {
namespace test {

TEST(SetValuesCpuKernelTest, FillsWholeMatrixNoPadding) {
    dim<2> size{2, 3};
    std::vector<double> values(6, -1.0);
    cpu::set_values_impl<double>(9.0, size, values.data(), 2);
    for (double v : values) EXPECT_DOUBLE_EQ(v, 9.0);
}

TEST(SetValuesCpuKernelTest, RespectsLeadingDimensionPadding) {
    dim<2> size{2, 2};
    std::vector<double> values(6, -1.0);
    cpu::set_values_impl<double>(9.0, size, values.data(), 3);
    EXPECT_DOUBLE_EQ(values[0], 9.0);
    EXPECT_DOUBLE_EQ(values[1], 9.0);
    EXPECT_DOUBLE_EQ(values[2], -1.0);  // padding, untouched
    EXPECT_DOUBLE_EQ(values[3], 9.0);
    EXPECT_DOUBLE_EQ(values[4], 9.0);
    EXPECT_DOUBLE_EQ(values[5], -1.0);  // padding, untouched
}

TEST(SetValuesCpuKernelTest, FloatType) {
    dim<2> size{2, 2};
    std::vector<float> values(4, -1.0f);
    cpu::set_values_impl<float>(2.5f, size, values.data(), 2);
    for (float v : values) EXPECT_FLOAT_EQ(v, 2.5f);
}

// uint32_t{0} is used to zero-fill NumericType::FP32_16 storage (see
// set_values_cpu_kernels.cpp's own comment): wh=wr=0 is an exact-zero
// encoding.
TEST(SetValuesCpuKernelTest, ZeroFillsUint32) {
    dim<2> size{2, 2};
    std::vector<uint32_t> values(4, 0xFFFFFFFFu);
    cpu::set_values_impl<uint32_t>(0u, size, values.data(), 2);
    for (uint32_t v : values) EXPECT_EQ(v, 0u);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

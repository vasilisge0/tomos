#include <gtest/gtest.h>

#include <vector>

#include "copy_cpu_kernels.hpp"
#include "dimensions.hpp"

namespace tomos {
namespace test {

TEST(StridedCopyKernelTest, SameTypeNoPadding) {
    dim<2> size{2, 3};
    std::vector<double> in = {1, 2, 3, 4, 5, 6};
    std::vector<double> out(6, -1.0);
    cpu::strided_copy_kernel(size, in.data(), 2, out.data(), 2);
    for (int i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(out[i], in[i]);
}

TEST(StridedCopyKernelTest, RespectsLeadingDimensionPadding) {
    // 2x2 logical matrix, source packed (ld=2), target padded (ld=3): only
    // the first two entries of each 3-element target column should be
    // touched.
    dim<2> size{2, 2};
    std::vector<double> in = {1, 2, 3, 4};
    std::vector<double> out(6, -1.0);
    cpu::strided_copy_kernel(size, in.data(), 2, out.data(), 3);
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 2.0);
    EXPECT_DOUBLE_EQ(out[2], -1.0);  // padding, untouched
    EXPECT_DOUBLE_EQ(out[3], 3.0);
    EXPECT_DOUBLE_EQ(out[4], 4.0);
    EXPECT_DOUBLE_EQ(out[5], -1.0);  // padding, untouched
}

TEST(StridedCopyKernelTest, ConvertsDoubleToFloat) {
    dim<2> size{2, 2};
    std::vector<double> in = {1.5, 2.5, 3.5, 4.5};
    std::vector<float> out(4, 0.0f);
    cpu::strided_copy_kernel(size, in.data(), 2, out.data(), 2);
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(in[i]));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

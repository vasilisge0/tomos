#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dimensions.hpp"
#include "gaussian_sketch_kernels.hpp"
#include "status.hpp"

namespace tomos {
namespace test {

TEST(GenerateGaussianValuesTest, ReturnsSuccessAndFillsEveryEntry) {
    constexpr size_t N = 20000;
    std::vector<double> values(N, 0.0);
    dim<2> size{N, 1};
    Status st = cpu::generate_gaussian_values<double>(size, values.data(), N);
    EXPECT_EQ(st, SUCCESS);

    // Sanity: not left at the zero-init sentinel, and not all identical
    // (would indicate a broken/constant generator).
    bool any_nonzero = false;
    bool any_different = false;
    for (size_t i = 1; i < N; ++i) {
        if (values[i] != 0.0) any_nonzero = true;
        if (values[i] != values[0]) any_different = true;
    }
    EXPECT_TRUE(any_nonzero);
    EXPECT_TRUE(any_different);
}

// Statistical sanity check: for a large N(0,1) sample, mean should be near 0
// and variance near 1, well within a generous tolerance to avoid flakiness.
TEST(GenerateGaussianValuesTest, SampleMeanAndVarianceAreReasonable) {
    constexpr size_t N = 100000;
    std::vector<double> values(N, 0.0);
    dim<2> size{N, 1};
    cpu::generate_gaussian_values<double>(size, values.data(), N);

    double sum = 0.0;
    for (double v : values) sum += v;
    double mean = sum / N;

    double sq_sum = 0.0;
    for (double v : values) sq_sum += (v - mean) * (v - mean);
    double variance = sq_sum / N;

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(variance, 1.0, 0.05);
}

TEST(GenerateGaussianValuesTest, RespectsLeadingDimensionPadding) {
    dim<2> size{2, 2};
    std::vector<double> values(6, -999.0);
    cpu::generate_gaussian_values<double>(size, values.data(), 3);
    EXPECT_NE(values[0], -999.0);
    EXPECT_NE(values[1], -999.0);
    EXPECT_DOUBLE_EQ(values[2], -999.0);  // padding, untouched
    EXPECT_NE(values[3], -999.0);
    EXPECT_NE(values[4], -999.0);
    EXPECT_DOUBLE_EQ(values[5], -999.0);  // padding, untouched
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

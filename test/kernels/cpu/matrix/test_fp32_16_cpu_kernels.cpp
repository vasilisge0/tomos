#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "dimensions.hpp"
#include "fp32_16_kernels.hpp"
#include "transpose.hpp"

namespace tomos {
namespace test {

// A (2x2, col-major): [[1.5, 2.25], [3.75, 4.125]]
// Chosen as exact binary fractions so split+combine round-trips exactly.
class Fp32_16CpuTest : public ::testing::Test {
protected:
    dim<2> size{2, 2};
    std::vector<float> A{1.5f, 3.75f, 2.25f, 4.125f};  // col-major
    std::vector<uint16_t> wh = std::vector<uint16_t>(4);
    std::vector<uint16_t> wr = std::vector<uint16_t>(4);

    void SetUp() override {
        cpu::split_dense_fp32_16(size, A.data(), 2, wh.data(), wr.data(), 2);
    }
};

TEST_F(Fp32_16CpuTest, SplitCombineRoundTripsExactly) {
    std::vector<float> reconstructed(4, 0.0f);
    cpu::combine_dense_fp32_16(size, wh.data(), wr.data(), 2,
                               reconstructed.data(), 2);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(reconstructed[i], A[i]);
}

TEST_F(Fp32_16CpuTest, MvAccurateNoTransposeMatchesNaiveReference) {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y = {0.0f, 0.0f};
    cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, size, wh.data(),
                             wr.data(), 2, x.data(), 1, 0.0, y.data(), 1);
    // A*x = (1.5+2.25, 3.75+4.125) = (3.75, 7.875)
    EXPECT_NEAR(y[0], 3.75f, 1e-4f);
    EXPECT_NEAR(y[1], 7.875f, 1e-4f);
}

TEST_F(Fp32_16CpuTest, MvAccurateTransposeMatchesNaiveReference) {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y = {0.0f, 0.0f};
    cpu::mv_fp32_16_accurate(TransposeType::TRANSPOSE, 1.0, size, wh.data(),
                             wr.data(), 2, x.data(), 1, 0.0, y.data(), 1);
    // A^T*x = (1.5+3.75, 2.25+4.125) = (5.25, 6.375)
    EXPECT_NEAR(y[0], 5.25f, 1e-4f);
    EXPECT_NEAR(y[1], 6.375f, 1e-4f);
}

TEST_F(Fp32_16CpuTest, MvAccurateBetaAccumulates) {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y = {10.0f, 10.0f};
    cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, size, wh.data(),
                             wr.data(), 2, x.data(), 1, 1.0, y.data(), 1);
    EXPECT_NEAR(y[0], 13.75f, 1e-4f);
    EXPECT_NEAR(y[1], 17.875f, 1e-4f);
}

// fast/pure_half only decode wh (fp16-level per-element accuracy); with
// exact-in-fp16 inputs like these, they should still match closely.
TEST_F(Fp32_16CpuTest, MvFastNoTransposeCloseToReference) {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y = {0.0f, 0.0f};
    cpu::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, size, wh.data(), 2,
                         x.data(), 1, 0.0, y.data(), 1);
    EXPECT_NEAR(y[0], 3.75f, 1e-3f);
    EXPECT_NEAR(y[1], 7.875f, 1e-3f);
}

TEST_F(Fp32_16CpuTest, MvPureHalfNoTransposeCloseToReference) {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y = {0.0f, 0.0f};
    cpu::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, size,
                              wh.data(), 2, x.data(), 1, 0.0, y.data(), 1);
    EXPECT_NEAR(y[0], 3.75f, 1e-2f);
    EXPECT_NEAR(y[1], 7.875f, 1e-2f);
}

TEST_F(Fp32_16CpuTest, MmAccurateNoTransposeMatchesNaiveReference) {
    // B (2x2, col-major identity)
    std::vector<float> B = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> C(4, 0.0f);
    cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                             TransposeType::NO_TRANSPOSE, 1.0, size, wh.data(),
                             wr.data(), 2, size, B.data(), 2, 0.0, C.data(), 2);
    // A * I == A
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(C[i], A[i], 1e-4f);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

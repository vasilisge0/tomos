#include <gtest/gtest.h>

#include <vector>

#include "dimensions.hpp"
#include "matrix_cpu_kernels.hpp"
#include "precision.hpp"
#include "transpose.hpp"

namespace tomos {
namespace test {

// A (2x3, CSR): [[1,2,0],[0,0,3]]
// row_ptrs = {0,2,3}, col_idxs = {0,1,2}, values = {1,2,3}
class CsrMvTest : public ::testing::Test {
protected:
    dim<2> size{2, 3};
    std::vector<int> row_ptrs{0, 2, 3};
    std::vector<int> col_idxs{0, 1, 2};
    std::vector<double> values{1.0, 2.0, 3.0};
};

TEST_F(CsrMvTest, NoTranspose) {
    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y = {0.0, 0.0};
    cpu::mv_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, 1.0, size, values.size(), values.data(),
        row_ptrs.data(), col_idxs.data(), x.data(), 1, 0.0, y.data(), 1);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
}

TEST_F(CsrMvTest, Transpose) {
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> y = {0.0, 0.0, 0.0};
    cpu::mv_csr<double, double, double, int>(
        TransposeType::TRANSPOSE, 1.0, size, values.size(), values.data(),
        row_ptrs.data(), col_idxs.data(), x.data(), 1, 0.0, y.data(), 1);
    EXPECT_DOUBLE_EQ(y[0], 1.0);
    EXPECT_DOUBLE_EQ(y[1], 2.0);
    EXPECT_DOUBLE_EQ(y[2], 3.0);
}

TEST_F(CsrMvTest, AlphaBetaScaling) {
    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y = {100.0, 100.0};
    cpu::mv_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, 2.0, size, values.size(), values.data(),
        row_ptrs.data(), col_idxs.data(), x.data(), 1, 0.5, y.data(), 1);
    // y = 2*(A*x) + 0.5*y = 2*3 + 50 = 56 for both rows.
    EXPECT_DOUBLE_EQ(y[0], 56.0);
    EXPECT_DOUBLE_EQ(y[1], 56.0);
}

TEST_F(CsrMvTest, BetaZeroDoesNotReadY) {
    std::vector<double> x = {1.0, 1.0, 1.0};
    // y starts uninitialized-looking (NaN-like sentinel); beta==0 must not
    // read it.
    std::vector<double> y = {-999.0, -999.0};
    cpu::mv_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, 1.0, size, values.size(), values.data(),
        row_ptrs.data(), col_idxs.data(), x.data(), 1, 0.0, y.data(), 1);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
}

TEST_F(CsrMvTest, MixedPrecisionAFloatXDoubleYDouble) {
    std::vector<float> a_vals(values.begin(), values.end());
    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y = {0.0, 0.0};
    cpu::mv_csr<float, double, double, int>(
        TransposeType::NO_TRANSPOSE, 1.0, size, a_vals.size(), a_vals.data(),
        row_ptrs.data(), col_idxs.data(), x.data(), 1, 0.0, y.data(), 1);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

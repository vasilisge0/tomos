#include <gtest/gtest.h>

#include "context.hpp"
#include "matrix.hpp"
#include "sketch_qr.hpp"

namespace tomos {
namespace test {

class SketchQrTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

// generate_sketch_qr_precond should leave R^{-1} in the leading n x n block
// of sketched_mtx, where R comes from the QR factorization of the original
// (k x n) sketched_mtx on entry. Verified via the identity
// M^T (A^T A) M == I, where M = R^{-1} and A^T A = R^T R for any valid QR
// factorization -- true regardless of sign/basis conventions.
TEST_F(SketchQrTest, ProducesRInverse) {
    dim<2> size = {4, 2};
    Dense sketched =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    auto v = as_strided_view<double>(sketched);

    // A (4x2), column-major: col0 = {1,0,1,1}, col1 = {0,1,1,-1}.
    double a_col0[4] = {1.0, 0.0, 1.0, 1.0};
    double a_col1[4] = {0.0, 1.0, 1.0, -1.0};
    for (size_t i = 0; i < 4; ++i) {
        v.values_[i + 0 * v.lead_dim_] = a_col0[i];
        v.values_[i + 1 * v.lead_dim_] = a_col1[i];
    }

    // A^T A, computed from the original A before it gets overwritten.
    double ata[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (size_t i = 0; i < 4; ++i) {
        ata[0][0] += a_col0[i] * a_col0[i];
        ata[0][1] += a_col0[i] * a_col1[i];
        ata[1][0] += a_col1[i] * a_col0[i];
        ata[1][1] += a_col1[i] * a_col1[i];
    }

    Sparse sketch =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {1, 1}, 1);
    generate_sketch_qr_precond(sketch, sketched);

    double m[2][2];
    m[0][0] = v.values_[0 + 0 * v.lead_dim_];
    m[1][0] = v.values_[1 + 0 * v.lead_dim_];
    m[0][1] = v.values_[0 + 1 * v.lead_dim_];
    m[1][1] = v.values_[1 + 1 * v.lead_dim_];

    // M^T (A^T A) M should be the 2x2 identity.
    double tmp[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) tmp[i][j] += m[k][i] * ata[k][j];

    double result[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) result[i][j] += tmp[i][k] * m[k][j];

    EXPECT_NEAR(result[0][0], 1.0, 1e-10);
    EXPECT_NEAR(result[1][1], 1.0, 1e-10);
    EXPECT_NEAR(result[0][1], 0.0, 1e-10);
    EXPECT_NEAR(result[1][0], 0.0, 1e-10);

    // R^{-1} is upper triangular; the sub-diagonal must be exactly zero.
    EXPECT_EQ(m[1][0], 0.0);
}

TEST_F(SketchQrTest, ThrowsWhenFewerRowsThanCols) {
    dim<2> size = {2, 4};
    Dense sketched =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    Sparse sketch =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {1, 1}, 1);
    EXPECT_THROW(generate_sketch_qr_precond(sketch, sketched),
                std::invalid_argument);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

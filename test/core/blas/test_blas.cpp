#include <gtest/gtest.h>

#include <variant>

#include "blas.hpp"
#include "context.hpp"
#include "matrix.hpp"

namespace tomos {
namespace test {

class BlasTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

// ---------------------------------------------------------------------------
// mv_apply (Dense A)
// ---------------------------------------------------------------------------

TEST_F(BlasTest, MvApplyDenseNoTranspose) {
    // A = [[1, 2], [3, 4]] (col-major), x = (1, 1), alpha = 2, beta = 0.
    // A * x = (3, 7); scaled by alpha -> (6, 14).
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 3.0;
    vA.values_[2] = 2.0;
    vA.values_[3] = 4.0;

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 1.0;
    vx.values_[1] = 1.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 2.0, A, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 6.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 14.0, 1e-12);
}

TEST_F(BlasTest, MvApplyDenseTranspose) {
    // Same A as above. A^T * x = (4, 6).
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 3.0;
    vA.values_[2] = 2.0;
    vA.values_[3] = 4.0;

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 1.0;
    vx.values_[1] = 1.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    mv_apply(Device::CPU, TransposeType::TRANSPOSE, 1.0, A, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 4.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 6.0, 1e-12);
}

TEST_F(BlasTest, MvApplyDenseBetaAccumulates) {
    // y starts at (100, 100); alpha = 1, beta = 1 -> y = A*x + y.
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 3.0;
    vA.values_[2] = 2.0;
    vA.values_[3] = 4.0;

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 1.0;
    vx.values_[1] = 1.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vy_init = as_strided_view<double>(y);
    vy_init.values_[0] = 100.0;
    vy_init.values_[1] = 100.0;

    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x, 1.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 103.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 107.0, 1e-12);
}

// ---------------------------------------------------------------------------
// mv_apply (Sparse A, CSR)
// ---------------------------------------------------------------------------

TEST_F(BlasTest, MvApplySparseDiagonal) {
    // A = diag(2, 3) as CSR, x = (5, 7) -> A*x = (10, 21).
    Sparse A = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {2, 2}, 2);
    auto vA = as_csr_view<double, int>(A);
    vA.row_ptrs_[0] = 0;
    vA.row_ptrs_[1] = 1;
    vA.row_ptrs_[2] = 2;
    vA.col_idxs_[0] = 0;
    vA.col_idxs_[1] = 1;
    vA.values_[0] = 2.0;
    vA.values_[1] = 3.0;

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 5.0;
    vx.values_[1] = 7.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 10.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 21.0, 1e-12);
}

// ---------------------------------------------------------------------------
// mv_apply (variant<Dense*, Sparse*> A) -- runtime-dispatched overload.
// ---------------------------------------------------------------------------

TEST_F(BlasTest, MvApplyVariantDispatchesToDense) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 0.0;
    vA.values_[2] = 0.0;
    vA.values_[3] = 1.0;  // identity

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 5.0;
    vx.values_[1] = 6.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    std::variant<Dense*, Sparse*> Avar = &A;
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, Avar, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 5.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 6.0, 1e-12);
}

TEST_F(BlasTest, MvApplyVariantDispatchesToSparse) {
    Sparse A = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {2, 2}, 2);
    auto vA = as_csr_view<double, int>(A);
    vA.row_ptrs_[0] = 0;
    vA.row_ptrs_[1] = 1;
    vA.row_ptrs_[2] = 2;
    vA.col_idxs_[0] = 0;
    vA.col_idxs_[1] = 1;
    vA.values_[0] = 2.0;
    vA.values_[1] = 3.0;

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 5.0;
    vx.values_[1] = 7.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    std::variant<Dense*, Sparse*> Avar = &A;
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, Avar, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 10.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 21.0, 1e-12);
}

// ---------------------------------------------------------------------------
// precond_apply -- std::monostate is a no-op (y left untouched); Dense*/
// Sparse* forward to mv_apply.
// ---------------------------------------------------------------------------

TEST_F(BlasTest, PrecondApplyMonostateLeavesYUntouched) {
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vy_init = as_strided_view<double>(y);
    vy_init.values_[0] = 42.0;
    vy_init.values_[1] = 43.0;

    std::variant<Dense*, Sparse*, std::monostate> P = std::monostate{};
    precond_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, P, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_EQ(vy.values_[0], 42.0);
    EXPECT_EQ(vy.values_[1], 43.0);
}

TEST_F(BlasTest, PrecondApplyDenseAppliesMv) {
    Dense P_mtx = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP64, {2, 2});
    auto vP = as_strided_view<double>(P_mtx);
    vP.values_[0] = 2.0;
    vP.values_[1] = 0.0;
    vP.values_[2] = 0.0;
    vP.values_[3] = 5.0;  // diag(2, 5)

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    auto vx = as_strided_view<double>(x);
    vx.values_[0] = 3.0;
    vx.values_[1] = 4.0;

    Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    std::variant<Dense*, Sparse*, std::monostate> P = &P_mtx;
    precond_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, P, x, 0.0, y);

    auto vy = as_strided_view<double>(y);
    EXPECT_NEAR(vy.values_[0], 6.0, 1e-12);
    EXPECT_NEAR(vy.values_[1], 20.0, 1e-12);
}

// ---------------------------------------------------------------------------
// mm_apply (Dense A, B, C)
// ---------------------------------------------------------------------------

TEST_F(BlasTest, MmApplyDenseNoTranspose) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (both col-major).
    // A * B = [[19, 22], [43, 50]].
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 3.0;
    vA.values_[2] = 2.0;
    vA.values_[3] = 4.0;

    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vB = as_strided_view<double>(B);
    vB.values_[0] = 5.0;
    vB.values_[1] = 7.0;
    vB.values_[2] = 6.0;
    vB.values_[3] = 8.0;

    Dense C = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    mm_apply(Device::CPU, TransposeType::NO_TRANSPOSE,
             TransposeType::NO_TRANSPOSE, 1.0, A, B, 0.0, C);

    auto vC = as_strided_view<double>(C);
    EXPECT_NEAR(vC.values_[0], 19.0, 1e-12);
    EXPECT_NEAR(vC.values_[1], 43.0, 1e-12);
    EXPECT_NEAR(vC.values_[2], 22.0, 1e-12);
    EXPECT_NEAR(vC.values_[3], 50.0, 1e-12);
}

TEST_F(BlasTest, MmApplyDenseTransposeA) {
    // Same A, B as above. A^T * B = [[26, 30], [38, 44]].
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 3.0;
    vA.values_[2] = 2.0;
    vA.values_[3] = 4.0;

    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vB = as_strided_view<double>(B);
    vB.values_[0] = 5.0;
    vB.values_[1] = 7.0;
    vB.values_[2] = 6.0;
    vB.values_[3] = 8.0;

    Dense C = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    mm_apply(Device::CPU, TransposeType::TRANSPOSE, TransposeType::NO_TRANSPOSE,
             1.0, A, B, 0.0, C);

    auto vC = as_strided_view<double>(C);
    EXPECT_NEAR(vC.values_[0], 26.0, 1e-12);
    EXPECT_NEAR(vC.values_[1], 38.0, 1e-12);
    EXPECT_NEAR(vC.values_[2], 30.0, 1e-12);
    EXPECT_NEAR(vC.values_[3], 44.0, 1e-12);
}

TEST_F(BlasTest, MmApplyDenseBetaAccumulates) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vA = as_strided_view<double>(A);
    vA.values_[0] = 1.0;
    vA.values_[1] = 0.0;
    vA.values_[2] = 0.0;
    vA.values_[3] = 1.0;  // identity

    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vB = as_strided_view<double>(B);
    vB.values_[0] = 5.0;
    vB.values_[1] = 7.0;
    vB.values_[2] = 6.0;
    vB.values_[3] = 8.0;

    Dense C = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    auto vC_init = as_strided_view<double>(C);
    for (int i = 0; i < 4; ++i) vC_init.values_[i] = 1.0;

    // C = 1 * (I * B) + 1 * C  ==  B + 1.
    mm_apply(Device::CPU, TransposeType::NO_TRANSPOSE,
             TransposeType::NO_TRANSPOSE, 1.0, A, B, 1.0, C);

    auto vC = as_strided_view<double>(C);
    EXPECT_NEAR(vC.values_[0], 6.0, 1e-12);
    EXPECT_NEAR(vC.values_[1], 8.0, 1e-12);
    EXPECT_NEAR(vC.values_[2], 7.0, 1e-12);
    EXPECT_NEAR(vC.values_[3], 9.0, 1e-12);
}

// ---------------------------------------------------------------------------
// set_values (Dense / Sparse)
// ---------------------------------------------------------------------------

TEST_F(BlasTest, SetValuesDense) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 2});
    set_values(3.5, A);
    auto vA = as_strided_view<double>(A);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
            EXPECT_EQ(vA.values_[i + j * vA.lead_dim_], 3.5);
}

TEST_F(BlasTest, SetValuesSparse) {
    Sparse A = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {2, 2}, 4);
    set_values(2.5, A);
    auto vA = as_csr_view<double, int>(A);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(vA.values_[i], 2.5);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

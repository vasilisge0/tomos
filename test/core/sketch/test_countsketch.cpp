#include <gtest/gtest.h>

#include <stdexcept>

#include "context.hpp"
#include "countsketch.hpp"
#include "matrix.hpp"

namespace tomos {
namespace test {

class CountsketchTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

// generate_count_sketch (nnz_per_row == 1) should populate the trivial
// row_ptrs = {0, 1, ..., nrows}, one in-range random column per row, and
// +-1 values.
TEST_F(CountsketchTest, CreateCountSketchStructure) {
    dim<2> size = {4, 10};  // 4 buckets, 10 input rows.
    size_t nnz = 4;         // must equal nrows (one entry per row).
    Sparse S =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    generate_count_sketch(S);

    auto vS = as_csr_view<double, int>(S);
    for (int i = 0; i <= 4; ++i) EXPECT_EQ(vS.row_ptrs_[i], i);
    for (size_t p = 0; p < nnz; ++p) {
        EXPECT_GE(vS.col_idxs_[p], 0);
        EXPECT_LT(vS.col_idxs_[p], 10);
        EXPECT_TRUE(vS.values_[p] == 1.0 || vS.values_[p] == -1.0);
    }
}

// nnz must equal nrows (nnz_per_row == 1 is the only supported structure).
TEST_F(CountsketchTest, CreateCountSketchThrowsOnNnzMismatch) {
    Sparse S = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {4, 10}, /*nnz=*/6);
    EXPECT_THROW(generate_count_sketch(S), std::invalid_argument);
}

// target = sketch (CSR) * source (dense), verified against a hand-built
// sketch matrix (independent of the random generator in
// generate_count_sketch).
TEST_F(CountsketchTest, ApplyDenseSource) {
    // sketch: 3x4, rows are buckets, columns are input coordinates.
    // Row 0: +1 * col 0, -1 * col 2.  Row 1: +1 * col 1.  Row 2: +1 * col 3.
    Sparse S =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {3, 4}, 4);
    auto vS = as_csr_view<double, int>(S);
    vS.row_ptrs_[0] = 0;
    vS.row_ptrs_[1] = 2;
    vS.row_ptrs_[2] = 3;
    vS.row_ptrs_[3] = 4;
    vS.col_idxs_[0] = 0;
    vS.col_idxs_[1] = 2;
    vS.col_idxs_[2] = 1;
    vS.col_idxs_[3] = 3;
    vS.values_[0] = 1.0;
    vS.values_[1] = -1.0;
    vS.values_[2] = 1.0;
    vS.values_[3] = 1.0;

    // source: 4x2 dense, columns are two independent vectors to sketch.
    Dense X = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    auto vX = as_strided_view<double>(X);
    double x_data[8] = {1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0};
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 4; ++i)
            vX.values_[i + j * vX.lead_dim_] = x_data[i + j * 4];

    Dense Y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 2});
    apply_count_sketch(S, X, Y);

    auto vY = as_strided_view<double>(Y);
    // Column 0: row0 = 1*1 - 1*3 = -2, row1 = 1*2 = 2, row2 = 1*4 = 4.
    EXPECT_NEAR(vY.values_[0 + 0 * vY.lead_dim_], -2.0, 1e-12);
    EXPECT_NEAR(vY.values_[1 + 0 * vY.lead_dim_], 2.0, 1e-12);
    EXPECT_NEAR(vY.values_[2 + 0 * vY.lead_dim_], 4.0, 1e-12);
    // Column 1: row0 = 1*10 - 1*30 = -20, row1 = 1*20 = 20, row2 = 1*40 = 40.
    EXPECT_NEAR(vY.values_[0 + 1 * vY.lead_dim_], -20.0, 1e-12);
    EXPECT_NEAR(vY.values_[1 + 1 * vY.lead_dim_], 20.0, 1e-12);
    EXPECT_NEAR(vY.values_[2 + 1 * vY.lead_dim_], 40.0, 1e-12);
}

// target = sketch (CSR) * source (CSR), materialized as dense. Same sketch
// and numeric values as ApplyDenseSource, but the source is itself sparse.
TEST_F(CountsketchTest, ApplySparseSource) {
    Sparse S =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {3, 4}, 4);
    auto vS = as_csr_view<double, int>(S);
    vS.row_ptrs_[0] = 0;
    vS.row_ptrs_[1] = 2;
    vS.row_ptrs_[2] = 3;
    vS.row_ptrs_[3] = 4;
    vS.col_idxs_[0] = 0;
    vS.col_idxs_[1] = 2;
    vS.col_idxs_[2] = 1;
    vS.col_idxs_[3] = 3;
    vS.values_[0] = 1.0;
    vS.values_[1] = -1.0;
    vS.values_[2] = 1.0;
    vS.values_[3] = 1.0;

    // source: 4x2 sparse. Column 0 has rows {0:1, 2:3, 3:4}; column 1 has
    // rows {1:20}. Stored as CSR (rows are the 4 input coordinates).
    Sparse B =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {4, 2}, 4);
    auto vB = as_csr_view<double, int>(B);
    vB.row_ptrs_[0] = 0;
    vB.row_ptrs_[1] = 1;
    vB.row_ptrs_[2] = 2;
    vB.row_ptrs_[3] = 3;
    vB.row_ptrs_[4] = 4;
    vB.col_idxs_[0] = 0;  // row 0, col 0 -> 1.0
    vB.col_idxs_[1] = 1;  // row 1, col 1 -> 20.0
    vB.col_idxs_[2] = 0;  // row 2, col 0 -> 3.0
    vB.col_idxs_[3] = 0;  // row 3, col 0 -> 4.0
    vB.values_[0] = 1.0;
    vB.values_[1] = 20.0;
    vB.values_[2] = 3.0;
    vB.values_[3] = 4.0;

    Dense Y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 2});
    apply_count_sketch(S, B, Y);

    auto vY = as_strided_view<double>(Y);
    // Column 0: row0 = 1*1 - 1*3 = -2, row1 = 0, row2 = 1*4 = 4.
    EXPECT_NEAR(vY.values_[0 + 0 * vY.lead_dim_], -2.0, 1e-12);
    EXPECT_NEAR(vY.values_[1 + 0 * vY.lead_dim_], 0.0, 1e-12);
    EXPECT_NEAR(vY.values_[2 + 0 * vY.lead_dim_], 4.0, 1e-12);
    // Column 1: row0 = 0, row1 = 1*20 = 20, row2 = 0.
    EXPECT_NEAR(vY.values_[0 + 1 * vY.lead_dim_], 0.0, 1e-12);
    EXPECT_NEAR(vY.values_[1 + 1 * vY.lead_dim_], 20.0, 1e-12);
    EXPECT_NEAR(vY.values_[2 + 1 * vY.lead_dim_], 0.0, 1e-12);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
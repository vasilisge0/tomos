#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "context.hpp"
#include "matrix.hpp"
#include "sketch.hpp"

namespace tomos {
namespace test {

class GaussianSketchTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};

    // Coarse statistical sanity check: sample mean ~ 0, sample variance ~ 1.
    // Tolerances are generous (many std devs) to avoid flakiness while still
    // catching gross bugs (e.g. all-zero, constant, or +-1 values).
    static void expect_standard_normal(const std::vector<double>& values) {
        double n = static_cast<double>(values.size());
        double mean = 0.0;
        for (double v : values) mean += v;
        mean /= n;

        double var = 0.0;
        for (double v : values) var += (v - mean) * (v - mean);
        var /= n;

        EXPECT_NEAR(mean, 0.0, 0.1);
        EXPECT_NEAR(var, 1.0, 0.3);
    }
};

TEST_F(GaussianSketchTest, CreateDenseStatistics) {
    dim<2> size = {200, 50};  // 10000 elements.
    Dense S = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    size);
    create_gaussian_sketch(S);

    auto v = as_strided_view<double>(S);
    std::vector<double> values;
    values.reserve(size[0] * size[1]);
    for (size_t j = 0; j < size[1]; ++j)
        for (size_t i = 0; i < size[0]; ++i)
            values.push_back(v.values_[i + j * v.lead_dim_]);

    expect_standard_normal(values);
}

TEST_F(GaussianSketchTest, CreateDenseTwoCallsDiffer) {
    dim<2> size = {10, 10};
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    size);
    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    size);
    create_gaussian_sketch(A);
    create_gaussian_sketch(B);

    auto vA = as_strided_view<double>(A);
    auto vB = as_strided_view<double>(B);
    bool any_different = false;
    for (size_t j = 0; j < size[1]; ++j)
        for (size_t i = 0; i < size[0]; ++i)
            if (vA.values_[i + j * vA.lead_dim_] !=
                vB.values_[i + j * vB.lead_dim_])
                any_different = true;
    EXPECT_TRUE(any_different);
}

TEST_F(GaussianSketchTest, CreateSparseStructureAndStatistics) {
    dim<2> size = {500, 20};  // 500 buckets, 20 input dims.
    size_t nnz = 500;         // nnz_per_row == 1: nnz must equal nrows.
    Sparse S =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    create_gaussian_sketch(S);

    auto vS = as_csr_view<double, int>(S);
    for (size_t i = 0; i <= size[0]; ++i)
        EXPECT_EQ(vS.row_ptrs_[i], static_cast<int>(i));

    std::vector<double> values;
    values.reserve(nnz);
    for (size_t p = 0; p < nnz; ++p) {
        EXPECT_GE(vS.col_idxs_[p], 0);
        EXPECT_LT(vS.col_idxs_[p], static_cast<int>(size[1]));
        values.push_back(vS.values_[p]);
    }
    expect_standard_normal(values);
}

TEST_F(GaussianSketchTest, CreateSparseThrowsOnNnzMismatch) {
    Sparse S = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {4, 10}, /*nnz=*/6);
    EXPECT_THROW(create_gaussian_sketch(S), std::invalid_argument);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

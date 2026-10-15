#include <gtest/gtest.h>

#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "matrix.hpp"

namespace tomos {
namespace test {

class DenseTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

TEST_F(DenseTest, CreateStrided) {
    dim<2> size = {100, 50};
    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    EXPECT_EQ(A.get_size()[0], 100);
    EXPECT_EQ(A.get_size()[1], 50);
    EXPECT_EQ(A.get_device(), Device::CPU);
    EXPECT_EQ(A.get_value_type(), NumericType::FP64);
    EXPECT_EQ(A.get_format(), MatrixFormat::STRIDED);
}

TEST_F(DenseTest, CreateStridedWithLeadDim) {
    dim<2> size = {100, 50};
    size_t lead_dim = 128;
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    size, lead_dim);
    EXPECT_EQ(A.get_size()[0], 100);
    EXPECT_EQ(A.get_size()[1], 50);
    EXPECT_EQ(A.get_value_type(), NumericType::FP32);
}

TEST_F(DenseTest, CreateStridedAllPrecisions) {
    dim<2> size = {10, 10};
    NumericType types[] = {NumericType::FP64, NumericType::FP32,
                           NumericType::FP16};
    for (auto type : types) {
        Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, type, size);
        EXPECT_EQ(A.get_value_type(), type);
    }
}

TEST_F(DenseTest, CreateVector) {
    dim<2> col_vec = {100, 1};
    Dense v = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    col_vec);
    EXPECT_EQ(v.get_size()[0], 100);
    EXPECT_EQ(v.get_size()[1], 1);

    dim<2> row_vec = {1, 100};
    Dense w = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    row_vec);
    EXPECT_EQ(w.get_size()[0], 1);
    EXPECT_EQ(w.get_size()[1], 100);
}

TEST_F(DenseTest, MoveConstructor) {
    dim<2> size = {100, 50};
    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    Dense B = std::move(A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_device(), Device::CPU);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
}

TEST_F(DenseTest, MoveAssignment) {
    dim<2> size_a = {100, 50};
    dim<2> size_b = {200, 30};
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    size_a);
    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    size_b);
    B = std::move(A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
}

TEST_F(DenseTest, CopyFrom) {
    dim<2> size = {100, 50};
    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    Dense B =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    B.copy_from(A);
    EXPECT_EQ(B.get_size()[0], A.get_size()[0]);
    EXPECT_EQ(B.get_size()[1], A.get_size()[1]);
}

TEST_F(DenseTest, CreateFrom) {
    dim<2> size = {100, 50};
    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    Dense B = Dense::create_from(ctx, A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
}

// wrap_strided should alias the backing Dense's memory rather than copy it,
// and must not free that memory when the wrapper is destroyed first.
TEST_F(DenseTest, WrapStridedAliasesBackingMemory) {
    dim<2> size = {4, 1};
    Dense backing =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, size);
    auto v = as_strided_view<double>(backing);
    for (size_t i = 0; i < 4; ++i) v.values_[i] = static_cast<double>(i + 1);

    {
        Dense wrapped = Dense::wrap_strided(
            ctx, Layout::COL_MAJOR, NumericType::FP64, size, size[0],
            static_cast<void*>(v.values_));
        EXPECT_EQ(wrapped.get_size()[0], 4);
        EXPECT_EQ(wrapped.get_value_type(), NumericType::FP64);

        // Writes through the wrapper should be visible via the backing Dense.
        auto vw = as_strided_view<double>(wrapped);
        vw.values_[0] = 42.0;
        EXPECT_EQ(v.values_[0], 42.0);
    }  // wrapped destroyed here; backing's memory must survive.

    EXPECT_EQ(v.values_[0], 42.0);
    EXPECT_EQ(v.values_[1], 2.0);
    EXPECT_EQ(v.values_[2], 3.0);
    EXPECT_EQ(v.values_[3], 4.0);
}

// wrap_strided'd operands should be usable directly in mv_apply, since it's
// the actual reason for adding it: LSQR's working vectors need to remain
// real Dense objects (mv_apply dispatches on their runtime NumericType) even
// if their memory comes from an external, caller-owned buffer.
TEST_F(DenseTest, WrapStridedWorksWithMvApply) {
    dim<2> a_size = {3, 3};
    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, a_size);
    auto vA = as_strided_view<double>(A);
    for (size_t i = 0; i < 9; ++i) vA.values_[i] = 0.0;
    vA.values_[0] = 1.0;
    vA.values_[4] = 2.0;
    vA.values_[8] = 4.0;  // A = diag(1, 2, 4)

    std::vector<double> x_buf = {1.0, 4.0, 8.0};
    std::vector<double> y_buf = {0.0, 0.0, 0.0};
    Dense x = Dense::wrap_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                  {3, 1}, 3, x_buf.data());
    Dense y = Dense::wrap_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                  {3, 1}, 3, y_buf.data());

    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, y);

    EXPECT_EQ(y_buf[0], 1.0);
    EXPECT_EQ(y_buf[1], 8.0);
    EXPECT_EQ(y_buf[2], 32.0);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

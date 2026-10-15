#include <gtest/gtest.h>

#include "context.hpp"
#include "matrix.hpp"

namespace tomos {
namespace test {

class SparseTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

TEST_F(SparseTest, CreateCsr) {
    dim<2> size = {100, 50};
    size_t nnz = 500;
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    EXPECT_EQ(A.get_size()[0], 100);
    EXPECT_EQ(A.get_size()[1], 50);
    EXPECT_EQ(A.get_device(), Device::CPU);
    EXPECT_EQ(A.get_value_type(), NumericType::FP64);
    EXPECT_EQ(A.get_index_type(), IntType::INT32);
    // Sparse is always CSR, no format check needed
}

TEST_F(SparseTest, CreateCsrAllPrecisions) {
    dim<2> size = {10, 10};
    size_t nnz = 20;
    NumericType types[] = {NumericType::FP64, NumericType::FP32,
                           NumericType::FP16};
    for (auto type : types) {
        Sparse A = Sparse::create_csr(ctx, type, IntType::INT32, size, nnz);
        EXPECT_EQ(A.get_value_type(), type);
    }
}

TEST_F(SparseTest, CreateCsrAllIndexTypes) {
    dim<2> size = {10, 10};
    size_t nnz = 20;
    IntType idx_types[] = {IntType::INT32, IntType::INT64};
    for (auto idx : idx_types) {
        Sparse A = Sparse::create_csr(ctx, NumericType::FP64, idx, size, nnz);
        EXPECT_EQ(A.get_index_type(), idx);
    }
}

TEST_F(SparseTest, MoveConstructor) {
    dim<2> size = {100, 50};
    size_t nnz = 500;
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    Sparse B = std::move(A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_device(), Device::CPU);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
    EXPECT_EQ(B.get_index_type(), IntType::INT32);
}

TEST_F(SparseTest, MoveAssignment) {
    dim<2> size_a = {100, 50};
    dim<2> size_b = {200, 30};
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size_a, 500);
    Sparse B =
        Sparse::create_csr(ctx, NumericType::FP32, IntType::INT64, size_b, 300);
    B = std::move(A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
    EXPECT_EQ(B.get_index_type(), IntType::INT32);
}

// TODO: enable once csr_copy_impl is wired up
TEST_F(SparseTest, CopyFrom) {
    dim<2> size = {100, 50};
    size_t nnz = 500;
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    Sparse B =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    B.copy_from(A);
    EXPECT_EQ(B.get_size()[0], A.get_size()[0]);
    EXPECT_EQ(B.get_size()[1], A.get_size()[1]);
}

TEST_F(SparseTest, CreateFrom) {
    dim<2> size = {100, 50};
    size_t nnz = 500;
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, size, nnz);
    Sparse B = Sparse::create_from(ctx, A);
    EXPECT_EQ(B.get_size()[0], 100);
    EXPECT_EQ(B.get_size()[1], 50);
    EXPECT_EQ(B.get_value_type(), NumericType::FP64);
    EXPECT_EQ(B.get_index_type(), IntType::INT32);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

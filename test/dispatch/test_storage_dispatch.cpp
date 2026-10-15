#include <gtest/gtest.h>

#include "dense_storage.hpp"
#include "device.hpp"
#include "sparse_storage.hpp"
#include "storage_dispatch.hpp"

namespace tomos {
namespace test {

// ---------------------------------------------------------------------------
// get_numeric_type / get_size / get_index_type -- pure accessors reading the
// active alternative out of the NumericPtrVariant/IntPtrVariant. Constructing
// StridedStorage/CsrStorage directly (rather than via Dense::create_strided/
// Sparse::create_csr) exercises these without needing a real allocation.
// ---------------------------------------------------------------------------

TEST(StorageDispatchTest, GetNumericTypeDense) {
    DenseStorage ds =
        StridedStorage(Layout::COL_MAJOR, {3, 4}, 3, NumericType::FP32);
    EXPECT_EQ(dispatch::get_numeric_type(ds), NumericType::FP32);
}

TEST(StorageDispatchTest, GetSizeDense) {
    DenseStorage ds =
        StridedStorage(Layout::COL_MAJOR, {3, 4}, 3, NumericType::FP64);
    dim<2> size = dispatch::get_size(ds);
    EXPECT_EQ(size[0], 3u);
    EXPECT_EQ(size[1], 4u);
}

TEST(StorageDispatchTest, GetNumericTypeAndIndexTypeSparse) {
    SparseStorage ss =
        CsrStorage({5, 6}, 10, NumericType::FP32, IntType::INT64);
    EXPECT_EQ(dispatch::get_numeric_type(ss), NumericType::FP32);
    EXPECT_EQ(dispatch::get_index_type(ss), IntType::INT64);
}

TEST(StorageDispatchTest, GetSizeSparse) {
    SparseStorage ss =
        CsrStorage({5, 6}, 10, NumericType::FP64, IntType::INT32);
    dim<2> size = dispatch::get_size(ss);
    EXPECT_EQ(size[0], 5u);
    EXPECT_EQ(size[1], 6u);
}

// ---------------------------------------------------------------------------
// initialize_dense_storage / free_dense_storage -- called directly here
// (rather than through Dense's constructor/destructor, which is what every
// other test in the suite exercises this through) to confirm the dispatch
// entry points themselves work in isolation.
// ---------------------------------------------------------------------------

TEST(StorageDispatchTest, InitializeAndFreeDenseStorage) {
    DenseStorage ds =
        StridedStorage(Layout::COL_MAJOR, {4, 4}, 4, NumericType::FP64);
    dispatch::initialize_dense_storage({4, 4}, Device::CPU, ds);

    auto& S = std::get<StridedStorage>(ds);
    double* values = std::get<double*>(S.values_);
    ASSERT_NE(values, nullptr);
    values[0] = 1.5;  // writable: a real allocation, not a dangling pointer.
    EXPECT_EQ(values[0], 1.5);

    dispatch::free_dense_storage(Device::CPU, ds);
}

TEST(StorageDispatchTest, CopyDenseStorageCopiesValues) {
    DenseStorage source =
        StridedStorage(Layout::COL_MAJOR, {2, 2}, 2, NumericType::FP64);
    DenseStorage target =
        StridedStorage(Layout::COL_MAJOR, {2, 2}, 2, NumericType::FP64);
    dispatch::initialize_dense_storage({2, 2}, Device::CPU, source);
    dispatch::initialize_dense_storage({2, 2}, Device::CPU, target);

    double* src_values = std::get<double*>(std::get<StridedStorage>(source).values_);
    for (int i = 0; i < 4; ++i) src_values[i] = static_cast<double>(i + 1);

    dispatch::copy_dense_storage({2, 2}, Device::CPU, Device::CPU, source, target);

    double* tgt_values = std::get<double*>(std::get<StridedStorage>(target).values_);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(tgt_values[i], static_cast<double>(i + 1));

    dispatch::free_dense_storage(Device::CPU, source);
    dispatch::free_dense_storage(Device::CPU, target);
}

TEST(StorageDispatchTest, InitializeAndFreeSparseStorage) {
    SparseStorage ss = CsrStorage({3, 3}, 3, NumericType::FP64, IntType::INT32);
    dispatch::initialize_sparse_storage(Device::CPU, ss);

    auto& S = std::get<CsrStorage>(ss);
    EXPECT_NE(std::get<double*>(S.values_), nullptr);
    EXPECT_NE(std::get<int*>(S.row_ptrs_), nullptr);
    EXPECT_NE(std::get<int*>(S.col_idxs_), nullptr);

    dispatch::free_sparse_storage(Device::CPU, ss);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

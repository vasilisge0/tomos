#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#include "copy_cuda_kernels.cuh"
#include "sparse_storage.hpp"

namespace tomos {
namespace test {

// Round-trips a 2x2 CSR matrix (3 nonzeros) host -> device -> host via
// csr_copy_to_device / csr_copy_from_device.
TEST(CsrCopyCudaKernelTest, HostToDeviceToHostRoundTrip) {
    dim<2> size{2, 2};
    size_t nnz = 3;

    std::vector<double> values_h = {2.0, 3.0, 4.0};
    std::vector<int> row_ptrs_h = {0, 1, 3};
    std::vector<int> col_idxs_h = {0, 0, 1};

    SparseStorage host_storage =
        CsrStorage(size, nnz, NumericPtrVariant{values_h.data()},
                  IntPtrVariant{row_ptrs_h.data()},
                  IntPtrVariant{col_idxs_h.data()}, IntType::INT32);

    double* values_d;
    int *row_ptrs_d, *col_idxs_d;
    cudaMalloc(&values_d, nnz * sizeof(double));
    cudaMalloc(&row_ptrs_d, (size[0] + 1) * sizeof(int));
    cudaMalloc(&col_idxs_d, nnz * sizeof(int));
    SparseStorage device_storage =
        CsrStorage(size, nnz, NumericPtrVariant{values_d},
                  IntPtrVariant{row_ptrs_d}, IntPtrVariant{col_idxs_d},
                  IntType::INT32);

    cuda::csr_copy_to_device<double, int>(size, host_storage, device_storage);

    std::vector<double> values_out(nnz, -1.0);
    std::vector<int> row_ptrs_out(size[0] + 1, -1);
    std::vector<int> col_idxs_out(nnz, -1);
    SparseStorage readback_storage =
        CsrStorage(size, nnz, NumericPtrVariant{values_out.data()},
                  IntPtrVariant{row_ptrs_out.data()},
                  IntPtrVariant{col_idxs_out.data()}, IntType::INT32);

    cuda::csr_copy_from_device<double, int>(size, device_storage,
                                            readback_storage);

    for (size_t i = 0; i < nnz; ++i) EXPECT_DOUBLE_EQ(values_out[i], values_h[i]);
    for (size_t i = 0; i < size[0] + 1; ++i)
        EXPECT_EQ(row_ptrs_out[i], row_ptrs_h[i]);
    for (size_t i = 0; i < nnz; ++i) EXPECT_EQ(col_idxs_out[i], col_idxs_h[i]);

    cudaFree(values_d);
    cudaFree(row_ptrs_d);
    cudaFree(col_idxs_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

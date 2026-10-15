#include <gtest/gtest.h>

#include <cusparse.h>
#include <cuda_runtime.h>

#include <vector>

#include "mv_cuda_kernels.cuh"
#include "transpose.hpp"

namespace tomos {
namespace test {

// A (2x3, CSR): [[1,2,0],[0,0,3]]
// row_ptrs = {0,2,3}, col_idxs = {0,1,2}, values = {1,2,3}
class CsrMvCudaTest : public ::testing::Test {
protected:
    dim<2> size{2, 3};
    std::vector<int> row_ptrs_h{0, 2, 3};
    std::vector<int> col_idxs_h{0, 1, 2};
    std::vector<double> values_h{1.0, 2.0, 3.0};
    cusparseHandle_t handle;

    int *row_ptrs_d, *col_idxs_d;
    double* values_d;

    void SetUp() override {
        cusparseCreate(&handle);
        cudaMalloc(&row_ptrs_d, row_ptrs_h.size() * sizeof(int));
        cudaMalloc(&col_idxs_d, col_idxs_h.size() * sizeof(int));
        cudaMalloc(&values_d, values_h.size() * sizeof(double));
        cudaMemcpy(row_ptrs_d, row_ptrs_h.data(),
                  row_ptrs_h.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(col_idxs_d, col_idxs_h.data(),
                  col_idxs_h.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(values_d, values_h.data(), values_h.size() * sizeof(double),
                  cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(row_ptrs_d);
        cudaFree(col_idxs_d);
        cudaFree(values_d);
        cusparseDestroy(handle);
    }
};

TEST_F(CsrMvCudaTest, NoTranspose) {
    std::vector<double> x_h = {1.0, 1.0, 1.0};
    double *x_d, *y_d;
    cudaMalloc(&x_d, 3 * sizeof(double));
    cudaMalloc(&y_d, 2 * sizeof(double));
    cudaMemcpy(x_d, x_h.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

    cuda::mv_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, 1.0, size, values_h.size(), values_d,
        row_ptrs_d, col_idxs_d, x_d, 1, 0.0, y_d, 1, handle);

    std::vector<double> y_h(2);
    cudaMemcpy(y_h.data(), y_d, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(y_h[0], 3.0, 1e-12);
    EXPECT_NEAR(y_h[1], 3.0, 1e-12);

    cudaFree(x_d);
    cudaFree(y_d);
}

TEST_F(CsrMvCudaTest, Transpose) {
    std::vector<double> x_h = {1.0, 1.0};
    double *x_d, *y_d;
    cudaMalloc(&x_d, 2 * sizeof(double));
    cudaMalloc(&y_d, 3 * sizeof(double));
    cudaMemcpy(x_d, x_h.data(), 2 * sizeof(double), cudaMemcpyHostToDevice);

    cuda::mv_csr<double, double, double, int>(
        TransposeType::TRANSPOSE, 1.0, size, values_h.size(), values_d,
        row_ptrs_d, col_idxs_d, x_d, 1, 0.0, y_d, 1, handle);

    std::vector<double> y_h(3);
    cudaMemcpy(y_h.data(), y_d, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(y_h[0], 1.0, 1e-12);
    EXPECT_NEAR(y_h[1], 2.0, 1e-12);
    EXPECT_NEAR(y_h[2], 3.0, 1e-12);

    cudaFree(x_d);
    cudaFree(y_d);
}

TEST_F(CsrMvCudaTest, AlphaBetaScaling) {
    std::vector<double> x_h = {1.0, 1.0, 1.0};
    std::vector<double> y_h = {100.0, 100.0};
    double *x_d, *y_d;
    cudaMalloc(&x_d, 3 * sizeof(double));
    cudaMalloc(&y_d, 2 * sizeof(double));
    cudaMemcpy(x_d, x_h.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h.data(), 2 * sizeof(double), cudaMemcpyHostToDevice);

    cuda::mv_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, 2.0, size, values_h.size(), values_d,
        row_ptrs_d, col_idxs_d, x_d, 1, 0.5, y_d, 1, handle);

    std::vector<double> y_out(2);
    cudaMemcpy(y_out.data(), y_d, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    // y = 2*(A*x) + 0.5*y = 6 + 50 = 56.
    EXPECT_NEAR(y_out[0], 56.0, 1e-12);
    EXPECT_NEAR(y_out[1], 56.0, 1e-12);

    cudaFree(x_d);
    cudaFree(y_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

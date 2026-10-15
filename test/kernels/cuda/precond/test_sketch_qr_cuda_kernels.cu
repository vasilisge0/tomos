#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "dimensions.hpp"
#include "sketch_qr_cuda_kernels.cuh"

namespace tomos {
namespace test {

// generate_sketch_qr_precond should leave R^{-1} in the leading n x n block
// of `values`, where R comes from the QR factorization of the original
// (k x n) matrix on entry. Verified via the identity M^T (A^T A) M == I,
// where M = R^{-1} and A^T A = R^T R for any valid QR factorization -- true
// regardless of sign/basis conventions. Same matrix and approach as
// test/core/precond/test_sketch_qr.cpp's CPU-level test, but exercising the
// CUDA kernel directly on device pointers.
TEST(SketchQrCudaKernelTest, ProducesRInverse) {
    dim<2> size{4, 2};
    // A (4x2), column-major: col0 = {1,0,1,1}, col1 = {0,1,1,-1}.
    double a_col0[4] = {1.0, 0.0, 1.0, 1.0};
    double a_col1[4] = {0.0, 1.0, 1.0, -1.0};
    std::vector<double> A_h(8);
    for (int i = 0; i < 4; ++i) {
        A_h[i] = a_col0[i];
        A_h[4 + i] = a_col1[i];
    }

    double ata[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < 4; ++i) {
        ata[0][0] += a_col0[i] * a_col0[i];
        ata[0][1] += a_col0[i] * a_col1[i];
        ata[1][0] += a_col1[i] * a_col0[i];
        ata[1][1] += a_col1[i] * a_col1[i];
    }

    double* A_d;
    cudaMalloc(&A_d, 8 * sizeof(double));
    cudaMemcpy(A_d, A_h.data(), 8 * sizeof(double), cudaMemcpyHostToDevice);

    cuda::generate_sketch_qr_precond<double>(size, A_d, 4);

    std::vector<double> M_h(8);
    cudaMemcpy(M_h.data(), A_d, 8 * sizeof(double), cudaMemcpyDeviceToHost);
    double m[2][2];
    m[0][0] = M_h[0];
    m[1][0] = M_h[1];
    m[0][1] = M_h[4];
    m[1][1] = M_h[5];

    double tmp[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) tmp[i][j] += m[k][i] * ata[k][j];

    double result[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) result[i][j] += tmp[i][k] * m[k][j];

    EXPECT_NEAR(result[0][0], 1.0, 1e-8);
    EXPECT_NEAR(result[1][1], 1.0, 1e-8);
    EXPECT_NEAR(result[0][1], 0.0, 1e-8);
    EXPECT_NEAR(result[1][0], 0.0, 1e-8);

    // R^{-1} is upper triangular; the sub-diagonal must be exactly zero.
    EXPECT_EQ(m[1][0], 0.0);

    cudaFree(A_d);
}

TEST(SketchQrCudaKernelTest, ThrowsWhenFewerRowsThanCols) {
    dim<2> size{2, 4};
    double* A_d;
    cudaMalloc(&A_d, 8 * sizeof(double));
    EXPECT_THROW(cuda::generate_sketch_qr_precond<double>(size, A_d, 2),
                std::invalid_argument);
    cudaFree(A_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

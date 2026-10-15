// Basic correctness tests for the "promoted" FP32_16 kernels actually wired
// into src/dispatch/fp32_16_registries.cpp (split_/combine_/mv_fast_/
// mv_accurate_/mm_accurate_ slots) -- NOT the dozens of exploratory kernels
// in fp32_16_cuda_kernels.cu that aren't referenced by any registry, which
// already have their own dedicated correctness sweeps
// (benchmarks/kernels/cuda/matrix/verify_mv_fp32_16_gemv.cu /
// verify_mv_fp32_transpose_promoted.cu). mv_pure_half's TRANSPOSE tier is
// intentionally not covered here: it takes an optional scratch workspace
// (see Workspace::mv_transpose_scratch_) and is env-var-gated between two
// kernel variants (TOMOS_FP32_16_PURE_HALF_FAST_DECODE) -- exactly the kind
// of tuning-sensitive surface those dedicated sweeps already own.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "fp32_16_cuda_kernels.cuh"
#include "transpose.hpp"

namespace tomos {
namespace test {

// A (2x2, col-major): [[1.5, 2.25], [3.75, 4.125]]
// Chosen as exact binary fractions so split+combine round-trips exactly.
class Fp32_16CudaKernelTest : public ::testing::Test {
protected:
    dim<2> size{2, 2};
    std::vector<float> A_h{1.5f, 3.75f, 2.25f, 4.125f};
    float* A_d;
    uint16_t *wh_d, *wr_d;

    void SetUp() override {
        cudaMalloc(&A_d, 4 * sizeof(float));
        cudaMalloc(&wh_d, 4 * sizeof(uint16_t));
        cudaMalloc(&wr_d, 4 * sizeof(uint16_t));
        cudaMemcpy(A_d, A_h.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        cuda::split_dense_fp32_16(size, A_d, 2, wh_d, wr_d, 2);
    }

    void TearDown() override {
        cudaFree(A_d);
        cudaFree(wh_d);
        cudaFree(wr_d);
    }
};

TEST_F(Fp32_16CudaKernelTest, SplitCombineRoundTripsExactly) {
    float* recon_d;
    cudaMalloc(&recon_d, 4 * sizeof(float));
    cuda::combine_dense_fp32_16(size, wh_d, wr_d, 2, recon_d, 2);

    std::vector<float> recon_h(4);
    cudaMemcpy(recon_h.data(), recon_d, 4 * sizeof(float),
              cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(recon_h[i], A_h[i]);
    cudaFree(recon_d);
}

TEST_F(Fp32_16CudaKernelTest, MvFastNoTransposeMatchesReference) {
    std::vector<float> x_h = {1.0f, 1.0f};
    float *x_d, *y_d;
    cudaMalloc(&x_d, 2 * sizeof(float));
    cudaMalloc(&y_d, 2 * sizeof(float));
    cudaMemcpy(x_d, x_h.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);

    cuda::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, size, wh_d, 2, x_d,
                          1, 0.0, y_d, 1);

    std::vector<float> y_h(2);
    cudaMemcpy(y_h.data(), y_d, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    // A*x = (1.5+2.25, 3.75+4.125) = (3.75, 7.875)
    EXPECT_NEAR(y_h[0], 3.75f, 1e-3f);
    EXPECT_NEAR(y_h[1], 7.875f, 1e-3f);

    cudaFree(x_d);
    cudaFree(y_d);
}

TEST_F(Fp32_16CudaKernelTest, MvAccurateNoTransposeMatchesReference) {
    std::vector<float> x_h = {1.0f, 1.0f};
    float *x_d, *y_d;
    cudaMalloc(&x_d, 2 * sizeof(float));
    cudaMalloc(&y_d, 2 * sizeof(float));
    cudaMemcpy(x_d, x_h.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);

    cuda::experimental::mv_fp32_16_accurate_fast(TransposeType::NO_TRANSPOSE,
                                                 1.0, size, wh_d, wr_d, 2, x_d,
                                                 1, 0.0, y_d, 1);

    std::vector<float> y_h(2);
    cudaMemcpy(y_h.data(), y_d, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(y_h[0], 3.75f, 1e-4f);
    EXPECT_NEAR(y_h[1], 7.875f, 1e-4f);

    cudaFree(x_d);
    cudaFree(y_d);
}

TEST_F(Fp32_16CudaKernelTest, MmAccurateNoTransposeMatchesReference) {
    // B (2x2, col-major identity)
    std::vector<float> B_h = {1.0f, 0.0f, 0.0f, 1.0f};
    float *B_d, *C_d;
    cudaMalloc(&B_d, 4 * sizeof(float));
    cudaMalloc(&C_d, 4 * sizeof(float));
    cudaMemcpy(B_d, B_h.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                              TransposeType::NO_TRANSPOSE, 1.0, size, wh_d,
                              wr_d, 2, size, B_d, 2, 0.0, C_d, 2);

    std::vector<float> C_h(4);
    cudaMemcpy(C_h.data(), C_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    // A * I == A
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(C_h[i], A_h[i], 1e-4f);

    cudaFree(B_d);
    cudaFree(C_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#include "dimensions.hpp"
#include "gaussian_sketch_cuda_kernels.cuh"

namespace tomos {
namespace test {

// Unlike the CPU version (Status generate_gaussian_values(...)), the CUDA
// kernel returns void and seeds from wall-clock time via Thrust
// (thrust::minstd_rand + a SplitMix64 hash), not curand -- not
// bit-reproducible across runs, so this only checks statistical properties,
// not exact values.

TEST(GenerateGaussianValuesCudaKernelTest, FillsEveryEntry) {
    constexpr size_t N = 20000;
    dim<2> size{N, 1};
    double* values_d;
    cudaMalloc(&values_d, N * sizeof(double));
    std::vector<double> init(N, 0.0);
    cudaMemcpy(values_d, init.data(), N * sizeof(double),
              cudaMemcpyHostToDevice);

    cuda::generate_gaussian_values<double>(size, values_d, N);

    std::vector<double> values_h(N);
    cudaMemcpy(values_h.data(), values_d, N * sizeof(double),
              cudaMemcpyDeviceToHost);

    bool any_nonzero = false;
    bool any_different = false;
    for (size_t i = 1; i < N; ++i) {
        if (values_h[i] != 0.0) any_nonzero = true;
        if (values_h[i] != values_h[0]) any_different = true;
    }
    EXPECT_TRUE(any_nonzero);
    EXPECT_TRUE(any_different);

    cudaFree(values_d);
}

TEST(GenerateGaussianValuesCudaKernelTest, SampleMeanAndVarianceAreReasonable) {
    constexpr size_t N = 100000;
    dim<2> size{N, 1};
    double* values_d;
    cudaMalloc(&values_d, N * sizeof(double));

    cuda::generate_gaussian_values<double>(size, values_d, N);

    std::vector<double> values_h(N);
    cudaMemcpy(values_h.data(), values_d, N * sizeof(double),
              cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (double v : values_h) sum += v;
    double mean = sum / N;

    double sq_sum = 0.0;
    for (double v : values_h) sq_sum += (v - mean) * (v - mean);
    double variance = sq_sum / N;

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(variance, 1.0, 0.05);

    cudaFree(values_d);
}

TEST(GenerateGaussianValuesCudaKernelTest, RespectsLeadingDimensionPadding) {
    dim<2> size{2, 2};
    double* values_d;
    cudaMalloc(&values_d, 6 * sizeof(double));
    std::vector<double> init(6, -999.0);
    cudaMemcpy(values_d, init.data(), 6 * sizeof(double),
              cudaMemcpyHostToDevice);

    cuda::generate_gaussian_values<double>(size, values_d, 3);

    std::vector<double> values_h(6);
    cudaMemcpy(values_h.data(), values_d, 6 * sizeof(double),
              cudaMemcpyDeviceToHost);
    EXPECT_NE(values_h[0], -999.0);
    EXPECT_NE(values_h[1], -999.0);
    EXPECT_DOUBLE_EQ(values_h[2], -999.0);  // padding, untouched
    EXPECT_NE(values_h[3], -999.0);
    EXPECT_NE(values_h[4], -999.0);
    EXPECT_DOUBLE_EQ(values_h[5], -999.0);  // padding, untouched

    cudaFree(values_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#include "dimensions.hpp"
#include "set_values_cuda_kernels.cuh"

namespace tomos {
namespace test {

TEST(SetValuesCudaKernelTest, FillsWholeMatrixNoPadding) {
    dim<2> size{2, 3};
    double* values_d;
    cudaMalloc(&values_d, 6 * sizeof(double));

    cuda::set_values_impl<double>(9.0, size, values_d, 2);

    std::vector<double> values_h(6, -1.0);
    cudaMemcpy(values_h.data(), values_d, 6 * sizeof(double),
              cudaMemcpyDeviceToHost);
    for (double v : values_h) EXPECT_DOUBLE_EQ(v, 9.0);

    cudaFree(values_d);
}

TEST(SetValuesCudaKernelTest, RespectsLeadingDimensionPadding) {
    dim<2> size{2, 2};
    double* values_d;
    cudaMalloc(&values_d, 6 * sizeof(double));
    std::vector<double> init(6, -1.0);
    cudaMemcpy(values_d, init.data(), 6 * sizeof(double),
              cudaMemcpyHostToDevice);

    cuda::set_values_impl<double>(9.0, size, values_d, 3);

    std::vector<double> values_h(6);
    cudaMemcpy(values_h.data(), values_d, 6 * sizeof(double),
              cudaMemcpyDeviceToHost);
    EXPECT_DOUBLE_EQ(values_h[0], 9.0);
    EXPECT_DOUBLE_EQ(values_h[1], 9.0);
    EXPECT_DOUBLE_EQ(values_h[2], -1.0);  // padding, untouched
    EXPECT_DOUBLE_EQ(values_h[3], 9.0);
    EXPECT_DOUBLE_EQ(values_h[4], 9.0);
    EXPECT_DOUBLE_EQ(values_h[5], -1.0);  // padding, untouched

    cudaFree(values_d);
}

TEST(SetValuesCudaKernelTest, FloatType) {
    dim<2> size{2, 2};
    float* values_d;
    cudaMalloc(&values_d, 4 * sizeof(float));

    cuda::set_values_impl<float>(2.5f, size, values_d, 2);

    std::vector<float> values_h(4, -1.0f);
    cudaMemcpy(values_h.data(), values_d, 4 * sizeof(float),
              cudaMemcpyDeviceToHost);
    for (float v : values_h) EXPECT_FLOAT_EQ(v, 2.5f);

    cudaFree(values_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

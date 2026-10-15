#include <gtest/gtest.h>

#include "device.hpp"
#include "dimensions.hpp"
#include "malloc.hpp"
#include "set_values.hpp"

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace tomos {
namespace test {

TEST(SetValuesTest, CpuSetValuesDoubleNoPadding) {
    double* values = nullptr;
    malloc<Device::CPU, double>(6, &values);
    dim<2> size = {2, 3};
    set_values_impl<double>(Device::CPU, 7.0, size, values, /*ld=*/2);
    for (size_t i = 0; i < 6; ++i) EXPECT_EQ(values[i], 7.0);
    free<Device::CPU, double>(values);
}

TEST(SetValuesTest, CpuSetValuesRespectsLeadDimPadding) {
    // 2x2 logical matrix stored with lead_dim 3: only the first two entries
    // of each 3-element column should be touched.
    double* values = nullptr;
    malloc<Device::CPU, double>(6, &values);
    for (size_t i = 0; i < 6; ++i) values[i] = -1.0;

    dim<2> size = {2, 2};
    set_values_impl<double>(Device::CPU, 9.0, size, values, /*ld=*/3);

    EXPECT_EQ(values[0], 9.0);
    EXPECT_EQ(values[1], 9.0);
    EXPECT_EQ(values[2], -1.0);  // padding row, untouched
    EXPECT_EQ(values[3], 9.0);
    EXPECT_EQ(values[4], 9.0);
    EXPECT_EQ(values[5], -1.0);  // padding row, untouched

    free<Device::CPU, double>(values);
}

TEST(SetValuesTest, CpuZerosImpl) {
    double* values = nullptr;
    malloc<Device::CPU, double>(4, &values);
    for (size_t i = 0; i < 4; ++i) values[i] = 42.0;

    dim<2> size = {2, 2};
    zeros_impl<double>(Device::CPU, size, values, /*ld=*/2);

    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(values[i], 0.0);
    free<Device::CPU, double>(values);
}

TEST(SetValuesTest, CpuSetValuesFloat) {
    float* values = nullptr;
    malloc<Device::CPU, float>(4, &values);
    dim<2> size = {2, 2};
    set_values_impl<float>(Device::CPU, 1.5f, size, values, /*ld=*/2);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(values[i], 1.5f);
    free<Device::CPU, float>(values);
}

#if defined(USE_CUDA)

TEST(SetValuesTest, CudaSetValuesDouble) {
    double* d_values = nullptr;
    malloc<Device::CUDA, double>(4, &d_values);
    dim<2> size = {2, 2};
    set_values_impl<double>(Device::CUDA, 3.0, size, d_values, /*ld=*/2);

    double h_values[4] = {0, 0, 0, 0};
    cudaMemcpy(h_values, d_values, sizeof(h_values), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(h_values[i], 3.0);

    free<Device::CUDA, double>(d_values);
}

#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>

#include <cstddef>

#include "device.hpp"
#include "malloc.hpp"
#include "status.hpp"

namespace tomos {
namespace test {

TEST(MallocTest, CpuMallocAndFreeDouble) {
    double* ptr = nullptr;
    Status s = malloc<Device::CPU, double>(16, &ptr);
    EXPECT_EQ(s, SUCCESS);
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < 16; ++i) ptr[i] = static_cast<double>(i);
    for (size_t i = 0; i < 16; ++i) EXPECT_EQ(ptr[i], static_cast<double>(i));
    EXPECT_EQ((free<Device::CPU, double>(ptr)), SUCCESS);
}

TEST(MallocTest, CpuMallocFloat) {
    float* ptr = nullptr;
    Status s = malloc<Device::CPU, float>(8, &ptr);
    EXPECT_EQ(s, SUCCESS);
    ASSERT_NE(ptr, nullptr);
    free<Device::CPU, float>(ptr);
}

TEST(MallocTest, CpuMallocStdByte) {
    std::byte* ptr = nullptr;
    Status s = malloc<Device::CPU, std::byte>(1024, &ptr);
    EXPECT_EQ(s, SUCCESS);
    ASSERT_NE(ptr, nullptr);
    free<Device::CPU, std::byte>(ptr);
}

#if defined(USE_CUDA)

TEST(MallocTest, CudaMallocAndFreeDouble) {
    double* ptr = nullptr;
    Status s = malloc<Device::CUDA, double>(16, &ptr);
    EXPECT_EQ(s, SUCCESS);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ((free<Device::CUDA, double>(ptr)), SUCCESS);
}

#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

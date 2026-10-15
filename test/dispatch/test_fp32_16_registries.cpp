#include <gtest/gtest.h>

#include <stdexcept>

#include "device.hpp"
#include "fp32_16_registries.hpp"

namespace tomos {
namespace test {

TEST(Fp32_16RegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::Fp32_16::get();
    const auto& b = dispatch::Fp32_16::get();
    EXPECT_EQ(&a, &b);
}

TEST(Fp32_16RegistriesTest, LookupSplitThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Fp32_16::get().lookup_split(Device::OMP),
                 std::runtime_error);
}

TEST(Fp32_16RegistriesTest, LookupMvPureHalfThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Fp32_16::get().lookup_mv_pure_half(Device::OMP),
                 std::runtime_error);
}

// lookup_mm_workspace is CUDA-only by design -- no CPU kernel exists at all,
// so Device::CPU throws unconditionally, regardless of whether this build
// has CUDA support (see fp32_16_registries.hpp's own comment on
// lookup_mm_workspace).
TEST(Fp32_16RegistriesTest, LookupMmWorkspaceThrowsOnCpu) {
    EXPECT_THROW(dispatch::Fp32_16::get().lookup_mm_workspace(Device::CPU),
                 std::runtime_error);
}

TEST(Fp32_16RegistriesTest, LookupSplitDoesNotThrowOnCpu) {
    EXPECT_NO_THROW(dispatch::Fp32_16::get().lookup_split(Device::CPU));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

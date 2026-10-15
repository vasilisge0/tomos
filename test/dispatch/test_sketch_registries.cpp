#include <gtest/gtest.h>

#include <stdexcept>

#include "device.hpp"
#include "precision.hpp"
#include "sketch_registries.hpp"

namespace tomos {
namespace test {

TEST(SketchRegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::Sketch::get();
    const auto& b = dispatch::Sketch::get();
    EXPECT_EQ(&a, &b);
}

TEST(SketchRegistriesTest, LookupGenerateValuesThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sketch::get().lookup_generate_count_sketch_values(
                     Device::OMP, NumericType::FP64),
                 std::runtime_error);
}

TEST(SketchRegistriesTest, LookupGenerateIdxsThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sketch::get().lookup_generate_count_sketch_idxs(
                     Device::OMP, IntType::INT32),
                 std::runtime_error);
}

TEST(SketchRegistriesTest, LookupApplyDenseThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sketch::get().lookup_apply_count_sketch_dense(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, IntType::INT32),
                 std::runtime_error);
}

TEST(SketchRegistriesTest, LookupGenerateValuesDoesNotThrowOnCpu) {
    EXPECT_NO_THROW(dispatch::Sketch::get().lookup_generate_count_sketch_values(
        Device::CPU, NumericType::FP64));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

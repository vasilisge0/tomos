#include <gtest/gtest.h>

#include <stdexcept>

#include "device.hpp"
#include "gaussian_sketch_registries.hpp"
#include "precision.hpp"

namespace tomos {
namespace test {

TEST(GaussianSketchRegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::GaussianSketch::get();
    const auto& b = dispatch::GaussianSketch::get();
    EXPECT_EQ(&a, &b);
}

TEST(GaussianSketchRegistriesTest, LookupCreateDenseThrowsOnUnregisteredDevice) {
    EXPECT_THROW(
        dispatch::GaussianSketch::get().lookup_create_dense(Device::OMP,
                                                             NumericType::FP64),
        std::runtime_error);
}

TEST(GaussianSketchRegistriesTest, LookupCreateSparseThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::GaussianSketch::get().lookup_create_sparse(
                     Device::OMP, NumericType::FP64, IntType::INT32),
                 std::runtime_error);
}

TEST(GaussianSketchRegistriesTest, LookupCreateDenseDoesNotThrowOnCpu) {
    EXPECT_NO_THROW(dispatch::GaussianSketch::get().lookup_create_dense(
        Device::CPU, NumericType::FP64));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

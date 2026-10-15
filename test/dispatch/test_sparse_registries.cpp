#include <gtest/gtest.h>

#include <stdexcept>

#include "device.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"
#include "sparse_registries.hpp"

namespace tomos {
namespace test {

TEST(SparseRegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::Sparse::get();
    const auto& b = dispatch::Sparse::get();
    EXPECT_EQ(&a, &b);
}

TEST(SparseRegistriesTest, LookupInitializeThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sparse::get().lookup_initialize(
                     MatrixFormat::CSR, Device::OMP, NumericType::FP64,
                     IntType::INT32),
                 std::runtime_error);
}

TEST(SparseRegistriesTest, LookupFreeThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sparse::get().lookup_free(
                     MatrixFormat::CSR, Device::OMP, NumericType::FP64,
                     IntType::INT32),
                 std::runtime_error);
}

TEST(SparseRegistriesTest, LookupSetValuesThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sparse::get().lookup_set_values(Device::OMP,
                                                            NumericType::FP64),
                 std::runtime_error);
}

TEST(SparseRegistriesTest, LookupMvThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Sparse::get().lookup_mv(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, IntType::INT32),
                 std::runtime_error);
}

TEST(SparseRegistriesTest, LookupCopyThrowsOnUnregisteredSourceDevice) {
    EXPECT_THROW(
        dispatch::Sparse::get().lookup_copy(
            MatrixFormat::CSR, MatrixFormat::CSR, Device::OMP, Device::CPU,
            NumericType::FP64, NumericType::FP64, IntType::INT32,
            IntType::INT32),
        std::runtime_error);
}

TEST(SparseRegistriesTest, LookupMvDoesNotThrowOnRegisteredCombination) {
    EXPECT_NO_THROW(dispatch::Sparse::get().lookup_mv(
        Device::CPU, NumericType::FP64, NumericType::FP64, NumericType::FP64,
        IntType::INT32));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

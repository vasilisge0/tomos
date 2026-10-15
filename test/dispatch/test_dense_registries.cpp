#include <gtest/gtest.h>

#include <stdexcept>

#include "dense_registries.hpp"
#include "device.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"

namespace tomos {
namespace test {

// Device::OMP is declared in the Device enum (device.hpp) but never included
// in ActiveDevices (storage_registry_base.hpp), so no registry ever
// registers a kernel for it. Every lookup_* array is still sized to fit it
// (num_registry_devices == Device::Count), so passing it in is a safe,
// in-bounds way to exercise the "nothing registered" throw path -- unlike,
// say, passing Dense a MatrixFormat::CSR, which Dense's registry is NOT
// sized for (num_registry_formats == DenseFormat::Count == 1) and would be
// out of bounds.

TEST(DenseRegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::Dense::get();
    const auto& b = dispatch::Dense::get();
    EXPECT_EQ(&a, &b);
}

TEST(DenseRegistriesTest, LookupInitializeThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Dense::get().lookup_initialize(
                     MatrixFormat::STRIDED, Device::OMP, NumericType::FP64),
                 std::runtime_error);
}

TEST(DenseRegistriesTest, LookupFreeThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Dense::get().lookup_free(
                     MatrixFormat::STRIDED, Device::OMP, NumericType::FP64),
                 std::runtime_error);
}

TEST(DenseRegistriesTest, LookupSetValuesThrowsOnUnregisteredDevice) {
    EXPECT_THROW(
        dispatch::Dense::get().lookup_set_values(Device::OMP, NumericType::FP64),
        std::runtime_error);
}

TEST(DenseRegistriesTest, LookupMvThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Dense::get().lookup_mv(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, NumericType::FP64),
                 std::runtime_error);
}

TEST(DenseRegistriesTest, LookupMmThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Dense::get().lookup_mm(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, /*use_workspace=*/false),
                 std::runtime_error);
}

TEST(DenseRegistriesTest, LookupCopyThrowsOnUnregisteredSourceDevice) {
    EXPECT_THROW(
        dispatch::Dense::get().lookup_copy(
            MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::OMP,
            Device::CPU, NumericType::FP64, NumericType::FP64),
        std::runtime_error);
}

// Native FP64/FP64/FP64 on Device::CPU with compute_type == FP64 (i.e. "no
// override") is always registered -- contrast with the throwing cases above.
TEST(DenseRegistriesTest, LookupMvDoesNotThrowOnRegisteredCombination) {
    EXPECT_NO_THROW(dispatch::Dense::get().lookup_mv(
        Device::CPU, NumericType::FP64, NumericType::FP64, NumericType::FP64,
        NumericType::FP64));
}

TEST(DenseRegistriesTest, LookupSetValuesDoesNotThrowOnRegisteredCombination) {
    EXPECT_NO_THROW(
        dispatch::Dense::get().lookup_set_values(Device::CPU, NumericType::FP64));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

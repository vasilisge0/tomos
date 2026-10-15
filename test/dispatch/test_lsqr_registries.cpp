#include <gtest/gtest.h>

#include <stdexcept>

#include "device.hpp"
#include "lsqr_registries.hpp"
#include "precision.hpp"

namespace tomos {
namespace test {

TEST(LsqrRegistriesTest, GetReturnsSameSingletonInstance) {
    const auto& a = dispatch::Lsqr::get();
    const auto& b = dispatch::Lsqr::get();
    EXPECT_EQ(&a, &b);
}

TEST(LsqrRegistriesTest, LookupSolveDenseThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Lsqr::get().lookup_solve_dense(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, NumericType::FP64),
                 std::runtime_error);
}

TEST(LsqrRegistriesTest, LookupSolveSparseThrowsOnUnregisteredDevice) {
    EXPECT_THROW(dispatch::Lsqr::get().lookup_solve_sparse(
                     Device::OMP, NumericType::FP64, NumericType::FP64,
                     NumericType::FP64, NumericType::FP64),
                 std::runtime_error);
}

// The constructor registers every (mv_type, value_type, refine_type,
// precond_type) combination drawn from FPTypes (double/float/__half/
// __nv_bfloat16) on every active device, so an all-FP64 combination on
// Device::CPU is always registered for both dense and sparse.
TEST(LsqrRegistriesTest, LookupSolveDenseDoesNotThrowOnRegisteredCombination) {
    EXPECT_NO_THROW(dispatch::Lsqr::get().lookup_solve_dense(
        Device::CPU, NumericType::FP64, NumericType::FP64, NumericType::FP64,
        NumericType::FP64));
}

TEST(LsqrRegistriesTest, LookupSolveSparseDoesNotThrowOnRegisteredCombination) {
    EXPECT_NO_THROW(dispatch::Lsqr::get().lookup_solve_sparse(
        Device::CPU, NumericType::FP64, NumericType::FP64, NumericType::FP64,
        NumericType::FP64));
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

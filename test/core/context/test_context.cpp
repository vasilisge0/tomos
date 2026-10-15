#include <gtest/gtest.h>

#include <variant>

#include "context.hpp"

namespace tomos {
namespace test {

TEST(ContextTest, ConstructCpu) {
    Context ctx(Device::CPU);
    EXPECT_EQ(ctx.get_device(), Device::CPU);
}

TEST(ContextTest, CopyConstruct) {
    Context ctx(Device::CPU);
    Context copy = ctx;
    EXPECT_EQ(copy.get_device(), Device::CPU);
}

#if defined(USE_CUDA)

TEST(ContextTest, ConstructCuda) {
    Context ctx(Device::CUDA);
    EXPECT_EQ(ctx.get_device(), Device::CUDA);
}

TEST(ContextTest, CudaHandlesAreUsable) {
    Context ctx(Device::CUDA);
    EXPECT_NE(ctx.cublas_handle(), nullptr);
    EXPECT_NE(ctx.cusparse_handle(), nullptr);
}

// context.hpp documents cublas_handle()/cusparse_handle() as throwing
// std::invalid_argument when the Context isn't on Device::CUDA, but the
// implementation (context.cpp) reaches the handle via
// std::get<CudaContext>(device_context_) with no explicit check, so a
// CPU-backed Context actually throws std::bad_variant_access instead. This
// test pins the ACTUAL behavior; if it starts failing, either the
// implementation was fixed to match the header's documented contract (in
// which case update this test and the header comment stays correct) or the
// two have drifted further.
TEST(ContextTest, CublasHandleOnCpuContextThrowsBadVariantAccess) {
    Context ctx(Device::CPU);
    EXPECT_THROW(ctx.cublas_handle(), std::bad_variant_access);
    EXPECT_THROW(ctx.cusparse_handle(), std::bad_variant_access);
}

// Every CudaContext shares the same process-wide handles (see context.hpp's
// own comment on cuda_cublas_handle()/cuda_cusparse_handle()).
TEST(ContextTest, CudaContextsShareGlobalHandles) {
    Context ctx1(Device::CUDA);
    Context ctx2(Device::CUDA);
    EXPECT_EQ(ctx1.cublas_handle(), ctx2.cublas_handle());
    EXPECT_EQ(ctx1.cusparse_handle(), ctx2.cusparse_handle());
}

#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

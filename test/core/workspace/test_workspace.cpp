#include <gtest/gtest.h>

#include <stdexcept>

#include "context.hpp"
#include "matrix.hpp"
#include "workspace.hpp"

namespace tomos {
namespace test {

class WorkspaceTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

TEST_F(WorkspaceTest, ConstructCpuAllocates) {
    Workspace ws(Device::CPU, 1024);
    EXPECT_EQ(ws.device_, Device::CPU);
    EXPECT_EQ(ws.capacity_, 1024u);
    EXPECT_NE(ws.data_, nullptr);
}

TEST_F(WorkspaceTest, DefaultPrecondStateIsUnset) {
    Workspace ws(Device::CPU, 64);
    EXPECT_EQ(ws.precond_kind_, PrecondKind::None);
    EXPECT_EQ(ws.precond_, nullptr);
}

TEST_F(WorkspaceTest, ZeroCapacityDoesNotCrash) {
    Workspace ws(Device::CPU, 0);
    EXPECT_EQ(ws.capacity_, 0u);
}

TEST_F(WorkspaceTest, AddPrecondDenseSetsKindAndPointer) {
    Workspace ws(Device::CPU, 64);
    ws.precond_type_ = NumericType::FP64;
    Dense P = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 2});
    add_precond(P, ws);
    EXPECT_EQ(ws.precond_kind_, PrecondKind::Dense);
    EXPECT_EQ(ws.precond_, static_cast<void*>(&P));
}

TEST_F(WorkspaceTest, AddPrecondSparseSetsKindAndPointer) {
    Workspace ws(Device::CPU, 64);
    ws.precond_type_ = NumericType::FP64;
    Sparse P = Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32,
                                  {2, 2}, 2);
    add_precond(P, ws);
    EXPECT_EQ(ws.precond_kind_, PrecondKind::Sparse);
    EXPECT_EQ(ws.precond_, static_cast<void*>(&P));
}

TEST_F(WorkspaceTest, AddPrecondThrowsOnNumericTypeMismatch) {
    Workspace ws(Device::CPU, 64);
    ws.precond_type_ = NumericType::FP64;
    Dense P = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {2, 2});
    EXPECT_THROW(add_precond(P, ws), std::invalid_argument);
}

#if defined(USE_CUDA)

TEST_F(WorkspaceTest, ConstructCudaAllocates) {
    Workspace ws(Device::CUDA, 4096);
    EXPECT_EQ(ws.device_, Device::CUDA);
    EXPECT_NE(ws.data_, nullptr);
}

#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

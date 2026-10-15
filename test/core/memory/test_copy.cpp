#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "context.hpp"
#include "matrix.hpp"
#include "status.hpp"

namespace tomos {
namespace test {

#define N_TEST 1000

TEST(CopyTests, TestCopyDense) {
    NumericType ctype[3] = {NumericType::FP64, NumericType::FP32,
                            NumericType::FP16};
    Device device_array[1] = {Device::CPU};

    for (auto p : ctype) {
        for (auto d : device_array) {
            Context ctx(d);
            size_t n = N_TEST;
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {i, i};
                Dense s = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                t.copy_from(s);
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distr(1, n);
            for (size_t i = 0; i < n; i += 10) {
                dim<2> size = {static_cast<size_t>(distr(gen)),
                               static_cast<size_t>(distr(gen))};
                Dense s = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                t.copy_from(s);
            }

            // i x 1 vectors
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {i, 1};
                Dense s = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                t.copy_from(s);
            }

            // 1 x i vectors
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {1, i};
                Dense s = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
                t.copy_from(s);
            }
        }
    }
}

TEST(CopyTests, TestCopyCsr) {
    NumericType ctype[3] = {NumericType::FP64, NumericType::FP32,
                            NumericType::FP16};
    Context ctx(Device::CPU);

    for (auto p : ctype) {
        size_t n = N_TEST;
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {i, i};
            Sparse s = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            t.copy_from(s);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, n);
        for (size_t i = 0; i < n; i += 10) {
            dim<2> size = {static_cast<size_t>(distr(gen)),
                           static_cast<size_t>(distr(gen))};
            Sparse s = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            t.copy_from(s);
        }

        // i x 1 vectors
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {i, 1};
            Sparse s = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            t.copy_from(s);
        }

        // 1 x i vectors
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {1, i};
            Sparse s = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
            t.copy_from(s);
        }
    }
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    std::srand(std::time(0));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

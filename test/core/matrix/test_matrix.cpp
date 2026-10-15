#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "context.hpp"
#include "dimensions.hpp"
#include "matrix.hpp"
#include "status.hpp"

namespace tomos {
namespace test {

#define N_TEST 1000

TEST(MatrixTests, TestInitializeDenseSquareMatrix) {
    NumericType ctype[3] = {NumericType::FP64, NumericType::FP32,
                            NumericType::FP16};
    Device device_array[1] = {Device::CPU};

    for (auto p : ctype) {
        for (auto d : device_array) {
            Context ctx(d);
            size_t n = N_TEST;
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {i, i};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distr(1, n);
            for (size_t i = 0; i < n; i += 10) {
                dim<2> size = {static_cast<size_t>(distr(gen)),
                               static_cast<size_t>(distr(gen))};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }

            // i x 1 vectors
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {i, 1};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }

            for (size_t i = 0; i < n; i += 10) {
                dim<2> size = {static_cast<size_t>(distr(gen)), 1};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }

            // 1 x i vectors
            for (size_t i = 1; i < n; i += 10) {
                dim<2> size = {1, i};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }

            for (size_t i = 0; i < n; i += 10) {
                dim<2> size = {1, static_cast<size_t>(distr(gen))};
                Dense t = Dense::create_strided(ctx, Layout::COL_MAJOR, p,
                                                size);
            }
        }
    }
}

TEST(MatrixTests, TestInitializeCsrMatrix) {
    NumericType ctype[3] = {NumericType::FP64, NumericType::FP32,
                            NumericType::FP16};
    Device device = Device::CPU;
    Context ctx(device);

    for (auto p : ctype) {
        size_t n = N_TEST;
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {i, i};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, n);
        for (size_t i = 0; i < n; i += 10) {
            dim<2> size = {static_cast<size_t>(distr(gen)),
                           static_cast<size_t>(distr(gen))};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
        }

        // i x 1 vectors
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {i, 1};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
        }

        for (size_t i = 0; i < n; i += 10) {
            dim<2> size = {static_cast<size_t>(distr(gen)), 1};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
        }

        // 1 x i vectors
        for (size_t i = 1; i < n; i += 10) {
            dim<2> size = {1, i};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
        }

        for (size_t i = 0; i < n; i += 10) {
            dim<2> size = {1, static_cast<size_t>(distr(gen))};
            Sparse t = Sparse::create_csr(ctx, p, IntType::INT32, size,
                                          size[0]);
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

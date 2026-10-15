#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cblas.h"
#include "matrix_cpu_kernels.hpp"

namespace tomos {
namespace test {

class MvStridedCpuTest : public ::testing::Test {
protected:
    static constexpr size_t M = 64;
    static constexpr size_t N = 32;

    std::vector<double> A_d, x_d, y_ref, y_test;

    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        A_d.resize(M * N);
        x_d.resize(N);
        y_ref.resize(M);
        y_test.resize(M);

        for (auto& v : A_d) v = dist(gen);
        for (auto& v : x_d) v = dist(gen);
        for (auto& v : y_ref) v = dist(gen);
        y_test = y_ref;
    }
};

TEST_F(MvStridedCpuTest, FP64NoTranspose) {
    double alpha = 2.0, beta = 0.5;

    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, alpha, A_d.data(), M,
                x_d.data(), 1, beta, y_ref.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_d.data(), 1, beta, y_test.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, FP64Transpose) {
    double alpha = 1.5, beta = 0.3;

    y_ref.resize(N);
    y_test.resize(N);
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : y_ref) v = dist(gen);
    y_test = y_ref;

    x_d.resize(M);
    for (auto& v : x_d) v = dist(gen);

    cblas_dgemv(CblasColMajor, CblasTrans, M, N, alpha, A_d.data(), M,
                x_d.data(), 1, beta, y_ref.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::TRANSPOSE, alpha, {M, N}, A_d.data(), M, x_d.data(),
        1, beta, y_test.data(), 1);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, FP32NoTranspose) {
    double alpha = 1.0, beta = 0.0;
    float f_alpha = static_cast<float>(alpha);
    float f_beta = static_cast<float>(beta);

    std::vector<float> A_f(A_d.begin(), A_d.end());
    std::vector<float> x_f(x_d.begin(), x_d.end());
    std::vector<float> y_ref_f(M, 0.0f);
    std::vector<float> y_test_f(M, 0.0f);

    cblas_sgemv(CblasColMajor, CblasNoTrans, M, N, f_alpha, A_f.data(), M,
                x_f.data(), 1, f_beta, y_ref_f.data(), 1);

    cpu::mv_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_f.data(), M,
        x_f.data(), 1, beta, y_test_f.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test_f[i], y_ref_f[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, FP32Transpose) {
    double alpha = 2.0, beta = 0.5;
    float f_alpha = static_cast<float>(alpha);
    float f_beta = static_cast<float>(beta);

    std::vector<float> A_f(A_d.begin(), A_d.end());
    std::vector<float> x_f(M);
    std::mt19937 gen(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x_f) v = dist(gen);

    std::vector<float> y_ref_f(N);
    for (auto& v : y_ref_f) v = dist(gen);
    std::vector<float> y_test_f = y_ref_f;

    cblas_sgemv(CblasColMajor, CblasTrans, M, N, f_alpha, A_f.data(), M,
                x_f.data(), 1, f_beta, y_ref_f.data(), 1);

    cpu::mv_strided<float, float, float>(
        TransposeType::TRANSPOSE, alpha, {M, N}, A_f.data(), M, x_f.data(),
        1, beta, y_test_f.data(), 1);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(y_test_f[i], y_ref_f[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, MixedFP64_A_FP32_x_FP64_y) {
    double alpha = 1.0, beta = 0.0;

    std::vector<float> x_f(x_d.begin(), x_d.end());
    std::vector<double> y_manual(M, 0.0);
    std::vector<double> y_kernel(M, 0.0);

    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j) {
            acc += A_d[i + j * M] * static_cast<double>(x_f[j]);
        }
        y_manual[i] = alpha * acc;
    }

    cpu::mv_strided<double, float, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_f.data(), 1, beta, y_kernel.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_kernel[i], y_manual[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, MixedFP32_A_FP64_x_FP64_y) {
    double alpha = 1.5, beta = 0.0;

    std::vector<float> A_f(A_d.begin(), A_d.end());
    std::vector<double> y_manual(M, 0.0);
    std::vector<double> y_kernel(M, 0.0);

    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j) {
            acc += static_cast<double>(A_f[i + j * M]) * x_d[j];
        }
        y_manual[i] = alpha * acc;
    }

    cpu::mv_strided<float, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_f.data(), M,
        x_d.data(), 1, beta, y_kernel.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_kernel[i], y_manual[i], 1e-6)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, MixedFP64_A_FP64_x_FP32_y) {
    double alpha = 1.0, beta = 0.0;

    std::vector<float> y_manual_f(M, 0.0f);
    std::vector<float> y_kernel_f(M, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j) {
            acc += A_d[i + j * M] * x_d[j];
        }
        y_manual_f[i] = static_cast<float>(alpha * acc);
    }

    cpu::mv_strided<double, double, float>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_d.data(), 1, beta, y_kernel_f.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_kernel_f[i], y_manual_f[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, AlphaZeroBetaOne) {
    double alpha = 0.0, beta = 1.0;
    std::vector<double> y_orig = y_test;

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_d.data(), 1, beta, y_test.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_orig[i], 1e-15)
            << "y should be unchanged when alpha=0, beta=1";
    }
}

TEST_F(MvStridedCpuTest, AlphaOneBetaZero) {
    double alpha = 1.0, beta = 0.0;

    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, alpha, A_d.data(), M,
                x_d.data(), 1, beta, y_ref.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_d.data(), 1, beta, y_test.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, NegativeAlpha) {
    double alpha = -3.0, beta = 2.0;

    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, alpha, A_d.data(), M,
                x_d.data(), 1, beta, y_ref.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d.data(), M,
        x_d.data(), 1, beta, y_test.data(), 1);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, SquareMatrix) {
    static constexpr size_t K = 50;
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> A_sq(K * K), x_sq(K), y_ref_sq(K), y_test_sq(K);
    for (auto& v : A_sq) v = dist(gen);
    for (auto& v : x_sq) v = dist(gen);
    for (auto& v : y_ref_sq) v = dist(gen);
    y_test_sq = y_ref_sq;

    double alpha = 1.0, beta = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, K, K, alpha, A_sq.data(), K,
                x_sq.data(), 1, beta, y_ref_sq.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {K, K}, A_sq.data(), K,
        x_sq.data(), 1, beta, y_test_sq.data(), 1);

    for (size_t i = 0; i < K; ++i) {
        EXPECT_NEAR(y_test_sq[i], y_ref_sq[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, TallMatrix) {
    static constexpr size_t ROWS = 500;
    static constexpr size_t COLS = 5;
    std::mt19937 gen(456);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> A_tall(ROWS * COLS), x_tall(COLS);
    std::vector<double> y_ref_tall(ROWS, 0.0), y_test_tall(ROWS, 0.0);
    for (auto& v : A_tall) v = dist(gen);
    for (auto& v : x_tall) v = dist(gen);

    double alpha = 1.0, beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, ROWS, COLS, alpha, A_tall.data(),
                ROWS, x_tall.data(), 1, beta, y_ref_tall.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {ROWS, COLS}, A_tall.data(), ROWS,
        x_tall.data(), 1, beta, y_test_tall.data(), 1);

    for (size_t i = 0; i < ROWS; ++i) {
        EXPECT_NEAR(y_test_tall[i], y_ref_tall[i], 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, WideMatrix) {
    static constexpr size_t ROWS = 5;
    static constexpr size_t COLS = 500;
    std::mt19937 gen(789);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> A_wide(ROWS * COLS), x_wide(COLS);
    std::vector<double> y_ref_wide(ROWS, 0.0), y_test_wide(ROWS, 0.0);
    for (auto& v : A_wide) v = dist(gen);
    for (auto& v : x_wide) v = dist(gen);

    double alpha = 1.0, beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, ROWS, COLS, alpha, A_wide.data(),
                ROWS, x_wide.data(), 1, beta, y_ref_wide.data(), 1);

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {ROWS, COLS}, A_wide.data(), ROWS,
        x_wide.data(), 1, beta, y_test_wide.data(), 1);

    for (size_t i = 0; i < ROWS; ++i) {
        EXPECT_NEAR(y_test_wide[i], y_ref_wide[i], 1e-11)
            << "Mismatch at index " << i;
    }
}

TEST_F(MvStridedCpuTest, SingleElement) {
    std::vector<double> A_1 = {3.0};
    std::vector<double> x_1 = {5.0};
    std::vector<double> y_1 = {7.0};

    cpu::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, 2.0, {1, 1}, A_1.data(), 1, x_1.data(),
        1, 1.0, y_1.data(), 1);

    EXPECT_NEAR(y_1[0], 2.0 * 3.0 * 5.0 + 1.0 * 7.0, 1e-15);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cblas.h"
#include "mm_cpu_kernels.hpp"

namespace tomos {
namespace test {

// Build a random dense matrix, sparsify it to CSR, and return the dense copy.
static void make_sparse_csr(size_t rows, size_t cols, double density,
                             unsigned seed, std::vector<double>& dense_out,
                             std::vector<double>& values_out,
                             std::vector<int>& row_ptrs_out,
                             std::vector<int>& col_idxs_out) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> zero_dist(0.0, 1.0);

    dense_out.assign(rows * cols, 0.0);
    row_ptrs_out.resize(rows + 1, 0);
    values_out.clear();
    col_idxs_out.clear();

    for (size_t i = 0; i < rows; ++i) {
        row_ptrs_out[i] = static_cast<int>(values_out.size());
        for (size_t j = 0; j < cols; ++j) {
            if (zero_dist(gen) < density) {
                double v = val_dist(gen);
                dense_out[i + j * rows] = v;
                values_out.push_back(v);
                col_idxs_out.push_back(static_cast<int>(j));
            }
        }
    }
    row_ptrs_out[rows] = static_cast<int>(values_out.size());
}

// ---- mm_strided tests ----

class MmStridedCpuTest : public ::testing::Test {
protected:
    static constexpr size_t M = 32;
    static constexpr size_t K = 24;
    static constexpr size_t N = 16;

    std::vector<double> A, B, C_ref, C_test;

    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        A.resize(M * K);
        B.resize(K * N);
        C_ref.resize(M * N);
        for (auto& v : A) v = dist(gen);
        for (auto& v : B) v = dist(gen);
        for (auto& v : C_ref) v = dist(gen);
        C_test = C_ref;
    }
};

TEST_F(MmStridedCpuTest, FP64NoTrans) {
    double alpha = 2.0, beta = 0.5;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A.data(), M, {K, N}, B.data(), K, beta, C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-12) << "index " << i;
}

TEST_F(MmStridedCpuTest, FP64TransA) {
    // op(A) is K×M transposed → op(A) is M×K; output is M×N with K inner dim.
    // Reuse A as K×M storage, B as K×N.
    std::vector<double> At(K * M), C_r2(M * N), C_t2(M * N);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : At) v = dist(gen);
    for (auto& v : C_r2) v = dist(gen);
    C_t2 = C_r2;

    double alpha = 1.5, beta = 0.3;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K,
                alpha, At.data(), K, B.data(), K, beta, C_r2.data(), M);
    cpu::mm_strided<double, double, double>(
        TransposeType::TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {K, M}, At.data(), K, {K, N}, B.data(), K, beta, C_t2.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_t2[i], C_r2[i], 1e-12) << "index " << i;
}

TEST_F(MmStridedCpuTest, FP64TransB) {
    // A is M×K, op(B) = B^T so B stored K×N but B^T is N×K → output M×N.
    // Actually: A M×K, B^T N×K → output M×N; B stored as N×K.
    std::vector<double> Bt(N * K), C_r2(M * N), C_t2(M * N);
    std::mt19937 gen(13);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : Bt) v = dist(gen);
    for (auto& v : C_r2) v = dist(gen);
    C_t2 = C_r2;

    double alpha = 0.7, beta = 1.2;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K,
                alpha, A.data(), M, Bt.data(), N, beta, C_r2.data(), M);
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::TRANSPOSE, alpha,
        {M, K}, A.data(), M, {N, K}, Bt.data(), N, beta, C_t2.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_t2[i], C_r2[i], 1e-12) << "index " << i;
}

TEST_F(MmStridedCpuTest, FP64TransAB) {
    // A stored K×M (so A^T is M×K), B stored N×K (so B^T is K×N).
    std::vector<double> At(K * M), Bt(N * K), C_r2(M * N), C_t2(M * N);
    std::mt19937 gen(21);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : At) v = dist(gen);
    for (auto& v : Bt) v = dist(gen);
    for (auto& v : C_r2) v = dist(gen);
    C_t2 = C_r2;

    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, M, N, K,
                alpha, At.data(), K, Bt.data(), N, beta, C_r2.data(), M);
    cpu::mm_strided<double, double, double>(
        TransposeType::TRANSPOSE, TransposeType::TRANSPOSE, alpha,
        {K, M}, At.data(), K, {N, K}, Bt.data(), N, beta, C_t2.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_t2[i], C_r2[i], 1e-12) << "index " << i;
}

TEST_F(MmStridedCpuTest, FP32NoTrans) {
    std::vector<float> Af(A.begin(), A.end());
    std::vector<float> Bf(B.begin(), B.end());
    std::vector<float> C_rf(M * N, 0.0f), C_tf(M * N, 0.0f);

    float alpha = 1.0f, beta = 0.0f;
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                alpha, Af.data(), M, Bf.data(), K, beta, C_rf.data(), M);
    cpu::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, Af.data(), M, {K, N}, Bf.data(), K, beta, C_tf.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_tf[i], C_rf[i], 1e-5f) << "index " << i;
}

TEST_F(MmStridedCpuTest, MixedFP64A_FP32B_FP64C) {
    std::vector<float> Bf(B.begin(), B.end());
    std::vector<double> C_manual(M * N, 0.0), C_kernel(M * N, 0.0);

    double alpha = 1.0, beta = 0.0;
    for (size_t j = 0; j < N; ++j)
        for (size_t l = 0; l < K; ++l)
            for (size_t i = 0; i < M; ++i)
                C_manual[i + j * M] += A[i + l * M] * static_cast<double>(Bf[l + j * K]);

    cpu::mm_strided<double, float, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A.data(), M, {K, N}, Bf.data(), K, beta, C_kernel.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_kernel[i], C_manual[i], 1e-10) << "index " << i;
}

TEST_F(MmStridedCpuTest, AlphaZeroBetaOne) {
    std::vector<double> C_orig = C_test;
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 0.0,
        {M, K}, A.data(), M, {K, N}, B.data(), K, 1.0, C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_orig[i], 1e-15) << "C should be unchanged";
}

TEST_F(MmStridedCpuTest, AlphaOneBetaZero) {
    std::fill(C_ref.begin(), C_ref.end(), 0.0);
    std::fill(C_test.begin(), C_test.end(), 0.0);

    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                alpha, A.data(), M, B.data(), K, beta, C_ref.data(), M);
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A.data(), M, {K, N}, B.data(), K, beta, C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-12) << "index " << i;
}

TEST_F(MmStridedCpuTest, SquareMatrix) {
    constexpr size_t S = 48;
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> As(S * S), Bs(S * S), Cr(S * S, 0.0), Ct(S * S, 0.0);
    for (auto& v : As) v = dist(gen);
    for (auto& v : Bs) v = dist(gen);

    double alpha = 2.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, S, S, S,
                alpha, As.data(), S, Bs.data(), S, beta, Cr.data(), S);
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {S, S}, As.data(), S, {S, S}, Bs.data(), S, beta, Ct.data(), S);
    for (size_t i = 0; i < S * S; ++i)
        EXPECT_NEAR(Ct[i], Cr[i], 1e-10) << "index " << i;
}

TEST_F(MmStridedCpuTest, SingleElement) {
    double A1 = 3.0, B1 = 5.0, C1 = 7.0;
    cpu::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 2.0,
        {1, 1}, &A1, 1, {1, 1}, &B1, 1, 1.0, &C1, 1);
    EXPECT_NEAR(C1, 2.0 * 3.0 * 5.0 + 1.0 * 7.0, 1e-15);
}

// ---- mm_csr tests ----

class MmCsrCpuTest : public ::testing::Test {
protected:
    static constexpr size_t M = 20;
    static constexpr size_t K = 15;
    static constexpr size_t N = 10;

    std::vector<double> A_dense, B, C_ref, C_test;
    std::vector<double> A_vals;
    std::vector<int> row_ptrs, col_idxs;

    void SetUp() override {
        std::mt19937 gen(77);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        make_sparse_csr(M, K, 0.4, 77, A_dense, A_vals, row_ptrs, col_idxs);

        B.resize(K * N);
        C_ref.resize(M * N);
        for (auto& v : B) v = dist(gen);
        for (auto& v : C_ref) v = dist(gen);
        C_test = C_ref;
    }
};

TEST_F(MmCsrCpuTest, FP64NoTrans) {
    double alpha = 1.5, beta = 0.5;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                alpha, A_dense.data(), M, B.data(), K, beta, C_ref.data(), M);
    cpu::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_vals.size(), A_vals.data(), row_ptrs.data(), col_idxs.data(),
        {K, N}, B.data(), K, beta, C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-11) << "index " << i;
}

TEST_F(MmCsrCpuTest, FP64TransA) {
    // A stored M×K in CSR; A^T is K×M. Output is K×N.
    double alpha = 1.0, beta = 0.0;
    std::vector<double> Cr(K * N, 0.0), Ct(K * N, 0.0);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, N, M,
                alpha, A_dense.data(), M, B.data(), M, beta, Cr.data(), K);

    // For transA: m_out=K, k_inner=M, need B to be M×N.
    // Reuse B resized to M×N.
    std::vector<double> Bmn(M * N);
    std::mt19937 gen(55);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : Bmn) v = dist(gen);

    std::fill(Cr.begin(), Cr.end(), 0.0);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, N, M,
                alpha, A_dense.data(), M, Bmn.data(), M, beta, Cr.data(), K);
    cpu::mm_csr<double, double, double, int>(
        TransposeType::TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_vals.size(), A_vals.data(), row_ptrs.data(), col_idxs.data(),
        {M, N}, Bmn.data(), M, beta, Ct.data(), K);
    for (size_t i = 0; i < K * N; ++i)
        EXPECT_NEAR(Ct[i], Cr[i], 1e-11) << "index " << i;
}

TEST_F(MmCsrCpuTest, FP64NoTransTransB) {
    // B stored N×K so B^T is K×N; output is M×N.
    std::vector<double> Bt(N * K);
    std::mt19937 gen(33);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : Bt) v = dist(gen);

    double alpha = 2.0, beta = 0.5;
    std::vector<double> Cr = C_ref, Ct = C_ref;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K,
                alpha, A_dense.data(), M, Bt.data(), N, beta, Cr.data(), M);
    cpu::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::TRANSPOSE, alpha,
        {M, K}, A_vals.size(), A_vals.data(), row_ptrs.data(), col_idxs.data(),
        {N, K}, Bt.data(), N, beta, Ct.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(Ct[i], Cr[i], 1e-11) << "index " << i;
}

TEST_F(MmCsrCpuTest, DiagonalMatrix) {
    // 4×4 diagonal matrix with diagonal = {1,2,3,4}, B identity → C = diag.
    constexpr size_t S = 4;
    std::vector<double> diag_vals = {1.0, 2.0, 3.0, 4.0};
    std::vector<int> rp = {0, 1, 2, 3, 4};
    std::vector<int> ci = {0, 1, 2, 3};
    std::vector<double> Bmat(S * S, 0.0);
    for (size_t i = 0; i < S; ++i) Bmat[i + i * S] = 1.0;
    std::vector<double> C4(S * S, 0.0);

    cpu::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {S, S}, S, diag_vals.data(), rp.data(), ci.data(),
        {S, S}, Bmat.data(), S, 0.0, C4.data(), S);

    for (size_t j = 0; j < S; ++j)
        for (size_t i = 0; i < S; ++i)
            EXPECT_NEAR(C4[i + j * S], (i == j) ? diag_vals[i] : 0.0, 1e-15);
}

TEST_F(MmCsrCpuTest, AlphaZeroBetaOne) {
    std::vector<double> C_orig = C_test;
    cpu::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 0.0,
        {M, K}, A_vals.size(), A_vals.data(), row_ptrs.data(), col_idxs.data(),
        {K, N}, B.data(), K, 1.0, C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_orig[i], 1e-15) << "C should be unchanged";
}

// ---- mm_csr_csr tests ----

class MmCsrCsrCpuTest : public ::testing::Test {
protected:
    static constexpr size_t M = 20;
    static constexpr size_t K = 15;
    static constexpr size_t N = 10;

    std::vector<double> A_dense, B_dense, C_ref, C_test;
    std::vector<double> A_vals, B_vals;
    std::vector<int> A_row_ptrs, A_col_idxs, B_row_ptrs, B_col_idxs;

    void SetUp() override {
        make_sparse_csr(M, K, 0.4, 77, A_dense, A_vals, A_row_ptrs, A_col_idxs);
        make_sparse_csr(K, N, 0.4, 33, B_dense, B_vals, B_row_ptrs, B_col_idxs);

        std::mt19937 gen(11);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        C_ref.resize(M * N);
        for (auto& v : C_ref) v = dist(gen);
        C_test = C_ref;
    }
};

TEST_F(MmCsrCsrCpuTest, FP64) {
    double alpha = 1.5, beta = 0.5;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                A_dense.data(), M, B_dense.data(), K, beta, C_ref.data(), M);
    cpu::mm_csr_csr<double, double, double, int>(
        alpha, {M, K}, A_row_ptrs.data(), A_col_idxs.data(), A_vals.data(),
        {K, N}, B_row_ptrs.data(), B_col_idxs.data(), B_vals.data(), beta,
        C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-11) << "index " << i;
}

TEST_F(MmCsrCsrCpuTest, AlphaZeroBetaOne) {
    std::vector<double> C_orig = C_test;
    cpu::mm_csr_csr<double, double, double, int>(
        0.0, {M, K}, A_row_ptrs.data(), A_col_idxs.data(), A_vals.data(),
        {K, N}, B_row_ptrs.data(), B_col_idxs.data(), B_vals.data(), 1.0,
        C_test.data(), M);
    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_orig[i], 1e-15) << "C should be unchanged";
}

TEST_F(MmCsrCsrCpuTest, DiagonalTimesDiagonal) {
    // diag(1,2,3,4) * diag(5,6,7,8) = diag(5,12,21,32).
    constexpr size_t S = 4;
    std::vector<double> a_vals = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b_vals = {5.0, 6.0, 7.0, 8.0};
    std::vector<int> rp = {0, 1, 2, 3, 4};
    std::vector<int> ci = {0, 1, 2, 3};
    std::vector<double> C4(S * S, 0.0);

    cpu::mm_csr_csr<double, double, double, int>(
        1.0, {S, S}, rp.data(), ci.data(), a_vals.data(), {S, S}, rp.data(),
        ci.data(), b_vals.data(), 0.0, C4.data(), S);

    std::vector<double> expected = {5.0, 12.0, 21.0, 32.0};
    for (size_t j = 0; j < S; ++j)
        for (size_t i = 0; i < S; ++i)
            EXPECT_NEAR(C4[i + j * S], (i == j) ? expected[i] : 0.0, 1e-15);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
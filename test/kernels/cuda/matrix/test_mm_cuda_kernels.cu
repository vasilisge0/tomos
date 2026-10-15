#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "mm_cuda_kernels.cuh"

namespace tomos {
namespace test {

// ---- mm_strided tests ----

class MmStridedCudaTest : public ::testing::Test {
protected:
    static constexpr size_t M = 32;
    static constexpr size_t K = 24;
    static constexpr size_t N = 16;

    cublasHandle_t handle;
    std::vector<double> A_h, B_h, C_h;

    void SetUp() override {
        cublasCreate(&handle);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        A_h.resize(M * K);
        B_h.resize(K * N);
        C_h.resize(M * N);
        for (auto& v : A_h) v = dist(gen);
        for (auto& v : B_h) v = dist(gen);
        for (auto& v : C_h) v = dist(gen);
    }

    void TearDown() override { cublasDestroy(handle); }
};

TEST_F(MmStridedCudaTest, FP64NoTrans) {
    double alpha = 2.0, beta = 0.5;

    double *A_d, *B_d, *C_ref_d, *C_test_d;
    cudaMalloc(&A_d, M * K * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(double));
    cudaMalloc(&C_ref_d, M * N * sizeof(double));
    cudaMalloc(&C_test_d, M * N * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_ref_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_test_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, A_d, M, B_d, K, &beta, C_ref_d, M);

    cuda::experimental::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_d, M, {K, N}, B_d, K, beta, C_test_d, M, handle);

    std::vector<double> C_ref(M * N), C_test(M * N);
    cudaMemcpy(C_ref.data(), C_ref_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_test.data(), C_test_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-12) << "index " << i;

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_ref_d); cudaFree(C_test_d);
}

TEST_F(MmStridedCudaTest, FP64TransA) {
    // A stored K×M, op(A)=A^T is M×K.
    std::vector<double> At_h(K * M);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : At_h) v = dist(gen);

    double alpha = 1.5, beta = 0.3;
    std::vector<double> C2_h = C_h;

    double *At_d, *B_d, *C_ref_d, *C_test_d;
    cudaMalloc(&At_d, K * M * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(double));
    cudaMalloc(&C_ref_d, M * N * sizeof(double));
    cudaMalloc(&C_test_d, M * N * sizeof(double));

    cudaMemcpy(At_d, At_h.data(), K * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_ref_d, C2_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_test_d, C2_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                &alpha, At_d, K, B_d, K, &beta, C_ref_d, M);
    cuda::experimental::mm_strided<double, double, double>(
        TransposeType::TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {K, M}, At_d, K, {K, N}, B_d, K, beta, C_test_d, M, handle);

    std::vector<double> C_ref(M * N), C_test(M * N);
    cudaMemcpy(C_ref.data(), C_ref_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_test.data(), C_test_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-12) << "index " << i;

    cudaFree(At_d); cudaFree(B_d); cudaFree(C_ref_d); cudaFree(C_test_d);
}

TEST_F(MmStridedCudaTest, FP32NoTrans) {
    std::vector<float> Af(A_h.begin(), A_h.end());
    std::vector<float> Bf(B_h.begin(), B_h.end());
    std::vector<float> Cf(C_h.begin(), C_h.end());

    float alpha = 1.0f, beta = 0.0f;

    float *A_d, *B_d, *C_ref_d, *C_test_d;
    cudaMalloc(&A_d, M * K * sizeof(float));
    cudaMalloc(&B_d, K * N * sizeof(float));
    cudaMalloc(&C_ref_d, M * N * sizeof(float));
    cudaMalloc(&C_test_d, M * N * sizeof(float));

    cudaMemcpy(A_d, Af.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bf.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_ref_d, 0, M * N * sizeof(float));
    cudaMemset(C_test_d, 0, M * N * sizeof(float));

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, A_d, M, B_d, K, &beta, C_ref_d, M);
    cuda::experimental::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_d, M, {K, N}, B_d, K, beta, C_test_d, M, handle);

    std::vector<float> C_ref(M * N), C_test(M * N);
    cudaMemcpy(C_ref.data(), C_ref_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_test.data(), C_test_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-5f) << "index " << i;

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_ref_d); cudaFree(C_test_d);
}

TEST_F(MmStridedCudaTest, MixedFP64A_FP32B_FP64C) {
    std::vector<float> Bf(B_h.begin(), B_h.end());
    double alpha = 1.0, beta = 0.0;

    double *A_d;
    float *B_d;
    double *C_d;
    cudaMalloc(&A_d, M * K * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(float));
    cudaMalloc(&C_d, M * N * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bf.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, M * N * sizeof(double));

    cuda::experimental::mm_strided<double, float, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_d, M, {K, N}, B_d, K, beta, C_d, M, handle);

    std::vector<double> C_result(M * N);
    cudaMemcpy(C_result.data(), C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < K; ++l)
                acc += A_h[i + l * M] * static_cast<double>(Bf[l + j * K]);
            EXPECT_NEAR(C_result[i + j * M], acc, 1e-10) << "i=" << i << " j=" << j;
        }
    }

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

TEST_F(MmStridedCudaTest, AlphaZeroBetaOne) {
    double alpha = 0.0, beta = 1.0;

    double *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, M * K * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(double));
    cudaMalloc(&C_d, M * N * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

    cuda::experimental::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, A_d, M, {K, N}, B_d, K, beta, C_d, M, handle);

    std::vector<double> C_result(M * N);
    cudaMemcpy(C_result.data(), C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_result[i], C_h[i], 1e-15) << "C should be unchanged";

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

TEST_F(MmStridedCudaTest, SingleElement) {
    double A1 = 3.0, B1 = 5.0, C1 = 7.0;
    double *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(double));
    cudaMalloc(&B_d, sizeof(double));
    cudaMalloc(&C_d, sizeof(double));
    cudaMemcpy(A_d, &A1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, &B1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, &C1, sizeof(double), cudaMemcpyHostToDevice);

    cuda::experimental::mm_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 2.0,
        {1, 1}, A_d, 1, {1, 1}, B_d, 1, 1.0, C_d, 1, handle);

    double result;
    cudaMemcpy(&result, C_d, sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(result, 2.0 * 3.0 * 5.0 + 1.0 * 7.0, 1e-15);

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

// ---- gemm_generic_kernel direct tests ----

class GemmGenericKernelTest : public ::testing::Test {
protected:
    static constexpr size_t M = 32;
    static constexpr size_t K = 24;
    static constexpr size_t N = 16;

    std::vector<double> A_h, B_h, C_h;

    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        A_h.resize(M * K);
        B_h.resize(K * N);
        C_h.resize(M * N);
        for (auto& v : A_h) v = dist(gen);
        for (auto& v : B_h) v = dist(gen);
        for (auto& v : C_h) v = dist(gen);
    }
};

TEST_F(GemmGenericKernelTest, DoublePrecision) {
    double alpha = 2.0, beta = 0.5;

    double *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, M * K * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(double));
    cudaMalloc(&C_d, M * N * sizeof(double));
    cudaMemcpy(A_d, A_h.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    cuda::experimental::gemm_generic_kernel<double, double, double><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        alpha, M, K, N, A_d, M, B_d, K, beta, C_d, M);
    cudaDeviceSynchronize();

    std::vector<double> C_result(M * N);
    cudaMemcpy(C_result.data(), C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < K; ++l)
                acc += A_h[i + l * M] * B_h[l + j * K];
            double expected = alpha * acc + beta * C_h[i + j * M];
            EXPECT_NEAR(C_result[i + j * M], expected, 1e-10) << "i=" << i << " j=" << j;
        }
    }

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

TEST_F(GemmGenericKernelTest, SinglePrecision) {
    std::vector<float> Af(A_h.begin(), A_h.end());
    std::vector<float> Bf(B_h.begin(), B_h.end());
    std::vector<float> Cf(C_h.begin(), C_h.end());
    double alpha = 1.5, beta = 0.0;

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, M * K * sizeof(float));
    cudaMalloc(&B_d, K * N * sizeof(float));
    cudaMalloc(&C_d, M * N * sizeof(float));
    cudaMemcpy(A_d, Af.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bf.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, M * N * sizeof(float));

    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    cuda::experimental::gemm_generic_kernel<float, float, float><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        alpha, M, K, N, A_d, M, B_d, K, beta, C_d, M);
    cudaDeviceSynchronize();

    std::vector<float> C_result(M * N);
    cudaMemcpy(C_result.data(), C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Reference: kernel accumulates in double then casts to float.
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < K; ++l)
                acc += static_cast<double>(Af[i + l * M]) *
                       static_cast<double>(Bf[l + j * K]);
            float expected = static_cast<float>(alpha * acc);
            EXPECT_NEAR(C_result[i + j * M], expected, 1e-5f) << "i=" << i << " j=" << j;
        }
    }

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

TEST_F(GemmGenericKernelTest, HalfPrecision) {
    std::vector<__half> Ah(M * K), Bh(K * N);
    for (size_t i = 0; i < M * K; ++i) Ah[i] = __float2half(static_cast<float>(A_h[i]));
    for (size_t i = 0; i < K * N; ++i) Bh[i] = __float2half(static_cast<float>(B_h[i]));

    double alpha = 1.0, beta = 0.0;

    __half *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, M * K * sizeof(__half));
    cudaMalloc(&B_d, K * N * sizeof(__half));
    cudaMalloc(&C_d, M * N * sizeof(__half));
    cudaMemcpy(A_d, Ah.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bh.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, M * N * sizeof(__half));

    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    cuda::experimental::gemm_generic_kernel<__half, __half, __half><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        alpha, M, K, N, A_d, M, B_d, K, beta, C_d, M);
    cudaDeviceSynchronize();

    std::vector<__half> C_result(M * N);
    cudaMemcpy(C_result.data(), C_d, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < K; ++l)
                acc += static_cast<double>(__half2float(Ah[i + l * M])) *
                       static_cast<double>(__half2float(Bh[l + j * K]));
            float expected = static_cast<float>(acc);
            EXPECT_NEAR(__half2float(C_result[i + j * M]), expected, 0.1f)
                << "i=" << i << " j=" << j;
        }
    }

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

// ---- mm_csr CUDA tests ----

class MmCsrCudaTest : public ::testing::Test {
protected:
    static constexpr size_t M = 16;
    static constexpr size_t K = 12;
    static constexpr size_t N = 8;

    cusparseHandle_t sp_handle;
    cublasHandle_t bl_handle;

    std::vector<double> A_dense_h, B_h, C_h;
    std::vector<double> A_vals_h;
    std::vector<int> row_ptrs_h, col_idxs_h;

    void SetUp() override {
        cusparseCreate(&sp_handle);
        cublasCreate(&bl_handle);

        std::mt19937 gen(55);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::uniform_real_distribution<double> zero(0.0, 1.0);

        A_dense_h.assign(M * K, 0.0);
        row_ptrs_h.resize(M + 1, 0);
        B_h.resize(K * N);
        C_h.resize(M * N);
        for (auto& v : B_h) v = dist(gen);
        for (auto& v : C_h) v = dist(gen);

        for (size_t i = 0; i < M; ++i) {
            row_ptrs_h[i] = static_cast<int>(A_vals_h.size());
            for (size_t j = 0; j < K; ++j) {
                if (zero(gen) < 0.4) {
                    double v = dist(gen);
                    A_dense_h[i + j * M] = v;
                    A_vals_h.push_back(v);
                    col_idxs_h.push_back(static_cast<int>(j));
                }
            }
        }
        row_ptrs_h[M] = static_cast<int>(A_vals_h.size());
    }

    void TearDown() override {
        cusparseDestroy(sp_handle);
        cublasDestroy(bl_handle);
    }
};

TEST_F(MmCsrCudaTest, FP64NoTrans) {
    double alpha = 1.5, beta = 0.5;

    // Reference: cublas dense gemm on A_dense.
    double *Ad_d, *B_d, *C_ref_d, *C_test_d;
    cudaMalloc(&Ad_d, M * K * sizeof(double));
    cudaMalloc(&B_d, K * N * sizeof(double));
    cudaMalloc(&C_ref_d, M * N * sizeof(double));
    cudaMalloc(&C_test_d, M * N * sizeof(double));

    cudaMemcpy(Ad_d, A_dense_h.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_ref_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_test_d, C_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemm(bl_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, Ad_d, M, B_d, K, &beta, C_ref_d, M);

    // CSR arrays on device.
    size_t nnz = A_vals_h.size();
    double* vals_d;
    int *rp_d, *ci_d;
    cudaMalloc(&vals_d, nnz * sizeof(double));
    cudaMalloc(&rp_d, (M + 1) * sizeof(int));
    cudaMalloc(&ci_d, nnz * sizeof(int));
    cudaMemcpy(vals_d, A_vals_h.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rp_d, row_ptrs_h.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ci_d, col_idxs_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);

    cuda::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        {M, K}, nnz, vals_d, rp_d, ci_d,
        {K, N}, B_d, K, beta, C_test_d, M, sp_handle);

    std::vector<double> C_ref(M * N), C_test(M * N);
    cudaMemcpy(C_ref.data(), C_ref_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_test.data(), C_test_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-10) << "index " << i;

    cudaFree(Ad_d); cudaFree(B_d); cudaFree(C_ref_d); cudaFree(C_test_d);
    cudaFree(vals_d); cudaFree(rp_d); cudaFree(ci_d);
}

TEST_F(MmCsrCudaTest, DiagonalMatrix) {
    constexpr size_t S = 8;
    std::vector<double> diag_vals = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> rp(S + 1);
    std::vector<int> ci(S);
    for (size_t i = 0; i <= S; ++i) rp[i] = static_cast<int>(i);
    for (size_t i = 0; i < S; ++i) ci[i] = static_cast<int>(i);

    // B = identity → C = diag
    std::vector<double> Bmat(S * S, 0.0);
    for (size_t i = 0; i < S; ++i) Bmat[i + i * S] = 1.0;
    std::vector<double> C_init(S * S, 0.0);

    double *vals_d, *B_d, *C_d;
    int *rp_d, *ci_d;
    cudaMalloc(&vals_d, S * sizeof(double));
    cudaMalloc(&rp_d, (S + 1) * sizeof(int));
    cudaMalloc(&ci_d, S * sizeof(int));
    cudaMalloc(&B_d, S * S * sizeof(double));
    cudaMalloc(&C_d, S * S * sizeof(double));

    cudaMemcpy(vals_d, diag_vals.data(), S * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rp_d, rp.data(), (S + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ci_d, ci.data(), S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, Bmat.data(), S * S * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, S * S * sizeof(double));

    cusparseHandle_t sp;
    cusparseCreate(&sp);
    cuda::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {S, S}, S, vals_d, rp_d, ci_d,
        {S, S}, B_d, S, 0.0, C_d, S, sp);
    cusparseDestroy(sp);

    std::vector<double> C_result(S * S);
    cudaMemcpy(C_result.data(), C_d, S * S * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < S; ++j)
        for (size_t i = 0; i < S; ++i)
            EXPECT_NEAR(C_result[i + j * S], (i == j) ? diag_vals[i] : 0.0, 1e-15);

    cudaFree(vals_d); cudaFree(rp_d); cudaFree(ci_d);
    cudaFree(B_d); cudaFree(C_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
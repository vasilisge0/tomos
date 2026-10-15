#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "malloc.hpp"
#include "mv_cuda_kernels.cuh"

namespace tomos {
namespace test {

class MvStridedCudaTest : public ::testing::Test {
protected:
    static constexpr size_t M = 64;
    static constexpr size_t N = 32;

    cublasHandle_t handle;
    std::vector<double> A_h, x_h, y_h;

    void SetUp() override {
        cublasCreate(&handle);

        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        A_h.resize(M * N);
        x_h.resize(N);
        y_h.resize(M);

        for (auto& v : A_h) v = dist(gen);
        for (auto& v : x_h) v = dist(gen);
        for (auto& v : y_h) v = dist(gen);
    }

    void TearDown() override { cublasDestroy(handle); }
};

TEST_F(MvStridedCudaTest, FP64NoTranspose) {
    double alpha = 2.0, beta = 0.5;

    double *A_d, *x_d, *y_ref_d, *y_test_d;
    cudaMalloc(&A_d, M * N * sizeof(double));
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_ref_d, M * sizeof(double));
    cudaMalloc(&y_test_d, M * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_ref_d, y_h.data(), M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_test_d, y_h.data(), M * sizeof(double),
               cudaMemcpyHostToDevice);

    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, A_d, M, x_d, 1, &beta,
                y_ref_d, 1);

    cuda::experimental::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d, M, x_d, 1, beta,
        y_test_d, 1, handle);

    std::vector<double> y_ref(M), y_test(M);
    cudaMemcpy(y_ref.data(), y_ref_d, M * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(y_test.data(), y_test_d, M * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_ref_d);
    cudaFree(y_test_d);
}

TEST_F(MvStridedCudaTest, FP64Transpose) {
    double alpha = 1.5, beta = 0.3;

    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x_trans(M), y_trans(N);
    for (auto& v : x_trans) v = dist(gen);
    for (auto& v : y_trans) v = dist(gen);

    double *A_d, *x_d, *y_ref_d, *y_test_d;
    cudaMalloc(&A_d, M * N * sizeof(double));
    cudaMalloc(&x_d, M * sizeof(double));
    cudaMalloc(&y_ref_d, N * sizeof(double));
    cudaMalloc(&y_test_d, N * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_trans.data(), M * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(y_ref_d, y_trans.data(), N * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(y_test_d, y_trans.data(), N * sizeof(double),
               cudaMemcpyHostToDevice);

    cublasDgemv(handle, CUBLAS_OP_T, M, N, &alpha, A_d, M, x_d, 1, &beta,
                y_ref_d, 1);

    cuda::experimental::mv_strided<double, double, double>(
        TransposeType::TRANSPOSE, alpha, {M, N}, A_d, M, x_d, 1, beta,
        y_test_d, 1, handle);

    std::vector<double> y_ref(N), y_test(N);
    cudaMemcpy(y_ref.data(), y_ref_d, N * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(y_test.data(), y_test_d, N * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-12)
            << "Mismatch at index " << i;
    }

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_ref_d);
    cudaFree(y_test_d);
}

TEST_F(MvStridedCudaTest, FP32NoTranspose) {
    double alpha = 1.0, beta = 0.0;
    float f_alpha = static_cast<float>(alpha);
    float f_beta = static_cast<float>(beta);

    std::vector<float> A_f(A_h.begin(), A_h.end());
    std::vector<float> x_f(x_h.begin(), x_h.end());

    float *A_d, *x_d, *y_ref_d, *y_test_d;
    cudaMalloc(&A_d, M * N * sizeof(float));
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_ref_d, M * sizeof(float));
    cudaMalloc(&y_test_d, M * sizeof(float));

    cudaMemcpy(A_d, A_f.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_f.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(y_ref_d, 0, M * sizeof(float));
    cudaMemset(y_test_d, 0, M * sizeof(float));

    cublasSgemv(handle, CUBLAS_OP_N, M, N, &f_alpha, A_d, M, x_d, 1, &f_beta,
                y_ref_d, 1);

    cuda::experimental::mv_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d, M, x_d, 1, beta,
        y_test_d, 1, handle);

    std::vector<float> y_ref(M), y_test(M);
    cudaMemcpy(y_ref.data(), y_ref_d, M * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(y_test.data(), y_test_d, M * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_test[i], y_ref[i], 1e-5f)
            << "Mismatch at index " << i;
    }

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_ref_d);
    cudaFree(y_test_d);
}

TEST_F(MvStridedCudaTest, MixedFP64_A_FP32_x_FP64_y) {
    double alpha = 1.0, beta = 0.0;

    std::vector<float> x_f(x_h.begin(), x_h.end());

    double* A_d;
    float* x_d;
    double *y_ref_d, *y_test_d;
    cudaMalloc(&A_d, M * N * sizeof(double));
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_ref_d, M * sizeof(double));
    cudaMalloc(&y_test_d, M * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_f.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(y_test_d, 0, M * sizeof(double));

    cuda::experimental::mv_strided<double, float, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d, M, x_d, 1, beta,
        y_test_d, 1, handle);

    std::vector<double> y_test(M);
    cudaMemcpy(y_test.data(), y_test_d, M * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j) {
            acc += A_h[i + j * M] * static_cast<double>(x_f[j]);
        }
        EXPECT_NEAR(y_test[i], alpha * acc, 1e-12)
            << "Mismatch at index " << i;
    }

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_ref_d);
    cudaFree(y_test_d);
}

TEST_F(MvStridedCudaTest, AlphaZeroBetaOne) {
    double alpha = 0.0, beta = 1.0;

    double *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, M * N * sizeof(double));
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_d, M * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h.data(), M * sizeof(double), cudaMemcpyHostToDevice);

    cuda::experimental::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, alpha, {M, N}, A_d, M, x_d, 1, beta,
        y_d, 1, handle);

    std::vector<double> y_result(M);
    cudaMemcpy(y_result.data(), y_d, M * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) {
        EXPECT_NEAR(y_result[i], y_h[i], 1e-15)
            << "y should be unchanged when alpha=0, beta=1";
    }

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_d);
}

TEST_F(MvStridedCudaTest, SingleElement) {
    double A_val = 3.0, x_val = 5.0, y_val = 7.0;

    double *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, sizeof(double));
    cudaMalloc(&x_d, sizeof(double));
    cudaMalloc(&y_d, sizeof(double));

    cudaMemcpy(A_d, &A_val, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, &x_val, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, &y_val, sizeof(double), cudaMemcpyHostToDevice);

    cuda::experimental::mv_strided<double, double, double>(
        TransposeType::NO_TRANSPOSE, 2.0, {1, 1}, A_d, 1, x_d, 1, 1.0, y_d,
        1, handle);

    double y_result;
    cudaMemcpy(&y_result, y_d, sizeof(double), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(y_result, 2.0 * 3.0 * 5.0 + 1.0 * 7.0, 1e-15);

    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_d);
}

// ---- Direct gemv_generic_kernel tests ----

class GemvGenericKernelTest : public ::testing::Test {
protected:
    static constexpr size_t M = 64;
    static constexpr size_t N = 32;

    std::vector<double> A_h, x_h, y_h;

    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        A_h.resize(M * N);
        x_h.resize(N);
        y_h.resize(M);
        for (auto& v : A_h) v = dist(gen);
        for (auto& v : x_h) v = dist(gen);
        for (auto& v : y_h) v = dist(gen);
    }
};

TEST_F(GemvGenericKernelTest, DoublePrecision) {
    double alpha = 2.0, beta = 0.5;

    double *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, M * N * sizeof(double));
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_d, M * sizeof(double));

    cudaMemcpy(A_d, A_h.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h.data(), M * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256, blocks = ((int)M + threads - 1) / threads;
    cuda::experimental::gemv_generic_kernel<double, double, double><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, alpha, M, N, A_d, M, x_d, beta, y_d);
    cudaDeviceSynchronize();

    std::vector<double> y_result(M);
    cudaMemcpy(y_result.data(), y_d, M * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j)
            acc += A_h[i + j * M] * x_h[j];
        EXPECT_NEAR(y_result[i], alpha * acc + beta * y_h[i], 1e-12)
            << "index " << i;
    }

    cudaFree(A_d); cudaFree(x_d); cudaFree(y_d);
}

TEST_F(GemvGenericKernelTest, SinglePrecision) {
    double alpha = 2.0, beta = 0.5;

    std::vector<float> A_f(A_h.begin(), A_h.end());
    std::vector<float> x_f(x_h.begin(), x_h.end());
    std::vector<float> y_f(y_h.begin(), y_h.end());

    float *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, M * N * sizeof(float));
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_d, M * sizeof(float));

    cudaMemcpy(A_d, A_f.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_f.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_f.data(), M * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = ((int)M + threads - 1) / threads;
    cuda::experimental::gemv_generic_kernel<float, float, float><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, alpha, M, N, A_d, M, x_d, beta, y_d);
    cudaDeviceSynchronize();

    std::vector<float> y_result(M);
    cudaMemcpy(y_result.data(), y_d, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Kernel accumulates in double then casts to float; replicate on CPU.
    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j)
            acc += static_cast<double>(A_f[i + j * M]) *
                   static_cast<double>(x_f[j]);
        float expected = static_cast<float>(
            alpha * acc + beta * static_cast<double>(y_f[i]));
        EXPECT_NEAR(y_result[i], expected, 1e-6f) << "index " << i;
    }

    cudaFree(A_d); cudaFree(x_d); cudaFree(y_d);
}

TEST_F(GemvGenericKernelTest, HalfPrecision) {
    double alpha = 1.0, beta = 0.0;

    std::vector<__half> A_half(M * N), x_half(N);
    for (size_t k = 0; k < M * N; ++k)
        A_half[k] = __float2half(static_cast<float>(A_h[k]));
    for (size_t k = 0; k < N; ++k)
        x_half[k] = __float2half(static_cast<float>(x_h[k]));

    __half *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, M * N * sizeof(__half));
    cudaMalloc(&x_d, N * sizeof(__half));
    cudaMalloc(&y_d, M * sizeof(__half));

    cudaMemcpy(A_d, A_half.data(), M * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_half.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(y_d, 0, M * sizeof(__half));

    int threads = 256, blocks = ((int)M + threads - 1) / threads;
    cuda::experimental::gemv_generic_kernel<__half, __half, __half><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, alpha, M, N, A_d, M, x_d, beta, y_d);
    cudaDeviceSynchronize();

    std::vector<__half> y_result(M);
    cudaMemcpy(y_result.data(), y_d, M * sizeof(__half), cudaMemcpyDeviceToHost);

    // Kernel accumulates half inputs in double then casts back to half.
    for (size_t i = 0; i < M; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < N; ++j)
            acc += static_cast<double>(__half2float(A_half[i + j * M])) *
                   static_cast<double>(__half2float(x_half[j]));
        float expected = static_cast<float>(alpha * acc);
        // Half has ~10-bit mantissa; for |values| <= 32, ULP ~= 0.002.
        EXPECT_NEAR(__half2float(y_result[i]), expected, 0.05f) << "index " << i;
    }

    cudaFree(A_d); cudaFree(x_d); cudaFree(y_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
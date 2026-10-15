#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "countsketch_cuda_kernels.cuh"

namespace tomos {
namespace test {

// ---- Host replicas of the hash functions used by the kernels ----

static uint64_t splitmix64_host(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static constexpr uint64_t BUCKET_SEED = 0x12345678abcdef00ull;
static constexpr uint64_t SIGN_SEED   = 0xf00dd00d42babe55ull;

// Reference for countsketch_apply_impl<float>: sign = bit 63 of hash
static void ref_float(const float* X, int n, int m, int B, int ldx,
                      float* Y, int ldy) {
    std::fill(Y, Y + m * B, 0.0f);
    for (int i = 0; i < n; ++i) {
        uint64_t hb = splitmix64_host((uint64_t)i ^ BUCKET_SEED);
        uint64_t hs = splitmix64_host((uint64_t)i ^ SIGN_SEED);
        int   bucket = (int)(hb % (uint64_t)m);
        float sign   = (hs & 1ull) ? 1.0f : -1.0f;
        for (int j = 0; j < B; ++j)
            Y[bucket + j * ldy] += sign * X[i + j * ldx];
    }
}

// Reference for countsketch_apply_impl(__half*, float*): sign = bit 0 of hash
static void ref_half2float(const __half* X, int n, int m, int B, int ldx,
                           float* Y, int ldy) {
    std::fill(Y, Y + m * B, 0.0f);
    for (int i = 0; i < n; ++i) {
        uint64_t hb = splitmix64_host((uint64_t)i ^ BUCKET_SEED);
        uint64_t hs = splitmix64_host((uint64_t)i ^ SIGN_SEED);
        int   bucket = (int)(hb % (uint64_t)m);
        float sign   = (hs & 1ULL) ? 1.0f : -1.0f;
        for (int j = 0; j < B; ++j)
            Y[bucket + j * ldy] += sign * __half2float(X[i + j * ldx]);
    }
}

// ---- Fixture ----

class CountsketchApplyTest : public ::testing::Test {
protected:
    static constexpr int N = 128;  // input rows
    static constexpr int M = 32;   // sketch rows (4× compression)
    static constexpr int B = 4;    // batch (columns)

    std::mt19937 gen{42};
};

// ---- countsketch_apply_impl<float> ----

TEST_F(CountsketchApplyTest, FloatZeroInput) {
    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * B * sizeof(float));
    cudaMalloc(&Y_d, M * B * sizeof(float));
    cudaMemset(X_d, 0, N * B * sizeof(float));
    cudaMemset(Y_d, 0, M * B * sizeof(float));

    cuda::countsketch_apply_impl<float>({N, B}, X_d, N, {M, B}, Y_d, M);

    std::vector<float> Y_h(M * B);
    cudaMemcpy(Y_h.data(), Y_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < M * B; ++k)
        EXPECT_EQ(Y_h[k], 0.0f) << "k=" << k;

    cudaFree(X_d); cudaFree(Y_d);
}

TEST_F(CountsketchApplyTest, FloatMatchesReference) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X_h(N * B);
    for (auto& v : X_h) v = dist(gen);

    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * B * sizeof(float));
    cudaMalloc(&Y_d, M * B * sizeof(float));
    cudaMemcpy(X_d, X_h.data(), N * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(Y_d, 0, M * B * sizeof(float));

    cuda::countsketch_apply_impl<float>({N, B}, X_d, N, {M, B}, Y_d, M);

    std::vector<float> Y_gpu(M * B), Y_ref(M * B);
    cudaMemcpy(Y_gpu.data(), Y_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);
    ref_float(X_h.data(), N, M, B, N, Y_ref.data(), M);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Y_gpu[k], Y_ref[k], 1e-4f) << "k=" << k;

    cudaFree(X_d); cudaFree(Y_d);
}

TEST_F(CountsketchApplyTest, FloatLinear) {
    constexpr float ALPHA = 3.5f;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X_h(N * B), Xs_h(N * B);
    for (int k = 0; k < N * B; ++k) { X_h[k] = dist(gen); Xs_h[k] = ALPHA * X_h[k]; }

    float *X_d, *Xs_d, *Y_d, *Ys_d;
    cudaMalloc(&X_d,  N * B * sizeof(float)); cudaMalloc(&Xs_d, N * B * sizeof(float));
    cudaMalloc(&Y_d,  M * B * sizeof(float)); cudaMalloc(&Ys_d, M * B * sizeof(float));
    cudaMemcpy(X_d,  X_h.data(),  N * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Xs_d, Xs_h.data(), N * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(Y_d, 0, M * B * sizeof(float)); cudaMemset(Ys_d, 0, M * B * sizeof(float));

    cuda::countsketch_apply_impl<float>({N, B}, X_d,  N, {M, B}, Y_d,  M);
    cuda::countsketch_apply_impl<float>({N, B}, Xs_d, N, {M, B}, Ys_d, M);

    std::vector<float> Y_h(M * B), Ys_h(M * B);
    cudaMemcpy(Y_h.data(),  Y_d,  M * B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ys_h.data(), Ys_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Ys_h[k], ALPHA * Y_h[k], 1e-4f) << "k=" << k;

    cudaFree(X_d); cudaFree(Xs_d); cudaFree(Y_d); cudaFree(Ys_d);
}

// CountSketch is an unbiased estimator: E[||Sx||²] = ||x||².
// Test this over many independent random inputs (B columns per trial).
TEST_F(CountsketchApplyTest, FloatNormUnbiased) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int TRIALS = 200;

    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * B * sizeof(float));
    cudaMalloc(&Y_d, M * B * sizeof(float));

    double ratio_sum = 0.0;
    int samples = 0;
    for (int t = 0; t < TRIALS; ++t) {
        std::vector<float> X_h(N * B);
        for (auto& v : X_h) v = dist(gen);
        cudaMemcpy(X_d, X_h.data(), N * B * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(Y_d, 0, M * B * sizeof(float));

        cuda::countsketch_apply_impl<float>({N, B}, X_d, N, {M, B}, Y_d, M);

        std::vector<float> Y_h(M * B);
        cudaMemcpy(Y_h.data(), Y_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);

        for (int j = 0; j < B; ++j) {
            float nx = 0.0f, ny = 0.0f;
            for (int i = 0; i < N; ++i) nx += X_h[i + j * N] * X_h[i + j * N];
            for (int i = 0; i < M; ++i) ny += Y_h[i + j * M] * Y_h[i + j * M];
            if (nx > 0) { ratio_sum += ny / nx; ++samples; }
        }
    }

    double avg = ratio_sum / samples;
    // Variance of ||Sx||²/||x||² ≈ 2/M = 2/32; std ≈ 0.25.
    // Over TRIALS*B=800 samples the mean std ≈ 0.25/sqrt(800) ≈ 0.009.
    EXPECT_NEAR(avg, 1.0, 0.05) << "E[||Sx||²/||x||²] should be 1 for CountSketch";

    cudaFree(X_d); cudaFree(Y_d);
}

// ---- countsketch_apply_impl(__half* X, float* Y) ----

TEST_F(CountsketchApplyTest, HalfToFloatZeroInput) {
    __half* X_d; float* Y_d;
    cudaMalloc(&X_d, N * B * sizeof(__half));
    cudaMalloc(&Y_d, M * B * sizeof(float));
    cudaMemset(X_d, 0, N * B * sizeof(__half));
    cudaMemset(Y_d, 0, M * B * sizeof(float));

    cuda::countsketch_apply_impl({N, B}, X_d, (size_t)N, {M, B}, Y_d, (size_t)M);

    std::vector<float> Y_h(M * B);
    cudaMemcpy(Y_h.data(), Y_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < M * B; ++k)
        EXPECT_EQ(Y_h[k], 0.0f) << "k=" << k;

    cudaFree(X_d); cudaFree(Y_d);
}

TEST_F(CountsketchApplyTest, HalfToFloatMatchesReference) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> X_h(N * B);
    for (auto& v : X_h) v = __float2half(dist(gen));

    __half* X_d; float* Y_d;
    cudaMalloc(&X_d, N * B * sizeof(__half));
    cudaMalloc(&Y_d, M * B * sizeof(float));
    cudaMemcpy(X_d, X_h.data(), N * B * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(Y_d, 0, M * B * sizeof(float));

    cuda::countsketch_apply_impl({N, B}, X_d, (size_t)N, {M, B}, Y_d, (size_t)M);

    std::vector<float> Y_gpu(M * B), Y_ref(M * B);
    cudaMemcpy(Y_gpu.data(), Y_d, M * B * sizeof(float), cudaMemcpyDeviceToHost);
    ref_half2float(X_h.data(), N, M, B, N, Y_ref.data(), M);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Y_gpu[k], Y_ref[k], 1e-3f) << "k=" << k;

    cudaFree(X_d); cudaFree(Y_d);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
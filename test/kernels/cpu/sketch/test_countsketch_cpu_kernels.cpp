#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "countsketch_kernels.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace test {

// ---- generate_countsketch_csr_values ----

TEST(GenerateCountsketchValuesTest, AllPlusMinus1Double) {
    constexpr size_t N = 1024;
    std::vector<double> vals(N);

    auto st = cpu::generate_countsketch_csr_values<double>(N, vals.data());
    EXPECT_EQ(st, SUCCESS);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(vals[i] == 1.0 || vals[i] == -1.0)
            << "value at " << i << " is " << vals[i];
    }
}

TEST(GenerateCountsketchValuesTest, AllPlusMinus1Float) {
    constexpr size_t N = 1024;
    std::vector<float> vals(N);

    auto st = cpu::generate_countsketch_csr_values<float>(N, vals.data());
    EXPECT_EQ(st, SUCCESS);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(vals[i] == 1.0f || vals[i] == -1.0f)
            << "value at " << i << " is " << vals[i];
    }
}

// For a large enough array, both signs should appear.
TEST(GenerateCountsketchValuesTest, BothSignsPresentDouble) {
    constexpr size_t N = 4096;
    std::vector<double> vals(N);
    cpu::generate_countsketch_csr_values<double>(N, vals.data());

    int pos = 0, neg = 0;
    for (double v : vals) { if (v > 0) ++pos; else ++neg; }
    EXPECT_GT(pos, 0) << "no positive entries";
    EXPECT_GT(neg, 0) << "no negative entries";
}

TEST(GenerateCountsketchValuesTest, BothSignsPresentFloat) {
    constexpr size_t N = 4096;
    std::vector<float> vals(N);
    cpu::generate_countsketch_csr_values<float>(N, vals.data());

    int pos = 0, neg = 0;
    for (float v : vals) { if (v > 0) ++pos; else ++neg; }
    EXPECT_GT(pos, 0) << "no positive entries";
    EXPECT_GT(neg, 0) << "no negative entries";
}

// Two independent calls should not be identical (uses srand(time(0))).
// This test is probabilistic but practically certain for N=1024.
TEST(GenerateCountsketchValuesTest, TwoCallsDifferDouble) {
    constexpr size_t N = 1024;
    std::vector<double> a(N), b(N);
    cpu::generate_countsketch_csr_values<double>(N, a.data());
    cpu::generate_countsketch_csr_values<double>(N, b.data());

    int diff = 0;
    for (size_t i = 0; i < N; ++i) diff += (a[i] != b[i]) ? 1 : 0;
    // Two independent Rademacher draws differ ~50% of entries in expectation.
    // Probability of seeing 0 differences is (0.5)^1024 — effectively impossible.
    EXPECT_GT(diff, 0) << "two calls produced identical patterns";
}

// ---- countsketch_apply_impl ----

// Host replica of the hash used by countsketch_apply_impl.
static uint64_t splitmix64_ref(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static constexpr uint64_t BUCKET_SEED = 0x12345678abcdef00ull;
static constexpr uint64_t SIGN_SEED   = 0xf00dd00d42babe55ull;

template <typename T>
static void ref_apply(const T* X, int n, int m, int B, int ldx,
                      T* Y, int ldy) {
    std::fill(Y, Y + m * B, T(0));
    for (int i = 0; i < n; ++i) {
        uint64_t hb = splitmix64_ref((uint64_t)i ^ BUCKET_SEED);
        uint64_t hs = splitmix64_ref((uint64_t)i ^ SIGN_SEED);
        int bucket = (int)(hb % (uint64_t)m);
        T sign = (hs & 1ull) ? T(1) : T(-1);
        for (int j = 0; j < B; ++j)
            Y[bucket + j * ldy] += sign * X[i + j * ldx];
    }
}

TEST(CountsketchApplyCpuTest, FloatZeroInput) {
    constexpr int N = 128, M = 32, B = 4;
    std::vector<float> X(N * B, 0.0f), Y(M * B, 0.0f);
    cpu::countsketch_apply_impl<float>({N, B}, X.data(), N, {M, B}, Y.data(), M);
    for (float v : Y) EXPECT_EQ(v, 0.0f);
}

TEST(CountsketchApplyCpuTest, FloatMatchesReference) {
    constexpr int N = 128, M = 32, B = 4;
    std::vector<float> X(N * B), Y(M * B, 0.0f), Y_ref(M * B);
    for (int k = 0; k < N * B; ++k) X[k] = (k % 7) * 0.1f - 0.3f;

    cpu::countsketch_apply_impl<float>({N, B}, X.data(), N, {M, B}, Y.data(), M);
    ref_apply<float>(X.data(), N, M, B, N, Y_ref.data(), M);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Y[k], Y_ref[k], 1e-5f) << "k=" << k;
}

TEST(CountsketchApplyCpuTest, FloatLinear) {
    constexpr int N = 128, M = 32, B = 4;
    constexpr float ALPHA = 2.5f;
    std::vector<float> X(N * B), Xs(N * B), Y(M * B, 0.0f), Ys(M * B, 0.0f);
    for (int k = 0; k < N * B; ++k) { X[k] = (k % 5) * 0.2f - 0.4f; Xs[k] = ALPHA * X[k]; }

    cpu::countsketch_apply_impl<float>({N, B}, X.data(),  N, {M, B}, Y.data(),  M);
    cpu::countsketch_apply_impl<float>({N, B}, Xs.data(), N, {M, B}, Ys.data(), M);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Ys[k], ALPHA * Y[k], 1e-5f) << "k=" << k;
}

TEST(CountsketchApplyCpuTest, DoubleMatchesReference) {
    constexpr int N = 256, M = 64, B = 3;
    std::vector<double> X(N * B), Y(M * B, 0.0), Y_ref(M * B);
    for (int k = 0; k < N * B; ++k) X[k] = (k % 11) * 0.1 - 0.5;

    cpu::countsketch_apply_impl<double>({N, B}, X.data(), N, {M, B}, Y.data(), M);
    ref_apply<double>(X.data(), N, M, B, N, Y_ref.data(), M);

    for (int k = 0; k < M * B; ++k)
        EXPECT_NEAR(Y[k], Y_ref[k], 1e-12) << "k=" << k;
}

TEST(CountsketchApplyCpuTest, FloatNormUnbiased) {
    constexpr int N = 128, M = 32, B = 4;
    constexpr int TRIALS = 200;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    double ratio_sum = 0.0;
    int samples = 0;
    std::vector<float> X(N * B), Y(M * B);
    for (int t = 0; t < TRIALS; ++t) {
        for (float& v : X) v = dist(rng);
        std::fill(Y.begin(), Y.end(), 0.0f);
        cpu::countsketch_apply_impl<float>({N, B}, X.data(), N, {M, B}, Y.data(), M);
        for (int j = 0; j < B; ++j) {
            float nx = 0.0f, ny = 0.0f;
            for (int i = 0; i < N; ++i) nx += X[i + j * N] * X[i + j * N];
            for (int i = 0; i < M; ++i) ny += Y[i + j * M] * Y[i + j * M];
            if (nx > 0) { ratio_sum += ny / nx; ++samples; }
        }
    }
    EXPECT_NEAR(ratio_sum / samples, 1.0, 0.05);
}

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
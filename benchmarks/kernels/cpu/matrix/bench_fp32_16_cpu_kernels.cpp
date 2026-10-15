#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"

#include <random>
#include <vector>

#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"
#include "matrix_cpu_kernels.hpp"
#include "mm_cpu_kernels.hpp"

namespace tomos {
namespace bench {

namespace {

// Builds a random A, split into its wh/wr halves via fp32_16_split. Values
// are drawn from [-1,1], comfortably within the exact/lossless normal range
// (see fp32_16_pack.hpp), so these benchmarks measure the kernels'
// throughput rather than any degraded-tier branch's cost.
void make_wh_wr(size_t n, std::mt19937& gen, std::vector<uint16_t>& wh,
                std::vector<uint16_t>& wr) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    wh.resize(n);
    wr.resize(n);
    for (size_t i = 0; i < n; ++i)
        fp32_16_split(static_cast<float>(dist(gen)), wh[i], wr[i]);
}

}  // namespace

// wh-only tiers (pure_half, fast) move half the bytes of a plain FP32 A
// (2B/elem vs 4B/elem); ext reads both wh and wr, the same footprint as FP32.
static int64_t bytes_per_mv(size_t M, size_t N, size_t a_bytes_per_elem) {
    return static_cast<int64_t>(M * N * a_bytes_per_elem + N * sizeof(float) +
                                2 * M * sizeof(float));
}

static int64_t bytes_per_mm(size_t M, size_t K, size_t N,
                           size_t a_bytes_per_elem) {
    return static_cast<int64_t>(M * K * a_bytes_per_elem +
                                K * N * sizeof(float) +
                                2 * M * N * sizeof(float));
}

// Small smoke-test points (kept for quick regression runs) plus three
// progressively larger sweeps: square (M=N), tall (N fixed at 1024, M
// scaling up), and wide (M fixed at 1024, N scaling up). mv cost is linear
// in M*N even through these naive per-element kernels, so the same ladder
// used for the plain FP32 baseline (and bench_mv_cpu_kernels.cpp) is safe to
// reuse here too — see there for the memory math (wh/wr are 2B/elem each, so
// this file's actual footprint at every point is at most half of that ladder's).
static void MvArgs(benchmark::internal::Benchmark* b) {
    b->Args({64, 32})->Args({1024, 512})->Args({4096, 2048})->Args({16384, 8192});
    for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}) b->Args({n, n});
    for (int64_t m : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({m, 1024});
    for (int64_t n : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({1024, n});
}

// Progressively larger sweeps for mm: square (M=K=N), tall (K and N fixed,
// M scaling up), and wide (M and K fixed, N scaling up). GEMM cost is cubic
// in the square sweep but only linear in tall/wide (only one dimension
// grows), so the square ladder is capped lower for pure_half/fast/accurate
// (naive per-element loops) than for the OpenBLAS-backed baseline, while all
// four share the same tall/wide ladder. pure_half is the slow outlier here
// (measured ~0.075 GFlop/s): its top square point (4096^3, left unchanged —
// already time-bound, not memory-bound) is expected to take on the order of
// half an hour, and its top tall/wide point (2097152 against a fixed
// dimension of 256) on the order of an hour — deliberate large-scale stress
// points, not quick regression checks. fast/accurate run roughly an order of
// magnitude faster than pure_half at the same size.
static void MmArgsSlow(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 512, 1024, 2048, 4096}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 256, 256});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({256, 256, n});
}

// Matches bench_mm_cpu_kernels.cpp's MmArgsFast ladder — see there for the
// memory math.
static void MmArgsFast(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 49152}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 512, 512});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({512, 512, n});
}

// ---- mv_fp32_16_* ----

static void BM_mv_fp32_16_pure_half(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * N, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> x(N), y(M, 0.0f);
    for (auto& v : x) v = static_cast<float>(dist(gen));

    cpu::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                              wh.data(), M, x.data(), 1, 0.0, y.data(), 1);
    for (auto _ : state) {
        cpu::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                  wh.data(), M, x.data(), 1, 0.0, y.data(), 1);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mv(M, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mv_fp32_16_pure_half)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mv_fp32_16_fast(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * N, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> x(N), y(M, 0.0f);
    for (auto& v : x) v = static_cast<float>(dist(gen));

    cpu::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh.data(),
                        M, x.data(), 1, 0.0, y.data(), 1);
    for (auto _ : state) {
        cpu::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                            wh.data(), M, x.data(), 1, 0.0, y.data(), 1);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mv(M, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mv_fp32_16_fast)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mv_fp32_16_accurate(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * N, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> x(N), y(M, 0.0f);
    for (auto& v : x) v = static_cast<float>(dist(gen));

    cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                             wh.data(), wr.data(), M, x.data(), 1, 0.0,
                             y.data(), 1);
    for (auto _ : state) {
        cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                 wh.data(), wr.data(), M, x.data(), 1, 0.0,
                                 y.data(), 1);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mv(M, N, 2 * sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mv_fp32_16_accurate)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Plain FP32 baseline for direct comparison against the three tiers above.
static void BM_mv_strided_fp32_baseline(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> A(M * N), x(N), y(M, 0.0f);
    for (auto& v : A) v = static_cast<float>(dist(gen));
    for (auto& v : x) v = static_cast<float>(dist(gen));

    cpu::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, 1.0,
                                        {M, N}, A.data(), M, x.data(), 1, 0.0,
                                        y.data(), 1);
    for (auto _ : state) {
        cpu::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, 1.0,
                                            {M, N}, A.data(), M, x.data(), 1,
                                            0.0, y.data(), 1);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * bytes_per_mv(M, N, 4));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mv_strided_fp32_baseline)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_* ----

static void BM_mm_fp32_16_pure_half(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * K, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> B(K * N), C(M * N, 0.0f);
    for (auto& v : B) v = static_cast<float>(dist(gen));

    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE,
                              TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                              wh.data(), M, {K, N}, B.data(), K, 0.0,
                              C.data(), M);
    for (auto _ : state) {
        cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE,
                                  TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                                  wh.data(), M, {K, N}, B.data(), K, 0.0,
                                  C.data(), M);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_pure_half)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mm_fp32_16_fast(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * K, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> B(K * N), C(M * N, 0.0f);
    for (auto& v : B) v = static_cast<float>(dist(gen));

    cpu::mm_fp32_16_fast(TransposeType::NO_TRANSPOSE,
                        TransposeType::NO_TRANSPOSE, 1.0, {M, K}, wh.data(),
                        M, {K, N}, B.data(), K, 0.0, C.data(), M);
    for (auto _ : state) {
        cpu::mm_fp32_16_fast(TransposeType::NO_TRANSPOSE,
                            TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                            wh.data(), M, {K, N}, B.data(), K, 0.0, C.data(),
                            M);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_fast)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mm_fp32_16_accurate(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    std::mt19937 gen(42);
    std::vector<uint16_t> wh, wr;
    make_wh_wr(M * K, gen, wh, wr);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> B(K * N), C(M * N, 0.0f);
    for (auto& v : B) v = static_cast<float>(dist(gen));

    cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                             TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                             wh.data(), wr.data(), M, {K, N}, B.data(), K,
                             0.0, C.data(), M);
    for (auto _ : state) {
        cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                                 TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                                 wh.data(), wr.data(), M, {K, N}, B.data(), K,
                                 0.0, C.data(), M);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, 2 * sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_accurate)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Plain FP32 baseline for direct comparison against the three tiers above.
static void BM_mm_strided_fp32_baseline(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
    for (auto& v : A) v = static_cast<float>(dist(gen));
    for (auto& v : B) v = static_cast<float>(dist(gen));

    cpu::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, A.data(), M, {K, N}, B.data(), K, 0.0, C.data(), M);
    for (auto _ : state) {
        cpu::mm_strided<float, float, float>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, A.data(), M, {K, N}, B.data(), K, 0.0, C.data(), M);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * bytes_per_mm(M, K, N, 4));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_strided_fp32_baseline)
    ->Apply(MmArgsFast)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

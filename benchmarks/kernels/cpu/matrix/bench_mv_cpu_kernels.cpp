#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"

#include <random>
#include <vector>

#include "matrix_cpu_kernels.hpp"

namespace tomos {
namespace bench {

template <typename a_t, typename x_t, typename y_t>
static int64_t bytes_per_iter(size_t M, size_t N) {
    return static_cast<int64_t>(M * N * sizeof(a_t) +
                                N * sizeof(x_t) +
                                2 * M * sizeof(y_t));
}

template <typename a_t, typename x_t, typename y_t>
static void BM_mv_strided(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<a_t> A(M * N);
    std::vector<x_t> x(N);
    std::vector<y_t> y(M, static_cast<y_t>(0));
    for (auto& v : A) v = static_cast<a_t>(dist(gen));
    for (auto& v : x) v = static_cast<x_t>(dist(gen));

    // Warm up — pulls data into cache for small sizes.
    cpu::mv_strided<a_t, x_t, y_t>(
        TransposeType::NO_TRANSPOSE, 1.0, {M, N},
        A.data(), M, x.data(), 1, 0.0, y.data(), 1);

    for (auto _ : state) {
        cpu::mv_strided<a_t, x_t, y_t>(
            TransposeType::NO_TRANSPOSE, 1.0, {M, N},
            A.data(), M, x.data(), 1, 0.0, y.data(), 1);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * bytes_per_iter<a_t, x_t, y_t>(M, N));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

// Small smoke-test points (kept for quick regression runs) plus three
// progressively larger sweeps: square (M=N), tall (N fixed at 1024, M
// scaling up), and wide (M fixed at 1024, N scaling up). mv cost is linear
// in M*N even through the generic (non-BLAS) fallback, so the same ladder is
// safe to reuse across every type combination below. Sized for the worst
// case (double, 8B/elem): the top square point (131072^2) is ~137GB, and the
// top tall/wide point (4194304 against a fixed dimension of 1024) is
// ~34GB — pick a host with enough RAM (or trim these) if that's too much
// for yours.
static void MvArgs(benchmark::internal::Benchmark* b) {
    b->Args({64, 32})->Args({1024, 512})->Args({4096, 2048})->Args({16384, 8192});
    for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}) b->Args({n, n});
    for (int64_t m : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({m, 1024});
    for (int64_t n : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({1024, n});
}

// ddd and fff go through OpenBLAS; mixed types use the generic fallback.
BENCHMARK_TEMPLATE(BM_mv_strided, double, double, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mv_strided, float, float, float)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mv_strided, double, float, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mv_strided, float, double, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mv_strided, double, double, float)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

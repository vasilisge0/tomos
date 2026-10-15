#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"

#include <random>
#include <vector>

#include "cblas.h"
#include "mm_cpu_kernels.hpp"

namespace tomos {
namespace bench {

// Bytes: read A + read B + read C + write C.
template <typename a_t, typename b_t, typename c_t>
static int64_t bytes_per_gemm(size_t M, size_t K, size_t N) {
    return static_cast<int64_t>(M * K * sizeof(a_t) + K * N * sizeof(b_t) +
                                2 * M * N * sizeof(c_t));
}

// Progressively larger sweeps: square (M=K=N), tall (K and N fixed, M
// scaling up), and wide (M and K fixed, N scaling up). GEMM cost is cubic in
// the square sweep but only linear in the tall/wide sweeps (only one
// dimension grows), so the square ladder is capped lower for the generic
// (non-BLAS) fallback used by the mixed-precision combos below, while all
// combos share the same tall/wide ladder. The top square point (49152^3,
// double A+B+C) is ~58GB and the top tall/wide point (2097152 against a
// fixed dimension of 512) is ~17.2GB — pick a host with enough RAM (or trim
// these) if that's too much for yours.
static void MmArgsFast(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 49152}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 512, 512});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({512, 512, n});
}

// The generic fallback (single-threaded, unvectorized) is measurably slower
// than the BLAS combos but not dramatically so (~0.2s at n=1024, so ~15s
// projected at the top square point below) — capped one tier below
// MmArgsFast's square ladder mainly because its cubic cost adds up faster,
// not because any single point here is expected to be slow in isolation.
// Square is left unchanged here (not pushed further like MmArgsFast's) since
// it's time-bound rather than memory-bound; the tall/wide top, though, is
// cheap (linear cost) and safe to extend the same as everywhere else.
static void MmArgsSlow(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 512, 1024, 2048, 4096}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 256, 256});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({256, 256, n});
}

// ---- mm_strided benchmarks ----

template <typename a_t, typename b_t, typename c_t>
static void BM_mm_strided(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<a_t> A(M * K);
    std::vector<b_t> B(K * N);
    std::vector<c_t> C(M * N, static_cast<c_t>(0));
    for (auto& v : A) v = static_cast<a_t>(dist(gen));
    for (auto& v : B) v = static_cast<b_t>(dist(gen));

    cpu::mm_strided<a_t, b_t, c_t>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, A.data(), M, {K, N}, B.data(), K, 0.0, C.data(), M);

    for (auto _ : state) {
        cpu::mm_strided<a_t, b_t, c_t>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, A.data(), M, {K, N}, B.data(), K, 0.0, C.data(), M);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() *
                            bytes_per_gemm<a_t, b_t, c_t>(M, K, N));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

// ddd and fff go through OpenBLAS cblas_dgemm/cblas_sgemm.
BENCHMARK_TEMPLATE(BM_mm_strided, double, double, double)
    ->Apply(MmArgsFast)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mm_strided, float, float, float)
    ->Apply(MmArgsFast)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Mixed types use the generic fallback.
BENCHMARK_TEMPLATE(BM_mm_strided, double, float, double)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mm_strided, float, double, double)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mm_strided, double, double, float)
    ->Apply(MmArgsSlow)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_csr benchmarks ----

static void BM_mm_csr(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));
    const double density = 0.01;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::uniform_real_distribution<double> keep(0.0, 1.0);

    std::vector<double> vals;
    std::vector<int> rp(M + 1, 0), ci;
    for (size_t i = 0; i < M; ++i) {
        rp[i] = static_cast<int>(vals.size());
        for (size_t j = 0; j < K; ++j) {
            if (keep(gen) < density) {
                vals.push_back(val(gen));
                ci.push_back(static_cast<int>(j));
            }
        }
    }
    rp[M] = static_cast<int>(vals.size());

    std::vector<double> B(K * N), C(M * N, 0.0);
    for (auto& v : B) v = val(gen);

    size_t nnz = vals.size();

    cpu::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, nnz, vals.data(), rp.data(), ci.data(),
        {K, N}, B.data(), K, 0.0, C.data(), M);

    for (auto _ : state) {
        cpu::mm_csr<double, double, double, int>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, nnz, vals.data(), rp.data(), ci.data(),
            {K, N}, B.data(), K, 0.0, C.data(), M);
        benchmark::ClobberMemory();
    }

    state.counters["nnz"] = static_cast<double>(nnz);
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * nnz * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

// Rows × inner-dim × output-cols; 1% density.
BENCHMARK(BM_mm_csr)
    ->Args({1024, 1024, 64})->Args({4096, 4096, 64})->Args({16384, 1024, 64})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

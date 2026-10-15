#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"
#include "bench_cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <random>
#include <vector>

#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_pack.hpp"
#include "mm_cuda_kernels.cuh"
#include "mv_cuda_kernels.cuh"

namespace tomos {
namespace bench {

namespace {

// Builds a random A host-side, split into wh/wr via fp32_16_split (values in
// [-1,1], comfortably within the exact/lossless normal range -- see
// fp32_16_pack.hpp), then uploads both halves to the given (already
// allocated — see DevBuf) device buffers.
void make_device_wh_wr(size_t n, std::mt19937& gen, DevBuf<uint16_t>& wh_d,
                       DevBuf<uint16_t>& wr_d) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<uint16_t> wh(n), wr(n);
    for (size_t i = 0; i < n; ++i)
        fp32_16_split(static_cast<float>(dist(gen)), wh[i], wr[i]);

    cudaMemcpy(wh_d, wh.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(wr_d, wr.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
}

}  // namespace

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
// scaling up), and wide (M fixed at 1024, N scaling up). wh/wr are 2B/elem
// each (4B/elem combined), so this ladder's device (and transient
// host-staging) footprint is lighter than bench_mv_cuda_kernels.cu's
// all-double ladder at the same size, letting the square top go higher:
// 90112^2 wh+wr is ~32.5GB, comfortably under a 40GB A100 with ~7.5GB
// headroom for CUDA context/driver overhead (versus that file's own 65536
// cap needing ~34.4GB for a single double buffer). The tall/wide top
// (4194304 x 1024) is only ~17.2GB, well within the same budget.
static void MvArgs(benchmark::internal::Benchmark* b) {
    b->Args({64, 32})->Args({1024, 512})->Args({4096, 2048})->Args({16384, 8192});
    for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 90112}) b->Args({n, n});
    for (int64_t m : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({m, 1024});
    for (int64_t n : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({1024, n});
}

// Progressively larger A100-scale sweeps for mm: square (M=K=N), tall (K and
// N fixed at 512, M scaling up), and wide (M and K fixed at 512, N scaling
// up). All four tiers below run as GPU kernels, so a single ladder is fast
// enough to share across pure_half/fast/accurate/baseline. wh is 2B/elem and
// B/C are float (4B/elem), so this ladder's footprint is lighter than
// bench_mm_cuda_kernels.cu's all-double one at the same size — its top
// square point (49152) needs only ~24.2GB, versus that file's 32768 cap
// needing ~25.8GB; the top tall/wide point (2097152 against fixed
// dimensions of 512) is only ~6.4GB.
static void MmArgs(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 49152}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 512, 512});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({512, 512, n});
}

// ---- mv_fp32_16_* ----

static void BM_mv_fp32_16_pure_half(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    DevBuf<uint16_t> wh_d(M * N), wr_d(M * N);
    DevBuf<float> x_d(N), y_d(M);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), x_d.status(), y_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * N, gen, wh_d, wr_d);

    cuda::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh_d,
                               M, x_d, 1, 0.0, y_d, 1);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                   wh_d, M, x_d, 1, 0.0, y_d, 1);
        if (!check_kernel_launch(state)) return;
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

    DevBuf<uint16_t> wh_d(M * N), wr_d(M * N);
    DevBuf<float> x_d(N), y_d(M);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), x_d.status(), y_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * N, gen, wh_d, wr_d);

    cuda::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh_d, M,
                          x_d, 1, 0.0, y_d, 1);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh_d,
                              M, x_d, 1, 0.0, y_d, 1);
        if (!check_kernel_launch(state)) return;
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

    DevBuf<uint16_t> wh_d(M * N), wr_d(M * N);
    DevBuf<float> x_d(N), y_d(M);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), x_d.status(), y_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * N, gen, wh_d, wr_d);

    cuda::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh_d,
                              wr_d, M, x_d, 1, 0.0, y_d, 1);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                  wh_d, wr_d, M, x_d, 1, 0.0, y_d, 1);
        if (!check_kernel_launch(state)) return;
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mv(M, N, 2 * sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mv_fp32_16_accurate)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Plain FP32 baseline (cuBLAS Sgemv via mv_strided) for direct comparison.
static void BM_mv_strided_fp32_baseline(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("cublasCreate failed");
        return;
    }

    DevBuf<float> A_d(M * N), x_d(N), y_d(M);
    if (!check_alloc(state, {A_d.status(), x_d.status(), y_d.status()})) {
        cublasDestroy(handle);
        return;
    }

    cuda::experimental::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, 1.0,
                                         {M, N}, A_d, M, x_d, 1, 0.0, y_d, 1,
                                         handle);
    if (!check_kernel_launch(state)) {
        cublasDestroy(handle);
        return;
    }
    for (auto _ : state) {
        cuda::experimental::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE,
                                             1.0, {M, N}, A_d, M, x_d, 1, 0.0,
                                             y_d, 1, handle);
        if (!check_kernel_launch(state)) {
            cublasDestroy(handle);
            return;
        }
    }
    state.SetBytesProcessed(state.iterations() * bytes_per_mv(M, N, 4));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cublasDestroy(handle);
}
BENCHMARK(BM_mv_strided_fp32_baseline)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_* ----

static void BM_mm_fp32_16_pure_half(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    DevBuf<uint16_t> wh_d(M * K), wr_d(M * K);
    DevBuf<float> B_d(K * N), C_d(M * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * K, gen, wh_d, wr_d);

    cuda::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE,
                               TransposeType::NO_TRANSPOSE, 1.0, {M, K}, wh_d,
                               M, {K, N}, B_d, K, 0.0, C_d, M);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE,
                                   TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                                   wh_d, M, {K, N}, B_d, K, 0.0, C_d, M);
        if (!check_kernel_launch(state)) return;
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_pure_half)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mm_fp32_16_fast(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    DevBuf<uint16_t> wh_d(M * K), wr_d(M * K);
    DevBuf<float> B_d(K * N), C_d(M * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * K, gen, wh_d, wr_d);

    cuda::mm_fp32_16_fast(TransposeType::NO_TRANSPOSE,
                          TransposeType::NO_TRANSPOSE, 1.0, {M, K}, wh_d, M,
                          {K, N}, B_d, K, 0.0, C_d, M);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mm_fp32_16_fast(TransposeType::NO_TRANSPOSE,
                              TransposeType::NO_TRANSPOSE, 1.0, {M, K}, wh_d,
                              M, {K, N}, B_d, K, 0.0, C_d, M);
        if (!check_kernel_launch(state)) return;
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_fast)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_mm_fp32_16_accurate(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    DevBuf<uint16_t> wh_d(M * K), wr_d(M * K);
    DevBuf<float> B_d(K * N), C_d(M * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    std::mt19937 gen(42);
    make_device_wh_wr(M * K, gen, wh_d, wr_d);

    cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                              TransposeType::NO_TRANSPOSE, 1.0, {M, K}, wh_d,
                              wr_d, M, {K, N}, B_d, K, 0.0, C_d, M);
    if (!check_kernel_launch(state)) return;
    for (auto _ : state) {
        cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE,
                                  TransposeType::NO_TRANSPOSE, 1.0, {M, K},
                                  wh_d, wr_d, M, {K, N}, B_d, K, 0.0, C_d, M);
        if (!check_kernel_launch(state)) return;
    }
    state.SetBytesProcessed(state.iterations() *
                            bytes_per_mm(M, K, N, 2 * sizeof(uint16_t)));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_mm_fp32_16_accurate)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Plain FP32 baseline (cuBLAS Sgemm via mm_strided) for direct comparison.
static void BM_mm_strided_fp32_baseline(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("cublasCreate failed");
        return;
    }

    DevBuf<float> A_d(M * K), B_d(K * N), C_d(M * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) {
        cublasDestroy(handle);
        return;
    }

    cuda::experimental::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, A_d, M, {K, N}, B_d, K, 0.0, C_d, M, handle);
    if (!check_kernel_launch(state)) {
        cublasDestroy(handle);
        return;
    }
    for (auto _ : state) {
        cuda::experimental::mm_strided<float, float, float>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, A_d, M, {K, N}, B_d, K, 0.0, C_d, M, handle);
        if (!check_kernel_launch(state)) {
            cublasDestroy(handle);
            return;
        }
    }
    state.SetBytesProcessed(state.iterations() * bytes_per_mm(M, K, N, 4));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cublasDestroy(handle);
}
BENCHMARK(BM_mm_strided_fp32_baseline)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

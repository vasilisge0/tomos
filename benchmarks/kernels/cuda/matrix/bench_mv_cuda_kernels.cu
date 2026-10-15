#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"
#include "bench_cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mv_cuda_kernels.cuh"

namespace tomos {
namespace bench {

// Bytes touched per matvec: read A + read x + read y + write y.
template <typename a_t, typename x_t, typename y_t>
static int64_t bytes_per_iter(size_t M, size_t N) {
    return static_cast<int64_t>(M * N * sizeof(a_t) +
                                N * sizeof(x_t) +
                                2 * M * sizeof(y_t));
}

// Small smoke-test points (kept for quick regression runs) plus three
// progressively larger sweeps: square (M=N), tall (N fixed at 1024, M
// scaling up), and wide (M fixed at 1024, N scaling up). The square sweep's
// top point is capped well below bench_mv_cpu_kernels.cpp's (which assumes
// generous host RAM) since it's real device VRAM on the line here: 65536^2
// double is ~34.4GB, leaving ~5.6GB of headroom on a 40GB A100 for CUDA
// context/driver overhead — tight, but the tall/wide sweeps land at exactly
// the same footprint (4194304 x 1024 is also ~34.4GB) despite scaling only
// one dimension, so there was no reason to cap them any lower.
static void MvArgs(benchmark::internal::Benchmark* b) {
    b->Args({64, 32})->Args({1024, 512})->Args({4096, 2048})->Args({16384, 8192});
    for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536}) b->Args({n, n});
    for (int64_t m : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({m, 1024});
    for (int64_t n : {4096, 16384, 65536, 262144, 1048576, 4194304}) b->Args({1024, n});
}

// ---- gemv_generic_kernel (direct kernel call, always the naive kernel) ----
//
// Registered before the cuBLAS benchmarks below (see BM_mv_cublas) so results
// print/save in that order: generic kernel numbers for a type combination
// appear right before its cuBLAS counterpart, making the two directly
// comparable side by side.

template <typename a_t, typename x_t, typename y_t>
static void BM_gemv_generic_kernel(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    DevBuf<a_t> A_d(M * N);
    DevBuf<x_t> x_d(N);
    DevBuf<y_t> y_d(M);
    if (!check_alloc(state, {A_d.status(), x_d.status(), y_d.status()})) return;

    const int threads = 256;
    const int blocks  = ((int)M + threads - 1) / threads;

    // Warm up — ensures kernel is JIT-compiled and caches are warm.
    cuda::experimental::gemv_generic_kernel<a_t, x_t, y_t><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, 1.0, M, N, A_d, M, x_d, 0.0, y_d);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemv_generic_kernel<a_t, x_t, y_t><<<blocks, threads>>>(
            TransposeType::NO_TRANSPOSE, 1.0, M, N, A_d, M, x_d, 0.0, y_d);
        if (!check_kernel_launch(state)) return;
    }

    state.SetBytesProcessed(state.iterations() * bytes_per_iter<a_t, x_t, y_t>(M, N));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

// ddd / fff / hhh
BENCHMARK_TEMPLATE(BM_gemv_generic_kernel, double, double, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_gemv_generic_kernel, float, float, float)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_gemv_generic_kernel, __half, __half, __half)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// dfd has no cuBLAS specialization (see BM_mv_cublas below), so it only ever
// runs through this kernel — benchmarked here rather than under the cuBLAS
// name so that name always means what it says.
BENCHMARK_TEMPLATE(BM_gemv_generic_kernel, double, float, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- BM_mv_cublas (cuda::mv_strided's cuBLAS specializations only) ----
//
// cuda::mv_strided has full template specializations for <double,double,
// double>, <float,float,float>, and <__half,__half,__half> that call
// cublasDgemv/cublasSgemv/cublasHgemm(n=1) instead of its generic-template
// body (which just calls gemv_generic_kernel — see mv_cuda_kernels.cu).
// Only those three combinations are instantiated below, so every result
// printed/saved under this name is genuinely cuBLAS, never a silent
// fallback to the generic kernel.

template <typename a_t, typename x_t, typename y_t>
static void BM_mv_cublas(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("cublasCreate failed");
        return;
    }

    DevBuf<a_t> A_d(M * N);
    DevBuf<x_t> x_d(N);
    DevBuf<y_t> y_d(M);
    if (!check_alloc(state, {A_d.status(), x_d.status(), y_d.status()})) {
        cublasDestroy(handle);
        return;
    }

    // Warm up
    cuda::experimental::mv_strided<a_t, x_t, y_t>(
        TransposeType::NO_TRANSPOSE, 1.0, {M, N}, A_d, M, x_d, 1, 0.0, y_d, 1, handle);
    if (!check_kernel_launch(state)) {
        cublasDestroy(handle);
        return;
    }

    for (auto _ : state) {
        cuda::experimental::mv_strided<a_t, x_t, y_t>(
            TransposeType::NO_TRANSPOSE, 1.0, {M, N}, A_d, M, x_d, 1, 0.0, y_d, 1, handle);
        if (!check_kernel_launch(state)) {
            cublasDestroy(handle);
            return;
        }
    }

    state.SetBytesProcessed(state.iterations() * bytes_per_iter<a_t, x_t, y_t>(M, N));
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cublasDestroy(handle);
}

BENCHMARK_TEMPLATE(BM_mv_cublas, double, double, double)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mv_cublas, float, float, float)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// cuBLAS has no native half-precision GEMV; this instantiation goes through
// cuda::experimental::mv_strided<__half,__half,__half>'s cublasHgemm(n=1) workaround (see
// mv_cuda_kernels.cu), so it's still genuinely cuBLAS/tensor-core-backed.
BENCHMARK_TEMPLATE(BM_mv_cublas, __half, __half, __half)
    ->Apply(MvArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

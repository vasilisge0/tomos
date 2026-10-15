#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"
#include "bench_cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "mm_cuda_kernels.cuh"

namespace tomos {
namespace bench {

// Square-only (M=K=N) fp32 CUDA GEMM sweep, scoped down from
// bench_mm_cuda_kernels.cu's full double/float/half x square/tall/wide matrix
// so a build+run cycle while tuning a new fp32 GEMM kernel stays fast. Same
// square ladder as that file's MmArgs; the top point (32768^3, A+B+C all
// fp32) is ~12.9GB, comfortable headroom on a 40GB A100 for CUDA
// context/driver overhead.
static void SquareFp32Args(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768}) b->Args({n});
}

// ---- gemm_generic_kernel (direct kernel call) ----
//
// Currently the only custom fp32 GEMM kernel -- the naive one-thread-per-
// output-element baseline from mm_cuda_kernels.cu. As tiled/shared-memory
// variants are added while iterating on performance, add their own
// BENCHMARK(...) block here alongside this one so their numbers print next
// to the naive baseline and the cuBLAS ceiling below.

static void BM_gemm_fp32_square_generic(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    cuda::experimental::gemm_generic_kernel<float, float, float><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_generic_kernel<float, float, float><<<blocks, threads>>>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_generic)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled (shared-memory-tiled, register-blocked custom kernel) ----
//
// Registered second so its numbers print between the naive baseline above
// and the cuBLAS ceiling below -- the two things it's meant to be compared
// against.

static void BM_gemm_fp32_square_tiled(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                          1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled_db (same tiling, double-buffered K-slice prefetch) ----

static void BM_gemm_fp32_square_tiled_db(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled_db(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                             1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled_db(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                 1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled_db)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled_cpasync (same tiling, cp.async-based prefetch) ----

static void BM_gemm_fp32_square_tiled_cpasync(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                  1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled_cpasync)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled_cpasync_vec (same cp.async, +float4 vectorized loads/reads) ----

static void BM_gemm_fp32_square_tiled_cpasync_vec(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled_cpasync_vec)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled_cpasync_vec2 (same, +vectorized B load) ----

static void BM_gemm_fp32_square_tiled_cpasync_vec2(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                       1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled_cpasync_vec2)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- gemm_fp32_tiled_cpasync_vec3 (same, +doubled K-tile for wider B coalescing) ----

static void BM_gemm_fp32_square_tiled_cpasync_vec3(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    cuda::experimental::gemm_fp32_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                       1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_fp32_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           1.0, N, N, N, A_d, N, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_gemm_fp32_square_tiled_cpasync_vec3)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- BM_gemm_fp32_square_cublas (cuda::experimental::mm_strided<float,float,float>'s
// cublasSgemm specialization) ----
//
// The performance ceiling every custom fp32 GEMM kernel above is being
// tuned against.

static void BM_gemm_fp32_square_cublas(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("cublasCreate failed");
        return;
    }

    DevBuf<float> A_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) {
        cublasDestroy(handle);
        return;
    }

    cuda::experimental::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {N, N}, A_d, N, {N, N}, B_d, N, 0.0, C_d, N, handle);
    if (!check_kernel_launch(state)) {
        cublasDestroy(handle);
        return;
    }

    for (auto _ : state) {
        cuda::experimental::mm_strided<float, float, float>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {N, N}, A_d, N, {N, N}, B_d, N, 0.0, C_d, N, handle);
        if (!check_kernel_launch(state)) {
            cublasDestroy(handle);
            return;
        }
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cublasDestroy(handle);
}

BENCHMARK(BM_gemm_fp32_square_cublas)
    ->Apply(SquareFp32Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

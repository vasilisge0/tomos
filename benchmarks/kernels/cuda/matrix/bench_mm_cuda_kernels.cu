#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"
#include "bench_cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <random>
#include <vector>

#include "mm_cuda_kernels.cuh"

namespace tomos {
namespace bench {

// Progressively larger A100-scale sweeps: square (M=K=N), tall (K and N
// fixed at 512, M scaling up), and wide (M and K fixed at 512, N scaling
// up). Every kernel below runs on-GPU (thousands of parallel threads even
// for the "generic" custom kernel), so unlike the CPU benchmarks a single
// ladder is fast enough to share across all of them. The top square point
// (32768^3, double A+B+C) is ~25.8GB — leaves ~14GB of headroom on a 40GB
// card for CUDA context/driver overhead; the top tall/wide point (M or N =
// 2097152 against fixed dimensions of 512) is ~17.2GB.
static void MmArgs(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768}) b->Args({n, n, n});
    for (int64_t m : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({m, 512, 512});
    for (int64_t n : {2048, 8192, 32768, 131072, 524288, 2097152}) b->Args({512, 512, n});
}

// ---- gemm_generic_kernel (direct kernel call, always the naive kernel) ----
//
// Registered before the cuBLAS benchmarks below (see BM_mm_cublas) so results
// print/save in that order: generic kernel numbers for a type combination
// appear right before its cuBLAS counterpart, making the two directly
// comparable side by side.

template <typename a_t, typename b_t, typename c_t>
static void BM_gemm_generic_kernel(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    DevBuf<a_t> A_d(M * K);
    DevBuf<b_t> B_d(K * N);
    DevBuf<c_t> C_d(M * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) return;

    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);

    cuda::experimental::gemm_generic_kernel<a_t, b_t, c_t><<<blocks, threads>>>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        1.0, M, K, N, A_d, M, B_d, K, 0.0, C_d, M);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::gemm_generic_kernel<a_t, b_t, c_t><<<blocks, threads>>>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, M, K, N, A_d, M, B_d, K, 0.0, C_d, M);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_gemm_generic_kernel, double, double, double)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_gemm_generic_kernel, float, float, float)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_gemm_generic_kernel, __half, __half, __half)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// dfd has no cuBLAS specialization (see BM_mm_cublas below), so it only ever
// runs through this kernel — benchmarked here rather than under the cuBLAS
// name so that name always means what it says.
BENCHMARK_TEMPLATE(BM_gemm_generic_kernel, double, float, double)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- BM_mm_cublas (cuda::mm_strided's cuBLAS specializations only) ----
//
// cuda::mm_strided has full template specializations for <double,double,
// double>, <float,float,float>, and <__half,__half,__half> that call
// cublasDgemm/cublasSgemm/cublasHgemm instead of its generic-template body
// (which just calls gemm_generic_kernel — see mm_cuda_kernels.cu). Only
// those three combinations are instantiated below, so every result
// printed/saved under this name is genuinely cuBLAS, never a silent
// fallback to the generic kernel.

template <typename a_t, typename b_t, typename c_t>
static void BM_mm_cublas(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("cublasCreate failed");
        return;
    }

    DevBuf<a_t> A_d(M * K);
    DevBuf<b_t> B_d(K * N);
    DevBuf<c_t> C_d(M * N);
    if (!check_alloc(state, {A_d.status(), B_d.status(), C_d.status()})) {
        cublasDestroy(handle);
        return;
    }

    cuda::experimental::mm_strided<a_t, b_t, c_t>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, A_d, M, {K, N}, B_d, K, 0.0, C_d, M, handle);
    if (!check_kernel_launch(state)) {
        cublasDestroy(handle);
        return;
    }

    for (auto _ : state) {
        cuda::experimental::mm_strided<a_t, b_t, c_t>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, A_d, M, {K, N}, B_d, K, 0.0, C_d, M, handle);
        if (!check_kernel_launch(state)) {
            cublasDestroy(handle);
            return;
        }
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * M * K * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cublasDestroy(handle);
}

BENCHMARK_TEMPLATE(BM_mm_cublas, double, double, double)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mm_cublas, __half, __half, __half)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_mm_cublas, float, float, float)
    ->Apply(MmArgs)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_csr CUDA (SpMM) ----

static void BM_mm_csr_cuda(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));
    const double density = 0.01;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::uniform_real_distribution<double> keep(0.0, 1.0);

    std::vector<double> vals_h;
    std::vector<int> rp_h(M + 1, 0), ci_h;
    for (size_t i = 0; i < M; ++i) {
        rp_h[i] = static_cast<int>(vals_h.size());
        for (size_t j = 0; j < K; ++j) {
            if (keep(gen) < density) {
                vals_h.push_back(val(gen));
                ci_h.push_back(static_cast<int>(j));
            }
        }
    }
    rp_h[M] = static_cast<int>(vals_h.size());
    size_t nnz = vals_h.size();

    std::vector<double> B_h(K * N), C_h(M * N, 0.0);
    for (auto& v : B_h) v = val(gen);

    DevBuf<double> vals_d(nnz);
    DevBuf<int> rp_d(M + 1);
    DevBuf<int> ci_d(nnz);
    DevBuf<double> B_d(K * N);
    DevBuf<double> C_d(M * N);
    if (!check_alloc(state, {vals_d.status(), rp_d.status(), ci_d.status(),
                             B_d.status(), C_d.status()})) {
        return;
    }

    cudaMemcpy(vals_d, vals_h.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rp_d, rp_h.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ci_d, ci_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, M * N * sizeof(double));

    cusparseHandle_t handle;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
        state.SkipWithError("cusparseCreate failed");
        return;
    }

    cuda::mm_csr<double, double, double, int>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
        {M, K}, nnz, vals_d, rp_d, ci_d,
        {K, N}, B_d, K, 0.0, C_d, M, handle);
    if (!check_kernel_launch(state)) {
        cusparseDestroy(handle);
        return;
    }

    for (auto _ : state) {
        cuda::mm_csr<double, double, double, int>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
            {M, K}, nnz, vals_d, rp_d, ci_d,
            {K, N}, B_d, K, 0.0, C_d, M, handle);
        if (!check_kernel_launch(state)) {
            cusparseDestroy(handle);
            return;
        }
    }

    state.counters["nnz"] = static_cast<double>(nnz);
    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * nnz * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);

    cusparseDestroy(handle);
}

BENCHMARK(BM_mm_csr_cuda)
    ->Args({1024, 1024, 64})->Args({4096, 4096, 64})->Args({16384, 1024, 64})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

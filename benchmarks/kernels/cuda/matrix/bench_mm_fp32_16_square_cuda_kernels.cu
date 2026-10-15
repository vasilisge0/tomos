#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"
#include "bench_cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_pack.hpp"
#include "mm_cuda_kernels.cuh"

namespace tomos {
namespace bench {

namespace {

// Device-side wh/wr generation (avoids a multi-GB host-generate-then-upload
// round trip at the larger square sizes below) -- same pseudo-random scheme
// used by verify_mv_fp32_16_gemv.cu's generator, duplicated here rather than
// shared since it's a few lines and each benchmark/verify file in this
// directory is self-contained.
__device__ float pseudo_rand_fp32_16(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_fp32_16(i), wh[i], wr[i]);
}

void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}

}  // namespace

// Square-only (M=K=N) fp32_16 CUDA GEMM sweep, scoped down like
// bench_mm_fp32_square_cuda_kernels.cu's plain-fp32 version, for a fast
// build+run cycle while tuning a custom fp32_16 GEMM kernel. Same square
// ladder as that file's SquareFp32Args.
static void SquareFp32_16Args(benchmark::internal::Benchmark* b) {
    for (int64_t n : {64, 256, 1024, 2048, 4096, 8192, 16384, 32768}) b->Args({n});
}

// ---- mm_fp32_16_accurate (direct kernel call) ----
//
// The naive baseline (one thread per output element, reconstructing A(i,l)
// from wh/wr via fp32_16_combine on every access -- see mm_fp32_16_
// accurate_kernel in fp32_16_cuda_kernels.cu) -- the kernel this benchmark
// exists to iteratively improve, mirroring gemm_fp32_tiled's evolution in
// bench_mm_fp32_square_cuda_kernels.cu. As tiled/vectorized fp32_16
// variants are added, add their own BENCHMARK(...) block here alongside
// this one.

static void BM_mm_fp32_16_accurate_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                  1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_accurate_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled (shared-memory-tiled: decode wh/wr once per load, not once per use) ----

static void BM_mm_fp32_16_tiled_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                           1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                               1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_vec (same, +vectorized wh/wr, B, and register loads) ----

static void BM_mm_fp32_16_tiled_vec_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                               1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                   1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_vec_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_vec (same, +cp.async prefetch for A's raw wh/wr) ----

static void BM_mm_fp32_16_tiled_cpasync_vec_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                       1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_vec_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_vec2 (same, +doubled K-tile for wider B coalescing) ----

static void BM_mm_fp32_16_tiled_cpasync_vec2_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_vec2_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_vec2_branchless (same, +branchless decode) ----

static void BM_mm_fp32_16_tiled_cpasync_vec2_branchless_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_vec2_branchless(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
        1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec2_branchless(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_vec2_branchless_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_vec3 (bigger tile, dynamic shared memory) ----

static void BM_mm_fp32_16_tiled_cpasync_vec3_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_vec3_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_vec4 (same block/grid as vec2, narrower per-thread tile) ----

static void BM_mm_fp32_16_tiled_cpasync_vec4_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_vec4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_vec4_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_warptile (vec2's tile, warp-aligned thread mapping) ----

static void BM_mm_fp32_16_tiled_cpasync_warptile_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_warptile_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_warptile2 (same, transposed warp/lane grid shape) ----

static void BM_mm_fp32_16_tiled_cpasync_warptile2_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_warptile2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile2(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_warptile2_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_warptile3 (extreme tall warp/lane grid shape) ----

static void BM_mm_fp32_16_tiled_cpasync_warptile3_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_warptile3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile3(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_warptile3_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf (both A and B prefetched via cp.async) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf2 (same, +BK doubled to 32) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf2_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf2(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf2_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_tripipe (3-stage pipeline, BM halved to fit) ----

static void BM_mm_fp32_16_tiled_cpasync_tripipe_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_tripipe_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf3 (dualbuf's tile, BK shrunk 16->8) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf3_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf3_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf4 (dualbuf's tile, BM shrunk 128->64) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf4_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf4_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf5 (dualbuf's tile, BN shrunk 128->64) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf5_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf5_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf6 (dualbuf + register double-buffering) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf6_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf6(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf6(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf6_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_splitk (dualbuf's tile, split across K) ----

static void BM_mm_fp32_16_tiled_cpasync_splitk_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_splitk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_splitk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_splitk_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- mm_fp32_16_tiled_cpasync_dualbuf7 (dualbuf, B loaded synchronously) ----

static void BM_mm_fp32_16_tiled_cpasync_dualbuf7_square(benchmark::State& state) {
    const size_t N = static_cast<size_t>(state.range(0));

    DevBuf<uint16_t> wh_d(N * N);
    DevBuf<uint16_t> wr_d(N * N);
    DevBuf<float> B_d(N * N);
    DevBuf<float> C_d(N * N);
    if (!check_alloc(state, {wh_d.status(), wr_d.status(), B_d.status(), C_d.status()}))
        return;
    gen_fp32_16(N * N, wh_d, wr_d);

    cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf7(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
    if (!check_kernel_launch(state)) return;

    for (auto _ : state) {
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf7(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {N, N}, wh_d, wr_d, N, {N, N}, B_d, N, 0.0, C_d, N);
        if (!check_kernel_launch(state)) return;
    }

    state.counters["GFlop/s"] = benchmark::Counter(
        2.0 * N * N * N * static_cast<double>(state.iterations()) / 1e9,
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mm_fp32_16_tiled_cpasync_dualbuf7_square)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- BM_mm_fp32_16_square_cublas_baseline (plain-fp32 cuBLAS Sgemm) ----
//
// The performance ceiling every custom fp32_16 GEMM kernel above is being
// tuned against -- same reference bench_mm_fp32_square_cuda_kernels.cu uses,
// operating on a fully-materialized fp32 A instead of the wh/wr split
// representation, so the gap between the two numbers is genuinely the cost
// of reconstructing fp32 from the compressed storage, not a different GEMM
// implementation entirely.

static void BM_mm_fp32_16_square_cublas_baseline(benchmark::State& state) {
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

BENCHMARK(BM_mm_fp32_16_square_cublas_baseline)
    ->Apply(SquareFp32_16Args)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

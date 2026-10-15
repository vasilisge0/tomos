// Trimmed-down sibling of benchmarks_gemm_fp32.cu, scoped to
// EXACTLY the methods scripts/plot_gemm_format_32_16_compute_32.m
// actually plots (its other series are commented out, not deleted --
// see that file's own series-construction block) -- instead of that
// file's/benchmarks_gemm_format_32_16_compute_32.sh's full ~15-method
// sweep, for fast iteration when only the plotted methods matter.
//
// Kernels compared, one process/GPU-state per run:
//   TOMOS Native: gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
//     midpoint_exact -- this codebase's current best native fp32 GEMM.
//   TOMOS FP32 / FP16: mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_
//     midpoint_exact -- the genuinely-fused (decode straight into the
//     tile pipeline) fp32_16 kernel built on midpoint_exact's tiling.
//     NOTE: whether this actually beats mm_fp32_16_tiled_cpasync_dualbuf
//     below is only confirmed at a single N=8192 data point (~3% faster,
//     from earlier this session) -- unlike the truncated pair below,
//     this pair has NOT yet been confirmed by a full-sweep run; the
//     dualbuf timing block exists specifically to get that confirmation.
//   TOMOS FP32 / FP16 (truncated): mm_fp32_16_tiled_cpasync_dualbuf_
//     truncated -- the plain, 0-spill truncated-decode reference kernel
//     (decode via fp32_16_combine_truncated: never-rounded, no round_
//     flag correction). Switched IN for this series after a full-sweep
//     head-to-head (this file itself) showed it beats the genuinely-
//     fused midpoint_exact-based truncated kernel (mm_fp32_16_
//     syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated) at
//     EVERY measured size, ~3% at N=1024 widening to ~8-10% for
//     N>=8192 -- the fused kernel compiles with a real spill here (26
//     STL + 17 LDL at 128 registers) that dualbuf_truncated doesn't
//     have. This is the OPPOSITE direction from the plain (non-
//     truncated) pair above.
//   TOMOS FP32 / FP16 (materialize): mm_fp32_16_gemm_materialize_
//     midpoint_exact -- decodes A to a scratch buffer, then calls
//     midpoint_exact as a black box. The best fp32_16 kernel of any
//     kind (beats the fused kernel above by ~15% at N=8192).
//   TOMOS FP32 / BF16: mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_
//     snake_midpoint_exact -- the fp32_bf16 counterpart to the fp32_16
//     fused kernel above (same tiling, decode via fp32_bf16_combine
//     instead of fp32_16_combine). Unlike fp32_16, where materialize
//     wins because fp32_16_combine's rounding logic is register-heavy,
//     fp32_bf16_combine is unconditional/branchless (bf16's 8-bit
//     exponent matches fp32's exactly, no degraded-range correction
//     branch) and compiles with 0 spill -- so here the fused kernel
//     wins outright over dualbuf below, by ~7-14% at every measured
//     size.
//   TOMOS FP32 / BF16 (workspace): mm_fp32_bf16_gemm_materialize_
//     rounded -- the ROUNDED-encoding bf16 format (wh formed by round-
//     to-nearest-even instead of truncation, round decision stored in
//     wr's own LSB -- see fp32_bf16_pack.hpp's fp32_bf16_split_rounded/
//     fp32_bf16_combine_rounded) decoded via the materialize strategy
//     (decode A to a scratch fp32 buffer up front, then run the native
//     GEMM as a black box). Built specifically to sidestep the fused
//     rounded kernel's own real spill (REG:128 STACK:56, 23 STL + 11
//     LDL -- investigated at length and not resolved), since
//     materialize's decode pass is a simple per-element kernel with no
//     GEMM tiling register pressure at all.
//   Plus mm_fp32_16_tiled_cpasync_dualbuf -- the older plain fused
//     reference this session's "TOMOS FP32 / FP16" series has NOT yet
//     been switched away from (see the NOTE above); tracked head-to-
//     head against the fused midpoint_exact kernel via the ratio print
//     below, extra raw data until that comparison is confirmed.
//   Plus mm_fp32_bf16_tiled_cpasync_dualbuf -- the older bf16 kernel
//     this series was switched out from above; tracked head-to-head
//     against the fused kernel (see the ratio print below), extra raw
//     data, not one of plot_gemm_format_32_16_compute_32.m's own
//     plotted series.
//   Plus mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_
//     exact_truncated -- the fused truncated kernel this session's
//     "TOMOS FP32 / FP16 (truncated)" series was switched OUT of, per
//     the NOTE above; tracked head-to-head against dualbuf_truncated,
//     extra raw data.
//   Plus real CUTLASS (fp32 SIMT device::Gemm) and cuBLAS Sgemm.
//
// CSV output uses the EXACT same column names scripts/benchmarks_gemm_
// format_32_16_compute_32.sh's own embedded parser produces for these
// eight methods (native_ms/gflops, fp32_16_fused_midpoint_exact_ms/
// gflops, fp32_16_dualbuf_trunc_ms/gflops, fp32_16_gemm_materialize_
// midpoint_exact_ms/gflops, fp32_bf16_fused_midpoint_exact_ms/gflops,
// fp32_bf16_materialize_rounded_ms/gflops, cutlass_ms/gflops, cublas_ms/
// gflops) -- so scripts/plot_gemm_format_32_16_compute_32.m can plot
// this file's own CSV output directly with ZERO changes, since every
// series it actually reads (the others are commented out in its own
// series-construction block) is present here under the identical column
// name.
//
// Usage:
//   fp32_16_plotted_vs_cutlass_ref
//       Sweeps the SAME square N list (1024/1536/2048/3072/4096/6144/
//       8192/12288/16384/24576/32768/49152) and rectangular K sweep at
//       M=N=49152 (K=1024..32768, same list minus 49152) as scripts/
//       benchmarks_gemm_format_32_16_compute_32.sh, so results are
//       directly comparable size-for-size.
//   fp32_16_plotted_vs_cutlass_ref N
//       Just that one square shape (M=K=N).
//   fp32_16_plotted_vs_cutlass_ref --rect K [outer]
//       Just that one rectangular shape (M=N=outer, default 49152).
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call the
// tomos kernels directly. See scripts/fp32_16_plotted_vs_cutlass_ref.sh.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
void mm_fp32_bf16_gemm_materialize_rounded(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

using ElementA = float;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = float;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;

// Same CUTLASS config as benchmarks_gemm_fp32.cu -- see that file's
// own header comment for why this uses CUTLASS's own per-arch defaults
// rather than a hand-picked tile/warp/instruction shape.
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80
>;

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_float_kernel(size_t n, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand(i);
}
void gen_float(size_t n, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out);
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}

__global__ void gen_fp32_16_truncated_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_truncated(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16_truncated(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_truncated_kernel<<<blocks, threads>>>(n, wh, wr);
}

__global__ void gen_fp32_bf16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_kernel<<<blocks, threads>>>(n, wh, wr);
}

__global__ void gen_fp32_bf16_rounded_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split_rounded(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16_rounded(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_rounded_kernel<<<blocks, threads>>>(n, wh, wr);
}

}  // namespace

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / reps;
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). The square sweep calls this
// with m=k=n; the rectangular sweep holds m=n fixed and varies k.
void run_shape(size_t m, size_t k, size_t n, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);

    float *B, *C;
    if (!alloc_or_skip(reinterpret_cast<void**>(&B), k * n * sizeof(float), "B", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&C), m * n * sizeof(float), "C", shape_desc)) {
        std::printf("  %-24s SKIPPED (out of device memory for B/C alone)\n", shape_desc);
        cudaFree(B);
        cudaFree(C);
        return;
    }
    gen_float(k * n, B);
    cudaMemset(C, 0, m * n * sizeof(float));
    cudaDeviceSynchronize();

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    size_t maxdim = std::max({m, k, n});
    int reps = (maxdim <= 2048) ? 20 : (maxdim <= 8192 ? 5 : 2);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    std::printf("  %s\n", shape_desc);

    double ms_16fused = -1.0;
    {
        uint16_t *wh16f = nullptr, *wr16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16f, wr16f);
            cudaDeviceSynchronize();
            ms_16fused = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16f, wr16f, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16fused, flops / ms_16fused / 1e6);
        }
        cudaFree(wh16f);
        cudaFree(wr16f);
    }

    // Original fused-decode reference kernel (plain, non-truncated) --
    // direct head-to-head with the fused midpoint_exact-based kernel
    // above (see the ratio print below) to confirm which is actually
    // faster across the full sweep. The only prior data point (a single
    // N=8192 measurement from earlier this session) showed the fused
    // kernel winning by ~3%, but that was never confirmed across sizes
    // the way the truncated pair below now has been -- this closes that
    // gap.
    double ms_16dualbuf = -1.0;
    {
        uint16_t *wh16d = nullptr, *wr16d = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16d), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16d), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16d, wr16d);
            cudaDeviceSynchronize();
            ms_16dualbuf = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16d, wr16d, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16dualbuf, flops / ms_16dualbuf / 1e6);
        }
        cudaFree(wh16d);
        cudaFree(wr16d);
    }

    double ms_16fused_trunc = -1.0;
    {
        uint16_t *wh16ft = nullptr, *wr16ft = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16ft), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16ft), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated (wr)", shape_desc)) {
            gen_fp32_16_truncated(m * k, wh16ft, wr16ft);
            cudaDeviceSynchronize();
            ms_16fused_trunc = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16ft, wr16ft, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16fused_trunc, flops / ms_16fused_trunc / 1e6);
        }
        cudaFree(wh16ft);
        cudaFree(wr16ft);
    }

    // Original fused-decode truncated reference kernel -- direct
    // head-to-head with the fused midpoint_exact-based truncated kernel
    // above (see the ratio print below) to confirm which is actually
    // faster, since the midpoint_exact one compiles with a real spill
    // (26 STL + 17 LDL at 128 registers) that the plain dualbuf one
    // doesn't have.
    double ms_16dualbuf_trunc = -1.0;
    {
        uint16_t *wh16dt = nullptr, *wr16dt = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16dt), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_truncated (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16dt), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_truncated (wr)", shape_desc)) {
            gen_fp32_16_truncated(m * k, wh16dt, wr16dt);
            cudaDeviceSynchronize();
            ms_16dualbuf_trunc = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16dt, wr16dt, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_truncated = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16dualbuf_trunc, flops / ms_16dualbuf_trunc / 1e6);
        }
        cudaFree(wh16dt);
        cudaFree(wr16dt);
    }

    double ms_16materialize = -1.0;
    {
        uint16_t *wh16m = nullptr, *wr16m = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16m), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16m), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16m, wr16m);
            cudaDeviceSynchronize();
            ms_16materialize = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16m, wr16m, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16materialize, flops / ms_16materialize / 1e6);
        }
        cudaFree(wh16m);
        cudaFree(wr16m);
    }

    double ms_bf16dualbuf = -1.0;
    {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_tiled_cpasync_dualbuf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_tiled_cpasync_dualbuf (wr)", shape_desc)) {
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16);
            cudaDeviceSynchronize();
            ms_bf16dualbuf = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16, wr_bf16, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_tiled_cpasync_dualbuf = %9.3fms (%10.1f GFlop/s)\n", ms_bf16dualbuf,
                       flops / ms_bf16dualbuf / 1e6);
        }
        cudaFree(wh_bf16);
        cudaFree(wr_bf16);
    }

    double ms_bf16fused = -1.0;
    {
        uint16_t *wh_bf16f = nullptr, *wr_bf16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16f), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16f), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wr)", shape_desc)) {
            gen_fp32_bf16(m * k, wh_bf16f, wr_bf16f);
            cudaDeviceSynchronize();
            ms_bf16fused = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16f, wr_bf16f, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n", ms_bf16fused,
                       flops / ms_bf16fused / 1e6);
        }
        cudaFree(wh_bf16f);
        cudaFree(wr_bf16f);
    }

    // TOMOS FP32 / BF16 (workspace): the ROUNDED-encoding bf16 format
    // (wh formed by round-to-nearest-even instead of truncation, round
    // decision stored in wr's own LSB -- see fp32_bf16_pack.hpp's
    // fp32_bf16_split_rounded/fp32_bf16_combine_rounded) decoded via the
    // materialize strategy (decode A to a scratch fp32 buffer up front,
    // then run the native GEMM as a black box) -- sidesteps the fused
    // rounded kernel's own real spill (REG:128 STACK:56, 23 STL + 11
    // LDL) entirely, since materialize's decode pass is a simple per-
    // element kernel with no GEMM tiling register pressure.
    double ms_bf16materialize_rounded = -1.0;
    {
        uint16_t *wh_bf16rm = nullptr, *wr_bf16rm = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16rm), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_gemm_materialize_rounded (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16rm), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_gemm_materialize_rounded (wr)", shape_desc)) {
            gen_fp32_bf16_rounded(m * k, wh_bf16rm, wr_bf16rm);
            cudaDeviceSynchronize();
            ms_bf16materialize_rounded = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_gemm_materialize_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16rm, wr_bf16rm, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_gemm_materialize_rounded = %9.3fms (%10.1f GFlop/s)\n", ms_bf16materialize_rounded,
                       flops / ms_bf16materialize_rounded / 1e6);
        }
        cudaFree(wh_bf16rm);
        cudaFree(wr_bf16rm);
    }

    double ms_native = -1.0, ms_cutlass = -1.0, ms_cublas = -1.0;
    {
        float* A_native = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "native/cutlass/cublas (A)",
                          shape_desc)) {
            gen_float(m * k, A_native);
            cudaDeviceSynchronize();

            ms_native = time_call(start, stop, [&]() {
                cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native) = %9.3fms (%10.1f GFlop/s)\n", ms_native,
                       flops / ms_native / 1e6);

            cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            ElementAccumulator alpha = 1.7f, beta = 0.3f;
            Gemm gemm_op;
            typename Gemm::Arguments args(problem_size, {A_native, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
            size_t workspace_size = Gemm::get_workspace_size(args);
            void* workspace = nullptr;
            if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

            cutlass::Status status = gemm_op.can_implement(args);
            if (status == cutlass::Status::kSuccess) status = gemm_op.initialize(args, workspace);
            if (status == cutlass::Status::kSuccess) {
                ms_cutlass = time_call(start, stop, [&]() { gemm_op(); }, reps);
                std::printf("    cutlass (real, SIMT fp32)          = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass,
                           flops / ms_cutlass / 1e6);
            } else {
                std::printf("    cutlass SKIPPED (%s)\n", cutlass::cutlassGetStatusString(status));
            }
            if (workspace) cudaFree(workspace);

            ms_cublas = time_call(start, stop, [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), &alpha, A_native, lda, B, ldb, &beta, C, ldc);
            }, reps);
            std::printf("    cublas (Sgemm)                     = %9.3fms (%10.1f GFlop/s)\n", ms_cublas,
                       flops / ms_cublas / 1e6);
        }
        cudaFree(A_native);
    }

    if (ms_16fused > 0 && ms_16fused_trunc > 0) {
        std::printf("    -- fused vs fused_truncated: %.3fx\n", (flops / ms_16fused_trunc) / (flops / ms_16fused));
    }
    if (ms_16dualbuf > 0 && ms_16fused > 0) {
        // Direct head-to-head between the old plain fused reference and
        // the new midpoint_exact-based fused kernel -- see the comment
        // on ms_16dualbuf's own timing block above for why this needed
        // a full-sweep confirmation beyond the one prior N=8192 point.
        std::printf("    -- dualbuf vs fused: %.3fx\n", (flops / ms_16fused) / (flops / ms_16dualbuf));
    }
    if (ms_16dualbuf > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (dualbuf)\n", (flops / ms_16dualbuf) / (flops / ms_cutlass));
    }
    if (ms_16dualbuf_trunc > 0 && ms_16fused_trunc > 0) {
        // The actual question this whole block exists to answer: does
        // the fused midpoint_exact-based truncated kernel's real spill
        // cost it the win against the plain, spill-free dualbuf
        // truncated reference?
        std::printf("    -- dualbuf_truncated vs fused_truncated: %.3fx\n",
                   (flops / ms_16fused_trunc) / (flops / ms_16dualbuf_trunc));
    }
    if (ms_16dualbuf_trunc > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (dualbuf_truncated)\n", (flops / ms_16dualbuf_trunc) / (flops / ms_cutlass));
    }
    if (ms_16fused > 0 && ms_16materialize > 0) {
        std::printf("    -- fused vs materialize: %.3fx\n", (flops / ms_16materialize) / (flops / ms_16fused));
    }
    if (ms_native > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (native)\n", (flops / ms_native) / (flops / ms_cutlass));
    }
    if (ms_16materialize > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (materialize)\n", (flops / ms_16materialize) / (flops / ms_cutlass));
    }
    if (ms_bf16dualbuf > 0 && ms_bf16fused > 0) {
        std::printf("    -- bf16dualbuf vs bf16fused: %.3fx\n", (flops / ms_bf16fused) / (flops / ms_bf16dualbuf));
    }
    if (ms_bf16fused > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16fused)\n", (flops / ms_bf16fused) / (flops / ms_cutlass));
    }
    if (ms_bf16fused > 0 && ms_bf16materialize_rounded > 0) {
        std::printf("    -- bf16fused vs bf16materialize_rounded: %.3fx\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_bf16fused));
    }
    if (ms_bf16materialize_rounded > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16materialize_rounded)\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_cutlass));
    }

    cudaFree(B);
    cudaFree(C);
}

int main(int argc, char** argv) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== Plotted methods only: TOMOS Native / FP32-FP16 / FP32-FP16 (truncated) / "
               "FP32-FP16 (workspace) / FP32-BF16 / FP32-BF16 (workspace) vs real CUTLASS (fp32 SIMT) vs cuBLAS Sgemm ===\n");

    // Same sweep sizes as scripts/benchmarks_gemm_format_32_16_compute_
    // 32.sh's own SQUARE_NS/RECT_OUTER/RECT_KS, so results (and the CSV
    // this produces) are directly comparable to -- and plottable by the
    // same plot_gemm_format_32_16_compute_32.m as -- that script's own
    // output.
    const std::vector<size_t> square_ns = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152};
    const std::vector<size_t> rect_ks = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768};
    const size_t kDefaultRectOuter = 49152;

    if (argc > 1 && std::string(argv[1]) == "--rect") {
        size_t outer = (argc > 3) ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) : kDefaultRectOuter;
        if (argc > 2) {
            size_t k = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
            run_shape(outer, k, outer, start, stop, handle);
        } else {
            for (size_t k : rect_ks) run_shape(outer, k, outer, start, stop, handle);
        }
        cublasDestroy(handle);
        return 0;
    }

    if (argc > 1) {
        size_t n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
        run_shape(n, n, n, start, stop, handle);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t n : square_ns) run_shape(n, n, n, start, stop, handle);
    for (size_t k : rect_ks) run_shape(kDefaultRectOuter, k, kDefaultRectOuter, start, stop, handle);

    cublasDestroy(handle);
    return 0;
}

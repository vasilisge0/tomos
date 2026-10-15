// Trimmed-down sibling of benchmarks_gemm_fp32.cu: times ONLY the
// kernels added by the warptile/snake/midpoint_exact investigation (and
// their immediate baselines for context), instead of that file's full
// ~40-kernel sweep -- for fast iteration when only these are relevant.
// Same binary layout, same CUTLASS config, same output/log conventions;
// see benchmarks_gemm_fp32.cu's own header comment for the full
// investigation writeup and scripts/fp32_recon_vs_cutlass_ref.sh for the
// exhaustive version.
//
// Kernels compared, one process/GPU-state per run:
//   Native fp32:
//     gemm_fp32_tiled_syncpipe3 -- the original "native" baseline
//       (0.951x vs CUTLASS at N=8192), pre-warptile/snake investigation.
//     gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake -- first
//       kernel to beat syncpipe3 (0.960x), via CUTLASS's real warp/lane
//       geometry + BK=8 + 4-stage pipeline + rolling-pointer addressing
//       + snake FFMA traversal, before the interleaved-checkpoint work.
//     gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_
//       exact -- current best (0.977x at N=8192, 0.982x at N=32768,
//       beats cuBLAS there): interleaves the prefetch into the FFMA row
//       loop at i==4, with bounds-check predicates dropped entirely
//       (host-side exact-multiple dispatch guard, falls back otherwise).
//   Packed fp32_16:
//     mm_fp32_16_tiled_cpasync_dualbuf -- the original fused-decode
//       reference kernel (BK=16, 2-stage, plain row-major lane split).
//     mm_fp32_16_gemm_materialize -- previously the fastest fp32_16
//       kernel: decodes A to a scratch fp32 buffer, then calls
//       gemm_fp32_tiled_cpasync_dualbuf as a black box.
//     mm_fp32_16_gemm_materialize_midpoint_exact -- same materialize
//       structure, second-stage call swapped to midpoint_exact.
//     mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact --
//       genuinely fused counterpart: midpoint_exact's entire tiling
//       structure ported into this file, wh/wr decoded straight into
//       the tile pipeline's shared-memory A buffer.
//
// Usage: identical to fp32_recon_vs_cutlass_ref (see that file's own
// usage comment) -- no args sweeps square N=1024..54000 then a
// rectangular K-sweep at M=N=44000; a bare N runs just that square
// shape; --rect [K] [outer] runs just the rectangular sweep or shape.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call the
// tomos kernels directly. See scripts/fp32_new_kernels_vs_cutlass_ref.sh.

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
#include "transpose.hpp"

namespace tomos {
namespace cuda {
namespace experimental { void gemm_fp32_tiled_syncpipe3(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                      double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_gemm_materialize(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

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

    double ms_16dualbuf = -1.0;
    {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16, wr16);
            cudaDeviceSynchronize();
            ms_16dualbuf = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                       dim<2>{m, k}, wh16, wr16, lda, dim<2>{k, n}, B, ldb, 0.3, C,
                                                       ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf   = %9.3fms (%10.1f GFlop/s)\n", ms_16dualbuf,
                       flops / ms_16dualbuf / 1e6);
        }
        cudaFree(wh16);
        cudaFree(wr16);
    }

    double ms_16gemm_materialize = -1.0;
    {
        uint16_t *wh16g = nullptr, *wr16g = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16g), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16g), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16g, wr16g);
            cudaDeviceSynchronize();
            ms_16gemm_materialize = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_gemm_materialize(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16g, wr16g, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize, flops / ms_16gemm_materialize / 1e6);
        }
        cudaFree(wh16g);
        cudaFree(wr16g);
    }

    double ms_16gemm_materialize_midpoint_exact = -1.0;
    {
        uint16_t *wh16gme = nullptr, *wr16gme = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16gme), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16gme), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16gme, wr16gme);
            cudaDeviceSynchronize();
            ms_16gemm_materialize_midpoint_exact = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16gme, wr16gme, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize_midpoint_exact, flops / ms_16gemm_materialize_midpoint_exact / 1e6);
        }
        cudaFree(wh16gme);
        cudaFree(wr16gme);
    }

    double ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = -1.0;
    {
        uint16_t *wh16f = nullptr, *wr16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wr)", shape_desc)) {
            gen_fp32_16(m * k, wh16f, wr16f);
            cudaDeviceSynchronize();
            ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16f, wr16f, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact, flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact / 1e6);
        }
        cudaFree(wh16f);
        cudaFree(wr16f);
    }

    double ms_syncpipe3 = -1.0, ms_wtnsnake = -1.0, ms_native = -1.0, ms_cutlass = -1.0, ms_cublas = -1.0;
    {
        float* A_native = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "native/cutlass/cublas (A)",
                          shape_desc)) {
            gen_float(m * k, A_native);
            cudaDeviceSynchronize();

            ms_syncpipe3 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3 = %9.3fms (%10.1f GFlop/s)\n", ms_syncpipe3,
                       flops / ms_syncpipe3 / 1e6);

            ms_wtnsnake = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake = %9.3fms (%10.1f GFlop/s)\n", ms_wtnsnake,
                       flops / ms_wtnsnake / 1e6);

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

    // Native fp32 family ratios.
    if (ms_syncpipe3 > 0 && ms_wtnsnake > 0) {
        std::printf("    -- syncpipe3 vs wtnsnake: %.3fx\n", (flops / ms_wtnsnake) / (flops / ms_syncpipe3));
    }
    if (ms_wtnsnake > 0 && ms_native > 0) {
        std::printf("    -- wtnsnake vs native (midpoint_exact): %.3fx\n", (flops / ms_native) / (flops / ms_wtnsnake));
    }
    if (ms_syncpipe3 > 0 && ms_native > 0) {
        std::printf("    -- syncpipe3 vs native (midpoint_exact): %.3fx\n", (flops / ms_native) / (flops / ms_syncpipe3));
    }
    if (ms_native > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (native)\n", (flops / ms_native) / (flops / ms_cutlass));
    }
    if (ms_native > 0 && ms_cublas > 0) {
        std::printf("    -- vs cublas: %.3fx (native)\n", (flops / ms_native) / (flops / ms_cublas));
    }

    // Packed fp32_16 family ratios.
    if (ms_16dualbuf > 0 && ms_16gemm_materialize > 0) {
        std::printf("    -- 16dualbuf vs 16gemm_materialize: %.3fx\n",
                   (flops / ms_16gemm_materialize) / (flops / ms_16dualbuf));
    }
    if (ms_16gemm_materialize > 0 && ms_16gemm_materialize_midpoint_exact > 0) {
        std::printf("    -- 16gemm_materialize vs 16gemm_materialize_midpoint_exact: %.3fx\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_16gemm_materialize));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0) {
        std::printf("    -- 16gemm_materialize_midpoint_exact vs 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_16gemm_materialize_midpoint_exact));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_native > 0) {
        std::printf("    -- 16gemm_materialize_midpoint_exact vs native: %.3fx\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_native));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_native > 0) {
        std::printf("    -- 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact vs native: %.3fx\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_native));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16gemm_materialize_midpoint_exact)\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_cutlass));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_cutlass));
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

    std::printf("=== New kernels only: gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact "
               "(native) and its fp32_16 counterparts vs syncpipe3/wtnsnake baselines, real CUTLASS (fp32 SIMT), "
               "and cuBLAS Sgemm ===\n");

    // Same sweep sizes/argument handling as benchmarks_gemm_fp32.cu
    // (see that file's own usage comment) -- kept identical so results
    // are directly comparable size-for-size.
    const std::vector<size_t> rect_ks = {1024, 2048, 4096, 8192, 16384, 32768};
    const size_t kDefaultRectOuter = 44000;

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

    for (size_t n : {1024, 2048, 4096, 8192, 16384, 32768, 48000, 54000}) {
        run_shape(n, n, n, start, stop, handle);
    }

    for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle);

    cublasDestroy(handle);
    return 0;
}

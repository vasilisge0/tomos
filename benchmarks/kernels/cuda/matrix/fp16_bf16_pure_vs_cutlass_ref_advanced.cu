// "_advanced" variant of benchmarks_gemm_fp16.cu -- byte-for-
// byte identical except for time_call's own timing methodology and two
// new CLI flags, --warmup N and --iters N (both default 10, see main()'s
// own argv-filtering preamble): instead of this file's own sibling's
// fixed, shape-size-dependent `reps` heuristic (a single un-timed
// warmup call, then `reps` timed calls averaged), this variant always
// does g_warmup_runs UN-TIMED warmup calls first, then times g_
// iterations calls as one block and divides by g_iterations -- both
// counts fully caller-controlled, independent of shape. No kernel call
// site, shape list, or dispatch logic differs from the sibling file.
//
// Standalone throughput comparison for the "pure/composite" (tensor-core
// compute, decode-only) tier of both packed-precision families --
// mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_
// precomp2_serp (src/kernels/cuda/matrix/mm_cuda_kernels.cu -- promoted
// here (2026-08-27) as this family's new best after real hardware
// measurement showed it beating plain inline_dirty by +13.1% at
// N=32768 up to +23.1% at N=2048, even slightly beating native fp16
// itself at N=4096/16384/32768 -- see that kernel's own comment in
// mm_cuda_kernels.cu for the full derivation) and mm_
// fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp (same file,
// promoted here (2026-08-28) -- see that kernel's own comment in mm_
// cuda_kernels.cu for why it can call the native bf16 kernel directly
// on wh reinterpreted, with no decode step at all) -- against this
// codebase's own best native (unpacked) fp16/bf16 GEMM (gemm_fp16_mma_
// ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp,
// gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_
// precomp2_serp), real CUTLASS's fp16/bf16
// tensor-op device::Gemm, and cuBLAS's Hgemm/GemmEx(CUDA_R_16BF).
// Mirrors benchmarks_gemm_fp32.cu's structure and staggered-
// allocation memory discipline, just for the pure/composite tier
// instead of the full-reconstruction tier.
//
// The fp16 "native" leg calls gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
// iter_epivec_acongruous_fastk_precomp2_serp, promoted here (2026-08-27)
// as this family's new best -- acongruous's own precomputed address
// tables (fastk_precomp2) plus real CUTLASS's own serpentine
// (boustrophedon) accumulate order, both validated on real hardware --
// see that kernel's own comment in mm_cuda_kernels.cu for the full
// derivation chain. Like iter_epivec before it, it requires k to be an
// exact multiple of its BK tile and falls back internally to the always-
// correct base acongruous kernel otherwise, so every shape in this
// file's sweeps (including the non-256-multiple M=N=44000 rect sweep)
// still runs correctly -- just not always on the fastest internal path.
// Plain deep4 and iter_epivec themselves are untouched and still
// available under those names.
//
// The bf16 "native" leg calls gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_
// iter_epivec_acongruous_fastk_precomp2_serp, promoted here (2026-08-28)
// as this family's new best -- a straight, mechanical port of the fp16
// leg's own refinement stack (same tile-geometry constants and index-
// math helpers, just __half -> __nv_bfloat16 and the mma.sync PTX
// instruction's f16.f16 -> bf16.bf16 suffix), MEASURED 1.17x-1.34x
// faster than plain deep4 (this family's prior only bf16 kernel) at
// every size N>=2048 -- see that kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. Requires m/n to be exact multiples
// of its own BigBM/SqBN tile (not just k, unlike the fp16 leg above,
// since no bf16 equivalent of iter_epivec's own boundary-safe
// intermediate fallback exists yet) and falls back to plain deep4
// otherwise, so every shape in this file's sweeps still runs correctly.
//
// Correctness for every kernel compared here is already covered
// elsewhere -- verify_mm_fp32_16_gemm.cu / verify_bf16_wide_bigreg_sq_
// deep4.cu for the packed kernels, verify_mm_fp16_gemm.cu for gemm_
// fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_
// precomp2_serp -- so this file is throughput-only, same convention
// every *_vs_cutlass_ref.cu file uses.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call the
// tomos kernels directly. See scripts/fp16_bf16_pure_vs_cutlass_ref.sh.
//
// The fp16 and bf16 comparisons are two independent groups within
// run_shape, each with its own B/C buffers (different element type) and
// its own transient wh/decoded-A buffers, freed before the next group
// starts -- so peak footprint is whichever single group's own peak is
// (never both groups' buffers alive at once), matching fp32_recon_vs_
// cutlass_ref.cu's staggered alloc/dealloc reasoning. At 2 bytes/element
// per buffer this is a much smaller footprint than the fp32
// reconstruction tier's, so no shape in the default sweep is expected to
// hit a real 40GB A100's memory ceiling.
//
// Usage:
//   fp16_bf16_pure_vs_cutlass_ref
//       Sweeps two families of shapes: square N=1024/2048/4096/8192/
//       16384/32768/65536 (M=K=N), then a rectangular sweep with the
//       output held fixed at M=N=44000 and the inner/contraction
//       dimension K swept across 1024/2048/4096/8192/16384/32768 (2^10
//       through 2^15) -- stress-testing how each kernel's performance
//       scales with K independent of output size, since every kernel
//       here was originally tuned assuming M=K=N.
//   fp16_bf16_pure_vs_cutlass_ref 8192
//       Just that one square size (M=K=N=8192).
//   fp16_bf16_pure_vs_cutlass_ref --rect
//       Just the rectangular K-sweep (M=N=44000, K=1024..32768), skipping
//       the square sweep entirely.
//   fp16_bf16_pure_vs_cutlass_ref --rect 8192
//       Just the rectangular shape M=N=44000, K=8192.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
void mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, const uint16_t* wh, size_t ld_a,
                                                                           const __half* B, size_t ld_b, double beta,
                                                                           __half* C, size_t ld_c);
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __half* A, size_t ld_a,
    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
void mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(TransposeType transA, TransposeType transB, double alpha,
                                                              size_t m, size_t k, size_t n, const uint16_t* wh,
                                                              size_t ld_a, const __nv_bfloat16* B, size_t ld_b,
                                                              double beta, __nv_bfloat16* C, size_t ld_c);
void gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __nv_bfloat16* A,
    size_t ld_a, __nv_bfloat16* B, size_t ld_b, double beta, __nv_bfloat16* C, size_t ld_c);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

// Same tile/warp/instruction shape for both configs -- CUTLASS supports
// bf16 tensor-core MMA at the same m16n8k16 granularity as fp16 on sm_80,
// so only the element types change between the two.
namespace cutlass_fp16_cfg {
using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;
}  // namespace cutlass_fp16_cfg

namespace cutlass_bf16_cfg {
using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;
}  // namespace cutlass_bf16_cfg

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
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

__global__ void wh_to_half_kernel(size_t n, const uint16_t* wh, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_16_half_only(wh[i]);
}
void wh_to_half(size_t n, const uint16_t* wh, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    wh_to_half_kernel<<<blocks, threads>>>(n, wh, out);
}

__global__ void wh_to_bf16_kernel(size_t n, const uint16_t* wh, __nv_bfloat16* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_bf16_bf16_only(wh[i]);
}
void wh_to_bf16(size_t n, const uint16_t* wh, __nv_bfloat16* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    wh_to_bf16_kernel<<<blocks, threads>>>(n, wh, out);
}

__global__ void gen_half_kernel(size_t n, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(pseudo_rand(i));
}
void gen_half(size_t n, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out);
}

__global__ void gen_bf16_kernel(size_t n, __nv_bfloat16* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(pseudo_rand(i));
}
void gen_bf16(size_t n, __nv_bfloat16* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_bf16_kernel<<<blocks, threads>>>(n, out);
}

}  // namespace

// Caller-controlled in place of every call site's own `reps` argument
// (still accepted, for source compatibility with the sibling file's own
// call sites, but ignored -- see main()'s own argv-filtering preamble
// for where these are actually set, from --warmup/--iters).
int g_warmup_runs = 10;
int g_iterations = 10;

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int /*reps*/) {
    for (int w = 0; w < g_warmup_runs; ++w) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < g_iterations; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / g_iterations;
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-58s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). The square sweep calls this
// with m=k=n; the rectangular sweep holds m=n=44000 fixed and varies k.
void run_shape(size_t m, size_t k, size_t n, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    size_t maxdim = std::max({m, k, n});
    int reps = (maxdim <= 2048) ? 20 : (maxdim <= 8192 ? 5 : 2);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    std::printf("  %s\n", shape_desc);

    // -- fp16 group: mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_
    // acongruous_fastk_precomp2_serp vs native/cutlass/cublas, all
    // sharing one __half B/C pair. wh/wr are transient (wr freed right
    // after generation, wh freed right after being decoded into A_half
    // for the native/cutlass/cublas leg).
    double ms_pure_half = -1.0, ms_native_fp16 = -1.0, ms_cutlass_fp16 = -1.0, ms_cublas_fp16 = -1.0;
    {
        __half *Bh = nullptr, *Ch = nullptr, *A_half = nullptr;
        uint16_t *wh16 = nullptr, *wr16 = nullptr;

        bool ok = alloc_or_skip(reinterpret_cast<void**>(&Bh), k * n * sizeof(__half), "fp16 group (B)", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&Ch), m * n * sizeof(__half), "fp16 group (C)", shape_desc);
        if (ok) {
            gen_half(k * n, Bh);
            cudaMemset(Ch, 0, m * n * sizeof(__half));
            cudaDeviceSynchronize();
        }

        ok = ok &&
             alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp (wh)",
                          shape_desc) &&
             alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp (wr)",
                          shape_desc);
        if (ok) {
            gen_fp32_16(m * k, wh16, wr16);
            cudaDeviceSynchronize();
            cudaFree(wr16);
            wr16 = nullptr;

            ms_pure_half = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, wh16, lda, Bh, ldb, 0.3,
                    Ch, ldc);
            }, reps);
            std::printf(
                "    mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp = "
                "%9.3fms (%10.1f GFlop/s)\n",
                ms_pure_half, flops / ms_pure_half / 1e6);

            // "Workspace" leg -- decodes wh into a scratch buffer ahead
            // of time (see mm_fp32_16_pure_half_gemm_materialize_cached_
            // vec_fastk_precomp2_serp's own comment in mm_cuda_
            // kernels.cu), then calls the native fp16 kernel completely
            // unmodified. Same wh16/Bh/Ch buffers as the leg above -- no
            // extra allocation needed here, the kernel's own internal
            // scratch buffer is separate and chunk-bounded. Uses the
            // CACHED+VECTORIZED variant specifically: the plain kernel's
            // own cudaMalloc/cudaFree-every-call cost dominates its
            // small-N numbers (not representative of the materialize
            // strategy's real performance, exactly the kind of setup
            // cost this file's own precompute legs elsewhere are careful
            // not to conflate with steady-state throughput either), and
            // the vectorized decode measured strictly better than the
            // scalar-decode cached kernel at every size on top of that.
            double ms_pure_half_workspace = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, wh16, lda, Bh, ldb, 0.3,
                    Ch, ldc);
            }, reps);
            std::printf(
                "    mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp = "
                "%9.3fms (%10.1f GFlop/s)\n",
                ms_pure_half_workspace, flops / ms_pure_half_workspace / 1e6);
        }

        ok = ok && alloc_or_skip(reinterpret_cast<void**>(&A_half), m * k * sizeof(__half),
                                 "native/cutlass/cublas fp16 (A)", shape_desc);
        if (ok) {
            wh_to_half(m * k, wh16, A_half);
            cudaDeviceSynchronize();
            cudaFree(wh16);
            wh16 = nullptr;

            ms_native_fp16 = time_call(start, stop, [&]() {
                cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, A_half, lda, Bh, ldb, 0.3,
                    Ch, ldc);
            }, reps);
            std::printf(
                "    gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp (native) = "
                "%9.3fms (%10.1f GFlop/s)\n",
                ms_native_fp16, flops / ms_native_fp16 / 1e6);

            cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            cutlass::half_t* A_ct = reinterpret_cast<cutlass::half_t*>(A_half);
            cutlass::half_t* B_ct = reinterpret_cast<cutlass::half_t*>(Bh);
            cutlass::half_t* C_ct = reinterpret_cast<cutlass::half_t*>(Ch);
            float alpha_f = 1.7f, beta_f = 0.3f;
            cutlass_fp16_cfg::Gemm gemm_op_fp16;
            typename cutlass_fp16_cfg::Gemm::Arguments args_fp16(problem_size, {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc},
                                                                  {C_ct, ldc}, {alpha_f, beta_f});
            size_t workspace_size_fp16 = cutlass_fp16_cfg::Gemm::get_workspace_size(args_fp16);
            void* workspace_fp16 = nullptr;
            if (workspace_size_fp16 > 0) cudaMalloc(&workspace_fp16, workspace_size_fp16);

            cutlass::Status status_fp16 = gemm_op_fp16.can_implement(args_fp16);
            if (status_fp16 == cutlass::Status::kSuccess) status_fp16 = gemm_op_fp16.initialize(args_fp16, workspace_fp16);
            if (status_fp16 == cutlass::Status::kSuccess) {
                ms_cutlass_fp16 = time_call(start, stop, [&]() { gemm_op_fp16(); }, reps);
                std::printf("    cutlass (real, fp16 tensor-op)                            = %9.3fms (%10.1f GFlop/s)\n",
                           ms_cutlass_fp16, flops / ms_cutlass_fp16 / 1e6);
            } else {
                std::printf("    cutlass fp16 SKIPPED (%s)\n", cutlass::cutlassGetStatusString(status_fp16));
            }
            if (workspace_fp16) cudaFree(workspace_fp16);

            __half alpha_h = __float2half(1.7f), beta_h = __float2half(0.3f);
            ms_cublas_fp16 = time_call(start, stop, [&]() {
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), &alpha_h, A_half, lda, Bh, ldb, &beta_h, Ch, ldc);
            }, reps);
            std::printf("    cublas (Hgemm)                                            = %9.3fms (%10.1f GFlop/s)\n",
                       ms_cublas_fp16, flops / ms_cublas_fp16 / 1e6);
        }

        cudaFree(A_half);
        cudaFree(wh16);
        cudaFree(wr16);
        cudaFree(Bh);
        cudaFree(Ch);
    }

    // -- bf16 group: mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_
    // serp vs native/cutlass/cublas, same staggered-allocation shape as
    // the fp16 group above (never alive at the same time as the fp16
    // group's own buffers, since that group's block has already fully
    // freed by now).
    double ms_pure_bf16 = -1.0, ms_native_bf16 = -1.0, ms_cutlass_bf16 = -1.0, ms_cublas_bf16 = -1.0;
    {
        __nv_bfloat16 *Bb = nullptr, *Cb = nullptr, *A_bf16 = nullptr;
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;

        bool ok = alloc_or_skip(reinterpret_cast<void**>(&Bb), k * n * sizeof(__nv_bfloat16), "bf16 group (B)",
                                shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&Cb), m * n * sizeof(__nv_bfloat16), "bf16 group (C)",
                                shape_desc);
        if (ok) {
            gen_bf16(k * n, Bb);
            cudaMemset(Cb, 0, m * n * sizeof(__nv_bfloat16));
            cudaDeviceSynchronize();
        }

        ok = ok &&
             alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp (wh)", shape_desc) &&
             alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp (wr)", shape_desc);
        if (ok) {
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16);
            cudaDeviceSynchronize();
            cudaFree(wr_bf16);
            wr_bf16 = nullptr;

            // Promoted here (2026-08-28) as this family's new best --
            // directly calls the promoted native bf16 kernel (below) on
            // wh_bf16, REINTERPRETED as __nv_bfloat16*, no decode kernel
            // or scratch buffer needed at all: fp32_bf16_bf16_only(wh) is
            // UNCONDITIONALLY exactly a bitcast of wh for every possible
            // input (bf16's exponent field matches fp32's own width/bias
            // exactly, unlike fp16's), so reinterpreting wh_bf16 directly
            // is byte-for-byte the same array the OLD fused kernel's own
            // inline decode produced -- see this kernel's own comment in
            // mm_cuda_kernels.cu for the full derivation.
            ms_pure_bf16 = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, wh_bf16, lda, Bb, ldb,
                    0.3, Cb, ldc);
            }, reps);
            std::printf(
                "    mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp   = %9.3fms (%10.1f GFlop/s)\n",
                ms_pure_bf16, flops / ms_pure_bf16 / 1e6);
        }

        ok = ok && alloc_or_skip(reinterpret_cast<void**>(&A_bf16), m * k * sizeof(__nv_bfloat16),
                                 "native/cutlass/cublas bf16 (A)", shape_desc);
        if (ok) {
            wh_to_bf16(m * k, wh_bf16, A_bf16);
            cudaDeviceSynchronize();
            cudaFree(wh_bf16);
            wh_bf16 = nullptr;

            // Promoted here (2026-08-28) as this family's new best -- a
            // straight, mechanical port of the fp16 leg's own iter_epivec/
            // acongruous/fastk_precomp2/serp refinement stack (see that
            // kernel's own comment in mm_cuda_kernels.cu for the full
            // derivation), MEASURED 1.17x-1.34x faster than plain deep4
            // (this family's prior best) at every size N>=2048 on real
            // hardware -- see verify_bf16_wide_bigreg_sq_deep4.sh's own
            // throughput sweep.
            ms_native_bf16 = time_call(start, stop, [&]() {
                cuda::gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, A_bf16, lda, Bb, ldb, 0.3,
                    Cb, ldc);
            }, reps);
            std::printf(
                "    gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp (native) = "
                "%9.3fms (%10.1f GFlop/s)\n",
                ms_native_bf16, flops / ms_native_bf16 / 1e6);

            cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            cutlass::bfloat16_t* A_ct = reinterpret_cast<cutlass::bfloat16_t*>(A_bf16);
            cutlass::bfloat16_t* B_ct = reinterpret_cast<cutlass::bfloat16_t*>(Bb);
            cutlass::bfloat16_t* C_ct = reinterpret_cast<cutlass::bfloat16_t*>(Cb);
            float alpha_f = 1.7f, beta_f = 0.3f;
            cutlass_bf16_cfg::Gemm gemm_op_bf16;
            typename cutlass_bf16_cfg::Gemm::Arguments args_bf16(problem_size, {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc},
                                                                  {C_ct, ldc}, {alpha_f, beta_f});
            size_t workspace_size_bf16 = cutlass_bf16_cfg::Gemm::get_workspace_size(args_bf16);
            void* workspace_bf16 = nullptr;
            if (workspace_size_bf16 > 0) cudaMalloc(&workspace_bf16, workspace_size_bf16);

            cutlass::Status status_bf16 = gemm_op_bf16.can_implement(args_bf16);
            if (status_bf16 == cutlass::Status::kSuccess) status_bf16 = gemm_op_bf16.initialize(args_bf16, workspace_bf16);
            if (status_bf16 == cutlass::Status::kSuccess) {
                ms_cutlass_bf16 = time_call(start, stop, [&]() { gemm_op_bf16(); }, reps);
                std::printf("    cutlass (real, bf16 tensor-op)                            = %9.3fms (%10.1f GFlop/s)\n",
                           ms_cutlass_bf16, flops / ms_cutlass_bf16 / 1e6);
            } else {
                std::printf("    cutlass bf16 SKIPPED (%s)\n", cutlass::cutlassGetStatusString(status_bf16));
            }
            if (workspace_bf16) cudaFree(workspace_bf16);

            ms_cublas_bf16 = time_call(start, stop, [&]() {
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                            static_cast<int>(k), &alpha_f, A_bf16, CUDA_R_16BF, lda, Bb, CUDA_R_16BF, ldb, &beta_f,
                            Cb, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }, reps);
            std::printf("    cublas (GemmEx, CUDA_R_16BF)                              = %9.3fms (%10.1f GFlop/s)\n",
                       ms_cublas_bf16, flops / ms_cublas_bf16 / 1e6);
        }

        cudaFree(A_bf16);
        cudaFree(wh_bf16);
        cudaFree(wr_bf16);
        cudaFree(Bb);
        cudaFree(Cb);
    }

    if (ms_pure_half > 0 && ms_native_fp16 > 0 && ms_cublas_fp16 > 0) {
        std::printf("    -- fp16 vs cublas:  %.3fx (pure_half)  %.3fx (native)\n",
                   (flops / ms_pure_half) / (flops / ms_cublas_fp16), (flops / ms_native_fp16) / (flops / ms_cublas_fp16));
        if (ms_cutlass_fp16 > 0) {
            std::printf("    -- fp16 vs cutlass: %.3fx (pure_half)  %.3fx (native)\n",
                       (flops / ms_pure_half) / (flops / ms_cutlass_fp16),
                       (flops / ms_native_fp16) / (flops / ms_cutlass_fp16));
        }
        std::printf("    -- fp16 vs native:  %.3fx (pure_half)\n", (flops / ms_pure_half) / (flops / ms_native_fp16));
    }
    if (ms_pure_bf16 > 0 && ms_native_bf16 > 0 && ms_cublas_bf16 > 0) {
        std::printf("    -- bf16 vs cublas:  %.3fx (pure_bf16)  %.3fx (native)\n",
                   (flops / ms_pure_bf16) / (flops / ms_cublas_bf16), (flops / ms_native_bf16) / (flops / ms_cublas_bf16));
        if (ms_cutlass_bf16 > 0) {
            std::printf("    -- bf16 vs cutlass: %.3fx (pure_bf16)  %.3fx (native)\n",
                       (flops / ms_pure_bf16) / (flops / ms_cutlass_bf16),
                       (flops / ms_native_bf16) / (flops / ms_cutlass_bf16));
        }
        std::printf("    -- bf16 vs native:  %.3fx (pure_bf16)\n", (flops / ms_pure_bf16) / (flops / ms_native_bf16));
    }
}

int main(int argc, char** argv) {
    // Extract --warmup N / --iters N from anywhere in argv (both
    // optional, default 10/10, see g_warmup_runs/g_iterations' own
    // comment above time_call) before this file's own original argument
    // parsing below runs -- argc/argv are reassigned to a filtered copy
    // with those two flags (and their values) removed, so every shape/
    // flag form the sibling file's own main() accepts still works
    // unchanged here.
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) {
            g_warmup_runs = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            g_iterations = std::atoi(argv[i + 1]);
            ++i;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== [advanced: warmup=%d iters=%d] ===\n", g_warmup_runs, g_iterations);
    std::printf("=== mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp / "
               "mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp "
               "vs native fp16/bf16 vs real CUTLASS (fp16/bf16 tensor-op) vs cuBLAS (Hgemm/GemmEx) ===\n");

    // Rectangular sweep: output held fixed at M=N=44000 (default), inner
    // dimension K swept across powers of two from 2^10 to 2^15. At 2
    // bytes/element per buffer even the largest K here is a small
    // footprint compared to the fp32 reconstruction tier's, so every K
    // is expected to produce real numbers, not skip.
    const std::vector<size_t> rect_ks = {1024, 2048, 4096, 8192, 16384, 32768};

    if (argc > 1 && std::string(argv[1]) == "--rect") {
        if (argc > 2) {
            size_t k = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
            // Optional 3rd arg overrides the M=N held fixed for this shape
            // (default 44000, matching the no-arg sweep below).
            size_t mn = (argc > 3) ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) : 44000;
            run_shape(mn, k, mn, start, stop, handle);
        } else {
            for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle);
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

    for (size_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536}) {
        run_shape(n, n, n, start, stop, handle);
    }

    for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle);

    cublasDestroy(handle);
    return 0;
}

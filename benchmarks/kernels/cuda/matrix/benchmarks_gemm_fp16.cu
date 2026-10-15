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
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp(
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

// Tunable-subnormal-density siblings of gen_fp32_16/gen_half above -- used
// only when --subnormal-density is passed (main() below), so every
// existing caller of gen_fp32_16/gen_half is untouched and produces
// bit-identical data to before. density is a FRACTION in [0, 1] (main()
// converts the CLI's percentage argument). For each element, a second,
// decorrelated hash off the same per-index seed decides (with
// probability `density`) whether that element is redrawn into the fp16
// subnormal band (0 < |x| < 2^-14, sign and magnitude uniform in that
// band) instead of pseudo_rand's own uniform-on-[-1,1] value -- so the
// realized subnormal fraction matches `density` in expectation (not
// exactly per-call, same as pseudo_rand's own approximate uniformity),
// and setting density=0 exactly reproduces pseudo_rand's own value at
// every index (the "redraw" branch is simply never taken).
__device__ float pseudo_rand_density(size_t idx, float density) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float u = (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // uniform [-1, 1], same grid as pseudo_rand

    uint32_t sel = s * 2246822519u + 3266489917u;  // decorrelated second hash off the same per-index seed
    sel ^= sel >> 13;
    bool subnormal = (sel % 1000000u) < static_cast<uint32_t>(density * 1000000.0f + 0.5f);
    if (subnormal) {
        const float kFp16SubnormalBound = 6.103515625e-5f;  // 2^-14, the fp16 normal/subnormal boundary
        return copysignf(fabsf(u) * kFp16SubnormalBound, u);  // uniform in (0, 2^-14), u's own sign
    }
    return u;
}

__global__ void gen_fp32_16_density_kernel(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_density(i, density), wh[i], wr[i]);
}
void gen_fp32_16_density(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_density_kernel<<<blocks, threads>>>(n, wh, wr, density);
}

__global__ void gen_half_density_kernel(size_t n, __half* out, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(pseudo_rand_density(i, density));
}
void gen_half_density(size_t n, __half* out, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_density_kernel<<<blocks, threads>>>(n, out, density);
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
    std::printf("    %-58s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). The square sweep calls this
// with m=k=n; the rectangular sweep holds m=n=44000 fixed and varies k.
// density: fraction in [0, 1] of A/B's own entries redrawn into the fp16
// subnormal band (see pseudo_rand_density's own comment above) -- 0.0
// (the default, all existing callers) reproduces this function's
// original behavior exactly. Scoped to the fp16 group's A (wh16/wr16)
// and B (Bh) only -- the bf16 group's inputs (Bb, wh16bf) are UNCHANGED
// regardless of density: bf16's normal/subnormal boundary is ~2^-126,
// astronomically smaller than fp16's 2^-14, so a density calibrated to
// fp16's own band would place essentially none of bf16's mass in ITS
// own (unreachably tiny) subnormal band -- a meaningless test for that
// tier, so this deliberately doesn't try to give it its own version.
void run_shape(size_t m, size_t k, size_t n, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle,
                float density = 0.0f) {
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
    double ms_pure_half = -1.0, ms_pure_half_fast = -1.0, ms_native_fp16 = -1.0, ms_cutlass_fp16 = -1.0,
           ms_cublas_fp16 = -1.0;
    {
        __half *Bh = nullptr, *Ch = nullptr, *A_half = nullptr;
        uint16_t *wh16 = nullptr, *wr16 = nullptr;

        bool ok = alloc_or_skip(reinterpret_cast<void**>(&Bh), k * n * sizeof(__half), "fp16 group (B)", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&Ch), m * n * sizeof(__half), "fp16 group (C)", shape_desc);
        if (ok) {
            if (density > 0.0f) {
                gen_half_density(k * n, Bh, density);
            } else {
                gen_half(k * n, Bh);
            }
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
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16, wr16, density);
            } else {
                gen_fp32_16(m * k, wh16, wr16);
            }
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

            // Fast-decode sibling of the leg above -- same wh16/Bh/Ch
            // buffers, only the decode inside the kernel differs
            // (fp32_16_half_only_fast instead of fp32_16_half_only, see
            // that function's own comment in fp32_16_pack.hpp). Added as
            // a new, separate call; the leg above is untouched.
            ms_pure_half_fast = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7, m, k, n, wh16, lda, Bh, ldb, 0.3,
                    Ch, ldc);
            }, reps);
            std::printf(
                "    mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp = "
                "%9.3fms (%10.1f GFlop/s)\n",
                ms_pure_half_fast, flops / ms_pure_half_fast / 1e6);

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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp / "
               "mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp "
               "vs native fp16/bf16 vs real CUTLASS (fp16/bf16 tensor-op) vs cuBLAS (Hgemm/GemmEx) ===\n");

    // --subnormal-density <PCT>: scans it out of the argument list up front
    // (may appear anywhere) so the existing --rect/N shape-selection logic
    // below is otherwise untouched -- it just sees whatever args remain.
    // Only affects the fp16 group's A/B generation (run_shape's own
    // density parameter, see its comment); 0 (the default, when this flag
    // is absent) reproduces every existing invocation's behavior exactly.
    std::vector<std::string> args(argv + 1, argv + argc);
    float density = 0.0f;
    for (size_t i = 0; i < args.size();) {
        if (args[i] == "--subnormal-density" && i + 1 < args.size()) {
            float pct = std::strtof(args[i + 1].c_str(), nullptr);
            density = pct / 100.0f;
            args.erase(args.begin() + static_cast<long>(i), args.begin() + static_cast<long>(i) + 2);
            std::printf("=== fp16-group A/B subnormal density: %.2f%% (fraction %.4f) ===\n", pct, density);
        } else {
            ++i;
        }
    }

    // Rectangular sweep: output held fixed at M=N=44000 (default), inner
    // dimension K swept across powers of two from 2^10 to 2^15. At 2
    // bytes/element per buffer even the largest K here is a small
    // footprint compared to the fp32 reconstruction tier's, so every K
    // is expected to produce real numbers, not skip.
    const std::vector<size_t> rect_ks = {1024, 2048, 4096, 8192, 16384, 32768};

    // Below: identical logic to before --subnormal-density was scanned out,
    // just reading from the filtered `args` vector (args[i]) in place of
    // (argv[i+1]/argc-1) -- same positions, since the flag has already
    // been erased above regardless of where it appeared.
    if (!args.empty() && args[0] == "--rect") {
        if (args.size() > 1) {
            size_t k = static_cast<size_t>(std::strtoull(args[1].c_str(), nullptr, 10));
            // Optional 3rd arg overrides the M=N held fixed for this shape
            // (default 44000, matching the no-arg sweep below).
            size_t mn = (args.size() > 2) ? static_cast<size_t>(std::strtoull(args[2].c_str(), nullptr, 10)) : 44000;
            run_shape(mn, k, mn, start, stop, handle, density);
        } else {
            for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle, density);
        }
        cublasDestroy(handle);
        return 0;
    }

    if (!args.empty()) {
        size_t n = static_cast<size_t>(std::strtoull(args[0].c_str(), nullptr, 10));
        run_shape(n, n, n, start, stop, handle, density);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t n : {1024, 2048, 4096, 8192, 16384, 32768, 65536}) {
        run_shape(n, n, n, start, stop, handle, density);
    }

    for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle, density);

    cublasDestroy(handle);
    return 0;
}

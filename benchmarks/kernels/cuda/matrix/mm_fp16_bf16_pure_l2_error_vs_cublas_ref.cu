// Standalone Frobenius-relative-error analysis for the "pure/composite"
// (tensor-core compute, decode-only) tier of GEMM -- the accuracy-vs-N
// companion to benchmarks_gemm_fp16.cu (which plots GFlop/s for
// the same square sweep, via scripts/plot_fp16_bf16_pure_vs_cutlass_
// ref.m). Measures how far each method's output matrix drifts from a
// genuine FP32-precision reference (cublasSgemm on genuine, unrounded
// float32 A and B), not a throughput number. See scripts/plot_gemm_
// fp16_bf16_pure_l2_error_vs_cublas_ref.m for the resulting plot. Named
// "l2_error" for consistency with every sibling *_l2_error_vs_cublas_
// ref.cu file, even though the actual metric computed is the FROBENIUS
// norm -- see mm_fp32_recon_l2_error_vs_cublas_ref.cu's own header for
// why that's textually identical to L2 on a flattened array (this file's
// Row-5 GEMM-recon-tier sibling).
//
// Five "TOMOS" series, verified directly against scripts/fp16_bf16_
// pure_vs_cutlass_ref_matched.sh's own current METHOD_MARKERS and
// plot_fp16_bf16_pure_vs_cutlass_ref.m's own current `series` struct,
// PLUS four external, non-TOMOS GEMM baselines added directly to the
// fp16 and bf16 groups (cuBLAS FP16/BF16, CUTLASS FP16/BF16 -- see the
// CutlassGemm/CutlassGemmBF16 type aliases below for the exact configs),
// each run on the SAME native-rounded A/B its own group's native TOMOS
// series uses:
//
//   TOMOS FP16              = gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
//                              iter_epivec_acongruous_fastk_precomp2_serp
//                              (native, genuinely __half-typed A/B/C
//                              throughout, no packed format involved)
//   cuBLAS FP16              = cublasHgemm, on the SAME native-rounded
//                              A/B as TOMOS FP16 (native) -- its own
//                              internal accumulation precision is not
//                              guaranteed to match (may accumulate in
//                              half rather than float).
//   CUTLASS FP16             = cutlass::gemm::device::Gemm, same config
//                              as scripts/cutlass_fp16_gemm_ref.cu's own
//                              binary, on the SAME native-rounded A/B.
//   TOMOS BF16               = gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_
//                              iter_epivec_acongruous_fastk_precomp2_serp
//                              (native, genuinely __nv_bfloat16-typed)
//   cuBLAS BF16              = cublasGemmEx (CUDA_R_16BF I/O,
//                              CUBLAS_COMPUTE_32F) -- cuBLAS has no
//                              simple "cublasBgemm" convenience wrapper
//                              the way cublasHgemm exists for fp16, on
//                              the SAME native-rounded A/B as TOMOS BF16
//                              (native).
//   CUTLASS BF16             = cutlass::gemm::device::Gemm, same
//                              tensorop config as CUTLASS FP16 (Ampere's
//                              bf16 tensor cores share fp16's own
//                              16x8x16 MMA shape), on the SAME native-
//                              rounded A/B.
//   TOMOS FP32 / FP16        = mm_fp32_16_pure_half_wide_bigreg_sq_
//                              inline_dirty_acongruous_fastk_precomp2_serp
//   TOMOS FP32 / FP16 (workspace) = mm_fp32_16_pure_half_gemm_
//                              materialize_cached_vec_fastk_precomp2_serp
//   TOMOS FP32 / BF16        = mm_fp32_bf16_pure_bf16_gemm_direct_
//                              fastk_precomp2_serp
//
// Unlike mm_fp32_recon_l2_error_vs_cublas_ref.cu's own "full
// reconstruction" tier (which decodes packed A back to genuine float
// before a float-accumulating GEMM, achieving near-fp32 accuracy), this
// tier is a fundamentally LOSSIER, TENSOR-CORE-COMPUTE tier: A is
// decoded only as far as a plain __half/__nv_bfloat16 value (fp32_16_
// half_only / fp32_bf16_bf16_only -- truncating straight to that
// format's own native storage precision, no mant_lo/extended-exponent
// correction at all), and B is likewise ALWAYS stored as __half/
// __nv_bfloat16 by every kernel's own signature here (no float* B
// overload exists in this tier -- confirmed by reading every one of the
// five forward declarations below) -- so, unlike every other error
// benchmark in this codebase, there is no "B/x stays uncompressed"
// convention to lean on: the genuine-FP32 reference below intentionally
// captures BOTH A's packing/truncation loss AND B's fp16/bf16 storage
// rounding, since the tier's own design requires both operands to pass
// through reduced-precision storage before the tensor cores ever see
// them.
//
// The actual multiply-accumulate for all five kernels happens in genuine
// float32 -- confirmed by reading gemm_fp16_mma_ptx_wide_bigreg_sq_
// deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm_kernel's own
// body (float acc[...][...][4], fed by "mma.sync.aligned.m16n8k16.row.
// col.f32.f16.f16.f32" PTX, i.e. the sm_80 tensor core's own hardware-
// mandated f32 accumulator for f16/f16 inputs -- the SAME PTX
// accumulator type the bf16.bf16 variant uses) -- so there is NO
// INTERNAL accumulator-overflow concern (a fixed hardware property, not
// a per-kernel choice).
//
// BUT the epilogue still writes that float32 accumulator down into
// C_half/C_bf16 (this tier's own __half/__nv_bfloat16-typed C, see this
// file's own header above) -- for the THREE fp16-group series (TOMOS
// FP16, TOMOS FP32/FP16, TOMOS FP32/FP16 (workspace)), that final
// narrowing cast is itself subject to fp16's own ~65504 max finite
// magnitude, and C[i,j] is a sum of K products (not a single stored
// value), so its own magnitude scales with sqrt(K) -- a K-DEPENDENT
// OUTPUT-STORAGE ceiling, the exact same shape of constraint as the
// GEVM "pure_half"/"hybrid" tier's own internal-accumulator bound (see
// mv_fp32_16_l2_error_vs_cublas_ref.cu's own header), just moved from
// mid-computation to the epilogue. A real bug this file's own first
// SCALES=16384 run caught: TOMOS FP16/FP32/FP16/FP32/FP16(workspace)
// all reported "inf" at EVERY shape at scale=4096 already, confirming a
// K-dependent output-storage overflow, not a fluke at the largest shape
// only. TOMOS BF16/TOMOS FP32/BF16 are unaffected (bf16 shares fp32's
// own 8-bit exponent, no realistic ceiling at any scale used here) --
// see scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh's
// own SCALES comment for the corrected, empirically-derived bound.
//
// A (float, m*k) and B (float, k*n) share ONE pseudo_rand-derived ground
// truth and ONE cublasSgemm reference (C_ref) across ALL FIVE series --
// gen_float, gen_fp32_16, gen_fp32_bf16, gen_half, and gen_bf16 are all
// pure, deterministic functions of (linear index, scale) alone, so
// calling each with the SAME n and scale is guaranteed to correspond to
// the SAME underlying "true" values.
//
// normal_scale multiplies A's own "normal"-range data only (same
// convention every sibling error benchmark uses) -- B is left permanently
// UNSCALED (scale=1.0), since B's own format ceiling (also fp16 for the
// fp16 group) is a fixed, shape-independent property, not something this
// sweep is meant to stress; only A's magnitude/exponent range is the
// thing actually varying across the packed-format comparison.
//
// Frobenius relative error: ||C_ref - C_method||_F / ||C_ref||_F,
// computed on the HOST over the OUTPUT matrix only (size M*N), after
// converting each method's own __half/__nv_bfloat16-typed C to float.
//
// Usage: mm_fp16_bf16_pure_l2_error_vs_cublas_ref [normal_scale [N]]
//   No args: runs the same SIZES sweep as scripts/fp16_bf16_pure_vs_
//            cutlass_ref_matched.sh's own square sweep (1024 through
//            65536, M=K=N), at normal_scale=1.0.
//   normal_scale: runs the full SIZES sweep with A's own "normal"-range
//            data widened to [-scale,scale] instead. See scripts/
//            benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh
//            for the actual scale tiers swept.
//   normal_scale N: runs just that one square shape (M=K=N=N) at that
//            scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemm_fp16_bf16_pure_l2_
// error_vs_cublas_ref.sh.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

// cuBLAS FP16 (cublasHgemm) and CUTLASS FP16 (cutlass::gemm::device::Gemm)
// baselines added to the fp16 group below -- external, non-TOMOS FP16 GEMM
// implementations run on the SAME native-rounded A/B every TOMOS FP16
// series in that group already uses, to check whether TOMOS FP16
// (native)'s own error is comparable to, better, or worse than these
// external implementations. Same "textbook" Ampere tensorop CUTLASS
// config (128x256x32/64x64x32/16x8x16, 3-stage, FP32 accumulator) as
// scripts/cutlass_fp16_gemm_ref.cu's own binary and mm_fp16_kernels_l2_
// error_vs_cublas_ref.cu's own CUTLASS series -- reused verbatim.
using CutlassElementA = cutlass::half_t;
using CutlassLayoutA = cutlass::layout::ColumnMajor;
using CutlassElementB = cutlass::half_t;
using CutlassLayoutB = cutlass::layout::ColumnMajor;
using CutlassElementC = cutlass::half_t;
using CutlassLayoutC = cutlass::layout::ColumnMajor;
using CutlassElementAccumulator = float;

using CutlassGemm = cutlass::gemm::device::Gemm<
    CutlassElementA, CutlassLayoutA,
    CutlassElementB, CutlassLayoutB,
    CutlassElementC, CutlassLayoutC,
    CutlassElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        CutlassElementC, 128 / cutlass::sizeof_bits<CutlassElementC>::value,
        CutlassElementAccumulator, CutlassElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// Same config, bf16 element types -- Ampere's bf16 tensor cores use the
// SAME 16x8x16 MMA instruction shape as fp16's own, so this is otherwise
// an identical instantiation to CutlassGemm above.
using CutlassElementA_BF16 = cutlass::bfloat16_t;
using CutlassLayoutA_BF16 = cutlass::layout::ColumnMajor;
using CutlassElementB_BF16 = cutlass::bfloat16_t;
using CutlassLayoutB_BF16 = cutlass::layout::ColumnMajor;
using CutlassElementC_BF16 = cutlass::bfloat16_t;
using CutlassLayoutC_BF16 = cutlass::layout::ColumnMajor;
using CutlassElementAccumulator_BF16 = float;

using CutlassGemmBF16 = cutlass::gemm::device::Gemm<
    CutlassElementA_BF16, CutlassLayoutA_BF16,
    CutlassElementB_BF16, CutlassLayoutB_BF16,
    CutlassElementC_BF16, CutlassLayoutC_BF16,
    CutlassElementAccumulator_BF16,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        CutlassElementC_BF16, 128 / cutlass::sizeof_bits<CutlassElementC_BF16>::value,
        CutlassElementAccumulator_BF16, CutlassElementAccumulator_BF16>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

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

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_float_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * pseudo_rand(i);
}
void gen_float(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out, scale);
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

__global__ void gen_fp32_bf16_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(scale * pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

// B is generated directly at each format's own storage precision (never
// a float* B overload in this tier -- see this file's own header) but
// from the SAME pseudo_rand(idx) values gen_float above would produce at
// scale=1.0, so it stays index-matched to A_true's own ground truth for
// the reference computation.
__global__ void gen_half_kernel(size_t n, __half* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(scale * pseudo_rand(i));
}
void gen_half(size_t n, __half* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out, scale);
}

__global__ void gen_bf16_kernel(size_t n, __nv_bfloat16* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(scale * pseudo_rand(i));
}
void gen_bf16(size_t n, __nv_bfloat16* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_bf16_kernel<<<blocks, threads>>>(n, out, scale);
}

__global__ void half_to_float_kernel(size_t n, const __half* in, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}
void half_to_float(size_t n, const __half* in, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    half_to_float_kernel<<<blocks, threads>>>(n, in, out);
}

__global__ void bf16_to_float_kernel(size_t n, const __nv_bfloat16* in, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(in[i]);
}
void bf16_to_float(size_t n, const __nv_bfloat16* in, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_to_float_kernel<<<blocks, threads>>>(n, in, out);
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||C_ref - C_gpu||_F / ||C_ref||_F, computed on the host over the
// OUTPUT matrix only (size M*N).
double frobenius_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). normal_scale widens/narrows A's
// own "normal"-range data by this multiplier -- B is left UNSCALED by
// normal_scale specifically (see this file's own header for why).
//
// normalized is a separate, orthogonal knob: when true, B (in all three
// of its forms below -- B_true, B_half, B_bf16, all generated from the
// SAME pseudo_rand(idx) values so they stay index-matched) is instead
// scaled by 1/sqrt(k), so C[i,j]'s own row-sum keeps magnitude
// ~normal_scale regardless of K instead of growing like normal_scale*
// sqrt(K). This is applied at DATA-GENERATION time, NOT via alpha --
// alpha here is only multiplied in at the very epilogue, after the
// K-reduction has already fully accumulated (confirmed by reading gemm_
// fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_
// precomp2_serp_kernel's own body: `v0 = alpha * static_cast<__half>(
// acc[wm][wn][0]);` -- the cast to __half happens BEFORE the multiply by
// alpha, so if the raw float accumulator already exceeds fp16 range, the
// cast alone produces Infinity and no subsequent alpha multiply can undo
// that). Scaling B's own VALUES down instead works regardless of that
// ordering, since the accumulator only ever sees the already-smaller
// products. See scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_
// cublas_ref.sh's own --normalized SCALES comment for the numbers this
// unlocks -- this is the row that found the real "inf at every shape"
// bug an alpha-based fix would NOT have caught.
void run_shape(size_t m, size_t k, size_t n, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);
    std::printf("  %s\n", shape_desc);

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;

    float b_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float *A_true = nullptr, *B_true = nullptr, *C_ref = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&A_true), m * k * sizeof(float), "A_true", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&B_true), k * n * sizeof(float), "B_true", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&C_ref), m * n * sizeof(float), "C_ref", shape_desc)) {
        cudaFree(A_true);
        cudaFree(B_true);
        cudaFree(C_ref);
        return;
    }
    gen_float(m * k, A_true, normal_scale);
    gen_float(k * n, B_true, b_scale);

    float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
               &f_alpha, A_true, lda, B_true, ldb, &f_beta, C_ref, ldc);
    std::vector<float> C_ref_h(m * n);
    cudaMemcpy(C_ref_h.data(), C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- fp16 group: TOMOS FP16 (native), TOMOS FP32/FP16, TOMOS
    // FP32/FP16 (workspace) -- own __half B, shared across all three. ----
    {
        __half* B_half = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&B_half), k * n * sizeof(__half), "B_half", shape_desc)) {
            gen_half(k * n, B_half, b_scale);

            // TOMOS FP16 (native): genuinely __half-typed A, no packed
            // format involved.
            {
                __half *A_half = nullptr, *C_half = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_half), m * k * sizeof(__half), "A_half", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_native",
                                  shape_desc)) {
                    gen_half(m * k, A_half, normal_scale);
                    cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half,
                        static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half,
                        static_cast<size_t>(ldc));
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_native_fp16",
                                      shape_desc)) {
                        half_to_float(m * n, C_half, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    tomos_fp16               = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                }
                cudaFree(A_half);
                cudaFree(C_half);
            }

            // cuBLAS FP16 baseline: cublasHgemm on freshly-generated
            // native-rounded A (same normal_scale, same generator as
            // TOMOS FP16 native above) and the SAME B_half this whole
            // fp16 group shares.
            {
                __half *A_half_cb = nullptr, *C_half_cb = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_half_cb), m * k * sizeof(__half), "A_half_cublas",
                                  shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_half_cb), m * n * sizeof(__half), "C_half_cublas",
                                  shape_desc)) {
                    gen_half(m * k, A_half_cb, normal_scale);
                    __half h_alpha = __float2half(static_cast<float>(alpha));
                    __half h_beta = __float2half(static_cast<float>(beta));
                    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                               static_cast<int>(k), &h_alpha, A_half_cb, lda, B_half, ldb, &h_beta, C_half_cb, ldc);
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_cublas_fp16",
                                      shape_desc)) {
                        half_to_float(m * n, C_half_cb, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    cublas_fp16              = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                }
                cudaFree(A_half_cb);
                cudaFree(C_half_cb);
            }

            // CUTLASS FP16 baseline: reinterpret_casts onto freshly-
            // generated native-rounded A and the SAME B_half this whole
            // fp16 group shares (cutlass::half_t and __half share the
            // same 16-bit IEEE layout).
            {
                __half *A_half_ct = nullptr, *C_half_ct = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_half_ct), m * k * sizeof(__half), "A_half_cutlass",
                                  shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_half_ct), m * n * sizeof(__half), "C_half_cutlass",
                                  shape_desc)) {
                    gen_half(m * k, A_half_ct, normal_scale);
                    cutlass::half_t* A_ct = reinterpret_cast<cutlass::half_t*>(A_half_ct);
                    cutlass::half_t* B_ct = reinterpret_cast<cutlass::half_t*>(B_half);
                    cutlass::half_t* C_ct = reinterpret_cast<cutlass::half_t*>(C_half_ct);

                    cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n),
                                                          static_cast<int>(k));
                    CutlassElementAccumulator ct_alpha(static_cast<float>(alpha)), ct_beta(static_cast<float>(beta));
                    typename CutlassGemm::Arguments args(problem_size, {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc},
                                                         {C_ct, ldc}, {ct_alpha, ct_beta});

                    size_t workspace_size = CutlassGemm::get_workspace_size(args);
                    void* workspace = nullptr;
                    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

                    CutlassGemm gemm_op;
                    cutlass::Status status = gemm_op.can_implement(args);
                    if (status == cutlass::Status::kSuccess) status = gemm_op.initialize(args, workspace);
                    if (status == cutlass::Status::kSuccess) status = gemm_op();

                    if (status == cutlass::Status::kSuccess) {
                        float* C_gpu = nullptr;
                        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float),
                                          "C_gpu_cutlass_fp16", shape_desc)) {
                            half_to_float(m * n, C_half_ct, C_gpu);
                            std::vector<float> C_gpu_h(m * n);
                            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                            std::printf("    cutlass_fp16             = %.6e\n",
                                       frobenius_rel_err(C_ref_h, C_gpu_h));
                            cudaFree(C_gpu);
                        }
                    } else {
                        std::printf("    cutlass_fp16             SKIPPED (cutlass status: %s at %s)\n",
                                   cutlass::cutlassGetStatusString(status), shape_desc);
                    }
                    if (workspace) cudaFree(workspace);
                }
                cudaFree(A_half_ct);
                cudaFree(C_half_ct);
            }

            // TOMOS FP32/FP16 and TOMOS FP32/FP16 (workspace): share the
            // SAME wh (both decode via fp32_16_half_only -- plain,
            // truncating decode, no algorithmic difference in what data
            // they need).
            {
                uint16_t *wh = nullptr, *wr = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t), "wh_fp16", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&wr), m * k * sizeof(uint16_t), "wr_fp16", shape_desc)) {
                    gen_fp32_16(m * k, wh, wr, normal_scale);

                    __half* C_half = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_pure_half",
                                      shape_desc)) {
                        cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
                            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, wh,
                            static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half,
                            static_cast<size_t>(ldc));
                        float* C_gpu = nullptr;
                        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float),
                                          "C_gpu_pure_half", shape_desc)) {
                            half_to_float(m * n, C_half, C_gpu);
                            std::vector<float> C_gpu_h(m * n);
                            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                            std::printf("    tomos_fp32_fp16          = %.6e\n",
                                       frobenius_rel_err(C_ref_h, C_gpu_h));
                            cudaFree(C_gpu);
                        }
                        cudaFree(C_half);
                    }

                    __half* C_half_ws = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_half_ws), m * n * sizeof(__half),
                                      "C_half_pure_half_workspace", shape_desc)) {
                        cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
                            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, wh,
                            static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half_ws,
                            static_cast<size_t>(ldc));
                        float* C_gpu = nullptr;
                        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float),
                                          "C_gpu_pure_half_workspace", shape_desc)) {
                            half_to_float(m * n, C_half_ws, C_gpu);
                            std::vector<float> C_gpu_h(m * n);
                            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                            std::printf("    tomos_fp32_fp16_workspace = %.6e\n",
                                       frobenius_rel_err(C_ref_h, C_gpu_h));
                            cudaFree(C_gpu);
                        }
                        cudaFree(C_half_ws);
                    }
                    cudaFree(wh);
                    cudaFree(wr);
                }
            }
            cudaFree(B_half);
        }
    }

    // ---- bf16 group: TOMOS BF16 (native), TOMOS FP32/BF16 -- own
    // __nv_bfloat16 B, shared across both. ----
    {
        __nv_bfloat16* B_bf16 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&B_bf16), k * n * sizeof(__nv_bfloat16), "B_bf16", shape_desc)) {
            gen_bf16(k * n, B_bf16, b_scale);

            // TOMOS BF16 (native): genuinely __nv_bfloat16-typed A.
            {
                __nv_bfloat16 *A_bf16 = nullptr, *C_bf16 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16), m * k * sizeof(__nv_bfloat16), "A_bf16",
                                  shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_bf16), m * n * sizeof(__nv_bfloat16), "C_bf16_native",
                                  shape_desc)) {
                    gen_bf16(m * k, A_bf16, normal_scale);
                    cuda::gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_bf16,
                        static_cast<size_t>(lda), B_bf16, static_cast<size_t>(ldb), beta, C_bf16,
                        static_cast<size_t>(ldc));
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_native_bf16",
                                      shape_desc)) {
                        bf16_to_float(m * n, C_bf16, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    tomos_bf16               = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                }
                cudaFree(A_bf16);
                cudaFree(C_bf16);
            }

            // cuBLAS BF16 baseline: cuBLAS has no simple "cublasBgemm"
            // convenience wrapper the way cublasHgemm exists for fp16 --
            // cublasGemmEx with an explicit FP32 compute type (bf16
            // tensor cores natively accumulate in fp32 on sm_80) is the
            // standard mixed-precision bf16 GEMM path. Freshly-generated
            // native-rounded A (same normal_scale, same generator as
            // TOMOS BF16 native above) and the SAME B_bf16 this whole
            // bf16 group shares.
            {
                __nv_bfloat16 *A_bf16_cb = nullptr, *C_bf16_cb = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_cb), m * k * sizeof(__nv_bfloat16),
                                  "A_bf16_cublas", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_bf16_cb), m * n * sizeof(__nv_bfloat16),
                                  "C_bf16_cublas", shape_desc)) {
                    gen_bf16(m * k, A_bf16_cb, normal_scale);
                    float f_alpha_cb = static_cast<float>(alpha), f_beta_cb = static_cast<float>(beta);
                    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                                static_cast<int>(k), &f_alpha_cb, A_bf16_cb, CUDA_R_16BF, lda, B_bf16, CUDA_R_16BF,
                                ldb, &f_beta_cb, C_bf16_cb, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_cublas_bf16",
                                      shape_desc)) {
                        bf16_to_float(m * n, C_bf16_cb, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    cublas_bf16              = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                }
                cudaFree(A_bf16_cb);
                cudaFree(C_bf16_cb);
            }

            // CUTLASS BF16 baseline: reinterpret_casts onto freshly-
            // generated native-rounded A and the SAME B_bf16 this whole
            // bf16 group shares (cutlass::bfloat16_t and __nv_bfloat16
            // share the same 16-bit layout).
            {
                __nv_bfloat16 *A_bf16_ct = nullptr, *C_bf16_ct = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_ct), m * k * sizeof(__nv_bfloat16),
                                  "A_bf16_cutlass", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&C_bf16_ct), m * n * sizeof(__nv_bfloat16),
                                  "C_bf16_cutlass", shape_desc)) {
                    gen_bf16(m * k, A_bf16_ct, normal_scale);
                    cutlass::bfloat16_t* A_ct = reinterpret_cast<cutlass::bfloat16_t*>(A_bf16_ct);
                    cutlass::bfloat16_t* B_ct = reinterpret_cast<cutlass::bfloat16_t*>(B_bf16);
                    cutlass::bfloat16_t* C_ct = reinterpret_cast<cutlass::bfloat16_t*>(C_bf16_ct);

                    cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n),
                                                          static_cast<int>(k));
                    CutlassElementAccumulator_BF16 ct_alpha(static_cast<float>(alpha)),
                        ct_beta(static_cast<float>(beta));
                    typename CutlassGemmBF16::Arguments args(problem_size, {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc},
                                                             {C_ct, ldc}, {ct_alpha, ct_beta});

                    size_t workspace_size = CutlassGemmBF16::get_workspace_size(args);
                    void* workspace = nullptr;
                    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

                    CutlassGemmBF16 gemm_op;
                    cutlass::Status status = gemm_op.can_implement(args);
                    if (status == cutlass::Status::kSuccess) status = gemm_op.initialize(args, workspace);
                    if (status == cutlass::Status::kSuccess) status = gemm_op();

                    if (status == cutlass::Status::kSuccess) {
                        float* C_gpu = nullptr;
                        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float),
                                          "C_gpu_cutlass_bf16", shape_desc)) {
                            bf16_to_float(m * n, C_bf16_ct, C_gpu);
                            std::vector<float> C_gpu_h(m * n);
                            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                            std::printf("    cutlass_bf16             = %.6e\n",
                                       frobenius_rel_err(C_ref_h, C_gpu_h));
                            cudaFree(C_gpu);
                        }
                    } else {
                        std::printf("    cutlass_bf16             SKIPPED (cutlass status: %s at %s)\n",
                                   cutlass::cutlassGetStatusString(status), shape_desc);
                    }
                    if (workspace) cudaFree(workspace);
                }
                cudaFree(A_bf16_ct);
                cudaFree(C_bf16_ct);
            }

            // TOMOS FP32/BF16: decode via fp32_bf16_bf16_only (plain,
            // truncating decode).
            {
                uint16_t *wh = nullptr, *wr = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&wr), m * k * sizeof(uint16_t), "wr_bf16", shape_desc)) {
                    gen_fp32_bf16(m * k, wh, wr, normal_scale);
                    __nv_bfloat16* C_bf16 = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_bf16), m * n * sizeof(__nv_bfloat16),
                                      "C_bf16_pure_bf16", shape_desc)) {
                        cuda::mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(
                            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, wh,
                            static_cast<size_t>(lda), B_bf16, static_cast<size_t>(ldb), beta, C_bf16,
                            static_cast<size_t>(ldc));
                        float* C_gpu = nullptr;
                        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float),
                                          "C_gpu_pure_bf16", shape_desc)) {
                            bf16_to_float(m * n, C_bf16, C_gpu);
                            std::vector<float> C_gpu_h(m * n);
                            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                            std::printf("    tomos_fp32_bf16          = %.6e\n",
                                       frobenius_rel_err(C_ref_h, C_gpu_h));
                            cudaFree(C_gpu);
                        }
                        cudaFree(C_bf16);
                    }
                    cudaFree(wh);
                    cudaFree(wr);
                }
            }
            cudaFree(B_bf16);
        }
    }

    cudaFree(A_true);
    cudaFree(B_true);
    cudaFree(C_ref);
}

}  // namespace

int main(int argc, char** argv) {
    // Extract --normalized (a bare flag, may appear anywhere in argv)
    // before this file's own original positional argument parsing below
    // runs -- see run_shape's own comment for what it does and why it's
    // implemented at data-generation time, not via alpha.
    bool normalized = false;
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--normalized") {
            normalized = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    cublasHandle_t handle;
    cublasCreate(&handle);

    // normal_scale (first arg, default 1.0) multiplies A's own "normal"-
    // range data. Every series here accumulates internally in genuine
    // float32 (see this file's own header -- a fixed sm_80 tensor-core
    // hardware property), so there is no INTERNAL accumulator-overflow
    // concern -- but the fp16-group series (TOMOS FP16, TOMOS FP32/FP16,
    // TOMOS FP32/FP16 (workspace)) still narrow that float32 result down
    // into __half-typed C at the epilogue, and C[i,j] sums K products,
    // so ITS OWN magnitude grows with sqrt(K) against fp16's ~65504 max
    // finite magnitude -- a real, K-dependent OUTPUT-STORAGE ceiling
    // (see this file's own header for the bug this caught: scale=4096
    // already produced "inf" for all three fp16-group series at every
    // shape). Without --normalized, DO NOT reuse mm_fp32_recon_l2_error_
    // vs_cublas_ref.cu's own wide SCALES here -- that file's own C is
    // genuine float throughout, with no such ceiling at any scale; see
    // scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh's
    // own SCALES comment for the corrected, empirically-derived bound
    // this file's own default (non-normalized) sweep uses. WITH --
    // normalized (see run_shape's own comment), C's own row-sum stays
    // ~normal_scale regardless of K, so the original wide SCALES range
    // (up to 16384) becomes safe again -- see that same .sh comment for
    // the --normalized-specific SCALES this script's own driver uses.
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== GEMM: pure/composite-tier (tensor-core) Frobenius relative error vs FP32 (cublasSgemm) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024,  1536,  2048,  3072,  4096, 6144, 8192,
                                              12288, 16384, 24576, 32768, 49152, 65536};

    if (argc > 2) {
        size_t n = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
        run_shape(n, n, n, handle, normal_scale, normalized);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t n : square_sizes) run_shape(n, n, n, handle, normal_scale, normalized);

    cublasDestroy(handle);
    return 0;
}

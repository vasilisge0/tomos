// Standalone L2-relative-error analysis for the TRANSPOSED (y = alpha*A^T*x
// + beta*y) genuine-16-bit-compute tier of GEMV -- the accuracy-vs-M
// companion to mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu (which
// plots GFlop/s for the same shapes) and the TRANSPOSE sibling of
// mv_fp32_16_l2_error_vs_cublas_ref.cu (this file's own NO_TRANSPOSE
// counterpart, whose header carries the full derivation of every
// generator/convention reused verbatim below). See scripts/plot_gemv_
// transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.m for the resulting
// plot.
//
// Five "TOMOS" series (was four; TOMOS FP16 (fast) added later, see its
// own comment below), verified directly against scripts/benchmarks_
// gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh's own current
// METHOD_MARKERS and plot_gemv_transpose_fp32_16_pure_half_vs_cublas_
// ref.m's own current `series` struct (NOT assumed from an older survey --
// this tier has accumulated many more experimental variants than the
// NO_TRANSPOSE tier since; only these below are actually plotted/
// promoted-adjacent there):
//
//   TOMOS FP16              = mv_fp32_16_pure_half_transpose_
//                              atomiconly_scaled_mblock_vec4_split
//                              (needs its own float* scratch workspace,
//                              sized via its own _workspace_size helper --
//                              unlike every other kernel in this file)
//   TOMOS FP16 (fast)       = mv_fp32_16_pure_half_fast_transpose_
//                              atomiconly_scaled_mblock_vec4_split --
//                              fp32_16_half_only_fast decode sibling of
//                              TOMOS FP16 above, safe on arbitrary data
//                              (shares TOMOS FP16's own wh/reference, no
//                              separate generator needed). NOT the same
//                              kernel TOMOS FP32/FP16 (implicit) below
//                              uses, despite the similar "fast" naming
//                              history -- see that series' own comment.
//   TOMOS FP32/FP16         = mv_fp32_16_hybrid_transpose_atomiconly_
//                              scaled_mblock_vec4_paired
//   TOMOS FP32/FP16 (implicit) = mv_fp32_16_pure_half_transpose_
//                              atomiconly_scaled_mblock_vec4_fast -- an
//                              OLDER, UNSAFE kernel (unconditional
//                              fp32_16_fast_half bit-reinterpret, requires
//                              control=0 always, hence its own magnitude-
//                              restricted generator below). Renamed from
//                              "(fast)" to "(implicit)" to match this
//                              session's GEVM performance-plot convention
//                              once the genuinely-tuned, safe fast-decode
//                              kernel above took the "(fast)" name.
//   TOMOS FP32/BF16         = mv_fp32_bf16_pure_bf16_transpose_
//                              atomiconly_scaled_mblock_vec4
//
// All of the fp32_16/bf16-packed series confirmed (by reading each
// kernel's own __global__ body in fp32_16_cuda_kernels.cu) to accumulate
// their row-sum in __half2/__nv_bfloat162 internally -- same tight,
// K-dependent (here M-dependent, since TRANSPOSE reduces over M -- see
// below) accumulator-overflow bound as this file's NO_TRANSPOSE sibling's
// own fp16-accumulating series, not the much wider bound the "recon"
// (float-accumulating) tier gets. See scripts/benchmarks_gemv_transpose_
// fp32_16_pure_half_l2_error_vs_cublas_ref.sh's own SCALES comment for the
// exact numbers.
//
// TRANSPOSE swaps which dimension is reduced over vs NO_TRANSPOSE: y[k] =
// sum_m A[m,k]*x[m], so x has length M (not K) and y has length K (not
// M), and the accumulated row-sum here runs over M terms, not K -- the
// fp16-accumulator safety margin below is therefore keyed to this sweep's
// largest COMPLETING M, not K (same memory footprint as the NO_TRANSPOSE
// tier, so the same shape -- M=K=65536 -- is the largest that doesn't OOM
// before M=K=131072 ever completes; verified against this file's own
// SQUARE_MS list, identical to the NO_TRANSPOSE tier's).
//
// TOMOS FP16 and TOMOS FP32/BF16 share the SAME plain float32 ground
// truth / cublasSgemv(CUBLAS_OP_T) reference (both decode unconditionally
// safe on arbitrary data -- fp32_16_half_only and the bf16 truncation
// decode, respectively; unlike this tier's NO_TRANSPOSE sibling, there is
// no separate plain "TOMOS BF16" series plotted here to also fold into
// this block -- see the METHOD_MARKERS/`series` cross-check above).
//
// x (float, length M) is never compressed by any of these methods, so a
// SINGLE shared x is generated once per shape and reused across all four
// series -- left permanently UNSCALED by normal_scale, same reasoning as
// the NO_TRANSPOSE sibling's own run_shape comment (L2 relative error is
// invariant to uniformly rescaling x, since none of these kernels
// compress x, only A).
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST but only over the OUTPUT vector y (length K) -- never the M*K
// matrix.
//
// Usage: mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref [normal_scale [M]]
//   No args: runs the same SQUARE_MS sweep as mv_fp32_16_pure_half_
//            transpose_vs_cublas_ref.cu's own square sweep (1024 through
//            131072), at normal_scale=1.0.
//   normal_scale: runs the full SQUARE_MS sweep with every method's own
//            "normal"-range data widened to [-scale,scale] instead. See
//            scripts/benchmarks_gemv_transpose_fp32_16_pure_half_l2_
//            error_vs_cublas_ref.sh for the actual scale tiers swept.
//   normal_scale M: runs just that one square shape (M=K=M) at that scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_transpose_fp32_16_
// pure_half_l2_error_vs_cublas_ref.sh.

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

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size); }  // namespace experimental

void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                         dim<2> size, const uint16_t* wh,
                                                                         size_t ld_a, const float* x, size_t ld_x,
                                                                         double beta, float* y, size_t ld_y,
                                                                         float* workspace);
void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(TransposeType trans, double alpha,
                                                                        dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                        const float* x, size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);
// Genuine safe-decode-sibling fast kernel (fp32_16_half_only_fast, safe on
// arbitrary data -- unlike the ...vec4_fast kernel above, which requires
// its own magnitude-restricted [0.5,1.0] generator). Shares the safe
// vec4_split kernel's own workspace-size function (decode-independent),
// so no separate _workspace_size declaration is needed here.
void mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              size_t ld_a, const float* x,
                                                                              size_t ld_x, double beta, float* y,
                                                                              size_t ld_y, float* workspace);
void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, size_t ld_x, double beta,
                                                                       float* y, size_t ld_y);
void mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
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

// Copied verbatim from mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// pseudo_rand_safe_normal (see that file's own header for the fp32_16_
// fast_half "always control=0" contract this must uphold).
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
}

// Copied verbatim from mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// (already-corrected) pseudo_rand_hybrid_mixed -- see that file's own
// comment for the real magnitude-range bug this fixed and why [70000,
// 4000000] is the correct encodable-extended-range window.
__device__ float pseudo_rand_hybrid_mixed(size_t idx, float normal_scale) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    if ((s % 10u) == 0u) {
        uint32_t s2 = s * 2246822519u + 3266489917u;
        s2 ^= s2 << 13;
        s2 ^= s2 >> 17;
        s2 ^= s2 << 5;
        float mag = 70000.0f + (static_cast<float>(s2 % 100001u) / 100000.0f) * 3930000.0f;
        return (s & 0x10000u) ? mag : -mag;
    }
    return normal_scale * ((static_cast<float>(s % 200001u) / 100000.0f) - 1.0f);  // [-scale, scale]
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

__global__ void gen_float_hybrid_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand_hybrid_mixed(i, scale);
}
void gen_float_hybrid(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_hybrid_kernel<<<blocks, threads>>>(n, out, scale);
}

__global__ void gen_float_safe_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * pseudo_rand_safe_normal(i);
}
void gen_float_safe(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_safe_kernel<<<blocks, threads>>>(n, out, scale);
}

// wr is written but never read by any kernel here (every kernel here only
// ever consumes wh) -- fp32_16_split/fp32_bf16_split still require a wr
// output parameter, so a scratch buffer is allocated, filled, and freed
// immediately, same convention this file's NO_TRANSPOSE sibling uses.
__global__ void gen_fp32_16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

__global__ void gen_fp32_16_wh_hybrid_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_hybrid_mixed(i, scale), wh[i], wr[i]);
}
void gen_fp32_16_wh_hybrid(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_hybrid_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

// scale is intended to be a power of 2 -- same fp32_16_fast_half "always
// control=0" contract as this file's NO_TRANSPOSE sibling's own
// gen_fp32_16_wh_safe_kernel.
__global__ void gen_fp32_16_wh_safe_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand_safe_normal(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh_safe(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_safe_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

__global__ void gen_fp32_bf16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(scale * pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_bf16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

// __half/__nv_bfloat16 generators for the cuBLAS FP16/BF16 baselines
// added below -- SAME pseudo_rand(idx) call, SAME idx, SAME scale as
// gen_float above, so these are index-matched to the plain-tier's own
// A_plain/x float32 ground truth, not a converted copy of any TOMOS
// series' own data. Mirrors this file's NO_TRANSPOSE sibling's own
// gen_half/gen_bf16.
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

// Device-side __half/__nv_bfloat16 -> float conversion for cuBLAS FP16/
// BF16's own output vector y (length K here, TRANSPOSE convention) --
// done on-device, matching this file's own "never touch the host except
// for the final L2 error reduction" convention.
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
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||cpu_ref - gpu||_2 / ||cpu_ref||_2, computed on the host over the
// OUTPUT vector only (length K here, TRANSPOSE convention).
double l2_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// x has length M here (TRANSPOSE: y[k] = sum_m A[m,k]*x[m]), y has length
// K -- swapped from the NO_TRANSPOSE convention, same as every other
// TRANSPOSE file in this codebase. A stored M x K column-major throughout
// (lda=m), matching mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu's own
// convention. Reduction is over M (not K) here, so normalized mode scales
// x by 1/sqrt(m) -- see mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// run_shape comment (this file's NO_TRANSPOSE sibling) for the full
// reasoning: all four of this tier's kernels accumulate in __half2
// internally, summing RAW products before alpha is ever applied, so only
// a data-generation-time scale (not alpha) can protect that accumulator.
void run_shape(size_t m, size_t k, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);
    std::printf("  M=%zu K=%zu\n", m, k);

    int lda = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    dim<2> size{m, k};

    float x_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(m)) : 1.0f;
    float* x = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&x), m * sizeof(float), "x", shape_desc)) return;
    gen_float(m, x, x_scale);

    // ---- TOMOS FP16 and (reusing the same plain float32 ground truth
    // and cublasSgemv(CUBLAS_OP_T) reference) TOMOS FP32/BF16 -- both
    // decode unconditionally safe on arbitrary data. ----
    {
        float *A_plain = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_plain), m * k * sizeof(float), "A_plain", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_plain", shape_desc)) {
            gen_float(m * k, A_plain, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_plain, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_fp16 = nullptr, *wr_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_fp16), m * k * sizeof(uint16_t), "wh_fp16", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_scratch), m * k * sizeof(uint16_t), "wr_scratch",
                              shape_desc)) {
                gen_fp32_16_wh(m * k, wh_fp16, wr_scratch, normal_scale);

                size_t split_ws_elems =
                    cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(size);
                float* split_ws = nullptr;
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&split_ws), split_ws_elems * sizeof(float),
                                  "pure_half_transpose_vec4_split (ws)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fp16", shape_desc)) {
                    cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(
                        TransposeType::TRANSPOSE, alpha, size, wh_fp16, static_cast<size_t>(lda), x, 1, beta, y_gpu,
                        1, split_ws);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }

                // ---- TOMOS FP16 (fast): genuine fast-decode sibling of
                // TOMOS FP16 above (fp32_16_half_only_fast instead of
                // fp32_16_half_only) -- safe on arbitrary data, so reuses
                // the SAME wh_fp16/y_ref_h/split_ws as TOMOS FP16, no
                // separate magnitude-restricted generator needed (unlike
                // the OLD "tomos_fp32_fp16_implicit" series below, which
                // predates this decode and still needs its own [0.5,1.0]-
                // restricted data). ----
                float* y_gpu_fast = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_fast), k * sizeof(float), "y_gpu_fp16_fast",
                                  shape_desc)) {
                    cuda::mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split(
                        TransposeType::TRANSPOSE, alpha, size, wh_fp16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu_fast, 1, split_ws);
                    std::vector<float> y_gpu_fast_h(k);
                    cudaMemcpy(y_gpu_fast_h.data(), y_gpu_fast, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_fast          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_fast_h));
                    cudaFree(y_gpu_fast);
                }

                if (split_ws) cudaFree(split_ws);
                cudaFree(wr_scratch);
                cudaFree(wh_fp16);
            }

            uint16_t *wh_bf16 = nullptr, *wr_bf16_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_scratch), m * k * sizeof(uint16_t), "wr_bf16_scratch",
                              shape_desc)) {
                gen_fp32_bf16_wh(m * k, wh_bf16, wr_bf16_scratch, normal_scale);
                float* y_gpu_bf16 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_bf16), k * sizeof(float), "y_gpu_bf16",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(
                        TransposeType::TRANSPOSE, alpha, size, wh_bf16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu_bf16, 1);
                    std::vector<float> y_gpu_bf16_h(k);
                    cudaMemcpy(y_gpu_bf16_h.data(), y_gpu_bf16, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_bf16_h));
                    cudaFree(y_gpu_bf16);
                }
                cudaFree(wr_bf16_scratch);
                cudaFree(wh_bf16);
            }

            // ---- cuBLAS FP16 (Hgemm, n=1, TRANSPOSE): own error vs the
            // SAME plain float32 reference as TOMOS FP16/BF16 above --
            // independent __half-typed A_h/x_h, index-matched to A_plain/x
            // via the SAME pseudo_rand(idx)/scale. alpha=1.0/beta=0.0, same
            // as every other method here. Dimensions/CUBLAS_OP_T convention
            // match mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu's own
            // cublasHgemm leg exactly (y[k] = alpha*A^T*x[m]). ----
            {
                __half *A_h = nullptr, *x_h = nullptr, *y_h = nullptr;
                float* y_h_f = nullptr;
                bool ok_h =
                    alloc_or_skip(reinterpret_cast<void**>(&A_h), m * k * sizeof(__half), "cublas_fp16 (A_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_h), m * sizeof(__half), "cublas_fp16 (x_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_h), k * sizeof(__half), "cublas_fp16 (y_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_h_f), k * sizeof(float), "cublas_fp16 (y_h_f)",
                                 shape_desc);
                if (ok_h) {
                    gen_half(m * k, A_h, normal_scale);
                    gen_half(m, x_h, x_scale);
                    cudaMemset(y_h, 0, k * sizeof(__half));
                    __half h_alpha = __float2half(static_cast<float>(alpha));
                    __half h_beta = __float2half(static_cast<float>(beta));
                    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(k), 1, static_cast<int>(m),
                               &h_alpha, A_h, lda, x_h, static_cast<int>(m), &h_beta, y_h, static_cast<int>(k));
                    half_to_float(k, y_h, y_h_f);
                    std::vector<float> y_h_host(k);
                    cudaMemcpy(y_h_host.data(), y_h_f, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cublas_fp16              = %.6e\n", l2_rel_err(y_ref_h, y_h_host));
                }
                cudaFree(y_h_f);
                cudaFree(y_h);
                cudaFree(x_h);
                cudaFree(A_h);
            }

            // ---- cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1, TRANSPOSE): own
            // error vs the SAME plain float32 reference, same "no native
            // Bgemv" reasoning as cublasHgemm above. Independent
            // __nv_bfloat16-typed A_bf/x_bf, alpha/beta as float
            // (CUBLAS_COMPUTE_32F's own precision, not bf16). ----
            {
                __nv_bfloat16 *A_bf = nullptr, *x_bf = nullptr, *y_bf = nullptr;
                float* y_bf_f = nullptr;
                bool ok_bf =
                    alloc_or_skip(reinterpret_cast<void**>(&A_bf), m * k * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (A_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_bf), m * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (x_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf), k * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (y_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf_f), k * sizeof(float), "cublas_bf16 (y_bf_f)",
                                 shape_desc);
                if (ok_bf) {
                    gen_bf16(m * k, A_bf, normal_scale);
                    gen_bf16(m, x_bf, x_scale);
                    cudaMemset(y_bf, 0, k * sizeof(__nv_bfloat16));
                    float alpha_f = static_cast<float>(alpha), beta_f = static_cast<float>(beta);
                    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(k), 1, static_cast<int>(m),
                                &alpha_f, A_bf, CUDA_R_16BF, lda, x_bf, CUDA_R_16BF, static_cast<int>(m), &beta_f,
                                y_bf, CUDA_R_16BF, static_cast<int>(k), CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    bf16_to_float(k, y_bf, y_bf_f);
                    std::vector<float> y_bf_host(k);
                    cudaMemcpy(y_bf_host.data(), y_bf_f, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cublas_bf16              = %.6e\n", l2_rel_err(y_ref_h, y_bf_host));
                }
                cudaFree(y_bf_f);
                cudaFree(y_bf);
                cudaFree(x_bf);
                cudaFree(A_bf);
            }

            cudaFree(y_ref);
            cudaFree(A_plain);
        }
    }

    // ---- TOMOS FP32/FP16: own mixed normal/extended-range ground truth
    // and reference. ----
    {
        float *A_hybrid = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_hybrid), m * k * sizeof(float), "A_hybrid", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_hybrid", shape_desc)) {
            gen_float_hybrid(m * k, A_hybrid, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_hybrid, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_hybrid = nullptr, *wr_hybrid = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid), m * k * sizeof(uint16_t), "wh_hybrid",
                              shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid), m * k * sizeof(uint16_t), "wr_hybrid",
                              shape_desc)) {
                gen_fp32_16_wh_hybrid(m * k, wh_hybrid, wr_hybrid, normal_scale);
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_hybrid", shape_desc)) {
                    cuda::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(
                        TransposeType::TRANSPOSE, alpha, size, wh_hybrid, static_cast<size_t>(lda), x, 1, beta, y_gpu,
                        1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(wr_hybrid);
                cudaFree(wh_hybrid);
            }
            cudaFree(y_ref);
            cudaFree(A_hybrid);
        }
    }

    // ---- TOMOS FP32/FP16 (implicit): own magnitude-restricted [0.5,1.0]
    // ground truth and reference -- UNSAFE on arbitrary data (unconditional
    // fp32_16_fast_half bit-reinterpret, requires control=0 always). Named
    // "(implicit)" not "(fast)" to match this session's own GEVM
    // performance-plot renaming (the genuinely-tuned, safe fast-decode
    // kernel now owns the "(fast)" name -- see TOMOS FP16 (fast) above). ----
    {
        float *A_safe = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_safe), m * k * sizeof(float), "A_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_safe", shape_desc)) {
            gen_float_safe(m * k, A_safe, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_safe, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_safe = nullptr, *wr_safe_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_safe), m * k * sizeof(uint16_t), "wh_safe", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_safe_scratch), m * k * sizeof(uint16_t), "wr_safe_scratch",
                              shape_desc)) {
                gen_fp32_16_wh_safe(m * k, wh_safe, wr_safe_scratch, normal_scale);
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fasthalf",
                                  shape_desc)) {
                    cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(
                        TransposeType::TRANSPOSE, alpha, size, wh_safe, static_cast<size_t>(lda), x, 1, beta, y_gpu,
                        1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16_implicit = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(wr_safe_scratch);
                cudaFree(wh_safe);
            }
            cudaFree(y_ref);
            cudaFree(A_safe);
        }
    }

    cudaFree(x);
}

}  // namespace

int main(int argc, char** argv) {
    // Extract --normalized (a bare flag, may appear anywhere in argv)
    // before this file's own original positional argument parsing below
    // runs -- see run_shape's own comment for what it does.
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

    // normal_scale (first arg, default 1.0) multiplies every method's own
    // "normal"-range data. All four series here accumulate their row-sum
    // in fp16/bf16 precision internally (see this file's own header), so
    // the SAME tight, empirically-derived bound as this file's
    // NO_TRANSPOSE sibling applies -- see scripts/benchmarks_gemv_
    // transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh's own SCALES
    // comment for the real numbers at this sweep's largest completing
    // shape (M=K=65536).
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== TRANSPOSE GEMV: 16-bit-compute L2 relative error vs FP32 (cublasSgemv CUBLAS_OP_T) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
                                              16384, 24576, 32768, 49152, 65536, 131072};

    if (argc > 2) {
        size_t m = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
        run_shape(m, m, handle, normal_scale, normalized);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t m : square_sizes) run_shape(m, m, handle, normal_scale, normalized);

    cublasDestroy(handle);
    return 0;
}

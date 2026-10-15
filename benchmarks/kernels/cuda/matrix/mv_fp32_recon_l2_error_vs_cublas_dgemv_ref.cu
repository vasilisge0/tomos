// Standalone L2-relative-error analysis for the fp32-reconstruction-tier
// GEVM (five "TOMOS" series -- same kernels as mv_fp32_recon_l2_error_vs_
// cublas_ref.cu, this file's own FP32-reference sibling), but measured
// against a TRUE FP64 reference instead of that sibling's FP32
// (cublasSgemv) one.
//
// Why FP64 instead of FP32: that sibling's own "TOMOS FP32" series is
// nearly a no-op comparison against a cublasSgemv reference -- both run
// at float precision with the same rounding model, so that series' own
// error stays near machine epsilon regardless of normal_scale, never
// actually showing the real accumulated rounding error a large float
// GEMV reduction produces. Referencing against genuine FP64 instead
// exposes that real error directly, and gives the four packed-format
// series' own ADDITIONAL error (from A's own lossy 16-bit encoding) a
// meaningful float-accumulation baseline to be measured against, rather
// than being measured against a reference sharing the same rounding
// error source. Same fix, same reasoning as mm_fp32_compute32_l2_error_
// vs_cublas_dgemm_ref.cu already applied to the GEMM analog of this tier.
//
// Five series, identical kernels/encodings to mv_fp32_recon_l2_error_vs_
// cublas_ref.cu's own set -- full parity, no subset:
//
//   TOMOS FP32                = mv_fp32_tiled
//   TOMOS FP32/FP16            = mv_fp32_16_accurate_ready4_atomiconly_
//                       scaled_kblock -- A packed via fp32_16_split
//                       (plain), safe on arbitrary data.
//   TOMOS FP32/FP16 (fast)     = mv_fp32_16_accurate_ready_atomiconly_
//                       scaled_kblock_fast_normal -- UNSAFE on arbitrary
//                       data, own magnitude-restricted [0.5,1.0)*scale
//                       ground truth.
//   TOMOS FP32/BF16            = mv_fp32_bf16_accurate_ready_atomiconly_
//                       scaled_kblock_rounded -- A packed via fp32_bf16_
//                       split_rounded.
//   TOMOS FP32/BF16 (workspace) = mv_fp32_bf16_accurate_ready_
//                       atomiconly_scaled_kblock_workspace -- A packed
//                       via plain fp32_bf16_split.
//
// Unlike mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu (ONE A/B pair
// and ONE reference shared across all seven series), each series here
// generates its OWN float32 ground-truth array and, in turn, its OWN FP64
// reference -- mirroring mv_fp32_recon_l2_error_vs_cublas_ref.cu's own
// per-series structure exactly (some series use pseudo_rand, "(fast)"
// uses the magnitude-restricted pseudo_rand_safe_normal instead, so a
// single shared reference across all five would silently test the wrong
// thing for that series). x, however, IS shared and widened to double
// ONCE (declared before any series block, freed at the very end) -- it is
// the SAME array for every series regardless of which A generator that
// series uses.
//
// FP64 reference: each series' own float32 A is WIDENED to double (exact,
// lossless), x is widened once and reused, then a single cublasDgemv call
// per series produces that series' own double-precision y_ref. This is
// deliberately NOT a from-scratch double-precision data generation -- the
// point is "how much does each method's own float-precision compute/
// encoding cost, relative to carrying the SAME starting values through in
// full double precision," same framing as the GEMM analog.
//
// Memory note: each series' own peak now needs FOUR concurrently-alive
// M*K-sized buffers instead of TWO -- float A (4 bytes/elem), double A_d
// (8 bytes/elem), plus (for the four packed-format series) wh/wr (2+2
// bytes/elem) -- 16 bytes/element at peak for those, vs the FP32-
// reference sibling's own 8 bytes/element (float A + wh + wr). That
// sibling's own sweep goes up to 131072 (float-only, ~68GB for a single
// array, already SKIPping near there) -- doubling the per-element cost
// here would reach that same ceiling at a much smaller N. This file's own
// square_sizes is capped at 24576 instead, the SAME conservative ceiling
// mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu already uses for
// exactly this reason (16*24576^2 ~ 9.7GB peak per series, comfortable
// margin on a single 40GB GPU) -- see that file's own header for the
// derivation this one reuses. Reference allocation still goes through
// alloc_or_skip like every other buffer; if it fails, that series prints
// an explicit SKIPPED line (not a silent zero-error row).
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST in double precision, over the OUTPUT vector only (length M) --
// y_ref itself is already double (the genuine FP64 result, never
// downcast), each method's own float y is widened to double only inside
// this per-element comparison.
//
// Usage: mv_fp32_recon_l2_error_vs_cublas_dgemv_ref [normal_scale [M]] [--normalized]
//   No args: runs the default square sweep (1024 through 24576, M=K) at
//            normal_scale=1.0.
//   normal_scale: runs the full sweep with every method's own "normal"-
//            range data widened to [-scale,scale] instead -- power of 2,
//            kept well under the 65536 normal/extended encoding boundary
//            (see mv_fp32_recon_l2_error_vs_cublas_ref.cu's own header).
//   normal_scale M: runs just that one square shape (M=K=M) at that scale.
//   --normalized: scales x by 1/sqrt(K) instead of leaving it at plain
//            1.0 -- see run_shape's own comment. Added for API
//            consistency with every other error benchmark in this
//            codebase; none of these five kernels accumulate in fp16/
//            bf16 internally (confirmed float-accumulating, see mv_fp32_
//            recon_l2_error_vs_cublas_ref.cu's own header), so there is
//            no accumulator-overflow bug for this flag to fix here
//            specifically.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_fp32_recon_scale_
// sweep_l2_error_vs_cublas_dgemv_ref.sh.

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
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y);
void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y);
void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                   const float* x, size_t ld_x, double beta, float* y,
                                                                   size_t ld_y, double* workspace);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y, double* workspace);
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

// Copied verbatim from mv_fp32_recon_l2_error_vs_cublas_ref.cu's own
// pseudo_rand_safe_normal -- see that file's own header for the fp32_16_
// fast_half "always control=0" contract this must uphold.
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
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

__global__ void gen_fp32_16_safe_normal_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand_safe_normal(i), wh[i], wr[i]);
}
void gen_fp32_16_safe_normal(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_safe_normal_kernel<<<blocks, threads>>>(n, wh, wr, scale);
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

__global__ void gen_fp32_bf16_rounded_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split_rounded(scale * pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16_rounded(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_rounded_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

// Exact, lossless widening (every float value is exactly representable in
// double) -- used both for x (shared, widened once) and for each series'
// own A array (widened inside its own block).
__global__ void widen_to_double_kernel(size_t n, const float* in, double* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<double>(in[i]);
}
void widen_to_double(size_t n, const float* in, double* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    widen_to_double_kernel<<<blocks, threads>>>(n, in, out);
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||y_ref - y_gpu||_2 / ||y_ref||_2 -- ref is the genuine FP64 result
// (never downcast), gpu is each method's own float output, widened to
// double only inside this comparison.
double l2_rel_err(const std::vector<double>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = ref[i] - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += ref[i] * ref[i];
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// normalized: when true, x is generated at scale 1/sqrt(k) instead of the
// plain 1.0 -- same reasoning as mv_fp32_recon_l2_error_vs_cublas_ref.cu's
// own run_shape comment. x is widened to double ONCE here (x_d), shared
// across every series' own cublasDgemv call below -- it does not depend
// on which A generator a given series uses.
void run_shape(size_t m, size_t k, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);
    std::printf("  M=%zu K=%zu\n", m, k);

    int lda = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    dim<2> size{m, k};

    float x_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float* x = nullptr;
    double* x_d = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&x), k * sizeof(float), "x", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&x_d), k * sizeof(double), "x_double", shape_desc)) {
        cudaFree(x);
        cudaFree(x_d);
        return;
    }
    gen_float(k, x, x_scale);
    widen_to_double(k, x, x_d);

    // ---- TOMOS FP32 (mv_fp32_tiled): plain float32 throughout, no 16-
    // bit storage -- its own reference is now the TRUE FP64 result on the
    // SAME A_native array widened to double, so this series' own error is
    // finally a real float-GEMV-accumulation measurement, not a sanity
    // check near machine epsilon. ----
    {
        float* A_native = nullptr;
        double* A_native_d = nullptr;
        double* y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "A_native", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_native_d), m * k * sizeof(double), "A_native_double",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), m * sizeof(double), "y_ref_native_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_native, normal_scale);
            widen_to_double(m * k, A_native, A_native_d);
            cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &alpha, A_native_d, lda, x_d,
                       1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref_d, m * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_native", shape_desc)) {
                cuda::mv_fp32_tiled(TransposeType::NO_TRANSPOSE, alpha, size, A_native, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }

            // ---- cuBLAS FP32: cublasSgemv on the SAME A_native/x every
            // other series in this block shares -- the "industry
            // standard" FP32 GEMV baseline, to see whether TOMOS FP32's
            // own error against the true FP64 reference is comparable to,
            // better, or worse than cuBLAS's own FP32 implementation. ----
            {
                float* y_gpu_cublas = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_cublas), m * sizeof(float), "y_gpu_cublas_fp32",
                                  shape_desc)) {
                    float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                    cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native,
                               lda, x, 1, &f_beta, y_gpu_cublas, 1);
                    std::vector<float> y_gpu_cublas_h(m);
                    cudaMemcpy(y_gpu_cublas_h.data(), y_gpu_cublas, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cublas_fp32              = %.6e\n", l2_rel_err(y_ref_h, y_gpu_cublas_h));
                    cudaFree(y_gpu_cublas);
                }
            }
        } else {
            std::printf("    tomos_fp32               SKIPPED (no FP64 reference at %s)\n", shape_desc);
            std::printf("    cublas_fp32              SKIPPED (no FP64 reference at %s)\n", shape_desc);
        }
        cudaFree(y_ref_d);
        cudaFree(A_native_d);
        cudaFree(A_native);
    }

    // ---- TOMOS FP32/FP16: plain data (safe on arbitrary data). ----
    {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        float* A_fp16_ref = nullptr;
        double *A_fp16_ref_d = nullptr, *y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t), "wh16", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t), "wr16", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_fp16_ref), m * k * sizeof(float), "A_fp16_ref",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_fp16_ref_d), m * k * sizeof(double), "A_fp16_ref_double",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), m * sizeof(double), "y_ref_fp16_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_fp16_ref, normal_scale);
            gen_fp32_16(m * k, wh16, wr16, normal_scale);
            widen_to_double(m * k, A_fp16_ref, A_fp16_ref_d);
            cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &alpha, A_fp16_ref_d, lda, x_d,
                       1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref_d, m * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fp16", shape_desc)) {
                cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, alpha, size,
                                                                            wh16, wr16, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
        } else {
            std::printf("    tomos_fp32_fp16          SKIPPED (no FP64 reference at %s)\n", shape_desc);
        }
        cudaFree(y_ref_d);
        cudaFree(A_fp16_ref_d);
        cudaFree(A_fp16_ref);
        cudaFree(wr16);
        cudaFree(wh16);
    }

    // ---- TOMOS FP32/FP16 (fast): own magnitude-restricted [0.5,1.0)*
    // scale ground truth -- UNSAFE on arbitrary data (see mv_fp32_recon_
    // l2_error_vs_cublas_ref.cu's own header). ----
    {
        uint16_t *wh16_safe = nullptr, *wr16_safe = nullptr;
        float* A_fast_ref = nullptr;
        double *A_fast_ref_d = nullptr, *y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16_safe), m * k * sizeof(uint16_t), "wh16_safe",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&wr16_safe), m * k * sizeof(uint16_t), "wr16_safe",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_fast_ref), m * k * sizeof(float), "A_fast_ref",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_fast_ref_d), m * k * sizeof(double),
                               "A_fast_ref_double", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), m * sizeof(double), "y_ref_fp16_fast_double",
                               shape_desc);
        if (ok) {
            gen_float_safe(m * k, A_fast_ref, normal_scale);
            gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe, normal_scale);
            widen_to_double(m * k, A_fast_ref, A_fast_ref_d);
            cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &alpha, A_fast_ref_d, lda,
                       x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref_d, m * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fp16_fast", shape_desc)) {
                cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(
                    TransposeType::NO_TRANSPOSE, alpha, size, wh16_safe, wr16_safe, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_fast     = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
        } else {
            std::printf("    tomos_fp32_fp16_fast     SKIPPED (no FP64 reference at %s)\n", shape_desc);
        }
        cudaFree(y_ref_d);
        cudaFree(A_fast_ref_d);
        cudaFree(A_fast_ref);
        cudaFree(wr16_safe);
        cudaFree(wh16_safe);
    }

    // ---- TOMOS FP32/BF16: own rounded-encoding ground truth. workspace
    // passed as nullptr -- this benchmark's own NO_TRANSPOSE, ld_a==m
    // shape always dispatches to the fast row-vectorized path, which
    // never reads or writes it (same reasoning as mv_fp32_recon_l2_
    // error_vs_cublas_ref.cu's own comment). ----
    {
        uint16_t *wh_bf16_rounded = nullptr, *wr_bf16_rounded = nullptr;
        float* A_bf16_rounded_ref = nullptr;
        double *A_bf16_rounded_ref_d = nullptr, *y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16_rounded), m * k * sizeof(uint16_t),
                               "wh_bf16_rounded", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_rounded), m * k * sizeof(uint16_t),
                               "wr_bf16_rounded", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_bf16_rounded_ref), m * k * sizeof(float),
                               "A_bf16_rounded_ref", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_bf16_rounded_ref_d), m * k * sizeof(double),
                               "A_bf16_rounded_ref_double", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), m * sizeof(double),
                               "y_ref_bf16_rounded_double", shape_desc);
        if (ok) {
            gen_float(m * k, A_bf16_rounded_ref, normal_scale);
            gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded, normal_scale);
            widen_to_double(m * k, A_bf16_rounded_ref, A_bf16_rounded_ref_d);
            cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &alpha, A_bf16_rounded_ref_d,
                       lda, x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref_d, m * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_bf16_rounded",
                              shape_desc)) {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(
                    TransposeType::NO_TRANSPOSE, alpha, size, wh_bf16_rounded, wr_bf16_rounded, lda, x, 1, beta,
                    y_gpu, 1, nullptr);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_bf16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
        } else {
            std::printf("    tomos_fp32_bf16          SKIPPED (no FP64 reference at %s)\n", shape_desc);
        }
        cudaFree(y_ref_d);
        cudaFree(A_bf16_rounded_ref_d);
        cudaFree(A_bf16_rounded_ref);
        cudaFree(wr_bf16_rounded);
        cudaFree(wh_bf16_rounded);
    }

    // ---- TOMOS FP32/BF16 (workspace): plain (non-rounded) bf16 ground
    // truth -- fp32_bf16_combine needs no unsafe-on-arbitrary-data
    // shortcut. workspace passed as nullptr for the same reason as the
    // rounded leg above. ----
    {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        float* A_bf16_ref = nullptr;
        double *A_bf16_ref_d = nullptr, *y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t), "wr_bf16",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_bf16_ref), m * k * sizeof(float), "A_bf16_ref",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_bf16_ref_d), m * k * sizeof(double), "A_bf16_ref_double",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), m * sizeof(double), "y_ref_bf16_ws_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_bf16_ref, normal_scale);
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16, normal_scale);
            widen_to_double(m * k, A_bf16_ref, A_bf16_ref_d);
            cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &alpha, A_bf16_ref_d, lda,
                       x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref_d, m * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_bf16_ws", shape_desc)) {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
                    TransposeType::NO_TRANSPOSE, alpha, size, wh_bf16, wr_bf16, lda, x, 1, beta, y_gpu, 1, nullptr);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_bf16_workspace = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
        } else {
            std::printf("    tomos_fp32_bf16_workspace SKIPPED (no FP64 reference at %s)\n", shape_desc);
        }
        cudaFree(y_ref_d);
        cudaFree(A_bf16_ref_d);
        cudaFree(A_bf16_ref);
        cudaFree(wr_bf16);
        cudaFree(wh_bf16);
    }

    cudaFree(x_d);
    cudaFree(x);
}

}  // namespace

int main(int argc, char** argv) {
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

    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== NO_TRANSPOSE GEVM: fp32-recon-tier L2 relative error vs TRUE FP64 (cublasDgemv) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576};

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

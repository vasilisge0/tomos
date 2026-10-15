// Standalone L2-relative-error analysis for the five "TOMOS" fp32-
// reconstruction-tier TRANSPOSE GEMV series scripts/plot_gemv_transpose_
// fp32_recon_vs_cublas_ref.m plots, across mv_fp32_transpose_recon_vs_
// cublas_ref.cu's own square sweep (M=K) -- measures how far each
// method's result drifts from a genuine FP32-precision reference, not a
// throughput number. See scripts/plot_gemv_transpose_fp32_recon_l2_
// error_vs_cublas_ref.m for the resulting plot.
//
// NOTE: mv_fp32_transpose_recon_vs_cublas_ref.cu's OWN header comment
// (as of this file's writing) describes an EARLIER promoted-kernel state
// that has since changed -- the actual CURRENTLY-promoted kernels (per
// scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh's own
// METHOD_MARKERS, the ground truth for what's plotted, confirmed against
// that file's own PROMOTED/DEMOTED comments) are TRANSPOSE-specialized
// kernels distinct from the NO_TRANSPOSE recon tier's own (mv_fp32_
// recon_l2_error_vs_cublas_ref.cu's), not the same five symbols called
// with trans=TRANSPOSE the way that stale header describes. Every kernel
// below was cross-checked directly against its own real call site in
// mv_fp32_transpose_recon_vs_cublas_ref.cu, not that file's header prose.
//
// Every kernel measured here accumulates its row-sum in FLOAT (confirmed
// by reading mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
// wrapborrow_kernel's own body: `float accum = 0.f;`), not fp16/
// __half2 -- so there is no K-dependent accumulator-overflow ceiling the
// way mv_fp32_16_l2_error_vs_cublas_ref.cu's own sweep had to work
// around. The binding safety constraint here instead is fp32_16_pack.
// hpp's own NORMAL/EXTENDED encoding boundary (magnitude 65536): see
// mv_fp32_recon_l2_error_vs_cublas_ref.cu's own header for the full
// reasoning (identical here, just TRANSPOSE-shaped kernels).
//
// The reference for each method is cublasSgemv(CUBLAS_OP_T) (real fp32
// GEMV, no storage compression) computed on a genuine float32 copy of
// that method's own underlying data, generated directly on the device --
// never materializing the full M*K matrix on the HOST, same reasoning as
// every other L2-error benchmark in this codebase. Device memory is
// still finite -- alloc_or_skip below gracefully SKIPs any method whose
// data doesn't fit at the largest shapes rather than crashing.
//
// A is stored MxK, column-major -- for TRANSPOSE, x has length M (the
// reduction dimension) and y has length K (the output dimension), the
// OPPOSITE of NO_TRANSPOSE's own x-length-K/y-length-M convention (see
// mv_fp32_transpose_recon_vs_cublas_ref.cu's own header for why).
//
// TOMOS FP32 (mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec)
// computes in plain float32 throughout, with no 16-bit storage at all --
// its own L2 error vs cublasSgemv (on the SAME underlying float32 array,
// not a separately-generated copy) should stay near machine epsilon
// regardless of normal_scale, a sanity check on this file's own
// methodology.
//
// pseudo_rand (and its safe-normal sibling) is a pure function of the
// linear index alone -- so a plain gen_float and gen_fp32_16/gen_fp32_
// bf16/gen_fp32_bf16_rounded called with the SAME n are guaranteed to
// correspond to the same underlying "true" values at every index,
// without ever decoding wh/wr back to float. Every generator here is
// copied from mv_fp32_transpose_recon_vs_cublas_ref.cu's own (standalone
// self-contained binaries, no shared translation unit between them).
//
// x (float, length M here) is never compressed by any of these methods,
// so a SINGLE shared x is used across all five, generated once per shape
// and deliberately NOT scaled by normal_scale (L2 relative error is
// invariant to uniformly rescaling x). alpha=1.0, beta=0.0.
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST but only over the OUTPUT vector y (length K here) -- never
// the M*K matrix.
//
// Usage: mv_fp32_transpose_recon_l2_error_vs_cublas_ref [normal_scale [M]]
//   No args: runs the same square sweep as mv_fp32_transpose_recon_vs_
//            cublas_ref.cu's own square_sizes (1024 through 131072), at
//            normal_scale=1.0.
//   normal_scale: runs the full sweep with every method's own data
//            widened to [-scale,scale] instead -- power of 2, kept well
//            under the 65536 normal/extended encoding boundary.
//   normal_scale M: runs just that one square shape (M=K=M) at that
//            scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_transpose_fp32_
// recon_l2_error_vs_cublas_ref.sh.

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
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              const uint16_t* wr, size_t ld_a,
                                                                              const float* x, size_t ld_x,
                                                                              double beta, float* y, size_t ld_y);
void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(TransposeType trans,
                                                                                          double alpha, dim<2> size,
                                                                                          const uint16_t* wh,
                                                                                          const uint16_t* wr,
                                                                                          size_t ld_a, const float* x,
                                                                                          size_t ld_x, double beta,
                                                                                          float* y, size_t ld_y);
void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                            const float* x, size_t ld_x, double beta, float* y,
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

// Safe-normal-range variant, matching mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_wrapborrow_vec4_dispatch's own contract --
// copied verbatim from mv_fp32_transpose_recon_vs_cublas_ref.cu's own
// pseudo_rand_safe_normal. normal_scale must stay well under 65536 for
// the SAME reason as mv_fp32_recon_l2_error_vs_cublas_ref.cu's own
// sibling generator.
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

// Float32 ground-truth sibling of gen_fp32_16_safe_normal above -- same
// idx, so guaranteed to correspond to the SAME underlying values.
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

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

double l2_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// x has length M, y has length K here (TRANSPOSE's own convention -- see
// this file's own header for why). Reduction is over M (not K), so
// normalized mode scales x by 1/sqrt(m), not 1/sqrt(k) -- see mv_fp32_
// recon_l2_error_vs_cublas_ref.cu's own run_shape comment (this file's
// NO_TRANSPOSE sibling) for the full reasoning on why this is applied at
// data-generation time, not via alpha.
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

    // ---- TOMOS FP32 (mv_fp32_transpose_atomiconly_scaled_mblock_vec4_
    // stagevec): plain float32 throughout -- its own reference is
    // cublasSgemv on the SAME A_native array, sanity-checking this file's
    // own methodology (expected near machine epsilon). ----
    {
        float *A_native = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "A_native", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_native", shape_desc)) {
            gen_float(m * k, A_native, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_native", shape_desc)) {
                cuda::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType::TRANSPOSE, alpha, size,
                                                                                A_native, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
            cudaFree(y_ref);
            cudaFree(A_native);
        }
    }

    // ---- TOMOS FP32/FP16: plain data (safe on arbitrary data). ----
    {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t), "wh16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t), "wr16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_fp16", shape_desc)) {
            float* A_fp16_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_fp16_ref), m * k * sizeof(float), "A_fp16_ref",
                              shape_desc)) {
                gen_float(m * k, A_fp16_ref, normal_scale);
                gen_fp32_16(m * k, wh16, wr16, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_fp16_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(k);
                cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fp16", shape_desc)) {
                    cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(
                        TransposeType::TRANSPOSE, alpha, size, wh16, wr16, lda, x, 1, beta, y_gpu, 1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(A_fp16_ref);
            }
            cudaFree(y_ref);
        }
        cudaFree(wr16);
        cudaFree(wh16);
    }

    // ---- TOMOS FP32/FP16 (fast): own magnitude-restricted [0.5,1.0)*
    // scale ground truth. ----
    {
        uint16_t *wh16_safe = nullptr, *wr16_safe = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16_safe), m * k * sizeof(uint16_t), "wh16_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16_safe), m * k * sizeof(uint16_t), "wr16_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_fp16_fast", shape_desc)) {
            float* A_fast_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_fast_ref), m * k * sizeof(float), "A_fast_ref",
                              shape_desc)) {
                gen_float_safe(m * k, A_fast_ref, normal_scale);
                gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_fast_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(k);
                cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fp16_fast",
                                  shape_desc)) {
                    cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
                        TransposeType::TRANSPOSE, alpha, size, wh16_safe, wr16_safe, lda, x, 1, beta, y_gpu, 1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16_fast     = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(A_fast_ref);
            }
            cudaFree(y_ref);
        }
        cudaFree(wr16_safe);
        cudaFree(wh16_safe);
    }

    // ---- TOMOS FP32/BF16: own rounded-encoding ground truth. ----
    {
        uint16_t *wh_bf16_rounded = nullptr, *wr_bf16_rounded = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16_rounded), m * k * sizeof(uint16_t), "wh_bf16_rounded",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_rounded), m * k * sizeof(uint16_t), "wr_bf16_rounded",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_bf16_rounded", shape_desc)) {
            float* A_bf16_rounded_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_rounded_ref), m * k * sizeof(float),
                              "A_bf16_rounded_ref", shape_desc)) {
                gen_float(m * k, A_bf16_rounded_ref, normal_scale);
                gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha,
                           A_bf16_rounded_ref, lda, x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(k);
                cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_bf16_rounded",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
                        TransposeType::TRANSPOSE, alpha, size, wh_bf16_rounded, wr_bf16_rounded, lda, x, 1, beta,
                        y_gpu, 1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(A_bf16_rounded_ref);
            }
            cudaFree(y_ref);
        }
        cudaFree(wr_bf16_rounded);
        cudaFree(wh_bf16_rounded);
    }

    // ---- TOMOS FP32/BF16 (truncated): plain (non-rounded) bf16 ground
    // truth. ----
    {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t), "wr_bf16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), k * sizeof(float), "y_ref_bf16_trunc", shape_desc)) {
            float* A_bf16_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_ref), m * k * sizeof(float), "A_bf16_ref",
                              shape_desc)) {
                gen_float(m * k, A_bf16_ref, normal_scale);
                gen_fp32_bf16(m * k, wh_bf16, wr_bf16, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_bf16_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(k);
                cudaMemcpy(y_ref_h.data(), y_ref, k * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_bf16_trunc",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType::TRANSPOSE, alpha, size,
                                                                                wh_bf16, wr_bf16, lda, x, 1, beta,
                                                                                y_gpu, 1);
                    std::vector<float> y_gpu_h(k);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16_truncated = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(A_bf16_ref);
            }
            cudaFree(y_ref);
        }
        cudaFree(wr_bf16);
        cudaFree(wh_bf16);
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

    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== TRANSPOSE GEMV: fp32-recon-tier L2 relative error vs FP32 (cublasSgemv) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

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

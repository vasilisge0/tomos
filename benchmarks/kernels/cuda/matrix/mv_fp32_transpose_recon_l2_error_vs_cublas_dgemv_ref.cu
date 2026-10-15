// Standalone L2-relative-error analysis for the fp32-reconstruction-tier
// TRANSPOSE GEVM (five "TOMOS" series -- same kernels as mv_fp32_
// transpose_recon_l2_error_vs_cublas_ref.cu, this file's own FP32-
// reference sibling), but measured against a TRUE FP64 reference instead
// of that sibling's FP32 (cublasSgemv(CUBLAS_OP_T)) one. TRANSPOSE
// counterpart of mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu -- see that
// file's own header for the full derivation of why FP64 matters here, not
// repeated below (identical reasoning, only the direction differs).
//
// Five series, identical kernels/encodings to mv_fp32_transpose_recon_l2_
// error_vs_cublas_ref.cu's own set -- full parity, no subset:
//
//   TOMOS FP32                 = mv_fp32_transpose_atomiconly_scaled_
//                       mblock_vec4_stagevec
//   TOMOS FP32/FP16             = mv_fp32_16_transpose_atomiconly_
//                       scaled_mblock_vec4_wrapborrow_dispatch
//   TOMOS FP32/FP16 (fast)      = mv_fp32_16_transpose_atomiconly_
//                       scaled_mblock_fast_normal_wrapborrow_vec4_
//                       dispatch -- UNSAFE on arbitrary data, own
//                       magnitude-restricted [0.5,1.0)*scale ground truth.
//   TOMOS FP32/BF16              = mv_fp32_bf16_transpose_atomiconly_
//                       scaled_mblock_vec4_rounded_dispatch
//   TOMOS FP32/BF16 (truncated) = mv_fp32_bf16_transpose_atomiconly_
//                       scaled_mblock_vec4
//
// A is stored MxK, column-major -- for TRANSPOSE, x has length M (the
// reduction dimension) and y has length K (the output dimension), the
// OPPOSITE of NO_TRANSPOSE's own convention -- same as every other
// TRANSPOSE file in this codebase. x is shared across all five series and
// widened to double ONCE (x_d); each series' own A array is generated and
// widened independently, mirroring mv_fp32_recon_l2_error_vs_cublas_
// dgemv_ref.cu's own per-series structure (this tier has no single shared
// A/reference the way the GEMM analog does, since "(fast)" needs its own
// magnitude-restricted ground truth).
//
// Memory note: same 16-bytes/element-at-peak reasoning (float A + double
// A_d + wh + wr, for the four packed-format series) and the same
// conservative square_sizes cap (1024 through 24576) as this file's own
// NO_TRANSPOSE sibling -- see that file's own header for the full
// derivation.
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST in double precision, over the OUTPUT vector only (length K
// here, TRANSPOSE convention) -- y_ref itself is already double, each
// method's own float y widened to double only inside the comparison.
//
// Usage: mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref [normal_scale [M]] [--normalized]
//   Same argument forms as mv_fp32_recon_l2_error_vs_cublas_dgemv_ref's
//   own usage -- see that file's own header.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_transpose_recon_
// scale_sweep_l2_error_vs_cublas_dgemv_ref.sh.

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

double l2_rel_err(const std::vector<double>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = ref[i] - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += ref[i] * ref[i];
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// x has length M, y has length K here (TRANSPOSE's own convention). x is
// widened to double ONCE (x_d), shared across every series below.
// normalized mode scales x by 1/sqrt(m), not 1/sqrt(k) -- reduction is
// over M here, same reasoning as mv_fp32_transpose_recon_l2_error_vs_
// cublas_ref.cu's own run_shape comment.
void run_shape(size_t m, size_t k, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);
    std::printf("  M=%zu K=%zu\n", m, k);

    int lda = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    dim<2> size{m, k};

    float x_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(m)) : 1.0f;
    float* x = nullptr;
    double* x_d = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&x), m * sizeof(float), "x", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&x_d), m * sizeof(double), "x_double", shape_desc)) {
        cudaFree(x);
        cudaFree(x_d);
        return;
    }
    gen_float(m, x, x_scale);
    widen_to_double(m, x, x_d);

    // ---- TOMOS FP32 (mv_fp32_transpose_atomiconly_scaled_mblock_vec4_
    // stagevec): plain float32 throughout -- its own reference is now the
    // TRUE FP64 result. ----
    {
        float* A_native = nullptr;
        double* A_native_d = nullptr;
        double* y_ref_d = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "A_native", shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&A_native_d), m * k * sizeof(double), "A_native_double",
                               shape_desc) &&
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), k * sizeof(double), "y_ref_native_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_native, normal_scale);
            widen_to_double(m * k, A_native, A_native_d);
            cublasDgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &alpha, A_native_d, lda, x_d,
                       1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref_d, k * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_native", shape_desc)) {
                cuda::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType::TRANSPOSE, alpha, size,
                                                                                A_native, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }

            // ---- cuBLAS FP32: cublasSgemv(CUBLAS_OP_T) on the SAME
            // A_native/x every other series in this block shares -- the
            // "industry standard" FP32 GEMV baseline. ----
            {
                float* y_gpu_cublas = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_cublas), k * sizeof(float), "y_gpu_cublas_fp32",
                                  shape_desc)) {
                    float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                    cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native,
                               lda, x, 1, &f_beta, y_gpu_cublas, 1);
                    std::vector<float> y_gpu_cublas_h(k);
                    cudaMemcpy(y_gpu_cublas_h.data(), y_gpu_cublas, k * sizeof(float), cudaMemcpyDeviceToHost);
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
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), k * sizeof(double), "y_ref_fp16_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_fp16_ref, normal_scale);
            gen_fp32_16(m * k, wh16, wr16, normal_scale);
            widen_to_double(m * k, A_fp16_ref, A_fp16_ref_d);
            cublasDgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &alpha, A_fp16_ref_d, lda,
                       x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref_d, k * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fp16", shape_desc)) {
                cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(
                    TransposeType::TRANSPOSE, alpha, size, wh16, wr16, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
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
    // scale ground truth -- UNSAFE on arbitrary data. ----
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
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), k * sizeof(double), "y_ref_fp16_fast_double",
                               shape_desc);
        if (ok) {
            gen_float_safe(m * k, A_fast_ref, normal_scale);
            gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe, normal_scale);
            widen_to_double(m * k, A_fast_ref, A_fast_ref_d);
            cublasDgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &alpha, A_fast_ref_d, lda,
                       x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref_d, k * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_fp16_fast", shape_desc)) {
                cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
                    TransposeType::TRANSPOSE, alpha, size, wh16_safe, wr16_safe, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
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

    // ---- TOMOS FP32/BF16: own rounded-encoding ground truth. ----
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
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), k * sizeof(double),
                               "y_ref_bf16_rounded_double", shape_desc);
        if (ok) {
            gen_float(m * k, A_bf16_rounded_ref, normal_scale);
            gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded, normal_scale);
            widen_to_double(m * k, A_bf16_rounded_ref, A_bf16_rounded_ref_d);
            cublasDgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &alpha, A_bf16_rounded_ref_d,
                       lda, x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref_d, k * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_bf16_rounded",
                              shape_desc)) {
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
                    TransposeType::TRANSPOSE, alpha, size, wh_bf16_rounded, wr_bf16_rounded, lda, x, 1, beta, y_gpu,
                    1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
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

    // ---- TOMOS FP32/BF16 (truncated): plain (non-rounded) bf16 ground
    // truth. ----
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
                  alloc_or_skip(reinterpret_cast<void**>(&y_ref_d), k * sizeof(double), "y_ref_bf16_trunc_double",
                               shape_desc);
        if (ok) {
            gen_float(m * k, A_bf16_ref, normal_scale);
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16, normal_scale);
            widen_to_double(m * k, A_bf16_ref, A_bf16_ref_d);
            cublasDgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &alpha, A_bf16_ref_d, lda,
                       x_d, 1, &beta, y_ref_d, 1);
            std::vector<double> y_ref_h(k);
            cudaMemcpy(y_ref_h.data(), y_ref_d, k * sizeof(double), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), k * sizeof(float), "y_gpu_bf16_trunc",
                              shape_desc)) {
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType::TRANSPOSE, alpha, size,
                                                                            wh_bf16, wr_bf16, lda, x, 1, beta, y_gpu,
                                                                            1);
                std::vector<float> y_gpu_h(k);
                cudaMemcpy(y_gpu_h.data(), y_gpu, k * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_bf16_truncated = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
        } else {
            std::printf("    tomos_fp32_bf16_truncated SKIPPED (no FP64 reference at %s)\n", shape_desc);
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
        "=== TRANSPOSE GEVM: fp32-recon-tier L2 relative error vs TRUE FP64 (cublasDgemv) reference "
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

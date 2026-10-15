// Standalone L2-relative-error analysis for the five "TOMOS" fp32-
// reconstruction-tier GEMV series scripts/plot_gemv_fp32_recon_vs_
// cutlass_ref.m plots, across mv_fp32_recon_vs_cutlass_ref.cu's own
// square sweep (M=K) -- measures how far each method's result drifts
// from a genuine FP32-precision reference, not a throughput number. See
// scripts/plot_gemv_fp32_recon_l2_error_vs_cublas_ref.m for the
// resulting plot.
//
// Unlike the pure_half/hybrid GEMV tier (mv_fp32_16_l2_error_vs_cublas_
// ref.cu), every kernel measured here accumulates its row-sum in FLOAT
// (confirmed by reading mv_fp32_16_ready4_atomiconly_scaled_kblock_
// kernel's own body: `float sum0 = 0.f, ...`), not fp16/__half2 -- so
// there is no K-dependent accumulator-overflow ceiling the way that
// file's own normal_scale sweep had to work around. The binding safety
// constraint here instead is fp32_16_pack.hpp's own NORMAL/EXTENDED
// encoding boundary: mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_
// fast_normal's data (from pseudo_rand_safe_normal, [0.5,1.0) scaled)
// must stay within the NORMAL band (fp32 exponent in [113,142], i.e.
// magnitude in [2^-14, 2^16) ~ [6.1e-5, 65536)) for that kernel's own
// "always control=0" contract to hold -- crossing into EXTENDED/
// DEGRADED territory there would be a real correctness bug, not just a
// precision measurement, the same class of mistake already found and
// fixed once for mv_fp32_16_l2_error_vs_cublas_ref.cu's own hybrid
// generator. normal_scale is kept well under that 65536 boundary.
//
// The reference for each method is cublasSgemv (real fp32 GEMV, no
// storage compression) computed on a genuine float32 copy of that
// method's own underlying data, generated directly on the device --
// never materializing the full M*K matrix on the HOST, which would OOM
// well before this sweep's own largest shape (same reasoning as
// verify_gemv_fp32_16_plot_promoted.cu's own header). Device memory is
// still finite, though -- alloc_or_skip below (same convention
// mv_fp32_recon_vs_cutlass_ref.cu's own performance sweep uses)
// gracefully SKIPs any method whose data doesn't fit at the largest
// shapes (131072x131072 needs ~68GB for a single float32 array) rather
// than crashing.
//
// TOMOS FP32 (mv_fp32_tiled) computes in plain float32 throughout, with
// no 16-bit storage at all -- its own L2 error vs cublasSgemv (on the
// SAME underlying float32 array, not a separately-generated copy) should
// stay near machine epsilon regardless of normal_scale, a useful sanity
// check that this file's own methodology is measuring real per-method
// storage/compute precision loss, not an artifact of the harness itself.
//
// pseudo_rand (and its safe-normal sibling) is a pure function of the
// linear index alone, with no external RNG state -- so a plain gen_float
// and gen_fp32_16/gen_fp32_bf16/gen_fp32_bf16_rounded called with the
// SAME n are guaranteed to correspond to the same underlying "true"
// values at every index, without ever decoding wh/wr back to float.
// Every generator here is copied verbatim from mv_fp32_recon_vs_cutlass_
// ref.cu's own (standalone self-contained binaries, no shared
// translation unit between them -- same convention every *_vs_cublas_
// ref.cu file in this codebase follows).
//
// x (float, length K) is never compressed by any of these methods, so a
// SINGLE shared x is used across all five, generated once per shape and
// deliberately NOT scaled by normal_scale (L2 relative error is
// invariant to uniformly rescaling x, since none of these methods
// compress x, only A -- same reasoning as mv_fp32_16_l2_error_vs_cublas_
// ref.cu's own run_shape). alpha=1.0, beta=0.0 (y = A*x exactly, no
// epilogue scaling to fold into the error analysis).
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST but only over the OUTPUT vector y (length M) -- never the M*K
// matrix, which is why this sweep can reach the same 131072x131072 scale
// the performance sweep does.
//
// Usage: mv_fp32_recon_l2_error_vs_cublas_ref [normal_scale [M]]
//   No args: runs the same square sweep as mv_fp32_recon_vs_cutlass_
//            ref.cu's own square_sizes (1024 through 131072), at
//            normal_scale=1.0 (every method's data in [-1,1], the
//            original/default magnitude range).
//   normal_scale: runs the full sweep with every method's own data
//            widened to [-scale,scale] instead -- intended to be a power
//            of 2, kept well under the 65536 normal/extended encoding
//            boundary (see this file's own header above for why).
//   normal_scale M: runs just that one square shape (M=K=M) at that
//            scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_fp32_recon_l2_
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

// Safe-normal-range variant, matching mv_fp32_16_accurate_ready_
// atomiconly_scaled_kblock_fast_normal's own contract (decodes
// unconditionally as if control were always 0) -- copied verbatim from
// mv_fp32_recon_vs_cutlass_ref.cu's own pseudo_rand_safe_normal.
// normal_scale multiplies this too (see gen_fp32_16_safe_normal_kernel's
// own comment for the resulting safety requirement on normal_scale).
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

// scale is intended to be a power of 2, kept well under 65536: pseudo_
// rand_safe_normal's own [0.5,1.0) output is exactly one binary exponent
// bracket, and multiplying by a power of 2 shifts it to another exponent
// bracket, [0.5*scale, 1.0*scale) -- this MUST stay within fp32_16's own
// NORMAL band (magnitude in [2^-14, 2^16)) for mv_fp32_16_accurate_
// ready_atomiconly_scaled_kblock_fast_normal's own "always control=0"
// contract to hold; scale >= 65536 would silently break it (the same
// class of bug already found and fixed once in mv_fp32_16_l2_error_vs_
// cublas_ref.cu's own hybrid generator).
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
// idx, so guaranteed to correspond to the SAME underlying values as that
// function's own wh/wr output, without ever decoding them back to float.
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

// Same convention mv_fp32_recon_vs_cutlass_ref.cu's own alloc_or_skip
// uses: at the largest shapes (131072x131072 needs ~68GB for a single
// float32 array), a cudaMalloc failure is expected, not a bug -- skip
// that one allocation/method gracefully instead of crashing.
bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||cpu_ref - gpu||_2 / ||cpu_ref||_2, computed on the host over the
// OUTPUT vector only (length M) -- never the M*K matrix. See this file's
// own header for why this is safe at every shape the performance sweep
// itself reaches, unlike a correctness check needing the full matrix on
// host.
double l2_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// normalized: when true, x is generated at scale 1/sqrt(k) instead of the
// plain 1.0 -- since y[i] sums K terms each ~normal_scale*x_scale in
// magnitude, this exactly cancels the sqrt(K) growth a raw (x_scale=1)
// row-sum would otherwise have, so y's own magnitude stays ~normal_scale
// regardless of K. Applied at DATA-GENERATION time (not via alpha, which
// several of this codebase's promoted kernels only apply in their own
// epilogue, AFTER their internal accumulator has already reduced the
// full-magnitude raw products -- scaling alpha down would not protect
// such an accumulator from overflowing mid-reduction; scaling the actual
// x VALUES down does, since the smaller values are what the accumulator
// itself ever sees). See scripts/benchmarks_gemv_fp32_recon_l2_error_vs_
// cublas_ref.sh's own --normalized SCALES comment for the numbers this
// unlocks.
void run_shape(size_t m, size_t k, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);
    std::printf("  M=%zu K=%zu\n", m, k);

    int lda = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    dim<2> size{m, k};

    float x_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float* x = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&x), k * sizeof(float), "x", shape_desc)) return;
    gen_float(k, x, x_scale);

    // ---- TOMOS FP32 (mv_fp32_tiled): plain float32 throughout, no 16-
    // bit storage -- its own reference is cublasSgemv on the SAME
    // A_native array (not a separately-generated copy), so this series'
    // own error is a sanity check on the harness itself (expected near
    // machine epsilon), not a real compression-precision measurement. ----
    {
        float *A_native = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "A_native", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_native", shape_desc)) {
            gen_float(m * k, A_native, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

            float* y_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_native", shape_desc)) {
                cuda::mv_fp32_tiled(TransposeType::NO_TRANSPOSE, alpha, size, A_native, lda, x, 1, beta, y_gpu, 1);
                std::vector<float> y_gpu_h(m);
                cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                cudaFree(y_gpu);
            }
            cudaFree(y_ref);
            cudaFree(A_native);
        }
    }

    // ---- TOMOS FP32/FP16: plain data (safe on arbitrary data, no
    // control=0 assumption). ----
    {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t), "wh16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t), "wr16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_fp16", shape_desc)) {
            // A separate float32 ground-truth array is needed for
            // cublasSgemv (it can't consume wh/wr directly) -- generated
            // with the SAME scale*pseudo_rand(idx) call gen_fp32_16_
            // kernel itself makes, so it corresponds to the same
            // underlying values without ever decoding wh/wr back.
            float* A_fp16_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_fp16_ref), m * k * sizeof(float), "A_fp16_ref",
                              shape_desc)) {
                gen_float(m * k, A_fp16_ref, normal_scale);
                gen_fp32_16(m * k, wh16, wr16, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_fp16_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(m);
                cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fp16", shape_desc)) {
                    cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, alpha,
                                                                               size, wh16, wr16, lda, x, 1, beta,
                                                                               y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
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
    // scale ground truth -- mv_fp32_16_accurate_ready_atomiconly_scaled_
    // kblock_fast_normal is UNSAFE on arbitrary data (see this file's
    // own header). ----
    {
        uint16_t *wh16_safe = nullptr, *wr16_safe = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16_safe), m * k * sizeof(uint16_t), "wh16_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16_safe), m * k * sizeof(uint16_t), "wr16_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_fp16_fast", shape_desc)) {
            float* A_fast_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_fast_ref), m * k * sizeof(float), "A_fast_ref",
                              shape_desc)) {
                gen_float_safe(m * k, A_fast_ref, normal_scale);
                gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_fast_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(m);
                cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fp16_fast",
                                  shape_desc)) {
                    cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh16_safe, wr16_safe, lda, x, 1, beta, y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
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

    // ---- TOMOS FP32/BF16: own rounded-encoding ground truth (fp32_
    // bf16_split_rounded, not the plain truncating split -- the two
    // encodings are not interchangeable, see mv_fp32_bf16_accurate_
    // ready_atomiconly_scaled_kblock_rounded's own comment in fp32_16_
    // cuda_kernels.cu). workspace passed as nullptr: this benchmark's
    // own NO_TRANSPOSE, ld_a==m shape always dispatches to the fast
    // row-vectorized path, which never reads or writes it (confirmed by
    // mv_fp32_recon_vs_cutlass_ref.cu's own direct A/B measurement). ----
    {
        uint16_t *wh_bf16_rounded = nullptr, *wr_bf16_rounded = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16_rounded), m * k * sizeof(uint16_t), "wh_bf16_rounded",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_rounded), m * k * sizeof(uint16_t), "wr_bf16_rounded",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_bf16_rounded", shape_desc)) {
            float* A_bf16_rounded_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_rounded_ref), m * k * sizeof(float),
                              "A_bf16_rounded_ref", shape_desc)) {
                gen_float(m * k, A_bf16_rounded_ref, normal_scale);
                gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha,
                           A_bf16_rounded_ref, lda, x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(m);
                cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

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
                cudaFree(A_bf16_rounded_ref);
            }
            cudaFree(y_ref);
        }
        cudaFree(wr_bf16_rounded);
        cudaFree(wh_bf16_rounded);
    }

    // ---- TOMOS FP32/BF16 (workspace): plain (non-rounded) bf16 ground
    // truth -- fp32_bf16_combine needs no unsafe-on-arbitrary-data
    // shortcut, so this reuses plain gen_fp32_bf16 rather than a
    // dedicated safe-range generator (see mv_fp32_bf16_accurate_ready_
    // atomiconly_scaled_kblock_workspace's own comment in
    // mv_fp32_recon_vs_cutlass_ref.cu). workspace passed as nullptr for
    // the same reason as the rounded leg above. ----
    {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        float* y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t), "wr_bf16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_bf16_ws", shape_desc)) {
            float* A_bf16_ref = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_bf16_ref), m * k * sizeof(float), "A_bf16_ref",
                              shape_desc)) {
                gen_float(m * k, A_bf16_ref, normal_scale);
                gen_fp32_bf16(m * k, wh_bf16, wr_bf16, normal_scale);
                float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
                cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_bf16_ref, lda,
                           x, 1, &f_beta, y_ref, 1);
                std::vector<float> y_ref_h(m);
                cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_bf16_ws",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_bf16, wr_bf16, lda, x, 1, beta, y_gpu, 1,
                        nullptr);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16_workspace = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
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
        "=== NO_TRANSPOSE GEMV: fp32-recon-tier L2 relative error vs FP32 (cublasSgemv) reference "
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

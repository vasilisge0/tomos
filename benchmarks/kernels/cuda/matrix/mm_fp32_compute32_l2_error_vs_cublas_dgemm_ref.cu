// Standalone Frobenius-relative-error analysis for the "fp32-compute"
// tier of GEMM (this file's own name for what mm_fp32_recon_l2_error_vs_
// cublas_ref.cu calls the "full fp32 reconstruction" tier: every series
// below decodes/consumes its A operand into genuine float and
// accumulates in float registers), but measured against a TRUE FP64
// reference instead of that sibling's FP32 (cublasSgemm) one.
//
// Why FP64 instead of FP32: mm_fp32_recon_l2_error_vs_cublas_ref.cu's
// own "TOMOS FP32" series is nearly a no-op comparison against a
// cublasSgemm reference -- both run at float precision with the same
// rounding model, so that series' own error stays near machine epsilon
// regardless of normal_scale, never actually showing the real
// accumulated rounding error an M=K=N-large float GEMM reduction
// produces. Referencing against genuine FP64 instead exposes that real
// error directly, and gives the TOMOS FP32/FP16 and TOMOS FP32/BF16
// series' own ADDITIONAL error (from A's own lossy 16-bit encoding) a
// meaningful float-accumulation baseline to be measured against, rather
// than being measured against a reference sharing the same rounding
// error source.
//
// Nine series -- ALL SIX "TOMOS" series from mm_fp32_recon_l2_error_vs_
// cublas_ref.cu's own full set (same kernels, same encodings, full
// parity -- no representative-only subset), a cuBLAS FP32 baseline that
// sibling file doesn't have at all, and two UNCONDITIONAL fast_normal
// decode variants of the fp32_16 dualbuf/workspace series (not present
// in that sibling file either):
//
//   cuBLAS FP32                = cublasSgemm, on the SAME A_plain/B
//                       every other series here shares -- the "industry
//                       standard" FP32 GEMM, to check whether TOMOS
//                       FP32's own error against the true FP64 reference
//                       is comparable to, better, or worse than cuBLAS's
//                       own FP32 implementation.
//   TOMOS FP32                 = gemm_fp32_tiled_syncpipe4_bk8_
//                       warptile_nostride_snake_midpoint_exact --
//                       consumes A directly, no 16-bit encoding at all.
//   TOMOS FP32/FP16             = mm_fp32_16_tiled_cpasync_dualbuf -- A
//                       packed via fp32_16_split (plain, round-to-
//                       nearest).
//   TOMOS FP32/FP16 (fast)      = mm_fp32_16_tiled_cpasync_dualbuf_
//                       fast_normal -- byte-for-byte identical to TOMOS
//                       FP32/FP16 above except A's decode calls
//                       fp32_16_combine_fast_normal instead of
//                       fp32_16_combine, unconditionally (no per-tile
//                       validity check). UNSAFE ON ARBITRARY DATA: only
//                       correct when every element of A has control bit
//                       (bit0 of wh) == 0 -- this series' generator does
//                       NOT guarantee that (see gen_fp32_16's own use
//                       below), so its own error is EXPECTED to include
//                       real reconstruction mistakes from elements below
//                       fp32_16_split's own control=1 boundary, not just
//                       encoding-precision rounding error. Plotted
//                       specifically to quantify that real-data cost.
//   TOMOS FP32/FP16 (truncated) = mm_fp32_16_tiled_cpasync_dualbuf_
//                       truncated -- A packed via fp32_16_split_
//                       truncated, NOT interchangeable with the plain
//                       wh/wr pair above.
//   TOMOS FP32/FP16 (workspace) = mm_fp32_16_gemm_materialize_
//                       midpoint_exact -- same plain fp32_16_split
//                       encoding as TOMOS FP32/FP16 above, different
//                       materialize strategy only.
//   TOMOS FP32/FP16 (workspace, fast) = mm_fp32_16_gemm_materialize_
//                       midpoint_exact_fast_normal -- the same
//                       unconditional fast_normal decode swap as TOMOS
//                       FP32/FP16 (fast) above, applied to the
//                       materialize/workspace strategy instead of
//                       dualbuf. Same UNSAFE-ON-ARBITRARY-DATA
//                       expectation and rationale.
//   TOMOS FP32/BF16 (truncated) = mm_fp32_bf16_syncpipe4_bk8_
//                       warptile_nostride_snake_midpoint_exact -- A
//                       packed via fp32_bf16_split (plain) -- labeled
//                       "(truncated)" to match the sibling file's own
//                       naming for this kernel, which refers to that
//                       file's own materialize-strategy name, not this
//                       encoding itself.
//   TOMOS FP32/BF16 (workspace) = mm_fp32_bf16_gemm_materialize_rounded
//                       -- A packed via fp32_bf16_split_rounded, NOT
//                       interchangeable with the plain wh/wr pair above.
//
// All nine accumulate in float -- cuBLAS FP32 and TOMOS FP32 directly
// (float register/accumulator, standard for an FP32 GEMM), and the eight
// 16-bit-encoded TOMOS kernels in a float tile-accumulator after
// decoding A (same confirmed-by-reading-the-kernel-body fact mm_fp32_
// recon_l2_error_vs_cublas_ref.cu's own header establishes for the
// family those belong to) -- so C is always genuine float-typed for
// every series here. Unlike mm_fp32_16_variable_rounding_l2_error_vs_
// cublas_ref.cu's own __half-typed C (three of its own series), there is
// NO fp16-output-storage-overflow concern here at any scale used in this
// file's own sweeps.
//
// FP64 reference: A and B are generated as plain float (same pseudo_
// rand-derived data every method's own A/B encoding is ALSO a function
// of), then WIDENED to double (exact, lossless -- float is a strict
// subset of double's own representable values) and run through a single
// cublasDgemm call. This is deliberately NOT a from-scratch double-
// precision data generation -- the point is to ask "how much does each
// method's own float-precision compute/encoding cost, relative to
// carrying the SAME starting values through in full double precision,"
// not "how does the method behave on genuinely double-range data."
//
// Memory note: computing the FP64 reference needs THREE additional
// double-typed buffers (A_double, B_double, C_ref_double) on top of the
// float A/B this file's own series already need -- peak device memory
// during that phase is N^2*(4+4+8+8+8)=32*N^2 bytes. This file's own
// largest shape (M=K=N=24576) keeps that at ~19GB, deliberately sized to
// fit comfortably within a single 40GB GPU with room to spare for CUDA
// context/cuBLAS workspace/the TOMOS series' own buffers afterward --
// unlike mm_fp32_recon_l2_error_vs_cublas_ref.cu's own sweep (which goes
// up to 49152, fine for its all-float buffers but ~77GB at peak for this
// file's own double-widened reference, likely exceeding even an 80GB
// card). Reference allocation still goes through alloc_or_skip like
// every other buffer in this file; if it ever fails anyway, run_shape
// prints an explicit SKIPPED line for all seven methods at that shape
// (not a silent zero-error row) and returns without computing anything
// further for it.
//
// Frobenius relative error: ||C_ref - C_method||_F / ||C_ref||_F,
// computed on the HOST in double precision, over the OUTPUT matrix only
// (size M*N) -- C_ref itself is already double (the genuine FP64 result,
// never downcast), and each method's own float C is widened to double
// only inside this per-element comparison.
//
// Usage: mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref [normal_scale [N]] [--normalized]
//   No args: runs the default square sweep (1024 through 24576, M=K=N)
//            at normal_scale=1.0.
//   normal_scale: runs the full sweep with A's own "normal"-range data
//            widened to [-scale,scale] instead.
//   normal_scale N: runs just that one square shape (M=K=N=N) at that
//            scale.
//   --normalized: scales B by 1/sqrt(K) instead of leaving it at plain
//            [-1,1] -- see run_shape's own comment. Added for API
//            consistency with every other error benchmark in this
//            codebase; C is float-typed here regardless, so there is no
//            output-storage-overflow bug for it to fix in this file
//            specifically.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase.

#include <cublas_v2.h>
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
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha, dim<2> size_A,
                                      const uint16_t* wh, const uint16_t* wr, size_t ld_a, dim<2> size_B,
                                      const float* B, size_t ld_b, double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType transA, TransposeType transB, double alpha,
                                                dim<2> size_A, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                dim<2> size_B, const float* B, size_t ld_b, double beta, float* C,
                                                size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                 dim<2> size_B, const float* B, size_t ld_b, double beta, float* C,
                                                 size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                   double beta, float* C, size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                                              dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                              size_t ld_a, dim<2> size_B, const float* B,
                                                              size_t ld_b, double beta, float* C, size_t ld_c);
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB,
                                                                        double alpha, dim<2> size_A,
                                                                        const uint16_t* wh, const uint16_t* wr,
                                                                        size_t ld_a, dim<2> size_B, const float* B,
                                                                        size_t ld_b, double beta, float* C,
                                                                        size_t ld_c);
void mm_fp32_bf16_gemm_materialize_rounded(TransposeType transA, TransposeType transB, double alpha, dim<2> size_A,
                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a, dim<2> size_B,
                                            const float* B, size_t ld_b, double beta, float* C, size_t ld_c);
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, float* A, size_t ld_a, float* B,
                                                                           size_t ld_b, double beta, float* C,
                                                                           size_t ld_c);
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

// CORRECTNESS CONTRACT: mm_fp32_16_tiled_cpasync_dualbuf_truncated
// requires wh/wr from fp32_16_split_truncated specifically, never
// fp32_16_split -- see fp32_16_combine_truncated's own comment in
// fp32_16_pack.hpp (copied verbatim from mm_fp32_recon_l2_error_vs_
// cublas_ref.cu's own gen_fp32_16_truncated).
__global__ void gen_fp32_16_truncated_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_truncated(scale * pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16_truncated(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_truncated_kernel<<<blocks, threads>>>(n, wh, wr, scale);
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

// Exact, lossless widening (every float value is exactly representable
// in double) -- the FP64 reference's own A/B, not a new data source.
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
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||C_ref - C_gpu||_F / ||C_ref||_F -- ref is the genuine FP64 result
// (never downcast), gpu is each method's own float output, widened to
// double only inside this comparison.
double frobenius_rel_err(const std::vector<double>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = ref[i] - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += ref[i] * ref[i];
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). normal_scale widens/narrows A's
// own "normal"-range data by this multiplier.
//
// normalized: when true, B is instead generated at scale 1/sqrt(k) --
// see mm_fp32_recon_l2_error_vs_cublas_ref.cu's own run_shape comment
// for the full derivation of what this does and why. C is genuine float
// here at every scale this file's own sweeps use, so there is no
// output-storage-ceiling bug for this flag to fix in this file
// specifically -- kept for API consistency across every error benchmark
// in this codebase.
void run_shape(size_t m, size_t k, size_t n, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);
    std::printf("  %s\n", shape_desc);

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;

    float b_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float *A_plain = nullptr, *B = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&A_plain), m * k * sizeof(float), "A_plain", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&B), k * n * sizeof(float), "B", shape_desc)) {
        cudaFree(A_plain);
        cudaFree(B);
        return;
    }
    gen_float(m * k, A_plain, normal_scale);
    gen_float(k * n, B, b_scale);

    // ---- TRUE FP64 reference: widen A, B to double, run cublasDgemm,
    // keep only the host-side double result. See this file's own header
    // for the memory cost at the largest shapes. ----
    std::vector<double> C_ref_h(m * n);
    bool have_ref = false;
    {
        double *A_d = nullptr, *B_d = nullptr, *C_ref_d = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_d), m * k * sizeof(double), "A_double", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&B_d), k * n * sizeof(double), "B_double", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&C_ref_d), m * n * sizeof(double), "C_ref_double", shape_desc)) {
            widen_to_double(m * k, A_plain, A_d);
            widen_to_double(k * n, B, B_d);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), &alpha, A_d, lda, B_d, ldb, &beta, C_ref_d, ldc);
            cudaMemcpy(C_ref_h.data(), C_ref_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            have_ref = true;
        }
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_ref_d);
    }
    if (!have_ref) {
        std::printf("    cublas_fp32               SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32                SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_fp16           SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_fp16_truncated SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_fp16_workspace SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_fp16_fast SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_fp16_workspace_fast SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_bf16_truncated SKIPPED (no FP64 reference at %s)\n", shape_desc);
        std::printf("    tomos_fp32_bf16_workspace SKIPPED (no FP64 reference at %s)\n", shape_desc);
        cudaFree(A_plain);
        cudaFree(B);
        return;
    }

    dim<2> size_A{m, k};
    dim<2> size_B{k, n};

    // ---- cuBLAS FP32: cublasSgemm on the SAME A_plain/B every other
    // series here shares -- the "industry standard" FP32 GEMM baseline,
    // to see whether TOMOS FP32's own error against the true FP64
    // reference is comparable to, better, or worse than cuBLAS's own FP32
    // implementation, not just internally self-consistent. ----
    {
        float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
        float* C_gpu = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_cublas_fp32",
                          shape_desc)) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), &f_alpha, A_plain, lda, B, ldb, &f_beta, C_gpu, ldc);
            std::vector<float> C_gpu_h(m * n);
            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            std::printf("    cublas_fp32     = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
            cudaFree(C_gpu);
        }
    }

    // ---- TOMOS FP32: native fp32 kernel, consumes A_plain directly (no
    // 16-bit encoding at all). ----
    {
        float* C_gpu = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32", shape_desc)) {
            cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_plain, lda, B, ldb, beta,
                C_gpu, ldc);
            std::vector<float> C_gpu_h(m * n);
            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            std::printf("    tomos_fp32      = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
            cudaFree(C_gpu);
        }
    }

    // ---- TOMOS FP32/FP16: A packed via fp32_16_split (plain), decoded +
    // accumulated in float. ----
    {
        uint16_t *wh = nullptr, *wr = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t), "wh_fp16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr), m * k * sizeof(uint16_t), "wr_fp16", shape_desc)) {
            gen_fp32_16(m * k, wh, wr, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_fp16",
                              shape_desc)) {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, size_A, wh, wr, static_cast<size_t>(lda), size_B, B,
                                                       static_cast<size_t>(ldb), beta, C_gpu,
                                                       static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16 = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh);
            cudaFree(wr);
        }
    }

    // ---- TOMOS FP32/FP16 (fast): UNCONDITIONAL fast_normal decode
    // sibling of TOMOS FP32/FP16 above -- same plain wh/wr pair (gen_
    // fp32_16, NOT a separate generator), same dualbuf tile/pipeline, but
    // A's decode skips the exponent-mode (normal/extended/degraded)
    // dispatch entirely via fp32_16_combine_fast_normal, with NO per-
    // element validity check. UNSAFE ON ARBITRARY DATA: gen_fp32_16 draws
    // pseudo_rand-derived values uniformly over [-normal_scale,
    // normal_scale], so a small but real fraction of elements land below
    // fp32_16_split's own ~6.1e-5*normal_scale control=1 boundary at
    // every shape/scale this file's own sweeps use -- this series is
    // EXPECTED to show inflated/wrong-magnitude error from exactly those
    // elements, not just the plain encoding's own rounding error. That is
    // the point of plotting it: quantifying the real-data cost of the
    // "take it for granted" assumption, same contract already accepted
    // for this codebase's GEVM "TOMOS FP32/FP16 (fast)" series (mv_
    // fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal). ----
    {
        uint16_t *wh_f = nullptr, *wr_f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_f), m * k * sizeof(uint16_t), "wh_fp16_fast", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_f), m * k * sizeof(uint16_t), "wr_fp16_fast", shape_desc)) {
            gen_fp32_16(m * k, wh_f, wr_f, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_fp16_fast",
                              shape_desc)) {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh_f, wr_f,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_fast = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_f);
            cudaFree(wr_f);
        }
    }

    // ---- TOMOS FP32/FP16 (truncated): own truncated encoding --
    // fp32_16_split_truncated, NOT interchangeable with the plain wh/wr
    // pair above (see gen_fp32_16_truncated's own comment). ----
    {
        uint16_t *wh_t = nullptr, *wr_t = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_t), m * k * sizeof(uint16_t), "wh_fp16_trunc", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_t), m * k * sizeof(uint16_t), "wr_fp16_trunc", shape_desc)) {
            gen_fp32_16_truncated(m * k, wh_t, wr_t, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_fp16_trunc",
                              shape_desc)) {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf_truncated(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh_t, wr_t,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_truncated = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_t);
            cudaFree(wr_t);
        }
    }

    // ---- TOMOS FP32/FP16 (workspace): decodes via fp32_16_combine, same
    // plain wh/wr pair as TOMOS FP32/FP16 above -- only the materialize
    // strategy differs, not the encoding. ----
    {
        uint16_t *wh_w = nullptr, *wr_w = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_w), m * k * sizeof(uint16_t), "wh_fp16_ws", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_w), m * k * sizeof(uint16_t), "wr_fp16_ws", shape_desc)) {
            gen_fp32_16(m * k, wh_w, wr_w, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_fp16_ws",
                              shape_desc)) {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh_w, wr_w,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_workspace = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_w);
            cudaFree(wr_w);
        }
    }

    // ---- TOMOS FP32/FP16 (workspace, fast): UNCONDITIONAL fast_normal
    // decode sibling of TOMOS FP32/FP16 (workspace) above -- same plain
    // wh/wr pair, same materialize-then-native-GEMM structure, but the
    // chunk-decode pass skips the exponent-mode dispatch entirely via
    // fp32_16_combine_fast_normal, with NO per-element validity check.
    // Same UNSAFE-ON-ARBITRARY-DATA expectation as TOMOS FP32/FP16
    // (fast) above -- see that series' own comment for the full
    // derivation (identical generator, identical risk). ----
    {
        uint16_t *wh_wf = nullptr, *wr_wf = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_wf), m * k * sizeof(uint16_t), "wh_fp16_ws_fast",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_wf), m * k * sizeof(uint16_t), "wr_fp16_ws_fast",
                          shape_desc)) {
            gen_fp32_16(m * k, wh_wf, wr_wf, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_fp16_ws_fast",
                              shape_desc)) {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh_wf, wr_wf,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_workspace_fast = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_wf);
            cudaFree(wr_wf);
        }
    }

    // ---- TOMOS FP32/BF16 (truncated): A packed via fp32_bf16_split
    // (plain), decoded + accumulated in float. Labeled "(truncated)" to
    // match mm_fp32_recon_l2_error_vs_cublas_ref.cu's own naming for this
    // exact kernel/encoding pair -- the label refers to that file's own
    // materialize-strategy name, not this encoding itself (see that
    // file's own header note). ----
    {
        uint16_t *wh_b = nullptr, *wr_b = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_b), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_b), m * k * sizeof(uint16_t), "wr_bf16", shape_desc)) {
            gen_fp32_bf16(m * k, wh_b, wr_b, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_bf16",
                              shape_desc)) {
                cuda::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh_b, wr_b,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_bf16_truncated = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_b);
            cudaFree(wr_b);
        }
    }

    // ---- TOMOS FP32/BF16 (workspace): ROUNDED bf16 split --
    // fp32_bf16_split_rounded, NOT interchangeable with the plain wh/wr
    // pair above. ----
    {
        uint16_t *wh_r = nullptr, *wr_r = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_r), m * k * sizeof(uint16_t), "wh_bf16_rounded", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_r), m * k * sizeof(uint16_t), "wr_bf16_rounded",
                          shape_desc)) {
            gen_fp32_bf16_rounded(m * k, wh_r, wr_r, normal_scale);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_bf16_ws",
                              shape_desc)) {
                cuda::mm_fp32_bf16_gemm_materialize_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                            alpha, size_A, wh_r, wr_r, static_cast<size_t>(lda),
                                                            size_B, B, static_cast<size_t>(ldb), beta, C_gpu,
                                                            static_cast<size_t>(ldc));
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_bf16_workspace = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(wh_r);
            cudaFree(wr_r);
        }
    }

    cudaFree(A_plain);
    cudaFree(B);
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

    // normal_scale (first arg, default 1.0) multiplies A's own "normal"-
    // range data. All seven series here accumulate in float, so the safe
    // range is governed by fp32_16_pack.hpp's/fp32_bf16_pack.hpp's own
    // NORMAL/EXTENDED encoding boundary (magnitude < 2^16=65536), same
    // reasoning as mm_fp32_recon_l2_error_vs_cublas_ref.cu's own
    // normal_scale comment -- not by any accumulator- or output-storage-
    // overflow bound (C is float throughout).
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== GEMM: fp32-compute-tier Frobenius relative error vs TRUE FP64 (cublasDgemm) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576};

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

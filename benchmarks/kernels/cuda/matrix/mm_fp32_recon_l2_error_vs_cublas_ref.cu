// Standalone Frobenius-relative-error analysis for the "full fp32
// reconstruction" tier of GEMM -- the accuracy-vs-N companion to
// benchmarks_gemm_fp32.cu (which plots GFlop/s for the same square
// sweep, via scripts/plot_gemm_format_32_16_compute_32.m). Measures how
// far each method's output matrix drifts from a genuine FP32-precision
// reference (cublasSgemm, real fp32, no storage compression), not a
// throughput number. See scripts/plot_gemm_fp32_recon_l2_error_vs_
// cublas_ref.m for the resulting plot. Named "l2_error" for consistency
// with every other *_l2_error_vs_cublas_ref.cu file in this codebase,
// even though the actual metric computed is the FROBENIUS norm (per
// explicit user confirmation: GEMM error benchmarks use Frobenius, not
// L2, since the output is a matrix, not a vector) -- Frobenius norm on a
// flattened M*N array reduces to the exact same sqrt(sum sq)/sqrt(sum
// sq) formula L2 uses on a vector, so this file's own frobenius_rel_err
// helper below is textually identical to every sibling's own l2_rel_err.
//
// Six "TOMOS" series, verified directly against scripts/benchmarks_
// gemm_format_32_16_compute_32.sh's own current METHOD_MARKERS and
// plot_gemm_format_32_16_compute_32.m's own current (non-commented-out)
// `series` struct:
//
//   TOMOS FP32                    = gemm_fp32_tiled_syncpipe4_bk8_
//                                    warptile_nostride_snake_midpoint_exact
//   TOMOS FP32/FP16                = mm_fp32_16_tiled_cpasync_dualbuf
//   TOMOS FP32/FP16 (truncated)    = mm_fp32_16_tiled_cpasync_dualbuf_truncated
//   TOMOS FP32/FP16 (workspace)    = mm_fp32_16_gemm_materialize_midpoint_exact
//   TOMOS FP32/BF16 (truncated)    = mm_fp32_bf16_syncpipe4_bk8_warptile_
//                                    nostride_snake_midpoint_exact
//   TOMOS FP32/BF16 (workspace)    = mm_fp32_bf16_gemm_materialize_rounded
//
// All six kernels decode their packed A operand (or, for TOMOS FP32,
// consume it directly) into genuine float and accumulate their tile's
// partial sums in a float register array -- confirmed by reading mm_
// fp32_16_tiled_cpasync_dualbuf_kernel's own body (float acc[TM][TN]),
// the family every one of these six ultimately shares -- so, unlike the
// GEMV "pure_half"/"hybrid" tier, there is NO fp16/bf16-accumulator
// overflow concern here: the much wider, encoding-boundary-governed
// SCALES range applies (same reasoning as mv_fp32_recon_l2_error_vs_
// cublas_ref.cu's own header, this file's GEMV-recon-tier sibling).
//
// Unlike that GEMV sibling (which needed three separate data-generation
// blocks -- plain/hybrid-mixed/safe-normal -- because some of its
// promoted kernels require magnitude-restricted or extended-range data),
// EVERY kernel here is safe on arbitrary data: none of the six promoted
// GEMM kernels is an unconditional-decode "(fast)"-style kernel, and
// none exercises a hybrid extended-exponent path (no "hybrid" GEMM
// kernel is promoted at all -- confirmed against the METHOD_MARKERS/
// `series` cross-check above). All six therefore share ONE plain
// pseudo_rand-derived ground truth (A_plain, B) and ONE cublasSgemm
// reference (C_ref) per shape -- gen_float, gen_fp32_16, gen_fp32_16_
// truncated, gen_fp32_bf16, and gen_fp32_bf16_rounded are all pure,
// deterministic functions of (linear index, scale) alone, so calling
// each with the SAME m*k and scale is guaranteed to correspond to the
// SAME underlying "true" A values, without ever decoding wh/wr back to
// float on the host.
//
// B (float, K x N) is never compressed by any of these methods, so a
// SINGLE shared B is generated once per shape and reused across all six
// series -- left permanently UNSCALED by normal_scale (only A's own
// magnitude/exponent range is the thing actually being 16-bit-
// compressed, same reasoning as every sibling's own x-left-unscaled
// convention).
//
// Frobenius relative error: ||C_ref - C_method||_F / ||C_ref||_F,
// computed on the HOST but only over the OUTPUT matrix C (size M*N) --
// never re-touching A or B after generation.
//
// Usage: mm_fp32_recon_l2_error_vs_cublas_ref [normal_scale [N]]
//   No args: runs the same SQUARE_NS sweep as benchmarks_gemm_format_
//            32_16_compute_32.sh's own square sweep (1024 through
//            49152, M=K=N), at normal_scale=1.0.
//   normal_scale: runs the full SQUARE_NS sweep with A's own "normal"-
//            range data widened to [-scale,scale] instead. See scripts/
//            benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh for
//            the actual scale tiers swept.
//   normal_scale N: runs just that one square shape (M=K=N=N) at that
//            scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemm_fp32_recon_l2_
// error_vs_cublas_ref.sh.

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

// CORRECTNESS CONTRACT: mm_fp32_16_tiled_cpasync_dualbuf_truncated
// requires wh/wr from fp32_16_split_truncated specifically, never
// fp32_16_split -- see fp32_16_combine_truncated's own comment in
// fp32_16_pack.hpp (copied verbatim from benchmarks_gemm_fp32.cu's
// own gen_fp32_16_truncated).
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
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||C_ref - C_gpu||_F / ||C_ref||_F, computed on the host over the
// OUTPUT matrix only (size M*N) -- see this file's own header for why
// this is textually identical to every sibling's own l2_rel_err (a
// flattened Frobenius norm IS an L2 norm).
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
// column-major, so lda=m, ldb=k, ldc=m) -- same convention fp32_recon_
// vs_cutlass_ref.cu's own run_shape uses. normal_scale widens/narrows
// A's own "normal"-range data by this multiplier -- B is deliberately
// left UNSCALED by normal_scale specifically (see this file's own header
// for why).
//
// normalized is a separate, orthogonal knob: when true, B is instead
// generated at scale 1/sqrt(k), so C[i,j]'s own row-sum (over K terms)
// keeps magnitude ~normal_scale regardless of K, instead of growing like
// normal_scale*sqrt(K) -- see mm_fp16_bf16_pure_l2_error_vs_cublas_
// ref.cu's own run_shape comment (this file's GEMM-tensor-core-tier
// sibling, where this actually matters -- its own C is __half-typed for
// three of five series) for the full derivation. Every series here
// writes to genuine float C (no format ceiling at any scale used in this
// file's own default sweep), so --normalized has no bug to fix in THIS
// file specifically -- added for API consistency across every error
// benchmark in this codebase, and to let the same normalized-vs-raw
// comparison be made uniformly.
void run_shape(size_t m, size_t k, size_t n, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);
    std::printf("  %s\n", shape_desc);

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;

    float b_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float *A_plain = nullptr, *B = nullptr, *C_ref = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&A_plain), m * k * sizeof(float), "A_plain", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&B), k * n * sizeof(float), "B", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&C_ref), m * n * sizeof(float), "C_ref", shape_desc)) {
        cudaFree(A_plain);
        cudaFree(B);
        cudaFree(C_ref);
        return;
    }
    gen_float(m * k, A_plain, normal_scale);
    gen_float(k * n, B, b_scale);

    float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
               &f_alpha, A_plain, lda, B, ldb, &f_beta, C_ref, ldc);
    std::vector<float> C_ref_h(m * n);
    cudaMemcpy(C_ref_h.data(), C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    dim<2> size_A{m, k};
    dim<2> size_B{k, n};

    // ---- TOMOS FP32: native fp32 kernel, consumes A_plain directly (no
    // wh/wr decode) -- own error is a sanity check on this file's own
    // methodology (should stay near machine epsilon regardless of
    // normal_scale). ----
    {
        float* C_gpu = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_native", shape_desc)) {
            cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_plain, lda, B, ldb, beta,
                C_gpu, ldc);
            std::vector<float> C_gpu_h(m * n);
            cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            std::printf("    tomos_fp32               = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
            cudaFree(C_gpu);
        }
    }

    // ---- TOMOS FP32/FP16 and TOMOS FP32/FP16 (workspace): both decode
    // via fp32_16_combine (plain, non-truncated), so both share the SAME
    // plain wh/wr encoding -- fp32_16_split. ----
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
                std::printf("    tomos_fp32_fp16          = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }

            float* C_gpu_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu_ws), m * n * sizeof(float), "C_gpu_fp32_fp16_workspace",
                              shape_desc)) {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, size_A, wh, wr,
                    static_cast<size_t>(lda), size_B, B, static_cast<size_t>(ldb), beta, C_gpu_ws,
                    static_cast<size_t>(ldc));
                std::vector<float> C_gpu_ws_h(m * n);
                cudaMemcpy(C_gpu_ws_h.data(), C_gpu_ws, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp32_fp16_workspace = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_ws_h));
                cudaFree(C_gpu_ws);
            }
            cudaFree(wh);
            cudaFree(wr);
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

    // ---- TOMOS FP32/BF16 (truncated): plain (non-rounded) bf16 split --
    // fp32_bf16_split. ----
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
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_fp32_bf16_workspace",
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
    cudaFree(C_ref);
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
    // range data. All six series here accumulate in float (see this
    // file's own header), so there is no fp16/bf16-accumulator overflow
    // concern -- the safe range is instead governed by fp32_16_pack.hpp's
    // own NORMAL/EXTENDED encoding boundary (magnitude < 2^16=65536 for
    // the packed formats' own "always control=0"/plain-normal-band
    // contract), same reasoning as mv_fp32_recon_l2_error_vs_cublas_
    // ref.cu's own normal_scale comment.
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== GEMM: full-fp32-reconstruction-tier Frobenius relative error vs FP32 (cublasSgemm) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144,
                                              8192, 12288, 16384, 24576, 32768, 49152};

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

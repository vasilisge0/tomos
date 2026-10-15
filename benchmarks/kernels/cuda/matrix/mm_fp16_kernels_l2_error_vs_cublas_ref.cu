// Standalone Frobenius-relative-error analysis comparing THREE different
// FP16 GEMM IMPLEMENTATIONS (not different A/B encodings -- see this
// file's own sibling, mm_fp32_16_variable_rounding_l2_error_vs_cublas_
// ref.cu, for that separate axis) on the EXACT SAME native-rounded A/B
// data, all measured against a genuine FP32 (cublasSgemm) reference:
//
//   TOMOS FP16   = gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
//                   acongruous_fastk_precomp2_serp -- tomos's own
//                   tensor-core kernel.
//   cuBLAS FP16  = cublasHgemm -- cuBLAS's own simple FP16 GEMM API. Its
//                   own internal accumulation precision is NOT
//                   guaranteed to match TOMOS FP16's own (TOMOS
//                   accumulates in float; cublasHgemm's simple API may
//                   accumulate in half -- see this file's own sibling
//                   plot's header for the empirical ~30x error gap this
//                   produced when first tried there), so this series
//                   isolates "cuBLAS's own default FP16 GEMM" as a
//                   whole, not accumulation precision alone.
//   CUTLASS FP16 = cutlass::gemm::device::Gemm, the SAME "textbook"
//                   Ampere tensorop config (128x256x32 threadblock tile,
//                   64x64x32 warp tile, 16x8x16 instruction shape,
//                   3-stage pipeline, FP32 accumulator) this codebase's
//                   own cutlass_fp16_gemm_ref.cu already uses as its own
//                   external throughput comparison point -- reused here
//                   verbatim, now for accuracy instead of speed.
//
// All three run on the IDENTICAL A_half/B_half buffers (generated once
// per shape, shared across all three series -- unlike the variable-
// rounding sibling, there is no per-series encoding variation here at
// all), so the only thing that ever differs between the three series is
// which GEMM implementation computed the result, never the input data.
//
// cutlass::half_t and __half are bit-layout-compatible (same convention
// cutlass_fp16_gemm_ref.cu's own header already establishes) -- the
// CUTLASS series reinterpret_casts the SAME __half* A_half/B_half
// buffers instead of regenerating a separate cutlass::half_t copy,
// keeping the comparison exact.
//
// Frobenius relative error: ||C_ref - C_method||_F / ||C_ref||_F,
// computed on the HOST in double precision, over the OUTPUT matrix only
// (size M*N) -- same helper, same convention, as every other
// *_l2_error_vs_cublas_ref.cu file in this codebase.
//
// Uses the SAME square-sweep sizes and the SAME normal_scale/--normalized
// SCALES tiers as mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu
// (see scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh and
// its two scale-sweep siblings) -- deliberately identical, so the two
// plot families are directly comparable at the same shapes/scales: this
// file isolates "which FP16 GEMM implementation", the variable-rounding
// sibling isolates "which A encoding/rounding method", against the SAME
// underlying pseudo_rand-derived ground truth and the SAME kind of FP32
// reference.
//
// Usage: mm_fp16_kernels_l2_error_vs_cublas_ref [normal_scale [N]] [--normalized]
//   No args: runs the default square sweep at normal_scale=1.0.
//   normal_scale: runs the full sweep with A's own "normal"-range data
//            widened to [-scale,scale] instead.
//   normal_scale N: runs just that one square shape (M=K=N=N) at that
//            scale.
//   --normalized: scales B by 1/sqrt(K) instead of leaving it at plain
//            [-1,1] -- see run_shape's own comment.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so AND against a standalone CUTLASS checkout (header-only,
// no separate library build -- see scripts/cutlass_fp16_gemm_ref.sh's own
// header for the CUTLASS_DIR convention this file's own .sh driver
// reuses), same convention as every other *_vs_cublas_ref.cu file in this
// codebase.

#include <cublas_v2.h>
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

#include "dimensions.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __half* A, size_t ld_a,
    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

// Same "textbook" Ampere tensorop config as cutlass_fp16_gemm_ref.cu's
// own Gemm alias -- copied verbatim, see that file's own header for the
// rationale (a standard CUTLASS example config, not a tuned search).
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

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

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
// column-major, so lda=m, ldb=k, ldc=m) -- same convention every sibling
// *_l2_error_vs_cublas_ref.cu file in this codebase uses. normal_scale
// widens/narrows A's own "normal"-range data by this multiplier.
//
// normalized: when true, B is instead generated at scale 1/sqrt(k) --
// see mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu's own
// run_shape comment for the full derivation. This file's own C is
// __half-typed for every series (all three are genuine FP16 GEMMs), so
// the same K-dependent output-storage-overflow concern applies here --
// see this file's own .sh drivers for the SCALES ranges chosen to avoid
// it (identical to the variable-rounding sibling's own).
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
    cudaFree(A_true);
    cudaFree(B_true);
    cudaFree(C_ref);

    // Shared native-rounded A/B -- the SAME buffers every one of the
    // three series below reads, so only the GEMM implementation varies.
    __half *A_half = nullptr, *B_half = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&A_half), m * k * sizeof(__half), "A_half", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&B_half), k * n * sizeof(__half), "B_half", shape_desc)) {
        cudaFree(A_half);
        cudaFree(B_half);
        return;
    }
    gen_half(m * k, A_half, normal_scale);
    gen_half(k * n, B_half, b_scale);

    // ---- TOMOS FP16 ----
    {
        __half* C_half = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_tomos", shape_desc)) {
            cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half,
                static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half, static_cast<size_t>(ldc));
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_tomos", shape_desc)) {
                half_to_float(m * n, C_half, C_gpu);
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp16   = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(C_half);
        }
    }

    // ---- cuBLAS FP16 ----
    {
        __half* C_half = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_cublas", shape_desc)) {
            __half h_alpha = __float2half(static_cast<float>(alpha));
            __half h_beta = __float2half(static_cast<float>(beta));
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), &h_alpha, A_half, lda, B_half, ldb, &h_beta, C_half, ldc);
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_cublas", shape_desc)) {
                half_to_float(m * n, C_half, C_gpu);
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    cublas_fp16  = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
            cudaFree(C_half);
        }
    }

    // ---- CUTLASS FP16 ---- (reinterpret_cast onto the SAME A_half/
    // B_half bytes -- cutlass::half_t and __half share the same 16-bit
    // IEEE layout).
    {
        __half* C_half = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_cutlass",
                          shape_desc)) {
            cutlass::half_t* A_ct = reinterpret_cast<cutlass::half_t*>(A_half);
            cutlass::half_t* B_ct = reinterpret_cast<cutlass::half_t*>(B_half);
            cutlass::half_t* C_ct = reinterpret_cast<cutlass::half_t*>(C_half);

            cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            CutlassElementAccumulator ct_alpha(static_cast<float>(alpha)), ct_beta(static_cast<float>(beta));
            typename CutlassGemm::Arguments args(problem_size, {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc}, {C_ct, ldc},
                                                 {ct_alpha, ct_beta});

            size_t workspace_size = CutlassGemm::get_workspace_size(args);
            void* workspace = nullptr;
            if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

            CutlassGemm gemm_op;
            cutlass::Status status = gemm_op.can_implement(args);
            if (status == cutlass::Status::kSuccess) status = gemm_op.initialize(args, workspace);
            if (status == cutlass::Status::kSuccess) status = gemm_op();

            if (status == cutlass::Status::kSuccess) {
                float* C_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_cutlass",
                                  shape_desc)) {
                    half_to_float(m * n, C_half, C_gpu);
                    std::vector<float> C_gpu_h(m * n);
                    cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cutlass_fp16 = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                    cudaFree(C_gpu);
                }
            } else {
                std::printf("    cutlass_fp16 SKIPPED (cutlass status: %s at %s)\n",
                           cutlass::cutlassGetStatusString(status), shape_desc);
            }
            if (workspace) cudaFree(workspace);
            cudaFree(C_half);
        }
    }

    cudaFree(A_half);
    cudaFree(B_half);
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
        "=== GEMM: TOMOS/cuBLAS/CUTLASS FP16 kernel comparison, Frobenius relative error vs FP32 (cublasSgemm) "
        "reference (normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    // Coarse log-spaced points up to 49152, then LINEAR steps of 1000 up
    // to 57152, PLUS a further zoom of LINEAR steps of 100 from 58152 to
    // 61152 (30 steps), then back to the coarse 1000-step tail up to
    // 65536 -- the error stayed flat through 49152 at coarse resolution,
    // so this densifies the upper range specifically to look for where
    // (if anywhere) it starts to increase, without paying for a full
    // linear sweep of the entire range (see this file's own header for
    // why float-accumulating kernels shouldn't show K-dependent growth
    // here at all, unlike a possibly-FP16-accumulating cuBLAS Hgemm).
    const std::vector<size_t> square_sizes = {
        1024,  1536,  2048,  3072,  4096,  6144,  8192,  12288, 16384, 24576, 32768, 49152, 50152, 51152,
        52152, 53152, 54152, 55152, 56152, 57152, 58152, 58252, 58352, 58452, 58552, 58652, 58752, 58852,
        58952, 59052, 59152, 59252, 59352, 59452, 59552, 59652, 59752, 59852, 59952, 60052, 60152, 60252,
        60352, 60452, 60552, 60652, 60752, 60852, 60952, 61052, 61152, 62152, 63152, 64152, 65152, 65536};

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

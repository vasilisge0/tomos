// Standalone correctness verification for ONLY the five TOMOS kernels
// scripts/plot_fp16_bf16_pure_vs_cutlass_ref.m currently plots (non-
// commented-out `series` entries; the benchmark those GFlop/s numbers
// come from is benchmarks_gemm_fp16.cu) -- a focused, fast
// check to run after any change to one of these five specific
// dispatchers, without paying for the much larger exploratory suites
// (verify_mm_fp32_16_pure_half_gemm.cu / verify_mm_fp16_gemm.cu / etc.).
// Pass/fail decided on FROBENIUS-relative error (see mm_fp32_recon_l2_
// error_vs_cublas_ref.cu's own header for why that's textually identical
// to every GEVM verify file's own L2-relative error).
//
// PROMOTED_KERNEL (mirrors plot_fp16_bf16_pure_vs_cutlass_ref.m's own
// current, non-commented-out `series` struct, verified this session
// against both that file and scripts/fp16_bf16_pure_vs_cutlass_ref_
// matched.sh's own METHOD_MARKERS):
//   TOMOS FP16              -> gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp
//   TOMOS BF16               -> gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp
//   TOMOS FP32/FP16          -> mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp
//   TOMOS FP32/FP16 (workspace) -> mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp
//   TOMOS FP32/BF16          -> mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp
//
// Unlike every other GEMM verify file in this codebase, this tier's own
// kernels are genuinely __half/__nv_bfloat16-typed throughout (A, B, AND
// C -- no float* B/C overload exists here, same design constraint
// documented in mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu's own
// header), and their compute happens on real tensor cores (sm_80
// mma.sync, hardware-mandated float32 accumulator regardless of the
// fp16/fp16 or bf16/bf16 input types -- confirmed by reading gemm_
// fp16_mma_ptx_..._ildsm_kernel's own body, same finding as that error
// benchmark's own header). So the CPU reference for every check here is
// built as: round A/B to their OWN target storage precision on the HOST
// (exactly what the GPU kernel's own inputs are), decode back to float,
// then run cpu::mm_strided<float,float,float> (float-accumulating) on
// those rounded-then-decoded values -- this is what a genuine float32
// accumulator over the SAME rounded inputs should produce, which is
// what these tensor-core kernels are actually expected to match (NOT a
// comparison against never-rounded, full-fp32-precision A/B, which
// would just be re-measuring the tier's OWN inherent storage-rounding
// error -- that measurement already exists, in mm_fp16_bf16_pure_l2_
// error_vs_cublas_ref.cu).
//
// TOMOS FP16 (native) and TOMOS FP32/FP16 (both plain-split, and its
// own "(workspace)" sibling) share the SAME B_half (their own B operand
// is always plain-rounded, no packed format ever applies to B here);
// TOMOS FP32/FP16's/​(workspace)'s own A comes from fp32_16_half_only(wh)
// (packed, truncating decode -- fp32_16_split's own encoding), NOT a
// plain __float2half round, though for in-range NORMAL-band magnitudes
// (this file's own [-1,1] data always is) the two decodes coincide to
// within the same rounding-mode margin; decoded explicitly via fp32_16_
// half_only here rather than assumed, for an exact match to what each
// kernel's own decode step actually does. Same structure for the bf16
// group (TOMOS BF16 native, TOMOS FP32/BF16 via fp32_bf16_bf16_only).
//
// Includes one non-exact-tile-multiple shape (1000x1000x1000) alongside
// exact-multiple shapes -- both native kernels fall back to a bounds-
// checked internal path when the problem size isn't an exact multiple of
// their own tile constants (see each kernel's own comment in mm_cuda_
// kernels.cu), a real regression check on that fallback path.
//
// Usage: verify_mm_fp16_bf16_pure_plot_promoted
// Exit code is non-zero if any check fails.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "mm_cpu_kernels.hpp"

using namespace tomos;

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

namespace {

// Frobenius-relative-error pass/fail, same convention as verify_mm_
// fp32_recon_plot_promoted.cu's own check().
bool check(const std::string& label, const std::vector<float>& cpu, const std::vector<float>& gpu, double tol) {
    cudaError_t pending_err = cudaGetLastError();
    if (pending_err != cudaSuccess) {
        std::printf("  [diag] CUDA ERROR pending before checking %-40s : %s\n", label.c_str(),
                   cudaGetErrorString(pending_err));
    }
    double max_abs_err = 0.0, max_rel_err = 0.0;
    double sum_sq_err = 0.0, sum_sq_cpu = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double cpu_i = static_cast<double>(cpu[i]);
        double err = std::fabs(cpu_i - static_cast<double>(gpu[i]));
        double rel = err / std::max(1.0, std::fabs(cpu_i));
        if (err > max_abs_err) worst_i = i;
        max_abs_err = std::max(max_abs_err, err);
        max_rel_err = std::max(max_rel_err, rel);
        sum_sq_err += err * err;
        sum_sq_cpu += cpu_i * cpu_i;
    }
    double frobenius_rel_err = (sum_sq_cpu > 0.0) ? std::sqrt(sum_sq_err / sum_sq_cpu) : std::sqrt(sum_sq_err);
    bool ok = frobenius_rel_err < tol;
    std::printf("  %-58s frobenius_rel_err=%.6g max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(),
               frobenius_rel_err, max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("    [diag] worst_i=%zu cpu=%.6f gpu=%.6f (of %zu elements)\n", worst_i,
                   static_cast<double>(cpu[worst_i]), static_cast<double>(gpu[worst_i]), cpu.size());
    }
    return ok;
}

// NO_TRANSPOSE only (this tier has no TRANSPOSE kernels anywhere in this
// codebase). tol=5e-2 (Frobenius-relative-error < 5%), same bound the
// GEVM "pure_half"/fp16-compute tier's own checks use -- this tier's
// own compute is genuinely fp16/bf16-precision throughout (only the
// accumulator is float), so a fp16/bf16-storage-level tolerance is
// appropriate, unlike the "recon" tier's own near-fp32 1e-3 bound.
bool run_case(size_t m, size_t k, size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    char label[160];
    double alpha = 1.7, beta = 0.3;

    std::vector<float> A_true(m * k), B_true(k * n), C_init(m * n);
    for (auto& v : A_true) v = dist(rng);
    for (auto& v : B_true) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    // ---- fp16 group: TOMOS FP16 (native), TOMOS FP32/FP16, TOMOS
    // FP32/FP16 (workspace) -- own __half B, shared across all three. ----
    {
        std::vector<__half> B_half(k * n);
        std::vector<float> B_half_f(k * n);
        for (size_t i = 0; i < k * n; ++i) {
            B_half[i] = __float2half(B_true[i]);
            B_half_f[i] = __half2float(B_half[i]);
        }
        __half* d_B_half = nullptr;
        cudaMalloc(&d_B_half, B_half.size() * sizeof(__half));
        cudaMemcpy(d_B_half, B_half.data(), B_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

        // TOMOS FP16 (native): genuinely __half-typed A, no packed
        // format involved.
        {
            std::vector<__half> A_half(m * k);
            std::vector<float> A_half_f(m * k);
            for (size_t i = 0; i < m * k; ++i) {
                A_half[i] = __float2half(A_true[i]);
                A_half_f[i] = __half2float(A_half[i]);
            }
            __half *d_A_half = nullptr, *d_C_half = nullptr;
            cudaMalloc(&d_A_half, A_half.size() * sizeof(__half));
            cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
            cudaMemcpy(d_A_half, A_half.data(), A_half.size() * sizeof(__half), cudaMemcpyHostToDevice);
            std::vector<__half> C_init_half(m * n);
            for (size_t i = 0; i < m * n; ++i) C_init_half[i] = __float2half(C_init[i]);
            cudaMemcpy(d_C_half, C_init_half.data(), C_init_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

            std::vector<float> C_cpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_cpu[i] = __half2float(C_init_half[i]);
            cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                 dim<2>{m, k}, A_half_f.data(), m, dim<2>{k, n}, B_half_f.data(), k,
                                                 beta, C_cpu.data(), m);
            cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_A_half,
                static_cast<size_t>(lda), d_B_half, static_cast<size_t>(ldb), beta, d_C_half,
                static_cast<size_t>(ldc));
            std::vector<__half> C_gpu_half(m * n);
            cudaMemcpy(C_gpu_half.data(), d_C_half, C_gpu_half.size() * sizeof(__half), cudaMemcpyDeviceToHost);
            std::vector<float> C_gpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_gpu[i] = __half2float(C_gpu_half[i]);
            std::snprintf(label, sizeof(label), "TOMOS FP16 (gemm_fp16_mma_ptx_..._serp) %s", tag);
            all_ok &= check(label, C_cpu, C_gpu, 5e-2);
            cudaFree(d_A_half);
            cudaFree(d_C_half);
        }

        // TOMOS FP32/FP16 and TOMOS FP32/FP16 (workspace): share the
        // SAME wh (both decode via fp32_16_half_only).
        {
            std::vector<uint16_t> wh(m * k), wr(m * k);
            for (size_t i = 0; i < m * k; ++i) fp32_16_split(A_true[i], wh[i], wr[i]);
            std::vector<float> A_decoded(m * k);
            for (size_t i = 0; i < m * k; ++i) A_decoded[i] = __half2float(fp32_16_half_only(wh[i]));
            uint16_t* d_wh = nullptr;
            cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            std::vector<__half> C_init_half(m * n);
            for (size_t i = 0; i < m * n; ++i) C_init_half[i] = __float2half(C_init[i]);

            {
                __half* d_C_half = nullptr;
                cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
                cudaMemcpy(d_C_half, C_init_half.data(), C_init_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

                std::vector<float> C_cpu(m * n);
                for (size_t i = 0; i < m * n; ++i) C_cpu[i] = __half2float(C_init_half[i]);
                cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                     dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n},
                                                     B_half_f.data(), k, beta, C_cpu.data(), m);
                cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh,
                    static_cast<size_t>(lda), d_B_half, static_cast<size_t>(ldb), beta, d_C_half,
                    static_cast<size_t>(ldc));
                std::vector<__half> C_gpu_half(m * n);
                cudaMemcpy(C_gpu_half.data(), d_C_half, C_gpu_half.size() * sizeof(__half), cudaMemcpyDeviceToHost);
                std::vector<float> C_gpu(m * n);
                for (size_t i = 0; i < m * n; ++i) C_gpu[i] = __half2float(C_gpu_half[i]);
                std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (wide_bigreg_sq_inline_dirty_..._serp) %s", tag);
                all_ok &= check(label, C_cpu, C_gpu, 5e-2);
                cudaFree(d_C_half);
            }
            {
                __half* d_C_half = nullptr;
                cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
                cudaMemcpy(d_C_half, C_init_half.data(), C_init_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

                std::vector<float> C_cpu(m * n);
                for (size_t i = 0; i < m * n; ++i) C_cpu[i] = __half2float(C_init_half[i]);
                cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                     dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n},
                                                     B_half_f.data(), k, beta, C_cpu.data(), m);
                cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
                    TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh,
                    static_cast<size_t>(lda), d_B_half, static_cast<size_t>(ldb), beta, d_C_half,
                    static_cast<size_t>(ldc));
                std::vector<__half> C_gpu_half(m * n);
                cudaMemcpy(C_gpu_half.data(), d_C_half, C_gpu_half.size() * sizeof(__half), cudaMemcpyDeviceToHost);
                std::vector<float> C_gpu(m * n);
                for (size_t i = 0; i < m * n; ++i) C_gpu[i] = __half2float(C_gpu_half[i]);
                std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (workspace) (gemm_materialize_cached_vec_..._serp) %s",
                             tag);
                all_ok &= check(label, C_cpu, C_gpu, 5e-2);
                cudaFree(d_C_half);
            }
            cudaFree(d_wh);
        }
        cudaFree(d_B_half);
    }

    // ---- bf16 group: TOMOS BF16 (native), TOMOS FP32/BF16 -- own
    // __nv_bfloat16 B, shared across both. ----
    {
        std::vector<__nv_bfloat16> B_bf16(k * n);
        std::vector<float> B_bf16_f(k * n);
        for (size_t i = 0; i < k * n; ++i) {
            B_bf16[i] = __float2bfloat16(B_true[i]);
            B_bf16_f[i] = __bfloat162float(B_bf16[i]);
        }
        __nv_bfloat16* d_B_bf16 = nullptr;
        cudaMalloc(&d_B_bf16, B_bf16.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(d_B_bf16, B_bf16.data(), B_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        // TOMOS BF16 (native): genuinely __nv_bfloat16-typed A.
        {
            std::vector<__nv_bfloat16> A_bf16(m * k);
            std::vector<float> A_bf16_f(m * k);
            for (size_t i = 0; i < m * k; ++i) {
                A_bf16[i] = __float2bfloat16(A_true[i]);
                A_bf16_f[i] = __bfloat162float(A_bf16[i]);
            }
            __nv_bfloat16 *d_A_bf16 = nullptr, *d_C_bf16 = nullptr;
            cudaMalloc(&d_A_bf16, A_bf16.size() * sizeof(__nv_bfloat16));
            cudaMalloc(&d_C_bf16, C_init.size() * sizeof(__nv_bfloat16));
            cudaMemcpy(d_A_bf16, A_bf16.data(), A_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            std::vector<__nv_bfloat16> C_init_bf16(m * n);
            for (size_t i = 0; i < m * n; ++i) C_init_bf16[i] = __float2bfloat16(C_init[i]);
            cudaMemcpy(d_C_bf16, C_init_bf16.data(), C_init_bf16.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyHostToDevice);

            std::vector<float> C_cpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_cpu[i] = __bfloat162float(C_init_bf16[i]);
            cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                 dim<2>{m, k}, A_bf16_f.data(), m, dim<2>{k, n}, B_bf16_f.data(), k,
                                                 beta, C_cpu.data(), m);
            cuda::gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_A_bf16,
                static_cast<size_t>(lda), d_B_bf16, static_cast<size_t>(ldb), beta, d_C_bf16,
                static_cast<size_t>(ldc));
            std::vector<__nv_bfloat16> C_gpu_bf16(m * n);
            cudaMemcpy(C_gpu_bf16.data(), d_C_bf16, C_gpu_bf16.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost);
            std::vector<float> C_gpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_gpu[i] = __bfloat162float(C_gpu_bf16[i]);
            std::snprintf(label, sizeof(label), "TOMOS BF16 (gemm_bf16_mma_ptx_..._serp) %s", tag);
            all_ok &= check(label, C_cpu, C_gpu, 5e-2);
            cudaFree(d_A_bf16);
            cudaFree(d_C_bf16);
        }

        // TOMOS FP32/BF16: decode via fp32_bf16_bf16_only.
        {
            std::vector<uint16_t> wh(m * k), wr(m * k);
            for (size_t i = 0; i < m * k; ++i) fp32_bf16_split(A_true[i], wh[i], wr[i]);
            std::vector<float> A_decoded(m * k);
            for (size_t i = 0; i < m * k; ++i) A_decoded[i] = __bfloat162float(fp32_bf16_bf16_only(wh[i]));
            uint16_t* d_wh = nullptr;
            cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            std::vector<__nv_bfloat16> C_init_bf16(m * n);
            for (size_t i = 0; i < m * n; ++i) C_init_bf16[i] = __float2bfloat16(C_init[i]);
            __nv_bfloat16* d_C_bf16 = nullptr;
            cudaMalloc(&d_C_bf16, C_init.size() * sizeof(__nv_bfloat16));
            cudaMemcpy(d_C_bf16, C_init_bf16.data(), C_init_bf16.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyHostToDevice);

            std::vector<float> C_cpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_cpu[i] = __bfloat162float(C_init_bf16[i]);
            cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                 dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n}, B_bf16_f.data(), k,
                                                 beta, C_cpu.data(), m);
            cuda::mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh,
                static_cast<size_t>(lda), d_B_bf16, static_cast<size_t>(ldb), beta, d_C_bf16,
                static_cast<size_t>(ldc));
            std::vector<__nv_bfloat16> C_gpu_bf16(m * n);
            cudaMemcpy(C_gpu_bf16.data(), d_C_bf16, C_gpu_bf16.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost);
            std::vector<float> C_gpu(m * n);
            for (size_t i = 0; i < m * n; ++i) C_gpu[i] = __bfloat162float(C_gpu_bf16[i]);
            std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (pure_bf16_gemm_direct_..._serp) %s", tag);
            all_ok &= check(label, C_cpu, C_gpu, 5e-2);
            cudaFree(d_wh);
            cudaFree(d_C_bf16);
        }
        cudaFree(d_B_bf16);
    }

    return all_ok;
}

}  // namespace

int main() {
    std::printf(
        "=== Correctness: TOMOS FP16/BF16 pure/composite GEMM plot's 5 promoted kernels (GPU vs CPU reference) ===\n");

    struct Case { size_t m, k, n; const char* tag; };
    std::vector<Case> cases = {
        {256, 256, 256, "square-small"},
        {1000, 1000, 1000, "square-non-exact-tile-multiple"},
        {1024, 1024, 1024, "square-exact-tile-multiple"},
        {2048, 3072, 1536, "rect"},
        {4096, 4096, 4096, "square-large"},
    };

    bool all_ok = true;
    for (const auto& c : cases) all_ok &= run_case(c.m, c.k, c.n, c.tag);

    std::printf("%s\n", all_ok ? "Correctness: ALL CHECKS PASSED"
                               : "Correctness: SOME CHECKS FAILED (see FAIL lines above)");
    return all_ok ? 0 : 1;
}

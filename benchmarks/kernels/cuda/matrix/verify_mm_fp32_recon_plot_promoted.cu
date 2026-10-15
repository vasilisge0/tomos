// Standalone correctness verification for ONLY the six TOMOS kernels
// scripts/plot_gemm_format_32_16_compute_32.m currently plots (non-
// commented-out `series` entries; the benchmark those GFlop/s numbers
// come from is benchmarks_gemm_fp32.cu) -- a focused, fast check to
// run after any change to one of these six specific dispatchers, without
// paying for the much larger exploratory sweeps in verify_mm_fp32_16_
// gemm.cu / verify_mm_fp32_gemm.cu. Pass/fail decided on FROBENIUS-
// relative error (same formula as every GEVM verify file's own L2-
// relative error, applied to the flattened M*N output array -- see
// mm_fp32_recon_l2_error_vs_cublas_ref.cu's own header for why that's
// textually identical), not the per-element max-relative-error metric
// verify_mm_fp32_16_gemm.cu's own giant suite still uses.
//
// PROMOTED_KERNEL (mirrors plot_gemm_format_32_16_compute_32.m's own
// current, non-commented-out `series` struct, verified this session
// against both that file and scripts/benchmarks_gemm_format_32_16_
// compute_32.sh's own METHOD_MARKERS):
//   TOMOS FP32                    -> gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact
//   TOMOS FP32/FP16                -> mm_fp32_16_tiled_cpasync_dualbuf
//   TOMOS FP32/FP16 (truncated)    -> mm_fp32_16_tiled_cpasync_dualbuf_truncated
//   TOMOS FP32/FP16 (workspace)    -> mm_fp32_16_gemm_materialize_midpoint_exact
//   TOMOS FP32/BF16 (truncated)    -> mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact
//   TOMOS FP32/BF16 (workspace)    -> mm_fp32_bf16_gemm_materialize_rounded
//
// TOMOS FP32/FP16 and TOMOS FP32/FP16 (workspace) both decode via plain
// fp32_16_combine (fp32_16_split-produced wh/wr) -- cpu::mm_fp32_16_
// accurate is the exact CPU counterpart, reused directly for both.
//
// TOMOS FP32/FP16 (truncated) requires wh/wr from fp32_16_split_
// truncated specifically (fp32_16_combine_truncated's own correctness
// contract, see that function's comment in fp32_16_pack.hpp) -- no
// cpu::mm_fp32_16_accurate_truncated-style helper exists, so its own
// reference decodes wh/wr to a plain float array on the HOST
// (fp32_16_combine_truncated) then reuses cpu::mm_strided, same
// technique the bf16 checks below (and every TRANSPOSE GEVM verify
// check in this codebase) use for a format with no dedicated cpu::
// helper.
//
// TOMOS FP32/BF16 (truncated) and TOMOS FP32/BF16 (workspace) both need
// their own host-side decode + cpu::mm_strided reference too -- no
// cpu::mm_fp32_bf16_accurate-style helper exists in this codebase at all
// (confirmed by grep) -- (truncated) via plain fp32_bf16_combine,
// (workspace) via fp32_bf16_combine_rounded (ROUNDED encoding, despite
// the "(workspace)" label referring to its own materialize-strategy
// name, not its own encoding -- see mm_fp32_recon_l2_error_vs_cublas_
// ref.cu's own header for the same naming note).
//
// Includes one non-exact-tile-multiple shape (1000x1000x1000) alongside
// exact-multiple shapes -- every promoted kernel here internally
// dispatches to a bounds-checked fallback path when the problem size
// isn't an exact multiple of its own tile constants (see each kernel's
// own comment in fp32_16_cuda_kernels.cu/mm_cuda_kernels.cu), so this is
// a real regression check on that fallback path, not just the fast path.
//
// Usage: verify_mm_fp32_recon_plot_promoted
// Exit code is non-zero if any check fails.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "mm_cpu_kernels.hpp"

using namespace tomos;

namespace tomos {
namespace cuda {
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, float* A, size_t ld_a, float* B,
                                                                           size_t ld_b, double beta, float* C,
                                                                           size_t ld_c);
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
}  // namespace cuda
}  // namespace tomos

namespace {

// Frobenius-relative-error pass/fail -- see mm_fp32_recon_l2_error_vs_
// cublas_ref.cu's own header for why this is textually identical to
// every GEVM verify file's own L2-relative-error check().
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
// codebase). tol=1e-3 (Frobenius-relative-error < 0.1%) for every check
// here -- this tier's own accuracy target is near-fp32 (full mant_hi+
// exponent reconstruction, or plain float for TOMOS FP32 itself), same
// tight bound verify_gemv_fp32_recon_plot_promoted.cu's own recon-tier
// checks use.
bool run_case(size_t m, size_t k, size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    char label[160];
    double alpha = 1.7, beta = 0.3;

    std::vector<float> A(m * k), B(k * n), C_init(m * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

    // ---- TOMOS FP32: gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
    // snake_midpoint_exact -- plain float A, no packed format. ----
    {
        float* d_A = nullptr;
        cudaMalloc(&d_A, A.size() * sizeof(float));
        cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<float> C_cpu = C_init;
        cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                             dim<2>{m, k}, A.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(),
                                             m);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_A,
            static_cast<size_t>(lda), d_B, static_cast<size_t>(ldb), beta, d_C, static_cast<size_t>(ldc));
        std::vector<float> C_gpu(m * n);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32 (gemm_fp32_..._midpoint_exact) %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        cudaFree(d_A);
    }

    // ---- TOMOS FP32/FP16 and TOMOS FP32/FP16 (workspace): both decode
    // via plain fp32_16_combine -- share cpu::mm_fp32_16_accurate as
    // their own exact CPU reference. ----
    {
        std::vector<uint16_t> wh(m * k), wr(m * k);
        for (size_t i = 0; i < m * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);
        uint16_t *d_wh = nullptr, *d_wr = nullptr;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr, wr.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr, wr.data(), wr.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        {
            std::vector<float> C_cpu = C_init;
            cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                                     wh.data(), wr.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);
            cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                   dim<2>{m, k}, d_wh, d_wr, static_cast<size_t>(lda), dim<2>{k, n},
                                                   d_B, static_cast<size_t>(ldb), beta, d_C,
                                                   static_cast<size_t>(ldc));
            std::vector<float> C_gpu(m * n);
            cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (tiled_cpasync_dualbuf) %s", tag);
            all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        }
        {
            std::vector<float> C_cpu = C_init;
            cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                                     wh.data(), wr.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);
            cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                             alpha, dim<2>{m, k}, d_wh, d_wr, static_cast<size_t>(lda),
                                                             dim<2>{k, n}, d_B, static_cast<size_t>(ldb), beta, d_C,
                                                             static_cast<size_t>(ldc));
            std::vector<float> C_gpu(m * n);
            cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (workspace) (gemm_materialize_midpoint_exact) %s",
                         tag);
            all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        }
        cudaFree(d_wh);
        cudaFree(d_wr);
    }

    // ---- TOMOS FP32/FP16 (truncated): requires wh/wr from fp32_16_
    // split_truncated specifically -- own host-side decode
    // (fp32_16_combine_truncated) + cpu::mm_strided reference. ----
    {
        std::vector<uint16_t> wh_t(m * k), wr_t(m * k);
        for (size_t i = 0; i < m * k; ++i) fp32_16_split_truncated(A[i], wh_t[i], wr_t[i]);
        std::vector<float> A_decoded(m * k);
        for (size_t i = 0; i < m * k; ++i) A_decoded[i] = fp32_16_combine_truncated(wh_t[i], wr_t[i]);

        uint16_t *d_wh_t = nullptr, *d_wr_t = nullptr;
        cudaMalloc(&d_wh_t, wh_t.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_t, wr_t.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_t, wh_t.data(), wh_t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_t, wr_t.data(), wr_t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> C_cpu = C_init;
        cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                             dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n}, B.data(), k, beta,
                                             C_cpu.data(), m);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                         alpha, dim<2>{m, k}, d_wh_t, d_wr_t,
                                                         static_cast<size_t>(lda), dim<2>{k, n}, d_B,
                                                         static_cast<size_t>(ldb), beta, d_C,
                                                         static_cast<size_t>(ldc));
        std::vector<float> C_gpu(m * n);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (truncated) (tiled_cpasync_dualbuf_truncated) %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        cudaFree(d_wh_t);
        cudaFree(d_wr_t);
    }

    // ---- TOMOS FP32/BF16 (truncated): plain (non-rounded) bf16 split
    // -- own host-side decode (fp32_bf16_combine) + cpu::mm_strided
    // reference (no cpu::mm_fp32_bf16_accurate-style helper exists). ----
    {
        std::vector<uint16_t> wh_b(m * k), wr_b(m * k);
        for (size_t i = 0; i < m * k; ++i) fp32_bf16_split(A[i], wh_b[i], wr_b[i]);
        std::vector<float> A_decoded(m * k);
        for (size_t i = 0; i < m * k; ++i) A_decoded[i] = fp32_bf16_combine(wh_b[i], wr_b[i]);

        uint16_t *d_wh_b = nullptr, *d_wr_b = nullptr;
        cudaMalloc(&d_wh_b, wh_b.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_b, wr_b.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_b, wh_b.data(), wh_b.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_b, wr_b.data(), wr_b.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> C_cpu = C_init;
        cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                             dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n}, B.data(), k, beta,
                                             C_cpu.data(), m);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, d_wh_b, d_wr_b,
            static_cast<size_t>(lda), dim<2>{k, n}, d_B, static_cast<size_t>(ldb), beta, d_C,
            static_cast<size_t>(ldc));
        std::vector<float> C_gpu(m * n);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (truncated) (syncpipe4_..._midpoint_exact) %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        cudaFree(d_wh_b);
        cudaFree(d_wr_b);
    }

    // ---- TOMOS FP32/BF16 (workspace): ROUNDED bf16 split (despite its
    // own "(workspace)" label referring to a materialize STRATEGY, not
    // its own encoding) -- own host-side decode (fp32_bf16_combine_
    // rounded) + cpu::mm_strided reference. ----
    {
        std::vector<uint16_t> wh_r(m * k), wr_r(m * k);
        for (size_t i = 0; i < m * k; ++i) fp32_bf16_split_rounded(A[i], wh_r[i], wr_r[i]);
        std::vector<float> A_decoded(m * k);
        for (size_t i = 0; i < m * k; ++i) A_decoded[i] = fp32_bf16_combine_rounded(wh_r[i], wr_r[i]);

        uint16_t *d_wh_r = nullptr, *d_wr_r = nullptr;
        cudaMalloc(&d_wh_r, wh_r.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_r, wr_r.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_r, wh_r.data(), wh_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_r, wr_r.data(), wr_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> C_cpu = C_init;
        cpu::mm_strided<float, float, float>(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                             dim<2>{m, k}, A_decoded.data(), m, dim<2>{k, n}, B.data(), k, beta,
                                             C_cpu.data(), m);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_gemm_materialize_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                    dim<2>{m, k}, d_wh_r, d_wr_r, static_cast<size_t>(lda),
                                                    dim<2>{k, n}, d_B, static_cast<size_t>(ldb), beta, d_C,
                                                    static_cast<size_t>(ldc));
        std::vector<float> C_gpu(m * n);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (workspace) (gemm_materialize_rounded) %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-3);
        cudaFree(d_wh_r);
        cudaFree(d_wr_r);
    }

    cudaFree(d_B);
    cudaFree(d_C);
    return all_ok;
}

}  // namespace

int main() {
    std::printf(
        "=== Correctness: TOMOS FP32-reconstruction GEMM plot's 6 promoted kernels (GPU vs CPU reference) ===\n");

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

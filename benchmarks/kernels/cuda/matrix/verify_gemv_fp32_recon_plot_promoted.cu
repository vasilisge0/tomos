// Standalone correctness verification for ONLY the five TOMOS kernels
// scripts/plot_gemv_fp32_recon_vs_cutlass_ref.m currently plots (the
// benchmark those GFlop/s numbers come from is mv_fp32_recon_vs_cutlass_
// ref.cu; verify_mv_fp32_16_gemv.cu is the full, much larger correctness
// suite) -- a focused, fast check to run after any change to one of
// these five specific dispatchers. Mirrors verify_gemv_fp32_16_plot_
// promoted.cu's own structure exactly, for this tier's own five kernels.
//
// PROMOTED_KERNEL (mirrors plot_gemv_fp32_recon_vs_cutlass_ref.m's own
// `series` struct, NO_TRANSPOSE only -- this tier's own scope; the
// separate TRANSPOSE tier is covered by verify_mv_fp32_transpose_
// promoted.cu):
//   TOMOS FP32              -> mv_fp32_tiled
//   TOMOS FP32/FP16          -> mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock
//   TOMOS FP32/FP16 (fast)   -> mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal
//   TOMOS FP32/BF16          -> mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded
//   TOMOS FP32/BF16 (workspace) -> mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace
//
// Unlike the "pure_half" tier's own verify_gemv_fp32_16_plot_promoted.cu,
// every kernel here does a FULL fp32 reconstruction decode (wh+wr, or
// wh-only for the safe-normal "(fast)" leg -- see below), so ALL FIVE
// checks use tight tolerances: this tier's own accuracy target is near-
// fp32, not fp16/bf16-level.
//
// mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal has an
// "always control=0" contract (see mv_fp32_recon_l2_error_vs_cublas_
// ref.cu's own gen_fp32_16_safe_normal comment) -- UNSAFE on arbitrary
// data, so it gets its own dedicated magnitude-restricted dataset below,
// same convention verify_gemv_fp32_16_plot_promoted.cu's own "(fast)"
// check uses for its tier's analogous kernel. Its own CPU reference is
// STILL the full cpu::mv_fp32_16_accurate decode (the "fast_normal"
// kernel-side optimization only skips a runtime branch, it doesn't
// change what a numerically-in-range input decodes to).
//
// TOMOS FP32/BF16 and TOMOS FP32/BF16 (workspace) both take a
// double* workspace parameter -- passed as nullptr here, same convention
// mv_fp32_recon_l2_error_vs_cublas_ref.cu's own calls use (see that
// file's own comment on why nullptr is safe for this benchmark's
// purposes). No cpu::mm_fp32_bf16_accurate-style helper exists for this
// codebase's bf16 CPU reference path (unlike the fp16 side's cpu::
// mv_fp32_16_accurate) -- both bf16 checks below decode wh/wr to a plain
// float array on the HOST (fp32_bf16_combine_rounded / fp32_bf16_combine
// respectively) and reuse cpu::mv_strided on that decoded array, same
// technique verify_mv_fp32_transpose_promoted.cu's own bf16 checks use.
//
// Usage: verify_gemv_fp32_recon_plot_promoted
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
#include "matrix_cpu_kernels.hpp"

using namespace tomos;

namespace tomos {
namespace cuda {
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A, size_t ld_a, const float* x,
                    size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y);
void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                   const uint16_t* wh, const uint16_t* wr,
                                                                   size_t ld_a, const float* x, size_t ld_x,
                                                                   double beta, float* y, size_t ld_y,
                                                                   double* workspace);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y,
                                                                     double* workspace);
}  // namespace cuda
}  // namespace tomos

namespace {

// L2-relative-error pass/fail, same convention/reasoning as verify_
// gemv_fp32_16_plot_promoted.cu's own check() -- see that file's own
// comment for the per-element-metric failure mode this avoids.
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
    double l2_rel_err = (sum_sq_cpu > 0.0) ? std::sqrt(sum_sq_err / sum_sq_cpu) : std::sqrt(sum_sq_err);
    bool ok = l2_rel_err < tol;
    std::printf("  %-58s l2_rel_err=%.6g max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(), l2_rel_err,
               max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("    [diag] worst_i=%zu cpu=%.6f gpu=%.6f (of %zu elements)\n", worst_i,
                   static_cast<double>(cpu[worst_i]), static_cast<double>(gpu[worst_i]), cpu.size());
    }
    return ok;
}

// NO_TRANSPOSE only -- this tier's own scope (see this file's own
// header). tol=1e-3 (L2-relative-error < 0.1%) for every check: this
// tier's own accuracy target is near-fp32 (a full mant_hi+exponent
// reconstruction decode, not a truncating fp16/bf16-level one), so a
// much tighter bound than the "pure_half" tier's own 5e-2 is appropriate
// and expected to pass comfortably.
bool run_case(size_t rows, size_t cols, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    size_t m = rows, k = cols;
    size_t ld_a = m;
    char label[128];
    double alpha = 1.7, beta = 0.3;

    std::vector<float> A(ld_a * k), x(k), y_init(m);
    for (auto& v : A) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_y, y_init.size() * sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    // ---- TOMOS FP32: mv_fp32_tiled -- plain float A, no packed format
    // involved. ----
    {
        float* d_A = nullptr;
        cudaMalloc(&d_A, A.size() * sizeof(float));
        cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, A.data(), ld_a,
                                             x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_tiled(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, d_A, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(m);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32 (mv_fp32_tiled) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 1e-3);
        cudaFree(d_A);
    }

    // ---- TOMOS FP32/FP16: mv_fp32_16_accurate_ready4_atomiconly_
    // scaled_kblock -- full wh/wr reconstruction decode, safe on
    // arbitrary data. ----
    {
        std::vector<uint16_t> wh(ld_a * k), wr(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);
        uint16_t *d_wh = nullptr, *d_wr = nullptr;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr, wr.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr, wr.data(), wr.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, wh.data(), wr.data(), ld_a,
                                 x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                                                                    d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(m);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (ready4_atomiconly_scaled_kblock) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 1e-3);
        cudaFree(d_wh);
        cudaFree(d_wr);
    }

    // ---- TOMOS FP32/FP16 (fast): mv_fp32_16_accurate_ready_
    // atomiconly_scaled_kblock_fast_normal -- UNSAFE on arbitrary data
    // ("always control=0" contract), own dedicated magnitude-in-
    // [0.5,1.0] dataset. CPU reference is still the FULL cpu::
    // mv_fp32_16_accurate decode (see this file's own header). ----
    {
        std::mt19937 safe_rng(43);
        std::uniform_real_distribution<float> mag_dist(0.5f, 1.0f);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::vector<float> A_safe(ld_a * k);
        for (auto& v : A_safe) {
            float mag = mag_dist(safe_rng);
            v = sign_dist(safe_rng) ? mag : -mag;
        }
        std::vector<uint16_t> wh_safe(ld_a * k), wr_safe(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_16_split(A_safe[i], wh_safe[i], wr_safe[i]);
        uint16_t *d_wh_safe = nullptr, *d_wr_safe = nullptr;
        cudaMalloc(&d_wh_safe, wh_safe.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_safe, wr_safe.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_safe, wh_safe.data(), wh_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_safe, wr_safe.data(), wr_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, wh_safe.data(), wr_safe.data(),
                                 ld_a, x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(
            TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(m);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (fast) (ready_atomiconly_scaled_kblock_fast_normal) %s",
                     tag);
        all_ok &= check(label, y_cpu, y_gpu, 1e-3);
        cudaFree(d_wh_safe);
        cudaFree(d_wr_safe);
    }

    // ---- TOMOS FP32/BF16: mv_fp32_bf16_accurate_ready_atomiconly_
    // scaled_kblock_rounded -- ROUNDED bf16 encoding, own host-side
    // decode + cpu::mv_strided reference (see this file's own header for
    // why). workspace passed as nullptr. ----
    {
        std::vector<uint16_t> wh_r(ld_a * k), wr_r(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_bf16_split_rounded(A[i], wh_r[i], wr_r[i]);
        std::vector<float> A_decoded(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) A_decoded[i] = fp32_bf16_combine_rounded(wh_r[i], wr_r[i]);

        uint16_t *d_wh_r = nullptr, *d_wr_r = nullptr;
        cudaMalloc(&d_wh_r, wh_r.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_r, wr_r.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_r, wh_r.data(), wh_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_r, wr_r.data(), wr_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, A_decoded.data(),
                                             ld_a, x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(
            TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, d_wh_r, d_wr_r, ld_a, d_x, 1, beta, d_y, 1, nullptr);
        std::vector<float> y_gpu(m);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (ready_atomiconly_scaled_kblock_rounded) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 1e-3);
        cudaFree(d_wh_r);
        cudaFree(d_wr_r);
    }

    // ---- TOMOS FP32/BF16 (workspace): mv_fp32_bf16_accurate_ready_
    // atomiconly_scaled_kblock_workspace -- PLAIN (non-rounded) bf16
    // encoding, own host-side decode + cpu::mv_strided reference.
    // workspace passed as nullptr. ----
    {
        std::vector<uint16_t> wh_p(ld_a * k), wr_p(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_bf16_split(A[i], wh_p[i], wr_p[i]);
        std::vector<float> A_decoded(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) A_decoded[i] = fp32_bf16_combine(wh_p[i], wr_p[i]);

        uint16_t *d_wh_p = nullptr, *d_wr_p = nullptr;
        cudaMalloc(&d_wh_p, wh_p.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_p, wr_p.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_p, wh_p.data(), wh_p.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_p, wr_p.data(), wr_p.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, A_decoded.data(),
                                             ld_a, x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
            TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k}, d_wh_p, d_wr_p, ld_a, d_x, 1, beta, d_y, 1, nullptr);
        std::vector<float> y_gpu(m);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (workspace) (ready_atomiconly_scaled_kblock_workspace) %s",
                     tag);
        all_ok &= check(label, y_cpu, y_gpu, 1e-3);
        cudaFree(d_wh_p);
        cudaFree(d_wr_p);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return all_ok;
}

}  // namespace

int main() {
    std::printf("=== Correctness: TOMOS FP32-reconstruction GEMV plot's 5 promoted kernels (GPU vs CPU reference) ===\n");

    // Same shape-scale reasoning as verify_gemv_fp32_16_plot_promoted.cu's
    // own cases list -- a correctness check needs the WHOLE ld_a*k matrix
    // on the HOST (unlike the device-only performance sweep), so shapes
    // stay well below mv_fp32_recon_vs_cutlass_ref.cu's own multi-billion-
    // element performance-sweep scale.
    struct Case { size_t rows, cols; const char* tag; };
    std::vector<Case> cases = {
        {2048, 2048, "square-small"},
        {8192, 8192, "square-mid"},
        {16384, 16384, "square-large"},
        {131072, 512, "tall-thin-K512-large"},
        {65536, 4096, "tall-thin-K4096-large"},
    };

    bool all_ok = true;
    for (const auto& c : cases) all_ok &= run_case(c.rows, c.cols, c.tag);

    std::printf("%s\n", all_ok ? "Correctness: ALL CHECKS PASSED"
                               : "Correctness: SOME CHECKS FAILED (see FAIL lines above)");
    return all_ok ? 0 : 1;
}

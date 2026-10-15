// Standalone correctness verification for ONLY the four TOMOS kernels
// scripts/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.m
// currently plots (the benchmark those GFlop/s numbers come from is
// mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu) -- a focused, fast
// check to run after any change to one of these four specific
// dispatchers. Mirrors verify_gemv_fp32_16_plot_promoted.cu's own
// structure (this tier's own NO_TRANSPOSE sibling), for the TRANSPOSE
// direction's own four kernels.
//
// PROMOTED_KERNEL (mirrors plot_gemv_transpose_fp32_16_pure_half_vs_
// cublas_ref.m's own current, non-commented-out `series` struct --
// verified this session against that file directly, since this tier has
// accumulated many more experimental variants than its NO_TRANSPOSE
// sibling and only these four are actually plotted):
//   TOMOS FP16              -> mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split
//   TOMOS FP32/FP16          -> mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired
//   TOMOS FP32/FP16 (fast)   -> mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast
//   TOMOS FP32/BF16          -> mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4
//
// TOMOS FP16 needs its own float* scratch workspace, sized via its own
// _workspace_size helper -- unlike every other kernel here.
//
// TOMOS FP32/FP16 (fast) has fp32_16_fast_half's own "always control=0"
// contract (UNSAFE on arbitrary data), so it gets its own dedicated
// magnitude-restricted dataset below, same convention verify_gemv_
// fp32_16_plot_promoted.cu's own "(fast)" check uses.
//
// TOMOS FP32/FP16 (hybrid) decodes via the SAME wh-only extended-
// exponent format as this tier's NO_TRANSPOSE sibling's own hybrid
// check (mv_fp32_16_hybrid_process_element/_branchless family, shared
// fp32_16_pack.hpp encoding constants) -- the host-side decode logic
// below is copied verbatim from verify_gemv_fp32_16_plot_promoted.cu's
// own hybrid check.
//
// TOMOS FP32/BF16 is safe on arbitrary data (plain truncating bf16
// decode) -- no cpu::mm_fp32_bf16-style helper exists in this codebase
// (confirmed by grep), so its own reference decodes wh host-side
// (fp32_bf16_bf16_only) then reuses cpu::mv_strided, same technique
// verify_gemv_fp32_16_plot_promoted.cu's own bf16 checks use.
//
// Usage: verify_gemv_fp32_16_pure_half_transpose_plot_promoted
// Exit code is non-zero if any check fails.

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

#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "matrix_cpu_kernels.hpp"

using namespace tomos;

namespace tomos {
namespace cuda {
namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size); }  // namespace experimental

void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                         dim<2> size, const uint16_t* wh,
                                                                         size_t ld_a, const float* x, size_t ld_x,
                                                                         double beta, float* y, size_t ld_y,
                                                                         float* workspace);
void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(TransposeType trans, double alpha,
                                                                        dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                        const float* x, size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);
void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, size_t ld_x, double beta,
                                                                       float* y, size_t ld_y);
void mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
}  // namespace cuda
}  // namespace tomos

namespace {

// L2-relative-error pass/fail, same convention as verify_gemv_fp32_16_
// plot_promoted.cu's own check().
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

// TRANSPOSE only -- this tier's own scope. x has length M, y has length
// K (swapped from NO_TRANSPOSE), A stored M x K column-major (ld_a=m),
// same convention every TRANSPOSE file in this codebase uses. tol=5e-2
// (L2-relative-error < 5%), same bound this tier's NO_TRANSPOSE sibling
// uses for its own fp16/bf16-compute-tier checks.
bool run_case(size_t rows, size_t cols, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    size_t m = rows, k = cols;
    size_t ld_a = m;
    char label[128];
    double alpha = 1.7, beta = 0.3;

    std::vector<float> A(ld_a * k), x(m), y_init(k);
    for (auto& v : A) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_y, y_init.size() * sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    // ---- TOMOS FP16: mv_fp32_16_pure_half_transpose_atomiconly_
    // scaled_mblock_vec4_split -- safe on arbitrary data, own float*
    // scratch workspace. ----
    {
        std::vector<uint16_t> wh(ld_a * k), wr_scratch(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_16_split(A[i], wh[i], wr_scratch[i]);
        uint16_t* d_wh = nullptr;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_pure_half(TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, wh.data(), ld_a, x.data(), 1, beta,
                                  y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);

        size_t ws_elems = cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
            dim<2>{m, k});
        float* d_ws = nullptr;
        cudaMalloc(&d_ws, ws_elems * sizeof(float));
        cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType::TRANSPOSE, alpha,
                                                                                  dim<2>{m, k}, d_wh, ld_a, d_x, 1,
                                                                                  beta, d_y, 1, d_ws);
        std::vector<float> y_gpu(k);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP16 (pure_half_transpose_..._vec4_split) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);
        cudaFree(d_ws);
        cudaFree(d_wh);
    }

    // ---- TOMOS FP32/FP16: mv_fp32_16_hybrid_transpose_atomiconly_
    // scaled_mblock_vec4_paired -- own mixed normal/extended-range data,
    // own host-side wh-only decode (copied verbatim from verify_gemv_
    // fp32_16_plot_promoted.cu's own hybrid check). ----
    {
        std::mt19937 hybrid_rng(73);
        std::uniform_real_distribution<float> hybrid_normal_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> hybrid_extended_mag_dist(1.0e5f, 1.0e8f);
        std::uniform_int_distribution<int> hybrid_extended_flag(0, 9);
        std::uniform_int_distribution<int> hybrid_sign_dist(0, 1);
        std::vector<float> A_hybrid(ld_a * k);
        for (auto& v : A_hybrid) {
            if (hybrid_extended_flag(hybrid_rng) == 0) {
                float mag = hybrid_extended_mag_dist(hybrid_rng);
                v = hybrid_sign_dist(hybrid_rng) ? mag : -mag;
            } else {
                v = hybrid_normal_dist(hybrid_rng);
            }
        }
        std::vector<uint16_t> wh_hybrid(ld_a * k), wr_hybrid_scratch(ld_a * k);
        for (size_t idx = 0; idx < ld_a * k; ++idx) fp32_16_split(A_hybrid[idx], wh_hybrid[idx], wr_hybrid_scratch[idx]);
        std::vector<float> A_hybrid_decoded(ld_a * k);
        for (size_t idx = 0; idx < ld_a * k; ++idx) {
            const uint16_t whv = wh_hybrid[idx];
            if ((whv & 0x1u) == 0u) {
                A_hybrid_decoded[idx] = __half2float(*reinterpret_cast<const __half*>(&whv));
            } else {
                const uint32_t offset = (static_cast<uint32_t>(whv) >> 10) & 0xFu;
                if (offset != fp32_16_detail::kOffsetSaturate) {
                    const uint32_t direction = (static_cast<uint32_t>(whv) >> 14) & 0x1u;
                    const uint32_t exp32 = direction ? (fp32_16_detail::kFp32ExpMax + offset)
                                                      : (fp32_16_detail::kFp32ExpMin - offset);
                    const uint32_t mant_hi = (static_cast<uint32_t>(whv) >> 1) & 0x1FFu;
                    const uint32_t mant32 = mant_hi << 14;
                    const uint32_t sign = (static_cast<uint32_t>(whv) & 0x8000u) << 16;
                    A_hybrid_decoded[idx] = fp32_16_detail::bits_to_float(sign | (exp32 << 23) | mant32);
                } else {
                    A_hybrid_decoded[idx] = 0.0f;
                }
            }
        }
        uint16_t* d_wh_hybrid = nullptr;
        cudaMalloc(&d_wh_hybrid, wh_hybrid.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_hybrid, wh_hybrid.data(), wh_hybrid.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, A_hybrid_decoded.data(),
                                             ld_a, x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(TransposeType::TRANSPOSE, alpha,
                                                                                dim<2>{m, k}, d_wh_hybrid, ld_a, d_x,
                                                                                1, beta, d_y, 1);
        std::vector<float> y_gpu(k);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (hybrid_transpose_..._vec4_paired) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);
        cudaFree(d_wh_hybrid);
    }

    // ---- TOMOS FP32/FP16 (fast): mv_fp32_16_pure_half_transpose_
    // atomiconly_scaled_mblock_vec4_fast -- UNSAFE on arbitrary data
    // (fp32_16_fast_half's own contract), own dedicated magnitude-in-
    // [0.5,1.0] dataset. ----
    {
        std::mt19937 safe_rng(43);
        std::uniform_real_distribution<float> mag_dist(0.5f, 1.0f);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::vector<float> A_safe(ld_a * k);
        for (auto& v : A_safe) {
            float mag = mag_dist(safe_rng);
            v = sign_dist(safe_rng) ? mag : -mag;
        }
        std::vector<uint16_t> wh_safe(ld_a * k), wr_safe_scratch(ld_a * k);
        for (size_t i = 0; i < ld_a * k; ++i) fp32_16_split(A_safe[i], wh_safe[i], wr_safe_scratch[i]);
        uint16_t* d_wh_safe = nullptr;
        cudaMalloc(&d_wh_safe, wh_safe.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_safe, wh_safe.data(), wh_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_pure_half(TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, wh_safe.data(), ld_a, x.data(), 1,
                                  beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(
            TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, d_wh_safe, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(k);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (fast) (pure_half_transpose_..._vec4_fast) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);
        cudaFree(d_wh_safe);
    }

    // ---- TOMOS FP32/BF16: mv_fp32_bf16_pure_bf16_transpose_
    // atomiconly_scaled_mblock_vec4 -- safe on arbitrary data (plain
    // truncating bf16 decode), own host-side decode + cpu::mv_strided
    // reference (no cpu::mm_fp32_bf16-style helper exists). ----
    {
        std::vector<uint16_t> wh_bf16(ld_a * k), wr_bf16_scratch(ld_a * k);
        for (size_t idx = 0; idx < ld_a * k; ++idx) fp32_bf16_split(A[idx], wh_bf16[idx], wr_bf16_scratch[idx]);
        std::vector<float> A_bf16_decoded(ld_a * k);
        for (size_t idx = 0; idx < ld_a * k; ++idx)
            A_bf16_decoded[idx] = __bfloat162float(fp32_bf16_bf16_only(wh_bf16[idx]));
        uint16_t* d_wh_bf16 = nullptr;
        cudaMalloc(&d_wh_bf16, wh_bf16.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_bf16, wh_bf16.data(), wh_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, A_bf16_decoded.data(),
                                             ld_a, x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType::TRANSPOSE, alpha,
                                                                              dim<2>{m, k}, d_wh_bf16, ld_a, d_x, 1,
                                                                              beta, d_y, 1);
        std::vector<float> y_gpu(k);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (pure_bf16_transpose_..._vec4) %s", tag);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);
        cudaFree(d_wh_bf16);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return all_ok;
}

}  // namespace

int main() {
    std::printf(
        "=== Correctness: TOMOS FP32/FP16 TRANSPOSE GEMV plot's 4 promoted kernels (GPU vs CPU reference) ===\n");

    struct Case { size_t rows, cols; const char* tag; };
    std::vector<Case> cases = {
        {2048, 2048, "square-small"},
        {8192, 8192, "square-mid"},
        {16384, 16384, "square-large"},
        {512, 131072, "tall-thin-M512-large"},
        {4096, 65536, "tall-thin-M4096-large"},
    };

    bool all_ok = true;
    for (const auto& c : cases) all_ok &= run_case(c.rows, c.cols, c.tag);

    std::printf("%s\n", all_ok ? "Correctness: ALL CHECKS PASSED"
                               : "Correctness: SOME CHECKS FAILED (see FAIL lines above)");
    return all_ok ? 0 : 1;
}

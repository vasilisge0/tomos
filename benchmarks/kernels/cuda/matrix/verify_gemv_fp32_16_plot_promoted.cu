// Standalone correctness verification for ONLY the five TOMOS kernels
// scripts/plot_gemv_fp32_16_pure_half_vs_cublas_ref.m currently plots (the
// benchmark those GFlop/s numbers come from is mv_fp32_16_pure_half_vs_
// cublas_ref.cu; verify_mv_fp32_16_gemv.cu is the full, much larger
// correctness suite covering every exploratory/"test"/negative-result
// kernel this session built along the way, including this family's own
// intmask/walign siblings) -- a focused, fast check to run after any
// change to one of these five specific dispatchers, without paying for
// the ~200 other checks the full suite carries.
//
// Covers exactly the five kernels the plot's own five TOMOS series
// currently resolve to (see plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's
// own header for how each series maps to a kernel -- update this list here
// too if that mapping ever changes, e.g. if intmask_best is promoted):
//   TOMOS FP16            -> mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best
//   TOMOS FP32/FP16       -> mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best
//   TOMOS FP32/FP16 (fast) -> mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best
//   TOMOS FP32/BF16       -> mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2
//   TOMOS BF16            -> mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best
// (cuBLAS FP16/BF16, the plot's other two series, are vendor reference
// implementations, not our own kernels -- nothing to check there.)
//
// Shapes bracket each dispatcher's own row-count crossover threshold
// (currently 4096 for all three "_best" families involved here, though
// each has its own separately-named constant -- see fp32_16_cuda_
// kernels.cu's kMv*BestCrossoverThreshold family): below it (bigtile
// leg), at/above it (wide4x leg), plus a large square and two tall-and-
// thin shapes. Deliberately NOT scaled up to match scripts/benchmarks_
// gemv_fp32_16_pure_half_vs_cublas_ref.sh's own performance-sweep sizes
// (up to 1048576x4096): that sweep is device-only and never needs the
// matrix on the host, but a correctness check inherently does (for the
// CPU reference), and at the sweep's own largest shapes that host array
// alone is tens of GB -- see main()'s own comment on its cases list for
// the OOM this hit at 65536x65536 before being scaled down. Each (rows,
// cols) case below runs BOTH TransposeType directions, which conveniently
// exercises BOTH legs at once for a tall-and-thin case (e.g. 131072x512:
// NO_TRANSPOSE gives rows_op=131072 >= threshold (wide4x), TRANSPOSE
// gives rows_op=512 < threshold (bigtile)).
//
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best is
// UNSAFE on arbitrary data (fp32_16_fast_half's own contract -- see its
// comment in fp32_16_cuda_kernels.cu), so it gets its own dedicated
// magnitude-restricted dataset below, not the plain data every other
// kernel here uses.
//
// Usage: verify_gemv_fp32_16_plot_promoted
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

namespace {

// Pass/fail is driven by the WHOLE-VECTOR L2-relative-error (||cpu-gpu||_2
// / ||cpu||_2), not a per-element max relative error: at these problem
// sizes (K up to 65536), a per-element metric with a max(1.0, |cpu|)
// floor lets a single row whose true value happens to land small (by
// ordinary statistical scatter -- a random +-1 signal summed over K terms
// has typical magnitude ~sqrt(K), but individual rows scatter around
// that, and plenty land well below it) blow up into a "relative error"
// over 1.0 from perfectly ordinary rounding noise, forcing the tolerance
// so loose (tol=5.0, seen accepting max_rel_err up to ~1.8 as "OK") that
// it stops meaningfully testing anything. The L2 norm averages over all
// elements instead of being driven by the single worst one, so it can't
// be blown up by an isolated near-zero row the same way, and can be set
// genuinely tight. The per-element max_abs_err/max_rel_err are still
// computed and printed as a debugging aid (which row is worst, by how
// much) -- just not what pass/fail is decided on.
bool check(const std::string& label, const std::vector<float>& cpu,
          const std::vector<float>& gpu, double tol) {
    cudaError_t pending_err = cudaGetLastError();
    if (pending_err != cudaSuccess) {
        std::printf("  [diag] CUDA ERROR pending before checking %-40s : %s\n",
                   label.c_str(), cudaGetErrorString(pending_err));
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
    std::printf("  %-58s l2_rel_err=%.6g max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(),
               l2_rel_err, max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("    [diag] worst_i=%zu cpu=%.6f gpu=%.6f (of %zu elements)\n",
                   worst_i, static_cast<double>(cpu[worst_i]), static_cast<double>(gpu[worst_i]), cpu.size());
    }
    return ok;
}

// Runs all five checks for one (rows, cols) case, both TransposeType
// directions. tol=5e-2 (L2-relative-error < 5%) for every check here --
// DELIBERATELY tighter than verify_mv_fp32_16_gemv.cu's own tol=5.0 (max
// PER-ELEMENT relative error < 500%) for this same "_best"/"_atomic"
// tier: that per-element metric's max(1.0, |cpu|) floor lets a single row
// whose true value happens to land small blow up into a huge "relative
// error" from ordinary rounding noise alone (see check()'s own comment),
// which is why that tolerance had to be so loose it stopped meaningfully
// testing anything -- confirmed empirically here (max_rel_err up to ~1.8
// accepted as "OK" under the old per-element metric at 8192x8192, before
// this switch). The L2 metric this file's own check() now uses doesn't
// have that failure mode, so 5% is a real, tight bound, not a rubber
// stamp -- a starting point, not yet confirmed against every shape this
// file's own cases list covers.
bool run_case(size_t rows, size_t cols, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    for (TransposeType trans : {TransposeType::NO_TRANSPOSE, TransposeType::TRANSPOSE}) {
        size_t m = rows, n = cols;
        size_t ld_a = m;
        size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
        size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
        const char* dir = (trans == TransposeType::NO_TRANSPOSE) ? "N" : "T";
        char label[128];
        double alpha = 1.7, beta = 0.3;

        std::vector<float> A(ld_a * n), x(cols_op), y_init(rows_op);
        for (auto& v : A) v = dist(rng);
        for (auto& v : x) v = dist(rng);
        for (auto& v : y_init) v = dist(rng);

        float* d_x = nullptr;
        float* d_y = nullptr;
        cudaMalloc(&d_x, x.size() * sizeof(float));
        cudaMalloc(&d_y, y_init.size() * sizeof(float));
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

        // ---- TOMOS FP16: mv_fp32_16_pure_half_wide_ilp_shared_half2_
        // atomic_best -- safe on arbitrary data, plain wh/x above. ----
        {
            std::vector<uint16_t> wh(ld_a * n), wr_scratch(ld_a * n);
            for (size_t i = 0; i < ld_a * n; ++i) fp32_16_split(A[i], wh[i], wr_scratch[i]);
            uint16_t* d_wh = nullptr;
            cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                          d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "TOMOS FP16 (pure_half_..._atomic_best) %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 5e-2);
            cudaFree(d_wh);
        }

        // ---- TOMOS FP32/FP16: mv_fp32_16_hybrid_wide_ilp_shared_half2_
        // atomic_branchless_best -- own mixed normal/extended-range data
        // (this decode's own two-tier contract needs both regimes
        // exercised), same generator/host-decode replication used by
        // verify_mv_fp32_16_gemv.cu's own hybrid_wide checks. ----
        {
            std::mt19937 hybrid_rng(73);
            std::uniform_real_distribution<float> hybrid_normal_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> hybrid_extended_mag_dist(1.0e5f, 1.0e8f);
            std::uniform_int_distribution<int> hybrid_extended_flag(0, 9);
            std::uniform_int_distribution<int> hybrid_sign_dist(0, 1);
            std::vector<float> A_hybrid(ld_a * n);
            for (auto& v : A_hybrid) {
                if (hybrid_extended_flag(hybrid_rng) == 0) {
                    float mag = hybrid_extended_mag_dist(hybrid_rng);
                    v = hybrid_sign_dist(hybrid_rng) ? mag : -mag;
                } else {
                    v = hybrid_normal_dist(hybrid_rng);
                }
            }
            std::vector<uint16_t> wh_hybrid(ld_a * n), wr_hybrid_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_hybrid[idx], wh_hybrid[idx], wr_hybrid_scratch[idx]);
            std::vector<float> A_hybrid_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) {
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
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(
                trans, alpha, dim<2>{m, n}, d_wh_hybrid, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (hybrid_..._branchless_best) %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 5e-2);
            cudaFree(d_wh_hybrid);
        }

        // ---- TOMOS FP32/FP16 (fast): mv_fp32_16_pure_half_wide_ilp_
        // shared_half2_atomic_fasthalf_best -- UNSAFE on arbitrary data
        // (fp32_16_fast_half's own contract), so uses its own dedicated
        // magnitude-in-[0.5,1.0] dataset instead of the plain A/x above,
        // same generator convention verify_mv_fp32_16_gemv.cu's own
        // fasthalf_atomic check uses. ----
        {
            std::mt19937 fasthalf_rng(44);
            std::uniform_real_distribution<float> fasthalf_mag_dist(0.5f, 1.0f);
            std::uniform_int_distribution<int> fasthalf_sign_dist(0, 1);
            std::vector<float> A_fasthalf(ld_a * n);
            for (auto& v : A_fasthalf) {
                float mag = fasthalf_mag_dist(fasthalf_rng);
                v = fasthalf_sign_dist(fasthalf_rng) ? mag : -mag;
            }
            std::vector<uint16_t> wh_fasthalf(ld_a * n), wr_fasthalf_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_fasthalf[idx], wh_fasthalf[idx], wr_fasthalf_scratch[idx]);
            uint16_t* d_wh_fasthalf = nullptr;
            cudaMalloc(&d_wh_fasthalf, wh_fasthalf.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_fasthalf, wh_fasthalf.data(), wh_fasthalf.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fasthalf.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(
                trans, alpha, dim<2>{m, n}, d_wh_fasthalf, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "TOMOS FP32/FP16 (fast) (..._fasthalf_best) %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 5e-2);
            cudaFree(d_wh_fasthalf);
        }

        // ---- TOMOS FP32/BF16 and TOMOS BF16: mv_fp32_bf16_pure_bf16_
        // wide_ilp_shared_atomic_best_v2 / ..._atomic_best -- genuinely
        // identical decode/compute (bf16 has no fast/safe split, see
        // fp32_bf16_bf16_only's own comment), safe on arbitrary data, so
        // both share the same plain A/x-derived bf16 data. ----
        {
            std::vector<uint16_t> wh_bf16(ld_a * n), wr_bf16_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) fp32_bf16_split(A[idx], wh_bf16[idx], wr_bf16_scratch[idx]);
            std::vector<float> A_bf16_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                A_bf16_decoded[idx] = __bfloat162float(fp32_bf16_bf16_only(wh_bf16[idx]));
            uint16_t* d_wh_bf16 = nullptr;
            cudaMalloc(&d_wh_bf16, wh_bf16.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_bf16, wh_bf16.data(), wh_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            {
                std::vector<float> y_cpu = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(trans, alpha, dim<2>{m, n}, d_wh_bf16,
                                                                            ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "TOMOS FP32/BF16 (..._atomic_best_v2) %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-2);
            }
            {
                std::vector<float> y_cpu = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(trans, alpha, dim<2>{m, n}, d_wh_bf16, ld_a,
                                                                         d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "TOMOS BF16 (..._atomic_best) %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-2);
            }
            cudaFree(d_wh_bf16);
        }

        cudaFree(d_x);
        cudaFree(d_y);
    }
    return all_ok;
}

}  // namespace

int main() {
    std::printf("=== Correctness: TOMOS FP32/FP16 GEMV plot's 5 promoted kernels (GPU vs CPU reference) ===\n");

    // Brackets each dispatcher's row-count crossover threshold (below,
    // at/above), plus large-scale square and tall-and-thin shapes
    // matching scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh's
    // own SQUARE_MS/TALL_KS/TALL_MS sweep -- see this file's own header.
    struct Case { size_t rows, cols; const char* tag; };
    // Kept well below scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_
    // ref.sh's own performance-sweep scale (up to 1048576x4096) ON
    // PURPOSE: unlike that sweep (device-only, no host copy of A ever
    // needed), a correctness check inherently needs the WHOLE ld_a*n
    // matrix on the HOST too, for cpu::mv_strided/cpu::mv_fp32_16_pure_
    // half to compute the reference -- at 65536x65536 that's 4.3 billion
    // elements, 17GB for a single float array, and the hybrid-tier check
    // alone keeps ~4 such arrays live at once (peak 60GB+), which is what
    // actually OOM-killed this script's first version. verify_mv_fp32_16_
    // gemv.cu's own correctness cases never exceed 8192x8192 (67M
    // elements, ~270MB/array) for the exact same reason -- these mirror
    // that same safe scale, stepped up by a further 4x for "large" (still
    // ~1GB/array, ~4-5GB peak per block) rather than matching the
    // performance sweep's own multi-billion-element shapes.
    std::vector<Case> cases = {
        {2048, 2048, "square-below-threshold"},
        {8192, 8192, "square-above-threshold"},
        {16384, 16384, "square-large"},
        {131072, 512, "tall-thin-K512-large"},
        {65536, 4096, "tall-thin-K4096-at-threshold-and-large"},
    };

    bool all_ok = true;
    for (const auto& c : cases) all_ok &= run_case(c.rows, c.cols, c.tag);

    std::printf("%s\n", all_ok ? "Correctness: ALL CHECKS PASSED"
                               : "Correctness: SOME CHECKS FAILED (see FAIL lines above)");
    return all_ok ? 0 : 1;
}

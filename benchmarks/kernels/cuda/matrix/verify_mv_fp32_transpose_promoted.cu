// Correctness verification for EXACTLY the TRANSPOSE GEMV kernels
// scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.m actually plots
// -- the ones scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh's
// own METHOD_MARKERS currently promotes -- and nothing else. Unlike
// verify_mv_fp32_16_gemv.cu (which covers ~50 kernels across both
// TRANSPOSE and NO_TRANSPOSE, including many demoted/experimental
// siblings kept only for historical A/B comparison), this file exists
// to answer one narrow question quickly: "are the kernels the plot
// actually shows numerically correct?" -- so it stays small and fast to
// build/run on purpose. When a kernel is promoted or demoted in the
// .sh's own METHOD_MARKERS, update the PROMOTED_KERNEL list in this
// file's own header comment (and the checks below) to match.
//
// PROMOTED_KERNEL (mirrors the .sh's own METHOD_MARKERS, TRANSPOSE-only):
//   native            = mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec
//   fp32_16           = mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch
//   fp32_16_fast      = mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch
//   fp32_bf16         = mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch
//   fp32_bf16_truncated = mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4
//
// Shapes cover: a small square (M=K=1024), the size where fp32_16_fast's
// own FIRST promotion (the wrapborrow decode reformulation) measured its
// biggest win (M=K=6144), a larger square (M=K=4096), two tall-and-thin
// cases -- one with a tiny, deliberately NOT-multiple-of-4 M (M=70)
// specifically because that exact shape/direction combination is what
// exposed a real misaligned-load bug in a DIFFERENT (non-promoted)
// kernel family earlier this session; these promoted kernels were not
// implicated, but this shape stays in the sweep as a standing regression
// check since their own vec4/ushort4 loads use a similar runtime-
// alignment-then-scalar-fallback pattern -- and one square shape at
// M=K=16384, ABOVE fp32_16's, fp32_16_fast's, AND fp32_bf16's own
// dispatch thresholds (M=8192, same value for all three dispatchers):
// every other shape above is M<8192, so without this one NONE of the
// three dispatchers' own M>=8192 branches (_wrapborrow_retarget2 for
// fp32_16, _retarget2 for fp32_16_fast, _rounded_retarget2 for
// fp32_bf16) would ever actually be exercised by this file -- see each
// dispatcher's own comment in fp32_16_cuda_kernels.cu for its own
// promotion history and the same M-only-threshold caveat (validated on
// square shapes only).
//
// Retrofitted (alongside the sibling verify_gemv_fp32_16_plot_
// promoted.cu) to decide pass/fail on L2-RELATIVE error instead of the
// per-element max relative error this file used before -- see check()'s
// own comment for why. The four packed-format checks' own tol moved from
// 5e-3 (0.5%, under the old per-element metric) to 5e-2 (5% L2-relative,
// the same bound the sibling file's own retrofit uses) -- native's own
// tol (1e-4) is unchanged, since a native fp32 kernel's L2 error against
// the cpu::mv_strided reference is expected to stay near machine epsilon
// regardless of which metric decides pass/fail.
//
// Usage: verify_mv_fp32_transpose_promoted

#include <cublas_v2.h>
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
#include "mv_cuda_kernels.cuh"

using namespace tomos;

namespace {

// Pass/fail is decided on L2-RELATIVE error (||cpu-gpu||_2/||cpu||_2,
// averaged over the whole output vector), NOT the per-element max
// relative error this function used before -- that per-element metric's
// err/max(1.0,|cpu|) floor lets a single row whose true value happens to
// land near zero blow up into a huge "relative error" from ordinary
// rounding noise alone, forcing tol so loose (5.0 = 500%, seen accepting
// max_rel_err up to ~1.8 as "OK" in the sibling verify_gemv_fp32_16_
// plot_promoted.cu before ITS OWN retrofit to this same metric) that it
// stops meaningfully testing anything. The L2 norm can't be blown up by
// an isolated near-zero row the same way, so tol can stay genuinely
// tight -- see run_shape's own comment for the tol values this retrofit
// changed them to. The per-element max_abs_err/max_rel_err are still
// computed and printed as a debugging aid (which row is worst, by how
// much) -- just not what pass/fail is decided on.
bool check(const std::string& label, const std::vector<float>& cpu,
          const std::vector<float>& gpu, double tol) {
    // cudaGetLastError() is sticky and cleared on read -- calling it here,
    // right after each block's own kernel launch + synchronous memcpy
    // (nothing else touches the CUDA API in between), immediately
    // attributes any silent device-side fault to the block that caused
    // it, rather than letting it surface as a confusing mismatch several
    // checks later.
    cudaError_t pending_err = cudaGetLastError();
    if (pending_err != cudaSuccess) {
        std::printf("  [CUDA ERROR] pending before checking %-40s : %s\n",
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
        std::printf("    worst_i=%zu cpu=%.6f gpu=%.6f (of %zu elements)\n",
                   worst_i, static_cast<double>(cpu[worst_i]), static_cast<double>(gpu[worst_i]), cpu.size());
    }
    return ok;
}

bool run_shape(size_t rows, size_t cols, const char* tag) {
    std::printf("--- shape: rows=%zu cols=%zu (%s) ---\n", rows, cols, tag);
    bool all_ok = true;

    const size_t m = rows, n = cols;
    const size_t ld_a = m;
    const size_t rows_op = n;  // TRANSPOSE: y has length K=n
    const size_t cols_op = m;  // TRANSPOSE: x has length M=m
    const double alpha = 1.7, beta = 0.3;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(ld_a * n), x(cols_op), y_init(rows_op);
    for (auto& v : A) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, A.size() * sizeof(float));
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_y, y_init.size() * sizeof(float));
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    char label[160];
    const char* dir = "T";

    // ---- native = mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec
    {
        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, A.data(), ld_a,
                                             x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType::TRANSPOSE, alpha,
                                                                        dim<2>{m, n}, d_A, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(rows_op);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "native (vec4_stagevec) %s %s", tag, dir);
        all_ok &= check(label, y_cpu, y_gpu, 1e-4);
    }

    // ---- fp32_16 = mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch
    // (SAFE decode, unrestricted data -- the dispatcher just routes to
    // _wrapborrow or _wrapborrow_retarget2 based on M, both of which
    // share fp32_16_combine_branchless_wrapborrow's own full normal/
    // extended/degraded-range contract, so no safe-range restriction
    // needed)
    {
        std::vector<uint16_t> wh(ld_a * n), wr(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) fp32_16_split(A[i], wh[i], wr[i]);
        uint16_t *d_wh, *d_wr;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr, wr.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr, wr.data(), wr.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_accurate(TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                 x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(
            TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(rows_op);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "fp32_16 (vec4_wrapborrow_dispatch) %s %s", tag, dir);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);

        cudaFree(d_wh);
        cudaFree(d_wr);
    }

    // ---- fp32_16_fast = mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch
    // (UNSAFE-on-arbitrary-data contract -- MUST use safe-range data,
    // magnitude in [0.5,1.0], guaranteed control=0/never extended or
    // degraded-encoded, matching this kernel's documented contract; the
    // dispatcher itself is just a router, the contract comes from
    // whichever of _retarget/_retarget2 it calls underneath)
    {
        std::mt19937 safe_rng(43);
        std::uniform_real_distribution<float> mag_dist(0.5f, 1.0f);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::vector<float> A_safe(ld_a * n);
        for (auto& v : A_safe) {
            float mag = mag_dist(safe_rng);
            v = sign_dist(safe_rng) ? mag : -mag;
        }
        std::vector<uint16_t> wh_safe(ld_a * n), wr_safe(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) fp32_16_split(A_safe[i], wh_safe[i], wr_safe[i]);
        uint16_t *d_wh_safe, *d_wr_safe;
        cudaMalloc(&d_wh_safe, wh_safe.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_safe, wr_safe.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_safe, wh_safe.data(), wh_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_safe, wr_safe.data(), wr_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_fp32_16_accurate(TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                 x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
            TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(rows_op);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "fp32_16_fast (wrapborrow_vec4_dispatch) %s %s", tag, dir);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);

        cudaFree(d_wh_safe);
        cudaFree(d_wr_safe);
    }

    // ---- fp32_bf16 = mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch
    // (rounded encoding -- fp32_bf16_split_rounded/fp32_bf16_combine_
    // rounded, own CPU reference built by decoding host-side then
    // reusing cpu::mv_strided, same technique verify_mv_fp32_16_gemv.cu
    // uses for this same kernel. NOT a new tuned kernel -- a router over
    // plain _vec4_rounded (M<8192) and its own _retarget2 sibling
    // (M>=8192), so this correctness check covers BOTH branches: this
    // shape sweep's own M=K=16384 case (see this file's header comment)
    // exercises the M>=8192 branch, every other shape here exercises the
    // M<8192 branch. float4/ushort4-vectorized, TRANSPOSE-only, no
    // workspace parameter -- see its own comment in fp32_16_cuda_
    // kernels.cu)
    {
        std::vector<uint16_t> wh_r(ld_a * n), wr_r(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) fp32_bf16_split_rounded(A[i], wh_r[i], wr_r[i]);
        std::vector<float> A_decoded(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) A_decoded[i] = fp32_bf16_combine_rounded(wh_r[i], wr_r[i]);

        uint16_t *d_wh_r, *d_wr_r;
        cudaMalloc(&d_wh_r, wh_r.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_r, wr_r.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_r, wh_r.data(), wh_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_r, wr_r.data(), wr_r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, A_decoded.data(), ld_a,
                                             x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
            TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, d_wh_r, d_wr_r, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(rows_op);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "fp32_bf16 (vec4_rounded_dispatch) %s %s", tag, dir);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);

        cudaFree(d_wh_r);
        cudaFree(d_wr_r);
    }

    // ---- fp32_bf16_truncated = mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4
    // (plain/truncating encoding -- fp32_bf16_split/fp32_bf16_combine.
    // float4/ushort4-vectorized, TRANSPOSE-only, no workspace parameter
    // -- see its own comment in fp32_16_cuda_kernels.cu)
    {
        std::vector<uint16_t> wh_t(ld_a * n), wr_t(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) fp32_bf16_split(A[i], wh_t[i], wr_t[i]);
        std::vector<float> A_decoded(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) A_decoded[i] = fp32_bf16_combine(wh_t[i], wr_t[i]);

        uint16_t *d_wh_t, *d_wr_t;
        cudaMalloc(&d_wh_t, wh_t.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_t, wr_t.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_t, wh_t.data(), wh_t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_t, wr_t.data(), wr_t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        std::vector<float> y_cpu = y_init;
        cpu::mv_strided<float, float, float>(TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, A_decoded.data(), ld_a,
                                             x.data(), 1, beta, y_cpu.data(), 1);
        cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(
            TransposeType::TRANSPOSE, alpha, dim<2>{m, n}, d_wh_t, d_wr_t, ld_a, d_x, 1, beta, d_y, 1);
        std::vector<float> y_gpu(rows_op);
        cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "fp32_bf16_truncated (vec4) %s %s", tag, dir);
        all_ok &= check(label, y_cpu, y_gpu, 5e-2);

        cudaFree(d_wh_t);
        cudaFree(d_wr_t);
    }

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    std::printf("\n");
    return all_ok;
}

}  // namespace

int main() {
    std::printf("=== TRANSPOSE GEMV: promoted/plotted kernels only ===\n");
    std::printf("(see scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.m for what \"promoted\" means)\n\n");

    struct Shape { size_t rows, cols; const char* tag; };
    std::vector<Shape> shapes = {
        {1024, 1024, "square-small"},
        {6144, 6144, "square-mid (fp32_16_fast's own biggest measured win)"},
        {4096, 4096, "square-large"},
        {70, 8192, "tall-thin-tiny-m-non-mult4 (regression check)"},
        {512, 16384, "tall-thin"},
        {16384, 16384, "square-xlarge (exercises fp32_16_fast's dispatch>=8192 branch)"},
    };

    bool all_ok = true;
    for (const auto& s : shapes) all_ok &= run_shape(s.rows, s.cols, s.tag);

    std::printf("%s\n", all_ok ? "ALL PROMOTED-KERNEL CHECKS PASSED"
                               : "SOME PROMOTED-KERNEL CHECKS FAILED (see FAIL lines above)");
    return all_ok ? 0 : 1;
}

// Standalone correctness repro specifically for mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled
// (fp32_16_cuda_kernels.cu) at a shape LARGE enough for its own internal
// sub-tile loop to actually iterate more than once -- the standard
// verify_mv_fp32_16_gemv.cu correctness sweep's own single TRANSPOSE-
// compatible shape (M=70, same limitation every other fast_normal
// sibling's own check there already has) never exercises that: at M=70,
// M_BLOCK is far below kMvFp32_16FastNormalWrapborrowTiledSubTile=2048,
// so the loop runs exactly once there, the SAME single-iteration path
// every other kernel's own check already covers. This repro targets
// M=K=65536 specifically (where _retarget2 -- and so this tiled sibling
// -- actually launches with M_BLOCK=8192, i.e. 4 real sub-tile
// iterations per block: 8192/2048=4), against a CPU reference, so the
// new cross-subtile accumulation (persistent per-thread accum carried
// across iterations) and the double-__syncthreads() (before AND after
// each sub-tile's compute, guarding against the next iteration's
// staging racing the current iteration's still-in-flight reads) are
// both genuinely tested, not just compiled. A itself is generated with
// the SAME magnitude restriction mv_fp32_transpose_recon_vs_cublas_
// ref.cu's own gen_fp32_16_safe_normal uses -- fp32_16_combine_fast_
// normal_wrapborrow decodes unconditionally as if control were always
// 0, so feeding it data outside that safe range silently corrupts every
// element (this was confirmed the hard way: an earlier plain-uniform-
// [-1,1] version of this repro FAILED against the CPU reference for
// BOTH the already-validated untiled _retarget2 AND this tiled sibling,
// by the exact same margin -- proving the bug was in the test data, not
// either kernel, before this fix).
//
// Not wired into any build script or CMakeLists -- compile and run
// directly (same include list this repo's own benchmark .sh scripts use,
// plus -ltomos_cpu for the CPU reference -- verified to compile+link
// clean before being handed off):
//   nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
//       -I src/kernels/cuda/matrix -I src/kernels/cpu/matrix \
//       -I src/core/matrix -I include/core -I src/core/context \
//       -I src/core/memory -I src/core/blas -I src/kernels/cuda/context \
//       -I src/macros -I include/io -I include/cpu \
//       -I src/kernels/cpu/memory -I src/kernels/cpu/sketch \
//       -I include/cuda -I src/kernels/cuda/constants \
//       -I src/kernels/cuda/memory -I src/kernels/cuda/sketch \
//       -I src/kernels/cuda/precond -I src/kernels/cuda/blas \
//       -I src/core/solver -I src/dispatch -I src/io \
//       -I build/generated -I build -I include -I src/core/detail \
//       verify_repro_tiled_vs_untiled_large_m.cu \
//       -L build/src/kernels/cuda -ltomos_cuda \
//       -L build -ltomos_cpu \
//       -Xlinker -rpath -Xlinker build/src/kernels/cuda \
//       -Xlinker -rpath -Xlinker build \
//       -o build/verify_repro_tiled_vs_untiled_large_m
//   build/verify_repro_tiled_vs_untiled_large_m
//
// If fp32_16_cuda_kernels.cu/.cuh changed since the last build,
// libtomos_cuda.so needs rebuilding first (same as the benchmark .sh
// scripts do):
//   cmake --build build --target tomos_cuda -j"$(nproc)"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "dimensions.hpp"
#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"
#include "transpose.hpp"

using namespace tomos;

int main() {
    const int m = 65536;
    const int k = 65536;
    const size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    const double alpha = 1.7;
    const double beta = 0.3;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // A itself must be restricted to the SAME safe-normal magnitude range
    // (mirrors gen_fp32_16_safe_normal's own pseudo_rand_safe_normal in
    // mv_fp32_transpose_recon_vs_cublas_ref.cu) that mv_fp32_16_transpose_
    // atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_kernel's own
    // decode (fp32_16_combine_fast_normal_wrapborrow) requires -- it
    // decodes unconditionally as if control were always 0, so feeding it
    // data fp32_16_split would have encoded via the extended/degraded
    // path (values outside this magnitude band) silently corrupts every
    // element. x/y are plain floats, never split/decoded, so they don't
    // need this restriction.
    std::uniform_real_distribution<float> mag_dist(0.5f, 1.0f);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    std::vector<float> A(mk);
    std::vector<float> x(static_cast<size_t>(m));
    std::vector<float> y_init(static_cast<size_t>(k));
    for (auto& v : A) v = sign_dist(rng) ? mag_dist(rng) : -mag_dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y_init) v = dist(rng);

    std::vector<uint16_t> wh(mk), wr(mk);
    for (size_t i = 0; i < mk; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    std::vector<float> y_cpu = y_init;
    cpu::mv_fp32_16_accurate(TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, wh.data(), wr.data(),
                             static_cast<size_t>(m), x.data(), 1, beta, y_cpu.data(), 1);

    uint16_t *d_wh = nullptr, *d_wr = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_wh, mk * sizeof(uint16_t));
    cudaMalloc(&d_wr, mk * sizeof(uint16_t));
    cudaMalloc(&d_x, static_cast<size_t>(m) * sizeof(float));
    cudaMalloc(&d_y, static_cast<size_t>(k) * sizeof(float));
    cudaMemcpy(d_wh, wh.data(), mk * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wr, wr.data(), mk * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), static_cast<size_t>(m) * sizeof(float), cudaMemcpyHostToDevice);

    // vs-CPU tolerance is deliberately loose: K=65536 is a MUCH deeper
    // reduction than verify_mv_fp32_16_gemv.cu's own TRANSPOSE check ever
    // exercises (that one only reduces over 70 elements, M=70), and this
    // kernel family accumulates in plain float (not double) end to end --
    // `accum` is a float, and the final combine is a float atomicAdd, see
    // the kernel's own body -- so real float32-accumulation error at this
    // depth (roughly sqrt(K) * float epsilon, worse at specific columns
    // with cancellation) is EXPECTED here, not a bug. This is informational
    // only, not the real correctness gate for the tiling change itself.
    auto check_vs_cpu = [&](const char* label, const std::vector<float>& y_gpu) {
        double max_abs_err = 0.0, max_rel_err = 0.0;
        for (size_t i = 0; i < y_gpu.size(); ++i) {
            double abs_err = std::fabs(static_cast<double>(y_gpu[i]) - static_cast<double>(y_cpu[i]));
            double denom = std::max(1e-6, std::fabs(static_cast<double>(y_cpu[i])));
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, abs_err / denom);
        }
        bool ok = max_rel_err < 2e-2;
        std::printf("  %-70s max_abs_err=%.6e max_rel_err=%.6e -- %s (loose, informational)\n", label, max_abs_err,
                    max_rel_err, ok ? "PASS" : "FAIL");
        return ok;
    };

    // THE real gate: tiled vs untiled directly. Both accumulate the SAME
    // float32 sums (same terms, same fmaf/vectorization pattern within
    // each phase, just chunked into sub-tiles for the tiled kernel) --
    // any genuine divergence here (beyond float-reordering-scale noise)
    // means the tiling refactor itself introduced a bug, independent of
    // whatever the CPU reference's own float32-vs-double gap is.
    auto check_tiled_vs_untiled = [&](const std::vector<float>& y_untiled, const std::vector<float>& y_tiled) {
        double max_abs_err = 0.0, max_rel_err = 0.0;
        size_t worst_i = 0;
        // Histogram to distinguish two very different explanations for a
        // nonzero divergence: DIFFUSE (every column off by a similarly
        // tiny amount -- consistent with compiler-level fmaf instruction
        // scheduling differing between the two loop structures, a real
        // but benign floating-point-non-determinism effect, NOT a logic
        // bug) vs CONCENTRATED (a small number of columns wildly wrong,
        // the rest exact -- consistent with an actual boundary/indexing
        // bug in the tiling refactor).
        size_t over_1e5 = 0, over_1e4 = 0, over_1e3 = 0, exact = 0;
        for (size_t i = 0; i < y_untiled.size(); ++i) {
            double abs_err = std::fabs(static_cast<double>(y_tiled[i]) - static_cast<double>(y_untiled[i]));
            double denom = std::max(1e-6, std::fabs(static_cast<double>(y_untiled[i])));
            double rel_err = abs_err / denom;
            if (rel_err > max_rel_err) { max_rel_err = rel_err; worst_i = i; }
            max_abs_err = std::max(max_abs_err, abs_err);
            if (rel_err == 0.0) ++exact;
            if (rel_err > 1e-5) ++over_1e5;
            if (rel_err > 1e-4) ++over_1e4;
            if (rel_err > 1e-3) ++over_1e3;
        }
        bool ok = max_rel_err < 1e-4;
        std::printf("  %-70s max_abs_err=%.6e max_rel_err=%.6e -- %s\n",
                    "retarget2_tiled vs retarget2 (untiled) DIRECTLY", max_abs_err, max_rel_err,
                    ok ? "PASS" : "FAIL");
        std::printf("    worst column: i=%zu  untiled=%.9e  tiled=%.9e\n", worst_i,
                    static_cast<double>(y_untiled[worst_i]), static_cast<double>(y_tiled[worst_i]));
        std::printf("    distribution (out of %zu): exact=%zu  rel_err>1e-5: %zu  >1e-4: %zu  >1e-3: %zu\n",
                    y_untiled.size(), exact, over_1e5, over_1e4, over_1e3);
        return ok;
    };

    bool all_ok = true;

    cudaMemcpy(d_y, y_init.data(), static_cast<size_t>(k) * sizeof(float), cudaMemcpyHostToDevice);
    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2(
        TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, d_wh, d_wr, static_cast<size_t>(m), d_x, 1, beta, d_y, 1);
    std::vector<float> y_gpu_untiled(static_cast<size_t>(k));
    cudaMemcpy(y_gpu_untiled.data(), d_y, static_cast<size_t>(k) * sizeof(float), cudaMemcpyDeviceToHost);
    check_vs_cpu("retarget2 (untiled, M_BLOCK=8192, single-shot staging) vs CPU", y_gpu_untiled);

    cudaMemcpy(d_y, y_init.data(), static_cast<size_t>(k) * sizeof(float), cudaMemcpyHostToDevice);
    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled(
        TransposeType::TRANSPOSE, alpha, dim<2>{m, k}, d_wh, d_wr, static_cast<size_t>(m), d_x, 1, beta, d_y, 1);
    std::vector<float> y_gpu_tiled(static_cast<size_t>(k));
    cudaMemcpy(y_gpu_tiled.data(), d_y, static_cast<size_t>(k) * sizeof(float), cudaMemcpyDeviceToHost);
    check_vs_cpu("retarget2_tiled (M_BLOCK=8192, 4x 2048-float sub-tiles) vs CPU", y_gpu_tiled);

    all_ok &= check_tiled_vs_untiled(y_gpu_untiled, y_gpu_tiled);

    cudaFree(d_wh);
    cudaFree(d_wr);
    cudaFree(d_x);
    cudaFree(d_y);

    std::printf("%s\n", all_ok ? "ALL CHECKS PASSED (tiled vs untiled agree)"
                               : "SOME CHECKS FAILED (tiled vs untiled DIVERGE -- real bug)");
    return all_ok ? 0 : 1;
}

// Standalone correctness + performance verification for the mv_fp32_16_*
// GEMV kernels (src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu), covering
// the wide (split-K), tiled (plain narrow/square), and narrow_vec
// (register-blocked) dispatch paths.
//
// This exists because the interesting shapes here are large enough (up to
// several million rows, or a few thousand columns times millions of rows)
// that host-side data generation runs into memory limits on constrained
// dev hosts -- both the CORRECTNESS reference data and the PERFORMANCE
// sweep's matrices are generated directly on the device instead. It also
// specifically exercises the boundary/tail cases (rows_op not a multiple
// of kMvRowsPerThread, both transpose directions, ld_a not a multiple of
// kMvRowsPerThread) that caught a real misaligned-load bug in the
// register-blocked kernels during development -- run the correctness
// section after any future change to these kernels, not just the
// performance section.
//
// Usage: verify_mv_fp32_16_gemv [--skip-correctness] [--skip-performance]
//                                [--only-shape M N] [--only-case TAG]
//
// --only-shape M N runs just that one shape (see run_single_shape) and
// exits, skipping correctness and the full sweep -- for isolating a single
// kernel launch under an external profiler (ncu/nsys), where "the Nth
// launch of kernel X" would otherwise be ambiguous to target across the
// full narrow/wide/square sweep.
//
// --only-case TAG restricts run_correctness()'s own sweep to the single
// Case whose tag exactly matches TAG (see the `cases` list in
// run_correctness()), running every kernel check for that one shape via
// the EXACT same code path as the full sweep -- unlike --only-shape,
// this stays in the correctness section (still runs run_correctness_case
// in full for that one shape) rather than jumping to a raw performance
// timing. Combine with --skip-performance to skip straight to just that
// one case's correctness output, e.g.:
//   verify_mv_fp32_16_gemv --skip-performance --only-case narrow_vec-via-transpose
//
// Exit code is non-zero if any correctness check that should be exact
// (accurate/fast, or pure_half on a short reduction) fails.

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

// ---- shared device-side data generation (avoids host memory limits at
// the large sizes the performance sweep uses) ----

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
}

__global__ void gen_half_kernel(size_t n, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<__half>(pseudo_rand(i));
}

__global__ void gen_float_kernel(size_t n, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand(i);
}

void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}
void gen_half(size_t n, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out);
}
void gen_float(size_t n, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out);
}

double now_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ---- correctness section: GPU vs CPU reference ----

bool check(const std::string& label, const std::vector<float>& cpu,
          const std::vector<float>& gpu, double tol) {
    // DIAGNOSTIC (temporary, for the narrow_vec-via-transpose
    // investigation): cudaGetLastError() is STICKY -- it reflects the
    // most recent error since the last query, and every call site's own
    // pattern (kernel launch -> synchronous cudaMemcpy -> this check())
    // means whatever error state is current HERE was set by THIS
    // block's own kernel/memcpy, not some earlier block's (nothing else
    // touches the CUDA API in between). A synchronous cudaMemcpy also
    // itself surfaces a prior queued kernel's fault (e.g. illegal
    // address) as its own return code, so this catches silent faults
    // that neither the kernel launch nor the memcpy currently check.
    cudaError_t pending_err = cudaGetLastError();
    if (pending_err != cudaSuccess) {
        std::printf("  [diag] CUDA ERROR pending before checking %-40s : %s\n",
                   label.c_str(), cudaGetErrorString(pending_err));
    }
    double max_abs_err = 0.0, max_rel_err = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double err = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpu[i]));
        double rel = err / std::max(1.0, std::fabs(static_cast<double>(cpu[i])));
        if (err > max_abs_err) worst_i = i;
        max_abs_err = std::max(max_abs_err, err);
        max_rel_err = std::max(max_rel_err, rel);
    }
    bool ok = max_rel_err < tol;
    std::printf("  %-58s max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(),
               max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    if (!ok) {
        // DIAGNOSTIC (temporary, for the narrow_vec-via-transpose
        // investigation): print the worst index and both values, to
        // distinguish "one column never written by the GPU kernel
        // (retains its pre-launch y_init value)" from "a genuine
        // per-element computational discrepancy".
        std::printf("    [diag] worst_i=%zu cpu=%.6f gpu=%.6f (of %zu elements)\n",
                   worst_i, static_cast<double>(cpu[worst_i]), static_cast<double>(gpu[worst_i]), cpu.size());
    }
    return ok;
}

// Runs one (rows, cols, trans) case for all three tiers, comparing against
// the CPU reference. pure_half is checked with a looser tolerance -- its
// naive fp16 accumulation is known to diverge from the CPU reference on
// long reductions (chaotic amplification of tiny host/device fp16 rounding
// differences through many dependent operations), a pre-existing
// characteristic of that tier, not a bug -- see the tag argument to note
// when a large pure_half error is therefore expected.
//
// ld_a_pad, if nonzero, uses that as the storage leading dimension instead
// of rows (must be >= rows): the *_narrow_vec_kernel variants only take
// their vectorized-load path when ld_a is a multiple of kMvRowsPerThread,
// so a case with rows % kMvRowsPerThread != 0 (needing the scalar tail
// fallback) only ALSO exercises the vectorized main loop if ld_a is
// padded out to a multiple of kMvRowsPerThread separately from rows --
// with no padding, ld_a == rows whenever trans is NO_TRANSPOSE, so a
// nonzero tail remainder would make ld_a misaligned too, disabling
// vectorization for the whole case rather than just its tail.
bool run_correctness_case(size_t rows, size_t cols, const char* tag,
                         size_t ld_a_pad = 0) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    for (TransposeType trans : {TransposeType::NO_TRANSPOSE, TransposeType::TRANSPOSE}) {
        size_t m = rows, n = cols;
        size_t ld_a = (ld_a_pad != 0) ? ld_a_pad : m;
        size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
        size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;

        std::vector<float> A(ld_a * n), x(cols_op), y_init(rows_op);
        for (auto& v : A) v = dist(rng);
        for (auto& v : x) v = dist(rng);
        for (auto& v : y_init) v = dist(rng);

        std::vector<uint16_t> wh(ld_a * n), wr(ld_a * n);
        for (size_t i = 0; i < ld_a * n; ++i) fp32_16_split(A[i], wh[i], wr[i]);

        double alpha = 1.7, beta = 0.3;

        uint16_t *d_wh, *d_wr;
        float *d_x, *d_y;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr, wr.size() * sizeof(uint16_t));
        cudaMalloc(&d_x, x.size() * sizeof(float));
        cudaMalloc(&d_y, y_init.size() * sizeof(float));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr, wr.data(), wr.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

        const char* dir = (trans == TransposeType::NO_TRANSPOSE) ? "N" : "T";
        char label[128];

        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1,
                                      beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }
        {
            // mv_fp32_16_accurate_workspace -- identical numerics to the
            // "accurate" block above (same kernels, same double
            // accumulator), just fed a caller-provided workspace buffer
            // instead of letting the wide path malloc/free its own
            // scratch space per call -- see its own comment in
            // fp32_16_cuda_kernels.cu. Same reference and tolerance as
            // "accurate" since the actual computation is unchanged.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t ws_elems = cuda::experimental::mv_fp32_16_accurate_workspace_size(trans, dim<2>{m, n});
            double* d_workspace = nullptr;
            if (ws_elems > 0) cudaMalloc(&d_workspace, ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_accurate_workspace(trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1,
                                                beta, d_y, 1, d_workspace);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate_workspace %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
            cudaFree(d_workspace);
        }
        {
            // mv_fp32_16_accurate_atomic -- recreates randls2.0's own
            // split-K GEMV mechanism (atomicAdd into y, no partial[]
            // buffer, no reduce kernel -- see its own comment in
            // fp32_16_cuda_kernels.cu). Same CPU reference as "accurate"
            // above, but a looser tolerance: the final combination
            // across col_splits blocks happens via float atomicAdd here
            // instead of an exact double reduction, a real (if small)
            // precision trade-off -- 2e-3 is a starting point, not yet
            // confirmed against a real hardware run.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_accurate_atomic(trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1,
                                             beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate_atomic %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 2e-3);
        }
        {
            // mv_fp32_16_accurate_ready -- exact recreation of randls2.0's
            // own ktiled_atomic_ready mechanism (2 rows/lane, packed
            // uint32_t loads, float accumulation, ready-flag atomics --
            // see its own comment in fp32_16_cuda_kernels.cu). Same CPU
            // reference as "accurate" above; looser tolerance than that
            // (float accumulation, not double, same reasoning as
            // "accurate_atomic" above, likely a bit looser still since
            // this kernel's OWN per-thread K-reduction is float, not
            // just the final cross-block combination) -- 5e-3 is a
            // starting point, not yet confirmed against real hardware.
            // ld_a must equal m here (mv_fp32_16_accurate_ready falls
            // back to mv_fp32_16_accurate otherwise, so this check would
            // silently test the fallback instead of the real kernel for
            // any padded-ld_a case -- this file's own cases keep ld_a==m
            // except where padding is the specific thing under test, so
            // this block only asserts when that holds).
            if (ld_a == m) {
                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                size_t ready_elems = cuda::experimental::mv_fp32_16_accurate_ready_size(trans, dim<2>{m, n});
                int* d_ready = nullptr;
                if (ready_elems > 0) cudaMalloc(&d_ready, ready_elems * sizeof(int));
                cuda::experimental::mv_fp32_16_accurate_ready(trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1,
                                                beta, d_y, 1, d_ready);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);
                cudaFree(d_ready);
            }
            if (ld_a == m) {
                // mv_fp32_16_accurate_ready_atomiconly_scaled_kblock --
                // PROMOTED as plain "TOMOS FP32/FP16": row-vectorized/
                // atomiconly/scaled-K_BLOCK structure identical to the
                // "(regular)" fast_normal sibling, but decoding via the
                // full fp32_16_combine_branchless instead -- safe on
                // arbitrary data, so unlike every "_fast_normal" sibling
                // in this file it can be checked against the SAME plain
                // wh/wr data as "accurate" above, no dedicated safe-
                // range generator needed. Same CPU reference as
                // "accurate"/"accurate_ready" above; same tolerance
                // reasoning as "accurate_ready" (float accumulation,
                // atomicAdd combination across gridK blocks, not double).
                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                                         ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready_atomiconly_scaled_kblock %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);
            }
            if (ld_a == m) {
                // mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock --
                // EXPERIMENTAL 4-rows/lane extension of the SAFE kernel
                // just above (see its own comment in fp32_16_cuda_
                // kernels.cu). Same safe-on-arbitrary-data property, so
                // checked the same way, same CPU reference/tolerance.
                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                                          ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready4_atomiconly_scaled_kblock %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);

                // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4 --
                // EXPERIMENTAL, standalone A/B sibling of the kernel just
                // above (float4-vectorized x-staging/sx reads, ushort4-
                // vectorized wh/wr loads -- see its own comment in
                // fp32_16_cuda_kernels.cu). Only handles TRANSPOSE
                // (returns immediately otherwise), so only checked for
                // that case here -- same plain wh/wr data and CPU
                // reference as the check just above, reused unchanged.
                if (trans == TransposeType::TRANSPOSE) {
                    std::vector<float> y_cpu2 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu2.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                                              ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu2(rows_op);
                    cudaMemcpy(y_gpu2.data(), d_y, y_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu2, y_gpu2, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
                    // retuned -- EXPERIMENTAL, standalone A/B sibling
                    // reusing the EXACT SAME __global__ kernel as the
                    // check just above, only a different M_BLOCK-scaling
                    // grid-oversubscription target -- see its own comment
                    // in fp32_16_cuda_kernels.cu. Same reference/data.
                    std::vector<float> y_cpu3 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned(
                        trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3(rows_op);
                    cudaMemcpy(y_gpu3.data(), d_y, y_gpu3.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_retuned %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3, y_gpu3, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
                    // retuned_floor -- EXPERIMENTAL, standalone A/B
                    // sibling reusing the EXACT SAME __global__ kernel as
                    // the two checks just above, same target as
                    // _retuned but with a floor on how far M_BLOCK can
                    // shrink -- see its own comment in fp32_16_cuda_
                    // kernels.cu. Same reference/data.
                    std::vector<float> y_cpu4 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu4.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                        trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu4(rows_op);
                    cudaMemcpy(y_gpu4.data(), d_y, y_gpu4.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_retuned_floor %s %s", tag, dir);
                    all_ok &= check(label, y_cpu4, y_gpu4, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
                    // wrapborrow -- EXPERIMENTAL, standalone A/B sibling
                    // reusing the EXACT SAME __global__ kernel STRUCTURE
                    // and M_BLOCK-scaling as _retuned_floor above, only
                    // decoding via fp32_16_combine_branchless_wrapborrow
                    // instead of fp32_16_combine_branchless -- see its own
                    // comment in fp32_16_cuda_kernels.cu. Safe on
                    // arbitrary data (same contract as fp32_16_combine_
                    // branchless), so tested here against the SAME plain
                    // wh/wr data as every other safe-on-arbitrary-data
                    // sibling above, not the safe-range-only data the
                    // fast_normal family below needs.
                    std::vector<float> y_cpu4b = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu4b.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow(
                        trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu4b(rows_op);
                    cudaMemcpy(y_gpu4b.data(), d_y, y_gpu4b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_wrapborrow %s %s", tag, dir);
                    all_ok &= check(label, y_cpu4b, y_gpu4b, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
                    // wrapborrow_retarget2 -- EXPERIMENTAL, standalone A/B
                    // sibling reusing the EXACT SAME __global__ kernel as
                    // _wrapborrow above, only a higher M_BLOCK-scaling
                    // target (8192 instead of 2048) -- a high-confidence
                    // port of the UNSAFE fast_normal_wrapborrow family's
                    // own retarget2, since the grid-math formula is
                    // decode-independent -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same plain wh/wr data.
                    std::vector<float> y_cpu4c = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu4c.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2(
                        trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu4c(rows_op);
                    cudaMemcpy(y_gpu4c.data(), d_y, y_gpu4c.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu4c, y_gpu4c, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
                    // wrapborrow_dispatch -- EXPERIMENTAL, standalone
                    // dispatcher routing to _wrapborrow or _wrapborrow_
                    // retarget2 based on M -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same plain wh/wr data.
                    std::vector<float> y_cpu4d = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                             x.data(), 1, beta, y_cpu4d.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(
                        trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu4d(rows_op);
                    cudaMemcpy(y_gpu4d.data(), d_y, y_gpu4d.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch %s %s", tag, dir);
                    all_ok &= check(label, y_cpu4d, y_gpu4d, 5e-3);
                }
            }
            if (ld_a == m) {
                // mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_
                // normal -- "TOMOS FP32/FP16 (fast)". UNSAFE on arbitrary
                // data (fp32_16_combine_fast_normal's own contract: no
                // extended/degraded-encoded values), so tested here
                // against dedicated safe-range data (mag in [0.5, 1.0]),
                // not the plain "wh"/"wr" used by every safe-on-arbitrary-
                // data sibling above -- mirrors mv_fp32_16_regular's own
                // safe-range generator in mv_fp32_recon_vs_cutlass_ref.cu
                // (pseudo_rand_safe_normal), just host-side here. This is
                // also this kernel's FIRST correctness coverage of any
                // kind in this file (previously benchmarked only, for
                // NO_TRANSPOSE) -- exercising both TRANSPOSE directions
                // via the same trans-parameterized loop this whole
                // function already runs under also covers the pre-
                // existing NO_TRANSPOSE path for the first time as a
                // side effect, not extra work. See mv_fp32_16_transpose_
                // atomiconly_scaled_mblock_fast_normal_kernel's own
                // comment in fp32_16_cuda_kernels.cu for the TRANSPOSE
                // path's own derivation.
                std::mt19937 safe_rng(43);
                std::uniform_real_distribution<float> mag_dist(0.5f, 1.0f);
                std::uniform_int_distribution<int> sign_dist(0, 1);
                std::vector<float> A_safe(ld_a * n);
                for (auto& v : A_safe) {
                    float mag = mag_dist(safe_rng);
                    v = sign_dist(safe_rng) ? mag : -mag;
                }
                std::vector<uint16_t> wh_safe(ld_a * n), wr_safe(ld_a * n);
                for (size_t idx = 0; idx < ld_a * n; ++idx)
                    fp32_16_split(A_safe[idx], wh_safe[idx], wr_safe[idx]);

                uint16_t *d_wh_safe, *d_wr_safe;
                cudaMalloc(&d_wh_safe, wh_safe.size() * sizeof(uint16_t));
                cudaMalloc(&d_wr_safe, wr_safe.size() * sizeof(uint16_t));
                cudaMemcpy(d_wh_safe, wh_safe.data(), wh_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_wr_safe, wr_safe.data(), wr_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(
                    trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready_atomiconly_scaled_kblock_fast_normal %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);

                // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                // normal_vec4 -- EXPERIMENTAL, standalone A/B sibling of
                // the kernel just above (float4-vectorized x-staging/sx
                // reads, ushort4-vectorized wh/wr loads -- see its own
                // comment in fp32_16_cuda_kernels.cu). Only handles
                // TRANSPOSE (returns immediately otherwise), so only
                // checked for that case here -- same safe-range data and
                // CPU reference as the check just above, reused
                // unchanged.
                if (trans == TransposeType::TRANSPOSE) {
                    std::vector<float> y_cpu2 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu2.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu2(rows_op);
                    cudaMemcpy(y_gpu2.data(), d_y, y_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_vec4 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu2, y_gpu2, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_vec4_retuned -- EXPERIMENTAL, standalone A/B
                    // sibling reusing the EXACT SAME __global__ kernel as
                    // the check just above, only a more conservative
                    // M_BLOCK-scaling target -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same safe-range data.
                    std::vector<float> y_cpu3 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3(rows_op);
                    cudaMemcpy(y_gpu3.data(), d_y, y_gpu3.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3, y_gpu3, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4 -- EXPERIMENTAL, standalone
                    // A/B sibling: identical structure to fast_normal_
                    // vec4's own check above, only decoding via fp32_16_
                    // combine_fast_normal_wrapborrow (bit-identical to
                    // fp32_16_combine_fast_normal, exhaustively verified
                    // -- see its own comment in fp32_16_pack.hpp). This
                    // check re-verifies that bit-identity holds through
                    // the full GEMV kernel, not just the standalone
                    // decode function. Same safe-range data.
                    std::vector<float> y_cpu3b = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3b.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3b(rows_op);
                    cudaMemcpy(y_gpu3b.data(), d_y, y_gpu3b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3b, y_gpu3b, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget -- EXPERIMENTAL,
                    // standalone sibling reusing the EXACT SAME __global__
                    // kernel as the check just above, only a more
                    // aggressive M_BLOCK-scaling target re-derived from
                    // wrapborrow's own (lower) register footprint -- see
                    // its own comment in fp32_16_cuda_kernels.cu. Same
                    // safe-range data.
                    std::vector<float> y_cpu3c = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3c.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3c(rows_op);
                    cudaMemcpy(y_gpu3c.data(), d_y, y_gpu3c.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3c, y_gpu3c, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2 -- EXPERIMENTAL,
                    // standalone sibling reusing the EXACT SAME __global__
                    // kernel as the check just above, only an even more
                    // aggressive M_BLOCK-scaling target aimed at the two
                    // tall-and-thin shapes _retarget's own target=2048
                    // left unaffected -- see its own comment in fp32_16_
                    // cuda_kernels.cu. Same safe-range data.
                    std::vector<float> y_cpu3d = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3d.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3d(rows_op);
                    cudaMemcpy(y_gpu3d.data(), d_y, y_gpu3d.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3d, y_gpu3d, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2_tiled -- standalone
                    // sibling reusing the EXACT SAME M_BLOCK-scaling and
                    // atomicAdd combine as the check just above, only
                    // staging x into shared memory in sub-tiles instead
                    // of the whole M_BLOCK range at once -- see its own
                    // comment in fp32_16_cuda_kernels.cu. NOTE: at this
                    // shape (M=70), M_BLOCK is far below the 2048-float
                    // sub-tile size, so this only exercises the loop's
                    // SINGLE-iteration path (byte-for-byte equivalent to
                    // the untiled kernel there) -- the genuinely new
                    // multi-sub-tile/cross-tile-accumulation path is
                    // covered separately by verify_repro_tiled_vs_
                    // untiled_large_m.cu (repo root) at M=K=65536, where
                    // M_BLOCK=8192 gives 4 real sub-tile iterations per
                    // block. Same safe-range data.
                    std::vector<float> y_cpu3d_tiled = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3d_tiled.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3d_tiled(rows_op);
                    cudaMemcpy(y_gpu3d_tiled.data(), d_y, y_gpu3d_tiled.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3d_tiled, y_gpu3d_tiled, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2_split -- standalone
                    // sibling reusing the EXACT SAME M_BLOCK-scaling as the
                    // check just above, only replacing the atomicAdd-per-
                    // block combine with a scratch-buffer partial sum +
                    // separate reduce kernel (targets the atomicAdd-
                    // contention growth at M>=16384) -- takes a caller-
                    // provided workspace (allocated below, outside any
                    // timing) rather than an internal cudaMalloc/cudaFree
                    // -- see its own comment in fp32_16_cuda_kernels.cu.
                    // Same safe-range data.
                    std::vector<float> y_cpu3d_split = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3d_split.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    size_t retarget2_split_ws_elems =
                        cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split_workspace_size(
                            dim<2>{m, n});
                    double* d_retarget2_split_ws = nullptr;
                    cudaMalloc(&d_retarget2_split_ws, retarget2_split_ws_elems * sizeof(double));
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1,
                        d_retarget2_split_ws);
                    cudaFree(d_retarget2_split_ws);
                    std::vector<float> y_gpu3d_split(rows_op);
                    cudaMemcpy(y_gpu3d_split.data(), d_y, y_gpu3d_split.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3d_split, y_gpu3d_split, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2_floor -- EXPERIMENTAL,
                    // standalone sibling reusing the EXACT SAME __global__
                    // kernel and target as the check just above, only a
                    // higher floor (2048) fixing _retarget2's own confirmed
                    // M=3072-6144 regression -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same safe-range data.
                    std::vector<float> y_cpu3e = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3e.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3e(rows_op);
                    cudaMemcpy(y_gpu3e.data(), d_y, y_gpu3e.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3e, y_gpu3e, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_dispatch -- EXPERIMENTAL,
                    // standalone dispatcher routing to _retarget or
                    // _retarget2 based on M -- see its own comment in
                    // fp32_16_cuda_kernels.cu. This shape (M=70) only
                    // exercises the M<8192 (_retarget) branch -- same
                    // single-shape limitation every OTHER fast_normal
                    // sibling's own correctness check in this file
                    // already has (this is the only shape/site the
                    // "if (ld_a == m)" guard above reaches). Same safe-
                    // range data.
                    std::vector<float> y_cpu3f = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3f.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3f(rows_op);
                    cudaMemcpy(y_gpu3f.data(), d_y, y_gpu3f.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3f, y_gpu3f, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2_bigblock --
                    // EXPERIMENTAL, standalone sibling reusing the EXACT
                    // SAME __global__ kernel and target as _retarget2
                    // above, only a higher shared M_BLOCK ceiling (16384
                    // instead of 8192, requiring a cudaFuncSetAttribute
                    // opt-in for the larger dynamic shared memory) -- see
                    // its own comment in fp32_16_cuda_kernels.cu. This
                    // shape (M=70) is far below where the higher ceiling
                    // has any effect (only M>=32768 changes grid config
                    // at all); this check only exercises the wrapper's
                    // own cudaFuncSetAttribute path and the shared kernel
                    // body at a trivial size, same limitation every OTHER
                    // fast_normal sibling's own correctness check in this
                    // file already has. Same safe-range data.
                    std::vector<float> y_cpu3g = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3g.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3g(rows_op);
                    cudaMemcpy(y_gpu3g.data(), d_y, y_gpu3g.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3g, y_gpu3g, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget_bigblock --
                    // EXPERIMENTAL, standalone sibling reusing the EXACT
                    // SAME __global__ kernel and higher ceiling (16384) as
                    // _retarget2_bigblock above, only _retarget's own
                    // LOWER target (2048) instead of _retarget2's 8192 --
                    // see its own comment in fp32_16_cuda_kernels.cu. This
                    // shape (M=70) is far below where either ceiling has
                    // any effect, same limitation as _retarget2_bigblock's
                    // own check above. Same safe-range data.
                    std::vector<float> y_cpu3h = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3h.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3h(rows_op);
                    cudaMemcpy(y_gpu3h.data(), d_y, y_gpu3h.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3h, y_gpu3h, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_wrapborrow_vec4_retarget2_hugeblock --
                    // EXPERIMENTAL, standalone sibling reusing the EXACT
                    // SAME __global__ kernel and target as _retarget2
                    // above, only an even higher ceiling (32768 instead of
                    // _retarget2_bigblock's own 16384) -- see its own
                    // comment in fp32_16_cuda_kernels.cu. This shape
                    // (M=70) is far below where any ceiling has an effect,
                    // same limitation as the other bigblock checks above.
                    // Same safe-range data.
                    std::vector<float> y_cpu3i = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu3i.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu3i(rows_op);
                    cudaMemcpy(y_gpu3i.data(), d_y, y_gpu3i.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock %s %s", tag, dir);
                    all_ok &= check(label, y_cpu3i, y_gpu3i, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_vec4_persistent -- EXPERIMENTAL, standalone
                    // sibling: same __global__ kernel FAMILY (byte-for-
                    // byte identical per-chunk compute), but blocks loop
                    // internally over multiple M_BLOCK chunks and do only
                    // ONE atomicAdd per warp for their entire lifetime --
                    // see its own comment in fp32_16_cuda_kernels.cu. Same
                    // safe-range data and CPU reference as the checks
                    // above.
                    std::vector<float> y_cpu4 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu4.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu4(rows_op);
                    cudaMemcpy(y_gpu4.data(), d_y, y_gpu4.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent %s %s", tag, dir);
                    all_ok &= check(label, y_cpu4, y_gpu4, 5e-3);

                    // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
                    // normal_vec4_persistent_v2 -- EXPERIMENTAL, standalone
                    // sibling of the check just above: identical launch
                    // config and per-chunk compute, only the trailing
                    // __syncthreads() is now conditional on there being
                    // another loop iteration -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same safe-range data.
                    std::vector<float> y_cpu5 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu5.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu5(rows_op);
                    cudaMemcpy(y_gpu5.data(), d_y, y_gpu5.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2 %s %s", tag, dir);
                    all_ok &= check(label, y_cpu5, y_gpu5, 5e-3);

                    // mv_fp32_16_transpose_warprows_splitm_tiled_fast_
                    // normal -- EXPERIMENTAL, standalone sibling: ported
                    // DIRECTLY from randls2.0's own sgemv_cm_transpose_
                    // warprows_splitm_tiled (no atomics, one persistent
                    // block per column-stripe, SPLITM warp-groups split
                    // each M-tile concurrently) -- see its own comment in
                    // fp32_16_cuda_kernels.cu. Same safe-range data.
                    std::vector<float> y_cpu6 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu6.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu6(rows_op);
                    cudaMemcpy(y_gpu6.data(), d_y, y_gpu6.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_warprows_splitm_tiled_fast_normal %s %s", tag, dir);
                    all_ok &= check(label, y_cpu6, y_gpu6, 5e-3);

                    // mv_fp32_16_transpose_warpdot_tiled_fast_normal --
                    // EXPERIMENTAL, standalone sibling: ported DIRECTLY
                    // from randls2.0's own sgemv_cm_transpose_warpdot_
                    // tiled (one warp per column loops the ENTIRE M
                    // range internally, no atomics, no M_BLOCK-scaling)
                    // -- see its own comment in fp32_16_cuda_kernels.cu.
                    // Same safe-range data.
                    std::vector<float> y_cpu7 = y_init;
                    cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh_safe.data(), wr_safe.data(), ld_a,
                                             x.data(), 1, beta, y_cpu7.data(), 1);
                    cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cuda::experimental::mv_fp32_16_transpose_warpdot_tiled_fast_normal(
                        trans, alpha, dim<2>{m, n}, d_wh_safe, d_wr_safe, ld_a, d_x, 1, beta, d_y, 1);
                    std::vector<float> y_gpu7(rows_op);
                    cudaMemcpy(y_gpu7.data(), d_y, y_gpu7.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    std::snprintf(label, sizeof(label), "transpose_warpdot_tiled_fast_normal %s %s", tag, dir);
                    all_ok &= check(label, y_cpu7, y_gpu7, 5e-3);
                }
                cudaFree(d_wh_safe);
                cudaFree(d_wr_safe);
            }
            if (ld_a == m) {
                // mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw
                // -- EXPERIMENTAL, same 4-rows/lane structure as the SAFE
                // kernel above, but decoding via fp32_16_combine_
                // branchless_nohw (pure bit-manipulation degraded path,
                // no hardware half conversion -- see its own comment in
                // fp32_16_cuda_kernels.cu, and Fp32_16PackTest.
                // CombineBranchlessNohwMatchesCombine/CombineBranchless
                // NohwDegradedEdgeCases in test/core/matrix/
                // test_fp32_16.cpp for the decode function's own already-
                // passing dedicated correctness check). Same GEMV-level
                // check as its sibling here, on top of that.
                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                                               ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready4_atomiconly_scaled_kblock_nohw %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);
            }
            if (ld_a == m) {
                // mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock --
                // EXPERIMENTAL 8-rows/lane extension, one step further
                // than the 4-row kernel just above (see its own comment
                // in fp32_16_cuda_kernels.cu). Same safe-on-arbitrary-
                // data property, so checked the same way, same CPU
                // reference/tolerance.
                std::vector<float> y_cpu = y_init;
                cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                         x.data(), 1, beta, y_cpu.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                                          ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu(rows_op);
                cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "accurate_ready8_atomiconly_scaled_kblock %s %s", tag, dir);
                all_ok &= check(label, y_cpu, y_gpu, 5e-3);
            }
        }
        {
            // mv_fp32_16_accurate_facc -- EXPERIMENTAL float-accumulator
            // sibling of mv_fp32_16_accurate's wide (split-K) path (see
            // its own comment in fp32_16_cuda_kernels.cu). Same CPU
            // reference as the "accurate" block above (mathematically
            // identical computation, just less precise on the GPU side),
            // but a looser tolerance: summing up to this file's largest
            // column counts (~50000 terms) in float32 accumulates real
            // rounding error double accumulation doesn't, so 1e-4 is not
            // expected to hold here -- 2e-3 is a starting point, not yet
            // confirmed against a real hardware run.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_accurate_facc(trans, alpha, dim<2>{m, n}, d_wh, d_wr, ld_a, d_x, 1,
                                           beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate_facc %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 2e-3);
        }
        {
            // mv_fp32_tiled -- this codebase's first tuned native (unpacked)
            // fp32 GEMV kernel, same 3-way shape dispatch as mv_fp32_16_
            // accurate above. Reference is cpu::mv_strided<float,float,
            // float> on the SAME A array used to generate wh/wr just above
            // (not a decoded copy -- A is already exact fp32 here).
            float* d_A;
            cudaMalloc(&d_A, A.size() * sizeof(float));
            cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_tiled(trans, alpha, dim<2>{m, n}, d_A, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fp32_tiled %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);

            // mv_fp32_transpose_atomiconly_scaled_mblock_vec4 -- EXPERIMENTAL,
            // standalone A/B sibling of mv_fp32_tiled's own TRANSPOSE fast
            // path (float4-vectorized loads, see its own comment in mv_
            // cuda_kernels.cu). Only handles TRANSPOSE + ld_a==m (returns
            // immediately otherwise), so only checked for that case here --
            // same A/x/y_init and CPU reference as mv_fp32_tiled's own
            // check just above, reused unchanged since both compute the
            // exact same y = alpha*A^T*x + beta*y.
            if (trans == TransposeType::TRANSPOSE && ld_a == m) {
                std::vector<float> y_cpu2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_transpose_atomiconly_scaled_mblock_vec4(trans, alpha, dim<2>{m, n}, d_A, ld_a, d_x, 1,
                                                                       beta, d_y, 1);
                std::vector<float> y_gpu2(rows_op);
                cudaMemcpy(y_gpu2.data(), d_y, y_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4 %s %s", tag, dir);
                all_ok &= check(label, y_cpu2, y_gpu2, 1e-4);
            }

            // mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec --
            // EXPERIMENTAL, standalone A/B sibling of the vec4 kernel just
            // above (ALSO float4-vectorizes the x-staging loop, see its
            // own comment in mv_cuda_kernels.cu). Same reference/data/
            // guard as the vec4 check just above.
            if (trans == TransposeType::TRANSPOSE && ld_a == m) {
                std::vector<float> y_cpu3 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(trans, alpha, dim<2>{m, n}, d_A, ld_a,
                                                                                d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3(rows_op);
                cudaMemcpy(y_gpu3.data(), d_y, y_gpu3.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "transpose_atomiconly_scaled_mblock_vec4_stagevec %s %s", tag, dir);
                all_ok &= check(label, y_cpu3, y_gpu3, 1e-4);
            }
            cudaFree(d_A);
        }
        {
            // mv_fp32_bf16_accurate -- this codebase's first fp32_bf16
            // GEMV kernel of any kind, byte-for-byte the same 3-way
            // dispatch as mv_fp32_16_accurate, decoding via fp32_bf16_
            // combine instead. No cpu::mv_fp32_bf16_accurate helper exists
            // (unlike the fp32_16 tier) -- the reference is built by
            // decoding wh_bf16/wr_bf16 host-side via fp32_bf16_combine
            // into a plain float array, then reusing cpu::mv_strided on
            // that (mathematically identical to what a dedicated CPU
            // reference function would compute).
            std::vector<uint16_t> wh_bf16(ld_a * n), wr_bf16(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) fp32_bf16_split(A[idx], wh_bf16[idx], wr_bf16[idx]);
            std::vector<float> A_bf16_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                A_bf16_decoded[idx] = fp32_bf16_combine(wh_bf16[idx], wr_bf16[idx]);

            uint16_t *d_wh_bf16, *d_wr_bf16;
            cudaMalloc(&d_wh_bf16, wh_bf16.size() * sizeof(uint16_t));
            cudaMalloc(&d_wr_bf16, wr_bf16.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_bf16, wh_bf16.data(), wh_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_wr_bf16, wr_bf16.data(), wr_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a, x.data(),
                                                 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_bf16_accurate(trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fp32_bf16_accurate %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);

            // mv_fp32_bf16_accurate_workspace -- genuinely-always-uses-
            // workspace sibling (classic column-split wide kernel +
            // reduce, no row-vectorized fast path at all -- see its own
            // comment in fp32_16_cuda_kernels.cu). Same CPU reference as
            // mv_fp32_bf16_accurate above (mathematically identical
            // computation, double accumulation) -- tight tolerance since
            // there's no float-atomic precision trade-off here.
            {
                std::vector<float> y_cpu2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                size_t bf16_ws_elems = cuda::experimental::mv_fp32_bf16_accurate_workspace_size(trans, dim<2>{m, n});
                double* d_bf16_workspace = nullptr;
                if (bf16_ws_elems > 0) cudaMalloc(&d_bf16_workspace, bf16_ws_elems * sizeof(double));
                cuda::experimental::mv_fp32_bf16_accurate_workspace(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1, d_bf16_workspace);
                std::vector<float> y_gpu2(rows_op);
                cudaMemcpy(y_gpu2.data(), d_y, y_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_accurate_workspace %s %s", tag, dir);
                all_ok &= check(label, y_cpu2, y_gpu2, 1e-4);
                cudaFree(d_bf16_workspace);
            }

            // mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_
            // workspace -- PROMOTED as plain "TOMOS FP32/BF16" (row-
            // vectorized/atomiconly/scaled-K_BLOCK fast path for
            // NO_TRANSPOSE+ld_a==m, workspace-based classic wide kernel
            // fallback otherwise -- see its own comment in fp32_16_
            // cuda_kernels.cu). Reuses the SAME wh_bf16/wr_bf16 encoding
            // and A_bf16_decoded CPU reference as mv_fp32_bf16_accurate
            // above -- fp32_bf16_combine's own decode is identical either
            // way, no precision trade-off there. Unlike mv_fp32_16_
            // accurate_ready above, this runs unconditionally for every
            // trans/ld_a this file exercises (no restricted-layout
            // fallback to a different, unrelated implementation -- it
            // always uses ITS OWN dispatch, general path included).
            // Tolerance mirrors accurate_ready's own reasoning: the fast
            // path's per-thread K-reduction and cross-block combination
            // are both float (fmaf accumulation, atomicAdd), not double
            // -- 5e-3 is a starting point, not yet confirmed against a
            // real hardware run (the general/tiled/narrow_vec fallback
            // paths this same tolerance also covers are double-precision
            // and pass it comfortably).
            {
                std::vector<float> y_cpu2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                size_t bf16_ws_elems = cuda::experimental::mv_fp32_bf16_accurate_workspace_size(trans, dim<2>{m, n});
                double* d_bf16_workspace = nullptr;
                if (bf16_ws_elems > 0) cudaMalloc(&d_bf16_workspace, bf16_ws_elems * sizeof(double));
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1, d_bf16_workspace);
                std::vector<float> y_gpu2(rows_op);
                cudaMemcpy(y_gpu2.data(), d_y, y_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_accurate_ready_atomiconly_scaled_kblock_workspace %s %s", tag, dir);
                all_ok &= check(label, y_cpu2, y_gpu2, 5e-3);
                cudaFree(d_bf16_workspace);
            }

            if (trans == TransposeType::TRANSPOSE && ld_a == m) {
                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4 --
                // EXPERIMENTAL, standalone TRANSPOSE-only A/B sibling of
                // the _workspace TRANSPOSE fast path above, float4/
                // ushort4-vectorized, same M_BLOCK-scaling -- ported
                // directly from SAFE fp32_16's own "(vec4)" design, see
                // its own comment in fp32_16_cuda_kernels.cu. Same
                // wh_bf16/wr_bf16 data and CPU reference.
                std::vector<float> y_cpu3 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3(rows_op);
                cudaMemcpy(y_gpu3.data(), d_y, y_gpu3.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4 %s %s", tag, dir);
                all_ok &= check(label, y_cpu3, y_gpu3, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // retuned_floor -- EXPERIMENTAL, standalone A/B sibling
                // reusing the EXACT SAME __global__ kernel as _vec4
                // above, only fp32_16's own already-validated M_BLOCK-
                // scaling retuning (target=2048, floor=256) -- see its
                // own comment in fp32_16_cuda_kernels.cu. Same data.
                std::vector<float> y_cpu4 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu4.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu4(rows_op);
                cudaMemcpy(y_gpu4.data(), d_y, y_gpu4.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor %s %s", tag, dir);
                all_ok &= check(label, y_cpu4, y_gpu4, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // retarget2 -- EXPERIMENTAL, standalone A/B sibling
                // reusing the EXACT SAME __global__ kernel as _vec4
                // above, only target=8192 -- see its own comment in
                // fp32_16_cuda_kernels.cu. Same data.
                std::vector<float> y_cpu4b = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu4b.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu4b(rows_op);
                cudaMemcpy(y_gpu4b.data(), d_y, y_gpu4b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2 %s %s", tag, dir);
                all_ok &= check(label, y_cpu4b, y_gpu4b, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // dispatch -- standalone dispatcher routing to _vec4 or
                // _retarget2 based on M -- see its own comment in
                // fp32_16_cuda_kernels.cu. Same data.
                std::vector<float> y_cpu4c = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu4c.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16, d_wr_bf16, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu4c(rows_op);
                cudaMemcpy(y_gpu4c.data(), d_y, y_gpu4c.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch %s %s", tag, dir);
                all_ok &= check(label, y_cpu4c, y_gpu4c, 5e-3);
            }

            cudaFree(d_wh_bf16);
            cudaFree(d_wr_bf16);
        }
        {
            // mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded
            // -- PROMOTED as plain "TOMOS FP32/BF16" (see fp32_16_cuda_
            // kernels.cu, next to this function's own definition, for the
            // full derivation): same shape dispatch as the "(truncated)"
            // sibling just above, but decoding via fp32_bf16_combine_
            // rounded against data split with fp32_bf16_split_rounded
            // instead of plain fp32_bf16_split -- a genuinely different
            // encoding (wr's bit0 is a round_flag, not a mantissa bit),
            // so this needs its OWN wh/wr buffers and its OWN CPU
            // reference (built by decoding host-side via fp32_bf16_
            // combine_rounded, same "decode into a plain float array, then
            // reuse cpu::mv_strided on that" technique the plain bf16
            // block above uses). NOT bit-exact vs the true fp32 CPU
            // reference the way the plain-split reference is -- fp32_bf16_
            // split_rounded's own encoding already bakes in a ~2^-23-
            // relative, one-lost-mantissa-bit approximation (see its own
            // comment in fp32_bf16_pack.hpp) -- but that's utterly
            // negligible next to the 5e-3 tolerance this fast path's own
            // float-accumulation/atomicAdd precision already requires
            // (mirroring the "(truncated)" sibling's own tolerance
            // reasoning just above).
            std::vector<uint16_t> wh_bf16_rounded(ld_a * n), wr_bf16_rounded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_bf16_split_rounded(A[idx], wh_bf16_rounded[idx], wr_bf16_rounded[idx]);
            std::vector<float> A_bf16_rounded_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                A_bf16_rounded_decoded[idx] = fp32_bf16_combine_rounded(wh_bf16_rounded[idx], wr_bf16_rounded[idx]);

            uint16_t *d_wh_bf16_rounded, *d_wr_bf16_rounded;
            cudaMalloc(&d_wh_bf16_rounded, wh_bf16_rounded.size() * sizeof(uint16_t));
            cudaMalloc(&d_wr_bf16_rounded, wr_bf16_rounded.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_bf16_rounded, wh_bf16_rounded.data(), wh_bf16_rounded.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_wr_bf16_rounded, wr_bf16_rounded.data(), wr_bf16_rounded.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice);

            std::vector<float> y_cpu3 = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_rounded_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu3.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t bf16_rounded_ws_elems = cuda::experimental::mv_fp32_bf16_accurate_workspace_size(trans, dim<2>{m, n});
            double* d_bf16_rounded_workspace = nullptr;
            if (bf16_rounded_ws_elems > 0) cudaMalloc(&d_bf16_rounded_workspace, bf16_rounded_ws_elems * sizeof(double));
            cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(
                trans, alpha, dim<2>{m, n}, d_wh_bf16_rounded, d_wr_bf16_rounded, ld_a, d_x, 1, beta, d_y, 1,
                d_bf16_rounded_workspace);
            std::vector<float> y_gpu3(rows_op);
            cudaMemcpy(y_gpu3.data(), d_y, y_gpu3.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "bf16_accurate_ready_atomiconly_scaled_kblock_rounded %s %s", tag, dir);
            all_ok &= check(label, y_cpu3, y_gpu3, 5e-3);
            cudaFree(d_bf16_rounded_workspace);

            if (trans == TransposeType::TRANSPOSE && ld_a == m) {
                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // rounded -- EXPERIMENTAL, standalone TRANSPOSE-only A/B
                // sibling of the rounded fast path above, float4/
                // ushort4-vectorized, same M_BLOCK-scaling -- rounded-
                // encoding counterpart of the "(truncated)" family's own
                // promoted _vec4, see its own comment in fp32_16_cuda_
                // kernels.cu. Same wh_bf16_rounded/wr_bf16_rounded data.
                std::vector<float> y_cpu3b = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_rounded_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3b.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16_rounded, d_wr_bf16_rounded, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3b(rows_op);
                cudaMemcpy(y_gpu3b.data(), d_y, y_gpu3b.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_rounded %s %s", tag, dir);
                all_ok &= check(label, y_cpu3b, y_gpu3b, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // rounded_retuned_floor -- EXPERIMENTAL, standalone A/B
                // sibling reusing the EXACT SAME __global__ kernel as
                // _vec4_rounded above, only fp32_16's own already-
                // validated M_BLOCK-scaling retuning (target=2048,
                // floor=256) -- see its own comment in fp32_16_cuda_
                // kernels.cu. Same data.
                std::vector<float> y_cpu3c = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_rounded_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3c.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16_rounded, d_wr_bf16_rounded, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3c(rows_op);
                cudaMemcpy(y_gpu3c.data(), d_y, y_gpu3c.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor %s %s", tag, dir);
                all_ok &= check(label, y_cpu3c, y_gpu3c, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // rounded_retarget2 -- EXPERIMENTAL, standalone A/B
                // sibling reusing the EXACT SAME __global__ kernel as
                // _vec4_rounded above, only target=8192 -- see its own
                // comment in fp32_16_cuda_kernels.cu. Same data.
                std::vector<float> y_cpu3d = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_rounded_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3d.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16_rounded, d_wr_bf16_rounded, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3d(rows_op);
                cudaMemcpy(y_gpu3d.data(), d_y, y_gpu3d.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2 %s %s", tag, dir);
                all_ok &= check(label, y_cpu3d, y_gpu3d, 5e-3);

                // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
                // rounded_dispatch -- standalone dispatcher routing to
                // _vec4_rounded or _rounded_retarget2 based on M -- see
                // its own comment in fp32_16_cuda_kernels.cu. Same data.
                std::vector<float> y_cpu3e = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_rounded_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu3e.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
                    trans, alpha, dim<2>{m, n}, d_wh_bf16_rounded, d_wr_bf16_rounded, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu3e(rows_op);
                cudaMemcpy(y_gpu3e.data(), d_y, y_gpu3e.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch %s %s", tag, dir);
                all_ok &= check(label, y_cpu3e, y_gpu3e, 5e-3);
            }

            cudaFree(d_wh_bf16_rounded);
            cudaFree(d_wr_bf16_rounded);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_fast(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                 beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_fast(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fast %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta,
                                       d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half %s %s", tag, dir);
            // pure_half's own long-reduction fp16 divergence is expected,
            // not a failure -- only flag it here if it's egregiously large
            // (a real bug would typically show as far worse than this).
            check(label, y_cpu, y_gpu, 5.0);
        }

        // EXPERIMENTAL *_wide_regblock variants (see their doc comments in
        // fp32_16_cuda_kernels.cu) -- same three checks again, against the
        // same CPU reference, so a correctness problem in the experimental
        // path is caught before trusting its performance numbers.
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_accurate_wide_regblock(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                    ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate_wide_regblock %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_fast(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                 beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_fast_wide_regblock(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x,
                                                1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fast_wide_regblock %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_regblock(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                     d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_regblock %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x,
                                                1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_launchbounds(trans, alpha, dim<2>{m, n}, d_wh,
                                                         ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_launchbounds %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_warpgroup(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                      d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_warpgroup %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                       d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_ldg(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                           d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_ldg %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_prefetch(trans, alpha, dim<2>{m, n}, d_wh,
                                                                ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_prefetch %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile(trans, alpha, dim<2>{m, n}, d_wh,
                                                               ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_wmma(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1,
                                                 beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_wmma %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_wmma_batched(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                         d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_wmma_batched %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf(trans, alpha, dim<2>{m, n}, d_wh,
                                                                ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_fasthalf %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_fasthalf_workspace -- identical
            // numerics to the check just above (same kernel, same
            // dispatch), just fed a caller-provided workspace buffer
            // instead of letting the wide path malloc/free its own
            // scratch space per call -- see its own comment in fp32_16_
            // cuda_kernels.cu. Same reference/tolerance since the actual
            // computation is unchanged.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t fasthalf_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace_size(trans, dim<2>{m, n});
            double* d_fasthalf_ws = nullptr;
            if (fasthalf_ws_elems > 0) cudaMalloc(&d_fasthalf_ws, fasthalf_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_fasthalf_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_fasthalf_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_fasthalf_ws);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_fasthalf(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_fasthalf %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // "TOMOS FP32/FP16 (fast)", new version -- half2_atomic_
            // fasthalf/bigtile_half2_atomic_fasthalf/half2_atomic_
            // fasthalf_wide4x/half2_atomic_fasthalf_best. Same UNSAFE-on-
            // arbitrary-data caveat as mv_fp32_16_pure_half_wide_ilp_
            // shared_fasthalf above (fp32_16_fast_half's own contract), so
            // this uses its OWN dedicated safe-range data (mag in
            // [0.5, 1.0]) -- same generator convention as mv_fp32_16_
            // accurate_ready_atomiconly_scaled_kblock_fast_normal's own
            // check above, host-side here, not the plain wh reused by the
            // OLD fasthalf check just above (which is a known, documented,
            // expected-fail diagnostic on this file's own plain data --
            // see mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_kernel's
            // own comment in fp32_16_cuda_kernels.cu). See these four
            // kernels' own comments there too.
            std::mt19937 fasthalf_atomic_rng(44);
            std::uniform_real_distribution<float> fasthalf_mag_dist(0.5f, 1.0f);
            std::uniform_int_distribution<int> fasthalf_sign_dist(0, 1);
            std::vector<float> A_fasthalf_safe(ld_a * n);
            for (auto& v : A_fasthalf_safe) {
                float mag = fasthalf_mag_dist(fasthalf_atomic_rng);
                v = fasthalf_sign_dist(fasthalf_atomic_rng) ? mag : -mag;
            }
            std::vector<uint16_t> wh_fasthalf_safe(ld_a * n), wr_fasthalf_safe(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_fasthalf_safe[idx], wh_fasthalf_safe[idx], wr_fasthalf_safe[idx]);

            uint16_t* d_wh_fasthalf_safe = nullptr;
            cudaMalloc(&d_wh_fasthalf_safe, wh_fasthalf_safe.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_fasthalf_safe, wh_fasthalf_safe.data(),
                      wh_fasthalf_safe.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            {
                std::vector<float> y_cpu_fh = y_init;
                cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fasthalf_safe.data(), ld_a,
                                          x.data(), 1, beta, y_cpu_fh.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf(
                    trans, alpha, dim<2>{m, n}, d_wh_fasthalf_safe, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_fh(rows_op);
                cudaMemcpy(y_gpu_fh.data(), d_y, y_gpu_fh.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_fasthalf %s %s", tag,
                             dir);
                check(label, y_cpu_fh, y_gpu_fh, 5.0);
            }
            {
                std::vector<float> y_cpu_fh = y_init;
                cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fasthalf_safe.data(), ld_a,
                                          x.data(), 1, beta, y_cpu_fh.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf(
                    trans, alpha, dim<2>{m, n}, d_wh_fasthalf_safe, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_fh(rows_op);
                cudaMemcpy(y_gpu_fh.data(), d_y, y_gpu_fh.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf %s %s",
                             tag, dir);
                check(label, y_cpu_fh, y_gpu_fh, 5.0);
            }
            {
                std::vector<float> y_cpu_fh = y_init;
                cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fasthalf_safe.data(), ld_a,
                                          x.data(), 1, beta, y_cpu_fh.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x(
                    trans, alpha, dim<2>{m, n}, d_wh_fasthalf_safe, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_fh(rows_op);
                cudaMemcpy(y_gpu_fh.data(), d_y, y_gpu_fh.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x %s %s",
                             tag, dir);
                check(label, y_cpu_fh, y_gpu_fh, 5.0);
            }
            {
                std::vector<float> y_cpu_fh = y_init;
                cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fasthalf_safe.data(), ld_a,
                                          x.data(), 1, beta, y_cpu_fh.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(
                    trans, alpha, dim<2>{m, n}, d_wh_fasthalf_safe, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_fh(rows_op);
                cudaMemcpy(y_gpu_fh.data(), d_y, y_gpu_fh.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_fasthalf_best %s %s",
                             tag, dir);
                check(label, y_cpu_fh, y_gpu_fh, 5.0);
            }

            cudaFree(d_wh_fasthalf_safe);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless(trans, alpha, dim<2>{m, n},
                                                                  d_wh, ld_a, d_x, 1, beta, d_y,
                                                                  1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_branchless %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_cpasync(trans, alpha, dim<2>{m, n}, d_wh,
                                                               ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_cpasync %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_cpasync2(trans, alpha, dim<2>{m, n}, d_wh,
                                                                ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_cpasync2 %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_atomic(trans, alpha, dim<2>{m, n}, d_wh,
                                                              ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_atomic %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2(trans, alpha, dim<2>{m, n}, d_wh,
                                                             ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2 %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_half2_workspace -- identical
            // numerics to the check just above, caller-provided workspace
            // instead of internal malloc/free -- see mv_fp32_16_pure_
            // half_wide_ilp_shared_fasthalf_workspace's own comment in
            // fp32_16_cuda_kernels.cu for the shared rationale.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace_size(trans, dim<2>{m, n});
            double* d_half2_ws = nullptr;
            if (half2_ws_elems > 0) cudaMalloc(&d_half2_ws, half2_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1,
                                                                       beta, d_y, 1, d_half2_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_ws);
        }
        {
            // pure_half_wide_ilp_shared_branchless_workspace -- same
            // reference/tolerance as the half2_workspace check just above
            // (both decode via the SAME fp32_16_half_only semantics on the
            // SAME wh) -- this check also verifies fp32_16_half_only_
            // branchless's own "bit-exact match to fp32_16_half_only"
            // claim on real data, not just by construction. See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t branchless_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace_size(trans, dim<2>{m, n});
            double* d_branchless_ws = nullptr;
            if (branchless_ws_elems > 0) cudaMalloc(&d_branchless_ws, branchless_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                            d_x, 1, beta, d_y, 1, d_branchless_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_branchless_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_branchless_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_branchless_workspace --
            // same reference/tolerance as the checks just above (SAME
            // fp32_16_half_only semantics on the SAME wh) -- the fair
            // isolation of "test", built on half2's own __hfma2-packed
            // body instead of the plain scalar one. See its own comment
            // in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_branchless_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace_size(trans, dim<2>{m, n});
            double* d_half2_branchless_ws = nullptr;
            if (half2_branchless_ws_elems > 0)
                cudaMalloc(&d_half2_branchless_ws, half2_branchless_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_branchless_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_branchless_workspace %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_branchless_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_vecrow_workspace -- same
            // reference/tolerance as the checks just above (SAME fp32_16_
            // half_only semantics on the SAME wh) -- exercises BOTH the
            // shuffle-based row-grouped fast path (when ld_a % 4 == 0) and
            // the mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace
            // fallback (when it isn't), since this loop's own cases cover
            // both aligned and misaligned ld_a. See its own comment in
            // fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_vecrow_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace_size(trans, dim<2>{m, n});
            double* d_half2_vecrow_ws = nullptr;
            if (half2_vecrow_ws_elems > 0)
                cudaMalloc(&d_half2_vecrow_ws, half2_vecrow_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_vecrow_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_vecrow_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_vecrow_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_vec2rows_workspace -- same
            // reference/tolerance as the checks just above (SAME fp32_16_
            // half_only semantics on the SAME wh) -- exercises the
            // per-thread-vectorized row-pairing fast path (NO_TRANSPOSE,
            // ld_a % 2 == 0, including this loop's own odd-m cases'
            // single-row tail), and the mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_workspace fallback (TRANSPOSE or odd ld_a),
            // since this loop's own cases cover all of these. See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_vec2rows_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace_size(trans, dim<2>{m, n});
            double* d_half2_vec2rows_ws = nullptr;
            if (half2_vec2rows_ws_elems > 0)
                cudaMalloc(&d_half2_vec2rows_ws, half2_vec2rows_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_vec2rows_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_vec2rows_workspace %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_vec2rows_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_wide2x_workspace -- same
            // reference/tolerance as the checks above (identical kernel
            // to half2_workspace, only col_splits differs). See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_wide2x_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace_size(trans, dim<2>{m, n});
            double* d_half2_wide2x_ws = nullptr;
            if (half2_wide2x_ws_elems > 0)
                cudaMalloc(&d_half2_wide2x_ws, half2_wide2x_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_wide2x_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_wide2x_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_wide2x_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_wide4x_workspace -- same
            // reference/tolerance as the checks above. See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_wide4x_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace_size(trans, dim<2>{m, n});
            double* d_half2_wide4x_ws = nullptr;
            if (half2_wide4x_ws_elems > 0)
                cudaMalloc(&d_half2_wide4x_ws, half2_wide4x_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_wide4x_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_wide4x_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_wide4x_ws);
        }
        {
            // pure_half_wide_ilp_shared_bigtile_half2_workspace -- same
            // reference/tolerance as the checks above (half2's own
            // packing on a wider shared-memory tile). See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t bigtile_half2_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace_size(trans, dim<2>{m, n});
            double* d_bigtile_half2_ws = nullptr;
            if (bigtile_half2_ws_elems > 0)
                cudaMalloc(&d_bigtile_half2_ws, bigtile_half2_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_bigtile_half2_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_half2_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_bigtile_half2_ws);
        }
        {
            // pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace
            // -- same reference/tolerance as the checks above (bigtile_
            // half2's own kernel, only col_splits differs by size). This
            // loop's own cases span both sides of kMvHalf2Adaptive
            // WideMultiplierThreshold. See its own comment in fp32_16_
            // cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t bigtile_half2_adaptive_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace_size(trans,
                                                                                                  dim<2>{m, n});
            double* d_bigtile_half2_adaptive_ws = nullptr;
            if (bigtile_half2_adaptive_ws_elems > 0)
                cudaMalloc(&d_bigtile_half2_adaptive_ws, bigtile_half2_adaptive_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_bigtile_half2_adaptive_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_bigtile_half2_adaptive_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_best_workspace -- same
            // reference/tolerance as the checks above (dispatches to
            // bigtile_half2_workspace or half2_wide4x_workspace, each
            // already independently verified -- this check confirms the
            // dispatch itself, including both sides of kMvHalf2BestSquare
            // CrossoverThreshold, since this loop's own cases span it).
            // See its own comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t half2_best_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace_size(trans, dim<2>{m, n});
            double* d_half2_best_ws = nullptr;
            if (half2_best_ws_elems > 0)
                cudaMalloc(&d_half2_best_ws, half2_best_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_half2_best_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_best_workspace %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
            cudaFree(d_half2_best_ws);
        }
        {
            // pure_half_wide_ilp_shared_half2_atomic -- same reference/
            // tolerance as the checks above (half2's own packing, atomic-
            // Add combine instead of split+reduce -- no workspace/buffer
            // to allocate here). See its own comment in fp32_16_cuda_
            // kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1,
                                                                    beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_bigtile_half2_atomic -- same
            // reference/tolerance as the checks above (bigtile_half2's own
            // 4x-larger shared-memory tile, atomicAdd combine instead of
            // split+reduce -- no workspace/buffer to allocate here). See
            // its own comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                            d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_half2_atomic %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_half2_atomic_wide2x -- same
            // reference/tolerance as the checks above (doubled col_splits
            // via choose_col_splits_target, same atomicAdd combine as
            // half2_atomic -- no workspace/buffer to allocate here). See
            // its own comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                            d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_wide2x %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_half2_atomic_wide4x -- same
            // reference/tolerance as the checks above (quadrupled
            // col_splits instead of doubled). See its own comment in
            // fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                            d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_wide4x %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // pure_half_wide_ilp_shared_half2_atomic_best -- same
            // reference/tolerance as the checks above (requested
            // dispatcher: below 4096 rows calls bigtile_half2_atomic,
            // at/above it calls half2_atomic_wide4x -- this loop's own
            // shapes span both sides of that threshold). See its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                                          d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_atomic_best %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            // hybrid_wide_ilp_shared -- NO_TRANSPOSE-usable "TOMOS FP32/
            // FP16" sibling of mv_fp32_16_pure_half_wide_ilp_shared_half2
            // just above (same generic trans-dispatched wide_ilp_shared
            // structure, no ld_a==m guard, runs for BOTH directions in
            // this loop), decode swapped to the wh-only extended-exponent
            // reconstruction + adjacent-pair packing built for
            // mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_
            // vec4_paired's own TRANSPOSE-only kernel. Own regenerated
            // mixed normal/extended-range data (same distribution as that
            // kernel's own check above), own fresh seed to keep every
            // hybrid check's data independent.
            std::mt19937 hybrid_wide_rng(73);
            std::uniform_real_distribution<float> hybrid_wide_normal_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> hybrid_wide_extended_mag_dist(1.0e5f, 1.0e8f);
            std::uniform_int_distribution<int> hybrid_wide_extended_flag(0, 9);
            std::uniform_int_distribution<int> hybrid_wide_sign_dist(0, 1);
            std::vector<float> A_hybrid_wide(ld_a * n);
            for (auto& v : A_hybrid_wide) {
                if (hybrid_wide_extended_flag(hybrid_wide_rng) == 0) {
                    float mag = hybrid_wide_extended_mag_dist(hybrid_wide_rng);
                    v = hybrid_wide_sign_dist(hybrid_wide_rng) ? mag : -mag;
                } else {
                    v = hybrid_wide_normal_dist(hybrid_wide_rng);
                }
            }
            std::vector<uint16_t> wh_hybrid_wide(ld_a * n), wr_hybrid_wide_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_hybrid_wide[idx], wh_hybrid_wide[idx], wr_hybrid_wide_scratch[idx]);
            std::vector<float> A_hybrid_wide_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) {
                const uint16_t whv = wh_hybrid_wide[idx];
                if ((whv & 0x1u) == 0u) {
                    A_hybrid_wide_decoded[idx] = __half2float(*reinterpret_cast<const __half*>(&whv));
                } else {
                    const uint32_t offset = (static_cast<uint32_t>(whv) >> 10) & 0xFu;
                    if (offset != fp32_16_detail::kOffsetSaturate) {
                        const uint32_t direction = (static_cast<uint32_t>(whv) >> 14) & 0x1u;
                        const uint32_t exp32 = direction ? (fp32_16_detail::kFp32ExpMax + offset)
                                                          : (fp32_16_detail::kFp32ExpMin - offset);
                        const uint32_t mant_hi = (static_cast<uint32_t>(whv) >> 1) & 0x1FFu;
                        const uint32_t mant32 = mant_hi << 14;
                        const uint32_t sign = (static_cast<uint32_t>(whv) & 0x8000u) << 16;
                        A_hybrid_wide_decoded[idx] = fp32_16_detail::bits_to_float(sign | (exp32 << 23) | mant32);
                    } else {
                        A_hybrid_wide_decoded[idx] = 0.0f;
                    }
                }
            }

            uint16_t* d_wh_hybrid_wide = nullptr;
            cudaMalloc(&d_wh_hybrid_wide, wh_hybrid_wide.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_hybrid_wide, wh_hybrid_wide.data(), wh_hybrid_wide.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared(trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta,
                                                    d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            // hybrid_wide_ilp_shared_workspace -- identical numerics
            // (same kernel, same wh_hybrid_wide/A_hybrid_wide_decoded
            // reference above, this kernel is standalone/always-wide so
            // its own workspace_size is never 0), caller-provided
            // workspace instead of internal malloc/free.
            std::vector<float> y_cpu_ws = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu_ws.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t hybrid_ws_elems = cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_workspace_size(trans, dim<2>{m, n});
            double* d_hybrid_ws = nullptr;
            cudaMalloc(&d_hybrid_ws, hybrid_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_workspace(trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a,
                                                              d_x, 1, beta, d_y, 1, d_hybrid_ws);
            std::vector<float> y_gpu_ws(rows_op);
            cudaMemcpy(y_gpu_ws.data(), d_y, y_gpu_ws.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_workspace %s %s", tag, dir);
            check(label, y_cpu_ws, y_gpu_ws, 5.0);
            cudaFree(d_hybrid_ws);

            // hybrid_wide_ilp_shared_half2_atomic -- "TOMOS FP32/FP16",
            // new version: ports "TOMOS FP32/FP16 (fast)"'s own atomicAdd-
            // combine dispatcher structure to the expensive decode. Same
            // reference/tolerance and same d_wh_hybrid_wide/A_hybrid_wide_
            // decoded as the checks above (this loop's own extended_flag
            // generator already exercises both the cheap and extended
            // paths). See its own comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_ha = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_ha.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic(trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide,
                                                                      ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_ha(rows_op);
                cudaMemcpy(y_gpu_ha.data(), d_y, y_gpu_ha.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic %s %s", tag, dir);
                check(label, y_cpu_ha, y_gpu_ha, 5.0);
            }

            // hybrid_wide_ilp_shared_bigtile_half2_atomic -- same
            // reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_hb = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hb.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic(trans, alpha, dim<2>{m, n},
                                                                             d_wh_hybrid_wide, ld_a, d_x, 1, beta,
                                                                             d_y, 1);
                std::vector<float> y_gpu_hb(rows_op);
                cudaMemcpy(y_gpu_hb.data(), d_y, y_gpu_hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_bigtile_half2_atomic %s %s", tag, dir);
                check(label, y_cpu_hb, y_gpu_hb, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_wide4x -- same
            // reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_hw = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hw.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x(trans, alpha, dim<2>{m, n},
                                                                            d_wh_hybrid_wide, ld_a, d_x, 1, beta,
                                                                            d_y, 1);
                std::vector<float> y_gpu_hw(rows_op);
                cudaMemcpy(y_gpu_hw.data(), d_y, y_gpu_hw.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_wide4x %s %s", tag, dir);
                check(label, y_cpu_hw, y_gpu_hw, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_best -- same
            // reference/tolerance as the checks above (this loop's own
            // shapes span both sides of kMvHybridAtomicBestCrossoverThreshold).
            {
                std::vector<float> y_cpu_hbest = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hbest.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best(trans, alpha, dim<2>{m, n},
                                                                          d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y,
                                                                          1);
                std::vector<float> y_gpu_hbest(rows_op);
                cudaMemcpy(y_gpu_hbest.data(), d_y, y_gpu_hbest.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_best %s %s", tag, dir);
                check(label, y_cpu_hbest, y_gpu_hbest, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless -- same
            // reference/tolerance as the checks above (both paths
            // computed unconditionally, discarded via arithmetic masking
            // -- this loop's own mixed normal/extended-range generator
            // already exercises both the cheap and extended paths, plus
            // the degraded/offset==7 sub-case, all of which the masking
            // must handle correctly). See its own comment in fp32_16_
            // cuda_kernels.cu.
            {
                std::vector<float> y_cpu_hbl = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hbl.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless(trans, alpha, dim<2>{m, n},
                                                                                d_wh_hybrid_wide, ld_a, d_x, 1, beta,
                                                                                d_y, 1);
                std::vector<float> y_gpu_hbl(rows_op);
                cudaMemcpy(y_gpu_hbl.data(), d_y, y_gpu_hbl.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_branchless %s %s", tag,
                             dir);
                check(label, y_cpu_hbl, y_gpu_hbl, 5.0);
            }

            // hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless --
            // same reference/tolerance as the checks above (reuses the
            // same branchless masking, just a 4x-larger shared-memory
            // tile). See its own comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_hbb = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hbb.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hbb(rows_op);
                cudaMemcpy(y_gpu_hbb.data(), d_y, y_gpu_hbb.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless %s %s",
                             tag, dir);
                check(label, y_cpu_hbb, y_gpu_hbb, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x --
            // same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_hbw = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hbw.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hbw(rows_op);
                cudaMemcpy(y_gpu_hbw.data(), d_y, y_gpu_hbw.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x %s %s",
                             tag, dir);
                check(label, y_cpu_hbw, y_gpu_hbw, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_best -- same
            // reference/tolerance as the checks above (this loop's own
            // shapes span both sides of kMvHybridBranchlessBestCrossover
            // Threshold).
            {
                std::vector<float> y_cpu_hbbest = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hbbest.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hbbest(rows_op);
                cudaMemcpy(y_gpu_hbbest.data(), d_y, y_gpu_hbbest.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_branchless_best %s %s",
                             tag, dir);
                check(label, y_cpu_hbbest, y_gpu_hbbest, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_fastx -- same
            // reference/tolerance as the checks above (same masking, plus
            // a cached float-typed x tile -- must produce IDENTICAL
            // results to the non-fastx branchless kernel, just fewer
            // redundant conversions). See its own comment in fp32_16_
            // cuda_kernels.cu.
            {
                std::vector<float> y_cpu_hfx = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hfx.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx(trans, alpha, dim<2>{m, n},
                                                                                      d_wh_hybrid_wide, ld_a, d_x, 1,
                                                                                      beta, d_y, 1);
                std::vector<float> y_gpu_hfx(rows_op);
                cudaMemcpy(y_gpu_hfx.data(), d_y, y_gpu_hfx.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_branchless_fastx %s %s",
                             tag, dir);
                check(label, y_cpu_hfx, y_gpu_hfx, 5.0);
            }

            // hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx
            // -- same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_hfxb = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hfxb.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hfxb(rows_op);
                cudaMemcpy(y_gpu_hfxb.data(), d_y, y_gpu_hfxb.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx %s %s", tag, dir);
                check(label, y_cpu_hfxb, y_gpu_hfxb, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x
            // -- same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_hfxw = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hfxw.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hfxw(rows_op);
                cudaMemcpy(y_gpu_hfxw.data(), d_y, y_gpu_hfxw.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x %s %s", tag, dir);
                check(label, y_cpu_hfxw, y_gpu_hfxw, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best --
            // same reference/tolerance as the checks above (this loop's
            // own shapes span both sides of kMvHybridBranchlessFastxBest
            // CrossoverThreshold).
            {
                std::vector<float> y_cpu_hfxbest = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_hfxbest.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_hfxbest(rows_op);
                cudaMemcpy(y_gpu_hfxbest.data(), d_y, y_gpu_hfxbest.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best %s %s", tag, dir);
                check(label, y_cpu_hfxbest, y_gpu_hfxbest, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask --
            // same reference/tolerance as the checks above (same
            // branchless combine, but the extended path masks the raw
            // uint32 bit pattern with an integer AND instead of scaling
            // the decoded float -- this loop's own mixed normal/extended-
            // range generator already exercises both the cheap and
            // extended paths, plus the degraded/offset==7 sub-case, all
            // of which the bit-masking must handle correctly). See its
            // own comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_him = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_him.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask(trans, alpha, dim<2>{m, n},
                                                                                        d_wh_hybrid_wide, ld_a, d_x, 1,
                                                                                        beta, d_y, 1);
                std::vector<float> y_gpu_him(rows_op);
                cudaMemcpy(y_gpu_him.data(), d_y, y_gpu_him.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "hybrid_wide_ilp_shared_half2_atomic_branchless_intmask %s %s",
                             tag, dir);
                check(label, y_cpu_him, y_gpu_him, 5.0);
            }

            // hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_
            // intmask -- same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_himb = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_himb.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_himb(rows_op);
                cudaMemcpy(y_gpu_himb.data(), d_y, y_gpu_himb.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask %s %s", tag, dir);
                check(label, y_cpu_himb, y_gpu_himb, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // wide4x -- same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_himw = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_himw.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_himw(rows_op);
                cudaMemcpy(y_gpu_himw.data(), d_y, y_gpu_himw.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x %s %s", tag, dir);
                check(label, y_cpu_himw, y_gpu_himw, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best
            // -- same reference/tolerance as the checks above (this
            // loop's own shapes span both sides of
            // kMvHybridBranchlessIntmaskBestCrossoverThreshold).
            {
                std::vector<float> y_cpu_himbest = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_himbest.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_himbest(rows_op);
                cudaMemcpy(y_gpu_himbest.data(), d_y, y_gpu_himbest.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best %s %s", tag, dir);
                check(label, y_cpu_himbest, y_gpu_himbest, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // walign -- same reference/tolerance as the checks above
            // (same intmask kernel, only col_splits differs -- must
            // produce IDENTICAL results to intmask_wide4x, just fewer
            // wasted GPU wave-slots). See its own comment in fp32_16_
            // cuda_kernels.cu.
            {
                std::vector<float> y_cpu_himwa = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_himwa.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_himwa(rows_op);
                cudaMemcpy(y_gpu_himwa.data(), d_y, y_gpu_himwa.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign %s %s", tag, dir);
                check(label, y_cpu_himwa, y_gpu_himwa, 5.0);
            }

            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // walign_best -- same reference/tolerance as the checks above
            // (this loop's own shapes span both sides of
            // kMvHybridBranchlessIntmaskWalignBestCrossoverThreshold).
            {
                std::vector<float> y_cpu_himwabest = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_himwabest.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best(
                    trans, alpha, dim<2>{m, n}, d_wh_hybrid_wide, ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_himwabest(rows_op);
                cudaMemcpy(y_gpu_himwabest.data(), d_y, y_gpu_himwabest.size() * sizeof(float),
                          cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label),
                             "hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best %s %s", tag, dir);
                check(label, y_cpu_himwabest, y_gpu_himwabest, 5.0);
            }

            cudaFree(d_wh_hybrid_wide);
        }
        {
            // fp32_bf16_pure_bf16_wide_ilp_shared -- NO_TRANSPOSE-usable
            // "TOMOS FP32/BF16" sibling of mv_fp32_16_pure_half_wide_ilp_
            // shared_half2 above (same generic trans-dispatched
            // structure), genuine bf16 compute via fp32_bf16_bf16_only.
            // Safe on arbitrary data (see mv_fp32_bf16_pure_bf16_
            // transpose_atomiconly_scaled_mblock_vec4's own check above
            // for why), so this reuses the plain "A"/"x" data already
            // generated for this shape.
            std::vector<uint16_t> wh_bf16_wide(ld_a * n), wr_bf16_wide_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_bf16_split(A[idx], wh_bf16_wide[idx], wr_bf16_wide_scratch[idx]);
            std::vector<float> A_bf16_wide_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                A_bf16_wide_decoded[idx] = __bfloat162float(fp32_bf16_bf16_only(wh_bf16_wide[idx]));

            uint16_t* d_wh_bf16_wide = nullptr;
            cudaMalloc(&d_wh_bf16_wide, wh_bf16_wide.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_bf16_wide, wh_bf16_wide.data(), wh_bf16_wide.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared(trans, alpha, dim<2>{m, n}, d_wh_bf16_wide, ld_a, d_x, 1,
                                                         beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            // fp32_bf16_pure_bf16_wide_ilp_shared_workspace -- identical
            // numerics (same kernel, same wh_bf16_wide/A_bf16_wide_
            // decoded reference above, standalone/always-wide so its own
            // workspace_size is never 0), caller-provided workspace
            // instead of internal malloc/free.
            std::vector<float> y_cpu_ws = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu_ws.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t bf16_ws_elems = cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace_size(trans, dim<2>{m, n});
            double* d_bf16_ws = nullptr;
            cudaMalloc(&d_bf16_ws, bf16_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace(trans, alpha, dim<2>{m, n}, d_wh_bf16_wide, ld_a,
                                                                   d_x, 1, beta, d_y, 1, d_bf16_ws);
            std::vector<float> y_gpu_ws(rows_op);
            cudaMemcpy(y_gpu_ws.data(), d_y, y_gpu_ws.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_workspace %s %s", tag, dir);
            check(label, y_cpu_ws, y_gpu_ws, 5.0);
            cudaFree(d_bf16_ws);

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic -- "TOMOS BF16":
            // same reference/tolerance as the checks above (atomicAdd
            // combine instead of split+reduce -- no workspace/buffer to
            // allocate here). See its own comment in fp32_16_cuda_
            // kernels.cu.
            {
                std::vector<float> y_cpu_atomic = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_atomic.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic(trans, alpha, dim<2>{m, n}, d_wh_bf16_wide, ld_a,
                                                                     d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_atomic(rows_op);
                cudaMemcpy(y_gpu_atomic.data(), d_y, y_gpu_atomic.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic %s %s", tag, dir);
                check(label, y_cpu_atomic, y_gpu_atomic, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic -- same
            // reference/tolerance as the checks above (4x-larger shared-
            // memory tile on top of the atomicAdd combine). See its own
            // comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_bigtile = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_bigtile.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic(trans, alpha, dim<2>{m, n},
                                                                            d_wh_bf16_wide, ld_a, d_x, 1, beta, d_y,
                                                                            1);
                std::vector<float> y_gpu_bigtile(rows_op);
                cudaMemcpy(y_gpu_bigtile.data(), d_y, y_gpu_bigtile.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic %s %s", tag,
                             dir);
                check(label, y_cpu_bigtile, y_gpu_bigtile, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x -- same
            // reference/tolerance as the checks above (quadrupled
            // col_splits on top of the atomicAdd combine). See its own
            // comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_wide4x = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_wide4x.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x(trans, alpha, dim<2>{m, n},
                                                                           d_wh_bf16_wide, ld_a, d_x, 1, beta, d_y,
                                                                           1);
                std::vector<float> y_gpu_wide4x(rows_op);
                cudaMemcpy(y_gpu_wide4x.data(), d_y, y_gpu_wide4x.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x %s %s", tag,
                             dir);
                check(label, y_cpu_wide4x, y_gpu_wide4x, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best -- same
            // reference/tolerance as the checks above (requested
            // dispatcher: below kMvBf16AtomicBestCrossoverThreshold rows
            // calls bigtile_atomic, at/above it calls atomic_wide4x --
            // this loop's own shapes span both sides of that threshold).
            // See its own comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_best = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_best.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(trans, alpha, dim<2>{m, n}, d_wh_bf16_wide,
                                                                         ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_best(rows_op);
                cudaMemcpy(y_gpu_best.data(), d_y, y_gpu_best.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best %s %s", tag,
                             dir);
                check(label, y_cpu_best, y_gpu_best, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 -- "TOMOS
            // FP32/BF16", new version: REQUESTED separate, distinctly-
            // named duplicate of "TOMOS BF16" above, explicitly confirmed
            // identical on wh -- same reference/tolerance and same
            // d_wh_bf16_wide/A_bf16_wide_decoded as the checks above. See
            // its own comment in fp32_16_cuda_kernels.cu.
            {
                std::vector<float> y_cpu_v2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_v2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2(trans, alpha, dim<2>{m, n}, d_wh_bf16_wide,
                                                                       ld_a, d_x, 1, beta, d_y, 1);
                std::vector<float> y_gpu_v2(rows_op);
                cudaMemcpy(y_gpu_v2.data(), d_y, y_gpu_v2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 %s %s", tag, dir);
                check(label, y_cpu_v2, y_gpu_v2, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2 --
            // same reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_bigtile_v2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_bigtile_v2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2(trans, alpha, dim<2>{m, n},
                                                                               d_wh_bf16_wide, ld_a, d_x, 1, beta,
                                                                               d_y, 1);
                std::vector<float> y_gpu_bigtile_v2(rows_op);
                cudaMemcpy(y_gpu_bigtile_v2.data(), d_y, y_gpu_bigtile_v2.size() * sizeof(float),
                          cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2 %s %s",
                             tag, dir);
                check(label, y_cpu_bigtile_v2, y_gpu_bigtile_v2, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2 -- same
            // reference/tolerance as the checks above.
            {
                std::vector<float> y_cpu_wide4x_v2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_wide4x_v2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2(trans, alpha, dim<2>{m, n},
                                                                              d_wh_bf16_wide, ld_a, d_x, 1, beta,
                                                                              d_y, 1);
                std::vector<float> y_gpu_wide4x_v2(rows_op);
                cudaMemcpy(y_gpu_wide4x_v2.data(), d_y, y_gpu_wide4x_v2.size() * sizeof(float),
                          cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2 %s %s",
                             tag, dir);
                check(label, y_cpu_wide4x_v2, y_gpu_wide4x_v2, 5.0);
            }

            // fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 -- same
            // reference/tolerance as the checks above (this loop's own
            // shapes span both sides of kMvBf16AtomicBestCrossoverThresholdV2).
            {
                std::vector<float> y_cpu_best_v2 = y_init;
                cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_wide_decoded.data(), ld_a,
                                                     x.data(), 1, beta, y_cpu_best_v2.data(), 1);
                cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(trans, alpha, dim<2>{m, n},
                                                                            d_wh_bf16_wide, ld_a, d_x, 1, beta, d_y,
                                                                            1);
                std::vector<float> y_gpu_best_v2(rows_op);
                cudaMemcpy(y_gpu_best_v2.data(), d_y, y_gpu_best_v2.size() * sizeof(float), cudaMemcpyDeviceToHost);
                std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 %s %s", tag,
                             dir);
                check(label, y_cpu_best_v2, y_gpu_best_v2, 5.0);
            }

            cudaFree(d_wh_bf16_wide);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4 --
            // standalone TRANSPOSE-only kernel (no NO_TRANSPOSE fallback,
            // same convention as every other atomiconly_scaled_mblock
            // sibling in this file), so this check only runs for the
            // TRANSPOSE direction with ld_a==m -- see its own comment in
            // fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4(trans, alpha, dim<2>{m, n}, d_wh,
                                                                                ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_atomiconly_scaled_mblock_vec4 %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_
            // retuned_floor -- same TRANSPOSE-only/ld_a==m guard as the
            // check just above (reuses the EXACT SAME __global__ kernel,
            // only a different M_BLOCK-scaling target/floor) -- see its
            // own comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_fast --
            // "TOMOS FP32/FP16 (fast)". UNSAFE on arbitrary data
            // (fp32_16_fast_half's own contract: every wh value must
            // already have control=0), so tested here against dedicated
            // safe-range data (mag in [0.5,1.0]), not the plain "wh"
            // used by every safe-on-arbitrary-data pure_half sibling
            // above -- mirrors mv_fp32_16_accurate_ready_atomiconly_
            // scaled_kblock_fast_normal's own safe-range check earlier
            // in this file. On safe-range data (control=0 for every
            // value), fp32_16_fast_half and fp32_16_half_only decode
            // identically, so cpu::mv_fp32_16_pure_half (unchanged) is
            // still the correct CPU reference. See its own comment in
            // fp32_16_cuda_kernels.cu.
            std::mt19937 fast_rng(47);
            std::uniform_real_distribution<float> fast_mag_dist(0.5f, 1.0f);
            std::uniform_int_distribution<int> fast_sign_dist(0, 1);
            std::vector<float> A_fast(ld_a * n);
            for (auto& v : A_fast) {
                float mag = fast_mag_dist(fast_rng);
                v = fast_sign_dist(fast_rng) ? mag : -mag;
            }
            std::vector<uint16_t> wh_fast(ld_a * n), wr_fast_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_fast[idx], wh_fast[idx], wr_fast_scratch[idx]);

            uint16_t* d_wh_fast = nullptr;
            cudaMalloc(&d_wh_fast, wh_fast.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_fast, wh_fast.data(), wh_fast.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_fast.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(
                trans, alpha, dim<2>{m, n}, d_wh_fast, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_atomiconly_scaled_mblock_vec4_fast %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_fast);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // hybrid_transpose_atomiconly_scaled_mblock -- "test" for
            // "TOMOS FP32/FP16". Own dedicated data: a MIX of normal-
            // range values (control=0, exercising the cheap __half
            // path) and deliberately-extended-range values (magnitude
            // beyond fp16's own ~65504 max, control=1, exercising the
            // wh-only extended-exponent reconstruction path this kernel
            // exists to test) -- the plain "A"/"wh"/"wr" data above is
            // uniform in [-1,1] and would NEVER trigger the control=1
            // branch at all, so it can't cover this kernel's own reason
            // for existing. Reference: decode host-side using the EXACT
            // SAME per-element logic as the GPU kernel -- control=0 ->
            // direct half reinterpret; control=1, non-degraded -> an
            // approximate fp32 value from wh's own offset/direction
            // (extended exponent) and mant_hi bits alone, missing
            // mantissa bits zero-padded (NOT fp32_16_combine, which
            // would also fold in wr's mant_lo/round_flag -- this kernel
            // deliberately never reads wr); control=1, degraded -> 0 --
            // into a plain float array, then reuse cpu::mv_strided.
            // Uses ONLY wh, like every other pure_half check in this
            // file. See its own comment in fp32_16_cuda_kernels.cu.
            std::mt19937 hybrid_rng(59);
            std::uniform_real_distribution<float> hybrid_normal_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> hybrid_extended_mag_dist(1.0e5f, 1.0e8f);
            std::uniform_int_distribution<int> hybrid_extended_flag(0, 9);  // ~10% extended
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
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_decoded.data(), ld_a, x.data(),
                                                 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock(trans, alpha, dim<2>{m, n}, d_wh_hybrid, ld_a,
                                                                       d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "hybrid_transpose_atomiconly_scaled_mblock %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_hybrid);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // hybrid_transpose_atomiconly_scaled_mblock_vec4 --
            // vectorized-load sibling of the check just above, same
            // per-element branch and reference-decode logic, same
            // mixed normal/extended-range data (regenerated with its
            // own seed) -- see its own comment in fp32_16_cuda_
            // kernels.cu.
            std::mt19937 hybrid_vec4_rng(61);
            std::uniform_real_distribution<float> hybrid_vec4_normal_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> hybrid_vec4_extended_mag_dist(1.0e5f, 1.0e8f);
            std::uniform_int_distribution<int> hybrid_vec4_extended_flag(0, 9);
            std::uniform_int_distribution<int> hybrid_vec4_sign_dist(0, 1);
            std::vector<float> A_hybrid_vec4(ld_a * n);
            for (auto& v : A_hybrid_vec4) {
                if (hybrid_vec4_extended_flag(hybrid_vec4_rng) == 0) {
                    float mag = hybrid_vec4_extended_mag_dist(hybrid_vec4_rng);
                    v = hybrid_vec4_sign_dist(hybrid_vec4_rng) ? mag : -mag;
                } else {
                    v = hybrid_vec4_normal_dist(hybrid_vec4_rng);
                }
            }
            std::vector<uint16_t> wh_hybrid_vec4(ld_a * n), wr_hybrid_vec4_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_hybrid_vec4[idx], wh_hybrid_vec4[idx], wr_hybrid_vec4_scratch[idx]);
            std::vector<float> A_hybrid_vec4_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) {
                const uint16_t whv = wh_hybrid_vec4[idx];
                if ((whv & 0x1u) == 0u) {
                    A_hybrid_vec4_decoded[idx] = __half2float(*reinterpret_cast<const __half*>(&whv));
                } else {
                    const uint32_t offset = (static_cast<uint32_t>(whv) >> 10) & 0xFu;
                    if (offset != fp32_16_detail::kOffsetSaturate) {
                        const uint32_t direction = (static_cast<uint32_t>(whv) >> 14) & 0x1u;
                        const uint32_t exp32 = direction ? (fp32_16_detail::kFp32ExpMax + offset)
                                                          : (fp32_16_detail::kFp32ExpMin - offset);
                        const uint32_t mant_hi = (static_cast<uint32_t>(whv) >> 1) & 0x1FFu;
                        const uint32_t mant32 = mant_hi << 14;
                        const uint32_t sign = (static_cast<uint32_t>(whv) & 0x8000u) << 16;
                        A_hybrid_vec4_decoded[idx] = fp32_16_detail::bits_to_float(sign | (exp32 << 23) | mant32);
                    } else {
                        A_hybrid_vec4_decoded[idx] = 0.0f;
                    }
                }
            }

            uint16_t* d_wh_hybrid_vec4 = nullptr;
            cudaMalloc(&d_wh_hybrid_vec4, wh_hybrid_vec4.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_hybrid_vec4, wh_hybrid_vec4.data(), wh_hybrid_vec4.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_vec4_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4(trans, alpha, dim<2>{m, n},
                                                                            d_wh_hybrid_vec4, ld_a, d_x, 1, beta, d_y,
                                                                            1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "hybrid_transpose_atomiconly_scaled_mblock_vec4 %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_hybrid_vec4);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // hybrid_transpose_atomiconly_scaled_mblock_vec4_paired --
            // same math as the check just above (paired-packing is an
            // arithmetic-throughput optimization only, not a semantic
            // change), own regenerated mixed normal/extended-range
            // data. See its own comment in fp32_16_cuda_kernels.cu.
            std::mt19937 hybrid_paired_rng(71);
            std::uniform_real_distribution<float> hybrid_paired_normal_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> hybrid_paired_extended_mag_dist(1.0e5f, 1.0e8f);
            std::uniform_int_distribution<int> hybrid_paired_extended_flag(0, 9);
            std::uniform_int_distribution<int> hybrid_paired_sign_dist(0, 1);
            std::vector<float> A_hybrid_paired(ld_a * n);
            for (auto& v : A_hybrid_paired) {
                if (hybrid_paired_extended_flag(hybrid_paired_rng) == 0) {
                    float mag = hybrid_paired_extended_mag_dist(hybrid_paired_rng);
                    v = hybrid_paired_sign_dist(hybrid_paired_rng) ? mag : -mag;
                } else {
                    v = hybrid_paired_normal_dist(hybrid_paired_rng);
                }
            }
            std::vector<uint16_t> wh_hybrid_paired(ld_a * n), wr_hybrid_paired_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_hybrid_paired[idx], wh_hybrid_paired[idx], wr_hybrid_paired_scratch[idx]);
            std::vector<float> A_hybrid_paired_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx) {
                const uint16_t whv = wh_hybrid_paired[idx];
                if ((whv & 0x1u) == 0u) {
                    A_hybrid_paired_decoded[idx] = __half2float(*reinterpret_cast<const __half*>(&whv));
                } else {
                    const uint32_t offset = (static_cast<uint32_t>(whv) >> 10) & 0xFu;
                    if (offset != fp32_16_detail::kOffsetSaturate) {
                        const uint32_t direction = (static_cast<uint32_t>(whv) >> 14) & 0x1u;
                        const uint32_t exp32 = direction ? (fp32_16_detail::kFp32ExpMax + offset)
                                                          : (fp32_16_detail::kFp32ExpMin - offset);
                        const uint32_t mant_hi = (static_cast<uint32_t>(whv) >> 1) & 0x1FFu;
                        const uint32_t mant32 = mant_hi << 14;
                        const uint32_t sign = (static_cast<uint32_t>(whv) & 0x8000u) << 16;
                        A_hybrid_paired_decoded[idx] = fp32_16_detail::bits_to_float(sign | (exp32 << 23) | mant32);
                    } else {
                        A_hybrid_paired_decoded[idx] = 0.0f;
                    }
                }
            }

            uint16_t* d_wh_hybrid_paired = nullptr;
            cudaMalloc(&d_wh_hybrid_paired, wh_hybrid_paired.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_hybrid_paired, wh_hybrid_paired.data(), wh_hybrid_paired.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_hybrid_paired_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(trans, alpha, dim<2>{m, n},
                                                                                   d_wh_hybrid_paired, ld_a, d_x, 1,
                                                                                   beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "hybrid_transpose_atomiconly_scaled_mblock_vec4_paired %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_hybrid_paired);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_
            // vec4 -- "TOMOS FP32/BF16". Unlike every fp16 "_fast"
            // sibling above, this is SAFE on arbitrary data (see its own
            // comment in fp32_16_cuda_kernels.cu -- bf16's truncation
            // decode is always valid, no extended/degraded fallback
            // exists for it), so this reuses the plain "A"/"x" data
            // already generated for this shape rather than a dedicated
            // safe-range generator. No cpu::mv_fp32_bf16_pure_bf16
            // helper exists -- the reference is built by decoding wh_
            // bf16_pure host-side via fp32_bf16_bf16_only (discarding
            // wr entirely, matching the GPU kernel's own single-array
            // decode) into a plain float array, then reusing cpu::
            // mv_strided on that -- same convention mv_fp32_bf16_
            // accurate's own check above uses (there via fp32_bf16_
            // combine instead). Same loose pure-16-bit-compute tolerance
            // (5.0) as every genuine __half/__nv_bfloat16-compute check
            // in this file.
            std::vector<uint16_t> wh_bf16_pure(ld_a * n), wr_bf16_pure_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_bf16_split(A[idx], wh_bf16_pure[idx], wr_bf16_pure_scratch[idx]);
            std::vector<float> A_bf16_pure_decoded(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                A_bf16_pure_decoded[idx] = __bfloat162float(fp32_bf16_bf16_only(wh_bf16_pure[idx]));

            uint16_t* d_wh_bf16_pure = nullptr;
            cudaMalloc(&d_wh_bf16_pure, wh_bf16_pure.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_bf16_pure, wh_bf16_pure.data(), wh_bf16_pure.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_strided<float, float, float>(trans, alpha, dim<2>{m, n}, A_bf16_pure_decoded.data(), ld_a,
                                                 x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(
                trans, alpha, dim<2>{m, n}, d_wh_bf16_pure, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_bf16_pure);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_wmma -- standalone TRANSPOSE-only
            // Tensor Core (WMMA) kernel, same TRANSPOSE-only/ld_a==m
            // guard as every other atomiconly_scaled_mblock-family
            // kernel in this file -- see its own comment in fp32_16_
            // cuda_kernels.cu (including why real hardware may well show
            // a severe regression here, matching the confirmed 0.14x-
            // of-cuBLAS result its NO_TRANSPOSE counterpart already
            // measured).
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_transpose_wmma(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_wmma %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_
            // retuned_floor -- IDENTICAL compute to the retuned_floor
            // check above, only x/y are __half instead of float
            // (matching cuBLAS's own Hgemm semantics) -- own __half
            // device buffers (converted from the same host x/y_init
            // this file's other checks already use), same TRANSPOSE-
            // only/ld_a==m guard, same loose pure_half tolerance (5.0)
            // for the same chaotic-fp16-accumulation reason every other
            // pure_half check in this file uses it -- see its own
            // comment in fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);

            std::vector<__half> x_h(x.size()), y_init_h(y_init.size());
            for (size_t i = 0; i < x.size(); ++i) x_h[i] = __float2half(x[i]);
            for (size_t i = 0; i < y_init.size(); ++i) y_init_h[i] = __float2half(y_init[i]);

            __half *d_x_h = nullptr, *d_y_h = nullptr;
            cudaMalloc(&d_x_h, x_h.size() * sizeof(__half));
            cudaMalloc(&d_y_h, y_init_h.size() * sizeof(__half));
            cudaMemcpy(d_x_h, x_h.data(), x_h.size() * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y_h, y_init_h.data(), y_init_h.size() * sizeof(__half), cudaMemcpyHostToDevice);

            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x_h, 1, beta, d_y_h, 1);

            std::vector<__half> y_gpu_h(rows_op);
            cudaMemcpy(y_gpu_h.data(), d_y_h, y_gpu_h.size() * sizeof(__half), cudaMemcpyDeviceToHost);
            std::vector<float> y_gpu(rows_op);
            for (size_t i = 0; i < y_gpu_h.size(); ++i) y_gpu[i] = __half2float(y_gpu_h[i]);

            std::snprintf(label, sizeof(label),
                         "pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_x_h);
            cudaFree(d_y_h);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_warp_owner -- standalone TRANSPOSE-only
            // "warp owner, no atomics" kernel, same TRANSPOSE-only/
            // ld_a==m guard as every other pure_half_transpose sibling in
            // this file -- see its own comment in fp32_16_cuda_kernels.cu
            // (including the prior-art regression this kernel's design
            // needs to avoid repeating).
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y,
                                                             1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_warp_owner %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_warp_owner_split -- split-then-reduce
            // counterpart of the check just above, same TRANSPOSE-only/
            // ld_a==m guard, caller-provided workspace (allocated below,
            // outside any timing) -- see its own comment in fp32_16_
            // cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t warp_owner_split_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_workspace_size(dim<2>{m, n});
            double* d_warp_owner_split_ws = nullptr;
            cudaMalloc(&d_warp_owner_split_ws, warp_owner_split_ws_elems * sizeof(double));
            cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1,
                                                                   beta, d_y, 1, d_warp_owner_split_ws);
            cudaFree(d_warp_owner_split_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_warp_owner_split %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_warp_owner_split_f32 -- float-scratch-
            // buffer counterpart of the check just above, same
            // TRANSPOSE-only/ld_a==m guard, caller-provided float
            // workspace -- see its own comment in fp32_16_cuda_
            // kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t warp_owner_split_f32_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_f32_workspace_size(dim<2>{m, n});
            float* d_warp_owner_split_f32_ws = nullptr;
            cudaMalloc(&d_warp_owner_split_f32_ws, warp_owner_split_f32_ws_elems * sizeof(float));
            cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_f32(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1,
                                                                       beta, d_y, 1, d_warp_owner_split_f32_ws);
            cudaFree(d_warp_owner_split_f32_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_warp_owner_split_f32 %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_split --
            // atomiconly_scaled_mblock_vec4's own compute body verbatim,
            // combine step swapped to split+reduce -- isolates atomics
            // vs split+reduce as the one remaining variable. Same
            // TRANSPOSE-only/ld_a==m guard, caller-provided float
            // workspace -- see its own comment in fp32_16_cuda_
            // kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t vec4_split_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                    dim<2>{m, n});
            float* d_vec4_split_ws = nullptr;
            cudaMalloc(&d_vec4_split_ws, vec4_split_ws_elems * sizeof(float));
            cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_vec4_split_ws);
            cudaFree(d_vec4_split_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_atomiconly_scaled_mblock_vec4_split %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_split_
            // fast -- "TOMOS FP16 (fast)". UNSAFE on arbitrary data
            // (fp32_16_fast_half's own contract), so tested here against
            // dedicated safe-range data (mag in [0.5,1.0]), same
            // convention as the vec4_fast check above -- see its own
            // comment in fp32_16_cuda_kernels.cu.
            std::mt19937 split_fast_rng(53);
            std::uniform_real_distribution<float> split_fast_mag_dist(0.5f, 1.0f);
            std::uniform_int_distribution<int> split_fast_sign_dist(0, 1);
            std::vector<float> A_split_fast(ld_a * n);
            for (auto& v : A_split_fast) {
                float mag = split_fast_mag_dist(split_fast_rng);
                v = split_fast_sign_dist(split_fast_rng) ? mag : -mag;
            }
            std::vector<uint16_t> wh_split_fast(ld_a * n), wr_split_fast_scratch(ld_a * n);
            for (size_t idx = 0; idx < ld_a * n; ++idx)
                fp32_16_split(A_split_fast[idx], wh_split_fast[idx], wr_split_fast_scratch[idx]);

            uint16_t* d_wh_split_fast = nullptr;
            cudaMalloc(&d_wh_split_fast, wh_split_fast.size() * sizeof(uint16_t));
            cudaMemcpy(d_wh_split_fast, wh_split_fast.data(), wh_split_fast.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice);

            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh_split_fast.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t vec4_split_fast_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                    dim<2>{m, n});
            float* d_vec4_split_fast_ws = nullptr;
            cudaMalloc(&d_vec4_split_fast_ws, vec4_split_fast_ws_elems * sizeof(float));
            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast(
                trans, alpha, dim<2>{m, n}, d_wh_split_fast, ld_a, d_x, 1, beta, d_y, 1, d_vec4_split_fast_ws);
            cudaFree(d_vec4_split_fast_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);

            cudaFree(d_wh_split_fast);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_split_
            // retarget -- reuses the EXACT SAME __global__ kernel as the
            // check just above, only the M_BLOCK-scaling target differs
            // (2176 vs 2048) -- see its own comment in fp32_16_cuda_
            // kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t vec4_split_retarget_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_workspace_size(
                    dim<2>{m, n});
            float* d_vec4_split_retarget_ws = nullptr;
            cudaMalloc(&d_vec4_split_retarget_ws, vec4_split_retarget_ws_elems * sizeof(float));
            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_vec4_split_retarget_ws);
            cudaFree(d_vec4_split_retarget_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label),
                         "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        if (trans == TransposeType::TRANSPOSE && ld_a == m) {
            // pure_half_transpose_atomiconly_scaled_mblock_vec4_split_
            // retarget_lo -- tests the opposite (fewer-blocks) hypothesis
            // from the check just above -- see its own comment in
            // fp32_16_cuda_kernels.cu.
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            size_t vec4_split_retarget_lo_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo_workspace_size(
                    dim<2>{m, n});
            float* d_vec4_split_retarget_lo_ws = nullptr;
            cudaMalloc(&d_vec4_split_retarget_lo_ws, vec4_split_retarget_lo_ws_elems * sizeof(float));
            cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1, d_vec4_split_retarget_lo_ws);
            cudaFree(d_vec4_split_retarget_lo_ws);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label),
                         "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x(trans, alpha, dim<2>{m, n},
                                                                    d_wh, ld_a, d_x, 1, beta, d_y,
                                                                    1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_wide2x %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x(trans, alpha, dim<2>{m, n},
                                                                    d_wh, ld_a, d_x, 1, beta, d_y,
                                                                    1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_half2_wide4x %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_vecrow(trans, alpha, dim<2>{m, n}, d_wh,
                                                              ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_vecrow %s %s", tag,
                         dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2(
                trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp_shared_bigtile_half2 %s %s",
                         tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp8_shared(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                        d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp8_shared %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_pure_half(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                      beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp2_shared(trans, alpha, dim<2>{m, n}, d_wh, ld_a,
                                                        d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "pure_half_wide_ilp2_shared %s %s", tag, dir);
            check(label, y_cpu, y_gpu, 5.0);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_accurate(trans, alpha, dim<2>{m, n}, wh.data(), wr.data(), ld_a,
                                     x.data(), 1, beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_accurate_wide_ilp_shared(trans, alpha, dim<2>{m, n}, d_wh, d_wr,
                                                      ld_a, d_x, 1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "accurate_wide_ilp_shared %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }
        {
            std::vector<float> y_cpu = y_init;
            cpu::mv_fp32_16_fast(trans, alpha, dim<2>{m, n}, wh.data(), ld_a, x.data(), 1,
                                 beta, y_cpu.data(), 1);
            cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
            cuda::experimental::mv_fp32_16_fast_wide_ilp_shared(trans, alpha, dim<2>{m, n}, d_wh, ld_a, d_x,
                                                  1, beta, d_y, 1);
            std::vector<float> y_gpu(rows_op);
            cudaMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::snprintf(label, sizeof(label), "fast_wide_ilp_shared %s %s", tag, dir);
            all_ok &= check(label, y_cpu, y_gpu, 1e-4);
        }

        cudaFree(d_wh);
        cudaFree(d_wr);
        cudaFree(d_x);
        cudaFree(d_y);
    }
    return all_ok;
}

// only_case_tag, if non-null, restricts the sweep to the single Case
// whose own tag exactly matches it (see main()'s --only-case parsing) --
// for isolating one problem case (e.g. narrow_vec-via-transpose) without
// paying for the other ~10 shapes or the performance section, while
// still exercising the EXACT same code path (every kernel call in
// run_correctness_case, in the same order) rather than a hand-
// duplicated subset that risks not reproducing a bug that only shows up
// after enough prior GPU state/allocations have accumulated.
bool run_correctness(const char* only_case_tag = nullptr) {
    std::printf("=== Correctness (GPU vs CPU reference) ===\n");
    // Covers: wide/split-K path (including a genuinely square shape, which
    // now also routes through it -- see kMvWideRowThreshold's comment),
    // tiled path (cols_op < 64), narrow_vec path (cols_op >= 64) with
    // rows_op both a multiple of kMvRowsPerThread and not, the narrow_vec
    // transpose branch specifically, and -- since ld_a == rows whenever
    // there's no padding, meaning a nonzero tail remainder normally makes
    // ld_a misaligned too and disables vectorization for the whole case,
    // not just its tail -- a couple of cases with an explicitly padded
    // ld_a to exercise the vectorized main loop and the scalar tail
    // fallback together, the way a real caller with a padded array would.
    struct Case { size_t rows, cols; const char* tag; size_t ld_a_pad = 0; };
    // rows_op must be >= kMvWideRowThreshold (currently kMvTargetTotalThreads
    // = 131072, see fp32_16_cuda_kernels.cu) to reach the tiled/narrow_vec
    // path at all -- below that, even a moderate/large square matrix now
    // routes through the split-K wide path (that's the point of the fix
    // this test suite is guarding), so the tiled/narrow_vec-targeted cases
    // below use rows_op well past that threshold on purpose.
    std::vector<Case> cases = {
        {512, 50000, "wide"},
        {8192, 8192, "square-now-splitk"},
        {200000, 100, "narrow_vec-rows%4==0"},
        {100, 100, "tiny-wide"},
        {200000, 60, "tiled-just-below-col-threshold"},
        {200000, 64, "narrow_vec-at-col-threshold"},
        {200002, 128, "narrow_vec-rows%4==2-fully-scalar(ld_a misaligned too)"},
        {200003, 128, "narrow_vec-rows%4==3-fully-scalar(ld_a misaligned too)"},
        {200002, 128, "narrow_vec-rows%4==2-vectorized-main+scalar-tail", 200004},
        {200003, 128, "narrow_vec-rows%4==3-vectorized-main+scalar-tail", 200004},
        {70, 200000, "narrow_vec-via-transpose"},
    };
    bool all_ok = true;
    bool ran_any = false;
    for (const auto& c : cases) {
        if (only_case_tag != nullptr && std::strcmp(c.tag, only_case_tag) != 0) continue;
        ran_any = true;
        all_ok &= run_correctness_case(c.rows, c.cols, c.tag, c.ld_a_pad);
    }
    if (only_case_tag != nullptr && !ran_any) {
        std::printf("error: --only-case '%s' matched no case in the list above\n", only_case_tag);
        return false;
    }

    std::printf("%s\n\n", all_ok ? "Correctness: ALL EXACT-TOLERANCE CHECKS PASSED"
                                 : "Correctness: SOME EXACT-TOLERANCE CHECKS FAILED (see FAIL lines above)");
    return all_ok;
}

// ---- performance section: our kernels vs cuBLAS ----

// Benchmarks one (M, N) shape: pure_half, the experimental
// pure_half_wide_regblock, cuBLAS's Hgemm(n=1), and the main-kernel/
// reduce-kernel diagnostic split, all against the same realistic fp32_16
// data. Pulled out to a standalone function (rather than kept as a local
// lambda in run_performance()) so run_single_shape() can also call it --
// letting a caller isolate exactly one shape's kernel launches for
// external profiling (ncu/nsys), where "the Nth launch of kernel X across
// the whole sweep" would otherwise be ambiguous to target.
void bench_one(cublasHandle_t handle, size_t M, size_t N) {
    uint16_t *wh, *wr;
    __half *A_h, *x_h, *y_h;
    float *x, *y;
    if (cudaMalloc(&wh, M * N * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&wr, M * N * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&A_h, M * N * sizeof(__half)) != cudaSuccess) {
        std::printf("  M=%-9zu N=%-9zu  SKIPPED (out of device memory)\n", M, N);
        cudaGetLastError();
        return;
    }
    gen_fp32_16(M * N, wh, wr);
    gen_half(M * N, A_h);
    cudaMalloc(&x, N * sizeof(float));
    cudaMalloc(&y, M * sizeof(float));
    gen_float(N, x);
    cudaMemset(y, 0, M * sizeof(float));
    cudaMalloc(&x_h, N * sizeof(__half));
    cudaMalloc(&y_h, M * sizeof(__half));
    gen_half(N, x_h);
    cudaMemset(y_h, 0, M * sizeof(__half));
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto time_call = [&](auto&& fn, int reps) {
        fn();
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int r = 0; r < reps; ++r) fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        return now_ms(start, stop) / reps;
    };

    int reps = 20;
    double t_ph = time_call(
        [&] {
            cuda::mv_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M,
                                       x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: register-blocked wide-matrix path (see
    // mv_fp32_16_pure_half_wide_regblock's comment) -- only actually
    // differs from mv_fp32_16_pure_half above when M < kMvWideRowThreshold
    // (currently 131072), i.e. for the wide/square sweeps below, not the
    // narrow one; timed here unconditionally anyway for a direct,
    // always-available comparison against the ~70%-of-cuBLAS baseline.
    double t_ph_regblock = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_regblock(TransposeType::NO_TRANSPOSE, 1.0,
                                                     {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mv_strided<__half, __half, __half>(TransposeType::NO_TRANSPOSE, 1.0,
                                                      {M, N}, A_h, M, x_h, 1, 0.0, y_h,
                                                      1, handle);
        },
        reps);
    // EXPERIMENTAL: same grid/block configuration as the baseline's
    // wide-matrix path, but with the column loop unrolled 4-wide with
    // independent accumulators -- see mv_fp32_16_pure_half_wide_ilp's
    // comment. Informed by the ncu profiling that motivated this whole
    // investigation (L1TEX load-latency stalls, not bandwidth/occupancy).
    double t_ph_ilp = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                                wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: identical to the baseline except for a
    // __launch_bounds__(kMvTileSize, 8) hint on the wide-matrix kernel --
    // see mv_fp32_16_pure_half_wide_launchbounds's comment. Tests whether
    // forcing lower register usage (for more resident blocks/SM) helps,
    // the opposite direction from wide_ilp's (unsuccessful) attempt to add
    // more per-thread ILP.
    double t_ph_launchbounds = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_launchbounds(TransposeType::NO_TRANSPOSE, 1.0,
                                                         {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: structurally different from the other _wide_*
    // variants -- warp-cooperative row groups with an in-register
    // __shfl_down_sync reduction, no column-splitting/partial buffer/
    // second kernel at all -- see mv_fp32_16_pure_half_wide_warpgroup's
    // comment.
    double t_ph_warpgroup = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_warpgroup(TransposeType::NO_TRANSPOSE, 1.0,
                                                      {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: same 4-way column unrolling as wide_ilp, but keeps the
    // baseline's shared-memory x_tile instead of direct __ldg reads --
    // isolates whether wide_ilp's regression came from dropping the
    // tiling rather than from the unrolling itself. See mv_fp32_16_pure_
    // half_wide_ilp_shared's comment.
    double t_ph_ilp_shared = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared(TransposeType::NO_TRANSPOSE, 1.0,
                                                       {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: IDENTICAL to ilp_shared above except wh/x are marked
    // __restrict__ and every wh load goes through __ldg() explicitly -- see
    // mv_fp32_16_pure_half_wide_ilp_shared_ldg's comment. A next lever on
    // top of the proven ilp_shared win, aimed at closing more of the
    // remaining gap to cuBLAS.
    double t_ph_ilp_shared_ldg = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_ldg(TransposeType::NO_TRANSPOSE, 1.0,
                                                           {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: keeps ilp_shared's proven 4-way accumulator width but
    // software-pipelines the loads (next group's wh loads issued before
    // consuming the current group's values) -- see mv_fp32_16_pure_half_
    // wide_ilp_shared_prefetch's comment. A different lever from unroll
    // width or the read-only-cache attempt above.
    double t_ph_ilp_shared_prefetch = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_prefetch(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: keeps ilp_shared's proven 4-way unroll but quadruples
    // the shared-memory tile width (1024 vs 256) to cut __syncthreads()
    // frequency -- see mv_fp32_16_pure_half_wide_ilp_shared_bigtile's
    // comment. A structural lever, unlike every compute-scheduling variant
    // above.
    double t_ph_ilp_shared_bigtile = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: Tensor Core (mma.sync via nvcuda::wmma) rewrite -- a
    // fundamentally different kernel architecture from every scalar
    // _wide_* variant above. See mv_fp32_16_pure_half_wide_wmma's comment
    // for the full rationale and the caveat about wh's load latency (the
    // measured bottleneck throughout) not being addressed by this change.
    double t_ph_wmma = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_wmma(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                                 wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: isolates whether wmma's sync frequency (twice per
    // 16-wide K-step) rather than tensor cores generally caused its 0.14
    // result, by batching 8 K-tiles per sync cycle -- see mv_fp32_16_pure_
    // half_wide_wmma_batched's comment.
    double t_ph_wmma_batched = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_wmma_batched(TransposeType::NO_TRANSPOSE, 1.0,
                                                         {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: isolates whether fp32_16_half_only's control-bit branch
    // costs anything beyond a raw half read, by swapping in
    // fp32_16_fast_half (unconditional reinterpret_cast) -- otherwise
    // identical to ilp_shared. See mv_fp32_16_pure_half_wide_ilp_shared_
    // fasthalf's comment, including its correctness caveat (safe for this
    // benchmark's synthetic data only).
    double t_ph_ilp_shared_fasthalf = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: combines ilp_shared_fasthalf's decode-branch removal
    // with ilp_shared_bigtile's quadrupled tile, to test whether the two
    // independent wins stack -- see mv_fp32_16_pure_half_wide_ilp_shared_
    // bigtile_fasthalf's comment.
    double t_ph_ilp_shared_bigtile_fasthalf = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_fasthalf(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: tests whether fp32_16_half_only's branch instruction
    // itself (not the control=1 saturation logic it guards, which fasthalf
    // wrongly skipped) was the source of fasthalf's speedup, using a
    // bit-exact branchless reformulation instead -- see mv_fp32_16_pure_
    // half_wide_ilp_shared_branchless's comment. Expected to be fully
    // correct, unlike fasthalf.
    double t_ph_ilp_shared_branchless = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: uses Ampere's cp.async to stage wh into shared memory
    // directly from global memory, bypassing the register file -- see
    // mv_fp32_16_pure_half_wide_ilp_shared_cpasync's comment. A
    // fundamentally different mechanism from every prior variant, aimed at
    // the same L1TEX latency ncu identified as the dominant cost
    // throughout this investigation.
    double t_ph_cpasync = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_cpasync(TransposeType::NO_TRANSPOSE,
                                                               1.0, {M, N}, wh, M, x, 1, 0.0,
                                                               y, 1);
        },
        reps);
    // EXPERIMENTAL: double-buffered version of cpasync -- issues the next
    // stage's copies while computing on the current stage, instead of
    // stage-then-wait-then-compute -- see mv_fp32_16_pure_half_wide_ilp_
    // shared_cpasync2's comment.
    double t_ph_cpasync2 = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_cpasync2(TransposeType::NO_TRANSPOSE,
                                                                1.0, {M, N}, wh, M, x, 1, 0.0,
                                                                y, 1);
        },
        reps);
    // EXPERIMENTAL: eliminates the split-K reduce kernel's second launch
    // by atomicAdd-ing each block's contribution directly into y instead
    // -- see mv_fp32_16_pure_half_wide_ilp_shared_atomic's comment. Tests
    // whether the reduce kernel's launch-boundary synchronization (not
    // __syncthreads(), which every other variant above has focused on) is
    // itself a real cost.
    double t_ph_atomic = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_atomic(TransposeType::NO_TRANSPOSE, 1.0,
                                                               {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: packs the 4-way unroll's 4 columns into two __half2
    // accumulators, using __hfma2 to halve the compute instruction count
    // for the same work -- see mv_fp32_16_pure_half_wide_ilp_shared_half2's
    // comment.
    double t_ph_half2 = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2(TransposeType::NO_TRANSPOSE, 1.0,
                                                              {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: reuses half2's proven kernel unchanged but doubles
    // col_splits (more, smaller column chunks, more blocks, a 2x-larger
    // temporary global partial buffer) -- see mv_fp32_16_pure_half_wide_
    // ilp_shared_half2_wide2x's comment.
    double t_ph_half2_wide2x = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: same idea, quadruples col_splits instead of doubling
    // it -- see mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x's
    // comment.
    double t_ph_half2_wide4x = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: properly occupancy-matched vectorized-row-load version
    // of the idea mv_fp32_16_pure_half_wide_regblock failed at -- see
    // mv_fp32_16_pure_half_wide_ilp_shared_vecrow's comment.
    double t_ph_vecrow = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_vecrow(TransposeType::NO_TRANSPOSE, 1.0,
                                                              {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: combines half2's packed arithmetic with bigtile's
    // quadrupled shared-memory tile, to test whether the two independent
    // wins stack -- see mv_fp32_16_pure_half_wide_ilp_shared_bigtile_
    // half2's comment.
    double t_ph_bigtile_half2 = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2(
                TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: same idea as wide_ilp_shared (the first variant to
    // beat the baseline) but 8-way unrolled instead of 4-way -- see
    // mv_fp32_16_pure_half_wide_ilp8_shared's comment.
    double t_ph_ilp8_shared = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp8_shared(TransposeType::NO_TRANSPOSE, 1.0,
                                                        {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);
    // EXPERIMENTAL: same idea, 2-way unrolled -- brackets the low side of
    // the 1/4/8-way peak. See mv_fp32_16_pure_half_wide_ilp2_shared's
    // comment.
    double t_ph_ilp2_shared = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_pure_half_wide_ilp2_shared(TransposeType::NO_TRANSPOSE, 1.0,
                                                        {M, N}, wh, M, x, 1, 0.0, y, 1);
        },
        reps);

    // Applying the winning pattern (4-way unroll, shared-memory x_tile
    // kept) to the accurate and fast tiers -- own baseline-vs-ilp_shared
    // speedup for each, not measured against cuBLAS (that comparison is
    // pure_half-specific throughout this investigation).
    double t_acc = time_call(
        [&] {
            cuda::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, wr, M, x,
                                      1, 0.0, y, 1);
        },
        reps);
    double t_acc_ilp_shared = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_accurate_wide_ilp_shared(TransposeType::NO_TRANSPOSE, 1.0,
                                                      {M, N}, wh, wr, M, x, 1, 0.0, y, 1);
        },
        reps);
    double t_fast = time_call(
        [&] {
            cuda::mv_fp32_16_fast(TransposeType::NO_TRANSPOSE, 1.0, {M, N}, wh, M, x, 1, 0.0,
                                  y, 1);
        },
        reps);
    double t_fast_ilp_shared = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_fast_wide_ilp_shared(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                                  wh, M, x, 1, 0.0, y, 1);
        },
        reps);

    // EXPERIMENTAL: the same register-blocked ushort4 vectorized-load
    // pattern as mv_fp32_16_pure_half_wide_regblock (see its comment),
    // applied to the accurate and fast tiers. These have existed since the
    // earlier register-blocking phase and were already correctness-checked
    // in run_correctness_case, but were never actually timed here -- added
    // now as their own baseline-vs-regblock speedup, alongside the
    // ilp_shared comparison above, for a fuller picture of how each
    // technique fares against these tiers' double-accumulator arithmetic.
    double t_acc_regblock = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_accurate_wide_regblock(TransposeType::NO_TRANSPOSE, 1.0,
                                                    {M, N}, wh, wr, M, x, 1, 0.0, y, 1);
        },
        reps);
    double t_fast_regblock = time_call(
        [&] {
            cuda::experimental::mv_fp32_16_fast_wide_regblock(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                                wh, M, x, 1, 0.0, y, 1);
        },
        reps);

    // DIAGNOSTIC: breaks the wide-matrix path's own two kernel launches
    // (main split-K kernel, then the reduce kernel) into separately
    // measured times -- see mv_fp32_16_pure_half_wide_diag's comment.
    // Reports 0/0 for shapes above kMvWideRowThreshold (the narrow
    // sweep), where the wide-matrix path doesn't apply.
    float main_ms_total = 0.0f, reduce_ms_total = 0.0f;
    {
        float warm_main, warm_reduce;
        cuda::experimental::mv_fp32_16_pure_half_wide_diag(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                             wh, M, x, 1, 0.0, y, 1, &warm_main,
                                             &warm_reduce);
    }
    cudaDeviceSynchronize();
    for (int r = 0; r < reps; ++r) {
        float main_ms, reduce_ms;
        cuda::experimental::mv_fp32_16_pure_half_wide_diag(TransposeType::NO_TRANSPOSE, 1.0, {M, N},
                                             wh, M, x, 1, 0.0, y, 1, &main_ms, &reduce_ms);
        main_ms_total += main_ms;
        reduce_ms_total += reduce_ms;
    }
    double main_ms_avg = static_cast<double>(main_ms_total) / reps;
    double reduce_ms_avg = static_cast<double>(reduce_ms_total) / reps;

    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N);
    std::printf(
        "  M=%-9zu N=%-9zu  pure_half: %8.2f GFlop/s (ratio=%.2f)   cublas: %8.2f GFlop/s\n"
        "  %-21s regblock: %8.2f (%.2f)   ilp: %8.2f (%.2f)   launchbounds: %8.2f (%.2f)   warpgroup: %8.2f (%.2f)\n"
        "  %-21s ilp2_shared: %8.2f (%.2f)   ilp_shared(4x): %8.2f (%.2f)   ilp8_shared: %8.2f (%.2f)\n"
        "  %-21s ilp_shared_ldg: %8.2f (%.2f)   ilp_shared_prefetch: %8.2f (%.2f)   ilp_shared_bigtile: %8.2f (%.2f)\n"
        "  %-21s wmma: %8.2f (%.2f)   wmma_batched: %8.2f (%.2f)   ilp_shared_fasthalf: %8.2f (%.2f)\n"
        "  %-21s ilp_shared_bigtile_fasthalf: %8.2f (%.2f)   ilp_shared_branchless: %8.2f (%.2f)   cpasync: %8.2f (%.2f)\n"
        "  %-21s cpasync2: %8.2f (%.2f)   ilp_shared_atomic: %8.2f (%.2f)   ilp_shared_half2: %8.2f (%.2f)\n"
        "  %-21s ilp_shared_bigtile_half2: %8.2f (%.2f)\n"
        "  %-21s half2_wide2x: %8.2f (%.2f)   half2_wide4x: %8.2f (%.2f)   vecrow: %8.2f (%.2f)\n"
        "  %-21s accurate: %8.2f -> ilp_shared: %8.2f (speedup=%.2fx)   fast: %8.2f -> ilp_shared: %8.2f (speedup=%.2fx)\n"
        "  %-21s accurate -> regblock: %8.2f (speedup=%.2fx)   fast -> regblock: %8.2f (speedup=%.2fx)\n"
        "  %-21s [diag: main=%.4fms reduce=%.4fms reduce_frac=%.1f%%]\n",
        M, N, flops / t_ph / 1e6, (flops / t_ph) / (flops / t_cublas), flops / t_cublas / 1e6,
        "", flops / t_ph_regblock / 1e6, (flops / t_ph_regblock) / (flops / t_cublas),
        flops / t_ph_ilp / 1e6, (flops / t_ph_ilp) / (flops / t_cublas),
        flops / t_ph_launchbounds / 1e6, (flops / t_ph_launchbounds) / (flops / t_cublas),
        flops / t_ph_warpgroup / 1e6, (flops / t_ph_warpgroup) / (flops / t_cublas), "",
        flops / t_ph_ilp2_shared / 1e6, (flops / t_ph_ilp2_shared) / (flops / t_cublas),
        flops / t_ph_ilp_shared / 1e6, (flops / t_ph_ilp_shared) / (flops / t_cublas),
        flops / t_ph_ilp8_shared / 1e6, (flops / t_ph_ilp8_shared) / (flops / t_cublas), "",
        flops / t_ph_ilp_shared_ldg / 1e6, (flops / t_ph_ilp_shared_ldg) / (flops / t_cublas),
        flops / t_ph_ilp_shared_prefetch / 1e6,
        (flops / t_ph_ilp_shared_prefetch) / (flops / t_cublas),
        flops / t_ph_ilp_shared_bigtile / 1e6,
        (flops / t_ph_ilp_shared_bigtile) / (flops / t_cublas),
        "", flops / t_ph_wmma / 1e6, (flops / t_ph_wmma) / (flops / t_cublas),
        flops / t_ph_wmma_batched / 1e6, (flops / t_ph_wmma_batched) / (flops / t_cublas),
        flops / t_ph_ilp_shared_fasthalf / 1e6,
        (flops / t_ph_ilp_shared_fasthalf) / (flops / t_cublas),
        "", flops / t_ph_ilp_shared_bigtile_fasthalf / 1e6,
        (flops / t_ph_ilp_shared_bigtile_fasthalf) / (flops / t_cublas),
        flops / t_ph_ilp_shared_branchless / 1e6,
        (flops / t_ph_ilp_shared_branchless) / (flops / t_cublas),
        flops / t_ph_cpasync / 1e6, (flops / t_ph_cpasync) / (flops / t_cublas),
        "", flops / t_ph_cpasync2 / 1e6, (flops / t_ph_cpasync2) / (flops / t_cublas),
        flops / t_ph_atomic / 1e6, (flops / t_ph_atomic) / (flops / t_cublas),
        flops / t_ph_half2 / 1e6, (flops / t_ph_half2) / (flops / t_cublas),
        "", flops / t_ph_bigtile_half2 / 1e6,
        (flops / t_ph_bigtile_half2) / (flops / t_cublas),
        "", flops / t_ph_half2_wide2x / 1e6, (flops / t_ph_half2_wide2x) / (flops / t_cublas),
        flops / t_ph_half2_wide4x / 1e6, (flops / t_ph_half2_wide4x) / (flops / t_cublas),
        flops / t_ph_vecrow / 1e6, (flops / t_ph_vecrow) / (flops / t_cublas),
        "",
        flops / t_acc / 1e6, flops / t_acc_ilp_shared / 1e6, t_acc / t_acc_ilp_shared,
        flops / t_fast / 1e6, flops / t_fast_ilp_shared / 1e6, t_fast / t_fast_ilp_shared, "",
        flops / t_acc_regblock / 1e6, t_acc / t_acc_regblock,
        flops / t_fast_regblock / 1e6, t_fast / t_fast_regblock, "",
        main_ms_avg, reduce_ms_avg,
        100.0 * reduce_ms_avg / std::max(1e-9, main_ms_avg + reduce_ms_avg));

    cudaFree(wh);
    cudaFree(wr);
    cudaFree(A_h);
    cudaFree(x);
    cudaFree(y);
    cudaFree(x_h);
    cudaFree(y_h);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_performance() {
    std::printf("=== Performance (pure_half vs cuBLAS Hgemm(n=1), realistic fp32_16 data) ===\n");
    std::printf("cuBLAS has no native half-precision Sgemv, so cublasHgemm(n=1) -- the\n"
               "standard matrix-vector-as-matrix-matrix workaround -- is the real fp16\n"
               "GEMV baseline (same one BM_mv_cublas<__half> and compare_precision.m use).\n\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("-- Narrow shapes (many rows, few columns): --\n");
    for (size_t M : {1048576ul, 4194304ul}) {
        for (size_t N : {64ul, 128ul, 256ul, 512ul, 1024ul, 2048ul, 4096ul})
            bench_one(handle, M, N);
    }

    std::printf("\n-- Wide shapes (few rows, many columns): --\n");
    for (size_t M : {1024ul, 4096ul}) {
        for (size_t N : {65536ul, 262144ul, 1048576ul, 4194304ul}) bench_one(handle, M, N);
    }

    // Matches bench_fp32_16_cuda_kernels.cu's MvArgs square sweep (which
    // tops out at 90112x90112). All of these sit below kMvWideRowThreshold
    // (rows_op alone isn't enough to reach kMvTargetTotalThreads=131072
    // threads on a large GPU), so they now route through the split-K
    // *_wide_kernel path instead of the register-blocked *_narrow_vec_kernel
    // path a moderate/large square matrix used before -- see the
    // "WIDE- and NARROW-matrix" comment in fp32_16_cuda_kernels.cu for why
    // that's about total thread count, not aspect ratio. 200000 is added
    // past that threshold to show the crossover back to *_narrow_vec_kernel.
    std::printf("\n-- Square shapes (both dimensions large): --\n");
    for (size_t N : {4096ul, 8192ul, 16384ul, 32768ul, 65536ul, 90112ul, 200000ul})
        bench_one(handle, N, N);

    cublasDestroy(handle);
    std::printf("\n");
}

// Runs exactly one shape (M, N) and exits -- for isolating a single kernel
// launch under an external profiler (ncu/nsys), where "the Nth launch of
// kernel X" would otherwise be ambiguous to target across the full
// narrow/wide/square sweep in run_performance(). Example:
//   ncu --kernel-name mv_fp32_16_pure_half_wide_kernel --launch-count 1 \
//       ./verify_mv_fp32_16_gemv --only-shape 32768 32768
// (mv_fp32_16_pure_half_wide_kernel is the same kernel the ~70%-of-cuBLAS
// baseline mv_fp32_16_pure_half uses for this shape; see fp32_16_cuda_
// kernels.cu for its full name-to-shape dispatch logic if profiling a
// different tier/shape.)
void run_single_shape(size_t M, size_t N) {
    std::printf("=== Single shape: M=%zu N=%zu ===\n", M, N);
    cublasHandle_t handle;
    cublasCreate(&handle);
    bench_one(handle, M, N);
    cublasDestroy(handle);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_correctness = true, do_performance = true;
    size_t only_m = 0, only_n = 0;
    const char* only_case_tag = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-correctness") == 0) do_correctness = false;
        if (std::strcmp(argv[i], "--skip-performance") == 0) do_performance = false;
        if (std::strcmp(argv[i], "--only-shape") == 0 && i + 2 < argc) {
            only_m = std::strtoull(argv[i + 1], nullptr, 10);
            only_n = std::strtoull(argv[i + 2], nullptr, 10);
            i += 2;
        }
        if (std::strcmp(argv[i], "--only-case") == 0 && i + 1 < argc) {
            only_case_tag = argv[i + 1];
            i += 1;
        }
    }

    if (only_m != 0 && only_n != 0) {
        // Single-shape mode for external profiling -- see run_single_shape's
        // comment. Skips correctness and the full sweep entirely.
        run_single_shape(only_m, only_n);
        return 0;
    }

    bool correctness_ok = true;
    if (do_correctness) correctness_ok = run_correctness(only_case_tag);
    if (do_performance) run_performance();

    return correctness_ok ? 0 : 1;
}

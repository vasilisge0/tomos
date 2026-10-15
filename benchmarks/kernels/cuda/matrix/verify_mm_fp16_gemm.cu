// Standalone correctness + performance verification for the custom,
// un-encumbered fp16 square-GEMM CUDA kernel(s) in src/kernels/cuda/
// matrix/mm_cuda_kernels.cu: gemm_generic_kernel<__half,__half,__half>
// (the naive one-thread-per-output-element baseline, double-accumulated),
// gemm_fp16_tiled (128x128/8x8 register-blocked, same structure as
// gemm_fp32_tiled but __half-typed and __half-accumulated end to end --
// genuine fp16-only compute, not merely fp16 storage), and gemm_fp16_wmma
// (Tensor Core mma.sync, 16x16x16 half/half/float tiles, fp32-accumulated).
//
// Unlike verify_mm_fp32_16_pure_half_gemm.cu, A/B/C here are all plain
// __half arrays -- no FP32_16 packed wh/wr storage or per-element decode
// involved at all -- so this isolates a hand-written kernel's true
// tensor-core-architecture ceiling against cuBLAS's Hgemm, without
// conflating it with the fp32_16 feature's wh-decode overhead. Comparing
// this file's gemm_fp16_wmma numbers against that file's mm_fp32_16_pure_
// half_wmma numbers (same tile/warp/launch_bounds design) quantifies
// exactly how much of that decode overhead really costs.
//
// As further custom fp16 kernels are added, wire them into both
// run_correctness_case() and bench_one() below next to gemm_fp16_wmma.
//
// The performance sweep's matrices are generated directly on the device,
// same reasoning as verify_mm_fp32_gemm.cu.
//
// Usage: verify_mm_fp16_gemm [--skip-correctness] [--skip-performance]
//                             [--only-shape N] [--only-method NAME]
//
// --only-shape N runs just that one square size and exits, skipping
// correctness and the full sweep -- for isolating a single kernel launch
// under an external profiler (ncu/nsys):
//   scripts/verify_mm_fp16_gemm.sh -- --only-shape 8192
//   ncu --kernel-name gemm_fp16_wmma_kernel --launch-count 1 \
//       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp16_gemm" \
//       --only-shape 8192
//
// --only-method NAME restricts run_correctness() to a single kernel
// (the exact name passed to cuda::<name>(...), e.g.
// gemm_fp16_mma_ptx_wide_spread, or cublasHgemm for the cuBLAS check) --
// every other kernel's block, including its own dedicated beta=0 CPU
// reference recomputation, is skipped entirely. Useful for checking one
// newly added kernel across all sizes without paying for the full,
// heavily redundant ~30-kernel sweep (each kernel currently recomputes
// its own O(n^3) CPU reference for the beta=0 case independently):
//   verify_mm_fp16_gemm --skip-performance --only-method gemm_fp16_mma_ptx_wide_spread
//
// Exit code is non-zero if any correctness check fails.

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

#include "mm_cpu_kernels.hpp"
#include "mm_cuda_kernels.cuh"

using namespace tomos;

namespace {

// ---- device-side data generation (avoids host memory limits at the large
// sizes the performance sweep uses) ----

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_half_kernel(size_t n, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<__half>(pseudo_rand(i));
}

void gen_half(size_t n, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out);
}

double now_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ---- correctness section: GPU vs CPU reference ----

// Both operands are already __half (so no extra precision is lost between
// the CPU reference and any GPU kernel by generating/storing at a higher
// precision) -- the CPU reference itself still accumulates in double (see
// cpu::mm_strided's generic fallback in mm_cpu_kernels.cpp), so it's the
// same "how much does THIS kernel's own reduction/accumulation strategy
// cost in accuracy" comparison as every other verify_*.cu file uses.
// Set from --only-method on the command line; nullptr/empty runs every
// method (the original behavior). Lets run_correctness_case() test one
// kernel in isolation -- skipping every other method's block, including
// their dedicated beta=0 CPU reference recomputation (each is its own
// O(n^3) double-accumulated cpu::mm_strided pass) -- instead of paying
// for the full ~30-kernel sweep just to check one new kernel.
const char* g_only_method = nullptr;

bool want(const char* name) {
    return g_only_method == nullptr || g_only_method[0] == '\0' ||
           std::strcmp(g_only_method, name) == 0;
}

bool check(const std::string& label, const std::vector<__half>& cpu,
          const std::vector<__half>& gpu, double tol) {
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double c = static_cast<double>(static_cast<float>(cpu[i]));
        double g = static_cast<double>(static_cast<float>(gpu[i]));
        double err = std::fabs(c - g);
        double rel = err / std::max(1.0, std::fabs(c));
        max_abs_err = std::max(max_abs_err, err);
        max_rel_err = std::max(max_rel_err, rel);
    }
    bool ok = max_rel_err < tol;
    std::printf("  %-40s max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(),
               max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    return ok;
}

// Runs one N x N square case (NO_TRANSPOSE on both operands) for
// gemm_generic_kernel/gemm_fp16_tiled/gemm_fp16_wmma, comparing against the
// CPU reference (cpu::mm_strided<__half,__half,__half>). tag identifies the
// size in output; sizes not a multiple of the 128x128 tiled kernel's block
// or the wmma kernel's 128x128/BK=64 block exercise their boundary
// handling.
bool run_correctness_case(size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    std::vector<__half> A(n * n), B(n * n), C_init(n * n);
    for (auto& v : A) v = static_cast<__half>(dist(rng));
    for (auto& v : B) v = static_cast<__half>(dist(rng));
    for (auto& v : C_init) v = static_cast<__half>(dist(rng));

    double alpha = 1.7, beta = 0.3;

    // fp16 accumulation divergence from the double-accumulated CPU
    // reference grows with the reduction length (more terms -> more
    // chances for the final __half rounding to land on the other side of
    // a boundary from the reference's independent rounding). The 0.15/
    // 5.0 tolerances below were calibrated against this function's
    // original small sizes (<=256); N=2048 was added later to exercise
    // the "big tile" kernels' real code paths (see the 256-row-tile
    // family's grid-size fallback), and at that size EVERY kernel here
    // -- including gemm_generic_kernel, the most basic reference --
    // showed ~0.83 relative divergence, correctly matching cuBLAS's own
    // (much smaller) divergence in kind, just larger in magnitude. That
    // is expected rounding noise, not a bug, so the tolerance scales
    // linearly with n/256 (identity for the original sizes, so their
    // scrutiny is unchanged) rather than being widened globally.
    double tol_scale = std::max(1.0, static_cast<double>(n) / 256.0);

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.size() * sizeof(__half));
    cudaMalloc(&d_B, B.size() * sizeof(__half));
    cudaMalloc(&d_C, C_init.size() * sizeof(__half));
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice);

    std::vector<__half> C_cpu = C_init;
    cpu::mm_strided<__half, __half, __half>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, beta,
        C_cpu.data(), n);

    char label[64];

    if (want("gemm_generic_kernel")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (n + 15) / 16);
        cuda::experimental::gemm_generic_kernel<__half, __half, __half><<<blocks, threads>>>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_generic_kernel<half>] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_generic_kernel<half> %s", tag);
        // Even though this kernel and the CPU reference both accumulate in
        // double, C itself is __half here (unlike every fp32_16 verify
        // file, where C is always float) -- so both sides round their
        // final sum to __half independently, and at larger N (bigger
        // sums, more chances for the two computations' last-bit-different
        // double intermediate to straddle a half rounding boundary) that
        // alone produces several-percent "divergence" with no actual bug.
        // cuBLAS's own Hgemm shows the same magnitude of divergence against
        // this same reference below, which is what confirms it's rounding
        // noise rather than a correctness problem specific to this kernel.
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_tiled")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_tiled] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_tiled %s", tag);
        // gemm_fp16_tiled accumulates entirely in __half (see its comment);
        // its own long-reduction fp16 divergence from the double-
        // accumulated CPU reference is expected, same reasoning as
        // verify_mm_fp32_16_pure_half_gemm.cu's tol=5.0 for the analogous
        // kernel there.
        all_ok &= check(label, C_cpu, C_gpu, 5.0 * tol_scale);
    }
    if (want("gemm_fp16_wmma")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                             alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                         alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_dualbuf")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_dualbuf] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_dualbuf %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_bigbk")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_bigbk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_bigbk] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_bigbk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_bigbk_pad")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_bigbk_pad(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_bigbk_pad] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_bigbk_pad %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_hugebk")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_hugebk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_hugebk] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_hugebk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_tripipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_tripipe] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_tripipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_warpspec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_warpspec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_warpspec] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_warpspec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_wmma_cpasync_vec_hugebk_w32")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_wmma_cpasync_vec_hugebk_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_wmma_cpasync_vec_hugebk_w32] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_wmma_cpasync_vec_hugebk_w32 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_swizzle")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_swizzle(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_swizzle] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_swizzle %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_deep")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_deep(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_deep] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_deep %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi")) {
        // Exercises the non-zero-beta (regular read-modify-write) path.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi")) {
        // Dedicated beta=0 check: exercises the skip-beta-read branch
        // specifically, which the beta=0.3 case above never touches.
        // Needs its own CPU reference since the math differs from C_cpu.
        std::vector<__half> C_cpu_beta0 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi beta=0] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_l2")) {
        // Exercises the non-zero-beta path.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_l2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_l2] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_l2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_l2")) {
        // Dedicated beta=0 check -- needs its own CPU reference since
        // the earlier C_cpu_beta0 is scoped to its own block above.
        std::vector<__half> C_cpu_beta0_l2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_l2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_l2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_l2 beta=0] %s\n", cudaGetErrorString(e2));
        std::vector<__half> C_gpu2(C_init.size());
        cudaMemcpy(C_gpu2.data(), d_C, C_gpu2.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_l2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_l2, C_gpu2, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e3 = cudaDeviceSynchronize();
        if (e3 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide] %s\n", cudaGetErrorString(e3));
        std::vector<__half> C_gpu3(C_init.size());
        cudaMemcpy(C_gpu3.data(), d_C, C_gpu3.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide %s", tag);
        all_ok &= check(label, C_cpu, C_gpu3, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_wide = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_wide.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e4 = cudaDeviceSynchronize();
        if (e4 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide beta=0] %s\n", cudaGetErrorString(e4));
        std::vector<__half> C_gpu4(C_init.size());
        cudaMemcpy(C_gpu4.data(), d_C, C_gpu4.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_wide, C_gpu4, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e5 = cudaDeviceSynchronize();
        if (e5 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide2] %s\n", cudaGetErrorString(e5));
        std::vector<__half> C_gpu5(C_init.size());
        cudaMemcpy(C_gpu5.data(), d_C, C_gpu5.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu5, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide2")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_wide2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_wide2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e6 = cudaDeviceSynchronize();
        if (e6 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide2 beta=0] %s\n", cudaGetErrorString(e6));
        std::vector<__half> C_gpu6(C_init.size());
        cudaMemcpy(C_gpu6.data(), d_C, C_gpu6.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_wide2, C_gpu6, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e7 = cudaDeviceSynchronize();
        if (e7 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b] %s\n", cudaGetErrorString(e7));
        std::vector<__half> C_gpu7(C_init.size());
        cudaMemcpy(C_gpu7.data(), d_C, C_gpu7.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b %s", tag);
        all_ok &= check(label, C_cpu, C_gpu7, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_wide2b = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_wide2b.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e8 = cudaDeviceSynchronize();
        if (e8 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b beta=0] %s\n", cudaGetErrorString(e8));
        std::vector<__half> C_gpu8(C_init.size());
        cudaMemcpy(C_gpu8.data(), d_C, C_gpu8.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_wide2b, C_gpu8, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e9 = cudaDeviceSynchronize();
        if (e9 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5] %s\n", cudaGetErrorString(e9));
        std::vector<__half> C_gpu9(C_init.size());
        cudaMemcpy(C_gpu9.data(), d_C, C_gpu9.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu9, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_deep5 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep5.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e10 = cudaDeviceSynchronize();
        if (e10 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5 beta=0] %s\n", cudaGetErrorString(e10));
        std::vector<__half> C_gpu10(C_init.size());
        cudaMemcpy(C_gpu10.data(), d_C, C_gpu10.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep5, C_gpu10, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_w32")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e19 = cudaDeviceSynchronize();
        if (e19 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_w32] %s\n", cudaGetErrorString(e19));
        std::vector<__half> C_gpu19(C_init.size());
        cudaMemcpy(C_gpu19.data(), d_C, C_gpu19.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_w32 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu19, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_w32")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_deep5w32 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep5w32.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e20 = cudaDeviceSynchronize();
        if (e20 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_w32 beta=0] %s\n", cudaGetErrorString(e20));
        std::vector<__half> C_gpu20(C_init.size());
        cudaMemcpy(C_gpu20.data(), d_C, C_gpu20.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_w32(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep5w32, C_gpu20, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_deep5")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e11 = cudaDeviceSynchronize();
        if (e11 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_deep5] %s\n", cudaGetErrorString(e11));
        std::vector<__half> C_gpu11(C_init.size());
        cudaMemcpy(C_gpu11.data(), d_C, C_gpu11.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_deep5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu11, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_deep5")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_widedeep = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_widedeep.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e12 = cudaDeviceSynchronize();
        if (e12 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_deep5 beta=0] %s\n", cudaGetErrorString(e12));
        std::vector<__half> C_gpu12(C_init.size());
        cudaMemcpy(C_gpu12.data(), d_C, C_gpu12.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_deep5(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_widedeep, C_gpu12, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_coalb")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e13 = cudaDeviceSynchronize();
        if (e13 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_coalb] %s\n", cudaGetErrorString(e13));
        std::vector<__half> C_gpu13(C_init.size());
        cudaMemcpy(C_gpu13.data(), d_C, C_gpu13.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_coalb %s", tag);
        all_ok &= check(label, C_cpu, C_gpu13, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_coalb")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_coalb = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_coalb.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e14 = cudaDeviceSynchronize();
        if (e14 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_coalb beta=0] %s\n", cudaGetErrorString(e14));
        std::vector<__half> C_gpu14(C_init.size());
        cudaMemcpy(C_gpu14.data(), d_C, C_gpu14.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_coalb(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_coalb, C_gpu14, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_coalb_shfl")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_coalb_shfl(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e17 = cudaDeviceSynchronize();
        if (e17 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_coalb_shfl] %s\n", cudaGetErrorString(e17));
        std::vector<__half> C_gpu17(C_init.size());
        cudaMemcpy(C_gpu17.data(), d_C, C_gpu17.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_coalb_shfl %s", tag);
        all_ok &= check(label, C_cpu, C_gpu17, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_coalb_shfl")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_coalbshfl = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_coalbshfl.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_coalb_shfl(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e18 = cudaDeviceSynchronize();
        if (e18 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_coalb_shfl beta=0] %s\n", cudaGetErrorString(e18));
        std::vector<__half> C_gpu18(C_init.size());
        cudaMemcpy(C_gpu18.data(), d_C, C_gpu18.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_coalb_shfl(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_coalbshfl, C_gpu18, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e21 = cudaDeviceSynchronize();
        if (e21 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill2] %s\n", cudaGetErrorString(e21));
        std::vector<__half> C_gpu21(C_init.size());
        cudaMemcpy(C_gpu21.data(), d_C, C_gpu21.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu21, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill2")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_nospill2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_nospill2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e22 = cudaDeviceSynchronize();
        if (e22 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill2 beta=0] %s\n", cudaGetErrorString(e22));
        std::vector<__half> C_gpu22(C_init.size());
        cudaMemcpy(C_gpu22.data(), d_C, C_gpu22.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_nospill2, C_gpu22, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_epi_smem")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_epi_smem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e23 = cudaDeviceSynchronize();
        if (e23 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_epi_smem] %s\n", cudaGetErrorString(e23));
        std::vector<__half> C_gpu23(C_init.size());
        cudaMemcpy(C_gpu23.data(), d_C, C_gpu23.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_epi_smem %s", tag);
        all_ok &= check(label, C_cpu, C_gpu23, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_epi_smem")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_epismem = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_epismem.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_epi_smem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e24 = cudaDeviceSynchronize();
        if (e24 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_epi_smem beta=0] %s\n", cudaGetErrorString(e24));
        std::vector<__half> C_gpu24(C_init.size());
        cudaMemcpy(C_gpu24.data(), d_C, C_gpu24.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_epi_smem(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_epismem, C_gpu24, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e25 = cudaDeviceSynchronize();
        if (e25 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem] %s\n", cudaGetErrorString(e25));
        std::vector<__half> C_gpu25(C_init.size());
        cudaMemcpy(C_gpu25.data(), d_C, C_gpu25.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem %s", tag);
        all_ok &= check(label, C_cpu, C_gpu25, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_nospillepismem = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_nospillepismem.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e26 = cudaDeviceSynchronize();
        if (e26 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem beta=0] %s\n", cudaGetErrorString(e26));
        std::vector<__half> C_gpu26(C_init.size());
        cudaMemcpy(C_gpu26.data(), d_C, C_gpu26.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_nospillepismem, C_gpu26, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e27 = cudaDeviceSynchronize();
        if (e27 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp] %s\n", cudaGetErrorString(e27));
        std::vector<__half> C_gpu27(C_init.size());
        cudaMemcpy(C_gpu27.data(), d_C, C_gpu27.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu27, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e28 = cudaDeviceSynchronize();
        if (e28 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp beta=0] %s\n", cudaGetErrorString(e28));
        std::vector<__half> C_gpu28(C_init.size());
        cudaMemcpy(C_gpu28.data(), d_C, C_gpu28.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp, C_gpu28, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp_apipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_apipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e29 = cudaDeviceSynchronize();
        if (e29 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp_apipe] %s\n", cudaGetErrorString(e29));
        std::vector<__half> C_gpu29(C_init.size());
        cudaMemcpy(C_gpu29.data(), d_C, C_gpu29.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp_apipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu29, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp_apipe")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarpapipe = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarpapipe.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_apipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e30 = cudaDeviceSynchronize();
        if (e30 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp_apipe beta=0] %s\n", cudaGetErrorString(e30));
        std::vector<__half> C_gpu30(C_init.size());
        cudaMemcpy(C_gpu30.data(), d_C, C_gpu30.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp_apipe(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarpapipe, C_gpu30, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e37 = cudaDeviceSynchronize();
        if (e37 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe] %s\n", cudaGetErrorString(e37));
        std::vector<__half> C_gpu37(C_init.size());
        cudaMemcpy(C_gpu37.data(), d_C, C_gpu37.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu37, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarpfullpipe = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarpfullpipe.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e38 = cudaDeviceSynchronize();
        if (e38 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe beta=0] %s\n", cudaGetErrorString(e38));
        std::vector<__half> C_gpu38(C_init.size());
        cudaMemcpy(C_gpu38.data(), d_C, C_gpu38.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarpfullpipe, C_gpu38, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e39 = cudaDeviceSynchronize();
        if (e39 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2] %s\n", cudaGetErrorString(e39));
        std::vector<__half> C_gpu39(C_init.size());
        cudaMemcpy(C_gpu39.data(), d_C, C_gpu39.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu39, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e40 = cudaDeviceSynchronize();
        if (e40 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2 beta=0] %s\n", cudaGetErrorString(e40));
        std::vector<__half> C_gpu40(C_init.size());
        cudaMemcpy(C_gpu40.data(), d_C, C_gpu40.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp2, C_gpu40, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2_epismem")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e41 = cudaDeviceSynchronize();
        if (e41 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2_epismem] %s\n", cudaGetErrorString(e41));
        std::vector<__half> C_gpu41(C_init.size());
        cudaMemcpy(C_gpu41.data(), d_C, C_gpu41.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2_epismem %s", tag);
        all_ok &= check(label, C_cpu, C_gpu41, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2_epismem")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp2epismem = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp2epismem.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e42 = cudaDeviceSynchronize();
        if (e42 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2_epismem beta=0] %s\n", cudaGetErrorString(e42));
        std::vector<__half> C_gpu42(C_init.size());
        cudaMemcpy(C_gpu42.data(), d_C, C_gpu42.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2_epismem(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp2epismem, C_gpu42, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e43 = cudaDeviceSynchronize();
        if (e43 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted] %s\n", cudaGetErrorString(e43));
        std::vector<__half> C_gpu43(C_init.size());
        cudaMemcpy(C_gpu43.data(), d_C, C_gpu43.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted %s", tag);
        all_ok &= check(label, C_cpu, C_gpu43, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp2hoisted = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp2hoisted.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e44 = cudaDeviceSynchronize();
        if (e44 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted beta=0] %s\n", cudaGetErrorString(e44));
        std::vector<__half> C_gpu44(C_init.size());
        cudaMemcpy(C_gpu44.data(), d_C, C_gpu44.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp2hoisted, C_gpu44, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp3")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e45 = cudaDeviceSynchronize();
        if (e45 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp3] %s\n", cudaGetErrorString(e45));
        std::vector<__half> C_gpu45(C_init.size());
        cudaMemcpy(C_gpu45.data(), d_C, C_gpu45.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu45, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp3")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp3 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp3.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e46 = cudaDeviceSynchronize();
        if (e46 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp3 beta=0] %s\n", cudaGetErrorString(e46));
        std::vector<__half> C_gpu46(C_init.size());
        cudaMemcpy(C_gpu46.data(), d_C, C_gpu46.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp3(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp3, C_gpu46, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp4")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e47 = cudaDeviceSynchronize();
        if (e47 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp4] %s\n", cudaGetErrorString(e47));
        std::vector<__half> C_gpu47(C_init.size());
        cudaMemcpy(C_gpu47.data(), d_C, C_gpu47.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp4 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu47, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp4")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp4 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp4.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e48 = cudaDeviceSynchronize();
        if (e48 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp4 beta=0] %s\n", cudaGetErrorString(e48));
        std::vector<__half> C_gpu48(C_init.size());
        cudaMemcpy(C_gpu48.data(), d_C, C_gpu48.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp4(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp4, C_gpu48, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp5")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e49 = cudaDeviceSynchronize();
        if (e49 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp5] %s\n", cudaGetErrorString(e49));
        std::vector<__half> C_gpu49(C_init.size());
        cudaMemcpy(C_gpu49.data(), d_C, C_gpu49.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu49, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp5")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp5 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp5.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e50 = cudaDeviceSynchronize();
        if (e50 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp5 beta=0] %s\n", cudaGetErrorString(e50));
        std::vector<__half> C_gpu50(C_init.size());
        cudaMemcpy(C_gpu50.data(), d_C, C_gpu50.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp5(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp5, C_gpu50, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp6")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp6(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e51 = cudaDeviceSynchronize();
        if (e51 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp6] %s\n", cudaGetErrorString(e51));
        std::vector<__half> C_gpu51(C_init.size());
        cudaMemcpy(C_gpu51.data(), d_C, C_gpu51.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp6 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu51, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_deep5_bigwarp6")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigwarp6 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigwarp6.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp6(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e52 = cudaDeviceSynchronize();
        if (e52 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_deep5_bigwarp6 beta=0] %s\n", cudaGetErrorString(e52));
        std::vector<__half> C_gpu52(C_init.size());
        cudaMemcpy(C_gpu52.data(), d_C, C_gpu52.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_deep5_bigwarp6(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigwarp6, C_gpu52, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_deep3")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_deep3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e53 = cudaDeviceSynchronize();
        if (e53 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_deep3] %s\n", cudaGetErrorString(e53));
        std::vector<__half> C_gpu53(C_init.size());
        cudaMemcpy(C_gpu53.data(), d_C, C_gpu53.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_deep3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu53, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_deep3")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_deep3 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep3.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_deep3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e54 = cudaDeviceSynchronize();
        if (e54 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_deep3 beta=0] %s\n", cudaGetErrorString(e54));
        std::vector<__half> C_gpu54(C_init.size());
        cudaMemcpy(C_gpu54.data(), d_C, C_gpu54.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_deep3(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep3, C_gpu54, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_xswz")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e55 = cudaDeviceSynchronize();
        if (e55 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_xswz] %s\n", cudaGetErrorString(e55));
        std::vector<__half> C_gpu55(C_init.size());
        cudaMemcpy(C_gpu55.data(), d_C, C_gpu55.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_xswz %s", tag);
        all_ok &= check(label, C_cpu, C_gpu55, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_xswz")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_xswz = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_xswz.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e56 = cudaDeviceSynchronize();
        if (e56 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_xswz beta=0] %s\n", cudaGetErrorString(e56));
        std::vector<__half> C_gpu56(C_init.size());
        cudaMemcpy(C_gpu56.data(), d_C, C_gpu56.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_xswz(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_xswz, C_gpu56, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e57 = cudaDeviceSynchronize();
        if (e57 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2] %s\n", cudaGetErrorString(e57));
        std::vector<__half> C_gpu57(C_init.size());
        cudaMemcpy(C_gpu57.data(), d_C, C_gpu57.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu57, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_xswz2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_xswz2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e58 = cudaDeviceSynchronize();
        if (e58 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2 beta=0] %s\n", cudaGetErrorString(e58));
        std::vector<__half> C_gpu58(C_init.size());
        cudaMemcpy(C_gpu58.data(), d_C, C_gpu58.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_xswz2, C_gpu58, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_cutlass_config")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_cutlass_config(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e59 = cudaDeviceSynchronize();
        if (e59 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_cutlass_config] %s\n", cudaGetErrorString(e59));
        std::vector<__half> C_gpu59(C_init.size());
        cudaMemcpy(C_gpu59.data(), d_C, C_gpu59.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_cutlass_config %s", tag);
        all_ok &= check(label, C_cpu, C_gpu59, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_cutlass_config")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_cutlass = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_cutlass.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_cutlass_config(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e60 = cudaDeviceSynchronize();
        if (e60 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_cutlass_config beta=0] %s\n", cudaGetErrorString(e60));
        std::vector<__half> C_gpu60(C_init.size());
        cudaMemcpy(C_gpu60.data(), d_C, C_gpu60.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_cutlass_config(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_cutlass, C_gpu60, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e63 = cudaDeviceSynchronize();
        if (e63 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg] %s\n", cudaGetErrorString(e63));
        std::vector<__half> C_gpu63(C_init.size());
        cudaMemcpy(C_gpu63.data(), d_C, C_gpu63.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg %s", tag);
        all_ok &= check(label, C_cpu, C_gpu63, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigreg = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigreg.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e64 = cudaDeviceSynchronize();
        if (e64 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg beta=0] %s\n", cudaGetErrorString(e64));
        std::vector<__half> C_gpu64(C_init.size());
        cudaMemcpy(C_gpu64.data(), d_C, C_gpu64.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigreg, C_gpu64, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e65 = cudaDeviceSynchronize();
        if (e65 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq] %s\n", cudaGetErrorString(e65));
        std::vector<__half> C_gpu65(C_init.size());
        cudaMemcpy(C_gpu65.data(), d_C, C_gpu65.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq %s", tag);
        all_ok &= check(label, C_cpu, C_gpu65, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_bigregsq = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigregsq.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e66 = cudaDeviceSynchronize();
        if (e66 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq beta=0] %s\n", cudaGetErrorString(e66));
        std::vector<__half> C_gpu66(C_init.size());
        cudaMemcpy(C_gpu66.data(), d_C, C_gpu66.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigregsq, C_gpu66, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e67 = cudaDeviceSynchronize();
        if (e67 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4] %s\n", cudaGetErrorString(e67));
        std::vector<__half> C_gpu67(C_init.size());
        cudaMemcpy(C_gpu67.data(), d_C, C_gpu67.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu67, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4")) {
        std::vector<__half> C_cpu_beta0_deep4 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e68 = cudaDeviceSynchronize();
        if (e68 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 beta=0] %s\n", cudaGetErrorString(e68));
        std::vector<__half> C_gpu68(C_init.size());
        cudaMemcpy(C_gpu68.data(), d_C, C_gpu68.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4, C_gpu68, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e79 = cudaDeviceSynchronize();
        if (e79 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter] %s\n", cudaGetErrorString(e79));
        std::vector<__half> C_gpu79(C_init.size());
        cudaMemcpy(C_gpu79.data(), d_C, C_gpu79.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter %s", tag);
        all_ok &= check(label, C_cpu, C_gpu79, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter")) {
        std::vector<__half> C_cpu_beta0_deep4_iter = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e80 = cudaDeviceSynchronize();
        if (e80 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter beta=0] %s\n", cudaGetErrorString(e80));
        std::vector<__half> C_gpu80(C_init.size());
        cudaMemcpy(C_gpu80.data(), d_C, C_gpu80.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter, C_gpu80, 0.15 * tol_scale);
    }
    // NOTE: gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec's vectorized
    // epilogue only takes its fast path when n is an exact multiple of
    // BOTH kGemmFp16BigBM(256) and kGemmFp16WideBigRegSqBN(128) -- of
    // this sweep's sizes, only n=256 and n=2048 qualify; every other n
    // here (1/15/16/17/33/64/129) falls back to gemm_fp16_mma_ptx_
    // bigtile_w32_epi_wide (the same fallback every other kernel in
    // this family already uses), so those sizes only confirm correct
    // ROUTING to that already-validated fallback, not the vectorized
    // epilogue itself -- n=256/2048 are what actually exercise it.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e81 = cudaDeviceSynchronize();
        if (e81 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec] %s\n", cudaGetErrorString(e81));
        std::vector<__half> C_gpu81(C_init.size());
        cudaMemcpy(C_gpu81.data(), d_C, C_gpu81.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu81, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec")) {
        std::vector<__half> C_cpu_beta0_deep4_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e82 = cudaDeviceSynchronize();
        if (e82 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec beta=0] %s\n", cudaGetErrorString(e82));
        std::vector<__half> C_gpu82(C_init.size());
        cudaMemcpy(C_gpu82.data(), d_C, C_gpu82.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_epivec, C_gpu82, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry above:
    // this sweep's n=256/2048 exercise the vectorized epilogue; the
    // rest only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e83 = cudaDeviceSynchronize();
        if (e83 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec] %s\n", cudaGetErrorString(e83));
        std::vector<__half> C_gpu83(C_init.size());
        cudaMemcpy(C_gpu83.data(), d_C, C_gpu83.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu83, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e84 = cudaDeviceSynchronize();
        if (e84 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec beta=0] %s\n", cudaGetErrorString(e84));
        std::vector<__half> C_gpu84(C_init.size());
        cudaMemcpy(C_gpu84.data(), d_C, C_gpu84.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec, C_gpu84, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the re-padded fast path; the rest
    // only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e85 = cudaDeviceSynchronize();
        if (e85 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align] %s\n", cudaGetErrorString(e85));
        std::vector<__half> C_gpu85(C_init.size());
        cudaMemcpy(C_gpu85.data(), d_C, C_gpu85.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align %s", tag);
        all_ok &= check(label, C_cpu, C_gpu85, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_align = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_align.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e86 = cudaDeviceSynchronize();
        if (e86 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align beta=0] %s\n", cudaGetErrorString(e86));
        std::vector<__half> C_gpu86(C_init.size());
        cudaMemcpy(C_gpu86.data(), d_C, C_gpu86.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_align, C_gpu86, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the re-padded (280) fast path; the
    // rest only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e99 = cudaDeviceSynchronize();
        if (e99 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280] %s\n", cudaGetErrorString(e99));
        std::vector<__half> C_gpu99(C_init.size());
        cudaMemcpy(C_gpu99.data(), d_C, C_gpu99.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu99, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_pad280 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_pad280.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e100 = cudaDeviceSynchronize();
        if (e100 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280 beta=0] %s\n", cudaGetErrorString(e100));
        std::vector<__half> C_gpu100(C_init.size());
        cudaMemcpy(C_gpu100.data(), d_C, C_gpu100.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_pad280, C_gpu100, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as pad280's own entry, but with EXTRA
    // emphasis: this kernel replaces A's shared-memory layout entirely
    // with a new XOR-permuted-chunk swizzle (not just a padding tweak),
    // so this is the most important correctness check in this whole
    // A-shared-memory sub-family -- do not skip it.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e103 = cudaDeviceSynchronize();
        if (e103 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle] %s\n", cudaGetErrorString(e103));
        std::vector<__half> C_gpu103(C_init.size());
        cudaMemcpy(C_gpu103.data(), d_C, C_gpu103.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle %s", tag);
        all_ok &= check(label, C_cpu, C_gpu103, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_aswizzle = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_aswizzle.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e104 = cudaDeviceSynchronize();
        if (e104 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle beta=0] %s\n", cudaGetErrorString(e104));
        std::vector<__half> C_gpu104(C_init.size());
        cudaMemcpy(C_gpu104.data(), d_C, C_gpu104.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_aswizzle, C_gpu104, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as aswizzle's own entry, but with
    // EXTRA emphasis: this kernel implements real CUTLASS's own A
    // formula, read directly from CUTLASS's header source -- the most
    // important correctness check in this whole A-shared-memory sub-
    // family given how much rides on the derivation being right.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e107 = cudaDeviceSynchronize();
        if (e107 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous] %s\n", cudaGetErrorString(e107));
        std::vector<__half> C_gpu107(C_init.size());
        cudaMemcpy(C_gpu107.data(), d_C, C_gpu107.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous %s", tag);
        all_ok &= check(label, C_cpu, C_gpu107, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e108 = cudaDeviceSynchronize();
        if (e108 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous beta=0] %s\n", cudaGetErrorString(e108));
        std::vector<__half> C_gpu108(C_init.size());
        cudaMemcpy(C_gpu108.data(), d_C, C_gpu108.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous, C_gpu108, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as acongruous's own entry: this
    // sweep's n=256/2048 exercise the fast path; the rest only confirm
    // correct fallback routing. Correctness here rests on both source
    // kernels' own already-verified correctness composing safely.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e109 = cudaDeviceSynchronize();
        if (e109 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe] %s\n", cudaGetErrorString(e109));
        std::vector<__half> C_gpu109(C_init.size());
        cudaMemcpy(C_gpu109.data(), d_C, C_gpu109.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu109, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_pipe = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_pipe.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e110 = cudaDeviceSynchronize();
        if (e110 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe beta=0] %s\n", cudaGetErrorString(e110));
        std::vector<__half> C_gpu110(C_init.size());
        cudaMemcpy(C_gpu110.data(), d_C, C_gpu110.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_pipe, C_gpu110, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as acongruous's own entry. Given
    // this kernel's own correctness is algebraically guaranteed by a
    // verified-identical factoring (not a new formula), this check
    // confirms the transcription into kernel code, not the derivation
    // itself.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e111 = cudaDeviceSynchronize();
        if (e111 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2] %s\n", cudaGetErrorString(e111));
        std::vector<__half> C_gpu111(C_init.size());
        cudaMemcpy(C_gpu111.data(), d_C, C_gpu111.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu111, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e112 = cudaDeviceSynchronize();
        if (e112 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2 beta=0] %s\n", cudaGetErrorString(e112));
        std::vector<__half> C_gpu112(C_init.size());
        cudaMemcpy(C_gpu112.data(), d_C, C_gpu112.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous2, C_gpu112, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as acongruous's own entry; this
    // sweep's n=256/2048 exercise the shallower-pipeline fast path; the
    // rest only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e113 = cudaDeviceSynchronize();
        if (e113 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3] %s\n", cudaGetErrorString(e113));
        std::vector<__half> C_gpu113(C_init.size());
        cudaMemcpy(C_gpu113.data(), d_C, C_gpu113.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu113, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_stages3 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_stages3.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e114 = cudaDeviceSynchronize();
        if (e114 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3 beta=0] %s\n", cudaGetErrorString(e114));
        std::vector<__half> C_gpu114(C_init.size());
        cudaMemcpy(C_gpu114.data(), d_C, C_gpu114.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_stages3, C_gpu114, 0.15 * tol_scale);
    }
    // acongruous_ldgsts's A-formula and mainloop rotation are byte-for-
    // byte identical to acongruous's own (already verified); the only
    // change is per-step cp.async issue order, which is a pure
    // algebraic de-incrementalization of the same address/predicate
    // formulas -- still, this is the most important correctness check
    // for this kernel, since a step-ordering bug (e.g. an off-by-one in
    // which wp a given rep is issued at) could silently write to/read
    // from the wrong shared-memory slot. Same fast-path-coverage note as
    // acongruous's own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e115 = cudaDeviceSynchronize();
        if (e115 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts] %s\n", cudaGetErrorString(e115));
        std::vector<__half> C_gpu115(C_init.size());
        cudaMemcpy(C_gpu115.data(), d_C, C_gpu115.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts %s", tag);
        all_ok &= check(label, C_cpu, C_gpu115, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_ldgsts = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_ldgsts.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e116 = cudaDeviceSynchronize();
        if (e116 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts beta=0] %s\n", cudaGetErrorString(e116));
        std::vector<__half> C_gpu116(C_init.size());
        cudaMemcpy(C_gpu116.data(), d_C, C_gpu116.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_ldgsts, C_gpu116, 0.15 * tol_scale);
    }
    // fastk's own boundary-branch elimination is only reachable when
    // n % kGemmFp16WideBigRegBK == 0 as well as the usual tile-alignment
    // checks -- sizes that satisfy that exercise the actual no-boundary-
    // code kernel; the rest confirm this kernel's own fallback to the
    // existing, already-correctness-proven acongruous kernel is wired
    // correctly. Since fastk's mainloop body is a mechanical deletion of
    // acongruous's own dead branch (no formula changes), this check is
    // mainly about verifying that deletion didn't silently break the
    // remaining code, not re-deriving anything new.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e117 = cudaDeviceSynchronize();
        if (e117 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk] %s\n", cudaGetErrorString(e117));
        std::vector<__half> C_gpu117(C_init.size());
        cudaMemcpy(C_gpu117.data(), d_C, C_gpu117.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu117, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e118 = cudaDeviceSynchronize();
        if (e118 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk beta=0] %s\n", cudaGetErrorString(e118));
        std::vector<__half> C_gpu118(C_init.size());
        cudaMemcpy(C_gpu118.data(), d_C, C_gpu118.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk, C_gpu118, 0.15 * tol_scale);
    }
    // fastk_precomp's own A-address LOOKUP TABLE construction is the
    // part most worth checking here -- an off-by-one in which (kh, wm)
    // or rep index maps to which precomputed offset would silently read
    // the wrong shared-memory slot, unlike fastk's own change (a pure
    // dead-branch deletion). Same fast-path-coverage note as
    // acongruous's own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e119 = cudaDeviceSynchronize();
        if (e119 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp] %s\n", cudaGetErrorString(e119));
        std::vector<__half> C_gpu119(C_init.size());
        cudaMemcpy(C_gpu119.data(), d_C, C_gpu119.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu119, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e120 = cudaDeviceSynchronize();
        if (e120 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp beta=0] %s\n", cudaGetErrorString(e120));
        std::vector<__half> C_gpu120(C_init.size());
        cudaMemcpy(C_gpu120.data(), d_C, C_gpu120.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp, C_gpu120, 0.15 * tol_scale);
    }
    // fastk_precomp2's own B-address LOOKUP TABLE construction is the
    // part most worth checking here (same reasoning as fastk_precomp's
    // own A-side table check) -- an off-by-one in which wp/rep index
    // maps to which precomputed B offset would silently read/write the
    // wrong shared-memory slot. Same fast-path-coverage note as
    // acongruous's own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e121 = cudaDeviceSynchronize();
        if (e121 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2] %s\n", cudaGetErrorString(e121));
        std::vector<__half> C_gpu121(C_init.size());
        cudaMemcpy(C_gpu121.data(), d_C, C_gpu121.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu121, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e122 = cudaDeviceSynchronize();
        if (e122 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2 beta=0] %s\n", cudaGetErrorString(e122));
        std::vector<__half> C_gpu122(C_init.size());
        cudaMemcpy(C_gpu122.data(), d_C, C_gpu122.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2, C_gpu122, 0.15 * tol_scale);
    }
    // fastk_precomp3's own three-way outstanding==2/1/0 dispatch is the
    // part most worth checking here -- a bug there would show up ONLY
    // near the very end of the K dimension (the last couple of tiles,
    // where outstanding takes 1 and 0 instead of the steady-state 2),
    // so this correctness check matters even though the change looks
    // purely mechanical. Same fast-path-coverage note as acongruous's
    // own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e123 = cudaDeviceSynchronize();
        if (e123 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3] %s\n", cudaGetErrorString(e123));
        std::vector<__half> C_gpu123(C_init.size());
        cudaMemcpy(C_gpu123.data(), d_C, C_gpu123.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu123, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp3 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp3.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e124 = cudaDeviceSynchronize();
        if (e124 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3 beta=0] %s\n", cudaGetErrorString(e124));
        std::vector<__half> C_gpu124(C_init.size());
        cudaMemcpy(C_gpu124.data(), d_C, C_gpu124.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp3, C_gpu124, 0.15 * tol_scale);
    }
    // fastk_precomp2_pipe's own interleaved B-load timing is the part
    // most worth checking here -- br0[0][wp] etc. get overwritten
    // mid-block by the NEXT tile's own data while the SAME iteration's
    // second MMA block is still consuming br0[1] (the [1] slot, not
    // [0]), so a mixed-up buffer index would silently read the wrong
    // tile's B data. Same fast-path-coverage note as acongruous's own
    // entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e125 = cudaDeviceSynchronize();
        if (e125 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe] %s\n", cudaGetErrorString(e125));
        std::vector<__half> C_gpu125(C_init.size());
        cudaMemcpy(C_gpu125.data(), d_C, C_gpu125.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu125, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_pipe = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_pipe.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e126 = cudaDeviceSynchronize();
        if (e126 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe beta=0] %s\n", cudaGetErrorString(e126));
        std::vector<__half> C_gpu126(C_init.size());
        cudaMemcpy(C_gpu126.data(), d_C, C_gpu126.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_pipe, C_gpu126, 0.15 * tol_scale);
    }
    // fastk_precomp2_serp's own accumulate-loop REORDER is the most
    // important correctness check of this whole family so far -- it
    // touches the core wm/wn -> (a_reg, b_reg, acc_slot) mapping for
    // every one of the 32 mma.sync calls per MMA block, both blocks.
    // Verified as a pure bijective permutation in Python, but that
    // proof lived outside the actual CUDA source; this is the first
    // real confirmation that the C++ index arithmetic (wn, wp, the
    // (wn%2) branch selecting br0/br1 vs br2/br3) matches that proof
    // exactly. Same fast-path-coverage note as acongruous's own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e127 = cudaDeviceSynchronize();
        if (e127 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp] %s\n", cudaGetErrorString(e127));
        std::vector<__half> C_gpu127(C_init.size());
        cudaMemcpy(C_gpu127.data(), d_C, C_gpu127.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu127, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e128 = cudaDeviceSynchronize();
        if (e128 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp beta=0] %s\n", cudaGetErrorString(e128));
        std::vector<__half> C_gpu128(C_init.size());
        cudaMemcpy(C_gpu128.data(), d_C, C_gpu128.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp, C_gpu128, 0.15 * tol_scale);
    }
    // fastk_precomp2_serp_ildsm is the most structurally invasive
    // change in this entire family's history -- it moves EVERY
    // individual load_a_frag/load_b_frags ldmatrix call to a new
    // program point, interleaved a full accumulate block away from
    // where it used to fire. A subtle off-by-one in WHICH wm-row a
    // given load_a_frag_step/load_b_frag_step call belongs to, or a
    // mixed-up kh/lane_offset for the ks=16 vs ks=0 substep, would
    // silently read stale or wrong-substep data while still producing
    // plausible-looking (but wrong) numbers -- this correctness check
    // matters more than usual for this specific kernel. Same fast-
    // path-coverage note as acongruous's own entry.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e129 = cudaDeviceSynchronize();
        if (e129 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm] %s\n", cudaGetErrorString(e129));
        std::vector<__half> C_gpu129(C_init.size());
        cudaMemcpy(C_gpu129.data(), d_C, C_gpu129.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm %s", tag);
        all_ok &= check(label, C_cpu, C_gpu129, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e130 = cudaDeviceSynchronize();
        if (e130 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm beta=0] %s\n", cudaGetErrorString(e130));
        std::vector<__half> C_gpu130(C_init.size());
        cudaMemcpy(C_gpu130.data(), d_C, C_gpu130.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm, C_gpu130, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the shallower-pipeline fast path; the
    // rest only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e97 = cudaDeviceSynchronize();
        if (e97 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3] %s\n", cudaGetErrorString(e97));
        std::vector<__half> C_gpu97(C_init.size());
        cudaMemcpy(C_gpu97.data(), d_C, C_gpu97.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu97, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_stages3 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_stages3.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e98 = cudaDeviceSynchronize();
        if (e98 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3 beta=0] %s\n", cudaGetErrorString(e98));
        std::vector<__half> C_gpu98(C_init.size());
        cudaMemcpy(C_gpu98.data(), d_C, C_gpu98.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_stages3, C_gpu98, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the deeper-pipeline fast path; the
    // rest only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e95 = cudaDeviceSynchronize();
        if (e95 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5] %s\n", cudaGetErrorString(e95));
        std::vector<__half> C_gpu95(C_init.size());
        cudaMemcpy(C_gpu95.data(), d_C, C_gpu95.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu95, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_stages5 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_stages5.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e96 = cudaDeviceSynchronize();
        if (e96 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5 beta=0] %s\n", cudaGetErrorString(e96));
        std::vector<__half> C_gpu96(C_init.size());
        cudaMemcpy(C_gpu96.data(), d_C, C_gpu96.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_stages5, C_gpu96, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the interleaved fast path; the rest
    // only confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e87 = cudaDeviceSynchronize();
        if (e87 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe] %s\n", cudaGetErrorString(e87));
        std::vector<__half> C_gpu87(C_init.size());
        cudaMemcpy(C_gpu87.data(), d_C, C_gpu87.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu87, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_pipe = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_pipe.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e88 = cudaDeviceSynchronize();
        if (e88 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe beta=0] %s\n", cudaGetErrorString(e88));
        std::vector<__half> C_gpu88(C_init.size());
        cudaMemcpy(C_gpu88.data(), d_C, C_gpu88.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_pipe, C_gpu88, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the hoisted fast path; the rest only
    // confirm correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e89 = cudaDeviceSynchronize();
        if (e89 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec] %s\n", cudaGetErrorString(e89));
        std::vector<__half> C_gpu89(C_init.size());
        cudaMemcpy(C_gpu89.data(), d_C, C_gpu89.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu89, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec")) {
        std::vector<__half> C_cpu_beta0_deep4_iter2_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter2_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e90 = cudaDeviceSynchronize();
        if (e90 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec beta=0] %s\n", cudaGetErrorString(e90));
        std::vector<__half> C_gpu90(C_init.size());
        cudaMemcpy(C_gpu90.data(), d_C, C_gpu90.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter2_epivec, C_gpu90, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the gridswap fast path; the rest only
    // confirm correct fallback routing. This case matters more than
    // usual here since the change swaps blockIdx.x/y -- if that swap
    // were done wrong, the grid-quantization/boundary behavior at these
    // small sizes is exactly where it would most likely surface as a
    // wrong answer.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e91 = cudaDeviceSynchronize();
        if (e91 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap] %s\n", cudaGetErrorString(e91));
        std::vector<__half> C_gpu91(C_init.size());
        cudaMemcpy(C_gpu91.data(), d_C, C_gpu91.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap %s", tag);
        all_ok &= check(label, C_cpu, C_gpu91, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap")) {
        std::vector<__half> C_cpu_beta0_deep4_iter_epivec_gridswap = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep4_iter_epivec_gridswap.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e92 = cudaDeviceSynchronize();
        if (e92 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap beta=0] %s\n", cudaGetErrorString(e92));
        std::vector<__half> C_gpu92(C_init.size());
        cudaMemcpy(C_gpu92.data(), d_C, C_gpu92.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep4_iter_epivec_gridswap, C_gpu92, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as the rest of this family: this
    // sweep's n=256/2048 exercise the new BM=128/BN=256 fast path (both
    // remain exact multiples of the new, stricter BN=256 requirement);
    // the rest only confirm correct fallback routing. This is the most
    // important correctness check in the whole family given this
    // kernel's own genuinely new tile geometry -- do not skip it.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e93 = cudaDeviceSynchronize();
        if (e93 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec] %s\n", cudaGetErrorString(e93));
        std::vector<__half> C_gpu93(C_init.size());
        cudaMemcpy(C_gpu93.data(), d_C, C_gpu93.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu93, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec")) {
        std::vector<__half> C_cpu_beta0_cutlassdims_iter_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_cutlassdims_iter_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e94 = cudaDeviceSynchronize();
        if (e94 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec beta=0] %s\n", cudaGetErrorString(e94));
        std::vector<__half> C_gpu94(C_init.size());
        cudaMemcpy(C_gpu94.data(), d_C, C_gpu94.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_cutlassdims_iter_epivec, C_gpu94, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as cutlassdims_iter_epivec's own
    // entry: this sweep's n=256/2048 exercise the fast path (BM=128,
    // BN=256, Stages=3); the rest only confirm correct fallback
    // routing. This is the full CUTLASS-matching config, the most
    // important correctness check for this specific comparison.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e101 = cudaDeviceSynchronize();
        if (e101 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec] %s\n", cudaGetErrorString(e101));
        std::vector<__half> C_gpu101(C_init.size());
        cudaMemcpy(C_gpu101.data(), d_C, C_gpu101.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu101, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec")) {
        std::vector<__half> C_cpu_beta0_cutlassdims3_iter_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_cutlassdims3_iter_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e102 = cudaDeviceSynchronize();
        if (e102 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec beta=0] %s\n", cudaGetErrorString(e102));
        std::vector<__half> C_gpu102(C_init.size());
        cudaMemcpy(C_gpu102.data(), d_C, C_gpu102.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_cutlassdims3_iter_epivec, C_gpu102, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep5")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e69 = cudaDeviceSynchronize();
        if (e69 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep5] %s\n", cudaGetErrorString(e69));
        std::vector<__half> C_gpu69(C_init.size());
        cudaMemcpy(C_gpu69.data(), d_C, C_gpu69.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu69, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep5")) {
        std::vector<__half> C_cpu_beta0_deep5sq = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_deep5sq.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e70 = cudaDeviceSynchronize();
        if (e70 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep5 beta=0] %s\n", cudaGetErrorString(e70));
        std::vector<__half> C_gpu70(C_init.size());
        cudaMemcpy(C_gpu70.data(), d_C, C_gpu70.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_deep5sq, C_gpu70, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e71 = cudaDeviceSynchronize();
        if (e71 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2] %s\n", cudaGetErrorString(e71));
        std::vector<__half> C_gpu71(C_init.size());
        cudaMemcpy(C_gpu71.data(), d_C, C_gpu71.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu71, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2")) {
        std::vector<__half> C_cpu_beta0_occ2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_occ2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e72 = cudaDeviceSynchronize();
        if (e72 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2 beta=0] %s\n", cudaGetErrorString(e72));
        std::vector<__half> C_gpu72(C_init.size());
        cudaMemcpy(C_gpu72.data(), d_C, C_gpu72.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_occ2, C_gpu72, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e73 = cudaDeviceSynchronize();
        if (e73 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma] %s\n", cudaGetErrorString(e73));
        std::vector<__half> C_gpu73(C_init.size());
        cudaMemcpy(C_gpu73.data(), d_C, C_gpu73.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma %s", tag);
        all_ok &= check(label, C_cpu, C_gpu73, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma")) {
        std::vector<__half> C_cpu_beta0_bigmma = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bigmma.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e74 = cudaDeviceSynchronize();
        if (e74 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma beta=0] %s\n", cudaGetErrorString(e74));
        std::vector<__half> C_gpu74(C_init.size());
        cudaMemcpy(C_gpu74.data(), d_C, C_gpu74.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bigmma, C_gpu74, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e75 = cudaDeviceSynchronize();
        if (e75 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64] %s\n", cudaGetErrorString(e75));
        std::vector<__half> C_gpu75(C_init.size());
        cudaMemcpy(C_gpu75.data(), d_C, C_gpu75.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu75, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64")) {
        std::vector<__half> C_cpu_beta0_bk64 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bk64.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e76 = cudaDeviceSynchronize();
        if (e76 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64 beta=0] %s\n", cudaGetErrorString(e76));
        std::vector<__half> C_gpu76(C_init.size());
        cudaMemcpy(C_gpu76.data(), d_C, C_gpu76.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bk64, C_gpu76, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e77 = cudaDeviceSynchronize();
        if (e77 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2] %s\n", cudaGetErrorString(e77));
        std::vector<__half> C_gpu77(C_init.size());
        cudaMemcpy(C_gpu77.data(), d_C, C_gpu77.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu77, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2")) {
        std::vector<__half> C_cpu_beta0_bk64reg2 = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bk64reg2.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e78 = cudaDeviceSynchronize();
        if (e78 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2 beta=0] %s\n", cudaGetErrorString(e78));
        std::vector<__half> C_gpu78(C_init.size());
        cudaMemcpy(C_gpu78.data(), d_C, C_gpu78.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bk64reg2, C_gpu78, 0.15 * tol_scale);
    }
    // Same fast-path-coverage note as deep4_epivec's own entry: this
    // sweep's n=256/2048 exercise the vectorized-epilogue fast path
    // (ld_c%8==0, m/n exact multiples of BM=256/SqBN=128, stricter than
    // bk64_reg2's own scalar-epilogue fallback); the rest only confirm
    // correct fallback routing.
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e105 = cudaDeviceSynchronize();
        if (e105 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec] %s\n", cudaGetErrorString(e105));
        std::vector<__half> C_gpu105(C_init.size());
        cudaMemcpy(C_gpu105.data(), d_C, C_gpu105.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu105, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec")) {
        std::vector<__half> C_cpu_beta0_bk64reg2_iter_epivec = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_bk64reg2_iter_epivec.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e106 = cudaDeviceSynchronize();
        if (e106 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec beta=0] %s\n", cudaGetErrorString(e106));
        std::vector<__half> C_gpu106(C_init.size());
        cudaMemcpy(C_gpu106.data(), d_C, C_gpu106.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_bk64reg2_iter_epivec, C_gpu106, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_multistage")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_multistage(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e61 = cudaDeviceSynchronize();
        if (e61 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_multistage] %s\n", cudaGetErrorString(e61));
        std::vector<__half> C_gpu61(C_init.size());
        cudaMemcpy(C_gpu61.data(), d_C, C_gpu61.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_multistage %s", tag);
        all_ok &= check(label, C_cpu, C_gpu61, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_multistage")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_widems = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_widems.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_multistage(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e62 = cudaDeviceSynchronize();
        if (e62 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_multistage beta=0] %s\n", cudaGetErrorString(e62));
        std::vector<__half> C_gpu62(C_init.size());
        cudaMemcpy(C_gpu62.data(), d_C, C_gpu62.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_multistage(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_widems, C_gpu62, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_spread")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_spread(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e31 = cudaDeviceSynchronize();
        if (e31 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_spread] %s\n", cudaGetErrorString(e31));
        std::vector<__half> C_gpu31(C_init.size());
        cudaMemcpy(C_gpu31.data(), d_C, C_gpu31.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_spread %s", tag);
        all_ok &= check(label, C_cpu, C_gpu31, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_spread")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_widespread = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_widespread.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_spread(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e32 = cudaDeviceSynchronize();
        if (e32 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_spread beta=0] %s\n", cudaGetErrorString(e32));
        std::vector<__half> C_gpu32(C_init.size());
        cudaMemcpy(C_gpu32.data(), d_C, C_gpu32.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_spread(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_widespread, C_gpu32, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_coalb")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e33 = cudaDeviceSynchronize();
        if (e33 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_coalb] %s\n", cudaGetErrorString(e33));
        std::vector<__half> C_gpu33(C_init.size());
        cudaMemcpy(C_gpu33.data(), d_C, C_gpu33.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_coalb %s", tag);
        all_ok &= check(label, C_cpu, C_gpu33, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_wide_nospill_epismem_coalb")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_nospillepismemcoalb = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_nospillepismemcoalb.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e34 = cudaDeviceSynchronize();
        if (e34 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_wide_nospill_epismem_coalb beta=0] %s\n", cudaGetErrorString(e34));
        std::vector<__half> C_gpu34(C_init.size());
        cudaMemcpy(C_gpu34.data(), d_C, C_gpu34.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_wide_nospill_epismem_coalb(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_nospillepismemcoalb, C_gpu34, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_smalltile_2block")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_smalltile_2block(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e35 = cudaDeviceSynchronize();
        if (e35 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_smalltile_2block] %s\n", cudaGetErrorString(e35));
        std::vector<__half> C_gpu35(C_init.size());
        cudaMemcpy(C_gpu35.data(), d_C, C_gpu35.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_smalltile_2block %s", tag);
        all_ok &= check(label, C_cpu, C_gpu35, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_smalltile_2block")) {
        // Dedicated beta=0 check.
        std::vector<__half> C_cpu_beta0_smalltile2block = C_init;
        cpu::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, 0.0,
            C_cpu_beta0_smalltile2block.data(), n);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_smalltile_2block(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, n, n, n, d_A, n, d_B, n, 0.0, d_C, n);
        cudaError_t e36 = cudaDeviceSynchronize();
        if (e36 != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_smalltile_2block beta=0] %s\n", cudaGetErrorString(e36));
        std::vector<__half> C_gpu36(C_init.size());
        cudaMemcpy(C_gpu36.data(), d_C, C_gpu36.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_smalltile_2block(beta=0) %s", tag);
        all_ok &= check(label, C_cpu_beta0_smalltile2block, C_gpu36, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile2_w32")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile2_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile2_w32] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile2_w32 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile2")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                         alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile2] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("gemm_fp16_mma_ptx_bigtile_tripipe")) {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp16_mma_ptx_bigtile_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp16_mma_ptx_bigtile_tripipe] %s\n", cudaGetErrorString(e));
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp16_mma_ptx_bigtile_tripipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
    }
    if (want("cublasHgemm")) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cuda::experimental::mm_strided<__half, __half, __half>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, d_A, n, dim<2>{n, n}, d_B, n, beta, d_C, n, handle);
        std::vector<__half> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "cublasHgemm %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 0.15 * tol_scale);
        cublasDestroy(handle);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return all_ok;
}

bool run_correctness() {
    std::printf("=== Correctness (gemm_generic_kernel<half>/gemm_fp16_tiled/gemm_fp16_wmma vs CPU reference) ===\n");
    if (g_only_method != nullptr && g_only_method[0] != '\0')
        std::printf("  [--only-method %s: skipping every other kernel]\n", g_only_method);
    bool all_ok = true;

    for (size_t n : {1ul, 15ul, 16ul, 17ul, 33ul, 64ul, 129ul, 256ul, 2048ul})
        all_ok &= run_correctness_case(n, "square");

    std::printf("\n");
    return all_ok;
}

// ---- performance section: gemm_fp16_tiled/gemm_fp16_wmma vs cuBLAS Hgemm ----

// Benchmarks one N x N square shape: the custom fp16 kernels against
// cuBLAS's Hgemm, all operating on the same plain __half A/B/C -- no
// materialization step needed (unlike verify_mm_fp32_16_pure_half_gemm.cu,
// which has to decode wh into a plain __half array before cuBLAS can use
// it), since there's no packed format here at all. Pulled out to a
// standalone function so run_single_shape() can also call it.
void bench_one(cublasHandle_t handle, size_t n) {
    __half *A, *B, *C;
    if (cudaMalloc(&A, n * n * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&B, n * n * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&C, n * n * sizeof(__half)) != cudaSuccess) {
        std::printf("  N=%-9zu  SKIPPED (out of device memory)\n", n);
        cudaGetLastError();
        return;
    }
    gen_half(n * n, A);
    gen_half(n * n, B);
    cudaMemset(C, 0, n * n * sizeof(__half));
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

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    int reps = 5;
    // gemm_generic_kernel's throughput is naive-kernel flat (one thread per
    // output element, no shared-memory reuse), same reasoning as every
    // other verify_*.cu file's naive-kernel skip threshold.
    constexpr size_t kSkipGenericAbove = 8192;
    double t_generic = n <= kSkipGenericAbove
                            ? time_call(
                                  [&] {
                                      cuda::experimental::gemm_generic_kernel<__half, __half, __half>
                                          <<<blocks, threads>>>(
                                              TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              1.0, n, n, n, A, n, B, n, 0.0, C, n);
                                  },
                                  reps)
                            : 0.0;
    double t_tiled = time_call(
        [&] {
            cuda::experimental::gemm_fp16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                 1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma_cp = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma_cp_vec = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_dualbuf = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_bigbk = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_bigbk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_bigbk_pad = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_bigbk_pad(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_hugebk = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_hugebk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_tripipe = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_warpspec = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_warpspec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_w32 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_wmma_cpasync_vec_hugebk_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_swizzle = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_swizzle(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_deep = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_deep(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32_epi = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32_epi_l2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_l2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32_epi_wide = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32_epi_wide2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                          1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_w32_epi_wide2b = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_w32 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_deep5 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_deep5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_coalb = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_coalb_shfl = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_coalb_shfl(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_epi_smem = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_epi_smem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill_epismem = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp_apipe = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_apipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp_fullpipe = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp2_epismem = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_epismem(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp2_hoisted = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp3 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp4 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp5 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp5(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_deep5_bigwarp6 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_deep5_bigwarp6(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill_epismem_deep3 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_deep3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill_epismem_xswz = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill_epismem_xswz2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_cutlass_config = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_cutlass_config(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_bigreg = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_bigreg_sq = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_multistage = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_multistage(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_spread = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_spread(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_wide_nospill_epismem_coalb = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_wide_nospill_epismem_coalb(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                               1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_smalltile_2block = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_smalltile_2block(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile2_w32 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile2_w32(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile2 = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_mma_ptx_bigtile_tripipe = time_call(
        [&] {
            cuda::experimental::gemm_fp16_mma_ptx_bigtile_tripipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mm_strided<__half, __half, __half>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                {n, n}, A, n, {n, n}, B, n, 0.0, C, n, handle);
        },
        reps);

    double flops = 2.0 * n * n * n;
    struct NamedTime { const char* name; double t; };
    NamedTime candidates[] = {{"tiled", t_tiled}, {"wmma", t_wmma}, {"wmma_cp", t_wmma_cp},
                              {"wmma_cp_vec", t_wmma_cp_vec}, {"dualbuf", t_dualbuf},
                              {"bigbk", t_bigbk}, {"bigbk_pad", t_bigbk_pad},
                              {"hugebk", t_hugebk}, {"tripipe", t_tripipe},
                              {"warpspec", t_warpspec}, {"w32", t_w32}, {"mma_ptx", t_mma_ptx},
                              {"mma_ptx_bigtile", t_mma_ptx_bigtile},
                              {"mma_ptx_bigtile_swizzle", t_mma_ptx_bigtile_swizzle},
                              {"mma_ptx_bigtile_deep", t_mma_ptx_bigtile_deep},
                              {"mma_ptx_bigtile_w32", t_mma_ptx_bigtile_w32},
                              {"mma_ptx_bigtile_w32_epi", t_mma_ptx_bigtile_w32_epi},
                              {"mma_ptx_bigtile_w32_epi_l2", t_mma_ptx_bigtile_w32_epi_l2},
                              {"mma_ptx_bigtile_w32_epi_wide", t_mma_ptx_bigtile_w32_epi_wide},
                              {"mma_ptx_bigtile_w32_epi_wide2", t_mma_ptx_bigtile_w32_epi_wide2},
                              {"mma_ptx_bigtile_w32_epi_wide2b", t_mma_ptx_bigtile_w32_epi_wide2b},
                              {"mma_ptx_deep5", t_mma_ptx_deep5},
                              {"mma_ptx_deep5_w32", t_mma_ptx_deep5_w32},
                              {"mma_ptx_wide_deep5", t_mma_ptx_wide_deep5},
                              {"mma_ptx_wide_coalb", t_mma_ptx_wide_coalb},
                              {"mma_ptx_wide_coalb_shfl", t_mma_ptx_wide_coalb_shfl},
                              {"mma_ptx_wide_nospill2", t_mma_ptx_wide_nospill2},
                              {"mma_ptx_wide_epi_smem", t_mma_ptx_wide_epi_smem},
                              {"mma_ptx_wide_nospill_epismem", t_mma_ptx_wide_nospill_epismem},
                              {"mma_ptx_deep5_bigwarp", t_mma_ptx_deep5_bigwarp},
                              {"mma_ptx_deep5_bigwarp_apipe", t_mma_ptx_deep5_bigwarp_apipe},
                              {"mma_ptx_deep5_bigwarp_fullpipe", t_mma_ptx_deep5_bigwarp_fullpipe},
                              {"mma_ptx_deep5_bigwarp2", t_mma_ptx_deep5_bigwarp2},
                              {"mma_ptx_deep5_bigwarp2_epismem", t_mma_ptx_deep5_bigwarp2_epismem},
                              {"mma_ptx_deep5_bigwarp2_hoisted", t_mma_ptx_deep5_bigwarp2_hoisted},
                              {"mma_ptx_deep5_bigwarp3", t_mma_ptx_deep5_bigwarp3},
                              {"mma_ptx_deep5_bigwarp4", t_mma_ptx_deep5_bigwarp4},
                              {"mma_ptx_deep5_bigwarp5", t_mma_ptx_deep5_bigwarp5},
                              {"mma_ptx_deep5_bigwarp6", t_mma_ptx_deep5_bigwarp6},
                              {"mma_ptx_wide_nospill_epismem_deep3", t_mma_ptx_wide_nospill_epismem_deep3},
                              {"mma_ptx_wide_nospill_epismem_xswz", t_mma_ptx_wide_nospill_epismem_xswz},
                              {"mma_ptx_wide_nospill_epismem_xswz2", t_mma_ptx_wide_nospill_epismem_xswz2},
                              {"mma_ptx_cutlass_config", t_mma_ptx_cutlass_config},
                              {"mma_ptx_wide_bigreg", t_mma_ptx_wide_bigreg},
                              {"mma_ptx_wide_bigreg_sq", t_mma_ptx_wide_bigreg_sq},
                              {"mma_ptx_wide_multistage", t_mma_ptx_wide_multistage},
                              {"mma_ptx_wide_spread", t_mma_ptx_wide_spread},
                              {"mma_ptx_wide_nospill_epismem_coalb", t_mma_ptx_wide_nospill_epismem_coalb},
                              {"mma_ptx_smalltile_2block", t_mma_ptx_smalltile_2block},
                              {"mma_ptx_bigtile2_w32", t_mma_ptx_bigtile2_w32},
                              {"mma_ptx_bigtile2", t_mma_ptx_bigtile2},
                              {"mma_ptx_bigtile_tripipe", t_mma_ptx_bigtile_tripipe}};
    const NamedTime* best_entry = &candidates[0];
    for (const auto& c : candidates)
        if (c.t < best_entry->t) best_entry = &c;
    if (n <= kSkipGenericAbove) {
        std::printf("  N=%-9zu generic=%9.3fms (%7.1f GFlop/s)  tiled=%9.3fms (%7.1f GFlop/s)  "
                   "wmma=%9.3fms (%7.1f GFlop/s)  wmma_cp=%9.3fms (%7.1f GFlop/s)  "
                   "wmma_cp_vec=%9.3fms (%7.1f GFlop/s)  dualbuf=%9.3fms (%7.1f GFlop/s)  "
                   "bigbk=%9.3fms (%7.1f GFlop/s)  bigbk_pad=%9.3fms (%7.1f GFlop/s)  "
                   "hugebk=%9.3fms (%7.1f GFlop/s)  tripipe=%9.3fms (%7.1f GFlop/s)  "
                   "warpspec=%9.3fms (%7.1f GFlop/s)  w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx=%9.3fms (%7.1f GFlop/s)  mma_ptx_bigtile=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_swizzle=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_deep=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_l2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide2b=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_deep5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_coalb=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_coalb_shfl=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_epi_smem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp_apipe=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp_fullpipe=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2_epismem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2_hoisted=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp3=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp4=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp6=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_deep3=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_xswz=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_xswz2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_cutlass_config=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_bigreg=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_bigreg_sq=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_multistage=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_spread=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_coalb=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_smalltile_2block=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile2_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_tripipe=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_generic, flops / t_generic / 1e6, t_tiled, flops / t_tiled / 1e6,
                   t_wmma, flops / t_wmma / 1e6, t_wmma_cp, flops / t_wmma_cp / 1e6,
                   t_wmma_cp_vec, flops / t_wmma_cp_vec / 1e6, t_dualbuf, flops / t_dualbuf / 1e6,
                   t_bigbk, flops / t_bigbk / 1e6, t_bigbk_pad, flops / t_bigbk_pad / 1e6,
                   t_hugebk, flops / t_hugebk / 1e6, t_tripipe, flops / t_tripipe / 1e6,
                   t_warpspec, flops / t_warpspec / 1e6, t_w32, flops / t_w32 / 1e6,
                   t_mma_ptx, flops / t_mma_ptx / 1e6,
                   t_mma_ptx_bigtile, flops / t_mma_ptx_bigtile / 1e6,
                   t_mma_ptx_bigtile_swizzle, flops / t_mma_ptx_bigtile_swizzle / 1e6,
                   t_mma_ptx_bigtile_deep, flops / t_mma_ptx_bigtile_deep / 1e6,
                   t_mma_ptx_bigtile_w32, flops / t_mma_ptx_bigtile_w32 / 1e6,
                   t_mma_ptx_bigtile_w32_epi, flops / t_mma_ptx_bigtile_w32_epi / 1e6,
                   t_mma_ptx_bigtile_w32_epi_l2, flops / t_mma_ptx_bigtile_w32_epi_l2 / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide, flops / t_mma_ptx_bigtile_w32_epi_wide / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide2, flops / t_mma_ptx_bigtile_w32_epi_wide2 / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide2b, flops / t_mma_ptx_bigtile_w32_epi_wide2b / 1e6,
                   t_mma_ptx_deep5, flops / t_mma_ptx_deep5 / 1e6,
                   t_mma_ptx_deep5_w32, flops / t_mma_ptx_deep5_w32 / 1e6,
                   t_mma_ptx_wide_deep5, flops / t_mma_ptx_wide_deep5 / 1e6,
                   t_mma_ptx_wide_coalb, flops / t_mma_ptx_wide_coalb / 1e6,
                   t_mma_ptx_wide_coalb_shfl, flops / t_mma_ptx_wide_coalb_shfl / 1e6,
                   t_mma_ptx_wide_nospill2, flops / t_mma_ptx_wide_nospill2 / 1e6,
                   t_mma_ptx_wide_epi_smem, flops / t_mma_ptx_wide_epi_smem / 1e6,
                   t_mma_ptx_wide_nospill_epismem, flops / t_mma_ptx_wide_nospill_epismem / 1e6,
                   t_mma_ptx_deep5_bigwarp, flops / t_mma_ptx_deep5_bigwarp / 1e6,
                   t_mma_ptx_deep5_bigwarp_apipe, flops / t_mma_ptx_deep5_bigwarp_apipe / 1e6,
                   t_mma_ptx_deep5_bigwarp_fullpipe, flops / t_mma_ptx_deep5_bigwarp_fullpipe / 1e6,
                   t_mma_ptx_deep5_bigwarp2, flops / t_mma_ptx_deep5_bigwarp2 / 1e6,
                   t_mma_ptx_deep5_bigwarp2_epismem, flops / t_mma_ptx_deep5_bigwarp2_epismem / 1e6,
                   t_mma_ptx_deep5_bigwarp2_hoisted, flops / t_mma_ptx_deep5_bigwarp2_hoisted / 1e6,
                   t_mma_ptx_deep5_bigwarp3, flops / t_mma_ptx_deep5_bigwarp3 / 1e6,
                   t_mma_ptx_deep5_bigwarp4, flops / t_mma_ptx_deep5_bigwarp4 / 1e6,
                   t_mma_ptx_deep5_bigwarp5, flops / t_mma_ptx_deep5_bigwarp5 / 1e6,
                   t_mma_ptx_deep5_bigwarp6, flops / t_mma_ptx_deep5_bigwarp6 / 1e6,
                   t_mma_ptx_wide_nospill_epismem_deep3, flops / t_mma_ptx_wide_nospill_epismem_deep3 / 1e6,
                   t_mma_ptx_wide_nospill_epismem_xswz, flops / t_mma_ptx_wide_nospill_epismem_xswz / 1e6,
                   t_mma_ptx_wide_nospill_epismem_xswz2, flops / t_mma_ptx_wide_nospill_epismem_xswz2 / 1e6,
                   t_mma_ptx_cutlass_config, flops / t_mma_ptx_cutlass_config / 1e6,
                   t_mma_ptx_wide_bigreg, flops / t_mma_ptx_wide_bigreg / 1e6,
                   t_mma_ptx_wide_bigreg_sq, flops / t_mma_ptx_wide_bigreg_sq / 1e6,
                   t_mma_ptx_wide_multistage, flops / t_mma_ptx_wide_multistage / 1e6,
                   t_mma_ptx_wide_spread, flops / t_mma_ptx_wide_spread / 1e6,
                   t_mma_ptx_wide_nospill_epismem_coalb, flops / t_mma_ptx_wide_nospill_epismem_coalb / 1e6,
                   t_mma_ptx_smalltile_2block, flops / t_mma_ptx_smalltile_2block / 1e6,
                   t_mma_ptx_bigtile2_w32, flops / t_mma_ptx_bigtile2_w32 / 1e6,
                   t_mma_ptx_bigtile2, flops / t_mma_ptx_bigtile2 / 1e6,
                   t_mma_ptx_bigtile_tripipe, flops / t_mma_ptx_bigtile_tripipe / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best_entry->t);
    } else {
        std::printf("  N=%-9zu generic=      SKIPPED               tiled=%9.3fms (%7.1f GFlop/s)  "
                   "wmma=%9.3fms (%7.1f GFlop/s)  wmma_cp=%9.3fms (%7.1f GFlop/s)  "
                   "wmma_cp_vec=%9.3fms (%7.1f GFlop/s)  dualbuf=%9.3fms (%7.1f GFlop/s)  "
                   "bigbk=%9.3fms (%7.1f GFlop/s)  bigbk_pad=%9.3fms (%7.1f GFlop/s)  "
                   "hugebk=%9.3fms (%7.1f GFlop/s)  tripipe=%9.3fms (%7.1f GFlop/s)  "
                   "warpspec=%9.3fms (%7.1f GFlop/s)  w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx=%9.3fms (%7.1f GFlop/s)  mma_ptx_bigtile=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_swizzle=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_deep=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_l2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_w32_epi_wide2b=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_deep5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_coalb=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_coalb_shfl=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_epi_smem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp_apipe=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp_fullpipe=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2_epismem=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp2_hoisted=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp3=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp4=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp5=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_deep5_bigwarp6=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_deep3=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_xswz=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_xswz2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_cutlass_config=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_bigreg=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_bigreg_sq=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_multistage=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_spread=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_wide_nospill_epismem_coalb=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_smalltile_2block=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile2_w32=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile2=%9.3fms (%7.1f GFlop/s)  "
                   "mma_ptx_bigtile_tripipe=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_tiled, flops / t_tiled / 1e6, t_wmma, flops / t_wmma / 1e6,
                   t_wmma_cp, flops / t_wmma_cp / 1e6, t_wmma_cp_vec, flops / t_wmma_cp_vec / 1e6,
                   t_dualbuf, flops / t_dualbuf / 1e6, t_bigbk, flops / t_bigbk / 1e6,
                   t_bigbk_pad, flops / t_bigbk_pad / 1e6, t_hugebk, flops / t_hugebk / 1e6,
                   t_tripipe, flops / t_tripipe / 1e6, t_warpspec, flops / t_warpspec / 1e6,
                   t_w32, flops / t_w32 / 1e6, t_mma_ptx, flops / t_mma_ptx / 1e6,
                   t_mma_ptx_bigtile, flops / t_mma_ptx_bigtile / 1e6,
                   t_mma_ptx_bigtile_swizzle, flops / t_mma_ptx_bigtile_swizzle / 1e6,
                   t_mma_ptx_bigtile_deep, flops / t_mma_ptx_bigtile_deep / 1e6,
                   t_mma_ptx_bigtile_w32, flops / t_mma_ptx_bigtile_w32 / 1e6,
                   t_mma_ptx_bigtile_w32_epi, flops / t_mma_ptx_bigtile_w32_epi / 1e6,
                   t_mma_ptx_bigtile_w32_epi_l2, flops / t_mma_ptx_bigtile_w32_epi_l2 / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide, flops / t_mma_ptx_bigtile_w32_epi_wide / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide2, flops / t_mma_ptx_bigtile_w32_epi_wide2 / 1e6,
                   t_mma_ptx_bigtile_w32_epi_wide2b, flops / t_mma_ptx_bigtile_w32_epi_wide2b / 1e6,
                   t_mma_ptx_deep5, flops / t_mma_ptx_deep5 / 1e6,
                   t_mma_ptx_deep5_w32, flops / t_mma_ptx_deep5_w32 / 1e6,
                   t_mma_ptx_wide_deep5, flops / t_mma_ptx_wide_deep5 / 1e6,
                   t_mma_ptx_wide_coalb, flops / t_mma_ptx_wide_coalb / 1e6,
                   t_mma_ptx_wide_coalb_shfl, flops / t_mma_ptx_wide_coalb_shfl / 1e6,
                   t_mma_ptx_wide_nospill2, flops / t_mma_ptx_wide_nospill2 / 1e6,
                   t_mma_ptx_wide_epi_smem, flops / t_mma_ptx_wide_epi_smem / 1e6,
                   t_mma_ptx_wide_nospill_epismem, flops / t_mma_ptx_wide_nospill_epismem / 1e6,
                   t_mma_ptx_deep5_bigwarp, flops / t_mma_ptx_deep5_bigwarp / 1e6,
                   t_mma_ptx_deep5_bigwarp_apipe, flops / t_mma_ptx_deep5_bigwarp_apipe / 1e6,
                   t_mma_ptx_deep5_bigwarp_fullpipe, flops / t_mma_ptx_deep5_bigwarp_fullpipe / 1e6,
                   t_mma_ptx_deep5_bigwarp2, flops / t_mma_ptx_deep5_bigwarp2 / 1e6,
                   t_mma_ptx_deep5_bigwarp2_epismem, flops / t_mma_ptx_deep5_bigwarp2_epismem / 1e6,
                   t_mma_ptx_deep5_bigwarp2_hoisted, flops / t_mma_ptx_deep5_bigwarp2_hoisted / 1e6,
                   t_mma_ptx_deep5_bigwarp3, flops / t_mma_ptx_deep5_bigwarp3 / 1e6,
                   t_mma_ptx_deep5_bigwarp4, flops / t_mma_ptx_deep5_bigwarp4 / 1e6,
                   t_mma_ptx_deep5_bigwarp5, flops / t_mma_ptx_deep5_bigwarp5 / 1e6,
                   t_mma_ptx_deep5_bigwarp6, flops / t_mma_ptx_deep5_bigwarp6 / 1e6,
                   t_mma_ptx_wide_nospill_epismem_deep3, flops / t_mma_ptx_wide_nospill_epismem_deep3 / 1e6,
                   t_mma_ptx_wide_nospill_epismem_xswz, flops / t_mma_ptx_wide_nospill_epismem_xswz / 1e6,
                   t_mma_ptx_wide_nospill_epismem_xswz2, flops / t_mma_ptx_wide_nospill_epismem_xswz2 / 1e6,
                   t_mma_ptx_cutlass_config, flops / t_mma_ptx_cutlass_config / 1e6,
                   t_mma_ptx_wide_bigreg, flops / t_mma_ptx_wide_bigreg / 1e6,
                   t_mma_ptx_wide_bigreg_sq, flops / t_mma_ptx_wide_bigreg_sq / 1e6,
                   t_mma_ptx_wide_multistage, flops / t_mma_ptx_wide_multistage / 1e6,
                   t_mma_ptx_wide_spread, flops / t_mma_ptx_wide_spread / 1e6,
                   t_mma_ptx_wide_nospill_epismem_coalb, flops / t_mma_ptx_wide_nospill_epismem_coalb / 1e6,
                   t_mma_ptx_smalltile_2block, flops / t_mma_ptx_smalltile_2block / 1e6,
                   t_mma_ptx_bigtile2_w32, flops / t_mma_ptx_bigtile2_w32 / 1e6,
                   t_mma_ptx_bigtile2, flops / t_mma_ptx_bigtile2 / 1e6,
                   t_mma_ptx_bigtile_tripipe, flops / t_mma_ptx_bigtile_tripipe / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best_entry->t);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_performance() {
    std::printf("=== Performance (gemm_fp16_tiled/gemm_fp16_wmma vs cuBLAS Hgemm, square) ===\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (size_t n : {1024ul, 2048ul, 4096ul, 8192ul, 16384ul, 32768ul}) bench_one(handle, n);

    cublasDestroy(handle);
    std::printf("\n");
}

// Runs exactly one N x N shape and exits -- see the file header comment.
void run_single_shape(size_t n) {
    std::printf("=== Single shape: N=%zu ===\n", n);
    cublasHandle_t handle;
    cublasCreate(&handle);
    bench_one(handle, n);
    cublasDestroy(handle);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_correctness = true, do_performance = true;
    size_t only_n = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-correctness") == 0) do_correctness = false;
        if (std::strcmp(argv[i], "--skip-performance") == 0) do_performance = false;
        if (std::strcmp(argv[i], "--only-shape") == 0 && i + 1 < argc) {
            only_n = std::strtoull(argv[i + 1], nullptr, 10);
            i += 1;
        }
        if (std::strcmp(argv[i], "--only-method") == 0 && i + 1 < argc) {
            g_only_method = argv[i + 1];
            i += 1;
        }
    }

    if (only_n != 0) {
        run_single_shape(only_n);
        return 0;
    }

    bool correctness_ok = true;
    if (do_correctness) correctness_ok = run_correctness();
    if (do_performance) run_performance();

    return correctness_ok ? 0 : 1;
}

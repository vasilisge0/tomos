// Standalone correctness + performance verification for the custom
// fp32_16 square-GEMM CUDA kernel(s) in src/kernels/cuda/matrix/fp32_16_
// cuda_kernels.cu: mm_fp32_16_accurate, the naive one-thread-per-output-
// element baseline that reconstructs each A(i,l) from its wh/wr halves
// (fp32_16_combine, see fp32_16_pack.hpp) before multiplying -- i.e. it
// uses BOTH halves to recover full ~fp32 accuracy per element, unlike
// mm_fp32_16_fast/pure_half (wh-only). Output is plain fp32, same as every
// other kernel in this file. mm_fp32_16_tiled is the shared-memory-tiled
// follow-up: same register-blocked structure as gemm_fp32_tiled (see
// mm_cuda_kernels.cu), decoding each wh/wr pair into a float once, when
// it's loaded into shared memory, instead of once per use the way the
// untiled kernel does. mm_fp32_16_tiled_vec adds vectorized wh/wr, B, and
// shared-memory register loads on top of that, mirroring gemm_fp32_tiled_
// cpasync_vec2's biggest wins in the plain-fp32 series. mm_fp32_16_tiled_
// cpasync_vec further prefetches A's raw wh/wr bytes one tile ahead via
// cp.async (decode still happens synchronously -- it can't run inside a
// cp.async transfer -- but the global memory fetch it depends on is
// hidden behind the previous tile's compute). Mirrors verify_mm_fp32_gemm.
// cu's structure and exists for the same reason: a harness to iteratively
// add and tune faster fp32_16 GEMM kernels against cuBLAS's Sgemm as the
// performance ceiling.
//
// As further custom fp32_16 kernels are added, wire them into both
// run_correctness_case() and bench_one() below next to mm_fp32_16_tiled_
// cpasync_vec.
//
// The performance sweep's matrices are generated directly on the device,
// same reasoning as verify_mm_fp32_gemm.cu.
//
// Usage: verify_mm_fp32_16_gemm [--skip-correctness] [--skip-performance]
//                                [--only-shape N]
//
// --only-shape N runs just that one square size and exits, skipping
// correctness and the full sweep -- for isolating a single kernel launch
// under an external profiler (ncu/nsys):
//   scripts/verify_mm_fp32_16_gemm.sh -- --only-shape 8192
//   ncu --kernel-name mm_fp32_16_accurate_kernel --launch-count 1 \
//       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp32_16_gemm" \
//       --only-shape 8192
//
// Exit code is non-zero if any correctness check fails.

#include <cublas_v2.h>
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
#include "fp32_bf16_cuda_kernels.cuh"
#include "fp32_bf16_pack.hpp"
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

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
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

// FP32_BF16 counterpart of gen_fp32_16_kernel/gen_fp32_16 -- same pseudo-
// random source values, split via fp32_bf16_split instead of fp32_16_split.
// A separate wh/wr pair is required: the two packing schemes are bit-
// incompatible (fp32_bf16_split's wh/wr are just the raw upper/lower 16
// bits of the float32 pattern -- see fp32_bf16_pack.hpp -- nothing like
// fp32_16_split's rebiased/control-bit encoding), so mm_fp32_16_*'s own
// wh/wr buffers can't be reused for the mm_fp32_bf16_* kernels below.
__global__ void gen_fp32_bf16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_kernel<<<blocks, threads>>>(n, wh, wr);
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
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        double err = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpu[i]));
        double rel = err / std::max(1.0, std::fabs(static_cast<double>(cpu[i])));
        max_abs_err = std::max(max_abs_err, err);
        max_rel_err = std::max(max_rel_err, rel);
    }
    bool ok = max_rel_err < tol;
    std::printf("  %-40s max_abs_err=%.6g max_rel_err=%.6g %s\n", label.c_str(),
               max_abs_err, max_rel_err, ok ? "OK" : "FAIL");
    return ok;
}

// Host-side reference for mm_fp32_bf16_accurate/mm_fp32_bf16_tiled --
// double-accumulated, reconstructing each A(i,l) via fp32_bf16_combine
// (exact, unconditionally, for every input -- see fp32_bf16_pack.hpp).
// No cpu::mm_fp32_bf16_accurate exists (unlike mm_fp32_16_accurate, which
// has both a CPU and CUDA implementation) -- this codebase's fp32_bf16
// kernels are CUDA-only so far, so this reference is computed inline here
// rather than adding a CPU-side production kernel the request didn't ask
// for.
void mm_fp32_bf16_cpu_reference(double alpha, size_t n, const std::vector<uint16_t>& wh,
                                const std::vector<uint16_t>& wr, const std::vector<float>& B,
                                double beta, std::vector<float>& C) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < n; ++l) {
                size_t idx = i + l * n;
                double a_val = static_cast<double>(fp32_bf16_combine(wh[idx], wr[idx]));
                double b_val = static_cast<double>(B[l + j * n]);
                acc += a_val * b_val;
            }
            size_t c_idx = i + j * n;
            C[c_idx] = static_cast<float>(alpha * acc + beta * static_cast<double>(C[c_idx]));
        }
    }
}

// Host-side reference for mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_
// snake_midpoint_exact_rounded -- double-accumulated, reconstructing
// each A(i,l) via fp32_bf16_combine_rounded (NOT bit-exact for values
// whose bf16 mantissa was actually rounded up by fp32_bf16_split_
// rounded, unlike fp32_bf16_combine/fp32_bf16_split's always-exact
// round-trip -- see fp32_bf16_pack.hpp's own comment on fp32_bf16_
// split_rounded). Same "verify the GPU kernel implements its OWN
// (lossy) decode contract, not the true original A values" philosophy
// as mm_fp32_16_truncated_cpu_reference below.
void mm_fp32_bf16_rounded_cpu_reference(double alpha, size_t n, const std::vector<uint16_t>& wh,
                                        const std::vector<uint16_t>& wr, const std::vector<float>& B,
                                        double beta, std::vector<float>& C) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < n; ++l) {
                size_t idx = i + l * n;
                double a_val = static_cast<double>(fp32_bf16_combine_rounded(wh[idx], wr[idx]));
                double b_val = static_cast<double>(B[l + j * n]);
                acc += a_val * b_val;
            }
            size_t c_idx = i + j * n;
            C[c_idx] = static_cast<float>(alpha * acc + beta * static_cast<double>(C[c_idx]));
        }
    }
}

// Host-side reference for mm_fp32_16_tiled_cpasync_dualbuf_truncated --
// double-accumulated, reconstructing each A(i,l) via fp32_16_combine_
// truncated (never exact for values whose mant_hi was actually truncated
// by fp32_16_split_truncated, unlike fp32_16_combine/fp32_16_split's
// always-exact round-trip -- this reference exists specifically to
// verify the GPU kernel correctly implements its OWN (lossy) decode
// contract, not to check it against the true original A values).
void mm_fp32_16_truncated_cpu_reference(double alpha, size_t n, const std::vector<uint16_t>& wh,
                                        const std::vector<uint16_t>& wr, const std::vector<float>& B,
                                        double beta, std::vector<float>& C) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t l = 0; l < n; ++l) {
                size_t idx = i + l * n;
                double a_val = static_cast<double>(fp32_16_combine_truncated(wh[idx], wr[idx]));
                double b_val = static_cast<double>(B[l + j * n]);
                acc += a_val * b_val;
            }
            size_t c_idx = i + j * n;
            C[c_idx] = static_cast<float>(alpha * acc + beta * static_cast<double>(C[c_idx]));
        }
    }
}

// Runs one N x N square case (NO_TRANSPOSE on both operands) for
// mm_fp32_16_accurate, comparing against the CPU reference
// (cpu::mm_fp32_16_accurate -- same wh/wr reconstruction, done on the
// host). tag identifies the size in output; sizes not a multiple of the
// 32x8 thread block mm_fp32_16_accurate_kernel launches with exercise its
// boundary handling (and any future tiled kernel's tail handling).
bool run_correctness_case(size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    std::vector<float> A(n * n), B(n * n), C_init(n * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(n * n), wr(n * n);
    for (size_t i = 0; i < n * n; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    double alpha = 1.7, beta = 0.3;

    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                             alpha, dim<2>{n, n}, wh.data(), wr.data(), n,
                             dim<2>{n, n}, B.data(), n, beta, C_cpu.data(), n);

    uint16_t *d_wh, *d_wr;
    float *d_B, *d_C;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_wr, wr.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wr, wr.data(), wr.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

    char label[64];
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                  alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                  dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_accurate] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_accurate %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                               alpha, dim<2>{n, n}, d_wh, d_wr, n,
                               dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                   alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                   dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_vec] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                           dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_vec] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                            dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_vec2] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_vec2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec2_branchless(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_vec2_branchless] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_vec2_branchless %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                            dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_vec3] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_vec3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_vec4(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                            dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_vec4] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_vec4 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, dim<2>{n, n}, d_wh, d_wr, n,
                                                dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_warptile] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_warptile %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile2(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_warptile2] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_warptile2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_warptile3(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_warptile3] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_warptile3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_tiled_cpasync_dualbuf(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Reuses the SAME d_wh/d_wr/C_cpu as mm_fp32_16_tiled_cpasync_
        // dualbuf just above -- mm_fp32_16_tiled_cpasync_dualbuf_adaptive
        // is expected to be bit-identical to it (its fast path's decode
        // body is exactly fp32_16_combine's own !control branch), so this
        // checks against the same rounding-aware CPU reference, not a
        // separate one.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_adaptive(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf_adaptive] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf_adaptive %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu as the two blocks above -- also expected
        // bit-identical (the warp-level fast/full split still only ever
        // calls fp32_16_combine or fp32_16_combine_fast_normal, same as
        // mm_fp32_16_tiled_cpasync_dualbuf_adaptive, just voted on at a
        // finer granularity).
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu -- mm_fp32_16_tiled_cpasync_dualbuf_multin
        // still decodes every element via plain fp32_16_combine, just
        // once per K-step instead of once per (K-step, N-tile), so it's
        // expected bit-identical too.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_multin(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf_multin] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf_multin %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu -- mm_fp32_16_gemm_materialize decodes A
        // to plain fp32 up front (still via fp32_16_combine) then calls
        // gemm_fp32_tiled_cpasync_dualbuf, so also expected bit-
        // identical.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_gemm_materialize(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_gemm_materialize] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_gemm_materialize %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu -- mm_fp32_16_gemm_materialize_midpoint_
        // exact decodes A to plain fp32 up front (still via fp32_16_
        // combine) then calls gemm_fp32_tiled_syncpipe4_bk8_warptile_
        // nostride_snake_midpoint_exact instead of gemm_fp32_tiled_
        // cpasync_dualbuf, so also expected bit-identical.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_gemm_materialize_midpoint_exact(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_gemm_materialize_midpoint_exact] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_gemm_materialize_midpoint_exact %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu -- mm_fp32_16_gemm_materialize_midpoint_
        // exact_vec is the vectorized-decode variant of the kernel
        // above (same fp32_16_combine, same materialize strategy, just
        // 4 rows/thread instead of 1), so also expected bit-identical.
        // Since ld_a=n here, this sweep covers the ld_a%4!=0 fallback
        // path (n=1,15,17,33,129) and the vectorized fast path with NO
        // row tail (n=16,64,256 -- chunk_rows=n<4096 is automatically a
        // multiple of 4 whenever n itself is, so ld_a%4==0 here always
        // implies chunk_rows%4==0 too). It does NOT exercise the
        // vectorized kernel's scalar row-tail branch on GPU (that needs
        // either a ragged last chunk, n not a multiple of 4096, or ld_a
        // > n with n%4!=0 -- neither happens in this small/fast sweep)
        // -- that branch's correctness was instead validated via a
        // host-side Python simulation (27 shapes, including ragged/
        // multi-chunk cases) before this kernel was ever compiled.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_gemm_materialize_midpoint_exact_vec(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_gemm_materialize_midpoint_exact_vec] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_vec(C_init.size());
        cudaMemcpy(C_gpu_vec.data(), d_C, C_gpu_vec.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_gemm_materialize_midpoint_exact_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu_vec, 1e-4);
    }
    {
        // Same d_wh/d_wr/C_cpu -- mm_fp32_16_syncpipe4_bk8_warptile_
        // nostride_snake_midpoint_exact is the fully-fused counterpart
        // (decode happens straight into the tile pipeline's shared-
        // memory A buffer, no separate scratch buffer), so also expected
        // bit-identical.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        // mm_fp32_16_tiled_cpasync_dualbuf_truncated needs wh/wr from
        // fp32_16_split_truncated specifically (see its own correctness
        // contract in fp32_16_pack.hpp) -- a fresh pair generated from
        // the SAME A values, plus its own CPU reference computed the
        // same (lossy, never-rounded) way, since this kernel's decode
        // never recovers A(i,l) exactly the way fp32_16_combine does.
        std::vector<uint16_t> wh_trunc(n * n), wr_trunc(n * n);
        for (size_t i = 0; i < n * n; ++i) fp32_16_split_truncated(A[i], wh_trunc[i], wr_trunc[i]);
        std::vector<float> C_cpu_trunc = C_init;
        mm_fp32_16_truncated_cpu_reference(alpha, n, wh_trunc, wr_trunc, B, beta, C_cpu_trunc);

        uint16_t *d_wh_trunc, *d_wr_trunc;
        cudaMalloc(&d_wh_trunc, wh_trunc.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_trunc, wr_trunc.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_trunc, wh_trunc.data(), wh_trunc.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_trunc, wr_trunc.data(), wr_trunc.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_tiled_cpasync_dualbuf_truncated(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh_trunc, d_wr_trunc, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf_truncated] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf_truncated %s", tag);
        all_ok &= check(label, C_cpu_trunc, C_gpu, 1e-4);

        // Same wh_trunc/wr_trunc/C_cpu_trunc -- the fused midpoint_exact-
        // based truncated sibling decodes via the same fp32_16_combine_
        // truncated call, so also expected to match C_cpu_trunc.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh_trunc, d_wr_trunc, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated] %s\n", cudaGetErrorString(e2));
        std::vector<float> C_gpu2(C_init.size());
        cudaMemcpy(C_gpu2.data(), d_C, C_gpu2.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated %s", tag);
        all_ok &= check(label, C_cpu_trunc, C_gpu2, 1e-4);

        cudaFree(d_wh_trunc);
        cudaFree(d_wr_trunc);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf2(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf2] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_tripipe(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_tripipe] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_tripipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf3(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf3] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf4(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf4] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf4 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf5(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf5] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf5 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf6(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf6] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf6 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_splitk(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_splitk] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_splitk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf7(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
            alpha, dim<2>{n, n}, d_wh, d_wr, n, dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_tiled_cpasync_dualbuf7] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_tiled_cpasync_dualbuf7 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }

    // FP32_BF16 kernels: bit-incompatible packing (see gen_fp32_bf16_
    // kernel's own comment), so re-split the SAME underlying A/B/C_init
    // data via fp32_bf16_split into a fresh wh/wr pair, and compare
    // against a fresh reference computed the same way (mm_fp32_bf16_cpu_
    // reference), rather than reusing d_wh/d_wr or C_cpu from above.
    {
        std::vector<uint16_t> wh_bf16(n * n), wr_bf16(n * n);
        for (size_t i = 0; i < n * n; ++i) fp32_bf16_split(A[i], wh_bf16[i], wr_bf16[i]);
        std::vector<float> C_cpu_bf16 = C_init;
        mm_fp32_bf16_cpu_reference(alpha, n, wh_bf16, wr_bf16, B, beta, C_cpu_bf16);

        uint16_t *d_wh_bf16, *d_wr_bf16;
        cudaMalloc(&d_wh_bf16, wh_bf16.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_bf16, wr_bf16.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_bf16, wh_bf16.data(), wh_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_bf16, wr_bf16.data(), wr_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_accurate(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                    alpha, dim<2>{n, n}, d_wh_bf16, d_wr_bf16, n,
                                    dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_accurate] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16a(C_init.size());
        cudaMemcpy(C_gpu_bf16a.data(), d_C, C_gpu_bf16a.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_accurate %s", tag);
        all_ok &= check(label, C_cpu_bf16, C_gpu_bf16a, 1e-4);

        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                 alpha, dim<2>{n, n}, d_wh_bf16, d_wr_bf16, n,
                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_tiled] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16t(C_init.size());
        cudaMemcpy(C_gpu_bf16t.data(), d_C, C_gpu_bf16t.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_tiled %s", tag);
        all_ok &= check(label, C_cpu_bf16, C_gpu_bf16t, 1e-4);

        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16, d_wr_bf16, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_tiled_cpasync_dualbuf] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16d(C_init.size());
        cudaMemcpy(C_gpu_bf16d.data(), d_C, C_gpu_bf16d.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_tiled_cpasync_dualbuf %s", tag);
        all_ok &= check(label, C_cpu_bf16, C_gpu_bf16d, 1e-4);

        // Same wh_bf16/wr_bf16/C_cpu_bf16 -- the fused midpoint_exact-
        // based kernel decodes via the same fp32_bf16_combine call, so
        // also expected to match C_cpu_bf16.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16, d_wr_bf16, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16f(C_init.size());
        cudaMemcpy(C_gpu_bf16f.data(), d_C, C_gpu_bf16f.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact %s", tag);
        all_ok &= check(label, C_cpu_bf16, C_gpu_bf16f, 1e-4);

        cudaFree(d_wh_bf16);
        cudaFree(d_wr_bf16);
    }

    // Rounded-encoding fp32_bf16 kernel: bit-incompatible with the plain
    // fp32_bf16_split output above (round_flag rides in wr's own bit0,
    // overwriting mant_lo's LSB -- see fp32_bf16_pack.hpp's own comment
    // on fp32_bf16_split_rounded), so re-split via fp32_bf16_split_
    // rounded into a fresh wh/wr pair and compare against a fresh
    // reference computed the same (lossy) way (mm_fp32_bf16_rounded_cpu_
    // reference), rather than reusing wh_bf16/wr_bf16/C_cpu_bf16 above.
    {
        std::vector<uint16_t> wh_bf16r(n * n), wr_bf16r(n * n);
        for (size_t i = 0; i < n * n; ++i) fp32_bf16_split_rounded(A[i], wh_bf16r[i], wr_bf16r[i]);
        std::vector<float> C_cpu_bf16r = C_init;
        mm_fp32_bf16_rounded_cpu_reference(alpha, n, wh_bf16r, wr_bf16r, B, beta, C_cpu_bf16r);

        uint16_t *d_wh_bf16r, *d_wr_bf16r;
        cudaMalloc(&d_wh_bf16r, wh_bf16r.size() * sizeof(uint16_t));
        cudaMalloc(&d_wr_bf16r, wr_bf16r.size() * sizeof(uint16_t));
        cudaMemcpy(d_wh_bf16r, wh_bf16r.data(), wh_bf16r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wr_bf16r, wr_bf16r.data(), wr_bf16r.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16r, d_wr_bf16r, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16r(C_init.size());
        cudaMemcpy(C_gpu_bf16r.data(), d_C, C_gpu_bf16r.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded %s", tag);
        all_ok &= check(label, C_cpu_bf16r, C_gpu_bf16r, 1e-4);

        // Same wh_bf16r/wr_bf16r/C_cpu_bf16r -- the materialize-rounded
        // kernel decodes via the same fp32_bf16_combine_rounded call
        // (just in its own separate decode-chunk kernel instead of
        // fused into a GEMM pipeline), so also expected to match
        // C_cpu_bf16r. This is the spill-free alternative to the fused
        // kernel above (see mm_fp32_bf16_gemm_materialize_rounded's own
        // comment in fp32_16_cuda_kernels.cu).
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_bf16_gemm_materialize_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16r, d_wr_bf16r, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_gemm_materialize_rounded] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16rm(C_init.size());
        cudaMemcpy(C_gpu_bf16rm.data(), d_C, C_gpu_bf16rm.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_gemm_materialize_rounded %s", tag);
        all_ok &= check(label, C_cpu_bf16r, C_gpu_bf16rm, 1e-4);

        // Same wh_bf16r/wr_bf16r/C_cpu_bf16r -- the adaptive kernel's
        // per-tile block-wide reduction just decides which of fp32_
        // bf16_combine/fp32_bf16_combine_rounded to use, both of which
        // still reconstruct the SAME values as the plain fused rounded
        // kernel above (verified host-side that the plain decode is an
        // exact substitute whenever round_flag=0 -- see mm_fp32_bf16_
        // syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_
        // adaptive_kernel's own comment), so also expected to match
        // C_cpu_bf16r exactly. This check is what actually confirms the
        // reduction picks the right branch, not just that the two decode
        // functions individually agree.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16r, d_wr_bf16r, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16ra(C_init.size());
        cudaMemcpy(C_gpu_bf16ra.data(), d_C, C_gpu_bf16ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive %s", tag);
        all_ok &= check(label, C_cpu_bf16r, C_gpu_bf16ra, 1e-4);

        // Same wh_bf16r/wr_bf16r/C_cpu_bf16r -- the warp-level adaptive
        // kernel's __any_sync vote decides the same fp32_bf16_combine/
        // fp32_bf16_combine_rounded choice as the block-wide kernel
        // above, just per-warp instead of per-block, so also expected
        // to match C_cpu_bf16r exactly.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                 alpha, dim<2>{n, n}, d_wh_bf16r, d_wr_bf16r, n,
                                                 dim<2>{n, n}, d_B, n, beta, d_C, n);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu_bf16raw(C_init.size());
        cudaMemcpy(C_gpu_bf16raw.data(), d_C, C_gpu_bf16raw.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp %s", tag);
        all_ok &= check(label, C_cpu_bf16r, C_gpu_bf16raw, 1e-4);

        cudaFree(d_wh_bf16r);
        cudaFree(d_wr_bf16r);
    }

    cudaFree(d_wh);
    cudaFree(d_wr);
    cudaFree(d_B);
    cudaFree(d_C);
    return all_ok;
}

bool run_correctness() {
    std::printf("=== Correctness (mm_fp32_16_accurate/tiled/tiled_vec/cpasync_vec/cpasync_vec2[_bl]/vec3/vec4/warptile[2][3]/dualbuf[2][3][4][5][6][7]/tripipe/splitk, mm_fp32_bf16_accurate/tiled vs CPU reference) ===\n");
    bool all_ok = true;

    for (size_t n : {1ul, 15ul, 16ul, 17ul, 33ul, 64ul, 129ul, 256ul})
        all_ok &= run_correctness_case(n, "square");

    std::printf("\n");
    return all_ok;
}

// ---- performance section: mm_fp32_16_accurate vs cuBLAS ----

// Benchmarks one N x N square shape: mm_fp32_16_accurate against cuBLAS's
// Sgemm operating on a fully-materialized fp32 A (via combine_dense_fp32_16
// -- so the gap measures the cost of reconstructing fp32 from wh/wr, not a
// different GEMM algorithm). Pulled out to a standalone function so
// run_single_shape() can also call it, for isolating one shape's kernel
// launches under an external profiler.
void bench_one(cublasHandle_t handle, size_t n) {
    uint16_t *wh, *wr;
    float *A_plain, *B, *C;
    if (cudaMalloc(&wh, n * n * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&wr, n * n * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&A_plain, n * n * sizeof(float)) != cudaSuccess) {
        std::printf("  N=%-9zu  SKIPPED (out of device memory)\n", n);
        cudaGetLastError();
        return;
    }
    gen_fp32_16(n * n, wh, wr);
    cuda::combine_dense_fp32_16(dim<2>{n, n}, wh, wr, n, A_plain, n);
    cudaMalloc(&B, n * n * sizeof(float));
    cudaMalloc(&C, n * n * sizeof(float));
    gen_float(n * n, B);
    cudaMemset(C, 0, n * n * sizeof(float));
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

    int reps = 5;
    // mm_fp32_16_accurate's throughput is flat (~148 GFlop/s, confirmed
    // across N=1024..16384 -- see the default sweep below), so at large N
    // timing it again just burns minutes for a number we already know.
    // Skip it past this threshold rather than paying for a call that can
    // take on the order of 8 minutes on its own; mm_fp32_16_tiled (the
    // kernel actually worth measuring at these sizes) still runs.
    constexpr size_t kSkipAccurateAbove = 16384;
    double t_accurate = n <= kSkipAccurateAbove
                            ? time_call(
                                  [&] {
                                      cuda::mm_fp32_16_accurate(
                                          TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
                                  },
                                  reps)
                            : 0.0;
    // Only the current-best lineage is timed here now that the kernel
    // count has grown large -- mm_fp32_16_tiled/tiled_vec/cpasync_vec/
    // cpasync_vec2_branchless/cpasync_vec3/cpasync_vec4 were each
    // superseded (see their own comments for why); their numbers are in
    // this conversation's history, not worth re-measuring every run.
    double t_cpasync_vec2 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // warptile (the 32x64-warp-tile shape) was itself superseded by
    // warptile2 (64x32) -- see that kernel's comment. Not re-timed here
    // for the same reason cpasync_vec/tiled/etc. were dropped above.
    double t_warptile2 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_warptile2(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // warptile3 (an even more extreme warp/lane shape) plateaued exactly
    // at warptile2's numbers -- see that kernel's comment. Not re-timed.
    double t_dualbuf = time_call(
        [&] {
            cuda::mm_fp32_16_tiled_cpasync_dualbuf(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // dualbuf2 (BK doubled to 32) regressed sharply -- see that kernel's
    // comment; the shared-memory-driven occupancy loss outweighed the
    // wider B coalescing it was chasing. Not re-timed here.
    double t_tripipe = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_tripipe(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // The inverse of dualbuf2: BK shrunk to 8 instead of grown to 32 --
    // see that kernel's comment for what this tests.
    double t_dualbuf3 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf3(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // BM shrunk to 64 (from dualbuf's 128) to fix small-N SM
    // underutilization -- see that kernel's comment for what this tests.
    double t_dualbuf4 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf4(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // The mirror of dualbuf4: BN shrunk to 64 instead of BM -- see that
    // kernel's comment for what this tests.
    double t_dualbuf5 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf5(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // dualbuf's exact tile shape, but with the inner k-loop's shared->
    // register load software-double-buffered -- see that kernel's
    // comment for what this tests.
    double t_dualbuf6 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf6(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // Split-K on dualbuf's exact tile -- see that kernel's comment for
    // what this tests.
    double t_splitk = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_splitk(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    // dualbuf, but B loaded synchronously (vec3-style) instead of via
    // async staging -- see that kernel's comment for what this tests.
    double t_dualbuf7 = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf7(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mm_strided<float, float, float>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                {n, n}, A_plain, n, {n, n}, B, n, 0.0, C, n, handle);
        },
        reps);

    double flops = 2.0 * n * n * n;
    struct NamedTime { const char* name; double t; };
    NamedTime candidates[] = {
        {"dualbuf", t_dualbuf}, {"tripipe", t_tripipe}, {"dualbuf3", t_dualbuf3},
        {"dualbuf4", t_dualbuf4}, {"dualbuf5", t_dualbuf5}, {"dualbuf6", t_dualbuf6},
        {"splitk", t_splitk}, {"dualbuf7", t_dualbuf7},
    };
    const NamedTime* best_entry = &candidates[0];
    for (const auto& c : candidates)
        if (c.t < best_entry->t) best_entry = &c;
    double best = best_entry->t;
    if (n <= kSkipAccurateAbove) {
        std::printf("  N=%-9zu accurate=%9.3fms (%7.1f GFlop/s)  cpasync_vec2=%9.3fms (%7.1f GFlop/s)  "
                   "warptile2=%9.3fms (%7.1f GFlop/s)  dualbuf=%9.3fms (%7.1f GFlop/s)  "
                   "tripipe=%9.3fms (%7.1f GFlop/s)  dualbuf3=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf4=%9.3fms (%7.1f GFlop/s)  dualbuf5=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf6=%9.3fms (%7.1f GFlop/s)  splitk=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf7=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_accurate, flops / t_accurate / 1e6, t_cpasync_vec2, flops / t_cpasync_vec2 / 1e6,
                   t_warptile2, flops / t_warptile2 / 1e6, t_dualbuf, flops / t_dualbuf / 1e6,
                   t_tripipe, flops / t_tripipe / 1e6, t_dualbuf3, flops / t_dualbuf3 / 1e6,
                   t_dualbuf4, flops / t_dualbuf4 / 1e6, t_dualbuf5, flops / t_dualbuf5 / 1e6,
                   t_dualbuf6, flops / t_dualbuf6 / 1e6, t_splitk, flops / t_splitk / 1e6,
                   t_dualbuf7, flops / t_dualbuf7 / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best);
    } else {
        std::printf("  N=%-9zu accurate=      SKIPPED (flat ~148 GFlop/s)  cpasync_vec2=%9.3fms (%7.1f GFlop/s)  "
                   "warptile2=%9.3fms (%7.1f GFlop/s)  dualbuf=%9.3fms (%7.1f GFlop/s)  "
                   "tripipe=%9.3fms (%7.1f GFlop/s)  dualbuf3=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf4=%9.3fms (%7.1f GFlop/s)  dualbuf5=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf6=%9.3fms (%7.1f GFlop/s)  splitk=%9.3fms (%7.1f GFlop/s)  "
                   "dualbuf7=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_cpasync_vec2, flops / t_cpasync_vec2 / 1e6, t_warptile2, flops / t_warptile2 / 1e6,
                   t_dualbuf, flops / t_dualbuf / 1e6, t_tripipe, flops / t_tripipe / 1e6,
                   t_dualbuf3, flops / t_dualbuf3 / 1e6, t_dualbuf4, flops / t_dualbuf4 / 1e6,
                   t_dualbuf5, flops / t_dualbuf5 / 1e6, t_dualbuf6, flops / t_dualbuf6 / 1e6,
                   t_splitk, flops / t_splitk / 1e6, t_dualbuf7, flops / t_dualbuf7 / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best);
    }

    cudaFree(wh);
    cudaFree(wr);
    cudaFree(A_plain);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Materializes a plain float32 array from an FP32_BF16 wh/wr pair --
// fp32_bf16 counterpart of cuda::combine_dense_fp32_16, kept local to this
// benchmark file (only needed here, to give cuBLAS Sgemm a fully-
// materialized A for a fair comparison) rather than added as a new
// production kernel the request didn't ask for.
__global__ void materialize_fp32_bf16_kernel(size_t n, const uint16_t* wh, const uint16_t* wr, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_bf16_combine(wh[i], wr[i]);
}

// Benchmarks mm_fp32_bf16_tiled against cuBLAS's Sgemm operating on a
// fully-materialized fp32 A (via materialize_fp32_bf16_kernel above) --
// same "isolate the reconstruction cost" reasoning bench_one's own
// comment gives for mm_fp32_16_accurate/combine_dense_fp32_16. Kept
// separate from bench_one rather than folded into its own already-large
// multi-kernel timing/printf structure -- see bench_one's own comment on
// why only the current-best fp32_16 lineage is timed there; adding a
// bit-incompatible packing scheme's kernel into the middle of that would
// risk a subtle mismatch in its dual-path printf format strings for no
// benefit, since fp32_bf16_tiled has nothing to compare against within
// that lineage anyway.
void bench_fp32_bf16(cublasHandle_t handle, size_t n) {
    uint16_t *wh, *wr;
    float *A_plain, *B, *C;
    if (cudaMalloc(&wh, n * n * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&wr, n * n * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&A_plain, n * n * sizeof(float)) != cudaSuccess) {
        std::printf("  N=%-9zu  SKIPPED (out of device memory)\n", n);
        cudaGetLastError();
        return;
    }
    gen_fp32_bf16(n * n, wh, wr);
    {
        int threads = 256;
        int blocks = static_cast<int>((n * n + threads - 1) / threads);
        materialize_fp32_bf16_kernel<<<blocks, threads>>>(n * n, wh, wr, A_plain);
    }
    cudaMalloc(&B, n * n * sizeof(float));
    cudaMalloc(&C, n * n * sizeof(float));
    gen_float(n * n, B);
    cudaMemset(C, 0, n * n * sizeof(float));

    // Same-run FP32_16 dualbuf comparison: a separate wh16/wr16 pair (bit-
    // incompatible packing, same reasoning as run_correctness_case's own
    // fp32_bf16 block), same B/C, so the two "best tiled" numbers share
    // this call's own GPU clock/thermal state instead of requiring a
    // scroll back to a different section's own separate run.
    uint16_t *wh16, *wr16;
    cudaMalloc(&wh16, n * n * sizeof(uint16_t));
    cudaMalloc(&wr16, n * n * sizeof(uint16_t));
    gen_fp32_16(n * n, wh16, wr16);
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

    int reps = 5;
    double t_tiled = time_call(
        [&] {
            cuda::mm_fp32_bf16_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                     1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_dualbuf = time_call(
        [&] {
            cuda::mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     1.0, {n, n}, wh, wr, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_dualbuf16 = time_call(
        [&] {
            cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, {n, n}, wh16, wr16, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mm_strided<float, float, float>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                {n, n}, A_plain, n, {n, n}, B, n, 0.0, C, n, handle);
        },
        reps);

    double flops = 2.0 * n * n * n;
    std::printf("  N=%-9zu mm_fp32_bf16_tiled=%9.3fms (%7.1f GFlop/s)  fp32_bf16_dualbuf=%9.3fms (%7.1f GFlop/s)  "
               "fp32_16_dualbuf=%9.3fms (%7.1f GFlop/s)  cublas=%9.3fms (%7.1f GFlop/s)  "
               "%.3fx cublas (fp32_bf16 dualbuf)  %.3fx fp32_16_dualbuf (fp32_bf16 dualbuf)\n",
               n, t_tiled, flops / t_tiled / 1e6, t_dualbuf, flops / t_dualbuf / 1e6, t_dualbuf16,
               flops / t_dualbuf16 / 1e6, t_cublas, flops / t_cublas / 1e6,
               (flops / t_dualbuf) / (flops / t_cublas), (flops / t_dualbuf) / (flops / t_dualbuf16));

    cudaFree(wh);
    cudaFree(wr);
    cudaFree(wh16);
    cudaFree(wr16);
    cudaFree(A_plain);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_performance() {
    std::printf("=== Performance (mm_fp32_16_accurate/cpasync_vec2/warptile2/dualbuf/tripipe/dualbuf3/dualbuf4/dualbuf5/dualbuf6/splitk/dualbuf7 vs cuBLAS Sgemm, square -- see git log for superseded kernels) ===\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    // N=32768 is back in the default sweep now that mm_fp32_16_tiled exists
    // (~30x faster than mm_fp32_16_accurate -- see the file header) and
    // bench_one skips timing mm_fp32_16_accurate above kSkipAccurateAbove,
    // so this size no longer costs the ~8 minutes it used to.
    for (size_t n : {1024ul, 2048ul, 4096ul, 8192ul, 16384ul, 32768ul}) bench_one(handle, n);

    std::printf("\n=== Performance (mm_fp32_bf16_tiled/tiled_cpasync_dualbuf vs cuBLAS Sgemm, square) ===\n");
    for (size_t n : {1024ul, 2048ul, 4096ul, 8192ul, 16384ul, 32768ul}) bench_fp32_bf16(handle, n);

    cublasDestroy(handle);
    std::printf("\n");
}

// Runs exactly one N x N shape and exits -- see the file header comment.
void run_single_shape(size_t n) {
    std::printf("=== Single shape: N=%zu ===\n", n);
    cublasHandle_t handle;
    cublasCreate(&handle);
    bench_one(handle, n);
    bench_fp32_bf16(handle, n);
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

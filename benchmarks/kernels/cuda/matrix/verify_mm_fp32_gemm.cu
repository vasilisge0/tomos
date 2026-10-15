// Standalone correctness + performance verification for the custom fp32
// square-GEMM CUDA kernel(s) in src/kernels/cuda/matrix/mm_cuda_kernels.cu:
// gemm_generic_kernel (the naive one-thread-per-output-element baseline),
// gemm_fp32_tiled (shared-memory-tiled, register-blocked), gemm_fp32_tiled_db
// (the same tiling, manually double-buffered against the next K-slice),
// gemm_fp32_tiled_cpasync (same double-buffering, but via sm_80's cp.async
// instead of a register round-trip), gemm_fp32_tiled_cpasync_vec (same
// cp.async, plus float4-vectorized shared-memory reads and A's load),
// gemm_fp32_tiled_cpasync_vec2 (same, plus a vectorized load for B too), and
// gemm_fp32_tiled_cpasync_vec3 (same, with the K-tile doubled from 8 to 16
// for wider coalescing on B's load). Used as the harness for iteratively
// adding and tuning faster kernels against cuBLAS's Sgemm as the
// performance ceiling.
//
// As new custom kernels are added (e.g. shared-memory tiled variants), wire
// them into both run_correctness_case() and bench_one() below next to
// gemm_generic_kernel so every new kernel gets its own correctness check
// (including the non-multiple-of-tile-size boundary sizes already covered
// here) before its performance numbers are trusted.
//
// The performance sweep's matrices are generated directly on the device
// (not read from disk or built up on the host) so it scales to large square
// sizes even on a host with a constrained memory cgroup -- point it at
// whatever GPU is available (e.g. an A100 in an interactive session) for the
// fullest picture; it will also just skip individual sizes that don't fit
// in whatever device memory is actually free at the time.
//
// Usage: verify_mm_fp32_gemm [--skip-correctness] [--skip-performance]
//                             [--only-shape N]
//
// --only-shape N runs just that one square size (see run_single_shape) and
// exits, skipping correctness and the full sweep -- for isolating a single
// kernel launch under an external profiler (ncu/nsys), where "the Nth
// launch of kernel X" would otherwise be ambiguous to target across the
// full sweep:
//   scripts/verify_mm_fp32_gemm.sh -- --only-shape 8192
//   ncu --kernel-name gemm_generic_kernel --launch-count 1 \
//       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp32_gemm" \
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

__global__ void gen_float_kernel(size_t n, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand(i);
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

// Runs one N x N square case (NO_TRANSPOSE on both operands) for every
// custom kernel plus cuBLAS, comparing against the CPU reference
// (cpu::mm_strided<float,float,float>). tag identifies the size in output
// (e.g. non-multiple-of-16 sizes exercise gemm_generic_kernel's boundary
// handling in the 16x16-thread-block grid).
bool run_correctness_case(size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool all_ok = true;

    std::vector<float> A(n * n), B(n * n), C_init(n * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    double alpha = 1.7, beta = 0.3;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.size() * sizeof(float));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> C_cpu = C_init;
    cpu::mm_strided<float, float, float>(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
        dim<2>{n, n}, A.data(), n, dim<2>{n, n}, B.data(), n, beta,
        C_cpu.data(), n);

    char label[64];

    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (n + 15) / 16);
        cuda::experimental::gemm_generic_kernel<float, float, float><<<blocks, threads>>>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_generic_kernel] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_generic_kernel %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_db(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                 alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_db] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_db %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync_vec] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync_vec2] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync_vec2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                           alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync_vec3] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync_vec3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync_dualbuf] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync_dualbuf %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf_regpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_cpasync_dualbuf_regpipe] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_cpasync_dualbuf_regpipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_dualbuf_syncpipe] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_dualbuf_syncpipe %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_dualbuf_syncpipe_incptr] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_dualbuf_syncpipe_incptr %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_lean] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_lean %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_lean2] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_lean2 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_nostride] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_nostride %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                             alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_kremain] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_kremain %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                             alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_finegrain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_finegrain] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_finegrain %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_dualacc(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_dualacc] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_dualacc %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_snake] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_snake %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe3_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe3_bk8] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe3_bk8 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8 %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                     alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    alpha, n, n, n, d_A, n, d_B, n, beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
    }
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_strided<float, float, float>(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
            dim<2>{n, n}, d_A, n, dim<2>{n, n}, d_B, n, beta, d_C, n, handle);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "cublasSgemm %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 1e-4);
        cublasDestroy(handle);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return all_ok;
}

bool run_correctness() {
    std::printf("=== Correctness (custom kernels vs CPU reference) ===\n");
    bool all_ok = true;

    // Small sizes, including several not a multiple of the 16x16 thread
    // block gemm_generic_kernel launches with -- exercises its boundary
    // handling (and any future tiled kernel's tail handling) up front.
    for (size_t n : {1ul, 15ul, 16ul, 17ul, 33ul, 64ul, 129ul, 256ul})
        all_ok &= run_correctness_case(n, "square");

    std::printf("\n");
    return all_ok;
}

// ---- performance section: our kernels vs cuBLAS ----

// Benchmarks one N x N square shape: every custom kernel plus cuBLAS's
// Sgemm, against the same device-generated data. Pulled out to a standalone
// function (rather than kept as a local lambda in run_performance()) so
// run_single_shape() can also call it -- letting a caller isolate exactly
// one shape's kernel launches for external profiling.
void bench_one(cublasHandle_t handle, size_t n) {
    float *A, *B, *C;
    if (cudaMalloc(&A, n * n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&B, n * n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&C, n * n * sizeof(float)) != cudaSuccess) {
        std::printf("  N=%-9zu  SKIPPED (out of device memory)\n", n);
        cudaGetLastError();
        return;
    }
    gen_float(n * n, A);
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

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    int reps = 5;
    double t_generic = time_call(
        [&] {
            cuda::experimental::gemm_generic_kernel<float, float, float><<<blocks, threads>>>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_tiled = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_tiled_db = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_db(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                     1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_tiled_cpasync = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                          1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cpasync_vec = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                              1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cpasync_vec2 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync_vec2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cpasync_vec3 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync_vec3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                               1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cpasync_dualbuf = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cpasync_dualbuf_regpipe = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf_regpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_dualbuf_syncpipe = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe_incptr = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                          1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                   1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_lean = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_lean2 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_nostride = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                            1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_nostride_kremain = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_nostride_bcol = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                 1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_kremain = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                            1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_dualbuf_syncpipe_incptr_nostride = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_incptr_nostride_lb1 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                 1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_finegrain = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_finegrain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_dualacc = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_dualacc(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_snake = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe3_bk8 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe3_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8 = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                         1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                  1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_interleaved = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_midpoint = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline = time_call(
        [&] {
            cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = time_call(
        [&] {
            cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.0, n, n, n, A, n, B, n, 0.0, C, n);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mm_strided<float, float, float>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                {n, n}, A, n, {n, n}, B, n, 0.0, C, n, handle);
        },
        reps);

    double flops = 2.0 * n * n * n;
    std::printf("  N=%-9zu generic=%9.3fms (%7.1f GFlop/s)  tiled=%9.3fms (%7.1f GFlop/s)  "
               "tiled_db=%9.3fms (%7.1f GFlop/s)  cpasync=%9.3fms (%7.1f GFlop/s)  "
               "cpasync_vec=%9.3fms (%7.1f GFlop/s)  cpasync_vec2=%9.3fms (%7.1f GFlop/s)  "
               "cpasync_vec3=%9.3fms (%7.1f GFlop/s)  cpasync_dualbuf=%9.3fms (%7.1f GFlop/s)  "
               "cpasync_dualbuf_regpipe=%9.3fms (%7.1f GFlop/s)  "
               "dualbuf_syncpipe=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe_incptr=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_lean=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_lean2=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_nostride=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_nostride_kremain=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_nostride_bcol=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_kremain=%9.3fms (%7.1f GFlop/s)  "
               "dualbuf_syncpipe_incptr_nostride=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_incptr_nostride_lb1=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_finegrain=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_dualacc=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_snake=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe3_bk8=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_interleaved=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_interleaved_late=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_midpoint=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_midpoint_inline=%9.3fms (%7.1f GFlop/s)  "
               "syncpipe4_bk8_warptile_nostride_snake_midpoint_exact=%9.3fms (%7.1f GFlop/s)  "
               "cublas=%9.3fms (%7.1f GFlop/s)  "
               "cpasync_vec3/cublas=%.3f  cpasync_dualbuf/cublas=%.3f  cpasync_dualbuf/cpasync_vec3=%.3f  "
               "regpipe/dualbuf=%.3f  syncpipe/dualbuf=%.3f  syncpipe3/dualbuf=%.3f  syncpipe3/syncpipe=%.3f  "
               "incptr/syncpipe3=%.3f  syncpipe_incptr/syncpipe=%.3f  incptr_lean/incptr=%.3f  "
               "incptr_lean/syncpipe3=%.3f  incptr_lean2/incptr_lean=%.3f  incptr_lean2/syncpipe3=%.3f  "
               "nostride/incptr=%.3f  nostride/incptr_lean=%.3f  nostride/syncpipe3=%.3f  "
               "kremain/nostride=%.3f  kremain/syncpipe3=%.3f  bcol/nostride=%.3f  bcol/syncpipe3=%.3f  "
               "kremain_alone/lean=%.3f  kremain_alone/syncpipe3=%.3f  kremain_alone/nostride_kremain=%.3f  "
               "2stage_nostride/syncpipe_incptr=%.3f  2stage_nostride/nostride=%.3f  2stage_nostride/dualbuf=%.3f  "
               "lb1/nostride=%.3f  lb1/syncpipe3=%.3f  lb1/cublas=%.3f  "
               "finegrain/syncpipe3=%.3f  finegrain/cublas=%.3f  "
               "dualacc/syncpipe3=%.3f  dualacc/cublas=%.3f  "
               "snake/syncpipe3=%.3f  snake/cublas=%.3f  "
               "bk8/syncpipe3=%.3f  bk8/cublas=%.3f  "
               "syncpipe4_bk8/syncpipe3_bk8=%.3f  syncpipe4_bk8/syncpipe3=%.3f  syncpipe4_bk8/cublas=%.3f  "
               "warptile/syncpipe4_bk8=%.3f  warptile/syncpipe3=%.3f  warptile/cublas=%.3f  "
               "wtnostride/warptile=%.3f  wtnostride/syncpipe4_bk8=%.3f  wtnostride/syncpipe3=%.3f  wtnostride/cublas=%.3f  "
               "wtnsnake/wtnostride=%.3f  wtnsnake/syncpipe3=%.3f  wtnsnake/cublas=%.3f  "
               "wtninter/wtnsnake=%.3f  wtninter/syncpipe3=%.3f  wtninter/cublas=%.3f  "
               "wtninterlate/wtninter=%.3f  wtninterlate/wtnsnake=%.3f  wtninterlate/syncpipe3=%.3f  wtninterlate/cublas=%.3f  "
               "wtnmid/wtnsnake=%.3f  wtnmid/syncpipe3=%.3f  wtnmid/cublas=%.3f  "
               "wtnmiduncapped/wtnmid=%.3f  wtnmiduncapped/wtnsnake=%.3f  wtnmiduncapped/syncpipe3=%.3f  wtnmiduncapped/cublas=%.3f  "
               "wtnmidnarrow/wtnmid=%.3f  wtnmidnarrow/wtnsnake=%.3f  wtnmidnarrow/syncpipe3=%.3f  wtnmidnarrow/cublas=%.3f  "
               "wtnmidinline/wtnmid=%.3f  wtnmidinline/wtnsnake=%.3f  wtnmidinline/syncpipe3=%.3f  wtnmidinline/cublas=%.3f  "
               "wtnmidexact/wtnmid=%.3f  wtnmidexact/wtnsnake=%.3f  wtnmidexact/syncpipe3=%.3f  wtnmidexact/cublas=%.3f\n",
               n, t_generic, flops / t_generic / 1e6, t_tiled, flops / t_tiled / 1e6,
               t_tiled_db, flops / t_tiled_db / 1e6, t_tiled_cpasync, flops / t_tiled_cpasync / 1e6,
               t_cpasync_vec, flops / t_cpasync_vec / 1e6, t_cpasync_vec2, flops / t_cpasync_vec2 / 1e6,
               t_cpasync_vec3, flops / t_cpasync_vec3 / 1e6, t_cpasync_dualbuf, flops / t_cpasync_dualbuf / 1e6,
               t_cpasync_dualbuf_regpipe, flops / t_cpasync_dualbuf_regpipe / 1e6,
               t_dualbuf_syncpipe, flops / t_dualbuf_syncpipe / 1e6,
               t_syncpipe_incptr, flops / t_syncpipe_incptr / 1e6,
               t_syncpipe3, flops / t_syncpipe3 / 1e6,
               t_syncpipe3_incptr, flops / t_syncpipe3_incptr / 1e6,
               t_syncpipe3_incptr_lean, flops / t_syncpipe3_incptr_lean / 1e6,
               t_syncpipe3_incptr_lean2, flops / t_syncpipe3_incptr_lean2 / 1e6,
               t_syncpipe3_incptr_nostride, flops / t_syncpipe3_incptr_nostride / 1e6,
               t_syncpipe3_incptr_nostride_kremain, flops / t_syncpipe3_incptr_nostride_kremain / 1e6,
               t_syncpipe3_incptr_nostride_bcol, flops / t_syncpipe3_incptr_nostride_bcol / 1e6,
               t_syncpipe3_incptr_kremain, flops / t_syncpipe3_incptr_kremain / 1e6,
               t_dualbuf_syncpipe_incptr_nostride, flops / t_dualbuf_syncpipe_incptr_nostride / 1e6,
               t_syncpipe3_incptr_nostride_lb1, flops / t_syncpipe3_incptr_nostride_lb1 / 1e6,
               t_syncpipe3_finegrain, flops / t_syncpipe3_finegrain / 1e6,
               t_syncpipe3_dualacc, flops / t_syncpipe3_dualacc / 1e6,
               t_syncpipe3_snake, flops / t_syncpipe3_snake / 1e6,
               t_syncpipe3_bk8, flops / t_syncpipe3_bk8 / 1e6,
               t_syncpipe4_bk8, flops / t_syncpipe4_bk8 / 1e6,
               t_syncpipe4_bk8_warptile, flops / t_syncpipe4_bk8_warptile / 1e6,
               t_syncpipe4_bk8_warptile_nostride, flops / t_syncpipe4_bk8_warptile_nostride / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake, flops / t_syncpipe4_bk8_warptile_nostride_snake / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_interleaved, flops / t_syncpipe4_bk8_warptile_nostride_snake_interleaved / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late, flops / t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint, flops / t_syncpipe4_bk8_warptile_nostride_snake_midpoint / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped, flops / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow, flops / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline, flops / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline / 1e6,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact, flops / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact / 1e6,
               t_cublas, flops / t_cublas / 1e6,
               t_cublas / t_cpasync_vec3, t_cublas / t_cpasync_dualbuf, t_cpasync_vec3 / t_cpasync_dualbuf,
               t_cpasync_dualbuf / t_cpasync_dualbuf_regpipe, t_cpasync_dualbuf / t_dualbuf_syncpipe,
               t_cpasync_dualbuf / t_syncpipe3, t_dualbuf_syncpipe / t_syncpipe3,
               t_syncpipe3 / t_syncpipe3_incptr, t_dualbuf_syncpipe / t_syncpipe_incptr,
               t_syncpipe3_incptr / t_syncpipe3_incptr_lean, t_syncpipe3 / t_syncpipe3_incptr_lean,
               t_syncpipe3_incptr_lean / t_syncpipe3_incptr_lean2, t_syncpipe3 / t_syncpipe3_incptr_lean2,
               t_syncpipe3_incptr / t_syncpipe3_incptr_nostride, t_syncpipe3_incptr_lean / t_syncpipe3_incptr_nostride,
               t_syncpipe3 / t_syncpipe3_incptr_nostride,
               t_syncpipe3_incptr_nostride / t_syncpipe3_incptr_nostride_kremain,
               t_syncpipe3 / t_syncpipe3_incptr_nostride_kremain,
               t_syncpipe3_incptr_nostride / t_syncpipe3_incptr_nostride_bcol,
               t_syncpipe3 / t_syncpipe3_incptr_nostride_bcol,
               t_syncpipe3_incptr_lean / t_syncpipe3_incptr_kremain,
               t_syncpipe3 / t_syncpipe3_incptr_kremain,
               t_syncpipe3_incptr_nostride_kremain / t_syncpipe3_incptr_kremain,
               t_syncpipe_incptr / t_dualbuf_syncpipe_incptr_nostride,
               t_syncpipe3_incptr_nostride / t_dualbuf_syncpipe_incptr_nostride,
               t_dualbuf_syncpipe / t_dualbuf_syncpipe_incptr_nostride,
               t_syncpipe3_incptr_nostride / t_syncpipe3_incptr_nostride_lb1,
               t_syncpipe3 / t_syncpipe3_incptr_nostride_lb1,
               t_cublas / t_syncpipe3_incptr_nostride_lb1,
               t_syncpipe3 / t_syncpipe3_finegrain,
               t_cublas / t_syncpipe3_finegrain,
               t_syncpipe3 / t_syncpipe3_dualacc,
               t_cublas / t_syncpipe3_dualacc,
               t_syncpipe3 / t_syncpipe3_snake,
               t_cublas / t_syncpipe3_snake,
               t_syncpipe3 / t_syncpipe3_bk8,
               t_cublas / t_syncpipe3_bk8,
               t_syncpipe3_bk8 / t_syncpipe4_bk8,
               t_syncpipe3 / t_syncpipe4_bk8,
               t_cublas / t_syncpipe4_bk8,
               t_syncpipe4_bk8 / t_syncpipe4_bk8_warptile,
               t_syncpipe3 / t_syncpipe4_bk8_warptile,
               t_cublas / t_syncpipe4_bk8_warptile,
               t_syncpipe4_bk8_warptile / t_syncpipe4_bk8_warptile_nostride,
               t_syncpipe4_bk8 / t_syncpipe4_bk8_warptile_nostride,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride,
               t_cublas / t_syncpipe4_bk8_warptile_nostride,
               t_syncpipe4_bk8_warptile_nostride / t_syncpipe4_bk8_warptile_nostride_snake,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_interleaved,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_interleaved,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_interleaved,
               t_syncpipe4_bk8_warptile_nostride_snake_interleaved / t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_interleaved_late,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_midpoint,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_midpoint,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_midpoint,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline,
               t_syncpipe4_bk8_warptile_nostride_snake_midpoint / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact,
               t_syncpipe4_bk8_warptile_nostride_snake / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact,
               t_syncpipe3 / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact,
               t_cublas / t_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_performance() {
    std::printf("=== Performance (custom kernels vs cuBLAS Sgemm, square fp32) ===\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Matches bench_mm_fp32_square_cuda_kernels.cu's SquareFp32Args full
    // ladder, topping out at 32768 (~12.9GB across A+B+C in fp32 -- fine on
    // a 40GB+ card, but a single call there can still run into the minutes
    // on a slower GPU with today's naive gemm_generic_kernel, ~10% of
    // cuBLAS). --only-shape 32768 isolates just that point if the full
    // sweep is too slow on a given machine.
    for (size_t n : {1024ul, 2048ul, 4096ul, 8192ul, 16384ul, 32768ul}) bench_one(handle, n);

    cublasDestroy(handle);
    std::printf("\n");
}

// Runs exactly one N x N shape and exits -- for isolating a single kernel
// launch under an external profiler (ncu/nsys), where "the Nth launch of
// kernel X" would otherwise be ambiguous to target across the full sweep in
// run_performance().
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
    }

    if (only_n != 0) {
        // Single-shape mode for external profiling -- see run_single_shape's
        // comment. Skips correctness and the full sweep entirely.
        run_single_shape(only_n);
        return 0;
    }

    bool correctness_ok = true;
    if (do_correctness) correctness_ok = run_correctness();
    if (do_performance) run_performance();

    return correctness_ok ? 0 : 1;
}

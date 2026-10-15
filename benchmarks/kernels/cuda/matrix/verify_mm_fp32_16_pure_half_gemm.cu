// Standalone correctness + performance verification for the custom
// fp32_16 "pure_half" square-GEMM CUDA kernel(s) in src/kernels/cuda/
// matrix/fp32_16_cuda_kernels.cu: mm_fp32_16_pure_half, the naive one-
// thread-per-output-element baseline that decodes each A(i,l) from its wh
// half only (fp32_16_half_only, see fp32_16_pack.hpp -- wr is never
// touched) and runs the ENTIRE multiply-accumulate reduction in __half
// arithmetic, i.e. genuine fp16-only compute, not merely fp16 storage.
// mm_fp32_16_pure_half_tiled is the shared-memory-tiled follow-up: same
// 128x128/8x8 register-blocked structure as mm_fp32_16_tiled (see mm_
// cuda_kernels.cu / fp32_16_cuda_kernels.cu), just with __half As/Bs tiles
// and __half accumulation instead of float. mm_fp32_16_pure_half_wmma is
// the tensor-core follow-up: real Tensor Core mma.sync (via nvcuda::wmma,
// 16x16x16 half/half/float tiles), which is how cuBLAS's own Hgemm gets
// its throughput on sm_70+ -- so this is the kernel meant to close the gap
// to cuBLAS, not the scalar ones above it.
//
// The performance section's baseline is cuBLAS's Hgemm (via cuda::mm_
// strided<__half,__half,__half>, see mm_cuda_kernels.cu), operating on a
// fully-materialized __half A (decoded from wh, matching how verify_mm_
// fp32_16_gemm.cu materializes a plain fp32 A for cuBLAS's Sgemm) and a
// __half B (cast from the same plain fp32 B every kernel here uses) -- so
// the gap measures how close a hand-written tensor-core kernel gets to
// cuBLAS's, not a different precision tier.
//
// Mirrors verify_mm_fp32_16_gemm.cu's structure and exists for the same
// reason: a harness to iteratively add and tune faster fp32_16 "pure_half"
// GEMM kernels against cuBLAS's Hgemm as the performance ceiling.
//
// As further custom pure_half kernels are added, wire them into both
// run_correctness_case() and bench_one() below next to mm_fp32_16_pure_
// half_wmma.
//
// The performance sweep's matrices are generated directly on the device,
// same reasoning as verify_mm_fp32_16_gemm.cu.
//
// Usage: verify_mm_fp32_16_pure_half_gemm [--skip-correctness]
//                                          [--skip-performance]
//                                          [--only-shape N]
//
// --only-shape N runs just that one square size and exits, skipping
// correctness and the full sweep -- for isolating a single kernel launch
// under an external profiler (ncu/nsys):
//   scripts/verify_mm_fp32_16_pure_half_gemm.sh -- --only-shape 8192
//   ncu --kernel-name mm_fp32_16_pure_half_wmma_kernel --launch-count 1 \
//       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp32_16_pure_half_gemm" \
//       --only-shape 8192
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

#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"
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

// Materializes a plain __half array from wh alone (fp32_16_half_only, never
// touching wr) -- the fp16-equivalent of verify_mm_fp32_16_gemm.cu's
// combine_dense_fp32_16 call, used to give cuBLAS's Hgemm a fully-
// materialized __half A for a fair "same working precision" comparison.
__global__ void wh_to_half_kernel(size_t n, const uint16_t* wh, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_16_half_only(wh[i]);
}

__global__ void float_to_half_kernel(size_t n, const float* in, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<__half>(in[i]);
}

__global__ void half_to_float_kernel(size_t n, const __half* in, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}

void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}
void gen_float(size_t n, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out);
}
void wh_to_half(size_t n, const uint16_t* wh, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    wh_to_half_kernel<<<blocks, threads>>>(n, wh, out);
}
void float_to_half(size_t n, const float* in, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    float_to_half_kernel<<<blocks, threads>>>(n, in, out);
}
void half_to_float(size_t n, const __half* in, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    half_to_float_kernel<<<blocks, threads>>>(n, in, out);
}

double now_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ---- correctness section: GPU vs CPU reference ----

// pure_half's own long-reduction fp16 divergence (against the CPU
// reference's serial __half accumulation) is expected and not a bug --
// mirrors verify_mv_fp32_16_gemv.cu's tol=5.0 for the same reason, and
// mm_fp32_16_pure_half_wmma additionally accumulates via mma.sync in fp32
// (not __half), so it diverges from the strictly-serial-__half CPU
// reference even more, by design (see its declaration's comment).
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

// Runs one N x N square case (NO_TRANSPOSE on both operands) for
// mm_fp32_16_pure_half/_tiled/_wmma, comparing against the CPU reference
// (cpu::mm_fp32_16_pure_half -- same wh-only decode, __half-only
// reduction, done on the host). tag identifies the size in output; sizes
// not a multiple of the 128x128/8x8 tiled kernel's block or the wmma
// kernel's 64x64/32 block exercise their boundary handling.
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
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              alpha, dim<2>{n, n}, wh.data(), n, dim<2>{n, n}, B.data(), n,
                              beta, C_cpu.data(), n);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    // mm_fp32_16_pure_half_wide_bigreg_sq deliberately takes B and C as
    // native __half (not this file's usual plain-float B/C), so its
    // memory traffic matches gemm_fp16_mma_ptx_wide_bigreg_sq's own
    // byte-for-byte -- see its declaration's comment in mm_cuda_
    // kernels.cuh.
    float_to_half(B.size(), d_B, d_B_half);

    // 128, not 64: mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_
    // acongruous_fastk_precomp2_serp_blockvote's own label (89 chars)
    // plus " " and a tag doesn't fit in 96 either -- same width run_
    // correctness_case_tile_dirty already uses below for the same
    // reason.
    char label[128];
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                   alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                   beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                         alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                         beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_tiled] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_tiled %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                        alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                        beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                                beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma_cpasync] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma_cpasync %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                                    beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma_cpasync_vec] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma_cpasync_vec %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                          alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                                          beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                                              beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                           alpha, dim<2>{n, n}, d_wh, n, dim<2>{n, n}, d_B, n,
                                                           beta, d_C, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk] %s\n", cudaGetErrorString(e));
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // wide_bigreg_sq takes C as native __half (see its declaration's
        // comment) -- stage C_init through d_C (float) then convert into
        // d_C_half, run the kernel, then convert the __half result back
        // to float in-place through d_C for the shared check() path.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  alpha, n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq] %s\n", cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same (ordinary, possibly-mixed-control) data as every other
        // kernel above -- mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty
        // must match mm_fp32_16_pure_half_wide_bigreg_sq bit-for-bit
        // regardless of which tiles its fp32_16_compute_tile_dirty_kernel
        // pass happens to mark clean vs dirty for this particular random
        // draw. The guaranteed-all-clean case (every tile skips decode)
        // is covered separately and deliberately by
        // run_correctness_case_tile_dirty below.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              alpha, n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty] %s\n", cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- mm_fp32_16_pure_half_wide_bigreg_
        // sq_inline_dirty must also match bit-for-bit regardless of
        // which individual 8-element chunks its per-thread inline check
        // happens to skip for this random draw. The guaranteed-all-clean
        // case is covered separately by run_correctness_case_tile_dirty
        // below (which exercises both siblings, not just tile_dirty).
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                alpha, n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty] %s\n", cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- mm_fp32_16_pure_half_wide_bigreg_
        // sq_inline_dirty_acongruous_fastk_precomp2_serp must also match
        // bit-for-bit: it's inline_dirty's own decode_a_tile_inline_
        // dirty logic (unchanged), just re-addressed through a_congruous_
        // idx/precomputed offset tables and reordered at the mma level
        // (see its own top comment in mm_cuda_kernels.cu), none of which
        // changes which elements are considered dirty or how they're
        // decoded.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the blockvote variant must also
        // match bit-for-bit: it decodes the exact same elements as
        // fastk_precomp2_serp (fp32_16_half_only is a correct decode
        // for both control=0 and control=1 bits, so unconditionally
        // decoding more chunks than strictly necessary -- because the
        // block-wide vote triggers more often than a per-warp vote --
        // never changes the VALUES written, only how many redundant-
        // but-idempotent chunks get rewritten). See its own top comment
        // in mm_cuda_kernels.cu for the full derivation.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr,
                         "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label),
                      "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote %s",
                      tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the alwaysdecode variant must also
        // match bit-for-bit: it calls the exact same fp32_16_half_only
        // decode on the exact same chunks fastk_precomp2_serp's own
        // conditional path would decode when triggered, just
        // unconditionally instead of behind a dirty check -- since
        // fp32_16_half_only is a correct decode for control=0 bits too,
        // this changes no value ever written. See its own top comment
        // in mm_cuda_kernels.cu for the full derivation.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(
                stderr,
                "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode] %s\n",
                cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(
            label, sizeof(label),
            "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the perthread variant must also
        // match bit-for-bit: dropping __any_sync means individual lanes
        // within a warp can now take different paths on this branch
        // (ordinary SIMT predication), but each thread's own decode
        // still depends only on data IT ITSELF just wrote via cp.async,
        // so which lanes take which path never changes any VALUE
        // written -- the same same-thread safety argument as decode_a_
        // tile_inline_dirty's own. This is the most important
        // correctness check for this specific kernel, since it's the
        // one that actually exercises real per-lane divergence on
        // real (mixed clean/dirty) random data.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(
                stderr,
                "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread] %s\n",
                cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(
            label, sizeof(label),
            "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the perthread_leanreg variant must
        // also match bit-for-bit: it's the exact same decode_a_tile
        // logic as perthread's own (each thread's dirty check depends
        // only on data it itself just wrote), just restructured so the
        // check doesn't hold the loaded tile across the branch, re-
        // reading from shared memory instead when dirty -- the VALUE
        // written is unaffected either way, only which registers/reads
        // are used to produce it.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr,
                         "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_"
                         "perthread_leanreg] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label),
                     "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_"
                     "leanreg %s",
                     tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the perrep variant must also
        // match bit-for-bit: each rep is checked and decoded fully
        // independently now (no combined OR-reduce across reps), but
        // every rep's own decode still depends only on data THIS thread
        // itself wrote via cp.async for THAT rep, so the same same-
        // thread safety argument applies at the finer per-rep grain
        // too. This is the important correctness check for this
        // specific kernel, since it's the one that actually exercises
        // independent per-rep divergence on real (mixed clean/dirty)
        // random data.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(
                stderr,
                "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep] %s\n",
                cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(
            label, sizeof(label),
            "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the materialize variant must also
        // match bit-for-bit: it decodes via the exact same fp32_16_
        // half_only function every other sibling uses, just into a
        // scratch buffer ahead of time instead of inline in shared
        // memory, then calls the native fp16 kernel (already covered by
        // its own correctness suite) on the decoded result -- the same
        // values, just computed via a different code path.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the cached variant must also
        // match bit-for-bit: it's byte-for-byte the same decode kernel
        // and native GEMM call as plain materialize, only the scratch
        // buffer's own allocation lifetime differs, which has no effect
        // on the VALUES computed. Calling this repeatedly (across n
        // values in this sweep, and across run_correctness_case's own
        // repeated calls at different n) also exercises the cache's own
        // grow-on-demand logic: each larger n after a smaller one forces
        // a real reallocation, not just a cache hit.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp %s",
                      tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- the cached+vectorized variant
        // must also match bit-for-bit: fp32_16_half_only is called on
        // the exact same elements either way, just 4-at-a-time via
        // ushort4/uint2 instead of one scalar element per thread, which
        // changes no value computed. Across this sweep's own n values,
        // most (n=1,15,17,33,129) aren't multiples of 4, so this also
        // exercises the vec kernel's own ld_a%4!=0 fallback to the
        // plain-scalar cached kernel, not just its fast path.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label),
                      "mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- mm_fp32_16_pure_half_wide_bigreg_
        // sq_warp_specialized must also match bit-for-bit: warps 0-3
        // decode the whole tile (their own share plus warp+4's), warps
        // 4-7 skip decode entirely, but the actual VALUES written are
        // identical to inline_dirty's (same fp32_16_half_only logic,
        // just relocated to different threads), so results must match
        // exactly regardless of which warps did the work. The guaranteed
        // -all-clean case is covered separately by run_correctness_case_
        // tile_dirty below.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    {
        // Same ordinary data again -- mm_fp32_16_pure_half_wide_bigreg_
        // sq_dualocc must also match bit-for-bit: A/K-side load, decode,
        // and compute are copied verbatim from inline_dirty, only the
        // B/N tile and thread-block occupancy target differ, neither of
        // which changes any computed value. The guaranteed-all-clean
        // case is covered separately by run_correctness_case_tile_dirty
        // below.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_dualocc(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_dualocc] %s\n", cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_dualocc %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
    }
    if (n % 8 == 0) {
        // mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty needs its
        // chunk_dirty metadata precomputed once from wh first (unlike its
        // two siblings, this isn't a same-signature drop-in -- see its
        // own comment in mm_cuda_kernels.cuh). Same ordinary data as
        // every other kernel above; the guaranteed-all-clean case is
        // covered separately by run_correctness_case_tile_dirty below.
        //
        // Guarded on n % 8 == 0: fp32_16_compute_chunk_dirty's own
        // vectorized (uint4) reads require ld_a % 8 == 0 to stay 16-byte
        // aligned (its documented precondition) -- for non-8-aligned n,
        // mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty itself
        // already falls back to the naive kernel without ever touching
        // chunk_dirty, so there is nothing this fast path would exercise
        // at those sizes anyway; calling the precompute step there would
        // just be a misaligned-address fault for no coverage gained.
        unsigned char* d_chunk_dirty;
        cudaMalloc(&d_chunk_dirty, (n / 8) * n * sizeof(unsigned char));
        cuda::experimental::fp32_16_compute_chunk_dirty(n, n, d_wh, d_chunk_dirty);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n, d_chunk_dirty);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);

        // Same chunk_dirty, same ordinary data -- mm_fp32_16_pure_half_
        // wide_bigreg_sq_prefetched_dirty must also match bit-for-bit.
        // This specifically exercises the corrected (plain synchronous
        // copy, not the 1-byte cp.async that was measured to compile to
        // a guaranteed trap) chunk_dirty_s prefetch path in load_tile_
        // async -- see that kernel's own comment.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n, d_chunk_dirty);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty %s", tag);
        all_ok &= check(label, C_cpu, C_gpu, 5.0);
        cudaFree(d_chunk_dirty);
    }

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return all_ok;
}

// Dedicated correctness case for both mm_fp32_16_pure_half_wide_bigreg_
// sq_tile_dirty and mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty's
// actual point: every A value is nudged away from the near-zero band
// (|v| < 2^-14, fp16's smallest normal exponent -- see fp32_16_pack.hpp)
// that fp32_16_split routes through the "degraded"/"extended" control=1
// encodings, guaranteeing wh's control bit (LSB) is 0 for EVERY element
// (checked explicitly below, not just assumed). That makes every tile --
// whole (block_row, k_tile) tiles for tile_dirty, individual 8-element
// chunks for inline_dirty -- come back clean, so decode is skipped for
// the WHOLE matrix by both methods: the scenario each exists to
// accelerate. run_correctness_case's ordinary pseudo-random data already
// exercises the mixed clean/dirty case incidentally (see its own
// tile_dirty/inline_dirty blocks above); this is the one that
// deliberately, guaranteed exercises the all-clean path for both.
bool run_correctness_case_tile_dirty(size_t n, const char* tag) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto away_from_zero = [](float v) { return std::fabs(v) < 1e-3f ? std::copysign(1e-3f, v) : v; };

    std::vector<float> A(n * n), B(n * n), C_init(n * n);
    for (auto& v : A) v = away_from_zero(dist(rng));
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(n * n), wr(n * n);
    for (size_t i = 0; i < n * n; ++i) fp32_16_split(A[i], wh[i], wr[i]);
    for (uint16_t w : wh) {
        if (w & 1u) {
            std::fprintf(stderr, "[run_correctness_case_tile_dirty] away_from_zero failed to clear control bit\n");
            return false;
        }
    }

    double alpha = 1.7, beta = 0.3;
    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                              alpha, dim<2>{n, n}, wh.data(), n, dim<2>{n, n}, B.data(), n,
                              beta, C_cpu.data(), n);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(B.size(), d_B, d_B_half);

    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                          alpha, n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty] %s\n", cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    std::vector<float> C_gpu(C_init.size());
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    char label[128];
    std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty %s (lsb=0)", tag);
    bool ok = check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- mm_fp32_16_pure_half_wide_bigreg_
    // sq_inline_dirty's per-thread, per-8-element-chunk check should skip
    // every single chunk's decode (every raw16 lane already has control=0
    // by construction), so this is that method's own "skip everywhere"
    // case, exercised deliberately rather than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                            alpha, n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty] %s\n", cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty %s (lsb=0)", tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- decode_a_tile's per-thread check
    // is inline_dirty's own decode_a_tile_inline_dirty unchanged (just
    // re-addressed through a_congruous_idx/precomputed offset tables),
    // so mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_
    // fastk_precomp2_serp should also skip decode everywhere here,
    // exercised deliberately rather than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp %s (lsb=0)", tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- __syncthreads_or's block-wide
    // vote should come back false everywhere (no thread anywhere in the
    // block has a dirty bit), so the blockvote variant should also skip
    // decode everywhere here, exercised deliberately rather than by
    // chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(label, sizeof(label),
                  "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote %s "
                  "(lsb=0)",
                  tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- the alwaysdecode variant has no
    // skip path at all (it always calls fp32_16_half_only), so this is
    // the important correctness check for it specifically: verifying
    // that decoding already-clean (control=0) data is truly a no-op, as
    // claimed in its own top comment, rather than assuming it from the
    // ordinary-data case above.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(
            stderr,
            "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode] %s\n",
            cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(
        label, sizeof(label),
        "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode %s (lsb=0)",
        tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- with no dirty bits set anywhere,
    // every thread's own per-lane `this_thread_dirty` predicate should
    // come back false, so the perthread variant should also skip decode
    // everywhere here, exercised deliberately rather than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(
            stderr,
            "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread] %s\n",
            cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(
        label, sizeof(label),
        "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread %s (lsb=0)",
        tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- perthread_leanreg's own check
    // loop should come back not-dirty for every thread here too, taking
    // the raw-copy-free path (nothing to write at all, As_wh/As alias
    // the same bytes) rather than ever exercising its re-read-on-dirty
    // fallback -- this test is really about the CLEAN path since the
    // ordinary-data check elsewhere already covers the dirty re-read.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_"
                     "perthread_leanreg] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(
        label, sizeof(label),
        "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg %s "
        "(lsb=0)",
        tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- with no dirty bits set anywhere,
    // every rep's own independent check should come back false too, so
    // the perrep variant should also skip decode everywhere here,
    // exercised deliberately rather than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(
            stderr,
            "[mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep] %s\n",
            cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(
        label, sizeof(label),
        "mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep %s (lsb=0)", tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- warps 0-3 should find nothing
    // dirty across their doubled check scope, so mm_fp32_16_pure_half_
    // wide_bigreg_sq_warp_specialized should also skip everywhere,
    // deliberately exercised here rather than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
        d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized] %s\n", cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized %s (lsb=0)", tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- mm_fp32_16_pure_half_wide_bigreg_
    // sq_dualocc's decode is copied verbatim from inline_dirty's own, so
    // it should also skip everywhere, deliberately exercised here rather
    // than by chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_dualocc(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha,
                                                       n, n, n, d_wh, n, d_B_half, n, beta, d_C_half, n);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_dualocc] %s\n", cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_dualocc %s (lsb=0)", tag);
    ok &= check(label, C_cpu, C_gpu, 5.0);

    // Same guaranteed-all-clean data -- every chunk_dirty byte should
    // come back 0, so mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_
    // dirty's decode should also skip everywhere, deliberately exercised
    // here rather than by chance. Guarded on n % 8 == 0 for the same
    // reason as run_correctness_case's own precomputed_dirty block: the
    // precompute pass needs ld_a % 8 == 0 to stay 16-byte aligned, and
    // the GEMM wrapper itself never touches chunk_dirty otherwise anyway.
    if (n % 8 == 0) {
        unsigned char* d_chunk_dirty;
        cudaMalloc(&d_chunk_dirty, (n / 8) * n * sizeof(unsigned char));
        cuda::experimental::fp32_16_compute_chunk_dirty(n, n, d_wh, d_chunk_dirty);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n, d_chunk_dirty);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty %s (lsb=0)", tag);
        ok &= check(label, C_cpu, C_gpu, 5.0);

        // Same guaranteed-all-clean chunk_dirty -- mm_fp32_16_pure_half_
        // wide_bigreg_sq_prefetched_dirty's decode should also skip
        // everywhere, deliberately exercised here rather than by chance.
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);
        cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty(
            TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, n, n, n, d_wh, n, d_B_half, n, beta,
            d_C_half, n, d_chunk_dirty);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty] %s\n",
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty %s (lsb=0)", tag);
        ok &= check(label, C_cpu, C_gpu, 5.0);
        cudaFree(d_chunk_dirty);
    }

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return ok;
}

// Dedicated correctness check for mm_fp32_16_pure_half_gemm_materialize_
// fastk_precomp2_serp's own chunking loop specifically: every shape
// run_correctness_case/run_correctness_case_tile_dirty test (max N=256)
// is far below its 4096-row chunk size, so the multi-chunk path (more
// than one native-GEMM launch, C+row0 pointer offsets into the second
// and later chunks, a ragged final chunk when M isn't a multiple of
// 4096) has zero coverage from those sweeps alone. M=5000 spans exactly
// two chunks (4096 full rows + a ragged 904-row final chunk); K=N=128
// keeps the actual GEMM compute trivial (this is a correctness check,
// not a benchmark) while still being large enough to exercise fastk_
// precomp2_serp's own fast tile-aligned path within each chunk.
bool run_correctness_case_materialize_multichunk() {
    constexpr size_t m = 5000, k = 128, n = 128;
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(m * k), B(k * n), C_init(m * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(m * k), wr(m * k);
    for (size_t i = 0; i < m * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    double alpha = 1.7, beta = 0.3;

    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                              wh.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(B.size(), d_B, d_B_half);
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);

    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp(TransposeType::NO_TRANSPOSE,
                                                                     TransposeType::NO_TRANSPOSE, alpha, m, k, n,
                                                                     d_wh, m, d_B_half, k, beta, d_C_half, m);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp (multichunk)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    std::vector<float> C_gpu(C_init.size());
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = check("mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp (M=5000, multichunk)", C_cpu, C_gpu,
                    5.0);

    // Same M=5000/K=N=128 shape, cached variant: exercises the cache's
    // own grow-on-demand path (this is the first call this process has
    // made to the cached kernel at this size, so it must allocate) AND
    // the multi-chunk loop simultaneously.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp (multichunk)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ok &= check("mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp (M=5000, multichunk)", C_cpu,
               C_gpu, 5.0);

    // Overlapped variant: M=5000 spans exactly 2 chunks (4096 + 904), the
    // minimum needed to exercise its ping-buffered decode/GEMM overlap at
    // all (num_chunks==2 means chunk 1's decode runs concurrently with
    // chunk 0's GEMM, but never reaches the c+2 lookahead branch -- see
    // run_correctness_case_materialize_overlap_3chunk for that).
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(
            stderr, "[mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp (multichunk)] %s\n",
            cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ok &= check("mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp (M=5000, multichunk)",
               C_cpu, C_gpu, 5.0);

    // vec64 variant: same decode as vec, but the write is a single
    // 64-bit integer store instead of two 32-bit ones (see its own
    // comment in mm_cuda_kernels.cu) -- exercised at the same multichunk
    // shape to confirm the repacked write is still numerically correct,
    // not just a different SASS shape.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp (multichunk)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ok &= check("mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp (M=5000, multichunk)", C_cpu,
               C_gpu, 5.0);

    // dirty variant: same decode as vec64, but skips fp32_16_half_only
    // entirely (raw bit copy instead) when a thread's own 4 elements are
    // all control=0 -- see its own comment in mm_cuda_kernels.cu. The
    // random [-1,1] data here is mostly clean (per the measured 99.99%
    // control=0 distribution that motivated this kernel), so this
    // exercises the fast (skip) path heavily; run_correctness_case_
    // materialize_dirty_mixed below deterministically forces both
    // branches within the same test instead of relying on chance.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp (multichunk)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ok &= check("mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp (M=5000, multichunk)", C_cpu,
               C_gpu, 5.0);

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return ok;
}

// Dedicated correctness check for mm_fp32_16_pure_half_gemm_
// materialize_cached_dirty_fastk_precomp2_serp's own thread-granularity
// dirty check specifically: deterministically constructs BOTH a group of
// 4 consecutive rows that are all control=0 ("clean" -- must take the
// raw-copy fast path) and a group of 4 that includes an exact zero
// (control=1, "degraded" -- must take the fp32_16_half_only fallback
// path), within the same column, rather than relying on random [-1,1]
// data to occasionally happen to include a dirty value (it does, per the
// measured 99.99%/0.006% split motivating this kernel, but only in
// expectation -- this test guarantees both branches run every time,
// matching this file's own convention of targeted tests for specific
// code paths, e.g. run_correctness_case_materialize_vec_padded_ld for
// the scalar-remainder branch).
bool run_correctness_case_materialize_dirty_mixed() {
    constexpr size_t m = 8, k = 4, n = 4;
    std::vector<float> A(m * k);
    // Rows 0-3 (group 0): row 0 is exact zero (forces control=1,
    // degraded) -- this whole group's any_dirty check must be true.
    // Rows 4-7 (group 1): all ordinary nonzero in-range values --
    // control=0 for every one, this group's any_dirty check must be
    // false, taking the raw-copy path. Same pattern repeated per column.
    // Same row pattern repeated identically for every column -- row 0
    // must stay EXACTLY 0.0f in every column for the dirty check to
    // reliably trigger group 0's fallback path everywhere, not just in
    // column 0.
    float row_vals[8] = {0.0f, 0.5f, -0.3f, 0.7f, 0.1f, -0.2f, 0.9f, -0.6f};
    for (size_t col = 0; col < k; ++col) {
        for (size_t row = 0; row < m; ++row) {
            A[row + col * m] = row_vals[row];
        }
    }
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> B(k * n), C_init(m * n);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(m * k), wr(m * k);
    for (size_t i = 0; i < m * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    double alpha = 1.7, beta = 0.3;

    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                              wh.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(B.size(), d_B, d_B_half);
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);

    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp (mixed)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    std::vector<float> C_gpu(C_init.size());
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = check("mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp (M=8, mixed clean/dirty)",
                    C_cpu, C_gpu, 5.0);

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return ok;
}

// Dedicated correctness check for mm_fp32_16_pure_half_gemm_
// materialize_cached_vec_overlap_fastk_precomp2_serp's own c+2 lookahead
// branch specifically: with only 2 ping-pong buffers, chunk c+2 reuses
// chunk c's own buffer slot and must wait (via the gemm_done[c%2] event)
// for chunk c's GEMM to finish reading it before decode(c+2) overwrites
// it -- run_correctness_case_materialize_multichunk's own M=5000 case
// (2 chunks) never reaches this branch (c+2 < num_chunks is never true
// for num_chunks==2). M=10000 spans 3 chunks (4096 + 4096 + 1808),
// forcing chunk 2's decode to reuse chunk 0's buffer while chunk 1's
// GEMM is still in flight, and specifically requiring the WAR-guard wait
// to be correct (not just present) -- a wrong/missing wait would show up
// as numerically corrupted rows in chunk 2's own output, not a crash.
bool run_correctness_case_materialize_overlap_3chunk() {
    constexpr size_t m = 10000, k = 128, n = 128;
    std::mt19937 rng(46);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(m * k), B(k * n), C_init(m * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(m * k), wr(m * k);
    for (size_t i = 0; i < m * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    double alpha = 1.7, beta = 0.3;

    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                              wh.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(B.size(), d_B, d_B_half);
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);

    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, m, d_B_half, k, beta,
        d_C_half, m);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(
            stderr, "[mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp (3chunk)] %s\n",
            cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    std::vector<float> C_gpu(C_init.size());
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = check("mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp (M=10000, 3chunk)",
                    C_cpu, C_gpu, 5.0);

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return ok;
}

// Dedicated correctness check for mm_fp32_16_pure_half_gemm_auto_
// fastk_precomp2_serp's own branch-selection logic specifically: every
// existing check above either tests perthread directly (never through
// the dispatcher) or tests materialize/materialize_cached directly at
// specific N values (never exercising the dispatcher's own routing
// decision). Rectangular M=K=8 keeps the CPU reference cheap (a full
// N=2048+ SQUARE CPU reference would take far too long for a routine
// check) while N alone -- the dispatcher's only branch variable --
// brackets its threshold as tightly as this file's own convention of
// round test sizes allows: N=1024 must route to perthread (just below
// the N>=2048 threshold), N=2048 itself must route to materialize_
// cached (the threshold boundary, inclusive). Verifies the routing
// itself, not the underlying kernels' own math (already covered by
// their own dedicated checks above).
bool run_correctness_case_auto_dispatch() {
    constexpr size_t m = 8, k = 8;
    bool ok = true;
    for (size_t n : {1024ul, 2048ul}) {
        std::mt19937 rng(44);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A(m * k), B(k * n), C_init(m * n);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        for (auto& v : C_init) v = dist(rng);

        std::vector<uint16_t> wh(m * k), wr(m * k);
        for (size_t i = 0; i < m * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);

        double alpha = 1.7, beta = 0.3;

        std::vector<float> C_cpu = C_init;
        cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                                  wh.data(), m, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);

        uint16_t* d_wh;
        float *d_B, *d_C;
        __half *d_B_half, *d_C_half;
        cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
        cudaMalloc(&d_B, B.size() * sizeof(float));
        cudaMalloc(&d_C, C_init.size() * sizeof(float));
        cudaMalloc(&d_B_half, B.size() * sizeof(__half));
        cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
        cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(B.size(), d_B, d_B_half);
        cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half(C_init.size(), d_C, d_C_half);

        cuda::experimental::mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp(TransposeType::NO_TRANSPOSE,
                                                                  TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh,
                                                                  m, d_B_half, k, beta, d_C_half, m);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            std::fprintf(stderr, "[mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp (N=%zu)] %s\n", n,
                         cudaGetErrorString(e));
        half_to_float(C_init.size(), d_C_half, d_C);
        std::vector<float> C_gpu(C_init.size());
        cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
        char label[96];
        std::snprintf(label, sizeof(label), "mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp (M=K=8, N=%zu)", n);
        ok &= check(label, C_cpu, C_gpu, 5.0);

        cudaFree(d_wh);
        cudaFree(d_B);
        cudaFree(d_B_half);
        cudaFree(d_C);
        cudaFree(d_C_half);
    }
    return ok;
}

// Dedicated correctness check for mm_fp32_16_pure_half_gemm_
// materialize_cached_vec_fastk_precomp2_serp's own decode kernel
// specifically: its scalar-remainder branch (chunk_rows not a multiple
// of 4) only triggers when the caller passes ld_a > m (a padded leading
// dimension) -- since kMmFp32_16PureHalfMaterializeChunkRows=4096 is
// itself a multiple of 4, any m that IS a multiple of 4 (required for
// ld_a=m to pass this kernel's own alignment gate at all) produces an
// ALSO-multiple-of-4 ragged final chunk, so run_correctness_case_
// materialize_multichunk's own M=5000 (ld_a=m, tightly packed) never
// actually exercises this branch despite genuinely spanning multiple
// chunks. M=4099 (not a multiple of 4) with ld_a=4100 (padded up to the
// next multiple of 4, so the vec kernel's own alignment gate passes)
// forces the SECOND chunk (rows 4096-4098, chunk_rows=3) into the
// scalar fallback path within an otherwise-vectorized kernel. Row m
// through ld_a-1 in the wh/wr buffers are genuine padding -- correct
// code must never read them, since chunk_rows itself (not ld_a) bounds
// every read/write -- so they're left as whatever the random fill
// happens to produce, deliberately not specially initialized.
bool run_correctness_case_materialize_vec_padded_ld() {
    constexpr size_t m = 4099, ld_a = 4100, k = 128, n = 128;
    std::mt19937 rng(45);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(ld_a * k), B(k * n), C_init(m * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : C_init) v = dist(rng);

    std::vector<uint16_t> wh(ld_a * k), wr(ld_a * k);
    for (size_t i = 0; i < ld_a * k; ++i) fp32_16_split(A[i], wh[i], wr[i]);

    double alpha = 1.7, beta = 0.3;

    std::vector<float> C_cpu = C_init;
    cpu::mm_fp32_16_pure_half(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, k},
                              wh.data(), ld_a, dim<2>{k, n}, B.data(), k, beta, C_cpu.data(), m);

    uint16_t* d_wh;
    float *d_B, *d_C;
    __half *d_B_half, *d_C_half;
    cudaMalloc(&d_wh, wh.size() * sizeof(uint16_t));
    cudaMalloc(&d_B, B.size() * sizeof(float));
    cudaMalloc(&d_C, C_init.size() * sizeof(float));
    cudaMalloc(&d_B_half, B.size() * sizeof(__half));
    cudaMalloc(&d_C_half, C_init.size() * sizeof(__half));
    cudaMemcpy(d_wh, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(B.size(), d_B, d_B_half);
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);

    cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, ld_a, d_B_half, k, beta,
        d_C_half, m);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp (padded ld_a)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    std::vector<float> C_gpu(C_init.size());
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = check(
        "mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp (M=4099, ld_a=4100, padded)", C_cpu,
        C_gpu, 5.0);

    // vec64 shares the exact same read/grid logic as vec (only the write
    // differs), so this same padded-ld_a shape exercises its own scalar-
    // remainder branch identically -- see run_correctness_case_
    // materialize_vec_padded_ld's own comment above for why this specific
    // shape (m not %4==0, ld_a padded up to the next multiple of 4) is
    // required to reach that branch at all.
    cudaMemcpy(d_C, C_init.data(), C_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    float_to_half(C_init.size(), d_C, d_C_half);
    cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp(
        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, d_wh, ld_a, d_B_half, k, beta,
        d_C_half, m);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        std::fprintf(stderr,
                     "[mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp (padded ld_a)] %s\n",
                     cudaGetErrorString(e));
    half_to_float(C_init.size(), d_C_half, d_C);
    cudaMemcpy(C_gpu.data(), d_C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ok &= check(
        "mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp (M=4099, ld_a=4100, padded)", C_cpu,
        C_gpu, 5.0);

    cudaFree(d_wh);
    cudaFree(d_B);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaFree(d_C_half);
    return ok;
}

bool run_correctness() {
    std::printf("=== Correctness (mm_fp32_16_pure_half/_tiled/_wmma vs CPU reference) ===\n");
    bool all_ok = true;

    for (size_t n : {1ul, 15ul, 16ul, 17ul, 33ul, 64ul, 129ul, 256ul})
        all_ok &= run_correctness_case(n, "square");

    for (size_t n : {1ul, 15ul, 16ul, 17ul, 33ul, 64ul, 129ul, 256ul})
        all_ok &= run_correctness_case_tile_dirty(n, "square");

    all_ok &= run_correctness_case_materialize_multichunk();
    all_ok &= run_correctness_case_materialize_vec_padded_ld();
    all_ok &= run_correctness_case_materialize_overlap_3chunk();
    all_ok &= run_correctness_case_materialize_dirty_mixed();
    all_ok &= run_correctness_case_auto_dispatch();

    std::printf("\n");
    return all_ok;
}

// ---- performance section: mm_fp32_16_pure_half* vs cuBLAS Hgemm ----

// Benchmarks one N x N square shape: the custom pure_half kernels against
// cuBLAS's Hgemm operating on a fully-materialized __half A (via
// wh_to_half -- so the gap measures the cost/efficiency of our tensor-core
// kernel, not a different GEMM algorithm) and __half B (cast from the same
// plain fp32 B). Pulled out to a standalone function so run_single_shape()
// can also call it, for isolating one shape's kernel launches under an
// external profiler.
void bench_one(cublasHandle_t handle, size_t n) {
    uint16_t* wh;
    float *B, *C;
    __half *A_half, *B_half, *C_half;
    uint16_t* wr_unused = nullptr;
    B = C = nullptr;
    A_half = B_half = C_half = nullptr;
    // Every allocation below is checked (not just the first two): on a
    // shared/multi-tenant GPU, free memory at the moment of this call is
    // whatever other jobs have left, not a fixed budget -- silently
    // ignoring a failed cudaMalloc left later kernels (gen_float, float_
    // to_half, ...) writing through an uninitialized/garbage pointer,
    // corrupting the CUDA context and crashing the host process with a
    // segfault rather than a clean skip (confirmed via compute-sanitizer:
    // an unchecked cudaMalloc failure for B/C/B_half/C_half at N=16384
    // fed float_to_half_kernel a garbage destination pointer).
    bool ok = cudaMalloc(&wh, n * n * sizeof(uint16_t)) == cudaSuccess &&
             cudaMalloc(&A_half, n * n * sizeof(__half)) == cudaSuccess &&
             cudaMalloc(&wr_unused, n * n * sizeof(uint16_t)) == cudaSuccess &&
             cudaMalloc(&B, n * n * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&C, n * n * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&B_half, n * n * sizeof(__half)) == cudaSuccess &&
             cudaMalloc(&C_half, n * n * sizeof(__half)) == cudaSuccess;
    if (!ok) {
        std::printf("  N=%-9zu  SKIPPED (out of device memory)\n", n);
        cudaFree(wh);
        cudaFree(wr_unused);
        cudaFree(B);
        cudaFree(C);
        cudaFree(A_half);
        cudaFree(B_half);
        cudaFree(C_half);
        cudaGetLastError();
        return;
    }
    gen_fp32_16(n * n, wh, wr_unused);
    wh_to_half(n * n, wh, A_half);
    cudaFree(wr_unused);

    gen_float(n * n, B);
    float_to_half(n * n, B, B_half);
    cudaMemset(C, 0, n * n * sizeof(float));
    cudaMemset(C_half, 0, n * n * sizeof(__half));
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
    // mm_fp32_16_pure_half's throughput is naive-kernel flat (one thread
    // per output element, no shared-memory reuse) the same way mm_fp32_16_
    // accurate's is in verify_mm_fp32_16_gemm.cu -- skip it past a
    // threshold so a full sweep doesn't pay minutes for an already-known
    // number.
    constexpr size_t kSkipNaiveAbove = 8192;
    double t_naive = n <= kSkipNaiveAbove
                        ? time_call(
                              [&] {
                                  cuda::mm_fp32_16_pure_half(
                                      TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                      1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
                              },
                              reps)
                        : 0.0;
    double t_tiled = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_tiled(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                             1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                            1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma_cp = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_wmma_cp_vec = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_bigbk = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_bigbk_pad = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                  1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_hugebk = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                               1.0, {n, n}, wh, n, {n, n}, B, n, 0.0, C, n);
        },
        reps);
    double t_wide_bigreg_sq = time_call(
        [&] {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      1.0, n, n, n, wh, n, B_half, n, 0.0, C_half, n);
        },
        reps);
    double t_cublas = time_call(
        [&] {
            cuda::experimental::mm_strided<__half, __half, __half>(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.0,
                {n, n}, A_half, n, {n, n}, B_half, n, 0.0, C_half, n, handle);
        },
        reps);

    double flops = 2.0 * n * n * n;
    struct NamedTime { const char* name; double t; };
    NamedTime candidates[] = {{"tiled", t_tiled}, {"wmma", t_wmma}, {"wmma_cp", t_wmma_cp},
                              {"wmma_cp_vec", t_wmma_cp_vec}, {"bigbk", t_bigbk},
                              {"bigbk_pad", t_bigbk_pad}, {"hugebk", t_hugebk},
                              {"wide_bigreg_sq", t_wide_bigreg_sq}};
    const NamedTime* best_entry = &candidates[0];
    for (const auto& c : candidates)
        if (c.t < best_entry->t) best_entry = &c;
    if (n <= kSkipNaiveAbove) {
        std::printf("  N=%-9zu naive=%9.3fms (%7.1f GFlop/s)  tiled=%9.3fms (%7.1f GFlop/s)  "
                   "wmma=%9.3fms (%7.1f GFlop/s)  wmma_cp=%9.3fms (%7.1f GFlop/s)  "
                   "wmma_cp_vec=%9.3fms (%7.1f GFlop/s)  bigbk=%9.3fms (%7.1f GFlop/s)  "
                   "bigbk_pad=%9.3fms (%7.1f GFlop/s)  hugebk=%9.3fms (%7.1f GFlop/s)  "
                   "wide_bigreg_sq=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_naive, flops / t_naive / 1e6, t_tiled, flops / t_tiled / 1e6,
                   t_wmma, flops / t_wmma / 1e6, t_wmma_cp, flops / t_wmma_cp / 1e6,
                   t_wmma_cp_vec, flops / t_wmma_cp_vec / 1e6, t_bigbk, flops / t_bigbk / 1e6,
                   t_bigbk_pad, flops / t_bigbk_pad / 1e6, t_hugebk, flops / t_hugebk / 1e6,
                   t_wide_bigreg_sq, flops / t_wide_bigreg_sq / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best_entry->t);
    } else {
        std::printf("  N=%-9zu naive=      SKIPPED               tiled=%9.3fms (%7.1f GFlop/s)  "
                   "wmma=%9.3fms (%7.1f GFlop/s)  wmma_cp=%9.3fms (%7.1f GFlop/s)  "
                   "wmma_cp_vec=%9.3fms (%7.1f GFlop/s)  bigbk=%9.3fms (%7.1f GFlop/s)  "
                   "bigbk_pad=%9.3fms (%7.1f GFlop/s)  hugebk=%9.3fms (%7.1f GFlop/s)  "
                   "wide_bigreg_sq=%9.3fms (%7.1f GFlop/s)  "
                   "cublas=%9.3fms (%7.1f GFlop/s)  best=%s (%.3f x cublas)\n",
                   n, t_tiled, flops / t_tiled / 1e6, t_wmma, flops / t_wmma / 1e6,
                   t_wmma_cp, flops / t_wmma_cp / 1e6, t_wmma_cp_vec, flops / t_wmma_cp_vec / 1e6,
                   t_bigbk, flops / t_bigbk / 1e6, t_bigbk_pad, flops / t_bigbk_pad / 1e6,
                   t_hugebk, flops / t_hugebk / 1e6, t_wide_bigreg_sq, flops / t_wide_bigreg_sq / 1e6,
                   t_cublas, flops / t_cublas / 1e6, best_entry->name, t_cublas / best_entry->t);
    }

    cudaFree(wh);
    cudaFree(A_half);
    cudaFree(B);
    cudaFree(C);
    cudaFree(B_half);
    cudaFree(C_half);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_performance() {
    std::printf("=== Performance (mm_fp32_16_pure_half/_tiled/_wmma vs cuBLAS Hgemm, square) ===\n");

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

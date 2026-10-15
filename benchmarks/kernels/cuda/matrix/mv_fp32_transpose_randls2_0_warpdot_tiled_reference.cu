// file: mv_fp32_transpose_randls2_0_warpdot_tiled_reference.cu
//
// Standalone, single-file reference benchmark -- NOT part of tomos_cuda,
// NOT linked against librandls/librandls_core. Exists to get a genuine,
// unmodified-code timing number for randls2.0's own plain-fp32 TRANSPOSE
// GEMV kernel, since no runnable benchmark for it exists anywhere in
// that repo (see the investigation notes below).
//
// Investigation context (temp-dependencies/randls2.0, this repo's own
// vendored copy):
//   - performance_tests/gemv/*.cu is NOT built by CMake at all -- the
//     top-level CMakeLists.txt has add_subdirectory(performance_tests)
//     commented out, and even performance_tests/CMakeLists.txt only
//     pulls in lsqr, not gemv.
//   - Every file there named gemv_fp32_ktiled_atomic_*t*/_n_and_t* is
//     MISLABELED: despite the "fp32" filename, the actual bodies
//     #include <cuda_fp16.h> and operate on __half/__half2 (copy-pasted
//     from a gemv_fp16_pair_tiled.cu template -- the header comments
//     even still say "file: gemv_fp16_pair_tiled.cu"). The one
//     genuinely-fp32 file, gemv_fp32_ktiled_atomic.cu, is NO_TRANSPOSE
//     only (no --trans flag exists in it).
//   - The real fp32 TRANSPOSE kernels live in src/kernels/cuda/matrix/
//     dense_kernels.cu, compiled into librandls/librandls_core, but
//     never exposed via any public header -- unreachable without
//     satisfying that project's full CHOLMOD/SuiteSparse/BLAS
//     dependency chain.
//   - Of the two fp32 TRANSPOSE kernel candidates in THIS vendored
//     snapshot (an older/different revision than ~/randls2.0 -- notably
//     lacking sgemv_cm_transpose_splitM_atomic_xshared, the kernel this
//     session's own TOMOS FP32 TRANSPOSE work was ported from), both
//     are dead code (grepped the whole repo, never invoked anywhere):
//       * sgemv_cm_transpose_warprows_splitm_tiled -- already ported
//         faithfully into tomos_cuda as mv_fp32_16_transpose_warprows_
//         splitm_tiled_fast_normal (see fp32_16_cuda_kernels.cu) and
//         MEASURED catastrophic (30-70x slower than cuBLAS) due to
//         gridDim.x fixed at 1, giving only 4-8 total launched blocks
//         at K=1024-2048 (over 90% of the A100 idle).
//       * sgemv_cm_transpose_warpdot_tiled (THIS file) -- structurally
//         almost identical to our own atomicAdd kernel's per-column
//         compute (warp-per-column, coalesced row reads, shared x tile,
//         warp-shfl reduction), but with NO M-splitting/atomics at all:
//         each warp loops the ENTIRE M range internally in M_TILE-sized
//         tiles. Grid is sized purely by K (blockIdx.y only, no
//         blockIdx.x), so -- UNLIKE warprows_splitm_tiled's cols_per_
//         cta=256 grouping -- block count scales 1:1 with K (e.g. 128
//         blocks at K=1024 with 8 warps/block, not 4), which may avoid
//         that kernel's severe under-occupancy failure mode. Genuinely
//         untested until now.
//
// Build: nvcc -O3 -arch=sm_80 mv_fp32_transpose_randls2_0_warpdot_tiled_reference.cu -lcublas -o mv_fp32_transpose_randls2_0_warpdot_tiled_reference
// Run:   ./mv_fp32_transpose_randls2_0_warpdot_tiled_reference [M] [K]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,  \
                    (int)_s);                                                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ============================================================
// VERBATIM copy of randls2.0's own sgemv_cm_transpose_warpdot_tiled
// (the plain-fp32 overload -- there is a second overload in the source
// operating on a packed uint16_t hi/lo format, not copied here), taken
// unmodified from temp-dependencies/randls2.0/src/kernels/cuda/matrix/
// dense_kernels.cu:6077-6136 (same author, no license/attribution issue,
// same precedent as every other randls2.0 port this session). Not one
// line of the kernel body below has been changed -- comments included,
// verbatim from the source.
// ============================================================

// ---------------------------------------------
// GEMV (column-major A, transpose):
//   y = alpha * (A^T * x) + beta * y
//   - A is MxK, column-major: A[row + k*M]
//   - Each warp computes one output y[k] (one column of A)
//   - Lanes iterate rows within a tile: consecutive rows -> coalesced
//   - Row dimension (M) is tiled into M_TILE and x tile cached in shared
// ---------------------------------------------
template<int M_TILE=256>
__global__ void sgemv_cm_transpose_warpdot_tiled(
    const float* __restrict__ A, // MxK, column-major
    const float* __restrict__ x, // M
    float*       __restrict__ y, // K
    int M, int K,
    float alpha, float beta)
{
  // Thread layout
  const int lane   = threadIdx.x;      // 0..31
  const int warpId = threadIdx.y;      // 0..(warpsPerBlock-1)
  const int warpsPerBlock = blockDim.y;

  // Warp-per-output: pick column k that this warp computes
  const int k = blockIdx.y * warpsPerBlock + warpId;

  // Shared tile of x (M_TILE floats)
  __shared__ float x_sh[M_TILE];

  float sum = 0.f; // warp-local accumulator (lane 0 will hold the reduced sum)

  // Tile over rows (M)
  for (int r0 = 0; r0 < M; r0 += M_TILE) {
    const int tileLen = min(M_TILE, M - r0);

    // Cooperative load of x tile into shared (all threads in CTA)
    for (int t = threadIdx.y * 32 + lane; t < tileLen; t += warpsPerBlock * 32) {
      x_sh[t] = x[r0 + t];
    }
    __syncthreads();

    if (k < K) {
      // For this column k, lanes iterate different rows within the tile (stride 32)
      // Column-major addressing: A[row + k*M]
      const size_t ld = (size_t)M;
      const size_t colBase = (size_t)k * ld;

      float accum = 0.f;
      // Unroll by 4 if you like; scalar version keeps it simple & safe
      for (int t = lane; t < tileLen; t += 32) {
        float a  = A[(size_t)(r0 + t) + colBase];  // consecutive rows -> coalesced across lanes
        float xv = x_sh[t];
        accum = fmaf(a, xv, accum);
      }

      // Warp reduction (sum across lanes)
      unsigned mask = 0xffffffffu;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        accum += __shfl_down_sync(mask, accum, off);

      // Lane 0 adds this tile's contribution into the running total
      if (lane == 0) sum += accum;
    }
    __syncthreads();
  }

  // Epilogue: write y[k]
  if (k < K && lane == 0) {
    float yv = y[k];
    y[k] = alpha * sum + beta * yv;
  }
}

// ============================================================
// Everything below this line is NEW driver code written for this
// comparison (data gen, launch configuration, timing, cuBLAS baseline,
// correctness check) -- NOT from randls2.0. The launch configuration
// (kWarpsPerBlock, kMTile, grid/block sizing) is this file's own
// reasonable default, since this kernel is dead code in its source repo
// with no launcher to copy verbatim -- grid.x MUST be 1 (the kernel
// never reads blockIdx.x, so any grid.x>1 would just duplicate work),
// grid.y = ceil(K / kWarpsPerBlock).
// ============================================================

static constexpr int kWarpsPerBlock = 8;
static constexpr int kMTile = 256;

__global__ void gen_kernel(size_t n, float* out, uint32_t seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s = static_cast<uint32_t>(i) * 2654435761u + seed;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    out[i] = static_cast<float>(s % 200001u) / 100000.0f - 1.0f;  // [-1, 1]
}
void gen_random(size_t n, float* out, uint32_t seed) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_kernel<<<blocks, threads>>>(n, out, seed);
}

float time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
    fn();  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / reps;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int K = (argc > 2) ? std::atoi(argv[2]) : 4096;

    std::printf("=== randls2.0 sgemv_cm_transpose_warpdot_tiled (VERBATIM kernel) vs cuBLAS ===\n");
    std::printf("M=%d K=%d\n", M, K);

    float *A = nullptr, *x = nullptr, *y = nullptr, *y_cublas = nullptr;
    CUDA_CHECK(cudaMalloc(&A, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x, static_cast<size_t>(M) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_cublas, static_cast<size_t>(K) * sizeof(float)));

    gen_random(static_cast<size_t>(M) * K, A, 1u);
    gen_random(static_cast<size_t>(M), x, 2u);
    gen_random(static_cast<size_t>(K), y, 3u);
    CUDA_CHECK(cudaMemcpy(y_cublas, y, static_cast<size_t>(K) * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    const float alpha = 1.7f, beta = 0.3f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int reps = 50;
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K);

    float ms_cublas = time_call(start, stop, [&]() {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, M, K, &alpha, A, M, x, 1, &beta, y_cublas, 1));
    }, reps);
    std::printf("    cublas (Sgemv)                             = %9.3fms (%10.1f GFlop/s)\n",
                ms_cublas, flops / ms_cublas / 1e6);

    dim3 block(32, kWarpsPerBlock);
    dim3 grid(1, (K + kWarpsPerBlock - 1) / kWarpsPerBlock);
    float ms_randls = time_call(start, stop, [&]() {
        sgemv_cm_transpose_warpdot_tiled<kMTile><<<grid, block>>>(A, x, y, M, K, alpha, beta);
    }, reps);
    std::printf("    randls2.0 sgemv_cm_transpose_warpdot_tiled = %9.3fms (%10.1f GFlop/s)\n",
                ms_randls, flops / ms_randls / 1e6);
    std::printf("    -- randls2.0_warpdot_tiled vs cublas: %.3fx\n", (flops / ms_randls) / (flops / ms_cublas));

    // Correctness sanity check: relative L2 error vs the cuBLAS leg
    // (used as ground truth here rather than a separate CPU reference,
    // to keep this file fully self-contained with no tomos dependency).
    std::vector<float> hy(K), hy_cublas(K);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, static_cast<size_t>(K) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy_cublas.data(), y_cublas, static_cast<size_t>(K) * sizeof(float), cudaMemcpyDeviceToHost));
    double num = 0.0, den = 0.0;
    for (int i = 0; i < K; ++i) {
        double d = static_cast<double>(hy[i]) - static_cast<double>(hy_cublas[i]);
        num += d * d;
        den += static_cast<double>(hy_cublas[i]) * static_cast<double>(hy_cublas[i]);
    }
    double relerr = std::sqrt(num) / std::max(1e-12, std::sqrt(den));
    std::printf("    correctness (rel. L2 error vs cuBLAS): %.3e %s\n",
                relerr, relerr < 1e-3 ? "[OK]" : "[FAIL]");

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(y_cublas);
    return 0;
}

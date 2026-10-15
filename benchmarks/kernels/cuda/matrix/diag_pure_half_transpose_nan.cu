// Standalone diagnostic (NOT a permanent benchmark) for the NaN found in
// Fp32_16CudaTest.MvLargeShapeTransposeAndBetaCombinations
// (mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split,
// TRANSPOSE, m=140000, k=100, seeds 301/302 -- exactly matching that
// test's own run_mv_large_shape_combination call). Downloads the kernel's
// intermediate `workspace` (the per-block partial-sum buffer BEFORE the
// separate reduce kernel sums it into y) directly, rather than just the
// final y, to identify exactly which (block, column) pair first goes
// non-finite -- host-side CPU repros and exact-arithmetic overflow checks
// already ruled out the decode math, memory safety (memcheck/initcheck
// both clean), and FP16 accumulator overflow (exact partial sums never
// exceed ~2453, far below the 65504 fp16 ceiling), so this probes the
// kernel's own actual GPU execution directly instead of guessing further.
//
// Build (mirrors this directory's other standalone diagnostic/verify
// binaries -- links directly against the already-built libtomos_cuda.so,
// no CMake target of its own):
//   nvcc -std=c++17 -O2 -arch=sm_80 \
//       -I$REPO_ROOT/src/kernels/cuda/matrix -I$REPO_ROOT/src/core/matrix \
//       -I$REPO_ROOT/include/core -I$REPO_ROOT/src/macros \
//       -I$REPO_ROOT/include/cuda -I$REPO_ROOT/src/kernels/cuda/constants \
//       $REPO_ROOT/benchmarks/kernels/cuda/matrix/diag_pure_half_transpose_nan.cu \
//       -L$REPO_ROOT/build/src/kernels/cuda -ltomos_cuda \
//       -Xlinker -rpath -Xlinker $REPO_ROOT/build/src/kernels/cuda \
//       -o /tmp/diag_pure_half_transpose_nan
//   /tmp/diag_pure_half_transpose_nan

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
namespace experimental {
size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size);
}  // namespace experimental
void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

namespace {
__global__ void split_kernel(size_t n, const float* src, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(src[i], wh[i], wr[i]);
}
}  // namespace

std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

int main() {
    const int m = 140000, k = 100;
    auto a_values = random_values((size_t)m * k, 301);
    auto x_values = random_values((size_t)m, 302);

    float *d_a = nullptr, *d_x = nullptr, *d_y = nullptr, *d_ws = nullptr;
    uint16_t *d_wh = nullptr, *d_wr = nullptr;
    size_t n_a = (size_t)m * k;
    cudaMalloc(&d_a, n_a * sizeof(float));
    cudaMalloc(&d_wh, n_a * sizeof(uint16_t));
    cudaMalloc(&d_wr, n_a * sizeof(uint16_t));
    cudaMalloc(&d_x, (size_t)m * sizeof(float));
    cudaMalloc(&d_y, (size_t)k * sizeof(float));
    cudaMemcpy(d_a, a_values.data(), n_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_values.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((n_a + threads - 1) / threads);
    split_kernel<<<blocks, threads>>>(n_a, d_a, d_wh, d_wr);
    cudaDeviceSynchronize();
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) { std::printf("split_kernel error: %s\n", cudaGetErrorString(err0)); return 1; }

    size_t ws_elems = cuda::experimental::
        mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2>{(size_t)m, (size_t)k});
    std::printf("workspace elems (gridM*k) = %zu\n", ws_elems);
    cudaMalloc(&d_ws, ws_elems * sizeof(float));
    // Poison the workspace before the call so any NEVER-WRITTEN slot is
    // obviously distinguishable (NaN bit pattern 0x7fc00000) from a
    // slot that was written but happens to legitimately be ~0.
    std::vector<float> poison(ws_elems, std::nanf(""));
    cudaMemcpy(d_ws, poison.data(), ws_elems * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_y, 0, (size_t)k * sizeof(float));
    cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(
        TransposeType::TRANSPOSE, 1.0, dim<2>{(size_t)m, (size_t)k}, d_wh, (size_t)m, d_x, 1, 0.0, d_y, 1, d_ws);
    cudaDeviceSynchronize();
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) { std::printf("mv kernel error: %s\n", cudaGetErrorString(err1)); return 1; }

    std::vector<float> ws_host(ws_elems);
    cudaMemcpy(ws_host.data(), d_ws, ws_elems * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> y_host(k);
    cudaMemcpy(y_host.data(), d_y, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);

    // gridM: recompute the SAME M_BLOCK formula the host wrapper itself
    // uses (see fp32_16_cuda_kernels.cu's own mv_fp32_16_pure_half_
    // transpose_atomiconly_scaled_mblock_vec4_split host wrapper), purely
    // to report human-readable (block, row-range) coordinates below --
    // does not affect the actual kernel call above, which computes this
    // internally on its own.
    const int kMvTransposeMBlock = 8192, kMvFp32_16TransposeMinMBlock = 256,
              kMvTransposeWarpsPerBlock = 8, kMvFp32_16TransposeTargetGridBlocks = 2048;
    int gridK = (k + kMvTransposeWarpsPerBlock - 1) / kMvTransposeWarpsPerBlock;
    int M_BLOCK = kMvTransposeMBlock;
    int target_gridM = (kMvFp32_16TransposeTargetGridBlocks + gridK - 1) / gridK;
    if (target_gridM > 1) {
        int scaled_block = (m + target_gridM - 1) / target_gridM;
        if (scaled_block < M_BLOCK) M_BLOCK = std::max(scaled_block, kMvFp32_16TransposeMinMBlock);
    }
    int gridM = (m + M_BLOCK - 1) / M_BLOCK;
    std::printf("M_BLOCK=%d gridM=%d (expected ws_elems=%d)\n", M_BLOCK, gridM, gridM * k);

    size_t bad_count = 0, unwritten_count = 0;
    for (size_t idx = 0; idx < ws_elems; ++idx) {
        float v = ws_host[idx];
        if (std::isnan(v) || std::isinf(v)) {
            int blk = (int)(idx / k);
            int col = (int)(idx % k);
            int mStart = blk * M_BLOCK;
            int mEnd = std::min(m, mStart + M_BLOCK);
            bool is_poison = (std::memcmp(&v, &poison[0], sizeof(float)) == 0);
            if (is_poison) unwritten_count++;
            if (bad_count < 20) {
                std::printf("BAD partial[%zu] (block=%d col=%d rows=[%d,%d)) = %g  %s\n",
                            idx, blk, col, mStart, mEnd, v, is_poison ? "(NEVER WRITTEN - still poison)" : "(written, non-finite)");
            }
            bad_count++;
        }
    }
    std::printf("total bad partial entries: %zu (of which never-written/poison: %zu) out of %zu\n",
                bad_count, unwritten_count, ws_elems);

    size_t bad_y = 0;
    for (int i = 0; i < k; ++i) {
        if (std::isnan(y_host[i]) || std::isinf(y_host[i])) {
            if (bad_y < 10) std::printf("BAD y[%d] = %g\n", i, y_host[i]);
            bad_y++;
        }
    }
    std::printf("total bad y entries: %zu of %d\n", bad_y, k);

    cudaFree(d_a); cudaFree(d_wh); cudaFree(d_wr); cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    return 0;
}

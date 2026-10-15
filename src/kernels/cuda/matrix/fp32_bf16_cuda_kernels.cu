#include "fp32_bf16_cuda_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include "fp32_bf16_pack.hpp"

// FP32_BF16 GEMM kernels: A is the packed wh/wr representation (see
// fp32_bf16_pack.hpp), reconstructed via fp32_bf16_combine(wh, wr) to the
// exact original float32 value -- unlike fp32_16_pack.hpp's own combine,
// this reconstruction is always exact (no degraded/approximate branch),
// since bfloat16's exponent field never needs the range-restricted
// fallback fp16's narrower exponent requires. B and C are plain float, and
// both kernels below accumulate in float (mm_fp32_bf16_accurate_kernel
// upgrades to double only for its own per-element accumulator, matching
// mm_fp32_16_accurate_kernel's own choice), matching ordinary single-
// precision GEMM numerics rather than truncating anywhere to bf16 or fp16
// precision -- this family exists specifically to answer "what if we
// recover full fp32 accuracy from the packed storage" as a numerics
// counterpart to mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4 (mm_cuda_
// kernels.cu), which instead decodes only wh (bf16-truncated) for tensor-
// core throughput.
//
// Structurally this file mirrors fp32_16_cuda_kernels.cu's own mm_fp32_16_
// accurate_kernel/mm_fp32_16_tiled_kernel pair exactly (same tile shape,
// same thread mapping, same accumulation precision) -- only the decode
// call differs (fp32_bf16_combine instead of fp32_16_combine). Kept in
// its own file/translation unit rather than added to fp32_16_cuda_
// kernels.cu, matching how that file itself is already separate from
// mm_cuda_kernels.cu: one file per packed-precision family.

namespace tomos {
namespace cuda {

namespace {

// One thread per output element C(i, j); mirrors mm_fp32_16_accurate_
// kernel's own untiled, always-correct-for-any-shape design -- the safe
// fallback for transposed operands or shapes mm_fp32_bf16_tiled_kernel's
// fast path doesn't cover, and a standalone reference point in its own
// right.
__global__ void mm_fp32_bf16_accurate_kernel(
    TransposeType transA, TransposeType transB, double alpha, size_t m,
    size_t k, size_t n, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* B, size_t ld_b, double beta, float* C, size_t ld_c) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;

    double acc = 0.0;
    for (size_t l = 0; l < k; ++l) {
        size_t a_row = (transA == TransposeType::NO_TRANSPOSE) ? i : l;
        size_t a_col = (transA == TransposeType::NO_TRANSPOSE) ? l : i;
        size_t idx = a_row + a_col * ld_a;
        double a_val = static_cast<double>(fp32_bf16_combine(wh[idx], wr[idx]));
        double b_val = (transB == TransposeType::NO_TRANSPOSE)
                          ? static_cast<double>(B[l + j * ld_b])
                          : static_cast<double>(B[j + l * ld_b]);
        acc += a_val * b_val;
    }
    C[i + j * ld_c] = (beta != 0.0)
                          ? static_cast<float>(alpha * acc + beta * static_cast<double>(
                                                                 C[i + j * ld_c]))
                          : static_cast<float>(alpha * acc);
}

constexpr int kMmFp32Bf16BM = 128;
constexpr int kMmFp32Bf16BN = 128;
constexpr int kMmFp32Bf16BK = 8;
constexpr int kMmFp32Bf16TM = 8;
constexpr int kMmFp32Bf16TN = 8;

// Shared-memory-tiled, register-blocked (8x8/thread) GEMM -- same tile
// shape and thread mapping as mm_fp32_16_tiled_kernel (fp32_16_cuda_
// kernels.cu), decode call swapped for fp32_bf16_combine. Each wh/wr pair
// is decoded once when loaded into shared memory, not once per use (same
// reuse argument mm_fp32_16_tiled_kernel's own comment makes). NO_
// TRANSPOSE x NO_TRANSPOSE only.
__global__ void mm_fp32_bf16_tiled_kernel(double alpha, size_t m, size_t k, size_t n,
                                          const uint16_t* __restrict__ wh,
                                          const uint16_t* __restrict__ wr, size_t ld_a,
                                          const float* __restrict__ B, size_t ld_b,
                                          double beta, float* __restrict__ C, size_t ld_c) {
    __shared__ float As[kMmFp32Bf16BK][kMmFp32Bf16BM];
    __shared__ float Bs[kMmFp32Bf16BK][kMmFp32Bf16BN];

    const size_t block_row = static_cast<size_t>(blockIdx.y) * kMmFp32Bf16BM;
    const size_t block_col = static_cast<size_t>(blockIdx.x) * kMmFp32Bf16BN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * (kMmFp32Bf16BN / kMmFp32Bf16TN) + tx;
    constexpr int kNumThreads =
        (kMmFp32Bf16BM / kMmFp32Bf16TM) * (kMmFp32Bf16BN / kMmFp32Bf16TN);

    float acc[kMmFp32Bf16TM][kMmFp32Bf16TN];
#pragma unroll
    for (int i = 0; i < kMmFp32Bf16TM; ++i)
#pragma unroll
        for (int j = 0; j < kMmFp32Bf16TN; ++j) acc[i][j] = 0.0f;

    for (size_t k0 = 0; k0 < k; k0 += kMmFp32Bf16BK) {
#pragma unroll
        for (int idx = tid; idx < kMmFp32Bf16BM * kMmFp32Bf16BK; idx += kNumThreads) {
            int i = idx % kMmFp32Bf16BM;
            int kk = idx / kMmFp32Bf16BM;
            size_t row = block_row + i;
            size_t kg = k0 + kk;
            if (row < m && kg < k) {
                size_t a_idx = row + kg * ld_a;
                As[kk][i] = fp32_bf16_combine(wh[a_idx], wr[a_idx]);
            } else {
                As[kk][i] = 0.0f;
            }
        }
#pragma unroll
        for (int idx = tid; idx < kMmFp32Bf16BK * kMmFp32Bf16BN; idx += kNumThreads) {
            int j = idx % kMmFp32Bf16BN;
            int kk = idx / kMmFp32Bf16BN;
            size_t col = block_col + j;
            size_t kg = k0 + kk;
            Bs[kk][j] = (col < n && kg < k) ? B[kg + col * ld_b] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kMmFp32Bf16BK; ++kk) {
            float a_reg[kMmFp32Bf16TM];
            float b_reg[kMmFp32Bf16TN];
#pragma unroll
            for (int i = 0; i < kMmFp32Bf16TM; ++i) a_reg[i] = As[kk][ty * kMmFp32Bf16TM + i];
#pragma unroll
            for (int j = 0; j < kMmFp32Bf16TN; ++j) b_reg[j] = Bs[kk][tx * kMmFp32Bf16TN + j];
#pragma unroll
            for (int i = 0; i < kMmFp32Bf16TM; ++i)
#pragma unroll
                for (int j = 0; j < kMmFp32Bf16TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kMmFp32Bf16TM; ++i) {
        size_t row = block_row + ty * kMmFp32Bf16TM + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < kMmFp32Bf16TN; ++j) {
            size_t col = block_col + tx * kMmFp32Bf16TN + j;
            if (col >= n) continue;
            size_t idx = row + col * ld_c;
            C[idx] = static_cast<float>(alpha * static_cast<double>(acc[i][j]) +
                                        beta * static_cast<double>(C[idx]));
        }
    }
}

}  // namespace

void mm_fp32_bf16_accurate(TransposeType transA, TransposeType transB,
                           double alpha, dim<2> size_A, const uint16_t* wh,
                           const uint16_t* wr, size_t ld_a, dim<2> size_B,
                           const float* B, size_t ld_b, double beta, float* C,
                           size_t ld_c) {
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];

    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    mm_fp32_bf16_accurate_kernel<<<blocks, threads>>>(
        transA, transB, alpha, m, k, n, wh, wr, ld_a, B, ld_b, beta, C, ld_c);
    cudaDeviceSynchronize();
}

void mm_fp32_bf16_tiled(TransposeType transA, TransposeType transB, double alpha,
                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                        dim<2> size_B, const float* B, size_t ld_b, double beta, float* C,
                        size_t ld_c) {
    if (transA != TransposeType::NO_TRANSPOSE || transB != TransposeType::NO_TRANSPOSE) {
        // Transposed fast path not implemented -- fall back to the
        // always-correct untiled kernel rather than getting it wrong, same
        // as mm_fp32_16_tiled's own fallback.
        mm_fp32_bf16_accurate(transA, transB, alpha, size_A, wh, wr, ld_a, size_B, B, ld_b,
                              beta, C, ld_c);
        return;
    }
    size_t m = size_A[0];
    size_t k = size_A[1];
    size_t n = size_B[1];

    dim3 threads(kMmFp32Bf16BN / kMmFp32Bf16TN, kMmFp32Bf16BM / kMmFp32Bf16TM);
    dim3 blocks((n + kMmFp32Bf16BN - 1) / kMmFp32Bf16BN, (m + kMmFp32Bf16BM - 1) / kMmFp32Bf16BM);
    mm_fp32_bf16_tiled_kernel<<<blocks, threads>>>(alpha, m, k, n, wh, wr, ld_a, B, ld_b, beta, C, ld_c);
}

// mm_fp32_bf16_tiled_cpasync_dualbuf: FP32_BF16 port of mm_fp32_16_tiled_
// cpasync_dualbuf_kernel (fp32_16_cuda_kernels.cu) -- that kernel is the
// consistent top performer across the fp32_16 tiled lineage (~16.3-16.6k
// GFlop/s at N>=8192 vs mm_fp32_16_tiled's own un-vectorized ~280 GFlop/s
// baseline; see verify_mm_fp32_16_gemm's own bench_one output). mm_fp32_
// bf16_tiled above was deliberately modeled on the UNTUNED base kernel
// (mm_fp32_16_tiled_kernel) for a numerics-matching first cut, not this
// lineage's tuning -- this kernel closes that gap by porting dualbuf's
// specific architecture wholesale: same 128x128/BK=16 tile, same 2x4-warp
// x 8x4-lane geometry, same double-buffered cp.async pipeline for BOTH
// operands (A's raw wh/wr bytes staged then bit-decoded via fp32_bf16_
// combine -- swapped in for fp32_16_combine, otherwise identical; B's
// raw bytes staged into a K-contiguous buffer matching its own global
// layout, then transposed via a cheap shared-to-shared scatter into the
// N-fast layout compute wants, since B is already plain float and needs
// no decode). Every index function, constant, and the full priming/main-
// loop/epilogue structure are ported verbatim; only the decode call and
// symbol names differ. See that kernel's own comment for why B needs the
// staging+transpose step cp.async alone can't skip (a straight copy can't
// scatter into a different-layout destination) and why __launch_bounds__
// (256, 2) matters here (without it, the staging state pushes registers
// high enough to drop occupancy from 2 resident blocks/SM to 1).
namespace {
constexpr int kMmFp32Bf16DualBM = 128;
constexpr int kMmFp32Bf16DualBN = 128;
constexpr int kMmFp32Bf16DualBK = 16;
constexpr int kMmFp32Bf16DualTM = 8;
constexpr int kMmFp32Bf16DualTN = 8;
constexpr int kMmFp32Bf16DualWarpsM = 2;
constexpr int kMmFp32Bf16DualWarpsN = 4;
constexpr int kMmFp32Bf16DualLanesM = 8;
constexpr int kMmFp32Bf16DualLanesN = 4;
constexpr int kMmFp32Bf16DualNumThreads =
    (kMmFp32Bf16DualBM / kMmFp32Bf16DualTM) * (kMmFp32Bf16DualBN / kMmFp32Bf16DualTN);  // 256

constexpr size_t kMmFp32Bf16DualStagingWhBytes =
    2 * static_cast<size_t>(kMmFp32Bf16DualBK) * kMmFp32Bf16DualBM * sizeof(uint16_t);
constexpr size_t kMmFp32Bf16DualStagingWrBytes = kMmFp32Bf16DualStagingWhBytes;
constexpr size_t kMmFp32Bf16DualAsBytes =
    2 * static_cast<size_t>(kMmFp32Bf16DualBK) * kMmFp32Bf16DualBM * sizeof(float);
constexpr size_t kMmFp32Bf16DualStagingBBytes =
    2 * static_cast<size_t>(kMmFp32Bf16DualBN) * kMmFp32Bf16DualBK * sizeof(float);
constexpr size_t kMmFp32Bf16DualBsBytes =
    2 * static_cast<size_t>(kMmFp32Bf16DualBK) * kMmFp32Bf16DualBN * sizeof(float);

constexpr size_t kMmFp32Bf16DualStagingWhOff = 0;
constexpr size_t kMmFp32Bf16DualStagingWrOff =
    kMmFp32Bf16DualStagingWhOff + kMmFp32Bf16DualStagingWhBytes;
constexpr size_t kMmFp32Bf16DualAsOff = kMmFp32Bf16DualStagingWrOff + kMmFp32Bf16DualStagingWrBytes;
constexpr size_t kMmFp32Bf16DualStagingBOff = kMmFp32Bf16DualAsOff + kMmFp32Bf16DualAsBytes;
constexpr size_t kMmFp32Bf16DualBsOff = kMmFp32Bf16DualStagingBOff + kMmFp32Bf16DualStagingBBytes;
constexpr size_t kMmFp32Bf16DualSharedBytes = kMmFp32Bf16DualBsOff + kMmFp32Bf16DualBsBytes;  // 64KB

// As/stagingWh/stagingWr share this [2][BK][BM] shape.
__device__ __forceinline__ int mm_fp32_bf16_dual_a_idx(int buf, int kk, int i) {
    return (buf * kMmFp32Bf16DualBK + kk) * kMmFp32Bf16DualBM + i;
}
// stagingB: [2][BN][BK], K fast -- matches B's own global layout.
__device__ __forceinline__ int mm_fp32_bf16_dual_stagingB_idx(int buf, int j, int kk) {
    return (buf * kMmFp32Bf16DualBN + j) * kMmFp32Bf16DualBK + kk;
}
// Bs: [2][BK][BN], N fast -- the compute-friendly layout.
__device__ __forceinline__ int mm_fp32_bf16_dual_bs_idx(int buf, int kk, int j) {
    return (buf * kMmFp32Bf16DualBK + kk) * kMmFp32Bf16DualBN + j;
}
}  // namespace

__global__ __launch_bounds__(256, 2) void mm_fp32_bf16_tiled_cpasync_dualbuf_kernel(
    double alpha, size_t m, size_t k, size_t n,
    const uint16_t* __restrict__ wh, const uint16_t* __restrict__ wr, size_t ld_a,
    const float* __restrict__ B, size_t ld_b,
    double beta, float* __restrict__ C, size_t ld_c) {
    extern __shared__ float4 smem_raw4[];
    unsigned char* smem_raw = reinterpret_cast<unsigned char*>(smem_raw4);
    uint16_t* stagingWh = reinterpret_cast<uint16_t*>(smem_raw + kMmFp32Bf16DualStagingWhOff);
    uint16_t* stagingWr = reinterpret_cast<uint16_t*>(smem_raw + kMmFp32Bf16DualStagingWrOff);
    float* As = reinterpret_cast<float*>(smem_raw + kMmFp32Bf16DualAsOff);
    float* stagingB = reinterpret_cast<float*>(smem_raw + kMmFp32Bf16DualStagingBOff);
    float* Bs = reinterpret_cast<float*>(smem_raw + kMmFp32Bf16DualBsOff);

    const size_t block_row = static_cast<size_t>(blockIdx.y) * kMmFp32Bf16DualBM;
    const size_t block_col = static_cast<size_t>(blockIdx.x) * kMmFp32Bf16DualBN;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_row = warp_id / kMmFp32Bf16DualWarpsN;
    const int warp_col = warp_id % kMmFp32Bf16DualWarpsN;
    const int lane_row = lane_id / kMmFp32Bf16DualLanesN;
    const int lane_col = lane_id % kMmFp32Bf16DualLanesN;
    const int row_off =
        warp_row * (kMmFp32Bf16DualBM / kMmFp32Bf16DualWarpsM) + lane_row * kMmFp32Bf16DualTM;
    const int col_off =
        warp_col * (kMmFp32Bf16DualBN / kMmFp32Bf16DualWarpsN) + lane_col * kMmFp32Bf16DualTN;

    const int a_i0 = (tid % (kMmFp32Bf16DualBM / 4)) * 4;
    const int a_kk_base = tid / (kMmFp32Bf16DualBM / 4);
    const int b_kk0_base = (tid % (kMmFp32Bf16DualBK / 4)) * 4;
    const int b_j_base = tid / (kMmFp32Bf16DualBK / 4);

    auto load_tile_async = [&](int buf, size_t k0) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            int a_kk = a_kk_base + v * (kMmFp32Bf16DualBK / 2);
            size_t a_row0 = block_row + a_i0;
            size_t a_kg = k0 + a_kk;
            if (a_row0 < m && a_kg < k) {
                size_t a_idx = a_row0 + a_kg * ld_a;
                __pipeline_memcpy_async(&stagingWh[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0)],
                                        &wh[a_idx], sizeof(ushort4));
                __pipeline_memcpy_async(&stagingWr[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0)],
                                        &wr[a_idx], sizeof(ushort4));
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) As[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0 + e)] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            int b_j = b_j_base + v * (kMmFp32Bf16DualNumThreads / (kMmFp32Bf16DualBK / 4));
            size_t b_col = block_col + b_j;
            size_t b_kg0 = k0 + b_kk0_base;
            if (b_col < n && b_kg0 < k) {
                __pipeline_memcpy_async(&stagingB[mm_fp32_bf16_dual_stagingB_idx(buf, b_j, b_kk0_base)],
                                        &B[b_kg0 + b_col * ld_b], sizeof(float4));
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    Bs[mm_fp32_bf16_dual_bs_idx(buf, b_kk0_base + e, b_j)] = 0.0f;
            }
        }
        __pipeline_commit();
    };

    // Turns this tile's staged raw bytes into the real compute buffers --
    // bit-decode for A via fp32_bf16_combine (the only line that differs
    // from mm_fp32_16_tiled_cpasync_dualbuf_kernel's own transform_tile),
    // a plain shared-to-shared layout transpose for B.
    auto transform_tile = [&](int buf, size_t k0) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            int a_kk = a_kk_base + v * (kMmFp32Bf16DualBK / 2);
            size_t a_row0 = block_row + a_i0;
            size_t a_kg = k0 + a_kk;
            if (a_row0 < m && a_kg < k) {
                ushort4 wh4 =
                    *reinterpret_cast<const ushort4*>(&stagingWh[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0)]);
                ushort4 wr4 =
                    *reinterpret_cast<const ushort4*>(&stagingWr[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0)]);
                As[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0 + 0)] = fp32_bf16_combine(wh4.x, wr4.x);
                As[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0 + 1)] = fp32_bf16_combine(wh4.y, wr4.y);
                As[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0 + 2)] = fp32_bf16_combine(wh4.z, wr4.z);
                As[mm_fp32_bf16_dual_a_idx(buf, a_kk, a_i0 + 3)] = fp32_bf16_combine(wh4.w, wr4.w);
            }
        }
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            int b_j = b_j_base + v * (kMmFp32Bf16DualNumThreads / (kMmFp32Bf16DualBK / 4));
            size_t b_col = block_col + b_j;
            size_t b_kg0 = k0 + b_kk0_base;
            if (b_col < n && b_kg0 < k) {
                float4 t = *reinterpret_cast<const float4*>(
                    &stagingB[mm_fp32_bf16_dual_stagingB_idx(buf, b_j, b_kk0_base)]);
                Bs[mm_fp32_bf16_dual_bs_idx(buf, b_kk0_base + 0, b_j)] = t.x;
                Bs[mm_fp32_bf16_dual_bs_idx(buf, b_kk0_base + 1, b_j)] = t.y;
                Bs[mm_fp32_bf16_dual_bs_idx(buf, b_kk0_base + 2, b_j)] = t.z;
                Bs[mm_fp32_bf16_dual_bs_idx(buf, b_kk0_base + 3, b_j)] = t.w;
            }
        }
    };

    float acc[kMmFp32Bf16DualTM][kMmFp32Bf16DualTN];
#pragma unroll
    for (int i = 0; i < kMmFp32Bf16DualTM; ++i)
#pragma unroll
        for (int j = 0; j < kMmFp32Bf16DualTN; ++j) acc[i][j] = 0.0f;

    load_tile_async(0, 0);
    __pipeline_wait_prior(0);
    transform_tile(0, 0);
    __syncthreads();

    int cur = 0;
    for (size_t k0 = 0; k0 < k; k0 += kMmFp32Bf16DualBK) {
        size_t next_k0 = k0 + kMmFp32Bf16DualBK;
        int nxt = cur ^ 1;
        if (next_k0 < k) load_tile_async(nxt, next_k0);

#pragma unroll
        for (int kk = 0; kk < kMmFp32Bf16DualBK; ++kk) {
            float a_reg[kMmFp32Bf16DualTM];
            float b_reg[kMmFp32Bf16DualTN];
#pragma unroll
            for (int v = 0; v < kMmFp32Bf16DualTM / 4; ++v) {
                float4 t = *reinterpret_cast<const float4*>(
                    &As[mm_fp32_bf16_dual_a_idx(cur, kk, row_off + v * 4)]);
                a_reg[v * 4 + 0] = t.x;
                a_reg[v * 4 + 1] = t.y;
                a_reg[v * 4 + 2] = t.z;
                a_reg[v * 4 + 3] = t.w;
            }
#pragma unroll
            for (int v = 0; v < kMmFp32Bf16DualTN / 4; ++v) {
                float4 t = *reinterpret_cast<const float4*>(
                    &Bs[mm_fp32_bf16_dual_bs_idx(cur, kk, col_off + v * 4)]);
                b_reg[v * 4 + 0] = t.x;
                b_reg[v * 4 + 1] = t.y;
                b_reg[v * 4 + 2] = t.z;
                b_reg[v * 4 + 3] = t.w;
            }
#pragma unroll
            for (int i = 0; i < kMmFp32Bf16DualTM; ++i)
#pragma unroll
                for (int j = 0; j < kMmFp32Bf16DualTN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }

        if (next_k0 < k) {
            __pipeline_wait_prior(0);
            transform_tile(nxt, next_k0);
        }
        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int i = 0; i < kMmFp32Bf16DualTM; ++i) {
        size_t row = block_row + row_off + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < kMmFp32Bf16DualTN; ++j) {
            size_t col = block_col + col_off + j;
            if (col >= n) continue;
            size_t idx = row + col * ld_c;
            C[idx] = static_cast<float>(alpha * static_cast<double>(acc[i][j]) +
                                        beta * static_cast<double>(C[idx]));
        }
    }
}

// Host wrapper -- same fallback shape as mm_fp32_16_tiled_cpasync_
// dualbuf's own, substituting this file's own two existing tiers
// (mm_fp32_bf16_accurate for the transpose case, mm_fp32_bf16_tiled --
// this file's own untiled-alignment-requirement base kernel -- for the
// unaligned-shape case) in place of the fp32_16 family's own accurate/
// vec2, since no fp32_bf16 "vec2" tier exists (or was asked for).
void mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c) {
    if (transA != TransposeType::NO_TRANSPOSE || transB != TransposeType::NO_TRANSPOSE) {
        mm_fp32_bf16_accurate(transA, transB, alpha, size_A, wh, wr, ld_a, size_B, B, ld_b,
                              beta, C, ld_c);
        return;
    }
    size_t m = size_A[0];
    size_t k = size_A[1];
    size_t n = size_B[1];

    if (ld_a % 4 != 0 || m % 4 != 0 || ld_b % 4 != 0 || k % 4 != 0) {
        mm_fp32_bf16_tiled(transA, transB, alpha, size_A, wh, wr, ld_a, size_B, B, ld_b,
                           beta, C, ld_c);
        return;
    }

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(mm_fp32_bf16_tiled_cpasync_dualbuf_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(kMmFp32Bf16DualSharedBytes));
        attr_set = true;
    }

    dim3 threads(kMmFp32Bf16DualNumThreads);  // 256, 1D
    dim3 blocks((n + kMmFp32Bf16DualBN - 1) / kMmFp32Bf16DualBN,
               (m + kMmFp32Bf16DualBM - 1) / kMmFp32Bf16DualBM);
    mm_fp32_bf16_tiled_cpasync_dualbuf_kernel<<<blocks, threads, kMmFp32Bf16DualSharedBytes>>>(
        alpha, m, k, n, wh, wr, ld_a, B, ld_b, beta, C, ld_c);
}

}  // namespace cuda
}  // namespace tomos

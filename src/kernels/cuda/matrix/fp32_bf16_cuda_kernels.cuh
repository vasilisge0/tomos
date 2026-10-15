#ifndef TOMOS_FP32_BF16_CUDA_KERNELS_CUH
#define TOMOS_FP32_BF16_CUDA_KERNELS_CUH

#include <cstdint>

#include <cuda_bf16.h>

#include "dimensions.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {

// C = alpha * op(A) * op(B) + beta * C, where A is stored as the packed
// FP32_BF16 wh/wr representation (see fp32_bf16_pack.hpp), reconstructed
// to its exact original float32 value via fp32_bf16_combine(wh, wr) per
// element, accumulated in double. One thread per output element -- the
// always-correct-for-any-shape/transpose reference kernel, and mm_fp32_
// bf16_tiled's own fallback for cases its fast path doesn't cover. Mirrors
// mm_fp32_16_accurate exactly (fp32_16_cuda_kernels.cuh), decode call
// swapped for fp32_bf16_combine.
void mm_fp32_bf16_accurate(TransposeType transA, TransposeType transB,
                           double alpha, dim<2> size_A, const uint16_t* wh,
                           const uint16_t* wr, size_t ld_a, dim<2> size_B,
                           const float* B, size_t ld_b, double beta, float* C,
                           size_t ld_c);

// Same as mm_fp32_bf16_accurate, but shared-memory-tiled: each wh/wr pair
// is decoded into a float once, when it's loaded into shared memory,
// instead of once per use, and the compute loop accumulates in float (the
// same numerics a standard, non-tensor-core single-precision GEMM uses --
// e.g. mm_fp32_16_tiled's own, which this kernel's tile shape and
// structure mirror directly), so results match ordinary fp32 GEMM to
// within normal fp32 rounding, not truncated to bf16 or fp16 precision
// anywhere. NO_TRANSPOSE x NO_TRANSPOSE only -- falls back to mm_fp32_
// bf16_accurate for any transposed operand. See fp32_bf16_cuda_kernels.cu
// for mm_fp32_bf16_tiled_kernel.
void mm_fp32_bf16_tiled(TransposeType transA, TransposeType transB, double alpha,
                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                        dim<2> size_B, const float* B, size_t ld_b, double beta, float* C,
                        size_t ld_c);

// FP32_BF16 port of mm_fp32_16_tiled_cpasync_dualbuf (fp32_16_cuda_
// kernels.cuh) -- the consistent top performer in that family's tiled
// lineage. Same 128x128/BK=16 tile, same warp/lane geometry, same double-
// buffered cp.async pipeline for both operands (A's raw bytes staged then
// decoded via fp32_bf16_combine; B staged into a K-contiguous buffer then
// transposed into the compute-friendly N-fast layout, since it needs no
// decode but cp.async can't scatter into a different layout directly).
// Every index function and the full structure are ported verbatim from
// that kernel; only the decode call and symbol names differ. NO_TRANSPOSE
// x NO_TRANSPOSE and ld_a/m/ld_b/k all multiples of 4 only -- falls back
// to mm_fp32_bf16_accurate (transpose) or mm_fp32_bf16_tiled (unaligned
// shape), this file's own two existing tiers, since no fp32_bf16 "vec2"
// tier exists. See mm_fp32_bf16_tiled_cpasync_dualbuf_kernel's own
// comment in fp32_bf16_cuda_kernels.cu for the full derivation.
void mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_FP32_BF16_CUDA_KERNELS_CUH

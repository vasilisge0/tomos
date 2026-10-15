// "_advanced" variant of mv_fp32_16_pure_half_vs_cublas_ref.cu -- byte-
// for-byte identical except for time_call's own timing methodology and
// two new CLI flags, --warmup N and --iters N (both default 10, see
// main()'s own argv-filtering preamble): instead of this file's own
// sibling's fixed, shape-size-dependent `reps` heuristic (a single
// un-timed warmup call, then `reps` timed calls averaged), this variant
// always does g_warmup_runs UN-TIMED warmup calls first, then times g_
// iterations calls as one block and divides by g_iterations -- both
// counts fully caller-controlled, independent of shape. No kernel call
// site, shape list, or dispatch logic differs from the sibling file.
//
// Standalone throughput comparison for the NO_TRANSPOSE (y = alpha*A*x +
// beta*y) GENUINE 16-BIT COMPUTE tier of GEMV -- the NO_TRANSPOSE mirror of
// mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu, covering the same final
// four-series "TOMOS" set that file's own plot settled on (see plot_gemv_
// transpose_fp32_16_pure_half_vs_cublas_ref.m's header), not that file's
// full exploratory kernel history.
//
// Checked before building (per the "check existing kernels first"
// instruction that motivated this file): mv_fp32_16_pure_half_wide_ilp_
// shared_half2 and mv_fp32_16_pure_half_wide_ilp_shared_fasthalf (src/
// kernels/cuda/matrix/fp32_16_cuda_kernels.cu) are ALREADY generic,
// trans-dispatched kernels -- they need no new __global__ code, only
// wiring with TransposeType::NO_TRANSPOSE below. No "TOMOS FP32/FP16"
// (wh-only extended-exponent reconstruction) or "TOMOS FP32/BF16" (genuine
// bf16 compute) kernel existed for NO_TRANSPOSE, so mv_fp32_16_hybrid_wide_
// ilp_shared and mv_fp32_bf16_pure_bf16_wide_ilp_shared were built fresh
// for this file -- both reuse the SAME wide_ilp_shared_half2 structure
// (K-split grid, shared x_tile, 4-way unroll of independent scalar wh
// loads) rather than the TRANSPOSE family's own warp-per-column/ushort4
// one: that TRANSPOSE structure vectorizes wh loads by pairing ADJACENT
// ROWS within one output column, which only exists because column-major
// storage makes a column's own rows contiguous. NO_TRANSPOSE has no
// per-thread analog of that -- for a fixed output row, adjacent K-columns
// are ld_a apart, not contiguous -- so no per-thread vector load of wh is
// possible there either way; NO_TRANSPOSE's own memory-bandwidth win
// instead comes from ADJACENT THREADS (adjacent output rows) reading
// adjacent addresses for the SAME column, already coalesced across the
// warp without any per-lane vectorization. See both new kernels' own
// comments in fp32_16_cuda_kernels.cu for the full derivation.
//
// Five TOMOS series (matching plot_gemv_fp32_16_pure_half_vs_cublas_ref.m):
//   TOMOS FP16             = mv_fp32_16_pure_half_wide_ilp_shared_half2_
//                             atomic_best (PROMOTED dispatcher, no merged
//                             kernel: below 4096 rows, calls
//                             mv_fp32_16_pure_half_wide_ilp_shared_
//                             bigtile_half2_atomic (atomicAdd combine + 4x
//                             larger shared-memory tile); at/above it,
//                             calls mv_fp32_16_pure_half_wide_ilp_shared_
//                             half2_atomic_wide4x (same atomicAdd combine,
//                             quadrupled col_splits instead). Both callees
//                             reuse mv_fp32_16_pure_half_wide_ilp_shared_
//                             half2_atomic's own genuine __half compute,
//                             safe fp32_16_half_only decode, and __hfma2
//                             packing unchanged -- SAME arithmetic as
//                             mv_fp32_16_pure_half_wide_ilp_shared_half2,
//                             the "pure_half_half2" leg measured below and
//                             still what every other TOMOS leg's own ratio
//                             line compares against. Built after Nsight
//                             Compute profiling of the PRIOR "TOMOS FP16"
//                             (plain half2_atomic) found it only reaching
//                             0.59 waves / 46.86% achieved occupancy at
//                             M=4096. NOTE the 4096 threshold matches that
//                             profiling shape, not a measured crossover
//                             between the two callees -- UNTESTED as of
//                             this promotion. See its own comment in
//                             fp32_16_cuda_kernels.cu.)
//   TOMOS FP32/FP16        = mv_fp32_16_hybrid_wide_ilp_shared_half2_
//                             atomic_branchless_best (PROMOTED dispatcher:
//                             below 4096 rows, calls mv_fp32_16_hybrid_
//                             wide_ilp_shared_bigtile_half2_atomic_
//                             branchless; at/above it, calls mv_fp32_16_
//                             hybrid_wide_ilp_shared_half2_atomic_
//                             branchless_wide4x -- same wh-only extended-
//                             exponent reconstruction the branchy kernel
//                             used, but computes BOTH the cheap and
//                             extended paths unconditionally per element,
//                             discarding the unwanted one via arithmetic
//                             masking instead of a data-dependent branch.
//                             Nsight Compute profiling found the branchy
//                             kernel's own Branch Efficiency only 54.09%
//                             at M=4096 (this benchmark's own 10%
//                             control=1 rate makes ~97% of warps pay for
//                             both code paths serially). CONFIRMED on
//                             real A100 hardware: beats the branchy
//                             mv_fp32_16_hybrid_wide_ilp_shared_half2_
//                             atomic_best by +5.7% to +18.7% at every
//                             square size measured. NOTE this took two
//                             attempts -- mv_fp32_16_hybrid_wide_ilp_
//                             shared_half2_atomic_branchless_fastx_best
//                             (same masking plus a cached float x_tile)
//                             was briefly promoted here first and
//                             CONFIRMED NEGATIVE instead (2%-17% slower)
//                             before being reverted and replaced with
//                             THIS dispatcher, which adds no such
//                             caching. See all relevant kernels' own
//                             comments in fp32_16_cuda_kernels.cu.)
//   TOMOS FP32/FP16 (fast) = mv_fp32_16_pure_half_wide_ilp_shared_half2_
//                             atomic_fasthalf_best (NEW dispatcher, ports
//                             "TOMOS FP16"'s own atomicAdd-combine
//                             promotion to the unconditional fp32_16_
//                             fast_half decode -- UNSAFE on arbitrary
//                             data, own dedicated safe-range wh, see its
//                             own generator below -- same 4096-row
//                             dispatch threshold as "TOMOS FP16" itself,
//                             carried over unchanged, not independently
//                             re-derived. UNTESTED as of this promotion.
//                             See its own comment in fp32_16_cuda_
//                             kernels.cu.)
//   TOMOS FP32/BF16        = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
//                             atomic_best_v2 (NEW dispatcher, REQUESTED as
//                             a separate, distinctly-named duplicate of
//                             "TOMOS BF16" below, explicitly confirmed
//                             identical on wh -- no decode/compute axis
//                             exists to differentiate these two series
//                             for bf16, unlike fp16's own reconstruction/
//                             fast-decode axes. UNTESTED as of this
//                             promotion. See its own comment in fp32_16_
//                             cuda_kernels.cu.)
//   TOMOS BF16             = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
//                             atomic_best (NEW dispatcher, ports "TOMOS
//                             FP16"'s own atomicAdd-combine promotion to
//                             genuine bf16 compute -- same 4096-row
//                             dispatch threshold, carried over unchanged.
//                             UNLIKE the fp16 family, no 3-way wide/
//                             tiled/narrow_vec dispatch -- always split-K,
//                             matching the plain bf16 kernel's own scope.
//                             UNTESTED as of this promotion. See its own
//                             comment in fp32_16_cuda_kernels.cu.)
// plus cuBLAS (Hgemm, n=1) and cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1) as
// the real fp16/bf16 GEMV baselines -- same "no native Hgemv/Bgemv, so
// GEMV-via-GEMM(n=1)" convention the TRANSPOSE file already established,
// here with CUBLAS_OP_N (NO_TRANSPOSE) instead of CUBLAS_OP_T.
//
// Plus one "test" series (not one of the four plotted TOMOS ones):
//   test = mv_fp32_16_pure_half_wide_ilp_shared_branchless (SAME safe
//          fp32_16_half_only semantics as "TOMOS FP16", via the bit-exact
//          fp32_16_half_only_branchless reformulation -- unconditional
//          bitwise masks instead of an if/else chain -- but the plain
//          scalar-__half compute body, no __hfma2 packing). Answers
//          whether removing just the decode branch (while staying
//          correct on arbitrary data, unlike "(fast)"'s unsafe shortcut)
//          recovers any of "(fast)"'s own measured gain over "TOMOS
//          FP16" -- see its own comment in fp32_16_cuda_kernels.cu.
//
// All four TOMOS legs use the `_workspace` sibling of their own kernel
// (mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace, etc.), not the
// plain function -- first real NO_TRANSPOSE numbers from this file showed
// every wide_ilp_shared-family series trailing cuBLAS far more at small/
// medium M than large M (e.g. "(fast)" at 0.37x-0.58x of cuBLAS for
// M<=4096, climbing to 0.8-0.9x by M>=16384), the classic signature of a
// fixed per-call cost, not a compute/decode problem: the plain functions'
// own wide (split-K) path does a fresh cudaMalloc+cudaFree of their
// "partial" scratch buffer on EVERY call, timed as part of this file's own
// `time_call` reps loop. The `_workspace` siblings take that same buffer
// as a caller-owned parameter instead -- allocated once per shape here,
// outside the timed loop, and reused across every rep -- eliminating that
// overhead entirely. See mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_
// workspace's own comment in fp32_16_cuda_kernels.cu for the full
// derivation (mirrors mv_fp32_16_accurate_workspace's own already-
// established fix for the identical issue on the fp32_16 reconstruction
// tier).
//
// A semantics note (same caveat the TRANSPOSE file's own header carries):
// the TOMOS legs take float* x/y (only A is 16-bit-encoded) but cuBLAS's
// Hgemm/GemmEx(bf16) require ALL of A/x/y to be half/bf16-typed -- so both
// cuBLAS legs use their own independent, separately-generated half/bf16
// A/x/y buffers, not a converted copy of the TOMOS legs' own data.
//
// Correctness: NOT re-tested here (throughput-only, same convention every
// *_vs_cublas_ref.cu file in this codebase follows) -- see verify_mv_
// fp32_16_gemv.cu for the NO_TRANSPOSE correctness cases added alongside
// this file for mv_fp32_16_hybrid_wide_ilp_shared and mv_fp32_bf16_pure_
// bf16_wide_ilp_shared (the two genuinely new kernels here); mv_fp32_16_
// pure_half_wide_ilp_shared_half2/_fasthalf already have existing
// NO_TRANSPOSE coverage there (predates this file).
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_fp32_16_pure_half_
// vs_cublas_ref.sh.
//
// Usage:
//   mv_fp32_16_pure_half_vs_cublas_ref
//       Sweeps two shape families, SAME lists mv_fp32_16_pure_half_
//       transpose_vs_cublas_ref.cu uses: square M=K=1024/1536/2048/3072/
//       4096/6144/8192/12288/16384/24576/32768/49152/65536/131072, then a
//       tall-and-thin sweep with K held fixed at 512 and then 4096, M swept
//       across 1024/2048/4096/8192/16384/32768/65536/131072/262144/524288/
//       1048576.
//   mv_fp32_16_pure_half_vs_cublas_ref 8192
//       Just that one square shape (M=K=8192).
//   mv_fp32_16_pure_half_vs_cublas_ref --tall 512
//       Just the tall-and-thin sweep at K=512, M swept across the default
//       size list.
//   mv_fp32_16_pure_half_vs_cublas_ref --tall 512 1048576
//       Just the single tall-and-thin shape M=1048576, K=512.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace(TransposeType trans,
                                                           double alpha, dim<2> size,
                                                           const uint16_t* wh,
                                                           size_t ld_a, const float* x,
                                                           size_t ld_x, double beta,
                                                           float* y, size_t ld_y,
                                                           double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace(TransposeType trans,
                                                              double alpha, dim<2> size,
                                                              const uint16_t* wh,
                                                              size_t ld_a, const float* x,
                                                              size_t ld_x, double beta,
                                                              float* y, size_t ld_y,
                                                              double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_hybrid_wide_ilp_shared_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_workspace(TransposeType trans, double alpha,
                                                 dim<2> size, const uint16_t* wh,
                                                 size_t ld_a, const float* x,
                                                 size_t ld_x, double beta, float* y,
                                                 size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic(TransposeType trans, double alpha,
                                                     dim<2> size, const uint16_t* wh,
                                                     size_t ld_a, const float* x,
                                                     size_t ld_x, double beta, float* y,
                                                     size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic(TransposeType trans, double alpha,
                                                             dim<2> size, const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta, float* y,
                                                             size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best(TransposeType trans, double alpha,
                                                          dim<2> size, const uint16_t* wh,
                                                          size_t ld_a, const float* x,
                                                          size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless(TransposeType trans, double alpha,
                                                                dim<2> size, const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y,
                                                                size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta, float* y,
                                                                       size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta, float* y,
                                                                       size_t ld_y); }  // namespace experimental

void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(TransposeType trans, double alpha,
                                                                     dim<2> size, const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx(TransposeType trans, double alpha,
                                                                      dim<2> size, const uint16_t* wh,
                                                                      size_t ld_a, const float* x,
                                                                      size_t ld_x, double beta, float* y,
                                                                      size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace(TransposeType trans, double alpha,
                                                       dim<2> size, const uint16_t* wh,
                                                       size_t ld_a, const float* x,
                                                       size_t ld_x, double beta, float* y,
                                                       size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic(TransposeType trans, double alpha,
                                                    dim<2> size, const uint16_t* wh,
                                                    size_t ld_a, const float* x,
                                                    size_t ld_x, double beta, float* y,
                                                    size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x(TransposeType trans, double alpha,
                                                           dim<2> size, const uint16_t* wh,
                                                           size_t ld_a, const float* x,
                                                           size_t ld_x, double beta, float* y,
                                                           size_t ld_y); }  // namespace experimental

void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(TransposeType trans, double alpha,
                                                         dim<2> size, const uint16_t* wh,
                                                         size_t ld_a, const float* x,
                                                         size_t ld_x, double beta, float* y,
                                                         size_t ld_y);
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2(TransposeType trans, double alpha,
                                                       dim<2> size, const uint16_t* wh,
                                                       size_t ld_a, const float* x,
                                                       size_t ld_x, double beta, float* y,
                                                       size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2(TransposeType trans, double alpha,
                                                               dim<2> size, const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2(TransposeType trans, double alpha,
                                                              dim<2> size, const uint16_t* wh,
                                                              size_t ld_a, const float* x,
                                                              size_t ld_x, double beta, float* y,
                                                              size_t ld_y); }  // namespace experimental

void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y);
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace(TransposeType trans,
                                                                double alpha, dim<2> size,
                                                                const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta,
                                                                float* y, size_t ld_y,
                                                                double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace(TransposeType trans,
                                                                      double alpha, dim<2> size,
                                                                      const uint16_t* wh,
                                                                      size_t ld_a, const float* x,
                                                                      size_t ld_x, double beta,
                                                                      float* y, size_t ld_y,
                                                                      double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace(TransposeType trans,
                                                                    double alpha, dim<2> size,
                                                                    const uint16_t* wh,
                                                                    size_t ld_a, const float* x,
                                                                    size_t ld_x, double beta,
                                                                    float* y, size_t ld_y,
                                                                    double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace(TransposeType trans,
                                                                   double alpha, dim<2> size,
                                                                   const uint16_t* wh,
                                                                   size_t ld_a, const float* x,
                                                                   size_t ld_x, double beta,
                                                                   float* y, size_t ld_y,
                                                                   double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace_size(TransposeType trans,
                                                                                  dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace(TransposeType trans,
                                                                            double alpha, dim<2> size,
                                                                            const uint16_t* wh,
                                                                            size_t ld_a, const float* x,
                                                                            size_t ld_x, double beta,
                                                                            float* y, size_t ld_y,
                                                                            double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace(TransposeType trans, double alpha,
                                                                dim<2> size, const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y,
                                                                size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic(TransposeType trans,
                                                                double alpha, dim<2> size,
                                                                const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta,
                                                                float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x(TransposeType trans,
                                                               double alpha, dim<2> size,
                                                               const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta,
                                                               float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x(TransposeType trans,
                                                               double alpha, dim<2> size,
                                                               const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta,
                                                               float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                             double alpha, dim<2> size,
                                                             const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta,
                                                             float* y, size_t ld_y);
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf(TransposeType trans,
                                                                 double alpha, dim<2> size,
                                                                 const uint16_t* wh,
                                                                 size_t ld_a, const float* x,
                                                                 size_t ld_x, double beta,
                                                                 float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf(TransposeType trans,
                                                                        double alpha, dim<2> size,
                                                                        const uint16_t* wh,
                                                                        size_t ld_a, const float* x,
                                                                        size_t ld_x, double beta,
                                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x(TransposeType trans,
                                                                       double alpha, dim<2> size,
                                                                       const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta,
                                                                       float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(TransposeType trans,
                                                                     double alpha, dim<2> size,
                                                                     const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta,
                                                                     float* y, size_t ld_y);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

namespace {

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

// wr is written but never read by any kernel in this file (every kernel
// here only ever consumes wh) -- fp32_16_split still requires a wr output
// parameter, so a scratch buffer is allocated, filled, and freed
// immediately, same convention mv_fp32_16_pure_half_transpose_vs_cublas_
// ref.cu's own gen_fp32_16_wh already uses.
__global__ void gen_fp32_16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch);
}

// Safe-normal-range variant of pseudo_rand -- required by "TOMOS FP32/FP16
// (fast)" (mv_fp32_16_pure_half_wide_ilp_shared_fasthalf): fp32_16_fast_
// half decodes unconditionally as if control were always 0, so its own
// data must PROVABLY never be extended/degraded-encoded -- same convention
// mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu's own pseudo_rand_safe_
// normal uses (copied rather than shared -- standalone self-contained
// binaries, no shared translation unit between them).
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
}

__global__ void gen_fp32_16_wh_safe_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_safe_normal(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh_safe(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_safe_kernel<<<blocks, threads>>>(n, wh, wr_scratch);
}

// Plain (not safe-range) generator for mv_fp32_bf16_pure_bf16_wide_ilp_
// shared ("TOMOS FP32/BF16") -- unlike the fp16 "(fast)" leg, that kernel
// has no unsafe-on-arbitrary-data caveat (bf16's truncation decode is
// always valid -- see fp32_bf16_bf16_only's own comment), so plain
// pseudo_rand data suffices, same convention gen_fp32_16_wh above uses.
__global__ void gen_fp32_bf16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_bf16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch);
}

// Mixed normal/extended-range generator for mv_fp32_16_hybrid_wide_ilp_
// shared ("TOMOS FP32/FP16") -- ~10% of values get a magnitude beyond
// fp16's own ~65504 max (control=1, exercising the wh-only extended-
// exponent reconstruction path), the rest stay in [-1,1] (control=0, the
// cheap __half path) -- same convention mv_fp32_16_pure_half_transpose_
// vs_cublas_ref.cu's own gen_fp32_16_wh_hybrid uses; plain pseudo_rand
// data would never trigger the branch this leg exists to test at all.
__device__ float pseudo_rand_hybrid_mixed(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    if ((s % 10u) == 0u) {
        uint32_t s2 = s * 2246822519u + 3266489917u;
        s2 ^= s2 << 13;
        s2 ^= s2 >> 17;
        s2 ^= s2 << 5;
        float mag = 1.0e5f + (static_cast<float>(s2 % 100001u) / 100000.0f) * 1.0e8f;
        return (s & 0x10000u) ? mag : -mag;
    }
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_fp32_16_wh_hybrid_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_hybrid_mixed(i), wh[i], wr[i]);
}
void gen_fp32_16_wh_hybrid(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_hybrid_kernel<<<blocks, threads>>>(n, wh, wr);
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

// bf16 counterpart of gen_half, for the "cuBLAS BF16" leg's own
// independent A/x/y buffers.
__global__ void gen_bf16_kernel(size_t n, __nv_bfloat16* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(pseudo_rand(i));
}
void gen_bf16(size_t n, __nv_bfloat16* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_bf16_kernel<<<blocks, threads>>>(n, out);
}

// Caller-controlled in place of every call site's own `reps` argument
// (still accepted, for source compatibility with the sibling file's own
// call sites, but ignored -- see main()'s own argv-filtering preamble
// for where these are actually set, from --warmup/--iters).
int g_warmup_runs = 10;
int g_iterations = 10;

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int /*reps*/) {
    for (int w = 0; w < g_warmup_runs; ++w) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < g_iterations; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / g_iterations;
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k follow BLAS convention: A is M x K, column-major, ld_a=m. For
// NO_TRANSPOSE (y = alpha*A*x + beta*y): x has length K (the reduction
// dimension), y has length M (the output dimension) -- the mirror image
// of mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu's own run_shape
// (x length M, y length K, for TRANSPOSE). A's own representation (wh
// for the TOMOS legs, A_h/x_h/y_h or A_bf/x_bf/y_bf for the cuBLAS legs)
// is transient, allocated right before and freed right after its own
// timing block -- same staggered-allocation discipline every run_shape
// in this codebase's benchmark files already uses.
void run_shape(size_t m, size_t k, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);

    int lda = static_cast<int>(m);
    size_t maxdim = std::max(m, k);
    int reps = (maxdim <= 4096) ? 50 : (maxdim <= 16384 ? 20 : 5);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k);

    std::printf("  %s\n", shape_desc);

    double ms_pure_half_half2 = -1.0, ms_hybrid = -1.0, ms_pure_half_fasthalf = -1.0, ms_fp32_bf16_pure_bf16 = -1.0,
           ms_cublas_hgemm = -1.0, ms_cublas_gemmex_bf16 = -1.0, ms_pure_half_branchless = -1.0,
           ms_pure_half_half2_branchless = -1.0, ms_pure_half_half2_vecrow = -1.0,
           ms_pure_half_half2_vec2rows = -1.0, ms_pure_half_half2_wide2x = -1.0,
           ms_pure_half_half2_wide4x = -1.0, ms_bigtile_half2 = -1.0,
           ms_bigtile_half2_adaptive = -1.0, ms_half2_best = -1.0, ms_half2_atomic = -1.0,
           ms_bigtile_half2_atomic = -1.0, ms_half2_atomic_wide2x = -1.0,
           ms_half2_atomic_wide4x = -1.0, ms_half2_atomic_best = -1.0,
           ms_bf16_atomic = -1.0, ms_bf16_bigtile_atomic = -1.0,
           ms_bf16_atomic_wide4x = -1.0, ms_bf16_atomic_best = -1.0,
           ms_fasthalf_atomic = -1.0, ms_fasthalf_bigtile_atomic = -1.0,
           ms_fasthalf_atomic_wide4x = -1.0, ms_fasthalf_atomic_best = -1.0,
           ms_bf16_atomic_v2 = -1.0, ms_bf16_bigtile_atomic_v2 = -1.0,
           ms_bf16_atomic_wide4x_v2 = -1.0, ms_bf16_atomic_best_v2 = -1.0,
           ms_hybrid_atomic = -1.0, ms_hybrid_bigtile_atomic = -1.0,
           ms_hybrid_atomic_wide4x = -1.0, ms_hybrid_atomic_best = -1.0,
           ms_hybrid_atomic_branchless = -1.0, ms_hybrid_branchless_bigtile = -1.0,
           ms_hybrid_branchless_wide4x = -1.0, ms_hybrid_branchless_best = -1.0,
           ms_hybrid_branchless_fastx = -1.0,
           ms_hybrid_branchless_fastx_bigtile = -1.0, ms_hybrid_branchless_fastx_wide4x = -1.0,
           ms_hybrid_branchless_fastx_best = -1.0, ms_hybrid_branchless_intmask = -1.0,
           ms_hybrid_branchless_intmask_bigtile = -1.0, ms_hybrid_branchless_intmask_wide4x = -1.0,
           ms_hybrid_branchless_intmask_best = -1.0, ms_hybrid_branchless_intmask_walign = -1.0,
           ms_hybrid_branchless_intmask_walign_best = -1.0;

    float *x = nullptr, *y = nullptr;
    bool ok_xy = alloc_or_skip(reinterpret_cast<void**>(&x), k * sizeof(float), "x", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&y), m * sizeof(float), "y", shape_desc);
    if (ok_xy) {
        gen_float(k, x);
        cudaMemset(y, 0, m * sizeof(float));
        cudaDeviceSynchronize();

        // "TOMOS FP16" -- mv_fp32_16_pure_half_wide_ilp_shared_half2's own
        // NO_TRANSPOSE path: genuine __half arithmetic, wh decoded via the
        // safe fp32_16_half_only (a direct bit reinterpret when control=0,
        // an Infinity/zero placeholder otherwise -- never accurate for
        // extended-range values, but well-defined on ANY data), 2 scalar
        // half FMAs packed per __hfma2. Plain uniform-[-1,1] wh.
        uint16_t *wh = nullptr, *wr_scratch = nullptr;
        bool ok_wh = alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t),
                                   "pure_half_wide_ilp_shared_half2 (wh)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&wr_scratch), m * k * sizeof(uint16_t),
                                  "pure_half_wide_ilp_shared_half2 (wr scratch)", shape_desc);
        if (ok_wh) {
            gen_fp32_16_wh(m * k, wh, wr_scratch);
            cudaDeviceSynchronize();
            cudaFree(wr_scratch);

            size_t half2_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace_size(TransposeType::NO_TRANSPOSE,
                                                                                dim<2>{m, k});
            double* half2_ws = nullptr;
            if (half2_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_ws), half2_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2 (workspace)", shape_desc)) {
                ms_pure_half_half2 = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2 = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_half2, flops / ms_pure_half_half2 / 1e6);
                if (half2_ws) cudaFree(half2_ws);
            }

            // "test" -- mv_fp32_16_pure_half_wide_ilp_shared_branchless's
            // own NO_TRANSPOSE path: SAME safe fp32_16_half_only semantics
            // as "TOMOS FP16" above (bit-exact match, via fp32_16_half_
            // only_branchless -- unconditional bitwise masks instead of an
            // if/else chain), but the plain scalar-__half compute body
            // (no __hfma2 packing). MEASURED on real A100 hardware to be
            // 20%-30% SLOWER than "TOMOS FP16" at every square size -- NOT
            // a decode-cost finding: this kernel is built on the OLDER,
            // pre-half2 scalar-accumulator body, so removing the branch is
            // confounded with losing __hfma2's 2-real-FMAs/instruction
            // packing, and that loss dominates. See "test (half2,
            // branchless)" below for the fair isolation. Same wh as "TOMOS
            // FP16" above (same safe semantics, no dedicated generator
            // needed).
            size_t branchless_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace_size(TransposeType::NO_TRANSPOSE,
                                                                                     dim<2>{m, k});
            double* branchless_ws = nullptr;
            if (branchless_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&branchless_ws), branchless_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_branchless (workspace)", shape_desc)) {
                ms_pure_half_branchless = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        branchless_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_branchless = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_branchless, flops / ms_pure_half_branchless / 1e6);
                if (branchless_ws) cudaFree(branchless_ws);
            }

            // "test (half2, branchless)" -- the fair isolation: mv_fp32_16_
            // pure_half_wide_ilp_shared_half2_branchless_workspace's own
            // NO_TRANSPOSE path, IDENTICAL to "TOMOS FP16" above (same
            // __hfma2-packed body) with ONLY the decode swapped to
            // fp32_16_half_only_branchless -- isolates the branch/no-
            // branch question alone, unconfounded by the packing loss the
            // "test" leg above has. Same wh.
            size_t half2_branchless_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* half2_branchless_ws = nullptr;
            if (half2_branchless_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_branchless_ws),
                             half2_branchless_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_branchless (workspace)", shape_desc)) {
                ms_pure_half_half2_branchless = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_branchless_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_half2_branchless, flops / ms_pure_half_half2_branchless / 1e6);
                if (half2_branchless_ws) cudaFree(half2_branchless_ws);
            }

            // "test (half2, vecrow)" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_vecrow_workspace's own NO_TRANSPOSE path:
            // targets the MEMORY side instead of decode -- vecrow's own
            // shuffle-based row-grouped load (1 of every 4 threads issues
            // a single 8-byte vectorized load covering its own row and
            // the next 3, the rest retrieve their value via __shfl_sync
            // instead of each issuing an independent scalar 16-bit load)
            // combined with "TOMOS FP16"'s own __hfma2-packed arithmetic
            // (not vecrow's own plain scalar accumulate, avoiding the same
            // packing-loss confound "test" above has). ncu-informed
            // hypothesis (already recorded elsewhere in this file): fewer,
            // wider load requests may complete more efficiently than many
            // independently-issued small ones, even though total bytes
            // moved are identical. Same wh, requires lda % 4 == 0 (true
            // for every shape this file sweeps).
            size_t half2_vecrow_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* half2_vecrow_ws = nullptr;
            if (half2_vecrow_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_vecrow_ws),
                             half2_vecrow_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_vecrow (workspace)", shape_desc)) {
                ms_pure_half_half2_vecrow = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_vecrow_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_half2_vecrow, flops / ms_pure_half_half2_vecrow / 1e6);
                if (half2_vecrow_ws) cudaFree(half2_vecrow_ws);
            }

            // "test (half2, vec2rows)" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_vec2rows_workspace's own NO_TRANSPOSE path:
            // ports temp-dependencies/randls2.0's own hgemv_cm_vec2rows_
            // fp16 row-vectorization mechanism -- each thread owns 2
            // ADJACENT rows and loads BOTH via a single vectorized read,
            // no cross-thread __shfl_sync needed (unlike "test (half2,
            // vecrow)" above, which was a severe negative result) since
            // the same thread already owns both values it needs. Same wh,
            // requires lda % 2 == 0 (true for every shape this file
            // sweeps).
            size_t half2_vec2rows_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* half2_vec2rows_ws = nullptr;
            if (half2_vec2rows_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_vec2rows_ws),
                             half2_vec2rows_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_vec2rows (workspace)", shape_desc)) {
                ms_pure_half_half2_vec2rows = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_vec2rows_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_half2_vec2rows, flops / ms_pure_half_half2_vec2rows / 1e6);
                if (half2_vec2rows_ws) cudaFree(half2_vec2rows_ws);
            }

            // "test (half2, wide2x)" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_wide2x_workspace's own NO_TRANSPOSE path:
            // REUSES "TOMOS FP16"'s own kernel completely unchanged, only
            // doubles col_splits (more, smaller column chunks, more total
            // blocks launched -- MORE parallelism, not less, unlike every
            // row-blocking "test" above). Same wh.
            size_t half2_wide2x_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* half2_wide2x_ws = nullptr;
            if (half2_wide2x_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_wide2x_ws),
                             half2_wide2x_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_wide2x (workspace)", shape_desc)) {
                ms_pure_half_half2_wide2x = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_wide2x_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_half2_wide2x, flops / ms_pure_half_half2_wide2x / 1e6);
                if (half2_wide2x_ws) cudaFree(half2_wide2x_ws);
            }

            // "test (half2, wide4x)" -- same as "test (half2, wide2x)"
            // above, quadruples col_splits instead of doubling it,
            // bracketing the high side. Same wh.
            size_t half2_wide4x_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* half2_wide4x_ws = nullptr;
            if (half2_wide4x_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_wide4x_ws),
                             half2_wide4x_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_wide4x (workspace)", shape_desc)) {
                ms_pure_half_half2_wide4x = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_wide4x_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_half2_wide4x, flops / ms_pure_half_half2_wide4x / 1e6);
                if (half2_wide4x_ws) cudaFree(half2_wide4x_ws);
            }

            // "test (bigtile, half2)" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_bigtile_half2_workspace's own NO_TRANSPOSE path:
            // "TOMOS FP16"'s own __hfma2 packing combined with a 4x
            // larger shared-memory x tile (1024 instead of 256), cutting
            // __syncthreads() frequency 4x -- same thread count/grid as
            // "TOMOS FP16" itself, only the tile width and its own
            // syncing cadence differ. Same wh.
            size_t bigtile_half2_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* bigtile_half2_ws = nullptr;
            if (bigtile_half2_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&bigtile_half2_ws),
                             bigtile_half2_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_bigtile_half2 (workspace)", shape_desc)) {
                ms_bigtile_half2 = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        bigtile_half2_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2 = %9.3fms (%10.1f GFlop/s)\n",
                    ms_bigtile_half2, flops / ms_bigtile_half2 / 1e6);
                if (bigtile_half2_ws) cudaFree(bigtile_half2_ws);
            }

            // "test (bigtile, half2, adaptive)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_bigtile_half2_adaptive_workspace's own
            // NO_TRANSPOSE path: bigtile_half2's own kernel (safe win
            // everywhere) combined with wide4x's own col_splits
            // quadrupling, but only above kMvHalf2AdaptiveWideMultiplier
            // Threshold rows (where that multiplier is a confirmed win,
            // not the confirmed loss it is below it). Tests whether the
            // two independent wins stack. Same wh.
            size_t bigtile_half2_adaptive_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* bigtile_half2_adaptive_ws = nullptr;
            if (bigtile_half2_adaptive_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&bigtile_half2_adaptive_ws),
                             bigtile_half2_adaptive_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_bigtile_half2_adaptive (workspace)", shape_desc)) {
                ms_bigtile_half2_adaptive = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        bigtile_half2_adaptive_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive = %9.3fms (%10.1f GFlop/s)\n",
                    ms_bigtile_half2_adaptive, flops / ms_bigtile_half2_adaptive / 1e6);
                if (bigtile_half2_adaptive_ws) cudaFree(bigtile_half2_adaptive_ws);
            }

            // "TOMOS FP16, best-of" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_best_workspace's own NO_TRANSPOSE path:
            // DISPATCHES to whichever already-proven kernel is better for
            // this shape (bigtile_half2 below kMvHalf2BestSquare
            // CrossoverThreshold rows, half2_wide4x at/above it) instead
            // of merging the two mechanisms (which "test (bigtile, half2,
            // adaptive)" above confirmed does NOT stack). Same wh.
            size_t half2_best_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace_size(TransposeType::NO_TRANSPOSE,
                                                                                     dim<2>{m, k});
            double* half2_best_ws = nullptr;
            if (half2_best_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_best_ws),
                             half2_best_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_half2_best (workspace)", shape_desc)) {
                ms_half2_best = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_best_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_best = %9.3fms (%10.1f GFlop/s)\n",
                           ms_half2_best, flops / ms_half2_best / 1e6);
                if (half2_best_ws) cudaFree(half2_best_ws);
            }

            // "test (half2, atomic)" -- mv_fp32_16_pure_half_wide_ilp_
            // shared_half2_atomic's own NO_TRANSPOSE path: checked
            // temp-dependencies/randls2.0's own gemv_fp16_cm_pair_
            // ktiled_atomic, which independently uses the SAME atomicAdd-
            // instead-of-reduce-kernel combine strategy mv_fp32_16_pure_
            // half_wide_ilp_shared_atomic already tried in this codebase
            // (cross-validating it) -- but paired here with half2's own
            // __hfma2 packing instead of that kernel's older scalar body.
            // No workspace needed (no buffer to own -- atomicAdd combines
            // directly into y, no malloc/free of any kind). Same wh.
            ms_half2_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic, flops / ms_half2_atomic / 1e6);

            // "test (bigtile, half2, atomic)" -- mv_fp32_16_pure_half_wide_
            // ilp_shared_bigtile_half2_atomic's own NO_TRANSPOSE path:
            // combines "TOMOS FP16"'s own atomicAdd combine with bigtile_
            // half2's own 4x-larger shared-memory tile. Unlike "test
            // (bigtile, half2, adaptive)" above (which stacked bigtile with
            // MORE col_splits/blocks and regressed), atomic combine doesn't
            // change col_splits/blocks at all -- same grid as bigtile_half2
            // itself -- so this pairing changes a different, independent
            // axis. No workspace needed (no buffer to own). Same wh.
            //
            // CONFIRMED NEGATIVE RESULT on real A100 hardware: no
            // meaningful win anywhere -- ratio vs half2_atomic stays
            // within noise (0.98x-1.02x) at small/medium square M and
            // every tall-and-thin shape, and is a mild but consistent
            // regression at large square M (M=16384-65536: 0.94x-0.96x).
            // See its own comment in fp32_16_cuda_kernels.cu.
            ms_bigtile_half2_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bigtile_half2_atomic, flops / ms_bigtile_half2_atomic / 1e6);

            // "test (half2, atomic, wide2x)" -- mv_fp32_16_pure_half_wide_
            // ilp_shared_half2_atomic_wide2x's own NO_TRANSPOSE path:
            // checked Nsight Compute profiling data for "TOMOS FP16" at
            // M=4096 -- the grid only fills 0.59 waves across the A100's
            // 108 SMs (achieved occupancy 46.86%, not register/shared-mem
            // limited). Doubles col_splits like "test (half2, wide2x)"
            // above does for the split+reduce kernel, but keeps atomicAdd
            // combine -- each extra split costs one more atomicAdd instead
            // of a whole extra malloc/launch/serial-reduce-loop, so this
            // should raise wave count without reintroducing the fixed
            // cost that made plain wide2x/wide4x lose at this size. No
            // workspace needed. Same wh.
            ms_half2_atomic_wide2x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_wide2x, flops / ms_half2_atomic_wide2x / 1e6);

            // "test (half2, atomic, wide4x)" -- same idea as "test (half2,
            // atomic, wide2x)" above, but quadruples col_splits instead of
            // doubling it, bracketing the high side. No workspace needed.
            // Same wh.
            ms_half2_atomic_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_wide4x, flops / ms_half2_atomic_wide4x / 1e6);

            // "test (half2, atomic, best-of)" -- mv_fp32_16_pure_half_wide_
            // ilp_shared_half2_atomic_best's own NO_TRANSPOSE path:
            // requested dispatcher for the atomic-combine family -- below
            // 4096 rows, calls bigtile_half2_atomic; at/above it, calls
            // half2_atomic_wide4x. No workspace needed. Same wh.
            ms_half2_atomic_best = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_best, flops / ms_half2_atomic_best / 1e6);

            cudaFree(wh);
        }

        // "TOMOS FP32/FP16" -- mv_fp32_16_hybrid_wide_ilp_shared's own
        // NO_TRANSPOSE path: wh-only extended-exponent reconstruction on
        // the control=1 path, paired __hfma2 packing when an adjacent pair
        // is both cheap (control=0). Own mixed normal/extended-range wh
        // (~10% deliberately beyond fp16's own range, exercising the
        // reconstruction path this kernel exists to test).
        uint16_t *wh_hybrid = nullptr, *wr_hybrid_scratch = nullptr;
        bool ok_wh_hybrid =
            alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid), m * k * sizeof(uint16_t),
                         "hybrid_wide_ilp_shared (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid_scratch), m * k * sizeof(uint16_t),
                         "hybrid_wide_ilp_shared (wr scratch)", shape_desc);
        if (ok_wh_hybrid) {
            gen_fp32_16_wh_hybrid(m * k, wh_hybrid, wr_hybrid_scratch);
            cudaDeviceSynchronize();
            cudaFree(wr_hybrid_scratch);

            size_t hybrid_ws_elems =
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_workspace_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* hybrid_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&hybrid_ws), hybrid_ws_elems * sizeof(double),
                              "hybrid_wide_ilp_shared (workspace)", shape_desc)) {
                ms_hybrid = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_workspace(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                      wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                                                                      y, 1, hybrid_ws);
                }, reps);
                std::printf("    mv_fp32_16_hybrid_wide_ilp_shared = %9.3fms (%10.1f GFlop/s)\n", ms_hybrid,
                           flops / ms_hybrid / 1e6);
                cudaFree(hybrid_ws);
            }

            // "TOMOS FP32/FP16", new version -- mv_fp32_16_hybrid_wide_
            // ilp_shared_half2_atomic's own NO_TRANSPOSE path: ports
            // "TOMOS FP32/FP16 (fast)"'s own atomicAdd-combine dispatcher
            // structure to the expensive (control-bit-checking) decode,
            // computing extended-range elements in float via fmaf instead
            // of fp32_16_fast_half's silent wrong answer or fp32_16_
            // half_only's own +-Infinity placeholder. No workspace needed.
            // Same wh_hybrid.
            ms_hybrid_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf("    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_hybrid_atomic, flops / ms_hybrid_atomic / 1e6);

            // "test (hybrid, bigtile, atomic)" -- mv_fp32_16_hybrid_wide_
            // ilp_shared_bigtile_half2_atomic's own NO_TRANSPOSE path.
            // Same wh_hybrid.
            ms_hybrid_bigtile_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf("    mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_hybrid_bigtile_atomic, flops / ms_hybrid_bigtile_atomic / 1e6);

            // "test (hybrid, atomic, wide4x)" -- mv_fp32_16_hybrid_wide_
            // ilp_shared_half2_atomic_wide4x's own NO_TRANSPOSE path. Same
            // wh_hybrid.
            ms_hybrid_atomic_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_atomic_wide4x, flops / ms_hybrid_atomic_wide4x / 1e6);

            // "TOMOS FP32/FP16, best-of" -- mv_fp32_16_hybrid_wide_ilp_
            // shared_half2_atomic_best's own NO_TRANSPOSE path: dispatcher
            // between the two legs above. Same wh_hybrid.
            ms_hybrid_atomic_best = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_atomic_best, flops / ms_hybrid_atomic_best / 1e6);

            // "test (hybrid, atomic, branchless)" -- mv_fp32_16_hybrid_
            // wide_ilp_shared_half2_atomic_branchless's own NO_TRANSPOSE
            // path: Nsight Compute profiling of the branchy atomic kernel
            // above at M=4096 found Branch Efficiency only 54.09% (this
            // benchmark's own 10% control=1 rate makes ~97% of warps pay
            // for both code paths serially). Computes both paths
            // unconditionally, discarding the unwanted one via arithmetic
            // masking instead of branching. CONFIRMED on real A100
            // hardware: +6.5% to +22% over the branchy kernel standalone
            // -- see the bigtile/wide4x/best legs right below that finish
            // this into "TOMOS FP32/FP16" itself now. No workspace
            // needed. Same wh_hybrid.
            ms_hybrid_atomic_branchless = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_atomic_branchless, flops / ms_hybrid_atomic_branchless / 1e6);

            // "TOMOS FP32/FP16" -- mv_fp32_16_hybrid_wide_ilp_shared_
            // bigtile_half2_atomic_branchless's own NO_TRANSPOSE path:
            // finishes the branchless base kernel's own CONFIRMED win
            // into a full bigtile/wide4x/best dispatcher, WITHOUT the
            // fastx caching that regressed (see further below). No
            // workspace needed. Same wh_hybrid.
            ms_hybrid_branchless_bigtile = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_bigtile, flops / ms_hybrid_branchless_bigtile / 1e6);

            // "test (hybrid, branchless, wide4x)" -- mv_fp32_16_hybrid_
            // wide_ilp_shared_half2_atomic_branchless_wide4x's own
            // NO_TRANSPOSE path. Same wh_hybrid.
            ms_hybrid_branchless_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_wide4x, flops / ms_hybrid_branchless_wide4x / 1e6);

            // "TOMOS FP32/FP16, best-of" -- mv_fp32_16_hybrid_wide_ilp_
            // shared_half2_atomic_branchless_best's own NO_TRANSPOSE
            // path: PROMOTED dispatcher between the two legs above. This
            // IS "TOMOS FP32/FP16" now -- CONFIRMED on real A100
            // hardware to beat mv_fp32_16_hybrid_wide_ilp_shared_half2_
            // atomic_best (the OLD branchy dispatcher, still what every
            // other series' own ratio line compares against) by +5.7% to
            // +18.7% at every square size measured. Same wh_hybrid.
            ms_hybrid_branchless_best = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_best, flops / ms_hybrid_branchless_best / 1e6);

            // "test (hybrid, branchless, fastx)" -- mv_fp32_16_hybrid_
            // wide_ilp_shared_half2_atomic_branchless_fastx's own
            // NO_TRANSPOSE path: attempted to finish the branchless base
            // kernel's own CONFIRMED win into a full bigtile/wide4x/best
            // family, plus cache a float-typed x tile to avoid
            // branchless's own redundant __half2float call. CONFIRMED
            // NEGATIVE RESULT instead (5%-26% slower than plain
            // branchless) -- see its own comment in fp32_16_cuda_
            // kernels.cu. No workspace needed. Same wh_hybrid.
            ms_hybrid_branchless_fastx = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_fastx, flops / ms_hybrid_branchless_fastx / 1e6);

            // "test (hybrid, branchless, fastx, bigtile)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_
            // fastx's own NO_TRANSPOSE path. Same wh_hybrid.
            ms_hybrid_branchless_fastx_bigtile = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_fastx_bigtile, flops / ms_hybrid_branchless_fastx_bigtile / 1e6);

            // "test (hybrid, branchless, fastx, wide4x)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x's
            // own NO_TRANSPOSE path. Same wh_hybrid.
            ms_hybrid_branchless_fastx_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_fastx_wide4x, flops / ms_hybrid_branchless_fastx_wide4x / 1e6);

            // "test (hybrid, branchless, fastx, best)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best's
            // own NO_TRANSPOSE path: dispatcher between the two legs
            // above. CONFIRMED NEGATIVE RESULT: 2%-17% slower than
            // mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best -- see
            // its own comment in fp32_16_cuda_kernels.cu. Same wh_hybrid.
            ms_hybrid_branchless_fastx_best = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_fastx_best, flops / ms_hybrid_branchless_fastx_best / 1e6);

            // "test (hybrid, branchless, intmask)" -- mv_fp32_16_hybrid_
            // wide_ilp_shared_half2_atomic_branchless_intmask's own
            // NO_TRANSPOSE path: same branchless combine as the CONFIRMED
            // base kernel above, but the extended path masks the raw
            // uint32 bit pattern with an integer AND instead of scaling
            // the decoded float by 0.0f/1.0f -- removes a standalone FP32
            // multiply ncu flagged as part of an ALU bottleneck (85.8%
            // Compute (SM) Throughput) in the promoted branchless kernel.
            // NOT YET CONFIRMED on real hardware. No workspace needed.
            // Same wh_hybrid.
            ms_hybrid_branchless_intmask = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask, flops / ms_hybrid_branchless_intmask / 1e6);

            // "test (hybrid, branchless, intmask, bigtile)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_
            // intmask's own NO_TRANSPOSE path. Same wh_hybrid.
            ms_hybrid_branchless_intmask_bigtile = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask_bigtile, flops / ms_hybrid_branchless_intmask_bigtile / 1e6);

            // "test (hybrid, branchless, intmask, wide4x)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // wide4x's own NO_TRANSPOSE path. Same wh_hybrid.
            ms_hybrid_branchless_intmask_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask_wide4x, flops / ms_hybrid_branchless_intmask_wide4x / 1e6);

            // "test (hybrid, branchless, intmask, best)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // best's own NO_TRANSPOSE path: dispatcher between the two
            // legs above. Candidate to replace mv_fp32_16_hybrid_wide_
            // ilp_shared_half2_atomic_branchless_best as "TOMOS FP32/
            // FP16" if it measures faster on real hardware. Same
            // wh_hybrid.
            ms_hybrid_branchless_intmask_best = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask_best, flops / ms_hybrid_branchless_intmask_best / 1e6);

            // "test (hybrid, branchless, intmask, walign)" -- mv_fp32_16_
            // hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_
            // walign's own NO_TRANSPOSE path: same intmask kernel as
            // intmask_wide4x above, but col_splits chosen to minimize
            // wasted GPU wave-slots instead of choose_col_splits_target's
            // own plain rounding. CONFIRMED NOISY/SHAPE-DEPENDENT RESULT,
            // NOT promoted: real wins/regressions alternate at nearby
            // moderate M with no clean threshold, but a consistent,
            // growing win at large tall-and-thin M -- see its own comment
            // in fp32_16_cuda_kernels.cu. No workspace needed. Same
            // wh_hybrid.
            ms_hybrid_branchless_intmask_walign = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask_walign, flops / ms_hybrid_branchless_intmask_walign / 1e6);

            // "test (hybrid, branchless, intmask, walign, best)" --
            // mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_
            // intmask_walign_best's own NO_TRANSPOSE path: dispatcher
            // between mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_
            // atomic_branchless_intmask (below threshold) and the walign
            // leg above (at/above it). CONFIRMED NOISY/SHAPE-DEPENDENT
            // RESULT, NOT promoted: real regressions at moderate square M
            // alternating with real wins at nearby sizes (no clean
            // threshold), but a consistent, growing win at large tall-
            // and-thin M -- see fp32_16_cuda_kernels.cu for the full
            // sweep. mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
            // branchless_intmask_best remains the strongest candidate for
            // "TOMOS FP32/FP16" since it wins everywhere. Same wh_hybrid.
            ms_hybrid_branchless_intmask_walign_best = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                    y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_hybrid_branchless_intmask_walign_best, flops / ms_hybrid_branchless_intmask_walign_best / 1e6);

            cudaFree(wh_hybrid);
        }

        // "TOMOS FP32/FP16 (fast)" -- mv_fp32_16_pure_half_wide_ilp_
        // shared_fasthalf's own NO_TRANSPOSE path: unconditional fp32_16_
        // fast_half decode (no control-bit branch at all) -- UNSAFE on
        // arbitrary data (its own contract requires control=0 for every
        // wh value), so this leg uses its OWN safe-range wh (mag in
        // [0.5,1.0], via gen_fp32_16_wh_safe) instead of the plain wh the
        // "TOMOS FP16" leg above uses.
        uint16_t *wh_fast = nullptr, *wr_fast_scratch = nullptr;
        bool ok_wh_fast =
            alloc_or_skip(reinterpret_cast<void**>(&wh_fast), m * k * sizeof(uint16_t),
                         "pure_half_wide_ilp_shared_fasthalf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_fast_scratch), m * k * sizeof(uint16_t),
                         "pure_half_wide_ilp_shared_fasthalf (wr scratch)", shape_desc);
        if (ok_wh_fast) {
            gen_fp32_16_wh_safe(m * k, wh_fast, wr_fast_scratch);
            cudaDeviceSynchronize();
            cudaFree(wr_fast_scratch);

            size_t fasthalf_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace_size(TransposeType::NO_TRANSPOSE,
                                                                                   dim<2>{m, k});
            double* fasthalf_ws = nullptr;
            if (fasthalf_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&fasthalf_ws), fasthalf_ws_elems * sizeof(double),
                             "pure_half_wide_ilp_shared_fasthalf (workspace)", shape_desc)) {
                ms_pure_half_fasthalf = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3,
                        y, 1, fasthalf_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_fasthalf = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_fasthalf, flops / ms_pure_half_fasthalf / 1e6);
                if (fasthalf_ws) cudaFree(fasthalf_ws);
            }

            // "TOMOS FP32/FP16 (fast)", new version -- mv_fp32_16_pure_
            // half_wide_ilp_shared_half2_atomic_fasthalf's own NO_TRANSPOSE
            // path: ports "TOMOS FP16"'s own atomicAdd-combine dispatcher
            // to the unsafe fasthalf decode. Same safe-range wh_fast as
            // the OLD "(fast)" leg above (required by fp32_16_fast_half's
            // own contract). No workspace needed. Same wh_fast.
            ms_fasthalf_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fasthalf_atomic, flops / ms_fasthalf_atomic / 1e6);

            // "test (fasthalf, bigtile, atomic)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_bigtile_half2_atomic_fasthalf's own
            // NO_TRANSPOSE path: adds a 4x-larger shared-memory tile on
            // top of the new "(fast)" atomicAdd combine. Same wh_fast.
            ms_fasthalf_bigtile_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf = %9.3fms (%10.1f GFlop/s)\n",
                ms_fasthalf_bigtile_atomic, flops / ms_fasthalf_bigtile_atomic / 1e6);

            // "test (fasthalf, atomic, wide4x)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_half2_atomic_fasthalf_wide4x's own
            // NO_TRANSPOSE path: quadruples col_splits on top of the new
            // "(fast)" atomicAdd combine. Same wh_fast.
            ms_fasthalf_atomic_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                ms_fasthalf_atomic_wide4x, flops / ms_fasthalf_atomic_wide4x / 1e6);

            // "TOMOS FP32/FP16 (fast), best-of" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_half2_atomic_fasthalf_best's own NO_TRANSPOSE
            // path: dispatcher between the two legs above. Same wh_fast.
            ms_fasthalf_atomic_best = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best = %9.3fms (%10.1f GFlop/s)\n",
                ms_fasthalf_atomic_best, flops / ms_fasthalf_atomic_best / 1e6);

            cudaFree(wh_fast);
        }

        // "TOMOS FP32/BF16" -- mv_fp32_bf16_pure_bf16_wide_ilp_shared's
        // own NO_TRANSPOSE path: genuine __nv_bfloat16/__nv_bfloat162
        // compute via fp32_bf16_bf16_only (bf16's only decode -- always
        // correct, no fast/safe split exists for it). Plain uniform data.
        uint16_t *wh_bf16 = nullptr, *wr_bf16_scratch = nullptr;
        bool ok_wh_bf16 =
            alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                         "fp32_bf16_pure_bf16_wide_ilp_shared (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_scratch), m * k * sizeof(uint16_t),
                         "fp32_bf16_pure_bf16_wide_ilp_shared (wr scratch)", shape_desc);
        if (ok_wh_bf16) {
            gen_fp32_bf16_wh(m * k, wh_bf16, wr_bf16_scratch);
            cudaDeviceSynchronize();
            cudaFree(wr_bf16_scratch);

            size_t bf16_ws_elems =
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace_size(TransposeType::NO_TRANSPOSE,
                                                                            dim<2>{m, k});
            double* bf16_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&bf16_ws), bf16_ws_elems * sizeof(double),
                              "fp32_bf16_pure_bf16_wide_ilp_shared (workspace)", shape_desc)) {
                ms_fp32_bf16_pure_bf16 = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace(TransposeType::NO_TRANSPOSE, 1.7,
                                                                           dim<2>{m, k}, wh_bf16,
                                                                           static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                                                                           bf16_ws);
                }, reps);
                std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared = %9.3fms (%10.1f GFlop/s)\n",
                           ms_fp32_bf16_pure_bf16, flops / ms_fp32_bf16_pure_bf16 / 1e6);
                cudaFree(bf16_ws);
            }

            // "TOMOS BF16" -- mv_fp32_bf16_pure_bf16_wide_ilp_shared_
            // atomic's own NO_TRANSPOSE path: ports "TOMOS FP16"'s own
            // atomicAdd-combine promotion to genuine bf16 compute. No
            // workspace needed (no buffer to own). Same wh_bf16.
            ms_bf16_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic, flops / ms_bf16_atomic / 1e6);

            // "test (bf16, bigtile, atomic)" -- mv_fp32_bf16_pure_bf16_
            // wide_ilp_shared_bigtile_atomic's own NO_TRANSPOSE path: adds
            // a 4x-larger shared-memory tile on top of "TOMOS BF16"'s own
            // atomicAdd combine. No workspace needed. Same wh_bf16.
            ms_bf16_bigtile_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_bigtile_atomic, flops / ms_bf16_bigtile_atomic / 1e6);

            // "test (bf16, atomic, wide4x)" -- mv_fp32_bf16_pure_bf16_
            // wide_ilp_shared_atomic_wide4x's own NO_TRANSPOSE path:
            // quadruples col_splits on top of "TOMOS BF16"'s own atomicAdd
            // combine. No workspace needed. Same wh_bf16.
            ms_bf16_atomic_wide4x = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic_wide4x, flops / ms_bf16_atomic_wide4x / 1e6);

            // "TOMOS BF16, best-of" -- mv_fp32_bf16_pure_bf16_wide_ilp_
            // shared_atomic_best's own NO_TRANSPOSE path: dispatcher
            // between the two legs above. No workspace needed. Same
            // wh_bf16.
            ms_bf16_atomic_best = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic_best, flops / ms_bf16_atomic_best / 1e6);

            // "TOMOS FP32/BF16", new version -- mv_fp32_bf16_pure_bf16_
            // wide_ilp_shared_atomic_v2's own NO_TRANSPOSE path: REQUESTED
            // separate, distinctly-named duplicate of "TOMOS BF16" above,
            // explicitly confirmed identical on wh (same decode, same
            // compute, same atomicAdd combine). No workspace needed. Same
            // wh_bf16.
            ms_bf16_atomic_v2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic_v2, flops / ms_bf16_atomic_v2 / 1e6);

            // "test (bf16, bigtile, atomic, v2)" -- mv_fp32_bf16_pure_
            // bf16_wide_ilp_shared_bigtile_atomic_v2's own NO_TRANSPOSE
            // path. Same wh_bf16.
            ms_bf16_bigtile_atomic_v2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_bigtile_atomic_v2, flops / ms_bf16_bigtile_atomic_v2 / 1e6);

            // "test (bf16, atomic, wide4x, v2)" -- mv_fp32_bf16_pure_bf16_
            // wide_ilp_shared_atomic_wide4x_v2's own NO_TRANSPOSE path.
            // Same wh_bf16.
            ms_bf16_atomic_wide4x_v2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic_wide4x_v2, flops / ms_bf16_atomic_wide4x_v2 / 1e6);

            // "TOMOS FP32/BF16, best-of", new version -- mv_fp32_bf16_
            // pure_bf16_wide_ilp_shared_atomic_best_v2's own NO_TRANSPOSE
            // path: dispatcher between the two legs above. Same wh_bf16.
            ms_bf16_atomic_best_v2 = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                    1);
            }, reps);
            std::printf("    mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_bf16_atomic_best_v2, flops / ms_bf16_atomic_best_v2 / 1e6);

            cudaFree(wh_bf16);
        }

        cudaFree(x);
        cudaFree(y);
    }

    // cuBLAS (Hgemm, n=1, NO_TRANSPOSE) -- the real fp16 GEMV baseline
    // (see this file's own header comment for why cublasSgemv is NOT used
    // here). Independent __half-typed A_h/x_h/y_h, NOT a converted copy
    // of the TOMOS legs' own wh/x/y.
    {
        __half *A_h = nullptr, *x_h = nullptr, *y_h = nullptr;
        bool ok_h = alloc_or_skip(reinterpret_cast<void**>(&A_h), m * k * sizeof(__half), "cublas Hgemm (A_h)",
                                  shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&x_h), k * sizeof(__half), "cublas Hgemm (x_h)",
                                 shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&y_h), m * sizeof(__half), "cublas Hgemm (y_h)",
                                 shape_desc);
        if (ok_h) {
            gen_half(m * k, A_h);
            gen_half(k, x_h);
            cudaMemset(y_h, 0, m * sizeof(__half));
            cudaDeviceSynchronize();

            __half h_alpha = __double2half(1.7);
            __half h_beta = __double2half(0.3);
            ms_cublas_hgemm = time_call(start, stop, [&]() {
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), 1, static_cast<int>(k), &h_alpha,
                           A_h, lda, x_h, static_cast<int>(k), &h_beta, y_h, static_cast<int>(m));
            }, reps);
            std::printf("    cublas (Hgemm n=1, NO_TRANSPOSE) = %9.3fms (%10.1f GFlop/s)\n", ms_cublas_hgemm,
                       flops / ms_cublas_hgemm / 1e6);

            cudaFree(A_h);
            cudaFree(x_h);
            cudaFree(y_h);
        }
    }

    // cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1, NO_TRANSPOSE) -- the real
    // bf16 GEMV baseline, same "no native Bgemv" reasoning as cublasHgemm
    // above. Independent __nv_bfloat16-typed A_bf/x_bf/y_bf, alpha/beta as
    // float (CUBLAS_COMPUTE_32F's own precision, not bf16).
    {
        __nv_bfloat16 *A_bf = nullptr, *x_bf = nullptr, *y_bf = nullptr;
        bool ok_bf = alloc_or_skip(reinterpret_cast<void**>(&A_bf), m * k * sizeof(__nv_bfloat16),
                                   "cublas GemmEx bf16 (A_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_bf), k * sizeof(__nv_bfloat16),
                                  "cublas GemmEx bf16 (x_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf), m * sizeof(__nv_bfloat16),
                                  "cublas GemmEx bf16 (y_bf)", shape_desc);
        if (ok_bf) {
            gen_bf16(m * k, A_bf);
            gen_bf16(k, x_bf);
            cudaMemset(y_bf, 0, m * sizeof(__nv_bfloat16));
            cudaDeviceSynchronize();

            float alpha_f = 1.7f;
            float beta_f = 0.3f;
            ms_cublas_gemmex_bf16 = time_call(start, stop, [&]() {
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), 1, static_cast<int>(k), &alpha_f,
                            A_bf, CUDA_R_16BF, lda, x_bf, CUDA_R_16BF, static_cast<int>(k), &beta_f, y_bf,
                            CUDA_R_16BF, static_cast<int>(m), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }, reps);
            std::printf("    cublas (GemmEx CUDA_R_16BF n=1, NO_TRANSPOSE) = %9.3fms (%10.1f GFlop/s)\n",
                       ms_cublas_gemmex_bf16, flops / ms_cublas_gemmex_bf16 / 1e6);

            cudaFree(A_bf);
            cudaFree(x_bf);
            cudaFree(y_bf);
        }
    }

    if (ms_pure_half_half2 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_branchless > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_branchless vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_branchless) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_branchless > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_branchless vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_branchless) / (flops / ms_pure_half_half2));
    if (ms_pure_half_branchless > 0 && ms_pure_half_fasthalf > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_branchless vs fp32_16_pure_half_wide_ilp_shared_fasthalf: %.3fx\n",
            (flops / ms_pure_half_branchless) / (flops / ms_pure_half_fasthalf));
    if (ms_pure_half_half2_branchless > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_branchless vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2_branchless) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_half2_branchless > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_branchless vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_half2_branchless) / (flops / ms_pure_half_half2));
    if (ms_pure_half_half2_branchless > 0 && ms_pure_half_fasthalf > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_branchless vs fp32_16_pure_half_wide_ilp_shared_fasthalf: %.3fx\n",
            (flops / ms_pure_half_half2_branchless) / (flops / ms_pure_half_fasthalf));
    if (ms_pure_half_half2_vecrow > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_vecrow vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2_vecrow) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_half2_vecrow > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_vecrow vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_half2_vecrow) / (flops / ms_pure_half_half2));
    if (ms_pure_half_half2_vecrow > 0 && ms_pure_half_fasthalf > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_vecrow vs fp32_16_pure_half_wide_ilp_shared_fasthalf: %.3fx\n",
            (flops / ms_pure_half_half2_vecrow) / (flops / ms_pure_half_fasthalf));
    if (ms_pure_half_half2_vec2rows > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_vec2rows vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2_vec2rows) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_half2_vec2rows > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_vec2rows vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_half2_vec2rows) / (flops / ms_pure_half_half2));
    if (ms_pure_half_half2_vec2rows > 0 && ms_pure_half_fasthalf > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_vec2rows vs fp32_16_pure_half_wide_ilp_shared_fasthalf: %.3fx\n",
            (flops / ms_pure_half_half2_vec2rows) / (flops / ms_pure_half_fasthalf));
    if (ms_pure_half_half2_wide2x > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_wide2x vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2_wide2x) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_half2_wide2x > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_wide2x vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_half2_wide2x) / (flops / ms_pure_half_half2));
    if (ms_pure_half_half2_wide4x > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_wide4x vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_half2_wide4x) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_half2_wide4x > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_wide4x vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_half2_wide4x) / (flops / ms_pure_half_half2));
    if (ms_bigtile_half2 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_bigtile_half2) / (flops / ms_cublas_hgemm));
    if (ms_bigtile_half2 > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2 vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_bigtile_half2) / (flops / ms_pure_half_half2));
    if (ms_bigtile_half2_adaptive > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive vs cublas_hgemm: %.3fx\n",
                   (flops / ms_bigtile_half2_adaptive) / (flops / ms_cublas_hgemm));
    if (ms_bigtile_half2_adaptive > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_bigtile_half2_adaptive) / (flops / ms_pure_half_half2));
    if (ms_bigtile_half2_adaptive > 0 && ms_bigtile_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive vs fp32_16_pure_half_wide_ilp_shared_bigtile_half2: %.3fx\n",
            (flops / ms_bigtile_half2_adaptive) / (flops / ms_bigtile_half2));
    if (ms_bigtile_half2_adaptive > 0 && ms_pure_half_half2_wide4x > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive vs fp32_16_pure_half_wide_ilp_shared_half2_wide4x: %.3fx\n",
            (flops / ms_bigtile_half2_adaptive) / (flops / ms_pure_half_half2_wide4x));
    if (ms_half2_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_best) / (flops / ms_cublas_hgemm));
    if (ms_half2_best > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_best vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_half2_best) / (flops / ms_pure_half_half2));
    if (ms_half2_atomic > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_half2_atomic) / (flops / ms_pure_half_half2));
    if (ms_half2_atomic > 0 && ms_half2_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic vs fp32_16_pure_half_wide_ilp_shared_half2_best: %.3fx\n",
            (flops / ms_half2_atomic) / (flops / ms_half2_best));
    if (ms_bigtile_half2_atomic > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic vs cublas_hgemm: %.3fx\n",
                   (flops / ms_bigtile_half2_atomic) / (flops / ms_cublas_hgemm));
    if (ms_bigtile_half2_atomic > 0 && ms_pure_half_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_bigtile_half2_atomic) / (flops / ms_pure_half_half2));
    if (ms_bigtile_half2_atomic > 0 && ms_half2_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic vs fp32_16_pure_half_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_bigtile_half2_atomic) / (flops / ms_half2_atomic));
    if (ms_half2_atomic_wide2x > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_wide2x) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_wide2x > 0 && ms_half2_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x vs fp32_16_pure_half_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_half2_atomic_wide2x) / (flops / ms_half2_atomic));
    if (ms_half2_atomic_wide4x > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_wide4x) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_wide4x > 0 && ms_half2_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x vs fp32_16_pure_half_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_half2_atomic_wide4x) / (flops / ms_half2_atomic));
    if (ms_half2_atomic_wide4x > 0 && ms_half2_atomic_wide2x > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x: %.3fx\n",
            (flops / ms_half2_atomic_wide4x) / (flops / ms_half2_atomic_wide2x));
    if (ms_half2_atomic_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_best) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_best > 0 && ms_half2_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_best vs fp32_16_pure_half_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_half2_atomic_best) / (flops / ms_half2_atomic));
    if (ms_hybrid > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid) / (flops / ms_cublas_hgemm));
    if (ms_hybrid > 0 && ms_pure_half_half2 > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
                   (flops / ms_hybrid) / (flops / ms_pure_half_half2));
    if (ms_hybrid_atomic > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_atomic) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_atomic > 0 && ms_hybrid > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic vs fp32_16_hybrid_wide_ilp_shared: %.3fx\n",
            (flops / ms_hybrid_atomic) / (flops / ms_hybrid));
    if (ms_hybrid_bigtile_atomic > 0 && ms_hybrid_atomic > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic vs fp32_16_hybrid_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_hybrid_bigtile_atomic) / (flops / ms_hybrid_atomic));
    if (ms_hybrid_atomic_wide4x > 0 && ms_hybrid_atomic > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x vs fp32_16_hybrid_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_hybrid_atomic_wide4x) / (flops / ms_hybrid_atomic));
    if (ms_hybrid_atomic_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_atomic_best) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_atomic_best > 0 && ms_hybrid_atomic > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_hybrid_atomic_best) / (flops / ms_hybrid_atomic));
    if (ms_hybrid_atomic_branchless > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_atomic_branchless) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_atomic_branchless > 0 && ms_hybrid_atomic > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless vs fp32_16_hybrid_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_hybrid_atomic_branchless) / (flops / ms_hybrid_atomic));
    if (ms_hybrid_branchless_bigtile > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_bigtile) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_bigtile > 0 && ms_hybrid_atomic_branchless > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless: %.3fx\n",
            (flops / ms_hybrid_branchless_bigtile) / (flops / ms_hybrid_atomic_branchless));
    if (ms_hybrid_branchless_wide4x > 0 && ms_hybrid_atomic_branchless > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless: %.3fx\n",
            (flops / ms_hybrid_branchless_wide4x) / (flops / ms_hybrid_atomic_branchless));
    if (ms_hybrid_branchless_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_best) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_best > 0 && ms_hybrid_atomic_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_hybrid_branchless_best) / (flops / ms_hybrid_atomic_best));
    if (ms_hybrid_branchless_fastx > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_fastx) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_fastx > 0 && ms_hybrid_atomic_branchless > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless: %.3fx\n",
            (flops / ms_hybrid_branchless_fastx) / (flops / ms_hybrid_atomic_branchless));
    if (ms_hybrid_branchless_fastx_bigtile > 0 && ms_hybrid_branchless_fastx > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx: %.3fx\n",
            (flops / ms_hybrid_branchless_fastx_bigtile) / (flops / ms_hybrid_branchless_fastx));
    if (ms_hybrid_branchless_fastx_wide4x > 0 && ms_hybrid_branchless_fastx > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx: %.3fx\n",
            (flops / ms_hybrid_branchless_fastx_wide4x) / (flops / ms_hybrid_branchless_fastx));
    if (ms_hybrid_branchless_fastx_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_fastx_best) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_fastx_best > 0 && ms_hybrid_branchless_fastx > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx: %.3fx\n",
            (flops / ms_hybrid_branchless_fastx_best) / (flops / ms_hybrid_branchless_fastx));
    if (ms_hybrid_branchless_fastx_best > 0 && ms_hybrid_atomic_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_hybrid_branchless_fastx_best) / (flops / ms_hybrid_atomic_best));
    if (ms_hybrid_branchless_intmask > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_intmask) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_intmask > 0 && ms_hybrid_atomic_branchless > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask) / (flops / ms_hybrid_atomic_branchless));
    if (ms_hybrid_branchless_intmask_bigtile > 0 && ms_hybrid_branchless_intmask > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_bigtile) / (flops / ms_hybrid_branchless_intmask));
    if (ms_hybrid_branchless_intmask_wide4x > 0 && ms_hybrid_branchless_intmask > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_wide4x) / (flops / ms_hybrid_branchless_intmask));
    if (ms_hybrid_branchless_intmask_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_intmask_best) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_intmask_best > 0 && ms_hybrid_atomic_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_best) / (flops / ms_hybrid_atomic_best));
    if (ms_hybrid_branchless_intmask_best > 0 && ms_hybrid_branchless_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_best) / (flops / ms_hybrid_branchless_best));
    if (ms_hybrid_branchless_intmask_walign > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_intmask_walign) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_intmask_walign > 0 && ms_hybrid_branchless_intmask_wide4x > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_walign) / (flops / ms_hybrid_branchless_intmask_wide4x));
    if (ms_hybrid_branchless_intmask_walign_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_branchless_intmask_walign_best) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_branchless_intmask_walign_best > 0 && ms_hybrid_branchless_intmask_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_walign_best) / (flops / ms_hybrid_branchless_intmask_best));
    if (ms_hybrid_branchless_intmask_walign_best > 0 && ms_hybrid_branchless_best > 0)
        std::printf(
            "    -- fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best vs fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best: %.3fx\n",
            (flops / ms_hybrid_branchless_intmask_walign_best) / (flops / ms_hybrid_branchless_best));
    if (ms_pure_half_fasthalf > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_fasthalf vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_fasthalf) / (flops / ms_cublas_hgemm));
    if (ms_fasthalf_atomic > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf vs cublas_hgemm: %.3fx\n",
                   (flops / ms_fasthalf_atomic) / (flops / ms_cublas_hgemm));
    if (ms_fasthalf_atomic > 0 && ms_pure_half_fasthalf > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf vs fp32_16_pure_half_wide_ilp_shared_fasthalf: %.3fx\n",
            (flops / ms_fasthalf_atomic) / (flops / ms_pure_half_fasthalf));
    if (ms_fasthalf_bigtile_atomic > 0 && ms_fasthalf_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf: %.3fx\n",
            (flops / ms_fasthalf_bigtile_atomic) / (flops / ms_fasthalf_atomic));
    if (ms_fasthalf_atomic_wide4x > 0 && ms_fasthalf_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf: %.3fx\n",
            (flops / ms_fasthalf_atomic_wide4x) / (flops / ms_fasthalf_atomic));
    if (ms_fasthalf_atomic_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_fasthalf_atomic_best) / (flops / ms_cublas_hgemm));
    if (ms_fasthalf_atomic_best > 0 && ms_fasthalf_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf: %.3fx\n",
            (flops / ms_fasthalf_atomic_best) / (flops / ms_fasthalf_atomic));
    if (ms_fp32_bf16_pure_bf16 > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf("    -- fp32_bf16_pure_bf16_wide_ilp_shared vs cublas_gemmex_bf16: %.3fx\n",
                   (flops / ms_fp32_bf16_pure_bf16) / (flops / ms_cublas_gemmex_bf16));
    if (ms_bf16_atomic > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf("    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic vs cublas_gemmex_bf16: %.3fx\n",
                   (flops / ms_bf16_atomic) / (flops / ms_cublas_gemmex_bf16));
    if (ms_bf16_atomic > 0 && ms_fp32_bf16_pure_bf16 > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic vs fp32_bf16_pure_bf16_wide_ilp_shared: %.3fx\n",
            (flops / ms_bf16_atomic) / (flops / ms_fp32_bf16_pure_bf16));
    if (ms_bf16_bigtile_atomic > 0 && ms_bf16_atomic > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic: %.3fx\n",
            (flops / ms_bf16_bigtile_atomic) / (flops / ms_bf16_atomic));
    if (ms_bf16_atomic_wide4x > 0 && ms_bf16_atomic > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic: %.3fx\n",
            (flops / ms_bf16_atomic_wide4x) / (flops / ms_bf16_atomic));
    if (ms_bf16_atomic_best > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf("    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best vs cublas_gemmex_bf16: %.3fx\n",
                   (flops / ms_bf16_atomic_best) / (flops / ms_cublas_gemmex_bf16));
    if (ms_bf16_atomic_best > 0 && ms_bf16_atomic > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic: %.3fx\n",
            (flops / ms_bf16_atomic_best) / (flops / ms_bf16_atomic));
    if (ms_bf16_atomic_v2 > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf("    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 vs cublas_gemmex_bf16: %.3fx\n",
                   (flops / ms_bf16_atomic_v2) / (flops / ms_cublas_gemmex_bf16));
    if (ms_bf16_atomic_v2 > 0 && ms_bf16_atomic > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic: %.3fx\n",
            (flops / ms_bf16_atomic_v2) / (flops / ms_bf16_atomic));
    if (ms_bf16_bigtile_atomic_v2 > 0 && ms_bf16_atomic_v2 > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2 vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2: %.3fx\n",
            (flops / ms_bf16_bigtile_atomic_v2) / (flops / ms_bf16_atomic_v2));
    if (ms_bf16_atomic_wide4x_v2 > 0 && ms_bf16_atomic_v2 > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2 vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2: %.3fx\n",
            (flops / ms_bf16_atomic_wide4x_v2) / (flops / ms_bf16_atomic_v2));
    if (ms_bf16_atomic_best_v2 > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf("    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 vs cublas_gemmex_bf16: %.3fx\n",
                   (flops / ms_bf16_atomic_best_v2) / (flops / ms_cublas_gemmex_bf16));
    if (ms_bf16_atomic_best_v2 > 0 && ms_bf16_atomic_v2 > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2: %.3fx\n",
            (flops / ms_bf16_atomic_best_v2) / (flops / ms_bf16_atomic_v2));
    if (ms_bf16_atomic_best_v2 > 0 && ms_bf16_atomic_best > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 vs fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best: %.3fx\n",
            (flops / ms_bf16_atomic_best_v2) / (flops / ms_bf16_atomic_best));
    if (ms_cublas_gemmex_bf16 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- cublas_gemmex_bf16 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_cublas_gemmex_bf16) / (flops / ms_cublas_hgemm));
}

}  // namespace

int main(int argc, char** argv) {
    // Extract --warmup N / --iters N from anywhere in argv (both
    // optional, default 10/10, see g_warmup_runs/g_iterations' own
    // comment above time_call) before this file's own original argument
    // parsing below runs -- argc/argv are reassigned to a filtered copy
    // with those two flags (and their values) removed, so every shape/
    // flag form the sibling file's own main() accepts still works
    // unchanged here.
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) {
            g_warmup_runs = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            g_iterations = std::atoi(argv[i + 1]);
            ++i;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== [advanced: warmup=%d iters=%d] ===\n", g_warmup_runs, g_iterations);
    std::printf("=== NO_TRANSPOSE GEMV: TOMOS FP32/FP16 (pure 16-bit compute) vs cuBLAS Hgemm(n=1) ===\n");

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
                                              16384, 24576, 32768, 49152, 65536, 131072};
    const std::vector<size_t> tall_ms = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    const std::vector<size_t> tall_ks = {512, 4096};

    if (argc > 1 && std::string(argv[1]) == "--tall") {
        size_t k = (argc > 2) ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : tall_ks[0];
        if (argc > 3) {
            size_t m = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));
            run_shape(m, k, start, stop, handle);
        } else {
            for (size_t m : tall_ms) run_shape(m, k, start, stop, handle);
        }
        cublasDestroy(handle);
        return 0;
    }

    if (argc > 1) {
        size_t m = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
        run_shape(m, m, start, stop, handle);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t m : square_sizes) run_shape(m, m, start, stop, handle);
    for (size_t k : tall_ks)
        for (size_t m : tall_ms) run_shape(m, k, start, stop, handle);

    cublasDestroy(handle);
    return 0;
}

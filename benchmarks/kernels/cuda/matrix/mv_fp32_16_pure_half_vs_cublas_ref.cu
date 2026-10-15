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
// Generic mixed-precision GEMV fallback (src/kernels/cuda/matrix/
// mv_cuda_kernels.cu/.cuh) -- NOT part of the fp32_16 packed-storage
// family above/below: A here is plain native __half (no wh/wr split at
// all), and this is the SAME kernel the generic Dense::lookup_mv
// dispatch registry silently falls onto whenever a caller requests an
// (A_type, x_type, y_type) triple with no dedicated tuned kernel -- e.g.
// lsqr_precision_runner's own --mv-type fp16 combined with --value-type
// fp32 and no --implicit-duplicates. Included here as "FP16 (double
// acc)" purely so that silent-fallback behavior is visible on the SAME
// performance plot as the genuinely-tuned "TOMOS FP16" series, instead
// of only being discoverable by reading dispatch code.
namespace experimental {
template <typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, dim<2> size, a_t* A,
                size_t ld_a, x_t* x, size_t ld_x, double beta, y_t* y,
                size_t ld_y, cublasHandle_t handle);
}  // namespace experimental
// Fully native fp16 GEMV -- A, x, AND y all __half, no decode at all (see
// its own comment in mv_cuda_kernels.cu). Dispatched here as "TOMOS FP16"
// -- the existing FP32_16-packed-storage series below that used to own
// that label is now "TOMOS FP16 (implicit)".
void mv_fp16_native_storage(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                            const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y);
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

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_split(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

// NEW, additive-only SWAR-decode sibling of the dispatcher immediately
// above -- see its own comment in fp32_16_cuda_kernels.cu. EXPERIMENTAL,
// not yet independently correctness-checked against this kernel's own
// packing (see that comment); throughput-only leg below, "test-latest"
// in the plot.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_combine_best(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const __half* A,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp16_native_mixed(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const __half* A,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_split_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_split_workspace(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y,
                                                        float* workspace); }  // namespace experimental

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

// Packed-storage implicit kernel with mv_fp16_native_mixed's accumulation
// (256-column tiles flushed into a double) -- see fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_nativeaccum(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                      size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                      size_t ld_y);
// LSQR's actual implicit NO_TRANSPOSE dispatcher (mv_pure_half_impl calls it)
// and its periodic-flush drop-ins -- see fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_no_transpose_fast2(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                             const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_pure_half_no_transpose_flush64(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                               const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_pure_half_no_transpose_flush256(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                                const float* x, size_t ld_x, double beta, float* y,
                                                size_t ld_y);
// LOSSY fast-decode + flush256 + SWAR sibling (underflow band -> 0).
void mv_fp32_16_pure_half_no_transpose_flush256_fast_swar(double alpha, dim<2> size, const uint16_t* wh,
                                                          size_t ld_a, const float* x, size_t ld_x, double beta,
                                                          float* y, size_t ld_y);
// Same kernel as flush256 with the SWAR (two wh per 32-bit register) decode.
void mv_fp32_16_pure_half_no_transpose_flush256_swar(double alpha, dim<2> size, const uint16_t* wh,
                                                     size_t ld_a, const float* x, size_t ld_x, double beta,
                                                     float* y, size_t ld_y);
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                             double alpha, dim<2> size,
                                                             const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta,
                                                             float* y, size_t ld_y);
void mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y);
void mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                   double alpha, dim<2> size,
                                                                   const uint16_t* wh,
                                                                   size_t ld_a, const float* x,
                                                                   size_t ld_x, double beta,
                                                                   float* y, size_t ld_y);
// Genuinely branchless-decode sibling: mirrors fast2_best's own shape-
// dispatch tree exactly, only swapping fp32_16_half_only_fast2's single
// data-dependent branch for fp32_16_half_only_branchless's unconditional-
// compute-then-blend decode (see that function's own comment in
// fp32_16_pack.hpp). MEASURED on the real matrix's own realistic-
// underflow data: a real but modest win over fast2 (removing warp
// divergence only saves the branch overhead itself, not the underlying
// decode arithmetic, which stays unconditional either way) -- superseded
// as "test-new" below by the no-overflow-guaranteed sibling, which cuts
// the arithmetic itself. Kept declared/callable for that comparison.
void mv_fp32_16_pure_half_branchless_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                        double alpha, dim<2> size,
                                                                        const uint16_t* wh,
                                                                        size_t ld_a, const float* x,
                                                                        size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);
// No-overflow-guaranteed sibling of the branchless dispatcher above --
// drops the direction extraction and cheap_bits select entirely under a
// caller-guaranteed "no overflow in this buffer" contract, MEASURED to
// genuinely hold on the real LSQR test matrix (0% overflow). See
// fp32_16_half_only_branchless_no_overflow's own comment in
// fp32_16_pack.hpp. Superseded as "test-new" below by the vec4store
// sibling, which ALSO fixes the write-back pattern ncu flagged
// separately from decode cost. Kept declared/callable for comparison.
void mv_fp32_16_pure_half_branchless_no_overflow_wide_ilp_shared_half2_atomic_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y);
// Vectorized-write sibling of the no_overflow narrow_vec kernel -- ncu
// flagged that kernel's own 4-separate-scalar-stores write-back
// directly (uncoalesced, scoreboard-stall-heavy), independent of decode
// cost. MEASURED: no real improvement over the non-vectorized write at
// the real LSQR shape (the flagged stall was apparently already hidden
// by other warps' work). Superseded as "test-new" below by the
// grid-stride sibling. Kept declared/callable for comparison.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store(TransposeType trans, double alpha, size_t m,
                                                                       size_t n, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, double beta, float* y);
// Grid-stride sibling of vec4store -- ncu also flagged this kernel
// family's own "4 rows per thread" grid sizing (977 blocks, only 864
// fit in one wave on this A100) as a partial-wave-tail problem (Waves
// Per SM: 1.13, Est. Speedup 50%). MEASURED: no real improvement either
// (the flagged tail wasn't on the critical path). Superseded as
// "test-new" below by the shiftenc sibling. Kept declared/callable.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);
// Shift-numbering sibling -- decode reads the 4-bit field as shift_by
// directly (offset-1, precomputed by a separate renumbering pass
// below), removing the subtraction + offset==0 zero-guard from the
// decode's own critical path. Lossless relabeling of the shift-amount
// field only; mant_hi (what the accurate wh+wr tier depends on) is
// untouched. MEASURED: real, substantial win (~14% over the previous
// best) -- shortening the dependency chain mattered more than the raw
// instruction count suggested. Superseded as "test-new" below by the
// half2acc sibling. Kept declared/callable for comparison.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_shiftenc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);
// half2-accumulation sibling of shiftenc -- decode stays scalar, but the
// accumulate step packs a0/a1 and a2/a3 into __half2 and uses __hfma2
// against a broadcast x, halving the FMA instruction count for that
// part of the loop (same technique mv_fp32_16_pure_half_wide_ilp_
// shared_half2_atomic_kernel already uses elsewhere). MEASURED: another
// real, substantial win (~90% of native_mixed at the real LSQR shape).
// Catastrophically under-parallelized at small M if called directly
// (narrow_vec's own 4-rows-per-thread grid shrinks to ~98 blocks on a
// 108-SM GPU at M=100,000) -- superseded as "test-new" below by the
// shape-aware dispatcher, which only routes here for large M. Kept
// declared/callable for comparison.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_shiftenc_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);
// "test-new": shape-aware composed dispatcher -- small M (<
// kMvWideRowThreshold) routes to cpasync_warpuniform (already fast
// there, unchanged), large M (the real LSQR shape) routes to the
// half2acc kernel above.
void mv_fp32_16_pure_half_test_new_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                             const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
// "test-new3": same dispatcher as test-new, but the large-M branch uses
// a kMvRowsPerThread=2 sibling kernel instead of 4 -- testing register-
// blocking vs. occupancy at this shape (see fp32_16_cuda_kernels.cu's
// own comment on the new kernel for the full rationale).
void mv_fp32_16_pure_half_test_new3_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
// "test-new4": same dispatcher, large-M branch uses a kMvRowsPerThread=8
// sibling kernel -- extends the register-blocking-vs-occupancy trend in
// the opposite direction from test-new3.
void mv_fp32_16_pure_half_test_new4_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed(TransposeType trans,
                                                              double alpha, dim<2> size,
                                                              const uint16_t* wh,
                                                              const uint16_t* wr,
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

// Underflow-rate-matched generator: ~18% of values fall in fp16's
// underflow band [1e-8, 6.104e-5) -- the shift-based subnormal-
// reconstruction path fp32_16_half_only_fast2 takes when is_underflow is
// true -- the rest stay plain uniform [-1,1] (control=0, the cheap
// path), same convention every other gen_fp32_16_wh_* generator here
// uses. Built to test a real, measured finding: sampling the actual
// LSQR .mtx test matrix (tall_thin_1000.mtx -- see
// documentation/exponent-range.md's own "Real-world case study" section
// for how that matrix was generated, via scripts/generate_tall_thin_mtx.sh)
// directly found a ~18% underflow rate (two independent
// 2M-value samples: 18.79% and 17.59%), vs. plain gen_fp32_16_wh's own
// pseudo_rand data (uniform [-1,1]), which essentially never triggers
// underflow at all (~0%). fp32_16_half_only_fast2's own is_underflow
// branch is genuinely data-dependent (see its comment in fp32_16_
// pack.hpp) -- if a real matrix's much higher underflow rate causes more
// warp divergence than this benchmark's own synthetic data ever
// exercises, that would explain why an isolated GEVM benchmark run
// against gen_fp32_16_wh's data can show fast2_best performing well
// while the SAME kernel, called on the SAME real matrix inside an actual
// LSQR solve, was measured (via ncu) running ~1.5x slower than that
// isolated number despite identical clocks/occupancy/register pressure/
// cache hit rate -- everything EXCEPT the data itself was already ruled
// out. This generator isolates that one remaining variable.
__device__ float pseudo_rand_realistic_underflow(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    if ((s % 100u) < 18u) {
        uint32_t s2 = s * 2246822519u + 3266489917u;
        s2 ^= s2 << 13;
        s2 ^= s2 >> 17;
        s2 ^= s2 << 5;
        // [1e-8, 6.104e-5) -- below fp16's normal-range floor (~6.104e-5,
        // 2^-14), above fp32_16_half_only_fast2's own hard-zero floor for
        // offset in [1,10] (real fp16 subnormals reach down to ~5.96e-8).
        float mag = 1.0e-8f + (static_cast<float>(s2 % 100001u) / 100000.0f) * (6.104e-5f - 1.0e-8f);
        return (s & 0x10000u) ? mag : -mag;
    }
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_fp32_16_wh_realistic_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_realistic_underflow(i), wh[i], wr[i]);
}
void gen_fp32_16_wh_realistic(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_realistic_kernel<<<blocks, threads>>>(n, wh, wr);
}

// NO LONGER NEEDED by "test-new" (fp32_16_split itself now writes the
// shift_by numbering directly -- see fp32_16_pack.hpp's own
// kOffsetSaturateStored comment for the full history: this renumbering
// pass is what it was verified against before being promoted globally).
// Applying this to data that's ALREADY shift_by-numbered would silently
// subtract 1 a second time and corrupt it -- kept defined (unused) only
// as a record of that verification step, not for any live caller.
__global__ void renumber_offset_to_shift_kernel(size_t n, const uint16_t* src, uint16_t* dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t wh = src[i];
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    if (!control) {
        dst[i] = wh;
        return;
    }
    uint32_t offset = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t shift_by = offset - 1u;
    uint32_t cleared = static_cast<uint32_t>(wh) & ~(0xFu << 10);
    dst[i] = static_cast<uint16_t>(cleared | (shift_by << 10));
}
void renumber_offset_to_shift(size_t n, const uint16_t* src, uint16_t* dst) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    renumber_offset_to_shift_kernel<<<blocks, threads>>>(n, src, dst);
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

// 10 warmup calls (not just 1) before the timed region: a single warmup
// call isn't enough for the GPU's clock boost governor to reach sustained
// full boost for some kernels, which was found this session to make
// reps=5 (the old formula's large-M value) measure a partially-ramped-up
// state -- stable and reproducible run to run, but consistently lower
// than the kernel's own true steady-state throughput (confirmed via
// repeated in-process re-measurement of the same kernel/buffers: the
// first few calls read high/inflated, then settle to a true, lower,
// stable value matching independent isolated-harness and ncu ground-
// truth measurements once enough iterations have run). 10 timed reps
// (not the old shape-dependent 5/20/50 formula) for the same reason,
// applied uniformly across every shape.
double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
    for (int i = 0; i < 10; ++i) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / reps;
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
    // Fixed at 10 for every shape now (was 5/20/50 depending on maxdim) --
    // see time_call's own comment: reps=5 at the large-M end was found
    // this session to measure a partially clock-ramped-up state, not
    // true steady-state throughput.
    int reps = 10;
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k);

    std::printf("  %s\n", shape_desc);

    double ms_pure_half_half2 = -1.0, ms_hybrid = -1.0, ms_pure_half_fasthalf = -1.0, ms_fp32_bf16_pure_bf16 = -1.0,
           ms_cublas_hgemm = -1.0, ms_cublas_gemmex_bf16 = -1.0, ms_pure_half_branchless = -1.0,
           ms_pure_half_half2_branchless = -1.0, ms_pure_half_half2_vecrow = -1.0,
           ms_pure_half_half2_vec2rows = -1.0, ms_pure_half_half2_wide2x = -1.0,
           ms_pure_half_half2_wide4x = -1.0, ms_bigtile_half2 = -1.0,
           ms_bigtile_half2_adaptive = -1.0, ms_half2_best = -1.0, ms_half2_atomic = -1.0,
           ms_half2_atomic_lowsplit = -1.0, ms_half2_split = -1.0, ms_half2_atomic_prefetch = -1.0,
           ms_half2_atomic_prefetch2 = -1.0, ms_half2_atomic_cpasync = -1.0, ms_half2_combine_best = -1.0,
           ms_native_mixed = -1.0,
           ms_half2_atomic_cpasync_warpuniform = -1.0,
           ms_half2_atomic_cpasync_warpuniform_swar = -1.0,
           ms_bigtile_half2_atomic = -1.0, ms_half2_atomic_wide2x = -1.0,
           ms_half2_atomic_wide4x = -1.0, ms_half2_atomic_best = -1.0,
           ms_nativeaccum = -1.0,
           ms_noT_fast2 = -1.0, ms_flush64 = -1.0, ms_flush256 = -1.0, ms_flush256_swar = -1.0, ms_flush256_fast_swar = -1.0,
           ms_half2_atomic_best_fast = -1.0, ms_half2_atomic_best_fast2 = -1.0,
           ms_half2_atomic_best_fast2_realistic_underflow = -1.0,
           ms_half2_atomic_cpasync_warpuniform_realistic_underflow = -1.0,
           ms_test_new = -1.0, ms_test_new2 = -1.0, ms_test_new3 = -1.0, ms_test_new4 = -1.0,
           ms_double_acc = -1.0, ms_native_storage = -1.0,
           ms_half2_atomic_mixed = -1.0,
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

            // "test (half2, atomic, lowsplit)" -- col_splits-retuned
            // sibling of wide4x (kMvTargetTotalThreads*1 instead of *4,
            // targeting the atomic-combine kernel's own contention cost
            // model, not the serial-reduce cost model choose_col_splits_
            // target was tuned around -- see that kernel's own comment in
            // fp32_16_cuda_kernels.cu). Same wh, same decode, only
            // col_splits differs from every other TOMOS FP16 series here.
            ms_half2_atomic_lowsplit = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_lowsplit, flops / ms_half2_atomic_lowsplit / 1e6);

            // "test (half2, split, no atomic)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_half2_split's own NO_TRANSPOSE path: SAME
            // col_splits as wide4x/"_best" (kMvTargetTotalThreads*4, NOT
            // lowsplit's *1 above), SAME decode, SAME per-block compute --
            // only the combine step differs, partial-buffer + separate
            // reduce kernel instead of atomicAdd-into-y directly, avoiding
            // the atomic-contention this kernel family's own Nsight
            // Compute profiling flagged as the likely bottleneck at
            // col_splits=16 (85% warp occupancy but only 33% DRAM / 20% SM
            // throughput -- time spent waiting on a contended atomic, not
            // doing useful work). A clean single-variable comparison
            // against "TOMOS FP16 (implicit)" above.
            //
            // Uses the _workspace sibling (caller-owned partial buffer,
            // allocated once here, outside time_call's timed reps loop)
            // instead of the plain _split wrapper's own internal
            // cudaMalloc/cudaFree-per-call -- the first "worse" measurement
            // of this candidate was confounded by paying that allocation
            // cost every single timed rep, unlike every atomic-combine
            // kernel it's compared against (zero allocation). Isolates the
            // atomic-vs-reduce-kernel question from the allocation-overhead
            // question.
            size_t half2_split_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_split_workspace_size(
                    TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            float* half2_split_ws = nullptr;
            if (half2_split_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&half2_split_ws),
                             half2_split_ws_elems * sizeof(float),
                             "pure_half_wide_ilp_shared_half2_split (workspace)", shape_desc)) {
                ms_half2_split = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_split_workspace(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        half2_split_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_split = %9.3fms (%10.1f GFlop/s)\n",
                           ms_half2_split, flops / ms_half2_split / 1e6);
                if (half2_split_ws) cudaFree(half2_split_ws);
            }

            // "test (half2, atomic, prefetch)" -- mv_fp32_16_pure_half_wide_
            // ilp_shared_half2_atomic_prefetch's own NO_TRANSPOSE path: SAME
            // col_splits/grid/decode/combine as wide4x/"_best" -- only the
            // underlying kernel double-buffers its wh loads (issues the
            // NEXT 4-wide batch before consuming the CURRENT one) instead
            // of load-then-immediately-use. Nsight Compute profiling of the
            // non-pipelined kernel found 88.7% of stall cycles are L1TEX
            // scoreboard stalls despite DRAM being far from saturated (Mem
            // Busy ~12%, Max Bandwidth ~25%) -- latency-bound, not
            // bandwidth- or atomic-contention-bound (ruled out separately
            // by "test (half2, split, no atomic)" above, which removed the
            // atomic entirely and measured no faster). This candidate tests
            // whether hiding that load latency behind independent FMA work
            // closes the gap to cuBLAS. Same wh, same decode.
            ms_half2_atomic_prefetch = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_prefetch, flops / ms_half2_atomic_prefetch / 1e6);

            // "test (half2, atomic, prefetch2)" -- depth-2 sibling of
            // "test (half2, atomic, prefetch)" above: re-profiling the
            // depth-1 prefetch kernel found the L1TEX-scoreboard-stall
            // share barely moved (88.7% -> 85.2%), consistent with ptxas
            // already scheduling that single independent load about as
            // well on its own. This candidate keeps 2 batches (8 elements)
            // of wh loads in flight per thread instead of 1, directly
            // doubling the number of independent outstanding memory
            // requests rather than relying on the compiler to find that
            // overlap from a 1-deep hint. SAME col_splits/grid/decode/
            // combine as wide4x/"_best" otherwise. Same wh.
            ms_half2_atomic_prefetch2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_prefetch2, flops / ms_half2_atomic_prefetch2 / 1e6);

            // "test (half2, atomic, cpasync)" -- register-mediated
            // lookahead (prefetch/prefetch2 above) measured no faster and
            // barely moved the L1TEX-stall share Nsight Compute found
            // dominant. This candidate uses Ampere's cp.async instead --
            // a fundamentally different mechanism: the async-copy engine
            // moves wh bytes from global straight into shared memory,
            // never occupying a register for the pending value and never
            // blocking the issuing warp until an explicit wait, which
            // register-mediated loads (ordinary or reordered) cannot do.
            // Reuses mv_fp32_16_pure_half_wide_ilp_shared_cpasync2_kernel's
            // own proven double-buffered staging mechanism (already in
            // this file from an earlier investigation), adapted to match
            // wide4x/"_best" on every other axis (same col_splits formula,
            // __hfma2 accumulation, atomicAdd combine) for a clean single-
            // variable comparison. CAVEAT: this kernel uses 55 registers
            // and ~32.5KB shared memory per block (vs ~512 bytes for every
            // prior candidate here) -- on an A100 that shared-memory
            // footprint alone limits occupancy to roughly 1 block/SM, far
            // below every other candidate's occupancy. If this measures no
            // faster (or worse), that occupancy cost is the prime suspect,
            // independent of whether cp.async itself helps.
            ms_half2_atomic_cpasync = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_cpasync, flops / ms_half2_atomic_cpasync / 1e6);

            // "test2" -- warp-uniform-issuer sibling of the cp.async
            // candidate above. Re-profiling that kernel (Warp State
            // Statistics + Occupancy) found its own biggest flagged
            // opportunity (Est. Local Speedup 25.51%) is warp divergence
            // during staging, not the memory latency cp.async already
            // fixed (warp cycles/instruction dropped 59.5->11.8 there):
            // only 24.67 of 32 threads active per warp on average, from
            // that kernel's own issuer = (threadIdx.x % kMvCpAsyncRowChunk
            // == 0) alternating even/odd WITHIN every 32-thread warp (16
            // issue, 16 idle, every warp, for the whole staging loop) --
            // real intra-warp predication cost. This candidate changes
            // ONLY which 128 of 256 threads issue (same total transfer
            // count, same shared-memory layout the compute phase reads
            // from, byte-for-byte unchanged): issuer = (threadIdx.x <
            // kMvTileSize/2) -- the first 4 warps, warp-uniform (every
            // thread in warps 0-3 takes the same branch, every thread in
            // warps 4-7 takes the other), eliminating intra-warp
            // divergence during staging entirely. Same col_splits/decode/
            // combine/occupancy footprint as cpasync above (56 vs 55
            // registers, same 33280 bytes shared mem) -- isolates the
            // issuer-selection scheme as the only variable.
            ms_half2_atomic_cpasync_warpuniform = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_cpasync_warpuniform, flops / ms_half2_atomic_cpasync_warpuniform / 1e6);

            // NEW, additive-only: SWAR-decode sibling of the leg just
            // above, same wh/x/y buffers -- see fp32_16_cuda_kernels.cu's
            // own comment on mv_fp32_16_pure_half_wide_ilp_shared_half2_
            // atomic_cpasync_warpuniform_swar for the full derivation.
            // "test-latest" in the plot.
            ms_half2_atomic_cpasync_warpuniform_swar = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_cpasync_warpuniform_swar, flops / ms_half2_atomic_cpasync_warpuniform_swar / 1e6);

            // SAME kernel as the leg directly above (mv_fp32_16_pure_half_
            // wide_ilp_shared_half2_atomic_cpasync_warpuniform -- the SAFE-
            // decode kernel mv_pure_half_impl's own default, non-fast2
            // NO_TRANSPOSE branch would reach for small M, and "TOMOS
            // FP32/FP16" in this plot), only the wh DATA differs:
            // gen_fp32_16_wh_realistic's own ~18% underflow rate (matched
            // to the real LSQR test matrix's own measured rate, see that
            // generator's comment above) instead of plain gen_fp32_16_wh's
            // ~0% rate. fp32_16_half_only (the SAFE decode this kernel
            // uses) has TWO sequential data-dependent branches (control,
            // then direction) -- more than fast2_realistic_underflow's one
            // combined check above -- so this is checking whether "TOMOS
            // FP32/FP16"'s own good-looking (~0%-underflow) numbers are
            // similarly not representative of real matrix data, per the
            // same blind spot fast2's own realistic-underflow leg just
            // confirmed.
            uint16_t *wh_safe_realistic = nullptr, *wr_safe_realistic_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_safe_realistic), m * k * sizeof(uint16_t),
                              "cpasync_warpuniform realistic-underflow (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_safe_realistic_scratch), m * k * sizeof(uint16_t),
                              "cpasync_warpuniform realistic-underflow (wr scratch)", shape_desc)) {
                gen_fp32_16_wh_realistic(m * k, wh_safe_realistic, wr_safe_realistic_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_safe_realistic_scratch);

                ms_half2_atomic_cpasync_warpuniform_realistic_underflow = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_safe_realistic, static_cast<size_t>(lda),
                        x, 1, 0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform (realistic ~18%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                    ms_half2_atomic_cpasync_warpuniform_realistic_underflow,
                    flops / ms_half2_atomic_cpasync_warpuniform_realistic_underflow / 1e6);
                if (ms_half2_atomic_cpasync_warpuniform_realistic_underflow > 0 && ms_half2_atomic_cpasync_warpuniform > 0)
                    std::printf(
                        "    -- cpasync_warpuniform (realistic underflow) vs (uniform, ~0%% underflow): %.3fx\n",
                        (flops / ms_half2_atomic_cpasync_warpuniform_realistic_underflow) /
                            (flops / ms_half2_atomic_cpasync_warpuniform));

                cudaFree(wh_safe_realistic);
            }

            // "test (combine strategy dispatch)" -- mirrors mv_fp16_native_
            // storage's own "_best"-style combine-strategy dispatch,
            // measured on real A100 hardware to fix a severe small/medium-M
            // atomicAdd-contention problem for that (no-decode, native
            // storage) kernel family: reduce-kernel combine (mv_fp32_16_
            // pure_half_wide_ilp_shared_half2, this investigation's
            // historically-fastest safe-decode series) below M=8192,
            // atomicAdd combine ("TOMOS FP16 (implicit)" itself) at/above
            // it. Tests whether the same mechanism (choose_col_splits_*
            // targeting a FIXED total thread count regardless of row
            // count, so col_splits -- and atomicAdd contention -- balloons
            // at small M) also afflicts this safe-decode packed-storage
            // family, which uses a different col_splits formula than
            // native storage's own. Crossover reused as-is from native
            // storage's own measurement, not independently re-tuned here.
            ms_half2_combine_best = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2_combine_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2_combine_best = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_combine_best, flops / ms_half2_combine_best / 1e6);

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

            // "TOMOS FP16 (implicit, native accumulation)" -- packed wh-only
            // storage and the same branchless no-overflow decode as the
            // implicit legs, but with mv_fp16_native_mixed's accumulation
            // (fp16 chains capped at 256 columns, flushed into a double per
            // tile -- the structure LSQR's --explicit-duplicates kernel
            // uses) for accuracy; timed here to see how much throughput that
            // costs relative to the leg above and to the native_mixed leg
            // ("native mixed LSQR" in the plot). Same wh, no workspace.
            ms_nativeaccum = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_nativeaccum(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_nativeaccum = %9.3fms (%10.1f GFlop/s)\n",
                       ms_nativeaccum, flops / ms_nativeaccum / 1e6);

            // LSQR's ACTUAL implicit NO_TRANSPOSE dispatcher (below
            // kMvWideRowThreshold rows: cpasync_warpuniform; at/above it --
            // LSQR's M=1e6 -- the narrow_vec kernel), the like-for-like
            // baseline for the two periodic-flush drop-ins right below it
            // (same routing, only the large-M kernel gains a float flush
            // every 64 / 256 columns to bound each fp16 chain -- see that
            // kernel's own comment). Below the threshold all three legs run
            // the same kernel, so any difference shows up only at M >= 131072.
            // Same wh, no workspace.
            ms_noT_fast2 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_no_transpose_fast2(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_no_transpose_fast2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_noT_fast2, flops / ms_noT_fast2 / 1e6);
            ms_flush64 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_no_transpose_flush64(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_no_transpose_flush64 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_flush64, flops / ms_flush64 / 1e6);
            ms_flush256 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_no_transpose_flush256(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_no_transpose_flush256 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_flush256, flops / ms_flush256 / 1e6);
            ms_flush256_swar = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_no_transpose_flush256_swar(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_no_transpose_flush256_swar = %9.3fms (%10.1f GFlop/s)\n",
                       ms_flush256_swar, flops / ms_flush256_swar / 1e6);
            ms_flush256_fast_swar = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_no_transpose_flush256_fast_swar(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_no_transpose_flush256_fast_swar = %9.3fms (%10.1f GFlop/s)\n",
                       ms_flush256_fast_swar, flops / ms_flush256_fast_swar / 1e6);

            // Fast-decode sibling of the leg above -- same wh, only the
            // decode inside the kernel differs (fp32_16_half_only_fast
            // instead of fp32_16_half_only, see that function's own
            // comment in fp32_16_pack.hpp). Added as a new, separate call;
            // the leg above is untouched.
            ms_half2_atomic_best_fast = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_best_fast, flops / ms_half2_atomic_best_fast / 1e6);

            // Fast2-decode sibling -- same wh, single-branch "is this
            // genuine underflow" decode (fp32_16_half_only_fast2, see its
            // own comment in fp32_16_pack.hpp): real shift-based subnormal
            // reconstruction kept for underflow (unlike the fast leg
            // above, which saturates that whole band to zero), but
            // collapsed to one branch instead of fp32_16_half_only's two
            // (control, then direction) for every other element. Added as
            // a new, separate call; both legs above are untouched.
            ms_half2_atomic_best_fast2 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best = %9.3fms (%10.1f GFlop/s)\n",
                       ms_half2_atomic_best_fast2, flops / ms_half2_atomic_best_fast2 / 1e6);

            // SAME kernel as the leg directly above (mv_fp32_16_pure_half_
            // fast2_wide_ilp_shared_half2_atomic_best -- the ACTUAL kernel
            // mv_pure_half_impl's NO_TRANSPOSE safe-decode branch dispatches
            // to for this shape, fp32_16_registries.cpp), only the
            // underlying wh DATA differs: gen_fp32_16_wh_realistic's own
            // ~18% underflow rate (matched to the real LSQR test matrix's
            // own measured rate, see that generator's comment above)
            // instead of plain gen_fp32_16_wh's ~0% rate. Isolates whether
            // fp32_16_half_only_fast2's data-dependent is_underflow branch
            // (see fp32_16_pack.hpp) is what's causing this exact kernel to
            // run slower inside the real LSQR solve than this benchmark's
            // own (until now, always ~0%-underflow) numbers predicted.
            uint16_t *wh_realistic = nullptr, *wr_realistic_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_realistic), m * k * sizeof(uint16_t),
                              "fast2_wide_ilp_shared_half2_atomic_best realistic-underflow (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_realistic_scratch), m * k * sizeof(uint16_t),
                              "fast2_wide_ilp_shared_half2_atomic_best realistic-underflow (wr scratch)",
                              shape_desc)) {
                gen_fp32_16_wh_realistic(m * k, wh_realistic, wr_realistic_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_realistic_scratch);

                ms_half2_atomic_best_fast2_realistic_underflow = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_realistic, static_cast<size_t>(lda), x, 1,
                        0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best (realistic ~18%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                    ms_half2_atomic_best_fast2_realistic_underflow,
                    flops / ms_half2_atomic_best_fast2_realistic_underflow / 1e6);
                if (ms_half2_atomic_best_fast2_realistic_underflow > 0 && ms_half2_atomic_best_fast2 > 0)
                    std::printf(
                        "    -- fast2_wide_ilp_shared_half2_atomic_best (realistic underflow) vs (uniform, ~0%% underflow): %.3fx\n",
                        (flops / ms_half2_atomic_best_fast2_realistic_underflow) / (flops / ms_half2_atomic_best_fast2));

                // "test-new": shape-aware dispatcher -- small M routes to
                // cpasync_warpuniform, large M (this shape) routes to the
                // half2acc kernel. fp32_16_split now writes the shift_by
                // numbering directly (promoted globally), so wh_realistic
                // already holds it -- no separate renumbering pass needed.
                ms_test_new = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_pure_half_test_new_dispatch(
                        1.7, dim<2>{m, k}, wh_realistic, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_test_new_dispatch (test-new, realistic ~18%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                    ms_test_new, flops / ms_test_new / 1e6);
                if (ms_test_new > 0 && ms_half2_atomic_best_fast2_realistic_underflow > 0)
                    std::printf(
                        "    -- test-new (shiftenc) vs fast2_best (single-branch), both realistic underflow: %.3fx\n",
                        (flops / ms_test_new) / (flops / ms_half2_atomic_best_fast2_realistic_underflow));

                cudaFree(wh_realistic);
            }

            // "test-new2": SAME test-new dispatcher/kernel as above, but
            // run against the plain wh buffer (gen_fp32_16_wh's own
            // ~0%-underflow uniform data, not wh_realistic's ~18%) --
            // isolates exactly how much of test-new's own remaining gap
            // to native_mixed is attributable to real data's underflow
            // rate, for the CURRENT most-optimized kernel specifically
            // (earlier in this investigation this same before/after
            // comparison was made for fast2_best; this repeats it for
            // test-new's own dispatcher/decode now that both have moved
            // on considerably since then).
            ms_test_new2 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_test_new_dispatch(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_test_new_dispatch (test-new2, uniform ~0%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                ms_test_new2, flops / ms_test_new2 / 1e6);
            if (ms_test_new > 0 && ms_test_new2 > 0)
                std::printf(
                    "    -- test-new (realistic underflow) vs test-new2 (uniform, ~0%% underflow): %.3fx\n",
                    (flops / ms_test_new) / (flops / ms_test_new2));

            // "test-new3": SAME dispatcher/data as test-new2 (uniform
            // ~0% underflow wh), but the large-M branch routes to the
            // kMvRowsPerThread=2 sibling kernel instead of test-new's own
            // proven kMvRowsPerThread=4 -- isolates the register-
            // blocking-vs-occupancy question as the ONLY variable
            // (same decode, same data, same shape-aware small-M branch).
            // See fp32_16_cuda_kernels.cu's own comment on the new
            // kernel for what this measures and why.
            ms_test_new3 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_test_new3_dispatch(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_test_new3_dispatch (test-new3, rows/thread=2, uniform ~0%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                ms_test_new3, flops / ms_test_new3 / 1e6);
            if (ms_test_new2 > 0 && ms_test_new3 > 0)
                std::printf(
                    "    -- test-new2 (rows/thread=4) vs test-new3 (rows/thread=2), both uniform ~0%% underflow: %.3fx\n",
                    (flops / ms_test_new2) / (flops / ms_test_new3));

            // "test-new4": SAME data/dispatcher as test-new2/test-new3,
            // but the large-M branch routes to the kMvRowsPerThread=8
            // sibling kernel -- extends the register-blocking-vs-
            // occupancy trend in the opposite direction from test-new3.
            // See fp32_16_cuda_kernels.cu's own comment on the new
            // kernel for what this measures and why.
            ms_test_new4 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half_test_new4_dispatch(
                    1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_test_new4_dispatch (test-new4, rows/thread=8, uniform ~0%% underflow) = %9.3fms (%10.1f GFlop/s)\n",
                ms_test_new4, flops / ms_test_new4 / 1e6);
            if (ms_test_new2 > 0 && ms_test_new4 > 0)
                std::printf(
                    "    -- test-new2 (rows/thread=4) vs test-new4 (rows/thread=8), both uniform ~0%% underflow: %.3fx\n",
                    (flops / ms_test_new2) / (flops / ms_test_new4));

            cudaFree(wh);

            // "FP16 (double acc)" -- NOT an fp32_16-packed kernel at all:
            // A here is plain native __half (own dedicated buffer, own
            // gen_half fill, same generator the cuBLAS Hgemm leg below
            // uses), x/y stay float. No tuned (half, float, float) kernel
            // exists in this codebase's generic Dense::lookup_mv registry,
            // so this is exactly gemv_generic_kernel (mv_cuda_kernels.cu):
            // every A/x element upcast to double, accumulated in double,
            // only the final sum cast down to y. Included here purely to
            // make that silent fallback's cost/behavior visible next to
            // the genuinely-tuned "TOMOS FP16" series above -- see this
            // file's own forward-declaration comment for mv_strided.
            __half* a_double_acc = nullptr;
            bool ok_a_double_acc = alloc_or_skip(reinterpret_cast<void**>(&a_double_acc), m * k * sizeof(__half),
                                                 "generic fp16/double-acc (A)", shape_desc);
            if (ok_a_double_acc) {
                gen_half(m * k, a_double_acc);
                cudaDeviceSynchronize();
                ms_double_acc = time_call(start, stop, [&]() {
                    cuda::experimental::mv_strided<__half, float, float>(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, a_double_acc, static_cast<size_t>(lda), x, 1,
                        0.3, y, 1, handle);
                }, reps);
                std::printf("    mv_generic_fp16_double_acc = %9.3fms (%10.1f GFlop/s)\n", ms_double_acc,
                           flops / ms_double_acc / 1e6);
                cudaFree(a_double_acc);
            }

            // "TOMOS FP32/FP16 (mixed precision)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_half2_atomic_mixed's own NO_TRANSPOSE path:
            // unlike plain "TOMOS FP16" above (which saturates control=1
            // entries to +-Infinity/0) and unlike "TOMOS FP32/FP16" hybrid
            // below (wh-only APPROXIMATE reconstruction of the extended
            // branch only, via mv_fp32_16_hybrid_process_element), this
            // decodes control=1 entries EXACTLY via fp32_16_combine(wh,wr)
            // -- covering the degraded branch too, which is unrecoverable
            // from wh alone -- at the cost of needing wr as a second input
            // array. Own persistent wh/wr pair (NOT the scratch-then-freed
            // wr_hybrid_scratch below -- this kernel's wr has to survive
            // the actual kernel launch, not just the generator). Same
            // mixed normal/extended/degraded-range generator as the hybrid
            // family below (gen_fp32_16_wh_hybrid, ~10% control=1), so
            // this exercises the same interesting entries they do.
            uint16_t *wh_mixed = nullptr, *wr_mixed = nullptr;
            bool ok_wh_mixed =
                alloc_or_skip(reinterpret_cast<void**>(&wh_mixed), m * k * sizeof(uint16_t),
                             "pure_half_wide_ilp_shared_half2_atomic_mixed (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_mixed), m * k * sizeof(uint16_t),
                             "pure_half_wide_ilp_shared_half2_atomic_mixed (wr)", shape_desc);
            if (ok_wh_mixed) {
                gen_fp32_16_wh_hybrid(m * k, wh_mixed, wr_mixed);
                cudaDeviceSynchronize();
                ms_half2_atomic_mixed = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed(
                        TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_mixed, wr_mixed,
                        static_cast<size_t>(lda), x, 1, 0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed = %9.3fms (%10.1f GFlop/s)\n",
                    ms_half2_atomic_mixed, flops / ms_half2_atomic_mixed / 1e6);
                cudaFree(wh_mixed);
                cudaFree(wr_mixed);
            } else if (wh_mixed) {
                cudaFree(wh_mixed);
            }
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

    // "TOMOS FP16" (native storage, "fastequiv") -- mv_fp16_native_storage_
    // fastequiv_wide_ilp_shared_half2_atomic_best's own NO_TRANSPOSE path:
    // a TRUE mechanical duplicate of "TOMOS FP16 (implicit, fast)"'s own
    // entire dispatch tree (see that function's own comment in fp32_16_
    // cuda_kernels.cu) with every fp32_16_half_only_fast(wh[idx]) decode
    // call replaced by a direct A[idx] read -- A is genuine native __half
    // here, not FP32_16-packed storage, so there is nothing to decode.
    // Same col_splits formulas, same bigtile/wide4x crossover, same
    // atomicAdd combine, same __hfma2 packing as the fast-decode family --
    // isolates decode-vs-no-decode as the ONLY variable, after three
    // earlier attempts at a native-storage "TOMOS FP16" (col-split-less,
    // then scalar-atomic, then a from-scratch col-split retune) each left
    // a real, measured gap against "TOMOS FP16 (implicit, fast)" from
    // independently reconstructing (not exactly duplicating) that
    // family's own tuning. x/y stay float, matching "TOMOS FP16
    // (implicit, fast)" exactly (only A's storage/decode is the isolated
    // variable) -- unlike this series' own prior all-half A/x/y
    // convention, so this is no longer a strict type-for-type match
    // against cuBLAS Hgemm's own all-half I/O the way mv_fp16_native_
    // storage itself (still available, unused by this series now) is.
    {
        __half* A_n = nullptr;
        float *x_n = nullptr, *y_n = nullptr;
        bool ok_n = alloc_or_skip(reinterpret_cast<void**>(&A_n), m * k * sizeof(__half), "native_storage (A_n)",
                                  shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&x_n), k * sizeof(float), "native_storage (x_n)",
                                 shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&y_n), m * sizeof(float), "native_storage (y_n)",
                                 shape_desc);
        if (ok_n) {
            gen_half(m * k, A_n);
            gen_float(k, x_n);
            cudaMemset(y_n, 0, m * sizeof(float));
            cudaDeviceSynchronize();

            ms_native_storage = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, A_n, static_cast<size_t>(lda), x_n, 1, 0.3, y_n, 1);
            }, reps);
            std::printf("    mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best = %9.3fms (%10.1f GFlop/s)\n", ms_native_storage,
                       flops / ms_native_storage / 1e6);

            // "test" -- mv_fp16_native_mixed (mv_cuda_kernels.cu): true
            // mechanical duplicate of "TOMOS FP16" (mv_fp16_native_storage)'s
            // own entire dispatch tree (same __hfma2 packing, same
            // col_splits formulas/crossover, same reduce-vs-atomic "_best"
            // split), but with x/y kept at float instead of __half --
            // matching mv_fp16_native's own existing mixed-precision
            // contract, the (CUDA, FP16, FP32, FP32, compute=FP16) slot
            // lsqr_precision_runner's --explicit-duplicates now actually
            // dispatches to (dense_registries.cpp). Reuses this block's own
            // A_n/x_n/y_n buffers (already the right types: A_n=__half,
            // x_n/y_n=float) since it's the same operand-type shape as the
            // series just above. Plotted here purely to sanity-check its
            // performance now that it is live in LSQR's own production
            // path, not because it belongs in the FP32_16-packed "test"
            // slot's own lineage.
            ms_native_mixed = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp16_native_mixed(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, A_n, static_cast<size_t>(lda), x_n, 1, 0.3, y_n, 1);
            }, reps);
            std::printf("    mv_fp16_native_mixed = %9.3fms (%10.1f GFlop/s)\n", ms_native_mixed,
                       flops / ms_native_mixed / 1e6);
            if (ms_test_new > 0 && ms_native_mixed > 0)
                std::printf("    -- test-new (branchless, realistic ~18%% underflow) vs native_mixed (explicit): %.3fx\n",
                           (flops / ms_test_new) / (flops / ms_native_mixed));

            cudaFree(A_n);
            cudaFree(x_n);
            cudaFree(y_n);
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
    if (ms_half2_atomic_best_fast2 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_best_fast2) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_best_fast2 > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_half2_atomic_best_fast2) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_best_fast2 > 0 && ms_half2_atomic_best_fast > 0)
        std::printf(
            "    -- fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best vs fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_half2_atomic_best_fast2) / (flops / ms_half2_atomic_best_fast));
    if (ms_half2_atomic_lowsplit > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_lowsplit) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_lowsplit > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_half2_atomic_lowsplit) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_lowsplit > 0 && ms_half2_atomic > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit vs fp32_16_pure_half_wide_ilp_shared_half2_atomic: %.3fx\n",
            (flops / ms_half2_atomic_lowsplit) / (flops / ms_half2_atomic));
    if (ms_half2_split > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_split vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_split) / (flops / ms_cublas_hgemm));
    if (ms_half2_split > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_split vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (\"TOMOS FP16 (implicit)\"): %.3fx\n",
            (flops / ms_half2_split) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_prefetch > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_prefetch) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_prefetch > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (\"TOMOS FP16 (implicit)\"): %.3fx\n",
            (flops / ms_half2_atomic_prefetch) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_prefetch2 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_prefetch2) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_prefetch2 > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2 vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (\"TOMOS FP16 (implicit)\"): %.3fx\n",
            (flops / ms_half2_atomic_prefetch2) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_cpasync > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_cpasync) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_cpasync > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (\"TOMOS FP16 (implicit)\"): %.3fx\n",
            (flops / ms_half2_atomic_cpasync) / (flops / ms_half2_atomic_best));
    if (ms_half2_atomic_cpasync_warpuniform > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_cpasync_warpuniform) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_cpasync_warpuniform > 0 && ms_half2_atomic_cpasync > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync (\"test\"): %.3fx\n",
            (flops / ms_half2_atomic_cpasync_warpuniform) / (flops / ms_half2_atomic_cpasync));
    if (ms_half2_combine_best > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_combine_best vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_combine_best) / (flops / ms_cublas_hgemm));
    if (ms_half2_combine_best > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_combine_best vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (\"TOMOS FP16 (implicit)\"): %.3fx\n",
            (flops / ms_half2_combine_best) / (flops / ms_half2_atomic_best));
    if (ms_double_acc > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- mv_generic_fp16_double_acc vs cublas_hgemm: %.3fx\n",
                   (flops / ms_double_acc) / (flops / ms_cublas_hgemm));
    if (ms_double_acc > 0 && ms_half2_atomic_best > 0)
        std::printf("    -- mv_generic_fp16_double_acc vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best: %.3fx\n",
                   (flops / ms_double_acc) / (flops / ms_half2_atomic_best));
    if (ms_native_storage > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- mv_fp16_native_storage vs cublas_hgemm: %.3fx\n",
                   (flops / ms_native_storage) / (flops / ms_cublas_hgemm));
    if (ms_native_storage > 0 && ms_half2_atomic_best > 0)
        std::printf("    -- mv_fp16_native_storage vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best: %.3fx\n",
                   (flops / ms_native_storage) / (flops / ms_half2_atomic_best));
    if (ms_native_mixed > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- mv_fp16_native_mixed vs cublas_hgemm: %.3fx\n",
                   (flops / ms_native_mixed) / (flops / ms_cublas_hgemm));
    if (ms_native_mixed > 0 && ms_native_storage > 0)
        std::printf("    -- mv_fp16_native_mixed vs mv_fp16_native_storage (\"TOMOS FP16\"): %.3fx\n",
                   (flops / ms_native_mixed) / (flops / ms_native_storage));
    if (ms_half2_atomic_mixed > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed vs cublas_hgemm: %.3fx\n",
                   (flops / ms_half2_atomic_mixed) / (flops / ms_cublas_hgemm));
    if (ms_half2_atomic_mixed > 0 && ms_half2_atomic_best > 0)
        std::printf(
            "    -- fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed vs fp32_16_pure_half_wide_ilp_shared_half2_atomic_best: %.3fx\n",
            (flops / ms_half2_atomic_mixed) / (flops / ms_half2_atomic_best));
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

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

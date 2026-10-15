// Standalone throughput comparison for the "full fp32 reconstruction"
// tier of both packed-precision families -- mm_fp32_16_tiled_cpasync_
// dualbuf (src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu) and mm_
// fp32_bf16_tiled_cpasync_dualbuf (src/kernels/cuda/matrix/fp32_bf16_
// cuda_kernels.cu) -- against this codebase's own best native (unpacked)
// fp32 GEMM, real CUTLASS's fp32 SIMT device::Gemm, and cuBLAS's Sgemm.
// All five run in one process so the numbers share the same GPU clock/
// thermal/contention state. Mirrors wide_bigreg_sq_fp32_16_vs_cutlass_
// ref.cu's own structure and reasoning.
//
// "native" is gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
// midpoint_exact (mm_cuda_kernels.cu). It was promoted from gemm_fp32_
// tiled_syncpipe3 (the no-cp.async, register-double-buffered pipeline
// modeled on CUTLASS's own MmaPipelined::gemm_iters; the call is left
// commented out below, not deleted, alongside the still-earlier
// gemm_fp32_tiled_cpasync_dualbuf call it had itself replaced) after a
// second, longer investigation on top of the syncpipe3 one described
// below. Short version of that second investigation: CUTLASS's actual
// warp/lane geometry (RowMajorInterleaved<2> lane mapping) plus BK=8
// plus a 4-stage pipeline plus rolling-pointer addressing plus a snake
// (boustrophedon) FFMA accumulate order together produced gemm_fp32_
// tiled_syncpipe4_bk8_warptile_nostride_snake, itself measured better
// than syncpipe3 (0.960x vs CUTLASS at N=8192, vs syncpipe3's 0.951x).
// From there, interleaving that kernel's next-kk/next-tile prefetch
// into the FFMA row loop (one checkpoint at i==4, instead of batching
// the prefetch entirely before the loop) raised the FFMA .reuse ratio
// but initially cost a register and picked up a light spill (2 STL + 2
// LDL), which measured WORSE than plain snake despite the higher reuse
// ratio -- until the checkpoint's bounds-check predicates (needed only
// for non-exact-multiple problem sizes) were dropped entirely, exactly
// at that pressure point, guarded by a host-side dispatch that falls
// back to the safe, bounds-checked kernel otherwise. That combination
// -- midpoint_exact -- eliminated the spill, pushed the reuse ratio to
// 83.8%, and measured 0.977x vs CUTLASS at N=8192 and 0.982x at
// N=32768 (where it also beats cuBLAS), the best native fp32 kernel
// found in either investigation.
//
// The syncpipe3-vs-CUTLASS investigation this was built on top of:
// syncpipe3 measured 0.951x vs CUTLASS at N=8192 vs dualbuf's 0.933x,
// and every attempt afterwards to improve on syncpipe3 via rolling-
// pointer addressing alone (the whole gemm_fp32_tiled_syncpipe3_incptr*
// family -- lean, lean2, nostride, nostride_kremain, nostride_bcol,
// kremain, the 2-stage variant, and even a __launch_bounds__(256,1)
// variant that eliminates spilling entirely, matching CUTLASS's own
// 0/0/0 spill profile) measured WORSE than plain syncpipe3, never
// better -- so within that family, syncpipe3 itself was the best
// native fp32 kernel found, until the warptile/snake/midpoint_exact
// line of kernels above surpassed it.
//
// Correctness for every kernel compared here is already covered
// elsewhere -- verify_mm_fp32_16_gemm.cu for both mm_fp32_16_* and
// mm_fp32_bf16_* (including these exact dualbuf kernels), verify_mm_
// fp32_gemm.cu for gemm_fp32_tiled_cpasync_dualbuf and gemm_fp32_
// tiled_syncpipe3 -- so this file is throughput-only, same convention
// every wide_bigreg_*_vs_cutlass_ref.cu file already uses.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call the
// tomos kernels directly. See scripts/fp32_recon_vs_cutlass_ref.sh.
//
// The CUTLASS config here is OpClassSimt (not OpClassTensorOp/tf32):
// this comparison is specifically about matching real fp32 numerics
// (mm_fp32_16_accurate/mm_fp32_bf16_accurate were both built to that
// requirement -- see their own comments), and tf32 tensor cores truncate
// the mantissa to 10 bits, which isn't real fp32. No plain-float CUTLASS
// config existed anywhere in this codebase's benchmark files before this
// one; rather than hand-picking a tile/warp/instruction shape (a real
// correctness/perf risk to guess wrong, unlike the half_t configs this
// file's siblings already had working examples of), this uses CUTLASS's
// own per-arch default Gemm template arguments for OpClassSimt+Sm80+
// float -- the library's own intended usage when not hand-tuning.
//
// Usage:
//   fp32_recon_vs_cutlass_ref
//       Sweeps two families of shapes: square N=1024/2048/4096/8192/
//       16384/32768/48000/54000 (M=K=N), then a rectangular sweep with
//       the output held fixed at M=N=44000 and the inner/contraction
//       dimension K swept across 1024/2048/4096/8192/16384/32768 (2^10
//       through 2^15) -- stress-testing how each kernel's performance
//       scales with K independent of output size, since every kernel
//       here was originally tuned assuming M=K=N.
//   fp32_recon_vs_cutlass_ref 8192
//       Just that one square size (M=K=N=8192).
//   fp32_recon_vs_cutlass_ref --rect
//       Just the rectangular K-sweep (M=N=44000, K=1024..32768), skipping
//       the square sweep entirely.
//   fp32_recon_vs_cutlass_ref --rect 8192
//       Just the rectangular shape M=N=44000, K=8192.
//   fp32_recon_vs_cutlass_ref --rect 8192 49152
//       Just the rectangular shape M=N=49152, K=8192 -- the optional 3rd
//       arg overrides the M=N outer dimension (default 44000).
//   fp32_recon_vs_cutlass_ref --subnormal-density 10 8192
//       M=K=N=8192, with 10% of every generator's own entries redrawn
//       into the fp32 subnormal band (0 < |x| < 2^-126) instead of
//       pseudo_rand's own uniform-on-[-1,1] -- see run_shape's own
//       comment. May appear anywhere in the argument list; every other
//       shape-selection flag/positional above still works unchanged.
//       See scripts/fp32_recon_vs_cutlass_ref_subnormal.sh.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                      double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType transA, TransposeType transB, double alpha,
                                                dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_tiled_cpasync_dualbuf_adaptive(TransposeType transA, TransposeType transB, double alpha,
                                               dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                               size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                               double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp(TransposeType transA, TransposeType transB, double alpha,
                                                     dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                     size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                     double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_tiled_cpasync_dualbuf_multin(TransposeType transA, TransposeType transB, double alpha,
                                             dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                             size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                             double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                   double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_gemm_materialize(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_16_gemm_materialize_midpoint_exact_vec(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c); }  // namespace experimental

void mm_fp32_bf16_gemm_materialize_rounded(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);
namespace experimental { void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_cpasync_dualbuf_regpipe(TransposeType transA, TransposeType transB, double alpha,
                                             size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                             float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_dualbuf_syncpipe(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_dualbuf_syncpipe_incptr(TransposeType transA, TransposeType transB, double alpha,
                                             size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                             float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_lean(TransposeType transA, TransposeType transB, double alpha,
                                           size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                           float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_lean2(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_nostride(TransposeType transA, TransposeType transB, double alpha,
                                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                               float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain(TransposeType transA, TransposeType transB, double alpha,
                                                        size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                        float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol(TransposeType transA, TransposeType transB, double alpha,
                                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_kremain(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                              float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride(TransposeType transA, TransposeType transB, double alpha,
                                                       size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                       float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                    float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_finegrain(TransposeType transA, TransposeType transB, double alpha,
                                         size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                         float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_dualacc(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                       float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_snake(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe3_bk8(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                   float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                   float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride(TransposeType transA, TransposeType transB, double alpha,
                                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c); }  // namespace experimental

void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

using ElementA = float;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = float;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;

// Deliberately using CUTLASS's own default template arguments for the
// tile/warp/instruction shape and pipeline depth (see this file's own
// header comment for why) -- only the element types, layouts, op class,
// and arch tag are specified.
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80
>;

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

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}

// Truncating counterpart of gen_fp32_16 -- feeds mm_fp32_16_tiled_
// cpasync_dualbuf_truncated, which requires wh/wr from fp32_16_split_
// truncated specifically (see that kernel's own correctness contract).
__global__ void gen_fp32_16_truncated_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_truncated(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16_truncated(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_truncated_kernel<<<blocks, threads>>>(n, wh, wr);
}

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

__global__ void gen_fp32_bf16_rounded_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split_rounded(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16_rounded(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_rounded_kernel<<<blocks, threads>>>(n, wh, wr);
}

// Tunable-subnormal-density siblings of every generator above -- used
// only when --subnormal-density is passed (main() below), so every
// existing caller of gen_float/gen_fp32_16/gen_fp32_16_truncated/
// gen_fp32_bf16/gen_fp32_bf16_rounded is untouched and produces bit-
// identical data to before. density is a FRACTION in [0, 1] (main()
// converts the CLI's percentage argument). For each element, a second,
// decorrelated hash off the same per-index seed decides (with
// probability `density`) whether that element is redrawn into the fp32
// subnormal band (0 < |x| < 2^-126, sign and magnitude uniform in that
// band) instead of pseudo_rand's own uniform-on-[-1,1] value -- so the
// realized subnormal fraction matches `density` in expectation (not
// exactly per-call, same as pseudo_rand's own approximate uniformity),
// and setting density=0 exactly reproduces pseudo_rand's own value at
// every index (the "redraw" branch is simply never taken). Unlike fp16_
// bf16_pure_vs_cutlass_ref.cu's own version of this helper (calibrated
// to fp16's much wider subnormal band, and deliberately NOT applied to
// its bf16 group since that format computes NATIVELY in bf16), every
// series in THIS file ultimately reconstructs to (or, for the native/
// cutlass/cublas leg, starts as) genuine float -- so one fp32-band
// density applies meaningfully to every generator here, fp32_16- and
// fp32_bf16-encoded alike, with no group excluded.
__device__ float pseudo_rand_density(size_t idx, float density) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float u = (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // uniform [-1, 1], same grid as pseudo_rand

    uint32_t sel = s * 2246822519u + 3266489917u;  // decorrelated second hash off the same per-index seed
    sel ^= sel >> 13;
    bool subnormal = (sel % 1000000u) < static_cast<uint32_t>(density * 1000000.0f + 0.5f);
    if (subnormal) {
        const float kFp32SubnormalBound = 1.1754943508222875e-38f;  // 2^-126, the fp32 normal/subnormal boundary
        return copysignf(fabsf(u) * kFp32SubnormalBound, u);  // uniform in (0, 2^-126), u's own sign
    }
    return u;
}

__global__ void gen_float_density_kernel(size_t n, float* out, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand_density(i, density);
}
void gen_float_density(size_t n, float* out, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_density_kernel<<<blocks, threads>>>(n, out, density);
}

__global__ void gen_fp32_16_density_kernel(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_density(i, density), wh[i], wr[i]);
}
void gen_fp32_16_density(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_density_kernel<<<blocks, threads>>>(n, wh, wr, density);
}

__global__ void gen_fp32_16_truncated_density_kernel(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_truncated(pseudo_rand_density(i, density), wh[i], wr[i]);
}
void gen_fp32_16_truncated_density(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_truncated_density_kernel<<<blocks, threads>>>(n, wh, wr, density);
}

__global__ void gen_fp32_bf16_density_kernel(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(pseudo_rand_density(i, density), wh[i], wr[i]);
}
void gen_fp32_bf16_density(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_density_kernel<<<blocks, threads>>>(n, wh, wr, density);
}

__global__ void gen_fp32_bf16_rounded_density_kernel(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split_rounded(pseudo_rand_density(i, density), wh[i], wr[i]);
}
void gen_fp32_bf16_rounded_density(size_t n, uint16_t* wh, uint16_t* wr, float density) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_rounded_density_kernel<<<blocks, threads>>>(n, wh, wr, density);
}

}  // namespace

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / reps;
}

// Peak footprint reasoning (see run_shape's own comment for the full
// derivation): B/C are shared and live for the whole call; wh16/wr16,
// wh_bf16/wr_bf16, and A_native are each transient, allocated right
// before and freed right after their own timing block, since no two of
// them are ever needed simultaneously.
bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). The square sweep calls this
// with m=k=n; the rectangular sweep holds m=n=44000 fixed and varies k.
// density: fraction in [0, 1] of every generator's own entries redrawn
// into the fp32 subnormal band (see pseudo_rand_density's own comment
// above) -- 0.0 (the default, all existing callers) reproduces this
// function's original behavior exactly.
void run_shape(size_t m, size_t k, size_t n, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle,
               float density = 0.0f) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);

    float *B, *C;
    if (!alloc_or_skip(reinterpret_cast<void**>(&B), k * n * sizeof(float), "B", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&C), m * n * sizeof(float), "C", shape_desc)) {
        std::printf("  %-24s SKIPPED (out of device memory for B/C alone)\n", shape_desc);
        cudaFree(B);
        cudaFree(C);
        return;
    }
    if (density > 0.0f) {
        gen_float_density(k * n, B, density);
    } else {
        gen_float(k * n, B);
    }
    cudaMemset(C, 0, m * n * sizeof(float));
    cudaDeviceSynchronize();

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    size_t maxdim = std::max({m, k, n});
    int reps = (maxdim <= 2048) ? 20 : (maxdim <= 8192 ? 5 : 2);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    std::printf("  %s\n", shape_desc);

    double ms_16dualbuf = -1.0;
    {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16, wr16, density);
            } else {
                gen_fp32_16(m * k, wh16, wr16);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                       dim<2>{m, k}, wh16, wr16, lda, dim<2>{k, n}, B, ldb, 0.3, C,
                                                       ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf   = %9.3fms (%10.1f GFlop/s)\n", ms_16dualbuf,
                       flops / ms_16dualbuf / 1e6);
        }
        cudaFree(wh16);
        cudaFree(wr16);
    }

    // UNCONDITIONAL fast_normal decode sibling of mm_fp32_16_tiled_
    // cpasync_dualbuf above -- byte-for-byte identical kernel except A's
    // decode skips the exponent-mode dispatch entirely (fp32_16_combine_
    // fast_normal, no per-tile validity check). Same wh/wr data as the
    // plain dualbuf timing above (gen_fp32_16, not a separate generator)
    // -- this is a pure decode-cost comparison, not a data-generation
    // change. UNSAFE ON ARBITRARY DATA (see that kernel's own comment in
    // fp32_16_cuda_kernels.cu); timed here purely for throughput, not
    // correctness -- see mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu
    // for the accompanying error measurement.
    double ms_16dualbuf_fast = -1.0;
    {
        uint16_t *wh16f = nullptr, *wr16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_fast_normal (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_fast_normal (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16f, wr16f, density);
            } else {
                gen_fp32_16(m * k, wh16f, wr16f);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf_fast = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    1.7, dim<2>{m, k}, wh16f, wr16f, lda, dim<2>{k, n}, B,
                                                                    ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_fast_normal = %9.3fms (%10.1f GFlop/s)\n", ms_16dualbuf_fast,
                       flops / ms_16dualbuf_fast / 1e6);
        }
        cudaFree(wh16f);
        cudaFree(wr16f);
    }

    // Same kernel architecture as above, but A/B'd against the never-
    // rounded, no-round_flag-correction decode this format used before
    // mant_hi rounding was added -- wh16t/wr16t come from fp32_16_split_
    // truncated specifically, decoded via mm_fp32_16_tiled_cpasync_
    // dualbuf_truncated's own fp32_16_combine_truncated call.
    double ms_16dualbuf_trunc = -1.0;
    {
        uint16_t *wh16t = nullptr, *wr16t = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16t), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_truncated (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16t), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_truncated (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_truncated_density(m * k, wh16t, wr16t, density);
            } else {
                gen_fp32_16_truncated(m * k, wh16t, wr16t);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf_trunc = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                 1.7, dim<2>{m, k}, wh16t, wr16t, lda, dim<2>{k, n}, B,
                                                                 ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_truncated = %9.3fms (%10.1f GFlop/s)\n", ms_16dualbuf_trunc,
                       flops / ms_16dualbuf_trunc / 1e6);
        }
        cudaFree(wh16t);
        cudaFree(wr16t);
    }

    // Fused midpoint_exact-based counterpart to mm_fp32_16_tiled_
    // cpasync_dualbuf_truncated above -- same relationship the plain
    // (non-truncated) fused kernel has to dualbuf. Compiles with a real
    // spill (26 STL + 17 LDL at 128 registers) -- fp32_16_combine_
    // truncated apparently has a meaningfully larger register footprint
    // than plain fp32_16_combine, unlike the clean 0-spill non-
    // truncated fused kernel -- so this timing is the direct test of
    // how much that costs in practice.
    double ms_16fused_midpoint_exact_trunc = -1.0;
    {
        uint16_t *wh16ft = nullptr, *wr16ft = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16ft), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16ft), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_truncated_density(m * k, wh16ft, wr16ft, density);
            } else {
                gen_fp32_16_truncated(m * k, wh16ft, wr16ft);
            }
            cudaDeviceSynchronize();
            ms_16fused_midpoint_exact_trunc = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                 1.7, dim<2>{m, k}, wh16ft, wr16ft, lda, dim<2>{k, n}, B,
                                                                 ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated = %9.3fms (%10.1f GFlop/s)\n", ms_16fused_midpoint_exact_trunc,
                       flops / ms_16fused_midpoint_exact_trunc / 1e6);
        }
        cudaFree(wh16ft);
        cudaFree(wr16ft);
    }

    // Same rounding-aware data as mm_fp32_16_tiled_cpasync_dualbuf above
    // (fresh wh16a/wr16a from the same fp32_16_split, not shared with it
    // since that block already freed its own buffers) -- this kernel is
    // expected to be bit-identical to it, just via a block-wide adaptive
    // fast/full decode split instead of always paying the full decode.
    double ms_16dualbuf_adaptive = -1.0;
    {
        uint16_t *wh16a = nullptr, *wr16a = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16a), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_adaptive (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16a), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_adaptive (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16a, wr16a, density);
            } else {
                gen_fp32_16(m * k, wh16a, wr16a);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf_adaptive = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_adaptive(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                1.7, dim<2>{m, k}, wh16a, wr16a, lda, dim<2>{k, n}, B,
                                                                ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_adaptive = %9.3fms (%10.1f GFlop/s)\n", ms_16dualbuf_adaptive,
                       flops / ms_16dualbuf_adaptive / 1e6);
        }
        cudaFree(wh16a);
        cudaFree(wr16a);
    }

    // Same fast/full split as mm_fp32_16_tiled_cpasync_dualbuf_adaptive,
    // but voted on per-warp (__any_sync) instead of per-block
    // (__syncthreads_or) -- see that kernel's own comment for why the
    // per-block version regressed and why per-warp should avoid it.
    double ms_16dualbuf_adaptive_warp = -1.0;
    {
        uint16_t *wh16aw = nullptr, *wr16aw = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16aw), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16aw), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16aw, wr16aw, density);
            } else {
                gen_fp32_16(m * k, wh16aw, wr16aw);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf_adaptive_warp = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                     1.7, dim<2>{m, k}, wh16aw, wr16aw, lda, dim<2>{k, n}, B,
                                                                     ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16dualbuf_adaptive_warp, flops / ms_16dualbuf_adaptive_warp / 1e6);
        }
        cudaFree(wh16aw);
        cudaFree(wr16aw);
    }

    // Attacks the redundant-decode problem by halving how many blocks
    // redundantly decode a given M-row range (each block covers 2
    // N-tiles, reusing one shared-memory A decode for both) instead of
    // making the decode itself cheaper.
    double ms_16dualbuf_multin = -1.0;
    {
        uint16_t *wh16m = nullptr, *wr16m = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16m), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_multin (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16m), m * k * sizeof(uint16_t),
                          "mm_fp32_16_tiled_cpasync_dualbuf_multin (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16m, wr16m, density);
            } else {
                gen_fp32_16(m * k, wh16m, wr16m);
            }
            cudaDeviceSynchronize();
            ms_16dualbuf_multin = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_tiled_cpasync_dualbuf_multin(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                               1.7, dim<2>{m, k}, wh16m, wr16m, lda, dim<2>{k, n}, B,
                                                               ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_tiled_cpasync_dualbuf_multin = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16dualbuf_multin, flops / ms_16dualbuf_multin / 1e6);
        }
        cudaFree(wh16m);
        cudaFree(wr16m);
    }

    // Attacks the same problem by eliminating the redundancy entirely:
    // materializes A to plain fp32 (in bounded-size row chunks) then
    // calls the native fp32 GEMM, which needs no decode at all.
    double ms_16gemm_materialize = -1.0;
    {
        uint16_t *wh16g = nullptr, *wr16g = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16g), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16g), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16g, wr16g, density);
            } else {
                gen_fp32_16(m * k, wh16g, wr16g);
            }
            cudaDeviceSynchronize();
            ms_16gemm_materialize = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_gemm_materialize(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16g, wr16g, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize, flops / ms_16gemm_materialize / 1e6);
        }
        cudaFree(wh16g);
        cudaFree(wr16g);
    }

    // Same materialize-then-GEMM structure as mm_fp32_16_gemm_materialize
    // above, but the second-stage native GEMM call is midpoint_exact
    // (the current best native fp32 kernel) instead of gemm_fp32_tiled_
    // cpasync_dualbuf.
    double ms_16gemm_materialize_midpoint_exact = -1.0;
    {
        uint16_t *wh16gme = nullptr, *wr16gme = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16gme), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16gme), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16gme, wr16gme, density);
            } else {
                gen_fp32_16(m * k, wh16gme, wr16gme);
            }
            cudaDeviceSynchronize();
            ms_16gemm_materialize_midpoint_exact = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16gme, wr16gme, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize_midpoint_exact, flops / ms_16gemm_materialize_midpoint_exact / 1e6);
        }
        cudaFree(wh16gme);
        cudaFree(wr16gme);
    }

    // UNCONDITIONAL fast_normal decode sibling of mm_fp32_16_gemm_
    // materialize_midpoint_exact above -- identical materialize-then-
    // native-GEMM structure (same chunking, same unmodified GEMM compute
    // stage), except the chunk-decode pass skips the exponent-mode
    // dispatch entirely (fp32_16_combine_fast_normal, no per-element
    // validity check). Same wh/wr data as the plain materialize timing
    // above. UNSAFE ON ARBITRARY DATA; timed here purely for throughput,
    // not correctness -- see mm_fp32_compute32_l2_error_vs_cublas_dgemm_
    // ref.cu for the accompanying error measurement.
    double ms_16gemm_materialize_midpoint_exact_fast = -1.0;
    {
        uint16_t *wh16gmef = nullptr, *wr16gmef = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16gmef), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16gmef), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16gmef, wr16gmef, density);
            } else {
                gen_fp32_16(m * k, wh16gmef, wr16gmef);
            }
            cudaDeviceSynchronize();
            ms_16gemm_materialize_midpoint_exact_fast = time_call(start, stop, [&]() {
                cuda::mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16gmef, wr16gmef, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize_midpoint_exact_fast, flops / ms_16gemm_materialize_midpoint_exact_fast / 1e6);
        }
        cudaFree(wh16gmef);
        cudaFree(wr16gmef);
    }

    // EXPERIMENT: vectorized-decode counterpart to the materialize
    // kernel above -- same materialize strategy, just 4 rows/thread via
    // ushort4/float4 instead of 1 scalar element/thread in the chunk-
    // decode pass (the ENTIRE extra cost materialize pays over plain
    // native fp32; the GEMM stage is unchanged either way). This timing
    // is what answers whether vectorizing that pass actually moves the
    // needle on materialize's total time.
    double ms_16gemm_materialize_midpoint_exact_vec = -1.0;
    {
        uint16_t *wh16gmev = nullptr, *wr16gmev = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16gmev), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact_vec (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16gmev), m * k * sizeof(uint16_t),
                          "mm_fp32_16_gemm_materialize_midpoint_exact_vec (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16gmev, wr16gmev, density);
            } else {
                gen_fp32_16(m * k, wh16gmev, wr16gmev);
            }
            cudaDeviceSynchronize();
            ms_16gemm_materialize_midpoint_exact_vec = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_gemm_materialize_midpoint_exact_vec(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16gmev, wr16gmev, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_gemm_materialize_midpoint_exact_vec = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16gemm_materialize_midpoint_exact_vec, flops / ms_16gemm_materialize_midpoint_exact_vec / 1e6);
        }
        cudaFree(wh16gmev);
        cudaFree(wr16gmev);
    }

    // The fully-fused counterpart: decode happens straight into the
    // tile pipeline's shared-memory A buffer (no separate scratch
    // buffer or elementwise decode kernel).
    double ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = -1.0;
    {
        uint16_t *wh16f = nullptr, *wr16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr16f), m * k * sizeof(uint16_t),
                          "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_16_density(m * k, wh16f, wr16f, density);
            } else {
                gen_fp32_16(m * k, wh16f, wr16f);
            }
            cudaDeviceSynchronize();
            ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                  1.7, dim<2>{m, k}, wh16f, wr16f, lda, dim<2>{k, n}, B,
                                                  ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact, flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact / 1e6);
        }
        cudaFree(wh16f);
        cudaFree(wr16f);
    }

    double ms_bf16dualbuf = -1.0;
    {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_tiled_cpasync_dualbuf (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_tiled_cpasync_dualbuf (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_density(m * k, wh_bf16, wr_bf16, density);
            } else {
                gen_fp32_bf16(m * k, wh_bf16, wr_bf16);
            }
            cudaDeviceSynchronize();
            ms_bf16dualbuf = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16, wr_bf16, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_tiled_cpasync_dualbuf = %9.3fms (%10.1f GFlop/s)\n", ms_bf16dualbuf,
                       flops / ms_bf16dualbuf / 1e6);
        }
        cudaFree(wh_bf16);
        cudaFree(wr_bf16);
    }

    // FP32/BF16 counterpart to the fused midpoint_exact-based fp32_16
    // kernel above -- same tiling structure, decode via fp32_bf16_
    // combine instead of fp32_16_combine.
    double ms_bf16fused_midpoint_exact = -1.0;
    {
        uint16_t *wh_bf16f = nullptr, *wr_bf16f = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16f), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16f), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_density(m * k, wh_bf16f, wr_bf16f, density);
            } else {
                gen_fp32_bf16(m * k, wh_bf16f, wr_bf16f);
            }
            cudaDeviceSynchronize();
            ms_bf16fused_midpoint_exact = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16f, wr_bf16f, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n", ms_bf16fused_midpoint_exact,
                       flops / ms_bf16fused_midpoint_exact / 1e6);
        }
        cudaFree(wh_bf16f);
        cudaFree(wr_bf16f);
    }

    // Rounded-encoding counterpart to the fused midpoint_exact-based
    // bf16 kernel above -- wh is formed by ROUNDING (round to nearest,
    // ties to even) instead of truncating, with the round decision
    // recorded in wr's own LSB (fp32_bf16_pack.hpp's fp32_bf16_split_
    // rounded/fp32_bf16_combine_rounded) so reconstruction can invert
    // it -- not bit-exact the way the plain kernel above is (loses the
    // original mantissa's own LSB), but accurate to within one ULP of
    // the full 23-bit fp32 mantissa. Compiles with a real spill (REG:128
    // STACK:56, 23 STL + 11 LDL) unlike the plain kernel's clean 0-spill
    // compile -- this timing exists to find out whether that costs it
    // the win the same way it did for fp32_16_combine_truncated vs
    // fp32_16_combine.
    double ms_bf16fused_rounded = -1.0;
    {
        uint16_t *wh_bf16r = nullptr, *wr_bf16r = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16r), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16r), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_rounded_density(m * k, wh_bf16r, wr_bf16r, density);
            } else {
                gen_fp32_bf16_rounded(m * k, wh_bf16r, wr_bf16r);
            }
            cudaDeviceSynchronize();
            ms_bf16fused_rounded = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16r, wr_bf16r, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded = %9.3fms (%10.1f GFlop/s)\n", ms_bf16fused_rounded,
                       flops / ms_bf16fused_rounded / 1e6);
        }
        cudaFree(wh_bf16r);
        cudaFree(wr_bf16r);
    }

    // Materialize (decode-once-then-black-box) alternative to the fused
    // rounded kernel above -- sidesteps its real spill entirely, since
    // materialize's decode pass is a simple per-element kernel (no GEMM
    // tiling register pressure), then hands off to the already-proven
    // 0-spill native GEMM. Same strategy that made materialize_
    // midpoint_exact win outright for fp32_16; this timing exists to
    // find out whether it does the same here.
    double ms_bf16materialize_rounded = -1.0;
    {
        uint16_t *wh_bf16rm = nullptr, *wr_bf16rm = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16rm), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_gemm_materialize_rounded (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16rm), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_gemm_materialize_rounded (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_rounded_density(m * k, wh_bf16rm, wr_bf16rm, density);
            } else {
                gen_fp32_bf16_rounded(m * k, wh_bf16rm, wr_bf16rm);
            }
            cudaDeviceSynchronize();
            ms_bf16materialize_rounded = time_call(start, stop, [&]() {
                cuda::mm_fp32_bf16_gemm_materialize_rounded(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16rm, wr_bf16rm, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_gemm_materialize_rounded = %9.3fms (%10.1f GFlop/s)\n", ms_bf16materialize_rounded,
                       flops / ms_bf16materialize_rounded / 1e6);
        }
        cudaFree(wh_bf16rm);
        cudaFree(wr_bf16rm);
    }

    // EXPERIMENT: same fused kernel as ms_bf16fused_rounded above, but
    // load_global_advance first checks -- once per BK=8 tile, via a
    // block-wide __syncthreads_or reduction -- whether the entire tile
    // has any rounded element at all, collapsing to the plain 0-spill
    // decode when it doesn't. Measured (cuobjdump): REG:128 STACK:16, 3
    // STL + 2 LDL -- a real reduction from the plain fused rounded
    // kernel's REG:128 STACK:56, 23 STL + 11 LDL, but NOT a full
    // elimination, and this adds 4 genuinely NEW BAR.RED.OR
    // instructions (real block-wide barriers, one per load_global_
    // advance call site) on top of the 2 plain BAR.SYNC both kernels
    // already share, which could easily cost more than the reduced
    // spill saves by undermining the 4-stage pipeline's own latency-
    // hiding. This timing is what actually answers that.
    double ms_bf16fused_rounded_adaptive = -1.0;
    {
        uint16_t *wh_bf16ra = nullptr, *wr_bf16ra = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16ra), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16ra), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_rounded_density(m * k, wh_bf16ra, wr_bf16ra, density);
            } else {
                gen_fp32_bf16_rounded(m * k, wh_bf16ra, wr_bf16ra);
            }
            cudaDeviceSynchronize();
            ms_bf16fused_rounded_adaptive = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16ra, wr_bf16ra, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive = %9.3fms (%10.1f GFlop/s)\n", ms_bf16fused_rounded_adaptive,
                       flops / ms_bf16fused_rounded_adaptive / 1e6);
        }
        cudaFree(wh_bf16ra);
        cudaFree(wr_bf16ra);
    }

    // EXPERIMENT: same idea as ms_bf16fused_rounded_adaptive above, but
    // the per-tile "needs correction" decision is made per-WARP via
    // __any_sync instead of per-BLOCK via __syncthreads_or. __any_sync
    // compiles to VOTE.ANY (confirmed via SASS) -- a cheap warp-only
    // predicate combine, NOT a barrier instruction -- so unlike the
    // block-wide kernel's 4 new BAR.RED.OR, this adds ZERO new barrier-
    // class instructions. Measured: REG:128 STACK:32, 6 STL + 6 LDL --
    // a smaller spill reduction than the block-wide kernel's STACK:16,
    // but at essentially none of its synchronization risk. This timing
    // is what answers whether that tradeoff nets out better in
    // practice.
    double ms_bf16fused_rounded_adaptive_warp = -1.0;
    {
        uint16_t *wh_bf16raw = nullptr, *wr_bf16raw = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16raw), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp (wh)", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_bf16raw), m * k * sizeof(uint16_t),
                          "mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp (wr)", shape_desc)) {
            if (density > 0.0f) {
                gen_fp32_bf16_rounded_density(m * k, wh_bf16raw, wr_bf16raw, density);
            } else {
                gen_fp32_bf16_rounded(m * k, wh_bf16raw, wr_bf16raw);
            }
            cudaDeviceSynchronize();
            ms_bf16fused_rounded_adaptive_warp = time_call(start, stop, [&]() {
                cuda::experimental::mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                         dim<2>{m, k}, wh_bf16raw, wr_bf16raw, lda, dim<2>{k, n}, B, ldb,
                                                         0.3, C, ldc);
            }, reps);
            std::printf("    mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp = %9.3fms (%10.1f GFlop/s)\n", ms_bf16fused_rounded_adaptive_warp,
                       flops / ms_bf16fused_rounded_adaptive_warp / 1e6);
        }
        cudaFree(wh_bf16raw);
        cudaFree(wr_bf16raw);
    }

    // A_native is shared by native/cutlass/cublas -- all three genuinely
    // need it live at once, so it's freed only after all three finish.
    double ms_native = -1.0, ms_native_regpipe = -1.0, ms_native_syncpipe = -1.0, ms_native_syncpipe_incptr = -1.0,
           ms_native_syncpipe3 = -1.0, ms_native_syncpipe3_incptr = -1.0, ms_native_syncpipe3_incptr_lean = -1.0,
           ms_native_syncpipe3_incptr_lean2 = -1.0, ms_native_syncpipe3_incptr_nostride = -1.0,
           ms_native_syncpipe3_incptr_nostride_kremain = -1.0, ms_native_syncpipe3_incptr_nostride_bcol = -1.0,
           ms_native_syncpipe3_incptr_kremain = -1.0, ms_native_dualbuf_syncpipe_incptr_nostride = -1.0,
           ms_native_syncpipe3_incptr_nostride_lb1 = -1.0, ms_native_syncpipe3_finegrain = -1.0,
           ms_native_syncpipe3_dualacc = -1.0, ms_native_syncpipe3_snake = -1.0,
           ms_native_syncpipe3_bk8 = -1.0, ms_native_syncpipe4_bk8 = -1.0,
           ms_native_syncpipe4_bk8_warptile = -1.0, ms_native_syncpipe4_bk8_warptile_nostride = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline = -1.0,
           ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = -1.0,
           ms_cutlass = -1.0, ms_cublas = -1.0;
    {
        float* A_native = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "native/cutlass/cublas (A)",
                          shape_desc)) {
            if (density > 0.0f) {
                gen_float_density(m * k, A_native, density);
            } else {
                gen_float(m * k, A_native);
            }
            cudaDeviceSynchronize();
            // "native" = gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
            // snake_midpoint_exact, promoted from gemm_fp32_tiled_
            // syncpipe3 (call kept below, commented out, not deleted) --
            // see the file-level comment above for the full rationale.
            // Short version: midpoint_exact measured 0.977x vs CUTLASS
            // at N=8192 and 0.982x at N=32768 (beating cuBLAS there),
            // vs syncpipe3's 0.951x, and it's the best native fp32
            // kernel found across both the syncpipe3_incptr_* family and
            // the warptile/snake/interleaved-checkpoint family it grew
            // out of.
            ms_native = time_call(start, stop, [&]() {
                cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
                                                m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native) = %9.3fms (%10.1f GFlop/s)\n", ms_native,
                       flops / ms_native / 1e6);
            // Previous native implementation (register-double-buffered
            // pipeline, no cp.async, no CUTLASS warp/lane geometry).
            // Left here for reference/rollback:
            //     ms_native = time_call(start, stop, [&]() {
            //         cuda::experimental::gemm_fp32_tiled_syncpipe3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
            //                                         m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            //     }, reps);
            //     std::printf("    gemm_fp32_tiled_syncpipe3 (native) = %9.3fms (%10.1f GFlop/s)\n", ms_native,
            //                flops / ms_native / 1e6);
            // Native implementation before that (both operands pipelined
            // through cp.async). Left here for reference/rollback:
            //     ms_native = time_call(start, stop, [&]() {
            //         cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, 1.7,
            //                                               m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            //     }, reps);
            //     std::printf("    gemm_fp32_tiled_cpasync_dualbuf (native) = %9.3fms (%10.1f GFlop/s)\n", ms_native,
            //                flops / ms_native / 1e6);

            ms_native_regpipe = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_cpasync_dualbuf_regpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_cpasync_dualbuf_regpipe  = %9.3fms (%10.1f GFlop/s)\n", ms_native_regpipe,
                       flops / ms_native_regpipe / 1e6);

            ms_native_syncpipe = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_dualbuf_syncpipe        = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe,
                       flops / ms_native_syncpipe / 1e6);

            ms_native_syncpipe_incptr = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                              1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_dualbuf_syncpipe_incptr = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe_incptr,
                       flops / ms_native_syncpipe_incptr / 1e6);

            // Now duplicates ms_native's own kernel (both are
            // gemm_fp32_tiled_syncpipe3, timed separately) -- kept
            // as-is since removing it would require re-plumbing every
            // "native_syncpipe3" ratio line below; the "native vs
            // native_syncpipe3" print further down is now a same-
            // kernel rerun, useful only as a run-to-run noise check.
            ms_native_syncpipe3 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3               = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe3,
                       flops / ms_native_syncpipe3 / 1e6);

            ms_native_syncpipe3_incptr = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                       1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr        = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe3_incptr,
                       flops / ms_native_syncpipe3_incptr / 1e6);

            ms_native_syncpipe3_incptr_lean = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_lean   = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe3_incptr_lean,
                       flops / ms_native_syncpipe3_incptr_lean / 1e6);

            ms_native_syncpipe3_incptr_lean2 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_lean2(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                             1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_lean2  = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe3_incptr_lean2,
                       flops / ms_native_syncpipe3_incptr_lean2 / 1e6);

            ms_native_syncpipe3_incptr_nostride = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_nostride = %9.3fms (%10.1f GFlop/s)\n", ms_native_syncpipe3_incptr_nostride,
                       flops / ms_native_syncpipe3_incptr_nostride / 1e6);

            ms_native_syncpipe3_incptr_nostride_kremain = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                        1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_incptr_nostride_kremain, flops / ms_native_syncpipe3_incptr_nostride_kremain / 1e6);

            ms_native_syncpipe3_incptr_nostride_bcol = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                     1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_incptr_nostride_bcol, flops / ms_native_syncpipe3_incptr_nostride_bcol / 1e6);

            ms_native_syncpipe3_incptr_kremain = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_kremain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                               1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_kremain = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_incptr_kremain, flops / ms_native_syncpipe3_incptr_kremain / 1e6);

            ms_native_dualbuf_syncpipe_incptr_nostride = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                       1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_dualbuf_syncpipe_incptr_nostride, flops / ms_native_dualbuf_syncpipe_incptr_nostride / 1e6);

            ms_native_syncpipe3_incptr_nostride_lb1 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                    1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_incptr_nostride_lb1, flops / ms_native_syncpipe3_incptr_nostride_lb1 / 1e6);

            ms_native_syncpipe3_finegrain = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_finegrain(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                          1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_finegrain = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_finegrain, flops / ms_native_syncpipe3_finegrain / 1e6);

            ms_native_syncpipe3_dualacc = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_dualacc(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                        1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_dualacc = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_dualacc, flops / ms_native_syncpipe3_dualacc / 1e6);

            ms_native_syncpipe3_snake = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                      1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_snake = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_snake, flops / ms_native_syncpipe3_snake / 1e6);

            ms_native_syncpipe3_bk8 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe3_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe3_bk8 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe3_bk8, flops / ms_native_syncpipe3_bk8 / 1e6);

            ms_native_syncpipe4_bk8 = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                    1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8, flops / ms_native_syncpipe4_bk8 / 1e6);

            ms_native_syncpipe4_bk8_warptile = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                             1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile, flops / ms_native_syncpipe4_bk8_warptile / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                      1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride, flops / ms_native_syncpipe4_bk8_warptile_nostride / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow / 1e6);

            ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline = time_call(start, stop, [&]() {
                cuda::experimental::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline / 1e6);

            // Now duplicates ms_native's own kernel (both are gemm_fp32_
            // tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact,
            // timed separately) -- kept as-is for the same reason
            // ms_native_syncpipe3 was left duplicating ms_native after
            // syncpipe3's own promotion: removing it would require
            // re-plumbing every "native_syncpipe4_bk8_warptile_nostride_
            // snake_midpoint_exact" ratio line below; the "-- vs cutlass"
            // line further down using this variable is now a same-
            // kernel rerun, useful only as a run-to-run noise check.
            ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = time_call(start, stop, [&]() {
                cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE,
                                                                            1.7, m, k, n, A_native, lda, B, ldb, 0.3, C, ldc);
            }, reps);
            std::printf("    gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact, flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact / 1e6);

            cutlass::gemm::GemmCoord problem_size(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            ElementAccumulator alpha = 1.7f, beta = 0.3f;
            Gemm gemm_op;
            typename Gemm::Arguments args(problem_size, {A_native, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
            size_t workspace_size = Gemm::get_workspace_size(args);
            void* workspace = nullptr;
            if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

            cutlass::Status status = gemm_op.can_implement(args);
            if (status == cutlass::Status::kSuccess) status = gemm_op.initialize(args, workspace);
            if (status == cutlass::Status::kSuccess) {
                ms_cutlass = time_call(start, stop, [&]() { gemm_op(); }, reps);
                std::printf("    cutlass (real, SIMT fp32)          = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass,
                           flops / ms_cutlass / 1e6);
            } else {
                std::printf("    cutlass SKIPPED (%s)\n", cutlass::cutlassGetStatusString(status));
            }
            if (workspace) cudaFree(workspace);

            ms_cublas = time_call(start, stop, [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), &alpha, A_native, lda, B, ldb, &beta, C, ldc);
            }, reps);
            std::printf("    cublas (Sgemm)                     = %9.3fms (%10.1f GFlop/s)\n", ms_cublas,
                       flops / ms_cublas / 1e6);
        }
        cudaFree(A_native);
    }

    if (ms_16dualbuf > 0 && ms_bf16dualbuf > 0 && ms_native > 0 && ms_cublas > 0) {
        std::printf("    -- vs cublas:  %.3fx (16dualbuf)  %.3fx (bf16dualbuf)  %.3fx (native)\n",
                   (flops / ms_16dualbuf) / (flops / ms_cublas), (flops / ms_bf16dualbuf) / (flops / ms_cublas),
                   (flops / ms_native) / (flops / ms_cublas));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (16dualbuf)  %.3fx (bf16dualbuf)  %.3fx (native)\n",
                       (flops / ms_16dualbuf) / (flops / ms_cutlass), (flops / ms_bf16dualbuf) / (flops / ms_cutlass),
                       (flops / ms_native) / (flops / ms_cutlass));
        }
        std::printf("    -- vs native:  %.3fx (16dualbuf)  %.3fx (bf16dualbuf)\n",
                   (flops / ms_16dualbuf) / (flops / ms_native), (flops / ms_bf16dualbuf) / (flops / ms_native));
    }
    if (ms_native > 0 && ms_native_regpipe > 0) {
        // Direct head-to-head: plain tile-level-double-buffered native
        // vs its register-level-software-pipelined sibling -- >1 means
        // regpipe is faster (the interleaving fix helped), <1 means it
        // regressed. See regpipe's own comment for the ncu/SASS
        // comparison against CUTLASS that motivated this attempt.
        std::printf("    -- native vs native_regpipe: %.3fx\n",
                   (flops / ms_native_regpipe) / (flops / ms_native));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_regpipe)\n",
                       (flops / ms_native_regpipe) / (flops / ms_cutlass));
        }
    }
    if (ms_native > 0 && ms_native_syncpipe > 0) {
        // Direct head-to-head: native vs its no-cp.async, CUTLASS-
        // MmaPipelined-modeled sibling -- >1 means syncpipe wins (the
        // real CUTLASS pipeline architecture helps), <1 means it
        // regressed. See syncpipe's own comment for exactly which real
        // CUTLASS source (mma_pipelined.h) this was modeled on.
        std::printf("    -- native vs native_syncpipe: %.3fx\n",
                   (flops / ms_native_syncpipe) / (flops / ms_native));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe)\n",
                       (flops / ms_native_syncpipe) / (flops / ms_cutlass));
        }
    }
    if (ms_native > 0 && ms_native_syncpipe3 > 0) {
        // Direct head-to-head: native vs the 3-stage (two-tile-
        // lookahead) extension of syncpipe -- >1 means the extra
        // pipeline depth helps further, <1 means it regressed.
        std::printf("    -- native vs native_syncpipe3: %.3fx\n",
                   (flops / ms_native_syncpipe3) / (flops / ms_native));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3)\n",
                       (flops / ms_native_syncpipe3) / (flops / ms_cutlass));
        }
        if (ms_native_syncpipe > 0) {
            // Direct head-to-head between the two syncpipe depths: >1
            // means the 3rd stage is worth it over the 2-stage version.
            std::printf("    -- syncpipe vs syncpipe3: %.3fx\n",
                       (flops / ms_native_syncpipe3) / (flops / ms_native_syncpipe));
        }
    }
    if (ms_native_syncpipe > 0 && ms_native_syncpipe_incptr > 0) {
        // Same rolling-pointer trick as syncpipe3_incptr, applied to
        // the simpler 2-stage syncpipe kernel instead -- tests whether
        // it avoids the register spilling syncpipe3_incptr hit.
        std::printf("    -- syncpipe vs syncpipe_incptr: %.3fx\n",
                   (flops / ms_native_syncpipe_incptr) / (flops / ms_native_syncpipe));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe_incptr)\n",
                       (flops / ms_native_syncpipe_incptr) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3 > 0 && ms_native_syncpipe3_incptr > 0) {
        // Direct head-to-head: syncpipe3's repeated-multiply address
        // computation vs its rolling-pointer sibling -- >1 means moving
        // address arithmetic off the FMA-shared IMAD path actually won
        // despite the register spilling it introduced (see incptr's own
        // comment for the ncu/SASS evidence and the spill tradeoff);
        // <1 means the spilling cost more than the IMAD reduction saved.
        std::printf("    -- syncpipe3 vs syncpipe3_incptr: %.3fx\n",
                   (flops / ms_native_syncpipe3_incptr) / (flops / ms_native_syncpipe3));
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr)\n",
                       (flops / ms_native_syncpipe3_incptr) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr > 0 && ms_native_syncpipe3_incptr_lean > 0) {
        // Direct head-to-head: does trimming the redundant live
        // register state (4 K-position counters -> 1 shared counter,
        // 4 tracked pointers -> 2) actually reduce the spilling enough
        // to matter? >1 means yes, it helped over the original incptr
        // port; <1 means the remaining spilling (56B stack frame, down
        // from 88B, but still nonzero) still costs more than it saves.
        std::printf("    -- syncpipe3_incptr vs syncpipe3_incptr_lean: %.3fx\n",
                   (flops / ms_native_syncpipe3_incptr_lean) / (flops / ms_native_syncpipe3_incptr));
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_lean: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_lean) / (flops / ms_native_syncpipe3));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_lean)\n",
                       (flops / ms_native_syncpipe3_incptr_lean) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_lean > 0 && ms_native_syncpipe3_incptr_lean2 > 0) {
        // Direct head-to-head: does combining predicate-collapsed
        // bounds checks + a 32-bit remaining-K counter (lean2) beat
        // lean's simpler trim? Static analysis showed lean2 actually
        // spills MORE than lean (64B/84B/108B stack/stores/loads vs
        // lean's 56B/60B/100B) -- this is the real measurement of
        // whether that static regression shows up at runtime too.
        std::printf("    -- syncpipe3_incptr_lean vs syncpipe3_incptr_lean2: %.3fx\n",
                   (flops / ms_native_syncpipe3_incptr_lean2) / (flops / ms_native_syncpipe3_incptr_lean));
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_lean2: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_lean2) / (flops / ms_native_syncpipe3));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_lean2)\n",
                       (flops / ms_native_syncpipe3_incptr_lean2) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_nostride > 0) {
        // Isolates option 3 alone (no precomputed stride locals --
        // inline the multiplications, let ptxas choose whether to
        // cache ld_a/ld_b or reload from cmem) against the other
        // incptr variants. Static analysis showed this is the BEST
        // spill result of the whole family (48B/48B/88B stack/stores/
        // loads, vs lean's 56B/60B/100B and lean2's 64B/84B/108B) --
        // this is the real measurement of whether that translates into
        // the best runtime result too.
        if (ms_native_syncpipe3_incptr > 0) {
            std::printf("    -- syncpipe3_incptr vs syncpipe3_incptr_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride) / (flops / ms_native_syncpipe3_incptr));
        }
        if (ms_native_syncpipe3_incptr_lean > 0) {
            std::printf("    -- syncpipe3_incptr_lean vs syncpipe3_incptr_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride) / (flops / ms_native_syncpipe3_incptr_lean));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride) / (flops / ms_native_syncpipe3));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_nostride)\n",
                       (flops / ms_native_syncpipe3_incptr_nostride) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_nostride > 0 && ms_native_syncpipe3_incptr_nostride_kremain > 0) {
        // Option 4 (decrementing k_remaining instead of tiles_fetched*BK)
        // added on top of nostride specifically. Static analysis showed
        // this is a mixed/slight regression vs nostride alone (same 48B
        // stack frame, but 68B spill stores vs 48B, 84B spill loads vs
        // 88B) -- this is the real measurement of whether removing that
        // one multiply helps or hurts at runtime.
        std::printf("    -- syncpipe3_incptr_nostride vs syncpipe3_incptr_nostride_kremain: %.3fx\n",
                   (flops / ms_native_syncpipe3_incptr_nostride_kremain) / (flops / ms_native_syncpipe3_incptr_nostride));
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_nostride_kremain: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_kremain) / (flops / ms_native_syncpipe3));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_nostride_kremain)\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_kremain) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_nostride > 0 && ms_native_syncpipe3_incptr_nostride_bcol > 0) {
        // Option 1 (deriving b_col[1] on demand instead of tracking it)
        // added on top of nostride specifically. Static analysis showed
        // this is a regression vs nostride alone on every spill metric
        // (56B stack frame vs 48B, 60B spill stores vs 48B, 100B spill
        // loads vs 88B) -- this is the real measurement of whether that
        // holds at runtime.
        std::printf("    -- syncpipe3_incptr_nostride vs syncpipe3_incptr_nostride_bcol: %.3fx\n",
                   (flops / ms_native_syncpipe3_incptr_nostride_bcol) / (flops / ms_native_syncpipe3_incptr_nostride));
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_nostride_bcol: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_bcol) / (flops / ms_native_syncpipe3));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_nostride_bcol)\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_bcol) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_kremain > 0) {
        // Option 4 (decrementing k_remaining instead of tiles_fetched*BK)
        // in ISOLATION -- applied on top of syncpipe3_incptr_lean's base
        // structure, NOT stacked on nostride like syncpipe3_incptr_
        // nostride_kremain was. Static analysis showed this alone
        // regresses vs lean on every spill metric (64B stack frame vs
        // 56B, 84B spill stores vs 60B, 108B spill loads vs 100B) --
        // in fact it lands on the exact same spill numbers as lean2
        // (predicates+narrow-ints combined), a coincidence worth noting.
        // This is the real measurement of option 4's own effect,
        // decoupled from option 3's.
        if (ms_native_syncpipe3_incptr_lean > 0) {
            std::printf("    -- syncpipe3_incptr_lean vs syncpipe3_incptr_kremain: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_kremain) / (flops / ms_native_syncpipe3_incptr_lean));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_kremain: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_kremain) / (flops / ms_native_syncpipe3));
        }
        if (ms_native_syncpipe3_incptr_nostride_kremain > 0) {
            std::printf("    -- syncpipe3_incptr_kremain vs syncpipe3_incptr_nostride_kremain (opt 4 alone vs opt 3+4): %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_kremain) / (flops / ms_native_syncpipe3_incptr_kremain));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_kremain)\n",
                       (flops / ms_native_syncpipe3_incptr_kremain) / (flops / ms_cutlass));
        }
    }
    if (ms_native_dualbuf_syncpipe_incptr_nostride > 0) {
        // syncpipe3_incptr_nostride's addressing (1 tracked pointer per
        // operand, 1 shared tiles_fetched counter, no precomputed
        // stride locals) applied to the SHALLOWER 2-stage dualbuf_
        // syncpipe pipeline instead of 3-stage syncpipe3 -- testing
        // whether less concurrent pipeline state (1 tile of lookahead
        // instead of 2) gets under the 128-register ceiling without
        // spilling. Static analysis showed it does NOT: 128 registers,
        // 48B stack frame, 52B spill stores, 88B spill loads --
        // essentially identical to (marginally worse than) 3-stage
        // nostride's 48B/48B/88B. This disproves the pipeline-depth
        // hypothesis: the addressing state itself is the register-
        // pressure bottleneck, independent of how many tiles are in
        // flight.
        if (ms_native_syncpipe_incptr > 0) {
            std::printf("    -- syncpipe_incptr vs dualbuf_syncpipe_incptr_nostride: %.3fx\n",
                       (flops / ms_native_dualbuf_syncpipe_incptr_nostride) / (flops / ms_native_syncpipe_incptr));
        }
        if (ms_native_syncpipe3_incptr_nostride > 0) {
            std::printf("    -- syncpipe3_incptr_nostride (3-stage) vs dualbuf_syncpipe_incptr_nostride (2-stage): %.3fx\n",
                       (flops / ms_native_dualbuf_syncpipe_incptr_nostride) / (flops / ms_native_syncpipe3_incptr_nostride));
        }
        if (ms_native > 0) {
            std::printf("    -- native (dualbuf, no syncpipe) vs dualbuf_syncpipe_incptr_nostride: %.3fx\n",
                       (flops / ms_native_dualbuf_syncpipe_incptr_nostride) / (flops / ms_native));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_dualbuf_syncpipe_incptr_nostride)\n",
                       (flops / ms_native_dualbuf_syncpipe_incptr_nostride) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_incptr_nostride_lb1 > 0) {
        // Same kernel as syncpipe3_incptr_nostride, EXCEPT
        // __launch_bounds__ minBlocksPerMultiprocessor: 1 instead of 2.
        // (256,2) hard-caps ptxas at exactly 128 registers/thread
        // (65536 register file / (256 threads * 2 blocks)); nostride's
        // peak live state sits just over that line, hence its spilling.
        // (256,1) gives ptxas up to 255 registers/thread instead --
        // static analysis confirms this ELIMINATES spilling entirely:
        // 146 registers, 0 bytes stack frame, 0 spill stores, 0 spill
        // loads (matching CUTLASS's own 0/0/0). The cost is halving max
        // resident blocks/SM from 2 to 1 -- every kernel in this family
        // was occupancy-tied with CUTLASS at ~24.7% achieved occupancy
        // AT the 2-block cap, so this trades that away. This is the
        // real measurement of whether removing the spill is worth the
        // occupancy it costs.
        if (ms_native_syncpipe3_incptr_nostride > 0) {
            std::printf("    -- syncpipe3_incptr_nostride vs syncpipe3_incptr_nostride_lb1 (2-block vs 1-block/SM): %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_lb1) / (flops / ms_native_syncpipe3_incptr_nostride));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_incptr_nostride_lb1: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_lb1) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe3_incptr_nostride_lb1: %.3fx\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_lb1) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_incptr_nostride_lb1)\n",
                       (flops / ms_native_syncpipe3_incptr_nostride_lb1) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_finegrain > 0) {
        // Applies to syncpipe3 the two structural traits confirmed from
        // CUTLASS's own MmaPipelined::gemm_iters and its MmaSimtPolicy
        // GemmShape<4,4,1> lane shape: (1) the 8x8 accumulator tile
        // split into four 4x4 quadrants, each interleaved with one of
        // next_slot's four float4 register loads instead of loading all
        // four upfront then running one monolithic 64-FFMA nest; (2)
        // A's and B's next-next-tile global loads issued at different
        // kk indices (kk==0, kk==BK/2) instead of bunched together at
        // kk==0. No addressing/register-pressure change versus
        // syncpipe3 otherwise. Static analysis found ptxas mostly
        // undid the source-level interleaving anyway (re-clustered the
        // loads together, ~30-FFMA bursts resumed, reuse ratio barely
        // moved: 69.3% vs syncpipe3's 68.5%, both far off CUTLASS's
        // 87.2%) and introduced a small new spill (24B/24B/24B) that
        // syncpipe3 didn't have -- the same lesson as the regpipe
        // reorder attempt: ptxas's own scheduling heuristics don't
        // reliably follow source-level restructuring. This is the real
        // runtime measurement of whether it helped anyway.
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_finegrain: %.3fx\n",
                       (flops / ms_native_syncpipe3_finegrain) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe3_finegrain: %.3fx\n",
                       (flops / ms_native_syncpipe3_finegrain) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_finegrain)\n",
                       (flops / ms_native_syncpipe3_finegrain) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_dualacc > 0) {
        // Byte-for-byte identical to syncpipe3 EXCEPT the accumulator:
        // each output element's sum is split into TWO independent
        // partial accumulators (alternating by kk parity, combined only
        // in the epilogue), breaking the RAW dependency chain across
        // kGemmFp32DualBK=16 kk-iterations into two 8-deep chains
        // instead of one 16-deep chain. Motivated by matched-occupancy
        // ncu data: syncpipe3's "wait" (fixed-latency execution
        // dependency) stall share is 15.4% of total vs CUTLASS's 11.6%,
        // while memory-latency categories (short/long_scoreboard) are
        // under 1% for both, ruling those out. Caveat found before
        // building: the wait gap is real but modest (CUTLASS is also
        // diffusely better on math_pipe_throttle/not_selected and
        // diffusely WORSE on barrier) -- not expected to close most of
        // the remaining gap alone. Confirmed cost: doubling accumulator
        // registers (64 -> 128 floats) produced severe spilling -- 680B
        // stack frame, 3672B spill stores, 3584B spill loads, by far
        // the worst of this whole investigation (previous worst was
        // incptr's 88B/176B/120B). This is the real measurement of
        // whether breaking the dependency chain was worth that cost.
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_dualacc: %.3fx\n",
                       (flops / ms_native_syncpipe3_dualacc) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe3_dualacc: %.3fx\n",
                       (flops / ms_native_syncpipe3_dualacc) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_dualacc)\n",
                       (flops / ms_native_syncpipe3_dualacc) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_snake > 0) {
        // Byte-for-byte identical to syncpipe3 EXCEPT the FFMA
        // accumulate loop's traversal order: a snake (boustrophedon)
        // sweep (even rows sweep j forward, odd rows sweep it backward)
        // instead of plain row-major, so consecutive rows connect at
        // the same j index instead of wrapping from j=TN-1 back to
        // j=0. Targets the measured FFMA .reuse ratio gap (syncpipe3:
        // 68.5%, CUTLASS: 87.2%) without touching addressing, load
        // structure, tile size, or accumulator structure. Static
        // analysis found this did NOT move the needle: reuse ratio
        // came in at 69.3%, essentially unchanged from syncpipe3's own
        // 68.5% -- ptxas apparently didn't honor the intended row-
        // boundary reuse opportunity -- while introducing a real spill
        // (120B stack frame, 208B spill stores, 136B spill loads) that
        // syncpipe3 didn't have. This is the real runtime measurement
        // of whether it helped anyway.
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_snake: %.3fx\n",
                       (flops / ms_native_syncpipe3_snake) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe3_snake: %.3fx\n",
                       (flops / ms_native_syncpipe3_snake) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_snake)\n",
                       (flops / ms_native_syncpipe3_snake) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe3_bk8 > 0) {
        // Same 3-stage rotation, TM=TN=8 register tile, and warp/lane
        // geometry as syncpipe3 -- the only change is the K-tile depth:
        // BK=8 instead of 16, matching CUTLASS's own confirmed
        // ThreadblockShape<128,128,8>. Every other kernel in this whole
        // investigation used the shared BK=16 constant; this is the
        // first one to actually match CUTLASS's own K-tile depth.
        // Requires a genuinely different thread-to-element mapping (one
        // float4 load per thread instead of two), not just a constant
        // substitution -- confirmed by dividing tile size by
        // (NumThreads*4) before writing any code. Static analysis: 128
        // registers, 0 bytes stack frame, 0 spill stores, 0 spill loads
        // -- matches syncpipe3's own clean profile exactly, no
        // register-pressure regression. This is the real measurement of
        // whether the shorter dependency chain / more frequent sync
        // tradeoff helps.
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe3_bk8: %.3fx\n",
                       (flops / ms_native_syncpipe3_bk8) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe3_bk8: %.3fx\n",
                       (flops / ms_native_syncpipe3_bk8) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe3_bk8)\n",
                       (flops / ms_native_syncpipe3_bk8) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8 > 0) {
        // Byte-for-byte identical to syncpipe3_bk8 EXCEPT the pipeline
        // depth: 4 rotating shared-memory stages instead of 3 (three
        // tiles of lookahead instead of two). Affordable because BK=8
        // halves each stage's cost versus BK=16, freeing up shared-
        // memory budget -- mirrors exactly how syncpipe (2-stage) was
        // extended to syncpipe3 (3-stage) earlier this session, the one
        // technique in this whole investigation with a confirmed
        // positive measured result, now applied to the bk8 base.
        // Targets bk8's one identified downside versus syncpipe3
        // (doubled sync/tile-transition frequency from the smaller BK).
        // Static analysis: 128 registers, 0 bytes stack frame, 0 spill
        // stores, 0 spill loads -- matches bk8's own clean profile, no
        // register-pressure regression from the deeper pipeline.
        if (ms_native_syncpipe3_bk8 > 0) {
            std::printf("    -- syncpipe3_bk8 vs syncpipe4_bk8 (3-stage vs 4-stage): %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8) / (flops / ms_native_syncpipe3_bk8));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8)\n",
                       (flops / ms_native_syncpipe4_bk8) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile > 0) {
        // Byte-for-byte identical to syncpipe4_bk8 EXCEPT the warp/lane
        // geometry used for the compute read path: CUTLASS's actual
        // confirmed WarpShape<32,64,8> (32 rows x 64 cols per warp, vs
        // this file's own 64x32) and its MmaSimtPolicy<MatrixShape<4,8>,
        // RowMajorInterleaved<2>,...> lane layout, including the exact
        // interleaved lane-to-(row,col) mapping read directly from
        // CUTLASS's own cutlass/layout/matrix.h (RowMajorInterleaved::
        // inverse()), not a plain divide/modulo split. The tile-fetch
        // path is untouched (addresses by flat thread id, not warp/lane
        // geometry). The biggest, most structural change attempted in
        // this whole investigation -- every other kernel kept the warp/
        // lane grid fixed. Static analysis: 128 registers, 0 bytes
        // stack frame, 0 spill stores, 0 spill loads -- no register-
        // pressure regression from the geometry change.
        if (ms_native_syncpipe4_bk8 > 0) {
            std::printf("    -- syncpipe4_bk8 vs syncpipe4_bk8_warptile: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile) / (flops / ms_native_syncpipe4_bk8));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride > 0) {
        // Byte-for-byte identical to syncpipe4_bk8_warptile EXCEPT the
        // global-load addressing: applies the syncpipe3_incptr_nostride
        // technique (one tracked rolling pointer per operand, no
        // precomputed stride locals) instead of recomputing &A[a_row0 +
        // (k0+a_kk)*ld_a] from scratch every call. Genuinely untried
        // combination -- the incptr/nostride address-reduction family
        // was only ever tried on the BK=16 base (and lost there); BK=8
        // was only ever tried with pipeline-depth and geometry changes,
        // never addressing. Static analysis: 126 registers, 0 bytes
        // stack frame, 0 spill stores, 0 spill loads -- the FIRST
        // kernel in this whole investigation to compile to fewer than
        // 128 registers; every other kernel either hit the 128-register
        // ceiling exactly or spilled past it. This naturally lands at
        // EXACTLY CUTLASS's own register count (126), with zero
        // spilling. This is the real measurement of whether that
        // translates into a runtime win.
        if (ms_native_syncpipe4_bk8_warptile > 0) {
            std::printf("    -- syncpipe4_bk8_warptile vs syncpipe4_bk8_warptile_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride) / (flops / ms_native_syncpipe4_bk8_warptile));
        }
        if (ms_native_syncpipe4_bk8 > 0) {
            std::printf("    -- syncpipe4_bk8 vs syncpipe4_bk8_warptile_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride) / (flops / ms_native_syncpipe4_bk8));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
        // Byte-for-byte identical to syncpipe4_bk8_warptile_nostride
        // EXCEPT the FFMA accumulate loop's traversal order: the same
        // snake (boustrophedon) sweep syncpipe3_snake used, retried
        // here instead of on syncpipe3. Motivation: snake failed on
        // syncpipe3 (introduced a new spill on a kernel already sitting
        // exactly at the 128-register ceiling, and the reuse ratio
        // didn't even move -- 69.3% vs syncpipe3's own 68.5%).
        // wtnostride naturally compiles to 126 registers -- 2 below the
        // ceiling -- and its own reuse ratio is 67.2%, still far from
        // CUTLASS's 87.2%. Static analysis this time: 127 registers (1
        // more than wtnostride, still 1 below the ceiling), 0 bytes
        // stack frame, 0 spill stores, 0 spill loads -- no spilling
        // this time, unlike the syncpipe3 attempt. And the reuse ratio
        // actually moved: 73.4%, a real +6.2 percentage point gain over
        // wtnostride's 67.2%, versus the negligible +0.8pp snake
        // achieved on syncpipe3 before. This is the real runtime
        // measurement of whether that reuse gain translates into a win.
        if (ms_native_syncpipe4_bk8_warptile_nostride > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride vs syncpipe4_bk8_warptile_nostride_snake: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake) / (flops / ms_native_syncpipe4_bk8_warptile_nostride));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved > 0) {
        // Identical geometry and snake traversal to
        // syncpipe4_bk8_warptile_nostride_snake, but the next-kk shared-
        // memory prefetch and next-tile global prefetch are issued
        // piecemeal, interleaved into the FFMA nest, instead of batched
        // up front -- mirroring the instruction interleaving pattern
        // found by disassembling CUTLASS's own compiled SASS mainloop
        // (same alternating-operand snake reuse pattern we already use,
        // but with LDS/LDG/address-math spread roughly one every 1-2
        // FFMAs through the chain instead of clustered at the front).
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_interleaved: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_interleaved: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_interleaved: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_interleaved)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late > 0) {
        // Same as syncpipe4_bk8_warptile_nostride_snake_interleaved, but
        // the next-kk/next-tile prefetch checkpoints inside the row loop
        // are placed late (i==5/6/7) instead of early (i==2/4/6), to
        // shrink how long the prefetched registers must stay live before
        // their real consumer (the next kk iteration) -- the interleaved
        // kernel hit exactly the 128-register ceiling with light
        // spilling (2 STL + 2 LDL); this checks whether reducing that
        // live range avoids the spill while keeping the reuse-ratio gain.
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake_interleaved vs syncpipe4_bk8_warptile_nostride_snake_interleaved_late: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved));
        }
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_interleaved_late: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_interleaved_late: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_interleaved_late: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_interleaved_late) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint > 0) {
        // Isolates whether the interleaved kernels' register/spill
        // increase came from splitting load_compute_regs into separate
        // a/b lambdas, or from moving the checkpoint into the row loop
        // at all -- interleaved and interleaved_late compiled to
        // byte-identical SASS, so source position was already shown to
        // be ignored for that split. This kernel keeps load_compute_regs
        // combined (one call site, as in plain snake) and moves only
        // that single call to one mid-loop checkpoint (i==4) instead of
        // before the FFMA block, with no split.
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_midpoint: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_midpoint: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_midpoint: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_midpoint)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped > 0) {
        // Same as syncpipe4_bk8_warptile_nostride_snake_midpoint, but
        // __launch_bounds__(256, 2) is relaxed to __launch_bounds__(256)
        // (no min-blocks term), letting ptxas pick registers freely --
        // matching how CUTLASS's own SIMT fp32 device::Gemm kernel is
        // compiled (no __launch_bounds__ at all; its 126-register
        // footprint is organic, not a forced ceiling). Tests whether the
        // (256, 2) ceiling itself, not the prefetch loop structure, was
        // the actual cause of the light spill seen in midpoint/
        // interleaved/interleaved_late (all of which converged to the
        // same 128-register/2 STL + 2 LDL compile under (256, 2)).
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake_midpoint vs syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint));
        }
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow > 0) {
        // Same as syncpipe4_bk8_warptile_nostride_snake_midpoint (same
        // combined checkpoint at i==4, same __launch_bounds__(256, 2) --
        // deliberately not relaxed, since the uncapped variant's
        // relaxation dropped occupancy from 2 blocks/SM to 1 and
        // measured dramatically slower despite eliminating the spill)
        // EXCEPT tiles_fetched is narrowed from size_t to unsigned int
        // to try to free the one register midpoint needed without
        // touching the interleaved-checkpoint structure at all.
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake_midpoint vs syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint));
        }
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline > 0) {
        // Same as syncpipe4_bk8_warptile_nostride_snake_midpoint (same
        // combined checkpoint at i==4, same __launch_bounds__(256, 2))
        // EXCEPT has_next_kk/has_next_tile are no longer precomputed
        // named locals -- the conditions are inlined directly at the
        // i==4 checkpoint, removing the one genuinely-live-longer
        // boolean (has_next_tile, only real for the kk==0 unrolled
        // instance) that midpoint's own manual hoisting introduced.
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake_midpoint vs syncpipe4_bk8_warptile_nostride_snake_midpoint_inline: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint));
        }
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_midpoint_inline: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_midpoint_inline: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_midpoint_inline: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline) / (flops / ms_cutlass));
        }
    }
    if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0) {
        // Isolates whether midpoint's own slowdown vs plain snake is
        // caused by its light spill (2 STL + 2 LDL at 128 registers), or
        // by something else about the interleaved checkpoint.
        // load_global_advance -- called right at midpoint's i==4
        // checkpoint -- normally bounds-checks every tile fetch even
        // though for an exact-multiple-of-tile-size problem that check
        // is always true; this kernel drops it entirely right at the
        // pressure point. Only dispatched when m/n are exact multiples
        // of 128 and k is an exact multiple of 8 (true for every size
        // here except N=54000, which falls back to the safe midpoint
        // kernel).
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake_midpoint vs syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint));
        }
        if (ms_native_syncpipe4_bk8_warptile_nostride_snake > 0) {
            std::printf("    -- syncpipe4_bk8_warptile_nostride_snake vs syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake));
        }
        if (ms_native_syncpipe3 > 0) {
            std::printf("    -- syncpipe3 vs syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_native_syncpipe3));
        }
        if (ms_cublas > 0) {
            std::printf("    -- cublas vs syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_cublas));
        }
        if (ms_cutlass > 0) {
            std::printf("    -- vs cutlass: %.3fx (native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                       (flops / ms_native_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_cutlass));
        }
    }
    if (ms_16dualbuf > 0 && ms_16dualbuf_trunc > 0) {
        // Direct head-to-head: rounding-aware decode (fp32_16_combine)
        // vs the never-rounded, no-correction-at-all decode (fp32_16_
        // combine_truncated) -- >1 means the rounded/round_flag-aware
        // kernel is faster than the truncated one at this shape, <1
        // means the reverse.
        std::printf("    -- 16dualbuf vs 16dualbuf_truncated: %.3fx\n",
                   (flops / ms_16dualbuf) / (flops / ms_16dualbuf_trunc));
    }
    if (ms_16dualbuf > 0 && ms_16dualbuf_adaptive > 0) {
        // >1 means the always-full-decode kernel is faster at this shape
        // (the adaptive kernel's extra per-tile __syncthreads_or barrier
        // costs more than skipping the dispatch saves); <1 means the
        // adaptive kernel wins.
        std::printf("    -- 16dualbuf vs 16dualbuf_adaptive: %.3fx\n",
                   (flops / ms_16dualbuf) / (flops / ms_16dualbuf_adaptive));
    }
    if (ms_16dualbuf > 0 && ms_16dualbuf_adaptive_warp > 0) {
        // >1 means the always-full-decode kernel is still faster; <1
        // means the per-warp adaptive split actually wins.
        std::printf("    -- 16dualbuf vs 16dualbuf_adaptive_warp: %.3fx\n",
                   (flops / ms_16dualbuf) / (flops / ms_16dualbuf_adaptive_warp));
    }
    if (ms_16dualbuf_adaptive > 0 && ms_16dualbuf_adaptive_warp > 0) {
        // Direct head-to-head between the two adaptive strategies: >1
        // means per-warp voting beats the block-wide barrier.
        std::printf("    -- 16dualbuf_adaptive vs 16dualbuf_adaptive_warp: %.3fx\n",
                   (flops / ms_16dualbuf_adaptive_warp) / (flops / ms_16dualbuf_adaptive));
    }
    if (ms_16dualbuf > 0 && ms_16dualbuf_multin > 0) {
        // >1 means the always-full-decode single-N-tile kernel is still
        // faster; <1 means halving decode redundancy via shared-memory
        // reuse across 2 N-tiles actually wins.
        std::printf("    -- 16dualbuf vs 16dualbuf_multin: %.3fx\n",
                   (flops / ms_16dualbuf) / (flops / ms_16dualbuf_multin));
    }
    if (ms_16dualbuf > 0 && ms_16gemm_materialize > 0) {
        // >1 means fused decode is still faster; <1 means eliminating
        // the redundancy entirely (materialize once, run native GEMM)
        // wins.
        std::printf("    -- 16dualbuf vs 16gemm_materialize: %.3fx\n",
                   (flops / ms_16dualbuf) / (flops / ms_16gemm_materialize));
    }
    if (ms_16gemm_materialize > 0 && ms_native > 0) {
        // How close the materialize strategy gets to the native fp32
        // GEMM ceiling -- since materialize literally calls that same
        // kernel, this ratio isolates the decode/chunking overhead.
        std::printf("    -- 16gemm_materialize vs native: %.3fx\n",
                   (flops / ms_16gemm_materialize) / (flops / ms_native));
    }
    if (ms_16gemm_materialize > 0 && ms_16gemm_materialize_midpoint_exact > 0) {
        // Direct head-to-head between the two materialize variants --
        // isolates the effect of swapping the second-stage native GEMM
        // (gemm_fp32_tiled_cpasync_dualbuf vs midpoint_exact) with the
        // decode pass held identical.
        std::printf("    -- 16gemm_materialize vs 16gemm_materialize_midpoint_exact: %.3fx\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_16gemm_materialize));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_native > 0) {
        // How close this materialize variant gets to the native fp32
        // GEMM ceiling (now midpoint_exact itself, since "native" was
        // promoted to it) -- isolates the decode/chunking overhead.
        std::printf("    -- 16gemm_materialize_midpoint_exact vs native: %.3fx\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_native));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16gemm_materialize_midpoint_exact)\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_cutlass));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_cublas > 0) {
        std::printf("    -- vs cublas: %.3fx (16gemm_materialize_midpoint_exact)\n",
                   (flops / ms_16gemm_materialize_midpoint_exact) / (flops / ms_cublas));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_16gemm_materialize_midpoint_exact_vec > 0) {
        // The actual question this timing exists to answer: does
        // vectorizing the chunk-decode pass (4 rows/thread via ushort4/
        // float4 instead of 1 scalar element/thread) measurably speed
        // up materialize's total time, given decode is the entire extra
        // cost it pays over plain native fp32?
        std::printf("    -- 16gemm_materialize_midpoint_exact vs its vectorized-decode sibling: %.3fx\n",
                   (flops / ms_16gemm_materialize_midpoint_exact_vec) / (flops / ms_16gemm_materialize_midpoint_exact));
    }
    if (ms_16gemm_materialize_midpoint_exact_vec > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16gemm_materialize_midpoint_exact_vec)\n",
                   (flops / ms_16gemm_materialize_midpoint_exact_vec) / (flops / ms_cutlass));
    }
    if (ms_16gemm_materialize_midpoint_exact > 0 && ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0) {
        // Direct head-to-head between the materialize-and-call-native
        // strategy and the genuinely fused kernel -- isolates whether
        // fusing wh/wr decode straight into the tile pipeline (no
        // separate scratch buffer or elementwise decode kernel, but
        // extra ushort4-load/decode instructions inside a register-
        // pressure-sensitive mainloop) beats paying extra global-memory
        // bandwidth for a wholly separate decode pass that keeps the
        // GEMM's own inner loop completely free of decode logic.
        std::printf("    -- 16gemm_materialize_midpoint_exact vs 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_16gemm_materialize_midpoint_exact));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_native > 0) {
        // How close the fused kernel gets to the native fp32 GEMM
        // ceiling -- isolates the pure decode-fusion overhead (no
        // separate scratch/chunking cost the materialize variant pays).
        std::printf("    -- 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact vs native: %.3fx\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_native));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_cutlass));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_cublas > 0) {
        std::printf("    -- vs cublas: %.3fx (16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                   (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact) / (flops / ms_cublas));
    }
    if (ms_16dualbuf_trunc > 0 && ms_16fused_midpoint_exact_trunc > 0) {
        // Direct head-to-head between the two truncated-decode kernels
        // -- isolates whether the fused midpoint_exact-based kernel's
        // real spill (26 STL + 17 LDL, vs 0 spill for the plain-decode
        // fused kernel) actually costs performance here.
        std::printf("    -- 16dualbuf_truncated vs 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated: %.3fx\n",
                   (flops / ms_16fused_midpoint_exact_trunc) / (flops / ms_16dualbuf_trunc));
    }
    if (ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact > 0 && ms_16fused_midpoint_exact_trunc > 0) {
        // Isolates the cost of the truncated decode's spill relative to
        // the plain-decode fused kernel, with tiling held identical.
        std::printf("    -- 16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact vs its truncated sibling: %.3fx\n",
                   (flops / ms_16fused_midpoint_exact_trunc) / (flops / ms_16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact));
    }
    if (ms_16fused_midpoint_exact_trunc > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated)\n",
                   (flops / ms_16fused_midpoint_exact_trunc) / (flops / ms_cutlass));
    }
    if (ms_bf16dualbuf > 0 && ms_bf16fused_midpoint_exact > 0) {
        // Direct head-to-head between the original bf16 fused kernel
        // and its midpoint_exact-based counterpart -- isolates whether
        // midpoint_exact's tiling improvements (CUTLASS geometry, BK=8,
        // 4-stage pipeline, snake traversal, interleaved checkpoint)
        // carry over to fp32_bf16 the same way they did for fp32_16
        // (where the fused kernel beat dualbuf by ~3%).
        std::printf("    -- bf16dualbuf vs bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact: %.3fx\n",
                   (flops / ms_bf16fused_midpoint_exact) / (flops / ms_bf16dualbuf));
    }
    if (ms_bf16fused_midpoint_exact > 0 && ms_native > 0) {
        std::printf("    -- bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact vs native: %.3fx\n",
                   (flops / ms_bf16fused_midpoint_exact) / (flops / ms_native));
    }
    if (ms_bf16fused_midpoint_exact > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                   (flops / ms_bf16fused_midpoint_exact) / (flops / ms_cutlass));
    }
    if (ms_bf16fused_midpoint_exact > 0 && ms_cublas > 0) {
        std::printf("    -- vs cublas: %.3fx (bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact)\n",
                   (flops / ms_bf16fused_midpoint_exact) / (flops / ms_cublas));
    }
    if (ms_bf16fused_midpoint_exact > 0 && ms_bf16fused_rounded > 0) {
        // The actual question this timing exists to answer: does the
        // rounded-decode kernel's real spill (23 STL + 11 LDL) cost it
        // the win against the plain, spill-free fused kernel above?
        std::printf("    -- bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact vs its rounded sibling: %.3fx\n",
                   (flops / ms_bf16fused_rounded) / (flops / ms_bf16fused_midpoint_exact));
    }
    if (ms_bf16fused_rounded > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded)\n",
                   (flops / ms_bf16fused_rounded) / (flops / ms_cutlass));
    }
    if (ms_bf16fused_rounded > 0 && ms_bf16materialize_rounded > 0) {
        // The other half of the question: does sidestepping the fused
        // kernel's spill via materialize (decode once, then the already-
        // proven 0-spill native GEMM as a black box) actually win, the
        // same way materialize beat fused for fp32_16?
        std::printf("    -- bf16fused_rounded vs bf16materialize_rounded: %.3fx\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_bf16fused_rounded));
    }
    if (ms_bf16materialize_rounded > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16materialize_rounded)\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_cutlass));
    }
    if (ms_bf16fused_rounded > 0 && ms_bf16fused_rounded_adaptive > 0) {
        // The actual question this timing exists to answer: does
        // trading a real extra per-tile barrier for a reduced (not
        // eliminated) spill -- REG:128 STACK:16 vs STACK:56 -- net out
        // as a win against the plain fused rounded kernel above?
        std::printf("    -- bf16fused_rounded vs bf16fused_rounded_adaptive: %.3fx\n",
                   (flops / ms_bf16fused_rounded_adaptive) / (flops / ms_bf16fused_rounded));
    }
    if (ms_bf16fused_rounded_adaptive > 0 && ms_bf16materialize_rounded > 0) {
        std::printf("    -- bf16fused_rounded_adaptive vs bf16materialize_rounded: %.3fx\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_bf16fused_rounded_adaptive));
    }
    if (ms_bf16fused_rounded_adaptive > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16fused_rounded_adaptive)\n",
                   (flops / ms_bf16fused_rounded_adaptive) / (flops / ms_cutlass));
    }
    if (ms_bf16fused_rounded > 0 && ms_bf16fused_rounded_adaptive_warp > 0) {
        // The other half of the question: does the warp-level variant
        // (0 new barrier-class instructions, VOTE.ANY only, but a
        // smaller spill reduction than the block-wide version) beat the
        // plain fused rounded kernel where the block-wide version's
        // real extra barriers might not?
        std::printf("    -- bf16fused_rounded vs bf16fused_rounded_adaptive_warp: %.3fx\n",
                   (flops / ms_bf16fused_rounded_adaptive_warp) / (flops / ms_bf16fused_rounded));
    }
    if (ms_bf16fused_rounded_adaptive > 0 && ms_bf16fused_rounded_adaptive_warp > 0) {
        std::printf("    -- bf16fused_rounded_adaptive vs bf16fused_rounded_adaptive_warp: %.3fx\n",
                   (flops / ms_bf16fused_rounded_adaptive_warp) / (flops / ms_bf16fused_rounded_adaptive));
    }
    if (ms_bf16fused_rounded_adaptive_warp > 0 && ms_bf16materialize_rounded > 0) {
        std::printf("    -- bf16fused_rounded_adaptive_warp vs bf16materialize_rounded: %.3fx\n",
                   (flops / ms_bf16materialize_rounded) / (flops / ms_bf16fused_rounded_adaptive_warp));
    }
    if (ms_bf16fused_rounded_adaptive_warp > 0 && ms_cutlass > 0) {
        std::printf("    -- vs cutlass: %.3fx (bf16fused_rounded_adaptive_warp)\n",
                   (flops / ms_bf16fused_rounded_adaptive_warp) / (flops / ms_cutlass));
    }

    cudaFree(B);
    cudaFree(C);
}

int main(int argc, char** argv) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== mm_fp32_16_tiled_cpasync_dualbuf / mm_fp32_bf16_tiled_cpasync_dualbuf / "
               "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native) vs real CUTLASS (fp32 SIMT) vs cuBLAS Sgemm ===\n");

    // --subnormal-density <PCT>: scans it out of the argument list up front
    // (may appear anywhere) so the existing --rect/N shape-selection logic
    // below is otherwise untouched -- it just sees whatever args remain.
    // Applies to every generator in every group (see run_shape's own
    // comment); 0 (the default, when this flag is absent) reproduces every
    // existing invocation's behavior exactly.
    std::vector<std::string> args(argv + 1, argv + argc);
    float density = 0.0f;
    for (size_t i = 0; i < args.size();) {
        if (args[i] == "--subnormal-density" && i + 1 < args.size()) {
            float pct = std::strtof(args[i + 1].c_str(), nullptr);
            density = pct / 100.0f;
            args.erase(args.begin() + static_cast<long>(i), args.begin() + static_cast<long>(i) + 2);
            std::printf("=== A/B subnormal density: %.2f%% (fraction %.4f) ===\n", pct, density);
        } else {
            ++i;
        }
    }

    // Rectangular sweep: output held fixed at M=N=44000 by default, inner
    // dimension K swept across powers of two from 2^10 to 2^15. Peak
    // footprint at the largest K here (32768) is still B+C (~13.5GB) plus
    // at most one transient m*k-sized buffer (~5.8GB) -- well under a
    // 40GB A100's ceiling, so every K in this sweep is expected to
    // produce real numbers, not skip.
    const std::vector<size_t> rect_ks = {1024, 2048, 4096, 8192, 16384, 32768};
    const size_t kDefaultRectOuter = 44000;

    // Below: identical logic to before --subnormal-density was scanned out,
    // just reading from the filtered `args` vector (args[i]) in place of
    // (argv[i+1]/argc-1) -- same positions, since the flag has already
    // been erased above regardless of where it appeared.
    if (!args.empty() && args[0] == "--rect") {
        // Optional 3rd arg overrides the M=N outer dimension (default
        // 44000, preserving every existing caller's behavior unchanged).
        size_t outer = (args.size() > 2) ? static_cast<size_t>(std::strtoull(args[2].c_str(), nullptr, 10)) : kDefaultRectOuter;
        if (args.size() > 1) {
            size_t k = static_cast<size_t>(std::strtoull(args[1].c_str(), nullptr, 10));
            run_shape(outer, k, outer, start, stop, handle, density);
        } else {
            for (size_t k : rect_ks) run_shape(outer, k, outer, start, stop, handle, density);
        }
        cublasDestroy(handle);
        return 0;
    }

    if (!args.empty()) {
        size_t n = static_cast<size_t>(std::strtoull(args[0].c_str(), nullptr, 10));
        run_shape(n, n, n, start, stop, handle, density);
        cublasDestroy(handle);
        return 0;
    }

    // 48000 and 54000 fit within run_shape's peak footprint (12 bytes/
    // element -- B+C always live, plus whichever single transient
    // representation -- wh16+wr16, wh_bf16+wr_bf16, or A_native -- is
    // currently allocated, never more than one at a time) on a 40GB
    // A100, producing real throughput numbers there. On a smaller card
    // (or one already under memory pressure), run_shape's own
    // cudaMalloc checks degrade gracefully, printing "SKIPPED (out of
    // device memory ...)" per method instead of aborting the whole run.
    for (size_t n : {1024, 2048, 4096, 8192, 16384, 32768, 48000, 54000}) {
        run_shape(n, n, n, start, stop, handle, density);
    }

    for (size_t k : rect_ks) run_shape(44000, k, 44000, start, stop, handle, density);

    cublasDestroy(handle);
    return 0;
}

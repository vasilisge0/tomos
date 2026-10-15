// "_advanced" variant of mv_fp32_transpose_recon_vs_cublas_ref.cu --
// byte-for-byte identical except for time_call's own timing methodology
// and two new CLI flags, --warmup N and --iters N (both default 10, see
// main()'s own argv-filtering preamble): instead of this file's own
// sibling's fixed, shape-size-dependent `reps` heuristic (a single
// un-timed warmup call, then `reps` timed calls averaged), this variant
// always does g_warmup_runs UN-TIMED warmup calls first, then times g_
// iterations calls as one block and divides by g_iterations -- both
// counts fully caller-controlled, independent of shape. No kernel call
// site, shape list, or dispatch logic differs from the sibling file.
//
// Standalone throughput comparison for the TRANSPOSED "full fp32
// reconstruction" tier of GEMV (y = alpha*A^T*x + beta*y): the SAME six
// series mv_fp32_recon_vs_cutlass_ref.cu compares for NO_TRANSPOSE --
// cuBLAS (Sgemv, CUBLAS_OP_T), TOMOS FP32 (native, mv_fp32_tiled), TOMOS
// FP32/FP16 (mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock), TOMOS
// FP32/FP16 (fast) (mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_
// fast_normal), TOMOS FP32/BF16 (mv_fp32_bf16_accurate_ready_atomiconly_
// scaled_kblock_rounded), TOMOS FP32/BF16 (truncated) (mv_fp32_bf16_
// accurate_ready_atomiconly_scaled_kblock_workspace) -- just with
// trans=TransposeType::TRANSPOSE for every call instead of NO_TRANSPOSE.
//
// No CUTLASS leg here (unlike the NO_TRANSPOSE file): this is a narrower,
// TOMOS-vs-cuBLAS-only comparison by request, not an oversight -- see this
// file's own name (_vs_cublas_ref, not _vs_cutlass_ref).
//
// Why this file exists at all, not just a --transpose flag on the
// existing one: every "fast"/"ready"/"atomiconly" row-vectorized TOMOS
// GEMV kernel in this codebase, before now, explicitly excluded TRANSPOSE
// and fell back to a generic double-accumulator dispatch whose own
// vectorized-load fast path is ALSO NO_TRANSPOSE-only -- there was no
// transpose-specialized GEMV kernel of any kind. Each of the five
// dispatcher functions above was extended with a genuine TRANSPOSE fast
// path (trans==TRANSPOSE && ld_a==size[0]) alongside its existing
// NO_TRANSPOSE one: warp-per-output-column, the reduction dimension (m)
// split into chunks across blockIdx.x with atomicAdd combining partial
// sums, x staged into shared memory once per block and reused by every
// warp in that block -- ported from randls2.0's own sgemv_cm_transpose_
// splitM_atomic_xshared (src/kernels/cuda/matrix/dense_kernels.cu, that
// sibling project -- same author, no separate license/attribution
// needed), adapted to this codebase's own conventions. See mv_fp32_
// transpose_atomiconly_scaled_mblock_kernel's own comment (mv_cuda_
// kernels.cu) for the full algorithmic derivation shared by all five
// TRANSPOSE kernels this exercises.
//
// A is stored MxK, column-major (A[row + col*ld_a], ld_a == m for the
// fast path every series here takes) -- for TRANSPOSE, x has length M
// (the reduction dimension) and y has length K (the output dimension),
// the OPPOSITE of NO_TRANSPOSE's own x-length-K/y-length-M convention.
// flops = 2*m*k either way (order-independent).
//
// Correctness for all five TOMOS series' TRANSPOSE paths is covered in
// verify_mv_fp32_16_gemv.cu -- mv_fp32_tiled, mv_fp32_16_accurate_ready4_
// atomiconly_scaled_kblock, mv_fp32_bf16_accurate_ready_atomiconly_
// scaled_kblock_workspace, and mv_fp32_bf16_accurate_ready_atomiconly_
// scaled_kblock_rounded all already run through that file's own
// TRANSPOSE/NO_TRANSPOSE loop (their signatures take `trans` and were
// already being called both ways before this file existed, so extending
// their dispatch got free TRANSPOSE coverage); mv_fp32_16_accurate_ready_
// atomiconly_scaled_kblock_fast_normal needed a NEW dedicated correctness
// case (safe-normal-range data, since it is UNSAFE on arbitrary data) --
// see that case's own comment in verify_mv_fp32_16_gemv.cu. So this file
// is throughput-only, same convention every *_vs_cutlass_ref.cu /
// *_vs_cublas_ref.cu file in this codebase already uses.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as mv_fp32_recon_vs_cutlass_ref.cu
// (minus that file's own CUTLASS include paths, not needed here). See
// scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh.
//
// Usage:
//   mv_fp32_transpose_recon_vs_cublas_ref
//       Sweeps two shape families: square M=K=1024/2048/4096/8192/16384/
//       32768/65536/131072 (matching mv_fp32_16_accurate's own
//       kMvWideRowThreshold=131072 boundary), then a tall-and-thin sweep
//       with K held fixed at 512 and then 4096, M swept across the same
//       size list -- same two families mv_fp32_recon_vs_cutlass_ref.cu
//       sweeps, same sizes.
//   mv_fp32_transpose_recon_vs_cublas_ref 8192
//       Just that one square shape (M=K=8192).
//   mv_fp32_transpose_recon_vs_cublas_ref --tall 512
//       Just the tall-and-thin sweep at K=512, M swept across the
//       default size list.
//   mv_fp32_transpose_recon_vs_cublas_ref --tall 512 1048576
//       Just the single tall-and-thin shape M=1048576, K=512.

#include <cublas_v2.h>
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
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y);
namespace experimental { void mv_fp32_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size, const float* A,
                                                       size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                                       size_t ld_y); }  // namespace experimental

void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                      const uint16_t* wh, const uint16_t* wr,
                                                                      size_t ld_a, const float* x, size_t ld_x,
                                                                      double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              const uint16_t* wr, size_t ld_a,
                                                                              const float* x, size_t ld_x, double beta,
                                                                              float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4(TransposeType trans, double alpha,
                                                                                 dim<2> size, const uint16_t* wh,
                                                                                 const uint16_t* wr, size_t ld_a,
                                                                                 const float* x, size_t ld_x,
                                                                                 double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget(TransposeType trans,
                                                                                          double alpha, dim<2> size,
                                                                                          const uint16_t* wh,
                                                                                          const uint16_t* wr,
                                                                                          size_t ld_a, const float* x,
                                                                                          size_t ld_x, double beta,
                                                                                          float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2(TransposeType trans,
                                                                                           double alpha, dim<2> size,
                                                                                           const uint16_t* wh,
                                                                                           const uint16_t* wr,
                                                                                           size_t ld_a, const float* x,
                                                                                           size_t ld_x, double beta,
                                                                                           float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled(TransposeType trans,
                                                                                                  double alpha,
                                                                                                  dim<2> size,
                                                                                                  const uint16_t* wh,
                                                                                                  const uint16_t* wr,
                                                                                                  size_t ld_a,
                                                                                                  const float* x,
                                                                                                  size_t ld_x,
                                                                                                  double beta,
                                                                                                  float* y,
                                                                                                  size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split_workspace_size(
    dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split(TransposeType trans,
                                                                                                  double alpha,
                                                                                                  dim<2> size,
                                                                                                  const uint16_t* wh,
                                                                                                  const uint16_t* wr,
                                                                                                  size_t ld_a,
                                                                                                  const float* x,
                                                                                                  size_t ld_x,
                                                                                                  double beta,
                                                                                                  float* y,
                                                                                                  size_t ld_y,
                                                                                                  double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor(TransposeType trans,
                                                                                                 double alpha,
                                                                                                 dim<2> size,
                                                                                                 const uint16_t* wh,
                                                                                                 const uint16_t* wr,
                                                                                                 size_t ld_a,
                                                                                                 const float* x,
                                                                                                 size_t ld_x,
                                                                                                 double beta, float* y,
                                                                                                 size_t ld_y); }  // namespace experimental

void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(TransposeType trans,
                                                                                          double alpha, dim<2> size,
                                                                                          const uint16_t* wh,
                                                                                          const uint16_t* wr,
                                                                                          size_t ld_a, const float* x,
                                                                                          size_t ld_x, double beta,
                                                                                          float* y, size_t ld_y);
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock(TransposeType trans,
                                                                                                     double alpha,
                                                                                                     dim<2> size,
                                                                                                     const uint16_t* wh,
                                                                                                     const uint16_t* wr,
                                                                                                     size_t ld_a,
                                                                                                     const float* x,
                                                                                                     size_t ld_x,
                                                                                                     double beta,
                                                                                                     float* y,
                                                                                                     size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock(TransposeType trans,
                                                                                                    double alpha,
                                                                                                    dim<2> size,
                                                                                                    const uint16_t* wh,
                                                                                                    const uint16_t* wr,
                                                                                                    size_t ld_a,
                                                                                                    const float* x,
                                                                                                    size_t ld_x,
                                                                                                    double beta,
                                                                                                    float* y,
                                                                                                    size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent(TransposeType trans, double alpha,
                                                                                 dim<2> size, const uint16_t* wh,
                                                                                 const uint16_t* wr, size_t ld_a,
                                                                                 const float* x, size_t ld_x,
                                                                                 double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2(TransposeType trans, double alpha,
                                                                                    dim<2> size, const uint16_t* wh,
                                                                                    const uint16_t* wr, size_t ld_a,
                                                                                    const float* x, size_t ld_x,
                                                                                    double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                              const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                              const float* x, size_t ld_x, double beta, float* y,
                                                              size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_warpdot_tiled_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                      const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                      const float* x, size_t ld_x, double beta, float* y,
                                                      size_t ld_y); }  // namespace experimental

void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y);
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned(TransposeType trans, double alpha, dim<2> size,
                                                                  const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                  const float* x, size_t ld_x, double beta, float* y,
                                                                  size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(TransposeType trans, double alpha, dim<2> size,
                                                                       const uint16_t* wh, const uint16_t* wr,
                                                                       size_t ld_a, const float* x, size_t ld_x,
                                                                       double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2(TransposeType trans, double alpha,
                                                                               dim<2> size, const uint16_t* wh,
                                                                               const uint16_t* wr, size_t ld_a,
                                                                               const float* x, size_t ld_x,
                                                                               double beta, float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              const uint16_t* wr, size_t ld_a,
                                                                              const float* x, size_t ld_x,
                                                                              double beta, float* y, size_t ld_y);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y, double* workspace);
void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                            const float* x, size_t ld_x, double beta, float* y,
                                                            size_t ld_y);
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(TransposeType trans, double alpha,
                                                                          dim<2> size, const uint16_t* wh,
                                                                          const uint16_t* wr, size_t ld_a,
                                                                          const float* x, size_t ld_x, double beta,
                                                                          float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2(TransposeType trans, double alpha, dim<2> size,
                                                                      const uint16_t* wh, const uint16_t* wr,
                                                                      size_t ld_a, const float* x, size_t ld_x,
                                                                      double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                   const float* x, size_t ld_x, double beta, float* y,
                                                                   size_t ld_y, double* workspace);
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                    const uint16_t* wh, const uint16_t* wr,
                                                                    size_t ld_a, const float* x, size_t ld_x,
                                                                    double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
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

// Safe-normal-range variant of pseudo_rand -- see gen_fp32_16_safe_
// normal's own comment in mv_fp32_recon_vs_cutlass_ref.cu for the full
// rationale (identical here, copied rather than shared since these two
// benchmark binaries are each self-contained standalone nvcc builds with
// no shared translation unit between them). Required by "TOMOS FP32/FP16
// (fast)" -- fp32_16_combine_fast_normal decodes unconditionally as if
// control were always 0, so its own data must PROVABLY never be extended/
// degraded-encoded.
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
}

__global__ void gen_fp32_16_safe_normal_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_safe_normal(i), wh[i], wr[i]);
}
void gen_fp32_16_safe_normal(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_safe_normal_kernel<<<blocks, threads>>>(n, wh, wr);
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

// Rounded-encoding sibling of gen_fp32_bf16 -- see gen_fp32_bf16_
// rounded's own comment in mv_fp32_recon_vs_cutlass_ref.cu (copied here
// for the same standalone-binary reason as gen_fp32_16_safe_normal
// above). Required by plain "TOMOS FP32/BF16": fp32_bf16_combine_rounded
// expects wr's bit0 to be a round_flag, not a real mantissa bit -- feeding
// it plain-truncated-split data would silently corrupt every element.
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

}  // namespace

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
// TRANSPOSE (y = alpha*A^T*x + beta*y): x has length M (the reduction
// dimension), y has length K (the output dimension) -- the OPPOSITE of
// mv_fp32_recon_vs_cutlass_ref.cu's own NO_TRANSPOSE run_shape, where x
// is length K and y is length M. A's own representation (plain float,
// wh/wr for fp32_16, wh/wr for fp32_bf16) is transient, allocated right
// before and freed right after its own timing block -- same staggered-
// allocation discipline the NO_TRANSPOSE file's own run_shape uses.
void run_shape(size_t m, size_t k, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);

    int lda = static_cast<int>(m);
    size_t maxdim = std::max(m, k);
    int reps = (maxdim <= 4096) ? 50 : (maxdim <= 16384 ? 20 : 5);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k);

    std::printf("  %s\n", shape_desc);

    float *x = nullptr, *y = nullptr;
    bool ok_xy = alloc_or_skip(reinterpret_cast<void**>(&x), m * sizeof(float), "x", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&y), k * sizeof(float), "y", shape_desc);
    if (ok_xy) {
        gen_float(m, x);
        cudaMemset(y, 0, k * sizeof(float));
        cudaDeviceSynchronize();
    }

    double ms_native = -1.0, ms_native_vec4 = -1.0, ms_native_vec4_stagevec = -1.0, ms_fp32_16 = -1.0,
           ms_fp32_16_vec4 = -1.0, ms_fp32_16_vec4_retuned = -1.0, ms_fp32_16_vec4_retuned_floor = -1.0,
           ms_fp32_16_vec4_wrapborrow = -1.0, ms_fp32_16_vec4_wrapborrow_retarget2 = -1.0,
           ms_fp32_16_vec4_wrapborrow_dispatch = -1.0,
           ms_fp32_16_fast = -1.0, ms_fp32_16_fast_vec4 = -1.0, ms_fp32_16_fast_vec4_retuned = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4 = -1.0, ms_fp32_16_fast_wrapborrow_vec4_retarget = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget2 = -1.0, ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget2_split = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_dispatch = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock = -1.0,
           ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock = -1.0,
           ms_fp32_16_fast_vec4_persistent = -1.0, ms_fp32_16_fast_vec4_persistent_v2 = -1.0,
           ms_fp32_16_fast_warprows_splitm_tiled = -1.0, ms_fp32_16_fast_warpdot_tiled = -1.0,
           ms_fp32_bf16 = -1.0, ms_fp32_bf16_truncated = -1.0,
           ms_fp32_bf16_vec4 = -1.0, ms_fp32_bf16_vec4_retuned_floor = -1.0,
           ms_fp32_bf16_vec4_retarget2 = -1.0, ms_fp32_bf16_vec4_dispatch = -1.0,
           ms_fp32_bf16_vec4_rounded = -1.0, ms_fp32_bf16_vec4_rounded_retuned_floor = -1.0,
           ms_fp32_bf16_vec4_rounded_retarget2 = -1.0, ms_fp32_bf16_vec4_rounded_dispatch = -1.0,
           ms_cublas = -1.0;

    if (ok_xy) {
        float* A_native = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "mv_fp32_tiled (A)",
                          shape_desc)) {
            gen_float(m * k, A_native);
            cudaDeviceSynchronize();

            ms_native = time_call(start, stop, [&]() {
                cuda::mv_fp32_tiled(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, A_native, static_cast<size_t>(lda),
                                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_tiled          = %9.3fms (%10.1f GFlop/s)\n", ms_native,
                       flops / ms_native / 1e6);

            // mv_fp32_transpose_atomiconly_scaled_mblock_vec4 -- EXPERIMENTAL,
            // standalone A/B sibling of mv_fp32_tiled's own TRANSPOSE fast
            // path (float4-vectorized loads of A and the shared x tile,
            // runtime 16-byte alignment check before committing to the
            // vectorized path -- see its own comment in mv_cuda_
            // kernels.cu). Reuses the SAME A_native/x/y data as mv_fp32_
            // tiled's own leg just above -- both compute the exact same
            // y = alpha*A^T*x + beta*y, so a direct back-to-back
            // comparison at unchanged data is valid.
            ms_native_vec4 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_transpose_atomiconly_scaled_mblock_vec4(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k},
                                                                       A_native, static_cast<size_t>(lda), x, 1, 0.3,
                                                                       y, 1);
            }, reps);
            std::printf("    mv_fp32_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_vec4, flops / ms_native_vec4 / 1e6);

            // mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec --
            // EXPERIMENTAL, standalone A/B sibling of the vec4 kernel just
            // above (ALSO float4-vectorizes the x-staging loop, not just
            // the main accumulation loop's A/sx reads -- see its own
            // comment in mv_cuda_kernels.cu). Same A_native/x/y data.
            ms_native_vec4_stagevec = time_call(start, stop, [&]() {
                cuda::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, A_native, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec = %9.3fms (%10.1f GFlop/s)\n",
                       ms_native_vec4_stagevec, flops / ms_native_vec4_stagevec / 1e6);

            float f_alpha = 1.7f, f_beta = 0.3f;
            ms_cublas = time_call(start, stop, [&]() {
                cublasSgemv(handle, CUBLAS_OP_T, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native, lda, x,
                           1, &f_beta, y, 1);
            }, reps);
            std::printf("    cublas (Sgemv)         = %9.3fms (%10.1f GFlop/s)\n", ms_cublas,
                       flops / ms_cublas / 1e6);

            cudaFree(A_native);
        }
    }

    if (ok_xy) {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock (wr)", shape_desc);
        if (ok) {
            gen_fp32_16(m * k, wh16, wr16);
            cudaDeviceSynchronize();

            // "TOMOS FP32/FP16" -- mv_fp32_16_accurate_ready4_atomiconly_
            // scaled_kblock's own NEW TRANSPOSE fast path (warp-per-
            // output-column/split-M/atomicAdd, see this file's own header
            // comment). Safe on arbitrary data (fp32_16_combine_
            // branchless), same plain wh16/wr16 data as the NO_TRANSPOSE
            // file's own "TOMOS FP32/FP16" leg.
            ms_fp32_16 = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k},
                                                                          wh16, wr16, static_cast<size_t>(lda),
                                                                          x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16, flops / ms_fp32_16 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4 --
            // EXPERIMENTAL, standalone A/B sibling of the leg just above
            // (float4-vectorized x-staging/sx reads, ushort4-vectorized
            // wh/wr loads -- see its own comment in fp32_16_cuda_
            // kernels.cu). Same plain wh16/wr16 data.
            ms_fp32_16_vec4 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k},
                                                                          wh16, wr16, static_cast<size_t>(lda),
                                                                          x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4, flops / ms_fp32_16_vec4 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned
            // -- EXPERIMENTAL, standalone A/B sibling reusing the EXACT
            // SAME __global__ kernel as the leg just above, only a
            // different M_BLOCK-scaling grid-oversubscription target --
            // see its own comment in fp32_16_cuda_kernels.cu.
            ms_fp32_16_vec4_retuned = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4_retuned, flops / ms_fp32_16_vec4_retuned / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
            // retuned_floor -- EXPERIMENTAL, standalone A/B sibling
            // reusing the EXACT SAME __global__ kernel and target as the
            // leg just above, but with a floor on how far M_BLOCK can
            // shrink -- see its own comment in fp32_16_cuda_kernels.cu.
            ms_fp32_16_vec4_retuned_floor = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4_retuned_floor, flops / ms_fp32_16_vec4_retuned_floor / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
            // wrapborrow -- "TOMOS FP32/FP16" (PROMOTED). Reuses the
            // EXACT SAME __global__ kernel structure and M_BLOCK-scaling
            // as _retuned_floor above, only decoding via fp32_16_combine_
            // branchless_wrapborrow instead of fp32_16_combine_branchless
            // -- see its own comment in fp32_16_cuda_kernels.cu.
            ms_fp32_16_vec4_wrapborrow = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4_wrapborrow, flops / ms_fp32_16_vec4_wrapborrow / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
            // wrapborrow_retarget2 -- EXPERIMENTAL, standalone A/B
            // sibling reusing the EXACT SAME __global__ kernel as
            // _wrapborrow above, only a higher M_BLOCK-scaling target
            // (8192 instead of 2048) -- a high-confidence port of the
            // UNSAFE fast_normal_wrapborrow family's own retarget2, since
            // the grid-math formula is decode-independent -- see its own
            // comment in fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_16_vec4_wrapborrow_retarget2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4_wrapborrow_retarget2, flops / ms_fp32_16_vec4_wrapborrow_retarget2 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
            // wrapborrow_dispatch -- EXPERIMENTAL, standalone dispatcher
            // routing to _wrapborrow or _wrapborrow_retarget2 based on M
            // -- see its own comment in fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_16_vec4_wrapborrow_dispatch = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_vec4_wrapborrow_dispatch, flops / ms_fp32_16_vec4_wrapborrow_dispatch / 1e6);
        }
        cudaFree(wh16);
        cudaFree(wr16);
    }

    if (ok_xy) {
        uint16_t *wh16_safe = nullptr, *wr16_safe = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16_safe), m * k * sizeof(uint16_t),
                               "mv_fp32_16_..._fast_normal (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr16_safe), m * k * sizeof(uint16_t),
                               "mv_fp32_16_..._fast_normal (wr)", shape_desc);
        if (ok) {
            gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe);
            cudaDeviceSynchronize();

            // "TOMOS FP32/FP16 (fast)" -- mv_fp32_16_accurate_ready_
            // atomiconly_scaled_kblock_fast_normal's own NEW TRANSPOSE
            // fast path. UNSAFE on arbitrary data (fp32_16_combine_fast_
            // normal's own contract), so tested against dedicated safe-
            // range data (mag in [0.5, 1.0]), not the plain wh16/wr16
            // used by the plain "TOMOS FP32/FP16" leg above.
            ms_fp32_16_fast = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast, flops / ms_fp32_16_fast / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // vec4 -- EXPERIMENTAL, standalone A/B sibling of the leg
            // just above (float4-vectorized x-staging/sx reads, ushort4-
            // vectorized wh/wr loads -- see its own comment in fp32_16_
            // cuda_kernels.cu). Same safe-range wh16_safe/wr16_safe data.
            ms_fp32_16_fast_vec4 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_vec4, flops / ms_fp32_16_fast_vec4 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // vec4_retuned -- EXPERIMENTAL, standalone A/B sibling reusing
            // the EXACT SAME __global__ kernel as the leg just above, only
            // a more conservative M_BLOCK-scaling grid target derived from
            // this kernel's own (higher) register footprint -- see its own
            // comment in fp32_16_cuda_kernels.cu. Same safe-range data.
            ms_fp32_16_fast_vec4_retuned = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_vec4_retuned, flops / ms_fp32_16_fast_vec4_retuned / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4 -- EXPERIMENTAL, standalone A/B sibling:
            // identical structure to fast_normal_vec4, only decoding via
            // fp32_16_combine_fast_normal_wrapborrow (bit-identical,
            // exhaustively verified). CONFIRMED via cuobjdump: eliminates
            // all 38 SEL/19 ISETP.EQ instructions the is_carry branch
            // compiled to, drops REG:39 -> REG:32 -- see its own comment
            // in fp32_16_cuda_kernels.cu. Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4, flops / ms_fp32_16_fast_wrapborrow_vec4 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget -- EXPERIMENTAL, standalone
            // sibling reusing the EXACT SAME __global__ kernel as the leg
            // just above, only a more aggressive M_BLOCK-scaling target
            // (2048, re-derived from wrapborrow's own REG:32 footprint,
            // matching the SAFE decode kernel's own REG:31/2048-target
            // residency profile) -- see its own comment in fp32_16_
            // cuda_kernels.cu. Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2 -- EXPERIMENTAL, standalone
            // sibling reusing the EXACT SAME __global__ kernel as the leg
            // above, only an even more aggressive M_BLOCK-scaling target
            // (8192) aimed specifically at the two tall-and-thin shapes
            // (K=4096, M=32768/65536) where target=2048 never actually
            // drops M_BLOCK below the 8192 ceiling -- see its own comment
            // in fp32_16_cuda_kernels.cu. Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2 / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2_tiled -- standalone sibling
            // reusing the EXACT SAME M_BLOCK-scaling and atomicAdd
            // combine as the leg just above, only staging x into shared
            // memory in 2048-float sub-tiles instead of the whole
            // M_BLOCK range at once -- ncu profiling (real A100
            // hardware, M=K=65536) found shared memory, not registers or
            // atomics, is the actual binding occupancy constraint there
            // ("Block Limit Shared Mem: 4" vs "Block Limit Registers: 8",
            // Achieved Occupancy 48.87% sitting right at the resulting
            // 50% ceiling) -- see its own comment in fp32_16_cuda_
            // kernels.cu. UNTESTED. Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2_split -- standalone sibling
            // reusing the EXACT SAME M_BLOCK-scaling as the leg just
            // above (target=8192, floor=256), only replacing the
            // atomicAdd-per-block combine with a scratch-buffer partial
            // sum + separate reduce kernel. Workspace allocated once
            // below, OUTSIDE the timed region, so this measurement
            // isolates the mechanism from allocator overhead. CLOSED,
            // CONFIRMED NEGATIVE RESULT on real A100 hardware even with
            // that confound removed: a small but consistent -0.2% to
            // -2.8% regression vs plain _retarget2 at every size tested,
            // LARGEST at M=8192 (gridM=1, zero atomic contention to
            // remove) and flat (not growing) from M=24576 up -- A100's
            // hardware atomics are cheap enough at this kernel family's
            // gridM range that removing them nets nothing; see its own
            // comment in fp32_16_cuda_kernels.cu for the full derivation.
            // Same safe-range data.
            size_t retarget2_split_ws_elems =
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split_workspace_size(
                    dim<2>{m, k});
            double* d_retarget2_split_ws = nullptr;
            cudaMalloc(&d_retarget2_split_ws, retarget2_split_ws_elems * sizeof(double));
            ms_fp32_16_fast_wrapborrow_vec4_retarget2_split = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1, d_retarget2_split_ws);
            }, reps);
            cudaFree(d_retarget2_split_ws);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2_split, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_split / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2_floor -- EXPERIMENTAL, standalone
            // sibling reusing the EXACT SAME __global__ kernel and target
            // as the leg above, only a higher floor (2048) fixing
            // _retarget2's own confirmed M=3072-6144 regression -- see
            // its own comment in fp32_16_cuda_kernels.cu. Same safe-
            // range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_dispatch -- EXPERIMENTAL, standalone
            // dispatcher routing to _retarget or _retarget2 based on M
            // -- see its own comment in fp32_16_cuda_kernels.cu. Should
            // track _retarget's own numbers for M<8192 and _retarget2's
            // own numbers for M>=8192 (modulo the small per-call
            // branch/dispatch overhead) -- included here mainly to
            // confirm the routing itself behaves as expected end-to-end,
            // not as a new independently-tuned kernel. Same safe-range
            // data.
            ms_fp32_16_fast_wrapborrow_vec4_dispatch = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_dispatch, flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2_bigblock -- EXPERIMENTAL,
            // standalone sibling reusing the EXACT SAME __global__ kernel
            // and target as _retarget2 above, only a higher shared
            // M_BLOCK ceiling (16384 instead of 8192) -- targets the
            // atomicAdd-per-column contention the shared ceiling forces
            // at very large SQUARE shapes (M=K>=32768), where gridK alone
            // is already large enough that the target formula's own math
            // wants a bigger M_BLOCK than the ceiling allows -- see its
            // own comment in fp32_16_cuda_kernels.cu for the full
            // derivation and the genuinely open occupancy tradeoff this
            // raises (fewer atomics vs fewer resident blocks/SM). UNTESTED.
            // Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget_bigblock -- EXPERIMENTAL,
            // standalone sibling reusing the EXACT SAME __global__ kernel
            // and higher ceiling (16384) as _retarget2_bigblock above,
            // only _retarget's own LOWER target (2048) instead of
            // _retarget2's 8192 -- fixes a gap _retarget2_bigblock's own
            // derivation missed: at M=24576 (square), target=8192 lands
            // on scaled_block=8192 NATURALLY (independent of the
            // ceiling), so a higher ceiling alone does nothing there --
            // see its own comment in fp32_16_cuda_kernels.cu for the full
            // grid-math derivation. UNTESTED. Same safe-range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // wrapborrow_vec4_retarget2_hugeblock -- EXPERIMENTAL,
            // standalone sibling reusing the EXACT SAME __global__ kernel
            // and target as _retarget2/_retarget2_bigblock above, only an
            // even higher ceiling (32768 instead of 16384) -- pushes
            // further at M=49152/65536 square where _retarget2_bigblock's
            // own ceiling still binds -- see its own comment in
            // fp32_16_cuda_kernels.cu for the full derivation and the
            // steeper occupancy question this raises (128KB shared
            // memory/block, ~1 resident block/SM). UNTESTED. Same safe-
            // range data.
            ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock, flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // vec4_persistent -- EXPERIMENTAL, standalone sibling: caps
            // the grid's M-width at the register-limited full-residency
            // ceiling and loops internally over extra M_BLOCK chunks
            // instead of launching more blocks, doing ONE atomicAdd per
            // warp for its entire lifetime instead of one per chunk --
            // see its own comment in fp32_16_cuda_kernels.cu. Same safe-
            // range data.
            ms_fp32_16_fast_vec4_persistent = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_vec4_persistent, flops / ms_fp32_16_fast_vec4_persistent / 1e6);

            // mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
            // vec4_persistent_v2 -- EXPERIMENTAL, standalone sibling of
            // the leg just above: identical launch config, only the
            // trailing __syncthreads() is now conditional on there being
            // another loop iteration -- see its own comment in fp32_16_
            // cuda_kernels.cu. Same safe-range data.
            ms_fp32_16_fast_vec4_persistent_v2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_vec4_persistent_v2, flops / ms_fp32_16_fast_vec4_persistent_v2 / 1e6);

            // mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal --
            // EXPERIMENTAL, standalone sibling: ported DIRECTLY from
            // randls2.0's own sgemv_cm_transpose_warprows_splitm_tiled --
            // no atomics, one persistent block per column-stripe, SPLITM
            // warp-groups split each M-tile concurrently -- see its own
            // comment in fp32_16_cuda_kernels.cu (including a real
            // uncoalesced-access caveat). Same safe-range data.
            ms_fp32_16_fast_warprows_splitm_tiled = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_warprows_splitm_tiled, flops / ms_fp32_16_fast_warprows_splitm_tiled / 1e6);

            // mv_fp32_16_transpose_warpdot_tiled_fast_normal --
            // EXPERIMENTAL, standalone sibling: ported DIRECTLY from
            // randls2.0's own sgemv_cm_transpose_warpdot_tiled -- one
            // warp per column loops the ENTIRE M range internally, no
            // atomics, no M_BLOCK-scaling -- see its own comment in
            // fp32_16_cuda_kernels.cu. The randls2.0 reference (plain
            // fp32, no decode) measured near-parity with cuBLAS via a
            // standalone driver (mv_fp32_transpose_randls2_0_warpdot_
            // tiled_reference.cu). Same safe-range data.
            ms_fp32_16_fast_warpdot_tiled = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_transpose_warpdot_tiled_fast_normal(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh16_safe, wr16_safe, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_transpose_warpdot_tiled_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_fast_warpdot_tiled, flops / ms_fp32_16_fast_warpdot_tiled / 1e6);
        }
        cudaFree(wh16_safe);
        cudaFree(wr16_safe);
    }

    if (ok_xy) {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_..._workspace (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_..._workspace (wr)", shape_desc);
        if (ok) {
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16);
            cudaDeviceSynchronize();

            // "TOMOS FP32/BF16 (truncated)" -- mv_fp32_bf16_accurate_
            // ready_atomiconly_scaled_kblock_workspace's own NEW
            // TRANSPOSE fast path, decoding via plain fp32_bf16_combine
            // against plain-truncating-split data. `workspace` passed as
            // nullptr -- this fast path never touches it (see this
            // file's own header comment / the dispatcher's own comment
            // in fp32_16_cuda_kernels.cu).
            ms_fp32_bf16_truncated = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1, nullptr);
            }, reps);
            std::printf("    mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_truncated, flops / ms_fp32_bf16_truncated / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4 --
            // EXPERIMENTAL, standalone TRANSPOSE-only A/B sibling of the
            // leg just above, float4/ushort4-vectorized, same M_BLOCK-
            // scaling -- ported directly from SAFE fp32_16's own "(vec4)"
            // design, per this codebase's "base the design on FP32/FP16
            // (fast)" convention -- see its own comment in fp32_16_cuda_
            // kernels.cu. UNTESTED. Same plain wh_bf16/wr_bf16 data.
            ms_fp32_bf16_vec4 = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4, flops / ms_fp32_bf16_vec4 / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // retuned_floor -- EXPERIMENTAL, standalone A/B sibling
            // reusing the EXACT SAME __global__ kernel as _vec4 above,
            // only fp32_16's own already-validated M_BLOCK-scaling
            // retuning (target=2048, floor=256) -- see its own comment
            // in fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_bf16_vec4_retuned_floor = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_retuned_floor, flops / ms_fp32_bf16_vec4_retuned_floor / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // retarget2 -- EXPERIMENTAL, standalone A/B sibling reusing
            // the EXACT SAME __global__ kernel as _vec4 above, only
            // target=8192 -- exists as _dispatch's own M>=8192 branch,
            // see its own comment in fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_bf16_vec4_retarget2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_retarget2, flops / ms_fp32_bf16_vec4_retarget2 / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // dispatch -- standalone dispatcher routing to plain _vec4
            // (M<8192, the currently-promoted "TOMOS FP32/BF16
            // (truncated)") or _retarget2 (M>=8192) -- targets large-M
            // performance specifically, see its own comment in
            // fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_bf16_vec4_dispatch = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16, static_cast<size_t>(lda),
                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_dispatch, flops / ms_fp32_bf16_vec4_dispatch / 1e6);
        }
        cudaFree(wh_bf16);
        cudaFree(wr_bf16);
    }

    if (ok_xy) {
        uint16_t *wh_bf16_rounded = nullptr, *wr_bf16_rounded = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16_rounded), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_..._rounded (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_rounded), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_..._rounded (wr)", shape_desc);
        if (ok) {
            gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded);
            cudaDeviceSynchronize();

            // "TOMOS FP32/BF16" -- mv_fp32_bf16_accurate_ready_
            // atomiconly_scaled_kblock_rounded's own NEW TRANSPOSE fast
            // path, decoding via fp32_bf16_combine_rounded against
            // rounded-split data (fp32_bf16_split_rounded). `workspace`
            // passed as nullptr, same reason as the truncated leg above.
            ms_fp32_bf16 = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1, nullptr);
            }, reps);
            std::printf("    mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16, flops / ms_fp32_bf16 / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // rounded -- standalone TRANSPOSE-only A/B sibling of the leg
            // just above, float4/ushort4-vectorized, same M_BLOCK-scaling
            // -- rounded-encoding counterpart of the "(truncated)"
            // family's own promoted _vec4, per this codebase's "base the
            // design on FP32/FP16 (fast)" convention -- see its own
            // comment in fp32_16_cuda_kernels.cu. DEMOTED now that
            // _rounded_dispatch (below) sources "TOMOS FP32/BF16" instead
            // -- kept as its own visible leg. Same wh_bf16_rounded/
            // wr_bf16_rounded data.
            ms_fp32_bf16_vec4_rounded = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_rounded, flops / ms_fp32_bf16_vec4_rounded / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // rounded_retuned_floor -- EXPERIMENTAL, standalone A/B
            // sibling reusing the EXACT SAME __global__ kernel as
            // _vec4_rounded above, only fp32_16's own already-validated
            // M_BLOCK-scaling retuning (target=2048, floor=256) -- see
            // its own comment in fp32_16_cuda_kernels.cu. UNTESTED.
            ms_fp32_bf16_vec4_rounded_retuned_floor = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_rounded_retuned_floor, flops / ms_fp32_bf16_vec4_rounded_retuned_floor / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // rounded_retarget2 -- EXPERIMENTAL, standalone A/B sibling
            // reusing the EXACT SAME __global__ kernel as _vec4_rounded
            // above, only target=8192 -- exists as _rounded_dispatch's
            // own M>=8192 branch, see its own comment in fp32_16_cuda_
            // kernels.cu. UNTESTED.
            ms_fp32_bf16_vec4_rounded_retarget2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_rounded_retarget2, flops / ms_fp32_bf16_vec4_rounded_retarget2 / 1e6);

            // mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_
            // rounded_dispatch -- standalone dispatcher routing to plain
            // _vec4_rounded (M<8192, the demoted predecessor above) or
            // _rounded_retarget2 (M>=8192). PROMOTED: this is "TOMOS
            // FP32/BF16" -- see its own comment in fp32_16_cuda_
            // kernels.cu for the real-hardware confirmation.
            ms_fp32_bf16_vec4_rounded_dispatch = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_vec4_rounded_dispatch, flops / ms_fp32_bf16_vec4_rounded_dispatch / 1e6);
        }
        cudaFree(wh_bf16_rounded);
        cudaFree(wr_bf16_rounded);
    }

    if (ms_native > 0 && ms_cublas > 0)
        std::printf("    -- native vs cublas: %.3fx\n", (flops / ms_native) / (flops / ms_cublas));
    if (ms_native_vec4 > 0 && ms_cublas > 0)
        std::printf("    -- native_vec4 vs cublas: %.3fx\n", (flops / ms_native_vec4) / (flops / ms_cublas));
    if (ms_native_vec4 > 0 && ms_native > 0)
        std::printf("    -- native_vec4 vs native (unvectorized): %.3fx\n", (flops / ms_native_vec4) / (flops / ms_native));
    if (ms_native_vec4_stagevec > 0 && ms_cublas > 0)
        std::printf("    -- native_vec4_stagevec vs cublas: %.3fx\n", (flops / ms_native_vec4_stagevec) / (flops / ms_cublas));
    if (ms_native_vec4_stagevec > 0 && ms_native_vec4 > 0)
        std::printf("    -- native_vec4_stagevec vs native_vec4: %.3fx\n", (flops / ms_native_vec4_stagevec) / (flops / ms_native_vec4));
    if (ms_fp32_16 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16 vs cublas: %.3fx\n", (flops / ms_fp32_16) / (flops / ms_cublas));
    if (ms_fp32_16_vec4 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4 vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4) / (flops / ms_cublas));
    if (ms_fp32_16_vec4 > 0 && ms_fp32_16 > 0)
        std::printf("    -- fp32_16_vec4 vs fp32_16 (scalar): %.3fx\n", (flops / ms_fp32_16_vec4) / (flops / ms_fp32_16));
    if (ms_fp32_16_vec4_retuned > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4_retuned vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4_retuned) / (flops / ms_cublas));
    if (ms_fp32_16_vec4_retuned > 0 && ms_fp32_16_vec4 > 0)
        std::printf("    -- fp32_16_vec4_retuned vs fp32_16_vec4: %.3fx\n", (flops / ms_fp32_16_vec4_retuned) / (flops / ms_fp32_16_vec4));
    if (ms_fp32_16_vec4_retuned_floor > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4_retuned_floor vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4_retuned_floor) / (flops / ms_cublas));
    if (ms_fp32_16_vec4_retuned_floor > 0 && ms_fp32_16_vec4 > 0)
        std::printf("    -- fp32_16_vec4_retuned_floor vs fp32_16_vec4: %.3fx\n", (flops / ms_fp32_16_vec4_retuned_floor) / (flops / ms_fp32_16_vec4));
    if (ms_fp32_16_vec4_wrapborrow > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow) / (flops / ms_cublas));
    if (ms_fp32_16_vec4_wrapborrow > 0 && ms_fp32_16_vec4_retuned_floor > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow vs fp32_16_vec4_retuned_floor: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow) / (flops / ms_fp32_16_vec4_retuned_floor));
    if (ms_fp32_16_vec4_wrapborrow_retarget2 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow_retarget2 vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow_retarget2) / (flops / ms_cublas));
    if (ms_fp32_16_vec4_wrapborrow_retarget2 > 0 && ms_fp32_16_vec4_wrapborrow > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow_retarget2 vs fp32_16_vec4_wrapborrow: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow_retarget2) / (flops / ms_fp32_16_vec4_wrapborrow));
    if (ms_fp32_16_vec4_wrapborrow_dispatch > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow_dispatch vs cublas: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow_dispatch) / (flops / ms_cublas));
    if (ms_fp32_16_vec4_wrapborrow_dispatch > 0 && ms_fp32_16_vec4_wrapborrow > 0)
        std::printf("    -- fp32_16_vec4_wrapborrow_dispatch vs fp32_16_vec4_wrapborrow: %.3fx\n", (flops / ms_fp32_16_vec4_wrapborrow_dispatch) / (flops / ms_fp32_16_vec4_wrapborrow));
    if (ms_fp32_16_fast > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast vs cublas: %.3fx\n", (flops / ms_fp32_16_fast) / (flops / ms_cublas));
    if (ms_fp32_16_fast_vec4 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_vec4 vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_vec4) / (flops / ms_cublas));
    if (ms_fp32_16_fast_vec4 > 0 && ms_fp32_16_fast > 0)
        std::printf("    -- fp32_16_fast_vec4 vs fp32_16_fast (scalar): %.3fx\n", (flops / ms_fp32_16_fast_vec4) / (flops / ms_fp32_16_fast));
    if (ms_fp32_16_fast_vec4_retuned > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_vec4_retuned vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_vec4_retuned) / (flops / ms_cublas));
    if (ms_fp32_16_fast_vec4_retuned > 0 && ms_fp32_16_fast_vec4 > 0)
        std::printf("    -- fp32_16_fast_vec4_retuned vs fp32_16_fast_vec4: %.3fx\n", (flops / ms_fp32_16_fast_vec4_retuned) / (flops / ms_fp32_16_fast_vec4));
    if (ms_fp32_16_fast_wrapborrow_vec4 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4 vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4 > 0 && ms_fp32_16_fast_vec4_retuned > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4 vs fp32_16_fast_vec4_retuned: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4) / (flops / ms_fp32_16_fast_vec4_retuned));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget > 0 && ms_fp32_16_fast_wrapborrow_vec4 > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget vs fp32_16_fast_wrapborrow_vec4: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget) / (flops / ms_fp32_16_fast_wrapborrow_vec4));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2 vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2 vs fp32_16_fast_wrapborrow_vec4_retarget: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_tiled vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_tiled vs fp32_16_fast_wrapborrow_vec4_retarget2: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_tiled) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_split > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_split vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_split) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_split > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_split vs fp32_16_fast_wrapborrow_vec4_retarget2: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_split) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_floor vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_floor vs fp32_16_fast_wrapborrow_vec4_retarget: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_floor vs fp32_16_fast_wrapborrow_vec4_retarget2: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_floor) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2));
    if (ms_fp32_16_fast_wrapborrow_vec4_dispatch > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_dispatch vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_dispatch > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_dispatch vs fp32_16_fast_wrapborrow_vec4_retarget: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_bigblock vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2 > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_bigblock vs fp32_16_fast_wrapborrow_vec4_retarget2: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_dispatch > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_bigblock vs fp32_16_fast_wrapborrow_vec4_dispatch: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget_bigblock vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget_bigblock vs fp32_16_fast_wrapborrow_vec4_retarget2_bigblock: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_dispatch > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget_bigblock vs fp32_16_fast_wrapborrow_vec4_dispatch: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget_bigblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock) / (flops / ms_cublas));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock vs fp32_16_fast_wrapborrow_vec4_retarget2_bigblock: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_bigblock));
    if (ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock > 0 && ms_fp32_16_fast_wrapborrow_vec4_dispatch > 0)
        std::printf("    -- fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock vs fp32_16_fast_wrapborrow_vec4_dispatch: %.3fx\n", (flops / ms_fp32_16_fast_wrapborrow_vec4_retarget2_hugeblock) / (flops / ms_fp32_16_fast_wrapborrow_vec4_dispatch));
    if (ms_fp32_16_fast_vec4_persistent > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_vec4_persistent vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_vec4_persistent) / (flops / ms_cublas));
    if (ms_fp32_16_fast_vec4_persistent > 0 && ms_fp32_16_fast_vec4_retuned > 0)
        std::printf("    -- fp32_16_fast_vec4_persistent vs fp32_16_fast_vec4_retuned: %.3fx\n", (flops / ms_fp32_16_fast_vec4_persistent) / (flops / ms_fp32_16_fast_vec4_retuned));
    if (ms_fp32_16_fast_vec4_persistent_v2 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_vec4_persistent_v2 vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_vec4_persistent_v2) / (flops / ms_cublas));
    if (ms_fp32_16_fast_vec4_persistent_v2 > 0 && ms_fp32_16_fast_vec4_retuned > 0)
        std::printf("    -- fp32_16_fast_vec4_persistent_v2 vs fp32_16_fast_vec4_retuned: %.3fx\n", (flops / ms_fp32_16_fast_vec4_persistent_v2) / (flops / ms_fp32_16_fast_vec4_retuned));
    if (ms_fp32_16_fast_vec4_persistent_v2 > 0 && ms_fp32_16_fast_vec4_persistent > 0)
        std::printf("    -- fp32_16_fast_vec4_persistent_v2 vs fp32_16_fast_vec4_persistent: %.3fx\n", (flops / ms_fp32_16_fast_vec4_persistent_v2) / (flops / ms_fp32_16_fast_vec4_persistent));
    if (ms_fp32_16_fast_warprows_splitm_tiled > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_warprows_splitm_tiled vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_warprows_splitm_tiled) / (flops / ms_cublas));
    if (ms_fp32_16_fast_warprows_splitm_tiled > 0 && ms_fp32_16_fast_vec4_retuned > 0)
        std::printf("    -- fp32_16_fast_warprows_splitm_tiled vs fp32_16_fast_vec4_retuned: %.3fx\n", (flops / ms_fp32_16_fast_warprows_splitm_tiled) / (flops / ms_fp32_16_fast_vec4_retuned));
    if (ms_fp32_16_fast_warpdot_tiled > 0 && ms_cublas > 0)
        std::printf("    -- fp32_16_fast_warpdot_tiled vs cublas: %.3fx\n", (flops / ms_fp32_16_fast_warpdot_tiled) / (flops / ms_cublas));
    if (ms_fp32_16_fast_warpdot_tiled > 0 && ms_fp32_16_fast_vec4_retuned > 0)
        std::printf("    -- fp32_16_fast_warpdot_tiled vs fp32_16_fast_vec4_retuned: %.3fx\n", (flops / ms_fp32_16_fast_warpdot_tiled) / (flops / ms_fp32_16_fast_vec4_retuned));
    if (ms_fp32_bf16 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16 vs cublas: %.3fx\n", (flops / ms_fp32_bf16) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_rounded > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_rounded vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_rounded > 0 && ms_fp32_bf16 > 0)
        std::printf("    -- fp32_bf16_vec4_rounded vs fp32_bf16: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded) / (flops / ms_fp32_bf16));
    if (ms_fp32_bf16_vec4_rounded_retuned_floor > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_retuned_floor vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_retuned_floor) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_rounded_retuned_floor > 0 && ms_fp32_bf16_vec4_rounded > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_retuned_floor vs fp32_bf16_vec4_rounded: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_retuned_floor) / (flops / ms_fp32_bf16_vec4_rounded));
    if (ms_fp32_bf16_vec4_rounded_retarget2 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_retarget2 vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_retarget2) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_rounded_retarget2 > 0 && ms_fp32_bf16_vec4_rounded > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_retarget2 vs fp32_bf16_vec4_rounded: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_retarget2) / (flops / ms_fp32_bf16_vec4_rounded));
    if (ms_fp32_bf16_vec4_rounded_dispatch > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_dispatch vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_dispatch) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_rounded_dispatch > 0 && ms_fp32_bf16_vec4_rounded > 0)
        std::printf("    -- fp32_bf16_vec4_rounded_dispatch vs fp32_bf16_vec4_rounded: %.3fx\n", (flops / ms_fp32_bf16_vec4_rounded_dispatch) / (flops / ms_fp32_bf16_vec4_rounded));
    if (ms_fp32_bf16_truncated > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_truncated vs cublas: %.3fx\n", (flops / ms_fp32_bf16_truncated) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4 vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4 > 0 && ms_fp32_bf16_truncated > 0)
        std::printf("    -- fp32_bf16_vec4 vs fp32_bf16_truncated: %.3fx\n", (flops / ms_fp32_bf16_vec4) / (flops / ms_fp32_bf16_truncated));
    if (ms_fp32_bf16_vec4_retuned_floor > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_retuned_floor vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_retuned_floor) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_retuned_floor > 0 && ms_fp32_bf16_vec4 > 0)
        std::printf("    -- fp32_bf16_vec4_retuned_floor vs fp32_bf16_vec4: %.3fx\n", (flops / ms_fp32_bf16_vec4_retuned_floor) / (flops / ms_fp32_bf16_vec4));
    if (ms_fp32_bf16_vec4_retarget2 > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_retarget2 vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_retarget2) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_retarget2 > 0 && ms_fp32_bf16_vec4 > 0)
        std::printf("    -- fp32_bf16_vec4_retarget2 vs fp32_bf16_vec4: %.3fx\n", (flops / ms_fp32_bf16_vec4_retarget2) / (flops / ms_fp32_bf16_vec4));
    if (ms_fp32_bf16_vec4_dispatch > 0 && ms_cublas > 0)
        std::printf("    -- fp32_bf16_vec4_dispatch vs cublas: %.3fx\n", (flops / ms_fp32_bf16_vec4_dispatch) / (flops / ms_cublas));
    if (ms_fp32_bf16_vec4_dispatch > 0 && ms_fp32_bf16_vec4 > 0)
        std::printf("    -- fp32_bf16_vec4_dispatch vs fp32_bf16_vec4: %.3fx\n", (flops / ms_fp32_bf16_vec4_dispatch) / (flops / ms_fp32_bf16_vec4));

    if (ok_xy) {
        cudaFree(x);
        cudaFree(y);
    }
}

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
    std::printf("=== TRANSPOSE GEMV: TOMOS FP32 / FP16 / BF16 vs cuBLAS Sgemv (CUBLAS_OP_T) ===\n");

    const std::vector<size_t> square_sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
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

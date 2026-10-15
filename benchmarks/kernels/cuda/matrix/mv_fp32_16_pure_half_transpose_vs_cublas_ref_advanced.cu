// "_advanced" variant of mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu
// -- byte-for-byte identical except for time_call's own timing
// methodology and two new CLI flags, --warmup N and --iters N (both
// default 10, see main()'s own argv-filtering preamble): instead of
// this file's own sibling's fixed, shape-size-dependent `reps` heuristic
// (a single un-timed warmup call, then `reps` timed calls averaged),
// this variant always does g_warmup_runs UN-TIMED warmup calls first,
// then times g_iterations calls as one block and divides by g_
// iterations -- both counts fully caller-controlled, independent of
// shape. No kernel call site, shape list, or dispatch logic differs
// from the sibling file.
//
// Standalone throughput comparison for the TRANSPOSED (y = alpha*A^T*x +
// beta*y) GENUINE 16-BIT COMPUTE tier of GEMV -- distinct from every
// other "fp32_16"/"fp32_bf16" benchmark in this codebase, which all
// decode packed 16-bit halves back to full fp32 BEFORE computing (the
// "full fp32 reconstruction" tier mv_fp32_transpose_recon_vs_cublas_
// ref.cu compares). Here A is stored as plain __half bit patterns (one
// uint16_t per element, no reconstruction pair) and the multiply-
// accumulate itself happens in __half/__half2 arithmetic -- this
// codebase's own existing "pure_half" kernel family (src/kernels/cuda/
// matrix/fp32_16_cuda_kernels.cu), built for NO_TRANSPOSE originally but
// (per that family's own host-wrapper trans-handling, confirmed by
// reading every mv_fp32_16_pure_half* wrapper) ALREADY genuinely
// TRANSPOSE-capable for the two kernels compared here -- no new __global__
// kernel was needed for this file, only new benchmark wiring.
//
// Two TOMOS series:
//   TOMOS FP32/FP16 (pure)      = mv_fp32_16_pure_half (baseline: plain
//                                  wide/tiled/narrow_vec dispatch, scalar
//                                  __half compute)
//   TOMOS FP32/FP16 (pure,      = mv_fp32_16_pure_half_wide_ilp_shared_
//   half2)                        half2 (EXPERIMENTAL: same dispatch,
//                                  but the wide/split-K path packs 2
//                                  scalar half FMAs into one __hfma2 --
//                                  an existing comment in fp32_16_cuda_
//                                  kernels.cu calls this "the current
//                                  best design (0.84 of cuBLAS)", though
//                                  that number predates this file (a
//                                  different shape sweep/cuBLAS baseline)
//                                  and is NOT assumed accurate here --
//                                  this file measures it fresh)
// plus cuBLAS as the SAME real fp16 baseline this codebase already
// established elsewhere (verify_mv_fp32_16_gemv.cu, bench_mv_cuda_
// kernels.cu, compare_precision.m all use it, and all state the same
// reasoning): cuBLAS has no native half-precision Sgemv (no
// cublasHgemv), so cublasHgemm with n=1 -- the standard matrix-vector-
// as-matrix-matrix workaround -- is the real fp16 GEMV baseline, NOT
// cublasSgemv (which is what mv_fp32_transpose_recon_vs_cublas_ref.cu's
// own "cublas (Sgemv)" column measures -- the two files' cuBLAS numbers
// are NOT directly comparable, by design, since they benchmark different
// cuBLAS entry points for different precision tiers).
//
// A semantics note the two TOMOS series and the cuBLAS series do NOT
// share: the TOMOS pure_half kernels take float* x/y (only A is
// __half-encoded; x/y stay fp32 throughout, matching every other
// "fp32_16"/"fp32_bf16" kernel's own x/y convention in this codebase) --
// but cublasHgemm requires ALL of A/x/y to be __half (there is no mixed-
// precision Hgemm variant), so the cuBLAS leg uses its own, entirely
// separate __half-typed A_h/x_h/y_h buffers, independently generated
// (same convention verify_mv_fp32_16_gemv.cu's own bench_one already
// uses for this exact comparison) -- NOT a converted copy of the TOMOS
// legs' own float x/y or wh-encoded A. This means the cuBLAS leg is
// measuring a strictly MORE aggressive (fully half-precision, including
// the vectors) computation than the TOMOS legs -- worth keeping in mind
// when reading the ratios, same caveat compare_precision.m's own header
// already carries for this identical comparison.
//
// Correctness: NOT re-tested here (throughput-only, same convention
// every *_vs_cublas_ref.cu/*_vs_cutlass_ref.cu file in this codebase
// follows) -- both mv_fp32_16_pure_half and mv_fp32_16_pure_half_wide_
// ilp_shared_half2 already have EXISTING TRANSPOSE correctness coverage
// in verify_mv_fp32_16_gemv.cu's own run_correctness_case (which loops
// over BOTH TransposeType directions for every case in its own `cases`
// list, non-square shapes included -- e.g. mv_fp32_16_pure_half_wide_
// ilp_shared_half2's own check there, alongside every other pure_half
// sibling) -- this predates this file and needed no new wiring.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu/
// *_vs_cutlass_ref.cu file in this codebase. See scripts/benchmarks_
// gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh.
//
// Usage:
//   mv_fp32_16_pure_half_transpose_vs_cublas_ref
//       Sweeps two shape families, SAME lists mv_fp32_transpose_recon_
//       vs_cublas_ref.cu uses (for exactly the same cases, per request):
//       square M=K=1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
//       32768/49152/65536/131072, then a tall-and-thin sweep with K held
//       fixed at 512 and then 4096, M swept across 1024/2048/4096/8192/
//       16384/32768/65536/131072/262144/524288/1048576.
//   mv_fp32_16_pure_half_transpose_vs_cublas_ref 8192
//       Just that one square shape (M=K=8192).
//   mv_fp32_16_pure_half_transpose_vs_cublas_ref --tall 512
//       Just the tall-and-thin sweep at K=512, M swept across the
//       default size list.
//   mv_fp32_16_pure_half_transpose_vs_cublas_ref --tall 512 1048576
//       Just the single tall-and-thin shape M=1048576, K=512.

#include <cublas_v2.h>
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
void mv_fp32_16_pure_half(TransposeType trans, double alpha, dim<2> size,
                          const uint16_t* wh, size_t ld_a, const float* x,
                          size_t ld_x, double beta, float* y, size_t ld_y);
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2(TransposeType trans,
                                                double alpha, dim<2> size,
                                                const uint16_t* wh,
                                                size_t ld_a, const float* x,
                                                size_t ld_x, double beta,
                                                float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                    const uint16_t* wh, size_t ld_a, const float* x,
                                                                    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(TransposeType trans, double alpha,
                                                                        dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                        const float* x, size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);
void mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
namespace experimental { void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock(TransposeType trans, double alpha, dim<2> size,
                                                           const uint16_t* wh, size_t ld_a, const float* x,
                                                           size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                const uint16_t* wh, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental

void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, size_t ld_x, double beta,
                                                                       float* y, size_t ld_y);
namespace experimental { void mv_fp32_16_pure_half_transpose_wmma(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                          size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                          size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const __half* x, size_t ld_x,
    double beta, __half* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                                size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                                size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_transpose_warp_owner_split_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner_split(TransposeType trans, double alpha, dim<2> size,
                                                      const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
                                                      double beta, float* y, size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_transpose_warp_owner_split_f32_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner_split_f32(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, size_t ld_a, const float* x,
                                                          size_t ld_x, double beta, float* y, size_t ld_y,
                                                          float* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size); }  // namespace experimental

void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                         dim<2> size, const uint16_t* wh,
                                                                         size_t ld_a, const float* x, size_t ld_x,
                                                                         double beta, float* y, size_t ld_y,
                                                                         float* workspace);
namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              size_t ld_a, const float* x,
                                                                              size_t ld_x, double beta, float* y,
                                                                              size_t ld_y, float* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace); }  // namespace experimental

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

// wr is written but never read by any kernel in this file (mv_fp32_16_
// pure_half* only ever consumes wh -- see fp32_16_half_only's own
// comment in fp32_16_pack.hpp) -- fp32_16_split still requires a wr
// output parameter, so a scratch buffer is allocated, filled, and freed
// immediately, same convention bench_fp32_16_cuda_kernels.cu's own
// make_device_wh_wr already uses for this exact family.
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

// Safe-normal-range variant of pseudo_rand -- required by "TOMOS FP32/
// FP16 (fast)" (mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
// vec4_fast): fp32_16_fast_half decodes unconditionally as if control
// were always 0, so its own data must PROVABLY never be extended/
// degraded-encoded -- same convention as mv_fp32_transpose_recon_vs_
// cublas_ref.cu's own pseudo_rand_safe_normal (copied rather than
// shared -- standalone self-contained binaries, no shared translation
// unit between them).
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

// Plain (not safe-range) generator for mv_fp32_bf16_pure_bf16_transpose_
// atomiconly_scaled_mblock_vec4 ("TOMOS FP32/BF16") -- unlike every fp16
// "_fast" sibling in this file, that kernel has no unsafe-on-arbitrary-
// data caveat (bf16's truncation decode is always valid -- see its own
// comment in fp32_16_cuda_kernels.cu), so plain pseudo_rand data
// suffices, same convention as gen_fp32_16_wh above.
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

// Mixed normal/extended-range generator for mv_fp32_16_hybrid_
// transpose_atomiconly_scaled_mblock ("test") -- ~10% of values get a
// magnitude beyond fp16's own ~65504 max (control=1, exercising that
// kernel's own wh-only extended-exponent reconstruction path), the rest
// stay in [-1,1] (control=0, the cheap __half path) -- same convention
// verify_mv_fp32_16_gemv.cu's own hybrid correctness check uses; plain
// pseudo_rand data (uniform in [-1,1]) would never trigger the branch
// this kernel exists to test at all.
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
// TRANSPOSE (y = alpha*A^T*x + beta*y): x has length M (the reduction
// dimension), y has length K (the output dimension) -- same convention
// mv_fp32_transpose_recon_vs_cublas_ref.cu's own run_shape uses. A's own
// representation (wh for the TOMOS legs, A_h/x_h/y_h for the cuBLAS leg)
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

    double ms_pure_half = -1.0, ms_pure_half_ilp_shared_half2 = -1.0, ms_pure_half_atomiconly = -1.0,
           ms_pure_half_atomiconly_retuned_floor = -1.0, ms_pure_half_wmma = -1.0,
           ms_pure_half_atomiconly_half_io_retuned_floor = -1.0, ms_pure_half_warp_owner = -1.0,
           ms_pure_half_warp_owner_split = -1.0, ms_pure_half_warp_owner_split_f32 = -1.0,
           ms_pure_half_atomiconly_vec4_split = -1.0, ms_pure_half_atomiconly_vec4_split_retarget = -1.0,
           ms_pure_half_atomiconly_vec4_split_retarget_lo = -1.0, ms_pure_half_atomiconly_vec4_fast = -1.0,
           ms_pure_half_atomiconly_vec4_split_fast = -1.0, ms_fp32_bf16_pure_bf16 = -1.0, ms_cublas_hgemm = -1.0,
           ms_cublas_gemmex_bf16 = -1.0, ms_hybrid = -1.0, ms_hybrid_vec4 = -1.0, ms_hybrid_vec4_paired = -1.0;

    float *x = nullptr, *y = nullptr;
    bool ok_xy = alloc_or_skip(reinterpret_cast<void**>(&x), m * sizeof(float), "x", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&y), k * sizeof(float), "y", shape_desc);
    if (ok_xy) {
        gen_float(m, x);
        cudaMemset(y, 0, k * sizeof(float));
        cudaDeviceSynchronize();

        uint16_t *wh = nullptr, *wr_scratch = nullptr;
        bool ok_wh = alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t),
                                   "mv_fp32_16_pure_half (wh)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&wr_scratch), m * k * sizeof(uint16_t),
                                  "mv_fp32_16_pure_half (wr scratch)", shape_desc);
        if (ok_wh) {
            gen_fp32_16_wh(m * k, wh, wr_scratch);
            cudaDeviceSynchronize();
            cudaFree(wr_scratch);

            // __half-typed x/y, own transient buffers alongside wh
            // (freed at the end of this same block) -- needed only by
            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_half_io_retuned_floor below, NOT shared with the
            // separate cuBLAS leg's own A_h/x_h/y_h further down (that
            // block generates its own independent __half data, same
            // convention as every other cuBLAS-vs-TOMOS comparison in
            // this codebase).
            __half *x_h = nullptr, *y_h = nullptr;
            bool ok_h_io = alloc_or_skip(reinterpret_cast<void**>(&x_h), m * sizeof(__half),
                                        "mv_fp32_16_pure_half_..._half_io (x_h)", shape_desc) &&
                          alloc_or_skip(reinterpret_cast<void**>(&y_h), k * sizeof(__half),
                                        "mv_fp32_16_pure_half_..._half_io (y_h)", shape_desc);
            if (ok_h_io) {
                gen_half(m, x_h);
                cudaMemset(y_h, 0, k * sizeof(__half));
                cudaDeviceSynchronize();
            }

            // "TOMOS FP32/FP16 (pure)" -- mv_fp32_16_pure_half's own
            // TRANSPOSE path (mv_fp32_16_pure_half_wide_kernel/_tiled_
            // kernel/_narrow_vec_kernel, dispatched by shape same as
            // every other tiled/narrow_vec/wide family in this codebase
            // -- see fp32_16_cuda_kernels.cu). Genuine __half arithmetic:
            // wh is decoded via fp32_16_half_only (a direct bit
            // reinterpret, no reconstruction pair) and every FMA in the
            // reduction happens in __half, only widened to double/float
            // at the final accumulate/store.
            ms_pure_half = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_pure_half(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda),
                                           x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half   = %9.3fms (%10.1f GFlop/s)\n", ms_pure_half,
                       flops / ms_pure_half / 1e6);

            // "TOMOS FP32/FP16 (pure, half2)" -- mv_fp32_16_pure_half_
            // wide_ilp_shared_half2's own TRANSPOSE path: EXACT SAME
            // dispatch/tiled/narrow_vec paths as the leg just above,
            // only the wide (split-K) path's kernel differs -- packs the
            // 4-way unrolled accumulators into two __half2 values and
            // uses __hfma2 (2 scalar half FMAs per instruction) instead
            // of 4 separate __hfma calls. Same wh/x/y data. EXPERIMENTAL
            // -- an existing comment elsewhere in fp32_16_cuda_kernels.cu
            // calls this "the current best design (0.84 of cuBLAS)," but
            // that number is from a different (NO_TRANSPOSE, and/or
            // cublasSgemv-baselined) measurement -- not assumed accurate
            // for this file's own TRANSPOSE-vs-cublasHgemm comparison.
            ms_pure_half_ilp_shared_half2 = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_wide_ilp_shared_half2(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh,
                                                                  static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_wide_ilp_shared_half2 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_pure_half_ilp_shared_half2, flops / ms_pure_half_ilp_shared_half2 / 1e6);

            // "TOMOS FP32/FP16 (pure, atomiconly)" -- mv_fp32_16_pure_
            // half_transpose_atomiconly_scaled_mblock_vec4's own NEW
            // standalone TRANSPOSE kernel: same warp-per-output-column/
            // split-M/atomicAdd/shared-x-staging structure as the fp32-
            // reconstruction tier's own TRANSPOSE kernel family, but
            // decoding via fp32_16_half_only and computing in __hfma2 on
            // ushort4-loaded wh -- a GENUINELY vectorized wh load, unlike
            // the leg just above (whose own wh load is two independent
            // scalar reads, only the arithmetic is packed) -- see its own
            // comment in fp32_16_cuda_kernels.cu for the full derivation,
            // including the randls2.0 cross-check that motivated it. Same
            // wh/x/y data.
            //
            // MEASURED on real A100 hardware (square M=K, M=1024-65536):
            // a dramatic win over both legs above -- 3.5x-4.5x faster
            // than plain/half2 at every size, reaching 0.95x-1.06x of
            // cuBLAS Hgemm at M=1024-2048/24576-65536, with a real dip
            // to 0.59x-0.79x at M=3072-8192 -- see _retuned_floor below,
            // built specifically to close that dip.
            ms_pure_half_atomiconly = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                       ms_pure_half_atomiconly, flops / ms_pure_half_atomiconly / 1e6);

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_retuned_floor -- EXPERIMENTAL, standalone A/B sibling
            // reusing the EXACT SAME __global__ kernel as the leg just
            // above, only a more aggressive M_BLOCK-scaling target (2048
            // instead of 512) with a floor (256) -- both already
            // validated for the fp32_16 reconstruction tier's own
            // TRANSPOSE family, reused directly since the M_BLOCK-
            // scaling formula is decode-independent. Targets the real
            // M=3072-8192 dip the leg above measured -- see its own
            // comment in fp32_16_cuda_kernels.cu. Same wh/x/y data.
            ms_pure_half_atomiconly_retuned_floor = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf(
                "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor = %9.3fms (%10.1f GFlop/s)\n",
                ms_pure_half_atomiconly_retuned_floor, flops / ms_pure_half_atomiconly_retuned_floor / 1e6);

            // mv_fp32_16_pure_half_transpose_wmma -- standalone Tensor
            // Core (WMMA) TRANSPOSE kernel. CONFIRMED, SEVERE NEGATIVE
            // RESULT on real A100 hardware: 0.043x of cuBLAS, WORSE than
            // even the NO_TRANSPOSE wmma kernel's own confirmed 0.14x,
            // and 15x slower than _retuned_floor at the same shape --
            // see its own comment in fp32_16_cuda_kernels.cu. Same
            // wh/x/y data.
            ms_pure_half_wmma = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_transpose_wmma(TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh,
                                                           static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_transpose_wmma = %9.3fms (%10.1f GFlop/s)\n", ms_pure_half_wmma,
                       flops / ms_pure_half_wmma / 1e6);

            // "TOMOS FP32/FP16 (pure, atomiconly, retuned_floor, half
            // I/O)" -- mv_fp32_16_pure_half_transpose_atomiconly_
            // scaled_mblock_vec4_half_io_retuned_floor's own kernel:
            // IDENTICAL compute to _retuned_floor above, only x and y
            // are __half* instead of float* -- matching cuBLAS's own
            // Hgemm(n=1) semantics exactly (A, x, AND y all __half),
            // closing the semantics-mismatch caveat this file's own
            // header comment notes for every other TOMOS leg here. Own
            // x_h/y_h data (allocated above, alongside wh), same wh.
            if (ok_h_io) {
                ms_pure_half_atomiconly_half_io_retuned_floor = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x_h, 1, 0.3, y_h,
                        1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_atomiconly_half_io_retuned_floor,
                    flops / ms_pure_half_atomiconly_half_io_retuned_floor / 1e6);

                cudaFree(x_h);
                cudaFree(y_h);
            }

            // mv_fp32_16_pure_half_transpose_warp_owner -- standalone
            // "warp owner, no atomics" TRANSPOSE kernel, ported from
            // randls2.0's own gemv_fp16_cm_T_warp_owner (dense_kernels.
            // cu:7533). ENTIRELY different grid/reduction strategy from
            // every other pure_half leg above: one warp owns a whole
            // output column for its whole lifetime (no M-splitting, no
            // atomics, no M_BLOCK-scaling), matching this file's own
            // mv_fp32_16_transpose_warpdot_tiled_fast_normal design one
            // tier up. That reconstruction-tier analog measured a real
            // -18.7% to -25.7% REGRESSION vs the atomicAdd/M_BLOCK-scaled
            // design -- see this kernel's own comment in fp32_16_cuda_
            // kernels.cu for why the cheaper pure_half decode and a
            // genuinely vectorized wh load make this a materially
            // different case, not a rerun of that same experiment. Same
            // wh/x/y data.
            ms_pure_half_warp_owner = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner(
                    TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_pure_half_transpose_warp_owner = %9.3fms (%10.1f GFlop/s)\n",
                       ms_pure_half_warp_owner, flops / ms_pure_half_warp_owner / 1e6);

            // mv_fp32_16_pure_half_transpose_warp_owner_split -- split-
            // then-reduce counterpart of the leg just above, fixing its
            // real cause of a worst-of-everything result (grid fixed at
            // (1, ceil(k/8)), never scales with M) by reusing the same
            // M_BLOCK-scaling target/floor already validated for this
            // pure_half tier's own retuned_floor kernel, combining
            // blocks' partial sums via a scratch buffer + reduce instead
            // of atomics -- see its own comment in fp32_16_cuda_
            // kernels.cu. Caller-provided workspace, allocated/freed
            // outside the timed loop (standing requirement this
            // session). Same wh/x/y data.
            size_t warp_owner_split_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_workspace_size(dim<2>{m, k});
            double* warp_owner_split_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&warp_owner_split_ws),
                              warp_owner_split_ws_elems * sizeof(double), "pure_half_transpose_warp_owner_split (ws)",
                              shape_desc)) {
                ms_pure_half_warp_owner_split = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        warp_owner_split_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_transpose_warp_owner_split = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_warp_owner_split, flops / ms_pure_half_warp_owner_split / 1e6);
                cudaFree(warp_owner_split_ws);
            }

            // mv_fp32_16_pure_half_transpose_warp_owner_split_f32 --
            // float-scratch-buffer counterpart of the leg just above,
            // testing whether the extra write+readback bandwidth a
            // double partial buffer costs (vs atomicAdd, which pays
            // none) explains _split's own measured trailing-off at the
            // largest M tested -- see its own comment in fp32_16_cuda_
            // kernels.cu. Same wh/x/y data, own float workspace
            // allocated/freed outside the timed loop.
            size_t warp_owner_split_f32_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_f32_workspace_size(dim<2>{m, k});
            float* warp_owner_split_f32_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&warp_owner_split_f32_ws),
                              warp_owner_split_f32_ws_elems * sizeof(float),
                              "pure_half_transpose_warp_owner_split_f32 (ws)", shape_desc)) {
                ms_pure_half_warp_owner_split_f32 = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_transpose_warp_owner_split_f32(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        warp_owner_split_f32_ws);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_transpose_warp_owner_split_f32 = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_warp_owner_split_f32, flops / ms_pure_half_warp_owner_split_f32 / 1e6);
                cudaFree(warp_owner_split_f32_ws);
            }

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_split -- atomiconly_scaled_mblock_vec4's own compute
            // body verbatim (ushort4 loads, __hfma2 accumulate), combine
            // step swapped to split+reduce -- isolates atomics vs
            // split+reduce as the one remaining variable after ruling
            // out both the scratch buffer type and the weaker warp_
            // owner-style compute body as the cause of the trailing-off
            // at large M -- see its own comment in fp32_16_cuda_
            // kernels.cu. Same wh/x/y data, own float workspace.
            size_t vec4_split_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                    dim<2>{m, k});
            float* vec4_split_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&vec4_split_ws), vec4_split_ws_elems * sizeof(float),
                              "pure_half_transpose_atomiconly_scaled_mblock_vec4_split (ws)", shape_desc)) {
                ms_pure_half_atomiconly_vec4_split = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        vec4_split_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_atomiconly_vec4_split, flops / ms_pure_half_atomiconly_vec4_split / 1e6);
                cudaFree(vec4_split_ws);
            }

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_split_retarget -- reuses the EXACT SAME __global__
            // kernel as the leg just above, only the M_BLOCK-scaling
            // target differs (2176 vs 2048) -- a grid-math-derived,
            // near-single-variable retune targeting the M=3072-8192 dip
            // (the worst gap to cuBLAS Hgemm once _split matched
            // _retuned_floor everywhere else) -- see its own comment in
            // fp32_16_cuda_kernels.cu. Same wh/x/y data, own float
            // workspace.
            size_t vec4_split_retarget_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_workspace_size(
                    dim<2>{m, k});
            float* vec4_split_retarget_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&vec4_split_retarget_ws),
                              vec4_split_retarget_ws_elems * sizeof(float),
                              "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget (ws)",
                              shape_desc)) {
                ms_pure_half_atomiconly_vec4_split_retarget = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        vec4_split_retarget_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_atomiconly_vec4_split_retarget,
                    flops / ms_pure_half_atomiconly_vec4_split_retarget / 1e6);
                cudaFree(vec4_split_retarget_ws);
            }

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_split_retarget_lo -- tests the OPPOSITE (fewer-
            // blocks) hypothesis from the leg just above -- see its own
            // comment in fp32_16_cuda_kernels.cu. Same wh/x/y data, own
            // float workspace.
            size_t vec4_split_retarget_lo_ws_elems =
                cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo_workspace_size(
                    dim<2>{m, k});
            float* vec4_split_retarget_lo_ws = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&vec4_split_retarget_lo_ws),
                              vec4_split_retarget_lo_ws_elems * sizeof(float),
                              "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo (ws)",
                              shape_desc)) {
                ms_pure_half_atomiconly_vec4_split_retarget_lo = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh, static_cast<size_t>(lda), x, 1, 0.3, y, 1,
                        vec4_split_retarget_lo_ws);
                }, reps);
                std::printf(
                    "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo = %9.3fms (%10.1f GFlop/s)\n",
                    ms_pure_half_atomiconly_vec4_split_retarget_lo,
                    flops / ms_pure_half_atomiconly_vec4_split_retarget_lo / 1e6);
                cudaFree(vec4_split_retarget_lo_ws);
            }

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_fast -- "TOMOS FP32/FP16 (fast)". UNSAFE on arbitrary
            // data (fp32_16_fast_half's own contract: every wh value
            // must already have control=0), so this leg uses its OWN
            // safe-range wh (mag in [0.5,1.0], via gen_fp32_16_wh_safe)
            // instead of the plain wh every other leg above shares --
            // see its own comment in fp32_16_cuda_kernels.cu. Own x/y
            // reused (decode-only difference, x/y semantics unchanged).
            uint16_t *wh_fast = nullptr, *wr_fast_scratch = nullptr;
            bool ok_wh_fast =
                alloc_or_skip(reinterpret_cast<void**>(&wh_fast), m * k * sizeof(uint16_t),
                             "pure_half_transpose_atomiconly_scaled_mblock_vec4_fast (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_fast_scratch), m * k * sizeof(uint16_t),
                             "pure_half_transpose_atomiconly_scaled_mblock_vec4_fast (wr scratch)", shape_desc);
            if (ok_wh_fast) {
                gen_fp32_16_wh_safe(m * k, wh_fast, wr_fast_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_fast_scratch);

                ms_pure_half_atomiconly_vec4_fast = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_fast, static_cast<size_t>(lda), x, 1, 0.3, y,
                        1);
                }, reps);
                std::printf("    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast = %9.3fms (%10.1f GFlop/s)\n",
                           ms_pure_half_atomiconly_vec4_fast, flops / ms_pure_half_atomiconly_vec4_fast / 1e6);

                cudaFree(wh_fast);
            }

            // mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock --
            // "test" for "TOMOS FP32/FP16". Own mixed normal/extended-
            // range wh (via gen_fp32_16_wh_hybrid) -- ~10% of values
            // deliberately beyond fp16's own range, exercising the
            // wh-only extended-exponent reconstruction path this kernel
            // exists to test, unlike the plain uniform-[-1,1] wh every
            // other leg here shares. wr is written by fp32_16_split but
            // never read by this kernel (uses ONLY wh) -- discarded
            // right after generation, same "scratch buffer" convention
            // every other single-array generator in this file uses. See
            // its own comment in fp32_16_cuda_kernels.cu.
            uint16_t *wh_hybrid = nullptr, *wr_hybrid_scratch = nullptr;
            bool ok_wh_hybrid =
                alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid_scratch), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock (wr scratch)", shape_desc);
            if (ok_wh_hybrid) {
                gen_fp32_16_wh_hybrid(m * k, wh_hybrid, wr_hybrid_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_hybrid_scratch);

                ms_hybrid = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid, static_cast<size_t>(lda), x, 1, 0.3,
                        y, 1);
                }, reps);
                std::printf("    mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock = %9.3fms (%10.1f GFlop/s)\n",
                           ms_hybrid, flops / ms_hybrid / 1e6);

                cudaFree(wh_hybrid);
            }

            // mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4
            // -- vectorized-load fix for the leg just above's own
            // measured falloff past M=3072ish. Own mixed normal/
            // extended-range wh (separate generation from the leg
            // above, same distribution). See its own comment in
            // fp32_16_cuda_kernels.cu.
            uint16_t *wh_hybrid_vec4 = nullptr, *wr_hybrid_vec4_scratch = nullptr;
            bool ok_wh_hybrid_vec4 =
                alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid_vec4), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock_vec4 (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid_vec4_scratch), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock_vec4 (wr scratch)", shape_desc);
            if (ok_wh_hybrid_vec4) {
                gen_fp32_16_wh_hybrid(m * k, wh_hybrid_vec4, wr_hybrid_vec4_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_hybrid_vec4_scratch);

                ms_hybrid_vec4 = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid_vec4, static_cast<size_t>(lda), x, 1,
                        0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                    ms_hybrid_vec4, flops / ms_hybrid_vec4 / 1e6);

                cudaFree(wh_hybrid_vec4);
            }

            // mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_
            // paired -- arithmetic-throughput fix for the leg just
            // above: packs adjacent same-control pairs into one real
            // __hfma2 instead of two zero-padded ones. Own mixed
            // normal/extended-range wh. See its own comment in
            // fp32_16_cuda_kernels.cu.
            uint16_t *wh_hybrid_paired = nullptr, *wr_hybrid_paired_scratch = nullptr;
            bool ok_wh_hybrid_paired =
                alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid_paired), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock_vec4_paired (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid_paired_scratch), m * k * sizeof(uint16_t),
                             "hybrid_transpose_atomiconly_scaled_mblock_vec4_paired (wr scratch)", shape_desc);
            if (ok_wh_hybrid_paired) {
                gen_fp32_16_wh_hybrid(m * k, wh_hybrid_paired, wr_hybrid_paired_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_hybrid_paired_scratch);

                ms_hybrid_vec4_paired = time_call(start, stop, [&]() {
                    cuda::mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_hybrid_paired, static_cast<size_t>(lda), x,
                        1, 0.3, y, 1);
                }, reps);
                std::printf(
                    "    mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired = %9.3fms (%10.1f GFlop/s)\n",
                    ms_hybrid_vec4_paired, flops / ms_hybrid_vec4_paired / 1e6);

                cudaFree(wh_hybrid_paired);
            }

            // mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_
            // vec4_split_fast -- "TOMOS FP16 (fast)". Same fast-decode/
            // safe-range-data contract as the vec4_fast leg above, but
            // applied to the split-combine kernel (vec4_split) instead
            // of the atomicAdd one -- isolates the decode win from the
            // combine-strategy variable. Own safe-range wh (separate
            // allocation from wh_fast above -- different M_BLOCK-scaling
            // target could in principle want a different-sized buffer,
            // though here it's the same, kept separate for clarity) and
            // own float workspace, both allocated/generated/freed
            // outside the timed loop. See its own comment in fp32_16_
            // cuda_kernels.cu.
            uint16_t *wh_split_fast = nullptr, *wr_split_fast_scratch = nullptr;
            bool ok_wh_split_fast =
                alloc_or_skip(reinterpret_cast<void**>(&wh_split_fast), m * k * sizeof(uint16_t),
                             "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_split_fast_scratch), m * k * sizeof(uint16_t),
                             "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast (wr scratch)",
                             shape_desc);
            if (ok_wh_split_fast) {
                gen_fp32_16_wh_safe(m * k, wh_split_fast, wr_split_fast_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_split_fast_scratch);

                size_t vec4_split_fast_ws_elems =
                    cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                        dim<2>{m, k});
                float* vec4_split_fast_ws = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&vec4_split_fast_ws),
                                  vec4_split_fast_ws_elems * sizeof(float),
                                  "pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast (ws)",
                                  shape_desc)) {
                    ms_pure_half_atomiconly_vec4_split_fast = time_call(start, stop, [&]() {
                        cuda::experimental::mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast(
                            TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_split_fast, static_cast<size_t>(lda), x,
                            1, 0.3, y, 1, vec4_split_fast_ws);
                    }, reps);
                    std::printf(
                        "    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast = %9.3fms (%10.1f GFlop/s)\n",
                        ms_pure_half_atomiconly_vec4_split_fast, flops / ms_pure_half_atomiconly_vec4_split_fast / 1e6);
                    cudaFree(vec4_split_fast_ws);
                }

                cudaFree(wh_split_fast);
            }

            // mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_
            // vec4 -- "TOMOS FP32/BF16". Unlike every fp16 "_fast"
            // sibling above, this is safe on arbitrary data (see its
            // own comment in fp32_16_cuda_kernels.cu), so plain
            // pseudo_rand-generated wh (via gen_fp32_bf16_wh) suffices --
            // own separate allocation, own x/y reused (decode-only
            // difference vs the fp16 legs).
            uint16_t *wh_bf16 = nullptr, *wr_bf16_scratch = nullptr;
            bool ok_wh_bf16 =
                alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                             "fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 (wh)", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_scratch), m * k * sizeof(uint16_t),
                             "fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 (wr scratch)",
                             shape_desc);
            if (ok_wh_bf16) {
                gen_fp32_bf16_wh(m * k, wh_bf16, wr_bf16_scratch);
                cudaDeviceSynchronize();
                cudaFree(wr_bf16_scratch);

                ms_fp32_bf16_pure_bf16 = time_call(start, stop, [&]() {
                    cuda::mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(
                        TransposeType::TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, static_cast<size_t>(lda), x, 1, 0.3, y,
                        1);
                }, reps);
                std::printf(
                    "    mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 = %9.3fms (%10.1f GFlop/s)\n",
                    ms_fp32_bf16_pure_bf16, flops / ms_fp32_bf16_pure_bf16 / 1e6);

                cudaFree(wh_bf16);
            }

            cudaFree(wh);
        }
    }

    // cuBLAS (Hgemm, n=1) -- the real fp16 GEMV baseline (see this
    // file's own header comment for why cublasSgemv is NOT used here).
    // Independent __half-typed A_h/x_h/y_h, NOT a converted copy of the
    // TOMOS legs' own wh/x/y -- same convention verify_mv_fp32_16_gemv.
    // cu's own bench_one already uses for this identical comparison.
    {
        __half *A_h = nullptr, *x_h = nullptr, *y_h = nullptr;
        bool ok_h = alloc_or_skip(reinterpret_cast<void**>(&A_h), m * k * sizeof(__half), "cublas Hgemm (A_h)",
                                  shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&x_h), m * sizeof(__half), "cublas Hgemm (x_h)",
                                 shape_desc) &&
                   alloc_or_skip(reinterpret_cast<void**>(&y_h), k * sizeof(__half), "cublas Hgemm (y_h)",
                                 shape_desc);
        if (ok_h) {
            gen_half(m * k, A_h);
            gen_half(m, x_h);
            cudaMemset(y_h, 0, k * sizeof(__half));
            cudaDeviceSynchronize();

            __half h_alpha = __double2half(1.7);
            __half h_beta = __double2half(0.3);
            ms_cublas_hgemm = time_call(start, stop, [&]() {
                cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(k), 1, static_cast<int>(m), &h_alpha,
                           A_h, lda, x_h, static_cast<int>(m), &h_beta, y_h, static_cast<int>(k));
            }, reps);
            std::printf("    cublas (Hgemm n=1, TRANSPOSE) = %9.3fms (%10.1f GFlop/s)\n", ms_cublas_hgemm,
                       flops / ms_cublas_hgemm / 1e6);

            cudaFree(A_h);
            cudaFree(x_h);
            cudaFree(y_h);
        }
    }

    // cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1) -- the real bf16 GEMV
    // baseline, same "no native cublasGemv for this dtype" reasoning as
    // cublasHgemm above (there is no cublasBgemv either) -- same
    // CUDA_R_16BF/CUBLAS_COMPUTE_32F/CUBLAS_GEMM_DEFAULT_TENSOR_OP call
    // convention already established in this codebase's own fp16_bf16_
    // pure_vs_cutlass_ref.cu (there for a real N>1 GEMM; here N=1, same
    // "GEMV via GEMM" convention cublasHgemm's own leg above uses).
    // Independent __nv_bfloat16-typed A_bf/x_bf/y_bf, own scope, alpha/
    // beta as float (CUBLAS_COMPUTE_32F's own precision, not bf16).
    {
        __nv_bfloat16 *A_bf = nullptr, *x_bf = nullptr, *y_bf = nullptr;
        bool ok_bf = alloc_or_skip(reinterpret_cast<void**>(&A_bf), m * k * sizeof(__nv_bfloat16),
                                   "cublas GemmEx bf16 (A_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_bf), m * sizeof(__nv_bfloat16),
                                  "cublas GemmEx bf16 (x_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf), k * sizeof(__nv_bfloat16),
                                  "cublas GemmEx bf16 (y_bf)", shape_desc);
        if (ok_bf) {
            gen_bf16(m * k, A_bf);
            gen_bf16(m, x_bf);
            cudaMemset(y_bf, 0, k * sizeof(__nv_bfloat16));
            cudaDeviceSynchronize();

            float alpha_f = 1.7f;
            float beta_f = 0.3f;
            ms_cublas_gemmex_bf16 = time_call(start, stop, [&]() {
                cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(k), 1, static_cast<int>(m), &alpha_f,
                            A_bf, CUDA_R_16BF, lda, x_bf, CUDA_R_16BF, static_cast<int>(m), &beta_f, y_bf,
                            CUDA_R_16BF, static_cast<int>(k), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }, reps);
            std::printf("    cublas (GemmEx CUDA_R_16BF n=1, TRANSPOSE) = %9.3fms (%10.1f GFlop/s)\n",
                       ms_cublas_gemmex_bf16, flops / ms_cublas_gemmex_bf16 / 1e6);

            cudaFree(A_bf);
            cudaFree(x_bf);
            cudaFree(y_bf);
        }
    }

    if (ms_pure_half > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half vs cublas_hgemm: %.3fx\n", (flops / ms_pure_half) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_ilp_shared_half2 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_ilp_shared_half2) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_ilp_shared_half2 > 0 && ms_pure_half > 0)
        std::printf("    -- fp32_16_pure_half_wide_ilp_shared_half2 vs fp32_16_pure_half: %.3fx\n",
                   (flops / ms_pure_half_ilp_shared_half2) / (flops / ms_pure_half));
    if (ms_pure_half_atomiconly > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_atomiconly) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly > 0 && ms_pure_half > 0)
        std::printf("    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 vs fp32_16_pure_half: %.3fx\n",
                   (flops / ms_pure_half_atomiconly) / (flops / ms_pure_half));
    if (ms_pure_half_atomiconly > 0 && ms_pure_half_ilp_shared_half2 > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 vs fp32_16_pure_half_wide_ilp_shared_half2: %.3fx\n",
            (flops / ms_pure_half_atomiconly) / (flops / ms_pure_half_ilp_shared_half2));
    if (ms_pure_half_atomiconly_retuned_floor > 0 && ms_cublas_hgemm > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor vs cublas_hgemm: %.3fx\n",
            (flops / ms_pure_half_atomiconly_retuned_floor) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_retuned_floor > 0 && ms_pure_half_atomiconly > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4: %.3fx\n",
            (flops / ms_pure_half_atomiconly_retuned_floor) / (flops / ms_pure_half_atomiconly));
    if (ms_pure_half_wmma > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_wmma vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_wmma) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_wmma > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_wmma vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_wmma) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_atomiconly_half_io_retuned_floor > 0 && ms_cublas_hgemm > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor vs cublas_hgemm: %.3fx\n",
            (flops / ms_pure_half_atomiconly_half_io_retuned_floor) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_half_io_retuned_floor > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_atomiconly_half_io_retuned_floor) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_warp_owner > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_warp_owner vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_warp_owner) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_warp_owner > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_warp_owner vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_warp_owner) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_warp_owner_split > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_warp_owner_split vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_warp_owner_split) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_warp_owner_split > 0 && ms_pure_half_warp_owner > 0)
        std::printf("    -- fp32_16_pure_half_transpose_warp_owner_split vs fp32_16_pure_half_transpose_warp_owner: %.3fx\n",
                   (flops / ms_pure_half_warp_owner_split) / (flops / ms_pure_half_warp_owner));
    if (ms_pure_half_warp_owner_split > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_warp_owner_split vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_warp_owner_split) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_warp_owner_split_f32 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_warp_owner_split_f32 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_warp_owner_split_f32) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_warp_owner_split_f32 > 0 && ms_pure_half_warp_owner_split > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_warp_owner_split_f32 vs fp32_16_pure_half_transpose_warp_owner_split: %.3fx\n",
            (flops / ms_pure_half_warp_owner_split_f32) / (flops / ms_pure_half_warp_owner_split));
    if (ms_pure_half_warp_owner_split_f32 > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_warp_owner_split_f32 vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_warp_owner_split_f32) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_atomiconly_vec4_split > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_atomiconly_vec4_split) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_vec4_split > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_atomiconly_vec4_split_retarget > 0 && ms_cublas_hgemm > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget vs cublas_hgemm: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_retarget) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_vec4_split_retarget > 0 && ms_pure_half_atomiconly_vec4_split > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_retarget) / (flops / ms_pure_half_atomiconly_vec4_split));
    if (ms_pure_half_atomiconly_vec4_split_retarget_lo > 0 && ms_cublas_hgemm > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo vs cublas_hgemm: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_retarget_lo) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_vec4_split_retarget_lo > 0 && ms_pure_half_atomiconly_vec4_split > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_retarget_lo) / (flops / ms_pure_half_atomiconly_vec4_split));
    if (ms_pure_half_atomiconly_vec4_fast > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast vs cublas_hgemm: %.3fx\n",
                   (flops / ms_pure_half_atomiconly_vec4_fast) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_vec4_fast > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_fast) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_hybrid > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- hybrid_transpose_atomiconly_scaled_mblock vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid) / (flops / ms_cublas_hgemm));
    if (ms_hybrid > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- hybrid_transpose_atomiconly_scaled_mblock vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_hybrid) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_hybrid_vec4 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- hybrid_transpose_atomiconly_scaled_mblock_vec4 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_vec4) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_vec4 > 0 && ms_hybrid > 0)
        std::printf("    -- hybrid_transpose_atomiconly_scaled_mblock_vec4 vs hybrid_transpose_atomiconly_scaled_mblock: %.3fx\n",
                   (flops / ms_hybrid_vec4) / (flops / ms_hybrid));
    if (ms_hybrid_vec4 > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- hybrid_transpose_atomiconly_scaled_mblock_vec4 vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_hybrid_vec4) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_hybrid_vec4_paired > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- hybrid_transpose_atomiconly_scaled_mblock_vec4_paired vs cublas_hgemm: %.3fx\n",
                   (flops / ms_hybrid_vec4_paired) / (flops / ms_cublas_hgemm));
    if (ms_hybrid_vec4_paired > 0 && ms_hybrid_vec4 > 0)
        std::printf(
            "    -- hybrid_transpose_atomiconly_scaled_mblock_vec4_paired vs hybrid_transpose_atomiconly_scaled_mblock_vec4: %.3fx\n",
            (flops / ms_hybrid_vec4_paired) / (flops / ms_hybrid_vec4));
    if (ms_hybrid_vec4_paired > 0 && ms_pure_half_atomiconly_retuned_floor > 0)
        std::printf(
            "    -- hybrid_transpose_atomiconly_scaled_mblock_vec4_paired vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor: %.3fx\n",
            (flops / ms_hybrid_vec4_paired) / (flops / ms_pure_half_atomiconly_retuned_floor));
    if (ms_pure_half_atomiconly_vec4_split_fast > 0 && ms_cublas_hgemm > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast vs cublas_hgemm: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_fast) / (flops / ms_cublas_hgemm));
    if (ms_pure_half_atomiconly_vec4_split_fast > 0 && ms_pure_half_atomiconly_vec4_fast > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_fast) / (flops / ms_pure_half_atomiconly_vec4_fast));
    if (ms_pure_half_atomiconly_vec4_split_fast > 0 && ms_pure_half_atomiconly_vec4_split > 0)
        std::printf(
            "    -- fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split: %.3fx\n",
            (flops / ms_pure_half_atomiconly_vec4_split_fast) / (flops / ms_pure_half_atomiconly_vec4_split));
    if (ms_fp32_bf16_pure_bf16 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_fp32_bf16_pure_bf16) / (flops / ms_cublas_hgemm));
    if (ms_fp32_bf16_pure_bf16 > 0 && ms_pure_half_atomiconly_vec4_fast > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 vs fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast: %.3fx\n",
            (flops / ms_fp32_bf16_pure_bf16) / (flops / ms_pure_half_atomiconly_vec4_fast));
    if (ms_cublas_gemmex_bf16 > 0 && ms_cublas_hgemm > 0)
        std::printf("    -- cublas_gemmex_bf16 vs cublas_hgemm: %.3fx\n",
                   (flops / ms_cublas_gemmex_bf16) / (flops / ms_cublas_hgemm));
    if (ms_fp32_bf16_pure_bf16 > 0 && ms_cublas_gemmex_bf16 > 0)
        std::printf(
            "    -- fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 vs cublas_gemmex_bf16: %.3fx\n",
            (flops / ms_fp32_bf16_pure_bf16) / (flops / ms_cublas_gemmex_bf16));

    if (ok_xy) {
        cudaFree(x);
        cudaFree(y);
    }
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
    std::printf("=== TRANSPOSE GEMV: TOMOS FP32/FP16 (pure 16-bit compute) vs cuBLAS Hgemm(n=1) ===\n");

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

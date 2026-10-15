// Standalone L2-relative-error analysis for the same five "TOMOS" 16-bit-
// storage GEMV series scripts/plot_gemv_fp32_16_pure_half_vs_cublas_ref.m
// plots, across mv_fp32_16_pure_half_vs_cublas_ref.cu's own SQUARE_MS
// sweep (M=K) -- measures how far each 16-bit-storage method's result
// drifts from a genuine FP32-precision reference, not a throughput
// number. See scripts/plot_gemv_fp32_16_l2_error_vs_cublas_ref.m for the
// resulting plot.
//
// The reference for each method is cublasSgemv (real fp32 GEMV, no
// storage compression) computed on a GENUINE float32 copy of that
// method's OWN underlying data, generated directly on the device --
// never materializing the full M*K matrix on the HOST, which would OOM
// well before this sweep's own largest shape (see verify_gemv_fp32_16_
// plot_promoted.cu's own header for the exact 65536x65536 crash that
// motivated keeping that correctness check's own shapes far smaller than
// this file's). Device memory is still finite, though -- alloc_or_skip
// below (same convention mv_fp32_16_pure_half_vs_cublas_ref.cu's own
// performance sweep uses) gracefully SKIPs any method whose data doesn't
// fit at the largest shapes (131072x131072 needs ~68GB for a single
// float32 array) rather than crashing.
//
// pseudo_rand (and its hybrid-mixed/safe-normal siblings) is a pure
// function of the linear index alone, with no external RNG state -- so a
// plain gen_float_kernel and gen_fp32_16_wh_kernel called with the SAME n
// are GUARANTEED to correspond to the same underlying "true" values at
// every index, without ever decoding wh back to float. Each method's own
// generator pair (float ground-truth, packed wh) is copied from
// mv_fp32_16_pure_half_vs_cublas_ref.cu's own generators (standalone
// self-contained binaries, no shared translation unit between them --
// same convention every *_vs_cublas_ref.cu file in this codebase
// follows), plus two NEW float-ground-truth siblings (gen_float_hybrid,
// gen_float_safe) for the two methods whose own packed-data generator
// isn't plain pseudo_rand. pseudo_rand_hybrid_mixed specifically is NOT
// a verbatim copy -- its own extended-magnitude range needed correcting
// to actually fit what the hybrid encoding can represent; see its own
// comment below for the real bug this caught.
//
// TOMOS FP32/BF16 and TOMOS BF16 share the SAME plain data (bf16's
// truncation decode has no unsafe-on-arbitrary-data caveat, unlike the
// fp16 "(fast)" leg) -- and so the SAME float32 ground truth / cublasSgemv
// reference as TOMOS FP16 (all three ultimately trace back to the same
// plain pseudo_rand generator), reused rather than regenerated.
//
// x (float, length K) is never compressed by any of these methods, so a
// SINGLE shared x is used across all five, generated once per shape.
// alpha=1.0, beta=0.0 (y = A*x exactly, no epilogue scaling to fold into
// the error analysis).
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST but only over the OUTPUT vector y (length M) -- never the M*K
// matrix, which is why this sweep can reach the same 131072x131072 scale
// the performance sweep does.
//
// Usage: mv_fp32_16_l2_error_vs_cublas_ref [normal_scale [M]]
//   No args: runs the same SQUARE_MS sweep as mv_fp32_16_pure_half_vs_
//            cublas_ref.cu's own square sweep (1024 through 131072), at
//            normal_scale=1.0 (every method's "normal" data in [-1,1],
//            the original/default magnitude range).
//   normal_scale: runs the full SQUARE_MS sweep with every method's own
//            "normal"-range data widened to [-scale,scale] instead (see
//            run_shape's own comment for exactly which data this does and
//            doesn't affect) -- intended to be a power of 2, kept well
//            under the ACCUMULATED row-sum's own safe range for the fp16-
//            accumulating series (a far tighter bound than fp16's largest
//            finite magnitude (65504) alone -- see main()'s own comment
//            just below for why). Used to sweep how L2 error changes as
//            data moves from near-1.0 magnitudes toward fp16's own
//            representable ceiling, still entirely within its NORMAL
//            (non-extended/degraded) range -- see scripts/benchmarks_
//            gemv_fp32_16_l2_error_vs_cublas_ref.sh for the actual scale
//            tiers swept and why.
//   normal_scale M: runs just that one square shape (M=K=M) at that scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemv_fp32_16_l2_error_vs_
// cublas_ref.sh.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                             double alpha, dim<2> size,
                                                             const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta,
                                                             float* y, size_t ld_y);
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(TransposeType trans,
                                                                     double alpha, dim<2> size,
                                                                     const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta,
                                                                     float* y, size_t ld_y);
void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(TransposeType trans, double alpha,
                                                                     dim<2> size, const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(TransposeType trans, double alpha,
                                                         dim<2> size, const uint16_t* wh,
                                                         size_t ld_a, const float* x,
                                                         size_t ld_x, double beta, float* y,
                                                         size_t ld_y);
void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y);
// "TOMOS FP16" in the GEMV PERFORMANCE plots (scripts/plot_gemv_fp32_16_pure_
// half_vs_cublas_ref.m): the native-fp16-INPUT kernel -- A is a plain
// __half array (no FP32_16 packing, nothing to decode), x/y float. Same
// definition mv_fp32_16_pure_half_vs_cublas_ref.cu's own "TOMOS FP16"
// series uses (its native_storage leg).
// The narrow_vec kernel LSQR's implicit NO_TRANSPOSE matvec runs at large M
// (plain wrapper, at every M here -- the square shapes below stay under the
// dispatcher's kMvWideRowThreshold, which would route elsewhere), and its
// periodic-flush siblings that add the per-row __half accumulators into
// float every 64 / 256 columns (see fp32_16_cuda_kernels.cu).
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_nomask_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);
void mv_fp32_16_pure_half_narrow_vec_flush64(TransposeType trans, double alpha, size_t m, size_t n,
                                             const uint16_t* wh, size_t ld_a, const float* x, double beta,
                                             float* y);
void mv_fp32_16_pure_half_narrow_vec_flush256(TransposeType trans, double alpha, size_t m, size_t n,
                                              const uint16_t* wh, size_t ld_a, const float* x, double beta,
                                              float* y);
void mv_fp32_16_pure_half_narrow_vec_flush256_swar(TransposeType trans, double alpha, size_t m, size_t n,
                                                   const uint16_t* wh, size_t ld_a, const float* x,
                                                   double beta, float* y);
// mv_fp16_native_mixed (mv_cuda_kernels.cu): the kernel lsqr_precision_
// runner's --explicit-duplicates actually dispatches to (the (CUDA, FP16,
// FP32, FP32, compute=FP16) slot in dense_registries.cpp) -- A native
// __half, x/y float. Its own NO_TRANSPOSE dispatch (rows_op < 8192: half2
// kernel, else the tiled atomic kernel that flushes each 256-column
// __half2 partial sum into a double) is NOT the same structure as the
// performance plots' "TOMOS FP16" (native_storage_fastequiv) above, which
// duplicates the packed kernels' col-split tree -- so the two can have
// different accumulation error, which is why both are measured here.
void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
namespace experimental {
void mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best(TransposeType trans, double alpha,
                                                                        dim<2> size, const __half* A,
                                                                        size_t ld_a, const float* x,
                                                                        size_t ld_x, double beta, float* y,
                                                                        size_t ld_y);
}  // namespace experimental
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

// Safe-normal-range variant, matching mv_fp32_16_pure_half_wide_ilp_
// shared_fasthalf's own contract (fp32_16_fast_half decodes unconditionally
// as if control were always 0, so its own data must PROVABLY never be
// extended/degraded-encoded) -- copied verbatim from mv_fp32_16_pure_half_
// vs_cublas_ref.cu's own pseudo_rand_safe_normal.
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
}

// Mixed normal/extended-range variant, matching mv_fp32_16_hybrid_wide_
// ilp_shared's own contract (~10% of values get a magnitude beyond fp16's
// own ~65504 max, exercising the wh-only extended-exponent reconstruction
// path; plain pseudo_rand data would never trigger it).
//
// NOT copied verbatim from mv_fp32_16_pure_half_vs_cublas_ref.cu's own
// pseudo_rand_hybrid_mixed -- that generator's own extended-magnitude
// range, [1e5, ~1e8], reaches ~24x beyond what fp32_16_pack.hpp's hybrid
// encoding can actually represent: the encodable overflow-direction
// window is exp32 in [kFp32ExpMax, kFp32ExpMax+kOffsetMax] = [142,148],
// i.e. magnitude in [2^16, 2^22) = [65536, 4194304) -- anything beyond
// that falls into the "degraded" bucket, which mv_fp32_16_hybrid_
// process_element_branchless (the kernel this generates data for) drops
// to a ZERO contribution entirely (a legitimate, documented tradeoff for
// real data, where degraded values are meant to be rare -- see that
// function's own comment). Across [1e5,1e8], ~96% of the range lies past
// the 4194304 encodable ceiling, so the vast majority of "extended"
// values -- also the LARGEST-magnitude terms in the whole row sum --
// were being silently dropped by the kernel while cublasSgemv's own
// reference kept their full (up to ~1e8) magnitude, driving L2 relative
// error to ~1.0 for every shape (a real bug this file's own first run
// caught, not a kernel bug: that generator was only ever previously used
// for TIMING, where an out-of-range/degraded-dropped value costs the same
// as an in-range one, so this was invisible until a result was actually
// checked against a reference). This magnitude range instead stays
// strictly inside [65536, 4194304), with margin on both edges.
//
// normal_scale multiplies the "normal" branch's own [-1,1] output only --
// the extended branch above is a fixed, already-encodable-range test
// independent of the magnitude-scale sweep (see run_shape's own comment).
__device__ float pseudo_rand_hybrid_mixed(size_t idx, float normal_scale) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    if ((s % 10u) == 0u) {
        uint32_t s2 = s * 2246822519u + 3266489917u;
        s2 ^= s2 << 13;
        s2 ^= s2 >> 17;
        s2 ^= s2 << 5;
        float mag = 70000.0f + (static_cast<float>(s2 % 100001u) / 100000.0f) * 3930000.0f;
        return (s & 0x10000u) ? mag : -mag;
    }
    return normal_scale * ((static_cast<float>(s % 200001u) / 100000.0f) - 1.0f);  // [-scale, scale]
}

// scale multiplies pseudo_rand's own [-1,1] output, widening/narrowing the
// magnitude/exponent range the generated data covers while leaving its
// underlying distribution SHAPE unchanged -- see run_shape's own comment
// for why every generator here threads the SAME scale through consistently
// (both the float32 ground-truth and the packed-wh siblings, so they stay
// index-matched at every scale, not just scale=1).
__global__ void gen_float_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * pseudo_rand(i);
}
void gen_float(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out, scale);
}

// Float32 ground-truth siblings of gen_float above, for the two methods
// whose own packed-data generator isn't plain pseudo_rand -- same idx, so
// guaranteed to correspond to the SAME underlying values as that method's
// own gen_fp32_16_wh_hybrid/gen_fp32_16_wh_safe call below.
__global__ void gen_float_hybrid_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand_hybrid_mixed(i, scale);
}
void gen_float_hybrid(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_hybrid_kernel<<<blocks, threads>>>(n, out, scale);
}
__global__ void gen_float_safe_kernel(size_t n, float* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * pseudo_rand_safe_normal(i);
}
void gen_float_safe(size_t n, float* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_safe_kernel<<<blocks, threads>>>(n, out, scale);
}

// wr is written but never read by any kernel here (every kernel here only
// ever consumes wh) -- fp32_16_split/fp32_bf16_split still require a wr
// output parameter, so a scratch buffer is allocated, filled, and freed
// immediately, same convention mv_fp32_16_pure_half_vs_cublas_ref.cu's own
// generators use.
__global__ void gen_fp32_16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

// scale is intended to be a power of 2: pseudo_rand_safe_normal's own
// [0.5,1.0) output is exactly one binary exponent bracket, and multiplying
// by a power of 2 shifts it to another exponent bracket, [0.5*scale,
// 1.0*scale), preserving fp32_16_fast_half's own "always control=0"
// contract exactly at every scale -- a non-power-of-2 scale could straddle
// two brackets, which this generator does not attempt to guard against.
__global__ void gen_fp32_16_wh_safe_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(scale * pseudo_rand_safe_normal(i), wh[i], wr_scratch[i]);
}
void gen_fp32_16_wh_safe(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_safe_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

__global__ void gen_fp32_16_wh_hybrid_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_hybrid_mixed(i, scale), wh[i], wr[i]);
}
void gen_fp32_16_wh_hybrid(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_wh_hybrid_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

__global__ void gen_fp32_bf16_wh_kernel(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(scale * pseudo_rand(i), wh[i], wr_scratch[i]);
}
void gen_fp32_bf16_wh(size_t n, uint16_t* wh, uint16_t* wr_scratch, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_wh_kernel<<<blocks, threads>>>(n, wh, wr_scratch, scale);
}

// __half/__nv_bfloat16 generators for the cuBLAS FP16/BF16 baselines
// added below -- SAME pseudo_rand(idx) call, SAME idx, SAME scale as
// gen_float above, so these are index-matched to the plain-tier's own
// A_plain/x float32 ground truth (just cast down to __half/bfloat16 at
// generation time instead of decoded from a packed wh), not a converted
// copy of any TOMOS series' own data. Mirrors mv_fp32_16_pure_half_vs_
// cublas_ref.cu's own gen_half/gen_bf16, plus a scale parameter (that
// file's own performance sweep has no normal_scale knob to thread
// through).
__global__ void gen_half_kernel(size_t n, __half* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(scale * pseudo_rand(i));
}
void gen_half(size_t n, __half* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out, scale);
}
__global__ void gen_bf16_kernel(size_t n, __nv_bfloat16* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(scale * pseudo_rand(i));
}
void gen_bf16(size_t n, __nv_bfloat16* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_bf16_kernel<<<blocks, threads>>>(n, out, scale);
}

// Device-side __half/__nv_bfloat16 -> float conversion for cuBLAS FP16/
// BF16's own output vector y (length M) -- done on-device rather than
// relying on a host-callable half/bfloat16-to-float conversion, matching
// this file's own "never touch the host except for the final L2 error
// reduction" convention.
__global__ void half_to_float_kernel(size_t n, const __half* in, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}
void half_to_float(size_t n, const __half* in, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    half_to_float_kernel<<<blocks, threads>>>(n, in, out);
}
__global__ void bf16_to_float_kernel(size_t n, const __nv_bfloat16* in, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(in[i]);
}
void bf16_to_float(size_t n, const __nv_bfloat16* in, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    bf16_to_float_kernel<<<blocks, threads>>>(n, in, out);
}

// Same convention mv_fp32_16_pure_half_vs_cublas_ref.cu's own alloc_or_
// skip uses: at the largest shapes (131072x131072 needs ~68GB for a
// single float32 array), a cudaMalloc failure is expected, not a bug --
// skip that one allocation/method gracefully instead of crashing.
bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||cpu_ref - gpu||_2 / ||cpu_ref||_2, computed on the host over the
// OUTPUT vector only (length M) -- never the M*K matrix. See this file's
// own header for why this is safe at every shape the performance sweep
// itself reaches, unlike a correctness check needing the full matrix on
// host.
double l2_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// normal_scale widens/narrows every method's own "normal"-range data by
// this multiplier (see the generator kernels' own comments above) -- x
// itself is left UNSCALED by normal_scale specifically (always plain
// [-1,1] baseline): L2 RELATIVE error is invariant to uniformly
// rescaling x (y_ref and y_gpu both scale by the same factor, canceling
// in the ratio, since none of these kernels compress x, only A), so
// scaling it BY normal_scale would add a knob with no effect while
// adding complexity -- isolating that sweep to A's own magnitude/
// exponent range, the thing actually being 16-bit-compressed, is the
// whole point.
//
// normalized is a SEPARATE, orthogonal knob from normal_scale, for a
// different purpose: three of this tier's five kernels accumulate their
// row-sum in __half2 registers internally (confirmed by reading each
// kernel's own body), summing RAW, un-normalized A*x products for the
// entire K-loop BEFORE alpha is ever applied (alpha only appears once,
// in the epilogue, after the accumulator has already reduced) -- so
// alpha-based scaling cannot protect that accumulator from overflowing
// mid-reduction. Scaling x itself at DATA-GENERATION time does, since
// the accumulator only ever sees the already-smaller values: x_scale =
// 1/sqrt(K) exactly cancels the sqrt(K) growth a raw (x_scale=1) row-sum
// would otherwise have, keeping the accumulated magnitude ~normal_scale
// regardless of K. This is the one case where uniformly rescaling x
// does NOT leave L2-relative-error invariant (the invariance argument
// above assumes no saturation; __half2 accumulator saturation to
// Infinity is exactly the non-linearity that breaks it) -- seeing a
// DIFFERENT (finite, informative) error curve under --normalized here is
// the whole point. See scripts/benchmarks_gemv_fp32_16_l2_error_vs_
// cublas_ref.sh's own --normalized SCALES comment for the numbers this
// unlocks.
void run_shape(size_t m, size_t k, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);
    std::printf("  M=%zu K=%zu\n", m, k);

    int lda = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    dim<2> size{m, k};

    float x_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;
    float* x = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&x), k * sizeof(float), "x", shape_desc)) return;
    gen_float(k, x, x_scale);

    // ---- TOMOS FP16 and (reusing the same plain float32 ground truth
    // and cublasSgemv reference) TOMOS FP32/BF16, TOMOS BF16 -- bf16's
    // truncation decode has no unsafe-on-arbitrary-data caveat, so all
    // three ultimately trace back to the SAME plain pseudo_rand data. ----
    {
        float *A_plain = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_plain), m * k * sizeof(float), "A_plain", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_plain", shape_desc)) {
            gen_float(m * k, A_plain, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_plain, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_fp16 = nullptr, *wr_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_fp16), m * k * sizeof(uint16_t), "wh_fp16", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_scratch), m * k * sizeof(uint16_t), "wr_scratch",
                              shape_desc)) {
                gen_fp32_16_wh(m * k, wh_fp16, wr_scratch, normal_scale);
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fp16", shape_desc)) {
                    cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType::NO_TRANSPOSE, alpha,
                                                                                  size, wh_fp16, lda, x, 1, beta,
                                                                                  y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));

                    // narrow_vec (LSQR's large-M implicit kernel) unflushed
                    // vs flushed every 64 / 256 columns -- same wh, x, and
                    // fp32 reference. narrow_vec writes y directly (beta=0),
                    // so no re-zeroing is needed between legs.
                    cuda::mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_nomask_half2acc(
                        TransposeType::NO_TRANSPOSE, alpha, m, k, wh_fp16, lda, x, beta, y_gpu);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_narrowvec     = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cuda::mv_fp32_16_pure_half_narrow_vec_flush64(TransposeType::NO_TRANSPOSE, alpha, m, k, wh_fp16,
                                                                  lda, x, beta, y_gpu);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_narrowvec_flush64  = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cuda::mv_fp32_16_pure_half_narrow_vec_flush256(TransposeType::NO_TRANSPOSE, alpha, m, k, wh_fp16,
                                                                   lda, x, beta, y_gpu);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_narrowvec_flush256 = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cuda::mv_fp32_16_pure_half_narrow_vec_flush256_swar(TransposeType::NO_TRANSPOSE, alpha, m, k,
                                                                        wh_fp16, lda, x, beta, y_gpu);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_narrowvec_flush256_swar = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(wr_scratch);
                cudaFree(wh_fp16);
            }

            uint16_t *wh_bf16 = nullptr, *wr_bf16_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_scratch), m * k * sizeof(uint16_t), "wr_bf16_scratch",
                              shape_desc)) {
                gen_fp32_bf16_wh(m * k, wh_bf16, wr_bf16_scratch, normal_scale);
                float* y_gpu_v2 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_v2), m * sizeof(float), "y_gpu_bf16_v2",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(TransposeType::NO_TRANSPOSE, alpha,
                                                                                size, wh_bf16, lda, x, 1, beta,
                                                                                y_gpu_v2, 1);
                    std::vector<float> y_gpu_v2_h(m);
                    cudaMemcpy(y_gpu_v2_h.data(), y_gpu_v2, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_v2_h));
                    cudaFree(y_gpu_v2);
                }
                float* y_gpu_bf16 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_bf16), m * sizeof(float), "y_gpu_bf16",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(TransposeType::NO_TRANSPOSE, alpha,
                                                                             size, wh_bf16, lda, x, 1, beta,
                                                                             y_gpu_bf16, 1);
                    std::vector<float> y_gpu_bf16_h(m);
                    cudaMemcpy(y_gpu_bf16_h.data(), y_gpu_bf16, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_bf16               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_bf16_h));
                    cudaFree(y_gpu_bf16);
                }
                cudaFree(wr_bf16_scratch);
                cudaFree(wh_bf16);
            }

            // ---- cuBLAS FP16 (Hgemm, n=1, NO_TRANSPOSE): own error vs
            // the SAME plain float32 reference as TOMOS FP16/BF16 above --
            // independent __half-typed A_h/x_h, index-matched to A_plain/x
            // via the SAME pseudo_rand(idx)/scale (see gen_half's own
            // comment), NOT a converted copy of A_plain itself. alpha=1.0/
            // beta=0.0, same as every other method here (no epilogue
            // scaling to fold into the error analysis). ----
            {
                __half *A_h = nullptr, *x_h = nullptr, *y_h = nullptr;
                float* y_h_f = nullptr;
                bool ok_h =
                    alloc_or_skip(reinterpret_cast<void**>(&A_h), m * k * sizeof(__half), "cublas_fp16 (A_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_h), k * sizeof(__half), "cublas_fp16 (x_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_h), m * sizeof(__half), "cublas_fp16 (y_h)",
                                 shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_h_f), m * sizeof(float), "cublas_fp16 (y_h_f)",
                                 shape_desc);
                if (ok_h) {
                    gen_half(m * k, A_h, normal_scale);
                    gen_half(k, x_h, x_scale);
                    cudaMemset(y_h, 0, m * sizeof(__half));
                    __half h_alpha = __float2half(static_cast<float>(alpha));
                    __half h_beta = __float2half(static_cast<float>(beta));
                    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), 1, static_cast<int>(k),
                               &h_alpha, A_h, lda, x_h, static_cast<int>(k), &h_beta, y_h, static_cast<int>(m));
                    half_to_float(m, y_h, y_h_f);
                    std::vector<float> y_h_host(m);
                    cudaMemcpy(y_h_host.data(), y_h_f, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cublas_fp16              = %.6e\n", l2_rel_err(y_ref_h, y_h_host));
                }
                cudaFree(y_h_f);
                cudaFree(y_h);
                cudaFree(x_h);
                cudaFree(A_h);
            }

            // ---- TOMOS FP16 (native fp16 inputs): the SAME kernel the GEMV
            // performance plots label "TOMOS FP16" (native_storage -- A is a
            // plain __half array, x/y float, nothing packed or decoded).
            // Own error vs the SAME plain float32 reference as every other
            // series here: A_n is generated by gen_half with the SAME
            // pseudo_rand(idx)/normal_scale as A_plain (index-matched, see
            // gen_half's own comment -- the same construction the cuBLAS
            // FP16 leg above uses), and x is the shared plain float x, so
            // the only difference from the packed "tomos_fp16" leg above
            // is A's STORAGE (native __half vs FP32_16 wh-only decode).
            // The packed leg keeps its "tomos_fp16" CSV column (relabeled
            // "TOMOS FP16 (implicit)" in the plot); this one is
            // "tomos_fp16_native" -- named so it can't be mistaken for it
            // by scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh's
            // substring-based marker matching (longer marker listed first).
            {
                __half* A_n = nullptr;
                float* y_n = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&A_n), m * k * sizeof(__half),
                                  "tomos_fp16_native (A_n)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_n), m * sizeof(float),
                                  "tomos_fp16_native (y_n)", shape_desc)) {
                    gen_half(m * k, A_n, normal_scale);
                    cudaMemset(y_n, 0, m * sizeof(float));
                    cuda::experimental::mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, A_n, lda, x, 1, beta, y_n, 1);
                    std::vector<float> y_n_h(m);
                    cudaMemcpy(y_n_h.data(), y_n, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_native        = %.6e\n", l2_rel_err(y_ref_h, y_n_h));

                    // Same A_n/x, but through mv_fp16_native_mixed -- the
                    // kernel LSQR's explicit-duplicates path really runs
                    // (see its declaration above). y re-zeroed first: its
                    // atomic path accumulates into y.
                    cudaMemset(y_n, 0, m * sizeof(float));
                    cuda::mv_fp16_native_mixed(TransposeType::NO_TRANSPOSE, alpha, size, A_n, lda, x, 1, beta, y_n, 1);
                    cudaMemcpy(y_n_h.data(), y_n, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_native_mixed  = %.6e\n", l2_rel_err(y_ref_h, y_n_h));
                }
                cudaFree(y_n);
                cudaFree(A_n);
            }

            // ---- cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1, NO_TRANSPOSE):
            // own error vs the SAME plain float32 reference, same "no
            // native Bgemv" reasoning as cublasHgemm above. Independent
            // __nv_bfloat16-typed A_bf/x_bf, alpha/beta as float
            // (CUBLAS_COMPUTE_32F's own precision, not bf16). ----
            {
                __nv_bfloat16 *A_bf = nullptr, *x_bf = nullptr, *y_bf = nullptr;
                float* y_bf_f = nullptr;
                bool ok_bf =
                    alloc_or_skip(reinterpret_cast<void**>(&A_bf), m * k * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (A_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&x_bf), k * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (x_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf), m * sizeof(__nv_bfloat16),
                                 "cublas_bf16 (y_bf)", shape_desc) &&
                    alloc_or_skip(reinterpret_cast<void**>(&y_bf_f), m * sizeof(float), "cublas_bf16 (y_bf_f)",
                                 shape_desc);
                if (ok_bf) {
                    gen_bf16(m * k, A_bf, normal_scale);
                    gen_bf16(k, x_bf, x_scale);
                    cudaMemset(y_bf, 0, m * sizeof(__nv_bfloat16));
                    float alpha_f = static_cast<float>(alpha), beta_f = static_cast<float>(beta);
                    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), 1, static_cast<int>(k),
                                &alpha_f, A_bf, CUDA_R_16BF, lda, x_bf, CUDA_R_16BF, static_cast<int>(k), &beta_f,
                                y_bf, CUDA_R_16BF, static_cast<int>(m), CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    bf16_to_float(m, y_bf, y_bf_f);
                    std::vector<float> y_bf_host(m);
                    cudaMemcpy(y_bf_host.data(), y_bf_f, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    cublas_bf16              = %.6e\n", l2_rel_err(y_ref_h, y_bf_host));
                }
                cudaFree(y_bf_f);
                cudaFree(y_bf);
                cudaFree(x_bf);
                cudaFree(A_bf);
            }

            cudaFree(y_ref);
            cudaFree(A_plain);
        }
    }

    // ---- TOMOS FP32/FP16: own mixed normal/extended-range ground truth
    // and reference (plain data would never exercise the extended-
    // exponent reconstruction path this kernel exists to test). ----
    {
        float *A_hybrid = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_hybrid), m * k * sizeof(float), "A_hybrid", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_hybrid", shape_desc)) {
            gen_float_hybrid(m * k, A_hybrid, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_hybrid, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_hybrid = nullptr, *wr_hybrid = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_hybrid), m * k * sizeof(uint16_t), "wh_hybrid",
                              shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_hybrid), m * k * sizeof(uint16_t), "wr_hybrid",
                              shape_desc)) {
                gen_fp32_16_wh_hybrid(m * k, wh_hybrid, wr_hybrid, normal_scale);
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_hybrid", shape_desc)) {
                    cuda::mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_hybrid, lda, x, 1, beta, y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(wr_hybrid);
                cudaFree(wh_hybrid);
            }
            cudaFree(y_ref);
            cudaFree(A_hybrid);
        }
    }

    // ---- TOMOS FP32/FP16 (fast): own magnitude-restricted [0.5,1.0]
    // ground truth and reference -- mv_fp32_16_pure_half_wide_ilp_shared_
    // fasthalf_best is UNSAFE on arbitrary data (fp32_16_fast_half's own
    // contract), so plain data would silently test the wrong thing. ----
    {
        float *A_safe = nullptr, *y_ref = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_safe), m * k * sizeof(float), "A_safe", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&y_ref), m * sizeof(float), "y_ref_safe", shape_desc)) {
            gen_float_safe(m * k, A_safe, normal_scale);
            float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
            cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_safe, lda, x, 1,
                       &f_beta, y_ref, 1);
            std::vector<float> y_ref_h(m);
            cudaMemcpy(y_ref_h.data(), y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);

            uint16_t *wh_safe = nullptr, *wr_safe_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_safe), m * k * sizeof(uint16_t), "wh_safe", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_safe_scratch), m * k * sizeof(uint16_t), "wr_safe_scratch",
                              shape_desc)) {
                gen_fp32_16_wh_safe(m * k, wh_safe, wr_safe_scratch, normal_scale);
                float* y_gpu = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu), m * sizeof(float), "y_gpu_fasthalf",
                                  shape_desc)) {
                    cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_safe, lda, x, 1, beta, y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16_fast     = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }
                cudaFree(wr_safe_scratch);
                cudaFree(wh_safe);
            }
            cudaFree(y_ref);
            cudaFree(A_safe);
        }
    }

    cudaFree(x);
}

}  // namespace

int main(int argc, char** argv) {
    // Extract --normalized (a bare flag, may appear anywhere in argv)
    // before this file's own original positional argument parsing below
    // runs -- see run_shape's own comment for what it does and why it's
    // orthogonal to normal_scale.
    bool normalized = false;
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--normalized") {
            normalized = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    cublasHandle_t handle;
    cublasCreate(&handle);

    // normal_scale (first arg, default 1.0) multiplies every method's own
    // "normal"-range data -- see run_shape's own comment for why this is
    // the whole point of a magnitude/exponent sweep, and why x itself is
    // deliberately excluded from it. Intended to be a power of 2 (see
    // gen_fp32_16_wh_safe_kernel's own comment for why). Values up to
    // 32768 keep every INDIVIDUAL A[i,j] safely under fp16's own largest
    // finite magnitude (65504), never crossing into the hybrid kernel's
    // extended/degraded encoding (a DIFFERENT, already-covered axis,
    // fixed regardless of this scale -- see pseudo_rand_hybrid_mixed's
    // own comment) -- but that is NOT the same as safe for TOMOS FP16/
    // FP32/FP16/FP32/FP16(fast), which accumulate their row-sum in fp16
    // precision INTERNALLY: for those three, it's the ACCUMULATED SUM
    // across up to K terms that must stay under 65504, a far tighter
    // bound that shrinks as K grows (roughly scale*sqrt(K)/3 in
    // magnitude) -- see scripts/benchmarks_gemv_fp32_16_l2_error_vs_
    // cublas_ref.sh's own SCALES comment for the real numbers at this
    // sweep's largest K (131072): scale=32768 overflows those three
    // series into NaN/Inf at every shape, confirmed empirically, while
    // scale=64 keeps real margin.
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== NO_TRANSPOSE GEMV: 16-bit-storage L2 relative error vs FP32 (cublasSgemv) reference (normal_scale=%g, "
        "normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
                                              16384, 24576, 32768, 49152, 65536, 131072};

    if (argc > 2) {
        size_t m = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
        run_shape(m, m, handle, normal_scale, normalized);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t m : square_sizes) run_shape(m, m, handle, normal_scale, normalized);

    cublasDestroy(handle);
    return 0;
}

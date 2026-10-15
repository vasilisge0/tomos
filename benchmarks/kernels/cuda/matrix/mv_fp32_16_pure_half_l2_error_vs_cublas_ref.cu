// Standalone L2-relative-error analysis for the NO_TRANSPOSE (y =
// alpha*A*x + beta*y) genuine-16-bit-compute tier of GEMV -- the
// accuracy-vs-M companion to mv_fp32_16_pure_half_vs_cublas_ref.cu (which
// plots GFlop/s for the same shapes) and the NO_TRANSPOSE sibling of
// mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu (this file's
// own TRANSPOSE counterpart). See scripts/plot_gemv_fp32_16_pure_half_vs_
// cublas_ref.m for the throughput-side plot this file's error numbers are
// meant to sit alongside.
//
// Six "TOMOS" series, matching mv_fp32_16_pure_half_vs_cublas_ref.cu's
// own current header/`series` set exactly (that file's own promoted
// dispatchers, not its full exploratory kernel history -- see its header
// for the derivation of each):
//
//   TOMOS FP16              = mv_fp32_16_pure_half_wide_ilp_shared_half2_
//                              atomic_best (no workspace parameter --
//                              unlike the TRANSPOSE pure_half family's own
//                              atomiconly_scaled_mblock_vec4_split, the
//                              NO_TRANSPOSE atomic-combine family here
//                              atomicAdd-combines directly into y, no
//                              caller-owned scratch buffer to allocate)
//   TOMOS FP16 (fast)       = mv_fp32_16_pure_half_fast_wide_ilp_shared_
//                              half2_atomic_best -- fp32_16_half_only_fast
//                              decode sibling of TOMOS FP16 above, safe on
//                              arbitrary data (shares TOMOS FP16's own
//                              wh/reference, no separate generator
//                              needed). NOT the same kernel TOMOS FP32/
//                              FP16 (implicit) below uses, despite the
//                              similar "fast" naming history -- see that
//                              series' own comment and mv_fp32_16_pure_
//                              half_transpose_l2_error_vs_cublas_ref.cu's
//                              own identical caveat for its TRANSPOSE
//                              sibling.
//   TOMOS FP32/FP16         = mv_fp32_16_hybrid_wide_ilp_shared_half2_
//                              atomic_branchless_best
//   TOMOS FP32/FP16 (implicit) = mv_fp32_16_pure_half_wide_ilp_shared_
//                              half2_atomic_fasthalf_best -- an OLDER,
//                              UNSAFE kernel (unconditional fp32_16_fast_
//                              half bit-reinterpret, requires control=0
//                              always, hence its own magnitude-restricted
//                              generator below). Named "(implicit)" not
//                              "(fast)" to match this session's GEVM
//                              performance-plot convention (mv_fp32_16_
//                              pure_half_vs_cublas_ref.cu's own header
//                              still calls this leg "TOMOS FP32/FP16
//                              (fast)" since that file predates the
//                              rename -- same kernel either way) once the
//                              genuinely-tuned, safe fast-decode kernel
//                              above took the "(fast)" name.
//   TOMOS FP32/BF16         = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
//                              atomic_best_v2
//   TOMOS BF16              = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
//                              atomic_best (the non-"v2" dispatcher --
//                              genuinely distinct from _best_v2, both
//                              separately declared and separately timed
//                              in mv_fp32_16_pure_half_vs_cublas_ref.cu,
//                              and both promoted as separate series in
//                              plot_gemv_fp32_16_pure_half_vs_cublas_ref.m
//                              (bf16_atomic_best_v2_gflops/bf16_atomic_
//                              best_gflops) -- so both are measured here
//                              too, against the SAME wh_bf16/reference
//                              (both decode wh via the same fp32_bf16_
//                              bf16_only truncation, no unsafe-on-
//                              arbitrary-data caveat for either), matching
//                              this file's non-pure_half sibling
//                              mv_fp32_16_l2_error_vs_cublas_ref.cu, which
//                              plots both for the same reason)
//
// Generators/helpers (pseudo_rand, gen_float/gen_float_safe/gen_float_
// hybrid, gen_fp32_16_wh/gen_fp32_16_wh_safe/gen_fp32_16_wh_hybrid,
// gen_fp32_bf16_wh, gen_half/gen_bf16, half_to_float/bf16_to_float,
// l2_rel_err, alloc_or_skip) are reused verbatim from mv_fp32_16_l2_error_
// vs_cublas_ref.cu (this file's own NO_TRANSPOSE, non-pure_half sibling --
// same canonical source mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_
// ref.cu's own header already credits for the TRANSPOSE side): they are
// already NO_TRANSPOSE-shaped (x length K, y length M) since that is this
// file's own native direction too, so no dimensional adaptation was
// needed, only the run_shape structure (which series, in what order,
// against which reference) below was ported from the TRANSPOSE pure_half
// file and re-pointed at the NO_TRANSPOSE kernels/conventions above.
//
// All of the fp32_16/bf16-packed series confirmed (by reading each
// kernel's own __global__ body in fp32_16_cuda_kernels.cu) to accumulate
// their row-sum in __half2/__nv_bfloat162 internally -- same tight,
// K-dependent accumulator-overflow bound as this file's non-pure_half
// sibling's own fp16-accumulating series (NO_TRANSPOSE reduces over K, not
// M -- see run_shape's own comment below), not the much wider bound the
// "recon" (float-accumulating) tier gets.
//
// TOMOS FP16 and TOMOS FP16 (fast) share the SAME plain float32 ground
// truth / cublasSgemv(CUBLAS_OP_N) reference (both decode unconditionally
// safe on arbitrary data -- fp32_16_half_only and fp32_16_half_only_fast
// respectively); TOMOS FP32/BF16 and TOMOS BF16 also share that same
// reference and each other's wh (bf16's truncation decode has no
// unsafe-on-arbitrary-data caveat either, and the two kernels are
// otherwise identical on wh -- see their own comment above).
//
// x (float, length K) is never compressed by any of these methods, so a
// SINGLE shared x is generated once per shape and reused across all six
// series -- left permanently UNSCALED by normal_scale (L2 relative error
// is invariant to uniformly rescaling x, since none of these kernels
// compress x, only A -- see run_shape's own comment for the one exception,
// --normalized, and why it does NOT break that invariance argument by
// accident).
//
// L2 relative error: ||y_ref - y_method||_2 / ||y_ref||_2, computed on
// the HOST but only over the OUTPUT vector y (length M) -- never the M*K
// matrix.
//
// Usage: mv_fp32_16_pure_half_l2_error_vs_cublas_ref [normal_scale [M]]
//   No args: runs the same SQUARE_MS sweep as mv_fp32_16_pure_half_vs_
//            cublas_ref.cu's own square sweep (1024 through 131072), at
//            normal_scale=1.0.
//   normal_scale: runs the full SQUARE_MS sweep with every method's own
//            "normal"-range data widened to [-scale,scale] instead. See
//            mv_fp32_16_l2_error_vs_cublas_ref.cu's own header/scripts/
//            benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh for the
//            general convention this follows (no dedicated driver script
//            exists yet for this pure_half-specific file).
//   normal_scale M: runs just that one square shape (M=K=M) at that scale.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase.

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
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType trans, double alpha, dim<2> size,
                                                             const uint16_t* wh, size_t ld_a, const float* x,
                                                             size_t ld_x, double beta, float* y, size_t ld_y);
// Genuine safe-decode fast-decode sibling of the kernel above
// (fp32_16_half_only_fast, safe on arbitrary data -- unlike the
// fasthalf-based kernel TOMOS FP32/FP16 (implicit) below uses). Shares
// TOMOS FP16's own wh/reference, so no separate generator is needed.
void mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(TransposeType trans, double alpha, dim<2> size,
                                                                  const uint16_t* wh, size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(TransposeType trans, double alpha, dim<2> size,
                                                            const uint16_t* wh, size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(TransposeType trans, double alpha, dim<2> size,
                                                         const uint16_t* wh, size_t ld_a, const float* x,
                                                         size_t ld_x, double beta, float* y, size_t ld_y);
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
// shared_half2_atomic_fasthalf_best's own contract (fp32_16_fast_half
// decodes unconditionally as if control were always 0, so its own data
// must PROVABLY never be extended/degraded-encoded) -- copied verbatim
// from mv_fp32_16_l2_error_vs_cublas_ref.cu's own pseudo_rand_safe_normal.
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
// path) -- copied verbatim from mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// (already-corrected) pseudo_rand_hybrid_mixed -- see that file's own
// comment for the real magnitude-range bug this fixed and why [70000,
// 4000000] is the correct encodable-extended-range window (its OWN
// pre-existing derivation, not re-derived here).
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
// immediately, same convention mv_fp32_16_l2_error_vs_cublas_ref.cu's own
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

// scale is intended to be a power of 2 -- same fp32_16_fast_half "always
// control=0" contract as mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// gen_fp32_16_wh_safe_kernel.
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
// A_plain/x float32 ground truth, not a converted copy of any TOMOS
// series' own data. Mirrors mv_fp32_16_l2_error_vs_cublas_ref.cu's own
// gen_half/gen_bf16.
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
// BF16's own output vector y (length M here, NO_TRANSPOSE convention) --
// done on-device, matching this file's own "never touch the host except
// for the final L2 error reduction" convention.
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

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-24s = SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||cpu_ref - gpu||_2 / ||cpu_ref||_2, computed on the host over the
// OUTPUT vector only (length M here, NO_TRANSPOSE convention).
double l2_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

// x has length K here (NO_TRANSPOSE: y[m] = sum_k A[m,k]*x[k]), y has
// length M -- OPPOSITE of mv_fp32_16_pure_half_transpose_l2_error_vs_
// cublas_ref.cu's own convention (x length M, y length K there). A stored
// M x K column-major throughout (lda=m), matching mv_fp32_16_pure_half_
// vs_cublas_ref.cu's own convention. Reduction is over K (not M) here, so
// normalized mode scales x by 1/sqrt(k) -- all five of this tier's
// kernels accumulate in __half2/__nv_bfloat162 internally, summing RAW
// products before alpha is ever applied, so only a data-generation-time
// scale (not alpha) can protect that accumulator; see mv_fp32_16_l2_
// error_vs_cublas_ref.cu's own run_shape comment for the full reasoning
// (identical here, just re-pointed at K instead of M since that file is
// ALSO NO_TRANSPOSE).
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

    // ---- TOMOS FP16, TOMOS FP16 (fast) and (reusing the same plain
    // float32 ground truth and cublasSgemv(CUBLAS_OP_N) reference) TOMOS
    // FP32/BF16 -- all three decode unconditionally safe on arbitrary
    // data. ----
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
                    cuda::mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_fp16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
                    cudaFree(y_gpu);
                }

                // ---- TOMOS FP16 (fast): genuine fast-decode sibling of
                // TOMOS FP16 above (fp32_16_half_only_fast instead of
                // fp32_16_half_only) -- safe on arbitrary data, so reuses
                // the SAME wh_fp16/y_ref_h as TOMOS FP16, no separate
                // magnitude-restricted generator needed (unlike the OLD
                // "tomos_fp32_fp16_implicit" series below, which predates
                // this decode and still needs its own [0.5,1.0]-restricted
                // data). ----
                float* y_gpu_fast = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_fast), m * sizeof(float), "y_gpu_fp16_fast",
                                  shape_desc)) {
                    cuda::mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_fp16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu_fast, 1);
                    std::vector<float> y_gpu_fast_h(m);
                    cudaMemcpy(y_gpu_fast_h.data(), y_gpu_fast, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp16_fast          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_fast_h));
                    cudaFree(y_gpu_fast);
                }

                cudaFree(wr_scratch);
                cudaFree(wh_fp16);
            }

            uint16_t *wh_bf16 = nullptr, *wr_bf16_scratch = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t), "wh_bf16", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_scratch), m * k * sizeof(uint16_t), "wr_bf16_scratch",
                              shape_desc)) {
                gen_fp32_bf16_wh(m * k, wh_bf16, wr_bf16_scratch, normal_scale);
                float* y_gpu_bf16 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_bf16), m * sizeof(float), "y_gpu_bf16",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_bf16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu_bf16, 1);
                    std::vector<float> y_gpu_bf16_h(m);
                    cudaMemcpy(y_gpu_bf16_h.data(), y_gpu_bf16, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_bf16          = %.6e\n", l2_rel_err(y_ref_h, y_gpu_bf16_h));
                    cudaFree(y_gpu_bf16);
                }

                // ---- TOMOS BF16: non-"v2" dispatcher, genuinely distinct
                // from TOMOS FP32/BF16 above (separately declared/timed in
                // mv_fp32_16_pure_half_vs_cublas_ref.cu, separately
                // promoted in its own plot) -- reuses the SAME wh_bf16/
                // reference, same reasoning as TOMOS FP16 (fast) reusing
                // TOMOS FP16's own data above. ----
                float* y_gpu_bf16_v1 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&y_gpu_bf16_v1), m * sizeof(float), "y_gpu_bf16_v1",
                                  shape_desc)) {
                    cuda::mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_bf16, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu_bf16_v1, 1);
                    std::vector<float> y_gpu_bf16_v1_h(m);
                    cudaMemcpy(y_gpu_bf16_v1_h.data(), y_gpu_bf16_v1, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_bf16               = %.6e\n", l2_rel_err(y_ref_h, y_gpu_bf16_v1_h));
                    cudaFree(y_gpu_bf16_v1);
                }

                cudaFree(wr_bf16_scratch);
                cudaFree(wh_bf16);
            }

            // ---- cuBLAS FP16 (Hgemm, n=1, NO_TRANSPOSE): own error vs
            // the SAME plain float32 reference as TOMOS FP16/BF16 above --
            // independent __half-typed A_h/x_h, index-matched to A_plain/x
            // via the SAME pseudo_rand(idx)/scale. alpha=1.0/beta=0.0,
            // same as every other method here. Dimensions/CUBLAS_OP_N
            // convention match mv_fp32_16_pure_half_vs_cublas_ref.cu's own
            // cublasHgemm leg exactly (y[m] = alpha*A*x[k]). ----
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
    // and reference. ----
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
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_hybrid, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu, 1);
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

    // ---- TOMOS FP32/FP16 (implicit): own magnitude-restricted [0.5,1.0]
    // ground truth and reference -- UNSAFE on arbitrary data (unconditional
    // fp32_16_fast_half bit-reinterpret, requires control=0 always). Named
    // "(implicit)" not "(fast)" to match this session's own GEVM
    // performance-plot renaming (the genuinely-tuned, safe fast-decode
    // kernel now owns the "(fast)" name -- see TOMOS FP16 (fast) above). ----
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
                        TransposeType::NO_TRANSPOSE, alpha, size, wh_safe, static_cast<size_t>(lda), x, 1, beta,
                        y_gpu, 1);
                    std::vector<float> y_gpu_h(m);
                    cudaMemcpy(y_gpu_h.data(), y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
                    std::printf("    tomos_fp32_fp16_implicit = %.6e\n", l2_rel_err(y_ref_h, y_gpu_h));
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
    // runs -- see run_shape's own comment for what it does.
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
    // "normal"-range data. All five series here accumulate their row-sum
    // in fp16/bf16 precision internally (see this file's own header), so
    // the SAME tight, empirically-derived bound as this file's non-
    // pure_half sibling mv_fp32_16_l2_error_vs_cublas_ref.cu applies.
    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== NO_TRANSPOSE GEMV: 16-bit-compute L2 relative error vs FP32 (cublasSgemv CUBLAS_OP_N) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
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

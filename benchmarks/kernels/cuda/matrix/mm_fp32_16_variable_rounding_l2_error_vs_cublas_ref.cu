// Standalone experiment: how much does the ENCODE-side rounding decision
// itself matter for the fp32_16 packed format's tensor-core ("pure_half")
// decode path? mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu (this tier's own
// sibling error benchmark) always encodes A via the real fp32_16_split,
// which examines all 13 "sticky" bits (mant_lo's own bits 12..0) plus the
// round bit (bit 13) to decide whether to round mant_hi up (round to
// nearest, ties to even -- see fp32_16_pack.hpp's own header). This file
// instead sweeps how many of those 13 sticky bits the rounding decision
// actually looks at -- 13 (identical to fp32_16_split), 9, 5, 1, and 0
// ("truncation", identical to fp32_16_split_truncated) -- to see how
// coarsening that decision degrades the resulting __half value's own
// accuracy, for a FIXED downstream kernel and a FIXED decode
// (fp32_16_half_only, which only ever reads wh -- wr, and hence
// round_flag, plays no role in this specific decode's own accuracy; only
// what ends up baked into wh's own mant_hi/control/offset bits matters).
//
// IMPORTANT: this file does NOT modify fp32_16_split or add a variant to
// fp32_16_pack.hpp. Per explicit instruction, the variable-rounding
// encoder lives ENTIRELY in this file, as a new, separate, additive
// function (fp32_16_split_variable_rounding, below) -- it duplicates
// fp32_16_split's own full-carry structure exactly (same normal/
// extended/degraded branches, same mant_hi-overflow-across-a-boundary
// handling), swapping in a parametrized round-up decision
// (round_up_variable) in place of fp32_16_detail::fp32_16_round_up.
// sticky_bits=13 is verified to reproduce fp32_16_split's own encoding
// bit-for-bit (fp32_16_round_up's own sticky mask is exactly 0x1FFFu --
// all 13 of mant_lo's bits 12..0), so the "13" series here is a direct
// cross-check that this duplicated logic matches the real encoder,
// not just a similar-looking one.
//
// ALL EIGHT SERIES RUN THROUGH THE SAME KERNEL: gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp (TOMOS
// FP16, the native tensor-core kernel) -- unlike this tier's own
// promoted "TOMOS FP32/FP16" series (mm_fp32_16_pure_half_wide_bigreg_
// sq_inline_dirty_..., a SEPARATE kernel body that decodes wh inline),
// this file decodes each rounding config's own wh into a genuine
// __half* array FIRST (via a small elementwise decode kernel calling
// fp32_16_half_only, the same decode function inline_dirty itself
// uses), then calls the identical native kernel every series uses --
// so the only thing that ever differs between any of the 8 series is
// what values ended up in A's own __half array, never which kernel
// computed the GEMM. This isolates the encode-side rounding-precision
// question from any incidental difference between kernel
// implementations (inline_dirty vs native are independently-written,
// even though they share the same tile/pipeline/PTX).
//
// A SECOND, SEPARATE AXIS IS ALSO ISOLATED: mantissa WIDTH. Every one
// of the 6 fp32_16_half_only-decoded series -- ALL SIX (the 5 sticky_
// bits/truncation configs plus the new stochastic-rounding one below)
// -- is structurally capped at 9 effective mantissa bits, never 10,
// because wh's own bit layout (sign(1)|exponent(5)|mant_hi(9)|
// control(1)) reinterpreted directly as a __half puts the control bit
// (always 0 in the in-range/control=0 case) where a real half's
// mantissa LSB would be. That conflates "9 vs 10 mantissa bits" (a
// fixed property of the wh-alone format) with "which rounding METHOD
// produced mant_hi" (round-to-nearest, truncate, or stochastic -- what
// the encode-side series below actually vary). tomos_fp16_trunc10
// isolates the first axis: a genuine 10-bit-mantissa TRUNCATING (not
// rounding) float->half conversion (float_to_half_truncate10, below),
// bypassing the packed wh/wr format entirely.
//
// A THIRD ROUNDING METHOD is added alongside round-to-nearest and
// truncation: STOCHASTIC ROUNDING (fp32_16_split_stochastic_rounding,
// below) -- rounds mant_hi up with probability exactly mant_lo/16384
// (the true fractional position between mant_hi_pre and mant_hi_pre+1)
// instead of always rounding to whichever is nearer. This makes the
// rounding UNBIASED IN EXPECTATION per element, unlike round-to-nearest
// (only unbiased in aggregate, under symmetric data) or truncation
// (never unbiased -- always rounds toward the cut). Standard technique
// in low-precision ML training (Gupta et al., "Deep Learning with
// Limited Numerical Precision," ICML 2015). Summary table (mantissa
// width x rounding method -- stochastic only tested at 9 bits here,
// alongside the sticky_bits sweep it's a direct alternative to):
//   10-bit, rounded      = tomos_fp16_native   (__float2half)
//   10-bit, truncated     = tomos_fp16_trunc10  (float_to_half_truncate10)
//    9-bit, rounded       = tomos_fp32_fp16_round_13 (sticky_bits=13)
//    9-bit, truncated      = tomos_fp32_fp16_trunc    (sticky_bits=0)
//    9-bit, stochastic     = tomos_fp32_fp16_stochastic
//
// Pipeline per shape (M=K=N from the square sweep below):
//   1. Generate A, B as float (same pseudo_rand-derived data every other
//      error benchmark in this codebase uses), compute C_ref via
//      cublasSgemm -- the genuine FP32 reference every series here is
//      measured against.
//   2. Convert A, B to native __half (A_half, B_half) directly
//      (__float2half) and compute C_half_native = TOMOS FP16 on them --
//      reported as a reference point (no packed-format encode/decode
//      step at all for A, beyond the plain float->half rounding).
//   3. Convert A to __half via float_to_half_truncate10 instead (B_half
//      unchanged from step 2) and compute tomos_fp16_trunc10 -- isolates
//      truncation vs rounding at a FIXED 10-bit mantissa.
//   4. Encode A via fp32_16_split_stochastic_rounding into (wh, wr);
//      decode via fp32_16_half_only; run through the SAME TOMOS FP16
//      kernel and the SAME B_half. Report as tomos_fp32_fp16_stochastic.
//   5. For each of the 5 rounding configs: encode A via
//      fp32_16_split_variable_rounding(..., sticky_bits) into (wh, wr);
//      decode wh into a genuine __half* array via fp32_16_half_only
//      (elementwise, see decode_pure_half_kernel below); call the SAME
//      TOMOS FP16 kernel from step 2 on that decoded array and the SAME
//      B_half. Report ||C_ref - C_half2||_F / ||C_ref||_F.
//
// B is shared across every series in a given shape -- so all 8 series
// (native + trunc10 + stochastic + 5 rounding configs) are directly
// comparable, differing only in how A's own __half values were produced.
//
// Two separate scripts drive this binary, for two different sweep axes:
//   scripts/benchmarks_gemm_fp32_16_variable_rounding_l2_error_vs_
//     cublas_ref.sh -- N-sweep at normal_scale=1.0 (this is the ORIGINAL
//     use case this file was built for: the rounding DECISION's own
//     precision, not magnitude/overflow behavior, so it never needed a
//     normal_scale sweep of its own).
//   scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_
//     error_vs_cublas_ref.sh -- normal_scale-sweep at a SINGLE FIXED
//     large shape (M=K=N=49152), --normalized always applied. This is
//     what actually stresses the packed format's own normal/extended/
//     degraded encoding branches (which only start diverging from the
//     "normal" case at large |A[i,j]|), something the N-sweep above
//     never varies. --normalized only rescales B (see run_shape's own
//     comment) -- it does NOT change which encoding branch any of A's
//     own elements land in, so it's safe to combine with a wide
//     normal_scale range without perturbing the actual thing being
//     measured, while avoiding the K-dependent output-storage-ceiling
//     "inf" bug this file's own sibling (mm_fp16_bf16_pure_l2_error_vs_
//     cublas_ref.cu) already found once at large, fixed K.
//
// Frobenius relative error: ||C_ref - C_method||_F / ||C_ref||_F,
// computed on the HOST over the OUTPUT matrix only (size M*N), after
// converting each method's own __half-typed C to float -- identical
// metric/convention to mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu.
//
// Usage: mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref [normal_scale [N]] [--normalized]
//   No args: runs the same square sweep as scripts/benchmarks_gemm_
//            fp16_bf16_pure_l2_error_vs_cublas_ref.sh's own SIZES
//            (1024 through 65536, M=K=N), at normal_scale=1.0.
//   normal_scale: runs the full sweep with A's own "normal"-range data
//            widened to [-scale,scale] instead.
//   normal_scale N: runs just that one square shape (M=K=N=N).
//   --normalized (may appear anywhere in argv): scales B by 1/sqrt(K)
//            instead of leaving it at scale=1.0 -- see run_shape's own
//            comment for why and when to use this.
//
// Deliberately NOT wired into tomos's CMake build -- built standalone via
// a direct nvcc invocation that links against the already-built
// libtomos_cuda.so, same convention as every other *_vs_cublas_ref.cu
// file in this codebase. See scripts/benchmarks_gemm_fp32_16_variable_
// rounding_l2_error_vs_cublas_ref.sh.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fp32_16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __half* A, size_t ld_a,
    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
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

// ---- New, benchmark-local, FULL-10-bit-mantissa truncating quantizer.
// See this file's own header for why: every fp32_16_half_only-decoded
// series above -- ALL FIVE, "trunc" (sticky_bits=0) included -- is
// structurally capped at 9 effective mantissa bits, never 10, because
// wh's own bit layout (sign(1)|exponent(5)|mant_hi(9)|control(1))
// reinterpreted directly as a __half puts the control bit (always 0 in
// the in-range/control=0 case) where a real half's mantissa LSB would
// be. That conflates two separate things: "9 vs 10 mantissa bits" (a
// fixed property of the wh-alone format, independent of sticky_bits)
// and "round-to-nearest vs truncate" (what sticky_bits actually
// controls). This function isolates the first axis: it goes straight
// from float to a genuine 16-bit half bit pattern -- sign(1)|
// exponent(5)|mantissa(10) -- with NO packed wh/wr format involved at
// all, and TRUNCATES (drops the low 13 mantissa bits outright, no
// round-to-nearest, no round/sticky computation) rather than rounding,
// giving a true "10-bit mantissa, truncated" data point that no
// existing series above provides. Same exponent-range boundaries as
// fp32_16_pack.hpp's own kFp32ExpMin/kFp32ExpMax (113/142, i.e. fp16's
// own valid normal-exponent window after rebias) for consistency with
// every other series' own encoding-branch behavior; underflow flushes
// to signed zero and overflow saturates to signed Infinity, matching
// this codebase's own general degraded/extended-range conventions
// rather than producing subnormals. ----
__host__ __device__ inline __half float_to_half_truncate10(float value) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    uint16_t result_bits;
    if (exp32 < kFp32ExpMin) {
        result_bits = static_cast<uint16_t>(sign);  // underflow -> truncate toward zero
    } else if (exp32 > kFp32ExpMax) {
        result_bits = static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> signed Infinity
    } else {
        uint32_t exp16 = exp32 - kExpRebias;
        uint32_t mant10 = mant32 >> 13;  // TRUNCATE to the top 10 bits -- no rounding at all
        result_bits = static_cast<uint16_t>(sign | (exp16 << 10) | mant10);
    }
    return *reinterpret_cast<const __half*>(&result_bits);
}

__global__ void gen_half_truncate10_kernel(size_t n, __half* out, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = float_to_half_truncate10(scale * pseudo_rand(i));
}
void gen_half_truncate10(size_t n, __half* out, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_truncate10_kernel<<<blocks, threads>>>(n, out, scale);
}

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

// ---- New, benchmark-local variable-rounding encoder (see this file's
// own header for why this is a SEPARATE function, not a change to
// fp32_16_split/fp32_16_pack.hpp). ----

// Same round-to-nearest-ties-to-even decision as fp32_16_detail::
// fp32_16_round_up, except the sticky computation only ORs together the
// TOP `sticky_bits` of mant_lo's own 13 sticky bits (bits 12..0, closest
// to the round bit first) -- the remaining, less-significant bits are
// ignored entirely, not just zeroed into the OR. sticky_bits>=13 (or
// exactly 13) reproduces fp32_16_round_up's own 0x1FFFu sticky mask bit-
// for-bit. sticky_bits<=0 skips the round bit too, so round-up never
// fires -- plain truncation, matching fp32_16_split_truncated's own
// output bit-for-bit.
__host__ __device__ inline bool round_up_variable(uint32_t mant_lo, bool mant_hi_pre_is_odd, int sticky_bits) {
    if (sticky_bits <= 0) return false;
    int n = (sticky_bits > 13) ? 13 : sticky_bits;
    bool round_bit = ((mant_lo >> 13) & 0x1u) != 0u;
    uint32_t sticky_mask = ((1u << n) - 1u) << (13 - n);
    bool sticky = (mant_lo & sticky_mask) != 0u;
    return round_bit && (sticky || mant_hi_pre_is_odd);
}

// Byte-for-byte the same full-carry structure as fp32_16_split (normal/
// extended/degraded branches, mant_hi-overflow-across-a-boundary
// handling via exp32_eff), with round_up_variable in place of
// fp32_16_detail::fp32_16_round_up as the only substantive change.
__host__ __device__ inline void fp32_16_split_variable_rounding(float value, uint16_t& wh, uint16_t& wr,
                                                                  int sticky_bits) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    if (exp32 >= kFp32ExpMin - kOffsetMax && exp32 <= kFp32ExpMax + kOffsetMax) {
        uint32_t mant_hi_pre = mant32 >> 14;
        uint32_t mant_lo = mant32 & 0x3FFFu;
        bool round_up = round_up_variable(mant_lo, (mant_hi_pre & 0x1u) != 0u, sticky_bits);

        uint32_t mant_hi_rounded = mant_hi_pre + (round_up ? 1u : 0u);
        bool overflow = mant_hi_rounded > 0x1FFu;
        uint32_t exp32_eff = exp32 + (overflow ? 1u : 0u);
        uint32_t mant_hi_written = overflow ? 0u : mant_hi_rounded;
        uint32_t round_flag = round_up ? 1u : 0u;

        if (exp32_eff >= kFp32ExpMin && exp32_eff <= kFp32ExpMax) {
            uint32_t exp16 = exp32_eff - kExpRebias;
            wh = static_cast<uint16_t>(sign | (exp16 << 10) | (mant_hi_written << 1) | 0u);
            wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
            return;
        }
        uint32_t direction_eff = (exp32_eff > kFp32ExpMax) ? 1u : 0u;
        uint32_t offset_eff =
            (exp32_eff > kFp32ExpMax) ? (exp32_eff - kFp32ExpMax) : (kFp32ExpMin - exp32_eff);
        if (offset_eff >= 1u && offset_eff <= kOffsetMax) {
            wh = static_cast<uint16_t>(sign | (direction_eff << 14) | (offset_eff << 10) | (mant_hi_written << 1) |
                                       1u);
            wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
            return;
        }
        // Falls through to the degraded path below, exactly as
        // fp32_16_split's own comment describes for this astronomically
        // narrow edge case.
    }

    // Saturate: matches fp32_16_split's own new saturate path (see
    // fp32_16_pack.hpp) -- exact zero and every other out-of-reach value
    // (deeper overflow/underflow, fp32 subnormals, inf, nan) all land here.
    uint32_t direction = (exp32 > kFp32ExpMax) ? 1u : 0u;
    wh = static_cast<uint16_t>(sign | (direction << 14) | (kOffsetSaturate << 10) | 1u);
    wr = 0;
}

__global__ void gen_fp32_16_variable_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale, int sticky_bits) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_variable_rounding(scale * pseudo_rand(i), wh[i], wr[i], sticky_bits);
}
void gen_fp32_16_variable(size_t n, uint16_t* wh, uint16_t* wr, float scale, int sticky_bits) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_variable_kernel<<<blocks, threads>>>(n, wh, wr, scale, sticky_bits);
}

// ---- New, benchmark-local STOCHASTIC-ROUNDING encoder (a third
// rounding strategy, alongside fp32_16_split's own round-to-nearest and
// fp32_16_split_truncated's plain truncation) -- again a SEPARATE,
// additive function, not a change to fp32_16_split/fp32_16_pack.hpp.
// ----

// Independent PRNG stream from pseudo_rand's own (different multiplier/
// salt), so the stochastic-rounding DECISION for element idx is not
// correlated with the DATA VALUE pseudo_rand(idx) itself produced for
// that same idx -- still a pure, deterministic function of idx alone
// (same "reproducible given (index, scale)" convention every generator
// in this codebase follows), just from an independent-looking stream.
// Returns a value uniform on 0..16383 inclusive (14 bits, matching mant_lo's
// own 14-bit width).
__device__ inline uint32_t stochastic_rand14(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 0x9E3779B1u + 0xC2B2AE35u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s & 0x3FFFu;
}

// Byte-for-byte the same full-carry structure as fp32_16_split/
// fp32_16_split_variable_rounding (normal/extended/degraded branches,
// mant_hi-overflow-across-a-boundary handling via exp32_eff), but the
// round-up decision is STOCHASTIC instead of round-to-nearest: round up
// with probability exactly mant_lo/16384 (the true fractional position
// between mant_hi_pre and mant_hi_pre+1), rather than always rounding to
// whichever is closer. This makes the rounding UNBIASED IN EXPECTATION
// per element (E[decoded value] == the true pre-rounding value) --
// round-to-nearest is only unbiased in aggregate under certain symmetric
// data distributions, and truncation is never unbiased (it always
// rounds toward zero-relative-to-the-cut). Stochastic rounding is the
// standard technique for this exact tradeoff in low-precision ML
// training (Gupta et al., "Deep Learning with Limited Numerical
// Precision," ICML 2015).
__device__ inline void fp32_16_split_stochastic_rounding(float value, uint16_t& wh, uint16_t& wr, size_t idx) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    if (exp32 >= kFp32ExpMin - kOffsetMax && exp32 <= kFp32ExpMax + kOffsetMax) {
        uint32_t mant_hi_pre = mant32 >> 14;
        uint32_t mant_lo = mant32 & 0x3FFFu;
        bool round_up = stochastic_rand14(idx) < mant_lo;

        uint32_t mant_hi_rounded = mant_hi_pre + (round_up ? 1u : 0u);
        bool overflow = mant_hi_rounded > 0x1FFu;
        uint32_t exp32_eff = exp32 + (overflow ? 1u : 0u);
        uint32_t mant_hi_written = overflow ? 0u : mant_hi_rounded;
        uint32_t round_flag = round_up ? 1u : 0u;

        if (exp32_eff >= kFp32ExpMin && exp32_eff <= kFp32ExpMax) {
            uint32_t exp16 = exp32_eff - kExpRebias;
            wh = static_cast<uint16_t>(sign | (exp16 << 10) | (mant_hi_written << 1) | 0u);
            wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
            return;
        }
        uint32_t direction_eff = (exp32_eff > kFp32ExpMax) ? 1u : 0u;
        uint32_t offset_eff =
            (exp32_eff > kFp32ExpMax) ? (exp32_eff - kFp32ExpMax) : (kFp32ExpMin - exp32_eff);
        if (offset_eff >= 1u && offset_eff <= kOffsetMax) {
            wh = static_cast<uint16_t>(sign | (direction_eff << 14) | (offset_eff << 10) | (mant_hi_written << 1) |
                                       1u);
            wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
            return;
        }
    }

    // Saturate: matches fp32_16_split's own new saturate path (see
    // fp32_16_pack.hpp) -- exact zero and every other out-of-reach value
    // (deeper overflow/underflow, fp32 subnormals, inf, nan) all land here.
    uint32_t direction = (exp32 > kFp32ExpMax) ? 1u : 0u;
    wh = static_cast<uint16_t>(sign | (direction << 14) | (kOffsetSaturate << 10) | 1u);
    wr = 0;
}

__global__ void gen_fp32_16_stochastic_kernel(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split_stochastic_rounding(scale * pseudo_rand(i), wh[i], wr[i], i);
}
void gen_fp32_16_stochastic(size_t n, uint16_t* wh, uint16_t* wr, float scale) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_stochastic_kernel<<<blocks, threads>>>(n, wh, wr, scale);
}

// Elementwise decode of wh alone into a genuine __half array, via the
// same fp32_16_half_only that mm_fp32_16_pure_half_wide_bigreg_sq_
// inline_dirty_... itself decodes wh through -- used here to materialize
// a real __half* A array so every rounding config can be run through the
// SAME TOMOS FP16 kernel tomos_fp16_native uses, instead of through a
// separate, independently-written kernel (see this file's own header for
// why that matters for this experiment specifically).
__global__ void decode_pure_half_kernel(size_t n, const uint16_t* wh, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_16_half_only(wh[i]);
}
void decode_pure_half(size_t n, const uint16_t* wh, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    decode_pure_half_kernel<<<blocks, threads>>>(n, wh, out);
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// ||C_ref - C_gpu||_F / ||C_ref||_F, computed on the host over the
// OUTPUT matrix only (size M*N).
double frobenius_rel_err(const std::vector<float>& ref, const std::vector<float>& gpu) {
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(ref[i]) - static_cast<double>(gpu[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);
}

struct RoundingConfig {
    int sticky_bits;
    const char* marker;
};
// Markers chosen so none is a substring of another (2-digit zero-padded
// counts), matching the "no substring collision" discipline every
// sibling *_l2_error_vs_cublas_ref.sh's own embedded parser uses.
const RoundingConfig kConfigs[] = {
    {13, "tomos_fp32_fp16_round_13"},
    {9, "tomos_fp32_fp16_round_09"},
    {5, "tomos_fp32_fp16_round_05"},
    {1, "tomos_fp32_fp16_round_01"},
    {0, "tomos_fp32_fp16_trunc"},
};

// m/k/n follow BLAS convention: A is M x K, B is K x N, C is M x N (all
// column-major, so lda=m, ldb=k, ldc=m). normal_scale widens/narrows A's
// own "normal"-range data. B's own scale depends on `normalized`: false
// (default) leaves B at scale=1.0, matching this file's own original
// "only A's magnitude sweeps" convention; true scales B by 1/sqrt(k)
// instead (identical mechanism/motivation to mm_fp16_bf16_pure_l2_
// error_vs_cublas_ref.cu's own --normalized, see that file's run_shape
// for the derivation) -- this keeps C[i,j]'s own row-sum magnitude
// ~normal_scale regardless of K instead of growing like normal_scale*
// sqrt(K), which matters here specifically because every series in this
// file narrows its float32 tensor-core accumulator down into __half-
// typed C, and C's own magnitude is what actually overflows fp16's
// ~65504 ceiling at large K/normal_scale -- NOT A's or B's own per-
// element storage magnitude (same K-dependent output-storage-ceiling
// bug class mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu's own header
// documents finding). Either way, B is shared unmodified across every
// series below.
void run_shape(size_t m, size_t k, size_t n, cublasHandle_t handle, float normal_scale, bool normalized) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu N=%zu", m, k, n);
    std::printf("  %s\n", shape_desc);

    int lda = static_cast<int>(m), ldb = static_cast<int>(k), ldc = static_cast<int>(m);
    double alpha = 1.0, beta = 0.0;
    float b_scale = normalized ? 1.0f / std::sqrt(static_cast<float>(k)) : 1.0f;

    float *A_true = nullptr, *B_true = nullptr, *C_ref = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&A_true), m * k * sizeof(float), "A_true", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&B_true), k * n * sizeof(float), "B_true", shape_desc) ||
        !alloc_or_skip(reinterpret_cast<void**>(&C_ref), m * n * sizeof(float), "C_ref", shape_desc)) {
        cudaFree(A_true);
        cudaFree(B_true);
        cudaFree(C_ref);
        return;
    }
    gen_float(m * k, A_true, normal_scale);
    gen_float(k * n, B_true, b_scale);

    float f_alpha = static_cast<float>(alpha), f_beta = static_cast<float>(beta);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
               &f_alpha, A_true, lda, B_true, ldb, &f_beta, C_ref, ldc);
    std::vector<float> C_ref_h(m * n);
    cudaMemcpy(C_ref_h.data(), C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_true);
    cudaFree(B_true);
    cudaFree(C_ref);

    __half* B_half = nullptr;
    if (!alloc_or_skip(reinterpret_cast<void**>(&B_half), k * n * sizeof(__half), "B_half", shape_desc)) {
        return;
    }
    gen_half(k * n, B_half, b_scale);

    // Native FP16 reference series (TOMOS FP16): genuinely __half-typed
    // A, no packed format involved -- reported purely as a comparison
    // point for the 5 rounding-config series below.
    {
        __half *A_half = nullptr, *C_half = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_half), m * k * sizeof(__half), "A_half", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&C_half), m * n * sizeof(__half), "C_half_native", shape_desc)) {
            gen_half(m * k, A_half, normal_scale);
            cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half,
                static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half, static_cast<size_t>(ldc));
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_native", shape_desc)) {
                half_to_float(m * n, C_half, C_gpu);
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp16_native        = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
        }
        cudaFree(A_half);
        cudaFree(C_half);
    }

    // 10-bit-mantissa TRUNCATED reference series: same native kernel,
    // same B_half, A generated via float_to_half_truncate10 (this
    // file's own new, separate quantizer -- see its own comment above)
    // instead of __float2half. Isolates "truncate vs round-to-nearest"
    // at a FIXED 10-bit mantissa, the missing comparison point against
    // tomos_fp16_native (10-bit, rounded) and tomos_fp32_fp16_trunc
    // (9-bit, truncated) below.
    {
        __half *A_half10 = nullptr, *C_half10 = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_half10), m * k * sizeof(__half), "A_half_trunc10",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&C_half10), m * n * sizeof(__half), "C_half_trunc10",
                          shape_desc)) {
            gen_half_truncate10(m * k, A_half10, normal_scale);
            cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half10,
                static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half10,
                static_cast<size_t>(ldc));
            float* C_gpu = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_trunc10",
                              shape_desc)) {
                half_to_float(m * n, C_half10, C_gpu);
                std::vector<float> C_gpu_h(m * n);
                cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                std::printf("    tomos_fp16_trunc10       = %.6e\n", frobenius_rel_err(C_ref_h, C_gpu_h));
                cudaFree(C_gpu);
            }
        }
        cudaFree(A_half10);
        cudaFree(C_half10);
    }

    // Stochastic-rounding series (9-bit mantissa, same as the round_XX/
    // trunc configs below, but rounded via fp32_16_split_stochastic_
    // rounding instead of round-to-nearest or truncation): encode, decode
    // via the same fp32_16_half_only, run through the same native kernel.
    {
        uint16_t *wh_st = nullptr, *wr_st = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh_st), m * k * sizeof(uint16_t), "wh_stochastic",
                          shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr_st), m * k * sizeof(uint16_t), "wr_stochastic",
                          shape_desc)) {
            gen_fp32_16_stochastic(m * k, wh_st, wr_st, normal_scale);

            __half* A_half_st = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_half_st), m * k * sizeof(__half), "A_half_stochastic",
                              shape_desc)) {
                decode_pure_half(m * k, wh_st, A_half_st);

                __half* C_half_st = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&C_half_st), m * n * sizeof(__half), "C_half_stochastic",
                                  shape_desc)) {
                    cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half_st,
                        static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half_st,
                        static_cast<size_t>(ldc));
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_stochastic",
                                      shape_desc)) {
                        half_to_float(m * n, C_half_st, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    tomos_fp32_fp16_stochastic = %.6e\n",
                                   frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                    cudaFree(C_half_st);
                }
                cudaFree(A_half_st);
            }
        }
        cudaFree(wh_st);
        cudaFree(wr_st);
    }

    // The 5 variable-rounding series -- encode via
    // fp32_16_split_variable_rounding (this file's own, at the sweep's
    // own sticky_bits), decode via fp32_16_half_only into a genuine
    // __half array, then run through the SAME native TOMOS FP16 kernel
    // (gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_
    // fastk_precomp2_serp) and the same B_half the native reference
    // series above used -- see this file's own header for why decoding
    // to a real __half array first, rather than calling the separate
    // inline_dirty kernel directly on wh, is deliberate here.
    for (const RoundingConfig& cfg : kConfigs) {
        uint16_t *wh = nullptr, *wr = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&wh), m * k * sizeof(uint16_t), "wh", shape_desc) &&
            alloc_or_skip(reinterpret_cast<void**>(&wr), m * k * sizeof(uint16_t), "wr", shape_desc)) {
            gen_fp32_16_variable(m * k, wh, wr, normal_scale, cfg.sticky_bits);

            __half* A_half2 = nullptr;
            if (alloc_or_skip(reinterpret_cast<void**>(&A_half2), m * k * sizeof(__half), "A_half2", shape_desc)) {
                decode_pure_half(m * k, wh, A_half2);

                __half* C_half2 = nullptr;
                if (alloc_or_skip(reinterpret_cast<void**>(&C_half2), m * n * sizeof(__half), "C_half2",
                                  shape_desc)) {
                    cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                        TransposeType::NO_TRANSPOSE, TransposeType::NO_TRANSPOSE, alpha, m, k, n, A_half2,
                        static_cast<size_t>(lda), B_half, static_cast<size_t>(ldb), beta, C_half2,
                        static_cast<size_t>(ldc));
                    float* C_gpu = nullptr;
                    if (alloc_or_skip(reinterpret_cast<void**>(&C_gpu), m * n * sizeof(float), "C_gpu_round",
                                      shape_desc)) {
                        half_to_float(m * n, C_half2, C_gpu);
                        std::vector<float> C_gpu_h(m * n);
                        cudaMemcpy(C_gpu_h.data(), C_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                        std::printf("    %s = %.6e\n", cfg.marker, frobenius_rel_err(C_ref_h, C_gpu_h));
                        cudaFree(C_gpu);
                    }
                    cudaFree(C_half2);
                }
                cudaFree(A_half2);
            }
        }
        cudaFree(wh);
        cudaFree(wr);
    }

    cudaFree(B_half);
}

}  // namespace

int main(int argc, char** argv) {
    // Extract --normalized (a bare flag, may appear anywhere in argv)
    // before the original positional argument parsing below runs -- see
    // run_shape's own comment for what it does and why. Same extraction
    // pattern mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu's own main()
    // uses.
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

    float normal_scale = (argc > 1) ? static_cast<float>(std::strtod(argv[1], nullptr)) : 1.0f;

    std::printf(
        "=== GEMM: fp32_16 encode-side rounding-precision sweep, TOMOS FP32/FP16 vs FP32 (cublasSgemm) reference "
        "(normal_scale=%g, normalized=%d) ===\n",
        normal_scale, normalized ? 1 : 0);

    const std::vector<size_t> square_sizes = {1024,  1536,  2048,  3072,  4096, 6144, 8192,
                                              12288, 16384, 24576, 32768, 49152, 65536};

    if (argc > 2) {
        size_t n = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
        run_shape(n, n, n, handle, normal_scale, normalized);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t n : square_sizes) run_shape(n, n, n, handle, normal_scale, normalized);

    cublasDestroy(handle);
    return 0;
}

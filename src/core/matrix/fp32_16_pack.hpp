#ifndef TOMOS_FP32_16_PACK_HPP
#define TOMOS_FP32_16_PACK_HPP

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>

#include "precision.hpp"

// Bit-level dissection of an IEEE-754 float32 into a pair of uint16_t words
// (wh, wr), following the same "split the matrix into two halves" idea used
// by randls2.0's C_FP32_FP16 matrix representation:
//
//   wh: sign(1) | exponent(5) | mant_hi(9) | control(1)
//   wr: mant_lo(14) | round_flag(1) | reserved(1)
//
// The exponent(5) field's meaning depends on control:
//   - control=0 ("normal"): a direct rebiased fp16-style exponent,
//     exp16 = fp32_exponent - 112, valid for fp32 exponent in [113,142]
//     (fp16's own normal exponent range).
//   - control=1 ("extended"): the MSB (bit14) is a direction bit
//     (1=overflow, 0=underflow) and the 4 LSBs (bits13-10) are an offset
//     (1-14) counting steps past the [113,142] boundary in that direction
//     (fp32 exponent = 142+offset for overflow, 113-offset for
//     underflow), bit-exact (full 23-bit mantissa preserved) throughout,
//     reserving offset=15 (kOffsetSaturate) as a marker for everything
//     still further out in that direction (deeper overflow/underflow,
//     subnormals, zero, inf, nan) -- those saturate to signed Inf
//     (overflow) or signed 0 (underflow); mantissa bits are don't-care and
//     written 0 by the encoder. There is no longer a third, reduced-
//     precision tier: every value is either exact or a clean 0/Inf
//     saturation.
//
// In the normal and extended (non-saturating) branches, mant_hi is ROUNDED
// (round to nearest, ties to even) from the top of the full 23-bit fp32
// mantissa, rather than truncated -- this is what makes wh, reinterpreted
// alone as a __half bit pattern, a properly-rounded half value instead of
// merely truncated (see fp32_16_half_only/fp32_16_fast_half). round_flag
// (wr's bit0) records whether that rounding stepped up (1) or not (0), so
// fp32_16_combine can invert it and recover mant_lo's sibling mant_hi_pre
// (the UNROUNDED top 9 bits) exactly -- mant_lo itself (wr's top 14 bits)
// is never touched by rounding, so full (wh,wr) reconstruction is always
// exact in these two branches, rounded mant_hi or not. When rounding
// overflows mant_hi (all-ones top 9 bits stepping past 0x1FF), the carry
// bumps the effective exponent by one -- fp32_16_split (the "full carry"
// encoder) re-derives which of normal/extended/saturate applies to that
// bumped exponent (now simply: does it still fit in the wider extended
// field, or does it saturate?), so wh always ends up a self-consistent,
// correctly rounded standalone half even across a normal<->extended
// boundary.
// fp32_16_split_clamped (the "clamped" encoder) instead simply skips the
// round-up in that one narrow case, sidestepping the boundary-crossing
// logic entirely at the cost of a stale truncation in that sliver of
// inputs. fp32_16_combine's decode logic is general enough to correctly
// invert output from either encoder.
//
// wh, reinterpreted directly as a __half bit pattern, is only meaningful
// when control=0 (a valid half value); when control=1 it holds
// direction/offset bits instead of a real half exponent, so fp16-only
// compute needs to check control explicitly rather than blindly
// reinterpreting the whole array.
//
// __host__/__device__ resolve to real CUDA qualifiers under nvcc and to
// nothing under a plain host compiler (via cuda_fp16.h, included through
// precision.hpp), so this single header serves both the CPU and CUDA
// kernels without duplication.

namespace tomos {

namespace fp32_16_detail {

constexpr uint32_t kFp32SignMask = 0x8000'0000u;
constexpr uint32_t kFp32ExpMask = 0x7F80'0000u;
constexpr uint32_t kFp32MantMask = 0x007F'FFFFu;
constexpr uint32_t kFp32ExpMin = 113;  // fp16 normal exponent range, rebiased
constexpr uint32_t kFp32ExpMax = 142;  // to fp32's 8-bit exponent field.
constexpr uint32_t kExpRebias = 112;   // fp32 bias(127) - fp16 bias(15)
constexpr uint32_t kOffsetSaturate = 15;  // reserved marker (see file comment)
constexpr uint32_t kOffsetMax = 14;       // max exact extended-range step

// wh's 4-bit extended-range field stores shift_by = offset-1 (0-13 for a
// real step, kOffsetSaturateStored=14 for the saturate marker), not the
// step-count `offset` (1-15) used throughout this file's own comments
// and the encoders' internal range checks -- every decoder needs the
// shift amount directly (sig10 >> (offset-1) for the pure_half tier;
// kFp32ExpMin/Max are ALSO offset by 1 below to absorb the same
// relabeling for the accurate tier's direct exponent formula), so the
// encoders write offset_eff-1 (or kOffsetSaturateStored for the
// saturate case) as their very last step, after using the ordinary
// offset_eff/kOffsetMax/kOffsetSaturate numbering for every range check
// leading up to it -- purely a storage relabeling, not a range or
// precision change (see fp32_16_half_only_branchless_no_overflow_
// shiftenc's own comment, this file, for the original derivation and
// CPU-exhaustive verification of this being lossless).
constexpr uint32_t kOffsetSaturateStored = kOffsetSaturate - 1u;

__host__ __device__ inline uint32_t float_to_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

__host__ __device__ inline float bits_to_float(uint32_t u) {
    float v;
    std::memcpy(&v, &u, sizeof(float));
    return v;
}

// Round to nearest, ties to even, when discarding mant_lo (the low 14 bits
// of the 23-bit fp32 mantissa) to leave a 9-bit mant_hi. round_bit is the
// discarded bit immediately below the cut (mant_lo's own bit13); sticky is
// whether any bit below THAT is set. mant_hi_pre_is_odd only matters (and
// is only read) at an exact tie (round_bit set, sticky clear).
__host__ __device__ inline bool fp32_16_round_up(uint32_t mant_lo, bool mant_hi_pre_is_odd) {
    bool round_bit = ((mant_lo >> 13) & 0x1u) != 0u;
    bool sticky = (mant_lo & 0x1FFFu) != 0u;
    return round_bit && (sticky || mant_hi_pre_is_odd);
}

}  // namespace fp32_16_detail

// Full-carry encoder: rounds mant_hi to nearest (ties to even), and re-
// derives which of normal/extended/saturate applies to the possibly-
// bumped effective exponent if that rounding overflows mant_hi's 9 bits
// -- so wh always ends up a correctly-rounded, self-consistent standalone
// half value, even across a normal<->extended boundary. See this file's
// own header comment for the full reasoning and wr's bit layout.
__host__ __device__ inline void fp32_16_split(float value, uint16_t& wh,
                                              uint16_t& wr) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    // Values whose exponent is within reach of the exact (normal or
    // extended) encoding at all -- exp32 in [kFp32ExpMin-kOffsetMax,
    // kFp32ExpMax+kOffsetMax] = [99,156] -- are eligible for rounding;
    // this is precisely the union of the normal range and both extended
    // directions' offset<=kOffsetMax windows.
    if (exp32 >= kFp32ExpMin - kOffsetMax && exp32 <= kFp32ExpMax + kOffsetMax) {
        uint32_t mant_hi_pre = mant32 >> 14;
        uint32_t mant_lo = mant32 & 0x3FFFu;
        bool round_up = fp32_16_round_up(mant_lo, (mant_hi_pre & 0x1u) != 0u);

        uint32_t mant_hi_rounded = mant_hi_pre + (round_up ? 1u : 0u);
        bool overflow = mant_hi_rounded > 0x1FFu;  // 0x1FF + 1 = 0x200, too wide for 9 bits
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
        uint32_t offset_eff = (exp32_eff > kFp32ExpMax) ? (exp32_eff - kFp32ExpMax)
                                                         : (kFp32ExpMin - exp32_eff);
        if (offset_eff >= 1u && offset_eff <= kOffsetMax) {
            wh = static_cast<uint16_t>(sign | (direction_eff << 14) | ((offset_eff - 1u) << 10) |
                                       (mant_hi_written << 1) | 1u);
            wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
            return;
        }
        // The carry pushed exp32_eff past even the extended range's own
        // outermost edge (needs mant_hi_pre==0x1FF, a round-up decision,
        // AND exp32 already at that edge -- an astronomically narrow
        // band). Falls through to the saturate path below using
        // direction_eff -- unlike the old degraded fallback, saturation
        // doesn't need the untouched original value, so there's nothing
        // special to preserve here.
        wh = static_cast<uint16_t>(sign | (direction_eff << 14) | (kOffsetSaturateStored << 10) | 1u);
        wr = 0;
        return;
    }

    // Saturate: this value's exponent is beyond even the extended range's
    // own reach in this direction (covers deeper overflow/underflow, fp32
    // subnormals, exact zero, inf, and nan -- all of which land here via
    // the exp32 bounds check above without any special-casing). Reconstructs
    // to exactly signed Inf (overflow) or signed 0 (underflow); nan loses
    // its payload and saturates to Inf like any other overflow, since this
    // marker has no spare bits to flag "was actually nan".
    uint32_t direction = (exp32 > kFp32ExpMax) ? 1u : 0u;
    wh = static_cast<uint16_t>(sign | (direction << 14) | (kOffsetSaturateStored << 10) | 1u);
    wr = 0;
}

// Clamped encoder: identical to fp32_16_split except when rounding would
// overflow mant_hi (top-9 bits == 0x1FF, round-up decision) -- there, it
// simply skips the round-up (round_flag=0, mant_hi stays 0x1FF, exponent
// untouched) rather than re-deriving which mode applies to a bumped
// exponent. Sidesteps the normal<->extended boundary-crossing logic
// entirely, at the cost of a stale truncation confined to that one narrow
// band of inputs (never worse than fp32_16_split's own pre-rounding
// behavior there). fp32_16_combine decodes output from either encoder.
__host__ __device__ inline void fp32_16_split_clamped(float value, uint16_t& wh,
                                                       uint16_t& wr) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    if (exp32 >= kFp32ExpMin && exp32 <= kFp32ExpMax) {
        uint32_t mant_hi_pre = mant32 >> 14;
        uint32_t mant_lo = mant32 & 0x3FFFu;
        bool round_up = fp32_16_round_up(mant_lo, (mant_hi_pre & 0x1u) != 0u);
        bool overflow = round_up && mant_hi_pre == 0x1FFu;
        uint32_t mant_hi_written = overflow ? mant_hi_pre : (mant_hi_pre + (round_up ? 1u : 0u));
        uint32_t round_flag = overflow ? 0u : (round_up ? 1u : 0u);

        uint32_t exp16 = exp32 - kExpRebias;
        wh = static_cast<uint16_t>(sign | (exp16 << 10) | (mant_hi_written << 1) | 0u);
        wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
        return;
    }

    uint32_t direction = (exp32 > kFp32ExpMax) ? 1u : 0u;
    uint32_t offset = (exp32 > kFp32ExpMax) ? (exp32 - kFp32ExpMax)
                                            : (kFp32ExpMin - exp32);

    if (offset <= kOffsetMax) {
        uint32_t mant_hi_pre = mant32 >> 14;
        uint32_t mant_lo = mant32 & 0x3FFFu;
        bool round_up = fp32_16_round_up(mant_lo, (mant_hi_pre & 0x1u) != 0u);
        bool overflow = round_up && mant_hi_pre == 0x1FFu;
        uint32_t mant_hi_written = overflow ? mant_hi_pre : (mant_hi_pre + (round_up ? 1u : 0u));
        uint32_t round_flag = overflow ? 0u : (round_up ? 1u : 0u);

        wh = static_cast<uint16_t>(sign | (direction << 14) | ((offset - 1u) << 10) |
                                   (mant_hi_written << 1) | 1u);
        wr = static_cast<uint16_t>((mant_lo << 2) | round_flag);
        return;
    }

    // Saturate: exp32 is beyond kOffsetMax steps past the extended range's
    // own edge (covers deeper overflow/underflow, fp32 subnormals, exact
    // zero, inf, nan -- all land here via the offset check above, no
    // special-casing needed). Reconstructs to exactly signed Inf/0; nan
    // saturates to Inf like any other overflow. See fp32_16_split's own
    // header comment for the full reasoning.
    wh = static_cast<uint16_t>(sign | (direction << 14) | (kOffsetSaturateStored << 10) | 1u);
    wr = 0;
}

// Truncating encoder: the plain, pre-rounding-feature behavior -- mant_hi
// is simply mant32's top 9 bits, never rounded, wr's round_flag bit
// always 0. Exists specifically to pair with fp32_16_combine_truncated
// below for a fast-decode comparison point: since round_flag is always
// 0 here, that decode can skip the is_carry correction entirely instead
// of computing (and discarding) it. wh alone is therefore back to being
// a merely-truncated (not properly-rounded) half value for this
// encoder's output -- see fp32_16_split's own comment for what that
// costs the fp16-only pure_half tier. Only ever pair this with
// fp32_16_combine_truncated (or fp32_16_combine, which also handles
// round_flag==0 correctly) -- never decode this encoder's output with
// code that assumes fp32_16_split/fp32_16_split_clamped's rounding
// contract holds, since it doesn't apply here.
__host__ __device__ inline void fp32_16_split_truncated(float value, uint16_t& wh,
                                                         uint16_t& wr) {
    using namespace fp32_16_detail;
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits & kFp32SignMask) >> 16;
    uint32_t exp32 = (bits & kFp32ExpMask) >> 23;
    uint32_t mant32 = bits & kFp32MantMask;

    if (exp32 >= kFp32ExpMin && exp32 <= kFp32ExpMax) {
        uint32_t exp16 = exp32 - kExpRebias;
        uint32_t mant_hi = mant32 >> 14;
        wh = static_cast<uint16_t>(sign | (exp16 << 10) | (mant_hi << 1) | 0u);
        wr = static_cast<uint16_t>((mant32 & 0x3FFFu) << 2);
        return;
    }

    uint32_t direction = (exp32 > kFp32ExpMax) ? 1u : 0u;
    uint32_t offset = (exp32 > kFp32ExpMax) ? (exp32 - kFp32ExpMax)
                                            : (kFp32ExpMin - exp32);

    if (offset <= kOffsetMax) {
        uint32_t mant_hi = mant32 >> 14;
        wh = static_cast<uint16_t>(sign | (direction << 14) | ((offset - 1u) << 10) |
                                   (mant_hi << 1) | 1u);
        wr = static_cast<uint16_t>((mant32 & 0x3FFFu) << 2);
        return;
    }

    // Saturate: exp32 is beyond kOffsetMax steps past the extended range's
    // own edge (covers deeper overflow/underflow, fp32 subnormals, exact
    // zero, inf, nan -- all land here via the offset check above, no
    // special-casing needed). Reconstructs to exactly signed Inf/0; nan
    // saturates to Inf like any other overflow. See fp32_16_split's own
    // header comment for the full reasoning.
    wh = static_cast<uint16_t>(sign | (direction << 14) | (kOffsetSaturateStored << 10) | 1u);
    wr = 0;
}

__host__ __device__ inline float fp32_16_combine(uint16_t wh, uint16_t wr) {
    using namespace fp32_16_detail;
    uint32_t sign = (static_cast<uint32_t>(wh) & 0x8000u) << 16;
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t mant_lo = (static_cast<uint32_t>(wr) >> 2) & 0x3FFFu;
    uint32_t round_flag = static_cast<uint32_t>(wr) & 0x1u;

    // Invert fp32_16_split's rounding: mant_hi==0 with round_flag set is
    // exactly (and only) the overflow-carry signature (0x1FF+1 wrapped),
    // never a genuine "rounded from mant_hi_pre=-1" -- so it alone
    // disambiguates the carry case from the plain "subtract 1" case,
    // whichever of fp32_16_split/fp32_16_split_clamped produced these
    // bits (the clamped encoder never sets round_flag when mant_hi would
    // need to wrap, so this never misfires on its output). is_carry is
    // computed once per branch below (each branch used to recompute this
    // same condition twice) -- scoped inside !control/offset!=saturate
    // rather than hoisted above them, so the saturate branch (which
    // never needed this at all) still doesn't pay for it.
    //
    // TRIED AND REVERTED: packing exp32/mant_hi into one combined integer
    // and subtracting round_flag from that (letting ordinary binary
    // borrow propagation do the carry correction, no is_carry/select at
    // all) compiled to fewer instructions and zero SEL/fewer BRA in this
    // kernel's SASS -- but measured ~1-2% SLOWER at N>=2048 than this
    // is_carry version, consistently across every size. Best guess: the
    // two SELs here both depend only on is_carry and can likely issue
    // independently, while the combined-subtract version is a strictly
    // serial dependency chain (build combined -> subtract -> shift ->
    // OR) -- fewer static instructions isn't the same as a shorter
    // critical path. Kept as a documented negative result; see git log
    // for the reverted version if revisiting this.
    if (!control) {
        uint32_t exp16 = (static_cast<uint32_t>(wh) >> 10) & 0x1Fu;
        bool is_carry = (round_flag != 0u) && (mant_hi == 0u);
        uint32_t mant_hi_pre = is_carry ? 0x1FFu : (mant_hi - round_flag);
        uint32_t exp32 = exp16 + kExpRebias - (is_carry ? 1u : 0u);
        uint32_t mant32 = (mant_hi_pre << 14) | mant_lo;
        return bits_to_float(sign | (exp32 << 23) | mant32);
    }

    // wh's field holds shift_by = offset-1 (see kOffsetSaturateStored's
    // own comment) -- kFp32ExpMax+1/kFp32ExpMin-1 absorb that relabeling
    // so the reconstructed exponent is identical to the pre-renumbering
    // kFp32ExpMax+offset/kFp32ExpMin-offset formula.
    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    if (shift_by != kOffsetSaturateStored) {
        bool is_carry = (round_flag != 0u) && (mant_hi == 0u);
        uint32_t mant_hi_pre = is_carry ? 0x1FFu : (mant_hi - round_flag);
        uint32_t exp32 = (direction ? (kFp32ExpMax + 1u + shift_by)
                                    : (kFp32ExpMin - 1u - shift_by)) - (is_carry ? 1u : 0u);
        uint32_t mant32 = (mant_hi_pre << 14) | mant_lo;
        return bits_to_float(sign | (exp32 << 23) | mant32);
    }

    // Saturate marker: exactly signed Inf (overflow) or signed 0 (underflow).
    return bits_to_float(direction ? (sign | 0x7F80'0000u) : sign);
}

// Fast decode for the "normal" (control=0) branch ONLY -- identical body
// to fp32_16_combine's own !control branch (same is_carry correction, so
// still exact/round_flag-aware), but skips the control-bit dispatch and
// the offset/direction/saturate branches entirely. CORRECTNESS CONTRACT:
// caller MUST guarantee wh's control bit (bit0) is 0 for every input --
// this function never checks it, and will silently misinterpret an
// extended/saturate value's offset/direction bits as exponent bits if
// called on one. Intended for a kernel-level fast path
// that first confirms (e.g. via a block-wide reduction over a tile's
// staged control bits) that an entire tile is control=0 before calling
// this instead of fp32_16_combine -- see mm_fp32_16_tiled_cpasync_
// dualbuf_adaptive_kernel in fp32_16_cuda_kernels.cu, which does exactly
// that and falls back to fp32_16_combine per-tile when it isn't.
__host__ __device__ inline float fp32_16_combine_fast_normal(uint16_t wh, uint16_t wr) {
    using namespace fp32_16_detail;
    uint32_t sign = (static_cast<uint32_t>(wh) & 0x8000u) << 16;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t mant_lo = (static_cast<uint32_t>(wr) >> 2) & 0x3FFFu;
    uint32_t round_flag = static_cast<uint32_t>(wr) & 0x1u;
    uint32_t exp16 = (static_cast<uint32_t>(wh) >> 10) & 0x1Fu;
    bool is_carry = (round_flag != 0u) && (mant_hi == 0u);
    uint32_t mant_hi_pre = is_carry ? 0x1FFu : (mant_hi - round_flag);
    uint32_t exp32 = exp16 + kExpRebias - (is_carry ? 1u : 0u);
    uint32_t mant32 = (mant_hi_pre << 14) | mant_lo;
    return bits_to_float(sign | (exp32 << 23) | mant32);
}

// EXPERIMENTAL, standalone sibling of fp32_16_combine_fast_normal above
// -- computes the EXACT SAME result via a different bit trick, replacing
// the is_carry ternary/SEL logic (2 comparisons, 1 AND, 2 SELECTs at the
// SASS level -- confirmed via cuobjdump, see mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_vec4_kernel's own SASS profile in
// fp32_16_cuda_kernels.cu) with a single unsigned WRAPAROUND subtraction:
// `mant_hi - round_flag`, when mant_hi==0 and round_flag==1, underflows
// to 0xFFFFFFFF in unsigned 32-bit arithmetic -- masking that with 0x1FF
// recovers the SAME 0x1FF the original ternary explicitly selects for
// mant_hi_pre, and shifting right by 9 then masking with 1 recovers the
// SAME borrow-into-exponent flag the original computes via mant_hi==0 &&
// round_flag!=0 -- with NO comparison or select instruction needed for
// either quantity, since the borrow is already latent in the wrapped
// subtraction's high bits. Exhaustively verified bit-identical to
// fp32_16_combine_fast_normal over all 1024 (mant_hi, round_flag)
// combinations (mant_hi is 9 bits, round_flag is 1 bit -- the only two
// inputs this branch's logic depends on) before this was written.
//
// UNTESTED at the SASS/hardware level -- see its own kernel-level A/B
// sibling, mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
// wrapborrow_vec4, in fp32_16_cuda_kernels.cu.
__host__ __device__ inline float fp32_16_combine_fast_normal_wrapborrow(uint16_t wh, uint16_t wr) {
    using namespace fp32_16_detail;
    uint32_t sign = (static_cast<uint32_t>(wh) & 0x8000u) << 16;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t mant_lo = (static_cast<uint32_t>(wr) >> 2) & 0x3FFFu;
    uint32_t round_flag = static_cast<uint32_t>(wr) & 0x1u;
    uint32_t exp16 = (static_cast<uint32_t>(wh) >> 10) & 0x1Fu;
    uint32_t sub = mant_hi - round_flag;
    uint32_t borrow = (sub >> 9) & 0x1u;
    uint32_t mant_hi_pre = sub & 0x1FFu;
    uint32_t exp32 = exp16 + kExpRebias - borrow;
    uint32_t mant32 = (mant_hi_pre << 14) | mant_lo;
    return bits_to_float(sign | (exp32 << 23) | mant32);
}

// Same contract and result as fp32_16_combine_branchless (fp32_16_cuda_
// kernels.cu): standalone, header-only (non-CUDA-file) branchless twin,
// built from ordinary integer ALU ops only, same normal/extended/saturate
// mask-select merge technique. Historically this function's distinguishing
// feature was reimplementing the old degraded branch's real-__half-cast
// reconstruction as pure bit manipulation, to avoid a measured-expensive
// hardware HADD2.F32 path; the saturate branch that replaced degraded
// needs no such thing (it's just a mask-selected constant), so that
// rationale no longer applies -- this function is now nearly identical to
// fp32_16_combine_branchless and may be redundant with it going forward.
//
// CORRECTNESS: verified bit-exact against fp32_16_combine across every
// fp32 exponent 0..255 with many random mantissa/sign trials each,
// including the saturate-bucket sub-cases (deep overflow/underflow,
// subnormal/zero/inf/nan fp32 inputs) -- see Fp32_16PackTest.
// CombineBranchlessNohwMatchesCombine in test/core/matrix/test_fp32_16.cpp.
__host__ __device__ inline float fp32_16_combine_branchless_nohw(uint16_t wh, uint16_t wr) {
    using namespace fp32_16_detail;
    uint32_t whb = static_cast<uint32_t>(wh);
    uint32_t wrb = static_cast<uint32_t>(wr);
    uint32_t sign = (whb & 0x8000u) << 16;
    uint32_t control = whb & 0x1u;
    uint32_t mant_hi = (whb >> 1) & 0x1FFu;
    uint32_t mant_lo = (wrb >> 2) & 0x3FFFu;

    // Same is_carry correction as fp32_16_combine_branchless's own normal/
    // extended paths -- unchanged, see that function's own comment.
    uint32_t round_flag = wrb & 0x1u;
    uint32_t is_carry = static_cast<uint32_t>(round_flag != 0u && mant_hi == 0u);
    uint32_t mask_carry = 0u - is_carry;
    uint32_t mant_hi_pre = (mask_carry & 0x1FFu) | (~mask_carry & (mant_hi - round_flag));
    uint32_t mant32 = (mant_hi_pre << 14) | mant_lo;
    uint32_t exp_adjust = is_carry;

    // Normal (control == 0).
    uint32_t exp16 = (whb >> 10) & 0x1Fu;
    uint32_t bitsNormal = sign | (((exp16 + kExpRebias) - exp_adjust) << 23) | mant32;

    // Extended (control == 1, wh's field holds shift_by = offset-1; see
    // kOffsetSaturateStored's own comment).
    uint32_t shift_by = (whb >> 10) & 0xFu;
    uint32_t direction = (whb >> 14) & 0x1u;
    uint32_t expExtended = direction ? (kFp32ExpMax + 1u + shift_by) : (kFp32ExpMin - 1u - shift_by);
    uint32_t bitsExtended = sign | ((expExtended - exp_adjust) << 23) | mant32;

    // Saturate (control == 1, shift_by == kOffsetSaturateStored): exactly
    // signed Inf/0 -- no hardware half-conversion needed anymore (that
    // was only ever required by the old degraded branch's real-half-cast
    // reconstruction), so this whole merge is already pure ALU, same as
    // the rest of this function.
    uint32_t bitsSaturate = direction ? (sign | 0x7F80'0000u) : sign;

    uint32_t is_saturate = static_cast<uint32_t>(shift_by == kOffsetSaturateStored);
    uint32_t mask_saturate = 0u - is_saturate;
    uint32_t mask_control = 0u - control;
    uint32_t bits_control1 = (mask_saturate & bitsSaturate) | (~mask_saturate & bitsExtended);
    uint32_t result_bits = (mask_control & bits_control1) | (~mask_control & bitsNormal);
    return bits_to_float(result_bits);
}

// Fast decode paired with fp32_16_split_truncated: no is_carry
// correction at all (not computed, not selected on) -- mant_hi is used
// as-is, exactly the fp32_16_combine logic that existed before mant_hi
// rounding was added to this format. CORRECTNESS CONTRACT: only valid
// on (wh,wr) produced by fp32_16_split_truncated. Decoding output from
// fp32_16_split/fp32_16_split_clamped with this function will silently
// reconstruct the WRONG value for any element where those encoders
// rounded mant_hi up (round_flag=1) -- it has no way to detect or
// correct for that, by design, since skipping that check is the entire
// point of this fast path. Exists to A/B this format's decode cost
// against fp32_16_combine's round_flag-aware version at the same
// problem sizes -- see mm_fp32_16_tiled_cpasync_dualbuf_truncated in
// fp32_16_cuda_kernels.cu, benchmarked head-to-head with mm_fp32_16_
// tiled_cpasync_dualbuf in benchmarks_gemm_fp32.cu.
__host__ __device__ inline float fp32_16_combine_truncated(uint16_t wh, uint16_t wr) {
    using namespace fp32_16_detail;
    uint32_t sign = (static_cast<uint32_t>(wh) & 0x8000u) << 16;
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t mant_lo = (static_cast<uint32_t>(wr) >> 2) & 0x3FFFu;

    if (!control) {
        uint32_t exp16 = (static_cast<uint32_t>(wh) >> 10) & 0x1Fu;
        uint32_t exp32 = exp16 + kExpRebias;
        uint32_t mant32 = (mant_hi << 14) | mant_lo;
        return bits_to_float(sign | (exp32 << 23) | mant32);
    }

    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    if (shift_by != kOffsetSaturateStored) {
        uint32_t exp32 = direction ? (kFp32ExpMax + 1u + shift_by) : (kFp32ExpMin - 1u - shift_by);
        uint32_t mant32 = (mant_hi << 14) | mant_lo;
        return bits_to_float(sign | (exp32 << 23) | mant32);
    }

    // Saturate marker: exactly signed Inf (overflow) or signed 0 (underflow).
    return bits_to_float(direction ? (sign | 0x7F80'0000u) : sign);
}

// Correctly decodes wh alone (never touching wr) into the best available
// __half approximation, explicitly handling both encodings rather than
// blindly reinterpreting the bits (which is only valid when control=0):
//   - control=0: wh's bits are directly a valid, properly-rounded half
//     value (fp32_16_split/fp32_16_split_clamped round mant_hi to nearest,
//     ties to even, rather than truncating it).
//   - control=1, direction=1 (overflow): the true value's fp32 exponent is
//     already beyond kFp32ExpMax, so it's unconditionally beyond fp16's own
//     representable range regardless of the specific offset value stored
//     (even offset=1 here is already past fp16's own max) -- always
//     +-Infinity.
//   - control=1, direction=0 (underflow): wh bits13-10 store shift_by =
//     offset-1 (see kOffsetSaturateStored's own comment) -- exactly the
//     real fp16 subnormal shift count already, no adjustment needed at
//     decode time -- so this reconstructs a genuine fp16 subnormal by
//     shifting the implicit-1-prefixed 10-bit significand (mant_hi's 9
//     bits, since wr's mant_lo isn't available to this wh-only function)
//     right by shift_by positions, exactly mirroring real IEEE subnormal
//     encoding. For shift_by>=10 (including the shift_by=14 saturate
//     marker) this shift naturally produces 0 with no extra branch
//     needed -- fp16 itself has only 10 subnormal positions, so anything
//     requiring more shift than that is correctly below fp16's own
//     smallest subnormal too.
// OVERFLOW/UNDERFLOW-DISTINGUISHING: reads `direction` (bit14) and branches
// on it, so overflow (control=1, direction=1, saturates to signed Infinity)
// and underflow (control=1, direction=0, shift-reconstructed subnormal) are
// handled correctly and distinctly. Safe to use on data with a nonzero
// overflow rate. Contrast with the `_no_overflow`/`_nomask` siblings below,
// which drop this check under a caller-guaranteed no-overflow contract.
__host__ __device__ inline __half fp32_16_half_only(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;
    if (!control) {
        return *reinterpret_cast<const __half*>(&wh);
    }
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    uint16_t result_bits;
    if (direction) {
        result_bits = static_cast<uint16_t>(sign_bits | 0x7C00u);
    } else {
        uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
        uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
        uint32_t sig10 = 0x200u | mant_hi;
        uint32_t half_mant = sig10 >> shift_by;
        result_bits = static_cast<uint16_t>(sign_bits | (half_mant & 0x3FFu));
    }
    return *reinterpret_cast<const __half*>(&result_bits);
}

// Cheaper sibling of fp32_16_half_only above: skips the shift-based fp16
// subnormal reconstruction entirely, saturating every control=1 value
// straight to +-Infinity (overflow) or +-0 (underflow) regardless of the
// stored offset. Loses the subset of underflowed values fp32_16_half_only
// can still recover as genuine fp16 subnormals (offset in [1,10]) -- an
// intentional accuracy-for-speed trade for a fast-path kernel that doesn't
// need that last sliver of dynamic range.
//
// LOSSY, NOT part of the no-overflow decode migration: this is a separate,
// already-established-as-problematic accuracy tradeoff (saturates ALL
// underflow to zero, previously implicated in LSQR convergence stalls on
// ill-conditioned real data) -- do not conflate with the _no_overflow/
// _nomask siblings below, which are lossless under a no-overflow contract.
__host__ __device__ inline __half fp32_16_half_only_fast(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;
    if (!control) {
        return *reinterpret_cast<const __half*>(&wh);
    }
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    uint16_t result_bits = static_cast<uint16_t>(sign_bits | (direction ? 0x7C00u : 0u));
    return *reinterpret_cast<const __half*>(&result_bits);
}

// Middle ground between fp32_16_half_only (full shift-based subnormal
// reconstruction, two sequential branches: control then direction) and
// fp32_16_half_only_fast (saturates the ENTIRE underflow band to signed
// zero, one branch): collapses the "do I need the shift" test into a
// SINGLE branch (is_underflow = control & !direction, computed branchless
// via bitwise ops -- safe even when control==0, since direction's own bit
// is then meaningless raw __half exponent data, but ANDing with control
// masks it out regardless, same "compute unconditionally, select via
// mask" idiom fp32_16_combine's own branchless paths already use), while
// STILL doing the real shift-based reconstruction for genuine underflow
// (offset in [1,10] recovers an exact fp16 subnormal, same as
// fp32_16_half_only) -- keeping the accuracy fast decode gives up, while
// still cutting fp32_16_half_only's second (direction) branch for every
// control=0 and every overflow element. Overflow (control=1, direction=1)
// and normal (control=0) both fall through the SAME cheap, branchless
// path (a select between the raw reinterpret and a signed-Inf pattern),
// same as fp32_16_half_only_fast's own cheap path.
//
// OVERFLOW/UNDERFLOW-DISTINGUISHING: is_underflow reads `direction` (bit14)
// explicitly, so overflow still saturates to signed Infinity via the cheap
// path rather than being misread as underflow. Safe to use on data with a
// nonzero overflow rate, same contract as fp32_16_half_only above.
__host__ __device__ inline __half fp32_16_half_only_fast2(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;
    uint32_t is_underflow = control & (1u - direction);
    if (is_underflow) {
        uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
        uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
        uint32_t sig10 = 0x200u | mant_hi;
        uint32_t half_mant = sig10 >> shift_by;
        uint16_t result_bits = static_cast<uint16_t>(sign_bits | (half_mant & 0x3FFu));
        return *reinterpret_cast<const __half*>(&result_bits);
    }
    uint16_t cheap_bits = control ? static_cast<uint16_t>(sign_bits | 0x7C00u) : wh;
    return *reinterpret_cast<const __half*>(&cheap_bits);
}

// Genuinely branchless sibling of fp32_16_half_only_fast2 above: computes
// BOTH the cheap path and the shift-based underflow reconstruction
// UNCONDITIONALLY for every element, then blends via a bitmask instead of
// an `if` on is_underflow (same "compute unconditionally, select via
// mask" idiom fp32_16_combine's own branchless paths already use).
// Motivation, not present when fast2 was written: sampling the real LSQR
// test matrix (tall_thin_1000.mtx) found a ~18% underflow rate -- with
// per-lane probability that far from 0 or 1, the odds all 32 lanes in a
// warp agree are negligible, so virtually every warp already pays for
// BOTH of fast2's paths serially via divergence; branchless removes that
// divergence (and its own branch/predication overhead) entirely, at the
// cost of always doing the shift-based work even when not needed --
// roughly a wash on old ~0%-underflow synthetic data, a real win on data
// with any meaningfully-nonzero underflow rate. wh's field holds
// shift_by directly (see kOffsetSaturateStored's own comment), always
// in [0,14] -- no zero-guard needed even in the discarded (non-
// underflow) case, unlike when this field held the 1-14 step-count
// numbering.
//
// OVERFLOW/UNDERFLOW-DISTINGUISHING: is_underflow is computed from `control
// & !direction`, so overflow (direction=1) still resolves to `cheap_bits`'s
// signed-Inf pattern rather than being misread as underflow, even though
// both paths are computed unconditionally and blended via mask. Safe to
// use on data with a nonzero overflow rate, same contract as
// fp32_16_half_only above.
__host__ __device__ inline __half fp32_16_half_only_branchless(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t direction = (static_cast<uint32_t>(wh) >> 14) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;
    uint32_t is_underflow = control & (1u - direction);

    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t sig10 = 0x200u | mant_hi;
    uint32_t half_mant = sig10 >> shift_by;
    uint32_t underflow_bits = sign_bits | (half_mant & 0x3FFu);

    uint32_t cheap_bits = control ? (sign_bits | 0x7C00u) : static_cast<uint32_t>(wh);

    uint32_t mask = 0u - is_underflow;
    uint16_t result_bits = static_cast<uint16_t>((underflow_bits & mask) | (cheap_bits & ~mask));
    return *reinterpret_cast<const __half*>(&result_bits);
}

// No-overflow-guaranteed sibling of fp32_16_half_only_branchless above:
// requires the CALLER to guarantee no wh in the buffer has control=1,
// direction=1 (a genuine overflow, magnitude beyond fp16's own ~65504
// max) -- same caller-contract axis fp32_16_combine_branchless_no_
// overflow already uses for the wh+wr "accurate" tier, applied here to
// the wh-only "pure_half" tier. Under that guarantee, is_underflow
// reduces to just `control` (direction is always 0 whenever control=1),
// so this drops direction's own extraction AND the cheap_bits select on
// control entirely (control=1 always means underflow, control=0 always
// means passthrough -- `wh` itself already IS cheap_bits in that case).
// Fewer live values than the general branchless version above (no
// separate direction/cheap_bits), not just fewer instructions -- verified
// bit-exact against fp32_16_half_only_fast2 on every wh value NOT
// excluded by this contract (overflow cases, and offset=0 which is
// separately unreachable from fp32_16_split regardless of direction).
// MEASURED on the real LSQR test matrix (tall_thin_1000.mtx, 0%
// overflow / ~18% underflow, two independent 2M-value samples): a real,
// if still incomplete, further improvement over fp32_16_half_only_
// branchless's own realistic-underflow number. wh's field now holds
// shift_by directly (see kOffsetSaturateStored's own comment), so this
// is now identical in structure to fp32_16_half_only_branchless_no_
// overflow_shiftenc below -- that sibling predates this function's own
// migration to the shift_by numbering and is now redundant with it,
// kept only so existing callers/benchmarks referencing it by name still
// resolve.
__host__ __device__ inline __half fp32_16_half_only_branchless_no_overflow(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;

    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t sig10 = 0x200u | mant_hi;
    uint32_t half_mant = sig10 >> shift_by;
    uint32_t underflow_bits = sign_bits | (half_mant & 0x3FFu);

    uint32_t mask = 0u - control;
    uint16_t result_bits = static_cast<uint16_t>((underflow_bits & mask) | (static_cast<uint32_t>(wh) & ~mask));
    return *reinterpret_cast<const __half*>(&result_bits);
}

// EXPERIMENTAL sibling of fp32_16_half_only_branchless_no_overflow above:
// drops the `& 0x3FFu` mask on half_mant, which is PROVABLY redundant,
// not just usually true -- sig10 = 0x200|mant_hi is always in [0x200,
// 0x3FF] (mant_hi is already masked to 9 bits, and OR-ing in bit9
// unconditionally caps sig10 at exactly 10 significant bits), and
// right-shifting a value that already fits in 10 bits by any amount can
// only shrink it, never grow it past 10 bits -- so `(sig10>>shift_by) &
// 0x3FFu` and `sig10>>shift_by` are IDENTICAL for every one of the 8192
// possible (mant_hi, shift_by) combinations (exhaustively verified on a
// host CPU harness before this was written, not just reasoned through).
// This isn't a source-level nicety either: cuobjdump on the ALREADY-
// promoted no_overflow kernel showed this mask compiled to a real,
// separate LOP3.LUT instruction (ptxas does NOT eliminate it on its
// own) -- one full instruction per decoded element that provably never
// changes the result. Test-only for now (mv_fp32_16_pure_half_vs_
// cublas_ref.cu's own "test-new" leg, both of its dispatched kernels),
// not yet promoted to the base function pending real-hardware
// measurement.
__host__ __device__ inline __half fp32_16_half_only_branchless_no_overflow_nomask(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;

    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t sig10 = 0x200u | mant_hi;
    uint32_t half_mant = sig10 >> shift_by;
    uint32_t underflow_bits = sign_bits | half_mant;

    uint32_t mask = 0u - control;
    uint16_t result_bits = static_cast<uint16_t>((underflow_bits & mask) | (static_cast<uint32_t>(wh) & ~mask));
    return *reinterpret_cast<const __half*>(&result_bits);
}

// Byte-for-byte identical to fp32_16_half_only_branchless_no_overflow
// above now that fp32_16_split's own numbering is shift_by (offset-1)
// directly -- this function predates that migration (it was built and
// verified first, as an isolated experiment, before the shift_by
// numbering became the standard encoding everywhere -- see
// kOffsetSaturateStored's own comment for the original derivation and
// CPU-exhaustive verification). Kept only so existing callers/
// benchmarks referencing it by name still resolve.
__host__ __device__ inline __half fp32_16_half_only_branchless_no_overflow_shiftenc(uint16_t wh) {
    uint32_t control = static_cast<uint32_t>(wh) & 0x1u;
    uint32_t sign_bits = static_cast<uint32_t>(wh) & 0x8000u;

    uint32_t shift_by = (static_cast<uint32_t>(wh) >> 10) & 0xFu;
    uint32_t mant_hi = (static_cast<uint32_t>(wh) >> 1) & 0x1FFu;
    uint32_t sig10 = 0x200u | mant_hi;
    uint32_t half_mant = sig10 >> shift_by;
    uint32_t underflow_bits = sign_bits | (half_mant & 0x3FFu);

    uint32_t mask = 0u - control;
    uint16_t result_bits = static_cast<uint16_t>((underflow_bits & mask) | (static_cast<uint32_t>(wh) & ~mask));
    return *reinterpret_cast<const __half*>(&result_bits);
}

// Decodes wh alone (never touching wr) into a __nv_bfloat16, preserving
// magnitude information fp32_16_half_only's own extended-range branch
// has to discard: there, any value outside fp16's own narrow 5-bit-
// exponent range saturates to +-Infinity (or flushes to +-0); here,
// since bf16's 8-bit exponent field is bit-for-bit the same width AND
// bias as fp32's own (see fp32_bf16_pack.hpp's own header), the
// control/offset/direction bits alone are enough to reconstruct the
// value's TRUE fp32 exponent -- no rebiasing needed, same arithmetic
// fp32_16_combine's own extended branch already uses. Only mant_hi (9
// bits) is available for the mantissa without wr, rounded down (to
// nearest, ties to even) to bf16's own 7-bit mantissa -- a second
// rounding pass on top of fp32_16_split's own mant_hi rounding, so this
// is not bit-exact the way a genuine fp32->bf16 conversion of the
// original value would be, only as accurate as mant_hi's own 9 bits
// allow.
//
//   - control=0 ("normal"): wh's bits are a valid, properly-rounded half
//     value (see fp32_16_split's own header) -- converted to bf16 via a
//     real half->float->bf16 widen (CUDA's own conversion intrinsics),
//     NOT a bit reinterpret -- half and bf16 have incompatible bit
//     layouts (5-bit vs 8-bit exponent), unlike wh's own direct
//     half-reinterpretation validity in fp32_16_half_only.
//   - control=1, offset in [1,kOffsetMax] ("extended"): the true fp32
//     exponent is recovered from direction+offset alone (no wr needed),
//     packed directly into bf16's own exponent field -- no rebias, since
//     bf16 and fp32 share the same 127 bias and 8-bit width -- with
//     mant_hi's 9 bits rounded down to bf16's own 7 (including the rare
//     mantissa-overflow carry into the exponent).
//   - control=1, offset=kOffsetSaturate ("saturate"): this value is beyond
//     what the container ever stored, for ANY consumer format -- the
//     encoder already discarded it, regardless of the fact that bf16's own
//     8-bit exponent could in principle have represented it exactly. Decodes
//     to signed Inf (overflow) or signed 0 (underflow), matching the
//     fp32-exact decoders' own saturate branch -- direction is always
//     available here (unlike the old degraded branch, which repurposed
//     wh[14] as a mantissa bit and so couldn't distinguish huge from tiny).
__host__ __device__ inline __nv_bfloat16 fp32_16_bf16_only(uint16_t wh) {
    using namespace fp32_16_detail;
    uint32_t bits = static_cast<uint32_t>(wh);
    uint32_t control = bits & 0x1u;
    uint32_t sign16 = (bits >> 15) & 0x1u;

    if (!control) {
        __half h = *reinterpret_cast<const __half*>(&wh);
        return __float2bfloat16(__half2float(h));
    }

    uint32_t shift_by = (bits >> 10) & 0xFu;
    uint32_t direction = (bits >> 14) & 0x1u;
    if (shift_by == kOffsetSaturateStored) {
        uint16_t sat_bits = static_cast<uint16_t>((sign16 << 15) | (direction ? 0x7F80u : 0u));
        return *reinterpret_cast<const __nv_bfloat16*>(&sat_bits);
    }

    uint32_t exp32 = direction ? (kFp32ExpMax + 1u + shift_by) : (kFp32ExpMin - 1u - shift_by);
    uint32_t mant_hi = (bits >> 1) & 0x1FFu;

    // Round mant_hi (9 bits) down to bf16's own 7-bit mantissa, to
    // nearest, ties to even -- the 2 bits being dropped are exact (no
    // sticky bits below them, since mant_hi is itself already the
    // rounded top of the real 23-bit fp32 mantissa fp32_16_split kept).
    bool round_bit = (mant_hi & 0x2u) != 0u;
    bool sticky = (mant_hi & 0x1u) != 0u;
    bool is_odd = ((mant_hi >> 2) & 0x1u) != 0u;
    bool round_up = round_bit && (sticky || is_odd);
    uint32_t bf_mant = (mant_hi >> 2) + (round_up ? 1u : 0u);
    if (bf_mant > 0x7Fu) {
        bf_mant = 0u;
        exp32 += 1u;
    }

    uint16_t bf_bits = static_cast<uint16_t>((sign16 << 15) | (exp32 << 7) | bf_mant);
    return *reinterpret_cast<const __nv_bfloat16*>(&bf_bits);
}

// wh, reinterpreted directly as a __half bit pattern. Only meaningful when
// control (bit0) is 0 -- a valid, properly-rounded half value; when control
// is 1, wh holds direction/offset bits instead of a real half exponent, so
// this reinterpretation is not meaningful there. Callers that need
// fp16-only compute across the whole array must use fp32_16_half_only
// (which checks control) rather than this unconditionally.
__host__ __device__ inline __half fp32_16_fast_half(uint16_t wh) {
    return *reinterpret_cast<const __half*>(&wh);
}

// A Dense/Sparse value array for NumericType::FP32_16 is allocated as
// num_elems uint32_t slots (matching every other NumericType's one-slot-per-
// element convention). Physically, that buffer holds two contiguous
// uint16_t arrays back to back: the first num_elems entries are the wh
// ("high") half, the next num_elems are the wr ("residual") half.
struct Fp32_16Storage {
    uint16_t* wh;
    uint16_t* wr;
};

inline Fp32_16Storage as_fp32_16_storage(uint32_t* base, size_t num_elems) {
    uint16_t* wh = reinterpret_cast<uint16_t*>(base);
    return Fp32_16Storage{wh, wh + num_elems};
}

}  // namespace tomos

#endif  // TOMOS_FP32_16_PACK_HPP

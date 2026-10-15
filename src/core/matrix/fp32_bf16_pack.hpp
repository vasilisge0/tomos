#ifndef TOMOS_FP32_BF16_PACK_HPP
#define TOMOS_FP32_BF16_PACK_HPP

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>

// Bit-level dissection of an IEEE-754 float32 into a pair of uint16_t words
// (wh, wr), analogous to fp32_16_pack.hpp's own wh/wr split but for
// bfloat16 instead of half -- and structurally much simpler, because
// bfloat16's 8-bit exponent field is bit-for-bit the SAME width and bias as
// float32's own (both biased by 127), unlike float16's narrower 5-bit
// field. A bfloat16 value IS, by construction, just the upper 16 bits of an
// IEEE float32 bit pattern (sign + 8-bit exponent + top 7 mantissa bits) --
// so there is no exponent range mismatch to handle, no rebiasing, no
// "extended"/"degraded" fallback encoding the way fp32_16_pack.hpp needs
// for fp16's narrower range:
//
//   wh: bits 31-16 of the float32 bit pattern (sign, exponent, mant_hi(7))
//       -- this IS a valid __nv_bfloat16 bit pattern already, unconditionally,
//       for every possible float32 value (normal, subnormal, zero, inf,
//       nan) -- reading wh's bits directly as bfloat16 needs no branching
//       at all, unlike fp32_16_half_only's three-way control-bit check.
//   wr: bits 15-0 of the float32 bit pattern (mant_lo(16), the remaining
//       16 of the 23 total mantissa bits truncated off of wh's own 7).
//
// wh is formed by TRUNCATION (bits>>16), not round-to-nearest: this is
// deliberate, not a missed accuracy opportunity -- fp32_bf16_combine(wh, wr)
// must reconstruct the ORIGINAL float32 bit-for-bit exactly for every
// input, with no special-casing, and simple concatenation ((wh<<16)|wr) can
// only be an exact inverse of a truncating split (a rounding split could
// carry into the exponent, which wr's lower-16-bits-only role has no way to
// represent). Same "split the raw bits, no rounding" philosophy fp32_16_
// split's own exact-mode branches already use, just unconditionally exact
// here rather than only within a limited exponent range.
//
// __host__/__device__ resolve to real CUDA qualifiers under nvcc and to
// nothing under a plain host compiler (via cuda_bf16.h, included directly
// above -- deliberately NOT via precision.hpp the way fp32_16_pack.hpp
// pulls in cuda_fp16.h, since that header's own transitive includes drag
// in this project's full dependency graph, which standalone nvcc builds
// like the wide_bigreg_*_vs_cutlass_ref.cu family can't easily satisfy;
// this header needs nothing else from precision.hpp), so this single
// header serves both the CPU and CUDA kernels without duplication.

namespace tomos {

namespace fp32_bf16_detail {

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

}  // namespace fp32_bf16_detail

// Splits value into its upper 16 bits (wh, a valid truncated bfloat16
// pattern) and lower 16 bits (wr, the remaining mantissa). Exact and
// unconditional for every float32 value -- no special cases.
__host__ __device__ inline void fp32_bf16_split(float value, uint16_t& wh,
                                                 uint16_t& wr) {
    uint32_t bits = fp32_bf16_detail::float_to_bits(value);
    wh = static_cast<uint16_t>(bits >> 16);
    wr = static_cast<uint16_t>(bits & 0xFFFFu);
}

// Reconstructs the exact original float32 bit pattern from wh and wr.
// Always exact -- no approximation, no range restriction, unlike fp32_16_
// combine's degraded branch.
__host__ __device__ inline float fp32_bf16_combine(uint16_t wh, uint16_t wr) {
    uint32_t bits = (static_cast<uint32_t>(wh) << 16) | static_cast<uint32_t>(wr);
    return fp32_bf16_detail::bits_to_float(bits);
}

// wh, reinterpreted directly as a __nv_bfloat16 bit pattern -- unlike
// fp32_16_half_only, this needs no control-bit check at all: wh's bits are
// ALWAYS a valid (truncated, not rounded) bfloat16 value for every possible
// float32 input, since bfloat16's exponent field never needs rebiasing or
// range-restricted fallback encoding the way fp16's does. Also valid,
// unchanged, on wh produced by fp32_bf16_split_rounded below -- a rounded
// wh is just as valid a standalone bfloat16 bit pattern as a truncated one,
// this function doesn't care which encoder produced it.
__host__ __device__ inline __nv_bfloat16 fp32_bf16_bf16_only(uint16_t wh) {
    return *reinterpret_cast<const __nv_bfloat16*>(&wh);
}

// Rounding variant of fp32_bf16_split: wh is formed by ROUNDING (to
// nearest, ties to even) rather than truncating -- mirroring fp32_16_
// split's own mant_hi rounding technique (fp32_16_pack.hpp) but for
// bfloat16's 7-bit (not fp16's 9-bit) retained mantissa. This is what
// makes wh, reinterpreted alone via fp32_bf16_bf16_only, a PROPERLY
// ROUNDED bfloat16 value instead of merely truncated -- the same
// motivation fp32_16_split has for rounding its own mant_hi.
//
// Unlike fp32_16_split, though, this is NOT a free upgrade: fp32_16's wr
// has 2 genuinely spare bits (mant_lo only needs 14 of wr's 16 bits) to
// carry round_flag on top of a fully-preserved mant_lo, so fp32_16_
// combine stays bit-exact. bf16's mant_lo is a full 16 bits wide (23-bit
// fp32 mantissa minus bf16's 7-bit mant_hi) -- wr has NO spare bit here.
// round_flag (recording whether rounding stepped wh up by 1, so fp32_
// bf16_combine_rounded can invert it) is stored in wr's own bit0,
// OVERWRITING mant_lo's own LSB. Full reconstruction is therefore no
// longer bit-exact the way plain fp32_bf16_split/fp32_bf16_combine is:
// the original mantissa's very last bit is unrecoverable and reconstructs
// as 0, costing at most 2^-23 relative error (one ULP of the full 23-bit
// fp32 mantissa) -- utterly negligible next to bf16's own ~2^-8 relative
// precision -- in exchange for a properly-rounded (not truncated)
// standalone bf16 value in wh.
__host__ __device__ inline void fp32_bf16_split_rounded(float value, uint16_t& wh,
                                                          uint16_t& wr) {
    uint32_t bits = fp32_bf16_detail::float_to_bits(value);
    uint16_t wh_truncated = static_cast<uint16_t>(bits >> 16);
    uint32_t mant_lo = bits & 0xFFFFu;
    uint32_t mant_hi_pre = static_cast<uint32_t>(wh_truncated) & 0x7Fu;

    // Round to nearest, ties to even: round_bit is mant_lo's own MSB
    // (bit15, the highest discarded bit); sticky is whether any bit below
    // THAT is set; an exact tie (round_bit set, sticky clear) rounds
    // toward whichever choice leaves mant_hi_pre even. Same technique as
    // fp32_16_round_up (fp32_16_pack.hpp), just against bf16's narrower
    // 7-bit retained mantissa.
    bool round_bit = ((mant_lo >> 15) & 0x1u) != 0u;
    bool sticky = (mant_lo & 0x7FFFu) != 0u;
    bool round_up = round_bit && (sticky || ((mant_hi_pre & 0x1u) != 0u));

    // Adding 1 to the truncated 16-bit high half (sign|exponent|mant_hi)
    // lets ordinary unsigned integer carry propagation ripple a mant_hi
    // overflow into the exponent field (and, in the astronomically rare
    // case of an already-max exponent with an all-ones mant_hi rounding
    // up -- e.g. a specific NaN payload -- further still, which round-
    // trips self-consistently via fp32_bf16_combine_rounded's own
    // subtract-back-out either way, even though wh alone would no longer
    // look like a sensible standalone bf16 value for that one input in
    // roughly 2^32).
    wh = static_cast<uint16_t>(wh_truncated + (round_up ? 1u : 0u));
    wr = static_cast<uint16_t>((mant_lo & 0xFFFEu) | (round_up ? 1u : 0u));
}

// Inverts fp32_bf16_split_rounded: recovers wh_truncated by subtracting
// round_flag (wr's bit0) back out -- the exact inverse of the add-based
// carry fp32_bf16_split_rounded used, correct even across the same rare
// wraparound edge case -- then reconstructs mant_lo from wr's remaining
// 15 bits, with the lost LSB (overwritten by round_flag) reconstructed as
// 0. NOT bit-exact (see fp32_bf16_split_rounded's own comment for the
// bound) -- only ever pair this with output from fp32_bf16_split_rounded,
// never with fp32_bf16_split's plain truncating output.
// TRIED AND REVERTED: two alternate bit-identical formulations -- (1)
// combine wh/wr into the packed 32-bit pattern FIRST, then apply a
// single subtraction with round_flag replicated into both the mantissa-
// LSB and wh-carry positions (delta = (round_flag<<16)|round_flag,
// bits = combined - delta), mirroring fp32_16_combine's own reverted
// "combined-subtract" experiment; and (2) the same combine-first idea
// but expressed as two separate subtracted terms instead of one pre-
// combined delta. Both compiled to the SAME register/spill footprint as
// the version below (REG:128 STACK:56, 23 STL + 11 LDL) when wired into
// mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_
// rounded_kernel -- ptxas evidently normalizes all three to equivalent
// code, the same "source-level reordering doesn't change the compiled
// result" pattern this session's midpoint_exact investigation already
// found for prefetch-checkpoint positioning. Relaxing __launch_bounds__
// from (256,2) to (256) alone (diagnostic only, NOT a fix) let ptxas use
// 138 registers with 0 spill -- confirming the spill is a genuine ~10-
// register shortfall against the 128-register ceiling 2-blocks/SM
// occupancy requires, not an artifact of any particular arithmetic
// sequence -- but that relaxation is the SAME occupancy-halving move
// midpoint_uncapped already measured as catastrophic (0.637x vs CUTLASS
// vs 0.884x), so it was reverted immediately rather than kept. The
// round_flag correction this decode needs (unlike the plain kernel's
// pure OR) appears to be an intrinsic register cost, not a fixable
// inefficiency -- see git log for the two reverted formulations if
// revisiting this.
__host__ __device__ inline float fp32_bf16_combine_rounded(uint16_t wh, uint16_t wr) {
    uint32_t round_flag = static_cast<uint32_t>(wr) & 0x1u;
    uint32_t wh_pretrunc = (static_cast<uint32_t>(wh) - round_flag) & 0xFFFFu;
    uint32_t mant_lo = static_cast<uint32_t>(wr) & 0xFFFEu;
    uint32_t bits = (wh_pretrunc << 16) | mant_lo;
    return fp32_bf16_detail::bits_to_float(bits);
}

// A Dense/Sparse value array for a packed FP32_BF16 representation is
// allocated as num_elems uint32_t slots (matching fp32_16's own one-slot-
// per-element convention). Physically, that buffer holds two contiguous
// uint16_t arrays back to back: the first num_elems entries are the wh
// ("high"/bfloat16) half, the next num_elems are the wr ("residual") half.
struct Fp32Bf16Storage {
    uint16_t* wh;
    uint16_t* wr;
};

inline Fp32Bf16Storage as_fp32_bf16_storage(uint32_t* base, size_t num_elems) {
    uint16_t* wh = reinterpret_cast<uint16_t*>(base);
    return Fp32Bf16Storage{wh, wh + num_elems};
}

}  // namespace tomos

#endif  // TOMOS_FP32_BF16_PACK_HPP

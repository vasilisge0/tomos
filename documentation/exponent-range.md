# Exponent range in the FP32_16 packed format

This note documents the exponent-range limitations of `NumericType::FP32_16`
(the packed `wh`/`wr` format defined in `src/core/matrix/fp32_16_pack.hpp`),
found while investigating why `--implicit-duplicates` (FP32_16 storage)
diverges under preconditioning on a real, extremely ill-conditioned matrix
(condition number ~4.4e9) where plain FP32 storage does not. It covers what
range the format's exponent-offset mechanism can represent with 3 offset
bits (the original implementation) vs 4 or 5, and how far short each falls
of covering both a real test case and FP32's own full range.

**Status: implemented.** The format now uses the 4-bit-per-direction scheme
described in "Implemented: 4 offset bits + explicit saturation" below,
replacing the original 3-bit-offset + real-`__half`-cast "degraded" fallback
described in the next two sections (kept for historical context — the
reasoning that motivated the change). The two genuinely idle bits this
investigation found are the reason the wider field was free: `wh[13]` is now
spent on the 4th offset bit; `wr[1]` remains idle (a 4-bit field alone
already had enough headroom for this matrix's needs once combined with
`fp32_16_half_only`'s new shift-based subnormal reconstruction — see below —
so the further 5-bit widening this file also explores was not needed).

## Format recap (original design — see "Implemented" section below for what shipped)

Each packed element is one `uint32_t`, split into two `uint16_t` halves:

```
wh: sign(1) | exponent(5) | mant_hi(9) | control(1)
wr: mant_lo(14) | round_flag(1) | reserved(1)
```

Three encoding tiers, selected by `control` (`wh` bit 0) and, when
`control=1`, by the `offset` sub-field:

- **Normal** (`control=0`): direct rebiased half-style exponent, valid for
  FP32 exponent field in `[113,142]` (`kFp32ExpMin`..`kFp32ExpMax` in
  `fp32_16_pack.hpp`). Bit-exact: the full 23-bit mantissa (`mant_hi`(9) +
  `mant_lo`(14), matching FP32's own mantissa width) is preserved.
- **Extended** (`control=1`, `offset` in `1..kOffsetMax`): the exponent bits
  hold a *direction* bit (`wh[14]`: 0=underflow, 1=overflow) and an *offset*
  — how many steps past the normal window's edge the value's true FP32
  exponent sits. Still bit-exact in the mantissa; only the exponent is
  encoded differently.
- **Degraded** (`control=1`, `offset=kOffsetDegraded=7`): the offset field
  can't reach far enough. Falls back to a literal `static_cast<__half>(value)`
  cast, splicing the resulting 16-bit half pattern across `wh`'s mantissa
  bits and `wr`'s bottom 5 bits. Reconstructs only to half's own precision
  and range — anything below half's own smallest subnormal (`~5.96e-8`)
  rounds to exactly zero here, same as real FP16 hardware underflow.

## Where "113" and "142" come from

`kFp32ExpMin=113` and `kFp32ExpMax=142` are **not** FP32's own
normal/subnormal boundary (that's field value `1`, actual exponent `-126`).
They're half-precision's own normal-range boundary (`-14` to `+15`,
biased by 15), re-expressed using FP32's bias (127) instead, so they can be
compared directly against an FP32 value's own (already 127-biased)
exponent field:

| | unbiased exponent | half's own biased field (bias 15) | FP32-biased field for the same value (bias 127) |
|---|---|---|---|
| half's smallest normal | `-14` | `1` | `113` |
| half's largest normal | `+15` | `30` | `142` |

`offset = kFp32ExpMin - exp32` (underflow) or `exp32 - kFp32ExpMax`
(overflow) is then a pure step-count — both sides share the FP32 bias, so
it cancels out.

## Original implementation: 3 offset bits (superseded)

`offset` is extracted/written everywhere in `fp32_16_pack.hpp` and
`fp32_16_cuda_kernels.cu` as `(wh >> 10) & 0x7` — 3 bits, values `0-7`.
One value (`7`, `kOffsetDegraded`) is reserved as the "fall back to
degraded" marker, leaving `1-6` usable as literal offsets
(`kOffsetMax = 6`).

| | FP32 exponent field | magnitude |
|---|---|---|
| Extended overflow ceiling (offset 6) | `148` | `2^21 ≈ 2.1e6` |
| Normal window | `[113,142]` | `[2^-14, 2^15] ≈ [6.1e-5, 3.3e4]` |
| Extended underflow floor (offset 6) | `107` | `2^-20 ≈ 9.5e-7` |
| Degraded fallback floor (real half cast) | — | `2^-24 ≈ 5.96e-8` (below this: exact `0`) |

So with 3 offset bits, values are bit-exact down to `~9.5e-7`; between
`~9.5e-7` and `~5.96e-8` they degrade to half-level precision; below
`~5.96e-8` they become exactly zero.

## Implemented: 4 offset bits + explicit saturation

The shipped design reorganizes the `control=1` branch entirely, removing the
real-`__half`-cast "degraded" tier rather than just widening its trigger
point:

- `wh[0]` (`control`, unchanged): `1` selects this branch.
- `wh[14]` (`direction`, unchanged position, but now purely a 2-way mode
  selector — never repurposed as a mantissa bit): `0`=underflow, `1`=overflow.
- `wh[13:10]` (4 bits — reclaims the idle `wh[13]`): `offset`, `1..14`
  (`kOffsetMax=14`), exact and bit-preserving in both directions (same plain
  step-count math as the original 3-bit scheme, just wider). `offset=15`
  (`kOffsetSaturate`, renamed from `kOffsetDegraded`) is a marker: the value
  is outside this format's reach entirely in that direction, and reconstructs
  to exactly signed `Inf` (overflow) or signed `0` (underflow) — mantissa
  bits are don't-care, written `0` by the encoder. There is no longer a
  reduced-precision middle tier: every value is either exact or a clean
  saturation.

| | FP32 exponent field | magnitude |
|---|---|---|
| Extended overflow ceiling (offset 14) | `156` | `2^29 ≈ 5.4e8` |
| Normal window | `[113,142]` | `[2^-14, 2^15] ≈ [6.1e-5, 3.3e4]` |
| Extended underflow floor (offset 14) | `99` | `2^-28 ≈ 3.7e-9` |
| Saturate marker (offset 15) | — | exactly signed `Inf` / signed `0` |

FP32 subnormals, exact zero, `Inf`, and `NaN` all naturally compute an offset
`>14` and land on the saturate marker with no special-casing needed. `NaN`
saturates to `Inf` (payload dropped) — accepted, since the marker has no
spare bits to flag "was actually NaN".

**`fp32_16_half_only`'s underflow branch also changed**, independent of the
above: rather than flushing every `control=1` value it can't otherwise
represent to `Inf`/`0`, it now reconstructs a genuine fp16 subnormal for
underflowed values by shifting `mant_hi`'s implicit-1-prefixed significand by
`offset-1` positions — exactly IEEE subnormal renormalization, just computed
from the stored offset directly instead of by searching for a leading bit.
Since `kFp32ExpMin` already re-biases FP16's own normal floor, `offset` *is*
the real FP16 subnormal shift count with no translation needed. This is
bit-exact against a genuine `static_cast<__half>(value)` for `offset` in
`1..10` (FP16's own 10 subnormal positions); for `offset` in `11..15` the
same shift formula naturally (no extra branch) produces `0`, matching real
FP16's own underflow-to-zero below its smallest subnormal (`2^-24`).
Overflow direction still always saturates to `Inf` regardless of the
specific offset stored, since even `offset=1` there is already beyond FP16's
own range. A cheaper sibling, `fp32_16_half_only_fast`, skips this shift
work entirely and always saturates control=1 values to `Inf`/`0` — an
explicit accuracy-for-speed trade for kernels that don't need that last
sliver of recoverable subnormal range.

**This narrows, but does not close, the real-matrix gap.** The "becomes
exactly zero" floor for the accurate/exact tier moves from `~5.96e-8` (old
degraded fallback's real-half subnormal floor) to `~3.7e-9` (new 4-bit exact
ceiling) — a real, ~16x deeper exact reach, so the `47,523`/`1e9`
zeroed-entry count from the "Real-world case study" below should shrink
(not yet re-measured against the shipped implementation). But this matrix's
own smallest entries (`~1.4e-12`, needing offset `~26`) are still well below
even the new `3.7e-9` floor, so they still saturate to exactly zero under
the shipped 4-bit scheme — closing this specific matrix's gap fully would
still require the 5-bit extension explored below (floor `~5.7e-14`), which
was not part of what was implemented.

## An idle bit in `wh`

`wh`'s 5-bit exponent field is split as 1 direction bit (`wh[14]`) + 3
offset bits (`wh[12:10]`) = 4 bits used. **`wh[13]` is never read or
written anywhere** in the encode or decode paths — confirmed by grepping
every `offset`/`direction` extraction and construction site in both
`fp32_16_pack.hpp` and `fp32_16_cuda_kernels.cu`. It's allocated to the
"exponent" field by the format's own layout comment but sits unused.

## An idle bit in `wr`

`wr = (mant_lo << 2) | round_flag` — `mant_lo` occupies bits `2-15` (14
bits), `round_flag` occupies bit `0`. **`wr[1]` is never touched** — this is
exactly the "reserved(1)" bit the layout comment already names.

## Why not a flat, unified exponent field instead?

`wh[0]` (`control`) plus the 5-bit exponent field is 6 bits total. A flat,
unified 6-bit field — one number, no separate mode-select bit, no
hierarchical branching between normal/extended/degraded — could represent
up to `63`-`64` distinct exponent values directly (reserving at most one
code as a fallback marker). The *current* scheme, spending those same 6
bits as `control(1) + field(5)`, only covers `42` distinct exponent values
exactly: `30` (normal) + `6` (extended overflow) + `6` (extended
underflow). So a flat encoding would reach further using the identical bit
budget.

This isn't simply wasted, though — the current layout is shaped the way it
is so the `pure_half` tier's decode (`fp32_16_half_only`) can skip
arithmetic entirely for the common case:

```cpp
if (!control) {
    return *reinterpret_cast<const __half*>(&wh);
}
```

When `control=0`, this does **no computation at all** — it reinterprets
`wh`'s raw bits directly as a valid IEEE half-precision value, because
"normal" mode's `exp16 = exp32 - kExpRebias` construction is deliberately
chosen to produce bits *bit-identical* to what a genuine `__half` would
already hold for that value. That free reinterpret only works because of
this specific (less bit-efficient) layout choice.

A flat, unified offset field would give up that property: *every* value,
including ones well within half's own native range, would need explicit
offset-to-exponent arithmetic on decode, not just the rare extreme ones —
a real cost on the tier that runs this decode on every element of every
matvec. So the `42`-vs-`63` gap is a deliberate range-vs-decode-cost
trade-off in the current design, not an oversight.

The two genuinely idle bits (`wh[13]`, `wr[1]`) are a separate matter,
though: they contribute to neither the bit-compatibility trick (which only
involves `control` and the 5-bit field as a whole, already fixed) nor to
reach in the current design. Reclaiming them doesn't touch the
normal-mode fast path at all — it only widens what `control=1` values can
reach, at no cost to the property above.

## What widening the offset field would buy

### 4 bits (reclaiming `wh[13]` only)

`kOffsetMax` → `14` (reserving one value, e.g. `15`, as the degraded
marker). Doubles the reach on both sides at zero cost to `mant_lo`:

| | FP32 exponent field | magnitude |
|---|---|---|
| Extended overflow ceiling (offset 14) | `156` | `2^29 ≈ 5.4e8` |
| Extended underflow floor (offset 14) | `99` | `2^-28 ≈ 3.7e-9` |

Still falls short of this matrix's own smallest values (`~1.4e-12`, needing
offset `~26`) and of half's own subnormal floor territory being fully
subsumed.

### 5 bits (`wh[13]` + `wr[1]`)

`kOffsetMax` → `30` (reserving `31` as the marker). Uses only the two bits
that are already idle in both halves — no reduction to `mant_lo`'s current
14 bits of real mantissa precision:

| | FP32 exponent field | magnitude |
|---|---|---|
| Extended overflow ceiling (offset 30) | `172` | `2^45 ≈ 3.5e13` |
| Extended underflow floor (offset 30) | `83` | `2^-44 ≈ 5.7e-14` |

This comfortably covers the real matrix's smallest values.

## Real-world case study

The matrix that motivated this investigation (a 1,000,000×1,000,
condition-number-≈4.4e9 problem generated via
[`scripts/generate_tall_thin_mtx.sh`](../scripts/generate_tall_thin_mtx.sh)
— note this condition number is higher than the `cond=1000`/`cond=500`
problems `scripts/generate_lsqr_test_matrices.sh` generates by default)
has entries ranging from
`6.27e-3` down to `1.4e-12` — needing an offset of roughly `26` to reach its
smallest values losslessly.

Direct measurement (`awk` scan over the raw `.mtx` data, and confirmed via
`test_fp32_16.cpp`'s `CombineBranchlessPackedBufferSmallMagnitudeUnderflow`
device-vs-host decode test) of what the *current* 3-bit-offset
implementation actually does to this matrix:

- **47,523 of 1,000,000,000 entries (0.0048%)** fall below the degraded
  fallback's `~5.96e-8` floor and round to **exactly zero**.
- **758,439 entries (0.0758%)** fall below the extended tier's own
  `~9.5e-7` floor (i.e. degrade to half-level precision, whether or not
  they end up exactly zero).
- Decode itself was verified bit-exact against a host reference throughout
  (`fp32_16_combine` and `fp32_16_combine_branchless`, both variants, at
  the real failing GEMV kernel shape) — the zeroing is the documented,
  correct behavior of the current 3-bit-offset degraded fallback, not a
  decoding bug.

With a 5-bit offset (using the two already-idle bits), this matrix's
`~26`-step requirement fits inside the *exact* extended tier entirely —
none of its entries would need the degraded fallback, and none would
round to zero.

## What it would take to cover *all* of FP32's range

The offset computation itself (`kFp32ExpMin - exp32` /
`exp32 - kFp32ExpMax`) is an unbounded integer subtraction — its maximum
possible value, across every finite FP32 exponent field (`0`-`254`), is:

- Underflow: `kFp32ExpMin - 0 = 113` (or `112` restricted to normal FP32
  inputs, `exp32 ≥ 1`).
- Overflow: `254 - kFp32ExpMax = 112`.

Storing any integer up to `113` needs `⌈log2(113)⌉ = 7` bits. That's more
than the 5 bits available from the two currently-idle bits alone — reaching
full FP32 coverage would require cutting into `mant_lo`'s real 14-bit
budget by ~2 bits (down to ~12), trading mantissa precision (for *every*
extended/degraded-eligible value, not just the extreme ones) for exponent
reach. This is a real design trade-off, not a bug fix, and is out of scope
for what this matrix specifically needs.

## Summary

| Offset width | Bits used | Max offset | Underflow floor (exact) | Covers this matrix (`~26`)? | Covers all of FP32 (`~113`)? | Cost |
|---|---|---|---|---|---|---|
| 3 (original) | `wh` only, partial | 6 | `9.5e-7` | No | No | — |
| **4 (implemented)** | `wh` only, full | 14 | `3.7e-9` | No | No | Reclaims 1 idle bit; also adds `fp32_16_half_only`'s shift-based fp16 subnormal recovery |
| 5 (not implemented) | `wh` + `wr` idle bits | 30 | `5.7e-14` | **Yes** | No | Reclaims 2 idle bits, zero mantissa cost |
| 7 (not implemented) | `wh` + `wr`, cutting into `mant_lo` | ~126 | far below FP32 subnormal range | Yes | Yes | Reclaims 2 idle bits + ~2 bits of real mantissa precision |

# FP32_16 pure_half decode: investigation and optimizations

This documents the investigation into why `--implicit-duplicates` (LSQR's
FP32_16-packed, FP16-compute storage tier) ran markedly slower per-iteration
than `--explicit-duplicates` (native FP16 storage) on the real LSQR matrix
(`tall_thin_1000.mtx`, M=1,000,000, K=1,000), and the sequence of decode- and
kernel-level changes that closed most of that gap. It's written as a record
of what was tried, what actually helped (measured, not assumed), and what
didn't — so the same ground doesn't get re-covered blind.

Every "wh" byte-layout fact here refers to `src/core/matrix/fp32_16_pack.hpp`'s
own header comment, which is the authoritative source for the encoding.

## Starting point: the mystery

Isolated GEVM benchmarks (`scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh`)
predicted only a ~12% gap between implicit's packed/FP16-compute kernels and
explicit's native-FP16 kernels at the real LSQR shape. The real LSQR solve
showed a stable, reproducible ~1.5-1.8x gap instead. Wrong-kernel dispatch,
GPU clock throttling, synchronization artifacts, and preconditioner
interaction were all checked directly and ruled out.

**Root cause**: the isolated benchmarks generated synthetic `wh` test data via
`pseudo_rand` — uniform on `[-1, 1]`, which essentially never underflows
fp16's normal range (~0% underflow rate). The real matrix, sampled directly
(`awk` over the raw `.mtx` values, two independent 2M-value samples), has a
genuine **~18% underflow rate** and **0% overflow rate**. The pure_half
decode functions (`fp32_16_half_only*`) are data-dependent: they take an
extra, expensive reconstruction path specifically for underflowed values.
Synthetic benchmarks with ~0% underflow never exercised that path; real
ill-conditioned data does, constantly. That gap between synthetic and real
data statistics — not a kernel bug — was the core of the mystery.

This was confirmed directly: adding a `gen_fp32_16_wh_realistic` data
generator (`~18%` underflow, `0%` overflow, matching the sampled real
distribution) to the isolated benchmark reproduced the real solve's slowdown
in isolation, closing the loop between "isolated benchmark" and "real LSQR".

## The encoding (context for everything below)

`wh` (16 bits) — `sign(1) | exponent-or-direction(1) | shift_by(4) |
mant_hi(9) | control(1)`:

- `control=0`: `wh` reinterpreted directly as a `__half` bit pattern is
  already correct (a properly-rounded fp16 value). Zero decode cost.
- `control=1, direction=1` (overflow): true magnitude is unconditionally
  beyond fp16's max — decodes to signed Infinity regardless of the other
  bits.
- `control=1, direction=0` (underflow): the value underflowed fp16's normal
  range. `mant_hi` (9 bits) plus an implicit leading 1 form a 10-bit
  significand; `shift_by` says how far to denormalize it to reconstruct a
  genuine fp16 subnormal (shift amounts ≥10 naturally flush to zero, no
  extra branch needed — fp16 only has 10 subnormal positions).

`shift_by` is **not** the "number of exponent steps past the boundary"
(`offset`, 1-14) originally used throughout this format — see "Global
change: shift-by numbering" below for why and how that got renumbered.

## What actually helped (in order applied, each independently measured)

All isolated numbers below are GFlop/s at the real LSQR shape (M=1,000,000,
K=1,000), NO_TRANSPOSE, measured via
`scripts/benchmarks_gemv_fp32_16_pure_half_tall_1000cols_vs_cublas_ref.sh`
against **realistic (~18% underflow) synthetic data**, not the old ~0%
data the rest of this codebase's GEMV benchmarks still default to.

| Stage | GFlop/s | vs. `native_mixed` (explicit) |
|---|---|---|
| `fp32_16_half_only_fast2` (single branch, wrong kernel selection at this shape) | ~705 | 62% |
| Correct kernel selection (`narrow_vec`-family, not `wide4x`'s own atomic path) | — | — |
| `fp32_16_half_only_branchless_no_overflow` | ~769 | 68% |
| + shift-by numbering (`_shiftenc`, later promoted globally) | ~880 | 77% |
| + `__half2`/`__hfma2`-packed accumulation (`_half2acc`) | ~1020 | **90%** |
| + shape-aware dispatch (fixes small-M regression) | — | (see below) |

### 1. Branch reduction: safe → fast2 → branchless

`fp32_16_half_only` (safe) has two sequential branches: `control`, then
`direction`. `fp32_16_half_only_fast2` collapses these into one combined
`is_underflow = control & !direction` test. `fp32_16_half_only_branchless`
goes further: computes **both** the cheap and underflow-reconstruction paths
unconditionally, blending via a bitmask (`0u - condition` → all-1s/all-0s)
instead of an `if`.

**Why branchless wins on real data despite "extra" unconditional work**: with
32 SIMT lanes per warp and any per-lane underflow probability meaningfully
away from 0% or 100% (here, ~18%), the probability that all 32 lanes agree
(`p^32 + (1-p)^32`) is astronomically small. Virtually every warp is already
mixed and already executes both code paths serially via divergence — a
branchy kernel pays nearly the same cost as branchless on this kind of data,
so branchless's removal of the branch/predication overhead itself is close
to a strict win, with no real downside.

### 2. No-overflow specialization

`fp32_16_half_only_branchless_no_overflow` requires the caller to guarantee
no element in the buffer genuinely overflows (`control=1, direction=1`).
Measured true for the real matrix (0% overflow, confirmed by direct
sampling). Under that guarantee, `is_underflow` reduces to just `control`
— drops the `direction` extraction and the `cheap_bits` select on `control`
entirely (control=1 always means underflow; control=0's own bits already
are `cheap_bits`).

### 3. Global change: shift-by numbering

**Not a decode-only tweak — a real change to the shared, production wire
format**, applied to *every* encoder and decoder in
`fp32_16_pack.hpp`/`fp32_16_cuda_kernels.cu` (17 call sites across 2 files,
including 5 independently hand-duplicated `hybrid`-tier decode helpers that
don't share a common function).

The original field stored `offset` (1-14, "steps past the boundary"), and
decode had to compute `sig10 >> (offset - 1)` plus a zero-guard
(`safe_offset = offset | (offset == 0)`) to avoid undefined behavior on a
raw-bit-pattern `offset=0`, which the real encoder never produces but the
decode function still has to be safe against. Storing `shift_by = offset -
1` directly (0-13, every value meaningful) removes both the subtraction and
the zero-guard from decode's critical path.

**Why this is lossless** (and why an earlier, rejected idea wasn't): this
only relabels the *shift-amount counter*, not the mantissa. `mant_hi` stays
completely untouched — raw and unshifted, exactly as the "accurate" (wh+wr)
tier needs it for exact fp32 reconstruction via `fp32_16_combine`. An
earlier idea to *pre-shift the mantissa itself* at encode time (storing the
final denormalized value directly) was considered and rejected: it would
permanently discard the low bits that fall off the shift, which the
accurate tier needs `wr`'s `mant_lo` to combine with for exact recovery.
Renumbering a 4-bit counter changes zero bits of actual information;
pre-shifting the mantissa destroys real information. That distinction is
the whole reason one was safe to promote and the other wasn't.

**Accurate-tier decoders** (which reconstruct the fp32 exponent directly,
not via a shift) needed a different edit: `kFp32ExpMax + offset` →
`kFp32ExpMax + 1 + shift_by`, `kFp32ExpMin - offset` → `kFp32ExpMin - 1 -
shift_by`, and the explicit `offset == kOffsetSaturate` check → `shift_by ==
kOffsetSaturateStored` (`kOffsetSaturateStored = kOffsetSaturate - 1`,
`fp32_16_pack.hpp`).

**Verification before any GPU rebuild**: a CPU-only host harness (no kernel
launch) checked all of:
- Direct roundtrip against the *original* fp32 input value (not just
  internal agreement between decoders) — 2,000,009 values, 0 failures.
- Every `pack.hpp` decoder variant cross-checked against every other — 0
  mismatches.
- All 5 hand-duplicated `.cu`-file implementations independently
  cross-checked against the verified `pack.hpp` reference — 0 mismatches.
- A full sweep of `src/` for any other hidden copy of this logic (`grep -rln
  ">> 10) & 0xF\|kOffsetSaturate"`) — confirmed only the 2 already-checked
  files touch this field anywhere in the codebase.

Measured impact was larger than instruction-count alone predicted (~14%
wall-clock, not the ~5-10% two fewer ops would suggest) — most likely
because it shortens a genuine serial dependency chain (`offset` →
`safe_offset` → shift amount → shift result), and chain *latency* matters
more than raw op count in a loop with limited ILP to hide it.

### 4. `__half2`/`__hfma2`-packed accumulation

The `narrow_vec` kernel family decoded 4 elements per thread (register
blocking) but accumulated them as 4 *separate scalar* `__half` FMAs. Other
kernel families in this codebase (`wide_ilp_shared_half2_atomic`) already
packed two elements into a `__half2` and used one `__hfma2` for two FMAs at
once — `narrow_vec` had just never adopted it. Packing `a0/a1` and `a2/a3`
into `__half2` pairs against a broadcast `x_val2 = __half2half2(x_val)`
halves the FMA instruction count for the accumulate step. This was the
single biggest individual win measured (~16% over the previous best),
because it's orthogonal to decode cost entirely — it fixes something decode
optimization could never touch.

Applying the *same* technique to `cpasync_warpuniform`'s own kernel (used
for the small-M path) turned out to be a no-op — that kernel already used
`__half2`/`__hfma2` accumulation from the start; only its decode was still
on the older `fp32_16_half_only` (safe, two-branch). Swapping just the
decode there (no accumulate change needed) gave a real, smaller win
consistent with the branch-reduction numbers above.

### 5. Redundant mask removal ("nomask")

`sig10 = 0x200 | mant_hi` is always in `[0x200, 0x3FF]` — exactly 10
significant bits, by construction (`mant_hi` is already masked to 9 bits,
and bit 9 is set unconditionally). Right-shifting a value already within 10
bits by any non-negative amount can only shrink it, never grow it past 10
bits. So `(sig10 >> shift_by) & 0x3FFu` and `sig10 >> shift_by` are
**identical** for all 8,192 possible `(mant_hi, shift_by)` combinations —
exhaustively verified, not just reasoned through. This isn't a
compiler-visible no-op either: `cuobjdump` on the already-optimized kernel
showed the mask compiled to a real, separate `LOP3.LUT` SASS instruction —
`ptxas` does not eliminate it on its own. Removing it from the C++ source
drops the per-element decode from ~11 non-FMA instructions to 10.

This is currently isolated to `fp32_16_half_only_branchless_no_overflow_nomask`,
exercised only via `test-new`'s dispatcher — not yet promoted to the base
`fp32_16_half_only_branchless_no_overflow` function pending a real-hardware
A/B measurement (the established pattern throughout: land a new, separate
function, measure it in isolation via `test-new`, then decide whether to
promote).

## Shape-aware dispatch (a non-decode fix, but necessary)

Every decode/accumulate optimization above was applied to `narrow_vec`
specifically — a kernel whose own "4 rows per thread" register-blocking
design only makes sense for large M (≥ `kMvWideRowThreshold` = 131,072,
`fp32_16_cuda_kernels.cu`). Calling it unconditionally at smaller M (e.g.
M=100,000) produces a catastrophically under-parallelized grid (977 blocks
shrinks to ~98 on a 108-SM A100 — most SMs idle). This was caught directly:
`test-new`'s first version called `narrow_vec` for every shape and collapsed
to ~25% of native at M=100,000, despite being ~90% of native at the real
M=1,000,000 shape.

The fix (`mv_fp32_16_pure_half_test_new_dispatch`,
`fp32_16_cuda_kernels.cu`) is the same shape-based composition the
production dispatcher (`mv_fp32_16_pure_half_no_transpose_fast2`) already
uses: route `rows_op < kMvWideRowThreshold` to `cpasync_warpuniform`'s own
kernel (a column-split/atomic-combine design that stays well-parallelized at
smaller M), and only the large-M branch to the optimized `narrow_vec`
variant. `cpasync_warpuniform` picked up the shift-by numbering and
no-overflow decode automatically (global format change + a mechanically
duplicated decode-only sibling), closing most — not all — of the gap at
smaller M too (61.8% → 74% of native at M=100,000).

## What did NOT help (real, measured negative results)

Both of these were flagged by `ncu`'s own "Est. Speedup" heuristics as
promising — neither delivered any real wall-clock improvement. Recorded
here specifically so they aren't re-tried blind:

- **Vectorized write-back (`vec4store`)**: `ncu` flagged the kernel's 4
  separate scalar `y[i]=` stores (and reads, when `beta≠0`) as
  uncoalesced, "Est. Speedup: 6-22.7%". Replacing them with a single
  `float4` load/store (safe: `i0` is always a multiple of 4 by
  construction, `y` comes from `cudaMalloc`) measured **statistically
  identical** to the non-vectorized version at the real shape.
- **Grid-stride launch (`gridstride`)**: `ncu` flagged a partial-wave tail
  (977 blocks, only 864 fit in one wave on this A100 — "Est. Speedup: 50%").
  Launching exactly one full wave's worth of blocks and looping internally
  (grid size sized at runtime via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, not hardcoded) also
  measured **no real improvement** — actually a hair slower, likely from
  the loop's own bookkeeping overhead.

Both stalls were real (confirmed by `ncu`), but neither was on the
*critical path* — they were apparently already hidden by other warps' work
under the kernel's real bottleneck (ALU throughput). The lesson: `ncu`'s
heuristic "Est. Speedup" estimates each flagged issue in isolation and can
overestimate real impact when something else is already the binding
constraint. Only changes that reduced the *dominant* bottleneck (decode
instruction count, then accumulate instruction count) produced measured
wall-clock gains.

## Where this leaves the format

`ncu` on the current best kernel at the real shape reports ALU as
"over-utilized and likely a performance bottleneck" (~82% Compute (SM)
Throughput) — a materially stronger signal than the earlier "well-utilized,
might become a bottleneck" reading, and `ncu`'s own guidance at this point
is explicit: further improvement needs work shifted off the currently
dominant unit, not another pass over the same one.

Inspecting the actual compiled SASS (not just C++ source) for the decode
confirms this isn't leaving free wins on the table: `ptxas` has already
fused essentially everything fuseable — bitfield extraction via `SHF` +
`LOP3` pairs, and the entire final branchless select (4 C++-level
operations: AND, NOT, AND, OR) collapses into a single `LOP3` instruction.
Reordering the bit layout to make one field cheaper to extract was checked
concretely and found to be a zero-sum trade (there is exactly one "free"
bit-0-aligned slot in the 16-bit word; whichever field doesn't occupy it
pays for a shift, so moving `control` off bit 0 to make `mant_hi` cheaper
just makes `control` correspondingly more expensive).

A genuinely cheaper decode at this point would need to eliminate an actual
*logical* field or step, not just relocate or fuse existing ones — i.e. a
real encoding redesign, not a kernel change. That's out of scope for this
document; see the git history around "Further improving decode" for where
that discussion left off.

## Current state (as of this writing)

- **Promoted globally** (production, `--implicit-duplicates`-live): the
  shift-by numbering (`fp32_16_split` and every consuming decoder), AND
  (as of the edit described below) the no-overflow, redundant-mask-removed
  branchless decode (`fp32_16_half_only_branchless_no_overflow_nomask`) in
  every GEVM (`mv_fp32_16_*`) kernel in `fp32_16_cuda_kernels.cu` that
  previously called `fp32_16_half_only`, `fp32_16_half_only_fast2`, or
  `fp32_16_half_only_branchless` — 65 distinct kernels, 252 call sites.
  `fp32_16_half_only_fast` (the separate, lossy saturate-to-zero decode)
  was explicitly excluded, as was the accurate/hybrid `wh`+`wr`
  `fp32_16_combine*` tier (a functionally different decode that still
  needs `wr`, not what "test-new's decode" refers to). The three
  `pack.hpp` source functions (`fp32_16_half_only`, `_fast2`,
  `_branchless`) were left in place, now explicitly commented as
  overflow/underflow-distinguishing, for any future caller that still
  needs that guarantee.
  **Correctness note**: this trades the overflow check away across a much
  larger surface than `test-new`'s original two kernels. It has only been
  verified against `tall_thin_1000.mtx`'s measured 0% overflow rate — any
  caller of these 65 kernels on data with a genuine overflow rate would
  now silently misdecode overflowed elements as underflow instead of
  saturating to Infinity.
  **Also included**: `mm_cuda_kernels.cu` (the separate GEMM, not GEMV,
  kernel family) — 31 call sites of `fp32_16_half_only` swapped the same
  way (this file didn't use `_fast2`/`_branchless` at all, and its own 3
  `fp32_16_half_only_fast` call sites, same lossy variant, were left
  alone). This is kernel territory never exercised by this session's
  GEVM-focused investigation — the same correctness caveat above applies,
  now to GEMM callers too, and the overflow-rate assumption has not been
  separately checked against any GEMM-specific dataset.
- **`test-new`-only, not yet wired into any dispatch registry**:
  `__half2`-packed accumulation and shape-aware dispatch (the decode swap
  above is now shared with `test-new` by construction, since `test-new`'s
  own kernels are part of the 65 updated). Production's actual dispatcher
  (`mv_pure_half_impl` in `fp32_16_registries.cpp`) still calls the older,
  non-shape-aware, non-half2acc kernel names — the decode change reaches
  production the moment those kernel *bodies* run, but production hasn't
  been repointed to `test-new`'s specific kernel *selection* (half2acc +
  shape dispatch) yet.
- Net measured result at the real LSQR shape: **~90% of `--explicit-duplicates`'
  own native-FP16 throughput**, up from the ~55-65% ratio that motivated
  this whole investigation.

## `branchless` vs `conditional` namespaces (kept for A/B comparison)

After the decode swap above was committed (`8fbbec9`, "Branchless decode."),
both `fp32_16_cuda_kernels.cu` and `mm_cuda_kernels.cu` were extended to keep
BOTH decode strategies simultaneously callable, so they can be measured
side by side (`ncu`, direct benchmark) instead of only being comparable via
git history:

- `tomos::cuda::experimental::branchless` — a namespace ALIAS (`namespace
  branchless = tomos::cuda::experimental;`), not a copy. Refers to the exact
  same entities as the unqualified `experimental::`/`cuda::` names used
  throughout both files today (the no-overflow, nomask, branchless decode).
  Exists purely to give that set a name symmetrical with `conditional`
  below, at zero cost and zero risk of the two diverging by accident.
- `tomos::cuda::experimental::conditional` / `tomos::cuda::conditional` —
  genuine duplicates: the exact pre-`8fbbec9` bodies (still using
  `fp32_16_half_only`/`_fast2`/`_branchless`, all overflow/underflow-
  distinguishing) of every function pulled in by the full call-graph
  closure over the swapped call sites — not just the 96 kernels whose
  decode call changed, but also every host-wrapper, `_best`-style
  delegator, and multi-hop caller needed to make those kernels
  independently reachable by name. Sizes: **145 functions (`fp32_16_cuda_
  kernels.cu`) + 25 (`mm_cuda_kernels.cu`) = 170 total**, generated
  mechanically from `git show 8fbbec9^:<path>` (the commit immediately
  before the decode swap) via a brace-matched function extractor + closure
  algorithm (not the earlier session's naive line-based approximation),
  verified by a from-scratch clean compile of both files (0 errors) and a
  full `tomos_cuda` library rebuild+link.
  Plain-scope wrapper duplicates (14 + 2) had their internal
  `experimental::X(...)` calls mechanically rewritten to
  `experimental::conditional::X(...)` so the conditional set is fully
  self-contained (never silently falls through to the branchless set);
  everything else is a byte-for-byte copy.
  **Not exposed anywhere else**: no dispatch registry, benchmark, or LSQR
  runner references `conditional::` yet — reaching these functions
  currently requires the fully-qualified name, e.g.
  `tomos::cuda::conditional::mv_fp32_16_pure_half_no_transpose_fast2(...)`.
  **Maintenance note**: `conditional` is a frozen snapshot, not a second
  copy to keep in sync — if the `branchless` set's non-decode logic
  changes going forward, `conditional` will silently go stale. It exists
  for a decode-only A/B comparison, nothing more.

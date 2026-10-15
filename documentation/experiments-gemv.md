# FP32 GEMV on FP32_16-Packed Data: Closing the Gap to Native FP32

**Kernels under study:** `mv_fp32_16_accurate_narrow_vec_kernel`, `mv_fp32_16_ready4_atomiconly_scaled_kblock_kernel`, `mv_fp32_16_ready8_atomiconly_scaled_kblock_kernel`, and their cp.async/no-overflow-decode descendants
**Files:** `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu` (kernels), `benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu` (benchmark), `scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh` / `scripts/plot_gemv_fp32_recon_vs_cutlass_ref.m` (driver/plot)
**Hardware:** NVIDIA A100 (sm_80), Purdue Gilbreth cluster
**Scope:** the **FP32-reconstruction** compute tier only (`mv_fp32_16_accurate` and its descendants: A stored packed as `wh`+`wr`, decoded back to genuine `float`, compute in `float`/`double`) — not the separate genuine-fp16-compute (`pure_half`) tier, which has its own, earlier optimization history (`cpasync_warpuniform`, documented separately in this session's LSQR work).

**Local setup:** the benchmark above compares against CUTLASS
(`scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh`), which defaults
to fetching/using it at `build/cutlass-ref`. For a persistent custom
location instead of passing `CUTLASS_DIR=` on every invocation, copy
`.configure.sh.example` (repo root) to `.configure.sh` and set
`CUTLASS_DIR` there — it's gitignored and sourced automatically. (The
same mechanism also covers `MATRIX_DIR` for LSQR's test matrices,
unrelated to this GEMV experiment.)

---

## 1. Background

`mv_fp32_16_accurate` (and its whole family) reads a matrix stored packed as two parallel `uint16_t` arrays, `wh` ("high" half, a self-contained fp16-like approximation with its own overflow/underflow escape encoding) and `wr` ("residual", the extra bits needed to reconstruct full fp32 precision when combined with `wh`), and reconstructs genuine `float` values via `fp32_16_combine_branchless(wh, wr)` before computing. This tier exists because LSQR's `--implicit-duplicates` mode stores A *only* in this packed format — there is no separate native-fp32 copy — so the solver's restart-boundary/residual-recompute step (which needs FP32 accuracy) has nowhere else to read from.

**The problem, quantified:** at the shapes that matter for real use, this tier ran meaningfully below native `mv_fp32_tiled`'s own throughput:

| Shape | native (`mv_fp32_tiled`) | packed FP32-reconstruction (plain, pre-investigation) | ratio |
|---|---|---|---|
| M=K=4096 (square) | 490 GFlop/s | 246–253 GFlop/s (`ready4`) | ~50% |
| M=K=65536 (square) | 700 GFlop/s | 608 GFlop/s (`ready4`) | ~87% |
| M=1,048,576, K=512 (tall/narrow) | 694 GFlop/s | 496 GFlop/s (`narrow_vec`, reached via `ready4`'s own fallback) | ~71% |

Two *different* kernels are responsible for this, depending on shape, both selected by the same top-level dispatcher (`mv_fp32_16_accurate`'s own `rows_op < kMvWideRowThreshold` check, threshold = 131072):

- **`mv_fp32_16_ready4_atomiconly_scaled_kblock_kernel`** — used for every shape in this benchmark's own **square** sweep (M=K never exceeds 65536 there) and for tall/narrow shapes with M below the threshold. K-tiled, 4-rows/lane register-blocked, shared-memory `x` staging, `atomicAdd`-combined across column-splits. This is the kernel actually plotted as plain `"TOMOS FP32 / FP16"`.
- **`mv_fp32_16_accurate_narrow_vec_kernel`** — used only once M reaches the threshold (i.e. the real LSQR matrix's own shape, M=1,000,000/K=1000, and every tall/narrow shape at the top of this benchmark's own M sweep). 4-rows/lane register-blocked, no shared-memory tiling, no atomics — a completely different design from `ready4`.

Critically: **every prior "test" experiment in this codebase's own history only ever touched `ready4`'s own smaller-row siblings (`ready`/`ready8`), never `narrow_vec`** — meaning the real LSQR matrix's own shape had never actually been targeted by any tuning pass before this investigation.

---

## 2. Investigation Timeline

### Phase 1 — Diagnosing `narrow_vec` (tall/narrow shapes)

Starting hypothesis: since `narrow_vec` decodes every element via `fp32_16_combine_branchless` (a genuinely expensive, branch-free-but-ALU-heavy sequence — computes the Normal, Extended, *and* Saturate bit-patterns unconditionally for every element, then mask-selects the right one, to avoid a data-dependent branch), the gap might be **memory-fetch latency**: `wh`/`wr` are read via direct, synchronous global loads every column iteration, with no staging.

- **`"test"`** — `mv_fp32_16_accurate_narrow_vec_cpasync`: added a 2-buffer (1-ahead), **per-thread** (not cross-thread-shared, unlike the GEMM cp.async kernels elsewhere in this file) double-buffered `cp.async` prefetch of the next column's `wh`/`wr` while decoding the current one. Measured at M=1,048,576: **532–534 GFlop/s**, a real but modest ~7% gain over the reused `ready4`-family baseline reachable at this shape.
- **`"test2"` (first form, superseded)** — `mv_fp32_16_accurate_narrow_vec_cpasync3`: deepened the pipeline to 3 buffers (2-ahead), using `__pipeline_wait_prior`'s runtime `size_t` overload (a plain `switch` over N=0–8 dispatching to the templated primitive, confirmed via `cuda_pipeline_primitives.h` — so a *runtime* wait-count is valid, not just a compile-time one) with an exactly-computed `allow = issued − j − 1` per iteration (FIFO completion order makes this provably correct, not a heuristic). **Measured IDENTICAL to `"test"`** — deepening the pipeline bought nothing.
  - **Conclusion:** `cp.async` had already fully hidden whatever fetch latency existed at 1-ahead. The remaining gap is **decode ALU/issue-throughput cost**, not memory latency — confirmed by a second, independent negative result (more buffering didn't help), not assumed.
- **`"test2"` (final form, promoted into this slot)** — `mv_fp32_16_accurate_narrow_vec_cpasync_no_overflow`: same `"test"` structure, but decoding via a new function, `fp32_16_combine_branchless_no_overflow` — see §3. Measured **596–600 GFlop/s** at M=1,048,576: a genuine **~12% win over `"test"`**, pushing the ratio to native from ~77% up to **~86%**.

### Phase 2 — Applying the same levers to `ready4` (square shapes)

Since `ready4` is what the square-shape benchmark (and every square/mid-size shape generally) actually measures as `"TOMOS FP32 / FP16"`, and it had never received either lever:

- **`"test3"`** — `mv_fp32_16_ready4_atomiconly_scaled_kblock_cpasync_no_overflow`: both levers (cp.async 1-ahead + no-overflow decode) applied to `ready4`'s own kernel. **Result: a real, consistent 7–34% win over plain `ready4` across the *entire* square sweep**, pushing the ceiling from ~87% of native up to **~93–94%** at large M:

  | M=K | native | `ready4` (plain) | `test3` | `test3`/`ready4` |
  |---|---|---|---|---|
  | 4096 | 491.9 | 246.2 | 325.7 | **1.32x** |
  | 8192 | 634.3 | 486.2 | 551.8 | **1.14x** |
  | 16384 | 683.8 | 579.0 | 630.5 | **1.09x** |
  | 32768 | 698.7 | 595.1 | 645.8 | **1.09x** |
  | 65536 | 700.5 | 608.1 | 650.8 | **1.07x** |

### Phase 3 — Revisiting 8-rows/lane

This codebase already had an **8-rows/lane** sibling of `ready4` (`mv_fp32_16_ready8_atomiconly_scaled_kblock_kernel`), from an earlier investigation, explicitly documented as **not panning out** in isolation (occupancy loss from halved `gridRows` outweighing the extra amortization). That finding predates both levers above — worth re-testing now that decode is cheaper and prefetched, since going wider changes the *ILP* available to hide whatever latency remains, which is a different mechanism than either lever tried alone.

- **`"test4"`** — `mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow`: both levers **and** 8-rows/lane combined.
- **`"test5"`** — `mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync`: identical to `"test4"` but decoding via the **general** (safe-on-arbitrary-data) `fp32_16_combine_branchless` instead, to isolate how much of `"test4"`'s win is the no-overflow assumption specifically vs. cp.async+8-rows on their own.

Full measured square-sweep comparison (same run, all five candidates + baselines):

| M=K | native | `ready4` | `test3` (4-row) | `test4` (8-row, no-ovf) | `test5` (8-row, safe) |
|---|---|---|---|---|---|
| 1024 | 68.3 | 27.0 | 30.4 | 26.7 | 24.8 |
| 2048 | 341.3 | 131.8 | 154.4 | 88.6 | 78.9 |
| 4096 | 487.0 | 249.3 | 325.8 | 337.7 | 307.4 |
| 8192 | 634.6 | 486.0 | 552.3 | 502.3 | 440.8 |
| 12288 | 660.9 | 545.3 | 600.5 | 593.7 | 521.8 |
| 16384 | 682.4 | 579.0 | 627.5 | 639.9 | 564.3 |
| 32768 | 697.3 | 600.5 | 645.8 | **692.8** | 608.2 |
| 65536 | 700.1 | 608.3 | 651.0 | **701.7** | 613.2 |

Two clear, important findings from this table:

1. **8-rows/lane is not uniformly better than 4-rows/lane.** `test4` trails `test3` below roughly M=12288–16384 (occupancy loss at small grids, same mechanism the original ready8 investigation found) but pulls decisively ahead above it — reaching **essential parity with native FP32** at M≥32768 (692.8/697.3 = 99.4%, 701.7/700.1 = **100.2%**).
2. **The no-overflow assumption is doing real, substantial work at 8-rows/lane** — `test4` vs `test5` is consistently ~13–15% apart at every size (e.g. 701.7 vs 613.2 at M=65536), confirming this isn't a negligible shortcut being taken for free.

---

## 3. `fp32_16_combine_branchless_no_overflow`: the Decode-Side Lever

`fp32_16_combine_branchless` unconditionally computes three candidate bit-patterns per element — Normal (`control==0`), Extended (`control==1`, `direction==0`: underflow escape), and Saturate (`control==1`, `offset==15`: complete over/underflow) — then mask-selects the right one, specifically to avoid a data-dependent branch (which would cause real warp divergence on random data).

`fp32_16_combine_branchless_no_overflow` (new, `fp32_16_cuda_kernels.cu`) drops the **overflow** half of that (the `direction==1` arm, saturate-to-Inf), under the explicit assumption — supplied by whoever calls it, not verified by the function itself — that the caller's real fp32 data is always representable in fp32 by construction, and the *only* reason any element would ever need `wh`'s escape encoding at all is **underflow** (small-magnitude values falling below fp16's normal exponent range), never overflow. Under that assumption, `direction` is never read, `expExtended`'s and `bitsSaturate`'s own direction-ternaries both collapse to their `direction==0` arm unconditionally, and the carry-correction logic (needed regardless of over/underflow, for `fp32_16_split`'s own mantissa-rounding) is untouched.

**This is silently, uncatchably wrong for any element that genuinely does overflow** — there is no saturate-to-Inf path left to catch it. It must only be used where the caller can actually guarantee this (e.g. a known-bounded matrix), not as a general-purpose replacement for the safe decode.

---

## 4. Promotion Decision

`"test4"` (`mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow`) was promoted to a named plot series, **`"TOMOS FP32 / FP16 (fast)"`** — explicitly not plain `"TOMOS FP32 / FP16"` — in `plot_gemv_fp32_recon_vs_cutlass_ref.m`, using the same `"(fast)"` naming convention already established by `"TOMOS FP16 (implicit, fast)"` on the sibling `pure_half`-tier plots, to signal the unsafe-on-arbitrary-data contract at a glance. The four other candidates (`"test"`, `"test2"`, `"test3"`, `"test5"`) remain defined and still measured by every run of the benchmark (their CSV columns are always populated), but are commented out of the active plot — not deleted — matching this codebase's own "reusable, throwaway `test` slot" convention throughout `benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh`'s own `METHOD_MARKERS` history.

`mv_accurate_impl` (`src/dispatch/fp32_16_registries.cpp`) — the actual LSQR-facing dispatch this whole tier feeds — **has not been changed** to call any of these new kernels yet. That remains a separate, explicit decision: promoting on the plot documents and surfaces the result; wiring it into the production dispatch path means accepting the no-overflow contract for every real matrix LSQR ever solves under `--implicit-duplicates`, which is a correctness-sensitive call the user needs to make deliberately, not a byproduct of a benchmark plot relabel.

---

## 5. Current State / Open Questions

- **Closed, with evidence:** `cp.async` prefetch depth beyond 1-ahead does not help (`"test"` vs. the original `"test2"`/`cpasync3` — identical). The bottleneck past 1-ahead is decode ALU/issue throughput, not memory latency.
- **Closed, with evidence:** the no-overflow decode shortcut is a real, substantial, non-negligible contributor (not just noise) — confirmed twice, independently, at both 4-rows/lane (narrow_vec `"test"`→`"test2"`, ~12%) and 8-rows/lane (`"test5"`→`"test4"`, ~13–15%).
- **Open:** `"test4"`'s own crossover point (~M=12288–16384, where 8-rows/lane stops losing to 4-rows/lane and starts winning) has not been used to build a size-adaptive dispatcher — the promoted series is `"test4"` alone, unconditionally, not a `test3`/`test4` hybrid. Whether such a hybrid is worth the added dispatch complexity is unexplored.
- **Open:** `narrow_vec`'s own no-overflow-decode variant (`"test2"`, final form) was never combined with `narrow_vec`'s own possible register-blocking width increase (an 8-rows/lane `narrow_vec` sibling doesn't exist) — the tall/narrow-shape investigation stopped at the 4-row/cp.async/no-overflow combination, one lever short of where the square-shape investigation ended up.
- **Open:** none of this touches the TRANSPOSE direction for this tier (`mv_fp32_16_transpose_atomiconly_scaled_mblock_kernel`), the `mv_fp32_bf16_accurate` sibling family, or `mv_accurate_impl`'s own production dispatch wiring.
- **Open:** `"test4"`'s real production risk (does the actual LSQR matrix data ever trigger `wh`'s overflow escape path?) has not been independently verified against the real matrix — the no-overflow assumption was supplied by the user as a property of their own data, not derived or checked by this investigation.

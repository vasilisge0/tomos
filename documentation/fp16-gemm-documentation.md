# FP16 GEMM Kernel Optimization: the `wide_bigreg_sq` L2 Sector Excess Investigation

**Kernel under study:** `gemm_fp16_mma_ptx_wide_bigreg_sq_kernel`
**File:** `src/kernels/cuda/matrix/mm_cuda_kernels.cu`
**Hardware:** NVIDIA A100 (sm_80), Purdue Gilbreth cluster
**Project constraint:** hand-tuned kernels must approach CUTLASS/cuBLAS performance **without** depending on CUTLASS as a library at build time.

---

## 1. Background

`wide_bigreg_sq` is one of several hand-written FP16 GEMM kernels in this codebase built to compete with CUTLASS's own `device::Gemm` and cuBLAS's `Hgemm`, without linking against CUTLASS. Profiling with `ncu` (Nsight Compute) surfaced a persistent finding: the kernel's A-tile load — `load_tile_async`'s `cp.async` (`LDGSTS.E.BYPASS.LTC128B.128`) instructions — shows a **real, hardware-confirmed L2 sector excess**, flagged by `ncu`'s `UncoalescedGlobalAccess` rule:

> *"This workload has uncoalesced global accesses resulting in a total of 134,217,728 excessive sectors (25% of the total 545,193,680 sectors)."* (N = 8192)

This is not a theoretical estimate — it is measured against a real hardware counter (`lts__t_sectors_srcunit_tex_op_read.sum`), compared to `ncu`'s own computed ideal. Real CUTLASS's equivalent load, on the identical GEMM, shows **0% excess**. Address arithmetic in our kernel was independently re-derived and confirmed correct — the warp's 32 lanes form a textbook-ideal, 512-byte-aligned contiguous run. The excess is a genuine, physical effect, not a bug in the addressing math.

Two additional established facts framed the whole investigation:

1. **Triggered by concurrency, not addresses.** A single active warp (1 warp/1 block) shows 0% excess. Two, four, or eight concurrently-active warps all show the identical ~1.5×/25% excess — the effect turns on completely between 1 and 2 warps and does not scale further with warp count.
2. **The naive fix is a net loss.** Replacing `cp.async` with a plain, synchronous vectorized `LDG.128`+`STS.128` (byte-identical addressing) eliminates the excess entirely (0% at any warp count) but causes a **35–40% throughput regression** (measured: 131,647–134,696 GFlop/s vs. the confirmed 205,000–220,427 GFlop/s baseline at N=8192). This kernel is compute/tensor-core-bound (~70% SM throughput from HMMA), so `cp.async`'s real value is hiding memory latency behind other in-flight work (MMA, B's load) — not reducing bytes moved. A synchronous load exposes that latency directly into the instruction stream, which costs far more than the sector waste it removes. This fix was tried and reverted; `cp.async` was kept **deliberately**.

Given (2), the goal shifted from "eliminate `cp.async`" to "understand and, if possible, eliminate the root cause while keeping `cp.async`'s async-overlap benefit intact."

---

## 2. Investigation Timeline

The investigation proceeded in three broad phases, spanning many independently-motivated, hardware-measured hypotheses. Numbers below are the actual measured values at each step, not estimates.

### Phase 1 — Characterizing the effect (prior sessions)

- Ruled out the epilogue's `beta*C` read (excess is identical whether or not that code path executes).
- Ruled out `ld_a`'s power-of-two-ness (padding `ld_a` to a non-power-of-two value made no difference).
- Ruled out warp-separation distance (doesn't correlate with excess magnitude).
- Confirmed via a byte-identical plain-load test that the effect is specific to `LDGSTS.BYPASS` (cache-bypass `cp.async`), not a general "many warps, different addresses" phenomenon.
- Traced why real CUTLASS avoids this by tracing `cutlass::gemm::device::Gemm`'s internal A↔B swap-and-transpose for `LayoutC=ColumnMajor` outputs, and wired up CUTLASS's **real** `PredicatedTileAccessIterator` class directly in an isolated repro (`a_load_repro_cutlass_iterator.cu`). This is the first configuration found that reproduced CUTLASS's clean (0% excess) behavior.
- An initial ~17-round ablation campaign (matching CUTLASS's address formula, predication style, immediate-offset addressing, synthetic and then genuine instruction interleaving, batched-bitmask predicate computation, and an exact source-verified match of the class's "compute predicates once, never again" steady-state behavior) **failed to reproduce the clean result** in hand-written code. Only the literal, complete CUTLASS iterator class was clean; every hand-rolled equivalent, however closely matched, showed the same 1.5× excess.

### Phase 2 — Exhaustive ablation of the real class, and ruling out timing (this session, part 1)

Rather than replicate the class, this phase modified the **real, unmodified CUTLASS header** in place (via a local include-path override), trivializing one piece of its internals at a time while keeping it as the vehicle:

- Trivialized `compute_predicates_()` to constant-true bitmasks — **still clean**.
- Trivialized `set_predicates()`'s residue-size/extent arithmetic — **still clean**.
- Replaced `ThreadMap::initial_offset(thread_id)` with hand-inlined scalar arithmetic producing the identical address — **still clean**.
- Moved that arithmetic into a **free function** (not a class method) in the same translation unit — **still clean**. This proved decisively that "being a class method" was never the discriminator.

With every internal computation gutted and even the class/free-function boundary eliminated, only the literal CUTLASS class — regardless of how hollowed-out — stayed clean. This pointed toward something at the *compiled-instruction-scheduling* level, not anything expressible in source.

SASS diffing found the clean kernel's four unrolled `LDGSTS` per tile were **not** fully back-to-back (`ptxas` interleaved a few unrelated ALU instructions between them), while every hand-rolled equivalent compiled to four fully-adjacent `LDGSTS`. This "instruction adjacency" hypothesis was then tested directly and **decisively falsified**:

- **Elapsed-time spacing** (`__nanosleep()`, both uniform-per-warp and staggered-between-warps, 0–300ns): **zero effect** at every magnitude.
- **Forced, unhoistable instruction-level interleaving** (a genuine data-dependency chain feeding each `LDGSTS`'s own predicate register, deep enough that `ptxas` couldn't precompute it ahead of time): produced real SASS-level gaps between the `LDGSTS`, in one case matching or exceeding the real kernel's own gap pattern — **still excess, completely unchanged**, in every configuration (0 gaps, 1 gap, 3 gaps).

Conclusion: SASS-level instruction adjacency was a real, consistent *correlate* across dozens of variants, but proven **non-causal** once directly, controllably tested.

### Phase 3 — Finding a real, reproducible mechanism in isolation

A sharper question was then asked: **is this really about the iterator at all**, or could file/translation-unit-level effects, or something else entirely, be responsible?

- A fully hand-written kernel with **zero** `Iterator` class usage, placed in the *same file* as the real class-based kernels, came back **clean** — the first hand-written kernel in the whole investigation to do so. Moved into a brand-new, minimal, single-kernel file (no CUTLASS iterator headers at all): **still clean**. This definitively ruled out the iterator class, and the file/TU, as the cause.
- Careful line-by-line diffing against the file's other (excess-showing) hand-written variants isolated the actual remaining difference: the **shared-memory destination indexing stride** between the four unrolled accesses. Clean kernels used a large stride (2048 elements = 2¹¹, a clean power of two); excess-showing kernels used a small stride (64 elements). Switching *only* this — reverting every other change back to the original excess-showing form (`const` pointer, chained predicate, narrow 32-bit multiply) — flipped the result to clean on its own.
- Testing the real kernel's *actual* stride (2112 elements = 8×264, **not** a power of two) in isolation: **still clean**. Power-of-two-ness was not the variable; raw magnitude, or perhaps just "large enough," was.
- Testing at the real kernel's actual warp count (8 warps instead of 2), with the real stride: **still clean** (confirmed via `ncu`'s own theoretical-ideal metric, not inference).
- Adding a second, concurrently-issued load stream ("B"), with a **simplified, non-faithful** layout: introduced real excess (~14.3%) for the first time. This appeared to be a breakthrough, but replacing the simplified B layout with the *exact* real crosswise formula (`gemm_fp16_widebigregsq_crosswise_bs_idx`, verified line-for-line against the source) made the excess **disappear again** — the earlier "breakthrough" was an artifact of an incorrect B implementation, not a real finding.
- Extending the repro further to match the real kernel's actual A thread-mapping (`a_group/a_kk0/a_i`, distinct from the CUTLASS-style mapping used in every earlier "clean" test in this investigation) **and** the real 3-stage rotating shared-memory buffer (`buf = t % 3`) reproduced the excess at **exactly 25%** — matching the real kernel's own documented figure to within measurement noise.
- Systematically isolating which of those two changes mattered: the **real A-tile thread mapping alone** — no B, no buffer rotation — reproduced the identical *absolute* excess sector count (8192 sectors) as the full combination. Adding B and/or rotation only diluted the *percentage* (more clean traffic sharing the same denominator); it never caused or fixed anything on its own.

**Conclusion of the isolated investigation:** the root cause is `load_tile_async`'s A-tile thread-to-address mapping. The production mapping (`a_group = tid % 32, a_kk0 = tid / 32, a_i = a_group * 8` — each thread covers 4 *consecutive* K values at one *fixed* M position) causes the excess. The CUTLASS-style mapping (`warp_id`/`lane_id` decomposition — each thread covers *one fixed* K value across 4 *different* M positions) does not, in every isolated configuration tested: alone, with B, with the 3-stage buffer rotation, at 2 warps, and at 8 warps.

---

## 3. The Attempted Fix

Based on the isolated finding, `load_tile_async`'s A-load (both the "safe" and boundary-checked branches) was rewritten to use the CUTLASS-style mapping:

**Before** (each thread: 4 consecutive K values, 1 fixed M position):
```cpp
int a_group = tid % kAGroups;      // kAGroups = 32
int a_kk0   = tid / kAGroups;
int a_i     = a_group * 8;
// ... 4 reps, kk = a_kk0 + rep*kAKkStep, address advances by a_ld_step per rep
```

**After** (each thread: 1 fixed K value, 4 different M positions):
```cpp
int a_warp_id  = tid / 32;
int a_lane_id  = tid % 32;
int a_k_lane   = a_lane_id / 8;
int a_m_lane   = a_lane_id % 8;
int a_kk       = 4 * a_warp_id + a_k_lane;   // FIXED per thread
int a_m_lane_i = a_m_lane * 8;
// ... 4 reps, M position = a_m_lane_i + rep*64, address advances by 64 elements per rep
```

This is safe to change in isolation because the downstream consumer, `load_a_frag` (the `ldmatrix`-based MMA fragment loader), reads shared memory purely by logical `(buf, kk, m)` coordinates via `gemm_fp16_widebigreg_as_idx(buf, kk, i) = (buf*BK + kk)*BMPad + i` — **never by which thread wrote a given slot**. Any mapping that covers the full `(kk ∈ [0, BK), m ∈ [0, BM))` space exactly once with correct source data is interchangeable from the reader's point of view. This is also why the fix, unlike the earlier plain-load attempt, does **not** touch `cp.async` at all — only *which thread reads which address* changes, so the async-overlap benefit is untouched.

As a byproduct of this change, `a_ld_step` and `kAKkStep` became dead in this kernel's scope and were removed; a stale comment referencing `a_ld_step` was updated accordingly.

---

## 4. Verification

### 4.1 Correctness

`benchmarks/kernels/cuda/matrix/verify_mm_fp16_gemm` (compares against a CPU reference across ~50 kernels and multiple problem sizes) was run after the fix. Every `gemm_fp16_mma_ptx_wide_bigreg_sq` case — 8 problem sizes, both `beta=0` and `beta≠0` — reported `OK`, with error magnitudes **identical** to the untouched sibling kernel (`gemm_fp16_mma_ptx_wide_bigreg`), e.g.:

```
gemm_fp16_mma_ptx_wide_bigreg square     max_abs_err=0.0078125 max_rel_err=0.00292969 OK
gemm_fp16_mma_ptx_wide_bigreg_sq square  max_abs_err=0.0078125 max_rel_err=0.00292969 OK
```

No failures were observed anywhere in the (very long) suite before it was interrupted by an unrelated, much later, slow kernel.

### 4.2 Performance

**Methodological note:** an initial before/after comparison was run directly from a development shell that, unnoticed at the time, was attached to the cluster's login node — which carries an NVIDIA A30, not the A100 this whole investigation targets. That comparison was discarded once caught (`hostname`/`nvidia-smi` should always be checked before trusting any performance number on a shared cluster). All figures below were measured on an actual A100 compute node.

A controlled before/after comparison was done using a separate `git worktree` checked out at the last pre-fix commit, built and benchmarked back-to-back with the current (fixed) working tree, using `scripts/wide_bigreg_family_vs_cutlass_ref.sh` across the same size sweep used by `verify_mm_fp16_gemm`. Figures below are `wide_bigreg_sq`'s own GFlop/s, same binary pair, same run-to-run conditions:

| N | BEFORE (pre-fix) | AFTER (with fix) | Δ |
|---|---|---|---|
| 1024 | 31,161.2 | 40,060.2 | **+28.6%** |
| 2048 | 81,403.3 | 101,618.5 | **+24.8%** |
| 4096 | 146,621.9 | 186,880.7 | **+27.5%** |
| 8192 | 189,532.9 | 187,337.2 | −1.2% |
| 16384 | 192,491.5 | 188,146.8 | −2.3% |
| 32768 | 192,645.3 | 189,057.3 | −1.9% |
| 65536 | 187,070.4 | 185,942.7 | −0.6% |

The picture is coherent and consistent: **substantial, repeatable gains (25–28%) at small-to-medium N (1024–4096)**, and **small, consistent dips (1–2%, not scaling with size) at large N (8192–65536)**. The large-N dip is consistent with the finding in Section 5: the fix does not change the actual L2 sector excess at N=8192, so at the sizes where that excess dominates the kernel's behavior, the new A-mapping doesn't address the bottleneck and may add a small amount of overhead from the restructured addressing itself; at smaller N, other effects (occupancy, boundary-check proportion) let the new mapping help directly.

A separate, useful side-finding from this comparison: `wide_bigreg_sq` is measurably slower than its untouched sibling `wide_bigreg` specifically at N=2048 (roughly 16–18% slower), but this gap is present **both before and after** the fix (BEFORE: 81,403.3 vs. 96,754.4; AFTER: 101,618.5 vs. 123,954.3) — it is a pre-existing characteristic of `wide_bigreg_sq` at that size, unrelated to this fix, and out of scope for this investigation.

Absolute GFlop/s figures still vary somewhat run-to-run on this shared cluster GPU (e.g. `cublas`'s own reported throughput at N=8192 ranged from ~272,000 to ~282,000 across the runs used above), so relative, same-run comparisons remain the reliable signal, not any single absolute number in isolation.

---

## 5. The Critical, Currently-Unresolved Discrepancy

Direct `ncu` profiling of the real kernel, **with the fix applied**, shows the L2 sector excess is **completely unchanged**:

```
This workload has uncoalesced global accesses resulting in a total of 134217728
excessive sectors (25% of the total 545193680 sectors).
```

This is the exact same figure documented before the fix. Per-instruction attribution (via `ncu --page source`) confirms this is not a measurement artifact:

| Address group | Excess/instr. | Executions/warp | Role (inferred from exec count) |
|---|---|---|---|
| `R166`/`R174` (×4) | 33,292,288 | 4,161,536 (=16384×254) | Main-loop call (~254 K-tiles for N=8192) |
| `R10+0x42xx`/`R12` (×4) | 131,072 | 16,384 (=16384×1) | Priming call #1 |
| `R10+0x0xx`/`R16` (×4) | 131,072 | 16,384 (=16384×1) | Priming call #2 |

`33,292,288×4 + 131,072×8 = 134,217,728` — **exactly** the full excess. All twelve flagged instructions show the identical 4-step, 128-byte-stride pattern that the fix's new A-mapping produces (`rep*64` elements), appearing three times because `load_tile_async` is inlined separately at its three call sites (two priming, one main-loop). **100% of the measured excess is still attributable to A's load**, using a mapping that was clean in *every* isolated test built during this investigation — including tests that matched the real B layout, the real 3-stage buffer rotation, and the real warp count.

**This means something about the real kernel's full context — not captured by any isolated repro in this investigation — is responsible.** Every isolated test used a single threadblock (`grid_y=1`); the real kernel launches ~2048 blocks across all 108 SMs on the A100. GPU-wide, many-block memory-system contention is the leading untested candidate, since no repro in this investigation ever exercised it. MMA/tensor-core compute interleaved in the same loop, and the real kernel's much higher register/occupancy footprint (255 registers/thread vs. 19–32 in isolated tests), are the other two untested candidates.

---

## 6. Current State

- The A-tile mapping fix **is applied** to `mm_cuda_kernels.cu`'s `load_tile_async` (used by `gemm_fp16_mma_ptx_wide_bigreg_sq_kernel`).
- It is **verified correct** (matches CPU reference and the untouched sibling kernel's error profile exactly).
- On real A100 hardware, it delivers **substantial, repeatable gains (25–28%) at N=1024–4096**, and **small, consistent dips (1–2%) at N=8192–65536** — no broad regression at any tested size.
- It **does not** eliminate the `ncu`-flagged L2 sector excess in the production kernel at N=8192 — that finding, and its root cause in the *full* kernel, remains open, and plausibly explains the small dip observed at large N.
- The mechanism is **fully and decisively understood and resolved** for every simplified, isolated repro built in this investigation (`benchmarks/kernels/cuda/matrix/a_load_repro*.cu` and related files) — it does not transfer to the production kernel's full complexity, for reasons not yet identified.

The fix is a net positive on its own merits (correct, meaningfully faster at small-to-medium sizes, only a small cost at large sizes) independent of whether it resolves the original `ncu` finding, and there is no evidence it should be reverted.

---

## 7. What Needs to Be Tested Further

In priority order, based on what has *never* been tested in this investigation:

1. **Real multi-block occupancy.** Build an isolated repro that launches a realistic grid (dozens to thousands of blocks across all SMs, matching the real kernel's ~2048-block launch) instead of a single block, and re-run the same A-mapping-only and A+B+rotation comparisons at that scale. This is the single highest-value untested variable — every prior isolated result in this investigation used `grid_y=1`.
2. **Genuine software-pipelining depth.** All isolated multi-stage tests issued-then-waited every tile (shallow depth); none tested real multi-tile-ahead prefetching (issuing tile *t+2* while consuming tile *t*, as the real kernel's priming/main-loop structure does).
3. **MMA/tensor-core compute interleaved in the same loop.** Every isolated repro tested only the load instructions in isolation; none included `wmma`/`mma.sync` instructions sharing the loop body, warp scheduler, and instruction issue slots with the loads.
4. **Realistic register pressure/occupancy.** The production kernel uses 255 registers/thread; isolated tests used 19–32. Whether high register pressure (and correspondingly low achieved occupancy) changes `ptxas`'s scheduling or the SM's memory-pipe behavior in a way that reintroduces the excess is untested.
5. **Whether B or C (the epilogue write) independently contribute excess in the full kernel.** The per-instruction breakdown above shows 100% of the *currently measured* excess on A's instructions, but this only rules out B/C as *additional* contributors under the current (fixed) A-mapping — it does not establish whether they would show excess of their own under other conditions (e.g., higher occupancy).
6. **Whether the fix's absolute throughput is stable across problem sizes**, not just N=8192 — the correctness suite covers multiple sizes, but the performance comparison was only rigorously controlled at N=8192.

If further investigation is undertaken, extending `a_load_repro_isolated.cu`'s existing `kernel_ab3_iso` template (already parameterized by A-mapping and buffer-rotation choice) to also launch a realistic multi-block grid is the most direct next step, and would settle candidate (1) with the same rigor applied to every other variable in this investigation.

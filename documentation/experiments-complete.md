# TOMOS

**Local setup:** benchmarks that compare against CUTLASS (any
`scripts/*vs_cutlass_ref*.sh`) default to fetching/using it at
`build/cutlass-ref`; LSQR's test matrices default to `build/matrices`. For
a persistent custom location instead of passing
`CUTLASS_DIR=`/`--matrix-dir` on every invocation, copy
`.configure.sh.example` (repo root) to `.configure.sh` and set
`CUTLASS_DIR`/`MATRIX_DIR` there — it's gitignored and sourced
automatically by every script that needs these.

## Complete list of methods

### FP32 GEMM

#### Experiment:

The complete "32-bit-compute" GEMM sweep as actually implemented in
`benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu`: cuBLAS (Sgemm), real
CUTLASS (SIMT fp32 `device::Gemm`), the full lineage of this codebase's own
native (unpacked) fp32 GEMM experiments (not just the promoted/plotted
`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`), and
every 32-bit-compute-tier variant tried for both packed formats (`fp32_16`
and `fp32_bf16`) -- decode `wh`/`wr` back to a real fp32 value (fused inline,
adaptively, or via a separate materialize pass) before computing, as opposed
to the "16-bit-compute" tier that computes directly in half/bf16. This is
the COMPLETE set of methods this benchmark driver times, not the 7-series
subset plotted in `plot_gemm_format_32_16_compute_32_square.png`/`_rect.png`
(see this document's own "Plots" section above for that promoted subset).

**Note on method count:** `benchmarks_gemm_fp32.cu` contains 51 lines
matching `ms_XXX = time_call(start, stop, ...)`, but 2 of those (both
reassigning `ms_native`, around lines 1175 and 1183) are commented-out
historical code -- one calls the superseded `gemm_fp32_tiled_syncpipe3`
(already timed separately below as its own live leg), the other calls
`gemm_fp32_tiled_cpasync_dualbuf` (the plain/unpacked cp.async dualbuf
kernel, which is NOT called anywhere else live in this file). Only **49**
`time_call` legs actually execute. This list documents all 49.

**Scripts:**
- `scripts/benchmarks_gemm_format_32_16_compute_32.sh` — builds and runs
  the benchmark.
- `scripts/plot_gemm_format_32_16_compute_32.sh` — regenerates the two
  `.png`/`.fig` plots from an already-saved CSV (no rebuild/rerun); wraps
  the MATLAB function `scripts/plot_gemm_format_32_16_compute_32.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu`.

**Kernels called (all methods tried):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| TOMOS FP32 / FP16 | `mm_fp32_16_tiled_cpasync_dualbuf` | `fp32_16_combine` | promoted/plotted; see this document's own promoted-subset entry above |
| TOMOS FP32 / FP16 (fast_normal) | `mm_fp32_16_tiled_cpasync_dualbuf_fast_normal` | `fp32_16_combine_fast_normal` | UNCONDITIONAL fast decode -- skips the control-bit dispatch entirely; UNSAFE on arbitrary data (silently misreads extended/saturate bits as exponent bits if any element has control=1), timed for a throughput ceiling only |
| TOMOS FP32 / FP16 (truncated) | `mm_fp32_16_tiled_cpasync_dualbuf_truncated` | `fp32_16_combine_truncated` | promoted/plotted |
| TOMOS FP32 / FP16 (truncated, fused midpoint_exact) | `mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated` | `fp32_16_combine_truncated` | fully-fused (not materialize) counterpart of the truncated dualbuf kernel above; compiles with a real spill (26 STL + 17 LDL at 128 registers) unlike the non-truncated fused kernel's 0-spill compile |
| TOMOS FP32 / FP16 (adaptive, block-wide) | `mm_fp32_16_tiled_cpasync_dualbuf_adaptive` | `fp32_16_combine_fast_normal` (clean tiles) / `fp32_16_combine` (fallback) | per-tile block-wide dispatch (`__syncthreads_or`-style) between the cheap and full decode |
| TOMOS FP32 / FP16 (adaptive, per-warp) | `mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp` | same pair as above | same fast/full split as the block-wide adaptive kernel, but voted per-warp (`__any_sync`) instead of per-block |
| TOMOS FP32 / FP16 (multin) | `mm_fp32_16_tiled_cpasync_dualbuf_multin` | `fp32_16_combine` | each block covers 2 N-tiles, reusing one shared-memory A decode across both, instead of making the decode itself cheaper |
| TOMOS FP32 / FP16 (workspace, pre-midpoint_exact) | `mm_fp32_16_gemm_materialize` | `fp32_16_combine` (via `mm_fp32_16_materialize_chunk_kernel`) | materialize-then-native-GEMM structure, but the second-stage native call is the OLDER `gemm_fp32_tiled_cpasync_dualbuf`, not `midpoint_exact` |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_gemm_materialize_midpoint_exact` | `fp32_16_combine` (via `mm_fp32_16_materialize_chunk_kernel`) | promoted/plotted |
| TOMOS FP32 / FP16 (workspace, fast_normal) | `mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal` | `fp32_16_combine_fast_normal` (via chunk kernel) | same materialize-then-native structure as the promoted workspace kernel, chunk-decode pass uses the unconditional fast decode instead; same unsafe-on-arbitrary-data caveat |
| TOMOS FP32 / FP16 (workspace, vectorized) | `mm_fp32_16_gemm_materialize_midpoint_exact_vec` | `fp32_16_combine` (via a vectorized `ushort4`/`float4` chunk kernel, 4 rows/thread) | tests whether vectorizing only the chunk-decode pass measurably speeds up materialize's total time |
| TOMOS FP32 / FP16 (fused, midpoint_exact) | `mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | `fp32_16_combine` | fully-fused counterpart: ports the entire native `midpoint_exact` tiling architecture unchanged; `load_global_advance` reads A's raw `wh`/`wr` `ushort4`s from global memory and decodes them via `fp32_16_combine` straight into a register-held `float4` (not shared memory), which a separate `store_to_smem` step then writes to the shared-memory `As` buffer -- decode-in-registers-then-store, not a direct-to-shared-memory decode; no separate scratch buffer or elementwise decode kernel either way |
| TOMOS FP32 / FP16 (accurate, untiled) | `mm_fp32_16_accurate` | `fp32_16_combine` | one-thread-per-output-element fallback, no shared-memory tiling; the always-correct fallback every other fp32_16 kernel in this lineage calls for transposed operands |
| TOMOS FP32 / FP16 (tiled) | `mm_fp32_16_tiled` | `fp32_16_combine` | first shared-memory-tiled kernel in the lineage (128x128/BK=8, 8x8 register-blocked), decodes each A element once into shared memory instead of once per use |
| TOMOS FP32 / FP16 (tiled, vec) | `mm_fp32_16_tiled_vec` | `fp32_16_combine` | same tiling as `mm_fp32_16_tiled`, A's wh/wr loaded as `ushort4` and B as `float4`, both scattered/decoded into shared memory; requires `ld_a`/`m`/`ld_b`/`k` multiples of 4, else falls back to `mm_fp32_16_tiled` |
| TOMOS FP32 / FP16 (cpasync, vec) | `mm_fp32_16_tiled_cpasync_vec` | `fp32_16_combine` | same tiling/vectorization as `tiled_vec`, but A's raw wh/wr bytes are prefetched one tile ahead via `cp.async` into a staging buffer before decode (decode itself stays synchronous; only the global fetch is hidden); B unchanged (still synchronous, nothing to hide) |
| TOMOS FP32 / FP16 (cpasync, vec2) | `mm_fp32_16_tiled_cpasync_vec2` | `fp32_16_combine` | doubles BK from 8 to 16 vs `cpasync_vec` for wider B coalescing; B drops back to a single (non-double-buffered) shared buffer since it was never cp.async-prefetched anyway, keeping total static shared memory at 40KB instead of the 48KB limit a naive double-buffer-everything approach would hit |
| TOMOS FP32 / FP16 (cpasync, vec2 branchless) | `mm_fp32_16_tiled_cpasync_vec2_branchless` | `fp32_16_combine_branchless` | identical to `vec2` except A's decode calls `fp32_16_combine_branchless` -- bit-exact same result as `fp32_16_combine` (all three normal/extended/saturate paths computed unconditionally, merged via arithmetic mask, no hardware conversion), targeting warp divergence rather than trading away any accuracy |
| TOMOS FP32 / FP16 (cpasync, vec3) | `mm_fp32_16_tiled_cpasync_vec3` | `fp32_16_combine` | enlarges the BLOCK tile itself (512 threads/block, same 16-warp occupancy as two of vec2's 256-thread blocks) rather than the thread tile; fewer, bigger blocks -- costs concurrent-block-per-GPU parallelism at small N |
| TOMOS FP32 / FP16 (cpasync, vec4) | `mm_fp32_16_tiled_cpasync_vec4` | `fp32_16_combine` | keeps vec2's exact block/grid size but reshapes the THREAD tile (TN 8->4, doubling BN/TN decode-reuse from 16 to 32 threads/decoded-A-value); 512 threads/block with half the accumulators each; A and B loads become single-pass (vs vec2/vec3's two-pass) since `(BM/4)*BK` and `BN*(BK/4)` now divide evenly by thread count |
| TOMOS FP32 / FP16 (cpasync, warptile) | `mm_fp32_16_tiled_cpasync_warptile` | `fp32_16_combine` | identical block tile/thread count/per-thread work to `vec2`; only the thread-to-output-position mapping changes, from a flat 16x16 (tx,ty) grid to an explicit 4x2-warp x 4x8-lane grid so each warp owns a warp-aligned, contiguous 32x64 output region (cutlass-style warp tiling), isolated from every other lever tried |
| TOMOS FP32 / FP16 (cpasync, warptile2) | `mm_fp32_16_tiled_cpasync_warptile2` | `fp32_16_combine` | identical to `warptile` except the warp/lane grid shape is transposed: 2x4 warps x 8x4 lanes (64x32 warp tile instead of 32x64), same total resources, isolating whether warptile's result came from warp-aligned addressing itself or this specific aspect ratio |
| TOMOS FP32 / FP16 (cpasync, warptile3) | `mm_fp32_16_tiled_cpasync_warptile3` | `fp32_16_combine` | continues the warptile->warptile2 trend (taller warp tile won) to its extreme: 1x8 warps x 16x2 lanes, each warp spanning the FULL BM=128 in a 128x16 tile; same total threads/accumulators/shared memory as warptile/warptile2 |
| TOMOS FP32 / FP16 (cpasync, dualbuf2) | `mm_fp32_16_tiled_cpasync_dualbuf2` | `fp32_16_combine` | same dual-cp.async structure as the promoted `dualbuf` kernel, BK doubled 16->32 for wider B coalescing; doubling BK roughly doubles every shared-memory region (~128KB/block), capping occupancy at 1 block/SM regardless of register count (a shared-memory, not register, occupancy ceiling) |
| TOMOS FP32 / FP16 (cpasync, tripipe) | `mm_fp32_16_tiled_cpasync_tripipe` | `fp32_16_combine` | genuine 3-stage cp.async pipeline (tile i+2 prefetched while tile i computes) vs dualbuf's 2-stage; needs 3 tiles' worth of buffers live simultaneously, so BM is halved to 64 to keep shared memory at ~72KB/block (2 blocks/SM); confounds pipeline depth with a smaller/different-shaped tile |
| TOMOS FP32 / FP16 (cpasync, dualbuf3) | `mm_fp32_16_tiled_cpasync_dualbuf3` | `fp32_16_combine` | inverse of dualbuf2: shrinks BK back to 8 (from dualbuf's 16) at dualbuf's same BM=BN=128, isolating BK alone (not confounded with pipeline depth or tile size the way tripipe is); every buffer lands at ~32KB/block, comfortably under the static 48KB limit; A and B loads become single-pass at this BK |
| TOMOS FP32 / FP16 (cpasync, dualbuf4) | `mm_fp32_16_tiled_cpasync_dualbuf4` | `fp32_16_combine` | keeps dualbuf's real cp.async prefetch for both A and B, BN=128/BK=16 unchanged; shrinks ONLY BM (128->64), doubling row-tile count to fill more SMs at small N without touching tile-shape/BK/pipeline-depth; measured non-monotonic vs dualbuf across N on real hardware (wins/loses at different N ranges, no simple grid/SM-count heuristic explains it) -- both kernels kept as directly-callable options, no dispatcher |
| TOMOS FP32 / FP16 (cpasync, dualbuf5) | `mm_fp32_16_tiled_cpasync_dualbuf5` | `fp32_16_combine` | mirror of dualbuf4: shrinks BN (128->64) instead of BM, keeping BM=128/BK=16 and the same real dual-cp.async machinery; tests whether dualbuf4's win is a general small-tile effect or specific to the M-axis (since B already carries an extra staging+transpose step A doesn't have) |
| TOMOS FP32 / FP16 (cpasync, dualbuf6) | `mm_fp32_16_tiled_cpasync_dualbuf6` | `fp32_16_combine` | same tile shape/double-buffering as dualbuf; only change is inside the inner k-loop -- explicitly software-pipelines the a_reg/b_reg shared-memory load (prefetch kk+1's registers while computing kk's) instead of relying on ptxas's own scheduling within the unrolled BK=16 loop; doubles a_reg/b_reg register footprint |
| TOMOS FP32 / FP16 (splitk) | `mm_fp32_16_tiled_cpasync_splitk` (calls `mm_fp32_16_splitk_prescale_kernel` then `mm_fp32_16_tiled_cpasync_splitk_kernel`) | `fp32_16_combine` | Split-K: dualbuf's exact 128x128/BK=16 tile/warp shape, but P blocks each reduce a K/P slice and `atomicAdd` their partial contribution into C, instead of one block reducing the full K range -- targets the too-few-output-tiles problem dualbuf4/5 were also fighting, without shrinking the (already-efficient) output tile; a separate prescale kernel applies `beta*C` exactly once before any slice's atomicAdd; `k_slice` is rounded up to a full BK tile to keep every slice's cp.async transfer alignment-safe |
| TOMOS FP32 / FP16 (cpasync, dualbuf7) | `mm_fp32_16_tiled_cpasync_dualbuf7` | `fp32_16_combine` | same tile/warp shape and real cp.async decode as dualbuf for A, but B REVERTS to a synchronous one-step global-read-then-transpose write (like the plain-fp32 `gemm_fp32_tiled_cpasync_vec3_kernel`'s B path) instead of dualbuf's two-step cp.async-staging-then-transform; motivated by `ncu` profiling showing dualbuf is compute-bound (86% SM/FMA, 20% DRAM) with no memory latency left to hide, testing whether B's staging round-trip is now pure overhead |
| TOMOS FP32 / FP16 (fast, untiled) | `mm_fp32_16_fast` (`tomos::cuda::experimental` / plain `tomos::cuda`) | `fp32_16_half_only_branchless_no_overflow_nomask` (wh only, wr unused) | one-thread-per-output-element naive kernel like `mm_fp32_16_accurate`, but decodes A from `wh` ALONE via a half-precision-only branchless decode (no overflow handling, no redundant post-shift mask), promotes the resulting `__half` to `double` for accumulation, outputs `float` C -- a 32-bit-output kernel with a cheaper, lossier single-word decode than the full `fp32_16_combine` path |
| TOMOS FP32 / FP16 (fast, untiled, conditional) | `mm_fp32_16_fast` (`tomos::cuda::experimental::conditional` / `tomos::cuda::conditional`) | `fp32_16_half_only` (wh only, wr unused) | byte-for-byte the same kernel shape as the row above, but decodes via `fp32_16_half_only` instead -- the full-safety sibling that branches on both `control` and `direction` (handles overflow AND underflow distinctly, including genuine fp16-subnormal reconstruction), vs the other kernel's branchless no-overflow-assumed variant; this is a genuinely different decode call, not a namespace copy of the same body |

*fp32_bf16 (BF16-family) 32-bit-compute variants -- additional rows:*
| TOMOS FP32 / BF16 | `mm_fp32_bf16_tiled_cpasync_dualbuf` | `fp32_bf16_combine` | cp.async double-buffered tiled GEMM, an explicit (per its own comment) port of `mm_fp32_16_tiled_cpasync_dualbuf_kernel`; real body lives in `fp32_bf16_cuda_kernels.cu` (not `fp32_16_cuda_kernels.cu`, which only forward-declares it) |
| TOMOS FP32 / BF16 (fused, truncated) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | `fp32_bf16_combine` | promoted/plotted (README's "TOMOS FP32 / BF16 (truncated)") |
| TOMOS FP32 / BF16 (rounded) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded` | `fp32_bf16_combine_rounded` | same fused architecture as the row above but via the rounding (not truncating) bf16 encode/decode pair; real measured register spill (REG:128 STACK:56, 23 STL + 11 LDL) vs the plain fused kernel's clean 0-spill compile |
| TOMOS FP32 / BF16 (workspace) | `mm_fp32_bf16_gemm_materialize_rounded` | `fp32_bf16_combine_rounded` (via `mm_fp32_bf16_materialize_rounded_chunk_kernel`) | promoted/plotted |
| TOMOS FP32 / BF16 (rounded, adaptive block-wide) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive` | `fp32_bf16_combine_rounded` (dirty tiles) / `fp32_bf16_combine` (clean fallback) | per-BK=8-tile block-wide `__syncthreads_or` reduction checks whether the tile has any rounded element at all; measured spill reduction (REG:128 STACK:16, 3 STL + 2 LDL) but adds 4 new `BAR.RED.OR` barrier instructions |
| TOMOS FP32 / BF16 (rounded, adaptive per-warp) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp` | same pair as above | same idea as the block-wide adaptive kernel, but the dispatch is per-WARP via `__any_sync` (compiles to `VOTE.ANY`, not a barrier instruction) instead of per-BLOCK; measured REG:128 STACK:32, 6 STL + 6 LDL -- a smaller spill reduction but none of the block-wide kernel's added-barrier risk |
| TOMOS FP32 / BF16 (accurate, untiled) | `mm_fp32_bf16_accurate` | `fp32_bf16_combine` | one-thread-per-output-element fallback mirroring `mm_fp32_16_accurate_kernel` exactly (same untiled design, same double-accumulator choice); lives in `fp32_bf16_cuda_kernels.cu`, the safe fallback for transposed operands or shapes `mm_fp32_bf16_tiled`'s fast path doesn't cover |
| TOMOS FP32 / BF16 (tiled) | `mm_fp32_bf16_tiled` | `fp32_bf16_combine` | shared-memory-tiled, register-blocked (8x8/thread) GEMM, same 128x128/BK=8 tile shape and thread mapping as `mm_fp32_16_tiled_kernel`, decode call swapped for `fp32_bf16_combine`; NO_TRANSPOSE x NO_TRANSPOSE only, lives in `fp32_bf16_cuda_kernels.cu` |
| TOMOS FP32 / BF16 (accurate, rounded) | `mm_fp32_bf16_accurate_rounded` | `fp32_bf16_combine_rounded` | naive always-correct-for-any-shape fallback for the ROUNDED encoding, mirroring `mm_fp32_bf16_accurate_kernel` exactly except the decode call; exists because (unlike fp32_16, where the plain `fp32_16_combine` safely decodes both its own output and the truncated variant's) `fp32_bf16_combine`(plain) silently produces WRONG results on rounded-encoded data -- this is the correctly-decoding fallback used for transpose/misalignment/malloc-failure by both `mm_fp32_bf16_gemm_materialize_rounded` and the fused `..._midpoint_exact_rounded` kernel |
| TOMOS FP32 | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | — (native, unpacked storage) | promoted/plotted |
| TOMOS FP32 (baseline tiled) | `gemm_fp32_tiled` | — (native) | earliest kernel in the whole lineage: register-blocked, shared-memory-tiled 128x128/BK=8 GEMM, 8x8 output elements per thread, no double buffering; the jump from the one-thread-per-output-element `gemm_generic_kernel` that closes most of the throughput gap to cuBLAS |
| TOMOS FP32 (double-buffered) | `gemm_fp32_tiled_db` | — (native) | manual double buffer: next K-slice's global loads issued into the other shared buffer before this slice's register accumulation; requires `__launch_bounds__(256,2)` to avoid a register-pressure occupancy regression that would otherwise make it slower than baseline |
| TOMOS FP32 (cp.async) | `gemm_fp32_tiled_cpasync` | — (native) | same double-buffered tiling as `_db`, but the prefetch goes through sm_80 `__pipeline_memcpy_async` cp.async instead of a manual load-into-register-then-store-to-shared round trip |
| TOMOS FP32 (cp.async, vec) | `gemm_fp32_tiled_cpasync_vec` | — (native) | adds two vectorizations on top of `_cpasync`: compute-loop shared-memory reads become `float4` instead of 8 scalar reads; A's cp.async global load becomes one 16-byte transfer of 4 rows instead of 4 separate 4-byte transfers. B's load deliberately stays scalar. Requires `ld_a`/`m` multiples of 4, else falls back to `_cpasync` |
| TOMOS FP32 (cp.async, vec2) | `gemm_fp32_tiled_cpasync_vec2` | — (native) | same as `_vec` plus vectorizing B's global load too: since B's compute-layout shared destination isn't contiguous, does one synchronous 16-byte `float4` global read then scatters its 4 lanes into shared with 4 ordinary stores (loses cp.async overlap for B specifically, still cuts instruction count 4x). Requires `ld_b`/`k` multiples of 4, else falls back down the chain |
| TOMOS FP32 (cp.async, vec3) | `gemm_fp32_tiled_cpasync_vec3` | — (native) | same structure as `_vec2`, K-slice doubled from BK=8 to BK=16 under its own distinct tile constants, specifically to widen B's coalesced access span per column (32B -> 64B); costs double the shared memory (32KB total, still under the 48KB static limit) |
| TOMOS FP32 (cp.async, dualbuf) | `gemm_fp32_tiled_cpasync_dualbuf` | — (native) | native sibling of the packed `mm_fp32_16_tiled_cpasync_dualbuf`/`mm_fp32_bf16_tiled_cpasync_dualbuf` kernels: unlike `_vec3` (only A through cp.async), routes BOTH A and B through cp.async, staging B in its native K-fast layout then transposing to N-fast via a shared-to-shared scatter; direct predecessor of the already-documented `cpasync_dualbuf_regpipe`. Only referenced live via a commented-out benchmark call (per this document's own "Note on method count") |
| TOMOS FP32 (regpipe) | `gemm_fp32_tiled_cpasync_dualbuf_regpipe` | — (native) | register-level software-pipelined sibling of the original cp.async dualbuf kernel |
| TOMOS FP32 (syncpipe, 2-stage) | `gemm_fp32_tiled_dualbuf_syncpipe` | — (native) | no-cp.async pipeline modeled on real CUTLASS's `MmaPipelined::gemm_iters` (`mma_pipelined.h`) |
| TOMOS FP32 (syncpipe, incptr) | `gemm_fp32_tiled_dualbuf_syncpipe_incptr` | — (native) | rolling-pointer addressing (as in `syncpipe3_incptr`) applied to the 2-stage syncpipe kernel instead |
| TOMOS FP32 (syncpipe3) | `gemm_fp32_tiled_syncpipe3` | — (native) | genuine 3-stage extension of dualbuf_syncpipe: two tiles of lookahead, third shared-memory buffer slot, no cp.async, no staging buffer |
| TOMOS FP32 (syncpipe3, incptr) | `gemm_fp32_tiled_syncpipe3_incptr` | — (native) | rolling-pointer addressing (fewer repeated multiplies) instead of recomputing addresses from scratch each iteration; regressed vs plain syncpipe3 |
| TOMOS FP32 (syncpipe3, incptr_lean) | `gemm_fp32_tiled_syncpipe3_incptr_lean` | — (native) | trims incptr's redundant live register state (4 K-position counters -> 1 shared counter, 4 tracked pointers -> 2) |
| TOMOS FP32 (syncpipe3, incptr_lean2) | `gemm_fp32_tiled_syncpipe3_incptr_lean2` | — (native) | predicate-collapsed bounds checks + a 32-bit remaining-K counter added on top of lean; measured MORE spilling than lean |
| TOMOS FP32 (syncpipe3, incptr_nostride) | `gemm_fp32_tiled_syncpipe3_incptr_nostride` | — (native) | drops precomputed stride locals, inlines the multiply at each use site; best spill result of the whole incptr family |
| TOMOS FP32 (syncpipe3, incptr_nostride_kremain) | `gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain` | — (native) | decrementing `k_remaining` instead of `tiles_fetched*BK`, added on top of nostride; mixed/slight regression |
| TOMOS FP32 (syncpipe3, incptr_nostride_bcol) | `gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol` | — (native) | derives `b_col[1]` on demand instead of tracking it, added on top of nostride; regression on every spill metric |
| TOMOS FP32 (syncpipe3, incptr_kremain) | `gemm_fp32_tiled_syncpipe3_incptr_kremain` | — (native) | the `k_remaining` change alone, applied to plain incptr_lean rather than stacked on nostride |
| TOMOS FP32 (dualbuf_syncpipe, incptr_nostride) | `gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride` | — (native) | nostride addressing applied to the shallower 2-stage pipeline instead of 3-stage syncpipe3, to isolate whether pipeline depth (not addressing) drove the spilling |
| TOMOS FP32 (syncpipe3, incptr_nostride_lb1) | `gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1` | — (native) | same kernel as nostride except `__launch_bounds__(256,1)` instead of `(256,2)`; eliminates spilling entirely (matches CUTLASS's own 0/0/0) at the cost of halving max resident blocks/SM |
| TOMOS FP32 (syncpipe3, finegrain) | `gemm_fp32_tiled_syncpipe3_finegrain` | — (native) | two CUTLASS-derived structural traits (quadrant-interleaved accumulator loads; staggered next-next-tile global loads) applied to syncpipe3; ptxas mostly undid the source-level interleaving |
| TOMOS FP32 (syncpipe3, dualacc) | `gemm_fp32_tiled_syncpipe3_dualacc` | — (native) | splits each output element's accumulator into two independent partial sums to break the RAW dependency chain; doubling accumulator registers produced severe spilling (worst of the whole investigation) |
| TOMOS FP32 (syncpipe3, snake) | `gemm_fp32_tiled_syncpipe3_snake` | — (native) | boustrophedon FFMA accumulate traversal order on syncpipe3; did NOT move the FFMA `.reuse` ratio and introduced a new spill |
| TOMOS FP32 (syncpipe3, bk8) | `gemm_fp32_tiled_syncpipe3_bk8` | — (native) | same 3-stage rotation as syncpipe3, K-tile depth BK=8 instead of 16 (matches CUTLASS's `ThreadblockShape<128,128,8>`) |
| TOMOS FP32 (syncpipe4, bk8) | `gemm_fp32_tiled_syncpipe4_bk8` | — (native) | byte-for-byte identical to syncpipe3_bk8 except 4 rotating pipeline stages instead of 3, affordable because BK=8 halves each stage's shared-memory cost |
| TOMOS FP32 (syncpipe4, bk8, warptile) | `gemm_fp32_tiled_syncpipe4_bk8_warptile` | — (native) | CUTLASS-matched `WarpShape<32,64,8>` + `RowMajorInterleaved<2>` lane geometry for the compute read path; biggest single structural change in the investigation |
| TOMOS FP32 (syncpipe4, bk8, warptile, nostride) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride` | — (native) | applies the nostride rolling-pointer technique to the BK=8 base for the first time; first kernel in the investigation to compile under 128 registers (126, matching CUTLASS exactly) |
| TOMOS FP32 (…nostride, snake) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake` | — (native) | retries the boustrophedon accumulate order (which failed on syncpipe3) on this leaner base; reuse ratio genuinely moves this time (73.4%, +6.2pp) |
| TOMOS FP32 (…snake, interleaved) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved` | — (native) | spreads next-kk/next-tile prefetch piecemeal through the FFMA nest instead of batching it up front, mirroring CUTLASS's own compiled SASS mainloop |
| TOMOS FP32 (…interleaved_late) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late` | — (native) | same as interleaved, but the prefetch checkpoints move later in the row loop (i==5/6/7 instead of i==2/4/6) |
| TOMOS FP32 (…midpoint) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint` | — (native) | isolates whether interleaved's register cost came from splitting `load_compute_regs` or the mid-loop checkpoint itself; keeps the call combined, one checkpoint at i==4 |
| TOMOS FP32 (…midpoint_uncapped) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped` | — (native) | same as midpoint but `__launch_bounds__(256,2)` relaxed to `(256)`, matching how CUTLASS itself is compiled; measured dramatically slower despite eliminating the spill |
| TOMOS FP32 (…midpoint_narrow) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow` | — (native) | same as midpoint (keeping the `(256,2)` cap) except `tiles_fetched` narrowed `size_t` -> `unsigned int` |
| TOMOS FP32 (…midpoint_inline) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline` | — (native) | same as midpoint except `has_next_kk`/`has_next_tile` are inlined at the checkpoint instead of precomputed named locals |
| TOMOS FP32 (…midpoint_exact, rerun) | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | — (native) | same kernel as the promoted TOMOS FP32 row above, timed a second time via the same call; kept only as a run-to-run noise check |
| CUTLASS | `cutlass::gemm::device::Gemm<...>` (SIMT fp32) | — | promoted/plotted |
| cuBLAS | `cublasSgemm` | — | promoted/plotted |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (declarations in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`); the
  bf16-family `mm_fp32_bf16_tiled_cpasync_dualbuf` kernel's real body is in
  the separate `src/kernels/cuda/matrix/fp32_bf16_cuda_kernels.cu` instead
  (only forward-declared in `fp32_16_cuda_kernels.cu`) -- every other
  `mm_fp32_bf16_*`/`mm_fp32_16_*` kernel in this table lives in
  `fp32_16_cuda_kernels.cu` itself.
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family: `fp32_16_split`/`_clamped`/`_truncated`, `fp32_16_combine`/
  `_truncated`), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family:
  `fp32_bf16_split`/`_rounded`, `fp32_bf16_combine`/`_rounded`).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu`.
- The native `gemm_fp32_tiled_*` lineage (all 28 non-promoted variants below)
  lives in `src/kernels/cuda/matrix/mm_cuda_kernels.cu`, entirely inside
  `namespace tomos::cuda::experimental` (that namespace block spans lines
  36-48727 of that file); the promoted kernel additionally has a thin
  forwarding wrapper outside `experimental` (lines 48729-48741) that the
  benchmark's own plain `cuda::gemm_fp32_tiled_syncpipe4_bk8_warptile_
  nostride_snake_midpoint_exact` call resolves to.

**Method Description:**

*fp32_16 (FP16-family) 32-bit-compute variants -- all built on top of the
`mm_fp32_16_tiled_cpasync_dualbuf` cp.async double-buffered architecture
already described in this document's promoted-subset entry above, unless
noted otherwise:*

- **TOMOS FP32 / FP16 (fast_normal)** (`mm_fp32_16_tiled_cpasync_dualbuf_fast_normal`) —
  byte-for-byte the same kernel as the plain dualbuf kernel, except A's
  decode calls `fp32_16_combine_fast_normal` instead of `fp32_16_combine`:
  this skips the control-bit dispatch entirely and reconstructs every
  element via the "normal" (control=0) branch only. Per that function's own
  correctness contract, this is only valid when every `wh` in the buffer
  genuinely has control=0 -- on any element that needed the extended or
  saturate encoding, it silently misinterprets the offset/direction bits as
  exponent bits. The benchmark uses the *same* rounding-aware generator as
  the plain dualbuf kernel (not a control=0-only generator), so this run is
  UNSAFE on that data and exists purely to measure a decode-cost ceiling, not
  a usable result; correctness is evaluated separately elsewhere.
- **TOMOS FP32 / FP16 (truncated, fused midpoint_exact)**
  (`mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated`) —
  the fully-fused (tiling-integrated, not materialize) counterpart to the
  promoted truncated dualbuf kernel, sharing the same `syncpipe4`/`bk8`/
  `warptile`/`nostride`/`snake`/`midpoint_exact` architecture as native TOMOS
  FP32, decoding via `fp32_16_combine_truncated` fused into the A-tile
  staging step. Unlike its non-truncated fused sibling below (a clean 0-spill
  compile), this one measures a real register spill (26 STL + 17 LDL at 128
  registers) -- the truncated decode function apparently has a meaningfully
  larger register footprint than the rounding-aware `fp32_16_combine`, and
  this timing exists specifically to isolate that spill's runtime cost.
- **TOMOS FP32 / FP16 (adaptive, block-wide)** (`mm_fp32_16_tiled_cpasync_dualbuf_adaptive`) —
  same cp.async dualbuf tiling as the plain kernel, but each staged tile's
  decode dispatches per-block between the cheap `fp32_16_combine_fast_normal`
  path (when the whole tile is control=0, i.e. "clean") and the full
  `fp32_16_combine` fallback (when any element is "dirty") using a
  block-wide reduction over the tile's own staged control bits -- exactly
  the usage `fp32_16_combine_fast_normal`'s own header comment names this
  kernel as the intended caller for. It is expected to be numerically
  bit-identical to the plain dualbuf kernel on the same rounding-aware
  input data, just potentially cheaper when a tile happens to be all-clean.
- **TOMOS FP32 / FP16 (adaptive, per-warp)** (`mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp`) —
  the same fast/full decode split as the block-wide adaptive kernel above,
  but the clean-vs-dirty decision is voted per-WARP via `__any_sync` instead
  of per-BLOCK via a block-wide barrier reduction, avoiding the block-wide
  version's extra synchronization cost at the granularity of a single warp's
  own tile slice instead.
- **TOMOS FP32 / FP16 (multin)** (`mm_fp32_16_tiled_cpasync_dualbuf_multin`) —
  attacks the redundant-decode problem from a different angle than the
  adaptive kernels: each block now covers 2 N-tiles instead of 1, reusing
  one shared-memory A-tile decode across both, since A's own decode doesn't
  depend on which N-tile is being multiplied against it -- this halves how
  many blocks redundantly decode a given M-row range, rather than making any
  single decode call itself cheaper.
- **TOMOS FP32 / FP16 (workspace, pre-midpoint_exact)** (`mm_fp32_16_gemm_materialize`) —
  the original two-pass materialize-then-native-GEMM kernel, structurally
  identical to the promoted workspace kernel (decodes the entire matrix to a
  scratch buffer via a chunked, one-thread-per-element decode kernel using
  `fp32_16_combine`, then calls a native fp32 GEMM unmodified), but its
  second-stage native call is the OLDER `gemm_fp32_tiled_cpasync_dualbuf`
  kernel, predating the `midpoint_exact` promotion -- this timing exists
  specifically to isolate the effect of swapping in `midpoint_exact` as the
  second stage (that comparison is the promoted workspace kernel below).
- **TOMOS FP32 / FP16 (workspace, fast_normal)**
  (`mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal`) — identical
  materialize-then-native-GEMM structure as the promoted workspace kernel
  (same chunking, same unmodified `midpoint_exact` GEMM stage), except the
  chunk-decode pass calls `fp32_16_combine_fast_normal` instead of
  `fp32_16_combine` -- the same unconditional, control=0-only decode used by
  the fast_normal dualbuf kernel above, with the same unsafe-on-arbitrary-
  data caveat, here isolated to only the one-time materialize pass rather
  than the whole fused mainloop.
- **TOMOS FP32 / FP16 (workspace, vectorized)** (`mm_fp32_16_gemm_materialize_midpoint_exact_vec`) —
  same materialize-then-native-GEMM strategy as the promoted workspace
  kernel, but the chunk-decode pass is vectorized: 4 rows decoded per thread
  via `ushort4`/`float4` loads and `fp32_16_combine`, instead of 1 scalar
  element per thread. Since decode is the entire extra cost materialize pays
  over plain native fp32 (the GEMM stage itself is unchanged either way),
  this timing directly answers whether vectorizing that one pass moves
  materialize's total time.
- **TOMOS FP32 / FP16 (fused, midpoint_exact)** (`mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`) —
  the fully-fused, non-truncated counterpart to the truncated fused kernel
  above, porting the ENTIRE native `midpoint_exact` tiling architecture
  (BK=8, CUTLASS warp/lane geometry, 4-stage pipeline, nostride addressing,
  snake traversal, the `i==4` interleaved checkpoint, bounds-check-free exact
  fast path) unchanged. The only change: `load_global_advance`'s A-side load
  reads two raw `ushort4` (`wh`,`wr`) straight from global memory and decodes
  them via `fp32_16_combine` into a register-held `float4` -- mechanically
  identical in shape to a single `float4` read in the un-decoded native
  kernel -- and only afterward does the separate `store_to_smem` lambda write
  that already-decoded value into shared memory. So this is decode-in-
  registers-then-store, not a direct-to-shared-memory decode; there is still
  no separate scratch buffer or elementwise decode kernel either way. Unlike
  its truncated sibling, this one compiles clean (0 spill), so its
  head-to-head against the workspace/materialize strategy isolates whether
  fusing decode into a register-pressure-sensitive mainloop beats paying
  extra global-memory bandwidth for a wholly separate decode pass.

*fp32_bf16 (BF16-family) 32-bit-compute variants:*

*fp32_16 (FP16-family) 32-bit-compute variants -- additional entries, same
lineage-predecessor/successor framing as the existing "Native (unpacked)
fp32 GEMM lineage" bullets: each states what changed vs its own nearest
sibling. None of these are called by `benchmarks_gemm_fp32.cu`'s 49 live
legs -- they are pre-`dualbuf`-promotion and post-`dualbuf` dead-lineage
kernels still present in source, per the same "list every method ever
implemented" mandate this document already applies to the native fp32
lineage:*

- **TOMOS FP32 / FP16 (accurate, untiled)** (`mm_fp32_16_accurate`) — the
  earliest kernel in the fp32_16 GEMM lineage: one thread per output element
  C(i,j), decoding A's `wh`/`wr` via `fp32_16_combine` fresh on every use (no
  shared-memory reuse at all), mirroring this codebase's existing
  `mm_csr_csr` untiled style rather than a tiled/tensor-core design. Every
  other kernel in this family falls back to it for transposed operands. Its
  own comment attributes it being ~15x further behind cuBLAS than the
  plain-fp32 naive kernel specifically to this zero-reuse decode pattern.
- **TOMOS FP32 / FP16 (tiled)** (`mm_fp32_16_tiled`) — introduces
  128x128/BK=8 shared-memory tiling with 8x8 register-blocked accumulation
  per thread (same structure as `gemm_fp32_tiled_kernel`), decoding each
  `wh`/`wr` pair via `fp32_16_combine` exactly once when staged into shared
  memory rather than once per use — eliminating both the redundant global
  traffic and redundant decode work `mm_fp32_16_accurate_kernel` paid for
  every output column sharing the same A row.
- **TOMOS FP32 / FP16 (tiled, vec)** (`mm_fp32_16_tiled_vec`) — same tiling
  as the row above, adds two vectorizations: A's `wh`/`wr` read as one
  `ushort4` per 4 contiguous rows (decoded via `fp32_16_combine`, then
  scattered into shared memory — the decode step means this can't be a
  cp.async transfer, unlike a plain copy), and B read as `float4` per
  column, plus `float4` shared-memory reads in the compute loop. Requires
  `ld_a`, `m`, `ld_b`, `k` all multiples of 4; falls back to `mm_fp32_16_
  tiled` otherwise.
- **TOMOS FP32 / FP16 (cpasync, vec)** (`mm_fp32_16_tiled_cpasync_vec`) —
  same tiling/vectorization as `tiled_vec`, but A's raw `wh`/`wr` bytes are
  now prefetched one tile ahead via `cp.async` into a small staging buffer
  instead of read synchronously right before decode. Since `fp32_16_combine`
  can't run inside a `cp.async` transfer (a raw byte copy, no compute), the
  decode itself still runs synchronously from the staged bytes — what's
  hidden is only the global-memory fetch latency, not the decode. B's load
  is unchanged (still synchronous; nothing to hide there).
- **TOMOS FP32 / FP16 (cpasync, vec2)** (`mm_fp32_16_tiled_cpasync_vec2`) —
  doubles BK from 8 to 16 versus `cpasync_vec`, the same "wider B
  coalescing" lever that helped the native fp32 lineage's own BK=8->16
  step: B's `float4` load groups BK/4 threads per column, so doubling BK
  doubles that group's byte span before the address jumps a full `ld_b`
  stride. B goes back to a single (non-double-buffered) shared buffer here
  — its load was never cp.async-prefetched, so double-buffering it never
  helped, and at BK=16 double-buffering everything would land at exactly
  48KB (the static shared-memory limit) with zero margin; dropping B to one
  buffer brings it to 40KB.
- **TOMOS FP32 / FP16 (cpasync, vec2 branchless)**
  (`mm_fp32_16_tiled_cpasync_vec2_branchless`) — byte-for-byte identical to
  `vec2` except A's decode calls `fp32_16_combine_branchless` instead of
  `fp32_16_combine`: this computes the EXACT same result (all three
  normal/extended/saturate reconstruction paths computed unconditionally as
  raw bit patterns, merged via arithmetic mask, no hardware conversion
  anywhere) — bit-exact, not an accuracy tradeoff — isolating whether
  removing per-element decode branch divergence is worth anything given
  decode only runs `BM*BK/kNumThreads` times per tile (not in the per-`kk`
  inner loop), so it's already a small fraction of the kernel's time.
- **TOMOS FP32 / FP16 (cpasync, vec3)** (`mm_fp32_16_tiled_cpasync_vec3`) —
  a qualitatively different lever than vec2/vec4: enlarges the BLOCK tile
  itself to 512 threads/block (fewer, bigger blocks — same 16-warp
  occupancy as two of vec2's 256-thread blocks combined, so no per-warp
  occupancy regression), at the cost of shrinking the grid (fewer
  concurrent blocks across the GPU, worst at small N).
- **TOMOS FP32 / FP16 (cpasync, vec4)** (`mm_fp32_16_tiled_cpasync_vec4`) —
  keeps vec2's exact block/grid size (same shared-memory footprint, same
  block count) but reshapes the THREAD tile instead: TN drops from 8 to 4,
  doubling BN/TN (how many threads share one decoded A value) from 16 to
  32 — the same decode-reuse goal vec3 chased via a bigger block, achieved
  here via a narrower per-thread output slice. 512 threads/block with half
  the accumulators each (TM*TN=32 vs 64); A and B loads become single-pass
  (vs vec2/vec3's two-pass) since `(BM/4)*BK` and `BN*(BK/4)` now divide
  evenly into exactly one group per thread.
- **TOMOS FP32 / FP16 (cpasync, warptile)**
  (`mm_fp32_16_tiled_cpasync_warptile`) — a controlled test of cutlass-style
  warp tiling, isolated from every other variable: identical block tile,
  thread count, and A/B load logic to vec2. The only change is how the 256
  threads map to output positions — vec2's flat 16x16 (tx,ty) grid (where a
  warp spans two different `ty` values, splitting the warp's As row-slice
  in half) becomes an explicit 4x2-warp x 4x8-lane grid, giving each warp a
  contiguous, warp-aligned 32x64 output region, the structural property
  cutlass's warp-tile hierarchy is built around.
- **TOMOS FP32 / FP16 (cpasync, warptile2)**
  (`mm_fp32_16_tiled_cpasync_warptile2`) — identical to `warptile` except
  the warp/lane grid's shape: 2x4 warps x 8x4 lanes (64x32 per warp) instead
  of 4x2 x 4x8 (32x64 per warp) — same total resources, transposed aspect
  ratio, isolating whether warptile's result came from warp-aligned
  addressing itself (shape-independent) or this specific shape's
  interaction with the tile's access pattern.
- **TOMOS FP32 / FP16 (cpasync, warptile3)**
  (`mm_fp32_16_tiled_cpasync_warptile3`) — continues the warptile->warptile2
  trend (taller won) to its extreme: 1x8 warps x 16x2 lanes, giving each
  warp a 128x16 tile spanning the FULL BM=128 — as tall and narrow as this
  geometry allows, at the same total thread/accumulator/shared-memory
  budget as warptile/warptile2.
- **TOMOS FP32 / FP16 (cpasync, dualbuf2)**
  (`mm_fp32_16_tiled_cpasync_dualbuf2`) — same dual-cp.async structure as
  the promoted `mm_fp32_16_tiled_cpasync_dualbuf_kernel`, BK doubled from 16
  to 32 (the same wider-B-coalescing lever that helped 8->16 earlier).
  Doubling BK roughly doubles every shared-memory region (staging + As +
  stagingB + Bs), landing at ~128KB/block — only 1 such block fits per SM
  regardless of register count, an occupancy ceiling `__launch_bounds__`
  can't fix since it's shared-memory-, not register-, bound. Measured
  anyway rather than assumed, per this file's stated experimental
  philosophy.
- **TOMOS FP32 / FP16 (cpasync, tripipe)**
  (`mm_fp32_16_tiled_cpasync_tripipe`) — a genuine 3-stage software
  pipeline: tile i+2's cp.async is issued while tile i+1 is mid-transform
  and tile i is computing, so all three tiles' worth of staging AND
  transformed buffers (both A and B) must stay valid simultaneously, unlike
  dualbuf's 2-stage design. Tripling every buffer at dualbuf's BM=128 would
  hit the same ~96KB/block occupancy wall that sank dualbuf2, so BM halves
  to 64 instead — meaning this measures "smaller tile + deeper pipeline"
  together, not pipelining in isolation, since BM=64 also reduces
  arithmetic intensity per tile load and changes the grid shape.
- **TOMOS FP32 / FP16 (cpasync, dualbuf3)**
  (`mm_fp32_16_tiled_cpasync_dualbuf3`) — the inverse of dualbuf2: shrinks
  BK back to 8 (from dualbuf's 16) at dualbuf's SAME BM=BN=128 tile, cleanly
  isolating BK alone (unlike tripipe, which confounded a smaller tile with
  a deeper pipeline). Every buffer scales down with BK to ~32KB/block,
  comfortably inside the static 48KB limit with no dynamic-shared-memory
  workaround needed. A structural side effect: at BK=8, `(BM/4)*BK` and
  `BN*(BK/4)` both equal `kNumThreads` exactly, so (unlike dualbuf's
  two-pass A/B loads) both loads become single-pass here.
- **TOMOS FP32 / FP16 (cpasync, dualbuf4)**
  (`mm_fp32_16_tiled_cpasync_dualbuf4`) — motivated by dualbuf's efficiency
  vs cuBLAS climbing steadily with N (45% at N=1024 to ~90% at N=32768), the
  classic signature of grid-size/wave-quantization loss rather than
  per-block inefficiency: dualbuf's 128x128 tile produces only an 8x8=64
  block grid at N=1024, leaving many SMs idle on a 108-SM GPU. Keeps
  dualbuf's exact machinery (real cp.async for both A and B, BN=128/BK=16)
  and shrinks ONLY BM (128->64), doubling the row-tile count without
  touching the tile-shape/BK/pipeline-depth axes dualbuf2/dualbuf3/tripipe
  already varied — isolating the grid-fill hypothesis tripipe's BM=64
  confounded with pipeline depth. A dispatcher choosing between dualbuf and
  dualbuf4 by a simple (grid-blocks-at-128x128) vs (SM-count) threshold was
  built and then DROPPED after measuring a real A100: there, dualbuf4 loses
  at N=1024 but wins at N=2048/4096 before losing again at N=8192+, a
  non-monotonic crossover the single-variable heuristic gets backwards on.
  Both kernels are kept as directly-callable options with no dispatcher.
- **TOMOS FP32 / FP16 (cpasync, dualbuf5)**
  (`mm_fp32_16_tiled_cpasync_dualbuf5`) — the mirror of dualbuf4: shrinks BN
  (128->64) instead of BM, keeping BM=128/BK=16 and the same real
  dual-cp.async machinery for both A and B. Tests whether dualbuf4's
  mid-range-N win is a general small-tile effect that should appear
  symmetrically on the N axis, or specific to the M-axis shrink — since B
  already carries dualbuf's extra staging+transpose step that A doesn't
  have, shrinking the N-driving dimension (BN) could behave differently
  than shrinking the M-driving one (BM).
- **TOMOS FP32 / FP16 (cpasync, dualbuf6)**
  (`mm_fp32_16_tiled_cpasync_dualbuf6`) — same tile shape and
  double-buffering as dualbuf (BM=BN=128, BK=16); the only change is INSIDE
  the inner k-loop: every earlier kernel loads `a_reg`/`b_reg` from shared
  memory for tile `kk` and consumes them immediately, leaving reordering to
  the compiler within the fully-unrolled BK=16 loop. This kernel explicitly
  software-pipelines that load instead — prefetching `kk+1`'s registers
  while computing on `kk`'s — doubling `a_reg`/`b_reg`'s register footprint
  (16 registers -> 32) to hold both the in-flight and in-use sets.
- **TOMOS FP32 / FP16 (splitk)** (`mm_fp32_16_tiled_cpasync_splitk`) —
  Split-K: reuses dualbuf's exact 128x128/BK=16 tile and warp/lane shape
  (the most efficient shape in this series) but, instead of one block
  reducing the full K range per output tile, P blocks each reduce a K/P
  slice and `atomicAdd` their partial contribution into C. This targets the
  same too-few-output-tiles problem dualbuf4/5 fought, but WITHOUT trading
  away per-block efficiency the way shrinking the output tile does — it
  keeps the efficient tile and multiplies the block count by dividing K
  instead. A grid-stride/persistent-kernel alternative was considered and
  rejected: it only amortizes launch overhead when there are MORE tiles
  than the GPU can hold at once (already the fine, large-N case), not the
  too-few-tiles small/mid-N regime this targets. Correctness requires
  `beta*C` applied exactly once before any slice's atomicAdd, handled by a
  separate `mm_fp32_16_splitk_prescale_kernel` that runs first on the same
  default stream (so its writes are guaranteed complete before the split-K
  kernel launches); each slice's own boundary check uses its LOCAL
  `k_end`, not the global `k`, so slices never double-count a K range. A
  separately-noted correctness fix: `k_slice` must be rounded up to a full
  BK-tile multiple, since an unrounded slice boundary isn't guaranteed
  aligned to A's stride the way `ld_a`-scaled offsets are, and a misaligned
  boundary was confirmed (via compute-sanitizer) to corrupt the cp.async
  transfer.
- **TOMOS FP32 / FP16 (cpasync, dualbuf7)**
  (`mm_fp32_16_tiled_cpasync_dualbuf7`) — same tile/warp shape and real
  cp.async decode path for A as dualbuf, but B REVERTS to a synchronous
  one-step global-read-then-transpose-write into shared memory (matching
  the plain-fp32 `gemm_fp32_tiled_cpasync_vec3_kernel`'s B path exactly),
  instead of dualbuf's two-step cp.async-into-staging-then-transform.
  Motivated by `ncu` profiling on an A100 showing dualbuf is compute-bound
  (86% SM/FMA throughput, only 20% DRAM throughput) with no meaningful
  memory latency left to hide — and by the plain-fp32 `vec3` kernel (no
  decode step at all, and also synchronous-B) still beating dualbuf's
  relative-to-cuBLAS efficiency despite dualbuf's async B and extra
  warp-tiling. This isolates whether B's staging round-trip is now pure
  overhead in dualbuf's compute-bound regime, changing only B's load
  strategy.
- **TOMOS FP32 / FP16 (fast, untiled)** (`mm_fp32_16_fast`, plain
  `tomos::cuda::experimental` scope) — a naive one-thread-per-output-element
  kernel like `mm_fp32_16_accurate_kernel`, but decodes A from `wh` ALONE
  (never reads `wr`) via `fp32_16_half_only_branchless_no_overflow_nomask`:
  a branchless decode that computes both the cheap (normal/overflow) and
  shift-based-underflow-reconstruction paths unconditionally and blends via
  a bitmask, under a caller-guaranteed no-overflow contract, and additionally
  drops a provably-redundant post-shift mask (confirmed via exhaustive
  host-CPU verification and `cuobjdump`, which showed ptxas does NOT
  eliminate that mask on its own -- one real `LOP3.LUT` instruction per
  element that never changes the result). The resulting `__half` is promoted
  to `double` for the per-element accumulator and stored back as `float` C
  -- so despite the single-word, half-precision-sourced decode, this is
  still a real-fp32-output ("32-bit-compute-tier" in this document's sense)
  kernel, distinct from both the full `wh`+`wr` `fp32_16_combine` path above
  and the `__half`-throughout `pure_half` family (out of this document's
  scope).
- **TOMOS FP32 / FP16 (fast, untiled, conditional)** (`mm_fp32_16_fast`,
  `tomos::cuda::experimental::conditional` / `tomos::cuda::conditional`
  scope) — byte-for-byte the same kernel shape (same naive one-thread-per-
  element loop, same `wh`-only decode-then-promote-to-double-then-store-
  float structure) as the row above, but calls `fp32_16_half_only` instead
  of the branchless no-overflow variant: this decode function branches
  explicitly on both `control` and `direction`, correctly distinguishing and
  handling overflow (saturate to signed Infinity) from underflow (genuine
  shift-based fp16-subnormal reconstruction) rather than assuming overflow
  never occurs. This is a genuinely different decode call in a genuinely
  different (non-anonymous) namespace, not a namespace copy of an identical
  body -- confirmed by diffing the two kernel bodies directly (only the
  named decode function call differs).

*fp32_bf16 (BF16-family) 32-bit-compute variants -- additional entries:*
- **TOMOS FP32 / BF16** (`mm_fp32_bf16_tiled_cpasync_dualbuf`) — an explicit
  port of `mm_fp32_16_tiled_cpasync_dualbuf_kernel` (per its own comment):
  same 128x128/BK=16 tile, same warp/lane geometry, same double-buffered
  cp.async pipeline for both operands, every index function/constant/
  priming/mainloop/epilogue carried over verbatim; only the decode call
  (`fp32_bf16_combine` in place of `fp32_16_combine`) and symbol names
  differ. Its real body lives in `src/kernels/cuda/matrix/fp32_bf16_cuda_
  kernels.cu` (not `fp32_16_cuda_kernels.cu`, which only forward-declares
  it). Decode (via `fp32_bf16_combine`, which per this file's own verified
  decode-function facts has no branch at all -- bf16's exponent field
  matches fp32's own bit-for-bit, so decode is a direct bit concatenation
  `(wh<<16)|wr`) is confirmed to run after `__pipeline_wait_prior(0)` and
  before `__syncthreads()`/the next tile's compute -- on the critical path,
  not overlapped with any transfer, same as the FP16 dualbuf kernel's own
  documented behavior. This is the predecessor the promoted "TOMOS FP32 /
  BF16 (truncated)" fused kernel is compared against to see whether that
  kernel's tiling improvements (CUTLASS geometry, BK=8, 4-stage pipeline,
  snake traversal, interleaved checkpoint) carry over to the bf16 format the
  same way they did for fp32_16.
- **TOMOS FP32 / BF16 (rounded)** (`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded`) —
  same fused `syncpipe4`/`bk8`/`warptile`/`nostride`/`snake`/`midpoint_exact`
  architecture as the promoted (truncated-encoding) fused bf16 kernel, but
  decoding via `fp32_bf16_combine_rounded` instead of the plain
  `fp32_bf16_combine`: `wh` here was formed by ROUNDING (round to nearest,
  ties to even) rather than truncating, with the round decision recorded in
  `wr`'s own LSB so decode can invert it -- not bit-exact (loses the original
  mantissa's own LSB) but accurate to within one ULP of the full 23-bit fp32
  mantissa, versus the plain kernel's bit-exact truncating reconstruction.
  This decode function has a real measured register spill (REG:128 STACK:56,
  23 STL + 11 LDL) that the plain (non-rounded) fused kernel's clean 0-spill
  compile doesn't pay, and this timing exists to find out whether that costs
  it the win.
- **TOMOS FP32 / BF16 (rounded, adaptive block-wide)**
  (`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive`) —
  same fused rounded kernel as above, but the tile-load step first checks --
  once per BK=8 tile, via a block-wide `__syncthreads_or`-style reduction --
  whether the tile has any rounded element at all, falling back to the plain
  0-spill decode when it doesn't. Measured (per the benchmark's own cuobjdump
  note): REG:128 STACK:16, 3 STL + 2 LDL -- a real spill reduction from the
  plain rounded kernel's REG:128 STACK:56, but NOT a full elimination, and
  this adds 4 genuinely new `BAR.RED.OR` (real block-wide barrier)
  instructions on top of the pipeline's existing syncs, which could plausibly
  cost more than the reduced spill saves by undermining the pipeline's own
  latency-hiding.
- **TOMOS FP32 / BF16 (rounded, adaptive per-warp)**
  (`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp`) —
  the same idea as the block-wide adaptive kernel, but the per-tile
  "needs correction" decision is made per-WARP via `__any_sync` instead of
  per-BLOCK via `__syncthreads_or`. `__any_sync` compiles to `VOTE.ANY` (a
  cheap warp-only predicate combine, not a barrier instruction), so unlike
  the block-wide kernel's 4 new `BAR.RED.OR` instructions, this adds zero new
  barrier-class instructions. Measured: REG:128 STACK:32, 6 STL + 6 LDL -- a
  smaller spill reduction than the block-wide kernel's STACK:16, but at
  essentially none of its synchronization risk; this timing exists to answer
  whether that tradeoff nets out better in practice.

- **TOMOS FP32 / BF16 (accurate, untiled)** (`mm_fp32_bf16_accurate`) — the
  bf16-family counterpart to `mm_fp32_16_accurate_kernel`: one thread per
  output element C(i,j), no shared-memory tiling, decoding A via
  `fp32_bf16_combine` (which per this file's own header comment is always
  an EXACT reconstruction -- unlike `fp32_16_combine`, bf16's exponent field
  never needs a range-restricted degraded/approximate fallback branch).
  Lives in `fp32_bf16_cuda_kernels.cu`, structurally mirroring the
  fp32_16 accurate/tiled pair "exactly (same tile shape, same thread
  mapping, same accumulation precision)" per that file's own top-of-file
  comment -- only the decode call differs. Serves as the always-correct
  fallback for transposed operands or shapes `mm_fp32_bf16_tiled`'s fast
  path doesn't cover.
- **TOMOS FP32 / BF16 (tiled)** (`mm_fp32_bf16_tiled`) — the bf16-family
  counterpart to `mm_fp32_16_tiled_kernel`: same 128x128/BK=8,
  8x8-register-blocked shared-memory tiling, same thread mapping, decoding
  each `wh`/`wr` pair via `fp32_bf16_combine` exactly once when staged into
  shared memory (same reuse argument as the fp32_16 tiled kernel's own
  comment). NO_TRANSPOSE x NO_TRANSPOSE only; transposed operands fall back
  to `mm_fp32_bf16_accurate`. Lives in `fp32_bf16_cuda_kernels.cu`.
- **TOMOS FP32 / BF16 (accurate, rounded)** (`mm_fp32_bf16_accurate_rounded`)
  — a naive, always-correct-for-any-shape fallback for the ROUNDED bf16
  encoding specifically, mirroring `mm_fp32_bf16_accurate_kernel` exactly
  (one thread per output element, scalar per-element loads, same transpose
  handling) except the decode call is `fp32_bf16_combine_rounded` instead
  of `fp32_bf16_combine`. This closes a real correctness gap the fp32_16
  family doesn't have: unlike `fp32_16_combine`, which safely decodes BOTH
  its own rounded output and the truncated variant's (since `round_flag=0`
  makes its correction a no-op), `fp32_bf16`'s two encodings are mutually
  INCOMPATIBLE in both directions -- plain `fp32_bf16_combine` silently
  produces wrong results on rounded-encoded data whenever `round_flag=1`.
  So `mm_fp32_bf16_accurate` (plain-decode) is NOT a safe transpose/
  misaligned fallback for anything downstream of the rounded encoding, even
  though the analogous plain fallback IS safe on the fp32_16 side. This
  kernel is the correctly-decoding replacement, used as the transpose/
  misaligned/malloc-failure fallback for both the already-documented
  `mm_fp32_bf16_gemm_materialize_rounded` (workspace) and
  `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded`
  (fused) kernels.

*Native (unpacked) fp32 GEMM lineage -- the full chain of isolated,
single-lever experiments that produced the promoted `gemm_fp32_tiled_
syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` kernel (see this
document's promoted-subset entry above for the fully decoded suffix
glossary: `syncpipe4`, `bk8`, `warptile`, `nostride`, `snake`, `midpoint`/
`midpoint_exact`). All of the following live in `namespace tomos::cuda::
experimental` in `mm_cuda_kernels.cu`, and each entry below states what
changed versus its own immediate predecessor in the lineage, per that
kernel's own comment in the source and/or the benchmark driver's own
detailed head-to-head comparison comments (`benchmarks_gemm_fp32.cu`,
roughly lines 1442-2165), which this codebase treats as an integral,
verified part of the experiment record rather than mere prose:*

- **TOMOS FP32 (baseline tiled)** (`gemm_fp32_tiled`) — the true starting
  point of the entire native fp32 lineage, predating even the
  cp.async-double-buffered kernels every other entry in this section builds
  on. Register-blocked, shared-memory-tiled GEMM: each block computes a
  128x128 tile of C (BM=BN=128) over BK=8 K-slices cached in shared memory
  (As/Bs), with each of the block's 256 threads (16x16, one per 8x8 output
  sub-tile) accumulating TM x TN = 8x8 output elements in registers before
  writing back -- adapted (per its own comment) from the row-major
  derivation in Simon Boehm's "How to Optimize a CUDA Matmul Kernel" to this
  codebase's column-major storage. No double buffering, no cp.async, no
  vectorization. Its own comment frames this single jump from the naive
  one-thread-per-output-element `gemm_generic_kernel` (zero cross-thread
  reuse of A/B loads) to this shared-memory + register tiling as where most
  of the throughput gap to cuBLAS closes. Falls back to `gemm_generic_kernel`
  for any transposed input (transpose not implemented for the tiled fast
  path).
- **TOMOS FP32 (double-buffered)** (`gemm_fp32_tiled_db`) — same 128x128/
  BK=8 tiling as the baseline kernel, but manually double-buffered: the
  next K-slice's global-memory loads are issued (into the *other* shared
  buffer) before the current slice's register-blocked accumulation runs,
  instead of after, aiming to overlap load latency with FMA work rather than
  sit behind a `__syncthreads()` waiting on it. Both buffers together are
  still only ~16KB of static shared memory, well under the 48KB default
  limit. Its own comment documents a real measured gotcha: without
  `__launch_bounds__(256, 2)`, the extra prefetch-related live state pushes
  ptxas from 126 to 132 registers/thread versus the baseline kernel -- just
  enough to drop occupancy from 2 resident blocks/SM to 1, which more than
  erases the prefetch's benefit and makes the double-buffered kernel
  *slower* than the single-buffered baseline it was meant to improve on;
  forcing 2 blocks/SM via the launch bound keeps register usage at parity
  with baseline so the prefetch is a net win instead.
- **TOMOS FP32 (cp.async)** (`gemm_fp32_tiled_cpasync`) — same 128x128/BK=8
  double-buffered tiling as `gemm_fp32_tiled_db`, but the prefetch itself
  goes through sm_80's cp.async (`__pipeline_memcpy_async`) instead of a
  plain load-into-register-then-store-to-shared round trip. Its own comment
  explains the motivation directly: `gemm_fp32_tiled_db` (once fixed to not
  regress occupancy) came out roughly even with the single-buffered
  baseline, because a *manual* double buffer still round-trips every
  prefetched value through a register before it lands in shared memory, so
  it doesn't free the SM to do anything else during the load's latency --
  with occupancy already at 2 blocks/SM, the warp scheduler was already
  hiding most of that latency by switching warps, leaving little for manual
  prefetching to add. cp.async is genuinely asynchronous (no register in the
  critical path), so the SM can keep retiring the *current* tile's FMAs
  while the *next* tile's bytes are still in flight;
  `__pipeline_wait_prior(0)`/`__syncthreads()` only block once that overlap
  has had a chance to pay for itself. Also introduces the zero-fill-then-
  async-copy boundary handling (`__pipeline_memcpy_async` has no notion of
  "copy nothing" for an out-of-range tail) that every later cp.async kernel
  in this lineage reuses unchanged.
- **TOMOS FP32 (cp.async, vec)** (`gemm_fp32_tiled_cpasync_vec`) — same
  double-buffered cp.async structure as `gemm_fp32_tiled_cpasync`, plus two
  independent vectorizations: (1) the compute loop's shared-memory reads
  (`As`/`Bs`) become two `float4` reads instead of eight scalar ones,
  always safe since both are the kernel's own 16-byte-aligned shared arrays
  with TM=TN=8; (2) A's global-to-shared cp.async load becomes a single
  16-byte transfer of 4 consecutive rows instead of 4 separate 4-byte
  transfers, safe because A's naturally-contiguous direction (rows,
  column-major storage) is the same direction the compute loop already
  reads it in. B's global load deliberately stays scalar -- vectorizing it
  under the current `Bs[BK][BN]` (N-fast) layout would need a transposed
  layout, left for a later kernel rather than bundled in here. Requires
  `ld_a` and `m` to both be multiples of 4 (for row-group alignment); its
  host dispatch checks that and falls back to `gemm_fp32_tiled_cpasync`
  otherwise, so the kernel itself never needs its own scalar fallback path.
- **TOMOS FP32 (cp.async, vec2)** (`gemm_fp32_tiled_cpasync_vec2`) — same
  as `_vec` above, plus vectorizing B's global load too -- the piece `_vec`
  deliberately left scalar. B can't use cp.async the way A does: cp.async
  copies N contiguous bytes from a contiguous global range straight into a
  contiguous shared range, but `Bs[BK][BN]`'s N-fast layout makes 4
  consecutive K-values for one column land in 4 *non-adjacent* shared slots,
  so there's no single contiguous shared destination for cp.async to
  target. The workaround: a plain synchronous 16-byte global read into a
  `float4` register (one instruction instead of four, still fully
  coalesced), then scattering its four lanes into their non-adjacent shared
  slots with four ordinary stores -- loses cp.async's latency-hiding for B
  specifically (the read blocks before the scatter can happen), but still
  cuts B's global load instruction count 4x, relying on the same 2-blocks/SM
  warp-switching occupancy that was already hiding most other latency.
  Requires `ld_b` and `k` to both be multiples of 4, mirroring `m`/`ld_a`'s
  role for A; falls back down the dispatch chain otherwise.
- **TOMOS FP32 (cp.async, vec3)** (`gemm_fp32_tiled_cpasync_vec3`) — same
  structure as `_vec2`, with only the K-slice depth changed: BK doubled from
  8 to 16, under its own distinct tile constants (`kV3BM`/`kV3BN`/`kV3BK`/
  `kV3TM`/`kV3TN`) so nothing about the earlier kernels' compiled behavior
  changes. Specifically targets B's coalescing, not just "fewer, bigger
  steps": B's vectorized load groups `BK/4` threads onto the same column
  before the address jumps a full `ld_b` stride to the next column -- at
  BK=8 that's only 2 threads sharing a 32-byte span per column, while at
  BK=16 it's 4 threads sharing a 64-byte span, a wider coalesced chunk. Costs
  double the shared memory per buffer (32KB total for both buffers), still
  comfortably under the 48KB static limit.
- **TOMOS FP32 (cp.async, dualbuf)** (`gemm_fp32_tiled_cpasync_dualbuf`) —
  the native (unpacked) fp32 sibling of `mm_fp32_bf16_tiled_cpasync_dualbuf`/
  `mm_fp32_16_tiled_cpasync_dualbuf` (per its own comment): same 128x128/
  BK=16 tile, same 2x4-warp x 8x4-lane geometry, same double-buffered
  cp.async pipeline for BOTH operands. `_vec3` (this sub-lineage's previous
  best) only routes A through cp.async -- B stays a synchronous `float4`
  load every tile iteration. This kernel closes that gap: B is staged via
  cp.async into a K-fast buffer matching its own global layout, then
  transposed into the N-fast compute layout via a shared-to-shared scatter
  (cp.async can't scatter into a different layout directly), mirroring
  `mm_fp32_bf16_tiled_cpasync_dualbuf`'s own B handling exactly. A needs no
  staging/transform step at all here (unlike the packed kernels' `wh`/`wr`
  decode), since it's already a plain float -- cp.async writes it directly
  into its final compute buffer, same as `_vec3` already does. Its own
  comment reports this measurably closes most of the gap that made the
  packed fp32_bf16/fp32_16 "full reconstruction" kernels outperform `_vec3`
  despite doing strictly more per-element work -- confirming that gap was a
  pipelining-maturity difference between kernel families, not something
  intrinsic to packing. This is the direct, explicitly-named predecessor of
  the already-documented `gemm_fp32_tiled_cpasync_dualbuf_regpipe` ("a
  register-level software-pipelined sibling of the original cp.async
  double-buffered dualbuf kernel"). Per this document's own "Note on method
  count," it is the one native kernel in this lineage NOT called anywhere
  live in the current benchmark driver -- only referenced inside a
  commented-out `ms_native` reassignment.
- **TOMOS FP32 (regpipe)** (`gemm_fp32_tiled_cpasync_dualbuf_regpipe`) — a
  register-level software-pipelined sibling of the original cp.async
  double-buffered dualbuf kernel (the tile-level-double-buffered baseline
  this whole lineage started from), motivated by an ncu/SASS comparison
  against real CUTLASS that this kernel's own source comment documents in
  full; the benchmark times it head-to-head against the promoted kernel to
  see whether that register-level interleaving fix actually helped.
- **TOMOS FP32 (syncpipe, 2-stage)** (`gemm_fp32_tiled_dualbuf_syncpipe`) —
  a no-cp.async pipeline, modeled directly on real CUTLASS's own
  `MmaPipelined::gemm_iters` (`mma_pipelined.h`), using plain synchronous
  global loads with one tile of independent lookahead instead of cp.async
  double-buffering.
- **TOMOS FP32 (syncpipe, incptr)** (`gemm_fp32_tiled_dualbuf_syncpipe_incptr`) —
  the rolling-pointer addressing technique later generalized as
  `syncpipe3_incptr` (tracking a small number of live pointers/counters
  instead of recomputing addresses from scratch every iteration), applied
  here to the simpler 2-stage syncpipe kernel instead, to test whether the
  shallower pipeline avoids the register spilling the 3-stage version hits.
- **TOMOS FP32 (syncpipe3)** (`gemm_fp32_tiled_syncpipe3`) — a genuine 3-stage
  extension of `dualbuf_syncpipe`: instead of fetching tile i+1 while
  consuming tile i (one tile of lookahead), this fetches tile i+2 while
  consuming tile i (two tiles of lookahead), at the cost of a third
  shared-memory buffer slot -- still no cp.async, no separate staging
  buffer, same underlying synchronous-load mechanism as syncpipe, just one
  stage deeper. The prologue primes two tiles (stages 0 and 1) before the
  mainloop starts. This is the base every `syncpipe3_*` variant below
  modifies one lever at a time, and (per the file's own header comment) the
  one technique in the whole `syncpipe3_incptr*` sub-investigation with a
  confirmed positive measured result (0.951x vs CUTLASS at N=8192, beating
  the earlier dualbuf kernel's 0.933x).
- **TOMOS FP32 (syncpipe3, incptr)** (`gemm_fp32_tiled_syncpipe3_incptr`) —
  rolling-pointer addressing (fewer repeated multiplies, off the FMA-shared
  IMAD path) instead of syncpipe3's own repeated address recomputation.
  Introduces real register spilling that the head-to-head measurement shows
  regressed versus plain syncpipe3 -- the first of several `incptr_*`
  attempts in this file to try to recover that loss.
- **TOMOS FP32 (syncpipe3, incptr_lean)** (`gemm_fp32_tiled_syncpipe3_incptr_lean`) —
  trims incptr's redundant live register state (4 K-position counters
  collapsed to 1 shared counter, 4 tracked pointers collapsed to 2),
  reducing the stack frame from 88B to 56B, though still nonzero; the
  benchmark's own comment frames this as testing whether that reduction is
  enough to matter at runtime.
- **TOMOS FP32 (syncpipe3, incptr_lean2)** (`gemm_fp32_tiled_syncpipe3_incptr_lean2`) —
  combines predicate-collapsed bounds checks with a 32-bit (not `size_t`)
  remaining-K counter, stacked on top of lean. Static analysis showed this
  actually spills MORE than lean (64B/84B/108B stack/stores/loads vs lean's
  56B/60B/100B); the timing exists to confirm whether that static regression
  shows up at runtime too.
- **TOMOS FP32 (syncpipe3, incptr_nostride)** (`gemm_fp32_tiled_syncpipe3_incptr_nostride`) —
  drops precomputed stride locals entirely, inlining the multiplication at
  each use site (the same "nostride" technique later reused, unmodified in
  meaning, by the promoted `warptile_nostride` kernel -- see this document's
  promoted-subset entry for the full verified explanation of why this is a
  register-pressure trick, not a memory-access-pattern change). Static
  analysis showed this is the best spill result of the whole incptr family
  (48B/48B/88B vs lean's 56B/60B/100B and lean2's 64B/84B/108B).
- **TOMOS FP32 (syncpipe3, incptr_nostride_kremain)** (`gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain`) —
  decrementing a `k_remaining` counter instead of computing
  `tiles_fetched*BK`, added on top of nostride specifically. Static analysis
  showed a mixed/slight regression versus nostride alone (same 48B stack
  frame, but 68B spill stores vs 48B, 84B spill loads vs 88B).
- **TOMOS FP32 (syncpipe3, incptr_nostride_bcol)** (`gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol`) —
  derives `b_col[1]` on demand instead of tracking it as a separate live
  value, added on top of nostride specifically. Static analysis showed a
  regression on every spill metric versus nostride alone (56B/60B/100B vs
  48B/48B/88B).
- **TOMOS FP32 (syncpipe3, incptr_kremain)** (`gemm_fp32_tiled_syncpipe3_incptr_kremain`) —
  the same `k_remaining` change as `incptr_nostride_kremain`, but applied in
  ISOLATION on top of plain `incptr_lean`'s base structure, not stacked on
  nostride. Static analysis showed this alone regresses versus lean on every
  spill metric (64B/84B/108B vs 56B/60B/100B) -- coincidentally landing on
  the exact same numbers as lean2 (predicates+narrow-ints combined).
- **TOMOS FP32 (dualbuf_syncpipe, incptr_nostride)** (`gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride`) —
  syncpipe3_incptr_nostride's addressing (one tracked pointer per operand, no
  precomputed stride locals) applied to the SHALLOWER 2-stage dualbuf_
  syncpipe pipeline instead of 3-stage syncpipe3, to test the
  pipeline-depth-vs-register-pressure hypothesis directly. Static analysis
  showed it does NOT help: 128 registers, 48B/52B/88B stack/stores/loads,
  essentially identical to (marginally worse than) the 3-stage nostride
  kernel's 48B/48B/88B -- disproving the hypothesis that less concurrent
  pipeline state alone would get under the register ceiling; the addressing
  state itself, independent of pipeline depth, is the bottleneck.
- **TOMOS FP32 (syncpipe3, incptr_nostride_lb1)** (`gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1`) —
  byte-for-byte the same kernel as `syncpipe3_incptr_nostride`, except
  `__launch_bounds__(256, 2)` is relaxed to `__launch_bounds__(256, 1)`: the
  `(256,2)` variant hard-caps ptxas at exactly 128 registers/thread (65536
  register file / (256 threads * 2 blocks)), which is exactly where
  nostride's peak live state sits, hence its spilling; `(256,1)` lets ptxas
  use up to 255 registers/thread instead, which static analysis confirms
  ELIMINATES spilling entirely (146 registers, 0 bytes stack frame, 0/0
  spill stores/loads, matching CUTLASS's own 0/0/0 profile). The cost is
  halving max resident blocks/SM from 2 to 1, trading away the ~24.7%
  achieved occupancy every kernel in this family shared with CUTLASS at the
  2-block cap.
- **TOMOS FP32 (syncpipe3, finegrain)** (`gemm_fp32_tiled_syncpipe3_finegrain`) —
  applies two structural traits confirmed from real CUTLASS's own
  `MmaPipelined::gemm_iters` and its `MmaSimtPolicy` `GemmShape<4,4,1>` lane
  shape: (1) the 8x8 accumulator tile split into four 4x4 quadrants, each
  interleaved with one of the next tile's four `float4` register loads
  instead of loading all four upfront then running one monolithic 64-FFMA
  nest; (2) A's and B's next-next-tile global loads issued at different `kk`
  indices (`kk==0`, `kk==BK/2`) instead of bunched together. Static analysis
  found ptxas mostly undid the source-level interleaving anyway (re-clustered
  the loads, reuse ratio barely moved: 69.3% vs syncpipe3's 68.5%, both far
  off CUTLASS's 87.2%) and introduced a small new spill (24B/24B/24B) that
  plain syncpipe3 didn't have.
- **TOMOS FP32 (syncpipe3, dualacc)** (`gemm_fp32_tiled_syncpipe3_dualacc`) —
  byte-for-byte identical to syncpipe3 except the accumulator: each output
  element's sum is split into two independent partial accumulators
  (alternating by `kk` parity, combined only in the epilogue), breaking the
  RAW dependency chain across syncpipe3's 16-deep `kk` loop into two 8-deep
  chains. Motivated by matched-occupancy `ncu` data showing syncpipe3's
  "wait" (fixed-latency execution dependency) stall share at 15.4% of total
  vs CUTLASS's 11.6%. Confirmed cost: doubling accumulator registers (64 ->
  128 floats) produced severe spilling -- 680B stack frame, 3672B spill
  stores, 3584B spill loads -- by far the worst of the whole investigation.
- **TOMOS FP32 (syncpipe3, snake)** (`gemm_fp32_tiled_syncpipe3_snake`) —
  byte-for-byte identical to syncpipe3 except the FFMA accumulate loop's own
  traversal order: a boustrophedon sweep (even rows sweep `j` forward, odd
  rows sweep it backward) instead of plain row-major, targeting the measured
  FFMA `.reuse` ratio gap (syncpipe3: 68.5%, CUTLASS: 87.2%) without touching
  addressing, load structure, tile size, or accumulator structure. Static
  analysis found this did NOT move the needle (69.3%, essentially unchanged)
  while introducing a real new spill (120B/208B/136B) -- this is the SAME
  boustrophedon idea the promoted kernel's own `snake` suffix later names,
  but it fails here on the BK=16 base and only succeeds later on the leaner
  BK=8/warptile/nostride base (see `syncpipe4_bk8_warptile_nostride_snake`
  below).
- **TOMOS FP32 (syncpipe3, bk8)** (`gemm_fp32_tiled_syncpipe3_bk8`) — the same
  3-stage rotation, 8x8 register tile, and warp/lane geometry as syncpipe3,
  with only the K-tile depth changed: BK=8 instead of 16, matching CUTLASS's
  own confirmed `ThreadblockShape<128,128,8>`. Requires a genuinely different
  thread-to-element mapping (one `float4` load per thread instead of two),
  confirmed by static analysis to compile clean (128 registers, 0 spill) --
  no register-pressure regression from the narrower K-tile.
- **TOMOS FP32 (syncpipe4, bk8)** (`gemm_fp32_tiled_syncpipe4_bk8`) —
  byte-for-byte identical to syncpipe3_bk8 except the pipeline depth: 4
  rotating shared-memory stages instead of 3 (three tiles of lookahead
  instead of two), affordable because BK=8 halves each stage's cost versus
  BK=16 -- mirrors exactly how the 2-stage syncpipe was earlier extended to
  3-stage syncpipe3 (the one confirmed-positive technique in this
  investigation), now applied to the bk8 base. Static analysis: 128
  registers, 0 spill -- matches bk8's own clean profile.
- **TOMOS FP32 (syncpipe4, bk8, warptile)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile`) —
  byte-for-byte identical to syncpipe4_bk8 except the warp/lane geometry used
  for the compute (read) path: CUTLASS's actual confirmed `WarpShape<32,64,8>`
  (32 rows x 64 cols per warp, vs this file's own prior 64x32) and its
  `MmaSimtPolicy<MatrixShape<4,8>, RowMajorInterleaved<2>,...>` lane layout,
  with the exact interleaved lane-to-(row,col) mapping read directly from
  CUTLASS's own `cutlass/layout/matrix.h`. The tile-fetch (load) path is
  untouched. The biggest, most structural change attempted in the whole
  investigation -- every other kernel kept the warp/lane grid fixed. Static
  analysis: 128 registers, 0 spill -- no register-pressure regression from
  the geometry change.
- **TOMOS FP32 (syncpipe4, bk8, warptile, nostride)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride`) —
  byte-for-byte identical to syncpipe4_bk8_warptile except the global-load
  addressing: applies the syncpipe3_incptr_nostride technique (one tracked
  rolling pointer per operand, no precomputed stride locals) to the BK=8
  base for the first time -- this exact combination (nostride + BK=8) was
  never tried before, since the incptr/nostride family was only ever tested
  on the BK=16 base (and lost there). Static analysis: 126 registers, 0
  spill -- the FIRST kernel in the whole investigation to compile to fewer
  than 128 registers, landing EXACTLY at CUTLASS's own register count with
  zero spilling.
- **TOMOS FP32 (…nostride, snake)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake`) —
  byte-for-byte identical to `..._warptile_nostride` except the FFMA
  accumulate loop's traversal order: the same boustrophedon sweep
  `syncpipe3_snake` tried (and failed on) the BK=16 base, retried here on
  this leaner base instead. This time it works: static analysis shows 127
  registers (1 more than the non-snake base, still 1 below the 128
  ceiling), 0 spill, and the FFMA `.reuse` ratio actually moves -- 73.4%, a
  real +6.2 percentage-point gain over the non-snake base's 67.2%, versus
  the negligible +0.8pp gain snake achieved on syncpipe3 before. This is the
  kernel whose own `snake` suffix carries the meaning documented in this
  document's promoted TOMOS FP32 entry (per-thread register-reuse traversal
  order, not a cache/tile-visitation scheme).
- **TOMOS FP32 (…snake, interleaved)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved`) —
  identical geometry and snake traversal to the kernel above, but the
  next-`kk` shared-memory prefetch and next-tile global prefetch are issued
  piecemeal, interleaved into the FFMA nest, instead of batched up front --
  mirroring the instruction-interleaving pattern found by disassembling real
  CUTLASS's own compiled SASS mainloop (LDS/LDG/address-math spread roughly
  one every 1-2 FFMAs through the chain instead of clustered at the front).
  This hits exactly the 128-register ceiling with a light spill (2 STL + 2
  LDL) -- the first spill reintroduced since `warptile_nostride`.
- **TOMOS FP32 (…interleaved_late)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late`) —
  same as `..._interleaved`, but the next-`kk`/next-tile prefetch checkpoints
  inside the row loop are placed late (`i==5/6/7`) instead of early
  (`i==2/4/6`), to shrink how long the prefetched registers must stay live
  before their real consumer (the next `kk` iteration), testing whether that
  reduces the interleaved kernel's own light spill while keeping its reuse
  gain.
- **TOMOS FP32 (…midpoint)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint`) —
  isolates whether the interleaved kernels' register/spill increase came
  from splitting `load_compute_regs` into separate A/B lambdas, or from
  moving the checkpoint into the row loop at all (interleaved and
  interleaved_late were found to compile to byte-identical SASS, showing
  source position alone doesn't matter for that split). This kernel keeps
  `load_compute_regs` combined as one call site (as in plain snake) and
  moves only that single call to one mid-loop checkpoint at `i==4`, with no
  split.
- **TOMOS FP32 (…midpoint_uncapped)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped`) —
  same as midpoint, but `__launch_bounds__(256, 2)` is relaxed to
  `__launch_bounds__(256)` (no min-blocks term), letting ptxas pick registers
  freely -- matching how real CUTLASS's own SIMT fp32 `device::Gemm` kernel
  is compiled (no `__launch_bounds__` at all; its 126-register footprint is
  organic). Tests whether the `(256, 2)` occupancy ceiling itself, not the
  prefetch loop structure, caused midpoint's own light spill. Per the
  benchmark's own comment on the later `_exact` kernel, this relaxation was
  measured "dramatically slower despite eliminating the spill" -- the same
  occupancy-halving cost `syncpipe3_incptr_nostride_lb1` already paid,
  confirming that keeping 2 blocks/SM matters more than avoiding this
  particular light spill.
- **TOMOS FP32 (…midpoint_narrow)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow`) —
  same as midpoint (same combined checkpoint at `i==4`, same
  `__launch_bounds__(256, 2)` -- deliberately NOT relaxed, since uncapped's
  relaxation regressed badly) except `tiles_fetched` is narrowed from
  `size_t` to `unsigned int`, trying to free the one register midpoint
  needed without touching the interleaved-checkpoint structure at all.
- **TOMOS FP32 (…midpoint_inline)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline`) —
  same as midpoint (same checkpoint, same launch bounds) except
  `has_next_kk`/`has_next_tile` are no longer precomputed named locals -- the
  conditions are inlined directly at the `i==4` checkpoint, removing the one
  genuinely-live-longer boolean (`has_next_tile`, only real for the `kk==0`
  unrolled instance) that midpoint's own manual hoisting had introduced.
- **TOMOS FP32 (…midpoint_exact, rerun)** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`) —
  this leg times the exact same kernel as the promoted "TOMOS FP32" row
  above (called via the same plain, non-`experimental::` entry point), a
  second time. The benchmark's own comment explains this is a leftover
  duplicate from the kernel's own promotion (removing it would require
  re-plumbing every downstream head-to-head ratio print that references it),
  now useful only as a run-to-run noise check rather than a distinct method.

*External baselines (see this document's promoted-subset entry above for
full descriptions):*

- **CUTLASS** (`cutlass::gemm::device::Gemm<...>`, SIMT fp32) — external
  reference baseline; not part of this codebase.
- **cuBLAS** (`cublasSgemm`) — external reference baseline; not part of this
  codebase.

### FP16 GEMM

#### Experiment:

The "16-bit-compute" GEMM sweep: cuBLAS, real CUTLASS, this codebase's own
native (unpacked) FP16/BF16 GEMM, and the packed-precision `fp32_16`/
`fp32_bf16` formats' 16-bit-compute tier (decode straight to a half/bf16
value and compute in that precision throughout, as opposed to the
"32-bit-compute" tier above that reconstructs a real FP32 value before
computing). `square` is the M=K=N sweep; `rect` holds M=N=44000 fixed and
sweeps the inner dimension K. This lists the COMPLETE set of methods this
benchmark times -- confirmed by `grep -c "= time_call(start, stop"
benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu`, which returns
exactly 10, one per row below.

**Scripts:**
- `scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh` — builds and runs the
  benchmark.
- `scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh` — regenerates the two
  `.png`/`.fig` plots from an already-saved CSV (no rebuild/rerun); wraps
  the MATLAB function `scripts/plot_fp16_bf16_pure_vs_cutlass_ref.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu`.

**Kernels called (all methods tried):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` | — | external library (cuBLAS) |
| cuBLAS BF16 | `cublasGemmEx` | — | external library (cuBLAS) |
| CUTLASS FP16 | `cutlass::gemm::device::Gemm<...>` (fp16 tensor-op) | — | external library (CUTLASS) |
| CUTLASS BF16 | `cutlass::gemm::device::Gemm<...>` (bf16 tensor-op) | — | external library (CUTLASS) |
| TOMOS FP16 (mma_ptx, base) | `gemm_fp16_mma_ptx_kernel` | -- (native) | first hand-written `mma.sync`+`ldmatrix` PTX kernel in the family, replacing an earlier `wmma::`-API kernel; register/address mapping reverse-engineered via a standalone delta-function test. Best-of-family at write time, 0 spill (104 regs) |
| TOMOS FP16 (bigtile) | `gemm_fp16_mma_ptx_bigtile_kernel` | -- (native) | doubles BM 128->256, BK 128->64, attacking HBM bandwidth; beats mma_ptx at every N, margin growing with N, 0 spill |
| TOMOS FP16 (bigtile, grid swizzle) | `gemm_fp16_mma_ptx_bigtile_swizzle_kernel` | -- (native) | groups blockIdx dispatch order for L2 reuse on top of bigtile; measured ~9-10% loss vs plain bigtile, not promoted |
| TOMOS FP16 (bigtile2) | `gemm_fp16_mma_ptx_bigtile2_kernel` | -- (native) | grows tile to 256x256; real register spill (1112B/1116B), worse than even plain mma_ptx; not promoted |
| TOMOS FP16 (bigtile, 3-stage pipe) | `gemm_fp16_mma_ptx_bigtile_tripipe_kernel` | -- (native) | extends cp.async prefetch depth 2->3 on bigtile's tile; measured neutral (tied with bigtile), not promoted |
| TOMOS FP16 (bigtile, lean/deep B) | `gemm_fp16_mma_ptx_bigtile_deep_kernel` | -- (native) | restructures B-fragment register lifetimes (lean, 2 live regs) instead of tile/pipeline; measured neutral, not promoted |
| TOMOS FP16 (bigtile, 32-warp) | `gemm_fp16_mma_ptx_bigtile_w32_kernel` | -- (native) | splits to 32 warps/1024 threads (occupancy lever, distinct from bandwidth/latency); decisive win, 62 regs 0 spill, margin growing with N |
| TOMOS FP16 (bigtile w32, skip-beta) | `gemm_fp16_mma_ptx_bigtile_w32_epi_kernel<SkipBeta>` | -- (native) | template-parameterized (not runtime-branched, to avoid a forced spill) skip of the beta*C read when beta==0.0; clean win at every N, new family best |
| TOMOS FP16 (bigtile w32 epi, L2 persistence) | `gemm_fp16_mma_ptx_bigtile_w32_epi_l2_kernel<SkipBeta>` | -- (native) | sets a CUDA L2 persistence window on B's column-strip per-column-tile launch; measured clear loss (~15%, per-launch overhead), also surfaced and fixed a real cross-kernel `cudaLimitPersistingL2CacheSize` state-leak bug; not promoted |
| TOMOS FP16 (bigtile2, 32-warp) | `gemm_fp16_mma_ptx_bigtile2_w32_kernel` | -- (native) | tests whether w32's occupancy win composes with bigtile2's 256x256 tile; register math unfavorable by design (64 regs = entire budget at 1024 threads); measured real spill, 5x slower than bigtile_w32; not promoted |
| TOMOS FP16 (bigtile w32 epi, wide N=160) | `gemm_fp16_mma_ptx_bigtile_w32_epi_wide_kernel<SkipBeta>` | -- (native) | grows BN only (128->160), exploiting B's lean-load asymmetry (growing FragN costs only accumulator regs, not operand cache); 64 regs, only 4B spill; beats w32_epi at every N; new family best |
| TOMOS FP16 (bigtile w32 epi, wide2 N=192) | `gemm_fp16_mma_ptx_bigtile_w32_epi_wide2_kernel<SkipBeta>` | -- (native) | next increment (BN 160->192); predicted real spill materializes (172-176B); never beats wide; not promoted, pins BN=160 as sweet spot |
| TOMOS FP16 (bigtile w32 epi, wide2b) | `gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b_kernel<SkipBeta>` | -- (native) | same 256x192 tile as wide2, different warp arrangement (16x2 vs 8x4); own comment has no separate MEASURED RESULT, later grouped with wide2 as "also a loss" in a subsequent kernel's comment; not promoted |
| TOMOS FP16 (deep5) | `gemm_fp16_mma_ptx_deep5_kernel` | -- (native) | matches cuBLAS's actual nsys-traced dispatched config (128x128/BK=32/5 stages/256 threads); consistent occupancy-bound loss (~0.47-0.50x of wide's throughput); not promoted |
| TOMOS FP16 (wide, deep5 pipeline) | `gemm_fp16_mma_ptx_wide_deep5_kernel` | -- (native) | isolates pipeline depth alone from deep5's 3-variable conflation (wide's tile, BK 64->32, stages 2->5); no improvement over wide, closes off pipeline depth as an independent lever |
| TOMOS FP16 (wide, coalesced B) | `gemm_fp16_mma_ptx_wide_coalb_kernel` | -- (native) | fixes uncoalesced B-tile global loads (swap fastest-varying index); measured regressed (per a later kernel's retrospective) |
| TOMOS FP16 (wide, coalesced B, shuffle) | `gemm_fp16_mma_ptx_wide_coalb_shfl_kernel` | -- (native) | reads B coalesced, `__shfl_sync`s into wide's original conflict-free shared layout instead of re-deriving one; also measured regressed |
| TOMOS FP16 (deep5, w32) | `gemm_fp16_mma_ptx_deep5_w32_kernel` | -- (native) | fills the untested (tile size) x (occupancy) cell: deep5's tile+stages with wide's 32-warp arrangement; no MEASURED RESULT stated in source |
| TOMOS FP16 (wide, nospill attempt 1) | `gemm_fp16_mma_ptx_wide_nospill_kernel` | -- (native) | attempts to fix wide's 4-byte register spill by collapsing `next_k0`'s live range into one boolean; measured no effect (compiler already did equivalent CSE) -- a documented negative result pinpointing the live range, not a missed optimization |
| TOMOS FP16 (wide, nospill attempt 2) | `gemm_fp16_mma_ptx_wide_nospill2_kernel` | -- (native) | removes `next_k0`'s second use entirely (unconditional `__pipeline_wait_prior(0)`); small but real, consistent win over wide at N>=8192 |
| TOMOS FP16 (wide, smem epilogue) | `gemm_fp16_mma_ptx_wide_epi_smem_kernel` | -- (native) | stages the epilogue through shared memory (reusing dead As/Bs space) for a coalesced final write instead of wide's scattered direct writes; consistent win over wide at N<=4096 |
| TOMOS FP16 (wide, nospill+epismem) | `gemm_fp16_mma_ptx_wide_nospill_epismem_kernel` | -- (native) | composes nospill2 + epi_smem (disjoint parts of the kernel); explicitly named "the best kernel from this investigation" by two later kernels' own comments |
| TOMOS FP16 (nospill_epismem, deep3) | `gemm_fp16_mma_ptx_wide_nospill_epismem_deep3_kernel` | -- (native) | tests a CUTLASS-derived 3-stage pipeline on top of nospill_epismem; real, consistent regression (-4% to -7%) -- already occupancy-capped at 1 block/SM even at 2 stages, no spare latency for deeper buffering to hide |
| TOMOS FP16 (nospill_epismem, xswz) | `gemm_fp16_mma_ptx_wide_nospill_epismem_xswz_kernel` | -- (native) | from-scratch fix of B-load coalescing via CUTLASS's real crosswise layout (4 documented attempts before landing on the correct `ldmatrix.x4.m8n8` + byte-offset-XOR formula, verified against real CUTLASS's compiled iterator); correctness confirmed via full verify suite on real GPU; throughput doesn't match nospill_epismem outright |
| TOMOS FP16 (nospill_epismem, xswz2) | `gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2_kernel` | -- (native) | same crosswise B-load fix as xswz, with a fetch/consume register fusion (FragN=10 is even here, unlike xswz's odd FragN=5); ~83-86% of nospill_epismem's speed, beats a sibling `coalb` combo kernel at every size; still doesn't match nospill_epismem |
| TOMOS FP16 (cutlass_config) | `gemm_fp16_mma_ptx_cutlass_config_kernel` | -- (native) | builds CUTLASS's real reference config from scratch (128x256x32 threadblock, 64x64x32 warp, 3 stages); switched to a 16-warp/512-thread split to fit the register ceiling; CONFIRMED win (+3-13% over the 8-warp split) from doubled occupancy despite modest new spilling |
| TOMOS FP16 (wide_bigreg, base) | `gemm_fp16_mma_ptx_wide_bigreg_kernel` | -- (native) | branch-point originator: BM=256/BN=160 (deliberately NOT matching CUTLASS's 128-float accumulator, to test whether tile/register shape matching mattered); two real in-kernel fixes measured (+5% from address hoisting despite new spill; >2x from converting B's load to genuine `__pipeline_memcpy_async`), ending above real CUTLASS's own measured throughput despite the "wrong" accumulator size |
| TOMOS FP16 (wide_bigreg_sq) | `gemm_fp16_mma_ptx_wide_bigreg_sq_kernel` | -- (native) | retunes BN 160->128, an exact match to CUTLASS's real accumulator/warp-tile size, motivated by direct SASS diffing; confirmed via measurement (0/4 byte spill vs wide_bigreg's real spilling); the SAME B-load cp.async fix applied again (~2x, above CUTLASS); plus an 18-point empirical investigation into a confirmed L2-sector overhead on A's loads, root-caused to CUTLASS's PredicatedTileAccessIterator machinery itself and fixed via a CUTLASS-style thread-to-address remap (NOT by adopting the bandwidth-costly synchronous-load workaround that was tried and reverted, -35-40%) |
| TOMOS FP16 (wide_bigreg_sq, deep4) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_kernel` | -- (native) | single-lever: 3-stage -> 4-stage pipeline; consistent win over wide_bigreg_sq at N>=4096 (+1.6% to +3.6%), 0 spill, became family's new best, edging past cuBLAS at N=32768/65536 |
| TOMOS FP16 (deep4, iter) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_kernel` | -- (native) | hoists `load_tile_async`'s per-call address arithmetic (provably-equivalent running-offset technique) after a SASS comparison found 3-4x more IMAD/LEA than CUTLASS despite identical HMMA/LDSM; UNTESTED explicitly |
| TOMOS FP16 (deep4, epivec) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec_kernel` | -- (native) | sibling of iter (built on deep4's unmodified mainloop): vectorizes the epilogue Cs<->C transfer to 8-half `float4` (register->Cs corner-turn deliberately left scalar, proven non-vectorizable without a much larger relayout stage); UNTESTED explicitly |
| TOMOS FP16 (deep4, iter+epivec) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_kernel` | -- (native) | combines both independently-measured fixes (iter: small win at large N/-3.35% loss at N=8192; epivec: large small-N win fading by N=16384+) to test additive composition; UNTESTED itself |
| TOMOS FP16 (iter_epivec, stages3) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3_kernel` | -- (native) | pipeline depth 4->3, matching real CUTLASS's actual `MmaMultistage` stage count confirmed via direct binary cuobjdump inspection; UNTESTED |
| TOMOS FP16 (iter_epivec, stages5) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5_kernel` | -- (native) | pipeline depth 4->5, motivated by ncu stall-sampling showing DEPBAR directly preceding the hottest-stalling HMMA (latency-bound evidence); explicitly distinguished from the earlier, unrelated `wide_bigreg_sq_deep5_kernel`; UNTESTED |
| TOMOS FP16 (iter_epivec, align) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align_kernel` | -- (native) | re-pads A's shared-memory row stride 264->272 for 32-byte alignment (algebraically proven mutually exclusive with the original bank-conflict-avoidance pad -- a genuine, acknowledged trade-off); UNTESTED |
| TOMOS FP16 (iter_epivec, pad280) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280_kernel` | -- (native) | sibling of align, the other untested valid pad residue (280 = 256+24); notes a standard bank-conflict model FAILS to predict ncu's actual measured conflict count -- explicitly empirical; UNTESTED |
| TOMOS FP16 (iter_epivec, aswizzle) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle_kernel` | -- (native) | XOR-permuted A shared-memory layout, later stated (in acongruous's own header) to have used a WRONG premise (self-invented, not CUTLASS's real scheme); own header says UNTESTED -- see the discrepancy flagged above |
| TOMOS FP16 (iter_epivec, pipe) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe_kernel` | -- (native) | interleaves next-tile B-fragment loads per-warp instead of one batch before the next MMA block; explicitly called "a genuinely uncertain experiment"; UNTESTED |
| TOMOS FP16 (deep4, iter2, epivec) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec_kernel` | -- (native) | applies iter's hoisting principle a SECOND time, now to the ldmatrix fragment-loading lambdas iter's original fix never touched; "same proof technique as deep4_iter's own hoist"; UNTESTED |
| TOMOS FP16 (iter_epivec, gridswap) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap_kernel` | -- (native) | swaps which grid axis maps to M vs N tiles, motivated by an ncu finding 99.4% of excessive-L2-sector count sits in A's cp.async load with no addressing bug found; pure relabeling, cannot break correctness, performance mechanism unverified; UNTESTED |
| TOMOS FP16 (wide_bigreg, cutlassdims) | `gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec_kernel` | -- (native) | forks off plain wide_bigreg (not _sq): new BM=128/BN=256 tile matching CUTLASS's real `ThreadblockShape<128,256,32>` exactly; a LATER kernel's comment reveals a severe, size-growing real loss (-24 to -25% at N=32768/65536) attributed to wave quantization against the A100's 108 SMs |
| TOMOS FP16 (cutlassdims, 3-stage) | `gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec_kernel` | -- (native) | forks off cutlassdims_iter_epivec: pipeline stages 4->3 matching CUTLASS's real stage count via direct cuobjdump inspection; explicitly stated to NOT fix cutlassdims' own wave-quantization regression (done for CUTLASS-comparison completeness); UNTESTED |
| TOMOS FP16 (wide_bigreg_sq, deep5) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep5_kernel` | -- (native) | forks off deep4, one stage deeper (5); MEASURED real hardware: the regression transfers from an analogous packed-fp32_16 sibling -- loses to deep4 at every N>=2048 (-3.2% to -8.8%), 0 spill in both, not spill-driven |
| TOMOS FP16 (deep4, occ2) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2_kernel` | -- (native) | cuts SqBN 128->64, `__launch_bounds__(...,2)` targeting 2 blocks/SM, direct analog of a packed-fp32_16 sibling; MEASURED real hardware: genuine split, real win at N=1024 (+60.4%) then worsening loss at every larger N (down to -38.7% at N=8192), light register spill once launch-underutilization stops being the bottleneck; deep4 remains best for N>=2048 |
| TOMOS FP16 (deep4, bigmma) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma_kernel` | -- (native) | restructures the mainloop to issue all 64 mma.sync as one flat un-barriered block instead of deep4's two 32-instruction halves; MEASURED real hardware: clear negative (-29% to -37% at N>=2048), 0 spill (not spill-driven) -- root cause: inadvertently eliminated a cross-iteration software-pipelining benefit deep4's split-load design already provided |
| TOMOS BF16 (plain deep4) | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_kernel` | -- (native) | explicit bfloat16 sibling of the plain fp16 deep4 kernel: same tile/pipeline/structure, `__nv_bfloat16` swapped for `__half`, `mma.sync` operand suffix `.f16.f16`->`.bf16.bf16`; first use of `cuda_bf16.h` in this codebase; UNTESTED, "not benchmarked or profiled before being handed over" |
| TOMOS FP16 (wide_bigreg_sq, bk64) | `gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_kernel` | -- (native) | forks off deep4, BK 32->64 (half the outer iterations, forced to 3 stages for shared-memory budget); required real re-derivation of two index-math pieces (traced and fixed a genuine silent-corruption bug in B's swizzle formula in the process); MEASURED real hardware: a flagged register risk materializes exactly -- consistent loss vs deep4 at every N>=2048 (-29% to -40%), 255/255 registers with real spill; the index-math re-derivation itself verified correct (loss is purely register-pressure) |
| TOMOS FP16 (bk64, reg2) | `gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_kernel` | -- (native) | forks off bk64, restructures K-substep register buffering back to deep4's shallow 2-slot depth; MEASURED real hardware: confirms the register diagnosis (+46-76% over bk64 at N>=2048, registers land at deep4's scale, 0 spill); not a clean win over deep4 (ties/slightly trails N>=8192, but actually BEATS deep4 at N=4096, +5.8%) |
| TOMOS FP16 (bk64_reg2, iter+epivec) | `gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec_kernel` | -- (native) | layers deep4_iter's address hoisting + deep4_epivec's vectorized epilogue on top of bk64_reg2's register-fixed mainloop, testing whether the same additive fixes close bk64_reg2's remaining gap to deep4_iter_epivec; UNTESTED, no run yet |
| TOMOS FP16 (acongruous, pipe retest) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe_kernel` | -- (native) | retests the earlier failed `pipe` technique on top of the FIXED acongruous baseline instead of the old overhead-heavy one, motivated by ncu now showing the hottest HMMA dominated by `stall_short_sb` (exactly pipe's target dependency); "GENUINELY EMPIRICAL"; UNTESTED |
| TOMOS FP16 (acongruous2) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2_kernel` | -- (native) | splits acongruous's A-address formula into a kk-only precompute half + m-only finish half, applying deep4_iter's hoisting principle to acongruous's more complex formula for the first time; correctness algebraically guaranteed via full Python sweep, zero mismatches; UNTESTED |
| TOMOS FP16 (acongruous, stages3) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3_kernel` | -- (native) | pipeline 4->3 stages layered on acongruous's real-CUTLASS A-formula (re-tested since acongruous genuinely changed A's cp.async/bank-conflict profile vs the old stages3 attempt); called "a cheap, low-risk test"; UNTESTED |
| TOMOS FP16 (acongruous, ldgsts) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts_kernel` | -- (native) | restructures cp.async issuance to interleave one-at-a-time instead of bursting all 6, matching CUTLASS's real compiled SASS pattern (static binary diff, no GPU execution used for the comparison itself); "GENUINELY EMPIRICAL"; UNTESTED |
| TOMOS FP16 (acongruous, fastk) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_kernel` | -- (native) | deletes the partial-last-K-tile boundary branch entirely (compile-time-unreachable, given a host-verified exact-K-multiple guarantee); the `fastk` lever cited in the promoted kernel's own entry; motivated by a 474 vs 140 instruction/K-tile gap vs CUTLASS despite identical HMMA/LDSM; UNTESTED |
| TOMOS FP16 (fastk, precomp) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp_kernel` | -- (native) | precomputes A's entire swizzled offset once per thread before the mainloop (hoisting across the whole K-loop, a harder invariance proof than intra-call CSE, confirmed via two prior negative compile-only tests that ptxas already handles the easier case); compile-only: 290->275 instructions/iter; UNTESTED on GPU |
| TOMOS FP16 (fastk_precomp2) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_kernel` | -- (native) | extends the same precomputed-offset technique to B's crosswise index; the `precomp2` lever cited in the promoted kernel's own entry; compile-only 275->247 instructions/iter (47.9% total vs original acongruous); later called "the standing champion of this family"; UNTESTED on GPU (compile-only only) |
| TOMOS FP16 (fastk_precomp3) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3_kernel` | -- (native) | sibling fork off fastk_precomp2 (not in the promoted chain): splits `__pipeline_wait_prior`'s DEPBAR dispatch into 3 literal-arg calls instead of a 9-case runtime dispatch; compile-only "by far the largest single reduction of this entire investigation" (247->182 instructions/iter, 61.6% total); own comment says UNTESTED, but a LATER kernel's comment reveals it WAS subsequently GPU-tested and REGRESSED ("a real, GPU-measured regression"; "had by far the best static numbers of this whole investigation and still lost at 5 of 6 sizes") -- the family's explicit cautionary example that static instruction counts stop predicting runtime performance once overhead is no longer the bottleneck |
| TOMOS FP16 (fastk_precomp2, pipe) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe_kernel` | -- (native) | sibling fork off fastk_precomp2 (not precomp3, "whose own outstanding-dispatch change is a real, GPU-measured regression"): applies pipe's B-load interleaving technique on this leaner base, motivated by real ncu profiling now showing a fundamentally different (compute/latency-bound, not overhead-bound) stall regime; explicitly "CANNOT predict the outcome here" per the precomp3 lesson; "GENUINELY UNCERTAIN"; UNTESTED |
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | also the tail call of "(workspace)" below |
| TOMOS FP16 (serp, ildsm) | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm_kernel` | -- (native) | descends from the promoted serp kernel: repositions individual ldmatrix calls to interleave 1-to-1 with HMMA from a different, already-ready block instead of an 8-LDSM burst, matching a pattern read directly from CUTLASS's own compiled mainloop; motivated by real ncu profiling showing opposite stall-math/stall-wait emphasis vs CUTLASS even after HMMA/LDSM counts and `.reuse` were already maximized; "the most structurally invasive change in this entire family's history"; UNTESTED |
| TOMOS BF16 | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | also the tail call of "TOMOS FP32 / BF16" below |
| TOMOS FP16 (wide_multistage) | `gemm_fp16_mma_ptx_wide_multistage_kernel` | -- (native) | forks off wide_epi_wide; a first literal CUTLASS-style double-buffered port MEASURED a clear failure (~5.6x worse, real spilling at the 64-register/1024-thread ceiling); redesigned as a register-conscious rolling pipeline extending the existing single-B-pair prefetch across the tile boundary instead |
| TOMOS FP16 (deep5, bigwarp) | `gemm_fp16_mma_ptx_deep5_bigwarp_kernel` | -- (native) | "a genuinely unexplored point in the design space": matches cuBLAS's real nsys-traced tile (128x128/BK=32/5 stages) AND CUTLASS's real 2x2/4-warp/128-thread split (from reading a real CUTLASS example); deliberately trades occupancy for per-warp compute granularity; measured ~40-48K GFlop/s, losing decisively to CUTLASS's ~226-240K (disclosed retroactively in a later kernel's comment) |
| TOMOS FP16 (bigwarp, apipe) | `gemm_fp16_mma_ptx_deep5_bigwarp_apipe_kernel` | -- (native) | adds register-level double-buffering for A only, since matching CUTLASS's tile/warp shape alone didn't transfer without matching its scheduling (real mainloop reading showed CUTLASS double-buffers both operands) |
| TOMOS FP16 (bigwarp, fullpipe) | `gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe_kernel` | -- (native) | finishes the port by also double-buffering B at the register level, since apipe's A-only fix "barely moved the needle (+2.9% at best)"; explicitly UNTESTED, "deliberately not benchmarked in this build" |
| TOMOS FP16 (bigwarp2) | `gemm_fp16_mma_ptx_deep5_bigwarp2_kernel` | -- (native) | a direct ncu comparison at N=16384 pinpointed the real gap: CUTLASS packs twice the warps/block (128x256 vs 128x128 threadblock) at near-identical register count; bigwarp2 doubles BN and widens the warp grid to isolate exactly that variable; UNTESTED, "not benchmarked in this build, per request" |
| TOMOS FP16 (bigwarp2, epismem) | `gemm_fp16_mma_ptx_deep5_bigwarp2_epismem_kernel` | -- (native) | applies the already-validated shared-memory-staged coalesced epilogue fix (from the `wide` family) to bigwarp2, after bigwarp2's own occupancy doubling "confirmed: +49-57% real throughput at N>=4096" but left a large residual epilogue-driven gap; no specific number given (self-referential comment) |
| TOMOS FP16 (bigwarp2, hoisted) | `gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted_kernel` | -- (native) | address-hoisting fork off bigwarp2 (independent of epismem): direct SASS comparison found 37.7% ISETP/IMAD vs CUTLASS's 11.9% despite near-identical HMMA counts, caused by per-thread-invariant address arithmetic recomputed every call; hoists it, leaves bounds checks fully dynamic for correctness |
| TOMOS FP16 (bigwarp3) | `gemm_fp16_mma_ptx_deep5_bigwarp3_kernel` | -- (native) | forks off bigwarp2_hoisted: SASS inspection found a costly `IMAD.WIDE.U32` reciprocal-multiply chain for `t % 5` (deep5's non-power-of-2 stage count); switches to 4 stages so the modulo becomes a cheap AND-mask |
| TOMOS FP16 (bigwarp4) | `gemm_fp16_mma_ptx_deep5_bigwarp4_kernel` | -- (native) | forks off bigwarp3: further SASS inspection found `col*ld_b`/`kg*ld_a` recomputed from scratch every call despite being thread-invariant or fixed-stride; hoists/replaces with a running accumulator, "exactly CUTLASS's own iterator pattern" |
| TOMOS FP16 (bigwarp5) | `gemm_fp16_mma_ptx_deep5_bigwarp5_kernel` | -- (native) | forks off bigwarp4: bigwarp3/4's fixes bought only ~1% real throughput despite a ~2x instruction-count gap -- now ILP/scheduling-bound, not instruction-count-bound; spreads next-tile cp.async issuance across the ks-loop (kept) matching CUTLASS's real mainloop; a second cross-iteration pipelining attempt was tried and REVERTED after compute-sanitizer racecheck caught a genuine data race; also documents an unrelated pre-existing race in a shared priming helper, confirmed present already in bigwarp3, not introduced here |
| TOMOS FP16 (bigwarp6) | `gemm_fp16_mma_ptx_deep5_bigwarp6_kernel` | -- (native) | forks off bigwarp5: implements CUTLASS's real B shared-memory swizzle (`RowMajorTensorOpMultiplicandCongruous<16,64>`, an XOR address swizzle, sourced directly from CUTLASS's own layout header) since bigwarp5 is already at the 255-register ceiling with no room for a register-shuffle-based fix; comment argues correctness holds even if the swizzle doesn't bit-match CUTLASS's own choice, since XOR partition swizzles are self-inverse |
| TOMOS FP16 (wide, spread cp.async) | `gemm_fp16_mma_ptx_wide_spread_kernel` | -- (native) | separate fork off the `wide` lineage (not bigwarp): spreads cp.async issuance across the ks-loop, read verbatim from CUTLASS's real mainloop source (not an AI-summarized paraphrase, per the comment); MEASURED real hardware: spill jumps from wide's 4B to 136/196B, ~30% below family-best at every N -- isolates the cost as a spill regression on wide's tight 64-register budget, unlike bigwarp's ample headroom; "wide_nospill_epismem remains the best kernel from this investigation" |
| TOMOS FP16 (wide, nospill+epismem+coalb) | `gemm_fp16_mma_ptx_wide_nospill_epismem_coalb_kernel` | -- (native) | combines three previously-separate `wide`-family fixes (nospill2 + epi_smem + a coalesced B-load shuffle) never tested together, since the coalesced-B fix was only ever benchmarked unfairly against wide's OLD uncoalesced epilogue before; no specific number given (self-referential comment) |
| TOMOS FP16 (smalltile, 2 blocks/SM) | `gemm_fp16_mma_ptx_smalltile_2block_kernel` | -- (native) | structurally distinct idea, not part of wide_bigreg or bigwarp: every prior kernel in the 32-warp family is stuck at 50% (1 block/SM) occupancy; shrinks the tile ~10x (FragM=1/FragN=1, the smallest non-degenerate step) specifically to fit 2 resident blocks/SM, "a real, deliberate gamble in the OPPOSITE direction of every previous tile-size experiment"; own comment explicitly flags even the intended occupancy gain as unconfirmed ("fitting the register budget on paper doesn't by itself guarantee the occupancy calculator agrees") |
| TOMOS FP32 / FP16 | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` | decode fused inline into the GEMM kernel; compute stays fp16 throughout |
| TOMOS FP32 / FP16 (fast) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp` | `fp32_16_half_only_fast` | lossy: underflow saturates to signed zero instead of a real fp16 subnormal |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` (via `mm_fp32_16_pure_half_materialize_chunk_vec_kernel`) | decodes the whole matrix to a cached scratch buffer up front, then calls the native TOMOS FP16 kernel above unmodified |
| TOMOS FP32/FP16 (naive scalar) | `mm_fp32_16_pure_half_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | fp32_16_cuda_kernels.cu; one-thread-per-C-element scalar loop, __half accumulate, float I/O; has a conditional-decode twin |
| TOMOS FP32/FP16 (naive tiled) | `mm_fp32_16_pure_half_tiled_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | 128x128/BK=8, 8x8 register-blocked tiling, __half compute; has a conditional-decode twin |
| TOMOS FP32/FP16 (wmma) | `mm_fp32_16_pure_half_wmma_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | `wmma::` API tensor-core kernel (not raw `mma.sync` PTX), fp32 accumulator, per-warp 16x16 shared staging buffer |
| TOMOS FP32/FP16 (wmma cp.async) | `mm_fp32_16_pure_half_wmma_cpasync_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | wh's raw bytes cp.async'd directly into As, decoded in place (no separate staging buffer needed) |
| TOMOS FP32/FP16 (wmma cp.async vec) | `mm_fp32_16_pure_half_wmma_cpasync_vec_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | widens wh's cp.async transfer (2->8 elements) and B's read to float4 |
| TOMOS FP32/FP16 (wmma cp.async vec bigbk) | `mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | BK widened 32->64 |
| TOMOS FP32/FP16 (wmma cp.async vec bigbk pad) | `mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | + shared-memory padding |
| TOMOS FP32/FP16 (wmma cp.async vec hugebk) | `mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | BK widened to 128, `__launch_bounds__(...,1)` |
| TOMOS FP32/FP16 (wide_bigreg_sq fallback) | `mm_fp32_16_pure_half_wide_bigreg_sq_fallback_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | always-correct 1-thread/element scalar fallback for shapes the tensor-core fast path rejects; has a conditional-decode twin |
| TOMOS FP32/FP16 (wide_bigreg_sq fast fallback) | `mm_fp32_16_pure_half_wide_bigreg_sq_fast_fallback_kernel` | `fp32_16_half_only_fast` | same as above, fast decode |
| TOMOS FP32/FP16 (wide_bigreg_sq / tile_dirty) | `mm_fp32_16_pure_half_wide_bigreg_sq_kernel<SkipBeta,UseTileDirty>` | `fp32_16_half_only_branchless_no_overflow_nomask` | base `mma.sync` PTX tensor-core kernel; one template body serves BOTH `wide_bigreg_sq` (UseTileDirty=false, unconditional per-element decode) and `wide_bigreg_sq_tile_dirty` (UseTileDirty=true, precomputed per-(block_row,k_tile) dirty flag via `fp32_16_compute_tile_dirty_kernel`) |
| TOMOS FP32/FP16 (inline_dirty) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | standalone sibling, deeper 4-stage pipeline + fully inline batched dirty check (precursor to the promoted kernel, missing acongruous/fastk/precomp2/serp) |
| TOMOS FP32/FP16 (inline_dirty fast) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_kernel<SkipBeta>` | `fp32_16_half_only_fast` | same as above, fast decode |
| TOMOS FP32/FP16 (materialize, uncached scalar) | `mm_fp32_16_pure_half_materialize_chunk_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | scalar decode-prepass kernel; host wrapper `mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp` `cudaMalloc`/`cudaFree`s scratch every call; has a conditional-decode twin |
| TOMOS FP32/FP16 (materialize, cached scalar) | `mm_fp32_16_pure_half_materialize_chunk_kernel` (same kernel as above) | `fp32_16_half_only_branchless_no_overflow_nomask` | host wrapper `mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp` reuses a `static` scratch buffer instead |
| TOMOS FP32/FP16 (materialize vec, overlap) | `mm_fp32_16_pure_half_materialize_chunk_vec_kernel` (same kernel as the promoted workspace row) | `fp32_16_half_only_branchless_no_overflow_nomask` | host wrapper `mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp` double-buffers decode/GEMM across 2 CUDA streams; no new kernel |
| TOMOS FP32/FP16 (materialize vec64) | `mm_fp32_16_pure_half_materialize_chunk_vec64_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | single 64-bit `STG.E.64` store instead of two 32-bit stores; has a conditional-decode twin |
| TOMOS FP32/FP16 (materialize dirty) | `mm_fp32_16_pure_half_materialize_chunk_dirty_kernel` | `fp32_16_half_only_branchless_no_overflow_nomask` | vec64 + per-thread (4-element) clean/dirty skip check; has a conditional-decode twin |
| TOMOS FP32/FP16 (auto dispatcher) | dispatches to `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` (n>=2048) or `..._perthread` (n<2048) | (inherits whichever branch's decode) | `mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp`; host-only size-threshold dispatcher, no own kernel |
| TOMOS FP32/FP16 (perrep) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | dirty check narrowed from per-thread (4 reps bundled) to per-rep (each of 4 reps checked independently) |
| TOMOS FP32/FP16 (perthread) | `..._perthread_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | drops `__any_sync`; each thread branches on its own combined dirty bit (no cross-lane vote) |
| TOMOS FP32/FP16 (perthread leanreg) | `..._perthread_leanreg_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | perthread + drops the held `raw[]/idx[]` array, re-reads from shared memory on the (rare) dirty path instead |
| TOMOS FP32/FP16 (alwaysdecode) | `..._alwaysdecode_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` (unconditional) | removes the dirty check entirely; every thread always decodes |
| TOMOS FP32/FP16 (blockvote) | `..._blockvote_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | replaces per-warp `__any_sync` with a block-wide `__syncthreads_or` vote+barrier |
| TOMOS FP32/FP16 (precomputed_dirty) | `mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | reads a precomputed per-chunk dirty byte from global memory (`fp32_16_compute_chunk_dirty_kernel`) instead of deriving it from `wh` inline |
| TOMOS FP32/FP16 (warp_specialized) | `mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | only warps 0-3 perform decode (each covering its own + warp+4's share); warps 4-7 skip decode entirely |
| TOMOS FP32/FP16 (prefetched_dirty) | `mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | same precomputed dirty byte as precomputed_dirty, but prefetched into shared memory via cp.async instead of a global read |
| TOMOS FP32/FP16 (dualocc) | `mm_fp32_16_pure_half_wide_bigreg_sq_dualocc_kernel<SkipBeta>` | `fp32_16_half_only_branchless_no_overflow_nomask` | BN halved 128->64, `__launch_bounds__(...,2)` for 2 resident CTAs/SM |
| TOMOS FP32 / BF16 | `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` | none — `wh` reinterpreted directly as `__nv_bfloat16*` | no decode kernel at all: bf16's exponent field matches fp32's own bit-for-bit, so truncated `wh` is always already a valid bf16 pattern; calls the native TOMOS BF16 kernel above directly |
| TOMOS FP32/BF16 (deep4, standalone) | `mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4_kernel<SkipBeta>` | none -- raw `wh` bytes cp.async'd straight into shared memory, reinterpreted as `__nv_bfloat16` in place | predates the promoted kernel; a fully independent, standalone tensor-core kernel body (mechanical copy of `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_kernel` with A retyped to `const uint16_t* wh`), NOT a thin wrapper around the native BF16 kernel; lacks the native family's later `iter`/`epivec`/`acongruous`/`fastk`/`precomp2`/`serp` refinements |
| TOMOS FP32/BF16 (deep4 fallback) | `mm_fp32_bf16_pure_bf16_fallback_kernel` | `fp32_bf16_bf16_only` | always-correct 1-thread/element scalar fallback for shapes/transposes the deep4 kernel's fast path doesn't cover; mirrors `mm_fp32_16_pure_half_wide_bigreg_sq_fallback_kernel`'s structure |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/mm_cuda_kernels.cu`
  (declarations in `src/kernels/cuda/matrix/mm_cuda_kernels.cuh`).
- Decode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16 family),
  `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu`.

**Method Description:**
- **cuBLAS FP16** (`cublasHgemm`) — NVIDIA's closed-source half-precision
  GEMM; external reference baseline, not part of this codebase.
- **cuBLAS BF16** (`cublasGemmEx`, `CUDA_R_16BF`) — NVIDIA's closed-source
  bfloat16 GEMM (via the generic `GemmEx` entry point rather than a
  dedicated `Hgemm`-style call, since cuBLAS has no bf16-specific
  top-level function); external reference baseline.
- **CUTLASS FP16** (`cutlass::gemm::device::Gemm<...>`, fp16 tensor-op
  config: `ElementA/B/C = cutlass::half_t`, `OpClassTensorOp`, `Sm80`,
  `ThreadblockShape<128,256,32>`/`WarpShape<64,64,32>`/
  `InstructionShape<16,8,16>`) — NVIDIA's templated tensor-core GEMM for
  fp16; external reference baseline.
- **CUTLASS BF16** (`cutlass::gemm::device::Gemm<...>`, bf16 tensor-op
  config: byte-for-byte the same template parameters as the FP16 config
  above except `ElementA/B/C = cutlass::bfloat16_t` — CUTLASS supports
  bf16 tensor-core MMA at the same `m16n8k16` granularity as fp16 on
  `sm_80`, so only the element types change) — NVIDIA's templated
  tensor-core GEMM for bf16; external reference baseline.

The native FP16/BF16 lineage below (74 `gemm_fp16_mma_ptx_*` + 2
`gemm_bf16_mma_ptx_*` kernels, organized as a five-chapter narrative rather
than raw source order) all operate on plain `__half*`/`__nv_bfloat16*`
storage with zero decode/unpack step -- confirmed per-kernel by grepping
each kernel's own body for any `fp32_16_combine`/`fp32_16_split`/`decode`/
`pack`/`wh`/`wr` reference; no exception found anywhere in this family.
Both promoted kernels' host-level "thin forwarding wrapper" structure
(`namespace experimental { ... }` closes in `mm_cuda_kernels.cu`,
immediately followed by two plain-`tomos::cuda::` forwarding wrappers that
call straight through to the `experimental::` implementation) was
independently confirmed by two separate research passes over this family.

**Flagged, unresolved source discrepancy** (do not silently pick a side):
`aswizzle`'s own header comment states "UNTESTED: no performance
measurement or correctness verification has been run yet." The later
`acongruous` kernel's own header instead refers back to "aswizzle's own
failure (~50% loss)" as a known, measured result. These two source
comments directly contradict each other and cannot be reconciled from
source alone -- flagged here (and cross-referenced from both the `aswizzle`
table row and its own Method Description bullet in Chapter 3 below) rather
than asserted either way.

*Chapter 1: the "bigtile"/occupancy prehistory (predates wide_bigreg)*

- **TOMOS FP16 (mma_ptx, base)** (`gemm_fp16_mma_ptx_kernel`) -- replaces an
  earlier `nvcuda::wmma`-API kernel with hand-written `mma.sync.aligned.m16n8k16`
  PTX plus `ldmatrix.trans` fragment loads (the real Ampere primitive, half of
  wmma's N=16 shape), writing the accumulator straight to global memory with no
  shared-memory staging round trip. The per-thread register layout for both mma
  operands and ldmatrix's own address/output mapping were reverse-engineered
  empirically (no vendor docs access) via a standalone delta-function test
  validated against a dense random CPU reference. MEASURED: best kernel across
  all tested N at write time, 0 register spill (104 registers/thread).
- **TOMOS FP16 (bigtile)** (`gemm_fp16_mma_ptx_bigtile_kernel`) -- doubles the
  threadblock tile to 256x128 (BM=256) and drops BK from 128 to 64 to fit
  shared memory, attacking HBM bandwidth rather than compute/occupancy: a
  roofline estimate showed mma_ptx's 128x128 tile was plausibly bandwidth-bound,
  and doubling BM halves redundant A/B HBM re-reads. Reuses mma_ptx's validated
  ldmatrix/mma.sync register mapping verbatim, only tile constants change.
  MEASURED: beats mma_ptx at every N, margin growing with N (~34% faster at
  N=1024, ~2x at N=16384), 0 spilling -- best result to that point.
- **TOMOS FP16 (bigtile, grid swizzle)** (`gemm_fp16_mma_ptx_bigtile_swizzle_kernel`)
  -- same tile/register footprint as bigtile, changes only the blockIdx-to-tile
  mapping: a "grouped" swizzle holds block_col fixed for a runtime-derived
  group_size of consecutive linear block IDs, hoping concurrently-scheduled
  blocks share a B-column strip for L2 reuse on top of bigtile's already-reduced
  traffic. Handles a non-dividing group_size correctly (a real correctness bug
  the comment explicitly warns a naive version would hit). MEASURED: a clear
  but modest loss, ~9-10% slower than plain bigtile at large N regardless of
  group_size; correctness passes throughout, a genuine negative result. Not
  promoted.
- **TOMOS FP16 (bigtile2)** (`gemm_fp16_mma_ptx_bigtile2_kernel`) -- pushes
  bigtile's bandwidth argument further, growing both dimensions to 256x256 (BK
  still 64), cutting HBM traffic another ~33% vs bigtile. MEASURED: a clear
  loss -- pushes register need past the 128-register cap with real spilling
  (1112B/1116B spill stores/loads); throughput drops below even the original
  128x128 mma_ptx. Marks where tile-scaling's bandwidth benefit stops
  outrunning its register-pressure cost. Not promoted.
- **TOMOS FP16 (bigtile, 3-stage pipeline)** (`gemm_fp16_mma_ptx_bigtile_tripipe_kernel`)
  -- same 256x128/BK=64 tile/register footprint as bigtile (avoiding bigtile2's
  spill risk), extends the cp.async prefetch depth from 2 to 3 stages instead,
  targeting latency-hiding/occupancy rather than bandwidth or registers. Costs
  only shared memory. MEASURED: essentially tied with bigtile at every N
  (within ~1-5%) -- bigtile's 2-stage buffer was already hiding the relevant
  latency; not promoted.
- **TOMOS FP16 (bigtile, lean/deep B pipeline)** (`gemm_fp16_mma_ptx_bigtile_deep_kernel`)
  -- restructures the compute loop's register lifetimes instead of tile size or
  pipeline depth: loads B one `wn` at a time (2 live registers, immediately
  consumed and overwritten) with the next wn's ldmatrix issued one step ahead,
  a finer-grained latency-hiding scheme than tripipe's whole-tile cp.async. A
  stays fully cached. MEASURED: neutral, tied with bigtile at every N, same
  register/spill profile (the compiler already found an equally good
  allocation) -- combined with tripipe's own neutral result, convergent
  evidence that latency-hiding wasn't the remaining bottleneck. Not promoted.
- **TOMOS FP16 (bigtile, 32-warp occupancy)** (`gemm_fp16_mma_ptx_bigtile_w32_kernel`)
  -- same 256x128/BK=64 tile/HBM traffic as bigtile, targets warp-scheduler
  OCCUPANCY instead (a mechanism distinct from bandwidth or latency-hiding,
  both separately ruled out above): splits across 32 warps (1024 threads,
  double bigtile's 512) instead of 16, halving the per-warp tile and
  accumulator registers (64->32/thread). Reuses bigtile_deep's lean B loading
  since register headroom is tight. MEASURED: decisive win -- 62 registers/
  thread, 0 spilling; beats bigtile at every N with margin GROWING at larger N
  (unlike every prior kernel in this chapter), confirming occupancy was the
  real remaining bottleneck.
- **TOMOS FP16 (bigtile w32, skip-beta epilogue)** (`gemm_fp16_mma_ptx_bigtile_w32_epi_kernel<SkipBeta>`)
  -- identical compute core to bigtile_w32, but when the host-side double beta
  is exactly 0.0 (checked before precision loss), skips the beta*C read
  entirely -- the one CUTLASS-style epilogue trick available, since true
  vectorized C stores aren't achievable given the fragment's memory-adjacent
  layout. SkipBeta is a compile-time template parameter, not a runtime bool: a
  first runtime-branching version pushed ptxas to a forced spill that scaled
  with N and lost at large N; splitting into two instantiations restored the
  exact 62-register/0-spill footprint for both branches. MEASURED: clean win
  at every N after the template fix, +3-8%, new family best.
- **TOMOS FP16 (bigtile w32 epi, L2 persistence)** (`gemm_fp16_mma_ptx_bigtile_w32_epi_l2_kernel<SkipBeta>`)
  -- identical compute core to w32_epi, but the host launches one column-tile's
  worth of blocks at a time and sets a CUDA L2 persistence window
  (`cudaStreamAttributeAccessPolicyWindow`, Ampere+) on B's column-strip before
  each launch -- a deterministic alternative to the swizzle kernel's hopeful
  approach, targeting B specifically since its column-strip is contiguous while
  A's row-strip (column-major storage) is not. MEASURED: clear loss at every N
  (~15%), attributed to per-launch overhead (one extra kernel launch + one
  `cudaStreamSetAttribute` call per column tile). Also surfaced and fixed a
  real bug: the L2 persistence limit is device-wide, not per-stream/kernel --
  an early version left it set statically, silently halving an unrelated later
  kernel's (bigtile2's) throughput in the same benchmark process. Not promoted.
- **TOMOS FP16 (bigtile2, 32-warp)** (`gemm_fp16_mma_ptx_bigtile2_w32_kernel`)
  -- tests whether bigtile_w32's occupancy win composes with bigtile2's larger
  256x256 tile at 32 warps (1024 threads, hardware max/block). The comment
  states plainly, before the result, that the register math is unfavorable:
  accumulator/thread at this tile/thread-count IS the entire per-thread
  register budget, leaving nothing for operands/addresses/loop counters, with
  no larger thread count to dilute into. MEASURED: confirms the prediction and
  worse -- hits the hard 64-register cap with real spilling, over 5x slower
  than bigtile_w32 and even slower than the original 16-warp bigtile2 despite
  twice the warps. Definitively closes the question: bigtile_w32's tile is
  very close to the largest that fits without spilling at max threads/block.
  Not promoted.
- **TOMOS FP16 (bigtile w32 epi, wide N=160)** (`gemm_fp16_mma_ptx_bigtile_w32_epi_wide_kernel<SkipBeta>`)
  -- grows the tile in N only (BN 128->160), exploiting an asymmetry the
  lean-B design creates that bigtile2/bigtile2_w32 (growing both dimensions
  together) missed: B never holds more than 2 live registers regardless of
  FragN, so growing FragN adds ONLY accumulator registers, not operand-cache
  overhead, unlike A (fully cached, ~4 registers/unit of FragM growth). The
  prediction (~70 registers, grounded in w32_epi's own measured 62=32+30
  accumulator/overhead split) landed almost exactly: 64 registers (hits the
  cap) with only 4 bytes of spill. MEASURED: beats w32_epi at every N (+9-31%),
  best result of the investigation to that point; new best in family.
- **TOMOS FP16 (bigtile w32 epi, wide2 N=192)** (`gemm_fp16_mma_ptx_bigtile_w32_epi_wide2_kernel<SkipBeta>`)
  -- the next increment in wide's direction (BN 160->192), explicitly framed as
  an empirical test of whether extra bandwidth reduction still outweighs an
  expected real spill, since wide had already used essentially all its
  register headroom. MEASURED: the predicted real spill materializes
  (172-176 bytes, vs wide's 4 bytes) and this time the spill cost wins --
  wide2 never beats wide at any tested N. Not promoted; pins BN=160 as the
  sweet spot and BN=192 as one increment too far.
- **TOMOS FP16 (bigtile w32 epi, wide2b, 16x2 warps)** (`gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b_kernel<SkipBeta>`)
  -- retries wide2's same 256x192 tile (same total accumulator, same mma.sync
  call count) with a different warp arrangement (16x2 instead of 8x4),
  halving A's per-warp cache (FragM 2->1) while B stays lean regardless,
  testing whether the smaller A-cache is enough headroom to avoid or shrink
  wide2's real spill. Unlike every other kernel in this chapter, its own
  preceding comment block contains no separate MEASURED RESULT paragraph with
  numbers -- its outcome is only stated in passing inside a later kernel's
  comment, which groups it with wide2 as "the opposite direction... also a
  loss," bracketing bigtile_w32_epi_wide as a genuine local optimum from both
  sides. Not promoted; wide remains the best kernel in this chapter (and,
  per that later kernel's comment, "overall" at that point in the file).

*Chapter 2: the parallel deep5/nospill/epismem/cutlass_config chapter*

- **TOMOS FP16 (deep5)** (`gemm_fp16_mma_ptx_deep5_kernel`) -- matches cuBLAS's
  actual dispatched SASS config, discovered via nsys (hardware ncu counters
  were unavailable): `ampere_h16816gemm_128x128_ldg8_stages_32x5_nn` decodes
  to a 128x128 tile, BK=32, 5-stage cp.async pipeline, 256 threads (targeting
  2 blocks/SM = 16 active warps/SM, half of wide's occupancy). MEASURED:
  consistent loss, ~0.47-0.50x of wide's throughput at N=2048-32768, plateaus
  at an occupancy-bound-looking level. Brackets `bigtile_w32_epi_wide` as a
  genuine local optimum alongside the wide2/wide2b bigger-tile experiments.
  Not promoted.
- **TOMOS FP16 (wide, deep5 pipeline)** (`gemm_fp16_mma_ptx_wide_deep5_kernel`)
  -- isolates pipeline depth alone from deep5's 3-variable conflation: wide's
  exact tile/warp/thread/epilogue, only BK shrunk 64->32 and pipeline deepened
  2->5 (matching cuBLAS's real config without sacrificing occupancy this
  time). MEASURED: does not improve on wide -- identical registers/spill/
  occupancy -- closing off pipeline depth as a lever independent of the tile/
  occupancy confound deep5 had. Not promoted.
- **TOMOS FP16 (wide, coalesced B)** (`gemm_fp16_mma_ptx_wide_coalb_kernel`)
  -- targets ncu's #1 flagged issue on `wide` (uncoalesced B-tile global
  loads, since B is column-major but consecutive threads originally varied
  by column): swaps the fastest-varying index so consecutive threads instead
  cover consecutive K-offsets within the same column. A separate, abandoned
  follow-on padding attempt (documented inline, not its own kernel) was found
  mathematically incompatible with ldmatrix's alignment and crashed
  compute-sanitizer. Confirmed "regressed" per a later kernel's retrospective
  comment.
- **TOMOS FP16 (wide, coalesced B, shuffle)** (`gemm_fp16_mma_ptx_wide_coalb_shfl_kernel`)
  -- sidesteps the coalescing/bank-conflict conflict differently: keeps wide's
  original conflict-free shared write layout untouched, reads B coalesced
  (coalb's mapping), then uses `__shfl_sync` to redistribute the loaded
  components within the warp so the shared write matches wide's original
  assignment byte-for-byte -- a pure read/write decoupling, no new layout.
  Also confirmed "regressed" per the same later retrospective.
- **TOMOS FP16 (deep5, w32)** (`gemm_fp16_mma_ptx_deep5_w32_kernel`) -- fills
  the untested cell in the (tile size) x (occupancy) design space: reuses
  deep5's exact tile/BK/stage-count/shared layout with wide's 32-warp/1024-
  thread arrangement instead of deep5's own 8-warp one, testing cuBLAS's tile
  shape without the occupancy handicap. No MEASURED RESULT paragraph appears
  in its own comment (unlike every sibling in this chapter) -- outcome not
  stated in source.
- **TOMOS FP16 (wide, nospill attempt 1)** (`gemm_fp16_mma_ptx_wide_nospill_kernel`)
  -- targets wide's one remaining ncu-flagged issue, a real reproduced 4-byte
  register spill, traced via ncu source-line correlation (after fixing a
  CMakeLists.txt flag conflict that was silently dropping PTX line tables) to
  `next_k0` needing to stay live across the whole unrolled loop between its
  two uses. Fix attempted: collapse into one explicit boolean. Per a later
  kernel's comment, this "didn't move the spill at all" -- the compiler
  already did the same CSE regardless of source syntax; a documented negative
  result pinpointing the live range, not a missed optimization.
- **TOMOS FP16 (wide, nospill attempt 2)** (`gemm_fp16_mma_ptx_wide_nospill2_kernel`)
  -- instead of renaming the live range, removes `next_k0`'s second use
  entirely: makes `__pipeline_wait_prior(0)` unconditional at the loop bottom
  (trivially true if nothing new was committed), so `next_k0` need not survive
  past the top of the loop; only behavioral difference is one harmless extra
  no-op wait on the final iteration. Per a later kernel's comment: "a small
  but real and consistent win over wide" at N>=8192.
- **TOMOS FP16 (wide, smem epilogue)** (`gemm_fp16_mma_ptx_wide_epi_smem_kernel`)
  -- fixes a different confirmed ncu issue: uncoalesced global epilogue
  stores (wide's scattered per-thread writes land far apart in column-major
  C). Confirmed against real CUTLASS source that production kernels stage
  through shared memory first; reuses As/Bs's now-unused shared space to
  stage, then sweeps in coalesced waves with consecutive threads on
  consecutive rows. Per a later kernel's comment: consistent win over wide
  at N<=4096.
- **TOMOS FP16 (wide, nospill+epismem)** (`gemm_fp16_mma_ptx_wide_nospill_epismem_kernel`)
  -- composes nospill2's unconditional wait with epi_smem's shared-memory-
  staged epilogue, since the two touch disjoint parts of the kernel (K-loop
  sequencing vs. epilogue) with no structural reason not to combine.
  Explicitly named "the best kernel from this investigation" in two later
  kernels' own comments -- the reference point for the rest of this chapter.
- **TOMOS FP16 (nospill_epismem, deep3)** (`gemm_fp16_mma_ptx_wide_nospill_epismem_deep3_kernel`)
  -- tests whether an earlier, unrelated kernel's (bigwarp3, chapter 5) deeper-
  pipeline win transfers to this tile: same tile/warp/epilogue as
  nospill_epismem, a CUTLASS-derived 3-stage pipeline (stage count picked the
  way real CUTLASS derives it from shared-memory budget, not a constant),
  necessarily giving up nospill2's unconditional-drain trick. MEASURED: real,
  consistent regression at every N tested (-4% to -7%); diagnosed as already
  occupancy-capped at 1 block/SM even at 2 stages, so this tile's MMA-heavy
  loop has no spare latency for a deeper pipeline to hide. nospill_epismem
  remains best.
- **TOMOS FP16 (nospill_epismem, xswz)** (`gemm_fp16_mma_ptx_wide_nospill_epismem_xswz_kernel`)
  -- a from-scratch fix of the B-load coalescing problem that wide_coalb/
  wide_coalb_shfl (and an earlier bigwarp6, chapter 5) failed to solve:
  reading real CUTLASS source revealed this B tile actually needs
  `ColumnMajorTensorOpMultiplicandCrosswise<16,32>` (kFactor=2), not the
  RowMajor-style layout those earlier kernels assumed. Documents 4
  B-fragment-read attempts before the fix (reused old addressing -> crashed
  misaligned; hand-derived Policy + transposed ldmatrix -> confirmed data
  race + wrong values; a role-swap -> still wrong values; reading CUTLASS's
  real `ldsm` dispatch -> the correct non-transposed `ldmatrix.x4.m8n8` plus
  an exact byte-offset K-group-advance toggle, verified 0/1024 mismatches
  against the actual compiled CUTLASS iterator). Correctness confirmed via
  the full `verify_mm_fp16_gemm` suite on real GPU runs, both beta=0 and
  beta!=0. Doesn't match nospill_epismem outright.
- **TOMOS FP16 (nospill_epismem, xswz2)** (`gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2_kernel`)
  -- same crosswise B-load fix as xswz. First tried CUTLASS's own 8-warp/
  256-thread reference count to escape a real spilling problem found via ncu
  -- eliminated spilling but measured slower overall, since shared memory
  already capped occupancy at 1 block/SM, so the thread-count cut cost a 4x
  latency-hiding loss; reverted to the original warp arrangement (spilling,
  but empirically faster) as a documented negative result. Unlike xswz,
  FragN is even here, so a fetch/consume register fusion that hurt xswz
  helps here instead: ~83-86% of nospill_epismem's speed (up from ~65-70%
  pre-fusion), "consistently beating" a sibling coalb-combo kernel at every
  size tested. Still doesn't match nospill_epismem outright.
- **TOMOS FP16 (cutlass_config)** (`gemm_fp16_mma_ptx_cutlass_config_kernel`)
  -- builds CUTLASS's actual reference config from scratch (128x256x32
  threadblock tile, 64x64x32 warp tile, 3-stage pipeline) rather than
  transplanting just the warp count in isolation (xswz2's aborted attempt);
  reuses the crosswise B-load fix validated on xswz/xswz2 and an already-
  validated 3-stage buffer-management pattern. Initial 8-warp/256-thread
  split hit a 255-register ceiling (~216 registers, matching CUTLASS's own
  compiled kernel by disassembly -- yet CUTLASS runs 3.5-4x faster, so the
  gap isn't "need more register headroom"); switched to a 16-warp/512-thread
  split to shrink the accumulator, the single largest register consumer.
  MEASURED (real GPU numbers + `ptxas -v` + racecheck, 0 hazards): CONFIRMED
  WIN -- 512 threads/block hits a hard 128-register ceiling causing modest
  spilling but doubled occupancy more than compensates, +3-13% over the
  8-warp split. Three separate CUTLASS-technique adaptations were tried and
  reverted here for hitting this same register ceiling with no headroom left.

*Chapter 3: the wide_bigreg -> wide_bigreg_sq branch point, through acongruous's origin*

- **TOMOS FP16 (wide_bigreg, base)** (`gemm_fp16_mma_ptx_wide_bigreg_kernel`)
  -- the branch-point originator: a large per-warp accumulator tile, BM=256/
  BN=160, 3-stage cp.async pipeline, deliberately NOT matching CUTLASS's
  actual 128-float accumulator (uses 160, "wrong" by design, to test whether
  matching CUTLASS's tile/register shape mattered on its own). Two real
  in-kernel fixes: (1) hoisting `load_tile_async`'s per-call divide/modulo
  into precomputed values -- MEASURED +5% but introduced real spilling
  (255 registers, previously zero-spill); an isolated A/B test confirmed this
  specific hoisted state, not the smaller fragment-loading hoists, is the
  sole spill source. (2) converting B's load from a blocking scalar
  load+unpack+store sequence to a straight `__pipeline_memcpy_async`
  (re-deriving that the crosswise destination indices are actually 8
  contiguous 16-byte-aligned slots, not scattered as originally assumed) --
  MEASURED >2x, still 255 registers with real spill, landing above real
  CUTLASS's own measured throughput despite the "wrong" accumulator size. A
  separate signed-division "fix as unsigned" attempt was tried and reverted
  (measured regression). Real-hardware validated throughout.
- **TOMOS FP16 (wide_bigreg_sq)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_kernel`)
  -- retunes wide_bigreg's BN from 160 to 128, an *exact* match to real
  CUTLASS's own accumulator size and square 64x64 warp tile, motivated by
  direct SASS diffing against real CUTLASS (wide_bigreg: 822 static
  instructions/80 HMMA vs CUTLASS: 140 instructions/64 HMMA per iteration --
  a 1:1 spill-reload-per-MMA ratio traced directly to the oversized
  accumulator). Implemented as a fully separate, standalone kernel. MEASURED:
  hypothesis confirmed -- only 4/0 bytes spill (vs wide_bigreg's real
  spilling at the same ceiling); briefly "the best kernel in this entire
  file." Then contains its own extensive follow-on history in the same body:
  (a) the identical B-load cp.async fix already proven on wide_bigreg,
  applied here too -- MEASURED ~2x again, above CUTLASS, registers dropped
  with 0 spill; (b) a very long (18-point) empirical investigation into a
  confirmed real L2-sector overhead on A's loads under >=2 concurrent warps
  (isolated via a standalone repro program), root-caused all the way down to
  CUTLASS's `PredicatedTileAccessIterator` machinery itself (not addressing,
  predication style, or instruction adjacency -- all tested and ruled out)
  but deliberately NOT adopted as a fix since it targets a bandwidth cost
  this compute-bound kernel doesn't pay (a synchronous-load variant that
  removed the excess was tried and reverted, -35-40% regression, since it
  exposes latency this kernel currently hides via cp.async overlap); the
  actual applied fix was cheaper -- switching A's thread-to-address mapping
  to a CUTLASS-style "one fixed K-value per thread across 4 M positions"
  scheme, confirmed via the isolated repro to remove the excess entirely.
- **TOMOS FP16 (wide_bigreg_sq, deep4)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_kernel`)
  -- single-lever change: bumps the pipeline from the family's shared
  3-stage baseline to 4 stages, same tile otherwise, built as a fully
  standalone kernel. Motivated by an already-measured win from the same 3->4
  change on a *different* (fp32_16, decode-bearing) sibling kernel, applied
  here for the first time to this native fp16 kernel. MEASURED: consistent
  win over wide_bigreg_sq at N>=4096 (+1.6% to +3.6%), 0 spill; became the
  family's new best, edging past cuBLAS at N=32768/65536. Flat vs
  wide_bigreg_sq at N=1024/2048 (grid too small).
- **TOMOS FP16 (deep4, iter)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_kernel`)
  -- sibling fork off deep4 (not epivec): hoists `load_tile_async`'s per-call
  address arithmetic out of the lambda into precomputed/incrementally-
  advanced values, since a real SASS comparison against CUTLASS's matching
  fp16 kernel found deep4 issuing 3.0-3.3x more IMAD and 3.9-4.1x more LEA
  instructions per K-tile despite identical HMMA/LDSM counts. Provably
  equivalent by induction; only the safe-path address math changes, boundary
  paths byte-for-byte identical to deep4. UNTESTED, explicitly "no
  performance measurement run yet on real hardware."
- **TOMOS FP16 (deep4, epivec)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec_kernel`)
  -- sibling of iter, built directly on deep4's mainloop unchanged (isolates
  the epilogue change alone). Vectorizes the epilogue's Cs<->C transfer from
  scalar `__half` stores to 8-half `float4` transfers, using a
  one-warp-per-column-group mapping. Motivated by a real ncu profile of deep4
  flagging real excessive-global-sector and excessive-shared-wavefront
  opportunities tracing to the scalar epilogue (CUTLASS's own matching
  config explicitly uses a vectorized epilogue). The register->Cs
  "corner turn" is deliberately left scalar (proven algebraically
  non-vectorizable without a much larger relayout stage, since none of the
  4 values a tensor-core accumulator produces together are memory-adjacent
  in Cs's column-major layout). Requires stricter alignment (no boundary
  tiles reach this kernel), proven algebraically safe. UNTESTED, explicitly
  stated.
- **TOMOS FP16 (deep4, iter+epivec)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_kernel`)
  -- combines both independently-measured/attempted deep4 forks in one
  kernel; shares no code path with deep4, iter, or epivec themselves. The
  comment reports the two source fixes' individually-measured effects from
  earlier isolated real-hardware runs: iter alone is a small win concentrated
  at large N (+1.8% at N=16384) but a real loss at N=8192 (-3.35%); epivec
  alone is a much larger win at small/medium N (+17.9% at N=2048, fading to
  ~flat by N=16384+, consistent with its fixed per-block, K-independent
  cost). This kernel itself tests whether the two fixes compose additively.
  UNTESTED itself, explicitly "no performance measurement or correctness
  verification has been run yet."
- **TOMOS FP16 (iter_epivec, stages3)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3_kernel`)
  -- single-knob fork off iter_epivec: pipeline depth reduced to 3 stages,
  byte-for-byte iter_epivec otherwise. Motivated by direct cuobjdump
  inspection of the real, already-built CUTLASS reference binary showing its
  actual `MmaMultistage` mainloop uses Stages=3 (not 4/5) on an identical
  warp tile/MMA shape, combined with a cited prior negative result for 5
  stages -- evidence CUTLASS's advantage isn't from deeper buffering. UNTESTED.
- **TOMOS FP16 (iter_epivec, stages5)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5_kernel`)
  -- single-knob fork off the same base: pipeline depth increased to 5
  stages. Motivated by ncu per-instruction stall sampling showing DEPBAR
  (i.e. `__pipeline_wait_prior`) immediately precedes the single
  hottest-stalling HMMA across 3 converging code paths -- evidence the
  bottleneck is cp.async/DRAM latency, not register latency. Explicitly
  distinguished from the pre-existing, unrelated `wide_bigreg_sq_deep5_kernel`
  (chapter 3's own deep5 sibling, which predates iter/epivec and already
  measured worse). UNTESTED.
- **TOMOS FP16 (iter_epivec, align)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align_kernel`)
  -- single-knob fork off iter_epivec: re-pads A's own shared-memory row
  stride from 264 elements/row to 272 (inlined at 3 A-only use sites; B and
  the epilogue untouched). Motivated by ncu still showing the same excessive
  global-sector/shared-wavefront findings as deep4's own original profile
  (proving neither iter nor epivec touched this), traced to As's row stride
  not being a multiple of 32. Proves algebraically that 32-byte alignment and
  the padding's original bank-conflict-avoidance purpose are mutually
  exclusive -- a genuine, acknowledged trade-off (doubled bank-conflict
  period), not a strict win. UNTESTED.
- **TOMOS FP16 (iter_epivec, pad280)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280_kernel`)
  -- sibling of align, testing the other remaining untested valid pad
  residue given the 16-byte-alignment constraint (only 4 valid residues mod
  32 exist; 280 is the one with no existing real-hardware data point).
  Notably documents that a standard bank-conflict model was tried and FAILS
  to reproduce ncu's actual reported conflict figure for the current pad --
  explicitly framed as a genuinely empirical test since the standard model
  can't predict the right answer here. UNTESTED.
- **TOMOS FP16 (iter_epivec, aswizzle)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle_kernel`)
  -- fork off iter_epivec replacing A's shared-memory layout entirely with a
  no-padding, XOR-permuted "chunk" swizzle, described in-comment as
  "structurally modeled on real CUTLASS's own A operand layout" but citing
  the WRONG CUTLASS layout name -- this premise is later stated (in
  acongruous's own header) to have been backwards. The comment itself is
  notably careful: verifies via ncu that A's read side is already 0%
  excessive under the current layout and that the flagged excess is 100%
  isolated to A's cp.async WRITE side, meaning a naive port of a read-
  oriented formula would target the wrong problem; also documents that a
  hand/Python bank-conflict simulation predicts the current layout is
  already at the theoretical minimum, directly contradicting ncu's measured
  excess -- "a genuinely empirical test," not a proven fix. Bijectivity/
  write-contiguity/read-match verified via full Python sweep. **Its own
  header explicitly says UNTESTED: no performance measurement or correctness
  verification has been run yet -- see this document's own flagged
  discrepancy note above regarding a LATER kernel's comment that
  contradicts this.**
- **TOMOS FP16 (iter_epivec, acongruous)** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_kernel`)
  -- the corrected replacement for aswizzle's self-invented scheme; matches
  this document's already-verified description of the `acongruous` suffix
  exactly (see the promoted-kernel entries). Header confirms real CUTLASS
  does NOT apply a crosswise/swizzled scheme to A at all --
  `SmemLayoutA = ColumnMajorTensorOpMultiplicandCongruous<16,64>`,
  `SmemLayoutB = ColumnMajorTensorOpMultiplicandCrosswise<16,32>` (B already
  gets the swizzle, matching this family's existing B-side indexing, which
  is why B already measured 0% excess); A gets the simpler Congruous scheme
  instead, obtained by reading real CUTLASS source directly ("not recalled
  from memory or reverse-engineered from SASS -- both were tried first for
  aswizzle and led to an entirely wrong, self-invented scheme"). Derivation
  worked through the real formula by hand, then verified via full Python
  sweep for bijectivity/write-contiguity/read-match, same rigor as aswizzle.
  **UNTESTED itself -- explicitly "no performance measurement or correctness
  verification has been run yet."** (Its own comment explicitly cautions
  that aswizzle's cited outcome should NOT be taken as predictive of this
  kernel's, since acongruous implements a structurally different, verified-
  correct formula rather than aswizzle's wrong-premise one.)

*Chapter 4: acongruous's direct descendant chain to the two promoted kernels*

- **TOMOS FP16 (acongruous, pipe retest)** (`..._acongruous_pipe_kernel`) --
  a retest of the earlier failed `pipe` technique on top of the FIXED
  acongruous baseline instead of the old overhead-heavy `iter_epivec`
  baseline pipe originally lost on, motivated by ncu now showing the two
  hottest HMMA instructions dominated by `stall_short_sb` (waiting on a
  just-issued ldmatrix result) -- exactly the dependency pipe targets.
  Correctness rests on composing two independently-proven changes touching
  disjoint kernel parts -- "GENUINELY EMPIRICAL on the performance side."
  UNTESTED.
- **TOMOS FP16 (acongruous2)** (`..._acongruous2_kernel`) -- splits
  acongruous's A-address formula into a kk-only "precompute" half (hoisted
  once per thread) and an m-only "finish" half, applying deep4_iter's
  established address-hoisting principle to acongruous's more complex
  real-CUTLASS formula for the first time. Motivated by a measured 9.6%
  extra dynamic-instruction overhead of acongruous vs. an older kernel at
  small N -- a per-tile overhead from the complex divide/mod/XOR formula
  that large-N coalescing wins increasingly dominate, but costs more at
  small N. Correctness algebraically guaranteed via full Python sweep, zero
  mismatches -- "a pure algebraic factoring, not a new formula." UNTESTED.
- **TOMOS FP16 (acongruous, stages3)** (`..._acongruous_stages3_kernel`) --
  reduces the pipeline from 4 stages to 3, applying the same prefetch-
  distance/priming change as an earlier stages3 sibling, but now layered on
  acongruous's real-CUTLASS A-formula rather than the old row-major+pad
  layout. Motivated because the old result was measured on a kernel with a
  totally different A shared-memory write pattern, and acongruous's formula
  genuinely changed A's cp.async latency/bank-conflict profile, so the old
  result isn't assumed to transfer. Called "a cheap, low-risk test." UNTESTED.
- **TOMOS FP16 (acongruous, ldgsts)** (`..._acongruous_ldgsts_kernel`) --
  restructures `load_tile_async` so its cp.async calls are issued
  one-at-a-time, interleaved into the prefetch's target MMA sub-block loop,
  instead of bursting all before a single DEPBAR (acongruous's baseline
  pattern). Motivated by directly diffing compiled SASS (static binary
  inspection, no GPU execution) of acongruous against real CUTLASS: CUTLASS's
  own mainloop issues loads one-at-a-time each followed by an independent
  HMMA, while acongruous stays clustered even though nothing data-dependent
  forces that -- confirming this originates from CUTLASS's own template
  source, not backend scheduling. "GENUINELY EMPIRICAL" -- matching
  instruction order isn't guaranteed to close CUTLASS's remaining gap.
  Explicitly UNTESTED, and its own comment states the comparison itself used
  no GPU execution.
- **TOMOS FP16 (acongruous, fastk)** (`..._acongruous_fastk_kernel`) --
  continues from plain acongruous (not any of the three forks above).
  Deletes `load_tile_async`'s boundary-safety branch entirely rather than
  predicating it off; only launched when the host wrapper has verified k is
  an exact multiple of the K-block, making the boundary case compile-time-
  unreachable -- this is the `fastk` lever cited in the promoted kernel's
  own entry. Motivated by a cuobjdump instruction count comparing
  acongruous's compiled loop (474 instructions/K-tile) vs. real CUTLASS's
  (140), despite identical HMMA/LDSM counts, concentrated in scalar/
  predicate/branch overhead because the compiler can't prove only the safe
  branch is reachable when the safety bools are runtime values. UNTESTED.
- **TOMOS FP16 (fastk, precomp)** (`..._acongruous_fastk_precomp_kernel`) --
  from fastk. Precomputes A's entire swizzled offset ONCE per thread before
  the mainloop, for the small fixed set of (m,kk) combos actually visited --
  reducing every steady-state A access to an add, zero divide/mod/xor in the
  hot path. Motivated by two PRIOR NEGATIVE compile-only experiments (that
  both produced byte-identical SASS to baseline, proving ptxas already does
  intra-call CSE) that this kernel instead avoids by hoisting across the
  ENTIRE outer K-loop, a harder invariance proof that DID survive: mainloop
  instructions/iteration dropped from fastk's 290 to 275 (42% total vs.
  original acongruous). Register count mixed by SkipBeta variant. Explicitly
  compile-only static analysis, UNTESTED on GPU.
- **TOMOS FP16 (fastk_precomp2)** (`..._acongruous_fastk_precomp2_kernel`) --
  from fastk_precomp. Extends the same precomputed-offset technique to B's
  crosswise index, which turns out even more invariant than A's -- only 6
  distinct offsets needed for the entire mainloop. This is the `precomp2`
  lever cited in the promoted kernel's own entry. Compile-only: 275->247
  instructions/iteration (47.9% total vs. acongruous's 474). Later described
  as "the standing champion of this family." UNTESTED on GPU (compile-only
  static analysis only).
- **TOMOS FP16 (fastk_precomp3)** (`..._acongruous_fastk_precomp3_kernel`)
  -- a sibling fork off fastk_precomp2 (NOT part of the promoted chain).
  Changes how `__pipeline_wait_prior`'s argument reaches the DEPBAR.LE
  instruction: although algebraically/simulation-proven the value only ever
  takes 3 possible values, DEPBAR.LE needs a compile-time literal and the
  compiler couldn't see that invariant, generating a 9-case defensive
  dispatch. A `__builtin_assume` hint was tried first and produced byte-
  identical SASS (didn't survive); this kernel instead splits the call site
  into 3 separate literal-arg calls. Compile-only: "by far the largest
  single reduction of this entire investigation," 247->182 instructions/iter
  (61.6% total vs. acongruous). **Its own comment states UNTESTED at time of
  writing, but a LATER kernel's comment reveals it WAS subsequently
  GPU-tested and REGRESSED: "a real, GPU-measured regression" -- and per yet
  another later kernel's comment, "precomp3 had by far the best static
  numbers of this whole investigation and still lost at 5 of 6 sizes."** The
  family's explicit cautionary example that static instruction counts stop
  predicting runtime performance once overhead is no longer the bottleneck.
- **TOMOS FP16 (fastk_precomp2, pipe)** (`..._acongruous_fastk_precomp2_pipe_kernel`)
  -- a sibling fork off fastk_precomp2 (NOT part of the promoted chain,
  explicitly built on "the standing champion... NOT precomp3, whose own
  outstanding-dispatch change is a real, GPU-measured regression") +
  `pipe`'s B-load interleaving technique, ported onto this leaner base. Real
  ncu profiling on fastk_precomp2 now shows a fundamentally different regime
  (stall_wait/stall_math concentrated on the HMMA instructions themselves,
  address/branch overhead already driven down to match CUTLASS's own HMMA/
  LDSM counts exactly), motivating a retest of pipe's hypothesis in this new
  compute/latency-bound regime. Compile-only register count matches
  precomp2's, no spilling, but explicitly "CANNOT predict the outcome here"
  per the precomp3 lesson. "GENUINELY UNCERTAIN" and UNTESTED.
- **TOMOS FP16** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`) —
  this codebase's own tuned native fp16 GEMM, issuing real tensor-core
  `mma.sync.aligned.m16n8k16` PTX instructions directly (confirmed by
  inline `asm volatile` in the kernel body, not a `wmma::` wrapper).
  Evolved through a long chain of isolated single-lever experiments (each
  suffix names one change kept over its predecessor):
  - `wide_bigreg`: a large per-warp accumulator tile (bigger register
    footprint per thread than CUTLASS's own 128, by design, to test
    whether matching CUTLASS's actual tile/register shape mattered).
  - `deep4`: a 4-stage `cp.async` software pipeline (vs. this family's
    shared 3-stage baseline).
  - `iter`: hoists per-`K`-tile address/index computations (`a_kg * ld_a`,
    `b_col0 * ld_b`, and several thread-constant lane indices) out of the
    per-iteration load lambda into precomputed/incrementally-updated
    values, cutting redundant `IMAD`/`LEA` address arithmetic that
    `cuobjdump`-level comparison against real CUTLASS showed this family
    was paying 3-4x more of, despite identical tensor-core instruction
    counts.
  - `epivec`: the epilogue (writing accumulated results from shared memory
    to `C`) uses vectorized 8-half `float4` transfers instead of scalar
    element-by-element stores -- a fixed per-block cost independent of `K`,
    so its relative benefit is largest at small/medium `N` and fades as
    `K` (and thus mainloop work) grows.
  - `acongruous`: matrix A's shared-memory addressing uses CUTLASS's own
    verified `ColumnMajorTensorOpMultiplicandCongruous<16,64>` layout
    (read directly from CUTLASS's real source, not guessed) -- **not** a
    warp-divergence or iteration-order change, a shared-memory
    address-swizzle scheme matching operand A's real memory layout.
  - `fastk`: the boundary-safety branch for a partial last K-tile is
    deleted entirely (not just predicated off) -- only ever launched when
    the caller has separately verified `k` is an exact multiple of the
    K-block, making the boundary case compile-time-unreachable.
  - `precomp2`: precomputed address-offset tables for *both* operand A's
    and operand B's indexing (a fixed, small set of offsets computed once
    before the main loop, since the underlying per-thread call pattern
    turns out fully predictable), further cutting address-computation
    instructions.
  - `serp`: a serpentine (boustrophedon) reordering of the `mma.sync` call
    sequence within each compute block -- the exact same operands/
    accumulator targets, just visited in a different order, chosen to
    match real CUTLASS's own pattern for maximizing consecutive-call
    operand reuse in the tensor core's `.reuse` hardware cache (confirmed
    via compiled-SASS inspection: `.reuse` hit rate rises from 75% to 97%
    with zero added instructions).

  No decode step -- operates on plain unpacked storage, and is the tail
  call the matching "(workspace)" kernel below calls after decoding.
- **TOMOS FP16 (serp, ildsm)** (`..._acongruous_fastk_precomp2_serp_ildsm_kernel`)
  -- descends from the promoted serp kernel ("the standing champion").
  Repositions individual `ldmatrix` calls to interleave 1-to-1 with HMMA
  instructions from a different, already-ready accumulate block, instead of
  an 8-LDSM burst before each block starts -- matching a pattern read
  directly out of real CUTLASS's own compiled mainloop. Motivated by real
  ncu profiling: CUTLASS shows the opposite stall_math/stall_wait emphasis
  vs. this family's own, even after HMMA/LDSM counts already matched exactly
  and `.reuse` was maximized (serp's own 62/64 reuse). Uses the SAME
  precomputed address tables from fastk_precomp2 -- "purely a
  REPOSITIONING," not new computation. LDGSTS placement deliberately left
  unchanged since that specific class of change already regressed once
  (the ldgsts kernel above). Compile-only confirmed the interleaving
  survives into SASS: genuine 1-to-1 alternation, mainloop instructions
  dropped further to 229, `.reuse` unchanged at 62/64. Explicitly "GENUINELY
  UNCERTAIN," "the most structurally invasive change in this entire family's
  history," and states the correctness check "matters more than usual."
  UNTESTED.

*Chapter 5: forks parallel to acongruous (pipe/gridswap/cutlassdims/bk64/bf16), plus the separate bigwarp/multistage family*

- **TOMOS FP16 (deep4, iter2, epivec)** (`..._deep4_iter2_epivec_kernel`) --
  builds on deep4_iter_epivec and applies deep4_iter's own hoisting
  principle a SECOND time, to the ldmatrix fragment-loading lambdas iter's
  original fix never touched -- hoisting lane-derived constants and
  call-invariant arrays, "same proof technique as deep4_iter's own hoist,"
  contrasted explicitly with the speculative/unproven reordering of the pipe
  kernel. UNTESTED.
- **TOMOS FP16 (iter_epivec, gridswap)** (`..._deep4_iter_epivec_gridswap_kernel`)
  -- byte-for-byte identical to deep4_iter_epivec except which grid axis
  maps to M vs N tiles (a pure relabeling, cannot introduce a correctness
  bug). Motivated by an ncu finding that 99.4% of A's own "excessive global
  sector" count sits in its cp.async load, with no per-thread addressing bug
  found on re-derivation -- pointing at a grid-level/L2-residency hypothesis
  instead, though the performance mechanism is explicitly flagged as
  unverified/plausible-only. UNTESTED.
- **TOMOS FP16 (wide_bigreg, cutlassdims)** (`..._wide_bigreg_cutlassdims_iter_epivec_kernel`)
  -- forks off plain wide_bigreg directly (NOT the _sq branch), a genuinely
  new tile geometry matching real CUTLASS's `ThreadblockShape<128,256,32>`
  exactly (the transpose of this family's usual BM/BN), confirmed by
  inspecting the config actually used in the benchmark file. Re-derives the
  address-hoisting and vectorized-epilogue fixes for the new geometry (not
  reused as-is). Motivated by the same L2-excess investigation as gridswap,
  after gridswap's own unverified result ruled out a pure grid-axis
  explanation. **A LATER kernel's comment quotes its real-hardware result:**
  severe, size-growing loss (-24.4% at N=32768, -25.3% at N=65536)
  attributed to wave quantization against the A100's 108 SMs.
- **TOMOS FP16 (cutlassdims, 3-stage)** (`..._wide_bigreg_cutlassdims3_iter_epivec_kernel`)
  -- forks off cutlassdims_iter_epivec, changing only pipeline stages from 4
  to 3, matching CUTLASS's real mainloop as confirmed via direct cuobjdump
  inspection of the built CUTLASS reference binary -- completing the "full
  CUTLASS-matching config." Explicitly stated to NOT address cutlassdims'
  own large-N wave-quantization regression (a grid-size effect, independent
  of stage count) -- done for completeness of the comparison, not because a
  fix was expected. UNTESTED.
- **TOMOS FP16 (wide_bigreg_sq, deep5)** (`..._wide_bigreg_sq_deep5_kernel`)
  -- forks directly off wide_bigreg_sq_deep4, one pipeline stage deeper (5).
  Motivated as untried territory since the analogous packed-fp32_16 sibling's
  4->5 step was a measured loss, but this kernel has no decode step so the
  trade-off wasn't assumed to repeat. MEASURED real hardware: the regression
  DOES transfer -- loses to deep4 at every N>=2048 (-3.2% to -8.8%), 0
  register spill in both, so not spill-driven. deep4 remains best.
- **TOMOS FP16 (deep4, occ2)** (`..._deep4_occ2_kernel`) -- forks off deep4,
  cuts BN in half and targets `__launch_bounds__(...,2)` instead of deep4's
  `(...,1)` -- direct analog of a packed-fp32_16 sibling kernel. Motivated
  by ncu finding deep4 at only 12.49% achieved occupancy, both register- and
  shared-mem-occupancy-limited, with the dominant stall a fixed-latency
  execution dependency. MEASURED real hardware: a genuine split -- a real
  win at N=1024 (+60.4%), then a worsening loss at every larger size (down
  to -38.7% at N=8192), attributed to light register spill once launch-
  underutilization is no longer the bottleneck. deep4 remains best for
  N>=2048.
- **TOMOS FP16 (deep4, bigmma)** (`..._deep4_bigmma_kernel`) -- forks off
  deep4, reusing its exact tile/shared-memory layout unchanged. NOT an
  MMA-instruction-shape change -- restructures the mainloop to load both
  K-substeps' fragments together and issue all 64 mma.sync instructions as
  one flat, un-barriered block (instead of deep4's own two 32-instruction
  halves split by a mid-tile sync), aiming to give ptxas a single larger
  pool of independent instructions to schedule fixed-latency-hiding within.
  MEASURED real hardware: clear negative result, worse than deep4 at every
  N>=2048 (-29.4% to -36.6%), 0 spill (not spill-driven). Root cause per the
  comment's own post-hoc analysis: the restructuring inadvertently
  eliminated a cross-iteration software-pipelining benefit deep4's
  split-load design already provided. deep4 remains best.
- **TOMOS BF16 (plain deep4)** (`gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_kernel`)
  -- comment states explicitly: "bfloat16 sibling of
  `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_kernel` -- same tile, same 4-stage
  pipeline, same load/decode-free/compute/epilogue structure," with
  `__nv_bfloat16` swapped in for `__half` throughout and the `mma.sync`
  operand-type suffix changed `.f16.f16`->`.bf16.bf16` (accumulator stays
  f32; same m16n8k16 shape/latency class on sm_80). A straight port,
  confirming the framing in the promoted-bf16 entry ("the family's prior
  plain deep4 bf16 kernel"). First use of `cuda_bf16.h` in the codebase,
  added "per explicit request," reusing deep4's index-math helpers directly
  since they're pure int/size_t arithmetic with no dtype dependence.
  UNTESTED -- explicitly "not benchmarked or profiled before being handed
  over," no claim made about throughput vs. fp16 deep4.
- **TOMOS BF16** (`gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`) —
  a straight, mechanical port of the TOMOS FP16 kernel's own
  `iter`/`epivec`/`acongruous`/`fastk`/`precomp2`/`serp` refinement stack
  immediately above: same tile-geometry constants and index-math helpers,
  just `__half` swapped for `__nv_bfloat16` and the `mma.sync` PTX
  instruction's `f16.f16` suffix swapped for `bf16.bf16` (confirmed by the
  benchmark driver's own header comment, which states this was measured
  1.17x-1.34x faster than the family's prior plain `deep4` bf16 kernel at
  every size N>=2048). No decode step -- operates on plain unpacked
  storage, and is the tail call "TOMOS FP32 / BF16" below calls directly
  (via a raw pointer reinterpret, not a real decode).
- **TOMOS FP16 (wide_bigreg_sq, bk64)** (`..._wide_bigreg_sq_bk64_kernel`)
  -- forks directly off deep4 (same tile/warp/frag/accumulator shape) but
  BK=64 instead of 32 -- half the outer tile iterations for the same K
  range. Forced to 3 stages (not deep4's 4: a 4-stage BK=64 pipe exceeds the
  A100's dynamic shared-memory limit) -- still more total K look-ahead than
  deep4. Required real re-derivation of two index-math pieces tuned for
  BK=32's bit-width (a genuine silent-corruption bug traced and fixed in B's
  swizzle formula during the process, not just a perf issue), plus an A-side
  fix adding an outer 2-iteration "band" loop. Register-level pipelining
  kept deep4's exact 2-phase shape (explicitly to avoid bigmma's flattening
  mistake), widened from 1+1 K-substeps to 2+2. Comment flags (not resolves)
  a real register risk from doubling K-substep buffering. MEASURED real
  hardware: the flagged risk materializes exactly -- correct, but a
  consistent loss vs deep4 at every N>=2048 (-29.0% to -39.9%), 255/255
  registers with real spill traffic. deep4 remains best; the index-math
  re-derivation itself is stated as verified-correct (the loss is purely
  register-pressure driven, not a derivation error).
- **TOMOS FP16 (bk64, reg2)** (`..._wide_bigreg_sq_bk64_reg2_kernel`) --
  forks off bk64 directly, reusing its exact 3-stage BK=64 tile/index math
  unchanged, but restructures K-substep register buffering to stay at
  deep4's shallow 2-live-slot depth instead of bk64's 4-deep buffering --
  cycling through the same 4 K-substeps via a load/MMA/overwrite rotation
  that keeps every load staggered at least one full MMA block ahead of its
  first use (deliberately avoiding bigmma's mistake). MEASURED real
  hardware: confirms the register diagnosis -- 46-76% faster than bk64 at
  every N>=2048, registers land exactly at deep4's scale, 0 spill. Not a
  clean win over deep4 though: ties/slightly trails at N>=8192 (-5.8% to
  -6.7%), but actually BEATS deep4 at N=4096 (+5.8%) and is close at
  N=2048. Remaining gap at large N attributed to bk64's inherently 4
  load/compute/overwrite transitions per tile vs. deep4's 2 (not a spill
  story). Net: deep4 remains best for N>=8192, this kernel a close second
  and the better choice around N=4096.
- **TOMOS FP16 (bk64_reg2, iter+epivec)** (`..._wide_bigreg_sq_bk64_reg2_iter_epivec_kernel`)
  -- forks off bk64_reg2, keeping its register-fixed 3-stage BK=64 mainloop
  completely untouched, and layers on two fixes ported from the deep4_iter_
  epivec side of the family: address-hoisting (extended to bk64's two-band
  A structure) and the vectorized epilogue (reused verbatim, doesn't depend
  on BK). Motivated because deep4_iter_epivec's fixes were previously
  verified as additive/orthogonal wins on top of plain deep4 across the
  whole N range, so this tests whether applying the same two fixes on top
  of bk64_reg2 closes enough ground to compete with deep4_iter_epivec
  itself. UNTESTED -- no performance or correctness run yet.
- **TOMOS FP16 (wide_multistage)** (`gemm_fp16_mma_ptx_wide_multistage_kernel`)
  -- forks off `wide_epi_wide` (reuses its B-fragment load pattern). A first
  version literally ported CUTLASS's own double-buffered technique and
  MEASURED a clear failure (~5.6x worse, real spilling at this kernel's
  64-register/1024-thread ceiling). Redesigned as a "register-conscious
  rolling pipeline" that extends `wide_epi_wide`'s single-B-pair-ahead
  prefetch discipline across the tile boundary instead of adding a second
  buffer.
- **TOMOS FP16 (deep5, bigwarp)** (`gemm_fp16_mma_ptx_deep5_bigwarp_kernel`)
  -- explicitly "a genuinely unexplored point in the design space, not a
  variant of anything tried so far," grounded in two pieces of real
  evidence: an nsys trace of cuBLAS's actual dispatched kernel gave a
  128x128/BK=32/5-stage tile (reuses deep5's own constants), and real
  CUTLASS source showed a 128x128 threadblock split into only a 2x2 warp
  grid (4 warps/128 threads, not 32) -- new constants deliberately trade
  occupancy for per-warp compute granularity, banking on the 5-stage
  pipeline to hide latency. Its own comment states expected low occupancy
  as a real risk but gives no number; the actual measured throughput
  (~40-48K GFlop/s, losing decisively to CUTLASS's ~226-240K) is disclosed
  retroactively in a later sibling's comment.
- **TOMOS FP16 (bigwarp, apipe)** (`gemm_fp16_mma_ptx_deep5_bigwarp_apipe_kernel`)
  -- direct fork of bigwarp. Comment states bigwarp "matched CUTLASS's real
  tile:warp ratio... but still lost decisively" -- matching tile dimensions
  alone didn't transfer without matching scheduling. Reading CUTLASS's real
  mainloop shows it register-double-buffers fragments for BOTH A and B;
  bigwarp already did the lean one-ahead version for B but reloaded all of A
  fresh every step. apipe adds register-level double-buffering for A only
  ("easily affordable given bigwarp's huge headroom"), testing whether that
  alone fixes what tile-matching couldn't.
- **TOMOS FP16 (bigwarp, fullpipe)** (`gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe_kernel`)
  -- forks off bigwarp_apipe. Real CUTLASS reference numbers reveal bigwarp's
  tile/warp shape is nearly identical to CUTLASS's own, yet bigwarp measured
  ~2x slower than CUTLASS -- and apipe's A-side double buffering "barely
  moved the needle," an incomplete port of CUTLASS's pipe state. fullpipe
  finishes the port by also double-buffering B at the register level.
  Deliberately does NOT add wide_spread's spread-cp.async-issuance
  technique, reasoning bigwarp's genuine 5-stage ring buffer already gives
  far more slack than wide's shallow 2-stage buffer had. Explicitly
  UNTESTED: "deliberately not benchmarked in this build."
- **TOMOS FP16 (bigwarp2)** (`gemm_fp16_mma_ptx_deep5_bigwarp2_kernel`) --
  forks off bigwarp_fullpipe. A direct ncu comparison at N=16384 pinpointed
  the real gap: fullpipe runs at 6.25% occupancy and only 15.66% Compute(SM)
  Throughput, stalled on fixed-latency, while CUTLASS's real kernel for the
  same problem runs at 12.5% occupancy and 94.94% Compute(SM) Throughput
  (compute-bound, a "good" stall profile) -- CUTLASS packs twice the warps
  per block at near-identical register count. bigwarp2 isolates exactly
  that variable: doubles BN, widens the warp grid 2x2->2x4, per-warp tile
  unchanged. Deliberately does NOT yet fix the uncoalesced epilogue or
  pipeline depth (both real, ncu-flagged, but isolating one variable at a
  time). Explicitly UNTESTED: "not benchmarked in this build, per request."
- **TOMOS FP16 (bigwarp2, epismem)** (`gemm_fp16_mma_ptx_deep5_bigwarp2_epismem_kernel`)
  -- forks off bigwarp2. bigwarp2's occupancy doubling "confirmed: +49-57%
  real throughput at N>=4096" but left a ~3x gap; still has the original
  direct/uncoalesced epilogue (unlike `wide`, never fixed for this family).
  Applies the same shared-memory-staged coalesced epilogue fix already
  validated on `wide` (reusing As/Bs's now-dead space). No specific number
  given in this excerpt (self-referential comment).
- **TOMOS FP16 (bigwarp2, hoisted)** (`gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted_kernel`)
  -- forks off bigwarp2 (address-hoisting layered independently of
  epismem). Direct SASS comparison vs CUTLASS found ISETP+IMAD at 37.7% of
  instructions vs CUTLASS's 11.9%, despite near-identical HMMA counts,
  caused by per-thread-invariant address components recomputed from scratch
  on every call. Hoists that arithmetic before the main loop; bounds checks
  explicitly left fully dynamic for correctness.
- **TOMOS FP16 (bigwarp3)** (`gemm_fp16_mma_ptx_deep5_bigwarp3_kernel`) --
  forks off bigwarp2_hoisted. SASS inspection found a costly reciprocal-
  multiply chain for `t % 5` (deep5's non-power-of-2 stage count, originally
  chosen to match cuBLAS's traced name), "the largest single contributor to
  the 69 IMAD instructions" in the loop. Switches to 4 stages so the modulo
  becomes a cheap AND-mask; notes an earlier wide_deep5 experiment already
  showed going one stage shallower "doesn't cost anything on its own when
  occupancy isn't already the bottleneck."
- **TOMOS FP16 (bigwarp4)** (`gemm_fp16_mma_ptx_deep5_bigwarp4_kernel`) --
  direct fork of bigwarp3. bigwarp3's stage fix cut the K-loop from 317 to
  297 instructions for the same 64 HMMA, but still far from CUTLASS's 2.19
  instr/HMMA. Further SASS inspection found `col*ld_b`/`kg*ld_a` recomputed
  from scratch on every call despite being thread-invariant or advancing by
  a fixed stride -- "exactly CUTLASS's own iterator pattern." Hoists/
  replaces with a running accumulator advanced by pure addition.
- **TOMOS FP16 (bigwarp5)** (`gemm_fp16_mma_ptx_deep5_bigwarp5_kernel`) --
  forks off bigwarp4. bigwarp3/4's instruction-count fixes bought only ~1%
  real throughput despite a ~2x instr/HMMA gap -- no longer instruction-
  count-bound, now ILP/scheduling-bound (CUTLASS hits 94.9% Compute(SM)
  Throughput at the same occupancy via better instruction-stream
  scheduling). First attempt to faithfully replicate CUTLASS's actual
  mainloop-iteration logic on two fronts: (1) spread next-tile cp.async
  issuance across the ks-loop instead of one upfront burst -- kept, noting
  `wide_spread` tried the same idea and lost to register pressure on wide's
  zero-slack budget, but bigwarp has far more headroom; (2) cross-t-
  iteration fragment pipelining was also attempted but REVERTED after
  compute-sanitizer racecheck caught a genuine data race -- any earlier
  throughput gain attributed to (2) is flagged as an artifact of the race's
  undefined timing, not a real, safe win. Only mechanism (1) ships. Also
  documents a separate, unrelated pre-existing race in a shared priming
  helper, confirmed present already in bigwarp3, not introduced here.
- **TOMOS FP16 (bigwarp6)** (`gemm_fp16_mma_ptx_deep5_bigwarp6_kernel`) --
  forks off bigwarp5. bigwarp5's ncu profile showed L1TEX-scoreboard as the
  top stall, the same uncoalesced-B-load issue this family never fixed
  (unlike `wide`); a register-shuffle fix "isn't safely affordable here"
  since bigwarp5 is already at the 255-register ceiling. Implements what
  CUTLASS actually does instead, sourced directly from CUTLASS's real
  layout header: for the reference config, B's shared-memory layout is a
  compile-time XOR address swizzle (not a runtime shuffle), confirmed from
  the compiled CUTLASS kernel's own mangled name. Comment explicitly argues
  correctness is preserved even if the swizzle doesn't bit-match CUTLASS's
  own choice, since XOR partition swizzles are self-inverse bijections and
  only the write site/read site need to agree with each other.
- **TOMOS FP16 (wide, spread cp.async)** (`gemm_fp16_mma_ptx_wide_spread_kernel`)
  -- a separate fork off the `wide` lineage (uses wide's own tile, NOT part
  of the bigwarp family). Grounded in CUTLASS's actual mainloop, "read
  verbatim via curl, not an AI-summarized paraphrase -- the summaries missed
  exactly this detail": CUTLASS spreads cp.async issuance across the ks-loop
  rather than one upfront burst (which is what wide's original load does).
  MEASURED (A100): register count unchanged, but spill jumps from wide's
  4 bytes to 136/196 bytes; real throughput lands ~30% below the family's
  then-best kernels at every N tested, consistently, with occupancy
  identical to wide -- isolating the cost as purely a spill regression from
  interleaving copy-issuance with compute on wide's tight 64-register
  budget (CUTLASS can afford the schedule because its register budget is
  built around it). Comment states: "wide_nospill_epismem remains the best
  kernel from this investigation."
- **TOMOS FP16 (wide, nospill+epismem+coalb combo)** (`gemm_fp16_mma_ptx_wide_nospill_epismem_coalb_kernel`)
  -- another separate fork off `wide`, sibling to wide_spread, also NOT part
  of bigwarp. Combines three previously-separate, orthogonal `wide`-family
  fixes never tested together (nospill2's unconditional wait,
  epi_smem's staged epilogue, and a coalesced B-tile global load with
  `__shfl_sync` redistribution). States earlier coalb/coalb_shfl kernels
  were only ever benchmarked against wide's OLD uncoalesced epilogue and
  thus never got a fair test: ncu on the then-best kernel showed
  uncoalesced global B-loads as the single largest remaining issue even
  though standalone coalb/coalb_shfl had lost in the earlier, unfair
  comparison. No inline numeric result for this specific combination
  (self-referential comment).
- **TOMOS FP16 (smalltile, 2 blocks/SM)** (`gemm_fp16_mma_ptx_smalltile_2block_kernel`)
  -- a structurally distinct idea, NOT part of the wide_bigreg or bigwarp
  lineages. Every kernel in the `wide`/`bigtile_w32` family is stuck at
  exactly 50% occupancy (1 block/SM) because the accumulator alone already
  saturates the register file for one block at 1024 threads; nothing in the
  investigation had ever run 2 resident blocks/SM. Reaching that needs a
  much smaller accumulator; the smallest non-degenerate step (FragM=1/
  FragN=1, a single 16x8 MMA/warp) gives a tile ~10x smaller per block than
  wide's. Per the roofline model used throughout the investigation, this
  implies ~3.8x more global memory traffic/flop than wide -- "a real,
  deliberate gamble in the OPPOSITE direction of every previous tile-size
  experiment" (shrinking the tile to grow occupancy, instead of the
  reverse). Reuses nospill2's unconditional wait and epi_smem's coalesced
  epilogue (free) but deliberately skips the B-load shuffle fix since it
  already showed a net loss elsewhere. No numeric measured result given;
  comment explicitly flags even the intended occupancy gain as unconfirmed:
  "whether ncu actually confirms 2 resident blocks/SM (fitting the register
  budget on paper doesn't by itself guarantee the occupancy calculator
  agrees)."
- **TOMOS FP32 / FP16** (`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp`) —
  same tensor-core kernel lineage as TOMOS FP16 above (`acongruous`/
  `fastk`/`precomp2`/`serp` mean the same things here -- confirmed by this
  kernel's own comment block, which calls it the "same three optimizations
  [congruous A-addressing, precomputed address tables, serpentine
  accumulate order]... applied on top of THIS family's own inline_dirty
  decode step instead of a plain `__half` A"), with one further change,
  `inline_dirty`: decode is fused directly into the staging loop, but
  batched rather than done element-by-element. Each thread first performs
  all of its shared-memory reads for this tile, OR-reducing their raw bits
  into one combined word and cheaply testing whether *any* of them have
  their control bit set (i.e. need real decoding at all); that per-thread
  result is then combined warp-wide via `__any_sync` specifically to avoid
  intra-warp divergence (this is where warp-uniform branching actually
  happens in this kernel family -- not in `acongruous`). Only if the whole
  warp's combined result says "yes" does it call
  `fp32_16_half_only_branchless_no_overflow_nomask` (safe on underflow,
  assumes no overflow) on that thread's own elements and write the decoded
  bits back to shared memory; if every element in the warp's data is
  already control-bit-0 ("clean"), both the decode and the write-back are
  skipped entirely, since the raw bits are already a valid half value.
  Confirmed directly in the kernel body (`decode_a_tile` lambda, source
  line ~37036 in `mm_cuda_kernels.cu`): the OR-reduce into `combined`, the
  `__any_sync(0xffffffffu, (combined & 0x00010001u) != 0)` check, and the
  `fp32_16_half_only_branchless_no_overflow_nomask` call are all present
  exactly as described. (Note: this kernel's own in-source comment is
  marked "UNTESTED", which appears to predate the actual hardware
  measurement the benchmark driver's own header comment references when
  promoting it -- the architecture described above is verified from the
  kernel body itself regardless of that stale annotation.)
- **TOMOS FP32 / FP16 (fast)** (`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp`) —
  byte-for-byte the same `inline_dirty` batched clean/dirty-check structure
  as the kernel above, but the dirty-path decode call is
  `fp32_16_half_only_fast` instead: this one does have a real branch
  (`if (!control)`), and for any element needing the underflow
  reconstruction path, saturates it to signed zero instead of recovering
  the genuine fp16 subnormal value -- lossy but always defined.
- **TOMOS FP32 / FP16 (workspace)** (`mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp`) —
  a two-pass approach: decodes the whole matrix up front via
  `mm_fp32_16_pure_half_materialize_chunk_vec_kernel` (vectorized `ushort4`
  loads / `uint2` stores, four consecutive rows -- i.e. four consecutive
  elements in this column-major layout -- decoded per thread in one shot
  via four calls to `fp32_16_half_only_branchless_no_overflow_nomask`,
  with a scalar fallback for a chunk's ragged final <4-row group) into a
  **cached** scratch buffer (a `static` device pointer, grown on demand and
  reused across calls rather than `cudaMalloc`/`cudaFree`-ed every call,
  the plain materialize kernel's dominant small-N cost), then calls the
  native TOMOS FP16 kernel above completely unmodified on that scratch
  buffer, chunked to bound scratch memory. Falls back to the scalar-decode
  cached kernel when `ld_a` isn't a multiple of 4 (can't safely issue
  aligned `ushort4` reads in that case).

*Naive / pre-tensor-core family (`fp32_16_cuda_kernels.cu`)*

This is a structurally separate, earlier evolutionary branch from the
`wide_bigreg_sq` `mma.sync`-PTX family below -- simple scalar/register-tiled
loops and `wmma::`-API kernels, all decoding only `wh` (never `wr`, hence
"pure_half") and computing in `__half`/fp16 tensor-core precision throughout,
which is why they belong in this "16-bit-compute" tier rather than the
32-bit-compute/"recon" tier above it in README_COMPLETE.md, even though their
A/C-side host interface is `float`.

- **`mm_fp32_16_pure_half_kernel`** (via host wrapper `mm_fp32_16_pure_half`)
  -- the simplest possible kernel in this family: one thread per output
  element, a scalar K loop, A decoded via
  `fp32_16_half_only_branchless_no_overflow_nomask`, accumulator kept in
  `__half` the whole way (`acc = acc + a_val * b_val`), only converted to
  `double`/`float` at the very final `alpha*acc + beta*C` combine. Has a
  byte-identical `experimental::conditional` twin using plain
  `fp32_16_half_only` instead (see "conditional decode A/B-test set" above).
- **`mm_fp32_16_pure_half_tiled_kernel`** -- same 128x128/BK=8, 8x8
  register-blocked tiling this file's plain-fp32 `mm_fp32_16_tiled_kernel`
  uses, but `As`/`Bs` are cached as `__half` and the whole accumulation runs
  in `__half` arithmetic instead of `float`, matching `mm_fp32_16_pure_half_
  kernel`'s own precision tier. Same conditional-decode-twin pattern as above.
- **`mm_fp32_16_pure_half_wmma_kernel`** -- first tensor-core kernel in this
  family, using the `wmma::` C++ API (not raw PTX `mma.sync`) with a real
  fp32 accumulator fragment. `__launch_bounds__(kMmFp32_16WmmaThreads, 2)`
  deliberately caps ptxas at 64 registers/thread (the natural 4-fragment
  version needs ~128, capping occupancy at 1 block/SM) to trade a few spilled
  registers for 2 blocks/SM (32 resident warps instead of 16). Output is
  staged through a small per-warp 16x16 shared buffer, reused sequentially
  across each warp's 4 fragments, instead of one 64KB block-wide buffer that
  would itself have capped occupancy.
- **`mm_fp32_16_pure_half_wmma_cpasync_kernel`** -- adds a `cp.async` load
  path for `wh`. Because `wh` (`uint16_t`, 2 bytes) and the decoded `__half`
  output are the same size, `wh`'s raw bytes can be `cp.async`'d directly
  into `As`'s own memory (reinterpreted as `uint16_t` for the transfer) and
  decoded in place once the transfer lands -- no second staging buffer is
  needed, unlike the fp32-compute tier's own cp.async kernel (which combines
  `wh`+`wr` into a wider value and therefore does need one). Decoding a
  zero-filled boundary tile unconditionally is safe:
  `fp32_16_half_only_branchless_no_overflow_nomask(0)` hits its `control=0`
  passthrough and returns `0.0f`. B stays a plain synchronous scalar load
  (same rationale used throughout this codebase: B's K-contiguous global
  layout conflicts with `Bs`'s N-fast compute layout). Requires `ld_a`/`m`
  both even; falls back to `mm_fp32_16_pure_half_wmma` otherwise.
- **`mm_fp32_16_pure_half_wmma_cpasync_vec_kernel`** -- widens both transfers:
  `wh`'s cp.async from 2 elements (4B) to 8 elements (16B) straight into
  `As`; B (plain `float`) widens to a `float4` (4 K-values, 16B) synchronous
  read, cast to `__half` when scattered into `Bs`. Ported from
  `gemm_fp16_wmma_cpasync_vec_kernel`'s own measured ~65% win there (B's
  unvectorized scalar load, not decode or the tensor cores, was the dominant
  cost). Requires `ld_a`/`m` multiples of 8 and `ld_b`/`k` multiples of 4;
  falls back to the plain cpasync kernel otherwise.
- **`mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_kernel`** -- same idea,
  K-block widened from 32 to 64 (`kMmFp32_16WmmaBigBK`).
- **`mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad_kernel`** -- bigbk +
  shared-memory padding (`kMmFp32_16WmmaBMPad`/`BNPad`) to reduce bank
  conflicts.
- **`mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk_kernel`** -- K-block
  widened further to 128 (`kMmFp32_16WmmaHugeBK`), `__launch_bounds__(...,1)`
  (drops the dual-occupancy target the smaller-BK siblings use, since the
  much larger per-block shared-memory footprint at BK=128 no longer leaves
  room for 2 resident blocks/SM).

*`wide_bigreg_sq` `mma.sync`-PTX tensor-core family (`mm_cuda_kernels.cu`)*

This is the direct lineage the 4 already-promoted kernels belong to. All of
these kernels reuse the native TOMOS FP16 kernel's own tile geometry
(`kGemmFp16WideBigRegThreads`/`BM`/`BN` etc.) and, where applicable, the same
`acongruous`/`fastk`/`precomp2`/`serp` refinements verified for the native
FP16/BF16 kernels elsewhere in this file (CUTLASS congruous A shared-memory
layout / compile-time exact-K dead-branch elimination / precomputed A+B
address tables / serpentine `mma.sync` reorder -- reused here exactly as
already verified for the native-kernel family, per this task's own briefing).

- **`mm_fp32_16_pure_half_wide_bigreg_sq_fallback_kernel`** /
  **`..._fast_fallback_kernel`** -- always-correct, one-thread-per-output-
  element scalar fallback kernels for any shape/transpose the tensor-core
  fast path below doesn't cover (non-`NO_TRANSPOSE`, misaligned dims, etc.).
  Decode is `fp32_16_half_only_branchless_no_overflow_nomask` (plain) /
  `fp32_16_half_only_fast` (fast) respectively -- mirrors the promoted
  kernel pair's own two decode tiers. The plain fallback has a conditional-
  decode twin (`fp32_16_half_only`); the fast fallback does not.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_kernel<SkipBeta, UseTileDirty>`**
  -- the base `mma.sync`-PTX tensor-core kernel this whole family branches
  from. One template body serves two host-callable variants:
  `UseTileDirty=false` is `mm_fp32_16_pure_half_wide_bigreg_sq` itself
  (`decode_a_tile` runs unconditionally every tile, per-element control-bit
  branch via the decode function); `UseTileDirty=true` is
  `mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty` (a companion
  `fp32_16_compute_tile_dirty_kernel` precomputes, per (block_row, k_tile),
  whether ANY element of that A slice has its control bit set, and decode is
  only invoked for tiles where it does -- a global "whole array clean" flag
  was tried first and rejected in favor of this per-tile granularity, per
  `fp32_16_compute_tile_dirty_kernel`'s own comment).
- **`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_kernel<SkipBeta>`** --
  a fully standalone sibling (NOT a template variant of the kernel above,
  shares no code path with it): identical load/mma/epilogue logic, but a
  deeper 4-stage `cp.async` pipeline instead of the shared 3-stage one, and
  the clean/dirty decision is made entirely inline per-thread per-8-element
  chunk from data already in registers -- no separate precompute kernel or
  tile_dirty buffer needed at all. This is the direct, un-refined precursor
  of the promoted kernel below (same `inline_dirty` mechanism the promoted
  kernel's own Method Description bullet already explains in full -- see
  that entry, unchanged here), just missing `acongruous`/`fastk`/`precomp2`/
  `serp`.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_kernel<SkipBeta>`**
  -- byte-for-byte identical to the kernel above except its dirty-path decode
  call is `fp32_16_half_only_fast` instead of `fp32_16_half_only_branchless_
  no_overflow_nomask`. Same fast/lossy relationship as the promoted pair.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_
  precomp2_serp_kernel<SkipBeta>`** -- **[PROMOTED, already documented in
  README_COMPLETE.md as "TOMOS FP32 / FP16"]**. Unchanged here; see the
  existing bullet.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_
  fastk_precomp2_serp_kernel<SkipBeta>`** -- **[PROMOTED, "TOMOS FP32 / FP16
  (fast)"]**. Unchanged here; see the existing bullet.
- **Materialize-family decode kernels** (two-pass: decode a strip of A into
  a scratch buffer, then call the native TOMOS FP16 kernel on it unmodified;
  all bounded to `kMmFp32_16PureHalfMaterializeChunkRows=4096` rows/chunk so
  total scratch memory stays O(chunk_rows*K) independent of M):
  - **`mm_fp32_16_pure_half_materialize_chunk_kernel`** -- scalar,
    one element decoded per thread. Two host wrappers share it: `..._gemm_
    materialize_fastk_precomp2_serp` (fresh `cudaMalloc`/`cudaFree` every
    call) and `..._gemm_materialize_cached_fastk_precomp2_serp` (a `static`
    scratch pointer grown on demand and reused, avoiding the allocator
    overhead the uncached version re-pays on every timed rep). Has a
    conditional-decode twin.
  - **`mm_fp32_16_pure_half_materialize_chunk_vec_kernel`** -- **[used by
    PROMOTED "TOMOS FP32 / FP16 (workspace)"]**. Unchanged here; see the
    existing bullet. Its host wrapper `..._gemm_materialize_cached_vec_
    overlap_fastk_precomp2_serp` is a separate, additional experiment that
    reuses this SAME kernel unmodified but overlaps chunk c+2's decode with
    chunk c's GEMM via a second `cudaStream_t` and two ping-ponged scratch
    buffers (standard double-buffered producer/consumer hazard handling via
    `cudaStreamWaitEvent`). MEASURED: correct (passes its own hazard tests)
    but not a win anywhere and a clear -2.4% regression at N=8192 -- ncu
    profiling at N=2048 found decode's own duration already accounts for
    the entire gap to native GEMM (no launch/dependency bubble to hide), and
    the GEMM kernel's `__launch_bounds__(...,1)` already claims every SM for
    its full duration, leaving the second stream's decode launch nowhere to
    actually run concurrently -- the double buffering is semantically
    correct but has no real SM capacity to exploit on this GEMM kernel's
    occupancy profile. Not wired into any dispatcher.
  - **`mm_fp32_16_pure_half_materialize_chunk_vec64_kernel`** -- same 4-
    row/thread `ushort4` read as the vec kernel, but computes the 64-bit
    output as ordinary 64-bit integer arithmetic so it compiles to a single
    `STG.E.64` instead of two `STG.E`s (confirmed via SASS re-inspection).
    MEASURED: no real effect vs. `materialize_cached_vec` -- every delta
    across the full square sweep fell within this file's own ~1% run-to-run
    noise band; changes instruction issue count, not memory transactions
    (the touched addresses were already fully coalesced either way). Has a
    conditional-decode twin. Host wrapper: `..._gemm_materialize_cached_
    vec64_fastk_precomp2_serp`.
  - **`mm_fp32_16_pure_half_materialize_chunk_dirty_kernel`** -- vec64's
    exact decode, plus a per-thread (all-4-elements) clean/dirty hoist: if
    all 4 of a thread's `ushort4` elements have `control=0`, it skips 4 calls
    to the decode function and does one raw `ushort4`->`uint2` bit copy
    instead (correctness-neutral, since a clean element's "decode" IS its
    own raw bits). Motivated by a host-only measurement of this file's own
    pseudo-random `[-1,1]` test data: 99.99355% clean across 200M samples.
    MEASURED: correct, but no clear win vs. vec64 -- deltas fell within
    noise; this is the THIRD distinct decode-kernel-internal optimization
    tested against this specific cost (after vec_overlap targeting launch/
    dependency bubbles and vec64 targeting store-instruction count) and all
    three showed no measurable effect, a consistent confirmation that
    materialize decode's cost is a bandwidth floor plus fixed launch
    latency, not anything happening inside the decode kernel. Has a
    conditional-decode twin. Host wrapper: `..._gemm_materialize_cached_
    dirty_fastk_precomp2_serp`.
  - **`mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp`** -- host-only
    size-threshold dispatcher, no kernel of its own: routes to
    `..._gemm_materialize_cached_vec_fastk_precomp2_serp` for N>=2048 and to
    `..._wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_
    perthread` (below) for N<2048. Threshold picked as the smallest size
    actually MEASURED to favor the cached-vec kernel (cached materialize
    beats perthread 1.04x-1.10x+ from N=2048 up, reaching 100.3% of native
    fp16 by N=32768; perthread still wins 0.91x at N=1024, where the extra
    kernel-launch overhead isn't yet amortized). Based on the same O(M*K)
    decode vs. O(M*K*N) GEMM cost-ratio reasoning as this codebase's
    (since-dropped) fp32-compute-tier dispatcher, but unlike that one this
    threshold's crossover was found to be monotonic (verified on one GPU).
- **The `inline_dirty_acongruous_fastk_precomp2_serp` sibling family**
  (all built directly on the promoted kernel, each changing exactly ONE
  thing about the `__any_sync` dirty-check mechanism, all confirmed via real
  `ncu` profiling data or documented as untested where noted):
  - **`..._perrep_kernel<SkipBeta>`** -- narrows the check from per-thread
    (perthread's own 4 bundled reps) to per-rep: each of a thread's 4 reps
    is checked and rewritten independently instead of one dirty rep dragging
    the other 3 along. MEASURED: a mild, consistent 2-4% REGRESSION vs.
    perthread/the promoted kernel -- the extra per-rep branch overhead (4
    independent predicated branches instead of 1) outweighs the further-
    reduced shared-memory-store traffic. Kept as a documented negative
    result; perthread is the right stopping point on this axis.
  - **`..._perthread_kernel<SkipBeta>`** -- drops `__any_sync` (the warp-wide
    vote) entirely; `decode_a_tile` still computes the same per-thread
    combined OR-reduce but branches on it directly, shrinking the "blast
    radius" of one dirty element from "the whole warp rewrites" to "just
    that thread rewrites." Motivated by `ncu` profiling of `alwaysdecode`
    (below), which confirmed the promoted kernel's per-warp vote really was
    doing valuable work-avoidance but traded it for elevated
    `stall_short_sb` (every thread doing a full LDS-read/STS-write cycle
    unconditionally). Source-commented "UNTESTED" at the point this kernel
    was written, but later referenced elsewhere in the file (the `auto`
    dispatcher's own comment, and `perrep`'s own comment) as measured
    "essentially a statistical tie" with the promoted kernel (within +-2% at
    every size, a small real edge at N=8192, no regression anywhere) --
    same stale-annotation pattern already flagged for the promoted kernel
    itself in README_COMPLETE.md's existing text.
  - **`..._perthread_leanreg_kernel<SkipBeta>`** -- perthread, but drops the
    held `raw[kAReps]`/`idx[kAReps]` arrays kept live across the dirty
    branch, re-reading from shared memory on the (rare, ~0.006%) dirty path
    instead, to test whether reducing live registers closes perthread's own
    17-register gap vs. native FP16 (251 vs 236 registers, compile-only
    check found almost no improvement -- 251 vs perthread's 253). MEASURED:
    a real, consistent regression vs. perthread (-3% to -7% at N=2048-32768,
    worst at N=8192) -- likely cause: recomputing idx in a second loop on
    the dirty path is more total address arithmetic than the original's
    single held pass. Closes this investigation axis: neither holding nor
    not-holding the tile beats materialize.
  - **`..._alwaysdecode_kernel<SkipBeta>`** -- removes the dirty check
    entirely; every thread unconditionally decodes every tile (no OR-reduce,
    no `__any_sync`, no `__syncthreads_or`, no tile-level branch at all).
    Motivated by real `ncu` profiling (N=4096) finding `stall_barrier` at
    15.18% vs native's 2.75%, attributed to warp-to-warp timing asymmetry
    right before the shared `__syncthreads()` that follows `decode_a_tile`.
    MEASURED: a REGRESSION, and a worse one than blockvote's (~31-36% slower
    than the promoted kernel at N=2048-32768, despite compiling to FEWER
    registers, 236/233 vs 253/253) -- the promoted kernel's per-warp vote
    was already skipping decode on most tiles in practice; removing it loses
    that work-avoidance benefit, which dominates the stall cost it removed.
  - **`..._blockvote_kernel<SkipBeta>`** -- replaces the per-warp
    `__any_sync` with `__syncthreads_or` (a block-wide OR-vote-plus-barrier
    CUDA builtin), so every warp learns the same "does ANY thread in the
    whole block need to decode" answer at the same point instead of each
    warp deciding independently, removing the warp-to-warp timing asymmetry
    feeding into the trailing `__syncthreads()`. MEASURED: a clean
    REGRESSION at every size (~20-24% slower than the promoted kernel,
    e.g. 147,460 vs 193,230 GFlop/s at N=4096) -- with 8192 elements checked
    per tile-decode call, even a small per-element dirty rate makes it
    likely SOME thread anywhere in the 256-thread block is dirty on nearly
    every tile, so the block-wide vote fires far more often than the
    per-warp vote would, and pays the full decode-and-rewrite cost for all
    256 threads almost every tile.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty_kernel<SkipBeta>`**
  -- a fully standalone sibling of the base `wide_bigreg_sq_kernel` (not
  built on `inline_dirty`'s own code path at all, shares no code with it or
  with `wide_bigreg_sq_kernel`): identical to `inline_dirty_kernel` in every
  respect except its dirty decision is read from a precomputed per-chunk
  dirty byte (`fp32_16_compute_chunk_dirty_kernel`, run by the caller first)
  instead of derived from `wh` itself inline -- removing both the
  `LDSM.16.MT88.4`-adjacent `stall_barrier` and the OR-reduce's own
  `stall_short_sb` from this kernel's own critical path, at the cost of
  requiring a separate precompute pass (NOT a drop-in same-signature
  replacement for `wide_bigreg_sq`). No specific throughput number is given
  in-source for this kernel alone; later comments (on `prefetched_dirty` and
  `blockvote`) group it among the family's documented negative results,
  behind `inline_dirty`/the promoted kernel.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized_kernel<SkipBeta>`**
  -- identical to `inline_dirty_kernel` in load/mma/epilogue/pipeline depth,
  but only warps 0-3 perform any decode work at all, each covering both its
  own original 1/8 share AND warp+4's share (a doubled loop over two
  "virtual warp id" values); warps 4-7 skip decode entirely and go straight
  to `load_a_frag`/`mma.sync`. Motivation: isolate the decode check onto a
  warp subset so the SM scheduler can interleave those warps' decode
  instructions with other warps' independent compute stream, rather than
  every warp serializing decode-then-compute internally. Source-commented as
  a genuine experiment with no confident directional prediction (the
  caveat: the block-wide `__syncthreads()` still waits for the slowest warp,
  and warps 0-3 now do DOUBLE decode work). No specific throughput number
  given in-source for this kernel alone; grouped by `prefetched_dirty`'s and
  `blockvote`'s own comments among the family's documented negative results.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty_kernel<SkipBeta>`**
  -- same precomputed per-chunk dirty byte as `precomputed_dirty` above
  (same `fp32_16_compute_chunk_dirty` precompute function, reused
  unchanged), but `load_tile_async` additionally prefetches each thread's
  own dirty bytes into shared memory via the same per-thread ownership
  mapping `wh`'s own cp.async load already uses, so the dirty check reads a
  shared-memory buffer instead of a global one. MEASURED: correct, and
  faster than `precomputed_dirty` (moving the check off the global-memory
  round trip did help, as predicted), but still consistently slower than
  `inline_dirty`/the promoted kernel at every size (e.g. 125,681 vs 176,162
  GFlop/s at N=8192) and slower than `warp_specialized` too -- structural,
  not tuning: every member of this "precomputed/prefetched dirty" sub-family
  introduces a whole separate `chunk_dirty` buffer (extra global loads,
  extra shared-memory footprint/traffic, extra writes) purely to
  reconstruct information `inline_dirty` already has for free from bits it
  loads anyway.
- **`mm_fp32_16_pure_half_wide_bigreg_sq_dualocc_kernel<SkipBeta>`** -- a
  fourth fully standalone sibling: A-side load/decode/compute copied
  verbatim from `inline_dirty_kernel` (unchanged logic, just locally
  renamed), but the B/N-side tile shrinks (BN 128->64,
  `kGemmFp16WideBigRegDualOccBN`) specifically to fit
  `__launch_bounds__(kGemmFp16WideBigRegThreads, 2)` -- 2 resident CTAs/SM
  instead of `inline_dirty`'s 1 -- trading register budget (confirmed real
  spilling by ptxas, 32-56B stack frames) for occupancy. MEASURED (real
  hardware): a genuine split result, not a uniform win/loss. At N=1024 (grid
  too small to saturate an A100 either way), dualocc WINS: 34,635 vs
  `inline_dirty`'s 31,347 GFlop/s, because more/smaller blocks plus 2
  CTAs/SM directly address the real bottleneck there (launch
  under-utilization) and outweigh the spill cost. At N=2048 and up (where
  block count already exceeds what's needed to fill the GPU), it's a clear
  and worsening loss -- 27% slower at N=2048, ~45-47% slower from N=4096
  through N=32768, plateauing at ~94-96k GFlop/s, the register-spill
  traffic becoming the per-tile-iteration cost ceiling once there's no
  launch-underutilization problem left to solve. `inline_dirty`/the
  promoted kernel remains the best kernel overall for the sizes this family
  targets; dualocc's N=1024 win is real but specific to that
  launch-underutilized regime.
- **TOMOS FP32 / BF16** (`mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp`) —
  no decode kernel exists for this one at all: bf16's exponent field
  matches fp32's own bit-for-bit, so the truncated `wh` half of the packed
  value is *already* a valid `__nv_bfloat16` bit pattern for any input,
  unconditionally. The kernel just reinterprets `wh` as `__nv_bfloat16*`
  directly (`reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(wh))`)
  and calls the native TOMOS BF16 kernel above on it, with zero decode
  instructions of any kind -- confirmed directly in this function's own
  one-line body in `mm_cuda_kernels.cu`. (Same caveat as TOMOS FP32/FP16
  above: this function's own comment is marked "UNTESTED", apparently
  stale relative to the benchmark driver's later promotion note; the
  reinterpret-only architecture is verified from the code itself.)

- **`mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp`** --
  **[PROMOTED, already documented in README_COMPLETE.md as "TOMOS FP32 /
  BF16"]**. Unchanged here; see the existing bullet.
- **`mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4_kernel<SkipBeta>`** -- an
  earlier, structurally distinct predecessor of the promoted kernel. Unlike
  the promoted kernel (a thin reinterpret-cast wrapper around the fully
  refined native BF16 tensor-core kernel), this is a fully independent,
  standalone `mma.sync`-PTX kernel: a mechanical copy of `gemm_bf16_mma_ptx_
  wide_bigreg_sq_deep4_kernel`'s own body with only A's type changed
  (`const uint16_t* wh` instead of `const __nv_bfloat16* A`) and its two
  `cp.async` source references retargeted to `&wh[...]`. Like the promoted
  kernel, it has NO decode step of any kind: `wh`'s raw bits are already a
  valid (truncated) bfloat16 bit pattern unconditionally for every possible
  packed value (bfloat16's exponent field never needs fp16's control-
  bit/range-restricted handling -- see `fp32_bf16_pack.hpp`'s own header
  comment), so `wh`'s bytes are `cp.async`'d directly into `As` with no
  decode call in between. It predates the native BF16 kernel's own later
  `iter`/`epivec`/`acongruous`/`fastk`/`precomp2`/`serp` refinement chain
  (only has `deep4`'s 4-stage pipeline), which is why the promoted kernel
  (a thin wrapper reusing the FULLY refined native kernel) superseded it
  rather than that refinement chain being re-applied to this standalone
  copy directly.
- **`mm_fp32_bf16_pure_bf16_fallback_kernel`** -- always-correct,
  one-thread-per-output-element scalar fallback for shapes/transposes
  `wide_bigreg_sq_deep4`'s fast path doesn't cover (NO_TRANSPOSE-only, same
  as the fast path itself supports); decode via `fp32_bf16_bf16_only`.
  Mirrors `mm_fp32_16_pure_half_wide_bigreg_sq_fallback_kernel`'s own
  structure exactly. Source-commented as deliberately NOT reusing the
  generic `gemm_generic_kernel<uint16_t,...>` template, since that template
  only ever `static_cast`s its A-type to `double` for accumulation, which
  would treat `wh`'s bits as a plain integer VALUE rather than
  reinterpreting them as a bfloat16 bit pattern -- silently wrong, not just
  imprecise.

---

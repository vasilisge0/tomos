# Scripts

This file indexes every performance, correctness, and error-analysis script in this
repository for the GEMV/GEMM packed-precision kernel families (`fp32_16` / `fp32_bf16`),
across both compute tiers (full fp32 reconstruction vs genuine 16-bit tensor-core/fp16-
arithmetic compute) and both matrix directions (NO_TRANSPOSE / TRANSPOSE, where GEMV has
both and GEMM has NO_TRANSPOSE only — this codebase has zero GEMM kernels that implement
`TransposeType::TRANSPOSE`).

## Methodology

Three distinct kinds of script exist for (almost) every operation/tier/direction
combination:

- **Performance** (`*_vs_cublas_ref.cu` / `*_vs_cutlass_ref.cu` + a `.sh` driver + a `.m`
  plot): measures GFlop/s against vendor baselines (cuBLAS and/or CUTLASS). Data is
  arbitrary (`pseudo_rand`-derived, no correctness claim); some scripts also cover a
  rectangular/tall-and-thin shape sweep in addition to the square one.
- **Correctness** (`verify_*_plot_promoted.cu` + a `.sh` wrapper): a small, fast,
  focused pass/fail check covering *only* the kernels a given plot currently plots (its
  "promoted" set), at a handful of representative shapes. Every one of these files'
  own header comment documents the exact kernel→series mapping it covers, and must be
  kept in sync with its corresponding `.m` plot's own `series` struct — the plot's
  `series` struct (not any file's header prose, which can go stale) is the ground-truth
  source for "what is currently promoted." Alongside these focused checks, several
  much larger **full exploratory suites** exist (`verify_mv_fp32_16_gevm.cu`,
  `verify_mm_fp32_16_gemm.cu`, etc.) covering every experimental/demoted kernel this
  project has ever built, not just the promoted set — see "Full exploratory suites"
  below.
- **Error** (`*_l2_error_vs_cublas_ref.cu` + a `.sh` driver + a `.m` plot): measures how
  far each promoted kernel's own result drifts from a genuine FP32-precision reference
  (`cublasSgemv`/`cublasSgemm` on uncompressed float32 data), as a function of shape and
  of a `normal_scale` magnitude-sweep parameter. Every data generator in these files is a
  pure, deterministic function of `(linear index, scale)` alone, so the packed-format
  input and its own float32 ground truth are always index-matched without ever decoding
  a packed value back to float — this is what lets these benchmarks reach the same
  multi-billion-element shapes the performance sweeps do without a host-memory blowup a
  correctness check's own CPU reference would hit.

**Metric**: GEVM (vector output) uses **L2-relative-error**; GEMM (matrix output) uses
**Frobenius-relative-error**. Both are the exact same formula,
`sqrt(sum((ref-gpu)^2)) / sqrt(sum(ref^2))`, computed over the whole output array (never
the input matrix) — "Frobenius" is just the name for that formula applied to a matrix's
flattened element array instead of a vector's. This replaced an earlier per-element
max-relative-error metric (`err / max(1.0, |ref|)` per element, worst element wins):
that metric's `max(1.0, |ref|)` floor lets a single output element whose true value
happens to land near zero blow up into a huge "relative error" from ordinary rounding
noise alone, forcing tolerances so loose (`tol=5.0`, i.e. 500%) that they stopped
meaningfully testing anything — confirmed empirically (`max_rel_err` up to ~1.8 accepted
as "OK" under the old metric, before the switch). The whole-array L2/Frobenius norm
averages over every element instead of being driven by the single worst one, so it can't
be blown up by an isolated near-zero element the same way, and tolerances can stay
genuinely tight. Every `check()`/pass-fail function below still computes and prints the
old per-element `max_abs_err`/`max_rel_err` too, purely as a debugging aid (which element
is worst, by how much) — just not what pass/fail is decided on.

**`normal_scale` sweep**: every error benchmark's own generators multiply their
"normal"-range test data (nominally `[-1,1]`) by a `normal_scale` parameter, sweeping
how error changes as data magnitude moves toward each packed format's own representable
ceiling. The safe range for this sweep is governed by one of two *different* constraints,
depending on what a given tier's kernels accumulate their row/tile sum in:

- **fp16/bf16-ACCUMULATING kernels** (GEVM "pure_half"/"hybrid" tier, both directions):
  the accumulated row-sum's own magnitude, not just each individual input value, must
  stay under fp16's ~65504 finite ceiling. For a row-sum of `K` (or `M`, for TRANSPOSE)
  terms each ~`scale` in magnitude, that sum's own standard deviation is roughly
  `scale*sqrt(K)/3` — a much tighter, shape-DEPENDENT bound. `SCALES=(1 2 4 8 16 32 64
  128)` for these tiers, confirmed empirically (a more aggressive early attempt,
  `scale=32768`, overflowed every row into NaN/Inf at every shape).
- **float-ACCUMULATING kernels** (GEVM/GEMM "recon" tier, and GEMM's tensor-core tier —
  hardware `mma.sync` always accumulates fp16×fp16/bf16×bf16 products in a float32
  register on sm_80, confirmed by reading each such kernel's own PTX): no accumulator
  concern at all — the binding constraint is instead each individual STORED value's own
  format ceiling (`fp32_16`'s NORMAL/EXTENDED encoding boundary, magnitude < 2^16=65536;
  or fp16's own ~65504 max finite magnitude for the tensor-core tier's __half-typed
  operands). `SCALES=(1 4 16 64 256 1024 4096 16384)` for these tiers — much wider,
  shape-INDEPENDENT.

## Row summary

| # | Operation | Tier | Direction | Perf `.cu` | Perf `.sh` | Perf `.m` | Correctness `.cu` | Correctness `.sh` | Error `.cu` | Error `.sh` | Error `.m` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GEVM | 32-bit (recon) | NO_TRANSPOSE | `mv_fp32_recon_vs_cutlass_ref.cu` | `benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh` | `plot_gemv_fp32_recon_vs_cutlass_ref.m` | `verify_gemv_fp32_recon_plot_promoted.cu` | `verify_gemv_fp32_recon_plot_promoted.sh` | `mv_fp32_recon_l2_error_vs_cublas_ref.cu` | `benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh` | `plot_gemv_fp32_recon_l2_error_vs_cublas_ref.m` |
| 2 | GEVM | 32-bit (recon) | TRANSPOSE | `mv_fp32_transpose_recon_vs_cublas_ref.cu` | `benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh` | `plot_gemv_transpose_fp32_recon_vs_cublas_ref.m` | `verify_mv_fp32_transpose_promoted.cu` | `verify_mv_fp32_transpose_promoted.sh` | `mv_fp32_transpose_recon_l2_error_vs_cublas_ref.cu` | `benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_ref.sh` | `plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_ref.m` |
| 3 | GEVM | 16-bit (pure_half) | NO_TRANSPOSE | `mv_fp32_16_pure_half_vs_cublas_ref.cu` | `benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh` | `plot_gemv_fp32_16_pure_half_vs_cublas_ref.m` | `verify_gemv_fp32_16_plot_promoted.cu` | `verify_gemv_fp32_16_plot_promoted.sh` | `mv_fp32_16_l2_error_vs_cublas_ref.cu` | `benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh` | `plot_gemv_fp32_16_l2_error_vs_cublas_ref.m` |
| 4 | GEVM | 16-bit (pure_half) | TRANSPOSE | `mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu` | `benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh` | `plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.m` | `verify_gemv_fp32_16_pure_half_transpose_plot_promoted.cu` | `verify_gemv_fp32_16_pure_half_transpose_plot_promoted.sh` | `mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu` | `benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh` | `plot_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.m` |
| 5 | GEMM | 32-bit (recon) | NO_TRANSPOSE only | `benchmarks_gemm_fp32.cu` | `benchmarks_gemm_format_32_16_compute_32.sh` | `plot_gemm_format_32_16_compute_32.m` | `verify_mm_fp32_recon_plot_promoted.cu` | `verify_mm_fp32_recon_plot_promoted.sh` | `mm_fp32_recon_l2_error_vs_cublas_ref.cu` | `benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh` | `plot_gemm_fp32_recon_l2_error_vs_cublas_ref.m` |
| 6 | GEMM | 16-bit tensor-core (pure/composite) | NO_TRANSPOSE only | `benchmarks_gemm_fp16.cu` | `fp16_bf16_pure_vs_cutlass_ref_matched.sh` | `plot_fp16_bf16_pure_vs_cutlass_ref.m` | `verify_mm_fp16_bf16_pure_plot_promoted.cu` | `verify_mm_fp16_bf16_pure_plot_promoted.sh` | `mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu` | `benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh` | `plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.m` |

All `.cu` files are under `benchmarks/kernels/cuda/matrix/`; all `.sh`/`.m` files are
under `scripts/`. Every performance and error `.cu` is built standalone (a direct `nvcc`
invocation linking against the already-built `libtomos_cuda.so`, driven by its own
`.sh`) — none are wired into the main CMake build. Every correctness `.cu` IS wired into
CMake (`benchmarks/kernels/cuda/matrix/CMakeLists.txt`), built via `cmake --build build
--target <name>`, same convention its own `.sh` wrapper uses.

## Row detail

### Row 1 — GEVM 32-bit reconstruction, NO_TRANSPOSE

Promoted kernels (reference: `cublasSgemv`):

| Series | Kernel |
|---|---|
| TOMOS FP32 | `mv_fp32_tiled` |
| TOMOS FP32/FP16 | `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal` |
| TOMOS FP32/BF16 | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded` |
| TOMOS FP32/BF16 (workspace) | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` |

Accumulates in float (full mant_hi+exponent reconstruction decode) — correctness
tolerance `1e-3` (L2-relative), error-sweep `SCALES=(1 4 16 64 256 1024 4096 16384)`.
`mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal` has an "always
control=0" contract and is UNSAFE on arbitrary data — both its correctness check and its
error benchmark use a dedicated magnitude-restricted `[0.5,1.0]`-range dataset.

### Row 2 — GEVM 32-bit reconstruction, TRANSPOSE

Promoted kernels (reference: `cublasSgemv(CUBLAS_OP_T)`):

| Series | Kernel |
|---|---|
| TOMOS FP32 | `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec` |
| TOMOS FP32/FP16 | `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch` |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch` |
| TOMOS FP32/BF16 | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch` |
| TOMOS FP32/BF16 (truncated) | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4` |

`mv_fp32_transpose_recon_vs_cublas_ref.cu`'s own header comment is STALE (describes an
outdated promoted-kernel state reusing the NO_TRANSPOSE recon tier's own five symbols) —
trust this table (derived from `benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh`'s
own METHOD_MARKERS and the `.cu`'s own actual call sites) over that file's header prose.
`verify_mv_fp32_transpose_promoted.cu` was retrofit this session from the old
per-element metric to L2-relative-error (native's own tol stayed `1e-4`; the four
packed-format checks moved from `5e-3` under the old metric to `5e-2` L2-relative,
matching every sibling correctness check's own bound). Error-sweep
`SCALES=(1 4 16 64 256 1024 4096 16384)` (float-accumulating, same reasoning as Row 1;
TRANSPOSE swaps which dimension is reduced over, but the memory-footprint-driven largest
completing shape is unaffected by direction).

**Not plotted (experimental, not promoted):** `mv_fp32_bf16_transpose_atomiconly_
scaled_mblock_vec4_dispatch` ("TOMOS FP32/BF16 (truncated, test)" in the .cu/.sh, column
`fp32_bf16_truncated_test_gflops`) — a host-side large-M dispatcher over "TOMOS FP32/BF16
(truncated)" above: below M=8192 it calls that exact same kernel directly, at/above
M=8192 it switches to a `retarget2` sibling with a more aggressive M_BLOCK-scaling
target (8192 vs 512). It exists to test whether the same M-threshold-routing trick
already promoted for "TOMOS FP32/FP16" and "TOMOS FP32/FP16 (fast)" above also wins for
BF16 — mirroring exactly how "TOMOS FP32/BF16" itself (the `_rounded_dispatch` series)
was promoted from this same kind of "(test)" slot after a confirmed M=8192 win. Neither
`_dispatch` nor `_retarget2` allocates a workspace — both combine via plain `atomicAdd`
into `y`, identical to every other kernel in this row. Was plotted as a dotted series in
`plot_gemv_transpose_fp32_recon_vs_cublas_ref.m`; removed from that plot (still compiled,
correctness-checked, and printed by the benchmark `.cu`/`.sh` either way) since no
promotion decision has been made yet — see `benchmarks_gemv_transpose_fp32_recon_vs_
cublas_ref.sh`'s own `METHOD_MARKERS` comment for the current state of that evaluation.

### Row 3 — GEVM 16-bit pure_half, NO_TRANSPOSE

Promoted kernels (reference: `cublasSgemv`, real fp32 — NOT the Hgemm/GemmEx baselines
the performance file's own cuBLAS series uses):

| Series | Kernel |
|---|---|
| TOMOS FP16 | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best` |
| TOMOS FP32/FP16 | `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best` |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best` |
| TOMOS FP32/BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2` |
| TOMOS BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best` |

Three of five series (TOMOS FP16, FP32/FP16, FP32/FP16 (fast)) accumulate their row-sum
in `__half2` internally — tight, K-dependent `SCALES=(1 2 4 8 16 32 64 128)` (`K=65536`
is this sweep's largest completing shape; `M=K=131072` always gets `SKIPPED`, needing
~68.7GB for a single float32 ground-truth array). TOMOS FP32/BF16 and TOMOS BF16
accumulate in float (no such ceiling) but share the same scale sweep for a fair,
apples-to-apples comparison across all five series. Correctness/error tolerance `5e-2`
L2-relative for every series.

### Row 4 — GEVM 16-bit pure_half, TRANSPOSE

Promoted kernels (reference: `cublasSgemv(CUBLAS_OP_T)`, matching Row 3's own choice of
a true fp32 reference over the performance file's own Hgemm/GemmEx baselines):

| Series | Kernel |
|---|---|
| TOMOS FP16 | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split` |
| TOMOS FP32/FP16 | `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired` |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast` |
| TOMOS FP32/BF16 | `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4` |

Only four series here (not five, unlike Row 3) — this tier has no separate plain "TOMOS
BF16" series plotted, only "TOMOS FP32/BF16." `mv_fp32_16_pure_half_transpose_
atomiconly_scaled_mblock_vec4_split` needs its own `float*` scratch workspace, sized via
its own `_workspace_size` helper — unlike every other kernel in this row. All four
accumulate in `__half2`/`__nv_bfloat162` internally (confirmed by reading each kernel's
own body) — same tight `SCALES=(1 2 4 8 16 32 64 128)` as Row 3, keyed to `M` (the
reduction length for TRANSPOSE) instead of `K`; same `M=K=65536` largest-completing-shape
reasoning applies (memory footprint is symmetric regardless of transpose direction).
Tolerance `5e-2` L2-relative for every series.

### Row 5 — GEMM 32-bit reconstruction, NO_TRANSPOSE (GEMM has no TRANSPOSE kernels)

Promoted kernels (reference: `cublasSgemm`):

| Series | Kernel |
|---|---|
| TOMOS FP32 | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` |
| TOMOS FP32/FP16 | `mm_fp32_16_tiled_cpasync_dualbuf` |
| TOMOS FP32/FP16 (truncated) | `mm_fp32_16_tiled_cpasync_dualbuf_truncated` |
| TOMOS FP32/FP16 (workspace) | `mm_fp32_16_gemm_materialize_midpoint_exact` |
| TOMOS FP32/BF16 (truncated) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` |
| TOMOS FP32/BF16 (workspace) | `mm_fp32_bf16_gemm_materialize_rounded` |

First Frobenius-norm row — the metric is identical to every GEVM row's own L2-relative
error, just applied to the flattened M×N output array. All six kernels accumulate their
tile's partial sum in a float register array (confirmed by reading `mm_fp32_16_tiled_
cpasync_dualbuf_kernel`'s own body: `float acc[TM][TN]`) — no accumulator-overflow
concern, `SCALES=(1 4 16 64 256 1024 4096 16384)`. `mm_fp32_16_tiled_cpasync_dualbuf_
truncated` requires wh/wr from `fp32_16_split_truncated` specifically (not interchangeable
with the plain `fp32_16_split`-produced pair the other two FP16 kernels share). "TOMOS
FP32/BF16 (workspace)" is the ROUNDED bf16 encoding (`fp32_bf16_split_rounded`) despite
its own label referring to a materialize strategy, not its encoding — "TOMOS FP32/BF16
(truncated)" is the plain (non-rounded) split. Correctness tolerance `1e-3` Frobenius-
relative (near-fp32 accuracy target, same as Row 1/2's own recon-tier bound).

### Row 6 — GEMM 16-bit tensor-core (pure/composite), NO_TRANSPOSE

Promoted kernels (reference: `cublasSgemm`, real fp32 — NOT the Hgemm/GemmEx baselines
the performance file's own cuBLAS series uses):

| Series | Kernel |
|---|---|
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` |
| TOMOS BF16 | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` |
| TOMOS FP32/FP16 | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` |
| TOMOS FP32/FP16 (workspace) | `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` |
| TOMOS FP32/BF16 | `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` |

Unlike every other row, EVERY kernel here stores B (and, for the two native kernels, A
too) in genuine `__half`/`__nv_bfloat16` precision by design — no `float* B` overload
exists in this tier at all. The multiply-accumulate itself still happens in genuine
float32 (sm_80 `mma.sync`'s own hardware-mandated accumulator for f16/f16 and bf16/bf16
inputs, confirmed by reading the native fp16 kernel's own PTX — `mma.sync.aligned.
m16n8k16.row.col.f32.f16.f16.f32`), so `SCALES=(1 4 16 64 256 1024 4096 16384)` applies
here too (same reasoning as Row 5, keyed to fp16's own ~65504 per-VALUE storage ceiling
rather than an accumulator bound — bf16 has no realistic ceiling at any scale used).
Because both operands are always precision-reduced here, the error benchmark's own
"genuine FP32" reference intentionally captures BOTH operands' storage rounding, unlike
every other row's own "only A is compressed" convention. The correctness check's own CPU
reference is built by rounding A/B to their target storage precision on the HOST first
(exactly what the GPU kernel's own inputs are), decoding back to float, THEN running a
float-accumulating CPU GEMM on those rounded-and-decoded values — this is what a genuine
float32 accumulator over the SAME rounded inputs should produce, which is what these
tensor-core kernels are actually expected to match (not a comparison against never-
rounded fp32 data, which would just re-measure this tier's own inherent storage-rounding
error — that measurement is what the error benchmark is for). Tolerance `5e-2`
Frobenius-relative (fp16/bf16-storage-level accuracy target, not near-fp32).

## Kernel techniques and references

Every kernel below was characterized by tracing its host wrapper to the actual
`__global__` kernel(s) it launches (`src/kernels/cuda/matrix/mv_cuda_kernels.cu`,
`fp32_16_cuda_kernels.cu`, `mm_cuda_kernels.cu`), reading the kernel body and its own
header comments directly, and — for the tensor-core tier — reading the emitted PTX. This
section cites the paper, whitepaper, or standard where each underlying technique was
first described, and is explicit about the alternative: a technique this codebase built
itself (no independent citation applies), or one it explicitly ported from a named prior
codebase (`randls2.0`) rather than the literature. References are numbered in order of
first appearance below; the numbered list itself is at the end of this section.

### Building blocks shared across rows

- **Split-K / split-M reduction, combined via `atomicAdd`.** The reduction dimension (K
  for NO_TRANSPOSE, M for TRANSPOSE) is partitioned across `blockIdx.x`/`.y` so multiple
  blocks contribute partial sums to the same output element; combined either via a
  separate reduce-kernel pass over a `float`/`double` partial-sum workspace, or — in
  every *promoted* kernel except Row 4's isolated `_vec4_split` — directly via
  `atomicAdd`, skipping the extra kernel launch and workspace buffer entirely. This is
  the standard GPU parallel-reduction tradeoff (tree/atomic combine vs. a second reduce
  pass) [1], applied here specifically to the register-blocked/vectorized GEMV tiling
  style set out in [2]. Scoped to GEVM (Rows 1-4) only — the GEMM tiers (Rows 5/6) tile
  K within a single block's own sequential loop instead of partitioning it across the
  grid, so no kernel there needs this combine step at all.
  **Kernels:** Row 1 — `mv_fp32_tiled` (wide-shape dispatch tier only, reduce-kernel
  combine), `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock`,
  `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` (all four: K-scaled
  K_BLOCK split-K, `atomicAdd` combine). Row 2 (all five, M_BLOCK split-M,
  `atomicAdd` combine) — `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec`,
  `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch`,
  `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch`,
  `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch`,
  `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4`. Row 3 (all five, `col_splits`
  split-K, `atomicAdd` combine) — `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`,
  `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best`,
  `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`. Row 4 —
  `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`,
  `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast`,
  `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4` (M_BLOCK split-M,
  `atomicAdd` combine); `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split`
  (same M_BLOCK split-M, but combined via a separate reduce kernel over a caller-owned
  workspace instead of `atomicAdd` — the one deliberate exception in the promoted set).
- **Warp-per-output-column + warp-shuffle-tree reduction.** Every TRANSPOSE (Row 2/4)
  kernel assigns one full warp to each output column, walks the 32 lanes together down
  the row (reduction) dimension exploiting column-major contiguity, and finishes with a
  `__shfl_down_sync` tree instead of shared-memory reduction. The shuffle instruction
  itself was introduced with Kepler [3]; using it for exactly this warp-level reduction
  tree (in place of a shared-memory scratch-and-sync) is the pattern popularized in [4].
  **Kernels:** all five Row 2 kernels and all four Row 4 kernels listed under the
  previous bullet (every promoted TRANSPOSE kernel in this codebase uses this
  architecture, with no exceptions).
- **Vectorized wide loads (`float4`/`uint32_t`/`uint64_t`/`ushort4`).** Every family
  uses the widest aligned load its packed layout allows, with a runtime alignment check
  and scalar prefix/suffix fallback — the standard GPU memory-coalescing lever for
  bandwidth-bound kernels like GEMV [2]. Row 3 is the one promoted family that does
  *not* use this: its NO_TRANSPOSE `wh` loads are 4 independent scalar 16-bit reads
  (only the arithmetic, not the load, is packed there — see the packed-FMA bullet
  below).
  **Kernels:** Row 1 — `mv_fp32_tiled` (narrow-shape tier, `float4` on A, 4 rows/thread),
  `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` (`uint64_t`, 4×wh+4×wr),
  `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` (all three:
  `uint32_t`, 2×wh+2×wr). Row 2 (all five, `float4` on the staged x vector, plus
  `ushort4` on wh/wr for every packed-format kernel — the native
  `..._vec4_stagevec` kernel vectorizes both its x-staging and A-reduction loops via
  `float4` instead) — same five kernels as the split-M list above. Row 4 (all four,
  `ushort4` on 4 consecutive wh values — the one real load-bandwidth win in this tier,
  vs. Row 3's arithmetic-only packing) — same four kernels as the split-M list above.
  Row 5 — `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` and
  `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` (`float4`,
  synchronous software-pipelined loads — its `cp.async`-based siblings
  `mm_fp32_16_tiled_cpasync_dualbuf[_truncated]` use `cp.async` bulk copies instead of
  a classic vectorized load; see Row 5/6's own cp.async note below).
- **Register-blocked "rows/lane" and "rows/thread" tiling (GEVM only).** Processing 2 or
  4 output rows per thread/lane in registers rather than one — the register-blocking
  technique for GPU dense linear algebra kernels set out in [2], which also established
  the general register-tile/vectorized-load combination this bullet's kernels follow.
  This is a NO_TRANSPOSE, Row-1-only pattern in the promoted set: Row 3's NO_TRANSPOSE
  16-bit kernels are one-thread-per-row instead (see the split-K bullet above), so they
  do not belong here despite superficially similar "ready"/"atomic" naming.
  **Kernels:** `mv_fp32_tiled` (narrow-shape tier, 4 rows/thread),
  `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` (4 rows/lane),
  `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`,
  `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` (all three: 2
  rows/lane).
- **Packed half-precision SIMD FMA (`__hfma2`/`__nv_bfloat162`).** Every "pure_half"
  and "hybrid" Row 3/4 kernel accumulates via genuine 2-wide packed FMA instructions
  (`__half2`/`__nv_bfloat162`), not scalar `__half` arithmetic — the packed-precision
6  cause (see main Row-summary methodology above) of why this tier's accumulated error
  grows with K while the float-accumulating recon/GEMM tiers' does not.
  **Kernels:** Row 3 (all five) — `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`,
  `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best`,
  `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`. Row 4 (all four) —
  `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split`,
  `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`,
  `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast`,
  `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4`.
- **Branch-elimination via unconditional computation + arithmetic masking.** Several
  kernels replace a per-element data-dependent branch with unconditional computation of
  every path plus a bitwise/arithmetic select — both the "wrapborrow" wraparound-
  subtraction reformulation of the packed format's own carry logic, and the hybrid
  family's `cheap_bits_mask = control - 1` masking — are applications of the general
  branch-free bit-manipulation techniques catalogued in [6]. The *decision* to eliminate
  the branch at all is this codebase's own quantitative argument (documented directly in
  several kernel headers): with each of 32 lanes independently having, e.g., a ~10%
  chance of needing a kernel's "rare" path, a whole warp skips it together only
  `0.9^32 ≈ 3.4%` of the time, so ~97% of warps already pay for both code paths serially
  under branching — branch-free computation is then free (or a net win). Note this is
  distinct from Row 6's `_inline_dirty` kernel below, which skips the rare path entirely
  via a warp-uniform vote instead of computing it unconditionally.
  **Kernels:** `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` (Row 1, branchless
  full decode, `fp32_16_combine_branchless`),
  `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch` (Row 2,
  `fp32_16_combine_branchless_wrapborrow`),
  `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch`
  (Row 2, `fp32_16_combine_fast_normal_wrapborrow`),
  `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best` (Row 3, dual
  cheap/extended accumulator with `cheap_bits_mask` masking). The other Row 3/4
  "fast"/"fast_normal" kernels use an unconditional bit-reinterpret instead
  (`fp32_16_fast_half`/`fp32_16_combine_fast_normal`) — cheaper still, but because it
  skips the safety check rather than computing-and-masking both paths, it is UNSAFE on
  arbitrary data rather than an application of this same technique. Row 4's
  `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired` also does not
  belong here despite its "hybrid" lineage: its per-pair check is itself a branch, and a
  mixed pair falls back to the ordinary branchy per-element decode, not to masking.
- **bfloat16's single, unconditionally-exact decode.** Every `fp32_bf16_*` kernel has
  exactly one decode function, with no "fast"/"safe" split and no reconstruction-vs-
  half-only distinction the `fp32_16` (fp16-based) family needs — because bfloat16's
  8-bit exponent field is bit-for-bit the same width and bias as float32's own [7],
  truncating/rounding the mantissa is the only lossy step, so a bf16 encoding of any
  finite fp32 value is always exactly representable and always correctly ranged. This
  covers every kernel that actually calls an `fp32_bf16_*` decode function — it excludes
  the two *native* bf16 tensor-core kernels (TOMOS BF16 in Rows 3 and 6), which operate
  on raw `__nv_bfloat16`/`bf16` storage directly and have no packed-format decode step
  to begin with.
  **Kernels:** Row 1 — `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`
  (`fp32_bf16_combine_rounded`), `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace`
  (`fp32_bf16_combine`). Row 2 — `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch`
  (`fp32_bf16_combine_rounded`), `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4`
  (`fp32_bf16_combine`). Row 3 — `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best` (both `fp32_bf16_bf16_only`).
  Row 4 — `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4`
  (`fp32_bf16_bf16_only`). Row 5 — `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`
  (fused-decode `fp32_bf16_combine`), `mm_fp32_bf16_gemm_materialize_rounded`
  (`fp32_bf16_combine_rounded` in its materialize-chunk kernel). Row 6 —
  `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` (`fp32_bf16_bf16_only`, applied
  as a zero-cost `reinterpret_cast`).

### Row 1 — GEVM 32-bit reconstruction, NO_TRANSPOSE

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP32 | `mv_fp32_tiled` | 3-way shape dispatch: split-K + separate reduce kernel (wide) / shared-x-tiled one-thread-per-row (mid) / 4-rows/thread register-blocked with `float4` A-loads (narrow) | [1][2] |
| TOMOS FP32/FP16 | `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` | split-K via K-scaled K_BLOCK tiling + `atomicAdd` combine; branchless full decode (`fp32_16_combine_branchless`, all 3 reconstruction paths computed unconditionally); 4-rows/lane packed `uint64_t` load | [1][2][6] |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal` | same split-K/atomicAdd/K_BLOCK structure; UNSAFE "always control=0" decode (`fp32_16_combine_fast_normal`) — bespoke unsafe fast-path, no independent citation beyond the packed format's own convention; 2-rows/lane packed `uint32_t` load | [1][2] |
| TOMOS FP32/BF16 | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded` | same structure, `fp32_bf16_combine_rounded` decode | [1][2][7] |
| TOMOS FP32/BF16 (workspace) | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` | same structure with a caller-owned workspace buffer (avoids per-launch malloc/free — standard CUDA allocation-avoidance practice, no specific paper), plain unconditional `fp32_bf16_combine` | [1][2][7] |

### Row 2 — GEVM 32-bit reconstruction, TRANSPOSE

All five kernels share the warp-per-column/split-M/shared-x-staging/warp-shuffle-
tree/atomicAdd architecture described under "Building blocks" above [1][2][3][4]. The
family's origin is explicitly attributed in its own header comment
(`mv_cuda_kernels.cu:371-417`) to a **prior codebase, not a paper**: ported from
`randls2.0`'s `sgemv_cm_transpose_splitM_atomic_xshared` — preserved here verbatim
rather than treated as needing an academic citation.

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP32 | `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec` | both the x-staging loop and the main A-reduction loop independently vectorized via `float4` | [2] |
| TOMOS FP32/FP16 | `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch` | `ushort4`-vectorized wh/wr loads; branchless wraparound-subtraction decode (`fp32_16_combine_branchless_wrapborrow`) replacing a select/ternary carry check | [2][6] |
| TOMOS FP32/FP16 (fast) | `..._fast_normal_wrapborrow_vec4_dispatch` | same structure, UNSAFE wrapborrow decode (`fp32_16_combine_fast_normal_wrapborrow`) | [2][6] |
| TOMOS FP32/BF16 | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch` | `fp32_bf16_combine_rounded` decode | [2][7] |
| TOMOS FP32/BF16 (truncated) | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4` | plain unconditional `fp32_bf16_combine` — "already maximally cheap," no wrapborrow/fast variant exists because there is no branch left to remove | [2][7] |

### Row 3 — GEVM 16-bit pure_half, NO_TRANSPOSE

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP16 | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best` | split-K + `atomicAdd` combine [1]; genuine packed `__half2` FMA accumulation (`__hfma2`) [5]; shared-memory `x_tile` staging | [1][5] |
| TOMOS FP32/FP16 | `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best` | dual-accumulator design: a packed `__half2` accumulator for the common (control=0) path, a scalar float accumulator for the rare extended-range path, combined only once at the end; branchless variant computes both paths unconditionally and discards the unwanted one via arithmetic masking, motivated by the warp-divergence-probability argument above | [5][6] |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best` | identical to TOMOS FP16's kernel, decode swapped to the UNSAFE unconditional bit-reinterpret `fp32_16_fast_half` (only valid on data generated with `gen_fp32_16_wh_safe`) | [1][5] |
| TOMOS FP32/BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2` | byte-for-byte identical body to TOMOS BF16's own kernel — packed `__nv_bfloat162` FMA via `fp32_bf16_bf16_only`, exact because of bf16's fp32-matching exponent field | [1][5][7] |
| TOMOS BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best` | same as above | [1][5][7] |

### Row 4 — GEVM 16-bit pure_half, TRANSPOSE

All four kernels share the warp-per-column/split-M/shared-x-staging(`__half`/
`__nv_bfloat16`)/warp-shuffle-tree architecture [2][3][4], each lane additionally doing a
vectorized `ushort4` load of 4 consecutive `wh` values (a real memory-bandwidth win here,
unlike Row 3's NO_TRANSPOSE tier where only the arithmetic, not the load, is packed).

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP16 | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split` | identical compute body to the plain `_vec4` sibling, but combines via a caller-owned workspace + separate reduce kernel instead of `atomicAdd` — built specifically to isolate atomics-vs-split+reduce as its own variable [1] | [1][2][5] |
| TOMOS FP32/FP16 | `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired` | "paired" packing: checks each adjacent `ushort4`-loaded pair's control bits together and, when both are cheap (control=0, the common case), issues one real 2-wide `__hfma2` reading the pair's shared x-values as a single `__half2` — fixing a throughput gap where the plain per-element hybrid decode wastes one lane of every packed FMA | [5][6] |
| TOMOS FP32/FP16 (fast) | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast` | UNSAFE unconditional `fp32_16_fast_half` decode on `ushort4`-vectorized loads, packed `__hfma2` accumulation | [2][5] |
| TOMOS FP32/BF16 | `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4` | bf16 port of the fp16 "fast" kernel; unlike its fp16 lineage, carries no unsafe-on-arbitrary-data caveat at all — there is only one bf16 decode function, and it was already unconditional [7] | [2][5][7] |

### Row 5 — GEMM 32-bit reconstruction, NO_TRANSPOSE

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP32 | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | BM=BN=128/BK=8/TM=TN=8 shared-memory + register tile hierarchy [2][8]; CUTLASS's own `RowMajorInterleaved<2>` warp/lane geometry, read directly from CUTLASS source [9]; snake (boustrophedon) accumulate order for shared-memory-register `.reuse`, also read directly from CUTLASS source [9]; 4-deep rotating shared-memory buffer, synchronous (`float4`) software-pipelined prefetch — no `cp.async` in this "syncpipe" family, distinct from Row 5/6's cp.async kernels below | [2][8][9][10] |
| TOMOS FP32/FP16 | `mm_fp32_16_tiled_cpasync_dualbuf` | BM=BN=128/BK=16/TM=TN=8, plain row-major warp/lane split (no CUTLASS interleave); genuine `cp.async`-based 2-stage double buffering via `__pipeline_memcpy_async`/`__pipeline_commit`/`__pipeline_wait_prior`, including a separate K-contiguous cp.async staging buffer for B with a shared-to-shared transpose scatter | [2][10][11] |
| TOMOS FP32/FP16 (truncated) | `mm_fp32_16_tiled_cpasync_dualbuf_truncated` | identical kernel body, decodes via `fp32_16_combine_truncated` (requires `wh`/`wr` from `fp32_16_split_truncated`, not interchangeable with the plain-split pair) | [2][10][11] |
| TOMOS FP32/FP16 (workspace) | `mm_fp32_16_gemm_materialize_midpoint_exact` | two-stage "materialize": an elementwise decode kernel (`fp32_16_combine`) writes a full fp32 scratch copy of A, then a native black-box call into the TOMOS FP32 kernel above runs unmodified on it — this composition (decode-to-scratch, then call an existing best native kernel, rather than fusing decode into the GEMM's own load path) is this codebase's own design, no independent citation | — |
| TOMOS FP32/BF16 (truncated) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | same tile/pipeline/snake structure as TOMOS FP32's own kernel, fused-decode `load_global_advance` calling `fp32_bf16_combine` (unconditional/exact, no degraded-fallback branch needed, per bf16's own exponent equivalence [7]) instead of `fp32_16_combine` | [2][7][8][9][10] |
| TOMOS FP32/BF16 (workspace) | `mm_fp32_bf16_gemm_materialize_rounded` | same materialize composition as TOMOS FP32/FP16 (workspace), decode via `fp32_bf16_combine_rounded` | [7] |

### Row 6 — GEMM 16-bit tensor-core (pure/composite), NO_TRANSPOSE

Every kernel here issues genuine `mma.sync` Tensor Core instructions — hardware
matrix-multiply-accumulate units introduced with Volta [12] and used here at Ampere's
(sm_80) `m16n8k16` granularity, with `f16`/`bf16` operands and a hardware-mandated `f32`
accumulator (confirmed directly from the emitted PTX,
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` / `...f32.bf16.bf16.f32`) — the
exact input/accumulate precision combination independently characterized in [13].
Fragment loads use `ldmatrix.sync.aligned` (introduced alongside `mma.sync`), whose
lane-to-fragment layout this codebase reads directly from CUTLASS's own confirmed source
rather than deriving independently [9]; [14] is the paper that first reverse-engineered
and documented this same `ldmatrix`/`mma.sync` fragment-layout behavior at the PTX level.
All five kernels use a 4-stage `cp.async` pipeline (`kGemmFp16WideBigRegDeep4Stages=4`)
[10][11], CUTLASS's serpentine (boustrophedon) `mma.sync` issue order [9], CUTLASS's
`ColumnMajorTensorOpMultiplicandCongruous` shared-memory swizzle (this codebase's own
"acongruous" naming) [9], and CUTLASS's 128-bit vectorized epilogue [9] — all four read
directly from CUTLASS header source, not reimplemented independently.

| Series | Kernel | Technique(s) | Ref |
|---|---|---|---|
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | BM=256/BN=128/BK=32 tile, 8 warps × 32 `mma.sync` calls/warp/k-step; falls back to smaller/simpler sibling kernels on misalignment or non-exact-K | [9][10][11][12][13][14] |
| TOMOS BF16 | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | mechanical bf16 port of the fp16 kernel above — only `__half`→`__nv_bfloat16` and the PTX element-type suffix change; compiles to the same register counts | [9][10][11][12][13][14][7] |
| TOMOS FP32/FP16 | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` | "inline_dirty" decode: `wh`'s raw bits are `cp.async`'d directly into the fp16 shared buffer (no separate decode-and-store step); a per-thread OR-reduce plus a warp-uniform `__any_sync` vote decides once whether *any* lane's data needs correction, so the whole warp takes the same branch — "batch first, branch once" using CUDA's own warp-vote primitive [15] rather than per-element predication | [9][10][11][12][13][14][15] |
| TOMOS FP32/FP16 (workspace) | `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` | two-stage materialize (vectorized `ushort4`→`fp32_16_half_only`→`uint2` decode into a *persistent, cached* scratch buffer never freed on the steady-state path, then a native call into the TOMOS FP16 kernel above) — bespoke composition, no independent citation | — |
| TOMOS FP32/BF16 | `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` | zero-cost `reinterpret_cast<__nv_bfloat16*>(wh)` — no decode kernel, no scratch buffer at all — then a direct call into the TOMOS BF16 kernel above; possible only because bf16's exponent field is bit-for-bit interchangeable with fp32's own [7], making this "the zero-cost limit of the materialize idea for this specific packed format" | [7] |

### References

1. M. Harris, "Optimizing Parallel Reduction in CUDA," NVIDIA developer technology
   whitepaper / GTC slides, 2007. (Tree-reduction vs. atomic-combine tradeoff for
   partial-sum GPU reductions.)
2. V. Volkov and J. Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra," *SC '08:
   Proceedings of the 2008 ACM/IEEE Conference on Supercomputing*. (Register blocking,
   vectorized wide loads, and software double-buffering for GPU GEMV/GEMM.)
3. NVIDIA, "NVIDIA GeForce GTX 680 / Kepler GK110" architecture whitepaper, 2012.
   (Introduction of the warp-shuffle instruction family, `__shfl_down` and siblings.)
4. J. Luitjens, "Faster Parallel Reductions on Kepler," NVIDIA Developer Blog, 2014.
   (Warp-shuffle-based reduction tree, replacing shared-memory reduction.)
5. NVIDIA, "CUDA Math API" reference documentation — `__half2`/`__nv_bfloat162` packed
   arithmetic intrinsics (`__hfma2` and related).
6. H. S. Warren Jr., *Hacker's Delight*, 2nd ed., Addison-Wesley, 2013. (Branch-free
   arithmetic and bitwise techniques, including wraparound-subtraction carry logic and
   mask-based select.)
7. D. Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training,"
   arXiv:1905.12322, 2019. (bfloat16's 8-bit exponent field, matching float32's own
   width and bias — the property every `fp32_bf16_*` decode in this codebase exploits.)
8. K. Goto and R. A. van de Geijn, "Anatomy of High-Performance Matrix Multiplication,"
   *ACM Transactions on Mathematical Software* 34(3), 2008. (Hierarchical
   register/cache blocking that this codebase's shared-memory-tile + register-tile
   structure mirrors at the GPU level.)
9. NVIDIA, *CUTLASS: CUDA Templates for Linear Algebra Subroutines* (open-source
   library and documentation), A. Kerr et al., 2017–present. Source of the
   `RowMajorInterleaved<2>` warp/lane mapping, the serpentine (boustrophedon)
   `mma.sync` issue order, the `TensorOpMultiplicandCongruous` shared-memory swizzle,
   and the 128-bit vectorized epilogue — all confirmed in this codebase by reading
   CUTLASS's own header source directly, not re-derived independently.
10. NVIDIA, "NVIDIA A100 Tensor Core GPU / Ampere GA100" architecture whitepaper, 2020.
    (Introduction of `cp.async` asynchronous global-to-shared memory copy at sm_80.)
11. NVIDIA, "Parallel Thread Execution ISA," `cp.async`/`ldmatrix`/`mma.sync`
    instruction reference. (Primary source for the exact semantics of every PTX
    instruction cited in this section.)
12. NVIDIA, "NVIDIA Tesla V100 GPU Architecture" whitepaper, 2017. (Introduction of
    Tensor Cores, the hardware matrix-multiply-accumulate unit `mma.sync` targets.)
13. S. Markidis, S. W. Der Chien, E. Laure, I. B. Peng, and J. S. Vetter, "NVIDIA
    Tensor Core Programmability, Performance & Precision," *2018 IEEE International
    Parallel and Distributed Processing Symposium Workshops (IPDPSW)*. (Empirical
    characterization of Tensor Cores' fp16-input/fp32-accumulate behavior.)
14. D. Yan, W. Wang, and X. Chu, "Demystifying Tensor Cores to Optimize Half-Precision
    Matrix Multiply," *2020 IEEE International Parallel and Distributed Processing
    Symposium (IPDPS)*. (Reverse-engineers the `ldmatrix`/`mma.sync` PTX
    lane-to-fragment layout this codebase reads from CUTLASS's own source.)
15. NVIDIA, "CUDA C++ Programming Guide," section "Warp Vote Functions"
    (`__any_sync` and related). (The warp-uniform-predicate primitive behind Row 6's
    "batch first, branch once" design.)

## Per-kernel algorithm descriptions

One paragraph per promoted kernel (the same 30 kernels tabulated row-by-row above),
describing how it actually works end to end. Reference numbers `[n]` point at the
numbered list under "Kernel techniques and references" above.

### Row 1 — GEVM 32-bit reconstruction, NO_TRANSPOSE

**`mv_fp32_tiled`** — TOMOS FP32 (compute: FP32; thread block: 1D, 256 threads —
`kMvFp32TileSize`, the same constant across all three dispatch tiers) — is the native
(unpacked) fp32 GEMV, with no decode step at all, and
picks one of three kernels via a 3-way shape dispatch that mirrors the packed-format
family's own dispatch shape-for-shape. When `rows_op` is below `kMvFp32WideRowThreshold`
(131072), it launches `mv_fp32_wide_kernel` with `grid.y = col_splits` partitioning K
into column chunks — split-K [1][2] — each block accumulating a partial dot product into
a `double*` scratch buffer, followed by a separate `mv_fp32_wide_reduce_kernel` that sums
the `col_splits` partials per row and applies alpha/beta. When `cols_op` is below
`kMvFp32NarrowVecColThreshold` (64), it launches `mv_fp32_tiled_kernel`: one thread per
output row, no split-K, x staged in shared memory, no atomics needed since each thread
owns its own `y[i]`. Otherwise it launches `mv_fp32_narrow_vec_kernel`: register-blocked
with 4 rows/thread [2], reading A via a single `float4` vectorized load [2] instead of 4
scalar loads whenever alignment allows. Its own header comment states the conclusion this
whole family follows: coalesced one-thread-per-row access beats warp-reduction tricks for
this memory-bound workload, a single wide vectorized load matters more than FMA-side
unrolling, and split-K column-partitioning — not a different reduction strategy — is
what actually fixes "too few rows to fill the GPU."

**`mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock`** — TOMOS FP32/FP16 (compute:
FP32; thread block: `dim3(32, 8)` = 256 threads, 8 warps × 32 lanes) — is a
row-vectorized, split-K
design for the NO_TRANSPOSE fast path (`rows_op < kMvWideRowThreshold`): K is partitioned
across `blockIdx.x` into `K_BLOCK`-sized tiles, scaled by `kMvReadyTargetKSplits=128` for
large K [1][2], with x staged once per K-tile into shared memory. `y` is pre-scaled by
beta in a separate pass, and every block then does a plain `atomicAdd` per row into `y` —
no ready-flag or spin-wait synchronization is needed. Each thread/lane handles 4 output
rows, reading `wh` (and `wr`) for those 4 adjacent rows via one 8-byte `uint64_t` load [2]
when alignment allows (scalar fallback otherwise). Decode is the SAFE
`fp32_16_combine_branchless`: the packed format's full reconstruction, computed via all
three (normal/extended/degraded) paths unconditionally rather than a data-dependent
branch [6], so no warp-divergence penalty applies regardless of how many elements land in
the rare "extended" bucket. Measured on real A100 hardware as a monotonic 1.070x win at
M=16384 over its own 1-row/lane sibling.

**`mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`** — TOMOS FP32/FP16
(fast) (compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — shares the same
split-K/K_BLOCK/shared-x-staging/atomicAdd architecture as the kernel above [1][2], but
processes 2 output rows/lane instead of 4, packing `wh`/`wr` for those 2 rows into a
single `uint32_t` load [2] and unrolling the K loop 4-way. Decode is
`fp32_16_combine_fast_normal` — an UNSAFE, unconditional bit-reinterpret that computes
only the cheap "normal" path; correctness depends entirely on the caller guaranteeing
every element's control bit is 0, which is why both its correctness check and its error
benchmark are run exclusively against a dedicated `[0.5,1.0]`-range dataset rather than
arbitrary data. Confirmed on real hardware to beat its own safety-checked sibling at every
size tested.

**`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`** — TOMOS FP32/BF16
(compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — reuses the identical
2-rows/lane, packed-`uint32_t`, prescale+atomicAdd, K-scaled-K_BLOCK structure as the
fp16 kernel above [1][2], decoding via `fp32_bf16_combine_rounded` in place of an
fp16-family decode. This is not a safety tradeoff at all: `fp32_bf16_combine_rounded` is
exact modulo only the single lost-mantissa-bit rounding every bf16 encoding pays,
because bfloat16's exponent field matches float32's own width and bias exactly [7]; other
transpose/shape branches fall back to a workspace-based wide kernel plus tiled/narrow_vec
rounded siblings.

**`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace`** — TOMOS FP32/BF16
(workspace) (compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — has an identical
2-rows/lane/packed-`uint32_t`/prescale+atomicAdd/K_BLOCK-scaled body to the kernel above,
decoding via the plain, unconditional `fp32_bf16_combine` (`(wh<<16)|wr`, no rounding)
[7] — unlike every `fp32_16` "fast_normal" sibling in this file, it needs no
unsafe-on-arbitrary-data disclaimer, because bfloat16's own exponent-width match to
float32 already makes this the cheapest possible exact decode for every input [7]. Its
distinguishing feature is that non-fast-path shape branches reuse a caller-provided
`workspace` buffer, avoiding a malloc/free on every launch, feeding the classic
wide/tiled/narrow_vec kernels.

### Row 2 — GEVM 32-bit reconstruction, TRANSPOSE

All five kernels below share one architecture, explicitly ported (per its own header
comment at `mv_cuda_kernels.cu:371-417`) from a **prior codebase, not a paper** —
`randls2.0`'s `sgemv_cm_transpose_splitM_atomic_xshared`: one warp per output column, the
32 lanes walking together down the row (reduction) dimension exploiting column-major
row-adjacency, M split across `blockIdx.x` into `M_BLOCK`-sized chunks (split-M [1][2]),
x staged into shared memory once per block, ending in a warp-shuffle-tree reduction
(`__shfl_down_sync` [3][4]) and a per-column `atomicAdd` (beta applied in a prescale pass
first).

**`mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec`** — TOMOS FP32 (compute:
FP32; thread block: `dim3(32, 8)` = 256 threads, one warp per output column) — is the
native fp32 member
of the family, with no decode step. Its distinguishing feature is that BOTH the
x-staging loop (global x → shared memory) and the main A-reduction loop are
independently vectorized via `float4` [2], each with its own runtime 16-byte alignment
check and a scalar prefix/suffix fallback — its own header comment confirms this is the
best-measured native fp32 TRANSPOSE kernel in the codebase, and specifically that
vectorizing the staging loop (not just the reduction loop) is what delivers the win.

**`mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch`** — TOMOS
FP32/FP16 (compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — size-forks
on `m` (threshold 8192) to one of two host wrappers with different M_BLOCK-scaling
targets (8192 vs 2048), but both launch the exact same kernel — only the
target/floor constants fed to the M_BLOCK-scaling heuristic differ. x-staging is
vectorized via `float4`; `wh`/`wr` are loaded via `ushort4` [2] (an 8-byte load,
independently alignment-checked against a 16-byte-load fallback path). Decode is
`fp32_16_combine_branchless_wrapborrow` — a bit-exact reformulation of the branchless
decode that replaces its select/ternary carry check with a wraparound-subtraction trick
[6] — measured as a clean, monotonic win at every size on real A100 hardware.

**`mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch`**
— TOMOS FP32/FP16 (fast) (compute: FP32; thread block: `dim3(32, 8)` = 256 threads) —
uses the same M-threshold(8192) dispatch pattern and the same vec4-x-staging/`ushort4`-
wh/wr-load structure as the kernel above, but decodes via
`fp32_16_combine_fast_normal_wrapborrow`: UNSAFE-on-arbitrary-data (only the cheap
normal path, requiring control=0 for every element), reformulated with the same
wraparound-subtraction bit trick [6]. The swap from a safe to an unsafe decode here was
motivated by a direct SASS instruction-count profile showing decode overhead, not the
FMA math itself, dominates this kernel's static instruction count.

**`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch`** — TOMOS
FP32/BF16 (compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — again
size-forks on the same M=8192 threshold to two wrappers targeting the same underlying
kernel, using vec4 x-staging and `ushort4` wh/wr loads [2], decoding via
`fp32_bf16_combine_rounded` [7]. The M-threshold retargeting alone was measured to add
+12.7% over the plain (non-retargeted) `_vec4_rounded` variant.

**`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4`** — TOMOS FP32/BF16 (truncated)
(compute: FP32; thread block: `dim3(32, 8)` = 256 threads) — uses a single, fixed M_BLOCK
target (512, floor 1) with no size fork, the same vec4 x-staging and `ushort4` wh/wr
loads [2], decoding via the plain, unconditional `fp32_bf16_combine` [7]. Its own header
comment notes there is nothing here for a wrapborrow-style reformulation to remove, and
no fast/unsafe branch-free variant left to build, since this decode is already maximally
cheap and unconditional for every input.

### Row 3 — GEVM 16-bit pure_half, NO_TRANSPOSE

**`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`** — TOMOS FP16 (compute: FP16,
packed `__half2`; thread block: 1D, 256 threads — `kMvTileSize`, one thread per row) —
size-forks at 4096 rows
between two host wrappers that ultimately launch one of two structurally identical
kernels (a "bigtile" variant below the threshold, a "wide4x" variant — the same kernel
the plain non-"_best" series uses, just with 4x more `col_splits` — at or above it), both
also falling back to tiled/narrow_vec kernels for narrow-column shapes. Architecture: one
thread per output row `i`; the grid is `(row_blocks, col_splits)`, a column split-K
tiling [1][2] combined directly via `atomicAdd` (no partial buffer, no separate reduce
launch, unlike Row 1's wide-tier kernel). x is staged once per K-tile into
`__shared__ __half x_tile[256]`. The accumulator is genuinely packed —
`__half2 acc01, acc23`, updated via `acc01 = __hfma2(a01, x01, acc01)` [5] — though the
`wh` load itself stays scalar for NO_TRANSPOSE (four independent 16-bit reads, `ld_a`
apart; only the arithmetic is packed here, not the memory access). Decode is
`fp32_16_half_only(wh[idx])`, the wh-alone, one-bit-coarser-than-real-fp16 decode
described in the main Row-summary methodology above. Real A100 numbers on the underlying
kernel: eliminating malloc/free plus a separate reduce launch via atomicAdd combine took
M=1024 from 90.5 to 188.2 GFlop/s (+108%).

**`mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best`** — TOMOS FP32/FP16
(compute: mixed FP16/FP32 — dual `__half2`+float accumulator; thread block: 1D, 256
threads) — shares the same
one-thread-per-row/split-K/shared-x-tile shell as the kernel above [1][2] but always
takes the wide split-K path (no narrow/tiled fallback). It carries a dual accumulator: a
packed `__half2 acc_cheap` for the common (control=0) path, and a scalar
`float acc_extended` for the rare extended-range path, combined only once at the very
end. `mv_fp32_16_hybrid_process_element_branchless` computes BOTH the cheap (direct
bit-reinterpret into `__half2`, `__hfma2`) and extended (reconstructing an approximate
fp32 value from `wh`'s own bits alone — an extended exponent from an offset/direction
pair, mantissa zero-padded for the 14 lower bits `wr` would normally supply — accumulated
in float) paths unconditionally every element, discarding the unwanted one via arithmetic
masking (`cheap_bits_mask = control - 1` zeroes the raw bits before reinterpret; an
`extended_keep`/`extended_scale` pair gates the extended contribution to zero) rather
than branching [6]. This is a direct application of the warp-divergence-probability
argument from the building-blocks section above: with each of 32 lanes independently
having roughly a 10% chance of needing the extended path, a whole warp skips it together
only ~3.4% of the time, so ~97% of warps already pay for both paths serially under
branching — making the branch genuinely free to eliminate. Confirmed on real A100
hardware: the base branchless kernel beats its branchy counterpart by +6.5% to +22% at
every square size measured (M=1024: 159.5→169.8 GFlop/s; M=65536: 652.6→754.8), and the
full "_best" dispatcher adds a further +5.7% to +18.7% on top of that.

**`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best`** — TOMOS FP32/FP16
(fast) (compute: FP16, packed `__half2`; thread block: 1D, 256 threads) — has the
identical
size-fork pattern, accumulator, and shared-memory structure as
`..._half2_atomic_best` above [1][2][5], differing only in decode:
`fp32_16_fast_half(wh[idx])`, an unconditional bit-reinterpret only correct when every
`wh` value already has control=0 — UNSAFE on arbitrary data, valid only on the dedicated
`gen_fp32_16_wh_safe`-generated dataset used by this kernel's own correctness/error
checks.

**`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`** — TOMOS FP32/BF16 (compute:
BF16, packed `__nv_bfloat162`; thread block: 1D, 256 threads) — and
**`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`** — TOMOS BF16 (compute: BF16,
packed `__nv_bfloat162`; thread block: 1D, 256 threads) — are byte-for-byte identical
kernel bodies, both size-forked at 4096 rows between a bigtile and a wide4x variant, and
both always split-K with no narrow/tiled fallback (unlike the fp16 kernels above).
`__shared__ __nv_bfloat16 x_tile[256]` is staged via `__float2bfloat16`; the accumulator
is `__nv_bfloat162 acc01, acc23`, updated via `__hfma2` [5] — genuine packed bf16x2 FMA.
Decode is `fp32_bf16_bf16_only(wh[idx])`, exact for every input because bf16 shares
float32's own 8-bit exponent field [7], leaving no fast/safe split or reconstruction
trick possible for this format. The "_v2" series exists purely to give this codebase's
"TOMOS FP32/BF16" plot slot its own distinctly-named kernel — confirmed byte-identical to
"TOMOS BF16" on purpose, since there is genuinely no decode/compute axis left to
differentiate the two series on for bf16. Both combine via `atomicAdd` [1] the same way
as the pure_half kernel above.

### Row 4 — GEVM 16-bit pure_half, TRANSPOSE

All four kernels below share the warp-per-output-column/split-M/shared-x-staging/
warp-shuffle-tree architecture [2][3][4] common to every TRANSPOSE row, with M split into
`M_BLOCK`-sized chunks (target grid blocks = 2048) and each lane doing a vectorized
`ushort4` load of 4 consecutive `wh` values — a real memory-bandwidth win here, unlike
Row 3's NO_TRANSPOSE tier where only the arithmetic, not the load, is packed.

**`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split`** — TOMOS FP16
(compute: FP16, packed `__half2`; thread block: `dim3(32, 8)` = 256 threads, one warp
per output column) — reuses its own
plain `_vec4` sibling's compute body verbatim — the same `__half`-typed shared x
staging, the same `ushort4`-vectorized `wh` loads [2], the same `__hfma2` accumulation
into `__half2 acc01, acc23` [5], the same `fp32_16_half_only` decode — changing only the
final combine step: instead of `atomicAdd`, it writes each block's partial sum into a
`float*` workspace, and the host wrapper then launches a separate reduce kernel over that
buffer [1]. This caller-owned-workspace split+reduce combine was built specifically to
isolate atomics-vs-split+reduce as the one remaining variable in an earlier performance
investigation into this kernel family's own scaling behavior at large M.

**`mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`** — TOMOS FP32/FP16
(compute: mixed FP16/FP32 — dual `__half2`+float accumulator; thread block:
`dim3(32, 8)` = 256 threads) — uses the same
warp-per-column/split-M/shared-x-staging/warp-shuffle/atomicAdd shell as its siblings
[1][2][3][4], and the same "cheap/extended" dual-accumulator decode design as Row 3's own
hybrid kernel above (`__half2 acc_cheap` + `float acc_extended`), but "paired": it checks
each adjacent `ushort4`-loaded pair's control bits together, and when BOTH are cheap
(control=0 — the common case) it packs them into one real 2-wide `__hfma2` [5] reading
the pair's shared x values as a single `__half2`, instead of the plain hybrid kernel's
per-element calls (each of which pads its own `__half2` with a zero in the unused lane,
wasting half of every FMA's throughput). Only when a pair is mixed does it fall back to
the ordinary branchy `mv_fp32_16_hybrid_process_element` per element for that pair — this
fallback is itself a genuine per-pair branch, not the arithmetic masking Row 3's own
branchless hybrid kernel uses.

**`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast`** — TOMOS FP32/FP16
(fast) (compute: FP16, packed `__half2`; thread block: `dim3(32, 8)` = 256 threads) —
shares the same
accumulator (`__half2 acc01, acc23`, `__hfma2` [5]) and `ushort4`-vectorized-load [2]
structure as the `_split` kernel above, decode swapped to `fp32_16_fast_half`: an
unconditional bit-reinterpret with no control check at all, correct only when every `wh`
value already has control=0 — UNSAFE on arbitrary data. Scalar prefix/suffix elements use
a padded `__hfma2` (one real lane); the aligned vectorized middle does true 2-wide FMA on
`ushort4`-loaded, `fp32_16_fast_half`-decoded pairs, with an alignment check and scalar
fallback.

**`mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4`** — TOMOS FP32/BF16
(compute: BF16, packed `__nv_bfloat162`; thread block: `dim3(32, 8)` = 256 threads) —
is a direct bf16
port of the `_fast` kernel above — `__half`/`__half2` swapped for
`__nv_bfloat16`/`__nv_bfloat162` throughout, `fp32_16_fast_half` swapped for
`fp32_bf16_bf16_only` [7] — but unlike its fp16 "fast" lineage, it carries no
unsafe-on-arbitrary-data caveat at all: a bf16 truncation of any finite fp32 value is
always valid and correctly ranged, so there is only ever one decode function for bf16,
and it was already unconditional.

### Row 5 — GEMM 32-bit reconstruction, NO_TRANSPOSE

**`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`** — TOMOS FP32
(compute: FP32; thread block: 1D, 256 threads — `kGemmFp32DualNumThreads`, logically 8
warps arranged WarpsM=4 × WarpsN=2) — is a single
kernel (falling back to a bounds-checked sibling only when m/n/k aren't exact multiples
of the tile). Tile: BM=BN=128, BK(chunk)=8, TM=TN=8 — 64 outputs/thread, a 2D register
tile [2][8] — with 8 warps arranged WarpsM=4×WarpsN=2 using CUTLASS's own confirmed
`RowMajorInterleaved<2>` warp/lane mapping, read directly from CUTLASS's own source [9].
Accumulation follows CUTLASS's own snake (boustrophedon) sweep instead of plain
row-major — row `i`'s j-direction alternates by `i`'s parity, adding a shared-memory
register-reuse (`.reuse`) opportunity at every row boundary [9], measured to raise this
kernel's own reuse rate from 68.5% (plain row-major) toward CUTLASS's own reference
figure of 87.2%. Global-to-shared prefetch uses a 4-deep rotating shared-memory buffer
with synchronous, vectorized `float4` loads — software-pipelined, but NOT
`cp.async`-based, distinguishing this "syncpipe" family from the cp.async kernels
elsewhere in this row and the next [2]. A mid-loop checkpoint at the 4th of 4 pipeline
stages gives the kernel its "midpoint" name; the "exact" variant drops bounds checks
entirely, valid only once the host wrapper has confirmed m/n are exact multiples of the
128 tile. This is the fastest native (unpacked) fp32 GEMM kernel in the codebase
(0.977x vs CUTLASS at N=8192), and is reused unmodified as the second stage of both
"materialize" kernels below it in this row.

**`mm_fp32_16_tiled_cpasync_dualbuf`** — TOMOS FP32/FP16 (compute: FP32; thread block:
1D, 256 threads — `kMmFp32_16DualNumThreads`) — uses BM=BN=128, BK=16, TM=TN=8, with warps
arranged as a plain 2×4 row-major split (not CUTLASS-interleaved, unlike the kernel
above). It issues genuine `cp.async` [10][11] via `__pipeline_memcpy_async`/
`__pipeline_commit`/`__pipeline_wait_prior`, 2-stage double-buffered — including a
dedicated cp.async prefetch path for B: B's raw bytes are staged into a separate,
K-contiguous staging buffer, then scattered into the actual shared-memory B tile via a
cheap shared-to-shared "transpose" pass, giving B the same asynchronous-copy treatment as
A rather than a synchronous fallback. Accumulates in plain row-major order (no snake).
Decode is `fp32_16_combine` (the full reconstruction decode), applied inline as each tile
is loaded rather than materialized ahead of time.

**`mm_fp32_16_tiled_cpasync_dualbuf_truncated`** — TOMOS FP32/FP16 (truncated) (compute:
FP32; thread block: 1D, 256 threads) — is byte-for-byte the same kernel body as
the one above, differing only in its decode call, `fp32_16_combine_truncated`, which
requires `wh`/`wr` produced by `fp32_16_split_truncated` specifically — not
interchangeable with the plain `fp32_16_split`-produced pair the other two FP16 kernels
in this row share.

**`mm_fp32_16_gemm_materialize_midpoint_exact`** — TOMOS FP32/FP16 (workspace) (compute:
FP32; thread block: its own decode-chunk kernel is 1D, 256 threads, flat elementwise;
the GEMM stage it calls into reuses `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
snake_midpoint_exact`'s own 1D, 256-thread block) — is a genuine two-stage composition, not
a single fused kernel: an elementwise `mm_fp32_16_materialize_chunk_kernel` first decodes
A via `fp32_16_combine` in 4096-row strips into a full fp32 scratch buffer, then a
native, unmodified, black-box call into
`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` (the kernel
described first in this row) runs on that decoded copy, reusing the codebase's own best
native fp32 GEMM kernel rather than re-deriving a fused-decode variant of it. This
decode-to-scratch-then-call-native-GEMM composition is this codebase's own design, with
no independent literature citation. On scratch-allocation failure it falls back to
`mm_fp32_16_tiled_cpasync_dualbuf` instead.

**`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`** — TOMOS FP32/BF16
(truncated) (compute: FP32; thread block: 1D, 256 threads —
`kMmFp32_16MidExactNumThreads`) — uses the same
tile constants (BM=BN=128, BK=8, TM=TN=8, warps 4×2) and the same 4-stage sync-pipe/
snake/midpoint structure as this row's native fp32 kernel [2][8][9], but its own
`load_global_advance` calls `fp32_bf16_combine` [7] inline instead of reading a plain
float — a fused-decode variant rather than a separate materialize pass. Its own header
comment notes bf16's decode here is unconditional and exact for every input, with no
degraded/approximate fallback branch the fp32_16 tier's own kernels need.

**`mm_fp32_bf16_gemm_materialize_rounded`** — TOMOS FP32/BF16 (workspace) (compute: FP32;
thread block: its own decode-chunk kernel is 1D, 256 threads, flat elementwise; the GEMM
stage it calls into reuses the native fp32 kernel's own 1D, 256-thread block) — is the
same two-stage materialize composition
as `mm_fp32_16_gemm_materialize_midpoint_exact` above:
`mm_fp32_bf16_materialize_rounded_chunk_kernel` decodes B via `fp32_bf16_combine_rounded`
[7] into a scratch buffer, then a native call into this row's own native fp32 kernel runs
the actual GEMM. Its plot label ("workspace") describes the materialize strategy rather
than the encoding choice — the ROUNDED bf16 encoding is what actually distinguishes it
from the plain (non-rounded) split used by "TOMOS FP32/BF16 (truncated)" above.

### Row 6 — GEMM 16-bit tensor-core (pure/composite), NO_TRANSPOSE

All five kernels below share a common tile geometry: BM=256, BN=128 (the "sq" tile),
BK=32; an MMA instruction tile of 16×8×16; 4 rows × 2 cols = 8 warps (256 threads), each
warp handling FragM=4 × FragN=8 fragments — 32 `mma.sync` calls per warp per k-step.
Fragment loads use `ldmatrix.sync.aligned` [9][14]; the actual multiply-accumulate is
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (or the bf16 equivalent) — f16/bf16
operands, f32 accumulator, exactly the precision combination independently characterized
in [13]. All five run a 4-stage `cp.async` pipeline, primed 3 tiles ahead [10][11], use
CUTLASS's own serpentine (boustrophedon) `mma.sync` issue order [9], CUTLASS's
`ColumnMajorTensorOpMultiplicandCongruous` shared-memory swizzle (this codebase's own
"acongruous" naming) [9], and CUTLASS's 128-bit vectorized epilogue [9] — all read
directly from CUTLASS header source, not reimplemented independently — plus a
compile-time-provable exact-K "fastk" fast path (roughly 140 instructions/K-tile vs 474
in the general bounds-checked path) and precomputed per-thread address tables for A and B
("precomp2").

**`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`**
— TOMOS FP16 (compute: FP32, hardware-mandated `mma.sync` accumulator over f16 operands;
thread block: 1D, 256 threads — `kGemmFp16WideBigRegThreads`, logically 4 warp-rows × 2
warp-cols × 32 lanes) —
is the reference native fp16 tensor-core implementation this row's other two fp32_16
kernels call into or share the load path with. It is a template on a `SkipBeta` flag
(two instantiations), with a size-based fallback chain to a smaller/simpler kernel for
misalignment or low occupancy, and to a non-fastk sibling when k isn't an exact multiple
of BK. Every shared building block above [9][10][11][12][13][14] applies directly to
this kernel.

**`gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`**
— TOMOS BF16 (compute: FP32, hardware-mandated `mma.sync` accumulator over bf16 operands;
thread block: 1D, 256 threads — the same `kGemmFp16WideBigRegThreads` constant reused
verbatim, confirmed via `__launch_bounds__(kGemmFp16WideBigRegThreads, 1)` on the bf16
kernel itself) —
is a straight, mechanical port of the fp16 kernel above — its own header comment states
the only real changes are `__half`→`__nv_bfloat16` throughout and the `mma.sync` PTX
instruction's own element-type suffix, `f16.f16`→`bf16.bf16` [7][9][10][11][12][13][14] —
confirmed to compile to exactly the same register counts as its fp16 counterpart
(236/218). Its own fallback chain is shorter (only one non-fastk sibling, no
intermediate two-tier fallback).

**`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp`**
— TOMOS FP32/FP16 (compute: FP32, hardware-mandated `mma.sync` accumulator; thread
block: 1D, 256 threads — the same `kGemmFp16WideBigRegThreads` constant) —
reads A as `const uint16_t* wh` only (no `wr` — this tier never needs the reconstruction
pair); B and C are genuine `__half`. It shares the same tile/pipeline/PTX as the native
fp16 kernel above [9][10][11][12][13][14]. Its distinguishing "inline_dirty" decode: A's
raw `wh` bits are `cp.async`'d directly into the same fp16 shared-memory buffer used for
genuine `__half` operands (aliasing `As_wh`/`As`) — there is no separate decode-and-store
step. A per-thread OR-reduce over the thread's own `wh` chunks, combined with a
warp-vote `__any_sync` [15], decides ONCE per warp whether ANY lane's data needs the
(expensive) correction pass via `fp32_16_half_only`, guaranteeing every lane in the warp
takes the same branch and removing SIMT reconvergence overhead entirely — "batch first,
branch once," in the kernel's own words. the

**`mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp`** — TOMOS
FP32/FP16 (workspace) (compute: FP32, hardware-mandated `mma.sync` accumulator; thread
block: its own decode-chunk kernel is 1D, 256 threads, flat elementwise; the GEMM stage
it calls into reuses the native fp16 tensor-core kernel's own 1D, 256-thread block) — is a
two-stage
composition like Row 5's own materialize kernels:
`mm_fp32_16_pure_half_materialize_chunk_vec_kernel` does a vectorized
`ushort4`→`fp32_16_half_only`→`uint2` decode, 4 rows/thread [2], into a persistent,
cached scratch buffer that is never freed on the steady-state path (avoiding a
malloc/free on every call — the feature distinguishing "cached_vec" from Row 5's own
plain materialize kernels), then a native, unmodified call into the fp16 tensor-core
kernel above runs the actual GEMM. No independent citation applies to the materialize
composition itself.

**`mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp`** — TOMOS FP32/BF16 (compute:
FP32, hardware-mandated `mma.sync` accumulator; thread block: no decode kernel of its
own — the whole call is a direct pass-through into the native bf16 tensor-core kernel's
own 1D, 256-thread block) — is not a materialize kernel
at all: a zero-cost `reinterpret_cast<__nv_bfloat16*>(wh)`, with no decode kernel and no
scratch buffer whatsoever, followed by a direct call into the bf16 tensor-core kernel
above. This is possible only because `fp32_bf16_bf16_only(wh)` is unconditionally,
exactly `reinterpret_cast<const __nv_bfloat16*>(&wh)` for every possible input —
bfloat16's 8-bit exponent field is bit-for-bit the same width and bias as float32's own
[7] — making this wrapper the zero-cost limit of the materialize idea for this specific
packed format, not an approximation of it.

## GEMV kernel walkthrough: Row 1 dispatch, and Row 1 vs Row 3 structure

This section works through several Row 1 and Row 3 GEMV kernels in more
implementation detail than the per-kernel summaries above — dispatch thresholds, thread
block shape, shared-memory usage, and a load/decode/FMA/epilogue breakdown — since their
internal structure is what explains *why* Row 1 (32-bit reconstruction) and Row 3
(16-bit pure_half) end up blocking work along genuinely different axes, not just
"the same kernel at a different precision."

### `mv_fp32_tiled`'s own 3-way shape dispatch

TOMOS FP32 picks one of three strategies per call, in this order (`mv_cuda_kernels.cu:927-954`):

| Condition | Strategy | Kernel(s) launched | Why |
|---|---|---|---|
| `rows_op < 131072` (`kMvFp32WideRowThreshold`) | **Wide / split-K** | `mv_fp32_wide_kernel` + `mv_fp32_wide_reduce_kernel` | Too few output rows to fill the GPU with one-thread-per-row alone — `grid.y` (`col_splits`) partitions K across extra blocks, each writing a partial dot product into a `double*` scratch buffer; a second kernel launch reduces the partials per row and applies α/β. Checked *first*, regardless of column count. |
| `rows_op ≥ 131072` and `cols_op < 64` (`kMvFp32NarrowVecColThreshold`) | **Tiled** | `mv_fp32_tiled_kernel` | Plenty of rows already, and K is small — one thread per row, `x` staged in shared memory tile-by-tile, no split-K, no atomics (each thread owns its own `y[i]`). |
| `rows_op ≥ 131072` and `cols_op ≥ 64` | **Narrow-vec / register-blocked** | `mv_fp32_narrow_vec_kernel` | Plenty of rows *and* a large K — register-blocks 4 rows/thread (`kMvFp32RowsPerThread`), one `float4` load of `A` per K-step instead of 4 scalar loads. |

Row count decides *whether split-K is needed at all* (checked first — too few rows means
too few blocks regardless of K); only once there are enough rows does column count
decide between the two non-split strategies. All three tiers share the same
256-thread, 1D thread block (`kMvFp32TileSize`), and this exact 3-way structure is what
the whole `fp32_16`/`fp32_bf16` packed-format family (Row 1's other four kernels)
mirrors shape-for-shape, per the kernel's own header comment.

### `mv_fp32_narrow_vec_kernel` — register-blocked, no shared memory at all

(`mv_cuda_kernels.cu:261-309`.) Each thread owns `kMvFp32RowsPerThread = 4` consecutive
output rows in registers (`i0 = (blockIdx.x*blockDim.x+threadIdx.x) * 4`), and the K-loop
runs *outer*, over `x`, not over rows — so each `x[j]` is read once (via `__ldg`, routed
through the read-only data cache) and reused across all 4 row-accumulators before moving
to the next `j`:

```cpp
double acc0=0, acc1=0, acc2=0, acc3=0;
for (size_t j = 0; j < cols_op; ++j) {
    double x_val = __ldg(&x[j]);
    if (vectorized) {                                    // NO_TRANSPOSE, in-bounds, ld_a%4==0
        float4 a4 = *reinterpret_cast<const float4*>(&A[i0 + j*ld_a]);   // 1 load, 4 rows
        acc0 += a4.x*x_val; acc1 += a4.y*x_val; acc2 += a4.z*x_val; acc3 += a4.w*x_val;
    } else { /* scalar fallback: 4 separate loads */ }
}
```

There is **no `__shared__` memory anywhere in this kernel** and no `__syncthreads()` —
a deliberate contrast to `mv_fp32_tiled_kernel` sitting right above it in the same file
(which explicitly stages `x` into `__shared__ float x_tile[kMvFp32TileSize]` behind two
syncs per tile). The design bet: every thread across the whole grid reads
`x[0], x[1], x[2], ...` in the same order, just at different row offsets, so the
GPU's own read-only cache already captures nearly all the reuse a shared-memory tile
would — without paying for the tiling loop or `__syncthreads()` overhead. Accumulation
runs in `double` for the *entire* reduction (not just the epilogue), even though
`A`/`x`/`y` are all `float`. The vectorized `float4` read on `A` and the scalar,
one-store-per-row write-back to `y` are the same "vectorized load, scalar store"
asymmetry seen in the GEMM epilogues elsewhere in this doc.

### `mv_fp32_16_ready4_atomiconly_scaled_kblock_kernel` — warp-structured, shared-memory-staged, split-K

(TOMOS FP32/FP16, `fp32_16_cuda_kernels.cu:5036-5138`.) Structurally different from
`narrow_vec` in almost every dimension:

- **Thread block**: `dim3(32, kMvReadyWarpsPerBlock)` = `dim3(32, 8)` = 256 threads — 2D,
  warp/lane-structured (`threadIdx.y` selects the warp, `threadIdx.x` is the lane), not
  `narrow_vec`'s flat 1D indexing.
- **Rows per lane**: 4 (`kMvReady4RowsPerWarp = 128` = 4 rows/lane × 32 lanes), so one
  256-thread block covers `8 warps × 128 rows/warp = 1024` output rows.
- **Grid**: `dim3(gridK, gridRows)` — genuinely 2D, because this kernel does split-K
  (unlike `narrow_vec`, which never splits K): `gridK` blocks cooperate on the *same*
  1024-row group, each handling a `K_BLOCK`-wide slice of the reduction.
- **Shared memory**: `extern __shared__ float sx[]`, sized `K_BLOCK * sizeof(float)` at
  launch. All 256 threads cooperatively stage the block's `K_BLOCK`-wide slice of `x`
  into it once, up front, before the K-loop starts:
  ```cpp
  for (int t = warpId*32+lane; t < tileN; t += warpsPerBlock*32) sx[t] = __ldg(&x[kStart+t]);
  __syncthreads();
  ```

**The K-reduction loop** (the "load" step) — a single pass, no software pipeline:

```cpp
#pragma unroll 4
for (int kk = kStart; kk < kEnd; ++kk) {
    uint64_t ah = *reinterpret_cast<const uint64_t*>(wh + row0 + kk*ld);   // 1 load, 4 rows' wh
    uint64_t ar = *reinterpret_cast<const uint64_t*>(wr + row0 + kk*ld);   // 1 load, 4 rows' wr
    float a0 = fp32_16_combine_branchless(ahp[0], arp[0]);                 // decode, x4
    ... a1, a2, a3 ...
    const float xk = sx[kk - kStart];                                      // shared memory, not global
    sum0 = fmaf(a0, xk, sum0); ... sum1, sum2, sum3 ...                     // FMA, x4
}
```

Unlike the GEMM kernels elsewhere in this doc, there is **no explicit prefetch/pipeline**
here: `x` is staged exactly once, up front, for the whole K-slice (removing it from the
hot loop entirely), but `wh`/`wr` are re-loaded from global memory every iteration with
no rotating buffer or `cp.async` overlap — reasonable given the tiny per-iteration
footprint (2×8 bytes) compared to a GEMM tile's kilobytes, leaving little latency worth
hiding with that extra complexity. Whatever overlap does happen is implicit: ordinary
warp-scheduler occupancy, plus whatever `#pragma unroll 4` lets the compiler interleave
across consecutive iterations.

**Epilogue**, per row (α baked into each *partial* sum, not deferred to after a full
combine — correct because `atomicAdd` combine is linear):

```cpp
atomicAdd(&y[row0], alpha * sum0);
if (row1_valid) atomicAdd(&y[row1], alpha * sum1);
if (row2_valid) atomicAdd(&y[row2], alpha * sum2);
if (row3_valid) atomicAdd(&y[row3], alpha * sum3);
```

`β` never appears in this kernel at all — it's applied once, in a separate
`mv_fp32_16_scale_y_kernel` launch, *before* this kernel runs, since it must only scale
the original `y` once, not once per contributing block. No workspace parameter and no
`cudaMalloc` anywhere in the host wrapper — the split-K combine across blocks happens
entirely through the `atomicAdd`s above.

| Step | What happens |
|---|---|
| 0. Pre-launch (host) | `y` pre-scaled by β in a separate kernel launch (only if β≠1.0) |
| 1. Index setup | each thread derives its 4 owned rows and its block's K-slice |
| 2. Shared-memory staging | whole block cooperatively loads its K-slice of `x` into `sx[]`, once |
| 3. K-reduction loop | per `kk`: vectorized `wh`/`wr` load (4 rows) → decode ×4 → shared-mem `x` read → FMA ×4 |
| 4. Epilogue | α applied per partial sum; `atomicAdd` into `y` (this *is* the split-K combine) |

### TOMOS FP16 vs TOMOS FP32/FP16: row-blocking vs column-blocking

`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_kernel` (TOMOS FP16,
`fp32_16_cuda_kernels.cu:18821-18889`) blocks work along the *opposite* axis from
`ready4` — one row per thread, but K-steps (columns) packed 2-at-a-time via genuine
`__half2` SIMD arithmetic:

```cpp
size_t i = blockIdx.x * blockDim.x + threadIdx.x;      // ONE row per thread
__half2 acc01 = ..., acc23 = ...;                        // packed accumulators, not scalar

for (tile_start = col_start; tile_start < col_end; tile_start += kMvTileSize) {   // tile-by-tile
    if (threadIdx.x < tile_len) x_tile[threadIdx.x] = (__half)x[...];
    __syncthreads();
    for (jj = 0; jj+4 <= tile_len; jj += 4) {             // 4 COLUMNS at a time
        __half a0 = fp32_16_half_only(wh[idx0]);           // 4 SEPARATE scalar loads
        ... a1, a2, a3 (each its own load — idx0..idx3 are ld_a apart, not contiguous) ...
        __half2 a01 = __halves2half2(a0, a1);
        __half2 x01 = *reinterpret_cast<__half2*>(&x_tile[jj]);   // x IS vectorized, 2-at-a-time
        acc01 = __hfma2(a01, x01, acc01);
        acc23 = __hfma2(a23, x23, acc23);
    }
    __syncthreads();
}
__half2 acc_sum2 = __hadd2(acc01, acc23);
__half acc = __low2half(acc_sum2) + __high2half(acc_sum2);
atomicAdd(&y[i], (float)alpha * (float)acc);
```

| | TOMOS FP32/FP16 (`ready4`, 32-bit compute) | TOMOS FP16 (`pure_half`, 16-bit compute) |
|---|---|---|
| **Decode** | `fp32_16_combine_branchless(wh, wr)` → `float` (full reconstruction, both wh & wr) | `fp32_16_half_only(wh)` → `__half` (wh-alone, saturates rather than reconstructing) |
| **Compute precision** | scalar `float` FMA (`fmaf`) | packed `__half2` SIMD FMA (`__hfma2`) |
| **Rows per thread** | **4** (register-blocked across *rows*) | **1** — no row-blocking |
| **What gets packed 2/4-at-a-time** | 4 **rows** of the same K-step, via one vectorized `uint64_t` load of `A` | 2 **columns** (K-steps) of the same row, via `__hfma2`'s packed arithmetic |
| **`wh` load** | vectorized (`uint64_t`, 4 rows in one load — contiguous for NO_TRANSPOSE) | **4 separate scalar loads** — the 4 columns are `ld_a` apart, not contiguous; only the arithmetic is packed, not the load |
| **`x` staging** | one shared-memory staging pass for the whole `K_BLOCK` slice, before the K-loop, no further syncs inside it | tile-by-tile staging loop, `__syncthreads()` before *and* after every tile — same pattern as `mv_fp32_tiled_kernel`'s own tiling loop |
| **Thread block shape** | `dim3(32, 8)` = 256 — 2D, warp/lane-structured | flat `256` — 1D, no warp-structured row assignment |
| **Combine** | split-K via `K_BLOCK`-sized grid.x chunks, `atomicAdd`, α baked into each of the 4 partial sums | split-K via `col_splits` grid.y chunks, `atomicAdd`, α baked into the single partial sum this thread owns |
| **Register accumulators** | 4 scalar `float`s (`sum0..sum3`) — one per owned row | 2 packed `__half2`s (`acc01`, `acc23`) — one per pair of owned columns |

The two kernels split parallelism along opposite axes: `ready4` blocks *rows* into a
single thread's registers and keeps arithmetic scalar; the FP16 kernel keeps one row
per thread but blocks *columns* into packed 2-wide arithmetic — which only makes sense
because `__half2` gives a genuine 2-element SIMD multiply-add plain `float` has no
equivalent for. This is also the concrete mechanism behind why this tier's error grows
with K (see the main Row-summary methodology above): the K-reduction itself, not just
storage, accumulates in `__half2`, term by term, rather than in `float` the way
`ready4`'s reduction does.

## Full exploratory suites

These predate (and are much larger than) the focused `verify_*_plot_promoted.cu` checks
above — they cover every experimental/demoted kernel variant this project has ever
built, not just the currently-promoted set, and use the OLDER per-element max-relative-
error metric (not yet retrofit to L2/Frobenius):

- `verify_mv_fp32_16_gevm.cu` (`verify_mv_fp32_16_gemv.sh`) — ~50 GEVM kernels, both
  TRANSPOSE directions, every `fp32_16`/`fp32_bf16` tier.
- `verify_mm_fp32_gemm.cu` (`verify_mm_fp32_gemm.sh`) — native fp32 GEMM kernel family.
- `verify_mm_fp32_16_gemm.cu` (`verify_mm_fp32_16_gemm.sh`) — full fp32-reconstruction
  GEMM tier, both `fp32_16` and `fp32_bf16`.
- `verify_mm_fp32_16_pure_half_gemm.cu` (`verify_mm_fp32_16_pure_half_gemm.sh`) — GEMM
  pure/composite tier's own `fp32_16` half-decode family.
- `verify_mm_fp16_gemm.cu` (`verify_mm_fp16_gemm.sh`) — native fp16/bf16 tensor-core GEMM
  kernel family.
- `verify_bf16_wide_bigreg_sq_deep4.cu` — native bf16 tensor-core GEMM, focused on the
  `wide_bigreg_sq_deep4` kernel family specifically.

Use a focused `verify_*_plot_promoted.cu` check when the question is "are the kernels
this specific plot shows still correct" (fast, run after any change to one of those
kernels); use a full exploratory suite when the question is "did I break anything across
every variant this project has ever tried" (slow, comprehensive regression sweep).

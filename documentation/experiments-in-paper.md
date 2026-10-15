# TOMOS

**Local setup:** benchmarks that compare against CUTLASS (any
`scripts/*vs_cutlass_ref*.sh`) default to fetching/using it at
`build/cutlass-ref`; LSQR's test matrices (see this document's own `##
LSQR` section) default to `build/matrices`. For a persistent custom
location instead of passing `CUTLASS_DIR=`/`--matrix-dir` on every
invocation, copy `.configure.sh.example` (repo root) to `.configure.sh`
and set `CUTLASS_DIR`/`MATRIX_DIR` there — it's gitignored and sourced
automatically by every script that needs these.

## Plots - Throughput

### FP32 GEMM

#### `build/output/benchmarks/gemm/fp32_compute/plots/plot_gemm_format_32_16_compute_32_square.png` / `build/output/benchmarks/gemm/fp32_compute/plots/plot_gemm_format_32_16_compute_32_rect.png`

##### Data

`build/output/benchmarks/gemm/fp32_compute/data/` — timestamped `.csv`/`.txt`
benchmark logs used to generate these plots, written by
`scripts/benchmarks_gemm_format_32_16_compute_32.sh`.

##### Experiment:

The "32-bit-compute" GEMM sweep: cuBLAS (Sgemm), real CUTLASS (SIMT fp32
`device::Gemm`), this codebase's own native (unpacked) fp32 GEMM, and the
32-bit-compute tier of both packed formats -- decode `wh`/`wr` back to a
real fp32 value before computing, as opposed to the "16-bit-compute" tier
below that computes directly in half/bf16. `square` is the M=K=N sweep;
`rect` holds M=N fixed and sweeps the inner dimension K.

**Scripts:**
- `scripts/benchmarks_gemm_format_32_16_compute_32.sh` — builds and runs
  the benchmark.
- `scripts/plot_gemm_format_32_16_compute_32.sh` — regenerates the two
  `.png`/`.fig` plots from an already-saved CSV (no rebuild/rerun); wraps
  the MATLAB function `scripts/plot_gemm_format_32_16_compute_32.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu`.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS | `cublasSgemm` | — | external library (cuBLAS) |
| CUTLASS | `cutlass::gemm::device::Gemm<...>` (SIMT fp32) | — | external library (CUTLASS) |
| TOMOS FP32 | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | — (native, unpacked storage) | also the tail call of both "(workspace)" series below |
| TOMOS FP32 / FP16 | `mm_fp32_16_tiled_cpasync_dualbuf` | `fp32_16_combine` | decode fused inline into the GEMM kernel; branchy (three-way normal/extended/saturate dispatch, see this session's own discussion of `fp32_16_combine`'s two real `if` branches) |
| TOMOS FP32 / FP16 (truncated) | `mm_fp32_16_tiled_cpasync_dualbuf_truncated` | `fp32_16_combine_truncated` | pairs with the truncating (non-rounding) encoder `fp32_16_split_truncated` |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_gemm_materialize_midpoint_exact` | `fp32_16_combine` (via `mm_fp32_16_materialize_chunk_kernel`) | decodes the whole matrix to a scratch buffer up front, then calls the native TOMOS FP32 kernel above unmodified; no subnormal-density sensitivity measured (see `scripts/fp32_16_workspace_subnormal_density_table.sh`), unlike dualbuf's real ~3.7% dip under 10% subnormal density |
| TOMOS FP32 / BF16 (truncated) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | `fp32_bf16_combine` | fused inline decode; no branch at all (bf16's exponent field matches fp32's own bit-for-bit, so there's no degraded-range correction to dispatch on) |
| TOMOS FP32 / BF16 (workspace) | `mm_fp32_bf16_gemm_materialize_rounded` | `fp32_bf16_combine_rounded` (via `mm_fp32_bf16_materialize_rounded_chunk_kernel`) | decodes to a scratch buffer up front (rounded encoder/decoder pair, not truncating), then calls the native TOMOS FP32 kernel above unmodified |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (declarations in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family: `fp32_16_split`/`_clamped`/`_truncated`, `fp32_16_combine`/
  `_truncated`), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family:
  `fp32_bf16_split`/`_rounded`, `fp32_bf16_combine`/`_rounded`).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu`.

**Method Description:**
- **cuBLAS** (`cublasSgemm`) — NVIDIA's closed-source, standard single-precision
  GEMM; external reference baseline, not part of this codebase.
- **CUTLASS** (`cutlass::gemm::device::Gemm<...>`, SIMT config) — NVIDIA's
  templated fp32 GEMM using plain FMA (SIMT) instructions, not tensor cores
  (fp32 tensor-core paths need TF32 truncation, which this comparison
  deliberately avoids); external reference baseline.
- **TOMOS FP32** (`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`) —
  this codebase's own tuned native fp32 GEMM, evolved through a long chain of
  isolated single-lever experiments (each suffix names one change kept over
  its predecessor):
  - `syncpipe4`: a 4-stage rotating shared-memory buffer using plain
    synchronous global loads (no `cp.async`) -- fetches tile *i+3* while
    consuming tile *i*, giving three tiles of independent lookahead work to
    hide load latency behind.
  - `bk8`: the K-dimension tile width is 8 (not a wider baseline), which
    halves each pipeline stage's shared-memory footprint -- specifically
    what makes the 4th `syncpipe4` stage affordable within the same memory
    budget.
  - `warptile`: the warp/lane-to-output-tile mapping used for the compute
    (read) path matches CUTLASS's own confirmed `WarpShape<32,64,8>` +
    `MmaSimtPolicy<..., RowMajorInterleaved<2>, ...>` geometry, rather than
    this file's earlier plain row-major warp/lane split; the tile-fetch
    (load) path is unaffected.
  - `nostride`: drops precomputed stride local variables (`a_stride`,
    `b_col_stride`, etc.), inlining the multiply at each use site instead --
    a register-pressure trick, not a memory-access-pattern change (lets
    `ptxas` reload from constant memory on demand rather than pin dedicated
    registers for the kernel's whole lifetime).
  - `snake`: the per-thread FFMA accumulate loop's own inner traversal order
    is boustrophedon -- even output rows sweep the inner `j` index forward,
    odd rows sweep it backward, so consecutive iterations connect at the
    same `j` instead of wrapping from `j=TN-1` back to `0` -- improving
    register-level operand reuse between consecutive FFMA instructions
    (this is a per-thread register-reuse pattern, not a tile-visitation or
    cache-locality scheme).
  - `midpoint`: an interleaved register-pressure checkpoint partway through
    the row loop, restructuring when loads happen relative to compute to
    manage spill. `_exact` additionally drops bounds-checking entirely
    (unconditional loads), valid only when `m`/`n` are exact multiples of
    the tile size and `k` is an exact multiple of `BK=8` (falls back to the
    bounds-checked `_midpoint` kernel otherwise).

  No decode step -- operates on plain unpacked fp32 storage throughout, and
  is the tail call every "(workspace)" kernel below calls after decoding.
- **TOMOS FP32 / FP16** (`mm_fp32_16_tiled_cpasync_dualbuf`) — a `cp.async`
  double-buffered tiled GEMM: while computing tile *N* (already staged and
  decoded), it concurrently issues the `cp.async` copy for tile *N+1*'s raw
  `wh`/`wr` bytes; decode itself (`fp32_16_combine`) runs strictly after
  that copy completes and before the next tile's compute, so it sits on the
  critical path rather than overlapping with any transfer.
- **TOMOS FP32 / FP16 (truncated)** (`mm_fp32_16_tiled_cpasync_dualbuf_truncated`) —
  byte-for-byte the same kernel as the one above, decoding via
  `fp32_16_combine_truncated` instead: `wh`/`wr` were packed by the
  truncating (non-rounding) encoder `fp32_16_split_truncated`, so this
  reconstructs a merely-truncated, not correctly-rounded, fp32 value.
- **TOMOS FP32 / FP16 (workspace)** (`mm_fp32_16_gemm_materialize_midpoint_exact`) —
  a two-pass approach: first decodes the *entire* matrix `wh`/`wr` -> fp32
  into a scratch buffer via `mm_fp32_16_materialize_chunk_kernel`
  (`fp32_16_combine`, one thread per element, no fused compute), chunked
  to bound scratch memory; then calls the native TOMOS FP32 kernel above,
  completely unmodified, on that scratch buffer. Because decode is now a
  separate pass rather than interleaved with compute, it shows no
  subnormal-density sensitivity (unlike dualbuf's real ~3.7% slowdown
  under 10% subnormal density -- see `scripts/fp32_16_workspace_subnormal_density_table.sh`).
- **TOMOS FP32 / BF16 (truncated)** (`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`) —
  despite the similar name, this is a sibling of **TOMOS FP32** above (same
  `syncpipe4`/`bk8`/`warptile`/`nostride`/`snake`/`midpoint_exact`
  synchronous-pipelined FFMA architecture, not the `cp.async` `dualbuf`
  architecture the FP16 kernels below it use), with decode fused directly
  into the staging step via `fp32_bf16_combine`, which has no branch at
  all: bf16's exponent field is bit-for-bit identical to fp32's own, so
  there is no degraded-range case to dispatch on, only a direct bit
  concatenation.
- **TOMOS FP32 / BF16 (workspace)** (`mm_fp32_bf16_gemm_materialize_rounded`) —
  same two-pass materialize-then-call-native structure as the FP16
  workspace kernel above, but via `fp32_bf16_combine_rounded` (paired with
  the rounding, not truncating, bf16 encoder `fp32_bf16_split_rounded`).

### FP16 GEMM

#### `build/output/benchmarks/gemm/fp16_compute/plots/plot_gemm_format_32_16_compute_16_square.png` / `build/output/benchmarks/gemm/fp16_compute/plots/plot_gemm_format_32_16_compute_16_rect.png`

##### Data

`build/output/benchmarks/gemm/fp16_compute/data/` — timestamped `.csv`/`.txt`
benchmark logs used to generate these plots, written by
`scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh`.

##### Experiment:

The "16-bit-compute" GEMM sweep: cuBLAS, real CUTLASS, this codebase's own
native (unpacked) FP16/BF16 GEMM, and the packed-precision `fp32_16`/
`fp32_bf16` formats' 16-bit-compute tier (decode straight to a half/bf16
value and compute in that precision throughout, as opposed to the
"32-bit-compute" tier above that reconstructs a real FP32 value before
computing). `square` is the M=K=N sweep; `rect` holds M=N=44000 fixed and
sweeps the inner dimension K.

**Scripts:**
- `scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh` — builds and runs the
  benchmark.
- `scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh` — regenerates the two
  `.png`/`.fig` plots from an already-saved CSV (no rebuild/rerun); wraps
  the MATLAB function `scripts/plot_fp16_bf16_pure_vs_cutlass_ref.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu`.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` | — | external library (cuBLAS) |
| cuBLAS BF16 | `cublasGemmEx` | — | external library (cuBLAS) |
| CUTLASS FP16 | `cutlass::gemm::device::Gemm<...>` (fp16 tensor-op) | — | external library (CUTLASS) |
| CUTLASS BF16 | `cutlass::gemm::device::Gemm<...>` (bf16 tensor-op) | — | external library (CUTLASS) |
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | also the tail call of "(workspace)" below |
| TOMOS BF16 | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | also the tail call of "TOMOS FP32 / BF16" below |
| TOMOS FP32 / FP16 | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` | decode fused inline into the GEMM kernel; compute stays fp16 throughout |
| TOMOS FP32 / FP16 (fast) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp` | `fp32_16_half_only_fast` | lossy: underflow saturates to signed zero instead of a real fp16 subnormal |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` (via `mm_fp32_16_pure_half_materialize_chunk_vec_kernel`) | decodes the whole matrix to a cached scratch buffer up front, then calls the native TOMOS FP16 kernel above unmodified |
| TOMOS FP32 / BF16 | `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` | none — `wh` reinterpreted directly as `__nv_bfloat16*` | no decode kernel at all: bf16's exponent field matches fp32's own bit-for-bit, so truncated `wh` is always already a valid bf16 pattern; calls the native TOMOS BF16 kernel above directly |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/mm_cuda_kernels.cu`
  (declarations in `src/kernels/cuda/matrix/mm_cuda_kernels.cuh`).
- Decode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16 family),
  `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu`.

**Method Description:**
- **cuBLAS FP16 / BF16** (`cublasHgemm` / `cublasGemmEx`) — NVIDIA's
  closed-source half/bfloat16 GEMM; external reference baselines, not part
  of this codebase.
- **CUTLASS FP16 / BF16** (`cutlass::gemm::device::Gemm<...>`, tensor-op
  config) — NVIDIA's templated tensor-core GEMM for each format; external
  reference baselines.
- **TOMOS FP16 / TOMOS BF16** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`,
  `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`) —
  this codebase's own tuned native GEMM for each format, issuing real
  tensor-core `mma.sync.aligned.m16n8k16` PTX instructions directly
  (confirmed by inline `asm volatile` in the kernel body, not a `wmma::`
  wrapper). Evolved through a long chain of isolated single-lever
  experiments, same convention as the FP32 kernel above:
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
  call the matching "(workspace)"/"TOMOS FP32 / BF16" kernel below calls
  after decoding.
- **TOMOS FP32 / FP16** (`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp`) —
  same tensor-core kernel lineage as TOMOS FP16 above (`acongruous`/
  `fastk`/`precomp2` mean the same things here), with one further change,
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
  loads, four elements decoded per instruction sequence via
  `fp32_16_half_only_branchless_no_overflow_nomask`) into a **cached**
  scratch buffer (reused across calls, avoiding the plain materialize
  kernel's repeated `cudaMalloc`/`cudaFree`), then calls the native TOMOS
  FP16 kernel above completely unmodified.
- **TOMOS FP32 / BF16** (`mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp`) —
  no decode kernel exists for this one at all: bf16's exponent field
  matches fp32's own bit-for-bit, so the truncated `wh` half of the packed
  value is *already* a valid `__nv_bfloat16` bit pattern for any input,
  unconditionally. The kernel just reinterprets `wh` as `__nv_bfloat16*`
  directly and calls the native TOMOS BF16 kernel above on it, with zero
  decode instructions of any kind.

### FP32 GEMV

#### `build/benchmarks/output/plots/plot_gemv_fp32_recon_vs_cutlass_ref_square.png` / `build/benchmarks/output/plots/plot_gemv_fp32_recon_vs_cutlass_ref_tall.png`

##### Data

`build/mv_fp32_recon_vs_cutlass_ref_output/` — timestamped `.csv`/`.txt`
benchmark logs used to generate these plots, written by
`scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh`.

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x + beta*y`) fp32-compute GEMV sweep:
cuBLAS (`Sgemv`), this codebase's own tuned native (unpacked) fp32 GEMV
kernel, and the fp32-compute ("full reconstruction") tier of both packed
formats -- decode `wh`/`wr` back to a real fp32 value, then accumulate in
fp32, mirroring the FP32 GEMM sweep's own "32-bit-compute" tier above.
Real CUTLASS (fp32 SIMT `device::Gemv`) is built and timed by the same
benchmark binary but deliberately not plotted (its own numbers are a
documented, non-tunable naive baseline, not a meaningful comparison
point). GEMV is bandwidth-bound rather than compute-bound, so plotted
values stay in GFlop/s (not TFlop/s, unlike the GEMM plots above) across
the whole sweep. `square` is the M=K sweep; `tall` is the tall-and-thin
M>>K sweep, with K held fixed at 512 and 4096 (both overlaid on the same
axes).

**Scripts:**
- `scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh` — builds and runs
  the benchmark.
- `scripts/plot_gemv_fp32_recon_vs_cutlass_ref.sh` — regenerates the two
  `.png`/`.fig` plots from an already-saved CSV (no rebuild/rerun); wraps
  the MATLAB function `scripts/plot_gemv_fp32_recon_vs_cutlass_ref.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu`.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS | `cublasSgemv` | — | external library (cuBLAS); CUTLASS's `device::Gemv` is also timed by this benchmark but not plotted (see Experiment above) |
| TOMOS FP32 | `mv_fp32_tiled` | — (native, unpacked storage) | 3-way shape dispatcher, thresholds shared with the packed-format kernels below |
| TOMOS FP32 / FP16 | `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` | `fp32_16_combine_branchless` | 4-rows/lane row-vectorized (packed `uint64_t` loads), prescale+`atomicAdd`, K-scaled `K_BLOCK`; safe on arbitrary data |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded` | `fp32_bf16_combine_rounded` | pairs with the rounding encoder `fp32_bf16_split_rounded` (`wr`'s bit0 is a round-flag, not a mantissa bit) |
| TOMOS FP32 / BF16 (truncated) | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` | `fp32_bf16_combine` | pairs with the truncating encoder `fp32_bf16_split`; bit-exact reconstruction, no approximation |
| TOMOS FP32 / FP16 (async) | `mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow` | `fp32_16_combine_branchless_no_overflow` | `cp.async` double-buffered prefetch of `wh`/`wr` + 8-rows/lane register blocking; decode itself is unchanged per-element cost, the name refers to the prefetch/blocking, not a decode trick; UNSAFE on arbitrary data (assumes no overflow-encoded values) |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`/`fp32_bf16` GEMV kernels, declarations in
  `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`), `src/kernels/cuda/matrix/mv_cuda_kernels.cu`
  (native fp32 GEMV kernels, declarations in `src/kernels/cuda/matrix/mv_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family). The
  "(async)" series' `fp32_16_combine_branchless_no_overflow` decode is
  instead defined directly in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`.
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu`.

**Method Description:**
- **cuBLAS** (`cublasSgemv`) — NVIDIA's closed-source, standard
  single-precision GEMV; external reference baseline, not part of this
  codebase.
- **TOMOS FP32** (`mv_fp32_tiled`) — this codebase's own tuned native fp32
  GEMV kernel, a 3-way shape dispatcher (same thresholds `mv_fp32_16_accurate`
  below shares): a wide-row split-column path (`mv_fp32_wide_kernel` +
  `mv_fp32_wide_reduce_kernel`, parallel-reduced via a scratch buffer) when
  the output dimension is small, a plain tiled kernel (`mv_fp32_tiled_kernel`)
  for narrow shapes, and a row-grouped vectorized kernel (`mv_fp32_narrow_vec_kernel`)
  otherwise. No decode step -- operates on plain unpacked fp32 storage
  throughout.
- **TOMOS FP32 / FP16** (`mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock`) —
  4-rows/lane row-vectorized (packed `uint64_t` loads of `wh`/`wr`),
  prescale + `atomicAdd` combine, K-scaled `K_BLOCK` structure, decoding
  via the FULL `fp32_16_combine_branchless` (all three of its
  normal/extended/degraded reconstruction paths, computed unconditionally)
  -- safe on arbitrary caller data, needs no external workspace/scratch
  buffer. Promoted over its 2-row predecessor (1.070x-1.115x, measured on
  a tall-and-thin M sweep at K=4096).
- **TOMOS FP32 / BF16** (`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`) —
  the same row-vectorized/`atomicAdd`/scaled-`K_BLOCK` structure as TOMOS
  FP32/FP16 above, decoding via `fp32_bf16_combine_rounded` against data
  split with `fp32_bf16_split_rounded`: `wh` is a properly ROUNDED (not
  truncated) standalone bfloat16 value, and `wr`'s bit0 is a round-flag
  (recording whether that rounding stepped `wh` up by 1) rather than a
  real mantissa bit, so decode subtracts it back out.
- **TOMOS FP32 / BF16 (truncated)** (`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace`) —
  the original fp32_bf16 GEMV series: decodes via plain `fp32_bf16_combine`
  against data split with plain `fp32_bf16_split` (`wh` formed by
  truncation, `wr` the exact remaining mantissa bits) -- bit-exact
  reconstruction, no approximation at all. Generalized to every
  `TransposeType`/`ld_a` via a caller-provided workspace buffer instead of
  a malloc/free-per-call, same as the "(rounded)" series above.
- **TOMOS FP32 / FP16 (async)** (`mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow`) —
  combines `cp.async` double-buffered prefetch of `wh`/`wr` with 8-rows/lane
  register-blocking and `fp32_16_combine_branchless_no_overflow` (skips
  the overflow escape path entirely). Named "(async)" for its actual
  performance driver -- the `cp.async` prefetch plus vectorized `uint4`
  loads/8-row register blocking, not a SIMD/paired decode trick (decode
  itself is still called once per scalar `(wh, wr)` element, same as plain
  "TOMOS FP32 / FP16"'s own decode). Still UNSAFE on arbitrary data --
  valid only under the caller's guarantee this matrix's data never needs
  `wh`'s overflow encoding. Measured (square sweep) up to ~1.16x plain
  "TOMOS FP32 / FP16" at large M, approaching native fp32 parity (e.g.
  701.7 vs native's 700.1 GFlop/s at M=K=65536).

#### `build/benchmarks/output/plots/plot_gemv_transpose_fp32_recon_vs_cublas_ref_square.png` / `build/benchmarks/output/plots/plot_gemv_transpose_fp32_recon_vs_cublas_ref_tall.png`

##### Data

`build/mv_fp32_transpose_recon_vs_cublas_ref_output/` — timestamped
`.csv`/`.txt` benchmark logs used to generate these plots, written by
`scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh`.

##### Experiment:

The TRANSPOSED (`y = alpha*A^T*x + beta*y`) fp32-compute GEMV sweep: same
six method slots as the NO_TRANSPOSE plot above, minus CUTLASS entirely
(TOMOS-vs-cuBLAS only, by request -- no CUTLASS clone/include-path setup
needed for this benchmark). Every TOMOS kernel here is sourced from a
dedicated TRANSPOSE fast path: warp-per-output-column, with the reduction
dimension (M) split into chunks across `blockIdx.x` and `atomicAdd`
combining partial sums, `x` staged into shared memory once per block and
reused by every warp in that block. `square` is the M=K sweep; `tall` is
the tall-and-thin M>>K sweep, K held fixed at 512 and 4096.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh` — builds
  and runs the benchmark.
- `scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.sh` —
  regenerates the two `.png`/`.fig` plots from an already-saved CSV (no
  rebuild/rerun); wraps the MATLAB function
  `scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.m`.
- Both read/write `benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_vs_cublas_ref.cu`.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS | `cublasSgemv` (`CUBLAS_OP_T`) | — | external library (cuBLAS) |
| TOMOS FP32 | `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec` | — (native, unpacked storage) | `float4`-vectorized loads of A, the shared `x` tile, AND the `x`-staging loop itself; runtime 16-byte alignment check with scalar fallback |
| TOMOS FP32 / FP16 | `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch` | `fp32_16_combine_branchless_wrapborrow` | thin host-side router (M<8192 vs M>=8192) over two independently-tuned siblings sharing this decode formula; safe on arbitrary data |
| TOMOS FP32 / FP16 (implicit) | `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch` | `fp32_16_combine_fast_normal_wrapborrow` | same dispatcher structure as the series above, over the UNSAFE "normal-path-only" decode reformulation; label distinct from the FP16 GEMM tier's own "(fast)" (a different lossy-underflow meaning) |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded` | `fp32_bf16_combine_rounded` | pairs with the rounding encoder `fp32_bf16_split_rounded` |
| TOMOS FP32 / BF16 (truncated) | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4` | `fp32_bf16_combine` | pairs with the truncating encoder `fp32_bf16_split`; bit-exact reconstruction |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`/`fp32_bf16` TRANSPOSE GEMV kernels, declarations in
  `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`), `src/kernels/cuda/matrix/mv_cuda_kernels.cu`
  (native fp32 TRANSPOSE GEMV kernel, declarations in `src/kernels/cuda/matrix/mv_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family; `fp32_16_combine_fast_normal_wrapborrow` lives here),
  `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family). The "(implicit)"
  series' safe counterpart, `fp32_16_combine_branchless_wrapborrow`, is
  instead defined directly in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`.
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_vs_cublas_ref.cu`.

**Method Description:**
- **cuBLAS** (`cublasSgemv`, `CUBLAS_OP_T`) — NVIDIA's closed-source,
  standard single-precision GEMV run in transpose mode; external
  reference baseline, not part of this codebase.
- **TOMOS FP32** (`mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec`) —
  warp-per-output-column, M split into chunks across `blockIdx.x` with
  `atomicAdd` combining partial sums; `float4`-vectorized loads of A and
  the shared `x` tile, AND the `x`-staging loop itself (global `x` ->
  shared `sx`), each behind a runtime 16-byte alignment check with a
  scalar fallback (always correct, vectorization is a pure throughput
  bonus when hit). Promoted through two rounds on real A100 hardware:
  first over "(unvectorized)" (`mv_fp32_tiled`, this codebase's original
  TRANSPOSE fast kernel, scalar everywhere) by vectorizing the main loop
  alone, then over that "(vec4)" intermediate by also vectorizing the
  staging loop -- reaches parity with or beats cuBLAS from M=12288 upward
  (0.999x-1.081x).
- **TOMOS FP32 / FP16** (`mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch`) —
  not a new tuned `__global__` kernel: a thin host-side router calling
  "(wrapborrow)" directly for M<8192, and an internal `retarget2` sibling
  (same M_BLOCK-scaling idea, no visible CSV column of its own) for
  M>=8192. The underlying kernel is warp-per-output-column/split-M/
  `atomicAdd` (same structure as TOMOS FP32 above), decoding via
  `fp32_16_combine_branchless_wrapborrow` -- a wraparound-subtraction
  reformulation of `fp32_16_combine_branchless` confirmed via `cuobjdump`
  to eliminate all 26 `SEL` instructions and reduce `ISETP` 46->33.
  Promoted on real A100 hardware: tracks "(wrapborrow)" almost exactly
  below M=8192 and its `retarget2` sibling almost exactly at/above it, no
  regression found anywhere tested (M=1024-12288 at promotion time).
- **TOMOS FP32 / FP16 (implicit)** (`mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch`) —
  same dispatcher structure as TOMOS FP32/FP16 above, over
  `fp32_16_combine_fast_normal_wrapborrow` (a bit-identical, wraparound-
  subtraction reformulation of `fp32_16_combine_fast_normal`, exhaustively
  verified over all 1024 possible `(mant_hi, round_flag)` combinations) --
  UNSAFE on arbitrary data, computing only the "normal" reconstruction
  path. Reaches 0.87x-0.96x of cuBLAS from M=3072 upward.
- **TOMOS FP32 / BF16** (`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded`) —
  byte-for-byte the same warp-per-output-column/split-M/`atomicAdd`/
  M_BLOCK-scaling structure as TOMOS FP32/FP16's own "(vec4)" TRANSPOSE
  kernel, ported directly per this codebase's "base the design on FP32/
  FP16" convention, decoding via `fp32_bf16_combine_rounded` (a single
  shift+OR against rounded-split data -- unlike fp16's decode, there is no
  branch here for a wrapborrow-style reformulation to remove).
  Float4-vectorized `x`-staging AND `ushort4`-vectorized `wh`/`wr` loads.
  Promoted: a monotonically growing win over "(rounded, workspace)" on
  real A100 hardware, already beating cuBLAS outright from M=4096 up.
- **TOMOS FP32 / BF16 (truncated)** (`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4`) —
  the truncating-encoding counterpart of TOMOS FP32/BF16 above (byte-for-
  byte identical except decoding via plain `fp32_bf16_combine` against
  plain-truncating-split data). Promoted: a dramatic, monotonically
  growing win over "(truncated, workspace)" on real A100 hardware, already
  beating cuBLAS outright from M=4096 up.

### FP16 GEMV

#### `build/benchmarks/output/plots/plot_gemv_fp32_16_pure_half_vs_cublas_ref_square_final.png` / `build/benchmarks/output/plots/plot_gemv_fp32_16_pure_half_vs_cublas_ref_tall_final.png`

##### Data

`build/mv_fp32_16_pure_half_vs_cublas_ref_output/` — timestamped
`.csv`/`.txt` benchmark logs used to generate these plots, written by
`scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh`.

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x + beta*y`) GENUINE 16-bit-compute GEMV
sweep: cuBLAS (fp16/bf16, via `n=1` GEMM -- no native `cublasHgemv`/
`cublasXgemv` for either dtype exists), this codebase's own native fp16/
bf16 GEMV, and the 16-bit-compute tier of packed `fp32_16` -- decode
straight to a `__half`/`__half2` value and accumulate in that precision
throughout, mirroring the FP16 GEMM sweep's own "16-bit-compute" tier
(there is no packed `fp32_bf16` 16-bit-compute NO_TRANSPOSE series here
distinct from native TOMOS BF16 -- see "TOMOS FP32 / BF16" below). The
`_square`/`_tall` pair for this same benchmark (no `_final` suffix) plots
the FULL, much larger exploratory series set (register-blocking-width
sweeps, cp.async/SWAR-decode A/B candidates, an LSQR-accuracy-motivated
"flush" family, several `test-new*` slots) this `_final` pair is reduced
from -- every series below is one of that full set's own headline/best
candidates, pulled out with byte-identical values/line style; see the
`.m` file's own header comment for the full exploratory derivation not
repeated here. `square` is the M=K sweep; `tall` is the tall-and-thin
M>>K sweep, K held fixed at 512 and 4096.

**Scripts:**
- `scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh` — builds
  and runs the benchmark; reads/writes `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_vs_cublas_ref.cu`.
- `scripts/plot_gemv_fp32_16_pure_half_vs_cublas_ref.m` — the MATLAB
  function that regenerates all four `.png`/`.fig` plots (full and
  `_final` pairs) from an already-saved CSV; unlike the FP32 GEMV/FP32
  GEMM/FP16 GEMM plots above, there is no separate `.sh` wrapper around
  it -- call the `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` (`n=1`) | — | external library (cuBLAS); GEMV-via-GEMM workaround, no native `cublasHgemv` |
| cuBLAS BF16 | `cublasGemmEx` (`CUDA_R_16BF`, `n=1`) | — | external library (cuBLAS); same GEMV-via-GEMM workaround, no native bf16 GEMV entry point |
| TOMOS FP16 | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best` | `fp32_16_half_only` | dispatcher: <4096 rows calls `..._bigtile_half2_atomic`, >=4096 calls `..._half2_atomic_wide4x`; genuine `__half2`/`__hfma2` compute |
| TOMOS BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best` | `fp32_bf16_bf16_only` | same dispatcher structure/threshold as TOMOS FP16, ported to `__nv_bfloat162`/`__hfma2` |
| TOMOS FP32 / FP16 | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform` | `fp32_16_half_only` | `cp.async`-staged `wh` (off the register-mediated critical path) + warp-uniform staging issuer (eliminates intra-warp divergence during staging); safe on arbitrary data |
| TOMOS FP32 / FP16 (SIMD decode) | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar` | `fp32_16_half2_decode_swar` | same kernel structure as the series above; SWAR-paired decode (two `wh` values per 32-bit register), bit-exact vs the scalar decode |
| TOMOS FP32 / FP16 (fast) | `mv_fp32_16_pure_half_no_transpose_flush256_fast_swar` | `fp32_16_half2_decode_fast_swar` (vectorized) / `fp32_16_half_only_fast` (scalar fallback) | lossy: whole underflow band saturates to signed zero; dispatcher below `kMvWideRowThreshold` calls a dedicated fast wide-ILP kernel, at/above it a flush-every-256-columns narrow_vec kernel |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2` | `fp32_bf16_bf16_only` | a deliberate duplicate of TOMOS BF16's own symbols under its own name, so this slot gets an independently-measured benchmark leg; no separate decode/compute axis exists for bf16 (unlike fp16) |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`/`fp32_bf16` GEMV kernels, declarations in
  `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`). `fp32_16_half2_decode_swar`
  and `fp32_16_half2_decode_fast_swar` are defined directly in this file,
  not in the core pack header.
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp`
  (`fp32_16_half_only`/`_fast`), `src/core/matrix/fp32_bf16_pack.hpp`
  (`fp32_bf16_bf16_only`).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_vs_cublas_ref.cu`.

**Method Description:**
- **cuBLAS FP16 / BF16** (`cublasHgemm` / `cublasGemmEx`, both `n=1`) —
  NVIDIA's closed-source half/bfloat16 GEMM run as a GEMV via the
  standard "N=1" workaround (neither dtype has a native `cublasXgemv`
  entry point in this codebase's CUDA toolkit); external reference
  baselines, not part of this codebase.
- **TOMOS FP16** (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`) —
  a host-side dispatcher over two siblings sharing the same genuine
  `__half2`/`__hfma2` compute and `fp32_16_half_only` decode: below 4096
  rows, a "bigtile" variant with a 4x-larger shared-memory `x` tile
  combined via `atomicAdd`; at/above it, the same `atomicAdd` combine
  with `col_splits` quadrupled instead ("wide4x"). Built after Nsight
  Compute profiling of the plain `half2_atomic` predecessor found it only
  reaching 46.86% achieved occupancy at M=4096 -- more `col_splits` was
  the profiling-motivated fix.
- **TOMOS BF16** (`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`) —
  ports TOMOS FP16's own `atomicAdd`-combine dispatcher structure to
  genuine bf16 compute (`__nv_bfloat162`/`__hfma2`, `fp32_bf16_bf16_only`
  decode), same 4096-row threshold reused unchanged. Unlike the fp16
  family, this does not add a 3-way (wide/tiled/narrow_vec) dispatch --
  no bf16 tiled/narrow_vec kernels exist in this codebase, so it stays
  always-split-K.
- **TOMOS FP32 / FP16** (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform`) —
  switches `wh` staging from a register-mediated load (subject to
  register renaming/scheduling limits regardless of source order) to
  Ampere's `cp.async`: the pending value never occupies a register and
  never blocks the issuing warp until an explicit wait, taking the
  expensive global-load wait off the decode's critical path. The
  "warpuniform" half changes which 128 of 256 threads issue the staging
  copy (first 4 warps, warp-uniform) instead of an even/odd-lane
  alternation, eliminating a real measured intra-warp divergence cost
  during staging (only 24.67 of 32 threads active per warp on average
  under the plain issuer). `fp32_16_half_only` decode itself is
  unchanged, safe on arbitrary data.
- **TOMOS FP32 / FP16 (SIMD decode)** (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar`) —
  byte-for-byte the same `cp.async`/warp-uniform-staging kernel as the
  series above, decoding via `fp32_16_half2_decode_swar` instead: decodes
  two `wh` values per 32-bit register (shared masks/shifts/selects run
  once per pair instead of once per element, output already packed as a
  `__half2`), bit-exact against the scalar decode (verified by an
  exhaustive host check over all 2^32 (lane0, lane1) input pairs). Named
  "(SIMD decode)" here for a more accessible label than the sibling
  family's own "(SWAR decode)" naming -- the underlying technique is
  still SWAR (bit-packed two-lanes-per-register tricks on scalar
  hardware), not dedicated vector instructions.
- **TOMOS FP32 / FP16 (fast)** (`mv_fp32_16_pure_half_no_transpose_flush256_fast_swar`) —
  a dispatcher: below `kMvWideRowThreshold`, calls a dedicated fast
  wide-ILP kernel; at/above it, a narrow_vec kernel that flushes its four
  `__half` accumulators into `float` every 256 columns (bounding
  underflow-heavy fp16 accumulation chains). Decodes via the SWAR pair
  form of the lossy fast decode, `fp32_16_half2_decode_fast_swar`
  (vectorized path) or scalar `fp32_16_half_only_fast` (unaligned/tail
  fallback) -- both saturate the whole underflow band to signed zero
  instead of recovering a genuine fp16 subnormal, ~7 ALU ops per pair
  instead of the exact decode's ~19. Lossy -- for throughput comparison
  only, not accuracy-safe on arbitrary data.
- **TOMOS FP32 / BF16** (`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`) —
  explicitly confirmed IDENTICAL to TOMOS BF16 above on `wh` (same
  `fp32_bf16_bf16_only` decode, same `__nv_bfloat162`/`__hfma2` packing,
  same `atomicAdd` combine, same bigtile/wide4x/best dispatch structure,
  same 4096-row threshold) -- duplicated under its own symbols, by
  request, purely so this comparison slot gets its own independently-
  measured benchmark leg rather than silently reusing TOMOS BF16's own
  numbers. There is genuinely no decode/compute axis left to
  differentiate the two series on for bf16 (unlike fp16, which has a
  real reconstruction axis for "TOMOS FP32 / FP16" and a real fast/safe
  decode axis for "TOMOS FP32 / FP16 (fast)") -- bf16 shares fp32's own
  8-bit exponent, so `fp32_bf16_bf16_only` is already exact across the
  full range, with no fast/safe split or reconstruction trick possible.

#### `build/benchmarks/output/plots/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_square.png` / `build/benchmarks/output/plots/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_tall.png`

##### Data

`build/mv_fp32_16_pure_half_transpose_vs_cublas_ref_output/` —
timestamped `.csv`/`.txt` benchmark logs used to generate these plots,
written by `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh`.

##### Experiment:

The TRANSPOSED (`y = alpha*A^T*x + beta*y`) GENUINE 16-bit-compute GEMV
sweep -- distinct from the FP32 GEMV TRANSPOSE plot above: that one
decodes packed halves back to a real fp32 value before computing; this
one multiply-accumulates directly in `__half`/`__half2`/`__nv_bfloat162`
arithmetic, so its cuBLAS baseline (`Hgemm`/`GemmEx`, `n=1`) is a
different cuBLAS entry point than that plot's own `Sgemv` column -- the
two plots' cuBLAS numbers are not directly comparable to each other even
though the shape axes line up. Unlike the NO_TRANSPOSE plot above, this
benchmark has no `_final`-reduced companion -- every series the `.m`
script currently plots is included in the table below, both the four
core method "families" (native fp16 storage, packed-fp16 pure-`__half`
compute, the wh-only hybrid reconstruction, and bf16) and several of
their own measured design-sweep siblings kept as their own named series
rather than folded together. `square` is the M=K sweep; `tall` is the
tall-and-thin M>>K sweep, K held fixed at 512 and 4096.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh` —
  builds and runs the benchmark; reads/writes `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu`.
- `scripts/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.m` — the
  MATLAB function that regenerates the two `.png`/`.fig` plots from an
  already-saved CSV; no separate `.sh` wrapper exists for this plot
  either -- call the `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` (`n=1`, `CUBLAS_OP_T`) | — | external library (cuBLAS) |
| cuBLAS BF16 | `cublasGemmEx` (`CUDA_R_16BF`, `n=1`, `CUBLAS_OP_T`) | — | external library (cuBLAS) |
| TOMOS FP16 | `mv_fp16_native_storage` | — (native `__half` storage) | `A`, `x`, AND `y` all genuine native `__half`; no decode of any kind |
| TOMOS FP16 (implicit) | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half_only` | packed `wh`-only storage, genuine fp16 compute; safe on arbitrary data; warp-per-output-column/split-M/`atomicAdd`, same structure as the FP32 GEMV TRANSPOSE family |
| TOMOS FP16 (implicit, fast) | `mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half_only_fast` | lossy underflow-to-zero sibling of the series above, same kernel structure otherwise; tested against its own dedicated safe-range data |
| TOMOS FP16 (implicit, fast2, vec4) | `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half_only_fast2` | bit-exact SAFE branchless decode despite the "fast2" name; register-blocking-width sweep's own baseline (N=4) |
| TOMOS FP16 (implicit, fast2, cp.async) | `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split_cpasync` | `fp32_16_half_only_fast2` | same decode as the series above plus double-buffered `cp.async` prefetch of `wh`; measured consistently slower, not what production calls |
| TOMOS FP16 (implicit, fast2, vec2) [production] | `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split` | `fp32_16_half_only_fast2` | N=2 register-blocking width; the actual kernel `mv_pure_half_impl`'s TRANSPOSE branch dispatches to at runtime (~7.6% faster than vec4 at M=1,000,000) |
| TOMOS FP16 (implicit, fast2, vec8) | `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec8_split` | `fp32_16_half_only_fast2` | N=8 register-blocking-width sweep sibling, same decode |
| mv_fp16_native_mixed (explicit, production) | `mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split` | — (native `__half` storage, mixed precision) | `A` genuine `__half`, `x`/`y` kept `float`; the actual kernel this codebase's `--explicit-duplicates` mode dispatches to |
| isolated mv_apply (implicit) | same kernel as "TOMOS FP16 (implicit, fast2, vec2) [production]" | `fp32_16_half_only_fast2` | called through `mv_apply(...)`, the exact dispatch entry point `lsqr.cpp` itself uses, instead of the raw kernel function directly |
| isolated mv_apply (explicit) | same kernel as "mv_fp16_native_mixed (explicit, production)" | — | same `mv_apply(...)` entry-point check for the explicit/mixed-precision path |
| TOMOS FP32 / FP16 | `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired` | `mv_fp32_16_hybrid_process_element` | reconstructs an approximate fp32 value from `wh` ALONE (extended exponent + `mant_hi`, no `wr` input) when `control≠0`, accumulating in genuine `float`; packs adjacent same-control-0 pairs into a real `__hfma2` for the common cheap case |
| TOMOS FP32 / FP16 (implicit) | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast` | `fp32_16_fast_half` | unconditional bit-reinterpret, UNSAFE on arbitrary data -- every `wh` value must already have `control=0` |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4` | `fp32_bf16_bf16_only` | safe on arbitrary data (bf16 shares fp32's own 8-bit exponent, no fast/safe split needed) |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`/`fp32_bf16`/hybrid TRANSPOSE GEMV kernels,
  declarations in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`),
  `src/kernels/cuda/matrix/mv_cuda_kernels.cu` (native-storage
  `mv_fp16_native_storage`/`mv_fp16_native_mixed`, declarations in
  `src/kernels/cuda/matrix/mv_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp`
  (`fp32_16_half_only`/`_fast`/`_fast2`/`fp32_16_fast_half`),
  `src/core/matrix/fp32_bf16_pack.hpp` (`fp32_bf16_bf16_only`).
  `mv_fp32_16_hybrid_process_element` (the wh-only hybrid decode) is
  defined directly in `fp32_16_cuda_kernels.cu`, not in a core pack
  header.
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu`.

**Method Description:**
- **cuBLAS FP16 / BF16** (`cublasHgemm` / `cublasGemmEx`, `CUBLAS_OP_T`) —
  NVIDIA's closed-source half/bfloat16 GEMM run as a transposed GEMV via
  the standard "N=1" workaround; external reference baselines, not part
  of this codebase.
- **TOMOS FP16** (`mv_fp16_native_storage`) — `A` is plain native `__half`
  storage at rest (not the packed `fp32_16` `wh`/`wr` format the rest of
  this table decodes from), so every element read is already the real
  fp16 value; no decode step of any kind. `x`/`y` also stay `__half`
  (staged into shared memory and rounded once per tile), matching
  cuBLAS Hgemm's own all-`__half` convention.
- **TOMOS FP16 (implicit)** (`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split`) —
  the packed-storage (`fp32_16` `wh`-only) counterpart of TOMOS FP16
  above: warp-per-output-column, M split across `blockIdx.x`,
  `atomicAdd` combining partial sums, decoding via the full
  `fp32_16_half_only` (safe on arbitrary data). "(implicit)" here refers
  to this codebase's own convention for a packed format used without an
  explicit dtype request, distinct from the unrelated "(implicit)"-style
  naming used elsewhere for a fast-decode axis.
- **TOMOS FP16 (implicit, fast)** (`mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split`) —
  byte-for-byte the same kernel as the series above, decoding via
  `fp32_16_half_only_fast` instead: saturates any underflowed value to
  signed zero rather than recovering a genuine fp16 subnormal. Tested
  against its own dedicated safe-range data (magnitude in [0.5, 1.0]),
  not the plain data TOMOS FP16 (implicit) uses.
- **TOMOS FP16 (implicit, fast2, vec4) / (fast2, cp.async) / (fast2,
  vec2) [production] / (fast2, vec8)** — four siblings sharing the same
  `fp32_16_half_only_fast2` decode (BIT-EXACT and safe on arbitrary
  data, despite the "fast2" name -- a branchless reformulation, not a
  lossy shortcut): "vec4" is the register-blocking-width sweep's
  baseline (N=4 per-lane vectorization); "cp.async" adds double-buffered
  prefetch of `wh` on top of vec4 but measured consistently slower;
  "vec2" (N=2) and "vec8" (N=8) are the width-sweep's other two points,
  added after `ncu` profiling found the vec4 kernel 86%
  Compute-SM-bound with ~50% of stall cycles on L1TEX scoreboard
  dependencies. vec2 measured the best overall (~7.6% faster than vec4
  at M=1,000,000) and is what `mv_pure_half_impl`'s TRANSPOSE branch
  (`fp32_16_registries.cpp`) actually dispatches to at runtime, hence its
  "[production]" tag.
- **mv_fp16_native_mixed (explicit, production)** (`mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split`) —
  a true mechanical duplicate of TOMOS FP16's own dispatch tree (same
  `__hfma2` packing, same `col_splits` formulas/crossover, same
  reduce-vs-atomic split) but with `x`/`y` kept `float` instead of
  `__half`, matching `mv_fp16_native`'s existing mixed-precision
  contract. This is the actual kernel this codebase's
  `--explicit-duplicates` mode dispatches to for its own TRANSPOSE
  matvec (`dense_registries.cpp`'s `mv_fp16_native_impl`).
- **isolated mv_apply (implicit) / (explicit)** — not new kernels: the
  same two kernels as "TOMOS FP16 (implicit, fast2, vec2) [production]"
  and "mv_fp16_native_mixed" respectively, but invoked through
  `mv_apply(Device, TransposeType, Dense&, ...)` -- the exact entry point
  `lsqr.cpp` itself calls -- instead of calling the raw kernel functions
  directly. Added because an external harness's own printed TRANSPOSE
  numbers for these kernels didn't cleanly reconcile with this
  benchmark's own raw-call numbers; plotted side by side as a direct
  consistency check between the two call paths.
- **TOMOS FP32 / FP16** (`mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`) —
  unlike a plain `fp32_16_half_only` decode (which flushes any value
  needing more than fp16's 5-bit exponent to signed Infinity/zero), this
  kernel reconstructs an APPROXIMATE fp32 value from `wh` alone (the
  extended exponent bits plus `mant_hi`, no `wr` input at all) via
  `mv_fp32_16_hybrid_process_element`, accumulating that reconstruction
  in genuine `float` -- while packing adjacent same-control-0 pairs into
  one real `__hfma2` for the common cheap case, matching a plain vec4
  kernel's own arithmetic throughput there.
- **TOMOS FP32 / FP16 (implicit)** (`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast`) —
  decodes via `fp32_16_fast_half`, an unconditional bit-reinterpret with
  no control-bit check at all -- UNSAFE on arbitrary data, valid only
  when the caller guarantees every `wh` value already has `control=0`
  (every source magnitude inside fp16's own normal representable
  range). Tested against its own dedicated safe-range data.
- **TOMOS FP32 / BF16** (`mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4`) —
  decodes via `fp32_bf16_bf16_only`, safe on arbitrary data unlike every
  fp16 "(implicit)"/"(fast)" sibling above: bf16 shares fp32's own 8-bit
  exponent field, so a truncated `wh` is already a valid bf16 bit
  pattern for any input, unconditionally.

## Plots - Error for Increasing Operand-Dimensions

Unlike the Throughput plots above (GFlop/s), these measure how far each
method's result drifts from a reference value -- Frobenius relative
error for GEMM, L2 relative error for GEMV (textually different names
for the same quantity on a flattened array). Every plot documented here
is one starting point picked out of a larger family of scale-tiered
sweeps each underlying benchmark can run (see each Experiment's own
`scale_tag`/`--normalized` note) -- not the full set.

### FP32 GEMM

#### `build/output/error_dimension_scaling/gemm/fp32_compute/plots/plot_gemm_fp32_c32_vs_dgemm_ref.png`

##### Data

`build/output/error_dimension_scaling/gemm/fp32_compute/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot,
written by `scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh`.

##### Experiment:

The fp32-compute-tier GEMM square sweep (M=K=N), measuring Frobenius
relative error against a genuine FP64 (`cublasDgemm`, on A/B widened to
double) reference -- NOT an FP32 reference, unlike this codebase's other
`*_recon_l2_error_vs_cublas_ref` benchmarks. Measured against FP32
instead, "TOMOS FP32"'s own error would stay near machine epsilon (both
share the same float rounding model), never showing the real accumulated
rounding error a large float GEMM reduction produces; measured against
FP64, that error becomes a genuine, non-trivial baseline the FP16/BF16-
encoded series' own additional error can be compared against. Every
series here accumulates in float (no fp16/bf16-accumulator-overflow
concern). Both axes are log-scale -- "TOMOS FP32"'s own error can sit
several decades below the FP16/BF16-encoded series' own error. Same
kernel set as the FP32 GEMM Throughput section above, plus cuBLAS FP32
(`cublasSgemm`) as an additional own-error baseline against the same FP64
reference.

**Scripts:**
- `scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh` —
  builds and runs the benchmark; reads/writes `benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu`.
- `scripts/plot_gemm_fp32_c32_vs_dgemm_ref.m` — the MATLAB function that
  regenerates the plot from an already-saved CSV; no separate `.sh`
  wrapper exists for this plot -- call the `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP32 | `cublasSgemm` | — | external library (cuBLAS); own error against the FP64 reference, on the SAME A/B every other series shares |
| TOMOS FP32 | `gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | — (native, unpacked storage) | same kernel as the FP32 GEMM Throughput section; A consumed directly, no 16-bit encoding |
| TOMOS FP32 / FP16 | `mm_fp32_16_tiled_cpasync_dualbuf` | `fp32_16_combine` | A packed via `fp32_16_split` (plain, round-to-nearest); same kernel as the FP32 GEMM Throughput section |
| TOMOS FP32 / FP16 (truncated) | `mm_fp32_16_tiled_cpasync_dualbuf_truncated` | `fp32_16_combine_truncated` | A packed via `fp32_16_split_truncated`; same kernel as the FP32 GEMM Throughput section |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_gemm_materialize_midpoint_exact` | `fp32_16_combine` | same kernel as the FP32 GEMM Throughput section |
| TOMOS FP32 / BF16 (truncated) | `mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact` | `fp32_bf16_combine` | A packed via `fp32_bf16_split` (plain, truncating); same kernel as the FP32 GEMM Throughput section |
| TOMOS FP32 / BF16 (workspace) | `mm_fp32_bf16_gemm_materialize_rounded` | `fp32_bf16_combine_rounded` | A packed via `fp32_bf16_split_rounded`; same kernel as the FP32 GEMM Throughput section |

**Files:**
- Kernel implementations: same as the FP32 GEMM Throughput section above
  (`src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`, declarations in
  `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu`.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP32
GEMM Throughput section above -- see that section's own Method
Description for the full architectural derivation of each (`syncpipe4`/
`bk8`/`warptile`/`nostride`/`snake`/`midpoint_exact`, the `cp.async`
`dualbuf` decode-fused pipeline, and the materialize-then-call-native
workspace kernels). What differs here is only the metric (Frobenius
relative error against a true FP64 reference, not GFlop/s) and the
addition of **cuBLAS FP32** (`cublasSgemm`) as its own measured baseline
against that same FP64 reference, run on the identical A/B every TOMOS
series shares -- checking whether TOMOS FP32's own rounding error is
comparable to, better, or worse than cuBLAS's, rather than assuming
parity.

### FP16 GEMM

#### `build/output/error_dimension_scaling/gemm/fp16_compute/plots/plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref_scale16384_normalized.png`

##### Data

`build/output/error_dimension_scaling/gemm/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot, written by
`scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh --normalized --scales 16384`.

##### Experiment:

The pure/composite-tier (tensor-core) GEMM square sweep (M=K=N), the
accuracy-vs-N companion to the FP16 GEMM Throughput plot above --
Frobenius relative error against a genuine FP32 (`cublasSgemm`) reference
(cuBLAS/CUTLASS are NOT the reference here, unlike other tiers -- they
are four additional own-error baselines, each run on the same
native-rounded A/B as its own group's native TOMOS series). Every series
genuinely stores at least A (and, for the two native kernels, both A and
B) in fp16/bf16 precision by design (this tier IS the tensor-core compute
tier), so unlike the FP32 GEMM accuracy plot above, there is no
near-machine-epsilon "TOMOS FP32" sanity-check series possible here.
`normal_scale` widens A's "normal"-range test data from `[-1,1]` to
`[-scale,scale]`; three of these five series narrow their float32
tensor-core accumulator down into `__half`-typed `C` at the epilogue, so
`C[i,j]` (a sum of `K` products) has a `K`-dependent OUTPUT-STORAGE
overflow ceiling distinct from any internal-accumulator concern.
`--normalized` scales `B` by `1/sqrt(K)` at data-generation time instead
of leaving it at plain `[-1,1]`, which keeps that row-sum magnitude
~`normal_scale` regardless of `K` rather than growing like
`normal_scale*sqrt(K)` -- the fix that makes the wide `scale=16384` tier
plotted here safe to run at all (an un-normalized run is capped at
`scale=64` for the same reason; see the `.sh`'s own header for the
overflow derivation and the real bug it caught). Both axes are log-scale.

**Scripts:**
- `scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh --normalized --scales 16384` —
  builds and runs the benchmark at this specific scale/normalization
  tier; reads/writes `benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.m` — the
  MATLAB function that regenerates the plot from an already-saved CSV
  (`scale_tag` argument `'16384_normalized'` produces this exact
  filename); no separate `.sh` wrapper exists for this plot -- call the
  `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` | — | external library (cuBLAS); own error, not the reference (`cublasSgemm` is) |
| CUTLASS FP16 | `cutlass::gemm::device::Gemm<...>` (fp16 tensor-op) | — | external library (CUTLASS); own error, same config as the FP16 GEMM Throughput section |
| cuBLAS BF16 | `cublasGemmEx` (`CUDA_R_16BF`, `CUBLAS_COMPUTE_32F`) | — | external library (cuBLAS); own error |
| CUTLASS BF16 | `cutlass::gemm::device::Gemm<...>` (bf16 tensor-op) | — | external library (CUTLASS); own error, same tensorop config as CUTLASS FP16 |
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | same kernel as the FP16 GEMM Throughput section |
| TOMOS BF16 | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | same kernel as the FP16 GEMM Throughput section |
| TOMOS FP32 / FP16 | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` | same kernel as the FP16 GEMM Throughput section |
| TOMOS FP32 / FP16 (workspace) | `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` | `fp32_16_half_only_branchless_no_overflow_nomask` | same kernel as the FP16 GEMM Throughput section |
| TOMOS FP32 / BF16 | `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` | none — `wh` reinterpreted directly as `__nv_bfloat16*` | same kernel as the FP16 GEMM Throughput section |

**Files:**
- Kernel implementations: same as the FP16 GEMM Throughput section above
  (`src/kernels/cuda/matrix/mm_cuda_kernels.cu`, declarations in
  `src/kernels/cuda/matrix/mm_cuda_kernels.cuh`).
- Decode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu`.

**Method Description:**

Every TOMOS/CUTLASS/cuBLAS kernel here is byte-for-byte the same one
documented in the FP16 GEMM Throughput section above -- see that
section's own Method Description for the full architectural derivation
of each (`wide_bigreg`/`deep4`/`iter`/`epivec`/`acongruous`/`fastk`/
`precomp2`/`serp`, the `inline_dirty` batched clean/dirty decode-skip
check, and the direct-bf16-reinterpret path). What differs here is only
the metric (Frobenius relative error against a true FP32 reference at a
specific `normal_scale`/`--normalized` data-generation tier, not
GFlop/s) -- and that cuBLAS/CUTLASS appear as four ADDITIONAL own-error
series here rather than as the reference itself (`cublasSgemm` is the
reference for every series, TOMOS and vendor alike).

### FP32 GEMV

#### `build/output/error_dimension_scaling/gemv/fp32_compute/plots/plot_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref_scale16384.png`

##### Data

`build/output/error_dimension_scaling/gemv/fp32_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot, written by
`scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales 16384`.

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x`) fp32-reconstruction-tier GEMV square
sweep (M=K, capped at 24576 -- each series here needs four concurrently-
alive M*K-sized buffers at peak, unlike the FP32-reference sibling
benchmark's own two), measuring L2 relative error against a TRUE FP64
(`cublasDgemv`) reference instead of FP32 (`cublasSgemv`) -- same "expose
the real accumulated rounding error, not a same-precision no-op" fix the
FP32 GEMM accuracy plot above applies, at this benchmark's own
`normal_scale=16384` data-generation tier. Y-axis is log-scale. Plus
cuBLAS FP32 (dashed) as its own measured error against that same FP64
reference, on the SAME `A_native`/`x` the TOMOS FP32 series shares.

**Scripts:**
- `scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales 16384` —
  builds and runs the benchmark at this specific scale tier; reads/writes
  `benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu`.
- `scripts/plot_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.m` — the
  MATLAB function that regenerates the plot from an already-saved CSV
  (`scale_tag` argument `'16384'` produces this exact filename); no
  separate `.sh` wrapper exists for this plot -- call the `.m` function
  directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP32 | `cublasSgemv` | — | external library (cuBLAS); own error against the FP64 reference, on the SAME `A_native`/`x` TOMOS FP32 shares |
| TOMOS FP32 | `mv_fp32_tiled` | — (native, unpacked storage) | same kernel as the FP32 GEMV Throughput section |
| TOMOS FP32 / FP16 | `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock` | `fp32_16_combine_branchless` | same kernel as the FP32 GEMV Throughput section |
| TOMOS FP32 / FP16 (fast) | `mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal` | `fp32_16_combine_fast_normal` | NOT the same kernel as the Throughput section's "(async)" series -- this is the plain scalar predecessor "(async)" was promoted over, kept here as its own accuracy series; UNSAFE on arbitrary data (only the cheap "normal" decode path) |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded` | `fp32_bf16_combine_rounded` | same kernel as the FP32 GEMV Throughput section |
| TOMOS FP32 / BF16 (workspace) | `mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace` | `fp32_bf16_combine` | same kernel as the FP32 GEMV Throughput section's "(truncated)" series -- labeled "(workspace)" here instead |

**Files:**
- Kernel implementations: same as the FP32 GEMV Throughput section above
  (`src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu` for the packed
  formats, `src/kernels/cuda/matrix/mv_cuda_kernels.cu` for native fp32).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu`.

**Method Description:**
- **cuBLAS FP32** (`cublasSgemv`) — own error measured against the same
  FP64 reference every TOMOS series here is measured against, on the
  identical `A_native`/`x`, so TOMOS FP32's own error can be judged
  against an industry-standard baseline, not just internally
  self-consistent.
- **TOMOS FP32 / TOMOS FP32/FP16 / TOMOS FP32/BF16 / TOMOS FP32/BF16
  (workspace)** — byte-for-byte the same kernels documented in the FP32
  GEMV Throughput section above; see that section's own Method
  Description for the full architectural derivation. Only the metric
  changes here (L2 relative error against true FP64, not GFlop/s).
- **TOMOS FP32 / FP16 (fast)** (`mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`) —
  the plain scalar (no row-vectorization) predecessor "TOMOS FP32 / FP16
  (async)" was promoted over in the Throughput section, kept here as its
  own dedicated accuracy series rather than reused: same 4-rows/lane
  `atomicAdd`/scaled-`K_BLOCK` family structure as "TOMOS FP32 / FP16"
  above, decoding via `fp32_16_combine_fast_normal` instead of the full
  `fp32_16_combine_branchless` -- computes ONLY the cheap "normal"
  (control=0) reconstruction path, UNSAFE on arbitrary data (the caller
  must independently guarantee no extended/degraded-encoded values
  exist), exposing exactly the additional error that shortcut costs
  relative to the full decode.

#### `build/output/error_dimension_scaling/gemv/fp32_compute_transpose/plots/plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref_scale16384.png`

##### Data

`build/output/error_dimension_scaling/gemv/fp32_compute_transpose/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot,
written by `scripts/benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales 16384`.

##### Experiment:

The TRANSPOSED (`y = alpha*A^T*x`) fp32-reconstruction-tier GEMV square
sweep (M=K, capped at 24576, same memory reasoning as the NO_TRANSPOSE
plot above), measuring L2 relative error against a TRUE FP64
(`cublasDgemv(CUBLAS_OP_T)`) reference instead of FP32
(`cublasSgemv(CUBLAS_OP_T)`) -- the TRANSPOSE counterpart of the FP32
GEMV NO_TRANSPOSE error plot above, identical reasoning for why FP64
matters here (only the direction differs), at this benchmark's own
`normal_scale=16384` data-generation tier. Y-axis is log-scale. Plus
cuBLAS FP32 (dashed) as its own measured error against that same FP64
reference, on the SAME `A_native`/`x` the TOMOS FP32 series shares.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales 16384` —
  builds and runs the benchmark at this specific scale tier; reads/writes
  `benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu`.
- `scripts/plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved CSV
  (`scale_tag` argument `'16384'` produces this exact filename); no
  separate `.sh` wrapper exists for this plot -- call the `.m` function
  directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP32 | `cublasSgemv` (`CUBLAS_OP_T`) | — | external library (cuBLAS); own error against the FP64 reference, on the SAME `A_native`/`x` TOMOS FP32 shares |
| TOMOS FP32 | `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec` | — (native, unpacked storage) | same kernel as the FP32 GEMV Throughput section's TRANSPOSE plot |
| TOMOS FP32 / FP16 | `mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch` | `fp32_16_combine_branchless_wrapborrow` | same kernel as the FP32 GEMV Throughput section's TRANSPOSE plot |
| TOMOS FP32 / FP16 (fast) | `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch` | `fp32_16_combine_fast_normal_wrapborrow` | same kernel as "TOMOS FP32 / FP16 (implicit)" in the FP32 GEMV Throughput section's TRANSPOSE plot -- labeled "(fast)" here instead of "(implicit)" |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch` | `fp32_bf16_combine_rounded` | same kernel as the FP32 GEMV Throughput section's TRANSPOSE plot |
| TOMOS FP32 / BF16 (truncated) | `mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4` | `fp32_bf16_combine` | same kernel as the FP32 GEMV Throughput section's TRANSPOSE plot |

**Files:**
- Kernel implementations: same as the FP32 GEMV Throughput section's
  TRANSPOSE plot above (`src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  for the packed formats, `src/kernels/cuda/matrix/mv_cuda_kernels.cu`
  for native fp32).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family; `fp32_16_combine_branchless_wrapborrow`/`_fast_normal_wrapborrow`
  live here too, see the Throughput section's own Files note),
  `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu`.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP32
GEMV Throughput section's TRANSPOSE plot above -- see that section's own
Method Description for the full architectural derivation (the
warp-per-output-column/split-M/`atomicAdd` structure, the wraparound-
subtraction `wrapborrow` decode reformulation, and the vec4/`stagevec`
vectorization history). Only the metric changes here (L2 relative error
against true FP64, not GFlop/s), plus **cuBLAS FP32** (`cublasSgemv`,
`CUBLAS_OP_T`) appears as its own measured baseline against that same
FP64 reference -- the TRANSPOSE counterpart of the same addition in the
FP32 GEMV NO_TRANSPOSE error plot above. The only naming difference from
the Throughput section: "TOMOS FP32 / FP16 (fast)" here is the identical
kernel the Throughput section calls "TOMOS FP32 / FP16 (implicit)" --
this error benchmark's own script predates that rename.

### FP16 GEMV

#### `build/output/error_dimension_scaling/gemv/fp16_compute/plots/plot_gemv_fp32_16_l2_error_vs_cublas_ref_scale16384_normalized.png`

##### Data

`build/output/error_dimension_scaling/gemv/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot, written by
`scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh --normalized --scales 16384`.

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x`) GENUINE 16-bit-compute GEMV square
sweep, the accuracy-vs-M companion to the FP16 GEMV Throughput plot
above -- L2 relative error against a genuine FP32-precision reference
(`cublasSgemv` on an uncompressed float32 copy of that method's own
data), not throughput. `cublasSgemv` itself (real fp32) remains this
file's own reference throughout; cuBLAS FP16/BF16 below are cuBLAS's OWN
16-bit compute paths being measured against that reference, not the
reference itself. `normal_scale` widens the input data's magnitude range
(see the `.cu`'s own header); `--normalized` scales `x` to keep the
accumulated magnitude from overflowing at wide scales, the fix that makes
this benchmark's own wider `scale=16384` tier (plotted here) safe to run.
Y-axis is log-scale.

**Scripts:**
- `scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh --normalized --scales 16384` —
  builds and runs the benchmark at this specific scale/normalization
  tier; reads/writes `benchmarks/kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemv_fp32_16_l2_error_vs_cublas_ref.m` — the MATLAB
  function that regenerates the plot from an already-saved CSV
  (`scale_tag` argument `'16384_normalized'` produces this exact
  filename); no separate `.sh` wrapper exists for this plot -- call the
  `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` (`n=1`) | — | external library (cuBLAS); own error, not the reference (`cublasSgemv` is); GEMV-via-GEMM workaround |
| cuBLAS BF16 | `cublasGemmEx` (`CUDA_R_16BF`, `n=1`) | — | external library (cuBLAS); own error, same GEMV-via-GEMM workaround |
| TOMOS FP16 (implicit) | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best` | `fp32_16_half_only` | the packed `fp32_16` `wh`-only kernel -- same kernel as "TOMOS FP16" in the FP16 GEMV Throughput section |
| TOMOS FP32 / FP16 | `mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best` | `mv_fp32_16_hybrid_process_element`-style branchless wh-only reconstruction | NOT the same kernel as the Throughput section's `cpasync_warpuniform` series -- this is the hybrid family's own NO_TRANSPOSE dispatcher (see the TRANSPOSE hybrid kernel documented in the FP32 GEMV Throughput section for the shared reconstruction technique) |
| TOMOS FP32 / FP16 (fast) | `mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best` | `fp32_16_fast_half` | NOT the same kernel as the Throughput section's SWAR `flush256_fast_swar` series -- this is the plain (non-SWAR) fasthalf dispatcher; UNSAFE on arbitrary data |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2` | `fp32_bf16_bf16_only` | same kernel as the FP16 GEMV Throughput section |
| TOMOS BF16 | `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best` | `fp32_bf16_bf16_only` | same kernel as the FP16 GEMV Throughput section |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (declarations in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp` (fp16
  family), `src/core/matrix/fp32_bf16_pack.hpp` (bf16 family).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu`.

**Method Description:**
- **cuBLAS FP16 / BF16** (`cublasHgemm` / `cublasGemmEx`, both `n=1`) —
  own error against the same FP32 reference every TOMOS series here is
  measured against, not the reference itself; same GEMV-via-GEMM
  workaround as the FP16 GEMV Throughput section (neither dtype has a
  native `cublasXgemv` entry point).
- **TOMOS FP16 (implicit) / TOMOS FP32/BF16 / TOMOS BF16** — byte-for-byte
  the same kernels documented in the FP16 GEMV Throughput section above
  (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`,
  `mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`); see that
  section's own Method Description for the full architectural
  derivation. Only the metric changes here (L2 relative error against
  true FP32, not GFlop/s).
- **TOMOS FP32 / FP16** (`mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best`) —
  the NO_TRANSPOSE dispatcher of the same hybrid kernel family as "TOMOS
  FP32 / FP16" in the FP32 GEMV Throughput section's TRANSPOSE plot
  (`mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`):
  per element, control=0 takes the cheap `__hfma2`-packed `__half` path;
  control=1 (extended or degraded) reconstructs an APPROXIMATE fp32 value
  from `wh` alone (no `wr` input), accumulating in genuine `float`. This
  is a genuinely different kernel from the FP16 GEMV Throughput section's
  own NO_TRANSPOSE "TOMOS FP32 / FP16" series
  (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform`,
  which decodes via `fp32_16_half_only` and is `cp.async`-staged) --
  same label, different underlying family, since this accuracy benchmark
  and the throughput benchmark were built to compare different points in
  this codebase's own kernel history.
- **TOMOS FP32 / FP16 (fast)** (`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best`) —
  a dispatcher (same bigtile/wide4x split as TOMOS FP16 above) over
  kernels decoding via `fp32_16_fast_half` (unconditional bit-reinterpret,
  UNSAFE on arbitrary data -- every `wh` value must already have
  `control=0`). Also NOT the same kernel as the FP16 GEMV Throughput
  section's own NO_TRANSPOSE "TOMOS FP32 / FP16 (fast)" series
  (`mv_fp32_16_pure_half_no_transpose_flush256_fast_swar`, which uses the
  SWAR-paired `fp32_16_half2_decode_fast_swar` decode and a flush-every-
  256-columns accumulation scheme) -- this accuracy series measures the
  plain, non-SWAR fasthalf family's own error instead.

#### `build/output/error_dimension_scaling/gemv/fp16_compute_transpose/plots/plot_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref_scale16384_normalized.png`

##### Data

`build/output/error_dimension_scaling/gemv/fp16_compute_transpose/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot,
written by `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh --normalized --scales 16384`.

##### Experiment:

The TRANSPOSE (`y = alpha*A^T*x`) GENUINE 16-bit-compute GEMV square
sweep, the accuracy-vs-M companion to the FP16 GEMV Throughput section's
TRANSPOSE plot above -- L2 relative error against a genuine
FP32-precision reference (`cublasSgemv(CUBLAS_OP_T)` on an uncompressed
float32 copy of that method's own data), not throughput. Only FOUR TOMOS
series here (not five, unlike this tier's NO_TRANSPOSE sibling above) --
this TRANSPOSE tier has no separate plain "TOMOS BF16" series, only
"TOMOS FP32/BF16". `cublasSgemv(CUBLAS_OP_T)` itself (real fp32) remains
this file's own reference throughout; cuBLAS FP16/BF16 below are
cuBLAS's OWN 16-bit compute paths being measured against that reference,
at this benchmark's own `--normalized --scales 16384` data-generation
tier (same overflow-safety reasoning as the NO_TRANSPOSE FP16-tier error
plot above). Y-axis is log-scale.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh --normalized --scales 16384` —
  builds and runs the benchmark at this specific scale/normalization
  tier; reads/writes `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved CSV
  (`scale_tag` argument `'16384_normalized'` produces this exact
  filename); no separate `.sh` wrapper exists for this plot -- call the
  `.m` function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| cuBLAS FP16 | `cublasHgemm` (`n=1`, `CUBLAS_OP_T`) | — | external library (cuBLAS); own error, not the reference (`cublasSgemv(CUBLAS_OP_T)` is); GEMV-via-GEMM workaround |
| cuBLAS BF16 | `cublasGemmEx` (`CUDA_R_16BF`, `n=1`, `CUBLAS_OP_T`) | — | external library (cuBLAS); own error, same GEMV-via-GEMM workaround |
| TOMOS FP16 | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half_only` | same kernel as "TOMOS FP16 (implicit)" in the FP16 GEMV Throughput section's TRANSPOSE plot -- labeled "TOMOS FP16" here instead |
| TOMOS FP16 (fast) | `mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half_only_fast` | same kernel as "TOMOS FP16 (implicit, fast)" in the FP16 GEMV Throughput section's TRANSPOSE plot -- labeled "TOMOS FP16 (fast)" here instead; safe on arbitrary data (lossy underflow-to-zero only) |
| TOMOS FP32 / FP16 | `mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired` | `mv_fp32_16_hybrid_process_element` | same kernel as the FP16 GEMV Throughput section's TRANSPOSE plot |
| TOMOS FP32 / FP16 (implicit) | `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast` | `fp32_16_fast_half` | same kernel as the FP16 GEMV Throughput section's TRANSPOSE plot; UNSAFE on arbitrary data |
| TOMOS FP32 / BF16 | `mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4` | `fp32_bf16_bf16_only` | same kernel as the FP16 GEMV Throughput section's TRANSPOSE plot |

**Files:**
- Kernel implementations: `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`/`fp32_bf16`/hybrid TRANSPOSE GEMV kernels,
  declarations in `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cuh`).
- Decode/encode helper functions: `src/core/matrix/fp32_16_pack.hpp`
  (`fp32_16_half_only`/`_fast`/`fp32_16_fast_half`),
  `src/core/matrix/fp32_bf16_pack.hpp` (`fp32_bf16_bf16_only`).
  `mv_fp32_16_hybrid_process_element` is defined directly in
  `fp32_16_cuda_kernels.cu`, not in a core pack header (see the FP16
  GEMV Throughput section's own Files note).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu`.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP16
GEMV Throughput section's TRANSPOSE plot above -- see that section's own
Method Description for the full architectural derivation (the
warp-per-output-column/split-M/`atomicAdd` structure shared by every
packed-storage series, the `mv_fp32_16_hybrid_process_element` wh-only
reconstruction, and each decode function's own safety contract). Only
the metric changes here (L2 relative error against true FP32 at the
`--normalized --scales 16384` tier, not GFlop/s), plus **cuBLAS FP16 /
BF16** appear as their own measured baselines against that same FP32
reference rather than the reference itself. Naming differences from the
Throughput section: "TOMOS FP16" here is that section's "TOMOS FP16
(implicit)"; "TOMOS FP16 (fast)" here is that section's "TOMOS FP16
(implicit, fast)" -- this error benchmark's own script uses the shorter,
pre-rename labels.

## Plots - Error for Increasing Magnitude Variance

Unlike the "Increasing Operand-Dimensions" plots above (X-axis is matrix
dimension at one fixed `normal_scale`), the plots here hold the shape
FIXED and sweep `normal_scale` on the X-axis instead -- the complementary
axis. `fp32_16_pack.hpp`'s/`fp32_bf16_pack.hpp`'s own encoding has
NORMAL/EXTENDED/degraded branches, and only large enough `|A[i,j]|`
values ever cross out of "normal"; a fixed-scale, N-swept plot can only
show rounding-precision loss WITHIN the normal branch, while these
plots instead push more of the operands' own elements across that
branch boundary as `normal_scale` grows.

### FP32 GEMM

#### `build/output/error_magnitude_scaling/gemm/fp32_compute/plots/plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemm/fp32_compute/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot
(same output directory/binary as the FP32 GEMM plot in the "Increasing
Operand-Dimensions" section above, run a different way), written by
`scripts/benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh`.

##### Experiment:

The fp32-compute-tier GEMM Frobenius-relative-error sweep at a SINGLE
FIXED SHAPE (M=K=N=24576, not the "Increasing Operand-Dimensions"
section's own N-sweep-at-fixed-scale=1.0), swept across `normal_scale`
instead -- same seven series, all measuring float-accumulating kernels
against a TRUE FP64 (`cublasDgemm`) reference, same reasoning for why
FP64 (not FP32) is used as the reference. Data was generated with
`--normalized` (B scaled by `1/sqrt(K)`) -- kept for API consistency with
every other error benchmark in this codebase, though unlike the FP16 GEMM
plot below, normalization isn't working around an output-storage-overflow
bug here: `C` is genuine float throughout, so this plot's raw
(unnormalized) sibling uses the identical `SCALES` range, not a reduced
one. Both axes are log-scale.

**Scripts:**
- `scripts/benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh` —
  builds and runs the benchmark (fixed `M=K=N=24576`, `SCALES=(1 4 16 64
  256 1024 4096 16384)`); reads/writes the SAME benchmark driver as the
  FP32 GEMM plot in the "Increasing Operand-Dimensions" section above,
  `benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu`.
- `scripts/plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref.m` — the MATLAB
  function that regenerates the plot from an already-saved CSV; no
  separate `.sh` wrapper exists for this plot -- call the `.m` function
  directly.

**Kernels called (one per plotted series):**

Identical kernel set to the FP32 GEMM plot in the "Plots - Error for
Increasing Operand-Dimensions" section above (`cublasSgemm`,
`gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`,
`mm_fp32_16_tiled_cpasync_dualbuf` and its `_truncated` sibling,
`mm_fp32_16_gemm_materialize_midpoint_exact`,
`mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact`, and
`mm_fp32_bf16_gemm_materialize_rounded`) -- see that section's own table
for the full Series/Kernel/Decode-function/Notes breakdown, not repeated
here.

**Files:**

Same files as the FP32 GEMM plot in the "Plots - Error for Increasing
Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP32
GEMM plot of the "Plots - Error for Increasing Operand-Dimensions"
section above -- see that section's own Method Description for the full
architectural derivation. Only the X-axis changes here: shape is held
fixed at `M=K=N=24576` and `normal_scale` is swept instead of `N`.

### FP16 GEMM

#### `build/output/error_magnitude_scaling/gemm/fp16_compute/plots/plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemm/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot (same output
directory/binary as the FP16 GEMM plot in the "Increasing
Operand-Dimensions" section above, run a different way), written by
`scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh`.

##### Experiment:

The pure/composite-tier (tensor-core) GEMM Frobenius-relative-error
sweep at a SINGLE FIXED SHAPE, swept across `normal_scale` instead of
`N` -- same nine series as the "Increasing Operand-Dimensions" section's
own FP16 GEMM plot. **Discrepancy flagged, not resolved:** the `.m`
script's own header comment claims this fixed shape is `M=K=N=65536`,
but the driving `.sh` script's own `FIXED_N` is actually **32768** (verified
directly in `scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh`)
-- the plot's title/legend inherit whatever shape the CSV it was run
against actually used, so the true fixed shape is whatever `FIXED_N` was
at run time, not necessarily what the `.m` comment says. Always run with
`--normalized`: without it, this tier's own three fp16-group series
(which narrow their float32 tensor-core accumulator down into
`__half`-typed `C` at the epilogue) overflow that epilogue at
`scale=4096` and up; `--normalized` removes that K-dependent ceiling,
making the full `SCALES` range safe -- same fix the FP16 GEMM plot in
the "Plots - Error for Increasing Operand-Dimensions" section's own
`_scale16384_normalized` variant applies. Both axes are log-scale.

**Scripts:**
- `scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh` —
  builds and runs the benchmark (fixed shape, `--normalized` always on,
  `SCALES=(1 4 16 64 256 1024 4096 16384)`); reads/writes the SAME
  benchmark driver as the FP16 GEMM plot in the "Increasing
  Operand-Dimensions" section above, `benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved
  CSV; no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly. Unlike its N-sweep sibling, this function takes no
  `scale_tag` argument -- always writes the same, untagged filename.

**Kernels called (one per plotted series):**

Identical kernel set to the FP16 GEMM plot in the "Plots - Error for
Increasing Operand-Dimensions" section above (`cublasHgemm`,
`cutlass::gemm::device::Gemm<...>` fp16/bf16, `cublasGemmEx`,
`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`,
`gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`,
`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp`,
`mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp`,
`mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp`) -- see that
section's own table for the full Series/Kernel/Decode-function/Notes
breakdown, not repeated here.

**Files:**

Same files as the FP16 GEMM plot in the "Plots - Error for Increasing
Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP16
GEMM plot of the "Plots - Error for Increasing Operand-Dimensions"
section above -- see that section's own Method Description for the full
architectural derivation. Only the X-axis changes here: shape is held
fixed (`FIXED_N=32768` per the driving `.sh` script -- see the
Experiment note above on the discrepancy with the `.m` file's own "65536"
comment) and `normal_scale` is swept instead of `N`.

### FP32 GEMV

#### `build/output/error_magnitude_scaling/gemv/fp32_compute/plots/plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemv/fp32_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot (same output
directory/binary as the FP32 GEMV NO_TRANSPOSE plot in the "Increasing
Operand-Dimensions" section above, run a different way), written by
`scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh`.

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x`) fp32-reconstruction-tier GEMV
L2-relative-error sweep at a SINGLE FIXED SHAPE (M=K=24576, smaller than
the FP32-reference sibling benchmark's own M=K=65536 since each series
here needs four concurrently-alive M*K-sized buffers at peak instead of
that sibling's own two), swept across `normal_scale` instead of across
M -- the complementary axis to the "Increasing Operand-Dimensions"
section's own M-sweep-at-one-scale-per-run. Same six series (five TOMOS
plus cuBLAS FP32), measured against a TRUE FP64 (`cublasDgemv`)
reference, same reasoning as that section's own FP32 GEMV plot for why
FP64 (not FP32) is used. Both axes are log-scale.

**Scripts:**
- `scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh` —
  builds and runs the benchmark (fixed `M=K=24576`, `SCALES=(1 4 16 64
  256 1024 4096 16384)`); reads/writes the SAME benchmark driver as the
  FP32 GEMV NO_TRANSPOSE plot in the "Increasing Operand-Dimensions"
  section above, `benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu`.
- `scripts/plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref.m` — the
  MATLAB function that regenerates the plot from an already-saved CSV;
  no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly.

**Kernels called (one per plotted series):**

Identical kernel set to the FP32 GEMV NO_TRANSPOSE plot in the "Plots -
Error for Increasing Operand-Dimensions" section above (`cublasSgemv`,
`mv_fp32_tiled`, `mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock`,
`mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal`,
`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded`,
`mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace`) -- see
that section's own table for the full Series/Kernel/Decode-function/Notes
breakdown, not repeated here.

**Files:**

Same files as the FP32 GEMV NO_TRANSPOSE plot in the "Plots - Error for
Increasing Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP32
GEMV NO_TRANSPOSE plot of the "Plots - Error for Increasing
Operand-Dimensions" section above -- see that section's own Method
Description for the full architectural derivation. Only the X-axis
changes here: shape is held fixed at `M=K=24576` and `normal_scale` is
swept instead of `M`.

#### `build/output/error_magnitude_scaling/gemv/fp32_compute_transpose/plots/plot_gemv_transpose_recon_scale_sweep_vs_cublas_dgemv_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemv/fp32_compute_transpose/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot
(same output directory/binary as the FP32 GEMV TRANSPOSE plot in the
"Increasing Operand-Dimensions" section above, run a different way),
written by `scripts/benchmarks_gemv_transpose_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh`.

##### Experiment:

The TRANSPOSED (`y = alpha*A^T*x`) fp32-reconstruction-tier GEMV
L2-relative-error sweep at a SINGLE FIXED SHAPE (M=K=24576), swept
across `normal_scale` instead of across M -- the TRANSPOSE counterpart
of the FP32 GEMV NO_TRANSPOSE scale-sweep plot just above (identical
reasoning, only the direction differs). Same six series (five TOMOS
plus cuBLAS FP32), measured against a TRUE FP64
(`cublasDgemv(CUBLAS_OP_T)`) reference. Both axes are log-scale.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh` —
  builds and runs the benchmark (fixed `M=K=24576`, `SCALES=(1 4 16 64
  256 1024 4096 16384)`); reads/writes the SAME benchmark driver as the
  FP32 GEMV TRANSPOSE plot in the "Increasing Operand-Dimensions"
  section above, `benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu`.
- `scripts/plot_gemv_transpose_recon_scale_sweep_vs_cublas_dgemv_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved
  CSV; no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly.

**Kernels called (one per plotted series):**

Identical kernel set to the FP32 GEMV TRANSPOSE plot in the "Plots -
Error for Increasing Operand-Dimensions" section above (`cublasSgemv`
`CUBLAS_OP_T`, `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec`,
`mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch`,
`mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch`,
`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch`,
`mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4`) -- see that
section's own table for the full Series/Kernel/Decode-function/Notes
breakdown, not repeated here (including the naming caveat: this
benchmark's own "TOMOS FP32 / FP16 (fast)" is that section's "TOMOS
FP32 / FP16 (implicit)").

**Files:**

Same files as the FP32 GEMV TRANSPOSE plot in the "Plots - Error for
Increasing Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP32
GEMV TRANSPOSE plot of the "Plots - Error for Increasing
Operand-Dimensions" section above -- see that section's own Method
Description for the full architectural derivation. Only the X-axis
changes here: shape is held fixed at `M=K=24576` and `normal_scale` is
swept instead of `M`.

### FP16 GEMV

#### `build/output/error_magnitude_scaling/gemv/fp16_compute/plots/plot_gemv_fp32_16_scale_sweep_l2_error_vs_cublas_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemv/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot (same output
directory/binary as the FP16 GEMV NO_TRANSPOSE plot in the "Increasing
Operand-Dimensions" section above, run a different way), written by
`scripts/benchmarks_gemv_fp32_16_scale_sweep_l2_error_vs_cublas_ref.sh`
(optionally `--normalized`).

##### Experiment:

The NO_TRANSPOSE (`y = alpha*A*x`) GENUINE 16-bit-compute GEMV
L2-relative-error sweep at a SINGLE FIXED SHAPE (M=K=65536), swept
across `normal_scale` instead of across M -- the complementary axis to
the "Increasing Operand-Dimensions" section's own M-sweep-at-one-scale-
per-run. Same seven series (five TOMOS plus cuBLAS FP16/BF16), measured
against a genuine FP32-precision reference (`cublasSgemv`). Both axes
are log-scale. **Caveat:** unlike the FP16 GEMM scale-sweep plot (always
run `--normalized`) or the NO_TRANSPOSE error plot's own
`_scale16384_normalized` filename tag, this `.m` function takes no
`scale_tag`/normalization argument at all -- it always writes the same
untagged filename regardless of whether its driving `.sh` script was run
with `--normalized` (wider, safer `SCALES` up to 16384) or without (the
default, capped `SCALES` up to 128), so the filename alone does not say
which data-generation tier produced it; check the source CSV's own
`scale` column range to tell them apart.

**Scripts:**
- `scripts/benchmarks_gemv_fp32_16_scale_sweep_l2_error_vs_cublas_ref.sh [--normalized]` —
  builds and runs the benchmark (fixed `M=K=65536`); reads/writes the
  SAME benchmark driver as the FP16 GEMV NO_TRANSPOSE plot in the
  "Increasing Operand-Dimensions" section above,
  `benchmarks/kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemv_fp32_16_scale_sweep_l2_error_vs_cublas_ref.m` — the
  MATLAB function that regenerates the plot from an already-saved CSV;
  no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly.

**Kernels called (one per plotted series):**

Identical kernel set to the FP16 GEMV NO_TRANSPOSE plot in the "Plots -
Error for Increasing Operand-Dimensions" section above
(`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best`,
`mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best`,
`mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best`,
`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2`,
`mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best`, `cublasHgemm`,
`cublasGemmEx`) -- see that section's own table for the full
Series/Kernel/Decode-function/Notes breakdown, not repeated here.

**Files:**

Same files as the FP16 GEMV NO_TRANSPOSE plot in the "Plots - Error for
Increasing Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP16
GEMV NO_TRANSPOSE plot of the "Plots - Error for Increasing
Operand-Dimensions" section above -- see that section's own Method
Description for the full architectural derivation, including the
naming caveats already flagged there (this benchmark's own "TOMOS FP32 /
FP16" and "TOMOS FP32 / FP16 (fast)" are NOT the same kernels as the
identically-labeled series in the FP16 GEMV Throughput section). Only
the X-axis changes here: shape is held fixed at `M=K=65536` and
`normal_scale` is swept instead of `M`.

#### `build/output/error_magnitude_scaling/gemv/fp16_compute_transpose/plots/plot_gemv_transpose_fp32_16_scale_sweep_l2_error_vs_cublas_ref.png`

##### Data

`build/output/error_magnitude_scaling/gemv/fp16_compute_transpose/data/` —
timestamped `.csv`/`.txt` benchmark logs used to generate this plot
(same output directory/binary as the FP16 GEMV TRANSPOSE plot in the
"Increasing Operand-Dimensions" section above, run a different way),
written by `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_scale_sweep_l2_error_vs_cublas_ref.sh`
(optionally `--normalized`).

##### Experiment:

The TRANSPOSE (`y = alpha*A^T*x`) GENUINE 16-bit-compute GEMV
L2-relative-error sweep at a SINGLE FIXED SHAPE (M=K=65536), swept
across `normal_scale` instead of across M -- the TRANSPOSE counterpart
of the FP16 GEMV NO_TRANSPOSE scale-sweep plot just above (identical
reasoning, only the direction differs). Only SIX series here (four TOMOS
plus cuBLAS FP16/BF16, not the NO_TRANSPOSE plot's own seven) -- this
TRANSPOSE tier has no separate plain "TOMOS BF16" series, only "TOMOS
FP32/BF16" (same convention already noted for this tier's Throughput and
"Increasing Operand-Dimensions" error plots). Both axes are log-scale.
**Same filename caveat as the NO_TRANSPOSE plot above:** this `.m`
function also takes no `scale_tag`/normalization argument, so its
untagged filename doesn't distinguish a `--normalized` run (`SCALES` up
to 16384) from the default, capped run (`SCALES` up to 128) -- check the
CSV's own `scale` column range.

**Scripts:**
- `scripts/benchmarks_gemv_transpose_fp32_16_pure_half_scale_sweep_l2_error_vs_cublas_ref.sh [--normalized]` —
  builds and runs the benchmark (fixed `M=K=65536`); reads/writes the
  SAME benchmark driver as the FP16 GEMV TRANSPOSE plot in the
  "Increasing Operand-Dimensions" section above,
  `benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemv_transpose_fp32_16_scale_sweep_l2_error_vs_cublas_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved
  CSV; no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly.

**Kernels called (one per plotted series):**

Identical kernel set to the FP16 GEMV TRANSPOSE plot in the "Plots -
Error for Increasing Operand-Dimensions" section above
(`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split`,
`mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired`,
`mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast`,
`mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4`,
`cublasHgemm` `CUBLAS_OP_T`, `cublasGemmEx` `CUBLAS_OP_T`) -- see that
section's own table for the full Series/Kernel/Decode-function/Notes
breakdown, not repeated here (including the naming caveat: this
benchmark's own "TOMOS FP16"/"TOMOS FP16 (fast)" are that section's
"TOMOS FP16 (implicit)"/"TOMOS FP16 (implicit, fast)").

**Files:**

Same files as the FP16 GEMV TRANSPOSE plot in the "Plots - Error for
Increasing Operand-Dimensions" section above.

**Method Description:**

Every kernel here is byte-for-byte the same one documented in the FP16
GEMV TRANSPOSE plot of the "Plots - Error for Increasing
Operand-Dimensions" section above -- see that section's own Method
Description for the full architectural derivation. Only the X-axis
changes here: shape is held fixed at `M=K=65536` and `normal_scale` is
swept instead of `M`.

## Plots - Rounding in Low-precision Encodings

Two complementary standalone experiments, deliberately built at
IDENTICAL shapes/`normal_scale` tiers so they're directly comparable:
one isolates "which FP16 GEMM *implementation*" (TOMOS/cuBLAS/CUTLASS,
all fed the exact same natively-rounded input data), the other isolates
"which encode-side *rounding method*" (round-to-nearest vs truncation,
and how many mantissa sticky bits the rounding decision examines,
holding the GEMM implementation fixed). Both plot Frobenius relative
error against a genuine FP32 (`cublasSgemm`) reference at a single
large, fixed shape swept across `normal_scale`.

### FP16 GEMM kernel-implementation comparison

#### `build/output/error_rounding/gemm/fp16_compute/plots/plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.png`

##### Data

`build/output/error_rounding/gemm/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot, written by
`scripts/benchmarks_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.sh`.
Same underlying binary (`mm_fp16_kernels_l2_error_vs_cublas_ref`) as this
benchmark's own N-sweep sibling, `scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh`,
but no longer the same output directory -- that sibling (not part of this
"Rounding" section) still writes to `build/mm_fp16_kernels_l2_error_vs_cublas_ref_output/`,
unmoved.

##### Experiment:

A standalone Frobenius-relative-error comparison of THREE different FP16
GEMM *implementations* (not different A/B encodings -- see the
fp32_16-round-sweep plot below for that separate axis), all run on the
IDENTICAL native-rounded `A_half`/`B_half` data at a single fixed shape
(M=K=N=49152), swept across `normal_scale` instead of across matrix
dimensions. Since every series shares the exact same input buffers, the
only thing that ever differs between them is which GEMM implementation
computed the result -- never the input data. Data was generated with
`--normalized` (B scaled by `1/sqrt(K)`), avoiding the K-dependent
output-storage-overflow "inf" this `__half`-typed-`C` tier is prone to
at a shape this large without it. `SCALES` is identical to the
fp32_16-round-sweep plot's own range, so the two plot families are
directly comparable at the same scales. X-axis (`normal_scale`) is
log-scale; Y-axis is LINEAR (unlike most other error plots in this
document) -- the observed error stays within roughly one decade across
these three implementations/scales, so linear reads the actual magnitude
of the differences more directly than a log axis would.

**Scripts:**
- `scripts/benchmarks_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.sh` —
  builds and runs the benchmark; reads/writes `benchmarks/kernels/cuda/matrix/mm_fp16_kernels_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved
  CSV; no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly.

**Kernels called (one per plotted series):**

| Series | Kernel | Decode function | Notes |
|---|---|---|---|
| TOMOS FP16 | `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp` | — (native, unpacked storage) | same kernel as the FP16 GEMM Throughput section; accumulates in genuine `float` internally |
| cuBLAS FP16 | `cublasHgemm` | — | external library (cuBLAS); own internal accumulation precision NOT guaranteed to match TOMOS FP16's own (may accumulate in `half`, not `float`) |
| CUTLASS FP16 | `cutlass::gemm::device::Gemm<...>` (fp16 tensor-op) | — | external library (CUTLASS); the SAME "textbook" Ampere tensorop config (128x256x32 threadblock tile, 64x64x32 warp tile, 16x8x16 instruction shape, 3-stage pipeline, FP32 accumulator) this codebase's own `cutlass_fp16_gemm_ref.cu` uses for throughput, reused here for accuracy |

**Files:**
- Kernel implementation: `src/kernels/cuda/matrix/mm_cuda_kernels.cu`
  (TOMOS FP16, same as the FP16 GEMM Throughput section; declarations in
  `src/kernels/cuda/matrix/mm_cuda_kernels.cuh`).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mm_fp16_kernels_l2_error_vs_cublas_ref.cu`
  (also contains the CUTLASS reference config).

**Method Description:**
- **TOMOS FP16** (`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`) —
  byte-for-byte the same kernel documented in the FP16 GEMM Throughput
  section above; see that section's own Method Description for the full
  architectural derivation. No decode step here either -- `A`/`B` are
  plain native `__half` arrays, generated once per shape and shared
  across all three series.
- **cuBLAS FP16** (`cublasHgemm`) — cuBLAS's own simple FP16 GEMM API,
  run on the SAME `A_half`/`B_half` buffers as TOMOS FP16. Its own
  internal accumulation precision is not guaranteed to match TOMOS
  FP16's own (TOMOS accumulates in `float` internally, confirmed by
  reading the kernel body; `cublasHgemm`'s simple API may accumulate in
  `half`) -- this series isolates "cuBLAS's own default FP16 GEMM" as a
  whole, not accumulation precision alone.
- **CUTLASS FP16** (`cutlass::gemm::device::Gemm<...>`) — `cutlass::half_t`
  and `__half` are bit-layout-compatible, so this series
  `reinterpret_cast`s the SAME `__half*` `A_half`/`B_half` buffers
  instead of regenerating a separate `cutlass::half_t` copy, keeping the
  comparison exact (no incidental data difference from a re-conversion
  step).

### FP32/FP16 encode-side rounding-method sweep

#### `build/output/error_rounding/gemm/fp16_compute/plots/plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref.png`

##### Data

`build/output/error_rounding/gemm/fp16_compute/data/` — timestamped
`.csv`/`.txt` benchmark logs used to generate this plot (same directory
the FP16 GEMM kernel-implementation comparison plot above also writes
its own data into), written by
`scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh`.
Same underlying binary (`mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref`)
as this benchmark's own N-sweep sibling, but no longer the same output
directory -- that sibling (and the `_raw` scale-sweep variant, neither
part of this "Rounding" section) still write to
`build/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_output/`,
unmoved.

##### Experiment:

A standalone experiment measuring how much the ENCODE-side rounding
*decision* itself matters for the packed `fp32_16` format's tensor-core
("pure_half") decode path, at a single fixed shape (M=K=N=49152), swept
across `normal_scale`. The real encoder, `fp32_16_split`, examines all 13
"sticky" bits of `mant_lo` (bits 12..0) plus the round bit to decide
whether to round `mant_hi` up (round-to-nearest, ties-to-even). This
benchmark sweeps how many of those 13 sticky bits the rounding decision
actually looks at -- 13 (bit-for-bit identical to `fp32_16_split`), 9, 5,
1, and 0 ("truncation", bit-for-bit identical to
`fp32_16_split_truncated`) -- to see how coarsening that decision
degrades the resulting decoded `__half` value's own accuracy, for a
FIXED downstream kernel (every series calls the SAME unmodified TOMOS
FP16 native tensor-core kernel -- see Notes below) and a FIXED decode
(`fp32_16_half_only`, which only ever reads `wh`; `wr`/`round_flag` play
no role in this specific decode's accuracy, only what ends up baked into
`wh`'s own `mant_hi`/`control`/`offset` bits matters). A second, separate
axis (mantissa WIDTH, 9 vs 10 bits) is also isolated via the "native"
and "trunc10" reference series -- see Method Description below. Data was
generated with `--normalized` (B scaled by `1/sqrt(K)`), same overflow
reasoning as the FP16 GEMM kernel-comparison plot above. `SCALES` is
identical to that plot's own range for direct comparability. X-axis
(`normal_scale`) is log-scale; Y-axis is LINEAR (same reasoning as
above -- the error spread here stays within roughly one decade).

**Scripts:**
- `scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh` —
  builds and runs the benchmark; reads/writes `benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu`.
- `scripts/plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref.m` —
  the MATLAB function that regenerates the plot from an already-saved
  CSV; no separate `.sh` wrapper exists for this plot -- call the `.m`
  function directly. Named "round" rather than the fully-consistent
  "variable_rounding" purely because the latter would exceed MATLAB's
  63-character function/file name limit -- the `.cu`/`.sh` files keep
  their own full, unabbreviated names.

**Kernels called (one per plotted series):**

All six plotted series call the SAME unmodified TOMOS FP16 kernel
(`gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`)
on a genuine `__half*` array decoded/converted ahead of time -- unlike
this tier's own promoted "TOMOS FP32/FP16" series elsewhere in this
document (`mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_...`, a
SEPARATE kernel body that decodes `wh` inline), so the only thing that
ever differs between any of these series is what values ended up in
`A`'s own `__half` array, never which kernel computed the GEMM:

| Series | Encode function | Decode function | Notes |
|---|---|---|---|
| Mantissa bits: 10, Sticky bits: 12 (FP32 to FP16) | `__float2half` | — (direct) | "tomos_fp16_native": plain native float->half rounding, no packed format involved at all; reference point |
| Mantissa bits: 10, Sticky bits: 0 (truncation) | `float_to_half_truncate10` | — (direct) | "tomos_fp16_trunc10": genuine 10-bit-mantissa TRUNCATING conversion, bypassing the packed `wh`/`wr` format entirely; isolates truncation vs rounding at a FIXED 10-bit mantissa |
| Mantissa bits: 9, Sticky bits: 13 (TOMOS FP32 / FP16) | `fp32_16_split_variable_rounding(..., sticky_bits=13)` | `fp32_16_half_only` | "tomos_fp32_fp16_round_13": verified bit-for-bit identical to the real `fp32_16_split`'s own encoding (its sticky mask is exactly `0x1FFFu`, all 13 of `mant_lo`'s bits) |
| Mantissa bits: 9, sticky bits: 5 | `fp32_16_split_variable_rounding(..., sticky_bits=5)` | `fp32_16_half_only` | "tomos_fp32_fp16_round_05" |
| Mantissa bits: 9, sticky bits: 1 | `fp32_16_split_variable_rounding(..., sticky_bits=1)` | `fp32_16_half_only` | "tomos_fp32_fp16_round_01" |
| Mantissa bits: 9, sticky bits: 0 | `fp32_16_split_variable_rounding(..., sticky_bits=0)` | `fp32_16_half_only` | "tomos_fp32_fp16_trunc": bit-for-bit identical to `fp32_16_split_truncated`'s own encoding |

Two further series are measured by the benchmark binary but NOT plotted
here (still in the CSV): `tomos_fp32_fp16_stochastic` (encoded via
`fp32_16_split_stochastic_rounding`, which rounds `mant_hi` up with
probability exactly `mant_lo/16384` -- unbiased in expectation per
element, unlike round-to-nearest or truncation) and the `sticky_bits=9`
config (`tomos_fp32_fp16_round_09`, between the plotted 13 and 5
configs).

**Files:**
- Kernel implementation: `src/kernels/cuda/matrix/mm_cuda_kernels.cu`
  (TOMOS FP16, same as the FP16 GEMM Throughput section).
- Encode functions: ALL defined directly in the benchmark driver
  `benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu`
  itself (`fp32_16_split_variable_rounding`, `float_to_half_truncate10`,
  `fp32_16_split_stochastic_rounding`), not in `fp32_16_pack.hpp` --
  deliberately additive, per an explicit instruction not to modify
  `fp32_16_split` or add a variant to the core pack header.
- Decode function: `fp32_16_half_only`, `src/core/matrix/fp32_16_pack.hpp`
  (the real, shared decode function -- unmodified).
- Benchmark driver: `benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu`.

**Method Description:**
- **TOMOS FP16 (native, `__float2half`)** — plain native float->half
  rounding (10-bit mantissa, no packed format at all); the reference
  point every other series in this experiment is compared against.
- **TOMOS FP16 (trunc10, `float_to_half_truncate10`)** — a genuine
  10-bit-mantissa TRUNCATING (not rounding) float->half conversion,
  isolating "truncation vs rounding" as its own axis, independent of the
  9-vs-10-bit-mantissa axis the packed-format series below are
  structurally stuck at.
- **TOMOS FP32/FP16 (round_13/round_05/round_01/trunc)** — all four
  decode via the real, unmodified `fp32_16_half_only`, differing only in
  how many of `mant_lo`'s 13 sticky bits
  `fp32_16_split_variable_rounding`'s own `round_up_variable` helper
  examines when deciding whether to round `mant_hi` up:
  `round_up_variable(mant_lo, mant_hi_pre_is_odd, sticky_bits)` builds a
  sticky mask over the TOP `sticky_bits` of those 13 bits (closest to the
  round bit) and ORs them together; bits below that window are ignored
  entirely, not zeroed into the OR. `sticky_bits=13` reproduces the real
  `fp32_16_split`'s own round-up decision bit-for-bit (a direct
  cross-check that this duplicated encoder logic matches the real one,
  not just a similar-looking one); `sticky_bits=0` skips the round bit
  too, degenerating to plain truncation (bit-for-bit identical to
  `fp32_16_split_truncated`). Every one of these four series is
  structurally capped at 9 effective mantissa bits, never 10 -- `wh`'s
  own bit layout (`sign(1)|exponent(5)|mant_hi(9)|control(1)`),
  reinterpreted directly as a `__half`, puts the control bit (always 0
  in the in-range case) where a real half's mantissa LSB would be. This
  is why "trunc10" above exists as a separate reference point: it
  isolates the pure mantissa-width axis (9 vs 10 bits) from the
  rounding-METHOD axis these four series actually vary.

## Note: packed-format kernels living outside `fp32_16_cuda_kernels.cu`

`src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu` is the primary home for
this codebase's packed `fp32_16`/`fp32_bf16` kernels -- every GEVM kernel
(both compute tiers, both transpose directions, both formats) documented
anywhere in this file lives there, and so does the entire 32-bit-compute
GEMM tier. It is NOT the only file with packed-format kernels, though:
four kernels, all from the 16-bit-compute GEMM tier documented in the
"FP16 GEMM" Throughput section above, are instead defined in
`src/kernels/cuda/matrix/mm_cuda_kernels.cu`, alongside the native fp16/bf16
GEMM kernels:

| Kernel | Series (FP16 GEMM Throughput) |
|---|---|
| `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp` | TOMOS FP32 / FP16 |
| `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp` | TOMOS FP32 / FP16 (fast) |
| `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp` | TOMOS FP32 / FP16 (workspace) |
| `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` | TOMOS FP32 / BF16 |

This isn't a random scattering -- each of these four is genuinely coupled
to a native GEMM kernel that lives in `mm_cuda_kernels.cu`, which is
presumably why it ended up there rather than with the rest of the packed
kernels:

- `mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp`
  and `mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp` each
  directly call a native GEMM kernel as their own tail call, in the same
  translation unit, with no forward declaration needed --
  `gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`
  and
  `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp`
  respectively (both documented as "TOMOS FP16"/"TOMOS BF16" in the same
  Throughput section).
- `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp`
  (and its `_fast` sibling) launch using
  `kGemmFp16WideBigRegInlineDirtyACongruousFastkPrecomp2SerpSharedBytes`,
  a file-local `constexpr` defined once in `mm_cuda_kernels.cu` and
  reused by the native fp16 kernel it was built to match tile/shared-memory
  geometry with (13 total references across the file).

None of the 30+ packed-format GEVM kernels, nor any 32-bit-compute-tier
packed GEMM kernel, has this kind of coupling to a same-file native
sibling -- this is specific to the 16-bit-compute GEMM tier, where the
packed kernel was deliberately built to reuse the native kernel's own
tuned tile/launch geometry (fused inline decode) or to fall back to
calling it directly (bf16, and the two-pass "(workspace)" materialize
kernel).

## LSQR

LSQR is this codebase's mixed-precision least-squares iterative solver
(`A*x = b` via `lsqr_solve()`, `benchmarks/solvers/cuda/lsqr_precision_runner.cu`
/ `src/solvers/`), tested here on large (`1,000,000`-row) tall-and-thin
`.mtx` problems, generated via
[`scripts/generate_lsqr_test_matrices.sh`](../scripts/generate_lsqr_test_matrices.sh)
into `build/matrices` by default. Every
plot in this section compares the SAME five precision/preconditioning
configurations (`--value-type`/`--refine-type`/`--precond-type` all
`fp32` throughout -- only `A`'s own storage/matvec precision and whether
a preconditioner is used changes):

| Config key | Description |
|---|---|
| `fp32_no_precond` | All FP32, no preconditioner. |
| `fp32_precond` | All FP32, with a count-sketch + QR right-preconditioner (`k=2n` sketch rows, `src/core/precond/sketch_qr.hpp`/`src/core/sketch/countsketch.cpp`) -- storage/application both run at `--precond-type` on a throwaway CPU context (no CUDA path exists for the sketch/QR step). |
| `fp16_explicit` | `--mv-type fp16 --explicit-duplicates`: `A` stored ONCE at native `__half` precision (no packing) for the per-iteration matvec, PLUS a second, independently-cast `fp32` copy used only for the restart-boundary/final residual recompute -- doubles `A`'s memory footprint. |
| `fp16_implicit` | `--mv-type fp16 --implicit-duplicates`: `A` stored ONCE, packed as `NumericType::FP32_16`; per-iteration matvecs run at the safe-decode fp16 compute tier from that packed buffer, restart-boundary/residual recompute reads the SAME buffer at its fp32 ("accurate") tier -- no second copy needed. |
| `fp16_implicit_fast` | Same as `fp16_implicit`, but with `TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1` set -- a comparison-only env-var toggle (not a real CLI flag) that switches `mv_pure_half_impl`'s decode to the lossy fast-decode sibling. Exists to make visible that fast decode's saturate-underflow-to-zero behavior has been observed to stall LSQR convergence entirely on real ill-conditioned matrices -- not because it's expected to win. |

`fp16_explicit`, `fp16_implicit`, and `fp16_implicit_fast` all run genuine
fp16 arithmetic for their per-iteration matvecs (different kernels,
different storage schemes, but none falls back to double-accumulate
compute the way plain `--mv-type fp16` with no duplicates flag still
does) -- a fair comparison of real fp16 storage/compute/decode
strategies, not one real and the rest misleading.

**Kernels called (one per configuration, NO_TRANSPOSE `A*x` / TRANSPOSE
`A^T*x` -- LSQR's own bidiagonalization needs both every iteration):**

| Config | NO_TRANSPOSE kernel | TRANSPOSE kernel | Decode function(s) | Notes |
|---|---|---|---|---|
| `fp32_no_precond` / `fp32_precond` | `mv_fp32_tiled` | `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec` | — (native, unpacked) | same kernels as the FP32 GEMV Throughput section; `fp32_precond` additionally applies the count-sketch+QR right-preconditioner each iteration |
| `fp16_explicit` | `mv_fp16_native_mixed` | `mv_fp16_native_mixed` | — (native `__half` storage, `x`/`y` kept `float`) | same kernel as "mv_fp16_native_mixed (explicit, production)" in the FP16 GEMV Throughput section's TRANSPOSE plot -- a single dispatcher handling both directions; registered at the `(FP16,FP32,FP32,compute=FP16)` dense-registry slot specifically so this triple gets real fp16 compute instead of silently upcasting to double |
| `fp16_implicit` | `mv_fp32_16_pure_half_no_transpose_flush256_swar` | `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split` | `fp32_16_half2_decode_swar`/`fp32_16_half_only` (NO_TRANSPOSE); `fp32_16_half_only_fast2` (TRANSPOSE) | NOT the same NO_TRANSPOSE kernel as "TOMOS FP16 (implicit)" in the FP16 GEMV Throughput section (`..._wide_ilp_shared_half2_atomic_best`) -- `mv_pure_half_impl`'s dispatch has since been rewired again (see Method Description); TRANSPOSE matches that section's own production series exactly |
| `fp16_implicit_fast` | `mv_fp32_16_pure_half_no_transpose_flush256_fast_swar` | `mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split` | `fp32_16_half2_decode_fast_swar`/`fp32_16_half_only_fast` (NO_TRANSPOSE); `fp32_16_half_only_fast` (TRANSPOSE) | both directions match the FP16 GEMV Throughput section's own "TOMOS FP32 / FP16 (fast)" / "TOMOS FP16 (implicit, fast)" series exactly |

**Files:**
- Dispatch/registration: `src/dispatch/dense_registries.cpp` (`mv_fp32_tiled_impl`,
  `mv_fp16_native_impl` → `mv_fp16_native_mixed`), `src/dispatch/fp32_16_registries.cpp`
  (`mv_pure_half_impl`, including the `TOMOS_FP32_16_PURE_HALF_FAST_DECODE`
  runtime toggle).
- Kernel implementations: `src/kernels/cuda/matrix/mv_cuda_kernels.cu`
  (native fp32/fp16), `src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu`
  (packed `fp32_16`) -- same files as the FP32/FP16 GEMV Throughput
  sections.
- Preconditioner: `src/core/precond/sketch_qr.hpp` (QR-based
  sketch-and-precondition, right-preconditioner), `src/core/sketch/countsketch.cpp`
  / `src/kernels/cuda/sketch/countsketch_cuda_kernels.cu`.
- Solver driver: `benchmarks/solvers/cuda/lsqr_precision_runner.cu`
  (CLI runner), wrapped by `scripts/run_lsqr_precision.sh`.

**Method Description:**
- **`fp32_no_precond` / `fp32_precond`** (`mv_fp32_tiled` / `mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec`) —
  byte-for-byte the same native fp32 kernels documented in the FP32 GEMV
  Throughput section above; see that section's own Method Description.
- **`fp16_explicit`** (`mv_fp16_native_mixed`) — a true mechanical
  duplicate of "TOMOS FP16" (`mv_fp16_native_storage`)'s entire
  col-split/`__half2`-packed/"\_best" dispatch tree, with `x`/`y` kept at
  `float` to match this dense-registry slot's own `(FP16,FP32,FP32,FP16)`
  type contract exactly (swapping in `mv_fp16_native_storage` directly,
  which has all-`__half` `x`/`y`, would silently corrupt every `x`/`y`
  read/write here). Was originally `mv_fp16_native` (a single-block-per-row
  scalar kernel, still defined/exported but no longer registered here) --
  measured faster on real A100 hardware once "TOMOS FP16" itself received
  the same col-split/half2-packing work.
- **`fp16_implicit`** (NO_TRANSPOSE: `mv_fp32_16_pure_half_no_transpose_flush256_swar`;
  TRANSPOSE: `mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split`) —
  `mv_pure_half_impl`'s current NO_TRANSPOSE dispatch, per a real,
  per-call `TOMOS_LSQR_TIMING=1` LSQR timing breakdown: below
  `kMvWideRowThreshold` calls the `cp.async`/warp-uniform-issuer kernel
  documented as "TOMOS FP32 / FP16" in the FP16 GEMV Throughput section
  (`..._wide_ilp_shared_half2_atomic_cpasync_warpuniform`, `fp32_16_half_only`
  decode); at/above it, a narrow_vec kernel that flushes its `__half2`
  accumulators into `float` every 256 columns (bounding the fp16
  accumulation chain's own error growth) and decodes two `wh` per 32-bit
  register via the bit-exact `fp32_16_half2_decode_swar` (scalar
  `fp32_16_half_only` fallback for the unaligned/tail case). This
  supersedes two earlier NO_TRANSPOSE dispatch choices found slower at
  the real LSQR shape (`M=1,000,000, K=1000`): the plain
  `..._wide_ilp_shared_half2_atomic_best` dispatcher documented as "TOMOS
  FP16 (implicit)" in the FP16 GEMV Throughput NO_TRANSPOSE plot (found
  ~1.6x slower under `--implicit-duplicates` than `--explicit-duplicates`
  at this shape, since that kernel's own `cp.async` win only applies
  below the wide-row threshold), and an intermediate `..._fast2`-shaped
  composed dispatcher (kept commented out in source for A/B revert). The
  TRANSPOSE kernel is unchanged from the FP16 GEMV Throughput section's
  own "TOMOS FP16 (implicit, fast2, vec2) [production]" series -- see
  that section's own Method Description for its own register-blocking-width
  promotion history (vec4 → vec2, `ncu`-profiling-motivated).
- **`fp16_implicit_fast`** (NO_TRANSPOSE: `mv_fp32_16_pure_half_no_transpose_flush256_fast_swar`;
  TRANSPOSE: `mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split`) —
  the lossy-decode sibling of `fp16_implicit` above, selected only when
  `TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1` is set: both kernels are
  byte-for-byte the ones documented as "TOMOS FP32 / FP16 (fast)" (FP16
  GEMV Throughput NO_TRANSPOSE plot) and "TOMOS FP16 (implicit, fast)"
  (TRANSPOSE plot) respectively -- see those entries for the full
  architectural derivation. Included in every plot in this section
  specifically to make its convergence-stalling failure mode on
  ill-conditioned matrices visible next to the safe-decode curve, not
  because it is expected to win.

### Plots - Performance

#### `build/output/lsqr/performance/plots/lsqr_performance_barplot.png`

##### Data

`build/output/lsqr/performance/data/lsqr_performance.csv` — one row per
(matrix, configuration): run count, iteration count, median/min/max solve
time, time per iteration, final residual, solution error. Raw per-run
console output lives alongside it in `build/output/lsqr/performance/data/logs/`
(`<matrix>__<config>__run<i>.log`).

##### Experiment:

A grouped bar chart of LSQR solve performance per precision/preconditioning
configuration (the five in the LSQR section intro above), one bar per test
matrix, run against the two established test problems ("cond=1000":
`tall_thin_1000.mtx`; "cond=500": `tall_thin_1000_cond500.mtx`, both
`1,000,000 x 1,000`). What is measured is `lsqr_solve()`'s own wall time
(ended with a `cudaDeviceSynchronize`) -- matrix loading, `A`'s precision
cast, and the preconditioner build are NOT included. Each configuration
runs `--repeats` times (default 3) as SEPARATE processes and the MEDIAN
solve time is plotted (min/max as error bars) -- every run is a fresh
process, so each timing includes that process's own one-time first-call
costs (cold GPU clocks, first workspace allocation) exactly as a real
single solve would pay them. Two stacked panels, same x-axis
(configurations, grouped; bars within a group = matrices): top is total
solve time, each bar annotated with its own iteration count (a config
needing more iterations to reach the same tolerance looks slower here
even if each matvec is fast); bottom is time PER ITERATION (median solve
time / iterations), isolating per-matvec cost from iteration-count
differences.

**Scripts:**
- `scripts/run_lsqr_precision_performance_compare.sh` — builds/runs the
  solver (via `scripts/run_lsqr_precision.sh`) across all (matrix,
  configuration, repeat) combinations and writes the CSV/logs; `--plot-only`
  skips the runs and just re-plots an existing CSV.
- `scripts/plot_lsqr_performance.m` — the MATLAB function that
  regenerates the plot from an already-saved CSV; no separate `.sh`
  wrapper exists for this plot -- call the `.m` function directly (or via
  `run_lsqr_precision_performance_compare.sh`, which calls it for you).

**Kernels called:** see the LSQR section intro above (same five
configurations, same kernels, for every plot in this section).

**Files:**
- Solver driver: `benchmarks/solvers/cuda/lsqr_precision_runner.cu`.
- See the LSQR section intro above for dispatch/kernel/preconditioner
  file locations.

**Method Description:** see the LSQR section intro above.

### Plots - Convergence

#### `build/output/lsqr/error/lsqr_residual_history_tall_thin_1000.png`, `build/output/lsqr/error/lsqr_residual_history_tall_thin_1000_cond500.png`, `build/output/lsqr/error/lsqr_residual_history_tall_thin_1000_cond1000_norm.png`

##### Data

Each of the following three subdirectories of `build/output/lsqr/error/`
holds the same five files -- `residual_fp32_no_precond.txt`,
`residual_fp32_precond.txt`, `residual_fp16_precond_explicit.txt`,
`residual_fp16_precond_implicit.txt`, `residual_fp16_precond_implicit_fast.txt`
(one plain-text "`iteration value`" per line, 1-based iteration index,
value already a relative residual) -- one subdirectory per matrix, so the
three matrices' identically-named files don't collide:

- `build/output/lsqr/error/tall_thin_1000/`
- `build/output/lsqr/error/tall_thin_1000_cond500/`
- `build/output/lsqr/error/tall_thin_1000_cond1000_norm/`

Every matrix's own plot instead lands flat in `build/output/lsqr/error/`
itself, one directory level up from its own source data:
`lsqr_residual_history_tall_thin_1000.png`/`.fig`,
`lsqr_residual_history_tall_thin_1000_cond500.png`/`.fig`,
`lsqr_residual_history_tall_thin_1000_cond1000_norm.png`/`.fig`.

This directory previously also held a fourth, separate run
(`error_tall_thin_1000/`, alongside a same-named, un-suffixed `error/` run)
-- both against the identical matrix, `tall_thin_1000.mtx`: their
`residual_fp32_no_precond.txt` values (unpreconditioned LSQR is
essentially deterministic given the same `(A, x_true, b)`) agreed to ~8
significant figures at every iteration, confirming the same underlying
problem, while the *preconditioned* configurations' values differed
starting at the 3rd significant digit and converged in a different
number of iterations -- consistent with the count-sketch preconditioner's
CUDA kernel building its sketch via concurrent `atomicAdd`s (fixed hash
seeds, but not a fixed reduction order), so tiny run-to-run
floating-point differences there get amplified through the QR
factorization and every iteration after. The duplicate run
(`error/`'s own un-suffixed files) was dropped; `tall_thin_1000/` is the
surviving copy. `tall_thin_1000_cond1000_norm/`'s own matrix was
confirmed (not just inferred from directory naming) by checking which of
the two candidate `.mtx` files under the matrix directory
actually has the `_xtrue.mtx`/`_b.mtx` companions
`run_lsqr_precision_compare.sh` requires -- `tall_thin_1000_norm.mtx` has
neither, `tall_thin_1000_cond1000_norm.mtx` has both.

##### Experiment:

A semilogy plot (residuals span orders of magnitude as LSQR converges) of
relative-residual-vs-iteration curves for the same five configurations
(see the LSQR section intro above), overlaid on one figure per matrix --
the convergence-curve counterpart of the Performance bar chart above (same
driver script family, same five configurations, same matrices in
principle, different metric: convergence behavior over iterations rather
than wall-clock time).

**Scripts:**
- `scripts/run_lsqr_precision_compare.sh` — builds/runs the solver (via
  `scripts/run_lsqr_precision.sh`) for a single `(A, x_true, b)` problem
  across all five configurations and writes the five residual-history
  files (under `<output-dir>/<matrix-name>/`, `<matrix-name>` derived from
  `--matrix`'s own basename) plus the plot
  (`<output-dir>/lsqr_residual_history_<matrix-name>.png`/`.fig`).
- `scripts/plot_lsqr_residual_history.m` — the MATLAB function that
  regenerates the plot from a given list of residual-history files/labels
  plus an output name; no separate `.sh` wrapper exists for this plot --
  call the `.m` function directly (or via `run_lsqr_precision_compare.sh`,
  which calls it for you) to combine an arbitrary subset of history files
  onto one figure.

**Kernels called:** see the LSQR section intro above.

**Files:** see the LSQR section intro above.

**Method Description:** see the LSQR section intro above.

### Plots - Speedup

#### `build/output/lsqr/speedup_by_columns/lsqr_speedup_by_columns_barplot.png`, `build/output/lsqr/speedup_by_columns/lsqr_speedup_by_columns_normalized_barplot.png`

##### Data

`build/output/lsqr/speedup_by_columns/lsqr_speedup_by_columns.csv` — one
row per (columns, condition-number variant, configuration): run count,
iteration count, median/min/max solve time, etc. (same schema as
`lsqr_performance.csv`). Raw per-run console output in
`build/output/lsqr/speedup_by_columns/logs/` (`n<cols>__<cond>__<config>__run<i>.log`).
Several `*_backup.csv`/`*_before_*.csv` snapshots also live in this
directory from earlier stages of this same sweep -- not read by the
current plot script, kept as an informal history.

##### Experiment:

The SAME speedup ratio as the plot above (`speedup = solve_ms_median(fp32_precond)
/ solve_ms_median(method)`, same three mixed-precision configurations),
but swept across problem WIDTH (number of columns: 1000/2000/3000) and
TWO condition-number variants per width, all `1,000,000`-row tall-and-thin
matrices generated via `scripts/generate_lsqr_test_matrices.sh`:

| Variant | Label | Matrix pattern |
|---|---|---|
| `cond1000` | `cond=1000` | `tall_thin_<n>.mtx` |
| `cond500` | `cond=500` | `tall_thin_<n>_cond500.mtx` |
| `cond1000norm` | `cond=1000-normalized` | `tall_thin_<n>_cond1000_norm.mtx` |

`cond1000` and `cond1000norm` are the SAME matrices at two scales (the
normalized one has its spectrum divided by its largest singular value).
Two bar charts are drawn from the one CSV: `lsqr_speedup_by_columns_barplot.png`
(bars grouped per column count; within a group, cond=1000 vs cond=500,
solid vs lighter shade of each method's color) and
`lsqr_speedup_by_columns_normalized_barplot.png` (cond=1000 unnormalized
vs normalized instead). Each bar is annotated with both its speedup
ratio and that run's own iteration count, same reasoning as the plot
above. A dashed line marks 1x parity with the `fp32_precond` baseline.

COST note (from the driver script's own header): every run re-reads its
`.mtx` from disk (~23.5/47/70.5 GB of ASCII for 1000/2000/3000 columns);
the script is resumable (appends to the CSV per completed
(columns, cond, configuration), skips pairs already present on a re-run)
specifically because of this cost.

**Scripts:**
- `scripts/run_lsqr_precision_speedup_by_columns.sh` — builds/runs the
  solver (via `scripts/run_lsqr_precision.sh`) across all (columns,
  variant, configuration, repeat) combinations and writes the CSV/logs;
  `--plot-only` skips the runs and just re-plots an existing CSV;
  `--fresh` discards the existing CSV and starts over; `--variants`
  selects which of the three variants above to run (default
  `"cond1000 cond500"`).
- `scripts/plot_lsqr_speedup_by_columns.m` — the MATLAB function that
  regenerates both plots from an already-saved CSV (one call per
  `cond_labels` pairing); no separate `.sh` wrapper exists for this plot
  -- call the `.m` function directly (or via
  `run_lsqr_precision_speedup_by_columns.sh`, which calls it for you).

**Kernels called:** the three mixed-precision configurations only
(`fp16_explicit`, `fp16_implicit`, `fp16_implicit_fast`) -- see the LSQR
section intro above; `fp32_precond` is the baseline divisor, not a
plotted bar.

**Files:** see the LSQR section intro above.

**Method Description:** see the LSQR section intro above.


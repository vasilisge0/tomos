# Benchmarks

This directory holds two distinct kinds of things, built two different ways:

- **`solvers/`** and the Google-Benchmark suites in `kernels/*/matrix/` (the
  `bench_*` files) are ordinary CMake targets: `cmake --build . --target
  <name>` produces an executable under `build/benchmarks/...` that you run
  directly.
- **Most of `kernels/cuda/matrix/`** (the GEMM/GEMV kernel-tuning files) are
  **not** CMake targets at all. They're compiled directly with `nvcc` by a
  matching `scripts/<name>.sh` wrapper script, which also knows the right
  problem sizes, warmup/iteration counts, and (for the `l2_error`/`verify_*`
  files) writes the data that a companion `scripts/plot_*.m` script later
  plots. Look for the file's own script in the tables below before assuming
  it needs wiring into CMake — most of them deliberately don't.

Every table entry below was produced by reading the file's own header
comment (and, for the CUDA kernel-tuning files, cross-checking which
`scripts/*.sh` actually builds/runs it) — not guessed from the filename.

---

## `kernels/cpu/matrix/` — CPU kernel timing (Google Benchmark)

All three are plain Google Benchmark executables (`BENCHMARK`/
`BENCHMARK_TEMPLATE`), timing-only (no correctness checks), built via
`benchmarks/kernels/cpu/matrix/CMakeLists.txt`.

| File | Benchmarks | Compared against |
|---|---|---|
| `bench_mv_cpu_kernels.cpp` | `mv_strided` (dense GEMV) across double/float/mixed type combos | — (pure timing, no baseline) |
| `bench_mm_cpu_kernels.cpp` | `mm_strided` (dense GEMM) across type combos, plus `mm_csr` (sparse GEMM) | — |
| `bench_fp32_16_cpu_kernels.cpp` | FP32_16 packed-format `mv`/`mm` pure_half/fast/accurate tiers | native `mv_strided`/`mm_strided` fp32 baseline, in the same file |

---

## `kernels/cuda/matrix/` — CUDA GEMV/GEMM kernel tuning

This is the bulk of the directory (48 files) and the active surface for the
project's ongoing CUDA GEMV/GEMM kernel-tuning work (see the FP32_16/
FP32_BF16 packed-precision formats in `src/kernels/cuda/matrix/
fp32_16_cuda_kernels.cu`). Grouped below by what each file is actually for.

### Google-Benchmark CMake targets (timing, broad sweeps)

Built via `benchmarks/kernels/cuda/matrix/CMakeLists.txt`, run directly
(no wrapper script).

| File | Benchmarks | Compared against |
|---|---|---|
| `bench_mv_cuda_kernels.cu` | `gevm_generic_kernel` across double/float/half, square/tall/wide sweeps | cuBLAS (`BM_mv_cublas`: Dgemv/Sgemv/Hgemm(n=1)) |
| `bench_mm_cuda_kernels.cu` | `gemm_generic_kernel` across type combos, plus `BM_mm_csr_cuda` | cuBLAS (`BM_mm_cublas`: Dgemm/Sgemm/Hgemm) |
| `bench_mm_fp32_square_cuda_kernels.cu` | Square-only (M=K=N) fp32 GEMM sweep (`gemm_fp32_tiled` family) — scoped down from `bench_mm_cuda_kernels.cu` for fast iteration | cuBLAS Sgemm |
| `bench_mm_fp32_16_square_cuda_kernels.cu` | Square-only FP32_16 GEMM sweep (accurate/tiled/cpasync/warptile/dualbuf/splitk family) | cuBLAS (baseline family, same convention as the fp32 sibling) |
| `bench_fp32_16_cuda_kernels.cu` | FP32_16 packed `mv`/`mm` pure_half/fast/accurate tiers (NO_TRANSPOSE only) | cuBLAS baseline |

### Native fp32 GEMV — "reconstruction" tier

Timing and accuracy of the plain native-fp32 GEMV kernel plus the FP32_16/
FP32_BF16 packed kernels that fully reconstruct fp32 before computing.

| File | Type | Transpose | Script(s) |
|---|---|---|---|
| `mv_fp32_recon_vs_cutlass_ref.cu` | Timing | NO_TRANSPOSE | `benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh` |
| `mv_fp32_recon_vs_cutlass_ref_advanced.cu` | Timing (warmup/iters CLI flags) | NO_TRANSPOSE | `benchmarks_gemv_fp32_recon_vs_cutlass_ref_advanced.sh` |
| `mv_fp32_recon_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed, vs genuine FP32 ref | NO_TRANSPOSE | `benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh` (+ scale-sweep variant) |
| `mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu` | Correctness, vs TRUE FP64 ref (exposes real accumulation error) | NO_TRANSPOSE | `benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh` (+ scale-sweep variant) |
| `mv_fp32_transpose_recon_vs_cublas_ref.cu` | Timing | TRANSPOSE | `benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh` |
| `mv_fp32_transpose_recon_vs_cublas_ref_advanced.cu` | Timing (warmup/iters flags) | TRANSPOSE | `benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh` |
| `mv_fp32_transpose_recon_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed | TRANSPOSE | `benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_ref.sh` (+ scale-sweep variant) |
| `mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu` | Correctness, vs TRUE FP64 ref | TRANSPOSE | `benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh` (+ scale-sweep variant) |
| `verify_gemv_fp32_recon_plot_promoted.cu` | Correctness-only, trimmed to exactly the 5 kernels the plot script plots (NO_TRANSPOSE) | NO_TRANSPOSE | `verify_gemv_fp32_recon_plot_promoted.sh` |
| `verify_mv_fp32_transpose_promoted.cu` | Correctness-only, trimmed to exactly the 5 promoted TRANSPOSE kernels — **your primary fast test loop, kept in sync on every promotion** | TRANSPOSE | `verify_mv_fp32_transpose_promoted.sh` |

### FP32_16 / FP32_BF16 packed GEVM — "pure_half" (genuine 16-bit compute) tier

| File | Type | Transpose | Script(s) |
|---|---|---|---|
| `mv_fp32_16_pure_half_vs_cublas_ref.cu` | Timing (5 series: FP16 dispatcher, hybrid, bf16) | NO_TRANSPOSE | `benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh` (+ tall-1000cols variant) |
| `mv_fp32_16_pure_half_vs_cublas_ref_advanced.cu` | Timing (warmup/iters flags, 4 series) | NO_TRANSPOSE | `benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref_advanced.sh` |
| `mv_fp32_16_pure_half_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed (6 series) | NO_TRANSPOSE | `benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh` |
| `mv_fp32_16_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed (across the square-M sweep) | NO_TRANSPOSE | `benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh` (+ scale-sweep variant) |
| `mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu` | Timing (2 series) | TRANSPOSE | `benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh` (+ tall-1000cols variant) |
| `mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced.cu` | Timing (warmup/iters flags) | TRANSPOSE | `benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh` |
| `mv_fp32_16_pure_half_transpose_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed (5 series) | TRANSPOSE | `benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh` (+ scale-sweep variant) |
| `verify_gemv_fp32_16_plot_promoted.cu` | Correctness-only, trimmed to the 5 kernels the NO_TRANSPOSE plot script plots | NO_TRANSPOSE | `verify_gemv_fp32_16_plot_promoted.sh` |
| `verify_gemv_fp32_16_pure_half_transpose_plot_promoted.cu` | Correctness-only, trimmed to the 4 kernels the TRANSPOSE plot script plots | TRANSPOSE | `verify_gemv_fp32_16_pure_half_transpose_plot_promoted.sh` |
| `verify_mv_fp32_16_gemv.cu` | The full ~50-kernel exploratory correctness + performance sweep (wide/split-K/tiled/narrow_vec, both transpose directions, boundary/tail cases) — the suite the `plot_promoted` files above are trimmed subsets of | both | `verify_mv_fp32_16_gemv.sh` |
| `diag_pure_half_transpose_nan.cu` | **Standalone diagnostic, not a permanent benchmark.** Downloads a kernel's intermediate per-block workspace to pin down exactly where a NaN first appears in `mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split` at m=140000,k=100 | TRANSPOSE | none (manual `nvcc` build) |
| `verify_repro_tiled_vs_untiled_large_m.cu` | **Standalone correctness repro, not a permanent benchmark.** Targets `mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled` at M=K=65536 specifically, large enough for its internal sub-tile loop to iterate more than once (M=70, the standard `verify_mv_fp32_16_gemv.cu` sweep's own shape, never exercises this) — checks the cross-sub-tile accumulation and double-`__syncthreads()` against a CPU reference. Moved here from the repo root, where it had been left behind. | TRANSPOSE | none (manual `nvcc` build) |
| `mv_fp32_transpose_randls2_0_warpdot_tiled_reference.cu` | Standalone reference: gets a genuine timing number for the external `randls2.0` project's own plain-fp32 TRANSPOSE GEMV kernel (that repo has no runnable benchmark for it), for comparison against tomos's own kernels elsewhere | TRANSPOSE | none (manual `nvcc` build) |

### GEMM — native fp32 / fp16 / bf16 tiers

| File | Type | Precision | Script(s) |
|---|---|---|---|
| `verify_mm_fp32_gemm.cu` | Correctness + perf, the original fp32 GEMM kernel-tuning harness | fp32 native | `verify_mm_fp32_gemm.sh` |
| `verify_mm_fp16_gemm.cu` | Correctness + perf for custom fp16 GEMM (`gemm_fp16_tiled`/`_wmma`) | fp16 native | `verify_mm_fp16_gemm.sh` |
| `cutlass_fp16_gemm_ref.cu` | Reference-only: real CUTLASS `device::Gemm` fp16, same shape as `verify_mm_fp16_gemm.cu`, vs cuBLAS Hgemm | fp16 native | `cutlass_fp16_gemm_ref.sh` |
| `verify_bf16_wide_bigreg_sq_deep4.cu` | Correctness + throughput for the bf16 sibling of the fp16 `wide_bigreg_sq_deep4` kernel | bf16 native | `verify_bf16_wide_bigreg_sq_deep4.sh` |
| `wide_bigreg_family_vs_cutlass_ref.cu` | Focused timing of 3 cp.async-B-fixed kernels (narrower than `verify_mm_fp16_gemm.cu`'s ~50-kernel sweep) | fp16 native | `wide_bigreg_family_vs_cutlass_ref.sh` |

### GEMM — FP32_16 / FP32_BF16 packed tiers

| File | Type | Precision | Script(s) |
|---|---|---|---|
| `benchmarks_gemm_fp32.cu` | Timing: full fp32-reconstruction tier vs native fp32, CUTLASS, cuBLAS Sgemm | fp32_16 + fp32_bf16 packed | `benchmarks_gemm_format_32_16_compute_32.sh` (+ rect/subnormal/density-table variants) |
| `fp32_recon_vs_cutlass_ref_advanced.cu` | Timing, `_advanced` (warmup/iters) variant of the above | same | `benchmarks_gemm_format_32_16_compute_32_advanced.sh` |
| `fp32_16_plotted_vs_cutlass_ref.cu` | Timing, trimmed to exactly what `plot_gemm_format_32_16_compute_32.m` plots | fp32_16 + fp32_bf16 packed | `fp32_16_plotted_vs_cutlass_ref.sh` |
| `fp32_new_kernels_vs_cutlass_ref.cu` | Timing, trimmed to only the warptile/snake/midpoint_exact investigation kernels | fp32 native + fp32_16 packed | `fp32_new_kernels_vs_cutlass_ref.sh` |
| `mm_fp32_recon_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed, vs genuine FP32 ref | fp32_16 + fp32_bf16 packed | `benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh` |
| `mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu` | Correctness ("fp32-compute" tier), vs TRUE FP64 ref | native fp32 + fp32_16 + fp32_bf16 packed | `benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh` (+ scale-sweep, raw variants) |
| `mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu` | Correctness: sweeps how many sticky bits (13/9/5/1/0) the encode-side rounding examines | fp32_16 packed | `benchmarks_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.sh` (+ scale-sweep, raw variants) |
| `verify_mm_fp32_16_gemm.cu` | Correctness + perf for the full wh+wr reconstruction GEMM family | fp32_16 packed | `verify_mm_fp32_16_gemm.sh` |
| `verify_mm_fp32_16_pure_half_gemm.cu` | Correctness + perf for the wh-only genuine-`__half`-compute GEMM family | fp32_16 packed (pure_half) | `verify_mm_fp32_16_pure_half_gemm.sh` |
| `verify_mm_fp32_recon_plot_promoted.cu` | Correctness-only, trimmed to the 6 kernels the plot script plots | fp32_16 + fp32_bf16 packed vs native fp32 | `verify_mm_fp32_recon_plot_promoted.sh` |
| `wide_bigreg_sq_fp32_16_vs_cutlass_ref.cu` | Focused timing: the FP32_16 pure_half `wide_bigreg_sq` kernel and its 2 best wmma siblings | fp32_16 packed (pure_half) | `wide_bigreg_sq_fp32_16_vs_cutlass_ref.sh` |
| `mm_fp32_16_stochastic_rounding_diagnostic.cu` | **Standalone diagnostic, not a permanent benchmark.** Statistically validates that `fp32_16_split_stochastic_rounding`'s PRNG pairing produces correct Bernoulli(p) draws (motivated by stochastic rounding measuring worse error than truncation, unexpectedly) | fp32_16 packed (encode-side only) | none (manual `nvcc` build) |

### GEMM — fp16/bf16 "pure/composite" tier (tensor-core, decode-only)

| File | Type | Precision | Script(s) |
|---|---|---|---|
| `benchmarks_gemm_fp16.cu` | Timing: pure/composite tier vs native fp16/bf16, real CUTLASS, cuBLAS Hgemm/GemmEx | fp16 + bf16 native, fp32_16 + fp32_bf16 packed | `fp16_bf16_pure_vs_cutlass_ref.sh` (+ matched/rect/subnormal variants) |
| `fp16_bf16_pure_vs_cutlass_ref_advanced.cu` | Timing, `_advanced` (warmup/iters) variant | same | `fp16_bf16_pure_vs_cutlass_ref_matched_advanced.sh` |
| `mm_fp16_kernels_l2_error_vs_cublas_ref.cu` | Correctness: 3 fp16 GEMM impls (native TOMOS, cuBLAS, CUTLASS) vs genuine FP32 ref | fp16 native | `benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh` (+ scale-sweep, raw variants) |
| `mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu` | Correctness + plot-data feed for the pure/composite tier | fp16 + bf16 native, fp32_16 + fp32_bf16 packed | `benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh` (+ scale-sweep variant) |
| `verify_mm_fp16_bf16_pure_plot_promoted.cu` | Correctness-only, trimmed to the 5 kernels the plot script plots | fp16 + bf16 native, fp32_16 + fp32_bf16 packed | `verify_mm_fp16_bf16_pure_plot_promoted.sh` |

### Standalone profiling diagnostics (not benchmarks, no script)

Self-documented in their own header comments as one-off investigation
artifacts, meant to be compiled manually with `nvcc` when needed — not
wired into any build or script, by design.

| File | Purpose |
|---|---|
| `a_load_repro.cu` | Isolated repro of a GEMM A-tile `cp.async` load loop, to check whether an `ncu` "L2 Theoretical Sectors Global Excessive" finding reproduces outside the full kernel |
| `a_load_repro_cutlass_iterator.cu` | Same question, using CUTLASS's own `PredicatedTileAccessIterator` directly, to compare against tomos's reimplementation |

---

## `solvers/cpu/` and `solvers/cuda/` — LSQR benchmarks

| File | Type | Notes |
|---|---|---|
| `solvers/cpu/bench_lsqr.cpp` | Google Benchmark, CPU | `BM_lsqr_dense`/`BM_lsqr_sparse` (double/float), plus sketch+QR-preconditioned and ill-conditioned variants, on synthetic consistent systems |
| `solvers/cuda/bench_lsqr_cuda.cu` | Google Benchmark, CUDA | `BM_lsqr_dense_cuda`, `BM_lsqr_dense_illconditioned_cuda`, same synthetic-system approach mirrored on the GPU |
| `solvers/cuda/lsqr_precision_runner.cu` | Standalone CLI tool, not Google Benchmark | Loads real `.mtx` files (via direct `mmio.c` calls — see the file's own comment on why, and `src/io/io.cpp`'s history), runs a full precision/timing sweep across `--mv-type`/`--value-type`/`--refine-type`/`--precond-type`/`--implicit-duplicates`/`--explicit-duplicates` flags |
| `solvers/cuda/lsqr_gemv_isolated.cu` | Standalone CLI tool | Isolates *only* the per-iteration GEMV at LSQR's real matrix shape (1,000,000×1,000), through the exact same `mv_apply(...)` entry point `lsqr.cpp` itself calls, for both FP32_16 and native FP16 storage — built to make two otherwise hard-to-compare timing numbers directly comparable from one harness |

---

## A note on scope

This README documents what each file *is* — not whether it's still worth
keeping. A handful of files in `kernels/cuda/matrix/` (the standalone
diagnostics above, plus a couple of square-only CMake sweeps with no wrapper
script) aren't referenced by anything else in the repo; that's expected for
this directory's workflow (ad hoc `nvcc` + manual runs are normal here), not
a sign of neglect. Nothing in this file changes any kernel, script, or build
target — it's documentation only.

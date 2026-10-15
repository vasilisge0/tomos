# TOMOS

A CUDA/CPU library for dense/sparse GEMM, GEMV, and LSQR, including two
packed low-precision storage formats (`FP32_16` and `FP32_BF16`) that trade
storage for either full-fp32 reconstruction or genuine 16-bit tensor-core
compute. This document covers building the project, running its tests, and
reproducing the experiments and plots behind the paper.

Five companion documents go into more depth than this file does:

- [`documentation/install.md`](documentation/install.md) — full build instructions, targeting a different GPU architecture, and the optional local `.configure.sh` mechanism for `CUTLASS_DIR`/`MATRIX_DIR`.
- [`documentation/experiments-in-paper.md`](documentation/experiments-in-paper.md) — every plot in the paper, figure by figure, with its exact data source and a description of the experiment.
- [`documentation/experiments-complete.md`](documentation/experiments-complete.md) — the exhaustive list of every kernel variant each benchmark driver times internally, not just the subset that made it into a plot.
- [`documentation/scripts.md`](documentation/scripts.md) and [`scripts/Documentation.md`](scripts/Documentation.md) — indexes of every script in `scripts/`, by experiment and by task/precision respectively.
- [`documentation/third-party-notices.md`](documentation/third-party-notices.md) — third-party software used in developing TOMOS (OpenBLAS, CUTLASS), their licenses, and exactly how each is used.

---

## 1. Install

```sh
scripts/configure/configure_sm_80.sh   # configures build/, targets sm_80 (A100)
scripts/configure/build.sh             # cd build; make -j4
```

See [`documentation/install.md`](documentation/install.md) for
prerequisites, targeting a different GPU architecture, the
`scripts/configure/configure.sh` pitfall (don't use it — it silently
builds a CPU-only tree), and the optional local `.configure.sh`
mechanism for script-level `CUTLASS_DIR`/`MATRIX_DIR` defaults.

---

## 2. Tests

There's no `ctest`/`make test` integration — `cmake --build` only compiles
test binaries, it never runs them. Build the target you want, then run its
binary (or the matching launcher script under `scripts/test/`, which mirrors
`test/`'s own directory layout and is just a 2-line `cd build/... && ./test_X`
each):

```sh
cmake --build build --target test_blas -j4
bash scripts/test/core/blas/test_blas.sh
# equivalently: ./build/test/core/blas/test_blas
```

Every test binary has a matching launcher script. A few representative
targets, one per layer:

| Layer | Example target | Launcher |
|---|---|---|
| Core library (`src/core/`) | `test_lsqr`, `test_workspace` | `scripts/test/core/solver/test_lsqr.sh`, `scripts/test/core/workspace/test_workspace.sh` |
| Dispatch/registries (`src/dispatch/`) | `test_dense_registries` | `scripts/test/dispatch/test_dense_registries.sh` |
| CPU kernels (`src/kernels/cpu/`) | `test_fp32_16_cpu_kernels` | `scripts/test/kernels/cpu/matrix/test_fp32_16_cpu_kernels.sh` |
| CUDA kernels (`src/kernels/cuda/`) | `test_sketch_qr_cuda_kernels` | `scripts/test/kernels/cuda/precond/test_sketch_qr_cuda_kernels.sh` |

There's no single "build and run everything" command — build and run each
target you care about individually.

---

## 3. Run experiments included in the paper

Four orchestrator scripts each run the exact set of experiments behind the
paper's plots — one "Row" per figure, matched 1:1 against
[`documentation/experiments-in-paper.md`](documentation/experiments-in-paper.md)
— and auto-generate the resulting plots as they go (each sub-script invokes
MATLAB itself at the end of its run):

```sh
scripts/benchmarks_all.sh            # throughput, 6 rows
scripts/benchmarks_all_advanced.sh   # throughput, fixed warmup/iters methodology, 8 rows
scripts/error_all.sh                 # L2-relative-error analysis, 8 rows
scripts/correctness_all.sh           # pass/fail correctness checks, 8 rows
```

These expect to run on the real GPU (an A100 on this project's cluster,
Purdue Gilbreth) — if not already inside a SLURM job allocation, each script
finds your running job on an `a100*` partition and re-execs itself into it.
This is a lot of GPU time (full shape sweeps, several reaching
multi-billion-element shapes) — expect hours, not minutes, for the full set.
Run one script at a time if you only need a subset.

LSQR's experiments are separate (not part of the four orchestrators above):

```sh
scripts/run_lsqr_precision_compare.sh              # convergence curves, 5 precision/precond configs
scripts/run_lsqr_precision_performance_compare.sh  # same 5 configs, solve-time bar chart
scripts/run_lsqr_precision_speedup_by_columns.sh    # speedup vs preconditioned FP32, across problem widths
```

These need real `.mtx` matrix files as input. Generate the exact set the
two scripts above expect (into `build/matrices`, their default
`--matrix-dir`) with:

```sh
scripts/generate_lsqr_test_matrices.sh              # full default set (columns 1000/2000/3000)
scripts/generate_lsqr_test_matrices.sh --columns 1000  # just what run_lsqr_precision_performance_compare.sh needs
```

or generate a single custom problem directly with the lower-level
generator both of the above call internally:

```sh
scripts/generate_tall_thin_mtx.sh -m ROWS -n COLS -o OUTPUT_PATH -x X_OUTPUT_PATH -b B_OUTPUT_PATH
```

See [`documentation/experiments-in-paper.md`](documentation/experiments-in-paper.md)'s own `## LSQR` section for the exact matrices/problem sizes used in the paper.

The scripts above, and every `scripts/*vs_cutlass_ref*.sh` comparison
benchmark in Section 4 (which similarly defaults to shallow-cloning
CUTLASS into `build/cutlass-ref`), read a `MATRIX_DIR`/`CUTLASS_DIR`
override from a gitignored `.configure.sh` at the repo root, if present —
copy `.configure.sh.example` to `.configure.sh` and set your own paths
there once instead of passing `--matrix-dir`/`CUTLASS_DIR=` on every
invocation.

---

## 4. Run the complete set of experiments

Beyond the promoted subset above, each benchmark driver actually times many
more kernel variants internally — every explored tuning attempt, not just
the one that got promoted into a plot. Run these directly, one family at a
time (each builds a fresh executable and runs it):

```sh
scripts/verify_mm_fp32_gemm.sh           # fp32 GEMM: full correctness + perf
scripts/verify_mm_fp16_gemm.sh           # fp16 GEMM
scripts/verify_mm_fp32_16_gemm.sh        # FP32_16/FP32_BF16, accurate/reconstruction tier
scripts/verify_mm_fp32_16_pure_half_gemm.sh  # FP32_16, pure_half tier
scripts/verify_bf16_wide_bigreg_sq_deep4.sh  # bf16 / FP32_BF16 pure_bf16 tier
scripts/verify_mv_fp32_16_gemv.sh        # FP32_16 GEVM: the ~50-kernel full sweep
scripts/run_cuda_benchmarks.sh           # google-benchmark-based exhaustive sweeps
```

Most accept `-- --skip-correctness` / `-- --skip-performance` / `-- --only-shape N`
to narrow scope — see each script's own header comment. For the full
precision/task quick-reference table (which script covers which kernel
family), see [`scripts/Documentation.md`](scripts/Documentation.md); for the
exhaustive method-by-method listing of what each driver times, see
[`documentation/experiments-complete.md`](documentation/experiments-complete.md).

---

## 5. Recreate plots included in the paper

Every script in Section 3 already plots as part of running it — you don't
normally need a separate plotting step. If you already have the data (a
previous run's `.csv`, or a log someone sent you) and just want to re-plot
without touching the GPU again, use the standalone plotting wrapper for that
experiment instead — these are pure post-processing, no GPU or SLURM
allocation needed, and finish in seconds. `CSV_PATH` is optional — omit it
and each script uses the most recently written matching CSV under `build/`:

```sh
scripts/plot_gemv_fp32_recon_vs_cutlass_ref.sh [CSV_PATH]
scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.sh [CSV_PATH]
scripts/plot_gemm_format_32_16_compute_32.sh [CSV_PATH]
scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh [CSV_PATH]
scripts/plot_lsqr_precision_speedup.sh        # reads lsqr_performance.csv, no args
```

For experiments without a dedicated `plot_*.sh` wrapper, call the matching
`.m` function directly (see Section 6).

---

## 6. Recreate the complete set of plots

`scripts/` has 41 MATLAB `.m` plotting scripts in total — one per experiment
(both the promoted ones in Section 5 and the rest of the complete set). Call
the one matching your data file directly:

```sh
matlab -batch "addpath('scripts'); plot_gemv_fp32_16_l2_error_vs_cublas_ref('path/to/data.csv')"
```

(falls back to `module load matlab/R2024a` automatically if `matlab` isn't
already on `PATH`, same as every benchmark script in Section 3/4 does). See
[`documentation/experiments-complete.md`](documentation/experiments-complete.md)
for which plot corresponds to which experiment driver.

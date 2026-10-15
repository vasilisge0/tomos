# scripts/ — which script to run for what

This indexes every benchmark/verification script in this directory by
**precision/packing variant** and **task** (correctness, performance,
GEMM vs GEMV). All of them build a fresh executable before running (so
results always reflect the current state of the kernels), and most
accept `-- <args>` or `[N]` to narrow scope — see each script's own
header comment for exact flags.

Prerequisite for all of them: the project must already be configured
(`scripts/configure/`) with a `build/` directory present.

## Quick reference

| Precision / packing                          | GEMM correctness + perf | GEMM vs CUTLASS/cuBLAS (throughput only) | GEMV |
|------------------------------------------------|--------------------------|-------------------------------------------|------|
| fp32 (plain)                                    | `verify_mm_fp32_gemm.sh` | *(native fp32 line inside `fp32_recon_vs_cutlass_ref.sh`)* | — none |
| fp16 (plain)                                    | `verify_mm_fp16_gemm.sh` | `wide_bigreg_family_vs_cutlass_ref.sh`, `cutlass_fp16_gemm_ref.sh` (CUTLASS+cuBLAS only, no tomos kernel) | — none |
| bf16 (plain)                                    | `verify_bf16_wide_bigreg_sq_deep4.sh` | *(native bf16 line inside `fp16_bf16_pure_vs_cutlass_ref.sh`)* | — none |
| FP32_16 packed, **pure_half** tier (fp16 compute, decodes `wh` only) | `verify_mm_fp32_16_pure_half_gemm.sh` | `wide_bigreg_sq_fp32_16_vs_cutlass_ref.sh`, `fp16_bf16_pure_vs_cutlass_ref.sh` / `fp16_bf16_pure_rect_vs_cutlass_ref.sh` | `verify_mv_fp32_16_gemv.sh` |
| FP32_16 packed, **accurate/reconstruction** tier (fp32 compute, decodes `wh`+`wr`, exact) | `verify_mm_fp32_16_gemm.sh` | `fp32_recon_vs_cutlass_ref.sh` / `fp32_recon_rect_vs_cutlass_ref.sh` | — none |
| FP32_BF16 packed, **pure_bf16** tier (bf16 compute, decodes `wh` only) | `verify_bf16_wide_bigreg_sq_deep4.sh` | `fp16_bf16_pure_vs_cutlass_ref.sh` / `fp16_bf16_pure_rect_vs_cutlass_ref.sh` | — none |
| FP32_BF16 packed, **accurate/reconstruction** tier (fp32 compute, decodes `wh`+`wr`, exact) | `verify_mm_fp32_16_gemm.sh` (same executable, also covers FP32_BF16) | `fp32_recon_vs_cutlass_ref.sh` / `fp32_recon_rect_vs_cutlass_ref.sh` | — none |
| Full project benchmark suite (google-benchmark) | — | `run_cuda_benchmarks.sh` | — |

"— none" means no dedicated script exists for that cell (not that the
kernel doesn't exist).

## GEMM correctness + performance (`verify_*_gemm.sh`)

Each of these builds and runs one `benchmarks/kernels/cuda/matrix/verify_*.cu`
executable that, by default, does BOTH a correctness section (compares
against a CPU reference, including odd/non-tile-aligned sizes) and a
performance section (throughput sweep vs cuBLAS at large square sizes).
All generate test data directly on the GPU (no host-side matrix
buildup), so they scale to large shapes even under a constrained host
memory cgroup, and gracefully skip individual sizes that don't fit in
whatever device memory happens to be free. Pass executable flags after
`--`:

- `-- --skip-correctness` — performance sweep only
- `-- --skip-performance` — correctness only
- `-- --only-shape N` — isolate one shape (for external profiling with `ncu`/`nsys`)

| Script | Precision | Kernels covered |
|---|---|---|
| `verify_mm_fp32_gemm.sh` | fp32 | tomos's plain fp32 GEMM kernel(s) vs cuBLAS Sgemm |
| `verify_mm_fp16_gemm.sh` | fp16 | `gemm_generic_kernel<half>`, `gemm_fp16_tiled`, `gemm_fp16_wmma` vs cuBLAS Hgemm |
| `verify_mm_fp32_16_pure_half_gemm.sh` | FP32_16, pure_half (fp16 compute) | `mm_fp32_16_pure_half`, `_tiled`, `_wmma` (decode `wh` only) vs cuBLAS Hgemm |
| `verify_mm_fp32_16_gemm.sh` | FP32_16 **and** FP32_BF16, accurate/reconstruction (fp32 compute) | `mm_fp32_16_accurate` vs cuBLAS Sgemm (on a materialized fp32 `A`); also covers `mm_fp32_bf16_accurate`/`_tiled`/`_tiled_cpasync_dualbuf` correctness in the same executable |
| `verify_bf16_wide_bigreg_sq_deep4.sh` | bf16 (plain) **and** FP32_BF16 pure_bf16 (bf16 compute) | `gemm_bf16_mma_ptx_wide_bigreg_sq_deep4` (native) vs real CUTLASS + cuBLAS; `mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4` (decodes `wh` only) throughput, compared against the native kernel's own number |

## GEMV correctness + performance

| Script | Precision | Kernels covered |
|---|---|---|
| `verify_mv_fp32_16_gemv.sh` | FP32_16, pure_half (fp16 compute) | `mv_fp32_16_pure_half_*` (narrow and wide-shape variants) vs cuBLAS Hgemv(n=1); also exercises boundary/tail/transpose edge cases |

No dedicated GEMV script exists yet for plain fp32/fp16/bf16 or for
FP32_BF16.

## Focused throughput-only comparisons vs real CUTLASS / cuBLAS

These are narrower than the `verify_*` scripts above — they compare a
specific handful of tomos kernels (not the whole family) against real
CUTLASS `device::Gemm` and cuBLAS in one process, so the numbers share
the same GPU clock/thermal state. None of them do correctness checking
(that's covered by the `verify_*` scripts above) — throughput only.
CUTLASS is fetched automatically (shallow clone) into
`build/cutlass-ref` on first run (override with the `CUTLASS_DIR` env
var, or persistently via a gitignored `.configure.sh` at the repo root —
see `.configure.sh.example`).

| Script | Precision / tier | Sweep |
|---|---|---|
| `cutlass_fp16_gemm_ref.sh` | fp16 (plain) | Real CUTLASS fp16 tensorop + cuBLAS Hgemm only (no tomos kernel — an external reference point to diff against `verify_mm_fp16_gemm.sh`'s own numbers). `N=1024/2048/4096/8192/16384` |
| `wide_bigreg_family_vs_cutlass_ref.sh` | fp16 (plain) | 3 tomos kernels (`gemm_fp16_mma_ptx_wide_bigreg_sq`, `_wide_bigreg`, `_cutlass_config`) vs CUTLASS vs cuBLAS. `N=1024/2048/4096/8192/16384` |
| `wide_bigreg_sq_fp32_16_vs_cutlass_ref.sh` | FP32_16, pure_half (fp16 compute) | `mm_fp32_16_pure_half_wide_bigreg_sq` + its 2 best wmma-tuned siblings vs CUTLASS vs cuBLAS. `N=1024..65536` |
| `fp16_bf16_pure_vs_cutlass_ref.sh` | FP32_16 pure_half **and** FP32_BF16 pure_bf16 (16-bit compute) | `mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty` (fp16 compute) + `mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4` (bf16 compute), each vs its own native kernel, CUTLASS, and cuBLAS. Square sweep `N=1024..65536` **then** a rectangular sweep (`M=N=44000`, `K=1024..32768`). Writes timestamped `.txt`+`.csv` logs; auto-targets the real A100 via `srun` if not already inside a job allocation. |
| `fp16_bf16_pure_rect_vs_cutlass_ref.sh` | same as above | Same binary, but **only** the rectangular sweep (`M=N=44000`, `K=1024..32768`) — skips the square sweep. Pass a single `K` to run just that one shape. |
| `fp32_recon_vs_cutlass_ref.sh` | FP32_16 **and** FP32_BF16, accurate/reconstruction (fp32 compute) | `mm_fp32_16_tiled_cpasync_dualbuf` + `mm_fp32_bf16_tiled_cpasync_dualbuf` (both fp32 compute, exact) vs native fp32 (`gemm_fp32_tiled_cpasync_dualbuf`), CUTLASS SIMT fp32, and cuBLAS Sgemm. Square sweep `N=1024..54000` **then** the same rectangular sweep as above. Writes `.txt`+`.csv` logs; auto-targets the real A100. |
| `fp32_recon_rect_vs_cutlass_ref.sh` | same as above | Same binary, **only** the rectangular sweep. Pass a single `K` to run just that one shape. |

See `cutlass_ref_benchmarks.txt` in this directory for a deeper writeup
of the four SLURM-auto-targeting, logging scripts (`fp32_recon_vs_
cutlass_ref.sh`, `fp32_recon_rect_vs_cutlass_ref.sh`, `fp16_bf16_pure_
vs_cutlass_ref.sh`, `fp16_bf16_pure_rect_vs_cutlass_ref.sh`) — CSV
column layouts, output paths, and the SLURM re-exec mechanism.

## Everything else

| Script | What it does |
|---|---|
| `run_cuda_benchmarks.sh` | Builds and runs the project's google-benchmark-based executables (`bench_mv_cuda_kernels`, `bench_mm_cuda_kernels`, `bench_fp32_16_cuda_kernels`, `bench_lsqr_cuda`) — each writes its own `.tex`/`.csv`/plot script to an `output/` dir next to itself under `build/`. Currently only `bench_fp32_16_cuda_kernels` is enabled in the script's own executable list; the others are present but commented out. |

## Known result: FP32_16 vs FP32_BF16 decode-cost gap (fp32-compute tier)

`mm_fp32_16_tiled_cpasync_dualbuf` and `mm_fp32_bf16_tiled_cpasync_dualbuf` are
architecturally identical (same tile/warp/lane geometry, same ~127-register
footprint) except for their decode step — `fp32_bf16_combine` is a trivial
unconditional `(wh<<16)|wr`, while `fp32_16_combine` dispatches across 3
exponent modes and unpacks 5 bitfields, plus an `is_carry` rounding
correction. Measured on the A100 via `fp32_recon_vs_cutlass_ref.sh`
(`fp32_recon_vs_cutlass_ref_20260823_195101.txt/.csv`), this costs FP32_16 a
consistent throughput deficit:

| N | FP32_BF16 (GFlop/s) | FP32_16, rounding | FP32_16, truncated | rounding vs BF16 |
|---|---|---|---|---|
| 1024 | 6412.3 | 5642.8 | 5758.2 | −12.0% |
| 8192 | 17215.2 | 16160.5 | 16312.6 | −6.1% |
| 32768 | 17399.8 | 16429.3 | 16555.6 | −5.6% |
| 49152 | 17411.0 | 16458.6 | 16575.5 | −5.5% |

`mm_fp32_16_tiled_cpasync_dualbuf_truncated` (plain truncation, no rounding,
no `is_carry` correction, decoded via `fp32_16_combine_truncated`) exists
specifically to separate the two causes. Static SASS analysis
(`cuobjdump --dump-sass`, no GPU execution) attributes the gap:

| Category | FP32_BF16 | FP32_16 truncated | FP32_16 rounding |
|---|---|---|---|
| Total instructions | 2728 | 3400 | 3592 |
| BRA | 20 | 88 | 88 |
| SEL | 0 | 0 | 48 |
| ISETP | 187 | 235 | 299 |
| LOP3 | 11 | 256 | 273 |

**~4.5–5 of the ~5.5–6 percentage points** are structural — the 3-way
exponent-mode dispatch and 5-bitfield unpack, not reducible without changing
`FP32_16`'s bit layout. Only **~0.7–1.5 percentage points** come from the
rounding correction itself (which compiles entirely to `SEL`/`ISETP`, never
`BRA` — confirmed branch-free). Two decode-only "optimizations" (a
branchless swap, a combined-integer borrow-propagation decode) were tried
and reverted — both measured *slower* despite fewer static SASS
instructions; see the comments in `fp32_16_pack.hpp` and the full writeup in
`fp32_16_vs_fp32_bf16_decode_investigation.tex` (repo root) for the complete
investigation.

## Picking a script — by task

- **"I changed a kernel, did I break it?"** → the matching `verify_*_gemm.sh` or `verify_mv_fp32_16_gemv.sh` (run without `--skip-correctness`).
- **"How fast is my kernel vs cuBLAS, at realistic sizes?"** → the matching `verify_*_gemm.sh` (or `verify_bf16_wide_bigreg_sq_deep4.sh`), performance section.
- **"How does tomos compare to a real CUTLASS implementation, not just cuBLAS?"** → one of the `*_vs_cutlass_ref.sh` scripts above, matching your precision/tier.
- **"I need saved results (txt+csv) for later analysis, or a rectangular (non-square) shape sweep."** → `fp32_recon_vs_cutlass_ref.sh` / `fp32_recon_rect_vs_cutlass_ref.sh` (fp32 compute) or `fp16_bf16_pure_vs_cutlass_ref.sh` / `fp16_bf16_pure_rect_vs_cutlass_ref.sh` (16-bit compute) — these are the only scripts with logging and rectangular-shape support.
- **"I want to profile one specific kernel launch with ncu/nsys."** → any `verify_*_gemm.sh` or `verify_mv_fp32_16_gemv.sh` with `-- --only-shape N`.

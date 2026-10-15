#!/bin/bash
# Builds benchmarks_gemm_fp32.cu (same binary as scripts/fp32_recon_
# vs_cutlass_ref.sh) and runs two custom sweeps -- comparing the "full
# fp32 reconstruction" tier of both packed-precision families -- mm_
# fp32_16_tiled_cpasync_dualbuf and mm_fp32_bf16_tiled_cpasync_dualbuf --
# against this codebase's own best native (unpacked) fp32 GEMM (gemm_
# fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact), real
# CUTLASS's fp32 SIMT device::Gemm, and cuBLAS's Sgemm, all in one run
# so the numbers share the same GPU clock/thermal/contention state
# (benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu).
#
# Unlike fp32_recon_vs_cutlass_ref.sh's own default sweeps (square N=
# 1024..54000, then a rectangular K=1024..32768 sweep at a fixed M=N=
# 44000), this script drives the SAME binary's single-shape modes (plain
# "<N>" for square, "--rect <K> <MN>" for rectangular) in loops over its
# own custom lists:
#   - Square (M=K=N): 1024/1536/2048/3072/4096/6144/8192/12288/16384/
#     24576/32768/49152.
#   - Rectangular: output held fixed at M=N=49152 (the top of the square
#     list above), inner/contraction dimension K swept across 1024/1536/
#     2048/3072/4096/6144/8192/12288/16384/24576/32768 (the same list
#     minus 49152, since that value is now the fixed outer dimension
#     instead of a K value).
# All shapes' output (both sweeps) is combined into one timestamped .txt
# log and parsed into one .csv, same as the other CUTLASS-comparison
# scripts.
#
# Usage:
#   scripts/benchmarks_gemm_format_32_16_compute_32.sh
#       Runs both sweeps above. Takes no arguments.
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/benchmarks_gemm_format_32_16_compute_32.sh
#
# Rebuilds tomos_cuda first (so this always reflects the current state
# of the kernels, not a stale libtomos_cuda.so) and links the standalone
# nvcc build against it directly -- CUTLASS itself is still NOT a
# dependency of tomos's own CMake build, only of this one comparison
# binary.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is
# only reachable inside an active job allocation. If SLURM_JOB_ID isn't
# already set, this script finds the caller's running job on an a100*
# partition and re-execs itself inside it via `srun --overlap`, so both
# the build and the run always land on the actual target GPU.
#
# All shapes' full console output (both sweeps) is saved to one
# timestamped .txt log, and a per-shape summary (ms + GFlop/s for each
# method) is parsed out into one .csv, both under build/fp32_recon_vs_
# cutlass_ref_output/ (same output directory as fp32_recon_vs_cutlass_
# ref.sh -- filenames are timestamped so runs from either script never
# collide).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu"
OUT="$BUILD_DIR/fp32_recon_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/benchmarks/gemm/fp32_compute/data"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@"
fi

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: fp32_recon_vs_cutlass_ref"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    -I"$REPO_ROOT/src/kernels/cuda/matrix" \
    -I"$REPO_ROOT/src/kernels/cpu/matrix" \
    -I"$REPO_ROOT/src/core/matrix" \
    -I"$REPO_ROOT/include/core" \
    -I"$REPO_ROOT/src/core/context" \
    -I"$REPO_ROOT/src/core/memory" \
    -I"$REPO_ROOT/src/core/blas" \
    -I"$REPO_ROOT/src/kernels/cuda/context" \
    -I"$REPO_ROOT/src/macros" \
    -I"$REPO_ROOT/include/io" \
    -I"$REPO_ROOT/include/cpu" \
    -I"$REPO_ROOT/src/kernels/cpu/memory" \
    -I"$REPO_ROOT/src/kernels/cpu/sketch" \
    -I"$REPO_ROOT/include/cuda" \
    -I"$REPO_ROOT/src/kernels/cuda/constants" \
    -I"$REPO_ROOT/src/kernels/cuda/memory" \
    -I"$REPO_ROOT/src/kernels/cuda/sketch" \
    -I"$REPO_ROOT/src/kernels/cuda/precond" \
    -I"$REPO_ROOT/src/kernels/cuda/blas" \
    -I"$REPO_ROOT/src/core/solver" \
    -I"$REPO_ROOT/src/dispatch" \
    -I"$REPO_ROOT/src/io" \
    -I"$BUILD_DIR/generated" \
    -I"$BUILD_DIR" \
    -I"$REPO_ROOT/include" \
    -I"$REPO_ROOT/src/core/detail" \
    "$SRC" -lcublas \
    -L"$TOMOS_CUDA_DIR" -ltomos_cuda -Xlinker -rpath -Xlinker "$TOMOS_CUDA_DIR" \
    -o "$OUT"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${TIMESTAMP}.csv"

# Square sweep (M=K=N).
SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152)

# Rectangular sweep: output held fixed at M=N=49152 (the top of the
# square list above), inner/contraction dimension K stepped through the
# same list minus 49152 (since that value is now the fixed outer
# dimension, not a K value).
RECT_OUTER=49152
RECT_KS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768)

: > "$TXT_LOG"
for n in "${SQUARE_NS[@]}"; do
    echo
    echo "==> Running fp32_recon_vs_cutlass_ref $n"
    "$OUT" "$n" | tee -a "$TXT_LOG"
done
for k in "${RECT_KS[@]}"; do
    echo
    echo "==> Running fp32_recon_vs_cutlass_ref --rect $k $RECT_OUTER"
    "$OUT" --rect "$k" "$RECT_OUTER" | tee -a "$TXT_LOG"
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    # fp32_16 method family, ranked BEST-TO-WORST by measured throughput
    # (N=8192, scripts/fp32_new_kernels_vs_cutlass_ref.sh 8192 and this
    # codebase's own prior measurements where noted; "?" = not freshly
    # re-measured in the warptile/snake/midpoint_exact investigation,
    # positioned by architecture family only, no specific number implied):
    #   1. mm_fp32_16_gemm_materialize_midpoint_exact -- 18450.2 GFlop/s,
    #      0.961x vs CUTLASS. NEW BEST. Decodes A to a plain fp32 scratch
    #      buffer up front (bounded-size row chunks), then calls
    #      midpoint_exact (this codebase's current best native fp32
    #      kernel) as a black box.
    #   2. mm_fp32_16_gemm_materialize -- 17586.9 GFlop/s. Previous best,
    #      superseded by #1: same materialize strategy, but its second-
    #      stage native GEMM call was gemm_fp32_tiled_cpasync_dualbuf
    #      (the OLD native fp32 GEMM) instead of midpoint_exact.
    #   3. mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact
    #      -- 16006.1 GFlop/s, 0.834x vs CUTLASS. The GENUINELY fused
    #      counterpart to #1/#2's materialize strategy: ports midpoint_
    #      exact's entire tiling structure into the fp32_16 file and
    #      decodes wh/wr straight into the tile pipeline's shared-memory
    #      A buffer (no separate scratch buffer or decode pass).
    #      Architecturally the "purer" approach, and the best kernel here
    #      if materialize-strategy kernels are excluded (it beats dualbuf,
    #      #5, by ~3%) -- but still measured 13% SLOWER than materialize_
    #      midpoint_exact -- midpoint_exact's tiling is already tightly
    #      register-bound (128 registers, 0 spill, no margin), and the
    #      extra decode instructions (two ushort4 loads + four fp32_16_
    #      combine calls) compete for the same tight budget its
    #      interleaved i==4 prefetch checkpoint depends on, disturbing a
    #      schedule that was clean with plain float loads. Not wired into
    #      this script (see below) since it lost head-to-head against
    #      materialize_midpoint_exact, but kept in the codebase and
    #      documented here for completeness -- see fp32_16_cuda_
    #      kernels.cu/.cuh.
    #   4. ? mm_fp32_16_tiled_cpasync_dualbuf_adaptive / _adaptive_warp /
    #      _multin -- same fused-into-shared-memory family as #5 below
    #      (minor per-tile decode-cost tweaks on top of it); not
    #      re-measured this pass -- see their own kernel comments in
    #      fp32_16_cuda_kernels.cu for original measurements.
    #   5. mm_fp32_16_tiled_cpasync_dualbuf -- 15529.0 GFlop/s. The
    #      original FUSED reference: decodes each BK=16 tile's wh/wr
    #      straight into shared memory (via fp32_16_combine, in
    #      transform_tile) as part of the same kernel that does the
    #      tiled compute -- no separate decode pass or scratch buffer,
    #      but every block redundantly re-decodes the same A elements
    #      once per N-tile it covers. Kept active below as the fixed
    #      "how much does fusing decode cost vs materializing separately"
    #      baseline.
    #   6. ? mm_fp32_16_tiled_cpasync_dualbuf_truncated -- not re-measured
    #      this pass (also numerically lossy vs the others -- never-
    #      rounded decode, see fp32_16_pack.hpp).
    #
    # dualbuf (#5, fused-into-shared-memory decode) vs gemm_materialize
    # (#2, decode-then-black-box-native-GEMM) vs the midpoint_exact
    # variants (#1 materialize-based, #3 genuinely fused) are three
    # genuinely different architectures, not just parameter tweaks of
    # each other -- see fp32_16_cuda_kernels.cu's own comments on
    # mm_fp32_16_tiled_cpasync_dualbuf_kernel, mm_fp32_16_gemm_
    # materialize, and mm_fp32_16_gemm_materialize_midpoint_exact for the
    # full architectural writeup.
    #
    # The truncated/adaptive_warp/adaptive/multin markers MUST come
    # before the plain dualbuf one, AND adaptive_warp MUST come before
    # plain adaptive -- "mm_fp32_16_tiled_cpasync_dualbuf" is a prefix
    # of "..._truncated", "..._adaptive", and "..._multin", and
    # "..._adaptive" is itself a prefix of "..._adaptive_warp"; this
    # loop takes the first match, so checking the longest/most specific
    # strings first avoids misfiling. "mm_fp32_16_gemm_materialize_
    # midpoint_exact" MUST come before the plain "mm_fp32_16_gemm_
    # materialize" marker below (now commented out, not deleted -- see
    # #2 above) for the same reason: it's a superstring/prefix relation.
    # Order between truncated, multin, and the two adaptive markers
    # doesn't matter among themselves -- none is a substring of another.
    ("mm_fp32_16_tiled_cpasync_dualbuf_truncated", "fp32_16_dualbuf_trunc"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp", "fp32_16_dualbuf_adaptive_warp"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive", "fp32_16_dualbuf_adaptive"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_multin", "fp32_16_dualbuf_multin"),
    # UNCONDITIONAL fast_normal decode siblings of dualbuf/materialize_
    # midpoint_exact (see fp32_16_cuda_kernels.cu's own comments) --
    # "mm_fp32_16_tiled_cpasync_dualbuf" is a literal prefix of
    # "..._fast_normal", and "mm_fp32_16_gemm_materialize_midpoint_exact"
    # is a literal prefix of "..._fast_normal" too, so both MUST come
    # before their own plain markers below, same superstring-first rule
    # as every other entry in this list.
    ("mm_fp32_16_tiled_cpasync_dualbuf_fast_normal", "fp32_16_dualbuf_fast"),
    ("mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal", "fp32_16_gemm_materialize_midpoint_exact_fast"),
    ("mm_fp32_16_gemm_materialize_midpoint_exact", "fp32_16_gemm_materialize_midpoint_exact"),
    # Switched out in favor of mm_fp32_16_gemm_materialize_midpoint_exact
    # above (measured 4.9% faster, same materialize strategy just
    # pointed at the newer native kernel) -- commented out, not deleted:
    #     ("mm_fp32_16_gemm_materialize", "fp32_16_gemm_materialize"),
    # The genuinely-fused midpoint_exact-based kernel (#3 in the ranking
    # above) -- best fp32_16 kernel once materialize-strategy kernels are
    # excluded (beats dualbuf below by ~3%), though still behind both
    # materialize variants overall. Still captured here even though it's
    # not this script's plotted "TOMOS FP32 / FP16" series (see
    # plot_gemm_format_32_16_compute_32.m) -- kept in the CSV as raw data
    # either way. Its truncated sibling MUST be checked first below,
    # since "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_
    # exact" is itself a prefix of "..._truncated" -- same superstring
    # ordering rule as the other markers here.
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated", "fp32_16_fused_midpoint_exact_trunc"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_16_fused_midpoint_exact"),
    ("mm_fp32_16_tiled_cpasync_dualbuf", "fp32_16_dualbuf"),
    # The fp32_bf16 counterpart to the fp32_16 fused midpoint_exact
    # kernel above (same tiling, decode via fp32_bf16_combine instead of
    # fp32_16_combine) -- unlike fp32_16, where materialize wins, this
    # one beats mm_fp32_bf16_tiled_cpasync_dualbuf below outright at
    # every measured size (~7-14%, see plot_gemm_format_32_16_compute_
    # 32.m's own comment on the TOMOS FP32 / BF16 series for the full
    # numbers), since fp32_bf16_combine is unconditional/branchless and
    # compiles with 0 spill. No prefix relation with the dualbuf marker
    # below, so ordering between them doesn't matter.
    # "_rounded" MUST come before the bare fused marker below -- it's a
    # superstring of it.
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded", "fp32_bf16_fused_rounded"),
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_bf16_fused_midpoint_exact"),
    ("mm_fp32_bf16_gemm_materialize_rounded", "fp32_bf16_materialize_rounded"),
    ("mm_fp32_bf16_tiled_cpasync_dualbuf", "fp32_bf16_dualbuf"),
    ("gemm_fp32_tiled_cpasync_dualbuf_regpipe", "native_regpipe"),
    # "native" is gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
    # midpoint_exact (promoted from gemm_fp32_tiled_syncpipe3, itself
    # promoted from gemm_fp32_tiled_cpasync_dualbuf -- see fp32_recon_vs_
    # cutlass_ref.cu's own file-level comment for why). The printed line
    # is "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_
    # exact (native) = ...ms (...GFlop/s)"; matching on the full string
    # including " (native)" (not just the bare kernel name) is required
    # here because the bare name is also a prefix of the plain
    # (non-native) timing line for the same kernel further down the same
    # run's output.
    ("gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native)", "native"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemm)", "cublas"),
]

FIELDS = [
    "M", "K", "N",
    "fp32_16_dualbuf_ms", "fp32_16_dualbuf_gflops",
    "fp32_16_dualbuf_trunc_ms", "fp32_16_dualbuf_trunc_gflops",
    "fp32_16_dualbuf_adaptive_ms", "fp32_16_dualbuf_adaptive_gflops",
    "fp32_16_dualbuf_adaptive_warp_ms", "fp32_16_dualbuf_adaptive_warp_gflops",
    "fp32_16_dualbuf_multin_ms", "fp32_16_dualbuf_multin_gflops",
    "fp32_16_dualbuf_fast_ms", "fp32_16_dualbuf_fast_gflops",
    "fp32_16_gemm_materialize_midpoint_exact_ms", "fp32_16_gemm_materialize_midpoint_exact_gflops",
    "fp32_16_gemm_materialize_midpoint_exact_fast_ms", "fp32_16_gemm_materialize_midpoint_exact_fast_gflops",
    # Switched out in favor of the _midpoint_exact fields above --
    # commented out, not deleted:
    #     "fp32_16_gemm_materialize_ms", "fp32_16_gemm_materialize_gflops",
    "fp32_16_fused_midpoint_exact_ms", "fp32_16_fused_midpoint_exact_gflops",
    "fp32_16_fused_midpoint_exact_trunc_ms", "fp32_16_fused_midpoint_exact_trunc_gflops",
    "fp32_bf16_dualbuf_ms", "fp32_bf16_dualbuf_gflops",
    "fp32_bf16_fused_midpoint_exact_ms", "fp32_bf16_fused_midpoint_exact_gflops",
    "fp32_bf16_fused_rounded_ms", "fp32_bf16_fused_rounded_gflops",
    "fp32_bf16_materialize_rounded_ms", "fp32_bf16_materialize_rounded_gflops",
    "native_ms", "native_gflops",
    "native_regpipe_ms", "native_regpipe_gflops",
    "cutlass_ms", "cutlass_gflops",
    "cublas_ms", "cublas_gflops",
]

rows = []
row = None


def flush():
    if row is not None:
        rows.append(row)


with open(txt_path) as f:
    for line in f:
        line = line.rstrip("\n")
        m = shape_re.match(line)
        if m:
            flush()
            row = {"M": m.group(1), "K": m.group(2), "N": m.group(3)}
            continue
        if row is None:
            continue

        if "native/cutlass/cublas (A)" in line and "SKIPPED" in line:
            for key in ("native", "cutlass", "cublas"):
                row[f"{key}_ms"] = "SKIPPED"
                row[f"{key}_gflops"] = "SKIPPED"
            continue

        for marker, key in METHOD_MARKERS:
            if marker not in line:
                continue
            if "SKIPPED" in line:
                row[f"{key}_ms"] = "SKIPPED"
                row[f"{key}_gflops"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_ms"] = vm.group(1)
                    row[f"{key}_gflops"] = vm.group(2)
            break
    flush()

with open(csv_path, "w") as out:
    out.write(",".join(FIELDS) + "\n")
    for r in rows:
        out.write(",".join(r.get(field, "") for field in FIELDS) + "\n")
PYEOF

echo
echo "==> Logs written:"
echo "      $TXT_LOG"
echo "      $CSV_LOG"

echo
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    # Both plot variants read the SAME CSV: plot_gemm_format_32_16_
    # compute_32 (unchanged, no fast_normal series) and its _fast sibling
    # (adds "TOMOS FP32 / FP16 (fast)" / "(workspace, fast)") -- see
    # plot_gemm_format_32_16_compute_32_fast.m's own header for why this
    # is a separate function rather than a flag on the original.
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$CSV_LOG'); plot_gemm_format_32_16_compute_32_fast('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$CSV_LOG'); plot_gemm_format_32_16_compute_32_fast('$CSV_LOG')\"" >&2
fi

#!/bin/bash
# Sibling of scripts/benchmarks_gemm_format_32_16_compute_32.sh that runs
# the SAME binary (benchmarks/kernels/cuda/matrix/fp32_recon_vs_cutlass_
# ref.cu) and the SAME square+rect sweeps (1024/1536/2048/3072/4096/6144/
# 8192/12288/16384/24576/32768/49152 square, then a rectangular sweep
# with the output held fixed at M=N=49152 and K swept across the same
# list minus 49152), but with every series' own A/B (native fp32 B,
# fp32_16-encoded A, fp32_bf16-encoded A, and the native/cutlass/cublas
# leg's plain fp32 A) drawn from a generator with a TUNABLE fraction of
# entries forced into the fp32 subnormal band (0 < |x| < 2^-126), instead
# of the .cu's own plain uniform-on-[-1,1] pseudo_rand.
#
# Exists for the same reason scripts/fp16_bf16_pure_vs_cutlass_ref_
# subnormal.sh does for the 16-bit-compute tier: the default uniform
# generator realizes essentially no subnormals (~0.006% of entries,
# matching a continuous uniform's own share of that tiny band) -- every
# existing GEMM performance plot for this "32-bit compute" (full fp32
# reconstruction) tier, including plot_gemm_format_32_16_compute_32_
# square.png/plot_gemm_format_32_16_compute_32_rect.png, is measuring
# decode/compute speed on data that essentially never exercises any
# subnormal-handling path -- fp32_16_combine's/fp32_bf16_combine's own
# exponent-mode dispatch, AND the native fp32 FFMA/CUTLASS/cuBLAS legs'
# own hardware handling of subnormal operands. This script measures every
# method at a realistic-or-worse subnormal fraction instead, to see
# whether that changes this tier's relative throughput.
#
# Unlike scripts/fp16_bf16_pure_vs_cutlass_ref_subnormal.sh, which
# deliberately leaves its bf16 group's inputs unchanged (that tier's bf16
# leg computes NATIVELY in bf16, whose own subnormal boundary is
# astronomically smaller than fp16's), every series in THIS file
# ultimately reconstructs to (or, for the native/cutlass/cublas leg,
# starts as) genuine float -- so one fp32-band density is applied
# uniformly to every generator, fp32_16- and fp32_bf16-encoded alike, no
# group excluded. See benchmarks_gemm_fp32.cu's own run_shape/
# pseudo_rand_density comments for the full derivation.
#
# Usage:
#   scripts/fp32_recon_vs_cutlass_ref_subnormal.sh [--density PCT] [N | --rect [K]]
#   --density PCT   Percent of every generator's own entries forced into
#                    the fp32 subnormal band. Default: 10 (i.e. 10%).
#   With no shape argument, sweeps the same square+rect lists as scripts/
#   benchmarks_gemm_format_32_16_compute_32.sh's own default. A single N
#   runs just that square shape; --rect [K] runs just that rectangular
#   shape (M=N=49152 fixed, K given or swept over the full rect list).
#
# Examples:
#   scripts/fp32_recon_vs_cutlass_ref_subnormal.sh
#       Full square+rect sweep at the default 10% density.
#   scripts/fp32_recon_vs_cutlass_ref_subnormal.sh --density 30 8192
#       Just N=8192, square, at 30% density.
#
# CUTLASS is fetched (shallow clone, header-only) into CUTLASS_DIR on
# first run and reused after that, same as the other *_vs_cutlass_ref.sh
# scripts. Override with the CUTLASS_DIR env var if you already have one:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp32_recon_vs_cutlass_ref_subnormal.sh
#
# Rebuilds tomos_cuda first (so this reflects the current kernel state)
# and links the standalone nvcc build directly against it -- same
# discipline as every other *_vs_cutlass_ref.sh script.
#
# On Gilbreth the shell this runs in is usually the login node (no A100
# there) -- if SLURM_JOB_ID isn't already set, this re-execs itself
# inside the caller's own running a100* job via `srun --overlap`.
#
# Writes, under build/fp32_recon_vs_cutlass_ref_subnormal_output/:
#   fp32_recon_vs_cutlass_ref_subnormal_<density>pct_<TIMESTAMP>.txt
#   fp32_recon_vs_cutlass_ref_subnormal_<density>pct_<TIMESTAMP>.csv
# and, via the EXISTING plot_gemm_format_32_16_compute_32.m/plot_gemm_
# format_32_16_compute_32_fast.m (unmodified -- same CSV schema, same
# methods, just different underlying data), under
# build/benchmarks/output/plots/:
#   plot_gemm_format_32_16_compute_32_square_subnormal<density>pct.png / .fig
#   plot_gemm_format_32_16_compute_32_rect_subnormal<density>pct.png   / .fig
# -- distinct names from the uniform-data plots (plot_gemm_format_32_16_
# compute_32_square.png etc.), so neither overwrites the other.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu"
OUT="$BUILD_DIR/fp32_recon_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp32_recon_vs_cutlass_ref_subnormal_output"

DENSITY="10"
SHAPE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --density) DENSITY="$2"; shift 2 ;;
        *) SHAPE_ARGS+=("$1"); shift 1 ;;
    esac
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" --density "$DENSITY" "${SHAPE_ARGS[@]}"
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
DENSITY_TAG="subnormal${DENSITY}pct"
TXT_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${DENSITY_TAG}_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${DENSITY_TAG}_${TIMESTAMP}.csv"

# Same custom square+rect lists as scripts/benchmarks_gemm_format_32_16_
# compute_32.sh -- see that script's own header for why these (not
# fp32_recon_vs_cutlass_ref.sh's own default 1024..54000 square list)
# are the shapes plot_gemm_format_32_16_compute_32.m's "_square"/"_rect"
# plots are built from.
SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152)
RECT_OUTER=49152
RECT_KS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768)

: > "$TXT_LOG"
if [ ${#SHAPE_ARGS[@]} -gt 0 ] && [ "${SHAPE_ARGS[0]}" = "--rect" ]; then
    if [ ${#SHAPE_ARGS[@]} -gt 1 ]; then
        echo
        echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $DENSITY --rect ${SHAPE_ARGS[1]} $RECT_OUTER"
        "$OUT" --subnormal-density "$DENSITY" --rect "${SHAPE_ARGS[1]}" "$RECT_OUTER" | tee -a "$TXT_LOG"
    else
        for k in "${RECT_KS[@]}"; do
            echo
            echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $DENSITY --rect $k $RECT_OUTER"
            "$OUT" --subnormal-density "$DENSITY" --rect "$k" "$RECT_OUTER" | tee -a "$TXT_LOG"
        done
    fi
elif [ ${#SHAPE_ARGS[@]} -gt 0 ]; then
    echo
    echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $DENSITY ${SHAPE_ARGS[0]}"
    "$OUT" --subnormal-density "$DENSITY" "${SHAPE_ARGS[0]}" | tee -a "$TXT_LOG"
else
    for n in "${SQUARE_NS[@]}"; do
        echo
        echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $DENSITY $n"
        "$OUT" --subnormal-density "$DENSITY" "$n" | tee -a "$TXT_LOG"
    done
    for k in "${RECT_KS[@]}"; do
        echo
        echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $DENSITY --rect $k $RECT_OUTER"
        "$OUT" --subnormal-density "$DENSITY" --rect "$k" "$RECT_OUTER" | tee -a "$TXT_LOG"
    done
fi

echo
echo "==> Writing CSV summary: $CSV_LOG"
# Identical parser to scripts/benchmarks_gemm_format_32_16_compute_32.sh's
# own embedded one -- the .cu's shape-line format (M=.. K=.. N=..) and
# method marker lines are completely unchanged by --subnormal-density
# (that flag only prints one extra banner line near the top of the log,
# outside any shape block, which this parser's shape_re/marker matching
# simply never touches).
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    ("mm_fp32_16_tiled_cpasync_dualbuf_truncated", "fp32_16_dualbuf_trunc"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp", "fp32_16_dualbuf_adaptive_warp"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive", "fp32_16_dualbuf_adaptive"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_multin", "fp32_16_dualbuf_multin"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_fast_normal", "fp32_16_dualbuf_fast"),
    ("mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal", "fp32_16_gemm_materialize_midpoint_exact_fast"),
    ("mm_fp32_16_gemm_materialize_midpoint_exact", "fp32_16_gemm_materialize_midpoint_exact"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated", "fp32_16_fused_midpoint_exact_trunc"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_16_fused_midpoint_exact"),
    ("mm_fp32_16_tiled_cpasync_dualbuf", "fp32_16_dualbuf"),
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded", "fp32_bf16_fused_rounded"),
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_bf16_fused_midpoint_exact"),
    ("mm_fp32_bf16_gemm_materialize_rounded", "fp32_bf16_materialize_rounded"),
    ("mm_fp32_bf16_tiled_cpasync_dualbuf", "fp32_bf16_dualbuf"),
    ("gemm_fp32_tiled_cpasync_dualbuf_regpipe", "native_regpipe"),
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
echo "==> Plotting (tag: $DENSITY_TAG): $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$CSV_LOG', '$DENSITY_TAG'); plot_gemm_format_32_16_compute_32_fast('$CSV_LOG', '$DENSITY_TAG');"
if command -v matlab >/dev/null 2>&1; then
    matlab -nodisplay -nosplash -nodesktop -r "$MATLAB_CALL exit;"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -nodisplay -nosplash -nodesktop -r \"$MATLAB_CALL exit;\"" >&2
fi

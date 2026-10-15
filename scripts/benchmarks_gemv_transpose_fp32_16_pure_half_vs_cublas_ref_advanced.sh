#!/bin/bash
# "_advanced" variant of scripts/benchmarks_gemv_transpose_fp32_16_pure_
# half_vs_cublas_ref.sh -- builds benchmarks/kernels/cuda/matrix/mv_
# fp32_16_pure_half_transpose_vs_cublas_ref_advanced.cu (byte-for-byte
# identical to its own sibling .cu except for a caller-controlled
# warmup/iteration count instead of a fixed, shape-size-dependent `reps`
# heuristic -- see that file's own header) instead of the plain mv_
# fp32_16_pure_half_transpose_vs_cublas_ref.cu, and runs the EXACT SAME
# two sweeps (square + tall-and-thin, including the SAME --square-only/
# --tall-only forms) that script does -- see that script's own header
# comment for the full shape-list/series reasoning, not repeated here.
#
# Usage:
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh
#       Runs both sweeps with 10 warmup runs and 10 timed-and-averaged
#       iterations per shape (the defaults).
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh --warmup 20 --iters 50
#       Same sweeps, 20 warmup runs and 50 timed-and-averaged iterations
#       per shape instead.
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh --square-only [M1 M2 ...] [--warmup N] [--iters N]
#       Square sweep only -- see the sibling script's own header for the
#       exact form; --warmup/--iters may appear anywhere among the args.
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh --tall-only [K M1 M2 ...] [--warmup N] [--iters N]
#       Tall-and-thin sweep only -- same form as the sibling script.
#
# All shapes' full console output is saved to one timestamped .txt log
# under build/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced_
# output/ (a SEPARATE directory from the sibling script's own output),
# parsed into one .csv (same schema as the sibling script's own CSV --
# the .cu's own printed method lines are unaffected by the timing-
# methodology change). Plots are written with an "_avg" tag (see plot_
# gemv_transpose_fp32_16_pure_half_vs_cublas_ref.m's own `tag`
# parameter), so they land ALONGSIDE the sibling script's own plots
# without overwriting them:
#   build/benchmarks/output/plots/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_square_avg.png / .fig
#   build/benchmarks/output/plots/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_tall_avg.png   / .fig

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced.cu"
OUT="$BUILD_DIR/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced_output"

# --warmup N / --iters N (both optional, default 10/10) -- extracted
# from anywhere in "$@" before this script's own --square-only/--tall-
# only parsing below runs, so both forms compose freely.
WARMUP=10
ITERS=10
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@" --warmup "$WARMUP" --iters "$ITERS"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
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
TXT_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced_${TIMESTAMP}.csv"

# Same shape lists as scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh's own.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)
TALL_KS=(512 4096)
TALL_MS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

RUN_SQUARE=1
RUN_TALL=1
if [ "${1:-}" = "--square-only" ]; then
    RUN_TALL=0
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_MS=("$@")
    fi
elif [ "${1:-}" = "--tall-only" ]; then
    RUN_SQUARE=0
    shift
    if [ "$#" -gt 0 ]; then
        TALL_KS=("$1")
        shift
        TALL_MS=("$@")
    fi
fi

: > "$TXT_LOG"
if [ "$RUN_SQUARE" -eq 1 ]; then
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced $m (warmup=$WARMUP iters=$ITERS)"
        "$OUT" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
    done
fi
if [ "$RUN_TALL" -eq 1 ]; then
    for k in "${TALL_KS[@]}"; do
        for m in "${TALL_MS[@]}"; do
            echo
            echo "==> Running mv_fp32_16_pure_half_transpose_vs_cublas_ref_advanced --tall $k $m (warmup=$WARMUP iters=$ITERS)"
            "$OUT" --tall "$k" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
        done
    done
fi

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Marker -> CSV field prefix. No marker here is a substring of another
# (verified via a Python collision-matrix check before this list was
# written, same discipline as every other *_vs_cutlass_ref.sh/*_vs_
# cublas_ref.sh script's own METHOD_MARKERS) -- so the first-match-wins
# `break` below never actually has to arbitrate a real ambiguity.
METHOD_MARKERS = [
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2 =", "pure_half_half2"),
    ("mv_fp32_16_pure_half   =", "pure_half"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 =", "pure_half_atomiconly"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor =", "pure_half_atomiconly_retuned_floor"),
    ("mv_fp32_16_pure_half_transpose_wmma =", "pure_half_wmma"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor =",
     "pure_half_atomiconly_half_io_retuned_floor"),
    ("mv_fp32_16_pure_half_transpose_warp_owner_split_f32 =", "pure_half_warp_owner_split_f32"),
    ("mv_fp32_16_pure_half_transpose_warp_owner_split =", "pure_half_warp_owner_split"),
    ("mv_fp32_16_pure_half_transpose_warp_owner =", "pure_half_warp_owner"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo =",
     "pure_half_atomiconly_vec4_split_retarget_lo"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget =",
     "pure_half_atomiconly_vec4_split_retarget"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast =",
     "pure_half_atomiconly_vec4_split_fast"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split =", "pure_half_atomiconly_vec4_split"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast =", "pure_half_atomiconly_vec4_fast"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired =", "hybrid_vec4_paired"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4 =", "hybrid_vec4"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock =", "hybrid"),
    ("mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 =", "fp32_bf16_pure_bf16"),
    ("cublas (Hgemm n=1, TRANSPOSE)", "cublas_hgemm"),
    ("cublas (GemmEx CUDA_R_16BF n=1, TRANSPOSE)", "cublas_gemmex_bf16"),
]

FIELDS = [
    "M", "K",
    "pure_half_ms", "pure_half_gflops",
    "pure_half_half2_ms", "pure_half_half2_gflops",
    "pure_half_atomiconly_ms", "pure_half_atomiconly_gflops",
    "pure_half_atomiconly_retuned_floor_ms", "pure_half_atomiconly_retuned_floor_gflops",
    "pure_half_wmma_ms", "pure_half_wmma_gflops",
    "pure_half_atomiconly_half_io_retuned_floor_ms", "pure_half_atomiconly_half_io_retuned_floor_gflops",
    "pure_half_warp_owner_ms", "pure_half_warp_owner_gflops",
    "pure_half_warp_owner_split_ms", "pure_half_warp_owner_split_gflops",
    "pure_half_warp_owner_split_f32_ms", "pure_half_warp_owner_split_f32_gflops",
    "pure_half_atomiconly_vec4_split_ms", "pure_half_atomiconly_vec4_split_gflops",
    "pure_half_atomiconly_vec4_split_retarget_ms", "pure_half_atomiconly_vec4_split_retarget_gflops",
    "pure_half_atomiconly_vec4_split_retarget_lo_ms", "pure_half_atomiconly_vec4_split_retarget_lo_gflops",
    "pure_half_atomiconly_vec4_fast_ms", "pure_half_atomiconly_vec4_fast_gflops",
    "hybrid_ms", "hybrid_gflops",
    "hybrid_vec4_ms", "hybrid_vec4_gflops",
    "hybrid_vec4_paired_ms", "hybrid_vec4_paired_gflops",
    "pure_half_atomiconly_vec4_split_fast_ms", "pure_half_atomiconly_vec4_split_fast_gflops",
    "fp32_bf16_pure_bf16_ms", "fp32_bf16_pure_bf16_gflops",
    "cublas_hgemm_ms", "cublas_hgemm_gflops",
    "cublas_gemmex_bf16_ms", "cublas_gemmex_bf16_gflops",
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
            row = {"M": m.group(1), "K": m.group(2)}
            continue
        if row is None:
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
echo "==> Plotting (avg): $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG', 'avg')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG', 'avg')\"" >&2
fi

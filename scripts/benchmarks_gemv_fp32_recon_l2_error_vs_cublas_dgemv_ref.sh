#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_
# dgemv_ref.cu and runs the SAME square sweep (M=K) as scripts/
# benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh (this script's own
# FP32-reference sibling), but measuring L2-relative-error against a TRUE
# FP64 (cublasDgemv) reference instead of FP32 (cublasSgemv) -- for the
# SAME five "TOMOS" fp32-reconstruction-tier GEMV series. See the .cu's
# own header for the full derivation of why FP64 matters here (the sibling
# script's own "TOMOS FP32" series is nearly a no-op comparison against a
# same-precision FP32 reference).
#
# Square sweep only, CAPPED AT 24576 (not the FP32-reference sibling's own
# 131072) -- each series now needs FOUR concurrently-alive M*K-sized
# buffers at peak (float A + double A_d + wh + wr for the four packed-
# format series) instead of that sibling's own two, so the safe ceiling is
# much smaller -- see the .cu's own header for the exact derivation (same
# 24576 ceiling scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_
# dgemm_ref.sh already uses for its own GEMM analog, and for the same
# reason).
#
# Runs the FULL square sweep once per entry in SCALES below, same list as
# the FP32-reference sibling script's own (every series here accumulates
# in float, so the binding constraint is fp32_16_pack.hpp's own NORMAL/
# EXTENDED encoding boundary at magnitude 65536, unaffected by which
# reference precision is used) -- see that script's own header for the
# full reasoning, not repeated here.
#
# --normalized (may appear anywhere among the arguments) scales x by
# 1/sqrt(K) at data-generation time, same as the FP32-reference sibling.
# Plots land with a "_normalized" tag so they don't overwrite the plain
# run's own.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh
#       Runs the full default SQUARE_MS list at every default SCALES tier.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales S1 [S2 ...] -- M1 [M2 ...]
#       Runs exactly those scale tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SQUARE_MS list.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh -- M1 [M2 ...]
#       Runs the default SCALES tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh --normalized [--scales ... [-- ...]]
#       Same forms as above, plus --normalized (may be combined with any
#       of them, in any position).
#
# Rebuilds tomos_cuda first, links the standalone nvcc build against it
# directly -- same convention as every other *_vs_cublas_ref.sh script.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (L2 relative error for each of the five series) is parsed
# out into one .csv, both under build/mv_fp32_recon_l2_error_vs_cublas_
# dgemv_ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref.cu"
OUT="$BUILD_DIR/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_dimension_scaling/gemv/fp32_compute/data"

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

NORMALIZED=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--normalized" ]; then
        NORMALIZED=1
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_recon_l2_error_vs_cublas_dgemv_ref"
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

SCALES=(1 4 16 64 256 1024 4096 16384)

# CAPPED at 24576 (not the FP32-reference sibling's own 131072) -- see
# this script's own header comment for the memory derivation.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576)

if [ "${1:-}" = "--scales" ]; then
    shift
    SCALES=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        SCALES+=("$1")
        shift
    done
    if [ "${1:-}" = "--" ]; then
        shift
        if [ "$#" -gt 0 ]; then
            SQUARE_MS=("$@")
        fi
    fi
elif [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_MS=("$@")
    fi
fi

NORM_SUFFIX=""
NORM_FLAG=()
if [ "$NORMALIZED" -eq 1 ]; then
    NORM_SUFFIX="_normalized"
    NORM_FLAG=(--normalized)
fi

for scale in "${SCALES[@]}"; do
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    TXT_LOG="$OUTPUT_DIR/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mv_fp32_recon_l2_error_vs_cublas_dgemv_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_recon_l2_error_vs_cublas_dgemv_ref $scale $m${NORM_SUFFIX:+ (normalized)}"
        "$OUT" "$scale" "$m" "${NORM_FLAG[@]}" | tee -a "$TXT_LOG"
    done

    echo
    echo "==> Writing CSV summary: $CSV_LOG"
    python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Same METHOD_MARKERS list/ordering as scripts/benchmarks_gemv_fp32_
# recon_l2_error_vs_cublas_ref.sh's own embedded parser (identical
# kernels/series, only the reference precision differs), plus cublas_fp32
# (this file's own extra baseline, no FP32-reference sibling equivalent).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_bf16_workspace", "tomos_fp32_bf16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp32", "tomos_fp32"),
    ("cublas_fp32", "cublas_fp32"),
]

FIELDS = ["M", "K"]
for _, key in METHOD_MARKERS:
    FIELDS.append(f"{key}_l2_rel_err")

rows = []
row = None

def flush():
    global row
    if row is not None:
        rows.append(row)
    row = None

with open(txt_path) as f:
    for line in f:
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
                row[f"{key}_l2_rel_err"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_l2_rel_err"] = vm.group(1)
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

    SCALE_TAG="${scale}${NORM_SUFFIX}"

    echo
    echo "==> Plotting (normal_scale=$scale, normalized=$NORMALIZED): $CSV_LOG"
    if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load matlab/R2024a 2>/dev/null || true
    fi
    if command -v matlab >/dev/null 2>&1; then
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

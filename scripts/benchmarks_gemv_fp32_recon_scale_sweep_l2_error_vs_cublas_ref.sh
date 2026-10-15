#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_
# ref.cu (same binary scripts/benchmarks_gemv_fp32_recon_l2_error_vs_
# cublas_ref.sh uses) and runs it at ONE FIXED SHAPE (M=K=65536, the
# largest square size that script's own sweep completes without SKIPping
# for out-of-device-memory) across a sweep of normal_scale values instead
# of across shapes -- the complementary axis to that sibling script's own
# M-sweep-at-one-scale-per-run. Same convention as scripts/benchmarks_
# gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh (this
# codebase's own first scale-sweep script, for the GEMM fp32-compute
# tier).
#
# Why this axis: fp32_16_pack.hpp's/fp32_bf16_pack.hpp's own encoding has
# NORMAL/EXTENDED/degraded branches, and only large enough |A[i,j]| values
# ever cross out of "normal" -- the M-sweep sibling script never varies
# normal_scale within one run, so it can only show rounding-precision loss
# WITHIN the normal branch at whatever one scale that run used. This
# script's own sweep instead pushes more of A's own elements toward and
# across that boundary at one large, fixed shape.
#
# --normalized (optional, may appear anywhere among the arguments) is
# passed straight through to the binary, same as the M-sweep sibling
# script's own --normalized handling -- see that script's own header for
# why it doesn't change which SCALES are safe here (every series in this
# tier accumulates in genuine float, no accumulator-overflow ceiling
# either way) but still changes the DATA (x scaled by 1/sqrt(K)) and is
# therefore still a meaningful axis to check independently. Plots land
# with a "_normalized" tag so they don't overwrite the plain run's own.
#
# SCALES = 1, 4, 16, 64, 256, 1024, 4096, 16384 -- same range as
# scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh's own
# SCALES (a 4x margin below fp32_16/fp32_bf16's own NORMAL/EXTENDED
# encoding boundary at magnitude 65536).
#
# Rebuilds tomos_cuda first, links the standalone nvcc build against it
# directly -- same convention as every other *_vs_cublas_ref.sh script.
#
# Full console output is saved to one timestamped .txt log, and a per-
# scale summary (L2 relative error for each of the five series, at the
# fixed shape) is parsed out into one .csv, both under build/mv_fp32_
# recon_l2_error_vs_cublas_ref_output/ (same output directory the M-sweep
# sibling script uses -- same binary). The .csv is then plotted via
# scripts/plot_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.m
# (x-axis = normal_scale, not M) -- auto-invoked below if MATLAB is
# available.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.sh
#       Runs the default SCALES list at the fixed M=K=65536 shape.
#   scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.sh --normalized
#       Same, with x scaled by 1/sqrt(K) at data-generation time.
#   scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.sh -- S1 [S2 ...]
#       Runs exactly those normal_scale values instead of the default list
#       (may be combined with --normalized, in any position).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mv_fp32_recon_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_recon_l2_error_vs_cublas_ref_output"

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

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

# --normalized (may appear anywhere among the arguments) -- extracted
# before the -- SCALES parsing below, so it composes freely with it.
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

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_recon_l2_error_vs_cublas_ref"
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

FIXED_M=65536
SCALES=(1 4 16 64 256 1024 4096 16384)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SCALES=("$@")
    fi
fi

NORM_SUFFIX=""
NORM_FLAG=()
if [ "$NORMALIZED" -eq 1 ]; then
    NORM_SUFFIX="_normalized"
    NORM_FLAG=(--normalized)
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref${NORM_SUFFIX}_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref${NORM_SUFFIX}_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for scale in "${SCALES[@]}"; do
    echo
    echo "==> Running mv_fp32_recon_l2_error_vs_cublas_ref $scale $FIXED_M${NORM_SUFFIX:+ --normalized}"
    "$OUT" "$scale" "$FIXED_M" "${NORM_FLAG[@]}" | tee -a "$TXT_LOG"
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

banner_re = re.compile(r"normal_scale=([0-9.eE+-]+),\s*normalized=(\d+)")
shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Same METHOD_MARKERS list/ordering as scripts/benchmarks_gemv_fp32_
# recon_l2_error_vs_cublas_ref.sh's own embedded parser (both scripts
# drive the same binary).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_bf16_workspace", "tomos_fp32_bf16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp32", "tomos_fp32"),
]

FIELDS = ["scale", "M", "K"]
for _, key in METHOD_MARKERS:
    FIELDS.append(f"{key}_l2_rel_err")

rows = []
row = None
current_scale = None

def flush():
    global row
    if row is not None:
        rows.append(row)
    row = None

with open(txt_path) as f:
    for line in f:
        bm = banner_re.search(line)
        if bm:
            current_scale = bm.group(1)
            continue

        m = shape_re.match(line)
        if m:
            flush()
            row = {"scale": current_scale, "M": m.group(1), "K": m.group(2)}
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

echo
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

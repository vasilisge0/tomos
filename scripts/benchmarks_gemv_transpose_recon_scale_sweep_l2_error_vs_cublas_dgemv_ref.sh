#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_l2_error_
# vs_cublas_dgemv_ref.cu (same binary scripts/benchmarks_gemv_transpose_
# fp32_recon_l2_error_vs_cublas_dgemv_ref.sh uses) and runs it at ONE
# FIXED SHAPE (M=K=24576) across a sweep of normal_scale values instead of
# across shapes -- the TRANSPOSE counterpart of scripts/benchmarks_gemv_
# fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh (see that
# script's own header for the full derivation, not repeated below --
# identical reasoning, only the direction differs; same six series, five
# TOMOS plus cuBLAS FP32, same TRUE FP64 reference).
#
# FIXED_M=24576 (not the FP32-reference sibling's own 65536) -- same
# memory reasoning as scripts/benchmarks_gemv_transpose_fp32_recon_l2_
# error_vs_cublas_dgemv_ref.sh's own square_sizes cap.
#
# SCALES = 1, 4, 16, 64, 256, 1024, 4096, 16384.
#
# --normalized (optional, may appear anywhere among the arguments) is
# passed straight through to the binary.
#
# Usage: same argument forms as scripts/benchmarks_gemv_fp32_recon_scale_
# sweep_l2_error_vs_cublas_dgemv_ref.sh's own usage.
#
# Rebuilds tomos_cuda first, links the standalone nvcc build against it
# directly -- same convention as every other *_vs_cublas_ref.sh script.
#
# Full console output is saved to one timestamped .txt log, and a per-
# scale summary (L2 relative error for each of the six series) is parsed
# out into one .csv, both under build/mv_fp32_transpose_recon_l2_error_
# vs_cublas_dgemv_ref_output/. The .csv is then plotted via scripts/
# plot_gemv_transpose_recon_scale_sweep_vs_cublas_dgemv_ref.m (x-axis =
# normal_scale, not M) -- auto-invoked below if MATLAB is available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu"
OUT="$BUILD_DIR/mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_magnitude_scaling/gemv/fp32_compute_transpose/data"

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

echo "==> Building: mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref"
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

FIXED_M=24576
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
TXT_LOG="$OUTPUT_DIR/mv_fp32_transpose_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref${NORM_SUFFIX}_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_transpose_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref${NORM_SUFFIX}_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for scale in "${SCALES[@]}"; do
    echo
    echo "==> Running mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref $scale $FIXED_M${NORM_SUFFIX:+ (normalized)}"
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

# Same METHOD_MARKERS list/ordering as scripts/benchmarks_gemv_transpose_
# fp32_recon_l2_error_vs_cublas_dgemv_ref.sh's own embedded parser (both
# scripts drive the same binary).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_bf16_truncated", "tomos_fp32_bf16_truncated"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp32", "tomos_fp32"),
    ("cublas_fp32", "cublas_fp32"),
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_recon_scale_sweep_vs_cublas_dgemv_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_recon_scale_sweep_vs_cublas_dgemv_ref('$CSV_LOG')\"" >&2
fi

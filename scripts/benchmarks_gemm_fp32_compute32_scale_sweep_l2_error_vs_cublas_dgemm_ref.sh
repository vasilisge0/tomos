#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_
# cublas_dgemm_ref.cu (same binary scripts/benchmarks_gemm_fp32_compute32_
# l2_error_vs_cublas_dgemm_ref.sh uses -- see that file's own header for
# the full three-series pipeline) and runs it at ONE FIXED SHAPE
# (M=K=N=24576) across a sweep of normal_scale values instead of across
# shapes -- the complementary sweep axis to that sibling script's own
# N-sweep-at-fixed-scale=1.0. 24576 (not the fp32-recon-tier's own 49152)
# keeps peak device memory during the FP64-reference phase at ~19GB,
# comfortably fitting a single 40GB GPU -- see the .cu's own header for
# the full derivation.
#
# Why this axis matters: fp32_16_pack.hpp's/fp32_bf16_pack.hpp's own
# encoding has NORMAL/EXTENDED/degraded branches, and only large enough
# |A[i,j]| values ever cross out of the "normal" branch -- the N-sweep
# sibling script never varies normal_scale, so it can only ever measure
# rounding-precision loss WITHIN the normal branch. Sweeping normal_scale
# at a large, fixed shape instead pushes more of A's own elements across
# that boundary.
#
# Always runs with --normalized (rescales B by 1/sqrt(K) -- see the .cu's
# own run_shape comment). Unlike mm_fp32_16_variable_rounding_l2_error_vs_
# cublas_ref.cu's own scale-sweep sibling, this is NOT working around an
# output-storage-overflow bug: every series here accumulates AND stores C
# in genuine float (see the .cu's own header), so C never comes close to
# overflowing at any of these scales regardless of normalization -- both
# --normalized and its own raw sibling (scripts/benchmarks_gemm_fp32_
# compute32_scale_sweep_raw_l2_error_vs_cublas_dgemm_ref.sh) use the
# IDENTICAL SCALES list for exactly this reason, letting the two be
# compared directly at every scale, not just a partial overlap. Kept here
# purely for API consistency with every other error benchmark in this
# codebase.
#
# SCALES = 1, 4, 16, 64, 256, 1024, 4096, 16384 -- same range as
# scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh's own
# SCALES (a 4x margin below fp32_16/fp32_bf16's own NORMAL/EXTENDED
# encoding boundary at magnitude 65536).
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# scale summary (Frobenius relative error for each series, at the fixed
# shape) is parsed out into one .csv, both under build/mm_fp32_compute32_
# l2_error_vs_cublas_dgemm_ref_output/ (same output directory the N-sweep
# sibling script uses -- same binary, just a different invocation
# pattern). The .csv is then plotted via scripts/plot_gemm_fp32_c32_
# scale_sweep_vs_dgemm_ref.m (x-axis = normal_scale, not N) -- auto-
# invoked below if MATLAB is available.
#
# Usage:
#   scripts/benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh
#       Runs the default SCALES list at the fixed M=K=N=24576 shape.
#   scripts/benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh -- S1 [S2 ...]
#       Runs exactly those normal_scale values instead of the default list.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu"
OUT="$BUILD_DIR/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_magnitude_scaling/gemm/fp32_compute/data"

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

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref"
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

FIXED_N=24576
SCALES=(1 4 16 64 256 1024 4096 16384)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SCALES=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for scale in "${SCALES[@]}"; do
    echo
    echo "==> Running mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref $scale $FIXED_N --normalized"
    "$OUT" "$scale" "$FIXED_N" --normalized | tee -a "$TXT_LOG"
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

banner_re = re.compile(r"normal_scale=([0-9.eE+-]+),\s*normalized=(\d+)")
shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Longest/most-specific marker first -- same METHOD_MARKERS list as
# scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh's
# own embedded parser (both scripts drive the same binary).
METHOD_MARKERS = [
    ("cublas_fp32", "cublas_fp32"),
    ("tomos_fp32_fp16_workspace_fast", "tomos_fp32_fp16_workspace_fast"),
    ("tomos_fp32_fp16_truncated", "tomos_fp32_fp16_truncated"),
    ("tomos_fp32_fp16_workspace", "tomos_fp32_fp16_workspace"),
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16_truncated", "tomos_fp32_bf16_truncated"),
    ("tomos_fp32_bf16_workspace", "tomos_fp32_bf16_workspace"),
    ("tomos_fp32", "tomos_fp32"),
]

FIELDS = ["scale", "M", "K", "N"]
for _, key in METHOD_MARKERS:
    FIELDS.append(f"{key}_frobenius_rel_err")

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
            row = {"scale": current_scale, "M": m.group(1), "K": m.group(2), "N": m.group(3)}
            continue
        if row is None:
            continue

        for marker, key in METHOD_MARKERS:
            if marker not in line:
                continue
            if "SKIPPED" in line:
                row[f"{key}_frobenius_rel_err"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_frobenius_rel_err"] = vm.group(1)
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref('$CSV_LOG'); plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref_fast('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref('$CSV_LOG'); plot_gemm_fp32_c32_scale_sweep_vs_dgemm_ref_fast('$CSV_LOG')\"" >&2
fi

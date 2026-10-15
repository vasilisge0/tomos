#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_
# error_vs_cublas_ref.cu (same binary scripts/benchmarks_gemm_fp32_16_
# variable_rounding_l2_error_vs_cublas_ref.sh and scripts/benchmarks_
# gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh
# both use) and runs it at the SAME FIXED SHAPE (M=K=N=49152) across the
# SAME normal_scale axis as that scale-sweep sibling script -- but WITHOUT
# --normalized, i.e. B stays at plain scale=1.0 (this codebase's own
# original "only A's magnitude sweeps" convention) instead of being
# rescaled by 1/sqrt(K).
#
# Why this script exists, alongside the --normalized one: normalizing B
# rescales the whole Frobenius-relative-error RATIO's numerator and
# denominator by the same factor, so in exact arithmetic it would not
# change the reported relative-error values at all -- its only real job
# is keeping C's own row-sum magnitude representable in fp16 at large
# normal_scale/K, which --normalized's own sibling script needs to reach
# fp32_16's extended/degraded encoding branches without overflowing
# first. This script exists to check that expectation directly: run the
# SAME rounding-precision comparison, at the SAME fixed shape, WITHOUT
# normalization, over the SCALES range that stays safely representable
# on its own -- if normalization really is "just" an overflow-avoidance
# mechanism and not something that changes the underlying comparison,
# this plot's values (where both sweeps cover the same normal_scale)
# should closely match scripts/benchmarks_gemm_fp32_16_variable_
# rounding_scale_sweep_l2_error_vs_cublas_ref.sh's own --normalized
# results at the SAME normal_scale, even though the two sweeps use
# different absolute data magnitudes for B and hence different absolute
# magnitudes for C.
#
# SCALES = 1, 2, 4, 8, 16, 32, 64 -- "the corresponding scalings" for
# this UN-normalized case: the same tight, K-dependent-overflow-safe
# range mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu's own (un-normalized)
# default SCALES uses, derived there for K up to 65536 via an expected-
# overflow-COUNT argument (not just a sigma bound, since a single
# overflowing element already poisons the whole Frobenius sum, and this
# shape reaches billions of output elements) -- see that file's own
# header for the full derivation. K=49152 here is smaller than the
# 65536 that range was derived for, so this SCALES list is comfortably
# (not just marginally) safe at this shape: std ~ scale*sqrt(49152)/3 ~
# scale*73.9, giving z = 65504/(64*73.9) ~ 13.85 sigma at the top of
# this range -- an even wider margin than the ~12-sigma the original
# derivation targeted.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# scale summary (Frobenius relative error for each series, at the fixed
# shape) is parsed out into one .csv, both under build/mm_fp32_16_
# variable_rounding_l2_error_vs_cublas_ref_output/ (same output directory
# every sibling script driving this binary uses). The .csv is then
# plotted via scripts/plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_
# vs_cublas_ref.m (x-axis = normal_scale, not N, same as the --normalized
# sibling plot but a SEPARATE file/function -- see that .m file's own
# comment for why "round_scale_sweep" and not "variable_rounding_scale_
# sweep": MATLAB's own 63-character function/file name limit) --
# auto-invoked below if MATLAB is available.
#
# Usage:
#   scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref.sh
#       Runs the default SCALES list at the fixed M=K=N=49152 shape.
#   scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref.sh -- S1 [S2 ...]
#       Runs exactly those normal_scale values instead of the default list.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_output"

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

echo "==> Building: mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref"
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

FIXED_N=49152
SCALES=(1 2 4 8 16 32 64)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SCALES=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for scale in "${SCALES[@]}"; do
    echo
    echo "==> Running mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref $scale $FIXED_N"
    "$OUT" "$scale" "$FIXED_N" | tee -a "$TXT_LOG"
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

# None of these markers is a substring of another -- same discipline as
# every sibling script driving this same binary.
METHOD_MARKERS = [
    ("tomos_fp16_native", "tomos_fp16_native"),
    ("tomos_fp16_trunc10", "tomos_fp16_trunc10"),
    ("tomos_fp32_fp16_stochastic", "tomos_fp32_fp16_stochastic"),
    ("tomos_fp32_fp16_round_13", "tomos_fp32_fp16_round_13"),
    ("tomos_fp32_fp16_round_09", "tomos_fp32_fp16_round_09"),
    ("tomos_fp32_fp16_round_05", "tomos_fp32_fp16_round_05"),
    ("tomos_fp32_fp16_round_01", "tomos_fp32_fp16_round_01"),
    ("tomos_fp32_fp16_trunc", "tomos_fp32_fp16_trunc"),
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

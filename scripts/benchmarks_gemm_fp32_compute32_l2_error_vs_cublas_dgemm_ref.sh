#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_
# cublas_dgemm_ref.cu and runs its default square sweep (M=K=N) at
# normal_scale=1.0 -- the N-sweep companion to scripts/benchmarks_gemm_
# fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh (fixed shape,
# swept normal_scale instead). See the .cu's own header for the full
# three-series design (TOMOS FP32, TOMOS FP32/FP16, TOMOS FP32/BF16, all
# accumulating in float) and for why the reference is a genuine FP64
# (cublasDgemm, on A/B widened to double) instead of the FP32 reference
# every *_recon_l2_error_vs_cublas_ref.cu sibling uses.
#
# Square sweep only, capped at 24576 (not the fp32-recon-tier's own
# 49152) -- see the .cu's own header for why: this file's FP64 reference
# needs three extra double-typed buffers on top of the float A/B, and at
# 49152 that peaks at ~77GB of device memory, likely exceeding even an
# 80GB GPU. 24576 keeps peak device memory during the reference phase at
# ~19GB, comfortably fitting a single 40GB GPU with room to spare.
# SQUARE_NS: 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576.
#
# Usage:
#   scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh
#       Runs the full default SIZES list.
#   scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh -- N1 [N2 ...]
#       Runs exactly those N values instead of the default SIZES list.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (Frobenius relative error for each series) is parsed out
# into one .csv, both under build/mm_fp32_compute32_l2_error_vs_cublas_
# dgemm_ref_output/. The .csv is then plotted via scripts/plot_gemm_
# fp32_c32_vs_dgemm_ref.m (auto-invoked below if MATLAB is available).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.cu"
OUT="$BUILD_DIR/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_dimension_scaling/gemm/fp32_compute/data"

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

SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_NS=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for n in "${SQUARE_NS[@]}"; do
    echo
    echo "==> Running mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref 1.0 $n"
    "$OUT" 1.0 "$n" | tee -a "$TXT_LOG"
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Longest/most-specific marker first -- "tomos_fp32" is a literal prefix
# of "tomos_fp32_fp16"/"tomos_fp32_bf16", which are themselves literal
# prefixes of "tomos_fp32_fp16_workspace"/"tomos_fp32_fp16_fast", which
# "tomos_fp32_fp16_workspace" is itself a literal prefix of
# "tomos_fp32_fp16_workspace_fast" (verified by inspection, same
# discipline as every other METHOD_MARKERS list in this codebase).
# "cublas_fp32" shares no prefix relationship with any tomos_* marker, so
# its own position in this list doesn't matter for correctness.
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

FIELDS = ["M", "K", "N"]
for _, key in METHOD_MARKERS:
    FIELDS.append(f"{key}_frobenius_rel_err")

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
            row = {"M": m.group(1), "K": m.group(2), "N": m.group(3)}
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_c32_vs_dgemm_ref('$CSV_LOG'); plot_gemm_fp32_c32_vs_dgemm_ref_fast('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_c32_vs_dgemm_ref('$CSV_LOG'); plot_gemm_fp32_c32_vs_dgemm_ref_fast('$CSV_LOG')\"" >&2
fi

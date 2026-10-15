#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp32_16_variable_rounding_l2_
# error_vs_cublas_ref.cu and runs the SAME square sweep (M=K=N) scripts/
# benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh's own SIZES
# uses, measuring how the fp32_16 packed format's ENCODE-side rounding
# decision itself affects the resulting __half value's own accuracy --
# every series (the native fp16 reference, a 10-bit-mantissa TRUNCATING
# reference, AND all 5 rounding configs) runs through the SAME,
# unmodified native tensor-core kernel (gemm_fp16_mma_ptx_wide_bigreg_
# sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp, "TOMOS FP16");
# only how A's own __half array was produced varies -- the number of
# "sticky" bits (0/1/5/9/13, out of the real encoder's own 13) the
# rounding decision examines when packing A, OR (for the 10-bit
# truncating reference) bypassing the packed format entirely. See the
# .cu's own header for the full pipeline, the mantissa-width-vs-
# rounding-method 2x2 design this isolates, and for why the new encoder
# (fp32_16_split_variable_rounding) lives entirely in that file, not in
# fp32_16_pack.hpp.
#
# Square sweep only, same list as scripts/benchmarks_gemm_fp16_bf16_pure_
# l2_error_vs_cublas_ref.sh's own SIZES: 1024/1536/2048/3072/4096/6144/
# 8192/12288/16384/24576/32768/49152/65536.
#
# No normal_scale SWEEP here (unlike that sibling script's own SCALES
# tiers) -- this experiment is about rounding-decision precision, not
# magnitude/overflow behavior, so the .cu binary always runs at its own
# default normal_scale=1.0. Run the .cu binary directly (see its own
# usage comment) if you want a different normal_scale or a single shape.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (Frobenius relative error for each series) is parsed out
# into one .csv, both under build/mm_fp32_16_variable_rounding_l2_error_
# vs_cublas_ref_output/. The .csv is then plotted via scripts/plot_gemm_
# fp32_16_variable_rounding_l2_error_vs_cublas_ref.m (auto-invoked below
# if MATLAB is available).
#
# Usage:
#   scripts/benchmarks_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.sh
#       Runs the full default SIZES list.
#   scripts/benchmarks_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.sh -- N1 [N2 ...]
#       Runs exactly those N values instead of the default SIZES list.

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

SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_NS=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for n in "${SQUARE_NS[@]}"; do
    echo
    echo "==> Running mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref 1.0 $n"
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

# None of these markers is a substring of another (2-digit zero-padded
# sticky-bit counts, "trunc" distinct from "native") -- verified by
# inspection, so unlike some sibling scripts' own METHOD_MARKERS, no
# specific ordering is required here for correctness.
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

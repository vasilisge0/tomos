#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_
# cublas_ref.cu (same binary scripts/benchmarks_gemm_fp16_bf16_pure_l2_
# error_vs_cublas_ref.sh uses) and runs it at ONE FIXED SHAPE (M=K=N=32768,
# the largest square size that script's own sweep tests) across a sweep of
# normal_scale values instead of across shapes -- the complementary axis to
# that sibling script's own N-sweep-at-one-scale-per-run. Same convention
# as scripts/benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_
# dgemm_ref.sh (this tier's own "B.1"/fp32-compute-tier analog).
#
# Always runs with --normalized: without it, this tier's own SCALES are
# capped at 64 (see the .cu's own header for the real scale=4096/"inf"-at-
# every-shape bug that forced that cap -- three of five series narrow
# their float32 accumulator down into __half-typed C at the epilogue, and
# C's own row-sum grows with sqrt(K) against fp16's ~65504 ceiling).
# --normalized removes that K-dependent growth (see the .cu's own run_
# shape comment), making the SAME wide SCALES range (up to 16384) safe at
# every shape -- the interesting range to actually sweep through.
#
# SCALES = 1, 4, 16, 64, 256, 1024, 4096, 16384 -- the --normalized-safe
# range scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh's
# own --normalized branch already uses.
#
# Rebuilds tomos_cuda first, links the standalone nvcc build against it
# directly -- same convention as every other *_vs_cublas_ref.sh script.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 -- if SLURM_JOB_ID isn't already set, this script
# finds the caller's running job on an a100* partition and re-execs itself
# inside it via `srun --overlap`.
#
# Full console output is saved to one timestamped .txt log, and a per-
# scale summary (Frobenius relative error for each of the 9 series, at
# the fixed shape) is parsed out into one .csv, both under build/mm_
# fp16_bf16_pure_l2_error_vs_cublas_ref_output/ (same output directory the
# N-sweep sibling script uses -- same binary). The .csv is then plotted
# via scripts/plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_
# ref.m (x-axis = normal_scale, not N) -- auto-invoked below if MATLAB is
# available.
#
# Usage:
#   scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh
#       Runs the default SCALES list at the fixed M=K=N=32768 shape.
#   scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh -- S1 [S2 ...]
#       Runs exactly those normal_scale values instead of the default list.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mm_fp16_bf16_pure_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_magnitude_scaling/gemm/fp16_compute/data"

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

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mm_fp16_bf16_pure_l2_error_vs_cublas_ref"
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

FIXED_N=32768
SCALES=(1 4 16 64 256 1024 4096 16384)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SCALES=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for scale in "${SCALES[@]}"; do
    echo
    echo "==> Running mm_fp16_bf16_pure_l2_error_vs_cublas_ref $scale $FIXED_N --normalized"
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

# Same METHOD_MARKERS list/ordering as scripts/benchmarks_gemm_fp16_bf16_
# pure_l2_error_vs_cublas_ref.sh's own embedded parser (both scripts
# drive the same binary).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_workspace", "tomos_fp32_fp16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp16", "tomos_fp16"),
    ("tomos_bf16", "tomos_bf16"),
    ("cublas_fp16", "cublas_fp16"),
    ("cutlass_fp16", "cutlass_fp16"),
    ("cublas_bf16", "cublas_bf16"),
    ("cutlass_bf16", "cutlass_bf16"),
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

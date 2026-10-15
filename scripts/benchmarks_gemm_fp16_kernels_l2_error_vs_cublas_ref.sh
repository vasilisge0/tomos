#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp16_kernels_l2_error_vs_
# cublas_ref.cu and runs its default square sweep (M=K=N) at
# normal_scale=1.0 -- the N-sweep companion to scripts/benchmarks_gemm_
# fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.sh (fixed shape, swept
# normal_scale instead). See the .cu's own header for the full three-
# series design: TOMOS FP16, cuBLAS FP16 (cublasHgemm), and CUTLASS FP16
# (cutlass::gemm::device::Gemm, the SAME config scripts/
# cutlass_fp16_gemm_ref.sh's own binary uses), all run on the IDENTICAL
# native-rounded A/B data, all measured against a genuine FP32
# (cublasSgemm) reference.
#
# Uses the SAME square-sweep sizes as scripts/benchmarks_gemm_fp32_16_
# variable_rounding_l2_error_vs_cublas_ref.sh (that file's own sibling
# plot family, which instead isolates A's own encoding/rounding method
# rather than the GEMM implementation) -- deliberately identical, so the
# two plot families are directly comparable at the same shapes.
#
# CUTLASS is fetched (shallow clone, header-only usage -- no separate
# library build needed) into CUTLASS_DIR on first run and reused after
# that, same convention as scripts/cutlass_fp16_gemm_ref.sh. Override the
# location with the CUTLASS_DIR env var if you already have a checkout
# elsewhere:
#   CUTLASS_DIR=/path/to/cutlass scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh
#
# Usage:
#   scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh
#       Runs the full default SIZES list.
#   scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh -- N1 [N2 ...]
#       Runs exactly those N values instead of the default SIZES list.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is only
# reachable inside an active job allocation. If SLURM_JOB_ID isn't already
# set, this script finds the caller's running job on an a100* partition and
# re-execs itself inside it via `srun --overlap`, so both the build and the
# run always land on the actual target GPU.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (Frobenius relative error for each series) is parsed out
# into one .csv, both under build/mm_fp16_kernels_l2_error_vs_cublas_ref_
# output/. The .csv is then plotted via scripts/plot_gemm_fp16_kernels_l2_
# error_vs_cublas_ref.m (auto-invoked below if MATLAB is available).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp16_kernels_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mm_fp16_kernels_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mm_fp16_kernels_l2_error_vs_cublas_ref_output"

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

echo "==> Building: mm_fp16_kernels_l2_error_vs_cublas_ref"
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

# Coarse log-spaced points up to 49152, then LINEAR steps of 1000 up to
# 57152, PLUS a further zoom of LINEAR steps of 100 from 58152 to 61152
# (30 steps), then back to the coarse 1000-step tail up to 65536 -- see
# the .cu's own square_sizes comment for why: the error stayed flat
# through 49152 at coarse resolution, so this densifies the upper range
# to look for where (if anywhere) it starts to increase.
SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 \
          50152 51152 52152 53152 54152 55152 56152 57152 \
          58152 58252 58352 58452 58552 58652 58752 58852 58952 59052 59152 \
          59252 59352 59452 59552 59652 59752 59852 59952 60052 60152 60252 \
          60352 60452 60552 60652 60752 60852 60952 61052 61152 \
          62152 63152 64152 65152 65536)

if [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_NS=("$@")
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mm_fp16_kernels_l2_error_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mm_fp16_kernels_l2_error_vs_cublas_ref_${TIMESTAMP}.csv"

: > "$TXT_LOG"
for n in "${SQUARE_NS[@]}"; do
    echo
    echo "==> Running mm_fp16_kernels_l2_error_vs_cublas_ref 1.0 $n"
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

# None of these markers is a substring of another -- verified by
# inspection.
METHOD_MARKERS = [
    ("tomos_fp16", "tomos_fp16"),
    ("cublas_fp16", "cublas_fp16"),
    ("cutlass_fp16", "cutlass_fp16"),
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_kernels_l2_error_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_kernels_l2_error_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

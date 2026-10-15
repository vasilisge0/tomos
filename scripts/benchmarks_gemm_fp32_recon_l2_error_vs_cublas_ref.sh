#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp32_recon_l2_error_vs_cublas_
# ref.cu and runs the SAME square sweep (M=K=N) scripts/benchmarks_gemm_
# format_32_16_compute_32.sh itself uses, but measuring Frobenius-
# relative-error against a genuine FP32 (cublasSgemm) reference instead
# of throughput -- for the six "TOMOS" full-fp32-reconstruction-tier GEMM
# series scripts/plot_gemm_format_32_16_compute_32.m plots: TOMOS FP32,
# TOMOS FP32/FP16, TOMOS FP32/FP16 (truncated), TOMOS FP32/FP16
# (workspace), TOMOS FP32/BF16 (truncated), TOMOS FP32/BF16 (workspace).
# See the .cu's own header comment for exactly how each method's own
# reference is generated (device-side only, to reach the same 49152-
# cubed scale the performance sweep does without a host-memory blowup),
# and for why this is only one data-generation block per series (unlike
# the GEMV recon tier's own error benchmark, no promoted GEMM kernel here
# needs magnitude-restricted or extended-range data).
#
# Square sweep only, same list as scripts/benchmarks_gemm_format_32_16_
# compute_32.sh's own SQUARE_NS: 1024/1536/2048/3072/4096/6144/8192/
# 12288/16384/24576/32768/49152. No rectangular sweep here (that driver
# has none either).
#
# Runs the FULL square sweep once per entry in SCALES below -- each is a
# "normal_scale" passed to the .cu binary (see its own header comment),
# widening A's own "normal"-range test data from [-1,1] to [-scale,
# scale]. Unlike the GEMV pure_half/hybrid tier's own SCALES (capped
# tightly by an fp16-accumulator overflow bound), every kernel here
# accumulates in float -- confirmed by reading mm_fp32_16_tiled_cpasync_
# dualbuf_kernel's own body (float acc[TM][TN]), the family every one of
# the six promoted kernels here ultimately shares -- so the binding
# constraint is instead fp32_16_pack.hpp's own NORMAL/EXTENDED encoding
# boundary: an individual per-VALUE magnitude must stay under 2^16=65536
# to remain in the packed formats' plain-normal-band encoding, a MUCH
# more generous ceiling than an accumulator-overflow bound, with no
# dependence on K at all. SCALES up to 16384 keeps a 4x margin below that
# 65536 boundary -- same values, same reasoning, as scripts/benchmarks_
# gemv_fp32_recon_l2_error_vs_cublas_ref.sh's own SCALES (this file's
# GEMV-recon-tier sibling). Each tier writes its OWN timestamped .txt/
# .csv (scale embedded in the filename) and its OWN plot (scale embedded
# in both the output filename and the plot title) -- one plot per tier,
# not overlaid, so each is independently readable.
#
# --normalized (may appear anywhere among the arguments) additionally
# scales B by 1/sqrt(K) at data-generation time instead of leaving it at
# plain [-1,1] (see the .cu's own run_shape comment). Every kernel here
# already accumulates in float (no accumulator-overflow concern either
# way), so --normalized doesn't unlock a wider SCALES range for THIS
# file specifically (added for API consistency across every error
# benchmark in this codebase -- see scripts/benchmarks_gemm_fp16_bf16_
# pure_l2_error_vs_cublas_ref.sh's own --normalized comment for the row
# where this actually matters). Plots land with a "_normalized" tag
# (e.g. ..._scale4096_normalized.png) so they don't overwrite the plain
# run's own plots.
#
# Usage:
#   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh
#       Runs the full default SQUARE_NS list at every default SCALES tier.
#   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...] -- N1 [N2 ...]
#       Runs exactly those scale tiers against exactly those N values.
#   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SQUARE_NS list.
#   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh -- N1 [N2 ...]
#       Runs the default SCALES tiers against exactly those N values.
#   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh --normalized [--scales ... [-- ...]]
#       Same forms as above, plus --normalized (may be combined with any
#       of them, in any position).
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (Frobenius relative error for each method) is parsed out
# into one .csv, both under build/mm_fp32_recon_l2_error_vs_cublas_ref_
# output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp32_recon_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mm_fp32_recon_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mm_fp32_recon_l2_error_vs_cublas_ref_output"

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

# --normalized (may appear anywhere among the arguments) -- extracted
# here, before the --scales/-- parsing below, so it composes freely with
# either form. See this script's own header comment for what it does.
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

echo "==> Building: mm_fp32_recon_l2_error_vs_cublas_ref"
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

# Same SCALES as scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_
# ref.sh (this file's GEMV-recon-tier sibling) -- see this script's own
# header comment for why the same reasoning (fp32_16_pack.hpp's own
# NORMAL/EXTENDED encoding boundary, not an accumulator-overflow bound)
# applies here too.
SCALES=(1 4 16 64 256 1024 4096 16384)

# Same list as scripts/benchmarks_gemm_format_32_16_compute_32.sh's own
# SQUARE_NS.
SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152)

# --scales S1 [S2 ...] [-- N1 [N2 ...]] / -- N1 [N2 ...] (see this
# script's own header comment for the exact forms).
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
            SQUARE_NS=("$@")
        fi
    fi
elif [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_NS=("$@")
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
    TXT_LOG="$OUTPUT_DIR/mm_fp32_recon_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mm_fp32_recon_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for n in "${SQUARE_NS[@]}"; do
        echo
        echo "==> Running mm_fp32_recon_l2_error_vs_cublas_ref $scale $n${NORM_SUFFIX:+ (normalized)}"
        "$OUT" "$scale" "$n" "${NORM_FLAG[@]}" | tee -a "$TXT_LOG"
    done

    echo
    echo "==> Writing CSV summary: $CSV_LOG"
    python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Longest/most-specific marker first -- "tomos_fp32_fp16" is a literal
# prefix of "tomos_fp32_fp16_workspace" and "tomos_fp32_fp16_truncated",
# and "tomos_fp32_bf16" is a literal prefix of "tomos_fp32_bf16_
# truncated" and "tomos_fp32_bf16_workspace", and "tomos_fp32" is itself
# a literal prefix of every other marker in this list -- REQUIRING this
# exact ordering (verified via a standalone Python collision simulation
# before finalizing this list, same discipline as every other *_recon_
# l2_error_vs_cublas_ref.sh script's own METHOD_MARKERS in this
# codebase).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_workspace", "tomos_fp32_fp16_workspace"),
    ("tomos_fp32_fp16_truncated", "tomos_fp32_fp16_truncated"),
    ("tomos_fp32_bf16_truncated", "tomos_fp32_bf16_truncated"),
    ("tomos_fp32_bf16_workspace", "tomos_fp32_bf16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
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

    SCALE_TAG="${scale}${NORM_SUFFIX}"

    echo
    echo "==> Plotting (normal_scale=$scale, normalized=$NORMALIZED): $CSV_LOG"
    if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load matlab/R2024a 2>/dev/null || true
    fi
    if command -v matlab >/dev/null 2>&1; then
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_recon_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp32_recon_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

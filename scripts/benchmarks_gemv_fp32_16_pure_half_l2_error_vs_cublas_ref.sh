#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_l2_error_vs_
# cublas_ref.cu and runs the SAME square sweep (M=K) mv_fp32_16_pure_
# half_vs_cublas_ref.cu itself uses, but measuring L2-relative-error
# against a genuine FP32 (cublasSgemv, CUBLAS_OP_N) reference instead of
# throughput -- for the six "TOMOS" series scripts/plot_gemv_fp32_16_
# pure_half_vs_cublas_ref.m plots: TOMOS FP16, TOMOS FP16 (fast), TOMOS
# FP32/FP16, TOMOS FP32/FP16 (implicit), TOMOS FP32/BF16, TOMOS BF16.
# See the .cu's own header comment for exactly how each method's own
# reference is generated (device-side only, to reach the same
# 131072x131072 scale the performance sweep does without the host-memory
# blowup a full-matrix CPU reference would hit), and for why this is six
# series, not four like the TRANSPOSE sibling (this tier plots a separate
# plain "TOMOS BF16" series in addition to "TOMOS FP32/BF16" -- the
# TRANSPOSE sibling's own performance companion doesn't).
#
# Square sweep only, same list as mv_fp32_16_pure_half_vs_cublas_ref.cu's
# own SQUARE_MS: 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
# 32768/49152/65536/131072. No tall-and-thin sweep here.
#
# Runs the FULL square sweep once per entry in SCALES below -- each is a
# "normal_scale" passed to the .cu binary (see its own header comment),
# widening every method's own "normal"-range test data from [-1,1] to
# [-scale,scale]. NOTE this stays far below fp16's own ~65504 per-VALUE
# ceiling on purpose: every series here except the recon/float-
# accumulating tiers (not plotted here at all) accumulates its row-sum in
# fp16/bf16 precision INTERNALLY (NO_TRANSPOSE reduces over K), so it's
# the ACCUMULATED SUM across up to K terms that must stay under that
# ceiling, a much tighter bound than "does one value fit in fp16" -- SAME
# empirically-derived SCALES as scripts/benchmarks_gemv_fp32_16_l2_error_
# vs_cublas_ref.sh (this codebase's non-pure_half NO_TRANSPOSE sibling),
# since the largest completing square shape and its own memory footprint
# are identical regardless of which tier is being measured. Each tier
# writes its OWN timestamped .txt/.csv (scale embedded in the filename)
# and its OWN plot (scale embedded in both the output filename and the
# plot title) -- one plot per tier, not overlaid, so each is independently
# readable.
#
# --normalized (may appear anywhere among the arguments) scales x by
# 1/sqrt(K) (the reduction dimension for NO_TRANSPOSE) at data-generation
# time instead of leaving it at plain [-1,1] (see the .cu's own run_shape
# comment) -- every fp16/bf16-accumulating kernel here sums RAW products
# BEFORE alpha is ever applied, so only a data-generation-time scale (not
# alpha) can keep that accumulator from overflowing. With --normalized,
# the accumulated magnitude stays ~normal_scale regardless of K instead of
# growing like normal_scale*sqrt(K), so the SAME wide SCALES this
# codebase's float-accumulating rows use becomes safe here too. Plots
# land with a "_normalized" tag (e.g. ..._scale4096_normalized.png) so
# they don't overwrite the plain run's own plots.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh
#       Runs the full default SQUARE_MS list at every default SCALES tier.
#   scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...] -- M1 [M2 ...]
#       Runs exactly those scale tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SQUARE_MS list.
#   scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh -- M1 [M2 ...]
#       Runs the default SCALES tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh --normalized [--scales ... [-- ...]]
#       Same forms as above, plus --normalized (may be combined with any
#       of them, in any position) -- switches the default SCALES to the
#       wider, normalized-safe list.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (L2 relative error for each method) is parsed out into
# one .csv, both under build/mv_fp32_16_pure_half_l2_error_vs_cublas_ref_
# output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mv_fp32_16_pure_half_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_16_pure_half_l2_error_vs_cublas_ref_output"

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

echo "==> Building: mv_fp32_16_pure_half_l2_error_vs_cublas_ref"
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

# Same SCALES as scripts/benchmarks_gemv_transpose_fp32_16_pure_half_l2_
# error_vs_cublas_ref.sh (this tier's TRANSPOSE sibling) -- for a row-sum
# of K terms each ~scale in magnitude, its own standard deviation is
# roughly scale*sqrt(K)/3. The largest completing square shape is M=K=
# 65536 (M=K=131072 always gets SKIPPED regardless of scale -- ~68.7GB
# single-float32-array ceiling), so scale=128 keeps that row-sum's own
# std ~6x (~6 sigma) below 65504 here too, by the identical calculation.
# TOMOS FP32/BF16 and TOMOS BF16 both accumulate in bf16 precision
# internally as well (see the .cu's own header), so the SAME cap applies
# uniformly across all six series plotted here.
#
# --normalized switches to a WIDER default -- see this script's own
# header comment for the derivation (identical to the TRANSPOSE sibling's
# own --normalized SCALES, just with K in place of M).
if [ "$NORMALIZED" -eq 1 ]; then
    SCALES=(1 4 16 64 256 1024 4096 16384)
else
    SCALES=(1 2 4 8 16 32 64 128)
fi

# Same list as mv_fp32_16_pure_half_vs_cublas_ref.cu's own SQUARE_MS.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)

# --scales S1 [S2 ...] [-- M1 [M2 ...]] / -- M1 [M2 ...] (see this
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
    TXT_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_16_pure_half_l2_error_vs_cublas_ref $scale $m${NORM_SUFFIX:+ (normalized)}"
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

# Longest/most-specific marker first -- "tomos_fp32_fp16" is a literal
# prefix of "tomos_fp32_fp16_implicit", and "tomos_fp16" is a literal
# prefix of "tomos_fp16_fast" -- both orderings REQUIRED (same discipline
# as scripts/benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_
# cublas_ref.sh's own METHOD_MARKERS, verified via the same standalone
# Python collision simulation). "tomos_bf16" is NOT a substring of
# "tomos_fp32_bf16" (has "fp32_" in between), so it doesn't need to
# precede anything -- confirmed by the same simulation.
METHOD_MARKERS = [
    ("tomos_fp16_fast", "tomos_fp16_fast"),
    ("tomos_fp32_fp16_implicit", "tomos_fp32_fp16_implicit"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp16", "tomos_fp16"),
    ("tomos_bf16", "tomos_bf16"),
    ("cublas_fp16", "cublas_fp16"),
    ("cublas_bf16", "cublas_bf16"),
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
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

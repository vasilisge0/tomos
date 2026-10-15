#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_ref.cu
# and runs the SAME square sweep (M=K) mv_fp32_recon_vs_cutlass_ref.cu's
# own driver uses, but measuring L2-relative-error against a genuine FP32
# (cublasSgemv) reference instead of throughput -- for the same five
# "TOMOS" fp32-reconstruction-tier GEMV series scripts/plot_gemv_fp32_
# recon_vs_cutlass_ref.m plots: TOMOS FP32, TOMOS FP32/FP16, TOMOS FP32/
# FP16 (fast), TOMOS FP32/BF16, TOMOS FP32/BF16 (workspace). See the
# .cu's own header comment for exactly how each method's own reference is
# generated (device-side only, to reach the same 131072x131072 scale the
# performance sweep does without the host-memory blowup a full-matrix CPU
# reference would hit).
#
# Square sweep only, same list as mv_fp32_16_l2_error_vs_cublas_ref.cu's
# own SQUARE_MS: 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
# 32768/49152/65536/131072. No tall-and-thin sweep here -- add one later
# if the square results motivate it.
#
# Runs the FULL square sweep once per entry in SCALES below -- each is a
# "normal_scale" passed to the .cu binary (see its own header comment),
# widening every method's own data from [-1,1] to [-scale,scale]. UNLIKE
# mv_fp32_16_l2_error_vs_cublas_ref.cu's own sweep, every kernel measured
# here accumulates its row-sum in FLOAT (confirmed by reading mv_fp32_16_
# ready4_atomiconly_scaled_kblock_kernel's own body), not fp16/__half2 --
# so there is no K-dependent accumulator-overflow ceiling to work around.
# The binding constraint instead is fp32_16_pack.hpp's own NORMAL/
# EXTENDED encoding boundary (magnitude 65536): mv_fp32_16_accurate_
# ready_atomiconly_scaled_kblock_fast_normal's own data must stay inside
# the NORMAL band for its "always control=0" contract to hold (see the
# .cu's own gen_fp32_16_safe_normal_kernel comment) -- a much more
# generous limit than Row 3's own accumulator-bound SCALES=(...128), so
# this sweep reaches much further before needing to stop. Each tier
# writes its OWN timestamped .txt/.csv (scale embedded in the filename)
# and its OWN plot (scale embedded in both the output filename and the
# plot title) -- one plot per tier, not overlaid, so each is
# independently readable.
#
# --normalized (may appear anywhere among the arguments) additionally
# scales x by 1/sqrt(K) at data-generation time (see the .cu's own
# run_shape comment) instead of leaving it at plain [-1,1] -- this keeps
# y's own magnitude ~normal_scale regardless of K instead of growing
# like normal_scale*sqrt(K). Every kernel here already accumulates in
# float (no accumulator-overflow concern either way), so --normalized
# doesn't unlock a wider SCALES range for THIS file specifically (added
# for API consistency across every error benchmark in this codebase --
# see scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh's
# own --normalized comment for the row where this actually matters).
# Plots land with a "_normalized" tag (e.g. ..._scale4096_normalized.png)
# so they don't overwrite the plain (non-normalized) run's own plots.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh
#       Runs the full default SQUARE_MS list at every default SCALES tier.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...] -- M1 [M2 ...]
#       Runs exactly those scale tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SQUARE_MS list.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh -- M1 [M2 ...]
#       Runs the default SCALES tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh --normalized [--scales ... [-- ...]]
#       Same forms as above, plus --normalized (may be combined with any
#       of them, in any position).
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (L2 relative error for each method) is parsed out into
# one .csv, both under build/mv_fp32_recon_l2_error_vs_cublas_ref_output/.

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

# Progressive magnitude-scale tiers -- powers of 2, kept well under fp32_
# 16's own 65536 normal/extended encoding boundary (see the .cu's own
# gen_fp32_16_safe_normal_kernel comment for why -- mv_fp32_16_accurate_
# ready_atomiconly_scaled_kblock_fast_normal's own "always control=0"
# contract needs its data to stay inside that boundary). Every kernel
# here accumulates in float (no accumulator-overflow ceiling the way Row
# 3's fp16-accumulating series had), so this sweep can go much further
# than that file's own SCALES=(...128) -- 16384 keeps real margin (4x)
# below 65536 even at the top of the sweep. Same list regardless of
# --normalized here (this row has no overflow concern either way to
# widen a range for) -- kept as one shared default for consistency.
SCALES=(1 4 16 64 256 1024 4096 16384)

# Same list as mv_fp32_16_l2_error_vs_cublas_ref.cu's own SQUARE_MS.
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
    TXT_LOG="$OUTPUT_DIR/mv_fp32_recon_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mv_fp32_recon_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_recon_l2_error_vs_cublas_ref $scale $m${NORM_SUFFIX:+ (normalized)}"
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

# Longest/most-specific marker FIRST -- REQUIRED here, not just
# defensive: "tomos_fp32" is a literal substring of every other marker
# below (e.g. "tomos_fp32_fp16"), and "tomos_fp32_fp16"/"tomos_fp32_bf16"
# are themselves substrings of their own "_fast"/"_workspace" siblings --
# checked, confirmed each maps to its own distinct key with this exact
# ordering.
METHOD_MARKERS = [
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_bf16_workspace", "tomos_fp32_bf16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp32", "tomos_fp32"),
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

    # scale_tag combines scale with "_normalized" when --normalized is
    # set (plot_gemv_fp32_recon_l2_error_vs_cublas_ref.m's own tag
    # mechanism appends "_<tag>" to the filename, so this produces
    # ..._scale<N>_normalized.png/.fig -- landing alongside, not
    # overwriting, the plain run's own ..._scale<N>.png/.fig).
    SCALE_TAG="${scale}${NORM_SUFFIX}"

    echo
    echo "==> Plotting (normal_scale=$scale, normalized=$NORMALIZED): $CSV_LOG"
    if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load matlab/R2024a 2>/dev/null || true
    fi
    if command -v matlab >/dev/null 2>&1; then
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

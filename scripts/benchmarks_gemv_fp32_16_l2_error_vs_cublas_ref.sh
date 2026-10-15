#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu
# and runs the SAME square sweep (M=K) mv_fp32_16_pure_half_vs_cublas_ref.cu
# itself uses, but measuring L2-relative-error against a genuine FP32
# (cublasSgemv) reference instead of throughput -- for the same five
# "TOMOS" 16-bit-storage GEMV series scripts/plot_gemv_fp32_16_pure_half_
# vs_cublas_ref.m plots: TOMOS FP16, TOMOS FP32/FP16, TOMOS FP32/FP16
# (fast), TOMOS FP32/BF16, TOMOS BF16. See the .cu's own header comment
# for exactly how each method's own reference is generated (device-side
# only, to reach the same 131072x131072 scale the performance sweep does
# without the host-memory blowup a full-matrix CPU reference would hit).
#
# Square sweep only, same list as mv_fp32_16_pure_half_vs_cublas_ref.cu's
# own SQUARE_MS: 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
# 32768/49152/65536/131072. No tall-and-thin sweep here -- add one later
# if the square results motivate it.
#
# Runs the FULL square sweep once per entry in SCALES below -- each is a
# "normal_scale" passed to the .cu binary (see its own header comment),
# widening every method's own "normal"-range test data from [-1,1] to
# [-scale,scale]. NOTE this stays far below fp16's own ~65504 per-VALUE
# ceiling on purpose: TOMOS FP16/FP32/FP16/FP32/FP16(fast) accumulate
# their row-sum in fp16 precision INTERNALLY, so it's the ACCUMULATED
# SUM across up to K=131072 terms that must stay under that ceiling, a
# much tighter bound than "does one value fit in fp16" -- see the SCALES
# array's own comment below for the real numbers (an earlier, far more
# aggressive attempt overflowed those three series into NaN/Inf at every
# shape). This scale is a separate axis from the hybrid-only extended/
# degraded-range test, which stays fixed regardless of it. Each tier
# writes its OWN timestamped .txt/.csv (scale embedded in the filename)
# and its OWN plot (scale embedded in both the output filename and the
# plot title) -- one plot per tier, not overlaid, so each is
# independently readable.
#
# --normalized (may appear anywhere among the arguments) scales x by
# 1/sqrt(K) at data-generation time instead of leaving it at plain
# [-1,1] (see the .cu's own run_shape comment) -- this DOES matter here,
# unlike this codebase's float-accumulating error benchmarks: three of
# this tier's five kernels accumulate their row-sum in __half2 registers
# internally, summing RAW products BEFORE alpha is ever applied, so only
# a data-generation-time scale (not alpha) can keep that accumulator
# from overflowing. With --normalized, the accumulated magnitude stays
# ~normal_scale regardless of K instead of growing like normal_scale*
# sqrt(K), so the SAME wide SCALES this codebase's float-accumulating
# rows use becomes safe here too -- see the SCALES array's own comment
# below for the default this switches to. Plots land with a
# "_normalized" tag (e.g. ..._scale4096_normalized.png) so they don't
# overwrite the plain run's own plots.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh
#       Runs the full default SQUARE_MS list at every default SCALES tier.
#   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...] -- M1 [M2 ...]
#       Runs exactly those scale tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SQUARE_MS list.
#   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh -- M1 [M2 ...]
#       Runs the default SCALES tiers against exactly those M values.
#   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh --normalized [--scales ... [-- ...]]
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
# one .csv, both under build/output/error_dimension_scaling/gemv/fp16_compute/data/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mv_fp32_16_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_dimension_scaling/gemv/fp16_compute/data"

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

echo "==> Building: mv_fp32_16_l2_error_vs_cublas_ref"
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

# Progressive magnitude-scale tiers -- powers of 2 (see the .cu's own
# gen_fp32_16_wh_safe_kernel comment for why), CAPPED FAR below fp16's own
# ~65504 per-VALUE ceiling on purpose: TOMOS FP16, TOMOS FP32/FP16, and
# TOMOS FP32/FP16 (fast) accumulate their row-sum in fp16/half2 precision
# INTERNALLY (not just fp16-STORED inputs), so it's the ACCUMULATED SUM
# across K terms that has to stay under that ceiling, not just each
# individual A[i,j] value -- a much tighter bound. For a row-sum of K
# terms each ~scale in magnitude, its own standard deviation is roughly
# scale*sqrt(K)/3 (confirmed empirically: scale=32768 overflowed EVERY
# row into NaN/Inf for all three fp16-accumulating series, even at the
# smallest shape here; scale=1024 overflowed only the larger shapes,
# M>24576 -- both real runs, not estimates). The worst case that actually
# matters is K=65536 (this SQUARE_MS list's largest shape that doesn't
# itself hit the OUT-OF-DEVICE-MEMORY ceiling -- M=K=131072 always gets
# SKIPPED regardless of scale, needing ~68.7GB for a single float32
# ground-truth array, so it never factors into this safety margin).
# scale=128 keeps that row-sum's own std ~6x (~6 sigma) below 65504 at
# K=65536 -- per-row overflow probability at 6 sigma is ~1e-9, and every
# smaller shape in this sweep has even more headroom, so no row should
# realistically overflow. TOMOS BF16/TOMOS FP32/BF16 accumulate in
# float (no such ceiling), so this cap is conservative for them
# specifically, but the SAME scale is used for a fair, apples-to-apples
# comparison across all five series.
#
# --normalized switches to a WIDER default: x is generated at scale
# 1/sqrt(K) instead of plain [-1,1] (see the .cu's own run_shape
# comment), so the accumulated row-sum std becomes ~scale/3 -- INDEPENDENT
# of K -- instead of ~scale*sqrt(K)/3. At scale=16384: std=16384/3~5461,
# z=65504/5461~12.0 sigma at EVERY shape (not just the smallest), a wider
# safety margin than the un-normalized scale=128 tier achieves even at
# its own worst case -- so the full original wide range becomes safe
# again, same list scripts/benchmarks_gemv_fp32_recon_l2_error_vs_
# cublas_ref.sh's own (float-accumulating, no overflow concern either
# way) SCALES already uses.
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
    TXT_LOG="$OUTPUT_DIR/mv_fp32_16_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mv_fp32_16_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_16_l2_error_vs_cublas_ref $scale $m${NORM_SUFFIX:+ (normalized)}"
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

# Longest/most-specific marker first (tomos_fp32_fp16_fast before
# tomos_fp32_fp16 before tomos_fp16, etc.) -- defensive, matching every
# other benchmark script's own convention in this codebase, though none
# of these seven actually collide as substrings of one another (each
# diverges before its own trailing " =") -- EXCEPT tomos_fp16_native, which
# (and tomos_fp16_native_mixed, which contains tomos_fp16_native) does contain
# tomos_fp16 as a substring and therefore MUST stay listed before it (tomos_fp16 is the packed FP32_16 wh-only kernel, relabeled
# "TOMOS FP16 (implicit)" in the plot; tomos_fp16_native is the native-fp16-
# input kernel the performance plots call "TOMOS FP16").
METHOD_MARKERS = [
    ("tomos_fp32_fp16_fast", "tomos_fp32_fp16_fast"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp16_narrowvec_flush256_swar", "tomos_fp16_narrowvec_flush256_swar"),
    ("tomos_fp16_narrowvec_flush256", "tomos_fp16_narrowvec_flush256"),
    ("tomos_fp16_narrowvec_flush64", "tomos_fp16_narrowvec_flush64"),
    ("tomos_fp16_narrowvec", "tomos_fp16_narrowvec"),
    ("tomos_fp16_native_mixed", "tomos_fp16_native_mixed"),
    ("tomos_fp16_native", "tomos_fp16_native"),
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
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

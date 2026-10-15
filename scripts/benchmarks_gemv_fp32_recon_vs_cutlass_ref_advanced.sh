#!/bin/bash
# "_advanced" variant of scripts/benchmarks_gemv_fp32_recon_vs_cutlass_
# ref.sh -- builds benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_
# cutlass_ref_advanced.cu (byte-for-byte identical to its own sibling
# .cu except for a caller-controlled warmup/iteration count instead of a
# fixed, shape-size-dependent `reps` heuristic -- see that file's own
# header) instead of the plain mv_fp32_recon_vs_cutlass_ref.cu, and runs
# the EXACT SAME two sweeps (square + tall-and-thin) that script does --
# see that script's own header comment for the full shape-list reasoning,
# not repeated here.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref_advanced.sh
#       Runs both sweeps with 10 warmup runs and 10 timed-and-averaged
#       iterations per shape (the defaults).
#   scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref_advanced.sh --warmup 20 --iters 50
#       Same sweeps, 20 warmup runs and 50 timed-and-averaged iterations
#       per shape instead.
#
# CUTLASS is fetched (shallow clone, header-only) into CUTLASS_DIR on
# first run and reused after that, same as the sibling script -- override
# with the CUTLASS_DIR env var if you already have a checkout.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 -- if SLURM_JOB_ID isn't already set, this script
# finds the caller's running job on an a100* partition and re-execs
# itself inside it via `srun --overlap`, same as the sibling script.
#
# All shapes' full console output is saved to one timestamped .txt log
# under build/mv_fp32_recon_vs_cutlass_ref_advanced_output/ (a SEPARATE
# directory from the sibling script's own output, so a regular and an
# advanced run never collide), parsed into one .csv (same schema as the
# sibling script's own CSV -- the .cu's own printed method lines are
# unaffected by the timing-methodology change). Plots are written with
# an "_avg" tag (see plot_gemv_fp32_recon_vs_cutlass_ref.m's own `tag`
# parameter), so they land ALONGSIDE the sibling script's own plots
# without overwriting them:
#   build/benchmarks/output/plots/plot_gemv_fp32_recon_vs_cutlass_ref_square_avg.png / .fig
#   build/benchmarks/output/plots/plot_gemv_fp32_recon_vs_cutlass_ref_tall_avg.png   / .fig

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref_advanced.cu"
OUT="$BUILD_DIR/mv_fp32_recon_vs_cutlass_ref_advanced"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_recon_vs_cutlass_ref_advanced_output"

# --warmup N / --iters N (both optional, default 10/10) -- this script
# takes no other arguments (same as its own sibling), so no interleaving
# with other flag parsing is needed.
WARMUP=10
ITERS=10
while [ "$#" -gt 0 ]; do
    case "$1" in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        *) echo "error: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" --warmup "$WARMUP" --iters "$ITERS"
fi

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_recon_vs_cutlass_ref_advanced"
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_recon_vs_cutlass_ref_advanced_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_recon_vs_cutlass_ref_advanced_${TIMESTAMP}.csv"

# Same shape lists as scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh's own.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)
TALL_KS=(512 4096)
TALL_MS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

: > "$TXT_LOG"
for m in "${SQUARE_MS[@]}"; do
    echo
    echo "==> Running mv_fp32_recon_vs_cutlass_ref_advanced $m (warmup=$WARMUP iters=$ITERS)"
    "$OUT" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
done
for k in "${TALL_KS[@]}"; do
    for m in "${TALL_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_recon_vs_cutlass_ref_advanced --tall $k $m (warmup=$WARMUP iters=$ITERS)"
        "$OUT" --tall "$k" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
    done
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Identical to scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh's own
# METHOD_MARKERS -- the .cu's own printed method lines are unaffected by
# the timing-methodology change (only the ms/GFlop/s VALUES differ, not
# the marker text), see that script's own comment for the full per-
# marker collision-safety reasoning.
METHOD_MARKERS = [
    ("mv_fp32_tiled", "native"),
    ("mv_fp32_16_accurate_workspace =", "fp32_16_ws"),
    ("mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal =", "fp32_16_regular"),
    ("mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw =", "fp32_16_test"),
    ("mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock =", "fp32_16"),
    ("mv_fp32_bf16_accurate_workspace =", "fp32_bf16_ws"),
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace =", "fp32_bf16_truncated"),
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded =", "fp32_bf16"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemv)", "cublas"),
]

FIELDS = [
    "M", "K",
    "native_ms", "native_gflops",
    "fp32_16_ms", "fp32_16_gflops",
    "fp32_16_test_ms", "fp32_16_test_gflops",
    "fp32_16_ws_ms", "fp32_16_ws_gflops",
    "fp32_16_regular_ms", "fp32_16_regular_gflops",
    "fp32_bf16_ms", "fp32_bf16_gflops",
    "fp32_bf16_truncated_ms", "fp32_bf16_truncated_gflops",
    "fp32_bf16_ws_ms", "fp32_bf16_ws_gflops",
    "cutlass_ms", "cutlass_gflops",
    "cublas_ms", "cublas_gflops",
]

rows = []
row = None


def flush():
    if row is not None:
        rows.append(row)


with open(txt_path) as f:
    for line in f:
        line = line.rstrip("\n")
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
                row[f"{key}_ms"] = "SKIPPED"
                row[f"{key}_gflops"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_ms"] = vm.group(1)
                    row[f"{key}_gflops"] = vm.group(2)
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
echo "==> Plotting (avg): $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref('$CSV_LOG', 'avg')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref('$CSV_LOG', 'avg')\"" >&2
fi

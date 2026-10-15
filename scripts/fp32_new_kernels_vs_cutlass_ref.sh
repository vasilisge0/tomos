#!/bin/bash
# Trimmed-down sibling of scripts/fp32_recon_vs_cutlass_ref.sh: builds and
# runs benchmarks/kernels/cuda/matrix/fp32_new_kernels_vs_cutlass_ref.cu,
# which times ONLY the kernels added by the warptile/snake/midpoint_exact
# investigation (and their immediate baselines for context) instead of
# benchmarks_gemm_fp32.cu's full ~40-kernel sweep -- for fast
# iteration when only these are relevant.
#
# Kernels compared, one process/GPU-state per run:
#   Native fp32: gemm_fp32_tiled_syncpipe3 (original baseline), gemm_
#     fp32_tiled_syncpipe4_bk8_warptile_nostride_snake (first kernel to
#     beat syncpipe3), gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
#     snake_midpoint_exact (current best, "native").
#   Packed fp32_16: mm_fp32_16_tiled_cpasync_dualbuf (original fused
#     reference), mm_fp32_16_gemm_materialize (previous best), mm_fp32_16_
#     gemm_materialize_midpoint_exact (materialize's 2nd stage swapped to
#     midpoint_exact), mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_
#     midpoint_exact (genuinely fused counterpart).
#   Plus real CUTLASS (fp32 SIMT device::Gemm) and cuBLAS Sgemm for both
#   families to compare against.
#
# See benchmarks_gemm_fp32.cu's own file-level comment for the full
# investigation writeup this trims down from.
#
# Usage:
#   scripts/fp32_new_kernels_vs_cutlass_ref.sh [N]
#
# With no args, sweeps 1024/2048/4096/8192/16384/32768/48000/54000, then a
# rectangular K-sweep at M=N=44000 -- identical sizes to fp32_recon_vs_
# cutlass_ref.sh, so results are directly comparable size-for-size. Pass a
# single N to benchmark just that size:
#   scripts/fp32_new_kernels_vs_cutlass_ref.sh 8192
# Or --rect [K] [outer] for just the rectangular sweep/shape, same as
# fp32_recon_vs_cutlass_ref.sh.
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp32_new_kernels_vs_cutlass_ref.sh
#
# Rebuilds tomos_cuda first (so this always reflects the current state
# of the kernels, not a stale libtomos_cuda.so) and links the standalone
# nvcc build against it directly -- CUTLASS itself is still NOT a
# dependency of tomos's own CMake build, only of this one comparison
# binary.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is
# only reachable inside an active job allocation. If SLURM_JOB_ID isn't
# already set, this script finds the caller's running job on an a100*
# partition and re-execs itself inside it via `srun --overlap`, so both
# the build and the run always land on the actual target GPU.
#
# Every run's full console output is saved to a timestamped .txt log, and
# a per-N summary (ms + GFlop/s for each method) is parsed out into a
# .csv, both under build/fp32_new_kernels_vs_cutlass_ref_output/ (a
# separate output directory from fp32_recon_vs_cutlass_ref.sh's, so runs
# from either script never collide).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/fp32_new_kernels_vs_cutlass_ref.cu"
OUT="$BUILD_DIR/fp32_new_kernels_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp32_new_kernels_vs_cutlass_ref_output"

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

echo "==> Building: fp32_new_kernels_vs_cutlass_ref"
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
TXT_LOG="$OUTPUT_DIR/fp32_new_kernels_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_new_kernels_vs_cutlass_ref_${TIMESTAMP}.csv"

echo
echo "==> Running fp32_new_kernels_vs_cutlass_ref $*"
"$OUT" "$@" | tee "$TXT_LOG"

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    # Longest/most-specific strings checked first -- "mm_fp32_16_gemm_
    # materialize" is a prefix of "..._midpoint_exact", and "gemm_fp32_
    # tiled_syncpipe4_bk8_warptile_nostride_snake" is a prefix of its own
    # "..._midpoint_exact (native)" line -- same ordering rule
    # fp32_recon_vs_cutlass_ref.sh's own METHOD_MARKERS uses.
    ("mm_fp32_16_gemm_materialize_midpoint_exact", "fp32_16_gemm_materialize_midpoint_exact"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_16_fused_midpoint_exact"),
    ("mm_fp32_16_gemm_materialize", "fp32_16_gemm_materialize"),
    ("mm_fp32_16_tiled_cpasync_dualbuf", "fp32_16_dualbuf"),
    ("gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native)", "native"),
    ("gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake", "wtnsnake"),
    ("gemm_fp32_tiled_syncpipe3", "syncpipe3"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemm)", "cublas"),
]

FIELDS = [
    "M", "K", "N",
    "fp32_16_dualbuf_ms", "fp32_16_dualbuf_gflops",
    "fp32_16_gemm_materialize_ms", "fp32_16_gemm_materialize_gflops",
    "fp32_16_gemm_materialize_midpoint_exact_ms", "fp32_16_gemm_materialize_midpoint_exact_gflops",
    "fp32_16_fused_midpoint_exact_ms", "fp32_16_fused_midpoint_exact_gflops",
    "syncpipe3_ms", "syncpipe3_gflops",
    "wtnsnake_ms", "wtnsnake_gflops",
    "native_ms", "native_gflops",
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
            row = {"M": m.group(1), "K": m.group(2), "N": m.group(3)}
            continue
        if row is None:
            continue

        if "native/cutlass/cublas (A)" in line and "SKIPPED" in line:
            for key in ("syncpipe3", "wtnsnake", "native", "cutlass", "cublas"):
                row[f"{key}_ms"] = "SKIPPED"
                row[f"{key}_gflops"] = "SKIPPED"
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

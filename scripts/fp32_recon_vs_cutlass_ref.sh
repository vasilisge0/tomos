#!/bin/bash
# Builds and runs a focused benchmark comparing the "full fp32
# reconstruction" tier of both packed-precision families -- mm_fp32_16_
# tiled_cpasync_dualbuf and mm_fp32_bf16_tiled_cpasync_dualbuf -- against
# this codebase's own best native (unpacked) fp32 GEMM (gemm_fp32_tiled_
# syncpipe4_bk8_warptile_nostride_snake_midpoint_exact), real CUTLASS's
# fp32 SIMT device::Gemm, and cuBLAS's
# Sgemm, all in one run so the numbers share the same GPU clock/thermal/
# contention state
# (benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu).
#
# The underlying .cu binary also times several newer kernels not
# mentioned above (the fp32_16/fp32_bf16 fused and materialize
# midpoint_exact-based kernels -- see that file's own header comment for
# the full list); this script's embedded CSV parser captures all of
# them under the SAME column names scripts/benchmarks_gemm_format_32_16_
# compute_32.sh and scripts/fp32_16_plotted_vs_cutlass_ref.sh use, so
# scripts/plot_gemm_format_32_16_compute_32.m can plot whichever
# script's CSV happens to be picked up from the shared build/fp32_
# recon_vs_cutlass_ref_output/ directory without a missing-column error.
#
# Mirrors scripts/wide_bigreg_sq_fp32_16_vs_cutlass_ref.sh's structure.
#
# Usage:
#   scripts/fp32_recon_vs_cutlass_ref.sh [N]
#
# With no args, sweeps 1024/2048/4096/8192/16384/32768/48000/54000. Pass a
# single N to benchmark just that size:
#   scripts/fp32_recon_vs_cutlass_ref.sh 8192
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp32_recon_vs_cutlass_ref.sh
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
# .csv, both under build/output/benchmarks/gemm/fp32_compute/data/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu"
OUT="$BUILD_DIR/fp32_recon_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/benchmarks/gemm/fp32_compute/data"

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

echo "==> Building: fp32_recon_vs_cutlass_ref"
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
TXT_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_recon_vs_cutlass_ref_${TIMESTAMP}.csv"

echo
echo "==> Running fp32_recon_vs_cutlass_ref $*"
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
    # The truncated/adaptive_warp/adaptive/multin markers MUST come
    # before the plain dualbuf one, AND adaptive_warp MUST come before
    # plain adaptive -- "mm_fp32_16_tiled_cpasync_dualbuf" is a prefix
    # of "..._truncated", "..._adaptive", and "..._multin", and
    # "..._adaptive" is itself a prefix of "..._adaptive_warp"; this
    # loop takes the first match, so checking the longest/most specific
    # strings first avoids misfiling. "mm_fp32_16_gemm_materialize"
    # shares no prefix with anything else here, so its position doesn't
    # matter beyond being a marker at all. Order between truncated,
    # multin, and the two adaptive markers doesn't matter among
    # themselves -- none is a substring of another.
    ("mm_fp32_16_tiled_cpasync_dualbuf_truncated", "fp32_16_dualbuf_trunc"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp", "fp32_16_dualbuf_adaptive_warp"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_adaptive", "fp32_16_dualbuf_adaptive"),
    ("mm_fp32_16_tiled_cpasync_dualbuf_multin", "fp32_16_dualbuf_multin"),
    # "mm_fp32_16_gemm_materialize_midpoint_exact" MUST come before the
    # plain "mm_fp32_16_gemm_materialize" marker below -- the former is a
    # superstring of the latter, and this loop takes the first match.
    ("mm_fp32_16_gemm_materialize_midpoint_exact", "fp32_16_gemm_materialize_midpoint_exact"),
    ("mm_fp32_16_gemm_materialize", "fp32_16_gemm_materialize"),
    # Same superstring rule: the truncated fused midpoint_exact marker
    # MUST come before the plain one below.
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated", "fp32_16_fused_midpoint_exact_trunc"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_16_fused_midpoint_exact"),
    ("mm_fp32_16_tiled_cpasync_dualbuf", "fp32_16_dualbuf"),
    # The fp32_bf16 counterpart to the fp32_16 fused midpoint_exact
    # kernel above -- unlike fp32_16, this one beats dualbuf below
    # outright at every measured size (see plot_gemm_format_32_16_
    # compute_32.m's own comment on the TOMOS FP32 / BF16 series). No
    # prefix relation with the dualbuf marker below.
    # "_rounded" MUST come before the bare fused marker below -- it's a
    # superstring of it.
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded", "fp32_bf16_fused_rounded"),
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_bf16_fused_midpoint_exact"),
    ("mm_fp32_bf16_gemm_materialize_rounded", "fp32_bf16_materialize_rounded"),
    ("mm_fp32_bf16_tiled_cpasync_dualbuf", "fp32_bf16_dualbuf"),
    ("gemm_fp32_tiled_cpasync_dualbuf_regpipe", "native_regpipe"),
    # "native" is gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
    # midpoint_exact (promoted from gemm_fp32_tiled_syncpipe3, itself
    # promoted from gemm_fp32_tiled_cpasync_dualbuf -- see fp32_recon_vs_
    # cutlass_ref.cu's own file-level comment for why). The printed line
    # is "gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_
    # exact (native) = ...ms (...GFlop/s)"; matching on the full string
    # including " (native)" (not just the bare kernel name) is required
    # here because the bare name is also a prefix of the plain
    # (non-native) timing line for the same kernel further down the same
    # run's output.
    ("gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native)", "native"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemm)", "cublas"),
]

FIELDS = [
    "M", "K", "N",
    "fp32_16_dualbuf_ms", "fp32_16_dualbuf_gflops",
    "fp32_16_dualbuf_trunc_ms", "fp32_16_dualbuf_trunc_gflops",
    "fp32_16_dualbuf_adaptive_ms", "fp32_16_dualbuf_adaptive_gflops",
    "fp32_16_dualbuf_adaptive_warp_ms", "fp32_16_dualbuf_adaptive_warp_gflops",
    "fp32_16_dualbuf_multin_ms", "fp32_16_dualbuf_multin_gflops",
    "fp32_16_gemm_materialize_ms", "fp32_16_gemm_materialize_gflops",
    "fp32_16_gemm_materialize_midpoint_exact_ms", "fp32_16_gemm_materialize_midpoint_exact_gflops",
    "fp32_16_fused_midpoint_exact_ms", "fp32_16_fused_midpoint_exact_gflops",
    "fp32_16_fused_midpoint_exact_trunc_ms", "fp32_16_fused_midpoint_exact_trunc_gflops",
    "fp32_bf16_dualbuf_ms", "fp32_bf16_dualbuf_gflops",
    "fp32_bf16_fused_midpoint_exact_ms", "fp32_bf16_fused_midpoint_exact_gflops",
    "fp32_bf16_fused_rounded_ms", "fp32_bf16_fused_rounded_gflops",
    "fp32_bf16_materialize_rounded_ms", "fp32_bf16_materialize_rounded_gflops",
    "native_ms", "native_gflops",
    "native_regpipe_ms", "native_regpipe_gflops",
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
            for key in ("native", "cutlass", "cublas"):
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

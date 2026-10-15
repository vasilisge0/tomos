#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_vs_cublas_ref.cu
# and runs two custom sweeps -- the NO_TRANSPOSE (y = alpha*A*x + beta*y)
# mirror of benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh,
# comparing the same final four-series "TOMOS" set that script's own plot
# settled on: TOMOS FP16 (mv_fp32_16_pure_half_wide_ilp_shared_half2),
# TOMOS FP32/FP16 (mv_fp32_16_hybrid_wide_ilp_shared), TOMOS FP32/FP16
# (fast) (mv_fp32_16_pure_half_wide_ilp_shared_fasthalf), TOMOS FP32/BF16
# (mv_fp32_bf16_pure_bf16_wide_ilp_shared), plus cuBLAS (Hgemm n=1, the
# real fp16 GEMV baseline) and cuBLAS BF16 (GemmEx CUDA_R_16BF n=1, the
# real bf16 GEMV baseline) -- see the .cu's own header comment for why
# these, not cublasSgemv. Plus one "test" leg (mv_fp32_16_pure_half_wide_
# ilp_shared_branchless): same safe decode as TOMOS FP16 but branchless
# (bit-exact via fp32_16_half_only_branchless), answering whether removing
# just the decode branch recovers any of "(fast)"'s own measured gain.
#
# Same two-sweep structure and shape lists as the TRANSPOSE script:
#   - Square (M=K): 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
#     32768/49152/65536/131072.
#   - Tall-and-thin: K held fixed at 512, then 4096; M swept across
#     1024/2048/4096/8192/16384/32768/65536/131072/262144/524288/1048576.
# All shapes' output (both sweeps) is combined into one timestamped .txt
# log and parsed into one .csv, same convention as every other benchmark
# script in this codebase.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh
#       Runs both sweeps above (default SQUARE_MS/TALL_KS/TALL_MS lists).
#   scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh --square-only [M1 M2 ...]
#       Runs ONLY the square sweep, skipping tall-and-thin entirely. With
#       no M values given, uses the default SQUARE_MS list below; pass one
#       or more M values to run exactly those instead.
#   scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh --tall-only [K M1 M2 ...]
#       Runs ONLY the tall-and-thin sweep, skipping square entirely. With
#       no arguments, uses the default TALL_KS x TALL_MS grid below; pass a
#       single K followed by one or more M values to run exactly that one K
#       against just those M's instead.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly.
#
# All shapes' full console output (both sweeps) is saved to one timestamped
# .txt log, and a per-shape summary (ms + GFlop/s for each method) is
# parsed out into one .csv, both under build/mv_fp32_16_pure_half_vs_
# cublas_ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mv_fp32_16_pure_half_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_16_pure_half_vs_cublas_ref_output"

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

echo "==> Building: mv_fp32_16_pure_half_vs_cublas_ref"
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_vs_cublas_ref_${TIMESTAMP}.csv"

# Square sweep (M=K) -- same list as the TRANSPOSE script's own SQUARE_MS.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)

# Tall-and-thin sweep: K held fixed at each of these, M swept across the
# same large-size list below -- same lists as the TRANSPOSE script's own
# TALL_KS/TALL_MS.
TALL_KS=(512 4096)
TALL_MS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# --square-only / --tall-only (see this script's own header comment) --
# both default to false, meaning "run both sweeps".
RUN_SQUARE=1
RUN_TALL=1
if [ "${1:-}" = "--square-only" ]; then
    RUN_TALL=0
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_MS=("$@")
    fi
elif [ "${1:-}" = "--tall-only" ]; then
    RUN_SQUARE=0
    shift
    if [ "$#" -gt 0 ]; then
        TALL_KS=("$1")
        shift
        TALL_MS=("$@")
    fi
fi

: > "$TXT_LOG"
if [ "$RUN_SQUARE" -eq 1 ]; then
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_16_pure_half_vs_cublas_ref $m"
        "$OUT" "$m" | tee -a "$TXT_LOG"
    done
fi
if [ "$RUN_TALL" -eq 1 ]; then
    for k in "${TALL_KS[@]}"; do
        for m in "${TALL_MS[@]}"; do
            echo
            echo "==> Running mv_fp32_16_pure_half_vs_cublas_ref --tall $k $m"
            "$OUT" --tall "$k" "$m" | tee -a "$TXT_LOG"
        done
    done
fi

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Marker -> CSV field prefix. No marker here is a substring of another
# (checked by inspection -- each diverges before its own trailing "="/
# directive text, e.g. "half2 =" vs "half2_branchless =" differ right
# after "half2"), so the first-match-wins `break` below never actually
# has to arbitrate a real ambiguity.
METHOD_MARKERS = [
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless =", "pure_half_half2_branchless"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow =", "pure_half_half2_vecrow"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows =", "pure_half_half2_vec2rows"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x =", "pure_half_half2_wide2x"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x =", "pure_half_half2_wide4x"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive =", "bigtile_half2_adaptive"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2 =", "bigtile_half2"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_best =", "half2_best"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit =", "half2_atomic_lowsplit"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_split =", "half2_split"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2 =", "half2_atomic_prefetch2"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar =", "half2_atomic_cpasync_warpuniform_swar"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform =", "half2_atomic_cpasync_warpuniform"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync =", "half2_atomic_cpasync"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_combine_best =", "half2_combine_best"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch =", "half2_atomic_prefetch"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic =", "half2_atomic"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic =", "bigtile_half2_atomic"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x =", "half2_atomic_wide2x"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x =", "half2_atomic_wide4x"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best =", "half2_atomic_best"),
    ("mv_fp32_16_pure_half_nativeaccum =", "nativeaccum"),
    ("mv_fp32_16_pure_half_no_transpose_fast2 =", "noT_fast2"),
    ("mv_fp32_16_pure_half_no_transpose_flush64 =", "noT_flush64"),
    ("mv_fp32_16_pure_half_no_transpose_flush256 =", "noT_flush256"),
    ("mv_fp32_16_pure_half_no_transpose_flush256_swar =", "noT_flush256_swar"),
    ("mv_fp32_16_pure_half_no_transpose_flush256_fast_swar =", "noT_flush256_fast_swar"),
    ("mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best =", "half2_atomic_best_fast"),
    ("mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best (realistic ~18% underflow) =",
     "half2_atomic_best_fast2_realistic_underflow"),
    ("mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best =", "half2_atomic_best_fast2"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform (realistic ~18% underflow) =",
     "half2_atomic_cpasync_warpuniform_realistic_underflow"),
    ("mv_fp32_16_pure_half_test_new_dispatch (test-new, realistic ~18% underflow) =",
     "test_new"),
    ("mv_fp32_16_pure_half_test_new_dispatch (test-new2, uniform ~0% underflow) =",
     "test_new2"),
    ("mv_fp32_16_pure_half_test_new3_dispatch (test-new3, rows/thread=2, uniform ~0% underflow) =",
     "test_new3"),
    ("mv_fp32_16_pure_half_test_new4_dispatch (test-new4, rows/thread=8, uniform ~0% underflow) =",
     "test_new4"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed =", "half2_atomic_mixed"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2 =", "pure_half_half2"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_branchless =", "pure_half_branchless"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic =", "hybrid_bigtile_atomic"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x =", "hybrid_atomic_wide4x"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best =", "hybrid_atomic_best"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx =", "hybrid_branchless_fastx_bigtile"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x =", "hybrid_branchless_fastx_wide4x"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best =", "hybrid_branchless_fastx_best"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx =", "hybrid_branchless_fastx"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless =", "hybrid_branchless_bigtile"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x =", "hybrid_branchless_wide4x"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best =", "hybrid_branchless_best"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless =", "hybrid_atomic_branchless"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask =", "hybrid_branchless_intmask_bigtile"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x =", "hybrid_branchless_intmask_wide4x"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best =", "hybrid_branchless_intmask_best"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best =", "hybrid_branchless_intmask_walign_best"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign =", "hybrid_branchless_intmask_walign"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask =", "hybrid_branchless_intmask"),
    ("mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic =", "hybrid_atomic"),
    ("mv_fp32_16_hybrid_wide_ilp_shared =", "hybrid"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf =", "fasthalf_bigtile_atomic"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x =", "fasthalf_atomic_wide4x"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best =", "fasthalf_atomic_best"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf =", "fasthalf_atomic"),
    ("mv_fp32_16_pure_half_wide_ilp_shared_fasthalf =", "pure_half_fasthalf"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic =", "bf16_bigtile_atomic"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x =", "bf16_atomic_wide4x"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best =", "bf16_atomic_best"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic =", "bf16_atomic"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2 =", "bf16_bigtile_atomic_v2"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2 =", "bf16_atomic_wide4x_v2"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2 =", "bf16_atomic_best_v2"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2 =", "bf16_atomic_v2"),
    ("mv_fp32_bf16_pure_bf16_wide_ilp_shared =", "fp32_bf16_pure_bf16"),
    ("mv_generic_fp16_double_acc =", "double_acc"),
    ("mv_fp16_native_storage_fastequiv_wide_ilp_shared_half2_atomic_best =", "native_storage"),
    ("mv_fp16_native_mixed =", "native_mixed"),
    ("cublas (Hgemm n=1, NO_TRANSPOSE)", "cublas_hgemm"),
    ("cublas (GemmEx CUDA_R_16BF n=1, NO_TRANSPOSE)", "cublas_gemmex_bf16"),
]

FIELDS = [
    "M", "K",
    "pure_half_half2_ms", "pure_half_half2_gflops",
    "pure_half_branchless_ms", "pure_half_branchless_gflops",
    "pure_half_half2_branchless_ms", "pure_half_half2_branchless_gflops",
    "pure_half_half2_vecrow_ms", "pure_half_half2_vecrow_gflops",
    "pure_half_half2_vec2rows_ms", "pure_half_half2_vec2rows_gflops",
    "pure_half_half2_wide2x_ms", "pure_half_half2_wide2x_gflops",
    "pure_half_half2_wide4x_ms", "pure_half_half2_wide4x_gflops",
    "bigtile_half2_ms", "bigtile_half2_gflops",
    "bigtile_half2_adaptive_ms", "bigtile_half2_adaptive_gflops",
    "half2_best_ms", "half2_best_gflops",
    "half2_atomic_ms", "half2_atomic_gflops",
    "half2_atomic_lowsplit_ms", "half2_atomic_lowsplit_gflops",
    "half2_split_ms", "half2_split_gflops",
    "half2_atomic_prefetch_ms", "half2_atomic_prefetch_gflops",
    "half2_atomic_prefetch2_ms", "half2_atomic_prefetch2_gflops",
    "half2_atomic_cpasync_ms", "half2_atomic_cpasync_gflops",
    "half2_atomic_cpasync_warpuniform_ms", "half2_atomic_cpasync_warpuniform_gflops",
    "half2_atomic_cpasync_warpuniform_swar_ms", "half2_atomic_cpasync_warpuniform_swar_gflops",
    "half2_atomic_cpasync_warpuniform_realistic_underflow_ms", "half2_atomic_cpasync_warpuniform_realistic_underflow_gflops",
    "test_new_ms", "test_new_gflops",
    "test_new2_ms", "test_new2_gflops",
    "test_new3_ms", "test_new3_gflops",
    "test_new4_ms", "test_new4_gflops",
    "half2_combine_best_ms", "half2_combine_best_gflops",
    "bigtile_half2_atomic_ms", "bigtile_half2_atomic_gflops",
    "half2_atomic_wide2x_ms", "half2_atomic_wide2x_gflops",
    "half2_atomic_wide4x_ms", "half2_atomic_wide4x_gflops",
    "half2_atomic_best_ms", "half2_atomic_best_gflops",
    "nativeaccum_ms", "nativeaccum_gflops",
    "noT_fast2_ms", "noT_fast2_gflops",
    "noT_flush64_ms", "noT_flush64_gflops",
    "noT_flush256_ms", "noT_flush256_gflops",
    "noT_flush256_swar_ms", "noT_flush256_swar_gflops",
    "noT_flush256_fast_swar_ms", "noT_flush256_fast_swar_gflops",
    "half2_atomic_best_fast_ms", "half2_atomic_best_fast_gflops",
    "half2_atomic_best_fast2_ms", "half2_atomic_best_fast2_gflops",
    "half2_atomic_best_fast2_realistic_underflow_ms", "half2_atomic_best_fast2_realistic_underflow_gflops",
    "half2_atomic_mixed_ms", "half2_atomic_mixed_gflops",
    "hybrid_ms", "hybrid_gflops",
    "hybrid_atomic_ms", "hybrid_atomic_gflops",
    "hybrid_bigtile_atomic_ms", "hybrid_bigtile_atomic_gflops",
    "hybrid_atomic_wide4x_ms", "hybrid_atomic_wide4x_gflops",
    "hybrid_atomic_best_ms", "hybrid_atomic_best_gflops",
    "hybrid_atomic_branchless_ms", "hybrid_atomic_branchless_gflops",
    "hybrid_branchless_bigtile_ms", "hybrid_branchless_bigtile_gflops",
    "hybrid_branchless_wide4x_ms", "hybrid_branchless_wide4x_gflops",
    "hybrid_branchless_best_ms", "hybrid_branchless_best_gflops",
    "hybrid_branchless_fastx_ms", "hybrid_branchless_fastx_gflops",
    "hybrid_branchless_fastx_bigtile_ms", "hybrid_branchless_fastx_bigtile_gflops",
    "hybrid_branchless_fastx_wide4x_ms", "hybrid_branchless_fastx_wide4x_gflops",
    "hybrid_branchless_fastx_best_ms", "hybrid_branchless_fastx_best_gflops",
    "hybrid_branchless_intmask_ms", "hybrid_branchless_intmask_gflops",
    "hybrid_branchless_intmask_bigtile_ms", "hybrid_branchless_intmask_bigtile_gflops",
    "hybrid_branchless_intmask_wide4x_ms", "hybrid_branchless_intmask_wide4x_gflops",
    "hybrid_branchless_intmask_best_ms", "hybrid_branchless_intmask_best_gflops",
    "hybrid_branchless_intmask_walign_ms", "hybrid_branchless_intmask_walign_gflops",
    "hybrid_branchless_intmask_walign_best_ms", "hybrid_branchless_intmask_walign_best_gflops",
    "pure_half_fasthalf_ms", "pure_half_fasthalf_gflops",
    "fasthalf_atomic_ms", "fasthalf_atomic_gflops",
    "fasthalf_bigtile_atomic_ms", "fasthalf_bigtile_atomic_gflops",
    "fasthalf_atomic_wide4x_ms", "fasthalf_atomic_wide4x_gflops",
    "fasthalf_atomic_best_ms", "fasthalf_atomic_best_gflops",
    "fp32_bf16_pure_bf16_ms", "fp32_bf16_pure_bf16_gflops",
    "bf16_atomic_ms", "bf16_atomic_gflops",
    "bf16_bigtile_atomic_ms", "bf16_bigtile_atomic_gflops",
    "bf16_atomic_wide4x_ms", "bf16_atomic_wide4x_gflops",
    "bf16_atomic_best_ms", "bf16_atomic_best_gflops",
    "bf16_atomic_v2_ms", "bf16_atomic_v2_gflops",
    "bf16_bigtile_atomic_v2_ms", "bf16_bigtile_atomic_v2_gflops",
    "bf16_atomic_wide4x_v2_ms", "bf16_atomic_wide4x_v2_gflops",
    "bf16_atomic_best_v2_ms", "bf16_atomic_best_v2_gflops",
    "double_acc_ms", "double_acc_gflops",
    "native_storage_ms", "native_storage_gflops",
    "native_mixed_ms", "native_mixed_gflops",
    "cublas_hgemm_ms", "cublas_hgemm_gflops",
    "cublas_gemmex_bf16_ms", "cublas_gemmex_bf16_gflops",
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
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG')\"" >&2
fi

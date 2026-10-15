#!/bin/bash
# TRANSPOSE mirror of scripts/benchmarks_gemv_fp32_16_pure_half_tall_
# 1000cols_vs_cublas_ref.sh -- same benchmarks/kernels/cuda/matrix/mv_
# fp32_16_pure_half_transpose_vs_cublas_ref.cu, same test cases, but
# scoped to a single tall-and-thin sweep at K=1000 columns, M swept
# across 100000 / 400000 / 800000 / 1000000 rows -- the real LSQR matrix
# shape (M=1,000,000, K=1000) this session's TRANSPOSE-tier debugging
# cares about. This .cu was just extended (this session) to also
# benchmark mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_
# mblock_vec4_split[_cpasync] -- the ACTUAL kernels mv_pure_half_impl's
# TRANSPOSE branch dispatches to at runtime (fp32_16_registries.cpp) --
# which the original transpose benchmark script never exercised (it only
# ever measured the plain safe/fast/retarget/retarget_lo variants, none
# of which are what LSQR actually calls). Written as a SEPARATE script so
# it always writes its own timestamped .txt/.csv/.png outputs and never
# collides with or overwrites the general-purpose script's own.
#
# Usage:
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_tall_1000cols_vs_cublas_ref.sh
#       Runs the default M sweep above at K=1000.
#   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_tall_1000cols_vs_cublas_ref.sh M1 [M2 ...]
#       Runs K=1000 against exactly the given M values instead.
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone
# nvcc build against it directly.
#
# Full console output is saved to one timestamped .txt log, and a
# per-shape summary (ms + GFlop/s for each method) is parsed out into one
# .csv, both under build/mv_fp32_16_pure_half_transpose_tall_1000cols_vs_
# cublas_ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_transpose_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref_output"

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

echo "==> Rebuilding tomos_cuda/tomos_core/tomos_cpu (so this reflects the current kernel/dispatch state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"
cmake --build "$BUILD_DIR" --target tomos_core -j"$(nproc)"
cmake --build "$BUILD_DIR" --target tomos_cpu -j"$(nproc)"

echo "==> Building: mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref"
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
    -L"$BUILD_DIR" -ltomos_core -ltomos_cpu -Xlinker -rpath -Xlinker "$BUILD_DIR" \
    -o "$OUT"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref_${TIMESTAMP}.csv"

# Tall-and-thin sweep: K fixed at 1000 columns, M swept across the real
# LSQR-shape-adjacent sizes this session cares about.
TALL_K=1000
TALL_MS=(100000 400000 800000 1000000)

if [ "$#" -gt 0 ]; then
    TALL_MS=("$@")
fi

: > "$TXT_LOG"
for m in "${TALL_MS[@]}"; do
    echo
    echo "==> Running mv_fp32_16_pure_half_transpose_tall_1000cols_vs_cublas_ref --tall $TALL_K $m"
    "$OUT" --tall "$TALL_K" "$m" | tee -a "$TXT_LOG"
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Marker -> CSV field prefix. Kept identical to scripts/benchmarks_gemv_
# transpose_fp32_16_pure_half_vs_cublas_ref.sh's own parser, PLUS the two
# fast2/fast2_cpasync markers this session just added to the .cu (see
# this script's own header comment).
METHOD_MARKERS = [
    ("mv_fp32_16_pure_half_wide_ilp_shared_half2 =", "pure_half_half2"),
    ("mv_fp32_16_pure_half   =", "pure_half"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4 =", "pure_half_atomiconly"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor =", "pure_half_atomiconly_retuned_floor"),
    ("mv_fp32_16_pure_half_transpose_wmma =", "pure_half_wmma"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor =",
     "pure_half_atomiconly_half_io_retuned_floor"),
    ("mv_fp32_16_pure_half_transpose_warp_owner_split_f32 =", "pure_half_warp_owner_split_f32"),
    ("mv_fp32_16_pure_half_transpose_warp_owner_split =", "pure_half_warp_owner_split"),
    ("mv_fp32_16_pure_half_transpose_warp_owner =", "pure_half_warp_owner"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo =",
     "pure_half_atomiconly_vec4_split_retarget_lo"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget =",
     "pure_half_atomiconly_vec4_split_retarget"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast =",
     "pure_half_atomiconly_vec4_split_fast"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split_cpasync =",
     "pure_half_fast2_atomiconly_vec4_split_cpasync"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split =",
     "pure_half_fast2_atomiconly_vec4_split"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split (~18% underflow, "
     "fast2/branching decode) =", "pure_half_fast2_underflow"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split (~18% underflow, "
     "branchless decode) =", "pure_half_branchless_underflow"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split (low underflow, "
     "branching decode) =", "pure_half_fast2_conditional_lowunderflow"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split =", "pure_half_fast2_vec2_split"),
    ("mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec8_split =", "pure_half_fast2_vec8_split"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split =", "pure_half_atomiconly_vec4_split"),
    ("mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split =",
     "pure_half_fast_atomiconly_vec4_split"),
    ("mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast =", "pure_half_atomiconly_vec4_fast"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired =", "hybrid_vec4_paired"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4 =", "hybrid_vec4"),
    ("mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock =", "hybrid"),
    ("mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4 =", "fp32_bf16_pure_bf16"),
    ("mv_fp16_native_storage   =", "native_storage"),
    ("mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split =", "native_mixed"),
    ("isolated_mv_apply(implicit, TRANSPOSE) =", "isolated_mv_apply_implicit"),
    ("isolated_mv_apply(explicit, TRANSPOSE) =", "isolated_mv_apply_explicit"),
    ("cublas (Hgemm n=1, TRANSPOSE)", "cublas_hgemm"),
    ("cublas (GemmEx CUDA_R_16BF n=1, TRANSPOSE)", "cublas_gemmex_bf16"),
]

FIELDS = [
    "M", "K",
    "pure_half_ms", "pure_half_gflops",
    "pure_half_half2_ms", "pure_half_half2_gflops",
    "pure_half_atomiconly_ms", "pure_half_atomiconly_gflops",
    "pure_half_atomiconly_retuned_floor_ms", "pure_half_atomiconly_retuned_floor_gflops",
    "pure_half_wmma_ms", "pure_half_wmma_gflops",
    "pure_half_atomiconly_half_io_retuned_floor_ms", "pure_half_atomiconly_half_io_retuned_floor_gflops",
    "pure_half_warp_owner_ms", "pure_half_warp_owner_gflops",
    "pure_half_warp_owner_split_ms", "pure_half_warp_owner_split_gflops",
    "pure_half_warp_owner_split_f32_ms", "pure_half_warp_owner_split_f32_gflops",
    "pure_half_atomiconly_vec4_split_ms", "pure_half_atomiconly_vec4_split_gflops",
    "pure_half_fast_atomiconly_vec4_split_ms", "pure_half_fast_atomiconly_vec4_split_gflops",
    "pure_half_fast2_atomiconly_vec4_split_ms", "pure_half_fast2_atomiconly_vec4_split_gflops",
    "pure_half_fast2_atomiconly_vec4_split_cpasync_ms", "pure_half_fast2_atomiconly_vec4_split_cpasync_gflops",
    "pure_half_fast2_underflow_ms", "pure_half_fast2_underflow_gflops",
    "pure_half_branchless_underflow_ms", "pure_half_branchless_underflow_gflops",
    "pure_half_fast2_conditional_lowunderflow_ms", "pure_half_fast2_conditional_lowunderflow_gflops",
    "pure_half_fast2_vec2_split_ms", "pure_half_fast2_vec2_split_gflops",
    "pure_half_fast2_vec8_split_ms", "pure_half_fast2_vec8_split_gflops",
    "pure_half_atomiconly_vec4_split_retarget_ms", "pure_half_atomiconly_vec4_split_retarget_gflops",
    "pure_half_atomiconly_vec4_split_retarget_lo_ms", "pure_half_atomiconly_vec4_split_retarget_lo_gflops",
    "pure_half_atomiconly_vec4_fast_ms", "pure_half_atomiconly_vec4_fast_gflops",
    "hybrid_ms", "hybrid_gflops",
    "hybrid_vec4_ms", "hybrid_vec4_gflops",
    "hybrid_vec4_paired_ms", "hybrid_vec4_paired_gflops",
    "pure_half_atomiconly_vec4_split_fast_ms", "pure_half_atomiconly_vec4_split_fast_gflops",
    "fp32_bf16_pure_bf16_ms", "fp32_bf16_pure_bf16_gflops",
    "native_storage_ms", "native_storage_gflops",
    "native_mixed_ms", "native_mixed_gflops",
    "isolated_mv_apply_implicit_ms", "isolated_mv_apply_implicit_gflops",
    "isolated_mv_apply_explicit_ms", "isolated_mv_apply_explicit_gflops",
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
echo "==> Key columns to check: pure_half_fast2_atomiconly_vec4_split_cpasync_gflops"
echo "    (the ACTUAL kernel LSQR's TRANSPOSE tier calls) vs cublas_hgemm_gflops"
echo "    and native_storage_gflops at M=1000000."

echo
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    # tag = 'lsqr_shape' so this writes its own plot_gemv_transpose_fp32_
    # 16_pure_half_vs_cublas_ref_tall_lsqr_shape.png/.fig instead of
    # overwriting the general-purpose script's own untagged tall plot.
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG', 'lsqr_shape')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref('$CSV_LOG', 'lsqr_shape')\"" >&2
fi

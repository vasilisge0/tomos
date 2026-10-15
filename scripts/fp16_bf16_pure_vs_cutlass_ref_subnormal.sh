#!/bin/bash
# Sibling of scripts/fp16_bf16_pure_vs_cutlass_ref.sh that runs the SAME
# binary (benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu)
# and the SAME square+rect sweeps (1024/2048/4096/8192/16384/32768/65536
# square, then M=N=44000 with K in 1024..32768 rectangular), but with A
# and B for the fp16 group (mm_fp32_16_pure_half_..., mm_fp32_16_pure_
# half_gemm_materialize_..., native fp16, real CUTLASS fp16, cuBLAS Hgemm
# -- everything sharing that group's own A/B pair) drawn from a generator
# with a TUNABLE fraction of entries forced into the fp16 subnormal band
# (0 < |x| < 2^-14), instead of the .cu's own plain uniform-on-[-1,1]
# pseudo_rand.
#
# Exists because the default uniform generator realizes essentially no
# subnormals (~0.006% of entries, matching a continuous uniform's own
# share of that tiny band) -- every existing GEMM performance plot in
# this codebase (this one included, at density 0) is measuring decode
# speed on data the branchless decode's underflow-handling code almost
# never actually executes. This script measures the same methods at a
# realistic-or-worse subnormal fraction instead, to see whether that
# changes the fp16-group kernels' relative throughput (the decode step
# itself is data-INDEPENDENT/branchless, so in principle it shouldn't --
# this is the test that confirms or refutes that expectation on real
# hardware, not just in the abstract).
#
# The bf16 group's inputs (Bb, wh16bf) are UNCHANGED by this script --
# bf16's own subnormal boundary (~2^-126) is astronomically smaller than
# fp16's, so a density calibrated to fp16's band is meaningless there
# (see benchmarks_gemm_fp16.cu's own run_shape comment).
#
# Usage:
#   scripts/fp16_bf16_pure_vs_cutlass_ref_subnormal.sh [--density PCT] [N | --rect [K [MN]]]
#   --density PCT   Percent of fp16-group A/B entries forced into the fp16
#                    subnormal band. Default: 10 (i.e. 10%).
#   With no shape argument, sweeps the same square+rect lists as scripts/
#   fp16_bf16_pure_vs_cutlass_ref.sh's own no-arg default. A single N runs
#   just that square shape; --rect [K [MN]] mirrors that script's own
#   rectangular single-shape mode.
#
# Examples:
#   scripts/fp16_bf16_pure_vs_cutlass_ref_subnormal.sh
#       Full square+rect sweep at the default 10% density.
#   scripts/fp16_bf16_pure_vs_cutlass_ref_subnormal.sh --density 30 8192
#       Just N=8192, square, at 30% density.
#
# CUTLASS is fetched (shallow clone, header-only) into CUTLASS_DIR on
# first run and reused after that, same as the other *_vs_cutlass_ref.sh
# scripts. Override with the CUTLASS_DIR env var if you already have one:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp16_bf16_pure_vs_cutlass_ref_subnormal.sh
#
# Rebuilds tomos_cuda first (so this reflects the current kernel state)
# and links the standalone nvcc build directly against it -- same
# discipline as every other *_vs_cutlass_ref.sh script.
#
# On Gilbreth the shell this runs in is usually the login node (no A100
# there) -- if SLURM_JOB_ID isn't already set, this re-execs itself
# inside the caller's own running a100* job via `srun --overlap`.
#
# Writes, under build/fp16_bf16_pure_vs_cutlass_ref_subnormal_output/:
#   fp16_bf16_pure_vs_cutlass_ref_subnormal_<density>pct_<TIMESTAMP>.txt
#   fp16_bf16_pure_vs_cutlass_ref_subnormal_<density>pct_<TIMESTAMP>.csv
# and, via the EXISTING plot_fp16_bf16_pure_vs_cutlass_ref.m (unmodified --
# same CSV schema, same methods, just different underlying data), under
# build/benchmarks/output/plots/:
#   plot_fp16_bf16_pure_vs_cutlass_ref_square_subnormal<density>pct.png / .fig
#   plot_fp16_bf16_pure_vs_cutlass_ref_rect_subnormal<density>pct.png / .fig
# -- distinct names from the uniform-data plots (plot_fp16_bf16_pure_vs_
# cutlass_ref_square.png etc.), so neither overwrites the other.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu"
OUT="$BUILD_DIR/fp16_bf16_pure_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp16_bf16_pure_vs_cutlass_ref_subnormal_output"

DENSITY="10"
BIN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --density) DENSITY="$2"; shift 2 ;;
        *) BIN_ARGS+=("$1"); shift 1 ;;
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
    exec srun --jobid="$JOBID" --overlap "$0" --density "$DENSITY" "${BIN_ARGS[@]}"
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

echo "==> Building: fp16_bf16_pure_vs_cutlass_ref"
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
DENSITY_TAG="subnormal${DENSITY}pct"
TXT_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_${DENSITY_TAG}_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_${DENSITY_TAG}_${TIMESTAMP}.csv"

echo
echo "==> Running fp16_bf16_pure_vs_cutlass_ref --subnormal-density $DENSITY ${BIN_ARGS[*]:-}"
"$OUT" --subnormal-density "$DENSITY" "${BIN_ARGS[@]}" | tee "$TXT_LOG"

echo
echo "==> Writing CSV summary: $CSV_LOG"
# Identical parser to scripts/fp16_bf16_pure_vs_cutlass_ref.sh's own
# embedded one -- the .cu's shape-line format (M=.. K=.. N=..) and method
# marker lines are completely unchanged by --subnormal-density (that flag
# only prints one extra banner line near the top of the log, outside any
# shape block, which this parser's shape_re/marker matching simply never
# touches).
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    ("mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty", "pure_half"),
    ("mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp", "pure_half_workspace"),
    ("gemm_fp16_mma_ptx_wide_bigreg_sq_deep4", "native_fp16"),
    ("cutlass (real, fp16 tensor-op)", "cutlass_fp16"),
    ("cutlass fp16 SKIPPED", "cutlass_fp16"),
    ("cublas (Hgemm)", "cublas_fp16"),
    ("mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp", "pure_bf16"),
    ("gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp", "native_bf16"),
    ("cutlass (real, bf16 tensor-op)", "cutlass_bf16"),
    ("cutlass bf16 SKIPPED", "cutlass_bf16"),
    ("cublas (GemmEx, CUDA_R_16BF)", "cublas_bf16"),
]

FIELDS = [
    "M", "K", "N",
    "pure_half_ms", "pure_half_gflops",
    "pure_half_workspace_ms", "pure_half_workspace_gflops",
    "native_fp16_ms", "native_fp16_gflops",
    "cutlass_fp16_ms", "cutlass_fp16_gflops",
    "cublas_fp16_ms", "cublas_fp16_gflops",
    "pure_bf16_ms", "pure_bf16_gflops",
    "native_bf16_ms", "native_bf16_gflops",
    "cutlass_bf16_ms", "cutlass_bf16_gflops",
    "cublas_bf16_ms", "cublas_bf16_gflops",
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
echo "==> Plotting (tag: $DENSITY_TAG): $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); plot_fp16_bf16_pure_vs_cutlass_ref('$CSV_LOG', '$DENSITY_TAG');"
if command -v matlab >/dev/null 2>&1; then
    matlab -nodisplay -nosplash -nodesktop -r "$MATLAB_CALL exit;"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -nodisplay -nosplash -nodesktop -r \"$MATLAB_CALL exit;\"" >&2
fi

#!/bin/bash
# Trimmed-down sibling of scripts/benchmarks_gemm_format_32_16_compute_
# 32.sh: builds and runs benchmarks/kernels/cuda/matrix/fp32_16_plotted_
# vs_cutlass_ref.cu, which times the eight methods scripts/plot_gemm_
# format_32_16_compute_32.m actually plots (its other series are
# commented out, not deleted -- see that file's own series-construction
# block) -- TOMOS Native, TOMOS FP32/FP16 (mm_fp32_16_syncpipe4_bk8_
# warptile_nostride_snake_midpoint_exact -- the fused kernel; whether it
# actually beats mm_fp32_16_tiled_cpasync_dualbuf below is only
# confirmed at one prior N=8192 point, NOT yet a full sweep), TOMOS
# FP32/FP16 (truncated) (mm_fp32_16_tiled_cpasync_dualbuf_truncated --
# the plain, 0-spill kernel, switched IN after a full-sweep run showed
# it beats the fused truncated kernel at every size, the OPPOSITE
# direction from the plain pair above), TOMOS FP32/FP16 (workspace),
# TOMOS FP32/BF16 (mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_
# midpoint_exact -- the fused bf16 kernel, switched in since it beats
# dualbuf outright at every size), TOMOS FP32/BF16 (workspace)
# (mm_fp32_bf16_gemm_materialize_rounded -- the ROUNDED-encoding bf16
# format, decoded via the materialize strategy to sidestep the fused
# rounded kernel's own real spill, REG:128 STACK:56, 23 STL + 11 LDL),
# CUTLASS, cuBLAS -- plus three extra methods not (yet) plotted:
# mm_fp32_16_tiled_cpasync_dualbuf, the older plain fused reference for
# the still-unconfirmed FP32/FP16 comparison above; mm_fp32_bf16_tiled_
# cpasync_dualbuf, the older bf16 kernel the BF16 series was switched
# out from; and mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_
# midpoint_exact_truncated, the fused truncated kernel the truncated
# series was switched out from -- all three tracked for their own head-
# to-head ratio prints -- instead of that script's full ~15-method
# sweep, for fast iteration when only these matter.
#
# CSV output uses the EXACT same column names benchmarks_gemm_format_32_
# 16_compute_32.sh's own parser produces for the eight plotted methods,
# so plot_gemm_format_32_16_compute_32.m can plot this script's own CSV
# directly with zero changes -- and this script calls that same
# plotting function automatically at the end, same convention as
# benchmarks_gemm_format_32_16_compute_32.sh itself. The three extra
# columns are captured as raw data alongside them either way.
#
# Usage:
#   scripts/fp32_16_plotted_vs_cutlass_ref.sh
#       Sweeps the SAME square N list (1024/1536/2048/3072/4096/6144/
#       8192/12288/16384/24576/32768/49152) and rectangular K sweep at
#       M=N=49152 (K=1024..32768, same list minus 49152) as scripts/
#       benchmarks_gemm_format_32_16_compute_32.sh, so results are
#       directly comparable size-for-size. Takes no arguments.
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp32_16_plotted_vs_cutlass_ref.sh
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
# All output is saved to a timestamped .txt log, and a per-shape summary
# (ms + GFlop/s for each method) is parsed out into a .csv, both under
# build/fp32_16_plotted_vs_cutlass_ref_output/ (a separate output
# directory from the other comparison scripts', so runs never collide).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/fp32_16_plotted_vs_cutlass_ref.cu"
OUT="$BUILD_DIR/fp32_16_plotted_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp32_16_plotted_vs_cutlass_ref_output"

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

echo "==> Building: fp32_16_plotted_vs_cutlass_ref"
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
TXT_LOG="$OUTPUT_DIR/fp32_16_plotted_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_16_plotted_vs_cutlass_ref_${TIMESTAMP}.csv"

echo
echo "==> Running fp32_16_plotted_vs_cutlass_ref (full sweep)"
"$OUT" | tee "$TXT_LOG"

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    # "..._truncated" MUST come before the plain fused marker below --
    # "mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact"
    # is itself a prefix of "..._truncated"; this loop takes the first
    # match, so checking the longer/more specific string first avoids
    # misfiling every truncated line as the plain fused column.
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated", "fp32_16_fused_midpoint_exact_trunc"),
    ("mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_16_fused_midpoint_exact"),
    ("mm_fp32_16_gemm_materialize_midpoint_exact", "fp32_16_gemm_materialize_midpoint_exact"),
    # "_rounded" MUST come before the bare marker below -- it's a
    # superstring of it. Not currently produced by this script's own
    # .cu file (the fused rounded kernel's own timing isn't wired in
    # here, only mm_fp32_bf16_gemm_materialize_rounded's is), but kept
    # for consistency with the other three scripts sharing fp32_recon_
    # vs_cutlass_ref.cu, and to avoid silently misfiling this line into
    # fp32_bf16_fused_midpoint_exact if that timing is ever added here.
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded", "fp32_bf16_fused_rounded"),
    ("mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact", "fp32_bf16_fused_midpoint_exact"),
    ("mm_fp32_bf16_gemm_materialize_rounded", "fp32_bf16_materialize_rounded"),
    ("mm_fp32_bf16_tiled_cpasync_dualbuf", "fp32_bf16_dualbuf"),
    # Plain, spill-free truncated-decode reference kernel -- now the
    # plotted TOMOS FP32/FP16 (truncated) series itself (see plot_gemm_
    # format_32_16_compute_32.m's own header comment for why it was
    # switched in over the fused truncated kernel). "mm_fp32_16_tiled_
    # cpasync_dualbuf" (the plain, non-truncated marker below) is a
    # PREFIX of this string, so this MUST be checked first.
    ("mm_fp32_16_tiled_cpasync_dualbuf_truncated", "fp32_16_dualbuf_trunc"),
    # Old plain fused reference for the still-unconfirmed (full-sweep)
    # FP32/FP16 head-to-head against the fused midpoint_exact kernel
    # above -- extra raw data, not yet a plotted series.
    ("mm_fp32_16_tiled_cpasync_dualbuf", "fp32_16_dualbuf"),
    ("gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact (native)", "native"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemm)", "cublas"),
]

# Column names for the eight methods plot_gemm_format_32_16_compute_
# 32.m actually plots (native, fp32_16_fused_midpoint_exact,
# fp32_16_dualbuf_trunc, fp32_16_gemm_materialize_midpoint_exact,
# fp32_bf16_fused_midpoint_exact, fp32_bf16_materialize_rounded,
# cutlass, cublas -- see that file's own series-construction block for
# which struct field backs which series) EXACTLY match scripts/
# benchmarks_gemm_format_32_16_compute_32.sh's own FIELDS, so that .m
# file can plot this CSV directly unchanged.
# fp32_16_fused_midpoint_exact_trunc, fp32_bf16_dualbuf, and
# fp32_16_dualbuf are captured as extra raw data alongside them -- NOT
# (currently) one of the .m file's own plotted series, so their column
# names don't need to match anything.
FIELDS = [
    "M", "K", "N",
    "fp32_16_fused_midpoint_exact_ms", "fp32_16_fused_midpoint_exact_gflops",
    "fp32_16_fused_midpoint_exact_trunc_ms", "fp32_16_fused_midpoint_exact_trunc_gflops",
    "fp32_16_gemm_materialize_midpoint_exact_ms", "fp32_16_gemm_materialize_midpoint_exact_gflops",
    "fp32_bf16_fused_midpoint_exact_ms", "fp32_bf16_fused_midpoint_exact_gflops",
    "fp32_bf16_materialize_rounded_ms", "fp32_bf16_materialize_rounded_gflops",
    "fp32_bf16_dualbuf_ms", "fp32_bf16_dualbuf_gflops",
    "fp32_16_dualbuf_trunc_ms", "fp32_16_dualbuf_trunc_gflops",
    "fp32_16_dualbuf_ms", "fp32_16_dualbuf_gflops",
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

echo
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$CSV_LOG')\"" >&2
fi

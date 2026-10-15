#!/bin/bash
# Focused subnormal-density sweep for a SINGLE kernel -- mm_fp32_16_gemm_
# materialize_midpoint_exact, the "TOMOS FP32 / FP16 (workspace)" series
# in plot_gemm_format_32_16_compute_32.m (the current-best fp32_16
# "32-bit compute"/full-fp32-reconstruction kernel: decodes A to a plain
# fp32 scratch buffer up front, then calls this codebase's best native
# fp32 GEMM, gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
# midpoint_exact, as a black box -- see fp32_16_cuda_kernels.cu's own
# comment on that kernel for the full derivation).
#
# Reuses the SAME binary as scripts/fp32_recon_vs_cutlass_ref_subnormal.sh
# (benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu, already
# supporting --subnormal-density) -- no source changes needed here, this
# script just drives it through a square-shape/density grid and reports
# ONLY this kernel's own throughput, as a standalone LaTeX table (the
# binary still runs its full ~20-method sweep per shape internally; every
# other method's numbers are simply not extracted).
#
# Square shapes: N = 49152, 32768, 24576 (M=K=N).
# Subnormal density: 0% (baseline -- reproduces the plain uniform
# generator exactly, see run_shape's own comment), 10%, 20%, 30%, 40%.
#
# Usage:
#   scripts/fp32_16_workspace_subnormal_density_table.sh
#       Runs all 3 sizes x 5 densities (15 runs). Takes no arguments.
#
# CUTLASS is fetched (shallow clone, header-only) into CUTLASS_DIR on
# first run and reused after that, same as the other *_vs_cutlass_ref.sh
# scripts. Override with the CUTLASS_DIR env var if you already have one:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp32_16_workspace_subnormal_density_table.sh
#
# Rebuilds tomos_cuda first (so this reflects the current kernel state)
# and links the standalone nvcc build directly against it -- same
# discipline as every other *_vs_cutlass_ref.sh script.
#
# On Gilbreth the shell this runs in is usually the login node (no A100
# there) -- if SLURM_JOB_ID isn't already set, this re-execs itself
# inside the caller's own running a100* job via `srun --overlap`.
#
# Writes, under build/fp32_16_workspace_subnormal_density_output/:
#   fp32_16_workspace_subnormal_density_<TIMESTAMP>.txt   (full console log, all 15 runs)
#   fp32_16_workspace_subnormal_density_<TIMESTAMP>.csv   (N, density, ms, GFlop/s)
#   fp32_16_workspace_subnormal_density_<TIMESTAMP>.tex   (standalone LaTeX table, booktabs style
#                                                           -- same style as benchmarks/bench_latex_reporter.hpp)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu"
OUT="$BUILD_DIR/fp32_recon_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp32_16_workspace_subnormal_density_output"

# The three square shapes and five densities this table reports --
# see this file's own header comment for why these specific values.
NS=(49152 32768 24576)
DENSITIES=(0 10 20 30 40)

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
TXT_LOG="$OUTPUT_DIR/fp32_16_workspace_subnormal_density_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp32_16_workspace_subnormal_density_${TIMESTAMP}.csv"
TEX_LOG="$OUTPUT_DIR/fp32_16_workspace_subnormal_density_${TIMESTAMP}.tex"

: > "$TXT_LOG"
for n in "${NS[@]}"; do
    for d in "${DENSITIES[@]}"; do
        echo
        echo "==> Running fp32_recon_vs_cutlass_ref --subnormal-density $d $n"
        "$OUT" --subnormal-density "$d" "$n" | tee -a "$TXT_LOG"
    done
done

echo
echo "==> Writing CSV/LaTeX: $CSV_LOG / $TEX_LOG"
# Extracts ONLY mm_fp32_16_gemm_materialize_midpoint_exact's own line --
# matched on the exact string "mm_fp32_16_gemm_materialize_midpoint_exact
# = " (trailing " = "), which is NOT a substring of its "_fast_normal" or
# "_vec" siblings' own lines (both have an underscore right after
# "exact", before their own " = "), so this can't misfile onto them.
python3 - "$TXT_LOG" "$CSV_LOG" "$TEX_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path, tex_path = sys.argv[1], sys.argv[2], sys.argv[3]

density_re = re.compile(r"=== A/B subnormal density: ([0-9.]+)% ")
shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
target_marker = "mm_fp32_16_gemm_materialize_midpoint_exact = "
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

results = {}  # (density_pct, N) -> (ms, gflops)
cur_density = None
cur_n = None

with open(txt_path) as f:
    for line in f:
        line = line.rstrip("\n")
        dm = density_re.search(line)
        if dm:
            cur_density = float(dm.group(1))
            continue
        sm = shape_re.match(line)
        if sm:
            cur_n = int(sm.group(3))
            continue
        if target_marker in line:
            vm = value_re.search(line)
            if vm and cur_density is not None and cur_n is not None:
                results[(cur_density, cur_n)] = (float(vm.group(1)), float(vm.group(2)))

ns = sorted({n for (_, n) in results}, reverse=True)
densities = sorted({d for (d, _) in results})

with open(csv_path, "w") as out:
    out.write("N,density_pct,ms,gflops\n")
    for n in ns:
        for d in densities:
            ms, gflops = results.get((d, n), ("", ""))
            out.write(f"{n},{d},{ms},{gflops}\n")

with open(tex_path, "w") as out:
    out.write("\\documentclass{article}\n"
               "\\usepackage{booktabs}\n"
               "\\usepackage[margin=1in]{geometry}\n"
               "\\begin{document}\n\n"
               "\\begin{table}[h]\n"
               "\\centering\n"
               "\\caption{mm\\_fp32\\_16\\_gemm\\_materialize\\_midpoint\\_exact "
               "(``TOMOS FP32 / FP16 (workspace)'') throughput (GFlop/s) "
               "vs.\\ subnormal density, square shapes (M=K=N)}\n")
    out.write("\\begin{tabular}{l" + "r" * len(densities) + "}\n")
    out.write("\\toprule\n")
    out.write("$N$ & " + " & ".join(f"{d:g}\\%" for d in densities) + " \\\\\n")
    out.write("\\midrule\n")
    for n in ns:
        cells = []
        for d in densities:
            if (d, n) in results:
                cells.append(f"{results[(d, n)][1]:,.1f}")
            else:
                cells.append("--")
        out.write(f"{n:,} & " + " & ".join(cells) + " \\\\\n")
    out.write("\\bottomrule\n"
               "\\end{tabular}\n"
               "\\end{table}\n\n"
               "\\end{document}\n")

print(f"Wrote {len(results)} of {len(ns) * len(densities)} expected (N, density) results.")
PYEOF

echo
echo "==> Logs written:"
echo "      $TXT_LOG"
echo "      $CSV_LOG"
echo "      $TEX_LOG"
echo
echo "==> LaTeX table:"
cat "$TEX_LOG"

#!/bin/bash
# Sibling of scripts/fp16_bf16_pure_vs_cutlass_ref.sh that runs the EXACT
# same binary (benchmarks/kernels/cuda/matrix/fp16_bf16_pure_vs_cutlass_
# ref.cu, otherwise unmodified) but swept across a FINER-GRAINED size
# list than that binary's own hardcoded default sweep: every size N in
# fp16_bf16_pure_vs_cutlass_ref.sh's own square sweep (1024, 2048, 4096,
# 8192, 16384, 32768, 65536) PLUS each size's own midpoint N*1.5, giving
# roughly double the resolution across the same overall range. Same idea
# for the rectangular sweep, but held at M=N=65536 (not that binary's own
# default 44000) -- see the --rect argument note below for how that's
# done without touching the .cu's own default sweep.
#
# Kernels compared (unchanged from benchmarks_gemm_fp16.cu --
# see that file's own header comment): mm_fp32_16_pure_half_wide_bigreg_
# sq_inline_dirty (fp32_16's fp16-compute tier) and mm_fp32_bf16_pure_
# bf16_wide_bigreg_sq_deep4 (fp32_bf16's bf16-compute tier) against this
# codebase's own best native (unpacked) fp16/bf16 GEMM (gemm_fp16_mma_
# ptx_wide_bigreg_sq_deep4 / gemm_bf16_mma_ptx_wide_bigreg_sq_deep4),
# real CUTLASS's fp16/bf16 tensor-op device::Gemm, and cuBLAS's Hgemm/
# GemmEx.
#
# IMPORTANT TRADE-OFF: the underlying fp16_bf16_pure_vs_cutlass_ref
# binary only accepts a SINGLE shape per invocation (a single N for
# square, or --rect <K> [MN] for one rectangular shape) -- there is no
# way to pass it a custom multi-size list in one process. So unlike a
# true single-process sweep, THIS script launches the binary ONCE PER
# SIZE in the lists below -- each size is its own separate process/
# cublas handle/CUTLASS workspace, not one continuous run. That means,
# unlike fp16_bf16_pure_vs_cutlass_ref.sh's own "all in one run so the
# numbers share the same GPU clock/thermal/contention state" property,
# results here can differ slightly run-to-run due to GPU clock/thermal
# state drifting between the separate process launches. Still the same
# GPU, same job allocation, same building block -- just not one
# unbroken process.
#
# The --rect mode's M=N was originally hardcoded to 44000 inside the
# .cu with no way to override it; a 3rd optional CLI arg (`--rect <K>
# <MN>`) was added to benchmarks_gemm_fp16.cu itself so this
# script can hold M=N=65536 instead, while the argument-less default
# (used by fp16_bf16_pure_vs_cutlass_ref.sh's own sweep) still falls
# back to 44000 unchanged.
#
# Usage:
#   scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh
#       Sweeps the square (M=K=N) and rectangular (M=N=65536, K swept)
#       size lists below, one binary invocation per size.
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh
#
# Rebuilds tomos_cuda first (so this always reflects the current state
# of the kernels, not a stale libtomos_cuda.so) and links the standalone
# nvcc build against it directly -- CUTLASS itself is still NOT a
# dependency of tomos's own CMake build, only of this one comparison
# binary. Reuses the EXACT same build/fp16_bf16_pure_vs_cutlass_ref
# binary path fp16_bf16_pure_vs_cutlass_ref.sh itself builds (same
# source, same flags -- byte-identical either way), so running this
# script and that one back to back doesn't rebuild redundantly if
# nothing changed.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is
# only reachable inside an active job allocation. If SLURM_JOB_ID isn't
# already set, this script finds the caller's running job on an a100*
# partition and re-execs itself inside it via `srun --overlap`, so both
# the build and the run always land on the actual target GPU.
#
# Every run's full console output (concatenated across all size
# invocations) is saved to a timestamped .txt log, and a per-N summary
# (ms + GFlop/s for each method) is parsed out into a .csv, both under
# build/output/benchmarks/gemm/fp16_compute/data/ (a separate output
# directory from fp16_bf16_pure_vs_cutlass_ref.sh's own runs, even
# though both scripts' CSV schemas are identical, so it's always
# unambiguous which sweep produced a given log).
#
# Also plots the fresh CSV via scripts/plot_fp16_bf16_pure_vs_cutlass_ref.m
# (same auto-invoke-matlab-or-warn-and-print-the-manual-command pattern
# scripts/benchmarks_gemm_format_32_16_compute_32.sh already uses for its
# own plots), writing plot_fp16_bf16_pure_vs_cutlass_ref_{square,rect}.png/
# .fig under build/benchmarks/output/plots/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu"
OUT="$BUILD_DIR/fp16_bf16_pure_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/benchmarks/gemm/fp16_compute/data"

# fp16_bf16_pure_vs_cutlass_ref.sh's own square-N sweep (1024, 2048,
# 4096, 8192, 16384, 32768, 65536), with each value's own 1.5x midpoint
# inserted right after it for finer resolution. Stops at 65536 --
# 65536*1.5=98304 runs out of device memory, so it's deliberately not
# included here (unlike the rectangular sweep below, which does reach
# its own list's 1.5x endpoint since that shape stays much smaller).
SIZES=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536)

# Same interpolation pattern applied to benchmarks_gemm_fp16.cu's
# own rect_ks list (1024 .. 32768), swept with M=N held fixed at
# RECT_MN via the --rect <K> <MN> form added to that .cu's main().
RECT_KS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152)
RECT_MN=65536

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
TXT_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_matched_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_matched_${TIMESTAMP}.csv"

echo
: > "$TXT_LOG"
for n in "${SIZES[@]}"; do
    echo "==> Running fp16_bf16_pure_vs_cutlass_ref $n"
    "$OUT" "$n" | tee -a "$TXT_LOG"
    echo
done
for k in "${RECT_KS[@]}"; do
    echo "==> Running fp16_bf16_pure_vs_cutlass_ref --rect $k $RECT_MN"
    "$OUT" --rect "$k" "$RECT_MN" | tee -a "$TXT_LOG"
    echo
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Identical to fp16_bf16_pure_vs_cutlass_ref.sh's own parser -- the
# underlying binary's printed output format is unaffected by which
# shell script invoked it, so the schema stays byte-for-byte the same.
METHOD_MARKERS = [
    # Checked BEFORE the plain "pure_half" marker below: its own text
    # ("...inline_dirty_fast_acongruous...") is a superset substring of the
    # plain marker's text ("...inline_dirty..."), so it must win the match
    # first or every fast-decode line would misclassify as the original.
    ("mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp", "pure_half_fast"),
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
    "pure_half_fast_ms", "pure_half_fast_gflops",
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
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_fp16_bf16_pure_vs_cutlass_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_fp16_bf16_pure_vs_cutlass_ref('$CSV_LOG')\"" >&2
fi

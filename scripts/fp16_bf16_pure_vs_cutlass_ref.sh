#!/bin/bash
# Builds and runs a focused benchmark comparing the "pure/composite"
# (tensor-core compute, decode-only) tier of both packed-precision
# families -- mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty and mm_
# fp32_bf16_pure_bf16_wide_bigreg_sq_deep4 -- against this codebase's own
# best native (unpacked) fp16/bf16 GEMM, real CUTLASS's fp16/bf16
# tensor-op device::Gemm, and cuBLAS's Hgemm/GemmEx, all in one run so
# the numbers share the same GPU clock/thermal/contention state
# (benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu).
#
# Mirrors scripts/fp32_recon_vs_cutlass_ref.sh's structure (SLURM
# auto-targeting, txt+csv logging) -- see that script's own comments for
# the reasoning behind each piece.
#
# Usage:
#   scripts/fp16_bf16_pure_vs_cutlass_ref.sh [N]
#
# With no args, sweeps 1024/2048/4096/8192/16384/32768/65536. Pass a
# single N to benchmark just that size:
#   scripts/fp16_bf16_pure_vs_cutlass_ref.sh 8192
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/fp16_bf16_pure_vs_cutlass_ref.sh
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
# .csv, both under build/fp16_bf16_pure_vs_cutlass_ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp16.cu"
OUT="$BUILD_DIR/fp16_bf16_pure_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/fp16_bf16_pure_vs_cutlass_ref_output"

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
TXT_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/fp16_bf16_pure_vs_cutlass_ref_${TIMESTAMP}.csv"

echo
echo "==> Running fp16_bf16_pure_vs_cutlass_ref $*"
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

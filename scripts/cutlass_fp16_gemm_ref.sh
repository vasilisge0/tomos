#!/bin/bash
# Builds and runs the CUTLASS FP16 tensorop GEMM reference benchmark
# (benchmarks/kernels/cuda/matrix/cutlass_fp16_gemm_ref.cu) -- used
# purely as an external comparison point against tomos's own custom
# kernels (see verify_mm_fp16_gemm.cu). CUTLASS is NOT a dependency of
# tomos itself: this is a standalone nvcc build, deliberately not wired
# into the project's CMake build.
#
# Usage:
#   scripts/cutlass_fp16_gemm_ref.sh [-- N]
#
# With no args, sweeps the same sizes as verify_mm_fp16_gemm's
# performance section (1024/2048/4096/8192/16384). Pass a single N to
# benchmark just that size:
#   scripts/cutlass_fp16_gemm_ref.sh -- 8192
#
# CUTLASS is fetched (shallow clone, header-only usage -- no separate
# library build needed) into CUTLASS_DIR on first run and reused after
# that. Override the location with the CUTLASS_DIR env var if you
# already have a checkout elsewhere:
#   CUTLASS_DIR=/path/to/cutlass scripts/cutlass_fp16_gemm_ref.sh
#
# Uses the same square-NxN, column-major, alpha=1.7/beta=0.3,
# cudaEvent_t timing methodology as verify_mm_fp16_gemm's bench_one(),
# so the reported GFlop/s numbers are directly comparable. This is one
# specific "textbook" Ampere config (128x256x32 threadblock tile,
# 64x64x32 warp tile, 16x8x16 instruction shape, 3 stages) taken from
# CUTLASS's own example programs -- not a tuned search over CUTLASS's
# full config space.
#
# Also times cublasHgemm directly in the same run, printing both
# GFlop/s numbers and their ratio per size -- same idea as
# verify_mm_fp16_gemm's own cuBLAS comparison, kept in one process so
# both numbers share the same GPU clock/thermal state.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/cutlass_fp16_gemm_ref.cu"
OUT="$BUILD_DIR/cutlass_fp16_gemm_ref"

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "==> Building: cutlass_fp16_gemm_ref"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    "$SRC" -lcublas -o "$OUT"

echo
echo "==> Running cutlass_fp16_gemm_ref $*"
"$OUT" "$@"

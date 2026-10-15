#!/bin/bash
# Builds and runs the fp32_16 "pure_half" (fp16-layout, fp16-compute)
# square-GEMM correctness + performance verification executable
# (benchmarks/kernels/cuda/matrix/verify_mm_fp32_16_pure_half_gemm.cu).
#
# Usage:
#   scripts/verify_mm_fp32_16_pure_half_gemm.sh [-- executable-args...]
#
# By default this runs both the correctness section (mm_fp32_16_pure_half,
# _tiled, and _wmma -- all of which decode each A(i,l) from its wh half
# only, via fp32_16_half_only, and run the multiply-accumulate in __half
# (or, for _wmma, tensor-core mma.sync with an fp32 accumulator) -- vs a
# CPU reference, including several sizes not a multiple of the tiled
# kernel's 128x128 or the wmma kernel's 64x64 thread block) and the
# performance section (the same three kernels vs cuBLAS's Hgemm, the fp16
# GEMM baseline, at large square sizes). Pass executable args through
# after `--`, e.g.:
#   scripts/verify_mm_fp32_16_pure_half_gemm.sh -- --skip-correctness
#   scripts/verify_mm_fp32_16_pure_half_gemm.sh -- --skip-performance
#
# To isolate one shape's kernel launches for external profiling (ncu/nsys)
# instead of running the full sweep:
#   scripts/verify_mm_fp32_16_pure_half_gemm.sh -- --only-shape 8192
#   ncu --kernel-name mm_fp32_16_pure_half_wmma_kernel --launch-count 1 \
#       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp32_16_pure_half_gemm" \
#       --only-shape 8192
#
# All test data is generated directly on the device (not read from disk or
# built up on the host), so this scales to the large shapes it exercises
# even on a host with a constrained memory cgroup -- point it at whatever
# GPU is available (e.g. an A100 in an interactive session) for the fullest
# picture; it will also just skip individual performance shapes that don't
# fit in whatever device memory is actually free at the time.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_mm_fp32_16_pure_half_gemm
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET $*"
"$EXECUTABLE" "$@"

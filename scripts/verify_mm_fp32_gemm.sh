#!/bin/bash
# Builds and runs the fp32 square-GEMM correctness + performance
# verification executable (benchmarks/kernels/cuda/matrix/verify_mm_fp32_gemm.cu).
#
# Usage:
#   scripts/verify_mm_fp32_gemm.sh [-- executable-args...]
#
# By default this runs both the correctness section (custom kernel(s) + GPU
# cuBLAS vs a CPU reference, including several sizes not a multiple of the
# 16x16 thread block gemm_generic_kernel launches with) and the performance
# section (custom kernel(s) vs cuBLAS's Sgemm, the fp32 GEMM baseline, at
# large square sizes). Pass executable args through after `--`, e.g.:
#   scripts/verify_mm_fp32_gemm.sh -- --skip-correctness
#   scripts/verify_mm_fp32_gemm.sh -- --skip-performance
#
# To isolate one shape's kernel launches for external profiling (ncu/nsys)
# instead of running the full sweep:
#   scripts/verify_mm_fp32_gemm.sh -- --only-shape 8192
#   ncu --kernel-name gemm_generic_kernel --launch-count 1 \
#       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp32_gemm" \
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

TARGET=verify_mm_fp32_gemm
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET $*"
"$EXECUTABLE" "$@"

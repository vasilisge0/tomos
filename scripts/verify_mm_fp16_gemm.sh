#!/bin/bash
# Builds and runs the plain fp16 (genuine __half A/B/C, no FP32_16 packed
# storage) square-GEMM correctness + performance verification executable
# (benchmarks/kernels/cuda/matrix/verify_mm_fp16_gemm.cu).
#
# Usage:
#   scripts/verify_mm_fp16_gemm.sh [-- executable-args...]
#
# By default this runs both the correctness section (gemm_generic_kernel
# <half> (double-accumulated), gemm_fp16_tiled (128x128/8x8 register-
# blocked, __half-accumulated end to end), and gemm_fp16_wmma (Tensor Core
# mma.sync, fp32-accumulated) -- vs a CPU reference, including several
# sizes not a multiple of the tiled kernel's 128x128 or the wmma kernel's
# 128x128/BK=64 thread block) and the performance section (the same three
# kernels vs cuBLAS's Hgemm, at large square sizes). Pass executable args
# through after `--`, e.g.:
#   scripts/verify_mm_fp16_gemm.sh -- --skip-correctness
#   scripts/verify_mm_fp16_gemm.sh -- --skip-performance
#
# To isolate one shape's kernel launches for external profiling (ncu/nsys)
# instead of running the full sweep:
#   scripts/verify_mm_fp16_gemm.sh -- --only-shape 8192
#   ncu --kernel-name gemm_fp16_wmma_kernel --launch-count 1 \
#       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mm_fp16_gemm" \
#       --only-shape 8192
#
# All test data is generated directly on the device (not read from disk or
# built up on the host), so this scales to the large shapes it exercises
# even on a host with a constrained memory cgroup -- point it at whatever
# GPU is available (e.g. an A100 in an interactive session) for the fullest
# picture; it will also just skip individual performance shapes that don't
# fit in whatever device memory is actually free at the time.
#
# This is the plain-fp16 sibling of scripts/verify_mm_fp32_16_pure_half_
# gemm.sh: same tuned tile/warp/launch_bounds wmma design, but with no
# FP32_16 wh-decode step at all, so diffing the two scripts' wmma numbers
# quantifies exactly how much that decode costs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_mm_fp16_gemm
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET $*"
"$EXECUTABLE"  --skip-correctness "$@"

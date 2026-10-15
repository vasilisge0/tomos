#!/bin/bash
# Builds and runs the mv_fp32_16_* GEMV correctness + performance
# verification executable (benchmarks/kernels/cuda/matrix/verify_mv_fp32_16_gemv.cu).
#
# Usage:
#   scripts/verify_mv_fp32_16_gemv.sh [-- executable-args...]
#
# By default this runs both the correctness section (GPU vs CPU reference,
# including the boundary/tail/transpose cases that once caught a real
# misaligned-load bug in the register-blocked narrow_vec kernels) and the
# performance section (pure_half vs cuBLAS's Hgemm(n=1), the real fp16 GEMV
# baseline, at both narrow and wide shapes). Pass executable args through
# after `--`, e.g.:
#   scripts/verify_mv_fp32_16_gemv.sh -- --skip-correctness
#   scripts/verify_mv_fp32_16_gemv.sh -- --skip-performance
#
# To isolate one shape's kernel launches for external profiling (ncu/nsys)
# instead of running the full sweep:
#   scripts/verify_mv_fp32_16_gemv.sh -- --only-shape 32768 32768
#   ncu --kernel-name mv_fp32_16_pure_half_wide_kernel --launch-count 1 \
#       "$BUILD_DIR/benchmarks/kernels/cuda/matrix/verify_mv_fp32_16_gemv" \
#       --only-shape 32768 32768
#
# All test data is generated directly on the device (not read from disk or
# built up on the host), so this scales to the large shapes it exercises
# even on a host with a constrained memory cgroup — point it at whatever
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

TARGET=verify_mv_fp32_16_gemv
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET $*"
"$EXECUTABLE" "$@"

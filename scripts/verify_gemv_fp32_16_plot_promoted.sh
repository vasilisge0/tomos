#!/bin/bash
# Builds and runs the promoted-kernels-only NO_TRANSPOSE GEMV correctness
# check (benchmarks/kernels/cuda/matrix/verify_gemv_fp32_16_plot_promoted.cu).
#
# Covers EXACTLY the 5 kernels scripts/plot_gemv_fp32_16_pure_half_vs_
# cublas_ref.m actually plots (TOMOS FP16, TOMOS FP32/FP16, TOMOS FP32/
# FP16 (fast), TOMOS FP32/BF16, TOMOS BF16) across 5 representative shapes
# -- much smaller and faster than scripts/verify_mv_fp32_16_gemv.sh's own
# much larger sweep (which also covers every exploratory/negative-result
# kernel this session built along the way), for when the question is just
# "are the promoted kernels still correct" rather than a full regression
# sweep of every experimental sibling. Same idea as scripts/verify_mv_
# fp32_transpose_promoted.sh, mirrored for this plot.
#
# Usage:
#   scripts/verify_gemv_fp32_16_plot_promoted.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_gemv_fp32_16_plot_promoted
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET"
"$EXECUTABLE"

#!/bin/bash
# Builds and runs the promoted-kernels-only TRANSPOSE GEVM 16-bit-compute
# correctness check (benchmarks/kernels/cuda/matrix/verify_gemv_fp32_16_
# pure_half_transpose_plot_promoted.cu).
#
# Covers EXACTLY the 4 kernels scripts/plot_gemv_transpose_fp32_16_pure_
# half_vs_cublas_ref.m actually plots (TOMOS FP16, TOMOS FP32/FP16,
# TOMOS FP32/FP16 (fast), TOMOS FP32/BF16) across 5 representative
# shapes -- much smaller and faster than a full exploratory sweep, for
# when the question is just "are the promoted kernels still correct."
#
# Usage:
#   scripts/verify_gemv_fp32_16_pure_half_transpose_plot_promoted.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_gemv_fp32_16_pure_half_transpose_plot_promoted
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET"
"$EXECUTABLE"

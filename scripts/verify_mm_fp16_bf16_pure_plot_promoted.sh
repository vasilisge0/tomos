#!/bin/bash
# Builds and runs the promoted-kernels-only NO_TRANSPOSE GEMM pure/
# composite (tensor-core) tier correctness check (benchmarks/kernels/
# cuda/matrix/verify_mm_fp16_bf16_pure_plot_promoted.cu).
#
# Covers EXACTLY the 5 kernels scripts/plot_fp16_bf16_pure_vs_cutlass_
# ref.m actually plots (TOMOS FP16, TOMOS BF16, TOMOS FP32/FP16, TOMOS
# FP32/FP16 (workspace), TOMOS FP32/BF16) across 5 representative shapes
# -- much smaller and faster than the full exploratory suites, for when
# the question is just "are the promoted kernels still correct."
#
# Usage:
#   scripts/verify_mm_fp16_bf16_pure_plot_promoted.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_mm_fp16_bf16_pure_plot_promoted
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET"
"$EXECUTABLE"

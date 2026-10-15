#!/bin/bash
# Builds and runs the promoted-kernels-only TRANSPOSE GEMV correctness
# check (benchmarks/kernels/cuda/matrix/verify_mv_fp32_transpose_promoted.cu).
#
# Covers EXACTLY the 5 kernels scripts/plot_gemv_transpose_fp32_recon_vs_
# cublas_ref.m actually plots (native, fp32_16, fp32_16_fast, fp32_bf16,
# fp32_bf16_truncated) across 5 representative shapes -- much smaller and
# faster than scripts/verify_mv_fp32_16_gemv.sh's own ~50-kernel sweep,
# for when the question is just "are the promoted kernels still correct"
# rather than a full regression sweep of every experimental sibling.
#
# Usage:
#   scripts/verify_mv_fp32_transpose_promoted.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_mv_fp32_transpose_promoted
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET"
"$EXECUTABLE"

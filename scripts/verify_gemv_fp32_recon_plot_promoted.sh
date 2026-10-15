#!/bin/bash
# Builds and runs the promoted-kernels-only NO_TRANSPOSE GEMV correctness
# check (benchmarks/kernels/cuda/matrix/verify_gemv_fp32_recon_plot_promoted.cu).
#
# Covers EXACTLY the 5 kernels scripts/plot_gemv_fp32_recon_vs_cutlass_
# ref.m actually plots (TOMOS FP32, TOMOS FP32/FP16, TOMOS FP32/FP16
# (fast), TOMOS FP32/BF16, TOMOS FP32/BF16 (workspace)) across 5
# representative shapes -- much smaller and faster than a full
# exploratory sweep, for when the question is just "are the promoted
# kernels still correct" rather than a full regression sweep. Same idea
# as scripts/verify_gemv_fp32_16_plot_promoted.sh, mirrored for this
# tier's own plot.
#
# Usage:
#   scripts/verify_gemv_fp32_recon_plot_promoted.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGET=verify_gemv_fp32_recon_plot_promoted
EXECUTABLE="$BUILD_DIR/benchmarks/kernels/cuda/matrix/$TARGET"

echo "==> Building: $TARGET"
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$(nproc)"

echo
echo "==> Running $TARGET"
"$EXECUTABLE"

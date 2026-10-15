#!/bin/bash
# Builds and runs every CUDA benchmark executable in the project.
#
# Usage:
#   scripts/run_cuda_benchmarks.sh [google-benchmark flags...]
#
# Any arguments are passed through to each benchmark executable, e.g.:
#   scripts/run_cuda_benchmarks.sh --benchmark_min_time=0.1s
#   scripts/run_cuda_benchmarks.sh --benchmark_filter=BM_mm_cublas
#
# Each executable writes its own results (console output, plus a .tex, .csv,
# and plot_benchmarks.m) to an output/ directory next to itself under build/
# — see bench_latex_reporter.hpp. Run compare_precision.m afterwards to
# compare FP16- vs FP32-compute kernels across all of them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR — configure the project first (see scripts/configure/)." >&2
    exit 1
fi

TARGETS=(
    bench_mv_cuda_kernels
    bench_mm_cuda_kernels
    bench_fp32_16_cuda_kernels
    bench_lsqr_cuda
)

EXECUTABLES=(
    # "$BUILD_DIR/benchmarks/kernels/cuda/matrix/bench_mv_cuda_kernels"
    # "$BUILD_DIR/benchmarks/kernels/cuda/matrix/bench_mm_cuda_kernels"
    "$BUILD_DIR/benchmarks/kernels/cuda/matrix/bench_fp32_16_cuda_kernels"
    # "$BUILD_DIR/benchmarks/solvers/cuda/bench_lsqr_cuda"
)

echo "==> Building: ${TARGETS[*]}"
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$(nproc)"

failed=()
for exe in "${EXECUTABLES[@]}"; do
    name="$(basename "$exe")"
    echo
    echo "==> Running $name $*"
    if ! "$exe" "$@"; then
        echo "!! $name exited with a non-zero status" >&2
        failed+=("$name")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "==> All CUDA benchmarks completed successfully."
else
    echo "!! Failed: ${failed[*]}" >&2
    exit 1
fi

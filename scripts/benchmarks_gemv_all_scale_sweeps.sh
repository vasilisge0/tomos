#!/bin/bash
# Runs EVERY GEMV magnitude-scale-sweep script in this codebase, one after
# another, each with its own default arguments (fixed shape, default
# SCALES list, no --normalized). Each sub-script already fully handles
# its own build, SLURM re-exec (via srun --overlap if not already inside
# a job allocation), scale loop, CSV writing, and plotting -- this script
# is purely a convenience batch driver, not a reimplementation of any of
# that.
#
# Covers all three GEVM tiers this codebase has a scale-sweep for, both
# directions where applicable:
#   fp32-recon tier, FP32 (cublasSgemv) reference, NO_TRANSPOSE + TRANSPOSE
#   fp32-recon tier, TRUE FP64 (cublasDgemv) reference, NO_TRANSPOSE + TRANSPOSE
#   16-bit (fp32_16/pure_half) tier, NO_TRANSPOSE + TRANSPOSE
#
# Does NOT abort on the first failure (a single script OOMing or a build
# error shouldn't block the rest of the batch) -- each sub-script's own
# exit code is tracked, and a pass/fail summary is printed at the end.
#
# Usage:
#   scripts/benchmarks_gemv_all_scale_sweeps.sh
#       Runs all six scripts below with their own default arguments.
#
# To run just one tier with custom arguments (--normalized, --scales,
# etc.), call that script directly instead -- see each one's own header.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemv_transpose_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh"
    "benchmarks_gemv_transpose_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh"
    "benchmarks_gemv_fp32_16_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemv_transpose_fp32_16_pure_half_scale_sweep_l2_error_vs_cublas_ref.sh"
)

FAILED=()
PASSED=()

for s in "${SCRIPTS[@]}"; do
    echo
    echo "################################################################"
    echo "==> Running $s"
    echo "################################################################"
    if "$SCRIPT_DIR/$s"; then
        PASSED+=("$s")
    else
        echo "!!! FAILED: $s (exit $?)" >&2
        FAILED+=("$s")
    fi
done

echo
echo "================================================================"
echo "GEMV scale-sweep batch summary"
echo "================================================================"
echo "Passed (${#PASSED[@]}):"
for s in "${PASSED[@]}"; do echo "  ok    $s"; done
echo "Failed (${#FAILED[@]}):"
for s in "${FAILED[@]}"; do echo "  FAIL  $s"; done

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0

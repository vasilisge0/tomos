#!/bin/bash
# Runs EVERY GEMM magnitude-scale-sweep script in this codebase, one
# after another, each with its own default arguments (fixed shape,
# default SCALES list). Each sub-script already fully handles its own
# build, SLURM re-exec (via srun --overlap if not already inside a job
# allocation), scale loop, CSV writing, and plotting -- this script is
# purely a convenience batch driver, not a reimplementation of any of
# that. GEVM counterpart: scripts/benchmarks_gemv_all_scale_sweeps.sh.
#
# Covers every GEMM tier this codebase has a scale-sweep for:
#   fp32-compute tier, TRUE FP64 (cublasDgemm) reference (normalized + raw)
#   16-bit tensor-core (pure/composite) tier
#   fp16 kernels tier (TOMOS FP16 vs cuBLAS/CUTLASS) (normalized + raw)
#   fp32_16 variable-rounding tier (normalized + raw)
#
# Does NOT abort on the first failure -- each sub-script's own exit code
# is tracked, and a pass/fail summary is printed at the end.
#
# Usage:
#   scripts/benchmarks_gemm_all_scale_sweeps.sh
#       Runs all seven scripts below with their own default arguments.
#
# To run just one tier with custom arguments, call that script directly
# instead -- see each one's own header.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "benchmarks_gemm_fp32_compute32_scale_sweep_l2_error_vs_cublas_dgemm_ref.sh"
    "benchmarks_gemm_fp32_compute32_scale_sweep_raw_l2_error_vs_cublas_dgemm_ref.sh"
    "benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemm_fp16_kernels_scale_sweep_raw_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh"
    "benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref.sh"
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
echo "GEMM scale-sweep batch summary"
echo "================================================================"
echo "Passed (${#PASSED[@]}):"
for s in "${PASSED[@]}"; do echo "  ok    $s"; done
echo "Failed (${#FAILED[@]}):"
for s in "${FAILED[@]}"; do echo "  FAIL  $s"; done

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0

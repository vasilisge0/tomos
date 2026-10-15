#!/bin/bash
# Runs every focused, promoted-kernels-only correctness check in this
# codebase (see scripts.md's own "Row summary" table for what each
# one covers), one after another, and reports a final PASS/FAIL summary.
# Does NOT run the much larger full exploratory suites (verify_mv_fp32_
# 16_gevm.sh, verify_mm_fp32_16_gemm.sh, etc. -- see scripts.md's own
# "Full exploratory suites" section) -- this is the fast "did I break a
# promoted kernel" sweep, not a full regression sweep.
#
# Each sub-script builds its own CMake target (only that one target --
# never rebuilds tomos_cuda's kernel sources unless something upstream of
# it actually changed) and runs the resulting binary directly; none of
# them write a CSV or generate a plot (correctness checks are pass/fail
# only, nothing to plot).
#
# A single script FAILING (a real correctness FAILURE, not a build
# error) does NOT stop this script from running the rest -- every check
# runs regardless, and the failures are called out in the final summary.
# This script's own exit code is the number of failed checks (0 if all
# passed).
#
# Usage:
#   scripts/correctness_all.sh
#       Runs all 6 checks with their own defaults (no per-script
#       customization here -- run an individual scripts/verify_*.sh
#       directly for that).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@"
fi

# One entry per Row in scripts.md's own "Row summary" table.
CHECKS=(
    "Row 1: GEVM 32-bit recon, NO_TRANSPOSE|verify_gemv_fp32_recon_plot_promoted.sh"
    "Row 2: GEVM 32-bit recon, TRANSPOSE|verify_mv_fp32_transpose_promoted.sh"
    "Row 3: GEVM 16-bit pure_half, NO_TRANSPOSE|verify_gemv_fp32_16_plot_promoted.sh"
    "Row 4: GEVM 16-bit pure_half, TRANSPOSE|verify_gemv_fp32_16_pure_half_transpose_plot_promoted.sh"
    "Row 5: GEMM 32-bit recon|verify_mm_fp32_recon_plot_promoted.sh"
    "Row 6: GEMM 16-bit tensor-core|verify_mm_fp16_bf16_pure_plot_promoted.sh"
)

declare -a RESULTS=()
FAIL_COUNT=0

for entry in "${CHECKS[@]}"; do
    desc="${entry%%|*}"
    script="${entry#*|}"

    echo
    echo "================================================================"
    echo "==> $desc"
    echo "==> $script"
    echo "================================================================"

    if "$REPO_ROOT/scripts/$script"; then
        RESULTS+=("PASS  $desc  ($script)")
    else
        RESULTS+=("FAIL  $desc  ($script)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo
echo "================================================================"
echo "==> Correctness summary"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "ALL $((${#CHECKS[@]})) CORRECTNESS CHECKS PASSED"
else
    echo "$FAIL_COUNT of ${#CHECKS[@]} CORRECTNESS CHECKS FAILED (see FAIL lines above)"
fi

exit "$FAIL_COUNT"

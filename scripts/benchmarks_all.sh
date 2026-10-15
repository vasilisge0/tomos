#!/bin/bash
# Runs every performance (GFlop/s vs vendor baseline) benchmark in this
# codebase (see scripts.md's own "Row summary" table), one after
# another, and generates every resulting plot. Five of the six already
# auto-invoke MATLAB themselves at the end of their own run; the sixth
# (fp16_bf16_pure_vs_cutlass_ref_matched.sh) only writes a CSV and does
# NOT plot itself, so this script explicitly calls its own plotting
# wrapper (scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh) right after it
# -- "also call the scripts to generate plots" is otherwise a no-op for
# the other five, which already do this on their own.
#
# This is a LOT of GPU time (six full shape sweeps, several reaching
# multi-billion-element/tens-of-GB shapes) -- expect this to run for
# hours, not minutes. Run an individual scripts/benchmarks_*.sh (or
# scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh) directly instead if
# you only need one row's own results.
#
# A single script FAILING does NOT stop this script from running the
# rest -- every benchmark runs regardless, and failures are called out
# in the final summary. This script's own exit code is the number of
# failed steps (0 if all succeeded).
#
# Usage:
#   scripts/benchmarks_all.sh
#       Runs all 6 benchmarks with their own default full sweeps (no
#       per-script customization here -- run an individual scripts/
#       benchmarks_*.sh directly, with its own arguments, for that).

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
# Fields: description|script|extra plot script (empty if the script
# above already plots itself).
BENCHMARKS=(
    "Row 1: GEVM 32-bit recon, NO_TRANSPOSE|benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh|"
    "Row 2: GEVM 32-bit recon, TRANSPOSE|benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh|"
    "Row 3: GEVM 16-bit pure_half, NO_TRANSPOSE|benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh|"
    "Row 4: GEVM 16-bit pure_half, TRANSPOSE|benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh|"
    "Row 5: GEMM 32-bit recon|benchmarks_gemm_format_32_16_compute_32.sh|"
    "Row 6: GEMM 16-bit tensor-core|fp16_bf16_pure_vs_cutlass_ref_matched.sh|plot_fp16_bf16_pure_vs_cutlass_ref.sh"
)

declare -a RESULTS=()
FAIL_COUNT=0

for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r desc script plot_script <<< "$entry"

    echo
    echo "================================================================"
    echo "==> $desc"
    echo "==> $script"
    echo "================================================================"

    if "$REPO_ROOT/scripts/$script"; then
        step_ok=true
    else
        step_ok=false
    fi

    if [ "$step_ok" = true ] && [ -n "$plot_script" ]; then
        echo
        echo "==> Plotting: $plot_script"
        if ! "$REPO_ROOT/scripts/$plot_script"; then
            step_ok=false
        fi
    fi

    if [ "$step_ok" = true ]; then
        RESULTS+=("OK    $desc  ($script)")
    else
        RESULTS+=("FAIL  $desc  ($script)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo
echo "================================================================"
echo "==> Benchmarks summary"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "ALL ${#BENCHMARKS[@]} BENCHMARKS COMPLETED (see build/benchmarks/output/plots/ for the plots)"
else
    echo "$FAIL_COUNT of ${#BENCHMARKS[@]} BENCHMARKS FAILED (see FAIL lines above)"
fi

exit "$FAIL_COUNT"

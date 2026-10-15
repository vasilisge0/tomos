#!/bin/bash
# Runs every L2/Frobenius-relative-error-vs-genuine-FP32 benchmark in
# this codebase (see scripts.md's own "Row summary" table), one
# after another -- each row TWICE: once with its own default (raw,
# un-normalized) SCALES, and once with --normalized (see any *_l2_
# error_vs_cublas_ref.sh's own --normalized comment for what that does
# and why the SCALES list itself can differ between the two runs). All
# 12 runs already auto-invoke MATLAB themselves, once per normal_scale
# tier in their own SCALES sweep -- "also call the scripts to generate
# plots" is a no-op here, every script below already does this on its
# own; nothing extra to add. The normalized run's own plots land with a
# "_normalized" tag (e.g. ..._scale4096_normalized.png), so they never
# overwrite the raw run's own plots.
#
# This is an ENORMOUS amount of GPU time -- 6 rows x 2 modes (raw +
# normalized) x up to 8 scale tiers each x a full shape sweep per tier
# (up to 14 shapes for GEVM, 12-13 for GEMM). Expect this to run for
# many hours. Run an individual scripts/benchmarks_*_l2_error_vs_
# cublas_ref.sh directly (optionally with its own --scales/--/--
# normalized overrides, see that script's own header) instead if you
# only need one row's own results, one mode, or a subset of scale
# tiers/shapes.
#
# A single script FAILING does NOT stop this script from running the
# rest -- every error sweep runs regardless, and failures are called out
# in the final summary. This script's own exit code is the number of
# failed steps (0 if all succeeded).
#
# Usage:
#   scripts/error_all.sh
#       Runs all 6 rows, both raw and --normalized, with their own
#       default SCALES x shape lists (no per-script customization here
#       -- run an individual scripts/benchmarks_*_l2_error_vs_cublas_
#       ref.sh directly, with its own arguments, for that).

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
ERROR_SWEEPS=(
    "Row 1: GEVM 32-bit recon, NO_TRANSPOSE|benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh"
    "Row 2: GEVM 32-bit recon, TRANSPOSE|benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_ref.sh"
    "Row 3: GEVM 16-bit pure_half, NO_TRANSPOSE|benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh"
    "Row 4: GEVM 16-bit pure_half, TRANSPOSE|benchmarks_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.sh"
    "Row 5: GEMM 32-bit recon|benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh"
    "Row 6: GEMM 16-bit tensor-core|benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh"
)

# Each row runs once per mode -- "" (raw/default SCALES) and
# "--normalized" (see this script's own header comment).
MODES=("" "--normalized")

declare -a RESULTS=()
FAIL_COUNT=0
TOTAL=0

for entry in "${ERROR_SWEEPS[@]}"; do
    desc="${entry%%|*}"
    script="${entry#*|}"

    for mode in "${MODES[@]}"; do
        TOTAL=$((TOTAL + 1))
        mode_desc="raw"
        mode_args=()
        if [ -n "$mode" ]; then
            mode_desc="normalized"
            mode_args=("$mode")
        fi

        echo
        echo "================================================================"
        echo "==> $desc ($mode_desc)"
        echo "==> $script ${mode_args[*]}"
        echo "================================================================"

        if "$REPO_ROOT/scripts/$script" "${mode_args[@]}"; then
            RESULTS+=("OK    $desc ($mode_desc)  ($script ${mode_args[*]})")
        else
            RESULTS+=("FAIL  $desc ($mode_desc)  ($script ${mode_args[*]})")
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
done

echo
echo "================================================================"
echo "==> Error sweeps summary"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "ALL $TOTAL ERROR SWEEPS COMPLETED (see build/benchmarks/output/plots/error/ for the plots)"
else
    echo "$FAIL_COUNT of $TOTAL ERROR SWEEPS FAILED (see FAIL lines above)"
fi

exit "$FAIL_COUNT"

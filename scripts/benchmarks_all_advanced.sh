#!/bin/bash
# "_advanced" variant of scripts/benchmarks_all.sh -- runs every "_
# advanced.sh" performance benchmark in this codebase (see scripts.md's
# own "Row summary" table for what each row covers; each "_advanced.sh"
# is its own row's sibling script, rebuilding a byte-for-byte-identical
# "_advanced.cu" binary that swaps the fixed, shape-size-dependent `reps`
# heuristic for a caller-controlled warmup/iteration count instead -- see
# any one of those .cu files' own header for the exact timing-methodology
# change), one after another. Unlike scripts/benchmarks_all.sh, EVERY one
# of the six "_advanced.sh" scripts below already self-plots (including
# the GEMM 16-bit tensor-core row, whose own non-advanced sibling does
# NOT self-plot -- its "_advanced.sh" was written to close that gap), so
# no extra plot-wrapper call is needed here.
#
# All plots land with an "_avg" tag (e.g. plot_gemv_fp32_recon_vs_
# cutlass_ref_square_avg.png), ALONGSIDE (never overwriting) the plain
# scripts/benchmarks_all.sh's own plots.
#
# This is a LOT of GPU time -- six full shape sweeps, each now ALSO
# multiplied by (warmup + iterations) GPU launches per shape instead of
# the plain scripts' own lighter single-warmup/shape-dependent-reps
# timing. Expect this to run for hours, not minutes. Run an individual
# scripts/*_advanced.sh directly instead if you only need one row's own
# results.
#
# A single script FAILING does NOT stop this script from running the
# rest -- every benchmark runs regardless, and failures are called out
# in the final summary. This script's own exit code is the number of
# failed steps (0 if all succeeded).
#
# Usage:
#   scripts/benchmarks_all_advanced.sh
#       Runs all 6 advanced benchmarks with 10 warmup runs and 10 timed-
#       and-averaged iterations per shape (the defaults).
#   scripts/benchmarks_all_advanced.sh --warmup 20 --iters 50
#       Same, but 20 warmup runs and 50 timed-and-averaged iterations
#       per shape for every row.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --warmup N / --iters N (both optional, default 10/10) -- forwarded
# verbatim to every sub-script below.
WARMUP=10
ITERS=10
while [ "$#" -gt 0 ]; do
    case "$1" in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        *) echo "error: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" --warmup "$WARMUP" --iters "$ITERS"
fi

# One entry per Row in scripts.md's own "Row summary" table.
BENCHMARKS=(
    "Row 1: GEVM 32-bit recon, NO_TRANSPOSE|benchmarks_gemv_fp32_recon_vs_cutlass_ref_advanced.sh"
    "Row 2: GEVM 32-bit recon, TRANSPOSE|benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh"
    "Row 3: GEVM 16-bit pure_half, NO_TRANSPOSE|benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref_advanced.sh"
    "Row 4: GEVM 16-bit pure_half, TRANSPOSE|benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_advanced.sh"
    "Row 5: GEMM 32-bit recon|benchmarks_gemm_format_32_16_compute_32_advanced.sh"
    "Row 6: GEMM 16-bit tensor-core|fp16_bf16_pure_vs_cutlass_ref_matched_advanced.sh"
)

declare -a RESULTS=()
FAIL_COUNT=0

for entry in "${BENCHMARKS[@]}"; do
    desc="${entry%%|*}"
    script="${entry#*|}"

    echo
    echo "================================================================"
    echo "==> $desc"
    echo "==> $script (warmup=$WARMUP iters=$ITERS)"
    echo "================================================================"

    if "$REPO_ROOT/scripts/$script" --warmup "$WARMUP" --iters "$ITERS"; then
        RESULTS+=("OK    $desc  ($script)")
    else
        RESULTS+=("FAIL  $desc  ($script)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo
echo "================================================================"
echo "==> Advanced benchmarks summary (warmup=$WARMUP iters=$ITERS)"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "ALL ${#BENCHMARKS[@]} ADVANCED BENCHMARKS COMPLETED (see build/benchmarks/output/plots/ for the *_avg plots)"
else
    echo "$FAIL_COUNT of ${#BENCHMARKS[@]} ADVANCED BENCHMARKS FAILED (see FAIL lines above)"
fi

exit "$FAIL_COUNT"

#!/bin/bash
# Regenerates plot_gemm_format_32_16_compute_16_{square,rect}.{png,fig}
# (scripts/plot_fp16_bf16_pure_vs_cutlass_ref.m) from already-saved CSV
# data -- does NOT run any benchmarks itself. Useful for re-plotting after
# changing the .m script's own formatting, or any time you just want the
# plots refreshed without re-running scripts/fp16_bf16_pure_vs_cutlass_
# ref_matched.sh.
#
# Usage:
#   scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh
#       Uses the most recently written fp16_bf16_pure_vs_cutlass_ref_
#       matched_*.csv under build/output/benchmarks/gemm/fp16_compute/data/
#       (see the .m function's own default).
#   scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh /path/to/some.csv
#       Plots that CSV directly instead.
#
# Writes build/output/benchmarks/gemm/fp16_compute/plots/plot_gemm_format_32_16_compute_16_
# {square,rect}.{png,fig}.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if ! command -v matlab >/dev/null 2>&1; then
    echo "error: matlab not found (tried 'module load matlab/R2024a')." >&2
    exit 1
fi

if [ $# -gt 0 ]; then
    echo "==> Plotting: $1"
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_fp16_bf16_pure_vs_cutlass_ref('$1')"
else
    echo "==> Plotting: most recent fp16_bf16_pure_vs_cutlass_ref_matched_*.csv"
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_fp16_bf16_pure_vs_cutlass_ref()"
fi

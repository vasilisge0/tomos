#!/bin/bash
# Regenerates plot_gemm_format_32_16_compute_32_{square,rect}.{png,fig}
# (scripts/plot_gemm_format_32_16_compute_32.m) from already-saved CSV
# data -- does NOT run any benchmarks itself. Useful for re-plotting after
# changing the .m script's own formatting, or any time you just want the
# plots refreshed without re-running scripts/benchmarks_gemm_format_32_
# 16_compute_32.sh.
#
# Usage:
#   scripts/plot_gemm_format_32_16_compute_32.sh
#       Uses the most recently written fp32_recon_vs_cutlass_ref_*.csv
#       under build/output/benchmarks/gemm/fp32_compute/data/ (see the .m
#       function's own default -- that directory is shared with
#       fp32_recon_vs_cutlass_ref.sh's own runs, so this picks up
#       whichever script ran last).
#   scripts/plot_gemm_format_32_16_compute_32.sh /path/to/some.csv
#       Plots that CSV directly instead.
#
# Writes build/output/benchmarks/gemm/fp32_compute/plots/plot_gemm_format_32_16_compute_32_
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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32('$1')"
else
    echo "==> Plotting: most recent fp32_recon_vs_cutlass_ref_*.csv"
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_format_32_16_compute_32()"
fi

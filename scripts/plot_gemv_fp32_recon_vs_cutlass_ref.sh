#!/bin/bash
# Regenerates plot_gemv_fp32_recon_vs_cutlass_ref_{square,tall}.{png,fig}
# (scripts/plot_gemv_fp32_recon_vs_cutlass_ref.m) from already-saved CSV
# data -- does NOT run any benchmarks itself. Useful for re-plotting after
# changing the .m script's own formatting, or any time you just want the
# plots refreshed without re-running scripts/benchmarks_gemv_fp32_recon_
# vs_cutlass_ref.sh.
#
# Usage:
#   scripts/plot_gemv_fp32_recon_vs_cutlass_ref.sh
#       Uses the most recently written mv_fp32_recon_vs_cutlass_ref_*.csv
#       under build/mv_fp32_recon_vs_cutlass_ref_output/ (see the .m
#       function's own default).
#   scripts/plot_gemv_fp32_recon_vs_cutlass_ref.sh /path/to/some.csv
#       Plots that CSV directly instead.
#
# Writes build/benchmarks/output/plots/plot_gemv_fp32_recon_vs_cutlass_
# ref_{square,tall}.{png,fig}.

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
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref('$1')"
else
    echo "==> Plotting: most recent mv_fp32_recon_vs_cutlass_ref_*.csv"
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref()"
fi

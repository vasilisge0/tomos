#!/bin/bash
# Plots, as a grouped bar chart, the SPEEDUP of the mixed-precision LSQR
# configurations over preconditioned FP32 (the non-preconditioned FP32
# configuration is intentionally left out), one bar per test matrix. Pure
# post-processing: it reads the lsqr_performance.csv that
# scripts/run_lsqr_precision_performance_compare.sh already wrote and runs
# scripts/plot_lsqr_speedup.m on it -- no solver runs, no GPU or SLURM
# allocation needed, finishes in seconds. (Re-running the performance
# script is what takes hours; this reuses its measurements.)
#
#   speedup = median solve time (FP32, precond)
#             / median solve time (method), per matrix
# for the three mixed-precision configurations:
#   fp16_explicit       TOMOS FP16 (native, explicit refine)
#   fp16_implicit       TOMOS FP16 (implicit, safe-decode)
#   fp16_implicit_fast  TOMOS FP16 (implicit, fast-decode)
# The ratio is of TOTAL solve time, so it includes iteration-count
# differences between configurations -- see plot_lsqr_speedup.m's own header
# comment.
#
# Usage:
#   scripts/plot_lsqr_precision_speedup.sh [--csv PATH] [--output-dir DIR]
#   --csv         Performance CSV to read (default:
#                 build/output/lsqr/performance/data/lsqr_performance.csv).
#   --output-dir  Where to write the plot (default:
#                 build/output/lsqr/performance/plots).
#
# Writes <output-dir>/lsqr_speedup_barplot.png / .fig -- a different name
# from the performance bar chart (lsqr_performance_barplot.*) and from the
# convergence plots (lsqr_residual_history.*), so nothing is overwritten.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="$REPO_ROOT/build/output/lsqr/performance/data/lsqr_performance.csv"
OUTPUT_DIR=""

usage() {
    echo "Usage: $0 [--csv PATH] [--output-dir DIR]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --csv) CSV_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

if [ ! -f "$CSV_PATH" ]; then
    echo "error: performance CSV not found: $CSV_PATH" >&2
    echo "       generate it first with scripts/run_lsqr_precision_performance_compare.sh" >&2
    echo "       (or point --csv at an existing lsqr_performance.csv)." >&2
    exit 1
fi
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$REPO_ROOT/build/output/lsqr/performance/plots"

echo "==> Plotting speedup over FP32 (precond) from $CSV_PATH"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); plot_lsqr_speedup('$CSV_PATH', '$OUTPUT_DIR');"
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "$MATLAB_CALL"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"$MATLAB_CALL\"" >&2
    exit 1
fi

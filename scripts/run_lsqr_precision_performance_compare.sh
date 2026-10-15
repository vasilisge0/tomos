#!/bin/bash
# Performance counterpart of scripts/run_lsqr_precision_compare.sh: runs the
# SAME five LSQR precision/preconditioning configurations (same flags, same
# tolerance/iteration/restart defaults) against one or more (A, x_true, b)
# problems, but instead of writing per-iteration residual histories and
# plotting convergence curves, it records how LONG each solve takes and
# plots that as a grouped bar chart (one group per configuration, one bar
# per matrix). Its outputs deliberately use different names/directory than
# the convergence plots, so the two never overwrite each other.
#
# The five configurations (identical to run_lsqr_precision_compare.sh --
# see that script's own header for the full writeup):
#   fp32_no_precond   1. All FP32, no preconditioner.
#   fp32_precond      2. All FP32, with the count-sketch + QR preconditioner.
#   fp16_explicit     3. --mv-type fp16, preconditioner, --explicit-duplicates.
#   fp16_implicit     4. --mv-type fp16, preconditioner, --implicit-duplicates
#                        (safe-decode "TOMOS FP16 (implicit)" kernels).
#   fp16_implicit_fast 5. Same as 4 with TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1
#                        (comparison-only env-var toggle, see
#                        src/dispatch/fp32_16_registries.cpp).
#
# What is measured: the runner's own "solve time" (lsqr_solve() wall time,
# ended with a cudaDeviceSynchronize; matrix loading, A's precision cast,
# and the preconditioner build are NOT included). Each configuration is run
# --repeats times as SEPARATE processes (default 3) and the MEDIAN solve time
# is plotted, with min/max as error bars -- every run is a fresh process, so
# each timing includes that process's own one-time first-call costs (cold
# GPU clocks, first workspace allocation) exactly as a real single solve
# would pay them. Because iteration counts differ between configurations
# (lower-precision configurations can need more iterations to reach the
# same tolerance), a second panel plots time PER ITERATION alongside the
# total, so a config that is slower per matvec is distinguishable from one
# that merely takes more iterations.
#
# COST WARNING: every run re-reads the .mtx file from disk (tens of GB of
# ASCII for the default 1,000,000 x 1,000 matrices, roughly 3-4 minutes per
# run in practice). Total runs = matrices x 5 configurations x --repeats --
# with the two default matrices and the default --repeats 3 that is 30 runs,
# on the order of 1.5-2 hours. Use --repeats 1 for a quick single-sample
# pass. Progress is appended to the CSV after each (matrix, configuration)
# finishes, so an interrupted run keeps everything completed so far.
#
# Usage:
#   scripts/run_lsqr_precision_performance_compare.sh
#       [--set LABEL A.mtx X.mtx B.mtx]...   (repeatable; LABEL must not
#                                             contain commas)
#       [--matrix-dir DIR] [--device cpu|cuda] [--mv-type T] [--value-type T]
#       [--refine-type T] [--precond-type T] [--tol TOL] [--max-iters N]
#       [--restarts N] [--repeats N] [--output-dir DIR] [--plot-only]
#   With no --set, runs the two established test problems, read from
#   --matrix-dir (default: $MATRIX_DIR if set via .configure.sh, else
#   build/matrices -- same convention as run_lsqr_precision_speedup_by_columns.sh):
#     "cond=1000" tall_thin_1000.mtx
#     "cond=500"  tall_thin_1000_cond500.mtx (same directory; _xtrue/_b siblings)
#   Generate both with: scripts/generate_lsqr_test_matrices.sh --columns 1000
#   --plot-only skips all solver runs and just re-plots an existing
#   DIR/lsqr_performance.csv.
#
# Writes, under DIR (default: build/output/lsqr/performance):
#   data/logs/<label>__<config>__run<i>.log   raw runner output per run
#   data/lsqr_performance.csv                 one row per (matrix, configuration)
#   plots/lsqr_performance_barplot.png / .fig via plot_lsqr_performance.m
# (The convergence plots live in build/output/lsqr/error*/ and are not
# touched.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_lsqr_precision.sh"
MATRIX_DIR="${MATRIX_DIR:-$REPO_ROOT/build/matrices}"

DEVICE="cuda"
MV_TYPE="fp16"
VALUE_TYPE="fp32"
REFINE_TYPE="fp32"
PRECOND_TYPE="fp32"
TOL="1e-7"
MAX_ITERS="5"
RESTARTS="40"
REPEATS="3"
OUTPUT_DIR="$REPO_ROOT/build/output/lsqr/performance"
PLOT_ONLY=0
SET_LABELS=()
SET_A=()
SET_X=()
SET_B=()

usage() {
    echo "Usage: $0 [--set LABEL A.mtx X.mtx B.mtx]... [--matrix-dir DIR] [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T] [--precond-type T] [--tol TOL] [--max-iters N] [--restarts N] [--repeats N] [--output-dir DIR] [--plot-only]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --set)
            [ $# -ge 5 ] || { echo "error: --set needs LABEL A.mtx X.mtx B.mtx." >&2; usage; }
            SET_LABELS+=("$2"); SET_A+=("$3"); SET_X+=("$4"); SET_B+=("$5"); shift 5 ;;
        --matrix-dir) MATRIX_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --mv-type) MV_TYPE="$2"; shift 2 ;;
        --value-type) VALUE_TYPE="$2"; shift 2 ;;
        --refine-type) REFINE_TYPE="$2"; shift 2 ;;
        --precond-type) PRECOND_TYPE="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --restarts) RESTARTS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --plot-only) PLOT_ONLY=1; shift 1 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

case "$REPEATS" in
    ''|*[!0-9]*|0) echo "error: --repeats must be a positive integer (got '$REPEATS')." >&2; exit 1 ;;
esac

if [ "${#SET_LABELS[@]}" -eq 0 ]; then
    SET_LABELS=("cond=1000" "cond=500")
    SET_A=("$MATRIX_DIR/tall_thin_1000.mtx" "$MATRIX_DIR/tall_thin_1000_cond500.mtx")
    SET_X=("$MATRIX_DIR/tall_thin_1000_xtrue.mtx" "$MATRIX_DIR/tall_thin_1000_cond500_xtrue.mtx")
    SET_B=("$MATRIX_DIR/tall_thin_1000_b.mtx" "$MATRIX_DIR/tall_thin_1000_cond500_b.mtx")
fi

CSV_PATH="$OUTPUT_DIR/data/lsqr_performance.csv"
LOG_DIR="$OUTPUT_DIR/data/logs"
PLOTS_DIR="$OUTPUT_DIR/plots"

# Same key -> (extra runner flags, extra env) mapping as
# run_lsqr_precision_compare.sh's five configurations. Flags are
# intentionally word-split when expanded below (none contain spaces or
# paths); the residual-history flag is deliberately NOT passed -- these
# runs are timing-only.
CONFIG_KEYS=(fp32_no_precond fp32_precond fp16_explicit fp16_implicit fp16_implicit_fast)
CONFIG_ARGS=(
    ""
    "--precond --precond-type $PRECOND_TYPE"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --explicit-duplicates"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --implicit-duplicates"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --implicit-duplicates"
)
CONFIG_ENV=(
    ""
    ""
    ""
    ""
    "TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1"
)

if [ "$PLOT_ONLY" -eq 0 ]; then
    if [ "$DEVICE" = "cuda" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
        if [ -z "$JOBID" ]; then
            echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
            echo "       start one first, or pass --device cpu, or use --plot-only to re-plot an existing CSV." >&2
            exit 1
        fi
        echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
        exec srun --jobid="$JOBID" --overlap "$0" "$@"
    fi

    for i in "${!SET_LABELS[@]}"; do
        for f in "${SET_A[$i]}" "${SET_X[$i]}" "${SET_B[$i]}"; do
            [ -f "$f" ] || { echo "error: file not found for set '${SET_LABELS[$i]}': $f" >&2; exit 1; }
        done
    done

    mkdir -p "$LOG_DIR"
    rm -f "$CSV_PATH"

    # Appends one CSV row for (matrix label, config key) aggregated over the
    # given per-run logs: median/min/max solve time, iteration count, ms per
    # iteration, final residual and solution error (from the first log --
    # they don't vary between repeats of the same deterministic solve).
    aggregate() {
        python3 - "$CSV_PATH" "$@" <<'PYEOF'
import csv, os, re, statistics, sys

csv_path, matrix_label, config_key, *logs = sys.argv[1:]
solve, iters, resid, err = [], [], [], []
for path in logs:
    text = open(path).read()
    m_t = re.search(r"==> solve time:\s+([0-9.eE+-]+) ms", text)
    m_i = re.search(r"==> iterations:\s+(\d+)", text)
    m_r = re.search(r"==> final residual:\s+([0-9.eE+-]+)", text)
    m_e = re.search(r"==> solution error:\s+([0-9.eE+-]+)", text)
    if not (m_t and m_i):
        print(f"warning: no timing found in {path}; skipping this repeat", file=sys.stderr)
        continue
    solve.append(float(m_t.group(1)))
    iters.append(int(m_i.group(1)))
    resid.append(float(m_r.group(1)) if m_r else float("nan"))
    err.append(float(m_e.group(1)) if m_e else float("nan"))

if not solve:
    print(f"warning: no successful runs for {matrix_label}/{config_key}; no CSV row written", file=sys.stderr)
    sys.exit(0)

median_ms = statistics.median(solve)
iterations = iters[0]
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["matrix_label", "config_key", "runs", "iterations", "solve_ms_median",
                    "solve_ms_min", "solve_ms_max", "ms_per_iter", "final_residual", "solution_error"])
    w.writerow([matrix_label, config_key, len(solve), iterations, f"{median_ms:.4f}",
                f"{min(solve):.4f}", f"{max(solve):.4f}", f"{median_ms / iterations:.4f}",
                f"{resid[0]:.6e}", f"{err[0]:.6e}"])
print(f"    {matrix_label} / {config_key}: median {median_ms:.3f} ms over {len(solve)} run(s), {iterations} iterations")
PYEOF
    }

    for i in "${!SET_LABELS[@]}"; do
        label="${SET_LABELS[$i]}"
        safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
        for c in "${!CONFIG_KEYS[@]}"; do
            key="${CONFIG_KEYS[$c]}"
            echo "==> [$label] ${key} (${REPEATS} run(s))"
            logs=()
            for r in $(seq 1 "$REPEATS"); do
                log="$LOG_DIR/${safe_label}__${key}__run${r}.log"
                echo "    run $r/$REPEATS -> $log"
                # ${CONFIG_ENV[$c]} and ${CONFIG_ARGS[$c]} are intentionally unquoted (word-split).
                if env ${CONFIG_ENV[$c]} bash "$RUNNER_SCRIPT" \
                        --matrix "${SET_A[$i]}" --x-true "${SET_X[$i]}" --rhs "${SET_B[$i]}" \
                        --device "$DEVICE" --value-type "$VALUE_TYPE" --refine-type "$REFINE_TYPE" \
                        --tol "$TOL" --max-iters "$MAX_ITERS" --restarts "$RESTARTS" \
                        ${CONFIG_ARGS[$c]} > "$log" 2>&1; then
                    logs+=("$log")
                else
                    echo "    warning: run $r failed (see $log); excluding it from the aggregate" >&2
                fi
            done
            if [ "${#logs[@]}" -gt 0 ]; then
                aggregate "$label" "$key" "${logs[@]}"
            else
                echo "    warning: every run failed for [$label] $key; no CSV row written" >&2
            fi
        done
    done

    echo
    echo "==> Performance CSV written to: $CSV_PATH"
fi

[ -f "$CSV_PATH" ] || { echo "error: $CSV_PATH not found -- nothing to plot." >&2; exit 1; }

echo
echo "==> Plotting performance bar chart"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); plot_lsqr_performance('$CSV_PATH', '$PLOTS_DIR');"
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "$MATLAB_CALL"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"$MATLAB_CALL\"" >&2
fi

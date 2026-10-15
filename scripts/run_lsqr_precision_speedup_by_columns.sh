#!/bin/bash
# Speedup (performance) tests of the mixed-precision LSQR configurations over
# preconditioned FP32, across several problem WIDTHS (number of columns) and
# two condition numbers, all plotted on ONE grouped bar chart -- bars are
# grouped per number of columns.
#
# Problems (1,000,000 rows each; default --matrix-dir is build/matrices --
# generate them with scripts/generate_lsqr_test_matrices.sh).
# Each problem "variant" is a set of matrices, one per column count n:
#   variant cond1000      label "cond=1000"             tall_thin_<n>.mtx
#   variant cond500       label "cond=500"              tall_thin_<n>_cond500.mtx
#   variant cond1000norm  label "cond=1000-normalized"  tall_thin_<n>_cond1000_norm.mtx
#                         (each with its _xtrue.mtx / _b.mtx companions)
# cond1000 and cond1000norm are the same matrices at two scales (spectrum
# divided by its largest singular value in the normalized one). By default
# the cond1000 and cond500 variants run (--variants "cond1000 cond500").
# Adding "--variants cond1000norm" runs only the normalized set, appending to
# the SAME CSV (rows already there are skipped), after which the plot step
# draws BOTH bar charts: the original cond=1000 vs cond=500 one, and a new
# unnormalized-vs-normalized cond=1000 one.
#
# Configurations run per problem (same flags/defaults as
# run_lsqr_precision_compare.sh / run_lsqr_precision_performance_compare.sh:
# tol 1e-7, 5 iterations per restart, 40 restarts, fp32 value/refine/precond
# types, --mv-type fp16). The non-preconditioned FP32 configuration is NOT
# run -- a speedup plot over FP32-with-preconditioner doesn't use it:
#   fp32_precond        baseline: all FP32, count-sketch + QR preconditioner
#   fp16_explicit       TOMOS FP32 / FP16 (explicit)
#   fp16_implicit       TOMOS FP32 / FP16 (implicit)
#   fp16_implicit_fast  TOMOS FP32 / FP16 (implicit, fast-decode;
#                       TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1)
#
# Speedup per (columns, condition number, method) =
#   median solve time (fp32_precond) / median solve time (method)
# -- of TOTAL solve time (lsqr_solve() wall time as reported by the runner),
# so it includes iteration-count differences between configurations.
#
# COST / RESUMING: every run re-reads its .mtx from disk -- ~23.5 / 47 / 70.5
# GB of ASCII for 1000 / 2000 / 3000 columns, on the order of 3.5 / 7 / 10
# minutes per run. One repeat of everything is 6 problems x 4 configurations
# = 24 runs, roughly 2.5-3 hours -- about a whole interactive job allocation,
# and --repeats 3 would be ~3x that. So this script is RESUMABLE: results are
# appended to the CSV after each (columns, cond, configuration) finishes, and
# pairs already present in the CSV are skipped on the next invocation, so you
# can simply re-run it inside a fresh job to continue. Use --fresh to discard
# the existing CSV and start over. Failed runs (e.g. the 3000-column case
# running out of host/GPU memory) are logged, excluded, and NOT recorded, so
# a later re-run retries them; the plot leaves those bars empty.
#
# Usage:
#   scripts/run_lsqr_precision_speedup_by_columns.sh
#       [--columns "1000 2000 3000"] [--variants "cond1000 cond500"]
#       [--matrix-dir DIR] [--repeats N]
#       [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T]
#       [--precond-type T] [--tol TOL] [--max-iters N] [--restarts N]
#       [--output-dir DIR] [--fresh] [--plot-only]
#   --columns     Space-separated column counts; each needs the matrix files
#                 of every selected variant (+ _xtrue/_b) in --matrix-dir.
#   --variants    Space-separated variants to run: cond1000, cond500,
#                 cond1000norm (see above). Default "cond1000 cond500".
#   --repeats     Separate-process runs per configuration, median plotted
#                 (default 1 -- see the cost note above).
#   --plot-only   Skip all solver runs; just re-plot the existing CSV.
#
# Writes, under DIR (default: build/output/lsqr/speedup_by_columns):
#   logs/n<cols>__<cond>__<config>__run<i>.log   raw runner output per run
#   lsqr_speedup_by_columns.csv                  one row per (cols, cond, config)
#   lsqr_speedup_by_columns_barplot.png / .fig   cond=1000 vs cond=500
#   lsqr_speedup_by_columns_normalized_barplot.png / .fig
#                                                cond=1000 unnormalized vs normalized
#                                                (drawn once cond1000norm rows exist)
#                                                both via plot_lsqr_speedup_by_columns.m
# (Separate from the convergence plots in build/output/lsqr/error*/ and from
# the performance/speedup plots in build/output/lsqr/performance/.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
# LSQR_RUNNER_SCRIPT is an override hook (used to test this driver against a
# stub without a GPU); normally unset.
RUNNER_SCRIPT="${LSQR_RUNNER_SCRIPT:-$REPO_ROOT/scripts/run_lsqr_precision.sh}"
MATRIX_DIR="${MATRIX_DIR:-$REPO_ROOT/build/matrices}"

COLUMNS_LIST="1000 2000 3000"
VARIANTS="cond1000 cond500"
DEVICE="cuda"
MV_TYPE="fp16"
VALUE_TYPE="fp32"
REFINE_TYPE="fp32"
PRECOND_TYPE="fp32"
TOL="1e-7"
MAX_ITERS="5"
RESTARTS="40"
REPEATS="1"
OUTPUT_DIR="$REPO_ROOT/build/output/lsqr/speedup_by_columns"
FRESH=0
PLOT_ONLY=0

usage() {
    echo "Usage: $0 [--columns \"1000 2000 3000\"] [--variants \"cond1000 cond500\"] [--matrix-dir DIR] [--repeats N] [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T] [--precond-type T] [--tol TOL] [--max-iters N] [--restarts N] [--output-dir DIR] [--fresh] [--plot-only]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --columns) COLUMNS_LIST="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --matrix-dir) MATRIX_DIR="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --mv-type) MV_TYPE="$2"; shift 2 ;;
        --value-type) VALUE_TYPE="$2"; shift 2 ;;
        --refine-type) REFINE_TYPE="$2"; shift 2 ;;
        --precond-type) PRECOND_TYPE="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --restarts) RESTARTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --fresh) FRESH=1; shift 1 ;;
        --plot-only) PLOT_ONLY=1; shift 1 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

case "$REPEATS" in
    ''|*[!0-9]*|0) echo "error: --repeats must be a positive integer (got '$REPEATS')." >&2; exit 1 ;;
esac

CSV_PATH="$OUTPUT_DIR/lsqr_speedup_by_columns.csv"
LOG_DIR="$OUTPUT_DIR/logs"

# Variant table: key -> (CSV label, matrix file suffix). Labels must not
# contain commas (CSV) or spaces (they appear in log file names).
VARIANT_KEYS=(cond1000 cond500 cond1000norm)
VARIANT_LABELS=("cond=1000" "cond=500" "cond=1000-normalized")
VARIANT_SUFFIXES=("" "_cond500" "_cond1000_norm")

COND_LABELS=()
COND_SUFFIXES=()
for v in $VARIANTS; do
    found=0
    for vi in "${!VARIANT_KEYS[@]}"; do
        if [ "${VARIANT_KEYS[$vi]}" = "$v" ]; then
            COND_LABELS+=("${VARIANT_LABELS[$vi]}")
            COND_SUFFIXES+=("${VARIANT_SUFFIXES[$vi]}")
            found=1
        fi
    done
    [ "$found" -eq 1 ] || { echo "error: unknown variant '$v' (valid: ${VARIANT_KEYS[*]})." >&2; exit 1; }
done

# Same key -> (runner flags, env) mapping as the other LSQR compare scripts,
# minus fp32_no_precond. Flags are intentionally word-split when expanded
# below (none contain spaces or paths); no residual-history flag -- these
# runs are timing-only.
CONFIG_KEYS=(fp32_precond fp16_explicit fp16_implicit fp16_implicit_fast)
CONFIG_ARGS=(
    "--precond --precond-type $PRECOND_TYPE"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --explicit-duplicates"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --implicit-duplicates"
    "--mv-type $MV_TYPE --precond --precond-type $PRECOND_TYPE --implicit-duplicates"
)
CONFIG_ENV=(
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

    # Validate every input file up front so a typo doesn't surface hours in.
    for n in $COLUMNS_LIST; do
        for s in "${COND_SUFFIXES[@]}"; do
            for f in "$MATRIX_DIR/tall_thin_${n}${s}.mtx" \
                     "$MATRIX_DIR/tall_thin_${n}${s}_xtrue.mtx" \
                     "$MATRIX_DIR/tall_thin_${n}${s}_b.mtx"; do
                [ -f "$f" ] || { echo "error: file not found: $f" >&2; exit 1; }
            done
        done
    done

    mkdir -p "$LOG_DIR"
    if [ "$FRESH" -eq 1 ]; then
        rm -f "$CSV_PATH"
    fi

    # Appends one CSV row for (columns, cond label, config key) aggregated
    # over the given per-run logs: median/min/max solve time, iteration
    # count, ms per iteration, final residual and solution error (from the
    # first log -- they don't vary between repeats of a deterministic solve).
    aggregate() {
        python3 - "$CSV_PATH" "$@" <<'PYEOF'
import csv, os, re, statistics, sys

csv_path, n_cols, cond_label, config_key, *logs = sys.argv[1:]
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
    print(f"warning: no successful runs for n={n_cols}/{cond_label}/{config_key}; no CSV row written", file=sys.stderr)
    sys.exit(0)

median_ms = statistics.median(solve)
iterations = iters[0]
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["columns", "cond_label", "config_key", "runs", "iterations", "solve_ms_median",
                    "solve_ms_min", "solve_ms_max", "ms_per_iter", "final_residual", "solution_error"])
    w.writerow([n_cols, cond_label, config_key, len(solve), iterations, f"{median_ms:.4f}",
                f"{min(solve):.4f}", f"{max(solve):.4f}", f"{median_ms / iterations:.4f}",
                f"{resid[0]:.6e}", f"{err[0]:.6e}"])
print(f"    n={n_cols} {cond_label} / {config_key}: median {median_ms:.3f} ms over {len(solve)} run(s), {iterations} iterations")
PYEOF
    }

    for n in $COLUMNS_LIST; do
        for ci in "${!COND_LABELS[@]}"; do
            label="${COND_LABELS[$ci]}"
            suffix="${COND_SUFFIXES[$ci]}"
            A="$MATRIX_DIR/tall_thin_${n}${suffix}.mtx"
            X="$MATRIX_DIR/tall_thin_${n}${suffix}_xtrue.mtx"
            B="$MATRIX_DIR/tall_thin_${n}${suffix}_b.mtx"
            for c in "${!CONFIG_KEYS[@]}"; do
                key="${CONFIG_KEYS[$c]}"
                if [ -f "$CSV_PATH" ] && grep -q "^${n},${label},${key}," "$CSV_PATH"; then
                    echo "==> [n=$n, $label] $key: already in $CSV_PATH -- skipping (use --fresh to redo)"
                    continue
                fi
                echo "==> [n=$n, $label] $key ($REPEATS run(s))"
                logs=()
                for r in $(seq 1 "$REPEATS"); do
                    log="$LOG_DIR/n${n}__${label}__${key}__run${r}.log"
                    echo "    run $r/$REPEATS -> $log"
                    # ${CONFIG_ENV[$c]} and ${CONFIG_ARGS[$c]} are intentionally unquoted (word-split).
                    if env ${CONFIG_ENV[$c]} bash "$RUNNER_SCRIPT" \
                            --matrix "$A" --x-true "$X" --rhs "$B" \
                            --device "$DEVICE" --value-type "$VALUE_TYPE" --refine-type "$REFINE_TYPE" \
                            --tol "$TOL" --max-iters "$MAX_ITERS" --restarts "$RESTARTS" \
                            ${CONFIG_ARGS[$c]} > "$log" 2>&1; then
                        logs+=("$log")
                    else
                        echo "    warning: run $r failed (see $log); excluding it from the aggregate" >&2
                    fi
                done
                if [ "${#logs[@]}" -gt 0 ]; then
                    aggregate "$n" "$label" "$key" "${logs[@]}"
                else
                    echo "    warning: every run failed for [n=$n, $label] $key; nothing recorded (a re-run will retry it)" >&2
                fi
            done
        done
    done

    echo
    echo "==> Speedup-by-columns CSV written to: $CSV_PATH"
fi

[ -f "$CSV_PATH" ] || { echo "error: $CSV_PATH not found -- nothing to plot." >&2; exit 1; }

echo
echo "==> Plotting speedup-by-columns bar chart"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
# Two figures from the one CSV. Each is drawn only if every condition it
# needs has rows in the CSV (plot_lsqr_speedup_by_columns errors otherwise);
# a missing one just prints a warning and the other is still drawn.
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); "
MATLAB_CALL+="try, plot_lsqr_speedup_by_columns('$CSV_PATH', '$OUTPUT_DIR', 'lsqr_speedup_by_columns_barplot', {'cond=1000','cond=500'}); catch e, warning('cond=1000 vs cond=500 plot skipped: %s', e.message); end; "
MATLAB_CALL+="try, plot_lsqr_speedup_by_columns('$CSV_PATH', '$OUTPUT_DIR', 'lsqr_speedup_by_columns_normalized_barplot', {'cond=1000','cond=1000-normalized'}); catch e, warning('normalized vs unnormalized plot skipped: %s', e.message); end;"
if command -v matlab >/dev/null 2>&1; then
    # -nodisplay -nosplash -r (the classic headless-script invocation)
    # instead of -batch: on this compute node, -batch's own graphics-stack
    # startup was hanging trying to reach a display/license resource only
    # the login node has -- -nodisplay skips that init. -r doesn't auto-exit
    # like -batch does, so "exit;" is appended explicitly.
    matlab -nodisplay -nosplash -nodesktop -r "$MATLAB_CALL exit;"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -nodisplay -nosplash -nodesktop -r \"$MATLAB_CALL exit;\"" >&2
fi

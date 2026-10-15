#!/bin/bash
# Thin CLI wrapper around the lsqr_precision_runner CMake target
# (benchmarks/solvers/cuda/lsqr_precision_runner.cu) -- solves A*x = b with
# LSQR at a caller-chosen combination of storage/internal/residual
# precisions, against .mtx files (e.g. from scripts/generate_tall_thin_mtx.sh),
# and reports iteration count, solve time, final residual, and solution
# error against the known x_true. See that .cu file's own header comment
# for the full precision-axis writeup, including --mv-type fp32_16
# (NumericType::FP32_16, the packed wh/wr format) and --implicit-duplicates.
#
# Usage:
#   scripts/run_lsqr_precision.sh --matrix A.mtx --x-true X.mtx --rhs B.mtx
#       [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T]
#       [--precond-type T]
#       [--tol TOL] [--max-iters N] [--max-relres R] [--restarts N]
#       [--precond] [--normalize-columns] [--explicit-duplicates]
#       [--implicit-duplicates] [--residual-history PATH]
#   T in {fp64,fp32,fp16,bf16}. All three type flags default to fp32 (i.e.
#   no flags at all = "all float precision"). --device defaults to cuda.
#   --max-iters is iterations PER restart (default 200); --restarts
#   (default 1) is the number of restart cycles, so the true iteration
#   ceiling is restarts * max-iters -- see the .cu file's own header
#   comment for what a "restart" actually does.
#   --precond adds a count-sketch + QR right-preconditioner (k=2n sketch
#   rows). Its storage AND application both run at --precond-type
#   (default: whatever --value-type is), independent of --mv-type (A's
#   own storage/matvec precision) -- see the .cu file's own header
#   comment for why the sketch/QR step always runs on a throwaway CPU
#   context regardless of --device (no CUDA path exists for it yet).
#   --normalize-columns scales each column of A to unit norm before
#   solving and un-scales the solution afterward -- reduces A's own
#   entry-magnitude spread before it's cast down to --mv-type, computed
#   in double precision from the .mtx data regardless of --mv-type.
#   --explicit-duplicates (only meaningful when --mv-type != --refine-type)
#   stores A ONCE at --mv-type precision (for fp16, PLAIN NATIVE __half,
#   routed to a genuinely-tuned real-fp16 GEMV kernel -- mv_fp16_native,
#   not the generic double-accumulate fallback other unmatched type
#   combinations get), plus a SECOND copy at --refine-type precision, used
#   for the residual recompute at every restart boundary (and this
#   script's own final residual report) instead of A's own --mv-type copy
#   -- so that check isn't capped by --mv-type's own quantization noise.
#   Doubles A's memory footprint while enabled. HARD mutually exclusive
#   with --implicit-duplicates (error if both are given).
#   --implicit-duplicates: a no-extra-memory alternative to
#   --explicit-duplicates. Forces --mv-type to fp32_16 (error if --mv-type
#   was also given as something else) and stores A ONCE, packed
#   (NumericType::FP32_16). Per-iteration matvecs run at the cheap FP16
#   ("pure_half", safe-decode variant -- "TOMOS FP16 (implicit)" in the
#   GEMV performance plots; the fast-decode sibling was tried here too,
#   but on real ill-conditioned matrices its saturate-the-underflow-band-
#   to-zero behavior cost enough accuracy to stall convergence, so this
#   reverted back) compute tier, but the restart-boundary residual
#   recompute (and final residual report) reads that SAME buffer at its
#   FP32 ("accurate") compute tier instead -- no second copy of A needed.
#   HARD mutually exclusive with --explicit-duplicates: the two now select
#   different, incompatible storage schemes for A (FP32_16-packed vs.
#   plain native fp16), not two independent options.
#   --residual-history, if given, writes one relative-residual value per
#   completed LSQR iteration to PATH as plain text ("iteration value",
#   one pair per line, 1-based iteration index).
#
# Examples:
#   # All float precision (mv/value/refine all FP32):
#   scripts/run_lsqr_precision.sh --matrix A.mtx --x-true x.mtx --rhs b.mtx
#
#   # Float residual, half internal precision, A stored as FP16:
#   scripts/run_lsqr_precision.sh --matrix A.mtx --x-true x.mtx --rhs b.mtx \
#       --mv-type fp16 --value-type fp16 --refine-type fp32
#
#   # Same but A stored as FP32 (storage-only sweep point vs the above):
#   scripts/run_lsqr_precision.sh --matrix A.mtx --x-true x.mtx --rhs b.mtx \
#       --mv-type fp32 --value-type fp16 --refine-type fp32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

DEVICE="cuda"
MV_TYPE=""
VALUE_TYPE="fp32"
REFINE_TYPE="fp32"
PRECOND_TYPE=""
TOL="1e-8"
MAX_ITERS="200"
MAX_RELRES="1e-8"
RESTARTS="1"
MATRIX_PATH=""
X_TRUE_PATH=""
RHS_PATH=""
RESIDUAL_HISTORY_PATH=""
PRECOND=0
NORMALIZE_COLUMNS=0
EXPLICIT_DUPLICATES=0
IMPLICIT_DUPLICATES=0

usage() {
    echo "Usage: $0 --matrix A.mtx --x-true X.mtx --rhs B.mtx [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T] [--precond-type T] [--tol TOL] [--max-iters N] [--max-relres R] [--restarts N] [--precond] [--normalize-columns] [--explicit-duplicates] [--implicit-duplicates] [--residual-history PATH]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --matrix) MATRIX_PATH="$2"; shift 2 ;;
        --x-true) X_TRUE_PATH="$2"; shift 2 ;;
        --rhs) RHS_PATH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --mv-type) MV_TYPE="$2"; shift 2 ;;
        --value-type) VALUE_TYPE="$2"; shift 2 ;;
        --refine-type) REFINE_TYPE="$2"; shift 2 ;;
        --precond-type) PRECOND_TYPE="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --max-relres) MAX_RELRES="$2"; shift 2 ;;
        --restarts) RESTARTS="$2"; shift 2 ;;
        --residual-history) RESIDUAL_HISTORY_PATH="$2"; shift 2 ;;
        --precond) PRECOND=1; shift 1 ;;
        --normalize-columns) NORMALIZE_COLUMNS=1; shift 1 ;;
        --explicit-duplicates) EXPLICIT_DUPLICATES=1; shift 1 ;;
        --implicit-duplicates) IMPLICIT_DUPLICATES=1; shift 1 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

# MV_TYPE defaults to empty (not fp32) so it's forwarded to the runner only
# when the caller actually passed --mv-type -- the runner tracks whether
# --mv-type was given explicitly to validate --implicit-duplicates (which
# forces fp32_16) against a conflicting explicit value; always forwarding a
# defaulted "fp32" here would falsely look like an explicit, conflicting
# --mv-type every time --implicit-duplicates is used without --mv-type.

[ -z "$MATRIX_PATH" ] && { echo "error: --matrix is required." >&2; usage; }
[ -z "$X_TRUE_PATH" ] && { echo "error: --x-true is required." >&2; usage; }
[ -z "$RHS_PATH" ] && { echo "error: --rhs is required." >&2; usage; }

case "$DEVICE" in
    cpu|cuda) ;;
    *) echo "error: --device must be 'cpu' or 'cuda' (got '$DEVICE')." >&2; exit 1 ;;
esac

if [ "$DEVICE" = "cuda" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        echo "       or pass --device cpu to run without a GPU allocation." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Building: lsqr_precision_runner"
cmake --build "$BUILD_DIR" --target lsqr_precision_runner -j"$(nproc)"

RUNNER="$BUILD_DIR/benchmarks/solvers/cuda/lsqr_precision_runner"
if [ ! -x "$RUNNER" ]; then
    echo "error: expected executable not found at $RUNNER" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ -n "$RESIDUAL_HISTORY_PATH" ]; then
    mkdir -p "$(dirname "$RESIDUAL_HISTORY_PATH")"
    EXTRA_ARGS+=(--residual-history "$RESIDUAL_HISTORY_PATH")
fi
if [ "$PRECOND" -eq 1 ]; then
    EXTRA_ARGS+=(--precond)
fi
if [ "$NORMALIZE_COLUMNS" -eq 1 ]; then
    EXTRA_ARGS+=(--normalize-columns)
fi
if [ -n "$PRECOND_TYPE" ]; then
    EXTRA_ARGS+=(--precond-type "$PRECOND_TYPE")
fi
if [ "$EXPLICIT_DUPLICATES" -eq 1 ]; then
    EXTRA_ARGS+=(--explicit-duplicates)
fi
if [ "$IMPLICIT_DUPLICATES" -eq 1 ]; then
    EXTRA_ARGS+=(--implicit-duplicates)
fi
if [ -n "$MV_TYPE" ]; then
    EXTRA_ARGS+=(--mv-type "$MV_TYPE")
fi

exec "$RUNNER" \
    --matrix "$MATRIX_PATH" --x-true "$X_TRUE_PATH" --rhs "$RHS_PATH" \
    --device "$DEVICE" --value-type "$VALUE_TYPE" \
    --refine-type "$REFINE_TYPE" --tol "$TOL" --max-iters "$MAX_ITERS" \
    --max-relres "$MAX_RELRES" --restarts "$RESTARTS" "${EXTRA_ARGS[@]}"

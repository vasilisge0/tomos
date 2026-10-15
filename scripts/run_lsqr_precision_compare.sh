#!/bin/bash
# Runs scripts/run_lsqr_precision.sh across five fixed precision/
# preconditioning configurations against the SAME (A, x_true, b) problem,
# each writing its own --residual-history file, then plots all five
# relative-residual-vs-iteration curves together via
# scripts/plot_lsqr_residual_history.m. A fixed comparison matrix, not a
# general sweep tool -- built to directly compare the effect of
# preconditioning and of A's storage precision (full fp32 vs an fp32_16-
# packed, fp16-compute A) side by side on the same problem.
#
# The five configurations (--value-type/--refine-type/--precond-type all
# fp32 throughout -- only A's own storage/matvec precision and whether a
# preconditioner is used changes):
#   1. All FP32, no preconditioner.
#   2. All FP32, with the count-sketch + QR preconditioner.
#   3. --explicit-duplicates only: A stored ONCE at --mv-type (fp16)
#      precision -- PLAIN NATIVE __half, no packing -- routed to the
#      genuinely-tuned mv_fp16_native kernel (real scalar __half
#      arithmetic, registered at the (FP16,FP32,FP32,compute=FP16) slot
#      in dense_registries.cpp), plus a real, independently-cast
#      --refine-type-precision (fp32) second copy of A used for the
#      restart-boundary/final-residual recompute. --implicit-duplicates
#      and --explicit-duplicates are now HARD mutually exclusive (they
#      select different, incompatible storage schemes for A -- see
#      run_lsqr_precision.sh's own header comment) -- passing both is a
#      CLI error, not a combinable pair of options.
#   4. --implicit-duplicates only: A stored packed as NumericType::
#      FP32_16, per-iteration matvecs at the --mv-type compute tier (fp16
#      -- the SAFE-decode kernel variant, "TOMOS FP16 (implicit)" in the
#      GEMV performance plots -- the default), restart-boundary/residual
#      recompute at the fp32 tier from that SAME buffer, no second copy
#      needed. See run_lsqr_precision.sh's own header comment for the
#      full --explicit/--implicit-duplicates writeup.
#   5. Same as 4, but with TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1 set,
#      switching mv_pure_half_impl (src/dispatch/fp32_16_registries.cpp)
#      to the FAST-decode kernel sibling ("TOMOS FP16 (implicit, fast)"
#      in the GEMV plots) instead -- a comparison-only env-var toggle, NOT
#      a real lsqr_precision_runner CLI flag (added purely so this one
#      extra curve doesn't need a new Workspace field/CLI flag/dispatch-
#      registry axis threaded through the whole solver). On real
#      ill-conditioned matrices, fast decode's saturate-the-underflow-
#      band-to-zero behavior (vs. safe decode's genuine fp16-subnormal
#      reconstruction) has been observed to stall LSQR convergence
#      entirely -- this curve exists to make that visible directly next
#      to the safe-decode one, not because fast decode is expected to win.
#
#   Configurations 3, 4, and 5 all run genuine fp16 arithmetic for their
#   per-iteration matvecs (different kernels, different storage schemes,
#   but none falls back to double-accumulate compute the way plain
#   --mv-type fp16 with no duplicates flag at all still does) -- a fair
#   comparison of real fp16 storage/compute/decode strategies, not one
#   real and the rest misleading.
#
# Usage:
#   scripts/run_lsqr_precision_compare.sh --matrix A.mtx --x-true X.mtx \
#       --rhs B.mtx [--device cpu|cuda] [--mv-type T] [--value-type T]
#       [--refine-type T] [--precond-type T] [--tol TOL]
#       [--max-iters N] [--restarts N] [--output-dir DIR]
#   T in {fp64,fp32,fp16,bf16}. --mv-type defaults to fp16 (only used by
#   configurations 3 and 4 above -- 1 and 2 are always all-fp32).
#
# Writes, under DIR (default: build/output/lsqr/error), where <name> is
# --matrix's own basename with the .mtx extension stripped:
#   <name>/residual_fp32_no_precond.txt
#   <name>/residual_fp32_precond.txt
#   <name>/residual_fp16_precond_explicit.txt
#   <name>/residual_fp16_precond_implicit.txt
#   <name>/residual_fp16_precond_implicit_fast.txt
#   lsqr_residual_history_<name>.png / .fig  (via plot_lsqr_residual_history.m,
#                                             flat in DIR itself, not <name>/,
#                                             so every matrix's plot lands in
#                                             the same shared directory)
#
# Each of the four runs goes through run_lsqr_precision.sh unmodified, so
# GPU allocation (srun --overlap re-exec onto a running a100* job) and the
# lsqr_precision_runner build are both handled there, once per run -- see
# that script's own header comment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_lsqr_precision.sh"

DEVICE="cuda"
MV_TYPE="fp16"
VALUE_TYPE="fp32"
REFINE_TYPE="fp32"
PRECOND_TYPE="fp32"
TOL="1e-7"
MAX_ITERS="5"
RESTARTS="40"
MATRIX_PATH=""
X_TRUE_PATH=""
RHS_PATH=""
OUTPUT_DIR="$REPO_ROOT/build/output/lsqr/error"

usage() {
    echo "Usage: $0 --matrix A.mtx --x-true X.mtx --rhs B.mtx [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T] [--precond-type T] [--tol TOL] [--max-iters N] [--restarts N] [--output-dir DIR]" >&2
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
        --restarts) RESTARTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

[ -z "$MATRIX_PATH" ] && { echo "error: --matrix is required." >&2; usage; }
[ -z "$X_TRUE_PATH" ] && { echo "error: --x-true is required." >&2; usage; }
[ -z "$RHS_PATH" ] && { echo "error: --rhs is required." >&2; usage; }

# Derived from --matrix's own filename (e.g. tall_thin_1000_cond500.mtx ->
# tall_thin_1000_cond500) -- both the residual-history files' own
# subdirectory name and the plot's own filename suffix, so distinct
# matrices run against the same --output-dir never collide.
MATRIX_NAME="$(basename "$MATRIX_PATH")"
MATRIX_NAME="${MATRIX_NAME%.mtx}"
MATRIX_DATA_DIR="$OUTPUT_DIR/$MATRIX_NAME"

mkdir -p "$MATRIX_DATA_DIR"

COMMON_ARGS=(--matrix "$MATRIX_PATH" --x-true "$X_TRUE_PATH" --rhs "$RHS_PATH"
    --device "$DEVICE" --value-type "$VALUE_TYPE" --refine-type "$REFINE_TYPE"
    --tol "$TOL" --max-iters "$MAX_ITERS" --restarts "$RESTARTS")

HIST_FP32_NO_PRECOND="$MATRIX_DATA_DIR/residual_fp32_no_precond.txt"
HIST_FP32_PRECOND="$MATRIX_DATA_DIR/residual_fp32_precond.txt"
HIST_FP16_EXPLICIT="$MATRIX_DATA_DIR/residual_fp16_precond_explicit.txt"
HIST_FP16_IMPLICIT="$MATRIX_DATA_DIR/residual_fp16_precond_implicit.txt"
HIST_FP16_IMPLICIT_FAST="$MATRIX_DATA_DIR/residual_fp16_precond_implicit_fast.txt"

echo "==> [1/5] All FP32, no preconditioner"
bash "$RUNNER_SCRIPT" "${COMMON_ARGS[@]}" \
    --residual-history "$HIST_FP32_NO_PRECOND"

echo "==> [2/5] All FP32, with preconditioner"
bash "$RUNNER_SCRIPT" "${COMMON_ARGS[@]}" \
    --precond --precond-type "$PRECOND_TYPE" \
    --residual-history "$HIST_FP32_PRECOND"

echo "==> [3/5] mv-type $MV_TYPE (TOMOS FP16 native), preconditioner, --explicit-duplicates (real fp32 refine copy)"
bash "$RUNNER_SCRIPT" "${COMMON_ARGS[@]}" \
    --mv-type "$MV_TYPE" --precond --precond-type "$PRECOND_TYPE" \
    --explicit-duplicates \
    --residual-history "$HIST_FP16_EXPLICIT"

echo "==> [4/5] mv-type $MV_TYPE (TOMOS FP16 implicit, safe-decode), preconditioner, --implicit-duplicates only (same-buffer refine)"
bash "$RUNNER_SCRIPT" "${COMMON_ARGS[@]}" \
    --mv-type "$MV_TYPE" --precond --precond-type "$PRECOND_TYPE" \
    --implicit-duplicates \
    --residual-history "$HIST_FP16_IMPLICIT"

echo "==> [5/5] mv-type $MV_TYPE (TOMOS FP16 implicit, FAST-decode), preconditioner, --implicit-duplicates only (same-buffer refine)"
echo "    (TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1 -- comparison-only env-var toggle, see fp32_16_registries.cpp's own comment; NOT the default)"
TOMOS_FP32_16_PURE_HALF_FAST_DECODE=1 bash "$RUNNER_SCRIPT" "${COMMON_ARGS[@]}" \
    --mv-type "$MV_TYPE" --precond --precond-type "$PRECOND_TYPE" \
    --implicit-duplicates \
    --residual-history "$HIST_FP16_IMPLICIT_FAST"

echo
echo "==> Residual histories written to: $MATRIX_DATA_DIR"

echo
echo "==> Plotting residual histories"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); plot_lsqr_residual_history({'$HIST_FP32_NO_PRECOND', '$HIST_FP32_PRECOND', '$HIST_FP16_EXPLICIT', '$HIST_FP16_IMPLICIT', '$HIST_FP16_IMPLICIT_FAST'}, {'FP32 (no precond)', 'FP32 (precond)', 'TOMOS FP32 / FP16 (explicit)', 'TOMOS FP32 / FP16 (implicit)', 'TOMOS FP32 / FP16 (implicit, fast-decode)'}, '$OUTPUT_DIR', 'lsqr_residual_history_$MATRIX_NAME');"
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "$MATLAB_CALL"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"$MATLAB_CALL\"" >&2
fi

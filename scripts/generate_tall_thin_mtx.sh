#!/bin/bash
# Thin CLI wrapper around scripts/generate_tall_thin_mtx.m -- builds a
# synthetic tall-and-thin (m x n, m >> n) test matrix A with a prescribed
# singular value spectrum (A = U*diag(s)*V, U/V genuinely orthonormal via
# QR, see that .m file's own header comment for the full derivation),
# plus a random Gaussian "true" solution x_true and the right-hand side
# b = A*x_true it induces -- a complete (A, b) least-squares test problem
# with a known exact answer, not just A alone. All three are written as
# MatrixMarket .mtx files this codebase's own reader
# (cpu::read_strided_storage, src/core/memory/dense_storage.cpp) can load
# directly. Pure CPU/MATLAB work -- no GPU/SLURM allocation needed to run
# this script itself (unlike the GEVM/GEMM benchmark scripts in this same
# directory); only loading the resulting matrix onto a GPU later needs one.
#
# Usage:
#   scripts/generate_tall_thin_mtx.sh -m ROWS -n COLS -o OUTPUT_PATH \
#       -x X_OUTPUT_PATH -b B_OUTPUT_PATH
#       [-k RANK] [-l SV_LOWER] [-c COND] [-d linear|geometric]
#       [--normalize-spectrum]
#
# Required:
#   -m, --rows ROWS            Number of rows (m).
#   -n, --cols COLS            Number of columns (n).
#   -o, --output OUTPUT_PATH   Where to write A's .mtx file.
#   -x, --x-output X_PATH      Where to write x_true's .mtx file (n x 1).
#   -b, --b-output B_PATH      Where to write b's .mtx file (m x 1).
#                              Parent directories are created if missing.
#
# Optional (defaults shown):
#   -k, --rank RANK           Number of singular values / rank of A.
#                             Default: same as COLS (full column rank).
#                             Must be <= min(ROWS, COLS).
#   -l, --sv-lower SV_LOWER   Smallest singular value. Default: 1.
#   -c, --cond COND           Condition number cond(A) = sv_upper/sv_lower
#                             (must be >= 1). Default: 1e6. The largest
#                             singular value is computed as SV_LOWER*COND
#                             -- pick COND directly instead of working out
#                             sv_upper by hand.
#   -d, --dist DIST           'linear' or 'geometric' spacing of the
#                             singular values across that range.
#                             Default: geometric.
#   --normalize-spectrum      Rescale the WHOLE matrix by 1/sv_upper AFTER
#                             the spectrum-controlled construction above
#                             (equivalent to dividing every singular value
#                             by sv_upper, since U/V are orthonormal), so
#                             the largest singular value of the written A
#                             is exactly 1 -- and build b from that
#                             already-rescaled A (so the written (A, b)
#                             pair is self-consistent -- no need to redo
#                             this at solve time, unlike
#                             lsqr_precision_runner's own
#                             --normalize-columns, which does a DIFFERENT,
#                             per-column rescaling at RUN time instead).
#                             Unlike per-column normalization, this is
#                             EXACTLY condition-number-preserving (cond(A)
#                             is a ratio, unaffected by uniformly scaling
#                             every singular value) -- SV_LOWER/COND/DIST
#                             are realized exactly, not just
#                             approximately.
#
# NOTE: ||b|| is typically dominated by the largest singular direction
# for a random x_true, so it can itself run up to roughly
# SV_LOWER*COND. A solver storing internal norms in a narrow-range type
# (e.g. lsqr_solve's own --value-type fp16, max representable magnitude
# ~65504) will silently overflow to Inf/NaN the moment ||b|| exceeds
# that range -- keep COND (at SV_LOWER=1) comfortably under whatever
# range limit applies if you need a low-precision solve to actually run
# to completion, not just to observe where it breaks.
#
# Example (2000 columns, sized to fit an A100 40GB at float32 -- see
# scripts/max_rows_for_gpu.m -- geometric spectrum, condition number 1e6):
#   scripts/generate_tall_thin_mtx.sh -m 4000000 -n 2000 \
#       -o build/test_matrices/tall_thin_a100_40gb.mtx \
#       -x build/test_matrices/tall_thin_a100_40gb_x.mtx \
#       -b build/test_matrices/tall_thin_a100_40gb_b.mtx

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROWS=""
COLS=""
RANK=""
SV_LOWER="1"
COND="1e6"
DIST="geometric"
OUTPUT_PATH=""
X_OUTPUT_PATH=""
B_OUTPUT_PATH=""
NORMALIZE_SPECTRUM=0

usage() {
    echo "Usage: $0 -m ROWS -n COLS -o OUTPUT_PATH -x X_OUTPUT_PATH -b B_OUTPUT_PATH [-k RANK] [-l SV_LOWER] [-c COND] [-d linear|geometric] [--normalize-spectrum]" >&2
    exit 1
}

# Resolves a possibly-relative path to absolute without requiring the
# path (or its parent) to exist yet -- readlink -f/realpath aren't
# portable enough across the shells this might run under, so this does
# the dirname/basename split by hand.
to_abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$(pwd)" "$1" ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--rows) ROWS="$2"; shift 2 ;;
        -n|--cols) COLS="$2"; shift 2 ;;
        -k|--rank) RANK="$2"; shift 2 ;;
        -l|--sv-lower) SV_LOWER="$2"; shift 2 ;;
        -c|--cond) COND="$2"; shift 2 ;;
        -d|--dist) DIST="$2"; shift 2 ;;
        -o|--output) OUTPUT_PATH="$2"; shift 2 ;;
        -x|--x-output) X_OUTPUT_PATH="$2"; shift 2 ;;
        -b|--b-output) B_OUTPUT_PATH="$2"; shift 2 ;;
        --normalize-spectrum) NORMALIZE_SPECTRUM=1; shift 1 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

[ -z "$ROWS" ] && { echo "error: -m/--rows is required." >&2; usage; }
[ -z "$COLS" ] && { echo "error: -n/--cols is required." >&2; usage; }
[ -z "$OUTPUT_PATH" ] && { echo "error: -o/--output is required." >&2; usage; }
[ -z "$X_OUTPUT_PATH" ] && { echo "error: -x/--x-output is required." >&2; usage; }
[ -z "$B_OUTPUT_PATH" ] && { echo "error: -b/--b-output is required." >&2; usage; }
[ -z "$RANK" ] && RANK="$COLS"

case "$DIST" in
    linear|geometric) ;;
    *) echo "error: -d/--dist must be 'linear' or 'geometric' (got '$DIST')." >&2; exit 1 ;;
esac

for name_val in "ROWS:$ROWS" "COLS:$COLS" "RANK:$RANK" "SV_LOWER:$SV_LOWER" "COND:$COND"; do
    name="${name_val%%:*}"
    val="${name_val#*:}"
    if ! [[ "$val" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        echo "error: $name must be numeric (got '$val')." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$X_OUTPUT_PATH")" "$(dirname "$B_OUTPUT_PATH")"
OUTPUT_ABS="$(to_abs_path "$OUTPUT_PATH")"
X_OUTPUT_ABS="$(to_abs_path "$X_OUTPUT_PATH")"
B_OUTPUT_ABS="$(to_abs_path "$B_OUTPUT_PATH")"

NORMALIZE_SPECTRUM_MATLAB="false"
[ "$NORMALIZE_SPECTRUM" -eq 1 ] && NORMALIZE_SPECTRUM_MATLAB="true"

if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
MATLAB_CALL="addpath('$REPO_ROOT/scripts'); generate_tall_thin_mtx($ROWS, $COLS, $RANK, $SV_LOWER, $COND, '$DIST', '$OUTPUT_ABS', '$X_OUTPUT_ABS', '$B_OUTPUT_ABS', $NORMALIZE_SPECTRUM_MATLAB);"
if ! command -v matlab >/dev/null 2>&1; then
    echo "error: matlab not found (tried 'module load matlab/R2024a')." >&2
    echo "       run manually later, e.g.:" >&2
    echo "         module load matlab/R2024a" >&2
    echo "         matlab -batch \"$MATLAB_CALL\"" >&2
    exit 1
fi

echo "==> Generating ${ROWS} x ${COLS} (rank ${RANK}, ${DIST} spectrum, sv_lower=${SV_LOWER}, cond=${COND}, normalize_spectrum=${NORMALIZE_SPECTRUM_MATLAB})"
echo "==> Writing A to: $OUTPUT_ABS"
echo "==> Writing x_true to: $X_OUTPUT_ABS"
echo "==> Writing b to: $B_OUTPUT_ABS"
matlab -batch "$MATLAB_CALL"

echo "==> Done."

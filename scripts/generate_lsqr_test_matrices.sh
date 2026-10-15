#!/bin/bash
# Generates the full set of tall-and-thin LSQR test problems that
# scripts/run_lsqr_precision_performance_compare.sh and
# scripts/run_lsqr_precision_speedup_by_columns.sh expect under their
# --matrix-dir, by calling scripts/generate_tall_thin_mtx.sh (MATLAB,
# see that script and scripts/generate_tall_thin_mtx.m) once per
# (columns, variant) pair. This replaces depending on a pre-existing,
# machine-specific matrix directory -- every file it writes is
# reproducible from scratch, from this one command.
#
# Variants (same naming/condition-number convention documented in
# run_lsqr_precision_speedup_by_columns.sh's own header):
#   cond1000      cond(A) = 1000                    tall_thin_<n>.mtx
#   cond500       cond(A) = 500                      tall_thin_<n>_cond500.mtx
#   cond1000norm  cond(A) = 1000, spectrum-normalized tall_thin_<n>_cond1000_norm.mtx
# Each variant also writes its _xtrue.mtx / _b.mtx companions.
#
# RESUMABLE: a (columns, variant) triple already present under
# --output-dir is skipped, so re-running after an interrupted/partial
# run only generates what's missing. Pass --fresh to regenerate
# everything regardless.
#
# PERSISTENT LOCAL OVERRIDE: if a (gitignored, not committed)
# .configure.sh exists at the repo root, every script here and in
# run_lsqr_precision_performance_compare.sh / speedup_by_columns.sh
# sources it before applying its own default -- set MATRIX_DIR=/your/path
# there once instead of passing --output-dir/--matrix-dir on every
# invocation. See .configure.sh.example for the template.
#
# COST: these are the same problem sizes documented in
# run_lsqr_precision_speedup_by_columns.sh (1,000,000 rows; ~23.5/47/70.5
# GB of ASCII .mtx per 1000/2000/3000 columns) -- generation needs
# proportional disk space and MATLAB runtime, not GPU/SLURM.
#
# Usage:
#   scripts/generate_lsqr_test_matrices.sh
#       [--rows M] [--columns "1000 2000 3000"]
#       [--variants "cond1000 cond500"] [--output-dir DIR]
#       [--sv-lower SV_LOWER] [--dist linear|geometric] [--fresh]
#   --rows         Rows per problem (default 1000000).
#   --columns      Space-separated column counts to generate.
#   --variants     Space-separated subset of: cond1000, cond500,
#                  cond1000norm (default "cond1000 cond500").
#   --output-dir   Where to write the .mtx files
#                  (default: $MATRIX_DIR if set -- see .configure.sh
#                  below -- else build/matrices).
#   --sv-lower     Smallest singular value, passed through to
#                  generate_tall_thin_mtx.sh (default 1).
#   --dist         Singular value spacing, passed through
#                  (default geometric).
#   --fresh        Regenerate every file even if already present.
#
# Example (just the two problems run_lsqr_precision_performance_compare.sh
# needs, default rows/output-dir):
#   scripts/generate_lsqr_test_matrices.sh --columns 1000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
GENERATOR="$REPO_ROOT/scripts/generate_tall_thin_mtx.sh"

ROWS="1000000"
COLUMNS_LIST="1000 2000 3000"
VARIANTS="cond1000 cond500"
OUTPUT_DIR="${MATRIX_DIR:-$REPO_ROOT/build/matrices}"
SV_LOWER="1"
DIST="geometric"
FRESH=0

usage() {
    echo "Usage: $0 [--rows M] [--columns \"1000 2000 3000\"] [--variants \"cond1000 cond500\"] [--output-dir DIR] [--sv-lower SV_LOWER] [--dist linear|geometric] [--fresh]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --rows) ROWS="$2"; shift 2 ;;
        --columns) COLUMNS_LIST="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sv-lower) SV_LOWER="$2"; shift 2 ;;
        --dist) DIST="$2"; shift 2 ;;
        --fresh) FRESH=1; shift 1 ;;
        -h|--help) usage ;;
        *) echo "error: unrecognized argument '$1'" >&2; usage ;;
    esac
done

[ -x "$GENERATOR" ] || { echo "error: $GENERATOR not found or not executable." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

# variant -> "COND:NORMALIZE_FLAG:SUFFIX"
variant_spec() {
    case "$1" in
        cond1000)     echo "1000:0:" ;;
        cond500)      echo "500:0:_cond500" ;;
        cond1000norm) echo "1000:1:_cond1000_norm" ;;
        *) echo "error: unknown variant '$1' (expected cond1000, cond500, or cond1000norm)." >&2; exit 1 ;;
    esac
}

GENERATED=0
SKIPPED=0

for n in $COLUMNS_LIST; do
    for variant in $VARIANTS; do
        spec="$(variant_spec "$variant")"
        cond="${spec%%:*}"
        rest="${spec#*:}"
        normalize_flag="${rest%%:*}"
        suffix="${rest#*:}"

        base="$OUTPUT_DIR/tall_thin_${n}${suffix}"
        a_path="${base}.mtx"
        x_path="${base}_xtrue.mtx"
        b_path="${base}_b.mtx"

        if [ "$FRESH" -ne 1 ] && [ -f "$a_path" ] && [ -f "$x_path" ] && [ -f "$b_path" ]; then
            echo "==> skip: $variant, columns=$n (already present at $a_path)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo
        echo "================================================================"
        echo "==> Generating: $variant, rows=$ROWS, columns=$n, cond=$cond, normalize-spectrum=$([ "$normalize_flag" -eq 1 ] && echo yes || echo no)"
        echo "================================================================"

        args=(-m "$ROWS" -n "$n" -c "$cond" -d "$DIST" -l "$SV_LOWER" \
              -o "$a_path" -x "$x_path" -b "$b_path")
        [ "$normalize_flag" -eq 1 ] && args+=(--normalize-spectrum)

        "$GENERATOR" "${args[@]}"
        GENERATED=$((GENERATED + 1))
    done
done

echo
echo "================================================================"
echo "==> Done: $GENERATED generated, $SKIPPED already present, under $OUTPUT_DIR"
echo "================================================================"

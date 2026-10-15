#!/bin/bash
# "_advanced" variant of scripts/benchmarks_gemv_transpose_fp32_recon_vs_
# cublas_ref.sh -- builds benchmarks/kernels/cuda/matrix/mv_fp32_
# transpose_recon_vs_cublas_ref_advanced.cu (byte-for-byte identical to
# its own sibling .cu except for a caller-controlled warmup/iteration
# count instead of a fixed, shape-size-dependent `reps` heuristic -- see
# that file's own header) instead of the plain mv_fp32_transpose_recon_
# vs_cublas_ref.cu, and runs the EXACT SAME two sweeps (square + tall-
# and-thin, including the SAME --square-only/--tall-only forms) that
# script does -- see that script's own header comment for the full
# shape-list/kernel-history reasoning, not repeated here.
#
# Usage:
#   scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh
#       Runs both sweeps with 10 warmup runs and 10 timed-and-averaged
#       iterations per shape (the defaults).
#   scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh --warmup 20 --iters 50
#       Same sweeps, 20 warmup runs and 50 timed-and-averaged iterations
#       per shape instead.
#   scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh --square-only [M1 M2 ...] [--warmup N] [--iters N]
#       Square sweep only -- see the sibling script's own header for the
#       exact form; --warmup/--iters may appear anywhere among the args.
#   scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_advanced.sh --tall-only [K M1 M2 ...] [--warmup N] [--iters N]
#       Tall-and-thin sweep only -- same form as the sibling script.
#
# All shapes' full console output is saved to one timestamped .txt log
# under build/mv_fp32_transpose_recon_vs_cublas_ref_advanced_output/ (a
# SEPARATE directory from the sibling script's own output), parsed into
# one .csv (same schema as the sibling script's own CSV -- the .cu's own
# printed method lines are unaffected by the timing-methodology change).
# Plots are written with an "_avg" tag (see plot_gemv_transpose_fp32_
# recon_vs_cublas_ref.m's own `tag` parameter), so they land ALONGSIDE
# the sibling script's own plots without overwriting them:
#   build/benchmarks/output/plots/plot_gemv_transpose_fp32_recon_vs_cublas_ref_square_avg.png / .fig
#   build/benchmarks/output/plots/plot_gemv_transpose_fp32_recon_vs_cublas_ref_tall_avg.png   / .fig

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_transpose_recon_vs_cublas_ref_advanced.cu"
OUT="$BUILD_DIR/mv_fp32_transpose_recon_vs_cublas_ref_advanced"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_transpose_recon_vs_cublas_ref_advanced_output"

# --warmup N / --iters N (both optional, default 10/10) -- extracted
# from anywhere in "$@" before this script's own --square-only/--tall-
# only parsing below runs, so both forms compose freely.
WARMUP=10
ITERS=10
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@" --warmup "$WARMUP" --iters "$ITERS"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_transpose_recon_vs_cublas_ref_advanced"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$REPO_ROOT/src/kernels/cuda/matrix" \
    -I"$REPO_ROOT/src/kernels/cpu/matrix" \
    -I"$REPO_ROOT/src/core/matrix" \
    -I"$REPO_ROOT/include/core" \
    -I"$REPO_ROOT/src/core/context" \
    -I"$REPO_ROOT/src/core/memory" \
    -I"$REPO_ROOT/src/core/blas" \
    -I"$REPO_ROOT/src/kernels/cuda/context" \
    -I"$REPO_ROOT/src/macros" \
    -I"$REPO_ROOT/include/io" \
    -I"$REPO_ROOT/include/cpu" \
    -I"$REPO_ROOT/src/kernels/cpu/memory" \
    -I"$REPO_ROOT/src/kernels/cpu/sketch" \
    -I"$REPO_ROOT/include/cuda" \
    -I"$REPO_ROOT/src/kernels/cuda/constants" \
    -I"$REPO_ROOT/src/kernels/cuda/memory" \
    -I"$REPO_ROOT/src/kernels/cuda/sketch" \
    -I"$REPO_ROOT/src/kernels/cuda/precond" \
    -I"$REPO_ROOT/src/kernels/cuda/blas" \
    -I"$REPO_ROOT/src/core/solver" \
    -I"$REPO_ROOT/src/dispatch" \
    -I"$REPO_ROOT/src/io" \
    -I"$BUILD_DIR/generated" \
    -I"$BUILD_DIR" \
    -I"$REPO_ROOT/include" \
    -I"$REPO_ROOT/src/core/detail" \
    "$SRC" -lcublas \
    -L"$TOMOS_CUDA_DIR" -ltomos_cuda -Xlinker -rpath -Xlinker "$TOMOS_CUDA_DIR" \
    -o "$OUT"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_transpose_recon_vs_cublas_ref_advanced_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_transpose_recon_vs_cublas_ref_advanced_${TIMESTAMP}.csv"

# Same shape lists as scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh's own.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)
TALL_KS=(512 4096)
TALL_MS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# --square-only / --tall-only (see this script's own header comment) --
# both default to false, meaning "run both sweeps."
RUN_SQUARE=1
RUN_TALL=1
if [ "${1:-}" = "--square-only" ]; then
    RUN_TALL=0
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_MS=("$@")
    fi
elif [ "${1:-}" = "--tall-only" ]; then
    RUN_SQUARE=0
    shift
    if [ "$#" -gt 0 ]; then
        TALL_KS=("$1")
        shift
        TALL_MS=("$@")
    fi
fi

: > "$TXT_LOG"
if [ "$RUN_SQUARE" -eq 1 ]; then
    for m in "${SQUARE_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_transpose_recon_vs_cublas_ref_advanced $m (warmup=$WARMUP iters=$ITERS)"
        "$OUT" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
    done
fi
if [ "$RUN_TALL" -eq 1 ]; then
    for k in "${TALL_KS[@]}"; do
        for m in "${TALL_MS[@]}"; do
            echo
            echo "==> Running mv_fp32_transpose_recon_vs_cublas_ref_advanced --tall $k $m (warmup=$WARMUP iters=$ITERS)"
            "$OUT" --tall "$k" "$m" --warmup "$WARMUP" --iters "$ITERS" | tee -a "$TXT_LOG"
        done
    done
fi

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

# Marker -> CSV field prefix. Order matters only in that no marker here is
# a substring of another (verified via a Python collision-matrix check
# before this list was written, same discipline as every other *_vs_
# cutlass_ref.sh/*_vs_cublas_ref.sh script's own METHOD_MARKERS) -- so the
# first-match-wins `break` below never actually has to arbitrate a real
# ambiguity.
METHOD_MARKERS = [
    # "TOMOS FP32 (unvectorized)" -- mv_fp32_tiled's own NEW TRANSPOSE fast
    # path (warp-per-output-column/split-M/atomicAdd, ported from
    # randls2.0 -- see mv_fp32_transpose_recon_vs_cublas_ref.cu's own
    # header comment). Was plain "native"/"TOMOS FP32"; DEMOTED now that
    # mv_fp32_transpose_atomiconly_scaled_mblock_vec4 (below) sources that
    # slot instead -- kept as its own visible series, not dropped, same
    # "keep both, promote the winner" convention every other promotion in
    # this file already follows.
    ("mv_fp32_tiled", "native_unvectorized"),
    # "TOMOS FP32 (vec4)" -- mv_fp32_transpose_atomiconly_scaled_mblock_
    # vec4's own TRANSPOSE kernel: byte-for-byte the same warp-per-
    # output-column/split-M/atomicAdd structure as the unvectorized
    # sibling above, but float4-vectorized loads of A and the shared x
    # tile (runtime 16-byte alignment check before committing to the
    # vectorized path -- see its own comment in mv_cuda_kernels.cu). Was
    # plain "native"/"TOMOS FP32"; DEMOTED now that mv_fp32_transpose_
    # atomiconly_scaled_mblock_vec4_stagevec (below) sources that slot
    # instead -- kept as its own visible series, same "keep both, promote
    # the winner" convention. Trailing " =" (one space) keeps this
    # disjoint from the "mv_fp32_tiled" marker above (neither is a
    # substring of the other) and from the "_stagevec" marker below
    # (diverges right after "vec4": " =" here vs "_stagevec =" there).
    ("mv_fp32_transpose_atomiconly_scaled_mblock_vec4 =", "native_vec4"),
    # "TOMOS FP32" (native) -- mv_fp32_transpose_atomiconly_scaled_mblock_
    # vec4_stagevec's own NEW TRANSPOSE kernel: same structure as the
    # vec4 sibling above, but ALSO float4-vectorizes the x-staging loop
    # (global x -> shared sx), not just the main accumulation loop's own
    # A/sx reads -- see its own comment in mv_cuda_kernels.cu. PROMOTED
    # here: a clean, consistent win over vec4 on real A100 hardware --
    # beats it at nearly every size tested (one small dip at M=1536,
    # -2.9%), with the biggest gains, +8%-+12%, exactly where vec4 itself
    # was weakest (M=2048/4096/8192), reaching parity with or beating
    # cuBLAS from M=12288 upward (0.999x-1.081x, vs vec4's own 0.95x-
    # 1.05x there).
    ("mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec =", "native"),
    ("cublas (Sgemv)", "cublas"),
    # "TOMOS FP32/FP16 (scalar)" -- mv_fp32_16_accurate_ready4_atomiconly_
    # scaled_kblock's own TRANSPOSE fast path. Safe on arbitrary data. Was
    # plain "fp32_16"/"TOMOS FP32/FP16"; DEMOTED now that mv_fp32_16_
    # transpose_atomiconly_scaled_mblock_vec4 (below) sources that slot
    # instead -- kept as its own visible series, same "keep both, promote
    # the winner" convention. Trailing " =" (one space) keeps this
    # disjoint from the "_fast_normal" marker below in either direction
    # (they diverge right after "accurate_ready": "4_atomiconly..." here
    # vs "_atomiconly...fast_normal" there).
    ("mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock =", "fp32_16_scalar"),
    # "TOMOS FP32/FP16 (vec4)" -- mv_fp32_16_transpose_atomiconly_scaled_
    # mblock_vec4's own TRANSPOSE kernel: same structure as "(scalar)"
    # above, but float4-vectorized x-staging/sx reads AND ushort4-
    # vectorized wh/wr loads (decode itself, fp32_16_combine_branchless,
    # stays scalar). Was plain "fp32_16"/"TOMOS FP32/FP16"; DEMOTED now
    # that mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_
    # floor (below) sources that slot instead -- kept as its own visible
    # series, same "keep both, promote the winner" convention. Trailing
    # " =" (one space) keeps this disjoint from the "_retuned"/"_retuned_
    # floor" markers below (diverge right after "vec4": " =" here vs
    # "_retuned..." there).
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4 =", "fp32_16_vec4"),
    # "TOMOS FP32/FP16 (retuned_floor)" -- mv_fp32_16_transpose_
    # atomiconly_scaled_mblock_vec4_retuned_floor's own TRANSPOSE kernel:
    # reuses the EXACT SAME __global__ kernel as "(vec4)" above unchanged,
    # just a more aggressive M_BLOCK-scaling grid-oversubscription target
    # (2048 vs 512) with a floor (256) capping how far M_BLOCK can shrink,
    # to avoid the excessive atomicAdd contention an unfloored version of
    # this same idea measured at small M -- see its own comment in
    # fp32_16_cuda_kernels.cu. Was plain "fp32_16"/"TOMOS FP32/FP16";
    # DEMOTED now that mv_fp32_16_transpose_atomiconly_scaled_mblock_
    # vec4_wrapborrow (below) sources that slot instead -- kept as its
    # own visible series, same "keep both, promote the winner" convention.
    # Its own promotion: a CLEAN win over "(vec4)" at every size tested on
    # real A100 hardware, never worse by more than measurement noise
    # (-0.3% at M=1024) -- +5.5%-+14.2% at M=3072-12288, ~parity
    # elsewhere.
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor =", "fp32_16_retuned_floor"),
    # "TOMOS FP32/FP16 (wrapborrow)" -- mv_fp32_16_transpose_atomiconly_
    # scaled_mblock_vec4_wrapborrow's own TRANSPOSE kernel: reuses the
    # EXACT SAME __global__ kernel STRUCTURE and M_BLOCK-scaling as
    # "(retuned_floor)" above unchanged, only decoding via fp32_16_
    # combine_branchless_wrapborrow instead of fp32_16_combine_branchless
    # -- the SAME wraparound-subtraction technique the UNSAFE fast_normal_
    # wrapborrow decode already proved out, applied here to the SAFE
    # decode's own identical is_carry/round_flag/mant_hi pattern -- see
    # its own comment in fp32_16_cuda_kernels.cu. Confirmed via cuobjdump:
    # eliminates all 26 SEL instructions and reduces ISETP 46->33
    # (864->832 total instructions), though register usage goes the WRONG
    # way (REG:32 vs REG:31, one MORE, unlike fast_normal_wrapborrow's own
    # reduction). Was plain "fp32_16"/"TOMOS FP32/FP16"; DEMOTED now that
    # mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_
    # dispatch (below) sources that slot instead -- kept as its own
    # visible series, same "keep both, promote the winner" convention.
    # Its own promotion: on real A100 hardware the fewer-instructions side
    # won decisively regardless of the extra register -- a remarkably
    # clean, MONOTONIC win at every square size tested (M=1024-65536),
    # from +4.0% at the smallest to +12.1% at the largest, no regressions.
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow =", "fp32_16_wrapborrow"),
    # "TOMOS FP32/FP16" -- mv_fp32_16_transpose_atomiconly_scaled_mblock_
    # vec4_wrapborrow_dispatch's own NEW TRANSPOSE entry point: NOT a new
    # tuned kernel, a thin host-side router over two already-independently-
    # tested siblings -- calls "(wrapborrow)" above directly for M<8192,
    # and mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_
    # retarget2 (target=8192, same shared floor=256, not its own visible
    # CSV/plot series) for M>=8192 -- a direct, high-confidence port of
    # the UNSAFE fast_normal_wrapborrow family's own _dispatch, since the
    # M_BLOCK-scaling grid math is decode-independent -- see both
    # kernels' own comments in fp32_16_cuda_kernels.cu. PROMOTED: on real
    # A100 hardware, tracks "(wrapborrow)" almost exactly below M=8192
    # (0.996x-1.006x, negligible routing overhead) and _retarget2 almost
    # exactly at/above it (1.006x-1.077x), capturing _retarget2's own
    # real M=8192-12288 win (+7.8% at M=12288) while completely avoiding
    # its severe M=3072-6144 regression (down to -20.4% for _retarget2
    # alone, 0.999x-1.001x for this dispatcher at those same sizes) -- no
    # regression found anywhere tested (M=1024-12288 at the time of this
    # promotion). retarget2 itself is deliberately NOT given its own CSV
    # column here -- it exists purely as this dispatcher's own internal
    # building block, same convention as the UNSAFE family's own
    # retarget2. CAVEAT: not yet confirmed at M>=16384, and (mirroring the
    # UNSAFE family's own dispatcher) this M-only threshold is unvalidated
    # for tall-and-thin/large-K shapes.
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch =", "fp32_16"),
    # "TOMOS FP32/FP16 (fast, scalar)" -- mv_fp32_16_accurate_ready_
    # atomiconly_scaled_kblock_fast_normal's own TRANSPOSE fast path.
    # UNSAFE on arbitrary data (tested against dedicated safe-range data
    # in the .cu). Was plain "fp32_16_fast"/"TOMOS FP32/FP16 (fast)";
    # DEMOTED now that mv_fp32_16_transpose_atomiconly_scaled_mblock_
    # fast_normal_vec4 (below) sources that slot instead -- kept as its
    # own visible series, same "keep both, promote the winner"
    # convention.
    ("mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal =", "fp32_16_fast_scalar"),
    # "TOMOS FP32/FP16 (fast, vec4)" -- mv_fp32_16_transpose_atomiconly_
    # scaled_mblock_fast_normal_vec4's own TRANSPOSE kernel: same
    # structure as "(fast, scalar)" above, but float4-vectorized x-
    # staging/sx reads AND ushort4-vectorized wh/wr loads (decode itself,
    # fp32_16_combine_fast_normal, stays scalar), using the shared
    # default grid target (512, not the SAFE decode's own higher retuned
    # target) -- see its own comment in fp32_16_cuda_kernels.cu. Was
    # plain "fp32_16_fast"/"TOMOS FP32/FP16 (fast)"; DEMOTED now that
    # mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_
    # retuned (below) sources that slot instead -- kept as its own
    # visible series, same "keep both, promote the winner" convention.
    # Trailing " =" (one space) keeps this disjoint from the "_retuned"
    # marker below (diverges right after "vec4": " =" here vs "_retuned
    # =" there).
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4 =", "fp32_16_fast_vec4"),
    # "TOMOS FP32/FP16 (fast, retuned)" -- mv_fp32_16_transpose_
    # atomiconly_scaled_mblock_fast_normal_vec4_retuned's own TRANSPOSE
    # kernel: reuses the EXACT SAME __global__ kernel as "(fast, vec4)"
    # above unchanged, just a more conservative M_BLOCK-scaling grid-
    # oversubscription target (1024, not the SAFE decode's own 2048 --
    # re-derived from this kernel's own higher register footprint,
    # REG:39 vs REG:31, rather than reused blindly) with the same shared
    # floor (256) -- see its own comment in fp32_16_cuda_kernels.cu. Was
    # plain "fp32_16_fast"/"TOMOS FP32/FP16 (fast)"; DEMOTED now that
    # mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
    # wrapborrow_vec4 (below) sources that slot instead -- kept as its
    # own visible series, same "keep both, promote the winner"
    # convention. Trailing " =" (one space) keeps this disjoint from the
    # "wrapborrow_vec4" marker below (diverges right after "fast_normal_":
    # "vec4_retuned =" here vs "wrapborrow_vec4 =" there).
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned =", "fp32_16_fast_vec4_retuned"),
    # "TOMOS FP32/FP16 (fast, wrapborrow)" -- mv_fp32_16_transpose_
    # atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4's own TRANSPOSE
    # kernel: byte-for-byte the SAME kernel/launch-config as "(fast,
    # retuned)" above, only decoding via fp32_16_combine_fast_normal_
    # wrapborrow instead of fp32_16_combine_fast_normal (bit-identical,
    # exhaustively verified over all 1024 possible (mant_hi, round_flag)
    # combinations -- see its own comment in fp32_16_pack.hpp). Motivated
    # by a direct SASS instruction-count profile showing decode overhead,
    # not FMA math, dominated the kernel's static instruction count --
    # confirmed via cuobjdump: eliminates all 38 SEL/19 ISETP.EQ.U32.AND
    # instructions the is_carry branch compiled to, drops register usage
    # REG:39 -> REG:32 (8 blocks/SM instead of 6, full-residency capacity
    # 864 instead of 648). Was plain "fp32_16_fast"/"TOMOS FP32/FP16
    # (fast)"; DEMOTED now that mv_fp32_16_transpose_atomiconly_scaled_
    # mblock_fast_normal_wrapborrow_vec4_retarget (below) sources that
    # slot instead -- kept as its own visible series, same "keep both,
    # promote the winner" convention. Trailing " =" keeps this disjoint
    # from the "_retarget" marker below (diverges right after "vec4":
    # " =" here vs "_retarget =" there).
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4 =", "fp32_16_fast_wrapborrow_vec4"),
    # "TOMOS FP32/FP16 (fast, retarget)" -- mv_fp32_16_transpose_
    # atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget's own
    # TRANSPOSE kernel: reuses the EXACT SAME __global__ kernel as "(fast,
    # wrapborrow)" above unchanged, just a more aggressive M_BLOCK-
    # scaling grid-oversubscription target (2048, re-derived from
    # wrapborrow's own measured REG:32 -- matching the SAFE decode
    # kernel's own REG:31/2048-target residency profile, not blindly
    # copied from it) with the same shared floor (256) -- see its own
    # comment in fp32_16_cuda_kernels.cu. Was plain "fp32_16_fast"/"TOMOS
    # FP32/FP16 (fast)"; DEMOTED now that mv_fp32_16_transpose_
    # atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch
    # (below) sources that slot instead -- kept as its own visible
    # series, same "keep both, promote the winner" convention. Its own
    # promotion history: a clean win over "(fast, wrapborrow)" at every
    # size where the grid config actually differs between the two
    # targets (verified by recomputing M_BLOCK for both, not just
    # trusting the measured ratio): +7.5% to +14.3% across M=3072-8192
    # square, and +15.2% at the one tall-and-thin shape where target=2048
    # actually shrinks M_BLOCK below the kMvTransposeMBlock=8192 ceiling
    # (K=4096, M=16384; target_gridM=4 there gives scaled_block=4096,
    # under the ceiling). At M=1024-1536, M_BLOCK is already floor-bound
    # (256) under BOTH targets, so the measured ~1.00x there is the
    # CORRECT, expected outcome, not noise to explain away. CAVEAT
    # verified the same way: at the two LARGEST tall-and-thin shapes
    # tested (K=4096, M=32768 and M=65536), target_gridM's own
    # scaled_block (16384 either way) never drops below the 8192 ceiling
    # under EITHER target -- M_BLOCK, grid, and the resulting kernel
    # launch are BYTE-FOR-BYTE IDENTICAL between "(fast, wrapborrow)" and
    # this kernel at those two shapes, so the +15.8%/+5.6% originally
    # measured there are almost certainly measurement noise, not a real
    # effect.
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget =", "fp32_16_fast_retarget"),
    # "TOMOS FP32/FP16 (fast)" -- mv_fp32_16_transpose_atomiconly_scaled_
    # mblock_fast_normal_wrapborrow_vec4_dispatch's own NEW TRANSPOSE
    # entry point: NOT a new tuned __global__ kernel, a thin host-side
    # router over two already-independently-validated siblings -- calls
    # "(fast, retarget)" above directly for M<8192, and mv_fp32_16_
    # transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_
    # retarget2 (target=8192, same floor=256, not its own visible CSV
    # series) for M>=8192 -- see both kernels' own comments in
    # fp32_16_cuda_kernels.cu. PROMOTED: validated on real A100 hardware
    # across the FULL square sweep (M=1024-65536), not just the two
    # shapes checked when the threshold was first chosen -- tracks
    # "(fast, retarget)" almost exactly below the M=8192 threshold
    # (0.998x-1.003x) and its retarget2 sibling almost exactly at and
    # above it (matches to 3 decimal places at M=24576/32768/49152/65536:
    # e.g. 1.820ms vs 1.819ms at M=24576), with dispatch overhead itself
    # never exceeding ~0.3% anywhere. Net effect: at least as good as
    # "(fast, retarget)" everywhere, and a real win from M=8192 up
    # (+2.8% to +10.7% vs "(fast, retarget)" across that range on this
    # run) by riding retarget2's own genuinely smaller M_BLOCK there.
    # CAVEAT: the M=8192 threshold itself was only validated on SQUARE
    # shapes -- the tall-and-thin sweep this file also runs only covers
    # M>=16384 (already past the threshold either way), so a small-M/
    # large-K combination (where gridK could shift where retarget2 starts
    # winning) remains untested. retarget2 itself is deliberately NOT
    # given its own CSV column/series here -- it exists purely as
    # dispatch's own internal building block, same as retarget2_floor (a
    # documented negative result, see its own comment in fp32_16_cuda_
    # kernels.cu -- fixes retarget2's M=3072-6144 regression but
    # introduces a worse one at M=1024-1536 from its floor exceeding m
    # itself, so it is not, and will not be, folded into this dispatcher).
    ("mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch =", "fp32_16_fast"),
    # "TOMOS FP32/BF16 (truncated, workspace)" -- mv_fp32_bf16_accurate_
    # ready_atomiconly_scaled_kblock_workspace's own TRANSPOSE fast path,
    # decoding via plain fp32_bf16_combine against plain-truncating-split
    # data, scalar-only loads, the ORIGINAL untuned M_BLOCK-scaling target
    # (512, floor=1). Was plain "fp32_bf16_truncated"/"TOMOS FP32/BF16
    # (truncated)"; DEMOTED now that mv_fp32_bf16_transpose_atomiconly_
    # scaled_mblock_vec4 (below) sources that slot instead -- kept as its
    # own visible series, same "keep both, promote the winner" convention.
    # Trailing " =" keeps this disjoint from the "_rounded" marker below
    # (diverges right after "kblock_": "workspace" vs "rounded").
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace =", "fp32_bf16_truncated_workspace"),
    # "TOMOS FP32/BF16 (truncated)" -- mv_fp32_bf16_transpose_atomiconly_
    # scaled_mblock_vec4's own NEW TRANSPOSE kernel: byte-for-byte the
    # SAME warp-per-output-column/split-M/atomicAdd/M_BLOCK-scaling
    # structure as SAFE fp32_16's own "(vec4)" TRANSPOSE kernel, ported
    # directly per this codebase's "base the design on FP32/FP16 (fast)"
    # convention -- float4-vectorized x-staging AND ushort4-vectorized
    # wh/wr loads, decode via fp32_bf16_combine unchanged (a single
    # shift+OR, no branch to reformulate the way fp16's decode had --
    # see its own comment in fp32_16_cuda_kernels.cu). SAME M_BLOCK-
    # scaling (target=512, floor=1) as "(truncated, workspace)" above, so
    # this isolates ONLY the vectorization change. PROMOTED: on real
    # A100 hardware, a dramatic, MONOTONICALLY GROWING win over "(workspace)"
    # at every square size tested (M=1024-6144) -- +0.6% at the smallest
    # up to +34.6% at M=6144, no regressions, already beating cuBLAS
    # outright from M=4096 up (1.000x-1.190x depending on retuning, see
    # below). Consistent with bf16's decode being essentially free (no
    # compute to hide memory latency behind, unlike fp16), making this
    # kernel MORE bandwidth-sensitive, not less, than its fp32_16
    # counterpart -- wider loads matter more here. A further M_BLOCK
    # retuning sibling, mv_fp32_bf16_transpose_atomiconly_scaled_mblock_
    # vec4_retuned_floor (reusing fp32_16's own target=2048/floor=256),
    # measured a REAL further win at M=3072-8192 (+2.2%-+5.3%) but a REAL
    # REGRESSION at M=2048 specifically in two independent runs
    # (-7.6%/-9.4% vs plain vec4) -- CROSS-VALIDATED as likely NOT noise:
    # its rounded-encoding counterpart (see fp32_bf16 below) shows the
    # SAME dip at the SAME M=2048 with a completely different decode
    # function. Grid-math root-cause candidate: at M=K=2048, gridK,
    # M_BLOCK, and gridM all land on clean powers of 2 simultaneously, a
    # triple alignment no neighboring size shares -- kept as a documented
    # open question, not promoted.
    ("mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4 =", "fp32_bf16_truncated"),
    # "TOMOS FP32/BF16 (truncated, test)" -- mv_fp32_bf16_transpose_
    # atomiconly_scaled_mblock_vec4_dispatch's own NEW TRANSPOSE entry
    # point: NOT a new tuned __global__ kernel, a thin host-side router
    # over "(truncated)" above (M<8192) and its own retarget2 sibling
    # (target=8192, same shared floor=256, not its own visible CSV/plot
    # series) for M>=8192 -- a direct, high-confidence port of fp32_16's
    # own dispatch pattern, since the M_BLOCK-scaling grid math is
    # decode-independent -- see both kernels' own comments in
    # fp32_16_cuda_kernels.cu. UNTESTED / NOT YET PROMOTED: plotted here
    # as a "test" series purely to compare against "(truncated)" before
    # any promotion decision -- see mv_fp32_bf16_transpose_atomiconly_
    # scaled_mblock_vec4_dispatch's own comment for the current state of
    # that evaluation.
    ("mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch =", "fp32_bf16_truncated_test"),
    # "TOMOS FP32/BF16 (rounded, workspace)" -- mv_fp32_bf16_accurate_
    # ready_atomiconly_scaled_kblock_rounded's own TRANSPOSE fast path,
    # decoding via fp32_bf16_combine_rounded against rounded-split data,
    # scalar-only loads, the ORIGINAL untuned M_BLOCK-scaling target (512,
    # floor=1). Was plain "fp32_bf16"/"TOMOS FP32/BF16"; DEMOTED now that
    # mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded (below)
    # sources that slot instead -- kept as its own visible series, same
    # "keep both, promote the winner" convention.
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded =", "fp32_bf16_rounded_workspace"),
    # "TOMOS FP32/BF16 (rounded, vec4)" -- mv_fp32_bf16_transpose_
    # atomiconly_scaled_mblock_vec4_rounded's own TRANSPOSE kernel:
    # rounded-encoding counterpart of mv_fp32_bf16_transpose_atomiconly_
    # scaled_mblock_vec4 above (byte-for-byte identical except decoding
    # via fp32_bf16_combine_rounded), same "base the design on FP32/FP16
    # (fast)" port. Same M_BLOCK-scaling (target=512, floor=1) as
    # "(rounded, workspace)" above, so this isolates ONLY the
    # vectorization change. Its own promotion: on real A100 hardware, a
    # dramatic, growing win over "(rounded, workspace)" mirroring the
    # truncating encoding's own curve closely -- roughly flat at
    # M=1024/1536 (0.972x-1.000x), then +4.3% at M=2048 climbing to
    # +21.6% at M=8192, no regressions anywhere tested. Was plain
    # "fp32_bf16"/"TOMOS FP32/BF16"; DEMOTED now that mv_fp32_bf16_
    # transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch (below)
    # sources that slot instead -- kept as its own visible series, same
    # "keep both, promote the winner" convention. A further M_BLOCK
    # retuning sibling, mv_fp32_bf16_transpose_atomiconly_scaled_mblock_
    # vec4_rounded_retuned_floor, measured a further win at every size
    # EXCEPT the SAME M=2048 anomaly the truncating encoding's own
    # retuned_floor showed (-4.7% here) -- cross-validating that anomaly
    # as decode-independent (see above) -- kept as a documented open
    # question, not promoted.
    ("mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded =", "fp32_bf16_vec4_rounded"),
    # "TOMOS FP32/BF16" -- mv_fp32_bf16_transpose_atomiconly_scaled_
    # mblock_vec4_rounded_dispatch's own NEW TRANSPOSE entry point: NOT a
    # new tuned __global__ kernel, a thin host-side router over "(rounded,
    # vec4)" above (M<8192) and its own rounded_retarget2 sibling
    # (target=8192, same shared floor=256, not its own visible CSV/plot
    # series) for M>=8192 -- rounded-encoding counterpart of the
    # "(truncated, test)" dispatcher below, same decode-independent port
    # of fp32_16's own dispatch pattern -- see both kernels' own comments
    # in fp32_16_cuda_kernels.cu. PROMOTED: on real A100 hardware at
    # M=8192 (the threshold itself, where grid math confirms a real
    # M_BLOCK/gridM difference vs "(rounded, vec4)", not noise): +12.7%
    # (689.5 vs 611.9 GFlop/s). CAVEAT: full confirmation across M=12288-
    # 24576 (where the win is predicted to taper to a no-op by M=24576,
    # per grid math) is still pending -- promoted on the strength of the
    # M=8192 result plus the grid-math analysis, not yet a complete sweep.
    ("mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch =", "fp32_bf16"),
]

FIELDS = [
    "M", "K",
    "native_ms", "native_gflops",
    "native_vec4_ms", "native_vec4_gflops",
    "native_unvectorized_ms", "native_unvectorized_gflops",
    "fp32_16_ms", "fp32_16_gflops",
    "fp32_16_retuned_floor_ms", "fp32_16_retuned_floor_gflops",
    "fp32_16_wrapborrow_ms", "fp32_16_wrapborrow_gflops",
    "fp32_16_vec4_ms", "fp32_16_vec4_gflops",
    "fp32_16_scalar_ms", "fp32_16_scalar_gflops",
    "fp32_16_fast_ms", "fp32_16_fast_gflops",
    "fp32_16_fast_vec4_ms", "fp32_16_fast_vec4_gflops",
    "fp32_16_fast_vec4_retuned_ms", "fp32_16_fast_vec4_retuned_gflops",
    "fp32_16_fast_wrapborrow_vec4_ms", "fp32_16_fast_wrapborrow_vec4_gflops",
    "fp32_16_fast_retarget_ms", "fp32_16_fast_retarget_gflops",
    "fp32_16_fast_scalar_ms", "fp32_16_fast_scalar_gflops",
    "fp32_bf16_ms", "fp32_bf16_gflops",
    "fp32_bf16_vec4_rounded_ms", "fp32_bf16_vec4_rounded_gflops",
    "fp32_bf16_rounded_workspace_ms", "fp32_bf16_rounded_workspace_gflops",
    "fp32_bf16_truncated_ms", "fp32_bf16_truncated_gflops",
    "fp32_bf16_truncated_test_ms", "fp32_bf16_truncated_test_gflops",
    "fp32_bf16_truncated_workspace_ms", "fp32_bf16_truncated_workspace_gflops",
    "cublas_ms", "cublas_gflops",
]

rows = []
row = None


def flush():
    if row is not None:
        rows.append(row)


with open(txt_path) as f:
    for line in f:
        line = line.rstrip("\n")
        m = shape_re.match(line)
        if m:
            flush()
            row = {"M": m.group(1), "K": m.group(2)}
            continue
        if row is None:
            continue

        for marker, key in METHOD_MARKERS:
            if marker not in line:
                continue
            if "SKIPPED" in line:
                row[f"{key}_ms"] = "SKIPPED"
                row[f"{key}_gflops"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_ms"] = vm.group(1)
                    row[f"{key}_gflops"] = vm.group(2)
            break
    flush()

with open(csv_path, "w") as out:
    out.write(",".join(FIELDS) + "\n")
    for r in rows:
        out.write(",".join(r.get(field, "") for field in FIELDS) + "\n")
PYEOF

echo
echo "==> Logs written:"
echo "      $TXT_LOG"
echo "      $CSV_LOG"

echo
echo "==> Plotting (avg): $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_recon_vs_cublas_ref('$CSV_LOG', 'avg')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_transpose_fp32_recon_vs_cublas_ref('$CSV_LOG', 'avg')\"" >&2
fi

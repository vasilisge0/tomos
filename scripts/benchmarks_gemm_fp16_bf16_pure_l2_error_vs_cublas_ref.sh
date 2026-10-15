#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_
# cublas_ref.cu and runs the SAME square sweep (M=K=N) scripts/fp16_
# bf16_pure_vs_cutlass_ref_matched.sh itself uses, but measuring
# Frobenius-relative-error against a genuine FP32 (cublasSgemm) reference
# instead of throughput -- for the five "TOMOS" pure/composite-tier
# (tensor-core) GEMM series scripts/plot_fp16_bf16_pure_vs_cutlass_
# ref.m plots: TOMOS FP16, TOMOS BF16, TOMOS FP32/FP16, TOMOS FP32/FP16
# (workspace), TOMOS FP32/BF16 -- PLUS four external, non-TOMOS GEMM
# baselines the .cu's own fp16/bf16 groups now also run on the SAME
# native-rounded A/B as each group's own native TOMOS series: cuBLAS
# FP16 (cublasHgemm), CUTLASS FP16 (cutlass::gemm::device::Gemm, same
# config as scripts/cutlass_fp16_gemm_ref.cu's own binary), cuBLAS BF16
# (cublasGemmEx, CUDA_R_16BF/CUBLAS_COMPUTE_32F), and CUTLASS BF16 (same
# tensorop config as CUTLASS FP16, bf16 element types). See the .cu's own
# header comment for exactly how the shared reference is generated
# (device-side only, to reach the same 65536-cubed scale the performance
# sweep does without a host-memory blowup) and for why, unlike every
# other error benchmark in this codebase, B is ALWAYS fp16/bf16-typed
# here (no uncompressed-B convention exists in this tensor-core tier).
#
# CUTLASS is fetched (shallow clone, header-only usage -- no separate
# library build needed) into CUTLASS_DIR on first run and reused after
# that, same convention as scripts/cutlass_fp16_gemm_ref.sh. Override the
# location with the CUTLASS_DIR env var if you already have a checkout
# elsewhere.
#
# Square sweep only, same list as scripts/fp16_bf16_pure_vs_cutlass_
# ref_matched.sh's own SIZES: 1024/1536/2048/3072/4096/6144/8192/12288/
# 16384/24576/32768/49152/65536. No rectangular sweep here.
#
# Runs the FULL square sweep once per entry in SCALES below -- each is a
# "normal_scale" passed to the .cu binary (see its own header comment),
# widening A's own "normal"-range test data from [-1,1] to [-scale,
# scale]. Every series here accumulates its tensor-core MMA output in
# genuine float32 internally (confirmed by reading gemm_fp16_mma_ptx_
# wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm_
# kernel's own body: float acc[...], fed by mma.sync's own f32.f16.f16.
# f32 PTX variant -- a fixed sm_80 hardware property, not a per-kernel
# choice), so there is no INTERNAL accumulator-overflow concern -- BUT
# three of this tier's five series (TOMOS FP16, TOMOS FP32/FP16, TOMOS
# FP32/FP16 (workspace)) still narrow that float32 accumulator down into
# __half-typed C at the epilogue, and C[i,j] is a SUM OF K PRODUCTS (not
# a single stored A/B value), so its own magnitude grows with sqrt(K) --
# a K-DEPENDENT OUTPUT-STORAGE ceiling, the same shape of constraint as
# the GEVM pure_half/hybrid tier's own internal-accumulator bound, just
# moved to the epilogue instead of mid-computation.
#
# A PREVIOUS version of this comment (and this list's own SCALES, up to
# 16384) reasoned only about each individual A[i,j]/B[i,j] STORAGE value
# staying under fp16's ~65504 ceiling -- true, but irrelevant: it never
# checked C's OWN magnitude, which is what actually overflows. This was
# a REAL bug, caught by a real run: at scale=4096, TOMOS FP16/FP32/FP16/
# FP32/FP16(workspace) all reported "inf" at EVERY shape (M=K=N=1024
# through 32768), not just the largest ones. Root cause: for A ~
# scale*Uniform(-1,1) and B ~ Uniform(-1,1) (B is left unscaled, see
# below), each product has std ~ scale/3, and C[i,j]'s own row-sum over
# K such products has std ~ scale*sqrt(K)/3. At the largest shape in
# this sweep (K=N=M=65536, ~4.3 BILLION output elements), that's
# scale*256/3 -- at scale=4096 that's ~350000, ~5.3x OVER fp16's ~65504
# ceiling already (not just close to it), so virtually every output
# element overflows, exactly matching the observed 100%-"inf" result.
# Because a SINGLE overflowing element already makes the whole Frobenius
# sum "inf" (poisoning the metric entirely) and this shape reaches
# billions of elements (unlike GEVM's own vectors, which top out at
# ~131072), the safety margin has to be far more conservative than the
# ~6-sigma bound the GEVM pure_half/hybrid tier's own SCALES uses --
# expected overflow COUNT (not just sigma) has to stay well under 1
# across the full element count. SCALES capped at 64 keeps std ~
# 64*256/3 ~ 5461 at the worst shape -- z = 65504/5461 ~ 12.0 sigma,
# whose two-tailed overflow probability (~1e-33) times ~4.3e9 elements
# gives an expected overflow count of effectively zero -- comfortably
# safe, unlike 128 (z ~ 6.0, tail ~2e-9, expected count ~8.6 -- NOT
# safe: almost certain to poison at least one element at the full
# sweep's largest shape). TOMOS BF16/TOMOS FP32/BF16 are unaffected
# (bf16 shares fp32's own 8-bit exponent, no realistic ceiling at any
# scale used here) -- capped at the same value anyway, for a fair,
# apples-to-apples comparison across all five series, same convention
# every sibling error benchmark's own SCALES uses. Each tier writes its
# OWN timestamped .txt/.csv (scale embedded in the filename) and its
# OWN plot (scale embedded in both the output filename and the plot
# title) -- one plot per tier, not overlaid, so each is independently
# readable.
#
# --normalized (may appear anywhere among the arguments) scales B (in
# all three of its forms -- B_true, B_half, B_bf16 -- see the .cu's own
# run_shape comment) by 1/sqrt(K) at data-generation time instead of
# leaving it at plain [-1,1]. This is the fix that actually matters for
# THIS row: it keeps C[i,j]'s own row-sum magnitude ~normal_scale
# regardless of K instead of growing like normal_scale*sqrt(K), so the
# K-dependent output-storage overflow this script's own SCALES=(...64)
# was capped to avoid (see above) no longer applies, and the ORIGINAL
# wide SCALES range (up to 16384) becomes safe again -- see the SCALES
# array's own comment below for the switch this triggers. NOTE this is
# implemented at data-generation time, NOT via alpha -- alpha here is
# only multiplied in AFTER the raw float accumulator is already cast to
# __half (`alpha * static_cast<__half>(acc[...])`), so an alpha-based
# fix would not have worked; see the .cu's own run_shape comment for the
# exact line this was confirmed against. Plots land with a "_normalized"
# tag (e.g. ..._scale4096_normalized.png) so they don't overwrite the
# plain run's own plots.
#
# Usage:
#   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh
#       Runs the full default SIZES list at every default SCALES tier.
#   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...] -- N1 [N2 ...]
#       Runs exactly those scale tiers against exactly those N values.
#   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh --scales S1 [S2 ...]
#       Runs exactly those scale tiers against the default SIZES list.
#   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh -- N1 [N2 ...]
#       Runs the default SCALES tiers against exactly those N values.
#   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh --normalized [--scales ... [-- ...]]
#       Same forms as above, plus --normalized (may be combined with any
#       of them, in any position) -- switches the default SCALES to the
#       wider, normalized-safe list (up to 16384).
#
# Rebuilds tomos_cuda first (so this always reflects the current state of
# the kernels, not a stale libtomos_cuda.so) and links the standalone nvcc
# build against it directly -- same convention as every other
# *_vs_cublas_ref.sh script in this codebase.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is only
# reachable inside an active job allocation. If SLURM_JOB_ID isn't already
# set, this script finds the caller's running job on an a100* partition and
# re-execs itself inside it via `srun --overlap`, so both the build and the
# run always land on the actual target GPU. Same pattern as scripts/
# benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh.
#
# Full console output is saved to one timestamped .txt log, and a per-
# shape summary (Frobenius relative error for each method) is parsed out
# into one .csv, both under build/mm_fp16_bf16_pure_l2_error_vs_cublas_
# ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mm_fp16_bf16_pure_l2_error_vs_cublas_ref.cu"
OUT="$BUILD_DIR/mm_fp16_bf16_pure_l2_error_vs_cublas_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/output/error_dimension_scaling/gemm/fp16_compute/data"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOBID="$(squeue -h -u "$USER" -o "%A %P" | awk '$2 ~ /a100/ {print $1; exit}')"
    if [ -z "$JOBID" ]; then
        echo "error: no running A100 job found for $USER (squeue shows none on an a100* partition)." >&2
        echo "       start one first, e.g.: sinteractive -A <account> -p a100-40gb --gres=gpu:1 ..." >&2
        exit 1
    fi
    echo "==> Not inside a job allocation -- re-running inside job $JOBID (srun --overlap)"
    exec srun --jobid="$JOBID" --overlap "$0" "$@"
fi

# --normalized (may appear anywhere among the arguments) -- extracted
# here, before the --scales/-- parsing below, so it composes freely with
# either form. See this script's own header comment for what it does.
NORMALIZED=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--normalized" ]; then
        NORMALIZED=1
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mm_fp16_bf16_pure_l2_error_vs_cublas_ref"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
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

# Without --normalized: NOT the same as scripts/benchmarks_gemm_fp32_
# recon_l2_error_vs_cublas_ref.sh's own SCALES (that file's own C is
# genuine float throughout, no ceiling at any scale) -- capped at 64
# here specifically for this tier's own K-dependent OUTPUT-STORAGE
# ceiling (three of five series narrow their float32 result down into
# __half-typed C) -- see this script's own header comment for the full
# derivation and the real scale=4096/"inf"-at-every-shape bug this
# fixed. WITH --normalized (see this script's own header comment), B is
# scaled by 1/sqrt(K), so C's own row-sum std becomes ~scale/3 --
# INDEPENDENT of K -- instead of ~scale*sqrt(K)/3: at scale=16384,
# std=16384/3~5461, giving the SAME ~12-sigma margin the un-normalized
# scale=64 tier achieves, but now at EVERY shape (not just because K
# stayed small) -- so the full original wide range is safe again.
if [ "$NORMALIZED" -eq 1 ]; then
    SCALES=(1 4 16 64 256 1024 4096 16384)
else
    SCALES=(1 2 4 8 16 32 64)
fi

# Same list as scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh's own
# SIZES.
SQUARE_NS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536)

# --scales S1 [S2 ...] [-- N1 [N2 ...]] / -- N1 [N2 ...] (see this
# script's own header comment for the exact forms).
if [ "${1:-}" = "--scales" ]; then
    shift
    SCALES=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        SCALES+=("$1")
        shift
    done
    if [ "${1:-}" = "--" ]; then
        shift
        if [ "$#" -gt 0 ]; then
            SQUARE_NS=("$@")
        fi
    fi
elif [ "${1:-}" = "--" ]; then
    shift
    if [ "$#" -gt 0 ]; then
        SQUARE_NS=("$@")
    fi
fi

NORM_SUFFIX=""
NORM_FLAG=()
if [ "$NORMALIZED" -eq 1 ]; then
    NORM_SUFFIX="_normalized"
    NORM_FLAG=(--normalized)
fi

for scale in "${SCALES[@]}"; do
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    TXT_LOG="$OUTPUT_DIR/mm_fp16_bf16_pure_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.txt"
    CSV_LOG="$OUTPUT_DIR/mm_fp16_bf16_pure_l2_error_vs_cublas_ref${NORM_SUFFIX}_scale${scale}_${TIMESTAMP}.csv"

    : > "$TXT_LOG"
    for n in "${SQUARE_NS[@]}"; do
        echo
        echo "==> Running mm_fp16_bf16_pure_l2_error_vs_cublas_ref $scale $n${NORM_SUFFIX:+ (normalized)}"
        "$OUT" "$scale" "$n" "${NORM_FLAG[@]}" | tee -a "$TXT_LOG"
    done

    echo
    echo "==> Writing CSV summary: $CSV_LOG"
    python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+) N=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.eE+-]+)")

# Longest/most-specific marker first -- "tomos_fp32_fp16" is a literal
# prefix of "tomos_fp32_fp16_workspace", and "tomos_fp16"/"tomos_bf16"
# are each substrings of "tomos_fp32_fp16"/"tomos_fp32_bf16" respectively
# (both share "fp16"/"bf16" preceded by an underscore) -- wait, actually
# "tomos_fp16" is NOT a substring of "tomos_fp32_fp16" (there's "32_"
# in between: "tomos_" + "fp32_" + "fp16" vs "tomos_" + "fp16"), so no
# collision there either -- REQUIRING only the workspace-vs-plain
# ordering below (verified via a standalone Python collision simulation
# before finalizing this list, same discipline as every other *_l2_
# error_vs_cublas_ref.sh script's own METHOD_MARKERS in this codebase).
METHOD_MARKERS = [
    ("tomos_fp32_fp16_workspace", "tomos_fp32_fp16_workspace"),
    ("tomos_fp32_fp16", "tomos_fp32_fp16"),
    ("tomos_fp32_bf16", "tomos_fp32_bf16"),
    ("tomos_fp16", "tomos_fp16"),
    ("tomos_bf16", "tomos_bf16"),
    ("cublas_fp16", "cublas_fp16"),
    ("cutlass_fp16", "cutlass_fp16"),
    ("cublas_bf16", "cublas_bf16"),
    ("cutlass_bf16", "cutlass_bf16"),
]

FIELDS = ["M", "K", "N"]
for _, key in METHOD_MARKERS:
    FIELDS.append(f"{key}_frobenius_rel_err")

rows = []
row = None

def flush():
    global row
    if row is not None:
        rows.append(row)
    row = None

with open(txt_path) as f:
    for line in f:
        m = shape_re.match(line)
        if m:
            flush()
            row = {"M": m.group(1), "K": m.group(2), "N": m.group(3)}
            continue
        if row is None:
            continue

        for marker, key in METHOD_MARKERS:
            if marker not in line:
                continue
            if "SKIPPED" in line:
                row[f"{key}_frobenius_rel_err"] = "SKIPPED"
            else:
                vm = value_re.search(line)
                if vm:
                    row[f"{key}_frobenius_rel_err"] = vm.group(1)
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

    SCALE_TAG="${scale}${NORM_SUFFIX}"

    echo
    echo "==> Plotting (normal_scale=$scale, normalized=$NORMALIZED): $CSV_LOG"
    if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load matlab/R2024a 2>/dev/null || true
    fi
    if command -v matlab >/dev/null 2>&1; then
        matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')"
    else
        echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plot for normal_scale=$scale." >&2
        echo "         run manually later, e.g.:" >&2
        echo "           module load matlab/R2024a" >&2
        echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref('$CSV_LOG', '$SCALE_TAG')\"" >&2
    fi
done

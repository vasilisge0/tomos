#!/bin/bash
# Builds benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu
# and runs two custom sweeps -- comparing the "full fp32 reconstruction"
# tier of GEMV: mv_fp32_16_accurate and mv_fp32_bf16_accurate (both
# src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu) -- against this
# codebase's own tuned native (unpacked) fp32 GEMV kernel (mv_fp32_
# tiled, src/kernels/cuda/matrix/mv_cuda_kernels.cu), real CUTLASS's
# fp32 SIMT device::Gemv, and cuBLAS's Sgemv, all in one run so the
# numbers share the same GPU clock/thermal/contention state. Mirrors
# scripts/benchmarks_gemm_format_32_16_compute_32.sh's own structure
# (that script's GEMM analog) -- see that script's own comments for the
# reasoning behind each piece not re-explained here.
#
# Drives the built binary's own single-shape modes (plain "<M>" for
# square, "--tall <K> <MN>" for tall-and-thin) in two custom loops,
# rather than using the binary's own default sweep:
#   - Square (M=K): 1024/1536/2048/3072/4096/6144/8192/12288/16384/24576/
#     32768/49152/65536/131072 -- same interleaved 1x/1.5x/2x pattern as
#     scripts/benchmarks_gemm_format_32_16_compute_32.sh's own SQUARE_NS
#     list (finer resolution than plain doubling up to ~8*10^4), with
#     131072 kept as the top anchor since it deliberately crosses
#     mv_fp32_16_accurate's own kMvWideRowThreshold=131072 boundary
#     (fp32_16_cuda_kernels.cu), so every kernel compared here (all three
#     share that same dispatch logic) exercises both its split-K "wide"
#     path and its "narrow"/"narrow_vec" path across this sweep.
#   - Tall-and-thin: K held fixed at 512, then 4096; M swept across
#     1024/2048/4096/8192/16384/32768/65536/131072/262144/524288/
#     1048576 -- the realistic "many rows, few columns" GEMV shape,
#     always keeping M >> K.
# All shapes' output (both sweeps) is combined into one timestamped .txt
# log and parsed into one .csv, same convention as every other
# CUTLASS-comparison script in this codebase.
#
# Usage:
#   scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh
#       Runs both sweeps above. Takes no arguments.
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh
#
# Rebuilds tomos_cuda first (so this always reflects the current state
# of the kernels, not a stale libtomos_cuda.so) and links the standalone
# nvcc build against it directly -- CUTLASS itself is still NOT a
# dependency of tomos's own CMake build, only of this one comparison
# binary.
#
# On Gilbreth the shell this script runs in is usually the login node,
# which has no A100 (only a shared front-end GPU) -- the real A100 is
# only reachable inside an active job allocation. If SLURM_JOB_ID isn't
# already set, this script finds the caller's running job on an a100*
# partition and re-execs itself inside it via `srun --overlap`, so both
# the build and the run always land on the actual target GPU.
#
# All shapes' full console output (both sweeps) is saved to one
# timestamped .txt log, and a per-shape summary (ms + GFlop/s for each
# method) is parsed out into one .csv, both under build/mv_fp32_recon_
# vs_cutlass_ref_output/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu"
OUT="$BUILD_DIR/mv_fp32_recon_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"
OUTPUT_DIR="$BUILD_DIR/mv_fp32_recon_vs_cutlass_ref_output"

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

if [ ! -d "$CUTLASS_DIR" ]; then
    echo "==> CUTLASS not found at $CUTLASS_DIR -- cloning (shallow, v3.5.1)"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory not found at $BUILD_DIR -- configure the project first (see scripts/configure/)." >&2
    exit 1
fi

echo "==> Rebuilding tomos_cuda (so this reflects the current kernel state)"
cmake --build "$BUILD_DIR" --target tomos_cuda -j"$(nproc)"

echo "==> Building: mv_fp32_recon_vs_cutlass_ref"
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TXT_LOG="$OUTPUT_DIR/mv_fp32_recon_vs_cutlass_ref_${TIMESTAMP}.txt"
CSV_LOG="$OUTPUT_DIR/mv_fp32_recon_vs_cutlass_ref_${TIMESTAMP}.csv"

# Square sweep (M=K). Interleaved 1x/1.5x/2x pattern (same as
# scripts/benchmarks_gemm_format_32_16_compute_32.sh's own SQUARE_NS) up
# through ~8*10^4, then 131072 kept as the top anchor to still cross
# mv_fp32_16_accurate's own kMvWideRowThreshold=131072 split-K boundary.
SQUARE_MS=(1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 131072)

# Tall-and-thin sweep: K held fixed at each of these, M swept across the
# same large-size list below (mirrors mv_fp32_16_accurate's own "many
# rows, few columns" realistic case).
TALL_KS=(512 4096)
TALL_MS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

: > "$TXT_LOG"
for m in "${SQUARE_MS[@]}"; do
    echo
    echo "==> Running mv_fp32_recon_vs_cutlass_ref $m"
    "$OUT" "$m" | tee -a "$TXT_LOG"
done
for k in "${TALL_KS[@]}"; do
    for m in "${TALL_MS[@]}"; do
        echo
        echo "==> Running mv_fp32_recon_vs_cutlass_ref --tall $k $m"
        "$OUT" --tall "$k" "$m" | tee -a "$TXT_LOG"
    done
done

echo
echo "==> Writing CSV summary: $CSV_LOG"
python3 - "$TXT_LOG" "$CSV_LOG" <<'PYEOF'
import re
import sys

txt_path, csv_path = sys.argv[1], sys.argv[2]

shape_re = re.compile(r"^\s*M=(\d+) K=(\d+)\s*$")
value_re = re.compile(r"=\s*([0-9.]+)ms \(\s*([0-9.]+) GFlop/s\)")

METHOD_MARKERS = [
    ("mv_fp32_tiled", "native"),
    # mv_fp32_16_accurate (the malloc/free-per-call baseline) is now
    # UNPROMOTED, no longer plotted -- superseded by mv_fp32_16_accurate_
    # ready_atomiconly_scaled_kblock_fast_normal below (see that marker's
    # own comment) -- so no marker for its own "mv_fp32_16_accurate    ="
    # printf line exists here anymore, matching every other unpromoted
    # sibling in this list (e.g. mv_fp32_16_accurate_facc).
    #
    ("mv_fp32_16_accurate_workspace =", "fp32_16_ws"),
    # "TOMOS FP32/FP16 (regular)" -- mv_fp32_16_accurate_ready_
    # atomiconly_scaled_kblock_fast_normal (SAME kernel as mv_fp32_16_
    # accurate_ready_atomiconly_fast_normal, but with a K-dependent
    # K_BLOCK targeting a roughly constant gridK instead of the fixed 128
    # -- caps the atomic contention on y[row] that otherwise grows
    # linearly with K; see its own comment in fp32_16_cuda_kernels.cu).
    # Ties the fixed-K_BLOCK version below the K=16384 crossover (noise)
    # and beats it consistently above it (1.09x-1.11x at K=24576-65536).
    # UNSAFE on arbitrary data (fp32_16_combine_fast_normal's own
    # contract) -- the SAFE, no-external-buffer counterpart is promoted
    # as plain "fp32_16" instead, see that marker's own comment below.
    # (mv_fp32_16_accurate_ready_atomiconly_fast_normal and mv_fp32_16_
    # accurate_ready_fast_normal are both still run/printed for direct
    # comparison, just no longer promoted -- neither's own text collides
    # with this marker: "ready_atomiconly_fast_normal =" isn't a
    # substring of "ready_atomiconly_scaled_kblock_fast_normal =" in
    # either direction.)
    ("mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal =", "fp32_16_regular"),
    # "test" -- a reusable, throwaway placeholder slot, not a permanent
    # series: whatever fp32_16 kernel is currently being A/B'd against
    # the promoted plain "TOMOS FP32/FP16" line gets plotted here under
    # the plain legend label "test", repointed as the experiment moves
    # on. Currently mv_fp32_16_accurate_narrow_vec_cpasync: targets
    # mv_fp32_16_accurate_narrow_vec_kernel directly -- the kernel EVERY
    # OTHER candidate in this family (ready4/ready8/_nohw) falls through
    # to, unchanged, once rows_op >= kMvWideRowThreshold (131072), which
    # is exactly the real LSQR matrix's own shape (M=1,000,000, K=1000).
    # None of the previous "test" experiments below ever actually
    # exercised that kernel. Measured (CSV, tall-and-thin M sweep) at the
    # closest available shape, M=1,048,576: plain safe decode there gets
    # ~71% of native fp32 mv_fp32_tiled's own GFlop/s, and an UNSAFE/fast
    # decode variant at the SAME shape recovers to ~90% -- so roughly two
    # thirds of the gap traces to fp32_16_combine_branchless's own
    # decode LATENCY (a real sequential dependency chain), not bytes
    # moved. This candidate double-buffers each thread's own next-column
    # wh/wr via cp.async into shared memory while decoding the current
    # column, to hide that latency behind the next fetch -- a different
    # lever than the two below, which either reduce work per element
    # (_nohw, measured worse) or amortize it over more rows (ready4/
    # ready8/2-row, where wider than 4 rows did not pan out). Previously
    # mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw (kept
    # below, no longer the "test" slot, as a documented negative-result
    # reference -- see its own printf line's own comment in the .cu for
    # the full derivation); before that mv_fp32_16_accurate_ready8_
    # atomiconly_scaled_kblock, and before that mv_fp32_16_accurate_
    # ready_atomiconly_scaled_kblock (the 2-row SAFE kernel). Trailing
    # " =" (one space) keeps this disjoint from every other marker in
    # this list -- no other printf line in this binary shares this
    # kernel's full name as a prefix.
    ("mv_fp32_16_accurate_narrow_vec_cpasync =", "fp32_16_test"),
    # "test2" -- mv_fp32_16_accurate_narrow_vec_cpasync_no_overflow: a
    # DIFFERENT experiment than what this slot previously held
    # (mv_fp32_16_accurate_narrow_vec_cpasync3, kept below as a
    # documented negative-result reference, no longer plotted here --
    # deepening the cp.async pipeline from 1-ahead to 2-ahead measured
    # IDENTICAL to plain "test", meaning cp.async had already fully hidden
    # fetch latency and the remaining gap to native is fp32_16_combine_
    # branchless's own ALU/issue-throughput cost, not memory latency).
    # This candidate instead reduces that ALU cost directly: same "test"
    # structure (4-rows/lane, 1-ahead cp.async), decoding via fp32_16_
    # combine_branchless_no_overflow instead -- skips the OVERFLOW escape
    # path entirely, valid only under the caller's guarantee that this
    # matrix's real fp32 data never needs wh's overflow encoding (only
    # ever underflow, if anything). See its own comment in fp32_16_cuda_
    # kernels.cu for the exact savings and the correctness caveat. Trailing
    # " =" keeps this disjoint from the "test" marker above and from its
    # own now-demoted predecessor's printf line below.
    ("mv_fp32_16_accurate_narrow_vec_cpasync_no_overflow =", "fp32_16_test2"),
    # "test3" -- mv_fp32_16_ready4_atomiconly_scaled_kblock_cpasync_
    # no_overflow: the SAME two levers ("test"/"test2" above) applied to
    # mv_fp32_16_ready4_atomiconly_scaled_kblock_kernel instead -- the
    # kernel actually used for every SQUARE shape in this benchmark's own
    # sweep (rows_op always < kMvWideRowThreshold there), unlike "test"/
    # "test2" which only ever fire at large, tall/narrow M (hence showing
    # up identical to plain "TOMOS FP32/FP16" on the square plot --
    # expected, that kernel was never exercised there). cp.async double-
    # buffers each thread's own wh/wr uint64_t reads; no_overflow decode
    # skips the OVERFLOW escape path entirely (same caller-guaranteed
    # assumption as "test2"). See its own comment in fp32_16_cuda_
    # kernels.cu for the full derivation.
    ("mv_fp32_16_ready4_atomiconly_scaled_kblock_cpasync_no_overflow =", "fp32_16_test3"),
    # "test4" -- mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_
    # no_overflow: "test3"'s two levers combined with 8-rows/lane
    # register-blocking (mv_fp32_16_ready8_atomiconly_scaled_kblock's own
    # design, previously found NOT to pan out in isolation -- see that
    # kernel's own comment/history -- worth re-testing now that decode is
    # cheaper and prefetched). See its own comment in fp32_16_cuda_
    # kernels.cu for the full derivation.
    #
    # PROMOTED: plot_gemv_fp32_recon_vs_cutlass_ref.m now plots this
    # fp32_16_test4 column as "TOMOS FP32 / FP16 (fast)" (not plain "test4"
    # -- see that .m file's own comment for the full promotion rationale
    # and experiments-gemv.md for the complete investigation writeup).
    # The CSV column key itself stays "fp32_16_test4" unchanged -- only
    # the plotted label changed, same convention "fp32_16_regular" (still
    # plotted as "TOMOS FP32 / FP16 (implicit)") already uses.
    ("mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow =", "fp32_16_test4"),
    # "test5" -- mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync:
    # byte-for-byte "test4" above, EXCEPT decoding via the GENERAL
    # fp32_16_combine_branchless instead of the no-overflow variant --
    # SAFE on arbitrary data, no caller-guaranteed assumption needed.
    # Isolates cp.async+8-rows' own contribution from the no-overflow
    # decode's contribution -- if this stays close to "test4", cp.async+
    # 8-rows alone is enough to promote safely; if it falls back toward
    # "test3"/plain ready4, the no-overflow assumption is still doing
    # most of the work. See its own comment in fp32_16_cuda_kernels.cu
    # for the full derivation. Trailing " =" keeps this disjoint from
    # "test4"'s own marker just above (that marker's own longer text
    # diverges right after "cpasync": "_no_overflow =" vs this marker's
    # own bare " =", so neither is a substring of the other's printf
    # line).
    ("mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync =", "fp32_16_test5"),
    #
    # "TOMOS FP32/FP16" (plain) -- mv_fp32_16_accurate_ready4_
    # atomiconly_scaled_kblock: 4-rows/lane extension of the 2-row SAFE
    # kernel above (packs 4 adjacent rows per lane instead of 2, uint64_t
    # loads) -- amortizing the expensive fp32_16_combine_branchless
    # decode over more rows won outright on real hardware (1.070x-1.115x
    # vs the 2-row kernel, tall-and-thin M sweep at K=4096; see its own
    # comment in fp32_16_cuda_kernels.cu). Safe on arbitrary data, needs
    # no external workspace/scratch buffer at all. Trailing " =" (one
    # space) is required, not just "mv_fp32_16_accurate_ready4_
    # atomiconly_scaled_kblock" alone -- no other printf line in this
    # binary shares that prefix, so this is really just consistency with
    # every other marker's own discipline in this list, not a live
    # collision risk here.
    ("mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock =", "fp32_16"),
    # mv_fp32_bf16_accurate (the malloc/free-per-call baseline) is now
    # UNPROMOTED, no longer plotted -- superseded by mv_fp32_bf16_
    # accurate_ready_atomiconly_scaled_kblock_workspace below (see that
    # marker's own comment) -- so no marker for its own "mv_fp32_bf16_
    # accurate  =" printf line exists here anymore, matching every other
    # unpromoted sibling in this list (e.g. mv_fp32_16_accurate_facc).
    #
    # "TOMOS FP32/BF16 (workspace)" -- mv_fp32_bf16_accurate_workspace:
    # the genuinely-always-uses-workspace sibling (classic column-split
    # wide kernel + reduce, no row-vectorized fast path at all -- see its
    # own comment in fp32_16_cuda_kernels.cu). Unlike the ready_
    # atomiconly_scaled_kblock_workspace marker below, `workspace` really
    # is read and written on every call here. Trailing " =" (one space,
    # the exact spacing the real printf line uses) is required, not just
    # "mv_fp32_bf16_accurate_workspace" alone -- see the next marker's own
    # comment for why a bare substring match would be unsafe here too.
    ("mv_fp32_bf16_accurate_workspace =", "fp32_bf16_ws"),
    # "TOMOS FP32/BF16 (truncated)" -- mv_fp32_bf16_accurate_ready_
    # atomiconly_scaled_kblock_workspace: mv_fp32_16_accurate_ready_
    # atomiconly_scaled_kblock_fast_normal's own row-vectorized/
    # atomiconly/scaled-K_BLOCK fast path, decoding via fp32_bf16_combine
    # against plain-truncating-split (fp32_bf16_split) data (needs no
    # unsafe-on-arbitrary-data shortcut at all, unlike every fp32_16
    # "_fast_normal" sibling -- see that kernel's own comment in fp32_16_
    # cuda_kernels.cu). Every shape this benchmark exercises calls it with
    # NO_TRANSPOSE/ld_a==m, which always takes the fast path and never
    # touches `workspace` at all (confirmed by direct A/B measurement) --
    # it beats mv_fp32_bf16_accurate (unpromoted) by 8-25% at every size
    # tested. Was PLAIN "fp32_bf16"; DEMOTED to "fp32_bf16_truncated" now
    # that mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded
    # (below) sources the plain "fp32_bf16" column instead. Full kernel
    # name is unique -- no substring-collision risk with the "fp32_bf16_
    # ws" marker above (that marker's own trailing " =" already excludes
    # this longer line), the "_rounded" marker below (diverges at
    # "workspace" vs "rounded"), or any other marker in this list.
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace =", "fp32_bf16_truncated"),
    # "TOMOS FP32/BF16" (plain) -- mv_fp32_bf16_accurate_ready_
    # atomiconly_scaled_kblock_rounded: byte-for-byte the same fast path
    # as the "(truncated)" marker above, but decoding via fp32_bf16_
    # combine_rounded against ROUNDED-split (fp32_bf16_split_rounded)
    # data instead -- wr's bit0 there is a round_flag (recording whether
    # bf16 rounding stepped wh up by 1), not a real mantissa bit, so this
    # is a genuinely different encoding, not just a different decode of
    # the same bits (see mv_fp32_bf16_accurate_ready_atomiconly_scaled_
    # kblock_rounded's own comment in fp32_16_cuda_kernels.cu). PROMOTED
    # here as the PLAIN "fp32_bf16" column.
    ("mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded =", "fp32_bf16"),
    ("cutlass (real, SIMT fp32)", "cutlass"),
    ("cutlass SKIPPED", "cutlass"),
    ("cublas (Sgemv)", "cublas"),
]

FIELDS = [
    "M", "K",
    "native_ms", "native_gflops",
    "fp32_16_ms", "fp32_16_gflops",
    "fp32_16_test_ms", "fp32_16_test_gflops",
    "fp32_16_test2_ms", "fp32_16_test2_gflops",
    "fp32_16_test3_ms", "fp32_16_test3_gflops",
    "fp32_16_test4_ms", "fp32_16_test4_gflops",
    "fp32_16_test5_ms", "fp32_16_test5_gflops",
    "fp32_16_ws_ms", "fp32_16_ws_gflops",
    "fp32_16_regular_ms", "fp32_16_regular_gflops",
    "fp32_bf16_ms", "fp32_bf16_gflops",
    "fp32_bf16_truncated_ms", "fp32_bf16_truncated_gflops",
    "fp32_bf16_ws_ms", "fp32_bf16_ws_gflops",
    "cutlass_ms", "cutlass_gflops",
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
echo "==> Plotting: $CSV_LOG"
if ! command -v matlab >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load matlab/R2024a 2>/dev/null || true
fi
if command -v matlab >/dev/null 2>&1; then
    matlab -batch "addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref('$CSV_LOG')"
else
    echo "warning: matlab not found (tried 'module load matlab/R2024a') -- skipping plots." >&2
    echo "         run manually later, e.g.:" >&2
    echo "           module load matlab/R2024a" >&2
    echo "           matlab -batch \"addpath('$REPO_ROOT/scripts'); plot_gemv_fp32_recon_vs_cutlass_ref('$CSV_LOG')\"" >&2
fi

#!/bin/bash
# Builds and runs a focused benchmark comparing just three tomos kernels
# (gemm_fp16_mma_ptx_wide_bigreg_sq, gemm_fp16_mma_ptx_wide_bigreg,
# gemm_fp16_mma_ptx_cutlass_config) against real CUTLASS's own
# device::Gemm and cuBLAS's Hgemm, all in one run so the numbers share
# the same GPU clock/thermal/contention state
# (benchmarks/kernels/cuda/matrix/wide_bigreg_family_vs_cutlass_ref.cu).
#
# This is narrower than scripts/verify_mm_fp16_gemm.sh's full ~50-kernel
# sweep -- just this family vs the two external references, with a
# one-line "best of ours vs cublas/cutlass" summary per size.
#
# Usage:
#   scripts/wide_bigreg_family_vs_cutlass_ref.sh [N]
#
# With no args, sweeps 1024/2048/4096/8192/16384. Pass a single N to
# benchmark just that size:
#   scripts/wide_bigreg_family_vs_cutlass_ref.sh 8192
#
# CUTLASS is fetched (shallow clone, header-only -- no separate library
# build) into CUTLASS_DIR on first run and reused after that. Override
# with the CUTLASS_DIR env var if you already have a checkout:
#   CUTLASS_DIR=/path/to/cutlass scripts/wide_bigreg_family_vs_cutlass_ref.sh
#
# Rebuilds tomos_cuda first (so this always reflects the current state
# of the three kernels, not a stale libtomos_cuda.so) and links the
# standalone nvcc build against it directly -- CUTLASS itself is still
# NOT a dependency of tomos's own CMake build, only of this one
# comparison binary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/wide_bigreg_family_vs_cutlass_ref.cu"
OUT="$BUILD_DIR/wide_bigreg_family_vs_cutlass_ref"
TOMOS_CUDA_DIR="$BUILD_DIR/src/kernels/cuda"

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

echo "==> Building: wide_bigreg_family_vs_cutlass_ref"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    "$SRC" -lcublas \
    -L"$TOMOS_CUDA_DIR" -ltomos_cuda -Xlinker -rpath -Xlinker "$TOMOS_CUDA_DIR" \
    -o "$OUT"

echo
echo "==> Running wide_bigreg_family_vs_cutlass_ref $*"
"$OUT" "$@"

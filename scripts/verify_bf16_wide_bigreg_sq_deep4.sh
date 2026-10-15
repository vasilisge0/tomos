#!/bin/bash
# Builds and runs the correctness + throughput check for
# gemm_bf16_mma_ptx_wide_bigreg_sq_deep4 (the bfloat16 sibling of
# gemm_fp16_mma_ptx_wide_bigreg_sq_deep4, against real CUTLASS and cuBLAS)
# AND mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4 (its FP32_BF16 "composite
# precision" sibling -- A decoded from the packed wh half only, see
# fp32_bf16_pack.hpp and that kernel's own comment in mm_cuda_kernels.cu)
# (benchmarks/kernels/cuda/matrix/verify_bf16_wide_bigreg_sq_deep4.cu).
#
# Usage:
#   scripts/verify_bf16_wide_bigreg_sq_deep4.sh
#       Runs correctness for both kernels (N=264/384/648/1024, vs a naive
#       CPU double-accumulated reference) then a throughput sweep vs real
#       CUTLASS and cuBLAS for gemm_bf16_mma_ptx_wide_bigreg_sq_deep4 (N=
#       1024/2048/4096/8192/16384/32768/65536), then a throughput sweep
#       for mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4 (same sizes,
#       compared against the first sweep's own "ours (deep4)" number).
#   scripts/verify_bf16_wide_bigreg_sq_deep4.sh 8192
#       Skips correctness, benchmarks just that one size (both kernels).
#
# CUTLASS is fetched (shallow clone, header-only) into CUTLASS_DIR on
# first run and reused after that, same as this repo's other CUTLASS-
# comparison scripts. Override with the CUTLASS_DIR env var if you
# already have a checkout.
#
# Rebuilds tomos_cuda first so this always reflects the current state of
# the kernel, then links the standalone nvcc build against it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.configure.sh" ] && source "$REPO_ROOT/.configure.sh"
BUILD_DIR="$REPO_ROOT/build"
CUTLASS_DIR="${CUTLASS_DIR:-$REPO_ROOT/build/cutlass-ref}"
SRC="$REPO_ROOT/benchmarks/kernels/cuda/matrix/verify_bf16_wide_bigreg_sq_deep4.cu"
OUT="$BUILD_DIR/verify_bf16_wide_bigreg_sq_deep4"
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

echo "==> Building: verify_bf16_wide_bigreg_sq_deep4"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_80 \
    -I"$CUTLASS_DIR/include" -I"$CUTLASS_DIR/tools/util/include" \
    -I"$REPO_ROOT/src/core/matrix" \
    "$SRC" -lcublas \
    -L"$TOMOS_CUDA_DIR" -ltomos_cuda -Xlinker -rpath -Xlinker "$TOMOS_CUDA_DIR" \
    -o "$OUT"

echo
echo "==> Running verify_bf16_wide_bigreg_sq_deep4 $*"
"$OUT" "$@"

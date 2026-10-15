#include "mv_cuda_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <type_traits>
#include <vector>

#include "type_macros.hpp"

namespace tomos {
namespace cuda {

// ====================================================================
// tomos::cuda::experimental -- every dense GEVM kernel/wrapper in this
// file EXCEPT the small set actually used by a plotted benchmark or by
// production dispatch (src/dispatch/dense_registries.cpp), re-exposed
// via thin forwarding wrappers just before the CSR SpMV section at the
// end of this file. Purely organizational -- no behavior change. The
// CSR SpMV kernels (mv_csr, spmv_csr_kernel, etc.) are a different
// operation family entirely, production-called from
// sparse_registries.cpp, and are deliberately left OUTSIDE this
// namespace, unchanged.
// ====================================================================
namespace experimental {

namespace {
void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[mv_strided] %s failed: status %d\n", what,
                static_cast<int>(status));
}
}  // namespace

// Generic mixed-precision dense matvec via a device kernel.
template <typename a_t, typename x_t, typename y_t>
__global__ void gemv_generic_kernel(TransposeType trans, double alpha,
                                    size_t m, size_t n, a_t* A, size_t ld_a,
                                    x_t* x, double beta, y_t* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_op) return;

    double acc = 0.0;
    for (size_t j = 0; j < cols_op; ++j) {
        size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
        size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
        acc += static_cast<double>(A[a_row + a_col * ld_a]) *
               static_cast<double>(x[j]);
    }
    // Per BLAS convention, beta == 0 means y need not be set on entry -- its
    // device memory may be uninitialized, so beta * y[i] must not be
    // evaluated in that case (0 * NaN is NaN, not 0).
    y[i] = (beta != 0.0)
              ? static_cast<y_t>(alpha * acc + beta * static_cast<double>(y[i]))
              : static_cast<y_t>(alpha * acc);
}

template <typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, dim<2> size, a_t* A,
                size_t ld_a, x_t* x, size_t ld_x, double beta, y_t* y,
                size_t ld_y, cublasHandle_t handle) {
    size_t rows_op =
        (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    int threads = 256;
    int blocks = (rows_op + threads - 1) / threads;
    cudaGetLastError();
    gemv_generic_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A,
                                             ld_a, x, beta, y);
    cudaError_t _e = cudaGetLastError();
    if (_e != cudaSuccess)
        fprintf(stderr, "[mv_strided] launch: %s\n", cudaGetErrorString(_e));
    _e = cudaDeviceSynchronize();
    if (_e != cudaSuccess)
        fprintf(stderr, "[mv_strided] sync: %s\n", cudaGetErrorString(_e));
}

// cuBLAS specialization for double.
template <>
void mv_strided<double, double, double>(TransposeType trans, double alpha,
                                        dim<2> size, double* A, size_t ld_a,
                                        double* x, size_t ld_x, double beta,
                                        double* y, size_t ld_y,
                                        cublasHandle_t handle) {
    cublasOperation_t op = (trans == TransposeType::NO_TRANSPOSE)
                               ? CUBLAS_OP_N
                               : CUBLAS_OP_T;
    check_cublas(cublasDgemv(handle, op, size[0], size[1], &alpha, A, ld_a, x,
                             1, &beta, y, 1),
                "cublasDgemv");
}

// cuBLAS specialization for float.
template <>
void mv_strided<float, float, float>(TransposeType trans, double alpha,
                                     dim<2> size, float* A, size_t ld_a,
                                     float* x, size_t ld_x, double beta,
                                     float* y, size_t ld_y,
                                     cublasHandle_t handle) {
    cublasOperation_t op = (trans == TransposeType::NO_TRANSPOSE)
                               ? CUBLAS_OP_N
                               : CUBLAS_OP_T;
    float f_alpha = static_cast<float>(alpha);
    float f_beta = static_cast<float>(beta);
    check_cublas(cublasSgemv(handle, op, size[0], size[1], &f_alpha, A, ld_a,
                             x, 1, &f_beta, y, 1),
                "cublasSgemv");
}

// cuBLAS specialization for __half. cuBLAS has no native half-precision
// GEMV routine (no cublasHgemv), so the matvec is expressed as a GEMM
// against a single-column "matrix" instead — the standard workaround for
// this gap: y (rows_op x 1) = alpha * op(A) * x (cols_op x 1) + beta * y,
// via cublasHgemm with n=1. x_t/y_t are treated as contiguous (unit-stride)
// vectors, same as the cublasDgemv/cublasSgemv specializations above (which
// likewise ignore ld_x/ld_y and always pass incx=incy=1).
template <>
void mv_strided<__half, __half, __half>(TransposeType trans, double alpha,
                                        dim<2> size, __half* A, size_t ld_a,
                                        __half* x, size_t ld_x, double beta,
                                        __half* y, size_t ld_y,
                                        cublasHandle_t handle) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];
    cublasOperation_t op = (trans == TransposeType::NO_TRANSPOSE)
                               ? CUBLAS_OP_N
                               : CUBLAS_OP_T;
    __half h_alpha = __double2half(alpha);
    __half h_beta = __double2half(beta);
    check_cublas(cublasHgemm(handle, op, CUBLAS_OP_N,
                             static_cast<int>(rows_op), 1,
                             static_cast<int>(cols_op), &h_alpha, A,
                             static_cast<int>(ld_a), x,
                             static_cast<int>(cols_op), &h_beta, y,
                             static_cast<int>(rows_op)),
                "cublasHgemm (mv)");
}

// Force device-code emission for all mixed-precision combinations.
// With separable compilation, __global__ templates require explicit
// device-side instantiation; implicit instantiation from the host-function
// INST_MV_STRIDED block is not sufficient.
#define INST_GEMV_KERNEL(a_t, x_t, y_t)                                   \
    template __global__ void gemv_generic_kernel<a_t, x_t, y_t>(          \
        TransposeType, double, size_t, size_t, a_t*, size_t, x_t*, double, \
        y_t*);
FOR_ALL_FP_TRIPLES(INST_GEMV_KERNEL)
#undef INST_GEMV_KERNEL

#define INST_MV_STRIDED(a_t, x_t, y_t)                                    \
    template void mv_strided<a_t, x_t, y_t>(TransposeType, double, dim<2>, \
                                            a_t*, size_t, x_t*, size_t,    \
                                            double, y_t*, size_t,          \
                                            cublasHandle_t);
FOR_ALL_FP_TRIPLES(INST_MV_STRIDED)
#undef INST_MV_STRIDED

// Tuned native (unpacked) fp32 GEMV family -- mirrors mv_fp32_16_accurate's
// own 3-way shape dispatch and kernel designs (src/kernels/cuda/matrix/
// fp32_16_cuda_kernels.cu) verbatim, with the packed-decode step
// (fp32_16_combine(wh,wr)) simply replaced by a plain A[idx] read: that
// family's own design history (coalesced one-thread-per-row access beats
// warp-reduction tricks for this memory-bound workload; a single wide
// vectorized load matters more than FMA-side unrolling; split-K column-
// partitioning is what actually fixes "too few rows to fill the GPU",
// not a different reduction strategy) is architecture, not precision,
// reasoning -- it applies identically to plain fp32 A. Same tile size,
// thresholds, and row-blocking factor (all reused directly, not
// redefined) so the two families stay directly comparable.
//
// Correctness-checked in benchmarks/kernels/cuda/matrix/verify_mv_fp32_16_
// gemv.cu ("fp32_tiled" case, both TRANSPOSE directions) and benchmarked in
// benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_cutlass_ref.cu as
// "TOMOS FP32" (native) -- no longer UNTESTED.
static constexpr int kMvFp32TileSize = 256;
static constexpr size_t kMvFp32TargetTotalThreads = 131072;
static constexpr size_t kMvFp32WideRowThreshold = kMvFp32TargetTotalThreads;
static constexpr size_t kMvFp32NarrowVecColThreshold = 64;
static constexpr int kMvFp32RowsPerThread = 4;

// M-chunk config for mv_fp32_transpose_atomiconly_scaled_mblock_kernel
// below. kMvFp32TransposeMBlock/kMvFp32TransposeWarpsPerBlock values
// match randls2.0's own launch_sgemv_cm_transpose_splitM's choice
// (WARPS_PER_BLOCK=8, M_CHUNK=8192, src/kernels/cuda/matrix/dense_
// kernels.cu in that sibling project) -- kMvFp32TransposeMBlock is now
// the CEILING M_BLOCK can reach, not the value always used: the dispatch
// below (mv_fp32_tiled's own TRANSPOSE branch) shrinks it for smaller m,
// targeting kMvFp32TransposeTargetGridBlocks total blocks, after real-
// hardware measurement showed the fixed value alone left small/medium M
// shapes underoccupied (see that dispatch code's own comment for the
// data and full derivation). kMvFp32TransposeTargetGridBlocks=512 is a
// round-number "several blocks per SM on a 100+ SM datacenter GPU"
// target, not derived from a specific device query -- same "fixed
// constant, not per-device-tuned" convention kMvReadyTargetKSplits (=128,
// fp32_16_cuda_kernels.cu) already uses for the analogous NO_TRANSPOSE
// scaling.
static constexpr int kMvFp32TransposeMBlock = 8192;
static constexpr int kMvFp32TransposeWarpsPerBlock = 8;
static constexpr int kMvFp32TransposeTargetGridBlocks = 512;

static int choose_col_splits_fp32(size_t rows_op, size_t cols_op) {
    size_t row_tiles = (rows_op + kMvFp32TileSize - 1) / kMvFp32TileSize;
    size_t total_row_threads = row_tiles * kMvFp32TileSize;
    size_t splits = kMvFp32TargetTotalThreads / std::max<size_t>(total_row_threads, 1);
    splits = std::max<size_t>(splits, size_t(1));
    return static_cast<int>(std::min(splits, cols_op));
}

// Same formula as choose_col_splits_fp32 above, but takes an explicit
// target-total-threads instead of the fixed kMvFp32TargetTotalThreads --
// mirrors fp32_16_cuda_kernels.cu's own choose_col_splits_target, which
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x uses at
// kMvTargetTotalThreads*4 (4x MORE column-splits than the plain *1
// formula at the same shape -- confirmed directly: col_splits=8 vs 2 at
// M=K=65536, 512 vs 128 at M=1024). mv_fp16_native_storage's own col-split
// tiers originally reused choose_col_splits_fp32 unmodified (matching
// this file's plain fp32 convention, not independently re-tuned for
// native fp16 storage's own, lighter per-element cost), which measured
// consistently 5-30% slower than "TOMOS FP16 (implicit, fast)" at every
// M despite doing zero decode work -- confirmed via this exact formula
// gap, not a decode-cost difference (native storage decodes nothing).
static int choose_col_splits_fp32_target(size_t rows_op, size_t cols_op, size_t target_total_threads) {
    size_t row_tiles = (rows_op + kMvFp32TileSize - 1) / kMvFp32TileSize;
    size_t total_row_threads = row_tiles * kMvFp32TileSize;
    size_t splits = target_total_threads / std::max<size_t>(total_row_threads, 1);
    splits = std::max<size_t>(splits, size_t(1));
    return static_cast<int>(std::min(splits, cols_op));
}

// Wide-matrix path only: sums the col_splits partial dot products (see
// mv_fp32_wide_kernel below) for each row and applies alpha/beta -- byte-
// for-byte identical to mv_fp32_16_wide_reduce_kernel's own logic (its
// partials are already plain double regardless of A's own source
// precision, so this reduction step never touched fp32_16 specifically
// in the first place).
__global__ void mv_fp32_wide_reduce_kernel(double alpha, double beta, size_t rows_op,
                                            size_t col_splits, const double* partial,
                                            float* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_op) return;
    double acc = 0.0;
    for (size_t s = 0; s < col_splits; ++s) acc += partial[s * rows_op + i];
    y[i] = (beta != 0.0)
              ? static_cast<float>(alpha * acc + beta * static_cast<double>(y[i]))
              : static_cast<float>(alpha * acc);
}

// Used for NARROW matrices with few columns (cols_op < kMvFp32NarrowVecColThreshold)
// -- see mv_fp32_16_accurate_tiled_kernel's own comment (fp32_16_cuda_
// kernels.cu) for the full design rationale, identical here except A is
// read directly instead of decoded.
__global__ void mv_fp32_tiled_kernel(TransposeType trans, double alpha, size_t m,
                                      size_t n, const float* A, size_t ld_a,
                                      const float* x, double beta, float* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float x_tile[kMvFp32TileSize];
    double acc = 0.0;
    for (size_t tile_start = 0; tile_start < cols_op; tile_start += kMvFp32TileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp32TileSize), cols_op - tile_start);
        if (threadIdx.x < tile_len) x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        __syncthreads();

        if (i < rows_op) {
            for (size_t jj = 0; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                acc += static_cast<double>(A[idx]) * static_cast<double>(x_tile[jj]);
            }
        }
        __syncthreads();
    }

    if (i >= rows_op) return;
    y[i] = (beta != 0.0)
              ? static_cast<float>(alpha * acc + beta * static_cast<double>(y[i]))
              : static_cast<float>(alpha * acc);
}

// Used for NARROW matrices once cols_op reaches kMvFp32NarrowVecColThreshold
// -- see mv_fp32_16_accurate_narrow_vec_kernel's own comment for the full
// design rationale (why wide vectorized loads beat FMA-side unrolling
// here). Each thread computes kMvFp32RowsPerThread consecutive output
// rows; for NO_TRANSPOSE (away from the trailing partial row group) those
// rows are contiguous in memory, so A is read via a single float4 load
// (16 bytes) instead of kMvFp32RowsPerThread separate 4-byte loads --
// the direct fp32 analog of the fp32_16 kernel's ushort4-pair-then-decode
// load, minus the decode.
__global__ void mv_fp32_narrow_vec_kernel(TransposeType trans, double alpha, size_t m,
                                           size_t n, const float* __restrict__ A,
                                           size_t ld_a, const float* __restrict__ x,
                                           double beta, float* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i0 = (blockIdx.x * blockDim.x + threadIdx.x) * static_cast<size_t>(kMvFp32RowsPerThread);
    if (i0 >= rows_op) return;
    bool vectorized = (trans == TransposeType::NO_TRANSPOSE) &&
                      (i0 + kMvFp32RowsPerThread <= rows_op) &&
                      (ld_a % static_cast<size_t>(kMvFp32RowsPerThread) == 0);

    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    for (size_t j = 0; j < cols_op; ++j) {
        double x_val = static_cast<double>(__ldg(&x[j]));
        if (vectorized) {
            size_t base = i0 + j * ld_a;
            float4 a4 = *reinterpret_cast<const float4*>(&A[base]);
            acc0 += static_cast<double>(a4.x) * x_val;
            acc1 += static_cast<double>(a4.y) * x_val;
            acc2 += static_cast<double>(a4.z) * x_val;
            acc3 += static_cast<double>(a4.w) * x_val;
        } else {
            for (int k = 0; k < kMvFp32RowsPerThread; ++k) {
                size_t i = i0 + static_cast<size_t>(k);
                if (i >= rows_op) break;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                double term = static_cast<double>(A[idx]) * x_val;
                switch (k) {
                    case 0: acc0 += term; break;
                    case 1: acc1 += term; break;
                    case 2: acc2 += term; break;
                    default: acc3 += term; break;
                }
            }
        }
    }

    double acc[kMvFp32RowsPerThread] = {acc0, acc1, acc2, acc3};
    for (int k = 0; k < kMvFp32RowsPerThread; ++k) {
        size_t i = i0 + static_cast<size_t>(k);
        if (i >= rows_op) break;
        y[i] = (beta != 0.0)
                  ? static_cast<float>(alpha * acc[k] + beta * static_cast<double>(y[i]))
                  : static_cast<float>(alpha * acc[k]);
    }
}

// For WIDE matrices (few rows, many columns, OR moderately-sized/square --
// see kMvFp32WideRowThreshold) only -- same coalesced one-thread-per-row-
// tile, shared-memory-x-tiled layout as mv_fp32_tiled_kernel, but each
// block only covers a slice of cols_op (blockIdx.y selects which of
// col_splits column-chunks) and writes its own partial dot product
// instead of finishing the row -- see mv_fp32_wide_reduce_kernel for the
// combining pass. Mirrors mv_fp32_16_accurate_wide_kernel exactly.
__global__ void mv_fp32_wide_kernel(TransposeType trans, size_t m, size_t n,
                                     const float* A, size_t ld_a, const float* x,
                                     size_t col_splits, double* partial) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t chunk = (cols_op + col_splits - 1) / col_splits;
    size_t col_start = blockIdx.y * chunk;
    size_t col_end = min(col_start + chunk, cols_op);

    __shared__ float x_tile[kMvFp32TileSize];
    double acc = 0.0;
    for (size_t tile_start = col_start; tile_start < col_end; tile_start += kMvFp32TileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp32TileSize), col_end - tile_start);
        if (threadIdx.x < tile_len) x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        __syncthreads();

        if (i < rows_op) {
            for (size_t jj = 0; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                acc += static_cast<double>(A[idx]) * static_cast<double>(x_tile[jj]);
            }
        }
        __syncthreads();
    }

    if (i < rows_op) partial[blockIdx.y * rows_op + i] = acc;
}

// Applies y[i] *= beta before mv_fp32_transpose_atomiconly_scaled_mblock_
// kernel's own atomicAdd accumulation -- byte-for-byte the same role as
// mv_fp32_16_scale_y_kernel (fp32_16_cuda_kernels.cu) plays for that
// family's own atomiconly kernels, just a separate copy since that one
// lives in a different translation unit with no shared declaration.
__global__ void mv_fp32_scale_y_kernel(size_t rows_op, double beta, float* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_op) return;
    y[i] = static_cast<float>(beta * static_cast<double>(y[i]));
}

// TRANSPOSE-specialized fast GEMV kernel: y = alpha * A^T * x + beta * y,
// A stored MxK column-major (A[row + col*ld_a], ld_a == m required -- see
// mv_fp32_tiled's own dispatch guard below). None of mv_fp32_tiled's three
// existing kernels (tiled/narrow_vec/wide, above) are actually fast for
// TRANSPOSE: narrow_vec's own float4-vectorized load is gated NO_TRANSPOSE-
// only (see its own comment), so every TRANSPOSE call there already fell
// back to a scalar, strided-access loop -- this codebase had NO transpose-
// specialized GEMV kernel of any kind before this one.
//
// Ported from randls2.0's own sgemv_cm_transpose_splitM_atomic_xshared
// (src/kernels/cuda/matrix/dense_kernels.cu, that sibling project -- same
// author, no separate license/attribution needed) -- not a verbatim copy,
// adapted to this codebase's own conventions (extern __shared__ sized at
// launch instead of a template parameter, kMvFp32Transpose* naming, beta
// handled via a separate prescale pass the way every other "atomiconly"
// kernel in this codebase does it, rather than threading beta through the
// kernel itself), but the core algorithmic idea is randls2.0's: this is
// the "transpose" analog of mv_fp32_16_ready_atomiconly_scaled_kblock_
// kernel's own row-vectorized design (fp32_16_cuda_kernels.cu), with the
// roles of "warp" and "lane" swapped to keep memory access coalesced under
// the transposed access pattern:
//   - NO_TRANSPOSE fast kernels: each LANE owns one (or more) whole row,
//     the WARP's 32 lanes cover 32 different rows in lockstep as they walk
//     together across K -- coalesced because A[row + k*ld] for 32
//     consecutive `row` values (one per lane) at a fixed k are contiguous.
//   - This kernel (TRANSPOSE): each WARP owns one whole output column
//     (one y[col]), and the 32 LANES within that warp walk together across
//     the row (reduction) dimension -- coalesced because A[row + col*ld]
//     for 32 consecutive `row` values (one per lane) at a fixed col are
//     STILL contiguous (column-major storage), so this is the same
//     coalescing trick, just with the warp now computing a REDUCTION
//     (dot product) instead of 32 independent FMA chains, hence the
//     warp-shuffle tree reduction at the end instead of one write per
//     lane.
// The reduction (row) dimension m is split into kMvFp32TransposeMBlock-
// sized chunks across blockIdx.x, each contributing a partial sum via
// atomicAdd -- the TRANSPOSE-axis analog of K_BLOCK-splitting K across
// blockIdx.x for the NO_TRANSPOSE fast kernels, needed for the same
// reason: without it, a shape with few output columns (small k, i.e. few
// warps needed) but a huge row count (m in the hundreds of thousands to
// millions, this codebase's own tall-and-thin GEMV benchmark shapes)
// would launch far too few blocks to fill the GPU. x is staged into
// shared memory once per block and reused by every warp in that block
// (all warps in a block share the same row-chunk, each computing a
// different output column), so A is the only operand streamed from
// global memory inside the reduction loop.
//
// Scalar per-lane loads (no float4 vectorization of A), matching
// randls2.0's own simpler gemv_fp16_cm_T_splitM_atomic_safe design for
// its packed-precision sibling rather than the plain-fp32 kernel's own
// alignment-gated float4 path -- a uniform, easy-to-verify-correct
// structure across every format this gets ported to (fp32_16, fp32_bf16
// below use the identical structure, just with a packed decode substituted
// for the plain A[idx] read) mattered more here than an optimization that
// would only apply to this one format; vectorizing A's loads is a natural
// follow-up once real hardware numbers confirm this base design is sound.
//
// MEASURED on real A100 hardware, ROUND 1 (square M=K, unscaled M_BLOCK,
// dispatch still called cudaDeviceSynchronize() before returning): 0.43x
// of cuBLAS at M=K=1024, climbing steadily to 0.92x-1.02x by M=K=12288
// and above -- the ratio tracked gridM (mv_fp32_tiled's own TRANSPOSE
// dispatch, below) going from 1 to 2+ almost exactly.
//
// ROUND 2 (M_BLOCK-scaling + right-sized shmem added, see the dispatch's
// own comment): only a small, consistent gain exactly where predicted --
// e.g. 0.43x->0.45x at M=1024, 0.72x->0.77x at M=3072, unchanged at
// M>=12288 as expected -- confirming grid occupancy was a REAL but MINOR
// factor, not the dominant one. The bulk of the gap was still
// unexplained after this fix.
//
// ROOT CAUSE, ROUND 3, CONFIRMED on real A100 hardware: the dispatch's
// own TRANSPOSE branch was calling cudaDeviceSynchronize() before
// returning (mirroring this function's pre-existing NO_TRANSPOSE/
// fallback trailing sync), which every fp32_16/fp32_bf16 "ready"/
// "atomiconly" dispatcher does NOT do. The benchmark's own time_call
// harness launches `reps` calls back-to-back before syncing ONCE at the
// end via a CUDA event -- an internal per-call sync instead forced the
// host to stall after every single rep, and that fixed sync latency was
// the DOMINANT cost at small/medium sizes, dwarfing the round-2 M_BLOCK
// fix: removing it (see the dispatch's own comment) gained 0.729x-0.043x
// = +63% relative at M=K=1024 (0.447x->0.729x), +55% at M=1536
// (0.556x->0.860x), +44% at M=2048 (0.615x->0.886x), +16% at M=3072,
// +10% at M=4096, tapering to ~0% by M=12288+ (kernel execution time
// already dwarfs the fixed per-call sync latency there) -- exactly the
// signature of a fixed per-call overhead, confirming the diagnosis.
// FINAL state, square M=K, real hardware: 0.73x-0.89x of cuBLAS at
// M=1024-8192 (a reasonable place for a latency-bound small GEMV to sit
// against a vendor kernel), climbing to 0.92x-0.98x from M=12288-32768,
// beating cuBLAS at M=49152 (1.021x). The M=65536 datapoint (4.575x) is
// NOT a TOMOS win -- cuBLAS's own Sgemv collapses there (700->141
// GFlop/s, a cuBLAS-side anomaly, not investigated further here).
//
// This kernel's own body is unchanged across all three rounds -- every
// fix was host-side launch config or synchronization policy, not the
// algorithm. Correctness is checked (verify_mv_fp32_16_gemv.cu,
// "fp32_tiled" TRANSPOSE case).
__global__ void mv_fp32_transpose_atomiconly_scaled_mblock_kernel(const float* __restrict__ A,
                                                                    const float* __restrict__ x,
                                                                    float* __restrict__ y,
                                                                    int m, int k, int M_BLOCK,
                                                                    float alpha) {
    const int lane = threadIdx.x;
    const int warpId = threadIdx.y;
    const int warpsPerBlock = blockDim.y;

    const int col = blockIdx.y * warpsPerBlock + warpId;
    if (col >= k) return;

    const int mStart = blockIdx.x * M_BLOCK;
    const int mEnd = min(m, mStart + M_BLOCK);
    if (mStart >= mEnd) return;

    extern __shared__ float sx[];
    const int tid = lane + 32 * warpId;
    const int nt = 32 * warpsPerBlock;
    for (int r = mStart + tid; r < mEnd; r += nt) {
        sx[r - mStart] = __ldg(&x[r]);
    }
    __syncthreads();

    const size_t ld = static_cast<size_t>(m);
    const size_t colBase = static_cast<size_t>(col) * ld;

    float accum = 0.f;
    for (int r = mStart + lane; r < mEnd; r += 32) {
        accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
    }

    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        accum += __shfl_down_sync(mask, accum, off);

    if (lane == 0) atomicAdd(&y[col], alpha * accum);
}

// EXPERIMENTAL sibling of mv_fp32_transpose_atomiconly_scaled_mblock_
// kernel above -- NOT wired into mv_fp32_tiled's own dispatch, kept as a
// separate, documented, standalone kernel per this session's own
// standing "add new kernels, don't modify existing ones" convention.
// Byte-for-byte the same warp-per-output-column/split-M/atomicAdd
// structure and shared-memory x-staging as its sibling, except the main
// accumulation loop reads A (and the shared x tile) via float4 instead
// of one float at a time per lane per step -- the vectorized-load
// optimization that kernel's own comment flagged as "a natural follow-up
// once real hardware numbers confirm this base design is sound" (now
// confirmed: the unvectorized sibling reached 0.73x-1.02x of cuBLAS on
// real A100 hardware after its own fixes, see its own comment).
//
// Ported from randls2.0's own sgemv_cm_transpose_splitM_atomic_xshared
// (src/kernels/cuda/matrix/dense_kernels.cu, that sibling project -- same
// author, no separate license/attribution needed) MORE completely than
// its unvectorized sibling here did: this now also carries over that
// kernel's own runtime 16-byte alignment check (`aligned16`) before
// committing to the vectorized path, rather than assuming index-based
// alignment. That check matters here specifically because this kernel's
// M_BLOCK (unlike its unvectorized sibling, which is dispatched with an
// M_BLOCK that's also runtime-scaled) can leave mStart, and therefore
// every row-relative shared-memory offset, at an arbitrary residue mod 4
// -- randls2.0's own runtime pointer check (not a compile-time/index
// assumption) is the only way to safely commit to float4 loads under
// that variability. Scalar prefix/suffix loops handle whatever the
// aligned range doesn't cover (a partial first/last group of rows, or
// the whole range when 16-byte alignment doesn't hold) -- always
// correct, the vectorized fast path is a pure throughput bonus when hit.
//
// MEASURED on real A100 hardware (square M=K), vs its own unvectorized
// sibling on the SAME data: a real but MODEST win, not a game-changer --
// +3%-+6% from M=3072 upward (e.g. 1.058x at M=4096, 1.042x at M=8192,
// 1.043x at M=16384, 1.040x at M=65536), but a genuine DIP at M=2048
// (0.913x, -8.7%) and a smaller one at M=1536 (0.976x, -2.4%) -- not
// chased further here; likely the extra register pressure (REG:38 vs
// REG:31 for the unvectorized sibling, cuobjdump -res-usage) and the
// alignment-check/prefix-suffix overhead costing more than the
// vectorized body saves when M_BLOCK (after auto-scaling, see the host
// wrapper's own comment) is itself small enough that the vectorized
// loop barely runs. Against cuBLAS: reaches parity or slightly beats it
// from M=24576 upward (1.01x-1.05x, vs the unvectorized sibling's own
// 0.93x-1.02x there) -- native fp32 GEMV TRANSPOSE is now genuinely
// competitive with vendor cuBLAS across nearly the whole size range
// tested, not just at the largest sizes.
__global__ void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_kernel(const float* __restrict__ A,
                                                                        const float* __restrict__ x,
                                                                        float* __restrict__ y,
                                                                        int m, int k, int M_BLOCK,
                                                                        float alpha) {
    const int lane = threadIdx.x;
    const int warpId = threadIdx.y;
    const int warpsPerBlock = blockDim.y;

    const int col = blockIdx.y * warpsPerBlock + warpId;
    if (col >= k) return;

    const int mStart = blockIdx.x * M_BLOCK;
    const int mEnd = min(m, mStart + M_BLOCK);
    if (mStart >= mEnd) return;

    extern __shared__ float sx[];
    const int tid = lane + 32 * warpId;
    const int nt = 32 * warpsPerBlock;
    for (int r = mStart + tid; r < mEnd; r += nt) {
        sx[r - mStart] = __ldg(&x[r]);
    }
    __syncthreads();

    const size_t ld = static_cast<size_t>(m);
    const size_t colBase = static_cast<size_t>(col) * ld;

    float accum = 0.f;

    // Vectorizable range [vBeg, vEnd), rounded to multiples of 4 (row
    // indices) -- matches randls2.0's own boundary technique. Whether
    // A[colBase+vBeg] and sx[vBeg-mStart] actually land on a 16-byte
    // boundary is checked at runtime below, not assumed from this
    // rounding alone (see this kernel's own header comment for why).
    const int vBeg = (mStart + 3) & ~3;
    const int vEnd = mEnd & ~3;

    // Scalar prefix: [mStart, min(vBeg, mEnd))
    for (int r = mStart + lane; r < min(vBeg, mEnd); r += 32) {
        accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
    }

    if (vBeg < vEnd) {
        const uintptr_t pA = reinterpret_cast<uintptr_t>(&A[colBase + static_cast<size_t>(vBeg)]);
        const uintptr_t pX = reinterpret_cast<uintptr_t>(&sx[vBeg - mStart]);
        const bool aligned16 = ((pA & 0xF) == 0) && ((pX & 0xF) == 0);

        if (aligned16) {
            for (int r = vBeg + lane * 4; r + 3 < vEnd; r += 32 * 4) {
                const float4 av = *reinterpret_cast<const float4*>(&A[colBase + static_cast<size_t>(r)]);
                const float4 xv = *reinterpret_cast<const float4*>(&sx[r - mStart]);
                accum = fmaf(av.x, xv.x, accum);
                accum = fmaf(av.y, xv.y, accum);
                accum = fmaf(av.z, xv.z, accum);
                accum = fmaf(av.w, xv.w, accum);
            }
        } else {
            // Unaligned fallback -- still correct, just not vectorized.
            for (int r = vBeg + lane; r < vEnd; r += 32) {
                accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
            }
        }
    }

    // Scalar suffix: [vEnd, mEnd)
    for (int r = vEnd + lane; r < mEnd; r += 32) {
        accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
    }

    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        accum += __shfl_down_sync(mask, accum, off);

    if (lane == 0) atomicAdd(&y[col], alpha * accum);
}

// Host wrapper for mv_fp32_transpose_atomiconly_scaled_mblock_vec4_kernel
// -- standalone, NOT called from mv_fp32_tiled (which keeps dispatching
// to its own unvectorized mv_fp32_transpose_atomiconly_scaled_mblock_
// kernel, unchanged). Same M_BLOCK auto-scaling and right-sized-shmem
// logic as that dispatcher's own TRANSPOSE branch (see its own comment,
// above mv_fp32_tiled's definition, for the full derivation) -- reused
// unchanged here since it's a proven, real occupancy win independent of
// the load-vectorization this kernel adds on top. Only handles the
// TRANSPOSE + ld_a==size[0] fast-path shape -- no fallback to a generic
// dispatch for other layouts, since this is a standalone A/B kernel, not
// a general-purpose entry point.
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size, const float* A,
                                                       size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                                       size_t ld_y) {
    if (trans != TransposeType::TRANSPOSE || ld_a != size[0]) return;

    int m = static_cast<int>(size[0]);
    int k = static_cast<int>(size[1]);
    size_t rows_op = size[1];

    if (beta == 0.0) {
        cudaMemsetAsync(y, 0, rows_op * sizeof(float));
    } else if (beta != 1.0) {
        int threads = kMvFp32TileSize;
        int blocks = (rows_op + threads - 1) / threads;
        mv_fp32_scale_y_kernel<<<blocks, threads>>>(rows_op, beta, y);
    }

    int gridK = (k + kMvFp32TransposeWarpsPerBlock - 1) / kMvFp32TransposeWarpsPerBlock;
    int M_BLOCK = kMvFp32TransposeMBlock;
    int target_gridM = (kMvFp32TransposeTargetGridBlocks + gridK - 1) / gridK;
    if (target_gridM > 1) {
        int scaled_block = (m + target_gridM - 1) / target_gridM;
        if (scaled_block < M_BLOCK) M_BLOCK = std::max(scaled_block, 1);
    }

    dim3 block(32, kMvFp32TransposeWarpsPerBlock);
    int gridM = (m + M_BLOCK - 1) / M_BLOCK;
    dim3 grid(gridM, gridK);
    size_t shmem = static_cast<size_t>(std::min(M_BLOCK, m)) * sizeof(float);

    mv_fp32_transpose_atomiconly_scaled_mblock_vec4_kernel<<<grid, block, shmem>>>(
        A, x, y, m, k, M_BLOCK, static_cast<float>(alpha));
}

// EXPERIMENTAL sibling of mv_fp32_transpose_atomiconly_scaled_mblock_
// vec4_kernel above -- NOT wired into mv_fp32_tiled's own dispatch, kept
// as a separate, documented, standalone kernel (same "add new kernels,
// don't modify existing ones" convention this whole TRANSPOSE kernel
// family has followed). Byte-for-byte the same structure as its vec4
// sibling, EXCEPT the x-staging loop (global x -> shared sx, done once
// per block before the main accumulation loop) is ALSO float4-vectorized
// now, with the same runtime-alignment-check-then-scalar-fallback
// technique the main loop already uses -- vec4 only vectorized the main
// loop's own A/sx reads, leaving this one-float-at-a-time.
//
// Motivation: on real A100 hardware, vec4's own gains were concentrated
// at M=3072 and up (+3%-+6%), with a dip at M=1536-2048 (see vec4's own
// comment) -- exactly the smaller shapes where the staging loop's own
// fixed cost (independent of M_BLOCK's vectorization) is proportionally
// largest relative to the main loop's own, now-vectorized, work. This
// targets that specific gap.
//
// MEASURED on real A100 hardware (square M=K), vs its vec4 sibling on the
// SAME data: a clean, consistent win -- beats vec4 at every size tested
// except one small dip at M=1536 (0.971x, -2.9%), with the biggest gains
// exactly where vec4 itself was weakest -- +11.6% at M=8192 (1.116x),
// +8.5% at M=2048 (1.085x), +8.4% at M=4096 (1.084x), settling to a
// steady +3%-+6% from M=12288 upward. Confirms the staging-loop
// hypothesis directly: vectorizing it fixed the dip vec4 showed at
// exactly the shapes where the (previously scalar) staging cost was
// proportionally largest. Against cuBLAS: reaches parity or wins from
// M=12288 upward (0.999x-1.081x, vs vec4's own 0.95x-1.05x there) --
// e.g. 1.017x at M=16384, 1.051x at M=24576, 1.081x at M=49152. This is
// now the best-measured native fp32 TRANSPOSE kernel in this file.
__global__ void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec_kernel(
    const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y,
    int m, int k, int M_BLOCK, float alpha) {
    const int lane = threadIdx.x;
    const int warpId = threadIdx.y;
    const int warpsPerBlock = blockDim.y;

    const int col = blockIdx.y * warpsPerBlock + warpId;
    if (col >= k) return;

    const int mStart = blockIdx.x * M_BLOCK;
    const int mEnd = min(m, mStart + M_BLOCK);
    if (mStart >= mEnd) return;

    extern __shared__ float sx[];
    const int tid = lane + 32 * warpId;
    const int nt = 32 * warpsPerBlock;

    // Vectorized staging: same rounded-boundary + runtime-alignment-check
    // technique as the main accumulation loop below, applied to the
    // global x -> shared sx copy instead of the A/sx reduction reads.
    const int sBeg = (mStart + 3) & ~3;
    const int sEnd = mEnd & ~3;

    for (int r = mStart + tid; r < min(sBeg, mEnd); r += nt) {
        sx[r - mStart] = __ldg(&x[r]);
    }
    if (sBeg < sEnd) {
        const uintptr_t pXg = reinterpret_cast<uintptr_t>(&x[sBeg]);
        const uintptr_t pXs = reinterpret_cast<uintptr_t>(&sx[sBeg - mStart]);
        const bool stageAligned16 = ((pXg & 0xF) == 0) && ((pXs & 0xF) == 0);

        if (stageAligned16) {
            for (int r = sBeg + tid * 4; r + 3 < sEnd; r += nt * 4) {
                const float4 xv = *reinterpret_cast<const float4*>(&x[r]);
                *reinterpret_cast<float4*>(&sx[r - mStart]) = xv;
            }
        } else {
            for (int r = sBeg + tid; r < sEnd; r += nt) {
                sx[r - mStart] = __ldg(&x[r]);
            }
        }
    }
    for (int r = sEnd + tid; r < mEnd; r += nt) {
        sx[r - mStart] = __ldg(&x[r]);
    }
    __syncthreads();

    const size_t ld = static_cast<size_t>(m);
    const size_t colBase = static_cast<size_t>(col) * ld;

    float accum = 0.f;

    const int vBeg = (mStart + 3) & ~3;
    const int vEnd = mEnd & ~3;

    for (int r = mStart + lane; r < min(vBeg, mEnd); r += 32) {
        accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
    }

    if (vBeg < vEnd) {
        const uintptr_t pA = reinterpret_cast<uintptr_t>(&A[colBase + static_cast<size_t>(vBeg)]);
        const uintptr_t pX = reinterpret_cast<uintptr_t>(&sx[vBeg - mStart]);
        const bool aligned16 = ((pA & 0xF) == 0) && ((pX & 0xF) == 0);

        if (aligned16) {
            for (int r = vBeg + lane * 4; r + 3 < vEnd; r += 32 * 4) {
                const float4 av = *reinterpret_cast<const float4*>(&A[colBase + static_cast<size_t>(r)]);
                const float4 xv = *reinterpret_cast<const float4*>(&sx[r - mStart]);
                accum = fmaf(av.x, xv.x, accum);
                accum = fmaf(av.y, xv.y, accum);
                accum = fmaf(av.z, xv.z, accum);
                accum = fmaf(av.w, xv.w, accum);
            }
        } else {
            for (int r = vBeg + lane; r < vEnd; r += 32) {
                accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
            }
        }
    }

    for (int r = vEnd + lane; r < mEnd; r += 32) {
        accum = fmaf(__ldg(&A[colBase + static_cast<size_t>(r)]), sx[r - mStart], accum);
    }

    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        accum += __shfl_down_sync(mask, accum, off);

    if (lane == 0) atomicAdd(&y[col], alpha * accum);
}

// Host wrapper for mv_fp32_transpose_atomiconly_scaled_mblock_vec4_
// stagevec_kernel -- standalone, same M_BLOCK auto-scaling/shmem-sizing
// logic as its vec4 sibling's own host wrapper (unchanged, reused).
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y) {
    if (trans != TransposeType::TRANSPOSE || ld_a != size[0]) return;

    int m = static_cast<int>(size[0]);
    int k = static_cast<int>(size[1]);
    size_t rows_op = size[1];

    if (beta == 0.0) {
        cudaMemsetAsync(y, 0, rows_op * sizeof(float));
    } else if (beta != 1.0) {
        int threads = kMvFp32TileSize;
        int blocks = (rows_op + threads - 1) / threads;
        mv_fp32_scale_y_kernel<<<blocks, threads>>>(rows_op, beta, y);
    }

    int gridK = (k + kMvFp32TransposeWarpsPerBlock - 1) / kMvFp32TransposeWarpsPerBlock;
    int M_BLOCK = kMvFp32TransposeMBlock;
    int target_gridM = (kMvFp32TransposeTargetGridBlocks + gridK - 1) / gridK;
    if (target_gridM > 1) {
        int scaled_block = (m + target_gridM - 1) / target_gridM;
        if (scaled_block < M_BLOCK) M_BLOCK = std::max(scaled_block, 1);
    }

    dim3 block(32, kMvFp32TransposeWarpsPerBlock);
    int gridM = (m + M_BLOCK - 1) / M_BLOCK;
    dim3 grid(gridM, gridK);
    size_t shmem = static_cast<size_t>(std::min(M_BLOCK, m)) * sizeof(float);

    mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec_kernel<<<grid, block, shmem>>>(
        A, x, y, m, k, M_BLOCK, static_cast<float>(alpha));
}

// Host dispatcher -- same 3-way shape dispatch as mv_fp32_16_accurate,
// same thresholds. ld_x/ld_y are accepted for signature symmetry with
// mv_strided but unused, matching that family's own established GEMV
// convention (x/y are always treated as unit-stride vectors).
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];

    // TRANSPOSE fast path -- mv_fp32_transpose_atomiconly_scaled_mblock_
    // kernel above, see its own comment for the full derivation. Requires
    // ld_a == size[0] the same way every NO_TRANSPOSE row-vectorized fast
    // kernel in this codebase requires ld_a == size[0] (the packed/
    // coalesced access pattern needs A's stored rows contiguous) -- any
    // other layout falls through to the generic dispatch below, same as
    // it always did.
    if (trans == TransposeType::TRANSPOSE && ld_a == size[0]) {
        int m = static_cast<int>(size[0]);
        int k = static_cast<int>(size[1]);

        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, rows_op * sizeof(float));
        } else if (beta != 1.0) {
            int threads = kMvFp32TileSize;
            int blocks = (rows_op + threads - 1) / threads;
            mv_fp32_scale_y_kernel<<<blocks, threads>>>(rows_op, beta, y);
        }

        // M_BLOCK tuning, added after the first real-hardware measurement
        // of this TRANSPOSE fast path: ratio vs cuBLAS climbed steadily
        // with size (0.43x at M=K=1024 up to 0.92x-1.02x by M=K=12288+)
        // and the inflection lined up exactly with gridM (below) going
        // from 1 to 2+ -- classic GPU-underoccupancy signature, not a
        // per-element decode cost issue (there's no decode here at all).
        // With the fixed kMvFp32TransposeMBlock alone, gridM stays stuck
        // at 1 for every m below it (8192), so total blocks = gridK alone
        // (k / kMvFp32TransposeWarpsPerBlock) -- e.g. only 128 blocks at
        // K=1024, too few to fill an A100's 108 SMs well when each
        // block's warps have to sync on the shared-memory x load before
        // their own reduction loop even starts.
        //
        // Mirrors kMvReadyTargetKSplits' own K_BLOCK-scaling idea
        // (fp32_16_cuda_kernels.cu) but targets TOTAL grid size, not just
        // one dimension, since TRANSPOSE splits parallelism across BOTH
        // gridM (via M_BLOCK, tunable) and gridK (via k /
        // kMvFp32TransposeWarpsPerBlock, fixed by k): shrink M_BLOCK,
        // growing gridM, only when the natural grid (gridM=1 * gridK)
        // falls short of kMvFp32TransposeTargetGridBlocks -- once gridK
        // alone already clears that target (large k), M_BLOCK is left at
        // its full 8192, identical to the unscaled behavior, so already-
        // well-performing large shapes are unaffected. UNTESTED: this
        // specific scaling has not itself been measured yet -- only the
        // unscaled kernel's own data above motivated it; re-run the same
        // benchmark to confirm before trusting the new ratios.
        int gridK = (k + kMvFp32TransposeWarpsPerBlock - 1) / kMvFp32TransposeWarpsPerBlock;
        int M_BLOCK = kMvFp32TransposeMBlock;
        int target_gridM = (kMvFp32TransposeTargetGridBlocks + gridK - 1) / gridK;
        if (target_gridM > 1) {
            int scaled_block = (m + target_gridM - 1) / target_gridM;
            if (scaled_block < M_BLOCK) M_BLOCK = std::max(scaled_block, 1);
        }

        dim3 block(32, kMvFp32TransposeWarpsPerBlock);
        int gridM = (m + M_BLOCK - 1) / M_BLOCK;
        dim3 grid(gridM, gridK);
        // Sized to the ACTUAL per-block row count (min(M_BLOCK, m), the
        // largest chunk any block ever stages), not blindly M_BLOCK --
        // fixes a real over-allocation: previously every launch reserved
        // the full 32KB (kMvFp32TransposeMBlock*4 bytes) even when m was
        // far smaller (e.g. 4KB actually used at m=1024, 28KB wasted),
        // needlessly capping how many blocks could be resident per SM.
        // Always safe -- never larger than before, only ever smaller or
        // equal.
        size_t shmem = static_cast<size_t>(std::min(M_BLOCK, m)) * sizeof(float);

        mv_fp32_transpose_atomiconly_scaled_mblock_kernel<<<grid, block, shmem>>>(
            A, x, y, m, k, M_BLOCK, static_cast<float>(alpha));
        // Deliberately NOT cudaDeviceSynchronize()'d here, unlike the
        // NO_TRANSPOSE/fallback paths' own trailing sync below (pre-
        // existing, left unchanged) -- nothing in this branch (no
        // cudaMalloc/cudaFree) needs the GPU work done before returning,
        // and every fp32_16/fp32_bf16 "ready"/"atomiconly" dispatcher in
        // fp32_16_cuda_kernels.cu already follows this same async
        // convention (launch, return, let the caller's own timing/
        // cudaMemcpy do the synchronizing). An earlier version of this
        // branch DID call cudaDeviceSynchronize() here (mirroring the
        // rest of this function), which MEASURABLY hurt small-shape
        // throughput on real A100 hardware: benchmarks/kernels/cuda/
        // matrix/time_call's own timed loop launches `reps` calls back-
        // to-back before syncing ONCE at the end via a CUDA event -- an
        // internal sync on every rep forced the host to stall after each
        // one instead of letting the GPU pipeline them. CONFIRMED as the
        // dominant cost, not a minor one: removing it gained +63% at
        // M=K=1024 (0.447x->0.729x of cuBLAS), +55% at M=1536, +44% at
        // M=2048, tapering to ~0% by M=12288+ (kernel execution time
        // already dwarfs the fixed per-call sync latency there) -- see
        // mv_fp32_transpose_atomiconly_scaled_mblock_kernel's own comment
        // above for the full round-by-round numbers.
        return;
    }

    if (rows_op < kMvFp32WideRowThreshold) {
        int col_splits = choose_col_splits_fp32(rows_op, cols_op);
        int row_blocks = (rows_op + kMvFp32TileSize - 1) / kMvFp32TileSize;
        double* partial = nullptr;
        cudaError_t err = cudaMalloc(&partial, rows_op * col_splits * sizeof(double));
        if (err != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        dim3 grid(row_blocks, col_splits);
        mv_fp32_wide_kernel<<<grid, kMvFp32TileSize>>>(trans, size[0], size[1], A, ld_a, x,
                                                        col_splits, partial);
        int reduce_blocks = (rows_op + kMvFp32TileSize - 1) / kMvFp32TileSize;
        mv_fp32_wide_reduce_kernel<<<reduce_blocks, kMvFp32TileSize>>>(alpha, beta, rows_op,
                                                                        col_splits, partial, y);
        cudaFree(partial);
    } else if (cols_op < kMvFp32NarrowVecColThreshold) {
        int threads = kMvFp32TileSize;
        int blocks = (rows_op + threads - 1) / threads;
        mv_fp32_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x,
                                                   beta, y);
    } else {
        int threads = kMvFp32TileSize;
        size_t row_groups = (rows_op + kMvFp32RowsPerThread - 1) / kMvFp32RowsPerThread;
        int blocks = static_cast<int>((row_groups + threads - 1) / threads);
        mv_fp32_narrow_vec_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a,
                                                        x, beta, y);
    }
    cudaDeviceSynchronize();
}

// TOMOS FP16 (native storage): plain native __half A -- NOT the
// NumericType::FP32_16 packed wh/wr format fp32_16_cuda_kernels.cu's
// "pure_half" family decodes from, so there's no decode step at all here,
// every A[idx] read is already the real fp16 value stored at rest. x/y
// stay float (x is staged into shared memory and rounded to __half once
// per tile, matching every other kernel in this codebase's "pure_half"/
// "native" GEMV families); the accumulation itself uses genuine scalar
// __half arithmetic (real fp16 hardware add/multiply, not double or
// float), one term at a time.
//
// Body mechanically ported from fp32_16_cuda_kernels.cu's own
// mv_fp32_16_pure_half_tiled_kernel -- identical control flow, only the
// fp32_16_half_only(wh[idx]) decode calls became plain A[idx] reads
// (nothing to decode). Deliberately does NOT replicate that family's
// full wide-row/narrow-vec/atomic-combine multi-kernel shape dispatch
// (each of those needs its own file-local tuning constants -- not shared
// across kernel files in this codebase -- and this session had no GPU
// access to validate a hand-ported multi-kernel tree): this single kernel
// handles every shape and both TRANSPOSE directions uniformly via A's own
// trans-dependent row/col mapping, at the cost of not chasing per-shape
// peak tuning. The `if (i < rows_op)` guard here is around the
// accumulation work only (not a return before the loop's own
// __syncthreads() calls), matching the barrier-safety pattern this
// session established for the packed-format kernels -- confirmed safe by
// direct comparison against mv_fp32_16_pure_half_tiled_kernel's own
// identical structure.
//
// CORRECTNESS FIX (found via a real LSQR run on this codebase's own
// TRANSPOSE direction, M=1,000,000 K=1,000 matrix): the FIRST version of
// this kernel accumulated the ENTIRE reduction length into a single
// persistent __half register across every tile -- fine for NO_TRANSPOSE
// here (cols_op = K, e.g. 1000), catastrophic for TRANSPOSE (cols_op = M,
// e.g. 1,000,000): __half's ~10-bit mantissa means that once the running
// sum's magnitude grows large enough, each further term's contribution
// rounds away to nothing (classic float-accumulation stagnation) --
// produces a finite but badly wrong result, not a crash or NaN, which is
// exactly what surfaced as LSQR's `mv_fp16_native`-driven solve
// diverging (residual/error orders of magnitude off) despite compiling
// and linking cleanly. mv_fp32_16_pure_half_tiled_kernel itself never hit
// this in its OWN original role: production dispatch only ever routes it
// to cols_op < kMvNarrowVecColThreshold (64) as a narrow-column fallback,
// never anywhere near a million-term reduction -- an assumption this
// kernel silently broke by reusing that body for BOTH directions at
// unbounded size. Fixed by widening to a double running total once per
// tile (kMvFp16NativeTileSize=256 terms) instead of once per whole
// reduction -- every individual multiply-accumulate is still genuine
// __half arithmetic (real fp16 rounding at every FMA within a tile), only
// the CROSS-tile combination widens, bounding the half-precision
// accumulation run-length to 256 terms regardless of the matrix's own
// M/K size.
//
// Registered at the (CUDA, FP16, FP32, FP32, compute=FP16) slot in
// dense_registries.cpp, overriding the generic double-accumulating
// gemv_generic_kernel fallback that slot would otherwise get -- see that
// registration's own comment for why plain --mv-type fp16 (paired with
// float x/y, no --implicit-duplicates) needed this.
static constexpr int kMvFp16NativeTileSize = 256;

__global__ void mv_fp16_native_tiled_kernel(TransposeType trans, double alpha, size_t m, size_t n,
                                            const __half* A, size_t ld_a, const float* x, double beta,
                                            float* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __half x_tile[kMvFp16NativeTileSize];
    double acc_wide = 0.0;
    for (size_t tile_start = 0; tile_start < cols_op; tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize), cols_op - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = static_cast<__half>(x[tile_start + threadIdx.x]);
        __syncthreads();

        if (i < rows_op) {
            __half tile_acc = static_cast<__half>(0.0f);
            for (size_t jj = 0; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                tile_acc = tile_acc + A[idx] * x_tile[jj];
            }
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
        }
        __syncthreads();
    }

    if (i >= rows_op) return;
    y[i] = (beta != 0.0)
              ? static_cast<float>(alpha * acc_wide + beta * static_cast<double>(y[i]))
              : static_cast<float>(alpha * acc_wide);
}

void mv_fp16_native(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                    const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    int threads = kMvFp16NativeTileSize;
    int blocks = static_cast<int>((rows_op + threads - 1) / threads);
    mv_fp16_native_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x, beta, y);
    cudaDeviceSynchronize();
}

// mv_fp16_native_storage: fully native fp16 GEMV -- A, x, AND y all
// __half (not just A, unlike mv_fp16_native above, which keeps x/y at
// float for LSQR's --explicit-duplicates mixed-precision solve and is
// UNCHANGED here). This variant exists for direct performance comparison
// against cuBLAS FP16 (Hgemm, n=1), which is ALSO genuinely all-half
// end-to-end in these benchmarks -- so both sides of that comparison use
// the same operand types, not a mixed-vs-uniform mismatch.
//
// Same bounded-tile + double-widened-across-tiles accumulation as
// mv_fp16_native_tiled_kernel above (see that kernel's own comment for
// why this matters -- the earlier all-in-one-__half-register version was
// found, via a real LSQR run, to diverge catastrophically on TRANSPOSE's
// million-term reduction). Only x_tile's fill and the final y write
// change from float to __half; the accumulation-safety fix carries over
// unchanged since it was never about x/y's own type, only the reduction
// length's own precision.
__global__ void mv_fp16_native_storage_tiled_kernel(TransposeType trans, double alpha, size_t m, size_t n,
                                                     const __half* A, size_t ld_a, const __half* x, double beta,
                                                     __half* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __half x_tile[kMvFp16NativeTileSize];
    double acc_wide = 0.0;
    for (size_t tile_start = 0; tile_start < cols_op; tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize), cols_op - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        __syncthreads();

        if (i < rows_op) {
            __half tile_acc = static_cast<__half>(0.0f);
            for (size_t jj = 0; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                tile_acc = tile_acc + A[idx] * x_tile[jj];
            }
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
        }
        __syncthreads();
    }

    if (i >= rows_op) return;
    double result = (beta != 0.0) ? (alpha * acc_wide + beta * static_cast<double>(static_cast<float>(y[i])))
                                   : (alpha * acc_wide);
    y[i] = static_cast<__half>(static_cast<float>(result));
}

// Column-split, half2-packed, atomic-combine sibling of mv_fp16_native_
// storage_tiled_kernel above -- mirrors this file's own mv_fp32_wide_
// kernel/choose_col_splits_fp32 col-splitting convention (see that kernel
// and its host wrapper above) to fix a severe underutilization the plain
// per-row kernel has at small/medium M: that kernel launches exactly
// ceil(rows_op/kMvFp16NativeTileSize) blocks REGARDLESS of cols_op, so
// e.g. at M=K=4096 only 16 blocks run on a 108-SM A100 no matter how large
// K is -- measured 62.9 GFlop/s vs mv_fp32_16_pure_half_wide_ilp_shared_
// half2_atomic_best's 530.9 (8.4x slower) in the GEMV performance
// benchmark. This kernel splits the column/reduction range across
// col_splits blocks per row-block (same grid shape as mv_fp32_wide_
// kernel), each block combining its own partial sum into y via
// atomicAdd(__half*, __half) (a native hardware built-in on compute
// capability 7.0+, the same choice already made by mv_fp32_16_pure_half_
// transpose_atomiconly_scaled_mblock_vec4_half_io_kernel in fp32_16_cuda_
// kernels.cu for the same all-half-I/O benchmark context) -- no partial-
// buffer/malloc/reduce-kernel needed, directly multiplying the number of
// blocks launched by col_splits independent of M.
//
// __half2/__hfma2-packed accumulation (2 real FMAs per instruction) --
// mechanically ported from mv_fp32_16_pure_half_wide_ilp_shared_half2_
// atomic_kernel's own accumulation body (fp32_16_cuda_kernels.cu, every
// fp32_16_half_only(wh[idx]) decode call became a direct A[idx] read,
// nothing else changed), replacing this kernel's own original SCALAR
// __half accumulation (one FMA at a time) -- confirmed via direct
// comparison against "TOMOS FP16 (implicit, fast)" (which IS half2-
// packed) to be a real, standing ~2x compute-throughput handicap at large
// M, on top of the col_splits/occupancy gap already fixed above (see
// mv_fp16_native_storage_atomic's own comment). No double-widened
// accumulation here (unlike this kernel's own prior scalar version, or
// mv_fp16_native_storage_tiled_kernel above): acc01/acc23 persist across
// the WHOLE column chunk, same as the reference kernel this was ported
// from, which has run in production (LSQR's --implicit-duplicates, via
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best) without ever
// surfacing the catastrophic-stagnation failure mode mv_fp16_native's own
// naive, col-split-less, million-term design hit -- col-split bounding
// keeps any one block's own chunk far short of that. mv_fp16_native_
// storage is benchmark-only (never registered in the dispatch registry),
// so there is no LSQR-correctness exposure either way.
// Tile-boundary double-widened accumulation (acc_wide, widened once per
// kMvFp16NativeTileSize=256-element tile) -- added for the same reason as
// mv_fp16_native_mixed_atomic_kernel's own identical fix (see that
// kernel's comment): this body has never actually been measured or tuned
// for TRANSPOSE (every optimization pass this session, col_splits/half2/
// cp.async, was explicitly NO_TRANSPOSE-only), so a long, unwidened
// __half2 accumulation chunk for that direction was never verified safe,
// and the reference kernel this was ported from never accumulates that
// long in any proven production TRANSPOSE path either (LSQR's own
// packed-storage dispatcher always uses a completely separate, dedicated
// TRANSPOSE kernel instead).
__global__ void mv_fp16_native_storage_atomic_kernel(TransposeType trans, size_t m, size_t n,
                                                      const __half* A, size_t ld_a,
                                                      const __half* x, size_t col_splits,
                                                      double alpha, __half* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t chunk = (cols_op + col_splits - 1) / col_splits;
    size_t col_start = blockIdx.y * chunk;
    size_t col_end = min(col_start + chunk, cols_op);

    __shared__ __align__(4) __half x_tile[kMvFp16NativeTileSize];
    __half2 acc01 = __half2half2(static_cast<__half>(0.0f));
    __half2 acc23 = __half2half2(static_cast<__half>(0.0f));
    double acc_wide = 0.0;
    for (size_t tile_start = col_start; tile_start < col_end; tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize), col_end - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        __syncthreads();

        if (i < rows_op) {
            size_t jj = 0;
            for (; jj + 4 <= tile_len; jj += 4) {
                size_t j0 = tile_start + jj, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
                size_t r0 = (trans == TransposeType::NO_TRANSPOSE) ? i : j0;
                size_t c0 = (trans == TransposeType::NO_TRANSPOSE) ? j0 : i;
                size_t r1 = (trans == TransposeType::NO_TRANSPOSE) ? i : j1;
                size_t c1 = (trans == TransposeType::NO_TRANSPOSE) ? j1 : i;
                size_t r2 = (trans == TransposeType::NO_TRANSPOSE) ? i : j2;
                size_t c2 = (trans == TransposeType::NO_TRANSPOSE) ? j2 : i;
                size_t r3 = (trans == TransposeType::NO_TRANSPOSE) ? i : j3;
                size_t c3 = (trans == TransposeType::NO_TRANSPOSE) ? j3 : i;
                size_t idx0 = r0 + c0 * ld_a, idx1 = r1 + c1 * ld_a;
                size_t idx2 = r2 + c2 * ld_a, idx3 = r3 + c3 * ld_a;
                __half a0 = A[idx0];
                __half a1 = A[idx1];
                __half a2 = A[idx2];
                __half a3 = A[idx3];
                __half2 a01 = __halves2half2(a0, a1);
                __half2 a23 = __halves2half2(a2, a3);
                __half2 x01 = *reinterpret_cast<__half2*>(&x_tile[jj]);
                __half2 x23 = *reinterpret_cast<__half2*>(&x_tile[jj + 2]);
                acc01 = __hfma2(a01, x01, acc01);
                acc23 = __hfma2(a23, x23, acc23);
            }
            for (; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                __half a_val = A[idx];
                __half2 a_pad = __halves2half2(a_val, static_cast<__half>(0.0f));
                __half2 x_pad = __halves2half2(x_tile[jj], static_cast<__half>(0.0f));
                acc01 = __hfma2(a_pad, x_pad, acc01);
            }
            __half2 acc_sum2 = __hadd2(acc01, acc23);
            __half tile_acc = __low2half(acc_sum2) + __high2half(acc_sum2);
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
            acc01 = __half2half2(static_cast<__half>(0.0f));
            acc23 = __half2half2(static_cast<__half>(0.0f));
        }
        __syncthreads();
    }

    if (i < rows_op) {
        float contribution = static_cast<float>(alpha) * static_cast<float>(acc_wide);
        atomicAdd(&y[i], static_cast<__half>(contribution));
    }
}

// Prescale step for the atomic-combine kernel above -- atomicAdd doesn't
// apply beta itself, so y must already be beta-scaled before any
// col_splits block's atomicAdd, same convention as mv_fp32_16_wide_atomic_
// prescale_kernel/mv_fp32_16_scale_y_half_kernel (fp32_16_cuda_kernels.cu)
// -- mechanically re-declared here rather than called cross-translation-
// unit, since __global__ kernels aren't shared across .cu files in this
// codebase.
__global__ void mv_fp16_native_storage_atomic_prescale_kernel(size_t rows_op, double beta, __half* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows_op)
        y[i] = __float2half(static_cast<float>(beta) * __half2float(y[i]));
}

// Column-split, __half2/__hfma2-packed, reduce-kernel-combine sibling of
// mv_fp16_native_storage_atomic_kernel above -- mechanically ported from
// fp32_16_cuda_kernels.cu's own mv_fp32_16_pure_half_wide_ilp_shared_
// half2_kernel (the "pure_half_half2" kernel, this whole GEMV
// investigation's historically-fastest FP32_16-family series -- measured
// 78-90% of cuBLAS in CSVs from 2026-08-31, traced via git history after
// a report that "TOMOS FP16" used to match cuBLAS closely). IDENTICAL
// control flow, shared x-tiling, and __hfma2-packed accumulation; the only
// change is every fp32_16_half_only(wh[idx]) decode call became a direct
// A[idx] read (A is already genuine __half here -- nothing to decode).
// Same partial[]-buffer + separate reduce-kernel combine as its source
// kernel (mv_fp16_native_storage_wide_reduce_kernel below), not
// mv_fp16_native_storage_atomic_kernel's atomicAdd(__half*, __half) --
// this is a deliberate mechanical port, not a redesign, to reproduce the
// historically-fast kernel's own combine strategy exactly, not guess at
// an equivalent.
// Same tile-boundary double-widened accumulation fix as mv_fp16_native_
// storage_atomic_kernel above -- see that kernel's own comment for why.
__global__ void mv_fp16_native_storage_wide_half2_kernel(
    TransposeType trans, size_t m, size_t n, const __half* A, size_t ld_a,
    const __half* x, size_t col_splits, double* partial) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t chunk = (cols_op + col_splits - 1) / col_splits;
    size_t col_start = blockIdx.y * chunk;
    size_t col_end = min(col_start + chunk, cols_op);

    __shared__ __align__(4) __half x_tile[kMvFp16NativeTileSize];
    __half2 acc01 = __half2half2(static_cast<__half>(0.0f));
    __half2 acc23 = __half2half2(static_cast<__half>(0.0f));
    double acc_wide = 0.0;
    for (size_t tile_start = col_start; tile_start < col_end;
         tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize),
                              col_end - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        __syncthreads();

        if (i < rows_op) {
            size_t jj = 0;
            for (; jj + 4 <= tile_len; jj += 4) {
                size_t j0 = tile_start + jj, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
                size_t r0 = (trans == TransposeType::NO_TRANSPOSE) ? i : j0;
                size_t c0 = (trans == TransposeType::NO_TRANSPOSE) ? j0 : i;
                size_t r1 = (trans == TransposeType::NO_TRANSPOSE) ? i : j1;
                size_t c1 = (trans == TransposeType::NO_TRANSPOSE) ? j1 : i;
                size_t r2 = (trans == TransposeType::NO_TRANSPOSE) ? i : j2;
                size_t c2 = (trans == TransposeType::NO_TRANSPOSE) ? j2 : i;
                size_t r3 = (trans == TransposeType::NO_TRANSPOSE) ? i : j3;
                size_t c3 = (trans == TransposeType::NO_TRANSPOSE) ? j3 : i;
                size_t idx0 = r0 + c0 * ld_a, idx1 = r1 + c1 * ld_a;
                size_t idx2 = r2 + c2 * ld_a, idx3 = r3 + c3 * ld_a;
                __half a0 = A[idx0];
                __half a1 = A[idx1];
                __half a2 = A[idx2];
                __half a3 = A[idx3];
                __half2 a01 = __halves2half2(a0, a1);
                __half2 a23 = __halves2half2(a2, a3);
                __half2 x01 = *reinterpret_cast<__half2*>(&x_tile[jj]);
                __half2 x23 = *reinterpret_cast<__half2*>(&x_tile[jj + 2]);
                acc01 = __hfma2(a01, x01, acc01);
                acc23 = __hfma2(a23, x23, acc23);
            }
            for (; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                __half a_val = A[idx];
                __half2 a_pad = __halves2half2(a_val, static_cast<__half>(0.0f));
                __half2 x_pad =
                    __halves2half2(x_tile[jj], static_cast<__half>(0.0f));
                acc01 = __hfma2(a_pad, x_pad, acc01);
            }
            __half2 acc_sum2 = __hadd2(acc01, acc23);
            __half tile_acc = __low2half(acc_sum2) + __high2half(acc_sum2);
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
            acc01 = __half2half2(static_cast<__half>(0.0f));
            acc23 = __half2half2(static_cast<__half>(0.0f));
        }
        __syncthreads();
    }

    if (i < rows_op) {
        partial[blockIdx.y * rows_op + i] = acc_wide;
    }
}

// __half-output counterpart of mv_fp32_16_wide_reduce_kernel/
// mv_fp32_wide_reduce_kernel (both write float* y) -- sums the col_splits
// partial dot products for each row and applies alpha/beta, narrowing to
// __half only on the final write, same convention mv_fp16_native_storage_
// tiled_kernel's own final combine already uses.
__global__ void mv_fp16_native_storage_wide_reduce_kernel(double alpha, double beta, size_t rows_op,
                                                            size_t col_splits, const double* partial,
                                                            __half* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_op) return;
    double acc = 0.0;
    for (size_t s = 0; s < col_splits; ++s) acc += partial[s * rows_op + i];
    double result = (beta != 0.0) ? (alpha * acc + beta * static_cast<double>(static_cast<float>(y[i])))
                                   : (alpha * acc);
    y[i] = static_cast<__half>(static_cast<float>(result));
}

// Column-split, half2-packed, reduce-combine dispatcher -- mechanically
// ported combine/arithmetic strategy from the historically-fastest
// FP32_16-family kernel (mv_fp32_16_pure_half_wide_ilp_shared_half2,
// 78-90% of cuBLAS in CSVs from 2026-08-31). MEASURED (bracketing CSVs
// 2026-09-16 01:37 vs 06:31, same shapes, both still using the *1 col_
// splits formula that has since been bumped to *4 below) to be
// dramatically faster than mv_fp16_native_storage_atomic below at small/
// medium M (16x at M=1024, 6.3x at M=2048, 1.7x at M=4096) but 3-9%
// SLOWER at large M (M>=12288) -- the reduce-kernel path pays for a
// second kernel launch plus reading/writing the double-precision
// partial[] buffer through global memory, overhead the atomic path never
// has (writes straight into y); the atomic path pays for that by having
// col_splits atomicAdd-contending blocks per row, made worse by more
// col_splits, which is exactly why the target multiplier stayed at
// choose_col_splits_fp32's own *1 in THIS dispatcher's own tier (small/
// medium M, where reduce-combine is used regardless) even after
// mv_fp16_native_storage_atomic's own tier below was bumped to *4 to
// close its own gap against "TOMOS FP16 (implicit, fast)" (see that
// function's own comment). See mv_fp16_native_storage below for the
// size-based dispatch between the two.
void mv_fp16_native_storage_half2(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                                   const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];
    if (rows_op < kMvFp32WideRowThreshold) {
        int col_splits = choose_col_splits_fp32(rows_op, cols_op);
        int row_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        double* partial = nullptr;
        cudaError_t err = cudaMalloc(&partial, rows_op * static_cast<size_t>(col_splits) * sizeof(double));
        if (err != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        dim3 grid(row_blocks, col_splits);
        mv_fp16_native_storage_wide_half2_kernel<<<grid, kMvFp16NativeTileSize>>>(
            trans, size[0], size[1], A, ld_a, x, col_splits, partial);
        int reduce_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        mv_fp16_native_storage_wide_reduce_kernel<<<reduce_blocks, kMvFp16NativeTileSize>>>(
            alpha, beta, rows_op, col_splits, partial, y);
        cudaFree(partial);
    } else {
        int threads = kMvFp16NativeTileSize;
        int blocks = static_cast<int>((rows_op + threads - 1) / threads);
        mv_fp16_native_storage_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x, beta, y);
    }
    cudaDeviceSynchronize();
}

// Column-split, scalar-__half, atomicAdd-combine dispatcher -- the
// original col-split fix for mv_fp16_native_storage's own severe small/
// medium-M underutilization (single-block-per-row design, only
// ceil(rows_op/256) blocks regardless of K). See mv_fp16_native_storage_
// half2 above for the measured atomic-vs-reduce trade-off this and that
// function make in opposite directions.
//
// col_splits target bumped from choose_col_splits_fp32's own *1 to *4
// (matching mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x's
// own target) after a direct comparison found this dispatcher's own tier
// launching 4x FEWER blocks than "TOMOS FP16 (implicit, fast)" at every
// shape (col_splits=2 vs 8 at M=K=65536, 128 vs 512 at M=1024, confirmed
// by computing both formulas directly) despite doing zero decode work --
// the ~5-30% gap against that series was pure under-occupancy from this
// formula mismatch, not a decode-cost difference (this kernel decodes
// nothing). Left at *1 in mv_fp16_native_storage_half2's own tier above,
// which already wins big against this dispatcher's own tier at small/
// medium M using the *1 formula unmodified (matches pure_half_half2's own
// historically-proven formula exactly) -- bumping that tier too would
// also grow the reduce-kernel's own per-row combine loop (one add per
// col_splits), a cost the atomic path here doesn't pay the same way.
void mv_fp16_native_storage_atomic(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                                    const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];
    if (rows_op < kMvFp32WideRowThreshold) {
        int col_splits = choose_col_splits_fp32_target(rows_op, cols_op, kMvFp32TargetTotalThreads * 4);
        int row_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        if (beta == 0.0) {
            cudaMemset(y, 0, rows_op * sizeof(__half));
        } else {
            int threads = kMvFp16NativeTileSize;
            int blocks = (rows_op + threads - 1) / threads;
            mv_fp16_native_storage_atomic_prescale_kernel<<<blocks, threads>>>(rows_op, beta, y);
        }
        dim3 grid(row_blocks, col_splits);
        mv_fp16_native_storage_atomic_kernel<<<grid, kMvFp16NativeTileSize>>>(
            trans, size[0], size[1], A, ld_a, x, col_splits, alpha, y);
    } else {
        int threads = kMvFp16NativeTileSize;
        int blocks = static_cast<int>((rows_op + threads - 1) / threads);
        mv_fp16_native_storage_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x, beta, y);
    }
    cudaDeviceSynchronize();
}

// "_best" dispatcher, same dispatch-not-merge idea as mv_fp32_16_pure_
// half_wide_ilp_shared_half2_best (which picks between bigtile_half2 and
// wide4x for the FP32_16-packed family): below the crossover, calls
// mv_fp16_native_storage_half2 (dramatically better at small/medium M,
// per its own comment); at/above it, calls mv_fp16_native_storage_atomic
// (slightly better at large M once atomic contention stops being severe).
// Threshold picked from the bracketing measurement itself: 8192 was
// roughly a wash (1012.9 GFlop/s either way), 4096 clearly favored half2
// (1.7x), 12288 clearly favored atomic (0.97x, i.e. atomic faster) -- not
// independently re-tuned beyond reading off that one data point.
static constexpr size_t kMvFp16NativeStorageBestCrossoverThreshold = 8192;

// Defined in fp32_16_cuda_kernels.cu (same tomos::cuda::experimental
// namespace, different translation unit -- needs a forward declaration
// here, same cross-file pattern used for mv_fp16_native_mixed's own
// TRANSPOSE kernel just below). A mechanical duplicate of mv_fp32_16_
// pure_half_transpose_atomiconly_scaled_mblock_vec4_split (LSQR's own
// --implicit-duplicates production TRANSPOSE kernel), adapted for
// mv_fp16_native_storage's own all-__half (A, x, y) contract -- see its
// own comment in fp32_16_cuda_kernels.cu for the full rationale.
size_t mv_fp16_native_storage_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size);
void mv_fp16_native_storage_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                            dim<2> size, const __half* A,
                                                                            size_t ld_a, const __half* x,
                                                                            size_t ld_x, double beta, __half* y,
                                                                            size_t ld_y, double* workspace);

// TRANSPOSE now dispatches to its own dedicated, M-block-scaled kernel
// above instead of falling through to the same rows_op-based half2/
// atomic crossover NO_TRANSPOSE uses -- that crossover was never tuned
// against TRANSPOSE's own access pattern (idx=j+i*ld_a, uncoalesced) and
// is exactly what the GEVM TRANSPOSE plot's "TOMOS FP16" series was
// measuring when the user reported "very low performance" this session.
// Same fix as mv_fp16_native_mixed's own TRANSPOSE dispatch just below,
// applied to this, its all-__half sibling. Also given the same
// persistent, grow-as-needed scratch buffer as mv_pure_half_impl's own
// TRANSPOSE branch (fp32_16_registries.cpp) and mv_fp16_native_mixed's
// own TRANSPOSE dispatch just below -- see either of those for the full
// rationale (avoids a cudaMalloc/cudaFree round-trip on every call).
void mv_fp16_native_storage(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                            const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y) {
    if (trans != TransposeType::NO_TRANSPOSE) {
        if (ld_a != size[0]) return;
        static double* ws = nullptr;
        static size_t ws_capacity = 0;
        size_t ws_elems = mv_fp16_native_storage_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(size);
        if (ws_elems > ws_capacity) {
            if (ws != nullptr) cudaFree(ws);
            ws = nullptr;
            ws_capacity = 0;
            cudaError_t err = cudaMalloc(&ws, ws_elems * sizeof(double));
            if (err != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ws_capacity = ws_elems;
        }
        mv_fp16_native_storage_transpose_atomiconly_scaled_mblock_vec4_split(trans, alpha, size, A, ld_a, x, ld_x,
                                                                              beta, y, ld_y, ws);
        return;
    }
    size_t rows_op = size[0];
    if (rows_op < kMvFp16NativeStorageBestCrossoverThreshold) {
        mv_fp16_native_storage_half2(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
    } else {
        mv_fp16_native_storage_atomic(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
    }
}

// mv_fp16_native_mixed: true mechanical duplicate of mv_fp16_native_
// storage's ENTIRE 7-piece dispatch tree above (tiled_kernel/atomic_
// kernel/atomic_prescale_kernel/wide_half2_kernel/wide_reduce_kernel/
// half2/atomic/_best -- same __hfma2-packed accumulation, same col_splits
// formulas/targets, same crossover threshold, same combine strategies),
// with x and y changed from __half to float throughout -- matching
// mv_fp16_native's own existing (A=half, x=float, y=float) type
// signature exactly, NOT mv_fp16_native_storage's own all-half one.
// Built because mv_fp16_native_storage itself cannot be swapped directly
// into --explicit-duplicates' own dispatch slot (encode_mv(CUDA, FP16,
// FP32, FP32, FP16) in dense_registries.cpp): that slot's x/y Dense
// buffers are genuinely float-typed, and reinterpreting a float buffer as
// __half would silently corrupt every read/write (wrong element size,
// not just a precision change). This kernel gives --explicit-duplicates
// the SAME performance work "TOMOS FP16" itself received (half2 packing,
// the col_splits/occupancy fix, the reduce-vs-atomic "_best" split)
// while keeping mv_fp16_native's own established mixed-precision
// semantics (A at fp16, x/y at float) exactly as before -- a faster
// implementation of the same contract, not a different one.
//
// x_tile staging narrows float->__half via static_cast (matching mv_fp16_
// native_tiled_kernel's own existing convention, unlike mv_fp16_native_
// storage's own direct __half-to-__half copy); y read-modify-write and
// atomicAdd stay in float throughout (removing the __half narrow/widen
// mv_fp16_native_storage's own y writes needed) -- atomicAdd(float*,
// float) is a plain, always-supported CUDA built-in besides, unlike
// atomicAdd(__half*, __half)'s own alignment caveat noted elsewhere in
// this file.
__global__ void mv_fp16_native_mixed_tiled_kernel(TransposeType trans, double alpha, size_t m, size_t n,
                                                    const __half* A, size_t ld_a, const float* x, double beta,
                                                    float* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __half x_tile[kMvFp16NativeTileSize];
    double acc_wide = 0.0;
    for (size_t tile_start = 0; tile_start < cols_op; tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize), cols_op - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = static_cast<__half>(x[tile_start + threadIdx.x]);
        __syncthreads();

        if (i < rows_op) {
            __half tile_acc = static_cast<__half>(0.0f);
            for (size_t jj = 0; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                tile_acc = tile_acc + A[idx] * x_tile[jj];
            }
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
        }
        __syncthreads();
    }

    if (i >= rows_op) return;
    y[i] = (beta != 0.0)
              ? static_cast<float>(alpha * acc_wide + beta * static_cast<double>(y[i]))
              : static_cast<float>(alpha * acc_wide);
}

// Tile-boundary double-widened accumulation (acc_wide, widened once per
// kMvFp16NativeTileSize=256-element tile, exactly like mv_fp16_native_
// mixed_tiled_kernel's own established fix) added after a real LSQR run
// on the actual production matrix (M=1,000,000, K=1000) exposed why this
// was needed: for TRANSPOSE at that shape, col_splits comes out to ~128
// (rows_op=K=1000 stays below the "_best" crossover, landing in the
// half2/reduce tier), giving a ~7813-element chunk per block -- this
// kernel body was mechanically ported from mv_fp32_16_pure_half_wide_
// ilp_shared_half2_atomic_kernel, which never accumulates unwidened
// __half2 over a chunk anywhere NEAR that long in ANY proven production
// path (LSQR's own packed-storage dispatcher always routes TRANSPOSE to
// a completely separate, dedicated kernel instead -- mv_fp32_16_pure_
// half_transpose_atomiconly_scaled_mblock_vec4_split -- never this
// NO_TRANSPOSE-oriented one), so there was no existing evidence this
// design tolerates a chunk this long without the same stagnation risk
// that caused mv_fp16_native_tiled_kernel's own original catastrophic-
// divergence bug (see that kernel's comment) -- not worth assuming safe.
__global__ void mv_fp16_native_mixed_atomic_kernel(TransposeType trans, size_t m, size_t n,
                                                     const __half* A, size_t ld_a,
                                                     const float* x, size_t col_splits,
                                                     double alpha, float* y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t chunk = (cols_op + col_splits - 1) / col_splits;
    size_t col_start = blockIdx.y * chunk;
    size_t col_end = min(col_start + chunk, cols_op);

    __shared__ __align__(4) __half x_tile[kMvFp16NativeTileSize];
    __half2 acc01 = __half2half2(static_cast<__half>(0.0f));
    __half2 acc23 = __half2half2(static_cast<__half>(0.0f));
    double acc_wide = 0.0;
    for (size_t tile_start = col_start; tile_start < col_end; tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize), col_end - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = static_cast<__half>(x[tile_start + threadIdx.x]);
        __syncthreads();

        if (i < rows_op) {
            size_t jj = 0;
            for (; jj + 4 <= tile_len; jj += 4) {
                size_t j0 = tile_start + jj, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
                size_t r0 = (trans == TransposeType::NO_TRANSPOSE) ? i : j0;
                size_t c0 = (trans == TransposeType::NO_TRANSPOSE) ? j0 : i;
                size_t r1 = (trans == TransposeType::NO_TRANSPOSE) ? i : j1;
                size_t c1 = (trans == TransposeType::NO_TRANSPOSE) ? j1 : i;
                size_t r2 = (trans == TransposeType::NO_TRANSPOSE) ? i : j2;
                size_t c2 = (trans == TransposeType::NO_TRANSPOSE) ? j2 : i;
                size_t r3 = (trans == TransposeType::NO_TRANSPOSE) ? i : j3;
                size_t c3 = (trans == TransposeType::NO_TRANSPOSE) ? j3 : i;
                size_t idx0 = r0 + c0 * ld_a, idx1 = r1 + c1 * ld_a;
                size_t idx2 = r2 + c2 * ld_a, idx3 = r3 + c3 * ld_a;
                __half a0 = A[idx0];
                __half a1 = A[idx1];
                __half a2 = A[idx2];
                __half a3 = A[idx3];
                __half2 a01 = __halves2half2(a0, a1);
                __half2 a23 = __halves2half2(a2, a3);
                __half2 x01 = *reinterpret_cast<__half2*>(&x_tile[jj]);
                __half2 x23 = *reinterpret_cast<__half2*>(&x_tile[jj + 2]);
                acc01 = __hfma2(a01, x01, acc01);
                acc23 = __hfma2(a23, x23, acc23);
            }
            for (; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                __half a_val = A[idx];
                __half2 a_pad = __halves2half2(a_val, static_cast<__half>(0.0f));
                __half2 x_pad = __halves2half2(x_tile[jj], static_cast<__half>(0.0f));
                acc01 = __hfma2(a_pad, x_pad, acc01);
            }
            __half2 acc_sum2 = __hadd2(acc01, acc23);
            __half tile_acc = __low2half(acc_sum2) + __high2half(acc_sum2);
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
            acc01 = __half2half2(static_cast<__half>(0.0f));
            acc23 = __half2half2(static_cast<__half>(0.0f));
        }
        __syncthreads();
    }

    if (i < rows_op) {
        float contribution = static_cast<float>(alpha) * static_cast<float>(acc_wide);
        atomicAdd(&y[i], contribution);
    }
}

__global__ void mv_fp16_native_mixed_atomic_prescale_kernel(size_t rows_op, double beta, float* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows_op)
        y[i] = static_cast<float>(beta * static_cast<double>(y[i]));
}

// Same tile-boundary double-widened accumulation fix as mv_fp16_native_
// mixed_atomic_kernel above -- see that kernel's own comment for why.
__global__ void mv_fp16_native_mixed_wide_half2_kernel(
    TransposeType trans, size_t m, size_t n, const __half* A, size_t ld_a,
    const float* x, size_t col_splits, double* partial) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t chunk = (cols_op + col_splits - 1) / col_splits;
    size_t col_start = blockIdx.y * chunk;
    size_t col_end = min(col_start + chunk, cols_op);

    __shared__ __align__(4) __half x_tile[kMvFp16NativeTileSize];
    __half2 acc01 = __half2half2(static_cast<__half>(0.0f));
    __half2 acc23 = __half2half2(static_cast<__half>(0.0f));
    double acc_wide = 0.0;
    for (size_t tile_start = col_start; tile_start < col_end;
         tile_start += kMvFp16NativeTileSize) {
        size_t tile_len = min(static_cast<size_t>(kMvFp16NativeTileSize),
                              col_end - tile_start);
        if (threadIdx.x < tile_len)
            x_tile[threadIdx.x] = static_cast<__half>(x[tile_start + threadIdx.x]);
        __syncthreads();

        if (i < rows_op) {
            size_t jj = 0;
            for (; jj + 4 <= tile_len; jj += 4) {
                size_t j0 = tile_start + jj, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
                size_t r0 = (trans == TransposeType::NO_TRANSPOSE) ? i : j0;
                size_t c0 = (trans == TransposeType::NO_TRANSPOSE) ? j0 : i;
                size_t r1 = (trans == TransposeType::NO_TRANSPOSE) ? i : j1;
                size_t c1 = (trans == TransposeType::NO_TRANSPOSE) ? j1 : i;
                size_t r2 = (trans == TransposeType::NO_TRANSPOSE) ? i : j2;
                size_t c2 = (trans == TransposeType::NO_TRANSPOSE) ? j2 : i;
                size_t r3 = (trans == TransposeType::NO_TRANSPOSE) ? i : j3;
                size_t c3 = (trans == TransposeType::NO_TRANSPOSE) ? j3 : i;
                size_t idx0 = r0 + c0 * ld_a, idx1 = r1 + c1 * ld_a;
                size_t idx2 = r2 + c2 * ld_a, idx3 = r3 + c3 * ld_a;
                __half a0 = A[idx0];
                __half a1 = A[idx1];
                __half a2 = A[idx2];
                __half a3 = A[idx3];
                __half2 a01 = __halves2half2(a0, a1);
                __half2 a23 = __halves2half2(a2, a3);
                __half2 x01 = *reinterpret_cast<__half2*>(&x_tile[jj]);
                __half2 x23 = *reinterpret_cast<__half2*>(&x_tile[jj + 2]);
                acc01 = __hfma2(a01, x01, acc01);
                acc23 = __hfma2(a23, x23, acc23);
            }
            for (; jj < tile_len; ++jj) {
                size_t j = tile_start + jj;
                size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
                size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
                size_t idx = a_row + a_col * ld_a;
                __half a_val = A[idx];
                __half2 a_pad = __halves2half2(a_val, static_cast<__half>(0.0f));
                __half2 x_pad =
                    __halves2half2(x_tile[jj], static_cast<__half>(0.0f));
                acc01 = __hfma2(a_pad, x_pad, acc01);
            }
            __half2 acc_sum2 = __hadd2(acc01, acc23);
            __half tile_acc = __low2half(acc_sum2) + __high2half(acc_sum2);
            acc_wide += static_cast<double>(static_cast<float>(tile_acc));
            acc01 = __half2half2(static_cast<__half>(0.0f));
            acc23 = __half2half2(static_cast<__half>(0.0f));
        }
        __syncthreads();
    }

    if (i < rows_op) {
        partial[blockIdx.y * rows_op + i] = acc_wide;
    }
}

__global__ void mv_fp16_native_mixed_wide_reduce_kernel(double alpha, double beta, size_t rows_op,
                                                          size_t col_splits, const double* partial,
                                                          float* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_op) return;
    double acc = 0.0;
    for (size_t s = 0; s < col_splits; ++s) acc += partial[s * rows_op + i];
    y[i] = (beta != 0.0) ? static_cast<float>(alpha * acc + beta * static_cast<double>(y[i]))
                         : static_cast<float>(alpha * acc);
}

void mv_fp16_native_mixed_half2(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                                 const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];
    if (rows_op < kMvFp32WideRowThreshold) {
        int col_splits = choose_col_splits_fp32(rows_op, cols_op);
        int row_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        double* partial = nullptr;
        cudaError_t err = cudaMalloc(&partial, rows_op * static_cast<size_t>(col_splits) * sizeof(double));
        if (err != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        dim3 grid(row_blocks, col_splits);
        mv_fp16_native_mixed_wide_half2_kernel<<<grid, kMvFp16NativeTileSize>>>(
            trans, size[0], size[1], A, ld_a, x, col_splits, partial);
        int reduce_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        mv_fp16_native_mixed_wide_reduce_kernel<<<reduce_blocks, kMvFp16NativeTileSize>>>(
            alpha, beta, rows_op, col_splits, partial, y);
        cudaFree(partial);
    } else {
        int threads = kMvFp16NativeTileSize;
        int blocks = static_cast<int>((rows_op + threads - 1) / threads);
        mv_fp16_native_mixed_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x, beta, y);
    }
    cudaDeviceSynchronize();
}

void mv_fp16_native_mixed_atomic(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                                  const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? size[0] : size[1];
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? size[1] : size[0];
    if (rows_op < kMvFp32WideRowThreshold) {
        int col_splits = choose_col_splits_fp32_target(rows_op, cols_op, kMvFp32TargetTotalThreads * 4);
        int row_blocks = (rows_op + kMvFp16NativeTileSize - 1) / kMvFp16NativeTileSize;
        if (beta == 0.0) {
            cudaMemset(y, 0, rows_op * sizeof(float));
        } else {
            int threads = kMvFp16NativeTileSize;
            int blocks = (rows_op + threads - 1) / threads;
            mv_fp16_native_mixed_atomic_prescale_kernel<<<blocks, threads>>>(rows_op, beta, y);
        }
        dim3 grid(row_blocks, col_splits);
        mv_fp16_native_mixed_atomic_kernel<<<grid, kMvFp16NativeTileSize>>>(
            trans, size[0], size[1], A, ld_a, x, col_splits, alpha, y);
    } else {
        int threads = kMvFp16NativeTileSize;
        int blocks = static_cast<int>((rows_op + threads - 1) / threads);
        mv_fp16_native_mixed_tiled_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1], A, ld_a, x, beta, y);
    }
    cudaDeviceSynchronize();
}

// Same crossover value as mv_fp16_native_storage's own -- not independently
// re-tuned for the mixed x/y=float variant yet (this is a mechanical port,
// not a fresh optimization pass).
static constexpr size_t kMvFp16NativeMixedBestCrossoverThreshold = 8192;

// Defined in fp32_16_cuda_kernels.cu (same tomos::cuda::experimental
// namespace, different translation unit -- needs a forward declaration
// here, same as every other cross-file reference in this codebase). A
// true mechanical duplicate of mv_fp32_16_pure_half_transpose_
// atomiconly_scaled_mblock_vec4_split (LSQR's own --implicit-duplicates
// production TRANSPOSE kernel) with every decode call removed -- see its
// own comment for the full rationale.
size_t mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size);
void mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                         dim<2> size, const __half* A,
                                                                         size_t ld_a, const float* x, size_t ld_x,
                                                                         double beta, float* y, size_t ld_y,
                                                                         float* workspace);

// TRANSPOSE now dispatches to its own dedicated, M-block-scaled kernel
// above (mechanically ported from LSQR's own proven --implicit-
// duplicates TRANSPOSE kernel) instead of reusing the NO_TRANSPOSE-tuned
// col-split "_best" path. History: this function originally routed
// TRANSPOSE straight to the plain, un-split tiled_kernel (reasoning that
// the col-split path's malloc/free/sync-per-call overhead and
// uncoalesced TRANSPOSE access pattern -- idx = j + i*ld_a strides by
// ld_a across a warp for that direction -- made it a bad fit). That
// missed a far bigger effect: on the actual LSQR matrix (M=1,000,000,
// K=1000), TRANSPOSE's rows_op is K, so the plain tiled_kernel's own grid
// (blocks = ceil(rows_op/256)) was only 4 blocks total, each thread
// serially reducing all 1,000,000 NO_TRANSPOSE-direction rows -- measured
// 16.4s solve time. Routing TRANSPOSE through the col-split path instead
// (same as NO_TRANSPOSE) fixed the catastrophic under-parallelization
// (512-2048 blocks instead of 4) but still paid the uncoalesced-access
// cost, and was never actually tuned for this shape either (every
// optimization pass this session was NO_TRANSPOSE-only) -- confirmed
// ~2x slower than --implicit-duplicates' own dedicated TRANSPOSE kernel.
// This dedicated kernel closes that gap by construction, matching the
// exact same design proven in production for the packed-storage family.
// Persistent, grow-as-needed scratch buffer for the TRANSPOSE kernel
// above -- same fix, same rationale as mv_pure_half_impl's own TRANSPOSE
// branch in fp32_16_registries.cpp (see that function's own comment for
// the full derivation): LSQR calls this ~240 times per solve at a FIXED
// problem shape, so a cudaMalloc/cudaFree on every call was pure,
// avoidable overhead completely invisible to the standalone GEVM
// benchmark (which allocates its own workspace once, outside its timed
// loop). Grows (frees the old buffer, allocates a bigger one) only the
// first time a call needs more elements than it currently holds -- one
// real allocation per distinct shape, not one per call. Freed only at
// process exit.
void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    if (trans != TransposeType::NO_TRANSPOSE) {
        if (ld_a != size[0]) return;
        static float* ws = nullptr;
        static size_t ws_capacity = 0;
        size_t ws_elems = mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(size);
        if (ws_elems > ws_capacity) {
            if (ws != nullptr) cudaFree(ws);
            ws = nullptr;
            ws_capacity = 0;
            cudaError_t err = cudaMalloc(&ws, ws_elems * sizeof(float));
            if (err != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ws_capacity = ws_elems;
        }
        mv_fp16_native_mixed_transpose_atomiconly_scaled_mblock_vec4_split(trans, alpha, size, A, ld_a, x, ld_x,
                                                                            beta, y, ld_y, ws);
        return;
    }
    size_t rows_op = size[0];
    if (rows_op < kMvFp16NativeMixedBestCrossoverThreshold) {
        mv_fp16_native_mixed_half2(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
    } else {
        mv_fp16_native_mixed_atomic(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
    }
}

}  // namespace experimental

// Thin forwarding wrappers -- see the matching banner comment above.
// Same signatures as before this reorganization; no caller (plotted
// benchmark, production dispatch, or otherwise) needs to change.
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y) {
    experimental::mv_fp32_tiled(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
}

void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y) {
    experimental::mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(
        trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
}

void mv_fp16_native(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                    const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    experimental::mv_fp16_native(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
}

void mv_fp16_native_storage(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                            const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y) {
    experimental::mv_fp16_native_storage(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
}

void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y) {
    experimental::mv_fp16_native_mixed(trans, alpha, size, A, ld_a, x, ld_x, beta, y, ld_y);
}

// Generic CSR SpMV kernel, untransposed: one thread per row of A, each
// thread owns its own y[i] so no atomics are needed.
template <typename a_t, typename x_t, typename y_t, typename index_t>
__global__ void spmv_csr_kernel(TransposeType trans, double alpha, size_t m,
                                size_t n, size_t nnz, a_t* values,
                                index_t* row_ptrs, index_t* col_idxs, x_t* x,
                                double beta, y_t* y) {
    if (trans == TransposeType::NO_TRANSPOSE) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m) return;

        double acc = 0.0;
        for (index_t k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            acc += static_cast<double>(values[k]) *
                   static_cast<double>(x[col_idxs[k]]);
        }
        // See the identical guard in gemv_generic_kernel above.
        y[i] = (beta != 0.0)
                  ? static_cast<y_t>(alpha * acc + beta * static_cast<double>(y[i]))
                  : static_cast<y_t>(alpha * acc);
    }
}

// y ← beta * y  (used to apply beta before atomic accumulation below). Only
// called with beta == 0 today (mv_csr's transpose path always passes
// through here first), so this always zeroes y rather than reading it --
// avoiding the same "0 * possibly-uninitialized-NaN" hazard noted above.
template <typename y_t>
__global__ void scale_kernel(size_t n, double beta, y_t* y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (beta != 0.0) ? static_cast<y_t>(beta * static_cast<double>(y[i]))
                        : y_t{0};
}

// Transposed CSR SpMV: y (length n) += alpha * A^T * x (length m). Each
// nonzero A(i, col) scatters alpha*A(i,col)*x[i] into y[col], so
// contributions from different rows can land on the same y[col].
//
// For double/float, one thread per row plus atomicAdd is safe and fast.
// __half's atomicAdd is NOT used here: on real hardware it operates on the
// containing 32-bit word (packing the target half with its memory
// neighbor), which can read/write past the end of an odd-length buffer's
// allocation -- confirmed by a real segfault when this was tried. So for
// __half output this kernel instead runs with a single thread (see the grid
// size chosen in mv_csr below), processing all rows sequentially with a
// plain (non-atomic) add -- correct by construction, just not parallel.
template <typename a_t, typename x_t, typename y_t, typename index_t>
__global__ void spmv_csr_transpose_kernel(double alpha, size_t m, a_t* values,
                                          index_t* row_ptrs, index_t* col_idxs,
                                          x_t* x, y_t* y) {
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        double xi = static_cast<double>(x[i]);
        for (index_t k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
            double contrib = alpha * static_cast<double>(values[k]) * xi;
            if constexpr (std::is_same_v<y_t, double> ||
                          std::is_same_v<y_t, float>) {
                atomicAdd(&y[col_idxs[k]], static_cast<y_t>(contrib));
            } else {
                y[col_idxs[k]] = static_cast<y_t>(
                    static_cast<double>(y[col_idxs[k]]) + contrib);
            }
        }
    }
}

template <typename a_t, typename x_t, typename y_t, typename index_t>
void mv_csr(TransposeType trans, double alpha, dim<2> size, size_t nnz,
            a_t* values, index_t* row_ptrs, index_t* col_idxs, x_t* x,
            size_t ld_x, double beta, y_t* y, size_t ld_y,
            cusparseHandle_t handle) {
    int threads = 256;
    if (trans == TransposeType::NO_TRANSPOSE) {
        size_t m = size[0];
        int blocks = (m + threads - 1) / threads;
        spmv_csr_kernel<<<blocks, threads>>>(trans, alpha, size[0], size[1],
                                              nnz, values, row_ptrs, col_idxs,
                                              x, beta, y);
    } else {
        size_t m = size[0];
        size_t n = size[1];
        int n_blocks = (n + threads - 1) / threads;
        scale_kernel<<<n_blocks, threads>>>(n, beta, y);
        if constexpr (std::is_same_v<y_t, double> || std::is_same_v<y_t, float>) {
            int m_blocks = (m + threads - 1) / threads;
            spmv_csr_transpose_kernel<<<m_blocks, threads>>>(
                alpha, m, values, row_ptrs, col_idxs, x, y);
        } else {
            // __half output: single thread, see the kernel comment above.
            spmv_csr_transpose_kernel<<<1, 1>>>(alpha, m, values, row_ptrs,
                                                col_idxs, x, y);
        }
    }
    cudaDeviceSynchronize();
}

#define INST_MV_CSR(a_t, x_t, y_t)                                          \
    template void mv_csr<a_t, x_t, y_t, int>(                               \
        TransposeType, double, dim<2>, size_t, a_t*, int*, int*, x_t*,       \
        size_t, double, y_t*, size_t, cusparseHandle_t);                     \
    template void mv_csr<a_t, x_t, y_t, long>(                              \
        TransposeType, double, dim<2>, size_t, a_t*, long*, long*, x_t*,     \
        size_t, double, y_t*, size_t, cusparseHandle_t);
FOR_ALL_FP_TRIPLES(INST_MV_CSR)
#undef INST_MV_CSR

}  // namespace cuda
}  // namespace tomos
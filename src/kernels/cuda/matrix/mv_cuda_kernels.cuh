#ifndef TOMOS_MV_CUDA_KERNELS_CUH
#define TOMOS_MV_CUDA_KERNELS_CUH

#include <cublas_v2.h>
#include <cusparse.h>

#include <cuda_fp16.h>

#include "dimensions.hpp"
#include "precision.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {

// See the matching banner comment in mv_cuda_kernels.cu -- everything
// below except the thin forwarding declarations just before mv_csr
// (a different, unrelated operation family, left untouched) is declared
// inside tomos::cuda::experimental; those forwarding wrappers re-expose
// exactly the plotted/production subset at plain tomos::cuda scope.
namespace experimental {

template <typename a_t, typename x_t, typename y_t>
__global__ void gemv_generic_kernel(TransposeType trans, double alpha,
                                    size_t m, size_t n, a_t* A, size_t ld_a,
                                    x_t* x, double beta, y_t* y);

template <typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, dim<2> size, a_t* A,
                size_t ld_a, x_t* x, size_t ld_x, double beta, y_t* y,
                size_t ld_y, cublasHandle_t handle);

// Tuned native (unpacked) fp32 GEMV -- mirrors mv_fp32_16_accurate's own
// 3-way shape dispatch (src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu)
// with the packed-decode step replaced by a plain A[idx] read for
// NO_TRANSPOSE or any layout with ld_a != size[0]. For TRANSPOSE with
// ld_a == size[0], dispatches instead to a dedicated fast kernel (warp-
// per-output-column, split-M/atomicAdd, ported from randls2.0's own
// sgemv_cm_transpose_splitM_atomic_xshared) -- see mv_cuda_kernels.cu,
// next to mv_fp32_transpose_atomiconly_scaled_mblock_kernel's own
// definition, for the full derivation. MEASURED on real A100 hardware,
// FINAL state after two rounds of tuning motivated by that same real
// data (an M_BLOCK auto-scaling fix for grid underoccupancy, then --
// the dominant fix -- removing an internal cudaDeviceSynchronize() this
// branch used to call before returning, which was silently dominating
// the timed-benchmark cost at small/medium sizes): 0.73x-0.89x of cuBLAS
// at M=K=1024-8192, climbing to 0.92x-0.98x by M=12288-32768, beating
// cuBLAS at M=49152 (1.021x). See mv_fp32_transpose_atomiconly_scaled_
// mblock_kernel's own comment for the full round-by-round numbers and
// the sync-removal diagnosis. Correctness-checked (verify_mv_fp32_16_
// gemv.cu, "fp32_tiled" case, both TRANSPOSE directions) and benchmarked
// as "TOMOS FP32" (native) in
// mv_fp32_recon_vs_cutlass_ref.cu and mv_fp32_transpose_recon_vs_cublas_
// ref.cu.
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of mv_fp32_tiled's own TRANSPOSE
// fast path -- NOT called from mv_fp32_tiled (which is unchanged), kept
// separate per this session's own standing "add new kernels, don't
// modify existing ones" convention. Adds float4-vectorized loads of A
// (and the shared-memory x tile) to the same warp-per-output-column/
// split-M/atomicAdd structure, with a runtime 16-byte alignment check
// before committing to the vectorized path -- see mv_fp32_transpose_
// atomiconly_scaled_mblock_vec4_kernel's own comment in mv_cuda_
// kernels.cu for the full derivation. Only handles TRANSPOSE + ld_a==
// size[0] (returns immediately otherwise) -- no general fallback, since
// this is a standalone comparison kernel, not a general-purpose entry
// point. MEASURED on real A100 hardware: a real but modest win over its
// unvectorized sibling (+3%-+6% from M=K=3072 upward, a genuine dip at
// M=2048 of -8.7% not chased further) -- reaches parity with or slightly
// beats cuBLAS from M=24576 upward (1.01x-1.05x). See mv_fp32_transpose_
// atomiconly_scaled_mblock_vec4_kernel's own comment for the full
// numbers.
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size, const float* A,
                                                       size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                                       size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of mv_fp32_transpose_atomiconly_
// scaled_mblock_vec4 above -- NOT wired into mv_fp32_tiled, kept
// separate per this session's own standing "add new kernels, don't
// modify existing ones" convention. Same structure as the vec4 sibling,
// but ALSO float4-vectorizes the x-staging loop (global x -> shared sx),
// not just the main accumulation loop's own A/sx reads -- targets the
// dip vec4 itself showed at M=K=1536-2048. MEASURED on real A100
// hardware: a clean, consistent win over vec4 at nearly every size
// (biggest gains, +8%-+12%, exactly where vec4 was weakest), reaching
// parity with or beating cuBLAS from M=12288 upward (0.999x-1.081x) --
// the best-measured native fp32 TRANSPOSE kernel in this file so far.
// See mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec_kernel's
// own comment in mv_cuda_kernels.cu for the full numbers.
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y);

// TOMOS FP16 (native storage) -- plain native __half A, no FP32_16
// packing/decode at all, genuine scalar __half arithmetic. See its own
// comment in mv_cuda_kernels.cu. Registered at the (CUDA, FP16, FP32,
// FP32, compute=FP16) slot in dense_registries.cpp for --explicit-
// duplicates (without --implicit-duplicates)'s per-iteration matvec.
void mv_fp16_native(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// Fully native fp16 GEMV -- A, x, AND y all __half (unlike mv_fp16_native
// above, which keeps x/y at float). See its own comment in
// mv_cuda_kernels.cu. Dispatched as "TOMOS FP16" in the GEVM performance
// plots (plot_gemv_fp32_16_pure_half_vs_cublas_ref.m and its TRANSPOSE
// sibling) -- the existing FP32_16-packed-storage series that used to own
// that name there is now "TOMOS FP16 (implicit)".
void mv_fp16_native_storage(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                            const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y);

// True mechanical duplicate of mv_fp16_native_storage's entire dispatch
// tree above, with x/y changed from __half to float (matching mv_fp16_
// native's own existing mixed-precision signature) -- see its own comment
// in mv_cuda_kernels.cu. Built to bring the same performance work "TOMOS
// FP16" received to --explicit-duplicates's own (CUDA, FP16, FP32, FP32,
// compute=FP16) dispatch slot in dense_registries.cpp, since mv_fp16_
// native_storage's own all-half x/y cannot be substituted there directly
// (that slot's x/y Dense buffers are genuinely float-typed).
void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

}  // namespace experimental

// Thin forwarding declarations -- see the matching definitions in
// mv_cuda_kernels.cu. Same signatures as before this reorganization.
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y);
void mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec(TransposeType trans, double alpha, dim<2> size,
                                                                const float* A, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp16_native(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp16_native_storage(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                            const __half* x, size_t ld_x, double beta, __half* y, size_t ld_y);
void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

template <typename a_t, typename x_t, typename y_t, typename index_t>
void mv_csr(TransposeType trans, double alpha, dim<2> size, size_t nnz,
            a_t* values, index_t* row_ptrs, index_t* col_idxs, x_t* x,
            size_t ld_x, double beta, y_t* y, size_t ld_y,
            cusparseHandle_t handle);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_MV_CUDA_KERNELS_CUH

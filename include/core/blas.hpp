
#ifndef TOMOS_BLAS_HPP
#define TOMOS_BLAS_HPP

#include <variant>

#include "device.hpp"
#include "matrix.hpp"
#include "status.hpp"
#include "transpose.hpp"

namespace tomos {

void zeros(Matrix& target);

// Dense API

// Statically dispatched BLAS functions.

template <Device device, typename value_t>
Status norm2(size_t num_elems, value_t* vec, size_t inc, value_t* norm);

template <Device device, typename value_t>
Status scale(size_t num_elems, value_t alpha, value_t* vec, size_t inc);

template <Device device, typename value_type>
Status axpy(size_t num_elems, value_type alpha, value_type* x, size_t incx,
            value_type* y, size_t incy);

template <Device device, typename value_t>
Status geqrf2(int m, int n, value_t* A, int lda, value_t* tau, value_t* work,
              int lwork, int* info);

template <Device device, typename value_t>
Status trtri(char uplo, char diag, int n, value_t* A, int lda, value_t* invA,
             size_t ldiv, int* info);

template <Device device, typename value_t>
Status extract_upper_triang(Context& context, int n, value_t* A, int lda);

template <typename value_t>
void set_values(value_t value, Dense& target);

// Dynamically dispatched BLAS functions.

// MV multiplication API. The type-erased mv_fn looked up from the registry will
// be cast to either mv_dense_fn or mv_csr_fn at the call site based on the
// MatrixType of A.

Status mv_apply(Device dev, TransposeType transpose, double alpha,
                Dense& A, Dense& x, double beta, Dense& y);

// Explicit compute_type overload: bypasses the plain overload's own default
// (A.get_value_type(), or FP16 when A is NumericType::FP32_16 -- see
// blas.cpp). Meaningful for FP32_16-stored A, which has two registered
// compute tiers (FP16 = "pure_half"/"TOMOS FP16", FP32 = "accurate"/hybrid
// -- see dense_registries.cpp's FP32_16 delegation block); for any other
// (native) storage type, only compute_type == A.get_value_type() is
// registered, so passing anything else throws.
//
// workspace: optional caller-owned device scratch buffer (float elements),
// forwarded verbatim to the resolved mv_dense_fn (see dense_registries.hpp's
// own comment on that typedef). Only the FP32_16 "pure_half" TRANSPOSE
// kernel currently uses it; every other kernel ignores it. Omit it (the
// nullptr default) unless you own a persistent workspace sized for this
// exact call -- dispatch::lsqr_dense_impl is the only current caller that
// does (see Workspace::mv_transpose_scratch_).
Status mv_apply(Device dev, TransposeType transpose, double alpha,
                Dense& A, Dense& x, double beta, Dense& y,
                NumericType compute_type, float* workspace = nullptr);

Status mv_apply(Context& context, TransposeType transpose, double alpha,
                Dense& A, Dense& x, double beta, Dense& y);

Status mv_apply_with_workspace(Context& context, TransposeType transpose,
                               double alpha, Dense& A, Dense& x, double beta,
                               Dense& y);

Status mv_apply_ext(Context& context, TransposeType transpose, double alpha,
                    Dense& A, Dense& x, double beta, Dense& y);

Status mv_apply_fast(Context& context, TransposeType transpose, double alpha,
                     Dense& A, Dense& x, double beta, Dense& y);

template <typename value_t>
Status mv_apply(Device dev, TransposeType transpose, double alpha, Dense& A,
                value_t* x, double beta, value_t* y);
//                 {
//     auto fn = dispatch::Dense::get().lookup_mv(dev, A.get_value_type(), x, y);
//     fn(transpose, alpha, A, x, beta, y);
//     return SUCCESS;
// }

// Variant overloads — dispatch to the Dense or Sparse overload at runtime.
// Precision is read from the matrix object on every call, enabling adaptive precision.

Status mv_apply(Device dev, TransposeType transpose, double alpha,
                std::variant<Dense*, Sparse*>& A, Dense& x, double beta, Dense& y);

// Explicit compute_type overload of the variant version above. compute_type
// is ignored when A holds a Sparse* (Sparse has no compute_type axis); so is
// workspace (see the Dense-only compute_type overload's own comment above).
Status mv_apply(Device dev, TransposeType transpose, double alpha,
                std::variant<Dense*, Sparse*>& A, Dense& x, double beta,
                Dense& y, NumericType compute_type, float* workspace = nullptr);

void precond_apply(Device dev, TransposeType transpose, double alpha,
                   std::variant<Dense*, Sparse*, std::monostate>& P,
                   Dense& x, double beta, Dense& y);

// MM multiplication API (Dense A). Type-erased registry lookup, mirroring
// mv_apply's own Dense-API design above: resolves purely from A/B/C's own
// NumericType via dispatch::Dense::lookup_mm, so this works for any native
// FP triple (routes to the plain mm_strided kernel, itself a thin cuBLAS
// wrapper for matched dense/dense/dense types) as well as
// NumericType::FP32_16-stored A against FP32 B/C (routes to
// dispatch::Fp32_16's own "accurate" tier via use_workspace=false, or
// "workspace" -- decode-into-scratch-then-native-GEMM -- via
// use_workspace=true; see dense_registries.cpp's own FP32_16 delegation
// block). Throws if no kernel is registered for the given combination
// (e.g. use_workspace=true for a native FP triple, which has only one real
// implementation).
Status mm_apply(Device dev, TransposeType transA, TransposeType transB,
                double alpha, Dense& A, Dense& B, double beta, Dense& C,
                bool use_workspace = false);

// Sparse API

Status mv_apply(Device dev, TransposeType transpose, double alpha,
                Sparse& A, Dense& x, double beta, Dense& y);

template <typename value_t>
void set_values(value_t value, Sparse& target);

template <typename value_t>
Status mv_apply(Device dev, TransposeType transpose, double alpha, Sparse& A,
                value_t* x, double beta, value_t* y);

// MM multiplication API. The type-erased mm_fn looked up from the registry will
// be cast to either mm_dense_fn or mm_csr_fn at the call site based on the
// MatrixType of A.

Status mm_apply(Context& context, TransposeType transpose, double alpha,
                Sparse& A, Dense& x, double beta, Dense& y);

Status mm_apply_with_workspace(Context& context, TransposeType transpose,
                               double alpha, Sparse& A, Dense& x, double beta,
                               Dense& y);

Status mm_apply_ext(Context& context, TransposeType transpose, double alpha,
                    Sparse& A, Dense& x, double beta, Dense& y);

Status mm_apply_fast(Context& context, TransposeType transpose, double alpha,
                     Sparse& A, Dense& x, double beta, Dense& y);

// Split-precision (NumericType::FP32_16) GEMV/GEMM API. A must be a Dense
// matrix stored as FP32_16 (see fp32_16_pack.hpp); every other operand must
// be a plain NumericType::FP32 Dense matrix. Three precision tiers, all
// using only A's wh half for data movement (half the footprint of a plain
// FP32 A) except "ext", which also reads wr:
//   - "pure_half": wh decoded once per element (fp32_16_half_only), then the
//     entire multiply-accumulate reduction runs in __half arithmetic --
//     genuine fp16-only compute.
//   - "fast": same per-element wh decode, but widened to float/double for
//     the reduction -- fp16-level per-element accuracy without fp16's own
//     accumulation error.
//   - "ext": combines wh and wr element-by-element (fp32_16_combine) for a
//     ~fp32-accurate product.

Status split_fp32_16(Dense& source, Dense& target);

Status combine_fp32_16(Dense& source, Dense& target);

Status mv_apply_fp32_16_pure_half(Device dev, TransposeType transpose,
                                  double alpha, Dense& A, Dense& x,
                                  double beta, Dense& y);

Status mv_apply_fp32_16_fast(Device dev, TransposeType transpose,
                             double alpha, Dense& A, Dense& x, double beta,
                             Dense& y);

Status mv_apply_fp32_16_ext(Device dev, TransposeType transpose, double alpha,
                            Dense& A, Dense& x, double beta, Dense& y);

Status mm_apply_fp32_16_pure_half(Device dev, TransposeType transA,
                                  TransposeType transB, double alpha,
                                  Dense& A, Dense& B, double beta, Dense& C);

Status mm_apply_fp32_16_fast(Device dev, TransposeType transA,
                             TransposeType transB, double alpha, Dense& A,
                             Dense& B, double beta, Dense& C);

Status mm_apply_fp32_16_ext(Device dev, TransposeType transA,
                            TransposeType transB, double alpha, Dense& A,
                            Dense& B, double beta, Dense& C);

}  // namespace tomos

#endif  // TOMOS_LINOP_HPP
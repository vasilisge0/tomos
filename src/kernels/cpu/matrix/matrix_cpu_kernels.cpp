#include "matrix_cpu_kernels.hpp"

#include <type_traits>

#include "type_macros.hpp"

#if defined USE_OPENBLAS
#include "cblas.h"
#endif

namespace tomos {
namespace cpu {

// CPU has no native FP16 ALU: any of these generic fallbacks that touches
// __half must promote to float (never operate on __half's own,
// software-emulated arithmetic operators). compute_t picks float when any
// of the operand types is __half, else keeps the existing double
// accumulation unchanged for every other combination.
template <typename a_t, typename x_t, typename y_t>
using mv_compute_t = std::conditional_t<
    std::is_same_v<a_t, __half> || std::is_same_v<x_t, __half> ||
        std::is_same_v<y_t, __half>,
    float, double>;

template <typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, dim<2> size, a_t* A,
                size_t ld_a, x_t* x, size_t ld_x, double beta, y_t* y,
                size_t ld_y) {
    size_t m = size[0];
    size_t n = size[1];

#if defined USE_OPENBLAS
    // Use cuBLAS when A, x, y are all the same native BLAS type.
    if constexpr (std::is_same_v<a_t, double> && std::is_same_v<x_t, double> &&
                  std::is_same_v<y_t, double>) {
        CBLAS_TRANSPOSE cblas_trans = (trans == TransposeType::NO_TRANSPOSE)
                                         ? CblasNoTrans
                                         : CblasTrans;
        cblas_dgemv(CblasColMajor, cblas_trans, m, n, alpha, A, ld_a,
                    reinterpret_cast<double*>(x), 1, beta,
                    reinterpret_cast<double*>(y), 1);
        return;
    }
    if constexpr (std::is_same_v<a_t, float> && std::is_same_v<x_t, float> &&
                  std::is_same_v<y_t, float>) {
        CBLAS_TRANSPOSE cblas_trans = (trans == TransposeType::NO_TRANSPOSE)
                                         ? CblasNoTrans
                                         : CblasTrans;
        float f_alpha = static_cast<float>(alpha);
        float f_beta = static_cast<float>(beta);
        cblas_sgemv(CblasColMajor, cblas_trans, m, n, f_alpha, A, ld_a,
                    reinterpret_cast<float*>(x), 1, f_beta,
                    reinterpret_cast<float*>(y), 1);
        return;
    }
#endif

    // Generic mixed-precision fallback.
    using compute_t = mv_compute_t<a_t, x_t, y_t>;
    compute_t c_alpha = static_cast<compute_t>(alpha);
    compute_t c_beta  = static_cast<compute_t>(beta);
    // Per BLAS convention, beta == 0 means y need not be set on entry -- its
    // storage may be uninitialized (e.g. freshly carved from a Workspace
    // arena), so beta * y[i] must not be evaluated in that case: 0 * NaN is
    // NaN, not 0, and would silently corrupt the result.
    bool y_is_read = (beta != 0.0);
    size_t rows_op = (trans == TransposeType::NO_TRANSPOSE) ? m : n;
    size_t cols_op = (trans == TransposeType::NO_TRANSPOSE) ? n : m;

    for (size_t i = 0; i < rows_op; ++i) {
        compute_t acc = compute_t{0};
        for (size_t j = 0; j < cols_op; ++j) {
            size_t a_row = (trans == TransposeType::NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == TransposeType::NO_TRANSPOSE) ? j : i;
            acc += static_cast<compute_t>(A[a_row + a_col * ld_a]) *
                   static_cast<compute_t>(x[j]);
        }
        y[i] = y_is_read ? static_cast<y_t>(c_alpha * acc +
                                            c_beta * static_cast<compute_t>(y[i]))
                         : static_cast<y_t>(c_alpha * acc);
    }
}

#define INST_MV_STRIDED(a_t, x_t, y_t)                                    \
    template void mv_strided<a_t, x_t, y_t>(TransposeType, double, dim<2>, \
                                             a_t*, size_t, x_t*, size_t,   \
                                             double, y_t*, size_t);
FOR_ALL_FP_TRIPLES(INST_MV_STRIDED)
#undef INST_MV_STRIDED

template <typename a_t, typename x_t, typename y_t, typename index_t>
void mv_csr(TransposeType trans, double alpha, dim<2> size, size_t nnz,
            a_t* values, index_t* row_ptrs, index_t* col_idxs, x_t* x,
            size_t ld_x, double beta, y_t* y, size_t ld_y) {
    size_t m = size[0];
    size_t n = size[1];
    using compute_t = mv_compute_t<a_t, x_t, y_t>;
    compute_t c_alpha = static_cast<compute_t>(alpha);
    compute_t c_beta  = static_cast<compute_t>(beta);
    // See the identical guard in mv_strided above: beta == 0 means y's
    // existing (possibly uninitialized) contents must not be read.
    bool y_is_read = (beta != 0.0);

    if (trans == TransposeType::NO_TRANSPOSE) {
        for (size_t i = 0; i < m; ++i) {
            compute_t acc = compute_t{0};
            for (index_t k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
                acc += static_cast<compute_t>(values[k]) *
                       static_cast<compute_t>(x[col_idxs[k]]);
            }
            y[i] = y_is_read
                       ? static_cast<y_t>(c_alpha * acc +
                                          c_beta * static_cast<compute_t>(y[i]))
                       : static_cast<y_t>(c_alpha * acc);
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            y[j] = y_is_read
                       ? static_cast<y_t>(c_beta * static_cast<compute_t>(y[j]))
                       : y_t{0};
        }
        for (size_t i = 0; i < m; ++i) {
            compute_t xi = static_cast<compute_t>(x[i]);
            for (index_t k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k) {
                y[col_idxs[k]] = static_cast<y_t>(
                    static_cast<compute_t>(y[col_idxs[k]]) +
                    c_alpha * static_cast<compute_t>(values[k]) * xi);
            }
        }
    }
}

#define INST_MV_CSR(a_t, x_t, y_t)                                       \
    template void mv_csr<a_t, x_t, y_t, int>(                            \
        TransposeType, double, dim<2>, size_t, a_t*, int*, int*, x_t*,    \
        size_t, double, y_t*, size_t);                                    \
    template void mv_csr<a_t, x_t, y_t, long>(                           \
        TransposeType, double, dim<2>, size_t, a_t*, long*, long*, x_t*,  \
        size_t, double, y_t*, size_t);
FOR_ALL_FP_TRIPLES(INST_MV_CSR)
#undef INST_MV_CSR

}  // namespace cpu
}  // namespace tomos
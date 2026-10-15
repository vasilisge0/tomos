#include "mm_cpu_kernels.hpp"

#include "type_macros.hpp"

#if defined USE_OPENBLAS
#include "cblas.h"
#endif

namespace tomos {
namespace cpu {

template <typename a_t, typename b_t, typename c_t>
void mm_strided(TransposeType transA, TransposeType transB, double alpha,
                dim<2> size_A, a_t* A, size_t ld_a,
                dim<2> size_B, b_t* B, size_t ld_b,
                double beta, c_t* C, size_t ld_c) {
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];

#if defined USE_OPENBLAS
    if constexpr (std::is_same_v<a_t, double> && std::is_same_v<b_t, double> &&
                  std::is_same_v<c_t, double>) {
        CBLAS_TRANSPOSE ta = (transA == TransposeType::NO_TRANSPOSE)
                                 ? CblasNoTrans
                                 : CblasTrans;
        CBLAS_TRANSPOSE tb = (transB == TransposeType::NO_TRANSPOSE)
                                 ? CblasNoTrans
                                 : CblasTrans;
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, ld_a, B, ld_b,
                    beta, C, ld_c);
        return;
    }
    if constexpr (std::is_same_v<a_t, float> && std::is_same_v<b_t, float> &&
                  std::is_same_v<c_t, float>) {
        CBLAS_TRANSPOSE ta = (transA == TransposeType::NO_TRANSPOSE)
                                 ? CblasNoTrans
                                 : CblasTrans;
        CBLAS_TRANSPOSE tb = (transB == TransposeType::NO_TRANSPOSE)
                                 ? CblasNoTrans
                                 : CblasTrans;
        float f_alpha = static_cast<float>(alpha);
        float f_beta = static_cast<float>(beta);
        cblas_sgemm(CblasColMajor, ta, tb, m, n, k, f_alpha, A, ld_a, B, ld_b,
                    f_beta, C, ld_c);
        return;
    }
#endif

    // Generic mixed-precision fallback (column-major, outer-product loop order).
    // Scale C by beta first.
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            C[i + j * ld_c] =
                static_cast<c_t>(beta * static_cast<double>(C[i + j * ld_c]));

    // Accumulate alpha * op(A) * op(B) column by column of B.
    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            double b_val =
                (transB == TransposeType::NO_TRANSPOSE)
                    ? static_cast<double>(B[l + j * ld_b])
                    : static_cast<double>(B[j + l * ld_b]);
            double scale = alpha * b_val;
            for (size_t i = 0; i < m; ++i) {
                double a_val =
                    (transA == TransposeType::NO_TRANSPOSE)
                        ? static_cast<double>(A[i + l * ld_a])
                        : static_cast<double>(A[l + i * ld_a]);
                C[i + j * ld_c] = static_cast<c_t>(
                    static_cast<double>(C[i + j * ld_c]) + scale * a_val);
            }
        }
    }
}

#define INST_MM_STRIDED(a_t, b_t, c_t)                                      \
    template void mm_strided<a_t, b_t, c_t>(                                \
        TransposeType, TransposeType, double, dim<2>, a_t*, size_t, dim<2>, \
        b_t*, size_t, double, c_t*, size_t);
FOR_ALL_FP_TRIPLES(INST_MM_STRIDED)
#undef INST_MM_STRIDED

template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr(TransposeType transA, TransposeType transB, double alpha,
            dim<2> size_A, size_t nnz, a_t* values, index_t* row_ptrs,
            index_t* col_idxs, dim<2> size_B, b_t* B, size_t ld_b,
            double beta, c_t* C, size_t ld_c) {
    // row_ptrs has size_A[0]+1 entries (stored row count).
    size_t m = (transA == TransposeType::NO_TRANSPOSE) ? size_A[0] : size_A[1];
    size_t k = (transA == TransposeType::NO_TRANSPOSE) ? size_A[1] : size_A[0];
    size_t n = (transB == TransposeType::NO_TRANSPOSE) ? size_B[1] : size_B[0];

    if (transA == TransposeType::NO_TRANSPOSE) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                double acc = 0.0;
                for (index_t p = row_ptrs[i]; p < row_ptrs[i + 1]; ++p) {
                    size_t l = static_cast<size_t>(col_idxs[p]);
                    double b_val =
                        (transB == TransposeType::NO_TRANSPOSE)
                            ? static_cast<double>(B[l + j * ld_b])
                            : static_cast<double>(B[j + l * ld_b]);
                    acc += static_cast<double>(values[p]) * b_val;
                }
                C[i + j * ld_c] = static_cast<c_t>(
                    alpha * acc +
                    beta * static_cast<double>(C[i + j * ld_c]));
            }
        }
    } else {
        // A^T: scatter-accumulate. Scale C by beta first.
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                C[i + j * ld_c] = static_cast<c_t>(
                    beta * static_cast<double>(C[i + j * ld_c]));

        for (size_t j = 0; j < n; ++j) {
            // Iterate stored rows of A (= columns of A^T = rows 0..k-1 of op(A)).
            for (size_t r = 0; r < size_A[0]; ++r) {
                double b_val =
                    (transB == TransposeType::NO_TRANSPOSE)
                        ? static_cast<double>(B[r + j * ld_b])
                        : static_cast<double>(B[j + r * ld_b]);
                double scale = alpha * b_val;
                for (index_t p = row_ptrs[r]; p < row_ptrs[r + 1]; ++p) {
                    size_t i = static_cast<size_t>(col_idxs[p]);
                    C[i + j * ld_c] = static_cast<c_t>(
                        static_cast<double>(C[i + j * ld_c]) +
                        scale * static_cast<double>(values[p]));
                }
            }
        }
    }
}

#define INST_MM_CSR(a_t, b_t, c_t)                                           \
    template void mm_csr<a_t, b_t, c_t, int>(                                \
        TransposeType, TransposeType, double, dim<2>, size_t, a_t*, int*,    \
        int*, dim<2>, b_t*, size_t, double, c_t*, size_t);                   \
    template void mm_csr<a_t, b_t, c_t, long>(                               \
        TransposeType, TransposeType, double, dim<2>, size_t, a_t*, long*,   \
        long*, dim<2>, b_t*, size_t, double, c_t*, size_t);
FOR_ALL_FP_TRIPLES(INST_MM_CSR)
#undef INST_MM_CSR

// C = alpha * A * B + beta * C, A (m x k) and B (k x n) in CSR, C dense
// column-major. Row-by-row CSR merge: for each nonzero A(i,l), scatter
// alpha * A(i,l) * B(l, :) into row i of C.
template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr_csr(double alpha, dim<2> size_A, index_t* row_ptrs_A,
               index_t* col_idxs_A, a_t* values_A, dim<2> size_B,
               index_t* row_ptrs_B, index_t* col_idxs_B, b_t* values_B,
               double beta, c_t* C, size_t ld_c) {
    size_t m = size_A[0];
    size_t n = size_B[1];

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            C[i + j * ld_c] =
                static_cast<c_t>(beta * static_cast<double>(C[i + j * ld_c]));

    for (size_t i = 0; i < m; ++i) {
        for (index_t p = row_ptrs_A[i]; p < row_ptrs_A[i + 1]; ++p) {
            size_t l = static_cast<size_t>(col_idxs_A[p]);
            double scale = alpha * static_cast<double>(values_A[p]);
            for (index_t q = row_ptrs_B[l]; q < row_ptrs_B[l + 1]; ++q) {
                size_t j = static_cast<size_t>(col_idxs_B[q]);
                C[i + j * ld_c] = static_cast<c_t>(
                    static_cast<double>(C[i + j * ld_c]) +
                    scale * static_cast<double>(values_B[q]));
            }
        }
    }
}

#define INST_MM_CSR_CSR(a_t, b_t, c_t)                                      \
    template void mm_csr_csr<a_t, b_t, c_t, int>(                           \
        double, dim<2>, int*, int*, a_t*, dim<2>, int*, int*, b_t*, double, \
        c_t*, size_t);                                                      \
    template void mm_csr_csr<a_t, b_t, c_t, long>(                          \
        double, dim<2>, long*, long*, a_t*, dim<2>, long*, long*, b_t*,     \
        double, c_t*, size_t);
FOR_ALL_FP_TRIPLES(INST_MM_CSR_CSR)
#undef INST_MM_CSR_CSR

}  // namespace cpu
}  // namespace tomos
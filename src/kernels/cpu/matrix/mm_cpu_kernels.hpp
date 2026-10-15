#ifndef MM_CPU_KERNELS_HPP
#define MM_CPU_KERNELS_HPP

#include "dimensions.hpp"
#include "precision.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cpu {

// C = alpha * op(A) * op(B) + beta * C  (column-major strided storage)
// size_A and size_B are the *storage* dimensions of A and B (before transpose).
template <typename a_t, typename b_t, typename c_t>
void mm_strided(TransposeType transA, TransposeType transB, double alpha,
                dim<2> size_A, a_t* A, size_t ld_a,
                dim<2> size_B, b_t* B, size_t ld_b,
                double beta, c_t* C, size_t ld_c);

// C = alpha * op(A) * op(B) + beta * C  (A in CSR, B and C column-major strided)
template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr(TransposeType transA, TransposeType transB, double alpha,
            dim<2> size_A, size_t nnz, a_t* values, index_t* row_ptrs,
            index_t* col_idxs, dim<2> size_B, b_t* B, size_t ld_b,
            double beta, c_t* C, size_t ld_c);

// C = alpha * A * B + beta * C  (A and B in CSR, C column-major strided).
// Only the untransposed product is supported.
template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr_csr(double alpha, dim<2> size_A, index_t* row_ptrs_A,
               index_t* col_idxs_A, a_t* values_A, dim<2> size_B,
               index_t* row_ptrs_B, index_t* col_idxs_B, b_t* values_B,
               double beta, c_t* C, size_t ld_c);

}  // namespace cpu
}  // namespace tomos

#endif  // MM_CPU_KERNELS_HPP
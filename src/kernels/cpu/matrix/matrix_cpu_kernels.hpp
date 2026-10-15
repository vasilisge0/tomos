#ifndef MATRIX_CPU_KERNELS_HPP
#define MATRIX_CPU_KERNELS_HPP

#include "dimensions.hpp"
#include "precision.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cpu {

template <typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, dim<2> size, a_t* A,
                size_t ld_a, x_t* x, size_t ld_x, double beta, y_t* y,
                size_t ld_y);

template <typename a_t, typename x_t, typename y_t, typename index_t>
void mv_csr(TransposeType trans, double alpha, dim<2> size, size_t nnz,
            a_t* values, index_t* row_ptrs, index_t* col_idxs, x_t* x,
            size_t ld_x, double beta, y_t* y, size_t ld_y);

}  // namespace cpu
}  // namespace tomos

#endif  // MATRIX_CPU_KERNELS_HPP

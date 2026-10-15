#include <iostream>

#include "dimensions.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace cpu {

// NOT USED. Declared and defined only here (no header), with zero callers
// anywhere in the codebase -- there is no forward declaration anyone else
// could even call it through. The real, dispatched CSR matvec kernel is
// mv_csr in matrix_cpu_kernels.cpp/.hpp, registered into Sparse::lookup_mv
// (see sparse_registries.cpp) and covered by test/kernels/cpu/matrix/
// test_csr_cpu_kernels.cpp. This function looks like an earlier,
// superseded attempt at the same operation, left behind when mv_csr took
// over. It also has a real latent bug that mv_csr's own beta==0 guard
// avoids: `out[i] = out[i] * beta;` below unconditionally reads out[i] even
// when beta == 0, so a caller passing an uninitialized `out` buffer with
// beta == 0 (the normal "don't care about y's prior contents" convention
// used everywhere else in this codebase) would read garbage rather than
// getting a clean overwrite -- worth fixing before ever reviving this
// function, not just deleting the dead-code label.
template <typename value_t, typename index_t>
Status csr_mv_kernel(dim<2> size, value_t alpha, index_t* row_ptrs,
                     index_t* col_idxs, value_t* values, value_t* vec,
                     value_t beta, value_t* out) {
    for (size_t i = 0; i < size[0]; ++i) {
        out[i] = out[i] * beta;
        for (size_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            out[i] = out[i] + alpha * values[j] * vec[col_idxs[j]];
        }
    }
    return SUCCESS;
}

template Status csr_mv_kernel<double, int>(dim<2> size, double alpha,
                                           int* row_ptrs, int* col_idxs,
                                           double* values, double* vec,
                                           double beta, double* out);
template Status csr_mv_kernel<float, int>(dim<2> size, float alpha,
                                          int* row_ptrs, int* col_idxs,
                                          float* values, float* vec, float beta,
                                          float* out);
template Status csr_mv_kernel<half, int>(dim<2> size, half alpha, int* row_ptrs,
                                         int* col_idxs, half* values, half* vec,
                                         half beta, half* out);

}  // end of namespace cpu
}  // namespace tomos

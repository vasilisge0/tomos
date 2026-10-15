#include <iostream>

#include "dimensions.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"
#include "status.hpp"

namespace tomos {
namespace cpu {

template <typename value_in_t, typename value_out_t>
void strided_copy_kernel(dim<2> size, value_in_t* values_in, size_t ld_in,
                         value_out_t* values_out, size_t ld_out) {
    size_t nrows = size[0];
    size_t ncols = size[1];
    if constexpr (std::is_same<value_in_t, value_out_t>::value) {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] = values_in[i + j * ld_in];
            }
        }
    } else {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] =
                    static_cast<value_out_t>(values_in[i + j * ld_in]);
            }
        }
    }
}

template <typename value_in_t, typename value_out_t, typename index_in_t,
          typename index_out_t>
__host__ void csr_copy_impl(dim<2> size, SparseStorage& source,
                            SparseStorage& target) {
    // SparseStorage is std::variant<CsrStorage> -- a single alternative, so
    // .index() is always 0 and can never equal MatrixFormat::CSR (1). A
    // previous format check here compared against that and was always
    // false, making every CSR CPU copy throw; SparseStorage can only ever
    // hold CsrStorage, so no format check is needed.
    CsrStorage& x = std::get<CsrStorage>(source);
    CsrStorage& y = std::get<CsrStorage>(target);
    strided_copy_kernel({x.nnz_, 1}, std::get<value_in_t*>(x.values_), x.nnz_,
                        std::get<value_out_t*>(y.values_), x.nnz_);
    strided_copy_kernel({size[0] + 1, 1}, std::get<index_in_t*>(x.row_ptrs_),
                        size[0] + 1, std::get<index_out_t*>(y.row_ptrs_),
                        size[0] + 1);
    strided_copy_kernel({x.nnz_, 1}, std::get<index_in_t*>(x.col_idxs_),
                        x.nnz_, std::get<index_out_t*>(y.col_idxs_), x.nnz_);
}

template void strided_copy_kernel<double, double>(dim<2>, double*, size_t, double*, size_t);
template void strided_copy_kernel<double, float>(dim<2>, double*, size_t, float*, size_t);
template void strided_copy_kernel<double, __half>(dim<2>, double*, size_t, __half*, size_t);
template void strided_copy_kernel<float, double>(dim<2>, float*, size_t, double*, size_t);
template void strided_copy_kernel<float, float>(dim<2>, float*, size_t, float*, size_t);
template void strided_copy_kernel<float, __half>(dim<2>, float*, size_t, __half*, size_t);
template void strided_copy_kernel<__half, double>(dim<2>, __half*, size_t, double*, size_t);
template void strided_copy_kernel<__half, float>(dim<2>, __half*, size_t, float*, size_t);
template void strided_copy_kernel<__half, __half>(dim<2>, __half*, size_t, __half*, size_t);
template void strided_copy_kernel<double, __nv_bfloat16>(dim<2>, double*, size_t, __nv_bfloat16*, size_t);
template void strided_copy_kernel<float, __nv_bfloat16>(dim<2>, float*, size_t, __nv_bfloat16*, size_t);
template void strided_copy_kernel<__half, __nv_bfloat16>(dim<2>, __half*, size_t, __nv_bfloat16*, size_t);
template void strided_copy_kernel<__nv_bfloat16, double>(dim<2>, __nv_bfloat16*, size_t, double*, size_t);
template void strided_copy_kernel<__nv_bfloat16, float>(dim<2>, __nv_bfloat16*, size_t, float*, size_t);
template void strided_copy_kernel<__nv_bfloat16, __half>(dim<2>, __nv_bfloat16*, size_t, __half*, size_t);
template void strided_copy_kernel<__nv_bfloat16, __nv_bfloat16>(dim<2>, __nv_bfloat16*, size_t, __nv_bfloat16*, size_t);

template void strided_copy_kernel<int32_t, int32_t>(dim<2>, int32_t*, size_t, int32_t*, size_t);
template void strided_copy_kernel<int32_t, int64_t>(dim<2>, int32_t*, size_t, int64_t*, size_t);
template void strided_copy_kernel<int64_t, int32_t>(dim<2>, int64_t*, size_t, int32_t*, size_t);
template void strided_copy_kernel<int64_t, int64_t>(dim<2>, int64_t*, size_t, int64_t*, size_t);

#define INST_CSR_COPY_IMPL(vi, vo, idx) \
    template void csr_copy_impl<vi, vo, idx, idx>(dim<2>, SparseStorage&,   \
                                                  SparseStorage&);

INST_CSR_COPY_IMPL(double, double, int32_t)
INST_CSR_COPY_IMPL(double, double, int64_t)
INST_CSR_COPY_IMPL(double, float, int32_t)
INST_CSR_COPY_IMPL(double, float, int64_t)
INST_CSR_COPY_IMPL(double, __half, int32_t)
INST_CSR_COPY_IMPL(double, __half, int64_t)
INST_CSR_COPY_IMPL(float, double, int32_t)
INST_CSR_COPY_IMPL(float, double, int64_t)
INST_CSR_COPY_IMPL(float, float, int32_t)
INST_CSR_COPY_IMPL(float, float, int64_t)
INST_CSR_COPY_IMPL(float, __half, int32_t)
INST_CSR_COPY_IMPL(float, __half, int64_t)
INST_CSR_COPY_IMPL(__half, double, int32_t)
INST_CSR_COPY_IMPL(__half, double, int64_t)
INST_CSR_COPY_IMPL(__half, float, int32_t)
INST_CSR_COPY_IMPL(__half, float, int64_t)
INST_CSR_COPY_IMPL(__half, __half, int32_t)
INST_CSR_COPY_IMPL(__half, __half, int64_t)
INST_CSR_COPY_IMPL(double, __nv_bfloat16, int32_t)
INST_CSR_COPY_IMPL(double, __nv_bfloat16, int64_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, double, int32_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, double, int64_t)
INST_CSR_COPY_IMPL(float, __nv_bfloat16, int32_t)
INST_CSR_COPY_IMPL(float, __nv_bfloat16, int64_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, float, int32_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, float, int64_t)
INST_CSR_COPY_IMPL(__half, __nv_bfloat16, int32_t)
INST_CSR_COPY_IMPL(__half, __nv_bfloat16, int64_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, __half, int32_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, __half, int64_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, __nv_bfloat16, int32_t)
INST_CSR_COPY_IMPL(__nv_bfloat16, __nv_bfloat16, int64_t)

#undef INST_CSR_COPY_IMPL

}  // namespace cpu
}  // namespace tomos

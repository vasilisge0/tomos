#ifndef TOMOS_COUNTSKETCH_CUDA_KERNELS_CUH
#define TOMOS_COUNTSKETCH_CUDA_KERNELS_CUH

#include "dimensions.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace cuda {

template <typename value_t>
Status generate_countsketch_csr_values(Sparse& S);

template <typename index_t>
Status generate_countsketch_csr_idxs(Sparse& S);

template <typename value_t>
void countsketch_apply_impl(dim<2> size, value_t* X, size_t ld_in,
                            dim<2> size_y, value_t* Y, size_t ld_out);

void countsketch_apply_impl(dim<2> size, __half* X, size_t ld_in, dim<2> size_y,
                            float* Y, size_t ld_out);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_COUNTSKETCH_CUDA_KERNELS_CUH

#ifndef TOMOS_GAUSSIAN_SKETCH_CUDA_KERNELS_CUH
#define TOMOS_GAUSSIAN_SKETCH_CUDA_KERNELS_CUH

#include <cstddef>

#include "dimensions.hpp"

namespace tomos {
namespace cuda {

// Fills the size[0] x size[1] (column-major, leading dimension `ld`) region
// of `values` with i.i.d. N(0,1) draws. Used both to fill a dense Gaussian
// sketch (size = matrix size) and to fill the nonzero values of a sparse one
// (size = {nnz, 1}, ld = nnz).
template <typename value_t>
void generate_gaussian_values(dim<2> size, value_t* values, size_t ld);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_GAUSSIAN_SKETCH_CUDA_KERNELS_CUH

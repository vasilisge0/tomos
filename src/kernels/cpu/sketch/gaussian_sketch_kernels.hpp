#ifndef TOMOS_GAUSSIAN_SKETCH_CPU_KERNELS_HPP
#define TOMOS_GAUSSIAN_SKETCH_CPU_KERNELS_HPP

#include <cstddef>

#include "dimensions.hpp"
#include "status.hpp"

namespace tomos {
namespace cpu {

// Fills the size[0] x size[1] (column-major, leading dimension `ld`) region
// of `values` with i.i.d. N(0,1) draws. Used both to fill a dense Gaussian
// sketch (size = matrix size) and to fill the nonzero values of a sparse one
// (size = {nnz, 1}, ld = nnz).
template <typename value_t>
Status generate_gaussian_values(dim<2> size, value_t* values, size_t ld);

}  // namespace cpu
}  // namespace tomos

#endif  // TOMOS_GAUSSIAN_SKETCH_CPU_KERNELS_HPP

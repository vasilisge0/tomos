#ifndef TOMOS_SKETCH_QR_CUDA_KERNELS_CUH
#define TOMOS_SKETCH_QR_CUDA_KERNELS_CUH

#include "dimensions.hpp"

namespace tomos {
namespace cuda {

// Factors the k x n (k >= n) device matrix `values` via QR (cusolverDn),
// then inverts the resulting n x n upper-triangular R factor via a cuBLAS
// triangular solve against the identity, leaving R^{-1} in the leading
// n x n block of `values`.
template <typename value_t>
void generate_sketch_qr_precond(dim<2> size, value_t* values, size_t ld);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_SKETCH_QR_CUDA_KERNELS_CUH

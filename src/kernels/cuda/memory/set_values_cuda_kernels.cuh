#ifndef TOMOS_SET_VALUES_CUDA_KERNELS_CUH
#define TOMOS_SET_VALUES_CUDA_KERNELS_CUH

#include <cuda_runtime.h>

namespace tomos {
namespace cuda {

template <typename value_t>
__host__ void set_values_impl(
    value_t val,
    dim<2> size,
    value_t* v_out,
    size_t ld_out);

}   // cuda
}   // tomos

#endif  // TOMOS_SET_VALUES_CUDA_KERNELS_CUH
#ifndef TOMOS_CPU_MEMORY_DENSE_STORAGE_CPU_KERNELS_HPP
#define TOMOS_CPU_MEMORY_DENSE_STORAGE_CPU_KERNELS_HPP

#include "dimensions.hpp"
#include "malloc.hpp"

namespace tomos {
namespace cpu {

template <typename value_in_t, typename value_out_t>
void strided_copy_kernel(dim<2> size, value_in_t* values_in, size_t ld_in,
                         value_out_t* values_out, size_t ld_out);

}  // namespace cpu
}  // namespace tomos

#endif  // TOMOS_CPU_MEMORY_DENSE_STORAGE_CPU_KERNELS_HPP
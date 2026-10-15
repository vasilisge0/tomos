#ifndef TOMOS_SET_VALUES_CPU_KERNELS_HPP
#define TOMOS_SET_VALUES_CPU_KERNELS_HPP

#include "dimensions.hpp"

namespace tomos {
namespace cpu {

template <typename value_t>
void set_values_impl(value_t val, dim<2> size, value_t* values, size_t ld);

}   // cpu
}   // tomos

#endif  // TOMOS_SET_VALUES_CPU_KERNELS_HPP
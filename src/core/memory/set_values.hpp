#ifndef TOMOS_SET_VALUES_HPP
#define TOMOS_SET_VALUES_HPP

#include "dimensions.hpp"

namespace tomos  {

template<typename value_t>
void set_values_impl(Device device, value_t val, dim<2> size, value_t* values, size_t ld);

template<typename value_t>
void zeros_impl(Device device, dim<2> size, value_t* values, size_t ld);

}   // tomos

#endif  // TOMOS_SET_VALUES_HPP
#include <iostream>
#include "dimensions.hpp"
#include "precision.hpp"

namespace tomos {
namespace cpu {

template <typename value_t>
void set_values_impl(value_t val, dim<2> size, value_t* values, size_t ld) {
    for (size_t i = 0; i < size[1]; ++i) {
        for (size_t j = 0; j < size[0]; ++j) {
            values[j + i * ld] = val;
        }
    }
}

template void set_values_impl(double val, dim<2> size, double* values,
                              size_t ld);
template void set_values_impl(float val, dim<2> size, float* values, size_t ld);
template void set_values_impl(__half val, dim<2> size, __half* values,
                              size_t ld);
template void set_values_impl(__nv_bfloat16 val, dim<2> size, __nv_bfloat16* values,
                              size_t ld);
// Used to zero-fill NumericType::FP32_16 storage (see fp32_16_pack.hpp) --
// value is a raw uint32_t bit pattern, not a packed float, but a fill of 0
// (the only value zeros() ever needs) is exactly wh=wr=0, which is a valid
// exact-zero encoding.
template void set_values_impl(uint32_t val, dim<2> size, uint32_t* values,
                              size_t ld);

}  // namespace cpu
}  // namespace tomos

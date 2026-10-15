#include "dimensions.hpp"

namespace tomos {
namespace cpu {

template <typename value_in_t, typename value_out_t>
void strided_copy_kernel(dim<2> size, value_in_t* values_in, size_t ld_in,
                         value_out_t* values_out, size_t ld_out);

}  // namespace cpu
}  // namespace tomos
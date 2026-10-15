#include "device.hpp"
#include "precision.hpp"
#include "set_values_cpu_kernels.hpp"
#if USE_CUDA
#include "set_values_cuda_kernels.cuh"
#endif

namespace tomos {

template <typename value_t>
void set_values_impl(Device device, value_t val, dim<2> size, value_t* values,
                     size_t ld) {
    if (device == Device::CPU) {
        cpu::set_values_impl<value_t>(val, size, values, ld);
    }
#if USE_CUDA
    else if (device == Device::CUDA) {
        cuda::set_values_impl<value_t>(val, size, values, ld);
    }
#endif
}

template void set_values_impl(Device, double, dim<2>, double*, size_t);
template void set_values_impl(Device, float, dim<2>, float*, size_t);
template void set_values_impl(Device, __half, dim<2>, __half*, size_t);
template void set_values_impl(Device, __nv_bfloat16, dim<2>, __nv_bfloat16*, size_t);

template <typename value_t>
void zeros_impl(Device device, dim<2> size, value_t* values, size_t ld) {
    set_values_impl<value_t>(device, static_cast<value_t>(0.0), size, values,
                             ld);
}

template void zeros_impl(Device, dim<2>, double*, size_t);
template void zeros_impl(Device, dim<2>, float*, size_t);
template void zeros_impl(Device, dim<2>, __half*, size_t);
template void zeros_impl(Device, dim<2>, __nv_bfloat16*, size_t);

}  // namespace tomos

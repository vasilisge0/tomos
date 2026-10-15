#include "malloc.hpp"

#include "device.hpp"
#include "precision.hpp"
#include "type_macros.hpp"

#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>

namespace tomos {

template <Device device, typename value_t>
Status malloc(size_t len, value_t** values) {
    if constexpr (device == tomos::Device::CPU) {
        *values = (value_t*)::malloc((sizeof **values) * len);
        return (*values != nullptr) ? tomos::SUCCESS : tomos::FAILURE;
    } else if constexpr (device == tomos::Device::CUDA) {
#if USE_CUDA
        cudaError_t status;
        status = cudaMalloc((void**)values, len * (sizeof **values));
        return (status == 0) ? tomos::SUCCESS : tomos::FAILURE;
#else
        throw std::invalid_argument("Compiled without Device::CUDA.");
#endif
    } else {
        throw std::invalid_argument(
            "Device has to be either tomos::Device::CPU or tomos::Device::CUDA");
    }
}  // malloc

#define MALLOC(device, value_t) \
    template Status malloc<device, value_t>(size_t, value_t**);

FOR_ALL_DEVICES_AND_TYPES(MALLOC)
template Status malloc<tomos::Device::CPU, int>(size_t, int**);
template Status malloc<tomos::Device::CUDA, int>(size_t, int**);
template Status malloc<tomos::Device::CPU, long>(size_t, long**);
template Status malloc<tomos::Device::CUDA, long>(size_t, long**);
template Status malloc<tomos::Device::CPU, std::byte>(size_t, std::byte**);
template Status malloc<tomos::Device::CUDA, std::byte>(size_t, std::byte**);

template <Device device, typename value_t>
Status free(value_t* values) {
    if constexpr (device == tomos::Device::CPU) {
        ::free(values);
        values = nullptr;
        return tomos::SUCCESS;
    } else if constexpr (device == tomos::Device::CUDA) {
#if USE_CUDA
        cudaError_t status = cudaFree((void*)values);
        values = nullptr;
        return (status == 0) ? tomos::SUCCESS : tomos::FAILURE;
#else
        throw std::invalid_argument("Compiled without Device::CUDA.");
#endif
    } else {
        throw std::invalid_argument(
            "Device has to be either tomos::Device::CPU or tomos::Device::CUDA");
    }
}  // free

#define FREE(device, value_t) template Status free<device, value_t>(value_t*);
FOR_ALL_DEVICES_AND_TYPES(FREE)
template Status free<tomos::Device::CPU, int>(int*);
template Status free<tomos::Device::CUDA, int>(int*);
template Status free<tomos::Device::CPU, long>(long*);
template Status free<tomos::Device::CUDA, long>(long*);
template Status free<tomos::Device::CPU, std::byte>(std::byte*);
template Status free<tomos::Device::CUDA, std::byte>(std::byte*);

}  // namespace tomos
#include "workspace.hpp"

#include "malloc.hpp"

namespace tomos {

Workspace::Workspace(Device device, size_t nbytes)
    : device_(device), capacity_(nbytes) {
    switch (device_) {
        case Device::CPU:
            malloc<Device::CPU, std::byte>(nbytes, &data_);
            break;
        case Device::CUDA:
            malloc<Device::CUDA, std::byte>(nbytes, &data_);
            break;
        default:
            break;
    }
}

Workspace::~Workspace() {
    switch (device_) {
        case Device::CPU:
            free<Device::CPU, std::byte>(data_);
            break;
        case Device::CUDA:
            free<Device::CUDA, std::byte>(data_);
            if (mv_transpose_scratch_ != nullptr)
                free<Device::CUDA, float>(mv_transpose_scratch_);
            break;
        default:
            break;
    }
    data_ = nullptr;
    mv_transpose_scratch_          = nullptr;
    mv_transpose_scratch_capacity_ = 0;
}

}  // namespace tomos

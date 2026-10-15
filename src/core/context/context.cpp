#include "context.hpp"

#include <stdexcept>

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "status.hpp"

namespace tomos {

#if defined(USE_CUDA)
namespace {

// Created on first use, never destroyed (leaked at process exit) -- see the
// rationale on cuda_cublas_handle()/cuda_cusparse_handle() in context.hpp.
struct CudaHandles {
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;

    CudaHandles() {
        if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
            perror("Call to cublasCreate failed.\n");
            exit(EXIT_FAILURE);
        }
        if (cusparseCreate(&cusparse) != CUSPARSE_STATUS_SUCCESS) {
            perror("Call to cusparseCreate failed.\n");
            exit(EXIT_FAILURE);
        }
    }
};

CudaHandles& cuda_handles() {
    static CudaHandles handles;
    return handles;
}

}  // namespace

cublasHandle_t cuda_cublas_handle() { return cuda_handles().cublas; }
cusparseHandle_t cuda_cusparse_handle() { return cuda_handles().cusparse; }

CudaContext::CudaContext()
    : cublas_handle_(cuda_cublas_handle()),
      cusparse_handle_(cuda_cusparse_handle()) {}
#endif

Context::Context(Device device) {
    if (device == Device::CUDA) {
#if defined(USE_CUDA)
        device_context_ = CudaContext();
#else
        throw std::invalid_argument(
            "Context: Device::CUDA requested but tomos was built without "
            "CUDA support.");
#endif
    } else {
        device_context_ = CpuContext();
    }
}

Device Context::get_device() {
    return std::visit(
        [](auto&& arg) -> Device {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CpuContext>) {
                return Device::CPU;
            } else if constexpr (std::is_same_v<T, CudaContext>) {
                return Device::CUDA;
            }
        },
        device_context_);
}

#if defined(USE_CUDA)
cublasHandle_t Context::cublas_handle() {
    return std::get<CudaContext>(device_context_).cublas_handle_;
}

cusparseHandle_t Context::cusparse_handle() {
    return std::get<CudaContext>(device_context_).cusparse_handle_;
}
#endif

}  // namespace tomos

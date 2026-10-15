#ifndef TOMOS_CONTEXT_HPP
#define TOMOS_CONTEXT_HPP

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <cusparse.h>
#endif

#include <cstdint>
#include <variant>

#include "device.hpp"

#if defined(USE_CUDA)
#include "context.cuh"
#else
#endif

namespace tomos {

#if defined(USE_CUDA)
// Process-wide cuBLAS/cuSPARSE handles, created once on first use and never
// destroyed (leaked at process exit). Every CudaContext instance shares the
// same pair, so Context stays cheaply copyable (a CudaContext copy just
// copies two pointers) while dispatch code that only has a Device -- not a
// Context -- can still reach the same persistent handles via these
// accessors, without threading a Context through every mv_apply call.
cublasHandle_t cuda_cublas_handle();
cusparseHandle_t cuda_cusparse_handle();
#endif

struct CudaContext {
#if defined(USE_CUDA)
    CudaContext();

    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
#endif
};

struct CpuContext {
    CpuContext() {}

    int num_threads_ = 1;
};

using DeviceContext = std::variant<CpuContext, CudaContext>;

struct Context {
public:
    Context() = delete;

    Context(Device device);

    ~Context() = default;

    Device get_device();

#if defined(USE_CUDA)
    // Throws std::invalid_argument if this Context isn't on Device::CUDA.
    cublasHandle_t cublas_handle();
    cusparseHandle_t cusparse_handle();
#endif

private:
    DeviceContext device_context_;
};

}  // namespace tomos

#endif  // TOMOS_CONTEXT_HPP

#include "sparse_storage.hpp"

#include "copy_cpu_kernels.hpp"
#include "dense_storage.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "malloc.hpp"
#include "precision.hpp"
#include "sparse_storage_cpu_kernels.hpp"
#include "type_macros.hpp"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#include "storage_cuda_kernels.cuh"
#endif

namespace tomos {

template <Device device, typename value_t, typename index_t>
void initialize_csr_storage(dim<2> size, SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* t;
    malloc<device, value_t>(S.nnz_, &t);
    S.values_ = t;

    index_t* rp;
    malloc<device, index_t>(size[0] + 1, &rp);
    S.row_ptrs_ = rp;

    index_t* ci;
    malloc<device, index_t>(S.nnz_, &ci);
    S.col_idxs_ = ci;
}  //  initialize_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(initialize_csr_storage,
                                    (dim<2>, SparseStorage&))

template <Device device, typename value_t, typename index_t>
void free_csr_storage(SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* v = std::get<value_t*>(S.values_);
    if (v != nullptr) free<device, value_t>(v);

    index_t* rp = std::get<index_t*>(S.row_ptrs_);
    if (rp != nullptr) free<device, index_t>(rp);

    index_t* ci = std::get<index_t*>(S.col_idxs_);
    if (ci != nullptr) free<device, index_t>(ci);
}  //  free_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(free_csr_storage, (SparseStorage&))

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t, typename index_in_t, typename index_out_t>
void copy_csr_storage(dim<2> size, SparseStorage& source,
                      SparseStorage& target) {
    CsrStorage& S_source = std::get<CsrStorage>(source);
    CsrStorage& S_target = std::get<CsrStorage>(target);
    if constexpr (device_source == Device::CPU &&
                  device_target == Device::CPU) {
        cpu::csr_copy_impl<value_in_t, value_out_t, index_in_t, index_out_t>(
            size, source, target);
    } else if constexpr (device_source == Device::CUDA &&
                         device_target == Device::CUDA) {
#ifdef __CUDACC__
        cuda::csr_copy_impl<value_in_t, value_out_t, index_in_t, index_out_t>(
            size, source, target);
#else
        throw std::invalid_argument("Compiled without CUDA support.");
#endif
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}  //  copy_csr_storage

#define INST_CSR_COPY_CPU_CPU(vi, vo, idx)                                 \
    template void copy_csr_storage<Device::CPU, Device::CPU, vi, vo, idx,  \
                                   idx>(dim<2>, SparseStorage&,             \
                                        SparseStorage&);

INST_CSR_COPY_CPU_CPU(double, double, int32_t)
INST_CSR_COPY_CPU_CPU(double, double, int64_t)
INST_CSR_COPY_CPU_CPU(double, float, int32_t)
INST_CSR_COPY_CPU_CPU(double, float, int64_t)
INST_CSR_COPY_CPU_CPU(double, __half, int32_t)
INST_CSR_COPY_CPU_CPU(double, __half, int64_t)
INST_CSR_COPY_CPU_CPU(float, double, int32_t)
INST_CSR_COPY_CPU_CPU(float, double, int64_t)
INST_CSR_COPY_CPU_CPU(float, float, int32_t)
INST_CSR_COPY_CPU_CPU(float, float, int64_t)
INST_CSR_COPY_CPU_CPU(float, __half, int32_t)
INST_CSR_COPY_CPU_CPU(float, __half, int64_t)
INST_CSR_COPY_CPU_CPU(__half, double, int32_t)
INST_CSR_COPY_CPU_CPU(__half, double, int64_t)
INST_CSR_COPY_CPU_CPU(__half, float, int32_t)
INST_CSR_COPY_CPU_CPU(__half, float, int64_t)
INST_CSR_COPY_CPU_CPU(__half, __half, int32_t)
INST_CSR_COPY_CPU_CPU(__half, __half, int64_t)
INST_CSR_COPY_CPU_CPU(double, __nv_bfloat16, int32_t)
INST_CSR_COPY_CPU_CPU(double, __nv_bfloat16, int64_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, double, int32_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, double, int64_t)
INST_CSR_COPY_CPU_CPU(float, __nv_bfloat16, int32_t)
INST_CSR_COPY_CPU_CPU(float, __nv_bfloat16, int64_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, float, int32_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, float, int64_t)
INST_CSR_COPY_CPU_CPU(__half, __nv_bfloat16, int32_t)
INST_CSR_COPY_CPU_CPU(__half, __nv_bfloat16, int64_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, __half, int32_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, __half, int64_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, __nv_bfloat16, int32_t)
INST_CSR_COPY_CPU_CPU(__nv_bfloat16, __nv_bfloat16, int64_t)

#undef INST_CSR_COPY_CPU_CPU

}  // namespace tomos
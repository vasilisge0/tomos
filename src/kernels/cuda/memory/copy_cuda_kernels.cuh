#ifndef COPY_CUDA_KERNELS_CUH
#define COPY_CUDA_KERNELS_CUH

#include <cuda_runtime.h>

#include "dimensions.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"

namespace tomos {
namespace cuda {

// template <typename in_value_t, typename out_value_t>
// __host__ Status copy_launcher(dim<2> size, in_value_t* values_in, size_t
// ld_in,
//                               out_value_t* values_out, size_t ld_out);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_impl(dim<2> size, StorageVariant& source,
//                         StorageVariant& target);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_to_device_impl(dim<2> size, StorageVariant& source,
//                                   StorageVariant& target);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_from_device_impl(dim<2> size, StorageVariant& source,
//                                     StorageVariant& target);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_to_device_impl(dim<2> size, StorageVariant& source,
//                                   StorageVariant& target);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_from_device_impl(dim<2> size, StorageVariant& source,
//                                     StorageVariant& target);

template <typename value_t, typename index_t>
void csr_copy_to_device(dim<2> size, SparseStorage& source,
                        SparseStorage& target);

template <typename value_t, typename index_t>
void csr_copy_from_device(dim<2> size, SparseStorage& source,
                          SparseStorage& target);

}  // namespace cuda
}  // namespace tomos

#endif  // COPY_CUDA_KERNELS_CUH
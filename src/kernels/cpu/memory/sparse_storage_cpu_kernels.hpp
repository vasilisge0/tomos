#ifndef TOMOS_CPU_MEMORY_SPARSE_STORAGE_CPU_KERNELS_HPP
#define TOMOS_CPU_MEMORY_SPARSE_STORAGE_CPU_KERNELS_HPP

#include "dimensions.hpp"
#include "malloc.hpp"
#include "sparse_storage.hpp"

namespace tomos {
namespace cpu {

template <typename value_in_t, typename value_out_t, typename index_in_t,
          typename index_out_t>
void csr_copy_impl(dim<2> size, SparseStorage& source, SparseStorage& target);

}  // namespace cpu
}  // namespace tomos

#endif  // TOMOS_CPU_MEMORY_SPARSE_STORAGE_CPU_KERNELS_HPP
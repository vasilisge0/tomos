#ifndef TOMOS_SPARSE_STORAGE_HPP
#define TOMOS_SPARSE_STORAGE_HPP

#include "dense_storage.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "malloc.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"
#include "type_macros.hpp"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace tomos {

struct CsrStorage {
    dim<2> size_ = {0, 0};
    size_t nnz_;
    NumericPtrVariant values_;
    IntPtrVariant row_ptrs_;
    IntPtrVariant col_idxs_;

    CsrStorage() = default;
    CsrStorage(dim<2> size, size_t nnz, NumericType value_type,
               IntType index_type)
        : size_{size}, nnz_{nnz} {
        switch (value_type) {
            case NumericType::FP64: values_.emplace<double*>(nullptr); break;
            case NumericType::FP32: values_.emplace<float*>(nullptr); break;
            case NumericType::FP16: values_.emplace<__half*>(nullptr); break;
            case NumericType::FP32_16: values_.emplace<uint32_t*>(nullptr); break;
            case NumericType::BF16: values_.emplace<__nv_bfloat16*>(nullptr); break;
            default: break;
        }
        switch (index_type) {
            case IntType::INT32: row_ptrs_.emplace<int*>(nullptr);
                                 col_idxs_.emplace<int*>(nullptr); break;
            case IntType::INT64: row_ptrs_.emplace<long int*>(nullptr);
                                 col_idxs_.emplace<long int*>(nullptr); break;
            default: break;
        }
    }
    CsrStorage(dim<2> size, size_t nnz, NumericPtrVariant values,
               IntPtrVariant row_ptrs, IntPtrVariant col_idxs, IntType)
        : size_{size},
          nnz_{nnz},
          values_{values},
          row_ptrs_{row_ptrs},
          col_idxs_{col_idxs} {}
};

enum class SparseFormat : std::size_t { CSR = 1, Count };

using SparseStorage = std::variant<CsrStorage>;

template <Device device, typename value_t, typename index_t>
void initialize_csr_storage(dim<2> size, SparseStorage& source);

template <Device device, typename value_t, typename index_t>
void free_csr_storage(SparseStorage& source);

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t, typename index_in_t, typename index_out_t>
void copy_csr_storage(dim<2> size, SparseStorage& source,
                      SparseStorage& target);

}  // namespace tomos

#endif
#ifndef TOMOS_DENSE_STORAGE_HPP
#define TOMOS_DENSE_STORAGE_HPP

#include "device.hpp"
#include "dimensions.hpp"
#include "precision.hpp"
#include "type_macros.hpp"

namespace tomos {

// Storage objects. These are used to store the actual values of the matrices,
// and do not own memory. They are used only in algorithm implementations
// to extract matix information.

struct StridedStorage {
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    size_t num_elems_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    NumericPtrVariant values_;

    StridedStorage() = default;
    StridedStorage(Layout layout, dim<2> size, size_t lead_dim,
                   NumericType store_type);
    StridedStorage(Layout layout, dim<2> size, size_t lead_dim,
                   NumericPtrVariant values);
};

using DenseStorage = std::variant<StridedStorage>;

template <Device device, typename value_t>
void initialize_strided_storage(dim<2> size, DenseStorage& source);

template <Device device, typename value_t>
void free_strided_storage(DenseStorage& source);

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_storage(dim<2> size, DenseStorage& source,
                          DenseStorage& target);

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t>
void copy_strided_data(dim<2> size, value_in_t* source, size_t ld_source,
                       value_out_t* target, size_t ld_target);

namespace cpu {

template <typename value_type>
void read_strided_storage(char* filename, DenseStorage& source);

template <typename value_type>
void write_strided_storage(const char* filename, DenseStorage& source);

template <typename value_t>
void write_strided_storage(const char* filename, DenseStorage& source,
                           dim<2> col_subset);

}  // namespace cpu

namespace cuda {
template <typename value_t>
void read_strided_storage(char* filename_mtx, DenseStorage& source);

template <typename value_t>
void write_strided_storage(char* filename_mtx, DenseStorage& source);

template <typename value_t>
void write_strided_storage(char* filename_mtx, DenseStorage& source,
                           dim<2> col_subset);
}  // namespace cuda

}  // namespace tomos

#endif  // TOMOS_DENSE_STORAGE_HPP
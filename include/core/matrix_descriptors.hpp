#ifndef TOMOS_MATRIX_DESCRIPTORS_HPP
#define TOMOS_MATRIX_DESCRIPTORS_HPP

#include "dimensions.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"

namespace tomos {

struct StridedDescriptor {
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    NumericType store_type_;

    StridedDescriptor() = delete;
    StridedDescriptor(Layout layout, NumericType store_type, dim<2> size,
                      size_t lead_dim)
        : size_(size), lead_dim_(lead_dim), layout_(layout),
          store_type_(store_type) {}
};

struct CsrDescriptor {
    dim<2> size_ = {0, 0};
    size_t nnz_ = 0;
    NumericType store_type_;
    IntType index_type_;

    CsrDescriptor() = delete;
    CsrDescriptor(dim<2> size, size_t nnz, NumericType store_type,
                  IntType index_type)
        : size_(size), nnz_(nnz), store_type_(store_type),
          index_type_(index_type) {}
};

struct MatrixDescriptor {
    std::variant<StridedDescriptor, CsrDescriptor> storage_descriptor_;

    MatrixDescriptor() = delete;

    MatrixDescriptor(StridedDescriptor dense_descr)
        : storage_descriptor_(dense_descr) {}

    MatrixDescriptor(CsrDescriptor csr_descr)
        : storage_descriptor_(csr_descr) {}

    MatrixFormat get_format() const {
        return static_cast<MatrixFormat>(storage_descriptor_.index());
    }

    dim<2> get_size() const {
        return std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                          storage_descriptor_);
    }
};

}  // namespace tomos

#endif  // TOMOS_MATRIX_DESCRIPTORS_HPP
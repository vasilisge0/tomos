#ifndef TOMOS_MATRIX_HPP
#define TOMOS_MATRIX_HPP

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dense_storage.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "malloc.hpp"
#include "matrix_descriptors.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"
#include "status.hpp"

//  #include "matrix_views.hpp"

// #include "matrix_cpu_kernels.hpp"
// #include "copy_cpu_kernels.hpp"

// Kernel includes moved to matrix.cpp to avoid exposing CUDA syntax in this
// header, which causes clang-format to hang.
// #include "copy_cuda_kernels.cuh"

namespace tomos {

struct Context;

// View objects used to access dense matrices without copying data. These are
// used as arguments to kernels, and are constructed from Dense objects. They do
// not own any data, and are not responsible for deallocating any data.

template <typename value_t>
struct StridedView;

// Dense interface to matrices. Only strided format is currently supported at
// the moment.

struct Dense {
    template <typename value_t>
    friend struct StridedView;

private:
    Device device_;

    DenseStorage storage_;

    // False for Dense objects built via wrap_strided(): the destructor then
    // skips freeing storage_, since it points at memory owned elsewhere.
    bool owns_storage_ = true;

protected:
    Dense(Context& context, MatrixFormat matrix_type, NumericType store_type,
          Layout layout, dim<2> size, size_t lead_dim);

    Dense(Context& context, Layout layout, NumericType store_type,
          std::string& filename);

    Dense(Context& context, MatrixDescriptor descriptor);

    // Non-owning: wraps an externally-owned buffer instead of allocating.
    Dense(Context& context, Layout layout, NumericType store_type, dim<2> size,
          size_t lead_dim, void* data);

public:
    static Dense create_strided(Context& context, Layout layout,
                                NumericType store_precision, dim<2> size);

    static Dense create_strided(Context& context, Layout layout,
                                NumericType store_precision, dim<2> size,
                                size_t lead_dim);

    static Dense create_strided(Context& context, Layout layout,
                                NumericType store_precision,
                                std::string& filename);

    static Dense create(Context& context, MatrixDescriptor descriptor);

    static Dense create_from(Context& context, Dense& source);

    static Dense create_from(Dense& source);

    // Wraps data (a buffer of at least lead_dim * (COL_MAJOR ? size[1] :
    // size[0]) elements of store_precision, already allocated on context's
    // device) as a non-owning Dense: the returned object never allocates or
    // frees data itself, so the caller remains responsible for its lifetime,
    // which must outlive the returned Dense.
    static Dense wrap_strided(Context& context, Layout layout,
                             NumericType store_precision, dim<2> size,
                             size_t lead_dim, void* data);

    Dense(const Dense&) = delete;

    Dense& operator=(const Dense&) = delete;

    Dense(Dense&& other) noexcept;

    Dense& operator=(Dense&& other) noexcept;

    void copy_from(Dense& source);

    MatrixDescriptor get_descriptor() const;

    dim<2> get_size() const;

    MatrixFormat get_format() const;

    Device get_device() const;

    NumericType get_value_type() const;

    DenseStorage& get_storage() { return storage_; }

    const DenseStorage& get_storage() const { return storage_; }

    ~Dense();
};

// Views for dense matrices. Only strided format is currently supported at the
// moment.

template <typename value_t>
StridedView<value_t> as_strided_view(Dense& source) {
    return StridedView<value_t>(source);
}

// Sparse interface. Only CSR format is currently supported.

struct Sparse {
    template <typename value_t, typename index_t>
    friend class CsrView;
    template <typename index_t>
    friend struct CsrSparsityView;

private:
    Device device_;
    SparseStorage storage_;

protected:
    Sparse(Context& context, MatrixDescriptor descriptor);

public:
    static Sparse create_csr(Context& context, NumericType value_type,
                             IntType index_type, dim<2> size, size_t nnz);

    static Sparse create(Context& context, MatrixDescriptor descriptor);

    static Sparse create_from(Context& context, Sparse& source);

    Sparse(const Sparse&) = delete;

    Sparse& operator=(const Sparse&) = delete;

    Sparse(Sparse&& other) noexcept;

    Sparse& operator=(Sparse&& other) noexcept;

    void copy_from(Sparse& source);

    MatrixDescriptor get_descriptor() const;

    dim<2> get_size() const;

    MatrixFormat get_format() const;

    Device get_device() const;

    NumericType get_value_type() const;

    IntType get_index_type() const;

    SparseStorage& get_storage() { return storage_; }
    const SparseStorage& get_storage() const { return storage_; }

    ~Sparse();
};

template <typename value_t>
struct StridedView {
    Device device_;
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    size_t num_elems_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    value_t* values_ = nullptr;

    StridedView(Dense& source);
};

// Views for sparse matrices. Only CSR format is currently supported at the
// moment.

template <typename value_t, typename index_t>
struct CsrView {
    Device device_;
    dim<2> size_;
    size_t nnz_;
    value_t* values_;
    index_t* row_ptrs_;
    index_t* col_idxs_;

    CsrView(Sparse& source)
        : device_(source.device_),
          size_(std::get<CsrStorage>(source.storage_).size_),
          nnz_(std::get<CsrStorage>(source.storage_).nnz_),
          values_(std::get<value_t*>(
              std::get<CsrStorage>(source.storage_).values_)),
          row_ptrs_(std::get<index_t*>(
              std::get<CsrStorage>(source.storage_).row_ptrs_)),
          col_idxs_(std::get<index_t*>(
              std::get<CsrStorage>(source.storage_).col_idxs_)) {}
};

template <typename index_t>
struct CsrSparsityView {
    Device device_;
    dim<2> size_;
    size_t nnz_;
    index_t* row_ptrs_;
    index_t* col_idxs_;

    CsrSparsityView(Sparse& source)
        : device_(source.device_),
          size_(std::get<CsrStorage>(source.storage_).size_),
          nnz_(std::get<CsrStorage>(source.storage_).nnz_),
          row_ptrs_(std::get<index_t*>(
              std::get<CsrStorage>(source.storage_).row_ptrs_)),
          col_idxs_(std::get<index_t*>(
              std::get<CsrStorage>(source.storage_).col_idxs_)) {}
};

template <typename value_t>
struct CsrNzView {
    Device device_;
    dim<2> size_;
    size_t nnz_;

    CsrNzView(Sparse& source);
};

template <typename value_t, typename index_t>
CsrView<value_t, index_t> as_csr_view(Sparse& source) {
    return CsrView<value_t, index_t>(source);
}

using Matrix = std::variant<Dense, Sparse>;

}  // namespace tomos

#endif
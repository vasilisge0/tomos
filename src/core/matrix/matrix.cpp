#include "status.hpp"

#include <iostream>

#include "copy_cpu_kernels.hpp"
#include "matrix.hpp"
#include "context.hpp"
#include "io.hpp"
#include "precision.hpp"

#if USE_CUDA
#include "copy_cuda_kernels.cuh"
#endif
#include "storage_dispatch.hpp"
#include "type_macros.hpp"

namespace tomos {

StridedStorage::StridedStorage(Layout layout, dim<2> size, size_t lead_dim,
                               NumericType store_type)
    : size_{size}, lead_dim_{lead_dim}, layout_{layout} {
    switch (store_type) {
        case NumericType::FP64:
            values_.emplace<double*>(nullptr);
            break;
        case NumericType::FP32:
            values_.emplace<float*>(nullptr);
            break;
        case NumericType::FP16:
            values_.emplace<__half*>(nullptr);
            break;
        case NumericType::FP32_16:
            values_.emplace<uint32_t*>(nullptr);
            break;
        case NumericType::BF16:
            values_.emplace<__nv_bfloat16*>(nullptr);
            break;
        default:
            break;
    }
}

StridedStorage::StridedStorage(Layout layout, dim<2> size, size_t lead_dim,
                               NumericPtrVariant values)
    : size_{size}, lead_dim_{lead_dim}, layout_{layout}, values_{values} {}

template <typename value_t>
StridedView<value_t>::StridedView(Dense& source) {
    StridedStorage storage = std::get<StridedStorage>(source.storage_);
    value_t* values = std::get<value_t*>(storage.values_);
    size_ = storage.size_;
    lead_dim_ = storage.lead_dim_;
    layout_ = storage.layout_;
    values_ = values;
    device_ = source.device_;
    num_elems_ = storage.num_elems_;
}

#define INST_STRIDED_VIEW(value_t) template struct StridedView<value_t>;
FOR_ALL_FP_SINGLES(INST_STRIDED_VIEW)
// NumericType::FP32_16 storage (see fp32_16_pack.hpp) is a plain uint32_t
// buffer, so it needs its own StridedView instantiation alongside the FP
// types above.
INST_STRIDED_VIEW(uint32_t)
#undef INST_STRIDED_VIEW

// Implementation of Dense class API.

Dense::Dense(Context& context, MatrixFormat matrix_type, NumericType store_type,
             Layout layout, dim<2> size, size_t lead_dim)
    : device_{context.get_device()} {
    if (matrix_type == MatrixFormat::STRIDED) {
        storage_ = StridedStorage(layout, size, lead_dim, store_type);
    } else {
        throw std::invalid_argument("Unsupported matrix type.");
    }
    dispatch::initialize_dense_storage(size, context.get_device(), storage_);
}

Dense::Dense(Context& context, Layout layout, NumericType store_type,
            std::string& filename)
    : device_{context.get_device()} {
    dim<2> size = read_mtx_size(filename);
    storage_ = StridedStorage(layout, size, /*lead_dim=*/size[0], store_type);
    dispatch::initialize_dense_storage(size, device_, storage_);
    dispatch::read_dense_storage(device_, storage_, filename);
}

namespace {

NumericPtrVariant wrap_ptr(NumericType store_type, void* data) {
    switch (store_type) {
        case NumericType::FP64:
            return NumericPtrVariant{static_cast<double*>(data)};
        case NumericType::FP32:
            return NumericPtrVariant{static_cast<float*>(data)};
        case NumericType::FP16:
            return NumericPtrVariant{static_cast<__half*>(data)};
        case NumericType::FP32_16:
            return NumericPtrVariant{static_cast<uint32_t*>(data)};
        case NumericType::BF16:
            return NumericPtrVariant{static_cast<__nv_bfloat16*>(data)};
        default:
            throw std::invalid_argument(
                "wrap_strided: unsupported NumericType.");
    }
}

}  // namespace

Dense::Dense(Context& context, Layout layout, NumericType store_type,
            dim<2> size, size_t lead_dim, void* data)
    : device_{context.get_device()}, owns_storage_(false) {
    storage_ = StridedStorage(layout, size, lead_dim,
                              wrap_ptr(store_type, data));
}

Dense::Dense(Context& context, MatrixDescriptor descriptor)
    : device_{context.get_device()} {
    if (descriptor.get_format() == MatrixFormat::STRIDED) {
        StridedDescriptor strided_descr =
            std::get<StridedDescriptor>(descriptor.storage_descriptor_);
        storage_ =
            StridedStorage(strided_descr.layout_, strided_descr.size_,
                           strided_descr.lead_dim_, strided_descr.store_type_);
        dispatch::initialize_dense_storage(strided_descr.size_,
                                           context.get_device(), storage_);
    } else {
        throw std::invalid_argument("Unsupported matrix type.");
    }
}

Dense::Dense(Dense&& other) noexcept
    : device_(other.device_),
      storage_(std::move(other.storage_)),
      owns_storage_(other.owns_storage_) {
    other.storage_ = StridedStorage{};  // empty, destructor won't free anything
    other.owns_storage_ = false;
}

Dense& Dense::operator=(Dense&& other) noexcept {
    if (this != &other) {
        if (owns_storage_)
            dispatch::free_dense_storage(device_,
                                         storage_);  // free our current data
        device_ = other.device_;
        storage_ = std::move(other.storage_);
        owns_storage_ = other.owns_storage_;
        other.storage_ = StridedStorage{};
        other.owns_storage_ = false;
    }
    return *this;
}

Dense Dense::create_strided(Context& context, Layout layout,
                            NumericType store_precision, dim<2> size) {
    return Dense(context, MatrixFormat::STRIDED, store_precision, layout, size,
                 size[0]);
}

Dense Dense::create_strided(Context& context, Layout layout,
                            NumericType store_precision, dim<2> size,
                            size_t lead_dim) {
    return Dense(context, MatrixFormat::STRIDED, store_precision, layout, size,
                 lead_dim);
}

Dense Dense::create_strided(Context& context, Layout layout,
                            NumericType store_precision,
                            std::string& filename) {
    return Dense(context, layout, store_precision, filename);
}

Dense Dense::create(Context& context, MatrixDescriptor descriptor) {
    return Dense(context, descriptor);
}

Dense Dense::create_from(Context& context, Dense& source) {
    Dense target = create(context, source.get_descriptor());
    target.copy_from(source);
    return target;
}

Dense Dense::create_from(Dense& source) {
    Context context = Context(source.get_device());
    Dense target = create(context, source.get_descriptor());
    target.copy_from(source);
    return target;
}

Dense Dense::wrap_strided(Context& context, Layout layout,
                         NumericType store_precision, dim<2> size,
                         size_t lead_dim, void* data) {
    return Dense(context, layout, store_precision, size, lead_dim, data);
}

dim<2> Dense::get_size() const {
    return std::visit([](auto&& arg) -> dim<2> { return arg.size_; }, storage_);
};

MatrixFormat Dense::get_format() const {
    return static_cast<MatrixFormat>(storage_.index());
}

void Dense::copy_from(Dense& source) {
    if (get_format() != source.get_format())
        throw std::invalid_argument(
            "Copying values from matrices of different types");

    dim<2> size = get_size();
    if (size != source.get_size())
        throw std::invalid_argument(
            "Copying from matrix of incompatible size.");

    dispatch::copy_dense_storage(size, source.device_, device_, source.storage_,
                                 storage_);
}

NumericType Dense::get_value_type() const {
    return std::visit(
        [](auto&& arg) -> NumericType {
            return std::visit(
                [](auto&& subarg) -> NumericType {
                    return numeric_type_of<
                        std::remove_pointer_t<std::decay_t<decltype(subarg)>>>;
                },
                arg.values_);
        },
        storage_);
}

Device Dense::get_device() const { return device_; }

MatrixDescriptor Dense::get_descriptor() const {
    NumericType data_type = get_value_type();
    return std::visit(
        [data_type](auto&& arg) -> MatrixDescriptor {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, StridedStorage>) {
                return MatrixDescriptor(StridedDescriptor(
                    arg.layout_, data_type, arg.size_, arg.lead_dim_));
            } else {
                throw std::invalid_argument("Unsupported matrix format.");
            }
        },
        storage_);
}

Dense::~Dense() {
    if (owns_storage_) dispatch::free_dense_storage(device_, storage_);
}

// Implementation of Sparse class API.

Sparse::Sparse(Context& context, MatrixDescriptor descriptor)
    : device_{context.get_device()} {
    std::visit(
        [this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CsrDescriptor>) {
                storage_ = CsrStorage(arg.size_, arg.nnz_, arg.store_type_,
                                      arg.index_type_);
            } else {
                throw std::invalid_argument("Unsupported matrix format.");
            }
        },
        descriptor.storage_descriptor_);
    dispatch::initialize_sparse_storage(device_, storage_);
}

Sparse Sparse::create_csr(Context& context, NumericType value_type,
                          IntType index_type, dim<2> size, size_t nnz) {
    return Sparse(context, MatrixDescriptor(CsrDescriptor(size, nnz, value_type,
                                                          index_type)));
}

Sparse Sparse::create(Context& context, MatrixDescriptor descriptor) {
    return Sparse(context, descriptor);
}

Sparse Sparse::create_from(Context& context, Sparse& source) {
    Sparse target = create(context, source.get_descriptor());
    target.copy_from(source);
    return target;
}

Sparse::Sparse(Sparse&& other) noexcept
    : device_(other.device_), storage_(std::move(other.storage_)) {
    other.storage_ = CsrStorage{};
}

Sparse& Sparse::operator=(Sparse&& other) noexcept {
    if (this != &other) {
        dispatch::free_sparse_storage(device_, storage_);
        device_ = other.device_;
        storage_ = std::move(other.storage_);
        other.storage_ = CsrStorage{};
    }
    return *this;
}

void Sparse::copy_from(Sparse& source) {
    if (get_format() != source.get_format())
        throw std::invalid_argument(
            "Copying values from matrices of different types");

    dim<2> size = get_size();
    if (size != source.get_size())
        throw std::invalid_argument(
            "Copying from matrix of incompatible size.");

    dispatch::copy_sparse_storage(size, source.device_, device_,
                                  source.storage_, storage_);
}

MatrixDescriptor Sparse::get_descriptor() const {
    NumericType value_type = get_value_type();
    IntType index_type = get_index_type();
    return std::visit(
        [value_type, index_type](auto&& arg) -> MatrixDescriptor {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CsrStorage>) {
                return MatrixDescriptor(
                    CsrDescriptor(arg.size_, arg.nnz_, value_type, index_type));
            } else {
                throw std::invalid_argument("Unsupported matrix format.");
            }
        },
        storage_);
}

dim<2> Sparse::get_size() const {
    return std::visit([](auto&& arg) -> dim<2> { return arg.size_; }, storage_);
}

MatrixFormat Sparse::get_format() const {
    return static_cast<MatrixFormat>(storage_.index());
}

Device Sparse::get_device() const { return device_; }

NumericType Sparse::get_value_type() const {
    return std::visit(
        [](auto&& arg) -> NumericType {
            return std::visit(
                [](auto&& subarg) -> NumericType {
                    return numeric_type_of<
                        std::remove_pointer_t<std::decay_t<decltype(subarg)>>>;
                },
                arg.values_);
        },
        storage_);
}

IntType Sparse::get_index_type() const {
    return std::visit(
        [](auto&& arg) -> IntType {
            return static_cast<IntType>(arg.row_ptrs_.index());
        },
        storage_);
}

Sparse::~Sparse() { dispatch::free_sparse_storage(device_, storage_); }

}  // namespace tomos
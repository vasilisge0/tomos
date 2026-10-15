#include "storage_dispatch.hpp"

#include <stdexcept>

#include "dense_registries.hpp"
#include "io.hpp"
#include "matrix_formats.hpp"
#include "sparse_registries.hpp"

namespace tomos {
namespace dispatch {

NumericType get_numeric_type(DenseStorage& source) {
    return static_cast<NumericType>(
        std::get<StridedStorage>(source).values_.index());
}

NumericType get_numeric_type(SparseStorage& source) {
    return static_cast<NumericType>(
        std::get<CsrStorage>(source).values_.index());
}

IntType get_index_type(SparseStorage& source) {
    return static_cast<IntType>(std::get<CsrStorage>(source).row_ptrs_.index());
}

dim<2> get_size(DenseStorage& source) {
    return std::get<StridedStorage>(source).size_;
}

dim<2> get_size(SparseStorage& source) {
    return std::get<CsrStorage>(source).size_;
}

void initialize_dense_storage(dim<2> size, Device dev, DenseStorage& source) {
    auto fn = Dense::get().lookup_initialize(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source));
    fn(size, source);
}

void free_dense_storage(Device dev, DenseStorage& source) {
    auto fn =
        Dense::get().lookup_free(static_cast<MatrixFormat>(source.index()), dev,
                                 get_numeric_type(source));
    fn(source);
}

void copy_dense_storage(dim<2> size, Device source_dev, Device target_dev,
                        DenseStorage& source, DenseStorage& target) {
    auto fn = Dense::get().lookup_copy(
        static_cast<MatrixFormat>(source.index()),
        static_cast<MatrixFormat>(target.index()), source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target));
    fn(size, source, target);
}

void read_dense_storage(Device device, DenseStorage& source,
                        std::string& filename) {
    if (device != Device::CPU)
        throw std::invalid_argument(
            "read_dense_storage: only Device::CPU is currently supported.");

    StridedStorage& S = std::get<StridedStorage>(source);
    switch (get_numeric_type(source)) {
        case NumericType::FP64:
            read_mtx_values<double>(filename, S.size_, S.layout_,
                                    std::get<double*>(S.values_),
                                    S.lead_dim_);
            break;
        case NumericType::FP32:
            read_mtx_values<float>(filename, S.size_, S.layout_,
                                   std::get<float*>(S.values_), S.lead_dim_);
            break;
        case NumericType::FP16:
            read_mtx_values<__half>(filename, S.size_, S.layout_,
                                    std::get<__half*>(S.values_),
                                    S.lead_dim_);
            break;
        case NumericType::BF16:
            read_mtx_values<__nv_bfloat16>(
                filename, S.size_, S.layout_,
                std::get<__nv_bfloat16*>(S.values_), S.lead_dim_);
            break;
        default:
            throw std::invalid_argument(
                "read_dense_storage: unsupported NumericType (the packed "
                "FP32_16 format has no file representation).");
    }
}

// NOT IMPLEMENTED: no write path exists (Dense/Sparse have no public
// write-to-file method), so this stays a no-op stub. See read_dense_storage
// above for the read path, which IS wired up (backs Dense's
// create_strided(..., filename) constructor).
void read_write_storage(Device device, DenseStorage& source,
                        std::string& filename) {}

void initialize_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = Sparse::get().lookup_initialize(
        MatrixFormat::CSR, dev,
        get_numeric_type(source), get_index_type(source));
    fn(get_size(source), source);
}

void free_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = Sparse::get().lookup_free(
        MatrixFormat::CSR, dev,
        get_numeric_type(source), get_index_type(source));
    fn(source);
}

void copy_sparse_storage(dim<2> size, Device source_dev, Device target_dev,
                         SparseStorage& source, SparseStorage& target) {
    auto fn = Sparse::get().lookup_copy(
        MatrixFormat::CSR, MatrixFormat::CSR, source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target),
        get_index_type(source), get_index_type(target));
    fn(size, source, target);
}

}  // namespace dispatch
}  // namespace tomos

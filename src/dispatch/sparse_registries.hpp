#ifndef TOMOS_SPARSE_REGISTRIES_HPP
#define TOMOS_SPARSE_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "storage_registry_base.hpp"
#include "transpose.hpp"

namespace tomos {
namespace dispatch {

using initialize_sparse_fn = void (*)(dim<2> size, SparseStorage& source);
using free_sparse_fn = void (*)(SparseStorage& source);
using copy_sparse_fn = void (*)(dim<2> size, SparseStorage& source,
                                SparseStorage& target);
using set_values_sparse_fn = void (*)(double val, tomos::Sparse& target);
using mv_sparse_fn = void (*)(TransposeType trans, double alpha, tomos::Sparse& A,
                              tomos::Dense& x, double beta, tomos::Dense& y);

struct Sparse {
    // Singleton instance accessor.
    static const Sparse& get();

    // Lookup functions used for dispatching; throw if no matching function is
    // found.
    initialize_sparse_fn lookup_initialize(MatrixFormat format, Device dev,
                                           NumericType numeric_type,
                                           IntType index_type) const;

    free_sparse_fn lookup_free(MatrixFormat format, Device dev,
                               NumericType numeric_type,
                               IntType index_type) const;

    copy_sparse_fn lookup_copy(MatrixFormat source_format,
                               MatrixFormat target_format, Device source_dev,
                               Device target_dev,
                               NumericType source_numeric_type,
                               NumericType target_numeric_type,
                               IntType source_index_type,
                               IntType target_index_type) const;

    set_values_sparse_fn lookup_set_values(Device dev,
                                           NumericType numeric_type) const;

    mv_sparse_fn lookup_mv(Device dev, NumericType type_A, NumericType type_x,
                           NumericType type_y, IntType index_type) const;

private:
    Sparse();

    // Encodings that map dispatch arguments to functions stored in dispatch
    // tables.
    static constexpr std::size_t encode(MatrixFormat format, Device dev,
                                        NumericType numeric_type,
                                        IntType index_type);

    static constexpr std::size_t encode_copy(
        MatrixFormat source_format, MatrixFormat target_format,
        Device source_dev, Device target_dev, NumericType source_numeric_type,
        NumericType target_numeric_type, IntType source_index_type,
        IntType target_index_type);

    static constexpr std::size_t encode_set_values(Device dev,
                                                   NumericType numeric_type);

    static constexpr std::size_t encode_mv(Device dev, NumericType type_A,
                                           NumericType type_x,
                                           NumericType type_y,
                                           IntType index_type);

    // Number of items in dispatch tables (by function). Sized on
    // MatrixFormat::Count (2: STRIDED, CSR), not num_registry_formats (which
    // is DenseFormat::Count == 1) -- Sparse always registers/looks up with
    // MatrixFormat::CSR == 1, which is out of range for the latter. Using
    // the wrong (too-small) basis here previously made every encode()/
    // encode_copy() index land past the end of its array, silently
    // overflowing into (and corrupting) the next data member.
    static constexpr std::size_t N =
        static_cast<std::size_t>(MatrixFormat::Count) * num_registry_devices *
        num_registry_numeric_types * num_registry_index_types;

    static constexpr std::size_t N_copy =
        static_cast<std::size_t>(MatrixFormat::Count) *
        static_cast<std::size_t>(MatrixFormat::Count) * num_registry_devices *
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_index_types *
        num_registry_index_types;

    static constexpr std::size_t N_set_values =
        num_registry_devices * num_registry_numeric_types;

    static constexpr std::size_t N_mv =
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_numeric_types *
        num_registry_index_types;

    // Dispatch tables for functions.
    std::array<initialize_sparse_fn, N> initialize_{};
    std::array<free_sparse_fn, N> free_{};
    std::array<copy_sparse_fn, N_copy> copy_{};
    std::array<set_values_sparse_fn, N_set_values> set_values_{};
    std::array<mv_sparse_fn, N_mv> mv_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_SPARSE_REGISTRIES_HPP

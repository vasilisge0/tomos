#ifndef TOMOS_SKETCH_REGISTRIES_HPP
#define TOMOS_SKETCH_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "status.hpp"
#include "storage_registry_base.hpp"

namespace tomos {
namespace dispatch {

// Type-erased function pointer types for registry entries.
using generate_count_sketch_values_fn = Status (*)(Sparse&);
using generate_count_sketch_idxs_fn = Status (*)(Sparse&);
// target = sketch (CSR) * source (dense)
using apply_count_sketch_dense_fn = void (*)(Sparse&, Dense&, Dense&);
// target = sketch (CSR) * source (CSR), materialized as dense
using apply_count_sketch_sparse_fn = void (*)(Sparse&, Sparse&, Dense&);

struct Sketch {
    // Singleton instance accessor.
    static const Sketch& get();

    // Lookup functions used for dispatching; throw if no matching function is
    // found.
    generate_count_sketch_values_fn lookup_generate_count_sketch_values(
        Device dev, NumericType type) const;

    generate_count_sketch_idxs_fn lookup_generate_count_sketch_idxs(
        Device dev, IntType type) const;

    apply_count_sketch_dense_fn lookup_apply_count_sketch_dense(
        Device dev, NumericType sketch_type, NumericType source_type,
        NumericType target_type, IntType index_type) const;

    apply_count_sketch_sparse_fn lookup_apply_count_sketch_sparse(
        Device dev, NumericType sketch_type, NumericType source_type,
        NumericType target_type, IntType index_type) const;

private:
    Sketch();

    // Encodings that map dispatch arguments to functions stored in dispatch
    // tables.
    static constexpr std::size_t encode_values(Device dev, NumericType type);
    static constexpr std::size_t encode_idxs(Device dev, IntType type);
    static constexpr std::size_t encode_apply(Device dev,
                                              NumericType sketch_type,
                                              NumericType source_type,
                                              NumericType target_type,
                                              IntType index_type);

    // Number of items in dispatch tables (by function).
    static constexpr std::size_t N_values =
        num_registry_devices * num_registry_numeric_types;

    static constexpr std::size_t N_idxs =
        num_registry_devices * num_registry_index_types;

    static constexpr std::size_t N_apply =
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_numeric_types *
        num_registry_index_types;

    // Dispatch tables for functions.
    std::array<generate_count_sketch_values_fn, N_values> generate_values_{};
    std::array<generate_count_sketch_idxs_fn, N_idxs> generate_idxs_{};
    std::array<apply_count_sketch_dense_fn, N_apply> apply_dense_{};
    std::array<apply_count_sketch_sparse_fn, N_apply> apply_sparse_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_SKETCH_REGISTRIES_HPP

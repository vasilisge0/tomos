#ifndef TOMOS_GAUSSIAN_SKETCH_REGISTRIES_HPP
#define TOMOS_GAUSSIAN_SKETCH_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "status.hpp"
#include "storage_registry_base.hpp"

namespace tomos {
namespace dispatch {

using create_gaussian_sketch_dense_fn = Status (*)(Dense&);
using create_gaussian_sketch_sparse_fn = Status (*)(Sparse&);

struct GaussianSketch {
    // Singleton instance accessor.
    static const GaussianSketch& get();

    // Lookup functions used for dispatching; throw if no matching function is
    // found.
    create_gaussian_sketch_dense_fn lookup_create_dense(
        Device dev, NumericType type) const;

    create_gaussian_sketch_sparse_fn lookup_create_sparse(
        Device dev, NumericType type, IntType index_type) const;

private:
    GaussianSketch();

    static constexpr std::size_t encode_dense(Device dev, NumericType type);
    static constexpr std::size_t encode_sparse(Device dev, NumericType type,
                                               IntType index_type);

    static constexpr std::size_t N_dense =
        num_registry_devices * num_registry_numeric_types;

    static constexpr std::size_t N_sparse = num_registry_devices *
                                            num_registry_numeric_types *
                                            num_registry_index_types;

    std::array<create_gaussian_sketch_dense_fn, N_dense> dense_{};
    std::array<create_gaussian_sketch_sparse_fn, N_sparse> sparse_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_GAUSSIAN_SKETCH_REGISTRIES_HPP

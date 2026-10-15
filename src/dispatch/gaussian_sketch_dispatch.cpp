#include "gaussian_sketch_dispatch.hpp"

namespace tomos {
namespace dispatch {

Status create_gaussian_sketch(Dense& S) {
    auto fn = dispatch::GaussianSketch::get().lookup_create_dense(
        S.get_device(), S.get_value_type());
    return fn(S);
}

Status create_gaussian_sketch(Sparse& S) {
    auto fn = dispatch::GaussianSketch::get().lookup_create_sparse(
        S.get_device(), S.get_value_type(), S.get_index_type());
    return fn(S);
}

}  // namespace dispatch
}  // namespace tomos

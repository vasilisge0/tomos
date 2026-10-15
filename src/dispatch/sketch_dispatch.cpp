#include "sketch_dispatch.hpp"

namespace tomos {
namespace dispatch {

void generate_count_sketch_values(Sparse& S) {
    auto fn = dispatch::Sketch::get().lookup_generate_count_sketch_values(
        S.get_device(), S.get_value_type());
    fn(S);
}

void generate_count_sketch_idxs(Sparse& S) {
    auto fn = dispatch::Sketch::get().lookup_generate_count_sketch_idxs(
        S.get_device(), S.get_index_type());
    fn(S);
}

void apply_count_sketch(Sparse& sketch, Dense& source, Dense& target) {
    auto fn = dispatch::Sketch::get().lookup_apply_count_sketch_dense(
        sketch.get_device(), sketch.get_value_type(), source.get_value_type(),
        target.get_value_type(), sketch.get_index_type());
    fn(sketch, source, target);
}

void apply_count_sketch(Sparse& sketch, Sparse& source, Dense& target) {
    auto fn = dispatch::Sketch::get().lookup_apply_count_sketch_sparse(
        sketch.get_device(), sketch.get_value_type(), source.get_value_type(),
        target.get_value_type(), sketch.get_index_type());
    fn(sketch, source, target);
}

}  // namespace dispatch
}  // namespace tomos
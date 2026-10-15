#ifndef TOMOS_SKETCH_DISPATCH_HPP
#define TOMOS_SKETCH_DISPATCH_HPP

#include "sketch_registries.hpp"

namespace tomos {
namespace dispatch {

void generate_count_sketch_values(Sparse& S);

void generate_count_sketch_idxs(Sparse& S);

void apply_count_sketch(Sparse& sketch, Dense& source, Dense& target);

void apply_count_sketch(Sparse& sketch, Sparse& source, Dense& target);

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_SKETCH_DISPATCH_HPP

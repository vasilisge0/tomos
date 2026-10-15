#ifndef TOMOS_GAUSSIAN_SKETCH_DISPATCH_HPP
#define TOMOS_GAUSSIAN_SKETCH_DISPATCH_HPP

#include "gaussian_sketch_registries.hpp"

namespace tomos {
namespace dispatch {

Status create_gaussian_sketch(Dense& S);

Status create_gaussian_sketch(Sparse& S);

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_GAUSSIAN_SKETCH_DISPATCH_HPP

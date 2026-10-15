#include "sketch.hpp"

#include "gaussian_sketch_dispatch.hpp"

namespace tomos {

Status create_gaussian_sketch(Dense& S) {
    return dispatch::create_gaussian_sketch(S);
}

Status create_gaussian_sketch(Sparse& S) {
    return dispatch::create_gaussian_sketch(S);
}

}  // namespace tomos
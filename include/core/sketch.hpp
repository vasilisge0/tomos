#ifndef TOMOS_SKETCH_HPP
#define TOMOS_SKETCH_HPP

#include "device.hpp"
#include "matrix.hpp"
#include "status.hpp"
#include "workspace.hpp"

namespace tomos {

// Fills S in-place with i.i.d. N(0,1) entries (a dense Gaussian random
// projection sketch).
Status create_gaussian_sketch(Dense& S);

// Populates S in-place as a sparse Gaussian sketch: the same nnz_per_row == 1
// structure as generate_count_sketch (one random column per row), but with
// N(0,1) nonzero values instead of +-1.
Status create_gaussian_sketch(Sparse& S);

// NOT YET IMPLEMENTED. Declared but defined nowhere in the codebase, so
// calling either overload would fail to link. Intended to apply an
// already-generated sketch (S, from create_gaussian_sketch) to X, using
// workspace for any scratch space needed -- mirroring apply_count_sketch's
// role for the count-sketch family.
Status apply_gaussian_sketch(Dense& S, Dense& X, Workspace& workspace);

Status apply_gaussian_sketch(Sparse& S, Dense& X, Workspace& workspace);

}  // namespace tomos

#endif  // TOMOS_SKETCH_HPP
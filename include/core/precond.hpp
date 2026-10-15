#ifndef TOMOS_PRECOND_HPP
#define TOMOS_PRECOND_HPP

#include "device.hpp"
#include "status.hpp"
#include "workspace.hpp"

namespace tomos {

// NOT YET IMPLEMENTED. These are declared but have no definition anywhere in
// the codebase, so they cannot currently be called (any use would fail to
// link). They are meant to eventually bundle the multi-step precond pipeline
// that callers currently do by hand -- generate_sketch_qr_precond()
// (src/core/precond/sketch_qr.hpp) to build the R^{-1} factor, followed by
// add_precond() (workspace.hpp) to attach it to a Workspace -- into single
// create/apply calls. Until implemented, use that manual pipeline instead
// (see generate_sketch_qr_precond's own doc comment for the exact steps).

Status create_sketch_qr_precond(Dense& sketch, Dense& mtx, Dense& precond,
                                Workspace& workspace);

Status create_sketch_qr_precond(Sparse& sketch, Dense& mtx, Dense& precond,
                                Workspace& workspace);

Status apply_sketch_qr_precond(Sparse& precond, Dense& rhs,
                               Workspace& workspace);

Status apply_sketch_qr_precond(Dense& precond, Dense& rhs,
                               Workspace& workspace);

}  // namespace tomos

#endif  // TOMOS_PRECOND_HPP
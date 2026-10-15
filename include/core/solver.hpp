#ifndef TOMOS_SOLVER_HPP
#define TOMOS_SOLVER_HPP

#include <vector>

#include "device.hpp"
#include "status.hpp"
#include "transpose.hpp"
#include "workspace.hpp"

namespace tomos {

// Returns a Workspace sized to hold LSQR's working vectors (u/v/w/temp1) in
// value_type. Ready to pass to lsqr_solve() for an unpreconditioned solve.
Workspace create_lsqr_workspace(Context& ctx, dim<2> size,
                                NumericType value_type);

// As above, but also sizes the preconditioner scratch vector (precond_v) in
// precond_type -- independent from value_type, since a preconditioner may
// be stored more compactly than the Krylov vectors it's applied to. Attach
// the preconditioner via add_precond() before solving; add_precond()
// validates the preconditioner's NumericType matches precond_type.
Workspace create_lsqr_workspace(Context& ctx, dim<2> size,
                                NumericType value_type,
                                NumericType precond_type);

// iterations, if non-null, receives the number of LSQR iterations actually
// completed (LsqrWorkspace::completed_iterations) before the solve returned
// -- either because it converged (tol/max_relres satisfied) or hit
// max_iters. Not settable/readable any other way: the internal
// LsqrWorkspace this count lives on is constructed and destroyed entirely
// inside the dispatched implementation, so without this out-parameter a
// caller has no way to distinguish "converged early" from "ran the full
// max_iters" short of re-solving with a different max_iters and comparing.
// residual_history, if non-null, has one relative residual value
// (|phi_bar| / rhsnorm, the same quantity the convergence check itself
// compares against tol/max_relres) appended per completed iteration, in
// order -- see LsqrWorkspace::residual_history_'s own comment
// (src/core/solver/lsqr_solver.hpp). Cleared/left untouched by
// lsqr_solve() itself: pass a fresh empty vector for a single solve's
// history, or a vector you keep appending to across repeated solves if
// that's more useful to you.
//
// restarts (default 1, matching every prior behavior of this function)
// is the number of OUTER restart cycles LSQR runs, each up to max_iters
// INNER iterations: max_iters is therefore "iterations per restart", not
// a total cap -- the true iteration ceiling is restarts * max_iters (and
// *iterations, above, reports how many of those actually ran in total,
// summed across every restart, not just the last one). Each restart
// re-initializes the bidiagonalization from the CURRENT solution x (see
// initialize() in lsqr.cpp, called fresh at the top of every restart),
// the standard technique for keeping a long LSQR run's Krylov basis from
// losing orthogonality/accuracy over very many iterations. Convergence
// (tol/max_relres) is checked both between restarts and after every
// inner iteration, so requesting more restarts than needed to converge
// is harmless -- it simply stops early, same as max_iters already does.
// mtx_refine_precision, if non-null, must be A stored at REFINE_T
// precision (i.e. the same NumericType as b/x -- validated against
// b.get_value_type() and throws on mismatch) and mathematically the SAME
// matrix as A itself, just at higher/different precision. When given,
// the residual recompute at every restart boundary (u = b - A*x inside
// initialize(), see lsqr.cpp) uses THIS copy instead of A's own mv_t
// precision -- so the early-termination decision isn't capped by mv_t's
// own quantization noise, while the bulk of the per-iteration matvecs
// still run at A's (cheaper) mv_t precision, unchanged. Storing/passing
// a second copy of A is the caller's responsibility -- this function
// does not allocate or convert one for you.
void lsqr_solve(Context context, Dense& A, Dense& b, Dense& x,
                Workspace& workspace, double tol, int max_iters,
                double max_relres, int* iterations = nullptr,
                std::vector<double>* residual_history = nullptr,
                int restarts = 1, Dense* mtx_refine_precision = nullptr);

void lsqr_solve(Context context, Sparse& A, Dense& b, Dense& x,
                Workspace& workspace, double tol, int max_iters,
                double max_relres, int* iterations = nullptr,
                std::vector<double>* residual_history = nullptr,
                int restarts = 1, Sparse* mtx_refine_precision = nullptr);

}  // namespace tomos

#endif  // TOMOS_SOLVER_HPP
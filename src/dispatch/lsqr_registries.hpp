#ifndef TOMOS_LSQR_REGISTRIES_HPP
#define TOMOS_LSQR_REGISTRIES_HPP

#include <array>
#include <vector>

#include "context.hpp"
#include "matrix.hpp"
#include "storage_registry_base.hpp"
#include "workspace.hpp"

namespace tomos {
namespace dispatch {

// Trailing int* receives LsqrWorkspace::completed_iterations if non-null,
// trailing std::vector<double>* receives one relative-residual value per
// completed iteration if non-null, trailing int is the number of restart
// cycles (max_iters is iterations PER restart), and the final Dense*/
// Sparse* is an optional refine_t-precision duplicate of A (see
// lsqr_solve's own comment in include/core/solver.hpp).
using lsqr_dense_fn = void (*)(Context context, tomos::Dense& A, tomos::Dense& b,
                                tomos::Dense& x, Workspace& workspace,
                                double tol, int max_iters, double max_relres,
                                int* iterations,
                                std::vector<double>* residual_history,
                                int restarts, tomos::Dense* mtx_refine_precision);

using lsqr_sparse_fn = void (*)(Context context, tomos::Sparse& A,
                                 tomos::Dense& b, tomos::Dense& x,
                                 Workspace& workspace, double tol, int max_iters,
                                 double max_relres, int* iterations,
                                 std::vector<double>* residual_history,
                                 int restarts, tomos::Sparse* mtx_refine_precision);

struct Lsqr {
    static const Lsqr& get();

    // Throws if no kernel is registered for the requested combination.
    // mv_type is A's own NumericType, value_type is the working vectors'
    // (LsqrWorkspace::value_type_), refine_type is b/x's NumericType, and
    // precond_type is the attached preconditioner's NumericType (or
    // value_type when no preconditioner is attached, matching the
    // LsqrWorkspace instantiation actually built for that case).
    lsqr_dense_fn lookup_solve_dense(Device dev, NumericType mv_type,
                                     NumericType value_type,
                                     NumericType refine_type,
                                     NumericType precond_type) const;
    lsqr_sparse_fn lookup_solve_sparse(Device dev, NumericType mv_type,
                                       NumericType value_type,
                                       NumericType refine_type,
                                       NumericType precond_type) const;

private:
    Lsqr();

    static constexpr std::size_t encode(Device dev, NumericType mv_type,
                                        NumericType value_type,
                                        NumericType refine_type,
                                        NumericType precond_type);

    static constexpr std::size_t N =
        num_registry_devices * num_registry_numeric_types *
        num_registry_numeric_types * num_registry_numeric_types *
        num_registry_numeric_types;

    std::array<lsqr_dense_fn, N> dense_{};
    std::array<lsqr_sparse_fn, N> sparse_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_LSQR_REGISTRIES_HPP
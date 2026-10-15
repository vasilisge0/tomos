#ifndef TOMOS_LSQR_IMPL_HPP
#define TOMOS_LSQR_IMPL_HPP

// Reference (single-precision, no-restart) LSQR implementation kept for
// test comparison against the production fused_lsqr_impl in lsqr_workspace.hpp.
// All names are prefixed ref_ / Ref to avoid ODR conflicts.

#include <cmath>
#include <variant>

#include "blas.hpp"
#include "context.hpp"
#include "matrix.hpp"
#include "precision.hpp"

#if defined USE_OPENBLAS
#include "cblas.h"
#endif

namespace tomos {

// ---------------------------------------------------------------------------
// CPU BLAS-1 helpers (reference copies; identical to lsqr_workspace.hpp).
// ---------------------------------------------------------------------------

template <typename T>
inline double ref_nrm2(int n, T* x) {
#if defined USE_OPENBLAS
    if constexpr (std::is_same_v<T, double>) return cblas_dnrm2(n, x, 1);
    if constexpr (std::is_same_v<T, float>)  return cblas_snrm2(n, x, 1);
#endif
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { double v = static_cast<double>(x[i]); acc += v * v; }
    return std::sqrt(acc);
}

template <typename T>
inline void ref_scal(int n, double alpha, T* x) {
#if defined USE_OPENBLAS
    if constexpr (std::is_same_v<T, double>) { cblas_dscal(n, alpha, x, 1); return; }
    if constexpr (std::is_same_v<T, float>)  { cblas_sscal(n, (float)alpha, x, 1); return; }
#endif
    for (int i = 0; i < n; ++i) x[i] = static_cast<T>(static_cast<double>(x[i]) * alpha);
}

template <typename T>
inline void ref_axpy(int n, double alpha, T* x, T* y) {
#if defined USE_OPENBLAS
    if constexpr (std::is_same_v<T, double>) { cblas_daxpy(n, alpha, x, 1, y, 1); return; }
    if constexpr (std::is_same_v<T, float>)  { cblas_saxpy(n, (float)alpha, x, 1, y, 1); return; }
#endif
    for (int i = 0; i < n; ++i)
        y[i] = static_cast<T>(static_cast<double>(y[i]) + alpha * static_cast<double>(x[i]));
}

template <typename T>
inline void ref_copy(int n, T* src, T* dst) {
#if defined USE_OPENBLAS
    if constexpr (std::is_same_v<T, double>) { cblas_dcopy(n, src, 1, dst, 1); return; }
    if constexpr (std::is_same_v<T, float>)  { cblas_scopy(n, src, 1, dst, 1); return; }
#endif
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Reference types (prefixed Ref* to avoid ODR conflict with production names).
// ---------------------------------------------------------------------------

template <typename value_t>
struct RefLsqrScalars {
    value_t alpha   = value_t{0};
    value_t beta    = value_t{0};
    value_t rho_bar = value_t{0};
    value_t phi_bar = value_t{0};
};

template <Device dev, typename value_t>
struct RefLsqrWorkspace {
    Context context_;
    dim<2>  size_;
    int     max_iter_;
    double  tol_;
    double  max_relres_;

    Dense u_d, v_d, w_d, temp1_d;

    tomos::Dense*  rhs_refine_ = nullptr;
    tomos::Dense*  sol_refine_ = nullptr;
    std::variant<tomos::Dense*, tomos::Sparse*> mtx_r_;

    double rhsnorm  = 0.0;
    int    completed_iterations             = 0;
    int    completed_iterations_per_restart = 0;

    RefLsqrWorkspace(Context& ctx, dim<2> size, int max_iter,
                     double tol, double max_relres)
        : context_(ctx), size_(size), max_iter_(max_iter),
          tol_(tol), max_relres_(max_relres),
          u_d(Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {size[0], 1})),
          v_d(Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {size[1], 1})),
          w_d(Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {size[1], 1})),
          temp1_d(Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        numeric_type_of<value_t>, {size[1], 1})) {}
};

// ---------------------------------------------------------------------------
// Reference sub-functions.
// ---------------------------------------------------------------------------

template <Device dev, typename value_t>
inline bool ref_check_stopping_criteria(RefLsqrWorkspace<dev, value_t>* ws,
                                        RefLsqrScalars<value_t>* sc) {
    if (ws->rhsnorm == 0.0) return true;
    double relres = std::abs(static_cast<double>(sc->phi_bar)) / ws->rhsnorm;
    return relres < ws->tol_ || relres < ws->max_relres_;
}

template <Device dev, typename value_t>
inline void ref_initialize(RefLsqrWorkspace<dev, value_t>* ws,
                           RefLsqrScalars<value_t>* sc) {
    int m = static_cast<int>(ws->size_[0]);
    int n = static_cast<int>(ws->size_[1]);
    auto u   = as_strided_view<value_t>(ws->u_d);
    auto v   = as_strided_view<value_t>(ws->v_d);
    auto w   = as_strided_view<value_t>(ws->w_d);
    auto rhs = as_strided_view<value_t>(*ws->rhs_refine_);

    ref_copy(m, rhs.values_, u.values_);
    double beta = ref_nrm2(m, u.values_);
    ref_scal(m, 1.0 / beta, u.values_);
    ws->rhsnorm = beta;

    std::visit([&](auto* mtx) {
        mv_apply(dev, TransposeType::TRANSPOSE, 1.0, *mtx, ws->u_d, 0.0, ws->v_d);
    }, ws->mtx_r_);
    double alpha = ref_nrm2(n, v.values_);
    ref_scal(n, 1.0 / alpha, v.values_);

    ref_copy(n, v.values_, w.values_);

    sc->alpha   = static_cast<value_t>(alpha);
    sc->beta    = static_cast<value_t>(beta);
    sc->rho_bar = static_cast<value_t>(alpha);
    sc->phi_bar = static_cast<value_t>(beta);
}

template <Device dev, typename value_t>
inline void ref_step_1_orth_basis(RefLsqrWorkspace<dev, value_t>* ws,
                                  RefLsqrScalars<value_t>* sc) {
    int m     = static_cast<int>(ws->size_[0]);
    int n     = static_cast<int>(ws->size_[1]);
    auto u     = as_strided_view<value_t>(ws->u_d);
    auto v     = as_strided_view<value_t>(ws->v_d);
    auto temp1 = as_strided_view<value_t>(ws->temp1_d);
    double alpha = static_cast<double>(sc->alpha);
    double beta  = static_cast<double>(sc->beta);

    ref_copy(n, v.values_, temp1.values_);

    std::visit([&](auto* mtx) {
        mv_apply(dev, TransposeType::NO_TRANSPOSE, 1.0, *mtx,
                 ws->temp1_d, -alpha, ws->u_d);
    }, ws->mtx_r_);
    beta = ref_nrm2(m, u.values_);
    ref_scal(m, 1.0 / beta, u.values_);

    std::visit([&](auto* mtx) {
        mv_apply(dev, TransposeType::TRANSPOSE, 1.0, *mtx,
                 ws->u_d, 0.0, ws->temp1_d);
    }, ws->mtx_r_);

    ref_axpy(n, -beta, v.values_, temp1.values_);
    alpha = ref_nrm2(n, temp1.values_);
    ref_scal(n, 1.0 / alpha, temp1.values_);
    ref_copy(n, temp1.values_, v.values_);

    sc->alpha = static_cast<value_t>(alpha);
    sc->beta  = static_cast<value_t>(beta);
}

template <Device dev, typename value_t>
inline void ref_step_2_update_sol(RefLsqrWorkspace<dev, value_t>* ws,
                                  RefLsqrScalars<value_t>* sc) {
    int n = static_cast<int>(ws->size_[1]);
    auto v   = as_strided_view<value_t>(ws->v_d);
    auto w   = as_strided_view<value_t>(ws->w_d);
    auto sol = as_strided_view<value_t>(*ws->sol_refine_);
    double rho_bar = static_cast<double>(sc->rho_bar);
    double phi_bar = static_cast<double>(sc->phi_bar);
    double alpha   = static_cast<double>(sc->alpha);
    double beta    = static_cast<double>(sc->beta);

    double rho   = std::hypot(rho_bar, beta);
    double c     = rho_bar / rho,  s = beta / rho;
    double theta = s * alpha;
    rho_bar      = -c * alpha;
    double phi   = c * phi_bar;
    phi_bar      = s * phi_bar;

    ref_axpy(n, phi / rho, w.values_, sol.values_);
    ref_scal(n, -theta / rho, w.values_);
    ref_axpy(n, 1.0, v.values_, w.values_);

    sc->rho_bar = static_cast<value_t>(rho_bar);
    sc->phi_bar = static_cast<value_t>(phi_bar);
}

// ---------------------------------------------------------------------------
// ref_fused_lsqr_impl: single-restart reference LSQR for test comparison.
// Structurally identical to fused_lsqr_impl in lsqr_workspace.hpp but uses
// Ref* types so the two implementations can coexist in the same test binary
// without ODR conflicts.
// ---------------------------------------------------------------------------

template <Device dev, typename value_t>
void ref_fused_lsqr_impl(RefLsqrWorkspace<dev, value_t>* ws) {
    ws->completed_iterations             = 0;
    ws->completed_iterations_per_restart = 0;

    RefLsqrScalars<value_t> sc;
    ref_initialize<dev, value_t>(ws, &sc);

    for (int iter = 0; iter < ws->max_iter_; ++iter) {
        ref_step_1_orth_basis<dev, value_t>(ws, &sc);
        ref_step_2_update_sol<dev, value_t>(ws, &sc);
        ws->completed_iterations             += 1;
        ws->completed_iterations_per_restart += 1;
        if (ref_check_stopping_criteria(ws, &sc)) return;
    }
}

}  // namespace tomos

#endif  // TOMOS_LSQR_IMPL_HPP
#ifndef TOMOS_LSQR_SOLVER_HPP
#define TOMOS_LSQR_SOLVER_HPP

// Struct definitions and extern-template declarations for the multi-precision
// fused_lsqr_impl.  All implementations live in lsqr_solve.cpp.

#include <variant>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "type_macros.hpp"
#include "workspace.hpp"

namespace tomos {

// ---------------------------------------------------------------------------
// LsqrScalars
// ---------------------------------------------------------------------------

template <typename... Types>
struct LsqrScalars {
    static constexpr int numargs       = sizeof...(Types);
    static constexpr int int_index     = 0;
    static constexpr int refine_index  = 1;
    static constexpr int val_index     = (numargs > 2) ? 2 : 0;
    static constexpr int mv_index      = (numargs > 3) ? 3 : 0;
    static constexpr int precond_index = (numargs > 4) ? 4 : 0;
    using index_t =
        typename std::tuple_element<int_index, std::tuple<Types...>>::type;
    using refine_t =
        typename std::tuple_element<refine_index, std::tuple<Types...>>::type;
    using value_t =
        typename std::tuple_element<val_index, std::tuple<Types...>>::type;
    using mv_t =
        typename std::tuple_element<mv_index, std::tuple<Types...>>::type;
    using precond_t =
        typename std::tuple_element<precond_index, std::tuple<Types...>>::type;

    value_t alpha   = value_t{0};
    value_t beta    = value_t{0};
    value_t rho_bar = value_t{0};
    value_t phi_bar = value_t{0};
};

// ---------------------------------------------------------------------------
// LsqrWorkspace — constructor body is in lsqr_solve.cpp.
// ---------------------------------------------------------------------------

template <Device device, typename... Types>
struct LsqrWorkspace {
    static constexpr int numargs       = sizeof...(Types);
    static constexpr int int_index     = 0;
    static constexpr int refine_index  = 1;
    static constexpr int val_index     = (numargs > 2) ? 2 : 0;
    static constexpr int mv_index      = (numargs > 3) ? 3 : 0;
    static constexpr int precond_index = (numargs > 4) ? 4 : 0;
    using index_t =
        typename std::tuple_element<int_index, std::tuple<Types...>>::type;
    using refine_t =
        typename std::tuple_element<refine_index, std::tuple<Types...>>::type;
    using value_t =
        typename std::tuple_element<val_index, std::tuple<Types...>>::type;
    using mv_t =
        typename std::tuple_element<mv_index, std::tuple<Types...>>::type;
    using precond_t =
        typename std::tuple_element<precond_index, std::tuple<Types...>>::type;

    // Config.
    dim<2>  size_ = {0, 0};
    Context context_;
    int     min_iter_                  = 0;
    int     max_iter_                  = 0;
    int     restarts_                  = 1;
    double  tol_                       = 0.0;
    double  max_relres_                = 0.0;
    bool    use_precond_               = false;
    bool    enable_high_precision_     = false;
    bool    enable_adaptive_precision_ = false;

    // Working vectors (Dense — required by mv_apply). Non-owning views built
    // over a Workspace's arena by whoever constructs this (see
    // dispatch::lsqr_dense_impl/lsqr_sparse_impl).
    Dense u_d, v_d, w_d, temp1_d;
    // Working vector for preconditioner application (n×1, precision precond_t).
    Dense precond_v_d;

    // Raw pointer aliases into the Dense working vectors (for BLAS-1).
    // Set by the constructor after allocating the Dense objects.
    value_t*   u         = nullptr;
    value_t*   v         = nullptr;
    value_t*   w         = nullptr;
    value_t*   temp1     = nullptr;
    precond_t* precond_v = nullptr;

    // Matrix reference (non-owning), at mv_t precision -- used for every
    // per-iteration matvec in step_1_orth_basis.
    std::variant<tomos::Dense*, tomos::Sparse*>                 mtx_r_;
    // Optional SECOND, non-owning reference to the SAME matrix, but at
    // refine_t precision -- set only when the caller passes
    // mtx_refine_precision to lsqr_solve() (see include/core/solver.hpp's
    // own comment) AND mv_t != refine_t (--explicit-duplicates in
    // lsqr_precision_runner.cu). When use_explicit_duplicate_ is true,
    // initialize() uses THIS for the residual recompute (u = b - A*x) at
    // every restart boundary instead of the mv_t-precision mtx_r_ above --
    // so the early-termination decision (and, if the caller reads it back
    // out, the reported residual) isn't itself capped by mv_t's own
    // quantization noise, while the bulk of the per-iteration matvecs in
    // step_1_orth_basis still run at the cheaper mv_t precision unchanged.
    bool use_explicit_duplicate_ = false;
    std::variant<tomos::Dense*, tomos::Sparse*>                 mtx_refine_;
    // True when mtx_r_ itself is stored as NumericType::FP32_16 and the
    // caller wants the restart-boundary residual recompute in initialize()
    // to read THAT SAME buffer at its FP32 ("accurate") compute tier
    // regardless of mv_compute_type_ below -- see
    // Workspace::use_implicit_duplicate_'s own comment (include/core/
    // workspace.hpp), which this is copied from in dispatch::lsqr_dense_impl.
    // Mutually exclusive in practice with use_explicit_duplicate_ (both
    // solve the same "cheap iterate, accurate residual check" problem, via
    // different storage strategies), but nothing enforces that here.
    bool use_implicit_duplicate_ = false;
    // Compute_type actually used for every mv_apply call on mtx_r_ inside
    // step_1_orth_basis (the per-iteration matvecs) -- a RUNTIME value,
    // independent of the compile-time MvT this struct is templated on,
    // since FP32_16-stored A (MvT == uint32_t here) supports two distinct
    // compute tiers (FP16/FP32) a single compile-time axis can't select
    // between. Set by dispatch::lsqr_dense_impl from the caller's
    // Workspace::mv_compute_type_/mv_compute_type_set_ (or a same-as-A
    // default when unset) -- see that field's own comment. For any mv
    // storage type other than FP32_16 this is just that type's own
    // NumericType, matching mv_apply's implicit default, so plumbing it
    // through costs those callers nothing.
    NumericType mv_compute_type_ = NumericType::FP64;
    // Non-owning scratch buffer for the FP32_16 "pure_half" TRANSPOSE
    // kernel's own workspace (see dense_registries.hpp's mv_dense_fn
    // comment). Points into the caller's Workspace::mv_transpose_scratch_
    // (set by dispatch::lsqr_dense_impl, sized/grown once at lsqr_solve()
    // setup time -- never inside step_1_orth_basis's per-iteration loop).
    // Stays null for any A not stored as NumericType::FP32_16; harmless to
    // pass through mv_apply either way (every non-pure_half kernel ignores
    // its own workspace parameter).
    float* mv_transpose_scratch_ = nullptr;
    // Right-preconditioner matrix reference (non-owning).
    // Holds Dense* or Sparse* when use_precond_ is true; monostate otherwise.
    std::variant<tomos::Dense*, tomos::Sparse*, std::monostate> precond_r_ =
        std::monostate{};

    // Problem references (non-owning).
    tomos::Dense* rhs_refine_ = nullptr;
    tomos::Dense* sol_refine_ = nullptr;

    // Convergence state.
    double rhsnorm                          = 0.0;
    double resnorm                          = 0.0;
    double resnorm_previous                 = 0.0;
    int    completed_iterations             = 0;
    int    completed_iterations_per_restart = 0;
    bool   new_restart_active               = false;

    // If non-null, check_stopping_criteria (lsqr.cpp) appends this
    // iteration's relative residual (|phi_bar| / rhsnorm) here every time
    // it's computed -- one entry per completed iteration, in order. Set by
    // the caller via lsqr_solve's own optional residual_history
    // out-parameter (see include/core/solver.hpp); nullptr (the default)
    // means "don't bother recording it" -- no per-iteration cost beyond a
    // null check when unused.
    std::vector<double>* residual_history_ = nullptr;

    // Constructor declared here; body is in lsqr_solve.cpp. u_d/v_d/w_d/
    // temp1_d/precond_v_d must already be Dense views (see
    // Dense::wrap_strided) over externally-owned memory -- LsqrWorkspace
    // itself never allocates.
    LsqrWorkspace(Context& ctx, dim<2> size, int miniter, int maxiter,
                  int restarts, double tol, double max_relres, Dense u_d,
                  Dense v_d, Dense w_d, Dense temp1_d, Dense precond_v_d);
};

// ---------------------------------------------------------------------------
// fused_lsqr_impl — declaration; body is in lsqr_solve.cpp.
// ---------------------------------------------------------------------------

template <Device device, bool fused, bool enable_precision_adaption,
          typename... Types>
void fused_lsqr_impl(LsqrWorkspace<device, Types...>* workspace);

// Extern-template declarations: linker resolves these from the explicit
// instantiations in lsqr.cpp. One per (RefineT, ValueT, MvT, PrecondT)
// combination, per device -- 256 combinations (FOR_ALL_FP_QUADRUPLES, MvT
// ranging over FPTypes) plus another 64 with MvT fixed to uint32_t
// (FOR_ALL_FP_QUADRUPLES_MV_U32 -- NumericType::FP32_16-stored A, see that
// macro's own comment in type_macros.hpp), matching dispatch::Lsqr's
// registry (src/dispatch/lsqr_registries.cpp).
#define EXTERN_INST_LSQR_CPU(RefineT, ValueT, MvT, PrecondT)                \
    extern template struct LsqrWorkspace<Device::CPU, int, RefineT, ValueT, \
                                         MvT, PrecondT>;                    \
    extern template void fused_lsqr_impl<Device::CPU, false, false, int,   \
                                         RefineT, ValueT, MvT, PrecondT>(   \
        LsqrWorkspace<Device::CPU, int, RefineT, ValueT, MvT, PrecondT>*);
FOR_ALL_FP_QUADRUPLES(EXTERN_INST_LSQR_CPU)
FOR_ALL_FP_QUADRUPLES_MV_U32(EXTERN_INST_LSQR_CPU)
#undef EXTERN_INST_LSQR_CPU

#if defined(USE_CUDA)
#define EXTERN_INST_LSQR_CUDA(RefineT, ValueT, MvT, PrecondT)                \
    extern template struct LsqrWorkspace<Device::CUDA, int, RefineT, ValueT, \
                                         MvT, PrecondT>;                     \
    extern template void fused_lsqr_impl<Device::CUDA, false, false, int,   \
                                         RefineT, ValueT, MvT, PrecondT>(    \
        LsqrWorkspace<Device::CUDA, int, RefineT, ValueT, MvT, PrecondT>*);
FOR_ALL_FP_QUADRUPLES(EXTERN_INST_LSQR_CUDA)
FOR_ALL_FP_QUADRUPLES_MV_U32(EXTERN_INST_LSQR_CUDA)
#undef EXTERN_INST_LSQR_CUDA
#endif

}  // namespace tomos

#endif  // TOMOS_LSQR_SOLVER_HPP
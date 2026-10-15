#include "lsqr_registries.hpp"

#include <stdexcept>
#include <string>

#include "lsqr_solver.hpp"
#include "malloc.hpp"
#include "matrix.hpp"
#include "precision.hpp"

#if USE_CUDA
#include "fp32_16_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

namespace {

// Advances offset to the next elem_size-aligned position, reserves nelems
// elements of that size there, and returns a pointer to the reserved slice.
template <typename value_t>
void* carve(Workspace& workspace, size_t& offset, size_t nelems) {
    constexpr size_t elem_size = sizeof(value_t);
    offset = (offset + elem_size - 1) / elem_size * elem_size;
    void* ptr = workspace.data_ + offset;
    offset += nelems * elem_size;
    return ptr;
}

// Slices workspace's arena into offsets for LsqrWorkspace's working vectors
// and wraps each slice as a non-owning Dense (Dense::wrap_strided), rather
// than letting LsqrWorkspace allocate its own memory. LsqrWorkspace itself
// never allocates -- constructing it is just bookkeeping over memory
// create_lsqr_workspace() already sized, so it's cheap to do fresh on every
// lsqr_solve() call. u/v/w/temp1 are carved as ValueT (the general vector
// arithmetic precision); precond_v is carved separately as PrecondT, since
// the preconditioner may be stored more compactly than the Krylov vectors
// it's applied to. RefineT/MvT don't drive any carving here -- they only
// appear so the return type matches LsqrWorkspace's full template
// signature (rhs/sol are refined in RefineT; A is read as MvT elsewhere).
template <Device dev, typename RefineT, typename ValueT, typename MvT,
          typename PrecondT>
LsqrWorkspace<dev, int, RefineT, ValueT, MvT, PrecondT> build_lsqr_workspace(
    Context& ctx, Workspace& workspace, dim<2> size, int miniter, int maxiter,
    int restarts, double tol, double max_relres) {
    if (workspace.device_ != ctx.get_device())
        throw std::invalid_argument(
            "lsqr_solve: workspace and context must be on the same device.");

    size_t m = size[0], n = size[1];
    size_t offset = 0;
    void* u_ptr        = carve<ValueT>(workspace, offset, m);
    void* v_ptr        = carve<ValueT>(workspace, offset, n);
    void* w_ptr        = carve<ValueT>(workspace, offset, n);
    void* temp1_ptr    = carve<ValueT>(workspace, offset, n);
    void* precond_v_ptr = carve<PrecondT>(workspace, offset, n);

    if (offset > workspace.capacity_)
        throw std::invalid_argument(
            "lsqr_solve: workspace is too small for LSQR's working vectors "
            "(need " + std::to_string(offset) + " bytes, have " +
            std::to_string(workspace.capacity_) + ").");

    return LsqrWorkspace<dev, int, RefineT, ValueT, MvT, PrecondT>(
        ctx, size, miniter, maxiter, restarts, tol, max_relres,
        Dense::wrap_strided(ctx, Layout::COL_MAJOR, numeric_type_of<ValueT>,
                            {m, 1}, m, u_ptr),
        Dense::wrap_strided(ctx, Layout::COL_MAJOR, numeric_type_of<ValueT>,
                            {n, 1}, n, v_ptr),
        Dense::wrap_strided(ctx, Layout::COL_MAJOR, numeric_type_of<ValueT>,
                            {n, 1}, n, w_ptr),
        Dense::wrap_strided(ctx, Layout::COL_MAJOR, numeric_type_of<ValueT>,
                            {n, 1}, n, temp1_ptr),
        Dense::wrap_strided(ctx, Layout::COL_MAJOR, numeric_type_of<PrecondT>,
                            {n, 1}, n, precond_v_ptr));
}

}  // namespace

// ---------------------------------------------------------------------------
// Registered implementation: dense A. A is read as MvT (independent of the
// working vectors' ValueT); the constructed workspace refines the solution
// in RefineT and (if a preconditioner is attached) applies it in PrecondT.
// ---------------------------------------------------------------------------
template <Device dev, typename MvT, typename ValueT, typename RefineT,
         typename PrecondT>
static void lsqr_dense_impl(Context context, tomos::Dense& A, tomos::Dense& b,
                            tomos::Dense& x, Workspace& workspace,
                            double tol, int max_iters, double max_relres,
                            int* iterations,
                            std::vector<double>* residual_history,
                            int restarts, tomos::Dense* mtx_refine_precision) {
    auto vA = as_strided_view<MvT>(A);
    auto ws = build_lsqr_workspace<dev, RefineT, ValueT, MvT, PrecondT>(
        context, workspace, vA.size_, /*miniter=*/0, max_iters, restarts,
        tol, max_relres);
    ws.mtx_r_           = &A;
    ws.rhs_refine_      = &b;
    ws.sol_refine_      = &x;
    ws.residual_history_ = residual_history;
    if (mtx_refine_precision) {
        ws.use_explicit_duplicate_ = true;
        ws.mtx_refine_             = mtx_refine_precision;
    }
    if (workspace.use_implicit_duplicate_) {
        if (numeric_type_of<MvT> != NumericType::FP32_16)
            throw std::invalid_argument(
                "lsqr_solve: Workspace::use_implicit_duplicate_ requires A "
                "to be stored as NumericType::FP32_16.");
        ws.use_implicit_duplicate_ = true;
    }
    // Per-iteration matvec compute_type (step_1_orth_basis): the caller's
    // explicit override (Workspace::mv_compute_type_, meaningful only for
    // FP32_16-stored A -- see that field's own comment) if given, else the
    // same default rule mv_apply's own plain overload uses (blas.cpp).
    ws.mv_compute_type_ = workspace.mv_compute_type_set_
        ? workspace.mv_compute_type_
        : ((numeric_type_of<MvT> == NumericType::FP32_16) ? NumericType::FP16
                                                           : numeric_type_of<MvT>);

#if USE_CUDA
    // FP32_16-stored A's TRANSPOSE matvec (mv_pure_half_impl,
    // fp32_16_registries.cpp) needs a scratch workspace. Size/grow it here,
    // once per lsqr_solve() call (cheap: a no-op check on every call after
    // the first, for a fixed-shape Workspace reused across solves), and
    // hand the caller-owned buffer down through every mv_apply TRANSPOSE
    // call below -- so no cudaMalloc/cudaFree ever runs inside
    // step_1_orth_basis's per-iteration timed loop (see that function's own
    // mv_apply calls, and mv_pure_half_impl's fallback path for callers
    // that don't provide a workspace of their own).
    if constexpr (dev == Device::CUDA && numeric_type_of<MvT> == NumericType::FP32_16) {
        size_t needed = cuda::experimental::
            mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                vA.size_);
        if (needed > workspace.mv_transpose_scratch_capacity_) {
            if (workspace.mv_transpose_scratch_ != nullptr)
                free<Device::CUDA, float>(workspace.mv_transpose_scratch_);
            workspace.mv_transpose_scratch_          = nullptr;
            workspace.mv_transpose_scratch_capacity_ = 0;
            malloc<Device::CUDA, float>(needed, &workspace.mv_transpose_scratch_);
            workspace.mv_transpose_scratch_capacity_ = needed;
        }
        ws.mv_transpose_scratch_ = workspace.mv_transpose_scratch_;
    }
#endif

    if (workspace.precond_kind_ == PrecondKind::Dense) {
        ws.use_precond_ = true;
        ws.precond_r_   = static_cast<tomos::Dense*>(workspace.precond_);
    } else if (workspace.precond_kind_ == PrecondKind::Sparse) {
        ws.use_precond_ = true;
        ws.precond_r_   = static_cast<tomos::Sparse*>(workspace.precond_);
    }

    set_values(RefineT{0}, x);
    fused_lsqr_impl<dev, false, false, int, RefineT, ValueT, MvT, PrecondT>(&ws);
    if (iterations) *iterations = ws.completed_iterations;
}

// ---------------------------------------------------------------------------
// Registered implementation: sparse A.
// ---------------------------------------------------------------------------
template <Device dev, typename MvT, typename ValueT, typename RefineT,
         typename PrecondT>
static void lsqr_sparse_impl(Context context, tomos::Sparse& A, tomos::Dense& b,
                             tomos::Dense& x, Workspace& workspace,
                             double tol, int max_iters, double max_relres,
                             int* iterations,
                             std::vector<double>* residual_history,
                             int restarts, tomos::Sparse* mtx_refine_precision) {
    auto vA = as_csr_view<MvT, int>(A);
    auto ws = build_lsqr_workspace<dev, RefineT, ValueT, MvT, PrecondT>(
        context, workspace, vA.size_, /*miniter=*/0, max_iters, restarts,
        tol, max_relres);
    ws.mtx_r_           = &A;
    ws.rhs_refine_      = &b;
    ws.sol_refine_      = &x;
    ws.residual_history_ = residual_history;
    if (mtx_refine_precision) {
        ws.use_explicit_duplicate_ = true;
        ws.mtx_refine_             = mtx_refine_precision;
    }

    if (workspace.precond_kind_ == PrecondKind::Dense) {
        ws.use_precond_ = true;
        ws.precond_r_   = static_cast<tomos::Dense*>(workspace.precond_);
    } else if (workspace.precond_kind_ == PrecondKind::Sparse) {
        ws.use_precond_ = true;
        ws.precond_r_   = static_cast<tomos::Sparse*>(workspace.precond_);
    }

    set_values(RefineT{0}, x);
    fused_lsqr_impl<dev, false, false, int, RefineT, ValueT, MvT, PrecondT>(&ws);
    if (iterations) *iterations = ws.completed_iterations;
}

// ---------------------------------------------------------------------------
// Registry singleton.
// ---------------------------------------------------------------------------

const Lsqr& Lsqr::get() {
    static const Lsqr instance;
    return instance;
}

lsqr_dense_fn Lsqr::lookup_solve_dense(Device dev, NumericType mv_type,
                                       NumericType value_type,
                                       NumericType refine_type,
                                       NumericType precond_type) const {
    auto fn = dense_[encode(dev, mv_type, value_type, refine_type, precond_type)];
    if (!fn)
        throw std::runtime_error(
            "No LSQR dense kernel registered for the requested "
            "device/mv_type/value_type/refine_type/precond_type combination.");
    return fn;
}

lsqr_sparse_fn Lsqr::lookup_solve_sparse(Device dev, NumericType mv_type,
                                         NumericType value_type,
                                         NumericType refine_type,
                                         NumericType precond_type) const {
    auto fn = sparse_[encode(dev, mv_type, value_type, refine_type, precond_type)];
    if (!fn)
        throw std::runtime_error(
            "No LSQR sparse kernel registered for the requested "
            "device/mv_type/value_type/refine_type/precond_type combination.");
    return fn;
}

constexpr std::size_t Lsqr::encode(Device dev, NumericType mv_type,
                                   NumericType value_type,
                                   NumericType refine_type,
                                   NumericType precond_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(mv_type) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types +
           static_cast<std::size_t>(value_type) * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(refine_type) * num_registry_numeric_types +
           static_cast<std::size_t>(precond_type);
}

Lsqr::Lsqr() {
    dense_.fill(nullptr);
    sparse_.fill(nullptr);

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto mv_tag) {
            using MvT = typename decltype(mv_tag)::type;
            for_each_type(FPTypes{}, [&](auto value_tag) {
                using ValueT = typename decltype(value_tag)::type;
                for_each_type(FPTypes{}, [&](auto refine_tag) {
                    using RefineT = typename decltype(refine_tag)::type;
                    for_each_type(FPTypes{}, [&](auto precond_tag) {
                        using PrecondT = typename decltype(precond_tag)::type;
                        auto key = encode(dev, numeric_type_of<MvT>,
                                          numeric_type_of<ValueT>,
                                          numeric_type_of<RefineT>,
                                          numeric_type_of<PrecondT>);
                        dense_[key] =
                            &lsqr_dense_impl<dev, MvT, ValueT, RefineT, PrecondT>;
                        sparse_[key] =
                            &lsqr_sparse_impl<dev, MvT, ValueT, RefineT, PrecondT>;
                    });
                });
            });
        });
    });

    // NumericType::FP32_16 as mv_type: A is stored packed (see
    // fp32_16_pack.hpp) -- MvT is uint32_t, its raw physical element type.
    // MvT only drives as_strided_view<MvT>(A)'s sizing in
    // build_lsqr_workspace above (dims/lead_dim, never a value read) --
    // the actual per-iteration matvecs run through mv_apply(Dense&)'s
    // runtime NumericType dispatch (blas.cpp), which for FP32_16-stored A
    // now routes to the tuned Fp32_16::lookup_mv_pure_half kernel (the
    // GEVM plots' "TOMOS FP16") instead of throwing -- see blas.cpp's own
    // comment. Dense only: no Sparse FP32_16 representation exists, so
    // sparse_ stays unregistered for this mv_type (lookup_solve_sparse
    // throws if ever requested, same "nothing registered" contract every
    // other gap in this codebase already uses).
    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto value_tag) {
            using ValueT = typename decltype(value_tag)::type;
            for_each_type(FPTypes{}, [&](auto refine_tag) {
                using RefineT = typename decltype(refine_tag)::type;
                for_each_type(FPTypes{}, [&](auto precond_tag) {
                    using PrecondT = typename decltype(precond_tag)::type;
                    auto key = encode(dev, NumericType::FP32_16,
                                      numeric_type_of<ValueT>,
                                      numeric_type_of<RefineT>,
                                      numeric_type_of<PrecondT>);
                    dense_[key] =
                        &lsqr_dense_impl<dev, uint32_t, ValueT, RefineT, PrecondT>;
                });
            });
        });
    });
}

}  // namespace dispatch
}  // namespace tomos

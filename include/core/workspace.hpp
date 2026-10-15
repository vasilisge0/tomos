#ifndef TOMOS_WORKSPACE_HPP
#define TOMOS_WORKSPACE_HPP

#include <cstddef>
#include <stdexcept>

#include "device.hpp"
#include "matrix.hpp"

namespace tomos {

// Identifies the concrete type of the object attached via add_precond(), so
// the (type-erased) precond_ pointer can be cast back safely. Mirrors the
// Dense*/Sparse*/monostate variant that solvers (e.g. LsqrWorkspace::precond_r_)
// store their preconditioner reference as -- apply_precond() applies either
// storage kind identically (a plain multiply), regardless of which algorithm
// produced it, so the tag only needs to distinguish storage kind, not
// preconditioner algorithm.
enum class PrecondKind : std::size_t { None = 0, Dense = 1, Sparse = 2 };

template <typename T>
struct PrecondKindTrait;

template <>
struct PrecondKindTrait<tomos::Dense> {
    static constexpr PrecondKind value = PrecondKind::Dense;
};

template <>
struct PrecondKindTrait<tomos::Sparse> {
    static constexpr PrecondKind value = PrecondKind::Sparse;
};

template <typename T>
constexpr PrecondKind precond_kind_of = PrecondKindTrait<T>::value;

struct Workspace {
    Device device_;
    size_t capacity_ = 0;
    std::byte* data_ = nullptr;

    // Non-owning; set via add_precond(). PrecondKind::None means unset.
    PrecondKind precond_kind_ = PrecondKind::None;
    void* precond_ = nullptr;

    // Precision of LSQR's working vectors (value_type_) and preconditioner
    // (precond_type_) -- set by create_lsqr_workspace(); precond_type_ is
    // validated against the precond's own type in add_precond() below since
    // the arena is already sized for it by the time add_precond() runs.
    // (The RHS/solution refinement precision isn't stored here -- it's read
    // straight from b's own NumericType at each lsqr_solve() call, since a
    // workspace can be reused across calls with different b/x.) Meaningless
    // for workspaces not used with lsqr_solve().
    NumericType value_type_   = NumericType::FP64;
    NumericType precond_type_ = NumericType::FP64;

    // True when A is stored as NumericType::FP32_16 (the packed wh/wr
    // format, see fp32_16_pack.hpp) and the caller wants LSQR to exploit
    // BOTH of FP32_16's registered compute tiers from that SINGLE buffer,
    // instead of building a second, separately-stored copy of A (compare
    // --explicit-duplicates / mtx_refine_precision, which needs its own
    // full-size Dense at refine_t precision): the restart-boundary residual
    // recompute in initialize() reads mtx_r_ at its FP32 ("accurate"/
    // hybrid) tier regardless of mv_compute_type_ below -- FP32_16's wr
    // half reconstructs near-FP32 accuracy from data already resident, at
    // no extra memory cost (an "implicit" duplicate, vs
    // --explicit-duplicates' explicit second copy). Set this directly
    // (mirroring how add_precond() below sets precond_kind_/precond_
    // directly rather than through a constructor parameter) before calling
    // lsqr_solve(). Meaningless for workspaces not used with lsqr_solve(),
    // and for any mv storage type other than FP32_16 (lsqr_dense_impl
    // throws if set otherwise -- see src/dispatch/lsqr_registries.cpp).
    bool use_implicit_duplicate_ = false;

    // Compute-precision override for A's PER-ITERATION matvecs
    // (step_1_orth_basis), independent of A's own storage NumericType --
    // meaningful only when A is stored as FP32_16, which has two
    // registered compute tiers (FP16 = "pure_half", the cheap tier; FP32 =
    // "accurate"/hybrid -- see dense_registries.cpp's FP32_16 delegation
    // block); every other mv storage type has only one (its own
    // NumericType), so setting this to anything else there just throws at
    // solve time. mv_compute_type_set_ == false (the default) means "use
    // lsqr_solve's own default rule" (FP16 for FP32_16-stored A, A's own
    // type otherwise -- see mv_apply's own comment in blas.cpp). Set both
    // fields directly, same as use_implicit_duplicate_ above.
    NumericType mv_compute_type_     = NumericType::FP16;
    bool        mv_compute_type_set_ = false;

    // Scratch device buffer (float elements) for the FP32_16 "pure_half"
    // TRANSPOSE kernel's own workspace (see mv_fp32_16_pure_half_transpose_
    // atomiconly_scaled_mblock_vec4_split_workspace_size, fp32_16_cuda_
    // kernels.cu). dispatch::lsqr_dense_impl allocates/grows this once, at
    // lsqr_solve() setup time, then passes it down through every mv_apply
    // TRANSPOSE call on A -- never allocated inside step_1_orth_basis's
    // per-iteration timed loop. Stays null (no allocation at all) unless A
    // is stored as NumericType::FP32_16 and the device is CUDA. Owned by
    // this Workspace (freed in ~Workspace()), so it persists -- and is
    // reused without reallocating -- across repeated lsqr_solve() calls on
    // the same Workspace at the same or smaller shape.
    float* mv_transpose_scratch_          = nullptr;
    size_t mv_transpose_scratch_capacity_ = 0;

    Workspace(Device device, size_t nbytes);

    ~Workspace();
};  // Workspace

// Attaches a non-owning pointer to precond on workspace, tagging it with
// precond's concrete type so consumers can retrieve it safely. Throws if
// precond's NumericType doesn't match what the workspace's arena was sized
// for (see create_lsqr_workspace's precond_type parameter).
template <typename T>
void add_precond(T& precond, Workspace& workspace) {
    if (precond.get_value_type() != workspace.precond_type_)
        throw std::invalid_argument(
            "add_precond: precond's NumericType doesn't match the "
            "precond_type the workspace was created with.");
    workspace.precond_kind_ = precond_kind_of<T>;
    workspace.precond_ = &precond;
}

}  // namespace tomos

#endif  // TOMOS_WORKSPACE_HPP

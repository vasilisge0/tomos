#include "lsqr_registries.hpp"
#include "lsqr_solver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <variant>

#include "matrix.hpp"
#include "precision.hpp"
#include "solver.hpp"
#include "type_macros.hpp"
#include "workspace.hpp"

namespace {
// TEMPORARY diagnostic instrumentation (TOMOS-side note: remove once the
// --implicit-duplicates + --precond divergence is root-caused) -- gated
// behind an env var so it costs a single getenv() call per solve and
// prints nothing for normal use. Enable with TOMOS_LSQR_DEBUG=1.
bool lsqr_debug_enabled() {
    static const bool enabled = std::getenv("TOMOS_LSQR_DEBUG") != nullptr;
    return enabled;
}

// TEMPORARY diagnostic instrumentation (TOMOS-side note: remove once the
// isolated-GEVM-benchmark-vs-real-LSQR gap for the pure_half NO_TRANSPOSE
// kernel is root-caused -- see this session's own investigation, decode.md).
// Strips step_1_orth_basis down to ONLY its two mv_apply calls (no
// precond_apply, no norm/scale/axpy/copy) and makes step_2_update_sol a
// full no-op, so TOMOS_LSQR_TIMING's iter_no_transpose/iter_transpose
// averages reflect JUST the matvec running inside LSQR's own real calling
// context (same stream, same interleaving with the OTHER matvec, same
// memory state) -- with nothing else happening between calls, ruling out
// whether something LSQR-specific (as opposed to the kernel itself) is
// responsible for the gap vs. the isolated benchmark's own number.
// Numerically meaningless once enabled (u/v/temp1 are never renormalized,
// so magnitudes drift unboundedly across iterations -- fine for a pure
// timing run, never for the residual/solution this produces). initialize()
// is NOT gated, so restart-boundary true-residual stopping still works
// normally; only the inner per-iteration cheap check effectively free-
// runs each restart's full max_iter_ (phi_bar never changes once step_2 is
// skipped). Enable with TOMOS_LSQR_GEMV_ONLY=1; pass the SAME --max-iters/
// --restarts as whatever real run you're comparing against, since the
// inner loop no longer exits early on tol.
bool lsqr_gemv_only_enabled() {
    static const bool enabled = std::getenv("TOMOS_LSQR_GEMV_ONLY") != nullptr;
    return enabled;
}
}  // namespace

#if defined USE_OPENBLAS
#include "cblas.h"
#endif

#if defined USE_CUDA
#include "context.hpp"
#include "lsqr_blas1_cuda_kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace {
// TEMPORARY diagnostic instrumentation (TOMOS-side note: remove once the
// --implicit-duplicates vs --explicit-duplicates wall-clock gap is root-
// caused -- see experiments-gemv.md and this session's own investigation).
// Own env var, separate from TOMOS_LSQR_DEBUG's own very verbose per-
// iteration alpha/beta/rho dump, so it can be enabled independently.
// Enable with TOMOS_LSQR_TIMING=1.
//
// Accumulates GPU-side wall time (via CUDA events, NOT host-side
// std::chrono -- mv_apply's own kernel launches are asynchronous, so a
// host-side timer would only measure launch/dispatch overhead, not
// actual kernel execution time) for this solver's four distinct
// mv_apply call sites, split by (once-per-restart, in initialize() vs.
// once-per-iteration, in step_1_orth_basis()) x (NO_TRANSPOSE vs.
// TRANSPOSE) -- exactly the granularity needed to tell whether a real
// wall-clock gap between --implicit-duplicates and --explicit-
// duplicates is coming from the per-restart FP32 residual-recompute
// tier (already retuned this session, ~13 calls for a 63-iteration/5-
// iters-per-restart solve) or the per-iteration FP16 tier (never
// retuned for this exact shape, ~63 calls) -- rather than continuing to
// infer it from isolated GEVM benchmarks, which have twice now failed
// to predict the real number.
bool lsqr_timing_enabled() {
    static const bool enabled = std::getenv("TOMOS_LSQR_TIMING") != nullptr;
    return enabled;
}

struct MvTimingBucket {
    double total_ms = 0.0;
    long count      = 0;
};

MvTimingBucket g_restart_no_transpose;
MvTimingBucket g_restart_transpose;
MvTimingBucket g_iter_no_transpose;
MvTimingBucket g_iter_transpose;

#if defined USE_CUDA
// RAII CUDA-event timer: records "start" at construction, "stop" plus the
// elapsed-time accumulation at destruction. Only touches the CUDA event
// API when lsqr_timing_enabled() is true -- the only cost when disabled
// is the one already-cached getenv() bool check. Forces a
// cudaEventSynchronize() per timed call (needed to read the elapsed time
// back immediately, rather than deferring it) -- this inflates ABSOLUTE
// numbers a little vs. untimed operation, but since --implicit-
// duplicates and --explicit-duplicates pay the identical per-call
// instrumentation overhead, the RELATIVE comparison between them stays
// valid, which is what this diagnostic is actually for.
class ScopedCudaMvTimer {
public:
    explicit ScopedCudaMvTimer(MvTimingBucket* bucket) : bucket_(bucket) {
        if (!lsqr_timing_enabled()) return;
        active_ = true;
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }
    ~ScopedCudaMvTimer() {
        if (!active_) return;
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        bucket_->total_ms += static_cast<double>(ms);
        bucket_->count += 1;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    ScopedCudaMvTimer(const ScopedCudaMvTimer&)            = delete;
    ScopedCudaMvTimer& operator=(const ScopedCudaMvTimer&) = delete;

private:
    MvTimingBucket* bucket_;
    cudaEvent_t start_{}, stop_{};
    bool active_ = false;
};
#endif

// Prints the accumulated per-category timing summary and resets it --
// called once per fused_lsqr_impl exit (see LsqrTimingSummaryGuard
// below), i.e. once per user-facing solve() call.
void lsqr_print_timing_summary() {
    if (!lsqr_timing_enabled()) return;
    auto print_bucket = [](const char* name, const MvTimingBucket& b) {
        std::printf(
            "[lsqr_timing] %-22s calls=%5ld total_ms=%12.3f avg_ms=%9.4f\n",
            name, b.count, b.total_ms, b.count ? b.total_ms / b.count : 0.0);
    };
    print_bucket("restart_no_transpose", g_restart_no_transpose);
    print_bucket("restart_transpose", g_restart_transpose);
    print_bucket("iter_no_transpose", g_iter_no_transpose);
    print_bucket("iter_transpose", g_iter_transpose);
    g_restart_no_transpose = MvTimingBucket{};
    g_restart_transpose    = MvTimingBucket{};
    g_iter_no_transpose    = MvTimingBucket{};
    g_iter_transpose       = MvTimingBucket{};
}

// Scope-exit guard: prints (and resets) the timing summary no matter
// which of fused_lsqr_impl's several return points is taken, without
// needing to duplicate the print call at each one.
struct LsqrTimingSummaryGuard {
    ~LsqrTimingSummaryGuard() { lsqr_print_timing_summary(); }
};
}  // namespace

namespace tomos {

// ---------------------------------------------------------------------------
// Internal BLAS-1 helpers — not exposed outside this TU. Templated on
// `device` because LsqrWorkspace's working vectors live on that device.
// Both CPU and CUDA support any Src/Dst combination (double/float/__half),
// matching the fully independent refine_t/value_t/mv_t/precond_t that
// LsqrWorkspace can now be instantiated with.
// ---------------------------------------------------------------------------

namespace {

// CPU has no native FP16 ALU, so any accumulation touching __half must
// promote to float first (never operate on __half's own, software-emulated
// arithmetic operators) -- compute_t picks float when T (or, for the
// two-type helpers below, Src/Dst) is __half, else keeps the existing
// double accumulation unchanged for every other combination.
template <typename T>
using cpu_compute_t =
    std::conditional_t<std::is_same_v<T, __half>, float, double>;

template <typename Src, typename Dst>
using cpu_compute2_t = std::conditional_t<
    std::is_same_v<Src, __half> || std::is_same_v<Dst, __half>, float, double>;

template <Device device, typename T>
double lsqr_nrm2(int n, T* x) {
    if constexpr (device == Device::CPU) {
#if defined USE_OPENBLAS
        if constexpr (std::is_same_v<T, double>) return cblas_dnrm2(n, x, 1);
        if constexpr (std::is_same_v<T, float>) return cblas_snrm2(n, x, 1);
#endif
        using compute_t = cpu_compute_t<T>;
        compute_t acc   = compute_t{0};
        for (int i = 0; i < n; ++i) {
            compute_t v = static_cast<compute_t>(x[i]);
            acc += v * v;
        }
        return static_cast<double>(std::sqrt(acc));
    }
#if defined USE_CUDA
    else if constexpr (device == Device::CUDA) {
        double result = cuda::lsqr_blas1_nrm2(n, x, cuda_cublas_handle());
        // TOMOS_LSQR_TIMING=1 only (see lsqr_timing_enabled()'s own
        // comment): block after every BLAS-1 call too, not just the
        // mv_apply calls ScopedCudaMvTimer already wraps -- makes the
        // whole per-iteration loop fully synchronous/non-overlapping
        // under instrumentation, so the matvec timing buckets aren't
        // muddied by async overlap with neighboring BLAS-1 work. No
        // effect on normal (uninstrumented) operation.
        if (lsqr_timing_enabled()) cudaDeviceSynchronize();
        return result;
    }
#endif
    else {
        throw std::invalid_argument("lsqr_nrm2: unsupported device.");
    }
}

template <Device device, typename T>
void lsqr_scal(int n, double alpha, T* x) {
    if constexpr (device == Device::CPU) {
#if defined USE_OPENBLAS
        if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(n, alpha, x, 1);
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            cblas_sscal(n, static_cast<float>(alpha), x, 1);
            return;
        }
#endif
        using compute_t   = cpu_compute_t<T>;
        compute_t c_alpha = static_cast<compute_t>(alpha);
        for (int i = 0; i < n; ++i)
            x[i] = static_cast<T>(static_cast<compute_t>(x[i]) * c_alpha);
    }
#if defined USE_CUDA
    else if constexpr (device == Device::CUDA) {
        cuda::lsqr_blas1_scal(n, alpha, x, cuda_cublas_handle());
        if (lsqr_timing_enabled()) cudaDeviceSynchronize();
    }
#endif
    else {
        throw std::invalid_argument("lsqr_scal: unsupported device.");
    }
}

// y ← alpha * x + y  (Src and Dst may differ for mixed-precision paths, CPU
// only -- the CUDA branch requires Src == Dst).
template <Device device, typename Src, typename Dst>
void lsqr_axpy(int n, double alpha, Src* x, Dst* y) {
    if constexpr (device == Device::CPU) {
        if constexpr (std::is_same_v<Src, Dst>) {
#if defined USE_OPENBLAS
            if constexpr (std::is_same_v<Src, double>) {
                cblas_daxpy(n, alpha, x, 1, y, 1);
                return;
            }
            if constexpr (std::is_same_v<Src, float>) {
                cblas_saxpy(n, static_cast<float>(alpha), x, 1, y, 1);
                return;
            }
#endif
        }
        using compute_t   = cpu_compute2_t<Src, Dst>;
        compute_t c_alpha = static_cast<compute_t>(alpha);
        for (int i = 0; i < n; ++i)
            y[i] = static_cast<Dst>(static_cast<compute_t>(y[i]) +
                                    c_alpha * static_cast<compute_t>(x[i]));
    }
#if defined USE_CUDA
    else if constexpr (device == Device::CUDA) {
        cuda::lsqr_blas1_axpy(n, alpha, x, y, cuda_cublas_handle());
        if (lsqr_timing_enabled()) cudaDeviceSynchronize();
    }
#endif
    else {
        throw std::invalid_argument("lsqr_axpy: unsupported device.");
    }
}

// dst ← src  (Src and Dst may differ for mixed-precision paths, CPU only --
// the CUDA branch requires Src == Dst).
template <Device device, typename Src, typename Dst>
void lsqr_copy(int n, Src* src, Dst* dst) {
    if constexpr (device == Device::CPU) {
        if constexpr (std::is_same_v<Src, Dst>) {
#if defined USE_OPENBLAS
            if constexpr (std::is_same_v<Src, double>) {
                cblas_dcopy(n, src, 1, dst, 1);
                return;
            }
            if constexpr (std::is_same_v<Src, float>) {
                cblas_scopy(n, src, 1, dst, 1);
                return;
            }
#endif
        }
        for (int i = 0; i < n; ++i) dst[i] = static_cast<Dst>(src[i]);
    }
#if defined USE_CUDA
    else if constexpr (device == Device::CUDA) {
        cuda::lsqr_blas1_copy(n, src, dst);
        if (lsqr_timing_enabled()) cudaDeviceSynchronize();
    }
#endif
    else {
        throw std::invalid_argument("lsqr_copy: unsupported device.");
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// LsqrWorkspace constructor — u_d/v_d/w_d/temp1_d/precond_v_d arrive
// pre-built (as Dense views over externally-owned memory); this constructor
// only stores them and derives the raw pointer aliases, it never allocates.
// ---------------------------------------------------------------------------

template <Device device, typename... Types>
LsqrWorkspace<device, Types...>::LsqrWorkspace(
    Context& ctx, dim<2> size, int miniter, int maxiter, int restarts,
    double tol, double max_relres, Dense u_d_in, Dense v_d_in, Dense w_d_in,
    Dense temp1_d_in, Dense precond_v_d_in)
    : size_(size),
      context_(ctx),
      min_iter_(miniter),
      max_iter_(maxiter),
      restarts_(restarts),
      tol_(tol),
      max_relres_(max_relres),
      u_d(std::move(u_d_in)),
      v_d(std::move(v_d_in)),
      w_d(std::move(w_d_in)),
      temp1_d(std::move(temp1_d_in)),
      precond_v_d(std::move(precond_v_d_in)) {
    u         = as_strided_view<value_t>(u_d).values_;
    v         = as_strided_view<value_t>(v_d).values_;
    w         = as_strided_view<value_t>(w_d).values_;
    temp1     = as_strided_view<value_t>(temp1_d).values_;
    precond_v = as_strided_view<precond_t>(precond_v_d).values_;
}

// ---------------------------------------------------------------------------
// create_lsqr_workspace helpers.
// ---------------------------------------------------------------------------

namespace {

size_t numeric_type_size(NumericType type) {
    switch (type) {
        case NumericType::FP64:
            return sizeof(double);
        case NumericType::FP32:
            return sizeof(float);
        case NumericType::FP16:
            return sizeof(__half);
        case NumericType::FP32_16:
            return sizeof(uint32_t);
        case NumericType::BF16:
            return sizeof(__nv_bfloat16);
        default:
            throw std::invalid_argument(
                "create_lsqr_workspace: unsupported NumericType.");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// create_lsqr_workspace — public entry point: returns a Workspace sized to
// hold LSQR's working vectors (u: m elements, v/w/temp1: n elements each, in
// value_type) plus the preconditioner scratch vector (precond_v: n elements,
// in precond_type -- independent of value_type). The returned Workspace
// already carries a preconditioner handle (PrecondKind/precond_, see
// workspace.hpp) that add_precond() can fill in before solving; add_precond()
// validates the attached preconditioner's NumericType against precond_type.
// ---------------------------------------------------------------------------

Workspace create_lsqr_workspace(Context& ctx, dim<2> size,
                                NumericType value_type,
                                NumericType precond_type) {
    size_t m = size[0], n = size[1];
    size_t precond_size = numeric_type_size(precond_type);
    // build_lsqr_workspace's carve<PrecondT> rounds the offset preceding
    // precond_v up to a multiple of precond_size, which can add up to
    // (precond_size - 1) bytes of padding beyond the four value_type
    // buffers below it (e.g. value_type=FP16 leaves an offset only
    // 2-byte-aligned, but precond_type=FP64 needs 8-byte alignment).
    size_t nbytes       = m * numeric_type_size(value_type) +
                    3 * n * numeric_type_size(value_type) + (precond_size - 1) +
                    n * precond_size;
    Workspace ws(ctx.get_device(), nbytes);
    ws.value_type_   = value_type;
    ws.precond_type_ = precond_type;
    return ws;
}

Workspace create_lsqr_workspace(Context& ctx, dim<2> size,
                                NumericType value_type) {
    return create_lsqr_workspace(ctx, size, value_type, value_type);
}

// ---------------------------------------------------------------------------
// check_stopping_criteria
// ---------------------------------------------------------------------------

template <Device device, typename... Types>
static bool check_stopping_criteria(LsqrWorkspace<device, Types...>* ws,
                                    LsqrScalars<Types...>* sc) {
    if (ws->completed_iterations == 0) return false;
    if (ws->rhsnorm == 0.0) return true;
    double relres = std::abs(static_cast<double>(sc->phi_bar)) / ws->rhsnorm;
    if (ws->residual_history_) ws->residual_history_->push_back(relres);
    return relres < ws->tol_ || relres < ws->max_relres_;
}

// ---------------------------------------------------------------------------
// initialize — β₁ u₁ = b,  α₁ v₁ = Aᵀ u₁,  w₁ = v₁.
// ---------------------------------------------------------------------------

template <Device device, bool fused, typename... Types>
static void initialize(LsqrWorkspace<device, Types...>* ws,
                       LsqrScalars<Types...>* sc) {
    using value_t = typename LsqrWorkspace<device, Types...>::value_t;
    int m         = static_cast<int>(ws->size_[0]);
    int n         = static_cast<int>(ws->size_[1]);
    auto rhs =
        as_strided_view<typename LsqrWorkspace<device, Types...>::refine_t>(
            *ws->rhs_refine_);

    // u = b - A*x, NOT just b: on the first restart x is still all-zero
    // (lsqr_dense_impl/lsqr_sparse_impl zero it right before calling
    // fused_lsqr_impl), so this reduces to the textbook u = b -- but on
    // restart 2+ x already holds whatever the previous restart(s)
    // accumulated, and re-bidiagonalizing from the raw b (ignoring that)
    // would re-solve the ORIGINAL system from scratch and add a whole
    // new correction on top of the already-good x, overshooting more
    // with every restart. Restarted LSQR (like restarted CG/GMRES) must
    // always restart from the CURRENT residual.
    lsqr_copy<device>(m, rhs.values_, ws->u);
    // Use the refine_t-precision duplicate of A here if the caller gave
    // one (--explicit-duplicates) -- this residual recompute is what the
    // early-termination check right after initialize() (fused_lsqr_impl)
    // and every subsequent restart's own starting point are built from,
    // so computing it against a higher-precision copy of A keeps that
    // decision from being capped by mv_t's own quantization noise, even
    // though the bulk of the per-iteration matvecs below in
    // step_1_orth_basis still run at mv_t (unchanged, for speed).
    auto do_restart_no_transpose = [&]() {
        if (ws->use_explicit_duplicate_) {
            mv_apply(device, TransposeType::NO_TRANSPOSE, -1.0, ws->mtx_refine_,
                     *ws->sol_refine_, 1.0, ws->u_d);
        } else if (ws->use_implicit_duplicate_) {
            // Same FP32_16-packed buffer as mtx_r_ -- just read at its FP32
            // ("accurate") compute tier instead of step_1_orth_basis's cheaper
            // FP16 ("pure_half") tier (see Workspace::use_implicit_duplicate_'s
            // own comment).
            mv_apply(device, TransposeType::NO_TRANSPOSE, -1.0, ws->mtx_r_,
                     *ws->sol_refine_, 1.0, ws->u_d, NumericType::FP32);
        } else {
            mv_apply(device, TransposeType::NO_TRANSPOSE, -1.0, ws->mtx_r_,
                     *ws->sol_refine_, 1.0, ws->u_d);
        }
    };
    if constexpr (device == Device::CUDA) {
        ScopedCudaMvTimer timer(&g_restart_no_transpose);
        do_restart_no_transpose();
    } else {
        do_restart_no_transpose();
    }
    double beta = lsqr_nrm2<device>(m, ws->u);
    lsqr_scal<device>(m, 1.0 / beta, ws->u);

    auto do_restart_transpose = [&]() {
        if (ws->use_precond_) {
            // v = P^T * (A^T * u):  compute A^T*u into temp1_d, then P^T*temp1_d into v_d.
            mv_apply(device, TransposeType::TRANSPOSE, 1.0, ws->mtx_r_, ws->u_d,
                     0.0, ws->temp1_d, ws->mv_compute_type_, ws->mv_transpose_scratch_);
        } else {
            mv_apply(device, TransposeType::TRANSPOSE, 1.0, ws->mtx_r_, ws->u_d,
                     0.0, ws->v_d, ws->mv_compute_type_, ws->mv_transpose_scratch_);
        }
    };
    if constexpr (device == Device::CUDA) {
        ScopedCudaMvTimer timer(&g_restart_transpose);
        do_restart_transpose();
    } else {
        do_restart_transpose();
    }
    if (ws->use_precond_) {
        precond_apply(device, TransposeType::TRANSPOSE, 1.0, ws->precond_r_,
                      ws->temp1_d, 0.0, ws->v_d);
    }
    double alpha = lsqr_nrm2<device>(n, ws->v);
    lsqr_scal<device>(n, 1.0 / alpha, ws->v);

    lsqr_copy<device>(n, ws->v, ws->w);

    sc->alpha   = static_cast<value_t>(alpha);
    sc->beta    = static_cast<value_t>(beta);
    sc->rho_bar = static_cast<value_t>(alpha);
    sc->phi_bar = static_cast<value_t>(beta);
}

// ---------------------------------------------------------------------------
// step_1_orth_basis — extend the bidiagonalization.
// ---------------------------------------------------------------------------

template <Device device, bool fused, bool enable_precision_adaption,
          typename... Types>
static void step_1_orth_basis(LsqrWorkspace<device, Types...>* ws,
                              LsqrScalars<Types...>* sc) {
    using value_t = typename LsqrWorkspace<device, Types...>::value_t;
    int m         = static_cast<int>(ws->size_[0]);
    int n         = static_cast<int>(ws->size_[1]);
    double alpha  = static_cast<double>(sc->alpha);
    double beta;

    if (lsqr_gemv_only_enabled()) {
        // See lsqr_gemv_only_enabled's own comment -- ONLY the two mv_apply
        // calls, same buffers/streams as the real (precond or not) path,
        // nothing else. Numerically meaningless; timing-only.
        Dense& mv_no_transpose_in = ws->use_precond_ ? ws->precond_v_d : ws->temp1_d;
        auto do_iter_no_transpose = [&]() {
            mv_apply(device, TransposeType::NO_TRANSPOSE, 1.0, ws->mtx_r_,
                     mv_no_transpose_in, -alpha, ws->u_d, ws->mv_compute_type_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_no_transpose);
            do_iter_no_transpose();
        } else {
            do_iter_no_transpose();
        }
        auto do_iter_transpose = [&]() {
            mv_apply(device, TransposeType::TRANSPOSE, 1.0, ws->mtx_r_, ws->u_d,
                     0.0, ws->temp1_d, ws->mv_compute_type_, ws->mv_transpose_scratch_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_transpose);
            do_iter_transpose();
        } else {
            do_iter_transpose();
        }
        return;
    }

    if (ws->use_precond_) {
        // u = A*(P*v) - alpha*u
        precond_apply(device, TransposeType::NO_TRANSPOSE, 1.0, ws->precond_r_,
                      ws->v_d, 0.0, ws->precond_v_d);
        auto do_iter_no_transpose = [&]() {
            mv_apply(device, TransposeType::NO_TRANSPOSE, 1.0, ws->mtx_r_,
                     ws->precond_v_d, -alpha, ws->u_d, ws->mv_compute_type_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_no_transpose);
            do_iter_no_transpose();
        } else {
            do_iter_no_transpose();
        }
        beta = lsqr_nrm2<device>(m, ws->u);
        lsqr_scal<device>(m, 1.0 / beta, ws->u);

        // v = P^T*(A^T*u) - beta*v
        auto do_iter_transpose = [&]() {
            mv_apply(device, TransposeType::TRANSPOSE, 1.0, ws->mtx_r_, ws->u_d,
                     0.0, ws->temp1_d, ws->mv_compute_type_, ws->mv_transpose_scratch_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_transpose);
            do_iter_transpose();
        } else {
            do_iter_transpose();
        }
        precond_apply(device, TransposeType::TRANSPOSE, 1.0, ws->precond_r_,
                      ws->temp1_d, 0.0, ws->precond_v_d);
        lsqr_axpy<device>(n, -beta, ws->v, ws->precond_v);
        alpha = lsqr_nrm2<device>(n, ws->precond_v);
        lsqr_scal<device>(n, 1.0 / alpha, ws->precond_v);
        lsqr_copy<device>(n, ws->precond_v, ws->v);
    } else {
        // u = A*v - alpha*u  (copy v to temp1 first so temp1_d is the mv input)
        lsqr_copy<device>(n, ws->v, ws->temp1);
        auto do_iter_no_transpose = [&]() {
            mv_apply(device, TransposeType::NO_TRANSPOSE, 1.0, ws->mtx_r_,
                     ws->temp1_d, -alpha, ws->u_d, ws->mv_compute_type_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_no_transpose);
            do_iter_no_transpose();
        } else {
            do_iter_no_transpose();
        }
        beta = lsqr_nrm2<device>(m, ws->u);
        lsqr_scal<device>(m, 1.0 / beta, ws->u);

        // v = A^T*u - beta*v
        auto do_iter_transpose = [&]() {
            mv_apply(device, TransposeType::TRANSPOSE, 1.0, ws->mtx_r_, ws->u_d,
                     0.0, ws->temp1_d, ws->mv_compute_type_, ws->mv_transpose_scratch_);
        };
        if constexpr (device == Device::CUDA) {
            ScopedCudaMvTimer timer(&g_iter_transpose);
            do_iter_transpose();
        } else {
            do_iter_transpose();
        }
        lsqr_axpy<device>(n, -beta, ws->v, ws->temp1);
        alpha = lsqr_nrm2<device>(n, ws->temp1);
        lsqr_scal<device>(n, 1.0 / alpha, ws->temp1);
        lsqr_copy<device>(n, ws->temp1, ws->v);
    }

    sc->alpha = static_cast<value_t>(alpha);
    sc->beta  = static_cast<value_t>(beta);
}

// ---------------------------------------------------------------------------
// step_2_update_sol — Givens rotation and solution update.
// ---------------------------------------------------------------------------

template <Device device, bool fused, typename... Types>
static void step_2_update_sol(LsqrWorkspace<device, Types...>* ws,
                              LsqrScalars<Types...>* sc) {
    if (lsqr_gemv_only_enabled()) return;
    int n          = static_cast<int>(ws->size_[1]);
    double rho_bar = static_cast<double>(sc->rho_bar);
    double phi_bar = static_cast<double>(sc->phi_bar);
    double alpha   = static_cast<double>(sc->alpha);
    double beta    = static_cast<double>(sc->beta);

    double rho = std::hypot(rho_bar, beta);
    double c = rho_bar / rho, s = beta / rho;
    double theta = s * alpha;
    rho_bar      = -c * alpha;
    double phi   = c * phi_bar;
    phi_bar      = s * phi_bar;

    if (lsqr_debug_enabled()) {
        std::printf(
            "[lsqr_debug] iter=%d alpha=%.9e beta=%.9e rho=%.9e c=%.9e "
            "s=%.9e theta=%.9e phi=%.9e phi_bar=%.9e rho_bar=%.9e "
            "phi/rho=%.9e\n",
            ws->completed_iterations, alpha, beta, rho, c, s, theta, phi,
            phi_bar, rho_bar, phi / rho);
    }

    auto sol =
        as_strided_view<typename LsqrWorkspace<device, Types...>::refine_t>(
            *ws->sol_refine_);

    if (ws->use_precond_) {
        // x += (phi/rho) * P*w  (right-preconditioning: actual search direction is P*w)
        precond_apply(device, TransposeType::NO_TRANSPOSE, 1.0, ws->precond_r_,
                      ws->w_d, 0.0, ws->precond_v_d);
        lsqr_axpy<device>(n, phi / rho, ws->precond_v, sol.values_);
    } else {
        lsqr_axpy<device>(n, phi / rho, ws->w, sol.values_);
    }
    lsqr_scal<device>(n, -theta / rho, ws->w);
    lsqr_axpy<device>(n, 1.0, ws->v, ws->w);

    ws->resnorm_previous = ws->resnorm;
    ws->resnorm          = std::abs(phi_bar);

    using value_t = typename LsqrWorkspace<device, Types...>::value_t;
    sc->rho_bar   = static_cast<value_t>(rho_bar);
    sc->phi_bar   = static_cast<value_t>(phi_bar);
}

// ---------------------------------------------------------------------------
// fused_lsqr_impl — restart loop + inner iteration loop.
// ---------------------------------------------------------------------------

template <Device device, bool fused, bool enable_precision_adaption,
          typename... Types>
void fused_lsqr_impl(LsqrWorkspace<device, Types...>* workspace) {
    using refine_t = typename LsqrWorkspace<device, Types...>::refine_t;

    // Prints (and resets) the TOMOS_LSQR_TIMING summary on every exit path
    // below, no matter which `return` is taken -- see this guard's own
    // definition near the top of this file.
    LsqrTimingSummaryGuard timing_summary_guard;

    auto rhs = as_strided_view<refine_t>(*workspace->rhs_refine_);
    workspace->rhsnorm =
        lsqr_nrm2<device>(static_cast<int>(workspace->size_[0]), rhs.values_);

    workspace->completed_iterations             = 0;
    workspace->completed_iterations_per_restart = 0;

    LsqrScalars<Types...> scalars;
    size_t iteration_count = 0;

    for (int restart = 0; restart < workspace->restarts_; ++restart) {
        workspace->completed_iterations_per_restart = 0;

        if (check_stopping_criteria<device>(workspace, &scalars)) return;

        workspace->new_restart_active = true;
        initialize<device, fused, Types...>(workspace, &scalars);

        // initialize() just recomputed the TRUE residual norm (beta =
        // ||b - A*x||, from an actual matvec against the current
        // solution) as a byproduct of restarting -- not the cheap
        // phi_bar-based estimate the inner loop's own per-iteration
        // checks below use. Check it here, for free (no extra matvec),
        // before committing to a whole restart's worth of iterations:
        // this is a strictly more trustworthy signal than the estimate,
        // since it's the actual residual rather than an accumulated
        // approximation that can drift over many low-precision
        // iterations (see check_stopping_criteria's own comment).
        if (workspace->rhsnorm == 0.0) return;
        double true_relres =
            static_cast<double>(scalars.beta) / workspace->rhsnorm;
        if (true_relres < workspace->tol_ ||
            true_relres < workspace->max_relres_) {
            return;
        }

        for (int iter = 0; iter < workspace->max_iter_; ++iter) {
            step_1_orth_basis<device, fused, enable_precision_adaption,
                              Types...>(workspace, &scalars);
            step_2_update_sol<device, fused, Types...>(workspace, &scalars);
            workspace->completed_iterations += 1;
            workspace->completed_iterations_per_restart += 1;

            if constexpr (enable_precision_adaption) {
                iteration_count += 1;
                // Precision adaptation logic placeholder.
                (void)iteration_count;
            }

            if (check_stopping_criteria<device>(workspace, &scalars)) return;
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations — makes templates available to other TUs via the
// extern template declarations in lsqr_solver.hpp. One instantiation per
// (RefineT, ValueT, MvT, PrecondT) combination, per device -- 256
// combinations (FOR_ALL_FP_QUADRUPLES, MvT ranging over FPTypes) plus
// another 64 with MvT fixed to uint32_t (FOR_ALL_FP_QUADRUPLES_MV_U32 --
// NumericType::FP32_16-stored A), matching dispatch::Lsqr's registry
// (src/dispatch/lsqr_registries.cpp).
// ---------------------------------------------------------------------------

#define INST_LSQR_CPU(RefineT, ValueT, MvT, PrecondT)                      \
    template struct LsqrWorkspace<Device::CPU, int, RefineT, ValueT, MvT,  \
                                  PrecondT>;                               \
    template void fused_lsqr_impl<Device::CPU, false, false, int, RefineT, \
                                  ValueT, MvT, PrecondT>(                  \
        LsqrWorkspace<Device::CPU, int, RefineT, ValueT, MvT, PrecondT>*);
FOR_ALL_FP_QUADRUPLES(INST_LSQR_CPU)
FOR_ALL_FP_QUADRUPLES_MV_U32(INST_LSQR_CPU)
#undef INST_LSQR_CPU

#if defined(USE_CUDA)
#define INST_LSQR_CUDA(RefineT, ValueT, MvT, PrecondT)                      \
    template struct LsqrWorkspace<Device::CUDA, int, RefineT, ValueT, MvT,  \
                                  PrecondT>;                                \
    template void fused_lsqr_impl<Device::CUDA, false, false, int, RefineT, \
                                  ValueT, MvT, PrecondT>(                   \
        LsqrWorkspace<Device::CUDA, int, RefineT, ValueT, MvT, PrecondT>*);
FOR_ALL_FP_QUADRUPLES(INST_LSQR_CUDA)
FOR_ALL_FP_QUADRUPLES_MV_U32(INST_LSQR_CUDA)
#undef INST_LSQR_CUDA
#endif

// ---------------------------------------------------------------------------
// Public API dispatch
// ---------------------------------------------------------------------------

void lsqr_solve(Context context, Dense& A, Dense& b, Dense& x,
                Workspace& workspace, double tol, int max_iters,
                double max_relres, int* iterations,
                std::vector<double>* residual_history, int restarts,
                Dense* mtx_refine_precision) {
    if (mtx_refine_precision &&
        mtx_refine_precision->get_value_type() != b.get_value_type()) {
        throw std::invalid_argument(
            "lsqr_solve: mtx_refine_precision's NumericType must match "
            "b/x's (refine_type).");
    }
    auto fn = dispatch::Lsqr::get().lookup_solve_dense(
        context.get_device(), A.get_value_type(), workspace.value_type_,
        b.get_value_type(), workspace.precond_type_);
    fn(context, A, b, x, workspace, tol, max_iters, max_relres, iterations,
       residual_history, restarts, mtx_refine_precision);
}

void lsqr_solve(Context context, Sparse& A, Dense& b, Dense& x,
                Workspace& workspace, double tol, int max_iters,
                double max_relres, int* iterations,
                std::vector<double>* residual_history, int restarts,
                Sparse* mtx_refine_precision) {
    if (mtx_refine_precision &&
        mtx_refine_precision->get_value_type() != b.get_value_type()) {
        throw std::invalid_argument(
            "lsqr_solve: mtx_refine_precision's NumericType must match "
            "b/x's (refine_type).");
    }
    auto fn = dispatch::Lsqr::get().lookup_solve_sparse(
        context.get_device(), A.get_value_type(), workspace.value_type_,
        b.get_value_type(), workspace.precond_type_);
    fn(context, A, b, x, workspace, tol, max_iters, max_relres, iterations,
       residual_history, restarts, mtx_refine_precision);
}

}  // namespace tomos

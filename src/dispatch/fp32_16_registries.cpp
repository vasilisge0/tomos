#include "fp32_16_registries.hpp"

#include <cstdlib>
#include <stdexcept>

#include "fp32_16_kernels.hpp"
#include "fp32_16_pack.hpp"

#if USE_CUDA
#include <cuda_runtime.h>

#include "fp32_16_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

namespace {

// A Dense's physical value buffer holds lead_dim_ * non_lead_dim elements
// (see initialize_strided_storage in dense_storage.cpp); for FP32_16 that
// full extent -- not just size[0]*size[1] -- is what as_fp32_16_storage
// needs to locate the second (wr) half.
template <typename value_t>
size_t physical_num_elems(StridedView<value_t>& view) {
    size_t non_lead_dim =
        (view.layout_ == Layout::ROW_MAJOR) ? view.size_[0] : view.size_[1];
    return view.lead_dim_ * non_lead_dim;
}

template <Device dev>
void split_fp32_16_impl(tomos::Dense& source, tomos::Dense& target) {
    auto src = as_strided_view<float>(source);
    auto tgt = as_strided_view<uint32_t>(target);
    Fp32_16Storage s = as_fp32_16_storage(tgt.values_, physical_num_elems(tgt));
    if constexpr (dev == Device::CPU) {
        cpu::split_dense_fp32_16(src.size_, src.values_, src.lead_dim_, s.wh,
                                 s.wr, tgt.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::split_dense_fp32_16(src.size_, src.values_, src.lead_dim_, s.wh,
                                  s.wr, tgt.lead_dim_);
    }
#endif
}

template <Device dev>
void combine_fp32_16_impl(tomos::Dense& source, tomos::Dense& target) {
    auto src = as_strided_view<uint32_t>(source);
    auto tgt = as_strided_view<float>(target);
    Fp32_16Storage s = as_fp32_16_storage(src.values_, physical_num_elems(src));
    if constexpr (dev == Device::CPU) {
        cpu::combine_dense_fp32_16(tgt.size_, s.wh, s.wr, src.lead_dim_,
                                   tgt.values_, tgt.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::combine_dense_fp32_16(tgt.size_, s.wh, s.wr, src.lead_dim_,
                                    tgt.values_, tgt.lead_dim_);
    }
#endif
}

template <Device dev>
void mv_pure_half_impl(TransposeType trans, double alpha, tomos::Dense& A,
                       tomos::Dense& x, double beta, tomos::Dense& y,
                       float* caller_workspace) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vx = as_strided_view<float>(x);
    auto vy = as_strided_view<float>(y);
    uint16_t* wh = reinterpret_cast<uint16_t*>(vA.values_);
    if constexpr (dev == Device::CPU) {
        cpu::mv_fp32_16_pure_half(trans, alpha, vA.size_, wh, vA.lead_dim_,
                                  vx.values_, vx.lead_dim_, beta, vy.values_,
                                  vy.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        // Runtime decode toggle, gated behind an env var so it costs a
        // single getenv() call per matvec and defaults to the safe decode
        // for normal use -- same "env-var-gated, off by default" pattern
        // lsqr.cpp's own TOMOS_LSQR_DEBUG uses. Exists purely so
        // lsqr_precision_runner's --implicit-duplicates comparison (see
        // scripts/run_lsqr_precision_compare.sh) can show BOTH the safe-
        // and fast-decode curves without threading a new CLI flag/
        // Workspace field/dispatch-registry axis through the whole solver
        // for a one-off diagnostic. Set TOMOS_FP32_16_PURE_HALF_FAST_
        // DECODE=1 to select the fast-decode sibling (fp32_16_half_only_
        // fast) instead of the default safe one (fp32_16_half_only) --
        // see this function's own git history for why fast decode is NOT
        // the default here (it stalled LSQR convergence on a real
        // ill-conditioned matrix).
        static const bool use_fast_decode =
            std::getenv("TOMOS_FP32_16_PURE_HALF_FAST_DECODE") != nullptr;
        if (trans == TransposeType::NO_TRANSPOSE) {
            if (use_fast_decode) {
                // cuda::mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(
                //     trans, alpha, vA.size_, wh, vA.lead_dim_, vx.values_,
                //     vx.lead_dim_, beta, vy.values_, vy.lead_dim_);
                //
                // NOW WIRED to mv_fp32_16_pure_half_no_transpose_flush256_fast_swar
                // (fp32_16_cuda_kernels.cu): same LOSSY fast decode as the call
                // above (underflow band -> 0, overflow -> Inf), but at M >=
                // kMvWideRowThreshold it runs the narrow_vec kernel with
                // float flush every 256 columns and the SWAR pair decode; below
                // the threshold it still dispatches to the fast wide-ilp kernel
                // commented out above. Previous call kept for A/B revert.
                cuda::mv_fp32_16_pure_half_no_transpose_flush256_fast_swar(
                    alpha, vA.size_, wh, vA.lead_dim_, vx.values_, vx.lead_dim_, beta, vy.values_, vy.lead_dim_);
            } else {
                // Safe decode -- was mv_fp32_16_pure_half_wide_ilp_shared_
                // half2_atomic_best (matching plot_gemv_fp32_16_pure_half_
                // vs_cublas_ref.m's own "TOMOS FP16 (implicit)" series,
                // half2_atomic_best_gflops), swapped for its cp.async +
                // warp-uniform-issuer sibling ("test2" in that same plot,
                // now promoted to "TOMOS FP32 / FP16" there) once it
                // measured faster on real A100 hardware -- SAME
                // fp32_16_half_only safe decode and __hfma2 accumulation,
                // only the wh-staging mechanism differs (cp.async shared-
                // memory staging instead of ordinary register-mediated
                // global loads), so this carries none of mv_accurate_
                // impl's own hybrid-family accuracy risk above (that
                // family skips wr entirely; this one still decodes
                // through the same wh-only-but-complete reconstruction
                // mv_pure_half_impl always used here).
                //
                // NOW WIRED to mv_fp32_16_pure_half_no_transpose_fast2 (a
                // composed dispatcher, fp32_16_cuda_kernels.cu), not
                // cpasync_warpuniform directly -- a real, per-call LSQR
                // timing breakdown (TOMOS_LSQR_TIMING=1) found this exact
                // per-iteration matvec running ~1.6x slower under
                // --implicit-duplicates than --explicit-duplicates at the
                // real LSQR matrix's shape (M=1,000,000, K=1000), because
                // cpasync_warpuniform's own cp.async optimization only
                // lives inside its rows_op < kMvWideRowThreshold branch --
                // the SAME class of "the fix doesn't apply at this shape"
                // gap already found and fixed once this session for the
                // FP32 tier (see mv_fp32_16_accurate_fast's own comment
                // above). The composed dispatcher keeps cpasync_
                // warpuniform's own proven win for rows_op < threshold and
                // routes to fp32_16_half_only_fast2 (BIT-EXACT with the
                // safe fp32_16_half_only -- see that function's own
                // comment in fp32_16_pack.hpp -- not the same axis as this
                // env-var toggle's own lossy "fast" path) for rows_op >=
                // threshold, i.e. the shape that actually matters here.
                // cuda::mv_fp32_16_pure_half_no_transpose_fast2(alpha, vA.size_, wh, vA.lead_dim_, vx.values_,
                //                                                vx.lead_dim_, beta, vy.values_, vy.lead_dim_);
                //
                // NOW WIRED to mv_fp32_16_pure_half_no_transpose_flush256_swar
                // (fp32_16_cuda_kernels.cu): same fast2-shaped dispatch as the
                // call above (cpasync_warpuniform below kMvWideRowThreshold,
                // narrow_vec at/above it), but the narrow_vec branch (a) adds
                // its __half2 accumulators into float every 256 columns
                // (bounds the fp16 accumulation chain -- the unflushed kernel
                // accumulates all n columns in half2, so its error grows with
                // n) and (b) decodes two wh per 32-bit register (SWAR,
                // bit-exact with the scalar decode on all 2^32 pairs) to
                // cut the ALU-bound decode's instruction count. The
                // cpasync_warpuniform branch (M < kMvWideRowThreshold) is
                // unchanged. Previous call kept commented out above for
                // easy A/B revert.
                cuda::mv_fp32_16_pure_half_no_transpose_flush256_swar(alpha, vA.size_, wh, vA.lead_dim_, vx.values_,
                                                                      vx.lead_dim_, beta, vy.values_, vy.lead_dim_);
            }
        } else {
            // TRANSPOSE needs a scratch buffer (see _workspace_size below).
            // Preferred path: the caller (e.g. dispatch::lsqr_dense_impl,
            // via Workspace::mv_transpose_scratch_) already owns a
            // persistently-sized buffer and passes it in directly -- no
            // allocation happens here at all in that case. LSQR sizes and
            // grows that buffer once at lsqr_solve() setup time (see
            // lsqr_registries.cpp's build_lsqr_workspace/lsqr_dense_impl),
            // not inside step_1_orth_basis's per-iteration timed loop.
            //
            // Fallback path (caller_workspace == nullptr): callers that
            // don't own a workspace of their own (raw benchmarks,
            // lsqr_gemv_isolated.cu, mv_apply_fp32_16_pure_half's own
            // convenience wrapper) still get correct behavior via a
            // function-local static that grows on demand and is freed only
            // at process exit (the CUDA driver reclaims device memory then
            // regardless -- same accepted process-lifetime-cache tradeoff
            // other statics in this codebase already make, e.g.
            // gaussian_sketch_kernels.cpp's own static thread_local
            // generator). Both safe and fast-decode kernels share the same
            // workspace-size function (decode-independent) and this same
            // buffer.
            float* ws = caller_workspace;
            if (ws == nullptr) {
                static float* fallback_ws = nullptr;
                static size_t fallback_ws_capacity = 0;
                size_t ws_elems = cuda::experimental::
                    mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(
                        vA.size_);
                if (ws_elems > fallback_ws_capacity) {
                    if (fallback_ws != nullptr) cudaFree(fallback_ws);
                    fallback_ws = nullptr;
                    fallback_ws_capacity = 0;
                    cudaError_t err = cudaMalloc(&fallback_ws, ws_elems * sizeof(float));
                    if (err != cudaSuccess) {
                        cudaGetLastError();
                        throw std::runtime_error(
                            "mv_pure_half_impl: cudaMalloc failed for the "
                            "transpose kernel's persistent scratch workspace.");
                    }
                    fallback_ws_capacity = ws_elems;
                }
                ws = fallback_ws;
            }
            if (use_fast_decode) {
                cuda::mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split(
                    trans, alpha, vA.size_, wh, vA.lead_dim_, vx.values_,
                    vx.lead_dim_, beta, vy.values_, vy.lead_dim_, ws);
            } else {
                // Safe decode -- fast2 (fp32_16_half_only_fast2, BIT-EXACT
                // with the safe fp32_16_half_only -- see that function's
                // own comment in fp32_16_pack.hpp; the kernel called here
                // actually runs the branchless no-overflow decode
                // internally now, same name kept from before that swap).
                //
                // WAS routed to the cp.async sibling (double-buffered
                // prefetch of wh) instead of a plain variant, on the
                // strength of real-LSQR TOMOS_LSQR_TIMING=1 evidence that
                // this per-iteration matvec ran ~1.37-1.52x slower under
                // --implicit-duplicates than --explicit-duplicates, at a
                // time when the isolated cp.async-vs-plain comparison had
                // never actually been run ("the lever that's actually
                // untested" -- see git history for that comment's full
                // text). It has since been run: a properly warmed-up
                // isolated benchmark (this session's own reps/warmup fix,
                // see scripts/benchmarks_gemv_transpose_fp32_16_pure_half_
                // tall_1000cols_vs_cublas_ref.sh) found the PLAIN vec4_
                // split variant consistently ~2-3% FASTER than cp.async
                // across every shape tested (100k/400k/800k/1M rows), not
                // slower -- cp.async's own staging overhead isn't earning
                // its keep here.
                //
                // THEN switched from vec4_split to vec2_split (this
                // function): ncu profiling of vec4_split at the real LSQR
                // shape (M=1,000,000, K=1000) found it 86% Compute-SM-
                // bound (ALU flagged as the over-utilized pipeline) with
                // ~50% of stall cycles waiting on L1TEX scoreboard
                // dependencies -- a register-blocking width sweep (vec2_
                // split/vec8_split, mechanical duplicates of vec4_split
                // with N=2/N=8 per-lane vectorization instead of N=4)
                // found BOTH neighbors beating vec4_split consistently
                // across every shape tested, with vec2_split the best
                // overall: ~7.6% faster than vec4_split at M=1,000,000
                // (1127.1 vs 1041.7 GFlop/s), and faster at 400k/800k/1M
                // too (roughly tied at 100k). Matches plot_gemv_transpose_
                // fp32_16_pure_half_vs_cublas_ref.m's own "TOMOS FP16
                // (implicit, fast2, vec2) [production]" series (pure_
                // half_fast2_vec2_split_gflops). Not yet re-confirmed via
                // a fresh TOMOS_LSQR_TIMING=1 real-LSQR run at the time of
                // this change.
                cuda::mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split(
                    trans, alpha, vA.size_, wh, vA.lead_dim_, vx.values_,
                    vx.lead_dim_, beta, vy.values_, vy.lead_dim_, ws);
            }
        }
    }
#endif
}

template <Device dev>
void mv_fast_impl(TransposeType trans, double alpha, tomos::Dense& A,
                  tomos::Dense& x, double beta, tomos::Dense& y,
                  float* /*workspace*/) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vx = as_strided_view<float>(x);
    auto vy = as_strided_view<float>(y);
    uint16_t* wh = reinterpret_cast<uint16_t*>(vA.values_);
    if constexpr (dev == Device::CPU) {
        cpu::mv_fp32_16_fast(trans, alpha, vA.size_, wh, vA.lead_dim_,
                            vx.values_, vx.lead_dim_, beta, vy.values_,
                            vy.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mv_fp32_16_fast(trans, alpha, vA.size_, wh, vA.lead_dim_,
                              vx.values_, vx.lead_dim_, beta, vy.values_,
                              vy.lead_dim_);
    }
#endif
}

template <Device dev>
void mv_accurate_impl(TransposeType trans, double alpha, tomos::Dense& A,
                     tomos::Dense& x, double beta, tomos::Dense& y,
                     float* /*workspace*/) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vx = as_strided_view<float>(x);
    auto vy = as_strided_view<float>(y);
    Fp32_16Storage s = as_fp32_16_storage(vA.values_, physical_num_elems(vA));
    if constexpr (dev == Device::CPU) {
        cpu::mv_fp32_16_accurate(trans, alpha, vA.size_, s.wh, s.wr,
                                 vA.lead_dim_, vx.values_, vx.lead_dim_, beta,
                                 vy.values_, vy.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        // NOT the "TOMOS FP32 / FP16" kernel family the plots measure
        // (mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best /
        // mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired)
        // -- that "hybrid" family never reads wr at all (only wh's own
        // extended-exponent escape hatch), so it doesn't actually meet
        // this tier's own near-machine-precision contract: test_fp32_16.
        // cpp's Fp32_16CudaTest.MvAllTierTransposeAndBetaCombinations
        // caught it failing ext_out's tight tolerance at NO_TRANSPOSE/
        // beta=0.7 AND TRANSPOSE/beta=0.0 -- the latter is exactly the
        // (trans, beta) LSQR's own restart-boundary/final residual check
        // always calls this tier with under --implicit-duplicates, and
        // was the direct cause of LSQR returning NaN there. Back to the
        // genuine wh+wr-combining reconstruction instead, which passes
        // that same test at every (trans, beta) combination -- "TOMOS
        // FP32/FP16" is a real, plotted, FASTER kernel, but it belongs in
        // a different registry slot than "accurate" if it's ever wired in
        // (its own accuracy tier, not a drop-in replacement for this one).
        //
        // NOW WIRED to mv_fp32_16_accurate_fast, a small composed
        // dispatcher (fp32_16_cuda_kernels.cu) that routes to whichever
        // of "test4" (ready8+cp.async+no_overflow, wins for rows_op <
        // kMvWideRowThreshold=131072) or "test2" (narrow_vec+cp.async+
        // no_overflow, wins for rows_op >= that threshold) actually has a
        // tuned kernel for the caller's shape. NOT "test4" alone -- an
        // earlier version of this wiring called mv_fp32_16_ready8_
        // atomiconly_scaled_kblock_cpasync_no_overflow directly, and was
        // confirmed (real LSQR run, solve time unchanged before/after) to
        // make NO difference: "test4"'s own optimization only lives
        // inside its own rows_op < kMvWideRowThreshold branch, and the
        // REAL LSQR restart-boundary matvec has rows_op = M = 1,000,000,
        // vastly above that threshold, where "test4" itself falls back to
        // plain ready4 -> the SAME untouched mv_fp32_16_accurate_
        // narrow_vec_kernel this whole investigation started from. See
        // mv_fp32_16_accurate_fast's own comment in fp32_16_cuda_
        // kernels.cu, and experiments-gemv.md, for the full derivation.
        // ("TOMOS FP32 / FP16 (fast)" on plot_gemv_fp32_recon_vs_
        // cutlass_ref.m plots "test4" alone, at square shapes only, where
        // that IS the right kernel -- the plot was never wrong, only this
        // registry's own wiring was.)
        //
        // Two distinct correctness questions, both already checked:
        //  1. The hybrid-family NaN bug above (skips wr entirely) does NOT
        //     apply to either "test4" or "test2" -- both genuinely read
        //     and combine wh AND wr (fp32_16_combine_branchless_
        //     no_overflow is a specialization of the full reconstruction,
        //     not a wr-skipping shortcut).
        //  2. TRANSPOSE calls (including this tier's own restart-boundary/
        //     final-residual check, exactly the call site that hit the
        //     historical bug above) never reach the no-overflow decode at
        //     all: mv_fp32_16_accurate_fast's own TRANSPOSE branch, and
        //     both "test4"'s and "test2"'s own internal TRANSPOSE
        //     fallbacks, all route to mv_fp32_16_accurate_ready4_
        //     atomiconly_scaled_kblock's genuine, SAFE (fp32_16_combine_
        //     branchless) TRANSPOSE fast path -- so this swap only ever
        //     changes NO_TRANSPOSE behavior.
        // The one REMAINING, NOT independently verified risk: NO_TRANSPOSE
        // calls now decode via fp32_16_combine_branchless_no_overflow,
        // which is silently wrong for any element whose true fp32
        // magnitude needs wh's OVERFLOW escape encoding -- see that
        // function's own comment in fp32_16_cuda_kernels.cu. Wired in on
        // the caller's explicit guarantee that this matrix's real data
        // never triggers that path (only ever underflow, if anything).
        cuda::experimental::mv_fp32_16_accurate_fast(trans, alpha, vA.size_, s.wh, s.wr, vA.lead_dim_, vx.values_,
                                                      vx.lead_dim_, beta, vy.values_, vy.lead_dim_);
    }
#endif
}

template <Device dev>
void mm_pure_half_impl(TransposeType transA, TransposeType transB,
                       double alpha, tomos::Dense& A, tomos::Dense& B,
                       double beta, tomos::Dense& C) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vB = as_strided_view<float>(B);
    auto vC = as_strided_view<float>(C);
    uint16_t* wh = reinterpret_cast<uint16_t*>(vA.values_);
    if constexpr (dev == Device::CPU) {
        cpu::mm_fp32_16_pure_half(transA, transB, alpha, vA.size_, wh,
                                  vA.lead_dim_, vB.size_, vB.values_,
                                  vB.lead_dim_, beta, vC.values_, vC.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mm_fp32_16_pure_half(transA, transB, alpha, vA.size_, wh,
                                   vA.lead_dim_, vB.size_, vB.values_,
                                   vB.lead_dim_, beta, vC.values_,
                                   vC.lead_dim_);
    }
#endif
}

template <Device dev>
void mm_fast_impl(TransposeType transA, TransposeType transB, double alpha,
                  tomos::Dense& A, tomos::Dense& B, double beta,
                  tomos::Dense& C) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vB = as_strided_view<float>(B);
    auto vC = as_strided_view<float>(C);
    uint16_t* wh = reinterpret_cast<uint16_t*>(vA.values_);
    if constexpr (dev == Device::CPU) {
        cpu::mm_fp32_16_fast(transA, transB, alpha, vA.size_, wh, vA.lead_dim_,
                            vB.size_, vB.values_, vB.lead_dim_, beta,
                            vC.values_, vC.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mm_fp32_16_fast(transA, transB, alpha, vA.size_, wh,
                              vA.lead_dim_, vB.size_, vB.values_, vB.lead_dim_,
                              beta, vC.values_, vC.lead_dim_);
    }
#endif
}

template <Device dev>
void mm_accurate_impl(TransposeType transA, TransposeType transB,
                     double alpha, tomos::Dense& A, tomos::Dense& B,
                     double beta, tomos::Dense& C) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vB = as_strided_view<float>(B);
    auto vC = as_strided_view<float>(C);
    Fp32_16Storage s = as_fp32_16_storage(vA.values_, physical_num_elems(vA));
    if constexpr (dev == Device::CPU) {
        cpu::mm_fp32_16_accurate(transA, transB, alpha, vA.size_, s.wh, s.wr,
                                 vA.lead_dim_, vB.size_, vB.values_,
                                 vB.lead_dim_, beta, vC.values_, vC.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::mm_fp32_16_accurate(transA, transB, alpha, vA.size_, s.wh, s.wr,
                                  vA.lead_dim_, vB.size_, vB.values_,
                                  vB.lead_dim_, beta, vC.values_, vC.lead_dim_);
    }
#endif
}

#if USE_CUDA
// CUDA-only: decodes A's wh/wr into a scratch float buffer (managed
// internally by the kernel itself -- cudaMalloc'd/freed per call, chunked
// so it never holds the whole matrix at once) then calls the codebase's
// fastest plain-fp32 GEMM on the decoded result. No CPU equivalent exists,
// so this is never registered for Device::CPU.
void mm_workspace_impl(TransposeType transA, TransposeType transB,
                       double alpha, tomos::Dense& A, tomos::Dense& B,
                       double beta, tomos::Dense& C) {
    auto vA = as_strided_view<uint32_t>(A);
    auto vB = as_strided_view<float>(B);
    auto vC = as_strided_view<float>(C);
    Fp32_16Storage s = as_fp32_16_storage(vA.values_, physical_num_elems(vA));
    cuda::mm_fp32_16_gemm_materialize_midpoint_exact(
        transA, transB, alpha, vA.size_, s.wh, s.wr, vA.lead_dim_, vB.size_,
        vB.values_, vB.lead_dim_, beta, vC.values_, vC.lead_dim_);
}
#endif

}  // namespace

const Fp32_16& Fp32_16::get() {
    static const Fp32_16 instance;
    return instance;
}

split_fp32_16_fn Fp32_16::lookup_split(Device dev) const {
    auto fn = split_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 split. Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

combine_fp32_16_fn Fp32_16::lookup_combine(Device dev) const {
    auto fn = combine_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 combine. Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mv_fp32_16_fn Fp32_16::lookup_mv_pure_half(Device dev) const {
    auto fn = mv_pure_half_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mv (pure_half). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mv_fp32_16_fn Fp32_16::lookup_mv_fast(Device dev) const {
    auto fn = mv_fast_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mv (fast). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mv_fp32_16_fn Fp32_16::lookup_mv_accurate(Device dev) const {
    auto fn = mv_accurate_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mv (accurate). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mm_fp32_16_fn Fp32_16::lookup_mm_pure_half(Device dev) const {
    auto fn = mm_pure_half_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mm (pure_half). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mm_fp32_16_fn Fp32_16::lookup_mm_fast(Device dev) const {
    auto fn = mm_fast_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mm (fast). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mm_fp32_16_fn Fp32_16::lookup_mm_accurate(Device dev) const {
    auto fn = mm_accurate_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mm (accurate). Rebuild with "
            "-DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mm_fp32_16_fn Fp32_16::lookup_mm_workspace(Device dev) const {
    auto fn = mm_workspace_[static_cast<std::size_t>(dev)];
    if (!fn)
        throw std::runtime_error(
            "No kernel registered for fp32_16 mm (workspace) on this device "
            "-- this tier is CUDA-only, no CPU implementation exists. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA and request Device::CUDA.");
    return fn;
}

Fp32_16::Fp32_16() {
    split_.fill(nullptr);
    combine_.fill(nullptr);
    mv_pure_half_.fill(nullptr);
    mv_fast_.fill(nullptr);
    mv_accurate_.fill(nullptr);
    mm_pure_half_.fill(nullptr);
    mm_fast_.fill(nullptr);
    mm_accurate_.fill(nullptr);
    mm_workspace_.fill(nullptr);

    split_[static_cast<std::size_t>(Device::CPU)] = &split_fp32_16_impl<Device::CPU>;
    combine_[static_cast<std::size_t>(Device::CPU)] =
        &combine_fp32_16_impl<Device::CPU>;
    mv_pure_half_[static_cast<std::size_t>(Device::CPU)] =
        &mv_pure_half_impl<Device::CPU>;
    mv_fast_[static_cast<std::size_t>(Device::CPU)] = &mv_fast_impl<Device::CPU>;
    mv_accurate_[static_cast<std::size_t>(Device::CPU)] =
        &mv_accurate_impl<Device::CPU>;
    mm_pure_half_[static_cast<std::size_t>(Device::CPU)] =
        &mm_pure_half_impl<Device::CPU>;
    mm_fast_[static_cast<std::size_t>(Device::CPU)] = &mm_fast_impl<Device::CPU>;
    mm_accurate_[static_cast<std::size_t>(Device::CPU)] =
        &mm_accurate_impl<Device::CPU>;

#if USE_CUDA
    split_[static_cast<std::size_t>(Device::CUDA)] =
        &split_fp32_16_impl<Device::CUDA>;
    combine_[static_cast<std::size_t>(Device::CUDA)] =
        &combine_fp32_16_impl<Device::CUDA>;
    mv_pure_half_[static_cast<std::size_t>(Device::CUDA)] =
        &mv_pure_half_impl<Device::CUDA>;
    mv_fast_[static_cast<std::size_t>(Device::CUDA)] = &mv_fast_impl<Device::CUDA>;
    mv_accurate_[static_cast<std::size_t>(Device::CUDA)] =
        &mv_accurate_impl<Device::CUDA>;
    mm_pure_half_[static_cast<std::size_t>(Device::CUDA)] =
        &mm_pure_half_impl<Device::CUDA>;
    mm_fast_[static_cast<std::size_t>(Device::CUDA)] = &mm_fast_impl<Device::CUDA>;
    mm_accurate_[static_cast<std::size_t>(Device::CUDA)] =
        &mm_accurate_impl<Device::CUDA>;
    mm_workspace_[static_cast<std::size_t>(Device::CUDA)] = &mm_workspace_impl;
#endif
}

}  // namespace dispatch
}  // namespace tomos

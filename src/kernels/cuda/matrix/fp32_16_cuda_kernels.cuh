#ifndef TOMOS_FP32_16_CUDA_KERNELS_CUH
#define TOMOS_FP32_16_CUDA_KERNELS_CUH

#include <cuda_fp16.h>
#include <cstdint>

#include "dimensions.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {

// Splits a plain float32 dense matrix into the wh/wr halves of the
// NumericType::FP32_16 representation (see fp32_16_pack.hpp).
void split_dense_fp32_16(dim<2> size, const float* src, size_t ld_src,
                         uint16_t* wh, uint16_t* wr, size_t ld_dst);

// Reconstructs a plain float32 dense matrix from the wh/wr halves.
void combine_dense_fp32_16(dim<2> size, const uint16_t* wh, const uint16_t* wr,
                           size_t ld_src, float* dst, size_t ld_dst);

// TEMPORARY diagnostic probe (remove once the fp32_16_combine_branchless
// large-magnitude divergence is root-caused) -- see its own comment in
// fp32_16_cuda_kernels.cu. Applies fp32_16_combine_branchless (the decode
// the actual GEMV kernels use, NOT fp32_16_combine/combine_dense_fp32_16
// above) to n (wh,wr) pairs, writing n float results to out.
void combine_branchless_probe(size_t n, const uint16_t* wh, const uint16_t* wr,
                              float* out);

// y = alpha * op(A) * x + beta * y, where A is stored as wh/wr (see
// fp32_16_pack.hpp) and combined element-by-element for ~fp32 accuracy.
void mv_fp32_16_accurate(TransposeType trans, double alpha, dim<2> size,
                         const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                         const float* x, size_t ld_x, double beta, float* y,
                         size_t ld_y);

// Returns the number of doubles mv_fp32_16_accurate_workspace's wide
// (split-K) path needs from a caller-provided workspace buffer for the
// given shape (0 if that shape doesn't dispatch to the wide path at
// all). See mv_fp32_16_accurate_workspace's own comment.
namespace experimental { size_t mv_fp32_16_accurate_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Same computation as mv_fp32_16_accurate, but the wide (split-K) path
// uses a caller-provided, caller-owned `workspace` buffer (sized via
// mv_fp32_16_accurate_workspace_size) instead of doing its own
// cudaMalloc/cudaFree on every call -- mirrors randls2.0's own
// DenseWorkspace/apply_with_workspace pattern (a persistent, matrix-
// lifetime-scoped scratch buffer, not a per-call allocation). See its
// own comment in fp32_16_cuda_kernels.cu, next to its definition, for
// the full reasoning and the workspace-sizing contract.
namespace experimental { void mv_fp32_16_accurate_workspace(TransposeType trans, double alpha, dim<2> size,
                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                   const float* x, size_t ld_x, double beta, float* y,
                                   size_t ld_y, double* workspace); }  // namespace experimental


// SUPERSEDED -- kept as a documented sibling, no longer used by mv_fp32_
// recon_vs_cutlass_ref.cu. Same workspace (buffered, no per-call malloc/
// free) mechanism as mv_fp32_16_accurate_workspace, decoding via
// fp32_16_combine_fast_normal instead of fp32_16_combine_branchless on
// the wide path -- but still 1 row/thread, same as mv_fp32_16_accurate_
// workspace itself, and measured NO improvement over it (load-
// instruction-bound, not decode-ALU-bound). "TOMOS FP32/FP16 (regular)"
// is now mv_fp32_16_accurate_ready_fast_normal instead (this same
// fast_normal decode, paired with 2-rows/lane + packed uint32_t loads),
// which measured 92% of native. CORRECTNESS CONTRACT (unchanged, still
// applies to this function): only valid when every element's control
// bit is 0 -- see its own comment in fp32_16_cuda_kernels.cu, next to
// mv_fp32_16_accurate_wide_fast_normal_kernel's own definition, for the
// full derivation and what data generator satisfies this (mv_fp32_
// recon_vs_cutlass_ref.cu's own gen_fp32_16_safe_normal).
namespace experimental { void mv_fp32_16_accurate_workspace_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                               const float* x, size_t ld_x, double beta, float* y,
                                               size_t ld_y, double* workspace); }  // namespace experimental


// Recreates randls2.0's own split-K GEMV mechanism (atomicAdd straight
// into y, no partial-sums buffer, no separate reduce kernel, no
// workspace of any kind needed by the caller) rather than just its
// allocation-lifetime idea -- contrast mv_fp32_16_accurate_workspace
// above, which keeps this codebase's own buffered design and only
// changes who owns the buffer. See its own comment in fp32_16_cuda_
// kernels.cu, next to its definition, for the full derivation, including
// why randls2.0's OTHER (ready-flag/spin-wait) split-K variant was
// deliberately not recreated. UNTESTED: no performance measurement or
// correctness verification has been run yet, and it carries a real
// precision trade-off (float atomicAdd across col_splits contributions
// instead of an exact double reduction) that needs checking on real
// hardware.
namespace experimental { void mv_fp32_16_accurate_atomic(TransposeType trans, double alpha, dim<2> size,
                                const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                const float* x, size_t ld_x, double beta, float* y,
                                size_t ld_y); }  // namespace experimental


// Returns how many ints the caller must provide as `ready` for
// mv_fp32_16_accurate_ready at the given shape (0 whenever that function
// would fall back to mv_fp32_16_accurate instead). See mv_fp32_16_
// accurate_ready's own comment.
namespace experimental { size_t mv_fp32_16_accurate_ready_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Recreates randls2.0's OWN split-K GEMV mechanism exactly (2 rows/lane,
// packed uint32_t loads, 4-way K unroll, float accumulation, ready-flag
// + atomicAdd combination, NO_TRANSPOSE-only) rather than just its
// atomics idea (contrast mv_fp32_16_accurate_atomic above, which uses
// this codebase's own 1-row/thread, double-accumulator kernel body).
// `ready` is caller-provided and reused across calls, matching
// randls2.0's own gemv_readiness_array_. Falls back to mv_fp32_16_
// accurate for anything the underlying kernel doesn't support
// (TRANSPOSE, or a shape past the wide-path threshold), so always
// correct to call. See its own comment in fp32_16_cuda_kernels.cu, next
// to mv_fp32_16_ready_kernel's own definition, for the full derivation,
// including the inter-block spin-wait correctness caveat carried over
// unchanged from randls2.0's own design. UNTESTED: no performance
// measurement or correctness verification has been run yet.
namespace experimental { void mv_fp32_16_accurate_ready(TransposeType trans, double alpha, dim<2> size,
                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                               const float* x, size_t ld_x, double beta, float* y,
                               size_t ld_y, int* ready); }  // namespace experimental


// "TOMOS FP32/FP16 (regular)" in mv_fp32_recon_vs_cutlass_ref.cu / plot_
// gemv_fp32_recon_vs_cutlass_ref.m -- UNSAFE ON ARBITRARY DATA. Decodes
// via fp32_16_combine_fast_normal instead of fp32_16_combine_branchless
// across ALL THREE shape-dispatch paths (wide/tiled/narrow_vec), not
// just wide, so large-M TALL-AND-THIN shapes (which dispatch to
// narrow_vec, not wide -- mv_fp32_recon_vs_cutlass_ref.cu's own tall-
// and-thin sweep, K fixed small, M swept up to 1048576) get the same win
// the square sweep measured. TRANSPOSE and a padded ld_a fall back to
// the fully general mv_fp32_16_accurate (mv_fp32_16_ready_fast_normal_
// kernel's own packed-load trick can't handle those).
//
// A self-checking revision of this function (a whole-matrix reduction
// pass before every call, verifying the control=0 contract instead of
// trusting the caller) was tried and reverted -- it measured WORSE on
// real hardware than this plain, caller-trust version, so the contract
// is back to: the CALLER must independently guarantee every element's
// control bit is 0 (e.g. via mv_fp32_recon_vs_cutlass_ref.cu's own gen_
// fp32_16_safe_normal generator) -- see its own comment in fp32_16_cuda_
// kernels.cu, next to its definition, for the full reasoning.
//
// Measured 92% of native fp32 GEMV throughput at M=K=65536 on the wide
// path (real hardware), matching randls2.0's own reported parity -- this
// needed BOTH mv_fp32_16_accurate_ready's row-vectorized load pattern
// (75% alone) AND the fast_normal decode swap (no measurable effect
// alone, on a still-1-row/thread kernel -- see mv_fp32_16_accurate_
// workspace_fast_normal's own, now-superseded comment) together.
//
// Deliberately not wired into verify_mv_fp32_16_gemv.cu (that file's own
// cases exist specifically to exercise the extended/degraded encoding
// this function cannot handle).
namespace experimental { void mv_fp32_16_accurate_ready_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                           const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                           const float* x, size_t ld_x, double beta, float* y,
                                           size_t ld_y, int* ready); }  // namespace experimental


// EXPERIMENTAL, MEASURED WORSE: 4 rows/lane instead of mv_fp32_16_
// accurate_ready_fast_normal's 2 -- extends the same row-vectorization
// lever one step further (packs 4 adjacent rows' wh, and separately wr,
// into a single uint64_t load instead of 2 rows/uint32_t), but real
// hardware measurement shows this consistently ~0.68-0.72x of the 2-row
// kernel's own throughput across M=K=1024..4096, not improving as M
// grows -- likely an occupancy regression (halving rows/thread without
// also adjusting K_BLOCK halves total block/thread count for the same
// M), see mv_fp32_16_ready4_fast_normal_kernel's own comment for the
// full reasoning. Kept as a documented negative result, not deleted.
// Decode is unconditionally fp32_16_combine_fast_normal, same UNSAFE-on-
// arbitrary-data contract as mv_fp32_16_accurate_ready_fast_normal --
// caller must independently guarantee every element's control bit is 0.
// Same dispatch/fallback structure as that function otherwise.
namespace experimental { void mv_fp32_16_accurate_ready4_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                            const float* x, size_t ld_x, double beta, float* y,
                                            size_t ld_y, int* ready); }  // namespace experimental


// CONFIRMED WINNER at small-to-medium M: reuses mv_fp32_16_ready4_fast_
// normal_kernel's own __global__ code unchanged, with a smaller K_BLOCK
// (64 instead of 128) to compensate that kernel's own measured occupancy
// regression -- doubling gridK to offset 4-rows/lane halving gridRows
// for the same M, restoring total block/thread count to parity with
// mv_fp32_16_accurate_ready_fast_normal's own (2-row) config. Confirmed
// on real hardware: recovers dramatically and edges PAST the 2-row
// kernel at M=K=1024/1536/2048 (1.03-1.09x), landing slightly below it
// by M=K=3072 (0.91x) -- not yet measured at the larger M (up to 65536)
// the 2-row kernel's own 92%-of-native figure was measured at, so
// whether this stays ahead, ties, or falls behind at scale is still
// open. See its own comment in fp32_16_cuda_kernels.cu, next to its
// definition, for the exact block-count arithmetic and the full numbers.
// Same UNSAFE-on-arbitrary-data contract as every other fast_normal
// sibling.
namespace experimental { void mv_fp32_16_accurate_ready4_kblock64_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                      const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                      const float* x, size_t ld_x, double beta, float* y,
                                                      size_t ld_y, int* ready); }  // namespace experimental


// CONFIRMED WINNER on real hardware -- currently the best-measured wide-
// path kernel in this file: 2 rows/lane + packed uint32_t loads (same as
// mv_fp32_16_accurate_ready_fast_normal), but with mv_fp32_16_accurate_
// atomic's own simpler prescale+atomic-only write-back instead of the
// ready-flag/spin-wait one -- no `ready` buffer needed at all (one fewer
// parameter than every other ready-flag-based function in this header).
// Beats mv_fp32_16_accurate_ready_fast_normal at EVERY size tested so
// far (M=K=1024 through 3072, 1.03-1.09x), not just some -- unlike the
// 4-row experiments' crossover. Pending confirmation at the larger M (up
// to 65536) mv_fp32_16_accurate_ready_fast_normal's own 92%-of-native
// figure was measured at. See its own comment in fp32_16_cuda_kernels.cu,
// next to mv_fp32_16_ready_atomiconly_fast_normal_kernel's own
// definition, for the full derivation and numbers. Same UNSAFE-on-
// arbitrary-data contract as every other fast_normal sibling.
namespace experimental { void mv_fp32_16_accurate_ready_atomiconly_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                       const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                       const float* x, size_t ld_x, double beta, float* y,
                                                       size_t ld_y); }  // namespace experimental


// MEASURED WORSE (real hardware): 2 rows/lane + packed uint32_t loads,
// same as mv_fp32_16_accurate_ready_atomiconly_fast_normal, but does NOT
// split K across blocks/launches at all -- each block loops over the
// entire K range internally and writes y once, directly, no atomics of
// any kind. Built to test whether removing split-K's own atomic
// contention (which becomes the dominant cost at large M/K, per real-
// hardware measurement showing mv_fp32_16_accurate_ready_atomiconly_
// fast_normal's own advantage over the ready-flag version collapsing to
// exactly 1.000x from M=K=24576 onward) would beat the split-K design at
// large sizes. It does not: loses at EVERY square size tested, 1024
// through 65536 (0.150x at M=1024, worst at 0.103x at M=8192, climbing
// to only 0.598x at M=65536). Removing split-K removes far more grid-
// level parallelism than it saves in atomic contention -- see its own
// comment in fp32_16_cuda_kernels.cu, next to mv_fp32_16_nosplit_fast_
// normal_kernel's own definition, for the full derivation. Kept as a
// documented negative result; not wired into a dispatcher. Same UNSAFE-
// on-arbitrary-data contract as every other fast_normal sibling.
namespace experimental { void mv_fp32_16_accurate_nosplit_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                             const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                             const float* x, size_t ld_x, double beta, float* y,
                                             size_t ld_y); }  // namespace experimental


// CONFIRMED WINNER (real hardware): SAME kernel as mv_fp32_16_accurate_
// ready_atomiconly_fast_normal (reused unchanged, K_BLOCK is already a
// runtime parameter), but with a K-dependent K_BLOCK instead of the fixed
// 128, targeting a roughly constant number of K-splits (gridK) instead of
// letting it grow linearly with K. Motivation: mv_fp32_16_accurate_ready_
// atomiconly_fast_normal's own advantage over the ready-flag version
// collapses to exactly 1.000x from M=K=24576 onward because gridK = K/128
// grows linearly, and atomic contention on the same y[row] across gridK
// blocks becomes the dominant cost -- at K=16384 (gridK=128) the
// advantage still held (1.017x); by K=24576 (gridK=192) it had already
// collapsed. This grows K_BLOCK once K exceeds 128*128 so gridK stays
// near 128, capping the contention. CONFIRMED: ties mv_fp32_16_accurate_
// ready_atomiconly_fast_normal below the K=16384 crossover (0.999x-1.011x,
// noise) and beats it consistently above it -- 1.097x at K=24576, 1.107x
// at 32768, 1.087x at 49152, 1.090x at 65536 -- also now beating native at
// every one of those sizes (0.990x-1.016x, vs the fixed-K_BLOCK version's
// 0.907x-0.926x there). Confirms the atomic-contention plateau was an
// artifact of the fixed K_BLOCK, not an inherent split-K cost -- unlike
// mv_fp32_16_accurate_nosplit_fast_normal (which eliminated split-K
// outright and lost at every size tested, see its own comment above),
// this keeps split-K's grid parallelism and just bounds its contention.
// "TOMOS FP32/FP16 (regular)": UNSAFE-on-arbitrary-data contract, same
// as every other fast_normal sibling -- but the fastest fp32_16 kernel
// in this file where that contract holds. See its own comment in
// fp32_16_cuda_kernels.cu for the full derivation and numbers, and see
// mv_fp32_16_accurate_ready_atomiconly_scaled_kblock's own comment
// (below, and its own comment in the .cu file) for the SAFE-on-
// arbitrary-data counterpart promoted as plain "TOMOS FP32/FP16"
// instead.
//
// Also has a TRANSPOSE fast path now (trans==TRANSPOSE && ld_a==size[0]):
// warp-per-output-column/split-M/atomicAdd, ported from randls2.0's own
// sgemv_cm_transpose_splitM_atomic_xshared -- see mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_kernel's own comment in fp32_16_
// cuda_kernels.cu for the full derivation. UNSAFE-on-arbitrary-data
// contract unchanged. UNTESTED (kernel-level, TRANSPOSE): no GEMV
// performance measurement has been run yet.
void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_accurate_ready_
// atomiconly_scaled_kblock_fast_normal's own TRANSPOSE fast path -- NOT
// wired into that dispatch (which is unchanged), kept separate per this
// session's own standing "add new kernels, don't modify existing ones"
// convention. That TRANSPOSE path was never vectorized at all (unlike
// the SAFE decode's own vec4/retuned_floor siblings) -- applies the
// same float4/ushort4 vectorization playbook here, decode via fp32_16_
// combine_fast_normal instead of fp32_16_combine_branchless. Same
// UNSAFE-on-arbitrary-data contract. Uses the shared default grid-
// oversubscription target (512), not the SAFE decode's own higher
// retuned target -- see mv_fp32_16_transpose_atomiconly_scaled_mblock_
// fast_normal_vec4_kernel's own comment in fp32_16_cuda_kernels.cu for
// the full derivation. MEASURED on real A100 hardware: a real win from
// M=K=4096 upward (+6.2% to +21.8% vs the scalar sibling), a mild dip
// at M=1024-3072 (-4.3% to -6.0%, far milder than the SAFE decode's own
// unfloored regression since this kernel uses the conservative default
// target). Reaches 0.78x-0.90x of cuBLAS from M=6144 upward -- the
// best-measured fp32_16 TRANSPOSE kernel in this file at every size
// M>=4096.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                      const uint16_t* wh, const uint16_t* wr,
                                                                      size_t ld_a, const float* x, size_t ld_x,
                                                                      double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_vec4 above -- byte-for-byte
// IDENTICAL structure, decoding via fp32_16_combine_fast_normal_
// wrapborrow instead of fp32_16_combine_fast_normal (bit-identical,
// exhaustively verified -- see its own comment in fp32_16_pack.hpp).
// Motivated by a direct SASS profile of the sibling kernel showing
// decode overhead (not FMA math) dominates its static instruction count
// -- see the kernel's own comment in fp32_16_cuda_kernels.cu.
// CONFIRMED via cuobjdump before any hardware measurement: eliminates
// ALL 38 SEL and 19 ISETP.EQ.U32.AND instructions the is_carry branch
// compiled to, and drops register usage from REG:39 to REG:32 (matching
// the SAFE decode kernel's own REG:31 -- 8 blocks/SM instead of 6, full-
// residency capacity 864 instead of 648, same math as kMvFp32_16Fast
// NormalPersistentResidencyCap's own derivation).
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4 above -- reuses
// the EXACT SAME __global__ kernel unchanged, only a more aggressive
// M_BLOCK-scaling target (2048, not the 1024 the wrapborrow wrapper
// above still inherits from the pre-wrapborrow fast_normal_vec4
// sibling's own REG:39 derivation) -- re-derived from wrapborrow's OWN
// measured register footprint (REG:32, matching the SAFE decode
// kernel's own REG:31/2048-target residency profile, not blindly copied
// from it) -- see kMvFp32_16FastNormalWrapborrowTransposeTargetGridBlocks's
// own comment in fp32_16_cuda_kernels.cu for the full derivation.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget above --
// reuses the EXACT SAME __global__ kernel unchanged, only an even more
// aggressive M_BLOCK-scaling target (8192 instead of 2048). Motivated by
// a specific, verified gap in _retarget's own promotion: at K=4096,
// M=32768/65536, target=2048 never actually drops M_BLOCK below the
// 8192 ceiling there, so the grid is byte-for-byte identical to the
// pre-retarget "(fast, wrapborrow)" sibling at those two shapes (the
// originally-measured differences there were noise, not real -- see
// the .sh's own METHOD_MARKERS comment for the full derivation).
// target=8192 is the smallest value that pushes scaled_block below the
// ceiling at BOTH shapes -- see kMvFp32_16FastNormalWrapborrowTranspose
// TargetGridBlocks2's own comment in fp32_16_cuda_kernels.cu for the
// full derivation and oversubscription-risk discussion.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 above --
// SAME M_BLOCK-scaling (target=8192, floor=256), SAME atomicAdd combine,
// isolates ONLY how much of x is staged into shared memory at once: the
// untiled kernel stages the entire [mStart, mEnd) M_BLOCK range in one
// shot; this one loops over 2048-float sub-tiles of it, staging/syncing/
// accumulating each into the same persistent per-thread accumulator
// (one final warp-reduce + atomicAdd after the loop, not per sub-tile).
// Motivated by ncu profiling on real A100 hardware at M=K=65536 (where
// _retarget2 launches): "Block Limit Shared Mem: 4" vs "Block Limit
// Registers: 8", with the profiler explicitly flagging shared memory as
// the binding occupancy constraint (Achieved Occupancy 48.87%, right AT
// the resulting 50% theoretical ceiling) -- a DIFFERENT lever from both
// prior large-M attempts (raising the M_BLOCK ceiling, which INCREASES
// this same footprint and confirmed WORSE occupancy; removing atomicAdd
// via _retarget2_split, confirmed NOT the bottleneck by this same
// profile -- both measured within 1 percentage point of _retarget2 on
// every metric). This lever touches neither M_BLOCK/gridM/gridK nor the
// combine mechanism at all. A pure superset of the untiled kernel when
// M_BLOCK is already <= the sub-tile size (the loop runs once, byte-for-
// byte equivalent) -- see its own comment in fp32_16_cuda_kernels.cu for
// the full derivation and exact profiler numbers.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_tiled(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Returns the number of doubles mv_fp32_16_transpose_atomiconly_scaled_
// mblock_fast_normal_wrapborrow_vec4_retarget2_split's own scratch buffer
// needs for the given shape (gridM * k) -- call once per distinct shape
// (outside any timed loop) to size and allocate a workspace buffer, then
// reuse it across every call at that shape. See that function's own
// comment for the full contract.
namespace experimental { size_t mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split_workspace_size(
    dim<2> size); }  // namespace experimental


// Split-then-reduce counterpart of mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 above: SAME
// M_BLOCK-scaling (target=8192, floor=256), isolates ONLY the mechanism
// used to combine each M-block's partial sum into y -- a scratch buffer
// (one double per (M-block, column) pair) plus mv_fp32_16_wide_reduce_
// kernel, instead of atomicAdd. Motivated by a gap in the M_BLOCK-ceiling
// investigation: once M_BLOCK saturates the shared 8192 ceiling, gridM
// (and therefore the number of atomicAdds landing on the SAME y[col])
// grows LINEARLY with M -- a contention source that gets WORSE at
// exactly the M>=16384 sizes this dispatcher's own threshold (8192)
// hands to _retarget2. Raising the ceiling to shrink gridM back down was
// tried and is a CONFIRMED, SEVERE negative result (occupancy collapse);
// this sibling removes the atomics instead, without touching M_BLOCK or
// shared-memory footprint at all, so it should not reproduce that
// failure.
//
// Takes a CALLER-PROVIDED `workspace` buffer (see the size helper above)
// instead of doing its own cudaMalloc/cudaFree on every call -- mirrors
// mv_fp32_16_accurate_workspace's own contract. This function's FIRST
// version did internal cudaMalloc/cudaFree per call; MEASURED on real
// A100 hardware that version was a CONFIRMED NEGATIVE RESULT (-0.8% to
// -6.5% vs plain _retarget2 at every size tested, no growing-with-M
// trend) -- but the WORST regression was at M=8192, where gridM=1 for
// BOTH kernels (zero atomic contention to remove), so that entire result
// was pure allocator overhead, not the mechanism failing. Switched to
// this workspace contract to re-test the mechanism with that confound
// removed.
//
// MEASURED with the allocator confound removed (square M=K, M=8192-
// 65536): STILL a CONFIRMED NEGATIVE RESULT, though much smaller now
// (-0.2% to -2.8%, vs -0.8% to -6.5% before) -- so the malloc/free WAS
// most of the original gap, but a small, flat ~0.5% residual loss
// remains at every size, including large M where gridM is well above 1
// and real atomic contention should show up if the original hypothesis
// were right. It doesn't: the residual is LARGEST at M=8192 (gridM=1,
// zero contention to remove, so pure mechanism overhead) and does NOT
// grow with M/gridM. CLOSED: A100's hardware atomics are cheap enough at
// this kernel family's gridM range that removing them nets nothing --
// see its own comment in fp32_16_cuda_kernels.cu for the full derivation
// and exact per-size numbers.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y, double* workspace); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 above --
// reuses the EXACT SAME __global__ kernel and target (8192) unchanged,
// only a HIGHER floor (2048 instead of 256). MEASURED on real A100
// hardware: DOES close the targeted M=3072-6144 regression, but
// introduces a NEW, real regression at M=1024 (-16.8% to -17.1%) and
// M=1536 (-7.7% to -8.0%) -- the floor (2048) exceeds m itself at these
// shapes, forcing M_BLOCK up instead of letting it shrink, collapsing
// gridM to 1 instead of _retarget/_retarget2's own 4-6 blocks there.
// DOCUMENTED NEGATIVE RESULT: kept as a standalone sibling, not
// promoted, not folded into the dispatcher below (which routes around
// it entirely) -- see kMvFp32_16FastNormalWrapborrowRetarget2MinMBlock's
// own comment in fp32_16_cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Standalone dispatcher for the fast_normal_wrapborrow TRANSPOSE
// family -- routes to _retarget2 (target=8192) for M>=8192, _retarget
// (target=2048) otherwise, based on real A100 hardware measurements
// confirming _retarget2 regresses at M=3072-6144 (M_BLOCK crashing
// toward its own floor) but is never worse than _retarget at M>=8192 (a
// real win at M=8192-16384, grid-identical at M>=24576). PROMOTED: this
// is "TOMOS FP32/FP16 (fast)" in scripts/plot_gemv_transpose_fp32_
// recon_vs_cublas_ref.m, validated on real A100 hardware across the FULL
// square sweep (M=1024-65536) -- tracks _retarget almost exactly below
// the threshold and _retarget2 almost exactly at/above it, with
// negligible routing overhead throughout. CAVEAT: the M=8192 threshold
// itself was only validated on SQUARE shapes -- see the .sh's own
// METHOD_MARKERS comment for the full promotion history. Does NOT route
// through _retarget2_floor, _retarget2_bigblock, _retarget_bigblock, or
// _retarget2_hugeblock (below) -- all four are confirmed negative results
// on real A100 hardware, see each one's own comment for details.
void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_dispatch(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2 above --
// reuses the EXACT SAME __global__ kernel and target (8192) unchanged,
// only a HIGHER shared M_BLOCK ceiling (kMvTransposeMBlockLarge=16384
// instead of the shared kMvTransposeMBlock=8192) -- see
// kMvTransposeMBlockLarge's own comment in fp32_16_cuda_kernels.cu for
// the full grid-math derivation. CONFIRMED REGRESSION on real A100
// hardware: -26.7% vs _dispatch at M=K=49152, -26.6% at M=K=65536 -- the
// occupancy loss from the bigger shared-memory footprint (~2 resident
// blocks/SM instead of ~5-8) completely dominates the reduced atomicAdd
// contention the grid math predicted. A true no-op (neither win nor
// loss) at tall-and-thin K=4096 shapes, where the higher ceiling never
// gets reached. CLOSED, CONFIRMED NEGATIVE RESULT, not wired into the
// dispatcher above.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_bigblock(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of _retarget2_bigblock above --
// SAME higher ceiling (16384), but _retarget's own LOWER target (2048)
// instead of _retarget2's 8192. CONFIRMED REGRESSED on real A100
// hardware in BOTH regimes tested: tall-and-thin (K=4096, -1.5% vs
// _dispatch at M=16384, -11.5% at M=32768 -- silently reduces to plain
// _retarget's own already-known-worse tall-and-thin tuning, since
// target=2048 never reaches the higher ceiling at that K) AND square
// (-26.7%/-26.6% at M=K=49152/65536, byte-identical to _retarget2_
// bigblock's own numbers there -- same occupancy collapse, inherited
// from the shared ceiling). CLOSED, CONFIRMED NEGATIVE RESULT everywhere
// tested. See kMvTransposeMBlockLarge's own comment in fp32_16_cuda_
// kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget_bigblock(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of _retarget2_bigblock above --
// reuses the EXACT SAME __global__ kernel and target (8192) unchanged,
// only an even higher ceiling (kMvTransposeMBlockHuge=32768 instead of
// kMvTransposeMBlockLarge=16384) -- see kMvTransposeMBlockHuge's own
// comment in fp32_16_cuda_kernels.cu for the full derivation. CONFIRMED
// REGRESSION on real A100 hardware, and a SEVERE one: -57.1% vs
// _dispatch at M=K=49152, -57.2% at M=K=65536 -- roughly double
// _retarget2_bigblock's own regression at the SAME shapes, tracking the
// further halving of resident blocks/SM (1 instead of ~2). Confirms
// occupancy collapse, not atomic-contention reduction, governs
// performance in this regime. CLOSED, CONFIRMED NEGATIVE RESULT -- do
// not push the ceiling higher still.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4_retarget2_hugeblock(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_fast_normal_vec4 above -- reuses the EXACT
// SAME __global__ kernel unchanged, only a different (more conservative
// than the SAFE decode's own retuned_floor) M_BLOCK-scaling target,
// scaled down from that kernel's own 2048 to account for this kernel's
// own LOWER register-limited residency ceiling (REG:39 here vs REG:31
// for the SAFE vec4 kernel) -- see kMvFp32_16FastNormalTransposeTarget
// GridBlocks's own comment in fp32_16_cuda_kernels.cu for the full
// derivation.
//
// MEASURED on real A100 hardware (square M=K, M=1024-65536): a CLEAN win
// over the plain vec4 sibling at every size tested, closing its mild
// M=1024-3072 dip entirely and settling to ~parity from M=8192 upward.
// Was plain "fp32_16_fast"/"TOMOS FP32/FP16 (fast)"; DEMOTED through
// several further promotion rounds since (wrapborrow decode, then
// _retarget's own retuning, then the _dispatch router) -- see the
// kernel's own comment in fp32_16_cuda_kernels.cu and the .sh's own
// METHOD_MARKERS comment for the full promotion history.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_retuned(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone sibling of mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_vec4 above -- ported from randls2.0's own
// persistent-block pattern (hgemv_cm_rows64_persistent_half2 in
// dense_kernels.cu) combined with this file's own atomicAdd TRANSPOSE
// reduction: caps the grid's M-width at the register-limited full-
// residency ceiling (kMvFp32_16FastNormalPersistentResidencyCap=864,
// REG:32) instead of growing 1:1 with the number of M_BLOCK chunks --
// each block loops internally over every chunk it owns, accumulating in
// a register, and does exactly ONE atomicAdd per warp for its entire
// lifetime.
//
// MEASURED on real A100 hardware (square M=K, M=1024-4096 so far): a NET
// REGRESSION vs the "(fast, vec4)" retuned sibling, -19.4% to -4.1%,
// worse at every size tested -- including sizes where the launched grid
// is IDENTICAL to the sibling's own (single loop iteration per block),
// ruling out grid-shape/atomic-count differences as the sole cause. The
// _v2 sibling below tests one specific hypothesis for that single-
// iteration-case cost (an avoidable unconditional __syncthreads()) and
// MEASURED NO improvement -- see both kernels' own comments in fp32_16_
// cuda_kernels.cu for the full, still-incomplete diagnosis. Kept as a
// documented negative result, not promoted; not recommended for further
// pursuit without a new hypothesis. Tall-and-thin remains untested but
// is not expected to recover given how poorly this performs even in the
// grid-identical case.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone sibling of mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_vec4_persistent above -- byte-for-byte
// identical launch config and per-chunk compute, with ONE change: the
// trailing __syncthreads() at the end of each loop iteration is now
// conditional on there being another iteration to run, instead of
// executing unconditionally. Isolates how much of the persistent
// sibling's own measured regression (see its comment above) was this
// avoidable per-launch overhead vs. its OTHER, architectural cause
// (large-M multi-iteration cases trading cross-block memory-level
// parallelism for fewer atomicAdd calls) -- see the kernel's own comment
// in fp32_16_cuda_kernels.cu for the full derivation.
//
// MEASURED on real A100 hardware (square M=K, M=1024-4096): NO
// meaningful difference vs the persistent sibling above (0.987x-1.003x,
// i.e. measurement noise) -- the hypothesis this kernel was built to
// test was WRONG. Both kernels regress -19.4% to -4.1% vs the non-
// persistent _retuned kernel across every size tested. Kept as a
// documented negative result; not recommended for further pursuit.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_vec4_persistent_v2(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone sibling of mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_vec4 above -- ported DIRECTLY from
// randls2.0's own sgemv_cm_transpose_warprows_splitm_tiled (dense_
// kernels.cu:9899-9959). A GENUINELY DIFFERENT reduction strategy from
// every other TRANSPOSE kernel in this file: no atomics, no cross-block
// combination, no warp-shuffle reduction -- EACH LANE within a 32-
// column stripe owns one full output column for the kernel's ENTIRE
// lifetime (gridDim.x is always 1: one persistent block per column-
// stripe covers the WHOLE M range, tiled internally), with
// kMvFp32_16TransposeSplitmFactor warp-groups splitting each tile's
// rows CONCURRENTLY (real parallelism, not a serial loop) and combining
// via shared memory once at the end.
//
// CAVEAT: this access pattern (col = col_base + lane) means consecutive
// lanes read consecutive COLUMNS at a fixed row -- UNCOALESCED in this
// column-major layout (M elements apart), the opposite of every other
// TRANSPOSE kernel here (one warp per column, coalesced row reads).
//
// MEASURED on real A100 hardware (square M=K=1024-2048): CATASTROPHIC --
// 30-70x slower than cuBLAS (0.013x-0.024x), 0.02x-0.03x of the promoted
// _retuned kernel. Confirmed root causes (not just inferred): (1)
// gridDim.x fixed at 1 means total launched blocks = ceil(K/256) -- just
// 4-8 blocks at these sizes, occupying 4-8 of the A100's 108 SMs, over
// 90% idle; (2) the uncoalesced access above compounds on top of that.
// This CONFIRMS why randls2.0's own author abandoned this design for
// TRANSPOSE in favor of the atomicAdd splitM design already ported
// above -- see the kernel's own comment in fp32_16_cuda_kernels.cu for
// the full derivation. Kept as a documented negative result, NOT
// recommended for further pursuit -- closing this gap would require
// abandoning the design's own core premise.
namespace experimental { void mv_fp32_16_transpose_warprows_splitm_tiled_fast_normal(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone sibling of mv_fp32_16_transpose_atomiconly_
// scaled_mblock_fast_normal_vec4 above -- ported DIRECTLY from
// randls2.0's own sgemv_cm_transpose_warpdot_tiled (temp-dependencies/
// randls2.0/src/kernels/cuda/matrix/dense_kernels.cu:6076-6136).
// Confirmed dead code there, but MEASURED near-parity with cuBLAS on
// real A100 hardware (0.985x at M=K=4096) via a standalone verbatim-
// kernel driver (benchmarks/kernels/cuda/matrix/mv_fp32_transpose_
// randls2_0_warpdot_tiled_reference.cu) -- the ONLY positively-measured
// randls2.0 TRANSPOSE design encountered this session (unlike
// warprows_splitm_tiled above, catastrophic). Structurally much simpler
// than every other TRANSPOSE kernel in this file: no atomics, no
// M_BLOCK-scaling/grid-oversubscription tuning -- ONE warp per column
// loops the ENTIRE M range internally in shared-mem-staged tiles, single
// shfl-reduce and write at the end.
//
// MEASURED on real A100 hardware (square M=K=1024-2048): a real
// regression vs the promoted _retuned kernel, -25.7% to -18.7% (0.743x-
// 0.813x) -- not catastrophic like warprows_splitm_tiled (0.02x), but
// not competitive either. Confirms the randls2.0 reference's near-
// cuBLAS-parity (plain fp32, no decode) does NOT carry over once every
// read pays fp32_16_combine_fast_normal's own decode cost -- see the
// kernel's own comment in fp32_16_cuda_kernels.cu for the full
// diagnosis. Kept as a documented negative result, not promoted.
namespace experimental { void mv_fp32_16_transpose_warpdot_tiled_fast_normal(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// SAFE (fp32_16_combine_branchless) counterpart to mv_fp32_16_accurate_
// ready_atomiconly_scaled_kblock_fast_normal above -- byte-for-byte the
// same row-vectorized (2 rows/lane, packed uint32_t loads)/prescale+
// atomicAdd/K-scaled-K_BLOCK structure, just decoding via the full
// branchless (normal/extended/degraded) reconstruction instead of the
// cheap "normal"-only shortcut. Correct on arbitrary caller data, the
// same correctness bar as mv_fp32_16_accurate itself -- no dedicated
// safe-normal-range test data needed, unlike every "_fast_normal"
// sibling in this file.
//
// SUPERSEDED as plain "TOMOS FP32/FP16" by mv_fp32_16_accurate_ready4_
// atomiconly_scaled_kblock below (still run/printed as an unpromoted
// sibling, not removed): best-measured fp32_16 kernel that is BOTH safe
// on arbitrary data AND needs no external workspace/scratch buffer of
// any kind -- its own fallback for non-fast-path shapes (transposed
// matrix, or ld_a != size[0]) is mv_fp32_16_accurate (the malloc-based
// baseline), not a caller-provided buffer. See its own comment in
// fp32_16_cuda_kernels.cu, next to its own definition, for the full
// derivation.
namespace experimental { void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental


// PROMOTED as plain "TOMOS FP32/FP16" (real hardware, tall-and-thin M
// sweep at K=4096): 4-rows/lane extension of mv_fp32_16_accurate_ready_
// atomiconly_scaled_kblock above -- same SAFE (fp32_16_combine_
// branchless) decode and atomiconly (no external buffer) mechanism, but
// packs 4 adjacent rows per lane (uint64_t loads) instead of 2. Unlike
// mv_fp32_16_ready4_fast_normal_kernel's own 4-row experiment (which
// lost to its 2-row sibling, occupancy-bound), this one wins outright
// with no K_BLOCK compensation needed -- 1.070x at M=16384, 1.052x at
// M=32768, 1.115x at M=65536, all vs the 2-row kernel. The SAFE decode's
// higher per-element cost tips the amortization-vs-occupancy balance the
// other way from the cheap fast_normal case. Square (M=K) shapes not
// separately confirmed before this promotion, only the tall-and-thin
// sweep. See its own comment in fp32_16_cuda_kernels.cu, next to its
// own definition, for the full derivation.
//
// Also has a TRANSPOSE fast path now (trans==TRANSPOSE && ld_a==size[0]):
// warp-per-output-column/split-M/atomicAdd, ported from randls2.0's own
// sgemv_cm_transpose_splitM_atomic_xshared -- see mv_fp32_16_transpose_
// atomiconly_scaled_mblock_kernel's own comment in fp32_16_cuda_
// kernels.cu for the full derivation. Safe on arbitrary data, same as
// the NO_TRANSPOSE path. UNTESTED (kernel-level, TRANSPOSE): no GEMV
// performance measurement has been run yet.
void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_accurate_ready4_
// atomiconly_scaled_kblock's own TRANSPOSE fast path -- NOT wired into
// that dispatch (which is unchanged), kept separate per this session's
// own standing "add new kernels, don't modify existing ones" convention.
// Mirrors mv_fp32_transpose_atomiconly_scaled_mblock_vec4_stagevec's own
// playbook (mv_cuda_kernels.cu, native fp32's own TRANSPOSE kernel) in
// one step: float4-vectorizes the x-staging loop AND the main
// accumulation loop's sx reads, plus vectorizes the wh/wr LOADS
// themselves (ushort4, 4 elements per load) -- though not the decode
// itself (fp32_16_combine_branchless stays scalar, called once per
// element). See mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
// kernel's own comment in fp32_16_cuda_kernels.cu for the full
// derivation. MEASURED on real A100 hardware: a sharp regime change, not
// a smooth gradient -- essentially no benefit (even a small loss) below
// M=K=8192, then a stable +22% vs the scalar sibling from M=16384
// upward (five consecutive size points, not noise). Against cuBLAS:
// 0.70x-0.74x from M=16384 up (vs the scalar kernel's own 0.57x-0.61x
// there), still behind every other packed-format TRANSPOSE series at
// those sizes. Cause of the sharp transition not yet understood. See
// mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_kernel's own
// comment for the full numbers.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_vec4 above -- reuses the EXACT SAME __global__
// kernel unchanged, only the M_BLOCK-scaling grid-oversubscription target
// differs (kMvFp32_16TransposeTargetGridBlocks=2048 vs the shared 512
// every other TRANSPOSE dispatcher uses), testing whether more aggressive
// oversubscription helps this specific ALU-bound (not memory-bound)
// kernel -- see that constant's own comment in fp32_16_cuda_kernels.cu
// for the full derivation (SASS evidence: 864 instructions vs fp32_bf16's
// own 168 for the same structural kernel, dominated by the decode's own
// LOP3.LUT/IADD3/SHF/SEL, only 18 LDG). MEASURED on real A100 hardware:
// a MIXED result, not a clean win -- severe regressions at the smallest
// sizes (0.662x at M=K=1024, 0.684x at M=1536, from quadrupled atomicAdd
// contention on y[col] once M_BLOCK shrinks too far), a genuine +6%-+16%
// win at M=3072-12288, and no difference at M>=16384 (both variants
// converge to the same M_BLOCK there). NOT recommended for promotion as
// a blanket replacement -- see kMvFp32_16TransposeTargetGridBlocks's own
// comment in fp32_16_cuda_kernels.cu for the full numbers and derivation.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned(TransposeType trans, double alpha, dim<2> size,
                                                                  const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                  const float* x, size_t ld_x, double beta, float* y,
                                                                  size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_transpose_
// atomiconly_scaled_mblock_vec4_retuned above -- reuses the EXACT SAME
// __global__ kernel and target (2048), but ADDS a floor on how far
// M_BLOCK is allowed to shrink (kMvFp32_16TransposeMinMBlock=256),
// testing whether that recovers _retuned's own severe M=1024/1536
// regressions (from excessive atomicAdd contention once M_BLOCK shrank
// too far) while preserving its real M=3072-12288 win -- see that
// constant's own comment in fp32_16_cuda_kernels.cu for the full
// derivation. MEASURED on real A100 hardware: CONFIRMED -- recovers
// _retuned's own catastrophic M=1024/1536 regressions to near-parity
// (0.997x, 1.048x) while preserving essentially all of its mid-range
// win (+5.5%-+14.2% at M=3072-12288). A clean win over plain vec4 at
// every size tested, never worse by more than measurement noise -- a
// genuine promotion candidate.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(TransposeType trans, double alpha, dim<2> size,
                                                                       const uint16_t* wh, const uint16_t* wr,
                                                                       size_t ld_a, const float* x, size_t ld_x,
                                                                       double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of _retuned_floor above -- reuses
// the EXACT SAME __global__ kernel STRUCTURE and M_BLOCK-scaling (target
// =2048, floor=256) unchanged, only decoding via fp32_16_combine_
// branchless_wrapborrow instead of fp32_16_combine_branchless -- the SAME
// wraparound-subtraction technique fast_normal_wrapborrow (this file's
// own UNSAFE TRANSPOSE family) already proved a real win for, applied
// here to the SAFE decode's own identical is_carry/round_flag/mant_hi
// pattern (only the normal/extended paths' shared sub-computation
// changes; the degraded path is untouched -- unlike fp32_16_combine_
// branchless_nohw's own attempt at THAT path, which measured a ~2x
// regression in a different kernel, see that function's own comment).
// See fp32_16_combine_branchless_wrapborrow's own comment near the top
// of fp32_16_cuda_kernels.cu for the full derivation and its exhaustive
// bit-identity verification. SASS profile (cuobjdump, not real hardware):
// eliminates all 26 SEL instructions, ISETP 46->33, 864->832 total
// instructions vs plain _retuned_floor's own kernel -- but register usage
// goes the WRONG way (REG:32 vs REG:31, one MORE), unlike fast_normal_
// wrapborrow's own REG reduction -- see the kernel's own comment in
// fp32_16_cuda_kernels.cu for the full account.
//
// MEASURED on real A100 hardware, FULL square sweep (M=1024-65536): a
// clean, MONOTONIC win over _retuned_floor at every size, no exceptions
// -- +4.0% at M=1024 climbing smoothly to +12.1% at M=32768-65536, the
// fewer-instructions side winning decisively regardless of the extra
// register. PROMOTED: this is "TOMOS FP32/FP16" in scripts/plot_gemv_
// transpose_fp32_recon_vs_cublas_ref.m -- see the kernel's own comment in
// fp32_16_cuda_kernels.cu for the full numbers and the .sh's own
// METHOD_MARKERS comment for the promotion history.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y); }  // namespace experimental


// M_BLOCK-scaling target for _wrapborrow_retarget2 below -- reuses the
// EXACT SAME value (8192) as the UNSAFE fast_normal_wrapborrow family's
// own retarget2 target, since the M_BLOCK-scaling formula depends only on
// m, k, target, and shared constants (kMvTransposeMBlock ceiling,
// kMvTransposeWarpsPerBlock) -- none of which differ between this SAFE
// kernel and that UNSAFE one, so the grid/M_BLOCK this produces is BYTE-
// FOR-BYTE IDENTICAL to what that family already measured on real
// hardware: a real win at M=8192-24576ish, a real regression at
// M=3072-6144 (M_BLOCK crashing toward the shared 256 floor), a no-op at
// M>=32768 (shared ceiling binds regardless of target) -- see
// fp32_16_cuda_kernels.cu for the full derivation. HIGH-CONFIDENCE port,
// not a blind guess -- the only open question is whether the SAFE
// decode's own heavier per-element cost changes the magnitude, not the
// existence or location of the win/regression.
//
// EXPERIMENTAL, standalone A/B sibling of _wrapborrow above -- reuses the
// EXACT SAME __global__ kernel unchanged, only this higher target.
// UNTESTED on THIS kernel specifically.
namespace experimental { void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_retarget2(TransposeType trans, double alpha,
                                                                               dim<2> size, const uint16_t* wh,
                                                                               const uint16_t* wr, size_t ld_a,
                                                                               const float* x, size_t ld_x,
                                                                               double beta, float* y, size_t ld_y); }  // namespace experimental


// Standalone dispatcher for the SAFE (branchless-decode) TRANSPOSE
// family -- routes to _wrapborrow_retarget2 (target=8192) for M>=8192,
// plain _wrapborrow (target=2048, the currently-promoted "TOMOS FP32/
// FP16") otherwise. Direct port of the UNSAFE fast_normal_wrapborrow
// family's own _dispatch, which real hardware validated across the FULL
// square sweep with no regressions anywhere -- see that function's own
// comment for the history and the CAVEAT it carries (this M-only
// threshold validated on SQUARE shapes only). UNTESTED on THIS kernel
// specifically -- not yet promoted/plotted.
void mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              const uint16_t* wr, size_t ld_a,
                                                                              const float* x, size_t ld_x,
                                                                              double beta, float* y, size_t ld_y);

// SAME structure as mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock
// above, byte-for-byte, EXCEPT decoding via fp32_16_combine_branchless_
// nohw (fp32_16_pack.hpp) instead of fp32_16_combine_branchless -- that
// function reimplements the degraded (control==1, offset==7) sub-path as
// pure integer bit manipulation instead of a real hardware static_cast<
// __half>/static_cast<float> conversion (which SASS analysis found
// compiles to disproportionately expensive HADD2.F32 instructions). See
// that function's own comment in fp32_16_pack.hpp for the full
// derivation, and Fp32_16PackTest.CombineBranchlessNohwMatchesCombine /
// CombineBranchlessNohwDegradedEdgeCases in test/core/matrix/
// test_fp32_16.cpp for its own dedicated, already-passing exhaustive
// correctness verification, run independently of this kernel.
//
// Attacks the decode cost directly rather than amortizing it over more
// rows the way the 4-row/8-row row-vectorization experiments do.
// MEASURED WORSE on real A100 hardware: well under 300 GFlop/s across
// every tested size, vs ~500-720 GFlop/s for the plain kernel above --
// a real ~2x+ regression, larger than the ~1.7x its own SASS
// instruction-count comparison (1744 vs 1032) predicted, most likely
// from the degraded path's subnormal-renormalization step being a
// sequential dependency chain rather than independent work. Kept as a
// documented negative result, not deleted. See its own comment in
// fp32_16_cuda_kernels.cu, next to its own definition, for the full
// derivation.
namespace experimental { void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// "test" candidate targeting mv_fp32_16_accurate_narrow_vec_kernel (the
// kernel actually used at the real LSQR matrix's shape, M=1,000,000 K=
// 1000 -- every ready4/ready8/_nohw kernel above only fires below
// rows_op<kMvWideRowThreshold=131072 and falls through to this same
// narrow_vec kernel unchanged at this shape). cp.async double-buffers
// each thread's own next-column wh/wr ushort4 into shared memory while
// decoding the current column, to hide fp32_16_combine_branchless's own
// latency behind the next fetch instead of amortizing/cheapening the
// decode itself (the two levers every prior experiment in this family
// tried). See its own comment in fp32_16_cuda_kernels.cu for the full
// derivation.
namespace experimental { void mv_fp32_16_accurate_narrow_vec_cpasync(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// "test2" -- deepens mv_fp32_16_accurate_narrow_vec_cpasync above from a
// 2-buffer (1-ahead) to a 3-buffer (2-ahead) cp.async software pipeline,
// to see whether more outstanding fetches lets the scheduler hide more
// of fp32_16_combine_branchless's own decode latency (still the same
// SAFE decode, untouched). See its own comment in fp32_16_cuda_kernels.cu
// for the full derivation.
namespace experimental { void mv_fp32_16_accurate_narrow_vec_cpasync3(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// New "test2" -- deepening the pipeline (cpasync3 above) measured
// identical to "test"'s own single-stage lookahead, meaning cp.async had
// already fully hidden fetch latency and the remaining gap is decode ALU
// cost, not memory latency. Same "test" structure (4-rows/lane, 1-ahead
// cp.async), decoding via fp32_16_combine_branchless_no_overflow (skips
// the OVERFLOW escape path entirely) instead of the general
// fp32_16_combine_branchless -- valid only under the caller's guarantee
// that this matrix's data never needs wh's overflow encoding. See its
// own comment in fp32_16_cuda_kernels.cu for the exact savings and the
// correctness caveat.
namespace experimental { void mv_fp32_16_accurate_narrow_vec_cpasync_no_overflow(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// "test3" -- applies the SAME two levers (cp.async prefetch of wh/wr,
// no-overflow decode) to mv_fp32_16_ready4_atomiconly_scaled_kblock_
// kernel instead -- the kernel actually used for every SQUARE shape in
// this benchmark's sweep (rows_op always < kMvWideRowThreshold there),
// unlike the narrow_vec work above which only ever applies at large,
// tall/narrow M. See its own comment in fp32_16_cuda_kernels.cu for the
// full derivation.
namespace experimental { void mv_fp32_16_ready4_atomiconly_scaled_kblock_cpasync_no_overflow(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// "test4" -- combines "test3"'s two levers with 8-rows/lane register-
// blocking (mv_fp32_16_ready8_atomiconly_scaled_kblock_kernel's own
// design, previously found NOT to pan out in isolation -- worth
// re-testing now that decode is cheaper and prefetched). See its own
// comment in fp32_16_cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync_no_overflow(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// Composed "fast" dispatcher: routes to "test4" (wins for rows_op <
// kMvWideRowThreshold) or "test2" (mv_fp32_16_accurate_narrow_vec_
// cpasync_no_overflow, wins for rows_op >= kMvWideRowThreshold -- this
// is the branch the REAL LSQR restart-boundary matvec shape,
// M=1,000,000/K=1000, actually falls into) depending on the caller's
// shape. "test4" ALONE was found to make no difference to real LSQR
// solve time, because its own optimization only lives inside its own
// rows_op < kMvWideRowThreshold branch -- see this function's own
// comment in fp32_16_cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_accurate_fast(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

// "test5" -- byte-for-byte "test4" above, EXCEPT decoding via the
// GENERAL fp32_16_combine_branchless instead of fp32_16_combine_
// branchless_no_overflow -- SAFE on arbitrary data, no caller-guaranteed
// assumption needed. Isolates cp.async+8-rows' own contribution from the
// no-overflow decode's contribution. See its own comment in fp32_16_
// cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_ready8_atomiconly_scaled_kblock_cpasync(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental


// EXPERIMENTAL: 8-rows/lane extension of mv_fp32_16_accurate_ready4_
// atomiconly_scaled_kblock above -- pushes the same lever one step
// further (packs 8 adjacent rows per lane via a single 16-byte uint4
// load instead of 4 via uint64_t), testing whether amortizing the SAFE
// decode over even more rows keeps paying off, or whether register
// pressure/occupancy loss catches up with it at this depth. UNTESTED:
// no performance measurement has been run yet -- check this build's own
// register/spill count (cuobjdump -res-usage) before trusting any
// timing, since 8 accumulators plus the K-loop's own unroll depth is a
// real spill risk this kernel hasn't been tuned against yet. See its
// own comment in fp32_16_cuda_kernels.cu, next to its own definition,
// for the full derivation.
namespace experimental { void mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL: identical to mv_fp32_16_accurate except the wide (split-K)
// path's per-thread accumulator is float, not double -- matching a
// sibling project's (randls2.0) own composite-precision GEMV design,
// which reaches ~85-95% of native fp32 throughput where this codebase's
// double-accumulator wide kernel sits around 50% for the same shapes.
// See fp32_16_cuda_kernels.cu, next to mv_fp32_16_accurate_wide_facc_
// kernel's own definition, for the full A/B reasoning. UNTESTED: no
// performance measurement or correctness verification has been run yet
// -- float accumulation over up to tens of thousands of terms is a real
// precision trade-off, unlike mv_fp32_bf16_accurate below (which is
// untested only for lack of a measurement run, not a precision change).
namespace experimental { void mv_fp32_16_accurate_facc(TransposeType trans, double alpha, dim<2> size,
                              const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                              const float* x, size_t ld_x, double beta, float* y,
                              size_t ld_y); }  // namespace experimental


// fp32_bf16 full-reconstruction GEMV, where A is stored as wh/wr (see
// fp32_bf16_pack.hpp) and combined element-by-element for ~fp32
// accuracy. Byte-for-byte the same 3-way shape dispatch as mv_fp32_16_
// accurate above, decoding via fp32_bf16_combine instead. See fp32_16_
// cuda_kernels.cu, next to this function's own definition, for the full
// derivation. UNTESTED: no performance measurement or correctness
// verification has been run yet.
namespace experimental { void mv_fp32_bf16_accurate(TransposeType trans, double alpha, dim<2> size,
                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                            const float* x, size_t ld_x, double beta, float* y,
                            size_t ld_y); }  // namespace experimental


// Sizing helper for mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_
// workspace's own `workspace` parameter -- same formula as mv_fp32_16_
// accurate_workspace_size (shape-only, not precision-dependent), see that
// function's own comment. Call once per distinct (trans, size) before a
// run of repeated calls at that shape; 0 whenever this shape's own wide
// path would use the row-vectorized fast kernel (no buffer needed) or the
// tiled/narrow_vec paths (also no buffer needed) -- but always safe to
// size and pass a real buffer regardless of which internal path a given
// call takes.
namespace experimental { size_t mv_fp32_bf16_accurate_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Composite fp32/bf16 GEMV, identical dispatch to mv_fp32_bf16_accurate,
// EXCEPT the wide (split-K) path reads/writes a caller-provided
// `workspace` buffer instead of doing its own cudaMalloc/cudaFree on
// every call -- byte-for-byte the same relationship mv_fp32_16_accurate_
// workspace has to mv_fp32_16_accurate, see that function's own comment
// for the full malloc/free-per-call overhead rationale. Unlike mv_fp32_
// bf16_accurate_ready_atomiconly_scaled_kblock_workspace (below), this
// has NO row-vectorized fast path -- the wide path ALWAYS uses the
// classic column-split-and-reduce mechanism whenever rows_op is below
// kMvWideRowThreshold, regardless of trans/ld_a, so `workspace` is
// genuinely read and written on every call that reaches it. This is the
// one plotted as "TOMOS FP32/BF16 (workspace)". See fp32_16_cuda_
// kernels.cu, next to this function's own definition, for the full
// derivation.
namespace experimental { void mv_fp32_bf16_accurate_workspace(TransposeType trans, double alpha, dim<2> size,
                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                     const float* x, size_t ld_x, double beta, float* y,
                                     size_t ld_y, double* workspace); }  // namespace experimental


// "Regular"-based, fully general FP32/BF16 GEMV: mv_fp32_16_accurate_
// ready_atomiconly_scaled_kblock_fast_normal's own optimizations (row-
// vectorization, prescale+atomic-only write-back, K-scaled K_BLOCK) for
// the fast common case (NO_TRANSPOSE, ld_a==size[0], rows_op below
// kMvWideRowThreshold), combined with mv_fp32_16_accurate_workspace's own
// no-malloc discipline (a caller-provided `workspace` buffer, sized via
// mv_fp32_bf16_accurate_workspace_size) for every other case, rather than
// falling back to a malloc/free-per-call function the way mv_fp32_16_
// accurate_ready_atomiconly_scaled_kblock_fast_normal falls back to
// mv_fp32_16_accurate when its own fast-path layout restriction isn't
// met. Handles ALL TransposeType/ld_a combinations, like mv_fp32_16_
// accurate_workspace -- unlike that restricted "regular" sibling, this
// function never needs to fall back to a different, unrelated function
// for any shape.
//
// PROMOTED as plain "TOMOS FP32/BF16" (not "(workspace)"): every shape
// the benchmark exercises calls with NO_TRANSPOSE/ld_a==size[0], so
// `workspace` is never touched there (confirmed by direct A/B
// measurement) -- it beats the malloc-based mv_fp32_bf16_accurate
// baseline by 8-25% at every size tested, so it replaced that baseline
// as the plotted "TOMOS FP32/BF16" series. mv_fp32_bf16_accurate_
// workspace (above) is the genuinely-always-uses-workspace sibling now
// plotted as "(workspace)" instead. `workspace` remains a real, load-
// bearing parameter for any OTHER caller that hits this function's own
// general fallback path (transposed matrix, or ld_a != size[0]).
//
// No safe-normal-range data requirement and no UNSAFE-on-arbitrary-data
// contract, unlike every fp32_16 "_fast_normal" sibling in this file:
// bfloat16's decode (fp32_bf16_combine) needs no control-bit dispatch or
// exponent-range restriction in the first place (see fp32_bf16_pack.hpp's
// own header comment), so there is no unsafe shortcut being taken here --
// this is safe on arbitrary caller data, the same correctness bar as
// mv_fp32_bf16_accurate itself. See fp32_16_cuda_kernels.cu, next to this
// function's own definition and mv_fp32_bf16_ready_atomiconly_scaled_
// kblock_kernel's own definition just above it, for the full derivation.
//
// Also has a TRANSPOSE fast path now (trans==TRANSPOSE && ld_a==size[0]):
// warp-per-output-column/split-M/atomicAdd, ported from randls2.0's own
// sgemv_cm_transpose_splitM_atomic_xshared -- see mv_fp32_bf16_transpose_
// atomiconly_scaled_mblock_kernel's own comment in fp32_16_cuda_
// kernels.cu for the full derivation. Same safe-on-arbitrary-data
// property as the NO_TRANSPOSE path. UNTESTED (kernel-level, TRANSPOSE):
// no GEMV performance measurement has been run yet.
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y, double* workspace);

// Standalone TRANSPOSE-only sibling of mv_fp32_bf16_accurate_ready_
// atomiconly_scaled_kblock_workspace's own TRANSPOSE fast path above --
// byte-for-byte the SAME warp-per-output-column/split-M/atomicAdd/
// M_BLOCK-scaling structure as SAFE fp32_16's own "(vec4)" TRANSPOSE
// kernel, ported directly per this codebase's "base the design on
// FP32/FP16 (fast)" convention: float4-vectorized x-staging AND
// ushort4-vectorized wh/wr loads, runtime alignment-checked with a
// scalar fallback. SAME M_BLOCK-scaling (target=512, floor=1) as
// "(truncated, workspace)"'s own TRANSPOSE fast path, so this isolates
// ONLY the vectorization change. NOTE: fp32_bf16_combine is
// `(wh<<16)|wr`, a single shift+OR with no branch and no degraded/
// extended fallback (see fp32_bf16_pack.hpp) -- so unlike fp32_16's
// UNSAFE "(fast)" family, there is no analogous decode-level
// reformulation to port here; this decode is already maximally cheap.
//
// MEASURED on real A100 hardware (square M=K, M=1024-6144 so far): a
// dramatic, MONOTONICALLY GROWING win over "(truncated, workspace)" at
// every size tested -- +0.6% at M=1024 up to +34.6% at M=6144, no
// regressions, already beating cuBLAS outright from M=4096 up.
// PROMOTED: this is "TOMOS FP32/BF16 (truncated)" in scripts/plot_gemv_
// transpose_fp32_recon_vs_cublas_ref.m. Not yet tested at M>=8192.
void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                            const float* x, size_t ld_x, double beta, float* y,
                                                            size_t ld_y);

// EXPERIMENTAL, standalone A/B sibling of _vec4 above -- reuses the
// EXACT SAME __global__ kernel unchanged, only fp32_16's own already-
// validated M_BLOCK-scaling retuning (target=2048, floor=256 instead of
// 512/1) -- decode-independent, directly reused rather than re-derived.
// See the kernel's own comment in fp32_16_cuda_kernels.cu for the full
// derivation and fp32_16's own real-hardware track record for this
// exact retuning.
//
// MEASURED on real A100 hardware, two independent runs (square M=K,
// M=1024-8192): a real further win over plain _vec4 at M=3072-8192
// (+2.2% to +5.3%), roughly flat at M=1024, a real REGRESSION at M=2048
// in BOTH runs (-7.6%/-9.4%) that grid math does not predict via M_BLOCK
// discontinuity (both kernels floor-bound there, block counts scaling
// smoothly across neighboring sizes), and a noisier M=1536 (flat to a
// mild dip, likely near the noise floor). CROSS-VALIDATED at M=2048,
// likely NOT noise: the rounded-encoding sibling shows the SAME dip at
// the SAME M=2048 with a completely different decode function -- see
// the kernel's own comment in fp32_16_cuda_kernels.cu for the grid-math
// root-cause candidate (a triple power-of-2 alignment unique to M=2048).
// DOCUMENTED OPEN QUESTION, NOT promoted -- plain _vec4 above is, since
// it has no such caveat anywhere tested.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(TransposeType trans, double alpha,
                                                                          dim<2> size, const uint16_t* wh,
                                                                          const uint16_t* wr, size_t ld_a,
                                                                          const float* x, size_t ld_x, double beta,
                                                                          float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_bf16_transpose_
// atomiconly_scaled_mblock_vec4 above -- reuses the EXACT SAME
// __global__ kernel unchanged, only the higher target=8192 instead of
// 512 -- a direct, high-confidence port of the SAFE fp32_16 family's own
// retarget2 (decode-independent grid math). NOT built to stand alone --
// exists as _dispatch's own M>=8192 branch below, which only ever calls
// it where M>=8192 is already guaranteed, sidestepping the mid-M
// regression zone this target causes elsewhere. See the kernel's own
// comment in fp32_16_cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_retarget2(TransposeType trans, double alpha, dim<2> size,
                                                                      const uint16_t* wh, const uint16_t* wr,
                                                                      size_t ld_a, const float* x, size_t ld_x,
                                                                      double beta, float* y, size_t ld_y); }  // namespace experimental


// Standalone dispatcher for the plain/truncating-encoding TRANSPOSE
// family -- routes to plain _vec4 (target=512, the currently-promoted
// "TOMOS FP32/BF16 (truncated)") for M<8192, _retarget2 (target=8192)
// for M>=8192. Direct port of the SAFE fp32_16 family's own dispatch --
// the M<8192 branch deliberately stays on the no-caveat target=512
// kernel rather than _retuned_floor (which carries a confirmed,
// cross-validated regression at M=2048). Deliberately does NOT reach for
// a higher M_BLOCK ceiling the way the UNSAFE fp32_16 family's own
// _retarget2_bigblock/_hugeblock did -- that lever is a CONFIRMED,
// SEVERE negative result on real A100 hardware for this exact shared
// kernel structure (decode-independent occupancy math), so there is no
// reason to re-run it for bf16. See the kernel's own comment in
// fp32_16_cuda_kernels.cu for the full derivation. UNTESTED -- not yet
// promoted/plotted.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr,
                                                                     size_t ld_a, const float* x, size_t ld_x,
                                                                     double beta, float* y, size_t ld_y); }  // namespace experimental


// Rounded-encoding sibling of mv_fp32_bf16_accurate_ready_atomiconly_
// scaled_kblock_workspace above: byte-for-byte the same 3-case shape
// dispatch (fast row-vectorized path for NO_TRANSPOSE/ld_a==size[0]/
// rows_op<kMvWideRowThreshold, workspace-based wide fallback for other
// layouts at that row count, tiled/narrow_vec for larger row counts), but
// every internal path decodes via fp32_bf16_combine_rounded instead of
// fp32_bf16_combine. wh/wr MUST come from fp32_bf16_split_rounded, never
// plain fp32_bf16_split -- the two fp32_bf16 encodings are NOT
// interchangeable (wr's bit0 means something different in each: a real
// mantissa bit for the plain split, a round_flag for the rounded split).
// Reuses mv_fp32_bf16_accurate_workspace_size for `workspace` sizing
// (shape-only formula, not decode-specific). Plotted as plain "TOMOS
// FP32/BF16"; mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_
// workspace above is now plotted as "TOMOS FP32/BF16 (truncated)". See
// fp32_16_cuda_kernels.cu, next to this function's own definition and its
// four "_rounded" kernel siblings just above it, for the full derivation.
//
// Also has a TRANSPOSE fast path now (trans==TRANSPOSE && ld_a==size[0]):
// warp-per-output-column/split-M/atomicAdd, ported from randls2.0's own
// sgemv_cm_transpose_splitM_atomic_xshared -- see mv_fp32_bf16_transpose_
// atomiconly_scaled_mblock_rounded_kernel's own comment in fp32_16_cuda_
// kernels.cu for the full derivation. wh/wr MUST come from fp32_bf16_
// split_rounded for this path too, same requirement as NO_TRANSPOSE.
// UNTESTED (kernel-level, TRANSPOSE): no GEMV performance measurement
// has been run yet.
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                   const float* x, size_t ld_x, double beta, float* y,
                                                                   size_t ld_y, double* workspace);

// Standalone TRANSPOSE-only sibling of mv_fp32_bf16_accurate_ready_
// atomiconly_scaled_kblock_rounded's own TRANSPOSE fast path (its
// demoted predecessor) -- rounded-encoding counterpart of mv_fp32_bf16_
// transpose_atomiconly_scaled_mblock_vec4 (fp32_16_cuda_kernels.cu),
// byte-for-byte identical except decoding via fp32_bf16_combine_rounded
// instead of fp32_bf16_combine. wh/wr MUST come from fp32_bf16_split_
// rounded. SAME "base the design on FP32/FP16 (fast)" port: float4-
// vectorized x-staging AND ushort4-vectorized wh/wr loads. SAME
// M_BLOCK-scaling (target=512, floor=1) as its predecessor's own
// TRANSPOSE fast path, so this isolates ONLY the vectorization change.
//
// MEASURED on real A100 hardware (square M=K, M=1024-8192): confirmed --
// a dramatic, growing win over its predecessor, mirroring the plain/
// truncating encoding's own curve closely: roughly flat at M=1024/1536
// (0.972x-1.000x), then +4.3% at M=2048 climbing to +21.6% at M=8192,
// no regressions anywhere tested. DEMOTED now that mv_fp32_bf16_
// transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch (below)
// sources "TOMOS FP32/BF16" instead -- kept as its own visible series.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                    const uint16_t* wh, const uint16_t* wr,
                                                                    size_t ld_a, const float* x, size_t ld_x,
                                                                    double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of _vec4_rounded above -- reuses
// the EXACT SAME __global__ kernel unchanged, only fp32_16's own
// already-validated M_BLOCK-scaling retuning (target=2048, floor=256) --
// rounded-encoding counterpart of mv_fp32_bf16_transpose_atomiconly_
// scaled_mblock_vec4_retuned_floor, built partly to check whether that
// sibling's own M=2048 anomaly is decode-specific or general.
//
// MEASURED on real A100 hardware (square M=K, M=1024-8192): CONFIRMS the
// M=2048 anomaly is NOT decode-specific -- this kernel, with a
// completely different decode function, shows the SAME regression at
// the SAME M=2048 (-4.7% vs plain _vec4_rounded), while winning at every
// other size tested (+0.2% to +10.1% at M=1536-8192). DOCUMENTED OPEN
// QUESTION, NOT promoted -- plain _vec4_rounded above is.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Rounded-encoding counterpart of mv_fp32_bf16_transpose_atomiconly_
// scaled_mblock_vec4_retarget2 (fp32_16_cuda_kernels.cu) -- reuses the
// EXACT SAME __global__ kernel (mv_fp32_bf16_transpose_atomiconly_
// scaled_mblock_vec4_rounded_kernel) unchanged, only target=8192. NOT
// built to stand alone -- exists as _rounded_dispatch's own M>=8192
// branch below. See the plain/truncating encoding's own _retarget2
// comment for the full derivation.
namespace experimental { void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_retarget2(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Standalone dispatcher for the rounded-encoding TRANSPOSE family --
// routes to plain _vec4_rounded (target=512, the demoted predecessor
// above) for M<8192, _rounded_retarget2 (target=8192) for M>=8192.
// Rounded-encoding counterpart of the plain/truncating encoding's own
// _dispatch -- see its own comment for the full derivation.
//
// MEASURED on real A100 hardware at M=8192 (grid math confirms a real
// M_BLOCK=8192/gridM=1 vs M_BLOCK=1024/gridM=8 difference there, not
// noise): +12.7% over plain _vec4_rounded (689.5 vs 611.9 GFlop/s).
// PROMOTED: this is "TOMOS FP32/BF16" in scripts/plot_gemv_transpose_
// fp32_recon_vs_cublas_ref.m. CAVEAT: full confirmation across M=12288-
// 24576 (where the win is predicted to taper to a no-op by M=24576) is
// still pending.
void mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_rounded_dispatch(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, const uint16_t* wr, size_t ld_a,
    const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// See the matching banner comment in fp32_16_cuda_kernels.cu -- everything
// below except the thin forwarding declarations right after this GEMM
// range (unrelated GEVM content resumes after those) is declared inside
// tomos::cuda::experimental.
namespace experimental {

// C = alpha * op(A) * op(B) + beta * C, where A is stored as wh/wr.
void mm_fp32_16_accurate(TransposeType transA, TransposeType transB,
                         double alpha, dim<2> size_A, const uint16_t* wh,
                         const uint16_t* wr, size_t ld_a, dim<2> size_B,
                         const float* B, size_t ld_b, double beta, float* C,
                         size_t ld_c);

// Same as mm_fp32_16_accurate, but shared-memory-tiled: each wh/wr pair is
// decoded into a float once, when it's loaded into shared memory, instead
// of once per use. NO_TRANSPOSE x NO_TRANSPOSE only -- falls back to
// mm_fp32_16_accurate for any transposed operand. See fp32_16_cuda_
// kernels.cu for mm_fp32_16_tiled_kernel.
void mm_fp32_16_tiled(TransposeType transA, TransposeType transB, double alpha,
                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                      double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_tiled, plus vectorizing A's wh/wr load (ushort4,
// decoded via fp32_16_combine) and B's load (float4), and the compute
// loop's shared-memory reads (float4) -- mirrors gemm_fp32_tiled_cpasync_
// vec2's vectorizations in the plain-fp32 series. Requires ld_a, m, ld_b,
// and k to all be multiples of 4 (checked internally); falls back to
// mm_fp32_16_tiled otherwise. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_tiled_vec_kernel.
void mm_fp32_16_tiled_vec(TransposeType transA, TransposeType transB, double alpha,
                          dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                          size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                          double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_tiled_vec, but A's raw wh/wr bytes are prefetched one
// tile ahead via cp.async into a staging buffer, hiding the global memory
// fetch (not the decode itself, which can't run inside a cp.async
// transfer) behind the previous tile's compute. Same alignment/boundary
// requirements as mm_fp32_16_tiled_vec; falls back to it otherwise. See
// fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_vec_kernel.
void mm_fp32_16_tiled_cpasync_vec(TransposeType transA, TransposeType transB, double alpha,
                                  dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                  size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                  double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_tiled_cpasync_vec, but with the K-tile doubled from 8
// to 16 -- gives B's vectorized load 4 threads sharing each column instead
// of 2 (a 64-byte coalesced chunk instead of 32), mirroring gemm_fp32_
// tiled_cpasync_vec3's improvement in the plain-fp32 series. Same
// alignment/boundary requirements; falls back through it as needed. See
// fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_vec2_kernel.
void mm_fp32_16_tiled_cpasync_vec2(TransposeType transA, TransposeType transB, double alpha,
                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                   double beta, float* C, size_t ld_c);

// EXPERIMENTAL: identical to mm_fp32_16_tiled_cpasync_vec2, but decode_A
// uses fp32_16_combine_branchless instead of fp32_16_combine, to test
// whether removing per-element decode branch divergence is worth anything
// on top of vec2's memory-side vectorizations. Same alignment/boundary
// requirements; falls back the same way. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_tiled_cpasync_vec2_branchless_kernel.
void mm_fp32_16_tiled_cpasync_vec2_branchless(TransposeType transA, TransposeType transB,
                                              double alpha, dim<2> size_A, const uint16_t* wh,
                                              const uint16_t* wr, size_t ld_a, dim<2> size_B,
                                              const float* B, size_t ld_b, double beta, float* C,
                                              size_t ld_c);

// Bigger tile (BM=128, BN=256, BK=16) than mm_fp32_16_tiled_cpasync_vec2 --
// doubling BN doubles how many threads reuse each decoded A value (BN/TN),
// halving the effective decode cost per FLOP. Uses dynamic shared memory
// (the resulting 48KB footprint sits exactly at the static-shared-memory
// limit) via cudaFuncSetAttribute at first call. Same alignment/boundary
// requirements as vec2; falls back to it otherwise. See fp32_16_cuda_
// kernels.cu for mm_fp32_16_tiled_cpasync_vec3_kernel.
void mm_fp32_16_tiled_cpasync_vec3(TransposeType transA, TransposeType transB, double alpha,
                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                   double beta, float* C, size_t ld_c);

// Same block/grid tile as mm_fp32_16_tiled_cpasync_vec2 (BM=128, BN=128,
// BK=16 -- same shared memory, same number of blocks), but TN drops from
// 8 to 4, doubling how many threads share each decoded A value (BN/TN:
// 16 -> 32) without shrinking the grid the way vec3's bigger block tile
// did. Same alignment/boundary requirements; falls back to vec2. See
// fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_vec4_kernel.
void mm_fp32_16_tiled_cpasync_vec4(TransposeType transA, TransposeType transB, double alpha,
                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                   double beta, float* C, size_t ld_c);

// A controlled test of cutlass-style warp tiling: identical block tile,
// thread count, and per-thread work to mm_fp32_16_tiled_cpasync_vec2 --
// the only change is the thread-to-output-position mapping, from a flat
// (tx, ty) grid to an explicit warp (4x2) x lane (4x8) decomposition, so
// each warp owns a contiguous, warp-aligned region of the block's output
// instead of straddling a ty boundary. Same alignment/boundary
// requirements as vec2; falls back to it otherwise. See fp32_16_cuda_
// kernels.cu for mm_fp32_16_tiled_cpasync_warptile_kernel.
void mm_fp32_16_tiled_cpasync_warptile(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// Identical to mm_fp32_16_tiled_cpasync_warptile except the warp/lane
// grid's shape: 2x4 warps with 8x4 lanes (was 4x2 warps, 4x8 lanes) --
// same total resources, transposed aspect ratio. Same alignment/boundary
// requirements; falls back to vec2. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_tiled_cpasync_warptile2_kernel.
void mm_fp32_16_tiled_cpasync_warptile2(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);

// Continues the warptile -> warptile2 trend (taller warp tiles won) to
// its extreme: 1x8 warps x 16x2 lanes, giving each warp a 128x16 tile.
// Same resources/load logic as warptile/warptile2; falls back to vec2.
// See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_warptile3_kernel.
void mm_fp32_16_tiled_cpasync_warptile3(TransposeType transA, TransposeType transB, double alpha,
                                        dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                        size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                        double beta, float* C, size_t ld_c);

// A genuinely different mechanism: B gets real cp.async prefetching too
// (every other kernel here keeps B synchronous, since its natural
// K-contiguous layout conflicts with Bs's N-fast compute layout). Stages
// B's raw bytes into a K-contiguous buffer matching its own layout, then
// does a cheap shared-to-shared transpose into the real Bs -- mirroring
// A's decode step. Uses dynamic shared memory (64KB) via
// cudaFuncSetAttribute. Same alignment/boundary requirements as vec2;
// falls back to it otherwise. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_tiled_cpasync_dualbuf_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                      double beta, float* C, size_t ld_c);

// Byte-for-byte identical to mm_fp32_16_tiled_cpasync_dualbuf except A's
// decode calls fp32_16_combine_truncated instead of fp32_16_combine (no
// round_flag correction at all). CORRECTNESS CONTRACT: wh/wr MUST come
// from fp32_16_split_truncated, never fp32_16_split/fp32_16_split_
// clamped -- see fp32_16_combine_truncated's own comment in fp32_16_
// pack.hpp. Exists to A/B the two decode strategies' cost head-to-head
// at the same problem sizes. See fp32_16_cuda_kernels.cu for mm_fp32_16_
// tiled_cpasync_dualbuf_truncated_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType transA, TransposeType transB, double alpha,
                                                dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                double beta, float* C, size_t ld_c);

// "TOMOS FP32/FP16 (fast)": byte-for-byte identical to mm_fp32_16_tiled_
// cpasync_dualbuf EXCEPT A's decode calls fp32_16_combine_fast_normal
// instead of fp32_16_combine, unconditionally (no per-tile validity
// check, unlike mm_fp32_16_tiled_cpasync_dualbuf_adaptive below).
// CORRECTNESS CONTRACT (UNSAFE ON ARBITRARY DATA): every element of A
// must have control bit (bit0 of wh) == 0 -- see fp32_16_combine_fast_
// normal's own contract in fp32_16_pack.hpp; never checked here. Same
// alignment/boundary requirements as vec2; falls back to it otherwise.
// See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_dualbuf_
// fast_normal_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                   double beta, float* C, size_t ld_c);

// Same tile/warp/lane geometry as mm_fp32_16_tiled_cpasync_dualbuf, but
// transform_tile first does a block-wide __syncthreads_or reduction over
// each tile's staged control bits to check whether the WHOLE tile is
// control=0 ("normal" exponent range). When it is, decode uses fp32_16_
// combine_fast_normal (skips the offset/direction/degraded dispatch);
// otherwise it falls back to the full fp32_16_combine, same as the plain
// dualbuf kernel always does. Expected to be bit-identical to mm_fp32_16_
// tiled_cpasync_dualbuf on the same input -- this only trades one extra
// block barrier per tile for skipping the dispatch when a tile doesn't
// need it. See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_
// dualbuf_adaptive_kernel, and fp32_16_vs_fp32_bf16_decode_investigation.
// tex (repo root) for the measurement/SASS background this responds to.
void mm_fp32_16_tiled_cpasync_dualbuf_adaptive(TransposeType transA, TransposeType transB, double alpha,
                                               dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                               size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                               double beta, float* C, size_t ld_c);

// Same idea as mm_fp32_16_tiled_cpasync_dualbuf_adaptive, but the
// fast/full decode decision is made per-WARP (__any_sync) instead of
// per-BLOCK (__syncthreads_or) -- valid because a_kk_base (=tid/32=
// warp_id) makes each warp's 2 A-rows disjoint from every other warp's,
// so no block-wide agreement is needed. Exists to test whether avoiding
// the block-wide adaptive kernel's extra barrier (which measured a
// consistent ~1.3-2.1% regression vs always doing the full decode)
// recovers a real win. See fp32_16_cuda_kernels.cu for mm_fp32_16_
// tiled_cpasync_dualbuf_adaptive_warp_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf_adaptive_warp(TransposeType transA, TransposeType transB, double alpha,
                                                     dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                     size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                     double beta, float* C, size_t ld_c);

// Attacks the SAME redundant-decode problem as the adaptive kernels
// above from a different angle: instead of making each block's decode
// cheaper, this halves how many blocks redundantly decode a given
// M-row range by having each block cover TWO adjacent N-tiles at once,
// decoding A into shared memory ONCE per K-step and reusing it for both
// N-groups' accumulation. Needs roughly double the accumulator register
// footprint (two live acc[TM][TN] arrays per thread), so this targets
// only 1 block/SM (__launch_bounds__(256,1)) instead of the plain
// kernel's 2 -- trading occupancy for a halved decode-redundancy
// factor. See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_
// dualbuf_multin_kernel, and fp32_16_vs_fp32_bf16_decode_investigation.
// tex (repo root) for the redundant-decode background.
void mm_fp32_16_tiled_cpasync_dualbuf_multin(TransposeType transA, TransposeType transB, double alpha,
                                             dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                             size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                             double beta, float* C, size_t ld_c);

// A third, more radical attack on the same redundant-decode problem:
// eliminates it entirely rather than reducing it. Decodes A into a
// plain fp32 GLOBAL-memory scratch buffer, kMmFp32_16MaterializeChunk
// Rows rows at a time (bounding scratch memory independent of M, unlike
// materializing the whole M x K matrix at once), then hands each chunk
// straight to gemm_fp32_tiled_cpasync_dualbuf -- this codebase's
// fastest plain fp32 GEMM, which needs no decode logic at all. Total
// decode work becomes O(M*K), independent of N, versus the fused
// kernels' O(M*K*N/BN). See fp32_16_cuda_kernels.cu for
// mm_fp32_16_materialize_chunk_kernel and mm_fp32_16_gemm_materialize.
void mm_fp32_16_gemm_materialize(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Identical structure to mm_fp32_16_gemm_materialize above EXCEPT the
// second-stage native GEMM call is gemm_fp32_tiled_syncpipe4_bk8_
// warptile_nostride_snake_midpoint_exact (the current best native fp32
// kernel -- 0.977x vs CUTLASS at N=8192, 0.982x at N=32768, beating
// both gemm_fp32_tiled_cpasync_dualbuf and gemm_fp32_tiled_syncpipe3)
// instead of gemm_fp32_tiled_cpasync_dualbuf. See fp32_16_cuda_kernels.cu
// for mm_fp32_16_gemm_materialize_midpoint_exact.
void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// "TOMOS FP32/FP16 (workspace, fast)": identical two-phase materialize-
// then-native-GEMM structure to mm_fp32_16_gemm_materialize_midpoint_
// exact above (same chunking, same unmodified GEMM compute stage),
// except the chunk-decode pass calls fp32_16_combine_fast_normal
// instead of fp32_16_combine, unconditionally. CORRECTNESS CONTRACT
// (UNSAFE ON ARBITRARY DATA): every element of A must have control bit
// (bit0 of wh) == 0 -- see fp32_16_combine_fast_normal's own contract in
// fp32_16_pack.hpp; never checked here. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_materialize_chunk_fast_normal_kernel.
void mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// EXPERIMENT: same materialize strategy as the kernel above, but the
// chunk-decode pass is vectorized -- 4 consecutive rows per thread via
// ushort4 reads / float4 writes instead of one scalar element per
// thread, with a scalar fallback for a chunk's final partial group of
// <4 rows. Decode is the ENTIRE extra cost materialize pays over plain
// native fp32 (a pure memory-bandwidth-bound kernel, no GEMM tiling
// register pressure), so this targets that directly rather than the
// GEMM stage, which is unchanged. Falls back to the plain scalar-decode
// mm_fp32_16_gemm_materialize_midpoint_exact when ld_a isn't a multiple
// of 4 (can't safely issue aligned ushort4 reads of wh/wr). See fp32_16_
// cuda_kernels.cu for mm_fp32_16_materialize_chunk_vec_kernel and its
// own comment on the alignment contract this relies on.
void mm_fp32_16_gemm_materialize_midpoint_exact_vec(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// A genuinely fused counterpart to mm_fp32_16_gemm_materialize_midpoint_
// exact above: ports gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
// snake_midpoint_exact's ENTIRE tiling structure (BK=8, CUTLASS
// RowMajorInterleaved<2> warp/lane geometry, 4-stage pipeline, rolling-
// pointer addressing, snake FFMA traversal, interleaved i==4 prefetch
// checkpoint, bounds-check-free "exact" fast path) into this file and
// decodes wh/wr straight into the tile pipeline's shared-memory A
// buffer -- the same "decode once per BK-tile into shared memory"
// placement mm_fp32_16_tiled_cpasync_dualbuf_kernel's own transform_tile
// already uses, just retargeted at midpoint_exact's newer tile shape
// instead of this file's own BK=16/2-stage/plain-row-major one. Falls
// back to mm_fp32_16_gemm_materialize_midpoint_exact (not a second,
// bounds-checked fused sibling) for non-exact-multiple problem sizes.
// See fp32_16_cuda_kernels.cu for mm_fp32_16_syncpipe4_bk8_warptile_
// nostride_snake_midpoint_exact_kernel.
void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Byte-for-byte identical to mm_fp32_16_syncpipe4_bk8_warptile_nostride_
// snake_midpoint_exact above EXCEPT its decode calls fp32_16_combine_
// truncated instead of fp32_16_combine -- same relationship mm_fp32_16_
// tiled_cpasync_dualbuf_truncated has to plain dualbuf. CORRECTNESS
// CONTRACT: wh/wr fed to this kernel MUST come from fp32_16_split_
// truncated, never fp32_16_split/fp32_16_split_clamped. Falls back to
// mm_fp32_16_tiled_cpasync_dualbuf_truncated (NOT mm_fp32_16_gemm_
// materialize_midpoint_exact -- that fallback decodes via plain fp32_
// 16_combine, which would silently corrupt truncated-split data) for
// non-exact-multiple problem sizes. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_
// truncated_kernel.
void mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_truncated(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// FP32/BF16 counterpart to mm_fp32_16_syncpipe4_bk8_warptile_nostride_
// snake_midpoint_exact above -- same tiling structure, EXCEPT it decodes
// via fp32_bf16_combine instead of fp32_16_combine (see fp32_bf16_
// pack.hpp -- same two-uint16_t wh/wr split as fp32_16, but simpler:
// unconditional, exact reconstruction, no degraded/approximate branch).
// Deliberately declared in this file alongside its fp32_16 sibling
// rather than in fp32_bf16_cuda_kernels.cuh, at the user's request.
// Falls back to mm_fp32_bf16_tiled_cpasync_dualbuf (fp32_bf16's own
// existing bounds-checked kernel -- there's no bf16 "materialize_
// midpoint_exact" equivalent) for non-exact-multiple problem sizes. See
// fp32_16_cuda_kernels.cu for mm_fp32_bf16_syncpipe4_bk8_warptile_
// nostride_snake_midpoint_exact_kernel.
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Naive, always-correct-for-any-shape fallback for the ROUNDED fp32_bf16
// format -- mirrors mm_fp32_bf16_accurate (fp32_bf16_cuda_kernels.cuh)
// exactly EXCEPT it decodes via fp32_bf16_combine_rounded. Necessary
// because, unlike fp32_16 (where fp32_16_combine safely decodes BOTH
// its own rounded output AND fp32_16_split_truncated's output, since
// round_flag=0 makes its correction a no-op), fp32_bf16's plain and
// rounded formats are mutually incompatible: fp32_bf16_combine silently
// produces wrong results on rounded-encoded data whenever round_flag=1.
// Used as the transpose/misaligned/malloc-failure fallback for both
// mm_fp32_bf16_gemm_materialize_rounded and mm_fp32_bf16_syncpipe4_bk8_
// warptile_nostride_snake_midpoint_exact_rounded below. See fp32_16_
// cuda_kernels.cu for mm_fp32_bf16_accurate_rounded_kernel.
void mm_fp32_bf16_accurate_rounded(TransposeType transA, TransposeType transB,
                           double alpha, dim<2> size_A, const uint16_t* wh,
                           const uint16_t* wr, size_t ld_a, dim<2> size_B,
                           const float* B, size_t ld_b, double beta, float* C,
                           size_t ld_c);

// Rounded-encoding counterpart to mm_fp32_16_gemm_materialize_midpoint_
// exact (fp32_16_cuda_kernels.cu) -- identical materialize structure
// (decode A into a scratch fp32 buffer up front, then call gemm_fp32_
// tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact as a black
// box) EXCEPT the decode is fp32_bf16_combine_rounded and the transpose/
// malloc-failure fallback is mm_fp32_bf16_accurate_rounded above (not
// the plain-decode mm_fp32_bf16_accurate). Since materialize's decode
// pass is a simple per-element kernel (no vectorized loads, no GEMM
// tiling register pressure), it sidesteps mm_fp32_bf16_syncpipe4_bk8_
// warptile_nostride_snake_midpoint_exact_rounded's own real spill
// entirely -- same "decode once, then hand off to the already-proven
// 0-spill native GEMM" strategy that made materialize_midpoint_exact win
// outright for fp32_16. See fp32_16_cuda_kernels.cu for mm_fp32_bf16_
// materialize_rounded_chunk_kernel.
void mm_fp32_bf16_gemm_materialize_rounded(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Rounded-encoding counterpart to mm_fp32_bf16_syncpipe4_bk8_warptile_
// nostride_snake_midpoint_exact above -- same tiling structure, EXCEPT
// it decodes via fp32_bf16_combine_rounded instead of fp32_bf16_combine
// (see fp32_bf16_pack.hpp's own comment on fp32_bf16_split_rounded/
// fp32_bf16_combine_rounded): wh is formed by ROUNDING (to nearest, ties
// to even) rather than truncating, so fp32_bf16_bf16_only(wh) is a
// properly-rounded standalone bf16 value; wr's LSB carries a round_flag
// bit instead of the original mantissa's own LSB, so full reconstruction
// here is NOT bit-exact the way the plain kernel above is -- only
// accurate to within one ULP of the full 23-bit fp32 mantissa (utterly
// negligible next to bf16's own precision). CORRECTNESS CONTRACT: wh/wr
// fed to this kernel MUST come from fp32_bf16_split_rounded, never
// fp32_bf16_split. Falls back to mm_fp32_bf16_accurate_rounded above
// (transpose/misaligned) or mm_fp32_bf16_gemm_materialize_rounded above
// (non-exact-multiple problem sizes) -- NEVER to the plain-decode mm_
// fp32_bf16_accurate/mm_fp32_bf16_tiled/mm_fp32_bf16_tiled_cpasync_
// dualbuf, which would silently corrupt rounded-split data. Compiles
// with a real spill (REG:128 STACK:56, 23 STL + 11 LDL) unlike the
// plain kernel above's clean 0-spill compile -- investigated at length
// and not resolved (see mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_
// snake_midpoint_exact_rounded_kernel's own comment in fp32_16_cuda_
// kernels.cu for the full investigation, including why mm_fp32_bf16_
// gemm_materialize_rounded above exists as the spill-free alternative).
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// EXPERIMENT: same as the kernel above, but load_global_advance first
// checks -- once per BK=8 tile, via a block-wide __syncthreads_or
// reduction -- whether the ENTIRE tile has any rounded element at all;
// when it doesn't, decode collapses to the plain, 0-spill fp32_bf16_
// combine (an exact substitute whenever round_flag=0 -- verified host-
// side), otherwise falls back to fp32_bf16_combine_rounded. UNLIKE the
// bounds-check removal that produced the exact-fast-path family, this
// is not a compile-time-provable elimination -- it trades REAL extra
// block-wide barriers (4 new BAR.RED.OR, one per load_global_advance
// call site, on top of the 2 plain BAR.SYNC both kernels already share)
// against sometimes skipping the corrected decode's extra live values,
// which could easily cost more than it saves by undermining the 4-
// stage pipeline's own latency-hiding. Measured: REG:128 STACK:16, 3
// STL + 2 LDL, vs the plain kernel's REG:128 STACK:56, 23 STL + 11 LDL.
// Same correctness contract as the non-adaptive kernel above. See
// fp32_16_cuda_kernels.cu for mm_fp32_bf16_syncpipe4_bk8_warptile_
// nostride_snake_midpoint_exact_rounded_adaptive_kernel.
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Same idea as the kernel above, but the "does this need the corrected
// decode" decision is made per-WARP via __any_sync instead of per-
// BLOCK via __syncthreads_or -- same technique as mm_fp32_16_tiled_
// cpasync_dualbuf_adaptive_warp, valid here because a_kk = tid /
// (kMmFp32_16MidExactBM / 4) = tid / 32 = warp_id, so a_kk (and
// therefore which A row a warp decodes) is warp-uniform: no other
// warp's decode work depends on this warp's choice. __any_sync compiles
// to VOTE.ANY (confirmed via SASS) -- a cheap warp-only predicate
// combine, NOT a barrier instruction at all -- so unlike the block-wide
// kernel's 4 new BAR.RED.OR, this adds ZERO new barrier-class
// instructions (only 4 VOTE.ANY, no shared-memory round trip, no wait
// on the other 7 warps). Measured: REG:128 STACK:32, 6 STL + 6 LDL --
// a smaller spill reduction than the block-wide kernel's STACK:16, but
// at essentially none of its synchronization risk. Same correctness
// contract as the non-adaptive kernel. See fp32_16_cuda_kernels.cu for
// mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_
// rounded_adaptive_warp_kernel.
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_rounded_adaptive_warp(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_tiled_cpasync_dualbuf, but with BK doubled from 16 to
// 32 -- 128KB/block shared memory, requiring dynamic shared memory and
// likely limiting occupancy to 1 block/SM (predicted risk; see
// fp32_16_cuda_kernels.cu for the measured outcome). Same alignment/
// boundary requirements; falls back to dualbuf otherwise. See fp32_16_
// cuda_kernels.cu for mm_fp32_16_tiled_cpasync_dualbuf2_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf2(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// A genuine 3-stage software pipeline (tile i+2 prefetched while tile i+1
// is mid-transform and tile i is being computed) at BM=64 (half of
// dualbuf's 128, to keep the tripled buffers within a 2-blocks/SM shared
// memory budget; BN/BK stay at dualbuf's 128/16). Note this measures
// "smaller tile + deeper pipeline" together, not pipelining alone. Same
// alignment/boundary requirements as vec2; falls back to dualbuf
// otherwise. See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_
// tripipe_kernel.
void mm_fp32_16_tiled_cpasync_tripipe(TransposeType transA, TransposeType transB, double alpha,
                                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                      double beta, float* C, size_t ld_c);

// The inverse of dualbuf2: same BM=BN=128 tile and warp/lane shape as
// dualbuf, but BK shrunk from 16 to 8 instead of grown to 32. Shared
// memory drops to ~32KB/block (fits the static 48KB limit, no dynamic
// shared memory needed), and A/B loads become single-pass (unlike
// dualbuf's two-pass loads at BK=16). Tests whether occupancy headroom
// beyond dualbuf's 2-blocks/SM is worth more than BK=16's wider B
// coalescing. Same alignment/boundary requirements as vec2; falls back
// to dualbuf otherwise. See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_
// cpasync_dualbuf3_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf3(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// Same real dual-cp.async machinery as dualbuf (BN=128, BK=16, decode/
// transpose logic unchanged), but BM shrunk from 128 to 64 -- doubling
// the row-tile count of blocks in the launch grid, to fix the SM
// underutilization dualbuf shows at small N (its 128x128 tile produces
// too few blocks to fill a large GPU's SM count). Isolates the tile-
// shape/grid-fill axis that mm_fp32_16_tiled_cpasync_tripipe's BM=64
// experiment confounded with a deeper pipeline. Same alignment/boundary
// requirements as vec2; falls back to dualbuf otherwise. See fp32_16_
// cuda_kernels.cu for mm_fp32_16_tiled_cpasync_dualbuf4_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf4(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// The mirror of dualbuf4: shrinks BN (128->64) instead of BM, same
// BM=128/BK=16 and real dual-cp.async machinery for both A and B. Tests
// whether dualbuf4's mid-range N win on an A100 (not just small-N as
// first measured elsewhere) is a general small-tile effect visible on
// the N-axis too, or specific to the M-axis shrink. Same alignment/
// boundary requirements as vec2; falls back to dualbuf otherwise. See
// fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_dualbuf5_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf5(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// Same tile shape/shared-memory scheme as dualbuf; only the inner k-loop
// changes -- register-double-buffers the shared->register load (prefetch
// kk+1's a_reg/b_reg while computing on kk's) instead of relying on
// ptxas's own scheduling inside the fully-unrolled loop. Same alignment/
// boundary requirements as vec2; falls back to dualbuf otherwise. See
// fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_dualbuf6_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf6(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

// Split-K on dualbuf's exact 128x128/BK=16 tile: instead of one block
// reducing the full K range per output tile, P blocks each reduce a K/P
// slice and atomicAdd their contribution into C, multiplying the launched
// block count without shrinking the output tile the way dualbuf4/5 did.
// P is chosen per call from M/N/K and the device's SM count. Requires a
// separate prescale pass (applies beta*C once, before any atomicAdd) --
// handled internally. Same alignment/boundary requirements as vec2; falls
// back to dualbuf otherwise. See fp32_16_cuda_kernels.cu for mm_fp32_16_
// tiled_cpasync_splitk_kernel and mm_fp32_16_splitk_prescale_kernel.
void mm_fp32_16_tiled_cpasync_splitk(TransposeType transA, TransposeType transB, double alpha,
                                     dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                     size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                     double beta, float* C, size_t ld_c);

// Same tile/warp shape as dualbuf, same real cp.async + decode for A, but
// B reverts to a synchronous one-step global-read-then-transpose write
// (matching gemm_fp32_tiled_cpasync_vec3_kernel's B path in mm_cuda_
// kernels.cu) instead of dualbuf's two-step cp.async-into-staging-then-
// transform. Tests whether dualbuf's B staging is worth its extra shared-
// memory round trip now that profiling shows dualbuf is compute-bound
// (86% SM throughput, 20% DRAM), not memory-latency-bound. Same
// alignment/boundary requirements as vec2; falls back to dualbuf
// otherwise. See fp32_16_cuda_kernels.cu for mm_fp32_16_tiled_cpasync_
// dualbuf7_kernel.
void mm_fp32_16_tiled_cpasync_dualbuf7(TransposeType transA, TransposeType transB, double alpha,
                                       dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                       size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                       double beta, float* C, size_t ld_c);

}  // namespace experimental

// Thin forwarding declarations -- see the matching definitions in
// fp32_16_cuda_kernels.cu. Same signatures as before this reorganization.
void mm_fp32_16_accurate(TransposeType transA, TransposeType transB,
                         double alpha, dim<2> size_A, const uint16_t* wh,
                         const uint16_t* wr, size_t ld_a, dim<2> size_B,
                         const float* B, size_t ld_b, double beta, float* C,
                         size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                      dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                      size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                      double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf_truncated(TransposeType transA, TransposeType transB, double alpha,
                                                dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                double beta, float* C, size_t ld_c);
void mm_fp32_16_tiled_cpasync_dualbuf_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                                   dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                                   size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                                   double beta, float* C, size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);
void mm_fp32_bf16_gemm_materialize_rounded(TransposeType transA, TransposeType transB, double alpha,
                                 dim<2> size_A, const uint16_t* wh, const uint16_t* wr,
                                 size_t ld_a, dim<2> size_B, const float* B, size_t ld_b,
                                 double beta, float* C, size_t ld_c);

// Same as mv_fp32_16_accurate, but using only wh (fp32_16_half_only decode,
// see fp32_16_pack.hpp): the dot product accumulates in double, giving
// fp16-level per-element accuracy but avoiding fp16 accumulation error.
void mv_fp32_16_fast(TransposeType trans, double alpha, dim<2> size,
                     const uint16_t* wh, size_t ld_a, const float* x,
                     size_t ld_x, double beta, float* y, size_t ld_y);

namespace experimental {
// Same as mm_fp32_16_accurate, but using only wh, accumulating in double.
void mm_fp32_16_fast(TransposeType transA, TransposeType transB, double alpha,
                     dim<2> size_A, const uint16_t* wh, size_t ld_a,
                     dim<2> size_B, const float* B, size_t ld_b, double beta,
                     float* C, size_t ld_c);
}  // namespace experimental

// Thin forwarding declaration -- see the matching definition in
// fp32_16_cuda_kernels.cu.
void mm_fp32_16_fast(TransposeType transA, TransposeType transB, double alpha,
                     dim<2> size_A, const uint16_t* wh, size_t ld_a,
                     dim<2> size_B, const float* B, size_t ld_b, double beta,
                     float* C, size_t ld_c);

// Same as mv_fp32_16_fast, but the entire reduction (multiply and
// accumulate) happens in __half arithmetic, not promoted to a wider type --
// genuine fp16-only compute, only widened to float for the final alpha/beta
// scale-and-store.
void mv_fp32_16_pure_half(TransposeType trans, double alpha, dim<2> size,
                          const uint16_t* wh, size_t ld_a, const float* x,
                          size_t ld_x, double beta, float* y, size_t ld_y);

// Standalone, TRANSPOSE-specialized genuine-__half-compute GEMV kernel:
// same warp-per-output-column/split-M/atomicAdd/shared-x-staging
// structure as mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4 (the
// fp32-reconstruction tier's own TRANSPOSE kernel), decode swapped to
// fp32_16_half_only and compute swapped to __hfma2 on ushort4-loaded wh
// packed into __half2 -- fixes a real gap in mv_fp32_16_pure_half_wide_
// ilp_shared_half2 (below), whose own wh load is never actually
// vectorized despite its packed __half2 arithmetic. Only handles
// TRANSPOSE + ld_a==size[0] -- no fallback to a general dispatch,
// standalone comparison kernel. See its own comment in fp32_16_cuda_
// kernels.cu for the full derivation (including why checking temp-
// dependencies/randls2.0's own genuine-half2-compute GEMV kernels
// confirmed the same gap on that side, and where the real fix idea
// came from instead).
//
// MEASURED on real A100 hardware: a dramatic win over both wide_ilp_
// shared_half2 and plain pure_half -- 3.5x-4.7x faster at every size
// tested, reaching 0.95x-1.06x of cuBLAS Hgemm at M=1024-2048/24576-
// 65536 -- with a real dip to 0.59x-0.79x at M=3072-8192, see
// _retuned_floor below.
namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                    const uint16_t* wh, size_t ld_a, const float* x,
                                                                    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL, standalone A/B sibling of mv_fp32_16_pure_half_transpose_
// atomiconly_scaled_mblock_vec4 above -- reuses the EXACT SAME __global__
// kernel unchanged, only a more aggressive M_BLOCK-scaling target (2048
// instead of 512) with a floor (256) -- BOTH constants already validated
// for the fp32_16 reconstruction tier's own TRANSPOSE family, reused
// directly since the M_BLOCK-scaling formula is decode-independent.
// Motivated by a real, measured dip in the plain kernel's own real-
// hardware numbers at M=3072-8192 (as low as 0.59x of cuBLAS Hgemm at
// M=4096) -- see its own comment in fp32_16_cuda_kernels.cu for the
// full derivation.
//
// MEASURED on real A100 hardware (M=3072-8192, the targeted dip range):
// a real, consistent win with no regressions in this range -- +3.0% to
// +18.1%, closing roughly half the remaining gap to cuBLAS Hgemm there.
// CAVEAT: not yet confirmed outside M=3072-8192.
namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y); }  // namespace experimental


// "TOMOS FP32/FP16 (fast)" -- UNSAFE-on-arbitrary-data sibling of the
// above: decode swapped from fp32_16_half_only (full, 3-path: checks
// control/offset/degraded) to fp32_16_fast_half (unconditional bit-
// reinterpret, only correct when every wh value already has control=0,
// i.e. every source magnitude is inside fp16's normal representable
// range). Same "fast" naming/contract convention as this codebase's own
// RECONSTRUCTION-tier sibling (fp32_16_combine_fast_normal). Everything
// else identical to _retuned_floor (same M_BLOCK-scaling target=2048/
// floor=256), isolating decode cost alone. See its own comment in
// fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_fast(TransposeType trans, double alpha,
                                                                        dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                        const float* x, size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);

// "TOMOS FP32/BF16" -- bf16 counterpart of the above, built on its exact
// structure (ushort4-vectorized load, packed accumulation, M_BLOCK-
// scaling target/floor) with __half/__half2 swapped for __nv_bfloat16/
// __nv_bfloat162 and fp32_16_fast_half swapped for fp32_bf16_bf16_only.
// UNLIKE its fp16 lineage, this has NO unsafe-on-arbitrary-data caveat:
// bfloat16 shares fp32's own 8-bit exponent, so its truncation decode
// is ALWAYS valid for any input -- safe on arbitrary data despite being
// built from the "fast" (unsafe) fp16 kernel. See its own comment in
// fp32_16_cuda_kernels.cu.
void mv_fp32_bf16_pure_bf16_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);

// "test" -- hybrid TRANSPOSE kernel for the "TOMOS FP32/FP16" family:
// uses the expensive decode (fp32_16_half_only's own control-bit check)
// per element -- control=0 (wh alone already a valid half) takes the
// cheap __half/__half2 path; control=1, non-degraded (the true value
// needed more range than fp16's 5-bit exponent) decodes an approximate
// fp32 value from wh ALONE -- offset/direction (bits 10-14) give an
// extended exponent, mant_hi (bits 1-9) the same top mantissa bits a
// control=0 value would use, missing lower mantissa bits zero-padded --
// instead of flushing to +-Infinity the way fp32_16_half_only itself
// does, accumulating that element's contribution in float. Uses ONLY
// wh, like every other pure_half kernel above -- no wr, no extra memory
// traffic. Not hand-vectorized (plain scalar loop) -- first-cut "test"
// kernel, not yet a tuned candidate. See its own comment in fp32_16_
// cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock(TransposeType trans, double alpha, dim<2> size,
                                                           const uint16_t* wh, size_t ld_a, const float* x,
                                                           size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Vectorized-load sibling of the above -- built to fix its measured
// falloff past M=3072ish: the untiled kernel never vectorized its wh
// load at all, unlike every other "TOMOS FP32/FP16" kernel (all
// ushort4). Since M_BLOCK grows with M (same shared target/floor
// formula), the untiled kernel's own scalar-load cost grows right
// alongside it; this sibling keeps its identical per-element branch
// (cheap __half2 path vs wh-only extended-exponent float path)
// unchanged, only vectorizing the LOAD (ushort4, same scalar-prefix/
// vectorized-middle/scalar-suffix structure every vec4 kernel here
// uses). See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4(TransposeType trans, double alpha, dim<2> size,
                                                                const uint16_t* wh, size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// Paired-packing sibling of the above -- fixes a real arithmetic-
// throughput gap vs "(atomiconly, retuned_floor)": the plain vec4
// kernel processes each of the 4 loaded elements individually, each
// __hfma2 wasting one lane on a zero pad. This kernel packs adjacent
// same-control (both cheap) pairs into one real __hfma2 (true half2
// SIMD, matching _retuned_floor's own approach exactly), falling back
// to the existing per-element helper only for mixed pairs. Same
// alignment contract as every other vec4 kernel here. See its own
// comment in fp32_16_cuda_kernels.cu.
void mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, size_t ld_x, double beta,
                                                                       float* y, size_t ld_y);

// __half-I/O counterpart of the above -- IDENTICAL compute (same
// fp32_16_half_only decode, same ushort4-vectorized wh load, same
// __hfma2 arithmetic, same M_BLOCK-scaling target=2048/floor=256), only
// x and y are __half* instead of float* -- matching cuBLAS's own
// Hgemm(n=1) semantics exactly (A, x, AND y all __half), unlike every
// other mv_fp32_16_pure_half* kernel in this file. Internal accumulation
// still happens in float (only the final atomicAdd narrows to __half) --
// this isolates the I/O type change from any accumulation-precision
// change. Only handles TRANSPOSE + ld_a==size[0], standalone comparison
// kernel. See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_retuned_floor(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const __half* x, size_t ld_x,
    double beta, __half* y, size_t ld_y); }  // namespace experimental


// Standalone TRANSPOSE pure_half sibling exploring a different grid/
// reduction strategy: "warp owner, no atomics" -- ported from randls2.0's
// own gemv_fp16_cm_T_warp_owner (dense_kernels.cu:7533, the pure single-
// __half*-array overload). One warp owns output column k for its ENTIRE
// lifetime (loops the whole M range internally, tiled, x staged into
// shared memory once per tile), single warp-shuffle reduction, single
// write per column -- no atomics, no M_BLOCK-scaling, unlike every other
// pure_half_transpose kernel above. Reuses genuinely-vectorized pair
// loads of wh (adjacent rows in the same column) decoded via fp32_16_
// half_only.
//
// PRIOR ART: this exact structural design was already tried for the
// fp32_16 RECONSTRUCTION tier (mv_fp32_16_transpose_warpdot_tiled_fast_
// normal) and regressed -18.7% to -25.7% there -- but that kernel could
// not vectorize its wh/wr load at all and paid a much more expensive
// decode. See its own comment in fp32_16_cuda_kernels.cu for why this
// pure_half version is a materially different case, not a rerun.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                                size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                                size_t ld_y); }  // namespace experimental


// Returns the scratch-buffer size (in doubles) mv_fp32_16_pure_half_
// transpose_warp_owner_split needs for the given shape -- call once per
// shape, allocate, reuse across calls (same workspace contract as
// mv_fp32_16_accurate_workspace_size).
namespace experimental { size_t mv_fp32_16_pure_half_transpose_warp_owner_split_workspace_size(dim<2> size); }  // namespace experimental


// Split-then-reduce counterpart of mv_fp32_16_pure_half_transpose_warp_
// owner above -- fixes that kernel's real, measured cause of a worst-of-
// everything result: its grid never scales with M (fixed at (1,
// ceil(k/8))), so for any shape where k isn't huge, too few blocks are
// launched to fill the GPU. Reuses the SAME M_BLOCK-scaling target/floor
// (2048/256) already validated for this pure_half tier's own retuned_
// floor kernel, applied here to split M across blocks -- each block
// writes its own partial sum to a unique scratch slot (no atomics,
// still true to the warp-owner design's own premise), combined by a
// separate reduce pass (mv_fp32_16_wide_reduce_kernel). Caller-provided
// workspace (see _workspace_size above), no internal cudaMalloc/
// cudaFree in the timed path. Only handles TRANSPOSE + ld_a==size[0].
// See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner_split(TransposeType trans, double alpha, dim<2> size,
                                                      const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
                                                      double beta, float* y, size_t ld_y, double* workspace); }  // namespace experimental


// float-scratch-buffer counterpart of mv_fp32_16_pure_half_transpose_
// warp_owner_split_workspace_size/mv_fp32_16_pure_half_transpose_warp_
// owner_split above -- identical compute, only the partial buffer is
// float instead of double, testing whether that buffer's own write+
// readback bandwidth explains _split's measured trailing-off at the
// largest M tested (pure_half's own loose 5.0-tolerance convention means
// there's no accuracy reason to prefer double here). See their own
// comments in fp32_16_cuda_kernels.cu.
namespace experimental { size_t mv_fp32_16_pure_half_transpose_warp_owner_split_f32_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_warp_owner_split_f32(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, size_t ld_a, const float* x,
                                                          size_t ld_x, double beta, float* y, size_t ld_y,
                                                          float* workspace); }  // namespace experimental


// Isolates the real remaining cause of the warp_owner_split family's own
// trailing-off vs _retuned_floor at large M: NOT the scratch buffer
// (already ruled out empirically), but warp_owner_split's own weaker
// 2-wide wh load/float-fmaf-accumulate compute body (ported from
// randls2.0's warp-owner design) vs _retuned_floor's proven 4-wide
// ushort4/__hfma2 compute. This kernel reuses mv_fp32_16_pure_half_
// transpose_atomiconly_scaled_mblock_vec4_kernel's own compute body
// VERBATIM, changing ONLY the final combine step (partial-buffer write
// instead of atomicAdd) -- isolates atomics-vs-split+reduce as the ONE
// remaining variable. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size(dim<2> size); }  // namespace experimental

void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split(TransposeType trans, double alpha,
                                                                         dim<2> size, const uint16_t* wh,
                                                                         size_t ld_a, const float* x, size_t ld_x,
                                                                         double beta, float* y, size_t ld_y,
                                                                         float* workspace);

// Fast-decode sibling of the wrapper above (uses fp32_16_half_only_fast
// instead of fp32_16_half_only) -- added as a new, separate function,
// original untouched. Shares the same workspace sizing (mv_fp32_16_
// pure_half_transpose_atomiconly_scaled_mblock_vec4_split_workspace_size
// above -- decode-independent), so no new sizing function is needed. See
// its own comment in fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);

// Fast2-decode sibling of the safe wrapper above (uses fp32_16_half_
// only_fast2 instead of fp32_16_half_only) -- BIT-EXACT with the safe
// decode, unlike the "fast"/"(fast)" siblings elsewhere in this file
// (see fp32_16_half_only_fast2's own comment in fp32_16_pack.hpp).
// Shares the same workspace sizing (decode-independent), so no new
// sizing function is needed. See its own comment in fp32_16_cuda_
// kernels.cu.
void mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);

// Register-blocking width sweep of the fast2/vec4_split wrapper above --
// N=2 and N=8 siblings (vec4_split itself is N=4). See their own comments
// in fp32_16_cuda_kernels.cu. Same workspace sizing (decode- and width-
// independent).
void mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec2_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);
void mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec8_split(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);

// Packed-storage (FP32_16 wh-only, branchless no-overflow decode) NO_TRANSPOSE
// GEMV with mv_fp16_native_mixed's ACCUMULATION structure (256-column tiles
// flushed into a double, same three-tier dispatch) -- see its own comment in
// fp32_16_cuda_kernels.cu. NO_TRANSPOSE only.
void mv_fp32_16_pure_half_nativeaccum(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                      size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                      size_t ld_y);

// Periodic-flush siblings of the narrow_vec kernel LSQR's implicit
// NO_TRANSPOSE matvec runs at large M: every 64 / 256 columns the per-row
// __half accumulators are added into float and reset, bounding every fp16
// chain (see the kernel's own comment in fp32_16_cuda_kernels.cu).
// narrow_vec_flush*: the kernel alone at every M (GEMV error benchmark).
// no_transpose_flush*: drop-in for mv_fp32_16_pure_half_no_transpose_fast2
// (same signature/routing; only the large-M branch changes).
void mv_fp32_16_pure_half_narrow_vec_flush64(TransposeType trans, double alpha, size_t m, size_t n,
                                             const uint16_t* wh, size_t ld_a, const float* x, double beta,
                                             float* y);
void mv_fp32_16_pure_half_narrow_vec_flush256(TransposeType trans, double alpha, size_t m, size_t n,
                                              const uint16_t* wh, size_t ld_a, const float* x, double beta,
                                              float* y);
void mv_fp32_16_pure_half_no_transpose_flush64(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                               const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
void mv_fp32_16_pure_half_no_transpose_flush256(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                                const float* x, size_t ld_x, double beta, float* y,
                                                size_t ld_y);

// flush256 with a SWAR (two wh per 32-bit register) decode -- bit-exact with
// the scalar decode (exhaustively checked on all 2^32 input pairs), aimed at
// the ALU-bound decode cost. Same two entry points as the flush siblings above.
void mv_fp32_16_pure_half_narrow_vec_flush256_swar(TransposeType trans, double alpha, size_t m, size_t n,
                                                   const uint16_t* wh, size_t ld_a, const float* x, double beta,
                                                   float* y);
void mv_fp32_16_pure_half_no_transpose_flush256_swar(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                                     const float* x, size_t ld_x, double beta, float* y,
                                                     size_t ld_y);
// LOSSY fast-decode (underflow band -> 0) sibling of the pair above; benchmark/plot only.
void mv_fp32_16_pure_half_narrow_vec_flush256_fast_swar(TransposeType trans, double alpha, size_t m, size_t n,
                                                        const uint16_t* wh, size_t ld_a, const float* x,
                                                        double beta, float* y);
void mv_fp32_16_pure_half_no_transpose_flush256_fast_swar(double alpha, dim<2> size, const uint16_t* wh,
                                                          size_t ld_a, const float* x, size_t ld_x, double beta,
                                                          float* y, size_t ld_y);

// cp.async sibling of the fast2 wrapper above: double-buffers each
// thread's own next-iteration wh ushort4 read (the aligned/vectorized
// path only) while decoding the current one -- x's own shared-memory
// staging is unchanged (already efficient, one-time per block). Same
// workspace sizing. See its own comment in fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_vec4_split_cpasync(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace);

// "TOMOS FP16 (fast)" -- UNSAFE-on-arbitrary-data sibling of the above:
// decode swapped from fp32_16_half_only (full, 3-path) to fp32_16_
// fast_half (unconditional bit-reinterpret, only correct when every wh
// value already has control=0) -- the same swap "TOMOS FP32/FP16
// (fast)" applied to _retuned_floor's own body, applied here to the
// split-combine kernel instead, isolating the decode win from the
// combine-strategy variable. Same M_BLOCK-scaling/workspace formula as
// _split (decode-independent, reuses that function's own _workspace_
// size). See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_fast(TransposeType trans, double alpha,
                                                                              dim<2> size, const uint16_t* wh,
                                                                              size_t ld_a, const float* x,
                                                                              size_t ld_x, double beta, float* y,
                                                                              size_t ld_y, float* workspace); }  // namespace experimental


// Retuned M_BLOCK-scaling target for the M=3072-8192 dip (worst gap to
// cuBLAS Hgemm once _split matched _retuned_floor's own compute) --
// grid-math search found target=2176 is a near-single-variable change
// vs the shared target=2048 (only M=4096/8192 actually move, everything
// else identical). MEASURED on real A100 hardware: negative result --
// no improvement at M=4096/6144, a real -9.2% regression at M=8192 --
// the wave-quantization model this was based on ignored the redundant
// x-restaging/scratch-buffer cost that grows with gridM. Not promoted;
// see the target constant's own comment in fp32_16_cuda_kernels.cu for
// the full diagnosis.
namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace); }  // namespace experimental


// Tests the OPPOSITE hypothesis of the retune above: _split_retarget's
// own confirmed -9.2% regression at M=8192 (where it INCREASED gridM)
// is diagnosed as the split design's redundant restaging/scratch-buffer
// cost growing with gridM, outweighing the occupancy gain -- so this
// sibling REDUCES gridM instead (target=1280 vs the shared 2048) at the
// M=3072-24576 dip (widened from the original 3072-8192 framing per the
// user's own observation), leaving M=8192 and everything above it
// completely untouched. See the target constant's own comment in
// fp32_16_cuda_kernels.cu for the full grid-math derivation.
namespace experimental { size_t mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo_workspace_size(dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y, float* workspace); }  // namespace experimental


namespace experimental {
// Same as mm_fp32_16_fast, but the entire reduction happens in __half
// arithmetic (see mv_fp32_16_pure_half).
void mm_fp32_16_pure_half(TransposeType transA, TransposeType transB,
                          double alpha, dim<2> size_A, const uint16_t* wh,
                          size_t ld_a, dim<2> size_B, const float* B,
                          size_t ld_b, double beta, float* C, size_t ld_c);

// Shared-memory-tiled version of mm_fp32_16_pure_half_kernel, same 128x128/
// 8x8 register-blocked structure as mm_fp32_16_tiled_kernel -- the only
// change is that As/Bs are cached as __half (not float) and the register-
// blocked accumulation runs in __half arithmetic end to end (native
// hardware fp16 FMA on sm_53+), matching mm_fp32_16_pure_half's "entire
// reduction in __half" semantics rather than mm_fp32_16_tiled's fp32 one.
// NO_TRANSPOSE x NO_TRANSPOSE only -- falls back to mm_fp32_16_pure_half
// for any transposed operand. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_pure_half_tiled_kernel.
void mm_fp32_16_pure_half_tiled(TransposeType transA, TransposeType transB,
                                double alpha, dim<2> size_A, const uint16_t* wh,
                                size_t ld_a, dim<2> size_B, const float* B,
                                size_t ld_b, double beta, float* C, size_t ld_c);

// A fundamentally different mechanism from mm_fp32_16_pure_half/_tiled: uses
// Tensor Core mma.sync instructions (via nvcuda::wmma, 16x16x16 half/half/
// float tiles) instead of scalar __half FMA, aiming to match cuBLAS's
// Hgemm -- which itself is tensor-core-backed on sm_70+ -- rather than
// just tuning the scalar kernel's memory/tiling the way mm_fp32_16_pure_
// half_tiled does. Unlike the GEMV wmma experiment in this file (mv_fp32_16_
// pure_half_wide_wmma, which wastes 15/16 of its tensor-core throughput
// replicating a width-1 x across all 16 output columns and measured only
// 0.14x cuBLAS because of it), a real GEMM has genuine N-wide reuse, so
// every output column does real, non-replicated work here.
//
// Tile shape: BM=128, BN=128, BK=64 per block, computed by 16 warps (4x4)
// each owning a 32x32 output region (2x2 = 4 accumulator fragments/warp,
// accumulated in fp32 the same way cuBLAS's own tensor-core kernels do --
// not __half -- since mma.sync's native mode is half/half/float and
// forcing a plain __half accumulator would mean an extra down/up-cast
// per k-step for no accuracy benefit). A's wh is decoded (fp32_16_half_
// only) and B is cast from float to __half once per tile, when staged into
// shared memory, same "decode on load" principle as every other tiled
// kernel in this file. __launch_bounds__ trades a modest amount of
// register spilling for 2 resident blocks/SM instead of 1 -- see
// fp32_16_cuda_kernels.cu for the register-pressure/occupancy history
// (a 64x64/BK=32/4-warp first attempt and a naive 128x128 scale-up both
// measured well below mm_fp32_16_pure_half_tiled's scalar kernel; this
// tile/warp/launch_bounds combination is what got it ahead, at ~7.4-7.9
// TFlop/s on an A30 vs cuBLAS's Hgemm ~135-140 TFlop/s there -- still a
// large gap, since matching cuBLAS's own multi-stage cp.async pipelining
// and swizzled shared-memory layouts is out of scope here). NO_TRANSPOSE
// x NO_TRANSPOSE only -- falls back to mm_fp32_16_pure_half_tiled for any
// transposed operand. See fp32_16_cuda_kernels.cu for mm_fp32_16_pure_
// half_wmma_kernel.
void mm_fp32_16_pure_half_wmma(TransposeType transA, TransposeType transB,
                               double alpha, dim<2> size_A, const uint16_t* wh,
                               size_t ld_a, dim<2> size_B, const float* B,
                               size_t ld_b, double beta, float* C, size_t ld_c);

// cp.async double-buffered version of mm_fp32_16_pure_half_wmma, ported
// from gemm_fp16_wmma_cpasync (mm_cuda_kernels.cuh/cu) after THAT kernel's
// cp.async pipelining measured a real ~8-10% win by hiding the global-load
// latency mm_fp32_16_pure_half_wmma's every outer-BK tile otherwise fully
// exposes behind a __syncthreads(). Unlike mm_fp32_16_tiled_cpasync_dualbuf
// (the accurate/fast tier's cp.async kernel, which needs a separate raw-
// bytes staging buffer because it decodes wh+wr TOGETHER into a wider
// float), a pure_half decode is wh alone -> a same-sized __half, so the
// raw bytes land via cp.async directly in As's own memory and get decoded
// in place once the transfer completes -- no second buffer needed. Same
// BK=32 (halved from 64, to fit double-buffered As/Bs in the 48KB static
// shared-memory budget) and B-stays-synchronous reasoning as gemm_fp16_
// wmma_cpasync. Requires ld_a and m to both be even; falls back to
// mm_fp32_16_pure_half_wmma otherwise. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_pure_half_wmma_cpasync_kernel.
void mm_fp32_16_pure_half_wmma_cpasync(TransposeType transA, TransposeType transB,
                                       double alpha, dim<2> size_A, const uint16_t* wh,
                                       size_t ld_a, dim<2> size_B, const float* B,
                                       size_t ld_b, double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_pure_half_wmma_cpasync, but wh's cp.async transfer
// widens to 8 elements (16 bytes) and B's synchronous read widens to a
// float4 (4 elements, 16 bytes) -- ported from gemm_fp16_wmma_cpasync_vec
// (mm_cuda_kernels.cuh/cu) after THAT kernel's widening measured a ~65%
// win there (B's unvectorized scalar load, not decode or the tensor cores
// themselves, was the dominant cost). Requires ld_a and m to both be
// multiples of 8, and ld_b and k to both be multiples of 4; falls back to
// mm_fp32_16_pure_half_wmma_cpasync otherwise. See fp32_16_cuda_kernels.cu
// for mm_fp32_16_pure_half_wmma_cpasync_vec_kernel.
void mm_fp32_16_pure_half_wmma_cpasync_vec(TransposeType transA, TransposeType transB,
                                           double alpha, dim<2> size_A, const uint16_t* wh,
                                           size_t ld_a, dim<2> size_B, const float* B,
                                           size_t ld_b, double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_pure_half_wmma_cpasync_vec (same wh/B handling --
// unlike mm_fp32_16_pure_half_wmma_cpasync's failed-in-the-fp16-analog
// fully-async-B idea, this does NOT change B), but BK doubles from 32 to
// 64 -- ported from gemm_fp16_wmma_cpasync_vec_bigbk (mm_cuda_kernels.cuh)
// after THAT kernel measured a consistent ~3.5-6% win from halving how
// often the outer loop pays its per-tile issue/barrier overhead. Needs
// 80KB of shared memory (double-buffered As+Bs at BK=64, plus the
// unchanged warp_c stage), so it opts into dynamic shared memory via
// cudaFuncSetAttribute. Requires ld_a and m to both be multiples of 8,
// and ld_b and k to both be multiples of 4; falls back to mm_fp32_16_
// pure_half_wmma_cpasync_vec otherwise. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_kernel.
void mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk(TransposeType transA, TransposeType transB,
                                                 double alpha, dim<2> size_A, const uint16_t* wh,
                                                 size_t ld_a, dim<2> size_B, const float* B,
                                                 size_t ld_b, double beta, float* C, size_t ld_c);

// Same as mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk, but As/Bs's shared-
// memory storage stride is padded from 128 halves (256 bytes -- exactly 2
// full sweeps of the 32 shared-memory banks) to 136 -- ported from gemm_
// fp16_wmma_cpasync_vec_bigbk_pad (mm_cuda_kernels.cuh) after THAT
// kernel's padding nearly DOUBLED throughput there. No logical tile size
// or compute changes -- only where each element physically lives in
// shared memory. Requires ld_a and m to both be multiples of 8, and ld_b
// and k to both be multiples of 4; falls back to mm_fp32_16_pure_half_
// wmma_cpasync_vec_bigbk otherwise. See fp32_16_cuda_kernels.cu for
// mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad_kernel.
void mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad(TransposeType transA, TransposeType transB,
                                                     double alpha, dim<2> size_A, const uint16_t* wh,
                                                     size_t ld_a, dim<2> size_B, const float* B,
                                                     size_t ld_b, double beta, float* C, size_t ld_c);

// Same padded layout as mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad,
// but BK doubles again, 64 -> 128 -- ported from gemm_fp16_wmma_cpasync_
// vec_hugebk (mm_cuda_kernels.cuh) after THAT kernel measured a further
// ~10-15% win despite dropping to 1 resident block/SM (the wider tile's
// arithmetic-intensity gain won that trade). Needs ~152KB of shared
// memory; if cudaFuncSetAttribute's opt-in fails on a GPU with a smaller
// ceiling, falls back to mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad.
// Requires ld_a and m to both be multiples of 8, and ld_b and k to both be
// multiples of 4. See fp32_16_cuda_kernels.cu for mm_fp32_16_pure_half_
// wmma_cpasync_vec_hugebk_kernel.
void mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk(TransposeType transA, TransposeType transB,
                                                  double alpha, dim<2> size_A, const uint16_t* wh,
                                                  size_t ld_a, dim<2> size_B, const float* B,
                                                  size_t ld_b, double beta, float* C, size_t ld_c);
}  // namespace experimental

// Thin forwarding declaration -- see the matching definition in
// fp32_16_cuda_kernels.cu.
void mm_fp32_16_pure_half(TransposeType transA, TransposeType transB,
                          double alpha, dim<2> size_A, const uint16_t* wh,
                          size_t ld_a, dim<2> size_B, const float* B,
                          size_t ld_b, double beta, float* C, size_t ld_c);

// ---- EXPERIMENTAL: register-blocked wide-matrix path, for comparison ----
//
// Identical to mv_fp32_16_accurate/fast/pure_half except in the wide-matrix
// (split-K) dispatch branch, which additionally register-blocks
// kMvRowsPerThread output rows per thread (see fp32_16_cuda_kernels.cu for
// the constant and the full design rationale, including an earlier,
// worse-performing attempt this one is specifically structured to avoid
// repeating). The narrow/tiled branches are unchanged. Kept as separate
// functions (not merged into the originals) so they can be benchmarked
// side by side against the already-validated ~70%-of-cuBLAS baseline.
namespace experimental { void mv_fp32_16_accurate_wide_regblock(TransposeType trans, double alpha,
                                       dim<2> size, const uint16_t* wh,
                                       const uint16_t* wr, size_t ld_a,
                                       const float* x, size_t ld_x,
                                       double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_fast_wide_regblock(TransposeType trans, double alpha,
                                   dim<2> size, const uint16_t* wh,
                                   size_t ld_a, const float* x, size_t ld_x,
                                   double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_regblock(TransposeType trans, double alpha,
                                        dim<2> size, const uint16_t* wh,
                                        size_t ld_a, const float* x,
                                        size_t ld_x, double beta, float* y,
                                        size_t ld_y); }  // namespace experimental


// DIAGNOSTIC (not a performance variant): identical to mv_fp32_16_pure_
// half's wide-matrix branch, but reports the main split-K kernel's and the
// reduce kernel's elapsed time separately via the two out-parameters
// (milliseconds), to test whether the reduce step accounts for a
// significant share of the wide-matrix path's ~70%-of-cuBLAS gap,
// independent of the (so far unsuccessful) register-blocking attempts
// above -- see fp32_16_cuda_kernels.cu for the full rationale. For
// rows_op >= kMvWideRowThreshold this just calls mv_fp32_16_pure_half and
// reports zeroed timings, so it's safe to call unconditionally.
namespace experimental { void mv_fp32_16_pure_half_wide_diag(TransposeType trans, double alpha,
                                    dim<2> size, const uint16_t* wh,
                                    size_t ld_a, const float* x, size_t ld_x,
                                    double beta, float* y, size_t ld_y,
                                    float* main_kernel_ms,
                                    float* reduce_kernel_ms); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path,
// informed by ncu profiling that found the baseline's dominant stall
// reason there is L1TEX load latency (waiting on wh[idx]'s result with no
// independent work to overlap it), not bandwidth or occupancy. Unlike
// mv_fp32_16_pure_half_wide_regblock, this keeps the exact same grid/block
// configuration as the baseline (same choose_col_splits(rows_op, cols_op)
// call) -- only the main kernel's column loop is unrolled 4-wide with 4
// independent accumulators, to have more loads in flight per thread. See
// fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp(TransposeType trans, double alpha,
                                   dim<2> size, const uint16_t* wh,
                                   size_t ld_a, const float* x, size_t ld_x,
                                   double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// IDENTICAL kernel body and grid/block configuration to the baseline --
// the only difference is a __launch_bounds__(kMvTileSize, 8) attribute on
// the launched kernel, asking the compiler to keep register usage low
// enough for 8 resident blocks/SM. Directly tests ncu's own finding that
// the baseline's occupancy there is register-limited ("Block Limit
// Registers: 6", the tightest of its occupancy constraints) -- see
// fp32_16_cuda_kernels.cu for the full rationale, including why mv_fp32_16_
// pure_half_wide_ilp's opposite approach (more registers for more ILP)
// likely made this exact constraint worse instead of better.
namespace experimental { void mv_fp32_16_pure_half_wide_launchbounds(TransposeType trans, double alpha,
                                            dim<2> size, const uint16_t* wh,
                                            size_t ld_a, const float* x,
                                            size_t ld_x, double beta,
                                            float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path,
// structurally different from the other _wide_* experiments above (which
// all still used column-splitting across the grid plus a separate reduce
// kernel/global-memory partial buffer): assigns one full warp (32 lanes)
// to each group of kMvRowsPerThread rows, using the same vectorized
// ushort4 load per lane and a __shfl_down_sync warp reduction to combine
// results, with no partial buffer or second kernel launch at all. See
// fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_warpgroup(TransposeType trans, double alpha,
                                         dim<2> size, const uint16_t* wh,
                                         size_t ld_a, const float* x,
                                         size_t ld_x, double beta, float* y,
                                         size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// isolates one variable that mv_fp32_16_pure_half_wide_ilp changed
// alongside its 4-way column unrolling -- x's shared-memory tiling.
// wide_ilp dropped the baseline's shared-memory x tile for direct __ldg
// reads while ALSO adding ILP, and measured worse than the baseline
// despite the added ILP (a hint the tiling removal cost more than the
// unrolling gained). This variant keeps the baseline's shared-memory
// x_tile but adds the same 4-way column unrolling, to test that hint in
// isolation. See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared(TransposeType trans, double alpha,
                                          dim<2> size, const uint16_t* wh,
                                          size_t ld_a, const float* x,
                                          size_t ld_x, double beta, float* y,
                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// IDENTICAL to mv_fp32_16_pure_half_wide_ilp_shared (same grid/col_splits,
// same 4-way unroll, same shared-memory x_tile) except wh/x are marked
// __restrict__ and every wh load is routed through __ldg() explicitly,
// testing whether the read-only data cache path closes more of the
// remaining gap to cuBLAS on top of the ilp_shared win (0.71 -> 0.83 of
// cuBLAS at 32768x32768). See fp32_16_cuda_kernels.cu for the full
// rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_ldg(TransposeType trans,
                                              double alpha, dim<2> size,
                                              const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x,
                                              double beta, float* y,
                                              size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path: keeps
// mv_fp32_16_pure_half_wide_ilp_shared's proven 4-way accumulator width
// (unlike ilp8_shared, which widened to 8 and regressed) but software-
// pipelines the column loop -- each iteration issues the NEXT group of 4
// wh loads before consuming the CURRENT group's already-loaded values,
// deepening the outstanding-load window from 4 to up to 8 without adding
// any new accumulator dependency chains. See fp32_16_cuda_kernels.cu for
// the full rationale, including why this is a different lever from the
// (regressive) __ldg-based mv_fp32_16_pure_half_wide_ilp_shared_ldg above.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_prefetch(TransposeType trans,
                                                   double alpha, dim<2> size,
                                                   const uint16_t* wh,
                                                   size_t ld_a, const float* x,
                                                   size_t ld_x, double beta,
                                                   float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path: a
// structural lever distinct from every compute-scheduling variant above
// (regblock, ilp, launchbounds, warpgroup, ilp_shared_ldg, ilp_shared_
// prefetch), all of which tuned unroll width, occupancy, caching, or load
// ordering around the same 256-wide shared-memory tile and regressed or
// landed flat past mv_fp32_16_pure_half_wide_ilp_shared's 0.83. Keeps the
// same proven 4-way unroll but quadruples the shared-memory tile width
// (1024 instead of 256), cutting __syncthreads() round-trips per column by
// 4x, while keeping blockDim.x/row_blocks/choose_col_splits identical to
// the baseline dispatch so occupancy is untouched. See
// fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile(TransposeType trans,
                                                  double alpha, dim<2> size,
                                                  const uint16_t* wh,
                                                  size_t ld_a, const float* x,
                                                  size_t ld_x, double beta,
                                                  float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path: a
// fundamentally different kernel architecture (Tensor Core mma.sync via
// nvcuda::wmma, 16x16x16 half/half/float tiles with x replicated across
// all 16 output columns) rather than a tuning variant of the scalar
// __half-FMA design every _wide_* variant above uses. Only reaches the
// tensor-core path for TransposeType::NO_TRANSPOSE; TRANSPOSE falls back
// to mv_fp32_16_pure_half_wide_ilp_shared. See fp32_16_cuda_kernels.cu for
// the full rationale, including the caveat that the actual measured
// bottleneck (wh's load latency) is in the staging step this kernel still
// performs via ordinary scalar loads, not in the arithmetic tensor cores
// accelerate.
namespace experimental { void mv_fp32_16_pure_half_wide_wmma(TransposeType trans, double alpha,
                                    dim<2> size, const uint16_t* wh,
                                    size_t ld_a, const float* x, size_t ld_x,
                                    double beta, float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// isolates whether mv_fp32_16_pure_half_wide_wmma's 0.14-of-cuBLAS result
// was caused specifically by its high __syncthreads() frequency (twice per
// 16-wide K-step) rather than tensor cores generally, by batching 8
// consecutive K-tiles into one larger shared-memory staging pass per sync
// cycle -- otherwise identical (same 16x16x16 tile shape, x-replication,
// fp32 accumulator). See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_wmma_batched(TransposeType trans, double alpha,
                                            dim<2> size, const uint16_t* wh,
                                            size_t ld_a, const float* x,
                                            size_t ld_x, double beta, float* y,
                                            size_t ld_y); }  // namespace experimental


// Standalone, TRANSPOSE-specific Tensor Core (WMMA) GEMV kernel -- the
// TRANSPOSE counterpart mv_fp32_16_pure_half_wide_wmma above explicitly
// does not have (falls back to mv_fp32_16_pure_half_wide_ilp_shared for
// TRANSPOSE). Same 16x16x16 tile/x-replication trick, dimensions swapped
// (one warp computes 16 output COLUMNS at once instead of 16 output
// rows, reduction walked in 16-ROW steps). Only handles TRANSPOSE +
// ld_a==size[0] -- no fallback to a general dispatch, standalone
// comparison kernel.
//
// STRONG PRIOR EVIDENCE OF LIKELY FAILURE: mv_fp32_16_pure_half_wide_
// wmma_kernel's own real-hardware result is 0.14x of cuBLAS -- WORSE
// than every one of the eleven scalar _wide_* attempts, including the
// worst one (regblock's 0.26x) -- because the actual bottleneck (wh's
// staging-load latency) sits BEFORE any tensor-core instruction even
// starts, and this kernel does the identical scalar-load staging. This
// TRANSPOSE kernel shares that same staging approach and the same
// twice-per-16-step __syncthreads() frequency the (never independently
// confirmed) _wide_wmma_batched sibling was built to test.
//
// MEASURED on real A100 hardware (M=K=4096): CONFIRMED, SEVERE NEGATIVE
// RESULT -- 0.043x of cuBLAS Hgemm, WORSE than even the NO_TRANSPOSE
// wmma kernel's own confirmed 0.14x, and 15x slower than the already-
// promoted-candidate _retuned_floor kernel at the same shape. Tensor
// Cores do not appear to help this memory-bound streaming GEMV shape --
// see its own comment in fp32_16_cuda_kernels.cu for the full derivation.
namespace experimental { void mv_fp32_16_pure_half_transpose_wmma(TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
                                          size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// isolates whether fp32_16_half_only's control-bit branch costs anything
// beyond a raw half read, by using fp32_16_fast_half (the pre-existing
// unconditional reinterpret_cast, no branch) instead -- otherwise identical
// to mv_fp32_16_pure_half_wide_ilp_shared. CORRECTNESS CAVEAT (CONFIRMED
// via run_correctness_case, not merely theoretical): fp32_16_fast_half is
// only bit-exact for control=0 elements, and this benchmark's synthetic
// test data DOES exercise the unsafe control=1 path -- gen_fp32_16 draws
// linearly uniform floats over [-1,1], and a small but real fraction land
// below the ~6.1e-5-magnitude control=1 boundary, which across a
// 32768x32768 matrix is tens of thousands of elements. This kernel's
// measured speedup (0.83 -> 0.88) reflects skipping work the data actually
// needs, not doing the same work faster -- see fp32_16_cuda_kernels.cu for
// the full mechanism. Kept only as a diagnostic negative result, not a
// valid performance comparison point.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_fasthalf(TransposeType trans,
                                                   double alpha, dim<2> size,
                                                   const uint16_t* wh,
                                                   size_t ld_a, const float* x,
                                                   size_t ld_x, double beta,
                                                   float* y, size_t ld_y); }  // namespace experimental


// Returns the number of doubles mv_fp32_16_pure_half_wide_ilp_shared_
// fasthalf_workspace's wide (split-K) path needs from a caller-provided
// workspace buffer for the given shape (0 if that shape doesn't dispatch
// to the wide path at all). See mv_fp32_16_pure_half_wide_ilp_shared_
// fasthalf_workspace's own comment.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered sibling of mv_fp32_16_pure_half_wide_ilp_shared_
// fasthalf above -- same dispatch/kernels/decode, only the wide path uses
// a caller-provided workspace buffer instead of a fresh cudaMalloc/
// cudaFree every call. Built after real A100 NO_TRANSPOSE measurements
// showed this whole wide_ilp_shared family trailing cuBLAS far more at
// small/medium M than large M -- the signature of per-call allocator
// overhead, not a compute/decode problem. See its own comment in
// fp32_16_cuda_kernels.cu for the full derivation (mirrors mv_fp32_16_
// accurate_workspace's own already-established fix for the same issue on
// the fp32_16 reconstruction tier).
//
// PROMOTED for square NO_TRANSPOSE shapes (CONFIRMED on real A100
// hardware); NOT promoted for tall-and-thin, large M (>=131072) collapses
// into mv_fp32_16_pure_half_narrow_vec_kernel via a diagnosed dispatch-
// threshold gap -- see kMvWideRowThreshold's own comment and this
// function's own comment in fp32_16_cuda_kernels.cu for the measurements.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace(TransposeType trans,
                                                              double alpha, dim<2> size,
                                                              const uint16_t* wh,
                                                              size_t ld_a, const float* x,
                                                              size_t ld_x, double beta,
                                                              float* y, size_t ld_y,
                                                              double* workspace); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// combines ilp_shared_fasthalf's decode-branch removal with ilp_shared_
// bigtile's quadrupled shared-memory tile (0.85 measured -- the two
// effects do not stack additively, landing between fasthalf alone's 0.88
// and bigtile alone's 0.80). Same CONFIRMED correctness failure as
// ilp_shared_fasthalf on this benchmark's data (see its comment and
// fp32_16_cuda_kernels.cu for the full mechanism) -- kept only as a
// diagnostic negative result, not a valid performance comparison point.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_fasthalf(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
    size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// tests whether fp32_16_half_only's branch instruction itself (rather than
// the control=1 saturation logic it guards, which ilp_shared_fasthalf
// wrongly skipped) was the source of that kernel's measured speedup, by
// using fp32_16_half_only_branchless (a bit-exact, branchless
// reformulation of the same logic -- see fp32_16_cuda_kernels.cu) instead.
// Unlike ilp_shared_fasthalf, this is expected to be fully correct on any
// input, not merely safe for one benchmark's data.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_branchless(TransposeType trans,
                                                     double alpha, dim<2> size,
                                                     const uint16_t* wh,
                                                     size_t ld_a,
                                                     const float* x,
                                                     size_t ld_x, double beta,
                                                     float* y, size_t ld_y); }  // namespace experimental


// Returns the number of doubles mv_fp32_16_pure_half_wide_ilp_shared_
// branchless_workspace's wide (split-K) path needs -- 0 for shapes that
// don't dispatch to the wide path. See mv_fp32_16_pure_half_wide_ilp_
// shared_branchless_workspace's own comment.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered sibling of mv_fp32_16_pure_half_wide_ilp_shared_
// branchless above -- same caller-owned-workspace fix as mv_fp32_16_pure_
// half_wide_ilp_shared_fasthalf_workspace (see its own comment in fp32_16_
// cuda_kernels.cu for the full rationale), built so this kernel's own
// "does removing the branch alone recover fasthalf's gain" question can
// be measured without the SAME per-call allocator overhead confounding
// the answer. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_branchless_workspace(TransposeType trans,
                                                                double alpha, dim<2> size,
                                                                const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta,
                                                                float* y, size_t ld_y,
                                                                double* workspace); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// uses Ampere's cp.async (via __pipeline_memcpy_async) to stage wh into
// shared memory directly from global memory, bypassing the register file
// -- a fundamentally different mechanism from every prior _wide_* variant,
// which all had every wh value pass through a register. Only implements
// the fast path for TransposeType::NO_TRANSPOSE (matching every
// benchmarked shape); TRANSPOSE falls back to mv_fp32_16_pure_half_wide_
// ilp_shared. ALIGNMENT CAVEAT: requires ld_a to be even (true for every
// power-of-two shape benchmarked in this investigation, not a general
// guarantee for arbitrary odd ld_a) -- see fp32_16_cuda_kernels.cu for the
// full rationale and mechanism.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_cpasync(TransposeType trans,
                                                  double alpha, dim<2> size,
                                                  const uint16_t* wh,
                                                  size_t ld_a, const float* x,
                                                  size_t ld_x, double beta,
                                                  float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// double-buffered version of mv_fp32_16_pure_half_wide_ilp_shared_cpasync
// -- issues the NEXT stage's cp.async copies into a second buffer while
// computing on the CURRENT (already-completed) stage, so the async copy
// engine has real concurrent work instead of being waited on immediately
// (which is what the single-buffered version did, and why it likely
// didn't show cp.async's intended benefit). Same NO_TRANSPOSE-only scope
// and ld_a-must-be-even alignment caveat as the single-buffered version --
// see fp32_16_cuda_kernels.cu for the full rationale and pipelining
// mechanism.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_cpasync2(TransposeType trans,
                                                   double alpha, dim<2> size,
                                                   const uint16_t* wh,
                                                   size_t ld_a, const float* x,
                                                   size_t ld_x, double beta,
                                                   float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// eliminates the split-K design's second kernel launch (mv_fp32_16_wide_
// reduce_kernel, used by every other _wide_* variant to combine col_splits
// blocks' partial sums) by having each block atomicAdd its column-chunk
// sum directly into y, replaced instead by a cheap cudaMemset (beta == 0)
// or small prescale kernel (beta != 0) before the main kernel. Same 4-way
// unroll and shared-memory x_tile as mv_fp32_16_pure_half_wide_ilp_shared.
// See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_atomic(TransposeType trans,
                                                 double alpha, dim<2> size,
                                                 const uint16_t* wh,
                                                 size_t ld_a, const float* x,
                                                 size_t ld_x, double beta,
                                                 float* y, size_t ld_y); }  // namespace experimental


// checked temp-dependencies/randls2.0's own gemv_fp16_cm_pair_ktiled_
// atomic after the split-K family (bigtile_half2, half2_wide4x,
// half2_best) plateaued -- that kernel independently arrives at the SAME
// atomicAdd-instead-of-reduce-kernel combine strategy mv_fp32_16_pure_
// half_wide_ilp_shared_atomic above already tried, cross-validating it.
// This is the fair version of that existing kernel: IDENTICAL grid/
// col_splits/beta-handling (cudaMemset or prescale kernel, no malloc/
// free of any kind -- a genuinely different combine axis from every
// split+reduce sibling in this family) but packs the 4-way unroll into
// __half2/__hfma2 like mv_fp32_16_pure_half_wide_ilp_shared_half2 does,
// instead of the older kernel's own scalar accumulation. No workspace
// parameter needed (no buffer to own).
//
// PROMOTED -- this is "TOMOS FP16" in the benchmark/plot (mv_fp32_16_
// pure_half_wide_ilp_shared_half2, the plain split+reduce kernel, is
// still available and still what every sibling's own comparison
// measures against, but is no longer what "TOMOS FP16" means). CONFIRMED
// on real A100 hardware: dramatically faster at small/medium M (M=1024:
// +108%; M=1536: +58%), competitive-to-ahead of every split+reduce
// sibling at large M too. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic(TransposeType trans,
                                                        double alpha, dim<2> size,
                                                        const uint16_t* wh,
                                                        size_t ld_a, const float* x,
                                                        size_t ld_x, double beta,
                                                        float* y, size_t ld_y); }  // namespace experimental


// "test" -- checked Nsight Compute profiling data for "TOMOS FP16" itself
// at M=4096: the grid only fills 0.59 waves across the A100's 108 SMs
// (achieved occupancy 46.86%, not register/shared-mem limited). Reuses
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_kernel UNCHANGED,
// only computing col_splits via choose_col_splits_target(rows_op,
// cols_op, kMvTargetTotalThreads*2) instead of the fixed
// kMvTargetTotalThreads -- more, smaller column chunks, raising wave
// count toward/past 1.0, but (unlike half2_wide2x's split+reduce) each
// extra split costs one more atomicAdd instead of a whole extra malloc/
// launch/serial-reduce-loop. No workspace needed. See its own comment in
// fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x(TransposeType trans,
                                                               double alpha, dim<2> size,
                                                               const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta,
                                                               float* y, size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// same idea as mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide2x
// above, but quadruples col_splits instead of doubling it
// (kMvTargetTotalThreads*4). See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x(TransposeType trans,
                                                               double alpha, dim<2> size,
                                                               const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta,
                                                               float* y, size_t ld_y); }  // namespace experimental

// col_splits-retuned sibling of wide4x above (kMvTargetTotalThreads*1
// instead of *4, targeting the atomic-combine kernel's own contention
// cost model rather than a serial-reduce cost model) -- see its own
// comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_lowsplit(TransposeType trans,
                                                               double alpha, dim<2> size,
                                                               const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta,
                                                               float* y, size_t ld_y); }  // namespace experimental


// "test" -- combines "TOMOS FP16"'s own atomicAdd combine (no reduce
// kernel, no partial buffer, no malloc/free) with bigtile_half2's own
// 4x-larger shared-memory tile (fewer __syncthreads() rounds). Unlike
// bigtile_half2_adaptive's own failed stacking attempt (bigger tile +
// MORE col_splits/blocks compounded on shared-memory-limited occupancy),
// this pairing changes a DIFFERENT axis: atomicAdd doesn't add blocks or
// change col_splits at all (same grid as bigtile_half2 itself) -- it only
// removes the SECOND kernel launch and its own partial-buffer traffic,
// a cost independent of tile width. See fp32_16_cuda_kernels.cu for the
// full rationale.
//
// CONFIRMED NEGATIVE RESULT on real A100 hardware: no meaningful win
// anywhere -- ratio vs half2_atomic stays within noise (0.98x-1.02x) at
// small/medium square M and every tall-and-thin shape tested, and is a
// mild but consistent regression at large square M (M=16384-65536:
// 0.94x-0.96x). See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic(TransposeType trans,
                                                                double alpha, dim<2> size,
                                                                const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta,
                                                                float* y, size_t ld_y); }  // namespace experimental


// "TOMOS FP16, atomic best-of" -- same dispatch-not-merge idea as
// mv_fp32_16_pure_half_wide_ilp_shared_half2_best (which picks between
// bigtile_half2 and half2_wide4x for the split+reduce family), applied to
// the atomic-combine family: below 4096 rows, calls mv_fp32_16_pure_half_
// wide_ilp_shared_bigtile_half2_atomic; at/above it, calls mv_fp32_16_
// pure_half_wide_ilp_shared_half2_atomic_wide4x. No workspace needed
// (neither callee owns a buffer). See its own comment in fp32_16_cuda_
// kernels.cu -- the 4096 threshold matches the profiling shape that
// motivated wide4x's atomic variant, not a measured crossover between the
// two callees themselves.
//
// UNTESTED: no GEMV performance measurement has been run yet.
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                             double alpha, dim<2> size,
                                                             const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta,
                                                             float* y, size_t ld_y);

// Winning "test2" GEVM-performance candidate: cp.async shared-memory
// staging + warp-uniform issuer selection, same fp32_16_half_only safe
// decode and col_splits/atomicAdd combine as the dispatcher above, only
// the staging mechanism differs (see its own comment in fp32_16_cuda_
// kernels.cu). NO_TRANSPOSE only -- falls back to the dispatcher above
// for TRANSPOSE.
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform(TransposeType trans,
                                                             double alpha, dim<2> size,
                                                             const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta,
                                                             float* y, size_t ld_y);

// No-overflow-guaranteed decode sibling of the dispatcher above -- its
// own accumulate step was already __half2/__hfma2-packed, so this
// isolates the decode-cost lever already proven on the narrow_vec
// family. See its own comment in fp32_16_cuda_kernels.cu. Test-only,
// direct launch (no multi-tier shape dispatch, only ever called for
// rows_op < kMvWideRowThreshold by the caller).
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_no_overflow(
    double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, double beta, float* y);

// "nomask" decode sibling of the dispatcher above: drops the provably-
// redundant `& 0x3FFu` after the subnormal shift -- see fp32_16_half_
// only_branchless_no_overflow_nomask's own comment in fp32_16_pack.hpp.
// Test-only, measuring before promoting the mask removal to the base
// decode function.
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_nomask(
    double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, double beta, float* y);

// Fast-decode sibling of the dispatcher above (uses fp32_16_half_only_fast
// instead of fp32_16_half_only throughout its whole wide4x/bigtile_half2_
// atomic/tiled/narrow_vec dispatch tree) -- added as a new, separate
// function, original untouched. See its own comment in fp32_16_cuda_
// kernels.cu.
void mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y);

// Fast2-decode sibling of the dispatcher above (uses fp32_16_half_only_
// fast2 throughout its whole wide4x/bigtile_half2_atomic/tiled/
// narrow_vec dispatch tree) -- added as a new, separate function,
// original untouched. See its own comment in fp32_16_cuda_kernels.cu and
// fp32_16_half_only_fast2's own comment in fp32_16_pack.hpp.
void mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                   double alpha, dim<2> size,
                                                                   const uint16_t* wh,
                                                                   size_t ld_a, const float* x,
                                                                   size_t ld_x, double beta,
                                                                   float* y, size_t ld_y);

// Genuinely branchless-decode sibling of the fast2 dispatcher above
// (uses fp32_16_half_only_branchless throughout its whole wide4x/
// bigtile_half2_atomic/tiled/narrow_vec dispatch tree, mirroring its
// exact shape-dispatch structure) -- added as a new, separate function,
// original untouched. Benchmark-only for now (mv_fp32_16_pure_half_vs_
// cublas_ref.cu's own "test-new" leg), not yet wired into any dispatch
// registry. See its own comment in fp32_16_cuda_kernels.cu and
// fp32_16_half_only_branchless's own comment in fp32_16_pack.hpp.
void mv_fp32_16_pure_half_branchless_wide_ilp_shared_half2_atomic_best(TransposeType trans,
                                                                        double alpha, dim<2> size,
                                                                        const uint16_t* wh,
                                                                        size_t ld_a, const float* x,
                                                                        size_t ld_x, double beta,
                                                                        float* y, size_t ld_y);

// No-overflow-guaranteed sibling of the branchless dispatcher above --
// requires the caller to guarantee no wh in the buffer ever overflows
// (control=1, direction=1). Measured directly on the real LSQR test
// matrix (tall_thin_1000.mtx): 0% overflow, so this contract genuinely
// holds there. See fp32_16_half_only_branchless_no_overflow's own
// comment in fp32_16_pack.hpp and fp32_16_cuda_kernels.cu. Benchmark-
// only for now (mv_fp32_16_pure_half_vs_cublas_ref.cu's own "test-new"
// leg), not yet wired into any dispatch registry.
void mv_fp32_16_pure_half_branchless_no_overflow_wide_ilp_shared_half2_atomic_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x, size_t ld_x,
    double beta, float* y, size_t ld_y);

// Vectorized-write sibling of the no_overflow narrow_vec kernel: `ncu`
// profiling of the real LSQR shape flagged that kernel's own write-back
// loop directly (4 separate scalar y[i]= stores/reads per thread instead
// of one float4), independent of decode cost. i0 is always a multiple
// of kMvRowsPerThread by construction and y comes from cudaMalloc, so
// this is always safely 16-byte-aligned when the 4-row group is in
// bounds. Direct single-kernel host wrapper (no multi-tier shape
// dispatch), mirroring mv_fp32_16_pure_half_fast2_narrow_vec_cpasync's
// own convention. See its own comment in fp32_16_cuda_kernels.cu.
// Benchmark-only for now (mv_fp32_16_pure_half_vs_cublas_ref.cu's own
// "test-new" leg), not yet wired into any dispatch registry.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store(TransposeType trans, double alpha, size_t m,
                                                                       size_t n, const uint16_t* wh, size_t ld_a,
                                                                       const float* x, double beta, float* y);

// Grid-stride sibling of vec4store above: launches exactly one full
// wave's worth of blocks (queried via cudaOccupancyMaxActiveBlocksPer
// Multiprocessor) instead of one block per row-group, and has each
// block loop internally over multiple row-groups -- fixes the partial-
// wave tail `ncu` flagged for this kernel family's own "4 rows per
// thread" grid sizing (977 blocks, only 864 fit in one wave on this
// A100). See its own comment in fp32_16_cuda_kernels.cu. Benchmark-only
// for now (mv_fp32_16_pure_half_vs_cublas_ref.cu's own "test-new" leg).
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// Shift-numbering sibling of the gridstride kernel above: decodes via
// fp32_16_half_only_branchless_no_overflow_shiftenc (reads the 4-bit
// field as shift_by directly, no subtraction, no zero-guard). REQUIRES
// wh's control=1 elements already renumbered (offset-1, not the
// standard offset fp32_16_split writes) -- see fp32_16_half_only_
// branchless_no_overflow_shiftenc's own comment in fp32_16_pack.hpp.
// Test-only, not wired into any dispatch registry or production
// encoder.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_shiftenc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// half2-accumulation sibling of shiftenc above: decode stays scalar
// (each of the 4 rows needs its own independent shift/mask), but the
// accumulate step packs a0/a1 and a2/a3 into __half2 and uses __hfma2
// against a broadcast x -- halves the FMA instruction count for that
// part of the loop, same technique mv_fp32_16_pure_half_wide_ilp_
// shared_half2_atomic_kernel already uses elsewhere in this file. See
// its own comment in fp32_16_cuda_kernels.cu. Test-only.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_shiftenc_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// "nomask" decode sibling of the kernel above: drops the provably-
// redundant `& 0x3FFu` after the subnormal shift -- see fp32_16_half_
// only_branchless_no_overflow_nomask's own comment in fp32_16_pack.hpp.
// Test-only, measuring before promoting the mask removal to the base
// decode function.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec_vec4store_gridstride_nomask_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// kMvRowsPerThread=2 sibling of the kernel above, used by test-new3 --
// see its own comment in fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec2_vec2store_gridstride_nomask_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// kMvRowsPerThread=8 sibling, used by test-new4 -- see its own comment
// in fp32_16_cuda_kernels.cu.
void mv_fp32_16_pure_half_branchless_no_overflow_narrow_vec4x2_vec4x2store_gridstride_nomask_half2acc(
    TransposeType trans, double alpha, size_t m, size_t n, const uint16_t* wh, size_t ld_a, const float* x,
    double beta, float* y);

// NO_TRANSPOSE-only composed dispatcher for mv_pure_half_impl's own safe-
// decode default: routes to cpasync_warpuniform (rows_op <
// kMvWideRowThreshold) or the bit-exact fast2-decode family (rows_op >=
// that threshold -- the real LSQR matrix's own shape, where
// cpasync_warpuniform's own cp.async optimization never fires). See its
// own comment in fp32_16_cuda_kernels.cu for the full derivation.
void mv_fp32_16_pure_half_no_transpose_fast2(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// Shape-aware composed dispatcher for "test-new"'s own kernel: same
// small-M branch as mv_fp32_16_pure_half_no_transpose_fast2 above
// (cpasync_warpuniform), but the large-M branch routes to the current
// best narrow_vec candidate (no-overflow branchless decode + shiftenc
// numbering + vectorized write-back + grid-stride launch + half2-packed
// accumulate) instead of that kernel's own fast2_best. See its own
// comment in fp32_16_cuda_kernels.cu. Benchmark-only for now.
void mv_fp32_16_pure_half_test_new_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                             const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// "test-new3": same dispatcher as test-new above, but the large-M branch
// uses a kMvRowsPerThread=2 sibling kernel instead of 4 -- testing
// register-blocking vs. occupancy at this shape. See its own comment in
// fp32_16_cuda_kernels.cu. Benchmark-only for now.
void mv_fp32_16_pure_half_test_new3_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// "test-new4": same dispatcher, large-M branch uses the kMvRowsPerThread=8
// sibling kernel -- extends the register-blocking-vs-occupancy trend in
// the opposite direction from test-new3. See its own comment in
// fp32_16_cuda_kernels.cu. Benchmark-only for now.
void mv_fp32_16_pure_half_test_new4_dispatch(double alpha, dim<2> size, const uint16_t* wh, size_t ld_a,
                                              const float* x, size_t ld_x, double beta, float* y, size_t ld_y);

// ====================================================================
// TOMOS FP32/FP16 (mixed precision): NEW, ADDITIVE only. Based on
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic (plain "TOMOS FP16"),
// but any wh entry needing the special 8-bit-exponent encoding (control=1,
// "extended" or "degraded") is decoded EXACTLY via fp32_16_combine(wh,wr)
// instead of saturating to +-Infinity/0 -- hence the extra wr parameter,
// unlike the rest of the wh-only pure_half family. For those entries, x
// is read as the original float (not the shared __half tile), the
// multiply happens in float, and only the result is narrowed to __half
// (the accumulator's own precision) before being added in. Scalar, no
// half2/ILP packing -- targets correctness on rare special-encoded
// entries, not throughput. Only the small-rows_op ("wide matrix") regime
// gets this treatment; narrow-column and large-rows_op shapes fall
// through to the existing, unmodified safe pure_half kernels. See its
// own comment in fp32_16_cuda_kernels.cu for the full derivation.
//
// UNTESTED: no correctness or performance measurement has been run yet.
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_mixed(TransposeType trans,
                                                              double alpha, dim<2> size,
                                                              const uint16_t* wh,
                                                              const uint16_t* wr,
                                                              size_t ld_a, const float* x,
                                                              size_t ld_x, double beta,
                                                              float* y, size_t ld_y);
// ====================================================================
// END TOMOS FP32/FP16 (mixed precision).
// ====================================================================

// "TOMOS FP32/FP16 (fast)", new version -- ports the OLD "(fast)"
// relationship (mv_fp32_16_pure_half_wide_ilp_shared_fasthalf: UNSAFE-on-
// arbitrary-data sibling of the OLD plain half2 kernel) to "TOMOS FP16"'s
// own CURRENT atomicAdd-combine dispatcher instead of that older plain
// kernel. Decode swapped from fp32_16_half_only to fp32_16_fast_half
// (unconditional bit-reinterpret) throughout. SAME CORRECTNESS CAVEAT AS
// THE OLD KERNEL: only valid on data PROVABLY never extended/degraded-
// encoded -- test against gen_fp32_16_wh_safe's own dedicated safe-range
// data, never plain data. See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf(TransposeType trans,
                                                                 double alpha, dim<2> size,
                                                                 const uint16_t* wh,
                                                                 size_t ld_a, const float* x,
                                                                 size_t ld_x, double beta,
                                                                 float* y, size_t ld_y); }  // namespace experimental


// "test" -- combines the new "(fast)" atomicAdd combine above with a
// 4x-larger shared-memory tile. Same correctness caveat. See its own
// comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_atomic_fasthalf(TransposeType trans,
                                                                        double alpha, dim<2> size,
                                                                        const uint16_t* wh,
                                                                        size_t ld_a, const float* x,
                                                                        size_t ld_x, double beta,
                                                                        float* y, size_t ld_y); }  // namespace experimental


// "test" -- REUSES mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_
// fasthalf_kernel unchanged, only quadrupling col_splits. Same
// correctness caveat. See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x(TransposeType trans,
                                                                       double alpha, dim<2> size,
                                                                       const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta,
                                                                       float* y, size_t ld_y); }  // namespace experimental


// "TOMOS FP32/FP16 (fast), best-of" -- same dispatch-not-merge idea as
// mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (the safe-decode
// dispatcher this is based on): below kMvFasthalfAtomicBestCrossover
// Threshold rows, calls mv_fp32_16_pure_half_wide_ilp_shared_bigtile_
// half2_atomic_fasthalf; at/above it, calls mv_fp32_16_pure_half_wide_
// ilp_shared_half2_atomic_fasthalf_wide4x. No workspace needed. Same
// correctness caveat as every callee here. See its own comment in
// fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
void mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_best(TransposeType trans,
                                                                     double alpha, dim<2> size,
                                                                     const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta,
                                                                     float* y, size_t ld_y);

// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// packs the 4-way unroll's 4 columns into two __half2 accumulators and
// uses __hfma2 to do 2 scalar half FMAs per instruction, halving the
// compute instruction count for the same work (wh's decode/loads are
// unchanged -- only the arithmetic consuming them is packed). Same grid/
// col_splits and shared-memory x_tile as mv_fp32_16_pure_half_wide_ilp_
// shared. See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2(TransposeType trans,
                                                double alpha, dim<2> size,
                                                const uint16_t* wh,
                                                size_t ld_a, const float* x,
                                                size_t ld_x, double beta,
                                                float* y, size_t ld_y); }  // namespace experimental


// "test" -- isolates whether removing fp32_16_half_only's own branch
// (via fp32_16_half_only_branchless, bit-exact) helps once it's not also
// confounded with losing __hfma2 packing the way mv_fp32_16_pure_half_
// wide_ilp_shared_branchless was (that one measured 20%-30% SLOWER than
// half2 itself -- diagnosed as the lost packing costing more than the
// branch removal saved, not a decode-cost finding).
//
// CONFIRMED NEGATIVE RESULT even in this fair isolation: 10%-23% slower
// than mv_fp32_16_pure_half_wide_ilp_shared_half2 at every square size --
// FLIPS the original hypothesis: on this workload's data (control=0
// vastly dominant), the branch itself is cheap/well-predicted, while the
// branchless bitwise masks pay real ALU cost on every element regardless.
// See its own comment in fp32_16_cuda_kernels.cu for the full writeup.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered host wrapper for the "test" kernel above -- IDENTICAL
// to mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace below except
// fp32_16_half_only swapped for fp32_16_half_only_branchless on the wide
// path (tiled/narrow_vec fall back to the existing, still-branching
// kernels, same as every sibling in this family).
//
// CONFIRMED NEGATIVE RESULT -- see its own comment in fp32_16_cuda_
// kernels.cu for the real A100 measurements.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless_workspace(TransposeType trans,
                                                                      double alpha, dim<2> size,
                                                                      const uint16_t* wh,
                                                                      size_t ld_a, const float* x,
                                                                      size_t ld_x, double beta,
                                                                      float* y, size_t ld_y,
                                                                      double* workspace); }  // namespace experimental


// Returns the number of doubles mv_fp32_16_pure_half_wide_ilp_shared_
// half2_workspace's wide (split-K) path needs -- same contract as
// mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace_size above.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered sibling of mv_fp32_16_pure_half_wide_ilp_shared_
// half2 above -- same caller-owned-workspace fix as mv_fp32_16_pure_half_
// wide_ilp_shared_fasthalf_workspace (see its own comment in fp32_16_
// cuda_kernels.cu for the full rationale).
//
// PROMOTED for square NO_TRANSPOSE shapes (CONFIRMED on real A100
// hardware); NOT promoted for tall-and-thin -- same dispatch-threshold
// gap as mv_fp32_16_pure_half_wide_ilp_shared_fasthalf_workspace above.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_workspace(TransposeType trans,
                                                           double alpha, dim<2> size,
                                                           const uint16_t* wh,
                                                           size_t ld_a, const float* x,
                                                           size_t ld_x, double beta,
                                                           float* y, size_t ld_y,
                                                           double* workspace); }  // namespace experimental


// NO_TRANSPOSE-usable "TOMOS FP32/FP16" kernel: same wh-only extended-
// exponent reconstruction + adjacent-pair __hfma2 packing as
// mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_vec4_paired, but
// built on mv_fp32_16_pure_half_wide_ilp_shared_half2's own generic
// trans-dispatched structure (4-way unroll of independent scalar wh
// loads) instead of that kernel's TRANSPOSE-only warp-per-column/
// ushort4 one -- NO_TRANSPOSE has no per-thread row-adjacent vector-load
// opportunity to exploit the way TRANSPOSE does. Standalone comparison
// kernel: wide split-K path only, no narrow/tiled fallback. Uses only
// wh, safe on arbitrary data. See its own comment in fp32_16_cuda_
// kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared(TransposeType trans, double alpha,
                                       dim<2> size, const uint16_t* wh,
                                       size_t ld_a, const float* x,
                                       size_t ld_x, double beta, float* y,
                                       size_t ld_y); }  // namespace experimental


// Returns the number of doubles mv_fp32_16_hybrid_wide_ilp_shared_
// workspace's split-K path needs -- unlike the wide_ilp_shared_half2/
// fasthalf workspace_size functions above, this kernel is standalone
// (wide path only, no tiled/narrow_vec fallback), so this always returns
// the split-K size, never 0.
namespace experimental { size_t mv_fp32_16_hybrid_wide_ilp_shared_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered sibling of mv_fp32_16_hybrid_wide_ilp_shared above --
// same caller-owned-workspace fix as mv_fp32_16_pure_half_wide_ilp_
// shared_fasthalf_workspace (see its own comment in fp32_16_cuda_
// kernels.cu for the full rationale). Always uses workspace (no fallback
// path that skips it).
//
// PROMOTED for square NO_TRANSPOSE shapes (CONFIRMED on real A100
// hardware) -- consistently the weakest of the four NO_TRANSPOSE "TOMOS"
// series (real per-element branch divergence), but a validated, not-
// UNTESTED choice. Tall-and-thin not separately investigated (this
// kernel has no narrow_vec/tiled fallback to hit the dispatch-threshold
// gap the other three siblings do) -- see its own comment in fp32_16_
// cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_workspace(TransposeType trans, double alpha,
                                                 dim<2> size, const uint16_t* wh,
                                                 size_t ld_a, const float* x,
                                                 size_t ld_x, double beta, float* y,
                                                 size_t ld_y, double* workspace); }  // namespace experimental


// "TOMOS FP32/FP16", new version -- ports "TOMOS FP32/FP16 (fast)"'s own
// atomicAdd-combine dispatcher structure to the EXPENSIVE decode instead
// of fasthalf's cheap unconditional one: per element, checks the control
// bit -- control=0 takes the cheap __hfma2-packed path; control=1
// reconstructs an approximate float from wh's own bits (needing the full
// 8 exponent bits fp32 itself uses, unlike fp16's own 5) and performs
// THAT element's multiply-accumulate in float. Always split-K (no 3-way
// dispatch), matching the plain hybrid kernel's own scope. See its own
// comment in fp32_16_cuda_kernels.cu.
//
// SUPERSEDED -- this WAS "TOMOS FP32/FP16" (via its own bigtile/wide4x/
// best dispatcher below), but mv_fp32_16_hybrid_wide_ilp_shared_half2_
// atomic_branchless below CONFIRMED a further +6.5% to +22% win over
// THIS kernel standalone, and that win was carried through to a full
// bigtile/wide4x/best dispatcher too (mv_fp32_16_hybrid_wide_ilp_shared_
// half2_atomic_branchless_best -- CONFIRMED +5.7% to +18.7% over this
// kernel's own dispatcher). "TOMOS FP32/FP16" now means that branchless
// dispatcher instead. Kept as a documented sibling, not deleted -- still
// what every OTHER series' own ratio line compares against.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic(TransposeType trans, double alpha,
                                                     dim<2> size, const uint16_t* wh,
                                                     size_t ld_a, const float* x,
                                                     size_t ld_x, double beta, float* y,
                                                     size_t ld_y); }  // namespace experimental


// "test" -- combines the new hybrid atomicAdd combine above with a
// 4x-larger shared-memory tile. See its own comment in fp32_16_cuda_
// kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic(TransposeType trans, double alpha,
                                                             dim<2> size, const uint16_t* wh,
                                                             size_t ld_a, const float* x,
                                                             size_t ld_x, double beta, float* y,
                                                             size_t ld_y); }  // namespace experimental


// "test" -- REUSES mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_kernel
// unchanged, only quadrupling col_splits. See its own comment in
// fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y); }  // namespace experimental


// "TOMOS FP32/FP16, best-of" -- SUPERSEDED: same dispatch-not-merge idea
// as mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best (the family
// this is based on): below kMvHybridAtomicBestCrossoverThreshold rows,
// calls mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic; at/above
// it, calls mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_wide4x. No
// workspace needed. This WAS "TOMOS FP32/FP16" -- CONFIRMED the
// strongest branchy option -- but mv_fp32_16_hybrid_wide_ilp_shared_
// half2_atomic_branchless_best CONFIRMED a further +5.7% to +18.7% win
// over THIS dispatcher. "TOMOS FP32/FP16" now means that branchless
// dispatcher instead. Kept as a documented sibling, not deleted -- still
// what every OTHER series' own ratio line compares against. See its own
// comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best(TransposeType trans, double alpha,
                                                          dim<2> size, const uint16_t* wh,
                                                          size_t ld_a, const float* x,
                                                          size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental


// "test" -- Nsight Compute profiling of mv_fp32_16_hybrid_wide_ilp_shared_
// half2_atomic_kernel at M=4096 found Branch Efficiency only 54.09% and
// heavy divergence (32 independent rows/warp, each independently rolling
// this benchmark's own 10% control=1 rate, means ~97% of warps pay for
// BOTH code paths serially on nearly every element). This variant
// computes both the cheap __half2 path and the extended float-
// reconstruction path UNCONDITIONALLY for every element (no data-
// dependent branch at all), discarding the unwanted one via arithmetic
// masking instead of control flow. See its own comment in fp32_16_cuda_
// kernels.cu for why this is provably safe (no NaN/Inf-through-zero risk)
// and why this is a genuinely different bet from mv_fp32_16_pure_half_
// wide_ilp_shared_half2_branchless's own confirmed loss (that kernel's
// own real-data extended rate is ~0.006%, a completely different
// divergence regime).
//
// CONFIRMED on real A100 hardware: +6.5% to +22% over the branchy
// mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic at every square size.
// NOT YET PROMOTED: mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
// branchless_fastx below tried to extend this into a full bigtile/
// wide4x/best dispatcher and CONFIRMED NEGATIVE instead -- see its own
// comment. Building bigtile/wide4x/best for THIS kernel specifically
// remains open.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless(TransposeType trans, double alpha,
                                                                dim<2> size, const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y,
                                                                size_t ld_y); }  // namespace experimental


// PROMOTED -- this IS "TOMOS FP32/FP16" now (via its own bigtile/wide4x/
// best dispatcher below). Finishes the branchless base kernel's own
// CONFIRMED win (+6.5% to +22% over the branchy kernel) into a full
// dispatcher, WITHOUT the fastx caching that regressed when tried on top
// of this same base kernel (see mv_fp32_16_hybrid_wide_ilp_shared_half2_
// atomic_branchless_fastx below -- CONFIRMED NEGATIVE). REUSES mv_fp32_
// 16_hybrid_process_element_branchless unchanged -- only new dispatch
// structure around it. CONFIRMED on real A100 hardware: the resulting
// dispatcher beats the branchy mv_fp32_16_hybrid_wide_ilp_shared_half2_
// atomic_best by +5.7% to +18.7% at every square size measured. See its
// own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta, float* y,
                                                                       size_t ld_y); }  // namespace experimental


// PROMOTED -- part of "TOMOS FP32/FP16" now. REUSES mv_fp32_16_hybrid_
// wide_ilp_shared_half2_atomic_branchless_kernel unchanged, only
// quadrupling col_splits. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_wide4x(TransposeType trans, double alpha,
                                                                       dim<2> size, const uint16_t* wh,
                                                                       size_t ld_a, const float* x,
                                                                       size_t ld_x, double beta, float* y,
                                                                       size_t ld_y); }  // namespace experimental


// "TOMOS FP32/FP16, best-of" -- PROMOTED: this IS "TOMOS FP32/FP16" now.
// Same dispatch-not-merge idea as mv_fp32_16_hybrid_wide_ilp_shared_
// half2_atomic_best (the branchy dispatcher this replaces): below
// kMvHybridBranchlessBestCrossoverThreshold rows, calls mv_fp32_16_
// hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless; at/above it,
// calls mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_
// wide4x. No workspace needed. UNLIKE the fastx_best dispatcher
// (CONFIRMED NEGATIVE, briefly promoted before being reverted), does NOT
// add the fastx x_tile caching. CONFIRMED on real A100 hardware: beats
// the branchy mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best (still
// what every other series' own ratio line compares against) by +5.7% to
// +18.7% at every square size measured. See its own comment in fp32_16_
// cuda_kernels.cu.
void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best(TransposeType trans, double alpha,
                                                                     dim<2> size, const uint16_t* wh,
                                                                     size_t ld_a, const float* x,
                                                                     size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);

// "test" -- attempted to extend the branchless base kernel's own
// CONFIRMED win into a full bigtile/wide4x/best dispatcher, plus caches a
// float-typed copy of the shared x tile (x_tile_f) meant to avoid
// branchless's own new redundant __half2float(xval_half) call on every
// element. CONFIRMED NEGATIVE RESULT on real A100 hardware instead: 5%-
// 26% SLOWER than plain branchless -- the extra shared-memory READ
// traffic from a second, float-typed tile cost more than the single
// cheap conversion instruction it was meant to save. See its own comment
// in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx(TransposeType trans, double alpha,
                                                                      dim<2> size, const uint16_t* wh,
                                                                      size_t ld_a, const float* x,
                                                                      size_t ld_x, double beta, float* y,
                                                                      size_t ld_y); }  // namespace experimental


// "test" -- combines the branchless_fastx combine above with a
// 4x-larger shared-memory tile (both half- and float-typed copies). See
// its own comment in fp32_16_cuda_kernels.cu.
//
// CONFIRMED NEGATIVE RESULT on real A100 hardware -- part of the whole
// branchless_fastx family's own regression.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_fastx(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test" -- REUSES mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
// branchless_fastx_kernel unchanged, only quadrupling col_splits. See its
// own comment in fp32_16_cuda_kernels.cu.
//
// CONFIRMED NEGATIVE RESULT on real A100 hardware -- part of the whole
// branchless_fastx family's own regression.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_wide4x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test" -- same dispatch-not-merge idea as mv_fp32_16_hybrid_wide_ilp_
// shared_half2_atomic_best: below kMvHybridBranchlessFastxBestCrossover
// Threshold rows, calls mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_
// atomic_branchless_fastx; at/above it, calls mv_fp32_16_hybrid_wide_ilp_
// shared_half2_atomic_branchless_fastx_wide4x. No workspace needed. See
// its own comment in fp32_16_cuda_kernels.cu.
//
// CONFIRMED NEGATIVE RESULT on real A100 hardware: 2%-17% SLOWER than
// mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_best. Briefly promoted
// as "TOMOS FP32/FP16" before being reverted once this was confirmed.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_fastx_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test", NOT YET PROMOTED -- same structure as mv_fp32_16_hybrid_wide_
// ilp_shared_half2_atomic_branchless_kernel, calling ..._intmask instead
// of ..._branchless. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test", NOT YET PROMOTED -- same structure as mv_fp32_16_hybrid_wide_
// ilp_shared_bigtile_half2_atomic_branchless, calling ..._intmask
// instead of ..._branchless. See its own comment in
// fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless_intmask(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test", NOT YET PROMOTED -- REUSES mv_fp32_16_hybrid_wide_ilp_shared_
// half2_atomic_branchless_intmask_kernel unchanged, only quadrupling
// col_splits. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_wide4x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// "test", NOT YET PROMOTED -- same dispatch-not-merge idea as mv_fp32_16_
// hybrid_wide_ilp_shared_half2_atomic_branchless_best. Pending real A100
// confirmation that removing the standalone FP32 multiply actually beats
// mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_best (still
// "TOMOS FP32/FP16" until this is confirmed). See its own comment in
// fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// CONFIRMED NOISY/SHAPE-DEPENDENT RESULT on real A100 hardware, NOT
// promoted -- picking col_splits to minimize wasted GPU wave-slots
// instead of choose_col_splits_target's own plain rounding: a full
// square + tall-and-thin sweep found real regressions at moderate square
// M alternating with real wins at nearby sizes (no clean threshold), but
// a consistent, growing win at large tall-and-thin M (K=4096, M>=262144:
// +4.2% to +6.6%). Real GPU schedulers backfill blocks dynamically
// rather than waiting for a whole wave to drain, so the actual tail-wave
// cost is far smaller and less predictable than ncu's own upper-bound
// estimate assumed. Kept as a documented result, not deleted -- a real
// lead for a future shape-adaptive dispatcher restricted to large tall-
// and-thin M, not a blanket replacement. See its own comment in
// fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// CONFIRMED NOISY/SHAPE-DEPENDENT RESULT on real A100 hardware, NOT
// promoted -- see mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
// branchless_intmask_walign's own comment above for the full sweep.
// mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_best
// remains the strongest candidate for "TOMOS FP32/FP16" (still pending
// correctness confirmation before promotion) since it wins everywhere,
// unlike this one. Kept as a documented result, not deleted. See its own
// comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_intmask_walign_best(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh, size_t ld_a, const float* x,
    size_t ld_x, double beta, float* y, size_t ld_y); }  // namespace experimental


// NO_TRANSPOSE-usable "TOMOS FP32/BF16" kernel: genuine __nv_bfloat16/
// __nv_bfloat162 compute via fp32_bf16_bf16_only (bf16's only decode --
// always correct, no fast/safe split needed), built on the same generic
// wide_ilp_shared structure as mv_fp32_16_hybrid_wide_ilp_shared above,
// for the same reason. Standalone comparison kernel: wide split-K path
// only. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared(TransposeType trans, double alpha,
                                            dim<2> size, const uint16_t* wh,
                                            size_t ld_a, const float* x,
                                            size_t ld_x, double beta, float* y,
                                            size_t ld_y); }  // namespace experimental


// Returns the number of doubles mv_fp32_bf16_pure_bf16_wide_ilp_shared_
// workspace's split-K path needs -- same "standalone, always split-K"
// reasoning as mv_fp32_16_hybrid_wide_ilp_shared_workspace_size above.
namespace experimental { size_t mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered sibling of mv_fp32_bf16_pure_bf16_wide_ilp_shared
// above -- same caller-owned-workspace fix as mv_fp32_16_pure_half_wide_
// ilp_shared_fasthalf_workspace (see its own comment in fp32_16_cuda_
// kernels.cu for the full rationale). Always uses workspace.
//
// PROMOTED for square NO_TRANSPOSE shapes (CONFIRMED on real A100
// hardware) -- comparable to mv_fp32_16_pure_half_wide_ilp_shared_
// fasthalf_workspace's own ratios (both pay one unconditional, branch-
// free decode per element). Tall-and-thin not separately investigated
// (no narrow_vec/tiled fallback to hit the dispatch-threshold gap) -- see
// its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_workspace(TransposeType trans, double alpha,
                                                       dim<2> size, const uint16_t* wh,
                                                       size_t ld_a, const float* x,
                                                       size_t ld_x, double beta, float* y,
                                                       size_t ld_y, double* workspace); }  // namespace experimental


// "TOMOS BF16" -- ports "TOMOS FP16"'s own atomicAdd-combine promotion to
// genuine bf16 compute: IDENTICAL to mv_fp32_bf16_pure_bf16_wide_ilp_
// shared_kernel (same __nv_bfloat162/__hfma2 packing, same fp32_bf16_
// bf16_only decode) except combine is a direct atomicAdd into y instead
// of a partial[] buffer + separate reduce-kernel launch. No 3-way
// dispatch (always split-K), matching the plain bf16 kernel's own scope.
// See its own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic(TransposeType trans, double alpha,
                                                    dim<2> size, const uint16_t* wh,
                                                    size_t ld_a, const float* x,
                                                    size_t ld_x, double beta, float* y,
                                                    size_t ld_y); }  // namespace experimental


// "test" -- combines "TOMOS BF16"'s own atomicAdd combine with a
// 4x-larger shared-memory tile -- the bf16 analog of mv_fp32_16_pure_
// half_wide_ilp_shared_bigtile_half2_atomic. See its own comment in
// fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y); }  // namespace experimental


// "test" -- REUSES mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_kernel
// unchanged, only quadrupling col_splits -- the bf16 analog of mv_fp32_
// 16_pure_half_wide_ilp_shared_half2_atomic_wide4x. See its own comment
// in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x(TransposeType trans, double alpha,
                                                           dim<2> size, const uint16_t* wh,
                                                           size_t ld_a, const float* x,
                                                           size_t ld_x, double beta, float* y,
                                                           size_t ld_y); }  // namespace experimental


// "TOMOS BF16, best-of" -- same dispatch-not-merge idea as mv_fp32_16_
// pure_half_wide_ilp_shared_half2_atomic_best (its own fp16 analog): below
// kMvBf16AtomicBestCrossoverThreshold rows, calls mv_fp32_bf16_pure_bf16_
// wide_ilp_shared_bigtile_atomic; at/above it, calls mv_fp32_bf16_pure_
// bf16_wide_ilp_shared_atomic_wide4x. No workspace needed. See its own
// comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best(TransposeType trans, double alpha,
                                                         dim<2> size, const uint16_t* wh,
                                                         size_t ld_a, const float* x,
                                                         size_t ld_x, double beta, float* y,
                                                         size_t ld_y);

// "TOMOS FP32/BF16", new version -- REQUESTED as a separate, distinctly-
// named kernel family for the "TOMOS FP32/BF16" plot slot, explicitly
// confirmed to be IDENTICAL to "TOMOS BF16" above on wh (same decode,
// same compute, same atomicAdd combine, same bigtile/wide4x/best dispatch
// structure) -- a "_v2" duplicate under its own symbols so this slot gets
// its own independently-measured benchmark leg and verify check. See its
// own comment in fp32_16_cuda_kernels.cu.
//
// UNTESTED: no GEMV performance measurement has been run yet.
namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_v2(TransposeType trans, double alpha,
                                                       dim<2> size, const uint16_t* wh,
                                                       size_t ld_a, const float* x,
                                                       size_t ld_x, double beta, float* y,
                                                       size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic_v2(TransposeType trans, double alpha,
                                                               dim<2> size, const uint16_t* wh,
                                                               size_t ld_a, const float* x,
                                                               size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x_v2(TransposeType trans, double alpha,
                                                              dim<2> size, const uint16_t* wh,
                                                              size_t ld_a, const float* x,
                                                              size_t ld_x, double beta, float* y,
                                                              size_t ld_y); }  // namespace experimental

void mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best_v2(TransposeType trans, double alpha,
                                                            dim<2> size, const uint16_t* wh,
                                                            size_t ld_a, const float* x,
                                                            size_t ld_x, double beta, float* y,
                                                            size_t ld_y);

// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// reuses mv_fp32_16_pure_half_wide_ilp_shared_half2's proven-correct
// kernel unchanged, but doubles col_splits (more, smaller column chunks,
// more blocks, a 2x-larger temporary global "partial" buffer) to test
// whether more column-level parallelism helps on top of the 0.84 result.
// See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
    size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// same idea as mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x, but
// quadruples col_splits instead of doubling it.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
    size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path: a
// properly occupancy-matched version of the vectorized-load idea
// mv_fp32_16_pure_half_wide_regblock tried and failed at (that one shrank
// the total thread count 4x by giving each thread multiple rows, forcing
// a losing col_splits/occupancy tradeoff). This kernel keeps exactly one
// row per thread (identical grid/col_splits/occupancy to the baseline)
// and instead has 1 of every 4 threads issue a single 8-byte vectorized
// load covering its own row and the next 3 (the naturally contiguous axis
// in this column-major layout), then all 4 threads retrieve their own
// element via __shfl_sync -- reducing the number of memory transactions
// per column 4x while moving identical total bytes. ALIGNMENT REQUIREMENT
// (ENFORCED): the 8-byte vectorized load needs ld_a to be a multiple of
// 4; this dispatch function checks that and falls back to the proven
// mv_fp32_16_pure_half_wide_ilp_shared whenever it doesn't hold (a
// misaligned load here is a hardware fault, not just a wrong answer --
// confirmed via the correctness suite's ld_a-misaligned test cases before
// this fallback was added). See fp32_16_cuda_kernels.cu for the full
// rationale and mechanism.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_vecrow(TransposeType trans,
                                                 double alpha, dim<2> size,
                                                 const uint16_t* wh,
                                                 size_t ld_a, const float* x,
                                                 size_t ld_x, double beta,
                                                 float* y, size_t ld_y); }  // namespace experimental


// "test" -- combines vecrow's own shuffle-based row-grouped load (above)
// with mv_fp32_16_pure_half_wide_ilp_shared_half2's own __hfma2-packed
// arithmetic, instead of vecrow's own plain scalar accumulate -- avoids
// the same "tested on the older scalar body" confound mv_fp32_16_pure_
// half_wide_ilp_shared_branchless fell into (measured 20%-30% slower than
// half2 for exactly that reason).
//
// CONFIRMED SEVERE NEGATIVE RESULT: ~4x SLOWER than mv_fp32_16_pure_half_
// wide_ilp_shared_half2 at every large square size -- the __shfl_sync
// mechanism itself is the problem, not just "no gain": half2 alone is
// already not L1TEX-stalled (~83% of cuBLAS Hgemm), so there is no
// latency problem left for the shuffles to hide, only their own real
// synchronization cost, paid unconditionally by every warp. See its own
// comment in fp32_16_cuda_kernels.cu for the full writeup.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered host wrapper for the "test" kernel above -- same
// ld_a%4 alignment requirement as mv_fp32_16_pure_half_wide_ilp_shared_
// vecrow's own wrapper, but falls back to mv_fp32_16_pure_half_wide_ilp_
// shared_half2_workspace (not the plain malloc/free function) when it
// doesn't hold, so the fallback path stays apples-to-apples with the
// other workspace-based legs too.
//
// CONFIRMED SEVERE NEGATIVE RESULT -- see its own comment in fp32_16_
// cuda_kernels.cu for the real A100 measurements.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_vecrow_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental


// "test" -- ports temp-dependencies/randls2.0's own hgemv_cm_vec2rows_
// fp16 row-vectorization mechanism (each thread owns 2 ADJACENT rows and
// loads BOTH via a single vectorized read -- no cross-thread __shfl_sync
// needed, unlike vecrow above, since the same thread already owns both
// values it needs) to this codebase's own K-split/shared-x-tile/partial-
// buffer structure and fp32_16_half_only decode. Built after both
// _half2_branchless (decode-side, mild negative) and _half2_vecrow
// (memory-side via cross-thread shuffle, SEVERE negative) failed to close
// the gap to cuBLAS. NO_TRANSPOSE only (needs column-major row-
// adjacency); falls back to mv_fp32_16_pure_half_wide_ilp_shared_half2_
// workspace for TRANSPOSE or odd ld_a. GRID-MATH CAVEAT (reused from
// mv_fp32_16_pure_half_wide_regblock_kernel's own documented regression):
// col_splits must be computed from the FULL un-halved rows_op, not the
// register-blocked thread count, or the reduce step's own cost inflates
// catastrophically. See its own comment in fp32_16_cuda_kernels.cu.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental


// Workspace-buffered host wrapper for the "test" kernel above. See its
// own comment in fp32_16_cuda_kernels.cu.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_vec2rows_workspace(TransposeType trans,
                                                                    double alpha, dim<2> size,
                                                                    const uint16_t* wh,
                                                                    size_t ld_a, const float* x,
                                                                    size_t ld_x, double beta,
                                                                    float* y, size_t ld_y,
                                                                    double* workspace); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// combines the two independent legitimate wins found so far --
// ilp_shared_half2's packed __half2/__hfma2 arithmetic (0.83 -> 0.84) and
// ilp_shared_bigtile's quadrupled shared-memory tile (0.80 alone) -- to
// test whether they stack. See fp32_16_cuda_kernels.cu for the full
// rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2(
    TransposeType trans, double alpha, dim<2> size, const uint16_t* wh,
    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
    size_t ld_y); }  // namespace experimental


// "test" workspace siblings for the three launch/structure-tuning
// variants above (half2_wide2x/half2_wide4x/bigtile_half2) -- built after
// three independent row-blocking experiments (branchless, vecrow,
// vec2rows) all confirmed occupancy/thread-count matters more than load-
// instruction-count for "TOMOS FP16" itself. These three take the
// opposite approach: none reduce thread count (wide2x/wide4x increase
// it via more column-splitting; bigtile_half2 keeps it identical, only
// widening the shared-memory tile), so they don't carry that risk.
// Reuse the existing kernels/dispatch unchanged -- only the malloc/free
// -> caller-owned-workspace buffer differs. See their own comment in
// fp32_16_cuda_kernels.cu.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide2x_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_wide4x_workspace(TransposeType trans,
                                                                  double alpha, dim<2> size,
                                                                  const uint16_t* wh,
                                                                  size_t ld_a, const float* x,
                                                                  size_t ld_x, double beta,
                                                                  float* y, size_t ld_y,
                                                                  double* workspace); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_workspace(TransposeType trans,
                                                                   double alpha, dim<2> size,
                                                                   const uint16_t* wh,
                                                                   size_t ld_a, const float* x,
                                                                   size_t ld_x, double beta,
                                                                   float* y, size_t ld_y,
                                                                   double* workspace); }  // namespace experimental


// "test" -- combines the two independent, independently-confirmed wins
// found after "TOMOS FP16" itself: bigtile_half2's own 4x-larger shared-
// memory tile (a safe, consistent win at essentially every shape
// measured, never meaningfully worse than half2) and half2_wide4x's own
// col_splits quadrupling (the single biggest win measured so far at
// large M, but a real regression at small M). Uses bigtile_half2's own
// kernel unchanged, with an ADAPTIVE col_splits target: 4x at/above
// kMvHalf2AdaptiveWideMultiplierThreshold rows (where wide4x's own
// multiplier is a confirmed win), 1x below it (where it's a confirmed
// loss) -- so this is expected to be at least as good as bigtile_half2
// alone everywhere, possibly better above the threshold if the two wins
// stack.
//
// CONFIRMED NEGATIVE RESULT: the two do NOT stack -- WORSE than both
// parents alone at every square size where the 4x multiplier applies
// (e.g. M=65536: wide4x=1193.4, bigtile_half2=1143.8, this=1072.8
// GFlop/s). Likely shared-memory-occupancy interference (bigger tile x
// more concurrent blocks compounds on the same resource). See mv_fp32_16_
// pure_half_wide_ilp_shared_half2_best_workspace below for the fix
// (dispatch to whichever proven kernel suits the shape, don't merge
// them) and its own comment in fp32_16_cuda_kernels.cu for the full
// writeup.
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace_size(TransposeType trans,
                                                                                  dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_adaptive_workspace(TransposeType trans,
                                                                            double alpha, dim<2> size,
                                                                            const uint16_t* wh,
                                                                            size_t ld_a, const float* x,
                                                                            size_t ld_x, double beta,
                                                                            float* y, size_t ld_y,
                                                                            double* workspace); }  // namespace experimental


// "TOMOS FP16, best-of" -- after bigtile_half2_adaptive above confirmed
// the two mechanisms don't stack when merged into one kernel, dispatches
// to whichever ALREADY-PROVEN kernel is better for the shape instead: no
// merging, no interaction risk. Below kMvHalf2BestSquareCrossoverThreshold
// rows, calls bigtile_half2_workspace (the safe, consistently-good-or-
// better choice at small/medium M); at/above it, calls half2_wide4x_
// workspace (the single biggest win measured in this whole
// investigation, +8% at large square M). See its own comment in
// fp32_16_cuda_kernels.cu for the measured crossover and its own caveats
// (tuned to the square case; narrow cols_op shapes have a much later,
// unmeasured crossover, but the worst case there is a near-tie, not a
// regression).
namespace experimental { size_t mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_pure_half_wide_ilp_shared_half2_best_workspace(TransposeType trans, double alpha,
                                                                dim<2> size, const uint16_t* wh,
                                                                size_t ld_a, const float* x,
                                                                size_t ld_x, double beta, float* y,
                                                                size_t ld_y, double* workspace); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path: the
// first change in this whole investigation to actually beat the baseline
// (mv_fp32_16_pure_half_wide_ilp_shared, 4-way unrolled + shared-memory
// x_tile, measured ~0.83 of cuBLAS vs the baseline's ~0.71 at
// 32768x32768). This variant pushes the same idea to 8-way unrolling
// instead of 4-way, to see if more independent accumulators keeps helping
// or has already hit diminishing returns/register pressure. See
// fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp8_shared(TransposeType trans, double alpha,
                                          dim<2> size, const uint16_t* wh,
                                          size_t ld_a, const float* x,
                                          size_t ld_x, double beta, float* y,
                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_pure_half's wide-matrix path:
// same idea as mv_fp32_16_pure_half_wide_ilp_shared/_ilp8_shared, but
// unrolled 2-wide. Measured so far at 32768x32768: 1-way (baseline) ~0.71
// of cuBLAS, 4-way ~0.83, 8-way ~0.74 -- non-monotonic with a peak around
// 4, so this brackets the low side to help characterize where the actual
// peak is. See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_pure_half_wide_ilp2_shared(TransposeType trans, double alpha,
                                          dim<2> size, const uint16_t* wh,
                                          size_t ld_a, const float* x,
                                          size_t ld_x, double beta, float* y,
                                          size_t ld_y); }  // namespace experimental


// EXPERIMENTAL alternative to mv_fp32_16_accurate/fast's wide-matrix paths:
// applies the winning pattern found for pure_half (mv_fp32_16_pure_half_
// wide_ilp_shared -- 4-way column unrolling kept alongside the baseline's
// shared-memory x_tile, not instead of it -- measured 0.83 of cuBLAS vs
// the plain baseline's 0.71 at 32768x32768) to the accurate and fast
// tiers. See fp32_16_cuda_kernels.cu for the full rationale.
namespace experimental { void mv_fp32_16_accurate_wide_ilp_shared(TransposeType trans, double alpha,
                                        dim<2> size, const uint16_t* wh,
                                        const uint16_t* wr, size_t ld_a,
                                        const float* x, size_t ld_x,
                                        double beta, float* y, size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_fast_wide_ilp_shared(TransposeType trans, double alpha,
                                    dim<2> size, const uint16_t* wh,
                                    size_t ld_a, const float* x, size_t ld_x,
                                    double beta, float* y, size_t ld_y); }  // namespace experimental


}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_FP32_16_CUDA_KERNELS_CUH

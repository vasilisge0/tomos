#ifndef TOMOS_MM_CUDA_KERNELS_CUH
#define TOMOS_MM_CUDA_KERNELS_CUH

#include <cublas_v2.h>
#include <cusparse.h>

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "dimensions.hpp"
#include "precision.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {

// See the matching banner comment in mm_cuda_kernels.cu -- everything
// below except the thin forwarding declarations just before mm_csr/
// mm_csr_csr (unrelated sparse operations, left untouched) is declared
// inside tomos::cuda::experimental; those forwarding wrappers re-expose
// exactly the plotted subset at plain tomos::cuda scope, unchanged.
namespace experimental {

template <typename a_t, typename b_t, typename c_t>
__global__ void gemm_generic_kernel(TransposeType transA, TransposeType transB,
                                    double alpha, size_t m, size_t k, size_t n,
                                    a_t* A, size_t ld_a, b_t* B, size_t ld_b,
                                    double beta, c_t* C, size_t ld_c);

// C = alpha * op(A) * op(B) + beta * C  (column-major strided storage)
template <typename a_t, typename b_t, typename c_t>
void mm_strided(TransposeType transA, TransposeType transB, double alpha,
                dim<2> size_A, a_t* A, size_t ld_a,
                dim<2> size_B, b_t* B, size_t ld_b,
                double beta, c_t* C, size_t ld_c, cublasHandle_t handle);

// Custom fp32 GEMM kernel, tuned against cuBLAS's Sgemm (mm_strided<float,
// float,float> above): 128x128 output tile per block, 8x8 register-blocked
// accumulation per thread, BK=8 shared-memory-cached K-slices. NO_TRANSPOSE
// x NO_TRANSPOSE only -- falls back to gemm_generic_kernel for any
// transposed operand. See mm_cuda_kernels.cu for the __global__ kernel
// (gemm_fp32_tiled_kernel) and this function's dispatch logic.
void gemm_fp32_tiled(TransposeType transA, TransposeType transB, double alpha,
                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                     float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same tiling as gemm_fp32_tiled, but double-buffered: prefetches the next
// K-slice into the other shared-memory buffer while accumulating the
// current one, instead of loading then computing then loading again. See
// mm_cuda_kernels.cu for gemm_fp32_tiled_db_kernel.
void gemm_fp32_tiled_db(TransposeType transA, TransposeType transB, double alpha,
                        size_t m, size_t k, size_t n, float* A, size_t ld_a,
                        float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same tiling and double-buffering as gemm_fp32_tiled_db, but the prefetch
// goes through sm_80's cp.async (__pipeline_memcpy_async) instead of a
// register round-trip -- genuinely asynchronous, so the SM can keep
// computing on the current K-slice while the next one's bytes are still in
// flight. See mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_kernel.
void gemm_fp32_tiled_cpasync(TransposeType transA, TransposeType transB, double alpha,
                             size_t m, size_t k, size_t n, float* A, size_t ld_a,
                             float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same cp.async double-buffering as gemm_fp32_tiled_cpasync, plus float4
// vectorization of the compute loop's shared-memory reads and of A's
// global-to-shared cp.async load. Requires ld_a and m to be multiples of 4
// (checked internally); falls back to gemm_fp32_tiled_cpasync otherwise.
// See mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_vec_kernel and why
// B's load stays scalar (its natural contiguous direction conflicts with
// the compute loop's under the current shared-memory layout).
void gemm_fp32_tiled_cpasync_vec(TransposeType transA, TransposeType transB, double alpha,
                                 size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                 float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same as gemm_fp32_tiled_cpasync_vec, plus vectorizing B's global load
// (via a synchronous float4 read scattered into shared memory, since B's
// shared-memory layout can't be a cp.async destination directly -- see
// mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_vec2_kernel). Requires
// ld_b and k to be multiples of 4 in addition to gemm_fp32_tiled_cpasync_
// vec's own ld_a/m requirement; falls back through that kernel, then
// gemm_fp32_tiled_cpasync, as needed.
void gemm_fp32_tiled_cpasync_vec2(TransposeType transA, TransposeType transB, double alpha,
                                  size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                  float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same as gemm_fp32_tiled_cpasync_vec2, but with the K-slice doubled from 8
// to 16 -- gives B's vectorized load 4 threads sharing each column instead
// of 2 (a 64-byte coalesced chunk instead of 32), at the cost of double the
// shared memory per buffer (still under the 48KB static limit). Same
// alignment/boundary requirements as vec2; falls back through it as
// needed. See mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_vec3_kernel.
void gemm_fp32_tiled_cpasync_vec3(TransposeType transA, TransposeType transB, double alpha,
                                  size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                  float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Native fp32 sibling of mm_fp32_bf16_tiled_cpasync_dualbuf/mm_fp32_16_
// tiled_cpasync_dualbuf -- same 128x128/BK=16 tile and warp/lane geometry,
// but with BOTH operands routed through a double-buffered cp.async
// pipeline (vec3 above only pipelines A; B is a synchronous load there).
// Same alignment/boundary requirements as vec3; falls back to it as
// needed. See mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_dualbuf_kernel.
void gemm_fp32_tiled_cpasync_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_cpasync_dualbuf except the
// inner kk-loop adds a register-level software-pipeline stage: a_reg/
// b_reg for kk+1 are prefetched from shared memory before kk's FFMAs
// run, instead of load-then-compute per kk. Motivated by an ncu/SASS
// comparison against real CUTLASS's SIMT fp32 kernel showing both
// kernels are equally register-bound (~24.7% occupancy each) and do
// identical per-kk work, but CUTLASS's compiled loop interleaves loads
// with FFMAs one-for-one while the plain kernel does them in separate
// bursts -- see mm_cuda_kernels.cu for gemm_fp32_tiled_cpasync_dualbuf_
// regpipe_kernel and fp32_recon_vs_cutlass_ref's own results for the
// measured outcome.
void gemm_fp32_tiled_cpasync_dualbuf_regpipe(TransposeType transA, TransposeType transB, double alpha,
                                             size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                             float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// A third native-fp32 pipeline strategy, this time modeled directly on
// real CUTLASS's MmaPipelined::gemm_iters (cutlass/gemm/threadblock/
// mma_pipelined.h) rather than guessing from SASS/PTX alone: no
// cp.async anywhere. A single plain synchronous global load for the
// next tile is issued at the FIRST kk-step and stored to shared memory
// only at the LAST kk-step (letting its latency hide behind ordinary,
// unconstrained instruction scheduling instead of a hardware async-
// copy engine's explicit synchronization primitives), combined with
// gemm_fp32_tiled_cpasync_dualbuf_regpipe's own register-level double-
// buffered shared-memory reads. No separate staging buffer needed, so
// this kernel's shared-memory footprint is half the cp.async kernels'
// (~32KB vs ~48KB). See mm_cuda_kernels.cu for gemm_fp32_tiled_dualbuf_
// syncpipe_kernel.
void gemm_fp32_tiled_dualbuf_syncpipe(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_dualbuf_syncpipe except its
// address computation is a rolling pointer advanced by a precomputed
// stride instead of recomputing row0+kg*ld (a fresh IMAD) every tile --
// the same technique gemm_fp32_tiled_syncpipe3_incptr applies to the
// 3-stage kernel, but tried here on the simpler 2-stage kernel too,
// since 3-stage's version of this change pushed register usage into
// real spilling and syncpipe has one fewer stage's worth of in-flight
// state, so it may have the register headroom to avoid that. See
// mm_cuda_kernels.cu for gemm_fp32_tiled_dualbuf_syncpipe_incptr_kernel.
void gemm_fp32_tiled_dualbuf_syncpipe_incptr(TransposeType transA, TransposeType transB, double alpha,
                                             size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                             float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// A genuine 3-stage extension of gemm_fp32_tiled_dualbuf_syncpipe: two
// tiles of lookahead (fetch tile i+2 while consuming tile i) instead of
// one, via a 3-deep rotating shared-memory buffer -- still no cp.async,
// same plain-synchronous-load mechanism as syncpipe, just one more
// pipeline stage deep. See mm_cuda_kernels.cu for gemm_fp32_tiled_
// syncpipe3_kernel.
void gemm_fp32_tiled_syncpipe3(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                               float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe3 except its
// address computation for each tile is a rolling pointer advanced by a
// precomputed stride, instead of recomputing row0+kg*ld (a fresh IMAD)
// on every tile -- mirroring CUTLASS's PredicatedTileIterator, which
// does the same thing. Motivated by ncu profiling data showing CUTLASS
// uses the ALU pipe ~63% more than syncpipe3 while syncpipe3 pays more
// in math_pipe_throttle/not_selected stalls, plus a SASS instruction-
// mix comparison (237 IMAD vs CUTLASS's 137) -- IMAD shares the FMA
// pipe on this architecture, so this change moves that address
// arithmetic off the pipe compute actually needs. See mm_cuda_
// kernels.cu for gemm_fp32_tiled_syncpipe3_incptr_kernel.
void gemm_fp32_tiled_syncpipe3_incptr(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same rolling-pointer technique as gemm_fp32_tiled_syncpipe3_incptr,
// but with the live register state trimmed: one tracked pointer per
// operand instead of two (the second v-group's pointer is a fixed
// offset from the first, derived on demand), and one shared tile-fetch
// counter instead of four separate absolute K-positions. Motivated by
// confirming CUTLASS's own kernel has ZERO register spilling (126
// registers, 0-byte stack frame, 0 STL/LDL) at the same rolling-pointer
// technique, while syncpipe3_incptr's more straightforward port spills
// -- this tests whether trimming the redundant live state closes that
// gap. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe3_incptr_
// lean_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_lean(TransposeType transA, TransposeType transB, double alpha,
                                           size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                           float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same as gemm_fp32_tiled_syncpipe3_incptr_lean, with two more register-
// pressure cuts: row/column validity collapsed to single-bit predicates
// (computed once, not kept alive as raw 64-bit values for repeated
// comparison), and the K bound tracked as a 32-bit remaining-K
// countdown instead of a 64-bit absolute tile-fetch position multiplied
// by BK every call. See mm_cuda_kernels.cu for gemm_fp32_tiled_
// syncpipe3_incptr_lean2_kernel for the full reasoning.
void gemm_fp32_tiled_syncpipe3_incptr_lean2(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same rolling-pointer/shared-counter structure as gemm_fp32_tiled_
// syncpipe3_incptr_lean (1 pointer/operand, 1 shared tile-fetch
// counter), WITHOUT that kernel's boolean-predicate or narrow-int
// trims -- the only change here is dropping the precomputed stride
// locals (a_stride, a_half_stride, b_col_stride) and inlining those
// multiplications at each point of use, letting ptxas choose whether
// to cache ld_a/ld_b in a register or reload from constant memory
// (cmem[0]) on demand. Isolates that one idea from the others tried in
// incptr_lean/incptr_lean2. See mm_cuda_kernels.cu for gemm_fp32_tiled_
// syncpipe3_incptr_nostride_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_nostride(TransposeType transA, TransposeType transB, double alpha,
                                               size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                               float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same as gemm_fp32_tiled_syncpipe3_incptr_nostride, plus option 4 from
// the original register-reduction brainstorm: the K bound is a
// decrementing k_remaining countdown instead of an absolute
// tiles_fetched*BK position -- removes a multiply from every call at
// no extra register cost. Deliberately not combined with the boolean-
// predicate/narrow-int trims tried in incptr_lean/incptr_lean2, since
// nostride alone already had the best spill result of the family. See
// mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe3_incptr_nostride_
// kremain_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_nostride_kremain(TransposeType transA, TransposeType transB, double alpha,
                                                        size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                        float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same as gemm_fp32_tiled_syncpipe3_incptr_nostride, with b_col reduced
// from a 2-element tracked array to a single tracked value (b_col0) --
// b_col1 is always a fixed compile-time-constant offset from b_col0,
// exactly the same relationship already exploited for a_ptr0/b_ptr0
// (only one pointer per operand tracked, the other derived on demand);
// this applies that same pattern to b_col itself. See mm_cuda_
// kernels.cu for gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_nostride_bcol(TransposeType transA, TransposeType transB, double alpha,
                                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Isolates option 4 (decrementing k_remaining countdown instead of an
// absolute tiles_fetched*BK position) ALONE, on top of gemm_fp32_tiled_
// syncpipe3_incptr_lean's base structure -- unlike gemm_fp32_tiled_
// syncpipe3_incptr_nostride_kremain, this does NOT also drop option 3's
// precomputed stride locals. Every prior "kremain" measurement was
// stacked on top of nostride, conflating option 4's own effect with
// option 3's; this kernel isolates it. See mm_cuda_kernels.cu for
// gemm_fp32_tiled_syncpipe3_incptr_kremain_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_kremain(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                              float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Applies gemm_fp32_tiled_syncpipe3_incptr_nostride's addressing pattern
// (1 tracked pointer per operand, 1 shared tiles_fetched counter, no
// precomputed stride locals) to the SHALLOWER 2-stage dualbuf_syncpipe
// pipeline instead of the 3-stage syncpipe3 one -- less concurrent live
// state (1 tile of lookahead instead of 2) going into the same
// addressing overhead. See mm_cuda_kernels.cu for gemm_fp32_tiled_
// dualbuf_syncpipe_incptr_nostride_kernel.
void gemm_fp32_tiled_dualbuf_syncpipe_incptr_nostride(TransposeType transA, TransposeType transB, double alpha,
                                                       size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                       float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Identical to gemm_fp32_tiled_syncpipe3_incptr_nostride EXCEPT for the
// __launch_bounds__ minBlocksPerMultiprocessor: 1 instead of 2, giving
// ptxas up to 255 registers/thread (vs the 128-register hard ceiling at
// 2 blocks/SM) so nostride's peak live state should no longer need to
// spill -- at the cost of halving max resident blocks/SM. See mm_cuda_
// kernels.cu for gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1_kernel.
void gemm_fp32_tiled_syncpipe3_incptr_nostride_lb1(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                    float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Applies to gemm_fp32_tiled_syncpipe3 the two structural traits
// confirmed from CUTLASS's own MmaPipelined::gemm_iters and its
// MmaSimtPolicy<... GemmShape<4,4,1>> lane shape: (1) fine-grained
// load/compute interleaving -- the 8x8 accumulator tile is split into
// four 4x4 quadrants, each interleaved with one of next_slot's four
// float4 register loads instead of loading all four upfront then
// running one monolithic 64-FFMA nest; (2) split global-load issue --
// A's and B's next-next-tile fetches are issued at different kk
// indices (kk==0, kk==BK/2) instead of bunched together at kk==0. Same
// total math, same addressing (no rolling pointers), same launch
// bounds as syncpipe3 -- this isolates the scheduling-granularity
// hypothesis from every syncpipe3_incptr_* variant's addressing
// angle. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe3_
// finegrain_kernel, including why this differs from gemm_fp32_tiled_
// cpasync_dualbuf_regpipe's earlier (null-result) reorder attempt.
void gemm_fp32_tiled_syncpipe3_finegrain(TransposeType transA, TransposeType transB, double alpha,
                                         size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                         float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe3 EXCEPT the
// accumulator: each output element's sum is split into TWO independent
// partial accumulators (alternating by kk parity, combined only in the
// epilogue), breaking the RAW dependency chain across kk-iterations
// into two 8-deep chains instead of one 16-deep chain. Motivated by
// matched-occupancy ncu data showing syncpipe3's "wait" (fixed-latency
// execution dependency) stall share is higher than CUTLASS's (15.4% vs
// 11.6%) while memory-latency stall categories are near zero for both --
// but this is a modest, not dominant, slice of the gap (CUTLASS is also
// diffusely better on math_pipe_throttle/not_selected and diffusely
// WORSE on barrier), and doubling the accumulator register footprint
// (64 -> 128 floats) is likely to reintroduce spilling. See mm_cuda_
// kernels.cu for gemm_fp32_tiled_syncpipe3_dualacc_kernel and its own
// comment for the full ncu numbers this was built from.
void gemm_fp32_tiled_syncpipe3_dualacc(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                       float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe3 EXCEPT the FFMA
// accumulate loop's traversal order: a snake (boustrophedon) sweep
// (even rows sweep j forward, odd rows sweep it backward) instead of
// plain row-major, so consecutive rows connect at the same j index
// instead of wrapping from j=TN-1 back to j=0. Targets the measured
// FFMA .reuse ratio gap (syncpipe3: 68.5%, CUTLASS: 87.2%) without
// touching addressing, load structure, tile size, or accumulator
// structure. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe3_
// snake_kernel and its own comment for the full reasoning.
void gemm_fp32_tiled_syncpipe3_snake(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                     float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Same 3-stage rotation, TM=TN=8 register tile, and warp/lane geometry
// as gemm_fp32_tiled_syncpipe3 -- the only change is the K-tile depth:
// BK=8 instead of 16, matching CUTLASS's own confirmed ThreadblockShape
// <128,128,8>. Uses a fully separate BK=8 shared-memory layout and
// indexing (not the shared kGemmFp32DualBK=16 constants every other
// kernel in this file depends on), including a genuinely different
// thread-to-element mapping (one float4 load per thread instead of
// two, since the BK=8 tile is half the size at the same thread count).
// See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe3_bk8_kernel and
// its own comment for the full reasoning.
void gemm_fp32_tiled_syncpipe3_bk8(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                   float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe3_bk8 EXCEPT the
// pipeline depth: 4 rotating shared-memory stages instead of 3 (three
// tiles of lookahead instead of two). Affordable because BK=8 halves
// each stage's cost versus BK=16, freeing up shared-memory budget --
// mirrors exactly how syncpipe (2-stage) was extended to syncpipe3
// (3-stage) earlier this session, the one technique in this whole
// investigation with a confirmed positive measured result, now applied
// to the bk8 base. Targets bk8's one identified downside versus
// syncpipe3 (doubled sync/tile-transition frequency from the smaller
// BK). See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_kernel.
void gemm_fp32_tiled_syncpipe4_bk8(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                   float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe4_bk8 EXCEPT the
// warp/lane geometry used for the compute read path: this uses
// CUTLASS's actual confirmed WarpShape<32,64,8> (32 rows x 64 cols per
// warp, vs this file's own 64x32) and its MmaSimtPolicy<MatrixShape
// <4,8>, RowMajorInterleaved<2>,...> lane layout -- including the exact
// interleaved lane-to-(row,col) mapping read from CUTLASS's own
// cutlass/layout/matrix.h (RowMajorInterleaved::inverse()), not a
// plain divide/modulo split. The tile-fetch (global load / store-to-
// shared-memory) path is untouched, since it addresses by flat thread
// id, not warp/lane geometry. The biggest, most structural change
// attempted in this investigation -- every other kernel kept the warp/
// lane grid fixed. See mm_cuda_kernels.cu for gemm_fp32_tiled_
// syncpipe4_bk8_warptile_kernel and its own comment for the full
// derivation.
void gemm_fp32_tiled_syncpipe4_bk8_warptile(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe4_bk8_warptile
// EXCEPT the global-load addressing: applies the syncpipe3_incptr_
// nostride technique (one tracked rolling pointer per operand, no
// precomputed stride locals) instead of recomputing &A[a_row0 +
// (k0+a_kk)*ld_a] from scratch every call. Genuinely untried
// combination -- the incptr/nostride address-reduction family was only
// ever tried on the BK=16 base (and lost there); BK=8 was only ever
// tried with pipeline-depth and geometry changes, never addressing.
// Motivating difference: at BK=8, load_global runs twice as often per
// unit of K than at BK=16, so the recomputed multiply this removes is
// a proportionally bigger fraction of the total instruction stream.
// See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_warptile_
// nostride_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride(TransposeType transA, TransposeType transB, double alpha,
                                                      size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                      float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to gemm_fp32_tiled_syncpipe4_bk8_warptile_
// nostride EXCEPT the FFMA accumulate loop's traversal order: the same
// snake (boustrophedon) sweep gemm_fp32_tiled_syncpipe3_snake used,
// retried here instead of on syncpipe3. Motivation: snake failed on
// syncpipe3 (introduced a new spill on a kernel already sitting exactly
// at the 128-register ceiling, and the reuse ratio didn't even move).
// wtnostride naturally compiles to 126 registers -- 2 below the
// ceiling -- and its own reuse ratio is still 67.2%, essentially
// unchanged from syncpipe3's 68.5% and still far from CUTLASS's 87.2%,
// confirming this gap is real and untouched by every other change that
// got wtnostride to 0.963x vs CUTLASS. See mm_cuda_kernels.cu for
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Identical geometry and snake FFMA traversal to
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake, but the next-kk
// shared-memory prefetch and next-tile global prefetch are issued
// piecemeal, interleaved into the 8x8 FFMA nest, instead of batched
// entirely before it. Motivated by disassembling CUTLASS's own compiled
// SASS mainloop: it uses the exact same alternating-operand snake reuse
// pattern we already have, but spreads its LDS/LDG/address-math
// instructions roughly one every 1-2 FFMAs through the chain rather than
// clustering them at the front, which may give ptxas's scheduler fewer
// legal reorderings and better preserve the snake .reuse adjacency. See
// mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
// snake_interleaved_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Identical to gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_
// interleaved EXCEPT the row-loop checkpoints where the next-kk and
// next-tile prefetches are issued: i==5/6/7 instead of i==2/4/6. The
// interleaved kernel hit exactly the 128-register ceiling and picked up
// light spilling (2 STL + 2 LDL) -- checkpointing mid-block means the
// prefetched a_reg[next_slot]/b_reg[next_slot] values must stay live
// from wherever they're written through the end of the current kk
// iteration's row loop before their real consumer (the next kk
// iteration); checkpointing early (i==2) extends that live range across
// rows 3-7, while checkpointing late (i==5) only extends it across rows
// 6-7. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_
// warptile_nostride_snake_interleaved_late_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_interleaved_late(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Isolates whether the interleaved kernels' register/spill increase came
// from splitting load_compute_regs into separate a/b lambdas, or from
// moving the checkpoint into the row loop at all -- interleaved and
// interleaved_late (checkpoints at i==2/4/6 vs i==5/6/7) compiled to
// byte-identical SASS, proving source position was ignored for that
// split entirely. This kernel keeps load_compute_regs combined (one
// lambda, one call site, as in the plain snake kernel) and moves only
// that single call to one mid-loop checkpoint (i==4) instead of before
// the FFMA block, with no split. See mm_cuda_kernels.cu for
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint EXCEPT
// __launch_bounds__(256, 2) is relaxed to __launch_bounds__(256) (no
// min-blocks-per-multiprocessor term), letting ptxas pick whatever
// register count it wants instead of being held under the 128-register
// ceiling (256, 2) implies. Motivated by reading CUTLASS's own source
// (include/cutlass/device_kernel.h): the classic SIMT fp32 device::Gemm
// kernel entry point carries no __launch_bounds__ at all -- its
// 126-register footprint is the compiler's organic choice, not a forced
// ceiling. Three prior restructurings of the prefetch (split vs
// combined, early vs late vs mid checkpoint) all converged to the same
// 128-register/light-spill compile under (256, 2), so this tests
// whether the ceiling itself was the actual cause. See
// mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_warptile_
// nostride_snake_midpoint_uncapped_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_uncapped(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint (same
// combined checkpoint at i==4, same __launch_bounds__(256, 2) --
// deliberately not relaxed, since midpoint_uncapped's relaxation dropped
// occupancy from 2 blocks/SM to 1 and measured dramatically slower
// despite eliminating the spill) EXCEPT tiles_fetched is narrowed from
// size_t to unsigned int. It's a small counter (0..k/BK8, well under a
// few thousand for every tested size) only ever multiplied by BK8 to
// form k_base -- unlike the pointer/index arithmetic elsewhere in the
// kernel, which must stay 64-bit (at N=54000, k*ld_a ~= 2.9 billion,
// over int32's ~2.1 billion range), this value never approaches that
// range. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_
// warptile_nostride_snake_midpoint_narrow_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_narrow(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Byte-for-byte identical to
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint (same
// combined checkpoint at i==4, same __launch_bounds__(256, 2), same
// size_t tiles_fetched -- narrowing it in midpoint_narrow didn't free a
// register and doubled the spill) EXCEPT has_next_kk/has_next_tile are
// no longer precomputed named locals before the row loop; the
// conditions are inlined directly at the i==4 checkpoint. has_next_kk
// folds to a compile-time constant for every unrolled kk anyway; the
// only genuine runtime value is has_next_tile for the kk==0 unrolled
// instance, which previously had to survive as a separately-named live
// value from the top of that kk block to row 4. next3_k0 is already
// alive regardless (reused after the loop), so inlining the comparison
// doesn't keep anything new alive -- it just removes our own manual
// hoisting. See mm_cuda_kernels.cu for gemm_fp32_tiled_syncpipe4_bk8_
// warptile_nostride_snake_midpoint_inline_kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_inline(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// Isolates whether midpoint's own slowdown vs plain snake (measured
// ~8-12% slower) is actually caused by its light spill (2 STL + 2 LDL
// at 128 registers), or by something else about the interleaved
// checkpoint. load_global_advance -- called right at midpoint's i==4
// checkpoint -- normally bounds-checks every tile fetch even though for
// an exact-multiple-of-tile-size problem that check is always true.
// This kernel drops it entirely (unconditional loads, tiles_fetched/
// k_base removed as dead code) right at the pressure point, and is only
// ever dispatched when m/n are exact multiples of the 128 tile and k is
// an exact multiple of BK=8 -- otherwise it falls back to the safe,
// bounds-checked midpoint kernel. See mm_cuda_kernels.cu for
// gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact_
// kernel.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);

// A genuine, un-encumbered fp16 GEMM -- A, B, and C are all plain __half
// arrays, not the FP32_16 feature's packed wh/wr representation (see
// mm_fp32_16_pure_half_tiled in fp32_16_cuda_kernels.cuh for that one).
// Same 128x128/8x8 register-blocked tiling as gemm_fp32_tiled, but __half-
// typed throughout (native hardware fp16 FMA on sm_53+) instead of float.
// Exists to isolate a hand-written kernel's true tensor-core-free ceiling
// against cuBLAS's Hgemm, without the fp32_16 wh-decode cost mixed in.
// NO_TRANSPOSE x NO_TRANSPOSE only -- falls back to gemm_generic_kernel
// for any transposed operand. See mm_cuda_kernels.cu for gemm_fp16_tiled_
// kernel.
void gemm_fp16_tiled(TransposeType transA, TransposeType transB, double alpha,
                     size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                     __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Tensor Core (nvcuda::wmma, 16x16x16 half/half/float) version of gemm_
// fp16_tiled: 128x128 output tile, BK=64, computed by 16 warps (4x4) each
// owning a 32x32 region (2x2 = 4 accumulator fragments/warp, fp32-
// accumulated the same way cuBLAS's own tensor-core kernels do).
// __launch_bounds__ trades a modest amount of register spilling for 2
// resident blocks/SM instead of 1. This exact tile/warp/launch_bounds
// combination is shared with mm_fp32_16_pure_half_wmma (see that kernel's
// comment in fp32_16_cuda_kernels.cuh for the register-pressure/occupancy
// investigation that arrived at it) -- comparing the two isolates how much
// of mm_fp32_16_pure_half_wmma's remaining gap to cuBLAS is the fp32_16
// wh-decode step versus the tile/occupancy architecture itself, since this
// kernel has no decode step at all. NO_TRANSPOSE x NO_TRANSPOSE only --
// falls back to gemm_fp16_tiled for any transposed operand. See mm_cuda_
// kernels.cu for gemm_fp16_wmma_kernel.
void gemm_fp16_wmma(TransposeType transA, TransposeType transB, double alpha,
                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same tile/warp/fragment shape as gemm_fp16_wmma (BK halved from 64 to 32
// to fit double-buffered As/Bs in the 48KB static shared-memory budget),
// but A's global-to-shared load goes through real cp.async (2 rows/4 bytes
// per transfer -- a single __half is below cp.async's 4-byte minimum)
// into the *other* buffer while the current buffer's mma.sync calls run,
// instead of every outer K-tile fully exposing the load's latency behind
// a __syncthreads(). B stays a plain synchronous scalar load (see mm_cuda_
// kernels.cu for why). Requires ld_a and m to both be even; falls back to
// gemm_fp16_wmma otherwise. See mm_cuda_kernels.cu for gemm_fp16_wmma_
// cpasync_kernel and the measurement that motivated it (gemm_fp16_wmma
// measured at or slightly behind the scalar gemm_fp16_tiled despite using
// tensor cores -- a sign of latency being fully exposed, not of mma.sync
// throughput being the bottleneck).
void gemm_fp16_wmma_cpasync(TransposeType transA, TransposeType transB, double alpha,
                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                            __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same as gemm_fp16_wmma_cpasync, but both A's cp.async transfer and B's
// synchronous read are widened from 4 bytes (2 halves) to 16 bytes (8
// halves) -- fewer, better-coalesced transactions on both sides, the same
// "vec2/vec3" lever used throughout this codebase's other GEMM kernel
// families, just not yet applied to this one. Requires ld_a, m, ld_b, and
// k to all be multiples of 8; falls back to gemm_fp16_wmma_cpasync
// otherwise. See mm_cuda_kernels.cu for gemm_fp16_wmma_cpasync_vec_kernel.
void gemm_fp16_wmma_cpasync_vec(TransposeType transA, TransposeType transB, double alpha,
                                size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same as gemm_fp16_wmma_cpasync_vec, but B's load is now genuinely
// asynchronous too (not just vectorized-and-synchronous): its raw bytes
// land via cp.async into a staging buffer matching B's own K-contiguous
// layout, then a cheap synchronous shared-to-shared transpose scatters
// them into the real N-fast Bs compute buffer -- the same stage-then-
// transpose technique mm_fp32_16_tiled_cpasync_dualbuf (fp32_16_cuda_
// kernels.cuh) already uses for its scalar accurate/fast tier, ported here
// on top of tensor cores. MEASURED NEGATIVE RESULT, kept for comparison:
// consistently 15-30% SLOWER than gemm_fp16_wmma_cpasync_vec (e.g. ~11.9
// vs ~14.8 TFlop/s at N=8192 on an A30) -- the added shared-to-shared
// transpose (plus possibly reduced occupancy from 64KB dynamic shared
// memory vs. vec's 48KB static) apparently costs more than the
// synchronous global read it replaces was still costing once THAT read
// was already a single coalesced 16-byte transaction. See mm_cuda_
// kernels.cu for the full measurement and the fp32_16_tiled_cpasync_
// dualbuf2 regression it parallels. Requires ld_a, m, ld_b, and k to all
// be multiples of 8; falls back to gemm_fp16_wmma_cpasync_vec otherwise.
// See mm_cuda_kernels.cu for gemm_fp16_wmma_dualbuf_kernel.
void gemm_fp16_wmma_dualbuf(TransposeType transA, TransposeType transB, double alpha,
                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                            __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same as gemm_fp16_wmma_cpasync_vec (same 8-wide A cp.async, same 8-wide
// synchronous-vectorized B -- unlike gemm_fp16_wmma_dualbuf, this does NOT
// change B's handling), but BK doubles from 32 to 64: halves how often the
// outer loop pays load_tile_async's issue overhead and the two per-tile
// barriers, for the same total compute -- the same lever gemm_fp32_tiled_
// cpasync_vec3 used doubling ITS K-tile (see mm_cuda_kernels.cu). Needs
// 80KB of shared memory (double-buffered As+Bs at BK=64, plus the
// unchanged warp_c stage), so it opts into dynamic shared memory via
// cudaFuncSetAttribute. Requires ld_a, m, ld_b, and k to all be multiples
// of 8; falls back to gemm_fp16_wmma_cpasync_vec otherwise. See mm_cuda_
// kernels.cu for gemm_fp16_wmma_cpasync_vec_bigbk_kernel.
void gemm_fp16_wmma_cpasync_vec_bigbk(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same as gemm_fp16_wmma_cpasync_vec_bigbk, but As/Bs's shared-memory
// storage stride (the "ld" load_matrix_sync uses) is padded from 128
// halves (256 bytes -- exactly 2 full sweeps of the 32 shared-memory
// banks) to 136, a classic wmma/ldmatrix bank-conflict-avoidance lever
// this codebase hadn't tried yet. No logical tile size or compute changes
// -- only where each element physically lives in shared memory. Requires
// ld_a, m, ld_b, and k to all be multiples of 8; falls back to gemm_fp16_
// wmma_cpasync_vec_bigbk otherwise. See mm_cuda_kernels.cu for gemm_fp16_
// wmma_cpasync_vec_bigbk_pad_kernel and its measured result.
void gemm_fp16_wmma_cpasync_vec_bigbk_pad(TransposeType transA, TransposeType transB, double alpha,
                                          size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                          __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same padded layout as gemm_fp16_wmma_cpasync_vec_bigbk_pad, but BK
// doubles again, 64 -> 128 -- testing whether the "wider BK helps" trend
// has more room now that padding removed the bank-conflict ceiling. Needs
// ~152KB of shared memory (fits an A30/A100's ~163KB dynamic-shared-memory
// opt-in ceiling, but leaves no room for a second resident block, unlike
// bigbk_pad's 2); if cudaFuncSetAttribute's opt-in fails on a GPU with a
// smaller ceiling, falls back to gemm_fp16_wmma_cpasync_vec_bigbk_pad.
// Requires ld_a, m, ld_b, and k to all be multiples of 8. See mm_cuda_
// kernels.cu for gemm_fp16_wmma_cpasync_vec_hugebk_kernel and its measured
// result versus bigbk_pad.
void gemm_fp16_wmma_cpasync_vec_hugebk(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// MEASURED NEGATIVE RESULT, kept for comparison: a different lever from
// bigbk/hugebk (which both widened BK on a 2-stage double buffer) -- a
// genuine 3-stage software pipeline at BK=64 (bigbk_pad's own BK), issuing
// tile t+2's load while computing tile t so up to two loads can be in
// flight at once. Consistently landed BETWEEN bigbk_pad and hugebk, never
// beating hugebk (e.g. ~27.7 vs hugebk's ~32.6 TFlop/s at N=8192): BK=64
// means 2x hugebk's outer-loop iteration count for the same total compute,
// and deeper per-iteration overlap didn't make up for paying that
// iteration overhead twice as often. Requires ld_a, m, ld_b, and k to all
// be multiples of 8. See mm_cuda_kernels.cu for gemm_fp16_wmma_cpasync_
// vec_tripipe_kernel and the full measurement.
void gemm_fp16_wmma_cpasync_vec_tripipe(TransposeType transA, TransposeType transB, double alpha,
                                        size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                        __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// MEASURED NEGATIVE RESULT, kept for comparison: warp specialization on
// top of gemm_fp16_wmma_cpasync_vec_hugebk's exact tile/warp/BigBK=128/
// padded-stride shape -- only the first 4 (of 16) warps execute any load-
// related instructions at all, the rest skip straight to compute every
// iteration. Consistently WORSE than hugebk, and even worse than tripipe
// (e.g. ~24.6 vs hugebk's ~32.6 TFlop/s at N=8192): concentrating the same
// total load work into 4 warps instead of 16 means those 4 must issue ~4x
// more cp.async instructions each, serially, extending the load phase's
// wall-clock length by more than the cleaner-instruction-stream benefit
// recovers. Third "beyond hugebk" architectural change this investigation
// tried (after full-async-B and 3-stage pipelining) that lost to hugebk's
// simpler design -- see mm_cuda_kernels.cu for the full measurement and
// what it implies about hugebk being near a practical optimum for the
// nvcuda::wmma API on this tile shape.
void gemm_fp16_wmma_cpasync_vec_warpspec(TransposeType transA, TransposeType transB, double alpha,
                                         size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                         __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same BM/BN/BigBK=128/padded-stride tile and shared memory layout as
// gemm_fp16_wmma_cpasync_vec_hugebk, but 32 warps (1024 threads, the
// hardware max/block) instead of 16, each owning a 16x32 region (1x2 = 2
// fragments) instead of hugebk's 32x32 (2x2 = 4) -- half the per-thread
// accumulator-register footprint (64 registers/thread with zero spilling,
// versus hugebk's same 64-register cap forcing real spills). MIXED
// MEASURED RESULT: a modest ~1.5-2.5% win over hugebk at large N (e.g.
// N=16384), but a ~1-2% loss at N<=4096 -- essentially on par, unlike
// every earlier lever this investigation tried. Kept as an alternative
// (not promoted over hugebk) since it may suit large-N-dominated
// workloads specifically; see mm_cuda_kernels.cu for the full measurement
// and what it implies about hugebk already sitting near this design's
// practical ceiling. Requires ld_a, m, ld_b, and k to all be multiples of
// 8. See mm_cuda_kernels.cu for gemm_fp16_wmma_cpasync_vec_hugebk_w32_
// kernel.
void gemm_fp16_wmma_cpasync_vec_hugebk_w32(TransposeType transA, TransposeType transB, double alpha,
                                           size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                           __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same BM/BN/HugeBK=128/padded-stride tile and cp.async double-buffered
// loading as gemm_fp16_wmma_cpasync_vec_hugebk, but the compute step is
// hand-written mma.sync.aligned.m16n8k16 PTX (fragments loaded via
// hand-written ldmatrix.trans PTX) instead of nvcuda::wmma. Since the
// accumulator is a plain per-thread float register array (not wmma's
// opaque fragment type), the output writes straight from registers to
// global memory -- no warp_c shared-memory staging buffer needed, unlike
// every wmma-based kernel above. MEASURED RESULT: the best kernel in this
// entire family across every tested N, beating hugebk/hugebk_w32
// everywhere (not just small N) once ldmatrix replaced an earlier
// scalar-gather version of this kernel; reaches ~0.39x cuBLAS at N=16384,
// the best ratio of the whole investigation. Requires ld_a, m, ld_b, and
// k to all be multiples of 8. See mm_cuda_kernels.cu for the validated
// per-thread register layout and gemm_fp16_mma_ptx_kernel.
void gemm_fp16_mma_ptx(TransposeType transA, TransposeType transB, double alpha,
                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same ldmatrix.trans/mma.sync compute core as gemm_fp16_mma_ptx, but a
// DOUBLED threadblock output tile (256x128 instead of 128x128, BK dropped
// 128->64 to keep shared memory in budget). Targets global memory
// bandwidth rather than compute/occupancy: every block re-reads its own
// A/B strip from HBM independently, so total HBM traffic scales as
// ~4*N^3/BM -- doubling BM halves that redundant traffic, attacking a
// bottleneck no amount of register/occupancy tuning on the 128x128 tile
// could fix. MEASURED RESULT: the best kernel in this entire investigation
// by a wide margin -- beats gemm_fp16_mma_ptx at every N, with the margin
// growing at larger N as the roofline argument predicts; reaches ~0.52x
// cuBLAS at N=16384 (vs. ~0.39x for the next-best kernel). See
// mm_cuda_kernels.cu for the bandwidth roofline estimate motivating this.
void gemm_fp16_mma_ptx_bigtile(TransposeType transA, TransposeType transB, double alpha,
                               size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                               __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same tile/register/compute core as gemm_fp16_mma_ptx_bigtile -- the
// ONLY difference is which output tile each block computes (a "grouped"
// blockIdx remapping, group_size derived from SM count at runtime), so
// that a wave of concurrently-scheduled blocks shares the same B-column
// strip and can serve repeated B reads from L2 instead of HBM. A
// genuinely different lever than tile size, registers, or pipeline
// depth -- doesn't touch any of those. MEASURED RESULT: a clear, if
// modest, LOSS -- ~9-10% slower than bigtile at large N, and no better
// with a 4x larger group_size either, so this isn't a tuning miss; the
// swizzle itself doesn't help on this hardware/scheduler. Correctness
// passes throughout (including the non-divisible-group_size case). Kept
// for comparison, not promoted; gemm_fp16_mma_ptx_bigtile remains the
// best kernel in this family. See mm_cuda_kernels.cu for the swizzle
// formula (and why the naive version of it would be a correctness bug)
// and the full measurement.
void gemm_fp16_mma_ptx_bigtile_swizzle(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same 256x128/BK=64 tile as gemm_fp16_mma_ptx_bigtile, but restructures
// the compute loop's register lifetimes: B is no longer cached as a
// FragN-wide array (FragN*2 registers alive at once) but loaded one
// fragment at a time, pipelined one wn ahead of the mma calls that
// consume it -- freeing register headroom bigtile's tile-size increase
// didn't use up (accumulator is only 64 of bigtile's 120 registers) and
// spending it on finer-grained shared-memory latency hiding than
// bigtile_tripipe's outer, whole-tile pipelining. MEASURED RESULT:
// neutral -- essentially tied with bigtile at every N, and ptxas reports
// the identical 120 registers/zero spill, so the expected register
// saving didn't materialize either. Combined with bigtile_tripipe's
// earlier neutral result, this is convergent evidence that latency
// hiding isn't the remaining gap to cuBLAS -- bigtile's simple 2-stage
// design was already sufficient. Kept for comparison, not promoted; See
// mm_cuda_kernels.cu for the full reasoning and measurement.
void gemm_fp16_mma_ptx_bigtile_deep(TransposeType transA, TransposeType transB, double alpha,
                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same 256x128/BK=64 bandwidth-optimized tile as gemm_fp16_mma_ptx_
// bigtile, but split across 32 warps (1024 threads, doubling resident
// threads/SM from 512 to 1024) instead of 16 -- tests warp-scheduler
// OCCUPANCY as the remaining bottleneck, a mechanism distinct from
// memory bandwidth (unchanged tile) or latency-hiding-via-buffering
// (already tested twice and ruled out). Uses lean single-buffered B
// loading (see gemm_fp16_mma_ptx_bigtile_deep) since fitting in the
// halved per-thread register budget this warp count implies is a hard,
// not automatic, target. MEASURED RESULT: the best kernel in this entire
// investigation -- fits in 62 registers/thread with zero spilling, and
// beats bigtile at every N with the margin GROWING at larger N (unlike
// every other post-bigtile attempt, which were flat or negative),
// reaching ~0.60x cuBLAS at N=32768. Strong evidence warp-scheduler
// occupancy, not bandwidth or latency-hiding, was the real remaining
// bottleneck. See mm_cuda_kernels.cu for the full reasoning and
// measurement.
void gemm_fp16_mma_ptx_bigtile_w32(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                   __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Identical to gemm_fp16_mma_ptx_bigtile_w32 except the epilogue skips
// the beta*C read entirely when the caller's beta is exactly 0 (checked
// once on the host, against the original double), via a compile-time
// template bool rather than a runtime branch -- a runtime branch version
// pushed registers from 62 to 64 with a small forced spill that actually
// lost to plain bigtile_w32 at large N; splitting into two template
// instantiations (one compiled per branch) restored the exact 62-
// register/zero-spill footprint for both. The other half of CUTLASS's
// epilogue optimization -- vectorized stores -- is NOT applicable here:
// none of a fragment's 4 output values are memory-adjacent given this
// kernel's column-major/row-contiguous C and its register-direct (no
// shared-memory staging) output path, so achieving it would mean
// reintroducing the staging this kernel deliberately avoids. MEASURED
// RESULT: a clean, consistent win at every N once the template fix
// landed -- +3-8% over bigtile_w32, reaching ~0.60x cuBLAS at N=32768,
// the best result of this entire investigation.
// gemm_fp16_mma_ptx_bigtile_w32_epi is the new best kernel in this
// family. See mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_bigtile_w32_epi(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same compute core as gemm_fp16_mma_ptx_bigtile_w32_epi, wrapped in a
// per-column-tile host loop that uses CUDA's explicit L2 persistence API
// to force B's contiguous column-strip to stay resident in L2 across the
// many blocks that read it, rather than the grid-swizzle approach's
// (which lost) indirect hope that scheduling order would provide reuse.
// A's row-strip is NOT persistable this way (not contiguous, given
// column-major storage), capping the max possible benefit at B's share
// of total redundant traffic (~1/3 here). MEASURED RESULT: a clear loss
// at every N (~15% slower than bigtile_w32_epi even at N=32768, where
// per-launch grids are comfortably larger than the SM count) -- the
// overhead of num_col_tiles separate kernel launches and L2 window
// reconfigurations outweighs B's reuse benefit. Also where a real bug
// was caught and fixed: cudaLimitPersistingL2CacheSize is device-wide,
// not per-kernel, and an early version left it permanently set, silently
// degrading unrelated kernels for the rest of the process. Kept for
// comparison, not promoted; gemm_fp16_mma_ptx_bigtile_w32_epi remains the
// best kernel in this family. See mm_cuda_kernels.cu for the full
// reasoning.
void gemm_fp16_mma_ptx_bigtile_w32_epi_l2(TransposeType transA, TransposeType transB, double alpha,
                                          size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                          __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// A resource-budgeted (not ad-hoc-doubled) tile grown ONLY in N (BN:
// 128->160, FragN 4->5), keeping BM/FragM unchanged, exploiting an
// asymmetry bigtile_w32_epi's lean-B design creates: B never holds more
// than 2 live registers regardless of FragN, so growing N adds only
// accumulator registers, not the operand-caching overhead that growing M
// (fully-cached A) would add on top. Grounded in bigtile_w32_epi's own
// measured 62 = 32 accumulator + 30 overhead breakdown, not a fresh
// guess. MEASURED RESULT: the prediction landed almost exactly (64
// registers, only 4 bytes of spill, vs. bigtile2's/bigtile2_w32's
// hundreds) -- and the payoff is real: beats bigtile_w32_epi at every N
// by +9-31%, reaching ~0.61-0.67x cuBLAS at N=16384/32768, the best
// result of this entire investigation.
// gemm_fp16_mma_ptx_bigtile_w32_epi_wide is the new best kernel in this
// family. See mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_bigtile_w32_epi_wide(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                            __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// The next increment in bigtile_w32_epi_wide's direction: BN 160->192,
// FragN 5->6. wide landed at 64 registers/4 bytes spill with essentially
// no headroom left, so this was expected to spill for real, not just a
// token amount. MEASURED RESULT: confirmed -- 172-176 bytes of spill,
// and this time the spill cost wins: never beats wide at any tested N.
// Pins down that BN=160 sits right at (not safely under) this design's
// sweet spot. Kept for comparison, not promoted;
// gemm_fp16_mma_ptx_bigtile_w32_epi_wide remains the best kernel in this
// family. See mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_bigtile_w32_epi_wide2(TransposeType transA, TransposeType transB, double alpha,
                                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                             __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Retries wide2's 256x192 tile with a 16x2 warp arrangement instead of
// 8x4: same accumulator size (48, tile-size-determined), but FragM drops
// to 1 (halving A's cache from 8 to 4 registers) while FragN rises to 12
// (B stays lean regardless, so no cost there). Tests whether that freed
// headroom avoids or shrinks the real spill wide2 hit at this tile size
// with the 8x4 arrangement. See mm_cuda_kernels.cu for the measured
// result.
void gemm_fp16_mma_ptx_bigtile_w32_epi_wide2b(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                              __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// A different trade, grounded in real profiling evidence: nsys (hardware
// counters unavailable) revealed cuBLAS's actual dispatched kernel name,
// ampere_h16816gemm_128x128_ldg8_stages_32x5_nn -- a SMALLER 128x128 tile
// than any tried here, with a 32-wide K-step and a 5-stage cp.async
// pipeline (everything above uses 2-stage double-buffering, plus one
// 3-stage experiment on a bigger tile found neutral). The smaller tile
// leaves far more shared-memory budget per stage, making 5 stages
// affordable here where it wasn't at wide's tile size. Tests the
// opposite trade from every kernel above: less per-block tile reuse, in
// exchange for deeper memory-latency hiding.
//
// MEASURED RESULT (A100): a clear, consistent loss -- ~0.47-0.50x of
// wide's throughput at every N from 2048-32768, plateauing at ~53-56k
// GFlop/s and never scaling further (occupancy-bound: 16 active warps/SM
// vs wide's 32). Combined with wide2/wide2b's loss in the opposite
// direction (bigger tile, same pipeline depth), this brackets
// bigtile_w32_epi_wide as a genuine local optimum from both sides. See
// mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_deep5(TransposeType transA, TransposeType transB, double alpha,
                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                             __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Fills the untested cell in the (tile size) x (pipeline depth) space:
// deep5 matched cuBLAS's tile/BK/stage-count but at only 256 threads (16
// active warps/SM, half of wide's occupancy), so its loss never
// separated "small tile + deep pipeline" from "low occupancy" as the
// cause. Reuses deep5's exact tile/BK/stage-count/shared-memory layout
// but with wide's 8x4/32-warp/1024-thread arrangement -- FragM=1/FragN=4
// drops the accumulator to just 16 registers, leaving ample headroom to
// reach the same 32-warps/SM occupancy wide gets. Directly tests
// whether cuBLAS's tile shape helps once it isn't handicapped by low
// occupancy. See mm_cuda_kernels.cu for the full reasoning and measured
// result.
void gemm_fp16_mma_ptx_deep5_w32(TransposeType transA, TransposeType transB, double alpha,
                                 size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                 __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Targets wide's one remaining ncu-flagged issue: a real register spill
// (~20% estimated speedup), traced via source-correlated ncu profiling
// (enabled by a CMakeLists.txt fix -- a redundant gencode flag was
// silently dropping embedded line tables) directly to the k0-loop
// header: `next_k0` (64-bit) is computed once at the top for the
// load_tile_async boundary check, then needed again at the bottom for
// the pipeline_wait check, so it has to stay live across the entire
// unrolled ks compute loop in between. This collapses it into a single
// explicit `has_next` boolean reused at both sites, testing whether a
// narrower predicate avoids the spill. Byte-for-byte identical to wide
// otherwise. See mm_cuda_kernels.cu for the full reasoning and measured
// result.
void gemm_fp16_mma_ptx_wide_nospill(TransposeType transA, TransposeType transB, double alpha,
                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// nospill's rename didn't move the spill (same CSE regardless of
// syntax); this removes the second use of next_k0 instead. The guard on
// __pipeline_wait_prior(0) at the bottom of the loop isn't needed for
// correctness -- it waits until AT MOST 0 groups are outstanding, which
// is already true if nothing new was committed, so making it
// unconditional costs one harmless no-op wait on the final iteration
// and lets next_k0's live range end at the top of the loop instead of
// spanning the whole unrolled ks compute loop. See mm_cuda_kernels.cu
// for the full reasoning and measured result.
void gemm_fp16_mma_ptx_wide_nospill2(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                     __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Fixes a different confirmed ncu-flagged issue than the B-load
// coalescing saga: uncoalesced global STORES in the epilogue (~34%
// estimated speedup), never fixed since we pivoted to the B-load issue
// instead. Validated against real CUTLASS source
// (epilogue/threadblock/epilogue.h): production kernels stage
// accumulators through shared memory before the global write for
// exactly this reason. Structurally safer than the B-load swizzle
// attempts -- only touches the epilogue, which runs once after the
// K-loop's accumulation is already complete, so there's no cross-K-step
// correctness risk. Reuses As/Bs's now-unused shared memory as the
// staging buffer rather than requesting more. See mm_cuda_kernels.cu
// for the full reasoning and measured result.
void gemm_fp16_mma_ptx_wide_epi_smem(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                     __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Combines nospill2's unconditional __pipeline_wait_prior(0) with
// epi_smem's shared-memory-staged epilogue -- two independent, real (if
// individually small) wins over wide at different N, touching disjoint
// parts of the kernel (K-loop sequencing vs. epilogue), tested together
// to see if they compose. See mm_cuda_kernels.cu for the measured
// result.
void gemm_fp16_mma_ptx_wide_nospill_epismem(TransposeType transA, TransposeType transB, double alpha,
                                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                            __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// wide_nospill_epismem's exact tile/warp/epilogue, but with nospill2's
// shallow 2-buffer pipeline (full-drain every iteration) replaced by a
// deeper cp.async pipeline (bigwarp3's prime-then-partial-wait_prior
// mechanism, never tried on this tile). Real CUTLASS
// (mma_multistage.h) sizes its stage count from the SM shared-memory
// budget, not a fixed constant -- this tile's per-stage footprint is 2x
// deep5-family's, so 3 stages is the deepest that fits the ~163KB/block
// A100 limit (4 would need ~216KB). 3 isn't a power of 2, so the stage
// index is tracked via CUTLASS's actual increment-with-wraparound
// counters (smem_read_stage_idx_/smem_write_stage_idx_), not bigwarp3's
// power-of-2 bitmask workaround.
//
// Measured result: a real, consistent regression vs wide_nospill_epismem
// (-4 to -7% at N=2048-8192) -- this tile is already occupancy-capped at
// 1 block/SM at 2 stages, so the deeper pipeline buys no extra latency
// hiding, only bookkeeping overhead. See mm_cuda_kernels.cu for the full
// reasoning.
void gemm_fp16_mma_ptx_wide_nospill_epismem_deep3(TransposeType transA, TransposeType transB, double alpha,
                                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Fixes ncu's #1 flagged issue on wide_nospill_epismem (uncoalesced B
// loads, 43% est. speedup, re-confirmed by a fresh profile) using the
// REAL CUTLASS mechanism for our REAL data layout -- our B is
// genuinely ColumnMajor (matches cutlass_fp16_gemm_ref.cu's own
// instantiation), which resolves to
// ColumnMajorTensorOpMultiplicandCrosswise<16,32> (kFactor=2), not the
// RowMajorTensorOpMultiplicandCongruous<16,64> (kFactor=1) bigwarp6's
// swizzle used -- confirmed against the actual
// <ColumnMajor,ColumnMajor> DefaultMmaCore specialization and the
// reference binary's SASS (LDGSTS.128 stride pattern), not inferred.
// 4 consecutive threads sweep 4 consecutive 8-element K-chunks for a
// fixed column (K is the physically contiguous dimension for
// ColumnMajor B) -- coalesced on read: the swizzled write target
// (derived from TensorOpMultiplicand's real operator()) is provably
// conflict-free for exactly this grouping, no __shfl_sync needed
// (unlike coalb_shfl) since the thread that reads is the thread that
// writes. Everything else (epilogue, wait_prior(0), 2-stage
// pipeline) is unchanged from wide_nospill_epismem, isolating this
// change to the B load/layout only.
//
// FIXED -- correctness confirmed against the full verify_mm_fp16_gemm
// suite (all sizes 1..2048, beta!=0 and beta=0). Root cause of every
// earlier failed attempt: the B-fragment ldmatrix.x4 was issued with the
// TRANSPOSED PTX variant (ldmatrix.x4.m8n8.trans); CUTLASS's own
// ldsm<RowMajor,4> dispatch (arch/memory_sm75.h) uses the NON-transposed
// ldmatrix.x4.m8n8 for this operand/layout combination, confirmed by
// diffing a hand-written address formula against CUTLASS's actual
// compiled MmaTensorOpMultiplicandTileIterator in an isolated scratchpad
// test (0/1024 mismatches once the instruction was corrected). The
// second K-group additionally needed the per-lane offset XORed by 16
// halfwords against a fixed kk=0 reference base, rather than recomputing
// the swizzle base with kk=16 directly. See mm_cuda_kernels.cu's
// load_b_fragment comment for the full investigation and final fix.
// wide_nospill_epismem is still faster; this fix establishes correctness,
// not a performance win. Original design rationale, still accurate for
// the write side:
//
// Fixes ncu's #1 flagged issue on wide_nospill_epismem (uncoalesced B
// loads, 43% est. speedup, re-confirmed by a fresh profile) using the
// REAL CUTLASS mechanism for our REAL data layout -- our B is
// genuinely ColumnMajor (matches cutlass_fp16_gemm_ref.cu's own
// instantiation), which resolves to
// ColumnMajorTensorOpMultiplicandCrosswise<16,32> (kFactor=2), not the
// RowMajorTensorOpMultiplicandCongruous<16,64> (kFactor=1) bigwarp6's
// swizzle used -- confirmed against the actual
// <ColumnMajor,ColumnMajor> DefaultMmaCore specialization and the
// reference binary's SASS (LDGSTS.128 stride pattern), not inferred.
// 4 consecutive threads sweep 4 consecutive 8-element K-chunks for a
// fixed column (K is the physically contiguous dimension for
// ColumnMajor B) -- coalesced on read: the swizzled write target
// (derived from TensorOpMultiplicand's real operator()) is provably
// conflict-free for exactly this grouping, no __shfl_sync needed
// (unlike coalb_shfl) since the thread that reads is the thread that
// writes. Everything else (epilogue, wait_prior(0), 2-stage
// pipeline) is unchanged from wide_nospill_epismem, isolating this
// change to the B load/layout only.
void gemm_fp16_mma_ptx_wide_nospill_epismem_xswz(TransposeType transA, TransposeType transB, double alpha,
                                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// FIXED -- same correctness fix and root cause as xswz (see that
// kernel's comment above), confirmed the same way. Originally built with
// a different warp grid (WarpRows=16/WarpCols=2 instead of 8/4, FragM=1/
// FragN=10 instead of 2/5, same 1280-element per-warp tile) to enable
// ldmatrix.x4 pairing -- and that pairing turned out to matter after all:
// unlike xswz (odd FragN=5, one orphan wn per warp), this kernel's
// WarpCols=2/FragN=10 makes every wn pair cleanly with no orphan, so
// fusing each x4 fetch's two halves with immediate consumption (see mm_
// cuda_kernels.cu's load_b_pair_raw comment) is a clean, unconditional
// win here: ~83-86% of wide_nospill_epismem's speed at N=2048/4096/8192
// (up from ~65-70%), now consistently beating mma_ptx_wide_nospill_
// epismem_coalb. Still short of wide_nospill_epismem itself.
void gemm_fp16_mma_ptx_wide_nospill_epismem_xswz2(TransposeType transA, TransposeType transB, double alpha,
                                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// CUTLASS's own reference config (cutlass_fp16_gemm_ref.cu): 128x256x32
// threadblock tile, 64x64x32 warp tile, 8 warps/256 threads, 3-stage
// pipeline -- built from scratch to match CUTLASS faithfully (tile,
// warp shape, AND pipeline depth together), unlike xswz2's earlier
// attempt at just copying CUTLASS's warp count onto a different
// threadblock tile (which tanked occupancy and lost). Uses the same
// validated crosswise B-load fix as xswz/xswz2. See mm_cuda_kernels.cu
// for the full derivation and this kernel's own comment.
//
// FOLLOW-UP: B was still being loaded synchronously (blocking global
// read, unpacked in registers, then 8 scalar shared-memory stores) --
// the same issue found and fixed on gemm_fp16_mma_ptx_wide_bigreg_sq,
// back-ported here too. Converting B to cp.async took this kernel from
// ~73726-73736 to 132302.7 GFlop/s at N=8192 (+79%), correct and
// racecheck-clean, despite pushing this kernel's already-tight 128-
// register ceiling into more spilling than before (96/180 and 88/172
// bytes). See mm_cuda_kernels.cu for the full derivation.
void gemm_fp16_mma_ptx_cutlass_config(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// ncu profiling showed CUTLASS's real kernel wins not via more registers
// per se but via a fundamentally different latency-hiding STRATEGY: few
// resident warps (12.49% occupancy, 216 registers/thread, zero spill)
// kept almost never stalled by deep per-warp software pipelining (93.62%
// Compute (SM) Throughput), vs this file's high-occupancy/tiny-register
// "wide" kernels (49.94% occupancy but only 38.70% throughput, 65% of
// L1TEX traffic being register-spill). This kernel is the first attempt
// at CUTLASS's actual strategy -- few warps, full register budget, deep
// pipelining -- applied to this file's own proven-best 256x160 tile
// (re-sliced to BK=32/3-stage): only 8 warps/256 threads (WarpM=64,
// WarpN=80, the closest-to-square split BN=160 allows for 8 warps).
// MEASURED: 255 registers, zero spill initially (this tile's
// accumulator alone is 160 floats/thread vs CUTLASS's own 128, right at
// the ceiling) -- yet only 84095 GFlop/s at N=8192, WORSE than the
// high-occupancy "wide" kernels, with Compute (SM) Throughput of just
// 30.21% (bottlenecked on the Shared pipe, not Tensor) despite matching
// CUTLASS's occupancy exactly. Root cause found via ncu: 3.76x more
// total instructions than real CUTLASS for the same GEMM, mostly from
// recomputing per-element index math (div/mod) from scratch every loop
// iteration instead of precomputing once like CUTLASS's real iterator
// classes do. Fixing that (see mm_cuda_kernels.cu) pushed the kernel
// BACK into real spilling (zero slack left at 255 registers even for
// hoisting-only changes) but still won on net: 88397 GFlop/s (+5%),
// correct and racecheck-clean.
//
// FOLLOW-UP (biggest win by far, discovered on gemm_fp16_mma_ptx_wide_
// bigreg_sq below and back-ported here): B was being loaded
// synchronously (blocking global read, unpacked in registers, then 8
// scalar shared-memory stores) on the mistaken assumption that the
// crosswise permutation layout couldn't accept a single contiguous
// cp.async. It can (see mm_cuda_kernels.cu for the derivation).
// Converting B to cp.async took this kernel from ~88400 to 187435.3
// GFlop/s at N=8192 -- more than double, and above real CUTLASS's own
// ~178500 measurement, DESPITE this kernel still having the "wrong"
// (160-float, not CUTLASS's 128) accumulator size. Registers stayed at
// 255 with real spilling (88/136 and 72/84 bytes for SkipBeta=false/
// true -- this kernel's bigger accumulator doesn't have the headroom
// wide_bigreg_sq's does to reach zero spill after this fix). That the
// win is this large even with spilling confirms the synchronous B load
// was fully serializing the loop regardless of tile shape or register
// budget. Correct, racecheck-clean.
void gemm_fp16_mma_ptx_wide_bigreg(TransposeType transA, TransposeType transB, double alpha,
                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// SASS diffing (cuobjdump -sass) of wide_bigreg's main K-loop against
// real CUTLASS's, isolated via each loop's backward-branch address
// range (both run num_tiles=256 times for N=8192): 822 instructions/80
// HMMA vs CUTLASS's 140 instructions/64 HMMA -- 5.9x more per
// iteration. Two standouts: IMAD 143 vs 1 (CUTLASS precomputes address
// strides once and adds; wide_bigreg recomputes multiplies every
// iteration) and LDL 80, exactly matching HMMA's 80 1:1 (a spill-load
// around literally every MMA call). Root cause: wide_bigreg's
// accumulator is 160 floats/thread (FragM=4*FragN=10*4) vs CUTLASS's
// own 128 (FragM=4*FragN=8*4), 25% bigger, because it kept BN=160 from
// the "wide" family instead of matching CUTLASS's per-warp size. This
// kernel re-tests with BN=128 instead -- WarpN becomes exactly 64
// (matching CUTLASS's own square 64x64 warp tile precisely, not just
// "closest available"), FragN=8, accumulator=128, an EXACT match.
// CONFIRMED: 255/254 registers, 4/0 bytes spill (essentially zero, vs
// wide_bigreg's real spilling at the same ceiling). 107391/114442/
// 111506 GFlop/s at N=4096/8192/16384 (vs wide_bigreg's 87793/88324/
// 81889) -- at N=8192 this is now THE BEST KERNEL IN THE FILE, beating
// gemm_fp16_mma_ptx_wide_nospill_epismem's long-standing 111802.
//
// FOLLOW-UP (biggest single win in this kernel's history): a further
// SASS diff against CUTLASS found its hot loop has ZERO shared-memory
// stores -- both A and B go through cp.async directly from global to
// shared -- while this kernel's B load was still fully synchronous (a
// blocking global read, unpacked in registers, then 8 scalar stores),
// on the mistaken assumption that the crosswise permutation layout
// couldn't accept a single contiguous cp.async. Re-deriving the
// permutation formula showed that assumption was wrong: for a fixed
// output column and a fixed 8-element K-octet, the destination indices
// are 8 CONSECUTIVE, 16-byte-aligned shared-memory slots -- exactly
// what cp.async needs. Converting B to cp.async (matching A and
// matching CUTLASS) took this kernel from 115008 to 220426.6 GFlop/s
// at N=8192 -- nearly double, and ABOVE real CUTLASS's own ~178500
// GFlop/s measurement. Registers dropped too (246/238, still 0/0/0
// spill). Correct across the full size range, racecheck-clean. See
// mm_cuda_kernels.cu for the full derivation.
//
// FOLLOW-UP (null result, kept anyway): same signed-vs-unsigned
// division-by-power-of-two fix applied to gemm_fp16_mma_ptx_cutlass_
// config's load loop (a real win there) was also tried on this
// kernel's once-per-call tid%/tid/ index math. Three repeated same-
// session runs landed at 209371.7/212672.7/210109.1 GFlop/s (avg
// ~210718, tight ~1.5% spread) -- statistically indistinguishable from
// the ~210209 GFlop/s predicted by scaling the confirmed 220426.6
// baseline by this node's own ~0.954x factor (derived from wide_
// bigreg's numbers in the same session). Correct, racecheck-clean;
// kept for consistency with cutlass_config rather than reverted.
//
// INVESTIGATED, root cause found, deliberately not fixed: ncu shows the
// A-load's steady-state LDGSTS instructions (mm_cuda_kernels.cu's
// load_tile_async) with a confirmed, real 1.5x L2 sector overhead vs.
// real CUTLASS's identical-instruction-type access (0% excess there) on
// the same GEMM. Root-caused via an isolated repro (a_load_repro.cu)
// after ruling out the epilogue's C-read, ld_a's power-of-two-ness,
// warp-separation distance, and a from-CUTLASS-source-derived
// alternative thread map (none of these changed the result): a
// byte-identical plain (non-async) vectorized LDG+STS shows ZERO excess
// at any warp count, while the cp.async version shows the 1.5x the
// moment 2+ warps are concurrently active -- isolating the cause to
// LDGSTS.BYPASS specifically. Tried switching the real kernel's A-load
// to the plain form: correct, racecheck-clean, but a confirmed 35-40%
// regression (131647-134696 vs. the confirmed 205000-220427 GFlop/s
// range) -- this kernel is compute/tensor-bound, so cp.async's real
// value is hiding memory latency behind other in-flight work, not the
// byte count moved. Reverted; kept as cp.async deliberately.
//
// FOLLOW-UP: separately chased WHY real CUTLASS avoids this entirely.
// Traced its actual mechanism (cutlass::gemm::device::Gemm swaps A<->B
// and transposes both to RowMajor internally for ColumnMajor-C outputs
// -- explains an earlier "RowMajor" mystery as iterator mislabeling on
// my part), then wired up CUTLASS's REAL PredicatedTileAccessIterator
// class directly in a second standalone repro (a_load_repro_cutlass_
// iterator.cu) -- confirmed it IS clean (0% excess), the only
// configuration found that reproduces CUTLASS's behavior. But neither
// its address formula, its predication, its immediate-offset
// addressing, nor all three combined (tested individually and
// together, SASS matching CUTLASS's opcodes closely) reproduce the
// clean result in our own code. Full instruction diffing suggested the
// cause was real bookkeeping keeping its LDGSTS calls apart (never
// back-to-back, unlike our steady-state loop) -- synthetic attempts to
// force this couldn't even be tested reliably (ptxas's own backend
// scheduler reorders even hand-written atomic inline-PTX blocks), but a
// GENUINE, non-synthetic test settled it: unifying our block-level
// safe/unsafe branches into one always-predicated path (matching
// CUTLASS's actual architecture) produces real, natural interleaving of
// boundary-check instructions between every LDGSTS -- and the excess is
// still completely unchanged. Also tried CUTLASS's batched-bitmask
// predicate computation style (precompute a validity mask once, extract
// bits via shift) instead of a live per-rep comparison -- also no
// change. Then read the ACTUAL predicate class implementation (not just
// its structure): its compute_predicates_() is only ever called once,
// at construction and the first tile transition, never again for the
// rest of the kernel -- unlike every variant above, which recomputes
// fresh every tile. Matched this exactly (mask computed once, before
// the tile loop, never touched again inside it) -- still no change.
// 18 independently-motivated hypotheses tested with hard measurements;
// only the complete, literal CUTLASS iterator class reproduces the
// clean result. This is a real, physical effect (confirmed via the raw
// hardware sector counter, not a theoretical estimate) whose mechanism
// remains unidentified -- plausibly Ampere memory-pipe/L2 request-
// batching behavior for concurrently-scheduled
// warps below what SASS comparison can resolve. Understood in
// exhaustive depth, not fixed -- full writeup in mm_cuda_kernels.cu
// just above load_tile_async.
void gemm_fp16_mma_ptx_wide_bigreg_sq(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq (identical
// signature, drop-in replacement, shares no code path -- the existing
// kernel is untouched by this): same tile, but a 4-stage pipeline
// instead of the shared 3-stage one every kernel on this tile has used
// until now. This exact stage-depth change was already validated as a
// real win (+5.4% to +7.7%) on the closest sibling of this tile,
// mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty, but never tried on
// this kernel itself. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
// kernel's own comment in mm_cuda_kernels.cu, just after gemm_fp16_mma_
// ptx_wide_bigreg_sq_kernel, for the full derivation -- including why
// this deliberately does NOT also copy inline_dirty's other structural
// change (moving the pipeline wait earlier in the loop), which exists
// specifically to overlap decode work this kernel has none of.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                             size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                             double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// signature, drop-in replacement, shares no code path with it -- deep4
// is completely untouched by this): same tile/pipeline/shared-memory
// layout, but load_tile_async's address arithmetic is restructured from
// per-call recomputation to precompute-once/increment, motivated by a
// real cuobjdump --dump-sass comparison against real CUTLASS's own fp16
// tensor-op kernel (same warp/instruction shape, matched register/
// occupancy) showing deep4 issuing 3-4x more address-computation
// instructions (IMAD/LEA) than CUTLASS for identical HMMA/LDSM (actual
// tensor-core work) counts. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
// iter_kernel's own comment in mm_cuda_kernels.cu, just after gemm_
// fp16_mma_ptx_wide_bigreg_sq_deep4's own host wrapper, for the full
// derivation (including the induction proof that the incremental
// rewrite is exactly equivalent to the original per-call recomputation)
// and the SASS evidence that motivated it. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                  double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// mainloop, shares no code path with deep4 or deep4_iter -- neither is
// touched by this): the epilogue (register->Cs->C) is rewritten to move
// 8 halfs (float4, 16 bytes) at a time instead of deep4's own scalar
// per-half stores, motivated by a real cuobjdump/ncu comparison against
// real CUTLASS's own fp16 tensor-op kernel showing large (~25-28%)
// estimated speedup available from uncoalesced global/shared accesses
// that don't line up with the (already-vectorized) mainloop -- CUTLASS's
// own config for this exact tile explicitly uses a 128-bit vectorized
// epilogue. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec_kernel's
// own comment in mm_cuda_kernels.cu for the full derivation, including
// the alignment proof and why the register->Cs "corner turn" itself is
// left scalar (not vectorizable without a much larger restructuring).
// STRICTER fast-path requirements than deep4's own: in addition to
// deep4's checks, also needs ld_c%8==0 and m/n to be exact multiples of
// the tile (no boundary/partial tile support in the vectorized
// epilogue) -- shapes failing this fall back the same way deep4 does.
// UNTESTED: no performance measurement or correctness verification has
// been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a, __half* B,
                                                    size_t ld_b, double beta, __half* C, size_t ld_c);

// Standalone sibling combining BOTH gemm_fp16_mma_ptx_wide_bigreg_sq_
// deep4_iter's mainloop (precompute-once/increment address arithmetic)
// and gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec's epilogue (8-half
// vectorized shared<->global transfers) in one kernel -- shares no code
// path with deep4, deep4_iter, or deep4_epivec, none of which are
// touched by this. Tests whether the two independently-measured fixes
// (deep4_iter: small, real win concentrated at large N; deep4_epivec:
// large win concentrated at small/medium N, since it targets a FIXED
// per-block cost that matters less as mainloop work grows) compose
// additively across the whole N range. See gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_kernel's own comment in mm_cuda_kernels.cu
// for the full derivation. Same stricter fast-path requirements as
// deep4_epivec's own (ld_c%8==0, m/n exact multiples of the tile).
// UNTESTED: no performance measurement or correctness verification has
// been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                         size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                         __half* B, size_t ld_b, double beta, __half* C,
                                                         size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec (shares no code path with it or any other kernel in this
// family): isolates ONE variable from that kernel's own fixes --
// pipeline depth alone, the OTHER direction from stages5 (3 buffers/
// prefetch-distance-2 instead of 4/3). Same generic gemm_fp16_prime_
// wait_n/outstanding-count formulas as stages5's own, so again a pure
// single-knob change. Motivated by direct cuobjdump --dump-sass
// inspection of the real, already-built CUTLASS reference binary
// (build/cutlass_fp16_gemm_ref): its actual MmaMultistage mainloop --
// on a warp tile (64x64x32) and MMA instruction shape (m16n8k16)
// ALREADY IDENTICAL to this family's own -- uses Stages=3, not 4 or 5.
// Combined with deep5's own negative result, this is direct evidence
// CUTLASS's real advantage does NOT come from deeper buffering; if
// anything the opposite. Tests that direction on TOMOS's own proven-
// best 256x128 tile shape (NOT CUTLASS's 128x256 -- gridswap/
// cutlassdims_iter_epivec already showed deviating from this shape's
// own blockIdx.x/N-tile-count axis mapping is a real, large loss at
// large N, so tile geometry is deliberately not touched here).
// UNTESTED: no performance measurement or correctness verification has
// been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(TransposeType transA, TransposeType transB,
                                                                 double alpha, size_t m, size_t k, size_t n,
                                                                 __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                 double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec (shares no code path with it or any other kernel in this
// family): isolates ONE variable from that kernel's own fixes --
// pipeline depth alone, 5 buffers/prefetch-distance-4 instead of 4
// buffers/prefetch-distance-3. Neither gemm_fp16_prime_wait_n(num_tiles,
// stages) nor the mainloop's own outstanding-count wait formula
// hardcodes the stage count -- both are already generic, verified
// algebraically before writing this -- so this is a pure single-knob
// change: no other index arithmetic, epilogue, or fallback logic is
// touched. Motivated by ncu --page source per-instruction stall
// sampling showing DEPBAR.LE (== __pipeline_wait_prior) immediately
// precedes the single hottest-stalling HMMA in deep4_iter_epivec --
// direct evidence the dominant remaining bottleneck is cp.async/DRAM
// latency, not ldmatrix/mma register latency. NOTE: an existing,
// unrelated kernel (gemm_fp16_mma_ptx_wide_bigreg_sq_deep5) already
// uses 5 stages, but it predates deep4_iter/deep4_epivec's own fixes
// and simultaneously shifts prefetch distance in a different way, so
// its own negative results do not cleanly isolate whether stage depth
// alone helps on top of the current best kernel -- this kernel is the
// single-variable version of that question. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(TransposeType transA, TransposeType transB,
                                                                 double alpha, size_t m, size_t k, size_t n,
                                                                 __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                 double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec (shares no code path with it or any other kernel in this
// family): A's own shared-memory tile is re-padded to 272 elements/row
// (32-byte aligned) instead of the shared kGemmFp16WideBigRegBMPad's
// 264 (16-byte aligned only) -- motivated by a real ncu profile showing
// deep4_iter_epivec still has the SAME uncoalesced-access counts as
// deep4's own original profile (25% excessive global sectors, 28%
// excessive shared wavefronts), unchanged by either of that kernel's
// own fixes, tracing to A's shared row-stride not being a multiple of
// 32 bytes. This does NOT touch kGemmFp16WmmaPad (a foundational
// constant used by ~15 other kernel tile configs) -- it defines its own
// local pad. PROVEN TRADE-OFF, not a strict improvement: any pad
// satisfying the 32-byte alignment this kernel wants must double the
// bank-conflict period versus the original's own carefully-chosen pad
// (gcd 8 vs 4) -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec_align_kernel's own comment in mm_cuda_kernels.cu for the full
// algebraic derivation of why no pad value can satisfy both properties
// at once. UNTESTED: no performance measurement or correctness
// verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(TransposeType transA, TransposeType transB,
                                                               double alpha, size_t m, size_t k, size_t n, __half* A,
                                                               size_t ld_a, __half* B, size_t ld_b, double beta,
                                                               __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec: same isolated A-shared-row-stride re-padding technique as
// _align's own, but a DIFFERENT pad value -- 280 elements/row (256+24)
// instead of _align's 272 (256+16). Motivated by direct ncu --page
// source analysis of deep4_iter_epivec's own already-collected profile
// showing A's cp.async store into shared memory is the single hottest
// instruction for BOTH the excessive-global-sectors finding AND (newly
// confirmed) the excessive-shared-wavefronts finding (27.7%) -- one
// root cause, not two. Because __pipeline_memcpy_async's 16-byte-
// aligned-destination requirement forces BMPad to be a multiple of 8,
// only 4 distinct residues mod 32 (the bank period) exist: 256 (no
// padding, already known severely conflicted), 264 (the shared/
// original pad), 272 (_align's own), and 280 -- the one residue with no
// existing real-hardware data point. A standard LDS/STS bank-conflict
// model was tried first but does not reproduce ncu's actually-reported
// conflict degree for cp.async/LDGSTS, so this is a genuinely empirical
// test, not a predicted improvement -- see gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_pad280_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(TransposeType transA, TransposeType transB,
                                                                double alpha, size_t m, size_t k, size_t n, __half* A,
                                                                size_t ld_a, __half* B, size_t ld_b, double beta,
                                                                __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec: replaces A's shared-memory layout entirely with an XOR-
// permuted-chunk swizzle (no linear padding at all), modeled on real
// CUTLASS's own A operand layout (RowMajorTensorOpMultiplicandCrosswise
// <16,32>, confirmed via cuobjdump on build/cutlass_fp16_gemm_ref --
// CUTLASS swizzles A specifically, the opposite assignment from this
// family's own, which swizzles B and only pads A). IMPORTANT: direct
// ncu analysis first confirmed A's own ldmatrix.trans reads are already
// 0.00% excessive (for both A and B) -- the entire excessive-shared-
// wavefronts finding is 100% isolated to A's own cp.async WRITE
// (58.62% excessive there alone), so this targets that specifically,
// not a read-side problem that doesn't exist. A naive port of B's own
// crosswise formula was checked and rejected (wrong contiguity
// direction for A's write -- B needs 8 consecutive K-values contiguous,
// A needs 8 consecutive M-values contiguous). CAVEAT: the standard
// bank-conflict model was checked twice against this specific LDGSTS
// instruction and both predict the CURRENT layout already hits the
// theoretical minimum, contradicting ncu's own reported excess -- so
// unlike this family's other siblings, this one's performance benefit
// is NOT independently provable in advance; it's motivated by
// CUTLASS's real tuned design, not proven analysis. Correctness IS
// independently verified: bijective onto [0,8192) and preserves the
// cp.async 16-byte-contiguous-destination requirement, confirmed via a
// full computational sweep, not assumed -- see gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_aswizzle_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(TransposeType transA, TransposeType transB,
                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                  double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec: replaces aswizzle's own self-invented A-shared-memory scheme
// with real CUTLASS's actual formula, read directly from CUTLASS's own
// header source (tensor_op_multiplicand_sm75.h's TensorOpMultiplicand
// <ElementSize,Crosswise>::operator(), as used via the exact
// DefaultMmaCore<...,layout::ColumnMajor,...,layout::ColumnMajor,...,
// arch::OpClassTensorOp,...> specialization matching this project's own
// cutlass_fp16_gemm_ref.cu config), not recalled from memory or
// reverse-engineered from SASS. CORRECTED UNDERSTANDING vs aswizzle's
// own premise: real CUTLASS does NOT swizzle A -- SmemLayoutA =
// ColumnMajorTensorOpMultiplicandCongruous<16,64>, SmemLayoutB =
// ColumnMajorTensorOpMultiplicandCrosswise<16,32> -- B gets the
// swizzle (already matching this family's own B indexing, hence B's
// own 0.00% excessive wavefronts), A gets the simpler Congruous
// scheme, which still keeps M as the physically-contiguous dimension
// (matching this family's own existing convention, not aswizzle's
// incorrect K-contiguous assumption). Correctness independently
// verified (bijective, preserves cp.async's contiguous-destination
// requirement, read/write agreement) via a full computational sweep --
// see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_
// kernel's own comment in mm_cuda_kernels.cu for the full derivation.
// aswizzle's own ~50% loss does NOT predict this kernel's outcome --
// that kernel implemented a different, wrong-premise scheme; this one
// implements CUTLASS's own real, verified algorithm. UNTESTED: no
// performance measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(TransposeType transA, TransposeType transB,
                                                                    double alpha, size_t m, size_t k, size_t n,
                                                                    __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                    double beta, __half* C, size_t ld_c);

// Combines acongruous's own real-CUTLASS A-formula (unchanged) with
// pipe's own interleaved second-MMA-block structure (unchanged) --
// each wp's B-fragment load for the next tile issued immediately
// before that same wp's own MMA sub-block, instead of batching all 4
// loads first. Motivated by ncu per-instruction stall-reason analysis
// of acongruous's own profile (N=8192): the hottest HMMA instructions
// are now dominated by stall_short_sb (waiting on a just-issued
// ldmatrix's output register) -- the exact dependency pipe's own
// interleaving targets. pipe's own prior test lost -2% to -6.5%, but
// that was measured against the OLD, unfixed baseline where DEPBAR.LE/
// cp.async latency (not this dependency) was still dominant -- since
// fixed by acongruous (L2 sectors excessive 24.8%->0.0%), so this is a
// genuinely different, fairer test of pipe's own hypothesis, not
// expected to repeat the same failure for the same reason.
// Correctness rests on both source kernels' own independently-verified
// correctness composing safely (they touch disjoint parts: A's address
// function vs second-MMA-block B-load timing) -- performance is
// GENUINELY EMPIRICAL, not mechanically proven, exactly as pipe's own
// comment states. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec_acongruous_pipe_kernel's own comment in mm_cuda_kernels.cu for
// the full derivation. UNTESTED: no performance measurement or
// correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(TransposeType transA, TransposeType transB,
                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                        double beta, __half* C, size_t ld_c);

// Identical to acongruous in every respect except a_congruous_idx is
// split into a kk-only precompute (invariant across load_tile_async's
// own 4-rep loop, hoisted to once-per-thread; and across load_a_frag's
// own 4-wm loop, hoisted to once-per-call) and an m-only finish --
// the same address-hoisting principle deep4_iter's own derivation
// already established for this family, applied to acongruous's own
// formula. Motivated by direct comparison of acongruous's own already-
// collected N=2048 and N=8192 profiles: the coalescing fix holds
// identically at both sizes (not size-dependent), but acongruous
// executes 9.6% more dynamic instructions than pad280 at N=2048 -- a
// real, measured per-tile overhead from the formula's own redundant
// per-loop recomputation, which this kernel eliminates via a pure
// algebraic factoring (verified via a full Python sweep over kk in
// [0,32) x m in [0,256): zero mismatches against acongruous's own
// single-pass formula -- ALGEBRAICALLY GUARANTEED correct, not
// empirical like pipe's own change). See gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_acongruous2_kernel's own comment in
// mm_cuda_kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(TransposeType transA, TransposeType transB,
                                                                     double alpha, size_t m, size_t k, size_t n,
                                                                     __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                     double beta, __half* C, size_t ld_c);

// Identical to acongruous (real CUTLASS's A-formula, unchanged) except
// Stages=3 instead of 4 (same prefetch-distance-2/2-tile-priming change
// as deep4_iter_epivec_stages3's own, applied on top of acongruous's
// own A-formula instead of the original row-major+pad one). Motivated
// by deep4_iter_epivec_stages3's own genuine, 5-run-confirmed win in a
// N=7168-12288 sweet spot on the OLD formula -- but acongruous's real-
// CUTLASS write pattern genuinely changes A's own cp.async latency/
// bank-conflict characteristics (LDGSTS's own excessive-wavefronts
// dropped from 58.6% to 25.0%), so the stage-count tradeoff that
// produced that sweet spot is not assumed to transfer unchanged. Same
// simple, cheap, low-risk mechanical technique already validated twice
// this session, not a scheduling gamble like pipe or the hoisting
// attempt (both of which failed on top of acongruous). See gemm_fp16_
// mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3_kernel's
// own comment in mm_cuda_kernels.cu for the full derivation. UNTESTED:
// no performance measurement or correctness verification has been run
// yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k, size_t n,
                                                                           __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                           double beta, __half* C, size_t ld_c);

// Identical to acongruous (same A-formula, same mainloop structure)
// except for how the t+3 prefetch's 6 per-iteration LDGSTS (global->
// shared cp.async) calls are issued. Motivated by directly comparing
// acongruous's own compiled SASS mainloop against real CUTLASS's own
// (both read from already-built binaries via cuobjdump --dump-sass, no
// GPU execution): CUTLASS's own loop issues its 6 LDGSTS ONE AT A TIME,
// each immediately followed by one independent HMMA, whereas
// acongruous's own load_tile_async bursts all 6 (12 SASS instructions)
// before a single LDGDEPBAR and only then starts HMMA -- confirmed NOT
// something ptxas reorders on its own (it DOES freely interleave the
// unrelated load_a_frag/load_b_frags LDSM calls into the same HMMA
// block, but leaves the LDGSTS burst clustered), so CUTLASS's own
// schedule must come from its own C++ template literally interleaving
// iterator.load()/store() calls with individual warp-mma() sub-steps in
// source. This kernel copies that structure: load_tile_async's 6
// individual cp.async calls are split out and issued one per wp
// iteration of the FIRST MMA sub-block (8 independent slots available,
// only 6 needed), instead of bursting them all before load_a_frag/
// load_b_frags. Priming (t=0,1,2) is UNCHANGED -- there is no
// independent MMA work to hide behind at that point. GENUINELY
// EMPIRICAL, same caveat as pipe's and stages3's own siblings:
// restructuring source order is not guaranteed to change ptxas's actual
// schedule, and matching CUTLASS's own instruction order is not
// guaranteed to close the remaining gap. See gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_acongruous_ldgsts_kernel's own comment in
// mm_cuda_kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k, size_t n,
                                                                           __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                           double beta, __half* C, size_t ld_c);

// Identical to acongruous (same A-formula, same mainloop structure, same
// HMMA/LDSM instruction mix) except load_tile_async's boundary-safety
// branch (the partial-last-K-tile `else` path) is deleted entirely
// instead of being a runtime-dispatched branch. This kernel is ONLY
// ever launched when its own host wrapper has separately verified
// k % kGemmFp16WideBigRegBK == 0 (on top of the m/n tile-alignment
// checks acongruous's own wrapper already makes); under that guarantee
// every tile is fully in-bounds, so the boundary code is unreachable
// and can be compiled away. Motivated by directly counting instructions
// in acongruous's own compiled steady-state loop vs real CUTLASS's own
// (both already-built binaries, cuobjdump --dump-sass, no GPU
// execution): despite IDENTICAL core math/register-load work (64 HMMA,
// 16 LDSM per iteration in both), acongruous's own loop totals 474
// instructions per K-tile vs CUTLASS's 140 -- concentrated almost
// entirely in scalar/predicate/branch overhead from safe_a/safe_b being
// runtime bool parameters (94 IMAD vs 1, 59 ISETP vs 6, 28 BRA vs 1, 18
// LDGSTS vs 6), even though for every problem size this benchmark
// actually tests (square, power-of-2 N) that branch is 100% dead code
// at runtime. Compile-only verification (nvcc --ptxas-options=-v plus
// cuobjdump --dump-sass on the resulting object, no GPU execution) on
// this kernel confirms the fix actually lands: 233-236 registers
// (down from acongruous's 254, which sat one below the 255 hardware
// cap) and 290 instructions per K-tile iteration (down from 474; LDGSTS
// now exactly 6, matching CUTLASS's own count, ISETP down to 2). For
// k NOT a multiple of BK, this kernel's own host wrapper falls back to
// calling the existing, already-correctness-proven acongruous kernel
// directly -- not a new boundary-handling path written here. See
// gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_
// kernel's own comment in mm_cuda_kernels.cu for the full derivation.
// UNTESTED: no performance measurement or correctness verification has
// been run yet (register/instruction counts above are compile-only
// static analysis, not a runtime measurement).
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(TransposeType transA, TransposeType transB,
                                                                          double alpha, size_t m, size_t k, size_t n,
                                                                          __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                          double beta, __half* C, size_t ld_c);

// Identical to fastk (same no-boundary-branch guarantee/dispatch rules)
// with ONE further change: A's own addressing is no longer recomputed
// via its full divide/mod/xor chain at every call site -- the entire
// non-buf swizzled offset is precomputed ONCE per thread, before the
// mainloop, for the small, genuinely fixed set of (m,kk) combinations
// actually used (kAReps=4 for load_tile_async's A-loop at the SAME
// thread-constant kk; 2 ks values x FragM=4 wm for load_a_frag).
// Motivated by two PRIOR negative compile-only results (block_row/
// block_col rematerialization, and acongruous2's own kk-only-hoist
// technique reapplied on fastk) that BOTH produced byte-for-byte
// identical compiled SASS to their own baseline -- proving ptxas
// already does that class of intra-call CSE on its own. This tests a
// different hypothesis: hoisting across the ENTIRE outer K-loop (not
// just within one lambda call, and across several opaque asm volatile
// blocks) is a harder invariance proof for ptxas -- confirmed via the
// same compile-only technique to actually change the compiled output
// this time: 275 mainloop instructions/iteration vs fastk's own 290 (a
// further 5.2% reduction; 42% total vs original acongruous's 474), IMAD
// 39->27, SHF 8->2, HMMA/LDSM unchanged at 64/16. Register count is
// mixed (SkipBeta=true drops 233->227; SkipBeta=false, the variant this
// benchmark's own beta=0.3 calls use, stays at 236). See gemm_fp16_
// mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp_
// kernel's own comment in mm_cuda_kernels.cu for the full derivation.
// UNTESTED: no performance measurement or correctness verification has
// been run yet (counts above are compile-only static analysis).
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(TransposeType transA, TransposeType transB,
                                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                  double beta, __half* C, size_t ld_c);

// Identical to fastk_precomp (same no-boundary-branch guarantee, same
// real precomputed-offset table for A) with ONE further change: the
// SAME precomputed-offset technique now also applied to B's own
// crosswise index (gemm_fp16_widebigregsq_crosswise_bs_idx), never
// checked for this before. B's own call sites turn out to be EVEN MORE
// invariant than A's own were: load_b_pair_raw always calls it with
// kk=0 (a compile-time literal, not even 2 variants like A needed), and
// load_tile_async's B-loop uses a thread-constant b_kk0 -- only 4+2=6
// distinct offsets are ever needed for the entire mainloop. Confirmed
// via the same compile-only technique to actually change the compiled
// output: 247 mainloop instructions/iteration vs fastk_precomp's own
// 275 (a further 10.2% reduction; 47.9% total vs original acongruous's
// 474), IMAD 27->14, HMMA/LDSM unchanged at 64/16. See gemm_fp16_mma_
// ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_
// kernel's own comment in mm_cuda_kernels.cu for the full derivation.
// UNTESTED: no performance measurement or correctness verification has
// been run yet (counts above are compile-only static analysis).
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(TransposeType transA, TransposeType transB,
                                                                                   double alpha, size_t m, size_t k, size_t n,
                                                                                   __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                   double beta, __half* C, size_t ld_c);

// Identical to fastk_precomp2 (same no-boundary-branch guarantee, same
// real precomputed-offset tables for A and B) with ONE further change:
// how __pipeline_wait_prior's own `outstanding` argument reaches the
// DEPBAR.LE instruction. Traced algebraically (confirmed by a Python
// simulation over every num_tiles from 1 to 2048): outstanding =
// commits_issued - (t+2) is PROVABLY always exactly 0, 1, or 2 for this
// algorithm's own priming/prefetch structure -- but DEPBAR.LE's operand
// must be a compile-time literal, so the single call site with a
// runtime-valued `outstanding` forced a full defensive dispatch cascade
// (measured: 9 cases, SB0=0x0 through 0x8). A direct __builtin_assume
// hint on that call site was tried FIRST and produced byte-for-byte
// IDENTICAL compiled SASS to fastk_precomp2's own -- the hint doesn't
// survive to whatever pass generates this dispatch. This kernel instead
// splits into three separate call sites, each with a LITERAL argument
// (0, 1, or 2) -- the same mechanism that already worked repeatedly
// this session for A/B's own address tables. Confirmed via the same
// compile-only technique to actually change the compiled output, and by
// far the largest single reduction of this entire investigation: 182
// mainloop instructions/iteration vs fastk_precomp2's own 247 (a
// further 26.3% reduction; 61.6% total vs original acongruous's 474) --
// DEPBAR 9->3, BRA 22->7, UISETP 26->8, PLOP3 14->6. HMMA/LDSM unchanged
// at 64/16. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp3_kernel's own comment in mm_cuda_kernels.cu
// for the full derivation. UNTESTED: no performance measurement or
// correctness verification has been run yet (counts above are
// compile-only static analysis).
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(TransposeType transA, TransposeType transB,
                                                                                   double alpha, size_t m, size_t k, size_t n,
                                                                                   __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                   double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2 (the standing champion of this family -- NOT
// precomp3, whose own outstanding-dispatch change is a real, GPU-
// measured regression) + pipe's own second-MMA-block B-load
// interleaving, ported onto this leaner base. pipe was tested once
// before, on the OLD, overhead-heavy acongruous baseline, and lost
// (-1.3% to -10.3%) -- at the time the kernel was overhead-bound, so
// there was little independent work for this kind of interleaving to
// hide behind. Real ncu profiling on fastk_precomp2 (--set full,
// N=8192) shows the kernel is now in a fundamentally different regime:
// stall_wait 46.3% and stall_math 22.0%, both concentrated on the HMMA
// instructions themselves (not on address computation or branching, all
// of which this session's own prior four changes already drove down to
// match real CUTLASS's own HMMA/LDSM counts exactly, 64/16). This
// retests pipe's own hypothesis in that new, compute/latency-bound
// regime. GENUINELY UNCERTAIN, unlike the four preceding wins in this
// family: this kernel's own compile-only register count (236, no
// spilling) is sane but CANNOT predict the outcome here -- precomp3
// already demonstrated that static counts stop reliably predicting
// runtime performance once address/branch overhead is no longer the
// bottleneck. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp2_pipe_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. This one requires an actual GPU
// run to know either way; no performance measurement or correctness
// verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(TransposeType transA, TransposeType transB,
                                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                        double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2 (the standing champion of this family) with
// ONE change: the warp-level accumulate loop's own ORDER, replaced with
// real CUTLASS's own -- read directly from cutlass/gemm/warp/
// mma_tensor_op.h's own MmaTensorOp::operator() (sm_80's kVerticalVisit
// =false path): a SERPENTINE (boustrophedon) sweep -- for (m) { for (n)
// { n_serp = (m%2) ? (kColumn-1-n) : n; mma(...); } } -- instead of this
// family's own reset-to-0-every-sub-block ordering. Verified via Python
// to be a PURE REORDERING of the same 32 mma.sync calls per MMA block
// (same operands, same accumulator targets, bijective) -- costs zero
// new instructions once unrolled (all indices are compile-time
// constants).
//
// Motivation: this family's own instruction mix already matches real
// CUTLASS's exactly (64 HMMA, 16 LDSM/iteration), yet ncu profiling
// shows stall_wait (46.3%) and stall_math (22.0%) concentrated on HMMA
// itself. The serpentine order targets the hardware operand-reuse cache
// (the `.reuse` SASS annotation): this family's own ordering gets a
// `.reuse`-eligible adjacent-operand match on 24/31 transitions per MMA
// block (all via B); CUTLASS's serpentine gets 31/31 (28 via A, 3 via
// B). CONFIRMED at the actual compiled-SASS level (not just
// algebraically): `.reuse` usage on HMMA jumps from 48/64 (75%) in
// fastk_precomp2's own compiled mainloop to 62/64 (97%) here, same
// total instruction count (247) and HMMA count (64) -- pure reordering.
// Mechanistically different from, and better-evidenced than, the two
// prior "one more try" attempts on this family (precomp3's dispatch
// restructuring and the pipe port), both of which regressed on GPU:
// those targeted general scheduling without a named hardware mechanism;
// this one's targeted effect is confirmed present in the compiled code,
// not just hoped for. Still GENUINELY UNCERTAIN like those two --
// `.reuse` presence is a real, measured proxy for reduced register-file
// read pressure, but not a guarantee stall_wait itself (HMMA's own
// pipeline latency, a related but distinct thing) drops comparably.
// Register count unchanged from fastk_precomp2's own (236/218), as
// expected for a pure reordering. See gemm_fp16_mma_ptx_wide_bigreg_sq_
// deep4_iter_epivec_acongruous_fastk_precomp2_serp_kernel's own comment
// in mm_cuda_kernels.cu for the full derivation. This one requires an
// actual GPU run to know either way; no performance measurement or
// correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                        double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2_serp (the standing champion) with ONE further
// change: how load_a_frag's and load_b_frags' own individual ldmatrix
// calls are POSITIONED relative to the accumulate block that doesn't
// need their result yet. Motivated by real ncu profiling comparing
// this family against real CUTLASS's own (--set full, N=8192): CUTLASS
// shows stall_math 39.8% / stall_wait 34.8% (tensor cores kept busy,
// few dependency stalls); ours showed the OPPOSITE emphasis (21.8% /
// 46.2%) even after HMMA/LDSM counts already matched exactly and
// operand reuse was already maximized. Reading CUTLASS's own compiled
// mainloop instruction-by-instruction shows it does not burst its own
// fragment loads before the block that needs them -- it interleaves
// each individual LDSM with an HMMA from a DIFFERENT, already-ready
// block, one per row. This kernel ports that: kGemmFp16WideBigRegFragM
// (4) equals kGemmFp16WideBigRegSqFragN/2 (4), so one A-load-step and
// one B-load-step are interleaved after each of the 4 wm-rows in BOTH
// accumulate blocks, using the SAME address tables already precomputed
// -- purely a repositioning of WHEN each ldmatrix fires, not a new
// computation. LDGSTS placement is unchanged (still a burst) --
// that specific interleaving already regressed once (acongruous_
// ldgsts), so it's deliberately not combined here.
//
// Confirmed via the same compile-only technique that the interleaving
// survives into the compiled SASS: genuine 1-to-1 LDSM/HMMA alternation
// (matching real CUTLASS's own observed pattern), same .reuse (62/64),
// and mainloop instructions/iteration drop further from serp's own 247
// to 229. This is the most structurally invasive change in this
// family's history -- bigger and more comprehensive than pipe's own
// (which only touched the second block's B-loads and regressed twice)
// or acongruous_ldgsts's own (also regressed) -- but motivated by a
// REAL, quantified profiling gap against real CUTLASS rather than a
// general scheduling hope. GENUINELY UNCERTAIN like several siblings:
// after precomp3's own lesson that large static wins don't guarantee
// GPU wins, this requires an actual GPU run to know either way; the
// correctness check matters more than usual given the scope of this
// change. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp2_serp_ildsm_kernel's own comment in
// mm_cuda_kernels.cu for the full derivation.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(TransposeType transA, TransposeType transB,
                                                                                              double alpha, size_t m, size_t k, size_t n,
                                                                                              __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                              double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec (NOT combined with _align's re-padding -- isolates this
// change's own effect): same mainloop/epilogue, except the second MMA
// block's B-fragment loads for the NEXT tile are interleaved per-wp
// with that same wp's own MMA sub-block (which uses already-resident,
// unrelated registers) instead of issued as one batch before any MMA in
// that block runs. Motivated by a real cuobjdump comparison against
// CUTLASS's own mainloop showing CUTLASS packs its LDSM/HMMA
// instructions far more tightly (consistent with ncu's finding that
// this kernel's dominant stall is still "fixed latency execution
// dependency" while CUTLASS's is "pipe busy"). Deliberately NOT the
// same change gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma_kernel made
// (that one flattened the two-block split and regressed -25% to -36%)
// -- this kernel leaves the two-block split fully intact and only
// reorders within one block. See gemm_fp16_mma_ptx_wide_bigreg_sq_
// deep4_iter_epivec_pipe_kernel's own comment in mm_cuda_kernels.cu for
// the full derivation. GENUINELY UNCERTAIN, not a mechanically-proven
// fix like this family's other siblings -- UNTESTED, no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(TransposeType transA, TransposeType transB,
                                                              double alpha, size_t m, size_t k, size_t n, __half* A,
                                                              size_t ld_a, __half* B, size_t ld_b, double beta,
                                                              __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec: applies the same address-hoisting principle deep4_iter used
// for load_tile_async a second time, to load_a_frag/load_b_frags (the
// ldmatrix fragment-loading lambdas, called twice per K-tile iteration,
// which deep4_iter's own fix never touched) -- hoisting several thread-
// constant/call-invariant quantities (lane-derived indices, per-wm/wp
// row/column base offsets, the 2-valued lane_offset) that were
// previously recomputed via IMAD/shift/mask on every call. A pure
// algebraic rewrite (same proof technique as deep4_iter's own hoist),
// not a speculative reorder like deep4_iter_epivec_pipe's (unsuccessful,
// -2% to -6.5%) attempt. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
// iter2_epivec_kernel's own comment in mm_cuda_kernels.cu for the full
// derivation. UNTESTED: no performance measurement or correctness
// verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                          size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                          __half* B, size_t ld_b, double beta, __half* C,
                                                          size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
// epivec, byte-for-byte identical except which grid axis (blockIdx.x
// vs blockIdx.y) maps to the M-tile vs N-tile index -- a pure grid-axis
// relabeling, touching no per-thread indexing, so it cannot introduce a
// correctness bug the way a per-thread addressing change could.
// Motivated by a real per-instruction ncu breakdown finding 99.4% of
// this kernel's L2-excessive-sector count concentrated in A's own
// global load, with careful per-warp addressing re-derivation finding
// no per-thread bug -- pointing at grid-level L2 residency instead (A's
// data is shared across MORE blocks than B's, yet shows far more DRAM
// excess). GENUINELY UNCERTAIN, not independently verified the way this
// family's provably-equivalent fixes were. See gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_iter_epivec_gridswap_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(TransposeType transA, TransposeType transB,
                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                  double beta, __half* C, size_t ld_c);

// A genuinely new tile geometry (BM=128, BN=256 -- the transpose of
// this whole family's own BM=256, BN=128) matching real CUTLASS's own
// ThreadblockShape<128,256,32> exactly. NOT a patch to an existing
// kernel -- a from-scratch re-derivation carrying forward deep4_iter's
// address-hoisting and deep4_epivec's vectorized-epilogue principles,
// both re-derived for the new dimensions (their exact formulas depended
// on the old BM/BN values). Motivated by a real per-instruction ncu
// breakdown finding 99.4% of this family's L2-excessive-sector count
// concentrated in A's own global load, with per-thread addressing and
// grid-axis-order explanations both ruled out by direct evidence
// (extensive re-derivation found no per-thread bug; gemm_fp16_mma_ptx_
// wide_bigreg_sq_deep4_iter_epivec_gridswap's own negative result ruled
// out simple grid-axis reordering) -- leaving CUTLASS's genuinely
// smaller A-tile (128x32=4096 elements vs this family's own 256x32=
// 8192) as the newly-identified structural difference this kernel
// tests. See gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec_
// kernel's own comment in mm_cuda_kernels.cu for the full, carefully
// verified derivation (accumulator size, shared memory footprint,
// alignment, and epilogue coverage all confirmed identical/correct
// before writing any code). This is the largest, highest-risk change
// in this family -- UNTESTED: no performance measurement or
// correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                            __half* B, size_t ld_b, double beta, __half* C,
                                                            size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_
// epivec: identical tile/warp/MMA shape (CUTLASS's own ThreadblockShape
// <128,256,32>/WarpShape<64,64,32>), but Stages=3 instead of 4 -- direct
// cuobjdump --dump-sass inspection of the real CUTLASS reference binary
// confirms its actual mainloop uses Stages=3, a gap cutlassdims_iter_
// epivec's own config never closed (it kept Stages=4 from the OTHER
// tile family). This is the FULL CUTLASS-matching config, all four
// verified parameters together. CAVEAT: cutlassdims_iter_epivec's own
// real result was a severe, size-growing loss (-24% to -25% at
// N=32768/65536) attributed to wave quantization against this A100's
// 108 SMs -- a grid-size effect independent of stage count, so this
// kernel is NOT expected to fix that regression; it completes the
// CUTLASS-matching comparison for correctness, not because stage count
// was expected to be the missing piece. See gemm_fp16_mma_ptx_wide_
// bigreg_cutlassdims3_iter_epivec_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation. UNTESTED: no performance
// measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                             __half* B, size_t ld_b, double beta, __half* C,
                                                             size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// signature, drop-in replacement, shares no code path with it or with
// gemm_fp16_mma_ptx_wide_bigreg_sq): one pipeline stage deeper still (5
// instead of 4). The analogous 4->5 step regressed on the closest FP32_16
// sibling (mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty), but that
// kernel has a decode step this one doesn't, so the result wasn't assumed
// to transfer -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep5_kernel's own
// comment in mm_cuda_kernels.cu, just after _deep4_kernel, for the full
// reasoning and the measured outcome.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                             size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                             double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// signature, drop-in replacement, shares no code path with it or any
// other kernel in this family): same tile/pipeline, but the B/N tile is
// cut to a quarter, targeting __launch_bounds__(..., 2) instead of
// deep4's (..., 1) -- direct analog of mm_fp32_16_pure_half_wide_bigreg_
// sq_dualocc, motivated by deep4 measuring only 12.49% achieved
// occupancy and a dominant fixed-latency-dependency stall (ncu, --set
// full). See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2_kernel's own
// comment in mm_cuda_kernels.cu, just after _deep4's own wrapper, for
// the full derivation and why dualocc's own register-spill outcome
// wasn't assumed to transfer either way.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                  double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// signature, drop-in replacement, shares no code path with it or any
// other kernel in this family): same tile/pipeline/shared-memory layout,
// but restructures the K-loop to load BOTH K-halves' register fragments
// together right when a tile becomes ready, then issue all 64 mma.sync
// instructions for that tile as one flat block (instead of deep4's own
// two 32-instruction halves split by a mid-tile barrier) -- giving
// ptxas a single, larger pool of independent instructions to hide any
// one ldmatrix's fixed latency behind, targeting the same fixed-latency-
// dependency stall dualocc's occupancy approach targets, via a different
// mechanism. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma_kernel's
// own comment in mm_cuda_kernels.cu for the full derivation, including
// why this is a genuinely different hypothesis from the plain source-
// level loop reorder that gemm_fp16_mma_ptx_wide_bigreg_sq_kernel's own
// header comment (investigation point #14) already found to be futile.
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a, __half* B,
                                                    size_t ld_b, double beta, __half* C, size_t ld_c);

// bfloat16 sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4: same tile,
// same 4-stage pipeline, same load/compute/epilogue structure, with
// __nv_bfloat16 operands (A/B/C) instead of __half and mma.sync's
// operand-type suffix changed accordingly (accumulator stays f32
// either way). First use of cuda_bf16.h in this codebase -- a genuinely
// new operand dtype, added on request, not motivated by a measured
// deficiency. See gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_kernel's own
// comment in mm_cuda_kernels.cu, just after gemm_fp16_mma_ptx_wide_
// bigreg_sq_deep4_bigmma's own wrapper, for the full derivation --
// including why it reuses deep4's own tile/pipeline constants and index-
// math helpers directly despite their "fp16" naming (they're pure
// arithmetic, dtype-agnostic). UNTESTED: unlike this file's other
// kernels, no performance measurement or correctness verification has
// been run yet -- this is new functionality, not a benchmarked result.
void gemm_bf16_mma_ptx_wide_bigreg_sq_deep4(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                             size_t k, size_t n, __nv_bfloat16* A, size_t ld_a, __nv_bfloat16* B,
                                             size_t ld_b, double beta, __nv_bfloat16* C, size_t ld_c);

// bf16 port of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp2_serp -- this codebase's own best fp16
// kernel (vectorized epilogue + swizzled shared-memory layout +
// precomputed offset tables + real CUTLASS's own serpentine mma.sync
// accumulate order), applied to bf16. A straight, mechanical port: every
// tile-geometry constant and index-math helper carries over unchanged
// (same reuse gemm_bf16_mma_ptx_wide_bigreg_sq_deep4 above already
// relies on), the only real changes are __half -> __nv_bfloat16 and the
// mma.sync PTX instruction's own f16.f16 -> bf16.bf16 suffix. Verified
// (compile-only, no GPU execution) to compile to EXACTLY the same
// register counts as its fp16 counterpart, both <SkipBeta> variants
// (236/218) -- strong evidence this is a faithful, correct port, not
// just a syntactically-valid one. Falls back to gemm_bf16_mma_ptx_wide_
// bigreg_sq_deep4 (simpler than fp16's own multi-tier fallback chain,
// which has no bf16 equivalent yet) for any disqualifying condition.
// See mm_cuda_kernels.cu, next to this kernel's own _kernel definition,
// for the full derivation.
//
// MEASURED on real hardware: correct (verify_bf16_wide_bigreg_sq_deep4's
// own correctness sweep, both this kernel's fallback path and its real
// fast path, all OK) and a clear win over gemm_bf16_mma_ptx_wide_bigreg_
// sq_deep4 at every size that reaches the fast path -- 1.17x-1.34x
// across N=2048-65536 (N=1024 a statistical tie, same as fp16's own
// analogous kernel). Confirms fp16's own refinement stack transfers
// directly to bf16. NOT YET wired in as the "native (bf16)" baseline
// benchmarks_gemm_fp16.cu's own comparisons use elsewhere.
void gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __nv_bfloat16* A,
    size_t ld_a, __nv_bfloat16* B, size_t ld_b, double beta, __nv_bfloat16* C, size_t ld_c);

// FP32_BF16 "composite precision" sibling of gemm_bf16_mma_ptx_wide_
// bigreg_sq_deep4: A is now the packed FP32_BF16 wh half (see fp32_bf16_
// pack.hpp) instead of a plain __nv_bfloat16 array -- the bf16 analog of
// this codebase's existing FP32_16 "pure_half" tier (mm_fp32_16_pure_
// half_wide_bigreg_sq_deep4), decoding only wh (never wr). Unlike that
// tier, there is no decode step at all here, not even a cheap one: wh's
// raw bits are always already a valid (truncated) bfloat16 bit pattern,
// unconditionally, for every possible packed value -- no control-bit
// branching the way fp16's narrower exponent range needs. B and C remain
// plain __nv_bfloat16. See mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4_
// kernel's own comment in mm_cuda_kernels.cu for the full derivation.
void mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                  const __nv_bfloat16* B, size_t ld_b, double beta,
                                                  __nv_bfloat16* C, size_t ld_c);

// Directly calls gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp2_serp (this codebase's own best bf16 kernel)
// on wh, REINTERPRETED as __nv_bfloat16*, with no decode kernel, no
// scratch buffer, and no extra launch at all. Unlike fp32_16's own
// materialize strategy (a genuinely separate decode pass, since fp32_
// 16_half_only's decode is conditional), fp32_bf16_bf16_only(wh) is
// UNCONDITIONALLY exactly a bitcast of wh for every possible input, with
// no exceptions (bfloat16's exponent field is bit-for-bit the same
// width and bias as float32's own, so wh -- float32's own top 16 bits,
// truncated -- is already a valid bf16 bit pattern unconditionally; see
// fp32_bf16_pack.hpp's own header comment). So reinterpreting wh as
// bf16 directly, with the same ld_a, is byte-for-byte the array a
// genuine decode pass would produce -- the zero-cost limit of the
// materialize idea for this specific packed format. See mm_cuda_
// kernels.cu, next to this kernel's own definition, for the full
// derivation. UNTESTED ON REAL HARDWARE: no performance measurement or
// correctness verification has been run yet.
void mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(TransposeType transA, TransposeType transB, double alpha,
                                                              size_t m, size_t k, size_t n, const uint16_t* wh,
                                                              size_t ld_a, const __nv_bfloat16* B, size_t ld_b,
                                                              double beta, __nv_bfloat16* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_deep4 (identical
// signature, drop-in replacement, shares no code path with it or any
// other kernel in this family): BK doubled from 32 to 64 (half as many
// outer tile iterations/barrier crossings for the same K range, twice
// the independent mma.sync work between them), forced to 3 pipeline
// stages instead of deep4's 4 by the resulting shared-memory footprint
// (still more total K look-ahead: 3*64=192 vs deep4's 4*32=128). This
// needed real re-derivation, not a mechanical BK swap: the B-operand
// ldmatrix swizzle and its read-offset toggle are both tuned for BK=32's
// specific bit-width, and reusing them as-is for BK=64 would silently
// alias shared-memory addresses (verified via a concrete collision
// trace, not just suspected). See gemm_fp16_mma_ptx_wide_bigreg_sq_
// bk64_kernel's own comment in mm_cuda_kernels.cu for the full
// derivation (both re-derivations proved via bit-decomposition/XOR
// identities, not empirically patched) and the open register-count
// risk from doubling the K-substep buffering.
void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                            size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                            double beta, __half* C, size_t ld_c);

// Standalone sibling of gemm_fp16_mma_ptx_wide_bigreg_sq_bk64 (identical
// signature, drop-in replacement, shares no code path with it or any
// other kernel in this family): same BK=64 tile, same index math, but
// the K-substep register buffering is restructured to stay [2]-deep
// (matching deep4's own footprint) instead of bk64's [4]-deep one, which
// bk64's own comment shows is what caused its 255-register spill and
// consistent loss. Cycles through the same 4 K-substeps per tile using
// only 2 live register slots at a time, via a specific load/compute
// rotation (worked out as a strict operation sequence, not just "buffer
// less") that keeps deep4's own cross-tile stagger for the first
// substep of every tile while still giving every other substep's load a
// full MMA block of unrelated work to hide behind -- deliberately not
// bigmma's flattened "load fresh right before use" mistake. See gemm_
// fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_kernel's own comment in mm_cuda_
// kernels.cu for the full derivation.
void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                 size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                 double beta, __half* C, size_t ld_c);

// Combines bk64_reg2's own register-fixed BK=64/3-stage mainloop
// (identical rotation choreography, identical index math, identical
// shared-memory sizing -- none of that touched) with two fixes from the
// deep4_iter_epivec side of this family: deep4_iter's own address-
// hoisting (load_tile_async's thread-constant global-address components
// computed once outside the lambda, with a running a_k0_ld_a
// accumulator, extended to bk64's own two-band A structure) and deep4_
// epivec's own vectorized (8-half float4) epilogue (verbatim reused,
// doesn't depend on BK). Motivated by bk64_reg2's own already-measured
// result against PLAIN deep4 (not deep4_iter_epivec): ties/slightly
// trails deep4 at N>=8192 (-5.8% to -6.7%) but beats it at N=4096
// (+5.8%), with its own remaining gap attributed to a structural
// transition-count cost, not registers/spill (already fixed, confirmed
// 222/232 regs, 0 spill). Since deep4_iter_epivec's own two fixes are
// additive wins on top of plain deep4, this tests whether applying them
// to bk64_reg2 instead closes enough of its gap to compete with deep4_
// iter_epivec itself -- expected to help most at N=4096, genuinely
// uncertain at large N given bk64_reg2's own remaining gap there is
// structural, not something either fix directly targets. See gemm_
// fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec_kernel's own comment
// in mm_cuda_kernels.cu for the full derivation. UNTESTED: no
// performance measurement or correctness verification has been run yet.
void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                             __half* B, size_t ld_b, double beta, __half* C,
                                                             size_t ld_c);

// FP32_16 "pure_half" variant of gemm_fp16_mma_ptx_wide_bigreg_sq: A is
// the packed FP32_16 representation (see src/core/matrix/fp32_16_pack.hpp),
// decoded via fp32_16_half_only() (the "wh" high half only -- "wr" is
// discarded, so this reconstructs fp16-level accuracy, not fp32). Unlike
// this codebase's other FP32_16 kernels, B and C here are native __half
// (not plain float) -- so this kernel's memory traffic (A/B reads every
// K-tile, C read/write once at the epilogue) exactly matches gemm_fp16_
// mma_ptx_wide_bigreg_sq's own byte-for-byte (wh and __half are both 2
// bytes/element), isolating the cost of A's packed-storage decode alone
// rather than also measuring a different memory footprint. Falls back
// to a naive scalar kernel (see mm_cuda_kernels.cu) for shapes/
// transposes outside this kernel's fast path (NO_TRANSPOSE, ld_a/m/
// ld_b/k all multiples of 8), matching gemm_fp16_mma_ptx_wide_bigreg_
// sq's own fallback pattern. See mm_cuda_kernels.cu, just after gemm_
// fp16_mma_ptx_wide_bigreg_sq, for the full derivation and the type
// table against that kernel.
void mm_fp32_16_pure_half_wide_bigreg_sq(TransposeType transA, TransposeType transB, double alpha,
                                          size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                          const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Sibling of mm_fp32_16_pure_half_wide_bigreg_sq (identical signature,
// drop-in replacement) that adds a per-(block_row, k_tile) precheck:
// tiles of A where every element already has control=0 (its bits are
// already a valid __half value, per fp32_16_pack.hpp) skip the decode
// pass entirely; only tiles containing a real control=1 element pay the
// decode cost. mm_fp32_16_pure_half_wide_bigreg_sq itself is unmodified
// by this -- it still decodes every tile unconditionally. See mm_cuda_
// kernels.cu, just after that function, for the full derivation
// (including why a single whole-array "is everything clean" check was
// tried first and rejected in favor of this per-tile version).
void mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                     size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                     const __half* B, size_t ld_b, double beta, __half* C,
                                                     size_t ld_c);

// Second sibling of mm_fp32_16_pure_half_wide_bigreg_sq (identical
// signature, drop-in replacement): same "skip decode for A data that's
// already valid __half bits" idea as mm_fp32_16_pure_half_wide_bigreg_
// sq_tile_dirty, but the check is made entirely inline, per thread, per
// 8-element chunk, using data already in registers from the same read
// the decode step needs anyway -- no separate precompute kernel, no
// tile_dirty buffer, no cross-thread communication at all. Runs on a
// fully standalone kernel; neither mm_fp32_16_pure_half_wide_bigreg_sq
// nor mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty is modified by or
// shares any code with this. See mm_cuda_kernels.cu, next to mm_fp32_16_
// pure_half_wide_bigreg_sq_inline_dirty_kernel's decode_a_tile_inline_
// dirty, for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                       size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                       const __half* B, size_t ld_b, double beta, __half* C,
                                                       size_t ld_c);

// Fast-decode sibling of mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty
// above (uses fp32_16_half_only_fast instead of fp32_16_half_only) -- added
// as a new, separate function, original untouched. See its own comment in
// mm_cuda_kernels.cu.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, const uint16_t* wh,
                                                            size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                            __half* C, size_t ld_c);

// This family's own counterpart to the native fp16 side's gemm_fp16_
// mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_
// serp: mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty's own decode
// step (per-thread, per-8-element-chunk, control-bit check via fp32_16_
// half_only), plus that native kernel's three optimizations layered on
// top -- real CUTLASS congruous A-addressing, precomputed per-thread/
// per-fragment address tables, and real CUTLASS's own serpentine
// accumulate order. Falls back to mm_fp32_16_pure_half_wide_bigreg_sq_
// inline_dirty itself (already boundary-safe for arbitrary m/n/k) on
// any disqualifying condition, including k not a multiple of the fast
// path's BK tile. See mm_cuda_kernels.cu, next to this kernel's own
// _kernel definition, for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Fast-decode sibling of the declaration above (uses fp32_16_half_only_fast
// instead of fp32_16_half_only, falling back to mm_fp32_16_pure_half_wide_
// bigreg_sq_inline_dirty_fast rather than the accurate original) -- added
// as a new, separate function, original untouched.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2_serp with one change, motivated by real ncu
// profiling: replaces decode_a_tile's per-warp __any_sync (each warp
// independently deciding whether it needs to decode, which was
// measured causing a large stall_barrier gap vs native fp16 -- 15.18%
// vs 2.75% at N=4096) with __syncthreads_or, a block-wide vote+barrier
// combined in one instruction, so every warp takes the same branch in
// lockstep instead of some warps idling at the following __syncthreads
// while others finish decoding. MEASURED: a clean regression vs fastk_
// precomp2_serp at every size (~20-24% slower, N=2048-32768) -- the
// block-wide vote fires far more often than a per-warp one, so it paid
// for both an extra barrier-vote AND nearly the same redundant work it
// was trying to make cheaper. See mm_cuda_kernels.cu, next to this
// kernel's own _kernel definition, for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2_serp with one change, mechanistically
// different from blockvote's own (already-regressed) attempt: instead
// of relocating decode_a_tile's dirty check to a different
// synchronization point, this REMOVES the check entirely -- every
// thread unconditionally decodes all its own chunks every tile. Despite
// compiling to fewer registers than fastk_precomp2_serp's own (236/233
// vs 253/253), MEASURED: an even larger regression than blockvote's
// (~31-36% slower vs blockvote's ~20-24%, N=2048-32768) -- the per-warp
// check being removed was already skipping decode on most tiles in
// practice, and losing that work-avoidance costs more than the
// stall_barrier it removed. See mm_cuda_kernels.cu, next to this
// kernel's own _kernel definition, for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Built on fastk_precomp2_serp with one change, motivated by real ncu
// profiling of alwaysdecode's own regression (see its own comment):
// that profile confirmed fastk_precomp2_serp's per-warp __any_sync
// check was doing valuable work-avoidance (stall_barrier dropped from
// 15.18% to 3.90% when removed) but traded it for elevated stall_
// short_sb (2.37% -> 9.27%) from unconditional shared-memory writes --
// pinpointing that __any_sync's own "whole warp rewrites if any lane is
// dirty" behavior is what inflates that write traffic. This kernel
// drops __any_sync (no cross-lane vote at all) and branches decode_a_
// tile directly on each thread's own dirty bit, shrinking the blast
// radius of one dirty element from "the whole warp rewrites" to "just
// that thread rewrites" -- narrower than blockvote's own (widened to
// block-level, regressed) and alwaysdecode's own (removed, regressed
// worse). Compiles to 253 registers, matching fastk_precomp2_serp's own
// (the idx[] array is unchanged; only the vote instruction is removed).
// See mm_cuda_kernels.cu, next to this kernel's own _kernel definition,
// for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Byte-for-byte identical to perthread above, except decode_a_tile no
// longer stores the whole loaded tile (raw[kAReps]/idx[kAReps], held
// live across the dirty branch in case it's needed again) -- instead it
// uses a single scalar raw/idx that dies each check iteration, and the
// (rare -- 99.99% of elements are clean, see mm_fp32_16_pure_half_gemm_
// materialize_cached_dirty_fastk_precomp2_serp's own comment) dirty
// branch re-reads from shared memory instead of reusing a held copy.
// COMPILE-ONLY CHECK (nvcc -Xptxas -v): the register-pressure hypothesis
// this kernel was built to test does NOT hold -- its own <false> variant
// (the one actually exercised, beta!=0) uses 251 registers, only 2 fewer
// than perthread's own 253, nowhere close to native's 236. Likely
// because #pragma unroll fully unrolls both loops regardless of array-
// vs-scalar source structure, and ptxas's own SSA-based liveness
// analysis on the unrolled result was probably already finding the same
// answer independent of this kernel's own source-level restructuring.
//
// MEASURED: a real, consistent regression vs perthread across the full
// square sweep (-3.1% at N=2048/4096, -7.1% at N=8192, -2.2% at N=32768;
// N=1024/16384 within noise) -- not just "no win" like materialize_
// cached_vec64/dirty's own analogous restructurings, an actual loss at
// 4 of 6 sizes. Both holding the tile (perthread, still the family's
// best fused kernel, still loses to materialize at N>=2048) and not
// holding it (this kernel) underperform materialize -- a structural
// limit, not a register-pressure problem with an available fix. NOT
// wired into gemm_auto_fastk_precomp2_serp. See mm_cuda_kernels.cu,
// next to this kernel's own _kernel definition, for the full derivation,
// register-count comparison, and per-size measured breakdown.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Built on perthread's own real measurement (a statistical tie with
// fastk_precomp2_serp, no regression -- see its own comment), narrowing
// the check's "blast radius" one step further: from per-thread (all
// kAReps chunks bundled into one combined check) to per-rep. MEASURED: a
// mild, consistent regression (~2-4% slower than perthread/fastk_
// precomp2_serp at N=2048/8192/16384/32768) despite fewer registers
// (236/234 vs perthread's own 253/253) -- the extra per-rep branch
// overhead outweighs the further-reduced STS traffic. perthread remains
// the right stopping point on this axis. See mm_cuda_kernels.cu, next to
// this kernel's own _kernel definition, for the full derivation.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// A structurally different attack on the same problem every sibling
// above targets, ported from a strategy already PROVEN to win outright
// in this codebase's own fp32-compute tier (see mm_fp32_16_gemm_
// materialize_midpoint_exact, fp32_16_cuda_kernels.cuh, and its own
// comment: the "decode once, then hand off to the already-proven
// 0-spill native GEMM" strategy). Instead of fusing decode into the
// wide_bigreg_sq tile pipeline (what every sibling above does, in five
// different variations), this decodes wh into a __half scratch buffer
// via a wholly separate, trivial, memory-bandwidth-bound prepass (16
// registers, 0 spills), chunked by row-strips to bound scratch memory,
// then calls gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
// acongruous_fastk_precomp2_serp completely UNMODIFIED as a black box
// -- so the GEMM stage is, by construction, exactly as fast as native
// fp16, not merely close to it. See mm_cuda_kernels.cu, next to this
// kernel's own definition, for the full derivation.
void mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                double alpha, size_t m, size_t k, size_t n,
                                                                const uint16_t* wh, size_t ld_a, const __half* B,
                                                                size_t ld_b, double beta, __half* C, size_t ld_c);

// Identical to the plain materialize kernel above except the scratch
// buffer is a CACHED, process-lifetime allocation instead of a fresh
// cudaMalloc/cudaFree on every call -- motivated by the plain version's
// own measured small-N weakness (cudaMalloc's own tens-to-low-hundreds-
// of-microsecond cost is comparable to or larger than the entire GEMM
// at N=1024/2048). NOT thread-safe for concurrent callers (plain
// static, matching this file's own existing caching convention
// elsewhere) -- see mm_cuda_kernels.cu, next to this kernel's own
// definition, for the full derivation and caveats.
void mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                       double alpha, size_t m, size_t k, size_t n,
                                                                       const uint16_t* wh, size_t ld_a,
                                                                       const __half* B, size_t ld_b, double beta,
                                                                       __half* C, size_t ld_c);

// Same cached-scratch-buffer strategy as the kernel above, plus a
// vectorized (4-rows-per-thread, ushort4/uint2) decode pass ported from
// this codebase's own fp32-compute tier -- targets the remaining
// N=2048-8192 gap to native fp16 (14-18% measured), where the decode
// pass's own O(M*K) execution time (not allocation, already fixed) is
// still a meaningful fraction of the O(M*K*N) GEMM's own time. Falls
// back to the plain-scalar cached kernel when ld_a isn't a multiple of
// 4. See mm_cuda_kernels.cu, next to this kernel's own definition, for
// the full derivation.
void mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, const uint16_t* wh, size_t ld_a,
                                                                           const __half* B, size_t ld_b, double beta,
                                                                           __half* C, size_t ld_c);

// TOMOS FP32/FP16 (bf16 compute) family -- decodes every element of wh
// (taking whichever of fp32_16_split's three encoding branches applies
// per element, see fp32_16_bf16_only's own comment in fp32_16_pack.hpp)
// into __nv_bfloat16 instead of __half, so magnitudes fp32_16_half_only
// would saturate to +-Infinity/flush to zero (anything outside fp16's
// own narrow 5-bit-exponent range) are preserved exactly instead, at
// bf16's own coarser 7-bit-mantissa precision. Same wh-only (no wr)
// convention as the pure_half family above. See mm_cuda_kernels.cu, next
// to this family's own definitions, for the full derivation and the
// always-correct generic fallback (mm_fp32_16_pure_bf16_generic) both
// kernels below share.
//
// UNTESTED: no performance measurement or correctness verification has
// been run yet for either kernel in this family.
void mm_fp32_16_pure_bf16_generic(TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k,
                                   size_t n, const uint16_t* wh, size_t ld_a, const __nv_bfloat16* B, size_t ld_b,
                                   double beta, __nv_bfloat16* C, size_t ld_c);

// Fused decode sibling -- ported from mm_fp32_16_pure_half_wide_bigreg_
// sq_inline_dirty_acongruous_fastk_precomp2_serp, minus the "dirty" bit
// skip (every element genuinely needs conversion work here, since wh's
// raw bits are never themselves a valid bf16 bit pattern), with the
// mma.sync dtype swapped from f16.f16 to bf16.bf16.
void mm_fp32_16_pure_bf16_wide_bigreg_sq_inline_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __nv_bfloat16* B, size_t ld_b, double beta, __nv_bfloat16* C, size_t ld_c);

// Workspace sibling -- decodes batches of rows into a cached bf16
// scratch buffer, then calls the EXISTING gemm_bf16_mma_ptx_wide_bigreg_
// sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp unmodified as a
// black box. Ported from mm_fp32_16_pure_half_gemm_materialize_cached_
// vec_fastk_precomp2_serp.
void mm_fp32_16_pure_bf16_gemm_materialize_cached_vec_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, const uint16_t* wh, size_t ld_a,
                                                                           const __nv_bfloat16* B, size_t ld_b,
                                                                           double beta, __nv_bfloat16* C,
                                                                           size_t ld_c);

// Same as materialize_cached_vec, but overlaps chunk c+2's decode with
// chunk c's GEMM via a second cudaStream_t and two ping-ponged scratch
// buffers, instead of running every chunk's decode-then-GEMM pair back
// to back on one stream. Motivated by real ncu profiling at N=2048 (a
// single-chunk case where no overlap is possible at all): the decode
// kernel's own duration (~14.75us) already accounts for the ENTIRE
// measured gap to native there (~93%, matching measured throughput),
// so overlap can only help where the row-chunk loop genuinely produces
// 2+ chunks (m > kMmFp32_16PureHalfMaterializeChunkRows, N>=8192 in the
// square sweep) -- a structural property of this pipeline shape, not a
// tuned threshold. Falls back to materialize_cached_vec when ld_a isn't
// a multiple of 4. See mm_cuda_kernels.cu, next to this kernel's own
// definition, for the full derivation and the hazard-handling reasoning
// (standard CUDA double-buffered producer/consumer via cudaStreamWait
// Event, verified by hand per-chunk).
//
// MEASURED: correct, but NOT a win anywhere, and a clear -2.4% loss at
// N=8192 (exactly 2 chunks -- the simplest case for the mechanism).
// N=16384/32768 (4/8 chunks) are a wash; single-chunk sizes show a
// small loss from the extra event/wait bookkeeping alone, as expected.
// Likely cause: the native GEMM kernel this calls is __launch_bounds__
// (..., 1) and each chunk's own launch already uses far more blocks
// than this GPU's SM count, so GEMM claims every SM for its full
// duration with no spare occupancy for the decode kernel to actually
// run concurrently on the second stream -- semantically correct double
// buffering with no real SM capacity to exploit on this kernel's
// occupancy profile. NOT wired into gemm_auto_fastk_precomp2_serp --
// see mm_cuda_kernels.cu, next to this kernel's own definition, for the
// full measured breakdown by size.
void mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same as materialize_cached_vec, but the decode kernel writes its 4
// decoded halves as a single 64-bit integer (built via __half_as_ushort
// + explicit shifts/ORs) instead of through a __half decoded[4] array
// reinterpreted as uint2. Motivated by inspecting materialize_chunk_
// vec_kernel's own SASS: its read was a genuine single LDG.E.64 as
// intended, but its write came out as TWO separate 32-bit STG.E
// instructions (the four decoded halves packed correctly into two
// 32-bit registers, but not a consecutive register pair, so the
// compiler couldn't fold them into one STG.E.64). This version's own
// SASS confirmed a single real STG.E.64 instead.
//
// MEASURED: no real effect -- every delta vs materialize_cached_vec
// across the full square sweep (-0.2% to +0.9%) fell within this file's
// own ~1% run-to-run noise band. Confirms the prediction: cutting one
// instruction per thread from an already warp-coalesced store changes
// issue count, not memory transactions or the dominant fixed-launch-
// latency cost, so there was nothing here to actually fix. NOT wired
// into gemm_auto_fastk_precomp2_serp -- kept as a documented, correct-
// but-no-measurable-difference sibling. Falls back to materialize_
// cached_vec when ld_a isn't a multiple of 4. See mm_cuda_kernels.cu,
// next to this kernel's own definition, for the full SASS comparison
// and per-size measured breakdown.
void mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp(TransposeType transA,
                                                                              TransposeType transB, double alpha,
                                                                              size_t m, size_t k, size_t n,
                                                                              const uint16_t* wh, size_t ld_a,
                                                                              const __half* B, size_t ld_b,
                                                                              double beta, __half* C, size_t ld_c);

// Same as materialize_cached_vec64, but hoists fp32_16_half_only's own
// per-element control-bit check to thread granularity first (4
// elements/thread): when all 4 have control=0 ("clean" -- fp32_16_
// half_only(wh) is then exactly a bitcast of wh itself, per that
// function's own control=0 fast path), skips calling it 4 times and
// does one raw ushort4->uint2 bit copy instead. Motivated by a host-
// only measurement (no GPU execution) of this file's own pseudo_rand
// test data: across 200M samples, 99.99355% have control=0. Same idea
// this codebase's own fused inline_dirty family already exploits (its
// own combined-OR-reduce + __any_sync dirty check), applied here to the
// separate materialize decode kernel instead. Does not change total
// memory traffic -- same bytes read/written either path -- so may show
// the same "no measurable effect" result materialize_cached_vec64 did,
// per the N=2048 ncu breakdown showing decode's cost is bandwidth- and
// launch-latency-dominated, not ALU-dominated; built to test that
// against the real data distribution rather than assume it.
//
// MEASURED: correct, but no clear win -- every delta vs materialize_
// cached_vec64 across the full square sweep (-2.3% to +1.3%) fell
// within this file's own ~1-2% run-to-run noise band. Now the THIRD
// distinct decode-kernel-internal optimization tested against this cost
// (after vec_overlap and vec64) with no measurable effect -- confirms
// decode's cost really is just a bandwidth floor plus fixed launch
// latency, unaffected by what happens inside the decode kernel itself.
// NOT wired into gemm_auto_fastk_precomp2_serp -- kept as a documented,
// correct-but-no-measurable-difference sibling. Falls back to
// materialize_cached_vec64 when ld_a isn't a multiple of 4. See
// mm_cuda_kernels.cu, next to this kernel's own definition, for the
// full derivation.
void mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp(TransposeType transA,
                                                                              TransposeType transB, double alpha,
                                                                              size_t m, size_t k, size_t n,
                                                                              const uint16_t* wh, size_t ld_a,
                                                                              const __half* B, size_t ld_b,
                                                                              double beta, __half* C, size_t ld_c);

// Size-threshold dispatcher: calls mm_fp32_16_pure_half_gemm_
// materialize_cached_vec_fastk_precomp2_serp for n >= 2048 (MEASURED
// strictly better than the non-vectorized cached kernel at every size,
// reaching 100.3% of native fp16 at N=32768) and mm_fp32_16_pure_half_
// wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread
// otherwise (the vec kernel still loses at N=1024 specifically, 0.91x
// -- two kernel launches vs one, even with the allocator warm and
// decode vectorized). REVISED TWICE from this function's own original
// version (first N>=8192 against the non-cached materialize kernel,
// then N>=2048 against the cached-but-scalar kernel) as each underlying
// kernel improved -- see mm_cuda_kernels.cu, next to this function's
// own definition, for the full derivation, including an important
// cautionary precedent from this codebase's own fp32-compute tier
// (a similar dispatcher there was dropped after its own crossover
// turned out non-monotonic and GPU-dependent, unlike this one).
void mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp(TransposeType transA, TransposeType transB, double alpha,
                                                         size_t m, size_t k, size_t n, const uint16_t* wh,
                                                         size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                         __half* C, size_t ld_c);

// Precomputes, once, the per-8-row-chunk "dirty" byte array (1 if any of
// those 8 elements of wh has its control bit set, 0 otherwise) that mm_
// fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty needs -- call this
// ONCE after packing A (e.g. right after fp32_16_split); the resulting
// buffer can be reused across as many GEMM calls on that same A as
// needed. chunk_dirty must be sized at least (ld_a/8)*k bytes (~1/16th
// the size of wh itself); ld_a must be a multiple of 8, matching this
// kernel family's existing tensor-core fast-path precondition.
void fp32_16_compute_chunk_dirty(size_t k, size_t ld_a, const uint16_t* wh, unsigned char* chunk_dirty);

// Third sibling of mm_fp32_16_pure_half_wide_bigreg_sq: same "skip
// decode for A data that's already valid __half bits" idea as _tile_
// dirty and _inline_dirty, but the dirty check is precomputed ONCE
// ahead of time (by fp32_16_compute_chunk_dirty above, on the same wh
// this is called with) instead of derived on every call. Profiling
// (ncu, --set full, N=8192) showed mm_fp32_16_pure_half_wide_bigreg_sq_
// inline_dirty's two largest remaining stall contributors versus native
// fp16 were both squarely inside its own inline check: stall_barrier
// elevated 3x over native (7.3% vs 2.4% of sampled stalls), concentrated
// on the LDSM.16.MT88.4 (ldmatrix) immediately after __syncthreads(),
// and stall_short_sb elevated ~21x (2.1% vs 0.1%), concentrated on the
// LOP3.LUT OR-reduce instructions the check itself compiles to. Reading
// a precomputed flag instead removes both of those from this kernel's
// critical path. Unlike _tile_dirty/_inline_dirty, this is NOT a drop-in
// same-signature replacement for mm_fp32_16_pure_half_wide_bigreg_sq --
// it takes an extra chunk_dirty parameter that must be precomputed
// first, so it only pays off when the same packed A is reused across
// calls (or when the one-time precompute cost is otherwise acceptable).
// Runs on a fully standalone kernel; none of the other three methods is
// touched by or shares code with this.
void mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, const uint16_t* wh,
                                                            size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                            __half* C, size_t ld_c,
                                                            const unsigned char* chunk_dirty);

// Fourth sibling of mm_fp32_16_pure_half_wide_bigreg_sq (identical
// signature, drop-in replacement): same load/compute/epilogue as mm_
// fp32_16_pure_half_wide_bigreg_sq_inline_dirty, but only warps 0-3 do
// any decode-check work at all (each covering both its own and warp+4's
// share, via a doubled loop), while warps 4-7 skip decode entirely and
// go straight to their own compute. An experiment in letting the SM's
// warp scheduler interleave decode-warps' and compute-warps' independent
// instruction streams, rather than every warp interleaving decode and
// compute within its own stream -- see mm_cuda_kernels.cu, next to mm_
// fp32_16_pure_half_wide_bigreg_sq_warp_specialized_kernel, for the full
// derivation, including the honest caveat that (unlike this file's other
// changes) there was no confident directional prediction for this one
// going in; it needed to be measured. None of the other three siblings
// is touched by or shares code with this.
void mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized(TransposeType transA, TransposeType transB, double alpha,
                                                           size_t m, size_t k, size_t n, const uint16_t* wh,
                                                           size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                           __half* C, size_t ld_c);

// Fifth sibling of mm_fp32_16_pure_half_wide_bigreg_sq: same idea as
// mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty (needs the same
// chunk_dirty, precomputed via fp32_16_compute_chunk_dirty first -- not
// a drop-in same-signature replacement), but each thread's own needed
// chunk_dirty bytes are prefetched into shared memory via cp.async
// alongside wh itself, instead of read from global memory at check
// time -- fixing the actual reason _precomputed_dirty lost to _inline_
// dirty. See mm_cuda_kernels.cu, next to mm_fp32_16_pure_half_wide_
// bigreg_sq_prefetched_dirty_kernel, for the full derivation. None of
// the other four siblings is touched by or shares code with this.
void mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                           size_t m, size_t k, size_t n, const uint16_t* wh,
                                                           size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                           __half* C, size_t ld_c,
                                                           const unsigned char* chunk_dirty);

// Sixth sibling of mm_fp32_16_pure_half_wide_bigreg_sq (identical
// signature, drop-in replacement): A/K-side load, decode, and compute
// copied verbatim from mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty
// (same tile, same 4-stage pipeline, same decode logic) -- only the B/N
// tile is cut to a quarter (128 -> 64), targeting __launch_bounds__(...,
// 2) instead of inline_dirty's (...,1). Motivated by ncu measuring
// inline_dirty at only 12.5% achieved occupancy with BOTH the register
// and shared-memory occupancy limiters at 1 block/SM: with only 8
// resident warps and one shared block-wide barrier, warp-to-warp
// variance in decode completion time has no other warp for the SM to
// schedule around it. See mm_cuda_kernels.cu, next to mm_fp32_16_pure_
// half_wide_bigreg_sq_dualocc_kernel, for the full derivation, including
// a directly relevant cautionary precedent already in this file
// (gemm_fp16_mma_ptx_deep5_kernel tried a related occupancy-for-tile-
// size trade and measured a clear loss) and why this kernel's case
// differs enough to still be worth measuring on its own. None of the
// other five siblings is touched by or shares code with this.
void mm_fp32_16_pure_half_wide_bigreg_sq_dualocc(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                  const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// gemm_fp16_mma_ptx_cutlass_config's 128x256x32 tile actually LOSES to
// gemm_fp16_mma_ptx_bigtile_w32_epi_wide's 256x160x64 tile on this GPU
// (35684 vs 54106 GFlop/s at N=8192) -- so instead of CUTLASS's literal
// tile shape, this ports cutlass_config's validated techniques (3-stage
// cp.async multistage, register-level ldmatrix double buffering across
// the tile-boundary __syncthreads()) onto wide_epi_wide's own tile/warp
// geometry, with BK shrunk 64->32 to make 3 stages affordable in shared
// memory. See mm_cuda_kernels.cu for the full derivation.
void gemm_fp16_mma_ptx_wide_multistage(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// A genuinely unexplored design point, not a variant of anything tried
// so far: deep5's exact tile/BK/stage-count (128x128, BK=32, 5 stages --
// matching cuBLAS's real dispatched kernel name from nsys), but with
// only a 2x2 WARP grid (64x64 per warp, 4 warps/128 threads total)
// instead of favoring occupancy -- matching the tile:warp ratio in real
// CUTLASS source (examples/14_ampere_tf32_tensorop_gemm, adapted from
// TF32 to FP16 instruction shape). FragM=4/FragN=8 gives a much bigger
// per-warp accumulator (~128 registers alone) than anything tried
// before, deliberately trading occupancy for per-warp compute
// granularity. See mm_cuda_kernels.cu for the full reasoning and
// measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp(TransposeType transA, TransposeType transB, double alpha,
                                     size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                     __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// bigwarp matched CUTLASS's real tile:warp ratio but still lost
// decisively -- tile dimensions alone don't transfer without matching
// the actual scheduling. CUTLASS's real mainloop (read directly, not
// summarized) double-buffers warp fragments at the REGISTER level for
// BOTH A and B, prefetching the next ks-step's fragments while computing
// the current ones. Our kernel already does this for B (lean-B); A is
// loaded fresh, all at once, every ks-step -- no overlap. This adds the
// same one-ahead register-level prefetch to A, doubling its register
// cost (easily affordable given bigwarp's 200-register headroom), to
// test whether this fine-grained pipelining is what actually lets
// CUTLASS succeed at low occupancy. See mm_cuda_kernels.cu for the full
// reasoning and measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp_apipe(TransposeType transA, TransposeType transB, double alpha,
                                           size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                           __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Real CUTLASS reference numbers (an unremarkable, un-tuned example --
// 128x256x32 threadblock, 64x64x32 warp tile, 3 stages -- nearly
// matching cuBLAS) reframe this family: that config's tile/warp shape
// and low-occupancy regime are almost identical to bigwarp's own, yet
// bigwarp lost decisively while CUTLASS roughly doubles tomos's best
// kernel. bigwarp_apipe's one added piece (A-side register double
// buffering) barely moved the needle, meaning it was still an
// incomplete port of CUTLASS's PipeState, which double-buffers BOTH A
// and B warp fragments at the register level. This finishes the port:
// B fragments are now loaded all at once per ks-step into the same
// kind of 2-slot ping-ponged register array as A, replacing the old
// within-ks-step lean single-fragment-ahead B prefetch. bigwarp_apipe
// measured 200 registers/thread at only 128 threads/block (512
// available) -- 300+ registers of unused headroom, the opposite
// situation from wide, where every register-costly CUTLASS technique
// tried so far lost specifically because wide has none to give.
// Deliberately does not also add wide_spread's spread-cp.async-issuance
// technique here: bigwarp already runs a genuine 5-stage ring buffer
// with far more load/compute overlap slack than wide's shallow 2-stage
// buffer had, so keeping this test isolated to the one CUTLASS
// mechanism bigwarp_apipe was actually missing. See mm_cuda_kernels.cu
// for the measured register count (not benchmarked in this build, per
// request).
void gemm_fp16_mma_ptx_deep5_bigwarp_fullpipe(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                              __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Direct ncu comparison against the real CUTLASS reference at N=16384
// found the actual gap: bigwarp_fullpipe runs at just 6.25% occupancy
// (4 warps/SM) and 15.66% Compute (SM) Throughput, stalls dominated by
// L1TEX-scoreboard and fixed-latency waits -- while CUTLASS's own
// kernel for this problem runs at 12.5% occupancy (8 warps/SM) and
// 94.94% Compute (SM) Throughput, stalls dominated by an oversubscribed
// math pipeline (i.e. genuinely compute-bound). Both hit "Block Limit
// Shared Mem: 1"; CUTLASS just packs twice the warps into that one
// block. bigwarp's actual tile (128x128, 2x2 warp grid) was never quite
// CUTLASS's (128x256, 2x4 warp grid) -- same 64x64 per-warp tile and
// near-identical register count (200 vs CUTLASS's 216), just half the
// warp count from never doubling the N dimension. This isolates exactly
// that one variable: BN doubles 128->256, warp grid widens 2x2->2x4,
// FragM=4/FragN=8 (still 64x64) unchanged, so accumulator cost
// shouldn't change. Deliberately does not also fix the epilogue's
// uncoalesced writes or reduce the 5-stage pipeline yet (both real,
// ncu-flagged differences from CUTLASS too) -- isolating the occupancy
// variable first. See mm_cuda_kernels.cu for the measured register
// count and occupancy (not benchmarked in this build, per request).
void gemm_fp16_mma_ptx_deep5_bigwarp2(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// bigwarp2 doubled occupancy to match CUTLASS's real 12.5% exactly
// (+49-57% real throughput at N>=4096) but still left a ~3x gap to
// CUTLASS untouched. bigwarp2 still has the ORIGINAL direct, uncoalesced
// epilogue -- never fixed for this family, unlike wide. CUTLASS's own
// profile showed no uncoalesced-global-access issue at all; ours showed
// 45% estimated speedup available from exactly that. This applies the
// same fix already validated on wide (wide_epi_smem/nospill_epismem):
// stage the epilogue through shared memory (reusing As/Bs's now-unused
// space), then a separate gather+write pass with a coalescing-friendly
// global write assignment. Costs no extra registers -- only touches
// code after the K-loop's accumulation is already complete. See
// mm_cuda_kernels.cu for the measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp2_epismem(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                              __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Direct SASS comparison (cuobjdump) against the real CUTLASS reference
// found ISETP+IMAD (address arithmetic and bounds-check predicates) at
// 37.7% of our instructions vs CUTLASS's 11.9%, despite near-identical
// total instruction and HMMA counts. load_a_fragments/load_b_fragments/
// load_tile_async recompute lane- and wm/wn-derived address components
// from scratch on every call, even though they're invariant across the
// whole kernel invocation for a given thread -- only k0/buf/ks actually
// change per call. This hoists that invariant arithmetic into a
// one-time precomputation before the main loop (per-warp m_base[]/
// n_base[] arrays, per-lane offset scalars, per-thread tile-loader
// index arrays), leaving only the k0/ks-dependent address part and the
// REQUIRED boundary checks (row<m, kg<k, col<n stay fully dynamic --
// correctness for non-tile-multiple m/n/k is unaffected) in the hot
// path. See mm_cuda_kernels.cu for the measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp2_hoisted(TransposeType transA, TransposeType transB, double alpha,
                                              size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                              __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Direct SASS inspection of bigwarp2_hoisted's compiled K-loop found a
// division-by-5: the 5-stage ring-buffer index (t % kGemmFp16DeepStages)
// is a 64-bit modulo by a non-power-of-2, which ptxas compiles to a
// multi-instruction reciprocal-multiplication sequence (the classic
// 0xCCCCCCCD magic constant) instead of a cheap AND. deep5's 5 stages
// were chosen to match cuBLAS's real dispatched kernel name from an
// nsys trace -- legitimate, but nobody had checked the instruction cost
// of a non-power-of-2 stage count. This switches to 4 stages (new
// dedicated constants -- kGemmFp16DeepStages is shared by other deep5-
// family kernels) with the ring-buffer index computed via bitmask
// (t & 3) instead of modulo. See mm_cuda_kernels.cu for the measured
// result.
void gemm_fp16_mma_ptx_deep5_bigwarp3(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// bigwarp3's stage-count fix cut the K-loop from 317 to 297 instructions
// for the same 64 HMMA, but still far from CUTLASS's 140. The remaining
// IMAD instructions are dominated by `col*ld_b` and `kg*ld_a` --
// address multiplication against RUNTIME kernel parameters, recomputed
// from scratch on every load. `col*ld_b` is thread-invariant and fully
// hoistable; `kg*ld_a` decomposes into a per-slot-invariant `kk*ld_a`
// plus a `k0*ld_a` term that increases by a fixed stride every
// iteration -- CUTLASS's own iterator pattern (precompute a stride
// once, advance via pure addition, never re-multiply by a runtime
// stride in steady state). This hoists both and replaces the
// recomputed `k0*ld_a` with a running accumulator. Boundary checks
// (row<m, kg<k, col<n) stay fully dynamic. See mm_cuda_kernels.cu for
// the measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp4(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// bigwarp3/4's instruction-count fixes only bought ~1% real throughput
// despite a ~2x loop-body instruction-efficiency gap vs CUTLASS -- this
// design is no longer instruction-COUNT-bound. CUTLASS hits 94.9%
// Compute (SM) Throughput at the same 12.5% occupancy we now match,
// meaning its SCHEDULING (not just instruction count) exposes far more
// per-warp ILP. This is the first attempt to faithfully replicate
// CUTLASS's actual mac_loop_iter() structure (read verbatim from
// mma_multistage.h) on two fronts: (1) spreading next-tile cp.async
// issuance across the ks-loop's 2 steps instead of one monolithic burst
// (wide_spread tried this and lost on wide's zero register slack;
// bigwarp has substantial headroom). A second mechanism -- cross-t-
// iteration fragment pipelining -- was also attempted (issuing t+1's
// ks=0 fragment load during t's last ks-step) but REVERTED: compute-
// sanitizer --tool racecheck caught a real data race (that buffer
// slot's cp.async write isn't guaranteed complete until iteration t's
// OWN wait_prior+syncthreads, one round later than the read). Any
// throughput gain previously attributed to that idea was an artifact of
// the race's undefined timing. Only the spread-issuance mechanism is
// actually in effect. See mm_cuda_kernels.cu for the measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp5(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// bigwarp5's ncu profile confirmed the largest remaining bottleneck is
// the same uncoalesced-B-load issue seen throughout this investigation
// (44.81% estimated speedup, top stall reason). coalb_shfl's register
// shuffle isn't safely affordable here (bigwarp5 is at 255/256
// registers). This implements what CUTLASS actually does instead,
// read directly from cutlass/layout/tensor_op_multiplicand_sm75.h: for
// our exact reference config, B's shared memory layout is
// RowMajorTensorOpMultiplicandCongruous<16,64> (confirmed from the
// compiled kernel's mangled name) -- a compile-time XOR ADDRESS
// SWIZZLE, not a runtime shuffle: the write/read address is permuted by
// XORing partition indices derived from (k mod 8, n mod 64), so the
// same coalesced-global thread mapping also lands bank-conflict-free
// for ldmatrix, at zero extra registers/instructions for data movement.
// Combined with coalb's coalesced global-read thread mapping. The exact
// bit-pattern doesn't need to match CUTLASS's own choice to be correct
// -- only identical between write and read sites for a given (k,n),
// which is guaranteed by using one shared swizzle function at both;
// XOR partition swizzles are self-inverse bijections by construction,
// so a mismatch from CUTLASS's own bit choices would only cost some of
// the intended bank-conflict avoidance (measurable via ncu), not
// correctness. See mm_cuda_kernels.cu for the measured result.
void gemm_fp16_mma_ptx_deep5_bigwarp6(TransposeType transA, TransposeType transB, double alpha,
                                      size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                      __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Grounded in CUTLASS's ACTUAL mainloop source (mma_multistage.h, read
// verbatim via curl, not an AI-summarized paraphrase). Their
// mac_loop_iter() does not issue the next tile's cp.async as one upfront
// burst: it spreads issuance across the ks-loop via
// copy_tiles_and_advance() with a growing group_start offset at every
// warp_mma_k step. wide's load_tile_async issues the entire next tile's
// copies in one burst before the ks-loop starts. This splits the same
// total copy work into 4 groups (matching the ks-loop's 4 iterations)
// and issues one group per ks-step, needing no new registers -- a clean
// test of a genuinely new mechanism rather than another tile/occupancy
// tradeoff.
//
// MEASURED RESULT (A100): register count unchanged (64), but spill
// jumps from wide's 4/4 bytes to 136/196 bytes -- interleaving copy-
// issuance with each ks-step's compute keeps more live values around at
// once. Throughput lands ~30% below wide_nospill2/epi_smem/
// nospill_epismem at every N tested, despite identical occupancy to
// wide -- isolating the spill cost cleanly from any occupancy effect.
// CUTLASS affords this schedule with a register budget built around it
// from the start; wide's tight lean-B design can't absorb it without
// spilling. wide_nospill_epismem remains the best kernel from this
// investigation. See mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_wide_spread(TransposeType transA, TransposeType transB, double alpha,
                                   size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                   __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Combines three independently-motivated, orthogonal fixes never tested
// together: nospill2's unconditional __pipeline_wait_prior(0), epi_smem's
// shared-memory-staged coalesced epilogue, and coalb_shfl's coalesced
// B-tile global load (read coalesced, then __shfl_sync back into wide's
// original bank-conflict-free shared-memory write assignment). ncu on
// wide_nospill_epismem (our prior best) flagged uncoalesced B loads as
// the single largest remaining issue (43% estimated speedup) -- but
// wide_coalb/wide_coalb_shfl were only ever benchmarked standalone, on
// top of wide's OLD uncoalesced-store epilogue, so they never got a fair
// test alongside epi_smem's fix. See mm_cuda_kernels.cu for the measured
// result of this three-way combination.
void gemm_fp16_mma_ptx_wide_nospill_epismem_coalb(TransposeType transA, TransposeType transB, double alpha,
                                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Every kernel in this family has been stuck at exactly 50% occupancy
// (1 block/SM) because 1024 threads x >=62 registers/thread already
// saturates the A100's entire register file for one block -- nothing
// here has ever actually run 2 blocks/SM. Getting there needs <=32
// registers/thread; the accumulator alone (FragM*FragN*4) is a hard
// floor, and bigtile_w32's own FragM=2/FragN=4 already costs 32 on the
// accumulator before any overhead. This uses the smallest non-degenerate
// step, FragM=1/FragN=1 (BM=128, BN=32 with the existing 8x4 warp grid)
// -- a ~10x smaller tile than wide's, implying ~3.8x more global memory
// traffic per the 2*N^3/BM + 2*N^3/BN roofline model. Deliberately the
// opposite gamble of every previous tile-size experiment (those all
// shrank occupancy to grow the tile; this shrinks the tile to grow
// occupancy). Reuses nospill2's unconditional wait and epi_smem's
// coalesced epilogue (free wins), skips coalb's B-load shuffle fix
// (already shown to net-lose from its own register pressure). See
// mm_cuda_kernels.cu for the measured register count and whether ncu
// confirms 2 resident blocks/SM.
void gemm_fp16_mma_ptx_smalltile_2block(TransposeType transA, TransposeType transB, double alpha,
                                        size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                        __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Isolates the ONE variable deep5 conflated with two others: this uses
// wide's exact tile (256x160), exact warp arrangement (32 warps, full
// 1024-thread occupancy), exact lean-B/SkipBeta epilogue -- only BK
// shrinks 64->32 and the pipeline deepens 2->5 stages (matching cuBLAS's
// actual config). Register cost is unaffected by BK/stage count (only
// shared-memory buffer size), so this should compile to essentially
// wide's own register footprint, cleanly testing whether deeper
// latency-hiding helps when occupancy isn't sacrificed to afford it.
//
// MEASURED RESULT (A100): does not improve on wide, even with the
// occupancy/tile-size confound fully removed (identical 64 registers/4
// bytes spill). The 2-stage double-buffer was already sufficient;
// closes off pipeline depth as a lever for this design entirely. See
// mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_wide_deep5(TransposeType transA, TransposeType transB, double alpha,
                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Targets the single largest issue real ncu profiling found: uncoalesced
// global loads on B's tile load (~40-44% estimated speedup). wide has
// consecutive threads reading consecutive COLUMNS of column-major B,
// which are ld_b elements apart -- no coalescing across the warp. This
// swaps the mapping so consecutive threads instead cover consecutive
// K-offsets within the SAME column (contiguous in memory), moving to the
// next column only every BK/8 threads. Everything else (tile size, warp
// arrangement, register footprint) is identical to wide, isolating just
// this access-pattern change. See mm_cuda_kernels.cu for the full
// reasoning and measured result.
void gemm_fp16_mma_ptx_wide_coalb(TransposeType transA, TransposeType transB, double alpha,
                                  size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                  __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// ATTEMPTED AND ABANDONED (gemm_fp16_mma_ptx_wide_coalb_pad): tried to
// fix coalb's shared-memory bank conflict by padding BNPad to an odd
// multiple-of-8 offset (169 instead of 168). Mathematically impossible:
// ldmatrix.sync.aligned requires BNPad to stay a multiple of 8 for every
// row's alignment, but any multiple-of-8 BNPad makes 8*BNPad a multiple
// of 64 -- exactly the quantity causing the bank collision in the first
// place. Confirmed via compute-sanitizer: a real misaligned __shared__
// read inside ldmatrix, which corrupted the CUDA context for the rest of
// the process. A genuine fix needs an address swizzle applied
// identically on both the store and ldmatrix-read sides, not a constant
// pad. Removed rather than left broken. See mm_cuda_kernels.cu for the
// full reasoning; gemm_fp16_mma_ptx_bigtile_w32_epi_wide remains the
// best kernel in this family.

// Sidesteps the structural dead end above instead of solving it within
// that framework: a kk-varying address transform (needed to actually
// fix the write conflict) is incompatible with mma.sync's accumulation
// (same output register sums contributions across all k-steps, so a
// per-k-step column relabeling would silently corrupt which logical
// column each partial sum belongs to), while a kk-invariant one can't
// change the conflicting addresses' relative spacing. This decouples
// "which thread reads which global bytes" from "which thread writes
// which shared address" using a WARP-LOCAL __shfl_sync instead of an
// address transform: read coalesced (coalb's mapping), then shuffle
// each lane's 16-byte read to the lane that owns wide's original,
// already-validated, bank-conflict-free write assignment. The shared
// memory layout and the ldmatrix read side are byte-for-byte unchanged
// from wide -- this is a pure decoupling, not a new layout, so it
// carries none of the padding/swizzle attempts' correctness risk. See
// mm_cuda_kernels.cu for the full reasoning and measured result.
void gemm_fp16_mma_ptx_wide_coalb_shfl(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Tests whether bigtile_w32's occupancy win composes with a bigger tile:
// bigtile2's 256x256 shape at 32 warps instead of the 16 that made it
// spill catastrophically. The register math is unfavorable and stated
// plainly in mm_cuda_kernels.cu: accumulator/thread = BM*BN/threads is
// fixed at 65536/1024=64 for this tile at the hardware's max threads/
// block, already the ENTIRE register budget with nothing left for
// operands or addresses -- bigtile_w32 itself only had 2 registers of
// slack, and there is no larger thread count to dilute a bigger
// accumulator into. Built to confirm this empirically rather than
// assume it. MEASURED RESULT: confirmed -- ptxas reports exactly 64
// registers (the hard cap) with real spilling, and the damage compounds
// because the accumulator itself (under the most pressure) is read-
// modify-written on every mma call: ~9.1 TFlop/s at N=16384, over 5x
// slower than bigtile_w32 and even slower than the original bigtile2.
// This closes out the tile-size-at-32-warps question: bigtile_w32's
// 256x128 tile is very close to the largest that fits without spilling
// at max threads/block. gemm_fp16_mma_ptx_bigtile_w32 remains the best
// kernel in this family. See mm_cuda_kernels.cu for the full reasoning.
void gemm_fp16_mma_ptx_bigtile2_w32(TransposeType transA, TransposeType transB, double alpha,
                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same idea as gemm_fp16_mma_ptx_bigtile, pushed further: a fully
// symmetric 256x256 threadblock tile (vs bigtile's asymmetric 256x128),
// which would cut redundant HBM traffic by another ~33% on top of
// bigtile's already-large reduction IF it worked -- but it doesn't.
// MEASURED RESULT: a clear LOSS, not a further win. Doubling FragN to 8
// (BN doubling to 256) pushes per-thread register need past the
// 128-register cap this GPU's register file allows at 512 threads/block,
// forcing real spilling (128 registers, 1112 bytes spill stores/1116
// bytes spill loads per ptxas, vs. bigtile's zero spilling at 120
// registers). The spill cost completely swamps the bandwidth benefit:
// ~13 TFlop/s at N=8192, worse than even the original 128x128
// gemm_fp16_mma_ptx (34 TFlop/s). Correctness passes -- this is a real
// performance regression, marking where tile-scaling's benefit stops
// outrunning its register-pressure cost. Kept for comparison, not
// promoted; gemm_fp16_mma_ptx_bigtile remains the best kernel in this
// family. Falls back to bigtile/mma_ptx via the same grid-vs-SM-count
// idea when its huge tile would underfill the GPU at small N.
void gemm_fp16_mma_ptx_bigtile2(TransposeType transA, TransposeType transB, double alpha,
                                size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

// Same 256x128/BK=64 tile and register footprint as gemm_fp16_mma_ptx_
// bigtile (none of bigtile2's register-spill risk), but with cp.async
// prefetch depth extended from 2 stages to 3 to hide more global-memory
// latency behind compute -- targets occupancy/latency, a different axis
// than bigtile's bandwidth fix or bigtile2's (failed) register-pressure
// gamble. Costs only shared memory (~150KB vs bigtile's ~100KB, still
// under this GPU's ceiling). MEASURED RESULT: neutral -- essentially
// tied with bigtile at every N (bigtile usually very slightly ahead),
// meaning bigtile's 2-stage buffer already hid the relevant latency.
// Kept for comparison; gemm_fp16_mma_ptx_bigtile remains the best kernel
// in this family. See mm_cuda_kernels.cu for the full measurement.
void gemm_fp16_mma_ptx_bigtile_tripipe(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);

}  // namespace experimental

// Thin forwarding declarations -- see the matching definitions in
// mm_cuda_kernels.cu. Same signatures as before this reorganization.
void gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_snake_midpoint_exact(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, float* A, size_t ld_a,
                                                            float* B, size_t ld_b, double beta, float* C, size_t ld_c);
void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                        double beta, __half* C, size_t ld_c);
void gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __nv_bfloat16* A,
    size_t ld_a, __nv_bfloat16* B, size_t ld_b, double beta, __nv_bfloat16* C, size_t ld_c);
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
// Fast-decode sibling of the forwarding declaration above -- added as a
// new, separate function, original untouched. See mm_cuda_kernels.cu.
void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
void mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, const uint16_t* wh, size_t ld_a,
                                                                           const __half* B, size_t ld_b, double beta,
                                                                           __half* C, size_t ld_c);
void mm_fp32_bf16_pure_bf16_gemm_direct_fastk_precomp2_serp(TransposeType transA, TransposeType transB, double alpha,
                                                              size_t m, size_t k, size_t n, const uint16_t* wh,
                                                              size_t ld_a, const __nv_bfloat16* B, size_t ld_b,
                                                              double beta, __nv_bfloat16* C, size_t ld_c);

// C = alpha * op(A) * op(B) + beta * C  (A in CSR, B and C column-major strided)
template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr(TransposeType transA, TransposeType transB, double alpha,
            dim<2> size_A, size_t nnz, a_t* values, index_t* row_ptrs,
            index_t* col_idxs, dim<2> size_B, b_t* B, size_t ld_b,
            double beta, c_t* C, size_t ld_c, cusparseHandle_t handle);

// C = alpha * A * B + beta * C  (A and B in CSR, C column-major strided).
// Only the untransposed product is supported. Custom kernel (no cuSPARSE
// handle required).
template <typename a_t, typename b_t, typename c_t, typename index_t>
void mm_csr_csr(double alpha, dim<2> size_A, index_t* row_ptrs_A,
               index_t* col_idxs_A, a_t* values_A, dim<2> size_B,
               index_t* row_ptrs_B, index_t* col_idxs_B, b_t* values_B,
               double beta, c_t* C, size_t ld_c);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_MM_CUDA_KERNELS_CUH
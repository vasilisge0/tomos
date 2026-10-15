// "_advanced" variant of mv_fp32_recon_vs_cutlass_ref.cu -- byte-for-
// byte identical except for time_call's own timing methodology and two
// new CLI flags, --warmup N and --iters N (both default 10, see main()'s
// own argv-filtering preamble): instead of this file's own sibling's
// fixed, shape-size-dependent `reps` heuristic (a single un-timed
// warmup call, then `reps` timed calls averaged), this variant always
// does g_warmup_runs UN-TIMED warmup calls first, then times g_
// iterations calls as one block and divides by g_iterations -- both
// counts fully caller-controlled, independent of shape. No kernel call
// site, shape list, or dispatch logic differs from the sibling file.
//
// Standalone throughput comparison for the "full fp32 reconstruction"
// tier of GEMV (matrix-vector, y = alpha*op(A)*x + beta*y): mv_fp32_16_
// accurate and mv_fp32_bf16_accurate (both src/kernels/cuda/matrix/
// fp32_16_cuda_kernels.cu) -- against this codebase's own new tuned
// native (unpacked) fp32 GEMV kernel (mv_fp32_tiled, src/kernels/cuda/
// matrix/mv_cuda_kernels.cu), real CUTLASS's fp32 SIMT device::Gemv, and
// cuBLAS's Sgemv. All five run in one process so the numbers share the
// same GPU clock/thermal/contention state. Mirrors fp32_recon_vs_
// cutlass_ref.cu's own structure and reasoning (that file's GEMM
// analog), adapted for GEMV: no N dimension, flops = 2*m*k, and two
// shape families -- square (M=K) and tall-and-thin (K held fixed at a
// small value, M swept large) -- instead of GEMM's square/rectangular
// split, since GEMV's own realistic "many rows, few columns" shape is
// tall-and-thin, not a K-varying rectangle.
//
// mv_fp32_tiled, mv_fp32_16_accurate, and mv_fp32_bf16_accurate all
// share the SAME 3-way shape dispatch (by thread count, not aspect
// ratio -- see mv_fp32_16_accurate's own comment in fp32_16_cuda_
// kernels.cu for the full derivation: coalesced one-thread-per-row
// access beats warp-reduction tricks for this memory-bound workload,
// and a split-K column-partitioning pass is what actually fixes "too
// few rows to fill the GPU"). mv_fp32_tiled and mv_fp32_bf16_accurate
// are both new (built alongside this file) -- mv_fp32_tiled is this
// codebase's first tuned native fp32 GEMV kernel (previously only a
// naive gemv_generic_kernel and cuBLAS's own Sgemv existed); mv_fp32_
// bf16_accurate is the first fp32_bf16 GEMV kernel of any kind.
//
// The CUTLASS config here is real cutlass::gemm::device::Gemv (NOT a
// device::Gemm with N=1 workaround) -- CUTLASS v3.5.1 (the vendored
// checkout) has a genuine, tested GEMV device API (cutlass-ref/test/
// unit/gemm/device/gemv.cu, SM50_Device_Gemv_f32n_f32_f32_simt_f32),
// with no size-minimum restrictions on M/K. ColumnMajor A is used
// (matching this codebase's own BLAS convention throughout) since
// CUTLASS's RowMajor-A Gemv specialization additionally requires
// K % kElementsPerAccess == 0, a restriction ColumnMajor doesn't have.
//
// Correctness for mv_fp32_tiled and mv_fp32_bf16_accurate is covered in
// verify_mv_fp32_16_gemv.cu (extended alongside this file); mv_fp32_16_
// accurate's own correctness is already covered there too -- so this
// file is throughput-only, same convention every *_vs_cutlass_ref.cu
// file in this codebase already uses.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call the
// tomos kernels directly. See scripts/benchmarks_gemv_fp32_recon_vs_
// cutlass_ref.sh.
//
// Usage:
//   mv_fp32_recon_vs_cutlass_ref
//       Sweeps two shape families: square M=K=1024/2048/4096/8192/16384/
//       32768/65536/131072 (matching mv_fp32_16_accurate's own
//       kMvWideRowThreshold=131072 boundary, so the sweep crosses from
//       the split-K "wide" dispatch path into the "narrow" one), then a
//       tall-and-thin sweep with K held fixed at 512 and then 4096, M
//       swept across the same size list -- covering both realistic small
//       and moderate column counts while M stays >> K throughout.
//   mv_fp32_recon_vs_cutlass_ref 8192
//       Just that one square shape (M=K=8192).
//   mv_fp32_recon_vs_cutlass_ref --tall 512
//       Just the tall-and-thin sweep at K=512, M swept across the
//       default size list.
//   mv_fp32_recon_vs_cutlass_ref --tall 512 1048576
//       Just the single tall-and-thin shape M=1048576, K=512.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/util/host_tensor.h"

#include "dimensions.hpp"
#include "fp32_16_pack.hpp"
#include "fp32_bf16_pack.hpp"
#include "transpose.hpp"

namespace tomos {
namespace cuda {
void mv_fp32_tiled(TransposeType trans, double alpha, dim<2> size, const float* A,
                    size_t ld_a, const float* x, size_t ld_x, double beta, float* y,
                    size_t ld_y);
void mv_fp32_16_accurate(TransposeType trans, double alpha, dim<2> size,
                         const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                         const float* x, size_t ld_x, double beta, float* y,
                         size_t ld_y);
namespace experimental { void mv_fp32_16_accurate_facc(TransposeType trans, double alpha, dim<2> size,
                              const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                              const float* x, size_t ld_x, double beta, float* y,
                              size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_accurate_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_workspace(TransposeType trans, double alpha, dim<2> size,
                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                   const float* x, size_t ld_x, double beta, float* y,
                                   size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_workspace_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                               const float* x, size_t ld_x, double beta, float* y,
                                               size_t ld_y, double* workspace); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_atomic(TransposeType trans, double alpha, dim<2> size,
                                const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                const float* x, size_t ld_x, double beta, float* y,
                                size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_16_accurate_ready_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready(TransposeType trans, double alpha, dim<2> size,
                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                               const float* x, size_t ld_x, double beta, float* y,
                               size_t ld_y, int* ready); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                           const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                           const float* x, size_t ld_x, double beta, float* y,
                                           size_t ld_y, int* ready); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready4_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                            const float* x, size_t ld_x, double beta, float* y,
                                            size_t ld_y, int* ready); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready4_kblock64_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                      const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                      const float* x, size_t ld_x, double beta, float* y,
                                                      size_t ld_y, int* ready); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready_atomiconly_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                       const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                       const float* x, size_t ld_x, double beta, float* y,
                                                       size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_nosplit_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                             const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                             const float* x, size_t ld_x, double beta, float* y,
                                             size_t ld_y); }  // namespace experimental

void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y);
namespace experimental { void mv_fp32_16_accurate_ready_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                         const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                         const float* x, size_t ld_x, double beta, float* y,
                                                         size_t ld_y); }  // namespace experimental

void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y);
namespace experimental { void mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw(TransposeType trans, double alpha, dim<2> size,
                                                               const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                               const float* x, size_t ld_x, double beta, float* y,
                                                               size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock(TransposeType trans, double alpha, dim<2> size,
                                                          const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                          const float* x, size_t ld_x, double beta, float* y,
                                                          size_t ld_y); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_accurate(TransposeType trans, double alpha, dim<2> size,
                            const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                            const float* x, size_t ld_x, double beta, float* y,
                            size_t ld_y); }  // namespace experimental

namespace experimental { size_t mv_fp32_bf16_accurate_workspace_size(TransposeType trans, dim<2> size); }  // namespace experimental

namespace experimental { void mv_fp32_bf16_accurate_workspace(TransposeType trans, double alpha, dim<2> size,
                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                     const float* x, size_t ld_x, double beta, float* y,
                                     size_t ld_y, double* workspace); }  // namespace experimental

void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(TransposeType trans, double alpha, dim<2> size,
                                                                     const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                     const float* x, size_t ld_x, double beta, float* y,
                                                                     size_t ld_y, double* workspace);
void mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(TransposeType trans, double alpha, dim<2> size,
                                                                   const uint16_t* wh, const uint16_t* wr, size_t ld_a,
                                                                   const float* x, size_t ld_x, double beta, float* y,
                                                                   size_t ld_y, double* workspace);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

// Real CUTLASS GEMV -- see this file's own header comment for why
// ColumnMajor A (matching this codebase's own BLAS convention, and
// avoiding RowMajor's K-alignment restriction) and why this uses
// device::Gemv directly rather than an N=1 device::Gemm workaround.
//
// This measures deliberately poorly against cuBLAS/the native kernel --
// confirmed by reading CUTLASS v3.5.1's own kernel/gemv.h source (the
// same tag vendored for this build). The ColumnMajor-A specialization
// used here is a bare reference implementation, not a tuned Level-2 BLAS
// routine: one thread per output row (grid = ceil(M/kThreadCount) blocks
// x kThreadCount=32 threads, so total threads launched = M, with no
// split-K/occupancy fallback the way this codebase's own mv_fp32_wide_
// kernel has for small M), a fully serial per-thread loop over every K
// element (CUTLASS_PRAGMA_NO_UNROLL), and every load scalar (`*ptr_A`,
// `*ptr_B`) -- kElementsPerAccess is a template parameter of the whole
// Gemv kernel struct but is NEVER referenced inside this ColumnMajor
// specialization's operator(), so it is a complete no-op here (verified
// against the literal source; left at 1 below rather than a higher value
// to avoid implying it does something). Vectorized loads and multi-
// thread-per-row reduction only exist in CUTLASS's separate RowMajor-A
// specialization, which would require A physically transposed into a
// row-major buffer first (this codebase's own storage is column-major,
// ld_a=M) -- not attempted here; the ColumnMajor numbers are kept and
// documented as CUTLASS's own naive-reference GEMV baseline.
using ElementInput = float;
using ElementOutput = float;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
constexpr int kCutlassGemvElementsPerAccess = 1;

using CutlassEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, kCutlassGemvElementsPerAccess, ElementAccumulator, ElementAccumulator>;

using CutlassGemv = cutlass::gemm::device::Gemv<
    cutlass::gemm::kernel::Gemv<
        ElementInput, LayoutA, ElementInput, ElementOutput,
        ElementAccumulator, CutlassEpilogueOp, kCutlassGemvElementsPerAccess>
>;

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_float_kernel(size_t n, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand(i);
}
void gen_float(size_t n, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out);
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}

// Safe-normal-range variant of pseudo_rand: magnitude uniformly between
// 0.5 and 1.0, never exact zero, never close enough to zero to reach
// fp32_16's extended/degraded encoding (that boundary is around 2^-14 ~
// 6.1e-5 magnitude -- see fp32_16_pack.hpp's own kFp32ExpMin comment).
// Exists specifically so gen_fp32_16_safe_normal's own wh output
// PROVABLY has control=0 for every element, not just "very likely" --
// plain pseudo_rand's [-1,1] range technically also produces the
// occasional exact-zero or near-zero extended-range value (see fp32_16_
// combine_fast_normal's own comment for why that matters: it decodes
// unconditionally as if control were always 0, silently misinterpreting
// any element where that's not true). Required by "TOMOS FP32/FP16
// (regular)" (mv_fp32_16_accurate_ready_fast_normal) and by the
// superseded mv_fp32_16_accurate_workspace_fast_normal -- neither
// verifies the control=0 contract itself (a self-checking revision of
// the "regular" leg was tried and reverted for measuring worse on real
// hardware), so both need data that provably satisfies it up front.
__device__ float pseudo_rand_safe_normal(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float mag = 0.5f + (static_cast<float>(s % 100001u) / 100000.0f) * 0.5f;  // [0.5, 1.0]
    return (s & 0x10000u) ? mag : -mag;
}

__global__ void gen_fp32_16_safe_normal_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand_safe_normal(i), wh[i], wr[i]);
}
void gen_fp32_16_safe_normal(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_safe_normal_kernel<<<blocks, threads>>>(n, wh, wr);
}

__global__ void gen_fp32_bf16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_kernel<<<blocks, threads>>>(n, wh, wr);
}

// Rounded-encoding sibling of gen_fp32_bf16: same pseudo_rand data, but
// split via fp32_bf16_split_rounded (properly-rounded bf16 in wh, wr's
// bit0 a round_flag) instead of plain truncating fp32_bf16_split. Needed
// because the two fp32_bf16 encodings are not interchangeable -- see
// mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded's own
// comment in fp32_16_cuda_kernels.cu.
__global__ void gen_fp32_bf16_rounded_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_bf16_split_rounded(pseudo_rand(i), wh[i], wr[i]);
}
void gen_fp32_bf16_rounded(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_bf16_rounded_kernel<<<blocks, threads>>>(n, wh, wr);
}

}  // namespace

// Caller-controlled in place of every call site's own `reps` argument
// (still accepted, for source compatibility with the sibling file's own
// call sites, but ignored -- see main()'s own argv-filtering preamble
// for where these are actually set, from --warmup/--iters).
int g_warmup_runs = 10;
int g_iterations = 10;

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int /*reps*/) {
    for (int w = 0; w < g_warmup_runs; ++w) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < g_iterations; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / g_iterations;
}

bool alloc_or_skip(void** ptr, size_t bytes, const char* what, const char* shape_desc) {
    if (cudaMalloc(ptr, bytes) == cudaSuccess) return true;
    std::printf("    %-38s SKIPPED (out of device memory at %s)\n", what, shape_desc);
    cudaGetLastError();
    *ptr = nullptr;
    return false;
}

// m/k follow BLAS convention: A is M x K, x is length K, y is length M
// (column-major A, so lda=m). x/y are shared across every kernel group
// below (small relative to A, always live); A's own representation
// (plain float, wh/wr for fp32_16, wh/wr for fp32_bf16) is transient,
// allocated right before and freed right after its own timing block --
// same staggered-allocation discipline benchmarks_gemm_fp32.cu's
// own run_shape uses, so peak footprint is x+y plus whichever single
// A representation is currently allocated, never more than one at once.
void run_shape(size_t m, size_t k, cudaEvent_t start, cudaEvent_t stop, cublasHandle_t handle) {
    char shape_desc[64];
    std::snprintf(shape_desc, sizeof(shape_desc), "M=%zu K=%zu", m, k);

    int lda = static_cast<int>(m);
    size_t maxdim = std::max(m, k);
    // GEMV's own O(M*K) work (no N multiplier) finishes much faster per
    // call than a comparable GEMM at similar dimensions, so more reps
    // are used here to keep fixed per-call overhead (kernel launch,
    // event recording) from dominating the timing at every size tier.
    int reps = (maxdim <= 4096) ? 50 : (maxdim <= 16384 ? 20 : 5);
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k);

    std::printf("  %s\n", shape_desc);

    float *x = nullptr, *y = nullptr;
    bool ok_xy = alloc_or_skip(reinterpret_cast<void**>(&x), k * sizeof(float), "x", shape_desc) &&
                alloc_or_skip(reinterpret_cast<void**>(&y), m * sizeof(float), "y", shape_desc);
    if (ok_xy) {
        gen_float(k, x);
        cudaMemset(y, 0, m * sizeof(float));
        cudaDeviceSynchronize();
    }

    double ms_native = -1.0, ms_fp32_16 = -1.0, ms_fp32_16_malloc = -1.0, ms_fp32_16_facc = -1.0, ms_fp32_16_ws = -1.0, ms_fp32_16_atomic = -1.0, ms_fp32_16_ready = -1.0, ms_fp32_16_ready_fast_normal = -1.0, ms_fp32_16_ready4_fast_normal = -1.0, ms_fp32_16_ready4_kblock64_fast_normal = -1.0, ms_fp32_16_ready_atomiconly_fast_normal = -1.0, ms_fp32_16_ready_atomiconly_scaled_kblock = -1.0, ms_fp32_16_ready4_atomiconly_scaled_kblock = -1.0, ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw = -1.0, ms_fp32_16_ready8_atomiconly_scaled_kblock = -1.0, ms_fp32_16_nosplit_fast_normal = -1.0, ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal = -1.0, ms_fp32_16_regular = -1.0, ms_fp32_bf16 = -1.0, ms_fp32_bf16_malloc = -1.0, ms_fp32_bf16_ws = -1.0, ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace = -1.0, ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded = -1.0, ms_cutlass = -1.0, ms_cublas = -1.0;

    if (ok_xy) {
        float* A_native = nullptr;
        if (alloc_or_skip(reinterpret_cast<void**>(&A_native), m * k * sizeof(float), "mv_fp32_tiled (A)",
                          shape_desc)) {
            gen_float(m * k, A_native);
            cudaDeviceSynchronize();

            ms_native = time_call(start, stop, [&]() {
                cuda::mv_fp32_tiled(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, A_native, static_cast<size_t>(lda),
                                    x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_tiled          = %9.3fms (%10.1f GFlop/s)\n", ms_native,
                       flops / ms_native / 1e6);

            // Real CUTLASS device::Gemv, same A_native/x/y as mv_fp32_tiled
            // above -- shares that same "same working precision, plain
            // storage" ceiling comparison this codebase's GEMM benchmarks
            // already use.
            cutlass::TensorRef<ElementInput, LayoutA> A_ref(A_native, LayoutA(lda));
            typename CutlassGemv::Arguments args{
                cutlass::MatrixCoord{static_cast<int>(m), static_cast<int>(k)},
                /*batch_count=*/1,
                {1.7f, 0.3f},
                A_ref, x, y, y,
                /*batch_stride_A=*/0, /*batch_stride_B=*/0, /*batch_stride_C=*/0, /*batch_stride_D=*/0
            };
            CutlassGemv gemv_op;
            cutlass::Status status = gemv_op.can_implement(args);
            if (status == cutlass::Status::kSuccess) status = gemv_op.initialize(args);
            if (status == cutlass::Status::kSuccess) {
                ms_cutlass = time_call(start, stop, [&]() { gemv_op(); }, reps);
                std::printf("    cutlass (real, SIMT fp32) = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass,
                           flops / ms_cutlass / 1e6);
            } else {
                std::printf("    cutlass SKIPPED (%s)\n", cutlass::cutlassGetStatusString(status));
            }

            float f_alpha = 1.7f, f_beta = 0.3f;
            ms_cublas = time_call(start, stop, [&]() {
                cublasSgemv(handle, CUBLAS_OP_N, static_cast<int>(m), static_cast<int>(k), &f_alpha, A_native, lda, x,
                           1, &f_beta, y, 1);
            }, reps);
            std::printf("    cublas (Sgemv)         = %9.3fms (%10.1f GFlop/s)\n", ms_cublas,
                       flops / ms_cublas / 1e6);

            cudaFree(A_native);
        }
    }

    if (ok_xy) {
        uint16_t *wh16 = nullptr, *wr16 = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr16), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate (wr)", shape_desc);
        if (ok) {
            gen_fp32_16(m * k, wh16, wr16);
            cudaDeviceSynchronize();

            // mv_fp32_16_accurate -- SUPERSEDED as plain "TOMOS FP32/FP16"
            // by mv_fp32_16_accurate_ready_atomiconly_scaled_kblock below
            // (still run here, for direct comparison, not removed):
            // malloc/free-per-call baseline, the wide (split-K) path's
            // own cudaMalloc+cudaFree on every call.
            ms_fp32_16_malloc = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_accurate(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16,
                                          static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate    = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_16_malloc,
                       flops / ms_fp32_16_malloc / 1e6);

            // EXPERIMENTAL: float-accumulator sibling, A/B'd against the
            // double-accumulator leg just above on the SAME wh16/wr16 --
            // see mv_fp32_16_accurate_wide_facc_kernel's own comment in
            // fp32_16_cuda_kernels.cu for what this tests.
            ms_fp32_16_facc = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_facc(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16,
                                               static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_facc = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_16_facc,
                       flops / ms_fp32_16_facc / 1e6);

            // mv_fp32_16_accurate_workspace -- same double-accumulator
            // computation and kernels as the plain "accurate" leg above,
            // but the wide (split-K) path's scratch buffer is allocated
            // ONCE here, outside time_call's timed loop, and reused
            // across every rep -- instead of mv_fp32_16_accurate's own
            // fresh cudaMalloc+cudaFree on every call. See mv_fp32_16_
            // accurate_workspace's own comment in fp32_16_cuda_kernels.cu
            // for why this specifically targets randls2.0's own
            // DenseWorkspace pattern (a sibling project's composite-
            // precision GEMV, which pre-allocates its own equivalent
            // scratch buffer once per matrix rather than per call).
            size_t ws_elems = cuda::experimental::mv_fp32_16_accurate_workspace_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* d_workspace = nullptr;
            bool ws_ok = ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&d_workspace), ws_elems * sizeof(double),
                              "mv_fp32_16_accurate_workspace (workspace)", shape_desc);
            if (ws_ok) {
                ms_fp32_16_ws = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_accurate_workspace(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16,
                                                        static_cast<size_t>(lda), x, 1, 0.3, y, 1, d_workspace);
                }, reps);
                std::printf("    mv_fp32_16_accurate_workspace = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_16_ws,
                           flops / ms_fp32_16_ws / 1e6);
            }
            cudaFree(d_workspace);

            // mv_fp32_16_accurate_atomic -- recreates randls2.0's own
            // split-K GEMV mechanism (atomicAdd into y, no partial[]
            // buffer, no reduce kernel, no workspace of any kind needed
            // by the caller at all) rather than just its allocation-
            // lifetime idea -- see its own comment in fp32_16_cuda_
            // kernels.cu for the full derivation and the real precision
            // trade-off (float atomicAdd across col_splits contributions
            // instead of an exact double reduction) this accepts.
            ms_fp32_16_atomic = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_atomic(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16,
                                                 static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_atomic = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_16_atomic,
                       flops / ms_fp32_16_atomic / 1e6);

            // mv_fp32_16_accurate_ready -- exact recreation of randls2.0's
            // own ktiled_atomic_ready mechanism (2 rows/lane, packed
            // uint32_t loads, float accumulation, ready-flag atomics --
            // see its own comment in fp32_16_cuda_kernels.cu). `ready` is
            // allocated ONCE here (outside time_call's timed loop) and
            // reused across every rep, matching randls2.0's own
            // gemv_readiness_array_ -- the kernel itself memsets it to 0
            // on every call (cheap linear write, not a malloc).
            size_t ready_elems = cuda::experimental::mv_fp32_16_accurate_ready_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            int* d_ready = nullptr;
            bool ready_ok = ready_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&d_ready), ready_elems * sizeof(int),
                              "mv_fp32_16_accurate_ready (ready)", shape_desc);
            if (ready_ok) {
                ms_fp32_16_ready = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_accurate_ready(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh16, wr16,
                                                    static_cast<size_t>(lda), x, 1, 0.3, y, 1, d_ready);
                }, reps);
                std::printf("    mv_fp32_16_accurate_ready = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_16_ready,
                           flops / ms_fp32_16_ready / 1e6);
            }
            cudaFree(d_ready);

            // mv_fp32_16_accurate_ready_atomiconly_scaled_kblock --
            // SUPERSEDED as plain "TOMOS FP32/FP16" by mv_fp32_16_
            // accurate_ready4_atomiconly_scaled_kblock below (still run
            // here, for direct comparison, not removed): SAME row-
            // vectorized/atomiconly/scaled-K_BLOCK structure as
            // "(regular)" (the fast_normal sibling, safe-normal-range-
            // data block above), but decoding via the full fp32_16_
            // combine_branchless instead -- safe on arbitrary data, so
            // this reuses the SAME plain wh16/wr16 data as the
            // "accurate"/"accurate_ready" legs above, not a dedicated
            // safe-range generator. Also needs no external workspace/
            // scratch buffer of any kind.
            ms_fp32_16_ready_atomiconly_scaled_kblock = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                         wh16, wr16, static_cast<size_t>(lda),
                                                                         x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready_atomiconly_scaled_kblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready_atomiconly_scaled_kblock, flops / ms_fp32_16_ready_atomiconly_scaled_kblock / 1e6);

            // mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock --
            // PROMOTED as plain "TOMOS FP32/FP16": 4-rows/lane extension
            // of the SAFE kernel just above -- unlike mv_fp32_16_ready4_
            // fast_normal_kernel's own 4-row experiment (which lost to
            // its 2-row sibling), amortizing the expensive branchless
            // decode over more rows wins outright here, no K_BLOCK
            // compensation needed -- 1.070x-1.115x vs the 2-row kernel
            // on real hardware (tall-and-thin M sweep at K=4096; square
            // shapes not separately confirmed before promotion). See its
            // own comment in fp32_16_cuda_kernels.cu for the full
            // derivation. Reuses the SAME plain wh16/wr16 data, safe on
            // arbitrary data, no external buffer.
            ms_fp32_16_ready4_atomiconly_scaled_kblock = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                          wh16, wr16, static_cast<size_t>(lda),
                                                                          x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready4_atomiconly_scaled_kblock, flops / ms_fp32_16_ready4_atomiconly_scaled_kblock / 1e6);
            // "TOMOS FP32/FP16" IS this same measurement -- reused, not
            // rerun. See the comment just above for the derivation.
            ms_fp32_16 = ms_fp32_16_ready4_atomiconly_scaled_kblock;

            // mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw --
            // MEASURED WORSE on real A100 hardware: well under 300 GFlop/s
            // across every tested size, vs ~500-720 GFlop/s for the plain
            // ready4_atomiconly_scaled_kblock kernel above -- a real,
            // large regression (roughly 2x or more slower), bigger than
            // this file's own SASS instruction-count comparison alone
            // predicted (1744 vs 1032, ~1.7x). Same 4-rows/lane structure
            // as the SAFE kernel above, decoding via fp32_16_combine_
            // branchless_nohw (pure bit-manipulation degraded path, no
            // hardware half conversion -- see its own comment in
            // fp32_16_cuda_kernels.cu); the extra slowdown beyond the
            // instruction-count ratio is most likely the subnormal-
            // renormalization step's sequential dependency chain (each
            // step of fp32_16_highest_bit_10 depends on the previous),
            // which a flat instruction count doesn't capture. Kept as a
            // documented negative result, not deleted -- reuses the SAME
            // plain wh16/wr16 data, safe on arbitrary data, no external
            // buffer.
            ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                               wh16, wr16, static_cast<size_t>(lda),
                                                                               x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock_nohw = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw, flops / ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw / 1e6);

            // mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock --
            // EXPERIMENTAL, NOT wired into the plot pipeline: 8-rows/lane
            // extension, one step further than the 4-row kernel just
            // above -- tests whether amortizing the SAFE decode over even
            // more rows keeps paying off, or whether register pressure
            // (43 registers vs the 4-row kernel's 32, though still 0
            // spills) catches up with it. See its own comment in
            // fp32_16_cuda_kernels.cu for the full derivation. Reuses the
            // SAME plain wh16/wr16 data, safe on arbitrary data, no
            // external buffer.
            ms_fp32_16_ready8_atomiconly_scaled_kblock = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                          wh16, wr16, static_cast<size_t>(lda),
                                                                          x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready8_atomiconly_scaled_kblock = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready8_atomiconly_scaled_kblock, flops / ms_fp32_16_ready8_atomiconly_scaled_kblock / 1e6);
        }
        cudaFree(wh16);
        cudaFree(wr16);
    }

    if (ok_xy) {
        // This block covers every mv_fp32_16_*_fast_normal wide-path
        // variant this codebase has built so far, all sharing the SAME
        // safe-normal-range data (wh16_safe/wr16_safe, below) since all
        // of them use fp32_16_combine_fast_normal unconditionally --
        // UNSAFE ON ARBITRARY DATA (a self-checking revision was tried
        // and reverted for measuring worse on real hardware -- see mv_
        // fp32_16_accurate_ready_fast_normal's own comment in fp32_16_
        // cuda_kernels.cu), so this data is NOT comparable 1:1 against
        // the other fp32_16 legs above, which use plain pseudo_rand data
        // instead.
        //
        // "TOMOS FP32 / FP16 (regular)" is currently mv_fp32_16_
        // accurate_ready_atomiconly_scaled_kblock_fast_normal (see its own
        // leg, further down this block) -- promoted from mv_fp32_16_
        // accurate_ready_atomiconly_fast_normal (still run here too, for
        // direct comparison, not superseded/removed), which was itself
        // promoted from mv_fp32_16_accurate_ready_fast_normal (also still
        // run). Each promotion beat its predecessor at every size measured
        // so far -- ties below the K=16384 crossover, wins consistently
        // above it (see mv_fp32_16_accurate_ready_atomiconly_scaled_
        // kblock_fast_normal's own comment in fp32_16_cuda_kernels.cu for
        // the full derivation and numbers).
        uint16_t *wh16_safe = nullptr, *wr16_safe = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh16_safe), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate_ready_fast_normal (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr16_safe), m * k * sizeof(uint16_t),
                               "mv_fp32_16_accurate_ready_fast_normal (wr)", shape_desc);
        if (ok) {
            gen_fp32_16_safe_normal(m * k, wh16_safe, wr16_safe);
            cudaDeviceSynchronize();

            size_t ready_elems = cuda::experimental::mv_fp32_16_accurate_ready_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            int* d_ready_fn = nullptr;
            bool ready_ok = ready_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&d_ready_fn), ready_elems * sizeof(int),
                              "mv_fp32_16_accurate_ready_fast_normal (ready)", shape_desc);
            if (ready_ok) {
                ms_fp32_16_ready_fast_normal = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_accurate_ready_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                                x, 1, 0.3, y, 1, d_ready_fn);
                }, reps);
                std::printf("    mv_fp32_16_accurate_ready_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                           ms_fp32_16_ready_fast_normal, flops / ms_fp32_16_ready_fast_normal / 1e6);
            }
            cudaFree(d_ready_fn);

            // mv_fp32_16_accurate_ready4_fast_normal -- EXPERIMENTAL, NOT
            // wired into the plot pipeline yet (benchmark-only comparison
            // leg): extends mv_fp32_16_accurate_ready_fast_normal's own 2-
            // rows/lane packed-load trick to 4 rows/lane (single uint64_t
            // load instead of uint32_t) -- see its own comment in fp32_16_
            // cuda_kernels.cu for the full derivation. Decode is
            // unconditionally fp32_16_combine_fast_normal, same as the
            // leg above (per instruction not to touch decode) -- reuses
            // the SAME safe-normal-range data (wh16_safe/wr16_safe), its
            // own separate `ready` buffer (identical sizing, kept
            // separate rather than shared for clarity).
            size_t ready4_elems = cuda::experimental::mv_fp32_16_accurate_ready_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            int* d_ready4 = nullptr;
            bool ready4_ok = ready4_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&d_ready4), ready4_elems * sizeof(int),
                              "mv_fp32_16_accurate_ready4_fast_normal (ready)", shape_desc);
            if (ready4_ok) {
                ms_fp32_16_ready4_fast_normal = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_accurate_ready4_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                 wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                                 x, 1, 0.3, y, 1, d_ready4);
                }, reps);
                std::printf("    mv_fp32_16_accurate_ready4_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                           ms_fp32_16_ready4_fast_normal, flops / ms_fp32_16_ready4_fast_normal / 1e6);

                // mv_fp32_16_accurate_ready4_kblock64_fast_normal --
                // EXPERIMENTAL, NOT wired into the plot pipeline yet:
                // SAME kernel as the leg just above, but K_BLOCK halved
                // (64 instead of 128) to compensate 4-rows/lane's own
                // halved gridRows, restoring total block/thread count to
                // parity with the 2-row kernel's own config -- see its
                // own comment in fp32_16_cuda_kernels.cu for the exact
                // arithmetic. Tests whether the plain ready4 leg's own
                // regression was an occupancy artifact. Reuses the SAME
                // d_ready4 buffer (identical sizing, independent of
                // K_BLOCK).
                ms_fp32_16_ready4_kblock64_fast_normal = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_16_accurate_ready4_kblock64_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                          wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                                          x, 1, 0.3, y, 1, d_ready4);
                }, reps);
                std::printf("    mv_fp32_16_accurate_ready4_kblock64_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                           ms_fp32_16_ready4_kblock64_fast_normal, flops / ms_fp32_16_ready4_kblock64_fast_normal / 1e6);
            }
            cudaFree(d_ready4);

            // mv_fp32_16_accurate_ready_atomiconly_fast_normal -- SUPERSEDED
            // as "TOMOS FP32 / FP16 (regular)" by mv_fp32_16_accurate_
            // ready_atomiconly_scaled_kblock_fast_normal below (still run
            // here, for direct comparison, not removed): 2 rows/lane +
            // packed loads (same as mv_fp32_16_accurate_ready_fast_normal),
            // but prescale+atomic-only write-back (mv_fp32_16_accurate_
            // atomic's own mechanism) instead of ready-flag/spin-wait -- no
            // `ready` buffer needed at all. Reuses the SAME safe-normal-
            // range data. See its own comment in fp32_16_cuda_kernels.cu
            // for the full derivation.
            ms_fp32_16_ready_atomiconly_fast_normal = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_ready_atomiconly_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                       wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                                       x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready_atomiconly_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready_atomiconly_fast_normal, flops / ms_fp32_16_ready_atomiconly_fast_normal / 1e6);

            // mv_fp32_16_accurate_nosplit_fast_normal -- MEASURED WORSE
            // (real hardware), NOT wired into the plot pipeline: same
            // 2-rows/lane + packed-load kernel family, but NO split-K at
            // all -- each block loops over the whole K range internally
            // and writes y once, non-atomically. Built to test whether
            // removing split-K's own atomic contention (which becomes the
            // dominant, shared cost at large M/K -- see the "regular"
            // leg's own comment above) would beat the split-K design at
            // large sizes. It does not: loses at every square size tested
            // from M=K=1024 through 65536 (0.150x-0.598x vs the "regular"
            // leg, never approaching parity) -- removing split-K costs far
            // more grid-level parallelism than it saves in atomic
            // contention. See mv_fp32_16_nosplit_fast_normal_kernel's own
            // comment in fp32_16_cuda_kernels.cu for the full derivation.
            // Kept as a documented negative result; no dispatcher.
            ms_fp32_16_nosplit_fast_normal = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_16_accurate_nosplit_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                              wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                              x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_nosplit_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_nosplit_fast_normal, flops / ms_fp32_16_nosplit_fast_normal / 1e6);

            // mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal
            // -- PROMOTED as plain "TOMOS FP32/FP16" (was "(regular)"):
            // SAME kernel as mv_fp32_16_accurate_ready_atomiconly_fast_
            // normal above, but with a K-dependent K_BLOCK (targeting a
            // roughly constant gridK instead of the fixed 128, which lets
            // gridK -- and the atomic contention on y[row] it causes --
            // grow linearly with K). Ties that leg below the K=16384
            // crossover (0.999x-1.011x, noise) and beats it consistently
            // above it -- 1.097x at K=24576, 1.107x at 32768, 1.087x at
            // 49152, 1.090x at 65536 -- also now beating native at every
            // one of those sizes, unlike the fixed-K_BLOCK leg. Keeps
            // split-K's own grid parallelism (unlike the nosplit leg
            // above, which gave up too much of it and lost everywhere).
            // This is also the best-measured kernel in the fp32_16 family
            // that never requires an external workspace/scratch buffer --
            // its own fallback for non-fast-path shapes (transposed, or
            // ld_a != size[0]) is mv_fp32_16_accurate (the malloc-based
            // baseline above), not a caller-provided buffer. See its own
            // comment in fp32_16_cuda_kernels.cu, next to its own
            // definition, for the full derivation.
            ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal = time_call(start, stop, [&]() {
                cuda::mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k},
                                                                                      wh16_safe, wr16_safe, static_cast<size_t>(lda),
                                                                                      x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal, flops / ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal / 1e6);
            // "TOMOS FP32/FP16 (regular)" IS this same measurement --
            // reused, not rerun. UNSAFE on arbitrary data (inherited from
            // fp32_16_combine_fast_normal), so tested here against wh16_
            // safe/wr16_safe, not plain wh16/wr16 -- see mv_fp32_16_
            // accurate_ready_atomiconly_scaled_kblock's own leg (below,
            // in the plain-data block) for the SAFE counterpart promoted
            // as plain "TOMOS FP32/FP16" instead.
            ms_fp32_16_regular = ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal;
        }
        cudaFree(wh16_safe);
        cudaFree(wr16_safe);
    }

    if (ok_xy) {
        uint16_t *wh_bf16 = nullptr, *wr_bf16 = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_accurate (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr_bf16), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_accurate (wr)", shape_desc);
        if (ok) {
            gen_fp32_bf16(m * k, wh_bf16, wr_bf16);
            cudaDeviceSynchronize();

            // mv_fp32_bf16_accurate -- SUPERSEDED as plain "TOMOS FP32/
            // BF16" by mv_fp32_bf16_accurate_ready_atomiconly_scaled_
            // kblock_workspace below (still run here, for direct
            // comparison, not removed): malloc/free-per-call baseline,
            // same dispatch mv_fp32_16_accurate itself uses.
            ms_fp32_bf16_malloc = time_call(start, stop, [&]() {
                cuda::experimental::mv_fp32_bf16_accurate(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16,
                                            static_cast<size_t>(lda), x, 1, 0.3, y, 1);
            }, reps);
            std::printf("    mv_fp32_bf16_accurate  = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_bf16_malloc,
                       flops / ms_fp32_bf16_malloc / 1e6);

            // mv_fp32_bf16_accurate_workspace -- genuinely-always-uses-
            // workspace sibling (classic column-split wide kernel +
            // reduce, no row-vectorized fast path at all -- see its own
            // comment in fp32_16_cuda_kernels.cu). `workspace` IS read
            // and written on every call this benchmark makes, unlike the
            // ready_atomiconly_scaled_kblock_workspace leg below -- this
            // is the one plotted as "TOMOS FP32/BF16 (workspace)".
            size_t bf16_ws_elems = cuda::experimental::mv_fp32_bf16_accurate_workspace_size(TransposeType::NO_TRANSPOSE, dim<2>{m, k});
            double* d_bf16_ws = nullptr;
            bool bf16_ws_ok = bf16_ws_elems == 0 ||
                alloc_or_skip(reinterpret_cast<void**>(&d_bf16_ws), bf16_ws_elems * sizeof(double),
                              "mv_fp32_bf16_accurate_workspace (workspace)", shape_desc);
            if (bf16_ws_ok) {
                ms_fp32_bf16_ws = time_call(start, stop, [&]() {
                    cuda::experimental::mv_fp32_bf16_accurate_workspace(TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16,
                                                          static_cast<size_t>(lda), x, 1, 0.3, y, 1, d_bf16_ws);
                }, reps);
                std::printf("    mv_fp32_bf16_accurate_workspace = %9.3fms (%10.1f GFlop/s)\n", ms_fp32_bf16_ws,
                           flops / ms_fp32_bf16_ws / 1e6);
            }
            cudaFree(d_bf16_ws);

            // mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_
            // workspace -- was PROMOTED as plain "TOMOS FP32/BF16"
            // ("(workspace)" before that): mv_fp32_16_accurate_ready_
            // atomiconly_scaled_kblock_fast_normal's own row-vectorized/
            // atomiconly/scaled-K_BLOCK optimizations, decoding via
            // fp32_bf16_combine -- which, unlike fp32_16_combine_fast_
            // normal, needs no unsafe-on-arbitrary-data shortcut at all
            // (see that kernel's own comment), so this reuses the SAME
            // plain wh_bf16/wr_bf16 data as the "accurate" leg above, not
            // a dedicated safe-range generator the way every fp32_16
            // "_fast_normal" leg needs.
            //
            // `workspace` passed as nullptr: this benchmark's own
            // NO_TRANSPOSE, ld_a==m shape always dispatches to the fast
            // row-vectorized path (see the host wrapper's own comment in
            // fp32_16_cuda_kernels.cu), which never reads or writes
            // `workspace` at all -- confirmed by direct A/B measurement
            // (allocating a real, unused buffer here changed nothing).
            // Beat mv_fp32_bf16_accurate (still run above, unpromoted) by
            // 8-25% at every size tested -- now DEMOTED to plotted
            // "TOMOS FP32/BF16 (truncated)" (its wh_bf16/wr_bf16 data
            // comes from plain truncating fp32_bf16_split, unlike the
            // rounded-encoding leg just below), since mv_fp32_bf16_
            // accurate_ready_atomiconly_scaled_kblock_rounded now sources
            // the plain "TOMOS FP32/BF16" series instead.
            ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16, wr_bf16,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1, nullptr);
            }, reps);
            std::printf("    mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_workspace = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace,
                       flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace / 1e6);
        }
        cudaFree(wh_bf16);
        cudaFree(wr_bf16);
    }

    if (ok_xy) {
        uint16_t *wh_bf16_rounded = nullptr, *wr_bf16_rounded = nullptr;
        bool ok = alloc_or_skip(reinterpret_cast<void**>(&wh_bf16_rounded), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded (wh)", shape_desc) &&
                 alloc_or_skip(reinterpret_cast<void**>(&wr_bf16_rounded), m * k * sizeof(uint16_t),
                               "mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded (wr)", shape_desc);
        if (ok) {
            gen_fp32_bf16_rounded(m * k, wh_bf16_rounded, wr_bf16_rounded);
            cudaDeviceSynchronize();

            // mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded
            // -- PROMOTED as plain "TOMOS FP32/BF16": byte-for-byte the
            // same row-vectorized/atomiconly/scaled-K_BLOCK fast path as
            // the "(truncated)" leg above, but decoding via fp32_bf16_
            // combine_rounded against wh_bf16_rounded/wr_bf16_rounded (own
            // generator, gen_fp32_bf16_rounded, using fp32_bf16_split_
            // rounded -- NOT the plain wh_bf16/wr_bf16 data, since the two
            // fp32_bf16 encodings are not interchangeable: wr's bit0 is a
            // real mantissa bit for the plain split, a round_flag
            // recording whether encoding rounded wh up by 1 for the
            // rounded split). See mv_fp32_bf16_accurate_ready_atomiconly_
            // scaled_kblock_rounded's own comment in fp32_16_cuda_
            // kernels.cu for the full derivation. `workspace` passed as
            // nullptr for the same reason as the "(truncated)" leg above
            // (this benchmark's shapes always dispatch to the fast path).
            ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded = time_call(start, stop, [&]() {
                cuda::mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded(
                    TransposeType::NO_TRANSPOSE, 1.7, dim<2>{m, k}, wh_bf16_rounded, wr_bf16_rounded,
                    static_cast<size_t>(lda), x, 1, 0.3, y, 1, nullptr);
            }, reps);
            std::printf("    mv_fp32_bf16_accurate_ready_atomiconly_scaled_kblock_rounded = %9.3fms (%10.1f GFlop/s)\n",
                       ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded,
                       flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded / 1e6);
            // "TOMOS FP32/BF16" IS this same measurement -- reused, not
            // rerun. See the comment just above for the derivation.
            ms_fp32_bf16 = ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded;
        }
        cudaFree(wh_bf16_rounded);
        cudaFree(wr_bf16_rounded);
    }

    if (ms_native > 0 && ms_cutlass > 0)
        std::printf("    -- native vs cutlass: %.3fx\n", (flops / ms_native) / (flops / ms_cutlass));
    if (ms_native > 0 && ms_cublas > 0)
        std::printf("    -- native vs cublas:  %.3fx\n", (flops / ms_native) / (flops / ms_cublas));
    if (ms_fp32_16 > 0 && ms_native > 0)
        std::printf("    -- fp32_16 vs native: %.3fx\n", (flops / ms_fp32_16) / (flops / ms_native));
    if (ms_fp32_16 > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16 vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_ready_atomiconly_scaled_kblock > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready_atomiconly_scaled_kblock vs native: %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_scaled_kblock) / (flops / ms_native));
    if (ms_fp32_16_ready_atomiconly_scaled_kblock > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_ready_atomiconly_scaled_kblock vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_scaled_kblock) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_ready4_atomiconly_scaled_kblock > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready4_atomiconly_scaled_kblock vs native: %.3fx\n", (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock) / (flops / ms_native));
    if (ms_fp32_16_ready4_atomiconly_scaled_kblock > 0 && ms_fp32_16_ready_atomiconly_scaled_kblock > 0)
        std::printf("    -- fp32_16_ready4_atomiconly_scaled_kblock vs fp32_16_ready_atomiconly_scaled_kblock (2 rows/lane): %.3fx\n", (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock) / (flops / ms_fp32_16_ready_atomiconly_scaled_kblock));
    if (ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready4_atomiconly_scaled_kblock_nohw vs native: %.3fx\n", (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw) / (flops / ms_native));
    if (ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw > 0 && ms_fp32_16_ready4_atomiconly_scaled_kblock > 0)
        std::printf("    -- fp32_16_ready4_atomiconly_scaled_kblock_nohw vs fp32_16_ready4_atomiconly_scaled_kblock (hardware half conversion): %.3fx\n", (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock_nohw) / (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock));
    if (ms_fp32_16_ready8_atomiconly_scaled_kblock > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready8_atomiconly_scaled_kblock vs native: %.3fx\n", (flops / ms_fp32_16_ready8_atomiconly_scaled_kblock) / (flops / ms_native));
    if (ms_fp32_16_ready8_atomiconly_scaled_kblock > 0 && ms_fp32_16_ready4_atomiconly_scaled_kblock > 0)
        std::printf("    -- fp32_16_ready8_atomiconly_scaled_kblock vs fp32_16_ready4_atomiconly_scaled_kblock (4 rows/lane): %.3fx\n", (flops / ms_fp32_16_ready8_atomiconly_scaled_kblock) / (flops / ms_fp32_16_ready4_atomiconly_scaled_kblock));
    if (ms_fp32_16_malloc > 0 && ms_native > 0)
        std::printf("    -- fp32_16_malloc vs native: %.3fx\n", (flops / ms_fp32_16_malloc) / (flops / ms_native));
    if (ms_fp32_16_facc > 0 && ms_native > 0)
        std::printf("    -- fp32_16_facc vs native: %.3fx\n", (flops / ms_fp32_16_facc) / (flops / ms_native));
    if (ms_fp32_16_facc > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_facc vs fp32_16_malloc (double): %.3fx\n", (flops / ms_fp32_16_facc) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_ws > 0 && ms_native > 0)
        std::printf("    -- fp32_16_workspace vs native: %.3fx\n", (flops / ms_fp32_16_ws) / (flops / ms_native));
    if (ms_fp32_16_ws > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_workspace vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16_ws) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_atomic > 0 && ms_native > 0)
        std::printf("    -- fp32_16_atomic vs native: %.3fx\n", (flops / ms_fp32_16_atomic) / (flops / ms_native));
    if (ms_fp32_16_atomic > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_atomic vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16_atomic) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_atomic > 0 && ms_fp32_16_ws > 0)
        std::printf("    -- fp32_16_atomic vs fp32_16_workspace: %.3fx\n", (flops / ms_fp32_16_atomic) / (flops / ms_fp32_16_ws));
    if (ms_fp32_16_ready > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready vs native: %.3fx\n", (flops / ms_fp32_16_ready) / (flops / ms_native));
    if (ms_fp32_16_ready > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_ready vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16_ready) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_ready > 0 && ms_fp32_16_atomic > 0)
        std::printf("    -- fp32_16_ready vs fp32_16_atomic (1 row/thread): %.3fx\n", (flops / ms_fp32_16_ready) / (flops / ms_fp32_16_atomic));
    if (ms_fp32_16_ready_fast_normal > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready_fast_normal vs native: %.3fx\n", (flops / ms_fp32_16_ready_fast_normal) / (flops / ms_native));
    if (ms_fp32_16_ready_fast_normal > 0 && ms_fp32_16_ready > 0)
        std::printf("    -- fp32_16_ready_fast_normal vs fp32_16_ready (branchless decode): %.3fx\n", (flops / ms_fp32_16_ready_fast_normal) / (flops / ms_fp32_16_ready));
    if (ms_fp32_16_regular > 0 && ms_native > 0)
        std::printf("    -- fp32_16_regular vs native: %.3fx\n", (flops / ms_fp32_16_regular) / (flops / ms_native));
    if (ms_fp32_16_regular > 0 && ms_fp32_16_ws > 0)
        std::printf("    -- fp32_16_regular vs fp32_16_workspace (1 row/thread, malloc-free): %.3fx\n", (flops / ms_fp32_16_regular) / (flops / ms_fp32_16_ws));
    if (ms_fp32_16_regular > 0 && ms_fp32_16_malloc > 0)
        std::printf("    -- fp32_16_regular vs fp32_16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_16_regular) / (flops / ms_fp32_16_malloc));
    if (ms_fp32_16_ready4_fast_normal > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready4_fast_normal vs native: %.3fx\n", (flops / ms_fp32_16_ready4_fast_normal) / (flops / ms_native));
    if (ms_fp32_16_ready4_fast_normal > 0 && ms_fp32_16_ready_fast_normal > 0)
        std::printf("    -- fp32_16_ready4_fast_normal vs fp32_16_ready_fast_normal (2 rows/lane): %.3fx\n", (flops / ms_fp32_16_ready4_fast_normal) / (flops / ms_fp32_16_ready_fast_normal));
    if (ms_fp32_16_ready4_kblock64_fast_normal > 0 && ms_fp32_16_ready_fast_normal > 0)
        std::printf("    -- fp32_16_ready4_kblock64_fast_normal vs fp32_16_ready_fast_normal (2 rows/lane): %.3fx\n", (flops / ms_fp32_16_ready4_kblock64_fast_normal) / (flops / ms_fp32_16_ready_fast_normal));
    if (ms_fp32_16_ready4_kblock64_fast_normal > 0 && ms_fp32_16_ready4_fast_normal > 0)
        std::printf("    -- fp32_16_ready4_kblock64_fast_normal vs fp32_16_ready4_fast_normal (K_BLOCK=128): %.3fx\n", (flops / ms_fp32_16_ready4_kblock64_fast_normal) / (flops / ms_fp32_16_ready4_fast_normal));
    if (ms_fp32_16_ready_atomiconly_fast_normal > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready_atomiconly_fast_normal vs native: %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_fast_normal) / (flops / ms_native));
    if (ms_fp32_16_ready_atomiconly_fast_normal > 0 && ms_fp32_16_ready_fast_normal > 0)
        std::printf("    -- fp32_16_ready_atomiconly_fast_normal vs fp32_16_ready_fast_normal (ready-flag): %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_fast_normal) / (flops / ms_fp32_16_ready_fast_normal));
    if (ms_fp32_16_nosplit_fast_normal > 0 && ms_native > 0)
        std::printf("    -- fp32_16_nosplit_fast_normal vs native: %.3fx\n", (flops / ms_fp32_16_nosplit_fast_normal) / (flops / ms_native));
    if (ms_fp32_16_nosplit_fast_normal > 0 && ms_fp32_16_ready_atomiconly_fast_normal > 0)
        std::printf("    -- fp32_16_nosplit_fast_normal vs fp32_16_ready_atomiconly_fast_normal (split-K): %.3fx\n", (flops / ms_fp32_16_nosplit_fast_normal) / (flops / ms_fp32_16_ready_atomiconly_fast_normal));
    if (ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal > 0 && ms_native > 0)
        std::printf("    -- fp32_16_ready_atomiconly_scaled_kblock_fast_normal vs native: %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal) / (flops / ms_native));
    if (ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal > 0 && ms_fp32_16_ready_atomiconly_fast_normal > 0)
        std::printf("    -- fp32_16_ready_atomiconly_scaled_kblock_fast_normal vs fp32_16_ready_atomiconly_fast_normal (fixed K_BLOCK): %.3fx\n", (flops / ms_fp32_16_ready_atomiconly_scaled_kblock_fast_normal) / (flops / ms_fp32_16_ready_atomiconly_fast_normal));
    if (ms_fp32_bf16 > 0 && ms_native > 0)
        std::printf("    -- fp32_bf16 vs native: %.3fx\n", (flops / ms_fp32_bf16) / (flops / ms_native));
    if (ms_fp32_bf16_malloc > 0 && ms_native > 0)
        std::printf("    -- fp32_bf16_malloc vs native: %.3fx\n", (flops / ms_fp32_bf16_malloc) / (flops / ms_native));
    if (ms_fp32_bf16_ws > 0 && ms_native > 0)
        std::printf("    -- fp32_bf16_workspace vs native: %.3fx\n", (flops / ms_fp32_bf16_ws) / (flops / ms_native));
    if (ms_fp32_bf16_ws > 0 && ms_fp32_bf16_malloc > 0)
        std::printf("    -- fp32_bf16_workspace vs fp32_bf16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_bf16_ws) / (flops / ms_fp32_bf16_malloc));
    if (ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace > 0 && ms_native > 0)
        std::printf("    -- fp32_bf16_ready_atomiconly_scaled_kblock_workspace vs native: %.3fx\n", (flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace) / (flops / ms_native));
    if (ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace > 0 && ms_fp32_bf16_malloc > 0)
        std::printf("    -- fp32_bf16_ready_atomiconly_scaled_kblock_workspace vs fp32_bf16_malloc (malloc/free per call): %.3fx\n", (flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace) / (flops / ms_fp32_bf16_malloc));
    if (ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded > 0 && ms_native > 0)
        std::printf("    -- fp32_bf16_ready_atomiconly_scaled_kblock_rounded vs native: %.3fx\n", (flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded) / (flops / ms_native));
    if (ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded > 0 && ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace > 0)
        std::printf("    -- fp32_bf16_ready_atomiconly_scaled_kblock_rounded vs fp32_bf16_ready_atomiconly_scaled_kblock_workspace (truncated): %.3fx\n", (flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_rounded) / (flops / ms_fp32_bf16_ready_atomiconly_scaled_kblock_workspace));

    cudaFree(x);
    cudaFree(y);
}

int main(int argc, char** argv) {
    // Extract --warmup N / --iters N from anywhere in argv (both
    // optional, default 10/10, see g_warmup_runs/g_iterations' own
    // comment above time_call) before this file's own original argument
    // parsing below runs -- argc/argv are reassigned to a filtered copy
    // with those two flags (and their values) removed, so every shape/
    // flag form the sibling file's own main() accepts still works
    // unchanged here.
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) {
            g_warmup_runs = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            g_iterations = std::atoi(argv[i + 1]);
            ++i;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== [advanced: warmup=%d iters=%d] ===\n", g_warmup_runs, g_iterations);
    std::printf("=== mv_fp32_16_accurate / mv_fp32_bf16_accurate / mv_fp32_tiled (native) "
               "vs real CUTLASS (fp32 SIMT GEMV) vs cuBLAS Sgemv ===\n");

    // Matches mv_fp32_16_accurate's own kMvWideRowThreshold=131072
    // boundary (fp32_16_cuda_kernels.cu) -- sweeping across it exercises
    // both the split-K "wide" dispatch path (below the threshold) and
    // the "narrow"/"narrow_vec" path (at/above it) for every kernel
    // compared here, since all three share that same dispatch logic.
    const std::vector<size_t> square_sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    const std::vector<size_t> tall_ms = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    const std::vector<size_t> tall_ks = {512, 4096};

    if (argc > 1 && std::string(argv[1]) == "--tall") {
        size_t k = (argc > 2) ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : tall_ks[0];
        if (argc > 3) {
            size_t m = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));
            run_shape(m, k, start, stop, handle);
        } else {
            for (size_t m : tall_ms) run_shape(m, k, start, stop, handle);
        }
        cublasDestroy(handle);
        return 0;
    }

    if (argc > 1) {
        size_t m = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
        run_shape(m, m, start, stop, handle);
        cublasDestroy(handle);
        return 0;
    }

    for (size_t m : square_sizes) run_shape(m, m, start, stop, handle);
    for (size_t k : tall_ks)
        for (size_t m : tall_ms) run_shape(m, k, start, stop, handle);

    cublasDestroy(handle);
    return 0;
}

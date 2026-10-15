// Focused benchmark: the FP32_16 "pure_half" wide_bigreg_sq kernel (mm_
// fp32_16_pure_half_wide_bigreg_sq, src/kernels/cuda/matrix/mm_cuda_
// kernels.cu -- the mma_ptx/wide_bigreg_sq architecture applied to A
// stored as packed FP32_16 wh/wr, decoded via fp32_16_half_only so only
// wr is discarded and precision is genuinely fp16-level, not fp32) and
// its two best-tuned wmma-family FP32_16 siblings (mm_fp32_16_pure_half_
// wmma_cpasync_vec_bigbk_pad, mm_fp32_16_pure_half_wmma_cpasync_vec_
// hugebk, src/kernels/cuda/matrix/fp32_16_cuda_kernels.cu) against real
// CUTLASS's own device::Gemm and cuBLAS's Hgemm, all in one run/process
// so the numbers share the same GPU clock/thermal/contention state.
// Mirrors wide_bigreg_family_vs_cutlass_ref.cu's structure and reasoning
// exactly, just for the FP32_16 "pure_half" tier instead of plain fp16.
//
// CUTLASS and cuBLAS have no packed-FP32_16 input format, so both
// references run on a fully-materialized __half A (decoded from wh via
// fp32_16_half_only, matching how the tomos kernels themselves decode
// it) and a __half B -- this measures how close the packed-storage
// kernels get to the "same working precision, plain storage" ceiling,
// not a different precision tier.
//
// Note an asymmetry in B's and C's types across the three tomos kernels
// here: wide_bigreg_sq takes B and C as native __half (matching gemm_
// fp16_mma_ptx_wide_bigreg_sq's own convention, so its memory traffic is
// byte-for-byte identical to the native-fp16 kernel and only A's
// packed-storage decode cost is being measured), while wmma_bigbk_pad/
// wmma_hugebk keep this codebase's usual FP32_16 convention of B and C
// as plain float (4 bytes/element, converted on the fly) -- so the gap
// between wide_bigreg_sq and its wmma-family siblings here reflects both
// the mma_ptx-vs-wmma architecture difference AND this type difference,
// not architecture alone. Square NxN, column-major, alpha=1.7/beta=0.3,
// cudaEvent_t timing.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS dependency
// in the main project) -- built standalone via a direct nvcc invocation
// that also links against the already-built libtomos_cuda.so to call
// the tomos kernels directly. See scripts/wide_bigreg_sq_fp32_16_vs_
// cutlass_ref.sh.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "fp32_16_cuda_kernels.cuh"
#include "fp32_16_pack.hpp"

// Minimal, exact-match declaration for mm_fp32_16_pure_half_wide_bigreg_
// sq -- it lives in mm_cuda_kernels.cuh, which pulls in cusparse.h and
// the rest of that header's ~1150-line declaration set; avoided here the
// same way wide_bigreg_family_vs_cutlass_ref.cu avoids it, by copying
// just this one signature verbatim and satisfying it at link time via
// libtomos_cuda.so. The wmma-family kernels below come from fp32_16_
// cuda_kernels.cuh instead, which is lightweight (just dimensions.hpp/
// transpose.hpp) and safe to include directly.
//
// Note B's and C's types here: __half, NOT float. Unlike this codebase's
// other FP32_16 kernels (which keep B/C as plain float), this kernel
// takes both as native __half so its memory traffic matches gemm_fp16_
// mma_ptx_wide_bigreg_sq's own byte-for-byte (wh and __half are both 2
// bytes/element) -- isolating the cost of A's packed-storage decode
// alone.
// Also declaring the native fp16 kernel this is architecturally based on,
// so this benchmark can report a true same-process/same-node/same-
// thermal-state "% of native" number directly, instead of comparing
// against a different session's documented numbers.
namespace tomos {
namespace cuda {
namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq(TransposeType transA, TransposeType transB, double alpha,
                                          size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                          const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                     size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                     const __half* B, size_t ld_b, double beta, __half* C,
                                                     size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                       size_t m, size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                       const __half* B, size_t ld_b, double beta, __half* C,
                                                       size_t ld_c); }  // namespace experimental

void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                double alpha, size_t m, size_t k, size_t n,
                                                                const uint16_t* wh, size_t ld_a, const __half* B,
                                                                size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                       double alpha, size_t m, size_t k, size_t n,
                                                                       const uint16_t* wh, size_t ld_a,
                                                                       const __half* B, size_t ld_b, double beta,
                                                                       __half* C, size_t ld_c); }  // namespace experimental

void mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k,
                                                                           size_t n, const uint16_t* wh, size_t ld_a,
                                                                           const __half* B, size_t ld_b, double beta,
                                                                           __half* C, size_t ld_c);
namespace experimental { void mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, const uint16_t* wh,
    size_t ld_a, const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp(TransposeType transA,
                                                                             TransposeType transB, double alpha,
                                                                             size_t m, size_t k, size_t n,
                                                                             const uint16_t* wh, size_t ld_a,
                                                                             const __half* B, size_t ld_b,
                                                                             double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp(TransposeType transA,
                                                                             TransposeType transB, double alpha,
                                                                             size_t m, size_t k, size_t n,
                                                                             const uint16_t* wh, size_t ld_a,
                                                                             const __half* B, size_t ld_b,
                                                                             double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp(TransposeType transA, TransposeType transB, double alpha,
                                                         size_t m, size_t k, size_t n, const uint16_t* wh,
                                                         size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                         __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void fp32_16_compute_chunk_dirty(size_t k, size_t ld_a, const uint16_t* wh, unsigned char* chunk_dirty); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, const uint16_t* wh,
                                                            size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                            __half* C, size_t ld_c,
                                                            const unsigned char* chunk_dirty); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized(TransposeType transA, TransposeType transB, double alpha,
                                                           size_t m, size_t k, size_t n, const uint16_t* wh,
                                                           size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                           __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty(TransposeType transA, TransposeType transB, double alpha,
                                                           size_t m, size_t k, size_t n, const uint16_t* wh,
                                                           size_t ld_a, const __half* B, size_t ld_b, double beta,
                                                           __half* C, size_t ld_c,
                                                           const unsigned char* chunk_dirty); }  // namespace experimental

namespace experimental { void mm_fp32_16_pure_half_wide_bigreg_sq_dualocc(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, const uint16_t* wh, size_t ld_a,
                                                  const __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
    TransposeType transA, TransposeType transB, double alpha, size_t m, size_t k, size_t n, __half* A, size_t ld_a,
    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c);
}  // namespace cuda
}  // namespace tomos

using namespace tomos;

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__global__ void gen_fp32_16_kernel(size_t n, uint16_t* wh, uint16_t* wr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fp32_16_split(pseudo_rand(i), wh[i], wr[i]);
}

__global__ void gen_float_kernel(size_t n, float* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pseudo_rand(i);
}

// Materializes a plain __half array from wh alone (fp32_16_half_only,
// never touching wr), giving CUTLASS/cuBLAS the same "fp16-level
// accuracy, decoded once" A that the tomos kernels themselves compute
// with -- a fair "same working precision" comparison.
__global__ void wh_to_half_kernel(size_t n, const uint16_t* wh, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp32_16_half_only(wh[i]);
}

__global__ void float_to_half_kernel(size_t n, const float* in, __half* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<__half>(in[i]);
}

void gen_fp32_16(size_t n, uint16_t* wh, uint16_t* wr) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_fp32_16_kernel<<<blocks, threads>>>(n, wh, wr);
}
void gen_float(size_t n, float* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_float_kernel<<<blocks, threads>>>(n, out);
}
void wh_to_half(size_t n, const uint16_t* wh, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    wh_to_half_kernel<<<blocks, threads>>>(n, wh, out);
}
void float_to_half(size_t n, const float* in, __half* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    float_to_half_kernel<<<blocks, threads>>>(n, in, out);
}

}  // namespace

double time_call(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms) / reps;
}

int main(int argc, char** argv) {
    // At N=65536 the packed wh/wr pair is 2 GiB each (uint16_t elements,
    // half the footprint of a plain fp32 A), plus B/C float (8 GiB each)
    // and the __half A/B/C references (2 GiB each) -- wr is freed right
    // after decoding A_half, so peak usage stays well under a 40/80 GiB
    // A100. This is a pure throughput benchmark (no correctness check),
    // so a single shared C_float across the tomos kernels and a single
    // shared C_half across CUTLASS/cuBLAS is fine, matching the original
    // wide_bigreg_family_vs_cutlass_ref.cu's single-C-buffer pattern.
    std::vector<size_t> sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    if (argc > 1) {
        sizes.clear();
        sizes.push_back(static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== FP32_16 pure_half: wide_bigreg_sq / tile_dirty / inline_dirty / inline_dirty_fastk_precomp2_serp / inline_dirty_fastk_precomp2_serp_blockvote / inline_dirty_fastk_precomp2_serp_alwaysdecode / inline_dirty_fastk_precomp2_serp_perthread / inline_dirty_fastk_precomp2_serp_perthread_leanreg / inline_dirty_fastk_precomp2_serp_perrep / gemm_materialize_fastk_precomp2_serp / gemm_materialize_cached_fastk_precomp2_serp / gemm_materialize_cached_vec_fastk_precomp2_serp / gemm_materialize_cached_vec_overlap_fastk_precomp2_serp / gemm_materialize_cached_vec64_fastk_precomp2_serp / gemm_materialize_cached_dirty_fastk_precomp2_serp / gemm_auto_fastk_precomp2_serp / precomputed_dirty / warp_specialized / prefetched_dirty / dualocc / wmma_bigbk_pad / wmma_hugebk vs real CUTLASS vs cuBLAS ===\n");
    for (size_t n : sizes) {
        uint16_t *wh, *wr;
        float *B, *C;
        cutlass::half_t *A_ct, *B_ct, *C_ct;
        unsigned char* chunk_dirty;
        bool ok = cudaMalloc(&wh, n * n * sizeof(uint16_t)) == cudaSuccess &&
                 cudaMalloc(&wr, n * n * sizeof(uint16_t)) == cudaSuccess &&
                 cudaMalloc(&B, n * n * sizeof(float)) == cudaSuccess &&
                 cudaMalloc(&C, n * n * sizeof(float)) == cudaSuccess &&
                 cudaMalloc(&A_ct, n * n * sizeof(cutlass::half_t)) == cudaSuccess &&
                 cudaMalloc(&B_ct, n * n * sizeof(cutlass::half_t)) == cudaSuccess &&
                 cudaMalloc(&C_ct, n * n * sizeof(cutlass::half_t)) == cudaSuccess &&
                 cudaMalloc(&chunk_dirty, (n / 8) * n * sizeof(unsigned char)) == cudaSuccess;
        if (!ok) {
            std::printf("  N=%-9zu SKIPPED (out of device memory)\n", n);
            cudaFree(wh); cudaFree(wr); cudaFree(B); cudaFree(C);
            cudaFree(A_ct); cudaFree(B_ct); cudaFree(C_ct); cudaFree(chunk_dirty);
            cudaGetLastError();
            continue;
        }

        gen_fp32_16(n * n, wh, wr);
        __half* A_half = reinterpret_cast<__half*>(A_ct);
        wh_to_half(n * n, wh, A_half);
        cudaFree(wr);

        gen_float(n * n, B);
        __half* B_half = reinterpret_cast<__half*>(B_ct);
        float_to_half(n * n, B, B_half);

        cudaMemset(C, 0, n * n * sizeof(float));
        cudaMemset(C_ct, 0, n * n * sizeof(cutlass::half_t));
        cudaDeviceSynchronize();

        cutlass::gemm::GemmCoord problem_size(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        ElementAccumulator alpha(1.7f), beta(0.3f);
        int lda = static_cast<int>(n), ldb = static_cast<int>(n), ldc = static_cast<int>(n);

        Gemm gemm_op;
        typename Gemm::Arguments args(
            problem_size,
            {A_ct, lda}, {B_ct, ldb}, {C_ct, ldc}, {C_ct, ldc},
            {alpha, beta});

        size_t workspace_size = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (cutlass can_implement failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(wh); cudaFree(B); cudaFree(C);
            cudaFree(A_ct); cudaFree(B_ct); cudaFree(C_ct); cudaFree(chunk_dirty);
            if (workspace) cudaFree(workspace);
            continue;
        }
        status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (cutlass initialize failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(wh); cudaFree(B); cudaFree(C);
            cudaFree(A_ct); cudaFree(B_ct); cudaFree(C_ct); cudaFree(chunk_dirty);
            if (workspace) cudaFree(workspace);
            continue;
        }

        int reps = (n <= 2048) ? 20 : (n <= 8192 ? 5 : 2);
        double ms_cutlass = time_call(start, stop, [&]() { gemm_op(); }, reps);

        __half alpha_h = static_cast<__half>(1.7f), beta_h = static_cast<__half>(0.3f);
        double ms_cublas = time_call(start, stop, [&]() {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                       &alpha_h, A_half, lda, B_half, ldb, &beta_h, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);

        double ms_wide_bigreg_sq = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same wh (this benchmark's ordinary pseudo-random data, not
        // forced clean) -- shows this method's real-world benefit: most
        // (block_row, k_tile) tiles come back clean and skip decode
        // entirely, while any tile a stray near-zero/degraded value
        // lands in still decodes correctly. See mm_cuda_kernels.cu, next
        // to mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty, for why a
        // per-tile check (not a single whole-array flag) is what makes
        // that payoff actually happen on data like this.
        double ms_tile_dirty = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_tile_dirty(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same wh again -- no separate precompute kernel or tile_dirty
        // buffer here, the clean/dirty decision is made entirely inline
        // per thread per 8-element chunk (see mm_fp32_16_pure_half_wide_
        // bigreg_sq_inline_dirty_kernel's decode_a_tile_inline_dirty
        // comment in mm_cuda_kernels.cu).
        double ms_inline_dirty = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // inline_dirty's own decode step (unchanged) layered under the
        // native fp16 family's own gemm_fp16_mma_ptx_wide_bigreg_sq_
        // deep4_iter_epivec_acongruous_fastk_precomp2_serp: real CUTLASS
        // congruous A-addressing, precomputed per-thread/per-fragment
        // address tables, and real CUTLASS's own serpentine accumulate
        // order (see that kernel's own comment in mm_cuda_kernels.cu).
        // Same wh (ordinary, possibly-mixed-control data) as every other
        // method above.
        double ms_inline_dirty_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same base kernel, with decode_a_tile's per-warp __any_sync
        // replaced by a block-wide __syncthreads_or vote -- motivated by
        // real ncu profiling showing fastk_precomp2_serp's own
        // stall_barrier at 5.5x native's (15.18% vs 2.75% at N=4096).
        // MEASURED: a clean regression, ~20-24% slower than fastk_
        // precomp2_serp at every size -- see its own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_inline_dirty_fastk_precomp2_serp_blockvote = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_blockvote(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same base kernel again, but instead of relocating the dirty
        // check (blockvote's own attempt, which regressed) this removes
        // it entirely. MEASURED: an even larger regression than
        // blockvote's own (~31-36% slower than fastk_precomp2_serp at
        // N=2048-32768) despite compiling to fewer registers (236/233
        // vs 253/253) -- the per-warp check was already skipping decode
        // on most tiles, and losing that work-avoidance cost more than
        // the stall it removed. See its own comment in mm_cuda_kernels.cu
        // for the full derivation.
        double ms_inline_dirty_fastk_precomp2_serp_alwaysdecode = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_alwaysdecode(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Built on alwaysdecode's own real profiling data (N=4096):
        // that profile confirmed __any_sync's own "whole warp rewrites
        // if any lane is dirty" behavior inflates STS traffic
        // (stall_short_sb 2.37% -> 9.27% when the check was removed
        // entirely). This kernel drops __any_sync but keeps the check,
        // branching each thread on its own dirty bit alone -- no
        // cross-lane vote, so one dirty element no longer forces the
        // whole warp to rewrite. Compiles to 253 registers, matching
        // fastk_precomp2_serp's own exactly. MEASURED: essentially a
        // statistical tie with fastk_precomp2_serp at every size (within
        // +-2%, N=2048-32768), a small real edge at N=8192 (+2.15%, new
        // best-of-family there), no regression anywhere -- unlike
        // blockvote/alwaysdecode, which both regressed.
        double ms_inline_dirty_fastk_precomp2_serp_perthread = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Byte-for-byte identical to perthread above, except decode_a_
        // tile no longer holds the loaded tile (raw[kAReps]/idx[kAReps])
        // across its own dirty branch -- a single scalar raw/idx dies
        // each check iteration instead, and the (rare, ~0.006% of
        // elements per this codebase's own measured data distribution)
        // dirty branch re-reads from shared memory rather than reusing a
        // held copy. Tests whether not holding the tile in flight closes
        // perthread's own 17-register gap to native (253 vs 236, both
        // <false> variants, the only ones actually exercised since every
        // benchmark here uses beta!=0). COMPILE-ONLY CHECK already
        // answered no: this kernel's own <false> variant compiles to 251
        // registers, only 2 fewer than perthread's 253 -- #pragma unroll
        // fully unrolls both loops regardless of array-vs-scalar source
        // structure, so ptxas's own SSA-based liveness analysis was
        // likely already finding the same answer either way. MEASURED: a
        // real, consistent regression vs perthread across the full
        // square sweep (-3.1% at N=2048/4096, -7.1% at N=8192, -2.2% at
        // N=32768; N=1024/16384 within noise) -- not just "no win," an
        // actual loss at 4 of 6 sizes. Both holding the tile (perthread)
        // and not holding it (this kernel) underperform materialize --
        // a structural limit, not a fixable register-pressure problem.
        // NOT wired into gemm_auto_fastk_precomp2_serp. See its own
        // comment in mm_cuda_kernels.cu for the full breakdown.
        double ms_inline_dirty_fastk_precomp2_serp_perthread_leanreg = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perthread_leanreg(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Built directly on perthread's own real (non-regressing)
        // measurement: narrows the check's blast radius one step
        // further, from per-thread to per-rep. MEASURED: a mild,
        // consistent regression (~2-4% slower than perthread/fastk_
        // precomp2_serp, N=2048/8192/16384/32768) despite fewer
        // registers (236/234 vs perthread's own 253/253) -- the extra
        // per-rep branch overhead outweighs the further-reduced STS
        // traffic. perthread remains the right stopping point here.
        double ms_inline_dirty_fastk_precomp2_serp_perrep = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_acongruous_fastk_precomp2_serp_perrep(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Structurally different from every decode-check sibling above:
        // decodes wh into a scratch buffer via a separate, trivial
        // prepass, then calls the native fp16 kernel completely
        // unmodified -- ported from a strategy already proven to win
        // outright in this codebase's own fp32-compute tier (see its
        // own comment in mm_cuda_kernels.cu for the full derivation).
        // MEASURED: a clean, MONOTONIC crossover on N -- loses badly at
        // small N (0.22x at N=1024, 0.41x at N=2048: two extra kernel
        // launches plus chunking overhead dominate) but wins
        // increasingly at large N (1.06x at N=8192, 1.18x at N=32768,
        // reaching 99.8% of native fp16 there) -- by far the best result
        // this whole family has produced, once N is large enough to
        // amortize the decode prepass's O(M*K) cost against the GEMM's
        // own O(M*K*N).
        double ms_gemm_materialize_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same as above, but the scratch buffer is a cached, process-
        // lifetime allocation instead of a fresh cudaMalloc/cudaFree on
        // every call -- targets the plain version's own small-N
        // weakness specifically (cudaMalloc's own tens-to-low-hundreds-
        // of-microsecond cost is comparable to or larger than the whole
        // GEMM at N=1024/2048). See its own comment in mm_cuda_
        // kernels.cu for the full derivation and thread-safety caveat.
        double ms_gemm_materialize_cached_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same as above plus a vectorized (4-rows-per-thread) decode
        // pass, ported from this codebase's own fp32-compute tier --
        // targets the remaining N=2048-8192 gap to native fp16 (14-18%
        // measured with the cached-but-scalar kernel), where the decode
        // pass's own O(M*K) execution time is still a meaningful
        // fraction of the O(M*K*N) GEMM's own time. See its own comment
        // in mm_cuda_kernels.cu for the full derivation.
        double ms_gemm_materialize_cached_vec_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::mm_fp32_16_pure_half_gemm_materialize_cached_vec_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same as above, but overlaps chunk c+2's decode with chunk c's
        // GEMM via a second stream and two ping-ponged scratch buffers
        // instead of running every chunk's decode-then-GEMM pair back to
        // back. Only structurally able to help once the row-chunk loop
        // produces 2+ chunks (N >= 8192 in this square sweep, m >
        // kMmFp32_16PureHalfMaterializeChunkRows) -- real ncu profiling
        // at N=2048 (a single-chunk case) showed the decode kernel's own
        // duration already accounts for the entire measured gap to
        // native there, with no launch/dependency bubble to hide via
        // overlap. See its own comment in mm_cuda_kernels.cu for the
        // full derivation and hazard-handling reasoning. MEASURED:
        // correct, but not a win anywhere -- a clear -2.4% loss at
        // N=8192 (exactly 2 chunks, the simplest case for the mechanism
        // to pay off), a wash at N=16384/32768 (4/8 chunks), and a small
        // loss at single-chunk sizes from the extra event/wait
        // bookkeeping alone. See its own comment in mm_cuda_kernels.cu
        // for the likely cause (native GEMM already saturates every SM,
        // leaving no spare occupancy for real concurrent execution) and
        // the full per-size breakdown. NOT wired into gemm_auto_fastk_
        // precomp2_serp.
        double ms_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same decode as materialize_cached_vec, but the write is a
        // single 64-bit integer store instead of two 32-bit ones. Found
        // by inspecting materialize_chunk_vec_kernel's own SASS (cuobjdump
        // --dump-sass on the built library): its read was already a
        // genuine LDG.E.64 as intended, but its write came out as two
        // separate 32-bit STG.E instructions instead of one STG.E.64 (the
        // four decoded halves packed into two non-adjacent 32-bit
        // registers, so the compiler couldn't fold them). This version
        // builds the output as a single 64-bit integer via explicit
        // shifts/ORs on __half_as_ushort's own bit pattern, confirmed via
        // SASS to compile to one real STG.E.64. Only changes instruction
        // ISSUE count, not memory transactions -- the addresses were
        // already fully coalesced across a warp either way. MEASURED: no
        // real effect -- every delta vs materialize_cached_vec across the
        // full square sweep (-0.2% to +0.9%) fell within this file's own
        // ~1% run-to-run noise band, confirming there was nothing here for
        // an instruction-count-only change to actually fix. NOT wired into
        // gemm_auto_fastk_precomp2_serp. See its own comment in mm_cuda_
        // kernels.cu for the full SASS comparison and per-size breakdown.
        double ms_gemm_materialize_cached_vec64_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_vec64_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Hoists fp32_16_half_only's own per-element control-bit check to
        // thread granularity (4 elements/thread): when all 4 are
        // control=0 ("clean"), skips calling fp32_16_half_only 4 times
        // and does one raw ushort4->uint2 bit copy instead, since that
        // function's own control=0 fast path IS exactly a bitcast of wh.
        // Motivated by a host-only measurement (no GPU execution) of this
        // file's own pseudo_rand test data: across 200M samples,
        // 99.99355% have control=0. Same idea this codebase's own fused
        // inline_dirty family already exploits, applied to the separate
        // materialize decode kernel instead. Does not change total memory
        // traffic either way, so may show the same "no measurable effect"
        // vec64 did -- built to test that against the real data
        // distribution rather than assume it. MEASURED: correct, but no
        // clear win -- every delta vs materialize_cached_vec64 across the
        // full square sweep fell within this file's own ~1-2% run-to-run
        // noise band. Third distinct decode-kernel-internal optimization
        // tested (after vec_overlap and vec64) with no measurable effect,
        // confirming decode's cost is a bandwidth floor plus fixed launch
        // latency, unaffected by what happens inside the kernel. NOT
        // wired into gemm_auto_fastk_precomp2_serp. See its own comment
        // in mm_cuda_kernels.cu for the full derivation and breakdown.
        double ms_gemm_materialize_cached_dirty_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_materialize_cached_dirty_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Size-threshold dispatcher combining perthread (N < 2048) and
        // materialize_cached_vec (N >= 2048) -- revised twice: first
        // down from an original N >= 8192 threshold against plain
        // (non-cached) materialize once the cached kernel's own
        // measurement showed plain materialize's small-N numbers were
        // dominated by repeated cudaMalloc/cudaFree overhead, not real
        // GEMM cost; then to route to the vectorized-decode kernel once
        // IT measured strictly better than the scalar cached kernel at
        // every size -- see its own comment in mm_cuda_kernels.cu for
        // the full derivation, including a cautionary precedent from
        // this codebase's own fp32-compute tier where a similar
        // dispatcher was dropped after its crossover turned out
        // non-monotonic and GPU-dependent (this one's own crossover is
        // cleanly monotonic and asymptotically grounded, unlike that
        // one).
        double ms_gemm_auto_fastk_precomp2_serp = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_gemm_auto_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Precompute cost is reported separately (ms_precompute), NOT
        // folded into precomputed_dirty's own GFlop/s below -- same
        // treatment as wh/wr generation itself, which isn't counted in
        // ANY of these numbers either. This only pays off amortized
        // across reuse of the same packed A; see mm_fp32_16_pure_half_
        // wide_bigreg_sq_precomputed_dirty's own comment.
        double ms_precompute = time_call(start, stop, [&]() {
            cuda::experimental::fp32_16_compute_chunk_dirty(n, lda, wh, chunk_dirty);
        }, reps);
        double ms_precomputed_dirty = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_precomputed_dirty(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc, chunk_dirty);
        }, reps);
        // Same signature as inline_dirty (no extra buffer needed) -- only
        // warps 0-3 decode (covering the whole tile between them), warps
        // 4-7 skip decode entirely (see mm_fp32_16_pure_half_wide_bigreg_
        // sq_warp_specialized_kernel's own comment in mm_cuda_kernels.cu
        // for the full derivation and the honest caveat that this one's
        // direction wasn't predictable going in).
        double ms_warp_specialized = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_warp_specialized(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Same chunk_dirty as precomputed_dirty (reused, already computed
        // above) -- each thread's own needed bytes are prefetched into
        // shared memory alongside wh itself, instead of read from global
        // memory at check time (see mm_fp32_16_pure_half_wide_bigreg_sq_
        // prefetched_dirty_kernel's own comment in mm_cuda_kernels.cu).
        double ms_prefetched_dirty = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_prefetched_dirty(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc, chunk_dirty);
        }, reps);
        // Same signature as inline_dirty (no extra buffer needed) -- A/K-
        // side load/decode/compute copied verbatim from inline_dirty, only
        // the B/N tile is cut to a quarter, targeting __launch_bounds__(
        // ..., 2) instead of inline_dirty's (..., 1) (see mm_fp32_16_pure_
        // half_wide_bigreg_sq_dualocc_kernel's own comment in mm_cuda_
        // kernels.cu -- includes the ncu occupancy numbers motivating this
        // and a directly relevant cautionary precedent already in this
        // file for a related trade that lost).
        double ms_dualocc = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wide_bigreg_sq_dualocc(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, wh, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        // Native fp16 counterpart this kernel is architecturally based
        // on, in the SAME process/node/thermal state -- gives a true
        // same-run "% of native" number instead of comparing against a
        // different session's documented numbers. Uses this family's own
        // best native kernel (deep4_iter_epivec_acongruous_fastk_
        // precomp2_serp), not plain gemm_fp16_mma_ptx_wide_bigreg_sq --
        // that stale comparison (from before this whole optimization
        // chain existed) understated the real gap to native by making it
        // look like near-parity against a kernel that was never the
        // actual best.
        double ms_native = time_call(start, stop, [&]() {
            cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                n, n, n, A_half, lda, B_half, ldb, 0.3, reinterpret_cast<__half*>(C_ct), ldc);
        }, reps);
        double ms_bigbk_pad = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_bigbk_pad(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                {n, n}, wh, lda, {n, n}, B, ldb, 0.3, C, ldc);
        }, reps);
        double ms_hugebk = time_call(start, stop, [&]() {
            cuda::experimental::mm_fp32_16_pure_half_wmma_cpasync_vec_hugebk(
                NO_TRANSPOSE, NO_TRANSPOSE, 1.7,
                {n, n}, wh, lda, {n, n}, B, ldb, 0.3, C, ldc);
        }, reps);

        double flops = 2.0 * static_cast<double>(n) * n * n;
        double g_cutlass = flops / ms_cutlass / 1e6;
        double g_cublas = flops / ms_cublas / 1e6;
        double g_wide_bigreg_sq = flops / ms_wide_bigreg_sq / 1e6;
        double g_tile_dirty = flops / ms_tile_dirty / 1e6;
        double g_inline_dirty = flops / ms_inline_dirty / 1e6;
        double g_inline_dirty_fastk_precomp2_serp = flops / ms_inline_dirty_fastk_precomp2_serp / 1e6;
        double g_inline_dirty_fastk_precomp2_serp_blockvote =
            flops / ms_inline_dirty_fastk_precomp2_serp_blockvote / 1e6;
        double g_inline_dirty_fastk_precomp2_serp_alwaysdecode =
            flops / ms_inline_dirty_fastk_precomp2_serp_alwaysdecode / 1e6;
        double g_inline_dirty_fastk_precomp2_serp_perthread =
            flops / ms_inline_dirty_fastk_precomp2_serp_perthread / 1e6;
        double g_inline_dirty_fastk_precomp2_serp_perthread_leanreg =
            flops / ms_inline_dirty_fastk_precomp2_serp_perthread_leanreg / 1e6;
        double g_inline_dirty_fastk_precomp2_serp_perrep =
            flops / ms_inline_dirty_fastk_precomp2_serp_perrep / 1e6;
        double g_gemm_materialize_fastk_precomp2_serp = flops / ms_gemm_materialize_fastk_precomp2_serp / 1e6;
        double g_gemm_materialize_cached_fastk_precomp2_serp =
            flops / ms_gemm_materialize_cached_fastk_precomp2_serp / 1e6;
        double g_gemm_materialize_cached_vec_fastk_precomp2_serp =
            flops / ms_gemm_materialize_cached_vec_fastk_precomp2_serp / 1e6;
        double g_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp =
            flops / ms_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp / 1e6;
        double g_gemm_materialize_cached_vec64_fastk_precomp2_serp =
            flops / ms_gemm_materialize_cached_vec64_fastk_precomp2_serp / 1e6;
        double g_gemm_materialize_cached_dirty_fastk_precomp2_serp =
            flops / ms_gemm_materialize_cached_dirty_fastk_precomp2_serp / 1e6;
        double g_gemm_auto_fastk_precomp2_serp = flops / ms_gemm_auto_fastk_precomp2_serp / 1e6;
        double g_precomputed_dirty = flops / ms_precomputed_dirty / 1e6;
        double g_warp_specialized = flops / ms_warp_specialized / 1e6;
        double g_prefetched_dirty = flops / ms_prefetched_dirty / 1e6;
        double g_dualocc = flops / ms_dualocc / 1e6;
        double g_native = flops / ms_native / 1e6;
        double g_bigbk_pad = flops / ms_bigbk_pad / 1e6;
        double g_hugebk = flops / ms_hugebk / 1e6;

        struct Named { const char* name; double gflops; };
        Named tomos_results[] = {
            {"wide_bigreg_sq", g_wide_bigreg_sq},
            {"wide_bigreg_sq_tile_dirty", g_tile_dirty},
            {"wide_bigreg_sq_inline_dirty", g_inline_dirty},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp", g_inline_dirty_fastk_precomp2_serp},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp_blockvote",
             g_inline_dirty_fastk_precomp2_serp_blockvote},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp_alwaysdecode",
             g_inline_dirty_fastk_precomp2_serp_alwaysdecode},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp_perthread",
             g_inline_dirty_fastk_precomp2_serp_perthread},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp_perthread_leanreg",
             g_inline_dirty_fastk_precomp2_serp_perthread_leanreg},
            {"wide_bigreg_sq_inline_dirty_fastk_precomp2_serp_perrep",
             g_inline_dirty_fastk_precomp2_serp_perrep},
            {"gemm_materialize_fastk_precomp2_serp", g_gemm_materialize_fastk_precomp2_serp},
            {"gemm_materialize_cached_fastk_precomp2_serp", g_gemm_materialize_cached_fastk_precomp2_serp},
            {"gemm_materialize_cached_vec_fastk_precomp2_serp", g_gemm_materialize_cached_vec_fastk_precomp2_serp},
            {"gemm_materialize_cached_vec_overlap_fastk_precomp2_serp",
             g_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp},
            {"gemm_materialize_cached_vec64_fastk_precomp2_serp", g_gemm_materialize_cached_vec64_fastk_precomp2_serp},
            {"gemm_materialize_cached_dirty_fastk_precomp2_serp",
             g_gemm_materialize_cached_dirty_fastk_precomp2_serp},
            {"gemm_auto_fastk_precomp2_serp", g_gemm_auto_fastk_precomp2_serp},
            {"wide_bigreg_sq_precomputed_dirty", g_precomputed_dirty},
            {"wide_bigreg_sq_warp_specialized", g_warp_specialized},
            {"wide_bigreg_sq_prefetched_dirty", g_prefetched_dirty},
            {"wide_bigreg_sq_dualocc", g_dualocc},
            {"wmma_bigbk_pad", g_bigbk_pad},
            {"wmma_hugebk", g_hugebk},
        };
        Named* best = &tomos_results[0];
        for (auto& r : tomos_results) if (r.gflops > best->gflops) best = &r;

        std::printf("  N=%zu\n", n);
        std::printf("    wide_bigreg_sq  = %9.3fms (%10.1f GFlop/s)\n", ms_wide_bigreg_sq, g_wide_bigreg_sq);
        std::printf("    tile_dirty      = %9.3fms (%10.1f GFlop/s)\n", ms_tile_dirty, g_tile_dirty);
        std::printf("    inline_dirty    = %9.3fms (%10.1f GFlop/s)\n", ms_inline_dirty, g_inline_dirty);
        std::printf("    inline_dirty_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp, g_inline_dirty_fastk_precomp2_serp);
        std::printf("    inline_dirty_fastk_precomp2_serp_blockvote = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp_blockvote, g_inline_dirty_fastk_precomp2_serp_blockvote);
        std::printf("    inline_dirty_fastk_precomp2_serp_alwaysdecode = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp_alwaysdecode, g_inline_dirty_fastk_precomp2_serp_alwaysdecode);
        std::printf("    inline_dirty_fastk_precomp2_serp_perthread = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp_perthread, g_inline_dirty_fastk_precomp2_serp_perthread);
        std::printf("    inline_dirty_fastk_precomp2_serp_perthread_leanreg = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp_perthread_leanreg,
                   g_inline_dirty_fastk_precomp2_serp_perthread_leanreg);
        std::printf("    inline_dirty_fastk_precomp2_serp_perrep = %9.3fms (%10.1f GFlop/s)\n",
                   ms_inline_dirty_fastk_precomp2_serp_perrep, g_inline_dirty_fastk_precomp2_serp_perrep);
        std::printf("    gemm_materialize_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_fastk_precomp2_serp, g_gemm_materialize_fastk_precomp2_serp);
        std::printf("    gemm_materialize_cached_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_cached_fastk_precomp2_serp, g_gemm_materialize_cached_fastk_precomp2_serp);
        std::printf("    gemm_materialize_cached_vec_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_cached_vec_fastk_precomp2_serp, g_gemm_materialize_cached_vec_fastk_precomp2_serp);
        std::printf("    gemm_materialize_cached_vec_overlap_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp,
                   g_gemm_materialize_cached_vec_overlap_fastk_precomp2_serp);
        std::printf("    gemm_materialize_cached_vec64_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_cached_vec64_fastk_precomp2_serp,
                   g_gemm_materialize_cached_vec64_fastk_precomp2_serp);
        std::printf("    gemm_materialize_cached_dirty_fastk_precomp2_serp = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_materialize_cached_dirty_fastk_precomp2_serp,
                   g_gemm_materialize_cached_dirty_fastk_precomp2_serp);
        std::printf("    gemm_auto_fastk_precomp2_serp  = %9.3fms (%10.1f GFlop/s)\n",
                   ms_gemm_auto_fastk_precomp2_serp, g_gemm_auto_fastk_precomp2_serp);
        std::printf("    precomputed_dirty=%9.3fms (%10.1f GFlop/s)  [+%.3fms one-time precompute, not counted above]\n",
                   ms_precomputed_dirty, g_precomputed_dirty, ms_precompute);
        std::printf("    warp_specialized= %9.3fms (%10.1f GFlop/s)\n", ms_warp_specialized, g_warp_specialized);
        std::printf("    prefetched_dirty= %9.3fms (%10.1f GFlop/s)  [+%.3fms one-time precompute, not counted above]\n",
                   ms_prefetched_dirty, g_prefetched_dirty, ms_precompute);
        std::printf("    dualocc         = %9.3fms (%10.1f GFlop/s)\n", ms_dualocc, g_dualocc);
        std::printf("    native (fp16)   = %9.3fms (%10.1f GFlop/s)\n", ms_native, g_native);
        std::printf("    wmma_bigbk_pad  = %9.3fms (%10.1f GFlop/s)\n", ms_bigbk_pad, g_bigbk_pad);
        std::printf("    wmma_hugebk     = %9.3fms (%10.1f GFlop/s)\n", ms_hugebk, g_hugebk);
        std::printf("    cutlass (real)  = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass, g_cutlass);
        std::printf("    cublas          = %9.3fms (%10.1f GFlop/s)\n", ms_cublas, g_cublas);
        std::printf("    best of ours: %s (%.1f GFlop/s) -- %.3fx cublas, %.3fx real cutlass, %.3fx native fp16\n",
                   best->name, best->gflops, best->gflops / g_cublas, best->gflops / g_cutlass,
                   best->gflops / g_native);

        cudaFree(wh); cudaFree(B); cudaFree(C);
        cudaFree(A_ct); cudaFree(B_ct); cudaFree(C_ct); cudaFree(chunk_dirty);
        if (workspace) cudaFree(workspace);
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

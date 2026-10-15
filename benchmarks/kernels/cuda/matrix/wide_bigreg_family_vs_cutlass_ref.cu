// Focused benchmark: just the three cp.async-B-fixed kernels (wide_
// bigreg_sq, wide_bigreg, cutlass_config) against real CUTLASS's own
// device::Gemm and cuBLAS's Hgemm, all in one run/process so the
// numbers share the same GPU clock/thermal/contention state. This is
// deliberately narrower than verify_mm_fp16_gemm's full ~50-kernel
// sweep -- a quick way to check just this specific family's standing
// against the two external references without wading through the rest.
//
// Same problem shape/methodology as verify_mm_fp16_gemm.cu and
// cutlass_fp16_gemm_ref.cu: square NxN, column-major, __half A/B/C
// (cutlass::half_t for the CUTLASS/data-gen path, bit-compatible),
// alpha=1.7/beta=0.3, cudaEvent_t timing.
//
// Deliberately NOT wired into tomos's CMake build (no CUTLASS
// dependency in the main project) -- built standalone via a direct
// nvcc invocation that also links against the already-built
// libtomos_cuda.so to call the three tomos kernels directly. See
// scripts/wide_bigreg_family_vs_cutlass_ref.sh.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

// Minimal, exact-match declarations for the three tomos kernels --
// avoids pulling mm_cuda_kernels.cuh's full ~1150-line declaration set
// (and its cusparse.h/dimensions.hpp/precision.hpp includes) into a
// standalone nvcc build; these three signatures are copied verbatim
// from mm_cuda_kernels.cuh and satisfied at link time by
// libtomos_cuda.so.
namespace tomos {
enum TransposeType { NO_TRANSPOSE, TRANSPOSE };
namespace cuda {
namespace experimental { void gemm_fp16_mma_ptx_cutlass_config(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg(TransposeType transA, TransposeType transB, double alpha,
                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                    __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq(TransposeType transA, TransposeType transB, double alpha,
                                       size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                       __half* B, size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                             size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                             double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                  double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a, __half* B,
                                                    size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                         size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                         __half* B, size_t ld_b, double beta, __half* C,
                                                         size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(TransposeType transA, TransposeType transB,
                                                               double alpha, size_t m, size_t k, size_t n, __half* A,
                                                               size_t ld_a, __half* B, size_t ld_b, double beta,
                                                               __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(TransposeType transA, TransposeType transB,
                                                                double alpha, size_t m, size_t k, size_t n, __half* A,
                                                                size_t ld_a, __half* B, size_t ld_b, double beta,
                                                                __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(TransposeType transA, TransposeType transB,
                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                  double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(TransposeType transA, TransposeType transB,
                                                                    double alpha, size_t m, size_t k, size_t n,
                                                                    __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                    double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(TransposeType transA, TransposeType transB,
                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                        double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(TransposeType transA, TransposeType transB,
                                                                     double alpha, size_t m, size_t k, size_t n,
                                                                     __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                     double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k, size_t n,
                                                                           __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                           double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(TransposeType transA, TransposeType transB,
                                                                           double alpha, size_t m, size_t k, size_t n,
                                                                           __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                           double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(TransposeType transA, TransposeType transB,
                                                                          double alpha, size_t m, size_t k, size_t n,
                                                                          __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                          double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(TransposeType transA, TransposeType transB,
                                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                  double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(TransposeType transA, TransposeType transB,
                                                                                   double alpha, size_t m, size_t k, size_t n,
                                                                                   __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                   double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(TransposeType transA, TransposeType transB,
                                                                                   double alpha, size_t m, size_t k, size_t n,
                                                                                   __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                   double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(TransposeType transA, TransposeType transB,
                                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                        double beta, __half* C, size_t ld_c); }  // namespace experimental

void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(TransposeType transA, TransposeType transB,
                                                                                        double alpha, size_t m, size_t k, size_t n,
                                                                                        __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                        double beta, __half* C, size_t ld_c);
namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(TransposeType transA, TransposeType transB,
                                                                                              double alpha, size_t m, size_t k, size_t n,
                                                                                              __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                                              double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(TransposeType transA, TransposeType transB,
                                                                 double alpha, size_t m, size_t k, size_t n,
                                                                 __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                 double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(TransposeType transA, TransposeType transB,
                                                                 double alpha, size_t m, size_t k, size_t n,
                                                                 __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                 double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(TransposeType transA, TransposeType transB,
                                                              double alpha, size_t m, size_t k, size_t n, __half* A,
                                                              size_t ld_a, __half* B, size_t ld_b, double beta,
                                                              __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                          size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                          __half* B, size_t ld_b, double beta, __half* C,
                                                          size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(TransposeType transA, TransposeType transB,
                                                                  double alpha, size_t m, size_t k, size_t n,
                                                                  __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                                  double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                            size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                            __half* B, size_t ld_b, double beta, __half* C,
                                                            size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                             __half* B, size_t ld_b, double beta, __half* C,
                                                             size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                             size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                             double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                  size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                  double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(TransposeType transA, TransposeType transB, double alpha,
                                                    size_t m, size_t k, size_t n, __half* A, size_t ld_a, __half* B,
                                                    size_t ld_b, double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                            size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                            double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(TransposeType transA, TransposeType transB, double alpha, size_t m,
                                                 size_t k, size_t n, __half* A, size_t ld_a, __half* B, size_t ld_b,
                                                 double beta, __half* C, size_t ld_c); }  // namespace experimental

namespace experimental { void gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(TransposeType transA, TransposeType transB, double alpha,
                                                             size_t m, size_t k, size_t n, __half* A, size_t ld_a,
                                                             __half* B, size_t ld_b, double beta, __half* C,
                                                             size_t ld_c); }  // namespace experimental

}  // namespace cuda
}  // namespace tomos

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

__global__ void gen_half_kernel(size_t n, cutlass::half_t* out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s = static_cast<uint32_t>(i) * 2654435761u + 1u;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;
    out[i] = static_cast<cutlass::half_t>(v);
}

void gen_half(size_t n, cutlass::half_t* out) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    gen_half_kernel<<<blocks, threads>>>(n, out);
}

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
    // At N=65536, each matrix is 8 GiB; with only 3 matrices allocated
    // (A, B, one shared C) that's 24 GiB -- fits a 40/80 GiB A100 with
    // room for CUTLASS's workspace and CUDA context overhead. This is
    // a pure throughput benchmark (no correctness check), so a single
    // shared C buffer across all five methods is fine -- each call just
    // reads/writes the same memory, same as the original cutlass_fp16_
    // gemm_ref.cu's single-C-buffer pattern.
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

    std::printf("=== wide_bigreg_sq / wide_bigreg / wide_bigreg_sq_deep4 / wide_bigreg_sq_deep4_iter / wide_bigreg_sq_deep4_epivec / wide_bigreg_sq_deep4_iter_epivec / wide_bigreg_sq_deep4_iter_epivec_align / wide_bigreg_sq_deep4_iter_epivec_pad280 / wide_bigreg_sq_deep4_iter_epivec_aswizzle / wide_bigreg_sq_deep4_iter_epivec_acongruous / wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe / wide_bigreg_sq_deep4_iter_epivec_acongruous2 / wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3 / wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2 / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3 / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp / wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm / wide_bigreg_sq_deep4_iter_epivec_stages3 / wide_bigreg_sq_deep4_iter_epivec_stages5 / wide_bigreg_sq_deep4_iter_epivec_pipe / wide_bigreg_sq_deep4_iter2_epivec / wide_bigreg_sq_deep4_iter_epivec_gridswap / wide_bigreg_cutlassdims_iter_epivec / wide_bigreg_cutlassdims3_iter_epivec / wide_bigreg_sq_deep5 / wide_bigreg_sq_deep4_occ2 / wide_bigreg_sq_deep4_bigmma / wide_bigreg_sq_bk64_reg2_iter_epivec / cutlass_config vs real CUTLASS vs cuBLAS ===\n");
    for (size_t n : sizes) {
        cutlass::half_t *A, *B, *C_ref;
        cudaMalloc(&A, n * n * sizeof(cutlass::half_t));
        cudaMalloc(&B, n * n * sizeof(cutlass::half_t));
        cudaMalloc(&C_ref, n * n * sizeof(cutlass::half_t));
        gen_half(n * n, A);
        gen_half(n * n, B);
        gen_half(n * n, C_ref);

        cutlass::gemm::GemmCoord problem_size(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        ElementAccumulator alpha(1.7f), beta(0.3f);
        int lda = static_cast<int>(n), ldb = static_cast<int>(n), ldc = static_cast<int>(n);

        Gemm gemm_op;
        typename Gemm::Arguments args(
            problem_size,
            {A, lda}, {B, ldb}, {C_ref, ldc}, {C_ref, ldc},
            {alpha, beta});

        size_t workspace_size = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (cutlass can_implement failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(A); cudaFree(B); cudaFree(C_ref);
            if (workspace) cudaFree(workspace);
            continue;
        }
        status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (cutlass initialize failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(A); cudaFree(B); cudaFree(C_ref);
            if (workspace) cudaFree(workspace);
            continue;
        }

        int reps = (n <= 2048) ? 20 : (n <= 8192 ? 5 : 2);
        double ms_cutlass = time_call(start, stop, [&]() { gemm_op(); }, reps);

        __half alpha_h = static_cast<__half>(1.7f), beta_h = static_cast<__half>(0.3f);
        __half* A_h = reinterpret_cast<__half*>(A);
        __half* B_h = reinterpret_cast<__half*>(B);
        // Single shared C buffer across all five methods (this is a
        // throughput-only benchmark, no correctness check, so it's fine
        // for beta*C to keep drifting across reps/methods -- same
        // pattern as the original cutlass_fp16_gemm_ref.cu). Keeps
        // memory use to 3 matrices total (A, B, C) instead of 7, which
        // matters at the larger sizes (8 GiB/matrix at N=65536).
        __half* C_h = reinterpret_cast<__half*>(C_ref);

        double ms_bigreg_sq = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        double ms_bigreg = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Same tile/signature as wide_bigreg_sq, just a 4-stage pipeline
        // instead of 3 -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
        // kernel's own comment in mm_cuda_kernels.cu for why this was
        // worth trying (already validated on the closest FP32_16 sibling)
        // and why it's a new, separate kernel rather than a modification.
        double ms_deep4 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Same tile/pipeline as deep4, but load_tile_async's address
        // arithmetic is restructured from per-call recomputation to
        // precompute-once/increment -- motivated by a real cuobjdump
        // --dump-sass comparison against real CUTLASS's own fp16
        // tensor-op kernel (matched register/occupancy, matched HMMA/
        // LDSM counts) showing deep4 issuing 3-4x more address-
        // computation instructions (IMAD/LEA) than CUTLASS for the same
        // actual compute. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
        // iter_kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation.
        double ms_deep4_iter = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Same mainloop as deep4 (untouched), epilogue rewritten to move
        // 8 halfs (float4, 16 bytes) at a time instead of deep4's own
        // scalar per-half stores -- motivated by a real ncu --set full
        // profile of deep4 flagging ~25-28% estimated speedup from
        // uncoalesced global/shared accesses that don't line up with the
        // (already-vectorized) mainloop. See gemm_fp16_mma_ptx_wide_
        // bigreg_sq_deep4_epivec_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation and alignment proof.
        double ms_deep4_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Combines deep4_iter's mainloop and deep4_epivec's epilogue in
        // one kernel -- tests whether the two independently-measured
        // fixes (deep4_iter: small win at large N; deep4_epivec: large
        // win at small/medium N) compose additively across the whole N
        // range. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
        // kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation.
        double ms_deep4_iter_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Re-pads A's own shared-memory tile to 32-byte-aligned rows
        // (272 elements instead of the shared BMPad's 264) -- motivated
        // by ncu showing deep4_iter_epivec still has the SAME
        // uncoalesced-access counts as deep4's own original profile.
        // Proven trade-off (doubles the bank-conflict period vs the
        // original pad's own carefully-chosen value) -- see gemm_fp16_
        // mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_align = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_align(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Same isolated A-shared-row-stride re-padding as _align's own,
        // different value (280 = 256+24 vs align's 272 = 256+16) --
        // completes the 4th untested bank-period residue after direct
        // ncu analysis showed A's own cp.async shared store is the
        // single hottest instruction for BOTH the excessive-global-
        // sectors AND the excessive-shared-wavefronts (27.7%) findings.
        // Genuinely empirical, not a predicted improvement -- see
        // gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280_
        // kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation.
        double ms_deep4_iter_epivec_pad280 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pad280(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Replaces A's shared-memory layout entirely with an XOR-
        // permuted-chunk swizzle (no padding), modeled on real CUTLASS's
        // own A operand layout (Crosswise, confirmed via cuobjdump --
        // CUTLASS swizzles A specifically, the opposite of this family's
        // own B-swizzle/A-pad assignment). ncu first confirmed A's own
        // ldmatrix.trans reads are already 0% excessive -- the problem
        // is 100% isolated to A's cp.async WRITE (58.62% excessive) --
        // so this targets that specifically. Correctness independently
        // verified (bijective, preserves cp.async's contiguous-
        // destination requirement); performance benefit is NOT provable
        // in advance (the standard bank-conflict model doesn't match
        // ncu's own reported LDGSTS conflict degree) -- genuinely
        // empirical, motivated by CUTLASS's real design. See gemm_fp16_
        // mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_aswizzle = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_aswizzle(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Replaces aswizzle's own self-invented scheme with real
        // CUTLASS's actual A formula, read directly from CUTLASS's own
        // header source -- corrected understanding: real CUTLASS does
        // NOT swizzle A (that's B's role, already matching this
        // family's own B indexing); A gets the simpler Congruous
        // scheme, M-contiguous like this family's own convention.
        // Correctness independently verified via a full computational
        // sweep. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
        // acongruous_kernel's own comment in mm_cuda_kernels.cu for the
        // full derivation.
        double ms_deep4_iter_epivec_acongruous = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Combines acongruous's own A-formula with pipe's own
        // interleaved second-MMA-block structure -- motivated by ncu
        // stall-reason analysis of acongruous's own profile showing
        // stall_short_sb (ldmatrix->mma latency) now dominates, the
        // exact dependency pipe's interleaving targets. Genuinely
        // empirical -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
        // epivec_acongruous_pipe_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_pipe = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to acongruous except a_congruous_idx's own kk-only
        // subexpressions are hoisted out of the per-rep/per-wm loops
        // that recompute them redundantly -- a pure algebraic factoring
        // (verified via a full Python sweep, zero mismatches), targeting
        // acongruous's own measured 9.6%-more-instructions overhead at
        // small N. ALGEBRAICALLY GUARANTEED correct, not empirical. See
        // gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2_
        // kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation.
        double ms_deep4_iter_epivec_acongruous2 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous2(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to acongruous except Stages=3 instead of 4 -- same
        // cheap, low-risk mechanical technique deep4_iter_epivec_
        // stages3 already used, retested here since acongruous's own
        // real-CUTLASS A-write pattern genuinely changes its latency
        // characteristics. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
        // iter_epivec_acongruous_stages3_kernel's own comment in
        // mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_stages3 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to acongruous except how the t+3 prefetch's 6 LDGSTS
        // are issued: one per wp iteration of the first MMA sub-block,
        // instead of all 6 bursted before load_a_frag/load_b_frags.
        // Motivated by directly comparing acongruous's own compiled SASS
        // mainloop against real CUTLASS's own: CUTLASS interleaves one
        // LDGSTS per independent HMMA; acongruous bursts all 6 first.
        // See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
        // acongruous_ldgsts_kernel's own comment in mm_cuda_kernels.cu
        // for the full derivation.
        double ms_deep4_iter_epivec_acongruous_ldgsts = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to acongruous except load_tile_async's boundary-
        // safety branch is deleted (compile-time-guaranteed unreachable
        // instead of runtime-checked) for k % BK == 0, which every N
        // tested here satisfies. Motivated by directly counting
        // instructions in acongruous's own compiled loop vs real
        // CUTLASS's own: identical HMMA/LDSM counts, but acongruous
        // executes 474 instructions/iteration vs CUTLASS's 140 --
        // concentrated in scalar/branch overhead from the runtime-
        // dispatched safe_a/safe_b parameters. Compile-only checks
        // (nvcc -Xptxas -v + cuobjdump on the object, no GPU execution)
        // confirm this kernel actually drops to 290 instructions/
        // iteration and 233-236 registers (down from acongruous's 254).
        // See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
        // acongruous_fastk_kernel's own comment in mm_cuda_kernels.cu
        // for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to fastk with ONE further change: A's own
        // addressing is precomputed ONCE per thread before the mainloop
        // (the small fixed set of (m,kk) combinations actually used)
        // instead of recomputed via its full divide/mod/xor chain at
        // every call site. Motivated by two PRIOR negative compile-only
        // results (block_row/block_col rematerialization, and
        // acongruous2's own kk-hoist reapplied on fastk) both producing
        // byte-for-byte identical compiled SASS -- ptxas already does
        // that class of intra-call CSE. This tests hoisting across the
        // ENTIRE outer K-loop instead, which ptxas can't prove on its
        // own given the loop's own asm volatile blocks -- confirmed via
        // the same compile-only technique to actually change the
        // compiled output: 275 instructions/iteration vs fastk's own
        // 290. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
        // acongruous_fastk_precomp_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to fastk_precomp with the SAME precomputed-offset
        // technique now also applied to B's own crosswise index, never
        // checked for this before -- B's own call sites turn out to be
        // even more invariant than A's own were (load_b_pair_raw always
        // calls it with kk=0, a compile-time literal). Confirmed via
        // the same compile-only technique to actually change the
        // compiled output: 247 instructions/iteration vs fastk_
        // precomp's own 275. See gemm_fp16_mma_ptx_wide_bigreg_sq_
        // deep4_iter_epivec_acongruous_fastk_precomp2_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp2 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Identical to fastk_precomp2 with ONE further change: how
        // __pipeline_wait_prior's own `outstanding` argument (provably
        // always 0, 1, or 2 -- traced algebraically and confirmed by a
        // Python simulation) reaches DEPBAR.LE. A __builtin_assume hint
        // on the single runtime-valued call site was tried FIRST and
        // had zero effect on the compiled output; splitting into three
        // separate call sites with literal arguments (0, 1, 2) instead
        // -- the same mechanism that already worked for A/B's own
        // address tables -- produced the largest single instruction-
        // count reduction of this entire investigation: 182/iteration
        // vs fastk_precomp2's own 247. See gemm_fp16_mma_ptx_wide_
        // bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3_
        // kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp3 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Built on fastk_precomp2 (the standing champion -- NOT
        // precomp3, a real GPU-measured regression) + pipe's own
        // second-MMA-block B-load interleaving, retested here after ncu
        // profiling (--set full, N=8192) showed fastk_precomp2 is now
        // compute/latency-bound (stall_wait 46.3%, stall_math 22.0%,
        // both on HMMA itself) rather than overhead-bound, the regime
        // pipe originally failed in on the old acongruous baseline.
        // GENUINELY UNCERTAIN -- unlike the other kernels in this
        // family, this one's outcome cannot be predicted from compile-
        // only counts (precomp3 already showed static counts can
        // mislead in this regime). See gemm_fp16_mma_ptx_wide_bigreg_sq_
        // deep4_iter_epivec_acongruous_fastk_precomp2_pipe_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp2_pipe = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Built on fastk_precomp2 with real CUTLASS's own warp-level
        // accumulate-loop order (a serpentine sweep, read directly from
        // cutlass/gemm/warp/mma_tensor_op.h) instead of this family's
        // own reset-every-sub-block order. Pure reordering of the same
        // 32 mma.sync calls per MMA block (verified bijective in
        // Python) -- targets the hardware operand-reuse cache
        // (`.reuse`): confirmed at the compiled-SASS level to raise
        // `.reuse` usage on HMMA from 48/64 to 62/64, same total
        // instruction/HMMA count. See gemm_fp16_mma_ptx_wide_bigreg_sq_
        // deep4_iter_epivec_acongruous_fastk_precomp2_serp_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp = time_call(start, stop, [&]() {
            tomos::cuda::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Built on fastk_precomp2_serp with the fragment loads
        // (load_a_frag/load_b_frags' own individual ldmatrix calls)
        // interleaved with the accumulate block that doesn't need them
        // yet, instead of bursted before it -- matching real CUTLASS's
        // own compiled mainloop pattern, motivated by ncu profiling
        // showing CUTLASS keeps its tensor cores busier (stall_math
        // 39.8% vs our 21.8%) and stalls on dependencies less
        // (stall_wait 34.8% vs our 46.2%) even after HMMA/LDSM counts
        // and operand reuse already matched. The most structurally
        // invasive change in this family's history. See gemm_fp16_
        // mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_
        // precomp2_serp_ildsm_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Isolates pipeline depth alone on top of deep4_iter_epivec's own
        // fixes -- the OTHER direction from stages5 (3 buffers/prefetch-
        // distance-2 instead of 4/3). Motivated by direct cuobjdump
        // --dump-sass inspection of the real, already-built CUTLASS
        // reference binary (build/cutlass_fp16_gemm_ref): its actual
        // MmaMultistage mainloop -- on a warp tile (64x64x32) and MMA
        // instruction shape (m16n8k16) already identical to this
        // family's own -- uses Stages=3, not 4 or 5. See gemm_fp16_
        // mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_stages3 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages3(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Isolates pipeline depth alone on top of deep4_iter_epivec's own
        // fixes -- 5 buffers/prefetch-distance-4 instead of 4/3, single
        // knob change (gemm_fp16_prime_wait_n and the mainloop's own
        // outstanding-count formula are both already generic in stage
        // count). Motivated by ncu --page source per-instruction stall
        // sampling showing DEPBAR.LE (__pipeline_wait_prior) immediately
        // precedes the hottest-stalling HMMA -- direct evidence of cp.
        // async/DRAM latency as the dominant remaining bottleneck. See
        // gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5_
        // kernel's own comment in mm_cuda_kernels.cu for the full
        // derivation and why the pre-existing wide_bigreg_sq_deep5 kernel
        // does NOT cleanly answer this same question.
        double ms_deep4_iter_epivec_stages5 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_stages5(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Interleaves the second MMA block's next-tile B-fragment loads
        // per-wp with that wp's own (independent) MMA work, instead of
        // batching all 4 loads before the block starts -- targets the
        // "fixed latency execution dependency" stall ncu still shows as
        // dominant even after today's other fixes, motivated by a real
        // SASS comparison showing CUTLASS packs its LDSM/HMMA far more
        // tightly. Genuinely uncertain, not a mechanically-proven fix
        // like the other siblings -- see gemm_fp16_mma_ptx_wide_bigreg_
        // sq_deep4_iter_epivec_pipe_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_pipe = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_pipe(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Extends deep4_iter's own address-hoisting principle to load_
        // a_frag/load_b_frags (never touched by deep4_iter's own fix) --
        // hoists thread-constant/call-invariant quantities those
        // lambdas previously recomputed on every one of their 2*
        // num_tiles calls. A pure algebraic rewrite, not a speculative
        // reorder. See gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_
        // epivec_kernel's own comment in mm_cuda_kernels.cu for the
        // full derivation.
        double ms_deep4_iter2_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter2_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Pure grid-axis relabeling of deep4_iter_epivec (blockIdx.x/y
        // swapped for which maps to the M-tile vs N-tile index) --
        // targets the finding that 99.4% of the kernel's L2-excessive-
        // sector count concentrated in A's own global load, with no
        // per-thread addressing bug found on re-derivation, pointing at
        // grid-level L2 residency instead. See gemm_fp16_mma_ptx_wide_
        // bigreg_sq_deep4_iter_epivec_gridswap_kernel's own comment in
        // mm_cuda_kernels.cu for the full derivation.
        double ms_deep4_iter_epivec_gridswap = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_gridswap(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Genuinely new tile geometry (BM=128, BN=256, matching real
        // CUTLASS's own ThreadblockShape<128,256,32> exactly) -- the
        // largest, highest-risk change in this family. See gemm_fp16_
        // mma_ptx_wide_bigreg_cutlassdims_iter_epivec_kernel's own
        // comment in mm_cuda_kernels.cu for the full derivation.
        double ms_cutlassdims_iter_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims_iter_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Same tile/warp/MMA shape as cutlassdims_iter_epivec's own, but
        // Stages=3 instead of 4 -- the full CUTLASS-matching config
        // (cuobjdump confirms real CUTLASS's own mainloop uses Stages=3).
        // NOT expected to fix cutlassdims_iter_epivec's own large-N
        // regression (that's a grid-wave-quantization effect independent
        // of stage count) -- see gemm_fp16_mma_ptx_wide_bigreg_
        // cutlassdims3_iter_epivec_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation.
        double ms_cutlassdims3_iter_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_cutlassdims3_iter_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // One stage deeper than deep4 (5 instead of 4) -- untried
        // territory for this specific kernel; see gemm_fp16_mma_ptx_
        // wide_bigreg_sq_deep5_kernel's own comment in mm_cuda_kernels.cu
        // for why the analogous FP32_16 regression at 5 stages wasn't
        // assumed to transfer here.
        double ms_deep5 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep5(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Occupancy-lever variant of deep4 (quarter B tile, 2 CTAs/SM
        // target) -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2_
        // kernel's own comment in mm_cuda_kernels.cu.
        double ms_deep4_occ2 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_occ2(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // ILP-restructured variant of deep4 (both K-halves' fragments
        // loaded together, all 64 mma.sync's issued as one flat block)
        // -- see gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma_kernel's
        // own comment in mm_cuda_kernels.cu.
        double ms_deep4_bigmma = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_bigmma(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // BK=64 tile redesign (half as many outer barrier crossings for
        // the same K range, 3 stages instead of deep4's 4) -- see
        // gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_kernel's own comment in
        // mm_cuda_kernels.cu for the full re-derivation this needed and
        // the register-spill risk confirmed by ptxas (255/255 registers,
        // real spilling, unlike every other kernel in this family).
        double ms_bk64 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Register-neutral BK=64: same tile as bk64, but the K-substep
        // buffering is kept [2]-deep (matching deep4's own footprint)
        // instead of bk64's [4]-deep one -- see gemm_fp16_mma_ptx_wide_
        // bigreg_sq_bk64_reg2_kernel's own comment in mm_cuda_kernels.cu.
        // Confirmed via ptxas: 222/232 registers, 0 spill (vs bk64's
        // 255/255 with real spilling).
        double ms_bk64_reg2 = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        // Combines bk64_reg2's own register-fixed BK=64/3-stage mainloop
        // with deep4_iter's address-hoisting and deep4_epivec's
        // vectorized epilogue -- see gemm_fp16_mma_ptx_wide_bigreg_sq_
        // bk64_reg2_iter_epivec_kernel's own comment in mm_cuda_
        // kernels.cu for the full derivation and realistic expectations
        // (likely helps at N=4096, uncertain at large N).
        double ms_bk64_reg2_iter_epivec = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_wide_bigreg_sq_bk64_reg2_iter_epivec(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        double ms_cutlass_cfg = time_call(start, stop, [&]() {
            tomos::cuda::experimental::gemm_fp16_mma_ptx_cutlass_config(
                tomos::NO_TRANSPOSE, tomos::NO_TRANSPOSE, 1.7,
                n, n, n, A_h, lda, B_h, ldb, 0.3, C_h, ldc);
        }, reps);
        double ms_cublas = time_call(start, stop, [&]() {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                       &alpha_h, A_h, lda, B_h, ldb, &beta_h, C_h, ldc);
        }, reps);

        double flops = 2.0 * static_cast<double>(n) * n * n;
        double g_cutlass = flops / ms_cutlass / 1e6;
        double g_cublas = flops / ms_cublas / 1e6;
        double g_bigreg_sq = flops / ms_bigreg_sq / 1e6;
        double g_bigreg = flops / ms_bigreg / 1e6;
        double g_deep4 = flops / ms_deep4 / 1e6;
        double g_deep4_iter = flops / ms_deep4_iter / 1e6;
        double g_deep4_epivec = flops / ms_deep4_epivec / 1e6;
        double g_deep4_iter_epivec = flops / ms_deep4_iter_epivec / 1e6;
        double g_deep4_iter_epivec_align = flops / ms_deep4_iter_epivec_align / 1e6;
        double g_deep4_iter_epivec_pad280 = flops / ms_deep4_iter_epivec_pad280 / 1e6;
        double g_deep4_iter_epivec_aswizzle = flops / ms_deep4_iter_epivec_aswizzle / 1e6;
        double g_deep4_iter_epivec_acongruous = flops / ms_deep4_iter_epivec_acongruous / 1e6;
        double g_deep4_iter_epivec_acongruous_pipe = flops / ms_deep4_iter_epivec_acongruous_pipe / 1e6;
        double g_deep4_iter_epivec_acongruous2 = flops / ms_deep4_iter_epivec_acongruous2 / 1e6;
        double g_deep4_iter_epivec_acongruous_stages3 = flops / ms_deep4_iter_epivec_acongruous_stages3 / 1e6;
        double g_deep4_iter_epivec_acongruous_ldgsts = flops / ms_deep4_iter_epivec_acongruous_ldgsts / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk = flops / ms_deep4_iter_epivec_acongruous_fastk / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp2 = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp2 / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp3 = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp3 / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp2_pipe = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp2_pipe / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp2_serp = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp / 1e6;
        double g_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm = flops / ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm / 1e6;
        double g_deep4_iter_epivec_stages3 = flops / ms_deep4_iter_epivec_stages3 / 1e6;
        double g_deep4_iter_epivec_stages5 = flops / ms_deep4_iter_epivec_stages5 / 1e6;
        double g_deep4_iter_epivec_pipe = flops / ms_deep4_iter_epivec_pipe / 1e6;
        double g_deep4_iter2_epivec = flops / ms_deep4_iter2_epivec / 1e6;
        double g_deep4_iter_epivec_gridswap = flops / ms_deep4_iter_epivec_gridswap / 1e6;
        double g_cutlassdims_iter_epivec = flops / ms_cutlassdims_iter_epivec / 1e6;
        double g_cutlassdims3_iter_epivec = flops / ms_cutlassdims3_iter_epivec / 1e6;
        double g_deep5 = flops / ms_deep5 / 1e6;
        double g_deep4_occ2 = flops / ms_deep4_occ2 / 1e6;
        double g_deep4_bigmma = flops / ms_deep4_bigmma / 1e6;
        double g_bk64 = flops / ms_bk64 / 1e6;
        double g_bk64_reg2 = flops / ms_bk64_reg2 / 1e6;
        double g_bk64_reg2_iter_epivec = flops / ms_bk64_reg2_iter_epivec / 1e6;
        double g_cutlass_cfg = flops / ms_cutlass_cfg / 1e6;

        struct Named { const char* name; double gflops; };
        Named tomos_results[] = {
            {"wide_bigreg_sq", g_bigreg_sq},
            {"wide_bigreg", g_bigreg},
            {"wide_bigreg_sq_deep4", g_deep4},
            {"wide_bigreg_sq_deep4_iter", g_deep4_iter},
            {"wide_bigreg_sq_deep4_epivec", g_deep4_epivec},
            {"wide_bigreg_sq_deep4_iter_epivec", g_deep4_iter_epivec},
            {"wide_bigreg_sq_deep4_iter_epivec_align", g_deep4_iter_epivec_align},
            {"wide_bigreg_sq_deep4_iter_epivec_pad280", g_deep4_iter_epivec_pad280},
            {"wide_bigreg_sq_deep4_iter_epivec_aswizzle", g_deep4_iter_epivec_aswizzle},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous", g_deep4_iter_epivec_acongruous},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe", g_deep4_iter_epivec_acongruous_pipe},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous2", g_deep4_iter_epivec_acongruous2},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3", g_deep4_iter_epivec_acongruous_stages3},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts", g_deep4_iter_epivec_acongruous_ldgsts},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk", g_deep4_iter_epivec_acongruous_fastk},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp", g_deep4_iter_epivec_acongruous_fastk_precomp},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2", g_deep4_iter_epivec_acongruous_fastk_precomp2},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3", g_deep4_iter_epivec_acongruous_fastk_precomp3},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe", g_deep4_iter_epivec_acongruous_fastk_precomp2_pipe},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp", g_deep4_iter_epivec_acongruous_fastk_precomp2_serp},
            {"wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm", g_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm},
            {"wide_bigreg_sq_deep4_iter_epivec_stages3", g_deep4_iter_epivec_stages3},
            {"wide_bigreg_sq_deep4_iter_epivec_stages5", g_deep4_iter_epivec_stages5},
            {"wide_bigreg_sq_deep4_iter_epivec_pipe", g_deep4_iter_epivec_pipe},
            {"wide_bigreg_sq_deep4_iter2_epivec", g_deep4_iter2_epivec},
            {"wide_bigreg_sq_deep4_iter_epivec_gridswap", g_deep4_iter_epivec_gridswap},
            {"wide_bigreg_cutlassdims_iter_epivec", g_cutlassdims_iter_epivec},
            {"wide_bigreg_cutlassdims3_iter_epivec", g_cutlassdims3_iter_epivec},
            {"wide_bigreg_sq_deep5", g_deep5},
            {"wide_bigreg_sq_deep4_occ2", g_deep4_occ2},
            {"wide_bigreg_sq_deep4_bigmma", g_deep4_bigmma},
            {"wide_bigreg_sq_bk64", g_bk64},
            {"wide_bigreg_sq_bk64_reg2", g_bk64_reg2},
            {"wide_bigreg_sq_bk64_reg2_iter_epivec", g_bk64_reg2_iter_epivec},
            {"cutlass_config", g_cutlass_cfg},
        };
        Named* best = &tomos_results[0];
        for (auto& r : tomos_results) if (r.gflops > best->gflops) best = &r;

        std::printf("  N=%zu\n", n);
        std::printf("    wide_bigreg_sq  = %9.3fms (%10.1f GFlop/s)\n", ms_bigreg_sq, g_bigreg_sq);
        std::printf("    wide_bigreg     = %9.3fms (%10.1f GFlop/s)\n", ms_bigreg, g_bigreg);
        std::printf("    wide_bigreg_sq_deep4 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4, g_deep4);
        std::printf("    wide_bigreg_sq_deep4_iter = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter, g_deep4_iter);
        std::printf("    wide_bigreg_sq_deep4_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_epivec, g_deep4_epivec);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec, g_deep4_iter_epivec);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_align = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_align, g_deep4_iter_epivec_align);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_pad280 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_pad280, g_deep4_iter_epivec_pad280);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_aswizzle = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_aswizzle, g_deep4_iter_epivec_aswizzle);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous, g_deep4_iter_epivec_acongruous);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_pipe = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_pipe, g_deep4_iter_epivec_acongruous_pipe);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous2 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous2, g_deep4_iter_epivec_acongruous2);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_stages3 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_stages3, g_deep4_iter_epivec_acongruous_stages3);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_ldgsts = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_ldgsts, g_deep4_iter_epivec_acongruous_ldgsts);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk, g_deep4_iter_epivec_acongruous_fastk);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp, g_deep4_iter_epivec_acongruous_fastk_precomp);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp2, g_deep4_iter_epivec_acongruous_fastk_precomp2);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp3 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp3, g_deep4_iter_epivec_acongruous_fastk_precomp3);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_pipe = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp2_pipe, g_deep4_iter_epivec_acongruous_fastk_precomp2_pipe);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp, g_deep4_iter_epivec_acongruous_fastk_precomp2_serp);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm, g_deep4_iter_epivec_acongruous_fastk_precomp2_serp_ildsm);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_stages3 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_stages3, g_deep4_iter_epivec_stages3);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_stages5 = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_stages5, g_deep4_iter_epivec_stages5);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_pipe = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_pipe, g_deep4_iter_epivec_pipe);
        std::printf("    wide_bigreg_sq_deep4_iter2_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter2_epivec, g_deep4_iter2_epivec);
        std::printf("    wide_bigreg_sq_deep4_iter_epivec_gridswap = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_iter_epivec_gridswap, g_deep4_iter_epivec_gridswap);
        std::printf("    wide_bigreg_cutlassdims_iter_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_cutlassdims_iter_epivec, g_cutlassdims_iter_epivec);
        std::printf("    wide_bigreg_cutlassdims3_iter_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_cutlassdims3_iter_epivec, g_cutlassdims3_iter_epivec);
        std::printf("    wide_bigreg_sq_deep5 = %6.3fms (%10.1f GFlop/s)\n", ms_deep5, g_deep5);
        std::printf("    wide_bigreg_sq_deep4_occ2   = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_occ2, g_deep4_occ2);
        std::printf("    wide_bigreg_sq_deep4_bigmma = %6.3fms (%10.1f GFlop/s)\n", ms_deep4_bigmma, g_deep4_bigmma);
        std::printf("    wide_bigreg_sq_bk64         = %6.3fms (%10.1f GFlop/s)\n", ms_bk64, g_bk64);
        std::printf("    wide_bigreg_sq_bk64_reg2    = %6.3fms (%10.1f GFlop/s)\n", ms_bk64_reg2, g_bk64_reg2);
        std::printf("    wide_bigreg_sq_bk64_reg2_iter_epivec = %6.3fms (%10.1f GFlop/s)\n", ms_bk64_reg2_iter_epivec, g_bk64_reg2_iter_epivec);
        std::printf("    cutlass_config  = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass_cfg, g_cutlass_cfg);
        std::printf("    cutlass (real)  = %9.3fms (%10.1f GFlop/s)\n", ms_cutlass, g_cutlass);
        std::printf("    cublas          = %9.3fms (%10.1f GFlop/s)\n", ms_cublas, g_cublas);
        std::printf("    best of ours: %s (%.1f GFlop/s) -- %.3fx cublas, %.3fx real cutlass\n",
                   best->name, best->gflops, best->gflops / g_cublas, best->gflops / g_cutlass);

        cudaFree(A); cudaFree(B); cudaFree(C_ref);
        if (workspace) cudaFree(workspace);
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

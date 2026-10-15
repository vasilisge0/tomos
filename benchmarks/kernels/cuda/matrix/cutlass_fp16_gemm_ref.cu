// Reference-only benchmark: CUTLASS's own device::Gemm for FP16, used
// purely as an external comparison point -- NOT a dependency of tomos's
// custom kernels. Matches tomos's verify_mm_fp16_gemm.cu problem shape
// exactly: square NxN, A/B/C all __half (cutlass::half_t), column-major
// (matching TransposeType::NO_TRANSPOSE with ld==n throughout tomos),
// FP32 accumulation, alpha=1.7/beta=0.3 (same as tomos's correctness
// harness, though this file only measures throughput). Same cudaEvent_t
// timing methodology as tomos's bench_one()/time_call() for an
// apples-to-apples GFlop/s comparison.
//
// This is one specific, "textbook" Ampere tensorop config (128x256x32
// threadblock tile, 64x64x32 warp tile, 16x8x16 instruction shape, 3
// pipeline stages) taken directly from CUTLASS's own example programs --
// not a hyper-tuned search over CUTLASS's full config space. It answers
// "what does CUTLASS's standard, undistinguished GEMM achieve here",
// which is a meaningful reference point even though it isn't
// necessarily CUTLASS's own best-case number for this problem size.
//
// Also times cublasHgemm directly in the SAME process/run (same
// alpha/beta, same cudaEvent_t methodology, same A/B/C buffers via a
// reinterpret_cast between cutlass::half_t* and __half* -- the two are
// bit-layout-compatible), exactly like verify_mm_fp16_gemm's bench_one()
// does for tomos's own kernels. Keeping both in one run avoids comparing
// numbers across separate invocations, which can differ from GPU clock/
// thermal state alone (see the N=2048 swing observed earlier this
// session between two separate verify_mm_fp16_gemm runs).
//
// Deliberately NOT wired into tomos's CMake build (no add_executable
// entry in this directory's CMakeLists.txt) -- built standalone via a
// direct nvcc invocation, see scripts/cutlass_fp16_gemm_ref.sh.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

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
    cutlass::gemm::GemmShape<128, 256, 32>,   // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // instruction shape (fp16)
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3  // pipeline stages
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
    std::vector<size_t> sizes = {1024, 2048, 4096, 8192, 16384};
    if (argc > 1) {
        sizes.clear();
        sizes.push_back(static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::printf("=== CUTLASS FP16 tensorop GEMM vs cuBLAS Hgemm (128x256x32/64x64x32/3-stage) ===\n");
    for (size_t n : sizes) {
        cutlass::half_t *A, *B, *C;
        cudaMalloc(&A, n * n * sizeof(cutlass::half_t));
        cudaMalloc(&B, n * n * sizeof(cutlass::half_t));
        cudaMalloc(&C, n * n * sizeof(cutlass::half_t));
        gen_half(n * n, A);
        gen_half(n * n, B);
        gen_half(n * n, C);

        cutlass::gemm::GemmCoord problem_size(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        ElementAccumulator alpha(1.7f), beta(0.3f);
        int lda = static_cast<int>(n), ldb = static_cast<int>(n), ldc = static_cast<int>(n);

        Gemm gemm_op;
        typename Gemm::Arguments args(
            problem_size,
            {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
            {alpha, beta});

        size_t workspace_size = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (can_implement failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(A); cudaFree(B); cudaFree(C);
            if (workspace) cudaFree(workspace);
            continue;
        }
        status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            std::printf("  N=%-9zu SKIPPED (initialize failed: %s)\n", n,
                       cutlass::cutlassGetStatusString(status));
            cudaFree(A); cudaFree(B); cudaFree(C);
            if (workspace) cudaFree(workspace);
            continue;
        }

        int reps = (n <= 2048) ? 20 : (n <= 8192 ? 5 : 2);
        double ms_cutlass = time_call(start, stop, [&]() { gemm_op(); }, reps);

        __half alpha_h = static_cast<__half>(1.7f), beta_h = static_cast<__half>(0.3f);
        __half* A_h = reinterpret_cast<__half*>(A);
        __half* B_h = reinterpret_cast<__half*>(B);
        __half* C_h = reinterpret_cast<__half*>(C);
        double ms_cublas = time_call(start, stop, [&]() {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                       &alpha_h, A_h, lda, B_h, ldb, &beta_h, C_h, ldc);
        }, reps);

        double flops = 2.0 * static_cast<double>(n) * n * n;
        double gflops_cutlass = flops / ms_cutlass / 1e6;
        double gflops_cublas = flops / ms_cublas / 1e6;
        std::printf("  N=%-9zu cutlass=%9.3fms (%9.1f GFlop/s)  cublas=%9.3fms (%9.1f GFlop/s)  "
                   "cutlass/cublas=%.3fx\n",
                   n, ms_cutlass, gflops_cutlass, ms_cublas, gflops_cublas,
                   gflops_cutlass / gflops_cublas);

        cudaFree(A); cudaFree(B); cudaFree(C);
        if (workspace) cudaFree(workspace);
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

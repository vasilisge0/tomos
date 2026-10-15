// Isolates ONLY the FP16-compute GEMV, at the real LSQR matrix's own shape
// (1,000,000 x 1,000), calling it through the EXACT SAME entry point
// lsqr.cpp itself calls -- mv_apply(Device, TransposeType, alpha, Dense&,
// Dense&, beta, Dense&, NumericType) -- for BOTH storage schemes LSQR can
// use: NumericType::FP32_16 (--implicit-duplicates) and plain native
// NumericType::FP16 (--explicit-duplicates), from the SAME controlled
// harness, so the two numbers are directly comparable without crossing
// between different benchmark files/methodologies (which turned out to
// disagree by ~10% on the native_mixed baseline alone -- see this
// session's own investigation).
//
// No solver, no preconditioner, no restart logic -- just A, x, y and a
// timed loop. Data is random (uniform, NOT the real matrix's ~18%
// underflow distribution) since the FP32_16 path's own decode is
// branchless (constant cost regardless of underflow rate -- see
// fp32_16_half_only_branchless_no_overflow_nomask's own comment in
// fp32_16_pack.hpp), so only the SHAPE matters for a pure timing
// measurement, not the values.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "fp32_16_registries.hpp"
#include "fp32_16_cuda_kernels.cuh"
#include "matrix.hpp"

using namespace tomos;

namespace tomos {
namespace cuda {
namespace experimental {
void mv_fp16_native_mixed(TransposeType trans, double alpha, dim<2> size, const __half* A, size_t ld_a,
                          const float* x, size_t ld_x, double beta, float* y, size_t ld_y);
}
}
}

namespace {

template <typename T>
void fill_dense_from_host(Dense& dense, const std::vector<double>& src) {
    auto v = as_strided_view<T>(dense);
    std::vector<T> typed(src.size());
    for (size_t k = 0; k < src.size(); ++k) typed[k] = static_cast<T>(static_cast<float>(src[k]));
    cudaMemcpy(v.values_, typed.data(), typed.size() * sizeof(T), cudaMemcpyHostToDevice);
}

// Mirrors lsqr_precision_runner.cu's own fill_fp32_16_dense_from_host.
void fill_fp32_16_dense_from_host(Context& ctx, Device dev, Dense& target,
                                  const std::vector<double>& src, dim<2> size) {
    Dense staging = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_from_host<float>(staging, src);
    dispatch::Fp32_16::get().lookup_split(dev)(staging, target);
}

double now_ms_via_events(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms);
}

// Runs the bulk-loop and per-call-synced timings for A/x/y already staged
// with real data, printing both under the given label. trans selects
// NO_TRANSPOSE (x length = A's cols, y length = A's rows) or TRANSPOSE
// (x length = A's rows, y length = A's cols) -- caller is responsible for
// sizing x/y to match.
void time_mv_apply(const char* label, TransposeType trans, Dense& A, Dense& x, double alpha, double beta, Dense& y,
                   int reps) {
    // Warm up (first call may pay one-time lazy-init/JIT/allocator costs).
    for (int i = 0; i < 5; ++i) {
        mv_apply(Device::CUDA, trans, alpha, A, x, beta, y, NumericType::FP16);
    }
    cudaDeviceSynchronize();

    // (1) Bulk-timed loop: one event pair around all reps, minimal per-call
    // overhead -- matches how test-new's own benchmark measures.
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < reps; ++i) {
            mv_apply(Device::CUDA, trans, alpha, A, x, beta, y, NumericType::FP16);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        double total_ms = now_ms_via_events(start, stop);
        std::printf("[%s] bulk-loop (1 sync for all %d reps):      avg_ms=%.4f\n", label, reps, total_ms / reps);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // (2) Per-call-synced loop: matches TOMOS_LSQR_TIMING's own
    // ScopedCudaMvTimer, which forces a cudaEventSynchronize() after every
    // single call -- isolates whether that per-call sync itself costs
    // something beyond the kernel's own execution time.
    {
        double total_ms = 0.0;
        for (int i = 0; i < reps; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            mv_apply(Device::CUDA, trans, alpha, A, x, beta, y, NumericType::FP16);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            total_ms += now_ms_via_events(start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        std::printf("[%s] per-call-synced loop (sync every rep):   avg_ms=%.4f\n", label, total_ms / reps);
    }
}

// Literal copy of mv_fp32_16_pure_half_vs_cublas_ref.cu's own time_call --
// testing whether ITS specific structure (std::function<void()> wrapper
// around the kernel call, vs. mv_apply called directly with no wrapper)
// reproduces the ~150us/call inflation that benchmark's own native_mixed_ms
// column shows relative to every other measurement of the same kernel.
double time_call_replica(cudaEvent_t start, cudaEvent_t stop, const std::function<void()>& fn, int reps) {
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

}  // namespace

int main(int argc, char** argv) {
    size_t m = 1000000, n = 1000;
    int reps = 100;
    if (argc > 1) m = static_cast<size_t>(std::atoll(argv[1]));
    if (argc > 2) n = static_cast<size_t>(std::atoll(argv[2]));
    if (argc > 3) reps = std::atoi(argv[3]);

    std::printf("==> Isolated FP16-compute GEMV via mv_apply (no solver): m=%zu n=%zu reps=%d\n", m, n, reps);

    // The benchmark being reconciled against (mv_fp32_16_pure_half_vs_
    // cublas_ref.cu) calls cublasCreate(&handle) once, before any
    // run_shape/timing happens -- this harness never did, despite linking
    // cublas. cuBLAS init can have context-wide side effects (math mode,
    // workspace reservations); testing whether that's the missing piece.
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    Context ctx{Device::CUDA};

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> a_host(m * n), x_host(n);
    for (auto& v : a_host) v = dist(rng);
    for (auto& v : x_host) v = dist(rng);

    double alpha = 1.7, beta = 0.3;

    // --implicit-duplicates' own storage: A packed as NumericType::FP32_16.
    {
        Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32_16, {m, n});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {n, 1});
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {m, 1});
        fill_fp32_16_dense_from_host(ctx, Device::CUDA, A, a_host, {m, n});
        fill_dense_from_host<float>(x, x_host);
        time_mv_apply("implicit (FP32_16-packed) NO_TRANSPOSE", TransposeType::NO_TRANSPOSE, A, x, alpha, beta, y,
                      reps);
    }

    // --explicit-duplicates' own storage: A as plain native __half, same
    // (FP16,FP32,FP32,compute=FP16) mv_apply slot -> mv_fp16_native_impl ->
    // cuda::mv_fp16_native_mixed (dense_registries.cpp) -- the exact same
    // kernel test-new's own benchmark measures as "native_mixed" and
    // --explicit-duplicates uses for its real per-iteration matvec.
    {
        Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP16, {m, n});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {n, 1});
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {m, 1});
        fill_dense_from_host<__half>(A, a_host);
        fill_dense_from_host<float>(x, x_host);
        time_mv_apply("explicit (native FP16) NO_TRANSPOSE", TransposeType::NO_TRANSPOSE, A, x, alpha, beta, y,
                      reps);
    }

    // TRANSPOSE direction: x length = A's rows (m), y length = A's cols
    // (n) -- mirror image of NO_TRANSPOSE above. Same A buffers can't be
    // reused since x/y need different lengths; fresh Dense objects, same
    // a_host/x_host data (truncated/reused as needed -- only the first n
    // entries of x_host are used here, matching TRANSPOSE's own shorter
    // x).
    std::vector<double> x_host_t(m);
    for (auto& v : x_host_t) v = dist(rng);

    // --implicit-duplicates' own storage, TRANSPOSE direction. Routes to
    // mv_pure_half_impl's TRANSPOSE branch (fp32_16_registries.cpp) --
    // now mv_fp32_16_pure_half_fast2_transpose_atomiconly_scaled_mblock_
    // vec4_split (plain, no cp.async -- see that dispatch's own comment
    // for why the cp.async sibling was dropped this session).
    {
        Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32_16, {m, n});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {m, 1});
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {n, 1});
        fill_fp32_16_dense_from_host(ctx, Device::CUDA, A, a_host, {m, n});
        fill_dense_from_host<float>(x, x_host_t);
        time_mv_apply("implicit (FP32_16-packed) TRANSPOSE", TransposeType::TRANSPOSE, A, x, alpha, beta, y, reps);
    }

    // --explicit-duplicates' own storage, TRANSPOSE direction. Routes to
    // mv_fp16_native_impl -> mv_fp16_native_mixed_transpose_
    // atomiconly_scaled_mblock_vec4_split (dense_registries.cpp).
    {
        Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP16, {m, n});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {m, 1});
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, {n, 1});
        fill_dense_from_host<__half>(A, a_host);
        fill_dense_from_host<float>(x, x_host_t);
        time_mv_apply("explicit (native FP16) TRANSPOSE", TransposeType::TRANSPOSE, A, x, alpha, beta, y, reps);
    }

    // (2.5) Simulate the ~50 OTHER transient cudaMalloc/cudaFree cycles
    // run_shape does (one per kernel variant it times) before it ever
    // reaches native_mixed's own measurement, even in single-shape --tall
    // mode -- my replica below runs in a fresh process with none of that
    // prior allocator churn. Sizes vary (matching this shape's A/x/y
    // buffers across different dtypes: __half A~2GB, uint32_t FP32_16
    // A~4GB, float x/y~tiny) to mimic the real mix, not just repeat one
    // size.
    {
        std::printf("==> Simulating ~50 prior alloc/free cycles (run_shape's own other measurements)...\n");
        for (int i = 0; i < 50; ++i) {
            size_t bytes = (i % 3 == 0) ? (m * n * sizeof(uint32_t))
                          : (i % 3 == 1) ? (m * n * sizeof(__half))
                                         : (m * n * sizeof(float));
            void* p = nullptr;
            cudaError_t err = cudaMalloc(&p, bytes);
            if (err == cudaSuccess) {
                cudaMemset(p, 0, bytes);
                cudaFree(p);
            } else {
                cudaGetLastError();
            }
        }
        cudaDeviceSynchronize();
        std::printf("==> Done simulating prior allocator churn.\n");
    }

    // (3) Literal replica of the benchmark's own call: raw cudaMalloc'd
    // buffers (not Dense), cuda::experimental::mv_fp16_native_mixed called
    // DIRECTLY (bypassing mv_apply's registry lookup entirely, exactly
    // like the benchmark does), wrapped in a std::function<void()> and
    // timed via time_call_replica -- isolates whether std::function
    // indirection and/or the direct-call (vs mv_apply) path is what's
    // actually responsible for the benchmark's own inflated reading.
    {
        __half* A_n;
        float *x_n, *y_n;
        cudaMalloc(&A_n, m * n * sizeof(__half));
        cudaMalloc(&x_n, n * sizeof(float));
        cudaMalloc(&y_n, m * sizeof(float));
        std::vector<__half> a_half(m * n);
        std::vector<float> x_float(n);
        for (size_t i = 0; i < m * n; ++i) a_half[i] = static_cast<__half>(static_cast<float>(a_host[i]));
        for (size_t i = 0; i < n; ++i) x_float[i] = static_cast<float>(x_host[i]);
        cudaMemcpy(A_n, a_half.data(), m * n * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(x_n, x_float.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        int lda = static_cast<int>(m);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        double ms = time_call_replica(start, stop, [&]() {
            cuda::experimental::mv_fp16_native_mixed(
                TransposeType::NO_TRANSPOSE, alpha, dim<2>{m, n}, A_n, static_cast<size_t>(lda), x_n, 1, beta, y_n, 1);
        }, reps);
        std::printf("[explicit, raw-buffer direct-call replica] time_call_replica: avg_ms=%.4f\n", ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A_n);
        cudaFree(x_n);
        cudaFree(y_n);
    }

    return 0;
}

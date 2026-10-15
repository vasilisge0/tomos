#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "solver.hpp"
#include "workspace.hpp"

namespace tomos {
namespace bench {

// Builds a well-conditioned, exactly-consistent M x N dense system on the
// GPU: random A, random x_true, b = A * x_true. Data is generated on the
// host and copied in, mirroring LsqrSolveCudaTest in test/core/solver.
static void make_dense_consistent_system_cuda(Context& ctx, size_t M, size_t N,
                                              Dense& A, Dense& b) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> a_host(M * N), x_host(N);
    for (auto& v : a_host) v = dist(gen);
    for (auto& v : x_host) v = dist(gen);

    auto vA = as_strided_view<double>(A);
    cudaMemcpy(vA.values_, a_host.data(), a_host.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    Dense x_true =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {N, 1});
    auto vx = as_strided_view<double>(x_true);
    cudaMemcpy(vx.values_, x_host.data(), x_host.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    set_values(0.0, b);
    mv_apply(Device::CUDA, TransposeType::NO_TRANSPOSE, 1.0, A, x_true, 0.0, b);
}

// Ill-conditioned variant: same construction, but column j of A is scaled by
// a smoothly decaying factor (10^-4 at j=N-1 relative to j=0), giving a
// condition number of roughly 1e4 -- large enough for a preconditioner to
// matter, unlike the well-conditioned system above.
static void make_ill_conditioned_dense_system_cuda(Context& ctx, size_t M,
                                                   size_t N, Dense& A,
                                                   Dense& b) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> a_host(M * N);
    for (size_t j = 0; j < N; ++j) {
        double scale = std::pow(1e-4, static_cast<double>(j) /
                                          static_cast<double>(N - 1));
        for (size_t i = 0; i < M; ++i) a_host[i + j * M] = dist(gen) * scale;
    }
    auto vA = as_strided_view<double>(A);
    cudaMemcpy(vA.values_, a_host.data(), a_host.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    std::vector<double> x_host(N);
    std::mt19937 gen2(43);
    for (auto& v : x_host) v = dist(gen2);
    Dense x_true =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {N, 1});
    auto vx = as_strided_view<double>(x_true);
    cudaMemcpy(vx.values_, x_host.data(), x_host.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    set_values(0.0, b);
    mv_apply(Device::CUDA, TransposeType::NO_TRANSPOSE, 1.0, A, x_true, 0.0, b);
}

// ||A*x - b|| / ||b||, computed on the host after copying x/b/r back.
static double relative_residual_cuda(Context& ctx, Dense& A, Dense& x,
                                     Dense& b, size_t M) {
    Dense r =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {M, 1});
    set_values(0.0, r);
    mv_apply(Device::CUDA, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, r);

    std::vector<double> r_host(M), b_host(M);
    auto vr = as_strided_view<double>(r);
    auto vb = as_strided_view<double>(b);
    cudaMemcpy(r_host.data(), vr.values_, M * sizeof(double),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host.data(), vb.values_, M * sizeof(double),
              cudaMemcpyDeviceToHost);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < M; ++i) {
        double ri = r_host[i] - b_host[i];
        num += ri * ri;
        den += b_host[i] * b_host[i];
    }
    return std::sqrt(num / den);
}

// ---- Dense LSQR on CUDA ----

static void BM_lsqr_dense_cuda(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CUDA};

    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {M, N});
    Dense b =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {M, 1});
    make_dense_consistent_system_cuda(ctx, M, N, A, b);

    Dense x =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {N, 1});
    Workspace ws = create_lsqr_workspace(ctx, {M, N}, NumericType::FP64);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(4 * N);

    for (auto _ : state) {
        set_values(0.0, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        cudaDeviceSynchronize();
    }

    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["relres"] = relative_residual_cuda(ctx, A, x, b, M);
}

BENCHMARK(BM_lsqr_dense_cuda)
    ->Args({200, 20})->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- Ill-conditioned dense LSQR on CUDA (unpreconditioned) ----

static void BM_lsqr_dense_illconditioned_cuda(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CUDA};

    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {M, N});
    Dense b =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {M, 1});
    make_ill_conditioned_dense_system_cuda(ctx, M, N, A, b);

    Dense x =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64, {N, 1});
    Workspace ws = create_lsqr_workspace(ctx, {M, N}, NumericType::FP64);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(50 * N);

    for (auto _ : state) {
        set_values(0.0, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        cudaDeviceSynchronize();
    }

    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["relres"] = relative_residual_cuda(ctx, A, x, b, M);
}

BENCHMARK(BM_lsqr_dense_illconditioned_cuda)
    ->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// NOTE: no CUDA sketch+QR-preconditioned benchmark here (unlike the CPU
// suite in benchmarks/solvers/cpu/bench_lsqr.cpp) -- apply_count_sketch's
// dense-source overload has no registered CUDA kernel yet (it goes through
// cpu::mm_csr's CUDA counterpart, which needs a cusparseHandle_t from a
// Context that isn't threaded through the current apply_count_sketch call
// signature; see the "CUDA path omitted" comment in
// src/dispatch/sketch_registries.cpp). Calling it throws
// std::runtime_error at solve time rather than silently misbehaving.

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

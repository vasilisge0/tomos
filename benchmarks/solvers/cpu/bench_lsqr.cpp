#include <benchmark/benchmark.h>
#include "bench_latex_reporter.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "countsketch.hpp"
#include "matrix.hpp"
#include "precision.hpp"
#include "sketch_qr.hpp"
#include "solver.hpp"
#include "workspace.hpp"

namespace tomos {
namespace bench {

// Fills A (M x N) with random entries and sets b = A * x_true for a random
// x_true, so the system is exactly consistent and relres is well-defined.
template <typename value_t>
static void make_dense_consistent_system(Context& ctx, size_t M, size_t N,
                                         Dense& A, Dense& b) {
    auto vA = as_strided_view<value_t>(A);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < M; ++i)
            vA.values_[i + j * vA.lead_dim_] = static_cast<value_t>(dist(gen));

    Dense x_true = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         numeric_type_of<value_t>, {N, 1});
    auto vx = as_strided_view<value_t>(x_true);
    for (size_t i = 0; i < N; ++i) vx.values_[i] = static_cast<value_t>(dist(gen));

    set_values(value_t{0}, b);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x_true, 0.0, b);
}

// Same as make_dense_consistent_system, but scales column j by a smoothly
// decaying factor (10^-4 at j=N-1 relative to j=0), giving A a condition
// number of roughly 1e4 -- large enough for a preconditioner to matter,
// unlike the well-conditioned random systems above.
template <typename value_t>
static void make_ill_conditioned_dense_system(Context& ctx, size_t M, size_t N,
                                              Dense& A, Dense& b) {
    make_dense_consistent_system<value_t>(ctx, M, N, A, b);
    auto vA = as_strided_view<value_t>(A);
    for (size_t j = 0; j < N; ++j) {
        double scale = std::pow(1e-4, static_cast<double>(j) /
                                          static_cast<double>(N - 1));
        for (size_t i = 0; i < M; ++i)
            vA.values_[i + j * vA.lead_dim_] = static_cast<value_t>(
                static_cast<double>(vA.values_[i + j * vA.lead_dim_]) * scale);
    }
    // b must be rebuilt from the now ill-conditioned A (b = A * x_true) --
    // reuse the same x_true implicitly by recomputing from the scaled A and
    // a fresh random x_true.
    Dense x_true = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         numeric_type_of<value_t>, {N, 1});
    auto vx = as_strided_view<value_t>(x_true);
    std::mt19937 gen(43);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) vx.values_[i] = static_cast<value_t>(dist(gen));
    set_values(value_t{0}, b);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x_true, 0.0, b);
}

// Random M x N CSR matrix at the given density, plus b = A * x_true.
template <typename value_t>
static Sparse make_sparse_consistent_system(Context& ctx, size_t M, size_t N,
                                            double density, Dense& b) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);

    std::vector<int> row_ptrs(M + 1, 0);
    std::vector<int> col_idxs;
    std::vector<double> values;
    for (size_t i = 0; i < M; ++i) {
        row_ptrs[i] = static_cast<int>(values.size());
        for (size_t j = 0; j < N; ++j) {
            if (keep_dist(gen) < density) {
                values.push_back(val_dist(gen));
                col_idxs.push_back(static_cast<int>(j));
            }
        }
    }
    row_ptrs[M] = static_cast<int>(values.size());

    Sparse A = Sparse::create_csr(ctx, numeric_type_of<value_t>,
                                  IntType::INT32, {M, N}, values.size());
    auto vA = as_csr_view<value_t, int>(A);
    std::copy(row_ptrs.begin(), row_ptrs.end(), vA.row_ptrs_);
    std::copy(col_idxs.begin(), col_idxs.end(), vA.col_idxs_);
    for (size_t k = 0; k < values.size(); ++k)
        vA.values_[k] = static_cast<value_t>(values[k]);

    Dense x_true = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         numeric_type_of<value_t>, {N, 1});
    auto vx = as_strided_view<value_t>(x_true);
    for (size_t i = 0; i < N; ++i)
        vx.values_[i] = static_cast<value_t>(val_dist(gen));

    set_values(value_t{0}, b);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x_true, 0.0, b);
    return A;
}

// ||A*x - b|| / ||b||, computed in double regardless of value_t.
template <typename value_t>
static double relative_residual(Context& ctx, Dense& x, Dense& b,
                                std::function<void(Dense&, Dense&)> apply_a,
                                size_t M) {
    Dense r = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    set_values(value_t{0}, r);
    apply_a(x, r);
    auto vr = as_strided_view<value_t>(r);
    auto vb = as_strided_view<value_t>(b);
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < M; ++i) {
        double ri = static_cast<double>(vr.values_[i]) -
                   static_cast<double>(vb.values_[i]);
        num += ri * ri;
        den += static_cast<double>(vb.values_[i]) * static_cast<double>(vb.values_[i]);
    }
    return std::sqrt(num / den);
}

// ---- Dense LSQR ----

template <typename value_t>
static void BM_lsqr_dense(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CPU};

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, N});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    make_dense_consistent_system<value_t>(ctx, M, N, A, b);

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {N, 1});
    Workspace ws = create_lsqr_workspace(ctx, {M, N}, numeric_type_of<value_t>);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(4 * N);

    for (auto _ : state) {
        set_values(value_t{0}, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        benchmark::ClobberMemory();
    }

    double relres = relative_residual<value_t>(
        ctx, x, b,
        [&](Dense& xv, Dense& r) {
            mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, xv, 0.0,
                    r);
        },
        M);
    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["relres"] = relres;
}

BENCHMARK_TEMPLATE(BM_lsqr_dense, double)
    ->Args({200, 20})->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_lsqr_dense, float)
    ->Args({200, 20})->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- Sparse LSQR ----

template <typename value_t>
static void BM_lsqr_sparse(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    const double density = 0.05;
    Context ctx{Device::CPU};

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    Sparse A =
        make_sparse_consistent_system<value_t>(ctx, M, N, density, b);

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {N, 1});
    Workspace ws = create_lsqr_workspace(ctx, {M, N}, numeric_type_of<value_t>);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(4 * N);

    for (auto _ : state) {
        set_values(value_t{0}, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        benchmark::ClobberMemory();
    }

    double relres = relative_residual<value_t>(
        ctx, x, b,
        [&](Dense& xv, Dense& r) {
            mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, xv, 0.0,
                    r);
        },
        M);
    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["nnz"] =
        static_cast<double>(as_csr_view<value_t, int>(A).nnz_);
    state.counters["relres"] = relres;
}

BENCHMARK_TEMPLATE(BM_lsqr_sparse, double)
    ->Args({2000, 100})->Args({10000, 200})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_lsqr_sparse, float)
    ->Args({2000, 100})->Args({10000, 200})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- Dense LSQR with a count-sketch + QR preconditioner ----
// Directly comparable to BM_lsqr_dense at the same (rows, cols): the
// preconditioner is built once (untimed setup, like A/b), then every timed
// iteration solves the identical system through the same lsqr_solve() entry
// point, just with add_precond() applied beforehand.

template <typename value_t>
static void BM_lsqr_dense_sketch_precond(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CPU};

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, N});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    make_dense_consistent_system<value_t>(ctx, M, N, A, b);

    // Sketch A down to k buckets (k > N), then QR-factor + invert R.
    const size_t k = std::min(M, 4 * N);
    Sparse S = Sparse::create_csr(ctx, numeric_type_of<value_t>,
                                  IntType::INT32, {k, M}, k);
    generate_count_sketch(S);
    Dense sketched = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                           numeric_type_of<value_t>, {k, N});
    apply_count_sketch(S, A, sketched);
    generate_sketch_qr_precond(S, sketched);

    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          numeric_type_of<value_t>, {N, N});
    auto vSketched = as_strided_view<value_t>(sketched);
    auto vPrecond = as_strided_view<value_t>(precond);
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i)
            vPrecond.values_[i + j * vPrecond.lead_dim_] =
                vSketched.values_[i + j * vSketched.lead_dim_];

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {N, 1});
    Workspace ws = create_lsqr_workspace(
        ctx, {M, N}, numeric_type_of<value_t>, numeric_type_of<value_t>);
    add_precond(precond, ws);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(4 * N);

    for (auto _ : state) {
        set_values(value_t{0}, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        benchmark::ClobberMemory();
    }

    double relres = relative_residual<value_t>(
        ctx, x, b,
        [&](Dense& xv, Dense& r) {
            mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, xv, 0.0,
                    r);
        },
        M);
    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["sketch_rows"] = static_cast<double>(k);
    state.counters["relres"] = relres;
}

BENCHMARK_TEMPLATE(BM_lsqr_dense_sketch_precond, double)
    ->Args({200, 20})->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- Ill-conditioned dense LSQR: unpreconditioned vs sketch+QR precond ----
// Same (rows, cols) and system-construction seed in both benchmarks below --
// only the presence of the preconditioner differs -- so the pair is a direct,
// apples-to-apples measurement of what preconditioning buys (or costs) on a
// system where it's actually expected to help, unlike the well-conditioned
// random systems benchmarked above.

template <typename value_t>
static void BM_lsqr_dense_illconditioned(benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CPU};

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, N});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    make_ill_conditioned_dense_system<value_t>(ctx, M, N, A, b);

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {N, 1});
    Workspace ws = create_lsqr_workspace(ctx, {M, N}, numeric_type_of<value_t>);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(50 * N);

    for (auto _ : state) {
        set_values(value_t{0}, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        benchmark::ClobberMemory();
    }

    double relres = relative_residual<value_t>(
        ctx, x, b,
        [&](Dense& xv, Dense& r) {
            mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, xv, 0.0,
                    r);
        },
        M);
    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["relres"] = relres;
}

BENCHMARK_TEMPLATE(BM_lsqr_dense_illconditioned, double)
    ->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename value_t>
static void BM_lsqr_dense_illconditioned_sketch_precond(
    benchmark::State& state) {
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    Context ctx{Device::CPU};

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, N});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {M, 1});
    make_ill_conditioned_dense_system<value_t>(ctx, M, N, A, b);

    const size_t k = std::min(M, 4 * N);
    Sparse S = Sparse::create_csr(ctx, numeric_type_of<value_t>,
                                  IntType::INT32, {k, M}, k);
    generate_count_sketch(S);
    Dense sketched = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                           numeric_type_of<value_t>, {k, N});
    apply_count_sketch(S, A, sketched);
    generate_sketch_qr_precond(S, sketched);

    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          numeric_type_of<value_t>, {N, N});
    auto vSketched = as_strided_view<value_t>(sketched);
    auto vPrecond = as_strided_view<value_t>(precond);
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i)
            vPrecond.values_[i + j * vPrecond.lead_dim_] =
                vSketched.values_[i + j * vSketched.lead_dim_];

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                    numeric_type_of<value_t>, {N, 1});
    Workspace ws = create_lsqr_workspace(
        ctx, {M, N}, numeric_type_of<value_t>, numeric_type_of<value_t>);
    add_precond(precond, ws);

    const double tol = 1e-10;
    const int max_iters = static_cast<int>(50 * N);

    for (auto _ : state) {
        set_values(value_t{0}, x);
        lsqr_solve(ctx, A, b, x, ws, tol, max_iters, tol);
        benchmark::ClobberMemory();
    }

    double relres = relative_residual<value_t>(
        ctx, x, b,
        [&](Dense& xv, Dense& r) {
            mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, xv, 0.0,
                    r);
        },
        M);
    state.counters["rows"] = static_cast<double>(M);
    state.counters["cols"] = static_cast<double>(N);
    state.counters["sketch_rows"] = static_cast<double>(k);
    state.counters["relres"] = relres;
}

BENCHMARK_TEMPLATE(BM_lsqr_dense_illconditioned_sketch_precond, double)
    ->Args({1000, 50})->Args({4000, 100})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace bench
}  // namespace tomos

TOMOS_BENCHMARK_MAIN()

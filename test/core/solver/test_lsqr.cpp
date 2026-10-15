#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "blas.hpp"
#include "cblas.h"
#include "context.hpp"
#include "countsketch.hpp"
#include "lsqr_impl.hpp"
#include "matrix.hpp"
#include "sketch_qr.hpp"
#include "solver.hpp"
#include "workspace.hpp"

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace tomos {
namespace test {

// Populate a Dense matrix (col-major) from a row-major std::vector.
static void fill_matrix(Dense& A, const std::vector<double>& row_major) {
    auto view = as_strided_view<double>(A);
    size_t m = view.size_[0], n = view.size_[1];
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            view.values_[i + j * view.lead_dim_] = row_major[i * n + j];
}

static void fill_vector(Dense& x, const std::vector<double>& data) {
    auto view = as_strided_view<double>(x);
    for (size_t i = 0; i < data.size(); ++i) view.values_[i] = data[i];
}

static std::vector<double> collect_vector(Dense& x) {
    auto view = as_strided_view<double>(x);
    size_t n  = view.size_[0];
    return std::vector<double>(view.values_, view.values_ + n);
}

// Minimal LSQR implementation using the dispatch layer.
// Solves min_x ||Ax - b||_2; terminates when |phi_bar|/||b|| < tol
// or after max_iter iterations.
// Returns {solution, final_relres}.
static std::pair<std::vector<double>, double> run_lsqr(Context& ctx, Dense& A,
                                                       Dense& b, int max_iter,
                                                       double tol) {
    auto Av  = as_strided_view<double>(A);
    size_t m = Av.size_[0], n = Av.size_[1];
    Device dev = A.get_device();

    Dense u_d = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                      {m, 1});
    Dense v_d = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                      {n, 1});
    Dense w_d = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                      {n, 1});
    Dense x_d = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                      {n, 1});

    set_values(0.0, u_d);
    set_values(0.0, v_d);
    set_values(0.0, w_d);
    set_values(0.0, x_d);

    auto bv = as_strided_view<double>(b);
    auto u  = as_strided_view<double>(u_d);
    auto v  = as_strided_view<double>(v_d);
    auto w  = as_strided_view<double>(w_d);
    auto x  = as_strided_view<double>(x_d);

    // β₁ u₁ = b
    cblas_dcopy(static_cast<int>(m), bv.values_, 1, u.values_, 1);
    double beta = cblas_dnrm2(static_cast<int>(m), u.values_, 1);
    cblas_dscal(static_cast<int>(m), 1.0 / beta, u.values_, 1);
    double bnorm = beta;

    // α₁ v₁ = Aᵀ u₁
    mv_apply(dev, TransposeType::TRANSPOSE, 1.0, A, u_d, 0.0, v_d);
    double alpha = cblas_dnrm2(static_cast<int>(n), v.values_, 1);
    cblas_dscal(static_cast<int>(n), 1.0 / alpha, v.values_, 1);

    // w = v
    cblas_dcopy(static_cast<int>(n), v.values_, 1, w.values_, 1);

    double phi_bar = beta, rho_bar = alpha;

    for (int iter = 0; iter < max_iter; ++iter) {
        // β u ← Av - α u
        mv_apply(dev, TransposeType::NO_TRANSPOSE, 1.0, A, v_d, -alpha, u_d);
        beta = cblas_dnrm2(static_cast<int>(m), u.values_, 1);
        cblas_dscal(static_cast<int>(m), 1.0 / beta, u.values_, 1);

        // α v ← Aᵀu - β v
        mv_apply(dev, TransposeType::TRANSPOSE, 1.0, A, u_d, -beta, v_d);
        alpha = cblas_dnrm2(static_cast<int>(n), v.values_, 1);
        cblas_dscal(static_cast<int>(n), 1.0 / alpha, v.values_, 1);

        // Givens rotation to eliminate β_{i+1}.
        double rho = std::hypot(rho_bar, beta);
        double c = rho_bar / rho, s = beta / rho;
        double theta = s * alpha;
        rho_bar      = -c * alpha;
        double phi   = c * phi_bar;
        phi_bar      = s * phi_bar;

        // x ← x + (φ/ρ) w
        cblas_daxpy(static_cast<int>(n), phi / rho, w.values_, 1, x.values_, 1);
        // w ← v - (θ/ρ) w
        cblas_dscal(static_cast<int>(n), -theta / rho, w.values_, 1);
        cblas_daxpy(static_cast<int>(n), 1.0, v.values_, 1, w.values_, 1);

        if (std::abs(phi_bar) / bnorm < tol) break;
    }

    // Compute ||Ax - b|| / ||b|| for the final x.
    Dense r_d = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                      {m, 1});
    set_values(0.0, r_d);
    mv_apply(dev, TransposeType::NO_TRANSPOSE, 1.0, A, x_d, 0.0, r_d);
    auto r = as_strided_view<double>(r_d);
    cblas_daxpy(static_cast<int>(m), -1.0, bv.values_, 1, r.values_, 1);
    double relres = cblas_dnrm2(static_cast<int>(m), r.values_, 1) / bnorm;

    return {collect_vector(x_d), relres};
}

class LsqrTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

// A = I₃, b = (1,2,3)ᵀ  →  exact solution in ~3 iterations.
TEST_F(LsqrTest, IdentitySystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    fill_vector(b, {1.0, 2.0, 3.0});

    auto [sol, relres] = run_lsqr(ctx, A, b, 50, 1e-10);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 3.0, 1e-8);
    EXPECT_LT(relres, 1e-8);
}

// A = diag(1,2,4), b = (1,4,8)ᵀ  →  x = (1,2,2)ᵀ.
TEST_F(LsqrTest, DiagonalSystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    fill_vector(b, {1.0, 4.0, 8.0});

    auto [sol, relres] = run_lsqr(ctx, A, b, 50, 1e-10);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 2.0, 1e-8);
    EXPECT_LT(relres, 1e-8);
}

// A is 4×2 with b = A * x_true (consistent), so LSQR finds the exact solution.
// x_true = (1, 1)ᵀ, A rows: [1,0],[0,1],[1,1],[1,-1].
TEST_F(LsqrTest, OverdeterminedConsistent) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    fill_matrix(A, {1, 0, 0, 1, 1, 1, 1, -1});
    fill_vector(b, {1.0, 1.0, 2.0, 0.0});

    auto [sol, relres] = run_lsqr(ctx, A, b, 100, 1e-10);
    EXPECT_NEAR(sol[0], 1.0, 1e-7);
    EXPECT_NEAR(sol[1], 1.0, 1e-7);
    EXPECT_LT(relres, 1e-8);
}

// A = [[1,1],[1,2],[1,3],[1,4]], b = [6,5,7,10].
// Least-squares solution via normal equations: x = (3.5, 1.4)ᵀ.
TEST_F(LsqrTest, OverdeterminedInconsistent) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    fill_matrix(A, {1, 1, 1, 2, 1, 3, 1, 4});
    fill_vector(b, {6.0, 5.0, 7.0, 10.0});

    auto [sol, relres] = run_lsqr(ctx, A, b, 200, 1e-10);
    EXPECT_NEAR(sol[0], 3.5, 1e-6);
    EXPECT_NEAR(sol[1], 1.4, 1e-6);
    // relres > 0 because the system is inconsistent; just verify convergence.
    EXPECT_LT(relres, 1.0);
}

// 6×3 consistent system; x_true = (1, -1, 2)ᵀ.
TEST_F(LsqrTest, WellConditionedOverdetermined) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {6, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {6, 1});
    // clang-format off
    std::vector<double> A_rows = {
        4, 1, 0,
        0, 3, 1,
        1, 0, 5,
        2, 1, 1,
        0, 2, 3,
        1, 1, 2
    };
    std::vector<double> x_true = {1.0, -1.0, 2.0};
    // clang-format on

    // b = A * x_true
    size_t m = 6, n = 3;
    std::vector<double> b_data(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            b_data[i] += A_rows[i * n + j] * x_true[j];

    fill_matrix(A, A_rows);
    fill_vector(b, b_data);

    auto [sol, relres] = run_lsqr(ctx, A, b, 200, 1e-10);
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(sol[i], x_true[i], 1e-7) << "x[" << i << "]";
    EXPECT_LT(relres, 1e-8);
}

// ---------------------------------------------------------------------------
// Tests using the public lsqr_solve() API (dispatch through registry).
// ---------------------------------------------------------------------------

// Helper: compute ||Ax - b|| / ||b|| to verify a solution.
static double dense_relres(Dense& A, Dense& x, Dense& b) {
    auto vA  = as_strided_view<double>(A);
    auto vx  = as_strided_view<double>(x);
    auto vb  = as_strided_view<double>(b);
    size_t m = vA.size_[0], n = vA.size_[1];
    std::vector<double> r(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            r[i] += vA.values_[i + j * vA.lead_dim_] *
                    static_cast<double>(vx.values_[j]);
    double bnorm = 0.0, rnorm = 0.0;
    for (size_t i = 0; i < m; ++i) {
        double ri = r[i] - static_cast<double>(vb.values_[i]);
        rnorm += ri * ri;
        double bi = static_cast<double>(vb.values_[i]);
        bnorm += bi * bi;
    }
    return std::sqrt(rnorm) / std::sqrt(bnorm);
}

class LsqrSolveTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

// Dense: A = I₃, b = (1,2,3)ᵀ  →  x = (1,2,3)ᵀ.
TEST_F(LsqrSolveTest, DenseIdentitySystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    fill_vector(b, {1.0, 2.0, 3.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 3.0, 1e-8);
    EXPECT_LT(dense_relres(A, x, b), 1e-8);
}

// Dense: A = diag(1,2,4), b = (1,4,8)ᵀ  →  x = (1,2,2)ᵀ.
TEST_F(LsqrSolveTest, DenseDiagonalSystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    fill_vector(b, {1.0, 4.0, 8.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 2.0, 1e-8);
    EXPECT_LT(dense_relres(A, x, b), 1e-8);
}

// Dense: 4×2 consistent overdetermined system, x_true = (1,1)ᵀ.
TEST_F(LsqrSolveTest, DenseOverdeterminedConsistent) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    fill_matrix(A, {1, 0, 0, 1, 1, 1, 1, -1});
    fill_vector(b, {1.0, 1.0, 2.0, 0.0});

    Workspace ws = create_lsqr_workspace(ctx, {4, 2}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 100, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-7);
    EXPECT_NEAR(sol[1], 1.0, 1e-7);
    EXPECT_LT(dense_relres(A, x, b), 1e-8);
}

// Dense: inconsistent overdetermined system; verify LS solution.
TEST_F(LsqrSolveTest, DenseOverdeterminedInconsistent) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    fill_matrix(A, {1, 1, 1, 2, 1, 3, 1, 4});
    fill_vector(b, {6.0, 5.0, 7.0, 10.0});

    Workspace ws = create_lsqr_workspace(ctx, {4, 2}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 200, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 3.5, 1e-6);
    EXPECT_NEAR(sol[1], 1.4, 1e-6);
}

// Dense: FP32 system — verifies the FP32 registry entry is populated.
TEST_F(LsqrSolveTest, DenseFP32System) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {3, 1});
    auto vA = as_strided_view<float>(A);
    auto vb = as_strided_view<float>(b);
    std::vector<double> Arows = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            vA.values_[i + j * vA.lead_dim_] =
                static_cast<float>(Arows[i * 3 + j]);
    vb.values_[0] = 1.0f;
    vb.values_[1] = 4.0f;
    vb.values_[2] = 9.0f;

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32);
    lsqr_solve(ctx, A, b, x, ws, 1e-5, 50, 1e-5);

    auto vx = as_strided_view<float>(x);
    EXPECT_NEAR(static_cast<double>(vx.values_[0]), 1.0, 1e-5);
    EXPECT_NEAR(static_cast<double>(vx.values_[1]), 2.0, 1e-5);
    EXPECT_NEAR(static_cast<double>(vx.values_[2]), 3.0, 1e-5);
}

// Sparse: diagonal 3×3 CSR matrix, b = (1,4,8)ᵀ  →  x = (1,2,2)ᵀ.
TEST_F(LsqrSolveTest, SparseDiagonalSystem) {
    // Build CSR: diag(1, 2, 4).
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {3, 3}, 3);
    auto vA         = as_csr_view<double, int>(A);
    vA.row_ptrs_[0] = 0;
    vA.row_ptrs_[1] = 1;
    vA.row_ptrs_[2] = 2;
    vA.row_ptrs_[3] = 3;
    vA.col_idxs_[0] = 0;
    vA.col_idxs_[1] = 1;
    vA.col_idxs_[2] = 2;
    vA.values_[0]   = 1.0;
    vA.values_[1]   = 2.0;
    vA.values_[2]   = 4.0;

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector(b, {1.0, 4.0, 8.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 2.0, 1e-8);
}

// Sparse: 4×2 overdetermined consistent system, x_true = (1,1)ᵀ.
// A rows: [1,0],[0,1],[1,1],[1,-1].  nnz = 6.
TEST_F(LsqrSolveTest, SparseOverdeterminedConsistent) {
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {4, 2}, 6);
    auto vA         = as_csr_view<double, int>(A);
    // row 0: col 0 → 1.0
    // row 1: col 1 → 1.0
    // row 2: col 0 → 1.0, col 1 → 1.0
    // row 3: col 0 → 1.0, col 1 → -1.0
    vA.row_ptrs_[0] = 0;
    vA.row_ptrs_[1] = 1;
    vA.row_ptrs_[2] = 2;
    vA.row_ptrs_[3] = 4;
    vA.row_ptrs_[4] = 6;
    vA.col_idxs_[0] = 0;
    vA.col_idxs_[1] = 1;
    vA.col_idxs_[2] = 0;
    vA.col_idxs_[3] = 1;
    vA.col_idxs_[4] = 0;
    vA.col_idxs_[5] = 1;
    vA.values_[0]   = 1.0;
    vA.values_[1]   = 1.0;
    vA.values_[2]   = 1.0;
    vA.values_[3]   = 1.0;
    vA.values_[4]   = 1.0;
    vA.values_[5]   = -1.0;

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    fill_vector(b, {1.0, 1.0, 2.0, 0.0});

    Workspace ws = create_lsqr_workspace(ctx, {4, 2}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 100, 1e-10);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-7);
    EXPECT_NEAR(sol[1], 1.0, 1e-7);
}

// ---------------------------------------------------------------------------
// Sketch-and-precondition LSQR: builds a right-preconditioner from a count
// sketch + QR (generate_count_sketch / apply_count_sketch /
// generate_sketch_qr_precond), attaches it to a Workspace via add_precond(),
// and solves through the public lsqr_solve() dispatch.
// ---------------------------------------------------------------------------

class LsqrSketchPrecondTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

TEST_F(LsqrSketchPrecondTest, ViaPublicLsqrSolveAPI) {
    const size_t m = 40, n = 5, k = 30;  // k >= n, k << m.

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {m, n});
    auto vA = as_strided_view<double>(A);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            vA.values_[i + j * vA.lead_dim_] = dist(gen);

    std::vector<double> x_true(n);
    for (auto& v : x_true) v = dist(gen);
    Dense x_true_d = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                           NumericType::FP64, {n, 1});
    fill_vector(x_true_d, x_true);

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {m, 1});
    set_values(0.0, b);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x_true_d, 0.0,
             b);

    Sparse S =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {k, m}, k);
    generate_count_sketch(S);
    Dense sketched = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                           NumericType::FP64, {k, n});
    apply_count_sketch(S, A, sketched);
    generate_sketch_qr_precond(S, sketched);

    Dense precond  = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                           NumericType::FP64, {n, n});
    auto vSketched = as_strided_view<double>(sketched);
    auto vPrecond  = as_strided_view<double>(precond);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            vPrecond.values_[i + j * vPrecond.lead_dim_] =
                vSketched.values_[i + j * vSketched.lead_dim_];

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {n, 1});

    Workspace ws = create_lsqr_workspace(ctx, {m, n}, NumericType::FP64);
    add_precond(precond, ws);
    lsqr_solve(ctx, A, b, x, ws, /*tol=*/1e-10, /*max_iters=*/100,
               /*max_relres=*/1e-10);

    auto sol = collect_vector(x);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(sol[i], x_true[i], 1e-6);

    Dense r = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {m, 1});
    set_values(0.0, r);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, r);
    auto vr = as_strided_view<double>(r);
    auto vb = as_strided_view<double>(b);
    cblas_daxpy(static_cast<int>(m), -1.0, vb.values_, 1, vr.values_, 1);
    double bnorm  = cblas_dnrm2(static_cast<int>(m), vb.values_, 1);
    double relres = cblas_dnrm2(static_cast<int>(m), vr.values_, 1) / bnorm;
    EXPECT_LT(relres, 1e-8);
}

// ---------------------------------------------------------------------------
// Comparison tests: ref_fused_lsqr_impl (lsqr_impl.hpp) vs lsqr_solve.
// Both are run on identical inputs and the results are compared element-wise.
// ---------------------------------------------------------------------------

class LsqrCompareTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};

    // Run ref_fused_lsqr_impl on a Dense system and return the solution.
    std::vector<double> run_ref(Dense& A, Dense& b, int max_iter, double tol) {
        auto vA = as_strided_view<double>(A);
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP64, {vA.size_[1], 1});
        set_values(0.0, x);

        RefLsqrWorkspace<Device::CPU, double> ws(ctx, vA.size_, max_iter, tol,
                                                 tol);
        ws.mtx_r_      = &A;
        ws.rhs_refine_ = &b;
        ws.sol_refine_ = &x;
        ref_fused_lsqr_impl<Device::CPU, double>(&ws);
        return collect_vector(x);
    }

    // Run lsqr_solve on a Dense system and return the solution.
    std::vector<double> run_prod(Dense& A, Dense& b, int max_iter, double tol) {
        auto vA = as_strided_view<double>(A);
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP64, {vA.size_[1], 1});
        Workspace ws_buf =
            create_lsqr_workspace(ctx, vA.size_, NumericType::FP64);
        lsqr_solve(ctx, A, b, x, ws_buf, tol, max_iter, tol);
        return collect_vector(x);
    }
};

// Identity system: both implementations should agree to high precision.
TEST_F(LsqrCompareTest, DenseIdentitySystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    fill_vector(b, {1.0, 2.0, 3.0});

    auto ref  = run_ref(A, b, 50, 1e-10);
    auto prod = run_prod(A, b, 50, 1e-10);
    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(ref[i], prod[i], 1e-8) << "x[" << i << "]";
}

// Diagonal system.
TEST_F(LsqrCompareTest, DenseDiagonalSystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    fill_vector(b, {1.0, 4.0, 8.0});

    auto ref  = run_ref(A, b, 50, 1e-10);
    auto prod = run_prod(A, b, 50, 1e-10);
    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(ref[i], prod[i], 1e-8) << "x[" << i << "]";
}

// Overdetermined consistent system.
TEST_F(LsqrCompareTest, DenseOverdeterminedConsistent) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 2});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    fill_matrix(A, {1, 0, 0, 1, 1, 1, 1, -1});
    fill_vector(b, {1.0, 1.0, 2.0, 0.0});

    auto ref  = run_ref(A, b, 100, 1e-10);
    auto prod = run_prod(A, b, 100, 1e-10);
    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(ref[i], prod[i], 1e-7) << "x[" << i << "]";
}

// ---------------------------------------------------------------------------
// Multiprecision tests: mv_type (A), value_type (Krylov vectors), refine_type
// (b/x), and precond_type are independently selectable. b/x stay FP64 so the
// existing fill_vector/collect_vector helpers apply unchanged; only A (and,
// in the preconditioned case, the preconditioner) need FP16 population,
// which -- since __half converts to/from float/double on the host too (see
// cuda_fp16.h) -- is just a direct static_cast into as_strided_view<__half>.
// ---------------------------------------------------------------------------

class LsqrMultiprecisionTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

static void fill_matrix_half(Dense& A, const std::vector<double>& row_major) {
    auto view = as_strided_view<__half>(A);
    size_t m = view.size_[0], n = view.size_[1];
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            view.values_[i + j * view.lead_dim_] =
                static_cast<__half>(row_major[i * n + j]);
}

// A: FP16 (mv_type), Krylov vectors: FP32 (value_type), b/x: FP64
// (refine_type) -- the storage/compute split the user asked for: A is
// stored compactly, arithmetic runs in FP32, the solution is refined in
// FP64.
TEST_F(LsqrMultiprecisionTest, DenseDiagonalMixedPrecision) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP16,
                                    {3, 3});
    fill_matrix_half(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector(b, {1.0, 4.0, 8.0});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32);
    lsqr_solve(ctx, A, b, x, ws, 1e-6, 50, 1e-6);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 5e-2);
    EXPECT_NEAR(sol[1], 2.0, 5e-2);
    EXPECT_NEAR(sol[2], 2.0, 5e-2);
}

// Same system, but also attaches a preconditioner stored in FP16
// (precond_type) while the Krylov vectors stay FP32 (value_type) -- the
// preconditioner is independently sized/typed via create_lsqr_workspace's
// precond_type parameter and validated by add_precond().
TEST_F(LsqrMultiprecisionTest, DensePreconditionedMixedPrecision) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    fill_matrix(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector(b, {1.0, 4.0, 8.0});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});

    // Precond ~ A's inverse diagonal, perturbed so it's approximate rather
    // than exact -- an exact inverse converges in exactly one iteration and
    // can drive the residual norm to precisely zero mid-iteration (before
    // the stopping check runs), hitting a divide-by-zero that's a
    // pre-existing LSQR robustness gap unrelated to precision; approximate
    // preconditioners (the realistic case, e.g. LsqrSketchPrecondTest above)
    // don't hit it. Stored FP16.
    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          NumericType::FP16, {3, 3});
    fill_matrix_half(precond, {0.9, 0, 0, 0, 0.45, 0, 0, 0, 0.2});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32,
                                         NumericType::FP16);
    add_precond(precond, ws);
    lsqr_solve(ctx, A, b, x, ws, 1e-8, 50, 1e-8);

    auto sol = collect_vector(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-2);
    EXPECT_NEAR(sol[1], 2.0, 1e-2);
    EXPECT_NEAR(sol[2], 2.0, 1e-2);
}

// add_precond() must reject a preconditioner whose NumericType doesn't match
// what create_lsqr_workspace's precond_type declared.
TEST_F(LsqrMultiprecisionTest, AddPrecondRejectsTypeMismatch) {
    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          NumericType::FP64, {3, 3});
    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32,
                                         NumericType::FP16);
    EXPECT_THROW(add_precond(precond, ws), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Exhaustive sweep: every (mv_type, value_type, refine_type, precond_type)
// combination dispatch::Lsqr registers -- 3^4 = 81 combinations, times every
// active device (CPU always; CPU+CUDA when built with CUDA support), times
// dense and sparse A -- 324 cases total on a CUDA build. Each case solves a
// small, well-conditioned preconditioned system; the tolerance is loosened
// whenever FP16 appears in any of the four slots, since FP16 storage (~3
// decimal digits) caps the achievable accuracy regardless of which slot it
// occupies. The preconditioner is approximate (not exact) throughout -- see
// DensePreconditionedMixedPrecision above for why an exact one isn't used.
// The sparse case reuses SparseOverdeterminedConsistent's 4x2 CSR structure
// (rows sharing columns 0 and 1) specifically so it exercises real multi-row
// atomic accumulation in the transpose SpMV kernel, not just a diagonal
// matrix where each output entry has exactly one contributor.
// ---------------------------------------------------------------------------

template <typename T>
static void fill_matrix_typed(Dense& A, const std::vector<double>& row_major) {
    auto view = as_strided_view<T>(A);
    size_t m = view.size_[0], n = view.size_[1];
    std::vector<T> col_major(view.lead_dim_ * n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            col_major[i + j * view.lead_dim_] =
                static_cast<T>(row_major[i * n + j]);
#if defined(USE_CUDA)
    if (A.get_device() == Device::CUDA) {
        cudaMemcpy(view.values_, col_major.data(),
                  col_major.size() * sizeof(T), cudaMemcpyHostToDevice);
        return;
    }
#endif
    std::copy(col_major.begin(), col_major.end(), view.values_);
}

template <typename T>
static std::vector<double> collect_vector_typed(Dense& x) {
    auto view = as_strided_view<T>(x);
    size_t n = view.size_[0];
    std::vector<T> buf(n);
#if defined(USE_CUDA)
    if (x.get_device() == Device::CUDA) {
        cudaMemcpy(buf.data(), view.values_, n * sizeof(T),
                  cudaMemcpyDeviceToHost);
    } else
#endif
    {
        std::copy(view.values_, view.values_ + n, buf.begin());
    }
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) result[i] = static_cast<double>(buf[i]);
    return result;
}

// Dispatches to the right instantiation of fill_matrix_typed/
// collect_vector_typed based on the Dense's own runtime NumericType --
// row_major doubles in, row_major doubles out, regardless of storage type.
static void fill_matrix_by_type(Dense& A, const std::vector<double>& row_major) {
    switch (A.get_value_type()) {
        case NumericType::FP64: fill_matrix_typed<double>(A, row_major); return;
        case NumericType::FP32: fill_matrix_typed<float>(A, row_major); return;
        case NumericType::FP16: fill_matrix_typed<__half>(A, row_major); return;
        default:
            throw std::invalid_argument(
                "fill_matrix_by_type: unsupported NumericType.");
    }
}

static std::vector<double> collect_vector_by_type(Dense& x) {
    switch (x.get_value_type()) {
        case NumericType::FP64: return collect_vector_typed<double>(x);
        case NumericType::FP32: return collect_vector_typed<float>(x);
        case NumericType::FP16: return collect_vector_typed<__half>(x);
        default:
            throw std::invalid_argument(
                "collect_vector_by_type: unsupported NumericType.");
    }
}

// Fills a 4x2 CSR matrix with rows [1,0],[0,1],[1,1],[1,-1] (nnz=6, matching
// SparseOverdeterminedConsistent above) -- columns 0 and 1 each receive
// contributions from 3 different rows, so A^T*x genuinely exercises the
// transpose SpMV kernel's cross-row atomic accumulation.
template <typename T>
static void fill_csr_4x2_typed(Sparse& A) {
    auto view = as_csr_view<T, int>(A);
    std::vector<int> row_ptrs = {0, 1, 2, 4, 6};
    std::vector<int> col_idxs = {0, 1, 0, 1, 0, 1};
    std::vector<T> values = {static_cast<T>(1.0), static_cast<T>(1.0),
                             static_cast<T>(1.0), static_cast<T>(1.0),
                             static_cast<T>(1.0), static_cast<T>(-1.0)};
#if defined(USE_CUDA)
    if (A.get_device() == Device::CUDA) {
        cudaMemcpy(view.row_ptrs_, row_ptrs.data(),
                  row_ptrs.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(view.col_idxs_, col_idxs.data(),
                  col_idxs.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(view.values_, values.data(), values.size() * sizeof(T),
                  cudaMemcpyHostToDevice);
        return;
    }
#endif
    std::copy(row_ptrs.begin(), row_ptrs.end(), view.row_ptrs_);
    std::copy(col_idxs.begin(), col_idxs.end(), view.col_idxs_);
    std::copy(values.begin(), values.end(), view.values_);
}

static void fill_csr_4x2_by_type(Sparse& A) {
    switch (A.get_value_type()) {
        case NumericType::FP64: fill_csr_4x2_typed<double>(A); return;
        case NumericType::FP32: fill_csr_4x2_typed<float>(A); return;
        case NumericType::FP16: fill_csr_4x2_typed<__half>(A); return;
        default:
            throw std::invalid_argument(
                "fill_csr_4x2_by_type: unsupported NumericType.");
    }
}

struct PrecisionCombo {
    Device device;
    bool sparse;
    NumericType mv_type, value_type, refine_type, precond_type;
};

static std::vector<PrecisionCombo> AllPrecisionCombos() {
    const NumericType types[] = {NumericType::FP64, NumericType::FP32,
                                 NumericType::FP16};
    std::vector<Device> devices = {Device::CPU};
#if defined(USE_CUDA)
    devices.push_back(Device::CUDA);
#endif
    const bool sparse_options[] = {false, true};

    std::vector<PrecisionCombo> combos;
    for (Device dev : devices)
        for (bool sparse : sparse_options)
            for (NumericType mv : types)
                for (NumericType val : types)
                    for (NumericType refine : types)
                        for (NumericType precond : types)
                            combos.push_back(
                                {dev, sparse, mv, val, refine, precond});
    return combos;
}

static std::string PrecisionComboName(
    const ::testing::TestParamInfo<PrecisionCombo>& info) {
    auto name_of = [](NumericType t) -> std::string {
        switch (t) {
            case NumericType::FP64: return "FP64";
            case NumericType::FP32: return "FP32";
            case NumericType::FP16: return "FP16";
            default: return "Unknown";
        }
    };
    const PrecisionCombo& p = info.param;
    std::string dev = (p.device == Device::CUDA) ? "CUDA" : "CPU";
    std::string fmt = p.sparse ? "Sparse" : "Dense";
    return dev + "_" + fmt + "_mv" + name_of(p.mv_type) + "_value" +
           name_of(p.value_type) + "_refine" + name_of(p.refine_type) +
           "_precond" + name_of(p.precond_type);
}

class LsqrAllPrecisionCombosTest
    : public ::testing::TestWithParam<PrecisionCombo> {};

TEST_P(LsqrAllPrecisionCombosTest, PreconditionedSystem) {
    const PrecisionCombo& p = GetParam();
    Context ctx{p.device};

    bool any_half = p.mv_type == NumericType::FP16 ||
                    p.value_type == NumericType::FP16 ||
                    p.refine_type == NumericType::FP16 ||
                    p.precond_type == NumericType::FP16;
    double tol = any_half ? 0.2 : 1e-3;

    auto check = [&](const std::vector<double>& sol,
                     const std::vector<double>& expected) {
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(sol[i], expected[i], tol) << "x[" << i << "]";
    };

    if (!p.sparse) {
        Dense A =
            Dense::create_strided(ctx, Layout::COL_MAJOR, p.mv_type, {3, 3});
        fill_matrix_by_type(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
        Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, p.refine_type,
                                        {3, 1});
        fill_matrix_by_type(b, {1.0, 4.0, 8.0});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, p.refine_type,
                                        {3, 1});

        // Approximate (not exact) inverse of A's diagonal.
        Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                              p.precond_type, {3, 3});
        fill_matrix_by_type(precond, {0.9, 0, 0, 0, 0.45, 0, 0, 0, 0.2});

        Workspace ws =
            create_lsqr_workspace(ctx, {3, 3}, p.value_type, p.precond_type);
        add_precond(precond, ws);
        lsqr_solve(ctx, A, b, x, ws, 1e-8, 100, 1e-8);

        check(collect_vector_by_type(x), {1.0, 2.0, 2.0});
    } else {
        Sparse A = Sparse::create_csr(ctx, p.mv_type, IntType::INT32, {4, 2}, 6);
        fill_csr_4x2_by_type(A);
        Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, p.refine_type,
                                        {4, 1});
        fill_matrix_by_type(b, {1.0, 1.0, 2.0, 0.0});
        Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, p.refine_type,
                                        {2, 1});

        // Mild, approximate (not exact) preconditioner.
        Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                              p.precond_type, {2, 2});
        fill_matrix_by_type(precond, {0.9, 0, 0, 0.9});

        Workspace ws =
            create_lsqr_workspace(ctx, {4, 2}, p.value_type, p.precond_type);
        add_precond(precond, ws);
        lsqr_solve(ctx, A, b, x, ws, 1e-8, 100, 1e-8);

        check(collect_vector_by_type(x), {1.0, 1.0});
    }
}

INSTANTIATE_TEST_SUITE_P(AllCombos, LsqrAllPrecisionCombosTest,
                        ::testing::ValuesIn(AllPrecisionCombos()),
                        PrecisionComboName);

#if defined(USE_CUDA)

// ---------------------------------------------------------------------------
// CUDA tests: same problems as LsqrSolveTest/LsqrSketchPrecondTest above, but
// with A/b/x/precond resident on Device::CUDA. Data is built on the host and
// copied in with cudaMemcpy directly (mirroring how test/kernels/cuda/*
// kernel tests move data), since Dense::copy_from has no registered
// CPU<->CUDA path yet.
// ---------------------------------------------------------------------------

static void fill_matrix_cuda(Dense& A, const std::vector<double>& row_major) {
    auto view = as_strided_view<double>(A);
    size_t m = view.size_[0], n = view.size_[1];
    std::vector<double> col_major(view.lead_dim_ * n, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            col_major[i + j * view.lead_dim_] = row_major[i * n + j];
    cudaMemcpy(view.values_, col_major.data(),
              col_major.size() * sizeof(double), cudaMemcpyHostToDevice);
}

static void fill_vector_cuda(Dense& x, const std::vector<double>& data) {
    auto view = as_strided_view<double>(x);
    cudaMemcpy(view.values_, data.data(), data.size() * sizeof(double),
              cudaMemcpyHostToDevice);
}

static std::vector<double> collect_vector_cuda(Dense& x) {
    auto view = as_strided_view<double>(x);
    size_t n = view.size_[0];
    std::vector<double> result(n);
    cudaMemcpy(result.data(), view.values_, n * sizeof(double),
              cudaMemcpyDeviceToHost);
    return result;
}

class LsqrSolveCudaTest : public ::testing::Test {
protected:
    Context ctx{Device::CUDA};
};

// Dense: A = I₃, b = (1,2,3)ᵀ  →  x = (1,2,3)ᵀ.
TEST_F(LsqrSolveCudaTest, DenseIdentitySystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix_cuda(A, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    fill_vector_cuda(b, {1.0, 2.0, 3.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 3.0, 1e-8);
}

// Dense: A = diag(1,2,4), b = (1,4,8)ᵀ  →  x = (1,2,2)ᵀ.
TEST_F(LsqrSolveCudaTest, DenseDiagonalSystem) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_matrix_cuda(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    fill_vector_cuda(b, {1.0, 4.0, 8.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 2.0, 1e-8);
}

// Sparse CSR, diagonal — exercises cuda::mv_csr for both A*v and A^T*u.
TEST_F(LsqrSolveCudaTest, SparseDiagonalSystem) {
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {3, 3}, 3);
    auto vA = as_csr_view<double, int>(A);
    std::vector<int> row_ptrs = {0, 1, 2, 3};
    std::vector<int> col_idxs = {0, 1, 2};
    std::vector<double> values = {1.0, 2.0, 4.0};
    cudaMemcpy(vA.row_ptrs_, row_ptrs.data(), row_ptrs.size() * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(vA.col_idxs_, col_idxs.data(), col_idxs.size() * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(vA.values_, values.data(), values.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector_cuda(b, {1.0, 4.0, 8.0});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 50, 1e-10);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-8);
    EXPECT_NEAR(sol[1], 2.0, 1e-8);
    EXPECT_NEAR(sol[2], 2.0, 1e-8);
}

// Sparse CSR, non-symmetric rectangular — A^T*u and A*v hit genuinely
// different code, so this exercises spmv_csr_transpose_kernel's atomic
// accumulation (not just the untransposed path the diagonal case above
// can't distinguish from a transpose).
TEST_F(LsqrSolveCudaTest, SparseOverdeterminedConsistent) {
    Sparse A =
        Sparse::create_csr(ctx, NumericType::FP64, IntType::INT32, {4, 2}, 6);
    auto vA = as_csr_view<double, int>(A);
    std::vector<int> row_ptrs = {0, 1, 2, 4, 6};
    std::vector<int> col_idxs = {0, 1, 0, 1, 0, 1};
    std::vector<double> values = {1.0, 1.0, 1.0, 1.0, 1.0, -1.0};
    cudaMemcpy(vA.row_ptrs_, row_ptrs.data(), row_ptrs.size() * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(vA.col_idxs_, col_idxs.data(), col_idxs.size() * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(vA.values_, values.data(), values.size() * sizeof(double),
              cudaMemcpyHostToDevice);

    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {4, 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {2, 1});
    fill_vector_cuda(b, {1.0, 1.0, 2.0, 0.0});

    Workspace ws = create_lsqr_workspace(ctx, {4, 2}, NumericType::FP64);
    lsqr_solve(ctx, A, b, x, ws, 1e-10, 100, 1e-10);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-7);
    EXPECT_NEAR(sol[1], 1.0, 1e-7);
}

// Preconditioned dense LSQR on CUDA: build A/b and the sketch-QR
// preconditioner on the CPU (reusing the exact pipeline from
// LsqrSketchPrecondTest above, which is already verified there), then copy
// A/b/precond to CUDA and solve entirely on the GPU. This isolates what
// this change actually touches -- LSQR's iterate loop and preconditioner
// *application* on CUDA -- from sketch_qr's own CUDA kernels, which are a
// separate, pre-existing piece of the codebase this change doesn't modify.
TEST_F(LsqrSolveCudaTest, DensePreconditioned) {
    const size_t m = 40, n = 5, k = 30;  // k >= n, k << m.

    Context cpu_ctx{Device::CPU};
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> A_row_major(m * n);
    for (auto& v : A_row_major) v = dist(gen);
    Dense A_h = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR,
                                      NumericType::FP64, {m, n});
    fill_matrix(A_h, A_row_major);

    std::vector<double> x_true(n);
    for (auto& v : x_true) v = dist(gen);
    Dense x_true_h = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR,
                                           NumericType::FP64, {n, 1});
    fill_vector(x_true_h, x_true);

    Dense b_h = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR,
                                      NumericType::FP64, {m, 1});
    set_values(0.0, b_h);
    mv_apply(Device::CPU, TransposeType::NO_TRANSPOSE, 1.0, A_h, x_true_h, 0.0,
             b_h);
    std::vector<double> b_data = collect_vector(b_h);

    Sparse S = Sparse::create_csr(cpu_ctx, NumericType::FP64, IntType::INT32,
                                  {k, m}, k);
    generate_count_sketch(S);
    Dense sketched = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR,
                                           NumericType::FP64, {k, n});
    apply_count_sketch(S, A_h, sketched);
    generate_sketch_qr_precond(S, sketched);

    auto vSketched = as_strided_view<double>(sketched);
    std::vector<double> precond_row_major(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            precond_row_major[i * n + j] =
                vSketched.values_[i + j * vSketched.lead_dim_];

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {m, n});
    fill_matrix_cuda(A, A_row_major);
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {m, 1});
    fill_vector_cuda(b, b_data);
    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          NumericType::FP64, {n, n});
    fill_matrix_cuda(precond, precond_row_major);
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {n, 1});

    Workspace ws = create_lsqr_workspace(ctx, {m, n}, NumericType::FP64);
    add_precond(precond, ws);
    lsqr_solve(ctx, A, b, x, ws, /*tol=*/1e-10, /*max_iters=*/100,
               /*max_relres=*/1e-10);

    auto sol = collect_vector_cuda(x);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(sol[i], x_true[i], 1e-6);
}

// A: FP16 (mv_type), Krylov vectors: FP32 (value_type), b/x: FP64
// (refine_type) -- the CUDA counterpart of
// LsqrMultiprecisionTest.DenseDiagonalMixedPrecision above.
static void fill_matrix_half_cuda(Dense& A,
                                  const std::vector<double>& row_major) {
    auto view = as_strided_view<__half>(A);
    size_t m = view.size_[0], n = view.size_[1];
    std::vector<__half> col_major(view.lead_dim_ * n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            col_major[i + j * view.lead_dim_] =
                static_cast<__half>(row_major[i * n + j]);
    cudaMemcpy(view.values_, col_major.data(),
              col_major.size() * sizeof(__half), cudaMemcpyHostToDevice);
}

TEST_F(LsqrSolveCudaTest, DenseDiagonalMixedPrecision) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP16,
                                    {3, 3});
    fill_matrix_half_cuda(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector_cuda(b, {1.0, 4.0, 8.0});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32);
    lsqr_solve(ctx, A, b, x, ws, 1e-6, 50, 1e-6);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 5e-2);
    EXPECT_NEAR(sol[1], 2.0, 5e-2);
    EXPECT_NEAR(sol[2], 2.0, 5e-2);
}

// Preconditioner stored FP16 (precond_type), independent of the FP32
// Krylov vectors (value_type) -- the CUDA counterpart of
// LsqrMultiprecisionTest.DensePreconditionedMixedPrecision above.
TEST_F(LsqrSolveCudaTest, DensePreconditionedMixedPrecision) {
    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 3});
    fill_matrix_cuda(A, {1, 0, 0, 0, 2, 0, 0, 0, 4});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});
    fill_vector_cuda(b, {1.0, 4.0, 8.0});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP64,
                                    {3, 1});

    // Approximate (not exact) inverse -- see the CPU counterpart test above
    // for why an exact preconditioner isn't used here.
    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          NumericType::FP16, {3, 3});
    fill_matrix_half_cuda(precond, {0.9, 0, 0, 0, 0.45, 0, 0, 0, 0.2});

    Workspace ws = create_lsqr_workspace(ctx, {3, 3}, NumericType::FP32,
                                         NumericType::FP16);
    add_precond(precond, ws);
    lsqr_solve(ctx, A, b, x, ws, 1e-8, 50, 1e-8);

    auto sol = collect_vector_cuda(x);
    EXPECT_NEAR(sol[0], 1.0, 1e-2);
    EXPECT_NEAR(sol[1], 2.0, 1e-2);
    EXPECT_NEAR(sol[2], 2.0, 1e-2);
}
#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos
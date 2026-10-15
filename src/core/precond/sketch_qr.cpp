#include "sketch_qr.hpp"

#include <lapacke.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "precision.hpp"

#if USE_CUDA
#include "sketch_qr_cuda_kernels.cuh"
#endif

namespace tomos {

namespace {

lapack_int geqrf_lapacke(int m, int n, double* a, int lda, double* tau) {
    return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
lapack_int geqrf_lapacke(int m, int n, float* a, int lda, float* tau) {
    return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}

lapack_int trtri_lapacke(int n, double* a, int lda) {
    return LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', n, a, lda);
}
lapack_int trtri_lapacke(int n, float* a, int lda) {
    return LAPACKE_strtri(LAPACK_COL_MAJOR, 'U', 'N', n, a, lda);
}

// Factors the k x n (k >= n) matrix `values` via QR, then inverts the
// resulting n x n upper-triangular R factor in-place, leaving R^{-1} in the
// leading n x n block.
template <typename value_t>
void generate_sketch_qr_precond_cpu(dim<2> size, value_t* values, size_t ld) {
    int k = static_cast<int>(size[0]);
    int n = static_cast<int>(size[1]);
    if (k < n)
        throw std::invalid_argument(
            "generate_sketch_qr_precond: sketched_mtx must have at least as "
            "many rows as columns (k >= n).");

    std::vector<value_t> tau(n);
    lapack_int           info =
        geqrf_lapacke(k, n, values, static_cast<int>(ld), tau.data());
    if (info != 0)
        throw std::runtime_error(
            "generate_sketch_qr_precond: geqrf failed (info=" +
            std::to_string(info) + ").");

    // geqrf packs Householder reflectors below the diagonal; zero them so
    // only the leading n x n upper-triangular R factor remains.
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i) values[i + j * ld] = value_t(0);

    info = trtri_lapacke(n, values, static_cast<int>(ld));
    if (info != 0)
        throw std::runtime_error(
            "generate_sketch_qr_precond: trtri failed (info=" +
            std::to_string(info) + ") -- R is singular.");
}

}  // namespace

void generate_sketch_qr_precond(Sparse& sketch, Dense& sketched_mtx) {
    if (sketch.get_device() != sketched_mtx.get_device())
        throw std::invalid_argument(
            "generate_sketch_qr_precond: sketch and sketched_mtx must be on "
            "the same device.");
    if (sketched_mtx.get_device() == Device::CPU) {
        switch (sketched_mtx.get_value_type()) {
            case NumericType::FP64: {
                auto view = as_strided_view<double>(sketched_mtx);
                generate_sketch_qr_precond_cpu<double>(
                    view.size_, view.values_, view.lead_dim_);
                break;
            }
            case NumericType::FP32: {
                auto view = as_strided_view<float>(sketched_mtx);
                generate_sketch_qr_precond_cpu<float>(
                    view.size_, view.values_, view.lead_dim_);
                break;
            }
            default:
                throw std::runtime_error(
                    "generate_sketch_qr_precond: unsupported NumericType "
                    "(only FP64/FP32 are supported).");
        }
        return;
    }

#if USE_CUDA
    if (sketched_mtx.get_device() == Device::CUDA) {
        switch (sketched_mtx.get_value_type()) {
            case NumericType::FP64: {
                auto view = as_strided_view<double>(sketched_mtx);
                cuda::generate_sketch_qr_precond<double>(
                    view.size_, view.values_, view.lead_dim_);
                break;
            }
            case NumericType::FP32: {
                auto view = as_strided_view<float>(sketched_mtx);
                cuda::generate_sketch_qr_precond<float>(
                    view.size_, view.values_, view.lead_dim_);
                break;
            }
            default:
                throw std::runtime_error(
                    "generate_sketch_qr_precond: unsupported NumericType "
                    "(only FP64/FP32 are supported).");
        }
        return;
    }
#endif

    throw std::runtime_error(
        "generate_sketch_qr_precond: unsupported device. Rebuild with "
        "-DDEVICE_BACKEND=CUDA to enable CUDA support.");
}

}  // namespace tomos

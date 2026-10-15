#include "sketch_qr_cuda_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdexcept>
#include <string>

namespace tomos {
namespace cuda {

namespace {

template <typename value_t>
__global__ void zero_subdiagonal_kernel(int n, value_t* a, int lda) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && i > j) a[i + j * lda] = value_t(0);
}

template <typename value_t>
__global__ void set_identity_kernel(int n, value_t* a, int lda) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) a[i + j * lda] = (i == j) ? value_t(1) : value_t(0);
}

cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t h, int m, int n,
                                   double* a, int lda, int* lwork) {
    return cusolverDnDgeqrf_bufferSize(h, m, n, a, lda, lwork);
}
cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t h, int m, int n,
                                   float* a, int lda, int* lwork) {
    return cusolverDnSgeqrf_bufferSize(h, m, n, a, lda, lwork);
}

cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, double* a, int lda,
                       double* tau, double* work, int lwork, int* devInfo) {
    return cusolverDnDgeqrf(h, m, n, a, lda, tau, work, lwork, devInfo);
}
cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, float* a, int lda,
                       float* tau, float* work, int lwork, int* devInfo) {
    return cusolverDnSgeqrf(h, m, n, a, lda, tau, work, lwork, devInfo);
}

cublasStatus_t trsm_solve_identity(cublasHandle_t h, int n, const double* alpha,
                                   const double* a, int lda, double* b,
                                   int ldb) {
    return cublasDtrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, alpha, a, lda,
                       b, ldb);
}
cublasStatus_t trsm_solve_identity(cublasHandle_t h, int n, const float* alpha,
                                   const float* a, int lda, float* b,
                                   int ldb) {
    return cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, alpha, a, lda,
                       b, ldb);
}

}  // namespace

template <typename value_t>
void generate_sketch_qr_precond(dim<2> size, value_t* values, size_t ld) {
    int k = static_cast<int>(size[0]);
    int n = static_cast<int>(size[1]);
    if (k < n)
        throw std::invalid_argument(
            "generate_sketch_qr_precond (CUDA): sketched_mtx must have at "
            "least as many rows as columns (k >= n).");
    int lda = static_cast<int>(ld);

    cusolverDnHandle_t solver_handle;
    if (cusolverDnCreate(&solver_handle) != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(
            "generate_sketch_qr_precond (CUDA): cusolverDnCreate failed.");
    cublasHandle_t blas_handle;
    if (cublasCreate(&blas_handle) != CUBLAS_STATUS_SUCCESS) {
        cusolverDnDestroy(solver_handle);
        throw std::runtime_error(
            "generate_sketch_qr_precond (CUDA): cublasCreate failed.");
    }

    value_t* tau = nullptr;
    value_t* work = nullptr;
    int* devInfo = nullptr;
    value_t* identity = nullptr;

    auto cleanup = [&]() {
        if (tau) cudaFree(tau);
        if (work) cudaFree(work);
        if (devInfo) cudaFree(devInfo);
        if (identity) cudaFree(identity);
        cublasDestroy(blas_handle);
        cusolverDnDestroy(solver_handle);
    };

    cudaMalloc(&tau, static_cast<size_t>(n) * sizeof(value_t));
    cudaMalloc(&devInfo, sizeof(int));

    int lwork = 0;
    geqrf_buffer_size(solver_handle, k, n, values, lda, &lwork);
    cudaMalloc(&work, static_cast<size_t>(lwork) * sizeof(value_t));

    cusolverStatus_t qr_status =
        geqrf(solver_handle, k, n, values, lda, tau, work, lwork, devInfo);

    int info_host = 0;
    cudaMemcpy(&info_host, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (qr_status != CUSOLVER_STATUS_SUCCESS || info_host != 0) {
        cleanup();
        throw std::runtime_error(
            "generate_sketch_qr_precond (CUDA): geqrf failed (status=" +
            std::to_string(static_cast<int>(qr_status)) +
            ", info=" + std::to_string(info_host) + ").");
    }

    // geqrf packs Householder reflectors below the diagonal; zero them so
    // only the leading n x n upper-triangular R factor remains.
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n + 15) / 16),
             static_cast<unsigned>((n + 15) / 16));
    zero_subdiagonal_kernel<<<grid, block>>>(n, values, lda);

    // Solve R * X = I in-place (cublas trsm overwrites the RHS with the
    // solution), giving X = R^{-1} in a separate n x n scratch buffer.
    int ld_id = n;
    cudaMalloc(&identity, static_cast<size_t>(n) * n * sizeof(value_t));
    set_identity_kernel<<<grid, block>>>(n, identity, ld_id);

    value_t alpha = value_t(1);
    cublasStatus_t trsm_status =
        trsm_solve_identity(blas_handle, n, &alpha, values, lda, identity,
                            ld_id);
    if (trsm_status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        throw std::runtime_error(
            "generate_sketch_qr_precond (CUDA): trsm failed (status=" +
            std::to_string(static_cast<int>(trsm_status)) +
            ") -- R is likely singular.");
    }

    // Copy R^{-1} back into the leading n x n block of `values`.
    cudaMemcpy2D(values, static_cast<size_t>(lda) * sizeof(value_t), identity,
                static_cast<size_t>(ld_id) * sizeof(value_t),
                static_cast<size_t>(n) * sizeof(value_t),
                static_cast<size_t>(n), cudaMemcpyDeviceToDevice);

    cleanup();
}

template void generate_sketch_qr_precond<double>(dim<2>, double*, size_t);
template void generate_sketch_qr_precond<float>(dim<2>, float*, size_t);

}  // namespace cuda
}  // namespace tomos

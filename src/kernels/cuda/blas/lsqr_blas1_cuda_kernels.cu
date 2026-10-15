// BLAS-1 primitives used within the LSQR solver's CUDA runtime path: called
// directly from src/core/solver/lsqr.cpp's lsqr_nrm2/lsqr_scal/lsqr_axpy/
// lsqr_copy<Device::CUDA> (cuda::lsqr_blas1_nrm2/_scal/_axpy/_copy), inside
// the per-iteration solve loop. See lsqr_blas1_cuda_kernels.cuh's own
// comment for the double/float-via-cuBLAS vs __half/__nv_bfloat16-via-
// generic-kernel split.

#include "lsqr_blas1_cuda_kernels.cuh"

#include <cmath>
#include <cuda_runtime.h>

#include "type_macros.hpp"

namespace tomos {
namespace cuda {

double lsqr_blas1_nrm2(int n, double* x, cublasHandle_t handle) {
    double result = 0.0;
    cublasDnrm2(handle, n, x, 1, &result);
    return result;
}

double lsqr_blas1_nrm2(int n, float* x, cublasHandle_t handle) {
    float result = 0.0f;
    cublasSnrm2(handle, n, x, 1, &result);
    return static_cast<double>(result);
}

// Sum-of-squares reduction for __half via atomicAdd (double accumulator --
// CUDA has real hardware/PTX-emulated __half arithmetic, but this generic
// path stays double-accumulated for consistency with the other generic
// kernels in this codebase, e.g. cuda::mv_strided/mv_csr's fallback path).
__global__ void nrm2_sumsq_half_kernel(int n, __half* x, double* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = static_cast<double>(x[i]);
    atomicAdd(result, v * v);
}

double lsqr_blas1_nrm2(int n, __half* x, cublasHandle_t handle) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nrm2_sumsq_half_kernel<<<blocks, threads>>>(n, x, d_result);
    double h_result = 0.0;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return std::sqrt(h_result);
}

// Sum-of-squares reduction for __nv_bfloat16 -- same double-accumulated
// atomicAdd pattern as the __half version above.
__global__ void nrm2_sumsq_bf16_kernel(int n, __nv_bfloat16* x, double* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = static_cast<double>(x[i]);
    atomicAdd(result, v * v);
}

double lsqr_blas1_nrm2(int n, __nv_bfloat16* x, cublasHandle_t handle) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nrm2_sumsq_bf16_kernel<<<blocks, threads>>>(n, x, d_result);
    double h_result = 0.0;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return std::sqrt(h_result);
}

void lsqr_blas1_scal(int n, double alpha, double* x, cublasHandle_t handle) {
    cublasDscal(handle, n, &alpha, x, 1);
}

void lsqr_blas1_scal(int n, double alpha, float* x, cublasHandle_t handle) {
    float f_alpha = static_cast<float>(alpha);
    cublasSscal(handle, n, &f_alpha, x, 1);
}

__global__ void scal_half_kernel(int n, double alpha, __half* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = static_cast<__half>(alpha * static_cast<double>(x[i]));
}

void lsqr_blas1_scal(int n, double alpha, __half* x, cublasHandle_t handle) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scal_half_kernel<<<blocks, threads>>>(n, alpha, x);
}

__global__ void scal_bf16_kernel(int n, double alpha, __nv_bfloat16* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = static_cast<__nv_bfloat16>(alpha * static_cast<double>(x[i]));
}

void lsqr_blas1_scal(int n, double alpha, __nv_bfloat16* x, cublasHandle_t handle) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scal_bf16_kernel<<<blocks, threads>>>(n, alpha, x);
}

// y <- alpha * x + y — generic mixed-precision elementwise kernel, double
// accumulator.
template <typename Src, typename Dst>
__global__ void axpy_generic_kernel(int n, double alpha, Src* x, Dst* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = static_cast<Dst>(alpha * static_cast<double>(x[i]) +
                            static_cast<double>(y[i]));
}

template <typename Src, typename Dst>
void lsqr_blas1_axpy(int n, double alpha, Src* x, Dst* y,
                     cublasHandle_t handle) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    axpy_generic_kernel<<<blocks, threads>>>(n, alpha, x, y);
}

// cuBLAS fast paths for same-precision double/float.
template <>
void lsqr_blas1_axpy<double, double>(int n, double alpha, double* x,
                                     double* y, cublasHandle_t handle) {
    cublasDaxpy(handle, n, &alpha, x, 1, y, 1);
}

template <>
void lsqr_blas1_axpy<float, float>(int n, double alpha, float* x, float* y,
                                   cublasHandle_t handle) {
    float f_alpha = static_cast<float>(alpha);
    cublasSaxpy(handle, n, &f_alpha, x, 1, y, 1);
}

#define INST_AXPY_GENERIC_KERNEL(Src, Dst) \
    template __global__ void axpy_generic_kernel<Src, Dst>(int, double, Src*, Dst*);
FOR_ALL_FP_PAIRS(INST_AXPY_GENERIC_KERNEL)
#undef INST_AXPY_GENERIC_KERNEL

// Explicit instantiation of the remaining 7 (Src, Dst) combinations not
// covered by the cuBLAS full specializations above.
template void lsqr_blas1_axpy<double, float>(int, double, double*, float*,
                                             cublasHandle_t);
template void lsqr_blas1_axpy<double, __half>(int, double, double*, __half*,
                                              cublasHandle_t);
template void lsqr_blas1_axpy<float, double>(int, double, float*, double*,
                                             cublasHandle_t);
template void lsqr_blas1_axpy<float, __half>(int, double, float*, __half*,
                                             cublasHandle_t);
template void lsqr_blas1_axpy<__half, double>(int, double, __half*, double*,
                                              cublasHandle_t);
template void lsqr_blas1_axpy<__half, float>(int, double, __half*, float*,
                                             cublasHandle_t);
template void lsqr_blas1_axpy<__half, __half>(int, double, __half*, __half*,
                                              cublasHandle_t);
template void lsqr_blas1_axpy<double, __nv_bfloat16>(int, double, double*,
                                                      __nv_bfloat16*, cublasHandle_t);
template void lsqr_blas1_axpy<__nv_bfloat16, double>(int, double, __nv_bfloat16*,
                                                      double*, cublasHandle_t);
template void lsqr_blas1_axpy<float, __nv_bfloat16>(int, double, float*,
                                                     __nv_bfloat16*, cublasHandle_t);
template void lsqr_blas1_axpy<__nv_bfloat16, float>(int, double, __nv_bfloat16*,
                                                     float*, cublasHandle_t);
template void lsqr_blas1_axpy<__half, __nv_bfloat16>(int, double, __half*,
                                                      __nv_bfloat16*, cublasHandle_t);
template void lsqr_blas1_axpy<__nv_bfloat16, __half>(int, double, __nv_bfloat16*,
                                                      __half*, cublasHandle_t);
template void lsqr_blas1_axpy<__nv_bfloat16, __nv_bfloat16>(int, double, __nv_bfloat16*,
                                                             __nv_bfloat16*, cublasHandle_t);

// dst <- src — generic elementwise cast kernel.
template <typename Src, typename Dst>
__global__ void copy_generic_kernel(int n, Src* src, Dst* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = static_cast<Dst>(src[i]);
}

template <typename Src, typename Dst>
void lsqr_blas1_copy(int n, Src* src, Dst* dst) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copy_generic_kernel<<<blocks, threads>>>(n, src, dst);
}

// Same-type fast paths: a plain device-to-device memcpy, no kernel needed.
template <>
void lsqr_blas1_copy<double, double>(int n, double* src, double* dst) {
    cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(double),
              cudaMemcpyDeviceToDevice);
}

template <>
void lsqr_blas1_copy<float, float>(int n, float* src, float* dst) {
    cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(float),
              cudaMemcpyDeviceToDevice);
}

template <>
void lsqr_blas1_copy<__half, __half>(int n, __half* src, __half* dst) {
    cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(__half),
              cudaMemcpyDeviceToDevice);
}

template <>
void lsqr_blas1_copy<__nv_bfloat16, __nv_bfloat16>(int n, __nv_bfloat16* src,
                                                    __nv_bfloat16* dst) {
    cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(__nv_bfloat16),
              cudaMemcpyDeviceToDevice);
}

#define INST_COPY_GENERIC_KERNEL(Src, Dst) \
    template __global__ void copy_generic_kernel<Src, Dst>(int, Src*, Dst*);
FOR_ALL_FP_PAIRS(INST_COPY_GENERIC_KERNEL)
#undef INST_COPY_GENERIC_KERNEL

// Explicit instantiation of the 6 mismatched (Src, Dst) combinations not
// covered by the memcpy fast paths above.
template void lsqr_blas1_copy<double, float>(int, double*, float*);
template void lsqr_blas1_copy<double, __half>(int, double*, __half*);
template void lsqr_blas1_copy<float, double>(int, float*, double*);
template void lsqr_blas1_copy<float, __half>(int, float*, __half*);
template void lsqr_blas1_copy<__half, double>(int, __half*, double*);
template void lsqr_blas1_copy<__half, float>(int, __half*, float*);
template void lsqr_blas1_copy<double, __nv_bfloat16>(int, double*, __nv_bfloat16*);
template void lsqr_blas1_copy<__nv_bfloat16, double>(int, __nv_bfloat16*, double*);
template void lsqr_blas1_copy<float, __nv_bfloat16>(int, float*, __nv_bfloat16*);
template void lsqr_blas1_copy<__nv_bfloat16, float>(int, __nv_bfloat16*, float*);
template void lsqr_blas1_copy<__half, __nv_bfloat16>(int, __half*, __nv_bfloat16*);
template void lsqr_blas1_copy<__nv_bfloat16, __half>(int, __nv_bfloat16*, __half*);

}  // namespace cuda
}  // namespace tomos

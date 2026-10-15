#ifndef TOMOS_LSQR_BLAS1_CUDA_KERNELS_CUH
#define TOMOS_LSQR_BLAS1_CUDA_KERNELS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tomos {
namespace cuda {

// BLAS-1 primitives backing LsqrWorkspace<Device::CUDA, ...> (see
// src/core/solver/lsqr.cpp). double/float use cuBLAS; any combination
// involving __half/__nv_bfloat16 (or a mismatched axpy/copy Src/Dst pair)
// falls back to a generic elementwise kernel accumulating in double,
// matching the existing generic-kernel convention already used by
// cuda::mv_strided/mv_csr.

double lsqr_blas1_nrm2(int n, double* x, cublasHandle_t handle);
double lsqr_blas1_nrm2(int n, float* x, cublasHandle_t handle);
double lsqr_blas1_nrm2(int n, __half* x, cublasHandle_t handle);
double lsqr_blas1_nrm2(int n, __nv_bfloat16* x, cublasHandle_t handle);

void lsqr_blas1_scal(int n, double alpha, double* x, cublasHandle_t handle);
void lsqr_blas1_scal(int n, double alpha, float* x, cublasHandle_t handle);
void lsqr_blas1_scal(int n, double alpha, __half* x, cublasHandle_t handle);
void lsqr_blas1_scal(int n, double alpha, __nv_bfloat16* x, cublasHandle_t handle);

// y <- alpha * x + y
template <typename Src, typename Dst>
void lsqr_blas1_axpy(int n, double alpha, Src* x, Dst* y,
                     cublasHandle_t handle);

// dst <- src
template <typename Src, typename Dst>
void lsqr_blas1_copy(int n, Src* src, Dst* dst);

}  // namespace cuda
}  // namespace tomos

#endif  // TOMOS_LSQR_BLAS1_CUDA_KERNELS_CUH

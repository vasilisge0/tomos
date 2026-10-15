#ifndef TOMOS_CONTEXT_CUH
#define TOMOS_CONTEXT_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

namespace tomos {

// struct CudaDescriptor {
//     cudaStream_t stream_         = nullptr;
//     cublasHandle_t cublas_       = nullptr;
//     cusparseHandle_t cusparse_   = nullptr;
//     cusolverDnHandle_t cusolver_ = nullptr;
//     curandGenerator_t curand_generator_ = nullptr;
// };

// struct CudaContext {

//     // Device::CUDA library descriptors
//     cudaStream_t stream_         = nullptr;
//     cublasHandle_t cublas_       = nullptr;
//     cusparseHandle_t cusparse_   = nullptr;
//     cusolverDnHandle_t cusolver_ = nullptr;
//     curandGenerator_t curand_generator_ = nullptr;

//     // @NOTE: TF32 is not used anywhere so far
//     bool use_tf32_ = false;

//     // internal memory that can be passed around via CudaContext
//     uint32_t *mem_ = nullptr;

//     CudaContext() {
//         // @TEMP: move this to an implementation file
//         //cudaError_t cuda_status = cudaStreamCreate(&stream_);
//         //if (cuda_status != cudaSuccess) {
//         //    perror("Call to cudaStreamCreate failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}

//         //cublasStatus_t cublas_status = cublasCreate(&cublas_);
//         //if (cublas_status != CUBLAS_STATUS_SUCCESS) {
//         //    perror("Call to cublasCreate failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}

//         //cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_);
//         //if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
//         //    perror("Call to cusparseCreate failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}

//         //curandStatus_t curand_status =
//         curandCreateGenerator(&curand_generator_,
//         //    CURAND_RNG_PSEUDO_DEFAULT);
//         //if (curand_status != CURAND_STATUS_SUCCESS) {
//         //    perror("Call to curandCreateGenerator failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}

//         //curand_status =
//         curandSetPseudoRandomGeneratorSeed(curand_generator_, time(NULL));
//         //if (curand_status != CURAND_STATUS_SUCCESS) {
//         //    perror("Call to curandSetPseudoRandomGeneratorSeed failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}

//         //cusolverStatus_t cusolver_status = cusolverDnCreate(&(cusolver_));
//         //if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
//         //    perror("Initialization of cuSolver failed.\n");
//         //    exit(EXIT_FAILURE);
//         //}
//     }
// };  // CudaContext

}  // namespace tomos

#endif  // TOMOS_CONTEXT_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "blas.hpp"
#include "context.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
template <>
Status norm2<Device::CUDA, double>(size_t num_elems, double *vec, size_t inc,
                                   double *norm) {
    cublasStatus_t status;
    // CudaDescriptor descr = context->get_cuda_descriptor();
    // status = cublasDnrm2(descr.cublas_,
    //                      static_cast<int>(num_elems), vec,
    //                      static_cast<int>(inc), norm);
    return SUCCESS;
}  // end of norm2<double>

template <>
Status norm2<Device::CUDA, float>(size_t num_elems, float *vec, size_t inc,
                                  float *norm) {
    cublasStatus_t status;
    // float norm;
    // status = cublasSnrm2(context->get_cuda_context().cublas_,
    //                     static_cast<int>(num_elems), vec,
    //                     static_cast<int>(inc), norm);
    return SUCCESS;
}  // end of norm2<double>

template <>
Status norm2<Device::CUDA, __half>(size_t num_elems, __half *vec, size_t inc,
                                   __half *norm) {
    cublasStatus_t status;
    // status = cublasDnrm2(context->get_cuda_context().cublas_,
    // static_cast<int>(num_elems), vec, static_cast<int>(inc), norm);
    // cudaDeviceSynchronize();
    // return (status == CUBLAS_STATUS_SUCCESS) ? SUCCESS : FAILURE;
    // return norm;
    return FAILURE;
}  // end of norm2<double>

// template <Device device, typename value_t>
// void scale<Device::CUDA, double>(Context* context, size_t num_rows,
//                   value_t alpha, value_t *v_vector, size_t inc) {
// if constexpr ((device == Device::CUDA) && (std::is_same<value_t,
// double>::value)) {
//     cublasDscal(context->get_cuda_context().cublas_, num_rows, &alpha,
//                 v_vector, inc);
//     cudaDeviceSynchronize();

//} else if constexpr ((device == Device::CUDA) &&
//                     (std::is_same<value_t, float>::value)) {
//    cublasSscal(context->get_cuda_context().cublas_, num_rows, &alpha,
//                v_vector, inc);
//    cudaDeviceSynchronize();
//}
//}   // scale

// template <Device device, typename value_t, typename index_t>
// inline void scale(Context &context, index_t num_rows, value_t alpha,
//                   value_t *v_vector, index_t inc) {
//     // if constexpr ((device == Device::CUDA) && (std::is_same<value_t,
//     // double>::value)) {
//     //     cublasDscal(context->get_cuda_context().cublas_, num_rows, &alpha,
//     //                 v_vector, inc);
//     //     cudaDeviceSynchronize();

//     //} else if constexpr ((device == Device::CUDA) &&
//     //                     (std::is_same<value_t, float>::value)) {
//     //    cublasSscal(context->get_cuda_context().cublas_, num_rows, &alpha,
//     //                v_vector, inc);
//     //    cudaDeviceSynchronize();
//     //}
// }

// template void scale<Device::CUDA, double, int>(Context &context, int
// num_rows,
//                                                double alpha, double
//                                                *v_vector, int inc);
// template void scale<Device::CUDA, float, int>(Context &context, int num_rows,
//                                               float alpha, float *v_vector,
//                                               int inc);
// template void scale<Device::CUDA, __half, int>(Context &context, int
// num_rows,
//                                                __half alpha, __half
//                                                *v_vector, int inc);

}  // namespace tomos

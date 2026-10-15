
#include "cublas_v2.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "blas.hpp"
#include "context.hpp"
#include "cuda_kernel_constants.cuh"
#include "device.hpp"
#include "dimensions.hpp"
#include "status.hpp"

namespace tomos {
namespace cuda {

template <typename index_type, typename value_type>
__global__ void set_upper_triang_2d_kernel(index_type num_rows,
                                           index_type num_cols,
                                           value_type* values, index_type ld) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_lower_triangular =
        (row > col) && (row < num_rows) && (col < num_cols);
    if (in_lower_triangular) values[row + ld * col] = 0.0;
}

template __global__ void set_upper_triang_2d_kernel(int num_rows, int num_cols,
                                                    double* values, int ld);
template __global__ void set_upper_triang_2d_kernel(int num_rows, int num_cols,
                                                    float* values, int ld);
template __global__ void set_upper_triang_2d_kernel(int num_rows, int num_cols,
                                                    __half* values, int ld);
template __global__ void set_upper_triang_2d_kernel(size_t num_rows,
                                                    size_t num_cols,
                                                    double* values, size_t ld);
template __global__ void set_upper_triang_2d_kernel(size_t num_rows,
                                                    size_t num_cols,
                                                    float* values, size_t ld);
template __global__ void set_upper_triang_2d_kernel(size_t num_rows,
                                                    size_t num_cols,
                                                    __half* values, size_t ld);

template <typename index_type, typename value_type>
void set_upper_triang(dim<2> size, value_type* values, index_type ld) {
    ::dim3 grid_size(size[1] / MAX_NUM_THREADS_PER_1D_THREAD_BLOCK + 1,
                     size[0] / MAX_NUM_THREADS_PER_1D_THREAD_BLOCK + 1);
    ::dim3 block_size(MAX_NUM_THREADS_PER_1D_THREAD_BLOCK,
                      MAX_NUM_THREADS_PER_1D_THREAD_BLOCK);
    set_upper_triang_2d_kernel<<<grid_size, block_size>>>(
        (int)size[0], (int)size[1], values, (int)ld);
    cudaDeviceSynchronize();
}

template void set_upper_triang(dim<2> size, double* values, int ld);
template void set_upper_triang(dim<2> size, float* values, int ld);
template void set_upper_triang(dim<2> size, __half* values, int ld);
template void set_upper_triang(dim<2> size, __nv_bfloat16* values, int ld);
template void set_upper_triang(dim<2> size, double* values, size_t ld);
template void set_upper_triang(dim<2> size, float* values, size_t ld);
template void set_upper_triang(dim<2> size, __half* values, size_t ld);
template void set_upper_triang(dim<2> size, __nv_bfloat16* values, size_t ld);

template <typename index_type, typename value_type>
__global__ void set_eye_2d_kernel(index_type num_rows, index_type num_cols,
                                  value_type* values, index_type ld) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < num_rows) && (col < num_cols)) {
        if (row == col) {
            values[row + ld * col] = 1.0;
        } else {
            values[row + ld * col] = 0.0;
        }
    }
}

template <typename index_type, typename value_type>
inline void set_eye(dim<2> size, value_type* values, index_type ld) {
    ::dim3 grid_size(size[1] / MAX_NUM_THREADS_PER_1D_THREAD_BLOCK + 1,
                     size[0] / MAX_NUM_THREADS_PER_1D_THREAD_BLOCK + 1);
    ::dim3 block_size(MAX_NUM_THREADS_PER_1D_THREAD_BLOCK,
                      MAX_NUM_THREADS_PER_1D_THREAD_BLOCK);
    set_eye_2d_kernel<<<grid_size, block_size>>>((int)size[0], (int)size[1],
                                                 values, (int)ld);
    cudaDeviceSynchronize();
}

template void set_eye(dim<2> size, double* values, int ld);
template void set_eye(dim<2> size, float* values, int ld);
template void set_eye(dim<2> size, __half* values, int ld);
template void set_eye(dim<2> size, __nv_bfloat16* values, int ld);

}  // end of namespace cuda
}  // namespace tomos

#include <cuda_runtime.h>

#include "cuda_kernel_constants.cuh"
#include "dimensions.hpp"
#include "precision.hpp"
#include "status.hpp"

namespace tomos {
namespace cuda {

template <typename value_t>
__global__ void set_values_kernel(value_t val, size_t nrows, size_t ncols,
                                  value_t* v_out, int ld_out) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < nrows) && (col < ncols)) v_out[row + ld_out * col] = val;
}  // set_values_kernel

template __global__ void set_values_kernel(double val, size_t nrows,
                                           size_t ncols, double* v_out,
                                           int ld_out);
template __global__ void set_values_kernel(float val, size_t nrows,
                                           size_t ncols, float* v_out,
                                           int ld_out);
template __global__ void set_values_kernel(__half val, size_t nrows,
                                           size_t ncols, __half* v_out,
                                           int ld_out);
template __global__ void set_values_kernel(uint32_t val, size_t nrows,
                                           size_t ncols, uint32_t* v_out,
                                           int ld_out);

template <typename value_t>
__host__ void set_values_impl(value_t val, dim<2> size, value_t* v_out,
                              size_t ld_out) {
    int num_threads = MAX_NUM_THREADS_PER_2D_THREAD_BLOCK;
    int num_rows = size[0];
    int num_cols = size[1];
    dim3 threads_per_block(num_threads, num_threads);
    dim3 num_blocks((num_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (num_rows + threads_per_block.y - 1) / threads_per_block.y);
    set_values_kernel<<<num_blocks, threads_per_block>>>(
        val, num_rows, num_cols, v_out, ld_out);
    cudaDeviceSynchronize();
}  // set_values_impl

template void set_values_impl(double val, dim<2> size, double* values,
                              size_t ld);
template void set_values_impl(float val, dim<2> size, float* values, size_t ld);
template void set_values_impl(__half val, dim<2> size, __half* values,
                              size_t ld);
template void set_values_impl(__nv_bfloat16 val, dim<2> size, __nv_bfloat16* values,
                              size_t ld);
template void set_values_impl(uint32_t val, dim<2> size, uint32_t* values,
                              size_t ld);

}  // namespace cuda
}  // namespace tomos

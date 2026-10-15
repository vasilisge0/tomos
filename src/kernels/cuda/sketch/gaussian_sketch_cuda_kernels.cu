#include "gaussian_sketch_cuda_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <time.h>

namespace tomos {
namespace cuda {

namespace {

__device__ __forceinline__ uint64_t gaussian_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

template <typename value_t>
__global__ void gaussian_values_kernel(uint64_t seed, size_t rows, size_t cols,
                                       value_t* values, size_t ld) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;
    uint64_t h = gaussian_splitmix64((j * rows + i) ^ seed);
    thrust::minstd_rand rng(static_cast<unsigned int>(h));
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    values[i + j * ld] = static_cast<value_t>(dist(rng));
}

}  // namespace

template <typename value_t>
void generate_gaussian_values(dim<2> size, value_t* values, size_t ld) {
    uint64_t seed = static_cast<uint64_t>(time(nullptr));
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((size[0] + 15) / 16),
             static_cast<unsigned>((size[1] + 15) / 16));
    gaussian_values_kernel<<<grid, block>>>(seed, size[0], size[1], values,
                                            ld);
    cudaDeviceSynchronize();
}

template void generate_gaussian_values<double>(dim<2>, double*, size_t);
template void generate_gaussian_values<float>(dim<2>, float*, size_t);
template void generate_gaussian_values<__half>(dim<2>, __half*, size_t);
template void generate_gaussian_values<__nv_bfloat16>(dim<2>, __nv_bfloat16*, size_t);

}  // namespace cuda
}  // namespace tomos

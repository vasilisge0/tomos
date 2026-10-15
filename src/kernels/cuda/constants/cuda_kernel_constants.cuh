#ifndef CUDA_KERNEL_CONSTANTS_CUH
#define CUDA_KERNEL_CONSTANTS_CUH

namespace tomos {
namespace cuda {

#define MAX_NUM_THREADS_PER_1D_THREAD_BLOCK 1024
#define MAX_NUM_THREADS_PER_2D_THREAD_BLOCK 32

}   // Device::CUDA
}   // tomos

#endif  // CUDA_KERNEL_CONSTANTS_CUH
#include <cuda_runtime.h>

#include "cuda_kernel_constants.cuh"
#include "dimensions.hpp"
#include "matrix_formats.hpp"
#include "precision.hpp"
#include "sparse_storage.hpp"
#include "status.hpp"
#include "type_macros.hpp"

namespace tomos {
namespace cuda {

template <typename value_t, typename value_out_t>
__global__ void copy_kernel(size_t nrows, size_t ncols, value_t* mtx_in,
                            int ld_in, value_out_t* mtx_out, int ld_out) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < nrows) && (col < ncols)) {
        if constexpr (std::is_same_v<value_t, value_out_t>) {
            mtx_out[row + ld_out * col] = mtx_in[row + ld_in * col];
        } else {
            mtx_out[row + ld_out * col] =
                static_cast<value_out_t>(mtx_in[row + ld_in * col]);
        }
    }
}

#define INST_COPY_KERNEL(T1, T2)                                           \
    template __global__ void copy_kernel<T1, T2>(size_t, size_t, T1*, int, \
                                                 T2*, int);

FOR_ALL_FP_PAIRS(INST_COPY_KERNEL)
FOR_ALL_INDEX_PAIRS(INST_COPY_KERNEL)

template <typename value_in_t, typename value_out_t>
__host__ void copy_launcher(dim<2> size, value_in_t* mtx_in, int ld_in,
                            value_out_t* mtx_out, int ld_out) {
    dim3 block_size(16, 16);
    dim3 grid_size((size[1] + block_size.x - 1) / block_size.x,
                   (size[0] + block_size.y - 1) / block_size.y);
    copy_kernel<<<grid_size, block_size>>>(size[0], size[1], mtx_in, ld_in,
                                           mtx_out, ld_out);
}

#define INST_COPY_LAUNCHER(value_in_t, value_out_t)                \
    template __host__ void copy_launcher<value_in_t, value_out_t>( \
        dim<2>, value_in_t*, int, value_out_t*, int);

FOR_ALL_FP_PAIRS(INST_COPY_LAUNCHER)
FOR_ALL_INDEX_PAIRS(INST_COPY_LAUNCHER)

template <typename value_in_t, typename value_out_t>
__host__ void dense_copy_impl(dim<2> size, DenseStorage& source,
                              DenseStorage& target) {
    std::visit(
        [&](auto& x, auto& y) {
            using Tx = std::decay_t<decltype(x)>;
            using Ty = std::decay_t<decltype(y)>;
            if constexpr ((std::is_same_v<Tx, DenseStorage>) &&
                          (std::is_same_v<Ty, DenseStorage>)) {
                copy_launcher(size, std::get<value_out_t*>(x.values_),
                              x.lead_dim_, std::get<value_in_t*>(y.values_),
                              y.lead_dim_);
            } else if constexpr ((std::is_same_v<Tx, CsrStorage>) &&
                                 (std::is_same_v<Ty, CsrStorage>)) {
                copy_launcher({x.nnz_, 1}, std::get<value_out_t*>(x.values_),
                              x.nnz_, std::get<value_in_t*>(y.values_), x.nnz_);
                copy_launcher({size[0] + 1, 1}, std::get<int*>(x.row_ptrs_),
                              size[0] + 1, std::get<int*>(y.row_ptrs_),
                              size[0] + 1);
                copy_launcher({x.nnz_, 1}, std::get<int*>(x.col_idxs_), x.nnz_,
                              std::get<int*>(y.col_idxs_), x.nnz_);
            } else {
                throw std::runtime_error(
                    "Copying values from matrixs of different types.");
            }
        },
        source, target);
}

#define INST_DENSE_COPY_IMPL(T1, T2)                                      \
    template __host__ void dense_copy_impl<T1, T2>(dim<2>, DenseStorage&, \
                                                   DenseStorage&);

FOR_ALL_FP_PAIRS(INST_DENSE_COPY_IMPL)

template <typename value_in_t, typename value_out_t, typename index_in_t,
          typename index_out_t>
__host__ void csr_copy_impl(dim<2> size, SparseStorage& source,
                            SparseStorage& target) {
    auto& x = std::get<CsrStorage>(source);
    auto& y = std::get<CsrStorage>(target);
    copy_launcher({x.nnz_, 1}, std::get<value_out_t*>(x.values_), x.nnz_,
                  std::get<value_in_t*>(y.values_), x.nnz_);
    copy_launcher({size[0] + 1, 1}, std::get<int*>(x.row_ptrs_), size[0] + 1,
                  std::get<int*>(y.row_ptrs_), size[0] + 1);
    copy_launcher({x.nnz_, 1}, std::get<int*>(x.col_idxs_), x.nnz_,
                  std::get<int*>(y.col_idxs_), x.nnz_);
}

#define INST_CSR_COPY_IMPL(value_in_t, value_out_t, index_in_t, index_out_t) \
    template __host__ void                                                   \
    csr_copy_impl<value_in_t, value_out_t, index_in_t, index_out_t>(         \
        dim<2>, SparseStorage&, SparseStorage&);

FOR_ALL_FP_PAIRS_INDEX_PAIRS(INST_CSR_COPY_IMPL)

// template <typename value_t>
// __global__ void copy_kernel(size_t nrows, size_t ncols, value_t* mtx_in,
//                             int ld_in, value_t* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] = mtx_in[row + ld_in * col];
//     }
// }

// template __global__ void copy_kernel<double>(size_t nrows, size_t ncols,
//                                              double* mtx_in, int ld_in,
//                                              double* mtx_out, int
//                                              ld_out);
// template __global__ void copy_kernel<float>(size_t nrows, size_t ncols,
//                                             float* mtx_in, int ld_in,
//                                             float* mtx_out, int ld_out);
// template __global__ void copy_kernel<__half>(size_t nrows, size_t ncols,
//                                              __half* mtx_in, int ld_in,
//                                              __half* mtx_out, int
//                                              ld_out);

// __global__ void copy_kernel(size_t nrows, size_t ncols, double* mtx_in,
//                             int ld_in, float* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] =
//             __double2float_rn(mtx_in[row + ld_in * col]);
//     }
// }  // convert_kernel

// __global__ void copy_kernel(size_t nrows, size_t ncols, double* mtx_in,
//                             int ld_in, __half* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] = __double2half(mtx_in[row + ld_in *
//         col]);
//     }
// }  // convert_kernel

// __global__ void copy_kernel(size_t nrows, size_t ncols, float* mtx_in,
//                             int ld_in, __half* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] = __float2half(mtx_in[row + ld_in *
//         col]);
//     }
// }  // convert_kernel

// __global__ void copy_kernel(size_t nrows, size_t ncols, float* mtx_in,
//                             int ld_in, double* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] =
//             static_cast<double>(mtx_in[row + ld_in * col]);
//     }
// }  // convert_kernel

// __global__ void copy_kernel(size_t nrows, size_t ncols, __half* mtx_in,
//                             int ld_in, double* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] =
//             static_cast<double>(__half2float(mtx_in[row + ld_in * col]));
//     }
// }  // convert_kernel

// __global__ void copy_kernel(size_t nrows, size_t ncols, __half* mtx_in,
//                             int ld_in, float* mtx_out, int ld_out) {
//     auto col = blockIdx.x * blockDim.x + threadIdx.x;
//     auto row = blockIdx.y * blockDim.y + threadIdx.y;
//     if ((row < nrows) && (col < ncols)) {
//         mtx_out[row + ld_out * col] = __half2float(mtx_in[row + ld_in *
//         col]);
//     }
// }  // convert_kernel

// template <typename in_value_t, typename out_value_t>
// __host__ Status copy_launcher(dim<2> size, in_value_t* values_in, size_t
// ld_in,
//                               out_value_t* values_out, size_t ld_out) {
//     int num_threads = MAX_NUM_THREADS_PER_2D_THREAD_BLOCK;
//     int num_rows = size[0];
//     int num_cols = size[1];
//     dim3 threads_per_block(num_threads, num_threads);
//     dim3 num_blocks((num_cols + threads_per_block.x - 1) /
//     threads_per_block.x,
//                     (num_rows + threads_per_block.y - 1) /
//                     threads_per_block.y);
//     copy_kernel<<<num_blocks, threads_per_block>>>(
//         num_rows, num_cols, values_in, ld_in, values_out, ld_out);
//     cudaDeviceSynchronize();
//     return SUCCESS;
// }  // copy_launcher

// template __host__ Status copy_launcher(dim<2> size, double* values_in,
//                                        size_t ld_in, double* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, double* values_in,
//                                        size_t ld_in, float* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, double* values_in,
//                                        size_t ld_in, __half* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, float* values_in,
//                                        size_t ld_in, double* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, float* values_in,
//                                        size_t ld_in, float* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, float* values_in,
//                                        size_t ld_in, __half* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, __half* values_in,
//                                        size_t ld_in, double* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, __half* values_in,
//                                        size_t ld_in, float* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, __half* values_in,
//                                        size_t ld_in, __half* values_out,
//                                        size_t ld_out);
// template __host__ Status copy_launcher(dim<2> size, int* values_in,
//                                        size_t ld_in, int* values_out,
//                                        size_t ld_out);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_impl(dim<2> size, StorageVariant& source,
//                         StorageVariant& target) {
//     std::visit(
//         [&](auto& x, auto& y) {
//             using Tx = std::decay_t<decltype(x)>;
//             using Ty = std::decay_t<decltype(y)>;
//             if constexpr ((std::is_same_v<Tx, DenseStorage>) &&
//                           (std::is_same_v<Ty, DenseStorage>)) {
//                 copy_launcher(size, std::get<value_out_t*>(x.values_),
//                               x.lead_dim_,
//                               std::get<value_in_t*>(y.values_),
//                               y.lead_dim_);
//             } else if constexpr ((std::is_same_v<Tx, CsrStorage>) &&
//                                  (std::is_same_v<Ty, CsrStorage>)) {
//                 copy_launcher({x.nnz_, 1},
//                 std::get<value_out_t*>(x.values_),
//                               x.nnz_, std::get<value_in_t*>(y.values_),
//                               x.nnz_);
//                 copy_launcher({size[0] + 1, 1},
//                 std::get<int*>(x.row_ptrs_),
//                               size[0] + 1, std::get<int*>(y.row_ptrs_),
//                               size[0] + 1);
//                 copy_launcher({x.nnz_, 1}, std::get<int*>(x.col_idxs_),
//                 x.nnz_,
//                               std::get<int*>(y.col_idxs_), x.nnz_);
//             } else {
//                 throw std::runtime_error(
//                     "Copying values from matrixs of different types.");
//             }
//         },
//         source, target);
// }  // copy_impl

// template __host__ void copy_impl<double, double>(dim<2> size,
//                                                  StorageVariant& source,
//                                                  StorageVariant& target);
// template __host__ void copy_impl<double, float>(dim<2> size,
//                                                 StorageVariant& source,
//                                                 StorageVariant& target);
// template __host__ void copy_impl<double, __half>(dim<2> size,
//                                                  StorageVariant& source,
//                                                  StorageVariant& target);
// template __host__ void copy_impl<float, double>(dim<2> size,
//                                                 StorageVariant& source,
//                                                 StorageVariant& target);
// template __host__ void copy_impl<float, float>(dim<2> size,
//                                                StorageVariant& source,
//                                                StorageVariant& target);
// template __host__ void copy_impl<float, __half>(dim<2> size,
//                                                 StorageVariant& source,
//                                                 StorageVariant& target);
// template __host__ void copy_impl<__half, double>(dim<2> size,
//                                                  StorageVariant& source,
//                                                  StorageVariant& target);
// template __host__ void copy_impl<__half, float>(dim<2> size,
//                                                 StorageVariant& source,
//                                                 StorageVariant& target);
// template __host__ void copy_impl<__half, __half>(dim<2> size,
//                                                  StorageVariant& source,
//

template <typename value_t, typename index_t>
__host__ void strided_copy_to_device(dim<2> size, DenseStorage& source,
                                     DenseStorage& target) {
    StridedStorage& strided_source = std::get<StridedStorage>(source);
    StridedStorage& strided_target = std::get<StridedStorage>(target);
    cudaMemcpy(std::get<value_t*>(strided_target.values_),
               std::get<value_t*>(strided_source.values_),
               strided_source.num_elems_ * sizeof(value_t),
               cudaMemcpyHostToDevice);
}

template <typename value_t, typename index_t>
__host__ void strided_copy_from_device(dim<2> size, DenseStorage& source,
                                       DenseStorage& target) {
    StridedStorage& strided_source = std::get<StridedStorage>(source);
    StridedStorage& strided_target = std::get<StridedStorage>(target);
    cudaMemcpy(std::get<value_t*>(strided_target.values_),
               std::get<value_t*>(strided_source.values_),
               strided_source.num_elems_ * sizeof(value_t),
               cudaMemcpyDeviceToHost);
}

template <typename value_t, typename index_t>
__host__ void csr_copy_to_device(dim<2> size, SparseStorage& source,
                                 SparseStorage& target) {
    CsrStorage& csr_source = std::get<CsrStorage>(source);
    CsrStorage& csr_target = std::get<CsrStorage>(target);
    cudaMemcpy(std::get<value_t*>(csr_target.values_),
               std::get<value_t*>(csr_source.values_),
               csr_source.nnz_ * sizeof(value_t), cudaMemcpyHostToDevice);
    cudaMemcpy(std::get<index_t*>(csr_target.row_ptrs_),
               std::get<index_t*>(csr_source.row_ptrs_),
               (size[0] + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(std::get<index_t*>(csr_target.col_idxs_),
               std::get<index_t*>(csr_source.col_idxs_),
               csr_source.nnz_ * sizeof(index_t), cudaMemcpyHostToDevice);
}

template <typename value_t, typename index_t>
__host__ void csr_copy_from_device(dim<2> size, SparseStorage& source,
                                   SparseStorage& target) {
    CsrStorage& csr_source = std::get<CsrStorage>(source);
    CsrStorage& csr_target = std::get<CsrStorage>(target);
    cudaMemcpy(std::get<value_t*>(csr_target.values_),
               std::get<value_t*>(csr_source.values_),
               csr_source.nnz_ * sizeof(value_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(std::get<index_t*>(csr_target.row_ptrs_),
               std::get<index_t*>(csr_source.row_ptrs_),
               (size[0] + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(std::get<index_t*>(csr_target.col_idxs_),
               std::get<index_t*>(csr_source.col_idxs_),
               csr_source.nnz_ * sizeof(index_t), cudaMemcpyDeviceToHost);
}

// template <typename value_t>
// __host__ void copy_to_device_impl(dim<2> size, SparseStorage& source,
//                                   SparseStorage& target) {
//     std::visit(
//         [&](auto& x, auto& y) {
//             using Tx = std::decay_t<decltype(x)>;
//             using Ty = std::decay_t<decltype(y)>;
//             if constexpr ((std::is_same_v<Tx, CsrStorage>) &&
//                           (std::is_same_v<Ty, CsrStorage>)) {
//                 size_t nnz = x.nnz_;
//                 cudaMemcpy(std::get<value_t*>(x.values_),
//                            std::get<value_t*>(y.values_), nnz * sizeof(value_t),
//                            cudaMemcpyHostToDevice);

//             } else {
//                 throw std::runtime_error(
//                     "Copying values from matrixs of different types.");
//             }
//         },
//         source, target);
// }

// template __host__ void copy_to_device_impl<double, double>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);
// template __host__ void copy_to_device_impl<float, float>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);
// template __host__ void copy_to_device_impl<__half, __half>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);

// template <typename value_in_t, typename value_out_t>
// __host__ void copy_from_device_impl(dim<2> size, StorageVariant& source,
//                                     StorageVariant& target) {
//     std::visit(
//         [&](auto& x, auto& y) {
//             using Tx = std::decay_t<decltype(x)>;
//             using Ty = std::decay_t<decltype(y)>;
//             if constexpr ((std::is_same_v<Tx, DenseStorage>) &&
//                           (std::is_same_v<Ty, DenseStorage>)) {
//                 size_t num_elems = size[0] * size[1];
//                 cudaMemcpy(std::get<value_out_t*>(x.values_),
//                            std::get<value_in_t*>(y.values_),
//                            num_elems * sizeof(value_in_t),
//                            cudaMemcpyDeviceToHost);
//             } else if constexpr ((std::is_same_v<Tx, CsrStorage>) &&
//                                  (std::is_same_v<Ty, CsrStorage>)) {
//                 size_t nnz = x.nnz_;
//                 cudaMemcpy(std::get<value_out_t*>(x.values_),
//                            std::get<value_in_t*>(y.values_),
//                            nnz * sizeof(value_in_t),
//                            cudaMemcpyDeviceToHost);
//             } else {
//                 throw std::runtime_error(
//                     "Copying values from matrixs of different types.");
//             }
//         },
//         source, target);
// }

// template __host__ void copy_from_device_impl<double, double>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);
// template __host__ void copy_from_device_impl<float, float>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);
// template __host__ void copy_from_device_impl<__half, __half>(
//     dim<2> size, StorageVariant& source, StorageVariant& target);

#define INST_CSR_COPY_TO_DEVICE(value_t, index_t)                \
    template __host__ void csr_copy_to_device<value_t, index_t>( \
        dim<2>, SparseStorage&, SparseStorage&);

#define INST_CSR_COPY_FROM_DEVICE(value_t, index_t)                \
    template __host__ void csr_copy_from_device<value_t, index_t>( \
        dim<2>, SparseStorage&, SparseStorage&);

#define INST_BOTH(value_t, index_t)           \
    INST_CSR_COPY_TO_DEVICE(value_t, index_t) \
    INST_CSR_COPY_FROM_DEVICE(value_t, index_t)

INST_BOTH(double, int)
INST_BOTH(double, long)
INST_BOTH(float, int)
INST_BOTH(float, long)
INST_BOTH(__half, int)
INST_BOTH(__half, long)
INST_BOTH(__nv_bfloat16, int)
INST_BOTH(__nv_bfloat16, long)

#undef INST_CSR_COPY_TO_DEVICE
#undef INST_CSR_COPY_FROM_DEVICE
#undef INST_BOTH

}  // namespace cuda
}  // namespace tomos

#include "countsketch.hpp"
#include "device.hpp"
#include "dimensions.hpp"
#include "matrix.hpp"

#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <time.h>

namespace tomos {
namespace cuda {

#define CUDA_MAX_NUM_THREADS_PER_BLOCK_1D 256
#define CUDA_MAX_NUM_THREADS_PER_BLOCK_2D 16

unsigned long hash(unsigned int in) {
    unsigned long hash = 5381;
    int c;

    while (c = in++) hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

__device__ unsigned int hash64shift(unsigned int key) {
    key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);  // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);  // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

template <typename index_type>
__global__ void set_col_idxs_kernel(double main_seed, int k, int rows, int cols,
                                    index_type* row_ptrs,
                                    index_type* col_idxs) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = 1;

    if (row < rows) {
        int seed = hash64shift(k * row + col) * ((int)main_seed);
        thrust::minstd_rand rng(seed);
        thrust::uniform_int_distribution<int> dist(0, cols - 1);
        col_idxs[row] = dist(rng);
    }
}

template <typename index_type>
__global__ void set_row_ptrs_kernel(unsigned int seed, int k, int rows,
                                    index_type* row_ptrs) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        row_ptrs[row] = row * k;
    }
}

template <typename index_type>
__global__ void shuffle_countsketch_init_col_idxs(double main_seed,
                                                  index_type k, index_type rows,
                                                  index_type* values) {
    int col_idx;
    int val_int;
    auto row = blockDim.y * blockIdx.y + threadIdx.y;
    auto col = blockDim.x * blockIdx.x + threadIdx.x;
    if ((row < rows) && (col < k)) {
        values[k * row + col] = col;
    }
}

template <typename index_type>
__global__ void shuffle_countsketch_set_col_idxs_kernel(
    unsigned int main_seed, index_type k, index_type rows, index_type cols,
    index_type* buffer, index_type* values) {
    int val_int;
    auto row = blockDim.y * blockIdx.y + threadIdx.y;
    auto col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows) {
        int seed = hash64shift(k * row + col) * ((int)main_seed);
        thrust::default_random_engine g(seed);
        thrust::shuffle(buffer + row * cols, buffer + row * cols + cols, g);
        index_type* v = values + row * k;
        for (int i = 0; i < k; i++) {
            v[i] = buffer[row * cols + i];
        }
    }
}

template <typename index_type>
void shuffle_countsketch_set_col_idxs(double main_seed, index_type nnz_per_col,
                                      dim<2> size, index_type* values) {
    // shuffle col_idxs
    {
        index_type* buffer;
        cudaMalloc(&buffer, sizeof(index_type) *
                                CUDA_MAX_NUM_THREADS_PER_BLOCK_2D * size[1]);
        int batch_length = CUDA_MAX_NUM_THREADS_PER_BLOCK_2D;
        int MAX_NUM_THREADS_PER_BLOCK_1D = batch_length;
        int num_batches = (size[0] + batch_length - 1) / batch_length;
        int last_row_ptrs = 0;
        for (auto batch = 0; batch < num_batches; batch++) {
            dim<2> batch_size = {0, 0};
            if (batch != num_batches - 1) {
                batch_size = dim<2>(
                    (batch + 1) * batch_length - batch_length * batch, 1);
            } else {
                batch_size = dim<2>(size[0] - batch_length * batch, 1);
            }

            ::dim3 grid_size(1,
                             batch_size[0] / MAX_NUM_THREADS_PER_BLOCK_1D + 1);
            ::dim3 grid_size2(1, 1);
            ::dim3 block_size(1, MAX_NUM_THREADS_PER_BLOCK_1D);

            shuffle_countsketch_init_col_idxs<<<grid_size2, block_size>>>(
                main_seed, static_cast<int>(size[1]), batch_length, buffer);
            cudaDeviceSynchronize();

            shuffle_countsketch_set_col_idxs_kernel<<<grid_size, block_size>>>(
                main_seed, static_cast<int>(batch_size[1]),
                static_cast<int>(size[1]), static_cast<int>(batch_size[0]),
                buffer, values + batch_length * batch * nnz_per_col);
            cudaDeviceSynchronize();
        }
        cudaFree(buffer);
    }
}

template <typename value_type>
__global__ void set_countsketch_values_kernel(double main_seed, int nelems,
                                              value_type* values) {
    int val_int;
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nelems) {
        int seed = hash64shift(idx) * ((int)main_seed);
        thrust::minstd_rand rng(seed);
        thrust::uniform_int_distribution<int> dist_vals(0, 1);
        //thrust::random::normal_distribution<float> dist_vals(0.0, 1.0);
        val_int = dist_vals(rng);
        values[idx] = (val_int == 0) ? -1.0 : 1.0;
    }
}

template <typename value_type, typename index_type>
__global__ void set_sparse_gaussian_values_kernel(double main_seed,
                                                  index_type k, index_type rows,
                                                  value_type* values) {
    int col_idx;
    value_type val;
    auto row = blockDim.y * blockIdx.y + threadIdx.y;
    auto col = blockDim.x * blockIdx.x + threadIdx.x;
    if ((row < rows) && (col < k)) {
        int seed = hash64shift(k * row + col) * ((int)main_seed);
        thrust::minstd_rand rng(seed);
        thrust::random::normal_distribution<float> dist_vals(0.0, 1.0);
        values[k * row + col] = dist_vals(rng);
    }
}

// implement this in a sequential manner
template <typename index_type>
__global__ void sort_col_idxs_kernel(size_t nnz_per_row, int rows, int cols,
                                     index_type* row_ptrs,
                                     index_type* col_idxs) {
    auto row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < rows) {
        auto rptrs = thrust::device_ptr<index_type>(row_ptrs);
        auto cidxs = thrust::device_ptr<index_type>(col_idxs);
        thrust::sort(thrust::seq, cidxs + row * nnz_per_row,
                     cidxs + (row + 1) * nnz_per_row);
    }
}

template <typename value_t>
Status generate_countsketch_csr_values(Sparse& S) {
    //auto size = dim<2>(sketch->get_size()[1], sketch->get_size()[0]);
    CsrView<value_t, int> view(S);
    size_t nelems{view.nnz_};
    double seed = time(NULL);
    ::dim3 grid_size((nelems + CUDA_MAX_NUM_THREADS_PER_BLOCK_1D - 1) /
                             CUDA_MAX_NUM_THREADS_PER_BLOCK_1D +
                         1,
                     1, 1);
    ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_1D, 1, 1);

    tomos::cuda::set_countsketch_values_kernel<<<grid_size, block_size>>>(
        seed, (int)nelems, view.values_);
    cudaDeviceSynchronize();

    return SUCCESS;
}  // end of generate_countsketch_csr_values function

template Status generate_countsketch_csr_values<double>(Sparse& S);
template Status generate_countsketch_csr_values<float>(Sparse& S);
template Status generate_countsketch_csr_values<__half>(Sparse& S);
template Status generate_countsketch_csr_values<__nv_bfloat16>(Sparse& S);

template <typename index_t>
Status generate_countsketch_csr_idxs(Sparse& S) {
    CsrSparsityView<index_t> view(S);
    size_t nptrs = view.size_[0] + 1;
    index_t* ptrs = view.row_ptrs_;
    size_t ncols = view.size_[1];
    index_t* idxs = view.col_idxs_;
    if (view.nnz_ != view.size_[0])
        throw std::invalid_argument(
            "generate_countsketch_csr_idxs: nnz_per_row == 1 only; the "
            "sketch's nnz must equal its number of rows.");
    unsigned int seed = static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    // Sets row_ptrs.
    {
        ::dim3 grid_size((nptrs + CUDA_MAX_NUM_THREADS_PER_BLOCK_1D - 1) /
                             CUDA_MAX_NUM_THREADS_PER_BLOCK_1D,
                         1);
        ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_1D, 1);
        tomos::cuda::set_row_ptrs_kernel<<<grid_size, block_size>>>(
            seed, 1, (int)nptrs, ptrs);
        cudaDeviceSynchronize();
    }

    // Sets col_idxs (nnz_per_row == 1 only; nnz_per_row > 1 is not yet implemented).
    {
        ::dim3 grid_size((nptrs + CUDA_MAX_NUM_THREADS_PER_BLOCK_1D - 1) /
                             CUDA_MAX_NUM_THREADS_PER_BLOCK_1D,
                         1);
        ::dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK_1D, 1);
        tomos::cuda::set_col_idxs_kernel<<<grid_size, block_size>>>(
            seed, 1, (int)(nptrs - 1), (int)ncols, ptrs, idxs);
        cudaDeviceSynchronize();
    }

    return SUCCESS;
}  // end of generate_countsketch_csr_idxs function

template Status generate_countsketch_csr_idxs<int>(Sparse& S);
template Status generate_countsketch_csr_idxs<long>(Sparse& S);

    // SplitMix64 hash and on-the-fly sign helpers
    __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        x = x ^ (x >> 31);
        return x;
    }

    __device__ __forceinline__ int sign_from_hash(uint64_t h) {
        return (h & 1ull) ? +1 : -1;
    }

    // On-the-fly hashing variant (no h/s arrays):
    template <typename T>
    __global__ void countsketch_apply_matrix_hashed(
        const T* __restrict__ X,  // n x B, col-major
        int ldx,
        T* __restrict__ Y,  // m x B, col-major (pre-zeroed)
        int ldy, int n, int m, int B, uint64_t bucket_seed,
        uint64_t sign_seed) {
        const int j = blockIdx.y * blockDim.y + threadIdx.y;  // column
        if (j >= B) return;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) {
            const T xi = X[i + (size_t)j * ldx];
            uint64_t hb = splitmix64((uint64_t)i ^ bucket_seed);
            uint64_t hs = splitmix64((uint64_t)i ^ sign_seed);
            int32_t bucket = (int32_t)(hb % (uint64_t)m);
            int sign = sign_from_hash(hs);
            atomicAdd(&Y[bucket + (size_t)j * ldy], (sign > 0 ? xi : -xi));
        }
    }

    // On-the-fly hashing variant (no h/s arrays):
    __global__ void countsketch_apply_matrix_hashed(
        const uint16_t* __restrict__ Xh,  // n x B, col-major
        const uint16_t* __restrict__ Xr,  // n x B, col-major
        int ldx,
        float* __restrict__ Y,  // m x B, col-major (pre-zeroed)
        int ldy, int n, int m, int B, uint64_t bucket_seed,
        uint64_t sign_seed) {
        const int j = blockIdx.y * blockDim.y + threadIdx.y;  // column
        if (j >= B) return;

        //@FIX: temp
        //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        //  float xi = 0.0;
        //  generic_pack_fp32(Xh[i + (size_t)j * ldx], Xr[i + (size_t)j * ldx], reinterpret_cast<uint16_t*>(&xi));
        //  uint64_t hb = splitmix64((uint64_t)i ^ bucket_seed);
        //  uint64_t hs = splitmix64((uint64_t)i ^ sign_seed);
        //  int32_t bucket = (int32_t)(hb % (uint64_t)m);
        //  int     sign   = sign_from_hash(hs);
        //  atomicAdd(&Y[bucket + (size_t)j * ldy], (sign > 0 ? xi : -xi));
        //}
    }

    // On-the-fly hashing variant (no h/s arrays):
    __global__ void countsketch_apply_matrix_hashed(
        const uint16_t* __restrict__ Xh,  // n x B, col-major
        const uint16_t* __restrict__ Xr,  // n x B, col-major
        int ldx,
        __half* __restrict__ Y,  // m x B, col-major (pre-zeroed)
        int ldy, int n, int m, int B, uint64_t bucket_seed,
        uint64_t sign_seed) {
        const int j = blockIdx.y * blockDim.y + threadIdx.y;  // column
        if (j >= B) return;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) {
            // Xh is (up to its stolen flag bit -- see fp32_16_pack.hpp)
            // directly a valid __half bit pattern; Xr only carries residual
            // bits needed to recombine to fp32 accuracy (the commented-out
            // generic_pack_fp32 call below), not a value in its own right.
            // A previous version of this line read `Xr[...]` here, which is
            // both the wrong array and an integer-to-half *value* conversion
            // rather than a bit-pattern reinterpret.
            __half xi = *reinterpret_cast<const __half*>(
                &Xh[i + (size_t)j * ldx]);
            // generic_pack_fp32(Xh[i + (size_t)j * ldx], Xr[i + (size_t)j * ldx], reinterpret_cast<uint16_t*>(&xi));
            uint64_t hb = splitmix64((uint64_t)i ^ bucket_seed);
            uint64_t hs = splitmix64((uint64_t)i ^ sign_seed);
            int32_t bucket = (int32_t)(hb % (uint64_t)m);
            int sign = sign_from_hash(hs);
            atomicAdd(&Y[bucket + (size_t)j * ldy], (sign > 0 ? xi : -xi));
        }
    }

    __global__ void countsketch_apply_matrix_hashed_half_tail(
        const __half* __restrict__ X, int ldx, float* __restrict__ Y, int ldy,
        int n, int m, int B, uint64_t bucket_seed, uint64_t sign_seed) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= B) return;
        int i = n - 1;
        float xi = __half2float(X[i + (size_t)j * ldx]);
        uint64_t hb = splitmix64((uint64_t)i ^ bucket_seed);
        uint64_t hs = splitmix64((uint64_t)i ^ sign_seed);
        int bucket = (int)(hb % (uint64_t)m);
        float sign = (hs & 1ULL) ? 1.f : -1.f;
        atomicAdd(&Y[bucket + (size_t)j * ldy], sign * xi);
    }

    // CountSketch using half2 loads and NumericType::FP32 accumulation
    template <int UNROLL = 2>
    __global__ void countsketch_apply_matrix_hashed_half2(
        const __half* __restrict__ X,  // n x B, col-major, base address is half*
        int ldx,
        float* __restrict__ Y,  // m x B, col-major (NumericType::FP32, pre-zeroed)
        int ldy, int n, int m, int B, uint64_t bucket_seed,
        uint64_t sign_seed) {
        const int j = blockIdx.y * blockDim.y + threadIdx.y;  // column
        if (j >= B) return;

        const int tid = threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        int i0 = blockIdx.x * blockDim.x * 2 + tid * 2;  // step by 2 for half2

        for (; i0 < n; i0 += UNROLL * stride * 2) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int i = i0 + u * stride * 2;
                if (i + 1 >= n) break;  // need two elements

                // reinterpret pointer as half2*
                const half2* X2 = reinterpret_cast<const half2*>(X + j * ldx);
                half2 h2 = X2[i >> 1];

                // convert to float2
                float2 xf = __half22float2(h2);

                // process element i
                uint64_t hb0 = splitmix64(((uint64_t)i) ^ bucket_seed);
                uint64_t hs0 = splitmix64(((uint64_t)i) ^ sign_seed);
                int bucket0 = hb0 % m;
                float s0 = (hs0 & 1ULL) ? 1.f : -1.f;
                atomicAdd(&Y[bucket0 + (size_t)j * ldy], s0 * xf.x);

                // process element i+1
                uint64_t hb1 = splitmix64(((uint64_t)(i + 1)) ^ bucket_seed);
                uint64_t hs1 = splitmix64(((uint64_t)(i + 1)) ^ sign_seed);
                int bucket1 = hb1 % m;
                float s1 = (hs1 & 1ULL) ? 1.f : -1.f;
                atomicAdd(&Y[bucket1 + (size_t)j * ldy], s1 * xf.y);
            }
        }
    }

    void countsketch_apply_impl(dim<2> size, uint16_t* Xh, uint16_t* Xr,
                                size_t ld_in, dim<2> size_y, __half* Y,
                                size_t ld_out) {
        dim3 block(128, 4);
        dim3 grid((size[0] + block.x - 1) / block.x,
                  (size[1] + block.y - 1) / block.y);
        grid.x = std::max(1u, std::min(grid.x, 65535u));
        grid.y = std::max(1u, std::min(grid.y, 65535u));
        cudaStream_t stream = 0;
        uint64_t bucket_seed = 0x12345678abcdef00ull;
        uint64_t sign_seed = 0xf00dd00d42babe55ull;
        countsketch_apply_matrix_hashed<<<grid, block, 0, stream>>>(
            Xh, Xr, size[0], Y, size_y[0], size[0], size_y[0], size[1],
            bucket_seed, sign_seed);
    }

    void countsketch_apply_impl(dim<2> size, uint16_t* Xh, uint16_t* Xr,
                                size_t ld_in, dim<2> size_y, float* Y,
                                size_t ld_out) {
        dim3 block(128, 4);
        dim3 grid((size[0] + block.x - 1) / block.x,
                  (size[1] + block.y - 1) / block.y);
        grid.x = std::max(1u, std::min(grid.x, 65535u));
        grid.y = std::max(1u, std::min(grid.y, 65535u));
        cudaStream_t stream = 0;

        uint64_t bucket_seed = 0x12345678abcdef00ull;
        uint64_t sign_seed = 0xf00dd00d42babe55ull;
        countsketch_apply_matrix_hashed<<<grid, block, 0, stream>>>(
            Xh, Xr, size[0], Y, size_y[0], size[0], size_y[0], size[1],
            bucket_seed, sign_seed);
    }

    template <typename value_t>
    void countsketch_apply_impl(dim<2> size, value_t * X, size_t ld_in,
                                dim<2> size_y, value_t* Y, size_t ld_out) {
        dim3 block(128, 4);
        dim3 grid((size[0] + block.x - 1) / block.x,
                  (size[1] + block.y - 1) / block.y);
        grid.x = std::max(1u, std::min(grid.x, 65535u));
        grid.y = std::max(1u, std::min(grid.y, 65535u));
        cudaStream_t stream = 0;

        uint64_t bucket_seed = 0x12345678abcdef00ull;
        uint64_t sign_seed = 0xf00dd00d42babe55ull;
        countsketch_apply_matrix_hashed<value_t><<<grid, block, 0, stream>>>(
            X, size[0], Y, size_y[0], size[0], size_y[0], size[1], bucket_seed,
            sign_seed);
        cudaDeviceSynchronize();
    }

    void countsketch_apply_impl(dim<2> size, __half * X, size_t ld_in,
                                dim<2> size_y, float* Y, size_t ld_out) {
        dim3 block(128, 4);
        dim3 grid((size[0] + block.x - 1) / block.x,
                  (size[1] + block.y - 1) / block.y);
        grid.x = std::max(1u, std::min(grid.x, 65535u));
        grid.y = std::max(1u, std::min(grid.y, 65535u));
        cudaStream_t stream = 0;

        uint64_t bucket_seed = 0x12345678abcdef00ull;
        uint64_t sign_seed = 0xf00dd00d42babe55ull;
        countsketch_apply_matrix_hashed_half2<4><<<grid, block, 0, stream>>>(
            X, size[0], Y, size_y[0], size[0], size_y[0], size[1], bucket_seed,
            sign_seed);
        if (size[0] & 1) {
            int tail_block = std::min((int)size[1], 128);
            int tail_grid = ((int)size[1] + tail_block - 1) / tail_block;
            countsketch_apply_matrix_hashed_half_tail<<<tail_grid, tail_block,
                                                        0, stream>>>(
                X, size[0], Y, size_y[0], size[0], size_y[0], size[1],
                bucket_seed, sign_seed);
        }
        cudaDeviceSynchronize();
    }

    template void countsketch_apply_impl<float>(dim<2> size, float* X,
                                                size_t ld_in, dim<2> size_y,
                                                float* Y, size_t ld_out);
    template void countsketch_apply_impl<__half>(dim<2> size, __half * X,
                                                 size_t ld_in, dim<2> size_y,
                                                 __half* Y, size_t ld_out);
    template void countsketch_apply_impl<__nv_bfloat16>(dim<2> size, __nv_bfloat16 * X,
                                                 size_t ld_in, dim<2> size_y,
                                                 __nv_bfloat16* Y, size_t ld_out);

    __device__ inline uint64_t splitmix64_dev(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    }

    __device__ inline int sign_from_hash_dev(uint64_t h) {
        return (h & 1ULL) ? 1 : -1;
    }

    __global__ void countsketch_build_hashes_gpu(
        int n, int m, uint64_t bucket_seed, uint64_t sign_seed,
        int32_t* __restrict__ h,  // [n]
        int8_t* __restrict__ s)   // [n]
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = tid; i < n; i += stride) {
            uint64_t hb = splitmix64_dev((uint64_t)i ^ bucket_seed);
            uint64_t hs = splitmix64_dev((uint64_t)i ^ sign_seed);

            h[i] = (int32_t)(hb % (uint64_t)m);
            s[i] = (int8_t)sign_from_hash_dev(hs);  // +- 1
        }
    }

    // Prehashed half2 CountSketch: X in half, Y in float
    template <int UNROLL = 2>
    __global__ void countsketch_apply_matrix_prehashed_half2(
        const __half* __restrict__ X,  // [n x B], col-major
        int ldx,
        float* __restrict__ Y,  // [m x B], col-major (NumericType::FP32, pre-zeroed)
        int ldy,
        const int32_t* __restrict__ h,  // [n] bucket indices
        const int8_t* __restrict__ s,   // [n] signs ±1
        int n, int m, int B) {
        // Column index
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= B) return;

        // Thread index along row dimension
        const int tx = threadIdx.x;
        const int blk = blockIdx.x;
        const int threads_per_block = blockDim.x;
        const int blocks_x = gridDim.x;

        // We work in units of 2 rows (half2)
        const int pair_stride = threads_per_block * blocks_x * 2;

        // First row index this thread handles (even)
        int i0 = (blk * threads_per_block + tx) * 2;

        // Base pointer for column j, as half2*
        const half2* Xcol2 =
            reinterpret_cast<const half2*>(X + (size_t)j * ldx);

        for (; i0 + 1 < n; i0 += UNROLL * pair_stride) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int i =
                    i0 +
                    u * pair_stride;  // row index for first element in the pair
                if (i + 1 >= n) break;  // need two elements

                // Index in half2 space
                int idx2 = i >> 1;  // i/2

                // Load two halves as one half2
                half2 h2 = Xcol2[idx2];
                float2 x2 =
                    __half22float2(h2);  // x2.x = X(i,j), x2.y = X(i+1,j)

                // Load buckets and signs
                int b0 = h[i];
                int b1 = h[i + 1];

                float s0 = (float)s[i];
                float s1 = (float)s[i + 1];

                // Fuse sign into the values (no extra memory traffic)
                float v0 = x2.x * s0;
                float v1 = x2.y * s1;

                // Atomic accumulations into Y(bucket, j)
                atomicAdd(&Y[b0 + (size_t)j * ldy], v0);
                atomicAdd(&Y[b1 + (size_t)j * ldy], v1);
            }
        }
    }

    __global__ void countsketch_apply_matrix_prehashed_half_tail(
        const __half* __restrict__ X, int ldx, float* __restrict__ Y, int ldy,
        const int32_t* __restrict__ h, const int8_t* __restrict__ s, int n,
        int m, int B) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= B) return;

        int i = n - 1;  // last row
        __half hx = X[i + (size_t)j * ldx];
        float xi = __half2float(hx);

        int bucket = h[i];
        float sign = (float)s[i];

        atomicAdd(&Y[bucket + (size_t)j * ldy], sign * xi);
    }

    __host__ void build_hash_impl(size_t m, size_t n, int32_t* __restrict__ d_h,
                                  int8_t* __restrict__ d_s) {
        uint64_t bucket_seed = 0x12345678abcdef00ull;
        uint64_t sign_seed = 0xf00dd00d42babe55ull;
        dim3 block_hash(256);
        dim3 grid_hash((n + block_hash.x - 1) / block_hash.x);
        countsketch_build_hashes_gpu<<<grid_hash, block_hash>>>(
            n, m, bucket_seed, sign_seed, d_h, d_s);
    }

    __host__ void apply_countsketch_prehashed(
        size_t m, size_t n, size_t B, int32_t* __restrict__ d_h,
        int8_t* __restrict__ d_s, const __half* __restrict__ d_X, size_t ldx,
        float* __restrict__ d_Y, size_t ldy) {
        cudaStream_t stream = 0;
        // 2) Launch main half2 kernel for even part of n
        {
            dim3 block(128, 2);  // 128 threads over rows, 2 columns per block
            dim3 grid(
                ((n / 2) + block.x - 1) / block.x,  // work in pairs => n/2
                (B + block.y - 1) / block.y);

            countsketch_apply_matrix_prehashed_half2<2>
                <<<grid, block, 0, stream>>>(d_X, ldx, d_Y, ldy, d_h, d_s, n, m,
                                             B);
        }

        // 3) Tail kernel if n is odd
        if (n & 1) {
            int block_tail = 128;
            int grid_tail = (B + block_tail - 1) / block_tail;
            countsketch_apply_matrix_prehashed_half_tail<<<
                grid_tail, block_tail, 0, stream>>>(d_X, ldx, d_Y, ldy, d_h,
                                                    d_s, n, m, B);
        }
    }

}  // end of namespace cuda
}  // end of namespace tomos

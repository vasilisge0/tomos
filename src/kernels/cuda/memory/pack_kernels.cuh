#ifndef PACK_KERNELS_CUH
#define PACK_KERNELS_CUH

#include <mma.h>
#include <cuda/pipeline>

using namespace nvcuda;

__device__ __forceinline__ void memcpy_async_16B(
    ::cuda::pipeline<::cuda::thread_scope_thread>& pipe, void* dst,
    const void* src) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
    //     cuda::memcpy_async(pipe, dst, src, ::cuda::aligned_size_t<16>(16));
    // #else
    cuda::memcpy_async(dst, src, (size_t)16, pipe);
#endif
}

namespace tomos {

extern __constant__ mask32_sign MASK_SIGN_V2;
extern __constant__ mask32_fp_exponent_from_fp16
    MASK_FP16_EXPONENT_FROM_FP16_V2;
extern __constant__ mask32_fp_exponent_from_fp16_s3
    MASK_FP16_EXPONENT_FROM_FP16_S3_V2;
extern __constant__ mask32_7 MASK_7_V2;
extern __constant__ mask32_bias_correction BIAS_CORRECTION_V2;
extern __constant__ mask32_bias_fp32_mantissa_from_fp16
    MASK_FP32_MANTISSA_FROM_FP16_V2;

// __device__ __forceinline__ void cp_async_commit() {
// #if __CUDA_ARCH__ >= 800
//   asm volatile("cp.async.commit_group;\n" ::);
// #endif
// }

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                              \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

#define NNN 2048

static inline __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}
static inline __host__ __device__ int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// --------- Tunables shared across variants ----------
constexpr int TB_M = 128;
constexpr int TB_N = 128;
constexpr int PAD_K = 1;   // smem padding to avoid bank conflicts (A)
constexpr int PAD_N = 1;   // smem padding to avoid bank conflicts (B)
constexpr int BLK_X = 32;  // 32 x 8 = 256 threads (8 warps)
constexpr int BLK_Y = 8;
constexpr int TMICRO = 8;  // per-thread rows
constexpr int NMICRO = 4;  // per-thread cols; we compute two strips: +0 and +16

#ifndef ENABLE_SAFE_TAILS
#define ENABLE_SAFE_TAILS \
    0  // 0 = fastest (requires K % TB_K == 0), 1 = safe tails on last K-chunk
#endif
// Replace your load_half2_unaligned with this:
__device__ inline uint32_t ld_half2_safe(const uint16_t* p) {
    uintptr_t addr = (uintptr_t)p;
    if ((addr & 0x3u) == 0u) {
        // naturally 4B-aligned → one 32-bit load is safe
        uint32_t w = *reinterpret_cast<const uint32_t*>(p);  // avoid __ldg here
        uint16_t lo = (uint16_t)(w & 0xFFFFu);
        uint16_t hi = (uint16_t)(w >> 16);
        // return __halves2half2(__ushort_as_half(lo), __ushort_as_half(hi));
        return w;
    } else {
        // 2B-misaligned → load two scalar halves safely
        uint32_t h;
        reinterpret_cast<uint16_t*>(&h)[0] = p[1];
        reinterpret_cast<uint16_t*>(&h)[1] = p[0];
        return h;
        // __half h0 = p[0];
        // __half h1 = p[1];
        // return __halves2half2(h0, h1);
    }
}

// original version
//__forceinline__ __device__
//float generic_pack_fp32_f(uint16_t xh, uint16_t xr)
//{
//    // Work in 32-bit
//    const uint32_t xh32 = xh;
//    const uint32_t xr32 = xr;
//
//    const uint32_t sign   = xh32 & MASK_SIGN;
//    const uint32_t mant_r = xh32 & 0x03FFu;
//    const uint32_t exp_r  = (xh32 & MASK_FP16_EXPONENT_FROM_FP16) >> 3;
//    const uint32_t modr   = xr32 >> 2;
//    const uint32_t special = xh32 & 1u;
//
//    uint32_t hi_r = mant_r >> 1;
//    uint32_t a;
//
//    if (!special) {
//        a = exp_r + BIAS_CORRECTION;
//
//        const uint32_t roundBit = xr32 & 1u;
//        const uint32_t nz       = (mant_r != 0u);
//        const uint32_t z        = 1u ^ nz;
//
//        const uint32_t opt0 = roundBit & nz;
//        const uint32_t opt1 = roundBit & z;
//
//        hi_r = hi_r - opt0 + opt1 * (HI_ORIG_INIT - hi_r);
//        a    = a - opt1 * ONE_SEVEN;
//    } else {
//        a = TMP113 - exp_r;
//    }
//
//    // Produce the exact same 16-bit halves your code wrote to tmp[]
//    const uint32_t hi16 = (sign | a | (hi_r >> 2)) & 0xFFFFu;
//    const uint32_t lo16 = ((hi_r << 14) | modr) & 0xFFFFu;
//
//    const uint32_t bits = (hi16 << 16) | lo16;
//    return __uint_as_float(bits);
//}

// optimized version
__forceinline__ __device__ float generic_pack_fp32_f(uint16_t xh, uint16_t xr) {
    const uint32_t xh32 = (uint32_t)xh;
    const uint32_t xr32 = (uint32_t)xr;

    const uint32_t sign = xh32 & MASK_SIGN;
    const uint32_t mant_r = xh32 & 0x03FFu;
    const uint32_t exp_r = (xh32 & MASK_FP16_EXPONENT_FROM_FP16) >> 3;
    const uint32_t modr = xr32 >> 2;
    const uint32_t special = xh32 & 1u;
    const uint32_t roundBit = xr32 & 1u;

    const uint32_t hi0 = mant_r >> 1;

    const uint32_t mant_nz = (mant_r != 0u);
    const uint32_t opt1 = roundBit & (mant_nz ^ 1u);
    const uint32_t opt0 = roundBit & mant_nz;

    const uint32_t hi_ns = hi0 - opt0 + opt1 * (HI_ORIG_INIT - hi0);
    const uint32_t a_ns = exp_r + BIAS_CORRECTION - opt1 * ONE_SEVEN;
    const uint32_t a_sp = TMP113 - exp_r;

    const uint32_t smask = 0u - special;

    const uint32_t hi_r = (hi0 & smask) | (hi_ns & ~smask);
    const uint32_t a = (a_sp & smask) | (a_ns & ~smask);

    const uint32_t bits = (((sign | a | (hi_r >> 2)) & 0xFFFFu) << 16) |
                          (((hi_r << 14) | modr) & 0xFFFFu);

    return __uint_as_float(bits);
}

// ---------------------------------------------
// Scale y by beta (in-place)
// ---------------------------------------------
extern __global__ void scale_y(float* __restrict__ y, int M, float beta);

}  // namespace tomos

#endif  // PACK_KERNELS_CUH
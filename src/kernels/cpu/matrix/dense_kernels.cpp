#include <cstdint>
#include "matrix.hpp"

#define BLOCK_SIZE 32

namespace tomos {

union D32 {
    float s;
    uint32_t u;

    __device__ void print_binary_cuda() {
        for (int i = static_cast<int>(sizeof(uint32_t) * 8 - 1); i >= 0; --i) {
            // std::cout << ((u >> i) & 1);
            printf("%d", ((u >> i) & 1));
        }
        // std::cout << std::endl;
        printf("\n");
    }

};  // end of union Display

union D16 {
    uint16_t u;

    __device__ void print_binary_cuda() {
        for (int i = static_cast<int>(sizeof(uint16_t) * 8 - 1); i >= 0; --i) {
            // std::cout << ((u >> i) & 1);
            printf("%d", ((u >> i) & 1));
            if (i == 48) {
                // std::cout << "|";
                printf("|");
            }
        }
        // std::cout << std::endl;
        printf("\n");
    }

};  // end of union Display

const int BLK = BLOCK_SIZE;

static const unsigned char scalar0_h = 2 ^ 6;
static const unsigned char scalar1_h = 2 ^ 2;

static __constant__ unsigned char scalar0;
static __constant__ unsigned char scalar1;

inline constexpr uint32_t create_bitmask(uint32_t l, uint32_t f) {
    uint32_t t = 0;
    for (uint32_t i = f; i <= l; ++i) {
        t = t | (static_cast<uint32_t>(1) << i);
    }
    return t;
}  // end of create_bitmask

inline constexpr unsigned char create_bitmask(unsigned char l,
                                              unsigned char f) {
    unsigned char t = 0;
    for (unsigned char i = f; i <= l; ++i) {
        t = t | (static_cast<unsigned char>(1) << i);
    }
    return t;
}  // end of create_bitmask

inline constexpr uint16_t create_bitmask16(uint16_t l, uint16_t f) {
    uint16_t t = 0;
    for (uint16_t i = f; i <= l; ++i) {
        t = t | (static_cast<uint16_t>(1) << i);
    }
    return t;
}  // end of create_bitmask

const unsigned char MU_64_16_0_h =
    create_bitmask((unsigned char)7, (unsigned char)7);
const unsigned char MU_64_16_1_h =
    create_bitmask((unsigned char)0, (unsigned char)0);
const unsigned char MU_64_16_2_h =
    create_bitmask((unsigned char)7, (unsigned char)2);
const unsigned char MU_64_16_3_h =
    create_bitmask((unsigned char)1, (unsigned char)0);
const unsigned char MU_64_16_4_h =
    create_bitmask((unsigned char)5, (unsigned char)0);
const unsigned char MU_64_48_0_h =
    create_bitmask((unsigned char)6, (unsigned char)1);
const unsigned char MU_64_48_1_h =
    create_bitmask((unsigned char)7, (unsigned char)2);

const unsigned char MP_16_64_0_h =
    create_bitmask((unsigned char)7, (unsigned char)7);
const unsigned char MP_16_64_1_h =
    create_bitmask((unsigned char)6, (unsigned char)0);
const unsigned char MP_48_64_0_h =
    create_bitmask((unsigned char)7, (unsigned char)2);
const unsigned char MP_48_64_1_h =
    create_bitmask((unsigned char)1, (unsigned char)0);

inline unsigned int create_bitmask32_0() {
    unsigned int t;
    ((unsigned char *)&t)[0] = MP_16_64_0_h;
    ((unsigned char *)&t)[1] = MP_16_64_1_h;
    ((unsigned char *)&t)[2] = MP_16_64_0_h;
    ((unsigned char *)&t)[3] = MP_16_64_1_h;
    return t;
}

inline unsigned int create_bitmask32_1() {
    unsigned int t;
    ((unsigned char *)&t)[0] = MP_48_64_0_h;
    ((unsigned char *)&t)[1] = MP_48_64_1_h;
    ((unsigned char *)&t)[2] = MP_48_64_0_h;
    ((unsigned char *)&t)[3] = MP_48_64_1_h;
    return t;
}

inline unsigned int create_bitmask32_2() {
    unsigned int t;
    ((unsigned char *)&t)[0] = MU_64_16_0_h;
    ((unsigned char *)&t)[1] = MU_64_16_1_h;
    ((unsigned char *)&t)[2] = MU_64_16_2_h;
    ((unsigned char *)&t)[3] = MU_64_16_3_h;
    return t;
}

const unsigned int MP_16_32_0_h = create_bitmask32_0();
const unsigned int MP_48_32_0_h = create_bitmask32_1();
const unsigned int MU32_64_16_0_h = create_bitmask32_2();

static __constant__ unsigned char MU_64_16_0;
static __constant__ unsigned char MU_64_16_1;
static __constant__ unsigned char MU_64_16_2;
static __constant__ unsigned char MU_64_16_3;
static __constant__ unsigned char MU_64_16_4;
static __constant__ unsigned char MU_64_48_0;
static __constant__ unsigned char MU_64_48_1;

static __constant__ unsigned char MP_16_64_0;
static __constant__ unsigned char MP_16_64_1;
static __constant__ unsigned char MP_48_64_0;
static __constant__ unsigned char MP_48_64_1;

static __constant__ unsigned int MU32_64_16_0;

struct W16 {
    unsigned char v_[2];
};

struct W48 {
    unsigned char v_[6];
};

const uint16_t MASK_SIGN = create_bitmask16(15, 15);
const uint16_t MASK_SIGN_AMD_MSB = create_bitmask16(15, 14);
const uint16_t MASK_BODY = create_bitmask16(14, 0);

const uint16_t MASK_NON_SIGN = create_bitmask16(15, 0);
const uint16_t BIAS_CORRECTION = ((uint16_t)112) << 7;
const uint16_t MASK_FP16_EXPONENT_FROM_FP16 = create_bitmask16(14, 10);
const uint16_t MASK_FP16_EXPONENT_FROM_FP16_3MSB = create_bitmask16(14, 12);
const uint16_t MASK_FP16_EXPONENT_FROM_FP16_4MSB = create_bitmask16(14, 11);
const uint16_t MASK_FP32_CORRECT_EXPONENT_BIAS = ((112) << 10);
const uint16_t MASK_FP32_SIGN = create_bitmask16(15, 15);
const uint16_t MASK_FP32_MANTISSA_UPPER = create_bitmask16(6, 0);
const uint16_t MASK_FP32_MANTISSA_FROM_FP16 = create_bitmask16(9, 0);
const uint16_t FP32_MANTISSA_LOWER = create_bitmask16(15, 0);

const uint16_t MASK_FP32_EXPONENT_FROM_FP16 = create_bitmask16(14, 7);
// const uint16_t MASK_FP32_EXPONENT_FROM_FP16_LSB4 = create_bitmask16(10, 7);
const uint16_t MASK_FP16_MANTISSA_FROM_FP32_UPPER = create_bitmask16(6, 0);
const uint16_t MASK_FP16_MANTISSA_FROM_FP16_UPPER = create_bitmask16(6, 0);
const uint16_t MASK_FP16_MANTISSA_FROM_FP16 = create_bitmask16(9, 0);

const uint16_t MASK_FP32_EXPONENT_FROM_FP16_LSB7 = create_bitmask16(13, 7);
const uint16_t MASK_FP32_EXPONENT_FROM_FP16_LSB5 = create_bitmask16(11, 7);
const uint16_t MASK_FP32_EXPONENT_FROM_FP16_LSB4 = create_bitmask16(10, 7);
const uint16_t MASK_FP32_EXPONENT_FROM_FP16_RES = create_bitmask16(14, 12);

const uint16_t MASK_FP16_MSB12 = create_bitmask16(15, 4);
const uint16_t MASK_FP16_MSB13 = create_bitmask16(15, 3);
const uint16_t MASK_FP16_MSB15 = create_bitmask16(15, 1);
const uint16_t MASK_FP32_EXPONENT_FROM_FP16_E13 = create_bitmask16(14, 12);
const uint16_t MASK_FP16_LSB3 = create_bitmask16(2, 0);
const uint16_t MASK_EXPONENT_FP16_LSB4 = create_bitmask16(13, 10);
const uint16_t MASK_MANTISSA_FP16_MSB9 = create_bitmask16(9, 1);
const uint16_t MASK_ROUND_BIT_FP16 = create_bitmask16(0, 0);
const uint16_t MASK_FP16_EXPONENT_MSB2 = create_bitmask16(15, 14);
const uint16_t MASK_FP16_EXPONENT_2NDBIT = create_bitmask16(14, 14);
const uint16_t MASK_FP16_MSB14 = create_bitmask16(15, 2);

const uint32_t MASK_FP32_EXPONENT_MSB9 = create_bitmask((uint32_t)31, 23);
const uint32_t MASK_FP32_EXPONENT = create_bitmask((uint32_t)30, 23);
const uint32_t MASK_FP32_EXPONENT_MSB2 = create_bitmask((uint32_t)31, 30);
const uint32_t MASK_FP32_SIGN_AND_EXPONENT = create_bitmask((uint32_t)31, 23);
const uint32_t MASK_FP32_LSB3 = create_bitmask((uint32_t)2, 0);
const uint32_t MASK_UPPER_16 = create_bitmask16(31, 16);
const uint32_t MASK_LOWER_16 = create_bitmask16(15, 0);

const uint HI_ORIG_INIT = (1u << 9) - 1;
const uint16_t TMP113 = (113u << 7);
const uint16_t ONE_NINE = (1u << 9) - 1;
const uint16_t ONE_TEN = 1u << 10;
const uint16_t ONE_SEVEN = 1u << 7;

inline uint32_t create_fp16_exponent_from_fp16_v2() {
    uint32_t m = 0;
    ((uint16_t *)&m)[1] = MASK_FP16_EXPONENT_FROM_FP16;
    ((uint16_t *)&m)[0] = MASK_FP16_EXPONENT_FROM_FP16;
    return m;
}

inline uint32_t create_bias_correction_v2() {
    uint32_t m = 0;
    ((uint16_t *)&m)[1] = BIAS_CORRECTION;
    ((uint16_t *)&m)[0] = BIAS_CORRECTION;
    return m;
}

inline uint32_t create_fp32_mantissa_from_fp16_v2() {
    uint32_t m = 0;
    ((uint16_t *)&m)[1] = MASK_FP32_MANTISSA_FROM_FP16;
    ((uint16_t *)&m)[0] = MASK_FP32_MANTISSA_FROM_FP16;
    return m;
}

inline uint32_t create_mask_7_v2() {
    uint32_t m = 0;
    ((uint16_t *)&m)[1] = (uint16_t)7;
    ((uint16_t *)&m)[0] = (uint16_t)7;
    return m;
}

union mask32_sign {
    uint32_t v32;
    uint16_t v16[2] = {MASK_SIGN, MASK_SIGN};
};

union mask32_fp_exponent_from_fp16 {
    uint32_t v32;
    uint16_t v16[2] = {MASK_FP16_EXPONENT_FROM_FP16,
                       MASK_FP16_EXPONENT_FROM_FP16};
};

union mask32_fp_exponent_from_fp16_s3 {
    uint32_t v32;
    uint16_t v16[2] = {MASK_FP16_EXPONENT_FROM_FP16 >> 3,
                       MASK_FP16_EXPONENT_FROM_FP16 >> 3};
};

union mask32_7 {
    uint32_t v32;
    uint16_t v16[2] = {(uint16_t)7, (uint16_t)7};
};

union mask32_bias_correction {
    uint32_t v32;
    uint16_t v16[2] = {BIAS_CORRECTION, BIAS_CORRECTION};
};

union mask32_bias_fp32_mantissa_from_fp16 {
    uint32_t v32;
    uint16_t v16[2] = {MASK_FP32_MANTISSA_FROM_FP16,
                       MASK_FP32_MANTISSA_FROM_FP16};
};

// @simple alternative of best
__forceinline__ __device__ void generic_pack_fp32(uint16_t xh, uint16_t xr,
                                                  uint16_t *__restrict__ tmp) {
    // Work in 32-bit (better match to A100 int units)
    const uint32_t xh32 = xh;
    const uint32_t xr32 = xr;

    const uint32_t sign = xh32 & MASK_SIGN;  // fits in 16 bits
    const uint32_t mant_r = xh32 & 0x03FFu;
    const uint32_t exp_r = (xh32 & MASK_FP16_EXPONENT_FROM_FP16) >> 3;
    const uint32_t modr = xr32 >> 2;

    const uint32_t special = xh32 & 1u;  // 1 if special, 0 otherwise

    uint32_t hi_r = mant_r >> 1;
    uint32_t a;

    if (!special) {
        // ===== Non-special / hot path =====
        a = exp_r + BIAS_CORRECTION;

        const uint32_t roundBit = xr32 & 1u;  // 0 or 1
        const uint32_t nz = (mant_r != 0u);   // 0 or 1
        const uint32_t z = 1u ^ nz;           // 1 if mant_r == 0

        const uint32_t opt0 = roundBit & nz;  // decrement hi_r
        const uint32_t opt1 = roundBit & z;   // reset hi_r, adjust exponent

        // hi_r: original semantics
        //   if (roundBit && mant_r != 0) hi_r--;
        //   if (roundBit && mant_r == 0) hi_r = HI_ORIG_INIT;
        hi_r = hi_r - opt0 + opt1 * (HI_ORIG_INIT - hi_r);

        // exponent:
        //   if (roundBit && mant_r == 0) a -= ONE_SEVEN;
        a = a - opt1 * ONE_SEVEN;
    } else {
        // ===== Special / rare path =====
        // Rounding was effectively a no-op in your original "special" case
        a = TMP113 - exp_r;
    }

    // Pack back to 16-bit
    tmp[1] = static_cast<uint16_t>(sign | a | (hi_r >> 2));
    tmp[0] = static_cast<uint16_t>((hi_r << 14) | modr);
}

inline void generic_unpack_fp32(uint16_t *tmp, uint16_t *wh, uint16_t *wr,
                                uint32_t *sflags, int idx) {
    float *f = (float *)tmp[0];
    *wh = 0;
    *wr = 0;

    // *wh = MASK_FP32_EXPONENT_FROM_FP16 & tmp[1];
    uint16_t t0 = MASK_FP32_EXPONENT_FROM_FP16 & tmp[1];

    uint16_t exp_32 = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >>
                      7;  // fp32 exponent fit in a fp16 variable.
    bool is_special_value = (exp_32 > 142) || (exp_32 < 113);

    {
        uint32_t in = *reinterpret_cast<uint32_t *>(tmp);
        uint32_t s32 = (in >> 31) & 0x1u;
        uint32_t e32 = (in >> 23) & 0xFFu;
        uint32_t m32 = in & 0x7FFFFFu;
        uint16_t s16 = (uint16_t)s32;

        // 3) Normal FP32: compute unbiased exponent
        uint32_t E32 = e32 - 127;  // FP32 bias = 127
        uint32_t e16i = E32 + 15;  // FP16 bias = 15

        // Overflow
        if (e32 >= 113u) {
            // printf("overflow_________<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, %f,
            // %u\n", *reinterpret_cast<float*>(tmp), );
            uint16_t e16 = (uint16_t)e16i;

            // 5) Build 24-bit mantissa with implicit leading 1: 1.mmmm...
            uint32_t mant24 = (1u << 23) | m32;

            // 6) Take the TOP 10 bits for our rounding scheme
            //    mant24 bits: [23 .. 0]
            //    top10 = bits [23..14]
            uint32_t top10 = (mant24 >> 13) & 0x3FFu;  // 10 bits: b9..b0

            // Split into:
            //   mant9 = b9..b1 (we keep these 9 bits)
            //   roundb = b0   (we use this as the round bit)
            uint16_t mant9 = (uint16_t)(top10 >> 1);   // 9 bits
            uint16_t roundb = (uint16_t)(top10 & 1u);  // 0 or 1

            // 7) Round at the 9th MSB:
            //    - roundb == 0 -> round down (truncate)
            //    - roundb == 1 -> round up (mant9++)

            // if (e16i <= 0) {
            //     // Too small for normal FP16; we drop to 0 here (no subnormals).
            //     uint16_t e16 = 0u;
            //     uint16_t m16 = 0u;
            //     // uint16_t m16 = (uint16_t)(mant9 << 1);
            //     *wh = (uint16_t)((s16 << 15) | (e16 << 10) | m16);
            //     *wr = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 12; // @FIX: use
            //     this for subnormals only *wr = *wr | (tmp[0] << 2) &
            //     MASK_FP16_MSB12;          // one bit will be lost in the wh part so
            //     we need to preserve it in wr, but in the special number we keep it!
            //     *wr = 0;
            //     return;
            // }
            // else {

            // }

            *wr = *wr |
                  (tmp[0]
                   << 2);  // one bit will be lost in the wh part so we need to
            // preserve it in wr, but in the special number we keep it!
            if (roundb) {
                mant9 += 1;
                *wr = *wr | 1u;
                if (mant9 ==
                    0x200u) {  // 0b10_0000_0000 (512) -> overflow of 9 bits
                    mant9 = 0;
                    e16 += 1;
                    if (e16 >= 31u) {
                        // exponent overflow -> Inf
                        uint16_t e_inf = 0x1Fu;
                        uint16_t m_inf = 0u;
                        // return (uint16_t)((s16 << 15) | (e_inf << 10) | m_inf);
                        *wh = (uint16_t)((s16 << 15) | (e_inf << 10) | m_inf);
                        return;
                    }
                }
            }

            // 8) Construct final 10-bit mantissa (LSB = 0: we only use 9 bits of
            // significance)
            uint16_t m16 =
                (uint16_t)(mant9
                           << 1);  // b9..b1 << 1 -> b9..b1,0 last bit is zero
                                   // because we don't have a special numbers

            // 9) Pack FP16 bits
            *wh = (uint16_t)((s16 << 15) | (e16 << 10) | m16);
            return;
        }

        // 5) Subnormal fp16: 103 <= e32 < 113
        if (e32 >= 103u) {
            *wh = 0u;

            // We are below the normal exponent range of fp16.
            // Let shift = 113 - e32 (1..10). We right-shift mant24 more to encode
            // subnormal. uint32_t mant24 = (1u << 23) | m32;
            uint32_t mant24 = m32;
            uint32_t shift = 113u - e32;  // how far below normal range we are

            // Idea: for subnormal fp16 we want a 10-bit mantissa m_sub from:
            //   m_sub ≈ mant24 >> (shift + 13)
            // Then we apply the same 9-MSB rounding rule on that 10-bit m_sub.

            // uint32_t top10 = (mant24 >> (shift + 13)) & 0x3FFu; // 10 bits
            uint32_t top10 = (mant24 >> (13)) & 0x3FFu;  // 10 bits

            uint16_t mant9 = (uint16_t)(top10 >> 1);   // 9 bits
            uint16_t roundb = (uint16_t)(top10 & 1u);  // 0 or 1

            *wr = *wr | (tmp[0] << 2);
            *wh = 1u;  // test

            if (roundb) {
                mant9 += 1;
                *wr = *wr | 1u;  // this should work but it doesn't
                if (top10 == 0x400u) {
                    // If mant9 overflows 9 bits here, that means we just crossed
                    // into the smallest *normal* fp16 value (exponent=1, mantissa=0).
                    uint16_t e16 = 0u;
                    uint16_t m16 = 0x200u;
                    *wh = 0u;  // test
                    *wh = *wh | (uint16_t)((s16 << 15) | (e16 << 10) | m16);
                    return;
                }
            }

            uint16_t m16 = (uint16_t)(mant9 << 1);  // LSB = 0
            // uint16_t e16 = 0u;                     // exponent = 0 → subnormal
            uint16_t e16 = (uint16_t)e16i;
            uint32_t tmp = top10;

            e16 = (uint16_t)(e32);

            uint16_t m32_16;
            if (e16 == 0) {
                m32_16 = m16;
            } else {
                m32_16 = (uint16_t)(m32 >> 7);
            }

            // // Underflow-to-zero if mant9 ended up 0
            // if (m16 == 0u) {
            //     *wh = *wh | (uint16_t)(s16 << 15);  // +0 or -0
            //     return;
            // }
            *wh = *wh | (uint16_t)((s16 << 15) | (shift << 10) |
                                   (MASK_FP16_MSB15 & (m32_16 >> 6)));
            return;
        } else {
            *wh = 0u;

            // We are below the normal exponent range of fp16.
            // Let shift = 113 - e32 (1..10). We right-shift mant24 more to encode
            // subnormal. uint32_t mant24 = (1u << 23) | m32;
            uint32_t mant24 = m32;
            uint32_t shift = 113u - e32;  // how far below normal range we are

            // Idea: for subnormal fp16 we want a 10-bit mantissa m_sub from:
            //   m_sub ≈ mant24 >> (shift + 13)
            // Then we apply the same 9-MSB rounding rule on that 10-bit m_sub.

            // uint32_t top10 = (mant24 >> (shift + 13)) & 0x3FFu; // 10 bits
            uint32_t top10 = (mant24 >> (13)) & 0x3FFu;  // 10 bits

            uint16_t mant9 = (uint16_t)(top10 >> 1);   // 9 bits
            uint16_t roundb = (uint16_t)(top10 & 1u);  // 0 or 1

            *wr = *wr | (tmp[0] << 2);
            if (roundb) {
                mant9 += 1;
                *wr = *wr | 1u;  // this should work but it doesn't
                if (mant9 == 0x200u) {
                    // If mant9 overflows 9 bits here, that means we just crossed
                    // into the smallest *normal* fp16 value (exponent=1, mantissa=0).
                    uint16_t e16 = 1u;
                    uint16_t m16 = 0u;
                    *wh = (uint16_t)((s16 << 15) | (e16 << 10) | m16);
                    return;
                }
            }

            *wh = 1u;                               // test
            uint16_t m16 = (uint16_t)(mant9 << 1);  // LSB = 0
            uint16_t e16 = (uint16_t)e16i;
            uint32_t tmp = top10;
            uint16_t k = 0;

            e16 = (uint16_t)(e32);

            uint16_t m32_16;
            if (e16 == 0) {
                m32_16 = m16;
            } else {
                m32_16 = (uint16_t)(m32 >> 7);
            }

            // Underflow-to-zero if mant9 ended up 0
            if (m16 == 0u) {
                *wh = *wh | (uint16_t)(s16 << 15);  // +0 or -0
                return;
            }

            *wh = *wh | (uint16_t)((s16 << 15) | (shift << 10) |
                                   (MASK_FP16_MSB15 & (m32_16 >> 6)));
            // *wh = *wh | (uint16_t)((s16 << 15) | ((shift-1u) << 10) |
            // (MASK_FP16_MSB15 & (m32_16 >> 6))); *wh = *wh | (uint16_t)((s16 << 15)
            // | ((shift) << 10) | (MASK_FP16_MSB15 & (m32_16 >> 6)));

            return;
        }
    }
}

// inline void cpu_pack_fp32(uint16_t xh, uint16_t xr, uint16_t* tmp)
// {
//    tmp[0] = 0;
//    tmp[1] = 0;

//    uint16_t t0 = MASK_SIGN & xh;
//    tmp[1] = (MASK_FP16_EXPONENT_FROM_FP16 & xh) >> 3;

//    uint16_t e = (uint16_t)7 & xr;
//    tmp[1] = tmp[1] | (e << 12);
//    tmp[1] = tmp[1] + BIAS_CORRECTION;
//    tmp[1] = tmp[1] | t0;

//    t0 = MASK_FP32_MANTISSA_FROM_FP16 & xh;
//    tmp[1] = tmp[1] | (t0 >> 3);
//    tmp[0] = (t0 << 13) | (xr >> 3);
// }

// inline void cpu_unpack_fp32(uint16_t* tmp, uint16_t* wh, uint16_t* wr,
// uint32_t* sflags, int idx)
// {
//    uint16_t sign = MASK_SIGN & tmp[1];
//    float* f = (float*)tmp[0];
//    *wh = 0;
//    *wh = MASK_FP32_EXPONENT_FROM_FP16 & tmp[1];

//    uint16_t e = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 7;
//    if ((e <= 142) && (e >= 113)) {

//       *wh = (*wh - BIAS_CORRECTION);
//       uint16_t t0 = MASK_FP32_EXPONENT_FROM_FP16 & (*wh);
//       *wr = 0;
//       *wr = *wr | (t0 >> 12);
//       *wh = *wh << 3;

//       t0 = MASK_FP16_MANTISSA_FROM_FP32_UPPER & tmp[1];
//       uint16_t t1 = tmp[0];
//       *wh = *wh | (t0 << 3);
//       *wh = *wh | (t1 >> 13);
//       *wh = *wh | sign;
//       *wr = *wr | (t1 << 3);
//    }
//    else {
//       *sflags = 0;
//       *sflags = *sflags + (1 << idx);

//       uint16_t t0 = MASK_FP16_EXPONENT_FROM_FP16 & tmp[1];

//       *wh = (*wh - BIAS_CORRECTION);
//       t0 = MASK_FP32_EXPONENT_FROM_FP16 & (*wh);
//       *wr = 0;
//       *wr = *wr | (t0 >> 12);
//       *wh = *wh << 3;

//       t0 = MASK_FP16_MANTISSA_FROM_FP32_UPPER & tmp[1];
//       uint16_t t1 = tmp[0];
//       *wh = *wh | (t0 << 3);
//       *wh = *wh | (t1 >> 13);
//       *wh = *wh | sign;
//       *wr = *wr | (t1 << 3);
//    }

// }

// inline void cpu_generic_unpack_fp32(uint16_t* tmp, uint16_t* wh, uint16_t*
// wr, uint32_t* sflags, int idx)
// {
//    float* f = (float*)tmp[0];
//    *wh = 0;
//    *wr = 0;

//    // *wh = MASK_FP32_EXPONENT_FROM_FP16 & tmp[1];
//    uint16_t t0 = MASK_FP32_EXPONENT_FROM_FP16 & tmp[1];

//    uint16_t exp_32 = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 7; // fp32
//    exponent fit in a fp16 variable. bool is_special_value = (exp_32 > 142) ||
//    (exp_32 < 113);

//    if (is_special_value) {
//       //
//       // *** Handles special values: overflow, underflow, inf or NaN. ***
//       //
//       // you need to handle subnormals in packfp16. They need to be handled
//       separately.

//       // Sets exponent and exponent residual.
//       // tmp[1]:
//       //
//       // | s | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3 | m4
//       | m5 | m7 |
//       // | * | *  |
//       // | s | e0 | 0  | 0  |  0 | 0  | 0  | 0  |  0 | 0  | 0  | 0  |  0 | 0
//       | 0  | 0  | uint16_t sign_and_msb = MASK_SIGN_AMD_MSB & tmp[1]; //
//       extract 2 MSB bits, sign and e0.

//       // tmp[1]:
//       //     s   e0  e1  e2  e3  e4  e5  e6  e7  m0  m1  m2  m3  m4  m5  m7
//       //   |   |   | * | * | * | * | * | * | * |   |   |   |   |   |   |   |
//       //   | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
//       //
//       //=> | 0 | 0 | 0 | 0 | 0 | e4| e5| e6| e7| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
//       *wh = MASK_FP32_EXPONENT_FROM_FP16_LSB7 & tmp[1];   // extracts 4 LSB
//       bits of exponent

//       // ignore this for a moment
//       // FIX@:
//       //    | 0 | 0 | 0 | 0 | 0 | e4| e5| e6| e7| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
//       // => | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | e4| e5| e6|
//       *wr = *wr | (*wh >> 11);                            // align unused
//       exponent bits with lsb of wr (e1-e3)

//       //   | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
//       //   | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
//       *wh = *wh & (~MASK_FP32_EXPONENT_FROM_FP16_E13);

//       // *wh:
//       //   | 0 | 0 | 0  | 0  | 0  | e4 | e5 | e6 | e7 | 0 | 0 | 0 | 0 | 0 | 0
//       | 0 |
//       //=> | 0 | 0 | e4 | e5 | e6 | e7 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 | 0
//       | 0 | *wh = *wh << 3;                                     // aligns
//       exponent

//       //   | 0 | 0  | e4 | e5 | e6 | e7 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |
//       0 | 0 |
//       //   | s | e0 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |
//       0 | 0 |
//       //=> | s | e0 | e4 | e5 | e6 | e7 | 0  | 0  | 0  | 0 | 0 | 0 | 0 | 0 |
//       0 | 0 | *wh = *wh | sign_and_msb;                           // add sign
//       and msb of exponent

//       // Set mantissa and mantissa residual.
//       // tmp[0]: | m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | m16 |
//       m17 | m18 | m19 | m20 | m21 | m22 |
//       // tmp[1]: | s  | e0 | e1 | e2  | e3  | e4  | e5  | e6  | e7  | m0  |
//       m1  | m2  | m3  | m4  | m5  | m6  |
//       //      M: | 0  | 0  | 0  | 0   | 0   | 0   | 0   | 0   | 0   | 1   | 1
//       | 1   | 1   | 1   | 1   | 1   |
//       // =>  t0: | 0  | 0  | 0  | 0   | 0   | 0   | 0   | 0   | 0   | m0  |
//       m1  | m2  | m3  | m4  | m5  | m7  |
//       //
//       t0 = MASK_FP16_MANTISSA_FROM_FP32_UPPER & tmp[1];   // extract the msb
//       part of the mantissa

//       // t1: | m10 | m11 | m12 | m13 | m14 | m15 | m16 | m17 | m18 | m19 |
//       m20 | m21 | m22 | m23 | m24 | m25 | uint16_t t1 = tmp[0]; // extract
//       the lsb part of the mantissa

//       //     *wh: | s | e0 | e4 | e5 | e6 | e7 | 0  | 0  | 0  | 0 | 0 | 0 | 0
//       | 0 | 0 | 0 |
//       // t0 << 3: | 0 | 0  | 0  | 0  | 0  | 0  | m0 | m1 | m2 | m3| m4| m5|
//       m7| 0 | 0 | 0 |
//       //  =>    : | s | e0 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3| m4| m5|
//       m7| 0 | 0 | 0 | *wh = *wh | (t0 << 3);

//       //      *wh: | s | e0 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3 | m4 | m5
//       | m6 | 0  | 0  | 0  |
//       // t1 >> 13: | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0
//       | 0  | 0  | m7 | m8 |
//       //  =>     : | s | e0 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3 | m4 | m5
//       | m6 | m7 | m8 | 0  | *wh = *wh | ((t1 >> 14) << 1);

//       //      : | s | e0 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3 | m4 | m5 |
//       m6 | m7 | m8 | 0 |
//       //      : | 0 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  |
//       0  | 0  | 0  | 1 |
//       //  =>  : | s | e0 | e4 | e5 | e6 | e7 | m0 | m1 | m2 | m3 | m4 | m5 |
//       m6 | m7 | m8 | 1 |
//       // adds 1 at the LSB because it encountered a special value
//       *wh = *wh | 1u;

//       //                              | 0   | 0   | 0   | 0   | 0   | 0   | 0
//       | 0   | 0   | 0   | 0   | 0   | 0   | e4  | e5 | e6 |
//       //                     t1 << 2: | m9  | m10 | m11 | m12 | m13 | m14 |
//       m15 | m16 | m17 | m18 | m19 | m20 | m21 | m22 |  0 | 0  |
//       // (t1 << 2) & MASK_FP16_MSB14: | m9  | m10 | m11 | m12 | m13 | m14 |
//       m15 | m16 | m17 | m18 | m19 | m20 | 0   | e4  |  e5| e6 |
//       // ^
//       // mr *wr = *wr | ((t1 << 2) & MASK_FP16_MSB12);          // one bit
//       will be lost in the wh part so we need to preserve it in wr, but in the
//       special number we keep it!
//       // mr = 0
//    }
//    else {
//       std::cout << "is not special\n";

//       // Convert to half and read the fp16 version of the algorithm.
//       __half h = __float2half_rn(*reinterpret_cast<float*>(tmp));
//       uint16_t u = hbits(h);
//       uint16_t sign = u & 0x8000u;
//       uint16_t exp = (u >> 10) & 0x1Fu;
//       uint16_t man = u & 0x03FFu;

//       // c0: clear LSB
//       uint16_t man0 = (uint16_t)(man & ~1u);
//       uint16_t exp0 = exp;

//       // c1: next 2-ULP step upward
//       uint16_t man1 = (uint16_t)(man0 + 2u);
//       uint16_t exp1 = exp;
//       if (man1 > 0x03FFu) {
//          // this requires precisise handling
//          man1 = 0;
//          if (exp1 < 0x1Eu) {
//             exp1 += 1;
//          }
//          else {
//             *wh = hmake((uint16_t)(sign | 0x7C00u));
//          } // Inf
//       }

//       uint16_t u0 = (uint16_t)(sign | (exp0 << 10) | man0);
//       uint16_t u1 = (uint16_t)(sign | (exp1 << 10) | man1);
//       D16 D;

//       __half c0 = hmake(u0);
//       __half c1 = hmake(u1);

//       float f0 = __half2float(c0);
//       float f1 = __half2float(c1);
//       float d0 = fabsf(*tmp - f0);
//       float d1 = fabsf(*tmp - f1);

//       // Assigns rounded number. Both lsb are 0
//       if (d1 < d0) {
//          *wh = *reinterpret_cast<uint16_t*>(&c1);
//          *wr = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 11;
//          *wr = *wr | ((tmp[0] << 2) & MASK_FP16_MSB12);          // one bit
//          will be lost in the wh part so we need to preserve it in wr, but in
//          the special number we keep it! *wr = *wr | 1u << 3; return;
//       }
//       else if (d0 < d1) {
//          *wh = *reinterpret_cast<uint16_t*>(&c0);
//          *wr = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 11;
//          *wr = *wr | ((tmp[0] << 2) & MASK_FP16_MSB12);          // one bit
//          will be lost in the wh part so we need to preserve it in wr, but in
//          the special number we keep it! return;
//       }
//       // Tie: choose even at bit1 (mantissa>>1 even)
//       bool c0_even_bit1 = (((u0 & 0x03FFu) >> 1) & 1u) == 0u;
//       *wh = c0_even_bit1 ? *reinterpret_cast<uint16_t*>(&c0) :
//       *reinterpret_cast<uint16_t*>(&c1);                               // you
//       need to modify the last bit of c1 and c0 +1 to one +0 to the other

//       *wr = (MASK_FP32_EXPONENT_FROM_FP16 & tmp[1]) >> 11;
//       *wr = *wr | ((tmp[0] << 2) & MASK_FP16_MSB12);              // one bit
//       will be lost in the wh part so we need to preserve it in wr, but in the
//       special number we keep it!
//       // *wr = *wr | c0_even_bit1 ? 0 : (1u << 3);
//       if (!c0_even_bit1) {
//          *wr = *wr | (1u << 3);
//       }
//    }
// }

void unpack_from_values_32_16_kernel(size_t num_elems, uint32_t *values,
                                     uint16_t *wh, uint16_t *wr,
                                     uint32_t *sflags) {
    uint32_t sflags_tmp;
    float *v = reinterpret_cast<float *>(values);
    for (size_t i = 0; i < num_elems; ++i) {
        if ((i < num_elems)) {
            __half *h = (half *)&wh[i];
            uint16_t *tmp = ((uint16_t *)&values[i]);
            generic_unpack_fp32(tmp, wh + i, wr + i, &sflags_tmp, i % 32);

            D32 A;
            D16 C;
            D32 B;
            A.u = values[i];
            float out;
            generic_pack_fp32(wh[i], wr[i], reinterpret_cast<uint16_t *>(&out));
            float diff =
                ((std::abs(out - reinterpret_cast<float *>(values)[i])) /
                 std::abs(reinterpret_cast<float *>(values)[i]));
            if (((std::abs(out - reinterpret_cast<float *>(values)[i])) /
                     std::abs(reinterpret_cast<float *>(values)[i]) >
                 1e-7)) {
                // debug
                std::cout << "\n\n";
                std::cout << "i: " << i << '\n';
                std::cout << "diff: "
                          << ((std::abs(out -
                                        reinterpret_cast<float *>(values)[i])) /
                              std::abs(reinterpret_cast<float *>(values)[i]))
                          << '\n';
                std::cout << "  in: " << reinterpret_cast<float *>(values)[i]
                          << ", out: " << out << '\n';
                std::cout << "  before: ";
                A.print_binary_cuda();
                B.s = out;
                std::cout << "   after: ";
                B.print_binary_cuda();
                std::cout << "    half: ";
                C.u = wh[i];
                C.print_binary_cuda();
                std::cout << "      xr: ";
                C.u = wr[i];
                C.print_binary_cuda();
                std::cout << "\n\n";
            }

            if (i % 32 == 0) {
                sflags[i / 32] = 0;
            } else {
                sflags[i / 32] = sflags[i / 32] | sflags_tmp;
            }
        }
    }

}  // end of unpack_from_values_kernel_32_16

}  // namespace tomos

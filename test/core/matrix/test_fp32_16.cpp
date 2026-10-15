#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "fp32_16_pack.hpp"
#include "matrix.hpp"

#if defined(USE_CUDA)
#include <cuda_runtime.h>

#include "fp32_16_cuda_kernels.cuh"
#endif

namespace tomos {
namespace test {

namespace {

// Values drawn from a modest range stay within fp32_16_split's lossless
// "hot path" (see fp32_16_pack.hpp), so the accurate path can be checked
// against a tight tolerance and the fast/pure_half paths against generous,
// purely fp16-precision-driven ones.
std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

// Mixes in occasional far-outside-[-8,8] magnitudes so fp32_16_split's
// "extended" (and occasionally "saturate") encoding branches actually get
// exercised -- see fp32_16_pack.hpp's own header comment for the
// two-tier-plus-saturation contract. random_values above stays entirely
// in the "normal" branch, so every check built on it (including the
// large-shape GEMV checks below) validates only that one branch, on real
// production kernels, at real production scale. A preconditioned Krylov
// vector in an actual ill-conditioned solve can plausibly span many more
// orders of magnitude than a uniform +-8 draw ever would.
std::vector<float> random_values_extreme(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> normal_dist(-8.0f, 8.0f);
    // Capped at 1e6, safely inside fp32_16_pack.hpp's "extended" branch
    // (finite up to ~5.4e8) -- beyond that, values fall into "saturate",
    // which correctly reconstructs to exactly signed Inf for magnitudes
    // beyond the extended tier's own ceiling (see Fp32_16PackTest.
    // ExactZeroAndInfPreserved). A wider range here doesn't test a bug --
    // it deliberately manufactures Infs by design, which is why an
    // earlier, wider version of this range (1e-9..1e9) produced NaN
    // results that were never actually a kernel defect.
    std::uniform_real_distribution<float> extreme_mag_dist(1.0e-6f, 1.0e6f);
    std::uniform_int_distribution<int> extreme_flag(0, 9);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    std::vector<float> v(n);
    for (auto& x : v) {
        if (extreme_flag(gen) == 0) {
            float mag = extreme_mag_dist(gen);
            x = sign_dist(gen) ? mag : -mag;
        } else {
            x = normal_dist(gen);
        }
    }
    return v;
}

// Log-uniform magnitude in [1e-15, 1e-5] -- specifically targets the
// UNDERFLOW/near-zero regime the real production matrix's own smallest
// entries live in (down to ~1.4e-12, see the matrix magnitude scan
// earlier in this investigation). random_values_extreme above samples
// LINEARLY uniform over a huge range (1e-6..1e6), which for a range that
// wide almost never actually lands anywhere near the small end -- it
// nominally "covers" underflow but essentially never exercises it. This
// generator samples the EXPONENT uniformly instead, so it actually
// reaches fp32_16_split's "extended" underflow branch and, deeper still,
// values below the extended tier's own floor (~3.7e-9) that the
// "saturate" branch correctly rounds to exactly zero.
std::vector<float> random_values_small_log_uniform(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> log_mag_dist(-15.0f, -5.0f);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    std::vector<float> v(n);
    for (auto& x : v) {
        float mag = std::pow(10.0f, log_mag_dist(gen));
        x = sign_dist(gen) ? mag : -mag;
    }
    return v;
}

void fill_dense_cpu(Dense& A, const std::vector<float>& col_major_values) {
    auto view = as_strided_view<float>(A);
    for (size_t j = 0; j < view.size_[1]; ++j)
        for (size_t i = 0; i < view.size_[0]; ++i)
            view.values_[i + j * view.lead_dim_] =
                col_major_values[i + j * view.size_[0]];
}

std::vector<float> collect_dense_cpu(Dense& A) {
    auto view = as_strided_view<float>(A);
    std::vector<float> out(view.size_[0] * view.size_[1]);
    for (size_t j = 0; j < view.size_[1]; ++j)
        for (size_t i = 0; i < view.size_[0]; ++i)
            out[i + j * view.size_[0]] = view.values_[i + j * view.lead_dim_];
    return out;
}

using FillFn = void (*)(Dense&, const std::vector<float>&);
using CollectFn = std::vector<float> (*)(Dense&);

const char* trans_name(TransposeType t) {
    return t == NO_TRANSPOSE ? "NO_TRANSPOSE" : "TRANSPOSE";
}

// Runs one (transpose, beta) combination of A*x through all three fp32_16
// GEMV tiers and checks each against a plain double-precision reference. A
// is square so x/y keep the same length regardless of transpose.
void run_mv_combination(Context& ctx, Device dev, FillFn fill,
                        CollectFn collect, TransposeType trans, double beta) {
    SCOPED_TRACE(std::string("mv trans=") + trans_name(trans) +
                " beta=" + std::to_string(beta));
    dim<2> size = {5, 5};
    auto a_values = random_values(size[0] * size[1], 101);
    auto x_values = random_values(size[1], 102);
    auto y_init = random_values(size[0], 103);

    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill(A, a_values);
    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(A, packed);

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {size[1], 1});
    fill(x, x_values);

    std::vector<double> y_ref(size[0]);
    for (size_t i = 0; i < size[0]; ++i)
        y_ref[i] = beta * static_cast<double>(y_init[i]);
    for (size_t i = 0; i < size[0]; ++i) {
        for (size_t j = 0; j < size[1]; ++j) {
            size_t a_row = (trans == NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == NO_TRANSPOSE) ? j : i;
            y_ref[i] += static_cast<double>(a_values[a_row + a_col * size[0]]) *
                        static_cast<double>(x_values[j]);
        }
    }

    auto make_y = [&]() {
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP32, {size[0], 1});
        fill(y, y_init);
        return y;
    };

    Dense y_pure_half = make_y();
    mv_apply_fp32_16_pure_half(dev, trans, 1.0, packed, x, beta, y_pure_half);
    auto pure_half_out = collect(y_pure_half);

    Dense y_fast = make_y();
    mv_apply_fp32_16_fast(dev, trans, 1.0, packed, x, beta, y_fast);
    auto fast_out = collect(y_fast);

    Dense y_ext = make_y();
    mv_apply_fp32_16_ext(dev, trans, 1.0, packed, x, beta, y_ext);
    auto ext_out = collect(y_ext);

    for (size_t i = 0; i < size[0]; ++i) {
        // pure_half additionally accumulates in __half (not just per-element
        // decode error), so it gets a slightly looser bound than fast.
        EXPECT_NEAR(pure_half_out[i], y_ref[i], 0.15 * std::abs(y_ref[i]) + 0.15)
            << "i=" << i;
        EXPECT_NEAR(fast_out[i], y_ref[i], 0.05 * std::abs(y_ref[i]) + 0.05)
            << "i=" << i;
        EXPECT_NEAR(ext_out[i], y_ref[i], 1e-3 * std::abs(y_ref[i]) + 1e-4)
            << "i=" << i;
    }
}

// Same as run_mv_combination, but for C = alpha*op(A)*op(B) + beta*C. A and
// B are both square so transA/transB don't change any operand's dimensions.
void run_mm_combination(Context& ctx, Device dev, FillFn fill,
                        CollectFn collect, TransposeType transA,
                        TransposeType transB, double beta) {
    SCOPED_TRACE(std::string("mm transA=") + trans_name(transA) +
                " transB=" + trans_name(transB) + " beta=" +
                std::to_string(beta));
    dim<2> size_A = {4, 4};
    dim<2> size_B = {4, 4};
    size_t m = 4, k = 4, n = 4;
    auto a_values = random_values(size_A[0] * size_A[1], 201);
    auto b_values = random_values(size_B[0] * size_B[1], 202);
    auto c_init = random_values(m * n, 203);

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    size_A);
    fill(A, a_values);
    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size_A);
    split_fp32_16(A, packed);

    Dense B = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    size_B);
    fill(B, b_values);

    std::vector<double> c_ref(m * n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            c_ref[i + j * m] = beta * static_cast<double>(c_init[i + j * m]);
    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            double b_val = (transB == NO_TRANSPOSE)
                              ? static_cast<double>(b_values[l + j * size_B[0]])
                              : static_cast<double>(b_values[j + l * size_B[0]]);
            for (size_t i = 0; i < m; ++i) {
                size_t a_row = (transA == NO_TRANSPOSE) ? i : l;
                size_t a_col = (transA == NO_TRANSPOSE) ? l : i;
                c_ref[i + j * m] +=
                    static_cast<double>(a_values[a_row + a_col * size_A[0]]) *
                    b_val;
            }
        }
    }

    auto make_c = [&]() {
        Dense C = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP32, {m, n});
        fill(C, c_init);
        return C;
    };

    Dense C_pure_half = make_c();
    mm_apply_fp32_16_pure_half(dev, transA, transB, 1.0, packed, B, beta,
                              C_pure_half);
    auto pure_half_out = collect(C_pure_half);

    Dense C_fast = make_c();
    mm_apply_fp32_16_fast(dev, transA, transB, 1.0, packed, B, beta, C_fast);
    auto fast_out = collect(C_fast);

    Dense C_ext = make_c();
    mm_apply_fp32_16_ext(dev, transA, transB, 1.0, packed, B, beta, C_ext);
    auto ext_out = collect(C_ext);

    for (size_t i = 0; i < c_ref.size(); ++i) {
        EXPECT_NEAR(pure_half_out[i], c_ref[i], 0.15 * std::abs(c_ref[i]) + 0.15)
            << "i=" << i;
        EXPECT_NEAR(fast_out[i], c_ref[i], 0.05 * std::abs(c_ref[i]) + 0.05)
            << "i=" << i;
        EXPECT_NEAR(ext_out[i], c_ref[i], 1e-3 * std::abs(c_ref[i]) + 1e-4)
            << "i=" << i;
    }
}

}  // namespace

class Fp32_16CpuTest : public ::testing::Test {
protected:
    Context ctx{Device::CPU};
};

uint32_t bits_of(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// Exhaustively sweeps every possible fp32 exponent (0..255) with several
// random mantissas/signs each, verifying fp32_16_split/fp32_16_combine's
// documented two-tier-plus-saturation contract (see fp32_16_pack.hpp):
//   - Normal (control=0, fp32 exponent in [113,142]): bit-exact.
//   - Extended (control=1, offset in [1,14]): still bit-exact -- same full
//     9+14 mantissa split, only the exponent field's role differs.
//   - Saturate (control=1, offset=15): reconstructs to exactly signed Inf
//     (overflow direction) or signed 0 (underflow direction) -- covers
//     deeper overflow/underflow, fp32 subnormals, exact zero, inf, and nan
//     (which saturates to Inf, payload dropped, like any other overflow).
TEST(Fp32_16PackTest, ExactWithinNormalAndExtendedRangeSaturateBeyond) {
    std::mt19937 gen(7);
    std::uniform_int_distribution<uint32_t> mant_dist(0, 0x7FFFFF);
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);

    for (uint32_t e32 = 0; e32 <= 255; ++e32) {
        for (int trial = 0; trial < 40; ++trial) {
            uint32_t mant = (e32 == 0 && trial == 0) ? 0 : mant_dist(gen);
            uint32_t sign = sign_dist(gen);
            uint32_t bits = (sign << 31) | (e32 << 23) | mant;
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            if (std::isnan(v)) continue;

            uint16_t wh, wr;
            fp32_16_split(v, wh, wr);
            float rec = fp32_16_combine(wh, wr);

            bool control = wh & 1u;
            uint32_t offset = (wh >> 10) & 0xFu;
            bool should_be_exact = !control || (offset != 15u);
            if (should_be_exact) {
                ASSERT_EQ(bits_of(rec), bits) << "e32=" << e32 << " mant=" << mant
                                              << " sign=" << sign
                                              << " control=" << control
                                              << " offset=" << offset;
            } else {
                uint32_t direction = (wh >> 14) & 0x1u;
                float expected = direction ? std::numeric_limits<float>::infinity() : 0.0f;
                if (sign) expected = -expected;
                ASSERT_EQ(bits_of(rec), bits_of(expected))
                    << "e32=" << e32 << " mant=" << mant << " sign=" << sign
                    << " direction=" << direction;
            }
        }
    }
}

// fp32_16_combine_branchless_nohw's saturate path is a mask-selected
// constant (see that function's own comment in fp32_16_pack.hpp), so it
// needs its own dedicated verification rather than just trusting the
// derivation: this reuses the SAME exponent sweep as
// ExactWithinNormalAndExtendedRangeSaturateBeyond above (which
// independently established fp32_16_combine's own bit-exact/saturate
// correctness against a direct memcpy'd ground truth) and checks
// fp32_16_combine_branchless_nohw against THAT already-validated result,
// bit-for-bit, for every (wh, wr) the sweep produces. NaN is excluded
// from this sweep (see the loop above) since bits_of() equality isn't a
// meaningful check for a value that itself now saturates to signed Inf --
// covered separately below.
TEST(Fp32_16PackTest, CombineBranchlessNohwMatchesCombine) {
    std::mt19937 gen(7);
    std::uniform_int_distribution<uint32_t> mant_dist(0, 0x7FFFFF);
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);

    for (uint32_t e32 = 0; e32 <= 255; ++e32) {
        for (int trial = 0; trial < 40; ++trial) {
            uint32_t mant = (e32 == 0 && trial == 0) ? 0 : mant_dist(gen);
            uint32_t sign = sign_dist(gen);
            uint32_t bits = (sign << 31) | (e32 << 23) | mant;
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            if (std::isnan(v)) continue;

            uint16_t wh, wr;
            fp32_16_split(v, wh, wr);
            float ref = fp32_16_combine(wh, wr);
            float nohw = fp32_16_combine_branchless_nohw(wh, wr);
            ASSERT_EQ(bits_of(nohw), bits_of(ref))
                << "e32=" << e32 << " mant=" << mant << " sign=" << sign
                << " wh=" << wh << " wr=" << wr;
        }
    }
}

// Explicit saturate-bucket edge cases the exponent sweep above either
// skips (NaN) or only reaches probabilistically -- constructs fp32 values
// specifically chosen to land in fp32_16_split's saturate bucket
// (exponent far outside the normal+extended union, see fp32_16_pack.hpp's
// own header comment) and drive fp32_16_combine_branchless_nohw's
// saturate branch deliberately, not just incidentally.
TEST(Fp32_16PackTest, CombineBranchlessNohwSaturateEdgeCases) {
    // Deep underflow (flushes to signed zero) and deep overflow (saturates
    // to signed infinity) -- both well beyond the extended tier's own
    // 14-step reach (~3.7e-9 floor, ~5.4e8 ceiling).
    for (float v : {std::ldexp(1.0f, -80), -std::ldexp(1.0f, -80),
                     std::ldexp(1.0f, 80), -std::ldexp(1.0f, 80)}) {
        uint16_t wh, wr;
        fp32_16_split(v, wh, wr);
        float ref = fp32_16_combine(wh, wr);
        float nohw = fp32_16_combine_branchless_nohw(wh, wr);
        ASSERT_EQ(bits_of(nohw), bits_of(ref)) << "v=" << v;
    }

    // NaN saturates to signed Inf now (see fp32_16_pack.hpp's own header
    // comment) -- check that explicitly, for both the CPU reference path
    // and this function, rather than bits_of() equality (not meaningful
    // for genuine NaN payloads, but this value isn't one anymore).
    {
        uint16_t wh, wr;
        fp32_16_split(std::numeric_limits<float>::quiet_NaN(), wh, wr);
        float ref = fp32_16_combine(wh, wr);
        float nohw = fp32_16_combine_branchless_nohw(wh, wr);
        ASSERT_TRUE(std::isinf(ref) && ref > 0.0f);
        ASSERT_EQ(bits_of(nohw), bits_of(ref));
    }
}

TEST(Fp32_16PackTest, ExactZeroAndInfPreserved) {
    uint16_t wh, wr;
    float rec;

    fp32_16_split(0.0f, wh, wr);
    rec = fp32_16_combine(wh, wr);
    EXPECT_EQ(rec, 0.0f);
    EXPECT_FALSE(std::signbit(rec));

    fp32_16_split(-0.0f, wh, wr);
    rec = fp32_16_combine(wh, wr);
    EXPECT_EQ(rec, 0.0f);
    EXPECT_TRUE(std::signbit(rec));

    fp32_16_split(std::numeric_limits<float>::infinity(), wh, wr);
    rec = fp32_16_combine(wh, wr);
    EXPECT_TRUE(std::isinf(rec) && rec > 0);

    fp32_16_split(-std::numeric_limits<float>::infinity(), wh, wr);
    rec = fp32_16_combine(wh, wr);
    EXPECT_TRUE(std::isinf(rec) && rec < 0);

    // NaN saturates to signed Inf now (the saturate marker has no spare
    // bits to preserve NaN-ness -- see fp32_16_pack.hpp's own header
    // comment) -- an explicit, documented behavior, not an accidental gap.
    fp32_16_split(std::numeric_limits<float>::quiet_NaN(), wh, wr);
    rec = fp32_16_combine(wh, wr);
    EXPECT_TRUE(std::isinf(rec) && rec > 0);
}

uint16_t half_bits_of(__half h) {
    return *reinterpret_cast<const uint16_t*>(&h);
}

// fp32_16_half_only's underflow branch reconstructs a genuine fp16
// subnormal by shifting mant_hi's implicit-1-prefixed significand (see
// fp32_16_pack.hpp's own header comment) -- bit-exact against a direct
// static_cast<__half>(value) reference for every magnitude real fp16
// hardware can still represent as a subnormal (offset 1..10, fp32
// exponent field 112..103), and correctly flushed to signed zero once
// offset exceeds that (11..15, including the saturate marker) -- exactly
// what real fp16 hardware does once a value underflows even its own
// smallest subnormal (2^-24). Overflow direction always decodes to
// signed Inf regardless of the specific offset value stored, since even
// offset=1 there is already beyond fp16's own representable range.
TEST(Fp32_16PackTest, HalfOnlyReconstructsRealFp16SubnormalsExactly) {
    std::mt19937 gen(11);
    std::uniform_int_distribution<uint32_t> mant_dist(0, 0x7FFFFF);
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);

    for (uint32_t offset = 1; offset <= 10; ++offset) {
        uint32_t e32 = fp32_16_detail::kFp32ExpMin - offset;
        for (int trial = 0; trial < 20; ++trial) {
            uint32_t mant = mant_dist(gen);
            uint32_t sign = sign_dist(gen);
            uint32_t bits = (sign << 31) | (e32 << 23) | mant;
            float v;
            std::memcpy(&v, &bits, sizeof(v));

            uint16_t wh, wr;
            fp32_16_split(v, wh, wr);
            __half ref = static_cast<__half>(v);
            ASSERT_EQ(half_bits_of(fp32_16_half_only(wh)), half_bits_of(ref))
                << "offset=" << offset << " v=" << v;
        }
    }

    for (uint32_t offset = 11; offset <= 15; ++offset) {
        uint32_t e32 = (offset <= 14) ? (fp32_16_detail::kFp32ExpMin - offset) : 40u;
        for (int trial = 0; trial < 10; ++trial) {
            uint32_t mant = mant_dist(gen);
            uint32_t sign = sign_dist(gen);
            uint32_t bits = (sign << 31) | (e32 << 23) | mant;
            float v;
            std::memcpy(&v, &bits, sizeof(v));

            uint16_t wh, wr;
            fp32_16_split(v, wh, wr);
            uint16_t expected_bits = static_cast<uint16_t>(sign << 15);
            ASSERT_EQ(half_bits_of(fp32_16_half_only(wh)), expected_bits)
                << "offset=" << offset << " v=" << v;
        }
    }

    for (uint32_t offset = 1; offset <= 15; ++offset) {
        uint32_t e32 = (offset <= 14) ? (fp32_16_detail::kFp32ExpMax + offset) : 220u;
        for (int trial = 0; trial < 10; ++trial) {
            uint32_t mant = mant_dist(gen);
            uint32_t sign = sign_dist(gen);
            uint32_t bits = (sign << 31) | (e32 << 23) | mant;
            float v;
            std::memcpy(&v, &bits, sizeof(v));

            uint16_t wh, wr;
            fp32_16_split(v, wh, wr);
            uint16_t expected_bits = static_cast<uint16_t>((sign << 15) | 0x7C00u);
            ASSERT_EQ(half_bits_of(fp32_16_half_only(wh)), expected_bits)
                << "offset=" << offset << " v=" << v;
        }
    }
}

// fp32_16_half_only_fast is the cheaper sibling that skips the shift-based
// subnormal reconstruction entirely: unlike fp32_16_half_only above, it
// flushes EVERY control=1 value straight to signed Inf/0 (overflow/
// underflow), including the offset 1..10 band that fp32_16_half_only can
// still recover as a genuine fp16 subnormal -- an intentional accuracy-
// for-speed approximation, asserted explicitly here (not bit-exact vs.
// real half for that band, by design).
TEST(Fp32_16PackTest, HalfOnlyFastAlwaysSaturates) {
    // control=0 (normal): identical to fp32_16_half_only, direct reinterpret.
    {
        uint16_t wh, wr;
        fp32_16_split(3.25f, wh, wr);
        ASSERT_EQ(half_bits_of(fp32_16_half_only_fast(wh)), half_bits_of(fp32_16_half_only(wh)));
    }

    // offset=1 underflow (2^-15): fp32_16_half_only recovers a genuine
    // subnormal (nonzero); fp32_16_half_only_fast flushes it to zero.
    {
        uint16_t wh, wr;
        fp32_16_split(std::ldexp(1.0f, -15), wh, wr);
        ASSERT_EQ(half_bits_of(fp32_16_half_only_fast(wh)), static_cast<uint16_t>(0u));
        ASSERT_NE(half_bits_of(fp32_16_half_only(wh)), static_cast<uint16_t>(0u));
    }

    // Any overflow-direction value: both functions agree (always Inf).
    {
        uint16_t wh, wr;
        fp32_16_split(std::ldexp(1.0f, 40), wh, wr);
        ASSERT_EQ(half_bits_of(fp32_16_half_only_fast(wh)), static_cast<uint16_t>(0x7C00u));
        ASSERT_EQ(half_bits_of(fp32_16_half_only(wh)), static_cast<uint16_t>(0x7C00u));
    }
}

// Saturation boundary, checked consistently across every decoder: a value
// just inside the extended tier's own reach (offset=14, still exact) vs.
// just beyond it (needs offset=15, saturates) -- see fp32_16_pack.hpp's
// own header comment for the exact magnitudes (~3.7e-9 floor, ~5.4e8
// ceiling).
TEST(Fp32_16PackTest, SaturationBoundaryAcrossAllDecoders) {
    const float kJustInsideUnderflow = std::ldexp(1.0f, -14 - 14);   // offset=14, exact
    const float kJustOutsideUnderflow = std::ldexp(1.0f, -14 - 15);  // offset=15, saturates
    const float kJustInsideOverflow = std::ldexp(1.0f, 15 + 14);     // offset=14, exact
    const float kJustOutsideOverflow = std::ldexp(1.0f, 15 + 15);    // offset=15, saturates

    for (float v : {kJustInsideUnderflow, kJustOutsideUnderflow, kJustInsideOverflow,
                     kJustOutsideOverflow}) {
        uint16_t wh, wr;
        fp32_16_split(v, wh, wr);
        float combine_out = fp32_16_combine(wh, wr);
        float nohw_out = fp32_16_combine_branchless_nohw(wh, wr);
        ASSERT_EQ(bits_of(combine_out), bits_of(nohw_out)) << "v=" << v;

        bool expect_saturated = (v == kJustOutsideUnderflow) || (v == kJustOutsideOverflow);
        if (expect_saturated) {
            bool is_underflow = (v == kJustOutsideUnderflow);
            ASSERT_EQ(combine_out, is_underflow ? 0.0f : std::numeric_limits<float>::infinity())
                << "v=" << v;
        } else {
            ASSERT_FLOAT_EQ(combine_out, v) << "v=" << v;
        }
    }

    // bf16_only agrees with the same saturate/exact split (wh-only, so it
    // can't be exact the way fp32_16_combine is -- only the saturate-vs-
    // exact BOUNDARY behavior is being checked here, not bit-exactness).
    {
        uint16_t wh, wr;
        fp32_16_split(kJustOutsideUnderflow, wh, wr);
        __nv_bfloat16 bf = fp32_16_bf16_only(wh);
        ASSERT_EQ(__bfloat162float(bf), 0.0f);

        fp32_16_split(kJustOutsideOverflow, wh, wr);
        bf = fp32_16_bf16_only(wh);
        ASSERT_TRUE(std::isinf(__bfloat162float(bf)));
    }
}

TEST_F(Fp32_16CpuTest, SplitCombineRoundTrip) {
    dim<2> size = {20, 15};
    auto values = random_values(size[0] * size[1], 1);

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cpu(source, values);

    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    Dense reconstructed =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    combine_fp32_16(packed, reconstructed);

    auto out = collect_dense_cpu(reconstructed);
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], values[i]);
}

// Covers every (tier x transpose x beta) combination for GEMV: 3 tiers x 2
// transpose settings x 2 beta settings (beta=0 exercises the "y need not be
// read" path, beta!=0 exercises the accumulate-into-existing-y path).
TEST_F(Fp32_16CpuTest, MvAllTierTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_combination(ctx, Device::CPU, fill_dense_cpu,
                              collect_dense_cpu, trans, beta);
}

// Covers every (tier x transA x transB x beta) combination for GEMM: 3
// tiers x 2 x 2 x 2 = 24 checks.
TEST_F(Fp32_16CpuTest, MmAllTierTransposeAndBetaCombinations) {
    for (TransposeType transA : {NO_TRANSPOSE, TRANSPOSE})
        for (TransposeType transB : {NO_TRANSPOSE, TRANSPOSE})
            for (double beta : {0.0, 0.7})
                run_mm_combination(ctx, Device::CPU, fill_dense_cpu,
                                  collect_dense_cpu, transA, transB, beta);
}

#if defined(USE_CUDA)

// Same problems as the CPU tests above, resident on Device::CUDA. Only the
// plain FP32 operands (A's source before splitting, x, B) need a host->device
// copy; the FP32_16 (wh/wr) packed matrix is produced by split_fp32_16
// running entirely on-device, mirroring how test_lsqr.cpp's CUDA tests move
// data (Dense::copy_from has no registered CPU<->CUDA path).

namespace {

void fill_dense_cuda(Dense& A, const std::vector<float>& col_major_values) {
    auto view = as_strided_view<float>(A);
    std::vector<float> padded(view.lead_dim_ * view.size_[1], 0.0f);
    for (size_t j = 0; j < view.size_[1]; ++j)
        for (size_t i = 0; i < view.size_[0]; ++i)
            padded[i + j * view.lead_dim_] =
                col_major_values[i + j * view.size_[0]];
    cudaMemcpy(view.values_, padded.data(), padded.size() * sizeof(float),
              cudaMemcpyHostToDevice);
}

std::vector<float> collect_dense_cuda(Dense& A) {
    auto view = as_strided_view<float>(A);
    std::vector<float> padded(view.lead_dim_ * view.size_[1]);
    cudaMemcpy(padded.data(), view.values_, padded.size() * sizeof(float),
              cudaMemcpyDeviceToHost);
    std::vector<float> out(view.size_[0] * view.size_[1]);
    for (size_t j = 0; j < view.size_[1]; ++j)
        for (size_t i = 0; i < view.size_[0]; ++i)
            out[i + j * view.size_[0]] = padded[i + j * view.lead_dim_];
    return out;
}

}  // namespace

class Fp32_16CudaTest : public ::testing::Test {
protected:
    Context ctx{Device::CUDA};
};

TEST_F(Fp32_16CudaTest, SplitCombineRoundTrip) {
    dim<2> size = {20, 15};
    auto values = random_values(size[0] * size[1], 11);

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cuda(source, values);

    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    Dense reconstructed =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    combine_fp32_16(packed, reconstructed);

    auto out = collect_dense_cuda(reconstructed);
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], values[i]);
}

// SplitCombineRoundTrip above only ever uses random_values (+-8, entirely
// within fp32_16_split's "normal" branch) -- it has never actually
// exercised the ON-DEVICE split_dense_fp32_16_kernel with anything that
// reaches the "extended" or "saturate" buckets. The isolation tests above
// (MvLargeShapeExtremeMatrixOnly vs ...VectorOnly) show ext_l2 goes NaN
// specifically when A's OWN values are extreme -- this reproduces that at
// the split/combine level directly, comparing the ON-DEVICE round-trip
// against a HOST-computed reference using the SAME (validated,
// exhaustively-swept-on-host) fp32_16_split/fp32_16_combine, rather than
// against the original value (the "saturate" bucket reconstructs to
// exactly signed 0/Inf, not the original value, so comparing to the
// original value directly would conflate that documented, intentional
// saturation with an actual device/host divergence bug -- comparing
// device output to a host reference computed via the identical functions
// isolates the latter).
TEST_F(Fp32_16CudaTest, SplitCombineRoundTripExtremeMagnitude) {
    dim<2> size = {20, 15};
    auto values = random_values_extreme(size[0] * size[1], 21);

    std::vector<float> host_ref(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint16_t wh, wr;
        fp32_16_split(values[i], wh, wr);
        host_ref[i] = fp32_16_combine(wh, wr);
    }

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cuda(source, values);

    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    Dense reconstructed =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    combine_fp32_16(packed, reconstructed);

    auto out = collect_dense_cuda(reconstructed);
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_TRUE(bits_of(out[i]) == bits_of(host_ref[i]) ||
                   (std::isnan(out[i]) && std::isnan(host_ref[i])))
            << "i=" << i << " input=" << values[i] << " device_out=" << out[i]
            << " host_ref=" << host_ref[i];
    }
}

// Same as SplitCombineRoundTripExtremeMagnitude above, but at the ACTUAL
// shape (140000 x 100) MvLargeShapeExtremeMatrixOnlyTransposeAndBeta
// Combinations fails at, instead of a tiny 20x15 -- the packed buffer's
// wr half lives at wh + num_elems (see as_fp32_16_storage,
// fp32_16_pack.hpp), so a bug specific to a large element count would
// only show up at this scale, not the small one above.
TEST_F(Fp32_16CudaTest, SplitCombineRoundTripExtremeMagnitudeLargeShape) {
    dim<2> size = {140000, 100};
    auto values = random_values_extreme(size[0] * size[1], 41);

    std::vector<float> host_ref(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint16_t wh, wr;
        fp32_16_split(values[i], wh, wr);
        host_ref[i] = fp32_16_combine(wh, wr);
    }

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cuda(source, values);

    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    Dense reconstructed =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    combine_fp32_16(packed, reconstructed);

    auto out = collect_dense_cuda(reconstructed);
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_TRUE(bits_of(out[i]) == bits_of(host_ref[i]) ||
                   (std::isnan(out[i]) && std::isnan(host_ref[i])))
            << "i=" << i << " input=" << values[i] << " device_out=" << out[i]
            << " host_ref=" << host_ref[i];
    }
}

// Directly targets fp32_16_combine_branchless (via combine_branchless_
// probe, see its own comment) -- the SEPARATE decode function the real
// GEMV kernels actually call, as opposed to fp32_16_combine (validated by
// SplitCombineRoundTripExtremeMagnitude above via combine_dense_fp32_16,
// a DIFFERENT call path). If this fails where the one above passes, the
// bug is isolated to fp32_16_combine_branchless itself.
TEST_F(Fp32_16CudaTest, CombineBranchlessDeviceMatchesHostExtremeMagnitude) {
    size_t n = 20 * 15;
    auto values = random_values_extreme(n, 31);

    std::vector<uint16_t> wh(n), wr(n);
    std::vector<float> host_ref(n);
    for (size_t i = 0; i < n; ++i) {
        fp32_16_split(values[i], wh[i], wr[i]);
        host_ref[i] = fp32_16_combine(wh[i], wr[i]);
    }

    uint16_t *d_wh = nullptr, *d_wr = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_wh, n * sizeof(uint16_t));
    cudaMalloc(&d_wr, n * sizeof(uint16_t));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_wh, wh.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wr, wr.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cuda::combine_branchless_probe(n, d_wh, d_wr, d_out);

    std::vector<float> out(n);
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_wh);
    cudaFree(d_wr);
    cudaFree(d_out);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(bits_of(out[i]) == bits_of(host_ref[i]) ||
                   (std::isnan(out[i]) && std::isnan(host_ref[i])))
            << "i=" << i << " input=" << values[i] << " wh=" << wh[i]
            << " wr=" << wr[i] << " branchless_out=" << out[i]
            << " combine_host_ref=" << host_ref[i];
    }
}

// Same as CombineBranchlessDeviceMatchesHostExtremeMagnitude above, but
// reading wh/wr from a REAL packed Dense's own buffer (via
// as_fp32_16_storage, exactly what mv_accurate_impl does -- wr lives at
// wh + num_elems inside ONE shared allocation, not two independent
// arrays) at the ACTUAL failing shape (140000 x 100, 14M elements), not
// the small, separately-allocated arrays and tiny element count the
// version above used. This is the one remaining untested combination:
// every other piece (split, fp32_16_combine, fp32_16_combine_branchless
// on independent small arrays) has already checked out fine.
TEST_F(Fp32_16CudaTest, CombineBranchlessPackedBufferLargeShapeExtremeMagnitude) {
    dim<2> size = {140000, 100};
    size_t n = size[0] * size[1];
    auto values = random_values_extreme(n, 51);

    std::vector<float> host_ref(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t wh, wr;
        fp32_16_split(values[i], wh, wr);
        host_ref[i] = fp32_16_combine(wh, wr);
    }

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cuda(source, values);
    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    auto vPacked = as_strided_view<uint32_t>(packed);
    Fp32_16Storage s = as_fp32_16_storage(vPacked.values_, n);

    float* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));
    cuda::combine_branchless_probe(n, s.wh, s.wr, d_out);

    std::vector<float> out(n);
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(bits_of(out[i]) == bits_of(host_ref[i]) ||
                   (std::isnan(out[i]) && std::isnan(host_ref[i])))
            << "i=" << i << " input=" << values[i] << " branchless_out=" << out[i]
            << " combine_host_ref=" << host_ref[i];
    }
}

// Same as CombineBranchlessPackedBufferLargeShapeExtremeMagnitude above,
// but with random_values_small_log_uniform instead of random_values_
// extreme -- specifically targets the UNDERFLOW regime (down to 1e-15)
// the real production matrix's smallest entries (down to ~1.4e-12) live
// in, which random_values_extreme's linear-uniform sampling essentially
// never reached despite nominally spanning that range. Investigates
// whether fp32_16_combine_branchless's DEVICE decode of near-zero/
// saturate-underflow values matches the HOST fp32_16_combine reference,
// reading from the real packed buffer at the real failing GEMV shape.
TEST_F(Fp32_16CudaTest, CombineBranchlessPackedBufferSmallMagnitudeUnderflow) {
    dim<2> size = {140000, 100};
    size_t n = size[0] * size[1];
    auto values = random_values_small_log_uniform(n, 61);

    std::vector<float> host_ref(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t wh, wr;
        fp32_16_split(values[i], wh, wr);
        host_ref[i] = fp32_16_combine(wh, wr);
    }

    Dense source =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill_dense_cuda(source, values);
    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(source, packed);

    auto vPacked = as_strided_view<uint32_t>(packed);
    Fp32_16Storage s = as_fp32_16_storage(vPacked.values_, n);

    float* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));
    cuda::combine_branchless_probe(n, s.wh, s.wr, d_out);

    std::vector<float> out(n);
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    size_t zero_count = 0, mismatch_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (host_ref[i] == 0.0f) ++zero_count;
        bool ok = bits_of(out[i]) == bits_of(host_ref[i]) ||
                 (std::isnan(out[i]) && std::isnan(host_ref[i]));
        if (!ok) ++mismatch_count;
        EXPECT_TRUE(ok) << "i=" << i << " input=" << values[i]
                       << " branchless_out=" << out[i]
                       << " combine_host_ref=" << host_ref[i];
    }
    std::printf("[CombineBranchlessPackedBufferSmallMagnitudeUnderflow] n=%zu "
               "zeroed_by_host_ref=%zu (%.4f%%) device_host_mismatches=%zu\n",
               n, zero_count, 100.0 * static_cast<double>(zero_count) / n,
               mismatch_count);
}

TEST_F(Fp32_16CudaTest, MvAllTierTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_combination(ctx, Device::CUDA, fill_dense_cuda,
                              collect_dense_cuda, trans, beta);
}

// MvAllTierTransposeAndBetaCombinations above only ever exercises a 5x5
// matrix -- every FP32_16 GEMV dispatcher (mv_fp32_16_pure_half,
// mv_fp32_16_accurate, mv_fp32_16_fast) picks between several internal
// __global__ kernels based on shape (see e.g. mv_fp32_16_accurate's own
// rows_op/cols_op branching in fp32_16_cuda_kernels.cu), and a 5x5 case
// only ever reaches the smallest-shape branch. This test picks m/n
// specifically to reach the OTHER branches on both sides of a TRANSPOSE
// call: m=140000 forces NO_TRANSPOSE (rows_op=m) into the "wide" kernel's
// row-threshold miss -> narrow_vec kernel, while the same A, TRANSPOSE
// (rows_op=n=100) lands in the "wide"/split-K kernel -- the two branches
// LSQR's own step_1_orth_basis exercises every iteration on a real
// tall-thin problem, which the 5x5 test can never reach.
using AGenFn = std::vector<float> (*)(size_t, unsigned);

void run_mv_large_shape_combination(Context& ctx, Device dev, FillFn fill,
                                    CollectFn collect, TransposeType trans,
                                    double beta, AGenFn a_gen = random_values,
                                    bool check_pure_half = true,
                                    AGenFn x_gen = nullptr,
                                    size_t m_override = 140000) {
    if (!x_gen) x_gen = a_gen;
    SCOPED_TRACE(std::string("large-shape mv trans=") + trans_name(trans) +
                " beta=" + std::to_string(beta) + " m=" + std::to_string(m_override));
    dim<2> size = {m_override, 100};
    auto a_values = a_gen(size[0] * size[1], 301);
    size_t x_len = (trans == NO_TRANSPOSE) ? size[1] : size[0];
    size_t y_len = (trans == NO_TRANSPOSE) ? size[0] : size[1];
    // x_gen defaults to a_gen but can be set independently: in the real
    // failing solve, x here is precond_v_d -- the OUTPUT of the
    // preconditioner (R^-1 * v), which for a condition-number-4.4e9
    // matrix could plausibly be far more extreme in magnitude than A's
    // own entries. Testing them independently isolates whether A's own
    // encoding (the "saturate" bucket) or x's own magnitude (a plain,
    // unpacked float -- no encoding involved at all) is what's actually
    // responsible for a failure seen with both extreme simultaneously.
    auto x_values = x_gen(x_len, 302);
    auto y_init = random_values(y_len, 303);

    Dense A =
        Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32, size);
    fill(A, a_values);
    Dense packed = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                         NumericType::FP32_16, size);
    split_fp32_16(A, packed);

    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, NumericType::FP32,
                                    {x_len, 1});
    fill(x, x_values);

    std::vector<double> y_ref(y_len);
    for (size_t i = 0; i < y_len; ++i)
        y_ref[i] = beta * static_cast<double>(y_init[i]);
    for (size_t i = 0; i < y_len; ++i) {
        for (size_t j = 0; j < x_len; ++j) {
            size_t a_row = (trans == NO_TRANSPOSE) ? i : j;
            size_t a_col = (trans == NO_TRANSPOSE) ? j : i;
            y_ref[i] += static_cast<double>(a_values[a_row + a_col * size[0]]) *
                        static_cast<double>(x_values[j]);
        }
    }

    auto make_y = [&]() {
        Dense y = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                        NumericType::FP32, {y_len, 1});
        fill(y, y_init);
        return y;
    };

    Dense y_pure_half = make_y();
    mv_apply_fp32_16_pure_half(dev, trans, 1.0, packed, x, beta, y_pure_half);
    auto pure_half_out = collect(y_pure_half);

    Dense y_ext = make_y();
    mv_apply_fp32_16_ext(dev, trans, 1.0, packed, x, beta, y_ext);
    auto ext_out = collect(y_ext);

    // L2-relative-error over the whole vector, not a per-element check --
    // at 100000+ elements some y_ref[i] inevitably land near zero by
    // chance, and a per-element relative tolerance spuriously fails there
    // even when the actual error is small in absolute/aggregate terms
    // (the same reasoning the GEVM plots' own L2-error benchmarks, e.g.
    // mv_fp32_16_l2_error_vs_cublas_ref.cu, are built around).
    auto l2_rel_err = [&](const std::vector<float>& out) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < y_len; ++i) {
            double d = static_cast<double>(out[i]) - y_ref[i];
            num += d * d;
            den += y_ref[i] * y_ref[i];
        }
        return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    };

    double pure_half_l2 = l2_rel_err(pure_half_out);
    double ext_l2 = l2_rel_err(ext_out);
    SCOPED_TRACE(std::string("pure_half l2_rel_err=") + std::to_string(pure_half_l2) +
                " ext l2_rel_err=" + std::to_string(ext_l2));
    // pure_half is genuine fp16 compute -- 5% L2-relative-error is the
    // same bound the GEVM plots' own pure_half L2-error checks use. Only
    // asserted when check_pure_half: with random_values_extreme's data,
    // pure_half's fp32_16_half_only decode is EXPECTED (by design, same
    // as real hardware half overflow) to saturate to Inf for magnitudes
    // beyond a genuine half float's own range -- not a bug, so that case
    // only checks ext, which is meant to represent that wider range.
    // ext reconstructs via wh+wr and should be near machine-precision
    // regardless of magnitude.
    if (check_pure_half) { EXPECT_LT(pure_half_l2, 5e-2); }
    EXPECT_LT(ext_l2, 1e-3);
}

TEST_F(Fp32_16CudaTest, MvLargeShapeTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_large_shape_combination(ctx, Device::CUDA, fill_dense_cuda,
                                          collect_dense_cuda, trans, beta);
}

// Same large shape (and thus same wide/narrow_vec kernel branches) as
// MvLargeShapeTransposeAndBetaCombinations above, but with A's own values
// drawn from random_values_extreme instead of random_values -- forces
// fp32_16_split's "extended" encoding branch to actually fire on real
// production data at real scale, through the real GPU GEMV kernels
// (mv_fp32_16_accurate_wide_kernel/_narrow_vec_kernel's own
// fp32_16_combine_branchless calls), rather than only through the
// separately-defined, host-testable fp32_16_combine that
// Fp32_16PackTest's exhaustive exponent sweep already validates.
TEST_F(Fp32_16CudaTest, MvLargeShapeExtremeMagnitudeTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_large_shape_combination(ctx, Device::CUDA, fill_dense_cuda,
                                          collect_dense_cuda, trans, beta,
                                          random_values_extreme,
                                          /*check_pure_half=*/false);
}

// Isolates WHICH operand's extreme magnitude actually causes the NaN seen
// in MvLargeShapeExtremeMagnitudeTransposeAndBetaCombinations above (which
// made both A and x extreme at once): A extreme (crossing into
// fp32_16_split's "saturate" bucket, unlike the narrower first attempt at
// this test), x an ordinary +-8 vector -- same as any REAL A but a
// well-scaled input vector.
TEST_F(Fp32_16CudaTest, MvLargeShapeExtremeMatrixOnlyTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_large_shape_combination(ctx, Device::CUDA, fill_dense_cuda,
                                          collect_dense_cuda, trans, beta,
                                          random_values_extreme,
                                          /*check_pure_half=*/false,
                                          /*x_gen=*/random_values);
}

// Every decode path (split, fp32_16_combine, fp32_16_combine_branchless,
// even reading from the real packed buffer at this exact scale) has now
// checked out fine in isolation, yet the full GEMV kernel still NaNs on
// A-only-extreme data. mv_fp32_16_accurate_narrow_vec_kernel (the
// NO_TRANSPOSE path here, since rows_op=m stays above kMvWideRowThreshold)
// has a vectorized fast path that loads FOUR uint16_t wh/wr values at
// once via a ushort4 load, active whenever ld_a % kMvRowsPerThread(4)==0
// -- true for m=140000 (used everywhere above), untested with it forced
// off. m=140001 keeps the exact same kernel-selection branch (rows_op/
// cols_op thresholds are unaffected by +1) but ld_a%4 != 0, forcing the
// scalar (non-ushort4) fallback for every row. If THIS passes where the
// m=140000 version fails, the bug is isolated to the vectorized load.
TEST_F(Fp32_16CudaTest, MvLargeShapeExtremeMatrixOnlyNonVectorizedLeadDim) {
    run_mv_large_shape_combination(ctx, Device::CUDA, fill_dense_cuda,
                                  collect_dense_cuda, NO_TRANSPOSE, 0.0,
                                  random_values_extreme,
                                  /*check_pure_half=*/false,
                                  /*x_gen=*/random_values,
                                  /*m_override=*/140001);
}

// The other half of the isolation: an ordinary +-8 A, but x (standing in
// for precond_v_d, the preconditioner's OWN output -- R^-1 * v, which for
// a condition-number-4.4e9 matrix could plausibly be far more extreme
// than A's own entries) drawn from random_values_extreme instead.
TEST_F(Fp32_16CudaTest, MvLargeShapeExtremeVectorOnlyTransposeAndBetaCombinations) {
    for (TransposeType trans : {NO_TRANSPOSE, TRANSPOSE})
        for (double beta : {0.0, 0.7})
            run_mv_large_shape_combination(ctx, Device::CUDA, fill_dense_cuda,
                                          collect_dense_cuda, trans, beta,
                                          random_values,
                                          /*check_pure_half=*/false,
                                          /*x_gen=*/random_values_extreme);
}

TEST_F(Fp32_16CudaTest, MmAllTierTransposeAndBetaCombinations) {
    for (TransposeType transA : {NO_TRANSPOSE, TRANSPOSE})
        for (TransposeType transB : {NO_TRANSPOSE, TRANSPOSE})
            for (double beta : {0.0, 0.7})
                run_mm_combination(ctx, Device::CUDA, fill_dense_cuda,
                                  collect_dense_cuda, transA, transB, beta);
}

#endif  // defined(USE_CUDA)

}  // namespace test
}  // namespace tomos

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

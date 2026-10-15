// Diagnostic: does fp32_16_split_stochastic_rounding's own PRNG pairing
// (stochastic_rand14 vs pseudo_rand -- both xorshift hashes of the same
// idx, different seed constants) actually produce statistically correct
// Bernoulli(p) draws, where p = mant_lo/16384 is the TRUE fractional
// rounding position for each element? And separately: is p itself close
// to uniformly distributed across pseudo_rand's own generated data, or
// does pseudo_rand's own coarse ~200001-value quantization skew it away
// from uniform?
//
// Motivated by an empirical result: mm_fp32_16_variable_rounding_l2_
// error_vs_cublas_ref.cu's own stochastic-rounding series measured
// WORSE Frobenius relative error than plain truncation, contrary to the
// usual "stochastic rounding beats truncation in aggregate" expectation
// -- which only holds when p is genuinely ~Uniform[0,1] across enough
// independent elements AND the per-element Bernoulli(p) draws are
// actually unbiased. This diagnostic checks both preconditions directly
// instead of guessing which one broke.
//
// For N elements (idx = 0..N-1) at a given scale: computes p = mant_lo/
// 16384 (the THEORETICAL round-up probability for that element, from
// its own true fp32 bit pattern) and whether stochastic_rand14(idx) <
// mant_lo (the ACTUAL round-up decision fp32_16_split_stochastic_
// rounding itself would make for that exact idx) directly on the
// device, copies both back, and on the HOST:
//   1. Reports the OVERALL mean(p) vs mean(round_up) -- if the PRNG
//      pairing is unbiased, E[round_up] should equal E[p] exactly (a
//      correct Bernoulli(p) draw has that property by construction), so
//      a large gap here directly implicates the PRNG pairing itself
//      (hypothesis #1).
//   2. Bins elements by p into 20 equal-width buckets and, WITHIN each
//      bucket, compares that bucket's own mean p against the OBSERVED
//      round-up FREQUENCY among elements in it. This is the stronger,
//      bin-conditional version of the same check -- correlation between
//      the two xorshift streams could show up as a systematic per-bin
//      deviation even if the overall (unconditional) means happen to
//      roughly match.
//   3. The same per-bin table's own "count" column doubles as a
//      histogram of p itself: if pseudo_rand's coarse quantization
//      skews p away from uniform (hypothesis #2), bin counts will be
//      visibly uneven rather than roughly equal across all 20 bins.
//
// Deliberately NOT wired into tomos's CMake build, and does NOT link
// against libtomos_cuda.so or include any of this codebase's own core
// headers -- purely self-contained bit-manipulation and RNG code,
// duplicated BYTE-FOR-BYTE from mm_fp32_16_variable_rounding_l2_error_
// vs_cublas_ref.cu's own pseudo_rand/stochastic_rand14 (see that file
// for the canonical, documented versions), so this diagnostic tests the
// exact mechanism that file's own stochastic series uses, not a
// reimplementation that could subtly differ. mant32's own mask
// (0x007FFFFFu) and mant_lo's own mask (0x3FFFu) are fp32_16_pack.hpp's
// kFp32MantMask and the low-14-bits mask respectively, inlined here as
// plain literals for the same reason.
//
// Usage: mm_fp32_16_stochastic_rounding_diagnostic [N [scale]]
//   N: number of elements to sample (default 10000000).
//   scale: normal_scale applied to A, same meaning as the error
//          benchmark's own normal_scale (default 1.0).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

__device__ float pseudo_rand(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 2654435761u + 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s % 200001u) / 100000.0f) - 1.0f;  // [-1, 1]
}

__device__ uint32_t stochastic_rand14(size_t idx) {
    uint32_t s = static_cast<uint32_t>(idx) * 0x9E3779B1u + 0xC2B2AE35u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s & 0x3FFFu;
}

__global__ void sample_kernel(size_t n, float scale, float* p_out, unsigned char* round_up_out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = scale * pseudo_rand(i);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t mant32 = bits & 0x007FFFFFu;
    uint32_t mant_lo = mant32 & 0x3FFFu;
    p_out[i] = static_cast<float>(mant_lo) / 16384.0f;
    round_up_out[i] = (stochastic_rand14(i) < mant_lo) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)) : 10000000ULL;
    float scale = (argc > 2) ? static_cast<float>(std::strtod(argv[2], nullptr)) : 1.0f;

    std::printf("=== fp32_16 stochastic-rounding PRNG diagnostic (N=%zu, scale=%g) ===\n", n, scale);

    float* p_dev = nullptr;
    unsigned char* round_up_dev = nullptr;
    if (cudaMalloc(&p_dev, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&round_up_dev, n * sizeof(unsigned char)) != cudaSuccess) {
        std::fprintf(stderr, "error: cudaMalloc failed for N=%zu\n", n);
        return 1;
    }

    int threads = 256;
    int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1) / threads);
    sample_kernel<<<blocks, threads>>>(n, scale, p_dev, round_up_dev);

    std::vector<float> p_host(n);
    std::vector<unsigned char> round_up_host(n);
    cudaMemcpy(p_host.data(), p_dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(round_up_host.data(), round_up_dev, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(p_dev);
    cudaFree(round_up_dev);

    double sum_p = 0.0, sum_round_up = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_p += p_host[i];
        sum_round_up += round_up_host[i];
    }
    double mean_p = sum_p / static_cast<double>(n);
    double mean_round_up = sum_round_up / static_cast<double>(n);
    std::printf("Overall: mean(p) = %.6f, mean(round_up) = %.6f, diff = %.6f\n", mean_p, mean_round_up,
               mean_round_up - mean_p);
    std::printf(
        "  (mean(p) far from 0.5 indicates pseudo_rand's own data does NOT give a\n"
        "   uniform fractional-position distribution at this scale -- hypothesis #2.\n"
        "   mean(round_up) far from mean(p) indicates the stochastic_rand14/pseudo_rand\n"
        "   PRNG pairing is NOT behaving as an unbiased Bernoulli(p) draw -- hypothesis\n"
        "   #1.)\n\n");

    const int kBins = 20;
    std::vector<double> bin_sum_p(kBins, 0.0);
    std::vector<size_t> bin_count(kBins, 0);
    std::vector<size_t> bin_round_up(kBins, 0);
    for (size_t i = 0; i < n; ++i) {
        float p = p_host[i];
        int bin = static_cast<int>(p * kBins);
        if (bin >= kBins) bin = kBins - 1;
        if (bin < 0) bin = 0;
        bin_sum_p[static_cast<size_t>(bin)] += p;
        bin_count[static_cast<size_t>(bin)] += 1;
        if (round_up_host[i]) bin_round_up[static_cast<size_t>(bin)] += 1;
    }
    std::printf("%-14s %10s %14s %18s %10s\n", "p range", "count", "mean p", "observed freq", "|diff|");
    for (int b = 0; b < kBins; ++b) {
        double lo = static_cast<double>(b) / kBins;
        double hi = static_cast<double>(b + 1) / kBins;
        size_t count = bin_count[static_cast<size_t>(b)];
        if (count == 0) {
            std::printf(" %.2f-%.2f    %10zu %14s %18s %10s\n", lo, hi, count, "n/a", "n/a", "n/a");
            continue;
        }
        double mean_bin_p = bin_sum_p[static_cast<size_t>(b)] / static_cast<double>(count);
        double observed_freq = static_cast<double>(bin_round_up[static_cast<size_t>(b)]) / static_cast<double>(count);
        std::printf(" %.2f-%.2f    %10zu %14.6f %18.6f %10.6f\n", lo, hi, count, mean_bin_p, observed_freq,
                   std::fabs(observed_freq - mean_bin_p));
    }

    return 0;
}

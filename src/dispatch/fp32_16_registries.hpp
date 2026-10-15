#ifndef TOMOS_FP32_16_REGISTRIES_HPP
#define TOMOS_FP32_16_REGISTRIES_HPP

#include <array>

#include "matrix.hpp"
#include "storage_registry_base.hpp"
#include "transpose.hpp"

namespace tomos {
namespace dispatch {

// Type-erased function pointer types for registry entries. Every function
// operates on plain Dense objects: "source"/"A" holds NumericType::FP32_16
// (the packed wh/wr representation, see fp32_16_pack.hpp), every other
// operand is a plain NumericType::FP32 Dense.
//
// tomos::Dense must be fully qualified here: inside namespace tomos::dispatch,
// unqualified "Dense" resolves to dispatch::Dense (the sibling registry
// struct below), not tomos::Dense -- the same collision dense_registries.hpp
// already works around.
using split_fp32_16_fn = void (*)(tomos::Dense& source, tomos::Dense& target);
using combine_fp32_16_fn = void (*)(tomos::Dense& source, tomos::Dense& target);
// workspace: see dense_registries.hpp's mv_dense_fn -- mv_pure_half_impl is
// registered into both that array and mv_pure_half_ below, so this typedef's
// shape must match mv_dense_fn's exactly.
using mv_fp32_16_fn = void (*)(TransposeType trans, double alpha,
                               tomos::Dense& A, tomos::Dense& x, double beta,
                               tomos::Dense& y, float* workspace);
using mm_fp32_16_fn = void (*)(TransposeType transA, TransposeType transB,
                               double alpha, tomos::Dense& A, tomos::Dense& B,
                               double beta, tomos::Dense& C);

// Singleton registry for the split-precision (FP32_16) GEMM/GEMV feature.
// Keyed only by Device, since every operand's NumericType is fixed by the
// feature itself (FP32_16 for A, FP32 for everything else).
struct Fp32_16 {
    static const Fp32_16& get();

    split_fp32_16_fn lookup_split(Device dev) const;
    combine_fp32_16_fn lookup_combine(Device dev) const;
    mv_fp32_16_fn lookup_mv_pure_half(Device dev) const;
    mv_fp32_16_fn lookup_mv_fast(Device dev) const;
    mv_fp32_16_fn lookup_mv_accurate(Device dev) const;
    mm_fp32_16_fn lookup_mm_pure_half(Device dev) const;
    mm_fp32_16_fn lookup_mm_fast(Device dev) const;
    mm_fp32_16_fn lookup_mm_accurate(Device dev) const;

    // Decode-into-scratch-then-native-GEMM tier (mm_fp32_16_gemm_materialize_
    // midpoint_exact): trades a workspace allocation for speed at large
    // shapes. CUDA-only -- no CPU counterpart exists (this is purely a GPU
    // performance optimization), so lookup_mm_workspace(Device::CPU) always
    // throws, even in a CUDA-enabled build.
    mm_fp32_16_fn lookup_mm_workspace(Device dev) const;

private:
    Fp32_16();

    static constexpr std::size_t N = num_registry_devices;

    std::array<split_fp32_16_fn, N> split_{};
    std::array<combine_fp32_16_fn, N> combine_{};
    std::array<mv_fp32_16_fn, N> mv_pure_half_{};
    std::array<mv_fp32_16_fn, N> mv_fast_{};
    std::array<mv_fp32_16_fn, N> mv_accurate_{};
    std::array<mm_fp32_16_fn, N> mm_pure_half_{};
    std::array<mm_fp32_16_fn, N> mm_fast_{};
    std::array<mm_fp32_16_fn, N> mm_accurate_{};
    std::array<mm_fp32_16_fn, N> mm_workspace_{};
};

}  // namespace dispatch
}  // namespace tomos

#endif  // TOMOS_FP32_16_REGISTRIES_HPP